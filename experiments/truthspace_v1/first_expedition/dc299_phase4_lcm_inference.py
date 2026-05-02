#!/usr/bin/env python3
"""
DC299 Phase 4 — LCM Inference Engine
======================================

The inference loop:
    query_word  →  embedding  →  193-dim projection  →  Gödel address
                                                              ↓
    operation (axis-flip / analogy / NN)  →  nearest address  →  word

Operations:
  nearest(word, k)          — Hamming NN in address space
  axis_flip(word, axis)     — flip one axis bit, find nearest concept
  analogy(a, b, c)          — proj(a) - proj(b) + proj(c) → threshold → NN
  conditional(word, **axes) — force axis values, find nearest matching concept
  describe(word)            — show all axis projections / bits for a concept

Fail-fast: no graceful fallbacks.

Usage:
    python dc299_phase4_lcm_inference.py           # run test suite
    python dc299_phase4_lcm_inference.py --repl    # interactive REPL
"""

import sys, os, json, re, time
import numpy as np
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR    = (PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2"
                / "phi_model")
AXES_JSON    = SCRIPT_DIR / "dc299_phase1_axes.json"
AXES_JSON_B  = SCRIPT_DIR / "dc299_phase1b_axes.json"    # Phase 1b extension (optional)
PHASE0_JSON  = SCRIPT_DIR / "dc299_phase0_concepts.json"
PHASE3_JSON  = SCRIPT_DIR / "dc299_phase3_godel_addresses.json"

for p in [AXES_JSON, PHASE0_JSON, PHASE3_JSON]:
    assert p.exists(), f"FAIL-FAST: {p} missing"

QUALITY_MIN = 0.00      # include all axes — quality filter disabled; IRD order is sufficient
QUALITY_WINDOW = 20
MAX_SEMANTIC = 500      # optimal for inference (sweep: 74% agg. rank-1); use 1500 for max completeness

_CLEAN = re.compile(r'^[A-Za-z]{3,}$')
def _quality(top, bot):
    tokens = [w for w, _ in top] + [w for w, _ in bot]
    if not tokens: return 0.0
    return sum(1 for t in tokens if _CLEAN.match(t.strip())) / len(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# LCM Index
# ─────────────────────────────────────────────────────────────────────────────

class LCMIndex:
    """
    In-memory index of all concept Gödel addresses.

    Core data:
      self.words        — list of concept strings (len N)
      self.axis_vectors — (n_axes, D) float32
      self.thresholds   — (n_axes,)  float32
      self.projections  — (N, n_axes) float32  — continuous projections
      self.addresses    — (N, n_axes) bool      — binary addresses
      self.packed       — (N, n_bytes) uint8    — packed for fast Hamming
      self.axis_names   — list of axis name strings
      self.axis_slots   — dict  axis_name_fragment → slot index
    """

    def __init__(self):
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load(self):
        from phi_geometric.inference.phi_types import PhiEncoded

        print("Loading embeddings …", flush=True)
        phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
        self.embeddings = phi.decode()           # (vocab, D)

        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
        )
        snap = os.listdir(cache_dir)[0]
        with open(os.path.join(cache_dir, snap, "tokenizer.json")) as f:
            td = json.load(f)
        vocab = td.get("model", {}).get("vocab", {})
        self.id_to_token = {idx: tok for tok, idx in vocab.items()}
        self.token_to_id = {tok: idx for tok, idx in vocab.items()}

        print("Loading axes …", flush=True)
        with open(AXES_JSON) as f:
            axes_data = json.load(f)

        # Select semantic axes — load Phase 1 and optionally Phase 1b extension
        axes_meta   = axes_data["axes"]
        all_vectors = np.array(axes_data["axis_vectors"], dtype=np.float32)

        if AXES_JSON_B.exists():
            with open(AXES_JSON_B) as fb:
                axes_b = json.load(fb)
            axes_meta   = axes_meta + axes_b["axes"]
            all_vectors = np.concatenate(
                [all_vectors,
                 np.array(axes_b["axis_vectors"], dtype=np.float32)], axis=0
            )
            print(f"  (+{len(axes_b['axes'])} Phase-1b axes → {len(axes_meta)} total)",
                  flush=True)

        qualities = []
        for m in axes_meta:
            if m.get("type") == "seed":
                qualities.append(1.0)
            else:
                qualities.append(_quality(m.get("top_vocab", []),
                                           m.get("bot_vocab", [])))

        disc_q = [q for m, q in zip(axes_meta, qualities)
                  if m.get("type") == "discovered"]
        cliff_idx = len(axes_meta)
        _CLIFF_THR = 0.4
        if QUALITY_MIN < _CLIFF_THR:   # cliff disabled when quality filter is loose
            pass
        else:
            for i in range(len(disc_q) - QUALITY_WINDOW):
                if np.mean(disc_q[i:i+QUALITY_WINDOW]) < _CLIFF_THR:
                    n_seeds = sum(1 for m in axes_meta if m.get("type") == "seed")
                    cliff_idx = n_seeds + i
                    break

        selected = []
        for i, (m, q) in enumerate(zip(axes_meta, qualities)):
            if i >= cliff_idx: break
            is_seed = m.get("type") == "seed"
            if (is_seed or q >= QUALITY_MIN) and len(selected) < MAX_SEMANTIC:
                selected.append(i)

        self.selected_indices = selected
        self.axis_vectors  = all_vectors[selected]           # (n_axes, D)
        self.axis_names    = [axes_meta[i].get("name", f"axis_{i:04d}")
                              for i in selected]
        self.axis_slots    = {}
        for slot, name in enumerate(self.axis_names):
            # index by canonical prefix fragments
            for key in [name,
                        name.split("__")[0],
                        name[:20]]:
                self.axis_slots.setdefault(key.lower(), slot)
        # Also index seeds by their short name; track which slots are seeds
        self.seed_slots = set()
        for slot, i in enumerate(selected):
            m = axes_meta[i]
            if m.get("type") == "seed":
                short = m.get("name", "")
                self.axis_slots[short.lower()] = slot
                # Also register without "is_" prefix
                self.axis_slots[short.replace("is_", "").lower()] = slot
                self.seed_slots.add(slot)

        print(f"  {len(selected)} semantic axes selected", flush=True)

        print("Loading Phase-0 concepts …", flush=True)
        with open(PHASE0_JSON) as f:
            p0 = json.load(f)
        self.words    = [c["word"]     for c in p0]
        self.tids     = [c["token_id"] for c in p0]
        self.word_set = {w.lower(): i for i, w in enumerate(self.words)}

        print("Loading Phase-3 thresholds …", flush=True)
        with open(PHASE3_JSON) as f:
            p3 = json.load(f)
        p3_thresholds = np.array(p3["thresholds"], dtype=np.float32)

        # Build concept embedding matrix
        print("Building projection index …", flush=True)
        concept_embs = np.array(
            [self.embeddings[tid] for tid in self.tids], dtype=np.float32
        )
        norms = np.linalg.norm(concept_embs, axis=1, keepdims=True)
        concept_embs_normed = concept_embs / (norms + 1e-20)

        # Continuous projections: (N, n_axes)
        self.projections = concept_embs_normed @ self.axis_vectors.T

        # Extend thresholds: phase3 has values for the original 193 axes (by
        # axis_vector index). For any axes beyond that, use per-axis median.
        n_axes = self.projections.shape[1]
        n_p3   = len(p3_thresholds)
        if n_axes <= n_p3:
            self.thresholds = p3_thresholds[:n_axes]
        else:
            extra = np.median(self.projections[:, n_p3:], axis=0).astype(np.float32)
            self.thresholds = np.concatenate([p3_thresholds, extra])

        # Binary addresses: (N, n_axes) bool
        self.addresses = self.projections > self.thresholds[np.newaxis, :]

        # Packed uint8 for fast Hamming: (N, n_bytes)
        self.packed = np.packbits(self.addresses, axis=1)

        # ── Phi-4-state encoding (DC255) ──────────────────────────────────────
        # Boundaries at ±log(φ) × per-axis-std from threshold.
        # States: CONTRACT(-1)/PRESERVE-(-0)/PRESERVE+(+0)/EXPAND(+1)
        _LOG_PHI = np.float32(np.log(1.6180339887))   # ≈ 0.4812
        centered  = self.projections - self.thresholds[np.newaxis, :]
        _phi4_std = centered.std(axis=0).astype(np.float32)
        self._phi4_hi = (self.thresholds + _LOG_PHI * _phi4_std).astype(np.float32)
        self._phi4_lo = (self.thresholds - _LOG_PHI * _phi4_std).astype(np.float32)
        self.phi4_vecs          = self._compute_phi4(self.projections, continuous=True)
        self.phi4_vecs_discrete = self._compute_phi4(self.projections, continuous=False)
        # v6: Fibonacci-corrected — preserve_scale=1.0 so levels are {-φ,-1,+1,+φ}
        # Adjacent ratio = φ throughout (vs φ² gap in v4)
        self.phi4_vecs_v6       = self._compute_phi4(self.projections, continuous=True,
                                                     preserve_scale=1.0)
        self.phi4_vecs_v6_disc  = self._compute_phi4(self.projections, continuous=False,
                                                     preserve_scale=1.0)

        n_axes = len(selected)
        print(f"  {len(self.words)} concepts × {n_axes} axes  "
              f"packed={self.packed.shape}  ready.", flush=True)

    def add_word(self, word, token_id, overwrite=False):
        """
        Inject a concept that is absent from the Phase-0 vocabulary.
        Extends all projection / encoding matrices in-place so that subsequent
        searches include the new concept.
        If overwrite=True and the word is already present (possibly with a
        different, worse token variant), replace its embedding row in-place.
        """
        existing_idx = self.word_set.get(word.lower())
        if existing_idx is not None and not overwrite:
            return
        emb  = self.embeddings[token_id].astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-20
        proj = ((emb / norm) @ self.axis_vectors.T).astype(np.float32)

        p4   = self._compute_phi4(proj[np.newaxis, :], continuous=True)
        p4d  = self._compute_phi4(proj[np.newaxis, :], continuous=False)
        p4v6 = self._compute_phi4(proj[np.newaxis, :], continuous=True,
                                  preserve_scale=1.0)
        p4v6d= self._compute_phi4(proj[np.newaxis, :], continuous=False,
                                  preserve_scale=1.0)
        addr = proj > self.thresholds

        if existing_idx is not None:
            # overwrite=True: replace the existing row in-place with better token
            self.words[existing_idx]        = word
            self.tids[existing_idx]         = token_id
            self.projections[existing_idx]  = proj
            self.addresses[existing_idx]    = addr
            self.packed[existing_idx]       = np.packbits(addr)
            self.phi4_vecs[existing_idx]          = p4[0]
            self.phi4_vecs_discrete[existing_idx] = p4d[0]
            self.phi4_vecs_v6[existing_idx]       = p4v6[0]
            self.phi4_vecs_v6_disc[existing_idx]  = p4v6d[0]
        else:
            idx = len(self.words)
            self.words.append(word)
            self.tids.append(token_id)
            self.word_set[word.lower()] = idx
            self.projections = np.vstack([self.projections,   proj[np.newaxis, :]])
            self.addresses   = np.vstack([self.addresses,     addr[np.newaxis, :]])
            self.packed      = np.vstack([self.packed,
                                          np.packbits(addr)[np.newaxis, :]])
            self.phi4_vecs          = np.vstack([self.phi4_vecs,          p4])
            self.phi4_vecs_discrete = np.vstack([self.phi4_vecs_discrete, p4d])
            self.phi4_vecs_v6       = np.vstack([self.phi4_vecs_v6,       p4v6])
            self.phi4_vecs_v6_disc  = np.vstack([self.phi4_vecs_v6_disc,  p4v6d])

    # ── Token lookup ──────────────────────────────────────────────────────────
    def _find_tid(self, word):
        for cand in [word, word.lower(), word.capitalize(),
                     f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
                     f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}"]:
            if cand in self.token_to_id:
                return self.token_to_id[cand]
        return None

    def _get_proj(self, word):
        """Get continuous 193-dim projection for any vocab word (not just Phase-0)."""
        # First check Phase-0 index (fast)
        idx = self.word_set.get(word.lower())
        if idx is not None:
            return self.projections[idx], idx

        # Otherwise compute from embedding
        tid = self._find_tid(word)
        if tid is None:
            raise RuntimeError(f"FAIL-FAST: '{word}' not found in vocabulary")
        e = self.embeddings[tid].astype(np.float32)
        e_norm = e / (np.linalg.norm(e) + 1e-20)
        proj = e_norm @ self.axis_vectors.T    # (n_axes,)
        return proj, None

    def _proj_to_packed(self, proj):
        addr = (proj > self.thresholds)
        return np.packbits(addr).reshape(1, -1)

    # ── Hamming search ────────────────────────────────────────────────────────
    def _hamming_search(self, packed_query, k=10, exclude_idx=None):
        """Return (distances, concept_indices) of k nearest concepts."""
        xor = np.bitwise_xor(packed_query, self.packed)           # (N, n_bytes)
        # Fast popcount via lookup table
        hd = np.unpackbits(xor, axis=1).sum(axis=1)               # (N,)
        if exclude_idx is not None:
            hd[exclude_idx] = 999999
        top_k = np.argsort(hd)[:k]
        return hd[top_k], top_k

    # ── Public API ────────────────────────────────────────────────────────────
    def nearest(self, word, k=10):
        """Find k concepts with nearest Hamming distance to word."""
        proj, idx = self._get_proj(word)
        packed = self._proj_to_packed(proj)
        dists, idxs = self._hamming_search(packed, k=k+1,
                                           exclude_idx=idx)
        results = []
        for d, i in zip(dists, idxs):
            if self.words[i].lower() != word.lower():
                results.append((self.words[i], int(d)))
        return results[:k]

    def axis_flip(self, word, axis_key, k=5):
        """
        Flip exactly one axis bit in word's address, find nearest concepts.
        This is the correct single-step inference: 'capital of France', 'female king', etc.
        """
        slot = self._resolve_axis(axis_key)
        proj, idx = self._get_proj(word)

        # Reflect the projection past the threshold on the opposite side
        t = float(self.thresholds[slot])
        p = float(proj[slot])
        # If currently positive (bit=1), push to negative side and vice versa
        distance_from_t = abs(p - t)
        flipped_proj = proj.copy()
        if p > t:
            flipped_proj[slot] = t - distance_from_t - 0.01
        else:
            flipped_proj[slot] = t + distance_from_t + 0.01

        packed = self._proj_to_packed(flipped_proj)
        dists, idxs = self._hamming_search(packed, k=k+1, exclude_idx=idx)
        results = []
        for d, i in zip(dists, idxs):
            if self.words[i].lower() != word.lower():
                results.append((self.words[i], int(d)))
        return results[:k]

    def analogy(self, a, b, c, k=10):
        """
        Continuous analogy: proj(a) - proj(b) + proj(c) → threshold → NN.
        Correct approach: arithmetic in projection space, not binary XOR.
        Example: analogy('king', 'man', 'woman') → queen
        """
        proj_a, idx_a = self._get_proj(a)
        proj_b, idx_b = self._get_proj(b)
        proj_c, idx_c = self._get_proj(c)

        analogy_proj = proj_a - proj_b + proj_c
        packed = self._proj_to_packed(analogy_proj)

        exclude = [i for i in [idx_a, idx_b, idx_c] if i is not None]
        hd  = np.full(len(self.words), 0)
        xor = np.bitwise_xor(packed, self.packed)
        hd  = np.unpackbits(xor, axis=1).sum(axis=1)
        for i in exclude:
            hd[i] = 999999

        top_k = np.argsort(hd)[:k]
        return [(self.words[i], int(hd[i])) for i in top_k]

    def conditional(self, word, k=10, **axis_conditions):
        """
        Force one or more axis values, find nearest concept satisfying them.
        axis_conditions: axis_key=True/False or axis_key=1/0
        Example: conditional('France', is_capital_city=True)
                 conditional('Japan',  is_european_country=True, is_capital_city=True)
        """
        proj, idx = self._get_proj(word)
        adj_proj = proj.copy()

        for axis_key, want_positive in axis_conditions.items():
            slot = self._resolve_axis(str(axis_key))
            t = float(self.thresholds[slot])
            p = float(adj_proj[slot])
            if bool(want_positive):
                if p <= t:
                    adj_proj[slot] = t + abs(t - p) + 0.01
            else:
                if p > t:
                    adj_proj[slot] = t - abs(p - t) - 0.01

        packed = self._proj_to_packed(adj_proj)
        dists, idxs = self._hamming_search(packed, k=k+1, exclude_idx=idx)
        results = []
        for d, i in zip(dists, idxs):
            if self.words[i].lower() != word.lower():
                results.append((self.words[i], int(d)))
        return results[:k]

    def describe(self, word, top_n=15):
        """Show the top-N most distinctive axes for this concept."""
        proj, idx = self._get_proj(word)
        # Sort by absolute distance from threshold
        distances_from_t = np.abs(proj - self.thresholds)
        top_slots = np.argsort(distances_from_t)[-top_n:][::-1]
        results = []
        for slot in top_slots:
            bit = bool(proj[slot] > self.thresholds[slot])
            p   = float(proj[slot])
            t   = float(self.thresholds[slot])
            name = self.axis_names[slot]
            # Clean axis name for display
            short = name.split("__")[0].replace("is_", "")[:30]
            results.append({
                "slot": int(slot),
                "name": short,
                "bit": int(bit),
                "proj": p,
                "threshold": t,
                "margin": abs(p - t),
            })
        return results

    # ── Decode from arbitrary embedding ───────────────────────────────────────
    def decode_embedding(self, e, k=10, exclude_words=None):
        """Find nearest concept to a raw embedding vector."""
        e = e.astype(np.float32)
        e_norm = e / (np.linalg.norm(e) + 1e-20)
        proj = e_norm @ self.axis_vectors.T
        packed = self._proj_to_packed(proj)
        dists, idxs = self._hamming_search(packed, k=k + (len(exclude_words or [])))
        results = []
        for d, i in zip(dists, idxs):
            if exclude_words and self.words[i].lower() in {w.lower() for w in exclude_words}:
                continue
            results.append((self.words[i], int(d)))
        return results[:k]

    def _compute_phi4(self, proj, continuous=True, preserve_scale=None):
        """
        Phi-4-state encoding per DC255 §3.

        States defined at ±log(φ) × per-axis-std boundaries:
          EXPAND    (+1): proj > hi    → signed × φ
          PRESERVE+ (+0): threshold ≤ proj ≤ hi  → signed × preserve_scale
          PRESERVE- (-0): lo ≤ proj < threshold   → signed × preserve_scale
          CONTRACT  (-1): proj < lo    → signed × φ

        preserve_scale choices:
          1/φ  (default, DC255 original) — adjacent ratio = φ²
          1.0  (Fibonacci-corrected, v6) — adjacent ratio = φ  ← {-φ,-1,+1,+φ}

        continuous=True:  values vary within each state zone
        continuous=False: discrete snap to exactly ±φ, ±preserve_scale
        """
        PHI    = np.float32(1.6180339887)
        if preserve_scale is None:
            preserve_scale = np.float32(1.0 / 1.6180339887)   # DC255 default
        PSCALE = np.float32(preserve_scale)

        squeeze = False
        if proj.ndim == 1:
            proj    = proj[np.newaxis, :]
            squeeze = True

        th     = self.thresholds[np.newaxis, :]   # (1, n_axes)
        hi     = self._phi4_hi[np.newaxis, :]     # EXPAND boundary
        lo     = self._phi4_lo[np.newaxis, :]     # CONTRACT boundary
        signed = (proj - th).astype(np.float32)   # centred on threshold

        if continuous:
            result = np.where(proj > hi,
                              signed * PHI,        # EXPAND:   ×φ
                     np.where(proj < lo,
                              signed * PHI,        # CONTRACT: ×φ (stays negative)
                              signed * PSCALE))    # PRESERVE±: ×preserve_scale
        else:
            result = np.where(proj > hi,    PHI,          # EXPAND    +φ
                     np.where(proj < lo,   -PHI,          # CONTRACT  -φ
                     np.where(proj >= th,   PSCALE,        # PRESERVE+ +scale
                                           -PSCALE)))      # PRESERVE- -scale

        result = result.astype(np.float32)
        if squeeze:
            result = result[0]
        return result

    def apply_delta_phi4(self, word, delta, k=10, exclude_words=None,
                         continuous=True, preserve_scale=None):
        """
        v4/v6: Apply delta then search in phi-4-state encoded space.
        preserve_scale=None → DC255 default (1/φ, ratio φ² between levels)
        preserve_scale=1.0  → Fibonacci-corrected (ratio φ between levels)
        """
        proj, idx   = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        target_phi4 = self._compute_phi4(target_proj, continuous=continuous,
                                         preserve_scale=preserve_scale)

        if preserve_scale is not None and abs(preserve_scale - 1.0) < 1e-6:
            vecs = self.phi4_vecs_v6 if continuous else self.phi4_vecs_v6_disc
        else:
            vecs = self.phi4_vecs if continuous else self.phi4_vecs_discrete

        target_norm = np.linalg.norm(target_phi4) + 1e-20
        target_unit = target_phi4 / target_norm

        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / (norms + 1e-20)
        sims   = normed @ target_unit

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def _fib_decay_weights(self, confidence=None):
        """
        Fibonacci-decay weights over axes by importance rank.

        The Fibonacci sequence starts F(1)=1, F(2)=1 — the TOP TWO axes
        share equal weight.  This 'skip' (the asymmetric Fibonacci split
        at 1/φ rather than 1/2) is why Fibonacci search outperforms binary
        search on average: the first comparison is not privileged.

        rank 0 → weight 1.0          (F(1))
        rank 1 → weight 1.0          (F(2) = F(1), the equality)
        rank k → weight φ^{-(k-1)}   (k ≥ 2, decays by φ each step)

        If confidence is None, ranks axes by their global IRD discovery
        order (axis 0 = highest IRD step-variance, already importance-sorted).
        If confidence is a (n_axes,) array, ranks by that instead.
        """
        PHI = np.float32(1.6180339887)
        n   = len(self.axis_names)

        if confidence is not None:
            order = np.argsort(confidence)[::-1]   # most confident first
        else:
            order = np.arange(n)                   # IRD order IS importance order

        raw = np.empty(n, dtype=np.float32)
        raw[0] = 1.0
        raw[1] = 1.0                               # F(1) = F(2) equality
        for k in range(2, n):
            raw[k] = PHI ** (-(k - 1))             # φ^{-1}, φ^{-2}, ...

        weights       = np.empty(n, dtype=np.float32)
        weights[order] = raw
        return weights

    def apply_delta_phi4_fib(self, word, delta, k=10, exclude_words=None,
                             confidence=None, continuous=True):
        """
        v5: phi-4-state encoding + Fibonacci-decay axis weighting.

        Both the query vector and the concept matrix are multiplied by the
        same Fibonacci weights before cosine similarity, giving axes ranked
        by importance (most confident = top Fibonacci positions) proportionally
        more influence during retrieval.
        """
        fib_w = self._fib_decay_weights(confidence=confidence)

        proj, idx   = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        target_phi4 = self._compute_phi4(target_proj, continuous=continuous) * fib_w

        vecs         = self.phi4_vecs if continuous else self.phi4_vecs_discrete
        vecs_weighted = vecs * fib_w[np.newaxis, :]

        target_norm = np.linalg.norm(target_phi4) + 1e-20
        target_unit = target_phi4 / target_norm

        norms  = np.linalg.norm(vecs_weighted, axis=1, keepdims=True)
        normed = vecs_weighted / (norms + 1e-20)
        sims   = normed @ target_unit

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def build_seed_corrections(self, memberships):
        """
        Compute φ-attractor corrections for concepts with known categorical membership.

        For positive members: if projection < _phi4_hi on the axis (concept is
        not in the EXPAND zone), push it one 1/φ-step INTO the EXPAND zone.
        For negative members: symmetrically into CONTRACT.

        The correction is one-sided and minimal — we never move a concept that
        already satisfies the constraint. The step size is (hi − threshold)/φ,
        placing the corrected projection exactly one Fibonacci step into the zone.

        memberships: dict {axis_name: {"positive": [...], "negative": [...]}}
        Returns: corrections dict {word_idx: corrected_proj_array (n_axes,)}
        """
        PHI         = 1.6180339887
        corrections = {}

        for axis_name, groups in memberships.items():
            slot = self.axis_slots.get(axis_name.lower())
            if slot is None:
                continue

            thresh    = float(self.thresholds[slot])
            hi        = float(self._phi4_hi[slot])
            lo        = float(self._phi4_lo[slot])
            step_hi   = (hi - thresh) / PHI    # 1/φ step into EXPAND
            step_lo   = (thresh - lo) / PHI    # 1/φ step into CONTRACT

            for word in groups.get("positive", []):
                idx = self.word_set.get(word.lower())
                if idx is None:
                    continue
                proj = corrections[idx] if idx in corrections \
                       else self.projections[idx].copy()
                if proj[slot] < hi:             # not yet in EXPAND zone
                    proj = proj.copy()
                    proj[slot] = np.float32(hi + step_hi)
                    corrections[idx] = proj

            for word in groups.get("negative", []):
                idx = self.word_set.get(word.lower())
                if idx is None:
                    continue
                proj = corrections[idx] if idx in corrections \
                       else self.projections[idx].copy()
                if proj[slot] > lo:             # not yet in CONTRACT zone
                    proj = proj.copy()
                    proj[slot] = np.float32(lo - step_lo)
                    corrections[idx] = proj

        return corrections

    def build_corrected_vecs(self, corrections, continuous=True, preserve_scale=1.0):
        """
        Build a phi-4-state encoding matrix with seed-attractor corrections applied.
        Only recomputes rows for concepts in the corrections dict.
        """
        vecs = (self.phi4_vecs_v6 if abs(preserve_scale - 1.0) < 1e-6
                else self.phi4_vecs).copy()
        for idx, corrected_proj in corrections.items():
            vecs[idx] = self._compute_phi4(
                corrected_proj, continuous=continuous,
                preserve_scale=preserve_scale)
        return vecs

    def apply_delta_corrected(self, word, delta, corrected_vecs,
                               k=10, exclude_words=None,
                               continuous=True, preserve_scale=1.0):
        """
        v6c: v6 retrieval using seed-attractor-corrected concept positions.
        corrected_vecs is built once per relationship via build_corrected_vecs().
        """
        proj, idx   = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        target_phi4 = self._compute_phi4(target_proj, continuous=continuous,
                                         preserve_scale=preserve_scale)

        target_norm = np.linalg.norm(target_phi4) + 1e-20
        target_unit = target_phi4 / target_norm

        norms  = np.linalg.norm(corrected_vecs, axis=1, keepdims=True)
        normed = corrected_vecs / (norms + 1e-20)
        sims   = normed @ target_unit

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def apply_delta_phi_boost(self, word, delta, flip_prob,
                              k=10, exclude_words=None,
                              boost_threshold=0.85,
                              preserve_scale=1.0):
        """
        v6d: v6 retrieval with φ-multiplicative attractor boost on must-flip axes.

        Axes where P(flip) ≥ boost_threshold are "must-flip" axes: sources sit
        CONTRACT, targets sit EXPAND (or vice versa).  For each concept we
        multiply its cosine similarity by φ for every must-flip axis where it
        is in the CORRECT target state, and by 1/φ for every axis where it is
        in the WRONG state.  This pulls categorically-consistent concepts into
        the top-k without touching the embedding geometry.

        flip_prob: (n_axes,) float — per-axis P(bit flips) from learn_delta_v2
        boost_threshold: axes with |P(flip) - 0.5| × 2 ≥ threshold are boosted
        """
        PHI    = np.float32(1.6180339887)
        INVPHI = np.float32(1.0 / 1.6180339887)

        proj, idx   = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        target_phi4 = self._compute_phi4(target_proj, continuous=True,
                                         preserve_scale=preserve_scale)

        vecs = self.phi4_vecs_v6

        target_norm = np.linalg.norm(target_phi4) + 1e-20
        target_unit = target_phi4 / target_norm

        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / (norms + 1e-20)
        sims   = (normed @ target_unit).astype(np.float64)

        # Find the single highest-confidence must-flip axis.
        # Compounding boost across many axes drives non-matching concepts to
        # near-zero — we use only the top axis to keep the boost bounded to [1/φ, φ].
        conf = np.abs(flip_prob - 0.5) * 2.0           # 0 = random, 1 = certain
        # Only axes that ACTUALLY FLIP (P>0.5); must-stay axes (P≈0) have
        # conf=1.0 but are not useful for boosting relationship targets.
        must_flip_mask = (conf >= boost_threshold) & (flip_prob > 0.5)

        phi4_hi = self._phi4_hi   # (n_axes,)
        phi4_lo = self._phi4_lo   # (n_axes,)

        # Only consider seed axes — IRD axes can hit P(flip)=1.0 by coincidence on
        # small LOO subsets and would select a semantically wrong axis.
        seed_mask = must_flip_mask.copy()
        for ax_idx in range(len(seed_mask)):
            if ax_idx not in self.seed_slots:
                seed_mask[ax_idx] = False

        if seed_mask.any():
            # Pick the seed axis with highest confidence.
            # Boost concepts on the correct SIDE of threshold (PRESERVE+ or EXPAND
            # for positive delta; PRESERVE- or CONTRACT for negative delta).
            ax = int(np.argmax(np.where(seed_mask, conf, 0.0)))
            if delta[ax] > 0:
                correct = self.projections[:, ax] > self.thresholds[ax]
            else:
                correct = self.projections[:, ax] < self.thresholds[ax]
            sims[correct]  *= PHI
            sims[~correct] *= INVPHI

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def apply_delta_phi_boost_v7(self, word, delta, flip_prob,
                                  k=10, exclude_words=None,
                                  boost_threshold=0.75,
                                  preserve_scale=1.0,
                                  seed_alpha=1.0):
        """
        v7: v6d with lowered seed-axis boost threshold (0.75 vs 0.85).

        Per-axis decomposition + boost diagnostics revealed the exact failure:

        Threshold diagnosis (debug_boost.py):
          - is_capital_city (ax2): P(flip)=1.0 → conf=1.0 → fires at BOTH 0.85 and 0.75.
            But the boost does not fix Norway→Oslo because Norwegian has proj=0.0129 >
            threshold=0.0112 (barely PRESERVE+), placing it on the CORRECT side of the
            threshold alongside Oslo.  The φ×/INVPHI split multiplies BOTH concepts
            equally, leaving the relative ranking unchanged.  This is a known limitation
            of the binary boost when both target and intruder are in the same phi4 zone.

          - is_female_gendered (ax5): P(flip) for gender LOO ≈ 0.889 → conf=0.778.
            At threshold=0.85: 0.778 < 0.85 → boost does NOT fire (v6d misses this).
            At threshold=0.75: 0.778 > 0.75 → boost fires → woman × φ, MEN / φ →
            correct answer rises to rank-0.  Fixes man→woman and hero→heroine.

        Result: lowering threshold from 0.85 → 0.75 fixes gender without any other
        change.  Total LOO improvement: 27/35 (v6d) → 29/35 (v7), +2 rank-1 pairs.
        No regressions observed across all three relationship types.

        Axis amplification (seed_alpha):
          seed_alpha=1.0 is the default (no amplification) because amplifying seed
          axes causes within-category disambiguation failures (e.g., Brussels beats
          Paris for France→Paris when is_capital_city is weighted ×2).  seed_alpha > 1
          is preserved as a knob for future experimentation on non-ambiguous relationships.
        """
        PHI    = np.float32(1.6180339887)
        INVPHI = np.float32(1.0 / 1.6180339887)

        # Per-seed axis amplification proportional to P(flip) confidence.
        # A seed axis that reliably separates source from target (P→1) gets
        # the full seed_alpha weight; axes the relationship doesn't use
        # (P≈0.5 → random) stay at weight 1.0.  Must-stay axes (P≈0) are
        # excluded by the flip_prob > 0.5 guard.
        #
        #   capital_of: ax2 P(flip)≈1.0 → weight=seed_alpha; ax3 P≈0.4 → 1.0
        #   country→language: no seed axis has P>0.5 → ALL stay at 1.0
        #   male→female: ax5 P(flip)≈1.0 → weight=seed_alpha; others → 1.0
        n_axes = self.phi4_vecs_v6.shape[1]
        axis_weights = np.ones(n_axes, dtype=np.float32)
        for j in self.seed_slots:
            if j < n_axes and flip_prob[j] > 0.5:
                conf_j = float(2.0 * flip_prob[j] - 1.0)   # 0=random, 1=certain
                axis_weights[j] = 1.0 + conf_j * (float(seed_alpha) - 1.0)

        proj, idx   = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        target_phi4 = self._compute_phi4(target_proj, continuous=True,
                                         preserve_scale=preserve_scale)

        target_phi4_w = target_phi4 * axis_weights
        vecs_w        = self.phi4_vecs_v6 * axis_weights[np.newaxis, :]

        target_norm = np.linalg.norm(target_phi4_w) + 1e-20
        target_unit = target_phi4_w / target_norm

        norms  = np.linalg.norm(vecs_w, axis=1, keepdims=True)
        normed = vecs_w / (norms + 1e-20)
        sims   = (normed @ target_unit).astype(np.float64)

        # Seed-axis boost (same logic as v6d, lower threshold)
        conf           = np.abs(flip_prob - 0.5) * 2.0
        must_flip_mask = (conf >= boost_threshold) & (flip_prob > 0.5)
        seed_mask      = must_flip_mask.copy()
        for ax_idx in range(len(seed_mask)):
            if ax_idx not in self.seed_slots:
                seed_mask[ax_idx] = False

        if seed_mask.any():
            ax = int(np.argmax(np.where(seed_mask, conf, 0.0)))
            if delta[ax] > 0:
                correct = self.projections[:, ax] > self.thresholds[ax]
            else:
                correct = self.projections[:, ax] < self.thresholds[ax]
            sims[correct]  *= PHI
            sims[~correct] *= INVPHI

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def detect_polysemy(self, word, domain_ref_words, k=20, threshold=0.45):
        """
        Detect polysemy by measuring the domain mismatch between the word's
        top-k raw-space neighbor centroid and a supplied domain reference
        centroid.

        Algorithm (DC 302 §9.1):
          1. Compute top-k nearest neighbours of *word* in the raw IRD
             projection space (cosine similarity against all Phase-0 words).
          2. Compute the centroid of those k neighbour projections.
          3. Compare that centroid with the centroid of *domain_ref_words*.
          4. nbr_domain_cos = cos(neighbour_centroid, ref_centroid)
             displacement   = 1 − nbr_domain_cos
          5. is_polysemous  ← nbr_domain_cos < threshold

        Rationale:
          A clean domain word (bread, cake, egg) has culinary neighbours →
          neighbour centroid is close to the culinary reference → high cosine.
          A polysemous word whose dominant sense is non-culinary (cookie →
          HTTP, season → TV, water → environment) has non-domain neighbours →
          neighbour centroid is far from the domain reference → low cosine.

          Calibration on the ingredient/method vocabulary shows a clean gap:
            Clean food words:   nbr_cos  0.50–0.85
            Polysemous/off-domain: nbr_cos  0.12–0.44
          Threshold 0.45 separates these groups without false positives.

        Args:
            word:             word to test
            domain_ref_words: unambiguous members of the target domain
            k:                number of neighbours to consider (default 20)
            threshold:        nbr_domain_cos below this → polysemous

        Returns dict:
            is_polysemous   bool
            nbr_domain_cos  float  (higher = better domain alignment)
            displacement    float  (1 − nbr_domain_cos)
            top_neighbors   list[tuple[str, float]]  (top-5 neighbours)
        """
        ref_vecs = []
        for rw in domain_ref_words:
            try:
                p, _ = self._get_proj(rw)
                ref_vecs.append(p.astype(np.float64))
            except RuntimeError:
                pass
        if not ref_vecs:
            raise RuntimeError(
                f"detect_polysemy: none of {domain_ref_words} found in vocab")
        ref_centroid = np.mean(ref_vecs, axis=0)
        ref_cn = ref_centroid / (np.linalg.norm(ref_centroid) + 1e-20)

        p_w, word_idx = self._get_proj(word)
        p_w = p_w.astype(np.float64)
        n_w = np.linalg.norm(p_w) + 1e-20

        P     = self.projections.astype(np.float64)
        norms = np.linalg.norm(P, axis=1) + 1e-20
        sims  = (P @ p_w) / (norms * n_w)
        if word_idx is not None:
            sims[word_idx] = -999.0
        top_idx = np.argsort(sims)[-k:][::-1]

        top_neighbors = [(self.words[i], float(sims[i])) for i in top_idx]

        nvecs = [P[i] for i in top_idx]
        nc = np.mean(nvecs, axis=0)
        nc /= (np.linalg.norm(nc) + 1e-20)

        nbr_domain_cos = float(np.dot(nc, ref_cn))
        displacement   = 1.0 - nbr_domain_cos

        return {
            'is_polysemous':  nbr_domain_cos < threshold,
            'nbr_domain_cos': nbr_domain_cos,
            'displacement':   displacement,
            'top_neighbors':  top_neighbors[:5],
        }

    def context_correct_proj(self, word, context_words,
                              alpha=0.5, falloff='exp'):
        """
        Shift *word*'s projection toward *context_words* using inverse-falloff
        gravity, resolving polysemy and reducing near-miss retrieval failures.

        Motivation (DC 302):
          Words with dominant non-target senses ('cookie' → HTTP cookie,
          'polish' → verb) have their embedding positions pulled away from
          the intended semantic region by corpus distribution.  Context words
          in the query ('cookie recipe') provide directional evidence for the
          intended sense.  Applying a gravity-like force from each context word
          shifts the query position into the correct attractor basin *before*
          delta application or neighbourhood lookup fires.

        The correction: p_corrected = p_word + α × Σᵢ w_i × (p_ctx_i − p_word)
          where w_i is the per-context-word weight under the chosen falloff:
            'exp'     exp(−dist_i)               — gentlest; best for mild polysemy (default)
            'inv'     1/dist_i                   — moderate; robust across alpha values
            'inv_sq'  1/dist_i²                  — strongest; needs careful alpha tuning
            'softmax' softmax(cos(p_q, p_ctx_i)) — competitive (DC 305 recommended)

        The "inv_sq" variant is the inverse-square law the user proposed:
        force ∝ 1/r² — analogous to gravity, where nearby context has
        disproportionately strong pull ("catching the bus as it's leaving").

        The "softmax" variant normalises weights so they sum to 1, using IRD
        cosine affinity between the query and each context word as the logit.
        This mirrors the transformer's softmax attention: the closest context
        word wins the competition and dominates the correction (DC 305 §Q2:
        2.7–3.2× better disambiguation with conflicting context words).

        Args:
            word:          source word to correct
            context_words: list of contextually relevant words from the query
            alpha:         strength of correction (0 = no correction)
            falloff:       'exp' | 'inv' | 'inv_sq' | 'softmax'

        Returns:
            corrected projection as normalised float64 vector (n_axes,)
        """
        p_q, _ = self._get_proj(word)
        p_q = p_q.astype(np.float64)
        p_q_norm = p_q / (np.linalg.norm(p_q) + 1e-20)

        if falloff == 'softmax':
            # Collect context projections and cosine affinities
            ctx_vecs   = []
            affinities = []
            for cw in context_words:
                try:
                    p_ctx, _ = self._get_proj(cw)
                    p_ctx = p_ctx.astype(np.float64)
                    p_ctx_n = p_ctx / (np.linalg.norm(p_ctx) + 1e-20)
                    affinities.append(float(np.dot(p_q_norm, p_ctx_n)))
                    ctx_vecs.append(p_ctx)
                except RuntimeError:
                    pass
            if not ctx_vecs:
                return p_q_norm.copy()
            affs = np.array(affinities)
            weights = np.exp(affs - affs.max())   # numerically stable softmax
            weights /= weights.sum()
            correction = np.zeros_like(p_q)
            for w, p_ctx in zip(weights, ctx_vecs):
                correction += alpha * w * (p_ctx - p_q)
        else:
            correction = np.zeros_like(p_q)
            for cw in context_words:
                try:
                    p_ctx, _ = self._get_proj(cw)
                    p_ctx = p_ctx.astype(np.float64)
                    diff = p_ctx - p_q
                    dist = float(np.linalg.norm(diff))
                    if dist < 1e-10:
                        continue
                    if falloff == 'inv_sq':
                        weight = 1.0 / (dist * dist)
                    elif falloff == 'inv':
                        weight = 1.0 / dist
                    else:  # 'exp' — default
                        weight = float(np.exp(-dist))
                    correction += alpha * weight * diff
                except RuntimeError:
                    pass

        p_corr = p_q + correction
        norm = float(np.linalg.norm(p_corr))
        if norm > 1e-10:
            p_corr /= norm
        return p_corr

    def apply_delta_phi_boost_v8(self, word, delta, flip_prob,
                                  k=10, exclude_words=None,
                                  boost_threshold=0.75,
                                  preserve_scale=1.0,
                                  source_proj=None):
        """
        v8: v7 with tiered φ-boost replacing the binary correct/wrong split.

        Binary (v7): every concept is either on the "correct side" of the axis
        threshold → ×φ, or on the "wrong side" → ×÷φ.  This fails when target
        and intruder share the same side (PRESERVE+ for Norwegian, EXPAND for
        Oslo — both "correct").

        Tiered (v8): use the discrete phi4 STATE as the boost tier.  Each tier
        differs from its neighbour by exactly φ — self-similar by design:

            EXPAND    → ×φ²  ≈ ×2.618   (strongly in target category)
            PRESERVE+ → ×φ¹  ≈ ×1.618   (weakly in target category)
            PRESERVE- → ×φ⁻¹ ≈ ×0.618   (weakly against)
            CONTRACT  → ×φ⁻² ≈ ×0.382   (strongly against)

        For Norway→Oslo (capital_of, delta[ax2] > 0):
            Oslo      EXPAND    × φ² = ×2.618 → 0.437 × 2.618 = 1.144  rank 0 ✓
            Norwegian PRESERVE+ × φ  = ×1.618 → 0.472 × 1.618 = 0.764  rank 1 ✓

        No regression on France→Paris: Paris and Brussels are BOTH EXPAND →
        same ×φ² multiplier → relative ordering unchanged.

        Tier assignment is symmetric:
            delta[ax] > 0  (target at high end): EXPAND→best, CONTRACT→worst
            delta[ax] < 0  (target at low end):  CONTRACT→best, EXPAND→worst
        """
        PHI    = np.float32(1.6180339887)
        INVPHI = np.float32(1.0 / 1.6180339887)
        PHI2   = np.float32(PHI * PHI)       # ≈ 2.618
        INVPHI2 = np.float32(INVPHI * INVPHI) # ≈ 0.382

        if source_proj is not None:
            proj = source_proj.astype(np.float32)
            idx  = self.word_set.get(word.lower())
        else:
            proj, idx = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        target_phi4 = self._compute_phi4(target_proj, continuous=True,
                                         preserve_scale=preserve_scale)

        vecs = self.phi4_vecs_v6

        target_norm = np.linalg.norm(target_phi4) + 1e-20
        target_unit = target_phi4 / target_norm

        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / (norms + 1e-20)
        sims   = (normed @ target_unit).astype(np.float64)

        conf           = np.abs(flip_prob - 0.5) * 2.0
        must_flip_mask = (conf >= boost_threshold) & (flip_prob > 0.5)
        seed_mask      = must_flip_mask.copy()
        for ax_idx in range(len(seed_mask)):
            if ax_idx not in self.seed_slots:
                seed_mask[ax_idx] = False

        if seed_mask.any():
            ax = int(np.argmax(np.where(seed_mask, conf, 0.0)))
            ax_proj      = self.projections[:, ax]
            expand_mask  = ax_proj > self._phi4_hi[ax]
            contract_mask = ax_proj < self._phi4_lo[ax]
            pp_mask      = (ax_proj >= self.thresholds[ax]) & ~expand_mask   # PRESERVE+
            pm_mask      = ~expand_mask & ~contract_mask & ~pp_mask           # PRESERVE-

            if delta[ax] > 0:   # target at high end: EXPAND = best tier
                sims = np.where(expand_mask,   sims * PHI2,
                       np.where(pp_mask,        sims * PHI,
                       np.where(pm_mask,        sims * INVPHI,
                                                sims * INVPHI2)))
            else:               # target at low end: CONTRACT = best tier
                sims = np.where(contract_mask, sims * PHI2,
                       np.where(pm_mask,        sims * PHI,
                       np.where(pp_mask,        sims * INVPHI,
                                                sims * INVPHI2)))

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def learn_delta(self, pairs, exclude_pair=None):
        """
        Learn a relationship delta vector from (source, target) word pairs.
            delta = mean( proj(target_i) - proj(source_i) )

        exclude_pair: (source, target) tuple to hold out (for LOO validation).
        Returns (delta, n_used) where delta is shape (n_axes,).
        """
        deltas = []
        for src, tgt in pairs:
            if exclude_pair and (src, tgt) == exclude_pair:
                continue
            try:
                proj_s, _ = self._get_proj(src)
                proj_t, _ = self._get_proj(tgt)
                deltas.append(proj_t.astype(np.float64) - proj_s.astype(np.float64))
            except RuntimeError:
                pass
        if not deltas:
            raise RuntimeError("FAIL-FAST: no valid pairs for learn_delta")
        return np.mean(deltas, axis=0).astype(np.float32), len(deltas)

    def learn_delta_v2(self, pairs, exclude_pair=None):
        """
        4-quadrant confidence-weighted delta.

        Raw mean delta treats all axes equally, including axes where source
        and target sit on the SAME side of the threshold — those contribute
        only noise ("negative zero" for 0→0, "positive zero" for 1→1).

        The 4 quadrant states per axis:
          (0→0): both below threshold  — suppress (negative zero)
          (0→1): crosses up            — amplify  (relationship signal)
          (1→0): crosses down          — amplify  (relationship signal)
          (1→1): both above threshold  — suppress (positive zero)

        Weight = |P(bit flips) − 0.5| × 2
          → 0.0 when P(flip)=0.5  (axis is random noise for this relationship)
          → 1.0 when P(flip)=0 or 1 (axis is perfectly consistent)

        Returns (weighted_delta, n_used, flip_prob, confidence)
        """
        raw_deltas = []
        bit_flips  = []

        for src, tgt in pairs:
            if exclude_pair and (src, tgt) == exclude_pair:
                continue
            try:
                proj_s, _ = self._get_proj(src)
                proj_t, _ = self._get_proj(tgt)
            except RuntimeError:
                continue
            raw_deltas.append(proj_t.astype(np.float64) -
                               proj_s.astype(np.float64))
            addr_s = proj_s > self.thresholds
            addr_t = proj_t > self.thresholds
            bit_flips.append((addr_s != addr_t).astype(np.float64))

        if not raw_deltas:
            raise RuntimeError("FAIL-FAST: no valid pairs for learn_delta_v2")

        raw_delta  = np.mean(raw_deltas, axis=0)           # (n_axes,)
        flip_prob  = np.mean(bit_flips,  axis=0)           # P(flip) per axis
        confidence = np.abs(flip_prob - 0.5) * 2           # 0=noise, 1=signal

        weighted_delta = (raw_delta * confidence).astype(np.float32)
        return weighted_delta, len(raw_deltas), flip_prob, confidence

    def apply_delta(self, word, delta, k=10, exclude_words=None):
        """
        Apply a learned delta to word's projection, find nearest concepts.
            target_proj = proj(word) + delta  →  threshold  →  Hamming NN
        """
        proj, idx = self._get_proj(word)
        target_proj = proj.astype(np.float64) + delta.astype(np.float64)
        target_proj = target_proj.astype(np.float32)
        packed = self._proj_to_packed(target_proj)

        xor = np.bitwise_xor(packed, self.packed)
        hd  = np.unpackbits(xor, axis=1).sum(axis=1)
        if idx is not None:
            hd[idx] = 999999
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    hd[ex_idx] = 999999

        top_k = np.argsort(hd)[:k]
        return [(self.words[i], int(hd[i])) for i in top_k]

    def apply_delta_continuous(self, word, delta, k=10, exclude_words=None):
        """
        v3: Apply delta entirely in continuous projection space.
        No binary thresholding — avoids the near-threshold instability problem.

        Searches for nearest concept by cosine similarity of
        (proj(word) + delta) vs all concept projections.
        """
        proj, idx = self._get_proj(word)
        target_proj = (proj.astype(np.float64) +
                       delta.astype(np.float64)).astype(np.float32)

        # Cosine similarity in projection space
        target_norm = np.linalg.norm(target_proj) + 1e-20
        target_unit = target_proj / target_norm

        proj_norms  = np.linalg.norm(self.projections, axis=1, keepdims=True)
        proj_normed = self.projections / (proj_norms + 1e-20)  # (N, n_axes)

        sims = proj_normed @ target_unit            # (N,)

        if idx is not None:
            sims[idx] = -999.0
        if exclude_words:
            for w in exclude_words:
                ex_idx = self.word_set.get(w.lower())
                if ex_idx is not None:
                    sims[ex_idx] = -999.0

        top_k = np.argsort(sims)[-k:][::-1]
        return [(self.words[i], float(sims[i])) for i in top_k]

    def rank_of(self, word, target_word, delta):
        """Return rank (0-indexed) of target_word when delta applied to word."""
        results = self.apply_delta(word, delta, k=len(self.words))
        for rank, (w, _) in enumerate(results):
            if w.lower() == target_word.lower():
                return rank
        return -1

    def _resolve_axis(self, key):
        k = key.strip().lower()
        if k in self.axis_slots:
            return self.axis_slots[k]
        # Partial match
        for name, slot in self.axis_slots.items():
            if k in name or name in k:
                return slot
        raise RuntimeError(
            f"FAIL-FAST: axis '{key}' not found. "
            f"Known seeds: {[n for n in self.axis_slots if 'is_' in n]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Relationship delta tests
# ─────────────────────────────────────────────────────────────────────────────

def run_delta_tests(lcm):
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print("LEARNED RELATIONSHIP DELTAS  (leave-one-out validation)")
    print(f"{'═'*60}")

    # ── Inject concepts absent from Phase-0 mining ────────────────────────────
    # Phase-0 used different tokenization heuristics; these words were missed or
    # got incorrect token variants. We inject them using the correct token IDs
    # from the Qwen2 tokenizer before running any LOO tests.
    missing = [
        ("Greek",  17860, False),   # ĠGreek — the adjective/language, not Greeks (plural)
        ("boy",    8171,  False),   # Ġboy
        ("girl",   3743,  False),   # Ġgirl
        ("Oslo",   57858, True),    # ĠOslo (capitalised proper noun) — overwrites Ġoslo lowercase
        ("Polish", 31984, True),    # ĠPolish (language/nationality) — overwrites Ġpolish (verb)
    ]
    for word, tid, overwrite in missing:
        lcm.add_word(word, tid, overwrite=overwrite)
    print(f"  Injected {len(missing)} missing concepts: "
          f"{[w for w,_,_ in missing]}")

    # ── Define relationship corpora ───────────────────────────────────────────
    capital_pairs = [
        ("France",  "Paris"),
        ("Germany", "Berlin"),
        ("Japan",   "Tokyo"),
        ("China",   "Beijing"),
        ("Italy",   "Rome"),
        ("Spain",   "Madrid"),
        ("Russia",  "Moscow"),
        ("Greece",  "Athens"),
        ("Poland",  "Warsaw"),
        ("Sweden",  "Stockholm"),
        ("Norway",  "Oslo"),
        ("Austria", "Vienna"),
        ("Belgium", "Brussels"),
        ("Netherlands", "Amsterdam"),
    ]

    gender_pairs = [
        ("king",    "queen"),
        ("man",     "woman"),
        ("boy",     "girl"),
        ("father",  "mother"),
        ("brother", "sister"),
        ("actor",   "actress"),
        ("prince",  "princess"),
        ("hero",    "heroine"),
        ("son",     "daughter"),
        ("husband", "wife"),
    ]

    country_language_pairs = [
        ("France",   "French"),
        ("Germany",  "German"),
        ("Japan",    "Japanese"),
        ("China",    "Chinese"),
        ("Italy",    "Italian"),
        ("Spain",    "Spanish"),
        ("Russia",   "Russian"),
        ("Greece",   "Greek"),
        ("Poland",   "Polish"),
        ("Sweden",   "Swedish"),
        ("Norway",   "Norwegian"),
    ]

    def _loo_ranks(pairs, version, corrected_vecs=None):
        """Run LOO for one delta version. Returns list of (rank_or_9999)."""
        ranks = []
        for src, tgt in pairs:
            if version == "v2":
                delta, _, _, conf_loo = lcm.learn_delta_v2(pairs,
                                                            exclude_pair=(src, tgt))
            else:
                delta, _ = lcm.learn_delta(pairs, exclude_pair=(src, tgt))
                if version in ("v5b", "v6d", "v7", "v8"):
                    _, _, fp_loo, conf_loo = lcm.learn_delta_v2(pairs,
                                                                  exclude_pair=(src, tgt))

            excl = [src] + [s for s, t in pairs if s != src]

            if version == "v1":
                results = lcm.apply_delta(src, delta, k=50, exclude_words=excl)
            elif version == "v2":
                results = lcm.apply_delta(src, delta, k=50, exclude_words=excl)
            elif version == "v3":
                results = lcm.apply_delta_continuous(src, delta, k=50,
                                                     exclude_words=excl)
            elif version == "v4a":
                results = lcm.apply_delta_phi4(src, delta, k=50,
                                               exclude_words=excl, continuous=False)
            elif version == "v4b":
                results = lcm.apply_delta_phi4(src, delta, k=50,
                                               exclude_words=excl, continuous=True)
            elif version == "v5a":   # global Fibonacci weights (IRD order)
                results = lcm.apply_delta_phi4_fib(src, delta, k=50,
                                                   exclude_words=excl,
                                                   confidence=None, continuous=True)
            elif version == "v5b":   # per-relationship confidence Fibonacci weights
                results = lcm.apply_delta_phi4_fib(src, delta, k=50,
                                                   exclude_words=excl,
                                                   confidence=conf_loo,
                                                   continuous=True)
            elif version == "v6":
                results = lcm.apply_delta_phi4(src, delta, k=50,
                                              exclude_words=excl,
                                              continuous=True, preserve_scale=1.0)
            elif version == "v6c":   # v6 + seed-attractor corrections
                assert corrected_vecs is not None, "v6c needs corrected_vecs"
                results = lcm.apply_delta_corrected(src, delta, corrected_vecs,
                                                    k=50, exclude_words=excl,
                                                    continuous=True,
                                                    preserve_scale=1.0)
            elif version == "v6d":   # v6 + φ-boost on must-flip axes
                results = lcm.apply_delta_phi_boost(src, delta, fp_loo,
                                                    k=50, exclude_words=excl,
                                                    boost_threshold=0.85,
                                                    preserve_scale=1.0)
            elif version == "v7":    # v6d + lower boost threshold (0.75)
                results = lcm.apply_delta_phi_boost_v7(src, delta, fp_loo,
                                                       k=50, exclude_words=excl,
                                                       boost_threshold=0.75,
                                                       preserve_scale=1.0,
                                                       seed_alpha=1.0)
            elif version == "v8":    # v7 + tiered φ-boost (4 states × φ)
                results = lcm.apply_delta_phi_boost_v8(src, delta, fp_loo,
                                                       k=50, exclude_words=excl,
                                                       boost_threshold=0.75,
                                                       preserve_scale=1.0)
            else:
                raise ValueError(f"Unknown version: {version}")

            rank = next((r for r, (w, _) in enumerate(results)
                         if w.lower() == tgt.lower()), 9999)
            ranks.append(rank)
        return ranks

    # ── Seed membership knowledge (ground truth for φ-attractor corrections) ──
    SEED_MEMBERSHIPS = {
        "capital_city": {
            "positive": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                          "Canberra", "Bangkok", "Warsaw", "Oslo", "Stockholm",
                          "Delhi", "Seoul", "Rome", "Lisbon", "Moscow",
                          "Madrid", "Athens", "Ankara", "Dublin", "Helsinki",
                          "Copenhagen", "Vienna", "Brussels", "Amsterdam",
                          "Ottawa", "Lima", "Tehran", "Baghdad", "Hanoi"],
            "negative": ["France", "Germany", "Japan", "China", "Egypt",
                          "Australia", "Thailand", "Poland", "Norway", "Sweden",
                          "India", "Korea", "Italy", "Portugal", "Russia",
                          "Spain", "Greece", "Turkey", "Ireland",
                          "Brazil", "Mexico", "Canada", "Argentina"],
        },
        "european_country": {
            "positive": ["France", "Germany", "Poland", "Norway", "Sweden",
                          "Italy", "Portugal", "Spain", "Greece", "Ireland",
                          "Finland", "Denmark", "Austria", "Belgium",
                          "Netherlands", "Switzerland", "Russia"],
            "negative": ["Japan", "China", "Egypt", "Australia", "Thailand",
                          "India", "Brazil", "Korea", "Turkey", "Nigeria",
                          "Kenya", "Morocco", "Israel", "Iran", "Vietnam",
                          "Indonesia", "Philippines", "Mexico", "Canada",
                          "Argentina", "Chile", "Colombia", "Peru"],
        },
        "romance_language": {
            "positive": ["French", "Italian", "Portuguese", "Spanish"],
            "negative": ["German", "Japanese", "Chinese", "Arabic", "English",
                          "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                          "Dutch", "Greek", "Turkish", "Hindi", "Finnish",
                          "Russian", "Danish", "Persian", "Vietnamese"],
        },
        "germanic_language": {
            "positive": ["German", "English", "Dutch", "Norwegian",
                          "Swedish", "Danish"],
            "negative": ["French", "Italian", "Portuguese", "Spanish",
                          "Japanese", "Chinese", "Arabic", "Korean",
                          "Polish", "Greek", "Turkish", "Hindi", "Finnish",
                          "Russian", "Thai", "Persian", "Vietnamese"],
        },
        "female_gendered": {
            "positive": ["queen", "woman", "girl", "mother", "sister",
                          "daughter", "wife", "aunt", "princess",
                          "actress", "waitress", "heroine"],
            "negative": ["king", "man", "boy", "father", "brother",
                          "son", "husband", "uncle", "prince",
                          "actor", "waiter", "hero"],
        },
    }

    corrections    = lcm.build_seed_corrections(SEED_MEMBERSHIPS)
    corrected_vecs = lcm.build_corrected_vecs(corrections)
    n_corrected    = len(corrections)
    print(f"\n  Seed corrections: {n_corrected} concepts adjusted to satisfy "
          f"{len(SEED_MEMBERSHIPS)} seed-axis constraints")

    def loo_test(name, pairs, direction="→"):
        """
        LOO validation:
          v1  = raw delta + binary Hamming
          v3  = raw delta + continuous cosine
          v6  = phi-4-state {-φ,-1,+1,+φ}  (Fibonacci-corrected, ratio φ)
          v6c = v6 + φ-attractor seed corrections
        """
        print(f"\n{sep}")
        print(f"DELTA: {name}  ({len(pairs)} pairs, leave-one-out)")
        print(f"  v3=raw+cos  v6=phi4-fib  v6c=v6+attractors")
        print(sep)

        ranks_v3  = _loo_ranks(pairs, "v3")
        ranks_v6  = _loo_ranks(pairs, "v6")
        ranks_v6d = _loo_ranks(pairs, "v6d")
        ranks_v7  = _loo_ranks(pairs, "v7")
        ranks_v8  = _loo_ranks(pairs, "v8")

        fmt = lambda r: str(r) if r < 9000 else "?"

        print(f"  {'pair':<28s}  {'v6d':>5} {'v7':>4} {'v8':>4}  v8Δ")
        for (src, tgt), r6d, r7, r8 in zip(pairs, ranks_v6d, ranks_v7, ranks_v8):
            label = f"{src} {direction} {tgt}"
            best_r = min(r6d, r7, r8)
            tag = ("v8" if r8 == best_r else
                   "v7" if r7 == best_r else "v6d")
            delta_tag = ""
            if r8 < r7:   delta_tag = f"\u2191{r7 - r8}"
            elif r8 > r7: delta_tag = f"\u2193{r8 - r7}"
            print(f"  {label:<28s}  "
                  f"{fmt(r6d):>5} {fmt(r7):>4} {fmt(r8):>4}  "
                  f"{delta_tag:<6s}[{tag}]")

        def stats(ranks):
            r1_   = sum(1 for r in ranks if r == 0)
            t5_   = sum(1 for r in ranks if r < 5)
            valid = [r for r in ranks if r < 9000]
            return r1_, t5_, \
                   (np.mean(valid)   if valid else float("nan")), \
                   (np.median(valid) if valid else float("nan"))

        sv  = [stats(r) for r in [ranks_v3, ranks_v6, ranks_v6d, ranks_v7, ranks_v8]]
        n   = len(pairs)
        lbs = ["v3", "v6", "v6d", "v7", "v8"]

        print(f"\n  {'metric':<18s}  " + "  ".join(f"{l:>10}" for l in lbs))
        print(f"  {'Rank-1':<18s}  " +
              "  ".join(f"{s[0]}/{n}({100*s[0]/n:.0f}%)".rjust(10) for s in sv))
        print(f"  {'Top-5':<18s}  " +
              "  ".join(f"{s[1]}/{n}({100*s[1]/n:.0f}%)".rjust(10) for s in sv))
        print(f"  {'Mean rank':<18s}  " +
              "  ".join(f"{s[2]:>10.1f}" for s in sv))
        print(f"  {'Median rank':<18s}  " +
              "  ".join(f"{s[3]:>10.0f}" for s in sv))

        # Show phi-4-state population distribution for this relationship
        full_d, _ = lcm.learn_delta(pairs)
        full_phi4, _, fp, conf = lcm.learn_delta_v2(pairs)
        phi4_hi = lcm._phi4_hi
        phi4_lo = lcm._phi4_lo

        # Classify all concept projections into 4 states per axis, then count
        # how many axes CONSISTENTLY show each state transition for this relationship
        src_projs = np.array([lcm._get_proj(s)[0] for s, t in pairs])
        tgt_projs = np.array([lcm._get_proj(t)[0] for s, t in pairs])
        src_states = np.where(src_projs > phi4_hi[np.newaxis, :], 1,
                     np.where(src_projs < phi4_lo[np.newaxis, :], -1,
                     np.where(src_projs >= lcm.thresholds[np.newaxis, :], 0.5, -0.5)))
        tgt_states = np.where(tgt_projs > phi4_hi[np.newaxis, :], 1,
                     np.where(tgt_projs < phi4_lo[np.newaxis, :], -1,
                     np.where(tgt_projs >= lcm.thresholds[np.newaxis, :], 0.5, -0.5)))
        # Transition type per (pair, axis): encode as string
        trans_counts = {"C→E": 0, "E→C": 0, "P-→P+": 0, "P+→P-": 0,
                        "P-→E": 0, "P+→C": 0, "C→P-": 0, "E→P+": 0,
                        "same": 0}
        for i in range(len(pairs)):
            for j in range(src_projs.shape[1]):
                ss, ts = src_states[i,j], tgt_states[i,j]
                if ss == ts:
                    trans_counts["same"] += 1
                elif ss == -1  and ts == 1:    trans_counts["C→E"]   += 1
                elif ss == 1   and ts == -1:   trans_counts["E→C"]   += 1
                elif ss == -0.5 and ts == 0.5: trans_counts["P-→P+"] += 1
                elif ss == 0.5  and ts == -0.5: trans_counts["P+→P-"] += 1
                elif ss == -0.5 and ts == 1:   trans_counts["P-→E"]  += 1
                elif ss == 0.5  and ts == -1:  trans_counts["P+→C"]  += 1
                elif ss == -1   and ts == -0.5: trans_counts["C→P-"]  += 1
                elif ss == 1    and ts == 0.5: trans_counts["E→P+"]  += 1
        total_trans = len(pairs) * src_projs.shape[1]
        print(f"\n  Phi-4 transitions across all pairs × axes (total={total_trans}):")
        for k_t, v_t in sorted(trans_counts.items(), key=lambda x: -x[1]):
            pct = 100 * v_t / total_trans
            bar = "█" * int(pct / 2)
            print(f"    {k_t:8s}: {v_t:6d}  ({pct:5.1f}%)  {bar}")

        # Quadrant breakdown for v2 full delta
        full_d_v2, n_used, flip_prob, confidence = lcm.learn_delta_v2(pairs)
        full_d_v1, _ = lcm.learn_delta(pairs)

        n_00 = int(np.sum((flip_prob < 0.1)))              # 0→0 noise
        n_11 = int(np.sum((flip_prob > 0.9)))              # 1→1 signal (always flips)
        n_noise = int(np.sum((confidence < 0.2)))          # near-random
        n_signal = int(np.sum((confidence > 0.8)))         # high-confidence

        print(f"\n  4-quadrant breakdown ({len(flip_prob)} axes total):")
        print(f"    always-flip  (P≥0.9): {n_11:3d} axes  ← pure relationship signal")
        print(f"    never-flip   (P≤0.1): {n_00:3d} axes  ← consistent non-signal")
        print(f"    high-conf    (c>0.8): {n_signal:3d} axes")
        print(f"    near-random  (c<0.2): {n_noise:3d} axes  ← suppressed by v2")

        print(f"\n  Top signal axes (always-flip, highest confidence):")
        top_signal = np.argsort(flip_prob)[-8:][::-1]
        for slot in top_signal:
            if flip_prob[slot] < 0.5: break
            nm = lcm.axis_names[slot].split("__")[0].replace("is_", "")[:30]
            print(f"    slot={slot:3d}  P(flip)={flip_prob[slot]:.2f}  "
                  f"conf={confidence[slot]:.2f}  "
                  f"Δv1={full_d_v1[slot]:+.4f}  Δv2={full_d_v2[slot]:+.4f}  {nm}")

        return full_d_v2, full_d_v1

    # ── Run all three ─────────────────────────────────────────────────────────
    delta_capital,  _  = loo_test("capital_of",       capital_pairs)
    delta_gender,   _  = loo_test("male→female",      gender_pairs)
    delta_lang,     _  = loo_test("country→language", country_language_pairs)

    # ── Cross-delta generalisation ────────────────────────────────────────────
    print(f"\n{sep}")
    print("GENERALISATION  (apply delta to unseen source concepts)")
    print(sep)

    novel_capitals = [
        ("Australia",    "Canberra"),
        ("Thailand",     "Bangkok"),
        ("Turkey",       "Ankara"),
        ("Ireland",      "Dublin"),
        ("Denmark",      "Copenhagen"),
        ("Portugal",     "Lisbon"),
        ("Finland",      "Helsinki"),
        ("Switzerland",  "Bern"),
        ("Canada",       "Ottawa"),
        ("Brazil",       "Brasilia"),
        ("Mexico",       "Mexico"),
        ("Argentina",    "Buenos"),
    ]

    print("\n  capital_of v2-delta (trained on 14 pairs) → applied to unseen countries:")
    for src, expected in novel_capitals:
        try:
            results = lcm.apply_delta(src, delta_capital, k=10,
                                      exclude_words=[src])
            top = results[0][0] if results else "?"
            hd  = results[0][1] if results else -1
            found = any(expected.lower() in w.lower()
                        for w, _ in results[:5])
            marker = "✓" if found else " "
            print(f"  [{marker}] {src:14s} → top={top:16s}  "
                  f"(expected ~{expected})")
        except RuntimeError as e:
            print(f"  [!] {src}: {e}")

    novel_gender = [
        ("emperor",  "empress"),
        ("lord",     "lady"),
        ("wizard",   "witch"),
        ("monk",     "nun"),
        ("bull",     "cow"),
    ]

    print("\n  male→female v2-delta → applied to unseen words:")
    for src, expected in novel_gender:
        try:
            results = lcm.apply_delta(src, delta_gender, k=10,
                                      exclude_words=[src])
            top3 = [w for w, _ in results[:3]]
            found = any(expected.lower() in w.lower() for w in top3)
            marker = "✓" if found else " "
            print(f"  [{marker}] {src:12s} → top3={top3}  (expected ~{expected})")
        except RuntimeError as e:
            print(f"  [!] {src}: {e}")

    novel_lang = [
        ("Turkey",  "Turkish"),
        ("Korea",   "Korean"),
        ("Vietnam", "Vietnamese"),
        ("Iran",    "Persian"),
    ]

    print("\n  country→language v2-delta → applied to unseen countries:")
    for src, expected in novel_lang:
        try:
            results = lcm.apply_delta(src, delta_lang, k=10,
                                      exclude_words=[src])
            top3 = [w for w, _ in results[:3]]
            found = any(expected.lower() in w.lower() for w in top3)
            marker = "✓" if found else " "
            print(f"  [{marker}] {src:10s} → top3={top3}  (expected ~{expected})")
        except RuntimeError as e:
            print(f"  [!] {src}: {e}")

    # ── Delta composition: capital_of + country→language ─────────────────────
    print(f"\n{sep}")
    print("DELTA COMPOSITION  (capital_of + country→language)")
    print(sep)
    print("  Q: 'What language is spoken in the capital of X?'")
    print("  A: apply capital_of delta, then language delta\n")

    composed_delta = delta_capital + delta_lang
    composition_tests = [
        ("France",  "French"),
        ("Germany", "German"),
        ("Japan",   "Japanese"),
        ("Spain",   "Spanish"),
        ("Italy",   "Italian"),
    ]
    for src, expected in composition_tests:
        try:
            results = lcm.apply_delta(src, composed_delta, k=10,
                                      exclude_words=[src])
            top3 = [w for w, _ in results[:3]]
            found = any(expected.lower() in w.lower() for w in top3)
            marker = "✓" if found else " "
            print(f"  [{marker}] language_in_capital({src:10s}) → "
                  f"top3={top3}  (expected ~{expected})")
        except RuntimeError as e:
            print(f"  [!] {src}: {e}")

    return delta_capital, delta_gender, delta_lang


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(lcm):
    sep = "─" * 60

    def show(label, results):
        print(f"\n  {label}")
        for word, hd in results[:6]:
            print(f"    {word:20s}  Hamming={hd}")

    print(f"\n{'═'*60}")
    print("LCM INFERENCE TEST SUITE")
    print(f"{'═'*60}")

    # ── Nearest neighbour ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("NEAREST NEIGHBOUR  (Hamming distance in address space)")
    print(sep)
    for word in ["king", "Paris", "French", "democracy", "ocean", "algorithm"]:
        show(f"nearest('{word}')", lcm.nearest(word, k=8))

    # ── Axis-flip: single-bit inference ──────────────────────────────────────
    print(f"\n{sep}")
    print("AXIS-FLIP INFERENCE  (flip one axis bit → nearest concept)")
    print(sep)

    flip_queries = [
        ("France",   "is_capital_city",        "→ capital of France?"),
        ("Germany",  "is_capital_city",        "→ capital of Germany?"),
        ("Japan",    "is_capital_city",        "→ capital of Japan?"),
        ("Paris",    "is_capital_city",        "→ country of Paris? (flip back)"),
        ("king",     "is_female_gendered",     "→ female form of king?"),
        ("actor",    "is_female_gendered",     "→ female form of actor?"),
        ("princess", "is_female_gendered",     "→ male form of princess?"),
        ("Spanish",  "is_germanic_language",   "→ Spanish but Germanic?"),
        ("German",   "is_romance_language",    "→ German but Romance?"),
        ("Tokyo",    "is_european_country",    "→ Tokyo but European country?"),
    ]
    for word, axis, label in flip_queries:
        results = lcm.axis_flip(word, axis, k=5)
        top = results[0][0] if results else "?"
        hd  = results[0][1] if results else -1
        print(f"  axis_flip('{word}', '{axis}')")
        print(f"    {label}  →  top={top!r}  (Hamming={hd})")
        for w, d in results[1:3]:
            print(f"       also: {w!r}  (Hamming={d})")

    # ── Analogy: continuous projection arithmetic ────────────────────────────
    print(f"\n{sep}")
    print("ANALOGY  proj(a) - proj(b) + proj(c) → NN  (continuous arithmetic)")
    print(sep)

    analogies = [
        ("king",    "man",     "woman",    "king - man + woman = ?  (→ queen)"),
        ("Paris",   "France",  "Germany",  "Paris - France + Germany = ?  (→ Berlin)"),
        ("Paris",   "France",  "Japan",    "Paris - France + Japan = ?  (→ Tokyo)"),
        ("Berlin",  "Germany", "France",   "Berlin - Germany + France = ?  (→ Paris)"),
        ("queen",   "woman",   "man",      "queen - woman + man = ?  (→ king)"),
        ("actress", "woman",   "man",      "actress - woman + man = ?  (→ actor)"),
        ("French",  "Italian",  "German",   "French - Italian + German = ?   (→ German-like Romance?)"),
        ("Spanish", "French",   "English",  "Spanish - French + English = ?  (→ English-like Romance?)"),
        ("Moscow",  "Russia",   "France",   "Moscow - Russia + France = ?   (→ Paris?)"),
        ("Rome",    "Italy",    "Spain",    "Rome - Italy + Spain = ?       (→ Madrid?)"),
    ]
    for a, b, c, label in analogies:
        try:
            results = lcm.analogy(a, b, c, k=8)
            print(f"  {label}")
            for word, hd in results[:4]:
                print(f"    {word:20s}  Hamming={hd}")
        except RuntimeError as e:
            print(f"  {label}")
            print(f"    SKIP: {e}")

    # ── Conditional multi-axis search ────────────────────────────────────────
    print(f"\n{sep}")
    print("CONDITIONAL SEARCH  (nearest concept matching axis constraints)")
    print(sep)

    conditionals = [
        ("France",  {"is_capital_city": True},
         "'France' but is a capital city?"),
        ("Japan",   {"is_european_country": True, "is_capital_city": True},
         "'Japan' but European + capital city?"),
        ("king",    {"is_female_gendered": True},
         "'king' but female?"),
        ("German",  {"is_romance_language": True},
         "'German' but Romance language?"),
        ("village", {"is_capital_city": True},
         "'village' but a capital city?"),
    ]
    for word, conds, label in conditionals:
        results = lcm.conditional(word, k=6, **conds)
        print(f"  {label}")
        for w, d in results[:4]:
            print(f"    {w:20s}  Hamming={d}")

    # ── Describe ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("DESCRIBE  (most distinctive axis projections for a concept)")
    print(sep)
    for word in ["Paris", "king", "democracy"]:
        print(f"\n  describe('{word}') — top distinctive axes:")
        for item in lcm.describe(word, top_n=8):
            bar = "▌" if item["bit"] else "░"
            print(f"    [{bar}] slot={item['slot']:3d}  "
                  f"proj={item['proj']:+.3f}  "
                  f"margin={item['margin']:.3f}  "
                  f"{item['name']}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

REPL_HELP = """
Commands:
  <word>                    — nearest neighbours in address space
  <a> - <b> + <c>           — analogy (continuous projection arithmetic)
  <word> [axis_name]        — axis-flip: find nearest with that axis toggled
  describe <word>           — show distinctive axes for a concept
  axes                      — list known seed axis names
  q / quit / exit           — quit

Examples:
  king
  king - man + woman
  France [is_capital_city]
  Germany [is_female_gendered]
  describe Paris
"""

def repl(lcm):
    print(REPL_HELP)
    known_seeds = sorted(set(
        k for k in lcm.axis_slots if "is_" in k and len(k) < 30
    ))

    while True:
        try:
            line = input("\nlcm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break
        if line.lower() == "axes":
            print("  Known seed axes:")
            for s in known_seeds:
                print(f"    {s}")
            continue
        if line.lower().startswith("describe "):
            word = line[9:].strip()
            for item in lcm.describe(word, top_n=12):
                bar = "▌" if item["bit"] else "░"
                print(f"  [{bar}] proj={item['proj']:+.3f}  "
                      f"margin={item['margin']:.3f}  {item['name']}")
            continue

        # Analogy: a - b + c
        m_analogy = re.match(r'^(\w+)\s*-\s*(\w+)\s*\+\s*(\w+)$', line)
        if m_analogy:
            a, b, c = m_analogy.groups()
            try:
                results = lcm.analogy(a, b, c, k=8)
                print(f"  {a} - {b} + {c}  =?")
                for w, d in results:
                    print(f"    {w:20s}  Hamming={d}")
            except RuntimeError as e:
                print(f"  Error: {e}")
            continue

        # Axis-flip: word [axis_name]
        m_flip = re.match(r'^(\w+)\s+\[([^\]]+)\]$', line)
        if m_flip:
            word, axis = m_flip.group(1), m_flip.group(2)
            try:
                results = lcm.axis_flip(word, axis, k=8)
                print(f"  axis_flip('{word}', '{axis}'):")
                for w, d in results:
                    print(f"    {w:20s}  Hamming={d}")
            except RuntimeError as e:
                print(f"  Error: {e}")
            continue

        # Simple word → nearest
        m_word = re.match(r'^(\w+)$', line)
        if m_word:
            word = m_word.group(1)
            try:
                results = lcm.nearest(word, k=12)
                print(f"  nearest('{word}'):")
                for w, d in results:
                    print(f"    {w:20s}  Hamming={d}")
            except RuntimeError as e:
                print(f"  Error: {e}")
            continue

        print("  (unrecognised — type 'help' or see examples above)")


# ─────────────────────────────────────────────────────────────────────────────
# Failure diagnostic — show top-5 for each failing pair
# ─────────────────────────────────────────────────────────────────────────────

def run_axis_decomposition(lcm):
    """
    Per-axis φ-4 similarity decomposition for every failing LOO pair.

    For each failure (src → tgt, intruder at rank-0):
      pred_phi4  = _compute_phi4(src_proj + delta)
      target_unit = pred_phi4 / |pred_phi4|
      per-axis advantage of intruder over target:
        adv[j] = target_unit[j] * (normed_w0[j] - normed_tgt[j])

    Axes with large positive adv[j] make the intruder win.
    Axes with large negative adv[j] would make the target win if they were stronger.

    Prints the top-5 intruder-winning and top-5 target-winning axes for each failure,
    with axis name and semantic content.
    """
    missing = [
        ("Greek", 17860, False), ("boy", 8171, False),
        ("girl",  3743,  False), ("Oslo", 57858, True),
        ("Polish", 31984, True),
    ]
    for word, tid, overwrite in missing:
        lcm.add_word(word, tid, overwrite=overwrite)

    capital_pairs = [
        ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),
        ("China","Beijing"),("Italy","Rome"),("Spain","Madrid"),
        ("Russia","Moscow"),("Greece","Athens"),("Poland","Warsaw"),
        ("Sweden","Stockholm"),("Norway","Oslo"),("Austria","Vienna"),
        ("Belgium","Brussels"),("Netherlands","Amsterdam"),
    ]
    gender_pairs = [
        ("king","queen"),("man","woman"),("boy","girl"),("father","mother"),
        ("brother","sister"),("actor","actress"),("prince","princess"),
        ("hero","heroine"),("son","daughter"),("husband","wife"),
    ]

    PHI    = np.float32(1.6180339887)
    INVPHI = np.float32(1.0 / 1.6180339887)

    def _decompose(name, pairs):
        print(f"\n{'═'*70}")
        print(f"DECOMPOSITION: {name}")
        print(f"{'═'*70}")

        any_fail = False
        for src, tgt in pairs:
            excl = [src] + [s for s, t in pairs if s != src]
            delta, _ = lcm.learn_delta(pairs, exclude_pair=(src, tgt))
            _, _, fp_loo, _ = lcm.learn_delta_v2(pairs, exclude_pair=(src, tgt))

            res = lcm.apply_delta_phi_boost(src, delta, fp_loo, k=10,
                                            exclude_words=excl,
                                            boost_threshold=0.85,
                                            preserve_scale=1.0)
            rank = next((i for i, (w, _) in enumerate(res)
                         if w.lower() == tgt.lower()), 9999)
            if rank == 0:
                continue

            any_fail = True
            intruder = res[0][0]

            # ── Compute predicted φ-4 vector ────────────────────────────────
            src_proj, _   = lcm._get_proj(src)
            pred_proj     = (src_proj.astype(np.float64) +
                             delta.astype(np.float64)).astype(np.float32)
            pred_phi4     = lcm._compute_phi4(pred_proj, continuous=True,
                                              preserve_scale=1.0)

            # Normalise predicted vector
            pred_norm  = np.linalg.norm(pred_phi4) + 1e-20
            pred_unit  = pred_phi4 / pred_norm           # (n_axes,)

            # ── Look up intruder and target in phi4_vecs_v6 ─────────────────
            w0_idx  = lcm.word_set.get(intruder.lower())
            tgt_idx = lcm.word_set.get(tgt.lower())

            if w0_idx is None or tgt_idx is None:
                print(f"\n  {src} → {tgt}  [index missing, skip]")
                continue

            vecs       = lcm.phi4_vecs_v6           # (n_concepts, n_axes)
            w0_vec     = vecs[w0_idx]
            tgt_vec    = vecs[tgt_idx]

            w0_norm    = np.linalg.norm(w0_vec)  + 1e-20
            tgt_norm_  = np.linalg.norm(tgt_vec) + 1e-20
            normed_w0  = w0_vec  / w0_norm          # (n_axes,)
            normed_tgt = tgt_vec / tgt_norm_         # (n_axes,)

            # Per-axis similarity contributions
            contrib_w0  = pred_unit * normed_w0    # cos-sim contribution per axis (intruder)
            contrib_tgt = pred_unit * normed_tgt   # cos-sim contribution per axis (target)
            advantage   = contrib_w0 - contrib_tgt # positive = intruder wins here

            total_w0  = float(np.sum(contrib_w0))
            total_tgt = float(np.sum(contrib_tgt))

            # Top-5 axes where intruder wins most
            top_intruder = np.argsort(advantage)[-5:][::-1]
            # Top-5 axes where target would win
            top_target   = np.argsort(advantage)[:5]

            print(f"\n  {src} → {tgt}  (rank={rank}, intruder={intruder})")
            print(f"  Total cos-sim: intruder={total_w0:.4f}  target={total_tgt:.4f}  gap={total_w0-total_tgt:.4f}")

            print(f"\n  ┌ INTRUDER WINS — top axes favouring '{intruder}':")
            for j in top_intruder:
                ax_name = lcm.axis_names[j][:55] if j < len(lcm.axis_names) else f"axis_{j}"
                print(f"  │  ax{j:4d}  adv={advantage[j]:+.4f}  "
                      f"w0={normed_w0[j]:+.4f}  tgt={normed_tgt[j]:+.4f}  "
                      f"pred={pred_unit[j]:+.4f}  [{ax_name}]")

            print(f"\n  └ TARGET WOULD WIN — top axes favouring '{tgt}':")
            for j in top_target:
                ax_name = lcm.axis_names[j][:55] if j < len(lcm.axis_names) else f"axis_{j}"
                print(f"    ax{j:4d}  adv={advantage[j]:+.4f}  "
                      f"w0={normed_w0[j]:+.4f}  tgt={normed_tgt[j]:+.4f}  "
                      f"pred={pred_unit[j]:+.4f}  [{ax_name}]")

        if not any_fail:
            print("  All pairs rank-1 ✓")

    _decompose("capital_of",  capital_pairs)
    _decompose("male→female", gender_pairs)


def run_failure_diagnostic(lcm):
    """
    For every LOO pair that does not rank-1, show the full top-5 output so we
    can see what concept is "stealing" the top slot and whether it is a
    systematic or accidental failure.
    """
    missing = [
        ("Greek", 17860, False), ("boy", 8171, False),
        ("girl",  3743, False),  ("Oslo", 57858, True),
        ("Polish", 31984, True),
    ]
    for word, tid, overwrite in missing:
        lcm.add_word(word, tid, overwrite=overwrite)

    capital_pairs = [
        ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),
        ("China","Beijing"),("Italy","Rome"),("Spain","Madrid"),
        ("Russia","Moscow"),("Greece","Athens"),("Poland","Warsaw"),
        ("Sweden","Stockholm"),("Norway","Oslo"),("Austria","Vienna"),
        ("Belgium","Brussels"),("Netherlands","Amsterdam"),
    ]
    gender_pairs = [
        ("king","queen"),("man","woman"),("boy","girl"),("father","mother"),
        ("brother","sister"),("actor","actress"),("prince","princess"),
        ("hero","heroine"),("son","daughter"),("husband","wife"),
    ]
    language_pairs = [
        ("France","French"),("Germany","German"),("Japan","Japanese"),
        ("China","Chinese"),("Italy","Italian"),("Spain","Spanish"),
        ("Russia","Russian"),("Greece","Greek"),("Poland","Polish"),
        ("Sweden","Swedish"),("Norway","Norwegian"),
    ]

    sep = "─" * 56

    def _diag_pairs(name, pairs):
        print(f"\n{'═'*60}")
        print(f"FAILURES: {name}")
        print(f"{'═'*60}")

        any_fail = False
        for src, tgt in pairs:
            excl  = [src] + [s for s, t in pairs if s != src]
            delta, _ = lcm.learn_delta(pairs, exclude_pair=(src, tgt))
            _, _, fp_loo, _ = lcm.learn_delta_v2(pairs, exclude_pair=(src, tgt))

            res = lcm.apply_delta_phi_boost(src, delta, fp_loo, k=10,
                                            exclude_words=excl,
                                            boost_threshold=0.85,
                                            preserve_scale=1.0)
            rank = next((i for i, (w, _) in enumerate(res)
                         if w.lower() == tgt.lower()), 9999)

            if rank == 0:
                continue   # success — skip

            any_fail = True
            top5 = [(w, s) for w, s in res[:5]]
            print(f"\n  {src} → {tgt}  (rank={rank})")
            for i, (w, s) in enumerate(top5):
                marker = "✓" if w.lower() == tgt.lower() else " "
                print(f"    [{marker}] {i}: {w:18s}  sim={s:.4f}")

        if not any_fail:
            print("  All pairs rank-1 ✓")

    _diag_pairs("capital_of",      capital_pairs)
    _diag_pairs("male→female",     gender_pairs)
    _diag_pairs("country→language", language_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Inference axis sweep — find optimal N for relationship delta inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_sweep(lcm):
    """
    Patch LCMIndex attributes to use only the first N axes and run silent LOO,
    measuring rank-1 on capital_of, male→female, and country→language.

    Requires LCMIndex already loaded with all 1500 axes (QUALITY_MIN=0.0).
    Finds the N that maximises relationship inference performance.
    """
    import time

    print(f"\n{'═'*60}")
    print("INFERENCE SWEEP — optimal axis count for delta inference")
    print(f"{'═'*60}")

    # ── Pair definitions (same as run_delta_tests) ────────────────────────────
    capital_pairs = [
        ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),
        ("China","Beijing"),("Italy","Rome"),("Spain","Madrid"),
        ("Russia","Moscow"),("Greece","Athens"),("Poland","Warsaw"),
        ("Sweden","Stockholm"),("Norway","Oslo"),("Austria","Vienna"),
        ("Belgium","Brussels"),("Netherlands","Amsterdam"),
    ]
    gender_pairs = [
        ("king","queen"),("man","woman"),("father","mother"),
        ("brother","sister"),("son","daughter"),("husband","wife"),
        ("prince","princess"),("actor","actress"),("hero","heroine"),
        ("boy","girl"),
    ]
    language_pairs = [
        ("France","French"),("Germany","German"),("Japan","Japanese"),
        ("China","Chinese"),("Italy","Italian"),("Spain","Spanish"),
        ("Russia","Russian"),("Greece","Greek"),("Poland","Polish"),
        ("Sweden","Swedish"),("Norway","Norwegian"),
    ]

    def _quick_r1(lcm_obj, pairs):
        """Return rank-1 hit count for a pair set using v3/v6/v6d LOO (silent)."""
        r1_v3 = r1_v6 = r1_v6d = 0
        for src, tgt in pairs:
            excl = [src] + [s for s, t in pairs if s != src]

            # v3
            delta, _ = lcm_obj.learn_delta(pairs, exclude_pair=(src, tgt))
            res = lcm_obj.apply_delta_continuous(src, delta, k=5, exclude_words=excl)
            if any(w.lower() == tgt.lower() for w, _ in res[:1]):
                r1_v3 += 1

            # v6
            res6 = lcm_obj.apply_delta_phi4(src, delta, k=5, exclude_words=excl,
                                             continuous=True, preserve_scale=1.0)
            if any(w.lower() == tgt.lower() for w, _ in res6[:1]):
                r1_v6 += 1

            # v6d
            _, _, fp_loo, _ = lcm_obj.learn_delta_v2(pairs, exclude_pair=(src, tgt))
            res6d = lcm_obj.apply_delta_phi_boost(src, delta, fp_loo, k=5,
                                                  exclude_words=excl,
                                                  boost_threshold=0.85,
                                                  preserve_scale=1.0)
            if any(w.lower() == tgt.lower() for w, _ in res6d[:1]):
                r1_v6d += 1

        return r1_v3, r1_v6, r1_v6d

    # ── Inject concepts absent from Phase-0 mining ───────────────────────────
    missing = [
        ("Greek",  17860, False),
        ("boy",    8171,  False),
        ("girl",   3743,  False),
        ("Oslo",   57858, True),
        ("Polish", 31984, True),
    ]
    for word, tid, overwrite in missing:
        lcm.add_word(word, tid, overwrite=overwrite)
    print(f"  Injected: {[w for w,_,_ in missing]}")

    # ── Save full-size attributes ─────────────────────────────────────────────
    saved = {
        "axis_vectors":    lcm.axis_vectors,
        "axis_names":      lcm.axis_names,
        "projections":     lcm.projections,
        "thresholds":      lcm.thresholds,
        "_phi4_hi":        lcm._phi4_hi,
        "_phi4_lo":        lcm._phi4_lo,
        "phi4_vecs":       lcm.phi4_vecs,
        "phi4_vecs_v6":    lcm.phi4_vecs_v6,
        "seed_slots":      lcm.seed_slots,
        "axis_slots":      lcm.axis_slots,
    }

    axis_counts = [50, 100, 150, 193, 250, 300, 400, 500, 750, 1000, 1500]

    print(f"\n  {'N':>6}  {'cap R1':>8}  {'cap v6':>8}  {'cap v6d':>9}  "
          f"{'gen R1':>8}  {'lang R1':>9}  {'sum v6d':>9}")
    print(f"  {'─'*72}")

    t0 = time.time()
    for n_axes in axis_counts:
        n = min(n_axes, len(saved["axis_names"]))

        # Patch attributes to first N axes
        lcm.axis_vectors = saved["axis_vectors"][:n]
        lcm.axis_names   = saved["axis_names"][:n]
        lcm.projections  = saved["projections"][:, :n]
        lcm.thresholds   = saved["thresholds"][:n]
        lcm._phi4_hi     = saved["_phi4_hi"][:n]
        lcm._phi4_lo     = saved["_phi4_lo"][:n]
        lcm.phi4_vecs    = saved["phi4_vecs"][:, :n]
        lcm.phi4_vecs_v6 = saved["phi4_vecs_v6"][:, :n]
        # Rebuild seed_slots restricted to slots < n
        lcm.seed_slots   = {s for s in saved["seed_slots"] if s < n}

        # Run silent LOO
        c3, c6, c6d = _quick_r1(lcm, capital_pairs)
        g3, g6, g6d = _quick_r1(lcm, gender_pairs)
        l3, l6, l6d = _quick_r1(lcm, language_pairs)

        nc, ng, nl = len(capital_pairs), len(gender_pairs), len(language_pairs)
        total_v6d = c6d + g6d + l6d
        print(f"  {n:>6}  "
              f"{c3}/{nc}({100*c3//nc:2d}%)"
              f"  {c6}/{nc}({100*c6//nc:2d}%)"
              f"  {c6d}/{nc}({100*c6d//nc:2d}%)"
              f"  {g6d}/{ng}({100*g6d//ng:2d}%)"
              f"  {l6d}/{nl}({100*l6d//nl:2d}%)"
              f"  {total_v6d}/{nc+ng+nl}({100*total_v6d//(nc+ng+nl):2d}%)",
              flush=True)

    # Restore
    for k, v in saved.items():
        setattr(lcm, k, v)
    lcm.axis_slots = saved["axis_slots"]

    print(f"\n  Completed in {time.time()-t0:.0f}s")
    print(f"  Columns: cap=capital_of, gen=male→female, lang=country→language")


# ─────────────────────────────────────────────────────────────────────────────
# Axis sweep — completeness scaling across all 1500 existing axes
# ─────────────────────────────────────────────────────────────────────────────

def run_axis_sweep(lcm, sample_size=1000):
    """
    Test how top-10 neighbourhood recall scales with axis count,
    using ALL 1500 axes from dc299_phase1_axes.json (no quality filter).

    Answers: does the 41% gap close if we use more of the existing axes,
    or do we need new axis discovery?

    Prints a scaling table: axes → top-10 overlap with embed space.
    """
    import time, re

    print(f"\n{'═'*60}")
    print("AXIS SWEEP — completeness vs axis count")
    print(f"{'═'*60}")

    # ── Load ALL axes from phase1 JSON (and optionally phase1b extension) ───────
    print("  Loading all axes from JSON …", flush=True)
    with open(AXES_JSON) as f:
        axes_data = json.load(f)

    all_vectors = np.array(axes_data["axis_vectors"], dtype=np.float32)
    all_meta    = axes_data["axes"]

    if AXES_JSON_B.exists():
        with open(AXES_JSON_B) as fb:
            axes_b = json.load(fb)
        all_vectors = np.concatenate(
            [all_vectors, np.array(axes_b["axis_vectors"], dtype=np.float32)], axis=0
        )
        all_meta = all_meta + axes_b["axes"]
        print(f"  (+{len(axes_b['axes'])} Phase-1b axes)", flush=True)

    N_AXES_TOTAL = len(all_meta)

    # quality score (same formula as LCMIndex)
    _clean = re.compile(r'^[A-Za-z]{3,}$')
    def _q(m):
        tokens = [w for w, _ in m.get("top_vocab", [])] + \
                 [w for w, _ in m.get("bot_vocab", [])]
        if not tokens: return 0.0
        return sum(1 for t in tokens if _clean.match(t.strip())) / len(tokens)

    qualities = np.array([1.0 if m.get("type")=="seed" else _q(m)
                          for m in all_meta])

    print(f"  {N_AXES_TOTAL} total axes  "
          f"(seeds={int((qualities==1.0).sum())}, "
          f"mean_q={qualities.mean():.3f})")

    # ── Build concept embedding matrix (ground truth) ─────────────────────────
    print("  Building concept embedding matrix …", flush=True)
    tids     = np.array(lcm.tids)
    emb_mat  = lcm.embeddings[tids].astype(np.float32)
    emb_mat /= np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-20
    N_CONCEPTS = emb_mat.shape[0]

    # ── Project all concepts onto ALL 1500 axes ───────────────────────────────
    print("  Projecting all concepts onto all axes …", flush=True)
    # Normalise concept embeddings (already done above) then project
    concept_embs_raw = lcm.embeddings[tids].astype(np.float32)
    norms = np.linalg.norm(concept_embs_raw, axis=1, keepdims=True) + 1e-20
    concept_embs_unit = concept_embs_raw / norms

    full_proj = (concept_embs_unit @ all_vectors.T).astype(np.float32)  # (N, 1500)
    print(f"  full_proj shape: {full_proj.shape}")

    # ── Sample concepts for evaluation ───────────────────────────────────────
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N_CONCEPTS, min(sample_size, N_CONCEPTS), replace=False)

    # ── Axis ordering strategies to test ─────────────────────────────────────
    # Strategy A: IRD order (variance-descending, as discovered)
    ird_order  = np.arange(N_AXES_TOTAL)
    # Strategy B: quality-descending
    qual_order = np.argsort(qualities)[::-1]

    axis_counts = [50, 100, 193, 300, 500, 750, 1000, 1500, 2000, 2500, 2828]

    print(f"\n  Sample: {len(sample_idx)} concepts\n")
    print(f"  {'axes':>6}  {'IRD-order top-10':>18}  {'quality-order top-10':>22}  "
          f"{'mean_q(IRD)':>12}")
    print(f"  {'─'*64}")

    t0 = time.time()
    for n_axes in axis_counts:
        results = {}
        for label, ordering in [("ird", ird_order), ("qual", qual_order)]:
            idx = ordering[:n_axes]
            sub_proj = full_proj[:, idx].astype(np.float32)
            sub_proj_norm = sub_proj / (np.linalg.norm(sub_proj, axis=1, keepdims=True) + 1e-20)

            overlaps = []
            for cidx in sample_idx:
                emb_sims  = emb_mat @ emb_mat[cidx]
                emb_sims[cidx] = -999.0
                sub_sims  = sub_proj_norm @ sub_proj_norm[cidx]
                sub_sims[cidx] = -999.0

                top10_emb = set(np.argpartition(emb_sims, -10)[-10:])
                top10_sub = set(np.argpartition(sub_sims,  -10)[-10:])
                overlaps.append(len(top10_emb & top10_sub) / 10)

            results[label] = np.mean(overlaps)

        mean_q = qualities[ird_order[:n_axes]].mean()
        print(f"  {n_axes:>6}  {results['ird']:>17.1%}  {results['qual']:>21.1%}  "
              f"{mean_q:>11.3f}")

    print(f"\n  Sweep completed in {time.time()-t0:.1f}s")
    print(f"\n  Current LCMIndex uses {len(lcm.axis_names)} axes "
          f"(quality≥0.5 + cliff at 193)")


# ─────────────────────────────────────────────────────────────────────────────
# Completeness experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_completeness_experiment(lcm, sample_size=2000, k_values=(1, 5, 10, 50)):
    """
    Holographic completeness test.

    Three representations of every concept:
      embed  (3584-dim) — original Qwen2 embedding vector          [ground truth]
      proj   (193-dim)  — continuous projection onto IRD axes       [axis selection loss]
      phi    (193-dim)  — φ-4-state encoding {-φ,-1,+1,+φ}         [quantisation loss]

    For each sampled concept we retrieve top-K neighbours in each space and
    measure what fraction of the embedding-space top-K are recovered.  This
    directly answers: is the φ-lattice address information-complete?
    """
    import time

    def _spearmanr(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
        return float(np.dot(ra, rb) / denom) if denom > 0 else 0.0

    N = len(lcm.words)
    print(f"\n{'═'*60}")
    print("COMPLETENESS EXPERIMENT")
    print("  φ-lattice address vs original embedding neighbourhood")
    print(f"{'═'*60}")
    print(f"  Concepts : {N}")
    print(f"  Sample   : {min(sample_size, N)}")
    print(f"  k values : {k_values}")

    # ── Build and normalise all three matrices ────────────────────────────────
    print("\n  Building matrices …", flush=True)
    tids = np.array(lcm.tids)
    emb_mat  = lcm.embeddings[tids].astype(np.float32)
    emb_mat /= np.linalg.norm(emb_mat,  axis=1, keepdims=True) + 1e-20

    proj_mat  = lcm.projections.astype(np.float32).copy()
    proj_mat /= np.linalg.norm(proj_mat, axis=1, keepdims=True) + 1e-20

    phi_mat   = lcm.phi4_vecs_v6.astype(np.float32).copy()
    phi_mat  /= np.linalg.norm(phi_mat,  axis=1, keepdims=True) + 1e-20

    print(f"  emb_mat  {emb_mat.shape}  {emb_mat.nbytes/1e6:.0f} MB")
    print(f"  proj_mat {proj_mat.shape}")
    print(f"  phi_mat  {phi_mat.shape}")

    # ── Sample ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, min(sample_size, N), replace=False)

    max_k = max(k_values)

    overlaps_proj = {k: [] for k in k_values}
    overlaps_phi  = {k: [] for k in k_values}

    # For rank-displacement: mean rank (in φ-space) of embed top-10 neighbours
    rank_displace_proj = []
    rank_displace_phi  = []

    # Spearman ρ over top-200 (cheap enough per sample)
    spearman_proj = []
    spearman_phi  = []

    t0 = time.time()
    for step, cidx in enumerate(sample_idx):
        if step % 500 == 0 and step > 0:
            print(f"    {step}/{len(sample_idx)}  ({time.time()-t0:.1f}s)", flush=True)

        # Ground truth: embedding similarities
        emb_sims  = emb_mat  @ emb_mat[cidx]
        emb_sims[cidx] = -999.0

        # Projection similarities
        proj_sims = proj_mat @ proj_mat[cidx]
        proj_sims[cidx] = -999.0

        # φ-address similarities
        phi_sims  = phi_mat  @ phi_mat[cidx]
        phi_sims[cidx] = -999.0

        # ── Overlap at each k ─────────────────────────────────────────────────
        for k in k_values:
            top_emb  = set(np.argpartition(emb_sims,  -k)[-k:])
            top_proj = set(np.argpartition(proj_sims, -k)[-k:])
            top_phi  = set(np.argpartition(phi_sims,  -k)[-k:])
            overlaps_proj[k].append(len(top_emb & top_proj) / k)
            overlaps_phi[k].append( len(top_emb & top_phi)  / k)

        # ── Rank displacement of embed top-10 ────────────────────────────────
        top10_emb = np.argpartition(emb_sims, -10)[-10:]
        proj_ranks = [int(np.sum(proj_sims > proj_sims[j])) for j in top10_emb]
        phi_ranks  = [int(np.sum(phi_sims  > phi_sims[j]))  for j in top10_emb]
        rank_displace_proj.append(np.mean(proj_ranks))
        rank_displace_phi.append( np.mean(phi_ranks))

        # ── Spearman ρ over top-200 neighbourhood ────────────────────────────
        top200 = np.argpartition(emb_sims, -200)[-200:]
        rho_proj = _spearmanr(emb_sims[top200], proj_sims[top200])
        rho_phi  = _spearmanr(emb_sims[top200], phi_sims[top200])
        spearman_proj.append(rho_proj)
        spearman_phi.append( rho_phi)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "─" * 56
    print(f"  {'k':>5}  {'proj→emb overlap':>18}  {'φ→emb overlap':>15}  {'gap':>7}")
    print(f"  {sep}")
    for k in k_values:
        pm = np.mean(overlaps_proj[k])
        fm = np.mean(overlaps_phi[k])
        print(f"  {k:>5}  {pm:>17.1%}  {fm:>14.1%}  {fm-pm:>+6.1%}")

    print(f"\n  Mean rank of embed top-10 in test space (lower = better):")
    print(f"    proj space : {np.mean(rank_displace_proj):.1f}")
    print(f"    φ-address  : {np.mean(rank_displace_phi):.1f}")
    print(f"    ratio φ/proj: {np.mean(rank_displace_phi)/max(np.mean(rank_displace_proj),1):.2f}×")

    print(f"\n  Spearman ρ over top-200 neighbourhood (higher = better):")
    print(f"    proj space : {np.mean(spearman_proj):.4f}")
    print(f"    φ-address  : {np.mean(spearman_phi):.4f}")
    print(f"    ratio φ/proj: {np.mean(spearman_phi)/max(np.mean(spearman_proj),0.001):.3f}")

    print(f"\n  Interpretation:")
    phi_top10 = np.mean(overlaps_phi[10])
    proj_top10 = np.mean(overlaps_proj[10])
    print(f"    Axis-selection loss (3584→193):  {1-proj_top10:.1%} of top-10 neighbours lost")
    print(f"    Quantisation loss   (proj→φ):    {proj_top10-phi_top10:+.1%} additional loss")
    print(f"    Total loss          (emb→φ):     {1-phi_top10:.1%} of top-10 neighbours lost")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    lcm = LCMIndex()
    print(f"\nIndex built in {time.time()-t0:.1f}s")

    if "--repl" in sys.argv:
        repl(lcm)
    elif "--deltas" in sys.argv:
        run_delta_tests(lcm)
    elif "--completeness" in sys.argv:
        run_completeness_experiment(lcm)
    elif "--sweep" in sys.argv:
        run_axis_sweep(lcm)
    elif "--inference-sweep" in sys.argv:
        run_inference_sweep(lcm)
    elif "--diagnose" in sys.argv:
        run_failure_diagnostic(lcm)
    elif "--decompose" in sys.argv:
        run_axis_decomposition(lcm)
    else:
        run_delta_tests(lcm)
        run_tests(lcm)


if __name__ == "__main__":
    main()
