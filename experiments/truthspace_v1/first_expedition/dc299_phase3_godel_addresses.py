#!/usr/bin/env python3
"""
DC299 Phase 3 — Gödel Address Assignment
=========================================

For each of the 25,671 Phase-0 concepts, compute a binary "Gödel address":
    bit_k = int( normalize(e) · axis_k  >  threshold_k )

where:
  - axis_k  = k-th semantic axis from Phase 1 (quality ≥ QUALITY_MIN)
  - threshold_k = median projection of all concept embeddings onto axis_k

Then run relationship delta tests:
  1. Gender analogy   : king - man + woman ≈ queen in address space
  2. Capital analogy  : France→Paris delta ≈ Germany→Berlin delta ≈ …
  3. Address decode   : reconstruct embedding from address, find nearest vocab token

Fail-fast: no graceful fallbacks.

Outputs:
  dc299_phase3_godel_addresses.json   — binary address per concept
  dc299_phase3_notes.md               — findings and test results
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
PHASE0_JSON  = SCRIPT_DIR / "dc299_phase0_concepts.json"
OUTPUT_JSON  = SCRIPT_DIR / "dc299_phase3_godel_addresses.json"
NOTES_PATH   = SCRIPT_DIR / "dc299_phase3_notes.md"

# ── Config ────────────────────────────────────────────────────────────────────
QUALITY_MIN   = 0.50   # min fraction of clean English tokens in top+bot vocab
QUALITY_WINDOW = 20    # rolling window for quality cliff detection
MAX_SEMANTIC  = 300    # hard cap on semantic axes to use (safety)
ANALOGY_K     = 10     # nearest neighbours to report in analogy tests

assert AXES_JSON.exists(),   f"FAIL-FAST: {AXES_JSON} missing"
assert PHASE0_JSON.exists(), f"FAIL-FAST: {PHASE0_JSON} missing"

# ── Notes ─────────────────────────────────────────────────────────────────────
class Notes:
    def __init__(self, path):
        self.f = open(path, "w")
        self._w("# DC299 Phase 3 — Gödel Address Notes\n\n")
    def _w(self, t):
        self.f.write(t); self.f.flush(); print(t, end="")
    def section(self, t): self._w(f"\n## {t}\n\n")
    def log(self, t):     self._w(t + "\n")
    def finding(self, t): self._w(f"\n> **FINDING:** {t}\n\n")
    def close(self):      self.f.close()


# ── Semantic quality scoring ───────────────────────────────────────────────────
_CLEAN = re.compile(r'^[A-Za-z]{3,}$')

def quality(top_vocab, bot_vocab):
    tokens = [w for w, _ in top_vocab] + [w for w, _ in bot_vocab]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _CLEAN.match(t.strip())) / len(tokens)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all():
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings …", flush=True)
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()   # (vocab_size, D_MODEL)

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file) as f:
        td = json.load(f)
    vocab = td.get("model", {}).get("vocab", {})
    id_to_token  = {idx: tok for tok, idx in vocab.items()}
    token_to_id  = {tok: idx for tok, idx in vocab.items()}

    print("Loading Phase-0 concepts …", flush=True)
    with open(PHASE0_JSON) as f:
        p0 = json.load(f)
    # p0 is a list of {name, token_id, token_str}
    concept_names   = [c["word"]     for c in p0]
    concept_tids    = [c["token_id"] for c in p0]
    concept_embs    = np.array([embeddings[tid] for tid in concept_tids])

    print("Loading Phase-1 axes …", flush=True)
    with open(AXES_JSON) as f:
        axes_data = json.load(f)

    print(f"  {len(concept_names)} concepts  |  "
          f"{len(axes_data['axis_vectors'])} axis vectors  |  "
          f"emb shape {embeddings.shape}", flush=True)
    return embeddings, id_to_token, token_to_id, concept_names, concept_tids, concept_embs, axes_data


# ── Semantic axis selection ───────────────────────────────────────────────────
def select_semantic_axes(axes_data):
    """
    Return indices into axis_vectors of axes that pass quality ≥ QUALITY_MIN.
    Seed axes (type='seed') are always included regardless of quality score.
    Applies the rolling-window cliff: stop when 20-window mean drops below 0.4.
    """
    axes_meta    = axes_data["axes"]
    axis_vectors = np.array(axes_data["axis_vectors"])   # (n_axes, D)

    qualities = []
    for m in axes_meta:
        if m.get("type") == "seed":
            qualities.append(1.0)    # seed axes are always trusted
        else:
            q = quality(m.get("top_vocab", []), m.get("bot_vocab", []))
            qualities.append(q)

    # Rolling mean cliff detection (only over discovered axes)
    discovered_qualities = [q for m, q in zip(axes_meta, qualities)
                            if m.get("type") == "discovered"]
    cliff_idx = len(axes_meta)
    window = QUALITY_WINDOW
    for i in range(len(discovered_qualities) - window):
        if np.mean(discovered_qualities[i:i+window]) < 0.4:
            # cliff_idx in terms of full axes_meta index
            n_seeds = sum(1 for m in axes_meta if m.get("type") == "seed")
            cliff_idx = n_seeds + i
            break

    selected = []
    for i, (m, q) in enumerate(zip(axes_meta, qualities)):
        if i >= cliff_idx:
            break
        is_seed = m.get("type") == "seed"
        if (is_seed or q >= QUALITY_MIN) and len(selected) < MAX_SEMANTIC:
            selected.append(i)

    return selected, cliff_idx, qualities, axis_vectors


# ── Threshold computation ─────────────────────────────────────────────────────
def compute_thresholds(concept_embs_normed, axis_vectors, selected_indices):
    """Median projection of all concept embeddings onto each selected axis."""
    thresholds = []
    for i in selected_indices:
        proj = concept_embs_normed @ axis_vectors[i]
        thresholds.append(float(np.median(proj)))
    return np.array(thresholds)


# ── Gödel address computation ─────────────────────────────────────────────────
def compute_addresses(concept_embs_normed, axis_vectors, selected_indices, thresholds):
    """
    Returns addresses: (n_concepts, n_semantic_axes) bool array.
    bit_k = proj_k > threshold_k
    """
    n_axes = len(selected_indices)
    n_concepts = concept_embs_normed.shape[0]

    # Build projection matrix: (n_semantic, D)
    A = axis_vectors[selected_indices]          # (n_semantic, D)
    projections = concept_embs_normed @ A.T     # (n_concepts, n_semantic)
    addresses = projections > thresholds[np.newaxis, :]   # (n_concepts, n_semantic) bool
    return addresses, projections


# ── Address utilities ─────────────────────────────────────────────────────────
def hamming(a, b):
    return int(np.sum(a != b))

def decode_address(address_bits, axis_vectors, selected_indices,
                   embeddings, id_to_token, k=ANALOGY_K):
    """
    Reconstruct approximate embedding from binary address, find nearest vocab tokens.
    e_approx = mean over axes of (2*bit - 1) * axis_k   (unit-weighted sum)
    """
    signs = (address_bits.astype(float) * 2 - 1)   # +1 or -1 per axis
    A = axis_vectors[selected_indices]              # (n_semantic, D)
    e_approx = (signs[:, np.newaxis] * A).mean(axis=0)
    e_approx /= (np.linalg.norm(e_approx) + 1e-20)

    # Cosine similarity to full vocab
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / (norms + 1e-20)
    sims = emb_normed @ e_approx
    top_idx = np.argsort(sims)[-k:][::-1]

    def clean(tok):
        return tok.replace("Ġ", " ").replace("▁", " ").strip()

    return [(clean(id_to_token.get(int(i), f"?{i}")), float(sims[i]))
            for i in top_idx]


# ── Token lookup ──────────────────────────────────────────────────────────────
def find_token_id(word, token_to_id):
    for cand in [word, word.lower(), word.capitalize(),
                 f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
                 f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}"]:
        if cand in token_to_id:
            return token_to_id[cand]
    return None


def get_address_for_word(word, token_to_id, embeddings, axis_vectors,
                         selected_indices, thresholds):
    tid = find_token_id(word, token_to_id)
    if tid is None:
        raise RuntimeError(f"FAIL-FAST: token '{word}' not found in vocabulary")
    e = embeddings[tid]
    e_norm = e / (np.linalg.norm(e) + 1e-20)
    A = axis_vectors[selected_indices]
    proj = e_norm @ A.T
    addr = proj > thresholds
    return addr, proj, tid


# ── Relationship delta tests ───────────────────────────────────────────────────
def run_relationship_tests(token_to_id, embeddings, id_to_token,
                           axis_vectors, selected_indices, thresholds,
                           axes_meta, notes):

    def addr(word):
        a, p, tid = get_address_for_word(word, token_to_id, embeddings,
                                         axis_vectors, selected_indices, thresholds)
        return a, p, tid

    def addr_or_none(word):
        try:
            return addr(word)
        except RuntimeError as e:
            notes.log(f"  WARNING: {e}")
            return None, None, None

    notes.section("Relationship Delta Tests")

    # ── Find which axis index is the gender axis ──────────────────────────────
    gender_axis_slot = None
    capital_axis_slot = None
    european_axis_slot = None
    romance_axis_slot = None
    germanic_axis_slot = None
    for slot, i in enumerate(selected_indices):
        name = axes_meta[i].get("name", "")
        if name.startswith("is_female_gendered"):   gender_axis_slot   = slot
        if name.startswith("is_capital_city"):      capital_axis_slot  = slot
        if name.startswith("is_european_country"):  european_axis_slot = slot
        if name.startswith("is_romance_language"):  romance_axis_slot  = slot
        if name.startswith("is_germanic_language"): germanic_axis_slot = slot
    notes.log(f"  Seed axis slots: gender={gender_axis_slot}  "
              f"capital={capital_axis_slot}  european={european_axis_slot}  "
              f"romance={romance_axis_slot}  germanic={germanic_axis_slot}")
    notes.log("")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Gender analogy   king - man + woman ≈ queen
    # ─────────────────────────────────────────────────────────────────────────
    notes.log("### Test 1: Gender Analogy (address space)")
    notes.log("")

    gender_pairs = [
        ("king",   "queen"),
        ("man",    "woman"),
        ("boy",    "girl"),
        ("father", "mother"),
        ("brother","sister"),
        ("actor",  "actress"),
        ("prince", "princess"),
        ("hero",   "heroine"),
    ]

    gender_deltas = []
    for w_male, w_female in gender_pairs:
        a_m, _, _ = addr_or_none(w_male)
        a_f, _, _ = addr_or_none(w_female)
        if a_m is None or a_f is None:
            continue
        delta = a_m ^ a_f          # XOR = bits that differ
        hd = hamming(a_m, a_f)
        gender_deltas.append(delta)
        notes.log(f"  {w_male:12s} vs {w_female:12s}  Hamming={hd:3d}  "
                  f"gender_bit_flipped={'YES' if gender_axis_slot is not None and delta[gender_axis_slot] else 'no'}")

    if len(gender_deltas) >= 2:
        # Consistency: how similar are the deltas?
        pairs_hd = []
        for i in range(len(gender_deltas)):
            for j in range(i+1, len(gender_deltas)):
                pairs_hd.append(hamming(gender_deltas[i], gender_deltas[j]))
        notes.log(f"\n  Delta consistency (Hamming between pair-deltas):")
        notes.log(f"    Mean  = {np.mean(pairs_hd):.1f}")
        notes.log(f"    Min   = {min(pairs_hd)}")
        notes.log(f"    Max   = {max(pairs_hd)}")
        notes.log(f"    (Lower = more consistent gender transform)")

    # king - man + woman → find nearest by flipping gender bit on king
    notes.log("\n  Analogy: king - man + woman = ?")
    a_king, p_king, tid_king = addr_or_none("king")
    a_man,  p_man,  tid_man  = addr_or_none("man")
    a_woman,p_woman,_        = addr_or_none("woman")
    if a_king is not None and a_man is not None and a_woman is not None:
        # Method: flip the bits that differ between man and woman, applied to king
        gender_flip_mask = a_man ^ a_woman
        analogy_addr = a_king ^ gender_flip_mask
        notes.log(f"  Bits flipped in man↔woman delta: {int(np.sum(gender_flip_mask))}")
        neighbours = decode_address(analogy_addr, axis_vectors, selected_indices,
                                    embeddings, id_to_token)
        notes.log(f"  king - man + woman → top neighbours:")
        for tok, sim in neighbours[:5]:
            notes.log(f"    {tok:20s}  cos={sim:.4f}")

    # Focused gender axis test: does the gender seed axis flip for ALL pairs?
    if gender_axis_slot is not None:
        notes.log(f"\n  Focused test on gender axis (slot {gender_axis_slot}):")
        for w_male, w_female in gender_pairs:
            a_m, p_m, _ = addr_or_none(w_male)
            a_f, p_f, _ = addr_or_none(w_female)
            if a_m is None or a_f is None: continue
            bit_m = bool(a_m[gender_axis_slot])
            bit_f = bool(a_f[gender_axis_slot])
            proj_m = float(p_m[gender_axis_slot])
            proj_f = float(p_f[gender_axis_slot])
            notes.log(f"    {w_male:12s}(bit={int(bit_m)}, proj={proj_m:+.3f})  "
                      f"{w_female:12s}(bit={int(bit_f)}, proj={proj_f:+.3f})  "
                      f"flipped={'YES' if bit_m != bit_f else 'NO'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Capital-of analogy   France→Paris  Germany→Berlin  …
    # ─────────────────────────────────────────────────────────────────────────
    notes.log("\n### Test 2: Capital-of Transform Consistency")
    notes.log("")

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
    ]

    capital_deltas = []
    for country, capital in capital_pairs:
        a_c, _, _ = addr_or_none(country)
        a_k, _, _ = addr_or_none(capital)
        if a_c is None or a_k is None:
            continue
        delta = a_c ^ a_k
        hd = hamming(a_c, a_k)
        capital_deltas.append((country, capital, delta, hd))
        notes.log(f"  {country:12s} → {capital:12s}  Hamming={hd:3d}")

    if len(capital_deltas) >= 2:
        deltas_only = [d for _, _, d, _ in capital_deltas]
        pairs_hd = []
        for i in range(len(deltas_only)):
            for j in range(i+1, len(deltas_only)):
                pairs_hd.append(hamming(deltas_only[i], deltas_only[j]))
        notes.log(f"\n  Delta consistency (Hamming between country→capital deltas):")
        notes.log(f"    Mean  = {np.mean(pairs_hd):.1f}")
        notes.log(f"    Min   = {min(pairs_hd)}")
        notes.log(f"    Max   = {max(pairs_hd)}")
        notes.log(f"    (Lower = capital-of is a single consistent address transform)")

        # Which bits are consistently flipped across ALL capital pairs?
        delta_array = np.array(deltas_only)   # (n_pairs, n_axes)
        consensus = delta_array.mean(axis=0)  # fraction of pairs where bit flips
        n_invariant = int(np.sum(consensus == 1.0))
        n_most      = int(np.sum(consensus >= 0.8))
        notes.log(f"\n  Bits flipped in ALL capital pairs:  {n_invariant}")
        notes.log(f"  Bits flipped in ≥80% of pairs:      {n_most}")
        notes.finding(
            f"Capital-of transform: {n_invariant} invariant bits, "
            f"{n_most} bits consistent in ≥80% of pairs. "
            f"Mean delta Hamming = {np.mean(pairs_hd):.1f}."
        )

    # Focused capital axis test
    if capital_axis_slot is not None:
        notes.log(f"\n  Focused test on capital axis (slot {capital_axis_slot}):")
        for country, capital in capital_pairs:
            a_c, p_c, _ = addr_or_none(country)
            a_k, p_k, _ = addr_or_none(capital)
            if a_c is None or a_k is None: continue
            bit_c = bool(a_c[capital_axis_slot])
            bit_k = bool(a_k[capital_axis_slot])
            proj_c = float(p_c[capital_axis_slot])
            proj_k = float(p_k[capital_axis_slot])
            notes.log(f"    {country:12s}(bit={int(bit_c)}, proj={proj_c:+.3f})  "
                      f"{capital:12s}(bit={int(bit_k)}, proj={proj_k:+.3f})  "
                      f"capital_bit={'1' if bit_k else '0'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Language family  (French vs Italian vs German vs Japanese)
    # ─────────────────────────────────────────────────────────────────────────
    notes.log("\n### Test 3: Language Family Clustering")
    notes.log("")

    language_groups = {
        "Romance":  ["French", "Italian", "Spanish", "Portuguese"],
        "Germanic": ["German", "English", "Dutch", "Swedish", "Norwegian"],
        "Asian":    ["Japanese", "Chinese", "Korean"],
        "Semitic":  ["Arabic", "Hebrew"],
    }

    # Focused seed axis projections for languages
    if romance_axis_slot is not None or germanic_axis_slot is not None:
        notes.log(f"\n  Focused seed axis projections:")
        all_lang_words = [(w, g) for g, ws in language_groups.items() for w in ws]
        for w, group in all_lang_words:
            a, p, _ = addr_or_none(w)
            if a is None: continue
            rom = f"{p[romance_axis_slot]:+.3f}" if romance_axis_slot is not None else "N/A"
            ger = f"{p[germanic_axis_slot]:+.3f}" if germanic_axis_slot is not None else "N/A"
            notes.log(f"    {w:14s} ({group:8s})  romance_proj={rom}  germanic_proj={ger}")

    group_addrs = {}
    for group, words in language_groups.items():
        addrs = []
        for w in words:
            a, _, _ = addr_or_none(w)
            if a is not None:
                addrs.append(a)
        group_addrs[group] = addrs
        notes.log(f"  {group}: {len(addrs)} words loaded")

    notes.log("")
    # Within-group vs between-group Hamming
    for g1 in group_addrs:
        for g2 in group_addrs:
            a1 = group_addrs[g1]
            a2 = group_addrs[g2]
            if not a1 or not a2:
                continue
            pairs = []
            if g1 == g2:
                for i in range(len(a1)):
                    for j in range(i+1, len(a1)):
                        pairs.append(hamming(a1[i], a1[j]))
            else:
                for x in a1:
                    for y in a2:
                        pairs.append(hamming(x, y))
            tag = "within" if g1 == g2 else "between"
            if pairs:
                notes.log(f"  {g1:10s} {tag} {g2:10s}: "
                          f"mean Hamming = {np.mean(pairs):.1f}  "
                          f"(n={len(pairs)})")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Address decode — reconstruct embedding from address
    # ─────────────────────────────────────────────────────────────────────────
    notes.log("\n### Test 4: Address Decode (address → nearest vocab token)")
    notes.log("")

    test_words = ["king", "queen", "France", "Paris", "Tokyo", "German", "French"]
    for w in test_words:
        a, _, tid = addr_or_none(w)
        if a is None:
            continue
        neighbours = decode_address(a, axis_vectors, selected_indices,
                                    embeddings, id_to_token)
        actual_tok = id_to_token.get(tid, "?")
        top_tok = neighbours[0][0] if neighbours else "?"
        rank = next((i for i, (t, _) in enumerate(neighbours) if t.strip().lower() == w.lower()), None)
        notes.log(f"  {w:12s}  →  top={top_tok:15s}  "
                  f"self_rank={rank if rank is not None else 'not in top-'+str(ANALOGY_K)}")

    return capital_deltas, gender_deltas


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    notes = Notes(NOTES_PATH)
    t0 = time.time()

    # 1. Load
    (embeddings, id_to_token, token_to_id,
     concept_names, concept_tids, concept_embs, axes_data) = load_all()

    # 2. Normalize concept embeddings
    norms = np.linalg.norm(concept_embs, axis=1, keepdims=True)
    concept_embs_normed = concept_embs / (norms + 1e-20)

    # 3. Select semantic axes
    notes.section("Semantic Axis Selection")
    selected_indices, cliff_idx, qualities, axis_vectors = select_semantic_axes(axes_data)
    axes_meta = axes_data["axes"]

    notes.log(f"  Quality cliff detected at axis index {cliff_idx}")
    notes.log(f"  Semantic axes selected:  {len(selected_indices)}")
    notes.log(f"  (quality ≥ {QUALITY_MIN}, cap ≤ {MAX_SEMANTIC})")
    notes.log(f"  Selected axis indices: [{selected_indices[0]} … {selected_indices[-1]}]")
    notes.finding(f"Using {len(selected_indices)} semantic axes for Gödel addressing.")

    # 4. Compute thresholds
    notes.section("Threshold Computation")
    notes.log("  threshold_k = median projection of all concept embeddings onto axis_k")
    thresholds = compute_thresholds(concept_embs_normed, axis_vectors, selected_indices)
    notes.log(f"  Threshold range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")
    notes.log(f"  Threshold mean:   {thresholds.mean():.4f}")

    # 5. Compute Gödel addresses for all concepts
    notes.section("Gödel Address Assignment")
    notes.log(f"  Computing {len(selected_indices)}-bit addresses for "
              f"{len(concept_names)} concepts …")
    t1 = time.time()
    addresses, projections = compute_addresses(
        concept_embs_normed, axis_vectors, selected_indices, thresholds
    )
    notes.log(f"  Done in {time.time()-t1:.1f}s")
    notes.log(f"  Address matrix: {addresses.shape}  "
              f"(concepts × semantic_axes)")

    # Bit statistics
    bit_rates = addresses.mean(axis=0)
    notes.log(f"  Bit +1 rate: mean={bit_rates.mean():.3f}  "
              f"min={bit_rates.min():.3f}  max={bit_rates.max():.3f}")
    notes.log(f"  (Ideal for information density: ~0.5 per bit)")

    # Hamming distance stats across all concept pairs (sampled)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(concept_names), size=min(500, len(concept_names)), replace=False)
    sample_addrs = addresses[sample]
    all_hd = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            all_hd.append(hamming(sample_addrs[i], sample_addrs[j]))
    notes.log(f"\n  Pairwise Hamming (500-concept sample):")
    notes.log(f"    Mean  = {np.mean(all_hd):.1f}")
    notes.log(f"    Std   = {np.std(all_hd):.1f}")
    notes.log(f"    Min   = {min(all_hd)}")
    notes.log(f"    Max   = {max(all_hd)}")
    notes.finding(
        f"Mean pairwise Hamming distance = {np.mean(all_hd):.1f} / {len(selected_indices)} bits. "
        f"(Expected for random: {len(selected_indices)//2})"
    )

    # 6. Relationship tests
    capital_deltas, gender_deltas = run_relationship_tests(
        token_to_id, embeddings, id_to_token,
        axis_vectors, selected_indices, thresholds,
        axes_meta, notes,
    )

    # 7. Serialize
    notes.section("Output")
    output = {
        "n_concepts":       len(concept_names),
        "n_semantic_axes":  len(selected_indices),
        "quality_cliff":    cliff_idx,
        "selected_axis_indices": selected_indices,
        "thresholds":       thresholds.tolist(),
        "concepts": [
            {
                "name":     concept_names[i],
                "token_id": int(concept_tids[i]),
                "address":  addresses[i].tolist(),   # list of bool
            }
            for i in range(len(concept_names))
        ],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    notes.log(f"  Output: {OUTPUT_JSON}")
    notes.log(f"  Total time: {time.time()-t0:.1f}s")
    notes.close()

    print(f"\n{'='*60}")
    print(f"Phase 3 complete.")
    print(f"  Concepts addressed : {len(concept_names)}")
    print(f"  Semantic axes used : {len(selected_indices)}")
    print(f"  Output             : {OUTPUT_JSON}")
    print(f"  Notes              : {NOTES_PATH}")


if __name__ == "__main__":
    main()
