#!/usr/bin/env python3
"""
DC299 Phase 1 — Iterative Residual Decomposition (IRD) Axis Discovery
======================================================================

Algorithm:
  0. Load Phase-0 concept set + known anchor directions (6 seed axes).
  1. Orthogonalise known axes via QR → current basis B.
  2. Project all concept embeddings onto B, compute residual matrix R.
  3. SVD on R → leading left singular vector u₁ = candidate axis.
  4. Validate candidate:
       a. Binary separation test   (top-K vs bottom-K concept projections)
       b. Vocabulary coherence     (top/bottom vocab words make semantic sense)
       c. Orthogonality check      (|dot(u₁, b_i)| < ORTH_TOL for all b_i ∈ B)
       d. Variance explained       (singular value² / total residual variance)
  5. If valid: add u₁ to B, increment axis count, go to 2.
  6. Stop when:
       • variance_explained_cumulative > MAX_VARIANCE
       • no valid axis found after PATIENCE consecutive attempts
       • axis count reaches MAX_AXES

Fail-fast: any missing dependency raises immediately; no graceful fallbacks.

Outputs:
  dc299_phase1_axes.json      — discovered axes + metadata
  dc299_phase1_notes.md       — field notes
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR   = (PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2"
               / "phi_model")
PHASE0_JSON = SCRIPT_DIR / "dc299_phase0_concepts.json"
OUTPUT_JSON = SCRIPT_DIR / "dc299_phase1_axes.json"
NOTES_PATH  = SCRIPT_DIR / "dc299_phase1_notes.md"

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_AXES          = 1500    # hard cap
MAX_VARIANCE      = 0.95    # stop when cumulative variance explained ≥ this
PATIENCE          = 10      # consecutive binary-rejected candidates before stopping
ORTH_TOL          = 0.10    # |cos θ| threshold for "sufficiently orthogonal"
BINARY_TOP_K      = 50      # concepts used in binary separation test
MIN_BINARY_ACC    = 0.75    # minimum separation accuracy to accept axis
MIN_VARIANCE_STEP = 0.001   # axis must explain ≥ 0.1 % of residual variance
VOCAB_DISPLAY_K   = 20      # top/bottom vocab tokens to log per axis
HOLDOUT_FRAC      = 0.20    # fraction of concepts held out for validation
HOLDOUT_SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================================
# Notes
# ============================================================================

class Notes:
    def __init__(self, path):
        self.f = open(path, "w")
        self._w("# DC299 Phase 1 — IRD Axis Discovery Notes\n\n")

    def _w(self, text):
        self.f.write(text)
        self.f.flush()
        print(text, end="")

    def section(self, title):
        self._w(f"\n## {title}\n\n")

    def log(self, text):
        self._w(text + "\n")

    def finding(self, text):
        self._w(f"\n> **FINDING:** {text}\n\n")

    def close(self):
        self.f.close()


# ============================================================================
# Data loading
# ============================================================================

def load_embeddings_and_vocab():
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings …", flush=True)
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()          # (vocab_size, D_MODEL)

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file) as f:
        tokenizer_data = json.load(f)

    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    print(f"  embeddings={embeddings.shape}  vocab={len(id_to_token)}", flush=True)
    return embeddings, id_to_token


def load_phase0_concepts(embeddings):
    if not PHASE0_JSON.exists():
        raise FileNotFoundError(
            f"FAIL-FAST: Phase 0 output not found at {PHASE0_JSON}. "
            "Run dc299_phase0_concept_mining.py first."
        )
    with open(PHASE0_JSON) as f:
        records = json.load(f)

    concept_embs = np.array([embeddings[r["token_id"]] for r in records])
    print(f"  Loaded {len(records)} Phase-0 concepts, "
          f"matrix shape {concept_embs.shape}", flush=True)
    return records, concept_embs


# ============================================================================
# Seed axes (from explore_platonic_residuals.py / DC298)
# ============================================================================

def build_seed_axes(embeddings):
    """Reproduce the 6 known anchor directions from DC298."""

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file) as f:
        tokenizer_data = json.load(f)
    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    token_to_id = {tok: idx for tok, idx in vocab.items()}

    def find_id(word):
        for cand in [word, word.lower(), word.capitalize(),
                     f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
                     f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}"]:
            if cand in token_to_id:
                return token_to_id[cand]
        return None

    def axis(positives, negatives):
        pos_ids = [find_id(w) for w in positives if find_id(w) is not None]
        neg_ids = [find_id(w) for w in negatives if find_id(w) is not None]
        if len(pos_ids) < 2 or len(neg_ids) < 2:
            raise RuntimeError(
                f"FAIL-FAST: insufficient token lookups for seed axis "
                f"(pos={len(pos_ids)}, neg={len(neg_ids)})"
            )
        pos_mean = np.mean([embeddings[i] for i in pos_ids], axis=0)
        neg_mean = np.mean([embeddings[i] for i in neg_ids], axis=0)
        d = pos_mean - neg_mean
        return d / (np.linalg.norm(d) + 1e-20)

    seed_defs = {
        "is_european_country": axis(
            ["France", "Germany", "Poland", "Norway", "Sweden",
             "Italy", "Portugal", "Spain", "Greece", "Ireland",
             "Finland", "Denmark", "Austria", "Belgium", "Netherlands",
             "Switzerland", "Russia"],
            ["Japan", "China", "Egypt", "Australia", "Thailand",
             "India", "Brazil", "Korea", "Turkey", "Nigeria",
             "Kenya", "Morocco", "Israel", "Iran", "Vietnam",
             "Indonesia", "Philippines", "Mexico", "Canada",
             "Argentina", "Chile", "Colombia", "Peru"],
        ),
        "is_asian_country": axis(
            ["Japan", "China", "Thailand", "India", "Korea",
             "Vietnam", "Indonesia", "Philippines", "Malaysia",
             "Singapore", "Iran", "Iraq", "Israel"],
            ["France", "Germany", "Poland", "Norway", "Sweden",
             "Italy", "Portugal", "Spain", "Egypt", "Australia",
             "Brazil", "Nigeria", "Kenya", "Morocco",
             "Mexico", "Canada", "Argentina"],
        ),
        "is_capital_city": axis(
            ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
             "Canberra", "Bangkok", "Warsaw", "Oslo", "Stockholm",
             "Delhi", "Seoul", "Rome", "Lisbon", "Moscow",
             "Madrid", "Athens", "Ankara", "Dublin", "Helsinki",
             "Copenhagen", "Vienna", "Brussels", "Amsterdam",
             "Ottawa", "Lima", "Tehran", "Baghdad", "Hanoi"],
            ["France", "Germany", "Japan", "China", "Egypt",
             "Australia", "Thailand", "Poland", "Norway", "Sweden",
             "India", "Korea", "Italy", "Portugal", "Russia",
             "Spain", "Greece", "Turkey", "Ireland",
             "Brazil", "Mexico", "Canada", "Argentina"],
        ),
        "is_romance_language": axis(
            ["French", "Italian", "Portuguese", "Spanish"],
            ["German", "Japanese", "Chinese", "Arabic", "English",
             "Korean", "Thai", "Polish", "Norwegian", "Swedish",
             "Dutch", "Greek", "Turkish", "Hindi", "Finnish",
             "Russian", "Danish", "Persian", "Vietnamese"],
        ),
        "is_germanic_language": axis(
            ["German", "English", "Dutch", "Norwegian", "Swedish", "Danish"],
            ["French", "Italian", "Portuguese", "Spanish",
             "Japanese", "Chinese", "Arabic", "Korean",
             "Polish", "Greek", "Turkish", "Hindi", "Finnish",
             "Russian", "Thai", "Persian", "Vietnamese"],
        ),
        "is_female_gendered": axis(
            ["queen", "woman", "girl", "mother", "sister",
             "daughter", "wife", "aunt", "princess", "actress",
             "waitress", "heroine"],
            ["king", "man", "boy", "father", "brother",
             "son", "husband", "uncle", "prince", "actor",
             "waiter", "hero"],
        ),
    }

    return seed_defs


# ============================================================================
# Validation helpers
# ============================================================================

def binary_separation_accuracy(train_embs, holdout_embs, candidate, top_k):
    """
    Derive threshold from train_embs top-K / bottom-K projections,
    then measure accuracy on holdout_embs.

    This prevents trivially-true accuracy caused by testing the same
    data that SVD was optimised against.
    """
    train_proj   = train_embs   @ candidate
    holdout_proj = holdout_embs @ candidate

    k = min(top_k, len(train_proj) // 4)
    order = np.argsort(train_proj)
    pos_mean = train_proj[order[-k:]].mean()
    neg_mean = train_proj[order[:k]].mean()
    threshold = (pos_mean + neg_mean) / 2
    gap       = float(pos_mean - neg_mean)

    # Classify ALL holdout concepts — not just the extreme ones
    predicted_pos = holdout_proj > threshold
    # "True" label: above-median = positive, below-median = negative
    median = float(np.median(holdout_proj))
    true_pos = holdout_proj > median

    correct = int((predicted_pos == true_pos).sum())
    total   = len(holdout_proj)
    return float(correct / total), gap


def vocab_top_bottom(embeddings, id_to_token, candidate, k=VOCAB_DISPLAY_K):
    """Top and bottom vocab tokens along the candidate direction."""
    projections = embeddings @ candidate
    top_idx = np.argsort(projections)[-k:][::-1]
    bot_idx = np.argsort(projections)[:k]

    def clean(tok):
        return tok.replace("Ġ", " ").replace("▁", " ").strip()

    top = [(clean(id_to_token.get(int(i), f"?{i}")), float(projections[i]))
           for i in top_idx]
    bot = [(clean(id_to_token.get(int(i), f"?{i}")), float(projections[i]))
           for i in bot_idx]
    return top, bot


def variance_explained_by_step(residual_matrix, candidate):
    """Fraction of total residual variance explained by projecting onto candidate."""
    projections = residual_matrix @ candidate          # (n_concepts,)
    var_on_axis  = float(np.var(projections) * residual_matrix.shape[0])
    total_var    = float(np.sum(residual_matrix ** 2))
    if total_var < 1e-30:
        return 0.0
    return var_on_axis / total_var


# ============================================================================
# Gram-Schmidt orthogonalisation (single step)
# ============================================================================

def orthogonalise_against_basis(v, basis_cols):
    """
    Remove components of v that lie in span(basis_cols).
    basis_cols: list of unit vectors.
    Returns normalised residual or None if v collapses.
    """
    v = v.copy()
    for b in basis_cols:
        v -= np.dot(v, b) * b
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return None
    return v / norm


# ============================================================================
# Main IRD loop
# ============================================================================

def ird_loop(concept_embs, concepts_meta, embeddings, id_to_token,
             seed_defs, notes):
    """
    Core iterative residual decomposition.
    Returns list of axis records.
    """
    D = concept_embs.shape[1]
    n_total = concept_embs.shape[0]

    # --- Train / holdout split ---------------------------------------------
    rng      = np.random.default_rng(HOLDOUT_SEED)
    all_idx  = np.arange(n_total)
    rng.shuffle(all_idx)
    n_hold   = max(1, int(n_total * HOLDOUT_FRAC))
    hold_idx = all_idx[:n_hold]
    train_idx= all_idx[n_hold:]

    train_embs   = concept_embs[train_idx]
    holdout_embs = concept_embs[hold_idx]
    notes.section("Train / Holdout Split")
    notes.log(f"  Total concepts : {n_total}")
    notes.log(f"  Train          : {len(train_idx)}")
    notes.log(f"  Holdout        : {len(hold_idx)}")
    notes.log(f"  SVD runs on train only; binary_acc measured on holdout.")

    # --- Initialise with seed axes ----------------------------------------
    axes = []                     # list of unit vectors (np arrays)
    axes_meta = []                # list of dicts with metadata

    notes.section("Seed Axes")
    for name, direction in seed_defs.items():
        acc, gap = binary_separation_accuracy(train_embs, holdout_embs,
                                              direction, BINARY_TOP_K)
        notes.log(f"  {name:30s}  binary_acc={acc:.3f}  gap={gap:.4f}")
        axes.append(direction.copy())
        axes_meta.append({
            "index":      len(axes) - 1,
            "type":       "seed",
            "name":       name,
            "binary_acc": round(acc, 4),
            "gap":        round(gap, 4),
        })

    notes.log(f"\nStarting IRD with {len(axes)} seed axes.")

    # --- Compute running residual ------------------------------------------
    # residual_matrix runs on TRAIN only (SVD must not see holdout)
    # We keep a parallel holdout_residual to update projections for validation

    residual_matrix  = train_embs.copy()
    holdout_residual = holdout_embs.copy()
    total_original_variance = float(np.sum(train_embs ** 2))
    cumulative_variance_removed = 0.0

    # Project out seed axes
    for b in axes:
        projections = residual_matrix @ b           # (n_train,)
        residual_matrix -= np.outer(projections, b)
        h_proj = holdout_residual @ b
        holdout_residual -= np.outer(h_proj, b)

    remaining_var = float(np.sum(residual_matrix ** 2))
    cumulative_variance_removed = (total_original_variance - remaining_var) / total_original_variance
    notes.log(f"After seeding: cumulative variance explained = "
              f"{cumulative_variance_removed:.4f}")

    # --- IRD loop ----------------------------------------------------------
    patience_count = 0
    iteration      = 0
    discovered     = []           # only newly discovered (non-seed)

    notes.section("IRD Discovery Loop")

    while True:
        iteration += 1

        # Stop conditions
        if cumulative_variance_removed >= MAX_VARIANCE:
            notes.finding(
                f"Stopped at iteration {iteration}: "
                f"cumulative variance {cumulative_variance_removed:.4f} ≥ {MAX_VARIANCE}"
            )
            break
        if patience_count >= PATIENCE:
            notes.finding(
                f"Stopped at iteration {iteration}: "
                f"{PATIENCE} consecutive invalid candidates."
            )
            break
        if len(axes) >= MAX_AXES:
            notes.finding(f"Stopped: reached MAX_AXES={MAX_AXES}.")
            break

        t0 = time.time()

        # 1. SVD on residual matrix → leading right singular vector
        # We want u s.t. residual ≈ (residual @ u) u^T + lower terms.
        # np.linalg.svd is expensive; use randomised SVD via power iteration.
        candidate = _leading_right_singular_vector(residual_matrix)
        if candidate is None:
            notes.log(f"  [{iteration:4d}] SVD returned None — residual collapsed. Stopping.")
            break

        # 2. Orthogonalise against existing basis
        candidate_orth = orthogonalise_against_basis(candidate, axes)
        if candidate_orth is None:
            notes.log(f"  [{iteration:4d}] Candidate collapsed after orthogonalisation. "
                      f"(patience {patience_count+1}/{PATIENCE})")
            patience_count += 1
            continue

        # 3. Orthogonality check
        max_dot = max(abs(np.dot(candidate_orth, b)) for b in axes)
        if max_dot > ORTH_TOL:
            notes.log(f"  [{iteration:4d}] max |dot| = {max_dot:.4f} > {ORTH_TOL} "
                      f"— not orthogonal enough. (patience {patience_count+1}/{PATIENCE})")
            patience_count += 1
            continue

        # 4. Variance explained by this step (train portion only)
        step_var = variance_explained_by_step(residual_matrix, candidate_orth)
        if step_var < MIN_VARIANCE_STEP:
            notes.log(f"  [{iteration:4d}] step_var={step_var:.6f} < {MIN_VARIANCE_STEP} "
                      f"— negligible. Stopping.")
            break

        # 5. Binary separation test (holdout only — real generalisation test)
        acc, gap = binary_separation_accuracy(train_embs, holdout_embs,
                                              candidate_orth, BINARY_TOP_K)

        # 6. Vocab coherence display
        top_words, bot_words = vocab_top_bottom(
            embeddings, id_to_token, candidate_orth, k=VOCAB_DISPLAY_K
        )

        elapsed = time.time() - t0
        top_str = " | ".join(f"{w}({v:.2f})" for w, v in top_words[:8])
        bot_str = " | ".join(f"{w}({v:.2f})" for w, v in bot_words[:8])

        notes.log(f"\n  [{iteration:4d}]  axes={len(axes)}  "
                  f"step_var={step_var:.4f}  binary_acc={acc:.3f}  "
                  f"gap={gap:.4f}  max_dot={max_dot:.4f}  ({elapsed:.1f}s)")
        notes.log(f"    TOP:  {top_str}")
        notes.log(f"    BOT:  {bot_str}")

        if acc < MIN_BINARY_ACC:
            notes.log(f"    REJECTED: binary_acc={acc:.3f} < {MIN_BINARY_ACC} "
                      f"(patience {patience_count+1}/{PATIENCE})")
            patience_count += 1
            # Deflate residuals by rejected candidate so SVD finds a NEW direction
            # next iteration. Do NOT add to axes (official basis).
            rej_proj = residual_matrix @ candidate_orth
            residual_matrix  -= np.outer(rej_proj, candidate_orth)
            rh_proj = holdout_residual @ candidate_orth
            holdout_residual -= np.outer(rh_proj, candidate_orth)
            continue

        # --- Accept axis ---------------------------------------------------
        patience_count = 0
        axis_idx = len(axes)

        # Suggest a name from top/bot vocab tokens
        top_labels = [w for w, _ in top_words[:4]]
        bot_labels = [w for w, _ in bot_words[:4]]
        suggested_name = (f"axis_{axis_idx:03d}__"
                          f"[{','.join(top_labels[:3])}]_vs_"
                          f"[{','.join(bot_labels[:3])}]")

        meta = {
            "index":          axis_idx,
            "type":           "discovered",
            "name":           suggested_name,
            "iteration":      iteration,
            "binary_acc":     round(acc, 4),
            "gap":            round(gap, 4),
            "step_var":       round(step_var, 6),
            "max_orth_dot":   round(max_dot, 6),
            "top_vocab":      [(w, round(v, 4)) for w, v in top_words[:10]],
            "bot_vocab":      [(w, round(v, 4)) for w, v in bot_words[:10]],
        }
        axes.append(candidate_orth)
        axes_meta.append(meta)
        discovered.append(meta)

        # Update residuals (both train and holdout)
        projections       = residual_matrix @ candidate_orth
        residual_matrix  -= np.outer(projections, candidate_orth)
        h_proj            = holdout_residual @ candidate_orth
        holdout_residual -= np.outer(h_proj, candidate_orth)

        remaining_var = float(np.sum(residual_matrix ** 2))
        cumulative_variance_removed = (
            (total_original_variance - remaining_var) / total_original_variance
        )
        meta["cumulative_var"] = round(cumulative_variance_removed, 6)

        notes.log(f"    ACCEPTED as axis_{axis_idx:03d}  "
                  f"cumulative_var={cumulative_variance_removed:.4f}")

    return axes, axes_meta, discovered


# ============================================================================
# Randomised leading right singular vector
# ============================================================================

def _leading_right_singular_vector(M, n_iter=4, seed=None):
    """
    Fast estimation of leading right singular vector of M (n_concepts × D)
    via randomised power iteration.  Returns unit vector in R^D.
    """
    rng   = np.random.default_rng(seed)
    n, D  = M.shape
    # Start with a random unit vector in D-space
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v)

    for _ in range(n_iter):
        u = M @ v            # (n,)
        u /= (np.linalg.norm(u) + 1e-20)
        v = M.T @ u          # (D,)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-20:
            return None
        v /= norm_v

    return v


# ============================================================================
# Main
# ============================================================================

def main():
    notes = Notes(NOTES_PATH)
    notes.section("Configuration")
    notes.log(f"  MAX_AXES={MAX_AXES}  MAX_VARIANCE={MAX_VARIANCE}")
    notes.log(f"  PATIENCE={PATIENCE}  ORTH_TOL={ORTH_TOL}")
    notes.log(f"  BINARY_TOP_K={BINARY_TOP_K}  MIN_BINARY_ACC={MIN_BINARY_ACC}")
    notes.log(f"  MIN_VARIANCE_STEP={MIN_VARIANCE_STEP}")

    embeddings, id_to_token = load_embeddings_and_vocab()
    concepts_meta, concept_embs = load_phase0_concepts(embeddings)

    # Normalise concept embeddings for projection (do NOT modify originals)
    concept_norms = np.linalg.norm(concept_embs, axis=1, keepdims=True)
    concept_embs_normed = concept_embs / (concept_norms + 1e-20)

    seed_defs = build_seed_axes(embeddings)
    notes.log(f"\nSeed axes loaded: {list(seed_defs.keys())}")

    axes, axes_meta, discovered = ird_loop(
        concept_embs_normed, concepts_meta, embeddings, id_to_token,
        seed_defs, notes,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    notes.section("Summary")
    notes.log(f"  Total axes in basis:     {len(axes)}")
    notes.log(f"  Seed axes:               {sum(1 for m in axes_meta if m['type'] == 'seed')}")
    notes.log(f"  Discovered axes:         {len(discovered)}")

    if discovered:
        notes.finding(
            f"Discovered {len(discovered)} new binary truth axes beyond the 6 seeds."
        )
    else:
        notes.finding(
            "No new axes discovered beyond seeds. "
            "Either concept set is too small or seed axes already span the "
            "meaningful variance. Consider larger Phase-0 concept set or "
            "lower MIN_BINARY_ACC threshold."
        )

    # ── Serialise ──────────────────────────────────────────────────────────
    output = {
        "total_axes":     len(axes),
        "seed_count":     sum(1 for m in axes_meta if m["type"] == "seed"),
        "discovered_count": len(discovered),
        "axes": axes_meta,
        # Store axis vectors for downstream use
        "axis_vectors": [v.tolist() for v in axes],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    notes.log(f"\nOutput: {OUTPUT_JSON}")
    notes.close()

    print(f"\n{'='*60}")
    print(f"Phase 1 complete.")
    print(f"  Total axes   : {len(axes)}")
    print(f"  Discovered   : {len(discovered)}")
    print(f"  Output       : {OUTPUT_JSON}")
    print(f"  Notes        : {NOTES_PATH}")


if __name__ == "__main__":
    main()
