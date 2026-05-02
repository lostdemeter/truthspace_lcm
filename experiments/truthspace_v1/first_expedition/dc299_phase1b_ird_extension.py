#!/usr/bin/env python3
"""
DC299 Phase 1b — IRD Axis Discovery Continuation
=================================================

Continues Phase 1 from where it stopped (1500 axes hit MAX_AXES cap).
Loads existing axis vectors as the initial basis, projects them out of
the residual matrix, then runs the IRD SVD loop to discover axes 1501+.

Algorithm:
  1. Load Phase-0 concept embeddings (normalised).
  2. Load all existing axes from dc299_phase1_axes.json.
  3. Compute the residual matrix by projecting out all existing axes.
  4. Continue IRD SVD loop — exactly as Phase 1 — until:
       * cumulative variance ≥ MAX_VARIANCE
       * PATIENCE consecutive rejections
       * new axis count reaches MAX_NEW_AXES
  5. Save new axes to dc299_phase1b_axes.json.

Fail-fast: no graceful fallbacks, missing dependency raises immediately.

Outputs:
  dc299_phase1b_axes.json  — newly discovered axes (indices 1500+)
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

MODEL_DIR      = (PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2"
                  / "phi_model")
PHASE0_JSON    = SCRIPT_DIR / "dc299_phase0_concepts.json"
PHASE1_JSON    = SCRIPT_DIR / "dc299_phase1_axes.json"
OUTPUT_JSON    = SCRIPT_DIR / "dc299_phase1b_axes.json"

for p in [PHASE0_JSON, PHASE1_JSON]:
    assert p.exists(), f"FAIL-FAST: {p} missing"

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_NEW_AXES      = 1500   # discover up to this many NEW axes (total → 3000)
MAX_VARIANCE      = 0.98   # stop when cumulative variance explained ≥ this
PATIENCE          = 10     # consecutive binary-rejected candidates before stopping
ORTH_TOL          = 0.10
BINARY_TOP_K      = 50
MIN_BINARY_ACC    = 0.75
MIN_VARIANCE_STEP = 0.0005  # lower threshold — later axes explain less variance
VOCAB_DISPLAY_K   = 10
HOLDOUT_FRAC      = 0.20
HOLDOUT_SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_embeddings_and_vocab():
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings …", flush=True)
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snap = os.listdir(cache_dir)[0]
    with open(os.path.join(cache_dir, snap, "tokenizer.json")) as f:
        td = json.load(f)
    vocab = td.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    print(f"  embeddings={embeddings.shape}  vocab={len(id_to_token)}", flush=True)
    return embeddings, id_to_token


def load_phase0_concepts(embeddings):
    with open(PHASE0_JSON) as f:
        records = json.load(f)
    concept_embs = np.array([embeddings[r["token_id"]] for r in records])
    norms = np.linalg.norm(concept_embs, axis=1, keepdims=True)
    concept_embs_normed = concept_embs / (norms + 1e-20)
    print(f"  {len(records)} Phase-0 concepts, shape {concept_embs_normed.shape}", flush=True)
    return records, concept_embs_normed


def load_existing_axes():
    print("Loading Phase-1 axes …", flush=True)
    with open(PHASE1_JSON) as f:
        data = json.load(f)
    vectors = np.array(data["axis_vectors"], dtype=np.float64)
    meta    = data["axes"]
    print(f"  {len(vectors)} existing axes loaded", flush=True)
    return vectors, meta


# ── Validation helpers (identical to Phase 1) ────────────────────────────────

def binary_separation_accuracy(train_embs, holdout_embs, candidate, top_k):
    train_proj   = train_embs   @ candidate
    holdout_proj = holdout_embs @ candidate

    k = min(top_k, len(train_proj) // 4)
    order    = np.argsort(train_proj)
    pos_mean = train_proj[order[-k:]].mean()
    neg_mean = train_proj[order[:k]].mean()
    threshold = (pos_mean + neg_mean) / 2
    gap       = float(pos_mean - neg_mean)

    predicted_pos = holdout_proj > threshold
    median        = float(np.median(holdout_proj))
    true_pos      = holdout_proj > median

    correct = int((predicted_pos == true_pos).sum())
    return float(correct / len(holdout_proj)), gap


def vocab_top_bottom(embeddings, id_to_token, candidate, k=VOCAB_DISPLAY_K):
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
    projections  = residual_matrix @ candidate
    var_on_axis  = float(np.var(projections) * residual_matrix.shape[0])
    total_var    = float(np.sum(residual_matrix ** 2))
    return var_on_axis / total_var if total_var > 1e-30 else 0.0


def orthogonalise_against_basis(v, basis_cols):
    v = v.copy()
    for b in basis_cols:
        v -= np.dot(v, b) * b
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return None
    return v / norm


def _leading_right_singular_vector(M, n_iter=4, seed=None):
    rng  = np.random.default_rng(seed)
    n, D = M.shape
    v    = rng.standard_normal(D)
    v   /= np.linalg.norm(v)
    for _ in range(n_iter):
        u  = M @ v
        u /= (np.linalg.norm(u) + 1e-20)
        v  = M.T @ u
        nv = np.linalg.norm(v)
        if nv < 1e-20:
            return None
        v /= nv
    return v


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    embeddings, id_to_token = load_embeddings_and_vocab()
    concepts_meta, concept_embs_normed = load_phase0_concepts(embeddings)
    existing_vectors, existing_meta    = load_existing_axes()

    n_total = concept_embs_normed.shape[0]

    # Train/holdout split (same seed as Phase 1 for reproducibility)
    rng      = np.random.default_rng(HOLDOUT_SEED)
    all_idx  = np.arange(n_total)
    rng.shuffle(all_idx)
    n_hold   = max(1, int(n_total * HOLDOUT_FRAC))
    hold_idx = all_idx[:n_hold]
    train_idx= all_idx[n_hold:]

    train_embs   = concept_embs_normed[train_idx].astype(np.float64)
    holdout_embs = concept_embs_normed[hold_idx].astype(np.float64)

    print(f"\n  Train: {len(train_idx)}  Holdout: {len(hold_idx)}", flush=True)

    # ── Build residual by projecting out all existing axes ────────────────────
    print(f"\nProjecting out {len(existing_vectors)} existing axes from residual …",
          flush=True)
    t0 = time.time()

    residual_matrix  = train_embs.copy()
    holdout_residual = holdout_embs.copy()
    total_original_variance = float(np.sum(train_embs ** 2))

    # Use the existing vectors directly as orthogonal basis
    # (Phase 1 already produced orthogonal vectors via Gram-Schmidt)
    axes = [v.astype(np.float64) for v in existing_vectors]

    for b in axes:
        b_unit = b / (np.linalg.norm(b) + 1e-20)
        proj   = residual_matrix @ b_unit
        residual_matrix  -= np.outer(proj, b_unit)
        h_proj = holdout_residual @ b_unit
        holdout_residual -= np.outer(h_proj, b_unit)

    remaining_var = float(np.sum(residual_matrix ** 2))
    cumulative_variance_removed = (total_original_variance - remaining_var) / total_original_variance
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    print(f"  Cumulative variance accounted for: {cumulative_variance_removed:.4f}", flush=True)
    print(f"  Residual variance remaining       : {remaining_var:.1f}", flush=True)

    if cumulative_variance_removed >= MAX_VARIANCE:
        print(f"  Already at MAX_VARIANCE={MAX_VARIANCE}. Nothing to discover.")
        return

    # ── Continue IRD loop ─────────────────────────────────────────────────────
    print(f"\nStarting Phase 1b IRD from axis {len(axes)} …", flush=True)

    new_axes_meta = []
    patience_count = 0
    iteration      = 0
    n_accepted     = 0

    while True:
        iteration += 1

        if cumulative_variance_removed >= MAX_VARIANCE:
            print(f"\n[{iteration}] Stopped: cumulative variance {cumulative_variance_removed:.4f} >= {MAX_VARIANCE}")
            break
        if patience_count >= PATIENCE:
            print(f"\n[{iteration}] Stopped: {PATIENCE} consecutive rejections.")
            break
        if n_accepted >= MAX_NEW_AXES:
            print(f"\n[{iteration}] Stopped: reached MAX_NEW_AXES={MAX_NEW_AXES}.")
            break

        t0 = time.time()

        candidate = _leading_right_singular_vector(residual_matrix)
        if candidate is None:
            print(f"  [{iteration:5d}] SVD returned None — residual collapsed. Stopping.")
            break

        candidate_orth = orthogonalise_against_basis(candidate, axes)
        if candidate_orth is None:
            patience_count += 1
            continue

        max_dot = max(abs(np.dot(candidate_orth, b / (np.linalg.norm(b)+1e-20)))
                      for b in axes[-50:])   # check recent basis only for speed
        if max_dot > ORTH_TOL:
            patience_count += 1
            continue

        step_var = variance_explained_by_step(residual_matrix, candidate_orth)
        if step_var < MIN_VARIANCE_STEP:
            print(f"\n  [{iteration:5d}] step_var={step_var:.7f} < {MIN_VARIANCE_STEP} — stopping.")
            break

        acc, gap = binary_separation_accuracy(train_embs, holdout_embs,
                                              candidate_orth, BINARY_TOP_K)

        top_words, bot_words = vocab_top_bottom(
            embeddings, id_to_token, candidate_orth, k=VOCAB_DISPLAY_K
        )

        elapsed = time.time() - t0
        top_str = " | ".join(f"{w}({v:.2f})" for w, v in top_words[:5])
        bot_str = " | ".join(f"{w}({v:.2f})" for w, v in bot_words[:5])

        axis_global_idx = len(axes)

        if n_accepted % 50 == 0 or acc < MIN_BINARY_ACC:
            print(f"  [{iteration:5d}]  axis={axis_global_idx}  "
                  f"step_var={step_var:.5f}  acc={acc:.3f}  gap={gap:.4f}  "
                  f"cum_var={cumulative_variance_removed:.4f}  ({elapsed:.1f}s)")
            print(f"    TOP: {top_str}")
            print(f"    BOT: {bot_str}", flush=True)

        if acc < MIN_BINARY_ACC:
            patience_count += 1
            # Deflate rejected direction so SVD finds something new next iter
            rej_proj = residual_matrix @ candidate_orth
            residual_matrix  -= np.outer(rej_proj, candidate_orth)
            h_proj = holdout_residual @ candidate_orth
            holdout_residual -= np.outer(h_proj, candidate_orth)
            continue

        # Accept
        patience_count = 0
        n_accepted    += 1

        top_labels = [w for w, _ in top_words[:4]]
        bot_labels = [w for w, _ in bot_words[:4]]
        name = (f"axis_{axis_global_idx:04d}__"
                f"[{','.join(top_labels[:3])}]_vs_"
                f"[{','.join(bot_labels[:3])}]")

        meta = {
            "index":         axis_global_idx,
            "type":          "discovered",
            "name":          name,
            "iteration":     iteration,
            "binary_acc":    round(acc, 4),
            "gap":           round(gap, 4),
            "step_var":      round(step_var, 7),
            "max_orth_dot":  round(max_dot, 6),
            "top_vocab":     [(w, round(v, 4)) for w, v in top_words],
            "bot_vocab":     [(w, round(v, 4)) for w, v in bot_words],
        }

        axes.append(candidate_orth)
        new_axes_meta.append(meta)

        # Update residual
        proj = residual_matrix @ candidate_orth
        residual_matrix  -= np.outer(proj, candidate_orth)
        h_proj = holdout_residual @ candidate_orth
        holdout_residual -= np.outer(h_proj, candidate_orth)

        remaining_var = float(np.sum(residual_matrix ** 2))
        cumulative_variance_removed = (total_original_variance - remaining_var) / total_original_variance
        meta["cumulative_var"] = round(cumulative_variance_removed, 6)

    # ── Serialise new axes only ───────────────────────────────────────────────
    print(f"\nPhase 1b complete: {n_accepted} new axes discovered.", flush=True)
    print(f"Total basis size: {len(axes)}  "
          f"({len(existing_vectors)} existing + {n_accepted} new)")
    print(f"Cumulative variance: {cumulative_variance_removed:.4f}")
    print(f"Total time: {time.time()-t_start:.1f}s")

    output = {
        "total_new_axes":       n_accepted,
        "first_new_index":      len(existing_vectors),
        "cumulative_var_final": round(cumulative_variance_removed, 6),
        "axes":                 new_axes_meta,
        "axis_vectors":         [v.tolist() for v in axes[len(existing_vectors):]],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
