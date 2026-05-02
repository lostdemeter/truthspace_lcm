#!/usr/bin/env python3
"""
Expedition Day 5 — Head-Axis Correspondence

Prediction from Day 4: if the IRD axes are the Killing vectors of the semantic
manifold, and trained transformers discover the same geometry through SGD, then
specific attention heads at Layer 23 should align with specific IRD axes.

Finding 38 established:
  - 28 total heads at Layer 23
  - 8 routing heads (6, 10, 16, 22, 23, 24, 25, 27) vary target per prompt
  - 20 fixed heads always attend to BOS (position 0)
  - Head 6 alone solves 6/6 factual probes, largest projection norm (11.61)

Prediction: Head 6's Q and K matrices are primarily sensitive to one of the
relationship-encoding IRD axes (2=geographic, 5=gender, 7=hypernym,
15=comparative, 17=verb-agent, 18=plural, 40=tense, 54=temperature).

Method:
  For each head h (0-27), extract W_q_head (128×3584) and W_k_kv (128×3584).
  For each IRD axis a_v (3584-dim unit vector), compute:
    response_Q(h, a) = ||W_q_head @ a_v||^2   [how much Q amplifies axis direction]
    response_K(h, a) = ||W_k_kv  @ a_v||^2   [how much K amplifies axis direction]
    combined(h, a)   = sqrt(response_Q * response_K)  [geometric mean]

  Normalise per head by the mean response (baseline = uniform random axis).
  High normalised response = head is specifically tuned to that axis direction.

All data loaded from pre-saved navigation_model/ directory (no Qwen2 model needed).
"""

import sys, os
import numpy as np

NAV_DIR = "/home/thorin/truthspace-lcm/navigation_model"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

# Relationship axes discovered in Day 4
DAY4_AXES = {
    2:  "geographic (capital/language)",
    5:  "gender",
    7:  "hypernym (is-a)",
    15: "comparative (degree)",
    17: "verb-to-agent",
    18: "plural (number)",
    40: "tense",
    54: "antonym-temperature",
}

# Qwen2-7B head structure (from navigation analysis)
N_HEADS    = 28
N_KV_HEADS = 4
HEAD_DIM   = 128   # 3584 / 28
LAYER      = 23

# The 8 routing heads identified in Finding 38
ROUTING_HEADS = {6, 10, 16, 22, 23, 24, 25, 27}


def load_layer_weights(layer_idx):
    """Load W_q, W_k from pre-saved navigation_model."""
    W_q = np.load(os.path.join(NAV_DIR, f"layer_{layer_idx}_W_q.npy")).astype(np.float64)
    W_k = np.load(os.path.join(NAV_DIR, f"layer_{layer_idx}_W_k.npy")).astype(np.float64)
    return W_q, W_k


def head_axis_response(W_q, W_k, A):
    """
    For all heads × all IRD axes, compute normalised response.

    Args:
        W_q: (3584, 3584) full Q weight matrix
        W_k: (512, 3584)  full K weight matrix (4 KV heads)
        A:   (n_axes, 3584) unit-normalised IRD axis vectors

    Returns:
        resp_q:    (n_heads, n_axes) normalised Q-response
        resp_k:    (n_heads, n_axes) normalised K-response (shared across Q-heads)
        resp_comb: (n_heads, n_axes) geometric mean
    """
    n_axes = A.shape[0]
    heads_per_kv = N_HEADS // N_KV_HEADS   # 7

    resp_q    = np.zeros((N_HEADS, n_axes))
    resp_k    = np.zeros((N_HEADS, n_axes))
    resp_comb = np.zeros((N_HEADS, n_axes))

    for h in range(N_HEADS):
        q_s = h * HEAD_DIM
        q_e = (h + 1) * HEAD_DIM
        W_q_h = W_q[q_s:q_e, :]            # (128, 3584)

        kv_h  = h // heads_per_kv
        k_s   = kv_h * HEAD_DIM
        k_e   = (kv_h + 1) * HEAD_DIM
        W_k_h = W_k[k_s:k_e, :]            # (128, 3584)

        # For each axis a_v, response = ||W @ a_v||^2
        # = sum_i (W[i,:] @ a_v)^2 = ||A @ W.T||_rows^2
        Aq = A @ W_q_h.T    # (n_axes, 128)
        Ak = A @ W_k_h.T    # (n_axes, 128)

        rq = (Aq ** 2).sum(axis=1)   # (n_axes,)
        rk = (Ak ** 2).sum(axis=1)   # (n_axes,)

        # Normalise by mean (baseline = all axes equally)
        rq_n = rq / (rq.mean() + 1e-20)
        rk_n = rk / (rk.mean() + 1e-20)

        resp_q[h]    = rq_n
        resp_k[h]    = rk_n
        resp_comb[h] = np.sqrt(rq_n * rk_n)

    return resp_q, resp_k, resp_comb


if __name__ == '__main__':
    print("Loading IRD axis vectors...")
    lcm = build_lcm()
    A   = lcm.axis_vectors.astype(np.float64)   # (500, 3584)
    n_axes, D = A.shape

    # Unit-normalise (should already be, but be safe)
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / (A_norms + 1e-20)

    print(f"  {n_axes} axes × {D} dims")

    print(f"Loading Layer {LAYER} weights from navigation_model/...")
    W_q, W_k = load_layer_weights(LAYER)
    print(f"  W_q: {W_q.shape}, W_k: {W_k.shape}")

    print("\nComputing head-axis response matrix...")
    resp_q, resp_k, resp_comb = head_axis_response(W_q, W_k, A)

    print(f"\n{'='*70}")
    print(f"DAY 5 — Head-Axis Correspondence (Layer {LAYER})")
    print(f"{'='*70}")

    # ── Head 6 (key routing head) detail ──────────────────────────────────────
    print(f"\n── Head 6 (dominant routing head, Finding 38) ──────────────")
    h6_comb = resp_comb[6]
    top10   = np.argsort(h6_comb)[-10:][::-1]
    print(f"  Top-10 IRD axes by combined Q×K response:")
    print(f"  {'Rank':<5}  {'Axis':<7}  {'Resp (×mean)':<14}  {'Day4 name'}")
    for rank, ax in enumerate(top10):
        name  = DAY4_AXES.get(ax, "—")
        print(f"  {rank+1:<5}  {ax:<7}  {h6_comb[ax]:.3f}          {name}")

    print(f"\n  ── Day 4 relationship axes specifically ──")
    print(f"  {'Axis':<7}  {'Day4 name':<35}  {'Q-resp':<10}  {'K-resp':<10}  {'Combined':<10}  Global rank")
    for ax, name in sorted(DAY4_AXES.items()):
        global_rank = int((h6_comb > h6_comb[ax]).sum()) + 1
        print(f"  {ax:<7}  {name:<35}  {resp_q[6, ax]:.3f}       {resp_k[6, ax]:.3f}       "
              f"{h6_comb[ax]:.3f}       {global_rank}/{n_axes}")

    # ── All routing heads ─────────────────────────────────────────────────────
    print(f"\n── All routing heads — top-3 IRD axes by combined response ──")
    print(f"  {'Head':<6}  {'#1 axis (resp)':<20}  {'#2 axis (resp)':<20}  {'#3 axis (resp)':<20}")
    for h in sorted(ROUTING_HEADS):
        top3  = np.argsort(resp_comb[h])[-3:][::-1]
        parts = []
        for ax in top3:
            n = DAY4_AXES.get(ax, "ax"+str(ax))
            parts.append(f"Ax{ax}({n[:12]}) {resp_comb[h, ax]:.2f}×")
        print(f"  H{h:<5}  {parts[0]:<20}  {parts[1]:<20}  {parts[2]:<20}")

    # ── All fixed heads for comparison ───────────────────────────────────────
    fixed_heads = sorted(set(range(N_HEADS)) - ROUTING_HEADS)
    print(f"\n── Fixed heads (always attend BOS) — top IRD axis ──────────")
    print(f"  {'Head':<6}  {'Top axis':<12}  {'Resp':<8}  Day4 name")
    for h in fixed_heads:
        top1 = int(np.argmax(resp_comb[h]))
        print(f"  H{h:<5}  Ax{top1:<9}  {resp_comb[h, top1]:.3f}    {DAY4_AXES.get(top1, '—')}")

    # ── Specialisation heatmap (routing heads × Day4 axes) ───────────────────
    print(f"\n── Specialisation heatmap: routing heads × Day 4 relationship axes ──")
    ax_list = sorted(DAY4_AXES.keys())
    print(f"  {'Head':<6}  " + "  ".join(f"Ax{a:<3}" for a in ax_list))
    print(f"  {'':6}  " + "  ".join(f"({DAY4_AXES[a][:4]})" for a in ax_list))
    for h in sorted(ROUTING_HEADS):
        row = [resp_comb[h, a] for a in ax_list]
        bars = ["████" if r > 3 else ("▓▓▓▓" if r > 2 else ("▒▒▒▒" if r > 1.5 else "░░░░"))
                for r in row]
        nums = " ".join(f"{r:.2f}" for r in row)
        print(f"  H{h:<5}  {nums}")

    # ── Is there a head-to-axis 1-1 mapping? ─────────────────────────────────
    print(f"\n── Head specialisation summary ─────────────────────────────")
    routing_resp = resp_comb[sorted(ROUTING_HEADS), :]  # (8, 500)
    for h in sorted(ROUTING_HEADS):
        top1 = int(np.argmax(resp_comb[h]))
        top1_resp = resp_comb[h, top1]
        # What fraction of total response is in the top axis?
        concentration = top1_resp / resp_comb[h].sum() * n_axes
        print(f"  H{h}: top axis={top1} ({DAY4_AXES.get(top1, 'unknown'):<30s}) "
              f"resp={top1_resp:.2f}×  concentration={concentration:.1f}×")

    # ── Cross-head diversity: do routing heads differ from each other? ────────
    print(f"\n── Cross-head agreement on Day 4 axes ──────────────────────")
    print(f"  (Are routing heads redundant or specialised?)")
    for a in ax_list:
        vals = [resp_comb[h, a] for h in sorted(ROUTING_HEADS)]
        print(f"  Ax{a:<3} ({DAY4_AXES[a]:<30s}): "
              f"max={max(vals):.2f}  min={min(vals):.2f}  "
              f"spread={max(vals)-min(vals):.2f}  "
              f"which heads high: {[h for h in sorted(ROUTING_HEADS) if resp_comb[h,a]>1.5]}")

    # ── Baseline: are IRD axes overall more aligned with routing vs fixed? ────
    print(f"\n── Routing vs Fixed heads: mean response to Day 4 axes ─────")
    routing_mean = np.mean([resp_comb[h, ax_list].mean() for h in ROUTING_HEADS])
    fixed_mean   = np.mean([resp_comb[h, ax_list].mean() for h in fixed_heads])
    all_mean     = np.mean([resp_comb[h, ax_list].mean() for h in range(N_HEADS)])
    print(f"  Routing heads mean response to Day4 axes: {routing_mean:.3f}×")
    print(f"  Fixed heads mean response to Day4 axes:   {fixed_mean:.3f}×")
    print(f"  All heads mean response to Day4 axes:     {all_mean:.3f}×")
    if routing_mean > fixed_mean * 1.2:
        print(f"  VERDICT: Routing heads MORE tuned to relationship axes than fixed heads ✓")
    elif routing_mean < fixed_mean * 0.8:
        print(f"  VERDICT: Fixed heads MORE tuned to relationship axes (unexpected)")
    else:
        print(f"  VERDICT: No significant difference between routing and fixed heads")
