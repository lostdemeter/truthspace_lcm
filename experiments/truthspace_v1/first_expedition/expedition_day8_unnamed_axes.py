#!/usr/bin/env python3
"""
Expedition Day 8 — Naming the Unnamed Axes

Day 5 found that the transformer's routing heads (H6, H10, H16, H22, H23,
H24, H25, H27) each peak at a different "unknown" IRD axis:
  H6  → Ax307   H10 → Ax168   H16 → Ax9
  H22 → Ax375   H23 → Ax236   H24 → Ax374
  H25 → Ax110   H27 → Ax171

None of these are the Day 4 semantic Killing vectors (axes 2, 5, 7, 15, 17,
18, 40, 54). So what ARE these axes?

Method: for each axis, retrieve the top-30 and bottom-30 concepts by
projection value. The vocabulary of extremes tells you what the axis captures.

Also: compare the named axes (Day 4, ~8 of them) vs unnamed axes — do the
named ones have more interpretable top/bottom vocabularies?

Also probe the Day 4 axes for comparison, so we can see the contrast between
"these are the grammar axes" (Day 4) and "these are the routing head axes" (Day 8).

Additional: cluster the top/bottom words by semantic field (manual inspection
of the word list). Identify which semantic dimension each axis captures.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

# Routing head peak axes (from Day 5)
ROUTING_PEAK_AXES = {
    6:  307,
    10: 168,
    16: 9,
    22: 375,
    23: 236,
    24: 374,
    25: 110,
    27: 171,
}

# Day 4 named axes for comparison
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

TOP_N = 30   # number of top/bottom concepts to show per axis


def describe_axis(lcm, axis_idx, top_n=TOP_N):
    """
    Return (top_words, bottom_words, axis_name) for a given axis.
    top_words = concepts with highest positive projection on this axis
    bottom_words = concepts with most negative projection on this axis
    """
    projections = lcm.projections[:, axis_idx]
    name = lcm.axis_names[axis_idx] if axis_idx < len(lcm.axis_names) else f"axis_{axis_idx}"
    top_idx    = np.argsort(projections)[-top_n:][::-1]
    bottom_idx = np.argsort(projections)[:top_n]
    top_words    = [(lcm.words[i], float(projections[i])) for i in top_idx]
    bottom_words = [(lcm.words[i], float(projections[i])) for i in bottom_idx]
    return top_words, bottom_words, name


def print_axis(axis_idx, top_words, bottom_words, name, tag=""):
    print(f"\n  ── Axis {axis_idx}  ({tag}name={name[:50]}) ──────────────────────────")
    top_str    = "  ".join(f"{w}({v:+.3f})" for w, v in top_words[:15])
    bottom_str = "  ".join(f"{w}({v:+.3f})" for w, v in bottom_words[:15])
    print(f"  POSITIVE (+): {top_str}")
    print(f"  NEGATIVE (−): {bottom_str}")


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float64)

    print(f"\n{'='*70}")
    print(f"DAY 8 — Naming the Unnamed Axes")
    print(f"{'='*70}")

    # ── Section 1: Routing head peak axes ─────────────────────────────────────
    print(f"\n── Section 1: Routing head peak axes ───────────────────────")
    print(f"  (What semantic dimension does each routing head care about?)\n")

    routing_axis_descriptions = {}
    for head_idx, axis_idx in sorted(ROUTING_PEAK_AXES.items()):
        top, bot, name = describe_axis(lcm, axis_idx)
        routing_axis_descriptions[axis_idx] = (top, bot, name)
        print_axis(axis_idx, top, bot, name, tag=f"H{head_idx} peak, ")

    # ── Section 2: Day 4 named axes for comparison ─────────────────────────────
    print(f"\n\n── Section 2: Day 4 Killing vector axes (named) ────────────")
    print(f"  (For comparison — these are the grammar/relationship axes)\n")

    for axis_idx, label in sorted(DAY4_AXES.items()):
        top, bot, name = describe_axis(lcm, axis_idx)
        print_axis(axis_idx, top, bot, name, tag=f"Day4={label}, ")

    # ── Section 3: Statistical comparison ─────────────────────────────────────
    print(f"\n\n── Section 3: Axis statistics ──────────────────────────────")
    print(f"  {'Axis':<8}  {'Tag':<32}  {'mean':<8}  {'std':<8}  {'max_abs':<10}  top_word")
    print("  " + "─" * 85)

    all_axes = ([(ax, f"H{h}-peak") for h, ax in sorted(ROUTING_PEAK_AXES.items())] +
                [(ax, f"Day4-{label[:16]}") for ax, label in sorted(DAY4_AXES.items())])
    for axis_idx, tag in all_axes:
        col = P[:, axis_idx]
        top1_idx = int(np.argmax(col))
        print(f"  {axis_idx:<8}  {tag:<32}  {col.mean():+.4f}    {col.std():.4f}    "
              f"{np.abs(col).max():.4f}      {lcm.words[top1_idx]}")

    # ── Section 4: Axis-axis cosine similarity (do routing axes correlate?) ───
    print(f"\n\n── Section 4: Cross-axis cosine similarity ──────────────────")
    print(f"  (Are the routing head axes orthogonal to each other and to Day4?)\n")
    routing_axes_list = [ROUTING_PEAK_AXES[h] for h in sorted(ROUTING_PEAK_AXES.keys())]
    all_axes_list     = routing_axes_list + list(sorted(DAY4_AXES.keys()))
    all_ax_labels     = [f"H{h}(Ax{ROUTING_PEAK_AXES[h]})" for h in sorted(ROUTING_PEAK_AXES.keys())] + \
                        [f"D4(Ax{a})" for a in sorted(DAY4_AXES.keys())]

    # Axis "direction vectors" in concept space = columns of projection matrix
    # cos similarity between axis i and j = correlation of their projection columns
    ax_cols = np.array([P[:, a] for a in all_axes_list])   # (n_axes, N_concepts)
    # Normalise
    norms   = np.linalg.norm(ax_cols, axis=1, keepdims=True)
    ax_cols_n = ax_cols / (norms + 1e-20)
    cos_matrix = ax_cols_n @ ax_cols_n.T   # (n_axes, n_axes)

    print(f"  {'':<16}" + "  ".join(f"{l[:12]:<12}" for l in all_ax_labels))
    for i, l1 in enumerate(all_ax_labels):
        row = "  ".join(f"{cos_matrix[i,j]:+.3f}       " for j in range(len(all_ax_labels)))
        print(f"  {l1:<16}  {row}")

    # ── Section 5: Top concept overlap ────────────────────────────────────────
    print(f"\n\n── Section 5: Top concept set overlap ───────────────────────")
    print(f"  (Do routing axes and Day 4 axes select the same concepts?)\n")

    def top_set(axis_idx, n=50):
        return set(lcm.words[i] for i in np.argsort(P[:, axis_idx])[-n:])

    routing_tops = {ax: top_set(ax) for ax in routing_axes_list}
    day4_tops    = {ax: top_set(ax) for ax in DAY4_AXES.keys()}

    print("  Routing × Day4 overlap (|top-50 intersection| / 50):")
    for r_ax in routing_axes_list:
        row = []
        for d_ax in sorted(DAY4_AXES.keys()):
            overlap = len(routing_tops[r_ax] & day4_tops[d_ax])
            row.append(f"{overlap:2d}")
        head = [h for h, a in ROUTING_PEAK_AXES.items() if a == r_ax][0]
        print(f"  H{head}(Ax{r_ax:<4}): " + "  ".join(row) + f"   (vs Day4: Ax{list(DAY4_AXES.keys())})")

    print("\n  Routing × Routing overlap:")
    for r_ax1 in routing_axes_list:
        row = []
        for r_ax2 in routing_axes_list:
            row.append(f"{len(routing_tops[r_ax1] & routing_tops[r_ax2]):2d}")
        head = [h for h, a in ROUTING_PEAK_AXES.items() if a == r_ax1][0]
        print(f"  H{head}(Ax{r_ax1:<4}): " + "  ".join(row))

    # ── Section 6: Manual semantic field annotation ───────────────────────────
    print(f"\n\n── Section 6: Proposed axis semantic interpretation ─────────")
    print(f"  (Based on the top/bottom vocabulary observed above)")
    print()

    for axis_idx, (top, bot, name) in sorted(routing_axis_descriptions.items()):
        head = [h for h, a in ROUTING_PEAK_AXES.items() if a == axis_idx][0]
        top_words_str  = ", ".join(w for w, _ in top[:10])
        bot_words_str  = ", ".join(w for w, _ in bot[:10])
        print(f"  Ax{axis_idx} (H{head} peak):")
        print(f"    + {top_words_str}")
        print(f"    - {bot_words_str}")
        print()
