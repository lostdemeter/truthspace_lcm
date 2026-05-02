"""
DC299 Phase 1 — Visualization Dashboard

Reads dc299_phase1_axes.json and produces a 6-panel figure:
  1. Cumulative variance explained (with 95% target line)
  2. Step variance per axis  (contribution decay)
  3. Binary accuracy scatter
  4. Gap value decay
  5. Semantic quality score (fraction of top/bot vocab that are clean English words)
  6. Semantic quality histogram (distribution across axes)

Also writes dc299_phase1_viz_summary.md with key numbers.

Usage:
    python dc299_phase1_visualize.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # headless — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
INPUT_JSON  = SCRIPT_DIR / "dc299_phase1_axes.json"
OUTPUT_PNG  = SCRIPT_DIR / "dc299_phase1_viz.png"
SUMMARY_MD  = SCRIPT_DIR / "dc299_phase1_viz_summary.md"

assert INPUT_JSON.exists(), f"Missing {INPUT_JSON} — run dc299_phase1_ird_axis_discovery.py first."


# ── Semantic quality scoring ───────────────────────────────────────────────────
_ASCII_ALPHA = re.compile(r'^[A-Za-z]{3,}$')

def _token_is_clean(tok: str) -> bool:
    """True if token looks like a real English word (ASCII alpha, length ≥ 3)."""
    tok = tok.strip()
    return bool(_ASCII_ALPHA.match(tok))

def semantic_quality(top_vocab, bot_vocab) -> float:
    """Fraction of top+bot tokens that are clean English words."""
    tokens = [w for w, _ in top_vocab] + [w for w, _ in bot_vocab]
    if not tokens:
        return 0.0
    clean = sum(1 for t in tokens if _token_is_clean(t))
    return clean / len(tokens)


# ── Load data ─────────────────────────────────────────────────────────────────
with open(INPUT_JSON) as f:
    data = json.load(f)

axes_list = data["axes"]

# Extract per-axis series (only discovered axes, not seeds, for trend plots;
# keep index relative to full basis for x-axis)
discovered = [a for a in axes_list if a["type"] == "discovered"]
seeds      = [a for a in axes_list if a["type"] == "seed"]

def get(ax_list, key):
    return [a.get(key, 0) for a in ax_list]

d_idx      = get(discovered, "index")
d_cumvar   = get(discovered, "cumulative_var")
d_stepvar  = get(discovered, "step_var")
d_acc      = get(discovered, "binary_acc")
d_gap      = get(discovered, "gap")
d_quality  = [semantic_quality(a.get("top_vocab", []), a.get("bot_vocab", []))
              for a in discovered]

# Full cumulative variance (seeds have no cumulative_var stored — prepend 0)
all_indices  = [0] + d_idx
all_cumvars  = [0.0] + d_cumvar

n_axes = len(axes_list)
n_disc = len(discovered)
n_seed = len(seeds)
max_cumvar = d_cumvar[-1] if d_cumvar else 0.0


# ── Compute key statistics ─────────────────────────────────────────────────────
# Knee detection on cumulative variance: point where second derivative is max
if len(d_cumvar) > 4:
    cv_arr    = np.array(d_cumvar)
    d1        = np.gradient(cv_arr)
    d2        = np.gradient(d1)
    knee_pos  = int(np.argmax(d2))                 # steepest acceleration → knee
else:
    knee_pos = 0

# Threshold crossings
def first_crossing(arr, threshold):
    for i, v in enumerate(arr):
        if v >= threshold:
            return i
    return None

cross_50  = first_crossing(d_cumvar, 0.50)
cross_75  = first_crossing(d_cumvar, 0.75)
cross_90  = first_crossing(d_cumvar, 0.90)
cross_95  = first_crossing(d_cumvar, 0.95)

# Semantic quality: fraction with quality ≥ 0.5
high_q = sum(1 for q in d_quality if q >= 0.5)
high_q_pct = 100 * high_q / len(d_quality) if d_quality else 0

# Quality "cliff" — first axis where quality drops permanently below 0.5
quality_cliff = None
window = 20
for i in range(len(d_quality) - window):
    if np.mean(d_quality[i:i+window]) < 0.4:
        quality_cliff = d_idx[i]
        break

# Axes still accepted at end
last_acc = d_acc[-1] if d_acc else 0
last_gap = d_gap[-1] if d_gap else 0


# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f0f1a")
ax_color = "#0f0f1a"
text_color = "#e0e0f0"
grid_color = "#2a2a3a"
accent1 = "#4fc3f7"    # cyan
accent2 = "#81c784"    # green
accent3 = "#ffb74d"    # amber
accent4 = "#ef9a9a"    # red/pink
accent5 = "#ce93d8"    # purple

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

def styled_ax(ax):
    ax.set_facecolor(ax_color)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    ax.grid(color=grid_color, linewidth=0.5, alpha=0.7)
    return ax


# ── Panel 1: Cumulative variance explained ────────────────────────────────────
ax1 = styled_ax(fig.add_subplot(gs[0, 0]))
ax1.plot(d_idx, d_cumvar, color=accent1, linewidth=1.8, label="Cumulative var")
ax1.axhline(0.95, color=accent3, linewidth=1.2, linestyle="--", label="95% target")
ax1.axhline(0.75, color=accent2, linewidth=0.8, linestyle=":", alpha=0.7, label="75%")
ax1.axhline(0.50, color=accent5, linewidth=0.8, linestyle=":", alpha=0.7, label="50%")
if cross_95 is not None:
    ax1.axvline(d_idx[cross_95], color=accent3, linewidth=1, linestyle="--", alpha=0.6)
    ax1.annotate(f"95% @ axis {d_idx[cross_95]}",
                 xy=(d_idx[cross_95], 0.95), xytext=(d_idx[cross_95]+10, 0.88),
                 color=accent3, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=accent3, lw=0.8))
else:
    ax1.annotate(f"95% not reached\n(max={max_cumvar:.1%})",
                 xy=(d_idx[-1]*0.6, 0.95), color=accent3, fontsize=8, ha="center")
ax1.set_xlabel("Axis index")
ax1.set_ylabel("Cumulative variance")
ax1.set_title("Cumulative Variance Explained", fontweight="bold")
ax1.set_ylim(0, 1.02)
ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=text_color, framealpha=0.8)


# ── Panel 2: Step variance per axis ───────────────────────────────────────────
ax2 = styled_ax(fig.add_subplot(gs[0, 1]))
ax2.scatter(d_idx, d_stepvar, color=accent2, s=2, alpha=0.6)
# Rolling mean
if len(d_stepvar) > 20:
    kernel = np.ones(20) / 20
    rolling = np.convolve(d_stepvar, kernel, mode="valid")
    roll_x  = d_idx[10: 10 + len(rolling)]
    ax2.plot(roll_x, rolling, color=accent3, linewidth=1.5, label="20-axis rolling mean")
ax2.axhline(0.001, color=accent4, linewidth=1, linestyle="--", alpha=0.7,
            label="MIN_VARIANCE_STEP=0.001")
ax2.set_xlabel("Axis index")
ax2.set_ylabel("Step variance")
ax2.set_title("Variance Contribution Per Axis", fontweight="bold")
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=text_color, framealpha=0.8)


# ── Panel 3: Binary accuracy ──────────────────────────────────────────────────
ax3 = styled_ax(fig.add_subplot(gs[1, 0]))
ax3.scatter(d_idx, d_acc, color=accent5, s=2, alpha=0.5)
if len(d_acc) > 20:
    kernel = np.ones(20) / 20
    roll_acc = np.convolve(d_acc, kernel, mode="valid")
    roll_x2  = d_idx[10: 10 + len(roll_acc)]
    ax3.plot(roll_x2, roll_acc, color=accent1, linewidth=1.5, label="20-axis rolling mean")
ax3.axhline(0.75, color=accent4, linewidth=1, linestyle="--", alpha=0.8,
            label="MIN_BINARY_ACC=0.75")
ax3.set_xlabel("Axis index")
ax3.set_ylabel("Holdout binary accuracy")
ax3.set_title("Holdout Binary Separation Accuracy", fontweight="bold")
ax3.set_ylim(0.70, 1.02)
ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=text_color, framealpha=0.8)


# ── Panel 4: Gap decay ────────────────────────────────────────────────────────
ax4 = styled_ax(fig.add_subplot(gs[1, 1]))
ax4.scatter(d_idx, d_gap, color=accent3, s=2, alpha=0.5)
if len(d_gap) > 20:
    kernel = np.ones(20) / 20
    roll_gap = np.convolve(d_gap, kernel, mode="valid")
    roll_x3  = d_idx[10: 10 + len(roll_gap)]
    ax4.plot(roll_x3, roll_gap, color=accent2, linewidth=1.5, label="20-axis rolling mean")
ax4.set_xlabel("Axis index")
ax4.set_ylabel("Separation gap (pos_mean − neg_mean)")
ax4.set_title("Axis Discrimination Gap Decay", fontweight="bold")
ax4.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=text_color, framealpha=0.8)


# ── Panel 5: Semantic quality ─────────────────────────────────────────────────
ax5 = styled_ax(fig.add_subplot(gs[2, 0]))
ax5.scatter(d_idx, d_quality, color=accent1, s=2, alpha=0.5)
if len(d_quality) > 20:
    kernel = np.ones(20) / 20
    roll_q = np.convolve(d_quality, kernel, mode="valid")
    roll_x4 = d_idx[10: 10 + len(roll_q)]
    ax5.plot(roll_x4, roll_q, color=accent5, linewidth=1.8, label="20-axis rolling mean")
ax5.axhline(0.5, color=accent4, linewidth=1, linestyle="--", alpha=0.8,
            label="Quality threshold 0.5")
if quality_cliff is not None:
    ax5.axvline(quality_cliff, color=accent3, linewidth=1.2, linestyle="--",
                label=f"Quality cliff ~axis {quality_cliff}")
ax5.set_xlabel("Axis index")
ax5.set_ylabel("Semantic quality (frac. clean English tokens)")
ax5.set_title("Semantic Quality Score Per Axis", fontweight="bold")
ax5.set_ylim(-0.05, 1.05)
ax5.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=text_color, framealpha=0.8)


# ── Panel 6: Quality histogram ────────────────────────────────────────────────
ax6 = styled_ax(fig.add_subplot(gs[2, 1]))
bins = np.linspace(0, 1, 21)
counts, edges = np.histogram(d_quality, bins=bins)
bar_colors = [accent2 if (e + bins[1]/2) >= 0.5 else accent4
              for e in edges[:-1]]
ax6.bar(edges[:-1], counts, width=(edges[1]-edges[0])*0.9,
        color=bar_colors, align="edge", alpha=0.85)
ax6.axvline(0.5, color=text_color, linewidth=1, linestyle="--", alpha=0.7)
ax6.set_xlabel("Semantic quality score")
ax6.set_ylabel("Count of axes")
ax6.set_title("Semantic Quality Distribution", fontweight="bold")

# Annotate counts
ax6.annotate(f"Semantic: {high_q} ({high_q_pct:.0f}%)",
             xy=(0.72, 0.88), xycoords="axes fraction", color=accent2, fontsize=9,
             fontweight="bold")
ax6.annotate(f"Structural: {n_disc-high_q} ({100-high_q_pct:.0f}%)",
             xy=(0.72, 0.78), xycoords="axes fraction", color=accent4, fontsize=9,
             fontweight="bold")


# ── Title ─────────────────────────────────────────────────────────────────────
title_parts = [
    f"DC299 Phase 1 — IRD Axis Discovery",
    f"{n_axes} total axes ({n_seed} seeds + {n_disc} discovered) | "
    f"Max var explained: {max_cumvar:.1%}",
]
if cross_95 is not None:
    title_parts[1] += f" | 95% @ axis {d_idx[cross_95]}"
else:
    title_parts[1] += " | 95% not yet reached"

fig.suptitle("\n".join(title_parts), color=text_color, fontsize=13, fontweight="bold",
             y=0.98)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUTPUT_PNG}")


# ── Summary markdown ──────────────────────────────────────────────────────────
with open(SUMMARY_MD, "w") as f:
    def w(s=""): f.write(s + "\n")

    w("# DC299 Phase 1 — Visualization Summary")
    w()
    w(f"**Input:** `{INPUT_JSON.name}`  ")
    w(f"**Plot:** `{OUTPUT_PNG.name}`")
    w()
    w("## Key Numbers")
    w()
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Total axes | {n_axes} ({n_seed} seeds + {n_disc} discovered) |")
    w(f"| Max cumulative variance | {max_cumvar:.4f} ({max_cumvar:.1%}) |")
    if cross_50 is not None:
        w(f"| Axes to reach 50% variance | {d_idx[cross_50]} |")
    if cross_75 is not None:
        w(f"| Axes to reach 75% variance | {d_idx[cross_75]} |")
    if cross_90 is not None:
        w(f"| Axes to reach 90% variance | {d_idx[cross_90]} |")
    if cross_95 is not None:
        w(f"| Axes to reach 95% variance | {d_idx[cross_95]} |")
    else:
        w(f"| Axes to reach 95% variance | Not reached (need more) |")
    w(f"| Semantic axes (quality ≥ 0.5) | {high_q} ({high_q_pct:.0f}%) |")
    w(f"| Structural axes (quality < 0.5) | {n_disc - high_q} ({100-high_q_pct:.0f}%) |")
    if quality_cliff is not None:
        w(f"| Semantic quality cliff | ~axis {quality_cliff} |")
    w(f"| Last axis binary_acc | {last_acc:.3f} |")
    w(f"| Last axis gap | {last_gap:.4f} |")
    w()
    w("## Variance Extrapolation")
    w()
    if len(d_stepvar) >= 50:
        last_50_mean = np.mean(d_stepvar[-50:])
        remaining = 0.95 - max_cumvar
        if remaining > 0 and last_50_mean > 0:
            est_more = int(np.ceil(remaining / last_50_mean))
            est_total = n_axes + est_more
            w(f"Mean step variance (last 50 axes): {last_50_mean:.6f}")
            w()
            w(f"If decay rate stays constant:")
            w(f"- Need ~{est_more} more axes to reach 95%")
            w(f"- Total axes needed: ~{est_total}")
        else:
            w("Already reached 95% or step_var = 0.")
    w()
    w("## Semantic Quality Summary")
    w()
    w("Quality score = fraction of top-20 + bottom-20 vocab tokens that are")
    w("pure ASCII alphabetic strings of length ≥ 3 (proxy for 'real English word').")
    w()
    if quality_cliff is not None:
        w(f"Rolling mean of quality score drops below 0.4 permanently at ~axis {quality_cliff}.")
        w(f"This is the approximate boundary between semantic and structural axes.")
    else:
        w("Quality cliff not detected within current axis range.")

print(f"Saved: {SUMMARY_MD}")
