#!/usr/bin/env python3
"""
Expedition Day 1 — The Mass Spectrum

Compute gravitational mass for all 25,671 IRD concepts under three definitions:

  M_binding  : mean cosine similarity to k=20 nearest neighbours (excluding self)
                → how tightly does a concept bind to its local cluster?
  M_global   : mean cosine similarity to a random sample of 500 concepts
                → how "globally central" is this concept?
  M_focus    : 1.0 / std(cosine to k=20 NN)
                → how focused (vs diffuse) is the local neighbourhood?
                  high focus = all neighbours are in the same semantic region

Three populations predicted:
  Stars   : high M_binding, high M_focus  — specific semantic anchors (bake, guitar, Paris)
  Black holes : high M_global, low M_focus — function words (the, a, is, and)
  Rogue planets : low M_binding, low M_global — rare isolated concepts

Also: sketch the compression landscape (SVD of projection matrix) to preview Day 2.

Usage:
    python expedition_day1_mass_spectrum.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

K_NN       = 20       # nearest neighbours for binding mass
N_SAMPLE   = 600      # random concepts for global mass estimate
BATCH_SIZE = 512      # batch size for pairwise cosine computation
TOP_N      = 30       # print this many in each ranking

rng = np.random.default_rng(42)


def batch_knn_cosines(P, k, batch_size=BATCH_SIZE):
    """
    For each row in P (n_concepts × d), return the cosine similarities to its
    k nearest neighbours (excluding self).
    Returns: (n_concepts, k) array of cosine similarities.
    """
    n = P.shape[0]
    # P is already unit-normed (IRD projections are normalised)
    top_cos = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = P[start:end]           # (batch, d)
        sims  = block @ P.T            # (batch, n) — full cosine matrix for this block
        # Exclude self by zeroing diagonal entries
        for local_i in range(end - start):
            global_i = start + local_i
            sims[local_i, global_i] = -2.0   # guaranteed not to be in top-k
        # Top-k for each row
        idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        for local_i in range(end - start):
            top_cos[start + local_i] = sims[local_i, idx[local_i]]
    return top_cos


def compute_global_mass(P, sample_idx):
    """
    Mean cosine similarity of each concept to a fixed random sample.
    Returns: (n_concepts,) array.
    """
    P_sample = P[sample_idx]    # (n_sample, d)
    sims = P @ P_sample.T       # (n_concepts, n_sample)
    return sims.mean(axis=1)


def ascii_histogram(values, bins=20, width=50):
    counts, edges = np.histogram(values, bins=bins)
    max_count = counts.max()
    lines = []
    for i, c in enumerate(counts):
        bar = '█' * int(c / max_count * width)
        lines.append(f"  [{edges[i]:+.3f}, {edges[i+1]:+.3f})  {bar} {c}")
    return '\n'.join(lines)


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float32)   # (25671, 500), unit-normed rows
    n   = P.shape[0]
    words = lcm.words

    # ── Mass computation ──────────────────────────────────────────────────────

    print(f"\nComputing k-NN cosines (k={K_NN}, n={n})...")
    top_cos = batch_knn_cosines(P, K_NN)        # (n, K_NN)

    M_binding  = top_cos.mean(axis=1)           # (n,)
    M_focus    = 1.0 / (top_cos.std(axis=1) + 1e-8)   # (n,) — higher = tighter cluster

    print(f"Computing global mass (sample={N_SAMPLE})...")
    sample_idx = rng.choice(n, N_SAMPLE, replace=False)
    M_global   = compute_global_mass(P, sample_idx)    # (n,)

    # ── Rankings ─────────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print("DAY 1 OBSERVATION LOG — The Mass Spectrum")
    print(f"{'='*65}")

    print(f"\n  Corpus: {n} concepts × {P.shape[1]} axes")
    print(f"\n  M_binding  mean={M_binding.mean():.4f}  std={M_binding.std():.4f}  "
          f"min={M_binding.min():.4f}  max={M_binding.max():.4f}")
    print(f"  M_global   mean={M_global.mean():.4f}  std={M_global.std():.4f}  "
          f"min={M_global.min():.4f}  max={M_global.max():.4f}")
    print(f"  M_focus    mean={M_focus.mean():.2f}   std={M_focus.std():.2f}  "
          f"min={M_focus.min():.2f}  max={M_focus.max():.2f}")

    # Stars: highest M_binding
    print(f"\n── STARS (highest M_binding — tightest semantic clusters) ──────")
    order = np.argsort(M_binding)[::-1]
    print(f"  {'Word':<22s}  M_bind  M_glob  M_focus")
    for i in order[:TOP_N]:
        print(f"  {words[i]:<22s}  {M_binding[i]:.4f}  "
              f"{M_global[i]:.4f}  {M_focus[i]:.1f}")

    # Black holes: highest M_global, lowest M_focus
    print(f"\n── BLACK HOLES (highest M_global — near everything) ─────────")
    order_g = np.argsort(M_global)[::-1]
    print(f"  {'Word':<22s}  M_bind  M_glob  M_focus")
    for i in order_g[:TOP_N]:
        print(f"  {words[i]:<22s}  {M_binding[i]:.4f}  "
              f"{M_global[i]:.4f}  {M_focus[i]:.1f}")

    # Rogue planets: lowest M_binding AND lowest M_global
    combined_isolation = M_binding + M_global
    order_iso = np.argsort(combined_isolation)
    print(f"\n── ROGUE PLANETS (lowest M_binding + M_global — isolated) ─────")
    print(f"  {'Word':<22s}  M_bind  M_glob  M_focus")
    for i in order_iso[:TOP_N]:
        print(f"  {words[i]:<22s}  {M_binding[i]:.4f}  "
              f"{M_global[i]:.4f}  {M_focus[i]:.1f}")

    # Distribution of M_binding
    print(f"\n── M_binding distribution ──────────────────────────────────────")
    print(ascii_histogram(M_binding))

    # ── Test specific words ───────────────────────────────────────────────────
    probe_words = [
        'the', 'a', 'is', 'and', 'of', 'to', 'in',   # predicted black holes
        'bake', 'guitar', 'recipe', 'flour', 'butter', # predicted stars (culinary/music)
        'paris', 'berlin', 'london',                   # geographic stars
        'bank', 'cookie', 'python', 'bass',            # predicted polysemous (high inertia)
        'algorithm', 'quark', 'mitochondria',          # predicted rogue planets (rare technical)
    ]
    print(f"\n── Probe words ─────────────────────────────────────────────────")
    print(f"  {'Word':<22s}  M_bind  M_glob  M_focus  rank_bind  rank_glob")
    bind_rank  = n - np.argsort(np.argsort(M_binding))    # rank 1 = highest
    glob_rank  = n - np.argsort(np.argsort(M_global))
    for w in probe_words:
        wl = w.lower()
        if wl in lcm.word_set:
            i = lcm.word_set[wl]
            print(f"  {w:<22s}  {M_binding[i]:.4f}  {M_global[i]:.4f}  "
                  f"{M_focus[i]:.1f}   {bind_rank[i]:<10d}  {glob_rank[i]}")
        else:
            print(f"  {w:<22s}  (not in vocabulary)")

    # ── Mass distribution — does it follow a power law? ───────────────────────
    print(f"\n── Power-law test on M_binding ─────────────────────────────────")
    sorted_mass = np.sort(M_binding)[::-1]
    ranks       = np.arange(1, n + 1)
    # Fit log(mass) ~ -alpha * log(rank) for a power law
    log_r = np.log(ranks)
    log_m = np.log(np.clip(sorted_mass, 1e-8, None))
    # Linear regression on log-log
    A = np.vstack([log_r, np.ones_like(log_r)]).T
    alpha, c = np.linalg.lstsq(A, log_m, rcond=None)[0]
    residuals = log_m - (alpha * log_r + c)
    r2 = 1 - residuals.var() / log_m.var()
    print(f"  Log-log fit: mass ~ rank^({alpha:.3f})  R²={r2:.4f}")
    if r2 > 0.9:
        print(f"  → POWER LAW confirmed (R²>{0.9:.1f})")
    elif r2 > 0.7:
        print(f"  → Approximate power law (R²>{0.7:.1f})")
    else:
        print(f"  → NOT a power law (R²={r2:.4f})")

    # Print a few rank samples
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int(n * pct / 100)
        print(f"  Percentile {pct:3d}%: M_binding = {sorted_mass[min(idx, n-1)]:.4f}  "
              f"(word: {words[np.argsort(M_binding)[::-1][min(idx, n-1)]]!r})")

    # ── Compression preview — SVD of projection matrix ────────────────────────
    print(f"\n── Compression preview — SVD energy ────────────────────────────")
    print(f"  Running SVD on ({n}, {P.shape[1]}) projection matrix...")
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    total_energy = (S ** 2).sum()
    print(f"  Total singular value energy: {total_energy:.2f}")
    print(f"  {'k components':<20s}  {'energy captured':<20s}  compression_ratio")
    thresholds = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    cumulative = np.cumsum(S ** 2) / total_energy
    for t in thresholds:
        k_needed = int(np.searchsorted(cumulative, t)) + 1
        ratio    = P.shape[1] / k_needed
        print(f"  k={k_needed:<17d}  {t*100:.1f}%{'':>15s}  {ratio:.1f}×")

    print(f"\n  Top-20 singular values (energy share):")
    top20_energy = (S[:20] ** 2) / total_energy
    cumul = 0.0
    for i in range(20):
        cumul += top20_energy[i]
        bar = '█' * int(top20_energy[i] * 500)
        print(f"  σ_{i+1:02d}: {S[i]:.2f}  ({top20_energy[i]*100:.2f}%  cumul={cumul*100:.1f}%)  {bar}")

    # ── Save mass arrays for future days ─────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'expedition_day1_masses.npz')
    np.savez(out_path,
             words=np.array(words),
             M_binding=M_binding,
             M_global=M_global,
             M_focus=M_focus,
             singular_values=S)
    print(f"\n  Saved: {out_path}")
    print(f"  (contains M_binding, M_global, M_focus, singular_values for all {n} concepts)")
