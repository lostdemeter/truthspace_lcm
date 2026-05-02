#!/usr/bin/env python3
"""
Expedition Day 52 — Betti Numbers of the Zone C Manifold

Persistent homology on the Zone C concept space to determine whether
its topological invariants (Betti numbers β0, β1, β2) are stable under
vocabulary perturbation — the definitive test for topological quantum character.

If the topology is protected:
  - Betti numbers should replicate across split-half body centroid sets
  - The persistence diagram should be stable (low bottleneck distance)
  - Topologically significant features should have long lifetimes (high persistence)

If not protected:
  - Betti numbers fluctuate across splits
  - Short-lived features dominate (noise-level topology)

Setup:
  Point cloud  = body centroids in φ-space (95 points, ~1536 dims → project to 43 axes)
  Distance     = cosine distance (1 - cos)
  Filtration   = Vietoris-Rips (ripser)
  Homology     = H0, H1, H2
  Stability    = bottleneck distance between persistence diagrams from 10 splits

Three scales:
  S1: 95 body centroids       — coarse topology (inter-body structure)
  S2: all 1647 Zone C words   — fine topology (word-level structure)
  S3: 10 split-half replications of S1 — stability test
"""

import json, random
import numpy as np
from pathlib import Path
from collections import defaultdict

import ripser
from persim import bottleneck, wasserstein
from persim import plot_diagrams  # noqa — not used for display but confirms import

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day52_betti_numbers.json")

N_AXES      = 43
N_SPLITS    = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("  Expedition Day 52 — Betti Numbers of Zone C")
print("=" * 70)


# ── Load + rebuild geometry ───────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words  = [w for w, v in wmap.items()
                 if v['phase'] == 2 and v.get('L14_body') not in ('B000','B001',None)
                 and w in w2i]
zone_c_bodies = {w: wmap[w]['L14_body'] for w in zone_c_words}
body_label_map = {}
for w, v in wmap.items():
    b = v.get('L14_body')
    if b and b not in body_label_map:
        body_label_map[b] = v.get('L14_label', '?')

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

def batch_phi(hs, z2):
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

# Z2
deltas = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        wa, wb = pfx+a, pfx+b
        if wa in w2i and wb in w2i:
            d = hs14_all[w2i[wb]] - hs14_all[w2i[wa]]
            dm = np.linalg.norm(d)
            if dm > 1e-20: deltas.append(d / dm)
            break
_, _, Vt_d = np.linalg.svd(np.stack(deltas), full_matrices=False)
z2 = Vt_d[0] / np.linalg.norm(Vt_d[0])

zone_c_idx = np.array([w2i[w] for w in zone_c_words])
phi_c14    = batch_phi(hs14_all[zone_c_idx], z2)  # (n_zc, D)

# Body centroids
body_members_full = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members_full[zone_c_bodies[w]].append(i)

def build_centroids(body_members, min_members=3):
    centroids = {}
    for body, idxs in body_members.items():
        if len(idxs) < min_members: continue
        vecs = phi_c14[idxs]
        c = vecs.mean(axis=0)
        nm = np.linalg.norm(c)
        if nm > 1e-20:
            centroids[body] = c / nm
    return centroids

full_centroids = build_centroids(body_members_full)

# Axis basis for dimensionality reduction
bodies_list = sorted(full_centroids.keys())
C_full      = np.stack([full_centroids[b] for b in bodies_list])   # (n_bodies, D)
_, _, Vt_c  = np.linalg.svd(C_full, full_matrices=False)
AXES        = Vt_c[:N_AXES]   # (43, D)

def to_axis_coords(centroid_dict):
    """Project body centroids into 43-dim axis space (unit-normalised)."""
    bodies = sorted(centroid_dict.keys())
    M = np.stack([centroid_dict[b] for b in bodies])
    coords = M @ AXES.T   # (n, 43)
    norms  = np.linalg.norm(coords, axis=1, keepdims=True)
    return coords / (norms + 1e-20), bodies

def cosine_dist_matrix(X):
    """Cosine distance matrix from row-normalised X."""
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-20)
    sims = X_n @ X_n.T
    np.clip(sims, -1, 1, out=sims)
    return 1.0 - sims

def run_ph(dist_matrix, maxdim=2):
    """Run persistent homology on a precomputed distance matrix."""
    result = ripser.ripser(dist_matrix, maxdim=maxdim, distance_matrix=True)
    return result['dgms']   # list: dgms[k] = (birth, death) array for H_k

def betti_at_threshold(dgms, t):
    """Count Betti numbers β0, β1, β2 at distance threshold t."""
    betti = []
    for k, dgm in enumerate(dgms):
        if len(dgm) == 0:
            betti.append(0)
            continue
        # Count features alive at t: born before t and dead after t
        # For H0, infinite bars are always alive
        count = 0
        for birth, death in dgm:
            if birth <= t and (np.isinf(death) or death > t):
                count += 1
        betti.append(count)
    return betti

def persistent_features(dgms, min_lifetime=0.05):
    """Return features with lifetime > min_lifetime (non-noise)."""
    features = []
    for k, dgm in enumerate(dgms):
        for birth, death in dgm:
            lifetime = (death - birth) if not np.isinf(death) else float('inf')
            if lifetime >= min_lifetime:
                features.append({'dim': k, 'birth': float(birth),
                                  'death': float(death) if not np.isinf(death) else None,
                                  'lifetime': float(lifetime) if not np.isinf(death) else None})
    return features


# ── Scale 1: Full body-centroid topology ──────────────────────────────────────
print(f"\n{'='*70}")
print(f"Scale 1 — Full body-centroid topology ({len(full_centroids)} bodies)")
print(f"  Point cloud: body centroids in 43-dim axis space")
print(f"  Distance:    cosine distance")
print(f"{'='*70}")

coords_full, bodies_full = to_axis_coords(full_centroids)
D_full  = cosine_dist_matrix(coords_full)
dgms_full = run_ph(D_full, maxdim=2)

print(f"\n  {len(bodies_full)} bodies  |  dist range: "
      f"[{D_full[D_full>0].min():.4f}, {D_full.max():.4f}]")

# Describe the persistence diagrams
print(f"\n  Persistence diagram summary:")
for k, dgm in enumerate(dgms_full):
    finite = [(b, d) for b, d in dgm if not np.isinf(d)]
    infinite = [(b, d) for b, d in dgm if np.isinf(d)]
    if len(finite) > 0:
        lifetimes = sorted([d-b for b, d in finite], reverse=True)
        print(f"    H{k}: {len(finite)} finite bars  |  "
              f"{len(infinite)} infinite bars  |  "
              f"top lifetimes: {', '.join(f'{lt:.4f}' for lt in lifetimes[:8])}")
    else:
        print(f"    H{k}: 0 finite bars  |  {len(infinite)} infinite bars")

# Betti numbers at several thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
print(f"\n  Betti numbers β0 β1 β2 at increasing distance thresholds:")
print(f"  {'Threshold':>10s}  {'β0':>5s}  {'β1':>5s}  {'β2':>5s}")
print(f"  {'-'*35}")
betti_full = {}
for t in thresholds:
    b = betti_at_threshold(dgms_full, t)
    while len(b) < 3: b.append(0)
    betti_full[t] = b
    print(f"  {t:>10.2f}  {b[0]:>5d}  {b[1]:>5d}  {b[2]:>5d}")

# Persistent (non-noise) features
pf = persistent_features(dgms_full, min_lifetime=0.05)
print(f"\n  Persistent features (lifetime > 0.05):")
for f in sorted(pf, key=lambda x: -(x['lifetime'] or 0)):
    lt_str = f"{f['lifetime']:.4f}" if f['lifetime'] else "∞"
    d_str  = f"{f['death']:.4f}"  if f['death']  else "∞"
    print(f"    H{f['dim']}  birth={f['birth']:.4f}  death={d_str}  lifetime={lt_str}")


# ── Scale 2: All Zone C words (subsample for tractability) ───────────────────
print(f"\n{'='*70}")
print(f"Scale 2 — Word-level Zone C topology (subsample)")
print(f"  Tests whether individual word positions show the same topology")
print(f"  as body centroids (coarse topology is self-similar across scales?)")
print(f"{'='*70}")

# Project all Zone C φ-vectors to axis space
coords_words = phi_c14 @ AXES.T   # (1647, 43)
norms_w      = np.linalg.norm(coords_words, axis=1, keepdims=True)
coords_words /= (norms_w + 1e-20)

# Subsample 300 words stratified by body (3 per body approx)
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = []
for body, idxs in body_members_full.items():
    n_sample = min(len(idxs), 3)
    chosen   = rng.choice(idxs, size=n_sample, replace=False)
    sample_idx.extend(chosen.tolist())
sample_idx = sorted(set(sample_idx))
print(f"\n  Subsampled {len(sample_idx)} Zone C words (≤3 per body)")

coords_sub = coords_words[sample_idx]
D_sub      = cosine_dist_matrix(coords_sub)
dgms_sub   = run_ph(D_sub, maxdim=2)

print(f"\n  Persistence diagram summary (word-level):")
for k, dgm in enumerate(dgms_sub):
    finite   = [(b, d) for b, d in dgm if not np.isinf(d)]
    infinite = [(b, d) for b, d in dgm if np.isinf(d)]
    if finite:
        lifetimes = sorted([d-b for b,d in finite], reverse=True)
        print(f"    H{k}: {len(finite)} finite  |  {len(infinite)} infinite  |  "
              f"top lifetimes: {', '.join(f'{lt:.4f}' for lt in lifetimes[:6])}")
    else:
        print(f"    H{k}: 0 finite  |  {len(infinite)} infinite")

print(f"\n  Betti numbers β0 β1 β2 at thresholds:")
print(f"  {'Threshold':>10s}  {'β0':>5s}  {'β1':>5s}  {'β2':>5s}")
print(f"  {'-'*35}")
betti_sub = {}
for t in thresholds:
    b = betti_at_threshold(dgms_sub, t)
    while len(b) < 3: b.append(0)
    betti_sub[t] = b
    print(f"  {t:>10.2f}  {b[0]:>5d}  {b[1]:>5d}  {b[2]:>5d}")


# ── Scale 3: Split-half stability of topology ────────────────────────────────
print(f"\n{'='*70}")
print(f"Scale 3 — Topological stability across {N_SPLITS} split-half replications")
print(f"  Each split: half of Zone C words → body centroids → PH")
print(f"  Stability measured by bottleneck distance between persistence diagrams")
print(f"{'='*70}")

split_dgms   = []    # list of dgms per split
split_bettis = []    # Betti numbers at t=0.3 per split
STAB_THRESHOLD = 0.3

for split_i in range(N_SPLITS):
    half = defaultdict(list)
    for body, idxs in body_members_full.items():
        shuffled = idxs[:]
        random.shuffle(shuffled)
        mid = max(1, len(shuffled) // 2)
        for idx in shuffled[:mid]: half[body].append(idx)

    cents_half = build_centroids(half, min_members=2)
    if len(cents_half) < 10: continue

    coords_half, _ = to_axis_coords(cents_half)
    D_half         = cosine_dist_matrix(coords_half)
    dgms_half      = run_ph(D_half, maxdim=2)
    split_dgms.append(dgms_half)

    b = betti_at_threshold(dgms_half, STAB_THRESHOLD)
    while len(b) < 3: b.append(0)
    split_bettis.append(b)

# Print Betti numbers per split
print(f"\n  Betti numbers at threshold={STAB_THRESHOLD} per split:")
print(f"  {'Split':>6s}  {'β0':>5s}  {'β1':>5s}  {'β2':>5s}  {'n_bodies':>10s}")
print(f"  {'-'*40}")
for i, b in enumerate(split_bettis):
    print(f"  {i+1:>6d}  {b[0]:>5d}  {b[1]:>5d}  {b[2]:>5d}")

# Compare to full
b_full = betti_full.get(STAB_THRESHOLD, [0,0,0])
print(f"\n  Full (95 bodies): β0={b_full[0]}  β1={b_full[1]}  β2={b_full[2]}")

# Betti stability stats
b0s = [b[0] for b in split_bettis]
b1s = [b[1] for b in split_bettis]
b2s = [b[2] for b in split_bettis]
print(f"\n  β0 across splits: mean={np.mean(b0s):.1f}  std={np.std(b0s):.2f}  "
      f"range=[{min(b0s)},{max(b0s)}]")
print(f"  β1 across splits: mean={np.mean(b1s):.1f}  std={np.std(b1s):.2f}  "
      f"range=[{min(b1s)},{max(b1s)}]")
print(f"  β2 across splits: mean={np.mean(b2s):.1f}  std={np.std(b2s):.2f}  "
      f"range=[{min(b2s)},{max(b2s)}]")

# Bottleneck distances between pairs of split persistence diagrams
print(f"\n  Bottleneck distances between pairs of split diagrams (H0, H1, H2):")
print(f"  (smaller = more stable topology)")
bn_h0, bn_h1, bn_h2 = [], [], []
for i in range(len(split_dgms)):
    for j in range(i+1, min(i+4, len(split_dgms))):  # sample pairs
        try:
            d0 = bottleneck(split_dgms[i][0], split_dgms[j][0])
            bn_h0.append(float(d0))
        except Exception:
            pass
        try:
            if len(split_dgms[i]) > 1 and len(split_dgms[j]) > 1:
                d1 = bottleneck(split_dgms[i][1], split_dgms[j][1])
                bn_h1.append(float(d1))
        except Exception:
            pass
        try:
            if len(split_dgms[i]) > 2 and len(split_dgms[j]) > 2:
                d2 = bottleneck(split_dgms[i][2], split_dgms[j][2])
                bn_h2.append(float(d2))
        except Exception:
            pass

if bn_h0: print(f"  H0 bottleneck: mean={np.mean(bn_h0):.4f}  max={np.max(bn_h0):.4f}")
if bn_h1: print(f"  H1 bottleneck: mean={np.mean(bn_h1):.4f}  max={np.max(bn_h1):.4f}")
if bn_h2: print(f"  H2 bottleneck: mean={np.mean(bn_h2):.4f}  max={np.max(bn_h2):.4f}")

# Bottleneck distance between full diagram and each split
print(f"\n  Bottleneck dist (full vs each split):")
bn_full_h1 = []
for i, dgms_split in enumerate(split_dgms):
    try:
        # H1 is the most informative (loops)
        d1 = bottleneck(dgms_full[1], dgms_split[1]) if (
            len(dgms_full) > 1 and len(dgms_split) > 1) else None
        d0 = bottleneck(dgms_full[0], dgms_split[0])
        d1_str = f"{d1:.4f}" if d1 is not None else "n/a"
        print(f"  Split {i+1}:  H0={d0:.4f}  H1={d1_str}")
        if d1 is not None: bn_full_h1.append(float(d1))
    except Exception as e:
        print(f"  Split {i+1}: error ({e})")


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"VERDICT — Is Zone C Topologically Quantum?")
print(f"{'='*70}")

b0_stable = np.std(b0s) <= 1 if b0s else False
b1_stable = np.std(b1s) <= 1 if b1s else False
b1_nonzero = np.mean(b1s) >= 1 if b1s else False
bn_stable  = np.mean(bn_full_h1) < 0.1 if bn_full_h1 else False

print(f"""
  Test                              Result          Threshold  Pass?
  ────────────────────────────────────────────────────────────────
  β0 stable across splits           std={np.std(b0s):.2f}        ≤1.0       {'✓' if b0_stable else '✗'}
  β1 stable across splits           std={np.std(b1s):.2f}        ≤1.0       {'✓' if b1_stable else '✗'}
  β1 > 0 (loops exist)              mean={np.mean(b1s):.1f}       ≥1         {'✓' if b1_nonzero else '✗'}
  H1 bottleneck(full,splits)        mean={np.mean(bn_full_h1):.4f}      <0.10      {'✓' if bn_stable else '✗'}
  ────────────────────────────────────────────────────────────────
""")

n_pass = sum([b0_stable, b1_stable, b1_nonzero, bn_stable])
if n_pass == 4:
    tq_verdict = "TOPOLOGICALLY QUANTUM — Betti numbers stable, non-trivial topology protected"
elif n_pass >= 2:
    tq_verdict = "PARTIALLY TOPOLOGICAL — some invariants stable, not all protected"
else:
    tq_verdict = "NOT TOPOLOGICAL — Betti numbers unstable, topology is noise"

print(f"  {tq_verdict}")

# Full comparison: body vs word topology
print(f"""
  Topology summary:
    Scale 1 (95 body centroids):
      Full range Betti at t=0.3: β0={b_full[0]}  β1={b_full[1]}  β2={betti_full[0.3][2] if 0.3 in betti_full else '?'}

    Scale 2 (~{len(sample_idx)} sampled Zone C words):
      Betti at t=0.3: β0={betti_sub.get(0.3,[0,0,0])[0]}  β1={betti_sub.get(0.3,[0,0,0])[1]}  β2={betti_sub.get(0.3,[0,0,0])[2]}

    The body centroids capture the COARSE topology;
    the word-level cloud shows finer structure.
    Comparison shows whether topology is self-similar across scales.
""")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_pytype(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_pytype(v) for v in x]
    if isinstance(x, dict): return {k: to_pytype(v) for k, v in x.items()}
    return x

def dgm_to_list(dgm):
    return [[float(b), float(d) if not np.isinf(d) else None]
            for b, d in dgm]

output = {
    'meta':            {'experiment': 'Day 52 — Betti Numbers of Zone C'},
    'scale1_bodies':   {
        'n_bodies': len(bodies_full),
        'betti_by_threshold': {str(t): betti_full[t] for t in thresholds},
        'persistent_features': pf,
        'dgms': [dgm_to_list(d) for d in dgms_full],
    },
    'scale2_words':    {
        'n_sampled': len(sample_idx),
        'betti_by_threshold': {str(t): betti_sub[t] for t in thresholds},
    },
    'scale3_stability': {
        'n_splits': len(split_bettis),
        'betti_per_split': split_bettis,
        'b0_std': float(np.std(b0s)), 'b0_mean': float(np.mean(b0s)),
        'b1_std': float(np.std(b1s)), 'b1_mean': float(np.mean(b1s)),
        'b2_std': float(np.std(b2s)), 'b2_mean': float(np.mean(b2s)),
        'bn_h0_mean': float(np.mean(bn_h0)) if bn_h0 else None,
        'bn_h1_mean': float(np.mean(bn_h1)) if bn_h1 else None,
        'bn_h2_mean': float(np.mean(bn_h2)) if bn_h2 else None,
        'bn_full_h1_mean': float(np.mean(bn_full_h1)) if bn_full_h1 else None,
    },
    'verdict':         tq_verdict,
    'tests_passed':    n_pass,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_pytype(output), f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 52 complete.")
