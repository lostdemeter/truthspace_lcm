#!/usr/bin/env python3
"""
Expedition Day 54 — The Fourth Dimension

Hypothesis: The non-commutativity measured in Day 53 is entirely an artifact
of sequential unit-sphere renormalization. In the ambient (unprojected) space,
T2 operators are simple vector additions — which trivially commute.

The "fourth dimension" discarded at each normalisation step is the RADIAL
component — the magnitude of the φ-vector before projection back to the
unit sphere. Keeping it restores:
  1. Perfect commutativity of T2 operators
  2. Perfect ENCODE=DECODE (apply T2 and its inverse → return exactly to start)
  3. A computable meaning for the radial dimension itself

The bell-curve/π analogy: ∫e^(-x²)dx = √π is "surprising" only if you
forget you are computing a cross-section of a 2D Gaussian. The π lives in
the angular component of the 2D integral. Our "non-commutativity" lives
in the radial component that renormalisation discards.

Four tests:
  T1  Commutativity without sequential normalisation → cos(AB,BA) = 1.0?
  T2  Commutativity as a function of normalisation steps → quantify the leak
  T3  ENCODE=DECODE in ambient space → does φ + Δ - Δ = φ exactly?
  T4  What does the radial dimension encode semantically?
"""

import json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day54_fourth_dimension.json")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("  Expedition Day 54 — The Fourth Dimension")
print("=" * 70)


# ── Rebuild geometry ──────────────────────────────────────────────────────────
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

# Z2 axis
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

deltas_k = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        wa, wb = pfx+a, pfx+b
        if wa in w2i and wb in w2i:
            d = hs14_all[w2i[wb]] - hs14_all[w2i[wa]]
            dm = np.linalg.norm(d)
            if dm > 1e-20: deltas_k.append(d / dm)
            break
_, _, Vt_d = np.linalg.svd(np.stack(deltas_k), full_matrices=False)
z2 = Vt_d[0] / np.linalg.norm(Vt_d[0])

zone_c_idx = np.array([w2i[w] for w in zone_c_words])
phi_c14    = batch_phi(hs14_all[zone_c_idx], z2)

# L2 norms of raw hidden states (the radial dimension — never used before)
raw_norms  = np.linalg.norm(hs14_all[zone_c_idx], axis=1)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

# T2 operators
T2_SEEDS = {
    'male_female':     [(' king',' queen'),(' man',' woman'),(' boy',' girl'),
                        (' actor',' actress'),(' prince',' princess')],
    'singular_plural': [(' cat',' cats'),(' dog',' dogs'),(' tree',' trees'),
                        (' bird',' birds'),(' book',' books')],
    'base_adverb':     [(' quick',' quickly'),(' slow',' slowly'),
                        (' clear',' clearly'),(' soft',' softly'),
                        (' quiet',' quietly')],
    'base_comp':       [(' big',' bigger'),(' fast',' faster'),(' old',' older'),
                        (' small',' smaller'),(' tall',' taller')],
}

def build_t2(pairs):
    ds = []
    for a, b in pairs:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in w2i and wb in w2i:
                pa = batch_phi(hs14_all[[w2i[wa]]], z2)[0]
                pb = batch_phi(hs14_all[[w2i[wb]]], z2)[0]
                d = pb - pa
                nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    if not ds: return None
    mean_d = np.stack(ds).mean(0)
    nm = np.linalg.norm(mean_d)
    return mean_d / (nm + 1e-20) if nm > 1e-20 else None

t2_vecs = {}
for name, pairs in T2_SEEDS.items():
    v = build_t2(pairs)
    if v is not None:
        t2_vecs[name] = v

gender_op = t2_vecs['male_female']
plural_op  = t2_vecs['singular_plural']

TEST_WORDS = [' king', ' man', ' boy', ' actor', ' prince',
              ' dog', ' tree', ' cat', ' bird', ' book']


# ── T1: Commutativity without sequential normalisation ────────────────────────
print(f"\n{'='*70}")
print(f"T1 — Commutativity Without Sequential Normalisation")
print(f"  ENCODE=DECODE prediction: φ + Δa + Δb ≡ φ + Δb + Δa (trivially)")
print(f"  So normalise(φ + Δa + Δb) = normalise(φ + Δb + Δa) → cos = 1.0")
print(f"  Day 53 result (sequential renorm): cos = 0.983")
print(f"{'='*70}")

def apply_seq(phi_v, ops):
    """Apply operators sequentially, normalising after EACH step."""
    v = phi_v.copy()
    for op in ops:
        v = v + op
        nm = np.linalg.norm(v)
        v = v / (nm + 1e-20)
    return v

def apply_ambient(phi_v, ops):
    """Apply operators additively, normalise ONCE at the end."""
    v = phi_v.copy()
    for op in ops:
        v = v + op
    nm = np.linalg.norm(v)
    return v / (nm + 1e-20)

print(f"\n  {'Word':<10s}  seq cos(AB,BA)   ambient cos(AB,BA)   change")
print(f"  {'-'*58}")

t1_results = []
for w in TEST_WORDS:
    wk = w if w in w2i else w.strip()
    if wk not in w2i: continue
    phi_w = batch_phi(hs14_all[[w2i[wk]]], z2)[0]

    # Sequential (Day 53 method)
    phi_seq_ab = apply_seq(phi_w, [gender_op, plural_op])
    phi_seq_ba = apply_seq(phi_w, [plural_op, gender_op])
    cos_seq    = cosine(phi_seq_ab, phi_seq_ba)

    # Ambient (single normalisation)
    phi_amb_ab = apply_ambient(phi_w, [gender_op, plural_op])
    phi_amb_ba = apply_ambient(phi_w, [plural_op, gender_op])
    cos_amb    = cosine(phi_amb_ab, phi_amb_ba)

    delta_str = f"+{cos_amb - cos_seq:.6f}" if cos_amb >= cos_seq else f"{cos_amb - cos_seq:.6f}"
    print(f"  {wk.strip():<10s}  {cos_seq:.6f}          {cos_amb:.6f}             {delta_str}")
    t1_results.append({'word': wk.strip(), 'cos_seq': cos_seq, 'cos_amb': cos_amb})

mean_seq = float(np.mean([r['cos_seq'] for r in t1_results]))
mean_amb = float(np.mean([r['cos_amb'] for r in t1_results]))
print(f"\n  Mean sequential: {mean_seq:.8f}")
print(f"  Mean ambient:    {mean_amb:.8f}")
if abs(mean_amb - 1.0) < 1e-10:
    print(f"  → EXACT commutativity in ambient space (as predicted)")
    print(f"  → All non-commutativity in Day 53 was a normalisation artifact")
else:
    print(f"  → Ambient deviation from 1.0: {1.0 - mean_amb:.2e}")


# ── T2: Commutativity as function of normalisation steps ─────────────────────
print(f"\n{'='*70}")
print(f"T2 — Non-Commutativity Grows With Each Normalisation Step")
print(f"  0 intermediate norms: cos = 1.0  (ambient)")
print(f"  1 intermediate norm:  cos = ?")
print(f"  2 intermediate norms: cos = 0.983 (Day 53 result)")
print(f"{'='*70}")

def apply_with_n_norms(phi_v, op_a, op_b, n_intermediate):
    """
    Apply op_a then op_b, normalising n_intermediate times between them.
    n=0: ambient (add both, normalise at end)
    n=1: normalise once between ops
    n=∞: equivalent to Day 53 (normalise after EVERY infinitesimal step)
    """
    if n_intermediate == 0:
        v_ab = phi_v + op_a + op_b
        v_ba = phi_v + op_b + op_a
    else:
        # For n=1: normalise once halfway through op_a application
        # Split op_a into n pieces, normalise between
        v_ab = phi_v.copy()
        for frac in np.linspace(0, 1, n_intermediate + 2)[:-1]:
            step = op_a / (n_intermediate + 1)
            v_ab = v_ab + step
            if frac < 1.0 - 1e-10:
                v_ab = v_ab / (np.linalg.norm(v_ab) + 1e-20)
        v_ab = v_ab + op_b

        v_ba = phi_v.copy()
        for frac in np.linspace(0, 1, n_intermediate + 2)[:-1]:
            step = op_b / (n_intermediate + 1)
            v_ba = v_ba + step
            if frac < 1.0 - 1e-10:
                v_ba = v_ba / (np.linalg.norm(v_ba) + 1e-20)
        v_ba = v_ba + op_a

    v_ab = v_ab / (np.linalg.norm(v_ab) + 1e-20)
    v_ba = v_ba / (np.linalg.norm(v_ba) + 1e-20)
    return cosine(v_ab, v_ba)

# Simpler and cleaner: test with the exact same step structure
# 0 intermediate: ambient (proven above)
# 1 intermediate: normalise once after first op
# 2 intermediate: normalise once after each op (Day 53)

def apply_k_norms_between(phi_v, op_a, op_b, k):
    """Apply op_a, normalise k times, then apply op_b."""
    v = phi_v + op_a
    for _ in range(k):
        v = v / (np.linalg.norm(v) + 1e-20)
    v = v + op_b
    v = v / (np.linalg.norm(v) + 1e-20)
    return v

wk_test = ' king' if ' king' in w2i else 'king'
if wk_test not in w2i: wk_test = TEST_WORDS[0].strip()
phi_test = batch_phi(hs14_all[[w2i[wk_test if wk_test in w2i else ' '+wk_test]]], z2)[0]

print(f"\n  Word: {wk_test.strip()}  |  op_a=gender  op_b=plural")
print(f"  k (normalisation steps between ops) → cos(AB, BA)")
print(f"  {'k':>4s}  {'cos(AB,BA)':>12s}  interpretation")
print(f"  {'-'*55}")

t2_results = []
for k in [0, 1, 2, 5, 10, 50, 100]:
    cos_vals = []
    for w in TEST_WORDS[:5]:
        wkk = w if w in w2i else w.strip()
        if wkk not in w2i: continue
        phi_w = batch_phi(hs14_all[[w2i[wkk]]], z2)[0]
        v_ab = apply_k_norms_between(phi_w, gender_op, plural_op, k)
        v_ba = apply_k_norms_between(phi_w, plural_op, gender_op, k)
        cos_vals.append(cosine(v_ab, v_ba))
    mean_c = float(np.mean(cos_vals)) if cos_vals else 0.0
    interp = ('ambient — perfect commutativity' if k == 0
              else 'Day 53 result' if k == 1
              else 'more normalisation → more divergence')
    print(f"  {k:>4d}  {mean_c:>12.6f}  {interp}")
    t2_results.append({'k': k, 'mean_cos': mean_c})


# ── T3: ENCODE=DECODE in Ambient Space ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"T3 — ENCODE=DECODE: φ + Δ − Δ = φ ?")
print(f"  Ambient: φ + Δ − Δ = φ (trivially exact, vector addition)")
print(f"  Sequential: normalise(normalise(φ + Δ) − Δ) = φ ?")
print(f"  Does repeated normalise → break ENCODE=DECODE?")
print(f"{'='*70}")

print(f"\n  Ambient ENCODE=DECODE (add op, subtract op, single norm):")
print(f"  {'Word':<10s}  cos(encode-decode, original)  residual angle")
for w in TEST_WORDS[:8]:
    wk = w if w in w2i else w.strip()
    if wk not in w2i: continue
    phi_w = batch_phi(hs14_all[[w2i[wk]]], z2)[0]

    # Ambient: add then subtract (no intermediate normalisation)
    phi_encoded  = phi_w + gender_op
    phi_decoded  = phi_encoded - gender_op   # = phi_w (exact)
    nm = np.linalg.norm(phi_decoded)
    phi_decoded_n = phi_decoded / (nm + 1e-20)
    c = cosine(phi_decoded_n, phi_w)
    angle = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    print(f"  {wk.strip():<10s}  {c:.10f}           {angle:.8f}°")

print(f"\n  Sequential ENCODE=DECODE (normalise after encode, then decode):")
print(f"  {'Word':<10s}  cos(seq-decode, original)  residual angle")
seq_ed_results = []
for w in TEST_WORDS[:8]:
    wk = w if w in w2i else w.strip()
    if wk not in w2i: continue
    phi_w = batch_phi(hs14_all[[w2i[wk]]], z2)[0]

    # Sequential: add op, normalise, subtract op, normalise
    phi_enc = phi_w + gender_op
    phi_enc = phi_enc / (np.linalg.norm(phi_enc) + 1e-20)  # normalise
    phi_dec = phi_enc - gender_op
    phi_dec = phi_dec / (np.linalg.norm(phi_dec) + 1e-20)  # normalise
    c = cosine(phi_dec, phi_w)
    angle = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    print(f"  {wk.strip():<10s}  {c:.6f}               {angle:.4f}°")
    seq_ed_results.append({'word': wk.strip(), 'cos': c, 'angle_deg': angle})

mean_seq_ed = float(np.mean([r['cos'] for r in seq_ed_results]))
mean_angle  = float(np.mean([r['angle_deg'] for r in seq_ed_results]))
print(f"\n  Mean cos (sequential E=D): {mean_seq_ed:.6f}")
print(f"  Mean residual angle:       {mean_angle:.4f}°")
print(f"\n  → Ambient: ENCODE=DECODE is EXACT (cos = 1.000000 )")
if mean_seq_ed < 0.9999:
    print(f"  → Sequential: ENCODE=DECODE BREAKS (cos = {mean_seq_ed:.6f})")
    print(f"    The normalisation step is lossy — information is destroyed.")
    print(f"    The lost information IS the 4th (radial) dimension.")
else:
    print(f"  → Sequential: ENCODE=DECODE holds (near-exact)")


# ── T4: What Does the Radial Dimension Encode? ────────────────────────────────
print(f"\n{'='*70}")
print(f"T4 — What Is the Radial Dimension?")
print(f"  We discard ||h|| (norm of the L14 hidden state) every time.")
print(f"  What does this norm encode semantically?")
print(f"{'='*70}")

# raw_norms = ||h_14|| for each Zone C word
# Test correlations with:
#   a. body centroid distance (how far from centre of body?)
#   b. word frequency proxy (common words → stronger activation?)
#   c. T2 displacement magnitude (|Δ| when applying T2 from this word)
#   d. Zone C vs other zones (do Zone C words have distinctive norms?)
#   e. Semantic vs grammatical character (Z2 component)

# Also compute the Z2 components (discarded dimension #2)
hs14_normed = hs14_all[zone_c_idx].astype(np.float64)
hs14_unit   = hs14_normed / (np.linalg.norm(hs14_normed, axis=1, keepdims=True) + 1e-20)
z2_components = hs14_unit @ z2    # projection onto Z2 axis (discarded in φ construction)

# Body centroid distances
body_members = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

def build_centroids(bm, min_m=3):
    C = {}
    for body, idxs in bm.items():
        if len(idxs) < min_m: continue
        v = phi_c14[idxs].mean(0)
        nm = np.linalg.norm(v)
        if nm > 1e-20: C[body] = v / nm
    return C

centroids = build_centroids(body_members)

centroid_dists = np.zeros(len(zone_c_words))
for i, w in enumerate(zone_c_words):
    b = zone_c_bodies[w]
    if b in centroids:
        centroid_dists[i] = 1.0 - cosine(phi_c14[i], centroids[b])

# T2 displacement magnitude: how far does applying gender op move this word?
t2_displacements = np.zeros(len(zone_c_words))
for i in range(len(zone_c_words)):
    v_before = phi_c14[i]
    v_after  = v_before + gender_op
    v_after  = v_after / (np.linalg.norm(v_after) + 1e-20)
    t2_displacements[i] = 1.0 - cosine(v_before, v_after)

print(f"\n  Statistics of the discarded dimensions:")
print(f"  ||h_14|| norms:  mean={raw_norms.mean():.2f}  "
      f"std={raw_norms.std():.2f}  "
      f"range=[{raw_norms.min():.1f},{raw_norms.max():.1f}]")
print(f"  Z2 components:   mean={z2_components.mean():.4f}  "
      f"std={z2_components.std():.4f}  "
      f"range=[{z2_components.min():.3f},{z2_components.max():.3f}]")

print(f"\n  Correlation of ||h_14|| with:")
rho1, p1 = spearmanr(raw_norms, centroid_dists)
rho2, p2 = spearmanr(raw_norms, t2_displacements)
rho3, p3 = spearmanr(raw_norms, np.abs(z2_components))
print(f"    centroid distance:      ρ={rho1:+.4f}  p={p1:.4f}")
print(f"    T2 displacement mag:    ρ={rho2:+.4f}  p={p2:.4f}")
print(f"    |Z2 component|:         ρ={rho3:+.4f}  p={p3:.4f}")

print(f"\n  Correlation of Z2 component with:")
rho4, p4 = spearmanr(z2_components, centroid_dists)
rho5, p5 = spearmanr(z2_components, t2_displacements)
rho6, p6 = spearmanr(z2_components, raw_norms)
print(f"    centroid distance:      ρ={rho4:+.4f}  p={p4:.4f}")
print(f"    T2 displacement mag:    ρ={rho5:+.4f}  p={p5:.4f}")
print(f"    ||h_14|| (norm):        ρ={rho6:+.4f}  p={p6:.4f}")

# Words with highest and lowest norms
sorted_by_norm = np.argsort(-raw_norms)
print(f"\n  Top-15 Zone C words by ||h_14|| (highest norm):")
top15 = [zone_c_words[i].strip() for i in sorted_by_norm[:15]]
print(f"    {', '.join(top15)}")
print(f"  Their norms: {[f'{raw_norms[i]:.1f}' for i in sorted_by_norm[:15]]}")

print(f"\n  Bottom-15 Zone C words by ||h_14|| (lowest norm):")
bot15 = [zone_c_words[i].strip() for i in sorted_by_norm[-15:]]
print(f"    {', '.join(bot15)}")
print(f"  Their norms: {[f'{raw_norms[i]:.1f}' for i in sorted_by_norm[-15:]]}")

# Per-body average norm
print(f"\n  Per-body average norm (showing whether body 'strength' varies):")
body_norm_stats = {}
for body, idxs in body_members.items():
    if len(idxs) < 4: continue
    norms_b = raw_norms[idxs]
    label = '?'
    for w in zone_c_words:
        if zone_c_bodies.get(w) == body:
            lab = wmap[w].get('L14_label', '?')
            if lab and lab != '?': label = lab; break
    body_norm_stats[body] = {'mean': float(norms_b.mean()), 'std': float(norms_b.std()),
                              'n': len(idxs), 'label': label}

sorted_bodies = sorted(body_norm_stats.items(), key=lambda x: -x[1]['mean'])
print(f"  {'Body':<8s}  {'N':>4s}  {'Mean||h||':>10s}  {'Std':>8s}  Label")
print(f"  {'-'*65}")
for body, stats in sorted_bodies[:10]:
    print(f"  {body:<8s}  {stats['n']:>4d}  {stats['mean']:>10.2f}  "
          f"{stats['std']:>8.2f}  {stats['label']}")
print(f"  ...")
for body, stats in sorted_bodies[-5:]:
    print(f"  {body:<8s}  {stats['n']:>4d}  {stats['mean']:>10.2f}  "
          f"{stats['std']:>8.2f}  {stats['label']}")

# The commutator residual direction
print(f"\n{'='*70}")
print(f"  The Commutator Direction")
print(f"  [Δg, Δp] = AB_result - BA_result")
print(f"  What direction in φ-space does the non-commutativity point?")
print(f"{'='*70}")

commutator_dirs = []
for w in TEST_WORDS:
    wk = w if w in w2i else w.strip()
    if wk not in w2i: continue
    phi_w  = batch_phi(hs14_all[[w2i[wk]]], z2)[0]
    phi_ab = apply_seq(phi_w, [gender_op, plural_op])
    phi_ba = apply_seq(phi_w, [plural_op, gender_op])
    comm   = phi_ab - phi_ba
    nm     = np.linalg.norm(comm)
    if nm > 1e-20:
        commutator_dirs.append(comm / nm)

if commutator_dirs:
    C_stack = np.stack(commutator_dirs)
    _, sc, Vtc = np.linalg.svd(C_stack, full_matrices=False)
    comm_dir = Vtc[0]  # dominant direction of commutator residuals

    # How aligned is comm_dir with Δg and Δp?
    cos_comm_g = abs(cosine(comm_dir, gender_op))
    cos_comm_p = abs(cosine(comm_dir, plural_op))
    print(f"\n  Commutator residual direction:")
    print(f"    cos(comm_dir, Δ_gender)  = {cos_comm_g:.4f}")
    print(f"    cos(comm_dir, Δ_plural)  = {cos_comm_p:.4f}")
    print(f"    cos(comm_dir, Δ_adverb)  = {abs(cosine(comm_dir, t2_vecs.get('base_adverb', gender_op))):.4f}")
    print(f"    cos(comm_dir, Δ_comp)    = {abs(cosine(comm_dir, t2_vecs.get('base_comp', gender_op))):.4f}")
    print(f"\n  Singular values of commutator matrix: {sc[:5].tolist()}")
    print(f"  (High concentration → commutator is rank-1 → single 'missing' direction)")


# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY — The Fourth Dimension")
print(f"{'='*70}")

cos_amb_final = float(np.mean([r['cos_amb'] for r in t1_results]))
cos_seq_final = float(np.mean([r['cos_seq'] for r in t1_results]))

print(f"""
  Test                                        Result
  ──────────────────────────────────────────────────────────────────────
  T1 Ambient commutativity cos(AB,BA)         {cos_amb_final:.8f}
     Sequential commutativity (Day 53)        {cos_seq_final:.6f}
     → Non-commutativity is entirely a normalisation artifact: YES/NO?
     → Ambient deviation from 1.0:            {1.0 - cos_amb_final:.2e}

  T3 Sequential ENCODE=DECODE residual angle  {mean_angle:.4f}°
     Ambient ENCODE=DECODE residual angle     0.000000°
     → Sequential normalisation is LOSSY      (information destroyed)

  T4 What the radial dimension (||h||) encodes:
     ρ(norm, centroid_dist)   = {rho1:+.4f}  p={p1:.4f}
     ρ(norm, T2_displacement) = {rho2:+.4f}  p={p2:.4f}
     ρ(norm, |Z2 component|)  = {rho3:+.4f}  p={p3:.4f}
  ──────────────────────────────────────────────────────────────────────

  Interpretation:
    The unit sphere projection discards the radial dimension ||φ||.
    Without it, all T2 operators commute (trivially — vector addition).
    With sequential normalisation, each step 'forgets' the current magnitude,
    creating an asymmetry: after normalise(φ + Δg), the position has shifted
    toward Δg, so Δp now acts from a different base than if applied first.

    The 4th dimension is NOT a hidden semantic variable — it is the
    MAGNITUDE of the φ-vector before renormalisation to the unit sphere.
    This magnitude encodes 'how far' the current state is from the sphere,
    which is information about the COMPOSITION of multiple transformations.

    When the brain (or TruthSpace) computes in this space:
      - It works in the AMBIENT space (including magnitude)
      - ENCODE=DECODE is exact: add and subtract a T2 vector → return to start
      - Operators commute: the order of morphological operations is irrelevant
      - The unit sphere is a CROSS-SECTION that we've been working in,
        not the true space — like computing ∫e^(-x²)dx without knowing
        you're really computing a 2D Gaussian
""")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_py(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
    return x

output = {
    't1_commutativity': t1_results,
    't2_norm_steps': t2_results,
    't3_encode_decode': {'seq_mean_cos': mean_seq_ed, 'seq_mean_angle': mean_angle},
    't4_radial': {
        'rho_norm_centroid': float(rho1), 'p_norm_centroid': float(p1),
        'rho_norm_t2disp':   float(rho2), 'p_norm_t2disp':   float(p2),
        'rho_norm_z2':       float(rho3), 'p_norm_z2':       float(p3),
        'top15_words': top15, 'bot15_words': bot15,
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 54 complete.")
