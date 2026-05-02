#!/usr/bin/env python3
"""
Expedition Day 35 — Zone C Analogy Arithmetic

Day 34 showed analogy arithmetic fails for degenerate pole words (man/woman/king)
because their φ-displacements converge to the same direction — the difference ≈ 0.
But Zone C words have large, body-specific displacements. This experiment tests:

  1. Do Zone C × Zone C analogies work? (b − a + c ≈ d, all Zone C)
  2. Does explicit centering on φ₀ improve Zone C analogy accuracy?
  3. Can we auto-discover morphological analogy pairs within Zone C?
  4. Does the "relationship vector" (b − a) generalise across body-pairs?
  5. Three-axis decomposition: project all words onto (Z2, φ₀, residual) and
     show the three-axis structure of φ-space.
"""

import os, json, re
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day35_zonec_analogies.json")

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def batch_phi(hs_matrix, z2):
    H  = hs_matrix.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

def cos_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-20)
    b = b / (np.linalg.norm(b) + 1e-20)
    return float(np.dot(a, b))

def nn_search(query, phi_mat, words, exclude, k=5):
    """Find top-k nearest neighbours in phi_mat (unit vectors)."""
    q = query / (np.linalg.norm(query) + 1e-20)
    sims = phi_mat @ q
    excl_set = set(exclude)
    for i, w in enumerate(words):
        if w in excl_set:
            sims[i] = -2.0
    top_k = np.argsort(sims)[-k:][::-1]
    return [(words[i], float(sims[i])) for i in top_k]

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n── Load ──────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words = [w for w, v in wmap.items() if v['phase']==2
                and v.get('L14_body') not in ('B000','B001',None) and w in w2i]
zone_d_words = [w for w, v in wmap.items() if v['phase']==2
                and v.get('L14_body') == 'B000' and w in w2i]

zone_c_bodies  = {w: wmap[w]['L14_body'] for w in zone_c_words}
zone_c_idx     = np.array([w2i[w] for w in zone_c_words])
zone_d_idx     = np.array([w2i[w] for w in zone_d_words])
zone_c_set     = set(zone_c_words)

body_label_map = {}
for w, v in wmap.items():
    b = v.get('L14_body')
    if b and b not in body_label_map:
        body_label_map[b] = v.get('L14_label', '?')

print(f"  Zone C: {len(zone_c_words)} words  Zone D: {len(zone_d_words)} words")

# ── Z2 axis ───────────────────────────────────────────────────────────────────
deltas = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        wa, wb = pfx+a, pfx+b
        if wa in w2i and wb in w2i:
            d = hs14_all[w2i[wb]] - hs14_all[w2i[wa]]
            dm = np.linalg.norm(d)
            if dm > 1e-20:
                deltas.append(d / dm)
            break
D = np.stack(deltas)
_, sv, Vt = np.linalg.svd(D, full_matrices=False)
z2   = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-20)
print(f"  Z2: {100*sv[0]**2/np.sum(sv**2):.1f}%")

# ── φ-vectors + φ₀ ────────────────────────────────────────────────────────────
phi_c14  = batch_phi(hs14_all[zone_c_idx],  z2)
phi_d14  = batch_phi(hs14_all[zone_d_idx],  z2)

phi0_14  = phi_d14.mean(axis=0)
phi0_14 /= (np.linalg.norm(phi0_14) + 1e-20)

# Full wmap φ-lookup (dictionary words only)
wmap_words = [w for w in wmap.keys() if w in w2i]
wmap_idx   = np.array([w2i[w] for w in wmap_words])
wmap_phi   = batch_phi(hs14_all[wmap_idx], z2)
wmap_w2l   = {w: i for i, w in enumerate(wmap_words)}

print(f"  φ₀ and wmap lookup computed")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 1 — Auto-Discover Zone C Morphological Pairs")
print(f"{'='*65}")

# Suffix rules: (suffix_b, suffix_a) → label for the relationship
# a has suffix_a, b has suffix_b, and base(b) == base(a) stripped of suffix_a
MORPHO_RULES = [
    # Comparative → Superlative:  bigger(er) → biggest(est)
    ('er',  'est',  'comparative→superlative'),
    # Base → Comparative:  big → bigger (but base might be Zone A)
    ('',    'er',   'base→comparative'),
    # Base → Superlative
    ('',    'est',  'base→superlative'),
    # Singular → Plural: cat → cats
    ('',    's',    'singular→plural'),
    ('',    'es',   'singular→plural'),
    # Base → Gerund
    ('',    'ing',  'base→gerund'),
    # Gerund → Past tense
    ('ing', 'ed',   'gerund→past'),
    # Adjective → Adverb
    ('',    'ly',   'base→adverb'),
    ('al',  'ally', 'adjective→adverb'),
    ('ic',  'ically','adjective→adverb'),
]

pairs_by_relation = defaultdict(list)
zone_c_set_local  = set(zone_c_words)

def strip_and_match(w, suf_a, suf_b):
    """If w ends with suf_b, return the word with suf_b replaced by suf_a."""
    if not w.endswith(suf_b):
        return None
    base_plus_a = w[:-len(suf_b)] + suf_a if suf_b else w + suf_a
    return base_plus_a

for w_b in zone_c_words:
    for suf_a, suf_b, label in MORPHO_RULES:
        if not suf_b:
            continue
        w_a = strip_and_match(w_b, suf_a, suf_b)
        if w_a and w_a != w_b and w_a in zone_c_set_local:
            if zone_c_bodies.get(w_a) == zone_c_bodies.get(w_b):
                pairs_by_relation[label].append((w_a, w_b, zone_c_bodies[w_a]))

print(f"\n  Morphological pairs found (both words in same Zone C body):")
for label, pairs in sorted(pairs_by_relation.items(), key=lambda x: -len(x[1])):
    print(f"    {label:<30s}: {len(pairs):>4d} pairs")
    for w_a, w_b, body in pairs[:4]:
        print(f"      {w_a:<20s} → {w_b:<20s}  [{body_label_map.get(body,'?')[:25]}]")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 2 — Zone C Analogy Tests (auto-generated)")
print(f"{'='*65}")

def analogy_test(a, b, c, phi_mat, word_list, phi_zero=None, exclude=None, k=5):
    """b − a + c in raw φ space (optionally centred on phi_zero)."""
    if not all(w in wmap_w2l for w in [a, b, c]):
        return None
    ia, ib, ic = wmap_w2l[a], wmap_w2l[b], wmap_w2l[c]
    if phi_zero is not None:
        va = wmap_phi[ia] - phi_zero
        vb = wmap_phi[ib] - phi_zero
        vc = wmap_phi[ic] - phi_zero
        v  = vb - va + vc + phi_zero
    else:
        v  = wmap_phi[ib] - wmap_phi[ia] + wmap_phi[ic]
    v /= (np.linalg.norm(v) + 1e-20)
    excl = (exclude or []) + [a, b, c]
    return nn_search(v, wmap_phi, wmap_words, excl, k)

# Build analogy tests from SAME-relation pairs across DIFFERENT words
analogy_tests_auto = []
for label, pairs in pairs_by_relation.items():
    if len(pairs) < 2:
        continue
    # Cross pairs: (a1, b1) as template, (a2, ?) as query → should give b2
    body_groups = defaultdict(list)
    for w_a, w_b, body in pairs:
        body_groups[body].append((w_a, w_b))
    # Within same body: b1 − a1 + a2 ≈ b2
    for body, bpairs in body_groups.items():
        if len(bpairs) < 2:
            continue
        for i in range(min(len(bpairs), 4)):
            for j in range(i+1, min(len(bpairs), 4)):
                a1, b1 = bpairs[i]
                a2, b2 = bpairs[j]
                analogy_tests_auto.append((a1, b1, a2, b2, label, body))
    # Cross-body: same relation, different body → should still give b2
    all_pairs_flat = [(w_a, w_b, body) for _, bp in pairs_by_relation.items()
                      for w_a, w_b, body in bp]
    bodies_list = list(body_groups.keys())
    if len(bodies_list) >= 2:
        b1_group = body_groups[bodies_list[0]]
        b2_group = body_groups[bodies_list[1]]
        if b1_group and b2_group:
            a1, b1 = b1_group[0]
            a2, b2 = b2_group[0]
            analogy_tests_auto.append((a1, b1, a2, b2, label+'_cross', bodies_list[0]))

print(f"\n  Generated {len(analogy_tests_auto)} auto-analogy tests")

# Run tests
raw_hits = 0
cent_hits = 0
raw_top5_hits = 0
cent_top5_hits = 0
n_valid = 0
results_by_label = defaultdict(lambda: {'raw':0,'cent':0,'total':0})
sample_results = []

for a, b, c, d_expected, label, body in analogy_tests_auto:
    raw_res  = analogy_test(a, b, c, wmap_phi, wmap_words, phi_zero=None)
    cent_res = analogy_test(a, b, c, wmap_phi, wmap_words, phi_zero=phi0_14)
    if raw_res is None:
        continue
    n_valid += 1
    raw_top1   = raw_res[0][0]
    cent_top1  = cent_res[0][0]
    raw_top5   = [w for w, _ in raw_res]
    cent_top5  = [w for w, _ in cent_res]
    raw_hit    = (raw_top1  == d_expected)
    cent_hit   = (cent_top1 == d_expected)
    raw_t5     = (d_expected in raw_top5)
    cent_t5    = (d_expected in cent_top5)
    raw_hits  += int(raw_hit)
    cent_hits += int(cent_hit)
    raw_top5_hits  += int(raw_t5)
    cent_top5_hits += int(cent_t5)
    results_by_label[label]['total'] += 1
    results_by_label[label]['raw']   += int(raw_hit)
    results_by_label[label]['cent']  += int(cent_hit)
    # Store interesting cases
    if len(sample_results) < 30:
        sample_results.append({
            'test': f"{b}-{a}+{c}={d_expected}",
            'raw_top1': raw_top1, 'cent_top1': cent_top1,
            'raw_hit': bool(raw_hit), 'cent_hit': bool(cent_hit),
            'label': label, 'body': body_label_map.get(body,'?'),
        })

print(f"\n  Results over {n_valid} valid Zone C analogy tests:")
print(f"    Raw φ:       top-1={raw_hits}/{n_valid} ({100*raw_hits/max(n_valid,1):.1f}%)  "
      f"top-5={raw_top5_hits}/{n_valid} ({100*raw_top5_hits/max(n_valid,1):.1f}%)")
print(f"    Centred φ:   top-1={cent_hits}/{n_valid} ({100*cent_hits/max(n_valid,1):.1f}%)  "
      f"top-5={cent_top5_hits}/{n_valid} ({100*cent_top5_hits/max(n_valid,1):.1f}%)")
print(f"    Improvement: Δtop1={cent_hits-raw_hits:+d}  Δtop5={cent_top5_hits-raw_top5_hits:+d}")

print(f"\n  Results by relationship type:")
print(f"  {'Relation':<35s}  {'Raw':>5s}  {'Cent':>5s}  {'Total':>5s}")
for label, r in sorted(results_by_label.items(), key=lambda x: -x[1]['total']):
    rt = r['total']
    print(f"  {label:<35s}  {r['raw']:>5d}  {r['cent']:>5d}  {rt:>5d}")

print(f"\n  Sample test results (first 20 non-trivial):")
print(f"  {'Test':<40s}  {'Raw':>15s}  {'Cent':>15s}  Status")
shown = 0
for r in sample_results:
    if shown >= 20:
        break
    flag = ("✓→✓" if r['raw_hit'] and r['cent_hit'] else
            "✗→✓" if not r['raw_hit'] and r['cent_hit'] else
            "✓→✗" if r['raw_hit'] and not r['cent_hit'] else "✗→✗")
    print(f"  {r['test']:<40s}  {r['raw_top1']:>15s}  {r['cent_top1']:>15s}  {flag}")
    shown += 1


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 3 — Relationship Vector Generalisation")
print(f"{'='*65}")

# For each morphological relation, compute the mean relationship vector
# and test how consistently it generalises across bodies
print(f"\n  Mean relationship vector consistency (cosine between all pair-vectors):")
for label, pairs in sorted(pairs_by_relation.items(), key=lambda x: -len(x[1])):
    if len(pairs) < 3:
        continue
    vecs = []
    for w_a, w_b, _ in pairs:
        if w_a in wmap_w2l and w_b in wmap_w2l:
            v = wmap_phi[wmap_w2l[w_b]] - wmap_phi[wmap_w2l[w_a]]
            nm = np.linalg.norm(v)
            if nm > 1e-20:
                vecs.append(v / nm)
    if len(vecs) < 3:
        continue
    V = np.stack(vecs)
    gram = V @ V.T
    upper = gram[np.triu_indices(len(V), k=1)]
    mean_cos = upper.mean()
    std_cos  = upper.std()
    mean_rel = V.mean(axis=0)
    mean_rel_norm = np.linalg.norm(mean_rel)
    # Cross-body: does the mean vector generalise?
    # Use mean vector to predict b given a for held-out pairs
    n = len(vecs)
    # How many would the mean vector retrieve correctly (top-1)?
    correct_mean = 0
    for i, (w_a, w_b, _) in enumerate(pairs[:20]):
        if w_a not in wmap_w2l:
            continue
        query = wmap_phi[wmap_w2l[w_a]] + mean_rel
        res = nn_search(query, wmap_phi, wmap_words, [w_a], k=1)
        if res and res[0][0] == w_b:
            correct_mean += 1
    pct_mean = 100*correct_mean/min(20,len(pairs))
    print(f"    {label:<30s}: mean_cos={mean_cos:.3f}±{std_cos:.3f}  "
          f"|mean_vec|={mean_rel_norm:.3f}  mean→correct={pct_mean:.0f}%  (n={n})")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 4 — Three-Axis Decomposition of φ-Space")
print(f"{'='*65}")

# Three axes:
#   1. Z2:  frequency axis
#   2. φ₀:  semantic zero direction (Zone D centroid)
#   3. φ_perp: component perpendicular to both Z2 and φ₀
# By construction φ₀ ⊥ Z2 (|cos|=0.000, confirmed Day 34).
# φ_perp is the "pure semantic body" direction.

# Verify φ₀ ⊥ Z2 here
cos_phi0_z2 = abs(float(phi0_14 @ z2))
print(f"\n  φ₀ ⊥ Z2: |cos| = {cos_phi0_z2:.6f}  ({'confirmed' if cos_phi0_z2 < 0.001 else 'NOT confirmed'})")

# Project all Zone C words onto the three axes
# Axis 1: Z2 projection of raw hidden state (before φ-transform)
hs_c_norm = hs14_all[zone_c_idx].astype(np.float64)
hs_c_nm   = np.linalg.norm(hs_c_norm, axis=1, keepdims=True)
hs_c_norm = hs_c_norm / (hs_c_nm + 1e-20)
z2_proj_c = hs_c_norm @ z2  # Z2 coordinate in hs-space

# Axis 2: φ₀ coordinate in φ-space
phi0_proj_c = phi_c14 @ phi0_14

# Axis 3: residual (perpendicular to both, in φ-space)
# Since φ₀ ⊥ Z2 in φ-space, the residual is simply |φ - (φ·φ₀)φ₀|
phi_perp_c  = phi_c14 - (phi0_proj_c[:, None] * phi0_14[None, :])
phi_perp_c_norm = np.linalg.norm(phi_perp_c, axis=1)

# Same for Zone D and Zone A/B
hs_d_norm    = hs14_all[zone_d_idx].astype(np.float64)
hs_d_nm      = np.linalg.norm(hs_d_norm, axis=1, keepdims=True)
hs_d_norm    = hs_d_norm / (hs_d_nm + 1e-20)
z2_proj_d    = hs_d_norm @ z2
phi0_proj_d  = phi_d14 @ phi0_14
phi_perp_d   = phi_d14 - (phi0_proj_d[:, None] * phi0_14[None, :])
phi_perp_d_norm = np.linalg.norm(phi_perp_d, axis=1)

zone_ab_idx  = np.array([w2i[w] for w, v in wmap.items()
                         if v['phase']==1 and w in w2i])
phi_ab14     = batch_phi(hs14_all[zone_ab_idx], z2)
hs_ab_norm   = hs14_all[zone_ab_idx].astype(np.float64)
hs_ab_nm     = np.linalg.norm(hs_ab_norm, axis=1, keepdims=True)
hs_ab_norm   = hs_ab_norm / (hs_ab_nm + 1e-20)
z2_proj_ab   = hs_ab_norm @ z2
phi0_proj_ab = phi_ab14 @ phi0_14
phi_perp_ab  = phi_ab14 - (phi0_proj_ab[:, None] * phi0_14[None, :])
phi_perp_ab_norm = np.linalg.norm(phi_perp_ab, axis=1)

print(f"\n  Three-axis zone statistics:")
print(f"  {'Axis':<20s}  {'Zone A/B':>12s}  {'Zone C':>12s}  {'Zone D':>12s}")
print(f"  {'-'*57}")

def fmt(arr, prec=4):
    return f"{arr.mean():.{prec}f}±{arr.std():.{prec}f}"

print(f"  {'Z2 (freq, hs-space)':<20s}  {fmt(z2_proj_ab):>12s}  {fmt(z2_proj_c):>12s}  {fmt(z2_proj_d):>12s}")
print(f"  {'φ₀ (semantic zero)':<20s}  {fmt(phi0_proj_ab):>12s}  {fmt(phi0_proj_c):>12s}  {fmt(phi0_proj_d):>12s}")
print(f"  {'‖φ_perp‖ (body dir)':<20s}  {fmt(phi_perp_ab_norm):>12s}  {fmt(phi_perp_c_norm):>12s}  {fmt(phi_perp_d_norm):>12s}")

# Zone C body centroids projected onto the three axes
body_members = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

print(f"\n  Zone C body centroids: spread along body-direction axis (φ_perp)")
print(f"  (showing bodies with n≥5, sorted by mean φ_perp norm)")
body_stats = []
for body, idxs in body_members.items():
    if len(idxs) < 5:
        continue
    perp_norms = phi_perp_c_norm[idxs]
    phi0_projs = phi0_proj_c[idxs]
    body_stats.append((body, len(idxs), perp_norms.mean(), phi0_projs.mean()))
body_stats.sort(key=lambda x: -x[2])
for body, n, perp_mean, phi0_mean in body_stats[:12]:
    lbl = body_label_map.get(body, '?')[:28]
    print(f"    {body}: {lbl:<28s}  n={n:>4d}  "
          f"‖φ_perp‖={perp_mean:.4f}  φ₀_proj={phi0_mean:.4f}")

print(f"\n  Interpretation:")
print(f"  Zone D (ocean): high φ₀_proj = AT semantic zero")
print(f"  Zone C (semantic): high ‖φ_perp‖ = displaced into body-specific direction")
print(f"  Zone A/B (pole): low φ₀_proj, low ‖φ_perp‖ = in a third direction (pole axis)")
print(f"  → Three mutually orthogonal regions: ocean (φ₀), semantic bodies (φ_perp), pole (residual)")

# Three-axis variance explained
print(f"\n  Three-axis decomposition:")
print(f"    Z2 variance (hs-space): ------  [external to φ-space, removed by construction]")
print(f"    φ₀ axis variance (Zone C): std={phi0_proj_c.std():.4f}")
print(f"    φ_perp axis variance (Zone C): std={phi_perp_c_norm.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 5 — Cross-Zone Analogy: Zone C → Zone D direction")
print(f"{'='*65}")

# What happens when we compute b − a + φ₀ (Zone D "target")?
# Does landing near φ₀ mean we land among Zone D words?
print(f"\n  Does b − a + φ₀ land in Zone D? (using Zone C word pairs as a, b)")
n_lands_in_d = 0
n_lands_in_c = 0
n_test_cd = 0
for label, pairs in pairs_by_relation.items():
    for w_a, w_b, body in pairs[:5]:
        if w_a not in wmap_w2l or w_b not in wmap_w2l:
            continue
        v = wmap_phi[wmap_w2l[w_b]] - wmap_phi[wmap_w2l[w_a]] + phi0_14
        v /= (np.linalg.norm(v) + 1e-20)
        res = nn_search(v, wmap_phi, wmap_words, [w_a, w_b], k=1)
        if res:
            top_w = res[0][0]
            is_d  = top_w in zone_c_bodies and zone_c_bodies.get(top_w) == 'B000'
            is_c  = top_w in zone_c_set
            n_lands_in_d += int(not is_c)
            n_lands_in_c += int(is_c)
            n_test_cd += 1

print(f"    Tests: {n_test_cd}  →  lands in Zone C: {n_lands_in_c}  "
      f"lands in Zone D or other: {n_lands_in_d}")
print(f"    → Zone C relationship vector + Zone D center "
      f"{'stays in Zone C (body geometry dominates)' if n_lands_in_c > n_lands_in_d else 'moves toward Zone D'}")


# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "meta": {"experiment": "Day 35 — Zone C Analogy Arithmetic"},
    "analogy_accuracy": {
        "n_tests": n_valid,
        "raw_top1": raw_hits, "raw_top5": raw_top5_hits,
        "centred_top1": cent_hits, "centred_top5": cent_top5_hits,
        "raw_top1_pct": round(100*raw_hits/max(n_valid,1), 2),
        "centred_top1_pct": round(100*cent_hits/max(n_valid,1), 2),
    },
    "pairs_found": {k: len(v) for k, v in pairs_by_relation.items()},
    "three_axis": {
        "phi0_perp_z2": float(cos_phi0_z2),
        "zone_c_phi0_mean": float(phi0_proj_c.mean()),
        "zone_d_phi0_mean": float(phi0_proj_d.mean()),
        "zone_ab_phi0_mean": float(phi0_proj_ab.mean()),
        "zone_c_perp_mean": float(phi_perp_c_norm.mean()),
        "zone_d_perp_mean": float(phi_perp_d_norm.mean()),
        "zone_ab_perp_mean": float(phi_perp_ab_norm.mean()),
    },
    "sample_results": sample_results[:20],
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 35 complete.")
