#!/usr/bin/env python3
"""
Expedition Day 34 — The Semantic Zero: Measuring φ₀ and Centre Shift

DC 314 established that the Zone D centroid is the implicit origin of all
φ-space arithmetic. This experiment measures it directly and answers:

  1. What IS φ₀ geometrically? Where does it sit relative to Z2 and the pole?
  2. Is φ₀ layer-stable? How much does it shift from L14 to L23?
  3. Does explicit centring improve nearest-neighbour body retrieval?
  4. What is the displacement magnitude distribution across all zones?
  5. Do morphological analogy vectors (king-man+woman) land closer to the
     correct answer in centred vs raw φ-space?
  6. Can we measure context shift as a vector from Day 29 vs Day 30 data?
"""

import os, json, time
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

CACHE_FILE   = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE   = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
PN_CACHE     = os.path.join(os.path.dirname(__file__), "day29_pn_cache.npz")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "day34_semantic_zero.json")

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]

ANALOGY_TESTS = [
    # (a, b, c, expected_d) — b is to a as d is to c
    ('man',   'woman', 'king',   'queen'),
    ('man',   'woman', 'boy',    'girl'),
    ('man',   'woman', 'father', 'mother'),
    ('cat',   'cats',  'dog',    'dogs'),
    ('cat',   'cats',  'tree',   'trees'),
    ('big',   'bigger','fast',   'faster'),
    ('big',   'bigger','old',    'older'),
    ('small', 'smaller','fast',  'faster'),
]

t0 = time.time()


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


# ── Load ─────────────────────────────────────────────────────────────────────
print(f"\n── Load ──────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
hs23_all  = npz['hs_23'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words = [w for w, v in wmap.items() if v['phase']==2
                and v.get('L14_body') not in ('B000','B001',None) and w in w2i]
zone_d_words = [w for w, v in wmap.items() if v['phase']==2
                and v.get('L14_body') == 'B000' and w in w2i]
zone_ab_words = [w for w, v in wmap.items() if v['phase']==1 and w in w2i]

zone_c_idx  = np.array([w2i[w] for w in zone_c_words])
zone_d_idx  = np.array([w2i[w] for w in zone_d_words])
zone_ab_idx = np.array([w2i[w] for w in zone_ab_words])

print(f"  Zone A/B: {len(zone_ab_idx)}  Zone C: {len(zone_c_idx)}  Zone D: {len(zone_d_idx)}")

# Body membership for Zone C
zone_c_bodies = [wmap[w]['L14_body'] for w in zone_c_words]
unique_bodies  = sorted(set(zone_c_bodies))

# ── Z2 axis ───────────────────────────────────────────────────────────────────
print(f"\n── Z2 axis ───────────────────────────────────────────────────────")
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
pct  = 100 * sv[0]**2 / (np.sum(sv**2) + 1e-20)
print(f"  Z2: {pct:.1f}%  ({len(deltas)} deltas)")

# ── φ-vectors ─────────────────────────────────────────────────────────────────
print(f"\n── φ-vectors ─────────────────────────────────────────────────────")
phi_c14  = batch_phi(hs14_all[zone_c_idx],  z2)
phi_d14  = batch_phi(hs14_all[zone_d_idx],  z2)
phi_ab14 = batch_phi(hs14_all[zone_ab_idx], z2)
phi_c23  = batch_phi(hs23_all[zone_c_idx],  z2)
phi_d23  = batch_phi(hs23_all[zone_d_idx],  z2)
print(f"  φ at L14 and L23 computed for all zones")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 1 — Measuring φ₀ (Zone D Centroid)")
print(f"{'='*65}")

phi0_14 = phi_d14.mean(axis=0)
phi0_14 /= (np.linalg.norm(phi0_14) + 1e-20)
phi0_23 = phi_d23.mean(axis=0)
phi0_23 /= (np.linalg.norm(phi0_23) + 1e-20)

phi_pole_14 = phi_ab14.mean(axis=0)
phi_pole_14 /= (np.linalg.norm(phi_pole_14) + 1e-20)

print(f"\n  φ₀(L14) and φ₀(L23) computed from Zone D centroids")
print(f"  Degenerate pole centroid computed from Zone A/B words")

# Key cosines
cos_z14_z23  = cos_sim(phi0_14, phi0_23)
cos_z14_pole = cos_sim(phi0_14, phi_pole_14)
cos_z14_z2   = abs(float(phi0_14 @ z2))
cos_z23_z2   = abs(float(phi0_23 @ z2))

print(f"\n  cos(φ₀(L14), φ₀(L23))   = {cos_z14_z23:+.6f}")
print(f"  cos(φ₀(L14), pole)       = {cos_z14_pole:+.6f}")
print(f"  |cos(φ₀(L14), Z2)|       = {cos_z14_z2:.6f}")
print(f"  |cos(φ₀(L23), Z2)|       = {cos_z23_z2:.6f}")

# How tight is Zone D around φ₀?
d_to_phi0_14 = phi_d14 @ phi0_14
d_to_phi0_23 = phi_d23 @ phi0_23
print(f"\n  Zone D word cosines to φ₀:")
print(f"    L14: mean={d_to_phi0_14.mean():.4f}  std={d_to_phi0_14.std():.4f}  "
      f"min={d_to_phi0_14.min():.4f}  max={d_to_phi0_14.max():.4f}")
print(f"    L23: mean={d_to_phi0_23.mean():.4f}  std={d_to_phi0_23.std():.4f}  "
      f"min={d_to_phi0_23.min():.4f}  max={d_to_phi0_23.max():.4f}")

# Interpretation
print(f"\n  Interpretation:")
if cos_z14_z23 > 0.99:
    print(f"  ✓ φ₀ is layer-STABLE (cos={cos_z14_z23:.4f}) — same direction at L14 and L23")
elif cos_z14_z23 > 0.90:
    print(f"  ~ φ₀ has MODERATE layer shift (cos={cos_z14_z23:.4f})")
else:
    print(f"  ✗ φ₀ has LARGE layer shift (cos={cos_z14_z23:.4f})")

if cos_z14_pole > 0.95:
    print(f"  ✓ φ₀ ≈ degenerate pole (cos={cos_z14_pole:.4f}) — center = default center")
elif cos_z14_pole > 0.80:
    print(f"  ~ φ₀ and pole are close but distinct (cos={cos_z14_pole:.4f})")
else:
    print(f"  ✗ φ₀ and degenerate pole are DISTINCT (cos={cos_z14_pole:.4f})")

if cos_z14_z2 < 0.1:
    print(f"  ✓ φ₀ is ORTHOGONAL to Z2 (|cos|={cos_z14_z2:.4f}) — center ⊥ frequency axis")
elif cos_z14_z2 < 0.5:
    print(f"  ~ φ₀ has partial alignment with Z2 (|cos|={cos_z14_z2:.4f})")
else:
    print(f"  ✗ φ₀ is aligned with Z2 (|cos|={cos_z14_z2:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 2 — Displacement Distribution Across Zones")
print(f"{'='*65}")

# Compute centred displacements: Δ = φ - φ₀
# In φ-space (unit sphere), "displacement" is angular distance from φ₀
# We measure it as: 1 - cos(φ, φ₀)

delta_c14  = 1.0 - (phi_c14  @ phi0_14)
delta_d14  = 1.0 - (phi_d14  @ phi0_14)
delta_ab14 = 1.0 - (phi_ab14 @ phi0_14)

print(f"\n  Displacement from φ₀ (= 1 − cos(φ, φ₀)):")
print(f"  {'Zone':<15s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
print(f"  {'-'*53}")
print(f"  {'Zone A/B (pole)':<15s}  {delta_ab14.mean():>8.4f}  {delta_ab14.std():>8.4f}  "
      f"{delta_ab14.min():>8.4f}  {delta_ab14.max():>8.4f}")
print(f"  {'Zone C (semantic)':<15s}  {delta_c14.mean():>8.4f}  {delta_c14.std():>8.4f}  "
      f"{delta_c14.min():>8.4f}  {delta_c14.max():>8.4f}")
print(f"  {'Zone D (ocean)':<15s}  {delta_d14.mean():>8.4f}  {delta_d14.std():>8.4f}  "
      f"{delta_d14.min():>8.4f}  {delta_d14.max():>8.4f}")

# Zone C: displacement by body
print(f"\n  Zone C displacement by body (mean Δ from φ₀) — top 10 most displaced:")
body_deltas = defaultdict(list)
for i, b in enumerate(zone_c_bodies):
    body_deltas[b].append(delta_c14[i])
body_mean_delta = {b: np.mean(v) for b, v in body_deltas.items()}
body_labels_snap = {w: wmap[w].get('L14_label','?') for w in zone_c_words}
# Get label for each body
body_label_map = {}
for w, v in wmap.items():
    b = v.get('L14_body')
    if b and b not in body_label_map:
        body_label_map[b] = v.get('L14_label','?')

top_displaced = sorted(body_mean_delta.items(), key=lambda x: -x[1])[:10]
for bid, d in top_displaced:
    lbl = body_label_map.get(bid,'?')[:30]
    print(f"    {bid}: {lbl:<30s}  Δ={d:.4f}  (n={len(body_deltas[bid])})")

# Correlation of displacement with token_id
all_phase2_words = zone_c_words + zone_d_words
all_deltas = np.concatenate([delta_c14, delta_d14])
all_tids   = np.array([wmap[w]['token_id'] for w in all_phase2_words])
r_d_tid, p_d_tid = spearmanr(all_tids, all_deltas)
print(f"\n  Spearman r(token_id, displacement): {r_d_tid:+.4f}  (p={p_d_tid:.2e})")
print(f"  → displacement is {'independent of' if abs(r_d_tid)<0.15 else 'weakly correlated with' if abs(r_d_tid)<0.4 else 'correlated with'} token_id")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 3 — Nearest-Neighbour Quality: Raw vs Centred φ")
print(f"{'='*65}")

# For each Zone C word, find nearest Zone C neighbour in:
#   (a) raw φ space
#   (b) centred φ space (= displacement from φ₀)
# Then check: does the nearest neighbour share the same body?
# Body-purity of NN retrieval = fraction where same body

# Centred φ vectors
phi_c14_centred = phi_c14 - phi0_14[None, :]     # subtract center
# Re-normalise on sphere
phi_c14_c_norm  = phi_c14_centred / (np.linalg.norm(phi_c14_centred, axis=1, keepdims=True) + 1e-20)

# Compute pairwise cosines — do batched to save memory
n = len(zone_c_words)
batch = 200
raw_purity   = []
cent_purity  = []
raw_rank     = []   # rank of same-body nearest among all neighbours
cent_rank    = []

for start in range(0, n, batch):
    end = min(start + batch, n)
    # Raw similarities
    raw_sims  = phi_c14[start:end]  @ phi_c14.T
    cent_sims = phi_c14_c_norm[start:end] @ phi_c14_c_norm.T
    for local_i, global_i in enumerate(range(start, end)):
        body_i = zone_c_bodies[global_i]
        # Mask self
        raw_sims[local_i,  global_i] = -2.0
        cent_sims[local_i, global_i] = -2.0

        raw_nn   = int(np.argmax(raw_sims[local_i]))
        cent_nn  = int(np.argmax(cent_sims[local_i]))
        raw_purity.append(int(zone_c_bodies[raw_nn]  == body_i))
        cent_purity.append(int(zone_c_bodies[cent_nn] == body_i))

raw_purity  = np.array(raw_purity)
cent_purity = np.array(cent_purity)

print(f"\n  Nearest-neighbour body purity (same body = correct):")
print(f"    Raw φ:      {raw_purity.mean():.4f}  ({raw_purity.sum()}/{len(raw_purity)} correct)")
print(f"    Centred φ:  {cent_purity.mean():.4f}  ({cent_purity.sum()}/{len(cent_purity)} correct)")
print(f"    Improvement: Δ = {cent_purity.mean()-raw_purity.mean():+.4f}")

# Where does centering HELP vs HURT?
helped = (raw_purity == 0) & (cent_purity == 1)
hurt   = (raw_purity == 1) & (cent_purity == 0)
print(f"\n  Centering HELPS:  {helped.sum()} words (wrong→right)")
print(f"  Centering HURTS:  {hurt.sum()} words (right→wrong)")
print(f"  Unchanged:        {(raw_purity == cent_purity).sum()} words")

# Examples where centering helps
if helped.sum() > 0:
    help_idx = np.where(helped)[0][:5]
    print(f"\n  Examples where centering helps:")
    for i in help_idx:
        print(f"    {zone_c_words[i]:<20s} body={zone_c_bodies[i]} ({body_label_map.get(zone_c_bodies[i],'?')[:25]})")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 4 — Analogy Arithmetic: Raw vs Centred φ")
print(f"{'='*65}")

# Build lookup: all Phase 2 φ-vectors for word lookup
all_p2_words  = zone_c_words + zone_d_words
all_p2_phi    = np.vstack([phi_c14, phi_d14])
all_p2_phi_c  = all_p2_phi - phi0_14[None, :]
all_p2_phi_cn = all_p2_phi_c / (np.linalg.norm(all_p2_phi_c, axis=1, keepdims=True) + 1e-20)
p2_w2i        = {w: i for i, w in enumerate(all_p2_words)}

# Lookup restricted to wmap dictionary words (avoids subword tokens)
wmap_words_ordered = [w for w in wmap.keys() if w in w2i]
wmap_phi_lookup    = batch_phi(hs14_all[[w2i[w] for w in wmap_words_ordered]], z2)
wmap_w2local       = {w: i for i, w in enumerate(wmap_words_ordered)}

def analogy_raw(a, b, c, exclude_words, k=5):
    """b−a+c in raw φ space, searched over dictionary words only."""
    if a not in wmap_w2local or b not in wmap_w2local or c not in wmap_w2local:
        return None, None
    v = wmap_phi_lookup[wmap_w2local[b]] - wmap_phi_lookup[wmap_w2local[a]] + wmap_phi_lookup[wmap_w2local[c]]
    v /= (np.linalg.norm(v) + 1e-20)
    sims = wmap_phi_lookup @ v
    for ew in exclude_words:
        if ew in wmap_w2local:
            sims[wmap_w2local[ew]] = -2.0
    top_k = np.argsort(sims)[-k:][::-1]
    return [wmap_words_ordered[i] for i in top_k], float(sims[top_k[0]])

def analogy_centred(a, b, c, exclude_words, k=5):
    """b−a+c in centred φ space, searched over dictionary words only."""
    if a not in wmap_w2local or b not in wmap_w2local or c not in wmap_w2local:
        return None, None
    va = wmap_phi_lookup[wmap_w2local[a]] - phi0_14
    vb = wmap_phi_lookup[wmap_w2local[b]] - phi0_14
    vc = wmap_phi_lookup[wmap_w2local[c]] - phi0_14
    v  = vb - va + vc + phi0_14
    v /= (np.linalg.norm(v) + 1e-20)
    sims = wmap_phi_lookup @ v
    for ew in exclude_words:
        if ew in wmap_w2local:
            sims[wmap_w2local[ew]] = -2.0
    top_k = np.argsort(sims)[-k:][::-1]
    return [wmap_words_ordered[i] for i in top_k], float(sims[top_k[0]])

print(f"\n  Analogy test: b − a + c ≈ d?  (top-5 results, star = correct answer)")
print(f"  {'Test':<35s}  {'Raw φ (top-1)':>20s}  {'Centred φ (top-1)':>20s}  Hit?")
print(f"  {'-'*90}")

raw_hits, cent_hits = 0, 0
results_table = []
for a, b, c, d_expected in ANALOGY_TESTS:
    excl = [a, b, c]
    raw_res,   raw_sim  = analogy_raw(a, b, c, excl)
    cent_res,  cent_sim = analogy_centred(a, b, c, excl)
    if raw_res is None:
        continue
    raw_top1  = raw_res[0]
    cent_top1 = cent_res[0]
    raw_hit   = (raw_top1  == d_expected)
    cent_hit  = (cent_top1 == d_expected)
    raw_hits  += int(raw_hit)
    cent_hits += int(cent_hit)
    flag = ("✓→✓" if raw_hit and cent_hit else
            "✗→✓" if not raw_hit and cent_hit else
            "✓→✗" if raw_hit and not cent_hit else "✗→✗")
    test_str = f"{a}-{b}+{c}={d_expected}"
    print(f"  {test_str:<35s}  {raw_top1:>20s}  {cent_top1:>20s}  {flag}")
    results_table.append({
        "a": a, "b": b, "c": c, "expected": d_expected,
        "raw_top1": raw_top1, "centred_top1": cent_top1,
        "raw_hit": bool(raw_hit), "centred_hit": bool(cent_hit),
    })
    # Also show top-5 for failures
    if not raw_hit or not cent_hit:
        raw_in_5  = d_expected in raw_res
        cent_in_5 = d_expected in cent_res
        print(f"    raw  top-5: {raw_res}  {'(answer in top-5)' if raw_in_5 else ''}")
        print(f"    cent top-5: {cent_res}  {'(answer in top-5)' if cent_in_5 else ''}")

n_tests = len(results_table)
print(f"\n  Raw φ:      {raw_hits}/{n_tests} correct ({100*raw_hits/max(n_tests,1):.0f}%)")
print(f"  Centred φ:  {cent_hits}/{n_tests} correct ({100*cent_hits/max(n_tests,1):.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 5 — Layer Shift: φ₀(L14) vs φ₀(L23)")
print(f"{'='*65}")

# Already computed φ0_14 and φ0_23 at top
# Now measure: how does the displacement of Zone C words change?
delta_c14_v2 = 1.0 - (phi_c14 @ phi0_14)
delta_c23_v2 = 1.0 - (phi_c23 @ phi0_23)

print(f"\n  Δ from φ₀ (Zone C words):")
print(f"    L14: mean={delta_c14_v2.mean():.4f}  std={delta_c14_v2.std():.4f}")
print(f"    L23: mean={delta_c23_v2.mean():.4f}  std={delta_c23_v2.std():.4f}")
print(f"    Δ(L23−L14): {delta_c23_v2.mean()-delta_c14_v2.mean():+.4f}")

# Are Zone C words displaced MORE from center at L23?
n_more_displaced = (delta_c23_v2 > delta_c14_v2).sum()
print(f"\n  Zone C words displaced MORE from φ₀ at L23: "
      f"{n_more_displaced}/{len(zone_c_words)} = {100*n_more_displaced/len(zone_c_words):.1f}%")
print(f"  → {'crystallisation: Zone C words move AWAY from center at L23 (as predicted)' if n_more_displaced > len(zone_c_words)*0.55 else 'no clear crystallisation trend'}")

# What about Zone D? Less displaced at L23 (ocean tightens)?
delta_d23_v2 = 1.0 - (phi_d23 @ phi0_23)
n_d_less_displaced = (delta_d23_v2 < delta_d14.flatten()[:len(delta_d23_v2)]).sum() if len(delta_d23_v2) == len(delta_d14) else 0
print(f"\n  Δ from φ₀ (Zone D words):")
print(f"    L14: mean={delta_d14.mean():.4f}  std={delta_d14.std():.4f}")
print(f"    L23: mean={delta_d23_v2.mean():.4f}  std={delta_d23_v2.std():.4f}")
print(f"    → {'ocean TIGHTENS (less displaced) at L23' if delta_d23_v2.mean() < delta_d14.mean() else 'ocean SPREADS (more displaced) at L23'}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 6 — Context Shift Vector from Day 29 Cache")
print(f"{'='*65}")

if os.path.exists(PN_CACHE):
    pn_npz   = np.load(PN_CACHE, allow_pickle=True)
    pn_words = list(pn_npz['words'])
    pn_hs14  = pn_npz['hs'].astype(np.float64)
    pn_phi   = batch_phi(pn_hs14, z2)
    pn_phi_c = pn_phi - phi0_14[None, :]
    pn_w2i   = {w: i for i, w in enumerate(pn_words)}

    # In isolation, proper nouns are near the degenerate pole
    # Measure displacement from φ₀ (vs Zone C words, Zone D words)
    delta_pn = 1.0 - (pn_phi @ phi0_14)
    print(f"\n  Proper nouns (Day 29 isolation cache):")
    print(f"    n = {len(pn_words)}  Δ from φ₀: mean={delta_pn.mean():.4f}  std={delta_pn.std():.4f}")
    print(f"    Compare Zone D: Δ={delta_d14.mean():.4f}  Zone C: Δ={delta_c14.mean():.4f}")

    # Context shift magnitude (from Day 30 findings):
    # Berlin in isolation: cos=0.997 to Zone A centroid → near pole
    # Berlin in context: cos=0.21 to Zone A centroid → in Zone C
    # The context shift vector magnitude in angular terms:
    pole_cos_isolation = 0.997
    pole_cos_context   = 0.21
    shift_magnitude    = float(np.arccos(np.clip(pole_cos_context, -1, 1)) -
                                np.arccos(np.clip(pole_cos_isolation, -1, 1)))
    print(f"\n  Context shift (Day 30 empirical):")
    print(f"    Proper noun isolation: cos to pole ≈ {pole_cos_isolation}")
    print(f"    Proper noun in context: cos to pole ≈ {pole_cos_context}")
    print(f"    Angular shift: {np.degrees(abs(shift_magnitude)):.1f}°")
    print(f"    This is the center shift: proper noun moves from degenerate pole baseline")
    print(f"    to a Zone C position — a displacement of {1-pole_cos_context:.2f} from the pole")

    # Can we estimate distance from φ₀ to degenerate pole?
    dist_phi0_to_pole = 1.0 - cos_z14_pole
    print(f"\n  Distance between φ₀ and degenerate pole:")
    print(f"    cos(φ₀, pole) = {cos_z14_pole:.4f}  →  Δ = {dist_phi0_to_pole:.4f}")
    print(f"    The context shift ({1-pole_cos_context:.2f}) is "
          f"{'much larger than' if (1-pole_cos_context) > dist_phi0_to_pole * 2 else 'comparable to'} "
          f"the φ₀-to-pole distance ({dist_phi0_to_pole:.4f})")
    print(f"    → context doesn't just move the word to φ₀; it moves it PAST φ₀ into Zone C")
else:
    print(f"\n  Day 29 cache not found; skipping proper noun context shift analysis")
    pn_words = []
    delta_pn = np.array([0.0])


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 7 — Summary: The Geometry of the Semantic Zero")
print(f"{'='*65}")

print(f"""
  φ₀ (Zone D centroid) properties:
    cos(φ₀(L14), φ₀(L23))   = {cos_z14_z23:.4f}   ← layer stability
    cos(φ₀, degenerate pole) = {cos_z14_pole:.4f}   ← vs default center
    |cos(φ₀, Z2)|            = {cos_z14_z2:.4f}   ← vs frequency axis

  Displacement from φ₀ by zone:
    Zone A/B (pole):  Δ = {delta_ab14.mean():.4f}  ← near center but not AT center
    Zone C (semantic): Δ = {delta_c14.mean():.4f}  ← displaced — has semantic content
    Zone D (ocean):   Δ = {delta_d14.mean():.4f}  ← AT center by definition

  Nearest-neighbour body purity:
    Raw φ:     {raw_purity.mean():.4f}
    Centred φ: {cent_purity.mean():.4f}  (Δ = {cent_purity.mean()-raw_purity.mean():+.4f})

  Analogy arithmetic:
    Raw φ:     {raw_hits}/{n_tests} correct
    Centred φ: {cent_hits}/{n_tests} correct
""")

# ── Save ─────────────────────────────────────────────────────────────────────
result = {
    "meta": {"experiment": "Day 34 — Semantic Zero and Centre Shift"},
    "phi0": {
        "cos_L14_L23":       float(cos_z14_z23),
        "cos_phi0_pole":     float(cos_z14_pole),
        "abs_cos_phi0_z2":   float(cos_z14_z2),
        "dist_phi0_to_pole": float(1.0 - cos_z14_pole),
    },
    "displacement": {
        "zone_ab_mean": float(delta_ab14.mean()),
        "zone_c_mean":  float(delta_c14.mean()),
        "zone_d_mean":  float(delta_d14.mean()),
        "zone_d_l23_mean": float(delta_d23_v2.mean()),
        "zone_c_l23_mean": float(delta_c23_v2.mean()),
        "spearman_tokenid": float(r_d_tid),
    },
    "nn_purity": {
        "raw":     float(raw_purity.mean()),
        "centred": float(cent_purity.mean()),
        "delta":   float(cent_purity.mean() - raw_purity.mean()),
    },
    "analogy": {
        "raw_hits":     raw_hits,
        "centred_hits": cent_hits,
        "n_tests":      n_tests,
        "results":      results_table,
    },
    "layer_shift": {
        "zone_c_more_displaced_at_l23_pct":
            float(100 * n_more_displaced / max(len(zone_c_words), 1)),
        "zone_c_delta_l14": float(delta_c14_v2.mean()),
        "zone_c_delta_l23": float(delta_c23_v2.mean()),
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 34 complete in {time.time()-t0:.1f}s")
