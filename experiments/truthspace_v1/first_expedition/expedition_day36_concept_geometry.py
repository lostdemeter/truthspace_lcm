#!/usr/bin/env python3
"""
Expedition Day 36 — The Geometry of a Concept

Three foundational mathematical questions before writing DC 315:

  Q1. Is a concept a POINT (body centroid), a DIRECTION (ray from φ₀),
      or a REGION with internal structure?
      → Measure within-body spread, PC1 variance, and whether the
        first within-body principal component correlates with
        interpretable word properties.

  Q2. Are relational concepts (Level 2) UNIVERSAL across bodies,
      or are they body-specific?
      → Cross-body held-out test: compute the mean relationship vector
        using pairs from BODY X only; apply it to words in BODY Y.
        If it generalises, "plural" / "gerund→past" / "adverb" are
        genuine universal concepts in φ-space — independent of
        body membership.

  Q3. What is the INTRINSIC DIMENSIONALITY of concept space?
      → SVD of the 95 body-centroid matrix.  If 95 bodies live in
        k << 95 dimensions, then there are only k fundamental concept
        axes, and the bodies are combinations of those axes.

  Bonus Q4. Can concepts be COMPOSED?
      → φ(body_A) + φ(body_B) → what word lands nearest the sum?
        If music_centroid + anatomy_centroid lands near a word that
        belongs to both domains, composition is geometric.
"""

import os, json
import numpy as np
from collections import defaultdict
from itertools import combinations

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day36_concept_geometry.json")

KILLING_PAIRS = [
    ('cat','cats'),('dog','dogs'),('tree','trees'),('bird','birds'),
    ('house','houses'),('man','woman'),('king','queen'),('boy','girl'),
    ('big','bigger'),('fast','faster'),('old','older'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def batch_phi(hs, z2):
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

def nn_search(query, phi_mat, words, exclude=None, k=5):
    q = query / (np.linalg.norm(query) + 1e-20)
    sims = phi_mat @ q
    for w in (exclude or []):
        if w in w2l:
            sims[w2l[w]] = -2.0
    top_k = np.argsort(sims)[-k:][::-1]
    return [(words[i], float(sims[i])) for i in top_k]

def rel_vec(w_a, w_b):
    """Unit relationship vector φ(b) − φ(a)."""
    v = wmap_phi[w2l[w_b]] - wmap_phi[w2l[w_a]]
    nm = np.linalg.norm(v)
    return v / nm if nm > 1e-20 else None

# ── Load ──────────────────────────────────────────────────────────────────────
print("── Load ───────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words = [w for w, v in wmap.items() if v['phase']==2
                and v.get('L14_body') not in ('B000','B001',None) and w in w2i]
zone_c_bodies = {w: wmap[w]['L14_body'] for w in zone_c_words}
zone_d_words  = [w for w, v in wmap.items() if v['phase']==2
                 and v.get('L14_body') == 'B000' and w in w2i]
body_label_map = {}
for w, v in wmap.items():
    b = v.get('L14_body')
    if b and b not in body_label_map:
        body_label_map[b] = v.get('L14_label', '?')

print(f"  Zone C: {len(zone_c_words)}  Zone D: {len(zone_d_words)}")

# Z2 axis
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
z2 = Vt[0] / np.linalg.norm(Vt[0])

# φ-vectors + φ₀
zone_c_idx   = np.array([w2i[w] for w in zone_c_words])
zone_d_idx   = np.array([w2i[w] for w in zone_d_words])
phi_c14      = batch_phi(hs14_all[zone_c_idx], z2)
phi_d14      = batch_phi(hs14_all[zone_d_idx], z2)
phi0_14      = phi_d14.mean(axis=0)
phi0_14     /= np.linalg.norm(phi0_14)

wmap_words   = [w for w in wmap.keys() if w in w2i]
wmap_idx     = np.array([w2i[w] for w in wmap_words])
wmap_phi     = batch_phi(hs14_all[wmap_idx], z2)
w2l          = {w: i for i, w in enumerate(wmap_words)}

# Body centroids
body_members   = defaultdict(list)   # body → list of indices into zone_c_words
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

body_centroids = {}
body_words_map = {}
for body, idxs in body_members.items():
    vecs = phi_c14[idxs]
    c = vecs.mean(axis=0)
    body_centroids[body] = c / (np.linalg.norm(c) + 1e-20)
    body_words_map[body] = [zone_c_words[i] for i in idxs]

print(f"  {len(body_centroids)} body centroids, φ₀ ready")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Q1 — Is a Concept a Point, a Direction, or a Region?")
print(f"{'='*65}")

# Q1a: Within-body spread and cohesion
# Spread = mean(1 - cos(word, centroid)) within body
print(f"\n  Q1a: Within-body spread and PC1 concentration")
print(f"  {'Body':<10s}  {'Label':<30s}  {'n':>4s}  {'Spread':>8s}  {'PC1%':>6s}  {'Gap':>6s}")
print(f"  {'-'*70}")
body_stats_q1 = []
for body, idxs in sorted(body_members.items(), key=lambda x: -len(x[1])):
    if len(idxs) < 5:
        continue
    vecs = phi_c14[idxs]
    c    = body_centroids[body]
    spread = float((1.0 - vecs @ c).mean())
    # PCA within body
    centered = vecs - c[None, :]
    _, sv_b, _ = np.linalg.svd(centered, full_matrices=False)
    total_var = float(np.sum(sv_b**2))
    pc1_pct   = float(sv_b[0]**2 / total_var * 100) if total_var > 0 else 0.0
    gap       = float((sv_b[0] - sv_b[1]) / (sv_b[1] + 1e-20)) if len(sv_b) > 1 else 0.0
    lbl = body_label_map.get(body, '?')[:30]
    print(f"  {body:<10s}  {lbl:<30s}  {len(idxs):>4d}  {spread:>8.4f}  {pc1_pct:>6.1f}%  {gap:>6.2f}")
    body_stats_q1.append({'body': body, 'n': len(idxs), 'spread': spread,
                          'pc1_pct': pc1_pct, 'sv1_sv2_gap': gap})

avg_spread = np.mean([x['spread'] for x in body_stats_q1])
avg_pc1    = np.mean([x['pc1_pct'] for x in body_stats_q1])
print(f"\n  Summary: mean within-body spread={avg_spread:.4f}  mean PC1%={avg_pc1:.1f}%")

# Q1b: Inter-body separation vs within-body spread
# "Concept clarity" = inter_body_distance / within_body_spread
# Compute mean pairwise BETWEEN centroids vs mean within spread
centroid_mat   = np.stack(list(body_centroids.values()))
gram           = centroid_mat @ centroid_mat.T
upper          = gram[np.triu_indices(len(centroid_mat), k=1)]
mean_inter_cos = float(upper.mean())
mean_inter_sep = float((1.0 - upper).mean())   # 1 - cos as "distance"
print(f"\n  Q1b: Inter-body vs within-body geometry")
print(f"    Mean inter-body separation (1-cos): {mean_inter_sep:.4f}")
print(f"    Mean within-body spread  (1-cos): {avg_spread:.4f}")
ratio = mean_inter_sep / avg_spread if avg_spread > 0 else 0
print(f"    Separation / Spread ratio: {ratio:.2f}  "
      f"({'concepts are well-separated' if ratio > 3 else 'concepts overlap substantially'})")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Q2 — Cross-Body Relationship Universality")
print(f"{'='*65}")

# Auto-discover morphological pairs as in Day 35
zone_c_set = set(zone_c_words)
MORPHO_RULES = [
    ('er',  'est',  'comparative→superlative'),
    ('',    'er',   'base→comparative'),
    ('',    's',    'singular→plural'),
    ('',    'es',   'singular→plural'),
    ('ing', 'ed',   'gerund→past'),
    ('',    'ly',   'base→adverb'),
    ('al',  'ally', 'adjective→adverb'),
    ('ic',  'ically','adjective→adverb'),
]

def strip_and_match(w, suf_a, suf_b):
    if not suf_b or not w.endswith(suf_b):
        return None
    base_plus_a = w[:-len(suf_b)] + suf_a
    return base_plus_a if base_plus_a != w else None

pairs_by_relation = defaultdict(list)  # relation → [(a, b, body)]
for w_b in zone_c_words:
    for suf_a, suf_b, label in MORPHO_RULES:
        w_a = strip_and_match(w_b, suf_a, suf_b)
        if w_a and w_a in zone_c_set and zone_c_bodies.get(w_a) == zone_c_bodies.get(w_b):
            pairs_by_relation[label].append((w_a, w_b, zone_c_bodies[w_a]))

print(f"\n  Cross-body held-out test:")
print(f"  Compute mean relationship vector from SOURCE body, apply to TARGET body")
print(f"  (tests whether the relationship is universal or body-specific)\n")

cross_body_results = {}
for label, pairs in sorted(pairs_by_relation.items(), key=lambda x: -len(x[1])):
    # Group pairs by body
    by_body = defaultdict(list)
    for w_a, w_b, body in pairs:
        if w_a in w2l and w_b in w2l:
            by_body[body].append((w_a, w_b))

    bodies_with_pairs = [b for b, ps in by_body.items() if len(ps) >= 2]
    if len(bodies_with_pairs) < 2:
        continue

    print(f"  Relation: {label}  ({len(pairs)} pairs across {len(by_body)} bodies)")
    print(f"  {'Source body':<12s}  {'Target body':<12s}  {'Source':>6s}  {'Target n':>8s}  "
          f"{'Top1':>5s}  {'Top5':>5s}  {'Example'}")
    print(f"  {'-'*80}")

    rel_results = []
    for src_body in bodies_with_pairs:
        src_pairs = by_body[src_body]
        # Mean relationship vector from source
        vecs = [rel_vec(a, b) for a, b in src_pairs if rel_vec(a, b) is not None]
        if not vecs:
            continue
        mean_rv = np.stack(vecs).mean(axis=0)
        mean_rv /= (np.linalg.norm(mean_rv) + 1e-20)

        for tgt_body in bodies_with_pairs:
            if tgt_body == src_body:
                continue
            tgt_pairs = by_body[tgt_body]
            top1, top5 = 0, 0
            ex_str = ''
            for w_a, w_b in tgt_pairs:
                if w_a not in w2l:
                    continue
                query = wmap_phi[w2l[w_a]] + mean_rv
                res = nn_search(query, wmap_phi, wmap_words, exclude=[w_a], k=5)
                top_words = [r[0] for r in res]
                if top_words[0] == w_b:
                    top1 += 1
                if w_b in top_words:
                    top5 += 1
                if not ex_str:
                    ex_str = f"{w_a}+vec→{top_words[0]} (want {w_b})"
            n = len(tgt_pairs)
            src_lbl = body_label_map.get(src_body, '?')[:10]
            tgt_lbl = body_label_map.get(tgt_body, '?')[:10]
            print(f"  {src_lbl:<12s}  {tgt_lbl:<12s}  {len(src_pairs):>6d}  {n:>8d}  "
                  f"{top1:>5d}  {top5:>5d}  {ex_str[:40]}")
            rel_results.append({'label': label, 'src': src_body, 'tgt': tgt_body,
                                 'src_n': len(src_pairs), 'tgt_n': n,
                                 'top1': top1, 'top5': top5})
    cross_body_results[label] = rel_results
    # Summary for this relation
    if rel_results:
        total_pairs_tested = sum(r['tgt_n'] for r in rel_results)
        total_top1 = sum(r['top1'] for r in rel_results)
        total_top5 = sum(r['top5'] for r in rel_results)
        pct1 = 100*total_top1/total_pairs_tested if total_pairs_tested else 0
        pct5 = 100*total_top5/total_pairs_tested if total_pairs_tested else 0
        print(f"  → Cross-body total: {total_top1}/{total_pairs_tested} top-1 ({pct1:.0f}%)  "
              f"{total_top5}/{total_pairs_tested} top-5 ({pct5:.0f}%)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Q3 — Intrinsic Dimensionality of Concept Space")
print(f"{'='*65}")

# SVD of the body-centroid matrix
# Rows = body centroids (unit vectors in φ-space)
# Columns = φ-space dimensions
C = centroid_mat  # shape (n_bodies, D)
_, sv_c, Vt_c = np.linalg.svd(C, full_matrices=False)

total_var = np.sum(sv_c**2)
cumvar    = np.cumsum(sv_c**2) / total_var
k_90 = int(np.searchsorted(cumvar, 0.90)) + 1
k_95 = int(np.searchsorted(cumvar, 0.95)) + 1
k_99 = int(np.searchsorted(cumvar, 0.99)) + 1

print(f"\n  Body-centroid SVD: {len(sv_c)} singular values over {len(body_centroids)} bodies")
print(f"    Variance explained:")
for k in [1, 2, 3, 5, 10, 20, 30, 50]:
    pct = float(cumvar[min(k-1, len(cumvar)-1)]) * 100
    print(f"      Top-{k:>2d}: {pct:.1f}%")
print(f"    Dimensions for 90% variance: {k_90}")
print(f"    Dimensions for 95% variance: {k_95}")
print(f"    Dimensions for 99% variance: {k_99}")
print(f"    Effective rank (∑sᵢ)²/∑sᵢ²: {float(np.sum(sv_c))**2 / float(np.sum(sv_c**2)):.1f}")

# Inspect top concept axes: what do they separate?
print(f"\n  Top concept axes (first 5 singular vectors):")
for axis in range(min(5, len(Vt_c))):
    # Project each body onto this axis
    proj   = C @ Vt_c[axis]
    top3   = np.argsort(proj)[-3:][::-1]
    bot3   = np.argsort(proj)[:3]
    bodies_list = list(body_centroids.keys())
    top3_lbl = [body_label_map.get(bodies_list[i],'?')[:20] for i in top3]
    bot3_lbl = [body_label_map.get(bodies_list[i],'?')[:20] for i in bot3]
    pct_axis  = float(sv_c[axis]**2 / total_var * 100)
    print(f"  Axis {axis+1} ({pct_axis:.1f}%):")
    print(f"    +end: {' | '.join(top3_lbl)}")
    print(f"    -end: {' | '.join(bot3_lbl)}")

# Effective dimensionality: number of dims needed to distinguish any two bodies
# Using nearest-neighbour accuracy in reduced dimensionality
print(f"\n  NN body-retrieval accuracy in reduced concept space:")
print(f"  {'Dims':>4s}  {'Accuracy':>10s}")
for k_test in [1, 2, 3, 5, 10, 20, 50, len(sv_c)]:
    k_test = min(k_test, len(sv_c))
    # Project body centroids into k_test dims
    C_low = C @ Vt_c[:k_test].T   # shape (n_bodies, k_test)
    gram_low = C_low @ C_low.T
    # For each body, its nearest neighbour should be itself (trivially true)
    # More useful: for each WORD in zone_c, project its φ-vector into the k_test-dim space
    # and find the nearest body centroid. Count correct body assignments.
    phi_c_low = phi_c14 @ Vt_c[:k_test].T  # (n_zone_c, k_test)
    C_low_norm = C_low / (np.linalg.norm(C_low, axis=1, keepdims=True) + 1e-20)
    phi_c_low_norm = phi_c_low / (np.linalg.norm(phi_c_low, axis=1, keepdims=True) + 1e-20)
    sims_low   = phi_c_low_norm @ C_low_norm.T   # (n_zone_c, n_bodies)
    pred_body  = np.argmax(sims_low, axis=1)
    bodies_list_arr = list(body_centroids.keys())
    correct = sum(1 for i, w in enumerate(zone_c_words)
                  if bodies_list_arr[pred_body[i]] == zone_c_bodies[w])
    acc = 100 * correct / len(zone_c_words)
    print(f"  {k_test:>4d}  {acc:>10.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Q4 — Can Concepts Be Composed?")
print(f"{'='*65}")

# Test: φ(body_A_centroid) + φ(body_B_centroid) → nearest words
# Check: are the nearest words plausibly "at the intersection" of both bodies?

# Pick interesting body pairs to compose
body_list_full = sorted(body_centroids.keys())
print(f"\n  Testing: body_A + body_B → what word lands nearest the sum?")
print(f"  {'Body A':<20s}  {'Body B':<20s}  {'Top-3 words at intersection'}")
print(f"  {'-'*75}")

# Pick body pairs with some semantic plausibility
composition_tests = []
# Grab pairs by finding bodies whose centroids are moderately close (not too close, not too far)
body_pairs_checked = 0
interesting_pairs  = []

for bi, bj in combinations(body_list_full[:30], 2):
    cos_ij = float(body_centroids[bi] @ body_centroids[bj])
    if 0.65 < cos_ij < 0.80:   # moderately related
        interesting_pairs.append((bi, bj, cos_ij))

interesting_pairs.sort(key=lambda x: -x[2])

for bi, bj, cos_ij in interesting_pairs[:15]:
    comp = body_centroids[bi] + body_centroids[bj]
    comp /= (np.linalg.norm(comp) + 1e-20)
    res = nn_search(comp, wmap_phi, wmap_words, exclude=[], k=3)
    top3 = [r[0] for r in res]
    top3_bodies = [wmap.get(w, {}).get('L14_body', '?') for w in top3]
    # Are the top words from either body?
    in_A = sum(1 for b in top3_bodies if b == bi)
    in_B = sum(1 for b in top3_bodies if b == bj)
    lbl_A = body_label_map.get(bi,'?')[:20]
    lbl_B = body_label_map.get(bj,'?')[:20]
    words_str = ', '.join(f"{w}({body_label_map.get(b,'?')[:8]})"
                          for w, b in zip(top3, top3_bodies))
    print(f"  {lbl_A:<20s}  {lbl_B:<20s}  {words_str}")
    composition_tests.append({'body_A': bi, 'body_B': bj, 'cos': cos_ij,
                               'top3': top3, 'top3_bodies': top3_bodies})

# Also test some intentional "semantic compositions"
print(f"\n  Intentional compositions (bodies with known semantic overlap):")
# Find bodies that might compose meaningfully
target_combos = [
    # Combining "action" bodies with "domain" bodies
    ('B003', 'B009'),  # action + political
    ('B003', 'B011'),  # action + vegetables (should give nonsense → negative control)
    ('B015', 'B013'),  # family + body parts
    ('B007', 'B006'),  # sports/leisure + nature
]
for bi, bj in target_combos:
    if bi not in body_centroids or bj not in body_centroids:
        continue
    comp = body_centroids[bi] + body_centroids[bj]
    comp /= (np.linalg.norm(comp) + 1e-20)
    res = nn_search(comp, wmap_phi, wmap_words, exclude=[], k=5)
    top5 = [r[0] for r in res]
    top5_bodies = [wmap.get(w, {}).get('L14_body', '?') for w in top5]
    lbl_A = body_label_map.get(bi,'?')[:22]
    lbl_B = body_label_map.get(bj,'?')[:22]
    in_A = sum(1 for b in top5_bodies[:3] if b == bi)
    in_B = sum(1 for b in top5_bodies[:3] if b == bj)
    print(f"  {bi}({lbl_A}) + {bj}({lbl_B})")
    print(f"    cos(A,B)={float(body_centroids[bi]@body_centroids[bj]):.3f}  "
          f"top-5: {top5}  in_A={in_A}/3  in_B={in_B}/3")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY — What is a Concept?")
print(f"{'='*65}")

print(f"\n  Q1 (Point vs Region):")
print(f"    Within-body spread:  {avg_spread:.4f} (1-cos from centroid)")
print(f"    Inter-body separation: {mean_inter_sep:.4f} (1-cos between centroids)")
print(f"    Sep/Spread ratio: {ratio:.2f}")
print(f"    Mean within-body PC1%: {avg_pc1:.1f}%")
if ratio > 5:
    print(f"    → Concepts are well-separated POINTS (ratio > 5)")
elif ratio > 2:
    print(f"    → Concepts are REGIONS with clear boundaries (2 < ratio ≤ 5)")
else:
    print(f"    → Concepts OVERLAP substantially (ratio ≤ 2)")

print(f"\n  Q3 (Intrinsic Dimensionality):")
print(f"    {len(body_centroids)} concepts live in ≈{k_95} effective dimensions (95% var)")
print(f"    Effective rank: {float(np.sum(sv_c))**2/float(np.sum(sv_c**2)):.1f}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "meta": {"experiment": "Day 36 — The Geometry of a Concept"},
    "q1_point_vs_region": {
        "mean_within_body_spread": float(avg_spread),
        "mean_inter_body_separation": float(mean_inter_sep),
        "separation_spread_ratio": float(ratio),
        "mean_pc1_pct": float(avg_pc1),
        "body_stats": body_stats_q1,
    },
    "q2_cross_body_universality": cross_body_results,
    "q3_intrinsic_dimensionality": {
        "n_bodies": len(body_centroids),
        "dims_for_90pct": k_90,
        "dims_for_95pct": k_95,
        "dims_for_99pct": k_99,
        "effective_rank": float(float(np.sum(sv_c))**2 / float(np.sum(sv_c**2))),
        "singular_values": sv_c[:20].tolist(),
        "cumvar": cumvar[:30].tolist(),
    },
    "q4_composition": composition_tests[:15],
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 36 complete.")
