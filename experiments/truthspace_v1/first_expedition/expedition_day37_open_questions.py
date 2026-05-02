#!/usr/bin/env python3
"""
Expedition Day 37 — Open Questions from DC 315

OQ1. Are there semantic Type 2 operators beyond morphology?
     Test: gender direction (within B015 family), antonym/degree direction
     within B012 comparatives, cross-body tests.

OQ2. Is Type 1 concept space hierarchical?
     Hierarchical clustering of 95 body centroids at k=2,4,8.
     What does the tree look like?

OQ3. Are Type 2 vectors orthogonal to Type 1 concept subspace?
     Project known Type 2 vectors (r_plural, r_adverb, r_gerund)
     onto the Type 1 subspace spanned by the top-43 concept axes.
     Report ||projection||² — how much Type 2 lives in Type 1 space.

OQ4. Can Type 2 operators be auto-discovered from within-body
     pairwise differences?
     Stack all within-body φ(b)−φ(a) vectors, SVD to find top
     directions, compare to known Type 2 vectors.

OQ5. Is ~43 effective dimensions stable?
     Bootstrap resampling of body centroids. Also report the
     prediction for Qwen2-7B based on what we know.
"""

import os, json
import numpy as np
from collections import defaultdict
from itertools import combinations

try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day37_open_questions.json")

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
    if w_a not in w2l or w_b not in w2l:
        return None
    v = wmap_phi[w2l[w_b]] - wmap_phi[w2l[w_a]]
    nm = np.linalg.norm(v)
    return v / nm if nm > 1e-20 else None

def mean_rel_vec(pairs):
    vecs = [rel_vec(a, b) for a, b in pairs if rel_vec(a, b) is not None]
    if not vecs:
        return None
    m = np.stack(vecs).mean(axis=0)
    nm = np.linalg.norm(m)
    return m / nm if nm > 1e-20 else None

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
_, _, Vt = np.linalg.svd(D, full_matrices=False)
z2 = Vt[0] / np.linalg.norm(Vt[0])

# φ-vectors
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

body_members   = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

body_centroids = {}
body_words_map = {}
for body, idxs in body_members.items():
    vecs = phi_c14[idxs]
    c = vecs.mean(axis=0)
    body_centroids[body] = c / (np.linalg.norm(c) + 1e-20)
    body_words_map[body] = [zone_c_words[i] for i in idxs]

# Type 1 subspace: top-43 SVD directions of body centroid matrix
centroid_mat = np.stack([body_centroids[b] for b in sorted(body_centroids)])
bodies_list  = sorted(body_centroids.keys())
_, sv_c, Vt_c = np.linalg.svd(centroid_mat, full_matrices=False)
total_var_c = float(np.sum(sv_c**2))
cumvar_c    = np.cumsum(sv_c**2) / total_var_c
k95 = int(np.searchsorted(cumvar_c, 0.95)) + 1
T1_basis = Vt_c[:k95]   # shape (k95, D) — the Type 1 concept subspace

zone_c_set = set(zone_c_words)

print(f"  Zone C: {len(zone_c_words)}  Zone D: {len(zone_d_words)}")
print(f"  {len(body_centroids)} bodies, T1 subspace dim={k95}")
print(f"  wmap words with φ: {len(wmap_words)}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ1 — Semantic Type 2 Operators Beyond Morphology")
print(f"{'='*65}")

# ── OQ1a: Gender direction in the family body ─────────────────────────────────
# Gender pairs: masc→fem, all within B015
gender_pairs_source = [
    ('grandfather', 'grandmother'),
    ('uncle', 'aunt'),
]
gender_pairs_target = [
    ('nephew', 'niece'),
    ('brother', 'sister'),
    ('grandson', 'granddaughter'),
    ('husband', 'wife'),
]
# Also try space-prefixed variants
def find_pair(a, b):
    for pa in [a, ' '+a, '▁'+a]:
        for pb in [b, ' '+b, '▁'+b]:
            if pa in w2l and pb in w2l:
                return (pa, pb)
    return None

gender_src = [find_pair(a, b) for a, b in gender_pairs_source]
gender_src = [p for p in gender_src if p]
gender_tgt = [find_pair(a, b) for a, b in gender_pairs_target]
gender_tgt = [p for p in gender_tgt if p]

print(f"\n  OQ1a: Gender direction (masculine → feminine)")
print(f"  Source pairs found: {[(a.strip(), b.strip()) for a,b in gender_src]}")
print(f"  Target pairs found: {[(a.strip(), b.strip()) for a,b in gender_tgt]}")

oq1a_results = []
if gender_src:
    r_gender = mean_rel_vec(gender_src)
    if r_gender is not None:
        # Consistency among source pairs
        src_vecs = [rel_vec(a, b) for a, b in gender_src]
        src_vecs = [v for v in src_vecs if v is not None]
        if len(src_vecs) >= 2:
            src_cos = float(np.mean([src_vecs[i] @ src_vecs[j]
                                     for i in range(len(src_vecs))
                                     for j in range(i+1, len(src_vecs))]))
        else:
            src_cos = float('nan')
        print(f"  Source pairwise cos: {src_cos:.3f}")
        print(f"  {'Pair':<30s}  {'Predicted':>12s}  {'Actual':>12s}  {'Top-1':>6s}  {'Top-5':>6s}")
        print(f"  {'-'*70}")
        top1, top5 = 0, 0
        for wa, wb in gender_tgt:
            query = wmap_phi[w2l[wa]] + r_gender
            res = nn_search(query, wmap_phi, wmap_words, exclude=[wa], k=5)
            top_words = [r[0].strip() for r in res]
            predicted = top_words[0]
            actual    = wb.strip()
            t1 = int(predicted == actual)
            t5 = int(actual in top_words)
            top1 += t1; top5 += t5
            marker = '✓' if t1 else ('~' if t5 else '✗')
            print(f"  {wa.strip()}+gender→{predicted:<12s}  (want {actual:<12s})  {marker}")
            oq1a_results.append({'pair': (wa.strip(), wb.strip()), 'predicted': predicted,
                                   'top1': t1, 'top5': t5})
        print(f"  Gender top-1: {top1}/{len(gender_tgt)}  top-5: {top5}/{len(gender_tgt)}")
        # Is gender direction cross-body?
        print(f"\n  Is gender direction cross-body? (apply to non-family words)")
        cross_gender_tests = [
            ('actor', 'actress'), ('waiter', 'waitress'), ('prince', 'princess'),
            ('lion', 'lioness'), ('king', 'queen'), ('emperor', 'empress'),
        ]
        cross_found = [find_pair(a, b) for a, b in cross_gender_tests]
        cross_found = [p for p in cross_found if p]
        print(f"  Cross-domain pairs found: {[(a.strip(), b.strip()) for a,b in cross_found]}")
        cross_top1 = 0
        for wa, wb in cross_found:
            query = wmap_phi[w2l[wa]] + r_gender
            res = nn_search(query, wmap_phi, wmap_words, exclude=[wa], k=5)
            top_words = [r[0].strip() for r in res]
            t1 = int(top_words[0] == wb.strip())
            t5 = int(wb.strip() in top_words)
            cross_top1 += t1
            marker = '✓' if t1 else ('~' if t5 else '✗')
            print(f"  {wa.strip()}+gender→{top_words[0]:<12s}  (want {wb.strip():<12s}) {marker}")

# ── OQ1b: Antonym direction within Comparative Adjectives body ────────────────
print(f"\n  OQ1b: Antonym/opposite direction (big↔small, fast↔slow, old↔young)")
antonym_source = [
    ('bigger', 'smaller'), ('larger', 'smaller'), ('faster', 'slower'),
    ('taller', 'shorter'), ('heavier', 'lighter'),
]
antonym_target = [
    ('older', 'younger'), ('stronger', 'weaker'), ('deeper', 'shallower'),
    ('louder', 'quieter'), ('richer', 'poorer'),
]
ant_src = [find_pair(a, b) for a, b in antonym_source]
ant_src = [p for p in ant_src if p]
ant_tgt = [find_pair(a, b) for a, b in antonym_target]
ant_tgt = [p for p in ant_tgt if p]

print(f"  Source pairs found: {[(a.strip(), b.strip()) for a,b in ant_src]}")
oq1b_results = []
if ant_src:
    r_antonym = mean_rel_vec(ant_src)
    if r_antonym is not None:
        src_vecs = [rel_vec(a, b) for a, b in ant_src if rel_vec(a, b) is not None]
        if len(src_vecs) >= 2:
            src_cos = float(np.mean([src_vecs[i] @ src_vecs[j]
                                     for i in range(len(src_vecs))
                                     for j in range(i+1, len(src_vecs))]))
        else:
            src_cos = float('nan')
        print(f"  Source pairwise cos: {src_cos:.3f}")
        top1, top5 = 0, 0
        for wa, wb in ant_tgt:
            query = wmap_phi[w2l[wa]] + r_antonym
            res = nn_search(query, wmap_phi, wmap_words, exclude=[wa], k=5)
            top_words = [r[0].strip() for r in res]
            t1 = int(top_words[0] == wb.strip())
            t5 = int(wb.strip() in top_words)
            top1 += t1; top5 += t5
            marker = '✓' if t1 else ('~' if t5 else '✗')
            print(f"  {wa.strip()}+antonym→{top_words[0]:<12s}  (want {wb.strip():<12s}) {marker}")
            oq1b_results.append({'pair': (wa.strip(), wb.strip()), 'top1': t1, 'top5': t5})
        print(f"  Antonym top-1: {top1}/{len(ant_tgt)}  top-5: {top5}/{len(ant_tgt)}")

# ── OQ1c: Comparative→Superlative direction ───────────────────────────────────
print(f"\n  OQ1c: Comparative → Superlative direction")
comp_sup_source = [
    ('bigger', 'biggest'), ('faster', 'fastest'), ('taller', 'tallest'),
    ('stronger', 'strongest'), ('older', 'oldest'),
]
comp_sup_target = [
    ('smaller', 'smallest'), ('slower', 'slowest'), ('shorter', 'shortest'),
    ('younger', 'youngest'), ('deeper', 'deepest'), ('wider', 'widest'),
]
cs_src = [find_pair(a, b) for a, b in comp_sup_source]
cs_src = [p for p in cs_src if p]
cs_tgt = [find_pair(a, b) for a, b in comp_sup_target]
cs_tgt = [p for p in cs_tgt if p]

print(f"  Source pairs found: {[(a.strip(), b.strip()) for a,b in cs_src]}")
oq1c_results = []
if cs_src:
    r_comp_sup = mean_rel_vec(cs_src)
    if r_comp_sup is not None:
        src_vecs = [rel_vec(a, b) for a, b in cs_src if rel_vec(a, b) is not None]
        if len(src_vecs) >= 2:
            src_cos = float(np.mean([src_vecs[i] @ src_vecs[j]
                                     for i in range(len(src_vecs))
                                     for j in range(i+1, len(src_vecs))]))
        else:
            src_cos = float('nan')
        print(f"  Source pairwise cos: {src_cos:.3f}")
        top1, top5 = 0, 0
        for wa, wb in cs_tgt:
            query = wmap_phi[w2l[wa]] + r_comp_sup
            res = nn_search(query, wmap_phi, wmap_words, exclude=[wa], k=5)
            top_words = [r[0].strip() for r in res]
            t1 = int(top_words[0] == wb.strip())
            t5 = int(wb.strip() in top_words)
            top1 += t1; top5 += t5
            marker = '✓' if t1 else ('~' if t5 else '✗')
            print(f"  {wa.strip()}→{top_words[0]:<12s}  (want {wb.strip():<12s}) {marker}")
            oq1c_results.append({'pair': (wa.strip(), wb.strip()), 'top1': t1, 'top5': t5})
        print(f"  Comp→Sup top-1: {top1}/{len(cs_tgt)}  top-5: {top5}/{len(cs_tgt)}")

# ── OQ1d: Do all known Type 2 vectors form an orthogonal basis? ───────────────
print(f"\n  OQ1d: Pairwise cosines between known Type 2 vectors")
t2_vectors = {}

# r_plural (from Day 35/36)
MORPHO_SOURCE = [
    ('brother', 'brothers'), ('sister', 'sisters'), ('cousin', 'cousins'),
    ('vegetable', 'vegetables'), ('potato', 'potatoes'), ('tomato', 'tomatoes'),
    ('kidney', 'kidneys'), ('partnership', 'partnerships'),
]
pairs_found = [find_pair(a, b) for a, b in MORPHO_SOURCE]
pairs_found = [p for p in pairs_found if p]
rv = mean_rel_vec(pairs_found)
if rv is not None: t2_vectors['plural'] = rv

# r_adverb
ADV_SOURCE = [
    ('respective', 'respectively'), ('unfortunate', 'unfortunately'),
    ('subsequent', 'subsequently'), ('intentional', 'intentionally'),
    ('drastic', 'drastically'), ('genetic', 'genetically'),
]
pairs_found = [find_pair(a, b) for a, b in ADV_SOURCE]
pairs_found = [p for p in pairs_found if p]
rv = mean_rel_vec(pairs_found)
if rv is not None: t2_vectors['adverb'] = rv

# r_gerund_to_past
GER_SOURCE = [
    ('deciding', 'decided'), ('killing', 'killed'), ('welcoming', 'welcomed'),
    ('crushing', 'crushed'),
]
pairs_found = [find_pair(a, b) for a, b in GER_SOURCE]
pairs_found = [p for p in pairs_found if p]
rv = mean_rel_vec(pairs_found)
if rv is not None: t2_vectors['gerund_to_past'] = rv

# Add new semantic directions if they worked above
if oq1a_results:
    r_gender_final = mean_rel_vec(gender_src)
    if r_gender_final is not None:
        t2_vectors['gender_masc_to_fem'] = r_gender_final
if oq1c_results:
    r_cs_final = mean_rel_vec(cs_src)
    if r_cs_final is not None:
        t2_vectors['comp_to_sup'] = r_cs_final
if oq1b_results and ant_src:
    r_ant_final = mean_rel_vec(ant_src)
    if r_ant_final is not None:
        t2_vectors['antonym'] = r_ant_final

t2_names = list(t2_vectors.keys())
print(f"\n  Type 2 vectors assembled: {t2_names}")
if len(t2_names) >= 2:
    print(f"  {'':20s}  " + "  ".join(f"{n[:10]:>10s}" for n in t2_names))
    t2_cos_mat = {}
    for i, ni in enumerate(t2_names):
        row = []
        for j, nj in enumerate(t2_names):
            c = float(t2_vectors[ni] @ t2_vectors[nj])
            row.append(c)
        print(f"  {ni[:20]:20s}  " + "  ".join(f"{c:>10.3f}" for c in row))
        t2_cos_mat[ni] = {nj: row[j] for j, nj in enumerate(t2_names)}


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ2 — Hierarchical Clustering of Type 1 Concept Space")
print(f"{'='*65}")

# Cosine distance matrix between body centroids
cos_dist = 1.0 - (centroid_mat @ centroid_mat.T)
cos_dist = np.clip(cos_dist, 0, 2)
# Force symmetry
cos_dist = (cos_dist + cos_dist.T) / 2.0
np.fill_diagonal(cos_dist, 0.0)

oq2_clusters = {}
if HAS_SCIPY:
    dist_condensed = squareform(cos_dist, checks=False)
    Z_link = linkage(dist_condensed, method='ward')

    print(f"\n  Hierarchical clustering (Ward's method) of {len(bodies_list)} bodies")
    for k in [2, 3, 4, 6, 8]:
        labels = fcluster(Z_link, t=k, criterion='maxclust')
        clusters = defaultdict(list)
        for i, lbl in enumerate(labels):
            clusters[int(lbl)].append(bodies_list[i])
        print(f"\n  k={k} clusters:")
        cluster_info = []
        for cl_id in sorted(clusters.keys()):
            members = clusters[cl_id]
            lbls = [body_label_map.get(b,'?')[:25] for b in members]
            print(f"    Cluster {cl_id} ({len(members)} bodies):")
            # Show top 5 body labels
            for lbl in sorted(lbls)[:6]:
                print(f"      {lbl}")
            if len(lbls) > 6:
                print(f"      ... +{len(lbls)-6} more")
            cluster_info.append({'cluster': cl_id, 'bodies': members,
                                  'labels': lbls})
        oq2_clusters[k] = cluster_info
else:
    print("  scipy not available, using manual greedy merge")
    # Simple greedy: find most similar pair, report
    print(f"  Most similar body pairs (top-10):")
    triu = [(float(cos_dist[i,j]), i, j)
            for i in range(len(bodies_list)) for j in range(i+1, len(bodies_list))]
    triu.sort()
    for dist, i, j in triu[:10]:
        li = body_label_map.get(bodies_list[i],'?')[:25]
        lj = body_label_map.get(bodies_list[j],'?')[:25]
        print(f"    cos_dist={dist:.3f}  {li} ↔ {lj}")

# Axis 1 analysis: what does the dominant concept axis separate?
print(f"\n  Axis 1 (56.6% variance) — projection of each body onto dominant concept direction")
body_axis1_proj = centroid_mat @ Vt_c[0]
sorted_by_axis1 = np.argsort(body_axis1_proj)
print(f"  Most NEGATIVE (−end, specific/domain-locked):")
for i in sorted_by_axis1[:8]:
    print(f"    {body_axis1_proj[i]:+.3f}  {body_label_map.get(bodies_list[i],'?')[:40]}")
print(f"  Most POSITIVE (+end, abstract/broadly-applicable):")
for i in sorted_by_axis1[-8:][::-1]:
    print(f"    {body_axis1_proj[i]:+.3f}  {body_label_map.get(bodies_list[i],'?')[:40]}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ3 — Type 2 Vectors ⊥ Type 1 Concept Subspace?")
print(f"{'='*65}")

print(f"\n  Type 1 subspace: top {k95} SVD directions (95% concept variance)")
print(f"  Test: project each Type 2 vector onto T1 subspace; measure ||proj||²\n")

oq3_results = {}
print(f"  {'Type 2 vector':<25s}  {'||T1 proj||²':>14s}  {'||residual||²':>14s}  {'cos(v,proj)':>12s}  {'Interpretation'}")
print(f"  {'-'*85}")
for name, rv in t2_vectors.items():
    # Project rv onto T1 subspace
    # proj = Vt_c[:k95].T @ (Vt_c[:k95] @ rv)
    coords   = T1_basis @ rv        # (k95,) — coordinates in T1 space
    proj_vec = T1_basis.T @ coords  # back in full φ-space
    proj_norm_sq = float(np.dot(proj_vec, proj_vec))
    resid_vec    = rv - proj_vec
    resid_norm_sq = float(np.dot(resid_vec, resid_vec))
    # rv is unit vector, so proj_norm_sq + resid_norm_sq should ≈ 1
    total = proj_norm_sq + resid_norm_sq
    proj_frac  = proj_norm_sq / total if total > 0 else 0.0
    resid_frac = resid_norm_sq / total if total > 0 else 0.0
    # cos between rv and its projection
    cos_rp = float(np.dot(rv, proj_vec) / (np.linalg.norm(proj_vec) + 1e-20))
    if proj_frac < 0.10:
        interp = "ORTHOGONAL to T1"
    elif proj_frac < 0.30:
        interp = "Mostly outside T1"
    elif proj_frac < 0.60:
        interp = "Partial overlap"
    else:
        interp = "Mostly inside T1"
    print(f"  {name:<25s}  {proj_frac:>14.3f}  {resid_frac:>14.3f}  {cos_rp:>12.3f}  {interp}")
    oq3_results[name] = {'proj_frac': proj_frac, 'resid_frac': resid_frac,
                          'cos_to_proj': cos_rp}

# Also test: where does each Type 2 vector land in T1 space?
# Report: which body centroids have highest projection onto each Type 2 direction
print(f"\n  Which bodies lie most along each Type 2 direction?")
for name, rv in t2_vectors.items():
    body_proj = centroid_mat @ rv  # (n_bodies,)
    top3_idx  = np.argsort(body_proj)[-3:][::-1]
    bot3_idx  = np.argsort(body_proj)[:3]
    top3_lbl  = [f"{body_label_map.get(bodies_list[i],'?')[:20]}({body_proj[i]:+.2f})" for i in top3_idx]
    bot3_lbl  = [f"{body_label_map.get(bodies_list[i],'?')[:20]}({body_proj[i]:+.2f})" for i in bot3_idx]
    print(f"  {name[:22]:22s}: +{' | '.join(top3_lbl[:2])}")
    print(f"  {'':22s}  -{' | '.join(bot3_lbl[:2])}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ4 — Auto-Discover Type 2 from Within-Body Pairwise Differences")
print(f"{'='*65}")

print(f"\n  Stack all within-body φ(b)−φ(a) vectors, then SVD")
print(f"  Hypothesis: top SVD directions = known Type 2 operators")

# Collect all within-body differences
diff_vecs = []
diff_labels = []  # (body, wa, wb)
for body, idxs in body_members.items():
    if len(idxs) < 3:
        continue
    vecs = phi_c14[idxs]
    ws   = [zone_c_words[i] for i in idxs]
    for i in range(len(idxs)):
        for j in range(len(idxs)):
            if i == j: continue
            d = vecs[j] - vecs[i]
            nm = np.linalg.norm(d)
            if nm > 0.05:   # filter trivial differences
                diff_vecs.append(d / nm)
                diff_labels.append((body, ws[i], ws[j]))

print(f"  Total within-body difference vectors: {len(diff_vecs)}")
if diff_vecs:
    D_mat = np.stack(diff_vecs)  # (N, dim)
    _, sv_d, Vt_d = np.linalg.svd(D_mat, full_matrices=False)
    total_var_d = float(np.sum(sv_d**2))

    print(f"  SVD of within-body difference matrix ({D_mat.shape}):")
    print(f"  {'Axis':>4s}  {'%var':>6s}  {'CumVar':>7s}  {'cos to plural':>14s}  "
          f"{'cos to adverb':>14s}  {'cos to gerund→past':>18s}")
    print(f"  {'-'*75}")

    oq4_results = []
    for axis in range(min(20, len(sv_d))):
        pct   = float(sv_d[axis]**2 / total_var_d * 100)
        cumv  = float(np.sum(sv_d[:axis+1]**2) / total_var_d * 100)
        v     = Vt_d[axis]
        cos_pl  = float(abs(v @ t2_vectors['plural']))   if 'plural' in t2_vectors else float('nan')
        cos_adv = float(abs(v @ t2_vectors['adverb']))   if 'adverb' in t2_vectors else float('nan')
        cos_ger = float(abs(v @ t2_vectors.get('gerund_to_past', v*0))) if 'gerund_to_past' in t2_vectors else float('nan')
        print(f"  {axis+1:>4d}  {pct:>6.2f}%  {cumv:>6.1f}%  {cos_pl:>14.3f}  {cos_adv:>14.3f}  {cos_ger:>18.3f}")
        oq4_results.append({'axis': axis+1, 'pct_var': pct, 'cumvar': cumv,
                             'cos_plural': cos_pl, 'cos_adverb': cos_adv,
                             'cos_gerund': cos_ger})

    # For the top 5 auto-discovered axes, show what body differences they capture best
    print(f"\n  Top auto-discovered Type 2 directions — what do they encode?")
    for axis in range(min(5, len(sv_d))):
        v = Vt_d[axis]
        # Find the word pairs whose difference vector most aligns with v
        sims = D_mat @ v   # signed projections
        top10_idx = np.argsort(np.abs(sims))[-10:][::-1]
        top_pairs = [(diff_labels[i][1].strip(), diff_labels[i][2].strip(),
                      float(sims[i])) for i in top10_idx]
        pct = float(sv_d[axis]**2 / total_var_d * 100)
        print(f"\n  Axis {axis+1} ({pct:.2f}% var):")
        for wa, wb, s in top_pairs[:6]:
            print(f"    {s:+.3f}  {wa} → {wb}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ5 — Stability of ~43 Effective Dimensions")
print(f"{'='*65}")

# Bootstrap: resample 80% of bodies N times, measure effective rank each time
N_bootstrap = 200
n_bodies    = len(bodies_list)
sample_size = int(0.80 * n_bodies)
eff_ranks   = []
k95_samples = []

rng = np.random.default_rng(42)
for _ in range(N_bootstrap):
    idx   = rng.choice(n_bodies, size=sample_size, replace=False)
    C_samp = centroid_mat[idx]
    _, sv_s, _ = np.linalg.svd(C_samp, full_matrices=False)
    eff_r = float(np.sum(sv_s)**2 / np.sum(sv_s**2))
    eff_ranks.append(eff_r)
    cumv_s = np.cumsum(sv_s**2) / np.sum(sv_s**2)
    k95_s  = int(np.searchsorted(cumv_s, 0.95)) + 1
    k95_samples.append(k95_s)

eff_ranks  = np.array(eff_ranks)
k95_arr    = np.array(k95_samples)

print(f"\n  Bootstrap effective rank (N={N_bootstrap}, 80% resample):")
print(f"    Mean = {eff_ranks.mean():.1f}  Std = {eff_ranks.std():.1f}")
print(f"    Range = [{eff_ranks.min():.1f}, {eff_ranks.max():.1f}]")
print(f"    95% CI: [{np.percentile(eff_ranks,2.5):.1f}, {np.percentile(eff_ranks,97.5):.1f}]")

print(f"\n  Bootstrap k(95% var) (N={N_bootstrap}):")
print(f"    Mean = {k95_arr.mean():.1f}  Std = {k95_arr.std():.1f}")
print(f"    Range = [{k95_arr.min()}, {k95_arr.max()}]")
print(f"    95% CI: [{np.percentile(k95_arr,2.5):.0f}, {np.percentile(k95_arr,97.5):.0f}]")

# Is the effective rank proportional to body count or square-root?
# Test by varying sample size
print(f"\n  Scaling of effective rank with number of bodies:")
print(f"  {'n_bodies':>8s}  {'eff_rank_mean':>14s}  {'ratio (eff/n)':>14s}")
for frac in [0.25, 0.4, 0.6, 0.8, 1.0]:
    n_samp  = max(5, int(frac * n_bodies))
    ranks_f = []
    for _ in range(50):
        idx   = rng.choice(n_bodies, size=n_samp, replace=False)
        C_s   = centroid_mat[idx]
        _, sv_s, _ = np.linalg.svd(C_s, full_matrices=False)
        ranks_f.append(float(np.sum(sv_s)**2 / np.sum(sv_s**2)))
    mean_r = float(np.mean(ranks_f))
    print(f"  {n_samp:>8d}  {mean_r:>14.1f}  {mean_r/n_samp:>14.3f}")

# 7B prediction
print(f"\n  OQ5 Prediction for Qwen2-7B:")
print(f"  If effective_rank scales as k*n_bodies^alpha, we need two data points.")
print(f"  At 95 bodies: eff_rank={eff_ranks.mean():.1f} (= {eff_ranks.mean()/n_bodies:.2f} × n)")
print(f"  Qwen2-7B likely has ~200-500 Zone C bodies (larger vocabulary, more capacity).")
print(f"  If alpha≈1.0 (linear): 7B eff_rank ≈ {eff_ranks.mean()/n_bodies*300:.0f}–{eff_ranks.mean()/n_bodies*500:.0f}")
print(f"  If alpha≈0.5 (sqrt):   7B eff_rank ≈ {eff_ranks.mean()*np.sqrt(300/n_bodies):.0f}–{eff_ranks.mean()*np.sqrt(500/n_bodies):.0f}")
print(f"  The TYPE 2 structure (morphological operators) should be the SAME: {len(t2_vectors)} known directions.")
print(f"  Type 2 is language-determined, not model-size-determined.")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY — Open Questions from DC 315")
print(f"{'='*65}")

print(f"\n  OQ1 (Semantic Type 2 operators):")
if oq1a_results:
    t1g = sum(r['top1'] for r in oq1a_results)
    print(f"    Gender: {t1g}/{len(oq1a_results)} in held-out family pairs")
if oq1c_results:
    t1c = sum(r['top1'] for r in oq1c_results)
    print(f"    Comp→Sup: {t1c}/{len(oq1c_results)} held-out")
if oq1b_results:
    t1b = sum(r['top1'] for r in oq1b_results)
    print(f"    Antonym: {t1b}/{len(oq1b_results)} held-out")
print(f"    Known Type 2 vectors: {list(t2_vectors.keys())}")

print(f"\n  OQ3 (Type 2 ⊥ Type 1?):")
if oq3_results:
    mean_proj = float(np.mean([v['proj_frac'] for v in oq3_results.values()]))
    print(f"    Mean ||T1 projection||² across all Type 2 vectors: {mean_proj:.3f}")
    print(f"    → {'MOSTLY ORTHOGONAL' if mean_proj < 0.25 else 'PARTIAL OVERLAP'}")

print(f"\n  OQ5 (Stability):")
print(f"    Bootstrap 95% CI for effective rank: "
      f"[{np.percentile(eff_ranks,2.5):.1f}, {np.percentile(eff_ranks,97.5):.1f}]")
print(f"    → {'STABLE' if eff_ranks.std() < 3 else 'VARIABLE'}")


# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "meta": {"experiment": "Day 37 — Open Questions from DC 315"},
    "oq1_semantic_type2": {
        "gender": oq1a_results,
        "antonym": oq1b_results,
        "comp_to_sup": oq1c_results,
        "type2_vectors_found": list(t2_vectors.keys()),
    },
    "oq2_hierarchy": {
        "clusters": {str(k): v for k, v in oq2_clusters.items()},
    },
    "oq3_orthogonality": oq3_results,
    "oq4_autodiscovery": oq4_results[:20] if 'oq4_results' in dir() else [],
    "oq5_stability": {
        "bootstrap_eff_rank_mean": float(eff_ranks.mean()),
        "bootstrap_eff_rank_std": float(eff_ranks.std()),
        "bootstrap_eff_rank_ci95": [float(np.percentile(eff_ranks,2.5)),
                                    float(np.percentile(eff_ranks,97.5))],
        "bootstrap_k95_mean": float(k95_arr.mean()),
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 37 complete.")
