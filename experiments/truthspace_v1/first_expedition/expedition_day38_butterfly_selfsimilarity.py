#!/usr/bin/env python3
"""
Expedition Day 38 — Is the T1/T2 Butterfly Self-Similar?

DC 315 §13 / DC 282 §8.5 predict: the T1/T2 mutual reinforcement recurs
at every scale. Three independent tests:

TEST A: Sub-clustering within the SCALE cluster (4 bodies)
  Ward's method on individual SCALE-cluster word φ-vectors. Do the words
  form sub-clusters that mirror the macro body structure? Are sub-clusters
  connected by sub-T2 operators?

TEST B: T1-consistency vs T2-consistency correlation (the "critical line")
  For every word in every known T2 pair, measure:
    T1-score = cos(φ(word), body_centroid)   [how central in body]
    T2-score = cos(φ(word)+r_T2, φ(partner)) [how well T2 predicts partner]
  Hypothesis: T1-score and T2-score are positively correlated.
  High-T1/high-T2 words are the "critical line zeros."

TEST C: Within-body directional operators
  Does the antonym direction work WITHIN the Comparative Adj body even
  though it fails cross-body? Tests whether T2-like structure exists at
  finer scale but is body-local rather than universal.

TEST D: Body-centroid Axis 1 projection vs T2-accuracy
  For each T2 pair, project both words onto Vt_c[0] (dominant concept axis).
  Is proximity to the body's Axis 1 centroid value correlated with T2-accuracy?
"""

import os, json
import numpy as np
from collections import defaultdict, Counter

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day38_butterfly_selfsimilarity.json")

KILLING_PAIRS = [
    ('cat','cats'),('dog','dogs'),('tree','trees'),('bird','birds'),
    ('house','houses'),('man','woman'),('king','queen'),('boy','girl'),
    ('big','bigger'),('fast','faster'),('old','older'),
]

def batch_phi(hs, z2):
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

def find_word(w):
    for v in [w, ' '+w, '▁'+w]:
        if v in w2l:
            return v
    return None

def mean_rel_vec(pairs):
    vecs = []
    for a, b in pairs:
        fa, fb = find_word(a), find_word(b)
        if fa and fb:
            d = wmap_phi[w2l[fb]] - wmap_phi[w2l[fa]]
            nm = np.linalg.norm(d)
            if nm > 1e-20:
                vecs.append(d / nm)
    if not vecs:
        return None
    m = np.stack(vecs).mean(axis=0)
    nm = np.linalg.norm(m)
    return m / nm if nm > 1e-20 else None

# ── Load ──────────────────────────────────────────────────────────────────────
print("── Load ──────────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words  = [w for w, v in wmap.items() if v['phase']==2
                 and v.get('L14_body') not in ('B000','B001',None) and w in w2i]
zone_c_bodies = {w: wmap[w]['L14_body'] for w in zone_c_words}
zone_d_words  = [w for w, v in wmap.items() if v['phase']==2
                 and v.get('L14_body') == 'B000' and w in w2i]
body_label_map = {}
for w, v in wmap.items():
    b = v.get('L14_body')
    if b and b not in body_label_map:
        body_label_map[b] = v.get('L14_label', '?')

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

zone_c_idx = np.array([w2i[w] for w in zone_c_words])
zone_d_idx = np.array([w2i[w] for w in zone_d_words])
phi_c14    = batch_phi(hs14_all[zone_c_idx], z2)
phi_d14    = batch_phi(hs14_all[zone_d_idx], z2)
phi0_14    = phi_d14.mean(axis=0)
phi0_14   /= np.linalg.norm(phi0_14)

wmap_words = [w for w in wmap.keys() if w in w2i]
wmap_idx   = np.array([w2i[w] for w in wmap_words])
wmap_phi   = batch_phi(hs14_all[wmap_idx], z2)
w2l        = {w: i for i, w in enumerate(wmap_words)}

body_members = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

body_centroids = {}
body_words_map = {}
for body, idxs in body_members.items():
    vecs = phi_c14[idxs]
    c    = vecs.mean(axis=0)
    body_centroids[body] = c / (np.linalg.norm(c) + 1e-20)
    body_words_map[body] = [zone_c_words[i] for i in idxs]

centroid_mat = np.stack([body_centroids[b] for b in sorted(body_centroids)])
bodies_list  = sorted(body_centroids.keys())
_, sv_c, Vt_c = np.linalg.svd(centroid_mat, full_matrices=False)
cumvar_c    = np.cumsum(sv_c**2) / np.sum(sv_c**2)
k95 = int(np.searchsorted(cumvar_c, 0.95)) + 1
T1_basis = Vt_c[:k95]

# Reconstruct T2 vectors
T2_PAIRS = {
    'plural':    [('brother','brothers'),('sister','sisters'),('cousin','cousins'),
                  ('vegetable','vegetables'),('potato','potatoes'),('tomato','tomatoes')],
    'adverb':    [('respective','respectively'),('unfortunate','unfortunately'),
                  ('subsequent','subsequently'),('intentional','intentionally'),
                  ('drastic','drastically'),('genetic','genetically')],
    'gerund_to_past': [('deciding','decided'),('killing','killed'),
                       ('welcoming','welcomed'),('crushing','crushed')],
    'comp_to_sup':    [('bigger','biggest'),('faster','fastest'),('taller','tallest'),
                       ('stronger','strongest'),('older','oldest')],
}
t2_vectors = {}
t2_pairs_found = {}
for name, pairs in T2_PAIRS.items():
    rv = mean_rel_vec(pairs)
    if rv is not None:
        t2_vectors[name] = rv
        found = [(a, b) for a, b in pairs if find_word(a) and find_word(b)]
        t2_pairs_found[name] = found

print(f"  Zone C: {len(zone_c_words)}  bodies: {len(body_centroids)}")
print(f"  T2 vectors: {list(t2_vectors.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"TEST A — Sub-clustering within SCALE Cluster (4 bodies)")
print(f"{'='*65}")

# Identify which bodies form the SCALE cluster
# From Day 37: Comparative Adj, Superlative Adj, Size Comparison, Thickness Variations
scale_keywords = ['comparative', 'superlative', 'size', 'thickness', 'comparison']
scale_bodies = []
for body in bodies_list:
    lbl = body_label_map.get(body, '').lower()
    if any(k in lbl for k in scale_keywords):
        scale_bodies.append(body)
        print(f"  SCALE body: {body} → {body_label_map.get(body,'?')}")

if not scale_bodies:
    print("  WARN: no SCALE bodies found by keyword — using bodies with most comp/sup words")
    # Fallback: find bodies with most comparative/superlative words
    comp_bodies = {}
    for body, ws in body_words_map.items():
        n_comp = sum(1 for w in ws if any(w.strip().endswith(sfx) for sfx in ['er','est','ly']))
        comp_bodies[body] = n_comp
    scale_bodies = sorted(comp_bodies, key=lambda b: -comp_bodies[b])[:4]
    for b in scale_bodies:
        print(f"  SCALE body (fallback): {b} → {body_label_map.get(b,'?')}")

# Gather all words from SCALE bodies
scale_word_list = []
scale_body_labels = []
for body in scale_bodies:
    for w in body_words_map.get(body, []):
        fw = find_word(w.strip())
        if fw:
            scale_word_list.append(fw)
            scale_body_labels.append(body_label_map.get(body,'?')[:20])

print(f"\n  {len(scale_word_list)} words across SCALE cluster bodies")
print(f"  Body sizes: " + ", ".join(f"{b_lbl}={sum(1 for l in scale_body_labels if l==b_lbl)}" 
                                      for b_lbl in dict.fromkeys(scale_body_labels)))

scale_phi = np.array([wmap_phi[w2l[w]] for w in scale_word_list])

# Ward's hierarchical clustering of SCALE words
testa_results = {}
if HAS_SCIPY and len(scale_word_list) >= 4:
    cos_d = np.clip(1.0 - (scale_phi @ scale_phi.T), 0, 2)
    cos_d = (cos_d + cos_d.T) / 2.0
    np.fill_diagonal(cos_d, 0.0)
    dist_cond = squareform(cos_d, checks=False)
    Z_link = linkage(dist_cond, method='ward')

    print(f"\n  Sub-clustering at k=4 (one per body expected if self-similar):")
    labels_k4 = fcluster(Z_link, t=4, criterion='maxclust')
    sub_clusters = defaultdict(list)
    for i, lbl in enumerate(labels_k4):
        sub_clusters[int(lbl)].append((scale_word_list[i], scale_body_labels[i]))

    # For each sub-cluster, show body composition and top words
    from_body_purity = []
    for cl_id in sorted(sub_clusters.keys()):
        members = sub_clusters[cl_id]
        body_counts = defaultdict(int)
        for w, b in members:
            body_counts[b] += 1
        total = len(members)
        purity = max(body_counts.values()) / total
        dom_body = max(body_counts, key=body_counts.get)
        from_body_purity.append(purity)
        print(f"\n  Sub-cluster {cl_id} ({total} words, purity={purity:.2f}, dominant={dom_body[:25]}):")
        for w, b in sorted(members, key=lambda x: x[1])[:8]:
            print(f"    {w.strip():<20s}  ← {b[:20]}")
        if len(members) > 8:
            print(f"    ... +{len(members)-8} more")

    mean_purity = float(np.mean(from_body_purity))
    print(f"\n  Mean cluster purity = {mean_purity:.3f}")
    print(f"  (1.0 = perfect: each sub-cluster contains only one body)")
    print(f"  (0.25 = random: sub-clusters are mixed)")
    testa_results['mean_purity_k4'] = mean_purity

    # Now test: are the sub-clusters connected by a T2-like direction?
    # Compute sub-cluster centroids
    sub_centroids = {}
    for cl_id in sorted(sub_clusters.keys()):
        members = sub_clusters[cl_id]
        vecs = np.array([wmap_phi[w2l[w]] for w, _ in members])
        c = vecs.mean(axis=0)
        sub_centroids[cl_id] = c / (np.linalg.norm(c) + 1e-20)

    print(f"\n  Sub-cluster centroid cosines (pairwise):")
    cl_ids = sorted(sub_centroids.keys())
    for i in cl_ids:
        for j in cl_ids:
            if i < j:
                c = float(sub_centroids[i] @ sub_centroids[j])
                print(f"    sub-cluster {i} ↔ {j}: cos = {c:.3f}")

    # Does the comp→sup T2 direction point from the Comparative sub-cluster toward Superlative?
    if 'comp_to_sup' in t2_vectors:
        r_cs = t2_vectors['comp_to_sup']
        print(f"\n  comp→sup T2 direction alignment with sub-cluster centroid differences:")
        for i in cl_ids:
            for j in cl_ids:
                if i == j: continue
                diff = sub_centroids[j] - sub_centroids[i]
                nm   = np.linalg.norm(diff)
                if nm < 1e-10: continue
                diff /= nm
                alignment = float(abs(r_cs @ diff))
                if alignment > 0.10:
                    lbl_i = Counter(b for _, b in sub_clusters[i]).most_common(1)[0][0]
                    lbl_j = Counter(b for _, b in sub_clusters[j]).most_common(1)[0][0]
                    print(f"    {lbl_i[:15]} → {lbl_j[:15]}: |cos(comp→sup, diff)| = {alignment:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"TEST B — T1-consistency vs T2-consistency (The Critical Line)")
print(f"{'='*65}")

print(f"\n  For every T2 pair (a,b): measure T1-score(a) and T2-score(a→b)")
print(f"  H: Pearson r(T1-score, T2-score) > 0 across all pairs\n")

testb_results = {}
all_t1_scores = []
all_t2_scores = []
all_pair_labels = []

for t2_name, pairs in T2_PAIRS.items():
    r_t2 = t2_vectors.get(t2_name)
    if r_t2 is None:
        continue

    t1_scores = []
    t2_scores = []

    for a, b in pairs:
        fa, fb = find_word(a), find_word(b)
        if not fa or not fb:
            continue

        # Body membership of word a
        a_plain = a.strip()
        body_a = None
        for w_orig, bd in zone_c_bodies.items():
            if w_orig.strip() == a_plain or w_orig == fa:
                body_a = bd
                break
        if body_a is None:
            # try plain
            for w_orig, bd in zone_c_bodies.items():
                if w_orig.strip() == fa.strip():
                    body_a = bd
                    break

        if body_a is None:
            continue

        c_a = body_centroids[body_a]
        phi_a = wmap_phi[w2l[fa]]
        phi_b = wmap_phi[w2l[fb]]

        t1 = float(phi_a @ c_a)
        t2 = float((phi_a + r_t2) @ phi_b / (np.linalg.norm(phi_a + r_t2) + 1e-20))
        t1_scores.append(t1)
        t2_scores.append(t2)
        all_t1_scores.append(t1)
        all_t2_scores.append(t2)
        all_pair_labels.append(t2_name)

    if len(t1_scores) >= 3:
        r = float(np.corrcoef(t1_scores, t2_scores)[0, 1])
        print(f"  {t2_name:<20s}: {len(t1_scores)} pairs, r(T1,T2) = {r:.3f}")
        print(f"    T1 mean={np.mean(t1_scores):.3f}±{np.std(t1_scores):.3f}  "
              f"T2 mean={np.mean(t2_scores):.3f}±{np.std(t2_scores):.3f}")
        testb_results[t2_name] = {'n': len(t1_scores), 'r_T1_T2': r,
                                    't1_mean': float(np.mean(t1_scores)),
                                    't2_mean': float(np.mean(t2_scores))}

# All pairs combined
if len(all_t1_scores) >= 5:
    r_all = float(np.corrcoef(all_t1_scores, all_t2_scores)[0, 1])
    print(f"\n  ALL T2 operators combined ({len(all_t1_scores)} pairs):")
    print(f"  r(T1-consistency, T2-accuracy) = {r_all:.3f}")
    if abs(r_all) > 0.3:
        print(f"  → MODERATE CORRELATION — butterfly structure confirmed at word level")
    elif abs(r_all) > 0.1:
        print(f"  → WEAK CORRELATION — some signal but noisy")
    else:
        print(f"  → NO CORRELATION — T1 and T2 are orthogonal at word level too")
    testb_results['combined'] = {'n': len(all_t1_scores), 'r_T1_T2': r_all}

# Show the "critical line words" — high T1 AND high T2
if all_t1_scores:
    t1a = np.array(all_t1_scores)
    t2a = np.array(all_t2_scores)
    # Z-score both
    t1z = (t1a - t1a.mean()) / (t1a.std() + 1e-20)
    t2z = (t2a - t2a.mean()) / (t2a.std() + 1e-20)
    combined_score = t1z + t2z
    # Find words by reconstructing pair list
    pair_list_flat = []
    for t2_name, pairs in T2_PAIRS.items():
        r_t2 = t2_vectors.get(t2_name)
        if r_t2 is None:
            continue
        for a, b in pairs:
            fa, fb = find_word(a), find_word(b)
            if not fa or not fb:
                continue
            a_plain = a.strip()
            body_a = None
            for w_orig, bd in zone_c_bodies.items():
                if w_orig.strip() == a_plain or w_orig == fa:
                    body_a = bd
                    break
            if body_a is None:
                continue
            pair_list_flat.append((a, b, t2_name))

    if len(pair_list_flat) == len(all_t1_scores):
        top_zeros = np.argsort(combined_score)[-5:][::-1]
        bot_zeros = np.argsort(combined_score)[:5]
        print(f"\n  Top 'critical line' words (high T1 AND high T2):")
        for idx in top_zeros:
            a, b, t2n = pair_list_flat[idx]
            print(f"    {a}→{b} ({t2n[:12]}): T1={all_t1_scores[idx]:.3f} T2={all_t2_scores[idx]:.3f}")
        print(f"\n  'Off-critical-line' words (low T1 or low T2):")
        for idx in bot_zeros:
            a, b, t2n = pair_list_flat[idx]
            print(f"    {a}→{b} ({t2n[:12]}): T1={all_t1_scores[idx]:.3f} T2={all_t2_scores[idx]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"TEST C — Within-body Directional Operators")
print(f"{'='*65}")

# Focus on Comparative Adj body and Superlative Adj body
print(f"\n  Does the antonym direction WITHIN Comparative Adj body succeed?")
print(f"  (Day 37: antonym fails cross-body 0/4. Does it work within-body?)")

comp_body = None
sup_body  = None
for b, lbl in body_label_map.items():
    if 'comparative' in lbl.lower() and 'adjective' in lbl.lower():
        comp_body = b
    if 'superlative' in lbl.lower():
        sup_body = b
if comp_body is None:
    for b, lbl in body_label_map.items():
        if 'comparative' in lbl.lower():
            comp_body = b; break
if sup_body is None:
    for b, lbl in body_label_map.items():
        if 'superlative' in lbl.lower():
            sup_body = b; break

testc_results = {}
for target_body, body_name in [(comp_body, 'Comparative Adj'), (sup_body, 'Superlative Adj')]:
    if target_body is None:
        print(f"  {body_name}: not found")
        continue
    ws = [w.strip() for w in body_words_map.get(target_body, [])]
    print(f"\n  {body_name} body ({target_body}): {len(ws)} words")
    print(f"  Words: {', '.join(ws[:15])}{'...' if len(ws)>15 else ''}")

    # Sub-cluster within this body
    fws = [find_word(w) for w in ws]
    fws = [fw for fw in fws if fw]
    if len(fws) < 4:
        print(f"  Too few words ({len(fws)}) to cluster")
        continue

    phi_body = np.array([wmap_phi[w2l[fw]] for fw in fws])

    # Ward's on just this body
    if HAS_SCIPY:
        cos_d_b = np.clip(1.0 - (phi_body @ phi_body.T), 0, 2)
        cos_d_b = (cos_d_b + cos_d_b.T) / 2
        np.fill_diagonal(cos_d_b, 0.0)
        nk = min(4, len(fws)-1)
        dist_b = squareform(cos_d_b, checks=False)
        Z_b = linkage(dist_b, method='ward')
        labels_b = fcluster(Z_b, t=nk, criterion='maxclust')
        sub_b = defaultdict(list)
        for i, lbl in enumerate(labels_b):
            sub_b[int(lbl)].append(fws[i].strip())
        print(f"  k={nk} sub-clusters:")
        for cl_id in sorted(sub_b.keys()):
            words_in = sub_b[cl_id]
            print(f"    {cl_id}: {', '.join(words_in)}")

    # Test antonym within-body: from positive to negative comparatives
    # Positive: bigger, larger, faster, older, taller, stronger
    # Negative: smaller, slower, younger, shorter, weaker
    if body_name == 'Comparative Adj':
        ant_within_source = [
            ('bigger','smaller'),('faster','slower'),('taller','shorter'),
            ('older','younger'),('stronger','weaker'),
        ]
        ant_within_src = [(a,b) for a,b in ant_within_source
                          if find_word(a) and find_word(b)]
        ant_within_tgt = [
            ('heavier','lighter'),('louder','quieter'),('wider','narrower'),
            ('richer','poorer'),
        ]
        ant_within_tgt_found = [(a,b) for a,b in ant_within_tgt
                                if find_word(a) and find_word(b)]
        print(f"\n  Within-body antonym test:")
        print(f"  Source pairs found: {[(a,b) for a,b in ant_within_src]}")
        print(f"  Target pairs found: {[(a,b) for a,b in ant_within_tgt_found]}")

        r_ant_within = None
        if ant_within_src:
            r_ant_within = mean_rel_vec([(a,b) for a,b in ant_within_src])
        if r_ant_within is not None and ant_within_tgt_found:
            src_vecs = [mean_rel_vec([(a,b)]) for a,b in ant_within_src
                        if mean_rel_vec([(a,b)]) is not None]
            pcos = float(np.mean([src_vecs[i]@src_vecs[j]
                                   for i in range(len(src_vecs))
                                   for j in range(i+1,len(src_vecs))])) if len(src_vecs)>1 else float('nan')
            print(f"  Within-body antonym source pairwise cos: {pcos:.3f}  (vs cross-body 0.213)")
            top1, top5 = 0, 0
            for wa, wb in ant_within_tgt_found:
                fa, fb = find_word(wa), find_word(wb)
                query = wmap_phi[w2l[fa]] + r_ant_within
                nm = np.linalg.norm(query)
                query /= (nm + 1e-20)
                sims = wmap_phi @ query
                sims[w2l[fa]] = -2.0
                top5_idx = np.argsort(sims)[-5:][::-1]
                top5_words = [wmap_words[i].strip() for i in top5_idx]
                t1 = int(top5_words[0] == wb.strip())
                t5 = int(wb.strip() in top5_words)
                top1 += t1; top5 += t5
                print(f"    {wa}+ant→{top5_words[0]:<12s}  (want {wb:<12s}) {'✓' if t1 else ('~' if t5 else '✗')}")
            print(f"  Within-body antonym top-1: {top1}/{len(ant_within_tgt_found)}")
            testc_results['antonym_within_comp_body'] = {
                'src_pairwise_cos': pcos,
                'top1': top1, 'total': len(ant_within_tgt_found)
            }


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"TEST D — Axis 1 Projection vs T2-accuracy")
print(f"{'='*65}")

print(f"\n  For each T2 pair word, project onto concept Axis 1 (Vt_c[0])")
print(f"  H: proximity to body's centroid Axis-1 value correlates with T2-accuracy")

concept_axis1 = Vt_c[0]  # dominant concept axis

testd_results = {}
for t2_name, pairs in T2_PAIRS.items():
    r_t2 = t2_vectors.get(t2_name)
    if r_t2 is None:
        continue

    ax1_vals    = []
    ax1_offsets = []  # |word_ax1 - body_centroid_ax1|
    t2_accs     = []

    for a, b in pairs:
        fa, fb = find_word(a), find_word(b)
        if not fa or not fb:
            continue
        a_plain = a.strip()
        body_a  = None
        for w_orig, bd in zone_c_bodies.items():
            if w_orig.strip() == a_plain or w_orig == fa:
                body_a = bd
                break
        if body_a is None:
            continue

        c_a    = body_centroids[body_a]
        phi_a  = wmap_phi[w2l[fa]]
        phi_b  = wmap_phi[w2l[fb]]

        ax1_word = float(phi_a @ concept_axis1)
        ax1_cent = float(c_a    @ concept_axis1)
        ax1_offset = abs(ax1_word - ax1_cent)

        predicted = phi_a + r_t2
        predicted /= (np.linalg.norm(predicted) + 1e-20)
        t2_acc = float(predicted @ phi_b)

        ax1_vals.append(ax1_word)
        ax1_offsets.append(ax1_offset)
        t2_accs.append(t2_acc)

    if len(ax1_vals) >= 3:
        r_val    = float(np.corrcoef(ax1_vals,    t2_accs)[0,1])
        r_offset = float(np.corrcoef(ax1_offsets, t2_accs)[0,1])
        print(f"  {t2_name:<20s}:  r(Axis1 value, T2-acc)={r_val:.3f}   "
              f"r(|offset from centroid|, T2-acc)={r_offset:.3f}")
        testd_results[t2_name] = {'r_ax1_t2': r_val, 'r_offset_t2': r_offset,
                                    'n': len(ax1_vals)}

# Summary across all operators
if testd_results:
    r_mean_val    = float(np.mean([v['r_ax1_t2']   for v in testd_results.values()]))
    r_mean_offset = float(np.mean([v['r_offset_t2'] for v in testd_results.values()]))
    print(f"\n  Mean across operators: r(Axis1,T2-acc)={r_mean_val:.3f}   "
          f"r(|offset|,T2-acc)={r_mean_offset:.3f}")
    if r_mean_offset < -0.1:
        print(f"  → Words CLOSER to body centroid on Axis 1 → higher T2-accuracy")
        print(f"  → CRITICAL LINE confirmed: centroid = locus of T1/T2 consistency")
    else:
        print(f"  → No clear critical-line structure from Axis 1 alone")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SYNTHESIS — Is the Butterfly Self-Similar?")
print(f"{'='*65}")

print(f"\n  TEST A (Sub-clustering):")
if testa_results:
    p = testa_results.get('mean_purity_k4', float('nan'))
    print(f"    Mean purity = {p:.3f}")
    if p > 0.70:
        print(f"    → Sub-clusters respect body boundaries: SELF-SIMILAR")
    elif p > 0.50:
        print(f"    → Partial alignment with body boundaries")
    else:
        print(f"    → No sub-cluster / body alignment")

print(f"\n  TEST B (T1-T2 correlation):")
if 'combined' in testb_results:
    r = testb_results['combined']['r_T1_T2']
    print(f"    r(T1-consistency, T2-accuracy) = {r:.3f}")
    if r > 0.30:
        print(f"    → Critical line confirmed: central words form cleaner T2 pairs")
    elif r > 0.10:
        print(f"    → Weak critical-line signal")
    else:
        print(f"    → No critical-line structure")

print(f"\n  TEST C (Within-body antonym):")
if 'antonym_within_comp_body' in testc_results:
    res = testc_results['antonym_within_comp_body']
    print(f"    Source pairwise cos = {res['src_pairwise_cos']:.3f}  "
          f"top-1 = {res['top1']}/{res['total']}")
    if res['top1'] > 0:
        print(f"    → Within-body antonym PARTIALLY works (within-body T2 analog)")
    else:
        print(f"    → Within-body antonym also fails (antonym is not geometric at any scale)")

print(f"\n  TEST D (Axis 1 critical line):")
if testd_results:
    print(f"    Mean r(|offset from centroid|, T2-acc) = {r_mean_offset:.3f}")


# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    'meta': {'experiment': 'Day 38 — Butterfly Self-Similarity Test'},
    'test_a': testa_results,
    'test_b': testb_results,
    'test_c': testc_results,
    'test_d': testd_results,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 38 complete.")
