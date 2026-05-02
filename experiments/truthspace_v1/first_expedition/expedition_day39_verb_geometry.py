#!/usr/bin/env python3
"""
Expedition Day 39 — Verb Geometry in φ-Space

Probe reveals three separate sub-problems:
  (A) Base-form verbs (walk, run, go): Phase != 2 → Zone A/B near the pole
  (B) Generic derived forms (walking, wrote): Phase 2 → B000/B001 catch-all bodies
  (C) Semantically loaded verb forms (killed, killing): Phase 2 → Zone C specific body

Open Questions:
  OQ6: What is the internal structure of B000 and B001?
       Are they truly featureless, or is there semantic sub-clustering?
  OQ7: Do tense operators work across zone boundaries?
       base(A/B) → past(B000) — is there a consistent tense direction?
  OQ8: Zone trajectory analysis
       For a given verb, trace all forms through φ-space. What path do they follow?
  OQ9: Can Zone C verb entry be predicted geometrically?
       Given a base form φ-vector, can we predict which derived forms land in Zone C?
"""

import os, json
import numpy as np
from collections import defaultdict, Counter

try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day39_verb_geometry.json")

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

def find_word(w, w2l):
    for v in [w, ' '+w, '▁'+w]:
        if v in w2l:
            return v
    return None

def mean_rel_vec(pairs, w2l, wmap_phi):
    vecs = []
    for a, b in pairs:
        fa, fb = find_word(a, w2l), find_word(b, w2l)
        if fa and fb:
            d = wmap_phi[w2l[fb]] - wmap_phi[w2l[fa]]
            nm = np.linalg.norm(d)
            if nm > 1e-20:
                vecs.append(d / nm)
    if not vecs:
        return None, 0
    m = np.stack(vecs).mean(axis=0)
    nm = np.linalg.norm(m)
    return (m / nm if nm > 1e-20 else None), len(vecs)

# ── Load ──────────────────────────────────────────────────────────────────────
print("── Load ──────────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

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

# All words in wmap that have hs14 entries
wmap_words = [w for w in wmap.keys() if w in w2i]
wmap_idx   = np.array([w2i[w] for w in wmap_words])
wmap_phi   = batch_phi(hs14_all[wmap_idx], z2)
w2l        = {w: i for i, w in enumerate(wmap_words)}

# Also include ALL words (including Zone A/B) in a global phi map
all_phi    = batch_phi(hs14_all, z2)

# Zone segmentation
zone_c_words = [w for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') not in ('B000','B001',None)]
b000_words   = [w for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') == 'B000']
b001_words   = [w for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') == 'B001']
ab_words     = [w for w, v in wmap.items() if v['phase'] != 2]

# Only keep words with hs14
zone_c_words = [w for w in zone_c_words if w in w2i]
b000_words   = [w for w in b000_words   if w in w2i]
b001_words   = [w for w in b001_words   if w in w2i]
ab_words     = [w for w in ab_words     if w in w2i]

zone_c_phi = all_phi[np.array([w2i[w] for w in zone_c_words])]
b000_phi   = all_phi[np.array([w2i[w] for w in b000_words])]
b001_phi   = all_phi[np.array([w2i[w] for w in b001_words])]
ab_phi     = all_phi[np.array([w2i[w] for w in ab_words])]

# Compute semantic zero from Zone D (B000 centroid — used as phi0 proxy)
zone_d_all_words = b000_words + b001_words
zone_d_all_phi   = np.vstack([b000_phi, b001_phi])
phi0 = zone_d_all_phi.mean(axis=0)
phi0 /= np.linalg.norm(phi0)

print(f"  Zone C: {len(zone_c_words)}  B000: {len(b000_words)}  B001: {len(b001_words)}  A/B: {len(ab_words)}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ6 — Internal Structure of B000 and B001")
print(f"{'='*65}")

def zone_intrinsic_rank(phi_mat, name, n_sample=2000):
    if len(phi_mat) > n_sample:
        idx = np.random.RandomState(42).choice(len(phi_mat), n_sample, replace=False)
        phi_mat = phi_mat[idx]
    svs = np.linalg.svd(phi_mat, compute_uv=False)
    sv2 = svs**2
    cumvar = np.cumsum(sv2) / sv2.sum()
    ax1_var = float(sv2[0] / sv2.sum())
    eff_rank = float(np.exp(-np.sum((sv2/sv2.sum()) * np.log(sv2/sv2.sum() + 1e-20))))
    k95 = int(np.searchsorted(cumvar, 0.95)) + 1
    spread = float(np.mean(np.linalg.norm(phi_mat - phi_mat.mean(axis=0), axis=1)))
    print(f"\n  {name} (n={len(phi_mat)}):")
    print(f"    Axis 1 var={ax1_var:.3f}  eff_rank={eff_rank:.1f}  k_for_95%={k95}  spread={spread:.4f}")
    return {'ax1_var': ax1_var, 'eff_rank': eff_rank, 'k95': k95, 'spread': spread, 'n': len(phi_mat)}

print("\n  Geometry comparison:")
oq6_stats = {}
oq6_stats['zone_c'] = zone_intrinsic_rank(zone_c_phi, "Zone C (all bodies)")
oq6_stats['b000']   = zone_intrinsic_rank(b000_phi,   "B000 (catch-all large)")
oq6_stats['b001']   = zone_intrinsic_rank(b001_phi,   "B001 (secondary catch-all)")
oq6_stats['ab']     = zone_intrinsic_rank(ab_phi,     "Zone A/B (pole words)")

# Pairwise centroid distances between zones
c000 = b000_phi.mean(axis=0); c000 /= np.linalg.norm(c000)
c001 = b001_phi.mean(axis=0); c001 /= np.linalg.norm(c001)
cC   = zone_c_phi.mean(axis=0); cC /= np.linalg.norm(cC)
cAB  = ab_phi.mean(axis=0); cAB /= np.linalg.norm(cAB)

print(f"\n  Zone centroid cosines:")
print(f"    B000 ↔ B001:    {c000@c001:.3f}")
print(f"    B000 ↔ Zone C:  {c000@cC:.3f}")
print(f"    B001 ↔ Zone C:  {c001@cC:.3f}")
print(f"    A/B  ↔ Zone C:  {cAB@cC:.3f}")
print(f"    A/B  ↔ B000:    {cAB@c000:.3f}")
print(f"    A/B  ↔ B001:    {cAB@c001:.3f}")

# Sub-clustering B000 (sample 300 for speed)
print(f"\n  B000 sub-clustering (k=8, sample 300):")
if HAS_SCIPY and len(b000_words) >= 8:
    rng = np.random.RandomState(42)
    idx_s = rng.choice(len(b000_words), min(300, len(b000_words)), replace=False)
    b000_sample_words = [b000_words[i] for i in idx_s]
    b000_sample_phi   = b000_phi[idx_s]
    cos_d = np.clip(1.0 - (b000_sample_phi @ b000_sample_phi.T), 0, 2)
    cos_d = (cos_d + cos_d.T)/2; np.fill_diagonal(cos_d, 0)
    Z_link = linkage(squareform(cos_d, checks=False), method='ward')
    labels_k8 = fcluster(Z_link, t=8, criterion='maxclust')
    sub_clusters = defaultdict(list)
    for i, lbl in enumerate(labels_k8):
        sub_clusters[int(lbl)].append(b000_sample_words[i])
    for cl_id in sorted(sub_clusters.keys()):
        ws = [w.strip() for w in sub_clusters[cl_id]]
        print(f"    Sub-cluster {cl_id} ({len(ws)} words): {', '.join(ws[:10])}{'...' if len(ws)>10 else ''}")

    # Intra vs inter cluster cosines
    cluster_phis = {}
    for cl_id in sorted(sub_clusters.keys()):
        idxs = [i for i, l in enumerate(labels_k8) if l == cl_id]
        c = b000_sample_phi[idxs].mean(axis=0)
        cluster_phis[cl_id] = c / (np.linalg.norm(c) + 1e-20)
    cl_ids = sorted(cluster_phis.keys())
    inter_cos = np.array([cluster_phis[i]@cluster_phis[j]
                           for i in cl_ids for j in cl_ids if i<j])
    intra_cos_list = []
    for cl_id in cl_ids:
        idxs = [i for i, l in enumerate(labels_k8) if l == cl_id]
        if len(idxs) > 1:
            sub = b000_sample_phi[idxs]
            for ii in range(len(idxs)):
                for jj in range(ii+1, len(idxs)):
                    intra_cos_list.append(float(sub[ii] @ sub[jj]))
    print(f"\n  B000 inter-cluster cos: mean={inter_cos.mean():.3f} min={inter_cos.min():.3f}")
    if intra_cos_list:
        print(f"  B000 intra-cluster cos: mean={np.mean(intra_cos_list):.3f}")
    oq6_stats['b000_subcluster_inter_cos'] = float(inter_cos.mean())
    oq6_stats['b000_subcluster_intra_cos'] = float(np.mean(intra_cos_list)) if intra_cos_list else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ7 — Tense Operators: Do They Cross Zone Boundaries?")
print(f"{'='*65}")

# All verb form groups we want to test
verb_groups = [
    # (base, past, gerund, 3sg)
    ('kill',    'killed',   'killing',   'kills'),
    ('walk',    'walked',   'walking',   'walks'),
    ('write',   'wrote',    'writing',   'writes'),
    ('speak',   'spoke',    'speaking',  'speaks'),
    ('eat',     'ate',      'eating',    'eats'),
    ('give',    'gave',     'giving',    'gives'),
    ('make',    'made',     'making',    'makes'),
    ('take',    'took',     'taking',    'takes'),
    ('use',     'used',     'using',     'uses'),
    ('come',    'came',     'coming',    'comes'),
    ('think',   'thought',  'thinking',  'thinks'),
    ('build',   'built',    'building',  'builds'),
    ('break',   'broke',    'breaking',  'breaks'),
    ('drive',   'drove',    'driving',   'drives'),
    ('sing',    'sang',     'singing',   'sings'),
    ('send',    'sent',     'sending',   'sends'),
    ('keep',    'kept',     'keeping',   'keeps'),
    ('cut',     'cut',      'cutting',   'cuts'),
    ('hold',    'held',     'holding',   'holds'),
    ('run',     'ran',      'running',   'runs'),
]

def get_zone(w):
    key = find_word(w, w2l)
    if not key:
        return '?', None
    v = wmap.get(key, {})
    bd = v.get('L14_body')
    if bd not in ('B000', 'B001', None) and v.get('phase') == 2:
        return 'C', bd
    elif bd == 'B000':
        return 'B000', bd
    elif bd == 'B001':
        return 'B001', bd
    elif v.get('phase') != 2:
        return 'A/B', None
    else:
        return 'D?', bd

print(f"\n  Verb form zone map and φ-positions:")
print(f"  {'base':<12s} {'past':<12s} {'gerund':<12s} {'3sg':<12s}")
print(f"  {'-'*60}")
for base, past, gerund, sg3 in verb_groups:
    zb, _ = get_zone(base)
    zp, _ = get_zone(past)
    zg, _ = get_zone(gerund)
    z3, _ = get_zone(sg3)
    print(f"  {base+'('+zb+')':<16s} {past+'('+zp+')':<16s} {gerund+'('+zg+')':<16s} {sg3+'('+z3+')':<12s}")

# Now compute tense direction vectors
print(f"\n  Tense operator vectors (base→past):")
base_to_past_vecs   = []
base_to_gerund_vecs = []
base_to_3sg_vecs    = []
past_to_gerund_vecs = []
pair_labels = []

for base, past, gerund, sg3 in verb_groups:
    fb = find_word(base, w2l)
    fp = find_word(past, w2l)
    fg = find_word(gerund, w2l)
    f3 = find_word(sg3, w2l)

    if fb and fp:
        d = wmap_phi[w2l[fp]] - wmap_phi[w2l[fb]]
        nm = np.linalg.norm(d)
        if nm > 1e-20:
            base_to_past_vecs.append(d/nm)
            pair_labels.append((base, past))

    if fb and fg:
        d = wmap_phi[w2l[fg]] - wmap_phi[w2l[fb]]
        nm = np.linalg.norm(d)
        if nm > 1e-20:
            base_to_gerund_vecs.append(d/nm)

    if fb and f3:
        d = wmap_phi[w2l[f3]] - wmap_phi[w2l[fb]]
        nm = np.linalg.norm(d)
        if nm > 1e-20:
            base_to_3sg_vecs.append(d/nm)

    if fp and fg:
        d = wmap_phi[w2l[fg]] - wmap_phi[w2l[fp]]
        nm = np.linalg.norm(d)
        if nm > 1e-20:
            past_to_gerund_vecs.append(d/nm)

def pairwise_cos(vecs):
    if len(vecs) < 2:
        return float('nan')
    V = np.stack(vecs)
    c = []
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            c.append(float(V[i] @ V[j]))
    return float(np.mean(c))

print(f"\n  Pairwise cos between individual tense vectors (universality test):")
print(f"  base→past   ({len(base_to_past_vecs)} pairs): {pairwise_cos(base_to_past_vecs):.3f}  "
      f"(plural=?, adverb=?, comp→sup=0.866 for reference)")
print(f"  base→gerund ({len(base_to_gerund_vecs)} pairs): {pairwise_cos(base_to_gerund_vecs):.3f}")
print(f"  base→3sg    ({len(base_to_3sg_vecs)} pairs): {pairwise_cos(base_to_3sg_vecs):.3f}")
print(f"  past→gerund ({len(past_to_gerund_vecs)} pairs): {pairwise_cos(past_to_gerund_vecs):.3f}")

# Mean tense vectors
r_b2p = None; r_b2g = None
if base_to_past_vecs:
    m = np.stack(base_to_past_vecs).mean(axis=0)
    r_b2p = m / np.linalg.norm(m)
    print(f"\n  Mean base→past direction computed ({len(base_to_past_vecs)} pairs)")

if base_to_gerund_vecs:
    m = np.stack(base_to_gerund_vecs).mean(axis=0)
    r_b2g = m / np.linalg.norm(m)
    print(f"  Mean base→gerund direction computed ({len(base_to_gerund_vecs)} pairs)")

# Orthogonality between tense operators and known T2 operators
known_t2_pairs = {
    'plural':     [('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees')],
    'adverb':     [('respective','respectively'),('unfortunate','unfortunately'),
                   ('intentional','intentionally'),('drastic','drastically')],
    'comp→sup':   [('bigger','biggest'),('faster','fastest'),('older','oldest')],
}
known_t2_vecs = {}
for name, pairs in known_t2_pairs.items():
    rv, n = mean_rel_vec(pairs, w2l, wmap_phi)
    if rv is not None:
        known_t2_vecs[name] = rv
        print(f"  Known T2 {name} reloaded ({n} pairs)")

tense_vecs = {}
if r_b2p is not None: tense_vecs['base→past']   = r_b2p
if r_b2g is not None: tense_vecs['base→gerund'] = r_b2g
if base_to_3sg_vecs:
    m = np.stack(base_to_3sg_vecs).mean(axis=0)
    tense_vecs['base→3sg'] = m / np.linalg.norm(m)
if past_to_gerund_vecs:
    m = np.stack(past_to_gerund_vecs).mean(axis=0)
    tense_vecs['past→gerund'] = m / np.linalg.norm(m)

print(f"\n  Cosines between tense operators and known T2 operators:")
for tname, tv in tense_vecs.items():
    for kname, kv in known_t2_vecs.items():
        print(f"    {tname:<15s} ↔ {kname:<12s}: cos = {tv@kv:.3f}")

print(f"\n  Tense operator mutual orthogonality:")
tnames = list(tense_vecs.keys())
for i in range(len(tnames)):
    for j in range(i+1, len(tnames)):
        print(f"    {tnames[i]:<15s} ↔ {tnames[j]:<12s}: cos = {tense_vecs[tnames[i]]@tense_vecs[tnames[j]]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ8 — Tense Operator Retrieval Test (Top-1 / Top-5)")
print(f"{'='*65}")

print(f"\n  Using mean base→past vector:")
test_pairs_retrieval = [
    ('build','built'),('break','broke'),('drive','drove'),('sing','sang'),
    ('send','sent'),('keep','kept'),('hold','held'),('run','ran'),
    ('find','found'),('lose','lost'),('grow','grew'),('throw','threw'),
]
oq8_results = {}
for operator_name, r_op in tense_vecs.items():
    print(f"\n  [{operator_name}]")
    src_pairs = {
        'base→past':    [('build','built'),('break','broke'),('drive','drove'),('sing','sang'),
                         ('send','sent'),('keep','kept'),('hold','held'),('run','ran'),
                         ('find','found'),('lose','lost'),('grow','grew'),('throw','threw')],
        'base→gerund':  [('build','building'),('break','breaking'),('drive','driving'),
                         ('sing','singing'),('send','sending'),('keep','keeping'),
                         ('hold','holding'),('run','running'),('find','finding'),
                         ('lose','losing'),('grow','growing'),('throw','throwing')],
        'base→3sg':     [('build','builds'),('break','breaks'),('drive','drives'),
                         ('sing','sings'),('send','sends'),('keep','keeps'),
                         ('hold','holds'),('run','runs'),('find','finds'),
                         ('lose','loses'),('grow','grows'),('throw','throws')],
        'past→gerund':  [('built','building'),('broke','breaking'),('drove','driving'),
                         ('sang','singing'),('sent','sending'),('kept','keeping'),
                         ('held','holding'),('ran','running'),('found','finding'),
                         ('lost','losing'),('grew','growing'),('threw','throwing')],
    }.get(operator_name, [])

    top1, top5, total = 0, 0, 0
    for src_w, tgt_w in src_pairs:
        fsrc = find_word(src_w, w2l)
        ftgt = find_word(tgt_w, w2l)
        if not fsrc or not ftgt:
            continue
        query = wmap_phi[w2l[fsrc]] + r_op
        nm = np.linalg.norm(query)
        query /= (nm + 1e-20)
        sims = wmap_phi @ query
        sims[w2l[fsrc]] = -2.0
        top5_idx = np.argsort(sims)[-5:][::-1]
        top5_w = [wmap_words[i].strip() for i in top5_idx]
        t1 = int(top5_w[0] == tgt_w)
        t5 = int(tgt_w in top5_w)
        top1 += t1; top5 += t5; total += 1
        status = '✓' if t1 else ('~' if t5 else '✗')
        print(f"    {src_w}→{top5_w[0]:<12s} (want {tgt_w:<12s}) {status}  top5={top5_w}")
    if total > 0:
        print(f"  Top-1: {top1}/{total}  Top-5: {top5}/{total}")
        oq8_results[operator_name] = {'top1': top1, 'top5': top5, 'total': total}


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"OQ9 — Zone Trajectory: Tracing Verb Forms Through φ-Space")
print(f"{'='*65}")

print(f"\n  For each verb, plot (base, past, gerund, 3sg) positions relative to each other:")
print(f"  Cosine distances between forms of the same verb")

oq9_results = {}
for base, past, gerund, sg3 in verb_groups:
    fb = find_word(base, w2l)
    fp = find_word(past, w2l)
    fg = find_word(gerund, w2l)
    f3 = find_word(sg3, w2l)
    forms = [(n, find_word(w, w2l)) for n, w in [('base',base),('past',past),('ger',gerund),('3sg',sg3)]]
    forms = [(n, f) for n, f in forms if f]
    if len(forms) < 2:
        continue
    phi_forms = {n: wmap_phi[w2l[f]] for n, f in forms}
    pairs_within = []
    for i, (n1, f1) in enumerate(forms):
        for j, (n2, f2) in enumerate(forms):
            if i < j:
                c = float(phi_forms[n1] @ phi_forms[n2])
                pairs_within.append(c)
    mean_within = float(np.mean(pairs_within))
    # Distance of each form from the B000 centroid (= ~φ₀ proxy)
    dists_from_phi0 = {n: float(phi_forms[n] @ phi0) for n, _ in forms}
    print(f"  {base:<8s} within-form mean cos={mean_within:.3f}  "
          f"φ₀-sims: " + " ".join(f"{n}={v:.2f}" for n, v in dists_from_phi0.items()))
    oq9_results[base] = {'within_mean_cos': mean_within, 'phi0_sims': dists_from_phi0}

# Aggregate: do all forms of a verb cluster more tightly than between-verb?
all_within_cos = [v['within_mean_cos'] for v in oq9_results.values()]
# Between-verb sample: random pairs across different verbs
verb_list = list(oq9_results.keys())
between_cos_samples = []
for i in range(len(verb_list)):
    for j in range(i+1, len(verb_list)):
        fb_i = find_word(verb_list[i], w2l)
        fb_j = find_word(verb_list[j], w2l)
        if fb_i and fb_j:
            between_cos_samples.append(float(wmap_phi[w2l[fb_i]] @ wmap_phi[w2l[fb_j]]))

print(f"\n  Within-verb mean cos: {np.mean(all_within_cos):.3f} ± {np.std(all_within_cos):.3f}")
print(f"  Between-verb mean cos: {np.mean(between_cos_samples):.3f} ± {np.std(between_cos_samples):.3f}")
sep_ratio = (np.mean(all_within_cos) - np.mean(between_cos_samples)) / (np.std(between_cos_samples) + 1e-20)
print(f"  Separation ratio: {sep_ratio:.2f}σ")
print(f"  (For Zone C bodies: Sep/Spread was ~2.20 — how does this compare?)")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SYNTHESIS — What IS Zone D?")
print(f"{'='*65}")

print(f"\n  OQ6: B000 eff_rank={oq6_stats['b000']['eff_rank']:.1f} vs Zone C={oq6_stats['zone_c']['eff_rank']:.1f}")
print(f"       B000 spread={oq6_stats['b000']['spread']:.4f} vs Zone C={oq6_stats['zone_c']['spread']:.4f}")
print(f"       B000 Axis1 var={oq6_stats['b000']['ax1_var']:.3f} vs Zone C={oq6_stats['zone_c']['ax1_var']:.3f}")
print(f"  Interpretation: " + ("B000 is MORE spread/higher rank than Zone C → truly featureless ocean"
                                if oq6_stats['b000']['eff_rank'] > oq6_stats['zone_c']['eff_rank']
                                else "B000 has structure comparable to Zone C"))

if all_within_cos:
    sep = (np.mean(all_within_cos) - np.mean(between_cos_samples)) / (np.std(between_cos_samples) + 1e-20)
    print(f"\n  OQ9: Verb form clustering: {sep:.2f}σ separation")
    if sep > 1.0:
        print(f"  → All morphological forms of a verb cluster together in φ-space")
        print(f"  → Verb identity is geometrically coherent even across zone boundaries")
    else:
        print(f"  → Verb forms scatter across φ-space — zone crossing destroys verb identity")

result = {
    'meta': {'experiment': 'Day 39 — Verb Geometry'},
    'oq6':  oq6_stats,
    'oq8':  oq8_results,
    'oq9':  oq9_results,
    'tense_pairwise_cos': {
        'base_to_past':   pairwise_cos(base_to_past_vecs),
        'base_to_gerund': pairwise_cos(base_to_gerund_vecs),
        'base_to_3sg':    pairwise_cos(base_to_3sg_vecs),
        'past_to_gerund': pairwise_cos(past_to_gerund_vecs),
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 39 complete.")
