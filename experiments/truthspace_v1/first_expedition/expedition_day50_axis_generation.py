#!/usr/bin/env python3
"""
Expedition Day 50 — Axis-Based Generation Targeting

The 43 concept axes (Day 42) form a coordinate system over Zone C concept space.
Every word with a Zone C body has a 43-dimensional address in axis space.

The question: can we USE this address space for generation without running the LM?

Operations to test:
  1. Body retrieval   — given a body label, find nearest words in axis space
  2. Coordinate inversion — given axis coords, reconstruct φ-vector, find words
  3. T2 navigation    — word A + T2 delta vector → find word B at new address
                        (king + gender_delta → queen?)
  4. Interpolation    — midpoint between two body centroids → what concept lives there?
  5. Precision test   — for known word pairs, does axis-space retrieval beat cosine-in-φ?

If T2 navigation works, the axis space is a generative coordinate system:
  - Specify target meaning as axis coordinates
  - Reconstruct approximate φ-vector
  - Find nearest word → retrieval without sampling
"""

import json, os
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day50_axis_generation.json")

N_AXES = 43   # effective rank

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

T2_SEEDS = [
    ('comp→sup',        [('bigger','biggest'), ('faster','fastest'), ('older','oldest'),
                         ('smaller','smallest'), ('stronger','strongest')]),
    ('base→comp',       [('big','bigger'), ('fast','faster'), ('old','older'),
                         ('small','smaller'), ('strong','stronger')]),
    ('singular→plural', [('cat','cats'), ('dog','dogs'), ('tree','trees'),
                         ('bird','birds'), ('house','houses'), ('book','books')]),
    ('male→female',     [('man','woman'), ('king','queen'), ('boy','girl'),
                         ('actor','actress'), ('son','daughter')]),
    ('base→adverb',     [('quick','quickly'), ('slow','slowly'), ('quiet','quietly'),
                         ('loud','loudly'), ('soft','softly')]),
    ('base→gerund',     [('run','running'), ('walk','walking'), ('sing','singing'),
                         ('play','playing'), ('talk','talking')]),
    ('gerund→past',     [('running','ran'), ('walking','walked'), ('singing','sang'),
                         ('playing','played'), ('talking','talked')]),
]

print("=" * 70)
print("  Expedition Day 50 — Axis-Based Generation Targeting")
print("=" * 70)


# ── Load ──────────────────────────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words = [w for w, v in wmap.items()
                if v['phase'] == 2 and v.get('L14_body') not in ('B000','B001',None)
                and w in w2i]
zone_c_bodies = {w: wmap[w]['L14_body'] for w in zone_c_words}
body_label_map = {}
for w, v in wmap.items():
    b = v.get('L14_body')
    if b and b not in body_label_map:
        body_label_map[b] = v.get('L14_label', '?')

wmap_words = [w for w in wmap.keys() if w in w2i]
w2l        = {w: i for i, w in enumerate(wmap_words)}
print(f"  {len(words_all):,} words total  |  Zone C: {len(zone_c_words)}  |  wmap: {len(wmap_words)}")


# ── Rebuild geometry ──────────────────────────────────────────────────────────
def batch_phi(hs, z2):
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

def mean_unit_vec(vecs):
    v = np.mean(vecs, axis=0)
    nm = np.linalg.norm(v)
    return v / nm if nm > 1e-20 else v

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

# φ-vectors for all wmap words
wmap_idx = np.array([w2i[w] for w in wmap_words])
wmap_phi = batch_phi(hs14_all[wmap_idx], z2)    # (n_wmap, D)

# φ-vectors for Zone C words only
zone_c_idx = np.array([w2i[w] for w in zone_c_words])
phi_c14    = batch_phi(hs14_all[zone_c_idx], z2)

# Body centroids
body_members = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

body_centroids = {}
body_words_map = {}
for body, idxs in body_members.items():
    vecs = phi_c14[idxs]
    c = vecs.mean(axis=0)
    body_centroids[body] = c / (np.linalg.norm(c) + 1e-20)
    body_words_map[body] = [zone_c_words[i] for i in idxs]

# SVD of body-centroid matrix → axis vectors
bodies_list = sorted(body_centroids.keys())
C           = np.stack([body_centroids[b] for b in bodies_list])  # (n_bodies, D)
_, sv_c, Vt_c = np.linalg.svd(C, full_matrices=False)

# Axis basis (first N_AXES rows of Vt_c)
AXES   = Vt_c[:N_AXES]   # (43, D)  — each row is one axis direction in φ-space
print(f"  Axis basis: {AXES.shape[0]} axes × {AXES.shape[1]} dims")

# Project all wmap words into axis-coordinate space
# coords_wmap[i] = 43-dim coordinate vector for wmap_words[i]
coords_wmap = wmap_phi @ AXES.T   # (n_wmap, 43)

# Project all body centroids into axis space
coords_bodies = C @ AXES.T        # (n_bodies, 43)
body_coords   = {b: coords_bodies[i] for i, b in enumerate(bodies_list)}

# T2 operator vectors in φ-space and in axis-coord space
t2_phi_vecs  = {}
t2_ax_vecs   = {}
for t2_label, pairs in T2_SEEDS:
    vecs = []
    for src, tgt in pairs:
        for pfx in [' ', '']:
            ws, wt = pfx+src, pfx+tgt
            if ws in w2l and wt in w2l:
                v = wmap_phi[w2l[wt]] - wmap_phi[w2l[ws]]
                nm = np.linalg.norm(v)
                if nm > 1e-20: vecs.append(v / nm)
                break
    if vecs:
        phi_vec             = mean_unit_vec(np.stack(vecs))
        t2_phi_vecs[t2_label] = phi_vec
        t2_ax_vecs[t2_label]  = phi_vec @ AXES.T   # project delta into axis space

print(f"  T2 operators: {list(t2_phi_vecs.keys())}")


# ── Helper: k-nearest neighbours in axis coord space ─────────────────────────
def knn_axis(target_coord, k=10, exclude=None):
    """Find k nearest wmap words to target_coord (43-dim) using L2 in axis space."""
    diff = coords_wmap - target_coord[None, :]
    dists = np.linalg.norm(diff, axis=1)
    order = np.argsort(dists)
    results = []
    for idx in order:
        w = wmap_words[idx]
        if exclude and w.strip() in exclude: continue
        results.append({
            'word':  w.strip(),
            'dist':  float(dists[idx]),
            'body':  wmap.get(w, {}).get('L14_body', '?'),
        })
        if len(results) >= k: break
    return results

def knn_phi(target_phi_vec, k=10, exclude=None):
    """Find k nearest wmap words to target_phi_vec by cosine similarity."""
    sims  = wmap_phi @ target_phi_vec
    order = np.argsort(sims)[::-1]
    results = []
    for idx in order:
        w = wmap_words[idx]
        if exclude and w.strip() in exclude: continue
        results.append({
            'word': w.strip(),
            'cos':  float(sims[idx]),
            'body': wmap.get(w, {}).get('L14_body', '?'),
        })
        if len(results) >= k: break
    return results


# ── Section 1: Body Retrieval ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 1 — Body Retrieval")
print(f"  Given a body label, retrieve nearest words in axis-coordinate space")
print(f"  Expected: top-k ≈ the members of that body")
print(f"{'='*70}")

# Test on 8 varied bodies
TEST_BODIES = ['B008', 'B009', 'B015', 'B025', 'B039', 'B047', 'B065', 'B083']
s1_results = {}
print(f"\n  {'Body':<8s}  {'Label':<30s}  Members(n)  Top-5 retrieved  [match %]")
print(f"  {'-'*90}")
for body in TEST_BODIES:
    if body not in body_coords: continue
    label   = body_label_map.get(body, '?')
    members = set(w.strip() for w in body_words_map.get(body, []))
    target  = body_coords[body]
    hits    = knn_axis(target, k=20, exclude=None)
    # how many of top-10 are actual body members?
    top10   = [h for h in hits[:10]]
    in_body = sum(1 for h in top10 if h['word'] in members)
    top5_str = ', '.join(h['word'] for h in hits[:5])
    pct = in_body / 10 * 100
    s1_results[body] = {'label': label, 'top10': top10, 'precision_at_10': pct/100}
    print(f"  {body:<8s}  {label[:30]:<30s}  n={len(members):<5d}  {top5_str}  [{pct:.0f}%]")


# ── Section 2: Coordinate Inversion ──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — Coordinate Inversion")
print(f"  Take a word's axis coords → reconstruct φ-vector → find nearest words")
print(f"  If inversion is clean, query word should rank #1")
print(f"{'='*70}")

INVERSION_WORDS = [
    'king', 'queen', 'quickly', 'running', 'cats',
    'decisions', 'humanity', 'beautiful', 'microscope', 'orchestra',
]

print(f"\n  {'Word':<15s}  Rank of self  Top-3 retrieved  Axis-cos  φ-cos")
print(f"  {'-'*70}")
s2_results = {}
for word in INVERSION_WORDS:
    for pfx in [' ', '']:
        w = pfx + word
        if w in w2l:
            orig_coord = coords_wmap[w2l[w]]           # 43-dim
            orig_phi   = wmap_phi[w2l[w]]              # full φ-vec

            # Reconstruct φ-vector from axis coords
            recon_phi  = orig_coord @ AXES              # (43,)·(43,D) = (D,)
            recon_phi /= (np.linalg.norm(recon_phi) + 1e-20)

            # Find self in axis-coord retrieval
            hits_ax = knn_axis(orig_coord, k=20)
            rank_ax = next((i+1 for i, h in enumerate(hits_ax) if h['word'] == word), '>20')

            # Find self via reconstructed φ-vector
            hits_ph = knn_phi(recon_phi, k=20)
            rank_ph = next((i+1 for i, h in enumerate(hits_ph) if h['word'] == word), '>20')

            # Cosine between original and reconstructed φ
            ax_cos  = float(orig_phi @ recon_phi)
            top3    = ', '.join(h['word'] for h in hits_ax[:3])
            s2_results[word] = {
                'rank_axis': rank_ax, 'rank_phi': rank_ph,
                'recon_cos': ax_cos, 'top3': top3
            }
            print(f"  {word:<15s}  ax:{rank_ax:<3} φ:{rank_ph:<3}  {top3:<35s}  {ax_cos:.4f}")
            break


# ── Section 3: T2 Navigation ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — T2 Navigation in Axis Space")
print(f"  word_A + T2_delta → nearest word at new coords = word_B?")
print(f"  This is word-analogy retrieval without running the LM")
print(f"{'='*70}")

ANALOGY_TESTS = [
    # (source_word, t2_operator, expected_target)
    ('king',    'male→female',     'queen'),
    ('man',     'male→female',     'woman'),
    ('boy',     'male→female',     'girl'),
    ('actor',   'male→female',     'actress'),
    ('son',     'male→female',     'daughter'),
    ('cat',     'singular→plural', 'cats'),
    ('dog',     'singular→plural', 'dogs'),
    ('tree',    'singular→plural', 'trees'),
    ('bird',    'singular→plural', 'birds'),
    ('big',     'base→comp',       'bigger'),
    ('fast',    'base→comp',       'faster'),
    ('old',     'base→comp',       'older'),
    ('bigger',  'comp→sup',        'biggest'),
    ('faster',  'comp→sup',        'fastest'),
    ('run',     'base→gerund',     'running'),
    ('walk',    'base→gerund',     'walking'),
    ('quick',   'base→adverb',     'quickly'),
    ('slow',    'base→adverb',     'slowly'),
    ('running', 'gerund→past',     'ran'),
    ('walking', 'gerund→past',     'walked'),
]

print(f"\n  Testing in AXIS-COORD space (L2 distance after adding T2 delta):")
print(f"  {'Source':<12s}  {'T2 op':<20s}  {'Expected':<12s}  "
      f"{'Rank(axis)':<12s}  {'Rank(φ)':<10s}  Top-3")
print(f"  {'-'*100}")

s3_results = {'axis': [], 'phi': []}
ax_correct = 0
ph_correct = 0
for src_word, t2_op, tgt_word in ANALOGY_TESTS:
    if t2_op not in t2_ax_vecs: continue

    # Find source word (try space-prefixed first)
    src_key = None
    for pfx in [' ', '']:
        if pfx + src_word in w2l:
            src_key = pfx + src_word; break
    if src_key is None: continue

    src_coord = coords_wmap[w2l[src_key]]
    src_phi   = wmap_phi[w2l[src_key]]

    # Apply T2 delta in axis-coord space
    delta_ax  = t2_ax_vecs[t2_op]
    target_coord = src_coord + delta_ax
    hits_ax = knn_axis(target_coord, k=20, exclude={src_word})
    rank_ax = next((i+1 for i, h in enumerate(hits_ax) if h['word'] == tgt_word), '>20')

    # Apply T2 delta in φ-space (for comparison)
    delta_ph  = t2_phi_vecs[t2_op]
    target_phi = src_phi + delta_ph
    target_phi /= np.linalg.norm(target_phi)
    hits_ph = knn_phi(target_phi, k=20, exclude={src_word})
    rank_ph = next((i+1 for i, h in enumerate(hits_ph) if h['word'] == tgt_word), '>20')

    top3 = ', '.join(h['word'] for h in hits_ax[:3])
    if isinstance(rank_ax, int) and rank_ax == 1: ax_correct += 1
    if isinstance(rank_ph, int) and rank_ph == 1: ph_correct += 1
    s3_results['axis'].append({'src': src_word, 't2': t2_op, 'tgt': tgt_word,
                                'rank': rank_ax, 'top3': top3})
    s3_results['phi'].append( {'src': src_word, 't2': t2_op, 'tgt': tgt_word,
                                'rank': rank_ph})
    flag_ax = '✓' if rank_ax == 1 else ('~' if isinstance(rank_ax,int) and rank_ax<=3 else '✗')
    flag_ph = '✓' if rank_ph == 1 else ('~' if isinstance(rank_ph,int) and rank_ph<=3 else '✗')
    print(f"  {src_word:<12s}  {t2_op:<20s}  {tgt_word:<12s}  "
          f"{flag_ax}rank={rank_ax!s:<8s}  {flag_ph}rank={rank_ph!s:<6s}  {top3}")

n_tests = len(s3_results['axis'])
print(f"\n  Axis-space top-1 accuracy: {ax_correct}/{n_tests} = {ax_correct/n_tests:.1%}")
print(f"  φ-space    top-1 accuracy: {ph_correct}/{n_tests} = {ph_correct/n_tests:.1%}")


# ── Section 4: Interpolation ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — Interpolation Between Body Centroids")
print(f"  Midpoint of two body centroids in axis space → nearest word")
print(f"  Tests whether axis space is convex (concepts blend smoothly)")
print(f"{'='*70}")

INTERP_PAIRS = [
    ('B015', 'B065', 'family + commanders → ?'),
    ('B008', 'B039', 'renewal + scientific → ?'),
    ('B025', 'B083', 'exchange + overdose → ?'),
    ('B009', 'B047', 'political + size → ?'),
]

print(f"\n  {'Pair':<8s}  {'Label A':<20s}  {'Label B':<20s}  "
      f"{'Note':<25s}  Midpoint top-3")
print(f"  {'-'*110}")
s4_results = []
for b1, b2, note in INTERP_PAIRS:
    if b1 not in body_coords or b2 not in body_coords: continue
    c1     = body_coords[b1]
    c2     = body_coords[b2]
    mid    = (c1 + c2) / 2
    hits   = knn_axis(mid, k=5)
    top3   = ', '.join(h['word'] for h in hits[:3])
    l1     = body_label_map.get(b1, '?')[:20]
    l2     = body_label_map.get(b2, '?')[:20]
    s4_results.append({'b1': b1, 'b2': b2, 'top3': top3})
    print(f"  {b1}+{b2}  {l1:<20s}  {l2:<20s}  {note:<25s}  {top3}")


# ── Section 5: Precision summary ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — Precision Summary")
print(f"{'='*70}")

# Body retrieval precision@10
p10_vals = [v['precision_at_10'] for v in s1_results.values()]
print(f"\n  Body retrieval precision@10: mean = {np.mean(p10_vals):.1%}  "
      f"min = {np.min(p10_vals):.1%}  max = {np.max(p10_vals):.1%}")

# Inversion self-rank
self_ranks_ax = [v['rank_axis'] for v in s2_results.values() if isinstance(v['rank_axis'], int)]
self_ranks_ph = [v['rank_phi']  for v in s2_results.values() if isinstance(v['rank_phi'], int)]
print(f"  Coord inversion self-rank: axis={np.mean(self_ranks_ax):.1f}  φ={np.mean(self_ranks_ph):.1f}")
recon_cos_vals = [v['recon_cos'] for v in s2_results.values()]
print(f"  Axis→φ reconstruction cosine: mean = {np.mean(recon_cos_vals):.4f}")

# T2 navigation
print(f"\n  T2 navigation (top-1 accuracy):")
print(f"    Axis-coord space: {ax_correct}/{n_tests} = {ax_correct/n_tests:.1%}")
print(f"    Full φ-space:     {ph_correct}/{n_tests} = {ph_correct/n_tests:.1%}")
print(f"    (φ-space is the oracle — axis-coord is ~43-dim compression)")

# T2 by operator
print(f"\n  T2 accuracy by operator (axis space):")
by_op = defaultdict(list)
for r in s3_results['axis']:
    by_op[r['t2']].append(r['rank'])
for op, ranks in sorted(by_op.items()):
    correct_1 = sum(1 for r in ranks if r == 1)
    top3_all  = sum(1 for r in ranks if isinstance(r,int) and r <= 3)
    print(f"    {op:<20s}: top-1 {correct_1}/{len(ranks)}  top-3 {top3_all}/{len(ranks)}")

print(f"""
  VERDICT:
    Axis space {'IS' if ax_correct/n_tests >= 0.5 else 'IS NOT'} sufficient for T2 navigation
    (>50% top-1 accuracy would confirm it as a generative coordinate system)

    The 43 concept axes {'capture' if ax_correct/n_tests >= 0.5 else 'do not fully capture'}
    enough structure for analogy-based word retrieval without the LM.
""")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta':            {'experiment': 'Day 50 — Axis Generation Targeting'},
    'body_retrieval':  to_json(s1_results),
    'coord_inversion': to_json(s2_results),
    't2_navigation':   to_json(s3_results),
    'interpolation':   to_json(s4_results),
    'summary': {
        'body_precision_at10_mean': float(np.mean(p10_vals)),
        'inversion_self_rank_mean': float(np.mean(self_ranks_ax)),
        'recon_cos_mean':           float(np.mean(recon_cos_vals)),
        't2_top1_axis':             ax_correct / n_tests,
        't2_top1_phi':              ph_correct / n_tests,
        'n_t2_tests':               n_tests,
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 50 complete.")
