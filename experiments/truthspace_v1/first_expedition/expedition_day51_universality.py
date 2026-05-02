#!/usr/bin/env python3
"""
Expedition Day 51 — Is TruthSpace Geometry Universal?

Three levels of evidence, cheapest first:

  Test 1: Split-half axis stability (no model needed)
    Split Zone C vocabulary into two random halves, build SVD axes independently
    from each half's body centroids, measure cosine alignment of corresponding axes.
    High alignment → geometry is a stable intrinsic property of the space.
    Low alignment → geometry is overfitting the specific word sample.

  Test 2: T2 operator leave-one-out generalization (no model needed)
    For each T2 operator, hold out one seed pair, build operator from rest,
    measure how well the held-out pair's direction is predicted.
    High consistency → T2 directions are stable geometric features.

  Test 3: Inter-body metric stability (no model needed)
    Compute pairwise cosine distances between all body centroids from two
    independent vocabulary halves.  Rank-correlate the two distance matrices.
    High correlation → the metric (not just topology) is stable.

  Test 4: Cross-lingual co-location (loads model)
    For a set of English–Chinese translation pairs, get L14 hidden states,
    project to Zone C (Z2 removal + φ-normalise), measure:
    (a) do translation partners occupy the same Zone C position?
    (b) is d(king_EN, queen_EN) ≈ d(king_ZH, queen_ZH)?
    This is the within-model universality test.

Verdict criteria:
    UNIVERSAL:   T1 mean cos > 0.7  AND  T2 LOO cos > 0.7  AND  T3 ρ > 0.9
    STABLE:      T1 > 0.5  AND  T2 > 0.5  AND  T3 > 0.7
    RELATIVE:    any of the above fails
"""

import json, os, random
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day51_universality.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

N_AXES      = 43
N_SPLITS    = 10       # number of random split-half replications
RANDOM_SEED = 42

print("=" * 70)
print("  Expedition Day 51 — Universality of TruthSpace Geometry")
print("=" * 70)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load cached data ──────────────────────────────────────────────────────────
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

wmap_words = [w for w in wmap.keys() if w in w2i]
w2l        = {w: i for i, w in enumerate(wmap_words)}
print(f"  Zone C: {len(zone_c_words)}  wmap: {len(wmap_words)}")


# ── Core geometry builder ─────────────────────────────────────────────────────
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

# Build Z2 and full φ-vectors once
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

wmap_idx  = np.array([w2i[w] for w in wmap_words])
wmap_phi  = batch_phi(hs14_all[wmap_idx], z2)

zone_c_idx = np.array([w2i[w] for w in zone_c_words])
phi_c14    = batch_phi(hs14_all[zone_c_idx], z2)

# Build full body centroids (reference)
body_members_full = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members_full[zone_c_bodies[w]].append(i)

def build_axes_from_subset(body_members_sub, min_members=3):
    """Build SVD axes from a subset of body members. Returns (Vt, bodies_list)."""
    body_centroids_sub = {}
    for body, idxs in body_members_sub.items():
        if len(idxs) < min_members: continue
        vecs = phi_c14[idxs]
        c = vecs.mean(axis=0)
        nm = np.linalg.norm(c)
        if nm > 1e-20:
            body_centroids_sub[body] = c / nm
    if len(body_centroids_sub) < 5:
        return None, []
    bodies_list = sorted(body_centroids_sub.keys())
    C = np.stack([body_centroids_sub[b] for b in bodies_list])
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:N_AXES], bodies_list, body_centroids_sub


# ── Test 1: Split-Half Axis Stability ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Test 1 — Split-Half Axis Stability")
print(f"  {N_SPLITS} random splits of Zone C vocabulary (by word, stratified by body)")
print(f"  Measure cosine between corresponding axes from each half")
print(f"{'='*70}")

split_cos_records = []

for split_i in range(N_SPLITS):
    # Stratified split: for each body, shuffle members and split 50/50
    half_A = defaultdict(list)
    half_B = defaultdict(list)
    for body, idxs in body_members_full.items():
        shuffled = idxs[:]
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        for idx in shuffled[:mid]:  half_A[body].append(idx)
        for idx in shuffled[mid:]:  half_B[body].append(idx)

    result_A = build_axes_from_subset(half_A)
    result_B = build_axes_from_subset(half_B)
    if result_A[0] is None or result_B[0] is None:
        continue

    Vt_A, bodies_A, _ = result_A
    Vt_B, bodies_B, _ = result_B

    # For each axis pair, find best match (handle possible sign flips)
    n = min(len(Vt_A), len(Vt_B), N_AXES)
    cos_matrix = np.abs(Vt_A[:n] @ Vt_B[:n].T)   # (n, n), absolute cosines

    # Greedy best-match assignment
    matched_cos = []
    used_B = set()
    for ax_a in range(n):
        row = [(cos_matrix[ax_a, ax_b], ax_b) for ax_b in range(n)
               if ax_b not in used_B]
        if not row: break
        best_cos, best_b = max(row)
        matched_cos.append(best_cos)
        used_B.add(best_b)

    mean_cos = float(np.mean(matched_cos))
    top5_mean = float(np.mean(matched_cos[:5]))
    split_cos_records.append({'split': split_i, 'mean_cos': mean_cos,
                               'top5_mean': top5_mean,
                               'per_axis': matched_cos[:N_AXES]})

overall_mean = np.mean([r['mean_cos'] for r in split_cos_records])
top5_mean    = np.mean([r['top5_mean'] for r in split_cos_records])
print(f"\n  Mean cosine across {len(split_cos_records)} splits:")
print(f"    All {N_AXES} axes: {overall_mean:.4f}")
print(f"    Top-5 axes:       {top5_mean:.4f}")

# Per-axis stability — pad each row to N_AXES so the 2D array is well-formed
padded_rows = []
for r in split_cos_records:
    row = r['per_axis'][:N_AXES]
    if len(row) < N_AXES:
        row = row + [float('nan')] * (N_AXES - len(row))
    padded_rows.append(row)
per_axis_means = np.nanmean(np.array(padded_rows), axis=0) if padded_rows else np.array([])
print(f"\n  Per-axis mean cosine (axes 1–20):")
for ax in range(min(20, len(per_axis_means))):
    v = per_axis_means[ax]
    bar = '█' * int(v * 20) if not np.isnan(v) else '(nan)'
    print(f"    Axis {ax+1:>2d}: {v:.4f}  {bar}" if not np.isnan(v) else f"    Axis {ax+1:>2d}: nan")

verdict_t1 = ('UNIVERSAL (>0.7)' if overall_mean > 0.7 else
              'STABLE (0.5-0.7)' if overall_mean > 0.5 else
              'RELATIVE (<0.5)')
print(f"\n  T1 verdict: mean={overall_mean:.4f}  → {verdict_t1}")


# ── Test 2: T2 Operator Leave-One-Out Generalization ─────────────────────────
print(f"\n{'='*70}")
print(f"Test 2 — T2 Operator Leave-One-Out Generalization")
print(f"  Hold out one seed pair per operator, build from rest, measure cos")
print(f"{'='*70}")

t2_loo_results = {}
print(f"\n  {'Operator':<22s}  {'Held-out pair':<25s}  cos(full, loo)  cos(held-out dir)")
print(f"  {'-'*80}")

all_loo_cos_full = []
all_loo_cos_held = []

for t2_label, pairs in T2_SEEDS:
    # Build full operator
    all_vecs = []
    for src, tgt in pairs:
        for pfx in [' ', '']:
            ws, wt = pfx+src, pfx+tgt
            if ws in w2l and wt in w2l:
                v = wmap_phi[w2l[wt]] - wmap_phi[w2l[ws]]
                nm = np.linalg.norm(v)
                if nm > 1e-20: all_vecs.append(v / nm)
                break
    if len(all_vecs) < 3: continue
    full_op = mean_unit_vec(np.stack(all_vecs))

    loo_cos_full_list = []
    loo_cos_held_list = []
    pair_labels = []
    for hold_i, (src, tgt) in enumerate(pairs):
        src_key = tgt_key = None
        for pfx in [' ', '']:
            if pfx+src in w2l and pfx+tgt in w2l:
                src_key, tgt_key = pfx+src, pfx+tgt; break
        if src_key is None: continue
        held_vec = wmap_phi[w2l[tgt_key]] - wmap_phi[w2l[src_key]]
        hnm = np.linalg.norm(held_vec)
        if hnm < 1e-20: continue
        held_vec /= hnm

        loo_vecs = [v for j, v in enumerate(all_vecs) if j != hold_i]
        if not loo_vecs: continue
        loo_op = mean_unit_vec(np.stack(loo_vecs))

        cos_full = float(abs(loo_op @ full_op))
        cos_held = float(abs(loo_op @ held_vec))
        loo_cos_full_list.append(cos_full)
        loo_cos_held_list.append(cos_held)
        pair_labels.append(f"{src}→{tgt}")

    if not loo_cos_full_list: continue
    mean_full = np.mean(loo_cos_full_list)
    mean_held = np.mean(loo_cos_held_list)
    all_loo_cos_full.extend(loo_cos_full_list)
    all_loo_cos_held.extend(loo_cos_held_list)
    t2_loo_results[t2_label] = {
        'mean_cos_full': float(mean_full),
        'mean_cos_held': float(mean_held),
        'per_pair': list(zip(pair_labels, loo_cos_held_list))
    }
    worst_pair = pair_labels[np.argmin(loo_cos_held_list)]
    print(f"  {t2_label:<22s}  worst={worst_pair:<20s}  "
          f"{mean_full:.4f}          {mean_held:.4f}")

mean_held_global = float(np.mean(all_loo_cos_held))
mean_full_global = float(np.mean(all_loo_cos_full))
print(f"\n  Global LOO cosine (operator vs held-out pair): {mean_held_global:.4f}")
print(f"  Global LOO cosine (loo_op vs full_op):         {mean_full_global:.4f}")

verdict_t2 = ('UNIVERSAL (>0.7)' if mean_held_global > 0.7 else
              'STABLE (0.5-0.7)' if mean_held_global > 0.5 else
              'RELATIVE (<0.5)')
print(f"  T2 verdict: → {verdict_t2}")


# ── Test 3: Inter-Body Metric Stability ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"Test 3 — Inter-Body Metric Stability")
print(f"  Do pairwise body-centroid distances replicate across vocabulary splits?")
print(f"  Rank-correlate two distance matrices from independent halves")
print(f"{'='*70}")

metric_rhos = []
for split_i in range(N_SPLITS):
    half_A = defaultdict(list)
    half_B = defaultdict(list)
    for body, idxs in body_members_full.items():
        shuffled = idxs[:]
        random.shuffle(shuffled)
        mid = max(1, len(shuffled) // 2)
        for idx in shuffled[:mid]:  half_A[body].append(idx)
        for idx in shuffled[mid:]:  half_B[body].append(idx)

    result_A = build_axes_from_subset(half_A)
    result_B = build_axes_from_subset(half_B)
    if result_A[0] is None or result_B[0] is None: continue

    _, _, cents_A = result_A
    _, _, cents_B = result_B

    # Common bodies with enough members in both halves
    common_bodies = sorted(set(cents_A) & set(cents_B))
    if len(common_bodies) < 5: continue

    C_A = np.stack([cents_A[b] for b in common_bodies])
    C_B = np.stack([cents_B[b] for b in common_bodies])

    # Pairwise cosine distances
    n = len(common_bodies)
    dist_A, dist_B = [], []
    for i in range(n):
        for j in range(i+1, n):
            dist_A.append(1 - float(C_A[i] @ C_A[j]))
            dist_B.append(1 - float(C_B[i] @ C_B[j]))

    rho, _ = spearmanr(dist_A, dist_B)
    metric_rhos.append(float(rho))

mean_rho = float(np.mean(metric_rhos))
print(f"\n  Mean Spearman ρ across {len(metric_rhos)} splits: {mean_rho:.4f}")
print(f"  Min: {min(metric_rhos):.4f}  Max: {max(metric_rhos):.4f}")

verdict_t3 = ('UNIVERSAL (>0.9)' if mean_rho > 0.9 else
              'STABLE (0.7-0.9)' if mean_rho > 0.7 else
              'RELATIVE (<0.7)')
print(f"  T3 verdict: ρ={mean_rho:.4f}  → {verdict_t3}")


# ── Test 4: Cross-Lingual Co-Location ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Test 4 — Cross-Lingual Co-Location (loads model)")
print(f"  Are translation partners at the same Zone C address?")
print(f"{'='*70}")

# Translation pairs: (English, Chinese, concept)
EN_ZH_PAIRS = [
    ('king',   '国王',  'royalty'),
    ('queen',  '女王',  'royalty'),
    ('man',    '男人',  'gender'),
    ('woman',  '女人',  'gender'),
    ('cat',    '猫',    'animal'),
    ('dog',    '狗',    'animal'),
    ('running','跑步',  'motion'),
    ('walking','走路',  'motion'),
    ('quickly','快速地','manner'),
    ('beautiful','美丽','quality'),
    ('decision','决定', 'cognition'),
    ('family',  '家庭', 'social'),
    ('soldier', '士兵', 'military'),
    ('scientist','科学家','occupation'),
    ('philosophy','哲学','abstract'),
]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map='cpu',
    output_hidden_states=True, attn_implementation='eager')
model.eval()

def get_l14_phi(text):
    """Get φ-vector (Z2-removed, L2-normalised) at L14 for the last token."""
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[14][0, -1].numpy().astype(np.float64)
    nm = np.linalg.norm(h)
    if nm < 1e-20: return None
    h_n = h / nm
    h_phi = h_n - (h_n @ z2) * z2
    pnm = np.linalg.norm(h_phi)
    return h_phi / pnm if pnm > 1e-20 else None

print(f"\n  Loading model and computing cross-lingual vectors...")
print(f"\n  {'Concept':<12s}  {'EN word':<12s}  {'ZH word':<10s}  "
      f"cos(EN,ZH)   d(EN,ZH)   Verdict")
print(f"  {'-'*75}")

t4_results = []
en_phi_vecs = {}
zh_phi_vecs = {}

for en_word, zh_word, concept in EN_ZH_PAIRS:
    phi_en = get_l14_phi(' ' + en_word)
    phi_zh = get_l14_phi(zh_word)
    if phi_en is None or phi_zh is None:
        print(f"  {concept:<12s}  {en_word:<12s}  {zh_word:<10s}  [SKIP]")
        continue

    cos_val = float(phi_en @ phi_zh)
    dist    = 1 - cos_val
    verdict = ('CLOSE (<0.3)' if dist < 0.3 else
               'NEAR  (0.3-0.6)' if dist < 0.6 else
               'FAR   (>0.6)')
    en_phi_vecs[en_word] = phi_en
    zh_phi_vecs[zh_word] = phi_zh
    t4_results.append({'en': en_word, 'zh': zh_word, 'concept': concept,
                        'cos': cos_val, 'dist': dist})
    print(f"  {concept:<12s}  {en_word:<12s}  {zh_word:<10s}  "
          f"{cos_val:.4f}       {dist:.4f}     {verdict}")

if t4_results:
    mean_cos_t4 = np.mean([r['cos'] for r in t4_results])
    mean_dist_t4 = np.mean([r['dist'] for r in t4_results])
    print(f"\n  Mean cos(EN,ZH): {mean_cos_t4:.4f}   Mean dist: {mean_dist_t4:.4f}")

    # Distance preservation test: does d(EN_A, EN_B) ≈ d(ZH_A, ZH_B)?
    print(f"\n  Metric preservation: d(en_A, en_B) vs d(zh_A, zh_B)")
    print(f"  (for translation pairs with both languages present)")
    dist_en_list, dist_zh_list = [], []
    en_words = list(en_phi_vecs.keys())
    for i in range(len(en_words)):
        for j in range(i+1, len(en_words)):
            wA, wB = en_words[i], en_words[j]
            # Find corresponding ZH words
            zh_A = next((r['zh'] for r in t4_results if r['en']==wA), None)
            zh_B = next((r['zh'] for r in t4_results if r['en']==wB), None)
            if zh_A not in zh_phi_vecs or zh_B not in zh_phi_vecs: continue
            d_en = 1 - float(en_phi_vecs[wA] @ en_phi_vecs[wB])
            d_zh = 1 - float(zh_phi_vecs[zh_A] @ zh_phi_vecs[zh_B])
            dist_en_list.append(d_en)
            dist_zh_list.append(d_zh)

    if len(dist_en_list) > 5:
        rho_t4, _ = spearmanr(dist_en_list, dist_zh_list)
        print(f"  Spearman ρ(d_EN, d_ZH) = {rho_t4:.4f}  ({len(dist_en_list)} pairs)")
        verdict_t4 = ('UNIVERSAL (>0.9)' if rho_t4 > 0.9 else
                      'STABLE (0.7-0.9)' if rho_t4 > 0.7 else
                      'RELATIVE (<0.7)')
        print(f"  T4 metric verdict: ρ={rho_t4:.4f}  → {verdict_t4}")
    else:
        rho_t4 = None
        print(f"  Not enough pairs for metric test")
else:
    rho_t4 = None


# ── Final Verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL VERDICT")
print(f"{'='*70}")

print(f"""
  T1 Split-half axis stability:       mean cos = {overall_mean:.4f}  → {verdict_t1}
  T2 T2 operator LOO generalisation:  mean cos = {mean_held_global:.4f}  → {verdict_t2}
  T3 Inter-body metric stability:     Spearman ρ = {mean_rho:.4f}  → {verdict_t3}
  T4 Cross-lingual co-location:       mean cos(EN,ZH) = {mean_cos_t4 if t4_results else float('nan'):.4f}
""")

universal = (overall_mean > 0.7 and mean_held_global > 0.7 and mean_rho > 0.9)
stable    = (overall_mean > 0.5 and mean_held_global > 0.5 and mean_rho > 0.7)

if universal:
    verdict = "UNIVERSAL — geometry is model-agnostic; TruthSpace may be discovering Platonic structure"
elif stable:
    verdict = "STABLE — geometry replicates within training distribution; cross-model test still needed"
else:
    verdict = "RELATIVE — geometry is sensitive to vocabulary sample; intrinsic claims need revision"
print(f"  OVERALL: {verdict}")

print(f"""
  Interpretation:
    Axis stability tells us whether the SVD basis is reproducible.
    T2 consistency tells us whether the transformation directions are stable.
    Metric stability tells us whether DISTANCES (the metric) are reproducible.
    Cross-lingual tells us whether the geometry is language-invariant.

    The metric (T3) is more fundamental than the axes (T1) because axes are
    just a chart. High metric stability + low axis stability would mean the
    geometry is real but we're measuring it with an unstable ruler.
    High metric + high axis = the geometry is stable AND we have a good chart.
""")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta':            {'experiment': 'Day 51 — Universality Investigation'},
    't1_split_half':   to_json({'mean_cos': overall_mean, 'top5_mean': top5_mean,
                                'per_axis_means': per_axis_means.tolist(),
                                'per_split': split_cos_records}),
    't2_t2_loo':       to_json({'mean_cos_held': mean_held_global,
                                'mean_cos_full': mean_full_global,
                                'per_operator': t2_loo_results}),
    't3_metric':       to_json({'mean_rho': mean_rho, 'per_split': metric_rhos}),
    't4_crosslingual': to_json({'pairs': t4_results,
                                'metric_rho': rho_t4}),
    'verdict':         verdict,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 51 complete.")
