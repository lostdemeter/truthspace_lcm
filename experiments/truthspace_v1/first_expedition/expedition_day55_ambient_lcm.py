#!/usr/bin/env python3
"""
Expedition Day 55 — Ambient LCM: Validating the 4th Dimension Correction

Day 54 proved that non-commutativity was entirely a normalisation artifact,
and that the correct composition rule is:
    φ_result = normalise(φ + Δa + Δb + ...)   ← ambient: normalise ONCE

Day 11 found "simultaneous composition fails" (king + Δgender + Δplural → queen
instead of queens). Day 54 predicts that failure was also a normalisation
artifact — we were using sequential norms.

This script:
  P1  Retest Day 11 "simultaneous fails" using ambient composition
  P2  Full LCM inference comparison: ambient vs sequential (all T2 types)
  P3  Multi-operator ambient composition (3+ operators simultaneously)
  P4  Round-trip accuracy: ambient vs sequential ENCODE=DECODE
  P5  Radial weight: does including ||h|| as a confidence score improve retrieval?
"""

import json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day55_ambient_lcm.json")

np.random.seed(42)

print("=" * 70)
print("  Expedition Day 55 — Ambient LCM")
print("  Does the ambient composition rule improve LCM inference quality?")
print("=" * 70)


# ── Load and build geometry ───────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

raw_norms = np.linalg.norm(hs14_all, axis=1)   # the 4th (radial) dimension

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

def build_z2(pairs, hs):
    ds = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx+a, pfx+b
            if wa in w2i and wb in w2i:
                d = hs[w2i[wb]] - hs[w2i[wa]]
                nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    _, _, Vt = np.linalg.svd(np.stack(ds), full_matrices=False)
    return Vt[0] / np.linalg.norm(Vt[0])

z2 = build_z2(KILLING_PAIRS, hs14_all)

def to_phi(hs_batch, z2):
    H   = hs_batch.astype(np.float64)
    nm  = np.linalg.norm(H, axis=1, keepdims=True)
    Hn  = H / (nm + 1e-20)
    perp = Hn - (Hn @ z2)[:, None] * z2
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

phi_all = to_phi(hs14_all, z2)  # unit φ-vectors for ALL words

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

def phi(w):
    wk = w if w in w2i else w.lstrip()
    if wk not in w2i: wk = ' '+w.lstrip() if ' '+w.lstrip() in w2i else None
    return phi_all[w2i[wk]] if wk else None

def hs(w):
    wk = w if w in w2i else w.lstrip()
    if wk not in w2i: wk = ' '+w.lstrip() if ' '+w.lstrip() in w2i else None
    return hs14_all[w2i[wk]] if wk else None

# retrieval: find nearest φ-vector to query
# Optional: weight by raw norm (radial dimension)
def retrieve(query_phi, exclude=None, top_k=5, norm_weight=0.0):
    """
    Find top_k nearest words by cosine similarity to query_phi.
    If norm_weight > 0, score = (1-norm_weight)*cos + norm_weight*normalised_norm.
    """
    sims = phi_all @ query_phi
    if norm_weight > 0.0:
        norms_norm = (raw_norms - raw_norms.min()) / (raw_norms.max() - raw_norms.min() + 1e-20)
        sims = (1.0 - norm_weight) * sims + norm_weight * norms_norm
    if exclude:
        for w in exclude:
            wk = w if w in w2i else ' '+w.lstrip()
            if wk in w2i: sims[w2i[wk]] = -1.0
    top_idx = np.argsort(-sims)[:top_k]
    return [(words_all[i].strip(), float(sims[i])) for i in top_idx]

def find_rank(query_phi, target, exclude=None):
    sims = phi_all @ query_phi
    if exclude:
        for w in exclude:
            wk = w if w in w2i else ' '+w.lstrip()
            if wk in w2i: sims[w2i[wk]] = -1.0
    order = np.argsort(-sims)
    target_k = target if target in w2i else ' '+target.lstrip()
    if target_k not in w2i: return 9999
    pos = np.where(order == w2i[target_k])[0]
    return int(pos[0]) + 1 if len(pos) > 0 else 9999


# ── Build T2 operators ────────────────────────────────────────────────────────
T2_SEEDS = {
    'male_female':     [(' king',' queen'),(' man',' woman'),(' boy',' girl'),
                        (' actor',' actress'),(' prince',' princess'),
                        (' brother',' sister'),(' father',' mother')],
    'singular_plural': [(' cat',' cats'),(' dog',' dogs'),(' tree',' trees'),
                        (' bird',' birds'),(' book',' books'),
                        (' king',' kings'),(' house',' houses')],
    'base_comp':       [(' big',' bigger'),(' fast',' faster'),(' old',' older'),
                        (' small',' smaller'),(' tall',' taller'),
                        (' cold',' colder'),(' warm',' warmer')],
    'base_past':       [(' walk',' walked'),(' talk',' talked'),
                        (' jump',' jumped'),(' play',' played'),
                        (' watch',' watched'),(' call',' called')],
    'base_adverb':     [(' quick',' quickly'),(' slow',' slowly'),
                        (' clear',' clearly'),(' soft',' softly'),
                        (' quiet',' quietly'),(' loud',' loudly')],
}

def build_t2(pairs):
    ds = []
    for a, b in pairs:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in w2i and wb in w2i:
                pa = phi_all[w2i[wa]]; pb = phi_all[w2i[wb]]
                d = pb - pa; nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    if not ds: return None
    m = np.stack(ds).mean(0); nm = np.linalg.norm(m)
    return m / nm if nm > 1e-20 else None

t2 = {k: build_t2(v) for k, v in T2_SEEDS.items()}

Δg = t2['male_female']     # gender
Δp = t2['singular_plural'] # plural
Δc = t2['base_comp']       # comparative
Δt = t2['base_past']       # past tense
Δa = t2['base_adverb']     # adverb


def apply_sequential(phi_v, ops):
    """Normalise after EVERY step (Day 11 / Day 53 method)."""
    v = phi_v.copy()
    for op in ops:
        v = v + op
        v = v / (np.linalg.norm(v) + 1e-20)
    return v

def apply_ambient(phi_v, ops):
    """Add ALL operators, normalise ONCE (Day 54 correction)."""
    v = phi_v.copy()
    for op in ops: v = v + op
    return v / (np.linalg.norm(v) + 1e-20)


# ── P1: Retest Day 11 — Simultaneous Composition ─────────────────────────────
print(f"\n{'='*70}")
print(f"P1 — Retest Day 11: Simultaneous Composition")
print(f"  Day 11 found: king + Δgender + Δplural → 'queen' (not 'queens')")
print(f"  Day 54 predicts: ambient rule → 'queens' (correct)")
print(f"{'='*70}")

TEST_MULTI = [
    # (source, ops, expected_target)
    (' king',  [Δg, Δp],  'queens'),   # male + singular → female + plural
    (' man',   [Δg, Δp],  'women'),
    (' boy',   [Δg, Δp],  'girls'),
    (' actor', [Δg, Δp],  'actresses'),
    (' cat',   [Δg, Δp],  'cats'),     # cat has no gender → should still pluralise
    (' dog',   [Δp, Δc],  'dogs'),     # plural + comparative (unusual)
    (' big',   [Δc, Δa],  'bigger'),   # comp + adverb
]

print(f"\n  {'Source':<10}  {'Ops':<20}  {'Seq top1':<14}  {'Seq rank':<10}  {'Amb top1':<14}  {'Amb rank'}")
print(f"  {'-'*80}")

p1_results = []
for source, ops, target in TEST_MULTI:
    p_src = phi(source)
    if p_src is None: continue

    ops_names = '+'.join(['Δg' if op is Δg else 'Δp' if op is Δp else
                          'Δc' if op is Δc else 'Δt' if op is Δt else 'Δa'
                          for op in ops])

    phi_seq = apply_sequential(p_src, ops)
    phi_amb = apply_ambient(p_src, ops)

    top_seq = retrieve(phi_seq, exclude=[source], top_k=1)[0][0]
    top_amb = retrieve(phi_amb, exclude=[source], top_k=1)[0][0]
    rank_seq = find_rank(phi_seq, target, exclude=[source])
    rank_amb = find_rank(phi_amb, target, exclude=[source])

    hit_seq = '✓' if rank_seq <= 5 else ' '
    hit_amb = '✓' if rank_amb <= 5 else ' '

    print(f"  {source.strip():<10}  {ops_names:<20}  "
          f"{top_seq:<12}  {hit_seq}{rank_seq:<8}   "
          f"{top_amb:<12}  {hit_amb}{rank_amb}")
    p1_results.append({'source': source.strip(), 'ops': ops_names, 'target': target,
                       'seq_top1': top_seq, 'seq_rank': rank_seq,
                       'amb_top1': top_amb, 'amb_rank': rank_amb})

amb_wins = sum(1 for r in p1_results if r['amb_rank'] < r['seq_rank'])
seq_wins = sum(1 for r in p1_results if r['seq_rank'] < r['amb_rank'])
ties     = len(p1_results) - amb_wins - seq_wins
print(f"\n  Ambient better: {amb_wins}/{len(p1_results)}  |  "
      f"Sequential better: {seq_wins}/{len(p1_results)}  |  "
      f"Tied: {ties}/{len(p1_results)}")


# ── P2: Full LCM Inference Comparison ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P2 — Full LCM Inference: Ambient vs Sequential (all T2 types)")
print(f"  Leave-one-out: train on N-1 seed pairs, test on held-out pair")
print(f"{'='*70}")

# LOO test: for each pair, build T2 from the other N-1 pairs, test on this pair
def loo_test_t2(seeds, method='ambient'):
    results = []
    for held_out in seeds:
        train = [s for s in seeds if s != held_out]
        delta = build_t2(train)
        if delta is None: continue

        source_w, target_w = held_out
        p_src = phi(source_w)
        if p_src is None: continue

        if method == 'ambient':
            p_out = apply_ambient(p_src, [delta])
        else:
            p_out = apply_sequential(p_src, [delta])

        rank = find_rank(p_out, target_w.strip(), exclude=[source_w])
        results.append({'source': source_w.strip(), 'target': target_w.strip(),
                        'rank': rank, 'hit5': rank <= 5})
    return results

print(f"\n  T2 type            N    Seq med   Seq@5    Amb med   Amb@5    Delta")
print(f"  {'-'*70}")

p2_results = {}
for name, seeds in T2_SEEDS.items():
    r_seq = loo_test_t2(seeds, 'sequential')
    r_amb = loo_test_t2(seeds, 'ambient')
    if not r_seq or not r_amb: continue

    med_seq = float(np.median([r['rank'] for r in r_seq]))
    med_amb = float(np.median([r['rank'] for r in r_amb]))
    at5_seq = sum(r['hit5'] for r in r_seq) / len(r_seq)
    at5_amb = sum(r['hit5'] for r in r_amb) / len(r_amb)
    delta_str = f"+{at5_amb-at5_seq:+.3f}"

    print(f"  {name:<20s} {len(r_seq):>3d}   {med_seq:>6.1f}    {at5_seq:.3f}    "
          f"{med_amb:>6.1f}    {at5_amb:.3f}    {delta_str}")
    p2_results[name] = {'n': len(r_seq),
                        'seq': {'med_rank': med_seq, 'at5': at5_seq},
                        'amb': {'med_rank': med_amb, 'at5': at5_amb}}

all_seq_at5 = np.mean([v['seq']['at5'] for v in p2_results.values()])
all_amb_at5 = np.mean([v['amb']['at5'] for v in p2_results.values()])
print(f"\n  OVERALL: Seq@5={all_seq_at5:.3f}  Amb@5={all_amb_at5:.3f}  "
      f"Delta={all_amb_at5-all_seq_at5:+.3f}")


# ── P3: Multi-Operator Ambient Composition ────────────────────────────────────
print(f"\n{'='*70}")
print(f"P3 — Multi-Operator Ambient Composition (3+ operators)")
print(f"  Can we compose 3 transformations simultaneously and find the right word?")
print(f"{'='*70}")

TRIPLE_TESTS = [
    # (source, ops, ops_desc, candidates)
    (' king',  [Δg, Δp, Δc],   'Δg+Δp+Δc',  ['queens', 'bigger', 'older', 'actresses']),
    (' cat',   [Δp, Δa, Δc],   'Δp+Δa+Δc',  ['cats', 'quickly', 'bigger']),
    (' big',   [Δc, Δa],       'Δc+Δa',      ['bigger', 'quickly', 'slowly', 'clearly']),
    (' walk',  [Δt, Δp],       'Δt+Δp',      ['walked', 'walks', 'walking']),
    (' man',   [Δg, Δp],       'Δg+Δp',      ['women', 'girls', 'ladies']),
    (' small', [Δc, Δa, Δg],   'Δc+Δa+Δg',  ['smaller', 'softly', 'quickly']),
]

print(f"\n  {'Source':<8}  {'Ops':<15}  Top-5 results (ambient)                          Rank of best candidate")
print(f"  {'-'*85}")

p3_results = []
for source, ops, ops_desc, candidates in TRIPLE_TESTS:
    p_src = phi(source)
    if p_src is None: continue

    phi_amb = apply_ambient(p_src, ops)
    top5    = retrieve(phi_amb, exclude=[source], top_k=5)
    top5_str = ', '.join(f"{w}({c:.3f})" for w, c in top5[:3])

    best_rank = min(find_rank(phi_amb, c.strip(), exclude=[source])
                    for c in candidates)
    best_cand = min(candidates, key=lambda c:
                    find_rank(phi_amb, c.strip(), exclude=[source]))

    hit = '✓' if best_rank <= 5 else '✗'
    print(f"  {source.strip():<8}  {ops_desc:<15}  {top5_str:<45}  {hit} rank={best_rank} ({best_cand})")
    p3_results.append({'source': source.strip(), 'ops': ops_desc,
                       'top1': top5[0][0] if top5 else '', 'best_rank': best_rank,
                       'best_cand': best_cand})

p3_hit_rate = sum(1 for r in p3_results if r['best_rank'] <= 5) / len(p3_results)
print(f"\n  Multi-operator hit@5 rate: {p3_hit_rate:.3f} ({sum(1 for r in p3_results if r['best_rank']<=5)}/{len(p3_results)})")


# ── P4: Round-Trip Accuracy ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P4 — Round-Trip: φ + Δ − Δ")
print(f"  Ambient:    φ + Δg − Δg = φ exactly")
print(f"  Sequential: normalise(normalise(φ + Δg) − Δg) = ?")
print(f"  Extended:   φ + Δg + Δp − Δp − Δg = φ (ambient, 2-step round trip)")
print(f"{'='*70}")

ROUNDTRIP_WORDS = [' king', ' man', ' cat', ' dog', ' big', ' fast',
                   ' walk', ' tree', ' actor', ' book']

print(f"\n  Single-op round trip (Δg):")
print(f"  {'Word':<10}  cos_seq   angle_seq°  cos_amb   angle_amb°")
print(f"  {'-'*55}")
p4_results = []
for w in ROUNDTRIP_WORDS:
    p_w = phi(w)
    if p_w is None: continue

    # Sequential
    p_enc_s = apply_sequential(p_w, [Δg])
    p_dec_s = p_enc_s - Δg
    nm = np.linalg.norm(p_dec_s)
    p_dec_s = p_dec_s / (nm + 1e-20)
    cos_s   = cosine(p_dec_s, p_w)
    ang_s   = float(np.degrees(np.arccos(np.clip(cos_s, -1, 1))))

    # Ambient
    p_enc_a = p_w + Δg
    p_dec_a = p_enc_a - Δg
    nm = np.linalg.norm(p_dec_a)
    p_dec_a = p_dec_a / (nm + 1e-20)
    cos_a   = cosine(p_dec_a, p_w)
    ang_a   = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    print(f"  {w.strip():<10}  {cos_s:.6f}  {ang_s:>10.4f}°  {cos_a:.6f}  {ang_a:>10.6f}°")
    p4_results.append({'word': w.strip(), 'cos_seq': cos_s, 'ang_seq': ang_s,
                       'cos_amb': cos_a, 'ang_amb': ang_a})

mean_ang_s = float(np.mean([r['ang_seq'] for r in p4_results]))
mean_ang_a = float(np.mean([r['ang_amb'] for r in p4_results]))
print(f"\n  Mean residual angle: sequential={mean_ang_s:.4f}°   ambient={mean_ang_a:.6f}°")

# Two-op round trip: φ + Δg + Δp - Δp - Δg
print(f"\n  Two-op round trip (Δg + Δp then reverse):")
print(f"  {'Word':<10}  cos_seq   cos_amb")
print(f"  {'-'*35}")
p4b_results = []
for w in ROUNDTRIP_WORDS[:6]:
    p_w = phi(w)
    if p_w is None: continue

    # Sequential
    p_enc_s = apply_sequential(p_w, [Δg, Δp])
    p_dec_s = apply_sequential(p_enc_s, [-Δp, -Δg])
    cos_s   = cosine(p_dec_s, p_w)

    # Ambient
    p_amb = p_w + Δg + Δp - Δp - Δg   # = p_w exactly
    nm = np.linalg.norm(p_amb); p_amb /= nm + 1e-20
    cos_a   = cosine(p_amb, p_w)

    print(f"  {w.strip():<10}  {cos_s:.6f}  {cos_a:.6f}")
    p4b_results.append({'word': w.strip(), 'cos_seq': cos_s, 'cos_amb': cos_a})


# ── P5: Radial Weight in Retrieval ────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P5 — Radial Weight: Does ||h|| improve retrieval?")
print(f"  score = (1 - α) * cos(φ, query) + α * normalised_||h||")
print(f"  α = 0: pure cosine (current)   α > 0: includes typicality bias")
print(f"{'='*70}")

# Build a set of T2 test cases with known correct answers
TEST_PAIRS = []
for name, seeds in T2_SEEDS.items():
    delta = build_t2(seeds)
    if delta is None: continue
    for a_w, b_w in seeds[:4]:
        p_src = phi(a_w)
        if p_src is None: continue
        p_out = apply_ambient(p_src, [delta])
        TEST_PAIRS.append({'source': a_w.strip(), 'target': b_w.strip(),
                           'query_phi': p_out, 'type': name})

print(f"\n  α         mean_rank   @rank1   @rank5")
print(f"  {'-'*42}")
p5_results = []
for alpha in [0.0, 0.05, 0.10, 0.20, 0.30]:
    ranks = []
    for tp in TEST_PAIRS:
        r = find_rank(tp['query_phi'], tp['target'])
        if alpha > 0:
            # Use radial-weighted scoring
            sims = phi_all @ tp['query_phi']
            norms_norm = (raw_norms - raw_norms.min()) / (raw_norms.max() - raw_norms.min() + 1e-20)
            scores = (1.0 - alpha) * sims + alpha * norms_norm
            target_k = tp['target'] if tp['target'] in w2i else ' '+tp['target']
            if target_k in w2i:
                order = np.argsort(-scores)
                pos = np.where(order == w2i[target_k])[0]
                r = int(pos[0]) + 1 if len(pos) > 0 else 9999
        ranks.append(r)
    mean_r = float(np.mean(ranks))
    at1 = sum(1 for r in ranks if r <= 1) / len(ranks)
    at5 = sum(1 for r in ranks if r <= 5) / len(ranks)
    marker = ' ← baseline' if alpha == 0.0 else ''
    print(f"  α={alpha:.2f}    {mean_r:>8.2f}    {at1:.3f}    {at5:.3f}{marker}")
    p5_results.append({'alpha': alpha, 'mean_rank': mean_r, 'at1': at1, 'at5': at5})


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY — Day 55")
print(f"{'='*70}")
print(f"""
  P1  Day 11 retest:
      Ambient better / Sequential better / Tied:
      {amb_wins}/{len(p1_results)} / {seq_wins}/{len(p1_results)} / {ties}/{len(p1_results)}

  P2  LCM inference overall accuracy:
      Sequential @5: {all_seq_at5:.3f}
      Ambient    @5: {all_amb_at5:.3f}
      Delta:         {all_amb_at5-all_seq_at5:+.3f}

  P3  Multi-operator (3+ ops) hit@5: {p3_hit_rate:.3f}

  P4  Round-trip residual angle:
      Sequential: {mean_ang_s:.4f}°
      Ambient:    {mean_ang_a:.6f}°

  P5  Best alpha for radial weighting:
      α=0.00 (baseline) @5={p5_results[0]['at5']:.3f}
      Best α:  {max(p5_results, key=lambda x: x['at5'])['alpha']:.2f}
               @5={max(p5_results, key=lambda x: x['at5'])['at5']:.3f}
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
    'p1_simultaneous': p1_results,
    'p2_lcm_inference': p2_results,
    'p3_multi_op': p3_results,
    'p4_roundtrip': p4_results,
    'p5_radial_weight': p5_results,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 55 complete.")
