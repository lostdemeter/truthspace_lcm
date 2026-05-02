#!/usr/bin/env python3
"""
Expedition Day 42 — The 43 Concept Axes: What Do They Mean?

Context (Day 36 Q3):
  - Body-centroid SVD yields eff_rank ≈ 42.8
  - Axis 1 (56.6% variance) is NOT a semantic axis — captures the common
    hemisphere shared by all Zone C concepts
  - 22:1 spectral gap after Axis 1
  - We need to know what axes 1–43 actually encode

Plan:
  S1. Recompute the body-centroid SVD (same as Day 36 Q3)
  S2. For each axis 1..N_AXES, project ALL wmap words onto it
      → top-20 / bottom-20 words at each pole
  S3. Align each axis with known T2 operators (comp→sup, plural, gender,
      adverb, base→comparative) — if cos > 0.5, label as that operator
  S4. Classify each axis:
      - TYPE_DOMAIN  : one or two specific bodies dominate both poles
      - TYPE_MORPHO  : known T2 operator aligns with it
      - TYPE_COMMON  : Axis 1 variant (all concepts same sign → hemisphere)
      - TYPE_UNKNOWN : no clear pattern
  S5. Save full axis map; print human-readable summary

Fail-fast philosophy: no label guessing. If the poles are ambiguous,
mark as UNKNOWN and record the pole words — the data IS the label.
"""

import os, json
import numpy as np
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day42_axis_identification.json")

N_AXES     = 50     # inspect first 50 axes (covers the eff_rank=43 range)
N_POLE     = 20     # words shown at each pole per axis
T2_COS_THR = 0.45   # threshold for calling an axis a T2 match

# ── Killing pairs for Z2 ──────────────────────────────────────────────────────
KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# ── Known T2 operator seed pairs ─────────────────────────────────────────────
# Format: (label, [(source, target), ...])
T2_SEEDS = [
    ('comp→sup',   [('bigger','biggest'), ('faster','fastest'), ('older','oldest'),
                    ('smaller','smallest'), ('stronger','strongest')]),
    ('base→comp',  [('big','bigger'), ('fast','faster'), ('old','older'),
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


# ── Helpers ───────────────────────────────────────────────────────────────────
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


# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Expedition Day 42 — The 43 Concept Axes")
print("=" * 70)

print("\n── Load ────────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}
print(f"  {len(words_all):,} words, hs dim = {hs14_all.shape[1]}")

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# Zone and body assignments
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
print(f"  Zone C: {len(zone_c_words)}  wmap total: {len(wmap_words)}")

# ── Z2 axis ───────────────────────────────────────────────────────────────────
print("\n── Z2 axis ─────────────────────────────────────────────────────────")
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
_, sv_d, Vt_d = np.linalg.svd(D, full_matrices=False)
z2 = Vt_d[0] / np.linalg.norm(Vt_d[0])
print(f"  Z2 from {len(deltas)} Killing pairs, SV0={sv_d[0]:.3f}")

# ── φ-vectors ─────────────────────────────────────────────────────────────────
print("\n── φ-vectors ────────────────────────────────────────────────────────")
zone_c_idx = np.array([w2i[w] for w in zone_c_words])
phi_c14    = batch_phi(hs14_all[zone_c_idx], z2)

wmap_idx   = np.array([w2i[w] for w in wmap_words])
wmap_phi   = batch_phi(hs14_all[wmap_idx], z2)
w2l        = {w: i for i, w in enumerate(wmap_words)}

# body centroids
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

print(f"  {len(body_centroids)} body centroids computed")

# ── SVD of body-centroid matrix ───────────────────────────────────────────────
print("\n── S1: Body-centroid SVD ────────────────────────────────────────────")
bodies_list = list(body_centroids.keys())
C = np.stack([body_centroids[b] for b in bodies_list])     # (n_bodies, D)
_, sv_c, Vt_c = np.linalg.svd(C, full_matrices=False)

total_var = float(np.sum(sv_c**2))
cumvar    = np.cumsum(sv_c**2) / total_var
eff_rank  = float(np.sum(sv_c))**2 / float(np.sum(sv_c**2))
k95       = int(np.searchsorted(cumvar, 0.95)) + 1

print(f"  {len(sv_c)} singular values over {len(bodies_list)} bodies")
print(f"  Effective rank: {eff_rank:.1f}   95% variance: {k95} dims")
print(f"  SV spectrum (first 15): {' '.join(f'{s:.2f}' for s in sv_c[:15])}")
print(f"  Axis 1 variance: {sv_c[0]**2/total_var*100:.1f}%  "
      f"Axis 2: {sv_c[1]**2/total_var*100:.1f}%  "
      f"Axis 3: {sv_c[2]**2/total_var*100:.1f}%")

# ── S2: T2 operator vectors ───────────────────────────────────────────────────
print("\n── S2: Known T2 operator vectors ────────────────────────────────────")
t2_vectors = {}
for t2_label, pairs in T2_SEEDS:
    vecs = []
    missing = []
    for src, tgt in pairs:
        for pfx in [' ', '']:
            ws, wt = pfx+src, pfx+tgt
            if ws in w2l and wt in w2l:
                v = wmap_phi[w2l[wt]] - wmap_phi[w2l[ws]]
                nm = np.linalg.norm(v)
                if nm > 1e-20:
                    vecs.append(v / nm)
                break
        else:
            missing.append(src)
    if vecs:
        t2_vectors[t2_label] = mean_unit_vec(np.stack(vecs))
        print(f"  {t2_label:<20s}  {len(vecs)} pairs  (missing: {missing[:3]})")
    else:
        print(f"  {t2_label:<20s}  NO pairs found — skip")

# ── S3: Axis characterisation ────────────────────────────────────────────────
print("\n── S3: Axis-by-axis characterisation ────────────────────────────────")
print(f"  {'Ax':>3s}  {'Var%':>5s}  {'Type':<10s}  {'T2 match':<22s}  "
      f"{'+ pole (bodies)': <35s}  {'- pole (bodies)'}")
print(f"  {'-'*130}")

axis_records = []

for ax in range(min(N_AXES, len(sv_c))):
    ax_vec    = Vt_c[ax]           # direction in φ-space
    var_pct   = sv_c[ax]**2 / total_var * 100

    # Project ALL wmap words onto this axis
    projs     = wmap_phi @ ax_vec  # (n_words,)

    top_idx   = np.argsort(projs)[-N_POLE:][::-1]
    bot_idx   = np.argsort(projs)[:N_POLE]

    top_words = [(wmap_words[i], float(projs[i]),
                  wmap.get(wmap_words[i], {}).get('L14_body', '?'))
                 for i in top_idx]
    bot_words = [(wmap_words[i], float(projs[i]),
                  wmap.get(wmap_words[i], {}).get('L14_body', '?'))
                 for i in bot_idx]

    # Body distribution at each pole
    top_bodies = [w[2] for w in top_words if w[2] not in ('?', None, 'B000', 'B001')]
    bot_bodies = [w[2] for w in bot_words if w[2] not in ('?', None, 'B000', 'B001')]

    def body_summary(bodies):
        counts = defaultdict(int)
        for b in bodies:
            counts[b] += 1
        top2 = sorted(counts.items(), key=lambda x: -x[1])[:2]
        return ' | '.join(f"{body_label_map.get(b,'?')[:14]}({n})" for b, n in top2) if top2 else '—'

    top_body_str = body_summary(top_bodies)
    bot_body_str = body_summary(bot_bodies)

    # T2 alignment
    best_t2_label = '—'
    best_t2_cos   = 0.0
    for t2_lbl, t2_vec in t2_vectors.items():
        c = float(ax_vec @ t2_vec)
        if abs(c) > abs(best_t2_cos):
            best_t2_cos   = c
            best_t2_label = t2_lbl if c > 0 else f'←{t2_lbl}'

    # Pole word summary (strip leading space for display)
    top_wstr = ' '.join(w[0].strip() for w in top_words[:6])
    bot_wstr = ' '.join(w[0].strip() for w in bot_words[:6])

    # Classification
    all_body_projs = C @ ax_vec  # project all 95 body centroids
    all_same_sign  = np.all(all_body_projs > 0) or np.all(all_body_projs < 0)
    frac_pos = float(np.sum(all_body_projs > 0)) / len(all_body_projs)

    if frac_pos > 0.90 or frac_pos < 0.10:
        axis_type = 'COMMON'
    elif abs(best_t2_cos) >= T2_COS_THR:
        axis_type = 'MORPHO'
    elif top_bodies or bot_bodies:
        axis_type = 'DOMAIN'
    else:
        axis_type = 'UNKNOWN'

    t2_str = f"{best_t2_label[:12]} ({best_t2_cos:+.2f})" if best_t2_label != '—' else '—'

    # Print condensed one-liner
    print(f"  {ax+1:>3d}  {var_pct:>5.1f}%  {axis_type:<10s}  {t2_str:<22s}  "
          f"{top_body_str:<35s}  {bot_body_str}")

    # Record
    axis_records.append({
        'axis': ax + 1,
        'var_pct': round(var_pct, 2),
        'type': axis_type,
        'frac_positive_bodies': round(frac_pos, 3),
        'best_t2_label': best_t2_label,
        'best_t2_cos': round(best_t2_cos, 3),
        'top_pole': [{'word': w.strip(), 'proj': round(p, 4), 'body': b}
                     for w, p, b in top_words],
        'bot_pole': [{'word': w.strip(), 'proj': round(p, 4), 'body': b}
                     for w, p, b in bot_words],
        'top_body_summary': top_body_str,
        'bot_body_summary': bot_body_str,
    })


# ── S4: Summary statistics ────────────────────────────────────────────────────
print(f"\n── S4: Summary ──────────────────────────────────────────────────────")
type_counts = defaultdict(int)
for r in axis_records:
    type_counts[r['type']] += 1
for t, n in sorted(type_counts.items()):
    print(f"  {t:<12s}: {n}")

# Print pole words for DOMAIN and MORPHO axes (the interpretable ones)
print(f"\n── Interpretable axes (DOMAIN + MORPHO) — pole words ────────────────")
for r in axis_records:
    if r['type'] not in ('DOMAIN', 'MORPHO'):
        continue
    top5 = ' | '.join(w['word'] for w in r['top_pole'][:8])
    bot5 = ' | '.join(w['word'] for w in r['bot_pole'][:8])
    t2   = f"  [{r['best_t2_label']} {r['best_t2_cos']:+.2f}]" if abs(r['best_t2_cos']) >= T2_COS_THR else ''
    print(f"\n  Axis {r['axis']:>2d} ({r['var_pct']:.1f}%){t2}")
    print(f"    + {top5}")
    print(f"    - {bot5}")

# ── S5: Save ──────────────────────────────────────────────────────────────────
output = {
    'meta': {
        'experiment': 'Day 42 — The 43 Concept Axes',
        'n_bodies': len(bodies_list),
        'eff_rank': round(eff_rank, 1),
        'dims_for_95pct': k95,
        'sv_spectrum': [round(float(s), 4) for s in sv_c[:50]],
    },
    't2_vectors_available': list(t2_vectors.keys()),
    'axes': axis_records,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 42 complete.")
