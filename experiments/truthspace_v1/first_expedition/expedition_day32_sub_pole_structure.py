#!/usr/bin/env python3
"""
Expedition Day 32 — Sub-Pole Structure and Self-Similarity

Open Question 3: Is the degenerate pole a single point or a small sphere?
  Zone A (monosyllabic), Zone B (secondary pole), and Zone E (proper nouns)
  all land within cos=0.9982 of each other. Are they coincident or separated?

Additional question (user): does the pole itself exhibit self-similar structure?
  Global φ-space: frequency drives degeneration (common → pole, rare → periphery)
  Self-similarity prediction: WITHIN the pole, frequency ALSO drives sub-structure
    - Most common words (the, and) → sub-pole (furthest from any semantic axis)
    - Less common Zone A words (dog, cat) → sub-periphery
    - Zone B words (return, public) → sub-periphery of pole
    - Zone E proper nouns → offset cluster

The definitive test: local SVD on pole hidden states.
  If the top axis of variance within the pole = global Z2 → full self-similarity.
  If it correlates with log(token_id) → same organising principle at smaller scale.

Data: pure matrix ops on cached hidden states. No forward passes.
"""

import os, json, time
import numpy as np
from scipy.stats import spearmanr

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
PN_CACHE    = os.path.join(os.path.dirname(__file__), "day29_pn_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day32_sub_pole.json")

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]


def batch_phi(hs_matrix, z2):
    H  = hs_matrix.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)


def centroid(vecs):
    c  = vecs.mean(axis=0)
    nm = np.linalg.norm(c)
    return c / (nm + 1e-20)


def phi_cos_to_centroid(phi_vecs, c):
    return (phi_vecs @ c).tolist()


def zone_stats(label, phi_vecs, token_ids=None):
    c    = centroid(phi_vecs)
    sims = phi_vecs @ c
    print(f"\n  {label} (n={len(phi_vecs)}):")
    print(f"    phi_cos to centroid: "
          f"min={sims.min():.5f}  median={np.median(sims):.5f}  max={sims.max():.5f}  "
          f"std={sims.std():.5f}")
    if token_ids is not None:
        r, p = spearmanr(token_ids, sims)
        print(f"    Spearman r(token_id, phi_cos): {r:+.4f}  (p={p:.3e})")
    return c, sims.tolist()


t0 = time.time()

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print(f"\n── Step 1: Load data ────────────────────────────────────────────")
npz   = np.load(CACHE_FILE, allow_pickle=True)
words_all  = list(npz['words'])
hs14_all   = npz['hs_14'].astype(np.float64)
hs23_all   = npz['hs_23'].astype(np.float64)
w2i        = {w: i for i, w in enumerate(words_all)}

pn         = np.load(PN_CACHE, allow_pickle=True)
pn_words   = list(pn['words'])
pn_hs14    = pn['hs'].astype(np.float64)   # L14 hidden states for 301 proper nouns
print(f"  Dict cache: {len(words_all)} words")
print(f"  PN cache:   {len(pn_words)} proper nouns")

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# Zone A: Phase 1 words (monosyllabic)
zone_a_entries = [(w, v) for w, v in wmap.items() if v['phase'] == 1 and w in w2i]
zone_a_words   = [w for w, _ in zone_a_entries]
zone_a_tids    = np.array([v['token_id'] for _, v in zone_a_entries])
zone_a_idx     = np.array([w2i[w] for w in zone_a_words])

# Zone B: Phase 2 words at secondary pole (L14_body == 'B001')
zone_b_entries = [(w, v) for w, v in wmap.items()
                  if v['phase'] == 2 and v.get('L14_body') == 'B001' and w in w2i]
zone_b_words   = [w for w, _ in zone_b_entries]
zone_b_tids    = np.array([v['token_id'] for _, v in zone_b_entries])
zone_b_cos14   = np.array([v['L14_phi_cos'] for _, v in zone_b_entries])
zone_b_cos23   = np.array([v['L23_phi_cos'] for _, v in zone_b_entries])
zone_b_idx     = np.array([w2i[w] for w in zone_b_words])

# Zone E: 301 proper nouns (isolated, from Day 29 PN cache)
zone_e_words   = pn_words

print(f"  Zone A: {len(zone_a_words)} words  (monosyllabic, Phase 1)")
print(f"  Zone B: {len(zone_b_words)} words  (Phase 2, secondary pole B001)")
print(f"  Zone E: {len(zone_e_words)} proper nouns  (Day 29 PN cache)")


# ── Step 2: Z2 axes ───────────────────────────────────────────────────────────
print(f"\n── Step 2: Z2 axes ──────────────────────────────────────────────")

def z2_from_cache(hs_mat, pairs, name):
    deltas = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx + a, pfx + b
            if wa in w2i and wb in w2i:
                d  = hs_mat[w2i[wb]] - hs_mat[w2i[wa]]
                dm = np.linalg.norm(d)
                if dm > 1e-20:
                    deltas.append(d / dm)
                break
    D = np.stack(deltas)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    z2  = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-20)
    pct = 100 * sv[0]**2 / (np.sum(sv**2) + 1e-20)
    print(f"  Z2 {name}: {pct:.1f}%  ({len(deltas)} deltas)")
    return z2

z2_14 = z2_from_cache(hs14_all, KILLING_PAIRS, "L14")
z2_23 = z2_from_cache(hs23_all, KILLING_PAIRS, "L23")


# ── Step 3: φ-vectors for all three zones at L14 and L23 ─────────────────────
print(f"\n── Step 3: φ-vectors ────────────────────────────────────────────")

# Zone A
phi_a14 = batch_phi(hs14_all[zone_a_idx], z2_14)
phi_a23 = batch_phi(hs23_all[zone_a_idx], z2_23)

# Zone B
phi_b14 = batch_phi(hs14_all[zone_b_idx], z2_14)
phi_b23 = batch_phi(hs23_all[zone_b_idx], z2_23)

# Zone E (PN cache has only L14)
phi_e14 = batch_phi(pn_hs14, z2_14)

print(f"  Zone A phi: L14={phi_a14.shape}  L23={phi_a23.shape}")
print(f"  Zone B phi: L14={phi_b14.shape}  L23={phi_b23.shape}")
print(f"  Zone E phi: L14={phi_e14.shape}")


# ── Section 1: Zone centroids and inter-zone cosines (sphere vs point) ────────
print(f"\n{'='*65}")
print(f"SECTION 1 — Sphere vs Point: Inter-Zone Cosines")
print(f"{'='*65}")

c_a14, sims_a14 = zone_stats("Zone A L14", phi_a14, zone_a_tids)
c_b14, sims_b14 = zone_stats("Zone B L14", phi_b14, zone_b_tids)
c_e14, sims_e14 = zone_stats("Zone E L14", phi_e14)

print(f"\n  Inter-zone centroid cosines (L14):")
cos_ab = float(c_a14 @ c_b14)
cos_ae = float(c_a14 @ c_e14)
cos_be = float(c_b14 @ c_e14)
print(f"    cos(Zone A, Zone B): {cos_ab:.6f}  Δ from 1.0 = {1-cos_ab:.6f}")
print(f"    cos(Zone A, Zone E): {cos_ae:.6f}  Δ from 1.0 = {1-cos_ae:.6f}")
print(f"    cos(Zone B, Zone E): {cos_be:.6f}  Δ from 1.0 = {1-cos_be:.6f}")

# Global pole centroid (Phase 1 only for comparison)
c_pole14 = centroid(phi_a14)
cos_b_pole = float(c_b14 @ c_pole14)
cos_e_pole = float(c_e14 @ c_pole14)
print(f"\n  Distances from Zone A centroid (= pole centre):")
print(f"    cos(Zone B, pole): {cos_b_pole:.6f}  Δ = {1-cos_b_pole:.6f}")
print(f"    cos(Zone E, pole): {cos_e_pole:.6f}  Δ = {1-cos_e_pole:.6f}")

# L23 comparison for A and B (E not available at L23)
c_a23, sims_a23 = zone_stats("Zone A L23", phi_a23, zone_a_tids)
c_b23, sims_b23 = zone_stats("Zone B L23", phi_b23, zone_b_tids)
cos_ab23 = float(c_a23 @ c_b23)
print(f"\n  Inter-zone centroid cosines (L23):")
print(f"    cos(Zone A, Zone B): {cos_ab23:.6f}  Δ from 1.0 = {1-cos_ab23:.6f}")


# ── Section 2: Within-Zone A frequency stratification ─────────────────────────
print(f"\n{'='*65}")
print(f"SECTION 2 — Within-Zone A: Frequency Stratification (Self-Similarity)")
print(f"{'='*65}")

# Sort by token_id and split into quartiles
order      = np.argsort(zone_a_tids)
n_a        = len(zone_a_tids)
quartile_n = n_a // 4
quartiles  = [order[i*quartile_n:(i+1)*quartile_n] for i in range(4)]
q_labels   = ["Q1 (most common)", "Q2", "Q3", "Q4 (rarest)"]

# For each quartile, compute phi_cos to Zone A centroid
c_a14 = centroid(phi_a14)
print(f"\n  Zone A L14 — φ_cos to centroid by token_id quartile:")
q_data = []
for qi, (qidx, ql) in enumerate(zip(quartiles, q_labels)):
    phi_q   = phi_a14[qidx]
    sims_q  = phi_q @ c_a14
    tids_q  = zone_a_tids[qidx]
    words_q = [zone_a_words[j] for j in qidx[:5]]
    print(f"  {ql}: mean_phi_cos={sims_q.mean():.5f}  "
          f"tid_range=[{tids_q.min()},{tids_q.max()}]  "
          f"sample={words_q}")
    q_data.append({'quartile': qi+1, 'label': ql,
                   'mean_phi_cos': float(sims_q.mean()),
                   'tid_min': int(tids_q.min()), 'tid_max': int(tids_q.max()),
                   'sample': words_q})

# Full Spearman correlation for Zone A
r_a14, p_a14 = spearmanr(zone_a_tids, sims_a14)
print(f"\n  Spearman r(token_id, phi_cos to centroid) for Zone A L14: "
      f"{r_a14:+.4f}  (p={p_a14:.3e})")

# Within Zone B: is phi_cos correlated with token_id?
c_b14_v = centroid(phi_b14)
sims_b14_c = phi_b14 @ c_b14_v
r_b14, p_b14 = spearmanr(zone_b_tids, sims_b14_c)
print(f"  Spearman r(token_id, phi_cos to centroid) for Zone B L14: "
      f"{r_b14:+.4f}  (p={p_b14:.3e})")

# Zone B broken into quartiles
order_b    = np.argsort(zone_b_tids)
q_b        = [order_b[i*len(order_b)//4:(i+1)*len(order_b)//4] for i in range(4)]
print(f"\n  Zone B L14 — φ_cos to centroid by token_id quartile:")
for qi, (qidx, ql) in enumerate(zip(q_b, q_labels)):
    phi_q   = phi_b14[qidx]
    sims_q  = phi_q @ c_b14_v
    words_q = [zone_b_words[j] for j in qidx[:5]]
    print(f"  {ql}: mean_phi_cos={sims_q.mean():.5f}  "
          f"sample={words_q}")

# Zone A + B combined: does a single frequency axis explain both?
all_phi14  = np.vstack([phi_a14, phi_b14])
all_tids   = np.concatenate([zone_a_tids, zone_b_tids])
c_all14    = centroid(all_phi14)
sims_all14 = all_phi14 @ c_all14
r_all14, p_all14 = spearmanr(all_tids, sims_all14)
print(f"\n  Spearman r(token_id, phi_cos) for Zone A+B combined L14: "
      f"{r_all14:+.4f}  (p={p_all14:.3e})")


# ── Section 3: Local SVD on pole hidden states (definitive self-similarity test)
print(f"\n{'='*65}")
print(f"SECTION 3 — Local SVD: Does the Pole Have Its Own Z2?")
print(f"{'='*65}")
print(f"\n  Running SVD on raw L14 hidden states of Zone A+B ({len(all_phi14)} words)...")

# Stack raw hidden states (not phi-vectors)
all_hs14 = np.vstack([hs14_all[zone_a_idx], hs14_all[zone_b_idx]])

# Centre within the pole
mean_hs   = all_hs14.mean(axis=0)
centred   = all_hs14 - mean_hs

# SVD — top components
U, sv, Vt = np.linalg.svd(centred, full_matrices=False)
variance_explained = sv**2 / (np.sum(sv**2) + 1e-20)

print(f"  Top-5 singular values (variance %):")
for i in range(5):
    print(f"    PC{i+1}: {100*variance_explained[i]:.2f}%")

# Top axis of local variation within the pole
local_axis1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-20)
local_axis2 = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-20)

# Cosine with global Z2
cos_local1_z2 = abs(float(local_axis1 @ z2_14))
cos_local2_z2 = abs(float(local_axis2 @ z2_14))
print(f"\n  cos(local_axis1, global_Z2_L14): {cos_local1_z2:.4f}")
print(f"  cos(local_axis2, global_Z2_L14): {cos_local2_z2:.4f}")
print(f"  (1.0 = same axis; 0.0 = orthogonal; self-similar → cos→1.0)")

# Project pole words onto local axes
proj1_ab = centred @ local_axis1
proj2_ab = centred @ local_axis2

# Correlation of projection with token_id
r1_tid, p1_tid = spearmanr(all_tids, proj1_ab)
r2_tid, p2_tid = spearmanr(all_tids, proj2_ab)
print(f"\n  Spearman r(token_id, projection on local_axis1): "
      f"{r1_tid:+.4f}  (p={p1_tid:.3e})")
print(f"  Spearman r(token_id, projection on local_axis2): "
      f"{r2_tid:+.4f}  (p={p2_tid:.3e})")

# Which PC most correlates with token_id?
print(f"\n  Token_id correlation for top 10 PCs:")
best_pc, best_r = 0, 0.0
for i in range(min(10, len(sv))):
    proj_i = centred @ (Vt[i] / np.linalg.norm(Vt[i]))
    ri, pi  = spearmanr(all_tids, proj_i)
    star   = " ← most correlated" if abs(ri) > abs(best_r) else ""
    if abs(ri) > abs(best_r):
        best_r, best_pc = ri, i+1
    print(f"    PC{i+1}: r={ri:+.4f} (p={pi:.2e})  var={100*variance_explained[i]:.2f}%{star}")

# Check if Zone A and Zone B are linearly separable on the local SVD
proj1_a  = (hs14_all[zone_a_idx] - mean_hs) @ local_axis1
proj1_b  = (hs14_all[zone_b_idx] - mean_hs) @ local_axis1
print(f"\n  Zone A vs Zone B on local_axis1:")
print(f"    Zone A: mean={proj1_a.mean():.4f}  std={proj1_a.std():.4f}")
print(f"    Zone B: mean={proj1_b.mean():.4f}  std={proj1_b.std():.4f}")
print(f"    Zone E (PN): mean={(((pn_hs14 - mean_hs) @ local_axis1)).mean():.4f}  "
      f"std={(((pn_hs14 - mean_hs) @ local_axis1)).std():.4f}")
sep_z = (proj1_a.mean() - proj1_b.mean()) / (
    np.sqrt(0.5 * (proj1_a.std()**2 + proj1_b.std()**2)) + 1e-20)
print(f"    Zone A - Zone B separation (Cohen's d): {sep_z:.3f}")


# ── Section 4: Fine structure within Zone A ────────────────────────────────────
print(f"\n{'='*65}")
print(f"SECTION 4 — Fine Structure Within Zone A (Most Common Words)")
print(f"{'='*65}")

# Top 50 most common Zone A words vs bottom 50
top50_idx = np.argsort(zone_a_tids)[:50]
bot50_idx = np.argsort(zone_a_tids)[-50:]

phi_top50 = phi_a14[top50_idx]
phi_bot50 = phi_a14[bot50_idx]

c_top50 = centroid(phi_top50)
c_bot50 = centroid(phi_bot50)
cos_top_bot = float(c_top50 @ c_bot50)

print(f"\n  Top-50 most common Zone A (the, and, for, this, that...):")
print(f"    token_id range: {zone_a_tids[top50_idx].min()}–{zone_a_tids[top50_idx].max()}")
print(f"    words: {[zone_a_words[j] for j in top50_idx[:12]]}")
print(f"    phi_cos to centroid: mean={float(np.mean(phi_top50 @ centroid(phi_top50))):.5f}")

print(f"\n  Bottom-50 rarest Zone A (monosyllabic but uncommon):")
print(f"    token_id range: {zone_a_tids[bot50_idx].min()}–{zone_a_tids[bot50_idx].max()}")
print(f"    words: {[zone_a_words[j] for j in bot50_idx[:12]]}")
print(f"    phi_cos to centroid: mean={float(np.mean(phi_bot50 @ centroid(phi_bot50))):.5f}")

print(f"\n  cos(top-50 centroid, bottom-50 centroid): {cos_top_bot:.6f}")
print(f"  Δ from unity: {1 - cos_top_bot:.6f}")
print(f"  → top-50 centroid cos to Zone A centroid: "
      f"{float(c_top50 @ c_a14):.6f}")
print(f"  → bottom-50 centroid cos to Zone A centroid: "
      f"{float(c_bot50 @ c_a14):.6f}")


# ── Section 5: L14 vs L23 pole concentration ──────────────────────────────────
print(f"\n{'='*65}")
print(f"SECTION 5 — Pole Concentration: L14 vs L23")
print(f"{'='*65}")

c_a23_v = centroid(phi_a23)
c_b23_v = centroid(phi_b23)

# Intra-zone spread
spread_a14 = (phi_a14 @ centroid(phi_a14)).std()
spread_a23 = (phi_a23 @ centroid(phi_a23)).std()
spread_b14 = (phi_b14 @ centroid(phi_b14)).std()
spread_b23 = (phi_b23 @ centroid(phi_b23)).std()

print(f"\n  Intra-zone angular spread (φ_cos std around centroid):")
print(f"    Zone A L14: {spread_a14:.5f}   L23: {spread_a23:.5f}   "
      f"{'CONCENTRATED↓' if spread_a23 < spread_a14 else 'DIFFUSED↑'}")
print(f"    Zone B L14: {spread_b14:.5f}   L23: {spread_b23:.5f}   "
      f"{'CONCENTRATED↓' if spread_b23 < spread_b14 else 'DIFFUSED↑'}")

# Mean phi_cos for each zone at L14 vs L23
mc_a14 = float(np.mean(phi_a14 @ centroid(phi_a14)))
mc_a23 = float(np.mean(phi_a23 @ centroid(phi_a23)))
mc_b14 = float(np.mean(phi_b14 @ centroid(phi_b14)))
mc_b23 = float(np.mean(phi_b23 @ centroid(phi_b23)))
print(f"\n  Mean φ_cos to own centroid (compactness):")
print(f"    Zone A: L14={mc_a14:.5f}  L23={mc_a23:.5f}  "
      f"Δ={mc_a23-mc_a14:+.5f}")
print(f"    Zone B: L14={mc_b14:.5f}  L23={mc_b23:.5f}  "
      f"Δ={mc_b23-mc_b14:+.5f}")

# Does Zone B separate from Zone A at L23?
cos_ab14_v = float(centroid(phi_a14) @ centroid(phi_b14))
cos_ab23_v = float(centroid(phi_a23) @ centroid(phi_b23))
print(f"\n  cos(Zone A centroid, Zone B centroid):")
print(f"    L14: {cos_ab14_v:.6f}   L23: {cos_ab23_v:.6f}")
print(f"    {'ZONES SEPARATING at L23 ↓' if cos_ab23_v < cos_ab14_v else 'ZONES CONVERGING at L23 ↑'}")


# ── Section 6: The self-similarity summary ────────────────────────────────────
print(f"\n{'='*65}")
print(f"SECTION 6 — Self-Similarity Assessment")
print(f"{'='*65}")

# Compute frequency correlation with distance from pole
# For Zone A+B together, is there a monotone relationship:
# low token_id → high phi_cos (at pole) → self-similar with global
sims_a14_from_za = phi_a14 @ c_a14
sims_b14_from_za = phi_b14 @ c_a14     # Zone B vs Zone A centroid
sims_e14_from_za = phi_e14 @ c_a14     # Zone E vs Zone A centroid

print(f"\n  φ_cos to Zone A centroid (= pole centre):")
print(f"    Zone A: {sims_a14_from_za.mean():.5f} ± {sims_a14_from_za.std():.5f}")
print(f"    Zone B: {sims_b14_from_za.mean():.5f} ± {sims_b14_from_za.std():.5f}")
print(f"    Zone E: {sims_e14_from_za.mean():.5f} ± {sims_e14_from_za.std():.5f}")

# Ordering: is Zone A closest to pole > Zone B > Zone E?
order_check = sims_a14_from_za.mean() > sims_b14_from_za.mean() > sims_e14_from_za.mean()
print(f"\n  Order: Zone A > Zone B > Zone E (pred by self-similarity): "
      f"{'YES ✓' if order_check else 'NO ✗'}")

# Combined frequency correlation
combined_phi_cos = np.concatenate([sims_a14_from_za, sims_b14_from_za])
combined_tids    = np.concatenate([zone_a_tids, zone_b_tids])
r_comb, p_comb   = spearmanr(combined_tids, combined_phi_cos)
print(f"  Spearman r(token_id, φ_cos to pole) — Zone A+B combined: "
      f"{r_comb:+.4f}  (p={p_comb:.2e})")
print(f"  → {'STRONG SELF-SIMILAR' if abs(r_comb)>0.3 else 'WEAK' if abs(r_comb)>0.1 else 'NO'} "
      f"frequency-driven sub-pole structure")

# Local SVD axis vs global Z2 summary
print(f"\n  Local SVD axis vs global Z2:")
print(f"    cos(local_axis1, Z2_L14) = {cos_local1_z2:.4f}")
print(f"    → {'SAME AXIS (full self-similarity)' if cos_local1_z2>0.8 else 'DIFFERENT AXIS (partial)' if cos_local1_z2>0.3 else 'ORTHOGONAL AXIS (different principle)'}")

print(f"\n  Token_id correlation on best PC:")
print(f"    PC{best_pc}: r={best_r:+.4f}")
print(f"    → {'STRONG' if abs(best_r)>0.3 else 'MODERATE' if abs(best_r)>0.15 else 'WEAK'} "
      f"frequency structure within the pole")


# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n── Saving results ───────────────────────────────────────────────")
result = {
    "meta": {
        "experiment": "Day 32 — Sub-pole structure and self-similarity",
        "zone_a_n": len(zone_a_words),
        "zone_b_n": len(zone_b_words),
        "zone_e_n": len(zone_e_words),
    },
    "sphere_vs_point": {
        "L14": {
            "cos_ZoneA_ZoneB": cos_ab,
            "cos_ZoneA_ZoneE": cos_ae,
            "cos_ZoneB_ZoneE": cos_be,
            "cos_ZoneB_pole":  cos_b_pole,
            "cos_ZoneE_pole":  cos_e_pole,
        },
        "L23": {
            "cos_ZoneA_ZoneB": cos_ab23,
        },
    },
    "zone_phi_cos": {
        "zone_a_L14_vs_pole": {
            "min": float(np.array(sims_a14).min()),
            "median": float(np.median(sims_a14)),
            "max": float(np.array(sims_a14).max()),
            "std": float(np.array(sims_a14).std()),
        },
        "zone_b_L14_vs_pole": {
            "min": float(sims_b14_from_za.min()),
            "median": float(np.median(sims_b14_from_za)),
            "max": float(sims_b14_from_za.max()),
            "std": float(sims_b14_from_za.std()),
        },
        "zone_e_L14_vs_pole": {
            "min": float(sims_e14_from_za.min()),
            "median": float(np.median(sims_e14_from_za)),
            "max": float(sims_e14_from_za.max()),
            "std": float(sims_e14_from_za.std()),
        },
    },
    "self_similarity": {
        "cos_local_axis1_global_z2": cos_local1_z2,
        "cos_local_axis2_global_z2": cos_local2_z2,
        "spearman_tokenid_proj1": r1_tid,
        "spearman_tokenid_proj2": r2_tid,
        "best_pc_tokenid_r": best_r,
        "best_pc_number": best_pc,
        "spearman_tokenid_phicos_zA": r_a14,
        "spearman_tokenid_phicos_zB": r_b14,
        "spearman_tokenid_phicos_combined": r_comb,
        "order_A_above_B_above_E": bool(order_check),
    },
    "concentration_L14_vs_L23": {
        "zone_a_spread_L14": float(spread_a14),
        "zone_a_spread_L23": float(spread_a23),
        "zone_b_spread_L14": float(spread_b14),
        "zone_b_spread_L23": float(spread_b23),
        "cos_ZoneA_ZoneB_L14": cos_ab14_v,
        "cos_ZoneA_ZoneB_L23": cos_ab23_v,
    },
    "quartile_data": q_data,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 32 complete in {time.time()-t0:.1f}s")
