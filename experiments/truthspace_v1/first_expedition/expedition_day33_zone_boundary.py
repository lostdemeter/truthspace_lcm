#!/usr/bin/env python3
"""
Expedition Day 33 — What Determines Zone D vs Zone C Membership?

Open Question 5: The boundary between the verb ocean (Zone D) and the semantic
periphery (Zone C) is empirically sharp but theoretically unexplained.
Co-occurrence specificity is the hypothesised factor: Zone C words have narrow
distributions (appear in specific contexts), Zone D words have wide distributions
(appear in almost any context).

Test strategy (pure matrix ops, no model load):
  Zone C words already occupy a specific φ-position in ISOLATION — their
  intrinsic semantic content is strong enough that even without context, the
  model assigns them a specific position.

  Zone D words land in the verb ocean in isolation — they have no preferred
  semantic direction. Their hidden state is the "average" across all contexts
  they appear in.

The "body-similarity entropy" test measures this directly:
  For each word, compute its cosine similarity to all 95 Zone C body centroids.
  Zone C member: HIGH similarity to ONE body (peaked → low entropy)
  Zone D member: MODERATE similarity to MANY bodies (flat → high entropy)

  If entropy separates Zone C from Zone D better than token_id/syllables,
  the co-occurrence specificity hypothesis is geometrically confirmed.

Secondary tests:
  1. Surface feature discriminant (token_id, syllables) — are they insufficient?
  2. Suffix/morphology patterns in Zone C vs Zone D
  3. φ-vector clustering: does Zone D form one big manifold or many sub-manifolds?
  4. What fraction of Zone D words are predicted-Zone-C by entropy alone?
"""

import os, json, re, time
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import entropy as scipy_entropy, spearmanr
from scipy.special import softmax

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day33_zone_boundary.json")

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]

LATINATE_SUFFIXES = [
    'ate', 'ize', 'ise', 'ify', 'fy', 'ment', 'tion', 'sion', 'ence', 'ance',
    'ness', 'ity', 'ive', 'ous', 'ful', 'able', 'ible', 'ial', 'ical',
    'atory', 'atory', 'ulate', 'alize', 'alize',
]
LATINATE_PREFIXES = [
    'dis', 'mis', 'pre', 'pro', 'sub', 'trans', 'over', 'under', 'inter',
    'intra', 'extra', 'counter', 'com', 'con', 'per', 'ex', 'de', 'im',
]


def batch_phi(hs_matrix, z2):
    H  = hs_matrix.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)


def body_sim_entropy(phi_vec, body_centroids_mat):
    """
    Compute cosine similarities of phi_vec to all body centroids,
    then return the entropy of the softmax-normalised distribution.
    High entropy → diffuse (Zone D-like); low entropy → peaked (Zone C-like).
    """
    sims = body_centroids_mat @ phi_vec          # (n_bodies,)
    probs = softmax(sims * 10.0)                 # temperature=0.1 → sharper
    return float(scipy_entropy(probs)), sims


def has_suffix(word, suffixes):
    return any(word.endswith(sfx) for sfx in suffixes)


def has_prefix(word, prefixes):
    return any(word.startswith(pfx) for pfx in prefixes)


t0 = time.time()

# ── Step 1: Load ─────────────────────────────────────────────────────────────
print(f"\n── Step 1: Load data ─────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# Zone C: Phase 2 words NOT in B000 (verb ocean) or B001 (secondary pole)
zone_c_entries = [(w, v) for w, v in wmap.items()
                  if v['phase'] == 2
                  and v.get('L14_body') not in ('B000', 'B001', None)
                  and w in w2i]
# Zone D: Phase 2 words in B000
zone_d_entries = [(w, v) for w, v in wmap.items()
                  if v['phase'] == 2
                  and v.get('L14_body') == 'B000'
                  and w in w2i]

zone_c_words = [w for w, _ in zone_c_entries]
zone_d_words = [w for w, _ in zone_d_entries]
zone_c_idx   = np.array([w2i[w] for w in zone_c_words])
zone_d_idx   = np.array([w2i[w] for w in zone_d_words])
zone_c_bodies_per_word = [v['L14_body'] for _, v in zone_c_entries]

print(f"  Zone C: {len(zone_c_words)} words in 95 bodies")
print(f"  Zone D: {len(zone_d_words)} words (verb ocean)")

# ── Step 2: Z2 axis ───────────────────────────────────────────────────────────
print(f"\n── Step 2: Z2 axis ───────────────────────────────────────────────")
deltas = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        wa, wb = pfx + a, pfx + b
        if wa in w2i and wb in w2i:
            d  = hs14_all[w2i[wb]] - hs14_all[w2i[wa]]
            dm = np.linalg.norm(d)
            if dm > 1e-20:
                deltas.append(d / dm)
            break
D = np.stack(deltas)
_, sv, Vt = np.linalg.svd(D, full_matrices=False)
z2  = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-20)
pct = 100 * sv[0]**2 / (np.sum(sv**2) + 1e-20)
print(f"  Z2: {pct:.1f}%  ({len(deltas)} deltas)")

# ── Step 3: φ-vectors ────────────────────────────────────────────────────────
print(f"\n── Step 3: φ-vectors ─────────────────────────────────────────────")
phi_c = batch_phi(hs14_all[zone_c_idx], z2)   # (1647, 1536)
phi_d = batch_phi(hs14_all[zone_d_idx], z2)   # (8778, 1536)
print(f"  φ_C: {phi_c.shape}   φ_D: {phi_d.shape}")

# Build Zone C body centroids
print(f"\n── Step 4: Zone C body centroids ─────────────────────────────────")
body_to_indices = defaultdict(list)
for i, body in enumerate(zone_c_bodies_per_word):
    body_to_indices[body].append(i)

body_ids       = sorted(body_to_indices.keys())
body_centroids = []
body_sizes     = []
body_labels_map = {}
for bid in body_ids:
    idxs = body_to_indices[bid]
    phis = phi_c[idxs]
    c    = phis.mean(axis=0)
    c   /= (np.linalg.norm(c) + 1e-20)
    body_centroids.append(c)
    body_sizes.append(len(idxs))
    body_labels_map[bid] = wmap[zone_c_words[idxs[0]]].get('L14_label', '?')

body_centroids_mat = np.stack(body_centroids)   # (95, 1536)
print(f"  Built {len(body_centroids)} body centroids")
print(f"  Body size range: {min(body_sizes)}–{max(body_sizes)} words")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 1 — Surface Feature Discriminant")
print(f"{'='*65}")

c_tids = np.array([v['token_id'] for _, v in zone_c_entries])
d_tids = np.array([v['token_id'] for _, v in zone_d_entries])
c_syls = np.array([v['syllables'] for _, v in zone_c_entries])
d_syls = np.array([v['syllables'] for _, v in zone_d_entries])
c_lens = np.array([len(w) for w in zone_c_words])
d_lens = np.array([len(w) for w in zone_d_words])

print(f"\n  Token_id:  Zone C mean={c_tids.mean():.0f}  Zone D mean={d_tids.mean():.0f}  "
      f"Δ={d_tids.mean()-c_tids.mean():.0f}")
print(f"  Syllables: Zone C mean={c_syls.mean():.2f}  Zone D mean={d_syls.mean():.2f}  "
      f"Δ={d_syls.mean()-c_syls.mean():.2f}")
print(f"  Word len:  Zone C mean={c_lens.mean():.2f}  Zone D mean={d_lens.mean():.2f}  "
      f"Δ={d_lens.mean()-c_lens.mean():.2f}")

# Naive threshold classifier: can token_id or syllables predict zone?
# Predict Zone D if token_id > threshold
for thresh_pct in [25, 50, 75]:
    tid_thresh = np.percentile(np.concatenate([c_tids, d_tids]), thresh_pct)
    tp = np.sum(d_tids > tid_thresh)
    fp = np.sum(c_tids > tid_thresh)
    tn = np.sum(c_tids <= tid_thresh)
    fn = np.sum(d_tids <= tid_thresh)
    acc = (tp + tn) / (len(c_tids) + len(d_tids))
    print(f"\n  tid threshold @p{thresh_pct} ({tid_thresh:.0f}): "
          f"accuracy={acc:.3f}  (tp={tp} fp={fp} tn={tn} fn={fn})")

# Suffix analysis
print(f"\n  Latinate-suffix words:")
c_lat = sum(1 for w in zone_c_words if has_suffix(w, LATINATE_SUFFIXES))
d_lat = sum(1 for w in zone_d_words if has_suffix(w, LATINATE_SUFFIXES))
print(f"    Zone C: {c_lat}/{len(zone_c_words)} = {100*c_lat/len(zone_c_words):.1f}%")
print(f"    Zone D: {d_lat}/{len(zone_d_words)} = {100*d_lat/len(zone_d_words):.1f}%")

# Most common suffixes per zone
def top_suffixes(words, n=8):
    sfx_counter = Counter()
    for w in words:
        for length in (3, 4, 5, 6):
            if len(w) > length:
                sfx_counter[w[-length:]] += 1
    return sfx_counter.most_common(n)

print(f"\n  Top Zone C suffixes: {top_suffixes(zone_c_words)}")
print(f"  Top Zone D suffixes: {top_suffixes(zone_d_words)}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 2 — Body-Similarity Entropy Test (Main Test)")
print(f"{'='*65}")
print(f"\n  Computing body-similarity entropy for all Zone C and Zone D words...")

# Zone C entropy
c_entropies  = []
c_max_sims   = []
c_own_sims   = []     # similarity to own body centroid
for i, (w, bid) in enumerate(zip(zone_c_words, zone_c_bodies_per_word)):
    body_idx_in_list = body_ids.index(bid)
    ent, sims = body_sim_entropy(phi_c[i], body_centroids_mat)
    c_entropies.append(ent)
    c_max_sims.append(float(sims.max()))
    c_own_sims.append(float(sims[body_idx_in_list]))

c_entropies = np.array(c_entropies)
c_max_sims  = np.array(c_max_sims)
c_own_sims  = np.array(c_own_sims)

# Zone D entropy (batched for speed)
print(f"  Zone D entropy (batched, {len(zone_d_words)} words)...")
# Compute all similarities at once: (8778, 95)
all_d_sims = phi_d @ body_centroids_mat.T    # (8778, 95)
d_entropies = np.array([
    float(scipy_entropy(softmax(row * 10.0)))
    for row in all_d_sims
])
d_max_sims = all_d_sims.max(axis=1)

print(f"\n  Body-similarity entropy:")
print(f"    Zone C: mean={c_entropies.mean():.4f}  median={np.median(c_entropies):.4f}  "
      f"std={c_entropies.std():.4f}")
print(f"    Zone D: mean={d_entropies.mean():.4f}  median={np.median(d_entropies):.4f}  "
      f"std={d_entropies.std():.4f}")
print(f"    Zone D − Zone C Δ: {d_entropies.mean()-c_entropies.mean():+.4f}")

print(f"\n  Max body similarity (best single-body match):")
print(f"    Zone C: mean={c_max_sims.mean():.4f}  median={np.median(c_max_sims):.4f}")
print(f"    Zone D: mean={d_max_sims.mean():.4f}  median={np.median(d_max_sims):.4f}")
print(f"    Zone D − Zone C Δ: {d_max_sims.mean()-c_max_sims.mean():+.4f}")

print(f"\n  Zone C own-body similarity:")
print(f"    mean={c_own_sims.mean():.4f}  median={np.median(c_own_sims):.4f}  "
      f"std={c_own_sims.std():.4f}")

# Entropy-based classifier: predict Zone D if entropy > threshold
combined_ent = np.concatenate([c_entropies, d_entropies])
labels       = np.concatenate([np.zeros(len(c_entropies)), np.ones(len(d_entropies))])
best_acc, best_thresh = 0.0, 0.0
for thresh in np.percentile(combined_ent, np.linspace(5, 95, 50)):
    pred = (combined_ent > thresh).astype(int)
    acc  = (pred == labels).mean()
    if acc > best_acc:
        best_acc, best_thresh = acc, thresh

print(f"\n  Entropy classifier:")
print(f"    Best threshold: {best_thresh:.4f}  Accuracy: {best_acc:.3f}")
pred_best = (combined_ent > best_thresh).astype(int)
tp = int(((pred_best == 1) & (labels == 1)).sum())
fp = int(((pred_best == 1) & (labels == 0)).sum())
tn = int(((pred_best == 0) & (labels == 0)).sum())
fn = int(((pred_best == 0) & (labels == 1)).sum())
prec = tp / (tp + fp + 1e-9)
rec  = tp / (tp + fn + 1e-9)
print(f"    Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {2*prec*rec/(prec+rec+1e-9):.3f}")
print(f"    tp={tp} fp={fp} tn={tn} fn={fn}")

# Spearman correlation of entropy with token_id
all_tids = np.concatenate([c_tids, d_tids])
r_ent_tid, p_ent_tid = spearmanr(all_tids, combined_ent)
print(f"\n  Spearman r(token_id, entropy): {r_ent_tid:+.4f}  (p={p_ent_tid:.2e})")
print(f"  → entropy is {'independent of' if abs(r_ent_tid)<0.1 else 'correlated with'} token_id")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 3 — Entropy Distribution Details")
print(f"{'='*65}")

# Entropy decile analysis
print(f"\n  Zone C words by entropy decile (low entropy = specific body):")
decile_edges = np.percentile(c_entropies, np.arange(0, 110, 10))
for d_lo, d_hi in zip(decile_edges[:-1], decile_edges[1:]):
    mask  = (c_entropies >= d_lo) & (c_entropies < d_hi)
    words = [zone_c_words[i] for i in np.where(mask)[0][:6]]
    print(f"    ent [{d_lo:.3f},{d_hi:.3f}]: {words}")

print(f"\n  Zone D words at lowest entropy (most body-like = escaped?):")
d_low_ent_idx = np.argsort(d_entropies)[:30]
print(f"    {[zone_d_words[i] for i in d_low_ent_idx]}")
print(f"    entropies: {d_entropies[d_low_ent_idx].tolist()[:10]}")

print(f"\n  Zone D words at highest entropy (most diffuse):")
d_high_ent_idx = np.argsort(d_entropies)[-20:]
print(f"    {[zone_d_words[i] for i in d_high_ent_idx[::-1]]}")

# What bodies do the Zone D low-entropy words resemble?
print(f"\n  For Zone D words with lowest entropy — which body do they resemble?")
for i in d_low_ent_idx[:15]:
    w     = zone_d_words[i]
    sims  = all_d_sims[i]
    top_b = np.argmax(sims)
    print(f"    {w:<20s} → best match: {body_ids[top_b]} "
          f"({body_labels_map[body_ids[top_b]][:30]}) "
          f"sim={sims[top_b]:.4f}  ent={d_entropies[i]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 4 — Z2 Projection: Zone C vs Zone D")
print(f"{'='*65}")

# What is the Z2 projection (frequency axis) for Zone C vs Zone D?
# This tells us if the boundary is related to the frequency axis at all
all_hs_c = hs14_all[zone_c_idx].astype(np.float64)
all_hs_d = hs14_all[zone_d_idx].astype(np.float64)

# Normalise then project onto z2
hs_c_n = all_hs_c / (np.linalg.norm(all_hs_c, axis=1, keepdims=True) + 1e-20)
hs_d_n = all_hs_d / (np.linalg.norm(all_hs_d, axis=1, keepdims=True) + 1e-20)
z2_proj_c = hs_c_n @ z2
z2_proj_d = hs_d_n @ z2

print(f"\n  Z2 projection (= frequency/degeneration axis):")
print(f"    Zone C: mean={z2_proj_c.mean():.4f}  std={z2_proj_c.std():.4f}")
print(f"    Zone D: mean={z2_proj_d.mean():.4f}  std={z2_proj_d.std():.4f}")
print(f"    Δ = {z2_proj_d.mean()-z2_proj_c.mean():+.4f}")

r_z2_ent_c, _ = spearmanr(z2_proj_c, c_entropies)
r_z2_ent_d, _ = spearmanr(z2_proj_d, d_entropies)
print(f"\n  Spearman r(Z2_projection, entropy):")
print(f"    Zone C: {r_z2_ent_c:+.4f}")
print(f"    Zone D: {r_z2_ent_d:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 5 — The Zone C/D Boundary: What Separates Them?")
print(f"{'='*65}")

# Multi-feature comparison
print(f"\n  Feature comparison (Zone C vs Zone D):")
print(f"  {'Feature':<30s}  {'Zone C':>12s}  {'Zone D':>12s}  {'Sep (d)':>8s}")
print(f"  {'-'*65}")

def cohen_d(a, b):
    pooled = np.sqrt(0.5 * (a.std()**2 + b.std()**2))
    return (a.mean() - b.mean()) / (pooled + 1e-20)

features = [
    ("token_id",        c_tids.astype(float),   d_tids.astype(float)),
    ("syllables",       c_syls.astype(float),   d_syls.astype(float)),
    ("word_length",     c_lens.astype(float),   d_lens.astype(float)),
    ("Z2_projection",   z2_proj_c,              z2_proj_d),
    ("body_sim_entropy",c_entropies,            d_entropies),
    ("max_body_sim",    c_max_sims,             d_max_sims),
]
feature_cohen_d = {}
for name, fc, fd in features:
    d = cohen_d(fc, fd)
    feature_cohen_d[name] = d
    print(f"  {name:<30s}  {fc.mean():>12.4f}  {fd.mean():>12.4f}  {d:>+8.3f}")

best_feat = max(feature_cohen_d, key=lambda k: abs(feature_cohen_d[k]))
print(f"\n  Best separator: {best_feat}  (Cohen's d = {feature_cohen_d[best_feat]:+.3f})")

# Combined logistic regression proxy (no sklearn needed — just LDA)
from numpy.linalg import lstsq
Xc = np.column_stack([c_entropies, c_max_sims, z2_proj_c,
                      c_tids/100000, c_syls])
Xd = np.column_stack([d_entropies, d_max_sims, z2_proj_d,
                      d_tids/100000, d_syls])
X  = np.vstack([Xc, Xd])
y  = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xd))])

# LDA-style: find direction that maximises between/within ratio
mu_c = Xc.mean(axis=0)
mu_d = Xd.mean(axis=0)
Sw = np.cov(Xc.T) * (len(Xc)-1) + np.cov(Xd.T) * (len(Xd)-1)
try:
    w_lda = np.linalg.solve(Sw, mu_d - mu_c)
    w_lda /= np.linalg.norm(w_lda) + 1e-20
    proj_c = Xc @ w_lda
    proj_d = Xd @ w_lda
    # Simple threshold
    thresh_lda = (proj_c.mean() + proj_d.mean()) / 2
    pred_lda = (np.concatenate([proj_c, proj_d]) > thresh_lda).astype(int)
    acc_lda = (pred_lda == y).mean()
    print(f"\n  LDA (5 features): accuracy = {acc_lda:.3f}")
    lda_weights = dict(zip(
        ['entropy', 'max_body_sim', 'Z2_proj', 'token_id', 'syllables'],
        w_lda.tolist()
    ))
    print(f"  LDA weights: {lda_weights}")
    print(f"  → dominant feature: "
          f"{max(lda_weights, key=lambda k: abs(lda_weights[k]))}")
except Exception as e:
    print(f"  LDA failed: {e}")
    acc_lda = None
    lda_weights = {}


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SECTION 6 — Interpretation: What IS the Boundary?")
print(f"{'='*65}")

entropy_sep  = abs(feature_cohen_d.get('body_sim_entropy', 0))
maxsim_sep   = abs(feature_cohen_d.get('max_body_sim', 0))
z2_sep       = abs(feature_cohen_d.get('Z2_projection', 0))
tid_sep      = abs(feature_cohen_d.get('token_id', 0))

print(f"""
  Summary of discriminant power (|Cohen's d|):
    body_sim_entropy:  {entropy_sep:.3f}  ← co-occurrence breadth (main hypothesis)
    max_body_sim:      {maxsim_sep:.3f}  ← does the word resemble ANY specific body?
    Z2_projection:     {z2_sep:.3f}  ← frequency axis
    token_id:          {tid_sep:.3f}  ← raw frequency rank
    syllables:         {abs(feature_cohen_d.get("syllables",0)):.3f}  ← surface form

  Interpretation:""")

if entropy_sep > 0.5 and entropy_sep > tid_sep * 2:
    print(f"  ✓ CONFIRMED: co-occurrence specificity (entropy) is the dominant predictor.")
    print(f"    The Zone C/D boundary is a SEMANTIC boundary, not a surface-form boundary.")
    print(f"    Zone C words: specific φ-body match → narrow co-occurrence → crystallise.")
    print(f"    Zone D words: diffuse body matches → wide co-occurrence → dissolve into ocean.")
elif entropy_sep > 0.3:
    print(f"  ~ PARTIAL: entropy is a moderate predictor. The boundary has both geometric")
    print(f"    and surface-form components.")
else:
    print(f"  ✗ NOT CONFIRMED: entropy doesn't cleanly separate the zones.")
    print(f"    The boundary may be determined by a factor not yet measured.")

# How many Zone D words have entropy < median Zone C entropy?
c_med_ent = np.median(c_entropies)
n_d_below_c_med = (d_entropies < c_med_ent).sum()
print(f"\n  Zone D words with entropy < Zone C median ({c_med_ent:.4f}): "
      f"{n_d_below_c_med} / {len(d_entropies)} = {100*n_d_below_c_med/len(d_entropies):.1f}%")
print(f"  → These are Zone D words that LOOK like Zone C by specificity alone.")
print(f"  → They may have crystallised at L23 (cf. Day 31: 218 escapees).")


# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n── Saving results ─────────────────────────────────────────────")
result = {
    "meta": {
        "experiment": "Day 33 — Zone D vs Zone C boundary",
        "zone_c_n": len(zone_c_words),
        "zone_d_n": len(zone_d_words),
        "n_bodies": len(body_ids),
    },
    "surface_features": {
        "c_token_id_mean": float(c_tids.mean()),
        "d_token_id_mean": float(d_tids.mean()),
        "c_syllables_mean": float(c_syls.mean()),
        "d_syllables_mean": float(d_syls.mean()),
        "c_len_mean": float(c_lens.mean()),
        "d_len_mean": float(d_lens.mean()),
    },
    "entropy_test": {
        "c_entropy_mean":    float(c_entropies.mean()),
        "c_entropy_median":  float(np.median(c_entropies)),
        "d_entropy_mean":    float(d_entropies.mean()),
        "d_entropy_median":  float(np.median(d_entropies)),
        "entropy_classifier_accuracy": float(best_acc),
        "entropy_best_threshold":      float(best_thresh),
        "c_max_body_sim_mean": float(c_max_sims.mean()),
        "d_max_body_sim_mean": float(d_max_sims.mean()),
        "c_own_body_sim_mean": float(c_own_sims.mean()),
        "spearman_entropy_tokenid": float(r_ent_tid),
    },
    "cohen_d": {k: float(v) for k, v in feature_cohen_d.items()},
    "lda": {
        "accuracy": float(acc_lda) if acc_lda is not None else None,
        "weights": {k: float(v) for k, v in lda_weights.items()},
    },
    "z2_projection": {
        "c_mean": float(z2_proj_c.mean()),
        "d_mean": float(z2_proj_d.mean()),
    },
    "d_low_entropy_words": [zone_d_words[i] for i in d_low_ent_idx[:20]],
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 33 complete in {time.time()-t0:.1f}s")
