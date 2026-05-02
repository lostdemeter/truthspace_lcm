"""
SECOND EXPEDITION — DAY 1
=========================
The Rotation Model Test

The First Expedition described semantic transformations as translations:
  pred = e(src) + scale * mean_chord_direction

This day tests a cleaner geometric claim:

  Semantic transformations ARE rotations on the unit sphere.
  The "axis" of a semantic relationship is the geodesic tangent n̂.
  The "scale" is the rotation angle θ = arccos(e_n(src)·e_n(tgt)).

If the rotation model is correct:
  - θ should be MORE CONSISTENT across word pairs than chord length |Δ|
    (lower coefficient of variation: CV(θ) < CV(|Δ|))
  - The geodesic tangent n̂ = (e_n(tgt) - cosθ·e_n(src)) / sinθ
    should be MORE COHERENT (higher pairwise cosine) than the raw chord direction
  - best_scale from grid search should satisfy: best_scale ≈ ||e_src|| · 2·sin(θ/2)
  - EN and ZH versions of each axis should show the same rotation angle θ

Natural observations to record as they arise:
  - Do rotation angles cluster near arccos(1/φ^n) — the golden angle series?
  - What does the distribution of θ across all axes look like?
  - How much "radial contamination" does the chord introduce vs the geodesic tangent?

Darwin's rule: record what you ACTUALLY find, not what you expect.

Script: second_expedition/day1_rotation_model_test.py
"""

import torch, numpy as np, sys

# ── model load ─────────────────────────────────────────────────────────────────
print("Loading Qwen2-1.5B-Instruct embedding matrix...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
norms_all = np.linalg.norm(W_E, axis=1)
W_n = (W_E / (norms_all[:, None] + 1e-8)).astype(np.float32)
print(f"  shape={W_E.shape}  "
      f"embedding norm: mean={norms_all.mean():.3f}  std={norms_all.std():.3f}")

# ── vocabulary masks ───────────────────────────────────────────────────────────
EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(len(W_E))], dtype=bool)

ZH_MASK = np.array([
    any('\u4e00' <= c <= '\u9fff' for c in tok.decode([i]).strip())
    for i in range(len(W_E))], dtype=bool)

# ── utilities ──────────────────────────────────────────────────────────────────
def normed(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def get_emb_any(word):
    ids = tok(word, add_special_tokens=False)['input_ids']
    if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def source_ids(word):
    ids = set()
    for p in [word, ' ' + word,
              word[0].upper() + word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

# ── φ reference constants ──────────────────────────────────────────────────────
PHI = (1 + 5**0.5) / 2
THETA_PHI  = np.degrees(np.arccos(1 / PHI))       # ≈ 51.83°
THETA_PHI2 = np.degrees(np.arccos(1 / PHI**2))    # ≈ 38.17°
THETA_PHI3 = np.degrees(np.arccos(1 / PHI**3))    # ≈ 27.95°
THETA_PHI4 = np.degrees(np.arccos(1 / PHI**4))    # ≈ 20.90°
# golden angle: 360°/φ² ≈ 137.5°, half of that ≈ 68.75°
GOLDEN_ANGLE = 360.0 / PHI**2

print(f"\nφ = {PHI:.6f}")
print(f"Reference rotation angles (arccos(1/φ^n)):")
print(f"  n=1: {THETA_PHI:.3f}°   n=2: {THETA_PHI2:.3f}°   "
      f"n=3: {THETA_PHI3:.3f}°   n=4: {THETA_PHI4:.3f}°")
print(f"  golden angle / 2 = {GOLDEN_ANGLE/2:.3f}°")

# ── word pair datasets ─────────────────────────────────────────────────────────
EN_GENDER = [('man','woman'),('king','queen'),('father','mother'),
             ('son','daughter'),('boy','girl'),('husband','wife'),
             ('uncle','aunt'),('prince','princess'),('brother','sister'),
             ('actor','actress')]
ZH_GENDER = [('男人','女人'),('国王','女王'),('父亲','母亲'),
             ('儿子','女儿'),('男孩','女孩'),('丈夫','妻子'),
             ('叔叔','阿姨'),('王子','公主'),('兄弟','姐妹')]
EN_SIZE   = [('big','small'),('large','tiny'),('huge','little'),
             ('tall','short'),('long','brief'),('fat','thin'),
             ('wide','narrow'),('heavy','light'),('strong','weak'),
             ('hot','cold')]
ZH_SIZE   = [('大','小'),('长','短'),('热','冷'),('高','低'),
             ('重','轻'),('厚','薄'),('宽','窄'),('强','弱')]
EN_SENT   = [('good','bad'),('happy','sad'),('love','hate'),
             ('bright','dark'),('beautiful','ugly'),('clean','dirty'),
             ('right','wrong'),('best','worst')]
ZH_SENT   = [('好','坏'),('美','丑'),('爱','恨'),('快乐','悲伤'),
             ('亮','暗'),('对','错')]
EN_PLURAL = [('cat','cats'),('dog','dogs'),('house','houses'),
             ('tree','trees'),('book','books'),('car','cars'),
             ('bird','birds'),('ship','ships')]
EN_CAPITAL = [('france','paris'),('germany','berlin'),('japan','tokyo'),
              ('china','beijing'),('italy','rome'),('spain','madrid')]

AXES = {
    'EN_gender':   (EN_GENDER,  get_emb,     EN_MASK),
    'ZH_gender':   (ZH_GENDER,  get_emb_any, ZH_MASK),
    'EN_size':     (EN_SIZE,    get_emb,     EN_MASK),
    'ZH_size':     (ZH_SIZE,    get_emb_any, ZH_MASK),
    'EN_sentiment':(EN_SENT,    get_emb,     EN_MASK),
    'ZH_sentiment':(ZH_SENT,    get_emb_any, ZH_MASK),
    'EN_plural':   (EN_PLURAL,  get_emb,     EN_MASK),
    'EN_capital':  (EN_CAPITAL, get_emb,     EN_MASK),
}

# ── core geometry for one pair ─────────────────────────────────────────────────
def pair_geometry(src, tgt, get_fn):
    """
    Returns the full rotation geometry for one word pair.
    All measurements are on the NORMALISED unit sphere.

    Returns dict with:
      theta      — rotation angle in degrees (arccos of dot product)
      chord      — Euclidean distance between unit vectors (= 2·sin(θ/2))
      tangent    — geodesic tangent at src pointing toward tgt
                   = (e_n_tgt - cosθ·e_n_src) / sinθ
      chord_dir  — raw chord direction (what the first expedition used)
      en_s, en_t — unit normalised embeddings
      raw_norm_s — norm of the raw source embedding
    """
    es, idx_s = get_fn(src)
    et, idx_t = get_fn(tgt)
    if es is None or et is None: return None

    raw_norm_s = float(np.linalg.norm(es))
    en_s = normed(es)
    en_t = normed(et)

    cos_th = float(np.clip(np.dot(en_s, en_t), -1.0, 1.0))
    theta  = float(np.degrees(np.arccos(cos_th)))
    chord  = float(np.linalg.norm(en_t - en_s))      # = 2·sin(θ/2)

    # Geodesic tangent: remove the component of en_t along en_s, then normalise
    perp   = en_t - cos_th * en_s
    sin_th = float(np.sqrt(max(0.0, 1.0 - cos_th**2)))
    tangent = perp / (sin_th + 1e-12)   # unit tangent (geodesic direction at src)

    chord_dir = normed(en_t - en_s)     # chord direction (non-geodesic)

    return dict(theta=theta, chord=chord, cos_th=cos_th,
                tangent=tangent, chord_dir=chord_dir,
                en_s=en_s, en_t=en_t,
                raw_norm_s=raw_norm_s, sin_th=sin_th)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Measure θ and chord for every pair in every axis.
#          Core test: is CV(θ) < CV(chord)?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — Is the Rotation Angle θ More Consistent than Chord Length?")
print("  Rotation model predicts: CV(θ) < CV(chord)")
print("  (CV = coefficient of variation = std/mean)")
print("═"*72)
print(f"\n{'Axis':<15} {'n':>3}  {'θ values (°)':>38}  {'CV_θ':>6}  {'CV_|Δ|':>7}  verdict")
print("─"*85)

all_geom = {}   # store for later phases

for ax_name, (pairs, get_fn, mask) in AXES.items():
    geoms = []
    for s, t in pairs:
        g = pair_geometry(s, t, get_fn)
        if g: geoms.append((s, t, g))

    if len(geoms) < 2: continue
    all_geom[ax_name] = geoms

    thetas = np.array([g['theta'] for _, _, g in geoms])
    chords = np.array([g['chord'] for _, _, g in geoms])

    cv_th = thetas.std() / thetas.mean()
    cv_ch = chords.std() / chords.mean()
    verdict = "ROTATION ✓" if cv_th < cv_ch else "translation"
    th_str = "  ".join(f"{t:.1f}" for t in thetas)

    print(f"{ax_name:<15} {len(geoms):>3}  [{th_str}]")
    print(f"{'':>15}      mean_θ={thetas.mean():.2f}°  std_θ={thetas.std():.2f}°  "
          f"CV_θ={cv_th:.3f}   CV_|Δ|={cv_ch:.3f}   {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: φ angle proximity — do semantic rotations fall near arccos(1/φ^n)?
#          Record the actual distances as a naturalist would: whatever we see.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — φ Angle Investigation")
print(f"  φ-series: {THETA_PHI:.2f}°  {THETA_PHI2:.2f}°  "
      f"{THETA_PHI3:.2f}°  {THETA_PHI4:.2f}°")
print("═"*72)

for ax_name, geoms in all_geom.items():
    thetas = np.array([g['theta'] for _, _, g in geoms])
    refs   = [THETA_PHI, THETA_PHI2, THETA_PHI3, THETA_PHI4]
    labels = ['1/φ', '1/φ²', '1/φ³', '1/φ⁴']
    dists  = [float(np.abs(thetas.mean() - r)) for r in refs]
    best_i = int(np.argmin(dists))
    flag   = " ◀◀ φ-MATCH" if dists[best_i] < 2.0 else \
             " ◀ φ-close"  if dists[best_i] < 5.0 else ""
    print(f"  {ax_name:<15}  mean_θ={thetas.mean():.2f}°  "
          f"closest: arccos({labels[best_i]})={refs[best_i]:.2f}°  "
          f"Δ={dists[best_i]:.2f}°{flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Cross-lingual rotation angle comparison.
#          Hypothesis: EN and ZH versions of the same semantic axis have the
#          same mean rotation angle (they are the same rotation).
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — Cross-Lingual θ: Do EN and ZH Execute the Same Rotation?")
print("═"*72)

for ax_type in ['gender', 'size', 'sentiment']:
    en_key = f'EN_{ax_type}';  zh_key = f'ZH_{ax_type}'
    if en_key not in all_geom or zh_key not in all_geom: continue
    en_th = np.array([g['theta'] for _, _, g in all_geom[en_key]])
    zh_th = np.array([g['theta'] for _, _, g in all_geom[zh_key]])
    delta  = abs(en_th.mean() - zh_th.mean())
    pooled = np.concatenate([en_th, zh_th]).std()
    if delta < 2.0:     verdict = "SAME rotation ✓"
    elif delta < 5.0:   verdict = "close (~same)"
    elif delta < pooled: verdict = "within spread"
    else:               verdict = "different rotations"
    print(f"\n  {ax_type.upper()}")
    print(f"    EN: mean={en_th.mean():.2f}°  std={en_th.std():.2f}°  "
          f"range=[{en_th.min():.1f}°, {en_th.max():.1f}°]")
    print(f"    ZH: mean={zh_th.mean():.2f}°  std={zh_th.std():.2f}°  "
          f"range=[{zh_th.min():.1f}°, {zh_th.max():.1f}°]")
    print(f"    |Δmean| = {delta:.2f}°  pooled_std={pooled:.2f}°  → {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Geodesic tangent coherence vs chord direction coherence.
#          The tangent direction n̂ is the correct spherical direction.
#          The chord direction is the Euclidean shortcut.
#          Measure pairwise cos(tangent_i, tangent_j) vs cos(chord_i, chord_j).
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — Geodesic Tangent Coherence vs Chord Direction Coherence")
print("  Higher pairwise cosine = more consistent direction across word pairs")
print("═"*72)
print(f"\n{'Axis':<15}  {'mean_cos(tan)':>13}  {'mean_cos(chord)':>15}  "
      f"{'Δ':>7}  verdict")
print("─"*65)

for ax_name, geoms in all_geom.items():
    tangents   = np.array([g['tangent']   for _, _, g in geoms])
    chord_dirs = np.array([g['chord_dir'] for _, _, g in geoms])
    tan_cos, crd_cos = [], []
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            tan_cos.append(float(np.dot(tangents[i],   tangents[j])))
            crd_cos.append(float(np.dot(chord_dirs[i], chord_dirs[j])))
    tc = np.array(tan_cos);  cc = np.array(crd_cos)
    delta = tc.mean() - cc.mean()
    verdict = "TANGENT more coherent ✓" if delta > 0.005 else \
              "chord more coherent"  if delta < -0.005 else "≈ same"
    print(f"{ax_name:<15}  {tc.mean():>13.4f}  {cc.mean():>15.4f}  "
          f"{delta:>+7.4f}  {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Scale prediction.
#          In normalized space: chord = 2·sin(θ/2)
#          In raw space: best_scale ≈ ||e_src|| · 2·sin(θ/2)
#          Test: does 2·sin(mean_θ/2) · mean(||e_src||) predict known scales?
#
#          Known best scales from First Expedition Day 355:
#            EN_gender = 0.429, ZH_gender = 0.429
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — Does mean_θ Predict the First Expedition's Optimal Scale?")
print("  Prediction: best_scale ≈ mean(||e_src||) · 2·sin(mean_θ / 2)")
print("═"*72)

KNOWN_SCALES = {'EN_gender': 0.429, 'ZH_gender': 0.429}

print(f"\n{'Axis':<15}  {'mean_θ':>7}  {'mean_||src||':>12}  "
      f"{'predicted':>10}  {'known':>8}  {'ratio(k/p)':>10}")
print("─"*70)

for ax_name, geoms in all_geom.items():
    thetas     = np.array([g['theta']       for _, _, g in geoms])
    raw_norms  = np.array([g['raw_norm_s']  for _, _, g in geoms])
    mean_th    = float(thetas.mean())
    mean_norm  = float(raw_norms.mean())
    predicted  = mean_norm * 2 * np.sin(np.radians(mean_th) / 2)
    known      = KNOWN_SCALES.get(ax_name)
    ratio_str  = f"{known/predicted:.4f}" if known else "  (not measured)"
    known_str  = f"{known:.4f}" if known else "     —"
    print(f"{ax_name:<15}  {mean_th:>7.2f}°  {mean_norm:>12.4f}  "
          f"{predicted:>10.4f}  {known_str:>8}  {ratio_str:>10}")

print("\n  Note: A constant ratio across axes means best_scale = k·predicted_scale")
print("  where k absorbs the un-normalized embedding geometry.")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Radial contamination in the chord direction.
#          Chord = (cosθ−1)·e_n_src + sinθ·tangent
#          The (cosθ−1) term is a radial component pulling toward the src.
#          This means the chord direction ≠ tangent direction by a known formula.
#          Measure: what fraction of each chord is the radial term vs tangential?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — Radial Contamination in the Chord Direction")
print("  Chord = (cosθ−1)·e_src  +  sinθ·tangent")
print("  Radial fraction = |cosθ−1| / (|cosθ−1| + sinθ)")
print("═"*72)
print(f"\n{'Axis':<15}  {'mean_radial_frac':>16}  {'std':>6}  "
      f"{'cos(chord_ax, src_centroid)':>27}")
print("─"*65)

for ax_name, geoms in all_geom.items():
    rad_fracs = []
    src_embs  = []
    chord_dirs = []
    for _, _, g in geoms:
        th = np.radians(g['theta'])
        radial_mag  = abs(np.cos(th) - 1)
        tangent_mag = abs(np.sin(th))
        rad_fracs.append(radial_mag / (radial_mag + tangent_mag + 1e-12))
        src_embs.append(g['en_s'])
        chord_dirs.append(g['chord_dir'])
    rf = np.array(rad_fracs)
    # How aligned is the mean chord axis with the source centroid?
    src_centroid = normed(np.mean(src_embs, axis=0))
    mean_chord   = normed(np.mean(chord_dirs, axis=0))
    centroid_cos = abs(float(np.dot(mean_chord, src_centroid)))
    print(f"{ax_name:<15}  {rf.mean():>16.4f}  {rf.std():>6.4f}  "
          f"{centroid_cos:>27.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Build geodesic tangent mean axis and compare retrieval accuracy
#          to the chord mean axis (train accuracy — same pairs).
#          This is the practical test: does the rotation model navigate better?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 7 — Navigation Accuracy: Geodesic Tangent vs Chord Axis")
print("  (train accuracy — same word pairs used for axis and test)")
print("  Rotation step: pred = cosθ·e_src + sinθ·tangent_axis  (applied to unit vecs)")
print("  Chord step:    pred = e_src_raw + scale·chord_axis     (raw, then normalize)")
print("═"*72)
print(f"\n{'Axis':<15}  {'n':>3}  {'chord_acc':>10}  scale   "
      f"{'geodesic_acc':>12}  θ(°)    {'chord_cos':>10}")
print("─"*72)

def eval_chord_axis(ax_dir, pairs, get_fn, mask):
    best_s, best_acc = 0.0, 0
    for s in np.linspace(0.02, 6.0, 60):
        c = sum(1 for src, tgt in pairs
                if get_fn(src)[0] is not None and
                nn_ret(get_fn(src)[0] + s * ax_dir,
                       source_ids(src), mask)[0] == tgt)
        if c > best_acc: best_acc = c; best_s = s
    return best_s, best_acc

def eval_geodesic_axis(tangent_ax, mean_th_deg, pairs, get_fn, mask):
    """
    Rotation-model prediction from the unit sphere:
      pred_n = cosθ·e_n_src + sinθ·tangent_axis
    Then find NN in the full vocabulary using normalised prediction.
    Optimise θ around mean_th.
    """
    best_th, best_acc = mean_th_deg, 0
    for th_deg in np.linspace(max(1, mean_th_deg - 30), mean_th_deg + 30, 61):
        th = np.radians(th_deg)
        c = 0
        for src, tgt in pairs:
            es, _ = get_fn(src)
            if es is None: continue
            en_s = normed(es)
            pred_n = np.cos(th) * en_s + np.sin(th) * tangent_ax
            w, _, _ = nn_ret(pred_n, source_ids(src), mask)
            if w == tgt: c += 1
        if c > best_acc: best_acc = c; best_th = th_deg
    return best_th, best_acc

for ax_name, (pairs, get_fn, mask) in AXES.items():
    geoms = all_geom.get(ax_name, [])
    if not geoms: continue
    n = len(geoms)

    # Build chord axis and tangent axis from all pairs
    chord_dirs = np.array([g['chord_dir'] for _, _, g in geoms])
    tangents   = np.array([g['tangent']   for _, _, g in geoms])
    chord_ax   = normed(np.mean(chord_dirs, axis=0))
    tangent_ax = normed(np.mean(tangents,   axis=0))
    mean_th    = float(np.mean([g['theta'] for _, _, g in geoms]))

    ax_cos     = float(np.dot(chord_ax, tangent_ax))

    scale_ch, acc_ch    = eval_chord_axis(chord_ax, pairs, get_fn, mask)
    th_geo, acc_geo     = eval_geodesic_axis(tangent_ax, mean_th, pairs, get_fn, mask)

    better = "GEODESIC ✓" if acc_geo > acc_ch else \
             "chord"      if acc_ch > acc_geo else "TIE"
    print(f"{ax_name:<15}  {n:>3}  {acc_ch:>3}/{n:<5}  {scale_ch:.3f}   "
          f"{acc_geo:>3}/{n:<5}  {th_geo:>5.1f}°   {ax_cos:>10.4f}  {better}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: Whole-vocabulary θ distribution.
#          Sample 200 random content-word pairs and measure θ distribution.
#          This gives us the "background" θ to compare our semantic pairs against.
#          Are semantic pairs special in their rotation angle?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 8 — Background θ Distribution (200 Random EN Content-Word Pairs)")
print("  Are semantic axis pairs special in their rotation angle?")
print("═"*72)

rng = np.random.default_rng(42)
en_ids = np.where(EN_MASK)[0]
sample_idx = rng.choice(en_ids, size=400, replace=False)
bg_thetas = []
for i in range(0, len(sample_idx), 2):
    e1 = normed(W_E[sample_idx[i]])
    e2 = normed(W_E[sample_idx[i+1]])
    cos_th = float(np.clip(np.dot(e1, e2), -1, 1))
    bg_thetas.append(np.degrees(np.arccos(cos_th)))
bg = np.array(bg_thetas)

print(f"\n  Background (random pairs):  mean={bg.mean():.2f}°  "
      f"std={bg.std():.2f}°  median={np.median(bg):.2f}°")
print(f"  Percentiles: p10={np.percentile(bg,10):.1f}°  "
      f"p25={np.percentile(bg,25):.1f}°  p75={np.percentile(bg,75):.1f}°  "
      f"p90={np.percentile(bg,90):.1f}°")

print(f"\n  Semantic axis mean θ values vs background:")
for ax_name, geoms in all_geom.items():
    thetas = np.array([g['theta'] for _, _, g in geoms])
    z = (thetas.mean() - bg.mean()) / bg.std()
    pct = float(np.mean(bg < thetas.mean())) * 100
    flag = " ◀ LOWER than random" if thetas.mean() < bg.mean() else \
           " ◀ HIGHER than random"
    print(f"    {ax_name:<15}  mean={thetas.mean():.2f}°  "
          f"z={z:+.2f}  pct={pct:.0f}%{flag}")

print(f"\n  φ references vs background:")
for label, ref in [('arccos(1/φ)',THETA_PHI), ('arccos(1/φ²)',THETA_PHI2),
                   ('arccos(1/φ³)',THETA_PHI3)]:
    pct = float(np.mean(bg < ref)) * 100
    print(f"    {label}={ref:.2f}°  falls at background pct={pct:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 1 SUMMARY")
print("═"*72)
print("""
Questions this day set out to answer:

1. Is CV(θ) < CV(chord)?         [Phase 1 — rotation model test]
2. Do angles cluster near φ?      [Phase 2 — φ proximity]
3. Same θ across EN and ZH?       [Phase 3 — cross-lingual]
4. Tangent more coherent?         [Phase 4 — geodesic tangent]
5. Does θ predict best_scale?     [Phase 5 — scale prediction]
6. Radial contamination size?     [Phase 6 — chord decomposition]
7. Better navigation?             [Phase 7 — practical accuracy test]
8. θ vs background distribution?  [Phase 8 — are semantic θ special?]

Record observations in second_expedition/expedition_log.md
""")
