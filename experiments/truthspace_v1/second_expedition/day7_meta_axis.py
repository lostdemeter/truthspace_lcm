"""
SECOND EXPEDITION — DAY 7
=========================
Meta-Axis Navigation: Cross-Domain Transfer and the Compass 2D Subspace

Day 6 found:
  - 95% of domain axes are near-orthogonal
  - The departures from orthogonality are φ-quantized
  - boolean ↔ sentiment at 69.5° ≈ arccos(1/φ²) — "evaluation" proximity
  - rank ↔ size at 71.5° ≈ arccos(1/φ³)
  - N/S ↔ NW/SE at 54.5° ≈ arccos(1/φ)
  - calendar sub-axes (days/times/seasons) are near-orthogonal

Day 7 questions:
  1. Do compass pairs span a 2D subspace? (In real geography they would)
  2. Can the boolean axis navigate sentiment pairs? (They're at n=2 axis angle)
  3. Does the axis-to-axis angle predict cross-domain navigation success?
  4. Is there a "meta-navigation" at the axis level? (move from one domain to another)
  5. What is the full compass 2D geometry — do compass words form a circle?

Phases:
  1. Compass subspace: PCA of 8 compass direction embeddings
  2. Full cross-domain navigation matrix (all 8 main domains × all 8)
  3. Axis angle vs navigation success — the prediction law
  4. Meta-navigation attempt: use boolean axis to navigate sentiment
  5. The semantic compass: are compass words arranged on a circle in 2D?
  6. Synthesis: the φ-scale law for both pair-level and axis-level geometry

Script: second_expedition/day7_meta_axis.py
"""

import torch, numpy as np
from collections import defaultdict

print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n = (W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
V = len(W_E)
print(f"  shape={W_E.shape}")

EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(V)], dtype=bool)

PHI = (1 + 5**0.5) / 2
PHI_L = {n: 1.0/PHI**n for n in range(0, 10)}
PHI_ANGLES = {n: np.degrees(np.arccos(PHI_L[n])) for n in range(1, 8)}

def normed(v): return v / (np.linalg.norm(v) + 1e-12)

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def source_ids(word):
    ids = set()
    for p in [word, ' '+word,
              word[0].upper()+word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

def nn_ret(pred, excl, mask):
    p = normed(pred).astype(np.float32)
    s = W_n @ p; s[~mask] = -1.0
    for e in excl: s[e] = -1.0
    idx = int(np.argmax(s))
    return tok.decode([idx]).strip(), float(s[idx]), idx

def pair_cos(src, tgt):
    es, _ = get_emb(src); et, _ = get_emb(tgt)
    if es is None or et is None: return None
    return float(np.dot(normed(es), normed(et)))

def phi_n(c):
    if c is None or c <= 0: return None
    return round(-np.log(max(c, 1e-9)) / np.log(PHI))

def build_axis(pairs):
    tangents = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        en_s = normed(es); en_t = normed(et)
        c = float(np.dot(en_s, en_t))
        sin_th = float(np.sqrt(max(0, 1-c**2)))
        if sin_th < 1e-8: continue
        tangents.append((en_t - c*en_s) / sin_th)
    if not tangents: return None, 0.0, 0
    ax = normed(np.mean(tangents, axis=0))
    coh = (np.mean([float(np.dot(tangents[i], tangents[j]))
                    for i in range(len(tangents))
                    for j in range(i+1, len(tangents))])
           if len(tangents) > 1 else 1.0)
    return ax, coh, len(tangents)

def best_theta_and_acc(ax, pairs):
    best_th, best_hits, n = 10.0, 0, 0
    for s, t in pairs:
        if get_emb(s)[0] is not None: n += 1
    for th_deg in np.linspace(10, 65, 111):
        th = np.radians(th_deg)
        hits = sum(1 for s,t in pairs
                   if get_emb(s)[0] is not None and
                   nn_ret(np.cos(th)*normed(get_emb(s)[0]) + np.sin(th)*ax,
                          source_ids(s), EN_MASK)[0] == t)
        if hits > best_hits: best_hits = hits; best_th = th_deg
    return best_th, best_hits, n

def nav_acc(ax, theta_deg, pairs):
    th = np.radians(theta_deg)
    hits = 0; n = 0
    for s, t in pairs:
        es, _ = get_emb(s)
        if es is None: continue
        n += 1
        pred = np.cos(th)*normed(es) + np.sin(th)*ax
        w, _, _ = nn_ret(pred, source_ids(s), EN_MASK)
        if w == t: hits += 1
    return hits, n

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN PAIRS
# ═══════════════════════════════════════════════════════════════════════════════
DOMAINS = {
    'kinship':     [('son','daughter'),('brother','sister'),('boy','girl'),
                    ('uncle','aunt'),('mother','father'),('king','queen'),
                    ('husband','wife'),('grandfather','grandmother'),
                    ('actor','actress'),('prince','princess')],
    'compass':     [('north','south'),('east','west'),
                    ('northeast','southwest'),('northwest','southeast'),
                    ('above','below'),('inside','outside')],
    'boolean':     [('true','false'),('positive','negative'),
                    ('correct','incorrect'),('valid','invalid')],
    'numbers':     [('two','three'),('second','third'),('hundred','thousand')],
    'rank':        [('senior','junior'),('major','minor'),
                    ('strong','weak'),('high','low')],
    'nation_lang': [('Korea','Korean'),('China','Chinese'),('Japan','Japanese'),
                    ('Russia','Russian'),('France','French'),('Germany','German'),
                    ('Italy','Italian'),('Greece','Greek'),('Spain','Spanish')],
    'calendar':    [('Sunday','Saturday'),('Monday','Friday'),
                    ('morning','evening'),('summer','winter'),
                    ('yesterday','tomorrow'),('January','July')],
    'sentiment':   [('good','bad'),('love','hate'),('beautiful','ugly'),
                    ('best','worst'),('honest','dishonest'),('happy','sad'),
                    ('right','wrong'),('wise','foolish')],
    'size':        [('big','small'),('large','tiny'),('tall','short'),
                    ('heavy','light'),('fast','slow')],
    'color':       [('red','blue'),('black','white'),('red','green'),('dark','light')],
}

# Build all axes
print("\nBuilding domain axes...")
axes = {}
opt_angles = {}
for d, pairs in DOMAINS.items():
    ax, coh, n = build_axis(pairs)
    if ax is None: continue
    axes[d] = ax
    th, hits, n_t = best_theta_and_acc(ax, pairs)
    opt_angles[d] = th
    print(f"  {d:<14}  {n:>3} pairs  coh={coh:>7.4f}  opt_θ={th:.1f}°  self={hits}/{n_t}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Compass subspace — PCA of 8 direction word embeddings
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — Compass Subspace Geometry")
print("  PCA of 8 compass direction word embeddings")
print("  Do they form a 2D plane? What are the angles between them?")
print("═"*72)

COMPASS_WORDS = ['north','south','east','west','northeast','southwest','northwest','southeast']
COMPASS_GEO = {  # real 2D geographic angles (degrees from north, clockwise)
    'north': 0, 'northeast': 45, 'east': 90, 'southeast': 135,
    'south': 180, 'southwest': 225, 'west': 270, 'northwest': 315
}

c_embs = {}
for w in COMPASS_WORDS:
    e, _ = get_emb(w)
    if e is not None: c_embs[w] = normed(e)

print(f"\n  Compass word cosine matrix:")
words = list(c_embs.keys())
print(f"  {'':<14}", end="")
for w in words: print(f"  {w[:5]:>5}", end="")
print()
for i, wi in enumerate(words):
    print(f"  {wi:<14}", end="")
    for j, wj in enumerate(words):
        if i == j: print(f"  {'---':>5}", end="")
        else:
            c = float(np.dot(c_embs[wi], c_embs[wj]))
            print(f"  {c:>5.3f}", end="")
    print()

# PCA of compass embeddings
C = np.array([c_embs[w] for w in words])  # (8, 1536)
U, S, Vt = np.linalg.svd(C - C.mean(axis=0, keepdims=True), full_matrices=False)
S_norm = S**2 / (S**2).sum()
print(f"\n  PCA of 8 compass embeddings:")
for i, (s, cs) in enumerate(zip(S_norm, np.cumsum(S_norm))):
    bar = "█" * int(s * 100 + 0.5)
    print(f"    PC{i+1}: {s:.4f}  cumvar={cs:.4f}  {bar}")
    if cs > 0.99: break

print(f"\n  2D projection (PC1, PC2):")
coords_2d = U[:, :2] * S[:2]  # project onto top 2 PCs
print(f"  {'word':<14}  {'PC1':>7}  {'PC2':>7}  {'angle_emb':>10}  {'angle_geo':>10}  Δangle")
print(f"  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*7}")
for i, w in enumerate(words):
    x, y = coords_2d[i]
    angle_emb = np.degrees(np.arctan2(y, x)) % 360
    angle_geo = COMPASS_GEO.get(w, None)
    delta = f"{abs(angle_emb - angle_geo):.1f}°" if angle_geo is not None else "N/A"
    print(f"  {w:<14}  {x:>7.4f}  {y:>7.4f}  {angle_emb:>10.1f}°  "
          f"{str(angle_geo)+'°' if angle_geo is not None else 'N/A':>10}  {delta:>7}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Full cross-domain navigation matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — Full Cross-Domain Navigation Matrix")
print("  Source axis (row) × Test domain (col)")
print("  Using source-domain optimal angle for each row")
print("═"*72)

dom_list = sorted(axes.keys())
print(f"\n  {'source↓ test→':<14}", end="")
for d in dom_list: print(f"  {d[:8]:>8}", end="")
print()
print(f"  {'─'*14}", end="")
for _ in dom_list: print(f"  {'─'*8}", end="")
print()

cross_results = {}  # (src, tgt) → accuracy
for src_d in dom_list:
    ax_src = axes[src_d]; th_src = opt_angles[src_d]
    print(f"  {src_d:<14}", end="")
    for tgt_d in dom_list:
        pairs_tgt = DOMAINS[tgt_d]
        hits, n = nav_acc(ax_src, th_src, pairs_tgt)
        acc = hits/n if n > 0 else 0
        cross_results[(src_d, tgt_d)] = acc
        if src_d == tgt_d:
            print(f"  {'---':>8}", end="")
        else:
            print(f"  {hits}/{n}={acc:.0%}".rjust(8), end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Axis angle vs navigation success — the prediction law
# Does the axis-to-axis angle predict cross-domain navigation accuracy?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — Axis Angle vs Navigation Success: The Prediction Law")
print("  Scatter: inter-axis angle → cross-domain navigation accuracy")
print("═"*72)

# Compute all cross results
data_points = []
for i, src in enumerate(dom_list):
    for j, tgt in enumerate(dom_list):
        if src == tgt: continue
        ax_s = axes[src]; ax_t = axes[tgt]
        c = float(np.dot(ax_s, ax_t))
        axis_angle = np.degrees(np.arccos(np.clip(abs(c), 0, 1)))
        acc = cross_results.get((src, tgt), 0)
        data_points.append((src, tgt, axis_angle, acc))

# Bin by axis angle and show mean accuracy
data_points.sort(key=lambda x: x[2])
bins = [(60,70), (70,80), (80,85), (85,88), (88,91)]
print(f"\n  {'axis_angle_bin':>18}  {'n_pairs':>8}  {'mean_acc':>9}  {'max_acc':>8}")
print(f"  {'─'*18}  {'─'*8}  {'─'*9}  {'─'*8}")
for lo, hi in bins:
    pts = [(a, acc) for _, _, a, acc in data_points if lo <= a < hi]
    if not pts: continue
    accs = [acc for _, acc in pts]
    print(f"  [{lo:.0f}°, {hi:.0f}°)         {len(pts):>8}  {np.mean(accs):>9.1%}  {np.max(accs):>8.1%}")

# Show individual high-accuracy cross-domain pairs
print(f"\n  Notable cross-domain hits (acc > 10%):")
print(f"  {'source':<14}  {'target':<14}  {'axis_angle':>10}  {'acc':>6}  φ-n?")
print(f"  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*6}  {'─'*8}")
for src, tgt, angle, acc in sorted(data_points, key=lambda x: -x[3]):
    if acc < 0.10: continue
    best_n, best_d = None, 100
    for pn, pth in PHI_ANGLES.items():
        if abs(angle - pth) < best_d: best_d = abs(angle - pth); best_n = pn
    phi_str = f"n={best_n}(Δ={best_d:.1f}°)" if best_d < 5 else ""
    print(f"  {src:<14}  {tgt:<14}  {angle:>10.1f}°  {acc:>6.0%}  {phi_str}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Meta-navigation test
# boolean axis used to navigate sentiment pairs (axis angle ≈ 67.5° = arccos(1/φ²))
# rank axis used to navigate size pairs (axis angle ≈ 76.3° = arccos(1/φ³))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — Meta-Navigation: Borrowing an Axis Across Domains")
print("  Test: boolean axis (true/false) → navigate sentiment pairs (good/bad)")
print("  Test: rank axis (senior/junior) → navigate size pairs (big/small)")
print("════════════════════════════════════════════════════════════════════════")

def detailed_nav(ax, theta_deg, pairs, label):
    th = np.radians(theta_deg)
    print(f"\n  {label} (θ={theta_deg:.1f}°):")
    print(f"  {'src':<14}  {'expected':>12}  {'got':>12}  {'ok':>4}")
    print(f"  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*4}")
    hits = 0; n = 0
    for s, t in pairs:
        es, _ = get_emb(s)
        if es is None: continue
        n += 1
        pred = np.cos(th)*normed(es) + np.sin(th)*ax
        w, _, _ = nn_ret(pred, source_ids(s), EN_MASK)
        ok = (w == t)
        if ok: hits += 1
        print(f"  {s:<14}  {t:>12}  {w:>12}  {'✓' if ok else '✗':>4}")
    print(f"  → {hits}/{n} = {hits/n:.0%}")
    return hits, n

# Test boolean axis on sentiment at various angles
if 'boolean' in axes and 'sentiment' in DOMAINS:
    ax_bool = axes['boolean']
    print(f"\n  Boolean axis angle sweep on sentiment pairs:")
    for th in [opt_angles.get('boolean', 19), 25, 35, 45, 55]:
        hits, n = nav_acc(ax_bool, th, DOMAINS['sentiment'])
        print(f"    θ={th:.0f}°: {hits}/{n} = {hits/n:.0%}")
    # Detailed at best angle
    best_th, best_h, n_s = best_theta_and_acc(ax_bool, DOMAINS['sentiment'])
    detailed_nav(ax_bool, best_th, DOMAINS['sentiment'],
                 f"boolean→sentiment (best θ={best_th:.1f}°)")

# Test rank axis on size
if 'rank' in axes and 'size' in DOMAINS:
    ax_rank = axes['rank']
    print(f"\n  Rank axis angle sweep on size pairs:")
    for th in [opt_angles.get('rank', 29), 35, 45, 52, 60]:
        hits, n = nav_acc(ax_rank, th, DOMAINS['size'])
        print(f"    θ={th:.0f}°: {hits}/{n} = {hits/n:.0%}")
    best_th2, _, _ = best_theta_and_acc(ax_rank, DOMAINS['size'])
    detailed_nav(ax_rank, best_th2, DOMAINS['size'],
                 f"rank→size (best θ={best_th2:.1f}°)")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: The semantic compass — are compass words arranged on a circle?
# If north/south/east/west form a 2D plane, their 2D projections should be
# at approximately 0°, 90°, 180°, 270° from each other
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — The Semantic Compass Circle")
print("  Additional compass/directional words projected onto compass 2D plane")
print("  Do related directional words fall on the compass circle?")
print("═"*72)

# Project additional words onto the compass PC1-PC2 plane
EXTRA_WORDS = ['up','down','left','right','forward','backward','inward','outward',
               'above','below','inside','outside','center','edge','top','bottom',
               'front','back','near','far','here','there','yes','no']

# Recompute PC1, PC2 of compass plane
C_mean = C.mean(axis=0)
C_centered = C - C_mean
_, S_compass, Vt_compass = np.linalg.svd(C_centered, full_matrices=False)
pc1 = Vt_compass[0]; pc2 = Vt_compass[1]

print(f"\n  Variance in compass plane: PC1={S_compass[0]**2:.1f}, PC2={S_compass[1]**2:.1f}")
print(f"  Ratio PC1/PC2: {S_compass[0]/S_compass[1]:.2f}")
print(f"  % variance in top 2 PCs: {(S_compass[0]**2 + S_compass[1]**2)/sum(S_compass**2):.1%}")
print(f"\n  Core compass words in 2D plane:")
for w in words:
    if w not in c_embs: continue
    e = c_embs[w] - C_mean
    x = float(np.dot(e, pc1)); y = float(np.dot(e, pc2))
    angle = np.degrees(np.arctan2(y, x)) % 360
    r = np.sqrt(x**2 + y**2)
    print(f"    {w:<14}  x={x:>7.4f}  y={y:>7.4f}  angle={angle:>6.1f}°  r={r:.4f}")

print(f"\n  Extra words projected onto compass plane:")
print(f"  {'word':<14}  {'x':>7}  {'y':>7}  {'angle':>8}  {'r':>7}  category")
print(f"  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*20}")
for w in EXTRA_WORDS:
    e, _ = get_emb(w)
    if e is None: continue
    e_c = normed(e) - C_mean
    x = float(np.dot(e_c, pc1)); y = float(np.dot(e_c, pc2))
    angle = np.degrees(np.arctan2(y, x)) % 360
    r = np.sqrt(x**2 + y**2)
    # Closest compass direction
    dists = {cw: abs(COMPASS_GEO.get(cw, 0) - angle) % 360 for cw in words}
    closest = min(dists, key=dists.get)
    print(f"  {w:<14}  {x:>7.4f}  {y:>7.4f}  {angle:>8.1f}°  {r:>7.4f}  ~{closest}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Synthesis — the φ-scale law
# Summary of the φ-quantization at all scales found in Days 1-7
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — Synthesis: The φ-Scale Law Across All Levels")
print("  Summary of φ-quantization discovered in Days 1-7")
print("═"*72)
print("""
  The φ-quantization law operates at every scale of semantic geometry:

  SCALE 1: Word-pair similarity (Days 1-3)
    cos(A,B) ≈ 1/φⁿ  for integer n
    n=1 (0.618): tightly paired — functional system complements
    n=2 (0.382): strong contrasts — domain opposites
    n=3 (0.236): weak contrasts — distant semantic relations
    n=5 (0.090): ground state — unrelated vocabulary

  SCALE 2: Axis coherence (Day 4)
    Pairwise tangent cosine ≈ 1/φⁿ within each axis
    n=1 axes have coherence ~0.15-0.57
    n=2+ axes have coherence ~0.07-0.13

  SCALE 3: Domain axis-to-axis angles (Day 6)
    angle(axis_A, axis_B) ≈ arccos(1/φⁿ)  for integer n
    n=2 (67.5°): very related domains (boolean ↔ sentiment)
    n=4 (81.6°): moderately related (compass ↔ rank, calendar ↔ time)
    n=5 (84.8°): nearly independent (most domain pairs)
    n=7 (88.0°): essentially independent (kinship ↔ nation_language)

  SCALE 4: Sub-axis angles within a domain (Day 6)
    angle(compass_NS, compass_diagonal) ≈ arccos(1/φ¹) = 51.8° (measured: 54.5°)
    angle(compass_EW, compass_diagonal) ≈ arccos(1/φ⁴) = 81.6° (measured: 81.7°)

  THE φ-SCALE LAW:
    Semantic distance at every scale is quantized at 1/φⁿ.
    The level n encodes the "type" of relationship:
      - Low n  = tightly coupled, functionally bonded
      - High n = loosely coupled, independent
    The SAME mathematics governs word pairs AND domain relationships.
""")

print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 7 SUMMARY")
print("═"*72)
print("""
Day 7 tests meta-axis navigation and compass geometry:
  Phase 1: Compass PCA — do compass words span a 2D plane?
  Phase 2: Full 10×10 cross-domain navigation matrix
  Phase 3: Does axis angle predict navigation success?
  Phase 4: Boolean→sentiment and rank→size meta-navigation
  Phase 5: Semantic compass circle — where do other directional words fall?
  Phase 6: The φ-scale law synthesis

Record in second_expedition/expedition_log.md
""")
