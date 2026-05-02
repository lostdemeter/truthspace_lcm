"""
SECOND EXPEDITION — DAY 6
=========================
The Axis Geometry: Are Semantic Module Axes Orthogonal?

Day 5 established that the embedding space is MODULAR:
  - Each functional system (compass, kinship, boolean, numbers, rank...)
    has its own axis direction
  - Cross-domain navigation fails (block-diagonal matrix)
  - The axes appear to be domain-specific

Day 6 question: Are the semantic axes truly orthogonal to each other?
Or do they form a specific geometric lattice (e.g. at φ-level angles)?

If orthogonal: the semantic space has an independent basis of n=1 axes —
  language organizes semantic knowledge into mutually independent subspaces.

If not orthogonal: the axes form a geometric structure, possibly at
  arccos(1/φⁿ) angles to each other, creating a φ-lattice in axis space.

Phases:
  1. Build all available n=1 axes (8+ domains)
  2. Measure all inter-axis angles — angle matrix
  3. Test the φ-angle hypothesis: do inter-axis angles cluster at arccos(1/φⁿ)?
  4. SVD of the axis matrix — how many independent directions are there?
  5. Compare axis directions to known embedding PCA components
  6. Does the axis basis span a subspace? What fraction of variance do they capture?

Script: second_expedition/day6_axis_geometry.py
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

def build_axis(pairs, get_fn=get_emb):
    tangents = []
    for s, t in pairs:
        es, _ = get_fn(s); et, _ = get_fn(t)
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

# ═══════════════════════════════════════════════════════════════════════════════
# ALL DOMAIN n=1 PAIRS (from Day 5 findings, curated)
# ═══════════════════════════════════════════════════════════════════════════════
DOMAIN_PAIRS = {
    'kinship':         [('son','daughter'),('brother','sister'),('boy','girl'),
                        ('uncle','aunt'),('mother','father'),('king','queen'),
                        ('husband','wife'),('grandfather','grandmother'),
                        ('actor','actress'),('prince','princess')],
    'compass_card':    [('north','south'),('east','west')],
    'compass_diag':    [('northeast','southwest'),('northwest','southeast')],
    'compass_all':     [('north','south'),('east','west'),
                        ('northeast','southwest'),('northwest','southeast'),
                        ('above','below'),('inside','outside')],
    'calendar_dow':    [('Sunday','Saturday'),('Monday','Friday')],
    'calendar_tod':    [('morning','evening')],
    'calendar_season': [('summer','winter')],
    'calendar_all':    [('Sunday','Saturday'),('Monday','Friday'),
                        ('morning','evening'),('summer','winter'),
                        ('yesterday','tomorrow'),('January','July')],
    'boolean':         [('true','false'),('positive','negative'),
                        ('correct','incorrect'),('valid','invalid')],
    'numbers':         [('two','three'),('second','third'),('hundred','thousand')],
    'rank':            [('senior','junior'),('major','minor'),
                        ('strong','weak'),('high','low')],
    'nation_lang':     [('Korea','Korean'),('China','Chinese'),('Japan','Japanese'),
                        ('Russia','Russian'),('France','French'),('Germany','German'),
                        ('Italy','Italian'),('Greece','Greek'),('Spain','Spanish')],
    'time_seq':        [('month','week'),('year','month')],
    'encode_decode':   [('encode','decode')],
    'early_late':      [('early','late')],
    # Additional pairs to test
    'size_adj':        [('big','small'),('large','tiny'),('tall','short'),
                        ('heavy','light'),('fast','slow')],
    'sentiment_core':  [('good','bad'),('love','hate'),('best','worst'),
                        ('happy','sad'),('right','wrong')],
    'color_core':      [('red','blue'),('black','white'),('red','green'),
                        ('dark','light')],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build all domain axes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — Build All Domain Axes")
print("  Building mean tangent axis for every domain")
print("═"*72)

axes = {}
print(f"\n  {'domain':<20}  {'pairs':>6}  {'coh':>7}  {'mean_n':>7}  axis_norm")
print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*10}")

for domain, pairs in DOMAIN_PAIRS.items():
    cos_vals = [pair_cos(s,t) for s,t in pairs if pair_cos(s,t) is not None]
    mean_n = np.mean([phi_n(c) for c in cos_vals if phi_n(c) is not None]) if cos_vals else None
    ax, coh, n = build_axis(pairs)
    if ax is None: continue
    axes[domain] = ax
    print(f"  {domain:<20}  {n:>6}  {coh:>7.4f}  {mean_n:>7.2f}  {np.linalg.norm(ax):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Inter-axis angle matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — Inter-Axis Angle Matrix")
print("  All pairwise angles between domain axes")
print("═"*72)

# Focus on the main semantic domains (not sub-splits)
MAIN_DOMAINS = ['kinship', 'compass_all', 'boolean', 'numbers', 'rank',
                'nation_lang', 'calendar_all', 'time_seq',
                'size_adj', 'sentiment_core', 'color_core']
main_axes = {d: axes[d] for d in MAIN_DOMAINS if d in axes}

print(f"\n  Axis-to-axis angles (degrees):")
dom_list = sorted(main_axes.keys())
# Header
header = f"  {'':>18}"
for d in dom_list: header += f"  {d[:10]:>10}"
print(header)
print(f"  {'─'*18}" + "  " + "  ".join(["─"*10]*len(dom_list)))

angle_matrix = np.zeros((len(dom_list), len(dom_list)))
for i, di in enumerate(dom_list):
    row = f"  {di:<18}"
    for j, dj in enumerate(dom_list):
        c = float(np.dot(main_axes[di], main_axes[dj]))
        c = np.clip(c, -1, 1)
        angle = np.degrees(np.arccos(abs(c)))  # use abs for unsigned angle
        angle_matrix[i][j] = angle
        if i == j:
            row += f"  {'---':>10}"
        else:
            # Mark if close to any φ-angle
            phi_mark = ""
            for pn, pth in PHI_ANGLES.items():
                if abs(angle - pth) < 3.0: phi_mark = f"≈φ{pn}"; break
            row += f"  {angle:>7.1f}° {phi_mark[:2]:>2}"
    print(row)

# Statistical summary of off-diagonal angles
off_diag = [angle_matrix[i][j] for i in range(len(dom_list)) 
            for j in range(len(dom_list)) if i != j]
print(f"\n  Off-diagonal angle statistics:")
print(f"    mean = {np.mean(off_diag):.1f}°")
print(f"    std  = {np.std(off_diag):.1f}°")
print(f"    min  = {np.min(off_diag):.1f}°")
print(f"    max  = {np.max(off_diag):.1f}°")
print(f"    frac near 90°±10° = {sum(1 for a in off_diag if abs(a-90)<10)/len(off_diag):.0%}")
print(f"    frac near 60°±10° = {sum(1 for a in off_diag if abs(a-60)<10)/len(off_diag):.0%}")

# Check which φ-angle is the most common
for pn, pth in sorted(PHI_ANGLES.items()):
    frac = sum(1 for a in off_diag if abs(a-pth) < 5) / len(off_diag)
    if frac > 0.05:
        print(f"    frac near arccos(1/φ^{pn})={pth:.1f}°±5° = {frac:.0%}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: φ-angle test on inter-axis angles
# Does the axis-to-axis angle distribution cluster at φ-levels?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — φ-Angle Test: Do Inter-Axis Angles Follow φ-Levels?")
print("  Compare observed axis-to-axis angles to arccos(1/φⁿ)")
print("═"*72)

all_domain_pairs = [(dom_list[i], dom_list[j], angle_matrix[i][j])
                    for i in range(len(dom_list))
                    for j in range(i+1, len(dom_list))]
all_domain_pairs.sort(key=lambda x: x[2])

print(f"\n  All inter-axis angles (sorted), with nearest φ-level:")
print(f"  {'domain_A':<20}  {'domain_B':<20}  {'angle':>7}  {'nearest φ-angle':>18}  Δ")
print(f"  {'─'*20}  {'─'*20}  {'─'*7}  {'─'*18}  {'─'*8}")
for da, db, angle in all_domain_pairs:
    best_n, best_delta = None, 100
    for pn, pth in PHI_ANGLES.items():
        if abs(angle - pth) < best_delta:
            best_delta = abs(angle - pth); best_n = pn
    phi_str = f"arccos(1/φ^{best_n})={PHI_ANGLES[best_n]:.1f}°"
    flag = " ◀" if best_delta < 3.0 else ""
    print(f"  {da:<20}  {db:<20}  {angle:>7.1f}°  {phi_str:>18}  {best_delta:>6.2f}°{flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: SVD of the axis matrix
# How many independent directions do the n=1 axes span?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — SVD of Axis Matrix")
print("  How many independent dimensions do the semantic axes span?")
print("═"*72)

# Stack all main axes into a matrix
ax_matrix = np.array([main_axes[d] for d in dom_list if d in main_axes])  # (n_axes, 1536)
print(f"\n  Axis matrix: {ax_matrix.shape[0]} axes × {ax_matrix.shape[1]} dims")

# SVD to find effective rank
U, S, Vt = np.linalg.svd(ax_matrix, full_matrices=False)
S_norm = S / S.sum()
cum_var = np.cumsum(S_norm)

print(f"\n  Singular values (normalized):")
for i, (s, cs) in enumerate(zip(S_norm, cum_var)):
    bar = "█" * int(s * 100)
    print(f"    PC{i+1:>2}: {s:.4f}  cumvar={cs:.4f}  {bar}")

print(f"\n  Effective rank summary:")
for thresh in [0.50, 0.75, 0.90, 0.95, 0.99]:
    rank = int(np.searchsorted(cum_var, thresh)) + 1
    print(f"    {thresh:.0%} variance captured by {rank} PCs")

# Top singular vectors — what words do they pick out?
print(f"\n  What concepts do the top principal axes represent?")
print(f"  (Top words in embedding space for each principal direction)")
for i in range(min(4, len(Vt))):
    direction = Vt[i]
    sims = W_n @ direction.astype(np.float32)
    top_ids = np.argsort(sims)[-10:][::-1]
    top_words = [tok.decode([j]).strip() for j in top_ids if
                 tok.decode([j]).strip().isalpha() and tok.decode([j]).strip().isascii()
                 and len(tok.decode([j]).strip()) >= 2][:6]
    neg_ids = np.argsort(sims)[:10]
    neg_words = [tok.decode([j]).strip() for j in neg_ids if
                 tok.decode([j]).strip().isalpha() and tok.decode([j]).strip().isascii()
                 and len(tok.decode([j]).strip()) >= 2][:6]
    print(f"  PC{i+1} (var={S_norm[i]:.3f}):")
    print(f"    +: {', '.join(top_words)}")
    print(f"    -: {', '.join(neg_words)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Compass sub-axis structure
# north/south and east/west are both compass but are THEY orthogonal?
# Same for calendar subsystems
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — Within-Domain Sub-Axis Structure")
print("  Are sub-axes within the same domain also orthogonal?")
print("═"*72)

# Compass: cardinal vs diagonal vs vertical
ax_ns = build_axis([('north','south')])[0]
ax_ew = build_axis([('east','west')])[0]
ax_nwse = build_axis([('northwest','southeast'),('northeast','southwest')])[0]
ax_updn = build_axis([('above','below')])[0]
ax_inout = build_axis([('inside','outside')])[0]

print(f"\n  Compass sub-axes:")
compass_axes = [('N/S', ax_ns), ('E/W', ax_ew), ('NW/SE+NE/SW', ax_nwse),
                ('above/below', ax_updn), ('inside/outside', ax_inout)]
for i, (n1, a1) in enumerate(compass_axes):
    if a1 is None: continue
    for j, (n2, a2) in enumerate(compass_axes):
        if j <= i or a2 is None: continue
        c = float(np.dot(a1, a2))
        angle = np.degrees(np.arccos(np.clip(abs(c), 0, 1)))
        print(f"    {n1:>20} · {n2:<20} = {angle:.1f}°")

# Calendar subsystems
ax_dow  = build_axis([('Sunday','Saturday'),('Monday','Friday')])[0]
ax_tod  = build_axis([('morning','evening')])[0]
ax_seas = build_axis([('summer','winter')])[0]
ax_yest = build_axis([('yesterday','tomorrow')])[0]

print(f"\n  Calendar sub-axes:")
cal_axes = [('days_of_week', ax_dow), ('time_of_day', ax_tod),
            ('seasons', ax_seas), ('yesterday/tomorrow', ax_yest)]
for i, (n1, a1) in enumerate(cal_axes):
    if a1 is None: continue
    for j, (n2, a2) in enumerate(cal_axes):
        if j <= i or a2 is None: continue
        c = float(np.dot(a1, a2))
        angle = np.degrees(np.arccos(np.clip(abs(c), 0, 1)))
        print(f"    {n1:>20} · {n2:<20} = {angle:.1f}°")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: The anti-axis — does each axis have a natural opposite?
# If we build the REVERSE axis (target→source instead of source→target),
# is it the negation of the forward axis, or a different direction?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — Axis Directionality: Forward vs Reverse")
print("  Is the A→B axis the negation of the B→A axis?")
print("  (Tests whether the axes are truly oriented or just directional)")
print("═"*72)

test_domains = {
    'kinship_fwd':  [('son','daughter'),('brother','sister'),('boy','girl'),
                     ('uncle','aunt'),('mother','father'),('king','queen')],
    'kinship_rev':  [('daughter','son'),('sister','brother'),('girl','boy'),
                     ('aunt','uncle'),('father','mother'),('queen','king')],
    'compass_fwd':  [('north','south'),('east','west')],
    'compass_rev':  [('south','north'),('west','east')],
    'boolean_fwd':  [('true','false'),('positive','negative')],
    'boolean_rev':  [('false','true'),('negative','positive')],
    'numbers_fwd':  [('two','three'),('second','third'),('hundred','thousand')],
    'numbers_rev':  [('three','two'),('third','second'),('thousand','hundred')],
}

print(f"\n  {'domain':<20}  axis_dot  angle  interpretation")
print(f"  {'─'*20}  {'─'*8}  {'─'*7}  {'─'*30}")

fwd_axes = {}
for d, pairs in test_domains.items():
    ax, _, _ = build_axis(pairs)
    if ax is not None: fwd_axes[d] = ax

for base in ['kinship', 'compass', 'boolean', 'numbers']:
    fwd_key = f"{base}_fwd"; rev_key = f"{base}_rev"
    if fwd_key in fwd_axes and rev_key in fwd_axes:
        af = fwd_axes[fwd_key]; ar = fwd_axes[rev_key]
        c = float(np.dot(af, ar))
        angle = np.degrees(np.arccos(np.clip(c, -1, 1)))
        if c < -0.9: interp = "near-perfect negation"
        elif c < -0.5: interp = "partial negation"
        elif abs(c) < 0.3: interp = "near-orthogonal (different direction)"
        else: interp = "same direction (not antisymmetric)"
        print(f"  {base:<20}  {c:>8.4f}  {angle:>6.1f}°  {interp}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 6 SUMMARY")
print("═"*72)
print("""
Day 6 tests the geometric structure of semantic axes:
  Phase 1: Build all available n=1 domain axes
  Phase 2: Inter-axis angle matrix — are they orthogonal?
  Phase 3: φ-angle test — do axis-to-axis angles follow φ-levels?
  Phase 4: SVD of axis matrix — effective rank and principal components
  Phase 5: Within-domain sub-axes — are N/S and E/W orthogonal?
  Phase 6: Axis directionality — is forward axis = negation of reverse axis?

Record in second_expedition/expedition_log.md
""")
