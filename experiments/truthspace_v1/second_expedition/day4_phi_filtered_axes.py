"""
SECOND EXPEDITION — DAY 4
=========================
φ-Filtered Axes: Does Restricting to n=1 Pairs Build Better Axes?

Day 3 established:
  - φ-level measures COHERENCE, not navigability
  - n=1 pairs have coherent chord directions (mean axis generalizes)
  - n≥3 pairs scatter (mean axis fails to generalize)

Day 4 hypothesis: If we build the gender axis using ONLY n=1 pairs (cos∈[0.55,0.68]),
the resulting axis should be more coherent and generalize better to held-out pairs,
including cross-lingual transfer.

The first expedition used ALL available pairs for axis building (including n≥3 pairs
like ram/ewe, cock/hen, his/her). These lower-coherence pairs may DEGRADE the mean axis
by pulling it away from the true n=1 direction.

Phases:
  1. φ-level audit of all available EN and ZH gender pairs
  2. φ-filtered axis (n=1 only) vs unfiltered axis — pairwise coherence comparison
  3. Cross-validation: held-out pair navigation accuracy for filtered vs unfiltered
  4. Cross-lingual transfer: EN→ZH using φ-filtered axes (both source and target principles)
  5. Axis coherence for size and sentiment — are n=1 pairs also better for these?
  6. The chain navigation test: A→B→C→D... does it stay on track?

Script: second_expedition/day4_phi_filtered_axes.py
"""

import torch, numpy as np

print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n = (W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
V   = len(W_E)
print(f"  shape={W_E.shape}")

EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(V)], dtype=bool)

ZH_MASK = np.array([
    any('\u4e00' <= c <= '\u9fff' for c in tok.decode([i]).strip())
    for i in range(V)], dtype=bool)

PHI   = (1 + 5**0.5) / 2
PHI_L = {n: 1.0/PHI**n for n in range(0, 10)}

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

def phi_n(c):
    if c <= 0: return None
    return round(-np.log(max(c, 1e-9)) / np.log(PHI))

def pair_cos(src, tgt):
    es, _ = get_emb(src); et, _ = get_emb(tgt)
    if es is None or et is None: return None
    return float(np.dot(normed(es), normed(et)))

def build_axis(pairs, get_fn=get_emb, label=""):
    """Build mean tangent axis from pairs. Returns (axis, tangents, cosines)."""
    tangents = []; cosines = []
    for s, t in pairs:
        es, _ = get_fn(s); et, _ = get_fn(t)
        if es is None or et is None: continue
        en_s = normed(es); en_t = normed(et)
        c = float(np.dot(en_s, en_t))
        sin_th = float(np.sqrt(max(0, 1-c**2)))
        if sin_th < 1e-8: continue
        tangent = (en_t - c*en_s) / sin_th
        tangents.append(tangent); cosines.append(c)
    if not tangents:
        return None, [], []
    axis = normed(np.mean(tangents, axis=0))
    pairwise = []
    for i in range(len(tangents)):
        for j in range(i+1, len(tangents)):
            pairwise.append(float(np.dot(tangents[i], tangents[j])))
    coherence = np.mean(pairwise) if pairwise else 0.0
    print(f"  {label}: {len(tangents)} pairs  coherence(mean pairwise cos)={coherence:.4f}"
          f"  mean_n={np.mean([phi_n(c) for c in cosines if phi_n(c) is not None]):.2f}")
    return axis, tangents, cosines

def test_axis(axis, theta_deg, pairs, mask, label="", get_fn=get_emb):
    """Test navigation accuracy for given axis and angle."""
    th = np.radians(theta_deg)
    hits = 0; n = 0; details = []
    for src, tgt in pairs:
        es, _ = get_fn(src)
        if es is None: continue
        n += 1
        pred = np.cos(th) * normed(es) + np.sin(th) * axis
        w, sim, _ = nn_ret(pred, source_ids(src), mask)
        ok = (w == tgt)
        if ok: hits += 1
        details.append((src, tgt, w, ok))
    return hits, n, details

def best_theta(axis, pairs, mask, get_fn=get_emb):
    """Find the optimal navigation angle for an axis over pairs."""
    best_th, best_acc = 10.0, 0
    for th_deg in np.linspace(5, 70, 130):
        th = np.radians(th_deg)
        acc = 0
        for src, tgt in pairs:
            es, _ = get_fn(src)
            if es is None: continue
            pred = np.cos(th)*normed(es) + np.sin(th)*axis
            w, _, _ = nn_ret(pred, source_ids(src), mask)
            if w == tgt: acc += 1
        if acc > best_acc: best_acc = acc; best_th = th_deg
    return best_th, best_acc

# ═══════════════════════════════════════════════════════════════════════════════
# FULL GENDER PAIR LIST (EN and ZH) from first expedition + day 2
# ═══════════════════════════════════════════════════════════════════════════════
EN_GENDER_ALL = [
    ('man','woman'),('king','queen'),('father','mother'),('son','daughter'),
    ('boy','girl'),('husband','wife'),('uncle','aunt'),('prince','princess'),
    ('brother','sister'),('actor','actress'),('hero','heroine'),('waiter','waitress'),
    ('monk','nun'),('wizard','witch'),('lord','lady'),('god','goddess'),
    ('male','female'),('he','she'),('his','her'),('him','her'),
    ('grandfather','grandfather'),('nephew','niece'),('groom','bride'),
    ('bull','cow'),('cock','hen'),('ram','ewe'),
]
# Restore correct grandmother
EN_GENDER_ALL[20] = ('grandfather', 'grandmother')

ZH_GENDER_ALL = [
    ('男人','女人'),('国王','王后'),('父亲','母亲'),('儿子','女儿'),
    ('男孩','女孩'),('丈夫','妻子'),('叔叔','阿姨'),('王子','公主'),
    ('兄弟','姐妹'),('男演员','女演员'),('英雄','女英雄'),('服务员','女服务员'),
    ('男孩','女孩'),('巫师','女巫'),('男神','女神'),('雄性','雌性'),
    ('他','她'),('他的','她的'),
]

SIZE_ALL = [
    ('big','small'),('large','tiny'),('huge','little'),('tall','short'),
    ('heavy','light'),('strong','weak'),('hot','cold'),('fast','slow'),
    ('hard','soft'),('loud','quiet'),('rich','poor'),('old','young'),
    ('high','low'),('full','empty'),('bright','dim'),('warm','cool'),
]

SENT_ALL = [
    ('good','bad'),('happy','sad'),('love','hate'),('beautiful','ugly'),
    ('right','wrong'),('best','worst'),('kind','cruel'),('honest','dishonest'),
    ('wise','foolish'),('gentle','harsh'),('success','failure'),('hope','despair'),
    ('truth','lie'),('friend','enemy'),('hero','villain'),('life','death'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: φ-level audit of all pairs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — φ-Level Audit of All Available Pairs")
print("  Classifying each pair by φ-level before building axes")
print("═"*72)

def audit_pairs(pairs, name, get_fn=get_emb):
    level_groups = {}
    for s, t in pairs:
        c = pair_cos(s, t) if get_fn == get_emb else pair_cos_fn(s, t, get_fn)
        if c is None: continue
        n = phi_n(c)
        if n is None: n = 99
        level_groups.setdefault(n, []).append((s, t, c))
    print(f"\n  {name}:")
    for n in sorted(level_groups.keys()):
        grp = level_groups[n]
        cos_vals = [c for _,_,c in grp]
        pairs_str = ', '.join(f"{s}/{t}" for s,t,_ in grp[:5])
        extra = f"...+{len(grp)-5}" if len(grp)>5 else ""
        print(f"    n={n} (1/φⁿ={PHI_L.get(n,0):.4f}): {len(grp):>3} pairs "
              f" mean_cos={np.mean(cos_vals):.4f}  [{pairs_str}{extra}]")
    n1 = [p for p in level_groups.get(1, [])]
    n2 = [p for p in level_groups.get(2, [])]
    return n1, n2, level_groups

def pair_cos_fn(s, t, gf):
    es, _ = gf(s); et, _ = gf(t)
    if es is None or et is None: return None
    return float(np.dot(normed(es), normed(et)))

en_n1, en_n2, en_levels = audit_pairs(EN_GENDER_ALL, "EN Gender")

# ZH — test single-token availability
def get_zh(word):
    ids = tok(word, add_special_tokens=False)['input_ids']
    if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

zh_valid = [(s,t) for s,t in ZH_GENDER_ALL
            if get_zh(s)[0] is not None and get_zh(t)[0] is not None]

def pair_cos_zh(s, t):
    es, _ = get_zh(s); et, _ = get_zh(t)
    if es is None or et is None: return None
    return float(np.dot(normed(es), normed(et)))

level_groups_zh = {}
for s, t in zh_valid:
    c = pair_cos_zh(s, t)
    if c is None: continue
    n = phi_n(c)
    if n is None: n = 99
    level_groups_zh.setdefault(n, []).append((s, t, c))

print(f"\n  ZH Gender ({len(zh_valid)} single-token pairs):")
for n in sorted(level_groups_zh.keys()):
    grp = level_groups_zh[n]
    cos_vals = [c for _,_,c in grp]
    pairs_str = ', '.join(f"{s}/{t}" for s,t,_ in grp[:4])
    print(f"    n={n} (1/φⁿ={PHI_L.get(n,0):.4f}): {len(grp):>3} pairs  "
          f"mean_cos={np.mean(cos_vals):.4f}  [{pairs_str}]")

zh_n1 = [(s,t) for s,t,_ in level_groups_zh.get(1, [])]
zh_n2 = [(s,t) for s,t,_ in level_groups_zh.get(2, [])]

# Size and sentiment
print()
for pairs, name in [(SIZE_ALL, "EN Size"), (SENT_ALL, "EN Sentiment")]:
    audit_pairs(pairs, name)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Axis coherence — filtered vs unfiltered
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — Axis Coherence: φ-Filtered vs Unfiltered")
print("  Building mean tangent axes and comparing pairwise coherence")
print("═"*72)

print("\n  Building EN axes:")
en_n1_pairs = [(s,t) for s,t,_ in en_n1]
en_n2_pairs = [(s,t) for s,t,_ in en_n2]
en_all_pairs = [(s,t) for n_grp in en_levels.values() for s,t,_ in n_grp]

ax_en_n1,   tan_en_n1, _ = build_axis(en_n1_pairs,  label="EN n=1 only")
ax_en_n12,  _, _         = build_axis(en_n1_pairs + en_n2_pairs, label="EN n≤2")
ax_en_all,  _, _         = build_axis(en_all_pairs,  label="EN all pairs")
ax_en_first, _, _        = build_axis(
    [p[:2] for p in EN_GENDER_ALL[:10]], label="EN first-expedition (first 10)")

# Compare axis similarity
if ax_en_n1 is not None and ax_en_all is not None:
    print(f"\n  Axis similarity (cos between axis vectors):")
    print(f"    cos(n=1, n≤2):    {float(np.dot(ax_en_n1, ax_en_n12)):.4f}")
    print(f"    cos(n=1, all):    {float(np.dot(ax_en_n1, ax_en_all)):.4f}")
    print(f"    cos(n=1, first10):{float(np.dot(ax_en_n1, ax_en_first)):.4f}")
    print(f"    cos(n≤2, all):    {float(np.dot(ax_en_n12, ax_en_all)):.4f}")

print("\n  Building ZH axes:")
if zh_n1:
    ax_zh_n1,  _, _ = build_axis(zh_n1, get_fn=get_zh, label="ZH n=1 only")
    ax_zh_all, _, _ = build_axis(zh_valid, get_fn=get_zh, label="ZH all pairs")
else:
    ax_zh_n1 = ax_zh_all = None
    print("  (No ZH n=1 pairs available)")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Cross-validation — held-out pair navigation
# 5-fold cross-validation on EN n=1 pairs: train on 4/5, test on 1/5
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — Cross-Validation on EN n=1 Pairs")
print("  5-fold CV: train on 4/5, test on 1/5")
print("  Comparing: n=1 axis vs all-pairs axis vs first-expedition axis")
print("═"*72)

# Only use n=1 pairs for CV
cv_pairs = en_n1_pairs.copy()
rng = np.random.default_rng(42)
rng.shuffle(cv_pairs)
n_cv = len(cv_pairs)
k = 5
fold_size = max(1, n_cv // k)

cv_results = {'n1_only': [], 'n12': [], 'all': []}

print(f"\n  {n_cv} n=1 pairs, {k} folds of ~{fold_size}")
print(f"\n  fold  test_pairs  n1_acc  n12_acc  all_acc")
print(f"  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*7}")

for fold in range(k):
    lo = fold * fold_size
    hi = min((fold+1) * fold_size, n_cv)
    test_p  = cv_pairs[lo:hi]
    train_p = cv_pairs[:lo] + cv_pairs[hi:]
    if not train_p or not test_p: continue

    ax_tr_n1, _, _ = build_axis(train_p, label=f"  fold{fold}_n1")
    # All-pairs axis uses all available (train + all lower-n pairs)
    ax_tr_all, _, _ = build_axis(en_all_pairs, label=f"  fold{fold}_all")
    # n≤2 axis
    ax_tr_n12, _, _ = build_axis(train_p + en_n2_pairs, label=f"  fold{fold}_n12")

    if ax_tr_n1 is None: continue

    # Find best theta for n1 axis on training set
    th_n1, _ = best_theta(ax_tr_n1, train_p, EN_MASK)

    h1, n1t, _ = test_axis(ax_tr_n1, th_n1, test_p, EN_MASK)
    h2, n2t, _ = test_axis(ax_tr_all, th_n1, test_p, EN_MASK)
    h3, n3t, _ = test_axis(ax_tr_n12, th_n1, test_p, EN_MASK)

    a1 = h1/n1t if n1t else 0; a2 = h2/n2t if n2t else 0; a3 = h3/n3t if n3t else 0
    cv_results['n1_only'].append(a1)
    cv_results['n12'].append(a3)
    cv_results['all'].append(a2)
    print(f"  {fold:>4}  {n1t:>10}  {a1:>6.0%}  {a3:>7.0%}  {a2:>7.0%}")

print(f"  {'─'*50}")
for key, vals in cv_results.items():
    if vals: print(f"  {key:<12}  mean={np.mean(vals):.1%}  std={np.std(vals):.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Cross-lingual transfer with φ-filtered axes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — Cross-Lingual Transfer: φ-Filtered Axes")
print("  EN→ZH gender transfer using various axis combinations")
print("  (Target language principle from Day 368 of first expedition)")
print("═"*72)

# Build EN and ZH axes at various φ-levels
axes_to_test = {}
if ax_en_n1 is not None:
    axes_to_test['EN_n1'] = (ax_en_n1, 'EN')
if ax_en_all is not None:
    axes_to_test['EN_all'] = (ax_en_all, 'EN')
if ax_zh_n1 is not None:
    axes_to_test['ZH_n1'] = (ax_zh_n1, 'ZH')
if ax_zh_all is not None:
    axes_to_test['ZH_all'] = (ax_zh_all, 'ZH')

# Test EN→ZH transfer: source=EN word, target=ZH word
EN_ZH_TRANSFER_PAIRS = [
    ('man','男人'),('woman','女人'),('king','国王'),('queen','王后'),
    ('father','父亲'),('mother','母亲'),('son','儿子'),('daughter','女儿'),
    ('boy','男孩'),('girl','女孩'),('husband','丈夫'),('wife','妻子'),
    ('brother','兄弟'),('sister','姐妹'),('god','男神'),('goddess','女神'),
]

# For EN→ZH, we start from EN src and try to land on ZH tgt
# pred = e_EN_src + scale * axis → NN in ZH space
print(f"\n  EN→ZH transfer (navigate from EN embedding → ZH nearest neighbor):")
print(f"  {'axis':>12}  {'scale':>6}  {'acc':>5}  details")
print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*30}")

valid_transfer = [(s,t) for s,t in EN_ZH_TRANSFER_PAIRS
                  if get_emb(s)[0] is not None and get_zh(t)[0] is not None]

for ax_name, (ax, lang) in axes_to_test.items():
    best_acc = 0; best_s_val = 0
    for s_val in np.linspace(0.1, 3.0, 58):
        acc = 0
        for en_w, zh_w in valid_transfer:
            es, idx_s = get_emb(en_w)
            if es is None: continue
            pred = es + s_val * ax
            w, _, _ = nn_ret(pred, source_ids(en_w), ZH_MASK)
            if w == zh_w or w == zh_w.strip(): acc += 1
        if acc > best_acc: best_acc = acc; best_s_val = s_val
    pct = best_acc / max(len(valid_transfer), 1)
    print(f"  {ax_name:>12}  {best_s_val:>6.3f}  {best_acc}/{len(valid_transfer)}={pct:.0%}")

# Show detailed results for best axis
if ax_zh_n1 is not None:
    print(f"\n  Detail (ZH_n1 axis, EN→ZH):")
    best_acc2 = 0; best_s2 = 0
    for s_val in np.linspace(0.1, 3.0, 58):
        acc = sum(1 for en_w,zh_w in valid_transfer
                  if get_emb(en_w)[0] is not None and
                  nn_ret(get_emb(en_w)[0]+s_val*ax_zh_n1,
                         source_ids(en_w), ZH_MASK)[0] in [zh_w, zh_w.strip()])
        if acc > best_acc2: best_acc2 = acc; best_s2 = s_val
    for en_w, zh_w in valid_transfer:
        es, _ = get_emb(en_w)
        if es is None: continue
        w, _, _ = nn_ret(es + best_s2*ax_zh_n1, source_ids(en_w), ZH_MASK)
        ok = (w == zh_w or w == zh_w.strip())
        print(f"    {en_w}→{zh_w}: {'✓' if ok else '✗'} (got: {w})")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: φ-filtered axes for size and sentiment
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — φ-Filtered Size and Sentiment Axes")
print("  For non-gender axes: which pairs are n=1 and does filtering help?")
print("═"*72)

for pairs, name, mask in [(SIZE_ALL, "Size", EN_MASK), (SENT_ALL, "Sentiment", EN_MASK)]:
    print(f"\n  {name}:")
    # Get n=1 pairs
    n1_p = []; n2_p = []; other_p = []
    for s, t in pairs:
        c = pair_cos(s, t)
        if c is None: continue
        n = phi_n(c)
        if n == 1: n1_p.append((s,t))
        elif n == 2: n2_p.append((s,t))
        else: other_p.append((s,t))
    print(f"    n=1: {len(n1_p)} pairs  n=2: {len(n2_p)} pairs  n≥3: {len(other_p)} pairs")
    if n1_p:
        print(f"    n=1 pairs: {', '.join(f'{s}/{t}' for s,t in n1_p)}")
    if not n1_p:
        print("    No n=1 pairs — cannot build filtered axis")
        continue
    ax_n1, _, _ = build_axis(n1_p, label=f"  {name}_n1")
    ax_all, _, _ = build_axis(pairs, label=f"  {name}_all")
    # Test on all pairs
    if ax_n1 is not None and ax_all is not None:
        th_n1_opt, acc_n1 = best_theta(ax_n1, n1_p, mask)
        th_all_opt, acc_all = best_theta(ax_all, pairs, mask)
        h_n1_on_all, n_all, _ = test_axis(ax_n1, th_n1_opt, pairs, mask)
        h_all_on_all, _, _    = test_axis(ax_all, th_all_opt, pairs, mask)
        h_n1_on_n1, n_n1, _  = test_axis(ax_n1, th_n1_opt, n1_p, mask)
        print(f"    n=1 axis  @ θ={th_n1_opt:.1f}°: {acc_n1}/{len(n1_p)} on n1, "
              f"{h_n1_on_all}/{n_all} on all")
        print(f"    all axis  @ θ={th_all_opt:.1f}°: {acc_all}/{len(pairs)} on all")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Chain navigation — A→B→C→D...
# Start from a word, apply the n=1 axis repeatedly, track where we go
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — Chain Navigation: A→B→C→D→...")
print("  Apply n=1 gender axis repeatedly, tracking the semantic path")
print("  Does it stay in the gender domain? Does it loop?")
print("═"*72)

if ax_en_n1 is None:
    print("  (No EN n=1 axis available)")
else:
    th_chain = np.radians(29.0)   # optimal from Day 2
    chain_seeds = ['king', 'man', 'father', 'son', 'boy', 'brother',
                   'actor', 'hero', 'good', 'large', 'dog', 'Paris']

    print(f"\n  Navigation angle: 29.0° (optimal from Day 2)")
    for seed in chain_seeds:
        es, idx_s = get_emb(seed)
        if es is None: continue
        chain = [seed]
        visited = {seed}
        for step in range(8):
            e_cur, idx_cur = get_emb(chain[-1])
            if e_cur is None: break
            pred = np.cos(th_chain)*normed(e_cur) + np.sin(th_chain)*ax_en_n1
            w, sim, _ = nn_ret(pred, source_ids(chain[-1]), EN_MASK)
            # Measure cos of consecutive pair
            e_next, _ = get_emb(w)
            c_step = None
            if e_next is not None:
                c_step = float(np.dot(normed(e_cur), normed(e_next)))
            phi_step = phi_n(c_step) if c_step is not None else None
            chain.append(w)
            if w in visited:
                chain.append("(loop)")
                break
            visited.add(w)
        # print chain with φ-levels
        cos_steps = []
        for i in range(len(chain)-1):
            if chain[i] in ['(loop)']: break
            es1, _ = get_emb(chain[i])
            es2, _ = get_emb(chain[i+1] if chain[i+1] != '(loop)' else chain[i])
            if es1 is not None and es2 is not None:
                c = float(np.dot(normed(es1), normed(es2)))
                cos_steps.append(c)
            else: cos_steps.append(None)
        chain_str = " → ".join(
            f"{w}[n={phi_n(c)}]" if c is not None else w
            for w, c in zip(chain, cos_steps + [None]))
        print(f"  {seed:>10}: {chain_str}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 4 SUMMARY")
print("═"*72)
print(f"""
Main questions for Day 4:
  Phase 1: φ-level distribution of EN and ZH gender pairs
  Phase 2: Does φ-filtering improve axis coherence?
  Phase 3: Cross-validation — does filtered axis generalize better?
  Phase 4: Cross-lingual transfer with filtered axes
  Phase 5: φ-filtering for size and sentiment
  Phase 6: Chain navigation — where does the axis lead repeatedly?

Record findings in second_expedition/expedition_log.md
""")
