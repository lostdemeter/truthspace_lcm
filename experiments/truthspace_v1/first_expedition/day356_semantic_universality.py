import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and len(w) >= 1: RELAXED_MASK[i] = True

EN_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and w.isalpha() and w.isascii() and len(w) >= 2: EN_MASK[i] = True

ZH_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and any('\u4e00' <= c <= '\u9fff' for c in w): ZH_MASK[i] = True

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def get_emb(word, zh=False):
    if zh:
        ids = tok(word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
        return None, None
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def source_ids(word, zh=False):
    ids = set()
    if zh:
        tks = tok(word, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
        return ids
    for p in [word, ' '+word, word[0].upper()+word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def nn_ret_top(pred_emb, excl_ids, mask, top_k=10):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    top = np.argsort(sims)[::-1][:top_k*3]
    out = []
    for idx in top:
        if len(out) >= top_k: break
        out.append((tok.decode([int(idx)]).strip(), float(sims[idx]), int(idx)))
    return out

def build_axis(pairs, zh=False):
    chords = []; usable = []; skipped = []
    for s, t in pairs:
        es, _ = get_emb(s, zh); et, _ = get_emb(t, zh)
        if es is None or et is None: skipped.append((s,t)); continue
        chords.append(et - es); usable.append((s,t))
    if not chords: return None, usable, skipped
    return normed(np.mean(chords, axis=0)), usable, skipped

def best_scale(ax_dir, pairs, mask, zh=False):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = 0
        for sr, tg in pairs:
            es, _ = get_emb(sr, zh)
            if es is None: continue
            w, _, _ = nn_ret(es + s*ax_dir, source_ids(sr, zh), mask)
            if w == tg: c += 1
        if c > best_a: best_a=c; best_s=s
    return best_s

def eval_axis(ax_dir, s, pairs, mask, zh=False):
    hits = 0; n = 0; details = []
    for src, tgt in pairs:
        es, _ = get_emb(src, zh)
        if es is None: continue
        n += 1
        pred = es + s * ax_dir
        w, sim, _ = nn_ret(pred, source_ids(src, zh), mask)
        ok = (w == tgt)
        if ok: hits += 1
        details.append((src, tgt, w, ok, sim))
    return hits, n, details

def axis_coherence(pairs, zh=False):
    chords = []
    for s, t in pairs:
        es, _ = get_emb(s, zh); et, _ = get_emb(t, zh)
        if es is None or et is None: continue
        chords.append(normed(et - es))
    if len(chords) < 2: return 0.0
    n = len(chords)
    total = sum(float(np.dot(chords[i].astype(np.float32), chords[j].astype(np.float32)))
                for i in range(n) for j in range(n) if i != j)
    return total / (n * (n-1))

# ====================================================================
# DATASETS
# ====================================================================

# English semantic axes (non-morphological)
EN_SIZE = [  # small → big antonym pairs
    ('small', 'big'), ('little', 'large'), ('tiny', 'huge'),
    ('narrow', 'wide'), ('short', 'tall'), ('shallow', 'deep'),
    ('thin', 'thick'), ('weak', 'strong'), ('slow', 'fast'),
    ('cold', 'hot'),
]
EN_SENTIMENT = [  # negative → positive
    ('bad', 'good'), ('ugly', 'beautiful'), ('hate', 'love'),
    ('sad', 'happy'), ('dark', 'bright'), ('wrong', 'right'),
    ('evil', 'good'), ('poor', 'rich'), ('sick', 'healthy'),
    ('dirty', 'clean'),
]
EN_AGE = [
    ('young', 'old'), ('new', 'ancient'), ('fresh', 'stale'),
    ('baby', 'adult'), ('child', 'elder'),
]

# Chinese semantic axes (parallel to English)
ZH_SIZE = [
    ('小', '大'),     # small → big
    ('窄', '宽'),     # narrow → wide
    ('短', '长'),     # short → long
    ('浅', '深'),     # shallow → deep
    ('薄', '厚'),     # thin → thick
    ('弱', '强'),     # weak → strong
    ('慢', '快'),     # slow → fast
    ('冷', '热'),     # cold → hot
]
ZH_SENTIMENT = [
    ('坏', '好'),     # bad → good
    ('丑', '美'),     # ugly → beautiful
    ('恨', '爱'),     # hate → love
    ('悲', '喜'),     # sad → happy
    ('暗', '亮'),     # dark → bright
    ('错', '对'),     # wrong → right
    ('穷', '富'),     # poor → rich
]
ZH_AGE = [
    ('年轻', '年老'),  # young → old
    ('新', '旧'),     # new → old
    ('小', '大'),     # young → old (size/age dual)
]

# English morphological axes (for comparison as CONTROL — expect NOT universal)
EN_PLURAL = [
    ('cat','cats'), ('dog','dogs'), ('house','houses'), ('car','cars'),
    ('tree','trees'), ('book','books'), ('bird','birds'), ('door','doors'),
]
ZH_PLURAL_EQUIV = [  # In Chinese: 猫们/猫 — but 们 suffix, mostly single-char
    # Chinese doesn't grammaticalise plural with a single morpheme the same way
    # Use quantifier constructions that are conceptually plural:
    # Actually, Chinese has 们 for animate nouns
    ('他', '他们'),   # he → they
    ('她', '她们'),   # she → they (fem)
    ('我', '我们'),   # I → we
    ('你', '你们'),   # you → you (plural)
]

# English gender axis (from Day 355 — the UNIVERSAL baseline)
EN_GENDER = [
    ('king','queen'), ('man','woman'), ('boy','girl'),
    ('father','mother'), ('son','daughter'), ('husband','wife'),
    ('uncle','aunt'), ('prince','princess'), ('actor','actress'),
    ('waiter','waitress'),
]
ZH_GENDER = [
    ('男人','女人'), ('国王','女王'), ('父亲','母亲'), ('儿子','女儿'),
    ('丈夫','妻子'), ('叔叔','阿姨'), ('王子','公主'), ('男孩','女孩'),
    ('兄弟','姐妹'),
]

print("\nDAY 356: Semantic Universality — Does it Extend Beyond Gender?")
print("="*70)
print("Hypothesis: SEMANTIC axes are universal, MORPHOLOGICAL axes are not")
print()

# ====================================================================
# PHASE 1: Build all axes and compute coherence
# ====================================================================
print("Phase 1: Build axes and measure coherence")
print()

axes = {}
configs = [
    ('EN_gender',    EN_GENDER,    False, RELAXED_MASK),
    ('EN_size',      EN_SIZE,      False, RELAXED_MASK),
    ('EN_sentiment', EN_SENTIMENT, False, RELAXED_MASK),
    ('EN_age',       EN_AGE,       False, RELAXED_MASK),
    ('EN_plural',    EN_PLURAL,    False, RELAXED_MASK),
    ('ZH_gender',    ZH_GENDER,    True,  ZH_MASK),
    ('ZH_size',      ZH_SIZE,      True,  ZH_MASK),
    ('ZH_sentiment', ZH_SENTIMENT, True,  ZH_MASK),
    ('ZH_age',       ZH_AGE,       True,  ZH_MASK),
    ('ZH_plural',    ZH_PLURAL_EQUIV, True, ZH_MASK),
]

for name, pairs, zh, mask in configs:
    ax_dir, usable, skipped = build_axis(pairs, zh=zh)
    if ax_dir is None:
        print("  %-16s  NO USABLE PAIRS" % name); continue
    coh = axis_coherence(usable, zh=zh)
    s   = best_scale(ax_dir, usable, mask, zh=zh)
    h, n, _ = eval_axis(ax_dir, s, usable, mask, zh=zh)
    axes[name] = (ax_dir, usable, s, mask, zh)
    print("  %-16s  n=%2d  coh=%.4f  scale=%.3f  train_acc=%d/%d=%.0f%%" % (
        name, len(usable), coh, s, h, n, 100*h/max(n,1)))

# ====================================================================
# PHASE 2: Cross-language axis alignment matrix
# ====================================================================
print("\nPhase 2: Cross-language axis alignment (cosine)")
print()

semantic_pairs = [
    ('EN_gender',    'ZH_gender'),
    ('EN_size',      'ZH_size'),
    ('EN_sentiment', 'ZH_sentiment'),
    ('EN_age',       'ZH_age'),
    ('EN_plural',    'ZH_plural'),
]

print("  %-16s  %-16s  cos     interpretation" % ('Axis1', 'Axis2'))
print("  " + "-"*60)
for a1, a2 in semantic_pairs:
    if a1 not in axes or a2 not in axes: print("  %-16s  %-16s  --" % (a1,a2)); continue
    d1 = axes[a1][0]; d2 = axes[a2][0]
    cos = float(np.dot(d1.astype(np.float32), d2.astype(np.float32)))
    interp = ('UNIVERSAL (>0.7)' if abs(cos) > 0.7 else
              'PARTIAL (0.4-0.7)' if abs(cos) > 0.4 else
              'LOW (0.2-0.4)'     if abs(cos) > 0.2 else
              'ORTHOGONAL (<0.2)')
    print("  %-16s  %-16s  %.4f  %s" % (a1, a2, cos, interp))

print()
print("  Reference: EN↔ZH gender cos = 0.7425 (Day 355, the universal baseline)")

# ====================================================================
# PHASE 3: Zero-shot cross-language transfer for each axis type
# ====================================================================
print("\nPhase 3: Zero-shot transfer — EN axis on ZH words, ZH axis on EN words")
print()

transfer_configs = [
    ('gender',    'EN_gender', 'ZH_gender', EN_GENDER[:5], ZH_GENDER),
    ('size',      'EN_size',   'ZH_size',   EN_SIZE[:5],   ZH_SIZE),
    ('sentiment', 'EN_sentiment','ZH_sentiment', EN_SENTIMENT[:5], ZH_SENTIMENT),
    ('age',       'EN_age',    'ZH_age',    EN_AGE,        ZH_AGE),
]

for ax_type, en_key, zh_key, en_test, zh_test in transfer_configs:
    if en_key not in axes or zh_key not in axes:
        print("  %s: one axis missing" % ax_type); continue

    en_dir, _, s_en, _, _ = axes[en_key]
    zh_dir, _, s_zh, _, _ = axes[zh_key]

    # EN axis → ZH test words
    h_en_on_zh, n, det = eval_axis(en_dir, s_en, zh_test, ZH_MASK, zh=True)
    # ZH axis → EN test words
    h_zh_on_en, n2, det2 = eval_axis(zh_dir, s_zh, en_test, RELAXED_MASK, zh=False)

    # Optimised scale for cross-language
    s_en_opt = best_scale(en_dir, zh_test, ZH_MASK, zh=True)
    h_en_opt, n, _ = eval_axis(en_dir, s_en_opt, zh_test, ZH_MASK, zh=True)

    s_zh_opt = best_scale(zh_dir, en_test, RELAXED_MASK, zh=False)
    h_zh_opt, n2, _ = eval_axis(zh_dir, s_zh_opt, en_test, RELAXED_MASK, zh=False)

    print("  %-12s  EN→ZH: %d/%d=%.0f%% (opt=%d/%d=%.0f%%)   ZH→EN: %d/%d=%.0f%% (opt=%d/%d=%.0f%%)" % (
        ax_type,
        h_en_on_zh, n,  100*h_en_on_zh/max(n,1),
        h_en_opt,   n,  100*h_en_opt/max(n,1),
        h_zh_on_en, n2, 100*h_zh_on_en/max(n2,1),
        h_zh_opt,   n2, 100*h_zh_opt/max(n2,1)))

    # Detail for ZH test
    for s,t,w,ok,sim in det:
        mark = '✓' if ok else '✗'
        print("    EN→ZH %s %-6s → %-6s  [found: %s %.4f]" % (mark, s, t, w, sim))
    print()

# ====================================================================
# PHASE 4: EN–EN axis alignment (within-language: semantic vs morphological)
# ====================================================================
print("Phase 4: Within-English axis alignment (semantic vs morphological)")
print()

en_axes = ['EN_gender', 'EN_size', 'EN_sentiment', 'EN_age', 'EN_plural']
header = "  " + "%-16s" % "" + "".join("%-12s" % k.replace('EN_','') for k in en_axes)
print(header)
for a1 in en_axes:
    if a1 not in axes: continue
    row = "  %-16s" % a1.replace('EN_','')
    for a2 in en_axes:
        if a2 not in axes: row += "%-12s" % "--"; continue
        cos = float(np.dot(axes[a1][0].astype(np.float32),
                           axes[a2][0].astype(np.float32)))
        row += "%-12.4f" % cos
    print(row)
print()
print("  (Off-diagonal values near 0 = orthogonal semantic directions)")

# ====================================================================
# PHASE 5: ZH–ZH axis alignment
# ====================================================================
print("\nPhase 5: Within-Chinese axis alignment")
print()

zh_axes = ['ZH_gender', 'ZH_size', 'ZH_sentiment', 'ZH_age', 'ZH_plural']
header = "  " + "%-16s" % "" + "".join("%-12s" % k.replace('ZH_','') for k in zh_axes)
print(header)
for a1 in zh_axes:
    if a1 not in axes: continue
    row = "  %-16s" % a1.replace('ZH_','')
    for a2 in zh_axes:
        if a2 not in axes: row += "%-12s" % "--"; continue
        cos = float(np.dot(axes[a1][0].astype(np.float32),
                           axes[a2][0].astype(np.float32)))
        row += "%-12.4f" % cos
    print(row)

# ====================================================================
# PHASE 6: Axis interpolation along the English gender axis
# ====================================================================
print("\n" + "="*70)
print("Phase 6: Axis interpolation — what lives BETWEEN male and female?")
print("="*70)
print()

en_gender_dir = axes['EN_gender'][0]
s_gender = axes['EN_gender'][2]

# Interpolate between 男人 (man) and 女人 (woman) in ZH space
zh_pairs_for_interp = [('man', 'woman'), ('king', 'queen'), ('boy', 'girl')]
for src, tgt in zh_pairs_for_interp:
    es, _ = get_emb(src)
    et, _ = get_emb(tgt)
    if es is None or et is None: continue
    print("  Interpolation: %s → %s" % (src, tgt))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        interp = es + frac * s_gender * en_gender_dir
        top = nn_ret_top(interp, set(), RELAXED_MASK, top_k=3)
        print("    α=%.2f  → %s" % (frac, [(w, round(c,4)) for w,c,_ in top]))
    print()

# Interpolation in Chinese
print("  Interpolation in Chinese: 男人 → 女人")
for frac in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    es_zh, _ = get_emb('男人', zh=True)
    interp = es_zh + frac * s_gender * en_gender_dir
    top = nn_ret_top(interp, set(), ZH_MASK, top_k=3)
    top_en = nn_ret_top(interp, set(), EN_MASK, top_k=2)
    print("    α=%.2f  ZH:%s  EN:%s" % (
        frac,
        [(w, round(c,4)) for w,c,_ in top],
        [(w, round(c,4)) for w,c,_ in top_en]))
print()

# ====================================================================
# PHASE 7: Axis extrapolation — beyond female
# ====================================================================
print("Phase 7: Axis extrapolation — past the poles")
print()

# Walk from 'man' in large steps along gender axis
print("  Walking from 'man' along gender axis (EN vocab):")
for scale in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
    es, _ = get_emb('man')
    pred = es + scale * en_gender_dir
    top = nn_ret_top(pred, set(), RELAXED_MASK, top_k=5)
    print("    scale=% .1f  → %s" % (scale, [(w, round(c,4)) for w,c,_ in top]))
print()

print("  Walking from '男人' along gender axis (ZH vocab):")
es_zh, _ = get_emb('男人', zh=True)
for scale in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
    pred = es_zh + scale * en_gender_dir
    top = nn_ret_top(pred, set(), ZH_MASK, top_k=4)
    print("    scale=% .1f  → %s" % (scale, [(w, round(c,4)) for w,c,_ in top]))

# ====================================================================
# PHASE 8: Summary table — universality verdict for each axis type
# ====================================================================
print("\n" + "="*70)
print("SUMMARY: Semantic vs Morphological Universality")
print("="*70)
print()
print("  Axis type     cos(EN,ZH)   Transfer?   Type")
print("  " + "-"*55)

verdict_order = [
    ('EN_gender', 'ZH_gender', 'SEMANTIC'),
    ('EN_size',   'ZH_size',   'SEMANTIC'),
    ('EN_sentiment','ZH_sentiment','SEMANTIC'),
    ('EN_age',    'ZH_age',    'SEMANTIC'),
    ('EN_plural', 'ZH_plural', 'MORPHOLOGICAL'),
]
for a1, a2, ax_type in verdict_order:
    if a1 not in axes or a2 not in axes:
        print("  %-14s  --           (missing pairs)" % a1.replace('EN_','')); continue
    d1 = axes[a1][0]; d2 = axes[a2][0]
    cos = float(np.dot(d1.astype(np.float32), d2.astype(np.float32)))
    level = ('UNIVERSAL' if abs(cos) > 0.7 else
             'PARTIAL'   if abs(cos) > 0.4 else
             'LOCAL'     if abs(cos) > 0.2 else
             'ORTHOGONAL')
    print("  %-14s  %.4f       %-12s  %s" % (
        a1.replace('EN_',''), cos, level, ax_type))
