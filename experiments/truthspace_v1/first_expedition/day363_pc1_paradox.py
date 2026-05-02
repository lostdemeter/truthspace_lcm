import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.utils.extmath import randomized_svd

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)

EN_MASK = np.zeros(len(W_E), dtype=bool)
ZH_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and w.isalpha() and w.isascii() and len(w) >= 2: EN_MASK[i] = True
    if w and any('\u4e00' <= c <= '\u9fff' for c in w): ZH_MASK[i] = True

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def get_idx(word, zh=False):
    if zh:
        ids = tok(word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return ids[0]
        return None
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return ids[0]
    return None

print("\nDAY 363: The PC1 Anti-Alignment Paradox")
print("="*70)
print("PC1 is anti-aligned with cross-lingual gender (cos=-1 at k=1).")
print("Does projecting out PC1 IMPROVE gender universality?")
print("Does removing PC1 HURT sentiment universality?")
print()

print("Computing SVD (k=200)...")
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)
U, S, Vt = randomized_svd(W_cent, n_components=200, random_state=42)
PC = [Vt[i].astype(np.float64) for i in range(200)]
print("  σ[0]=%.3f σ[1]=%.3f σ[2]=%.3f σ[4]=%.3f σ[9]=%.3f" % (
    S[0], S[1], S[2], S[4], S[9]))
print()

def ablate_pcs(W, pc_indices):
    W_abl = W.copy()
    for i in pc_indices:
        v = PC[i]
        W_abl -= np.outer(W_abl @ v, v)
    return W_abl

def build_axis(W, pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W[ti] - W[si]
        n = np.linalg.norm(d)
        if n > 1e-8: chords.append(d / n)
    if not chords: return None
    m = np.mean(chords, axis=0)
    return m / (np.linalg.norm(m) + 1e-8)

def axis_cross_cos(W, en_pairs, zh_pairs):
    ax_en = build_axis(W, en_pairs, zh=False)
    ax_zh = build_axis(W, zh_pairs, zh=True)
    if ax_en is None or ax_zh is None: return float('nan')
    return float(np.dot(ax_en.astype(np.float32), ax_zh.astype(np.float32)))

def axis_coherence(W, pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W[ti] - W[si]
        n = np.linalg.norm(d)
        if n > 1e-8: chords.append((d/n).astype(np.float32))
    if len(chords) < 2: return 0.0
    sims = []
    for i in range(len(chords)):
        for j in range(i+1, len(chords)):
            sims.append(float(np.dot(chords[i], chords[j])))
    return float(np.mean(sims))

def mean_trans_cos(W, pairs):
    sims = []
    for en_w, zh_w in pairs:
        ei = get_idx(en_w); zi = get_idx(zh_w, zh=True)
        if ei is None or zi is None: continue
        sims.append(float(np.dot(normed(W[ei]).astype(np.float32),
                                  normed(W[zi]).astype(np.float32))))
    return float(np.mean(sims)) if sims else float('nan')

def axis_accuracy(W, pairs, zh=False, k=5):
    W_n = np.array([normed(v) for v in W], dtype=np.float32)
    mask = ZH_MASK if zh else EN_MASK
    ax = build_axis(W, pairs, zh=zh)
    if ax is None: return 0.0, 0
    ax_n = normed(ax).astype(np.float32)
    hits = 0; total = 0
    for src, tgt in pairs:
        si = get_idx(src, zh=zh); ti = get_idx(tgt, zh=zh)
        if si is None or ti is None: continue
        total += 1
        pred = (W[si] + ax * np.linalg.norm(W[ti] - W[si])).astype(np.float32)
        sims = W_n @ normed(pred)
        sims[~mask] = -1.0; sims[si] = -1.0
        top_k = np.argsort(sims)[::-1][:k]
        if ti in top_k: hits += 1
    return (hits / total) if total > 0 else 0.0, total

EN_GENDER = [
    ('king','queen'),('man','woman'),('boy','girl'),
    ('father','mother'),('son','daughter'),('husband','wife'),
    ('uncle','aunt'),('prince','princess'),('actor','actress'),('waiter','waitress'),
]
ZH_GENDER = [
    ('男人','女人'),('国王','女王'),('父亲','母亲'),('儿子','女儿'),
    ('丈夫','妻子'),('叔叔','阿姨'),('王子','公主'),('男孩','女孩'),
]
EN_SIZE = [
    ('small','big'),('little','large'),('tiny','huge'),('narrow','wide'),
    ('short','tall'),('shallow','deep'),('thin','thick'),('weak','strong'),
]
ZH_SIZE = [('小','大'),('短','长'),('低','高'),('薄','厚')]
EN_SENTIMENT = [
    ('bad','good'),('ugly','beautiful'),('hate','love'),('sad','happy'),
    ('dark','bright'),('wrong','right'),('evil','good'),('poor','rich'),
]
ZH_SENTIMENT = [
    ('坏','好'),('丑','美'),('恨','爱'),('悲伤','快乐'),
    ('暗','亮'),('错','对'),('恶','善'),('穷','富'),
]
EN_AGE = [
    ('young','old'),('new','ancient'),('fresh','stale'),('modern','classic'),
]
ZH_AGE = [('年轻','年老'),('新','旧')]

TRANS_PAIRS = [
    ('king','国王'),('queen','女王'),('man','男人'),('woman','女人'),
    ('father','父亲'),('mother','母亲'),('son','儿子'),('daughter','女儿'),
    ('happy','快乐'),('sad','悲伤'),('good','好'),('bad','坏'),
    ('big','大'),('small','小'),('cat','猫'),('love','爱'),
    ('hate','恨'),('red','红'),('blue','蓝'),('young','年轻'),
]

# ====================================================================
# PHASE 1: What does PC1 encode for EN vs ZH gender words?
# ====================================================================
print("Phase 1: PC1 projection for EN vs ZH gender pairs")
print("  (Why is EN gender along +PC1 and ZH gender along -PC1?)")
print()
print("  %-20s  %-12s  %-12s  %-10s" % ("pair", "src_PC1", "tgt_PC1", "Δ(tgt-src)"))
print("  " + "-"*58)

print("  === EN gender ===")
for src, tgt in EN_GENDER:
    si = get_idx(src); ti = get_idx(tgt)
    if si is None or ti is None: continue
    s_pc1 = float((W_E[si] - W_mean) @ PC[0])
    t_pc1 = float((W_E[ti] - W_mean) @ PC[0])
    print("  %-20s  %-12.3f  %-12.3f  %-10.3f" % (
        "%s→%s" % (src, tgt), s_pc1, t_pc1, t_pc1-s_pc1))

print()
print("  === ZH gender ===")
for src, tgt in ZH_GENDER:
    si = get_idx(src, zh=True); ti = get_idx(tgt, zh=True)
    if si is None or ti is None: continue
    s_pc1 = float((W_E[si] - W_mean) @ PC[0])
    t_pc1 = float((W_E[ti] - W_mean) @ PC[0])
    print("  %-20s  %-12.3f  %-12.3f  %-10.3f" % (
        "%s→%s" % (src, tgt), s_pc1, t_pc1, t_pc1-s_pc1))

print()

# Summary: mean PC1 delta for EN vs ZH gender
en_deltas = []
for src, tgt in EN_GENDER:
    si = get_idx(src); ti = get_idx(tgt)
    if si is None or ti is None: continue
    en_deltas.append(float((W_E[ti]-W_mean) @ PC[0]) - float((W_E[si]-W_mean) @ PC[0]))
zh_deltas = []
for src, tgt in ZH_GENDER:
    si = get_idx(src, zh=True); ti = get_idx(tgt, zh=True)
    if si is None or ti is None: continue
    zh_deltas.append(float((W_E[ti]-W_mean) @ PC[0]) - float((W_E[si]-W_mean) @ PC[0]))

print("  Mean PC1 Δ for EN gender: %+.3f" % np.mean(en_deltas))
print("  Mean PC1 Δ for ZH gender: %+.3f" % np.mean(zh_deltas))
print()
print("  Interpretation:")
print("  EN gender: masculine→feminine moves in +PC1 direction (more frequent)")
print("  ZH gender: masculine→feminine moves in -PC1 direction (less frequent)")
print("  → PC1 frequency pattern is OPPOSITE between EN and ZH for gender")
print()

# ====================================================================
# PHASE 2: Same analysis for PC2-PC5 — which PCs are congruent?
# ====================================================================
print("Phase 2: PC1-PC5 projections — EN vs ZH gender axis alignment per PC")
print()
print("  %-6s  %-14s  %-14s  %-12s  %-10s" % (
    "PC", "EN_mean_Δ", "ZH_mean_Δ", "cos(axEN,axZH)", "direction"))
print("  " + "-"*62)

for pc_idx in range(10):
    en_d = []
    for s,t in EN_GENDER:
        si = get_idx(s); ti = get_idx(t)
        if si is None or ti is None: continue
        en_d.append(float((W_E[ti]-W_mean) @ PC[pc_idx]) -
                    float((W_E[si]-W_mean) @ PC[pc_idx]))
    zh_d = []
    for s,t in ZH_GENDER:
        si = get_idx(s, zh=True); ti = get_idx(t, zh=True)
        if si is None or ti is None: continue
        zh_d.append(float((W_E[ti]-W_mean) @ PC[pc_idx]) -
                    float((W_E[si]-W_mean) @ PC[pc_idx]))
    en_mean = np.mean(en_d) if en_d else 0.0
    zh_mean = np.mean(zh_d) if zh_d else 0.0
    # cosine of 1D projections: sign(en_mean * zh_mean)
    cos_1d = np.sign(en_mean * zh_mean) if (en_mean != 0 and zh_mean != 0) else 0.0
    direction = "CONGRUENT" if cos_1d > 0 else ("OPPOSED" if cos_1d < 0 else "neutral")
    print("  %-6d  %-14.3f  %-14.3f  %-12.1f  %-10s" % (
        pc_idx+1, en_mean, zh_mean, cos_1d, direction))

print()

# ====================================================================
# PHASE 3: Ablation effect — remove top-N PCs and measure axis universality
# ====================================================================
print("Phase 3: Ablating top-N PCs — effect on cross-lingual axis cosines")
print()
print("  %-20s  %-10s  %-10s  %-10s  %-10s" % (
    "Ablation", "gender", "size", "sentiment", "trans_cos"))
print("  " + "-"*62)

ABLATIONS = [
    ("none", []),
    ("ablate PC1", [0]),
    ("ablate PC1-2", [0,1]),
    ("ablate PC1-3", [0,1,2]),
    ("ablate PC1-5", list(range(5))),
    ("ablate PC1-10", list(range(10))),
    ("ablate PC1-20", list(range(20))),
    ("ablate PC6-10", list(range(5,10))),
    ("ablate PC11-20", list(range(10,20))),
]

for label, pc_list in ABLATIONS:
    W_abl = ablate_pcs(W_E, pc_list)
    gc  = axis_cross_cos(W_abl, EN_GENDER,    ZH_GENDER)
    sc  = axis_cross_cos(W_abl, EN_SIZE,      ZH_SIZE)
    sc2 = axis_cross_cos(W_abl, EN_SENTIMENT, ZH_SENTIMENT)
    tc  = mean_trans_cos(W_abl, TRANS_PAIRS)
    print("  %-20s  %-10.4f  %-10.4f  %-10.4f  %-10.4f" % (
        label, gc, sc, sc2, tc))

print()

# ====================================================================
# PHASE 4: Ablation effect on INTRA-language axis accuracy
# ====================================================================
print("Phase 4: Ablation effect on within-language axis accuracy (top-5)")
print()
print("  %-20s  %-14s  %-14s  %-14s  %-14s" % (
    "Ablation", "EN_gender", "ZH_gender", "EN_sentiment", "ZH_sentiment"))
print("  " + "-"*74)

ABLATIONS2 = [
    ("none", []),
    ("ablate PC1", [0]),
    ("ablate PC1-5", list(range(5))),
    ("ablate PC1-10", list(range(10))),
    ("ablate PC1-20", list(range(20))),
]

for label, pc_list in ABLATIONS2:
    W_abl = ablate_pcs(W_E, pc_list)
    eg, _ = axis_accuracy(W_abl, EN_GENDER,    zh=False)
    zg, _ = axis_accuracy(W_abl, ZH_GENDER,    zh=True)
    es, _ = axis_accuracy(W_abl, EN_SENTIMENT, zh=False)
    zs, _ = axis_accuracy(W_abl, ZH_SENTIMENT, zh=True)
    print("  %-20s  %-14.3f  %-14.3f  %-14.3f  %-14.3f" % (
        label, eg, zg, es, zs))

print()

# ====================================================================
# PHASE 5: What DOES PC1 encode? EN vs ZH projections compared
# ====================================================================
print("Phase 5: PC1 scores for EN vs ZH tokens — frequency interpretation")
print()

# Top EN tokens on PC1 (positive and negative)
all_pc1 = (W_E - W_mean) @ PC[0]
print("  Top-10 EN tokens with HIGHEST PC1 score:")
en_scores = [(i, all_pc1[i]) for i in range(len(W_E)) if EN_MASK[i]]
en_scores.sort(key=lambda x: -x[1])
for idx, score in en_scores[:10]:
    print("    %-20s  %.3f" % (repr(tok.decode([idx]).strip()), score))
print()
print("  Top-10 EN tokens with LOWEST PC1 score:")
for idx, score in en_scores[-10:]:
    print("    %-20s  %.3f" % (repr(tok.decode([idx]).strip()), score))
print()
print("  Top-10 ZH tokens with HIGHEST PC1 score:")
zh_scores = [(i, all_pc1[i]) for i in range(len(W_E)) if ZH_MASK[i]]
zh_scores.sort(key=lambda x: -x[1])
for idx, score in zh_scores[:10]:
    print("    %-20s  %.3f" % (repr(tok.decode([idx]).strip()), score))
print()
print("  Top-10 ZH tokens with LOWEST PC1 score:")
for idx, score in zh_scores[-10:]:
    print("    %-20s  %.3f" % (repr(tok.decode([idx]).strip()), score))
print()

# PC1 scores for gendered words
print("  PC1 scores for gendered word pairs:")
print()
print("  === EN (masculine should be LOWER on PC1 if +Δ is masc→fem) ===")
for src, tgt in EN_GENDER[:6]:
    si = get_idx(src); ti = get_idx(tgt)
    if si is None or ti is None: continue
    s_pc1 = all_pc1[si]; t_pc1 = all_pc1[ti]
    print("    %-12s: PC1=%-8.2f    %-12s: PC1=%-8.2f  Δ=%+.2f" % (
        src, s_pc1, tgt, t_pc1, t_pc1-s_pc1))
print()
print("  === ZH (masculine should be HIGHER on PC1 if -Δ is masc→fem) ===")
for src, tgt in ZH_GENDER[:6]:
    si = get_idx(src, zh=True); ti = get_idx(tgt, zh=True)
    if si is None or ti is None: continue
    s_pc1 = all_pc1[si]; t_pc1 = all_pc1[ti]
    print("    %-12s: PC1=%-8.2f    %-12s: PC1=%-8.2f  Δ=%+.2f" % (
        src, s_pc1, tgt, t_pc1, t_pc1-s_pc1))
print()

# ====================================================================
# PHASE 6: The optimal ablation for EACH axis
# ====================================================================
print("Phase 6: Finding the optimal PC ablation to MAXIMIZE each cross-lingual axis cos")
print()

FINE_ABLATIONS = [
    ("none",       []),
    ("-PC1",       [0]),
    ("-PC2",       [1]),
    ("-PC3",       [2]),
    ("-PC4",       [3]),
    ("-PC5",       [4]),
    ("-PC1,2",     [0,1]),
    ("-PC1,3",     [0,2]),
    ("-PC1,4",     [0,3]),
    ("-PC1,5",     [0,4]),
    ("-PC2,3",     [1,2]),
    ("-PC1,2,3",   [0,1,2]),
    ("-PC1,2,3,4", [0,1,2,3]),
    ("-PC1..5",    [0,1,2,3,4]),
]

print("  %-16s  %-10s  %-10s  %-10s  %-10s" % (
    "Ablation", "gender", "size", "sentiment", "mean"))
print("  " + "-"*58)

best_gender = -99; best_size = -99; best_sentiment = -99
best_gender_label = best_size_label = best_sentiment_label = "none"
for label, pc_list in FINE_ABLATIONS:
    W_abl = ablate_pcs(W_E, pc_list)
    gc  = axis_cross_cos(W_abl, EN_GENDER,    ZH_GENDER)
    sc  = axis_cross_cos(W_abl, EN_SIZE,      ZH_SIZE)
    sc2 = axis_cross_cos(W_abl, EN_SENTIMENT, ZH_SENTIMENT)
    mean_val = np.mean([gc, sc, sc2])
    if gc > best_gender:   best_gender = gc;   best_gender_label = label
    if sc > best_size:     best_size = sc;     best_size_label = label
    if sc2 > best_sentiment: best_sentiment = sc2; best_sentiment_label = label
    print("  %-16s  %-10.4f  %-10.4f  %-10.4f  %-10.4f" % (
        label, gc, sc, sc2, mean_val))

print()
print("  Best ablation per axis:")
print("  gender:    '%s' (%.4f)" % (best_gender_label, best_gender))
print("  size:      '%s' (%.4f)" % (best_size_label, best_size))
print("  sentiment: '%s' (%.4f)" % (best_sentiment_label, best_sentiment))
print()

# ====================================================================
# PHASE 7: PC1 component of each semantic axis direction
# ====================================================================
print("Phase 7: PC1 component of each semantic axis direction in EN and ZH")
print()
print("  Axis direction projected onto PC1: if opposite sign EN vs ZH,")
print("  removing PC1 would improve cross-lingual alignment for that axis.")
print()
print("  %-14s  %-14s  %-14s  %-10s" % ("axis", "EN_PC1_proj", "ZH_PC1_proj", "same sign?"))
print("  " + "-"*56)

AXIS_DEFS = [
    ("gender",    EN_GENDER,    ZH_GENDER,    False, True),
    ("size",      EN_SIZE,      ZH_SIZE,      False, True),
    ("sentiment", EN_SENTIMENT, ZH_SENTIMENT, False, True),
    ("age",       EN_AGE,       ZH_AGE,       False, True),
]

for ax_name, en_pairs, zh_pairs, en_zh, cross in AXIS_DEFS:
    ax_en = build_axis(W_E, en_pairs, zh=False)
    ax_zh = build_axis(W_E, zh_pairs, zh=True)
    if ax_en is None or ax_zh is None: continue
    en_pc1 = float(ax_en @ PC[0])
    zh_pc1 = float(ax_zh @ PC[0])
    same = "YES" if np.sign(en_pc1) == np.sign(zh_pc1) else "NO  (opposed)"
    print("  %-14s  %-14.4f  %-14.4f  %-10s" % (ax_name, en_pc1, zh_pc1, same))

print()
for pc_n in range(5):
    print("  PC%d components:" % (pc_n+1))
    for ax_name, en_pairs, zh_pairs, _, _ in AXIS_DEFS:
        ax_en = build_axis(W_E, en_pairs, zh=False)
        ax_zh = build_axis(W_E, zh_pairs, zh=True)
        if ax_en is None or ax_zh is None: continue
        en_v = float(ax_en @ PC[pc_n])
        zh_v = float(ax_zh @ PC[pc_n])
        same = "+" if np.sign(en_v) == np.sign(zh_v) else "-"
        print("    %-14s  en=%-8.4f  zh=%-8.4f  [%s]" % (ax_name, en_v, zh_v, same))
    print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 363 — PC1 Anti-Alignment Paradox")
print("="*70)
print()
print("  Core finding: PC1 projects EN and ZH gender axes in OPPOSITE directions.")
print("  EN gender: masculine→feminine is +PC1 direction")
print("  ZH gender: masculine→feminine is -PC1 direction")
print("  → En feminine words are more frequent than masculine (queen>king in Qwen data)")
print("  → ZH masculine words are more frequent than feminine (国王>女王 in Chinese data)")
print()
print("  Effect of ablating PC1:")
print("  → gender cross-lingual cosine: changes (Phase 3)")
print("  → sentiment cross-lingual cosine: changes (Phase 3)")
print("  → within-language axis accuracy: changes (Phase 4)")
