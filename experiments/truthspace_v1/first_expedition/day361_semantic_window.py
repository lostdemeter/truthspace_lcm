import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.utils.extmath import randomized_svd
from collections import defaultdict

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)

EN_MASK = np.zeros(len(W_E), dtype=bool)
ZH_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and w.isalpha() and w.isascii() and len(w) >= 2: EN_MASK[i] = True
    if w and any('\u4e00' <= c <= '\u9fff' for c in w): ZH_MASK[i] = True

BOTH_MASK = EN_MASK | ZH_MASK

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

print("\nDAY 361: The k=50 Semantic Window — Cross-Lingual Clustering")
print("="*70)
print()

# ====================================================================
# BUILD PROJECTED MATRICES at k=50 and k=200 (reference)
# ====================================================================
print("Computing SVD and building k=50 and k=200 projected matrices...")
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)
U, S, Vt = randomized_svd(W_cent, n_components=300, random_state=42)
print("  σ spectrum: %.2f  %.2f  %.2f  %.2f  %.2f ... %.4f" % (
    S[0], S[1], S[2], S[3], S[4], S[-1]))

def project_k(k):
    Vk = Vt[:k].astype(np.float64)
    coords = W_cent.astype(np.float64) @ Vk.T
    W_proj = coords @ Vk + W_mean
    return W_proj

def make_Wn(W):
    return np.array([normed(v) for v in W], dtype=np.float32)

print("  Building W_proj for k=50...")
W50  = project_k(50)
W50n = make_Wn(W50)
print("  Building W_proj for k=200...")
W200  = project_k(200)
W200n = make_Wn(W200)
W_n_full = make_Wn(W_E)
print("  Done.\n")

# ====================================================================
# PHASE 1: Semantic NN at k=50 — cross-lingual neighbors
# ====================================================================
print("Phase 1: Nearest neighbours at k=50 — cross-lingual semantic merging")
print()
print("  At k=50, semantic signal dominates identity. Do EN and ZH words")
print("  cluster together by meaning rather than by language?")
print()

TEST_WORDS_EN = [
    'king','queen','man','woman','boy','girl','father','mother',
    'good','bad','happy','sad','big','small','fast','slow',
    'cat','dog','tree','house','love','hate','red','blue',
]

ZH_SEMANTIC = {
    'king':   ('国王','王国'),
    'queen':  ('女王','皇后'),
    'man':    ('男人','男子'),
    'woman':  ('女人','女子'),
    'father': ('父亲','爸爸'),
    'mother': ('母亲','妈妈'),
    'good':   ('好','良好'),
    'bad':    ('坏','不好'),
    'happy':  ('快乐','高兴'),
    'sad':    ('悲伤','难过'),
    'big':    ('大','巨大'),
    'small':  ('小','微小'),
    'cat':    ('猫','小猫'),
    'dog':    ('狗','小狗'),
}

def top_k_nn(W_proj_n, idx, mask, k=5, excl=None):
    sims = W_proj_n @ W_proj_n[idx]
    sims[~mask] = -1.0
    sims[idx] = -1.0
    if excl:
        for e in excl: sims[e] = -1.0
    top_idx = np.argsort(sims)[::-1][:k]
    return [(tok.decode([int(i)]).strip(), float(sims[i])) for i in top_idx]

print("  EN word  →  top-5 NNs in BOTH-language mask (k=50)")
print("  (* = Chinese token, ! = exact ZH translation)")
print()
print("  %-10s  %-14s  %-55s" % ("word", "k=200 top-1 NN", "k=50 top-5 NNs (EN+ZH)"))
print("  " + "-"*85)

cross_lingual_hits = 0; tested = 0
for word in TEST_WORDS_EN:
    idx = get_idx(word)
    if idx is None: continue
    # k=200 top-1 (should be case variant)
    nn200 = top_k_nn(W200n, idx, BOTH_MASK, k=1)
    # k=50 top-5
    nn50  = top_k_nn(W50n, idx, BOTH_MASK, k=5)
    # Check if any ZH translation appears in top-5
    zh_targets = ZH_SEMANTIC.get(word, ())
    zh_in_top5 = [zh for zh in zh_targets
                  if any(get_idx(zh, zh=True) == get_idx(n[0], zh=True) or n[0] == zh
                         for n in nn50)]
    labels = []
    for n, sim in nn50:
        marker = '*' if any('\u4e00' <= c <= '\u9fff' for c in n) else ' '
        labels.append("%s%s" % (marker, n[:8]))
    tested += 1
    zh_hit = len(zh_in_top5) > 0
    if zh_hit: cross_lingual_hits += 1
    zh_mark = " [ZH✓]" if zh_hit else ""
    print("  %-10s  %-14s  %-55s%s" % (
        word, nn200[0][0] if nn200 else "?",
        "  ".join("%-10s" % l for l in labels), zh_mark))

print()
print("  Cross-lingual top-5 hit rate: %d / %d (%.0f%%)" % (
    cross_lingual_hits, tested, 100*cross_lingual_hits/tested if tested else 0))
print()

# ====================================================================
# PHASE 2: ZH words — do they cluster with EN translations at k=50?
# ====================================================================
print("Phase 2: ZH words find EN neighbours at k=50 (reverse cross-lingual)")
print()
print("  %-10s  %-14s  %-55s" % ("ZH word", "k=200 top-1", "k=50 top-5 NNs (EN+ZH)"))
print("  " + "-"*85)

ZH_TEST = [
    ('男人','man'),('女人','woman'),('国王','king'),('女王','queen'),
    ('父亲','father'),('母亲','mother'),('快乐','happy'),('悲伤','sad'),
    ('大','big'),('小','small'),('猫','cat'),('狗','dog'),
]
zh_to_en_hits = 0; zh_tested = 0
for zh_word, en_target in ZH_TEST:
    idx = get_idx(zh_word, zh=True)
    if idx is None: print("  %-10s  (no single token)" % zh_word); continue
    nn200 = top_k_nn(W200n, idx, BOTH_MASK, k=1)
    nn50  = top_k_nn(W50n, idx, BOTH_MASK, k=5)
    labels = []
    en_in_top5 = False
    for n, sim in nn50:
        marker = '*' if any('\u4e00' <= c <= '\u9fff' for c in n) else ' '
        labels.append("%s%s" % (marker, n[:8]))
        # Check if EN target (or its form) is in top-5
        en_idx = get_idx(en_target)
        n_idx = get_idx(n) or get_idx(n, zh=True)
        if n.lower().strip() == en_target.lower() or n.strip() == en_target: en_in_top5=True
    zh_tested += 1
    if en_in_top5: zh_to_en_hits += 1
    en_mark = " [EN✓]" if en_in_top5 else ""
    print("  %-10s  %-14s  %-55s%s" % (
        zh_word, nn200[0][0] if nn200 else "?",
        "  ".join("%-10s" % l for l in labels), en_mark))

print()
print("  ZH→EN cross-lingual hit rate: %d / %d (%.0f%%)" % (
    zh_to_en_hits, zh_tested, 100*zh_to_en_hits/zh_tested if zh_tested else 0))
print()

# ====================================================================
# PHASE 3: Semantic CLUSTER analysis at k=50
# ====================================================================
print("Phase 3: Semantic cluster analysis at k=50")
print("  Cluster by finding communities of mutually close tokens")
print()

# Use small curated set across EN and ZH
CLUSTER_PROBE = {
    'royalty_EN':  ['king','queen','prince','princess','royal','emperor'],
    'royalty_ZH':  ['国王','女王','王子','公主','皇帝'],
    'family_EN':   ['father','mother','son','daughter','husband','wife','uncle','aunt'],
    'family_ZH':   ['父亲','母亲','儿子','女儿','丈夫','妻子'],
    'animal_EN':   ['cat','dog','bird','fish','horse','wolf','lion','tiger'],
    'animal_ZH':   ['猫','狗','鸟','鱼','马','狼','狮子'],
    'emotion_EN':  ['happy','sad','angry','fear','joy','love','hate'],
    'emotion_ZH':  ['快乐','悲伤','愤怒','恐惧','喜悦','爱','恨'],
    'size_EN':     ['big','small','huge','tiny','large','little','massive','mini'],
    'size_ZH':     ['大','小','巨大','微小'],
}

print("  Average pairwise cosine similarity at k=50 and k=full:")
print()
print("  %-16s  %-10s  %-10s  %-10s" % ("category", "k=50 sim", "k=full sim", "ratio"))
print("  " + "-"*55)

for cat, words in CLUSTER_PROBE.items():
    is_zh = cat.endswith('_ZH')
    embs_50 = []; embs_full = []
    for w in words:
        idx = get_idx(w, zh=is_zh)
        if idx is not None:
            embs_50.append(normed(W50[idx]).astype(np.float32))
            embs_full.append(normed(W_E[idx]).astype(np.float32))
    if len(embs_50) < 2: continue
    def avg_cos(embs):
        sims = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                sims.append(float(np.dot(embs[i], embs[j])))
        return np.mean(sims)
    s50 = avg_cos(embs_50); sfull = avg_cos(embs_full)
    ratio = s50/sfull if sfull != 0 else float('nan')
    print("  %-16s  %-10.4f  %-10.4f  %-10.2f" % (cat, s50, sfull, ratio))

print()

# ====================================================================
# PHASE 4: Cross-lingual semantic alignment at k=50
# ====================================================================
print("Phase 4: Cross-lingual semantic alignment at k=50 vs full")
print("  Average cosine between EN concept and ZH translation")
print()
print("  %-16s  %-10s  %-10s  %-10s" % ("concept pair", "k=50", "k=full", "Δ"))
print("  " + "-"*50)

TRANSLATION_PAIRS = [
    ('king', '国王'), ('queen', '女王'), ('man', '男人'), ('woman', '女人'),
    ('father', '父亲'), ('mother', '母亲'), ('son', '儿子'), ('daughter', '女儿'),
    ('happy', '快乐'), ('sad', '悲伤'), ('good', '好'), ('bad', '坏'),
    ('big', '大'), ('small', '小'), ('cat', '猫'), ('dog', '狗'),
    ('love', '爱'), ('hate', '恨'), ('red', '红'), ('blue', '蓝'),
]

cos_50_list = []; cos_full_list = []
for en_w, zh_w in TRANSLATION_PAIRS:
    en_idx = get_idx(en_w); zh_idx = get_idx(zh_w, zh=True)
    if en_idx is None or zh_idx is None: continue
    c50   = float(np.dot(normed(W50[en_idx]).astype(np.float32),
                         normed(W50[zh_idx]).astype(np.float32)))
    cfull = float(np.dot(W_n_full[en_idx], W_n_full[zh_idx]))
    delta = c50 - cfull
    cos_50_list.append(c50); cos_full_list.append(cfull)
    sign = "+" if delta >= 0 else ""
    print("  %-16s  %-10.4f  %-10.4f  %s%.4f" % (
        "%s/%s" % (en_w, zh_w), c50, cfull, sign, delta))

print()
print("  MEAN cross-lingual cosine:")
print("  k=50:  %.4f" % np.mean(cos_50_list))
print("  k=full: %.4f" % np.mean(cos_full_list))
delta_mean = np.mean(cos_50_list) - np.mean(cos_full_list)
sign = "+" if delta_mean >= 0 else ""
print("  Δ:     %s%.4f" % (sign, delta_mean))
print()
print("  If k=50 has HIGHER cross-lingual cosine than full:")
print("  → The semantic window compresses language-specific features")
print("  → Cross-lingual alignment is STRONGER in low-dim projection")
print("  If k=50 is LOWER:")
print("  → Identity features (partially cross-lingual) dominate at high-k")
print()

# ====================================================================
# PHASE 5: Axis direction alignment at k=50 and full space
# ====================================================================
print("Phase 5: Do EN and ZH gender axes ALIGN better at k=50?")
print()

def build_axis_at(W_proj, pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        chords.append(normed(W_proj[ti] - W_proj[si]))
    if not chords: return None
    return normed(np.mean(chords, axis=0))

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
EN_SENTIMENT = [
    ('bad','good'),('ugly','beautiful'),('hate','love'),('sad','happy'),
    ('dark','bright'),('wrong','right'),('evil','good'),('poor','rich'),
]

print("  Cosine between EN and ZH gender axis at different k:")
for k, W_proj, label in [(50, W50, 'k=50'), (200, W200, 'k=200'), (1536, W_E, 'k=full')]:
    ax_en = build_axis_at(W_proj, EN_GENDER)
    ax_zh = build_axis_at(W_proj, ZH_GENDER, zh=True)
    if ax_en is None or ax_zh is None: print("  %s: N/A" % label); continue
    cos = float(np.dot(ax_en.astype(np.float32), ax_zh.astype(np.float32)))
    print("  %s: EN-ZH gender axis cos = %.4f" % (label, cos))

print()
print("  Intra-language coherence at k=50 vs full:")
print("  %-16s  %-10s  %-10s" % ("axis", "k=50", "k=full"))
print("  " + "-"*38)

def axis_coherence(W_proj, pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        chords.append(normed(W_proj[ti] - W_proj[si]).astype(np.float32))
    if len(chords) < 2: return 0.0
    sims = []
    for i in range(len(chords)):
        for j in range(i+1, len(chords)):
            sims.append(float(np.dot(chords[i], chords[j])))
    return float(np.mean(sims))

for ax_name, pairs, zh in [
    ('EN_gender',  EN_GENDER,    False),
    ('ZH_gender',  ZH_GENDER,    True),
    ('EN_size',    EN_SIZE,      False),
    ('EN_sentiment', EN_SENTIMENT, False),
]:
    c50   = axis_coherence(W50, pairs, zh=zh)
    cfull = axis_coherence(W_E, pairs, zh=zh)
    delta = c50 - cfull
    sign = "+" if delta >= 0 else ""
    print("  %-16s  %-10.4f  %-10.4f  (%s%.4f)" % (ax_name, c50, cfull, sign, delta))

print()

# ====================================================================
# PHASE 6: "Semantic window" top-50 NN cross-lingual composition
# ====================================================================
print("Phase 6: For each EN word, what fraction of top-50 NNs are ZH? (at k=50 vs full)")
print()
print("  %-12s  %-16s  %-16s" % ("word", "k=50 ZH%", "k=full ZH%"))
print("  " + "-"*48)

for word in ['king','man','woman','father','mother','cat','dog',
             'happy','sad','big','small','good','bad']:
    idx = get_idx(word)
    if idx is None: continue
    for W_proj_n, label in [(W50n, "k=50"), (W_n_full, "k=full")]:
        sims = W_proj_n @ W_proj_n[idx]
        sims[~BOTH_MASK] = -1.0; sims[idx] = -1.0
        top50_idx = np.argsort(sims)[::-1][:50]
        zh_count = sum(1 for i in top50_idx if ZH_MASK[i])
        en_count = sum(1 for i in top50_idx if EN_MASK[i])
    # Report both
    for W_proj_n_l, lbl in [(W50n, "k=50"), (W_n_full, "k=full")]:
        sims = W_proj_n_l @ W_proj_n_l[idx]
        sims[~BOTH_MASK] = -1.0; sims[idx] = -1.0
        top50 = np.argsort(sims)[::-1][:50]
        zh_frac = 100 * sum(1 for i in top50 if ZH_MASK[i]) / 50
        if lbl == "k=50": zh50 = zh_frac
        else: zhfull = zh_frac
    print("  %-12s  %-16s  %-16s" % (
        word, "%.1f%%" % zh50, "%.1f%%" % zhfull))

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 361 — k=50 Semantic Window Cross-Lingual Analysis")
print("="*70)
print()
print("  Phase 1: EN words at k=50 find ZH translations in top-5? (above)")
print("  Phase 2: ZH words at k=50 find EN translations in top-5? (above)")
print()
print("  Phase 4: Cross-lingual cosine (EN/ZH translation pairs)")
print("  k=50 mean:  %.4f" % np.mean(cos_50_list))
print("  k=full mean: %.4f" % np.mean(cos_full_list))
print()
print("  Key question: Is the semantic window (k=50) a natural cross-lingual")
print("  space? If so, this confirms: SEMANTIC information in embeddings is")
print("  inherently language-universal; IDENTITY information is language-specific.")
