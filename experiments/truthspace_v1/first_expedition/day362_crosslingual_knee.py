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

print("\nDAY 362: Cross-Lingual Universality Knee")
print("="*70)
print("Where exactly does EN-ZH axis alignment jump as k increases?")
print()

# ====================================================================
# SVD — compute up to 500 components (covers full range of interest)
# ====================================================================
print("Computing SVD (k=500)...")
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)
U, S, Vt = randomized_svd(W_cent, n_components=500, random_state=42)
print("  σ[0]=%.2f σ[49]=%.2f σ[99]=%.2f σ[199]=%.2f σ[299]=%.2f σ[499]=%.4f" % (
    S[0], S[49], S[99], S[199], S[299], S[499]))
print()

def project_k(k):
    Vk = Vt[:k].astype(np.float64)
    coords = W_cent.astype(np.float64) @ Vk.T
    return coords @ Vk + W_mean

# ====================================================================
# AXIS DEFINITIONS
# ====================================================================
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
ZH_SIZE = [
    ('小','大'),('短','长'),('低','高'),('薄','厚'),
]
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
    ('recent','historic'),('early','late'),
]
ZH_AGE = [
    ('年轻','年老'),('新','旧'),('早','晚'),
]

TRANSLATION_PAIRS = [
    ('king','国王'),('queen','女王'),('man','男人'),('woman','女人'),
    ('father','父亲'),('mother','母亲'),('son','儿子'),('daughter','女儿'),
    ('happy','快乐'),('sad','悲伤'),('good','好'),('bad','坏'),
    ('big','大'),('small','小'),('cat','猫'),('dog','狗'),
    ('love','爱'),('hate','恨'),('red','红'),('blue','蓝'),
    ('young','年轻'),('old','年老'),('new','新'),('old','旧'),
]

def build_axis(W_proj, pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_proj[ti] - W_proj[si]
        n = np.linalg.norm(d)
        if n > 1e-8: chords.append(d / n)
    if not chords: return None
    m = np.mean(chords, axis=0)
    return m / (np.linalg.norm(m) + 1e-8)

def axis_coherence(W_proj, pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_proj[ti] - W_proj[si]
        n = np.linalg.norm(d)
        if n > 1e-8: chords.append((d/n).astype(np.float32))
    if len(chords) < 2: return 0.0
    sims = []
    for i in range(len(chords)):
        for j in range(i+1, len(chords)):
            sims.append(float(np.dot(chords[i], chords[j])))
    return float(np.mean(sims))

def cross_ax_cos(W_proj, en_pairs, zh_pairs):
    ax_en = build_axis(W_proj, en_pairs, zh=False)
    ax_zh = build_axis(W_proj, zh_pairs, zh=True)
    if ax_en is None or ax_zh is None: return float('nan')
    return float(np.dot(ax_en.astype(np.float32), ax_zh.astype(np.float32)))

def mean_translation_cos(W_proj, trans_pairs):
    sims = []
    for en_w, zh_w in trans_pairs:
        en_i = get_idx(en_w); zh_i = get_idx(zh_w, zh=True)
        if en_i is None or zh_i is None: continue
        e = normed(W_proj[en_i]).astype(np.float32)
        z = normed(W_proj[zh_i]).astype(np.float32)
        sims.append(float(np.dot(e, z)))
    return float(np.mean(sims)) if sims else float('nan')

# ====================================================================
# PHASE 1: Cross-lingual axis alignment vs k
# ====================================================================
DIMS = [5, 10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 1536]

print("Phase 1: EN-ZH axis cosine similarity vs k PCs")
print()
print("  %-8s  %-10s  %-10s  %-10s  %-10s  %-14s" % (
    "k", "gender", "size", "sentiment", "age", "mean_trans_cos"))
print("  " + "-"*70)

gender_cos_by_k = {}; size_cos_by_k = {}; sentiment_cos_by_k = {}
age_cos_by_k = {}; trans_cos_by_k = {}

for k in DIMS:
    if k == 1536:
        W_proj = W_E.copy()
    else:
        W_proj = project_k(k)
    gc  = cross_ax_cos(W_proj, EN_GENDER,    ZH_GENDER)
    sc  = cross_ax_cos(W_proj, EN_SIZE,      ZH_SIZE)
    sc2 = cross_ax_cos(W_proj, EN_SENTIMENT, ZH_SENTIMENT)
    ac  = cross_ax_cos(W_proj, EN_AGE,       ZH_AGE)
    tc  = mean_translation_cos(W_proj, TRANSLATION_PAIRS)
    gender_cos_by_k[k] = gc; size_cos_by_k[k] = sc
    sentiment_cos_by_k[k] = sc2; age_cos_by_k[k] = ac; trans_cos_by_k[k] = tc
    print("  %-8d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-14.4f" % (
        k, gc, sc, sc2, ac, tc))

print()

# ====================================================================
# PHASE 2: Rate of change — where does alignment grow fastest?
# ====================================================================
print("Phase 2: Rate of change in EN-ZH gender axis cosine (Δcos per Δk)")
print()
print("  Identifying the 'universality knee' — where alignment grows fastest")
print()

prev_k = None; prev_gc = None
print("  %-8s  %-10s  %-10s  %-10s" % ("k range", "Δgender", "Δsize", "Δsentiment"))
print("  " + "-"*45)
prev = {ax: None for ax in ['g', 's', 'sent', 'age']}
DIMS2 = [d for d in DIMS if d != 1536]
for i in range(1, len(DIMS2)):
    k0, k1 = DIMS2[i-1], DIMS2[i]
    dg = gender_cos_by_k[k1] - gender_cos_by_k[k0]
    ds = size_cos_by_k[k1] - size_cos_by_k[k0]
    dsen = sentiment_cos_by_k[k1] - sentiment_cos_by_k[k0]
    mark = " ← MAX" if dg > 0.05 else ""
    print("  %-8s  %-10.4f  %-10.4f  %-10.4f%s" % (
        "%d→%d" % (k0, k1), dg, ds, dsen, mark))

print()

# ====================================================================
# PHASE 3: Per-translation-pair cosine curve
# ====================================================================
print("Phase 3: Per-translation-pair cosine at selected k values")
print("  Reveals which concept types align early vs late")
print()

K_SAMPLE = [5, 20, 50, 100, 150, 200, 300, 500, 1536]
TRANS_SAMPLE = [
    ('king','国王'),('man','男人'),('father','父亲'),('mother','母亲'),
    ('happy','快乐'),('sad','悲伤'),('good','好'),('bad','坏'),
    ('big','大'),('small','小'),('cat','猫'),('love','爱'),
    ('hate','恨'),('red','红'),('blue','蓝'),('young','年轻'),
]

# Cache projections for k in K_SAMPLE
W_projs = {}
for k in K_SAMPLE:
    if k == 1536: W_projs[k] = W_E.copy()
    else: W_projs[k] = project_k(k)

print("  %-14s  %s" % ("pair", "  ".join("%-8s" % ("k=%d" % k) for k in K_SAMPLE)))
print("  " + "-"*(14 + 10*len(K_SAMPLE)))

for en_w, zh_w in TRANS_SAMPLE:
    en_i = get_idx(en_w); zh_i = get_idx(zh_w, zh=True)
    if en_i is None or zh_i is None: print("  %-14s  N/A" % ("%s/%s"%(en_w,zh_w))); continue
    row = []
    for k in K_SAMPLE:
        W = W_projs[k]
        c = float(np.dot(normed(W[en_i]).astype(np.float32), normed(W[zh_i]).astype(np.float32)))
        row.append("%-8.3f" % c)
    print("  %-14s  %s" % ("%s/%s" % (en_w, zh_w), "  ".join(row)))

print()

# ====================================================================
# PHASE 4: Axis coherence vs k for each language
# ====================================================================
print("Phase 4: Axis coherence (intra-language consistency) vs k")
print()
print("  %-8s  %-12s  %-12s  %-12s  %-12s" % (
    "k", "EN_gender", "ZH_gender", "EN_sentiment", "ZH_sentiment"))
print("  " + "-"*60)

for k in [5, 10, 20, 50, 75, 100, 150, 200, 300, 500, 1536]:
    if k == 1536: W_proj = W_E.copy()
    else: W_proj = project_k(k)
    eg = axis_coherence(W_proj, EN_GENDER,    zh=False)
    zg = axis_coherence(W_proj, ZH_GENDER,    zh=True)
    es = axis_coherence(W_proj, EN_SENTIMENT, zh=False)
    zs = axis_coherence(W_proj, ZH_SENTIMENT, zh=True)
    print("  %-8d  %-12.4f  %-12.4f  %-12.4f  %-12.4f" % (k, eg, zg, es, zs))

print()

# ====================================================================
# PHASE 5: What PCs 50-200 encode — interpret the "universality zone"
# ====================================================================
print("Phase 5: What information lives in PCs 50-200 (universality zone)?")
print()
print("  Testing: does adding PCs 51-200 incrementally to PCs 1-50")
print("  explain the jump in cross-lingual alignment?")
print()

# Build additive projection: start with k=50, add PCs one band at a time
BANDS = [(1, 50), (51, 75), (76, 100), (101, 125), (126, 150), (151, 175), (176, 200),
         (201, 250), (251, 300), (301, 400), (401, 500)]

print("  %-14s  %-12s  %-12s  %-12s" % ("PC band", "gender_cos", "trans_cos", "ZH_frac_good"))
print("  " + "-"*56)

# ZH fraction in top-20 of 'good'
good_idx = get_idx('good')
W50n_all = (lambda W: np.array([normed(v) for v in W], dtype=np.float32))(project_k(50))

for k_start, k_end in BANDS:
    if k_end <= 500:
        W_proj = project_k(k_end)
    else:
        W_proj = W_E.copy()
    gc = cross_ax_cos(W_proj, EN_GENDER, ZH_GENDER)
    tc = mean_translation_cos(W_proj, TRANSLATION_PAIRS)
    # ZH fraction in top-20 of 'good'
    Wn = np.array([normed(v) for v in W_proj], dtype=np.float32)
    sims = Wn @ normed(W_proj[good_idx]).astype(np.float32)
    sims[~(EN_MASK | ZH_MASK)] = -1.0; sims[good_idx] = -1.0
    top20 = np.argsort(sims)[::-1][:20]
    zh_frac = 100*sum(1 for i in top20 if ZH_MASK[i])/20
    print("  %-14s  %-12.4f  %-12.4f  %-12.1f%%" % (
        "k=1-%d" % k_end, gc, tc, zh_frac))

print()

# ====================================================================
# PHASE 6: Incremental contribution — which PCs drive cross-lingual alignment?
# ====================================================================
print("Phase 6: Per-PC contribution to EN-ZH gender axis cosine")
print("  (how much does adding each single PC change the cross-lingual alignment?)")
print()

# Efficient: project onto cumulative top-k and measure gender axis cos
# We sample k=1,5,10,15,20,30,40,50,60,70,80,90,100,...,200

TEST_KS = list(range(1, 21)) + list(range(25, 105, 5)) + list(range(110, 210, 10))

print("  %-6s  %-10s  %-10s  %-10s" % ("k", "gender", "size", "sentiment"))
print("  " + "-"*42)

prev_gc = prev_sc = prev_sc2 = 0.0
max_delta_gc = 0.0; max_delta_k = 0
for k in TEST_KS:
    W_proj = project_k(k)
    gc  = cross_ax_cos(W_proj, EN_GENDER,    ZH_GENDER)
    sc  = cross_ax_cos(W_proj, EN_SIZE,      ZH_SIZE)
    sc2 = cross_ax_cos(W_proj, EN_SENTIMENT, ZH_SENTIMENT)
    delta_gc = gc - prev_gc
    if abs(delta_gc) > abs(max_delta_gc): max_delta_gc = delta_gc; max_delta_k = k
    mark = " ← largest jump" if k == max_delta_k and k > 1 else ""
    print("  %-6d  %-10.4f  %-10.4f  %-10.4f%s" % (k, gc, sc, sc2, mark))
    prev_gc = gc; prev_sc = sc; prev_sc2 = sc2

print()
print("  Largest single-PC jump in gender axis cosine: k=%d (Δ=%.4f)" % (
    max_delta_k, max_delta_gc))
print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 362 — Cross-Lingual Universality Knee")
print("="*70)
print()
print("  EN-ZH axis cosine at key checkpoints:")
print("  %-8s  %-10s  %-10s  %-10s  %-12s" % (
    "k", "gender", "size", "sentiment", "trans_cos"))
for k in [5, 50, 100, 200, 500, 1536]:
    if k in gender_cos_by_k:
        print("  %-8d  %-10.4f  %-10.4f  %-10.4f  %-12.4f" % (
            k, gender_cos_by_k[k], size_cos_by_k.get(k, float('nan')),
            sentiment_cos_by_k.get(k, float('nan')), trans_cos_by_k.get(k, float('nan'))))
print()
print("  Zone 3 (k=50-200) hypothesis: EN-ZH alignment jumps here")
print("  if confirmed by Phase 1 and Phase 6 results.")
