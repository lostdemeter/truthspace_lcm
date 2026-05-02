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

print("\nDAY 364: The Gender Universality Zone — PCs 51-125")
print("="*70)
print("What linguistic content lives in the mid-range PCs that drive")
print("cross-lingual gender axis alignment from 0.27 to 0.64?")
print()

print("Computing SVD (k=150)...")
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)
U, S, Vt = randomized_svd(W_cent, n_components=150, random_state=42)
PC = [Vt[i].astype(np.float64) for i in range(150)]
all_scores = [(W_E - W_mean).astype(np.float64) @ PC[i] for i in range(150)]
print("  Done. σ range: %.2f → %.4f" % (S[0], S[149]))
print()

EN_GENDER = [
    ('king','queen'),('man','woman'),('boy','girl'),
    ('father','mother'),('son','daughter'),('husband','wife'),
    ('uncle','aunt'),('prince','princess'),('actor','actress'),('waiter','waitress'),
]
ZH_GENDER = [
    ('男人','女人'),('国王','女王'),('父亲','母亲'),('儿子','女儿'),
    ('丈夫','妻子'),('叔叔','阿姨'),('王子','公主'),('男孩','女孩'),
]
EN_SENTIMENT = [
    ('bad','good'),('ugly','beautiful'),('hate','love'),('sad','happy'),
    ('dark','bright'),('wrong','right'),
]
ZH_SENTIMENT = [('坏','好'),('丑','美'),('恨','爱'),('悲伤','快乐')]

def build_axis(pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_E[ti] - W_E[si]
        n = np.linalg.norm(d)
        if n > 1e-8: chords.append(d / n)
    if not chords: return None
    m = np.mean(chords, axis=0)
    return m / (np.linalg.norm(m) + 1e-8)

ax_en_gender = build_axis(EN_GENDER, zh=False)
ax_zh_gender = build_axis(ZH_GENDER, zh=True)
ax_en_sent   = build_axis(EN_SENTIMENT, zh=False)
ax_zh_sent   = build_axis(ZH_SENTIMENT, zh=True)

# ====================================================================
# PHASE 1: Vocabulary extremes for each PC in range 50-125
# ====================================================================
print("Phase 1: Vocabulary extremes for key PCs in the gender universality zone")
print("  Identify what linguistic feature each PC encodes")
print()

PROBE_PCS = [50, 51, 60, 70, 75, 80, 90, 100, 110, 115, 120, 125]

for pc_idx in PROBE_PCS:
    scores = all_scores[pc_idx]
    en_scored = [(i, scores[i]) for i in range(len(W_E)) if EN_MASK[i]]
    zh_scored = [(i, scores[i]) for i in range(len(W_E)) if ZH_MASK[i]]
    en_scored.sort(key=lambda x: -x[1])
    zh_scored.sort(key=lambda x: -x[1])

    # Cross-lingual gender cosine contribution of this PC
    en_comp = float(ax_en_gender @ PC[pc_idx])
    zh_comp = float(ax_zh_gender @ PC[pc_idx])
    same = "+" if np.sign(en_comp) == np.sign(zh_comp) else "-"
    contrib = en_comp * zh_comp  # positive = helps cross-lingual alignment

    print("  PC%d  (σ=%.4f)  gender[EN=%.4f, ZH=%.4f, %s, contrib=%.5f]" % (
        pc_idx+1, S[pc_idx], en_comp, zh_comp, same, contrib))

    top5_en = [tok.decode([i]).strip() for i,_ in en_scored[:5]]
    bot5_en = [tok.decode([i]).strip() for i,_ in en_scored[-5:]]
    top5_zh = [tok.decode([i]).strip() for i,_ in zh_scored[:5]]
    bot5_zh = [tok.decode([i]).strip() for i,_ in zh_scored[-5:]]

    print("    EN+: %-40s  ZH+: %s" % (str(top5_en), str(top5_zh)))
    print("    EN-: %-40s  ZH-: %s" % (str(bot5_en), str(bot5_zh)))
    print()

# ====================================================================
# PHASE 2: Per-PC contribution to cross-lingual gender cosine
# ====================================================================
print("Phase 2: Per-PC contribution to cross-lingual gender cosine (PCs 40-130)")
print("  contrib = (EN_gender · PC_i) × (ZH_gender · PC_i)")
print("  Positive contrib = this PC helps EN-ZH alignment")
print("  Negative contrib = this PC hurts EN-ZH alignment")
print()

print("  %-6s  %-10s  %-10s  %-10s  %-10s" % (
    "PC", "EN_proj", "ZH_proj", "contrib", "cumul_cos"))
print("  " + "-"*52)

# Full-space gender cosine decomposed per PC:
# cos(ax_EN, ax_ZH) = sum_i (ax_EN · v_i)(ax_ZH · v_i) + residual
# where residual = ax_EN_rest · ax_ZH_rest (orthogonal to top 150 PCs)

en_g_coords = np.array([float(ax_en_gender @ PC[i]) for i in range(150)])
zh_g_coords = np.array([float(ax_zh_gender @ PC[i]) for i in range(150)])
contribs = en_g_coords * zh_g_coords
cumul = np.cumsum(contribs)

for i in range(39, 130):
    mark = ""
    if contribs[i] > 0.002: mark = " ← +LARGE"
    elif contribs[i] < -0.002: mark = " ← -LARGE"
    if i in [49, 74, 99, 109, 119, 124] or abs(contribs[i]) > 0.002:
        print("  %-6d  %-10.5f  %-10.5f  %-10.5f  %-10.5f%s" % (
            i+1, en_g_coords[i], zh_g_coords[i], contribs[i], cumul[i], mark))

print()
print("  Full-space gender cos (approx from top-150 PCs): %.5f" % cumul[-1])
print("  Actual full-space gender cos: %.5f" % float(
    np.dot(ax_en_gender.astype(np.float32), ax_zh_gender.astype(np.float32))))
print()

# Find the top-10 contributing PCs in range 50-130
top_contribs = sorted([(i, contribs[i]) for i in range(49, 130)], key=lambda x: -x[1])
print("  Top-10 contributing PCs (range 50-130) for gender universality:")
for i, c in top_contribs[:10]:
    print("    PC%d  contrib=%.5f  EN_proj=%.5f  ZH_proj=%.5f" % (
        i+1, c, en_g_coords[i], zh_g_coords[i]))
print()
print("  Top-10 ANTI-contributing PCs (range 50-130):")
for i, c in sorted(top_contribs, key=lambda x: x[1])[:10]:
    print("    PC%d  contrib=%.5f  EN_proj=%.5f  ZH_proj=%.5f" % (
        i+1, c, en_g_coords[i], zh_g_coords[i]))
print()

# ====================================================================
# PHASE 3: Do the gender-universal PCs also carry gender signal
#           for other languages? (French, Spanish, Japanese)
# ====================================================================
print("Phase 3: Do the gender-universal PCs also carry signal for other languages?")
print("  Testing Romance (FR, ES) and Japanese pronoun gender pairs")
print()

FR_GENDER = [
    ('homme','femme'),('garcon','fille'),('fils','fille'),('roi','reine'),
    ('pere','mere'),('mari','femme'),('prince','princesse'),('acteur','actrice'),
]
ES_GENDER = [
    ('hombre','mujer'),('chico','chica'),('hijo','hija'),('rey','reina'),
    ('padre','madre'),('principe','princesa'),('actor','actriz'),
]
JA_GENDER = [
    ('男','女'),('彼','彼女'),
]
AR_GENDER = [
    ('رجل','امرأة'),('ولد','بنت'),('أب','أم'),('زوج','زوجة'),
]

def count_single_token(pairs, zh=False):
    ok = sum(1 for s,t in pairs
             if get_idx(s, zh=zh) is not None and get_idx(t, zh=zh) is not None)
    return ok, len(pairs)

# Check coverage
for lang, pairs, zh in [("FR", FR_GENDER, False), ("ES", ES_GENDER, False),
                         ("JA", JA_GENDER, True), ("AR", AR_GENDER, True)]:
    ok, total = count_single_token(pairs, zh=zh)
    print("  %s single-token coverage: %d/%d" % (lang, ok, total))

print()

# Build axes for each language
def build_axis_raw(pairs, zh=False):
    chords = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_E[ti] - W_E[si]
        n = np.linalg.norm(d)
        if n > 1e-8: chords.append(d / n)
    if len(chords) < 2: return None
    m = np.mean(chords, axis=0)
    return m / (np.linalg.norm(m) + 1e-8)

axes = {}
for lang, pairs, zh in [("EN", EN_GENDER, False), ("ZH", ZH_GENDER, True),
                          ("FR", FR_GENDER, False), ("ES", ES_GENDER, False),
                          ("JA", JA_GENDER, True), ("AR", AR_GENDER, True)]:
    ax = build_axis_raw(pairs, zh=zh)
    if ax is not None: axes[lang] = ax
    print("  %s axis built: %s" % (lang, "YES" if ax is not None else "NO"))

print()

# Cross-lingual cosine matrix
print("  Pairwise cross-lingual gender axis cosine:")
langs = [k for k in ["EN","ZH","FR","ES","JA","AR"] if k in axes]
print("  %-6s" % "", end="")
for l in langs: print("  %-8s" % l, end="")
print()
for l1 in langs:
    print("  %-6s" % l1, end="")
    for l2 in langs:
        c = float(np.dot(axes[l1].astype(np.float32), axes[l2].astype(np.float32)))
        print("  %-8.4f" % c, end="")
    print()
print()

# Project each language axis onto the top-5 gender-contributing PCs
print("  Projection of each language gender axis onto top-5 contributing PCs (50-130):")
print()
top5_pcs = [i for i,c in top_contribs[:5]]
print("  %-6s" % "lang", end="")
for i in top5_pcs: print("  PC%-5d" % (i+1), end="")
print()
for lang in langs:
    print("  %-6s" % lang, end="")
    for i in top5_pcs:
        proj = float(axes[lang] @ PC[i])
        print("  %-7.4f" % proj, end="")
    print()
print()

# ====================================================================
# PHASE 4: What kind of words score high/low on the top gender-universal PCs?
# ====================================================================
print("Phase 4: Vocabulary analysis of the top gender-contributing PCs")
print("  Do these PCs separate masculine/feminine vocabulary?")
print()

# For each top-5 PC in the gender zone, show vocabulary extremes
# AND test: do EN masculine words score higher/lower than EN feminine words?

EN_MASC = ['king', 'man', 'boy', 'father', 'son', 'husband', 'uncle', 'prince',
           'actor', 'waiter', 'brother', 'grandfather', 'nephew', 'groom']
EN_FEM  = ['queen', 'woman', 'girl', 'mother', 'daughter', 'wife', 'aunt', 'princess',
           'actress', 'waitress', 'sister', 'grandmother', 'niece', 'bride']
ZH_MASC = ['男人', '国王', '父亲', '儿子', '丈夫', '叔叔', '王子', '男孩', '兄弟', '祖父']
ZH_FEM  = ['女人', '女王', '母亲', '女儿', '妻子', '阿姨', '公主', '女孩', '姐妹', '祖母']

print("  Mean score of masculine vs feminine words on top gender-contributing PCs:")
print()

for pc_rank, (pc_idx, contrib) in enumerate(top_contribs[:5]):
    scores = all_scores[pc_idx]
    en_masc_s = [scores[get_idx(w)] for w in EN_MASC if get_idx(w) is not None]
    en_fem_s  = [scores[get_idx(w)] for w in EN_FEM  if get_idx(w) is not None]
    zh_masc_s = [scores[get_idx(w, zh=True)] for w in ZH_MASC if get_idx(w, zh=True) is not None]
    zh_fem_s  = [scores[get_idx(w, zh=True)] for w in ZH_FEM  if get_idx(w, zh=True) is not None]

    en_sep = np.mean(en_masc_s) - np.mean(en_fem_s) if en_masc_s and en_fem_s else 0
    zh_sep = np.mean(zh_masc_s) - np.mean(zh_fem_s) if zh_masc_s and zh_fem_s else 0

    # Full vocabulary extremes
    en_scored = [(i, scores[i]) for i in range(len(W_E)) if EN_MASK[i]]
    zh_scored = [(i, scores[i]) for i in range(len(W_E)) if ZH_MASK[i]]
    en_scored.sort(key=lambda x: -x[1])
    zh_scored.sort(key=lambda x: -x[1])

    print("  PC%d (rank %d contributing, contrib=%.5f)" % (pc_idx+1, pc_rank+1, contrib))
    print("    EN: masc_mean=%.4f  fem_mean=%.4f  Δ=%.4f" % (
        np.mean(en_masc_s) if en_masc_s else 0,
        np.mean(en_fem_s)  if en_fem_s  else 0, en_sep))
    print("    ZH: masc_mean=%.4f  fem_mean=%.4f  Δ=%.4f" % (
        np.mean(zh_masc_s) if zh_masc_s else 0,
        np.mean(zh_fem_s)  if zh_fem_s  else 0, zh_sep))
    print("    EN extremes+: %s" % [tok.decode([i]).strip() for i,_ in en_scored[:8]])
    print("    EN extremes-: %s" % [tok.decode([i]).strip() for i,_ in en_scored[-8:]])
    print("    ZH extremes+: %s" % [tok.decode([i]).strip() for i,_ in zh_scored[:8]])
    print("    ZH extremes-: %s" % [tok.decode([i]).strip() for i,_ in zh_scored[-8:]])
    print()

# ====================================================================
# PHASE 5: Does gender signal on each PC predict gender?
# ====================================================================
print("Phase 5: Is each contributing PC a 'gender axis' in the traditional sense?")
print("  For each PC, use the PC direction as axis and measure nearest-neighbor accuracy")
print()
print("  %-6s  %-12s  %-12s  %-12s" % ("PC", "EN_acc(top5)", "ZH_acc(top5)", "cross_cos"))
print("  " + "-"*48)

W_n = np.array([normed(v) for v in W_E], dtype=np.float32)

for pc_idx, contrib in top_contribs[:8]:
    ax_dir = PC[pc_idx].astype(np.float32)
    # EN accuracy
    en_hits = 0; en_total = 0
    for src, tgt in EN_GENDER:
        si = get_idx(src); ti = get_idx(tgt)
        if si is None or ti is None: continue
        en_total += 1
        # Scale: use mean chord magnitude along this PC
        scale = abs(float((W_E[ti] - W_E[si]) @ PC[pc_idx]))
        pred = normed(W_E[si].astype(np.float32) + scale * ax_dir)
        sims = W_n @ pred
        sims[~EN_MASK] = -1.0; sims[si] = -1.0
        top5 = np.argsort(sims)[::-1][:5]
        if ti in top5: en_hits += 1
    # ZH accuracy
    zh_hits = 0; zh_total = 0
    for src, tgt in ZH_GENDER:
        si = get_idx(src, zh=True); ti = get_idx(tgt, zh=True)
        if si is None or ti is None: continue
        zh_total += 1
        scale = abs(float((W_E[ti] - W_E[si]) @ PC[pc_idx]))
        pred = normed(W_E[si].astype(np.float32) + scale * ax_dir)
        sims = W_n @ pred
        sims[~ZH_MASK] = -1.0; sims[si] = -1.0
        top5 = np.argsort(sims)[::-1][:5]
        if ti in top5: zh_hits += 1
    en_acc = en_hits/en_total if en_total > 0 else 0
    zh_acc = zh_hits/zh_total if zh_total > 0 else 0
    cross = float(np.dot(axes.get("EN", ax_dir).astype(np.float32), ax_dir))
    print("  %-6d  %-12.3f  %-12.3f  %-12.5f" % (
        pc_idx+1, en_acc, zh_acc, contrib))

print()

# ====================================================================
# PHASE 6: Incremental gender cosine — which INDIVIDUAL PCs matter most?
# ====================================================================
print("Phase 6: Incremental cross-lingual gender cosine — per-PC isolated contribution")
print("  Projecting ONLY onto a single PC — what does each PC 'say' about cross-lingual gender?")
print()
print("  Note: sum of per-PC contributions ≠ full-space cosine (off-diagonal terms)")
print()
print("  %-6s  %-10s  %-10s  %-10s" % ("PC", "contrib", "EN_gend", "ZH_gend"))
print("  " + "-"*40)

top10_zone = sorted([(i, contribs[i]) for i in range(49, 130)], key=lambda x: -abs(x[1]))
for pc_idx, contrib in top10_zone[:15]:
    print("  %-6d  %-10.5f  %-10.5f  %-10.5f" % (
        pc_idx+1, contrib, en_g_coords[pc_idx], zh_g_coords[pc_idx]))

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 364 — The Gender Universality Zone (PCs 51-125)")
print("="*70)
print()

print("  Cross-lingual gender cosine from top-150 PCs: %.5f" % cumul[-1])
print()
print("  Top-5 contributing PCs in zone 50-130:")
for i, c in top_contribs[:5]:
    print("    PC%d  contrib=%.5f  σ=%.4f" % (i+1, c, S[i]))
print()
print("  Interpretation: these mid-range PCs encode [see Phase 1 vocabulary extremes]")
print("  The gender universality zone is about [grammatical structure / syntax /")
print("  animacy / argument structure] — not frequency (PC1) nor identity (PC200+)")
