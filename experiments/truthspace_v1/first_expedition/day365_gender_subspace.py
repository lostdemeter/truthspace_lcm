import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

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

print("\nDAY 365: Learned Cross-Lingual Gender Subspace via SVD on Chord Differences")
print("="*70)
print("If gender is distributed across all 1536 dims, can we find the optimal")
print("cross-lingual subspace by doing SVD directly on gender chord differences?")
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
EN_SIZE = [
    ('small','big'),('little','large'),('tiny','huge'),('narrow','wide'),
    ('short','tall'),('shallow','deep'),('thin','thick'),('weak','strong'),
]
ZH_SIZE = [('小','大'),('短','长'),('低','高'),('薄','厚'),('弱','强')]
EN_SENTIMENT = [
    ('bad','good'),('ugly','beautiful'),('hate','love'),('sad','happy'),
    ('dark','bright'),('wrong','right'),('evil','good'),('poor','rich'),
]
ZH_SENTIMENT = [('坏','好'),('丑','美'),('恨','爱'),('悲伤','快乐'),('暗','亮')]

# Build chord matrices for each language/axis
def get_chords(pairs, zh=False):
    chords = []
    valid = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_E[ti] - W_E[si]
        n = np.linalg.norm(d)
        if n > 1e-8:
            chords.append(d / n)
            valid.append((s, t, si, ti))
    return np.array(chords, dtype=np.float64), valid

def mean_axis(chords):
    m = chords.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-8)

# ====================================================================
# PHASE 1: Baseline — single-axis (mean chord) cross-lingual cosine
# ====================================================================
print("Phase 1: Baseline single-axis approach (mean chord)")
print()

for ax_name, en_pairs, zh_pairs in [
    ("gender", EN_GENDER, ZH_GENDER),
    ("size",   EN_SIZE,   ZH_SIZE),
    ("sentiment", EN_SENTIMENT, ZH_SENTIMENT),
]:
    en_chords, _ = get_chords(en_pairs, zh=False)
    zh_chords, _ = get_chords(zh_pairs, zh=True)
    ax_en = mean_axis(en_chords)
    ax_zh = mean_axis(zh_chords)
    cos_val = float(np.dot(ax_en.astype(np.float32), ax_zh.astype(np.float32)))
    print("  %-14s  EN_chords=%d  ZH_chords=%d  EN-ZH cos=%.5f" % (
        ax_name, len(en_chords), len(zh_chords), cos_val))

print()

# ====================================================================
# PHASE 2: Learn the optimal cross-lingual gender subspace
# ====================================================================
print("Phase 2: Learning cross-lingual gender subspace via SVD")
print()
print("  Strategy 1: SVD on COMBINED chord matrix (EN + ZH chords stacked)")
print("  Strategy 2: Cross-covariance SVD: C = EN_chords.T @ ZH_chords")
print("  Strategy 3: Joint SVD of mean chord matrix")
print()

en_chords_g, en_valid_g = get_chords(EN_GENDER, zh=False)
zh_chords_g, zh_valid_g = get_chords(ZH_GENDER, zh=True)

# Strategy 1: SVD on combined chord matrix
combined_g = np.vstack([en_chords_g, zh_chords_g])
U1, S1, Vt1 = np.linalg.svd(combined_g, full_matrices=False)
print("  Strategy 1: Combined SVD singular values:")
print("    ", S1[:10])
print()

# Strategy 2: Cross-covariance SVD C = EN^T @ ZH
# But EN and ZH have different numbers of chords — pad with zeros or use CCA
# For cross-covariance: C[i,j] = correlation between EN chord component i and ZH chord component j
# This is equivalent to: do SVD of EN^T ZH (if same number of chords, use direct product)
# With different numbers, use E[en · zh] type formulation
# Cross-covariance matrix: C = (1/n) * sum_k (en_k ⊗ zh_k) won't work directly
# Better: C = EN.T @ EN_to_ZH_mapped
# Actually the cleanest cross-covariance approach: CCA
# But for simplicity, use the outer product: average over random pairings

# For cross-covariance, just use the matrix product of means:
# The cross-covariance between EN chord space and ZH chord space:
en_n = len(en_chords_g); zh_n = len(zh_chords_g)
# Compute pair-wise products: for each EN chord, for each ZH chord
# C = sum_{i,j} en_i ⊗ zh_j / (en_n * zh_n) = (mean en) ⊗ (mean zh)
# But that's just the outer product of means — same as CCA rank-1
# More informative: SVD of the direct "cross-chord" matrix
# by pairing each EN chord with corresponding ZH chord (pad shorter list)
min_n = min(en_n, zh_n)
cross_mat = en_chords_g[:min_n].T @ zh_chords_g[:min_n] / min_n
U2, S2, Vt2 = np.linalg.svd(cross_mat)
print("  Strategy 2: Cross-covariance SVD singular values:")
print("    ", S2[:10])
print()

# Strategy 3: Joint PCA of gender axes
# For each language, compute the chord matrix; SVD of stacked matrix with equal weight
en_w = en_chords_g / en_n
zh_w = zh_chords_g / zh_n
joint = np.vstack([en_w, zh_w])
U3, S3, Vt3 = np.linalg.svd(joint, full_matrices=False)
print("  Strategy 3: Equal-weight joint SVD singular values:")
print("    ", S3[:10])
print()

# ====================================================================
# PHASE 3: Multi-dimensional subspace transfer accuracy
# ====================================================================
print("Phase 3: Multi-dimensional subspace transfer — which strategy works best?")
print()
print("  For each strategy and dimensionality k, we:")
print("  1. Project EN gender axis onto k-dim subspace")
print("  2. Use the projected direction for zero-shot ZH transfer")
print("  3. Measure top-5 accuracy on ZH pairs")
print()

W_n = np.array([normed(v) for v in W_E], dtype=np.float32)

def subspace_transfer_acc(pairs_src, pairs_tgt, subspace_dirs, zh_src=False, zh_tgt=True, k=5):
    """
    Build axis from src pairs projected onto subspace_dirs,
    then use that projected direction to transfer to tgt language.
    """
    # Build src axis
    chords_src, _ = get_chords(pairs_src, zh=zh_src)
    if len(chords_src) == 0: return 0.0, 0
    ax_src_full = mean_axis(chords_src)

    # Project onto subspace
    coords = np.array([float(ax_src_full @ d) for d in subspace_dirs])
    ax_src_proj = sum(c * d for c, d in zip(coords, subspace_dirs))
    n = np.linalg.norm(ax_src_proj)
    if n < 1e-8: return 0.0, 0
    ax_src_proj = ax_src_proj / n

    # Evaluate on tgt pairs
    mask = ZH_MASK if zh_tgt else EN_MASK
    hits = 0; total = 0
    for s, t in pairs_tgt:
        si = get_idx(s, zh=zh_tgt); ti = get_idx(t, zh=zh_tgt)
        if si is None or ti is None: continue
        total += 1
        # Find best scale on src pairs
        chord_src_along_proj = [
            float((W_E[get_idx(tgt, zh=zh_src)] - W_E[get_idx(src, zh=zh_src)]) @ ax_src_proj)
            for src, tgt in pairs_src
            if get_idx(src, zh=zh_src) is not None and get_idx(tgt, zh=zh_src) is not None
        ]
        if not chord_src_along_proj: continue
        scale = abs(np.mean(chord_src_along_proj))

        # Use same scale for tgt (zero-shot)
        pred = normed((W_E[si] + scale * ax_src_proj).astype(np.float32))
        sims = W_n @ pred
        sims[~mask] = -1.0; sims[si] = -1.0
        top_k = np.argsort(sims)[::-1][:k]
        if ti in top_k: hits += 1
    return (hits/total) if total > 0 else 0.0, total

def full_axis_acc(pairs_src, pairs_eval, zh_src=False, zh_eval=True, k=5):
    chords_src, _ = get_chords(pairs_src, zh=zh_src)
    if len(chords_src) == 0: return 0.0, 0
    ax = mean_axis(chords_src)
    mask = ZH_MASK if zh_eval else EN_MASK
    hits = 0; total = 0
    for s, t in pairs_eval:
        si = get_idx(s, zh=zh_eval); ti = get_idx(t, zh=zh_eval)
        if si is None or ti is None: continue
        total += 1
        chord_along = [
            float((W_E[get_idx(tgt, zh=zh_src)] - W_E[get_idx(src, zh=zh_src)]) @ ax)
            for src, tgt in pairs_src
            if get_idx(src, zh=zh_src) is not None and get_idx(tgt, zh=zh_src) is not None
        ]
        scale = abs(np.mean(chord_along)) if chord_along else 1.0
        pred = normed((W_E[si] + scale * ax).astype(np.float32))
        sims = W_n @ pred
        sims[~mask] = -1.0; sims[si] = -1.0
        top_k = np.argsort(sims)[::-1][:k]
        if ti in top_k: hits += 1
    return (hits/total) if total > 0 else 0.0, total

# Full-space baseline: EN axis → transfer to ZH
base_acc, base_n = full_axis_acc(EN_GENDER, ZH_GENDER, zh_src=False, zh_eval=True, k=5)
base_rev, _ = full_axis_acc(ZH_GENDER, EN_GENDER, zh_src=True, zh_eval=False, k=5)
print("  BASELINE (full-space EN axis → ZH zero-shot top-5): %.3f (%d pairs)" % (base_acc, base_n))
print("  BASELINE (full-space ZH axis → EN zero-shot top-5): %.3f" % base_rev)
print()

# Now test subspace strategies at various k
print("  %-12s  %-6s  %-12s  %-12s" % ("Strategy", "dims", "EN→ZH acc", "ZH→EN acc"))
print("  " + "-"*44)

for strat_name, dirs in [("Combined",    list(Vt1)),
                          ("CrossCov",    list(Vt2)),
                          ("Joint",       list(Vt3))]:
    for n_dims in [1, 2, 3, 5, 10, 20, 50]:
        sub = dirs[:n_dims]
        acc_en_zh, _ = subspace_transfer_acc(EN_GENDER, ZH_GENDER, sub, zh_src=False, zh_tgt=True, k=5)
        acc_zh_en, _ = subspace_transfer_acc(ZH_GENDER, EN_GENDER, sub, zh_src=True, zh_tgt=False, k=5)
        better = " *" if acc_en_zh > base_acc else ""
        print("  %-12s  %-6d  %-12.3f  %-12.3f%s" % (
            strat_name, n_dims, acc_en_zh, acc_zh_en, better))
    print()

# ====================================================================
# PHASE 4: The cross-lingual gender subspace directions — what are they?
# ====================================================================
print("Phase 4: Interpreting the learned cross-lingual gender subspace")
print("  (Using Strategy 1: Combined SVD as it is simplest)")
print()

# For top-3 combined SVD directions, show:
# 1. Cosine with EN/ZH mean axes
# 2. Mean masculine vs feminine score
# 3. Cross-lingual cosine if used as single axis
EN_MASC = ['king', 'man', 'boy', 'father', 'son', 'husband', 'uncle', 'prince', 'actor', 'waiter']
EN_FEM  = ['queen', 'woman', 'girl', 'mother', 'daughter', 'wife', 'aunt', 'princess', 'actress', 'waitress']
ZH_MASC = ['男人', '国王', '父亲', '儿子', '丈夫', '叔叔', '王子', '男孩']
ZH_FEM  = ['女人', '女王', '母亲', '女儿', '妻子', '阿姨', '公主', '女孩']

ax_en_g = mean_axis(en_chords_g)
ax_zh_g = mean_axis(zh_chords_g)

print("  %-6s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s" % (
    "Dir", "cos(EN)", "cos(ZH)", "EN_Δ(m-f)", "ZH_Δ(m-f)", "EN_acc", "ZH_acc"))
print("  " + "-"*66)

for i, d in enumerate(Vt1[:6]):
    cos_en = float(np.dot(d.astype(np.float32), ax_en_g.astype(np.float32)))
    cos_zh = float(np.dot(d.astype(np.float32), ax_zh_g.astype(np.float32)))

    en_masc_s = [float(W_E[get_idx(w)] @ d) for w in EN_MASC if get_idx(w) is not None]
    en_fem_s  = [float(W_E[get_idx(w)] @ d) for w in EN_FEM  if get_idx(w) is not None]
    zh_masc_s = [float(W_E[get_idx(w, zh=True)] @ d) for w in ZH_MASC if get_idx(w, zh=True) is not None]
    zh_fem_s  = [float(W_E[get_idx(w, zh=True)] @ d) for w in ZH_FEM  if get_idx(w, zh=True) is not None]

    en_delta = np.mean(en_masc_s) - np.mean(en_fem_s) if en_masc_s and en_fem_s else 0
    zh_delta = np.mean(zh_masc_s) - np.mean(zh_fem_s) if zh_masc_s and zh_fem_s else 0

    # Accuracy using this single direction
    acc_en, _ = subspace_transfer_acc(EN_GENDER, EN_GENDER, [d], zh_src=False, zh_tgt=False, k=5)
    acc_zh, _ = subspace_transfer_acc(ZH_GENDER, ZH_GENDER, [d], zh_src=True, zh_tgt=True, k=5)

    print("  %-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.3f  %-10.3f" % (
        i+1, cos_en, cos_zh, en_delta, zh_delta, acc_en, acc_zh))

print()

# ====================================================================
# PHASE 5: Cross-covariance SVD = CCA — the true cross-lingual axes
# ====================================================================
print("Phase 5: Cross-covariance SVD directions — interpretive analysis")
print("  C = (1/n) EN^T ZH: singular vectors maximize EN-ZH correlation")
print("  Left singular vectors = EN directions most correlated with ZH")
print("  Right singular vectors = ZH directions most correlated with EN")
print()

print("  Correlation between top cross-cov pairs:")
for i in range(5):
    u_dir = U2[:, i]  # shape 1536 — EN direction
    v_dir = Vt2[i]    # shape 1536 — ZH direction
    cos_uv = float(np.dot(u_dir.astype(np.float32), v_dir.astype(np.float32)))
    cos_en_g = float(np.dot(u_dir.astype(np.float32), ax_en_g.astype(np.float32)))
    cos_zh_g = float(np.dot(v_dir.astype(np.float32), ax_zh_g.astype(np.float32)))
    print("  CC%d  σ=%.5f  cos(u,v)=%.4f  cos(u, EN_axis)=%.4f  cos(v, ZH_axis)=%.4f" % (
        i+1, S2[i], cos_uv, cos_en_g, cos_zh_g))

print()

# ====================================================================
# PHASE 6: Does cross-lingual gender subspace also work for OTHER axes?
# ====================================================================
print("Phase 6: Is the learned gender subspace specific to gender,")
print("  or does it generalize to other semantic axes?")
print()

# Test if combining EN + ZH gender chords finds directions that also
# transfer other axes (size, sentiment)
gender_subspace = list(Vt1[:5])  # top-5 combined gender SVD directions

print("  Using top-5 Combined gender SVD directions as subspace:")
print()
print("  %-14s  %-12s  %-12s  %-12s" % ("Axis", "EN→ZH acc", "ZH→EN acc", "baseline_EN→ZH"))
print("  " + "-"*54)

for ax_name, en_pairs, zh_pairs, base_en_zh in [
    ("gender",    EN_GENDER,    ZH_GENDER,    base_acc),
    ("size",      EN_SIZE,      ZH_SIZE,      None),
    ("sentiment", EN_SENTIMENT, ZH_SENTIMENT, None),
]:
    acc_en_zh, _ = subspace_transfer_acc(en_pairs, zh_pairs, gender_subspace, zh_src=False, zh_tgt=True, k=5)
    acc_zh_en, _ = subspace_transfer_acc(zh_pairs, en_pairs, gender_subspace, zh_src=True, zh_tgt=False, k=5)
    base_val = base_en_zh if base_en_zh is not None else full_axis_acc(en_pairs, zh_pairs, zh_src=False, zh_eval=True, k=5)[0]
    print("  %-14s  %-12.3f  %-12.3f  %-12.3f" % (ax_name, acc_en_zh, acc_zh_en, base_val))

print()

# ====================================================================
# PHASE 7: The UNIVERSAL semantic subspace — combine all axes
# ====================================================================
print("Phase 7: Building a universal semantic subspace from ALL axes")
print("  Stack all semantic chords (EN+ZH gender+size+sentiment) → SVD")
print()

en_chords_sz, _ = get_chords(EN_SIZE,      zh=False)
zh_chords_sz, _ = get_chords(ZH_SIZE,      zh=True)
en_chords_st, _ = get_chords(EN_SENTIMENT, zh=False)
zh_chords_st, _ = get_chords(ZH_SENTIMENT, zh=True)

all_chords = np.vstack([
    en_chords_g, zh_chords_g,
    en_chords_sz, zh_chords_sz,
    en_chords_st, zh_chords_st,
])
print("  Total chords: %d  (dim=1536)" % len(all_chords))
U_all, S_all, Vt_all = np.linalg.svd(all_chords, full_matrices=False)
print("  Singular values:", S_all[:10])
print()

universal_sub = list(Vt_all)
print("  Zero-shot transfer using top-k universal subspace directions:")
print()
print("  %-10s  %-14s  %-14s  %-14s" % ("dims", "gender EN→ZH", "size EN→ZH", "sent EN→ZH"))
print("  " + "-"*54)
for n_dims in [1, 2, 3, 5, 10, 20]:
    sub = universal_sub[:n_dims]
    g_acc, _ = subspace_transfer_acc(EN_GENDER, ZH_GENDER, sub, zh_src=False, zh_tgt=True, k=5)
    s_acc, _ = subspace_transfer_acc(EN_SIZE, ZH_SIZE, sub, zh_src=False, zh_tgt=True, k=5)
    st_acc, _ = subspace_transfer_acc(EN_SENTIMENT, ZH_SENTIMENT, sub, zh_src=False, zh_tgt=True, k=5)
    print("  %-10d  %-14.3f  %-14.3f  %-14.3f" % (n_dims, g_acc, s_acc, st_acc))

print()
print("  Full-space baselines: gender=%.3f  size=%.3f  sentiment=%.3f" % (
    base_acc,
    full_axis_acc(EN_SIZE, ZH_SIZE, zh_src=False, zh_eval=True, k=5)[0],
    full_axis_acc(EN_SENTIMENT, ZH_SENTIMENT, zh_src=False, zh_eval=True, k=5)[0]
))
print()

# ====================================================================
# PHASE 8: The cross-lingual cosine of each universal subspace direction
# ====================================================================
print("Phase 8: Cross-lingual axis cosines of universal subspace directions")
print("  Which directions of the universal subspace carry cross-lingual gender?")
print()
print("  %-6s  %-10s  %-10s  %-10s  %-10s  %-10s" % (
    "Dir", "cos(EN_g)", "cos(ZH_g)", "cos(EN_sz)", "cos(ZH_sz)", "cross_g"))
print("  " + "-"*58)

ax_en_sz = mean_axis(en_chords_sz)
ax_zh_sz = mean_axis(zh_chords_sz)
for i, d in enumerate(Vt_all[:15]):
    d = d.astype(np.float32)
    ceg = float(np.dot(d, ax_en_g.astype(np.float32)))
    czg = float(np.dot(d, ax_zh_g.astype(np.float32)))
    ces = float(np.dot(d, ax_en_sz.astype(np.float32)))
    czs = float(np.dot(d, ax_zh_sz.astype(np.float32)))
    cross_g = ceg * czg
    print("  %-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.5f" % (
        i+1, ceg, czg, ces, czs, cross_g))

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 365 — Learned Cross-Lingual Gender Subspace")
print("="*70)
print()
print("  Key question: does an N-dim learned subspace beat the full-space")
print("  single mean axis for zero-shot cross-lingual transfer?")
print()
print("  Baseline (full EN axis → ZH top-5): %.3f" % base_acc)
