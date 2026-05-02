import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.linalg import svd as scipy_svd

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

def build_axis(pairs, zh=False):
    chords = []; usable = []
    for s, t in pairs:
        es, _ = get_emb(s, zh); et, _ = get_emb(t, zh)
        if es is None or et is None: continue
        chords.append(et - es); usable.append((s,t))
    if not chords: return None, usable
    return normed(np.mean(chords, axis=0)), usable

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

# Semantic axes
EN_GENDER = [
    ('king','queen'), ('man','woman'), ('boy','girl'),
    ('father','mother'), ('son','daughter'), ('husband','wife'),
    ('uncle','aunt'), ('prince','princess'), ('actor','actress'),
    ('waiter','waitress'),
]
EN_SIZE = [
    ('small','big'), ('little','large'), ('tiny','huge'),
    ('narrow','wide'), ('short','tall'), ('shallow','deep'),
    ('thin','thick'), ('weak','strong'), ('slow','fast'), ('cold','hot'),
]
EN_SENTIMENT = [
    ('bad','good'), ('ugly','beautiful'), ('hate','love'),
    ('sad','happy'), ('dark','bright'), ('wrong','right'),
    ('evil','good'), ('poor','rich'), ('sick','healthy'), ('dirty','clean'),
]
EN_AGE = [
    ('young','old'), ('new','ancient'), ('fresh','stale'),
    ('baby','adult'), ('child','elder'),
]
EN_PLURAL = [
    ('cat','cats'), ('dog','dogs'), ('house','houses'), ('car','cars'),
    ('tree','trees'), ('book','books'), ('bird','birds'), ('door','doors'),
]
ZH_GENDER = [
    ('男人','女人'), ('国王','女王'), ('父亲','母亲'), ('儿子','女儿'),
    ('丈夫','妻子'), ('叔叔','阿姨'), ('王子','公主'), ('男孩','女孩'),
    ('兄弟','姐妹'),
]
ZH_SIZE = [
    ('小','大'), ('窄','宽'), ('短','长'), ('浅','深'),
    ('薄','厚'), ('弱','强'), ('慢','快'), ('冷','热'),
]
ZH_SENTIMENT = [
    ('坏','好'), ('丑','美'), ('恨','爱'), ('悲','喜'),
    ('暗','亮'), ('错','对'), ('穷','富'),
]

# Build all semantic axis directions
semantic_axes = {}
for name, pairs, zh in [
    ('EN_gender',    EN_GENDER,    False),
    ('EN_size',      EN_SIZE,      False),
    ('EN_sentiment', EN_SENTIMENT, False),
    ('EN_age',       EN_AGE,       False),
    ('EN_plural',    EN_PLURAL,    False),
    ('ZH_gender',    ZH_GENDER,    True),
    ('ZH_size',      ZH_SIZE,      True),
    ('ZH_sentiment', ZH_SENTIMENT, True),
]:
    ax, _ = build_axis(pairs, zh=zh)
    if ax is not None: semantic_axes[name] = ax

print("\nDAY 357: PCA Alignment — Do Semantic Axes Match Principal Components?")
print("="*70)
print("Hypothesis: semantic axes ARE principal components of the embedding matrix")
print()

# ====================================================================
# PHASE 1: SVD of the FULL embedding matrix
# ====================================================================
print("Phase 1: SVD of embed_tokens weight matrix [151643 × 1536]")
print("  (Randomized SVD for speed — top 200 components)")
print()

from sklearn.utils.extmath import randomized_svd
# Centre the embedding matrix first
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)

print("  Running randomized SVD (n_components=200)...")
U, S, Vt = randomized_svd(W_cent, n_components=200, random_state=42)
# Vt: [200, 1536] — rows are principal directions in embedding space
print("  Done. SV range: [%.2f, %.2f]" % (S[0], S[-1]))
print("  Explained variance ratio (top 10): %s" % [round(float(s**2/np.sum(S**2)*100),2) for s in S[:10]])
print()

# ====================================================================
# PHASE 2: Alignment of semantic axes with PCs
# ====================================================================
print("Phase 2: Cosine alignment of semantic axes with top PCs")
print()

# For each semantic axis, find its best matching PC and the alignment
print("  %-16s  best_PC  |cos|   rank   PC_label" % "Axis")
print("  " + "-"*60)

pc_labels = {}  # Will be filled as we identify PCs

axis_alignments = {}
for ax_name, ax_dir in semantic_axes.items():
    ax_n = ax_dir.astype(np.float32)
    # Compute cosine with each PC direction (Vt rows)
    dots = np.abs(Vt.astype(np.float32) @ ax_n)  # [200]
    best_pc = int(np.argmax(dots))
    best_cos = float(dots[best_pc])
    axis_alignments[ax_name] = (best_pc, best_cos, dots)
    print("  %-16s  PC%-3d    %.4f  rank=%d" % (ax_name, best_pc+1, best_cos, best_pc+1))

# ====================================================================
# PHASE 3: What do the top PCs represent?
# ====================================================================
print("\nPhase 3: Semantic interpretation of top PCs (vocabulary extremes)")
print()

for pc_idx in range(20):
    pc_dir = Vt[pc_idx].astype(np.float32)
    # Find vocabulary extremes (projected onto this PC)
    # Project ALL embeddings onto this PC
    proj = W_n @ pc_dir  # Use normalised embeddings to avoid frequency effects
    
    # Top positive tokens
    top_pos_idx = np.argsort(proj)[::-1][:30]
    top_neg_idx = np.argsort(proj)[:30]
    
    # Filter to readable EN tokens
    pos_en = [(tok.decode([int(i)]).strip(), float(proj[i]))
              for i in top_pos_idx if EN_MASK[i]][:6]
    neg_en = [(tok.decode([int(i)]).strip(), float(proj[i]))
              for i in top_neg_idx if EN_MASK[i]][:6]
    pos_zh = [(tok.decode([int(i)]).strip(), float(proj[i]))
              for i in top_pos_idx if ZH_MASK[i]][:4]
    neg_zh = [(tok.decode([int(i)]).strip(), float(proj[i]))
              for i in top_neg_idx if ZH_MASK[i]][:4]
    
    print("  PC%2d (σ=%.1f):" % (pc_idx+1, S[pc_idx]))
    print("    + EN: %s" % [(w,round(c,3)) for w,c in pos_en])
    print("    - EN: %s" % [(w,round(c,3)) for w,c in neg_en])
    print("    + ZH: %s" % [(w,round(c,3)) for w,c in pos_zh])
    print("    - ZH: %s" % [(w,round(c,3)) for w,c in neg_zh])
    
    # Check alignment with all semantic axes
    aligns = [(name, float(np.abs(np.dot(Vt[pc_idx].astype(np.float32), d.astype(np.float32)))))
              for name, d in semantic_axes.items()]
    aligns.sort(key=lambda x: -x[1])
    top_align = ["%s=%.3f" % (n.replace('EN_','').replace('ZH_','ZH-'), c)
                 for n,c in aligns[:3] if c > 0.1]
    if top_align: print("    AXES: %s" % ', '.join(top_align))
    print()

# ====================================================================
# PHASE 4: Direct cosine of semantic axes with PCs (matrix)
# ====================================================================
print("Phase 4: Semantic axis alignment matrix — top 30 PCs")
print()

ax_names_short = [k.replace('EN_','') for k in list(semantic_axes.keys())[:5]]
print("  %s" % "   ".join("PC%-3d" % (i+1) for i in range(30)))
for ax_name in list(semantic_axes.keys())[:5]:
    ax_n = semantic_axes[ax_name].astype(np.float32)
    dots = np.abs(Vt[:30].astype(np.float32) @ ax_n)
    name_s = ax_name.replace('EN_','')
    # Find if any PC has high alignment
    top_pcs = np.argsort(dots)[::-1][:5]
    top_str = " | top: " + " ".join("PC%d=%.3f" % (i+1, dots[i]) for i in top_pcs)
    bar = "".join("█" if d > 0.4 else "▒" if d > 0.2 else "░" for d in dots)
    print("  %-12s  %s%s" % (name_s, bar, top_str))

print()
# Also show top 200 for gender
print("  EN_gender alignment with top 200 PCs:")
ax_n = semantic_axes['EN_gender'].astype(np.float32)
dots_gender = np.abs(Vt.astype(np.float32) @ ax_n)
top200 = np.argsort(dots_gender)[::-1][:10]
print("  " + " ".join("PC%d=%.4f" % (i+1, dots_gender[i]) for i in top200))

# ====================================================================
# PHASE 5: How much of each semantic axis is captured by top-K PCs?
# ====================================================================
print("\nPhase 5: Cumulative variance captured by top-K PCs for each axis")
print()

print("  %-16s  " % "Axis", end="")
for k in [1, 5, 10, 20, 50, 100, 200]:
    print("top%-3d  " % k, end="")
print()
print("  " + "-"*75)

for ax_name, ax_dir in semantic_axes.items():
    ax_n = ax_dir.astype(np.float32)
    dots = (Vt.astype(np.float32) @ ax_n)  # signed
    print("  %-16s  " % ax_name, end="")
    for k in [1, 5, 10, 20, 50, 100, 200]:
        # How much of the unit axis vector is captured by projection onto top-K PCs?
        top_k_idx = np.argsort(np.abs(dots))[::-1][:k]
        recon = sum(float(dots[i])**2 for i in top_k_idx)
        print("%.3f   " % recon, end="")
    print()

# ====================================================================
# PHASE 6: Semantic subspace — does a low-dim semantic subspace exist?
# ====================================================================
print("\nPhase 6: Semantic subspace — joint PCA of the semantic axes themselves")
print()

# Stack all semantic axis directions into a matrix and SVD it
ax_matrix = np.stack([v.astype(np.float32) for v in semantic_axes.values()], axis=0)
# ax_matrix: [8, 1536]

U_ax, S_ax, Vt_ax = np.linalg.svd(ax_matrix, full_matrices=False)
print("  SVD of 8-axis matrix: singular values =", [round(float(s),4) for s in S_ax])
cum_var = np.cumsum(S_ax**2) / np.sum(S_ax**2)
print("  Cumulative variance: %s" % [round(float(v),4) for v in cum_var])
print()

# The first few Vt_ax rows ARE the semantic subspace principal directions
# How well does each semantic axis project onto top-1, top-2, top-3 semantic PCs?
print("  Projection of each axis onto semantic subspace PCs:")
for i, (ax_name, ax_dir) in enumerate(semantic_axes.items()):
    ax_n = ax_dir.astype(np.float32)
    projs = [float(np.dot(ax_n, Vt_ax[j].astype(np.float32)))**2 for j in range(len(S_ax))]
    cum = np.cumsum(projs)
    print("  %-16s: PC1=%.3f  top2=%.3f  top3=%.3f  top4=%.3f" % (
        ax_name, projs[0], cum[1], cum[2], cum[3]))

print()
# How aligned is the semantic subspace (Vt_ax top rows) with W_E PCA (Vt top rows)?
print("  Alignment of semantic subspace with W_E principal components:")
print("  (cosine of semantic PC with each W_E PC, top 3 matches)")
for sem_pc_idx in range(4):
    sem_pc = Vt_ax[sem_pc_idx].astype(np.float32)
    # Alignment with W_E PCs
    aligns = np.abs(Vt.astype(np.float32) @ sem_pc)
    top3 = np.argsort(aligns)[::-1][:5]
    print("  SemPC%d (σ=%.3f): " % (sem_pc_idx+1, S_ax[sem_pc_idx]), end="")
    print(" ".join("W_E_PC%d=%.4f" % (j+1, aligns[j]) for j in top3))

# ====================================================================
# PHASE 7: Surprise check — what do the TOP W_E PCs actually encode?
# ====================================================================
print("\nPhase 7: Top W_E PCs — language vs semantic vs frequency signal?")
print()

# For the top 5 PCs, check how much they correlate with:
# (a) token frequency (approximate: use |W_E| as proxy since high-freq tokens are more trained)
# (b) language (EN vs ZH projection)
# (c) semantic axes

# Language signal: EN tokens project onto +/- some PC?
print("  PC  σ        EN_mean    ZH_mean    EN-ZH    top_axes")
print("  " + "-"*65)
for pc_idx in range(15):
    pc_dir = Vt[pc_idx].astype(np.float32)
    proj = W_n @ pc_dir
    en_mean = float(np.mean(proj[EN_MASK]))
    zh_mean = float(np.mean(proj[ZH_MASK]))
    # Top axes
    aligns = [(name, float(np.abs(np.dot(Vt[pc_idx].astype(np.float32), d.astype(np.float32)))))
              for name, d in semantic_axes.items()]
    aligns.sort(key=lambda x: -x[1])
    top_a = ["%s=%.3f" % (n.replace('EN_','').replace('ZH_','zh_'), c)
              for n,c in aligns[:2] if c > 0.1]
    print("  PC%-2d σ=%-6.1f  EN=% .4f  ZH=% .4f  Δ=% .4f  %s" % (
        pc_idx+1, S[pc_idx], en_mean, zh_mean, en_mean-zh_mean,
        ', '.join(top_a) if top_a else '(none)'))

# ====================================================================
# PHASE 8: The key question — is gender axis a PC?
# ====================================================================
print("\n" + "="*70)
print("Phase 8: Is the gender axis a principal component?")
print("="*70)
print()

ax_n = semantic_axes['EN_gender'].astype(np.float32)
dots_gender_signed = Vt.astype(np.float32) @ ax_n
top5_gender = np.argsort(np.abs(dots_gender_signed))[::-1][:10]
print("  EN gender axis alignment with top W_E PCs:")
for i in top5_gender[:10]:
    print("    PC%-3d (σ=%.1f): cos=% .4f" % (i+1, S[i], dots_gender_signed[i]))

print()
# Reconstruct gender axis from top matching PCs and see how much accuracy is preserved
print("  Gender axis reconstruction from top-N matching PCs:")
for n_pcs in [1, 2, 3, 5, 10, 20]:
    top_n = np.argsort(np.abs(dots_gender_signed))[::-1][:n_pcs]
    recon = sum(dots_gender_signed[i] * Vt[i] for i in top_n).astype(np.float32)
    # Normalize reconstruction
    recon_n = recon / (np.linalg.norm(recon) + 1e-8)
    cos_with_orig = float(np.dot(recon_n, ax_n))
    print("    top-%2d PCs: cos_with_original=%.4f  |recon|=%.4f" % (
        n_pcs, cos_with_orig, float(np.dot(dots_gender_signed[top_n], dots_gender_signed[top_n])**0.5)))

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "="*70)
print("SUMMARY: Day 357 — PCA Alignment of Semantic Axes")
print("="*70)
print()
print("  Key question: are semantic axes principal components of W_E?")
print()
print("  %-16s  best_PC  cos(axis,PC)  interpretation" % "Axis")
print("  " + "-"*60)
for ax_name in list(semantic_axes.keys()):
    if ax_name not in axis_alignments: continue
    best_pc, best_cos, _ = axis_alignments[ax_name]
    interp = ("STRONG alignment" if best_cos > 0.5 else
              "MODERATE alignment" if best_cos > 0.3 else
              "WEAK alignment" if best_cos > 0.15 else
              "no alignment — NOT a PC")
    print("  %-16s  PC%-3d    %.4f        %s" % (
        ax_name, best_pc+1, best_cos, interp))
print()
print("  If best_cos << 1.0 for all axes: semantic axes are NOT principal components.")
print("  They are oblique directions cutting ACROSS multiple PCs.")
print("  The embedding space is NOT primarily organised around semantic dimensions.")
print()
print("  The semantic axes live in a low-dimensional SEMANTIC SUBSPACE")
print("  that is SHARED across languages but is NOT the dominant variance direction.")
