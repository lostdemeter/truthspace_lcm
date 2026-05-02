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

def make_W_n(W): return np.array([normed(v) for v in W], dtype=np.float32)

W_n = make_W_n(W_E)

def get_emb_from(W, word, zh=False):
    if zh:
        ids = tok(word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W[ids[0]].copy(), ids[0]
        return None, None
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W[ids[0]].copy(), ids[0]
    return None, None

def nn_ret_from(W_n_local, pred_emb, excl_ids, mask, top_k=1):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n_local @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def source_ids_tok(word):
    ids = set()
    for p in [' ', '']:
        r = tok(p+word, add_special_tokens=False)['input_ids']
        ids.update(r)
    return ids

def build_axis_from(W, pairs, zh=False):
    chords = []
    for s, t in pairs:
        es, _ = get_emb_from(W, s, zh)
        et, _ = get_emb_from(W, t, zh)
        if es is None or et is None: continue
        chords.append(et - es)
    if not chords: return None
    return normed(np.mean(chords, axis=0))

def eval_axis(W, W_n_local, ax_dir, pairs, mask, zh=False):
    hits = 0; n = 0; scale_scores = {}
    for scale in np.linspace(0.05, 6.0, 30):
        c = 0
        for s, t in pairs:
            es, _ = get_emb_from(W, s, zh)
            if es is None: continue
            w, _, _ = nn_ret_from(W_n_local, es + scale*ax_dir, source_ids_tok(s if not zh else ''), mask)
            if w == t: c += 1
        scale_scores[scale] = c
    best_scale = max(scale_scores, key=scale_scores.get)
    best_acc = scale_scores[best_scale]
    # Count valid pairs
    n = sum(1 for s, t in pairs if get_emb_from(W, s, zh)[0] is not None)
    return best_acc, n, best_scale

# Semantic axis pairs
EN_GENDER = [
    ('king','queen'),('man','woman'),('boy','girl'),
    ('father','mother'),('son','daughter'),('husband','wife'),
    ('uncle','aunt'),('prince','princess'),('actor','actress'),('waiter','waitress'),
]
EN_SIZE = [
    ('small','big'),('little','large'),('tiny','huge'),('narrow','wide'),
    ('short','tall'),('shallow','deep'),('thin','thick'),('weak','strong'),
    ('slow','fast'),('cold','hot'),
]
EN_SENTIMENT = [
    ('bad','good'),('ugly','beautiful'),('hate','love'),('sad','happy'),
    ('dark','bright'),('wrong','right'),('evil','good'),('poor','rich'),
    ('sick','healthy'),('dirty','clean'),
]
EN_PLURAL = [
    ('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
    ('tree','trees'),('book','books'),('bird','birds'),('door','doors'),
]
ZH_GENDER = [
    ('男人','女人'),('国王','女王'),('父亲','母亲'),('儿子','女儿'),
    ('丈夫','妻子'),('叔叔','阿姨'),('王子','公主'),('男孩','女孩'),
]

print("\nDAY 359: PC1 Ablation and Semantic Subspace Reconstruction")
print("="*70)
print()

# ====================================================================
# SETUP: Compute PC1 and ablated embedding matrices
# ====================================================================
print("Setup: Computing PC1 direction and building ablated matrices...")
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)
U, S, Vt = randomized_svd(W_cent, n_components=50, random_state=42)
pc1_dir = Vt[0].astype(np.float64)
print("  SVD done. σ₁=%.2f σ₂=%.2f" % (S[0], S[1]))

# Ablated matrices
# A) Remove PC1 component from ORIGINAL (not centred) embeddings
W_abl_pc1 = W_E - np.outer(W_E @ pc1_dir, pc1_dir)
# B) Remove top-5 PCs
W_abl_pc5 = W_E.copy()
for k in range(5):
    pc = Vt[k].astype(np.float64)
    W_abl_pc5 = W_abl_pc5 - np.outer(W_abl_pc5 @ pc, pc)
# C) Remove top-20 PCs
W_abl_pc20 = W_E.copy()
for k in range(20):
    pc = Vt[k].astype(np.float64)
    W_abl_pc20 = W_abl_pc20 - np.outer(W_abl_pc20 @ pc, pc)
# D) Remove PC1 from centred matrix only
W_abl_cent_pc1 = W_cent - np.outer(W_cent @ pc1_dir, pc1_dir)

# Build normalised versions
print("  Building W_n for original, ablated-PC1, ablated-PC5, ablated-PC20...")
W_n_orig   = make_W_n(W_E)
W_n_abl1   = make_W_n(W_abl_pc1)
W_n_abl5   = make_W_n(W_abl_pc5)
W_n_abl20  = make_W_n(W_abl_pc20)
print("  Done.\n")

# ====================================================================
# PHASE 1: Quick accuracy test — 4 conditions × 4 axes
# ====================================================================
print("Phase 1: Axis accuracy under PC ablation")
print()

conditions = [
    ('Original',   W_E,        W_n_orig),
    ('Abl-PC1',    W_abl_pc1,  W_n_abl1),
    ('Abl-PC5',    W_abl_pc5,  W_n_abl5),
    ('Abl-PC20',   W_abl_pc20, W_n_abl20),
]

axis_defs = [
    ('EN_gender',    EN_GENDER,    False, EN_MASK),
    ('EN_size',      EN_SIZE,      False, EN_MASK),
    ('EN_sentiment', EN_SENTIMENT, False, EN_MASK),
    ('EN_plural',    EN_PLURAL,    False, EN_MASK),
    ('ZH_gender',    ZH_GENDER,    True,  ZH_MASK),
]

results = {}
for cond_name, W_local, W_n_local in conditions:
    results[cond_name] = {}
    for ax_name, pairs, zh, mask in axis_defs:
        ax_dir = build_axis_from(W_local, pairs, zh=zh)
        if ax_dir is None:
            results[cond_name][ax_name] = (0, 0, 0.0)
            continue
        acc, n, scale = eval_axis(W_local, W_n_local, ax_dir, pairs, mask, zh=zh)
        results[cond_name][ax_name] = (acc, n, scale)

print("  Accuracy (hits/total) per condition:")
print("  %-16s  %s" % ("Axis", "  ".join("%-12s" % c for c, _, _ in conditions)))
print("  " + "-"*75)
for ax_name, pairs, zh, mask in axis_defs:
    vals = []
    for cond_name, _, _ in conditions:
        acc, n, scale = results[cond_name][ax_name]
        vals.append("%d/%-2d (%.0f%%)" % (acc, n, 100*acc/n if n else 0))
    print("  %-16s  %s" % (ax_name, "  ".join("%-12s" % v for v in vals)))

print()

# Summarize changes
print("  Change in accuracy (Original → Abl-PC1):")
for ax_name, pairs, zh, mask in axis_defs:
    orig_acc, orig_n, _ = results['Original'][ax_name]
    abl_acc,  abl_n,  _ = results['Abl-PC1'][ax_name]
    delta = abl_acc - orig_acc
    pct_change = 100*(abl_acc - orig_acc)/max(orig_acc, 1)
    sign = "+" if delta >= 0 else ""
    print("  %-16s  %s%d (%.0f%% → %.0f%%)" % (
        ax_name, sign, delta,
        100*orig_acc/orig_n if orig_n else 0,
        100*abl_acc/abl_n if abl_n else 0))

print()

# ====================================================================
# PHASE 2: Axis coherence before/after ablation
# ====================================================================
print("Phase 2: Axis coherence (intra-pair cosine similarity) under ablation")
print()

def axis_coherence(W_local, pairs, zh=False):
    chords = []
    for s, t in pairs:
        es, _ = get_emb_from(W_local, s, zh)
        et, _ = get_emb_from(W_local, t, zh)
        if es is None or et is None: continue
        chords.append(normed(et - es))
    if len(chords) < 2: return 0.0, 0
    sims = []
    for i in range(len(chords)):
        for j in range(i+1, len(chords)):
            sims.append(float(np.dot(chords[i].astype(np.float32), chords[j].astype(np.float32))))
    return float(np.mean(sims)), len(chords)

print("  %-16s  %s" % ("Axis", "  ".join("%-12s" % c for c, _, _ in conditions)))
print("  " + "-"*70)
for ax_name, pairs, zh, mask in axis_defs:
    vals = []
    for cond_name, W_local, _ in conditions:
        coh, n = axis_coherence(W_local, pairs, zh=zh)
        vals.append("%.4f (n=%d)" % (coh, n))
    print("  %-16s  %s" % (ax_name, "  ".join("%-14s" % v for v in vals)))

print()

# ====================================================================
# PHASE 3: Nearest neighbor quality — does PC1 ablation reduce
# false positives from function words appearing as top matches?
# ====================================================================
print("Phase 3: NN quality — function word contamination in top-10 results")
print()

FUNCTION_SET = {
    'the','a','an','of','in','to','is','was','are','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'may','might','shall','can','must','and','but','or','for','that','this',
    'it','he','she','we','they','you','not','no','all','any','some','so',
    'if','by','on','at','as','with','from','more','most','less','very',
    'just','also','then','when','where','while','about','after','before',
    'than','how','what','which','who','there','here','my','your','its',
}

def count_fn_in_top10(W_n_local, query_emb, excl_ids, mask):
    pred_n = normed(query_emb).astype(np.float32)
    sims = W_n_local @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    top10 = np.argsort(sims)[::-1][:10]
    fn_count = sum(1 for i in top10 if tok.decode([int(i)]).strip().lower() in FUNCTION_SET)
    return fn_count

print("  Function word contamination in top-10 gender axis predictions:")
print("  (lower is better — function words don't belong in semantic queries)")
print()
print("  %-12s→%-12s  %s" % ("source", "target",
    "  ".join("%-12s" % c for c, _, _ in conditions)))
print("  " + "-"*70)

for src, tgt in EN_GENDER[:8]:
    row = []
    for cond_name, W_local, W_n_local in conditions:
        ax_dir = build_axis_from(W_local, EN_GENDER)
        if ax_dir is None: row.append("N/A"); continue
        # Best scale for this condition
        _, _, best_scale = results[cond_name]['EN_gender']
        es, _ = get_emb_from(W_local, src)
        if es is None: row.append("N/A"); continue
        pred = es + best_scale * ax_dir
        fn_count = count_fn_in_top10(W_n_local, pred, source_ids_tok(src), EN_MASK)
        # Also get the actual top-1
        w, sim, _ = nn_ret_from(W_n_local, pred, source_ids_tok(src), EN_MASK)
        row.append("%s(%d)" % (w, fn_count))
    print("  %-12s→%-12s  %s" % (src, tgt, "  ".join("%-12s" % r for r in row)))

print()

# ====================================================================
# PHASE 4: Semantic subspace reconstruction
# ====================================================================
print("Phase 4: Semantic subspace reconstruction")
print("  Project all embeddings into the 8-axis semantic subspace")
print("  and test axis retrieval accuracy within that subspace")
print()

# Build the 8-axis semantic directions in ORIGINAL space
raw_axes = {}
ax_defs_full = [
    ('EN_gender',    EN_GENDER,    False),
    ('EN_size',      EN_SIZE,      False),
    ('EN_sentiment', EN_SENTIMENT, False),
    ('EN_plural',    EN_PLURAL,    False),
    ('ZH_gender',    ZH_GENDER,    True),
]
for ax_name, pairs, zh in ax_defs_full:
    ax = build_axis_from(W_E, pairs, zh=zh)
    if ax is not None: raw_axes[ax_name] = ax.astype(np.float32)

# Stack axes into semantic basis matrix [n_axes × 1536]
ax_matrix = np.stack(list(raw_axes.values()), axis=0).astype(np.float32)
print("  Semantic axis matrix: %s" % str(ax_matrix.shape))

# Orthonormalize via QR to get clean semantic subspace basis
Q, R = np.linalg.qr(ax_matrix.T)  # Q: [1536 × n_axes], columns are ON basis
Q = Q.astype(np.float32)
print("  Orthonormal semantic basis: %s (columns)" % str(Q.shape))
print()

# Project embeddings into semantic subspace
# W_sem: [151643 × n_axes] — coordinates in semantic space
W_sem = (W_E.astype(np.float32)) @ Q  # [N × n_axes]
print("  Projecting all tokens into semantic subspace...")
print("  W_sem shape: %s" % str(W_sem.shape))
print()

# Normalise for cosine retrieval in semantic subspace
W_sem_n = np.array([normed(v) for v in W_sem], dtype=np.float32)

def eval_axis_in_subspace(ax_name, pairs, zh=False):
    # The axis direction in semantic subspace coordinates
    ax_full = raw_axes[ax_name]
    ax_sem = Q.T @ ax_full  # [n_axes] — axis in semantic coords
    ax_sem_n = normed(ax_sem).astype(np.float32)
    n_axes_sem = Q.shape[1]
    # Scale search in semantic space
    best_scale = 0; best_acc = 0
    for scale in np.linspace(0.01, 2.0, 30):
        c = 0
        for s, t in pairs:
            for p in [' ', '']:
                ids = tok(p+s, add_special_tokens=False)['input_ids']
                if len(ids)==1:
                    es_sem = W_sem[ids[0]]
                    break
            else:
                if zh:
                    ids2 = tok(s, add_special_tokens=False)['input_ids']
                    if len(ids2)==1: es_sem = W_sem[ids2[0]]
                    else: continue
                else: continue
            pred_sem = es_sem + scale * ax_sem
            pred_sem_n = normed(pred_sem).astype(np.float32)
            sims = W_sem_n @ pred_sem_n
            mask_to_use = ZH_MASK if zh else EN_MASK
            sims[~mask_to_use] = -1.0
            # Exclude source
            for p in [' ', '']:
                ids3 = tok(p+s, add_special_tokens=False)['input_ids']
                for i in ids3: sims[i] = -1.0
            best_idx = int(np.argmax(sims))
            if tok.decode([best_idx]).strip() == t: c += 1
        if c > best_acc: best_acc = c; best_scale = scale
    n_pairs = sum(1 for s, t in pairs
                  if any(len(tok(p+s,add_special_tokens=False)['input_ids'])==1
                         for p in [' ','']) or
                     (zh and len(tok(s,add_special_tokens=False)['input_ids'])==1))
    return best_acc, n_pairs, best_scale

print("  Axis accuracy in semantic subspace vs full space:")
print()
print("  %-16s  %-18s  %-18s  sem/full" % ("Axis", "Full space", "Sem subspace"))
print("  " + "-"*65)
for ax_name, pairs, zh in ax_defs_full:
    if ax_name not in raw_axes: continue
    full_acc, full_n, full_scale = results['Original'][ax_name]
    sem_acc, sem_n, sem_scale = eval_axis_in_subspace(ax_name, pairs, zh=zh)
    ratio = sem_acc / max(full_acc, 1)
    print("  %-16s  %d/%d (%.0f%%)       %d/%d (%.0f%%)       %.2f" % (
        ax_name, full_acc, full_n, 100*full_acc/full_n if full_n else 0,
        sem_acc, sem_n, 100*sem_acc/sem_n if sem_n else 0, ratio))

print()

# ====================================================================
# PHASE 5: Cross-axis semantic retrieval in subspace
# ====================================================================
print("Phase 5: Semantic subspace cross-axis retrieval")
print("  Can we find a token using ONLY its semantic subspace coordinates?")
print()

test_words = ['king','queen','man','woman','cat','dog','big','small','good','bad',
              'happy','sad','father','mother','red','blue','run','sit','three','seven']

print("  %-12s  full-space-nn  sem-subspace-nn  same?" % "word")
print("  " + "-"*55)
for word in test_words:
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            idx = ids[0]
            # Full space NN (excluding self)
            excl = {idx}
            w_full, sim_full, _ = nn_ret_from(W_n_orig, W_E[idx], excl, EN_MASK)
            # Semantic subspace NN
            esem = W_sem_n[idx]
            sims_sem = W_sem_n @ esem; sims_sem[~EN_MASK] = -1.0; sims_sem[idx] = -1.0
            idx_sem = int(np.argmax(sims_sem))
            w_sem = tok.decode([idx_sem]).strip()
            same = "✓" if w_full == w_sem else "✗"
            print("  %-12s  %-14s %-16s  %s" % (word, w_full, w_sem, same))
            break

print()

# ====================================================================
# PHASE 6: Does PC1 ablation change the semantic axis directions?
# ====================================================================
print("Phase 6: Do semantic axis DIRECTIONS change under PC1 ablation?")
print()

print("  Cosine similarity of axis direction: Original vs Abl-PC1")
print()
for ax_name, pairs, zh, mask in axis_defs:
    ax_orig = build_axis_from(W_E, pairs, zh=zh)
    ax_abl  = build_axis_from(W_abl_pc1, pairs, zh=zh)
    if ax_orig is None or ax_abl is None: continue
    cos = float(np.dot(ax_orig.astype(np.float32), ax_abl.astype(np.float32)))
    print("  %-16s  cos(orig,abl-PC1) = %.6f  (Δaxis = %.6f)" % (
        ax_name, cos, 1.0-cos))

print()
print("  Note: if cos ≈ 1.0 the axis is unchanged by removing PC1.")
print("  If cos << 1.0, PC1 was distorting the axis direction.")

print()

# ====================================================================
# PHASE 7: What fills the gap in NN space after removing PC1?
# ====================================================================
print("Phase 7: How does NN space change after PC1 ablation?")
print()
print("  Top-5 nearest neighbors (EN): Original vs Abl-PC1")
print()
test_probe = ['king', 'good', 'run', 'the', 'cat', 'happy']
for word in test_probe:
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            idx = ids[0]
            excl = {idx}
            # Original
            sims_o = W_n_orig @ (normed(W_E[idx]).astype(np.float32))
            sims_o[~EN_MASK] = -1.0; sims_o[idx] = -1.0
            top5_o = [tok.decode([int(i)]).strip() for i in np.argsort(sims_o)[::-1][:5]]
            # Ablated
            sims_a = W_n_abl1 @ (normed(W_abl_pc1[idx]).astype(np.float32))
            sims_a[~EN_MASK] = -1.0; sims_a[idx] = -1.0
            top5_a = [tok.decode([int(i)]).strip() for i in np.argsort(sims_a)[::-1][:5]]
            print("  '%s'" % word)
            print("    Original:  %s" % top5_o)
            print("    Abl-PC1:   %s" % top5_a)
            break

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 359 — PC1 Ablation and Semantic Subspace")
print("="*70)
print()
print("  PC1 ablation impact on accuracy:")
for ax_name, pairs, zh, mask in axis_defs:
    o_acc, o_n, _ = results['Original'][ax_name]
    a_acc, a_n, _ = results['Abl-PC1'][ax_name]
    delta = a_acc - o_acc
    sign = "+" if delta >= 0 else ""
    impact = ("IMPROVED" if delta > 0 else "DEGRADED" if delta < 0 else "unchanged")
    print("  %-16s  %d→%d (%s%d)  %s" % (
        ax_name, o_acc, a_acc, sign, delta, impact))
print()
print("  Semantic subspace reconstruction:")
print("  (details above in Phase 4)")
print()
print("  If PC1 ablation does NOT change accuracy: PC1 is orthogonal to")
print("  semantic axes (confirmed by Day 358: gender axis 98.7%% outside PC1).")
print("  If removing PC1 IMPROVES accuracy: it was adding interference noise.")
print("  If removing PC1 DEGRADES accuracy: semantic axes partially need PC1.")
