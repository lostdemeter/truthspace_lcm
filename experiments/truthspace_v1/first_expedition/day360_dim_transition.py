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

def get_emb(word, zh=False):
    if zh:
        ids = tok(word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
        return None, None
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def source_ids(word):
    ids = set()
    for p in [' ', '']:
        r = tok(p+word, add_special_tokens=False)['input_ids']
        ids.update(r)
    return ids

def build_axis(pairs, W_local=None, zh=False):
    W = W_local if W_local is not None else W_E
    chords = []
    for s, t in pairs:
        es = et = None
        for p in [' ', '']:
            ids = tok(p+s, add_special_tokens=False)['input_ids']
            if len(ids) == 1: es = W[ids[0]].copy(); break
        if zh and es is None:
            ids = tok(s, add_special_tokens=False)['input_ids']
            if len(ids) == 1: es = W[ids[0]].copy()
        for p in [' ', '']:
            ids = tok(p+t, add_special_tokens=False)['input_ids']
            if len(ids) == 1: et = W[ids[0]].copy(); break
        if zh and et is None:
            ids = tok(t, add_special_tokens=False)['input_ids']
            if len(ids) == 1: et = W[ids[0]].copy()
        if es is not None and et is not None:
            chords.append(et - es)
    if not chords: return None
    return normed(np.mean(chords, axis=0))

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

# Probe pairs for NN-only retrieval (no axis, just: does it retrieve itself?)
NN_PROBE = ['king','queen','man','woman','cat','dog','tree','house','car','book',
            'good','bad','happy','sad','big','small','fast','slow','red','blue',
            'father','mother','son','daughter','run','jump','eat','drink','three','seven']

print("\nDAY 360: Dimensionality Transition — When Does Retrieval Become Exact?")
print("="*70)
print()

# ====================================================================
# COMPUTE SVD for different k values
# ====================================================================
print("Computing SVD (k=1536 components)...")
W_mean = W_E.mean(axis=0)
W_cent = (W_E - W_mean).astype(np.float32)
# Use full SVD up to 1536 components
U_full, S_full, Vt_full = randomized_svd(W_cent, n_components=1024, random_state=42)
print("  Done. σ range: %.2f → %.4f (top 1024 PCs)" % (S_full[0], S_full[-1]))
print()

# ====================================================================
# PHASE 1: Self-retrieval accuracy vs dimensionality
# ====================================================================
# For each dimensionality k, project all embeddings into the top-k PC subspace
# and test how many words retrieve themselves as NN

print("Phase 1: Self-retrieval accuracy at each dimensionality k")
print("  (what % of probe words retrieve themselves as nearest neighbour)")
print()

dims_to_test = [5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1024, 1536]
# For 1536, we just use original W_E

# Build projected embedding matrices for each k
def project_to_k_dims(k):
    """Project W_E onto top-k PCs and return the projected matrix."""
    if k >= 1024:
        # Use all 1024 computed PCs
        Vk = Vt_full[:k if k <= 1024 else 1024].astype(np.float64)
    else:
        Vk = Vt_full[:k].astype(np.float64)
    # Project: W_proj[i] = sum_j (W_E[i] · v_j) * v_j
    coords = W_cent.astype(np.float64) @ Vk.T  # [N × k]
    W_proj = coords @ Vk                         # [N × 1536] reconstruction
    W_proj += W_mean                              # add mean back
    return W_proj

def self_retrieval(W_proj, words, mask):
    W_proj_n = np.array([normed(v) for v in W_proj], dtype=np.float32)
    hits = 0; n = 0
    for word in words:
        for p in [' ', '']:
            ids = tok(p+word, add_special_tokens=False)['input_ids']
            if len(ids) == 1:
                idx = ids[0]
                # Find self — does this token retrieve itself?
                sims = W_proj_n @ (normed(W_proj[idx]).astype(np.float32))
                sims[~mask] = -1.0
                # DON'T exclude self — check if self is top-1
                best_idx = int(np.argmax(sims))
                if best_idx == idx: hits += 1
                n += 1
                break
    return hits, n

print("  %-8s  %-12s  %-12s" % ("dims", "self-retrieval", "accuracy"))
print("  " + "-"*35)

self_ret_results = {}
for k in dims_to_test:
    if k == 1536:
        W_proj = W_E.copy()
    else:
        W_proj = project_to_k_dims(k)
    hits, n = self_retrieval(W_proj, NN_PROBE, EN_MASK)
    acc = 100*hits/n if n else 0
    self_ret_results[k] = (hits, n, acc)
    print("  %-8d  %d/%-2d           %.1f%%" % (k, hits, n, acc))

print()

# ====================================================================
# PHASE 2: Semantic axis accuracy vs dimensionality
# ====================================================================
print("Phase 2: Semantic axis accuracy at each dimensionality k")
print()

def axis_acc_at_k(W_proj, pairs, mask, zh=False):
    ax_dir = build_axis(pairs, W_local=W_proj, zh=zh)
    if ax_dir is None: return 0, 0, 0.0
    W_proj_n = np.array([normed(v) for v in W_proj], dtype=np.float32)
    best_acc = 0; best_scale = 0
    for scale in np.linspace(0.05, 6.0, 25):
        c = 0
        for s, t in pairs:
            es = et = None
            for p in [' ', '']:
                ids = tok(p+s, add_special_tokens=False)['input_ids']
                if len(ids) == 1: es = W_proj[ids[0]].copy(); break
            if zh and es is None:
                ids = tok(s, add_special_tokens=False)['input_ids']
                if len(ids) == 1: es = W_proj[ids[0]].copy()
            if es is None: continue
            pred = es + scale * ax_dir
            pred_n = normed(pred).astype(np.float32)
            sims = W_proj_n @ pred_n
            sims[~mask] = -1.0
            for p in [' ', '']:
                ids2 = tok(p+s, add_special_tokens=False)['input_ids']
                for ii in ids2: sims[ii] = -1.0
            best_idx = int(np.argmax(sims))
            if tok.decode([best_idx]).strip() == t: c += 1
        if c > best_acc: best_acc = c; best_scale = scale
    n_pairs = sum(1 for s, t in pairs
                  if any(len(tok(p+s, add_special_tokens=False)['input_ids'])==1
                         for p in [' ','']))
    return best_acc, n_pairs, best_scale

axis_configs = [
    ('EN_gender',    EN_GENDER,    False, EN_MASK),
    ('EN_size',      EN_SIZE,      False, EN_MASK),
    ('EN_sentiment', EN_SENTIMENT, False, EN_MASK),
    ('EN_plural',    EN_PLURAL,    False, EN_MASK),
    ('ZH_gender',    ZH_GENDER,    True,  ZH_MASK),
]

axis_results = {ax: {} for ax, _, _, _ in axis_configs}
print("  %-8s  %s" % ("dims", "  ".join("%-16s" % ax for ax, _, _, _ in axis_configs)))
print("  " + "-"*85)

for k in dims_to_test:
    if k == 1536:
        W_proj = W_E.copy()
    else:
        W_proj = project_to_k_dims(k)
    row = []
    for ax_name, pairs, zh, mask in axis_configs:
        acc, n, scale = axis_acc_at_k(W_proj, pairs, mask, zh=zh)
        axis_results[ax_name][k] = (acc, n, scale)
        row.append("%d/%d (%.0f%%)" % (acc, n, 100*acc/n if n else 0))
    print("  %-8d  %s" % (k, "  ".join("%-16s" % r for r in row)))

print()

# ====================================================================
# PHASE 3: Token distinguish-ability — how many tokens are "same NN" at k dims?
# ====================================================================
print("Phase 3: Token distinguishability — duplicate NN clusters at k dims")
print("  (how many EN tokens share the same nearest neighbour?)")
print()
print("  At low k, many tokens collapse to the same NN (clusters form).")
print("  At high k, every token has a unique nearest neighbour.")
print()

# For a SAMPLE of EN tokens, measure how many unique NNs exist
EN_SAMPLE_SIZE = 2000
EN_SAMPLE_IDS = np.where(EN_MASK)[0][:EN_SAMPLE_SIZE]

print("  %-8s  %-20s  %-20s  %-20s" % (
    "dims", "unique NNs / sample", "mean cluster size", "max cluster size"))
print("  " + "-"*75)

for k in [5, 10, 20, 50, 100, 200, 500, 1024, 1536]:
    if k == 1536:
        W_proj = W_E.copy()
    else:
        W_proj = project_to_k_dims(k)
    W_proj_n = np.array([normed(v) for v in W_proj[EN_SAMPLE_IDS]], dtype=np.float32)
    W_proj_n_all = np.array([normed(v) for v in W_proj], dtype=np.float32)
    # Find NN for each sample token
    nn_ids = []
    for i, idx in enumerate(EN_SAMPLE_IDS):
        sims = W_proj_n_all @ W_proj_n[i]
        sims[~EN_MASK] = -1.0; sims[idx] = -1.0
        nn_ids.append(int(np.argmax(sims)))
    nn_array = np.array(nn_ids)
    unique_nns = len(set(nn_ids))
    from collections import Counter
    counts = Counter(nn_ids)
    cluster_sizes = list(counts.values())
    mean_clust = np.mean(cluster_sizes)
    max_clust = max(cluster_sizes)
    print("  %-8d  %d / %d (%.1f%%)       %.2f                  %d" % (
        k, unique_nns, EN_SAMPLE_SIZE, 100*unique_nns/EN_SAMPLE_SIZE,
        mean_clust, max_clust))

print()

# ====================================================================
# PHASE 4: Semantic similarity (top-1 semantic match) vs dimensionality
# ====================================================================
print("Phase 4: Semantic clustering quality vs dimensionality")
print("  (are semantically related words nearest neighbours at each k?)")
print()
print("  Test: does 'king' retrieve 'queen' as NN at each k?")
print()

SEMANTIC_PAIRS = [
    ('king','queen'), ('man','woman'), ('good','bad'), ('big','small'),
    ('happy','sad'), ('cat','dog'), ('run','walk'), ('fast','slow'),
    ('father','son'), ('red','blue'),
]

print("  %-8s  %s" % ("dims", "  ".join("%-14s" % ("%s→%s" % (s,t)) for s,t in SEMANTIC_PAIRS)))
print("  " + "-"*115)

for k in dims_to_test:
    if k == 1536:
        W_proj = W_E.copy()
    else:
        W_proj = project_to_k_dims(k)
    W_proj_n = np.array([normed(v) for v in W_proj], dtype=np.float32)
    results_row = []
    for s, t in SEMANTIC_PAIRS:
        es = None
        for p in [' ', '']:
            ids = tok(p+s, add_special_tokens=False)['input_ids']
            if len(ids) == 1: es = W_proj[ids[0]]; break
        if es is None: results_row.append("N/A"); continue
        sims = W_proj_n @ normed(es).astype(np.float32)
        sims[~EN_MASK] = -1.0
        for p in [' ', '']:
            ids2 = tok(p+s, add_special_tokens=False)['input_ids']
            for ii in ids2: sims[ii] = -1.0
        best = tok.decode([int(np.argmax(sims))]).strip()
        mark = "✓" if best == t else " "
        results_row.append("%s%s" % (mark, best[:10]))
    print("  %-8d  %s" % (k, "  ".join("%-14s" % r for r in results_row)))

print()

# ====================================================================
# PHASE 5: Transition ANALYSIS — find the knee point
# ====================================================================
print("Phase 5: Transition analysis — identifying the semantic resolution knee")
print()

# Compute composite score: avg of self-retrieval and axis accuracies
print("  %-8s  %-14s  %-14s  %-12s  composite" % (
    "dims", "self-ret%", "gender%", "plural%"))
print("  " + "-"*60)
for k in dims_to_test:
    sr = self_ret_results[k][2]
    gen = 100 * axis_results['EN_gender'][k][0] / max(axis_results['EN_gender'][k][1], 1)
    plu = 100 * axis_results['EN_plural'][k][0] / max(axis_results['EN_plural'][k][1], 1)
    composite = (sr + gen + plu) / 3
    print("  %-8d  %-14.1f  %-14.1f  %-12.1f  %.1f" % (k, sr, gen, plu, composite))

print()

# ====================================================================
# PHASE 6: Random vs structured tokens — which need more dims?
# ====================================================================
print("Phase 6: Structured (common) vs random (rare) tokens — retrieval dimensionality")
print()

COMMON_WORDS  = ['the','a','of','to','in','is','it','be','as','at',
                  'king','man','cat','good','run','big','happy','red','three','love']
RARE_WORDS    = ['minstrel','oblong','somnolent','scullery','lachrymose',
                 'prestidigitator','filigree','obsequious','pugnacious','vituperate']

print("  At what k does each word first retrieve ITSELF as NN?")
print()
print("  %-20s  first-exact-k" % "word")
print("  " + "-"*35)

for word in COMMON_WORDS + RARE_WORDS:
    ids_list = []
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: ids_list = ids; break
    if not ids_list: print("  %-20s  (no single token)" % word); continue
    idx = ids_list[0]
    first_k = None
    for k in dims_to_test:
        if k == 1536:
            W_proj = W_E.copy()
        else:
            W_proj = project_to_k_dims(k)
        W_proj_n = np.array([normed(v) for v in W_proj], dtype=np.float32)
        sims = W_proj_n @ normed(W_proj[idx]).astype(np.float32)
        sims[~EN_MASK] = -1.0
        best_idx = int(np.argmax(sims))
        if best_idx == idx:
            first_k = k; break
    print("  %-20s  %s" % (word, str(first_k) if first_k else ">1536"))

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 360 — Dimensionality Transition")
print("="*70)
print()
print("  Self-retrieval reaches 100% somewhere between k=? and k=1536")
print("  (see Phase 1 output above)")
print()
print("  Axis accuracy:")
print("  EN_gender  (coherent, high): reaches full accuracy at k=?")
print("  EN_plural  (morphological):  reaches full accuracy at k=?")
print("  EN_size    (diffuse):         partially solved at low k?")
print()
print("  The dimensionality transition reveals the 'information density'")
print("  of each word type: common/clustered words need fewer dims,")
print("  rare/unique words need more dims to be distinguishable.")
