import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

def normed(v): return v/(np.linalg.norm(v)+1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None
def compute_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, 0.0, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    coh = float(np.mean([np.dot(c, md.astype(np.float32)) for c in cn]))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, coh, valid, pc

print("DAY 304: PC2 VERIFICATION, v_ord DECOMPOSITION, MORPHOLOGICAL SUBSPACE PCA")
print("="*70)
print()

# Build PC1, PC2, PC3 via power iteration
rng = np.random.default_rng(42)
N_SAMPLE = 8000
sample_ids = rng.integers(0, len(W_E), size=N_SAMPLE)
W_sample = W_E[sample_ids].astype(np.float32)
mu = W_sample.mean(axis=0)
W_c = W_sample - mu

pcs = []
W_deflated = W_c.copy()
for k in range(6):
    vk = rng.standard_normal(W_deflated.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(200):
        vk = W_deflated.T @ (W_deflated @ vk)
        vk /= np.linalg.norm(vk)
    proj = W_deflated @ vk
    W_deflated = W_deflated - np.outer(proj, vk)
    lam = float(np.var(W_c @ vk)) * W_c.shape[1]
    pcs.append((vk, lam))

tot_var = float(np.sum(np.var(W_c, axis=0)))
mu_f = mu.astype(np.float64)
pc_vecs = [v.astype(np.float64) for v, _ in pcs]
pc_vars = [lam/tot_var*100 for _, lam in pcs]

print("PC variances: " + "  ".join("PC%d=%.4f%%" % (i+1, v) for i,v in enumerate(pc_vars)))
print()

# ====================================================================
# PART A: PC2 TOP/BOTTOM TOKEN SCAN
# ====================================================================
print("PART A: PC2 top and bottom tokens (scan 50k tokens)")
print("-"*70)

BATCH = 100
n_scan = min(50000, len(W_E))
pc2_projs = []
step = max(1, n_scan // 10000)
for i in range(0, n_scan, step):
    ec = (W_E[i] - mu_f).astype(np.float64)
    pc2_projs.append((float(np.dot(ec, pc_vecs[1])), i))

pc2_projs.sort()

print("  Bottom 25 tokens on PC2 (most negative):")
for p2, tid in pc2_projs[:25]:
    w = tok.decode([tid]).strip()
    ec = (W_E[tid] - mu_f).astype(np.float64)
    p1 = float(np.dot(ec, pc_vecs[0]))
    print("    PC2=%+.4f  PC1=%+.4f  id=%-6d  '%s'" % (p2, p1, tid, w))
print()
print("  Top 25 tokens on PC2 (most positive):")
for p2, tid in pc2_projs[-25:][::-1]:
    w = tok.decode([tid]).strip()
    ec = (W_E[tid] - mu_f).astype(np.float64)
    p1 = float(np.dot(ec, pc_vecs[0]))
    print("    PC2=%+.4f  PC1=%+.4f  id=%-6d  '%s'" % (p2, p1, tid, w))
print()

# Specifically check all single-digit tokens
print("  All single-digit tokens on PC2:")
for d in ['0','1','2','3','4','5','6','7','8','9']:
    for pfx in ['', ' ']:
        ids = tok(pfx+d, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            ec = (W_E[ids[0]] - mu_f).astype(np.float64)
            p1 = float(np.dot(ec, pc_vecs[0]))
            p2 = float(np.dot(ec, pc_vecs[1]))
            print("    '%s' (id=%d)  PC1=%+.4f  PC2=%+.4f" % (pfx+d, ids[0], p1, p2))
            break
print()

# ====================================================================
# PART B: v_ord DECOMPOSITION IN PC BASIS
# ====================================================================
print("PART B: v_ord decomposition in PC1–PC6 basis")
print("-"*70)

# Build v_ord
MONTHS   = ['January','February','March','April','May','June',
            'July','August','September']
WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDS    = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ace']
CARD_N   = ['2','3','4','5','6','7','8','9','1']
SEASONS  = ['Spring','Summer','Autumn','Winter']
PLANETS  = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']

fwd_axes = []
for pairs in [
    [(MONTHS[i], str(i+1)) for i in range(9)],
    [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    list(zip(CARDS, CARD_N)),
    [(SEASONS[i], str(i+1)) for i in range(4)],
    [(PLANETS[i], str(i+1)) for i in range(8)],
]:
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, _ = compute_axis(avail)
    if ax is not None: fwd_axes.append(ax)
v_ord = normed(np.mean(fwd_axes, axis=0)).astype(np.float64)

# All known semantic/morphological axes
AXIS_PAIRS = {
    'month->num':     [(MONTHS[i], str(i+1)) for i in range(9)],
    'weekday->num':   [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    'card->num':      list(zip(CARDS, CARD_N)),
    'digit->word':    [(str(i+1), ['one','two','three','four','five','six','seven','eight','nine'][i]) for i in range(9)],
    '+er':            [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                       ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner')],
    '+est':           [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                       ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest')],
    'er->est':        [('faster','fastest'),('slower','slowest'),('taller','tallest'),('shorter','shortest'),
                       ('brighter','brightest'),('darker','darkest'),('deeper','deepest')],
    'gender':         [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                       ('son','daughter'),('brother','sister')],
    'past_irr':       [('go','went'),('come','came'),('run','ran'),('see','saw'),
                       ('eat','ate'),('know','knew'),('take','took')],
    '+ed':            [('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                       ('end','ended'),('look','looked'),('call','called')],
    '+ness':          [('sad','sadness'),('happy','happiness'),('dark','darkness'),
                       ('kind','kindness'),('bright','brightness')],
    'un-':            [('happy','unhappy'),('kind','unkind'),('fair','unfair'),
                       ('known','unknown'),('usual','unusual')],
    'country->dem':   [('France','French'),('Germany','German'),('Italy','Italian'),
                       ('Spain','Spanish'),('Japan','Japanese'),('China','Chinese')],
}

# v_ord decomposition
print("  v_ord decomposition (component along each PC):")
vord_comps = []
for i, pcv in enumerate(pc_vecs):
    c = float(np.dot(v_ord, pcv))
    vord_comps.append(c)
    print("    cos(v_ord, PC%d) = %+.4f   (var=%.4f%%)" % (i+1, c, pc_vars[i]))
total_explained = sum(c**2 for c in vord_comps)
print("  Total R² in PC1-PC6 space: %.4f (%.2f%% of v_ord direction)" %
      (total_explained, 100*total_explained))
print()

# Decompose ALL axes in PC1-PC6
print("  PC decomposition of all known axes:")
print("  %-20s  PC1      PC2      PC3      PC4      PC5      PC6" % "axis")
print("  " + "-"*70)
for nm, pairs in AXIS_PAIRS.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, pc = compute_axis(avail)
    if ax is None: continue
    ax_f = ax.astype(np.float64)
    comps = [float(np.dot(ax_f, pcv)) for pcv in pc_vecs]
    r2 = sum(c**2 for c in comps)
    print("  %-20s" % nm + "".join("  %+.4f" % c for c in comps) + "  R²=%.3f" % r2)
print()

# ====================================================================
# PART C: DEGREE TRIANGLE e2 AXIS SEMANTICS
# ====================================================================
print("PART C: What does the degree e2 (superlative elevation) axis encode?")
print("-"*70)

DEGREE_TRIPLES = [
    ('fast','faster','fastest'), ('slow','slower','slowest'),
    ('tall','taller','tallest'), ('short','shorter','shortest'),
    ('bright','brighter','brightest'), ('dark','darker','darkest'),
    ('deep','deeper','deepest'), ('clean','cleaner','cleanest'),
    ('light','lighter','lightest'), ('strong','stronger','strongest'),
]

# Build basis for degree 2D space
avail_triples = [(b,c,s) for b,c,s in DEGREE_TRIPLES
                 if get_emb(b)[0] is not None
                 and get_emb(c)[0] is not None
                 and get_emb(s)[0] is not None]

# e1 = +er direction, e2 = orthogonal complement in base-sup plane
chords_bc = [normed(get_emb(c)[0]-get_emb(b)[0]) for b,c,s in avail_triples]
chords_bs = [normed(get_emb(s)[0]-get_emb(b)[0]) for b,c,s in avail_triples]
e1_d = normed(np.mean(chords_bc, axis=0)).astype(np.float64)
e2_raw = normed(np.mean(chords_bs, axis=0)).astype(np.float64)
e2_orth = normed(e2_raw - np.dot(e2_raw, e1_d) * e1_d).astype(np.float64)

# What are e1 and e2 in terms of PCs?
print("  e1 (+er direction) PC decomposition:")
e1_comps = [(float(np.dot(e1_d, pcv)), i+1) for i, pcv in enumerate(pc_vecs)]
for c, i in sorted(e1_comps, key=lambda x: -abs(x[0])):
    if abs(c) > 0.05: print("    PC%d: %+.4f" % (i, c))
print()
print("  e2_orth (superlative elevation) PC decomposition:")
e2_comps = [(float(np.dot(e2_orth, pcv)), i+1) for i, pcv in enumerate(pc_vecs)]
for c, i in sorted(e2_comps, key=lambda x: -abs(x[0])):
    if abs(c) > 0.05: print("    PC%d: %+.4f" % (i, c))
print()

# Does e2 align with v_ord?
c_e2_vord = float(np.dot(e2_orth, v_ord))
print("  cos(e2_orth, v_ord) = %+.4f" % c_e2_vord)
print()

# What known axes align with e2?
print("  Known axis alignments with e2_orth:")
for nm, pairs in AXIS_PAIRS.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, _ = compute_axis(avail)
    if ax is None: continue
    c = float(np.dot(ax.astype(np.float64), e2_orth))
    if abs(c) > 0.1:
        print("    %-22s  cos=%+.4f" % (nm, c))
print()

# ====================================================================
# PART D: MORPHOLOGICAL SUBSPACE PCA
# ====================================================================
print("PART D: PCA of morphological transformation subspace")
print("-"*70)

# Collect all morphological chord vectors
MORPH_PAIRS = [
    ('fast','faster'), ('slow','slower'), ('tall','taller'), ('short','shorter'),
    ('bright','brighter'), ('dark','darker'), ('deep','deeper'), ('clean','cleaner'),
    ('fast','fastest'), ('slow','slowest'), ('tall','tallest'), ('short','shortest'),
    ('bright','brightest'), ('dark','darkest'), ('deep','deepest'), ('clean','cleanest'),
    ('king','queen'), ('man','woman'), ('boy','girl'), ('father','mother'),
    ('go','went'), ('come','came'), ('run','ran'), ('see','saw'),
    ('walk','walked'), ('talk','talked'), ('jump','jumped'), ('start','started'),
    ('cat','cats'), ('dog','dogs'), ('house','houses'), ('car','cars'),
    ('sad','sadness'), ('happy','happiness'), ('dark','darkness'),
    ('hope','hopeful'), ('care','careful'), ('use','useful'),
    ('happy','unhappy'), ('kind','unkind'), ('fair','unfair'),
    ('achieve','achievement'), ('manage','management'), ('move','movement'),
]

morph_chords = []
morph_labels = []
for s, t in MORPH_PAIRS:
    es, sid = get_emb(s)
    et, tid = get_emb(t)
    if es is None or et is None: continue
    morph_chords.append(normed(et-es).astype(np.float32))
    morph_labels.append((s,t))

M_mat = np.array(morph_chords)  # shape: (n_pairs, d)
print("  Morphological chord matrix shape: %s" % str(M_mat.shape))

# PCA on the chord matrix
M_mean = M_mat.mean(axis=0)
M_c = M_mat - M_mean

rng2 = np.random.default_rng(0)
M_pcs = []
M_deflated = M_c.copy()
for k in range(8):
    vk = rng2.standard_normal(M_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(100):
        vk = M_deflated.T @ (M_deflated @ vk)
        vk /= np.linalg.norm(vk)
    proj = M_deflated @ vk
    M_deflated = M_deflated - np.outer(proj, vk)
    lam = float(np.var(M_c @ vk)) * M_c.shape[1]
    M_pcs.append((vk, lam))

M_tot = float(np.sum(np.var(M_c, axis=0)))
print("  Morphological subspace PC variances:")
for i, (vk, lam) in enumerate(M_pcs):
    print("    mPC%d: %.4f%%" % (i+1, 100*lam/M_tot))

# What do the morphological PCs align with?
print()
print("  Morphological PC alignment with known W_E axes:")
for k, (vk, _) in enumerate(M_pcs[:4]):
    vk_f = vk.astype(np.float64)
    print("  mPC%d alignments (|cos|>0.15):" % (k+1))
    aligns = []
    for nm, pairs in AXIS_PAIRS.items():
        avail = [(s,t) for s,t in pairs
                 if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
        if len(avail) < 2: continue
        ax, _, _, _ = compute_axis(avail)
        if ax is None: continue
        c = float(np.dot(ax.astype(np.float64), vk_f))
        if abs(c) > 0.15: aligns.append((abs(c), c, nm))
    aligns.sort(reverse=True)
    for _, c, nm in aligns:
        print("    %-22s  cos=%+.4f" % (nm, c))
    # Also check alignment with global PCs
    for pi, pcv in enumerate(pc_vecs[:4]):
        c = float(np.dot(vk_f, pcv))
        if abs(c) > 0.10:
            print("    %-22s  cos=%+.4f  (global PC%d)" % ('', c, pi+1))
    print()

# ====================================================================
# PART E: THE W_E SEMANTIC MAP — WHAT WE KNOW SO FAR
# ====================================================================
print("="*70)
print("PART E: W_E semantic map summary (after Day 304)")
print("="*70)
print()
print("  Known directions in W_E (1536-dimensional space):")
print("  %-28s  var%%    interpretation" % "Direction")
print("  " + "-"*65)
print("  %-28s  3.35%%   token frequency/specificity" % "PC1")
print("  %-28s  ~?%%     digit symbol axis" % "PC2")
print("  %-28s  ~?%%     morphological modification" % "PC3")
print("  %-28s  1.91%%   ordinal direction (name->digit)" % "v_ord")
print("  %-28s  0.16%%   degree comparation (horizontal)" % "e1 (+er)")
print("  %-28s  0.XX%%   superlative elevation" % "e2_orth (+est-specific)")
print("  %-28s  0.79%%   morphological subspace (5 axes)" % "subspace M")
print()
print("  Subspace relationships:")
print("  cos(v_ord, PC1) = -0.720  [v_ord anti-aligned with frequency]")
print("  cos(v_ord, all morph axes) < 0.20  [labelling orthogonal to morphology]")
print("  cos(past_irr, past_reg) = +0.442  [same function, different form]")
print("  cos(gender, +er) = +0.063  [cross-domain orthogonality]")
print()
# Re-print v_ord PC decomposition
print("  v_ord PC decomposition (PC1-PC6):")
for i, c in enumerate(vord_comps):
    print("    PC%d: %+.4f  (%.2f%% of v_ord²)" % (i+1, c, 100*c**2))
