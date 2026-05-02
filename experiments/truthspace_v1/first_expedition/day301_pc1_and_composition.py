import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr

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
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 301: PC1 EXPLORATION, MULTI-AXIS RETRIEVAL, AXIS COMPOSITION")
print("="*65)
print()

MONTHS   = ['January','February','March','April','May','June',
            'July','August','September','October','November','December']
WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDINALS_WORD = ['one','two','three','four','five','six','seven','eight','nine']
CARDINALS_STR  = ['1','2','3','4','5','6','7','8','9']

# ====================================================================
# PART A: GLOBAL PC1 OF W_E
# What is the most important direction in W_E?
# ====================================================================
print("PART A: Global PC1 of W_E")
print("-"*65)

rng = np.random.default_rng(42)
# Use a larger, structured sample: top-k frequent tokens by index
# (lower token IDs tend to be more frequent in BPE)
N_SAMPLE = 8000
sample_ids = rng.integers(0, len(W_E), size=N_SAMPLE)
W_sample = W_E[sample_ids].astype(np.float32)
mu = W_sample.mean(axis=0)
W_c = W_sample - mu

# Power iteration for top eigenvector
v1 = rng.standard_normal(W_c.shape[1]).astype(np.float32)
v1 = v1 / np.linalg.norm(v1)
for _ in range(200):
    v1 = W_c.T @ (W_c @ v1)
    v1 = v1 / np.linalg.norm(v1)

# Get PC2 as well
proj1 = W_c @ v1
W_c2 = W_c - np.outer(proj1, v1)
v2 = rng.standard_normal(W_c.shape[1]).astype(np.float32)
v2 = v2 / np.linalg.norm(v2)
for _ in range(200):
    v2 = W_c2.T @ (W_c2 @ v2)
    v2 = v2 / np.linalg.norm(v2)

# Variance explained
lambda1 = float(np.var(W_c @ v1)) * W_c.shape[1]
lambda2 = float(np.var(W_c @ v2)) * W_c.shape[1]
total_var = float(np.sum(np.var(W_c, axis=0)))
print("  PC1 explains: %.4f%%  PC2 explains: %.4f%%" % (
    100*lambda1/total_var, 100*lambda2/total_var))
print()

# Project known tokens onto PC1 and PC2
print("  Projections of key tokens onto PC1 and PC2:")
test_groups = [
    ('Digits',      ['1','2','3','4','5','6','7','8','9']),
    ('CardWords',   ['one','two','three','four','five','six','seven','eight','nine']),
    ('Months',      ['January','February','March','July','December']),
    ('Weekdays',    ['Monday','Tuesday','Wednesday','Saturday','Sunday']),
    ('Function',    ['the','and','is','of','to','a','in','that','it','for']),
    ('Punct/Sym',   ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')']),
    ('Common adj',  ['big','small','good','bad','hot','cold','fast','slow']),
    ('Common verb', ['run','walk','go','come','see','know','make','take']),
]
mu_f = mu.astype(np.float64)
v1_f = v1.astype(np.float64)
v2_f = v2.astype(np.float64)

print("  %-16s  %-14s  PC1      PC2" % ("group", "word"))
print("  " + "-"*52)
for group, words in test_groups:
    for w in words:
        e, sid = get_emb(w)
        if e is None: continue
        p1 = float(np.dot(e - mu_f, v1_f))
        p2 = float(np.dot(e - mu_f, v2_f))
        print("  %-16s  %-14s  %+.4f  %+.4f" % (group, w, p1, p2))
    print()

# What is PC1 measuring? Check correlation with:
# 1. Token frequency (approximated by token ID -- lower ID = more frequent in BPE)
# 2. Word length
# 3. Punctuation flag
print("  PC1 interpretation test:")
sample_size = 2000
test_ids = list(range(0, min(sample_size * 2, len(W_E)), 2))[:sample_size]
pc1_vals = []
tok_ids  = []
for tid in test_ids:
    e = W_E[tid]
    p1 = float(np.dot(e - mu_f, v1_f))
    pc1_vals.append(p1)
    tok_ids.append(tid)

r_freq, p_freq = pearsonr(tok_ids, pc1_vals)
print("  Pearson r(token_ID, PC1) = %.4f  p=%.4e" % (r_freq, p_freq))

# Word length correlation
lengths = [len(tok.decode([i]).strip()) for i in tok_ids]
r_len, p_len = pearsonr(lengths, pc1_vals)
print("  Pearson r(word_length, PC1) = %.4f  p=%.4e" % (r_len, p_len))
print()

# ====================================================================
# PART B: MULTI-AXIS RETRIEVAL
# Can we combine two axes to retrieve a word from TWO constraints?
# Test: weekday "Monday" + ordinal offset=2 -> "Wednesday" (3rd weekday)
# ====================================================================
print("PART B: Multi-axis retrieval (two simultaneous constraints)")
print("-"*65)
print("  Idea: source + scale1*ax1 + scale2*ax2 -> target")
print()

# Axis 1: weekday shift (Monday->Tuesday->Wednesday pattern)
# Use Monday->Wednesday as training chord
wd_shift_pairs = [(WEEKDAYS[i], WEEKDAYS[i+2]) for i in range(5)
                  if get_emb(WEEKDAYS[i])[0] is not None
                  and get_emb(WEEKDAYS[i+2])[0] is not None]
ax_wd2, _, valid_wd2, pc_wd2 = compute_axis(wd_shift_pairs)
print("  Weekday +2 shift axis: pc=%.4f  n=%d" % (pc_wd2, len(wd_shift_pairs)))

# Axis 2: ordinal +1 direction (month->number shift)
mn_pairs = [(MONTHS[i], str(i+1)) for i in range(9)
            if get_emb(MONTHS[i])[0] is not None and get_emb(str(i+1))[0] is not None]
ax_mn, _, valid_mn, _ = compute_axis(mn_pairs)

# Test: given "Monday" (weekday 1), shift by +2 weekdays
print("  Test A: Monday + wd_shift_+2 -> Wednesday?")
e_mon, sid_mon = get_emb('Monday')
if e_mon is not None and ax_wd2 is not None:
    s_wd2, acc_wd2 = best_scale(ax_wd2, valid_wd2)
    pred = W_E[sid_mon] + s_wd2 * ax_wd2
    r = nn_retrieve(pred, [sid_mon], top_n=5)
    print("  scale=%.2f  top5: %s" % (s_wd2, ', '.join(w for w,_,_ in r)))

# Test: month consecutive shift axis
month_shift_pairs = [(MONTHS[i], MONTHS[i+2]) for i in range(10)
                     if get_emb(MONTHS[i])[0] is not None
                     and get_emb(MONTHS[i+2])[0] is not None]
ax_m2, _, valid_m2, pc_m2 = compute_axis(month_shift_pairs)
print()
print("  Month +2 shift axis: pc=%.4f  n=%d" % (pc_m2, len(month_shift_pairs)))
print("  Test B: January + month_shift_+2 -> March?")
e_jan, sid_jan = get_emb('January')
if e_jan is not None and ax_m2 is not None:
    s_m2, acc_m2 = best_scale(ax_m2, valid_m2)
    pred = W_E[sid_jan] + s_m2 * ax_m2
    r = nn_retrieve(pred, [sid_jan], top_n=5)
    print("  scale=%.2f  acc=%d/%d  top5: %s" % (
        s_m2, acc_m2, len(valid_m2), ', '.join(w for w,_,_ in r)))
    print("  Full month +2 results:")
    for s, t, sid, _ in valid_m2:
        r = nn_retrieve(W_E[sid]+s_m2*ax_m2, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-12s +2-> %-12s  got=%-12s [%s]" % (
            s, t, r[0][0], 'HIT' if hit else '---'))
print()

# ====================================================================
# PART C: AXIS COMPOSITION
# Does ax_A->B + ax_B->C ≈ ax_A->C?
# ====================================================================
print("PART C: Axis composition")
print("-"*65)
print("  Testing: ax(A->B) + ax(B->C) ?= ax(A->C)")
print()

# Test 1: month->number + number->cardinal_word ?= month->cardinal_word
mn_avail = [(MONTHS[i], str(i+1)) for i in range(9)
            if get_emb(MONTHS[i])[0] is not None and get_emb(str(i+1))[0] is not None]
nw_avail = [(str(i+1), CARDINALS_WORD[i]) for i in range(9)
            if get_emb(str(i+1))[0] is not None and get_emb(CARDINALS_WORD[i])[0] is not None]
mw_avail = [(MONTHS[i], CARDINALS_WORD[i]) for i in range(9)
            if get_emb(MONTHS[i])[0] is not None and get_emb(CARDINALS_WORD[i])[0] is not None]

ax_mn_d, _, _, pc_mn = compute_axis(mn_avail)   # month -> digit
ax_nw_d, _, _, pc_nw = compute_axis(nw_avail)   # digit -> cardinal word
ax_mw_d, _, valid_mw, pc_mw = compute_axis(mw_avail)  # month -> cardinal word (direct)

if ax_mn_d is not None and ax_nw_d is not None and ax_mw_d is not None:
    # Composed axis (not normalised)
    ax_mn_raw = np.mean([normed(get_emb(t)[0] - get_emb(s)[0]) for s,t in mn_avail
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_nw_raw = np.mean([normed(get_emb(t)[0] - get_emb(s)[0]) for s,t in nw_avail
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_composed = normed(ax_mn_raw + ax_nw_raw)
    ax_mw_norm = ax_mw_d

    cos_comp_direct = float(np.dot(ax_composed.astype(np.float32),
                                    ax_mw_norm.astype(np.float32)))
    cos_mn_mw = float(np.dot(ax_mn_d.astype(np.float32), ax_mw_norm.astype(np.float32)))
    cos_nw_mw = float(np.dot(ax_nw_d.astype(np.float32), ax_mw_norm.astype(np.float32)))

    print("  Test 1: month->digit + digit->word ?= month->word")
    print("  ax_month->digit: pc=%.4f" % pc_mn)
    print("  ax_digit->word:  pc=%.4f" % pc_nw)
    print("  ax_month->word (direct): pc=%.4f" % pc_mw)
    print()
    print("  cos(composed, direct) = %.4f  [1.0 = perfect composition]" % cos_comp_direct)
    print("  cos(month->digit, month->word) = %.4f" % cos_mn_mw)
    print("  cos(digit->word, month->word) = %.4f" % cos_nw_mw)
    print()

    # Test composed axis for retrieval
    s_comp, acc_comp = best_scale(ax_composed, valid_mw)
    print("  Composed axis retrieval (month->word):")
    print("  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        s_comp, acc_comp, len(valid_mw), 100*acc_comp/max(1,len(valid_mw))))
    for s, t, sid, _ in valid_mw:
        r = nn_retrieve(W_E[sid]+s_comp*ax_composed, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-12s -> %-8s  got=%-8s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# Test 2: weekday->number + number->ordinal_word ?= weekday->ordinal
ORDINALS_WORD = ['first','second','third','fourth','fifth','sixth','seventh']
wo_avail = [(str(i+1), ORDINALS_WORD[i]) for i in range(7)
            if get_emb(str(i+1))[0] is not None and get_emb(ORDINALS_WORD[i])[0] is not None]
wdo_avail = [(WEEKDAYS[i], ORDINALS_WORD[i]) for i in range(7)
             if get_emb(WEEKDAYS[i])[0] is not None and get_emb(ORDINALS_WORD[i])[0] is not None]

wd_avail = [(WEEKDAYS[i], str(i+1)) for i in range(7)
            if get_emb(WEEKDAYS[i])[0] is not None and get_emb(str(i+1))[0] is not None]

ax_wd_d, _, _, _ = compute_axis(wd_avail)
ax_no_d, _, _, pc_no = compute_axis(wo_avail)     # num -> ordinal
ax_wdo_d, _, valid_wdo, pc_wdo = compute_axis(wdo_avail)  # weekday -> ordinal (direct)

if ax_wd_d is not None and ax_no_d is not None and ax_wdo_d is not None:
    ax_wd_raw = np.mean([normed(get_emb(t)[0] - get_emb(s)[0]) for s,t in wd_avail
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_no_raw = np.mean([normed(get_emb(t)[0] - get_emb(s)[0]) for s,t in wo_avail
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_comp2 = normed(ax_wd_raw + ax_no_raw)

    cos2 = float(np.dot(ax_comp2.astype(np.float32), ax_wdo_d.astype(np.float32)))
    print("  Test 2: weekday->num + num->ordinal ?= weekday->ordinal")
    print("  ax_num->ordinal:  pc=%.4f" % pc_no)
    print("  ax_weekday->ordinal (direct): pc=%.4f" % pc_wdo)
    print("  cos(composed, direct) = %.4f" % cos2)
    print()
    s_c2, acc_c2 = best_scale(ax_comp2, valid_wdo)
    print("  Composed axis retrieval (weekday->ordinal):")
    print("  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        s_c2, acc_c2, len(valid_wdo), 100*acc_c2/max(1,len(valid_wdo))))
    for s, t, sid, _ in valid_wdo:
        r = nn_retrieve(W_E[sid]+s_c2*ax_comp2, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-12s -> %-8s  got=%-8s [%s]" % (s, t, r[0][0], 'HIT' if hit else '---'))
    print()

# Test 3: Morphological composition
# +er + +er ?= +est   (more+more ≈ most)?
er_pairs = [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
            ('bright','brighter'),('dark','darker'),('clean','cleaner'),('deep','deeper')]
est_pairs = [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
             ('bright','brightest'),('dark','darkest'),('clean','cleanest'),('deep','deepest')]
er_er_pairs = [('faster','fastest'),('slower','slowest'),('taller','tallest'),
               ('shorter','shortest'),('brighter','brightest'),('darker','darkest')]

ax_er, _, _, pc_er = compute_axis(er_pairs)
ax_est, _, valid_est, pc_est = compute_axis(est_pairs)
ax_er_er, _, valid_erer, pc_erer = compute_axis(er_er_pairs)

if ax_er is not None and ax_est is not None:
    ax_er_raw = np.mean([normed(get_emb(t)[0] - get_emb(s)[0]) for s,t in er_pairs
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_comp3 = normed(ax_er_raw + ax_er_raw)  # +er twice
    cos3 = float(np.dot(ax_comp3.astype(np.float32), ax_est.astype(np.float32)))
    print("  Test 3: +er + +er ?= +est  (comp-comp composition)")
    print("  ax_+er: pc=%.4f  ax_+est: pc=%.4f" % (pc_er, pc_est))
    if ax_er_er is not None:
        print("  ax_comparative->superlative (er->est): pc=%.4f" % pc_erer)
        cos_er_est = float(np.dot(ax_er.astype(np.float32), ax_est.astype(np.float32)))
        print("  cos(+er, +est) = %.4f  [do they point the same way?]" % cos_er_est)
    print("  cos(+er+er composed, +est direct) = %.4f" % cos3)
    print()

# ====================================================================
# PART D: WHAT DOES PC1 MEASURE? -- DEEPER ANALYSIS
# ====================================================================
print("PART D: PC1 measurement -- deeper analysis")
print("-"*65)

# Top and bottom tokens on PC1
proj_all = []
for i in range(0, min(50000, len(W_E)), 5):
    e = W_E[i]
    p = float(np.dot(e - mu_f, v1_f))
    proj_all.append((p, i))

proj_all.sort()
print("  Bottom 20 tokens on PC1 (lowest projection):")
for p, tid in proj_all[:20]:
    w = tok.decode([tid]).strip()
    print("    %+.4f  id=%-6d  '%s'" % (p, tid, w))
print()
print("  Top 20 tokens on PC1 (highest projection):")
for p, tid in proj_all[-20:][::-1]:
    w = tok.decode([tid]).strip()
    print("    %+.4f  id=%-6d  '%s'" % (p, tid, w))
print()

# ====================================================================
# PART E: AXIS ALGEBRA SUMMARY
# ====================================================================
print("="*65)
print("PART E: Axis algebra summary (Day 301)")
print("="*65)

# Compile all composition tests
print()
print("  Composition test results:")
print("  %-40s  cos" % "Test")
print("  " + "-"*50)

if ax_mn_d is not None and ax_nw_d is not None and ax_mw_d is not None:
    print("  %-40s  %.4f" % ("month->digit + digit->word = month->word", cos_comp_direct))
if ax_wd_d is not None and ax_no_d is not None and ax_wdo_d is not None:
    print("  %-40s  %.4f" % ("weekday->num + num->ordinal = weekday->ord", cos2))
if ax_er is not None and ax_est is not None:
    print("  %-40s  %.4f" % ("+er + +er = +est", cos3))
    print("  %-40s  %.4f" % ("cos(+er, +est)", float(np.dot(
        ax_er.astype(np.float32), ax_est.astype(np.float32)))))

print()
print("  Interpretation:")
print("  cos > 0.9: composition works (axes nearly additive)")
print("  cos 0.7-0.9: approximate composition (partial additivity)")
print("  cos < 0.7: composition fails (non-linear intermediate)")
