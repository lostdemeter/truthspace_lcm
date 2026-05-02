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

print("DAY 300 -- GRAND SYNTHESIS: THE COMPLETE LINEARITY MAP")
print("="*65)
print("Milestone day: 300 days of W_E geometry exploration.")
print("1. Universal ordinal direction vector")
print("2. Card->number holdout")
print("3. Ordinal direction as W_E coordinate")
print("4. Complete linearity spectrum -- all axes on one scale")
print()

MONTHS  = ['January','February','March','April','May','June',
           'July','August','September','October','November','December']
WEEKDAYS= ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDINALS_WORD = ['one','two','three','four','five','six','seven','eight','nine']
CARD_NAMES = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ace']
CARD_NUMS  = ['2',  '3',    '4',    '5',    '6',    '7',    '8',    '9',  '1']
PLANETS    = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
SEASONS    = ['Spring','Summer','Autumn','Winter']
LETTERS    = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# ====================================================================
# PART A: UNIVERSAL ORDINAL DIRECTION (mean of all labelling axes)
# ====================================================================
print("PART A: Universal ordinal direction vector")
print("-"*65)

labelling_axis_pairs = {
    'month->num':    [(MONTHS[i], str(i+1)) for i in range(9)],
    'weekday->num':  [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    'card->num':     list(zip(CARD_NAMES, CARD_NUMS)),
    'season->qtr':   [(SEASONS[i], str(i+1)) for i in range(4)],
    'planet->orb':   [(PLANETS[i], str(i+1)) for i in range(8)],
    'letter->pos':   [(LETTERS[i], str(i+1)) for i in range(26)
                      if get_emb(LETTERS[i])[0] is not None and
                         get_emb(str(i+1))[0] is not None],
    'digit->word':   [(str(i+1), CARDINALS_WORD[i]) for i in range(9)],
}

axes_computed = {}
print("  Computing individual labelling axes:")
for name, pairs in labelling_axis_pairs.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, coh, valid, pc = compute_axis(avail)
    if ax is None: continue
    axes_computed[name] = (ax, pc, coh, len(valid))
    print("  %-16s  n=%-2d  pc=%+.4f  coh=%.4f" % (name, len(valid), pc, coh))

# Compute mean ordinal direction (weighted by coh for reliability)
print()
print("  Computing universal ordinal direction:")

# Version 1: simple mean of all labelling axes (except digit->word which is reversed)
forward_axes = {k: v for k, v in axes_computed.items() if k != 'digit->word'}
ax_vectors_fwd = [ax for ax, _, _, _ in forward_axes.values()]
v_ord_simple = normed(np.mean(ax_vectors_fwd, axis=0))

# Version 2: include digit->word reversed
ax_dw = axes_computed.get('digit->word', (None,))[0]
if ax_dw is not None:
    ax_dw_rev = normed(-ax_dw)
    all_forward = ax_vectors_fwd + [ax_dw_rev]
    v_ord_all = normed(np.mean(all_forward, axis=0))
else:
    v_ord_all = v_ord_simple

# Measure each axis's alignment with universal direction
print("  Cosines with universal ordinal direction (v_ord):")
for name, (ax, pc, coh, n) in axes_computed.items():
    c_fwd = float(np.dot(ax.astype(np.float32), v_ord_simple.astype(np.float32)))
    print("  %-16s  cos(axis, v_ord) = %+.4f" % (name, c_fwd))
print()

# Internal coherence of the universal direction
all_ax_vecs = [ax for ax, _, _, _ in axes_computed.values()]
ax_norms = [normed(a).astype(np.float32) for a in all_ax_vecs]
pairwise_c = [np.dot(ax_norms[i], ax_norms[j])
              for i in range(len(ax_norms)) for j in range(i+1, len(ax_norms))]
print("  Inter-axis mean cosine (all labelling axes incl. digit->word): %.4f" %
      np.mean(pairwise_c))
print("  (Note: digit->word is reversed, so expected ~-0.7 on average)")

# Compute without digit->word
fwd_norms = [normed(ax).astype(np.float32) for ax in ax_vectors_fwd]
pairwise_fwd = [np.dot(fwd_norms[i], fwd_norms[j])
                for i in range(len(fwd_norms)) for j in range(i+1, len(fwd_norms))]
print("  Inter-axis mean cosine (forward axes only, excl. digit->word): %.4f" %
      np.mean(pairwise_fwd))
print()

# ====================================================================
# PART B: CARD->NUMBER HOLDOUT
# ====================================================================
print("PART B: Card->number holdout")
print("-"*65)

# Train: Two, Three, Four, Five, Six
# Hold:  Seven, Eight, Nine, Ace
CARD_TRAIN = [('Two','2'),('Three','3'),('Four','4'),('Five','5'),('Six','6')]
CARD_HOLD  = [('Seven','7'),('Eight','8'),('Nine','9'),('Ace','1')]

tr_avail = [(s,t) for s,t in CARD_TRAIN
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
ho_avail = [(s,t) for s,t in CARD_HOLD
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]

ax_ct, coh_ct, valid_ct, pc_ct = compute_axis(tr_avail)
if ax_ct is not None:
    s_ct, acc_ct = best_scale(ax_ct, valid_ct)
    print("  Train (Two-Six): pc=%.4f  coh=%.4f  scale=%.2f  %d/%d (%.0f%%)" % (
        pc_ct, coh_ct, s_ct, acc_ct, len(valid_ct), 100*acc_ct/max(1,len(valid_ct))))
    print()
    hold_hits = 0
    print("  Holdout results:")
    for s, t in ho_avail:
        es, sid = get_emb(s)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+s_ct*ax_ct, [sid], top_n=5)
        got = r[0][0]
        hit = got == t
        if hit: hold_hits += 1
        print("    %-8s -> %s  got=%-6s [%s]  top5: %s" % (
            s, t, got, 'HIT' if hit else '---',
            ', '.join(w for w,_,_ in r)))
    print("  Holdout: %d/%d (%.0f%%)" % (hold_hits, len(ho_avail), 100*hold_hits/max(1,len(ho_avail))))
print()

# ====================================================================
# PART C: ORDINAL DIRECTION AS W_E COORDINATE
# How much variance in W_E does the ordinal direction explain?
# ====================================================================
print("PART C: Ordinal direction as W_E coordinate")
print("-"*65)

# Project ALL embeddings onto the universal ordinal direction
v_ord_fp = v_ord_simple.astype(np.float64)
v_ord_norm = v_ord_fp / (np.linalg.norm(v_ord_fp) + 1e-8)

# Sample 5000 random tokens for efficiency
rng = np.random.default_rng(42)
sample_ids = rng.integers(0, len(W_E), size=5000)
sample_embs = W_E[sample_ids]

# Project onto ordinal direction
proj_ord = sample_embs @ v_ord_norm  # shape: (5000,)

# Total variance in sample
var_total = float(np.var(sample_embs))  # mean over all dimensions
var_ord   = float(np.var(proj_ord))
print("  Variance explained by ordinal direction: %.6f / %.6f = %.4f%%" % (
    var_ord, var_total * sample_embs.shape[1], 100*var_ord / (var_total * sample_embs.shape[1])))

# Compare: how much variance does the global PC1 of W_E explain?
# Sample for PCA
W_sample = sample_embs.astype(np.float32)
W_c = W_sample - W_sample.mean(axis=0)
# Use fast covariance approach
cov = (W_c.T @ W_c) / len(W_c)
# Get top eigenvalue via power iteration (fast)
v = rng.standard_normal(W_c.shape[1]).astype(np.float32)
v = v / np.linalg.norm(v)
for _ in range(50):
    v = cov @ v
    v = v / np.linalg.norm(v)
lambda_1 = float(v @ cov @ v)
total_var = float(np.trace(cov))
print("  Global PC1 explains: %.4f%% of total variance" % (100*lambda_1/total_var))
print()

# Project known ordinal words onto ordinal direction
print("  Projections of key tokens onto ordinal direction:")
test_words = ['1','2','3','4','5','6','7','8','9',
              'one','two','three','four','five','six','seven','eight','nine',
              'January','February','March','July','December',
              'Monday','Tuesday','Wednesday','Saturday','Sunday',
              'first','second','third','tenth',
              'Two','Three','Four','Nine','Ace',
              'the','and','is','of','to']  # common words for comparison
print("  %-14s  proj_ord" % "word")
print("  " + "-"*25)
for w in test_words:
    e, sid = get_emb(w)
    if e is None: continue
    p = float(np.dot(e, v_ord_norm))
    print("  %-14s  %+.4f" % (w, p))
print()

# ====================================================================
# PART D: ENCODE=DECODE SYMMETRY FOR LABELLING AXES
# ====================================================================
print("PART D: ENCODE=DECODE symmetry for labelling axes")
print("-"*65)

print("  Testing cos(forward, reverse) for each labelling axis:")
for name, pairs in labelling_axis_pairs.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax_fwd, _, _, _ = compute_axis(avail)
    ax_rev, _, _, _ = compute_axis([(t,s) for s,t in avail])
    if ax_fwd is None or ax_rev is None: continue
    c = float(np.dot(ax_fwd.astype(np.float32), ax_rev.astype(np.float32)))
    fwd_scale = best_scale(ax_fwd, [v for _,_,v,_ in [compute_axis(avail)]], lo=0.02, hi=4.0, n=50)[0] if False else 0.0
    print("  %-16s  cos(fwd, rev) = %+.4f" % (name, c))
print()

# ====================================================================
# PART E: COMPLETE LINEARITY SPECTRUM (ALL 35+ AXES)
# ====================================================================
print("="*65)
print("COMPLETE LINEARITY SPECTRUM — DAY 300")
print("All axes measured across 300 days of experiment")
print("="*65)

FULL_SPECTRUM = [
    # LABELLING (name/symbol -> ordinal number)
    ("digit->word",         0.851, "LABEL", "digit symbols -> spoken names"),
    ("weekday->number",     0.842, "LABEL", "Mon-Sun -> 1-7"),
    ("month->number",       0.803, "LABEL", "Jan-Sep -> 1-9"),
    ("card->number",        0.789, "LABEL", "Two-Nine,Ace -> 1-9"),
    ("season->quarter",     0.691, "LABEL", "Spring-Winter -> 1-4, n=4"),
    ("planet->orbital",     0.609, "LABEL", "Mercury-Neptune -> 1-8, *attractor"),
    ("ordinal->cardinal",   0.582, "LABEL", "first-tenth -> one-ten"),
    ("letter->alpha-pos",   0.504, "LABEL", "A-I -> 1-9"),
    # SEMANTIC
    ("country->demonym",    0.563, "SEMAN", "France->French"),
    ("country->lang",       0.474, "SEMAN", "France->French *inflated"),
    ("elem:single-letter",  0.390, "SEMAN", "H,C,N,O -> symbol"),
    ("country->capital",    0.317, "SEMAN", "France->Paris"),
    ("animal->class",       0.254, "SEMAN", "dog->mammal"),
    ("person->nationality", 0.246, "SEMAN", "French->France"),
    ("elem:double-letter",  0.163, "SEMAN", "Ca,Fe -> symbol"),
    ("element->symbol",     0.139, "SEMAN", "Hydrogen->H"),
    ("field->concept",      0.087, "SEMAN", "physics->energy"),
    ("word->antonym",       0.020, "SEMAN", "hot->cold (diversified)"),
    # INFLECTIONAL
    ("+est (superlative)",  0.436, "INFL ", "fast->fastest"),
    ("+er (comparative)",   0.393, "INFL ", "fast->faster"),
    ("past_irr",            0.230, "INFL ", "go->went"),
    ("gender",              0.213, "INFL ", "king->queen"),
    ("+ed (past_reg)",      0.174, "INFL ", "walk->walked"),
    ("+s plural",           0.155, "INFL ", "cat->cats"),
    # DERIVATIONAL
    ("+ness",               0.211, "DERIV", "sad->sadness"),
    ("+ment",               0.124, "DERIV", "achieve->achievement"),
    ("un-",                 0.121, "DERIV", "happy->unhappy"),
    ("in-/im-",             0.133, "DERIV", "possible->impossible"),
    ("+less",               0.133, "DERIV", "hope->hopeless"),
    ("+tion",               0.130, "DERIV", "act->action"),
    ("+ful",                0.104, "DERIV", "hope->hopeful"),
    # TEMPORAL/NUMERIC (cyclic/non-uniform -> negative pc)
    ("month (consec)",     -0.090, "CYCL ", "Jan->Feb->...->Dec"),
    ("digit n->n+1",       -0.115, "NONUN", "1->2->...->9"),
    ("digit n->n+2",       -0.076, "NONUN", "1->3, 2->4, ..."),
    ("digit n->n+3",       -0.006, "NONUN", "1->4, 2->5, ..."),
    ("weekday (consec)",   -0.153, "CYCL ", "Mon->Tue->...->Sun"),
]

# Compute actual values for the new labelling axes today
new_pc = {}
for name, pairs in [
    ('card->number', list(zip(CARD_NAMES, CARD_NUMS))),
    ('season->quarter', [(SEASONS[i], str(i+1)) for i in range(4)]),
    ('planet->orbital', [(PLANETS[i], str(i+1)) for i in range(8)]),
    ('letter->alpha-pos', [(LETTERS[i], str(i+1)) for i in range(26)]),
    ('digit->word', [(str(i+1), CARDINALS_WORD[i]) for i in range(9)]),
    ('weekday->number', [(WEEKDAYS[i], str(i+1)) for i in range(7)]),
    ('month->number', [(MONTHS[i], str(i+1)) for i in range(9)]),
]:
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) >= 2:
        _, _, _, pc = compute_axis(avail)
        if pc is not None: new_pc[name] = pc

# Update FULL_SPECTRUM with measured values
updated = []
for row in FULL_SPECTRUM:
    name, pc, cat, desc = row
    if name in new_pc:
        updated.append((name, new_pc[name], cat, desc))
    else:
        updated.append(row)
updated.sort(key=lambda x: -x[1])

print()
print("  %-28s  %+.4s  %-5s  %s" % ("Axis", "pc", "Type", "Description"))
print("  " + "-"*72)
prev_sign = 1
for name, pc, cat, desc in updated:
    if pc < 0 and prev_sign >= 0:
        print("  " + "-"*72 + "  [ pc < 0 boundary ]")
        prev_sign = -1
    print("  %-28s  %+.4f  %-5s  %s" % (name, pc, cat, desc))

print()
print("SUMMARY:")
pos_axes = [(n,p,c) for n,p,c,_ in updated if p >= 0]
neg_axes = [(n,p,c) for n,p,c,_ in updated if p < 0]
for cat in ['LABEL','SEMAN','INFL ','DERIV','CYCL ','NONUN']:
    c_pos = [p for n,p,c in pos_axes if c==cat]
    c_neg = [p for n,p,c in neg_axes if c==cat]
    all_c = c_pos + c_neg
    if not all_c: continue
    print("  %-6s  n=%-2d  mean_pc=%+.4f  range=[%+.4f, %+.4f]" % (
        cat, len(all_c), np.mean(all_c), min(all_c), max(all_c)))

print()
print("Tier boundaries:")
print("  TIER 1 (pc > 0.58):  LABELLING axes only  (%d axes)" %
      sum(1 for _,p,_ in pos_axes if p > 0.58))
print("  TIER 2 (pc 0.30-0.58): SEMANTIC high + INFLECTIONAL top  (%d axes)" %
      sum(1 for _,p,_ in pos_axes if 0.30 <= p <= 0.58))
print("  TIER 3 (pc 0.10-0.30): INFLECTIONAL + DERIVATIONAL  (%d axes)" %
      sum(1 for _,p,_ in pos_axes if 0.10 <= p < 0.30))
print("  TIER 4 (pc 0.00-0.10): LOW / BORDERLINE  (%d axes)" %
      sum(1 for _,p,_ in pos_axes if 0.00 <= p < 0.10))
print("  TIER 5 (pc < 0.00):  CYCLIC / NON-UNIFORM sequences  (%d axes)" %
      len(neg_axes))
