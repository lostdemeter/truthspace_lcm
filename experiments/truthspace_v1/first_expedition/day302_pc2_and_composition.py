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

print("DAY 302: PC2 EXPLORATION AND MORPHOLOGICAL COMPOSITION")
print("="*65)
print()

rng = np.random.default_rng(42)
N_SAMPLE = 8000
sample_ids = rng.integers(0, len(W_E), size=N_SAMPLE)
W_sample = W_E[sample_ids].astype(np.float32)
mu = W_sample.mean(axis=0)
W_c = W_sample - mu

# Power iteration for PC1 and PC2
v1 = rng.standard_normal(W_c.shape[1]).astype(np.float32)
v1 /= np.linalg.norm(v1)
for _ in range(200):
    v1 = W_c.T @ (W_c @ v1); v1 /= np.linalg.norm(v1)
proj1 = W_c @ v1
W_c2 = W_c - np.outer(proj1, v1)
v2 = rng.standard_normal(W_c.shape[1]).astype(np.float32)
v2 /= np.linalg.norm(v2)
for _ in range(200):
    v2 = W_c2.T @ (W_c2 @ v2); v2 /= np.linalg.norm(v2)
proj2 = W_c2 @ v2
W_c3 = W_c2 - np.outer(proj2, v2)
v3 = rng.standard_normal(W_c.shape[1]).astype(np.float32)
v3 /= np.linalg.norm(v3)
for _ in range(200):
    v3 = W_c3.T @ (W_c3 @ v3); v3 /= np.linalg.norm(v3)

lam1 = float(np.var(W_c  @ v1)) * W_c.shape[1]
lam2 = float(np.var(W_c2 @ v2)) * W_c.shape[1]
lam3 = float(np.var(W_c3 @ v3)) * W_c.shape[1]
tot  = float(np.sum(np.var(W_c, axis=0)))

mu_f  = mu.astype(np.float64)
v1_f  = v1.astype(np.float64)
v2_f  = v2.astype(np.float64)
v3_f  = v3.astype(np.float64)

# ====================================================================
# PART A: PC2 AND PC3 EXPLORATION
# ====================================================================
print("PART A: PC1, PC2, PC3 of W_E")
print("-"*65)
print("  PC1: %.4f%%  PC2: %.4f%%  PC3: %.4f%%" % (
    100*lam1/tot, 100*lam2/tot, 100*lam3/tot))
print()

test_groups = [
    ('Digits',    ['1','2','3','4','5','6','7','8','9']),
    ('CardWords', ['one','two','three','four','five','six','seven','eight','nine']),
    ('Months',    ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']),
    ('Weekdays',  ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']),
    ('Function',  ['the','and','is','of','to','a','in','that','it','for']),
    ('Punct',     ['.', ',', '!', '?', ':', '"']),
    ('Adj',       ['big','small','good','bad','hot','cold','fast','slow',
                   'bright','dark','happy','sad']),
    ('Verb',      ['run','walk','go','come','see','know','make','take',
                   'went','walked','ran','came']),
    ('Adverbs',   ['quickly','slowly','very','really','quite','already',
                   'never','always','often','sometimes']),
    ('Countries', ['France','Germany','Italy','Japan','China','Brazil',
                   'India','Russia','Spain','Canada']),
]

print("  %-14s  %-14s  PC1      PC2      PC3" % ("group", "word"))
print("  " + "-"*60)
for group, words in test_groups:
    grp_p1, grp_p2, grp_p3 = [], [], []
    for w in words:
        e, sid = get_emb(w)
        if e is None: continue
        ec = (e - mu_f).astype(np.float64)
        p1 = float(np.dot(ec, v1_f))
        p2 = float(np.dot(ec, v2_f))
        p3 = float(np.dot(ec, v3_f))
        grp_p1.append(p1); grp_p2.append(p2); grp_p3.append(p3)
    if grp_p1:
        print("  %-14s  %-14s  %+.4f  %+.4f  %+.4f  [GROUP MEAN]" % (
            group, '', np.mean(grp_p1), np.mean(grp_p2), np.mean(grp_p3)))
        for i, w in enumerate(words):
            e, sid = get_emb(w)
            if e is None: continue
            ec = (e - mu_f).astype(np.float64)
            print("  %-14s  %-14s  %+.4f  %+.4f  %+.4f" % (
                '', w, float(np.dot(ec,v1_f)), float(np.dot(ec,v2_f)), float(np.dot(ec,v3_f))))
        print()

# Correlation tests for PC2
test_ids_seq = list(range(0, min(4000, len(W_E)), 2))
pc2_vals = []
for tid in test_ids_seq:
    ec = (W_E[tid] - mu_f).astype(np.float64)
    pc2_vals.append(float(np.dot(ec, v2_f)))
r2_id, p2_id = pearsonr(test_ids_seq, pc2_vals)
lengths2 = [len(tok.decode([i]).strip()) for i in test_ids_seq]
r2_len, p2_len = pearsonr(lengths2, pc2_vals)
print("  PC2 interpretation tests:")
print("  r(token_ID, PC2) = %.4f  p=%.4e" % (r2_id, p2_id))
print("  r(word_length, PC2) = %.4f  p=%.4e" % (r2_len, p2_len))
print()

# ====================================================================
# PART B: v_ord ALIGNMENT WITH ALL KNOWN AXES
# ====================================================================
print("PART B: v_ord alignment with all semantic/morphological axes")
print("-"*65)

MONTHS   = ['January','February','March','April','May','June',
            'July','August','September','October','November','December']
WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDS    = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ace']
CARD_N   = ['2','3','4','5','6','7','8','9','1']
SEASONS  = ['Spring','Summer','Autumn','Winter']
PLANETS  = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
LETTERS  = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
CARDINALS= ['one','two','three','four','five','six','seven','eight','nine']

# Build v_ord
fwd_pairs_map = {
    'month->num':   [(MONTHS[i], str(i+1)) for i in range(9)],
    'weekday->num': [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    'card->num':    list(zip(CARDS, CARD_N)),
    'season->qtr':  [(SEASONS[i], str(i+1)) for i in range(4)],
    'planet->orb':  [(PLANETS[i], str(i+1)) for i in range(8)],
    'letter->pos':  [(LETTERS[i], str(i+1)) for i in range(26)
                     if get_emb(LETTERS[i])[0] is not None
                     and get_emb(str(i+1))[0] is not None],
}
fwd_axes = []
for nm, pairs in fwd_pairs_map.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    ax, _, _, _ = compute_axis(avail)
    if ax is not None: fwd_axes.append(ax)
v_ord = normed(np.mean(fwd_axes, axis=0)).astype(np.float32)

# All known axes
ALL_PAIRS = {
    'digit->word':      [(str(i+1), CARDINALS[i]) for i in range(9)],
    'month->num':       [(MONTHS[i], str(i+1)) for i in range(9)],
    'weekday->num':     [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    'card->num':        list(zip(CARDS, CARD_N)),
    'ordinal->card':    [('first','one'),('second','two'),('third','three'),
                         ('fourth','four'),('fifth','five'),('sixth','six'),
                         ('seventh','seven'),('eighth','eight'),('ninth','nine'),('tenth','ten')],
    'country->demonym': [('France','French'),('Germany','German'),('Italy','Italian'),
                         ('Spain','Spanish'),('Japan','Japanese'),('China','Chinese'),
                         ('Russia','Russian'),('Greece','Greek'),('Brazil','Brazilian'),
                         ('India','Indian')],
    'country->capital': [('France','Paris'),('Germany','Berlin'),('Italy','Rome'),
                         ('Spain','Madrid'),('Japan','Tokyo'),('China','Beijing'),
                         ('Russia','Moscow'),('Greece','Athens'),('Brazil','Brasilia')],
    '+est':             [('fast','fastest'),('slow','slowest'),('tall','tallest'),
                         ('short','shortest'),('bright','brightest'),('dark','darkest'),
                         ('clean','cleanest'),('deep','deepest'),('light','lightest'),
                         ('strong','strongest')],
    '+er':              [('fast','faster'),('slow','slower'),('tall','taller'),
                         ('short','shorter'),('bright','brighter'),('dark','darker'),
                         ('clean','cleaner'),('deep','deeper'),('light','lighter'),
                         ('strong','stronger')],
    'gender':           [('king','queen'),('man','woman'),('boy','girl'),
                         ('father','mother'),('son','daughter'),('brother','sister'),
                         ('husband','wife'),('uncle','aunt')],
    'past_irr':         [('go','went'),('come','came'),('run','ran'),
                         ('see','saw'),('eat','ate'),('know','knew'),
                         ('take','took'),('make','made')],
    '+ed':              [('walk','walked'),('talk','talked'),('jump','jumped'),
                         ('start','started'),('end','ended'),('look','looked'),
                         ('call','called'),('help','helped')],
    '+s plural':        [('cat','cats'),('dog','dogs'),('house','houses'),
                         ('car','cars'),('tree','trees'),('book','books'),
                         ('bird','birds'),('fish','fishes')],
    '+ness':            [('sad','sadness'),('happy','happiness'),('dark','darkness'),
                         ('kind','kindness'),('bright','brightness'),('fit','fitness')],
    'un-':              [('happy','unhappy'),('kind','unkind'),('fair','unfair'),
                         ('known','unknown'),('usual','unusual'),('clear','unclear')],
    '+ment':            [('achieve','achievement'),('manage','management'),
                         ('develop','development'),('argue','argument'),
                         ('move','movement'),('treat','treatment')],
    '+ful':             [('hope','hopeful'),('care','careful'),('use','useful'),
                         ('power','powerful'),('peace','peaceful'),('harm','harmful')],
}

print("  %-22s  cos(axis, v_ord)  pc" % "Axis")
print("  " + "-"*50)
for nm, pairs in ALL_PAIRS.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, pc = compute_axis(avail)
    if ax is None: continue
    c = float(np.dot(ax.astype(np.float32), v_ord))
    print("  %-22s  %+.4f           %+.4f" % (nm, c, pc))
print()

# ====================================================================
# PART C: MORPHOLOGICAL COMPOSITION TESTS
# ====================================================================
print("PART C: Morphological chain composition")
print("-"*65)

# Test: un- + +ness ?= un-ness axis (happy->unhappiness)
un_pairs   = [('happy','unhappy'),('kind','unkind'),('fair','unfair'),
              ('known','unknown'),('usual','unusual'),('clear','unclear')]
ness_pairs = [('sad','sadness'),('happy','happiness'),('dark','darkness'),
              ('kind','kindness'),('bright','brightness'),('fit','fitness')]
unness_pairs = [(s, 'un'+s+'ness') if get_emb('un'+s+'ness')[0] is not None
                else (s, None) for s in ['happy','kind','fair']
                if get_emb(s)[0] is not None]
unness_pairs = [(s,t) for s,t in [('happy','unhappiness'),('kind','unkindness'),
                                    ('fair','unfairness'),('known','unknownness'),
                                    ('usual','unusualness'),('clear','unclearness')]
                if get_emb(s)[0] is not None and get_emb(t)[0] is not None]

ax_un,   _, valid_un,   pc_un   = compute_axis(un_pairs)
ax_ness, _, valid_ness, pc_ness = compute_axis(ness_pairs)
ax_unness, _, valid_unness, pc_unness = compute_axis(unness_pairs) if unness_pairs else (None, 0, [], 0)

print("  Test A: un- + +ness -> un-ness (un-happy-ness)")
print("  ax_un-:   pc=%.4f  n=%d" % (pc_un, len(valid_un)))
print("  ax_+ness: pc=%.4f  n=%d" % (pc_ness, len(valid_ness)))
if unness_pairs:
    print("  ax_un+ness direct: pc=%.4f  n=%d" % (pc_unness, len(valid_unness)))
else:
    print("  ax_un+ness direct: no single-token un-X-ness words available")

if ax_un is not None and ax_ness is not None:
    ax_un_raw   = np.mean([normed(get_emb(t)[0]-get_emb(s)[0])
                            for s,t in un_pairs
                            if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_ness_raw = np.mean([normed(get_emb(t)[0]-get_emb(s)[0])
                            for s,t in ness_pairs
                            if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_comp_un_ness = normed(ax_un_raw + ax_ness_raw)

    cos_un_ness = float(np.dot(ax_un.astype(np.float32), ax_ness.astype(np.float32)))
    print("  cos(un-, +ness) = %.4f  [alignment of component axes]" % cos_un_ness)

    if ax_unness is not None and len(valid_unness) >= 2:
        cos_comp = float(np.dot(ax_comp_un_ness.astype(np.float32), ax_unness.astype(np.float32)))
        print("  cos(composed, direct) = %.4f" % cos_comp)

    # Test composition on known words
    print()
    print("  Testing composition: happy + un- + ness =?= unhappiness")
    test_words_base = [('happy', 'unhappiness'), ('kind', 'unkindness'),
                       ('fair', 'unfairness'), ('clear', 'unclearness')]
    for base, target in test_words_base:
        e_b, sid_b = get_emb(base)
        e_t, tid_t = get_emb(target)
        if e_b is None: continue
        scale_un, _ = best_scale(ax_un, valid_un, lo=0.1, hi=5.0, n=50)
        scale_ness, _ = best_scale(ax_ness, valid_ness, lo=0.1, hi=5.0, n=50)
        pred = W_E[sid_b] + scale_un * ax_un + scale_ness * ax_ness
        r = nn_retrieve(pred, [sid_b], top_n=5)
        got = r[0][0]
        hit_exact = (e_t is not None and got == target)
        print("    %-8s -> %-16s  got=%-16s  top5: %s" % (
            base, target if e_t is not None else target+'(NA)',
            got, ', '.join(w for w,_,_ in r[:3])))
print()

# Test B: base->comparative + comparative->superlative chain
er_pairs  = [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
             ('bright','brighter'),('dark','darker'),('clean','cleaner'),('deep','deeper')]
ersup_pairs = [('faster','fastest'),('slower','slowest'),('taller','tallest'),
               ('shorter','shortest'),('brighter','brightest'),('darker','darkest'),
               ('cleaner','cleanest'),('deeper','deepest')]
est_pairs = [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
             ('bright','brightest'),('dark','darkest'),('clean','cleanest'),('deep','deepest')]

ax_er,    _, valid_er,    pc_er    = compute_axis(er_pairs)
ax_ersup, _, valid_ersup, pc_ersup = compute_axis(ersup_pairs)
ax_est,   _, valid_est,   pc_est   = compute_axis(est_pairs)

if ax_er is not None and ax_ersup is not None and ax_est is not None:
    ax_er_raw    = np.mean([normed(get_emb(t)[0]-get_emb(s)[0]) for s,t in er_pairs
                             if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_ersup_raw = np.mean([normed(get_emb(t)[0]-get_emb(s)[0]) for s,t in ersup_pairs
                             if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_comp_est  = normed(ax_er_raw + ax_ersup_raw)
    cos_chain = float(np.dot(ax_comp_est.astype(np.float32), ax_est.astype(np.float32)))
    cos_er_ersup = float(np.dot(ax_er.astype(np.float32), ax_ersup.astype(np.float32)))

    print("  Test B: +er + er->est = +est  (two-step morphological chain)")
    print("  ax_+er:          pc=%.4f  n=%d" % (pc_er, len(valid_er)))
    print("  ax_er->est:      pc=%.4f  n=%d" % (pc_ersup, len(valid_ersup)))
    print("  ax_+est (direct): pc=%.4f  n=%d" % (pc_est, len(valid_est)))
    print("  cos(+er, er->est) = %.4f  [component alignment]" % cos_er_ersup)
    print("  cos(+er + er->est, +est direct) = %.4f" % cos_chain)
    s_chain, acc_chain = best_scale(ax_comp_est, valid_est)
    print("  Composed axis retrieval: %d/%d (%.0f%%)  scale=%.2f" % (
        acc_chain, len(valid_est), 100*acc_chain/max(1,len(valid_est)), s_chain))
    print()
    for s, t, sid, _ in valid_est[:5]:
        r = nn_retrieve(W_E[sid]+s_chain*ax_comp_est, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-10s -> %-12s  got=%-12s [%s]" % (s, t, r[0][0], 'HIT' if hit else '---'))
print()

# ====================================================================
# PART D: PAIRWISE AXIS COSINES (ALL KNOWN AXES)
# ====================================================================
print("PART D: Pairwise axis cosines — which axes are most aligned?")
print("-"*65)

axis_names = list(ALL_PAIRS.keys())
axis_vectors = {}
axis_pcs = {}
for nm, pairs in ALL_PAIRS.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, pc = compute_axis(avail)
    if ax is None: continue
    axis_vectors[nm] = ax.astype(np.float32)
    axis_pcs[nm] = pc

names = sorted(axis_vectors.keys())
n = len(names)
cos_mat = np.zeros((n, n))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        cos_mat[i, j] = float(np.dot(axis_vectors[ni], axis_vectors[nj]))

print("  Top 20 most-aligned axis pairs (|cos| > 0.3, excluding same-axis):")
pairs_ranked = []
for i in range(n):
    for j in range(i+1, n):
        pairs_ranked.append((abs(cos_mat[i,j]), cos_mat[i,j], names[i], names[j]))
pairs_ranked.sort(reverse=True)
for absc, c, ni, nj in pairs_ranked[:20]:
    print("  %-22s <-> %-22s  cos=%+.4f" % (ni, nj, c))
print()
print("  Top 10 most ANTI-aligned pairs (cos most negative):")
pairs_neg = sorted(pairs_ranked, key=lambda x: x[1])
for absc, c, ni, nj in pairs_neg[:10]:
    print("  %-22s <-> %-22s  cos=%+.4f" % (ni, nj, c))
print()

# ====================================================================
# PART E: PC2 vs v_ord alignment
# ====================================================================
print("PART E: PC1/PC2/PC3 alignment with known axes")
print("-"*65)

for pc_name, pc_vec in [('PC1', v1_f), ('PC2', v2_f), ('PC3', v3_f)]:
    pc_n = normed(pc_vec).astype(np.float32)
    print("  %s alignments:" % pc_name)
    pc_aligns = []
    for nm, ax in axis_vectors.items():
        c = float(np.dot(ax, pc_n))
        pc_aligns.append((abs(c), c, nm))
    pc_aligns.sort(reverse=True)
    for _, c, nm in pc_aligns[:8]:
        print("    %-22s  cos=%+.4f" % (nm, c))
    # v_ord alignment
    c_vord = float(np.dot(v_ord, pc_n))
    print("    %-22s  cos=%+.4f  (v_ord)" % ('v_ord', c_vord))
    print()
