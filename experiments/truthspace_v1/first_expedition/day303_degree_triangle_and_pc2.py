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

print("DAY 303: DEGREE TRIANGLE GEOMETRY, PC2 PROBE, COMPOSITION CATALOG")
print("="*65)
print()

# ====================================================================
# PART A: DEGREE TRIANGLE — PRECISE 2D GEOMETRY
# ====================================================================
print("PART A: Degree triangle — 2D geometry")
print("-"*65)

# Adjective triples (base, comparative, superlative) as single tokens
DEGREE_TRIPLES = [
    ('fast',   'faster',   'fastest'),
    ('slow',   'slower',   'slowest'),
    ('tall',   'taller',   'tallest'),
    ('short',  'shorter',  'shortest'),
    ('bright', 'brighter', 'brightest'),
    ('dark',   'darker',   'darkest'),
    ('clean',  'cleaner',  'cleanest'),
    ('deep',   'deeper',   'deepest'),
    ('light',  'lighter',  'lightest'),
    ('strong', 'stronger', 'strongest'),
    ('weak',   'weaker',   'weakest'),
    ('soft',   'softer',   'softest'),
]

avail = [(b,c,s) for b,c,s in DEGREE_TRIPLES
         if get_emb(b)[0] is not None
         and get_emb(c)[0] is not None
         and get_emb(s)[0] is not None]

print("  Available triples: %d" % len(avail))
print()

# For each triple, measure the triangle
# Use vectors: v_b, v_c, v_s
# Angles at each vertex, side lengths, and whether c is "above" the b-s line

print("  Per-triple analysis:")
print("  %-8s  |b-c|   |c-s|   |b-s|   cos(bc,bs)  cos(cb,cs)  'height'" % "base")
print("  " + "-"*72)

# Also collect chord vectors for axis computation
chords_bc = []  # base -> comparative
chords_cs = []  # comparative -> superlative
chords_bs = []  # base -> superlative

heights = []
for base, comp, sup in avail:
    v_b = get_emb(base)[0]
    v_c = get_emb(comp)[0]
    v_s = get_emb(sup)[0]
    if v_b is None or v_c is None or v_s is None: continue

    bc = v_c - v_b  # base -> comp chord
    cs = v_s - v_c  # comp -> sup chord
    bs = v_s - v_b  # base -> sup chord

    chords_bc.append(normed(bc))
    chords_cs.append(normed(cs))
    chords_bs.append(normed(bs))

    # Side lengths (L2 distances)
    len_bc = float(np.linalg.norm(bc))
    len_cs = float(np.linalg.norm(cs))
    len_bs = float(np.linalg.norm(bs))

    # Angle at base vertex: cos(bc, bs)
    cos_base = float(np.dot(normed(bc), normed(bs)))

    # Angle at comp vertex: cos(cb, cs) = cos(-bc, cs)
    cos_comp = float(np.dot(normed(-bc), normed(cs)))

    # "Height" = component of bc orthogonal to bs (how far comp is off the b-s line)
    bs_n = normed(bs)
    proj_along_bs = float(np.dot(bc, bs_n))
    bc_parallel = proj_along_bs * bs_n
    bc_perp = bc - bc_parallel
    height = float(np.linalg.norm(bc_perp))

    heights.append(height)
    print("  %-8s  %.3f  %.3f  %.3f   %+.4f       %+.4f      %.4f" % (
        base, len_bc, len_cs, len_bs, cos_base, cos_comp, height))

# Summary statistics for the degree triangle
print()
ax_bc = normed(np.mean(chords_bc, axis=0))  # mean base->comp direction
ax_cs = normed(np.mean(chords_cs, axis=0))  # mean comp->sup direction
ax_bs = normed(np.mean(chords_bs, axis=0))  # mean base->sup direction

cos_bc_cs = float(np.dot(ax_bc.astype(np.float32), ax_cs.astype(np.float32)))
cos_bc_bs = float(np.dot(ax_bc.astype(np.float32), ax_bs.astype(np.float32)))
cos_cs_bs = float(np.dot(ax_cs.astype(np.float32), ax_bs.astype(np.float32)))

print("  Mean triangle properties:")
print("  cos(base->comp, comp->sup) = %+.4f  [bc vs cs directions]" % cos_bc_cs)
print("  cos(base->comp, base->sup) = %+.4f  [bc vs bs directions]" % cos_bc_bs)
print("  cos(comp->sup, base->sup)  = %+.4f  [cs vs bs directions]" % cos_cs_bs)
print("  Mean height (off-line dist): %.4f" % np.mean(heights))
print("  Height std:                  %.4f" % np.std(heights))
print()

# Project all base/comp/sup embeddings onto 2D plane spanned by ax_bc, ax_bs
# Use Gram-Schmidt to make ax_bs orthogonal to ax_bc
e1 = ax_bc.astype(np.float64)
e2 = ax_bs.astype(np.float64)
e2_orth = normed(e2 - np.dot(e2, e1) * e1)

print("  2D projections of all base/comp/sup embeddings:")
print("  %-10s  e1(bc)  e2_orth" % "word")
print("  " + "-"*32)
for base, comp, sup in avail:
    for word, label in [(base,'base'),(comp,'comp'),(sup,'sup')]:
        e, _ = get_emb(word)
        if e is None: continue
        p1 = float(np.dot(e, e1))
        p2 = float(np.dot(e, e2_orth))
        print("  %-10s  %+.4f  %+.4f  [%s]" % (word, p1, p2, label))
    print()

# ====================================================================
# PART B: PC2 SYNTACTIC AND SEMANTIC PROBE
# ====================================================================
print("PART B: PC2 syntactic/semantic probe")
print("-"*65)

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

mu_f = mu.astype(np.float64)
v2_f = v2.astype(np.float64)
v1_f = v1.astype(np.float64)

# Probe with syntactic categories
SYNTACTIC_GROUPS = {
    'Nouns-concrete':   ['dog','cat','house','car','tree','book','city','country'],
    'Nouns-abstract':   ['love','hate','truth','beauty','justice','peace','time','life'],
    'Nouns-proper':     ['Paris','London','Rome','Tokyo','France','England','Japan'],
    'Adjectives':       ['fast','slow','big','small','good','bad','hot','cold'],
    'Comparatives':     ['faster','slower','bigger','smaller','better','worse','hotter','colder'],
    'Superlatives':     ['fastest','slowest','biggest','smallest','best','worst','hottest','coldest'],
    'Verbs-base':       ['run','walk','go','come','see','know','make','take'],
    'Verbs-past-reg':   ['walked','talked','jumped','started','ended','looked','helped','called'],
    'Verbs-past-irr':   ['went','came','saw','knew','made','took','ran','said'],
    'Adverbs':          ['quickly','slowly','very','really','quite','already','never','always'],
    'Prepositions':     ['in','on','at','by','for','with','from','about'],
    'Conjunctions':     ['and','but','or','so','yet','nor','for','either'],
    'Pronouns':         ['he','she','it','they','we','you','I','me'],
    'Determiners':      ['the','a','an','this','that','these','those','some'],
    'Numbers-digits':   ['1','2','3','4','5','6','7','8','9'],
    'Numbers-words':    ['one','two','three','four','five','six','seven','eight'],
    'Months':           ['January','February','March','April','May','June','July'],
    'Weekdays':         ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    'Punct/Special':    ['.', ',', '!', '?', ':', ';'],
    'Derivations-ness': ['sadness','happiness','darkness','kindness','brightness','fitness'],
    'Derivations-ment': ['achievement','management','development','argument','movement'],
    'Derivations-un':   ['unhappy','unkind','unfair','unknown','unusual','unclear'],
}

print("  %-22s  n   PC2_mean  PC2_std   PC1_mean" % "group")
print("  " + "-"*55)
group_pc2 = {}
group_pc1 = {}
for grp, words in SYNTACTIC_GROUPS.items():
    pc1s, pc2s = [], []
    for w in words:
        e, _ = get_emb(w)
        if e is None: continue
        ec = (e - mu_f).astype(np.float64)
        pc1s.append(float(np.dot(ec, v1_f)))
        pc2s.append(float(np.dot(ec, v2_f)))
    if pc2s:
        group_pc2[grp] = np.mean(pc2s)
        group_pc1[grp] = np.mean(pc1s)
        print("  %-22s  %-3d %+.4f   %.4f    %+.4f" % (
            grp, len(pc2s), np.mean(pc2s), np.std(pc2s), np.mean(pc1s)))

print()
# Sort by PC2 to show the spectrum
print("  Groups sorted by PC2 (highest to lowest):")
for grp, pc2v in sorted(group_pc2.items(), key=lambda x: -x[1]):
    print("    %-22s  PC2=%+.4f  PC1=%+.4f" % (grp, pc2v, group_pc1[grp]))
print()

# ====================================================================
# PART C: ADDITIONAL COMPOSITION TESTS
# ====================================================================
print("PART C: Additional composition tests")
print("-"*65)

# Test: +s + +ness = ?  (plural of an abstract noun)
# e.g., sad -> sadness -> sadnesses  (does sadness->sadnesses axis match sad->sadnesses?)
# Limited by single-token availability

# Test: gender + +er = ?  (faster sister?)
# gender axis (king->queen, man->woman), +er axis (fast->faster)
# These are orthogonal in different semantic domains -- should NOT compose

gender_pairs = [('king','queen'),('man','woman'),('boy','girl'),
                ('father','mother'),('son','daughter'),('brother','sister')]
er_pairs = [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
            ('bright','brighter'),('dark','darker'),('clean','cleaner'),('deep','deeper')]
est_pairs = [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
             ('bright','brightest'),('dark','darkest'),('clean','cleanest'),('deep','deepest')]
past_irr_pairs = [('go','went'),('come','came'),('run','ran'),
                  ('see','saw'),('eat','ate'),('know','knew'),('take','took')]
past_reg_pairs = [('walk','walked'),('talk','talked'),('jump','jumped'),
                  ('start','started'),('end','ended'),('look','looked'),('call','called')]

# Compute all base axes
ax_gender, _, _, _ = compute_axis(gender_pairs)
ax_er, _, valid_er, _ = compute_axis(er_pairs)
ax_est, _, valid_est, _ = compute_axis(est_pairs)
ax_irr, _, valid_irr, _ = compute_axis(past_irr_pairs)
ax_reg, _, valid_reg, _ = compute_axis(past_reg_pairs)

# Test A: gender axis is orthogonal to +er axis?
if ax_gender is not None and ax_er is not None:
    c = float(np.dot(ax_gender.astype(np.float32), ax_er.astype(np.float32)))
    print("  cos(gender, +er) = %+.4f  [should be ~0 if truly orthogonal]" % c)

# Test B: past_irr vs past_reg -- do they compose?
# past_reg_verb -> past_irr_verb (if such axis exists)
# Or: are they in the same subspace?
if ax_irr is not None and ax_reg is not None:
    c_irr_reg = float(np.dot(ax_irr.astype(np.float32), ax_reg.astype(np.float32)))
    print("  cos(past_irr, past_reg) = %+.4f  [same tense, different morphology]" % c_irr_reg)

# Test C: +er + gender = ?  (composition across independent domains)
if ax_er is not None and ax_gender is not None:
    ax_er_raw = np.mean([normed(get_emb(t)[0]-get_emb(s)[0]) for s,t in er_pairs
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_g_raw  = np.mean([normed(get_emb(t)[0]-get_emb(s)[0]) for s,t in gender_pairs
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_comp_er_gender = normed(ax_er_raw + ax_g_raw)
    # What does fast + gender composition give?
    e_fast, sid_fast = get_emb('fast')
    if e_fast is not None:
        s_er, _ = best_scale(ax_er, valid_er)
        pred = W_E[sid_fast] + s_er * ax_comp_er_gender
        r = nn_retrieve(pred, [sid_fast], top_n=5)
        print("  fast + (er+gender) -> %s  [expected: nothing meaningful]" %
              ', '.join(w for w,_,_ in r[:3]))

print()

# Test D: Composition of morphological chains that ARE connected
# +s + 's (genitive) ?  -- limited by tokenization
# +er + rev(+er) = identity?  (does forward + reverse = stay at source?)
if ax_er is not None:
    ax_er_raw = np.mean([normed(get_emb(t)[0]-get_emb(s)[0]) for s,t in er_pairs
                          if get_emb(s)[0] is not None and get_emb(t)[0] is not None], axis=0)
    ax_er_rev = normed(-ax_er_raw)
    ax_round_trip = normed(ax_er_raw + (-ax_er_raw))  # should be zero vector
    print("  |+er + rev(+er)| = %.6f  [expected: ~0 -- identity]" %
          float(np.linalg.norm(ax_er_raw + (-ax_er_raw))))
    # What does round-trip do to a word?
    e_fast, sid_fast = get_emb('fast')
    if e_fast is not None:
        s_er, _ = best_scale(ax_er, valid_er)
        # Forward
        pred_fwd = W_E[sid_fast] + s_er * ax_er_raw / (np.linalg.norm(ax_er_raw) + 1e-8)
        r_fwd = nn_retrieve(pred_fwd, [sid_fast], top_n=3)
        print("  fast +er -> %s" % ', '.join(w for w,_,_ in r_fwd[:3]))
        # Round trip: fast -> faster -> fast?
        e_faster, sid_faster = get_emb('faster')
        if e_faster is not None:
            pred_rt = W_E[sid_faster] + s_er * (-ax_er_raw / (np.linalg.norm(ax_er_raw)+1e-8))
            r_rt = nn_retrieve(pred_rt, [sid_faster], top_n=3)
            print("  faster -er -> %s  [should recover 'fast']" %
                  ', '.join(w for w,_,_ in r_rt[:3]))
print()

# ====================================================================
# PART D: SUBSPACE L + M VARIANCE TEST
# ====================================================================
print("PART D: Subspace L + M combined variance")
print("-"*65)

# Build v_ord (subspace L representative)
MONTHS   = ['January','February','March','April','May','June','July','August','September']
WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDS    = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ace']
CARD_N   = ['2','3','4','5','6','7','8','9','1']
SEASONS  = ['Spring','Summer','Autumn','Winter']
PLANETS  = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
LETTERS  = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

fwd_axes = []
for pairs in [
    [(MONTHS[i], str(i+1)) for i in range(9)],
    [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    list(zip(CARDS, CARD_N)),
    [(SEASONS[i], str(i+1)) for i in range(4)],
]:
    avail_p = [(s,t) for s,t in pairs
               if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail_p) < 2: continue
    ax, _, _, _ = compute_axis(avail_p)
    if ax is not None: fwd_axes.append(ax)
v_ord = normed(np.mean(fwd_axes, axis=0)).astype(np.float64)

# Build subspace M: mean of all morphological axes
morph_pairs_list = [er_pairs, est_pairs, gender_pairs, past_irr_pairs, past_reg_pairs]
morph_names = ['+er', '+est', 'gender', 'past_irr', '+ed']
morph_axes = []
for pairs, nm in zip(morph_pairs_list, morph_names):
    avail_p = [(s,t) for s,t in pairs
               if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail_p) < 2: continue
    ax, _, _, _ = compute_axis(avail_p)
    if ax is not None: morph_axes.append(ax)

# Gram-Schmidt to make M axes orthonormal
def gram_schmidt(vecs):
    basis = []
    for v in vecs:
        v = v.copy().astype(np.float64)
        for b in basis:
            v -= np.dot(v, b) * b
        n = np.linalg.norm(v)
        if n > 1e-8: basis.append(v / n)
    return basis

M_basis = gram_schmidt(morph_axes)
print("  Subspace M basis size: %d (from %d morphological axes)" %
      (len(M_basis), len(morph_axes)))

# Sample embeddings
N = min(5000, len(W_E))
sample_ids2 = rng.integers(0, len(W_E), size=N)
W_s = W_E[sample_ids2].astype(np.float64)
W_s_c = W_s - W_s.mean(axis=0)

total_var = float(np.mean(np.sum(W_s_c**2, axis=1)))

# Variance along v_ord (subspace L)
proj_L = W_s_c @ v_ord
var_L = float(np.var(proj_L))

# Variance along M basis
var_M_total = 0.0
for b in M_basis:
    proj = W_s_c @ b
    var_M_total += float(np.var(proj))

# Combined variance (L + M)
var_LM = var_L + var_M_total

print("  Total W_E variance (sample): %.6f" % total_var)
print("  Variance in subspace L (v_ord): %.6f  (%.4f%%)" %
      (var_L, 100*var_L/total_var))
print("  Variance in subspace M (%d axes): %.6f  (%.4f%%)" %
      (len(M_basis), var_M_total, 100*var_M_total/total_var))
print("  Combined L+M: %.6f  (%.4f%%)" %
      (var_LM, 100*var_LM/total_var))
print()
print("  For reference:")
print("  PC1 (single direction): ~3.35%")
print("  L+M (%d directions total): %.4f%%" % (1+len(M_basis), 100*var_LM/total_var))
print()

# ====================================================================
# PART E: DEGREE SYSTEM SUMMARY
# ====================================================================
print("="*65)
print("PART E: Degree system geometry summary")
print("="*65)
print()

# Collect mean chord lengths
all_bc_lengths = []
all_cs_lengths = []
all_bs_lengths = []
all_heights = []

for base, comp, sup in avail:
    v_b = get_emb(base)[0]
    v_c = get_emb(comp)[0]
    v_s = get_emb(sup)[0]
    if v_b is None or v_c is None or v_s is None: continue
    bc = v_c - v_b
    cs = v_s - v_c
    bs = v_s - v_b
    all_bc_lengths.append(float(np.linalg.norm(bc)))
    all_cs_lengths.append(float(np.linalg.norm(cs)))
    all_bs_lengths.append(float(np.linalg.norm(bs)))
    bs_n = normed(bs)
    bc_perp = bc - float(np.dot(bc, bs_n)) * bs_n
    all_heights.append(float(np.linalg.norm(bc_perp)))

print("  Mean side lengths:")
print("    base->comp  (|bc|): %.4f ± %.4f" % (np.mean(all_bc_lengths), np.std(all_bc_lengths)))
print("    comp->sup   (|cs|): %.4f ± %.4f" % (np.mean(all_cs_lengths), np.std(all_cs_lengths)))
print("    base->sup   (|bs|): %.4f ± %.4f" % (np.mean(all_bs_lengths), np.std(all_bs_lengths)))
print("    height (comp off-line): %.4f ± %.4f" % (np.mean(all_heights), np.std(all_heights)))
print()
print("  Ratios:")
print("    |bc|/|bs| = %.4f  [comp step as fraction of full range]" %
      (np.mean(all_bc_lengths)/np.mean(all_bs_lengths)))
print("    |cs|/|bs| = %.4f  [sup step as fraction of full range]" %
      (np.mean(all_cs_lengths)/np.mean(all_bs_lengths)))
print("    height/|bc| = %.4f  [relative off-axis displacement]" %
      (np.mean(all_heights)/np.mean(all_bc_lengths)))
print()
print("  Direction cosines:")
print("    cos(bc, cs) = %+.4f  [sequential steps]" % cos_bc_cs)
print("    cos(bc, bs) = %+.4f  [comp vs full-range]" % cos_bc_bs)
print("    cos(cs, bs) = %+.4f  [sup step vs full-range]" % cos_cs_bs)
