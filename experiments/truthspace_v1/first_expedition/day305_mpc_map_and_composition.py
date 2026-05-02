import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

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

print("DAY 305: FULL mPC MAP, CROSS-MORPHOLOGICAL COMPOSITION, mPC TOKEN ATLAS")
print("="*70)
print()

# ====================================================================
# Build comprehensive morphological chord matrix
# ====================================================================
ALL_MORPH = {
    '+er':      [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                 ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                 ('light','lighter'),('strong','stronger'),('weak','weaker'),('soft','softer')],
    '+est':     [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                 ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest'),
                 ('light','lightest'),('strong','strongest'),('weak','weakest')],
    'er->est':  [('faster','fastest'),('slower','slowest'),('taller','tallest'),('shorter','shortest'),
                 ('brighter','brightest'),('darker','darkest'),('deeper','deepest'),('cleaner','cleanest'),
                 ('lighter','lightest'),('stronger','strongest'),('weaker','weakest')],
    'gender':   [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                 ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
    'past_irr': [('go','went'),('come','came'),('run','ran'),('see','saw'),
                 ('eat','ate'),('know','knew'),('take','took'),('make','made')],
    '+ed':      [('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                 ('end','ended'),('look','looked'),('call','called'),('help','helped')],
    '+ness':    [('sad','sadness'),('happy','happiness'),('dark','darkness'),('kind','kindness'),
                 ('bright','brightness'),('fit','fitness'),('mad','madness'),('glad','gladness')],
    '+ful':     [('hope','hopeful'),('care','careful'),('use','useful'),('power','powerful'),
                 ('peace','peaceful'),('harm','harmful'),('thank','thankful'),('help','helpful')],
    'un-':      [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('known','unknown'),
                 ('usual','unusual'),('clear','unclear'),('lock','unlock'),('wrap','unwrap')],
    '+ment':    [('achieve','achievement'),('manage','management'),('develop','development'),
                 ('move','movement'),('treat','treatment'),('argue','argument')],
    '+s':       [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
    '+tion':    [('act','action'),('direct','direction'),('collect','collection'),
                 ('connect','connection'),('protect','protection'),('select','selection')],
}

# Collect all chord vectors
all_chords = []
all_chord_labels = []  # (source, target, axis_name)
for nm, pairs in ALL_MORPH.items():
    for s, t in pairs:
        es, sid = get_emb(s)
        et, tid = get_emb(t)
        if es is None or et is None: continue
        all_chords.append(normed(et-es).astype(np.float32))
        all_chord_labels.append((s, t, nm))

M_mat = np.array(all_chords)
print("  Total morphological chord vectors: %d" % len(M_mat))
print("  Axis breakdown:")
for nm in ALL_MORPH:
    n = sum(1 for _,_,a in all_chord_labels if a==nm)
    print("    %-12s  %d pairs" % (nm, n))
print()

# PCA on morphological subspace
M_mean = M_mat.mean(axis=0)
M_c = (M_mat - M_mean).astype(np.float32)
rng = np.random.default_rng(0)
M_pcs = []
M_deflated = M_c.copy()
for k in range(8):
    vk = rng.standard_normal(M_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(200):
        vk = M_deflated.T @ (M_deflated @ vk)
        vk /= np.linalg.norm(vk)
    proj = M_deflated @ vk
    M_deflated = M_deflated - np.outer(proj, vk)
    var_k = float(np.var(M_c @ vk))
    M_pcs.append((vk.astype(np.float64), var_k))
M_tot_var = float(np.sum(np.var(M_c, axis=0)))

# ====================================================================
# PART A: FULL mPC1-5 ALIGNMENT MAP
# ====================================================================
print("PART A: Full mPC1-5 alignment with all morphological axes")
print("-"*70)

# First compute each named axis
named_axes = {}
named_pcs = {}
for nm, pairs in ALL_MORPH.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, pc = compute_axis(avail)
    if ax is None: continue
    named_axes[nm] = ax.astype(np.float64)
    named_pcs[nm] = pc

print("  %-14s" % "axis" + "".join("   mPC%d " % (k+1) for k in range(5)) + "   pc")
print("  " + "-"*72)
for nm in ALL_MORPH:
    if nm not in named_axes: continue
    comps = [float(np.dot(named_axes[nm], M_pcs[k][0])) for k in range(5)]
    print("  %-14s" % nm + "".join("  %+.4f" % c for c in comps) + "  %+.4f" % named_pcs[nm])
print()

# ====================================================================
# PART B: NEAREST TOKENS TO mPCs (TOKEN ATLAS)
# ====================================================================
print("PART B: Nearest tokens to each mPC direction (atlas)")
print("-"*70)

# For mPCs, compute the projection of ALL tokens onto the axis
# and find those closest to the +/- ends
for k in range(5):
    mpc = M_pcs[k][0].astype(np.float32)
    projs = W_n @ mpc   # cosine similarity (W_n is already normalised)
    top_pos = np.argsort(projs)[-20:][::-1]
    top_neg = np.argsort(projs)[:20]

    print("  mPC%d  top 10 tokens (most POSITIVE — most similar to mPC%d direction):" % (k+1, k+1))
    for tid in top_pos[:10]:
        w = tok.decode([tid]).strip()
        print("    cos=%+.4f  id=%-6d  '%s'" % (float(projs[tid]), tid, w))
    print("  mPC%d  bottom 10 tokens (most NEGATIVE — most opposite):" % (k+1))
    for tid in top_neg[:10]:
        w = tok.decode([tid]).strip()
        print("    cos=%+.4f  id=%-6d  '%s'" % (float(projs[tid]), tid, w))
    print()

# ====================================================================
# PART C: CROSS-DOMAIN COMPOSITION TEST
# ====================================================================
print("PART C: Cross-domain composition tests")
print("-"*70)

# Test 1: gender + past_irr — should be meaningless (different subspaces)
ax_gender  = named_axes.get('gender')
ax_past    = named_axes.get('past_irr')
ax_er      = named_axes.get('+er')
ax_s       = named_axes.get('+s')
ax_ness    = named_axes.get('+ness')
ax_ed      = named_axes.get('+ed')
ax_un      = named_axes.get('un-')

# Compute raw (unnormalised means) for composition
def raw_axis(pairs):
    chords = []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(normed(et-es).astype(np.float64))
    return np.mean(chords, axis=0) if chords else None

raw_gender  = raw_axis(ALL_MORPH['gender'])
raw_past    = raw_axis(ALL_MORPH['past_irr'])
raw_er      = raw_axis(ALL_MORPH['+er'])
raw_s       = raw_axis(ALL_MORPH['+s'])
raw_ness    = raw_axis(ALL_MORPH['+ness'])
raw_ed      = raw_axis(ALL_MORPH['+ed'])
raw_un      = raw_axis(ALL_MORPH['un-'])
raw_ersup   = raw_axis(ALL_MORPH['er->est'])
raw_est     = raw_axis(ALL_MORPH['+est'])

_, _, valid_gender, _ = compute_axis(ALL_MORPH['gender'])
_, _, valid_past,   _ = compute_axis(ALL_MORPH['past_irr'])
_, _, valid_er_v,   _ = compute_axis(ALL_MORPH['+er'])
_, _, valid_s_v,    _ = compute_axis(ALL_MORPH['+s'])
_, _, valid_ness_v, _ = compute_axis(ALL_MORPH['+ness'])

print("  Test A: gender + past_irr (should produce no linguistic meaning)")
if raw_gender is not None and raw_past is not None:
    ax_comp = normed(raw_gender + raw_past)
    c_g = float(np.dot(ax_comp, ax_gender))
    c_p = float(np.dot(ax_comp, ax_past))
    print("    cos(composed, gender) = %+.4f" % c_g)
    print("    cos(composed, past)   = %+.4f" % c_p)
    # Apply to a test word: king
    e_king, sid_king = get_emb('king')
    if e_king is not None:
        s_g, _ = best_scale(ax_gender, valid_gender, lo=0.5, hi=4.0, n=50)
        s_p, _ = best_scale(ax_past,   valid_past,   lo=0.5, hi=4.0, n=50)
        pred = W_E[sid_king] + s_g * raw_gender + s_p * raw_past
        r = nn_retrieve(pred, [sid_king], top_n=5)
        print("    king + gender + past -> %s  [top5: %s]" %
              (r[0][0], ', '.join(w for w,_,_ in r[:5])))
    print()

print("  Test B: +s + gender — pluralise then gender-flip")
if raw_s is not None and raw_gender is not None:
    # brothers -> sisters? (brothers = brother+s, then gender flip)
    ax_comp_sg = normed(raw_s + raw_gender)
    e_brothers, sid = get_emb('brothers')
    if e_brothers is not None:
        s_s, _ = best_scale(ax_s, valid_s_v, lo=0.5, hi=4.0, n=50)
        s_g, _ = best_scale(ax_gender, valid_gender, lo=0.5, hi=4.0, n=50)
        # Apply s then gender
        pred = W_E[sid] + s_g * raw_gender  # brothers -> sisters?
        r = nn_retrieve(pred, [sid], top_n=5)
        print("    brothers + gender -> %s  [%s]" % (r[0][0], ', '.join(w for w,_,_ in r[:5])))
    # son -> daughters? (son + gender + plural)
    e_son, sid_son = get_emb('son')
    if e_son is not None:
        pred = W_E[sid_son] + s_s * raw_s + s_g * raw_gender
        r = nn_retrieve(pred, [sid_son], top_n=5)
        print("    son + gender + plural -> %s  [%s]" % (r[0][0], ', '.join(w for w,_,_ in r[:5])))
    print()

print("  Test C: un- + +er — uncomparative? (e.g. happy -> unhappier)")
if raw_un is not None and raw_er is not None:
    ax_un_er = normed(raw_un + raw_er)
    e_happy, sid_happy = get_emb('happy')
    if e_happy is not None:
        _, _, valid_un_v, _ = compute_axis([('happy','unhappy'),('kind','unkind'),('fair','unfair')])
        s_u, _ = best_scale(ax_un, valid_un_v, lo=0.5, hi=4.0, n=50)
        s_er, _ = best_scale(ax_er, valid_er_v, lo=0.5, hi=4.0, n=50)
        pred = W_E[sid_happy] + s_u * raw_un + s_er * raw_er
        r = nn_retrieve(pred, [sid_happy], top_n=5)
        print("    happy + un- + +er -> %s  [%s]" % (r[0][0], ', '.join(w for w,_,_ in r[:5])))
        # Check if 'unhappier' is a single token
        e_uh, _ = get_emb('unhappier')
        if e_uh is not None:
            print("    'unhappier' is a single token -- target accessible")
        else:
            print("    'unhappier' is NOT a single token -- can't verify")
    print()

print("  Test D: +ness + un- — unsadness? (reverse derivation)")
if raw_ness is not None and raw_un is not None:
    ax_ness_un = normed(raw_ness + raw_un)
    # Does this compose to anything useful?
    c_u = float(np.dot(ax_ness_un, ax_un))
    c_n = float(np.dot(ax_ness_un, ax_ness))
    print("    cos(ness+un, +ness)  = %+.4f" % c_n)
    print("    cos(ness+un, un-)    = %+.4f" % c_u)
    e_sad, sid_sad = get_emb('sad')
    if e_sad is not None:
        _, _, valid_un_v2, _ = compute_axis([('happy','unhappy'),('kind','unkind')])
        s_u, _ = best_scale(ax_un, valid_un_v2, lo=0.5, hi=4.0, n=50)
        s_n, _ = best_scale(ax_ness, valid_ness_v, lo=0.5, hi=4.0, n=50)
        pred = W_E[sid_sad] + s_n * raw_ness + s_u * raw_un
        r = nn_retrieve(pred, [sid_sad], top_n=5)
        print("    sad + +ness + un- -> %s  [%s]" % (r[0][0], ', '.join(w for w,_,_ in r[:5])))
    print()

# ====================================================================
# PART D: v_ord 39% VERIFICATION
# ====================================================================
print("PART D: v_ord 39% verification — is it signal or noise?")
print("-"*70)

# Build v_ord from labelling axes
MONTHS   = ['January','February','March','April','May','June','July','August','September']
WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDS    = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ace']
CARD_N   = ['2','3','4','5','6','7','8','9','1']

fwd_axes = []
for pairs in [
    [(MONTHS[i], str(i+1)) for i in range(9)],
    [(WEEKDAYS[i], str(i+1)) for i in range(7)],
    list(zip(CARDS, CARD_N)),
]:
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, _ = compute_axis(avail)
    if ax is not None: fwd_axes.append(ax)
v_ord = normed(np.mean(fwd_axes, axis=0)).astype(np.float64)

# Project v_ord onto top 20 global PCs (extend previous analysis)
rng2 = np.random.default_rng(42)
N_SAMPLE = 8000
sample_ids = rng2.integers(0, len(W_E), size=N_SAMPLE)
W_sample = W_E[sample_ids].astype(np.float32)
mu = W_sample.mean(axis=0)
W_c = W_sample - mu

global_pcs = []
W_defl = W_c.copy()
for k in range(20):
    vk = rng2.standard_normal(W_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(100):
        vk = W_defl.T @ (W_defl @ vk)
        vk /= np.linalg.norm(vk)
    proj = W_defl @ vk
    W_defl = W_defl - np.outer(proj, vk)
    lam = float(np.var(W_c @ vk))
    global_pcs.append((vk.astype(np.float64), lam))

total_glob = float(np.sum(np.var(W_c, axis=0)))

print("  v_ord decomposition in PC1-PC20 basis:")
r2_running = 0.0
for k, (pcv, lam) in enumerate(global_pcs):
    c = float(np.dot(v_ord, pcv))
    r2_running += c**2
    var_pct = lam / (len(W_c[0]) * total_glob / len(W_c[0])) * 100
    if abs(c) > 0.02 or k < 10:
        print("    PC%-2d: cos=%+.6f  r\u00b2_contrib=%.4f%%  cumR\u00b2=%.4f  (var=%.4f%%)" %
              (k+1, c, 100*c**2, r2_running, lam/total_glob*100*len(W_c[0])))

print()
print("  Cumulative R\u00b2 in PC1-20: %.4f" % r2_running)

# Test whether the residual v_ord direction (after PC1-6 removal) is consistent
# Build residual v_ord
v_ord_resid = v_ord.copy()
for k in range(6):
    pcv = global_pcs[k][0]
    v_ord_resid -= np.dot(v_ord_resid, pcv) * pcv
v_ord_resid_n = normed(v_ord_resid)

# Check if v_ord_resid is consistent across different labelling sets
print("  Residual v_ord (after PC1-6 removal) alignment with labelling axes:")
for pairs_name, pairs in [
    ('month->num', [(MONTHS[i], str(i+1)) for i in range(9)]),
    ('weekday->num', [(WEEKDAYS[i], str(i+1)) for i in range(7)]),
    ('card->num', list(zip(CARDS, CARD_N))),
]:
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    ax, _, _, _ = compute_axis(avail)
    if ax is None: continue
    # Compute residual of each axis
    ax_resid = ax.copy()
    for k in range(6):
        pcv = global_pcs[k][0]
        ax_resid -= np.dot(ax_resid, pcv) * pcv
    ax_resid_n = normed(ax_resid)
    c = float(np.dot(v_ord_resid_n, ax_resid_n))
    print("    %-16s  cos(residuals) = %+.4f" % (pairs_name, c))

print()
# If residuals are consistent (high cosine between each other), the 39% is SIGNAL
# If not, it's noise

# ====================================================================
# PART E: mPC PROJECTION OF ALL MORPH CHORD VECTORS
# ====================================================================
print("PART E: Per-pair mPC1-3 projections (visualize the morphological landscape)")
print("-"*70)

mpc1 = M_pcs[0][0]; mpc2 = M_pcs[1][0]; mpc3 = M_pcs[2][0]

print("  %-12s  %-10s  %-10s  mPC1     mPC2     mPC3    axis" %
      ("source", "target", ""))
print("  " + "-"*68)
for (s, t, nm), chord in zip(all_chord_labels, all_chords):
    chord_f = chord.astype(np.float64)
    p1 = float(np.dot(chord_f, mpc1))
    p2 = float(np.dot(chord_f, mpc2))
    p3 = float(np.dot(chord_f, mpc3))
    print("  %-12s  %-10s         %+.4f   %+.4f   %+.4f  %s" %
          (s, t, p1, p2, p3, nm))
print()

# Group means on mPC1-3
print("  Group means on mPC1-3:")
print("  %-14s  mPC1_mean  mPC2_mean  mPC3_mean  pc_axis" % "axis")
print("  " + "-"*60)
for nm in ALL_MORPH:
    if nm not in named_axes: continue
    chords_nm = [all_chords[i].astype(np.float64)
                 for i, (_,_,a) in enumerate(all_chord_labels) if a==nm]
    if not chords_nm: continue
    p1s = [float(np.dot(c, mpc1)) for c in chords_nm]
    p2s = [float(np.dot(c, mpc2)) for c in chords_nm]
    p3s = [float(np.dot(c, mpc3)) for c in chords_nm]
    print("  %-14s  %+.4f     %+.4f     %+.4f     %.4f" %
          (nm, np.mean(p1s), np.mean(p2s), np.mean(p3s), named_pcs[nm]))
