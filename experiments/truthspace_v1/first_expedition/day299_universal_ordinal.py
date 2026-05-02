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
def axis_summary(label, pairs):
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2:
        print("  %-24s  SKIP (only %d avail)" % (label, len(avail)))
        return None, None, avail, None
    ax, coh, valid, pc = compute_axis(avail)
    if ax is None:
        print("  %-24s  SKIP" % label)
        return None, None, avail, None
    s_opt, acc = best_scale(ax, valid)
    print("  %-24s  n=%-2d  pc=%+.4f  coh=%.4f  scale=%.2f  %d/%d (%.0f%%)" % (
        label, len(valid), pc, coh, s_opt, acc, len(valid), 100*acc/max(1,len(valid))))
    return ax, s_opt, avail, pc

print("DAY 299: UNIVERSAL ORDINAL DIRECTION")
print("="*65)
print("Testing whether labelling axis (name->number) is universal")
print("across multiple ordinal categories.")
print()

# ====================================================================
# PART A: LETTER -> ALPHABETICAL POSITION
# ====================================================================
print("PART A: Letter -> alphabetical position (A=1, B=2, ...)")
print("-"*65)

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
LETTER_PAIRS = [(LETTERS[i], str(i+1)) for i in range(len(LETTERS))]

# Check token availability
avail_letters = [(l, n, get_emb(l)[1] is not None, get_emb(n)[1] is not None)
                 for l, n in LETTER_PAIRS]
ok_pairs = [(l, n) for l, n, al, an in avail_letters if al and an]
print("  Single-token letter+number pairs: %d/26" % len(ok_pairs))
print("  Available: %s" % ', '.join('%s=%s' % (l,n) for l,n in ok_pairs))
print()

ax_la, s_la, _, pc_la = axis_summary('letter->alpha-pos', ok_pairs)
if ax_la is not None:
    print()
    print("  Per-pair results:")
    for s, t in ok_pairs:
        es, sid = get_emb(s)
        r = nn_retrieve(W_E[sid]+s_la*ax_la, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-4s -> %-4s  got=%-4s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# lowercase letters
lc_letters = list('abcdefghijklmnopqrstuvwxyz')
lc_pairs = [(lc_letters[i], str(i+1)) for i in range(len(lc_letters))
            if get_emb(lc_letters[i])[0] is not None and get_emb(str(i+1))[0] is not None]
print("  Single-token lowercase+number pairs: %d/26" % len(lc_pairs))
ax_lc, s_lc, _, pc_lc = axis_summary('lc-letter->alpha-pos', lc_pairs)
print()

# ====================================================================
# PART B: PLANETS -> ORBITAL ORDER
# ====================================================================
print("PART B: Planets -> orbital order (Mercury=1, ..., Neptune=8)")
print("-"*65)

PLANETS = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
PLANET_PAIRS = [(PLANETS[i], str(i+1)) for i in range(len(PLANETS))]

pl_avail = [(s,t) for s,t in PLANET_PAIRS
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available: %s" % ', '.join('%s->%s' % (s,t) for s,t in pl_avail))
ax_pl, s_pl, _, pc_pl = axis_summary('planet->orbital-pos', pl_avail)
if ax_pl is not None:
    _, _, valid_pl, _ = compute_axis(pl_avail)
    print()
    for s, t, sid, _ in valid_pl:
        r = nn_retrieve(W_E[sid]+s_pl*ax_pl, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-10s -> %-4s  got=%-4s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
print()

# ====================================================================
# PART C: SEASONS -> QUARTER
# ====================================================================
print("PART C: Seasons -> quarter (Spring=1, Summer=2, Autumn=3, Winter=4)")
print("-"*65)

SEASONS = ['Spring','Summer','Autumn','Winter']
SEASON_PAIRS = [(SEASONS[i], str(i+1)) for i in range(4)]

se_avail = [(s,t) for s,t in SEASON_PAIRS
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available: %s" % ', '.join('%s->%s' % (s,t) for s,t in se_avail))
ax_se, s_se, _, pc_se = axis_summary('season->quarter', se_avail)
if ax_se is not None:
    _, _, valid_se, _ = compute_axis(se_avail)
    for s, t, sid, _ in valid_se:
        r = nn_retrieve(W_E[sid]+s_se*ax_se, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-10s -> %-4s  got=%-4s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
print()

# ====================================================================
# PART D: PLAYING CARD RANK -> NUMBER
# ====================================================================
print("PART D: Playing card names -> number (Two=2, ..., Nine=9, Ten=10)")
print("-"*65)

CARD_NAMES = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten',
              'Jack','Queen','King','Ace']
CARD_NUMS  = ['2','3','4','5','6','7','8','9','10','11','12','13','1']
CARD_PAIRS = list(zip(CARD_NAMES, CARD_NUMS))

cd_avail = [(s,t) for s,t in CARD_PAIRS
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available: %d/%d" % (len(cd_avail), len(CARD_PAIRS)))
ax_cd, s_cd, _, pc_cd = axis_summary('card->number', cd_avail)
if ax_cd is not None:
    _, _, valid_cd, _ = compute_axis(cd_avail)
    for s, t, sid, _ in valid_cd:
        r = nn_retrieve(W_E[sid]+s_cd*ax_cd, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-8s -> %-4s  got=%-4s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
print()

# ====================================================================
# PART E: CROSS-DOMAIN COSINES vs REFERENCE AXES
# ====================================================================
print("PART E: Cross-domain cosines (vs month->num and weekday->num)")
print("-"*65)

# Build reference axes from Day 298
MONTHS  = ['January','February','March','April','May','June',
           'July','August','September','October','November','December']
WEEKDAYS= ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

mn_pairs = [(MONTHS[i], str(i+1)) for i in range(9)
            if get_emb(MONTHS[i])[0] is not None and get_emb(str(i+1))[0] is not None]
wd_pairs = [(WEEKDAYS[i], str(i+1)) for i in range(7)
            if get_emb(WEEKDAYS[i])[0] is not None and get_emb(str(i+1))[0] is not None]
dw_pairs = [(str(i+1), ['one','two','three','four','five','six','seven','eight','nine'][i])
            for i in range(9) if get_emb(str(i+1))[0] is not None
            and get_emb(['one','two','three','four','five','six','seven','eight','nine'][i])[0] is not None]

ax_mn, _, _, _ = compute_axis(mn_pairs)
ax_wd, _, _, _ = compute_axis(wd_pairs)
ax_dw, _, _, _ = compute_axis(dw_pairs)

test_axes = {
    'letter->pos': ax_la,
    'lc-letter->pos': ax_lc,
    'planet->orbital': ax_pl,
    'season->quarter': ax_se,
    'card->number': ax_cd,
}

ref_axes = {
    'month->num': ax_mn,
    'weekday->num': ax_wd,
    'digit->word': ax_dw,
}

print("  Inter-axis cosines:")
for new_label, ax_new in test_axes.items():
    if ax_new is None: continue
    for ref_label, ax_ref in ref_axes.items():
        if ax_ref is None: continue
        c = float(np.dot(ax_new.astype(np.float32), ax_ref.astype(np.float32)))
        print("  %-22s <-> %-14s  cos=%+.4f" % (new_label, ref_label, c))
    print()

# ====================================================================
# PART F: LETTER CLUSTER ANALYSIS
# Does PC1 of letters correlate with alphabetical position?
# ====================================================================
print("PART F: Letter cluster analysis (SVD)")
print("-"*65)

uc_letters_avail = [l for l in LETTERS if get_emb(l)[0] is not None]
lc_letters_avail = [l for l in lc_letters if get_emb(l)[0] is not None]

for group_name, letters_avail in [
    ('Uppercase letters', uc_letters_avail),
    ('Lowercase letters', lc_letters_avail),
]:
    if len(letters_avail) < 10: continue
    L_mat = np.array([get_emb(l)[0] for l in letters_avail])
    L_c = L_mat - L_mat.mean(axis=0)
    U, S, Vt = np.linalg.svd(L_c, full_matrices=False)
    var_ratio = S**2 / (S**2).sum()
    proj = U[:, :2] * S[:2]
    idxs = [LETTERS.index(l) if l in LETTERS else lc_letters.index(l) for l in letters_avail]
    r1, p1 = pearsonr(idxs, proj[:,0])
    r2, p2 = pearsonr(idxs, proj[:,1])
    src_pc_l = float(np.mean([
        np.dot(normed(get_emb(letters_avail[i])[0]).astype(np.float32),
               normed(get_emb(letters_avail[j])[0]).astype(np.float32))
        for i in range(len(letters_avail)) for j in range(i+1,len(letters_avail))
    ]))
    print("  %s:" % group_name)
    print("  n=%d  src_pc=%.4f" % (len(letters_avail), src_pc_l))
    print("  Var explained: PC1=%.3f  PC2=%.3f" % (var_ratio[0], var_ratio[1]))
    print("  Pearson r(alpha_idx, PC1) = %.4f  p=%.4f" % (r1, p1))
    print("  Pearson r(alpha_idx, PC2) = %.4f  p=%.4f" % (r2, p2))
    print()

# ====================================================================
# PART G: ARITHMETIC AXIS (n -> n+1, n -> n+2, n -> n+5)
# Is the "increment by k" operation linear in W_E?
# ====================================================================
print("PART G: Arithmetic axes (digit n -> n+k)")
print("-"*65)

single_digits = [d for d in '123456789' if get_emb(d)[0] is not None]

for k in [1, 2, 3]:
    arith_pairs = [(single_digits[i], single_digits[i+k])
                   for i in range(len(single_digits)-k)
                   if i+k < len(single_digits)]
    avail = [(s,t) for s,t in arith_pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, pc = compute_axis(avail)
    if ax is None: continue
    _, _, valid, _ = compute_axis(avail)
    s_opt, acc = best_scale(ax, valid)
    print("  n->n+%d:  n=%-2d  pc=%+.4f  train=%d/%d (%.0f%%)  pairs: %s" % (
        k, len(avail), pc, acc, len(avail), 100*acc/max(1,len(avail)),
        ', '.join('%s->%s'%(s,t) for s,t in avail)))

    # Are n->n+k and n->n+1 aligned? (test for linearity of arithmetic)
    if k > 1:
        ax1, _, _, _ = compute_axis([(single_digits[i], single_digits[i+1])
                                      for i in range(len(single_digits)-1)
                                      if i+1 < len(single_digits)])
        if ax1 is not None:
            c = float(np.dot(ax.astype(np.float32), ax1.astype(np.float32)))
            print("    cos(n->n+%d, n->n+1) = %.4f" % (k, c))
print()

# ====================================================================
# PART H: UPDATED LINEARITY SPECTRUM
# ====================================================================
print("="*65)
print("UPDATED LINEARITY SPECTRUM (Day 299)")
print("="*65)

new_entries = []
for label, pairs in [
    ('letter->alpha (UC)', ok_pairs),
    ('lc-letter->alpha',   lc_pairs),
    ('planet->orbital',    pl_avail),
    ('season->quarter',    se_avail),
    ('card->number',       cd_avail),
]:
    if len([(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]) < 2:
        continue
    _, _, _, pc = compute_axis([(s,t) for s,t in pairs
                                 if get_emb(s)[0] is not None and get_emb(t)[0] is not None])
    if pc is not None:
        new_entries.append((label, pc, 'ORDINAL'))

for k in [1, 2, 3]:
    arith_pairs = [(single_digits[i], single_digits[i+k])
                   for i in range(len(single_digits)-k)
                   if i+k < len(single_digits)]
    avail = [(s,t) for s,t in arith_pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    _, _, _, pc = compute_axis(avail)
    if pc is not None:
        new_entries.append(('digit n->n+%d' % k, pc, 'NUMERIC'))

PREV = [
    ("digit->word",        0.851, "NUMERIC"),
    ("weekday->number",    0.842, "NUMERIC"),
    ("month->number",      0.803, "TEMPORAL"),
    ("ordinal->cardinal",  0.582, "NUMERIC"),
    ("country->demonym",   0.563, "SEMANTIC"),
    ("+est (sup)",         0.436, "INFL"),
    ("+er (comp)",         0.393, "INFL"),
    ("elem:single",        0.390, "SEMANTIC"),
    ("country->cap",       0.317, "SEMANTIC"),
    ("past_irr",           0.230, "INFL"),
    ("gender",             0.213, "INFL"),
    ("+ness",              0.211, "DERIV"),
    ("+ed (past_r)",       0.174, "INFL"),
    ("+s plural",          0.155, "INFL"),
    ("+ment",              0.124, "DERIV"),
    ("+ful",               0.104, "DERIV"),
    ("word->antonym",      0.020, "SEMANTIC"),
    ("month (consec)",    -0.090, "TEMPORAL"),
    ("digit consec",      -0.115, "NUMERIC"),
    ("weekday (consec)",  -0.153, "TEMPORAL"),
]

all_axes = new_entries + PREV
all_axes.sort(key=lambda x: -(x[1] if x[1] is not None else -99))
print()
print("  %-28s  pc_cos   type" % "Axis")
print("  " + "-"*52)
for name, pc, atype in all_axes:
    if pc is None or (isinstance(pc, float) and pc != pc): continue
    print("  %-28s  %+.4f   %s" % (name, pc, atype))
