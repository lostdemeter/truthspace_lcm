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
def nn_retrieve(pred_emb, exclude_ids, top_n=3):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=80):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def eval_axis(pairs, label, hold_pairs=None):
    ax, coh, valid, pc = compute_axis(pairs)
    if ax is None: print("  %-22s  SKIP (no valid pairs)" % label); return None, None, None
    s_opt, acc_tr = best_scale(ax, valid)
    n_tr = len(valid)
    hold_str = ''
    if hold_pairs:
        hold_r = [(s,t,*get_emb(s)[:1]) for s,t in hold_pairs]
        acc_h = 0; n_h = 0
        for s, t, es in hold_r:
            sid = get_emb(s)[1]
            if es is None: continue
            got = nn_retrieve(W_E[sid]+s_opt*ax, [sid])[0][0]
            if got == t: acc_h += 1
            n_h += 1
        hold_str = '  hold=%d/%d (%.0f%%)' % (acc_h, n_h, 100*acc_h/max(1,n_h))
    print("  %-22s  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)%s" % (
        label, pc, coh, s_opt, acc_tr, n_tr, 100*acc_tr/max(1,n_tr), hold_str))
    return ax, s_opt, valid
def eval_full(pairs, label, ax, scale):
    if ax is None: return
    results = []
    for s, t in pairs:
        es, sid = get_emb(s)
        if es is None: results.append((s, t, None, '?', False)); continue
        r = nn_retrieve(W_E[sid]+scale*ax, [sid])
        got = r[0][0] if r else '?'
        results.append((s, t, sid, got, got==t))
    acc = sum(1 for _,_,sid,_,hit in results if hit and sid is not None)
    n   = sum(1 for _,_,sid,_,_ in results if sid is not None)
    print("  %-22s  %d/%d (%.0f%%)" % (label, acc, n, 100*acc/max(1,n)))
    for s, t, sid, got, hit in results:
        if sid is None: continue
        print("    %-14s -> %-16s  got=%-16s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    return results

print("DAY 295: DERIVATIONAL MORPHOLOGY AXES")
print("="*65)
print("Testing: +ness (adj->noun), un- prefix, -ful, -less, -ment")
print("Hypothesis: derivational morphology obeys same linearity law")
print("as inflectional morphology and semantic axes.")
print()

# ====================================================================
# AXIS DEFINITIONS
# ====================================================================

# A: adjective -> abstract noun via +ness
NESS_TRAIN = [
    ('happy','happiness'),('sad','sadness'),('kind','kindness'),
    ('dark','darkness'),('soft','softness'),('hard','hardness'),
    ('warm','warmth'),('cold','coldness'),('bright','brightness'),
    ('clean','cleanness'),('loud','loudness'),('sweet','sweetness'),
    ('weak','weakness'),('bold','boldness'),('calm','calmness'),
]
NESS_HOLD = [
    ('neat','neatness'),('sharp','sharpness'),('smooth','smoothness'),
    ('quick','quickness'),('still','stillness'),('dark','darkness'),
]

# B: adjective -> negative adj via un-
UNPREF_TRAIN = [
    ('happy','unhappy'),('kind','unkind'),('fair','unfair'),
    ('clear','unclear'),('safe','unsafe'),('lucky','unlucky'),
    ('common','uncommon'),('known','unknown'),('likely','unlikely'),
    ('usual','unusual'),('even','uneven'),('real','unreal'),
]
UNPREF_HOLD = [
    ('well','unwell'),('able','unable'),('fit','unfit'),
    ('worthy','unworthy'),('sound','unsound'),('fair','unfair'),
]

# C: adjective -> negative adj via in- / im- / il- / ir-  (mixed)
INPREF_MIXED = [
    ('possible','impossible'),('logical','illogical'),
    ('regular','irregular'),('complete','incomplete'),
    ('correct','incorrect'),('direct','indirect'),
    ('formal','informal'),('active','inactive'),
    ('visible','invisible'),('legal','illegal'),
    ('rational','irrational'),('relevant','irrelevant'),
]
INPREF_HOLD = [
    ('proper','improper'),('moral','immoral'),('patient','impatient'),
    ('personal','impersonal'),('adequate','inadequate'),('credible','incredible'),
]

# D: +ful suffix (adj from noun)
FUL_TRAIN = [
    ('hope','hopeful'),('care','careful'),('help','helpful'),
    ('wonder','wonderful'),('color','colorful'),('power','powerful'),
    ('peace','peaceful'),('grace','graceful'),('skill','skillful'),
    ('use','useful'),('cheer','cheerful'),('faith','faithful'),
]
FUL_HOLD = [
    ('harm','harmful'),('delight','delightful'),('respect','respectful'),
    ('thought','thoughtful'),('beauty','beautiful'),('play','playful'),
]

# E: +less suffix (opposite of +ful in many cases)
LESS_TRAIN = [
    ('hope','hopeless'),('care','careless'),('help','helpless'),
    ('power','powerless'),('use','useless'),('worth','worthless'),
    ('home','homeless'),('end','endless'),('sleep','sleepless'),
    ('harm','harmless'),('fear','fearless'),('count','countless'),
]
LESS_HOLD = [
    ('thought','thoughtless'),('taste','tasteless'),('meaning','meaningless'),
    ('rest','restless'),('fault','faultless'),('pain','painless'),
]

# F: +ment (verb -> noun)
MENT_TRAIN = [
    ('move','movement'),('treat','treatment'),('agree','agreement'),
    ('judge','judgment'),('manage','management'),('pay','payment'),
    ('state','statement'),('replace','replacement'),('require','requirement'),
    ('achieve','achievement'),('improve','improvement'),('develop','development'),
]
MENT_HOLD = [
    ('employ','employment'),('invest','investment'),('govern','government'),
    ('argue','argument'),('amuse','amusement'),('amaze','amazement'),
]

# G: +tion / +ion (verb -> noun, most common derivational)
TION_TRAIN = [
    ('act','action'),('connect','connection'),('direct','direction'),
    ('select','selection'),('protect','protection'),('collect','collection'),
    ('produce','production'),('reduce','reduction'),('conduct','conduction'),
    ('reflect','reflection'),('predict','prediction'),('inspect','inspection'),
]
TION_HOLD = [
    ('correct','correction'),('affect','affection'),('detect','detection'),
    ('inject','injection'),('project','projection'),('elect','election'),
]

# H: +ful vs +less axis (are they anti-parallel?)
# This tests whether ful and less form a single +/- dimension
FULLESS_TRAIN = FUL_TRAIN[:6]  # hope/care/help/wonder/color/power -> +ful
FULLESS_LESS  = LESS_TRAIN[:6]  # hope/care/help/power/use/worth -> +less

# ====================================================================
# PART A: OVERALL AXIS QUALITY
# ====================================================================
print("PART A: Axis quality for each derivational rule")
print("-"*65)

axes = {}
for label, train, hold in [
    ('+ness',     NESS_TRAIN,    NESS_HOLD),
    ('un-',       UNPREF_TRAIN,  UNPREF_HOLD),
    ('in-/im-',   INPREF_MIXED,  INPREF_HOLD),
    ('+ful',      FUL_TRAIN,     FUL_HOLD),
    ('+less',     LESS_TRAIN,    LESS_HOLD),
    ('+ment',     MENT_TRAIN,    MENT_HOLD),
    ('+tion',     TION_TRAIN,    TION_HOLD),
]:
    ax, scale, valid = eval_axis(train, label, hold)
    axes[label] = (ax, scale, valid)
print()

# ====================================================================
# PART B: DETAILED BREAKDOWN FOR BEST AND WORST AXES
# ====================================================================
print("PART B: Detailed results for +ness and un-")
print("-"*65)

for label, train, hold in [
    ('+ness (train)', NESS_TRAIN, None),
    ('+ness (hold)',  NESS_HOLD,  None),
    ('un- (train)',   UNPREF_TRAIN, None),
    ('un- (hold)',    UNPREF_HOLD,  None),
]:
    lk = label.split(' ')[0]
    ax_key = '+ness' if 'ness' in label else 'un-'
    ax, scale, valid = axes.get(ax_key, (None, None, None))
    if ax is None: continue
    if 'train' in label:
        base_pairs = NESS_TRAIN if 'ness' in label else UNPREF_TRAIN
    else:
        base_pairs = NESS_HOLD if 'ness' in label else UNPREF_HOLD
    eval_full(base_pairs, label, ax, scale)
    print()

# ====================================================================
# PART C: ful vs less — ANTI-PARALLEL AXIS TEST
# ====================================================================
print("PART C: +ful vs +less — anti-parallel test")
print("-"*65)

ax_ful, _, valid_ful, pc_ful = compute_axis(FULLESS_TRAIN)
ax_less, _, valid_less, pc_less = compute_axis(FULLESS_LESS)

if ax_ful is not None and ax_less is not None:
    cos_fl = float(np.dot(ax_ful.astype(np.float32), ax_less.astype(np.float32)))
    print("  cos(+ful axis, +less axis) = %.4f" % cos_fl)
    print("  (expect -1.0 if +ful and +less are anti-parallel)")
    print("  pc(+ful)  = %.4f" % pc_ful)
    print("  pc(+less) = %.4f" % pc_less)
    print()
    # Test: does the +ful axis REVERSE reach +less targets?
    # hope + ful_axis  => hopeful
    # hope - ful_axis  => hopeless?
    s_ful, _ = best_scale(ax_ful, valid_ful)
    s_less, _ = best_scale(ax_less, valid_less)
    print("  Testing +ful forward: hop/care/help/power/use")
    for s, t, sid, tid in valid_ful[:5]:
        r = nn_retrieve(W_E[sid]+s_ful*ax_ful, [sid])
        print("    %-10s -> %-12s  got=%s" % (s, t, r[0][0] if r else '?'))
    print()
    print("  Testing reversed +ful (= ful_axis * -1.0) -> should reach +less")
    for s, t, sid, tid in valid_ful[:5]:
        # what's the +less form of this word?
        s_less_form, _ = get_emb(s.rstrip('e')+'less' if s.endswith('e') else s+'less')
        r_rev = nn_retrieve(W_E[sid]-s_ful*ax_ful, [sid])
        print("    %-10s [-ful]  got=%s" % (s, r_rev[0][0] if r_rev else '?'))
    print()

# ====================================================================
# PART D: SCALE COMPARISONS (density hypothesis)
# ====================================================================
print("PART D: Scale comparison across derivational axes")
print("-"*65)
print("  Scale ratio tests the ENCODE=DECODE density hypothesis")
print()

for label, train in [
    ('+ness',   NESS_TRAIN),
    ('un-',     UNPREF_TRAIN),
    ('+ful',    FUL_TRAIN),
    ('+less',   LESS_TRAIN),
    ('+ment',   MENT_TRAIN),
    ('+tion',   TION_TRAIN),
]:
    ax, _, valid, pc = compute_axis(train)
    if ax is None: continue
    rev_pairs = [(t, s) for s, t, sid, tid in valid]
    ax_rev, _, valid_rev, _ = compute_axis(rev_pairs)
    if ax_rev is None: continue
    s_fwd, acc_fwd = best_scale(ax, valid)
    s_rev, acc_rev = best_scale(ax_rev, valid_rev)
    cos_fr = float(np.dot(ax.astype(np.float32), ax_rev.astype(np.float32)))
    print("  %-8s  fwd_scale=%.2f  rev_scale=%.2f  ratio=%.3f  cos=%.4f" % (
        label, s_fwd, s_rev, s_fwd/max(0.001,s_rev), cos_fr))
print()

# ====================================================================
# PART E: CROSS-DERIVATIONAL GENERALISATION
# Do the un- and in-/im- axes point in the same direction?
# ====================================================================
print("PART E: Negation axes — un- vs in-/im- cosine")
print("-"*65)

ax_un, _, _, pc_un = compute_axis(UNPREF_TRAIN)
ax_in, _, _, pc_in = compute_axis(INPREF_MIXED)
ax_ful2, _, _, _ = compute_axis(FUL_TRAIN)
ax_less2, _, _, _ = compute_axis(LESS_TRAIN)

if ax_un is not None and ax_in is not None:
    cos_ui = float(np.dot(ax_un.astype(np.float32), ax_in.astype(np.float32)))
    print("  cos(un-, in-/im-) = %.4f" % cos_ui)
    print("  Are both 'negation axes' pointing in the same direction?")
    print()

for lx, ax_x, ly, ax_y in [
    ('un-', ax_un, 'in-/im-', ax_in),
    ('un-', ax_un, '+less', ax_less2),
    ('in-/im-', ax_in, '+less', ax_less2),
    ('+ful', ax_ful2, '+less', ax_less2),
]:
    if ax_x is None or ax_y is None: continue
    cos_xy = float(np.dot(ax_x.astype(np.float32), ax_y.astype(np.float32)))
    print("  %-10s <-> %-10s  cos=%.4f" % (lx, ly, cos_xy))
print()

# Cross-test: does un- axis predict in-/im- targets?
if ax_un is not None:
    ax_un_full, _, valid_un, _ = compute_axis(UNPREF_TRAIN)
    s_un, _ = best_scale(ax_un_full, valid_un)
    cross_results = []
    for s, t in INPREF_HOLD:
        es, sid = get_emb(s)
        if es is None: continue
        got = nn_retrieve(W_E[sid]+s_un*ax_un_full, [sid])[0][0]
        cross_results.append((s, t, got, got==t))
    acc_cross = sum(1 for _,_,_,hit in cross_results if hit)
    print("  un- axis applied to in-/im- holdout: %d/%d (%.0f%%)" % (
        acc_cross, len(cross_results), 100*acc_cross/max(1,len(cross_results))))
    for s, t, got, hit in cross_results:
        print("    %-14s -> %-16s  got=%-16s [%s]" % (s, t, got, 'HIT' if hit else '---'))
print()

# ====================================================================
# PART F: UPDATED LINEARITY SPECTRUM (add derivational axes)
# ====================================================================
print("="*65)
print("UPDATED LINEARITY SPECTRUM (Days 290-295)")
print("="*65)
print()

# Compute fresh pc for all current derivational axes
new_axes = []
for label, pairs in [
    ('+ness', NESS_TRAIN), ('un-', UNPREF_TRAIN),
    ('in-/im-', INPREF_MIXED), ('+ful', FUL_TRAIN),
    ('+less', LESS_TRAIN), ('+ment', MENT_TRAIN),
    ('+tion', TION_TRAIN),
]:
    _, _, _, pc = compute_axis(pairs)
    new_axes.append((label, pc, 'DERIV'))

PREV = [
    ("country->demonym", 0.563, "SEMANTIC"),
    ("country->lang*",   0.474, "SEMANTIC"),
    ("+est (sup)",       0.436, "INFL"),
    ("+er (comp)",       0.393, "INFL"),
    ("elem:single-lett", 0.390, "SEMANTIC"),
    ("country->cap",     0.317, "SEMANTIC"),
    ("animal->class",    0.254, "SEMANTIC"),
    ("person->nat",      0.246, "SEMANTIC"),
    ("past_irr",         0.230, "INFL"),
    ("gender",           0.213, "INFL"),
    ("+ed (past_r)",     0.174, "INFL"),
    ("elem:double-lett", 0.163, "SEMANTIC"),
    ("+s plural",        0.155, "INFL"),
    ("element->sym",     0.139, "SEMANTIC"),
    ("elem:latin-deriv", 0.104, "SEMANTIC"),
    ("field->concept",   0.087, "SEMANTIC"),
    ("word->antonym",    0.020, "SEMANTIC"),
]
all_axes = new_axes + PREV
all_axes.sort(key=lambda x: -x[1])

print("  %-28s  pc_cos   type" % "Axis")
print("  " + "-"*52)
for name, pc, atype in all_axes:
    print("  %-28s  %.4f   %s" % (name, pc, atype))
print()
print("  INFL=inflectional, DERIV=derivational, SEMANTIC=non-morphological")
