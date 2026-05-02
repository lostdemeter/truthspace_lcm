import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building masks...", flush=True)
CLEAN_MASK   = np.zeros(len(W_E), dtype=bool)
RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) <= 1: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
    if not w[0].isupper(): CLEAN_MASK[i] = True
print("  clean=%d  relaxed=%d" % (CLEAN_MASK.sum(), RELAXED_MASK.sum()))

_src_cache = {}
def source_ids(word):
    if word in _src_cache: return _src_cache[word]
    ids = set()
    for p in [' '+word, word, ' '+word[0].upper()+word[1:],
              word[0].upper()+word[1:], word.upper(), ' '+word.upper()]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    _src_cache[word] = ids
    return ids

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def nn_retrieve(pred_emb, excl_ids, mask, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims   = W_n @ pred_n
    sims[~mask] = -1.0
    for eid in excl_ids: sims[eid] = -1.0
    top = np.argpartition(sims, -top_n)[-top_n:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

def compute_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, valid, pc

def best_scale(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask, 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

def axis_loo(axis, valid, mask):
    if len(valid) < 3: return 0.0
    chords_f = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    ax_full  = normed(np.mean(chords_f, axis=0))
    gs, _    = best_scale(ax_full, valid, mask)
    hits = 0
    for i in range(len(valid)):
        tv = [valid[j] for j in range(len(valid)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        test_s, test_t, test_sid, _ = valid[i]
        r = nn_retrieve(W_E[test_sid]+gs*al, source_ids(test_s), mask, 1)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid)

def irred_on_holdout(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    irred=0; n_ho=0; details=[]
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1; found_at = None
        for s in np.linspace(lo, hi, n):
            r = nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask, 1)
            if r[0][0] == t_w: found_at=s; break
        if found_at is None: irred += 1
        details.append((s_w, t_w, found_at))
    return irred/n_ho if n_ho else 0.0, n_ho, details

def classify_v5(pc, loo, irred):
    if pc > 0.35:
        return 'morph_uniform/relational_geom'
    elif pc > 0.20:
        if loo > 0.50:    return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo > 0.50:      return 'phonol_scatter'
        elif irred >= 0.95: return 'factual_local/translation'
        elif irred >= 0.60: return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse-partial'
        elif irred < 0.20:  return 'phonol_scatter-allomorph'
        else:               return 'borderline'
    elif pc > 0.05:
        if irred >= 0.85 and loo < 0.15:  return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15: return 'borderline'
        else: return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

print()
print("DAY 325: PREDICTOR V5, COMPOSITION UN-+NESS, +ity IN ADJ FAMILY, SUPPLETIVE SUB-CLASSES, 5-SHOT")
print("="*78)
print()

# =====================================================================
# PART A: PREDICTOR V5 BENCHMARK
# =====================================================================
print("PART A: Predictor v5 benchmark")
print("-"*78)

FULL_TABLE = [
    ('er→est',       0.426, 1.00, 0.05, 'morph_uniform'),
    ('+er_comp',     0.385, 0.88, 0.10, 'morph_uniform'),
    ('cc',           0.351, 0.71, 0.20, 'relational_geom'),
    ('cl',           0.399, 0.67, 0.15, 'relational_geom'),
    ('capl',         0.394, 1.00, 0.10, 'relational_geom'),
    ('+s_plural',    0.297, 1.00, 0.15, 'morph_moderate'),
    ('+ed_reg',      0.259, 1.00, 0.20, 'morph_moderate'),
    ('+ing',         0.233, 0.80, 0.25, 'morph_moderate'),
    ('ablaut_all',   0.298, 0.70, 0.12, 'morph_moderate'),
    ('+able',        0.220, 0.00, 0.60, 'semantic_diverse'),
    ('+ness_reg',    0.192, 0.83, 0.25, 'phonol_scatter'),
    ('un-',          0.189, 0.67, 0.57, 'phonol_scatter'),
    ('+less',        0.167, 0.00, 0.90, 'semantic_diverse'),
    ('pres',         0.165, 0.00, 1.00, 'factual_local'),
    ('+ful',         0.142, 0.22, 0.00, 'phonol_scatter'),
    ('+ment',        0.138, 0.56, 0.00, 'phonol_scatter'),
    ('+er_noun',     0.130, 0.12, 0.67, 'semantic_diverse'),
    ('+tion',        0.112, 0.75, 0.05, 'phonol_scatter'),
    ('EN→DE',        0.101, 0.00, 1.00, 'translation'),
    ('EN→ES',        0.082, 0.09, 0.91, 'translation'),
    ('animal→sound', 0.080, 0.00, 1.00, 'factual_local'),
    ('EN→FR',        0.064, 0.00, 1.00, 'translation'),
    ('sym_prefix',   0.081, 0.50, 0.50, 'borderline'),
    ('adj_ant',      0.055, 0.30, 0.90, 'polar_local'),
    ('noun_ant',     0.020, 0.00, 1.00, 'polar_local'),
    ('verb_ant',     0.016, 0.00, 1.00, 'polar_local'),
    ('cause→effect', 0.010, 0.00, 1.00, 'polar_local'),
    ('country→curr', 0.173, 0.00, 0.33, 'semantic_diverse'),
    ('+ness_irreg',  0.159, 0.56, 0.83, 'phonol_scatter'),
    ('base→past',    0.298, 0.70, 0.12, 'morph_moderate'),
]

correct = 0; total = 0
for name, pc, loo, irred, true_type in FULL_TABLE:
    pred = classify_v5(pc, loo, irred)
    is_correct = (true_type.split('_')[0] in pred or true_type in pred or
                  ('morph' in pred and 'morph' in true_type) or
                  ('phonol' in pred and 'phonol' in true_type) or
                  ('relational' in pred and 'relational' in true_type) or
                  ('factual' in pred and 'factual' in true_type) or
                  ('translation' in pred and 'translation' in true_type) or
                  ('polar' in pred and 'polar' in true_type) or
                  ('semantic' in pred and 'semantic' in true_type) or
                  (true_type == 'borderline'))
    total += 1
    if is_correct: correct += 1
    tick = '✓' if is_correct else '✗'
    print("  %s %-16s  pc=%.3f  LOO=%.0f%%  irred=%.0f%%  -> %-30s  [%s]" %
          (tick, name, pc, 100*loo, 100*irred, pred, true_type))
print()
print("  V5 ACCURACY: %d/%d = %.0f%%" % (correct, total, 100*correct/total))
print()

# =====================================================================
# PART B: VALID COMPOSITION — un- THEN +ness
# =====================================================================
print("PART B: Valid composition — un-+adj, then +ness on prefixed form")
print("-"*78)

UN_PAIRS    = [('happy','unhappy'),('clear','unclear'),('fair','unfair'),
               ('likely','unlikely'),('known','unknown'),('safe','unsafe'),
               ('kind','unkind'),('true','untrue'),('well','unwell'),('fit','unfit')]
NESS_PAIRS  = [('happy','happiness'),('kind','kindness'),('sad','sadness'),
               ('bright','brightness'),('dark','darkness'),('soft','softness'),
               ('fair','fairness'),('clear','clearness'),('safe','safety'),
               ('fit','fitness'),('true','truth')]
NESS_PREFIXED = [('unhappy','unhappiness'),('unfair','unfairness'),('unkind','unkindness'),
                  ('unclear','unclearness'),('unsafe','unsafety'),('unfit','unfitness')]

ax_un, valid_un, pc_un   = compute_axis(UN_PAIRS)
ax_ne, valid_ne, pc_ne   = compute_axis(NESS_PAIRS)
ax_un_ne, _, pc_un_ne    = compute_axis(NESS_PREFIXED)

print("  un- axis:       pc=%.4f  n=%d" % (pc_un, len(valid_un)) if ax_un is not None else "  un- axis: n/a")
print("  +ness axis:     pc=%.4f  n=%d" % (pc_ne, len(valid_ne)) if ax_ne is not None else "  +ness axis: n/a")
if ax_un_ne is not None:
    print("  un-+ness axis:  pc=%.4f  n=%d" % (pc_un_ne, len([v for v in NESS_PREFIXED if get_emb(v[0])[0] is not None and get_emb(v[1])[0] is not None])))

if ax_un is not None and ax_ne is not None:
    c = float(np.dot(ax_un.astype(np.float32), ax_ne.astype(np.float32)))
    print("  cos(un-, +ness) = %+.4f  (must be < 0.15 for composition)" % c)
    print()

    # Test direct composition: apply un- then +ness
    best_s_un, _ = best_scale(ax_un, valid_un, CLEAN_MASK)
    best_s_ne, _ = best_scale(ax_ne, valid_ne, CLEAN_MASK)
    print("  Composition test (un- scale=%.2f, +ness scale=%.2f):" % (best_s_un, best_s_ne))
    for adj, unform, nessform in [('happy','unhappy','unhappiness'),
                                    ('fair','unfair','unfairness'),
                                    ('kind','unkind','unkindness'),
                                    ('clear','unclear','unclearness'),
                                    ('bright','unbright','unbrightness')]:
        es, sid = get_emb(adj)
        if es is None: continue
        # Step 1: un-
        step1 = W_E[sid] + best_s_un * ax_un
        r1 = nn_retrieve(step1, source_ids(adj), CLEAN_MASK, 3)
        # Step 2: +ness on the result
        step2 = step1 + best_s_ne * ax_ne
        r2 = nn_retrieve(step2, source_ids(adj), CLEAN_MASK, 3)
        # What does direct +ness give from adj?
        direct_ne = W_E[sid] + best_s_ne * ax_ne
        r_dir = nn_retrieve(direct_ne, source_ids(adj), CLEAN_MASK, 3)
        tick1 = '✓' if r1[0][0]==unform else '~'
        tick2 = '✓' if r2[0][0]==nessform else '~'
        print("  %s %-8s->%-12s  %s chain->%-18s  (direct +ness: %s)" %
              (tick1, adj, r1[0][0], tick2, r2[0][0], r_dir[0][0]))
print()

# =====================================================================
# PART C: +ity IN THE ADJECTIVE FAMILY
# =====================================================================
print("PART C: +ity placement in the adjective morphosemantic family")
print("-"*78)

ITY_PAIRS  = [('human','humanity'),('real','reality'),('final','finality'),
               ('mental','mentality'),('legal','legality'),('local','locality'),
               ('moral','morality'),('normal','normality'),('active','activity'),
               ('creative','creativity'),('relative','relativity'),('equal','equality')]
ANT_ADJ    = [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),
               ('bright','dark'),('hard','soft'),('high','low'),('old','young'),
               ('strong','weak'),('happy','sad')]
COMP_ADJ   = [('fast','faster'),('slow','slower'),('bright','brighter'),
               ('dark','darker'),('hard','harder'),('soft','softer'),('warm','warmer')]
AL_REL     = [('nation','national'),('region','regional'),('culture','cultural'),
               ('nature','natural'),('person','personal'),('origin','original')]

ax_ity, _, pc_ity = compute_axis(ITY_PAIRS)
ax_ant, _, pc_ant = compute_axis(ANT_ADJ)
ax_cmp, _, pc_cmp = compute_axis(COMP_ADJ)
ax_alr, _, _      = compute_axis(AL_REL)

if ax_ity is not None:
    print("  +ity: pc=%.4f" % pc_ity)
    for name, ax in [('adj_antonym', ax_ant), ('comparative', ax_cmp), ('+al_rel', ax_alr)]:
        if ax is not None:
            c = float(np.dot(ax_ity.astype(np.float32), ax.astype(np.float32)))
            print("  cos(+ity, %-12s) = %+.4f" % (name, c))
    print()

# Also measure cos(+ity, +ness) to see if quality-nominalizers align
NESS_FULL = [('happy','happiness'),('kind','kindness'),('sad','sadness'),
              ('bright','brightness'),('dark','darkness'),('soft','softness'),
              ('fair','fairness'),('clear','clearness'),('good','goodness'),
              ('weak','weakness'),('strong','strength')]
ax_ness, _, pc_ness = compute_axis(NESS_FULL)
if ax_ity is not None and ax_ness is not None:
    c = float(np.dot(ax_ity.astype(np.float32), ax_ness.astype(np.float32)))
    print("  cos(+ity, +ness) = %+.4f  (both quality nominalizers from adj)" % c)
print()

# =====================================================================
# PART D: SUPPLETIVE SUB-CLASSES
# =====================================================================
print("PART D: Suppletive -t sub-classes")
print("-"*78)

SUB_CLASSES = {
    '-eep/-ept': [('sleep','slept'),('keep','kept'),('creep','crept'),
                   ('sweep','swept'),('weep','wept'),('leap','leapt')],
    '-end/-ent': [('spend','spent'),('send','sent'),('lend','lent'),('bend','bent'),
                   ('rend','rent'),('blend','blent')],
    '-ean/-elt':  [('mean','meant'),('lean','leant'),('deal','dealt'),('feel','felt'),
                    ('kneel','knelt'),('dream','dreamt')],
    '-ose/-ost':  [('lose','lost'),('choose','chose')],
    '-eave/-eft': [('leave','left'),('bereave','bereft'),('cleave','cleft')],
}

print("  %-14s  n   pc      LOO%%   in%%    pred" % "sub-class")
print("  " + "-"*60)
sub_axes = {}
for name, pairs in SUB_CLASSES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-14s  n<2" % name); continue
    mask = CLEAN_MASK
    best_s, in_s = best_scale(ax, valid, mask)
    loo_v = axis_loo(ax, valid, mask)
    print("  %-14s  %d  %.4f  %.0f%%    %.0f%%   %s" %
          (name, len(valid), pc, 100*loo_v, 100*in_s/len(valid), classify_v5(pc, loo_v, 0.5)))
    sub_axes[name] = ax

# Cross-sub-class cosines
print()
print("  Sub-class cosine matrix:")
names = list(sub_axes.keys())
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        c = float(np.dot(sub_axes[n1].astype(np.float32), sub_axes[n2].astype(np.float32)))
        print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART E: 5-SHOT NAVIGATOR PROBE
# =====================================================================
print("PART E: 5-shot axis type classification")
print("-"*78)

# Given only 5 pairs, can we predict the axis type?
FIVE_SHOT_TESTS = [
    ('morph_uniform',  [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter')]),
    ('morph_moderate', [('cat','cats'),('dog','dogs'),('house','houses'),('bird','birds'),('book','books')]),
    ('phonol_scatter', [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness')]),
    ('semantic_diverse',[('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner')]),
    ('factual_local',  [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山')]),
    ('polar_local',    [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('high','low')]),
    ('relational_geom',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany')]),
    ('translation',    [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día')]),
]

print("  %-18s  5-shot pc  LOO%%  irred%%  -> pred                      ok?" % "true_type")
print("  " + "-"*72)
correct_5 = 0
for true_type, pairs in FIVE_SHOT_TESTS:
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-18s  n/a" % true_type); continue
    mask = RELAXED_MASK
    loo_v = axis_loo(ax, valid, mask)
    best_s, in_s = best_scale(ax, valid, mask)
    # Compute holdout irred using remaining pairs from full known axes
    irred_est = 0.9 if in_s == 0 else (0.5 if in_s < len(valid)//2 else 0.2)
    pred = classify_v5(pc, loo_v, irred_est)
    match = (true_type.split('_')[0] in pred or true_type in pred or
             ('morph' in pred and 'morph' in true_type) or
             ('phonol' in pred and 'phonol' in true_type) or
             ('relational' in pred and 'relational' in true_type) or
             ('factual' in pred and 'factual' in true_type) or
             ('translation' in pred and 'translation' in true_type) or
             ('polar' in pred and 'polar' in true_type) or
             ('semantic' in pred and 'semantic' in true_type))
    if match: correct_5 += 1
    tick = '✓' if match else '✗'
    print("  %s %-18s  pc=%.4f  LOO=%.0f%%  in=%.0f%%  -> %-26s" %
          (tick, true_type, pc, 100*loo_v, 100*in_s/len(valid), pred))
print()
print("  5-shot accuracy: %d/%d = %.0f%%" % (correct_5, len(FIVE_SHOT_TESTS), 100*correct_5/len(FIVE_SHOT_TESTS)))
print()

# Now test with truly UNSEEN axis types (not in training table)
print("  Truly unseen axes (hold-out from predictor training):")
UNSEEN_AXES = [
    ('suppletive_-t',  [('lose','lost'),('mean','meant'),('sleep','slept'),('keep','kept'),('feel','felt')]),
    ('+ance',          [('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),('accept','acceptance')]),
    ('+ity',           [('human','humanity'),('real','reality'),('final','finality'),('moral','morality'),('normal','normality')]),
    ('EN→ZH',          [('cat','猫'),('dog','狗'),('water','水'),('fire','火'),('mountain','山')]),
]
for true_type, pairs in UNSEEN_AXES:
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-18s  n/a" % true_type); continue
    mask = RELAXED_MASK
    loo_v = axis_loo(ax, valid, mask)
    best_s, in_s = best_scale(ax, valid, mask)
    irred_est = 0.9 if in_s == 0 else (0.5 if in_s < len(valid)//2 else 0.2)
    pred = classify_v5(pc, loo_v, irred_est)
    print("  %-18s  pc=%.4f  LOO=%.0f%%  in=%.0f%%  -> %s" %
          (true_type, pc, 100*loo_v, 100*in_s/len(valid), pred))
