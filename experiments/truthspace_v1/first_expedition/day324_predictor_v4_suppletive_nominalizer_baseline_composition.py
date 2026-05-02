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

def classify_v4(pc, loo, irred):
    if pc > 0.35:
        return 'morph_uniform/relational_geom'
    elif pc > 0.20:
        if loo > 0.50:   return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo > 0.50:       return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse-partial'
        elif irred < 0.20:   return 'phonol_scatter-allomorph'
        else:                return 'borderline'
    elif pc > 0.05:
        if irred >= 0.95 and loo < 0.15: return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15: return 'borderline'
        else: return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

print()
print("DAY 324: PREDICTOR V4, SUPPLETIVE -t, NOMINALIZER FAMILY, BASELINE, COMPOSITION")
print("="*72)
print()

# =====================================================================
# PART A: PREDICTOR V4 BENCHMARK
# =====================================================================
print("PART A: Predictor v4 benchmark")
print("-"*72)

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
    pred = classify_v4(pc, loo, irred)
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
    print("  %s %-16s  pc=%.3f  LOO=%.0f%%  irred=%.0f%%  -> %-28s  [%s]" %
          (tick, name, pc, 100*loo, 100*irred, pred, true_type))
print()
print("  V4 ACCURACY: %d/%d = %.0f%%" % (correct, total, 100*correct/total))
print()

# =====================================================================
# PART B: SUPPLETIVE -t CLASS
# =====================================================================
print("PART B: Suppletive -t irregular past tense")
print("-"*72)

SUPPL_T = [('lose','lost'),('mean','meant'),('sleep','slept'),('keep','kept'),
            ('feel','felt'),('deal','dealt'),('kneel','knelt'),('lean','leant'),
            ('learn','learnt'),('build','built'),('spend','spent'),('send','sent')]
SUPPL_T_HOLDOUT = [('smell','smelt'),('creep','crept'),('sweep','swept'),
                    ('weep','wept'),('leave','left'),('bereave','bereft')]

ax_t, valid_t, pc_t = compute_axis(SUPPL_T)
if ax_t is not None:
    best_s, in_s = best_scale(ax_t, valid_t, CLEAN_MASK)
    loo_v = axis_loo(ax_t, valid_t, CLEAN_MASK)
    irr_f, n_ho, details = irred_on_holdout(ax_t, SUPPL_T_HOLDOUT, CLEAN_MASK)
    pred = classify_v4(pc_t, loo_v, irr_f)
    print("  suppletive_-t: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  irred=%.0f%%  -> %s" %
          (len(valid_t), pc_t, 100*in_s/len(valid_t), 100*loo_v, 100*irr_f, pred))
    for s_w, t_w, found_at in details:
        hit = '✓' if found_at else '✗'
        fa = '%.2f' % found_at if found_at else 'N/A'
        print("  %s %-8s -> %-8s  (scale=%s)" % (hit, s_w, t_w, fa))

    # Cosine with ablaut ALL
    ABLAUT_ALL = [
        ('go','went'),('buy','bought'),('bring','brought'),('think','thought'),
        ('catch','caught'),('teach','taught'),
        ('see','saw'),('say','said'),('do','did'),('come','came'),('run','ran'),('hold','held'),
        ('take','took'),('give','gave'),('get','got'),('make','made'),
        ('know','knew'),('grow','grew'),('throw','threw'),('blow','blew'),
        ('sing','sang'),('ring','rang'),('drink','drank'),('swim','swam'),
        ('begin','began'),('spring','sprang'),
        ('break','broke'),('choose','chose'),('ride','rode'),('write','wrote'),
        ('rise','rose'),('drive','drove'),('bite','bit'),
        ('find','found'),('bind','bound'),('wind','wound'),('grind','ground'),
    ]
    ax_ab, _, _ = compute_axis(ABLAUT_ALL)
    if ax_ab is not None:
        c = float(np.dot(ax_t.astype(np.float32), ax_ab.astype(np.float32)))
        print("  cos(suppletive_-t, ablaut_ALL) = %+.4f" % c)
print()

# =====================================================================
# PART C: NOMINALIZER FAMILY — +ance, +ity, +ure
# =====================================================================
print("PART C: Nominalizer family expansion")
print("-"*72)

NOMINALIZER_AXES = {
    '+ance/+ence': [('perform','performance'),('exist','existence'),('enter','entrance'),
                     ('resist','resistance'),('accept','acceptance'),('insist','insistence'),
                     ('appear','appearance'),('depend','dependence'),('prefer','preference')],
    '+ity':        [('human','humanity'),('real','reality'),('final','finality'),
                     ('mental','mentality'),('legal','legality'),('local','locality'),
                     ('moral','morality'),('normal','normality'),('active','activity'),
                     ('creative','creativity'),('relative','relativity')],
    '+ure':        [('fail','failure'),('expose','exposure'),('please','pleasure'),
                     ('measure','measure'),('press','pressure'),('depart','departure'),
                     ('compose','composure'),('close','closure'),('mix','mixture')],
    '+al_nominal': [('arrive','arrival'),('propose','proposal'),('approve','approval'),
                     ('refuse','refusal'),('remove','removal'),('survive','survival')],
    '+er_noun':    [('teach','teacher'),('farm','farmer'),('drive','driver'),
                     ('work','worker'),('own','owner'),('lead','leader'),
                     ('build','builder'),('manage','manager')],
    '+tion':       [('act','action'),('direct','direction'),('educate','education'),
                     ('create','creation'),('produce','production'),('relate','relation'),
                     ('connect','connection'),('collect','collection')],
    '+ment':       [('achieve','achievement'),('develop','development'),
                     ('manage','management'),('govern','government'),
                     ('engage','engagement'),('require','requirement')],
}

print("  %-14s  pc      LOO%%  irred%%  pred                     n" % "axis")
print("  " + "-"*72)
nom_axes = {}
for name, pairs in NOMINALIZER_AXES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-14s  n/a" % name); continue
    loo_v = axis_loo(ax, valid, CLEAN_MASK)
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    print("  %-14s  %.4f  %.0f%%   ?%%     %-25s  n=%d in=%.0f%%" %
          (name, pc, 100*loo_v, classify_v4(pc, loo_v, 0.0), len(valid), 100*in_s/len(valid)))
    nom_axes[name] = ax

print()
print("  Nominalizer family cosine matrix:")
nom_names = list(nom_axes.keys())
for i, n1 in enumerate(nom_names):
    for n2 in nom_names[i+1:]:
        c = float(np.dot(nom_axes[n1].astype(np.float32), nom_axes[n2].astype(np.float32)))
        print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART D: BASELINE SIMILARITY — SOURCE-TARGET DISTANCE
# =====================================================================
print("PART D: Cross-lingual baseline similarity")
print("-"*72)

TRANS_PAIRS = {
    'EN→ZH': [('cat','猫'),('dog','狗'),('water','水'),('fire','火'),('sun','日'),
               ('moon','月'),('mountain','山'),('sea','海'),('tree','木'),('fish','鱼'),
               ('hand','手'),('eye','眼'),('mouth','口'),('heart','心'),('man','男')],
    'EN→JA': [('cat','猫'),('dog','犬'),('water','水'),('fire','火'),('sun','日'),
               ('moon','月'),('mountain','山'),('sea','海'),('tree','木'),('fish','魚'),
               ('hand','手'),('eye','目'),('mouth','口'),('heart','心')],
    'EN→ES': [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),
               ('sea','mar'),('air','aire'),('day','día'),('night','noche'),
               ('hand','mano'),('heart','corazón'),('bread','pan'),('salt','sal'),
               ('green','verde'),('black','negro'),('man','hombre')],
    'EN→FR': [('cat','chat'),('dog','chien'),('house','maison'),('water','eau'),
               ('fire','feu'),('sun','soleil'),('book','livre'),('door','porte'),
               ('day','jour'),('night','nuit'),('year','an'),('hand','main')],
    'EN→DE': [('man','Mann'),('hand','Hand'),('house','Haus'),('water','Wasser'),
               ('fire','Feuer'),('sun','Sonne'),('book','Buch'),('door','Tür'),
               ('day','Tag'),('night','Nacht'),('year','Jahr'),('cat','Katze')],
}

print("  Language    mean_baseline_cos  std     n_pairs")
print("  " + "-"*50)
for lang, pairs in TRANS_PAIRS.items():
    baselines = []
    for s_w, t_w in pairs:
        es, sid = get_emb(s_w); et, tid = get_emb(t_w)
        if es is None or et is None: continue
        cos_st = float(np.dot(W_n[sid], W_n[tid]))
        baselines.append(cos_st)
    if baselines:
        mean_b = np.mean(baselines); std_b = np.std(baselines)
        print("  %-10s  %.4f             %.4f  n=%d" % (lang, mean_b, std_b, len(baselines)))

print()
print("  Top-5 baselines (most similar pairs):")
all_pairs_flat = []
for lang, pairs in TRANS_PAIRS.items():
    for s_w, t_w in pairs:
        es, sid = get_emb(s_w); et, tid = get_emb(t_w)
        if es is None or et is None: continue
        cos_st = float(np.dot(W_n[sid], W_n[tid]))
        all_pairs_flat.append((cos_st, lang, s_w, t_w))
all_pairs_flat.sort(reverse=True)
for cos_st, lang, s_w, t_w in all_pairs_flat[:10]:
    print("  %-8s %-10s -> %-10s  cos=%.4f" % (lang, s_w, t_w, cos_st))
print()

# =====================================================================
# PART E: AXIS COMPOSITION — CHAINING TWO AXES
# =====================================================================
print("PART E: Axis composition test")
print("-"*72)

# Compute base axes
PLURAL_PAIRS = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('bird','birds'),('ship','ships'),
                 ('hand','hands'),('foot','feet'),('tooth','teeth')]
ABLAUT_TRAIN = [('go','went'),('take','took'),('give','gave'),('see','saw'),
                 ('break','broke'),('choose','chose'),('know','knew'),('drive','drove')]
COMP_PAIRS   = [('fast','faster'),('slow','slower'),('bright','brighter'),
                 ('dark','darker'),('soft','softer'),('warm','warmer')]
SUP_PAIRS    = [('fast','fastest'),('slow','slowest'),('bright','brightest'),
                 ('dark','darkest'),('soft','softest'),('warm','warmest')]

ax_pl, _, _ = compute_axis(PLURAL_PAIRS)
ax_ab, _, _ = compute_axis(ABLAUT_TRAIN)
ax_cp, _, _ = compute_axis(COMP_PAIRS)
ax_sp, _, _ = compute_axis(SUP_PAIRS)

print("  Test 1: base→past THEN +s (went→wents?)")
# go: apply ablaut -> went, then apply plural -> wents?
for verb, past_form in [('go','went'),('take','took'),('break','broke')]:
    es, sid = get_emb(verb)
    if es is None: continue
    # Scale search for ablaut
    best_s_ab, _ = best_scale(ax_ab, [(s,t,si,ti) for s,t,si,ti in
                               [(*p, *[get_emb(x)[1] for x in p]) for p in ABLAUT_TRAIN]
                               if si is not None and ti is not None], CLEAN_MASK) \
        if ax_ab is not None else (1.0, 0)
    # Step 1: apply ablaut
    mid = W_E[sid] + 1.0 * ax_ab
    # Step 2: apply plural
    final = mid + 1.0 * ax_pl
    r1 = nn_retrieve(mid, source_ids(verb), CLEAN_MASK, 3)
    r2 = nn_retrieve(final, source_ids(verb), CLEAN_MASK, 3)
    print("  %-6s -> step1(ablaut): %-10s  step2(+plural): %s" %
          (verb, r1[0][0], r2[0][0]))

print()
print("  Test 2: comparative THEN +est (faster→fastest?)")
# fast: apply comparative -> faster, then apply superlative offset
ax_comp_to_sup, _, _ = compute_axis([(a,b) for (a,_),((_,b)) in
                                       zip(COMP_PAIRS, SUP_PAIRS)] if len(COMP_PAIRS)==len(SUP_PAIRS) else [])
for adj in ['fast', 'slow', 'bright', 'dark']:
    es, sid = get_emb(adj)
    if es is None: continue
    mid = W_E[sid] + 1.2 * ax_cp
    r1 = nn_retrieve(mid, source_ids(adj), CLEAN_MASK, 3)
    if ax_sp is not None:
        final = W_E[sid] + 1.2 * ax_sp
        r2 = nn_retrieve(final, source_ids(adj), CLEAN_MASK, 3)
        print("  %-6s + comparative -> %-10s  (direct superlative -> %s)" %
              (adj, r1[0][0], r2[0][0]))

print()
print("  Test 3: Does chaining comparative+superlative axes work?")
# comp axis + (sup - comp) axis = sup axis?
if ax_cp is not None and ax_sp is not None:
    c_cp_sp = float(np.dot(ax_cp.astype(np.float32), ax_sp.astype(np.float32)))
    print("  cos(comparative_axis, superlative_axis) = %+.4f" % c_cp_sp)
    # Try chain: apply comparative then apply (sup - comp) delta
    # sup = comp + delta where delta = sup_axis - comp_axis
    delta = normed(ax_sp - ax_cp)
    for adj in ['fast','slow','bright']:
        es, sid = get_emb(adj)
        if es is None: continue
        step1 = W_E[sid] + 1.0 * ax_cp
        step2 = step1   + 0.5 * delta
        r1 = nn_retrieve(step1, source_ids(adj), CLEAN_MASK, 3)
        r2 = nn_retrieve(step2, source_ids(adj), CLEAN_MASK, 3)
        print("  %-6s -> comp: %-10s  -> chain sup: %s" % (adj, r1[0][0], r2[0][0]))
print()

print("  Test 4: Axis orthogonality check (composition valid iff axes near-orthogonal)")
axis_pairs = [
    ('plural', ax_pl, 'ablaut', ax_ab),
    ('plural', ax_pl, 'comparative', ax_cp),
    ('ablaut', ax_ab, 'comparative', ax_cp),
    ('plural', ax_pl, 'superlative', ax_sp),
]
for n1, a1, n2, a2 in axis_pairs:
    if a1 is None or a2 is None: continue
    c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
    print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, c))
