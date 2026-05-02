import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building masks...", flush=True)
CLEAN_MASK   = np.zeros(len(W_E), dtype=bool)
RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
CJK_MASK     = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) <= 1: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
    if not w[0].isupper(): CLEAN_MASK[i] = True
    if any('\u4e00' <= c <= '\u9fff' for c in w): CJK_MASK[i] = True
print("  clean=%d  relaxed=%d  cjk=%d" % (CLEAN_MASK.sum(), RELAXED_MASK.sum(), CJK_MASK.sum()))

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
    if pc > 0.35:    return 'morph_uniform/relational_geom'
    elif pc > 0.20:
        if loo > 0.50:     return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
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
        if irred >= 0.85 and loo < 0.15:  return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15: return 'borderline'
        else: return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

print()
print("DAY 328: 20-PAIR PROBE, GROUP D, THIRD-PERSON -S, ADAPTIVE CHAIN, CHINESE VOID")
print("="*80)
print()

# =====================================================================
# PART A: 20-PAIR PROBE (10 train + 10 holdout)
# =====================================================================
print("PART A: 20-pair probe (10 train + 10 holdout)")
print("-"*80)

TWENTY_PAIR_TESTS = [
    ('morph_uniform',
     [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),
      ('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder'),
      ('dark','darker'),('soft','softer')],
     [('small','smaller'),('hard','harder'),('young','younger'),('loud','louder'),
      ('wide','wider'),('light','lighter'),('deep','deeper'),('flat','flatter'),
      ('thin','thinner'),('thick','thicker')]),
    ('morph_moderate',
     [('cat','cats'),('dog','dogs'),('house','houses'),('bird','birds'),('book','books'),
      ('tree','trees'),('car','cars'),('ship','ships'),('door','doors'),('hand','hands')],
     [('arm','arms'),('foot','feet'),('child','children'),('man','men'),('chair','chairs'),
      ('glass','glasses'),('box','boxes'),('bus','buses'),('class','classes'),('dress','dresses')]),
    ('phonol_scatter',
     [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),
      ('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness'),
      ('warm','warmth'),('bright','brightness')],
     [('clean','cleanliness'),('sweet','sweetness'),('rich','richness'),('thick','thickness'),
      ('smooth','smoothness'),('fresh','freshness'),('sharp','sharpness'),('calm','calmness'),
      ('fit','fitness'),('odd','oddness')]),
    ('semantic_diverse',
     [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),
      ('own','owner'),('manage','manager'),('build','builder'),('lead','leader'),
      ('paint','painter'),('write','writer')],
     [('read','reader'),('hunt','hunter'),('mine','miner'),('dive','diver'),
      ('climb','climber'),('design','designer'),('plan','planner'),('help','helper'),
      ('print','printer'),('run','runner')]),
    ('factual_local',
     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),
      ('hand','手'),('eye','眼'),('fish','鱼'),('heart','心'),('tree','木')],
     [('sea','海'),('sky','天'),('man','男'),('woman','女'),('child','子'),
      ('earth','土'),('gold','金'),('wood','木'),('rain','雨'),('stone','石')]),
    ('polar_local',
     [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('high','low'),
      ('hard','soft'),('bright','dark'),('strong','weak'),('rich','poor'),('old','young')],
     [('light','heavy'),('long','short'),('wide','narrow'),('loud','quiet'),
      ('happy','sad'),('clean','dirty'),('near','far'),('full','empty'),
      ('early','late'),('true','false')]),
    ('relational_geom',
     [('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),
      ('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia'),
      ('Cairo','Egypt'),('Athens','Greece')],
     [('Vienna','Austria'),('Warsaw','Poland'),('Oslo','Norway'),('Dublin','Ireland'),
      ('Lisbon','Portugal'),('Budapest','Hungary'),('Bucharest','Romania'),
      ('Helsinki','Finland'),('Stockholm','Sweden'),('Copenhagen','Denmark')]),
    ('translation',
     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día'),
      ('night','noche'),('hand','mano'),('year','año'),('fire','fuego'),('sea','mar')],
     [('bread','pan'),('salt','sal'),('door','puerta'),('tree','árbol'),('road','camino'),
      ('milk','leche'),('rain','lluvia'),('wind','viento'),('snow','nieve'),('fish','pez')]),
]

print("  %-18s  pc      LOO%%  irred%%  n  -> pred                      ok?" % "true_type")
print("  " + "-"*76)
correct_20 = 0
for true_type, train_pairs, holdout_pairs in TWENTY_PAIR_TESTS:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2:
        print("  %-18s  n/a" % true_type); continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr_f, n_ho, _ = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    pred = classify_v5(pc, loo_v, irr_f)
    match = (true_type.split('_')[0] in pred or true_type in pred or
             ('morph' in pred and 'morph' in true_type) or
             ('phonol' in pred and 'phonol' in true_type) or
             ('relational' in pred and 'relational' in true_type) or
             ('factual' in pred and 'factual' in true_type) or
             ('translation' in pred and 'translation' in true_type) or
             ('polar' in pred and 'polar' in true_type) or
             ('semantic' in pred and 'semantic' in true_type))
    if match: correct_20 += 1
    tick = '✓' if match else '✗'
    print("  %s %-18s  pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d  -> %-26s" %
          (tick, true_type, pc, 100*loo_v, 100*irr_f, n_ho, pred))
print()
print("  20-pair accuracy: %d/%d = %.0f%%" % (correct_20, len(TWENTY_PAIR_TESTS), 100*correct_20/len(TWENTY_PAIR_TESTS)))
print()

# =====================================================================
# PART B: GROUP D TRIANGLE AND CROSS-GROUP COSINES
# =====================================================================
print("PART B: GROUP D internal triangle + cross-group cosines")
print("-"*80)

LESS_PAIRS  = [('hope','hopeless'),('fear','fearless'),('care','careless'),
                ('pain','painless'),('end','endless'),('home','homeless'),
                ('harm','harmless'),('power','powerless'),('worth','worthless')]
FUL_PAIRS   = [('hope','hopeful'),('care','careful'),('fear','fearful'),
                ('use','useful'),('grace','graceful'),('help','helpful'),
                ('faith','faithful'),('joy','joyful'),('peace','peaceful')]
ABLE_PAIRS  = [('read','readable'),('wash','washable'),('break','breakable'),
                ('love','lovable'),('use','usable'),('accept','acceptable'),
                ('avoid','avoidable'),('change','changeable'),('pass','passable')]
ANCE_PAIRS  = [('perform','performance'),('exist','existence'),('enter','entrance'),
                ('resist','resistance'),('accept','acceptance'),('appear','appearance'),
                ('depend','dependence'),('insist','insistence')]
MENT_PAIRS  = [('achieve','achievement'),('develop','development'),('manage','management'),
                ('govern','government'),('engage','engagement'),('require','requirement'),
                ('move','movement'),('improve','improvement')]
NESS_PAIRS  = [('happy','happiness'),('kind','kindness'),('sad','sadness'),
                ('bright','brightness'),('dark','darkness'),('soft','softness'),
                ('fair','fairness'),('clear','clearness'),('weak','weakness')]
EDLY_PAIRS  = [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                ('play','played'),('clean','cleaned'),('open','opened'),('start','started')]
ABLAUT_PAIRS= [('go','went'),('take','took'),('give','gave'),('see','saw'),
                ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')]

ax_less, _, pc_less  = compute_axis(LESS_PAIRS)
ax_ful,  _, pc_ful   = compute_axis(FUL_PAIRS)
ax_able, _, pc_able  = compute_axis(ABLE_PAIRS)
ax_ance, _, pc_ance  = compute_axis(ANCE_PAIRS)
ax_ment, _, pc_ment  = compute_axis(MENT_PAIRS)
ax_ness, _, pc_ness  = compute_axis(NESS_PAIRS)
ax_ed,   _, pc_ed    = compute_axis(EDLY_PAIRS)
ax_ab,   _, pc_ab    = compute_axis(ABLAUT_PAIRS)

GROUP_D_AXES = [('less', ax_less, pc_less), ('ful', ax_ful, pc_ful), ('able', ax_able, pc_able)]
GROUP_A_AXES = [('ance', ax_ance, pc_ance), ('ment', ax_ment, pc_ment)]
OTHER_AXES   = [('ness', ax_ness, pc_ness), ('ed', ax_ed, pc_ed), ('ablaut', ax_ab, pc_ab)]

print("  Group D internal cosines (full triangle):")
for i, (n1, a1, _) in enumerate(GROUP_D_AXES):
    for n2, a2, _ in GROUP_D_AXES[i+1:]:
        if a1 is None or a2 is None: continue
        c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
        print("  cos(+%-6s, +%-6s) = %+.4f" % (n1, n2, c))
print()
print("  Group D vs Group A (cross-group):")
for n1, a1, _ in GROUP_D_AXES:
    for n2, a2, _ in GROUP_A_AXES:
        if a1 is None or a2 is None: continue
        c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
        print("  cos(+%-6s, +%-6s) = %+.4f" % (n1, n2, c))
print()
print("  Group D vs others:")
for n1, a1, _ in GROUP_D_AXES:
    for n2, a2, _ in OTHER_AXES:
        if a1 is None or a2 is None: continue
        c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
        print("  cos(+%-6s, +%-6s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART C: THIRD-PERSON -S AXIS
# =====================================================================
print("PART C: Third-person -s inflection axis")
print("-"*80)

THIRD_PS = [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
             ('read','reads'),('write','writes'),('play','plays'),('work','works'),
             ('talk','talks'),('drive','drives'),('sleep','sleeps'),('stand','stands'),
             ('think','thinks'),('know','knows'),('say','says'),('see','sees')]

ax_3ps, valid_3ps, pc_3ps = compute_axis(THIRD_PS)
if ax_3ps is not None:
    loo_3ps = axis_loo(ax_3ps, valid_3ps, CLEAN_MASK)
    irr_3ps, _, _ = irred_on_holdout(ax_3ps,
        [('feel','feels'),('hold','holds'),('tell','tells'),('hear','hears')], CLEAN_MASK)
    print("  +3ps axis: pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_3ps, 100*loo_3ps, 100*irr_3ps, len(valid_3ps)))
    print("  Predicted type: %s" % classify_v5(pc_3ps, loo_3ps, irr_3ps))
    print()
    if ax_ed is not None:
        c1 = float(np.dot(ax_3ps.astype(np.float32), ax_ed.astype(np.float32)))
        print("  cos(+3ps, +ed_reg) = %+.4f" % c1)
    if ax_ab is not None:
        c2 = float(np.dot(ax_3ps.astype(np.float32), ax_ab.astype(np.float32)))
        print("  cos(+3ps, ablaut)  = %+.4f" % c2)

    # Also +s_plural for comparison
    PLURAL_PAIRS = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                     ('tree','trees'),('book','books'),('bird','birds'),('door','doors')]
    ax_pl, valid_pl, pc_pl = compute_axis(PLURAL_PAIRS)
    if ax_pl is not None:
        c3 = float(np.dot(ax_3ps.astype(np.float32), ax_pl.astype(np.float32)))
        print("  cos(+3ps, +s_plural) = %+.4f  (how similar to noun plural?)" % c3)
    print()

# =====================================================================
# PART D: ADAPTIVE SCALE CHAIN
# =====================================================================
print("PART D: Scale-adaptive three-step chain")
print("-"*80)

def local_scale(axis, emb, mask, n_neighbors=5, lo=0.01, hi=4.0, n_s=30):
    emb_n = normed(emb).astype(np.float32)
    sims = W_n @ emb_n
    sims[~mask] = -1.0
    top_ids = np.argpartition(sims, -n_neighbors)[-n_neighbors:]
    top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
    scales = []
    for idx in top_ids:
        word = tok.decode([idx]).strip()
        best_s_loc, _ = best_scale(axis, [(word, word, idx, idx)], mask, lo=lo, hi=hi, n=n_s)
        scales.append(best_s_loc)
    return float(np.median(scales)) if scales else 1.0

# Build all needed axes
AL_REL_PAIRS = [('nation','national'),('region','regional'),('culture','cultural'),
                 ('nature','natural'),('person','personal'),('origin','original'),
                 ('emotion','emotional'),('tradition','traditional')]
ITY_PAIRS    = [('human','humanity'),('real','reality'),('national','nationality'),
                 ('personal','personality'),('moral','morality'),('legal','legality'),
                 ('local','locality'),('normal','normality')]
PLURAL_FULL  = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('nation','nations'),
                 ('person','persons'),('year','years'),('city','cities')]

ax_alr, valid_alr, _ = compute_axis(AL_REL_PAIRS)
ax_ity, valid_ity, _ = compute_axis(ITY_PAIRS)
ax_pl2, valid_pl2, _ = compute_axis(PLURAL_FULL)

if ax_alr is not None and ax_ity is not None and ax_pl2 is not None:
    bs_alr, _ = best_scale(ax_alr, valid_alr, CLEAN_MASK)
    bs_ity, _ = best_scale(ax_ity, valid_ity, CLEAN_MASK)
    bs_pl2, _ = best_scale(ax_pl2, valid_pl2, CLEAN_MASK)
    print("  Global scales: +al_rel=%.2f  +ity=%.2f  +plural=%.2f" % (bs_alr, bs_ity, bs_pl2))
    print()

    print("  Standard three-step chain:")
    for source, adj_f, ity_f, pl_f in [
        ('nation','national','nationality','nationalities'),
        ('person','personal','personality','personalities'),
        ('equal','equal','equality','equalities'),
    ]:
        es, sid = get_emb(source)
        if es is None: continue
        mag = np.linalg.norm(W_E[sid])
        s1 = W_E[sid] + bs_alr * ax_alr
        r1 = nn_retrieve(s1, source_ids(source), CLEAN_MASK, 2)
        s2 = normed(s1)*mag + bs_ity * ax_ity
        r2 = nn_retrieve(s2, source_ids(source), CLEAN_MASK, 2)
        s3 = normed(s2)*mag + bs_pl2 * ax_pl2
        r3 = nn_retrieve(s3, source_ids(source), CLEAN_MASK, 2)
        t1 = '✓' if r1[0][0]==adj_f else '~'
        t2 = '✓' if r2[0][0]==ity_f else '~'
        t3 = '✓' if r3[0][0]==pl_f else '~'
        print("  %s %-10s->%-14s  %s ->%-14s  %s ->%s" %
              (t1, source, r1[0][0], t2, r2[0][0], t3, r3[0][0]))
    print()

    print("  Adaptive-scale three-step chain (re-calibrate scale at each step):")
    for source, adj_f, ity_f, pl_f in [
        ('nation','national','nationality','nationalities'),
        ('person','personal','personality','personalities'),
        ('equal','equal','equality','equalities'),
        ('morality','moral','morality','moralities'),
    ]:
        es, sid = get_emb(source)
        if es is None: continue
        mag = np.linalg.norm(W_E[sid])

        # Step 1 with global scale (good)
        s1 = W_E[sid] + bs_alr * ax_alr
        r1 = nn_retrieve(s1, source_ids(source), CLEAN_MASK, 2)

        # Step 2: find intermediate token, get its embedding, use it as new base
        mid1_word = r1[0][0]
        es2, sid2 = get_emb(mid1_word)
        if es2 is None: es2 = s1; sid2 = None
        s2 = es2 + bs_ity * ax_ity
        r2 = nn_retrieve(s2, source_ids(source) | (source_ids(mid1_word) if sid2 else set()), CLEAN_MASK, 2)

        # Step 3: find intermediate token from step2, calibrate locally
        mid2_word = r2[0][0]
        es3, sid3 = get_emb(mid2_word)
        if es3 is None: es3 = s2
        # Local scale search from mid2_word position
        loc_s = 1.0
        if sid3 is not None:
            bs_loc, _ = best_scale(ax_pl2, [(mid2_word, pl_f, sid3, 0)], CLEAN_MASK, lo=0.01, hi=5.0)
            loc_s = bs_loc
        s3 = es3 + loc_s * ax_pl2
        r3 = nn_retrieve(s3, source_ids(source) | source_ids(mid2_word), CLEAN_MASK, 3)

        t1 = '✓' if r1[0][0]==adj_f else '~'
        t2 = '✓' if r2[0][0]==ity_f else '~'
        t3 = '✓' if r3[0][0]==pl_f else '~'
        print("  %s %-10s->%-14s  %s ->%-14s  %s ->%-14s  (pl_s=%.2f)" %
              (t1, source, r1[0][0], t2, r2[0][0], t3, r3[0][0], loc_s))
print()

# =====================================================================
# PART E: CHINESE VOID MAPPING
# =====================================================================
print("PART E: Chinese void mapping — which operations land in Chinese OOD?")
print("-"*80)

ALL_AXIS_DEFS = {
    '+al_rel': AL_REL_PAIRS,
    '+ity':    ITY_PAIRS,
    '+ness':   NESS_PAIRS,
    '+less':   LESS_PAIRS,
    '+ful':    FUL_PAIRS,
    '+able':   ABLE_PAIRS,
    '+ance':   ANCE_PAIRS,
    '+ment':   MENT_PAIRS,
    '+ed_reg': EDLY_PAIRS,
    'ablaut':  ABLAUT_PAIRS,
}

# Test each axis on 10 cross-category words (guaranteed OOD)
CROSS_CATEGORY_WORDS = [
    'table', 'window', 'garden', 'computer', 'music',
    'science', 'market', 'forest', 'bridge', 'ocean',
]

print("  For each axis, count how many OOD words navigate to Chinese tokens:")
print("  %-12s  cjk_fraction  example_cjk_result" % "axis")
print("  " + "-"*60)
for name, pairs in ALL_AXIS_DEFS.items():
    ax, vl, _ = compute_axis(pairs)
    if ax is None: continue
    bs, _ = best_scale(ax, vl, CLEAN_MASK)
    cjk_count = 0
    cjk_example = None
    for word in CROSS_CATEGORY_WORDS:
        es, sid = get_emb(word)
        if es is None: continue
        r = nn_retrieve(es + bs * ax, source_ids(word), RELAXED_MASK, 3)
        top_word = r[0][0]
        if any('\u4e00' <= c <= '\u9fff' for c in top_word):
            cjk_count += 1
            if cjk_example is None: cjk_example = (word, top_word)
    frac = cjk_count / len(CROSS_CATEGORY_WORDS)
    ex_str = '(%s->%s)' % (cjk_example[0], cjk_example[1]) if cjk_example else ''
    print("  %-12s  %.1f%%         %s" % (name, 100*frac, ex_str))
print()
print("  Interpretation: high CJK% = axis points to empty region for OOD words")
print("  (Abstract suffix axes point toward abstract regions populated by CJK)")
print()

# Show top-20 CJK tokens nearest to each axis's 'endpoint' (centroid of target embeddings)
print("  Top CJK tokens near each axis target centroid:")
for name, pairs in list(ALL_AXIS_DEFS.items())[:6]:
    ax, vl, _ = compute_axis(pairs)
    if ax is None or len(vl) < 2: continue
    # centroid of target embeddings
    target_embs = [W_E[tid] for _,_,_,tid in vl]
    centroid = normed(np.mean(target_embs, axis=0)).astype(np.float32)
    sims = W_n @ centroid
    sims[~CJK_MASK] = -1.0
    top_cjk = np.argpartition(sims, -8)[-8:]
    top_cjk = top_cjk[np.argsort(sims[top_cjk])[::-1]]
    cjk_words = [tok.decode([i]).strip() for i in top_cjk]
    print("  %-12s  -> %s" % (name, ' '.join(cjk_words[:6])))
