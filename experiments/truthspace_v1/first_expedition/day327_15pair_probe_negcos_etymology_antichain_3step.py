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
print("DAY 327: 15-PAIR PROBE, NEGATIVE COSINES, ETYMOLOGY SPLIT, ANTI-CHAIN, 3-STEP CHAIN")
print("="*80)
print()

# =====================================================================
# PART A: 15-PAIR PROBE TEST
# =====================================================================
print("PART A: 15-pair probe (5 train + 10 holdout)")
print("-"*80)

FIFTEEN_PAIR_TESTS = [
    ('morph_uniform',
     [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter')],
     [('warm','warmer'),('long','longer'),('cold','colder'),('dark','darker'),('soft','softer'),
      ('small','smaller'),('hard','harder'),('young','younger'),('loud','louder'),('wide','wider')]),
    ('morph_moderate',
     [('cat','cats'),('dog','dogs'),('house','houses'),('bird','birds'),('book','books')],
     [('tree','trees'),('car','cars'),('ship','ships'),('door','doors'),('hand','hands'),
      ('arm','arms'),('foot','feet'),('child','children'),('man','men'),('woman','women')]),
    ('phonol_scatter',
     [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness')],
     [('weak','weakness'),('good','goodness'),('hard','hardness'),('warm','warmth'),
      ('bright','brightness'),('clean','cleanliness'),('sweet','sweetness'),('rich','richness'),
      ('thick','thickness'),('smooth','smoothness')]),
    ('semantic_diverse',
     [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner')],
     [('manage','manager'),('build','builder'),('lead','leader'),('paint','painter'),
      ('write','writer'),('read','reader'),('hunt','hunter'),('mine','miner'),
      ('dive','diver'),('climb','climber')]),
    ('factual_local',
     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山')],
     [('hand','手'),('eye','眼'),('fish','鱼'),('heart','心'),('tree','木'),
      ('sea','海'),('sky','天'),('man','男'),('woman','女'),('child','子')]),
    ('polar_local',
     [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('high','low')],
     [('hard','soft'),('bright','dark'),('strong','weak'),('rich','poor'),('old','young'),
      ('light','heavy'),('long','short'),('wide','narrow'),('loud','quiet'),('happy','sad')]),
    ('relational_geom',
     [('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany')],
     [('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia'),('Cairo','Egypt'),
      ('Athens','Greece'),('Vienna','Austria'),('Warsaw','Poland'),('Oslo','Norway'),
      ('Dublin','Ireland'),('Lisbon','Portugal')]),
    ('translation',
     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día')],
     [('night','noche'),('hand','mano'),('year','año'),('fire','fuego'),('sea','mar'),
      ('bread','pan'),('salt','sal'),('door','puerta'),('tree','árbol'),('road','camino')]),
]

print("  %-18s  pc      LOO%%  irred%%  n_ho  -> pred                      ok?" % "true_type")
print("  " + "-"*80)
correct_15 = 0
for true_type, train_pairs, holdout_pairs in FIFTEEN_PAIR_TESTS:
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
    if match: correct_15 += 1
    tick = '✓' if match else '✗'
    print("  %s %-18s  pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d  -> %-26s" %
          (tick, true_type, pc, 100*loo_v, 100*irr_f, n_ho, pred))
print()
print("  15-pair accuracy: %d/%d = %.0f%%" % (correct_15, len(FIFTEEN_PAIR_TESTS), 100*correct_15/len(FIFTEEN_PAIR_TESTS)))
print()

# =====================================================================
# PART B: FULL NEGATIVE COSINE SURVEY
# =====================================================================
print("PART B: Full negative cosine survey across all suffix/semantic axes")
print("-"*80)

ALL_AXIS_DEFS = {
    '+al_rel':   [('nation','national'),('region','regional'),('culture','cultural'),
                   ('nature','natural'),('person','personal'),('origin','original'),
                   ('emotion','emotional'),('tradition','traditional')],
    '+al_nom':   [('arrive','arrival'),('propose','proposal'),('approve','approval'),
                   ('refuse','refusal'),('remove','removal'),('survive','survival')],
    '+ance':     [('perform','performance'),('exist','existence'),('enter','entrance'),
                   ('resist','resistance'),('accept','acceptance'),('insist','insistence'),
                   ('appear','appearance'),('depend','dependence')],
    '+ment':     [('achieve','achievement'),('develop','development'),('manage','management'),
                   ('govern','government'),('engage','engagement'),('require','requirement')],
    '+tion':     [('act','action'),('direct','direction'),('educate','education'),
                   ('create','creation'),('produce','production'),('relate','relation')],
    '+ity':      [('human','humanity'),('real','reality'),('final','finality'),
                   ('moral','morality'),('normal','normality'),('national','nationality'),
                   ('personal','personality'),('legal','legality'),('local','locality')],
    '+ness':     [('happy','happiness'),('kind','kindness'),('sad','sadness'),
                   ('bright','brightness'),('dark','darkness'),('soft','softness'),
                   ('fair','fairness'),('clear','clearness'),('weak','weakness')],
    '+er_noun':  [('teach','teacher'),('farm','farmer'),('drive','driver'),
                   ('work','worker'),('own','owner'),('lead','leader')],
    '+er_comp':  [('fast','faster'),('slow','slower'),('bright','brighter'),
                   ('dark','darker'),('soft','softer'),('warm','warmer')],
    '+s_plural': [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                   ('tree','trees'),('book','books'),('bird','birds')],
    '+ed_reg':   [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                   ('play','played'),('clean','cleaned'),('open','opened')],
    '+ing':      [('go','going'),('take','taking'),('run','running'),('see','seeing'),
                   ('give','giving'),('make','making'),('write','writing'),('read','reading')],
    'ablaut':    [('go','went'),('take','took'),('give','gave'),('see','saw'),
                   ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],
    '+able':     [('read','readable'),('wash','washable'),('break','breakable'),
                   ('love','lovable'),('use','usable'),('accept','acceptable')],
    '+less':     [('hope','hopeless'),('fear','fearless'),('care','careless'),
                   ('pain','painless'),('end','endless'),('home','homeless')],
    '+ful':      [('hope','hopeful'),('care','careful'),('fear','fearful'),
                   ('use','useful'),('grace','graceful'),('help','helpful')],
    'un-':       [('happy','unhappy'),('clear','unclear'),('fair','unfair'),
                   ('likely','unlikely'),('known','unknown'),('safe','unsafe')],
    'adj_ant':   [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),
                   ('bright','dark'),('hard','soft'),('high','low')],
    'cc':        [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),
                   ('car','Car'),('tree','Tree'),('river','River')],
}

print("  Building all axes...", flush=True)
all_axes = {}
for name, pairs in ALL_AXIS_DEFS.items():
    ax, _, pc = compute_axis(pairs)
    if ax is not None: all_axes[name] = (ax, pc)
print("  Built %d axes" % len(all_axes))
print()

axis_names = list(all_axes.keys())
pairs_neg = []
for i, n1 in enumerate(axis_names):
    for n2 in axis_names[i+1:]:
        c = float(np.dot(all_axes[n1][0].astype(np.float32),
                         all_axes[n2][0].astype(np.float32)))
        if c < -0.10: pairs_neg.append((c, n1, n2))

pairs_neg.sort()
print("  All negative cosine pairs (cos < -0.10):")
if pairs_neg:
    for c, n1, n2 in pairs_neg:
        print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, c))
else:
    print("  None found below -0.10")
print()
print("  Top-10 most positive cosine pairs:")
pairs_pos = []
for i, n1 in enumerate(axis_names):
    for n2 in axis_names[i+1:]:
        c = float(np.dot(all_axes[n1][0].astype(np.float32),
                         all_axes[n2][0].astype(np.float32)))
        pairs_pos.append((c, n1, n2))
pairs_pos.sort(reverse=True)
for c, n1, n2 in pairs_pos[:10]:
    print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART C: +ness vs +ity ETYMOLOGY SPLIT
# =====================================================================
print("PART C: +ness vs +ity etymology split (Latin vs Germanic adjectives)")
print("-"*80)

# Germanic-origin adjectives (Old English roots, typically take +ness)
GERMANIC_ADJ = [
    ('dark','darkness','darkity'),    ('soft','softness','softity'),
    ('hard','hardness','hardity'),    ('cold','coldness','coldity'),
    ('warm','warmness','warmity'),    ('bright','brightness','brightity'),
    ('weak','weakness','weakity'),    ('sweet','sweetness','sweetity'),
    ('sad','sadness','sadity'),       ('glad','gladness','gladity'),
    ('sick','sickness','sickity'),    ('bold','boldness','boldity'),
    ('old','oldness','oldity'),       ('kind','kindness','kindity'),
    ('mild','mildness','mildity'),
]
# Latin-origin adjectives (typically take +ity or have both forms)
LATIN_ADJ = [
    ('moral','morality','moralness'),   ('legal','legality','legalness'),
    ('local','locality','localness'),   ('real','reality','realness'),
    ('final','finality','finalness'),   ('equal','equality','equalness'),
    ('noble','nobility','nobleness'),   ('civil','civility','civilness'),
    ('agile','agility','agileness'),    ('fertile','fertility','fertileness'),
    ('humble','humility','humbleness'), ('mobile','mobility','mobileness'),
    ('normal','normality','normalness'),('stable','stability','stableness'),
    ('vital','vitality','vitalness'),
]

if all_axes.get('+ity') and all_axes.get('+ness'):
    ax_ity = all_axes['+ity'][0]
    ax_ness = all_axes['+ness'][0]
    # Quick scale estimate
    bs_ity = 1.0; bs_ness = 1.0
    _ax_ity2, _vl_ity, _ = compute_axis([('real','reality'),('moral','morality'),
                                          ('human','humanity'),('legal','legality')])
    _ax_ness2, _vl_ness, _ = compute_axis([('happy','happiness'),('kind','kindness'),
                                             ('sad','sadness'),('dark','darkness')])
    if _ax_ity2 is not None and len(_vl_ity) > 0:
        bs_ity, _ = best_scale(ax_ity, _vl_ity, CLEAN_MASK, lo=0.1, hi=3.0)
    if _ax_ness2 is not None and len(_vl_ness) > 0:
        bs_ness, _ = best_scale(ax_ness, _vl_ness, CLEAN_MASK, lo=0.1, hi=3.0)

    print("  Scale: +ity=%.2f  +ness=%.2f" % (bs_ity, bs_ness))
    print()
    print("  Germanic adjectives — which axis wins?")
    print("  %-10s  +ity->          +ness->         winner" % "adj")
    for adj, ness_form, _ in GERMANIC_ADJ:
        es, sid = get_emb(adj)
        if es is None: continue
        r_ity  = nn_retrieve(W_E[sid]+bs_ity*ax_ity, source_ids(adj), CLEAN_MASK, 3)
        r_ness = nn_retrieve(W_E[sid]+bs_ness*ax_ness, source_ids(adj), CLEAN_MASK, 3)
        ity_w = r_ity[0][0]; ness_w = r_ness[0][0]
        ity_ok  = (ity_w == ness_form)
        ness_ok = (ness_w == ness_form)
        winner = '+ness' if ness_ok and not ity_ok else ('+ity' if ity_ok and not ness_ok else ('both' if ness_ok and ity_ok else 'neither'))
        print("  %-10s  %-16s %-16s %s" % (adj, ity_w, ness_w, winner))
    print()
    print("  Latin adjectives — which axis wins?")
    print("  %-10s  +ity->          +ness->         winner" % "adj")
    for adj, ity_form, _ in LATIN_ADJ:
        es, sid = get_emb(adj)
        if es is None: continue
        r_ity  = nn_retrieve(W_E[sid]+bs_ity*ax_ity, source_ids(adj), CLEAN_MASK, 3)
        r_ness = nn_retrieve(W_E[sid]+bs_ness*ax_ness, source_ids(adj), CLEAN_MASK, 3)
        ity_w = r_ity[0][0]; ness_w = r_ness[0][0]
        ity_ok  = (ity_w == ity_form)
        ness_ok = (ness_w == ity_form)
        winner = '+ity' if ity_ok and not ness_ok else ('+ness' if ness_ok and not ity_ok else ('both' if ness_ok and ity_ok else 'neither'))
        print("  %-10s  %-16s %-16s %s" % (adj, ity_w, ness_w, winner))
print()

# =====================================================================
# PART D: ANTI-ALIGNED CHAIN (+al_rel + +ity extended)
# =====================================================================
print("PART D: Anti-aligned chain exploration (+al_rel → +ity in detail)")
print("-"*80)

if all_axes.get('+al_rel') and all_axes.get('+ity'):
    ax_alr = all_axes['+al_rel'][0]; ax_ity = all_axes['+ity'][0]
    # Try larger vocabulary
    test_nouns = [
        ('nation','national','nationality'),('person','personal','personality'),
        ('origin','original','originality'),('region','regional','regionality'),
        ('material','material','materiality'),('equal','equal','equality'),
        ('universe','universal','universality'),('spirit','spiritual','spirituality'),
        ('virgin','virginal','virginity'),('mortal','mortal','mortality'),
        ('brutal','brutal','brutality'),('neutral','neutral','neutrality'),
        ('royal','royal','royalty'),('loyal','loyal','loyalty'),
        ('fatal','fatal','fatality'),('final','final','finality'),
        ('vital','vital','vitality'),('local','local','locality'),
        ('vocal','vocal','vocality'),('total','total','totality'),
    ]
    _ax_alr2, _vl_alr, _ = compute_axis([('nation','national'),('person','personal')])
    bs_alr, _ = best_scale(ax_alr, _vl_alr, CLEAN_MASK)
    bs_ity_est = 0.84
    print("  Scales: +al_rel=%.2f  +ity=%.2f" % (bs_alr, bs_ity_est))
    chain_hits = 0; chain_total = 0
    for source, adj_form, final_form in test_nouns:
        es, sid = get_emb(source)
        if es is None: continue
        chain_total += 1
        step1 = W_E[sid] + bs_alr * ax_alr
        r1 = nn_retrieve(step1, source_ids(source), CLEAN_MASK, 3)
        # Scale-free step 2
        step1_sf = normed(step1) * np.linalg.norm(W_E[sid])
        step2 = step1_sf + bs_ity_est * ax_ity
        r2 = nn_retrieve(step2, source_ids(source), CLEAN_MASK, 3)
        # Direct from source with +ity
        direct = W_E[sid] + bs_ity_est * ax_ity
        r_dir = nn_retrieve(direct, source_ids(source), CLEAN_MASK, 3)
        t1 = '✓' if r1[0][0]==adj_form else '~'
        t2 = '✓' if r2[0][0]==final_form else '~'
        if t2 == '✓': chain_hits += 1
        print("  %s %-10s->%-14s  %s chain->%-14s  (direct: %s)" %
              (t1, source, r1[0][0], t2, r2[0][0], r_dir[0][0]))
    print()
    print("  Chain hits: %d/%d = %.0f%%" % (chain_hits, chain_total, 100*chain_hits/chain_total))
print()

# =====================================================================
# PART E: THREE-STEP CHAIN — nation → national → nationality → nationalities
# =====================================================================
print("PART E: Three-step chain test")
print("-"*80)

if all_axes.get('+al_rel') and all_axes.get('+ity') and all_axes.get('+s_plural'):
    ax_alr = all_axes['+al_rel'][0]
    ax_ity = all_axes['+ity'][0]
    ax_pl  = all_axes['+s_plural'][0]
    bs_pl, _ = best_scale(ax_pl, [v for v in []], CLEAN_MASK) if False else (1.0, 0)
    # estimate plural scale
    _ax_pl2, _vl2, _ = compute_axis([('cat','cats'),('dog','dogs'),('house','houses')])
    if _ax_pl2 is not None and _vl2:
        bs_pl, _ = best_scale(_ax_pl2, _vl2, CLEAN_MASK)
    print("  Scales: +al_rel=0.64  +ity=0.84  +plural=%.2f" % bs_pl)
    print()
    print("  Three-step (scale-free between each step):")
    for source, adj_form, ity_form, plural_form in [
        ('nation','national','nationality','nationalities'),
        ('person','personal','personality','personalities'),
        ('morality','moral','morality','moralities'),
        ('tradition','traditional','traditionality','traditionalities'),
    ]:
        es, sid = get_emb(source)
        if es is None: continue
        mag = np.linalg.norm(W_E[sid])
        # Step 1
        s1 = W_E[sid] + 0.64 * ax_alr
        r1 = nn_retrieve(s1, source_ids(source), CLEAN_MASK, 2)
        # Scale-free step 2
        s2 = normed(s1) * mag + 0.84 * ax_ity
        r2 = nn_retrieve(s2, source_ids(source), CLEAN_MASK, 2)
        # Scale-free step 3
        s3 = normed(s2) * mag + bs_pl * ax_pl
        r3 = nn_retrieve(s3, source_ids(source), CLEAN_MASK, 2)
        t1 = '✓' if r1[0][0]==adj_form else '~'
        t2 = '✓' if r2[0][0]==ity_form else '~'
        t3 = '✓' if r3[0][0]==plural_form else '~'
        print("  %s %-12s->%-12s  %s ->%-12s  %s ->%s" %
              (t1, source, r1[0][0], t2, r2[0][0], t3, r3[0][0]))
    print()

    # Also test: ablaut → +s_plural (cross-category chain, should fail)
    print("  Control: ablaut → +s_plural (cross-category, should fail):")
    if all_axes.get('ablaut'):
        ax_ab = all_axes['ablaut'][0]
        _ax_ab2, _vl_ab, _ = compute_axis([('go','went'),('take','took'),('give','gave')])
        bs_ab = 1.0
        if _ax_ab2 is not None and _vl_ab:
            bs_ab, _ = best_scale(_ax_ab2, _vl_ab, CLEAN_MASK)
        for verb, past_form in [('go','went'),('take','took'),('run','ran')]:
            es, sid = get_emb(verb)
            if es is None: continue
            s1 = W_E[sid] + bs_ab * ax_ab
            r1 = nn_retrieve(s1, source_ids(verb), CLEAN_MASK, 2)
            s2 = normed(s1) * np.linalg.norm(W_E[sid]) + bs_pl * ax_pl
            r2 = nn_retrieve(s2, source_ids(verb), CLEAN_MASK, 2)
            print("  %-6s -> %-8s -> %s (expected: %ss)" % (verb, r1[0][0], r2[0][0], past_form))
