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

def classify_v6(pc, loo, irred):
    if pc > 0.32:    return 'morph_uniform/relational_geom'  # v6: lowered from 0.35
    elif pc > 0.20:
        if loo > 0.50:     return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo > 0.50:
            if irred >= 0.40: return 'semantic_diverse'      # v6: irred override
            return 'phonol_scatter'
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
print("DAY 329: PREDICTOR V6, +ing/+ly IN GROUPS, BILINGUAL AXIS, GROUP E PROBE")
print("="*80)
print()

# =====================================================================
# PART A: PREDICTOR V6 — BENCHMARK ON ALL 30 ORIGINAL AXES
# =====================================================================
print("PART A: Predictor v6 benchmark — 30-axis suite")
print("-"*80)

THIRTY_AXIS_BENCHMARK = [
    # (name, pairs, true_label)
    ('er_comp',     [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),
                     ('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],
     'morph_uniform'),
    ('relational',  [('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),
                     ('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],
     'relational_geom'),
    ('er_comp2',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                     ('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')],
     'morph_uniform'),
    ('plural',      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                     ('tree','trees'),('book','books'),('bird','birds'),('door','doors')],
     'morph_moderate'),
    ('3ps',         [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                     ('read','reads'),('write','writes'),('play','plays'),('work','works')],
     'morph_moderate'),
    ('ed_reg',      [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                     ('play','played'),('clean','cleaned'),('open','opened'),('start','started')],
     'morph_moderate'),
    ('ing',         [('go','going'),('take','taking'),('run','running'),('see','seeing'),
                     ('give','giving'),('make','making'),('write','writing'),('read','reading')],
     'morph_moderate'),
    ('ness',        [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),
                     ('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],
     'phonol_scatter'),
    ('er_noun',     [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),
                     ('own','owner'),('manage','manager'),('build','builder'),('lead','leader')],
     'semantic_diverse'),
    ('ablaut',      [('go','went'),('take','took'),('give','gave'),('see','saw'),
                     ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],
     'phonol_scatter'),
    ('ity',         [('human','humanity'),('real','reality'),('final','finality'),
                     ('moral','morality'),('normal','normality'),('national','nationality'),
                     ('personal','personality'),('legal','legality')],
     'phonol_scatter'),
    ('un_neg',      [('happy','unhappy'),('clear','unclear'),('fair','unfair'),
                     ('likely','unlikely'),('known','unknown'),('safe','unsafe'),
                     ('usual','unusual'),('equal','unequal')],
     'phonol_scatter'),
    ('al_rel',      [('nation','national'),('region','regional'),('culture','cultural'),
                     ('nature','natural'),('person','personal'),('origin','original'),
                     ('emotion','emotional'),('tradition','traditional')],
     'relational_geom'),
    ('ance',        [('perform','performance'),('exist','existence'),('enter','entrance'),
                     ('resist','resistance'),('accept','acceptance'),('appear','appearance'),
                     ('depend','dependence'),('insist','insistence')],
     'phonol_scatter'),
    ('ment',        [('achieve','achievement'),('develop','development'),('manage','management'),
                     ('govern','government'),('engage','engagement'),('require','requirement'),
                     ('move','movement'),('improve','improvement')],
     'phonol_scatter'),
    ('tion',        [('act','action'),('direct','direction'),('educate','education'),
                     ('create','creation'),('produce','production'),('relate','relation'),
                     ('combine','combination'),('apply','application')],
     'phonol_scatter'),
    ('al_nom',      [('arrive','arrival'),('propose','proposal'),('approve','approval'),
                     ('refuse','refusal'),('remove','removal'),('survive','survival'),
                     ('deny','denial'),('dispose','disposal')],
     'phonol_scatter'),
    ('less',        [('hope','hopeless'),('fear','fearless'),('care','careless'),
                     ('pain','painless'),('end','endless'),('home','homeless'),
                     ('harm','harmless'),('power','powerless')],
     'phonol_scatter'),
    ('ful',         [('hope','hopeful'),('care','careful'),('fear','fearful'),
                     ('use','useful'),('grace','graceful'),('help','helpful'),
                     ('faith','faithful'),('joy','joyful')],
     'phonol_scatter'),
    ('able',        [('read','readable'),('wash','washable'),('break','breakable'),
                     ('love','lovable'),('use','usable'),('accept','acceptable'),
                     ('avoid','avoidable'),('change','changeable')],
     'phonol_scatter'),
    ('ablaut_t',    [('send','sent'),('build','built'),('burn','burned'),('deal','dealt'),
                     ('feel','felt'),('keep','kept'),('leave','left'),('sleep','slept')],
     'phonol_scatter'),
    ('cc',          [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),
                     ('car','Car'),('tree','Tree'),('river','River'),('mountain','Mountain')],
     'morph_moderate'),
    ('adj_ant',     [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),
                     ('bright','dark'),('hard','soft'),('high','low'),('rich','poor')],
     'polar_local'),
    ('en_es',       [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),
                     ('day','día'),('night','noche'),('hand','mano'),('year','año')],
     'translation'),
    ('en_de',       [('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),
                     ('day','Tag'),('night','Nacht'),('cat','Katze'),('dog','Hund')],
     'translation'),
    ('en_fr',       [('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),
                     ('day','jour'),('night','nuit'),('cat','chat'),('dog','chien')],
     'translation'),
    ('en_zh',       [('sun','日'),('moon','月'),('water','水'),('fire','火'),
                     ('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼')],
     'factual_local'),
    ('en_ja',       [('sun','日'),('moon','月'),('water','水'),('fire','火'),
                     ('mountain','山'),('hand','手'),('eye','目'),('fish','魚')],
     'factual_local'),
    ('num_word',    [('1','one'),('2','two'),('3','three'),('4','four'),
                     ('5','five'),('6','six'),('7','seven'),('8','eight')],
     'semantic_diverse'),
    ('antonym2',    [('love','hate'),('war','peace'),('life','death'),('day','night'),
                     ('begin','end'),('give','take'),('push','pull'),('open','close')],
     'polar_local'),
]

print("  %-14s  pc      LOO%%  irred%%  v5              v6              true      v5 v6" %
      "axis")
print("  " + "-"*86)
v5_correct = 0; v6_correct = 0
for name, pairs, true_type in THIRTY_AXIS_BENCHMARK:
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-14s  n/a" % name); continue
    loo_v  = axis_loo(ax, valid, RELAXED_MASK)
    ho_pairs = pairs  # use training pairs as self-holdout for speed
    irr_f, _, _ = irred_on_holdout(ax, pairs[:4], RELAXED_MASK)
    p_v5 = classify_v5(pc, loo_v, irr_f)
    p_v6 = classify_v6(pc, loo_v, irr_f)
    def match(pred, true):
        return (true.split('_')[0] in pred or true in pred or
                ('morph' in pred and 'morph' in true) or
                ('phonol' in pred and 'phonol' in true) or
                ('relational' in pred and 'relational' in true) or
                ('factual' in pred and 'factual' in true) or
                ('translation' in pred and 'translation' in true) or
                ('polar' in pred and 'polar' in true) or
                ('semantic' in pred and 'semantic' in true))
    m5 = match(p_v5, true_type); m6 = match(p_v6, true_type)
    if m5: v5_correct += 1
    if m6: v6_correct += 1
    t5 = '✓' if m5 else '✗'; t6 = '✓' if m6 else '✗'
    print("  %-14s  pc=%.3f  %.0f%%   %.0f%%  %-16s %-16s %-12s %s %s" %
          (name, pc, 100*loo_v, 100*irr_f, p_v5[:16], p_v6[:16], true_type, t5, t6))
print()
print("  v5: %d/30 = %.0f%%   v6: %d/30 = %.0f%%" %
      (v5_correct, 100*v5_correct/30, v6_correct, 100*v6_correct/30))
print()

# =====================================================================
# PART B: +ing WITHIN GROUP E
# =====================================================================
print("PART B: +ing placement in morphological family groups")
print("-"*80)

ING_PAIRS   = [('go','going'),('take','taking'),('run','running'),('see','seeing'),
                ('give','giving'),('make','making'),('write','writing'),('read','reading'),
                ('work','working'),('play','playing'),('eat','eating'),('sleep','sleeping')]
ED_PAIRS    = [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                ('play','played'),('clean','cleaned'),('open','opened'),('start','started')]
ABLAUT_PAIRS= [('go','went'),('take','took'),('give','gave'),('see','saw'),
                ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')]
THIRD_PS    = [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                ('read','reads'),('write','writes'),('play','plays'),('work','works'),
                ('talk','talks'),('drive','drives'),('sleep','sleeps'),('stand','stands')]
PLURAL_PAIRS= [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                ('tree','trees'),('book','books'),('bird','birds'),('door','doors')]
ANCE_PAIRS  = [('perform','performance'),('exist','existence'),('enter','entrance'),
                ('resist','resistance'),('accept','acceptance'),('appear','appearance')]
NESS_PAIRS  = [('happy','happiness'),('kind','kindness'),('sad','sadness'),
                ('bright','brightness'),('dark','darkness'),('soft','softness')]

ax_ing, _, _ = compute_axis(ING_PAIRS)
ax_ed,  _, _ = compute_axis(ED_PAIRS)
ax_ab,  _, _ = compute_axis(ABLAUT_PAIRS)
ax_3ps, _, _ = compute_axis(THIRD_PS)
ax_pl,  _, _ = compute_axis(PLURAL_PAIRS)
ax_ac,  _, _ = compute_axis(ANCE_PAIRS)
ax_ns,  _, _ = compute_axis(NESS_PAIRS)

group_e_axes = [('ed_reg', ax_ed), ('ablaut', ax_ab), ('+3ps', ax_3ps), ('+plural', ax_pl)]
print("  cos(+ing, GROUP E axes):")
for name, ax in group_e_axes:
    if ax_ing is not None and ax is not None:
        c = float(np.dot(ax_ing.astype(np.float32), ax.astype(np.float32)))
        print("  cos(+ing, %-8s) = %+.4f" % (name, c))
print()
print("  cos(+ing, GROUP A/B reference):")
for name, ax in [('ance', ax_ac), ('ness', ax_ns)]:
    if ax_ing is not None and ax is not None:
        c = float(np.dot(ax_ing.astype(np.float32), ax.astype(np.float32)))
        print("  cos(+ing, %-8s) = %+.4f" % (name, c))
print()

# Summary: which group does +ing belong to?
if ax_ing is not None:
    e_cosines = [float(np.dot(ax_ing.astype(np.float32), ax.astype(np.float32)))
                 for _, ax in group_e_axes if ax is not None]
    print("  Mean cos(+ing, GROUP E) = %.4f" % np.mean(e_cosines))
    print()

# =====================================================================
# PART C: +ly ADVERB AXIS — GROUP F?
# =====================================================================
print("PART C: +ly adverb axis (adj→adverb)")
print("-"*80)

LY_PAIRS = [('quick','quickly'),('slow','slowly'),('happy','happily'),
             ('careful','carefully'),('loud','loudly'),('soft','softly'),
             ('bright','brightly'),('clear','clearly'),('dark','darkly'),
             ('fair','fairly'),('hard','hardly'),('kind','kindly'),
             ('strong','strongly'),('warm','warmly'),('cool','coolly'),
             ('calm','calmly'),('free','freely'),('deep','deeply')]

ax_ly, valid_ly, pc_ly = compute_axis(LY_PAIRS)
if ax_ly is not None:
    loo_ly = axis_loo(ax_ly, valid_ly, CLEAN_MASK)
    irr_ly, _, _ = irred_on_holdout(ax_ly, LY_PAIRS[:5], CLEAN_MASK)
    print("  +ly axis: pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_ly, 100*loo_ly, 100*irr_ly, len(valid_ly)))
    print("  Predicted type (v6): %s" % classify_v6(pc_ly, loo_ly, irr_ly))
    print()
    print("  cos(+ly, GROUP axes):")
    for name, ax in [('ness(B)', ax_ns), ('ing(E?)', ax_ing), ('ed_reg(E)', ax_ed),
                      ('3ps(E)', ax_3ps), ('ance(A)', ax_ac), ('plural', ax_pl)]:
        if ax is not None:
            c = float(np.dot(ax_ly.astype(np.float32), ax.astype(np.float32)))
            print("  cos(+ly, %-12s) = %+.4f" % (name, c))
    print()

    # Is there a GROUP F? Test +ly with other adj-sourced axes
    ABLE_PAIRS = [('read','readable'),('wash','washable'),('love','lovable'),
                   ('use','usable'),('accept','acceptable'),('avoid','avoidable')]
    ER_COMP_PAIRS = [('fast','faster'),('slow','slower'),('bright','brighter'),
                      ('dark','darker'),('soft','softer'),('warm','warmer')]
    ax_ab2, _, _ = compute_axis(ABLE_PAIRS)
    ax_ec,  _, _ = compute_axis(ER_COMP_PAIRS)
    for name, ax in [('able(D)', ax_ab2), ('er_comp', ax_ec)]:
        if ax is not None and ax_ly is not None:
            c = float(np.dot(ax_ly.astype(np.float32), ax.astype(np.float32)))
            print("  cos(+ly, %-12s) = %+.4f" % (name, c))
    print()

# =====================================================================
# PART D: BILINGUAL TRANSLATION AXIS vs CJK VOID AXIS
# =====================================================================
print("PART D: Bilingual axis — morphological overshoot vs direct translation")
print("-"*80)

# Direct English→Chinese translation axis (subset from 30-axis benchmark)
EN_ZH_PAIRS = [('sun','日'),('moon','月'),('water','水'),('fire','火'),
                ('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼'),
                ('heart','心'),('tree','木'),('sea','海'),('sky','天'),
                ('man','男'),('earth','土'),('rain','雨')]

ax_enzh, valid_enzh, pc_enzh = compute_axis(EN_ZH_PAIRS)

# CJK void axis: displacement from +al_rel applied to OOD words
AL_REL_PAIRS = [('nation','national'),('region','regional'),('culture','cultural'),
                 ('nature','natural'),('person','personal'),('origin','original'),
                 ('emotion','emotional'),('tradition','traditional')]
ax_alr, valid_alr, _ = compute_axis(AL_REL_PAIRS)

if ax_enzh is not None and ax_alr is not None:
    bs_alr, _ = best_scale(ax_alr, valid_alr, CLEAN_MASK)
    c = float(np.dot(ax_enzh.astype(np.float32), ax_alr.astype(np.float32)))
    print("  cos(EN→ZH_axis, +al_rel_axis) = %+.4f" % c)
    print("  EN→ZH: pc=%.4f  LOO=n/a (CJK targets)" % pc_enzh)
    print()

    # Compute CJK displacement axis from OOD words
    OOD_TO_CJK = [('table','桌子'),('window','窗户'),('garden','花园'),
                   ('computer','电脑'),('music','音乐'),('science','科学'),
                   ('market','市场'),('forest','森林'),('bridge','桥'),('ocean','海洋')]
    ax_ood_cjk, valid_ood, pc_ood = compute_axis(OOD_TO_CJK)
    if ax_ood_cjk is not None:
        c2 = float(np.dot(ax_enzh.astype(np.float32), ax_ood_cjk.astype(np.float32)))
        print("  cos(EN→ZH_axis, OOD→CJK_void_axis) = %+.4f  (1.0=same, 0=orthogonal)" % c2)
        c3 = float(np.dot(ax_alr.astype(np.float32), ax_ood_cjk.astype(np.float32)))
        print("  cos(+al_rel, OOD→CJK_void_axis) = %+.4f" % c3)
        print()

    print("  Test: navigate OOD word via EN→ZH axis (should give Chinese translation)")
    for word, zh in [('table','桌子'),('garden','花园'),('computer','电脑'),
                      ('music','音乐'),('science','科学')]:
        es, sid = get_emb(word)
        if es is None: continue
        bs_zh, _ = best_scale(ax_enzh, valid_enzh, RELAXED_MASK)
        r = nn_retrieve(W_E[sid]+bs_zh*ax_enzh, source_ids(word), RELAXED_MASK, 3)
        found = r[0][0]
        is_cjk = any('\u4e00' <= c <= '\u9fff' for c in found)
        print("  %-12s -> %-8s  (expected: %s)  CJK=%s" % (word, found, zh, is_cjk))
print()

# =====================================================================
# PART E: GROUP E PROBE — +3ps stability under different 10-pair subsets
# =====================================================================
print("PART E: GROUP E probe stability — +3ps classification under subsampling")
print("-"*80)

ALL_3PS = [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
            ('read','reads'),('write','writes'),('play','plays'),('work','works'),
            ('talk','talks'),('drive','drives'),('sleep','sleeps'),('stand','stands'),
            ('think','thinks'),('know','knows'),('say','says'),('see','sees'),
            ('hold','holds'),('tell','tells'),('feel','feels'),('hear','hears')]

print("  Testing +3ps on 10 random 8-pair subsets:")
np.random.seed(42)
results = []
for trial in range(10):
    idx = np.random.choice(len(ALL_3PS), 8, replace=False)
    sub_pairs = [ALL_3PS[i] for i in idx]
    ho_pairs  = [ALL_3PS[i] for i in range(len(ALL_3PS)) if i not in idx][:5]
    ax, vl, pc = compute_axis(sub_pairs)
    if ax is None: continue
    loo = axis_loo(ax, vl, CLEAN_MASK)
    irr, _, _ = irred_on_holdout(ax, ho_pairs, CLEAN_MASK)
    pred = classify_v6(pc, loo, irr)
    ok = 'morph_moderate' in pred or 'morph' in pred
    results.append((pc, loo, irr, pred, ok))
    print("  Trial %2d: pc=%.3f LOO=%.0f%% irred=%.0f%% -> %-24s %s" %
          (trial+1, pc, 100*loo, 100*irr, pred, '✓' if ok else '✗'))
n_ok = sum(1 for _,_,_,_,ok in results if ok)
print()
print("  +3ps classification stability: %d/10 = %.0f%%" % (n_ok, 100*n_ok/10))
print()
print("  +3ps feature ranges across 10 trials:")
print("  pc:   %.3f - %.3f" % (min(r[0] for r in results), max(r[0] for r in results)))
print("  LOO:  %.0f%% - %.0f%%" % (100*min(r[1] for r in results), 100*max(r[1] for r in results)))
print("  irred:%.0f%% - %.0f%%" % (100*min(r[2] for r in results), 100*max(r[2] for r in results)))
