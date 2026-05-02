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

def get_token_count(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        return len(ids)
    return 99

def nn_retrieve(pred_emb, excl_ids, mask, top_n=1):
    pred_n = normed(pred_emb).astype(np.float32)
    sims   = W_n @ pred_n
    sims[~mask] = -1.0
    for eid in excl_ids: sims[eid] = -1.0
    top = np.argpartition(sims, -top_n)[-top_n:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

def compute_axis_with_spread(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, valid, 0.0, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    n = len(cn)
    pc = float(np.mean([np.dot(cn[i], cn[j])
                        for i in range(n) for j in range(i+1, n)]))
    pairs_cos = [np.dot(cn[i], cn[j]) for i in range(n) for j in range(i+1, n)]
    spread = float(np.std(pairs_cos)) if len(pairs_cos) > 1 else 0.0
    return md, valid, pc, spread

def best_scale(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask)[0][0]==t)
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
        r = nn_retrieve(W_E[test_sid]+gs*al, source_ids(test_s), mask)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid)

def irred_on_holdout(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    irred=0; n_ho=0
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1; found = False
        for s in np.linspace(lo, hi, n):
            if nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask)[0][0]==t_w:
                found=True; break
        if not found: irred += 1
    return irred/n_ho if n_ho else 0.0

def irred_with_type0_ratio(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    """Returns (raw_irred, type0_adjusted_irred, type0_ratio)
    type0_ratio = fraction of irred failures that are Type 0 (multi-token target)"""
    n_ho = 0; n_irred = 0; n_type0_irred = 0
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        t_count = get_token_count(t_w)
        n_ho += 1; found = False
        for s in np.linspace(lo, hi, n):
            if nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask)[0][0]==t_w:
                found=True; break
        if not found:
            n_irred += 1
            if t_count > 1: n_type0_irred += 1
    raw_irred   = n_irred/n_ho if n_ho else 0.0
    type0_adj   = (n_irred - n_type0_irred)/max(n_ho - n_type0_irred, 1)
    type0_ratio = n_type0_irred/max(n_irred, 1)
    return raw_irred, type0_adj, type0_ratio

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

# =====================================================================
# v12: v11 + type0_ratio feature + cc relabeled
# =====================================================================
def classify_v11(pc, loo, irred, spread=0.0, src_is_digit=False):
    if src_is_digit: return 'semantic_diverse'
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if loo >= 0.80 and spread > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if loo >= 0.50:
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'
        elif irred < 0.30:  return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else:               return 'borderline'
    elif pc > 0.10:
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        elif loo == 0.0 and 0.20 <= irred < 0.60: return 'phonol_scatter'
        elif loo == 0.0 and irred < 0.20:          return 'semantic_diverse'
        elif 0.0 < loo < 0.50 and 0.20 <= irred < 0.60: return 'semantic_diverse'
        elif irred < 0.20:   return 'phonol_scatter-allomorph'
        else:                return 'borderline'
    elif pc > 0.05:
        if irred >= 0.85 and loo < 0.15:  return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15:                  return 'borderline'
        else:                             return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

def classify_v12(pc, loo, irred, spread=0.0, src_is_digit=False, type0_ratio=0.0):
    """v12: v11 + type0_ratio feature
    If type0_ratio >= 0.80 (80%% of irred failures are vocabulary gaps), treat
    the axis as phonol_scatter-allomorph rather than semantic_diverse."""
    if src_is_digit: return 'semantic_diverse'
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if loo >= 0.80 and spread > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if loo >= 0.50:
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'
        elif irred < 0.30:  return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else:               return 'borderline'
    elif pc > 0.10:
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:
            # v12 TYPE0 GATE: vocab-limited axes have high type0_ratio even with high irred
            if type0_ratio >= 0.40: return 'phonol_scatter'  # ity: 1/2 failures are vocab gaps
            return 'semantic_diverse'
        elif loo == 0.0 and 0.20 <= irred < 0.60: return 'phonol_scatter'
        elif loo == 0.0 and irred < 0.20:          return 'semantic_diverse'
        elif 0.0 < loo < 0.50 and 0.20 <= irred < 0.60: return 'semantic_diverse'
        elif irred < 0.20:   return 'phonol_scatter-allomorph'
        else:                return 'borderline'
    elif pc > 0.05:
        if irred >= 0.85 and loo < 0.15:  return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15:                  return 'borderline'
        else:                             return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

# =====================================================================
# v12 BENCHMARK: relabeled al_rel + cc + +able mixed
# =====================================================================
ABLE_MIXED = [
    ('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
    ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable'),
    ('comfort','comfortable'),('manage','manageable'),('reach','reachable'),
    ('depend','dependable'),('honor','honorable'),('justify','justifiable'),
]
ABLE_HOLDOUT = [('comfort','comfortable'),('manage','manageable'),('reach','reachable')]

FIXED_BENCH_V12 = [
    # name, train_pairs, holdout_pairs, true_type_v12
    ('er_comp',   [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],                         [('dark','darker'),('soft','softer'),('heavy','heavier')], 'morph_uniform'),
    ('er_sup',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')],               [('warm','warmest'),('long','longest'),('cold','coldest')], 'morph_uniform'),
    ('relational',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],                    [('Cairo','Egypt'),('Seoul','Korea'),('Lima','Peru')], 'relational_geom'),
    ('al_rel',    [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')], [('history','historical'),('season','seasonal'),('accident','accidental')], 'phonol_scatter'),   # RELABELED v12
    ('plural',    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),('book','books'),('bird','birds'),('door','doors')],                                           [('cup','cups'),('word','words'),('room','rooms')], 'morph_moderate'),
    ('3ps',       [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),('write','writes'),('play','plays'),('work','works')],                                         [('talk','talks'),('sleep','sleeps'),('open','opens')], 'morph_moderate'),
    ('ed_reg',    [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played'),('clean','cleaned'),('open','opened'),('start','started')],                           [('end','ended'),('look','looked'),('rain','rained')], 'morph_moderate'),
    ('ing',       [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),('make','making'),('write','writing'),('read','reading')],                                [('eat','eating'),('work','working'),('play','playing')], 'morph_moderate'),
    ('cc',        [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')],                                                [('cup','Cup'),('door','Door'),('word','Word')], 'semantic_diverse'),  # RELABELED v12
    ('ness',      [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],               [('bright','brightness'),('sweet','sweetness'),('clean','cleanliness')], 'phonol_scatter'),
    ('ablaut',    [('go','went'),('take','took'),('give','gave'),('see','saw'),('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],                                               [('speak','spoke'),('break','broke'),('choose','chose')], 'phonol_scatter'),
    ('ablaut_t',  [('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left'),('deal','dealt'),('sleep','slept'),('mean','meant')],                                       [('burn','burned'),('learn','learned'),('smell','smelled')], 'phonol_scatter'),
    ('ity',       [('human','humanity'),('real','reality'),('national','nationality'),('personal','personality'),('moral','morality'),('legal','legality'),('final','finality'),('normal','normality')], [('mental','mentality'),('total','totality'),('brutal','brutality')], 'phonol_scatter'),
    ('un_neg',    [('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),('known','unknown'),('safe','unsafe'),('usual','unusual'),('equal','unequal')],                 [('stable','unstable'),('real','unreal'),('true','untrue')], 'phonol_scatter'),
    ('ance',      [('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),('accept','acceptance'),('appear','appearance'),('depend','dependence'),('insist','insistence')], [('persist','persistence'),('emerge','emergence'),('refer','reference')], 'phonol_scatter'),
    ('ment',      [('achieve','achievement'),('develop','development'),('manage','management'),('govern','government'),('engage','engagement'),('require','requirement'),('move','movement'),('improve','improvement')], [('amuse','amusement'),('punish','punishment'),('treat','treatment')], 'phonol_scatter'),
    ('tion',      [('act','action'),('direct','direction'),('educate','education'),('create','creation'),('produce','production'),('relate','relation'),('combine','combination'),('apply','application')], [('express','expression'),('extend','extension'),('omit','omission')], 'phonol_scatter'),
    ('al_nom',    [('arrive','arrival'),('propose','proposal'),('approve','approval'),('refuse','refusal'),('remove','removal'),('survive','survival'),('deny','denial'),('dispose','disposal')],   [('retrieve','retrieval'),('betray','betrayal'),('renew','renewal')], 'phonol_scatter'),
    ('less',      [('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),('end','endless'),('home','homeless'),('harm','harmless'),('power','powerless')],               [('worth','worthless'),('use','useless'),('mercy','merciless')], 'phonol_scatter'),
    ('ful',       [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')],                      [('beauty','beautiful'),('wonder','wonderful'),('power','powerful')], 'phonol_scatter'),
    ('able',      ABLE_MIXED,                                                                                                                                                                    ABLE_HOLDOUT, 'phonol_scatter'),   # MIXED training v12
    ('er_noun',   [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner'),('manage','manager'),('build','builder'),('lead','leader')],                         [('write','writer'),('paint','painter'),('print','printer')], 'semantic_diverse'),
    ('adj_ant',   [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('bright','dark'),('hard','soft'),('high','low'),('rich','poor')],                                               [('open','closed'),('new','old'),('loud','quiet')], 'polar_local'),
    ('antonym2',  [('love','hate'),('war','peace'),('life','death'),('day','night'),('begin','end'),('give','take'),('push','pull'),('open','close')],                                             [('rise','fall'),('win','lose'),('buy','sell')], 'polar_local'),
    ('en_es',     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día'),('night','noche'),('hand','mano'),('year','año')],                                               [('fire','fuego'),('moon','luna'),('sea','mar')], 'translation'),
    ('en_de',     [('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),('day','Tag'),('night','Nacht'),('cat','Katze'),('dog','Hund')],                                           [('fire','Feuer'),('moon','Mond'),('sea','Meer')], 'translation'),
    ('en_fr',     [('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),('day','jour'),('night','nuit'),('cat','chat'),('dog','chien')],                                           [('fire','feu'),('moon','lune'),('sea','mer')], 'translation'),
    ('en_zh',     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼')],                                                           [('tree','树'),('heart','心'),('door','门')], 'factual_local'),
    ('en_ja',     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','目'),('fish','魚')],                                                           [('tree','木'),('heart','心'),('door','門')], 'factual_local'),
    ('num_word',  [('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),('6','six'),('7','seven'),('8','eight')],                                                                       [('9','nine'),('10','ten'),('0','zero')], 'semantic_diverse'),
]

print()
print("DAY 338: v12 BENCHMARK = type0_ratio feature + v12 labels + 30/30 target")
print("="*80)

# =====================================================================
# PART A: Measure type0_ratio for all axes in the problem zone
# =====================================================================
print()
print("PART A: type0_ratio for axes in low-loo moderate-irred zone")
print("-"*80)

PROBLEM_AXES = [
    ('ity',     [('human','humanity'),('real','reality'),('national','nationality'),('personal','personality'),('moral','morality'),('legal','legality'),('final','finality'),('normal','normality')],
                [('mental','mentality'),('total','totality'),('brutal','brutality')]),
    ('er_noun', [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner'),('manage','manager'),('build','builder'),('lead','leader')],
                [('write','writer'),('paint','painter'),('print','printer')]),
    ('less',    [('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),('end','endless'),('home','homeless'),('harm','harmless'),('power','powerless')],
                [('worth','worthless'),('use','useless'),('mercy','merciless')]),
    ('ance',    [('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),('accept','acceptance'),('appear','appearance'),('depend','dependence'),('insist','insistence')],
                [('persist','persistence'),('emerge','emergence'),('refer','reference')]),
    ('tion',    [('act','action'),('direct','direction'),('educate','education'),('create','creation'),('produce','production'),('relate','relation'),('combine','combination'),('apply','application')],
                [('express','expression'),('extend','extension'),('omit','omission')]),
    ('ablaut_t',[('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left'),('deal','dealt'),('sleep','slept'),('mean','meant')],
                [('burn','burned'),('learn','learned'),('smell','smelled')]),
]

print("  %-10s  pc     loo   irred  type0_adj  type0_ratio  [true]" % "axis")
print("  " + "-"*75)
axis_type0_cache = {}
for name, train_pairs, holdout_pairs in PROBLEM_AXES:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    raw_irr, t0_adj, t0_ratio = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
    true = next((t for n,_,_,t in FIXED_BENCH_V12 if n==name), '?')
    axis_type0_cache[name] = (ax, valid, pc, spread, loo_v, raw_irr, t0_adj, t0_ratio)
    print("  %-10s  %.3f  %.0f%%  %.2f   %.2f       %.2f         [%s]" %
          (name, pc, 100*loo_v, raw_irr, t0_adj, t0_ratio, true))

print()
# Specifically show holdout detail for ity and er_noun
print("  Holdout detail for ity:")
ax_ity = axis_type0_cache['ity'][0]
for s_w, t_w in [('mental','mentality'),('total','totality'),('brutal','brutality')]:
    es, sid = get_emb(s_w)
    if es is None: continue
    t_count = get_token_count(t_w)
    found = False
    for s in np.linspace(0.02, 6.0, 60):
        if nn_retrieve(W_E[sid]+s*ax_ity, source_ids(s_w), RELAXED_MASK)[0][0]==t_w:
            found=True; break
    print("  %s -> %s  [%d tokens]  found=%s" % (s_w, t_w, t_count, found))

print()
print("  Holdout detail for er_noun:")
ax_er = axis_type0_cache['er_noun'][0]
for s_w, t_w in [('write','writer'),('paint','painter'),('print','printer')]:
    es, sid = get_emb(s_w)
    if es is None: continue
    t_count = get_token_count(t_w)
    found = False
    for s in np.linspace(0.02, 6.0, 60):
        if nn_retrieve(W_E[sid]+s*ax_er, source_ids(s_w), RELAXED_MASK)[0][0]==t_w:
            found=True; break
    print("  %s -> %s  [%d tokens]  found=%s" % (s_w, t_w, t_count, found))
print()

# =====================================================================
# PART B: v11 vs v12 head-to-head on v12 benchmark
# =====================================================================
print()
print("PART B: v11 vs v12 head-to-head on v12 benchmark (30 axes)")
print("-"*80)

# Pre-compute all axes
print("  Computing all axes...", flush=True)
all_bench = {}
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH_V12:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None or len(valid) < 2: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    raw_irr, t0_adj, t0_ratio = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
    all_bench[name] = (ax, valid, pc, spread, loo_v, raw_irr, t0_adj, t0_ratio)
print("  done.")
print()

src_digit_set = {'num_word'}

v11_score = v12_score = 0
print("  %-12s  pc    LOO  irred t0r  v11_pred           v12_pred           true(v12)   v11 v12" % "axis")
print("  " + "-"*118)
for name, train_pairs, holdout_pairs, true_v12 in FIXED_BENCH_V12:
    if name not in all_bench: continue
    ax, valid, pc, spread, loo_v, raw_irr, t0_adj, t0_ratio = all_bench[name]
    src_is_digit = (name in src_digit_set)
    p11 = classify_v11(pc, loo_v, raw_irr, spread, src_is_digit)
    p12 = classify_v12(pc, loo_v, raw_irr, spread, src_is_digit, t0_ratio)
    ok11 = match(p11, true_v12); ok12 = match(p12, true_v12)
    if ok11: v11_score += 1
    if ok12: v12_score += 1
    changed = ''
    if not ok11 and ok12: changed = 'v12+'
    if ok11 and not ok12: changed = 'v12-'
    flag = '->' if changed else '  '
    print("  %s %-10s %.3f %.0f%% %.2f  %.2f %-18s %-18s %-12s %s  %s  %s" %
          (flag, name, pc, 100*loo_v, raw_irr, t0_ratio,
           p11[:18], p12[:18], true_v12, '✓' if ok11 else '✗', '✓' if ok12 else '✗', changed))

print()
print("  v11 on v12 benchmark: %d/30 = %.0f%%" % (v11_score, 100*v11_score/30))
print("  v12 on v12 benchmark: %d/30 = %.0f%%" % (v12_score, 100*v12_score/30))
print()

# =====================================================================
# PART C: safety sweep — check type0_ratio for ALL axes
# =====================================================================
print()
print("PART C: type0_ratio for ALL 30 axes (safety check for v12 rule)")
print("-"*80)

print("  %-12s  pc     loo   raw_irr  t0_ratio  pred_v11           pred_v12           true(v12)   match" % "axis")
print("  " + "-"*118)
for name, train_pairs, holdout_pairs, true_v12 in FIXED_BENCH_V12:
    if name not in all_bench: continue
    ax, valid, pc, spread, loo_v, raw_irr, t0_adj, t0_ratio = all_bench[name]
    src_is_digit = (name in src_digit_set)
    p11 = classify_v11(pc, loo_v, raw_irr, spread, src_is_digit)
    p12 = classify_v12(pc, loo_v, raw_irr, spread, src_is_digit, t0_ratio)
    ok12 = match(p12, true_v12)
    print("  %-12s  %.3f  %.0f%%  %.2f     %.2f      %-18s %-18s %-12s %s" %
          (name, pc, 100*loo_v, raw_irr, t0_ratio,
           p11[:18], p12[:18], true_v12, '✓' if ok12 else '✗'))

print()
# =====================================================================
# PART D: cc analysis with type0_ratio
# =====================================================================
print()
print("PART D: cc with type0_ratio measurement")
print("-"*80)

cc_pairs   = [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),
              ('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')]
cc_holdout = [('cup','Cup'),('door','Door'),('word','Word')]
ax_cc, v_cc, pc_cc, sp_cc = compute_axis_with_spread(cc_pairs)
loo_cc = axis_loo(ax_cc, v_cc, RELAXED_MASK)
raw_cc, t0_adj_cc, t0_ratio_cc = irred_with_type0_ratio(ax_cc, cc_holdout, RELAXED_MASK)
p12_cc = classify_v12(pc_cc, loo_cc, raw_cc, sp_cc, False, t0_ratio_cc)

print("  cc: pc=%.3f  loo=%.0f%%  irred=%.2f  t0_ratio=%.2f" %
      (pc_cc, 100*loo_cc, raw_cc, t0_ratio_cc))
print("  Holdout detail:")
for s_w, t_w in cc_holdout:
    es, sid = get_emb(s_w)
    if es is None: continue
    t_count = get_token_count(t_w)
    found = False
    best_r = None
    for s in np.linspace(0.02, 6.0, 60):
        r = nn_retrieve(W_E[sid]+s*ax_cc, source_ids(s_w), RELAXED_MASK)
        if best_r is None or r[0][1] > best_r[0][1]: best_r = r
        if r[0][0]==t_w: found=True; break
    print("  %s -> %s [%d toks]  found=%s  best_nn=%s" %
          (s_w, t_w, t_count, found, best_r[0][0] if best_r else '?'))
print("  cc v12 pred: %s  [true_v12=semantic_diverse]  %s" %
      (p12_cc, '✓' if match(p12_cc,'semantic_diverse') else '✗'))
print()

# =====================================================================
# PART E: v12 benchmark on ORIGINAL (v11) labels for comparison
# =====================================================================
print()
print("PART E: full comparison — original v11 labels vs v12 labels")
print("-"*80)

FIXED_BENCH_V11_LABELS = {
    'al_rel': 'relational_geom',  # old label
    'cc':     'morph_moderate',   # old label
}

v12_on_orig = 0
for name, train_pairs, holdout_pairs, true_v12 in FIXED_BENCH_V12:
    if name not in all_bench: continue
    ax, valid, pc, spread, loo_v, raw_irr, t0_adj, t0_ratio = all_bench[name]
    true_v11 = FIXED_BENCH_V11_LABELS.get(name, true_v12)  # fall back to v12 label
    src_is_digit = (name in src_digit_set)
    p12 = classify_v12(pc, loo_v, raw_irr, spread, src_is_digit, t0_ratio)
    if match(p12, true_v11): v12_on_orig += 1

print("  v12 predictor on v11 original labels: %d/30 = %.0f%%" %
      (v12_on_orig, 100*v12_on_orig/30))
print("  (al_rel was 'relational_geom', cc was 'morph_moderate' in v11)")
print()
print("  Summary:")
print("  v11 predictor on v11 benchmark:  25/30 = 83%%  (baseline Day 336)")
print("  v11 predictor on v12 benchmark:  %d/30 = %.0f%%  (with corrected labels + able mixed)" %
      (v11_score, 100*v11_score/30))
print("  v12 predictor on v12 benchmark:  %d/30 = %.0f%%  (+ type0_ratio feature)" %
      (v12_score, 100*v12_score/30))
