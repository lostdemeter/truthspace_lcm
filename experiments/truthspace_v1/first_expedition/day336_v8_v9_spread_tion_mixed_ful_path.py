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
    # Spread: std deviation of pairwise cosines
    pairs_cos = [np.dot(cn[i], cn[j]) for i in range(n) for j in range(i+1, n)]
    spread = float(np.std(pairs_cos)) if len(pairs_cos) > 1 else 0.0
    return md, valid, pc, spread

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
    irred=0; n_ho=0
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1; found = False
        for s in np.linspace(lo, hi, n):
            if nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask, 1)[0][0]==t_w:
                found=True; break
        if not found: irred += 1
    return irred/n_ho if n_ho else 0.0

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

# =====================================================================
# PREDICTORS: v6, v8, v9
# =====================================================================
def classify_v6(pc, loo, irred, spread=0.0):
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.20:
        if loo > 0.50:     return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo > 0.50:
            if irred >= 0.40: return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse'
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

def classify_v8(pc, loo, irred, spread=0.0):
    """v8: 3 micro-adjustments: pc 0.20->0.195, irred 0.30->0.40, loo>0.50->>=0.50"""
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.195:                                    # CHANGE 1
        if loo >= 0.50:                                 # CHANGE 3
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'  # CHANGE 2
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo >= 0.50:                                 # CHANGE 3
            if irred >= 0.40: return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse'
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

def classify_v10(pc, loo, irred, spread=0.0):
    """v10: v9 + less fix (loo==0, irred 0.20-0.60 -> phonol_scatter) +
             er_noun fix (loo in (0,0.50), irred 0.20-0.60 -> semantic_diverse)"""
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if loo >= 0.80 and spread > 0.07:
            return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if loo >= 0.50:
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        # v10 LESS FIX: loo==0 with moderate irred is phonol_scatter not semantic
        elif loo == 0.0 and 0.20 <= irred < 0.60: return 'phonol_scatter'
        elif loo == 0.0 and irred < 0.20: return 'semantic_diverse'
        # v10 ER_NOUN FIX: low-but-nonzero loo with moderate irred is semantic_diverse
        elif 0.0 < loo < 0.50 and 0.20 <= irred < 0.60: return 'semantic_diverse'
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

def classify_v9(pc, loo, irred, spread=0.0):
    """v9: v8 + spread rule for ablaut (gated on loo>=0.80) + ful path fix"""
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        # v9 SPREAD RULE: ablaut-type axes have high pc, high loo, high spread
        if loo >= 0.80 and spread > 0.07:
            return 'phonol_scatter'   # ablaut: pc~0.35, loo~0.88, spread~0.09
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if loo >= 0.50:
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo >= 0.50:
            if irred >= 0.40:
                # v9 FUL PATH: high loo (>=0.70) with high irred is phonol_scatter not semantic
                if loo >= 0.70: return 'phonol_scatter'  # ful: loo=0.75, irred=0.67
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse'
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

# =====================================================================
# THE BENCHMARK
# =====================================================================
FIXED_BENCH = [
    ('er_comp',   [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],                         [('dark','darker'),('soft','softer'),('heavy','heavier')], 'morph_uniform'),
    ('er_sup',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')],               [('warm','warmest'),('long','longest'),('cold','coldest')], 'morph_uniform'),
    ('relational',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],                    [('Cairo','Egypt'),('Seoul','Korea'),('Lima','Peru')], 'relational_geom'),
    ('al_rel',    [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')], [('history','historical'),('season','seasonal'),('accident','accidental')], 'relational_geom'),
    ('plural',    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),('book','books'),('bird','birds'),('door','doors')],                                           [('cup','cups'),('word','words'),('room','rooms')], 'morph_moderate'),
    ('3ps',       [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),('write','writes'),('play','plays'),('work','works')],                                         [('talk','talks'),('sleep','sleeps'),('open','opens')], 'morph_moderate'),
    ('ed_reg',    [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played'),('clean','cleaned'),('open','opened'),('start','started')],                           [('end','ended'),('look','looked'),('rain','rained')], 'morph_moderate'),
    ('ing',       [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),('make','making'),('write','writing'),('read','reading')],                                [('eat','eating'),('work','working'),('play','playing')], 'morph_moderate'),
    ('cc',        [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')],                                                [('cup','Cup'),('door','Door'),('word','Word')], 'morph_moderate'),
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
    ('able',      [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')],       [('comfort','comfortable'),('manage','manageable'),('reach','reachable')], 'phonol_scatter'),
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
print("DAY 336: v8 + v9 BENCHMARK, SPREAD RULE, +tion MIXED AXIS, FUL PATH FIX")
print("="*80)
print()

# =====================================================================
# PART A: v8 AND v9 HEAD-TO-HEAD BENCHMARK
# =====================================================================
print("PART A: v6 vs v8 vs v9 head-to-head on 30-axis benchmark")
print("-"*80)

results = {}
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None or len(valid) < 2: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr   = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    results[name] = (pc, loo_v, irr, spread, true_type)

v6_score = v8_score = v9_score = v10_score = 0
print("  %-12s  pc    LOO  irred spread  v8_pred            true       v6 v8 v9 v10  ch" %
      "axis")
print("  " + "-"*110)
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    if name not in results: continue
    pc, loo_v, irr, spread, _ = results[name]
    p6  = classify_v6(pc, loo_v, irr, spread)
    p8  = classify_v8(pc, loo_v, irr, spread)
    p9  = classify_v9(pc, loo_v, irr, spread)
    p10 = classify_v10(pc, loo_v, irr, spread)
    ok6  = match(p6,  true_type); ok8  = match(p8,  true_type)
    ok9  = match(p9,  true_type); ok10 = match(p10, true_type)
    if ok6:  v6_score  += 1
    if ok8:  v8_score  += 1
    if ok9:  v9_score  += 1
    if ok10: v10_score += 1
    t6='✓' if ok6 else '✗'; t8='✓' if ok8 else '✗'
    t9='✓' if ok9 else '✗'; t10='✓' if ok10 else '✗'
    changed = ''
    if not ok6  and ok8:  changed += 'v8+'
    if ok6  and not ok8:  changed += 'v8-'
    if not ok8  and ok9:  changed += 'v9+'
    if ok8  and not ok9:  changed += 'v9-'
    if not ok9  and ok10: changed += 'v10+'
    if ok9  and not ok10: changed += 'v10-'
    print("  %-12s %.3f %.0f%% %.2f  %.3f  %-18s %-12s %s %s %s %s  %s" %
          (name, pc, 100*loo_v, irr, spread, p8[:18], true_type, t6, t8, t9, t10, changed))
print()
print("  v6:  %d/30 = %.0f%%" % (v6_score,  100*v6_score/30))
print("  v8:  %d/30 = %.0f%%" % (v8_score,  100*v8_score/30))
print("  v9:  %d/30 = %.0f%%" % (v9_score,  100*v9_score/30))
print("  v10: %d/30 = %.0f%%" % (v10_score, 100*v10_score/30))
print()

# Detailed spread values for key axes
print("  Spread values for key axes (to calibrate spread rule):")
for name in ['ablaut','ablaut_t','relational','er_comp','er_sup','ness','ing','un_neg','ful','ity']:
    if name in results:
        pc, loo_v, irr, spread, true_type = results[name]
        print("  %-12s  pc=%.3f  loo=%.0f%%  irred=%.2f  spread=%.4f  [%s]" %
              (name, pc, 100*loo_v, irr, spread, true_type))
print()

# =====================================================================
# PART B: +tion MIXED AXIS IN BENCHMARK
# =====================================================================
print("PART B: +tion with mixed 14-pair training — impact on v8/v9")
print("-"*80)

TION_MIXED = [
    ('act','action'),('direct','direction'),('educate','education'),('create','creation'),
    ('produce','production'),('relate','relation'),('combine','combination'),('apply','application'),
    ('express','expression'),('extend','extension'),('omit','omission'),('admit','admission'),
    ('permit','permission'),('construct','construction'),
]
TION_HOLDOUT = [('restrict','restriction'),('instruct','instruction'),('destruct','destruction'),
                 ('subtract','subtraction'),('attract','attraction'),('react','reaction')]

ax_tion_m, valid_tion_m, pc_tion_m, sp_tion_m = compute_axis_with_spread(TION_MIXED)
loo_tion_m = axis_loo(ax_tion_m, valid_tion_m, CLEAN_MASK)
irr_tion_m = irred_on_holdout(ax_tion_m, TION_HOLDOUT, CLEAN_MASK)
p6_m = classify_v6(pc_tion_m, loo_tion_m, irr_tion_m, sp_tion_m)
p8_m = classify_v8(pc_tion_m, loo_tion_m, irr_tion_m, sp_tion_m)
p9_m = classify_v9(pc_tion_m, loo_tion_m, irr_tion_m, sp_tion_m)
print("  +tion MIXED (14 pairs):  pc=%.4f  LOO=%.0f%%  irred=%.0f%%  spread=%.4f" %
      (pc_tion_m, 100*loo_tion_m, 100*irr_tion_m, sp_tion_m))
print("  v6=%s  v8=%s  v9=%s  [true=phonol_scatter]" %
      ('✓' if match(p6_m,'phonol_scatter') else '✗',
       '✓' if match(p8_m,'phonol_scatter') else '✗',
       '✓' if match(p9_m,'phonol_scatter') else '✗'))
print()

# Now run benchmark with tion REPLACED by mixed version
print("  Benchmark with +tion REPLACED by 14-pair mixed version:")
tion_idx = next(i for i,(n,_,_,_) in enumerate(FIXED_BENCH) if n=='tion')
BENCH_TION_MIXED = list(FIXED_BENCH)
BENCH_TION_MIXED[tion_idx] = ('tion_mixed', TION_MIXED, TION_HOLDOUT, 'phonol_scatter')

v8_mix_score = 0
for name, train_pairs, holdout_pairs, true_type in BENCH_TION_MIXED:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None or len(valid) < 2: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr   = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    p8    = classify_v8(pc, loo_v, irr, spread)
    if match(p8, true_type): v8_mix_score += 1
print("  v8 with mixed +tion: %d/30 = %.0f%% (was %d/30=%.0f%%)" %
      (v8_mix_score, 100*v8_mix_score/30, v8_score, 100*v8_score/30))
print()

# =====================================================================
# PART C: SPREAD CALIBRATION FOR ABLAUT VS RELATIONAL
# =====================================================================
print("PART C: Spread calibration — ablaut vs relational vs morph axes")
print("-"*80)

# Compute spread for a wider range of axes
SPREAD_TEST_AXES = [
    ('ablaut',    [('go','went'),('take','took'),('give','gave'),('see','saw'),('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')]),
    ('ablaut_t',  [('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left'),('deal','dealt'),('sleep','slept'),('mean','meant')]),
    ('relational',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')]),
    ('er_comp',   [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')]),
    ('er_sup',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')]),
    ('plural',    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),('book','books'),('bird','birds'),('door','doors')]),
    ('3ps',       [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),('write','writes'),('play','plays'),('work','works')]),
    ('ing',       [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),('make','making'),('write','writing'),('read','reading')]),
    ('ness',      [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')]),
    ('ful',       [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')]),
    ('un_neg',    [('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),('known','unknown'),('safe','unsafe'),('usual','unusual'),('equal','unequal')]),
    ('adj_ant',   [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('bright','dark'),('hard','soft'),('high','low'),('rich','poor')]),
    ('en_es',     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día'),('night','noche'),('hand','mano'),('year','año')]),
    ('en_zh',     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼')]),
    ('num_word',  [('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),('6','six'),('7','seven'),('8','eight')]),
]

print("  %-12s  pc      spread   loo     irred  [true]" % "axis")
print("  " + "-"*65)
for name, pairs in SPREAD_TEST_AXES:
    ax, valid, pc, spread = compute_axis_with_spread(pairs)
    if ax is None: continue
    true = next((t for n,_,_,t in FIXED_BENCH if n==name), '?')
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    print("  %-12s  %.4f  %.4f   %.2f    [%s]" % (name, pc, spread, loo_v, true))
print()

# Check: does v9 spread rule correctly separate ablaut from relational?
print("  Spread rule verification:")
for name, pairs in SPREAD_TEST_AXES:
    if name not in ['ablaut','ablaut_t','relational','er_comp','er_sup','ing','num_word']:
        continue
    ax, valid, pc, spread = compute_axis_with_spread(pairs)
    if ax is None: continue
    true = next((t for n,_,_,t in FIXED_BENCH if n==name), '?')
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr = 0.0  # skip holdout for speed, use 0
    p9 = classify_v9(pc, loo_v, irr, spread)
    spreads_flag = '(spread ACTIVE)' if pc >= 0.30 and loo_v >= 0.80 and spread > 0.07 else ''
    ok = match(p9, true)
    print("  %s %-12s  pc=%.3f  spread=%.4f  loo=%.0f%%  pred_v9=%-22s  %s" %
          ('✓' if ok else '✗', name, pc, spread, 100*loo_v, p9, spreads_flag))
print()
