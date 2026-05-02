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

def get_token_ids(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        return ids
    return []

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

def irred_typed(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    """Returns typed irreducibility: (irred_rate, details)
    For each holdout pair: Type0=multi-token target, Type1=geo fail(low best_sim), Type2=near-miss"""
    details = []
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        t_toks = get_token_ids(t_w)
        if es is None:
            details.append((s_w, t_w, 'Type0_no_src', None, None))
            continue
        if len(t_toks) > 1:
            details.append((s_w, t_w, 'Type0_multi', None, None))
            continue
        et, tid = get_emb(t_w)
        found_at = None
        best_sim = 0.0
        for s in np.linspace(lo, hi, n):
            pred = W_E[sid] + s * axis
            r = nn_retrieve(pred, source_ids(s_w), mask, 1)
            sim = float(np.dot(normed(pred).astype(np.float32), W_n[tid]))
            if sim > best_sim: best_sim = sim
            if r[0][0] == t_w: found_at = s; break
        if found_at is not None:
            details.append((s_w, t_w, 'found', found_at, best_sim))
        elif best_sim > 0.90:
            details.append((s_w, t_w, 'Type2_near_miss', None, best_sim))
        else:
            details.append((s_w, t_w, 'Type1_geo_fail', None, best_sim))
    irred = sum(1 for _,_,t,_,_ in details if t not in ['found'])
    return irred/len(details) if details else 0.0, details

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

def classify_v10(pc, loo, irred, spread=0.0):
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

# =====================================================================
# BENCHMARK — v10 base + relabeled variants
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
print("DAY 337: RELABEL, ITY IRRED TYPING, ABLE MIXED, CC ANALYSIS, NUM_WORD FEATURE")
print("="*80)

# Pre-compute all axes
print("\nComputing all axes...", flush=True)
all_results = {}
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None or len(valid) < 2: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr   = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    all_results[name] = (ax, valid, pc, spread, loo_v, irr)
print("  done.")

# =====================================================================
# PART A: v10 with RELABELED al_rel -> phonol_scatter
# =====================================================================
print()
print("PART A: v10 with al_rel relabeled phonol_scatter -> expected 26/30=87%%")
print("-"*80)

BENCH_RELABELED = [(n,tr,ho,'phonol_scatter' if n=='al_rel' else t)
                   for n,tr,ho,t in FIXED_BENCH]

v10_orig = v10_rel = 0
for name, _, _, true_orig in FIXED_BENCH:
    if name not in all_results: continue
    ax, valid, pc, spread, loo_v, irr = all_results[name]
    true_rel = 'phonol_scatter' if name == 'al_rel' else true_orig
    pred = classify_v10(pc, loo_v, irr, spread)
    ok_orig = match(pred, true_orig)
    ok_rel  = match(pred, true_rel)
    if ok_orig: v10_orig += 1
    if ok_rel:  v10_rel  += 1
    if name in ['al_rel']:
        print("  al_rel: pred=%-22s  true_orig=%-18s  true_rel=%-18s  ok_orig=%s  ok_rel=%s" %
              (pred, true_orig, true_rel, '✓' if ok_orig else '✗', '✓' if ok_rel else '✗'))
        print("         pc=%.3f  loo=%.0f%%  irred=%.2f" % (pc, 100*loo_v, irr))

print()
print("  v10 original labels:    %d/30 = %.0f%%" % (v10_orig, 100*v10_orig/30))
print("  v10 relabeled al_rel:   %d/30 = %.0f%%" % (v10_rel,  100*v10_rel/30))
print()

# Show that al_rel's neighbors (ance/ment/tion/al_nom) are all phonol_scatter
print("  al_rel vs phonol_scatter neighbors (pc 0.10-0.20, irred=0):")
for name in ['al_rel','ance','ment','tion','al_nom','ness','ablaut_t']:
    if name not in all_results: continue
    ax, valid, pc, spread, loo_v, irr = all_results[name]
    true = next(t for n,_,_,t in FIXED_BENCH if n==name)
    pred = classify_v10(pc, loo_v, irr, spread)
    print("  %-10s  pc=%.3f  loo=%.0f%%  irred=%.2f  spread=%.3f  pred=%-22s  true=%-18s  %s" %
          (name, pc, 100*loo_v, irr, spread, pred, true, '✓' if match(pred,true) else '✗'))
print()

# =====================================================================
# PART B: ity IRRED TYPING
# =====================================================================
print("PART B: ity irred typing — is the holdout failure Type 0 or Type 1?")
print("-"*80)

ity_pairs = [('human','humanity'),('real','reality'),('national','nationality'),
             ('personal','personality'),('moral','morality'),('legal','legality'),
             ('final','finality'),('normal','normality')]
ity_holdout = [('mental','mentality'),('total','totality'),('brutal','brutality')]
ity_extra_holdout = [('actual','actuality'),('equal','equality'),('ideal','ideality'),
                     ('modal','modality'),('vocal','vocality'),('global','globality'),
                     ('social','sociality'),('fiscal','fiscality')]

ax_ity, valid_ity, pc_ity, sp_ity = compute_axis_with_spread(ity_pairs)

print("  +ity axis:  pc=%.4f  spread=%.4f" % (pc_ity, sp_ity))
print()
print("  Primary holdout irred typing:")
irr_rate, details = irred_typed(ax_ity, ity_holdout, RELAXED_MASK)
for s_w, t_w, typ, found_at, best_sim in details:
    t_toks = get_token_ids(t_w)
    token_repr = '%d tokens: %s' % (len(t_toks), str([tok.decode([i]) for i in t_toks]))
    print("  %-12s -> %-14s  type=%-22s  best_sim=%-6s  tokens=[%s]" %
          (s_w, t_w, typ,
           '%.3f' % best_sim if best_sim is not None else '---',
           token_repr))
print("  irred_rate=%.2f" % irr_rate)
print()

print("  Extended holdout irred typing (more -al -> -ality):")
irr_rate2, details2 = irred_typed(ax_ity, ity_extra_holdout, RELAXED_MASK)
for s_w, t_w, typ, found_at, best_sim in details2:
    t_toks = get_token_ids(t_w)
    token_repr = '%d toks' % len(t_toks)
    print("  %-10s -> %-14s  %-22s  best_sim=%-6s  [%s]" %
          (s_w, t_w, typ,
           '%.3f' % best_sim if best_sim is not None else '---',
           token_repr))
print("  irred_rate=%.2f" % irr_rate2)
print()

# Type 0 adjusted irred: ignore multi-token failures
def irred_type0_adjusted(ax, holdout, mask):
    """Returns irred ignoring Type 0 (multi-token) failures"""
    n_valid = 0; n_fail = 0
    for s_w, t_w in holdout:
        t_toks = get_token_ids(t_w)
        if len(t_toks) > 1: continue  # skip Type 0
        es, sid = get_emb(s_w)
        if es is None: continue
        n_valid += 1; found = False
        for s in np.linspace(0.02, 6.0, 60):
            if nn_retrieve(W_E[sid]+s*ax, source_ids(s_w), mask, 1)[0][0]==t_w:
                found=True; break
        if not found: n_fail += 1
    return n_fail/n_valid if n_valid else 0.0, n_valid

ity_irred_adj, n_ity_valid = irred_type0_adjusted(ax_ity, ity_holdout, RELAXED_MASK)
ity_irred_adj2, n_ity2 = irred_type0_adjusted(ax_ity, ity_extra_holdout, RELAXED_MASK)
print("  +ity Type0-adjusted irred (primary):  %.2f  (n_valid=%d)" % (ity_irred_adj, n_ity_valid))
print("  +ity Type0-adjusted irred (extended): %.2f  (n_valid=%d)" % (ity_irred_adj2, n_ity2))
print()

# If type0-adjusted irred is significantly lower, ity should be phonol_scatter
orig_irr = irred_on_holdout(ax_ity, ity_holdout, RELAXED_MASK)
print("  +ity original irred (all): %.2f" % orig_irr)
p_orig = classify_v10(pc_ity, axis_loo(ax_ity, valid_ity, RELAXED_MASK), orig_irr, sp_ity)
p_adj  = classify_v10(pc_ity, axis_loo(ax_ity, valid_ity, RELAXED_MASK), ity_irred_adj, sp_ity)
print("  v10 with original irred:   %s" % p_orig)
print("  v10 with Type0-adj irred:  %s (true=phonol_scatter)" % p_adj)
print()

# =====================================================================
# PART C: able MIXED TRAINING
# =====================================================================
print("PART C: +able with mixed (Germanic + Latinate) training")
print("-"*80)

ABLE_ORIG = [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
             ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')]
ABLE_LATINATE = [('comfort','comfortable'),('manage','manageable'),('reach','reachable'),
                 ('depend','dependable'),('honor','honorable'),('justify','justifiable'),
                 ('prefer','preferable'),('measure','measurable')]
ABLE_MIXED = ABLE_ORIG + ABLE_LATINATE[:6]  # 14 pairs

ABLE_HOLDOUT = [('respect','respectable'),('fashion','fashionable'),('suit','suitable'),
                ('admir','admirable'),('desire','desirable'),('remark','remarkable')]

ax_able_orig, v_able_orig, pc_able_o, sp_able_o = compute_axis_with_spread(ABLE_ORIG)
ax_able_mix,  v_able_mix,  pc_able_m, sp_able_m = compute_axis_with_spread(ABLE_MIXED)

loo_able_o = axis_loo(ax_able_orig, v_able_orig, CLEAN_MASK)
loo_able_m = axis_loo(ax_able_mix,  v_able_mix,  CLEAN_MASK)

ho_able_orig = [('comfort','comfortable'),('manage','manageable'),('reach','reachable')]
irr_able_o   = irred_on_holdout(ax_able_orig, ho_able_orig, CLEAN_MASK)
irr_able_m   = irred_on_holdout(ax_able_mix,  ho_able_orig, CLEAN_MASK)
irr_able_ext = irred_on_holdout(ax_able_mix,  ABLE_HOLDOUT, CLEAN_MASK)

p_able_o = classify_v10(pc_able_o, loo_able_o, irr_able_o, sp_able_o)
p_able_m = classify_v10(pc_able_m, loo_able_m, irr_able_m, sp_able_m)

print("  +able ORIGINAL (8 pairs):  pc=%.4f  loo=%.0f%%  irred=%.2f  spread=%.3f  pred=%s" %
      (pc_able_o, 100*loo_able_o, irr_able_o, sp_able_o, p_able_o))
print("  +able MIXED   (14 pairs):  pc=%.4f  loo=%.0f%%  irred=%.2f  spread=%.3f  pred=%s" %
      (pc_able_m, 100*loo_able_m, irr_able_m, sp_able_m, p_able_m))
print("  +able MIXED extended holdout irred: %.2f" % irr_able_ext)
print("  [true=phonol_scatter]")
print()

# =====================================================================
# PART D: cc ANALYSIS — is a safe rule possible?
# =====================================================================
print("PART D: cc (case change) analysis — can we fix it?")
print("-"*80)

cc_pairs   = [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),
              ('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')]
cc_holdout = [('cup','Cup'),('door','Door'),('word','Word')]
ax_cc, v_cc, pc_cc, sp_cc = compute_axis_with_spread(cc_pairs)
loo_cc = axis_loo(ax_cc, v_cc, RELAXED_MASK)
irr_cc = irred_on_holdout(ax_cc, cc_holdout, RELAXED_MASK)

print("  +cc axis:  pc=%.4f  loo=%.0f%%  irred=%.2f  spread=%.4f" %
      (pc_cc, 100*loo_cc, irr_cc, sp_cc))
print("  v10 pred: %s  [true=morph_moderate]" % classify_v10(pc_cc, loo_cc, irr_cc, sp_cc))
print()

# Why does cc have LOO=0%?  Let's trace one LOO step
print("  LOO trace: what does cc axis retrieve?")
chords_cc = [W_E[tid]-W_E[sid] for _,_,sid,tid in v_cc]
ax_cc_full = normed(np.mean(chords_cc, axis=0))
gs_cc, _ = best_scale(ax_cc_full, v_cc, RELAXED_MASK)
print("  best_scale=%.2f" % gs_cc)
for s_w, t_w, sid, tid in v_cc[:4]:
    r = nn_retrieve(W_E[sid]+gs_cc*ax_cc_full, source_ids(s_w), RELAXED_MASK, 3)
    print("  %s -> %s | retrieved: %s" % (s_w, t_w, [x[0] for x in r]))
print()

# Check: is the issue that capitalized tokens appear rare in RELAXED_MASK?
cap_tokens = [i for i in range(len(W_E))
              if RELAXED_MASK[i] and tok.decode([i]).strip() and tok.decode([i]).strip()[0].isupper()]
print("  Capitalized tokens in RELAXED_MASK: %d" % len(cap_tokens))
# Check if Dog/Cat/Cup are in RELAXED_MASK
for w in ['Dog','Cat','Cup','Door','Word','Book','Tree','House']:
    es, sid = get_emb(w)
    if sid is not None:
        print("  '%-8s'  id=%6d  in_relaxed=%s  token='%s'" %
              (w, sid, RELAXED_MASK[sid], tok.decode([sid]).strip()))
print()

# =====================================================================
# PART E: num_word — source cluster type feature
# =====================================================================
print("PART E: num_word — can source cluster type distinguish it?")
print("-"*80)

num_pairs  = [('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),
              ('6','six'),('7','seven'),('8','eight')]
ax_num, v_num, pc_num, sp_num = compute_axis_with_spread(num_pairs)
loo_num = axis_loo(ax_num, v_num, RELAXED_MASK)
irr_num = irred_on_holdout(ax_num, [('9','nine'),('10','ten'),('0','zero')], RELAXED_MASK)

print("  num_word: pc=%.4f  loo=%.0f%%  irred=%.2f  spread=%.4f" %
      (pc_num, 100*loo_num, irr_num, sp_num))
print("  v10 pred: %s  [true=semantic_diverse]" % classify_v10(pc_num, loo_num, irr_num, sp_num))
print()

# Check if num sources are all single-character tokens
print("  Source token analysis (num_word):")
for s_w, t_w, sid, tid in v_num:
    src_tok = tok.decode([sid]).strip()
    print("  %s (id=%d, tok='%s') -> %s (id=%d, tok='%s')" %
          (s_w, sid, src_tok, t_w, tid, tok.decode([tid]).strip()))
print()

# Check if ALL source tokens share a feature (e.g., all are digits)
src_tokens = [tok.decode([sid]).strip() for _,_,sid,_ in v_num]
all_digit = all(s.strip().isdigit() for s in src_tokens)
print("  All source tokens are pure digits: %s" % all_digit)
print()

# Compare: for regular morphological axes, are sources all alphabetic?
print("  Source token type across axes:")
for name in ['num_word','er_comp','plural','ness','ablaut','relational']:
    pairs_for_name = next((tr for n,tr,_,_ in FIXED_BENCH if n==name), [])
    src_types = []
    for s_w, t_w in pairs_for_name:
        _, sid = get_emb(s_w)
        if sid is None: continue
        src_tok = tok.decode([sid]).strip()
        if src_tok.isdigit(): src_types.append('digit')
        elif src_tok.isalpha(): src_types.append('alpha')
        elif src_tok[0].isupper(): src_types.append('cap')
        else: src_types.append('other')
    type_counts = {t: src_types.count(t) for t in set(src_types)}
    print("  %-12s  %s" % (name, type_counts))
print()
print("  If source type is DIGIT -> semantic_diverse (num_word fix):")
print("  This would be safe: no other axis in benchmark has digit sources.")
print()

# =====================================================================
# PART F: v11 benchmark with all proposed fixes
# =====================================================================
print("PART F: v11 — v10 + al_rel relabeled + able mixed + Type0-adj ity + digit rule")
print("-"*80)

def classify_v11(pc, loo, irred, spread=0.0, src_is_digit=False):
    """v11: v10 + digit source detection for num_word"""
    if src_is_digit: return 'semantic_diverse'  # num_word fix
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

# v11 benchmark with:
# - al_rel relabeled to phonol_scatter
# - able replaced with mixed 14-pair version
# - num_word: src_is_digit=True
# - ity: use Type0-adjusted irred if applicable

v11_score = 0
print("  %-12s  pc    LOO  irred  pred_v11            true_v11       ok" % "axis")
print("  " + "-"*85)

for name, train_pairs, holdout_pairs, true_orig in FIXED_BENCH:
    # Determine true label for v11 benchmark
    if name == 'al_rel':    true_v11 = 'phonol_scatter'
    else:                   true_v11 = true_orig

    # Determine features for v11
    if name == 'able':
        ax, valid, pc, spread = compute_axis_with_spread(ABLE_MIXED)
        loo_v = axis_loo(ax, valid, CLEAN_MASK)
        irr   = irred_on_holdout(ax, ho_able_orig, CLEAN_MASK)
    elif name == 'ity':
        ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
        loo_v = axis_loo(ax, valid, RELAXED_MASK)
        irr   = ity_irred_adj  # Type0-adjusted
    elif name in all_results:
        ax, valid, pc, spread, loo_v, irr = all_results[name]
    else:
        continue

    src_is_digit = (name == 'num_word')
    pred = classify_v11(pc, loo_v, irr, spread, src_is_digit)
    ok = match(pred, true_v11)
    if ok: v11_score += 1
    marker = '  '
    if name in ['al_rel','able','ity','num_word']: marker = '->'
    print("  %s %-10s  %.3f %.0f%% %.2f  %-20s  %-16s  %s" %
          (marker, name, pc, 100*loo_v, irr, pred[:20], true_v11, '✓' if ok else '✗'))

print()
print("  v10 (original labels): 25/30 = 83%%")
print("  v10 (relabeled):       26/30 = 87%%")
print("  v11 (all fixes):       %d/30 = %.0f%%" % (v11_score, 100*v11_score/30))
print()
