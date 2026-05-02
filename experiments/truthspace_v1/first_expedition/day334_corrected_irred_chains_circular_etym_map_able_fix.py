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

def is_single_token(word):
    for p in [' '+word, word]:
        ids = tok(p, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return True
    return False

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

def irred_corrected(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    """Returns (irred_raw, irred_corrected, type0, type1, details)."""
    raw=0; type0=0; type1_geom=0; n_ho=0; details=[]
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1
        # Check if target is single-token
        tgt_single = is_single_token(t_w)
        found_at = None; best_sim = 0.0
        for s in np.linspace(lo, hi, n):
            r = nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask, 1)
            if r[0][0] == t_w: found_at=s; break
        if found_at is None:
            raw += 1
            et, _ = get_emb(t_w)
            if not tgt_single:
                type0 += 1  # vocabulary-limited
                irr_type = 'Type0(vocab)'
            else:
                # Measure best_sim for direction diagnosis
                if et is not None:
                    for s in np.linspace(lo, hi, n):
                        pred = W_E[sid] + s * axis
                        sim = float(np.dot(normed(pred).astype(np.float32),
                                           normed(et).astype(np.float32)))
                        if sim > best_sim: best_sim = sim
                type1_geom += 1
                irr_type = 'Type1(geom,sim=%.3f)' % best_sim
            details.append((s_w, t_w, 'IRRED', irr_type))
        else:
            details.append((s_w, t_w, 'OK@%.2f'%found_at, 'OK'))
    irred_raw  = raw/n_ho if n_ho else 0.0
    irred_corr = type1_geom/n_ho if n_ho else 0.0
    return irred_raw, irred_corr, type0, type1_geom, n_ho, details

print()
print("DAY 334: CORRECTED IRRED BENCHMARK, VALID CHAINS, CIRCULAR +al_rel, ETYM MAP, +able FIX")
print("="*80)
print()

# =====================================================================
# PART A: v6 BENCHMARK WITH CORRECTED IRRED
# =====================================================================
print("PART A: v6 benchmark with corrected irred (irred_corrected ignores Type0/vocab failures)")
print("-"*80)

def classify_v6(pc, loo, irred, n_train=8):
    if pc > 0.35:
        return 'morph_uniform/relational_geom'
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

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

FIXED_BENCH = [
    ('er_comp',
     [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),
      ('warm','warmer'),('long','longer'),('cold','colder')],
     [('dark','darker'),('soft','softer'),('heavy','heavier')], 'morph_uniform'),
    ('er_sup',
     [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),
      ('bright','brightest'),('dark','darkest'),('soft','softest')],
     [('warm','warmest'),('long','longest'),('cold','coldest')], 'morph_uniform'),
    ('relational',
     [('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),
      ('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],
     [('Cairo','Egypt'),('Seoul','Korea'),('Lima','Peru')], 'relational_geom'),
    ('al_rel',
     [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),
      ('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')],
     [('history','historical'),('season','seasonal'),('accident','accidental')], 'relational_geom'),
    ('plural',
     [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),
      ('book','books'),('bird','birds'),('door','doors')],
     [('cup','cups'),('word','words'),('room','rooms')], 'morph_moderate'),
    ('3ps',
     [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),
      ('write','writes'),('play','plays'),('work','works')],
     [('talk','talks'),('sleep','sleeps'),('open','opens')], 'morph_moderate'),
    ('ed_reg',
     [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played'),
      ('clean','cleaned'),('open','opened'),('start','started')],
     [('end','ended'),('look','looked'),('rain','rained')], 'morph_moderate'),
    ('ing',
     [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),
      ('make','making'),('write','writing'),('read','reading')],
     [('eat','eating'),('work','working'),('play','playing')], 'morph_moderate'),
    ('cc',
     [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car'),
      ('tree','Tree'),('river','River'),('bird','Bird')],
     [('cup','Cup'),('door','Door'),('word','Word')], 'morph_moderate'),
    ('ness',
     [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),
      ('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],
     [('bright','brightness'),('sweet','sweetness'),('clean','cleanliness')], 'phonol_scatter'),
    ('ablaut',
     [('go','went'),('take','took'),('give','gave'),('see','saw'),('know','knew'),
      ('drive','drove'),('write','wrote'),('ride','rode')],
     [('speak','spoke'),('break','broke'),('choose','chose')], 'phonol_scatter'),
    ('ablaut_t',
     [('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left'),
      ('deal','dealt'),('sleep','slept'),('mean','meant')],
     [('burn','burned'),('learn','learned'),('smell','smelled')], 'phonol_scatter'),
    ('ity',
     [('human','humanity'),('real','reality'),('national','nationality'),('personal','personality'),
      ('moral','morality'),('legal','legality'),('final','finality'),('normal','normality')],
     [('mental','mentality'),('total','totality'),('brutal','brutality')], 'phonol_scatter'),
    ('un_neg',
     [('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),
      ('known','unknown'),('safe','unsafe'),('usual','unusual'),('equal','unequal')],
     [('stable','unstable'),('real','unreal'),('true','untrue')], 'phonol_scatter'),
    ('ance',
     [('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),
      ('accept','acceptance'),('appear','appearance'),('depend','dependence'),('insist','insistence')],
     [('persist','persistence'),('emerge','emergence'),('refer','reference')], 'phonol_scatter'),
    ('ment',
     [('achieve','achievement'),('develop','development'),('manage','management'),('govern','government'),
      ('engage','engagement'),('require','requirement'),('move','movement'),('improve','improvement')],
     [('amuse','amusement'),('punish','punishment'),('treat','treatment')], 'phonol_scatter'),
    ('tion',
     [('act','action'),('direct','direction'),('educate','education'),('create','creation'),
      ('produce','production'),('relate','relation'),('combine','combination'),('apply','application')],
     [('express','expression'),('extend','extension'),('omit','omission')], 'phonol_scatter'),
    ('al_nom',
     [('arrive','arrival'),('propose','proposal'),('approve','approval'),('refuse','refusal'),
      ('remove','removal'),('survive','survival'),('deny','denial'),('dispose','disposal')],
     [('retrieve','retrieval'),('betray','betrayal'),('renew','renewal')], 'phonol_scatter'),
    ('less',
     [('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),
      ('end','endless'),('home','homeless'),('harm','harmless'),('power','powerless')],
     [('worth','worthless'),('use','useless'),('mercy','merciless')], 'phonol_scatter'),
    ('ful',
     [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),
      ('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')],
     [('beauty','beautiful'),('wonder','wonderful'),('power','powerful')], 'phonol_scatter'),
    ('able',
     [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
      ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')],
     [('comfort','comfortable'),('manage','manageable'),('reach','reachable')], 'phonol_scatter'),
    ('er_noun',
     [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),
      ('own','owner'),('manage','manager'),('build','builder'),('lead','leader')],
     [('write','writer'),('paint','painter'),('print','printer')], 'semantic_diverse'),
    ('adj_ant',
     [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),
      ('bright','dark'),('hard','soft'),('high','low'),('rich','poor')],
     [('open','closed'),('new','old'),('loud','quiet')], 'polar_local'),
    ('antonym2',
     [('love','hate'),('war','peace'),('life','death'),('day','night'),
      ('begin','end'),('give','take'),('push','pull'),('open','close')],
     [('rise','fall'),('win','lose'),('buy','sell')], 'polar_local'),
    ('en_es',
     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día'),
      ('night','noche'),('hand','mano'),('year','año')],
     [('fire','fuego'),('moon','luna'),('sea','mar')], 'translation'),
    ('en_de',
     [('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),('day','Tag'),
      ('night','Nacht'),('cat','Katze'),('dog','Hund')],
     [('fire','Feuer'),('moon','Mond'),('sea','Meer')], 'translation'),
    ('en_fr',
     [('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),('day','jour'),
      ('night','nuit'),('cat','chat'),('dog','chien')],
     [('fire','feu'),('moon','lune'),('sea','mer')], 'translation'),
    ('en_zh',
     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),
      ('hand','手'),('eye','眼'),('fish','鱼')],
     [('tree','树'),('heart','心'),('door','门')], 'factual_local'),
    ('en_ja',
     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),
      ('hand','手'),('eye','目'),('fish','魚')],
     [('tree','木'),('heart','心'),('door','門')], 'factual_local'),
    ('num_word',
     [('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),
      ('6','six'),('7','seven'),('8','eight')],
     [('9','nine'),('10','ten'),('0','zero')], 'semantic_diverse'),
]

v6_raw = 0; v6_corr = 0
print("  %-12s  pc    LOO  irred_raw irred_corr  pred_raw       pred_corr      true       raw? corr?" % "axis")
print("  " + "-"*105)
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2: continue
    loo_v  = axis_loo(ax, valid, RELAXED_MASK)
    irr_raw, irr_corr, t0, t1, n_ho, dets = irred_corrected(ax, holdout_pairs, RELAXED_MASK)
    pred_raw  = classify_v6(pc, loo_v, irr_raw)
    pred_corr = classify_v6(pc, loo_v, irr_corr)
    ok_raw  = match(pred_raw, true_type)
    ok_corr = match(pred_corr, true_type)
    if ok_raw:  v6_raw  += 1
    if ok_corr: v6_corr += 1
    tick_r = '✓' if ok_raw  else '✗'
    tick_c = '✓' if ok_corr else '✗'
    changed = '*' if ok_raw != ok_corr else ' '
    print("  %s/%s%s %-10s  %.3f  %.0f%%  %.2f      %.2f        %-14s %-14s %-12s" %
          (tick_r, tick_c, changed, name, pc, 100*loo_v, irr_raw, irr_corr,
           pred_raw[:14], pred_corr[:14], true_type))
print()
print("  v6 (raw irred):  %d/30 = %.0f%%" % (v6_raw,  100*v6_raw/30))
print("  v6 (corr irred): %d/30 = %.0f%%" % (v6_corr, 100*v6_corr/30))
print()

# =====================================================================
# PART B: VALID MORPHOLOGICAL CHAIN TESTS
# =====================================================================
print("PART B: Valid chain tests (linguistically compatible suffix chains)")
print("-"*80)

# Build axes
ax_ize, valid_ize, _ = compute_axis(
    [('organ','organize'),('moral','moralize'),('legal','legalize'),('real','realize'),
     ('local','localize'),('final','finalize'),('general','generalize'),('national','nationalize')])
ax_tion, valid_tion, _ = compute_axis(
    [('act','action'),('direct','direction'),('educate','education'),('create','creation'),
     ('produce','production'),('relate','relation'),('combine','combination'),('apply','application')])
ax_alrel, valid_alrel, _ = compute_axis(
    [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),
     ('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')])
ax_ity, valid_ity, _ = compute_axis(
    [('human','humanity'),('real','reality'),('national','nationality'),('personal','personality'),
     ('moral','morality'),('legal','legality'),('final','finality'),('normal','normality')])
ax_ness, valid_ness, _ = compute_axis(
    [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),
     ('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')])
ax_3ps, valid_3ps, _ = compute_axis(
    [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),
     ('write','writes'),('play','plays'),('work','works')])
ax_ment, valid_ment, _ = compute_axis(
    [('achieve','achievement'),('develop','development'),('manage','management'),
     ('govern','government'),('engage','engagement'),('require','requirement'),
     ('move','movement'),('improve','improvement')])
ax_ing, valid_ing, _ = compute_axis(
    [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),
     ('make','making'),('write','writing'),('read','reading')])

bs_ize  = best_scale(ax_ize, valid_ize, CLEAN_MASK)[0]
bs_tion = best_scale(ax_tion, valid_tion, CLEAN_MASK)[0]
bs_alrel= best_scale(ax_alrel, valid_alrel, CLEAN_MASK)[0]
bs_ity  = best_scale(ax_ity, valid_ity, CLEAN_MASK)[0]
bs_ness = best_scale(ax_ness, valid_ness, CLEAN_MASK)[0]
bs_3ps  = best_scale(ax_3ps, valid_3ps, CLEAN_MASK)[0]
bs_ment = best_scale(ax_ment, valid_ment, CLEAN_MASK)[0]
bs_ing  = best_scale(ax_ing, valid_ing, CLEAN_MASK)[0]
print("  Scales: ize=%.2f  tion=%.2f  al_rel=%.2f  ity=%.2f  ness=%.2f  3ps=%.2f  ment=%.2f  ing=%.2f" %
      (bs_ize, bs_tion, bs_alrel, bs_ity, bs_ness, bs_3ps, bs_ment, bs_ing))
print()

# Chain 1: adj -> +ize -> +tion (Latin)
print("  Chain 1: adj(Latin) -> +ize(verb) -> +tion(event_noun)")
chain1_adjs = [('local','localize','localization'),('real','realize','realization'),
                ('final','finalize','finalization'),('legal','legalize','legalization'),
                ('general','generalize','generalization'),('moral','moralize','moralization'),
                ('national','nationalize','nationalization')]
hits_c1 = 0
for adj, verb, noun in chain1_adjs:
    es, sid = get_emb(adj)
    if es is None: continue
    step1 = W_E[sid] + bs_ize * ax_ize
    r1 = nn_retrieve(step1, source_ids(adj), CLEAN_MASK, 1)
    got_v = r1[0][0]; v1_id = r1[0][2]
    step2 = W_E[v1_id] + bs_tion * ax_tion
    r2 = nn_retrieve(step2, source_ids(got_v), CLEAN_MASK, 1)
    got_n = r2[0][0]
    ok1 = '✓' if got_v == verb else '~'
    ok2 = '✓' if got_n == noun else '~'
    if got_n == noun: hits_c1 += 1
    print("  %s/%s %-10s -> %-14s -> %-18s [expected: %s]" %
          (ok1, ok2, adj, got_v, got_n, noun))
print("  Chain 1 end-to-end: %d/%d" % (hits_c1, len(chain1_adjs)))
print()

# Chain 2: verb -> +3ps -> can we get a gerund from the 3ps form?
print("  Chain 2: verb -> +3ps -> +ment (action_noun from 3ps?)")
chain2_verbs = [('run','runs'),('move','moves'),('improve','improves'),('develop','develops')]
for verb, verb_3ps in chain2_verbs:
    ev, sid = get_emb(verb)
    if ev is None: continue
    step1 = W_E[sid] + bs_3ps * ax_3ps
    r1 = nn_retrieve(step1, source_ids(verb), CLEAN_MASK, 1)
    got_3ps = r1[0][0]; v1_id = r1[0][2]
    step2 = W_E[v1_id] + bs_ment * ax_ment
    r2 = nn_retrieve(step2, source_ids(got_3ps), CLEAN_MASK, 1)
    ok1 = '✓' if got_3ps == verb_3ps else '~'
    print("  %s %-10s -> %-12s -> %-15s" % (ok1, verb, got_3ps, r2[0][0]))
print()

# Chain 3: verb -> +ing -> usage as noun (gerund) — single-step test
print("  Chain 3: verb -> +ing (gerund) — does gerund navigate like a noun?")
ax_er_noun, valid_er_noun, _ = compute_axis(
    [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),
     ('own','owner'),('manage','manager'),('build','builder'),('lead','leader')])
bs_er_noun = best_scale(ax_er_noun, valid_er_noun, CLEAN_MASK)[0]
test_verb_ings = [('run','running'),('write','writing'),('build','building'),('manage','managing')]
for verb, gerund in test_verb_ings:
    ev, sid = get_emb(verb)
    if ev is None: continue
    step1 = W_E[sid] + bs_ing * ax_ing
    r1 = nn_retrieve(step1, source_ids(verb), CLEAN_MASK, 1)
    got_ing = r1[0][0]; v1_id = r1[0][2]
    # Now apply er_noun to the gerund — does "running" -> "runner"?
    step2 = W_E[v1_id] + bs_er_noun * ax_er_noun
    r2 = nn_retrieve(step2, source_ids(got_ing), CLEAN_MASK, 1)
    ok1 = '✓' if got_ing == gerund else '~'
    print("  %s %-10s -> %-12s -> %-12s" % (ok1, verb, got_ing, r2[0][0]))
print()

# =====================================================================
# PART C: +al_rel CIRCULAR CHAIN
# =====================================================================
print("PART C: +al_rel circular chain — noun -> adj -> back_to_noun?")
print("-"*80)

# Test 1: noun -> +al_rel -> adj -> +ity -> back_to_quality_noun
print("  Test 1: noun -> (+al_rel) -> adj -> (+ity) -> quality_noun")
print("  [Expected: nation->national->nationality, person->personal->personality]")
test_nouns = [('nation','national','nationality'),('person','personal','personality'),
               ('region','regional','regionality?'),('moral','moral','morality'),
               ('culture','cultural','?')]
for noun, adj_exp, noun2_exp in test_nouns:
    en, sid = get_emb(noun)
    if en is None: continue
    step1 = W_E[sid] + bs_alrel * ax_alrel
    r1 = nn_retrieve(step1, source_ids(noun), RELAXED_MASK, 1)
    got_adj = r1[0][0]; a1_id = r1[0][2]
    step2 = W_E[a1_id] + bs_ity * ax_ity
    r2 = nn_retrieve(step2, source_ids(got_adj), CLEAN_MASK, 1)
    got_n2 = r2[0][0]
    ok1 = '✓' if got_adj == adj_exp else '~'
    print("  %s %-10s -> %-12s -> %-14s [expected: %s]" %
          (ok1, noun, got_adj, got_n2, noun2_exp))
print()

# Test 2: adj -> +ity -> noun -> -al_rel (reversed) -> back_to_adj?
print("  Test 2: adj -> (+ity) -> noun -> (-al_rel reverse) -> back_to_adj?")
print("  [This should be circular: national->nationality->back_to_national?]")
ax_alrel_rev = -ax_alrel
bs_alrel_rev = best_scale(ax_alrel_rev, valid_alrel, CLEAN_MASK)[0]
test_adjs = [('national','nationality'),('personal','personality'),
              ('moral','morality'),('legal','legality'),('final','finality')]
for adj, noun_exp in test_adjs:
    ea, aid = get_emb(adj)
    if ea is None: continue
    step1 = W_E[aid] + bs_ity * ax_ity
    r1 = nn_retrieve(step1, source_ids(adj), CLEAN_MASK, 1)
    got_n = r1[0][0]; n1_id = r1[0][2]
    step2 = W_E[n1_id] + bs_alrel_rev * ax_alrel_rev
    r2 = nn_retrieve(step2, source_ids(got_n), CLEAN_MASK, 1)
    got_back = r2[0][0]
    ok1 = '✓' if got_n == noun_exp else '~'
    ok2 = '✓' if got_back == adj else '~'
    print("  %s/%s %-12s -> %-14s -> %-12s [should return: %s]" %
          (ok1, ok2, adj, got_n, got_back, adj))
print()

# Test 3: noun -> +al_rel -> adj -> -al_rel (reverse) -> back_to_noun?
print("  Test 3: noun -> (+al_rel) -> adj -> (-al_rel) -> back_to_noun?")
test_nouns3 = [('nation','national'),('person','personal'),('emotion','emotional'),
                ('tradition','traditional'),('origin','original')]
for noun, adj_exp in test_nouns3:
    en, sid = get_emb(noun)
    if en is None: continue
    step1 = W_E[sid] + bs_alrel * ax_alrel
    r1 = nn_retrieve(step1, source_ids(noun), RELAXED_MASK, 1)
    got_adj = r1[0][0]; a1_id = r1[0][2]
    step2 = W_E[a1_id] + bs_alrel_rev * ax_alrel_rev
    r2 = nn_retrieve(step2, source_ids(got_adj), CLEAN_MASK, 1)
    got_back = r2[0][0]
    ok1 = '✓' if got_adj == adj_exp else '~'
    ok2 = '✓' if got_back == noun else '~'
    print("  %s/%s %-12s -> %-12s -> %-12s [should return: %s]" %
          (ok1, ok2, noun, got_adj, got_back, noun))
print()

# =====================================================================
# PART D: ETYMOLOGICAL SUB-CLUSTER MAP OF adj SPACE
# =====================================================================
print("PART D: Etymology map — Germanic vs Latin adj clusters")
print("-"*80)

# Compute axes for classification
ax_germ_adj, _, _ = compute_axis(
    [('bright','brighter'),('dark','darker'),('soft','softer'),('hard','harder'),
     ('warm','warmer'),('cold','colder'),('deep','deeper'),('wide','wider')])  # +er_comp Germanic
ax_latin_adj, _, _ = compute_axis(
    [('human','humanity'),('real','reality'),('national','nationality'),('moral','morality'),
     ('legal','legality'),('final','finality'),('personal','personality'),('local','locality')])  # +ity Latin

if ax_germ_adj is not None and ax_latin_adj is not None:
    print("  Scoring adj vocabulary: cos(adj_emb, +er_comp) vs cos(adj_emb, +ity)")
    germ_adj_tokens = []
    latin_adj_tokens = []
    both_adj_tokens = []
    germ_thresh = 0.20
    latin_thresh = 0.20
    for i in range(len(W_E)):
        if not CLEAN_MASK[i]: continue
        w = tok.decode([i]).strip()
        if len(w) < 3: continue
        emb_n = W_n[i]
        cos_g = float(np.dot(emb_n, ax_germ_adj.astype(np.float32)))
        cos_l = float(np.dot(emb_n, ax_latin_adj.astype(np.float32)))
        if cos_g > germ_thresh and cos_l < 0.05:
            germ_adj_tokens.append((w, cos_g, cos_l))
        elif cos_l > latin_thresh and cos_g < 0.05:
            latin_adj_tokens.append((w, cos_g, cos_l))
        elif cos_g > 0.15 and cos_l > 0.15:
            both_adj_tokens.append((w, cos_g, cos_l))
    
    germ_adj_tokens.sort(key=lambda x: -x[1])
    latin_adj_tokens.sort(key=lambda x: -x[2])
    both_adj_tokens.sort(key=lambda x: -(x[1]+x[2]))
    
    print("  Top Germanic adj (high cos(+er_comp), low cos(+ity)):")
    for w, cg, cl in germ_adj_tokens[:20]:
        print("    %-14s  cos_germ=%+.3f  cos_latin=%+.3f" % (w, cg, cl))
    print()
    print("  Top Latin adj (high cos(+ity), low cos(+er_comp)):")
    for w, cg, cl in latin_adj_tokens[:20]:
        print("    %-14s  cos_germ=%+.3f  cos_latin=%+.3f" % (w, cg, cl))
    print()
    print("  Adj in BOTH clusters (high cos on both):")
    for w, cg, cl in both_adj_tokens[:10]:
        print("    %-14s  cos_germ=%+.3f  cos_latin=%+.3f" % (w, cg, cl))
    print()
    print("  Germanic count: %d  Latin count: %d  Both: %d" %
          (len(germ_adj_tokens), len(latin_adj_tokens), len(both_adj_tokens)))
print()

# =====================================================================
# PART E: FIXED +able AXIS WITH MIXED TRAINING
# =====================================================================
print("PART E: Fix +able with mixed Germanic+Latin training set")
print("-"*80)

ABLE_MIXED = [
    # Germanic (original training)
    ('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
    ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable'),
    # Latinate (new)
    ('comfort','comfortable'),('manage','manageable'),('reach','reachable'),
    ('note','notable'),('remark','remarkable'),('reason','reasonable'),
    ('prefer','preferable'),('rely','reliable'),
]
ABLE_HOLDOUT_MIXED = [
    ('adapt','adaptable'),('agree','agreeable'),('suit','suitable'),
    ('value','valuable'),('enjoy','enjoyable'),('honor','honorable'),
    ('depend','dependable'),('consider','considerable'),
]
ax_able_m, valid_able_m, pc_able_m = compute_axis(ABLE_MIXED)
if ax_able_m is not None:
    loo_m = axis_loo(ax_able_m, valid_able_m, CLEAN_MASK)
    irr_raw, irr_corr, t0, t1, n_ho, details = irred_corrected(ax_able_m, ABLE_HOLDOUT_MIXED, CLEAN_MASK)
    print("  +able MIXED axis: pc=%.4f  LOO=%.0f%%  irred_raw=%.0f%%  irred_corr=%.0f%%  n=%d" %
          (pc_able_m, 100*loo_m, 100*irr_raw, 100*irr_corr, len(valid_able_m)))
    print()
    print("  Holdout details:")
    for sw, tw, status, irr_type in details:
        ids_t = tok(' '+tw, add_special_tokens=False)['input_ids']
        print("  %-14s -> %-14s  %-20s  [%d tok]" % (sw, tw, status, len(ids_t)))
    print()

    # Compare with original Germanic-only +able:
    ABLE_GERM = [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
                  ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')]
    ax_able_g, valid_able_g, pc_able_g = compute_axis(ABLE_GERM)
    if ax_able_g is not None:
        loo_g = axis_loo(ax_able_g, valid_able_g, CLEAN_MASK)
        irr_g, _, _, _, _, _ = irred_corrected(ax_able_g, ABLE_HOLDOUT_MIXED, CLEAN_MASK)
        print("  +able GERMANIC-ONLY: pc=%.4f  LOO=%.0f%%  irred_raw=%.0f%%" %
              (pc_able_g, 100*loo_g, 100*irr_g))
        c_axes = float(np.dot(ax_able_m.astype(np.float32), ax_able_g.astype(np.float32)))
        print("  cos(mixed, germanic) = %.4f  (are they the same axis?)" % c_axes)
print()
