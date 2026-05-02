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

def source_centroid(words):
    """Compute normalized centroid of source word embeddings."""
    embs = []
    for w in words:
        e, _ = get_emb(w)
        if e is not None: embs.append(e)
    if not embs: return None
    return normed(np.mean(embs, axis=0))

print()
print("DAY 335: ETYMOLOGY CENTROID MAP, CHAIN DEEP-DIVE, V6 BOUNDARY, MULTI-SUB-AXIS")
print("="*80)
print()

# =====================================================================
# PART A: ETYMOLOGY SUB-CLUSTER MAP (CENTROID METHOD)
# =====================================================================
print("PART A: Etymology sub-cluster map using SOURCE CENTROIDS (corrected method)")
print("-"*80)

# Compute source centroids for key morphological groups
GERM_ADJ_SRC = ['bright','dark','warm','cold','deep','wide','soft','hard',
                  'fast','slow','tall','short','clean','heavy','young','old']
LATIN_ADJ_SRC = ['national','moral','legal','final','personal','local',
                   'general','real','mental','total','central','natural',
                   'original','cultural','regional','traditional']
GERM_VERB_SRC = ['run','walk','jump','eat','read','write','play','work',
                   'talk','sleep','sit','stand','go','give','take','make']
LATIN_VERB_SRC = ['organize','create','produce','educate','relate','combine',
                    'perform','appear','accept','require','achieve','manage',
                    'develop','improve','engage','exist']

cent_germ_adj  = source_centroid(GERM_ADJ_SRC).astype(np.float32)
cent_latin_adj = source_centroid(LATIN_ADJ_SRC).astype(np.float32)
cent_germ_verb = source_centroid(GERM_VERB_SRC).astype(np.float32)
cent_latin_verb = source_centroid(LATIN_VERB_SRC).astype(np.float32)

# Cross-centroid cosines
print("  Cross-centroid cosines:")
pairs_to_check = [('germ_adj', cent_germ_adj, 'latin_adj', cent_latin_adj),
                   ('germ_adj', cent_germ_adj, 'germ_verb', cent_germ_verb),
                   ('latin_adj', cent_latin_adj, 'latin_verb', cent_latin_verb),
                   ('germ_verb', cent_germ_verb, 'latin_verb', cent_latin_verb)]
for n1, c1, n2, c2 in pairs_to_check:
    print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, float(np.dot(c1, c2))))
print()

# Find top-N words near each centroid
def top_near_centroid(centroid, mask, n=25, excl_words=None):
    excl_words = excl_words or []
    excl_ids = set()
    for w in excl_words:
        for p in [' '+w, w]:
            ids = tok(p, add_special_tokens=False)['input_ids']
            if len(ids) == 1: excl_ids.add(ids[0])
    sims = W_n @ centroid
    sims[~mask] = -1.0
    for eid in excl_ids: sims[eid] = -1.0
    top = np.argpartition(sims, -n)[-n:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(tok.decode([i]).strip(), float(sims[i])) for i in top]

print("  Top 20 tokens near GERMANIC ADJ centroid:")
for w, s in top_near_centroid(cent_germ_adj, CLEAN_MASK, 20, GERM_ADJ_SRC):
    print("    %-14s  %.4f" % (w, s))
print()
print("  Top 20 tokens near LATIN ADJ centroid:")
for w, s in top_near_centroid(cent_latin_adj, CLEAN_MASK, 20, LATIN_ADJ_SRC):
    print("    %-14s  %.4f" % (w, s))
print()
print("  Top 20 tokens near GERMANIC VERB centroid:")
for w, s in top_near_centroid(cent_germ_verb, CLEAN_MASK, 20, GERM_VERB_SRC):
    print("    %-14s  %.4f" % (w, s))
print()
print("  Top 20 tokens near LATIN VERB centroid:")
for w, s in top_near_centroid(cent_latin_verb, CLEAN_MASK, 20, LATIN_VERB_SRC):
    print("    %-14s  %.4f" % (w, s))
print()

# Test words: are they classified correctly?
TEST_ADJ = [('bright','germ'),('dark','germ'),('warm','germ'),('cold','germ'),
             ('moral','latin'),('legal','latin'),('national','latin'),('personal','latin'),
             ('quick','germ'),('strong','germ'),('final','latin'),('general','latin'),
             ('human','latin'),('clean','germ'),('central','latin'),('deep','germ')]
print("  Test adj etymology classification (cos to germ_adj vs latin_adj centroid):")
for w, expected in TEST_ADJ:
    e, _ = get_emb(w)
    if e is None: continue
    en = normed(e).astype(np.float32)
    cg = float(np.dot(en, cent_germ_adj))
    cl = float(np.dot(en, cent_latin_adj))
    pred = 'germ' if cg > cl else 'latin'
    ok = '✓' if pred == expected else '✗'
    print("  %s %-12s  cos_germ=%+.4f  cos_latin=%+.4f  pred=%-6s  exp=%s" %
          (ok, w, cg, cl, pred, expected))
print()

# =====================================================================
# PART B: write->writing->writer CHAIN DEEP-DIVE
# =====================================================================
print("PART B: write->writing->writer chain — full test on all training verbs")
print("-"*80)

# Build the two axes
ax_ing, valid_ing, pc_ing = compute_axis(
    [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),
     ('make','making'),('write','writing'),('read','reading')])
ax_er_noun, valid_er, pc_er = compute_axis(
    [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),
     ('own','owner'),('manage','manager'),('build','builder'),('lead','leader')])

bs_ing = best_scale(ax_ing, valid_ing, CLEAN_MASK)[0]
bs_er  = best_scale(ax_er_noun, valid_er, CLEAN_MASK)[0]

print("  +ing axis:     pc=%.4f  scale=%.2f" % (pc_ing, bs_ing))
print("  +er_noun axis: pc=%.4f  scale=%.2f" % (pc_er, bs_er))
print()

# Test verbs: diverse set covering different POS families
TEST_CHAIN_VERBS = [
    # Training set members
    ('go', 'going', 'goer'),     # less common agent form
    ('take', 'taking', 'taker'), # taker exists
    ('run', 'running', 'runner'),
    ('see', 'seeing', 'seer'),
    ('give', 'giving', 'giver'),
    ('make', 'making', 'maker'),
    ('write', 'writing', 'writer'),
    ('read', 'reading', 'reader'),
    # Out-of-distribution
    ('teach', 'teaching', 'teacher'),
    ('farm', 'farming', 'farmer'),
    ('drive', 'driving', 'driver'),
    ('own', 'owning', 'owner'),
    ('build', 'building', 'builder'),
    ('lead', 'leading', 'leader'),
    ('manage', 'managing', 'manager'),
    ('work', 'working', 'worker'),
    ('print', 'printing', 'printer'),
    ('paint', 'painting', 'painter'),
    ('cook', 'cooking', 'cook'),    # cook is both verb and agent noun
    ('think', 'thinking', 'thinker'),
    ('swim', 'swimming', 'swimmer'),
    ('play', 'playing', 'player'),
    ('act', 'acting', 'actor'),      # actor not -er form
    ('direct', 'directing', 'director'),  # director not -er form
]

print("  %-10s  %-14s  %-12s  %-12s  ok1  ok2" % ('verb', 'expected_ing', 'got_ing', 'got_agent'))
chain_hits = 0; step1_hits = 0
for verb, exp_ing, exp_agent in TEST_CHAIN_VERBS:
    ev, sid = get_emb(verb)
    if ev is None: continue
    # Step 1: verb → +ing
    step1 = W_E[sid] + bs_ing * ax_ing
    r1 = nn_retrieve(step1, source_ids(verb), RELAXED_MASK, 1)
    got_ing = r1[0][0]; ing_id = r1[0][2]
    # Step 2: +ing → +er_noun (from the gerund position)
    step2 = W_E[ing_id] + bs_er * ax_er_noun
    r2 = nn_retrieve(step2, source_ids(got_ing), CLEAN_MASK, 3)
    got_agent = r2[0][0]
    ok1 = '✓' if got_ing == exp_ing else '~'
    ok2 = '✓' if got_agent == exp_agent else '~'
    if ok1 == '✓': step1_hits += 1
    if ok1 == '✓' and ok2 == '✓': chain_hits += 1
    alt = '/'.join(x[0] for x in r2[1:3]) if ok2 == '~' else ''
    print("  %-10s  %-14s  %-12s  %-12s  %-4s %-4s  %s" %
          (verb, exp_ing, got_ing, got_agent, ok1, ok2, alt))
print()
print("  Step 1 (verb→+ing): %d/%d" % (step1_hits, len(TEST_CHAIN_VERBS)))
print("  Full chain (verb→+ing→+er_noun): %d/%d" % (chain_hits, len(TEST_CHAIN_VERBS)))
print()

# Also test direct verb -> +er_noun (skip the gerund step)
print("  Comparison: direct verb -> +er_noun (1-step, no gerund):")
direct_hits = 0
for verb, exp_ing, exp_agent in TEST_CHAIN_VERBS:
    ev, sid = get_emb(verb)
    if ev is None: continue
    r = nn_retrieve(W_E[sid] + bs_er * ax_er_noun, source_ids(verb), CLEAN_MASK, 1)
    got = r[0][0]
    ok = '✓' if got == exp_agent else '~'
    if ok == '✓': direct_hits += 1
    if ok == '~': print("    ~ %-10s -> %-12s (expected: %s)" % (verb, got, exp_agent))
print("  Direct verb->+er_noun: %d/%d" % (direct_hits, len(TEST_CHAIN_VERBS)))
print()

# =====================================================================
# PART C: v6 BOUNDARY ANALYSIS — which 12 axes are STILL wrong?
# =====================================================================
print("PART C: v6 boundary analysis — diagnosing the 12/30 failures")
print("-"*80)

def classify_v6(pc, loo, irred):
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

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

FIXED_BENCH = [
    ('er_comp',   [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],       [('dark','darker'),('soft','softer'),('heavy','heavier')], 'morph_uniform'),
    ('er_sup',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')], [('warm','warmest'),('long','longest'),('cold','coldest')], 'morph_uniform'),
    ('relational',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],    [('Cairo','Egypt'),('Seoul','Korea'),('Lima','Peru')], 'relational_geom'),
    ('al_rel',    [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')], [('history','historical'),('season','seasonal'),('accident','accidental')], 'relational_geom'),
    ('plural',    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),('book','books'),('bird','birds'),('door','doors')],       [('cup','cups'),('word','words'),('room','rooms')], 'morph_moderate'),
    ('3ps',       [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),('write','writes'),('play','plays'),('work','works')],    [('talk','talks'),('sleep','sleeps'),('open','opens')], 'morph_moderate'),
    ('ed_reg',    [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played'),('clean','cleaned'),('open','opened'),('start','started')], [('end','ended'),('look','looked'),('rain','rained')], 'morph_moderate'),
    ('ing',       [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),('make','making'),('write','writing'),('read','reading')], [('eat','eating'),('work','working'),('play','playing')], 'morph_moderate'),
    ('cc',        [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')],               [('cup','Cup'),('door','Door'),('word','Word')], 'morph_moderate'),
    ('ness',      [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')], [('bright','brightness'),('sweet','sweetness'),('clean','cleanliness')], 'phonol_scatter'),
    ('ablaut',    [('go','went'),('take','took'),('give','gave'),('see','saw'),('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],             [('speak','spoke'),('break','broke'),('choose','chose')], 'phonol_scatter'),
    ('ablaut_t',  [('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left'),('deal','dealt'),('sleep','slept'),('mean','meant')],     [('burn','burned'),('learn','learned'),('smell','smelled')], 'phonol_scatter'),
    ('ity',       [('human','humanity'),('real','reality'),('national','nationality'),('personal','personality'),('moral','morality'),('legal','legality'),('final','finality'),('normal','normality')], [('mental','mentality'),('total','totality'),('brutal','brutality')], 'phonol_scatter'),
    ('un_neg',    [('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),('known','unknown'),('safe','unsafe'),('usual','unusual'),('equal','unequal')], [('stable','unstable'),('real','unreal'),('true','untrue')], 'phonol_scatter'),
    ('ance',      [('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),('accept','acceptance'),('appear','appearance'),('depend','dependence'),('insist','insistence')], [('persist','persistence'),('emerge','emergence'),('refer','reference')], 'phonol_scatter'),
    ('ment',      [('achieve','achievement'),('develop','development'),('manage','management'),('govern','government'),('engage','engagement'),('require','requirement'),('move','movement'),('improve','improvement')], [('amuse','amusement'),('punish','punishment'),('treat','treatment')], 'phonol_scatter'),
    ('tion',      [('act','action'),('direct','direction'),('educate','education'),('create','creation'),('produce','production'),('relate','relation'),('combine','combination'),('apply','application')], [('express','expression'),('extend','extension'),('omit','omission')], 'phonol_scatter'),
    ('al_nom',    [('arrive','arrival'),('propose','proposal'),('approve','approval'),('refuse','refusal'),('remove','removal'),('survive','survival'),('deny','denial'),('dispose','disposal')], [('retrieve','retrieval'),('betray','betrayal'),('renew','renewal')], 'phonol_scatter'),
    ('less',      [('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),('end','endless'),('home','homeless'),('harm','harmless'),('power','powerless')], [('worth','worthless'),('use','useless'),('mercy','merciless')], 'phonol_scatter'),
    ('ful',       [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')],        [('beauty','beautiful'),('wonder','wonderful'),('power','powerful')], 'phonol_scatter'),
    ('able',      [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')], [('comfort','comfortable'),('manage','manageable'),('reach','reachable')], 'phonol_scatter'),
    ('er_noun',   [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner'),('manage','manager'),('build','builder'),('lead','leader')], [('write','writer'),('paint','painter'),('print','printer')], 'semantic_diverse'),
    ('adj_ant',   [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('bright','dark'),('hard','soft'),('high','low'),('rich','poor')], [('open','closed'),('new','old'),('loud','quiet')], 'polar_local'),
    ('antonym2',  [('love','hate'),('war','peace'),('life','death'),('day','night'),('begin','end'),('give','take'),('push','pull'),('open','close')], [('rise','fall'),('win','lose'),('buy','sell')], 'polar_local'),
    ('en_es',     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día'),('night','noche'),('hand','mano'),('year','año')], [('fire','fuego'),('moon','luna'),('sea','mar')], 'translation'),
    ('en_de',     [('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),('day','Tag'),('night','Nacht'),('cat','Katze'),('dog','Hund')], [('fire','Feuer'),('moon','Mond'),('sea','Meer')], 'translation'),
    ('en_fr',     [('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),('day','jour'),('night','nuit'),('cat','chat'),('dog','chien')], [('fire','feu'),('moon','lune'),('sea','mer')], 'translation'),
    ('en_zh',     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼')], [('tree','树'),('heart','心'),('door','门')], 'factual_local'),
    ('en_ja',     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','目'),('fish','魚')], [('tree','木'),('heart','心'),('door','門')], 'factual_local'),
    ('num_word',  [('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),('6','six'),('7','seven'),('8','eight')], [('9','nine'),('10','ten'),('0','zero')], 'semantic_diverse'),
]

FAILURE_MODES = {
    'pc_wrong_range': [],
    'irred_wrong':    [],
    'loo_wrong':      [],
    'boundary_case':  [],
    'true_label_ambiguous': [],
}
print("  Diagnosing v6 failures:")
print("  %-12s  pc    LOO   irred  pred                   true            failure_mode" % "axis")
print("  " + "-"*95)
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr   = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    pred  = classify_v6(pc, loo_v, irr)
    ok    = match(pred, true_type)
    if ok: continue  # skip correct ones
    # Diagnose failure mode
    # Check: what pc range would give correct prediction?
    mode = 'unknown'
    # Try shifting pc
    correct_with_higher_pc = match(classify_v6(pc+0.15, loo_v, irr), true_type)
    correct_with_lower_pc  = match(classify_v6(max(0,pc-0.15), loo_v, irr), true_type)
    correct_with_higher_irr= match(classify_v6(pc, loo_v, min(1.0,irr+0.3)), true_type)
    correct_with_lower_irr = match(classify_v6(pc, loo_v, max(0,irr-0.3)), true_type)
    correct_with_higher_loo= match(classify_v6(pc, min(1.0,loo_v+0.3), irr), true_type)
    if correct_with_higher_pc or correct_with_lower_pc:
        mode = 'pc_threshold'
    elif correct_with_higher_irr or correct_with_lower_irr:
        mode = 'irred_threshold'
    elif correct_with_higher_loo:
        mode = 'loo_threshold'
    else:
        mode = 'structural'
    print("  ✗ %-12s  %.3f %.0f%%  %.2f   %-22s %-16s %s" %
          (name, pc, 100*loo_v, irr, pred[:22], true_type, mode))
print()

# =====================================================================
# PART D: MULTI-SUB-AXIS TEST (+ness, +ance, +tion have two sub-axes?)
# =====================================================================
print("PART D: Multi-sub-axis test — does mixing sub-populations improve +ness/+ance/+tion?")
print("-"*80)

MULTI_TESTS = [
    ('+ness',
     # Germanic (current training):
     [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),
      ('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],
     # Latinate additions:
     [('abstract','abstractness'),('direct','directness'),('explicit','explicitness'),
      ('distinct','distinctness'),('evident','evidentness'),('precise','preciseness')],
     # Holdout:
     [('bright','brightness'),('sweet','sweetness'),('clean','cleanliness'),
      ('sharp','sharpness'),('warm','warmness'),('calm','calmness')]),
    ('+ance',
     # Current training:
     [('perform','performance'),('exist','existence'),('enter','entrance'),
      ('resist','resistance'),('accept','acceptance'),('appear','appearance'),
      ('depend','dependence'),('insist','insistence')],
     # More Latin additions:
     [('assist','assistance'),('persist','persistence'),('emerge','emergence'),
      ('refer','reference'),('tolerate','tolerance'),('guide','guidance')],
     # Holdout:
     [('dominate','dominance'),('maintain','maintenance'),('endure','endurance'),
      ('ensure','insurance'),('allow','allowance'),('balance','balance')]),
    ('+tion',
     # Current training:
     [('act','action'),('direct','direction'),('educate','education'),('create','creation'),
      ('produce','production'),('relate','relation'),('combine','combination'),('apply','application')],
     # More additions:
     [('express','expression'),('extend','extension'),('omit','omission'),
      ('admit','admission'),('permit','permission'),('construct','construction')],
     # Holdout:
     [('restrict','restriction'),('instruct','instruction'),('destruct','destruction'),
      ('subtract','subtraction'),('attract','attraction'),('react','reaction')]),
]

for name, orig_train, extra_train, holdout in MULTI_TESTS:
    ax_orig, v_orig, pc_orig = compute_axis(orig_train)
    ax_mixed, v_mixed, pc_mixed = compute_axis(orig_train + extra_train)
    if ax_orig is None or ax_mixed is None: continue
    loo_orig  = axis_loo(ax_orig, v_orig, CLEAN_MASK)
    loo_mixed = axis_loo(ax_mixed, v_mixed, CLEAN_MASK)
    irr_orig  = irred_on_holdout(ax_orig, holdout, CLEAN_MASK)
    irr_mixed = irred_on_holdout(ax_mixed, holdout, CLEAN_MASK)
    c_axes    = float(np.dot(ax_orig.astype(np.float32), ax_mixed.astype(np.float32)))
    print("  %s:" % name)
    print("    ORIGINAL:  pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_orig, 100*loo_orig, 100*irr_orig, len(v_orig)))
    print("    MIXED:     pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d  cos_to_orig=%.3f" %
          (pc_mixed, 100*loo_mixed, 100*irr_mixed, len(v_mixed), c_axes))
    delta_loo  = 100*(loo_mixed - loo_orig)
    delta_irr  = 100*(irr_mixed - irr_orig)
    arrow_loo  = '▲' if delta_loo > 0 else ('▼' if delta_loo < 0 else '=')
    arrow_irr  = '▲' if delta_irr > 0 else ('▼' if delta_irr < 0 else '=')
    print("    CHANGE:    LOO %s%.0f%%  irred %s%.0f%%" %
          (arrow_loo, abs(delta_loo), arrow_irr, abs(delta_irr)))
    print()
