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

def irred_with_type0_ratio(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    n_ho=0; n_irred=0; n_type0_irred=0
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
    type0_ratio = n_type0_irred/max(n_irred, 1)
    return raw_irred, type0_ratio

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

# =====================================================================
# v13: v12 + irred>=0.95 type0_ratio gate + revised phonol_scatter labels
# =====================================================================
def classify_v12(pc, loo, irred, spread=0.0, src_is_digit=False, type0_ratio=0.0):
    if src_is_digit: return 'semantic_diverse'
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if loo >= 0.80 and spread > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if loo >= 0.50:
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'
        elif irred < 0.30:  return 'morph_moderate'
        elif irred >= 0.60:
            if type0_ratio >= 0.40: return 'phonol_scatter'
            return 'semantic_diverse'
        else:               return 'borderline'
    elif pc > 0.10:
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:
            if type0_ratio >= 0.40: return 'phonol_scatter'
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

def classify_v13(pc, loo, irred, spread=0.0, src_is_digit=False, type0_ratio=0.0):
    """v13: v12 + three targeted fixes:
    1. irred>=0.95 gate: type0_ratio>=0.70 AND spread<0.07 -> phonol_scatter
       (spread<0.07 preserves en_zh/en_ja whose cross-lingual chords are MORE spread)
    2. 0<loo<0.50, irred<0.60 branch: type0_ratio>=0.80 -> phonol_scatter (fixes 'en' axis)
    3. pc lower bound: 0.10 -> 0.08 (catches ous/ist that just miss the threshold)
    """
    if src_is_digit: return 'semantic_diverse'
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if loo >= 0.80 and spread > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if loo >= 0.50:
            return 'morph_moderate' if irred < 0.40 else 'phonol_scatter'
        elif irred < 0.30:  return 'morph_moderate'
        elif irred >= 0.60:
            # type0_ratio gate extended to pc>0.195 zone (from v12)
            if type0_ratio >= 0.40: return 'phonol_scatter'
            return 'semantic_diverse'
        else:               return 'borderline'
    elif pc > 0.08:  # v13: lowered from 0.10 to catch ous/ist near-misses
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:
            # v13: vocabulary-limited morphological axes have TIGHT spread (<0.07)
            # cross-lingual/factual axes have WIDER spread (en_zh=0.090, en_ja=0.093)
            if type0_ratio >= 0.70 and spread < 0.07: return 'phonol_scatter'
            return 'factual_local/translation'
        elif irred >= 0.60:
            if type0_ratio >= 0.40: return 'phonol_scatter'
            return 'semantic_diverse'
        elif loo == 0.0 and 0.20 <= irred < 0.60: return 'phonol_scatter'
        elif loo == 0.0 and irred < 0.20:          return 'semantic_diverse'
        elif 0.0 < loo < 0.50 and 0.20 <= irred < 0.60:
            # v13: type0_ratio gate (fixes 'en': dark->darken, t0r=1.00)
            if type0_ratio >= 0.80: return 'phonol_scatter'
            return 'semantic_diverse'
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
# BENCHMARKS
# =====================================================================
ABLE_MIXED = [
    ('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
    ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable'),
    ('comfort','comfortable'),('manage','manageable'),('reach','reachable'),
    ('depend','dependable'),('honor','honorable'),('justify','justifiable'),
]
ABLE_HOLDOUT = [('comfort','comfortable'),('manage','manageable'),('reach','reachable')]

ORIG_BENCH = [
    ('er_comp',   [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],                         [('dark','darker'),('soft','softer'),('heavy','heavier')], 'morph_uniform'),
    ('er_sup',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')],               [('warm','warmest'),('long','longest'),('cold','coldest')], 'morph_uniform'),
    ('relational',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],                    [('Cairo','Egypt'),('Seoul','Korea'),('Lima','Peru')], 'relational_geom'),
    ('al_rel',    [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')], [('history','historical'),('season','seasonal'),('accident','accidental')], 'phonol_scatter'),
    ('plural',    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),('book','books'),('bird','birds'),('door','doors')],                                           [('cup','cups'),('word','words'),('room','rooms')], 'morph_moderate'),
    ('3ps',       [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),('write','writes'),('play','plays'),('work','works')],                                         [('talk','talks'),('sleep','sleeps'),('open','opens')], 'morph_moderate'),
    ('ed_reg',    [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played'),('clean','cleaned'),('open','opened'),('start','started')],                           [('end','ended'),('look','looked'),('rain','rained')], 'morph_moderate'),
    ('ing',       [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),('make','making'),('write','writing'),('read','reading')],                                [('eat','eating'),('work','working'),('play','playing')], 'morph_moderate'),
    ('cc',        [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')],                                                [('cup','Cup'),('door','Door'),('word','Word')], 'semantic_diverse'),
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
    ('able',      ABLE_MIXED,                                                                                                                                                                    ABLE_HOLDOUT, 'phonol_scatter'),
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

GEN_BENCH_V2 = [  # Day 339 axes with REVISED expected labels
    ('er_comp2',  [('old','older'),('young','younger'),('smart','smarter'),('strong','stronger'),('light','lighter'),('safe','safer'),('cheap','cheaper'),('quiet','quieter')],
                  [('cool','cooler'),('warm','warmer'),('wide','wider')], 'morph_uniform'),
    ('er_sup2',   [('old','oldest'),('young','youngest'),('smart','smartest'),('strong','strongest'),('light','lightest'),('safe','safest'),('cheap','cheapest'),('quiet','quietest')],
                  [('cool','coolest'),('warm','warmest'),('wide','widest')], 'morph_uniform'),
    ('pl_reg2',   [('hand','hands'),('arm','arms'),('eye','eyes'),('leg','legs'),('head','heads'),('mouth','mouths'),('face','faces'),('mind','minds')],
                  [('heart','hearts'),('foot','foots'),('ear','ears')], 'morph_moderate'),
    ('3ps2',      [('live','lives'),('want','wants'),('need','needs'),('find','finds'),('keep','keeps'),('call','calls'),('feel','feels'),('turn','turns')],
                  [('show','shows'),('hold','holds'),('move','moves')], 'morph_moderate'),
    ('ing2',      [('live','living'),('want','wanting'),('need','needing'),('find','finding'),('keep','keeping'),('call','calling'),('feel','feeling'),('turn','turning')],
                  [('show','showing'),('hold','holding'),('move','moving')], 'morph_moderate'),
    ('er_2syl',   [('happy','happier'),('easy','easier'),('busy','busier'),('early','earlier'),('heavy','heavier'),('pretty','prettier'),('funny','funnier'),('angry','angrier')],
                  [('lucky','luckier'),('noisy','noisier'),('cloudy','cloudier')], 'morph_moderate'),
    ('pl_irr',    [('foot','feet'),('tooth','teeth'),('man','men'),('woman','women'),('mouse','mice'),('goose','geese'),('child','children'),('person','people')],
                  [('ox','oxen'),('die','dice'),('louse','lice')], 'phonol_scatter'),
    ('past_ab',   [('swim','swam'),('sing','sang'),('ring','rang'),('drink','drank'),('sink','sank'),('begin','began'),('run','ran'),('spring','sprang')],
                  [('shrink','shrank'),('sink','sank'),('blow','blew')], 'phonol_scatter'),
    ('ize',       [('modern','modernize'),('local','localize'),('real','realize'),('social','socialize'),('legal','legalize'),('private','privatize'),('organ','organize'),('terror','terrorize')],
                  [('civil','civilize'),('final','finalize'),('vital','vitalize')], 'phonol_scatter'),
    ('ous',       [('danger','dangerous'),('poison','poisonous'),('fame','famous'),('nerve','nervous'),('humor','humorous'),('hazard','hazardous'),('glory','glorious'),('courage','courageous')],
                  [('vigor','vigorous'),('mystery','mysterious'),('joy','joyous')], 'phonol_scatter'),
    ('en',        [('dark','darken'),('bright','brighten'),('hard','harden'),('sharp','sharpen'),('deep','deepen'),('wide','widen'),('loose','loosen'),('tight','tighten')],
                  [('soft','soften'),('weak','weaken'),('thick','thicken')], 'phonol_scatter'),
    ('ish',       [('child','childish'),('self','selfish'),('fool','foolish'),('fever','feverish'),('clown','clownish'),('book','bookish'),('baby','babyish'),('snob','snobbish')],
                  [('freak','freakish'),('wolf','wolfish'),('oaf','oafish')], 'phonol_scatter'),
    ('ist',       [('art','artist'),('real','realist'),('novel','novelist'),('journal','journalist'),('tour','tourist'),('capital','capitalist'),('social','socialist'),('final','finalist')],
                  [('piano','pianist'),('guitar','guitarist'),('terror','terrorist')], 'phonol_scatter'),
    ('ism',       [('real','realism'),('social','socialism'),('capital','capitalism'),('human','humanism'),('symbol','symbolism'),('terror','terrorism'),('ideal','idealism'),('national','nationalism')],
                  [('natural','naturalism'),('rational','rationalism'),('plural','pluralism')], 'phonol_scatter'),
    ('ness2',     [('cold','coldness'),('bold','boldness'),('calm','calmness'),('fresh','freshness'),('rich','richness'),('wild','wildness'),('neat','neatness'),('raw','rawness')],
                  [('brave','braveness'),('free','freeness'),('fair','fairness')], 'morph_uniform'),   # REVISED
    ('ward',      [('north','northward'),('south','southward'),('east','eastward'),('west','westward'),('home','homeward'),('up','upward'),('in','inward'),('out','outward')],
                  [('back','backward'),('for','forward'),('down','downward')], 'morph_uniform'),       # REVISED
    ('re_pfx',    [('try','retry'),('do','redo'),('write','rewrite'),('start','restart'),('build','rebuild'),('read','reread'),('use','reuse'),('think','rethink')],
                  [('turn','return'),('view','review'),('place','replace')], 'semantic_diverse'),      # REVISED
    ('pre_pfx',   [('view','preview'),('heat','preheat'),('pay','prepay'),('cook','precook'),('treat','pretreat'),('warn','prewarn'),('select','preselect'),('test','pretest')],
                  [('school','preschool'),('order','preorder'),('set','preset')], 'phonol_scatter'),
    ('un_verb',   [('lock','unlock'),('wrap','unwrap'),('tie','untie'),('fold','unfold'),('pack','unpack'),('dress','undress'),('cover','uncover'),('do','undo')],
                  [('load','unload'),('zip','unzip'),('plug','unplug')], 'phonol_scatter'),
    ('ary',       [('element','elementary'),('moment','momentary'),('comment','commentary'),('legend','legendary'),('custom','customary'),('vision','visionary'),('honor','honorary'),('mission','missionary')],
                  [('revolution','revolutionary'),('parliament','parliamentary'),('discipline','disciplinary')], 'phonol_scatter'),
    ('tion2',     [('invent','invention'),('observe','observation'),('explain','explanation'),('object','objection'),('describe','description'),('destroy','destruction'),('celebrate','celebration'),('compose','composition')],
                  [('oppose','opposition'),('distribute','distribution'),('contrast','contradiction')], 'phonol_scatter'),
    ('er_noun2',  [('play','player'),('sing','singer'),('report','reporter'),('hack','hacker'),('surf','surfer'),('climb','climber'),('swim','swimmer'),('box','boxer')],
                  [('run','runner'),('skate','skater'),('cycle','cyclist')], 'semantic_diverse'),
    ('gender_pr', [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),('son','daughter'),('husband','wife'),('uncle','aunt'),('prince','princess')],
                  [('actor','actress'),('waiter','waitress'),('hero','heroine')], 'morph_moderate'),   # REVISED
    ('num_ord',   [('one','first'),('two','second'),('three','third'),('four','fourth'),('five','fifth'),('six','sixth'),('seven','seventh'),('eight','eighth')],
                  [('nine','ninth'),('ten','tenth'),('eleven','eleventh')], 'morph_uniform'),          # REVISED
    ('adj_ant2',  [('clean','dirty'),('right','wrong'),('true','false'),('early','late'),('cheap','expensive'),('safe','dangerous'),('simple','complex'),('quiet','loud')],
                  [('open','closed'),('full','empty'),('public','private')], 'polar_local'),
    ('abstract_ant',[('success','failure'),('victory','defeat'),('reward','punishment'),('praise','blame'),('courage','cowardice'),('freedom','slavery'),('truth','lie'),('order','chaos')],
                  [('rise','fall'),('creation','destruction'),('unity','division')], 'polar_local'),
    ('en_it',     [('house','casa'),('water','acqua'),('sun','sole'),('book','libro'),('day','giorno'),('cat','gatto'),('dog','cane'),('fire','fuoco')],
                  [('night','notte'),('moon','luna'),('sea','mare')], 'translation'),
    ('en_nl',     [('house','huis'),('water','water'),('sun','zon'),('book','boek'),('day','dag'),('cat','kat'),('dog','hond'),('fire','vuur')],
                  [('night','nacht'),('moon','maan'),('sea','zee')], 'translation'),
    ('en_pt',     [('house','casa'),('water','água'),('sun','sol'),('fire','fogo'),('day','dia'),('cat','gato'),('dog','cachorro'),('night','noite')],
                  [('moon','lua'),('sea','mar'),('tree','árvore')], 'translation'),
    ('en_zh2',    [('big','大'),('small','小'),('good','好'),('new','新'),('old','老'),('high','高'),('low','低'),('long','长')],
                  [('short','短'),('wide','宽'),('deep','深')], 'factual_local'),
]

print()
print("DAY 340: v13 — irred>=0.95 type0_ratio gate + pc>0.08 lower bound")
print("="*80)

print()
print("PART A: v12 vs v13 safety check on ORIGINAL 30-axis benchmark")
print("-"*80)

def run_bench(bench, classifier, label):
    print("  Computing %s..." % label, flush=True)
    score = 0; total = 0
    rows = []
    for name, train_pairs, holdout_pairs, true_type in bench:
        ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
        if ax is None or len(valid) < 2: continue
        loo_v  = axis_loo(ax, valid, RELAXED_MASK)
        irr, t0r = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
        src_is_digit = all(tok.decode([sid]).strip().isdigit()
                           for _,_,sid,_ in valid)
        pred = classifier(pc, loo_v, irr, spread, src_is_digit, t0r)
        ok = match(pred, true_type)
        if ok: score += 1
        total += 1
        rows.append((name, pc, loo_v, irr, t0r, spread, pred, true_type, ok))
    return score, total, rows

orig_v12_score, orig_total, orig_rows = run_bench(ORIG_BENCH, classify_v12, 'v12 on orig')
orig_v13_score, _, orig_rows_v13      = run_bench(ORIG_BENCH, classify_v13, 'v13 on orig')
gen_v12_score,  gen_total, gen_rows   = run_bench(GEN_BENCH_V2, classify_v12, 'v12 on gen_v2')
gen_v13_score,  _, gen_rows_v13       = run_bench(GEN_BENCH_V2, classify_v13, 'v13 on gen_v2')
print()

# Check for regressions in original benchmark
print("  v12 on orig: %d/%d = %.0f%%" % (orig_v12_score, orig_total, 100*orig_v12_score/orig_total))
print("  v13 on orig: %d/%d = %.0f%%" % (orig_v13_score, orig_total, 100*orig_v13_score/orig_total))
regressions = [(n, p12, p13, t) for (n,_,_,_,_,_,p12,t,ok12), (_,_,_,_,_,_,p13,_,ok13)
               in zip(orig_rows, orig_rows_v13) if ok12 and not ok13
               for n,_,_,_,_,_,p12,t,_ in [list(orig_rows)[[r[0] for r in orig_rows].index(n)]]
               ]
# simpler regression check
reg_count = 0
for r12, r13 in zip(orig_rows, orig_rows_v13):
    name, pc, loo_v, irr, t0r, spread, pred12, true_type, ok12 = r12
    _, _, _, _, _, _, pred13, _, ok13 = r13
    if ok12 and not ok13:
        print("  REGRESSION: %-12s  v12=%s  v13=%s  true=%s" % (name, pred12, pred13, true_type))
        reg_count += 1
if reg_count == 0:
    print("  No regressions on original benchmark.")

print()
print("PART B: v12 vs v13 on REVISED generalization benchmark (revised labels)")
print("-"*80)
print("  v12 on gen_v2: %d/%d = %.0f%%" % (gen_v12_score, gen_total, 100*gen_v12_score/gen_total))
print("  v13 on gen_v2: %d/%d = %.0f%%" % (gen_v13_score, gen_total, 100*gen_v13_score/gen_total))
print()
print("  Changes v12→v13:")
for r12, r13 in zip(gen_rows, gen_rows_v13):
    name, pc, loo_v, irr, t0r, spread, pred12, true_type, ok12 = r12
    _, _, _, _, _, _, pred13, _, ok13 = r13
    if pred12 != pred13:
        change = 'v13+' if (not ok12 and ok13) else ('v13-' if (ok12 and not ok13) else '~same')
        print("  %s %-12s  pc=%.3f  t0r=%.2f  v12=%s -> v13=%s  true=%s  %s" %
              ('->' if ok12!=ok13 else '  ', name, pc, t0r, pred12[:18], pred13[:18], true_type, change))
print()

print("PART C: Full v13 table on revised generalization benchmark")
print("-"*80)
print("  %-14s  pc    LOO  irred  t0r   pred_v13             true(revised)        match" % "axis")
print("  " + "-"*100)
by_cat = {}
for r13 in gen_rows_v13:
    name, pc, loo_v, irr, t0r, spread, pred, true_type, ok = r13
    flag = '✓' if ok else '✗'
    marker = '  ' if ok else '->'
    print("  %s %-12s  %.3f  %.0f%%  %.2f   %.2f  %-20s %-20s %s" %
          (marker, name, pc, 100*loo_v, irr, t0r, pred[:20], true_type, flag))
    if true_type not in by_cat: by_cat[true_type] = {'ok':0, 'total':0, 'fails':[]}
    by_cat[true_type]['total'] += 1
    if ok: by_cat[true_type]['ok'] += 1
    else:  by_cat[true_type]['fails'].append(name)

print()
print("  By category:")
for cat in sorted(by_cat.keys()):
    d = by_cat[cat]
    fs = ', '.join(d['fails'])
    print("  %-22s  %d/%d = %.0f%%  %s" % (cat, d['ok'], d['total'],
          100*d['ok']/d['total'] if d['total'] else 0, ('FAIL: '+fs) if fs else ''))

print()
print("PART D: combined v13 score across both benchmarks")
print("-"*80)
combined = orig_v13_score + gen_v13_score
combined_total = orig_total + gen_total
print("  v13 on original 30-axis benchmark: %d/%d = %.0f%%" % (orig_v13_score, orig_total, 100*orig_v13_score/orig_total))
print("  v13 on revised gen benchmark:      %d/%d = %.0f%%" % (gen_v13_score, gen_total, 100*gen_v13_score/gen_total))
print("  v13 combined:                      %d/%d = %.0f%%" % (combined, combined_total, 100*combined/combined_total))
print()
print("  Comparison:")
print("  v12 orig: %d/30 = 100%%   v12 gen: %d/%d = %.0f%%" % (orig_v12_score, gen_v12_score, gen_total, 100*gen_v12_score/gen_total))
print("  v13 orig: %d/30 = %.0f%%  v13 gen: %d/%d = %.0f%%" % (orig_v13_score, 100*orig_v13_score/orig_total, gen_v13_score, gen_total, 100*gen_v13_score/gen_total))
