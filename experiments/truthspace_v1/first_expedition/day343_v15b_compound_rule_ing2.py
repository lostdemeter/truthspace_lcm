import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building masks...", flush=True)
RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) <= 1: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
print("  relaxed=%d" % RELAXED_MASK.sum())

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
# v14 (reference)
# =====================================================================
def classify_v14(pc, loo, irred, spread=0.0, src_is_digit=False, type0_ratio=0.0):
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
            if type0_ratio >= 0.40 and spread < 0.07: return 'phonol_scatter'
            return 'semantic_diverse'
        else:               return 'borderline'
    elif pc > 0.08:
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:
            if type0_ratio >= 0.70 and spread < 0.07: return 'phonol_scatter'
            return 'factual_local/translation'
        elif irred >= 0.60:
            if type0_ratio >= 0.40: return 'phonol_scatter'
            return 'semantic_diverse'
        elif loo == 0.0 and 0.20 <= irred < 0.60: return 'phonol_scatter'
        elif loo == 0.0 and irred < 0.20:          return 'semantic_diverse'
        elif 0.0 < loo < 0.50 and 0.20 <= irred < 0.60:
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
# v15b: v14 + compound rule for ing2
# Rule: loo >= 0.70 AND irred > 0.05 AND spread > 0.11 -> morph_moderate
#
# Rationale:
#   ing2: loo=0.75, irred=0.33, spread=0.122 -> all three fire ✓
#   ness: loo=0.75, irred=0.00 -> irred NOT > 0.05 -> doesn't fire ✓
#   al_nom: loo=0.75, irred=0.00 -> irred NOT > 0.05 -> doesn't fire ✓
#   tion2: loo=0.88, irred=0.33, spread=0.109 < 0.11 -> doesn't fire ✓
#
# The threshold spread=0.11 is between tion2 (0.109) and ing2 (0.122), margin=0.013.
# =====================================================================
def classify_v15b(pc, loo, irred, spread=0.0, src_is_digit=False, type0_ratio=0.0):
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
            if type0_ratio >= 0.40 and spread < 0.07: return 'phonol_scatter'
            return 'semantic_diverse'
        else:               return 'borderline'
    elif pc > 0.08:
        if loo >= 0.50:
            if irred >= 0.40:
                if loo >= 0.70: return 'phonol_scatter'
                return 'semantic_diverse'
            # v15b: compound gate catches ing2 (loo high, irred nonzero, spread high)
            # without firing for ness/al_nom (irred=0.00) or tion2 (spread=0.109 < 0.11)
            if loo >= 0.70 and irred > 0.05 and spread > 0.11: return 'morph_moderate'
            return 'phonol_scatter'
        elif irred >= 0.95:
            if type0_ratio >= 0.70 and spread < 0.07: return 'phonol_scatter'
            return 'factual_local/translation'
        elif irred >= 0.60:
            if type0_ratio >= 0.40: return 'phonol_scatter'
            return 'semantic_diverse'
        elif loo == 0.0 and 0.20 <= irred < 0.60: return 'phonol_scatter'
        elif loo == 0.0 and irred < 0.20:          return 'semantic_diverse'
        elif 0.0 < loo < 0.50 and 0.20 <= irred < 0.60:
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
    ('able',      ABLE_MIXED, ABLE_HOLDOUT, 'phonol_scatter'),
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

GEN_BENCH_V2 = [
    ('er_comp2',  [('old','older'),('young','younger'),('smart','smarter'),('strong','stronger'),('light','lighter'),('safe','safer'),('cheap','cheaper'),('quiet','quieter')], [('cool','cooler'),('warm','warmer'),('wide','wider')], 'morph_uniform'),
    ('er_sup2',   [('old','oldest'),('young','youngest'),('smart','smartest'),('strong','strongest'),('light','lightest'),('safe','safest'),('cheap','cheapest'),('quiet','quietest')], [('cool','coolest'),('warm','warmest'),('wide','widest')], 'morph_uniform'),
    ('pl_reg2',   [('hand','hands'),('arm','arms'),('eye','eyes'),('leg','legs'),('head','heads'),('mouth','mouths'),('face','faces'),('mind','minds')], [('heart','hearts'),('foot','foots'),('ear','ears')], 'morph_moderate'),
    ('3ps2',      [('live','lives'),('want','wants'),('need','needs'),('find','finds'),('keep','keeps'),('call','calls'),('feel','feels'),('turn','turns')], [('show','shows'),('hold','holds'),('move','moves')], 'morph_moderate'),
    ('ing2',      [('live','living'),('want','wanting'),('need','needing'),('find','finding'),('keep','keeping'),('call','calling'),('feel','feeling'),('turn','turning')], [('show','showing'),('hold','holding'),('move','moving')], 'morph_moderate'),
    ('er_2syl',   [('happy','happier'),('easy','easier'),('busy','busier'),('early','earlier'),('heavy','heavier'),('pretty','prettier'),('funny','funnier'),('angry','angrier')], [('lucky','luckier'),('noisy','noisier'),('cloudy','cloudier')], 'morph_moderate'),
    ('pl_irr',    [('foot','feet'),('tooth','teeth'),('man','men'),('woman','women'),('mouse','mice'),('goose','geese'),('child','children'),('person','people')], [('ox','oxen'),('die','dice'),('louse','lice')], 'phonol_scatter'),
    ('past_ab',   [('swim','swam'),('sing','sang'),('ring','rang'),('drink','drank'),('sink','sank'),('begin','began'),('run','ran'),('spring','sprang')], [('shrink','shrank'),('sink','sank'),('blow','blew')], 'phonol_scatter'),
    ('ize',       [('modern','modernize'),('local','localize'),('real','realize'),('social','socialize'),('legal','legalize'),('private','privatize'),('organ','organize'),('terror','terrorize')], [('civil','civilize'),('final','finalize'),('vital','vitalize')], 'phonol_scatter'),
    ('ous',       [('danger','dangerous'),('poison','poisonous'),('fame','famous'),('nerve','nervous'),('humor','humorous'),('hazard','hazardous'),('glory','glorious'),('courage','courageous')], [('vigor','vigorous'),('mystery','mysterious'),('joy','joyous')], 'phonol_scatter'),
    ('en',        [('dark','darken'),('bright','brighten'),('hard','harden'),('sharp','sharpen'),('deep','deepen'),('wide','widen'),('loose','loosen'),('tight','tighten')], [('soft','soften'),('weak','weaken'),('thick','thicken')], 'phonol_scatter'),
    ('ish',       [('child','childish'),('self','selfish'),('fool','foolish'),('fever','feverish'),('clown','clownish'),('book','bookish'),('baby','babyish'),('snob','snobbish')], [('freak','freakish'),('wolf','wolfish'),('oaf','oafish')], 'phonol_scatter'),
    ('ist',       [('art','artist'),('real','realist'),('novel','novelist'),('journal','journalist'),('tour','tourist'),('capital','capitalist'),('social','socialist'),('final','finalist')], [('piano','pianist'),('guitar','guitarist'),('terror','terrorist')], 'phonol_scatter'),
    ('ism',       [('real','realism'),('social','socialism'),('capital','capitalism'),('human','humanism'),('symbol','symbolism'),('terror','terrorism'),('ideal','idealism'),('national','nationalism')], [('natural','naturalism'),('rational','rationalism'),('plural','pluralism')], 'phonol_scatter'),
    ('ness2',     [('cold','coldness'),('bold','boldness'),('calm','calmness'),('fresh','freshness'),('rich','richness'),('wild','wildness'),('neat','neatness'),('raw','rawness')], [('brave','braveness'),('free','freeness'),('fair','fairness')], 'morph_uniform'),
    ('ward',      [('north','northward'),('south','southward'),('east','eastward'),('west','westward'),('home','homeward'),('up','upward'),('in','inward'),('out','outward')], [('back','backward'),('for','forward'),('down','downward')], 'morph_uniform'),
    ('re_pfx',    [('try','retry'),('do','redo'),('write','rewrite'),('start','restart'),('build','rebuild'),('read','reread'),('use','reuse'),('think','rethink')], [('turn','return'),('view','review'),('place','replace')], 'semantic_diverse'),
    ('pre_pfx',   [('view','preview'),('heat','preheat'),('pay','prepay'),('cook','precook'),('treat','pretreat'),('warn','prewarn'),('select','preselect'),('test','pretest')], [('school','preschool'),('order','preorder'),('set','preset')], 'phonol_scatter'),
    ('un_verb',   [('lock','unlock'),('wrap','unwrap'),('tie','untie'),('fold','unfold'),('pack','unpack'),('dress','undress'),('cover','uncover'),('do','undo')], [('load','unload'),('zip','unzip'),('plug','unplug')], 'phonol_scatter'),
    ('ary',       [('element','elementary'),('moment','momentary'),('comment','commentary'),('legend','legendary'),('custom','customary'),('vision','visionary'),('honor','honorary'),('mission','missionary')], [('revolution','revolutionary'),('parliament','parliamentary'),('discipline','disciplinary')], 'phonol_scatter'),
    ('tion2',     [('invent','invention'),('observe','observation'),('explain','explanation'),('object','objection'),('describe','description'),('destroy','destruction'),('celebrate','celebration'),('compose','composition')], [('oppose','opposition'),('distribute','distribution'),('contrast','contradiction')], 'phonol_scatter'),
    ('er_noun2',  [('play','player'),('sing','singer'),('report','reporter'),('hack','hacker'),('surf','surfer'),('climb','climber'),('swim','swimmer'),('box','boxer')], [('run','runner'),('skate','skater'),('cycle','cyclist')], 'semantic_diverse'),
    ('gender_pr', [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),('son','daughter'),('husband','wife'),('uncle','aunt'),('prince','princess')], [('actor','actress'),('waiter','waitress'),('hero','heroine')], 'morph_moderate'),
    ('num_ord',   [('one','first'),('two','second'),('three','third'),('four','fourth'),('five','fifth'),('six','sixth'),('seven','seventh'),('eight','eighth')], [('nine','ninth'),('ten','tenth'),('eleven','eleventh')], 'morph_uniform'),
    ('adj_ant2',  [('clean','dirty'),('right','wrong'),('true','false'),('early','late'),('cheap','expensive'),('safe','dangerous'),('simple','complex'),('quiet','loud')], [('open','closed'),('full','empty'),('public','private')], 'polar_local'),
    ('abstract_ant',[('success','failure'),('victory','defeat'),('reward','punishment'),('praise','blame'),('courage','cowardice'),('freedom','slavery'),('truth','lie'),('order','chaos')], [('rise','fall'),('creation','destruction'),('unity','division')], 'polar_local'),
    ('en_it',     [('house','casa'),('water','acqua'),('sun','sole'),('book','libro'),('day','giorno'),('cat','gatto'),('dog','cane'),('fire','fuoco')], [('night','notte'),('moon','luna'),('sea','mare')], 'translation'),
    ('en_nl',     [('house','huis'),('water','water'),('sun','zon'),('book','boek'),('day','dag'),('cat','kat'),('dog','hond'),('fire','vuur')], [('night','nacht'),('moon','maan'),('sea','zee')], 'translation'),
    ('en_pt',     [('house','casa'),('water','água'),('sun','sol'),('fire','fogo'),('day','dia'),('cat','gato'),('dog','cachorro'),('night','noite')], [('moon','lua'),('sea','mar'),('tree','árvore')], 'translation'),
    ('en_zh2',    [('big','大'),('small','小'),('good','好'),('new','新'),('old','老'),('high','高'),('low','低'),('long','长')], [('short','短'),('wide','宽'),('deep','深')], 'semantic_diverse'),
]

print()
print("DAY 343: v15b compound rule (loo>=0.70 AND irred>0.05 AND spread>0.11) for ing2")
print("="*80)

def run_bench(bench, classifier, label):
    print("  Computing %s..." % label, flush=True)
    score = 0; rows = []
    for name, train_pairs, holdout_pairs, true_type in bench:
        ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
        if ax is None or len(valid) < 2: continue
        loo_v  = axis_loo(ax, valid, RELAXED_MASK)
        irr, t0r = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
        src_is_digit = all(tok.decode([sid]).strip().isdigit() for _,_,sid,_ in valid)
        pred = classifier(pc, loo_v, irr, spread, src_is_digit, t0r)
        ok = match(pred, true_type)
        if ok: score += 1
        rows.append((name, pc, loo_v, irr, t0r, spread, pred, true_type, ok))
    return score, len(rows), rows

# Print feature values for key axes first
print()
print("PART A: feature check on axes near the compound rule threshold")
print("-"*80)
KEY_AXES = [
    ('ing2',  [('live','living'),('want','wanting'),('need','needing'),('find','finding'),('keep','keeping'),('call','calling'),('feel','feeling'),('turn','turning')],
              [('show','showing'),('hold','holding'),('move','moving')], 'morph_moderate'),
    ('tion2', [('invent','invention'),('observe','observation'),('explain','explanation'),('object','objection'),('describe','description'),('destroy','destruction'),('celebrate','celebration'),('compose','composition')],
              [('oppose','opposition'),('distribute','distribution'),('contrast','contradiction')], 'phonol_scatter'),
    ('ness',  [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],
              [('bright','brightness'),('sweet','sweetness'),('clean','cleanliness')], 'phonol_scatter'),
    ('al_nom',[('arrive','arrival'),('propose','proposal'),('approve','approval'),('refuse','refusal'),('remove','removal'),('survive','survival'),('deny','denial'),('dispose','disposal')],
              [('retrieve','retrieval'),('betray','betrayal'),('renew','renewal')], 'phonol_scatter'),
    ('ful',   [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')],
              [('beauty','beautiful'),('wonder','wonderful'),('power','powerful')], 'phonol_scatter'),
    ('ment',  [('achieve','achievement'),('develop','development'),('manage','management'),('govern','government'),('engage','engagement'),('require','requirement'),('move','movement'),('improve','improvement')],
              [('amuse','amusement'),('punish','punishment'),('treat','treatment')], 'phonol_scatter'),
]
print("  %-12s  pc     LOO   irred  spread  rule_fires?  true_label" % "axis")
print("  " + "-"*72)
for name, train_pairs, holdout_pairs, true_label in KEY_AXES:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None: continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr, _ = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
    fires = (loo_v >= 0.70 and irr > 0.05 and spread > 0.11)
    flag = 'YES ***' if fires else 'no'
    print("  %-12s  %.3f  %.0f%%  %.2f   %.3f   %-11s  %s" % (name, pc, 100*loo_v, irr, spread, flag, true_label))

orig_v14, orig_total, orig_rows_v14 = run_bench(ORIG_BENCH,   classify_v14,  'v14 on orig')
orig_v15b, _, orig_rows_v15b        = run_bench(ORIG_BENCH,   classify_v15b, 'v15b on orig')
gen_v14,   gen_total, gen_rows_v14  = run_bench(GEN_BENCH_V2, classify_v14,  'v14 on gen')
gen_v15b,  _, gen_rows_v15b         = run_bench(GEN_BENCH_V2, classify_v15b, 'v15b on gen')

print()
print("PART B: safety check on original benchmark")
print("-"*80)
print("  v14  on orig: %d/%d = %.0f%%" % (orig_v14, orig_total, 100*orig_v14/orig_total))
print("  v15b on orig: %d/%d = %.0f%%" % (orig_v15b, orig_total, 100*orig_v15b/orig_total))
any_change = False
for r14, r15b in zip(orig_rows_v14, orig_rows_v15b):
    name, pc, loo_v, irr, t0r, spread, pred14, true_type, ok14 = r14
    _, _, _, _, _, _, pred15b, _, ok15b = r15b
    if pred14 != pred15b:
        any_change = True
        change = 'v15b+' if (not ok14 and ok15b) else ('v15b-' if (ok14 and not ok15b) else '~same')
        print("  -> %-12s  pc=%.3f  spread=%.3f  irred=%.2f  loo=%.0f%%  v14=%s -> v15b=%s  true=%s  %s" %
              (name, pc, spread, irr, 100*loo_v, pred14[:14], pred15b[:14], true_type, change))
if not any_change:
    print("  No changes on original benchmark.")

print()
print("PART C: generalization benchmark (revised labels)")
print("-"*80)
print("  v14  on gen: %d/%d = %.0f%%" % (gen_v14, gen_total, 100*gen_v14/gen_total))
print("  v15b on gen: %d/%d = %.0f%%" % (gen_v15b, gen_total, 100*gen_v15b/gen_total))
print()
print("  Changes v14->v15b:")
any_change = False
for r14, r15b in zip(gen_rows_v14, gen_rows_v15b):
    name, pc, loo_v, irr, t0r, spread, pred14, true_type, ok14 = r14
    _, _, _, _, _, _, pred15b, _, ok15b = r15b
    if pred14 != pred15b:
        any_change = True
        change = 'v15b+' if (not ok14 and ok15b) else ('v15b-' if (ok14 and not ok15b) else '~same')
        print("  -> %-12s  pc=%.3f  spread=%.3f  irred=%.2f  loo=%.0f%%  v14=%s -> v15b=%s  true=%s  %s" %
              (name, pc, spread, irr, 100*loo_v, pred14[:14], pred15b[:14], true_type, change))
if not any_change:
    print("  No changes on gen benchmark.")

print()
print("PART D: full v15b table on gen benchmark")
print("-"*80)
print("  %-14s  pc    LOO  irred  spread  pred_v15b            true_rev             match" % "axis")
print("  " + "-"*100)
by_cat = {}
for r15b in gen_rows_v15b:
    name, pc, loo_v, irr, t0r, spread, pred, true_type, ok = r15b
    flag = '✓' if ok else '✗'
    marker = '  ' if ok else '->'
    print("  %s %-12s  %.3f  %.0f%%  %.2f   %.3f   %-20s %-20s %s" %
          (marker, name, pc, 100*loo_v, irr, spread, pred[:20], true_type, flag))
    if true_type not in by_cat: by_cat[true_type] = {'ok':0,'total':0,'fails':[]}
    by_cat[true_type]['total'] += 1
    if ok: by_cat[true_type]['ok'] += 1
    else: by_cat[true_type]['fails'].append(name)

print()
print("  By category:")
for cat in sorted(by_cat.keys()):
    d = by_cat[cat]
    fs = ', '.join(d['fails'])
    print("  %-22s  %d/%d = %.0f%%  %s" % (cat, d['ok'], d['total'],
          100*d['ok']/d['total'] if d['total'] else 0, ('FAIL: '+fs) if fs else ''))

print()
print("PART E: combined summary")
print("-"*80)
total = orig_total + gen_total
print("  v14:  orig=%d/%d  gen=%d/%d  combined=%d/%d = %.0f%%" %
      (orig_v14, orig_total, gen_v14, gen_total, orig_v14+gen_v14, total, 100*(orig_v14+gen_v14)/total))
print("  v15b: orig=%d/%d  gen=%d/%d  combined=%d/%d = %.0f%%" %
      (orig_v15b, orig_total, gen_v15b, gen_total, orig_v15b+gen_v15b, total, 100*(orig_v15b+gen_v15b)/total))
