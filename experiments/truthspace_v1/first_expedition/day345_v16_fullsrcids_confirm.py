import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and len(w) > 1 and not w.startswith('-') and not w.startswith('_'):
        RELAXED_MASK[i] = True

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
def get_tc(word):
    for p in [' ', '']:
        return len(tok(p+word, add_special_tokens=False)['input_ids'])
    return 99

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    i = int(np.argmax(sims))
    return tok.decode([i]).strip()

def axis_spread(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, valid, 0.0, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    n = len(cn)
    cosines = [np.dot(cn[i], cn[j]) for i in range(n) for j in range(i+1,n)]
    return md, valid, float(np.mean(cosines)), float(np.std(cosines)) if len(cosines)>1 else 0.0

def best_scale(ax, valid, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 6.0, 30):
        c = sum(1 for _,t,sid,_ in valid
                if nn_ret(W_E[sid]+s*ax, source_ids(tok.decode([sid]).strip()), mask)==t)
        if c > best_a: best_a=c; best_s=s
    return best_s

def loo(ax, valid, mask):
    if len(valid) < 3: return 0.0
    cf = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    gs = best_scale(normed(np.mean(cf, axis=0)), valid, mask)
    hits = 0
    for i in range(len(valid)):
        tv = [valid[j] for j in range(len(valid)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        s, t, sid, _ = valid[i]
        if nn_ret(W_E[sid]+gs*al, source_ids(s), mask) == t: hits += 1
    return hits/len(valid)

def irred(ax, holdout, mask):
    nh=0; ni=0; nt0=0
    for sw, tw in holdout:
        es, sid = get_emb(sw)
        if es is None: continue
        tc = get_tc(tw); nh += 1; found = False
        for s in np.linspace(0.02, 6.0, 60):
            if nn_ret(W_E[sid]+s*ax, source_ids(sw), mask) == tw: found=True; break
        if not found:
            ni += 1
            if tc > 1: nt0 += 1
    return (ni/nh if nh else 0.0), (nt0/max(ni,1))

def match(p, t):
    return (t.split('_')[0] in p or t in p or
            ('morph' in p and 'morph' in t) or ('phonol' in p and 'phonol' in t) or
            ('relational' in p and 'relational' in t) or ('factual' in p and 'factual' in t) or
            ('translation' in p and 'translation' in t) or ('polar' in p and 'polar' in t) or
            ('semantic' in p and 'semantic' in t))

def v15b(pc, lv, ir, sp=0.0, dig=False, t0r=0.0):
    if dig: return 'semantic_diverse'
    if pc > 0.35: return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if lv >= 0.80 and sp > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if lv >= 0.50: return 'morph_moderate' if ir < 0.40 else 'phonol_scatter'
        elif ir < 0.30: return 'morph_moderate'
        elif ir >= 0.60:
            if t0r >= 0.40 and sp < 0.07: return 'phonol_scatter'
            return 'semantic_diverse'
        return 'borderline'
    elif pc > 0.08:
        if lv >= 0.50:
            if ir >= 0.40: return 'phonol_scatter' if lv >= 0.70 else 'semantic_diverse'
            if lv >= 0.70 and ir > 0.05 and sp > 0.11: return 'morph_moderate'
            return 'phonol_scatter'
        elif ir >= 0.95:
            if t0r >= 0.70 and sp < 0.07: return 'phonol_scatter'
            return 'factual_local/translation'
        elif ir >= 0.60:
            return 'phonol_scatter' if t0r >= 0.40 else 'semantic_diverse'
        elif lv == 0.0 and 0.20 <= ir < 0.60: return 'phonol_scatter'
        elif lv == 0.0 and ir < 0.20: return 'semantic_diverse'
        elif 0.0 < lv < 0.50 and 0.20 <= ir < 0.60:
            return 'phonol_scatter' if t0r >= 0.80 else 'semantic_diverse'
        elif ir < 0.20: return 'phonol_scatter-allomorph'
        return 'borderline'
    elif pc > 0.05:
        if ir >= 0.85 and lv < 0.15: return 'translation/factual_local'
        elif lv > 0.15 and ir > 0.80: return 'polar_local-partial'
        elif lv > 0.15: return 'borderline'
        return 'polar_local'
    else:
        return 'polar_local-partial' if lv > 0.15 else 'polar_local'

def v16(pc, lv, ir, sp=0.0, dig=False, t0r=0.0):
    if dig: return 'semantic_diverse'
    if pc > 0.35: return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if lv >= 0.80 and sp > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if lv >= 0.50: return 'morph_moderate' if ir < 0.40 else 'phonol_scatter'
        elif ir < 0.30: return 'morph_moderate'
        elif ir >= 0.60:
            if t0r >= 0.40 and sp < 0.07: return 'phonol_scatter'
            return 'semantic_diverse'
        return 'borderline'
    elif pc > 0.08:
        if lv >= 0.50:
            if ir >= 0.40: return 'phonol_scatter' if lv >= 0.70 else 'semantic_diverse'
            if lv >= 0.70 and ir > 0.05 and sp > 0.11: return 'morph_moderate'
            return 'phonol_scatter'
        elif ir >= 0.95:
            if t0r >= 0.70 and sp < 0.07: return 'phonol_scatter'
            if sp < 0.03: return 'phonol_scatter'           # Rule A: un_verb
            return 'factual_local/translation'
        elif ir >= 0.60:
            if t0r >= 0.40:
                if lv == 0.0: return 'semantic_diverse'     # Rule C: er_noun2
                return 'phonol_scatter'
            return 'semantic_diverse'
        elif lv == 0.0 and 0.20 <= ir < 0.60: return 'phonol_scatter'
        elif lv == 0.0 and ir < 0.20: return 'semantic_diverse'
        elif 0.0 < lv < 0.50 and 0.20 <= ir < 0.60:
            return 'phonol_scatter' if t0r >= 0.80 else 'semantic_diverse'
        elif ir < 0.20: return 'phonol_scatter-allomorph'
        return 'borderline'
    elif pc > 0.05:
        if ir >= 0.85 and lv < 0.15: return 'translation/factual_local'
        elif lv > 0.15 and ir > 0.80: return 'polar_local-partial'
        elif lv > 0.15: return 'borderline'
        return 'polar_local'
    else:
        if ir < 0.20: return 'phonol_scatter'               # Rule B: ary
        return 'polar_local-partial' if lv > 0.15 else 'polar_local'

ABLE_M = [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable'),('comfort','comfortable'),('manage','manageable'),('reach','reachable'),('depend','dependable'),('honor','honorable'),('justify','justifiable')]
ABLE_H = [('comfort','comfortable'),('manage','manageable'),('reach','reachable')]

ORIG = [
    ('er_comp',[('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],[('dark','darker'),('soft','softer'),('heavy','heavier')],'morph_uniform'),
    ('er_sup',[('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest'),('bright','brightest'),('dark','darkest'),('soft','softest')],[('warm','warmest'),('long','longest'),('cold','coldest')],'morph_uniform'),
    ('relational',[('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')],[('Cairo','Egypt'),('Seoul','Korea'),('Lima','Peru')],'relational_geom'),
    ('al_rel',[('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal'),('origin','original'),('emotion','emotional'),('tradition','traditional')],[('history','historical'),('season','seasonal'),('accident','accidental')],'phonol_scatter'),
    ('plural',[('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees'),('book','books'),('bird','birds'),('door','doors')],[('cup','cups'),('word','words'),('room','rooms')],'morph_moderate'),
    ('3ps',[('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads'),('write','writes'),('play','plays'),('work','works')],[('talk','talks'),('sleep','sleeps'),('open','opens')],'morph_moderate'),
    ('ed_reg',[('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played'),('clean','cleaned'),('open','opened'),('start','started')],[('end','ended'),('look','looked'),('rain','rained')],'morph_moderate'),
    ('ing',[('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving'),('make','making'),('write','writing'),('read','reading')],[('eat','eating'),('work','working'),('play','playing')],'morph_moderate'),
    ('cc',[('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car'),('tree','Tree'),('river','River'),('bird','Bird')],[('cup','Cup'),('door','Door'),('word','Word')],'semantic_diverse'),
    ('ness',[('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],[('bright','brightness'),('sweet','sweetness'),('clean','cleanliness')],'phonol_scatter'),
    ('ablaut',[('go','went'),('take','took'),('give','gave'),('see','saw'),('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],[('speak','spoke'),('break','broke'),('choose','chose')],'phonol_scatter'),
    ('ablaut_t',[('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left'),('deal','dealt'),('sleep','slept'),('mean','meant')],[('burn','burned'),('learn','learned'),('smell','smelled')],'phonol_scatter'),
    ('ity',[('human','humanity'),('real','reality'),('national','nationality'),('personal','personality'),('moral','morality'),('legal','legality'),('final','finality'),('normal','normality')],[('mental','mentality'),('total','totality'),('brutal','brutality')],'phonol_scatter'),
    ('un_neg',[('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),('known','unknown'),('safe','unsafe'),('usual','unusual'),('equal','unequal')],[('stable','unstable'),('real','unreal'),('true','untrue')],'phonol_scatter'),
    ('ance',[('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),('accept','acceptance'),('appear','appearance'),('depend','dependence'),('insist','insistence')],[('persist','persistence'),('emerge','emergence'),('refer','reference')],'phonol_scatter'),
    ('ment',[('achieve','achievement'),('develop','development'),('manage','management'),('govern','government'),('engage','engagement'),('require','requirement'),('move','movement'),('improve','improvement')],[('amuse','amusement'),('punish','punishment'),('treat','treatment')],'phonol_scatter'),
    ('tion',[('act','action'),('direct','direction'),('educate','education'),('create','creation'),('produce','production'),('relate','relation'),('combine','combination'),('apply','application')],[('express','expression'),('extend','extension'),('omit','omission')],'phonol_scatter'),
    ('al_nom',[('arrive','arrival'),('propose','proposal'),('approve','approval'),('refuse','refusal'),('remove','removal'),('survive','survival'),('deny','denial'),('dispose','disposal')],[('retrieve','retrieval'),('betray','betrayal'),('renew','renewal')],'phonol_scatter'),
    ('less',[('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),('end','endless'),('home','homeless'),('harm','harmless'),('power','powerless')],[('worth','worthless'),('use','useless'),('mercy','merciless')],'phonol_scatter'),
    ('ful',[('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')],[('beauty','beautiful'),('wonder','wonderful'),('power','powerful')],'phonol_scatter'),
    ('able',ABLE_M,ABLE_H,'phonol_scatter'),
    ('er_noun',[('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner'),('manage','manager'),('build','builder'),('lead','leader')],[('write','writer'),('paint','painter'),('print','printer')],'semantic_diverse'),
    ('adj_ant',[('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('bright','dark'),('hard','soft'),('high','low'),('rich','poor')],[('open','closed'),('new','old'),('loud','quiet')],'polar_local'),
    ('antonym2',[('love','hate'),('war','peace'),('life','death'),('day','night'),('begin','end'),('give','take'),('push','pull'),('open','close')],[('rise','fall'),('win','lose'),('buy','sell')],'polar_local'),
    ('en_es',[('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día'),('night','noche'),('hand','mano'),('year','año')],[('fire','fuego'),('moon','luna'),('sea','mar')],'translation'),
    ('en_de',[('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),('day','Tag'),('night','Nacht'),('cat','Katze'),('dog','Hund')],[('fire','Feuer'),('moon','Mond'),('sea','Meer')],'translation'),
    ('en_fr',[('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),('day','jour'),('night','nuit'),('cat','chat'),('dog','chien')],[('fire','feu'),('moon','lune'),('sea','mer')],'translation'),
    ('en_zh',[('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼')],[('tree','树'),('heart','心'),('door','门')],'factual_local'),
    ('en_ja',[('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),('hand','手'),('eye','目'),('fish','魚')],[('tree','木'),('heart','心'),('door','門')],'factual_local'),
    ('num_word',[('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),('6','six'),('7','seven'),('8','eight')],[('9','nine'),('10','ten'),('0','zero')],'semantic_diverse'),
]

GEN = [
    ('er_comp2',[('old','older'),('young','younger'),('smart','smarter'),('strong','stronger'),('light','lighter'),('safe','safer'),('cheap','cheaper'),('quiet','quieter')],[('cool','cooler'),('warm','warmer'),('wide','wider')],'morph_uniform'),
    ('er_sup2',[('old','oldest'),('young','youngest'),('smart','smartest'),('strong','strongest'),('light','lightest'),('safe','safest'),('cheap','cheapest'),('quiet','quietest')],[('cool','coolest'),('warm','warmest'),('wide','widest')],'morph_uniform'),
    ('pl_reg2',[('hand','hands'),('arm','arms'),('eye','eyes'),('leg','legs'),('head','heads'),('mouth','mouths'),('face','faces'),('mind','minds')],[('heart','hearts'),('foot','foots'),('ear','ears')],'morph_moderate'),
    ('3ps2',[('live','lives'),('want','wants'),('need','needs'),('find','finds'),('keep','keeps'),('call','calls'),('feel','feels'),('turn','turns')],[('show','shows'),('hold','holds'),('move','moves')],'morph_moderate'),
    ('ing2',[('live','living'),('want','wanting'),('need','needing'),('find','finding'),('keep','keeping'),('call','calling'),('feel','feeling'),('turn','turning')],[('show','showing'),('hold','holding'),('move','moving')],'morph_moderate'),
    ('er_2syl',[('happy','happier'),('easy','easier'),('busy','busier'),('early','earlier'),('heavy','heavier'),('pretty','prettier'),('funny','funnier'),('angry','angrier')],[('lucky','luckier'),('noisy','noisier'),('cloudy','cloudier')],'morph_moderate'),
    ('pl_irr',[('foot','feet'),('tooth','teeth'),('man','men'),('woman','women'),('mouse','mice'),('goose','geese'),('child','children'),('person','people')],[('ox','oxen'),('die','dice'),('louse','lice')],'phonol_scatter'),
    ('past_ab',[('swim','swam'),('sing','sang'),('ring','rang'),('drink','drank'),('sink','sank'),('begin','began'),('run','ran'),('spring','sprang')],[('shrink','shrank'),('sink','sank'),('blow','blew')],'phonol_scatter'),
    ('ize',[('modern','modernize'),('local','localize'),('real','realize'),('social','socialize'),('legal','legalize'),('private','privatize'),('organ','organize'),('terror','terrorize')],[('civil','civilize'),('final','finalize'),('vital','vitalize')],'phonol_scatter'),
    ('ous',[('danger','dangerous'),('poison','poisonous'),('fame','famous'),('nerve','nervous'),('humor','humorous'),('hazard','hazardous'),('glory','glorious'),('courage','courageous')],[('vigor','vigorous'),('mystery','mysterious'),('joy','joyous')],'phonol_scatter'),
    ('en',[('dark','darken'),('bright','brighten'),('hard','harden'),('sharp','sharpen'),('deep','deepen'),('wide','widen'),('loose','loosen'),('tight','tighten')],[('soft','soften'),('weak','weaken'),('thick','thicken')],'phonol_scatter'),
    ('ish',[('child','childish'),('self','selfish'),('fool','foolish'),('fever','feverish'),('clown','clownish'),('book','bookish'),('baby','babyish'),('snob','snobbish')],[('freak','freakish'),('wolf','wolfish'),('oaf','oafish')],'phonol_scatter'),
    ('ist',[('art','artist'),('real','realist'),('novel','novelist'),('journal','journalist'),('tour','tourist'),('capital','capitalist'),('social','socialist'),('final','finalist')],[('piano','pianist'),('guitar','guitarist'),('terror','terrorist')],'phonol_scatter'),
    ('ism',[('real','realism'),('social','socialism'),('capital','capitalism'),('human','humanism'),('symbol','symbolism'),('terror','terrorism'),('ideal','idealism'),('national','nationalism')],[('natural','naturalism'),('rational','rationalism'),('plural','pluralism')],'phonol_scatter'),
    ('ness2',[('cold','coldness'),('bold','boldness'),('calm','calmness'),('fresh','freshness'),('rich','richness'),('wild','wildness'),('neat','neatness'),('raw','rawness')],[('brave','braveness'),('free','freeness'),('fair','fairness')],'morph_uniform'),
    ('ward',[('north','northward'),('south','southward'),('east','eastward'),('west','westward'),('home','homeward'),('up','upward'),('in','inward'),('out','outward')],[('back','backward'),('for','forward'),('down','downward')],'morph_uniform'),
    ('re_pfx',[('try','retry'),('do','redo'),('write','rewrite'),('start','restart'),('build','rebuild'),('read','reread'),('use','reuse'),('think','rethink')],[('turn','return'),('view','review'),('place','replace')],'semantic_diverse'),
    ('pre_pfx',[('view','preview'),('heat','preheat'),('pay','prepay'),('cook','precook'),('treat','pretreat'),('warn','prewarn'),('select','preselect'),('test','pretest')],[('school','preschool'),('order','preorder'),('set','preset')],'phonol_scatter'),
    ('un_verb',[('lock','unlock'),('wrap','unwrap'),('tie','untie'),('fold','unfold'),('pack','unpack'),('dress','undress'),('cover','uncover'),('do','undo')],[('load','unload'),('zip','unzip'),('plug','unplug')],'phonol_scatter'),
    ('ary',[('element','elementary'),('moment','momentary'),('comment','commentary'),('legend','legendary'),('custom','customary'),('vision','visionary'),('honor','honorary'),('mission','missionary')],[('revolution','revolutionary'),('parliament','parliamentary'),('discipline','disciplinary')],'phonol_scatter'),
    ('tion2',[('invent','invention'),('observe','observation'),('explain','explanation'),('object','objection'),('describe','description'),('destroy','destruction'),('celebrate','celebration'),('compose','composition')],[('oppose','opposition'),('distribute','distribution'),('contrast','contradiction')],'phonol_scatter'),
    ('er_noun2',[('play','player'),('sing','singer'),('report','reporter'),('hack','hacker'),('surf','surfer'),('climb','climber'),('swim','swimmer'),('box','boxer')],[('run','runner'),('skate','skater'),('cycle','cyclist')],'semantic_diverse'),
    ('gender_pr',[('king','queen'),('man','woman'),('boy','girl'),('father','mother'),('son','daughter'),('husband','wife'),('uncle','aunt'),('prince','princess')],[('actor','actress'),('waiter','waitress'),('hero','heroine')],'morph_moderate'),
    ('num_ord',[('one','first'),('two','second'),('three','third'),('four','fourth'),('five','fifth'),('six','sixth'),('seven','seventh'),('eight','eighth')],[('nine','ninth'),('ten','tenth'),('eleven','eleventh')],'morph_uniform'),
    ('adj_ant2',[('clean','dirty'),('right','wrong'),('true','false'),('early','late'),('cheap','expensive'),('safe','dangerous'),('simple','complex'),('quiet','loud')],[('open','closed'),('full','empty'),('public','private')],'polar_local'),
    ('abstract_ant',[('success','failure'),('victory','defeat'),('reward','punishment'),('praise','blame'),('courage','cowardice'),('freedom','slavery'),('truth','lie'),('order','chaos')],[('rise','fall'),('creation','destruction'),('unity','division')],'polar_local'),
    ('en_it',[('house','casa'),('water','acqua'),('sun','sole'),('book','libro'),('day','giorno'),('cat','gatto'),('dog','cane'),('fire','fuoco')],[('night','notte'),('moon','luna'),('sea','mare')],'translation'),
    ('en_nl',[('house','huis'),('water','water'),('sun','zon'),('book','boek'),('day','dag'),('cat','kat'),('dog','hond'),('fire','vuur')],[('night','nacht'),('moon','maan'),('sea','zee')],'translation'),
    ('en_pt',[('house','casa'),('water','água'),('sun','sol'),('fire','fogo'),('day','dia'),('cat','gato'),('dog','cachorro'),('night','noite')],[('moon','lua'),('sea','mar'),('tree','árvore')],'translation'),
    ('en_zh2',[('big','大'),('small','小'),('good','好'),('new','新'),('old','老'),('high','高'),('low','低'),('long','长')],[('short','短'),('wide','宽'),('deep','深')],'semantic_diverse'),
]

def run_bench(bench, fn, lbl):
    print("  %s..." % lbl, flush=True)
    sc = 0; rows = []
    for name, train, ho, true in bench:
        ax, valid, pc, sp = axis_spread(train)
        if ax is None or len(valid) < 2: continue
        lv = loo(ax, valid, RELAXED_MASK)
        ir, t0r = irred(ax, ho, RELAXED_MASK)
        dig = all(tok.decode([sid]).strip().isdigit() for _,_,sid,_ in valid)
        p = fn(pc, lv, ir, sp, dig, t0r)
        ok = match(p, true)
        if ok: sc += 1
        rows.append((name, pc, lv, ir, t0r, sp, p, true, ok))
    return sc, len(rows), rows

print("\nDAY 345: v16 confirmation with full source_ids")
print("="*65)

o15, ot, or15 = run_bench(ORIG, v15b, 'v15b orig')
o16, _,  or16 = run_bench(ORIG, v16,  'v16  orig')
g15, gt, gr15 = run_bench(GEN,  v15b, 'v15b gen')
g16, _,  gr16 = run_bench(GEN,  v16,  'v16  gen')

total = ot + gt

print()
print("PART A: orig bench")
print("  v15b: %d/%d = %.0f%%" % (o15, ot, 100*o15/ot))
print("  v16:  %d/%d = %.0f%%" % (o16, ot, 100*o16/ot))
diffs = [(r5,r6) for r5,r6 in zip(or15,or16) if r5[6]!=r6[6]]
if diffs:
    for r5, r6 in diffs:
        n,pc,lv,ir,t0r,sp,p5,true,ok5=r5; _,_,_,_,_,_,p6,_,ok6=r6
        tag = 'v16+' if (not ok5 and ok6) else ('v16-' if (ok5 and not ok6) else '~')
        print("  -> %-12s pc=%.3f sp=%.3f ir=%.2f lv=%.0f%% v15b=%s v16=%s true=%s %s"%(n,pc,sp,ir,lv*100,p5[:12],p6[:12],true,tag))
else:
    print("  No changes between v15b and v16 on orig.")
failing = [(r[0],r[6],r[7]) for r in or16 if not r[8]]
if failing:
    for n,p,t in failing:
        print("  FAIL: %-12s pred=%s true=%s" % (n, p[:20], t))

print()
print("PART B: gen bench")
print("  v15b: %d/%d = %.0f%%" % (g15, gt, 100*g15/gt))
print("  v16:  %d/%d = %.0f%%" % (g16, gt, 100*g16/gt))
for r5, r6 in zip(gr15, gr16):
    n,pc,lv,ir,t0r,sp,p5,true,ok5=r5; _,_,_,_,_,_,p6,_,ok6=r6
    if p5 != p6:
        tag = 'v16+' if (not ok5 and ok6) else ('v16-' if (ok5 and not ok6) else '~')
        print("  -> %-12s pc=%.3f sp=%.3f ir=%.2f lv=%.0f%% v15b=%s v16=%s true=%s %s"%(n,pc,sp,ir,lv*100,p5[:12],p6[:12],true,tag))
failing_g = [(r[0],r[6],r[7]) for r in gr16 if not r[8]]
if failing_g:
    print("  Remaining failures:")
    for n,p,t in failing_g:
        print("    FAIL: %-12s pred=%s true=%s" % (n, p[:20], t))

print()
print("PART C: combined summary")
print("  v15b: orig=%d/%d gen=%d/%d combined=%d/%d = %.0f%%" % (o15,ot,g15,gt,o15+g15,total,100*(o15+g15)/total))
print("  v16:  orig=%d/%d gen=%d/%d combined=%d/%d = %.0f%%" % (o16,ot,g16,gt,o16+g16,total,100*(o16+g16)/total))
