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

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or
            ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or
            ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

# =====================================================================
# 30 NEW axes — not seen during v12 design
# =====================================================================
GEN_BENCH = [
    # --- morph_uniform ---
    ('er_comp2',  [('old','older'),('young','younger'),('smart','smarter'),('strong','stronger'),
                   ('light','lighter'),('safe','safer'),('cheap','cheaper'),('quiet','quieter')],
                  [('cool','cooler'),('warm','warmer'),('wide','wider')],           'morph_uniform'),
    ('er_sup2',   [('old','oldest'),('young','youngest'),('smart','smartest'),('strong','strongest'),
                   ('light','lightest'),('safe','safest'),('cheap','cheapest'),('quiet','quietest')],
                  [('cool','coolest'),('warm','warmest'),('wide','widest')],         'morph_uniform'),
    # --- morph_moderate ---
    ('pl_reg2',   [('hand','hands'),('arm','arms'),('eye','eyes'),('leg','legs'),
                   ('head','heads'),('mouth','mouths'),('face','faces'),('mind','minds')],
                  [('heart','hearts'),('foot','foots'),('ear','ears')],             'morph_moderate'),
    ('3ps2',      [('live','lives'),('want','wants'),('need','needs'),('find','finds'),
                   ('keep','keeps'),('call','calls'),('feel','feels'),('turn','turns')],
                  [('show','shows'),('hold','holds'),('move','moves')],             'morph_moderate'),
    ('ing2',      [('live','living'),('want','wanting'),('need','needing'),('find','finding'),
                   ('keep','keeping'),('call','calling'),('feel','feeling'),('turn','turning')],
                  [('show','showing'),('hold','holding'),('move','moving')],        'morph_moderate'),
    ('er_2syl',   [('happy','happier'),('easy','easier'),('busy','busier'),('early','earlier'),
                   ('heavy','heavier'),('pretty','prettier'),('funny','funnier'),('angry','angrier')],
                  [('lucky','luckier'),('noisy','noisier'),('cloudy','cloudier')],  'morph_moderate'),
    # --- phonol_scatter ---
    ('pl_irr',    [('foot','feet'),('tooth','teeth'),('man','men'),('woman','women'),
                   ('mouse','mice'),('goose','geese'),('child','children'),('person','people')],
                  [('ox','oxen'),('die','dice'),('louse','lice')],                  'phonol_scatter'),
    ('past_ab',   [('swim','swam'),('sing','sang'),('ring','rang'),('drink','drank'),
                   ('sink','sank'),('begin','began'),('run','ran'),('spring','sprang')],
                  [('shrink','shrank'),('sink','sank'),('blow','blew')],            'phonol_scatter'),
    ('ize',       [('modern','modernize'),('local','localize'),('real','realize'),('social','socialize'),
                   ('legal','legalize'),('private','privatize'),('organ','organize'),('terror','terrorize')],
                  [('civil','civilize'),('final','finalize'),('vital','vitalize')],  'phonol_scatter'),
    ('ous',       [('danger','dangerous'),('poison','poisonous'),('fame','famous'),('nerve','nervous'),
                   ('humor','humorous'),('hazard','hazardous'),('glory','glorious'),('courage','courageous')],
                  [('vigor','vigorous'),('mystery','mysterious'),('joy','joyous')],  'phonol_scatter'),
    ('en',        [('dark','darken'),('bright','brighten'),('hard','harden'),('sharp','sharpen'),
                   ('deep','deepen'),('wide','widen'),('loose','loosen'),('tight','tighten')],
                  [('soft','soften'),('weak','weaken'),('thick','thicken')],        'phonol_scatter'),
    ('ish',       [('child','childish'),('self','selfish'),('fool','foolish'),('fever','feverish'),
                   ('clown','clownish'),('book','bookish'),('baby','babyish'),('snob','snobbish')],
                  [('freak','freakish'),('wolf','wolfish'),('oaf','oafish')],        'phonol_scatter'),
    ('ist',       [('art','artist'),('real','realist'),('novel','novelist'),('journal','journalist'),
                   ('tour','tourist'),('capital','capitalist'),('social','socialist'),('final','finalist')],
                  [('piano','pianist'),('guitar','guitarist'),('terror','terrorist')], 'phonol_scatter'),
    ('ism',       [('real','realism'),('social','socialism'),('capital','capitalism'),('human','humanism'),
                   ('symbol','symbolism'),('terror','terrorism'),('ideal','idealism'),('national','nationalism')],
                  [('natural','naturalism'),('rational','rationalism'),('plural','pluralism')], 'phonol_scatter'),
    ('ness2',     [('cold','coldness'),('bold','boldness'),('calm','calmness'),('fresh','freshness'),
                   ('rich','richness'),('wild','wildness'),('neat','neatness'),('raw','rawness')],
                  [('brave','braveness'),('free','freeness'),('fair','fairness')],  'phonol_scatter'),
    ('ward',      [('north','northward'),('south','southward'),('east','eastward'),('west','westward'),
                   ('home','homeward'),('up','upward'),('in','inward'),('out','outward')],
                  [('back','backward'),('for','forward'),('down','downward')],      'phonol_scatter'),
    ('re_pfx',    [('try','retry'),('do','redo'),('write','rewrite'),('start','restart'),
                   ('build','rebuild'),('read','reread'),('use','reuse'),('think','rethink')],
                  [('turn','return'),('view','review'),('place','replace')],         'phonol_scatter'),
    ('pre_pfx',   [('view','preview'),('heat','preheat'),('pay','prepay'),('cook','precook'),
                   ('treat','pretreat'),('warn','prewarn'),('select','preselect'),('test','pretest')],
                  [('school','preschool'),('order','preorder'),('set','preset')],    'phonol_scatter'),
    ('un_verb',   [('lock','unlock'),('wrap','unwrap'),('tie','untie'),('fold','unfold'),
                   ('pack','unpack'),('dress','undress'),('cover','uncover'),('do','undo')],
                  [('load','unload'),('zip','unzip'),('plug','unplug')],            'phonol_scatter'),
    ('ary',       [('element','elementary'),('moment','momentary'),('comment','commentary'),
                   ('legend','legendary'),('custom','customary'),('vision','visionary'),
                   ('honor','honorary'),('mission','missionary')],
                  [('revolution','revolutionary'),('parliament','parliamentary'),('discipline','disciplinary')], 'phonol_scatter'),
    ('tion2',     [('invent','invention'),('observe','observation'),('explain','explanation'),
                   ('object','objection'),('describe','description'),('destroy','destruction'),
                   ('celebrate','celebration'),('compose','composition')],
                  [('oppose','opposition'),('distribute','distribution'),('contrast','contradiction')], 'phonol_scatter'),
    # --- semantic_diverse ---
    ('er_noun2',  [('play','player'),('sing','singer'),('report','reporter'),('hack','hacker'),
                   ('surf','surfer'),('climb','climber'),('swim','swimmer'),('box','boxer')],
                  [('run','runner'),('skate','skater'),('cycle','cyclist')],        'semantic_diverse'),
    ('gender_pr', [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                   ('son','daughter'),('husband','wife'),('uncle','aunt'),('prince','princess')],
                  [('actor','actress'),('waiter','waitress'),('hero','heroine')],   'semantic_diverse'),
    ('num_ord',   [('one','first'),('two','second'),('three','third'),('four','fourth'),
                   ('five','fifth'),('six','sixth'),('seven','seventh'),('eight','eighth')],
                  [('nine','ninth'),('ten','tenth'),('eleven','eleventh')],          'semantic_diverse'),
    # --- polar_local ---
    ('adj_ant2',  [('clean','dirty'),('right','wrong'),('true','false'),('early','late'),
                   ('cheap','expensive'),('safe','dangerous'),('simple','complex'),('quiet','loud')],
                  [('open','closed'),('full','empty'),('public','private')],        'polar_local'),
    ('abstract_ant',[('success','failure'),('victory','defeat'),('reward','punishment'),
                     ('praise','blame'),('courage','cowardice'),('freedom','slavery'),
                     ('truth','lie'),('order','chaos')],
                  [('rise','fall'),('creation','destruction'),('unity','division')], 'polar_local'),
    # --- translation ---
    ('en_it',     [('house','casa'),('water','acqua'),('sun','sole'),('book','libro'),
                   ('day','giorno'),('cat','gatto'),('dog','cane'),('fire','fuoco')],
                  [('night','notte'),('moon','luna'),('sea','mare')],               'translation'),
    ('en_nl',     [('house','huis'),('water','water'),('sun','zon'),('book','boek'),
                   ('day','dag'),('cat','kat'),('dog','hond'),('fire','vuur')],
                  [('night','nacht'),('moon','maan'),('sea','zee')],                'translation'),
    ('en_pt',     [('house','casa'),('water','água'),('sun','sol'),('fire','fogo'),
                   ('day','dia'),('cat','gato'),('dog','cachorro'),('night','noite')],
                  [('moon','lua'),('sea','mar'),('tree','árvore')],                 'translation'),
    # --- factual_local ---
    ('en_zh2',    [('big','大'),('small','小'),('good','好'),('new','新'),
                   ('old','老'),('high','高'),('low','低'),('long','长')],
                  [('short','短'),('wide','宽'),('deep','深')],                     'factual_local'),
]

print()
print("DAY 339: v12 GENERALIZATION TEST — 30 NEW AXES")
print("="*80)

# Category breakdown
from collections import Counter
cat_dist = Counter(t for _,_,_,t in GEN_BENCH)
print("\n  Axis category distribution:")
for cat, count in sorted(cat_dist.items()): print("  %-22s %d" % (cat, count))
print()

# Check single-token coverage
print("  Checking single-token coverage for each axis...")
n_total = 0; n_valid = 0
for name, pairs, holdout, true in GEN_BENCH:
    n_p = 0; n_ok = 0
    for s, t in pairs:
        n_p += 1
        es, sid = get_emb(s)
        et, tid = get_emb(t)
        if es is not None and et is not None: n_ok += 1
    if n_ok < n_p:
        print("  %-14s  %d/%d train pairs have single-token forms" % (name, n_ok, n_p))
    n_total += n_p; n_valid += n_ok
print("  Overall: %d/%d = %.0f%% train pairs are single-token" %
      (n_valid, n_total, 100*n_valid/n_total))
print()

# Run v12 on all 30 new axes
print("Computing metrics and running v12...")
print()
print("  %-14s  pc     loo   irred  t0r   spread  v12_pred             expected_type        match" % "axis")
print("  " + "-"*110)

score = 0
total = 0
results_by_cat = {}
for name, train_pairs, holdout_pairs, expected_type in GEN_BENCH:
    ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
    if ax is None or len(valid) < 2:
        print("  %-14s  SKIP (too few valid pairs)" % name)
        continue
    loo_v  = axis_loo(ax, valid, RELAXED_MASK)
    irr, t0r = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
    src_is_digit = all(tok.decode([sid]).strip().isdigit()
                       for _,_,sid,_ in valid)
    pred = classify_v12(pc, loo_v, irr, spread, src_is_digit, t0r)
    ok = match(pred, expected_type)
    if ok: score += 1
    total += 1
    flag = '✓' if ok else '✗'
    marker = '  ' if ok else '->'
    print("  %s %-12s  %.3f  %.0f%%  %.2f   %.2f  %.3f  %-20s %-20s %s" %
          (marker, name, pc, 100*loo_v, irr, t0r, spread, pred[:20], expected_type, flag))
    if expected_type not in results_by_cat:
        results_by_cat[expected_type] = {'ok': 0, 'total': 0, 'failures': []}
    results_by_cat[expected_type]['total'] += 1
    if ok: results_by_cat[expected_type]['ok'] += 1
    else:  results_by_cat[expected_type]['failures'].append((name, pred))

print()
print("  GENERALIZATION RESULT: %d/%d = %.0f%%" % (score, total, 100*score/total))
print()
print("  By category:")
for cat in sorted(results_by_cat.keys()):
    d = results_by_cat[cat]
    pct = 100*d['ok']/d['total'] if d['total'] else 0
    fails = ', '.join('%s→%s' % (n,p) for n,p in d['failures'])
    fail_str = ('  FAIL: '+fails) if fails else ''
    print("  %-22s  %d/%d = %.0f%%%s" % (cat, d['ok'], d['total'], pct, fail_str))
print()

# Cross-reference: compare generalization vs original benchmark scores
print("  Summary comparison:")
print("  v12 on ORIGINAL 30-axis benchmark (v12 labels): 30/30 = 100%%")
print("  v12 on NEW     30-axis generalization test:     %d/30 = %.0f%%" %
      (score, 100*score/total))
print()
if score < total:
    print("  Failures analysis:")
    for name, train_pairs, holdout_pairs, expected_type in GEN_BENCH:
        ax, valid, pc, spread = compute_axis_with_spread(train_pairs)
        if ax is None or len(valid) < 2: continue
        loo_v  = axis_loo(ax, valid, RELAXED_MASK)
        irr, t0r = irred_with_type0_ratio(ax, holdout_pairs, RELAXED_MASK)
        src_is_digit = all(tok.decode([sid]).strip().isdigit() for _,_,sid,_ in valid)
        pred = classify_v12(pc, loo_v, irr, spread, src_is_digit, t0r)
        ok = match(pred, expected_type)
        if not ok:
            print("  FAIL %-14s  pc=%.3f  loo=%.0f%%  irred=%.2f  t0r=%.2f  spread=%.3f" %
                  (name, pc, 100*loo_v, irr, t0r, spread))
            print("       pred=%-22s  expected=%-18s" % (pred, expected_type))
            print("       Is expected label wrong? Geometric profile suggests: %s" % pred)
            print()
