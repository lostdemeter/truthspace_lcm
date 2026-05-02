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

def chord_spread(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return 0.0, 0.0, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    mean_axis = normed(np.mean(chords, axis=0)).astype(np.float32)
    cos_to_mean = [float(np.dot(cn[i], mean_axis)) for i in range(len(cn))]
    mag_chords = [np.linalg.norm(c) for c in chords]
    pc = float(np.mean([np.dot(cn[i], cn[j])
                        for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return pc, float(np.std(cos_to_mean)), float(np.mean(mag_chords))

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

def irred_on_holdout_typed(axis, holdout, mask, lo=0.02, hi=6.0, n=60, top_k=10):
    """Returns (irred_rate, type1_count, type2_count, details)."""
    irred=0; type1=0; type2=0; n_ho=0; details=[]
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1; found_at = None; best_nn = None; best_sim = 0.0
        for s in np.linspace(lo, hi, n):
            pred = W_E[sid] + s * axis
            r = nn_retrieve(pred, source_ids(s_w), mask, top_k)
            if r[0][0] == t_w:
                found_at = s; break
            # Track best sim to target at any scale
            pred_n = normed(pred).astype(np.float32)
            et, tid = get_emb(t_w)
            if et is not None:
                sim = float(np.dot(pred_n, normed(et).astype(np.float32)))
                if sim > best_sim:
                    best_sim = sim
                    best_nn = [x[0] for x in r[:3]]
        if found_at is None:
            irred += 1
            # Check if top-k NN at best scale includes semantically related word
            # Type 2: target is multi-token but a synonym/related word is in top-k
            # Heuristic: if best_sim > 0.85, target is geometrically reachable (Type 2)
            # If best_sim < 0.75, target region is wrong (Type 1)
            et, _ = get_emb(t_w)
            if et is not None:
                if best_sim >= 0.85:
                    type2 += 1
                    irred_class = 'Type2(vocab)'
                elif best_sim >= 0.75:
                    type2 += 1
                    irred_class = 'Type2(borderline)'
                else:
                    type1 += 1
                    irred_class = 'Type1(geometric)'
            else:
                type1 += 1
                irred_class = 'Type1(no_emb)'
            details.append((s_w, t_w, 'IRRED', irred_class, best_sim))
        else:
            details.append((s_w, t_w, 'found@%.2f'%found_at, 'OK', 0.0))
    return irred/n_ho if n_ho else 0.0, type1, type2, n_ho, details

print()
print("DAY 333: PREDICTOR V7, IRRED TYPE 1/2, SOURCE POP HOMOLOGY, CHAIN GRAPH, +al_rel")
print("="*80)
print()

# =====================================================================
# PART A: PREDICTOR V7 — SPREAD RULE FOR HIGH-pc AXES
# =====================================================================
print("PART A: Predictor v7 benchmark (spread rule: pc > 0.30 → spread disambiguates)")
print("-"*80)

def classify_v7(pc, loo, irred, spread, n_train=8):
    """Predictor v7: adds spread disambiguation for high-pc axes."""
    if pc > 0.35:
        if spread > 0.07:  return 'phonol_scatter'   # ablaut-type
        else:              return 'morph_uniform'
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
     [('talk','talked'),('end','ended'),('look','looked')], 'morph_moderate'),
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
     [('bright','brightness'),('clean','cleanliness'),('sweet','sweetness')], 'phonol_scatter'),
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

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

v7_correct = 0
print("  %-12s  pc    spread  pred                   true        ok?" % "axis")
print("  " + "-"*72)
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2: continue
    pc2, sp, _ = chord_spread(train_pairs)
    from itertools import islice
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr_f, _, _ = irred_on_holdout_typed(ax, holdout_pairs, RELAXED_MASK)[:3]
    n = len(valid)
    pred = classify_v7(pc, loo_v, irr_f, sp, n)
    ok = match(pred, true_type)
    if ok: v7_correct += 1
    tick = '✓' if ok else '✗'
    print("  %s %-12s  pc=%.3f  s=%.3f  %-22s %-12s" %
          (tick, name, pc, sp, pred[:22], true_type))
print()
print("  v7 accuracy: %d/30 = %.0f%%" % (v7_correct, 100*v7_correct/30))
print()

# =====================================================================
# PART B: TYPE 1 vs TYPE 2 IRRED — GEOMETRIC vs VOCABULARY
# =====================================================================
print("PART B: Type 1 vs Type 2 irred for key axes")
print("-"*80)

IRRED_TEST_AXES = [
    ('+ize',
     [('organ','organize'),('legal','legalize'),('minimal','minimize'),('real','realize'),
      ('local','localize'),('final','finalize'),('general','generalize'),('moral','moralize')],
     [('popular','popularize'),('equal','equalize'),('visual','visualize'),
      ('crystal','crystallize'),('neutral','neutralize'),('normal','normalize'),
      ('standard','standardize'),('active','activate')]),
    ('+tion',
     [('act','action'),('direct','direction'),('educate','education'),('create','creation'),
      ('produce','production'),('relate','relation'),('combine','combination'),('apply','application')],
     [('express','expression'),('extend','extension'),('omit','omission'),
      ('admit','admission'),('permit','permission'),('submit','submission')]),
    ('+ment',
     [('achieve','achievement'),('develop','development'),('manage','management'),
      ('govern','government'),('engage','engagement'),('require','requirement'),
      ('move','movement'),('improve','improvement')],
     [('amuse','amusement'),('punish','punishment'),('treat','treatment'),
      ('judge','judgment'),('argue','argument'),('settle','settlement')]),
    ('+able',
     [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
      ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')],
     [('comfort','comfortable'),('manage','manageable'),('reach','reachable'),
      ('note','notable'),('remark','remarkable'),('reason','reasonable')]),
]

for axis_name, train_pairs, holdout_pairs in IRRED_TEST_AXES:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None: continue
    irr, type1, type2, n_ho, details = irred_on_holdout_typed(ax, holdout_pairs, CLEAN_MASK)
    print("  %s: pc=%.4f  irred=%.0f%%  Type1(geom)=%d  Type2(vocab)=%d  n=%d" %
          (axis_name, pc, 100*irr, type1, type2, n_ho))
    for s_w, t_w, status, irred_class, best_sim in details:
        ids_t = tok(' '+t_w, add_special_tokens=False)['input_ids']
        print("    %-12s -> %-15s  %-20s  sim=%.3f  [%d tok]" %
              (s_w, t_w, irred_class if status.startswith('IRRED') else 'OK', best_sim, len(ids_t)))
    print()

# =====================================================================
# PART C: SOURCE POPULATION HOMOLOGY SEARCH
# =====================================================================
print("PART C: Source population homology — finding high cross-group cosines")
print("-"*80)

# Test various axis pairs and measure their cosines + source population overlap
HOMOLOGY_TESTS = [
    # Germanic adj sources for both
    ('+en', [('bright','brighten'),('dark','darken'),('hard','harden'),('wide','widen'),
              ('soft','soften'),('fresh','freshen'),('weak','weaken'),('sharp','sharpen')],
     '+er_comp', [('fast','faster'),('slow','slower'),('bright','brighter'),('dark','darker'),
                   ('soft','softer'),('warm','warmer'),('tall','taller'),('clean','cleaner')]),
    # Latin adj for both
    ('+ize', [('organ','organize'),('moral','moralize'),('legal','legalize'),('real','realize'),
               ('local','localize'),('final','finalize'),('general','generalize'),('national','nationalize')],
     '+ity', [('human','humanity'),('real','reality'),('national','nationality'),
               ('moral','morality'),('legal','legality'),('final','finality'),
               ('personal','personality'),('mental','mentality')]),
    # Germanic verb sources for both
    ('+able_germ', [('read','readable'),('wash','washable'),('break','breakable'),
                     ('love','lovable'),('use','usable'),('take','takeable'),
                     ('make','makeable'),('work','workable')],
     '+3ps_germ', [('run','runs'),('walk','walks'),('read','reads'),('break','breaks'),
                    ('take','takes'),('make','makes'),('work','works'),('play','plays')]),
    # Noun sources for translation axes
    ('en_es', [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),
                ('day','día'),('night','noche'),('hand','mano'),('year','año')],
     'en_de', [('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),
                ('day','Tag'),('night','Nacht'),('cat','Katze'),('dog','Hund')]),
    # Adj antonym pairs (same source pop)
    ('adj_ant1', [('good','bad'),('hot','cold'),('fast','slow'),('big','small')],
     'adj_ant2', [('bright','dark'),('hard','soft'),('high','low'),('rich','poor')]),
    # Related verb families
    ('+3ps_motion', [('run','runs'),('walk','walks'),('jump','jumps'),('fly','flies'),
                      ('swim','swims'),('fall','falls'),('climb','climbs'),('move','moves')],
     '+3ps_cognit', [('think','thinks'),('know','knows'),('feel','feels'),('want','wants'),
                      ('see','sees'),('hear','hears'),('say','says'),('mean','means')]),
]

print("  Testing source population homology:")
for n1, p1, n2, p2 in HOMOLOGY_TESTS:
    ax1, _, pc1 = compute_axis(p1)
    ax2, _, pc2 = compute_axis(p2)
    if ax1 is None or ax2 is None: continue
    c = float(np.dot(ax1.astype(np.float32), ax2.astype(np.float32)))
    # Source population overlap: count shared source words
    src1 = set(s for s,t in p1)
    src2 = set(s for s,t in p2)
    overlap = len(src1 & src2)
    flag = '***' if abs(c) > 0.30 else ('  *' if abs(c) > 0.20 else '   ')
    print("  %s cos(%-16s, %-16s) = %+.4f  overlap=%d/%d" %
          (flag, n1, n2, c, overlap, min(len(src1), len(src2))))
print()

# =====================================================================
# PART D: MORPHOLOGICAL CHAIN GRAPH
# =====================================================================
print("PART D: Morphological chain graph — which adj→noun chains work?")
print("-"*80)

# Build all axes needed
ax_en, valid_en, _ = compute_axis(
    [('bright','brighten'),('dark','darken'),('hard','harden'),('wide','widen'),
     ('soft','soften'),('fresh','freshen'),('weak','weaken'),('sharp','sharpen')])
ax_ize, valid_ize, _ = compute_axis(
    [('organ','organize'),('moral','moralize'),('legal','legalize'),('real','realize'),
     ('local','localize'),('final','finalize'),('general','generalize'),('national','nationalize')])
ax_ance, valid_ance, _ = compute_axis(
    [('perform','performance'),('exist','existence'),('enter','entrance'),
     ('resist','resistance'),('accept','acceptance'),('appear','appearance')])
ax_ment, valid_ment, _ = compute_axis(
    [('achieve','achievement'),('develop','development'),('manage','management'),
     ('govern','government'),('engage','engagement'),('require','requirement')])
ax_tion, valid_tion, _ = compute_axis(
    [('act','action'),('direct','direction'),('educate','education'),
     ('create','creation'),('produce','production'),('relate','relation')])
ax_ness, valid_ness, _ = compute_axis(
    [('happy','happiness'),('kind','kindness'),('sad','sadness'),
     ('bright','brightness'),('dark','darkness'),('soft','softness')])
ax_ity, valid_ity, _ = compute_axis(
    [('human','humanity'),('real','reality'),('national','nationality'),
     ('personal','personality'),('moral','morality'),('legal','legality')])

# Scales
bs_en = best_scale(ax_en, valid_en, CLEAN_MASK)[0] if ax_en is not None else 0.64
bs_ize = best_scale(ax_ize, valid_ize, CLEAN_MASK)[0] if ax_ize is not None else 0.84
bs_ance = best_scale(ax_ance, valid_ance, CLEAN_MASK)[0] if ax_ance is not None else 0.5
bs_ment = best_scale(ax_ment, valid_ment, CLEAN_MASK)[0] if ax_ment is not None else 0.5
bs_tion = best_scale(ax_tion, valid_tion, CLEAN_MASK)[0] if ax_tion is not None else 0.5
bs_ness = best_scale(ax_ness, valid_ness, CLEAN_MASK)[0] if ax_ness is not None else 0.5
bs_ity  = best_scale(ax_ity, valid_ity, CLEAN_MASK)[0] if ax_ity is not None else 0.5

# Chain 1: adj -> verb (+en) -> event_noun (+ance/ment/tion)
print("  Chain 1: adj → verb(+en) → event_noun")
test_adj_chain1 = [('bright','brighten','brightness'),('dark','darken','darkness'),
                    ('wide','widen','wideness'),('soft','soften','softness'),
                    ('weak','weaken','weakness'),('hard','harden','hardness')]
print("  %-8s  %-12s  %-12s  %-12s  %-12s" % ('adj', '+en(verb)', '+ance', '+ment', '+tion'))
for adj, verb, noun_expected in test_adj_chain1:
    es, sid = get_emb(adj)
    if es is None or ax_en is None: continue
    # Step 1: adj -> verb
    step1 = W_E[sid] + bs_en * ax_en
    r1 = nn_retrieve(step1, source_ids(adj), RELAXED_MASK, 1)
    got_verb = r1[0][0]; v1_id = r1[0][2]
    # Step 2a: verb -> +ance noun
    step2a = W_E[v1_id] + bs_ance * ax_ance
    r2a = nn_retrieve(step2a, source_ids(got_verb), CLEAN_MASK, 1)
    # Step 2b: verb -> +ment noun
    step2b = W_E[v1_id] + bs_ment * ax_ment
    r2b = nn_retrieve(step2b, source_ids(got_verb), CLEAN_MASK, 1)
    # Step 2c: verb -> +tion noun
    step2c = W_E[v1_id] + bs_tion * ax_tion
    r2c = nn_retrieve(step2c, source_ids(got_verb), CLEAN_MASK, 1)
    ok = '✓' if got_verb == verb else '~'
    print("  %s %-8s  %-12s  %-12s  %-12s  %-12s" %
          (ok, adj, got_verb, r2a[0][0], r2b[0][0], r2c[0][0]))
print()

# Chain 2: adj -> +ize -> event_noun (ize then ance/ment/tion)
print("  Chain 2: adj → verb(+ize) → event_noun")
test_adj_chain2 = [('moral','moralize'),('legal','legalize'),('national','nationalize'),
                    ('local','localize'),('real','realize'),('final','finalize')]
print("  %-10s  %-14s  %-14s  %-14s" % ('adj', '+ize(verb)', '+tion/noun', '+ment/noun'))
for adj, verb in test_adj_chain2:
    es, sid = get_emb(adj)
    if es is None or ax_ize is None: continue
    step1 = W_E[sid] + bs_ize * ax_ize
    r1 = nn_retrieve(step1, source_ids(adj), RELAXED_MASK, 1)
    got_verb = r1[0][0]; v1_id = r1[0][2]
    step2 = W_E[v1_id] + bs_tion * ax_tion
    r2 = nn_retrieve(step2, source_ids(got_verb), CLEAN_MASK, 1)
    step3 = W_E[v1_id] + bs_ment * ax_ment
    r3 = nn_retrieve(step3, source_ids(got_verb), CLEAN_MASK, 1)
    ok = '✓' if got_verb == verb else '~'
    print("  %s %-10s  %-14s  %-14s  %-14s" % (ok, adj, got_verb, r2[0][0], r3[0][0]))
print()

# =====================================================================
# PART E: +al_rel vs GROUP D arrival analysis
# =====================================================================
print("PART E: +al_rel vs GROUP D — do they share an 'adj cluster arrival' direction?")
print("-"*80)

ax_alrel, valid_alrel, pc_alrel = compute_axis(
    [('nation','national'),('region','regional'),('culture','cultural'),
     ('nature','natural'),('person','personal'),('origin','original'),
     ('emotion','emotional'),('tradition','traditional')])
ax_less, valid_less, _ = compute_axis(
    [('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),
     ('end','endless'),('home','homeless'),('harm','harmless'),('power','powerless')])
ax_ful, valid_ful, _ = compute_axis(
    [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),
     ('grace','graceful'),('help','helpful'),('faith','faithful'),('joy','joyful')])
ax_able, valid_able, _ = compute_axis(
    [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),
     ('use','usable'),('accept','acceptable'),('avoid','avoidable'),('change','changeable')])

if ax_alrel is not None:
    # Cosine of +al_rel with individual GROUP D members
    print("  +al_rel vs individual GROUP D members:")
    for name, ax in [('+less', ax_less), ('+ful', ax_ful), ('+able', ax_able)]:
        if ax is not None:
            c = float(np.dot(ax_alrel.astype(np.float32), ax.astype(np.float32)))
            print("    cos(+al_rel, %-6s) = %+.4f" % (name, c))
    print()

    # Key test: is the REVERSED +al_rel axis (noun<-adj = adj->noun) similar to GROUP B?
    ax_alrel_rev = -ax_alrel
    ax_ity_ref, _, _ = compute_axis(
        [('human','humanity'),('real','reality'),('national','nationality'),
         ('personal','personality'),('moral','morality'),('legal','legality')])
    ax_ness_ref, _, _ = compute_axis(
        [('happy','happiness'),('kind','kindness'),('sad','sadness'),
         ('bright','brightness'),('dark','darkness'),('soft','softness')])
    if ax_ity_ref is not None and ax_ness_ref is not None:
        print("  REVERSED +al_rel (adj->noun direction) vs GROUP B:")
        for name, ax in [('+ity', ax_ity_ref), ('+ness', ax_ness_ref)]:
            c_fwd = float(np.dot(ax_alrel.astype(np.float32), ax.astype(np.float32)))
            c_rev = float(np.dot(ax_alrel_rev.astype(np.float32), ax.astype(np.float32)))
            print("    cos(+al_rel, %-6s) = %+.4f    cos(-al_rel, %-6s) = %+.4f" %
                  (name, c_fwd, name, c_rev))
        print()

    # Does +al_rel navigate to the same adj positions as GROUP D?
    # Test: take a common verb, navigate +less, then see if +al_rel reverse
    #        takes us back to a noun
    print("  Functional test: noun -> adj (+al_rel) vs noun -> qual_adj (+less):")
    print("  Both should produce adj, but different types (rel_adj vs qual_adj)")
    test_nouns = [('nation','national'),('person','personal'),('region','regional')]
    test_verbs_less = [('home','homeless'),('hope','hopeless'),('pain','painless')]
    bs_alrel = best_scale(ax_alrel, valid_alrel, RELAXED_MASK)[0]
    bs_less  = best_scale(ax_less, valid_less, CLEAN_MASK)[0]
    print("  %-12s  +al_rel(noun->adj)    %-12s  +less(v->adj)" % ('noun', 'verb'))
    for (n, exp_n), (v, exp_v) in zip(test_nouns, test_verbs_less):
        en, sid_n = get_emb(n); ev, sid_v = get_emb(v)
        if en is None or ev is None: continue
        r_n = nn_retrieve(W_E[sid_n]+bs_alrel*ax_alrel, source_ids(n), RELAXED_MASK, 1)
        r_v = nn_retrieve(W_E[sid_v]+bs_less*ax_less, source_ids(v), CLEAN_MASK, 1)
        ok_n = '✓' if r_n[0][0]==exp_n else '~'
        ok_v = '✓' if r_v[0][0]==exp_v else '~'
        print("  %s %-10s -> %-12s    %s %-10s -> %-12s" %
              (ok_n, n, r_n[0][0], ok_v, v, r_v[0][0]))
