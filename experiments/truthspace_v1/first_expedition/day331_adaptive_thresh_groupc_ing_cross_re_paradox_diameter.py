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
    """Compute spread of chords: std of chord cosines relative to mean axis."""
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

def irred_on_holdout(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    irred=0; n_ho=0; details=[]
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1; found_at = None
        for s in np.linspace(lo, hi, n):
            r = nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask, 1)
            if r[0][0] == t_w: found_at=s; break
        if found_at is None: irred += 1
        details.append((s_w, t_w, found_at))
    return irred/n_ho if n_ho else 0.0, n_ho, details

def classify_adaptive(pc, loo, irred, n_train):
    """Size-adaptive predictor: scale thresholds by (8/n)^0.3."""
    if n_train < 2: return 'insufficient'
    scale = (8.0 / max(n_train, 2)) ** 0.3
    t_top = 0.35 * scale      # normally 0.35
    t_mid = 0.20 * scale      # normally 0.20
    t_low = 0.10 * scale      # normally 0.10
    t_vlow= 0.05 * scale      # normally 0.05
    if pc > t_top:    return 'morph_uniform/relational_geom'
    elif pc > t_mid:
        if loo > 0.50:     return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > t_low:
        if loo > 0.50:
            if irred >= 0.40: return 'semantic_diverse'
            return 'phonol_scatter'
        elif irred >= 0.95:  return 'factual_local/translation'
        elif irred >= 0.60:  return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse-partial'
        elif irred < 0.20:   return 'phonol_scatter-allomorph'
        else:                return 'borderline'
    elif pc > t_vlow:
        if irred >= 0.85 and loo < 0.15:  return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15: return 'borderline'
        else: return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

print()
print("DAY 331: ADAPTIVE THRESHOLDS, GROUP C, +ing CROSS-GROUP, +re- PARADOX, DIAMETER")
print("="*80)
print()

# =====================================================================
# PART A: SIZE-ADAPTIVE THRESHOLD BENCHMARK
# =====================================================================
print("PART A: Size-adaptive threshold benchmark (5 train + 3 holdout)")
print("-"*80)

FIXED_BENCH = [
    ('er_comp',
     [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter')],
     [('warm','warmer'),('long','longer'),('cold','colder')], 'morph_uniform'),
    ('er_sup',
     [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('clean','cleanest')],
     [('bright','brightest'),('dark','darkest'),('soft','softest')], 'morph_uniform'),
    ('relational',
     [('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany')],
     [('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')], 'relational_geom'),
    ('al_rel',
     [('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal')],
     [('origin','original'),('emotion','emotional'),('tradition','traditional')], 'relational_geom'),
    ('plural',
     [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),('tree','trees')],
     [('book','books'),('bird','birds'),('door','doors')], 'morph_moderate'),
    ('3ps',
     [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads')],
     [('write','writes'),('play','plays'),('work','works')], 'morph_moderate'),
    ('ed_reg',
     [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played')],
     [('clean','cleaned'),('open','opened'),('start','started')], 'morph_moderate'),
    ('ing',
     [('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving')],
     [('make','making'),('write','writing'),('read','reading')], 'morph_moderate'),
    ('cc',
     [('dog','Dog'),('house','House'),('cat','Cat'),('book','Book'),('car','Car')],
     [('tree','Tree'),('river','River'),('mountain','Mountain')], 'morph_moderate'),
    ('ness',
     [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness')],
     [('weak','weakness'),('good','goodness'),('hard','hardness')], 'phonol_scatter'),
    ('ablaut',
     [('go','went'),('take','took'),('give','gave'),('see','saw'),('know','knew')],
     [('drive','drove'),('write','wrote'),('ride','rode')], 'phonol_scatter'),
    ('ablaut_t',
     [('send','sent'),('build','built'),('feel','felt'),('keep','kept'),('leave','left')],
     [('deal','dealt'),('sleep','slept'),('burn','burned')], 'phonol_scatter'),
    ('ity',
     [('human','humanity'),('real','reality'),('final','finality'),('moral','morality'),('normal','normality')],
     [('national','nationality'),('personal','personality'),('legal','legality')], 'phonol_scatter'),
    ('un_neg',
     [('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),('known','unknown')],
     [('safe','unsafe'),('usual','unusual'),('equal','unequal')], 'phonol_scatter'),
    ('ance',
     [('perform','performance'),('exist','existence'),('enter','entrance'),('resist','resistance'),('accept','acceptance')],
     [('appear','appearance'),('depend','dependence'),('insist','insistence')], 'phonol_scatter'),
    ('ment',
     [('achieve','achievement'),('develop','development'),('manage','management'),('govern','government'),('engage','engagement')],
     [('require','requirement'),('move','movement'),('improve','improvement')], 'phonol_scatter'),
    ('tion',
     [('act','action'),('direct','direction'),('educate','education'),('create','creation'),('produce','production')],
     [('relate','relation'),('combine','combination'),('apply','application')], 'phonol_scatter'),
    ('al_nom',
     [('arrive','arrival'),('propose','proposal'),('approve','approval'),('refuse','refusal'),('remove','removal')],
     [('survive','survival'),('deny','denial'),('dispose','disposal')], 'phonol_scatter'),
    ('less',
     [('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),('end','endless')],
     [('home','homeless'),('harm','harmless'),('power','powerless')], 'phonol_scatter'),
    ('ful',
     [('hope','hopeful'),('care','careful'),('fear','fearful'),('use','useful'),('grace','graceful')],
     [('help','helpful'),('faith','faithful'),('joy','joyful')], 'phonol_scatter'),
    ('able',
     [('read','readable'),('wash','washable'),('break','breakable'),('love','lovable'),('use','usable')],
     [('accept','acceptable'),('avoid','avoidable'),('change','changeable')], 'phonol_scatter'),
    ('er_noun',
     [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner')],
     [('manage','manager'),('build','builder'),('lead','leader')], 'semantic_diverse'),
    ('adj_ant',
     [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('bright','dark')],
     [('hard','soft'),('high','low'),('rich','poor')], 'polar_local'),
    ('antonym2',
     [('love','hate'),('war','peace'),('life','death'),('day','night'),('begin','end')],
     [('give','take'),('push','pull'),('open','close')], 'polar_local'),
    ('en_es',
     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día')],
     [('night','noche'),('hand','mano'),('year','año')], 'translation'),
    ('en_de',
     [('house','Haus'),('water','Wasser'),('sun','Sonne'),('book','Buch'),('day','Tag')],
     [('night','Nacht'),('cat','Katze'),('dog','Hund')], 'translation'),
    ('en_fr',
     [('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),('day','jour')],
     [('night','nuit'),('cat','chat'),('dog','chien')], 'translation'),
    ('en_zh',
     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山')],
     [('hand','手'),('eye','眼'),('fish','鱼')], 'factual_local'),
    ('en_ja',
     [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山')],
     [('hand','手'),('eye','目'),('fish','魚')], 'factual_local'),
    ('num_word',
     [('1','one'),('2','two'),('3','three'),('4','four'),('5','five')],
     [('6','six'),('7','seven'),('8','eight')], 'semantic_diverse'),
]

def match(pred, true):
    return (true.split('_')[0] in pred or true in pred or
            ('morph' in pred and 'morph' in true) or ('phonol' in pred and 'phonol' in true) or
            ('relational' in pred and 'relational' in true) or ('factual' in pred and 'factual' in true) or
            ('translation' in pred and 'translation' in true) or ('polar' in pred and 'polar' in true) or
            ('semantic' in pred and 'semantic' in true))

adap_correct = 0
print("  %-12s  pc       scaled_thresh  pred                   true        ok?" % "axis")
print("  " + "-"*76)
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2: continue
    n = len(valid)
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr_f, _, _ = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    pred = classify_adaptive(pc, loo_v, irr_f, n)
    scale_f = (8.0/max(n,2))**0.3
    ok = match(pred, true_type)
    if ok: adap_correct += 1
    tick = '✓' if ok else '✗'
    print("  %s %-12s  pc=%.3f  t=[%.3f,%.3f,%.3f]  %-22s %-12s" %
          (tick, name, pc, 0.10*scale_f, 0.20*scale_f, 0.35*scale_f,
           pred[:22], true_type))
print()
print("  Adaptive accuracy: %d/30 = %.0f%%" % (adap_correct, 100*adap_correct/30))
print()

# =====================================================================
# PART B: GROUP C SEARCH (denominal verbs, deadjectival verbs)
# =====================================================================
print("PART B: GROUP C search — noun→verb and adj→verb")
print("-"*80)

# Denominal verbs: noun used as verb (verb meaning derived from noun)
DENOM_PAIRS = [
    ('water','water'),   # noun=water(liquid) vs verb=water(plants) - same token!
    ('chair','chair'),   # same token
    ('fish','fish'),     # same token
    # These fail because source=target for same-form words
    # Use -en/-ify/-ize instead:
    ('strength','strengthen'),('length','lengthen'),('depth','deepen'),
    ('bright','brighten'),('dark','darken'),('hard','harden'),
    ('wide','widen'),('short','shorten'),('tight','tighten'),
    ('soft','soften'),('fresh','freshen'),('weak','weaken'),
]
# Filter out self-pairs
DENOM_PAIRS = [(s,t) for s,t in DENOM_PAIRS if s != t]

# Denominalisation via -ize: noun → verb
IZE_PAIRS = [
    ('memory','memorize'),('symbol','symbolize'),('organ','organize'),
    ('crystal','crystallize'),('moral','moralize'),('legal','legalize'),
    ('minimal','minimize'),('maximal','maximize'),('real','realize'),
    ('national','nationalize'),('local','localize'),('modern','modernize'),
]

# Deadjectival verbs via -en
EN_VERB_PAIRS = [
    ('bright','brighten'),('dark','darken'),('hard','harden'),
    ('wide','widen'),('short','shorten'),('tight','tighten'),
    ('soft','soften'),('fresh','freshen'),('weak','weaken'),
    ('sharp','sharpen'),('flat','flatten'),('sweet','sweeten'),
    ('deep','deepen'),('light','lighten'),('thick','thicken'),
    ('white','whiten'),('black','blacken'),('red','redden'),
]

ax_en, valid_en, pc_en = compute_axis(EN_VERB_PAIRS)
ax_ize, valid_ize, pc_ize = compute_axis(IZE_PAIRS)

if ax_en is not None:
    loo_en = axis_loo(ax_en, valid_en, CLEAN_MASK)
    irr_en, _, _ = irred_on_holdout(ax_en,
        [('loose','loosen'),('long','lengthen'),('broad','broaden'),('clear','clarify')],
        CLEAN_MASK)
    print("  adj→verb (+en): pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_en, 100*loo_en, 100*irr_en, len(valid_en)))

if ax_ize is not None:
    loo_ize = axis_loo(ax_ize, valid_ize, CLEAN_MASK)
    irr_ize, _, _ = irred_on_holdout(ax_ize,
        [('general','generalize'),('civil','civilize'),('final','finalize')],
        CLEAN_MASK)
    print("  adj→verb (+ize): pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_ize, 100*loo_ize, 100*irr_ize, len(valid_ize)))
print()

# Inter-group cosines for potential GROUP C
GROUP_REF_PAIRS = {
    'GROUP_A(+ance)': [('perform','performance'),('exist','existence'),('enter','entrance'),
                        ('resist','resistance'),('accept','acceptance'),('appear','appearance')],
    'GROUP_D(+less)': [('hope','hopeless'),('fear','fearless'),('care','careless'),
                        ('pain','painless'),('end','endless'),('home','homeless')],
    'GROUP_E(+3ps)':  [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                        ('read','reads'),('write','writes'),('play','plays'),('work','works')],
    'GROUP_B(+ness)': [('happy','happiness'),('kind','kindness'),('sad','sadness'),
                        ('bright','brightness'),('dark','darkness'),('soft','softness')],
}

print("  GROUP C candidates vs reference groups:")
for cand_name, cand_ax in [('adj->verb +en', ax_en), ('adj->verb +ize', ax_ize)]:
    if cand_ax is None: continue
    print("  %s:" % cand_name)
    for ref_name, ref_pairs in GROUP_REF_PAIRS.items():
        ax_ref, _, _ = compute_axis(ref_pairs)
        if ax_ref is not None:
            c = float(np.dot(cand_ax.astype(np.float32), ax_ref.astype(np.float32)))
            print("    cos(%-20s, %-15s) = %+.4f" % (cand_name, ref_name, c))
    print()

# Are +en and +ize related to each other?
if ax_en is not None and ax_ize is not None:
    c = float(np.dot(ax_en.astype(np.float32), ax_ize.astype(np.float32)))
    print("  cos(+en, +ize) = %+.4f  (GROUP C internal cosine)" % c)
print()

# =====================================================================
# PART C: +ing COMPLETE CROSS-GROUP COSINES
# =====================================================================
print("PART C: +ing complete cross-group cosines")
print("-"*80)

ING_PAIRS = [('go','going'),('take','taking'),('run','running'),('see','seeing'),
              ('give','giving'),('make','making'),('write','writing'),('read','reading'),
              ('work','working'),('play','playing'),('eat','eating'),('sleep','sleeping')]

ax_ing, _, _ = compute_axis(ING_PAIRS)

ALL_REF = [
    ('GROUP_A +ance', [('perform','performance'),('exist','existence'),('enter','entrance'),
                       ('resist','resistance'),('accept','acceptance'),('appear','appearance')]),
    ('GROUP_A +ment', [('achieve','achievement'),('develop','development'),('manage','management'),
                       ('govern','government'),('engage','engagement'),('require','requirement')]),
    ('GROUP_B +ness', [('happy','happiness'),('kind','kindness'),('dark','darkness'),
                       ('soft','softness'),('weak','weakness'),('bright','brightness')]),
    ('GROUP_B +ity',  [('human','humanity'),('real','reality'),('national','nationality'),
                       ('personal','personality'),('moral','morality'),('legal','legality')]),
    ('GROUP_D +less', [('hope','hopeless'),('fear','fearless'),('care','careless'),
                       ('pain','painless'),('end','endless'),('home','homeless')]),
    ('GROUP_D +ful',  [('hope','hopeful'),('care','careful'),('fear','fearful'),
                       ('use','useful'),('grace','graceful'),('help','helpful')]),
    ('GROUP_D +able', [('read','readable'),('wash','washable'),('break','breakable'),
                       ('love','lovable'),('use','usable'),('accept','acceptable')]),
    ('GROUP_E +3ps',  [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                       ('read','reads'),('write','writes'),('play','plays'),('work','works')]),
    ('GROUP_E +ed',   [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                       ('play','played'),('clean','cleaned'),('open','opened'),('start','started')]),
    ('GROUP_E ablaut',[('go','went'),('take','took'),('give','gave'),('see','saw'),
                       ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')]),
    ('STANDALONE +ly',[('quick','quickly'),('slow','slowly'),('happy','happily'),
                       ('careful','carefully'),('loud','loudly'),('soft','softly')]),
    ('STANDALONE +pl',[('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                       ('tree','trees'),('book','books'),('bird','birds'),('door','doors')]),
    ('REVERSE +al_rel',[('nation','national'),('region','regional'),('culture','cultural'),
                        ('nature','natural'),('person','personal'),('origin','original')]),
]

if ax_ing is not None:
    print("  cos(+ing, ALL reference axes):")
    for ref_name, ref_pairs in ALL_REF:
        ax_ref, _, _ = compute_axis(ref_pairs)
        if ax_ref is not None:
            c = float(np.dot(ax_ing.astype(np.float32), ax_ref.astype(np.float32)))
            group = ref_name.split()[0]
            print("  cos(+ing, %-20s) = %+.4f" % (ref_name, c))
print()

# =====================================================================
# PART D: THE +re- PARADOX — DIRECTION vs MAGNITUDE
# =====================================================================
print("PART D: +re- paradox — overfitting direction vs magnitude")
print("-"*80)

RE_PAIRS = [('do','redo'),('write','rewrite'),('build','rebuild'),('think','rethink'),
             ('place','replace'),('make','remake'),('view','review'),('read','reread'),
             ('use','reuse'),('check','recheck'),('play','replay'),('test','retest'),
             ('start','restart'),('open','reopen'),('form','reform'),('call','recall')]

ax_re, valid_re, pc_re = compute_axis(RE_PAIRS)
if ax_re is not None:
    chords_re = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid_re]
    mags = [np.linalg.norm(c) for c in chords_re]
    dots_to_mean = [float(np.dot(normed(c).astype(np.float32), ax_re.astype(np.float32)))
                    for c in chords_re]
    print("  +re- chord analysis:")
    print("  pc=%.4f  n=%d" % (pc_re, len(valid_re)))
    print("  Chord magnitudes: mean=%.4f  std=%.4f  range=[%.4f, %.4f]" %
          (np.mean(mags), np.std(mags), min(mags), max(mags)))
    print("  Chord-to-axis alignment: mean=%.4f  std=%.4f  range=[%.4f, %.4f]" %
          (np.mean(dots_to_mean), np.std(dots_to_mean), min(dots_to_mean), max(dots_to_mean)))
    print()

    # Test: does the axis direction work even when scale is wrong?
    # Compare: navigate using (a) MEAN axis direction + per-pair scale
    #          vs              (b) MEAN axis direction + mean scale
    bs_re, _ = best_scale(ax_re, valid_re, CLEAN_MASK)
    print("  Global scale: %.2f" % bs_re)
    print()

    # LOO test with fixed scale vs variable scale
    print("  LOO fixed-scale:   ", end='')
    hits_fixed = 0
    for i in range(len(valid_re)):
        tv = [valid_re[j] for j in range(len(valid_re)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        test_s, test_t, test_sid, _ = valid_re[i]
        r = nn_retrieve(W_E[test_sid]+bs_re*al, source_ids(test_s), CLEAN_MASK, 1)
        if r[0][0] == test_t: hits_fixed += 1
    print("%.0f%%" % (100*hits_fixed/len(valid_re)))

    # LOO with per-pair optimal scale
    print("  LOO per-pair scale:", end='')
    hits_var = 0
    for i in range(len(valid_re)):
        tv = [valid_re[j] for j in range(len(valid_re)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        test_s, test_t, test_sid, _ = valid_re[i]
        # Find best scale just for this pair
        best_s_local = bs_re
        for s in np.linspace(0.1, 4.0, 50):
            r = nn_retrieve(W_E[test_sid]+s*al, source_ids(test_s), CLEAN_MASK, 1)
            if r[0][0] == test_t:
                best_s_local = s; break
        r = nn_retrieve(W_E[test_sid]+best_s_local*al, source_ids(test_s), CLEAN_MASK, 1)
        if r[0][0] == test_t: hits_var += 1
    print("%.0f%%" % (100*hits_var/len(valid_re)))
    print()
    print("  Diagnosis: if per-pair >> fixed, +re- axis is SCALE-SENSITIVE (direction ok)")
    print("  If both low, +re- axis has an unstable DIRECTION")
print()

# =====================================================================
# PART E: AXIS DIAMETER — SPREAD vs pc/LOO/irred
# =====================================================================
print("PART E: Axis diameter — chord spread correlates with axis type")
print("-"*80)

DIAMETER_TEST_AXES = [
    ('er_comp',  [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),
                   ('bright','brighter'),('warm','warmer'),('long','longer'),('cold','colder')],
     'morph_uniform'),
    ('3ps',      [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                   ('read','reads'),('write','writes'),('play','plays'),('work','works')],
     'morph_moderate'),
    ('ed_reg',   [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                   ('play','played'),('clean','cleaned'),('open','opened'),('start','started')],
     'morph_moderate'),
    ('ness',     [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),
                   ('soft','softness'),('weak','weakness'),('good','goodness'),('hard','hardness')],
     'phonol_scatter'),
    ('ablaut',   [('go','went'),('take','took'),('give','gave'),('see','saw'),
                   ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],
     'phonol_scatter'),
    ('ance',     [('perform','performance'),('exist','existence'),('enter','entrance'),
                   ('resist','resistance'),('accept','acceptance'),('appear','appearance'),
                   ('depend','dependence'),('insist','insistence')],
     'phonol_scatter'),
    ('er_noun',  [('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),
                   ('own','owner'),('manage','manager'),('build','builder'),('lead','leader')],
     'semantic_diverse'),
    ('adj_ant',  [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),
                   ('bright','dark'),('hard','soft'),('high','low'),('rich','poor')],
     'polar_local'),
    ('en_zh',    [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山'),
                   ('hand','手'),('eye','眼'),('fish','鱼')],
     'factual_local'),
    ('en_es',    [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),
                   ('day','día'),('night','noche'),('hand','mano'),('year','año')],
     'translation'),
    ('+re-',     [('do','redo'),('write','rewrite'),('build','rebuild'),('think','rethink'),
                   ('place','replace'),('make','remake'),('view','review'),('read','reread')],
     'standalone'),
    ('+ize',     IZE_PAIRS[:8] if len(IZE_PAIRS)>=8 else IZE_PAIRS,
     'GROUP_C?'),
]

print("  %-14s  pc      spread  mean_mag  type")
print("  " + "-"*62)
for name, pairs, true_type in DIAMETER_TEST_AXES:
    pc, spread, mean_mag = chord_spread(pairs)
    if pc == 0.0: continue
    print("  %-14s  pc=%.4f  s=%.4f  m=%.4f  %s" %
          (name, pc, spread, mean_mag, true_type))
print()
print("  Hypothesis: spread correlates inversely with pc (less tight = more spread)")
print("  Mean_mag: how far each word moves on average (large = global operation)")
