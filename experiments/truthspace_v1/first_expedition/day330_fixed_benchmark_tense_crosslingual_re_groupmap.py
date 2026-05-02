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

def classify_v5(pc, loo, irred):
    if pc > 0.35:    return 'morph_uniform/relational_geom'
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
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse-partial'
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

print()
print("DAY 330: FIXED BENCHMARK, TENSE SUB-CLUSTER, CROSS-LINGUAL CHAIN, +re-, GROUP MAP")
print("="*80)
print()

# =====================================================================
# PART A: FIXED 30-AXIS BENCHMARK (5 train + 3 proper holdout)
# =====================================================================
print("PART A: Fixed benchmark — 5 train + 3 SEPARATE holdout per axis")
print("-"*80)

# Each entry: (name, train_pairs[5], holdout_pairs[3], true_label)
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

print("  %-12s  pc      LOO%%  irred%%  n  -> pred                   true       ok?" %
      "axis")
print("  " + "-"*76)
v5_correct = 0
for name, train_pairs, holdout_pairs, true_type in FIXED_BENCH:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2:
        print("  %-12s  n/a" % name); continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr_f, n_ho, _ = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    pred = classify_v5(pc, loo_v, irr_f)
    def match(p, t):
        return (t.split('_')[0] in p or t in p or
                ('morph' in p and 'morph' in t) or ('phonol' in p and 'phonol' in t) or
                ('relational' in p and 'relational' in t) or ('factual' in p and 'factual' in t) or
                ('translation' in p and 'translation' in t) or ('polar' in p and 'polar' in t) or
                ('semantic' in p and 'semantic' in t))
    ok = match(pred, true_type)
    if ok: v5_correct += 1
    tick = '✓' if ok else '✗'
    print("  %s %-12s  pc=%.3f  %.0f%%  %.0f%%  %d  %-22s %-12s" %
          (tick, name, pc, 100*loo_v, 100*irr_f, n_ho, pred[:22], true_type))
print()
print("  Fixed benchmark accuracy: %d/%d = %.0f%%" %
      (v5_correct, len(FIXED_BENCH), 100*v5_correct/len(FIXED_BENCH)))
print()

# =====================================================================
# PART B: TENSE SUB-CLUSTER — PRESENT vs PAST WITHIN GROUP E
# =====================================================================
print("PART B: Tense sub-cluster verification (10 random subsamples)")
print("-"*80)

ALL_ING  = [('go','going'),('take','taking'),('run','running'),('see','seeing'),
             ('give','giving'),('make','making'),('write','writing'),('read','reading'),
             ('work','working'),('play','playing'),('eat','eating'),('sleep','sleeping')]
ALL_3PS  = [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
             ('read','reads'),('write','writes'),('play','plays'),('work','works'),
             ('talk','talks'),('drive','drives'),('sleep','sleeps'),('stand','stands')]
ALL_ED   = [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
             ('play','played'),('clean','cleaned'),('open','opened'),('start','started'),
             ('work','worked'),('wash','washed'),('turn','turned'),('move','moved')]
ALL_ABL  = [('go','went'),('take','took'),('give','gave'),('see','saw'),
             ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode'),
             ('break','broke'),('fall','fell'),('stand','stood'),('hold','held')]

print("  Computing tense sub-cluster cosines (5 random 8-pair subsets per axis):")
np.random.seed(99)
ing_3ps = []; ing_ed = []; ing_ab = []; ed_ab = []; ps_ed = []; ps_ab = []
for _ in range(5):
    def rsub(pool): return [pool[i] for i in np.random.choice(len(pool), 8, replace=False)]
    ax1, _, _ = compute_axis(rsub(ALL_ING))
    ax2, _, _ = compute_axis(rsub(ALL_3PS))
    ax3, _, _ = compute_axis(rsub(ALL_ED))
    ax4, _, _ = compute_axis(rsub(ALL_ABL))
    if all(a is not None for a in [ax1,ax2,ax3,ax4]):
        ing_3ps.append(np.dot(ax1.astype(np.float32), ax2.astype(np.float32)))
        ing_ed.append(np.dot(ax1.astype(np.float32), ax3.astype(np.float32)))
        ing_ab.append(np.dot(ax1.astype(np.float32), ax4.astype(np.float32)))
        ed_ab.append(np.dot(ax3.astype(np.float32), ax4.astype(np.float32)))
        ps_ed.append(np.dot(ax2.astype(np.float32), ax3.astype(np.float32)))
        ps_ab.append(np.dot(ax2.astype(np.float32), ax4.astype(np.float32)))

def stats(v): return "%.4f ± %.4f" % (np.mean(v), np.std(v))
print("  PRESENT-PRESENT:")
print("  cos(+ing, +3ps)   = %s" % stats(ing_3ps))
print()
print("  PAST-PAST:")
print("  cos(+ed_reg, ablaut) = %s" % stats(ed_ab))
print()
print("  CROSS-TENSE:")
print("  cos(+ing, +ed_reg) = %s" % stats(ing_ed))
print("  cos(+ing, ablaut)  = %s" % stats(ing_ab))
print("  cos(+3ps, +ed_reg) = %s" % stats(ps_ed))
print("  cos(+3ps, ablaut)  = %s" % stats(ps_ab))
print()
if ing_3ps and ed_ab and ing_ed:
    within = np.mean(ing_3ps + ed_ab)
    cross  = np.mean(ing_ed + ing_ab + ps_ed + ps_ab)
    print("  Within-tense mean: %.4f  Cross-tense mean: %.4f" % (within, cross))
    print("  Tense sub-clustering confirmed: within > cross by %.4f" % (within - cross))
print()

# =====================================================================
# PART C: CROSS-LINGUAL CHAIN — EN→ZH + +s_plural
# =====================================================================
print("PART C: Cross-lingual chain — EN→ZH then +s_plural")
print("-"*80)

EN_ZH_PAIRS = [('sun','日'),('moon','月'),('water','水'),('fire','火'),
                ('mountain','山'),('hand','手'),('eye','眼'),('fish','鱼'),
                ('heart','心'),('tree','木'),('sea','海'),('sky','天')]
PLURAL_PAIRS = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('bird','birds'),('door','doors')]

ax_zh, valid_zh, pc_zh = compute_axis(EN_ZH_PAIRS)
ax_pl, valid_pl, pc_pl = compute_axis(PLURAL_PAIRS)

if ax_zh is not None and ax_pl is not None:
    bs_zh, _ = best_scale(ax_zh, valid_zh, RELAXED_MASK)
    bs_pl, _ = best_scale(ax_pl, valid_pl, CLEAN_MASK)
    print("  Scales: EN->ZH=%.2f  +plural=%.2f" % (bs_zh, bs_pl))
    print()
    print("  Chain: EN word -> ZH word -> ZH plural (expect Chinese plural or same word)")
    for en_word, zh_word in [('sun','日'),('moon','月'),('water','水'),
                               ('hand','手'),('eye','眼'),('man','男')]:
        es, sid = get_emb(en_word)
        if es is None: continue
        step1 = W_E[sid] + bs_zh * ax_zh
        r1 = nn_retrieve(step1, source_ids(en_word), RELAXED_MASK, 3)
        mag = np.linalg.norm(W_E[sid])
        step2 = normed(step1)*mag + bs_pl * ax_pl
        r2 = nn_retrieve(step2, source_ids(en_word), RELAXED_MASK, 3)
        t1 = '✓' if r1[0][0]==zh_word else '~'
        is_cjk = any('\u4e00' <= c <= '\u9fff' for c in r2[0][0])
        print("  %s %-6s->%-6s  ->  %-6s  (CJK=%s)" %
              (t1, en_word, r1[0][0], r2[0][0], is_cjk))
    print()
    print("  Direct plural chain (control): EN -> plural EN")
    for en_word, pl_word in [('sun','suns'),('eye','eyes'),('hand','hands'),('day','days')]:
        es, sid = get_emb(en_word)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+bs_pl*ax_pl, source_ids(en_word), CLEAN_MASK, 2)
        print("  %-6s -> %-8s  (expected: %s)" % (en_word, r[0][0], pl_word))
print()

# =====================================================================
# PART D: +re- PREFIX AXIS
# =====================================================================
print("PART D: +re- prefix axis (verb → reversed verb)")
print("-"*80)

RE_PAIRS = [('do','redo'),('write','rewrite'),('build','rebuild'),('think','rethink'),
             ('place','replace'),('make','remake'),('view','review'),('read','reread'),
             ('use','reuse'),('check','recheck'),('play','replay'),('test','retest'),
             ('start','restart'),('open','reopen'),('form','reform'),('call','recall')]

ax_re, valid_re, pc_re = compute_axis(RE_PAIRS)
if ax_re is not None:
    loo_re = axis_loo(ax_re, valid_re, CLEAN_MASK)
    irr_re, _, _ = irred_on_holdout(ax_re,
        [('turn','return'),('load','reload'),('set','reset'),('run','rerun'),('name','rename')],
        CLEAN_MASK)
    print("  +re- axis: pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_re, 100*loo_re, 100*irr_re, len(valid_re)))
    print("  Predicted type: %s" % classify_v5(pc_re, loo_re, irr_re))
    print()

    # Inter-group cosines for +re-
    print("  cos(+re-, GROUP axes):")
    for ref_pairs, ref_name in [
        ([('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),('read','reads')], '+3ps'),
        ([('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),('play','played')], '+ed_reg'),
        ([('go','going'),('take','taking'),('run','running'),('see','seeing'),('give','giving')], '+ing'),
        ([('hope','hopeless'),('fear','fearless'),('care','careless'),('pain','painless'),('end','endless')], '+less'),
        ([('nation','national'),('region','regional'),('culture','cultural'),('nature','natural'),('person','personal')], '+al_rel'),
        ([('happy','unhappy'),('clear','unclear'),('fair','unfair'),('likely','unlikely'),('known','unknown')], 'un-'),
    ]:
        ax_ref, _, _ = compute_axis(ref_pairs)
        if ax_ref is not None:
            c = float(np.dot(ax_re.astype(np.float32), ax_ref.astype(np.float32)))
            print("  cos(+re-, %-8s) = %+.4f" % (ref_name, c))

    # Most importantly: cos(+re-, un-) — both are reversal/negation prefixes
    print()
    # Navigability test
    bs_re, _ = best_scale(ax_re, valid_re, CLEAN_MASK)
    print("  Scale: %.2f" % bs_re)
    print("  Navigation test:")
    for src, tgt in [('do','redo'),('write','rewrite'),('build','rebuild'),
                      ('think','rethink'),('turn','return'),('name','rename')]:
        es, sid = get_emb(src)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+bs_re*ax_re, source_ids(src), CLEAN_MASK, 3)
        ok = '✓' if r[0][0]==tgt else '~'
        print("  %s %-8s -> %-10s  (expected: %s)" % (ok, src, r[0][0], tgt))
print()

# =====================================================================
# PART E: COMPLETE GROUP MAP TABLE
# =====================================================================
print("PART E: Complete morphological family group map")
print("-"*80)

GROUP_DEFS = {
    'A: verb->event_noun': [
        ('+ance', [('perform','performance'),('exist','existence'),('enter','entrance'),
                    ('resist','resistance'),('accept','acceptance'),('appear','appearance')]),
        ('+al_nom',[('arrive','arrival'),('propose','proposal'),('approve','approval'),
                    ('refuse','refusal'),('remove','removal'),('survive','survival')]),
        ('+tion', [('act','action'),('direct','direction'),('educate','education'),
                    ('create','creation'),('produce','production'),('relate','relation')]),
        ('+ment', [('achieve','achievement'),('develop','development'),('manage','management'),
                    ('govern','government'),('engage','engagement'),('require','requirement')]),
    ],
    'B: adj->quality_noun': [
        ('+ity(lat)',[('human','humanity'),('real','reality'),('national','nationality'),
                      ('personal','personality'),('moral','morality'),('legal','legality')]),
        ('+ness(ger)',[('happy','happiness'),('sad','sadness'),('kind','kindness'),
                       ('dark','darkness'),('soft','softness'),('weak','weakness')]),
    ],
    'D: verb->adj_modifier': [
        ('+less', [('hope','hopeless'),('fear','fearless'),('care','careless'),
                    ('pain','painless'),('end','endless'),('home','homeless')]),
        ('+ful',  [('hope','hopeful'),('care','careful'),('fear','fearful'),
                    ('use','useful'),('grace','graceful'),('help','helpful')]),
        ('+able', [('read','readable'),('wash','washable'),('break','breakable'),
                    ('love','lovable'),('use','usable'),('accept','acceptable')]),
    ],
    'E: verb->inflected': [
        ('+3ps',  [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                    ('read','reads'),('write','writes'),('play','plays'),('work','works')]),
        ('+ed_reg',[('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                    ('play','played'),('clean','cleaned'),('open','opened'),('start','started')]),
        ('+ing',  [('go','going'),('take','taking'),('run','running'),('see','seeing'),
                    ('give','giving'),('make','making'),('write','writing'),('read','reading')]),
        ('ablaut',[('go','went'),('take','took'),('give','gave'),('see','saw'),
                   ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')]),
    ],
}

print("  Building group axes...", flush=True)
group_axes = {}
for gname, members in GROUP_DEFS.items():
    group_axes[gname] = {}
    for mname, pairs in members:
        ax, _, pc = compute_axis(pairs)
        if ax is not None:
            group_axes[gname][mname] = (ax, pc)

print("  Intra-group cosine summary:")
for gname, members in group_axes.items():
    names = list(members.keys())
    cosines = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = float(np.dot(members[names[i]][0].astype(np.float32),
                             members[names[j]][0].astype(np.float32)))
            cosines.append(c)
    if cosines:
        print("  %-25s n=%d  mean=%.4f  range=[%.4f, %.4f]" %
              (gname, len(cosines), np.mean(cosines), min(cosines), max(cosines)))
print()
print("  Inter-group cosine matrix (group means):")
gnames = list(group_axes.keys())
header = "  %-25s" % "" + "".join("  %-12s" % g[:12] for g in gnames)
print(header)
for g1 in gnames:
    row = "  %-25s" % g1[:25]
    for g2 in gnames:
        if g1 == g2:
            row += "  %-12s" % "---"
            continue
        cosines = []
        for n1, (a1,_) in group_axes[g1].items():
            for n2, (a2,_) in group_axes[g2].items():
                cosines.append(float(np.dot(a1.astype(np.float32), a2.astype(np.float32))))
        row += "  %+.4f      " % np.mean(cosines) if cosines else "  n/a         "
    print(row)
print()
print("  Reverse pair: cos(+al_rel, +ity) = %.4f" %
      float(np.dot(group_axes['B: adj->quality_noun'].get('+ity(lat)',(np.zeros(1),0))[0].astype(np.float32),
                   compute_axis([('nation','national'),('person','personal'),('culture','cultural')])[0].astype(np.float32)))
      if group_axes['B: adj->quality_noun'].get('+ity(lat)') else 0)
