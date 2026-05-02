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
    if pc > 0.35:   return 'morph_uniform/relational_geom'
    elif pc > 0.20:
        if loo > 0.50:    return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred >= 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo > 0.50:      return 'phonol_scatter'
        elif irred >= 0.95: return 'factual_local/translation'
        elif irred >= 0.60: return 'semantic_diverse'
        elif loo == 0.0 and irred < 0.60: return 'semantic_diverse-partial'
        elif irred < 0.20:  return 'phonol_scatter-allomorph'
        else:               return 'borderline'
    elif pc > 0.05:
        if irred >= 0.85 and loo < 0.15:  return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15: return 'borderline'
        else: return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

print()
print("DAY 326: REVERSE CHAIN, 8-PAIR 5-SHOT, NEGATIVE COSINES, +ness/+ity TARGETS, SUPPLETIVE CHAIN")
print("="*80)
print()

# =====================================================================
# PART A: REVERSE OPERATION CHAIN +al_rel → +ity
# =====================================================================
print("PART A: Reverse operation chain test (+al_rel then +ity)")
print("-"*80)

AL_REL_PAIRS = [('nation','national'),('region','regional'),('culture','cultural'),
                 ('nature','natural'),('person','personal'),('origin','original'),
                 ('emotion','emotional'),('tradition','traditional')]
ITY_PAIRS    = [('human','humanity'),('real','reality'),('final','finality'),
                 ('mental','mentality'),('legal','legality'),('local','locality'),
                 ('moral','morality'),('normal','normality'),('national','nationality'),
                 ('personal','personality'),('emotional','emotionality'),
                 ('regional','regionality'),('original','originality'),
                 ('cultural','culturality')]

ax_alr, valid_alr, pc_alr = compute_axis(AL_REL_PAIRS)
ax_ity, valid_ity, pc_ity = compute_axis(ITY_PAIRS)

if ax_alr is not None and ax_ity is not None:
    c = float(np.dot(ax_alr.astype(np.float32), ax_ity.astype(np.float32)))
    print("  cos(+al_rel, +ity) = %+.4f" % c)
    best_s_alr, _ = best_scale(ax_alr, valid_alr, CLEAN_MASK)
    best_s_ity, _ = best_scale(ax_ity, valid_ity, CLEAN_MASK)
    print("  +al_rel scale=%.2f  +ity scale=%.2f" % (best_s_alr, best_s_ity))
    print()
    print("  Chain test: noun --[+al_rel]--> adj --[+ity]--> quality_noun")
    for noun, adj_form, noun2 in [('nation','national','nationality'),
                                    ('person','personal','personality'),
                                    ('origin','original','originality'),
                                    ('region','regional','regionality'),
                                    ('emotion','emotional','emotionality'),
                                    ('culture','cultural','culturality'),
                                    ('tradition','traditional','traditionality')]:
        es, sid = get_emb(noun)
        if es is None: continue
        # Step 1: apply +al_rel
        step1 = W_E[sid] + best_s_alr * ax_alr
        r1 = nn_retrieve(step1, source_ids(noun), CLEAN_MASK, 3)
        # Step 2: apply +ity on the displaced embedding
        step2 = step1 + best_s_ity * ax_ity
        r2 = nn_retrieve(step2, source_ids(noun), CLEAN_MASK, 3)
        # Direct +ity from original noun (control)
        direct = W_E[sid] + best_s_ity * ax_ity
        r_dir = nn_retrieve(direct, source_ids(noun), CLEAN_MASK, 3)
        t1 = '✓' if r1[0][0]==adj_form else '~'
        t2 = '✓' if r2[0][0]==noun2 else '~'
        print("  %s %-12s->%-14s  %s chain->%-18s  (direct: %s)" %
              (t1, noun, r1[0][0], t2, r2[0][0], r_dir[0][0]))
    print()

    # Direct +ity on the ADJ form (to compare with chain)
    print("  Direct +ity from adj form (is 'national' in +ity training dist?):")
    for adj_form, noun2 in [('national','nationality'),('personal','personality'),
                              ('original','originality'),('regional','regionality'),
                              ('cultural','culturality'),('emotional','emotionality')]:
        es, sid = get_emb(adj_form)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+best_s_ity*ax_ity, source_ids(adj_form), CLEAN_MASK, 3)
        tick = '✓' if r[0][0]==noun2 else '~'
        print("  %s %-14s -> %s" % (tick, adj_form, r[0][0]))
    print()

# =====================================================================
# PART B: 8-PAIR 5-SHOT RE-TEST (5 training + 3 holdout)
# =====================================================================
print("PART B: 8-pair probe (5 train + 3 holdout → measure irred)")
print("-"*80)

EIGHT_PAIR_TESTS = [
    ('morph_uniform',   [('big','bigger'),('fast','faster'),('tall','taller'),('clean','cleaner'),('bright','brighter')],
                        [('warm','warmer'),('long','longer'),('cold','colder')]),
    ('morph_moderate',  [('cat','cats'),('dog','dogs'),('house','houses'),('bird','birds'),('book','books')],
                        [('tree','trees'),('car','cars'),('ship','ships')]),
    ('phonol_scatter',  [('happy','happiness'),('sad','sadness'),('kind','kindness'),('dark','darkness'),('soft','softness')],
                        [('weak','weakness'),('good','goodness'),('hard','hardness')]),
    ('semantic_diverse',[('teach','teacher'),('farm','farmer'),('drive','driver'),('work','worker'),('own','owner')],
                        [('manage','manager'),('build','builder'),('lead','leader')]),
    ('factual_local',   [('sun','日'),('moon','月'),('water','水'),('fire','火'),('mountain','山')],
                        [('hand','手'),('eye','眼'),('fish','鱼')]),
    ('polar_local',     [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),('high','low')],
                        [('hard','soft'),('bright','dark'),('strong','weak')]),
    ('relational_geom', [('London','England'),('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany')],
                        [('Tokyo','Japan'),('Beijing','China'),('Moscow','Russia')]),
    ('translation',     [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),('day','día')],
                        [('night','noche'),('hand','mano'),('year','año')]),
]

print("  %-18s  pc      LOO%%  irred%%  -> pred                      ok?" % "true_type")
print("  " + "-"*76)
correct_8 = 0
for true_type, train_pairs, holdout_pairs in EIGHT_PAIR_TESTS:
    ax, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2:
        print("  %-18s  n/a" % true_type); continue
    loo_v = axis_loo(ax, valid, RELAXED_MASK)
    irr_f, _, _ = irred_on_holdout(ax, holdout_pairs, RELAXED_MASK)
    pred = classify_v5(pc, loo_v, irr_f)
    match = (true_type.split('_')[0] in pred or true_type in pred or
             ('morph' in pred and 'morph' in true_type) or
             ('phonol' in pred and 'phonol' in true_type) or
             ('relational' in pred and 'relational' in true_type) or
             ('factual' in pred and 'factual' in true_type) or
             ('translation' in pred and 'translation' in true_type) or
             ('polar' in pred and 'polar' in true_type) or
             ('semantic' in pred and 'semantic' in true_type))
    if match: correct_8 += 1
    tick = '✓' if match else '✗'
    print("  %s %-18s  pc=%.4f  LOO=%.0f%%  irred=%.0f%%  -> %-28s" %
          (tick, true_type, pc, 100*loo_v, 100*irr_f, pred))
print()
print("  8-pair accuracy: %d/%d = %.0f%%" % (correct_8, len(EIGHT_PAIR_TESTS), 100*correct_8/len(EIGHT_PAIR_TESTS)))
print()

# =====================================================================
# PART C: NEGATIVE COSINES — +al_rel FAMILY
# =====================================================================
print("PART C: Negative cosines — +al_rel vs all nominalizer axes")
print("-"*80)

NOMINALIZER_AXES = {
    '+ance':    [('perform','performance'),('exist','existence'),('enter','entrance'),
                  ('resist','resistance'),('accept','acceptance'),('insist','insistence'),
                  ('appear','appearance'),('depend','dependence')],
    '+ment':    [('achieve','achievement'),('develop','development'),('manage','management'),
                  ('govern','government'),('engage','engagement'),('require','requirement')],
    '+tion':    [('act','action'),('direct','direction'),('educate','education'),
                  ('create','creation'),('produce','production'),('relate','relation')],
    '+al_nom':  [('arrive','arrival'),('propose','proposal'),('approve','approval'),
                  ('refuse','refusal'),('remove','removal'),('survive','survival')],
    '+ity':     [('human','humanity'),('real','reality'),('final','finality'),
                  ('moral','morality'),('normal','normality'),('national','nationality')],
    '+ness':    [('happy','happiness'),('kind','kindness'),('sad','sadness'),
                  ('bright','brightness'),('dark','darkness'),('soft','softness'),
                  ('fair','fairness'),('clear','clearness'),('weak','weakness')],
    '+er_noun': [('teach','teacher'),('farm','farmer'),('drive','driver'),
                  ('work','worker'),('own','owner'),('lead','leader')],
    '+er_comp': [('fast','faster'),('slow','slower'),('bright','brighter'),
                  ('dark','darker'),('soft','softer'),('warm','warmer')],
    '+s_plural':[('cat','cats'),('dog','dogs'),('house','houses'),('bird','birds'),
                  ('book','books'),('tree','trees'),('car','cars')],
    'adj_ant':  [('good','bad'),('hot','cold'),('fast','slow'),('big','small'),
                  ('bright','dark'),('hard','soft'),('high','low')],
}

print("  Computing all axes...")
all_axes = {'al_rel': ax_alr}
for name, pairs in NOMINALIZER_AXES.items():
    ax, _, _ = compute_axis(pairs)
    if ax is not None: all_axes[name] = ax

print("  cos(+al_rel, X):")
for name, ax in sorted(all_axes.items()):
    if name == 'al_rel': continue
    c = float(np.dot(ax_alr.astype(np.float32), ax.astype(np.float32)))
    sign = '←' if c < -0.15 else ('→' if c > 0.15 else '≈0')
    print("  cos(+al_rel, %-12s) = %+.4f  %s" % (name, c, sign))
print()
print("  Full cross-axis cosine table (selected):")
key_axes = ['+ance', '+ment', '+tion', '+al_nom', '+ity', '+ness', '+er_noun', '+er_comp', 'adj_ant']
print("  " + "%-12s" % "", end="")
for n in key_axes: print(" %-8s" % n[:8], end="")
print()
for n1 in key_axes:
    if n1 not in all_axes: continue
    print("  %-12s" % n1, end="")
    for n2 in key_axes:
        if n2 not in all_axes:
            print(" %-8s" % "n/a", end="")
        elif n1 == n2:
            print(" %-8s" % "1.000", end="")
        else:
            c = float(np.dot(all_axes[n1].astype(np.float32), all_axes[n2].astype(np.float32)))
            print(" %+.3f  " % c, end="")
    print()
print()

# =====================================================================
# PART D: +ness vs +ity TARGET TOKEN ANALYSIS
# =====================================================================
print("PART D: +ness vs +ity target vocabulary analysis")
print("-"*80)

if ax_ity is not None and all_axes.get('+ness') is not None:
    ax_ne = all_axes['+ness']
    # Compute 'centroid' of training targets for each
    ity_targets = ['humanity','reality','finality','morality','normality',
                    'nationality','personality','originality','legality','locality']
    ness_targets = ['happiness','kindness','sadness','brightness','darkness',
                     'softness','fairness','clearness','weakness','goodness']

    ity_embs  = [W_E[tok(' '+w, add_special_tokens=False)['input_ids'][0]]
                  for w in ity_targets
                  if len(tok(' '+w, add_special_tokens=False)['input_ids'])==1]
    ness_embs = [W_E[tok(' '+w, add_special_tokens=False)['input_ids'][0]]
                  for w in ness_targets
                  if len(tok(' '+w, add_special_tokens=False)['input_ids'])==1]

    if len(ity_embs) > 1 and len(ness_embs) > 1:
        ity_centroid  = normed(np.mean(ity_embs, axis=0)).astype(np.float32)
        ness_centroid = normed(np.mean(ness_embs, axis=0)).astype(np.float32)
        sep = float(np.dot(ity_centroid, ness_centroid))
        print("  +ity target centroid: n=%d  +ness target centroid: n=%d" %
              (len(ity_embs), len(ness_embs)))
        print("  cos(ity_targets, ness_targets) = %+.4f" % sep)
        print("  (0.0 = separate clusters, 1.0 = same cluster)")
        print()
        # Top 10 nearest tokens to each centroid
        for name, centroid in [('+ity', ity_centroid), ('+ness', ness_centroid)]:
            sims = W_n @ centroid
            sims[~CLEAN_MASK] = -1.0
            top = np.argpartition(sims, -15)[-15:]
            top = top[np.argsort(sims[top])[::-1]]
            words = [tok.decode([i]).strip() for i in top]
            print("  Nearest to %s centroid: %s" % (name, ', '.join(words[:12])))
        print()

    # Compare axis directions
    c_axis = float(np.dot(ax_ity.astype(np.float32), ax_ne.astype(np.float32)))
    print("  cos(+ity_axis, +ness_axis) = %+.4f  (Day 325 confirmed: +0.394)" % c_axis)
    print()
    # Do the axes point to the same centroid?
    # Displace a common adjective word (fast, slow, dark, bright) with each
    test_words = ['fair','clear','equal','legal','moral','human']
    print("  Axis navigation comparison (+ity vs +ness from same source):")
    print("  %-8s  +ity result      +ness result" % "adj")
    for adj in test_words:
        es, sid = get_emb(adj)
        if es is None: continue
        r_ity  = nn_retrieve(W_E[sid]+1.0*ax_ity, source_ids(adj), CLEAN_MASK, 3)
        r_ness = nn_retrieve(W_E[sid]+1.0*ax_ne,  source_ids(adj), CLEAN_MASK, 3)
        print("  %-8s  %-16s %-16s" % (adj, r_ity[0][0], r_ness[0][0]))
    print()

# =====================================================================
# PART E: SUPPLETIVE/IRREGULAR CHAIN — PAST + +ing
# =====================================================================
print("PART E: Suppletive irregular chain test — past tense + +ing")
print("-"*80)

ING_PAIRS = [('go','going'),('take','taking'),('run','running'),('see','seeing'),
              ('give','giving'),('get','getting'),('make','making'),('write','writing'),
              ('read','reading'),('think','thinking'),('know','knowing'),('feel','feeling'),
              ('come','coming'),('bring','bringing'),('hold','holding'),('keep','keeping')]
ABLAUT_CORE = [('go','went'),('take','took'),('give','gave'),('see','saw'),
                ('break','broke'),('choose','chose'),('know','knew'),('drive','drove'),
                ('write','wrote'),('ride','rode'),('bite','bit'),('hide','hid')]

ax_ing, valid_ing, pc_ing = compute_axis(ING_PAIRS)
ax_ab,  valid_ab,  pc_ab  = compute_axis(ABLAUT_CORE)

if ax_ing is not None and ax_ab is not None:
    c = float(np.dot(ax_ing.astype(np.float32), ax_ab.astype(np.float32)))
    print("  cos(+ing, ablaut) = %+.4f  (need < 0.15 for composition)" % c)
    best_s_ab  = 1.0
    best_s_ing = 1.0
    for lo, hi, ax, vl in [(0.1, 3.0, ax_ab, valid_ab), (0.1, 3.0, ax_ing, valid_ing)]:
        bs, _ = best_scale(ax, vl, CLEAN_MASK, lo=lo, hi=hi, n=30)
        if ax is ax_ab: best_s_ab = bs
        else: best_s_ing = bs
    print("  ablaut scale=%.2f  +ing scale=%.2f" % (best_s_ab, best_s_ing))
    print()

    print("  Chain: base_verb --[ablaut]--> past_form --[+ing]--> past+ing?")
    for verb, past_form, ing_form in [('go','went','going'),('take','took','taking'),
                                       ('give','gave','giving'),('write','wrote','writing'),
                                       ('break','broke','breaking'),('know','knew','knowing')]:
        es, sid = get_emb(verb)
        if es is None: continue
        step1 = W_E[sid] + best_s_ab * ax_ab
        r1 = nn_retrieve(step1, source_ids(verb), CLEAN_MASK, 3)
        step2 = step1 + best_s_ing * ax_ing
        r2 = nn_retrieve(step2, source_ids(verb), CLEAN_MASK, 3)
        direct_ing = W_E[sid] + best_s_ing * ax_ing
        r_dir = nn_retrieve(direct_ing, source_ids(verb), CLEAN_MASK, 3)
        t1 = '✓' if r1[0][0]==past_form else '~'
        t2 = '✓' if r2[0][0]==ing_form else '~'
        print("  %s %-7s->%-10s  %s chain->%-12s  (direct +ing: %s)" %
              (t1, verb, r1[0][0], t2, r2[0][0], r_dir[0][0]))
    print()

    # Reverse: +ing first, then ablaut?
    print("  Reverse chain: +ing first, then ablaut?")
    for verb, ing_form, past_form in [('go','going','went'),('take','taking','took'),
                                       ('write','writing','wrote'),('know','knowing','knew')]:
        es, sid = get_emb(verb)
        if es is None: continue
        step1 = W_E[sid] + best_s_ing * ax_ing
        r1 = nn_retrieve(step1, source_ids(verb), CLEAN_MASK, 3)
        step2 = step1 + best_s_ab * ax_ab
        r2 = nn_retrieve(step2, source_ids(verb), CLEAN_MASK, 3)
        t1 = '✓' if r1[0][0]==ing_form else '~'
        t2 = '✓' if r2[0][0] in [past_form, 'going','taking'] else '~'
        print("  %s %-7s->%-10s  chain->%s" % (t1, verb, r1[0][0], r2[0][0]))
print()

# =====================================================================
# PART F: SCALE-FREE COMPOSITION — normalized chain
# =====================================================================
print("PART F: Scale-free composition test")
print("-"*80)
print("  Can we compose morphological operations by normalizing at each step?")
print()
if ax_ab is not None and ax_ing is not None:
    for verb, past_form, ing_form in [('go','went','going'),('take','took','taking'),
                                       ('write','wrote','writing'),('break','broke','breaking')]:
        es, sid = get_emb(verb)
        if es is None: continue
        # Scale-free: normalize the embedding at each step, then apply
        v0_n = normed(W_E[sid])
        # Find best retrieval scale for ablaut on normalized source
        v1_n = normed(W_E[sid] + best_s_ab * ax_ab)
        # Now start from v1_n * ||W_E[sid]|| to preserve magnitude
        v1_mag = np.linalg.norm(W_E[sid])
        step1 = v1_n * v1_mag
        r1 = nn_retrieve(step1, source_ids(verb), CLEAN_MASK, 3)
        step2 = normed(step1 + best_s_ing * ax_ing) * v1_mag
        r2 = nn_retrieve(step2, source_ids(verb), CLEAN_MASK, 3)
        t1 = '✓' if r1[0][0]==past_form else '~'
        t2 = '✓' if r2[0][0]==ing_form else '~'
        print("  %s %-7s (normalized) ->%-10s  %s chain->%s" %
              (t1, verb, r1[0][0], t2, r2[0][0]))
