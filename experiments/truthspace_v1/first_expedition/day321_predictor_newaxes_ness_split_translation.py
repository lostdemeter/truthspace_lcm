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

def classify_axis(pc, loo, irred):
    if pc > 0.35: return 'morph_uniform/relational_geom'
    elif pc > 0.20 and loo > 0.50: return 'morph_moderate/phonol_scatter-high'
    elif pc > 0.20 and irred < 0.30: return 'morph_moderate'
    elif pc > 0.20 and irred > 0.60: return 'semantic_diverse'
    elif pc > 0.10 and loo > 0.50: return 'phonol_scatter'
    elif pc > 0.10 and irred < 0.10: return 'phonol_scatter-allomorph'
    elif pc > 0.10 and irred > 0.60: return 'semantic_diverse'
    elif pc > 0.05 and irred > 0.90: return 'translation/factual_local'
    elif pc > 0.05: return 'translation-partial/semantic_diverse'
    elif loo > 0: return 'antonym-partial'
    else: return 'polar_local'

print()
print("DAY 321: 3-FEATURE PREDICTOR, NESS DOMAIN SPLIT, TRANSLATION VOCABULARY, NEW AXES")
print("="*72)
print()

# =====================================================================
# PART A: 3-FEATURE PREDICTOR — 5 NEW AXES
# =====================================================================
print("PART A: 3-feature predictor on new axes")
print("-"*72)

NEW_AXES = {
    '+ing': [('walk','walking'),('run','running'),('play','playing'),
              ('talk','talking'),('jump','jumping'),('write','writing'),
              ('speak','speaking'),('think','thinking'),('eat','eating'),('sleep','sleeping')],
    '+er_noun': [('teach','teacher'),('farm','farmer'),('drive','driver'),
                  ('work','worker'),('own','owner'),('lead','leader'),
                  ('build','builder'),('manage','manager')],
    '+ment': [('treat','treatment'),('develop','development'),('govern','government'),
               ('manage','management'),('achieve','achievement'),('improve','improvement'),
               ('judge','judgment'),('move','movement'),('invest','investment')],
    'base→past_tense': [('go','went'),('come','came'),('take','took'),('give','gave'),
                         ('get','got'),('say','said'),('make','made'),('know','knew'),
                         ('see','saw'),('find','found')],
    'sym_prefix': [('symmetric','asymmetric'),('moral','amoral'),('typical','atypical'),
                    ('normal','abnormal'),('social','antisocial'),('legal','illegal'),
                    ('visible','invisible'),('honest','dishonest'),('happy','unhappy')],
}

NEW_HOLDOUTS = {
    '+ing': [('read','reading'),('know','knowing'),('feel','feeling'),('drive','driving'),
              ('make','making'),('come','coming'),('take','taking'),('give','giving')],
    '+er_noun': [('sing','singer'),('fight','fighter'),('hunt','hunter'),
                  ('paint','painter'),('swim','swimmer'),('kill','killer')],
    '+ment': [('engage','engagement'),('adjust','adjustment'),('punish','punishment'),
               ('excite','excitement'),('commit','commitment'),('equip','equipment')],
    'base→past_tense': [('run','ran'),('drink','drank'),('sing','sang'),('write','wrote'),
                         ('break','broke'),('choose','chose'),('fly','flew'),('grow','grew')],
    'sym_prefix': [('regular','irregular'),('rational','irrational'),('possible','impossible'),
                    ('patient','impatient'),('formal','informal'),('fair','unfair')],
}

print("  %-20s  pc      LOO%%  irred%%  pred                       actual?" % "axis")
print("  " + "-"*80)
for name, pairs in NEW_AXES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-20s  n/a" % name); continue
    loo_v = axis_loo(ax, valid, CLEAN_MASK)
    irr_f, n_ho, details = irred_on_holdout(ax, NEW_HOLDOUTS.get(name,[]), CLEAN_MASK)
    pred = classify_axis(pc, loo_v, irr_f)
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    print("  %-20s  %.4f  %.0f%%   %.0f%%    %-30s  n=%d in=%.0f%%" %
          (name, pc, 100*loo_v, 100*irr_f, pred, len(valid), 100*in_s/len(valid)))
    # Details for irred
    if NEW_HOLDOUTS.get(name):
        for s_w, t_w, found_at in details[:4]:
            hit = '✓' if found_at else '✗'
            if found_at:
                print("    %s %-10s -> %-12s  scale=%.3f" % (hit, s_w, t_w, found_at))
            else:
                es, sid = get_emb(s_w)
                if es is not None:
                    r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), CLEAN_MASK, 1)
                    print("    %s %-10s -> %-12s  got: %s" % (hit, s_w, t_w, r[0][0]))
print()

# =====================================================================
# PART B: TRANSLATION VOCABULARY — WHY ONLY n=4 FOR SPANISH?
# =====================================================================
print("PART B: Spanish vocabulary — what IS single-token in Qwen2?")
print("-"*72)

SPANISH_WORDS = [
    # Basic nouns
    ('cat','gato'),('dog','perro'),('house','casa'),('water','agua'),
    ('fire','fuego'),('sun','sol'),('moon','luna'),('star','estrella'),
    ('book','libro'),('car','coche'),('tree','árbol'),('door','puerta'),
    # Food
    ('bread','pan'),('wine','vino'),('milk','leche'),('fish','pez'),
    ('meat','carne'),('cheese','queso'),('salt','sal'),('rice','arroz'),
    # Colors
    ('red','rojo'),('blue','azul'),('green','verde'),('black','negro'),
    ('white','blanco'),('yellow','amarillo'),
    # Body
    ('hand','mano'),('foot','pie'),('eye','ojo'),('head','cabeza'),
    ('heart','corazón'),('mouth','boca'),
    # Nature
    ('sea','mar'),('sky','cielo'),('earth','tierra'),('air','aire'),
    ('day','día'),('night','noche'),('year','año'),('time','tiempo'),
]

single_tok_pairs = []
print("  Checking %d Spanish words for single-token status..." % len(SPANISH_WORDS))
for en, es in SPANISH_WORDS:
    e_en, sid_en = get_emb(en)
    e_es, sid_es = get_emb(es)
    en_ok = e_en is not None
    es_ok = e_es is not None
    if en_ok and es_ok:
        single_tok_pairs.append((en, es))

print("  Single-token pairs: %d/%d" % (len(single_tok_pairs), len(SPANISH_WORDS)))
print("  Valid pairs:")
for en, es in single_tok_pairs:
    print("    %-10s -> %s" % (en, es))
print()

# Now compute a better EN->ES axis with all valid pairs
if len(single_tok_pairs) >= 4:
    half = len(single_tok_pairs) // 2
    train_pairs = single_tok_pairs[:half]
    test_pairs  = single_tok_pairs[half:]
    ax_es, valid_es, pc_es = compute_axis(train_pairs)
    if ax_es is not None:
        best_s, in_s = best_scale(ax_es, valid_es, CLEAN_MASK)
        loo_v = axis_loo(ax_es, valid_es, CLEAN_MASK)
        irr_f, n_ho, _ = irred_on_holdout(ax_es, test_pairs, CLEAN_MASK)
        print("  EN->ES (n=%d train, n=%d holdout):" % (len(valid_es), len(test_pairs)))
        print("    pc=%.4f  in=%.0f%%  LOO=%.0f%%  irred=%.0f%%  scale=%.3f" %
              (pc_es, 100*in_s/len(valid_es), 100*loo_v, 100*irr_f, best_s))
        print("    Predicted type: %s" % classify_axis(pc_es, loo_v, irr_f))
print()

# =====================================================================
# PART C: +NESS DOMAIN SPLIT — FORMAL AXIS COMPARISON
# =====================================================================
print("PART C: +ness domain split — two separate axes")
print("-"*72)

NESS_REGULAR = [('happy','happiness'),('sad','sadness'),('kind','kindness'),
                 ('dark','darkness'),('hard','hardness'),('soft','softness'),
                 ('weak','weakness'),('cold','coldness'),('loud','loudness'),
                 ('bold','boldness')]
NESS_IRREGULAR = [('wide','width'),('long','length'),('high','height'),
                   ('broad','breadth'),('deep','depth'),('strong','strength'),
                   ('young','youth'),('wise','wisdom'),('free','freedom')]

REG_HOLDOUT  = [('sick','sickness'),('thick','thickness'),('rich','richness'),
                 ('fresh','freshness'),('bright','brightness'),('cool','coolness'),
                 ('rough','roughness'),('sweet','sweetness')]
IRREG_HOLDOUT = [('good','goodness'),('great','greatness'),('old','age'),
                  ('hot','heat'),('cool','cold'),('fast','speed')]

for name, pairs, holdout in [
    ('+ness_regular', NESS_REGULAR, REG_HOLDOUT),
    ('+ness_irregular', NESS_IRREGULAR, IRREG_HOLDOUT),
]:
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %s: n/a (too few valid pairs, n=%d)" % (name, len(valid))); continue
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    loo_v = axis_loo(ax, valid, CLEAN_MASK)
    irr_f, n_ho, details = irred_on_holdout(ax, holdout, CLEAN_MASK)
    print("  %-18s  n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  irred=%.0f%%  -> %s" %
          (name, len(valid), pc, 100*in_s/len(valid), 100*loo_v, 100*irr_f,
           classify_axis(pc, loo_v, irr_f)))
    for s_w, t_w, found_at in details:
        hit = '✓' if found_at else '✗'
        if found_at:
            print("    %s %-10s -> %-12s  scale=%.3f" % (hit, s_w, t_w, found_at))
        else:
            es, sid = get_emb(s_w)
            if es is not None:
                r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), CLEAN_MASK, 1)
                print("    %s %-10s -> %-12s  got: %s" % (hit, s_w, t_w, r[0][0]))
print()

# cos between the two +ness axes
ax_reg, _, _ = compute_axis(NESS_REGULAR)
ax_irr, _, _ = compute_axis(NESS_IRREGULAR)
if ax_reg is not None and ax_irr is not None:
    c = float(np.dot(ax_reg.astype(np.float32), ax_irr.astype(np.float32)))
    print("  cos(+ness_regular, +ness_irregular) = %+.4f" % c)
    print("  (positive = same direction, 0 = orthogonal, negative = opposite)")
print()

# =====================================================================
# PART D: BOUNDARY CASES — AXES IN 0.05-0.15 RANGE
# =====================================================================
print("PART D: Boundary cases — 0.05 < pc < 0.15 zone")
print("-"*72)

BOUNDARY_AXES = {
    # Known ambiguous
    '+ful':      [('hope','hopeful'),('care','careful'),('harm','harmful'),
                   ('use','useful'),('help','helpful'),('play','playful'),
                   ('power','powerful'),('wonder','wonderful'),('color','colorful')],
    '+ness_mix': [('happy','happiness'),('sad','sadness'),('kind','kindness'),
                   ('wide','width'),('long','length'),('deep','depth'),
                   ('strong','strength'),('soft','softness'),('hard','hardness')],
    # New candidates in this range
    'cause→effect': [('rain','flood'),('heat','drought'),('cold','snow'),
                      ('wind','storm'),('fire','smoke'),('sun','tan')],
    'animal→sound': [('dog','bark'),('cat','meow'),('cow','moo'),('duck','quack'),
                      ('lion','roar'),('bee','buzz'),('horse','neigh'),('snake','hiss')],
    'job→tool':     [('carpenter','hammer'),('artist','brush'),('chef','knife'),
                     ('doctor','stethoscope'),('soldier','gun'),('farmer','plow')],
    'country→currency': [('usa','dollar'),('uk','pound'),('japan','yen'),
                          ('china','yuan'),('india','rupee'),('europe','euro'),
                          ('russia','ruble'),('brazil','real')],
}

BOUNDARY_HOLDOUTS = {
    '+ful': [('beauty','beautiful'),('joy','joyful'),('stress','stressful'),
              ('cheer','cheerful'),('dread','dreadful'),('grace','graceful')],
    'cause→effect': [('smoke','fire'),('drought','thirst'),('war','destruction')],
    'animal→sound': [('frog','croak'),('wolf','howl'),('bird','chirp'),('pig','oink')],
    'job→tool': [('teacher','chalk'),('painter','brush'),('writer','pen')],
    'country→currency': [('mexico','peso'),('australia','dollar'),('canada','dollar')],
}

print("  %-20s  pc      LOO%%  irred%%  pred                     n" % "axis")
print("  " + "-"*75)
for name, pairs in BOUNDARY_AXES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-20s  n/a  (n=%d valid)" % (name, len(valid))); continue
    mask = RELAXED_MASK if name in ('country→currency',) else CLEAN_MASK
    loo_v = axis_loo(ax, valid, mask)
    irr_f, n_ho, _ = irred_on_holdout(ax, BOUNDARY_HOLDOUTS.get(name,[]), mask)
    pred = classify_axis(pc, loo_v, irr_f)
    best_s, in_s = best_scale(ax, valid, mask)
    print("  %-20s  %.4f  %.0f%%   %.0f%%    %-25s  n=%d in=%.0f%%" %
          (name, pc, 100*loo_v, 100*irr_f, pred, len(valid), 100*in_s/len(valid)))
print()
