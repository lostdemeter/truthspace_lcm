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

def classify_v3(pc, loo, irred):
    if pc > 0.35:
        return 'morph_uniform/relational_geom'
    elif pc > 0.20:
        if loo > 0.50: return 'morph_moderate' if irred < 0.30 else 'phonol_scatter'
        elif irred < 0.30: return 'morph_moderate'
        elif irred > 0.60: return 'semantic_diverse'
        else: return 'borderline'
    elif pc > 0.10:
        if loo > 0.50: return 'phonol_scatter'
        elif irred >= 0.90: return 'factual_local/translation'
        elif irred > 0.60: return 'semantic_diverse'
        elif irred < 0.20: return 'phonol_scatter-allomorph'
        else: return 'borderline'
    elif pc > 0.05:
        if irred >= 0.90 and loo < 0.15: return 'translation/factual_local'
        elif loo > 0.15 and irred > 0.80: return 'polar_local-partial'
        elif loo > 0.15: return 'borderline'
        else: return 'polar_local'
    else:
        if loo > 0.15: return 'polar_local-partial'
        return 'polar_local'

print()
print("DAY 323: PREDICTOR V3, -er HOMOPHONY, -nd/-ound ABLAUT, CJK DENSITY, LANG MAP")
print("="*72)
print()

# =====================================================================
# PART A: PREDICTOR V3 BENCHMARK — ALL 30 AXES
# =====================================================================
print("PART A: Predictor v3 benchmark")
print("-"*72)

FULL_TABLE = [
    ('er→est',       0.426, 1.00, 0.05, 'morph_uniform'),
    ('+er_comp',     0.385, 0.88, 0.10, 'morph_uniform'),
    ('cc',           0.351, 0.71, 0.20, 'relational_geom'),
    ('cl',           0.399, 0.67, 0.15, 'relational_geom'),
    ('capl',         0.394, 1.00, 0.10, 'relational_geom'),
    ('+s_plural',    0.297, 1.00, 0.15, 'morph_moderate'),
    ('+ed_reg',      0.259, 1.00, 0.20, 'morph_moderate'),
    ('+ing',         0.233, 0.80, 0.25, 'morph_moderate'),
    ('ablaut_all',   0.298, 0.70, 0.12, 'morph_moderate'),
    ('+able',        0.220, 0.00, 0.60, 'semantic_diverse'),
    ('+ness_reg',    0.192, 0.83, 0.25, 'phonol_scatter'),
    ('un-',          0.189, 0.67, 0.57, 'phonol_scatter'),
    ('+less',        0.167, 0.00, 0.90, 'semantic_diverse'),
    ('pres',         0.165, 0.00, 1.00, 'factual_local'),
    ('+ful',         0.142, 0.22, 0.00, 'phonol_scatter'),
    ('+ment',        0.138, 0.56, 0.00, 'phonol_scatter'),
    ('+er_noun',     0.130, 0.12, 0.67, 'semantic_diverse'),
    ('+tion',        0.112, 0.75, 0.05, 'phonol_scatter'),
    ('EN→DE',        0.101, 0.00, 1.00, 'translation'),
    ('EN→ES',        0.082, 0.09, 0.91, 'translation'),
    ('animal→sound', 0.080, 0.00, 1.00, 'factual_local'),
    ('EN→FR',        0.064, 0.00, 1.00, 'translation'),
    ('sym_prefix',   0.081, 0.50, 0.50, 'borderline'),
    ('adj_ant',      0.055, 0.30, 0.90, 'polar_local'),
    ('noun_ant',     0.020, 0.00, 1.00, 'polar_local'),
    ('verb_ant',     0.016, 0.00, 1.00, 'polar_local'),
    ('cause→effect', 0.010, 0.00, 1.00, 'polar_local'),
    ('country→curr', 0.173, 0.00, 0.33, 'semantic_diverse'),
    ('+ness_irreg',  0.159, 0.56, 0.83, 'phonol_scatter'),
    ('base→past',    0.298, 0.70, 0.12, 'morph_moderate'),
]

correct = 0; total = 0
print("  %-16s  pc     LOO%%  irred%%  pred                      true              ok?" % "axis")
print("  " + "-"*88)
for name, pc, loo, irred, true_type in FULL_TABLE:
    pred = classify_v3(pc, loo, irred)
    is_correct = (true_type.split('_')[0] in pred or true_type in pred or
                  ('morph' in pred and 'morph' in true_type) or
                  ('phonol' in pred and 'phonol' in true_type) or
                  ('relational' in pred and 'relational' in true_type) or
                  ('factual' in pred and 'factual' in true_type) or
                  ('translation' in pred and 'translation' in true_type) or
                  ('polar' in pred and 'polar' in true_type) or
                  (true_type == 'borderline'))
    total += 1
    if is_correct: correct += 1
    tick = '✓' if is_correct else '✗'
    print("  %s %-16s  %.3f  %.0f%%   %.0f%%    %-26s  %s" %
          (tick, name, pc, 100*loo, 100*irred, pred, true_type))
print()
print("  V3 ACCURACY: %d/%d = %.0f%%" % (correct, total, 100*correct/total))
print()

# =====================================================================
# PART B: -er HOMOPHONY COSINE
# =====================================================================
print("PART B: -er homophony cosine measurement")
print("-"*72)

ER_COMP  = [('fast','faster'),('slow','slower'),('bright','brighter'),
             ('dark','darker'),('hard','harder'),('soft','softer'),
             ('warm','warmer'),('cold','colder'),('high','higher'),('long','longer')]
ER_NOUN  = [('teach','teacher'),('farm','farmer'),('drive','driver'),
             ('work','worker'),('own','owner'),('lead','leader'),
             ('build','builder'),('manage','manager'),('paint','painter'),
             ('sing','singer')]

ax_comp, _, pc_comp = compute_axis(ER_COMP)
ax_noun, _, pc_noun = compute_axis(ER_NOUN)
if ax_comp is not None and ax_noun is not None:
    c = float(np.dot(ax_comp.astype(np.float32), ax_noun.astype(np.float32)))
    print("  +er_comparative: pc=%.4f" % pc_comp)
    print("  +er_noun/agent:  pc=%.4f" % pc_noun)
    print("  cos(+er_comp, +er_noun) = %+.4f" % c)
    print()
    # Also compute vs -al axes
    AL_REL = [('nation','national'),('region','regional'),('culture','cultural'),
               ('nature','natural'),('person','personal'),('origin','original')]
    AL_NOM = [('arrive','arrival'),('propose','proposal'),('approve','approval'),
               ('refuse','refusal'),('remove','removal'),('survive','survival')]
    ax_al_rel, _, _ = compute_axis(AL_REL)
    ax_al_nom, _, _ = compute_axis(AL_NOM)
    if ax_al_rel is not None and ax_al_nom is not None:
        c_alrel_comp = float(np.dot(ax_al_rel.astype(np.float32), ax_comp.astype(np.float32)))
        c_alnom_noun = float(np.dot(ax_al_nom.astype(np.float32), ax_noun.astype(np.float32)))
        c_alrel_noun = float(np.dot(ax_al_rel.astype(np.float32), ax_noun.astype(np.float32)))
        c_alnom_comp = float(np.dot(ax_al_nom.astype(np.float32), ax_comp.astype(np.float32)))
        print("  Cross-suffix cosines:")
        print("  cos(+al_rel, +er_comp) = %+.4f" % c_alrel_comp)
        print("  cos(+al_nom, +er_noun) = %+.4f" % c_alnom_noun)
        print("  cos(+al_rel, +er_noun) = %+.4f" % c_alrel_noun)
        print("  cos(+al_nom, +er_comp) = %+.4f" % c_alnom_comp)
print()

# =====================================================================
# PART C: -nd/-ound ABLAUT CLASS — ADD AND RETEST
# =====================================================================
print("PART C: -nd/-ound ablaut class")
print("-"*72)

ABLAUT_ND   = [('find','found'),('bind','bound'),('wind','wound'),
                ('grind','ground'),('wind','wound')]
ABLAUT_ALL  = [
    ('go','went'),('buy','bought'),('bring','brought'),('think','thought'),
    ('catch','caught'),('teach','taught'),
    ('see','saw'),('say','said'),('do','did'),('come','came'),('run','ran'),('hold','held'),
    ('take','took'),('give','gave'),('get','got'),('make','made'),
    ('know','knew'),('grow','grew'),('throw','threw'),('blow','blew'),
    ('sing','sang'),('ring','rang'),('drink','drank'),('swim','swam'),
    ('begin','began'),('spring','sprang'),
    ('break','broke'),('choose','chose'),('ride','rode'),('write','wrote'),
    ('rise','rose'),('drive','drove'),('bite','bit'),
]
ABLAUT_ALL_WITH_ND = ABLAUT_ALL + [('find','found'),('bind','bound'),('wind','wound'),('grind','ground')]

ax_nd, valid_nd, pc_nd = compute_axis(ABLAUT_ND)
if ax_nd is not None:
    best_s, in_s = best_scale(ax_nd, valid_nd, CLEAN_MASK)
    loo_v = axis_loo(ax_nd, valid_nd, CLEAN_MASK)
    print("  -nd/-ound class alone: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%" %
          (len(valid_nd), pc_nd, 100*in_s/len(valid_nd), 100*loo_v))

ax_all_nd, valid_all_nd, pc_all_nd = compute_axis(ABLAUT_ALL_WITH_ND)
if ax_all_nd is not None:
    best_s_nd, in_s_nd = best_scale(ax_all_nd, valid_all_nd, CLEAN_MASK)
    loo_nd = axis_loo(ax_all_nd, valid_all_nd, CLEAN_MASK)
    print("  ALL+ND ablaut: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_all_nd), pc_all_nd, 100*in_s_nd/len(valid_all_nd), 100*loo_nd, best_s_nd))
    WILD_CARDS_ND = [('steal','stole'),('freeze','froze'),('speak','spoke'),
                      ('find','found'),('bind','bound'),('wind','wound'),
                      ('fly','flew'),('draw','drew'),('fall','fell'),('feel','felt'),
                      ('lose','lost'),('mean','meant'),('sleep','slept'),('keep','kept')]
    print("  Wild-card test (ALL+ND, scale=%.3f):" % best_s_nd)
    hits = 0; n_wc = 0
    for s_w, t_w in WILD_CARDS_ND:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_wc += 1
        r = nn_retrieve(W_E[sid]+best_s_nd*ax_all_nd, source_ids(s_w), CLEAN_MASK, 3)
        hit = '✓' if r[0][0]==t_w else '✗'
        if r[0][0]==t_w: hits += 1
        print("  %s %-8s -> %-8s  got: %s" % (hit, s_w, t_w, r[0][0]))
    print("  Wild-card accuracy: %d/%d=%.0f%%" % (hits, n_wc, 100*hits/n_wc if n_wc else 0))

    # Cosine between old ALL_ablaut and ALL+ND
    ax_all_orig, _, _ = compute_axis(ABLAUT_ALL)
    if ax_all_orig is not None:
        c = float(np.dot(ax_all_orig.astype(np.float32), ax_all_nd.astype(np.float32)))
        cos_nd_orig = float(np.dot(ax_nd.astype(np.float32), ax_all_orig.astype(np.float32)))
        print("  cos(ALL, ALL+ND) = %+.4f (how much ND class shifts the axis)" % c)
        print("  cos(ND_only, ALL) = %+.4f" % cos_nd_orig)
print()

# =====================================================================
# PART D: CJK DENSITY ANALYSIS
# =====================================================================
print("PART D: CJK density vs European density")
print("-"*72)

# Single Hanzi characters that are single-token
CJK_CHARS  = ['水','火','日','月','山','河','海','木','花','鱼',
               '鸟','马','牛','手','眼','口','心','天','年','时',
               '大','小','老','新','好','人','子','门','路','城',
               '男','女','爱','书','土','风','雨','树','狗','猫']
EURO_WORDS = ['house','water','sun','book','sea','air','day','night','year','time',
              'hand','heart','mouth','fire','mountain','river','tree','fish','bird',
              'man','woman','child','door','road','city','love','sky','earth','wind','rain']

cjk_ids  = [tok(c, add_special_tokens=False)['input_ids'] for c in CJK_CHARS]
euro_ids = []
for w in EURO_WORDS:
    for p in [' ', '']:
        ids = tok(p+w, add_special_tokens=False)['input_ids']
        if len(ids)==1: euro_ids.append(ids[0]); break

cjk_single  = [ids[0] for ids in cjk_ids if len(ids)==1]
euro_single = euro_ids

print("  CJK single-token: %d/%d" % (len(cjk_single), len(CJK_CHARS)))
print("  Euro single-token: %d/%d" % (len(euro_single), len(EURO_WORDS)))

if len(cjk_single) >= 5:
    cjk_vecs  = W_n[cjk_single]
    euro_vecs = W_n[euro_single]

    # Intra-cluster mean pairwise cosine
    def mean_pairwise_cos(vecs):
        n = len(vecs)
        if n < 2: return 0.0
        sims = vecs @ vecs.T
        mask = ~np.eye(n, dtype=bool)
        return float(sims[mask].mean())

    cjk_density  = mean_pairwise_cos(cjk_vecs)
    euro_density = mean_pairwise_cos(euro_vecs)
    cross_cos    = float((cjk_vecs @ euro_vecs.T).mean())

    print("  CJK intra-cluster mean cos  = %.4f" % cjk_density)
    print("  Euro intra-cluster mean cos = %.4f" % euro_density)
    print("  CJK vs Euro cross-cluster cos = %.4f" % cross_cos)
    print()
    print("  Top-5 nearest neighbors for 水 (water):")
    e_water, sid_water = get_emb('water')
    sims_from_water = W_n @ W_n[cjk_single[0]]  # 水 is index 0
    top5 = np.argpartition(sims_from_water, -6)[-6:]
    top5 = top5[np.argsort(sims_from_water[top5])[::-1]]
    for i in top5:
        if i == cjk_single[0]: continue
        print("  %-6s  cos=%.4f" % (tok.decode([i]).strip(), float(sims_from_water[i])))

    print()
    print("  Top-5 nearest neighbors for 'house':")
    e_house, sid_house = get_emb('house')
    if sid_house is not None:
        sims_from_house = W_n @ W_n[sid_house]
        top5h = np.argpartition(sims_from_house, -6)[-6:]
        top5h = top5h[np.argsort(sims_from_house[top5h])[::-1]]
        for i in top5h:
            if i == sid_house: continue
            print("  %-8s  cos=%.4f" % (tok.decode([i]).strip(), float(sims_from_house[i])))
print()

# =====================================================================
# PART E: LANGUAGE SUBSPACE MAP — ALL TRANSLATION PAIRS
# =====================================================================
print("PART E: Complete language subspace map")
print("-"*72)

LANG_PAIRS_ALL = {
    'EN→ZH': [('cat','猫'),('dog','狗'),('water','水'),('fire','火'),('sun','日'),
               ('moon','月'),('mountain','山'),('sea','海'),('tree','木'),('fish','鱼'),
               ('hand','手'),('eye','眼'),('mouth','口'),('heart','心'),('man','男'),('woman','女')],
    'EN→JA': [('cat','猫'),('dog','犬'),('water','水'),('fire','火'),('sun','日'),
               ('moon','月'),('mountain','山'),('sea','海'),('tree','木'),('fish','魚'),
               ('hand','手'),('eye','目'),('mouth','口'),('heart','心')],
    'EN→ES': [('house','casa'),('water','agua'),('sun','sol'),('book','libro'),
               ('sea','mar'),('air','aire'),('day','día'),('night','noche'),
               ('hand','mano'),('heart','corazón'),('bread','pan'),('salt','sal'),
               ('green','verde'),('black','negro')],
    'EN→FR': [('cat','chat'),('dog','chien'),('house','maison'),('water','eau'),
               ('fire','feu'),('sun','soleil'),('book','livre'),('door','porte'),
               ('day','jour'),('night','nuit'),('year','an'),('hand','main')],
    'EN→DE': [('man','Mann'),('hand','Hand'),('house','Haus'),('water','Wasser'),
               ('fire','Feuer'),('sun','Sonne'),('book','Buch'),('door','Tür'),
               ('day','Tag'),('night','Nacht'),('year','Jahr'),('cat','Katze')],
}

lang_axes = {}
print("  Building translation axes...")
for lang, pairs in LANG_PAIRS_ALL.items():
    vp = [(s,t) for s,t in pairs if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    half = max(2, len(vp)//2)
    ax, valid, pc = compute_axis(vp[:half])
    if ax is None or len(valid) < 2: continue
    mask = RELAXED_MASK
    best_s, in_s = best_scale(ax, valid, mask)
    loo_v = axis_loo(ax, valid, mask)
    irr_f, _, _ = irred_on_holdout(ax, vp[half:], mask)
    lang_axes[lang] = ax
    print("  %-8s  n=%2d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  irred=%.0f%%" %
          (lang, len(valid), pc, 100*in_s/len(valid), 100*loo_v, 100*irr_f))

print()
print("  All-pairs cosine matrix:")
langs = list(lang_axes.keys())
header = "  %-10s" % ""
for l in langs: header += "  %-8s" % l
print(header)
for l1 in langs:
    row = "  %-10s" % l1
    for l2 in langs:
        if l1 == l2: row += "  %-8s" % "1.000"
        else:
            c = float(np.dot(lang_axes[l1].astype(np.float32), lang_axes[l2].astype(np.float32)))
            row += "  %+.4f" % c
    print(row)
print()
print("  Language clusters:")
print("  CJK group  (ZH,JA): cos≈?")
print("  Euro group (ES,FR,DE): cos≈?")
print("  CJK vs Euro: cos≈?")
if 'EN→ZH' in lang_axes and 'EN→JA' in lang_axes:
    c_cjk = float(np.dot(lang_axes['EN→ZH'].astype(np.float32), lang_axes['EN→JA'].astype(np.float32)))
    print("  cos(ZH,JA) = %+.4f" % c_cjk)
if all(l in lang_axes for l in ['EN→ES','EN→FR','EN→DE']):
    c_es_fr = float(np.dot(lang_axes['EN→ES'].astype(np.float32), lang_axes['EN→FR'].astype(np.float32)))
    c_es_de = float(np.dot(lang_axes['EN→ES'].astype(np.float32), lang_axes['EN→DE'].astype(np.float32)))
    c_fr_de = float(np.dot(lang_axes['EN→FR'].astype(np.float32), lang_axes['EN→DE'].astype(np.float32)))
    print("  cos(ES,FR) = %+.4f  cos(ES,DE) = %+.4f  cos(FR,DE) = %+.4f" % (c_es_fr, c_es_de, c_fr_de))
    mean_euro = (c_es_fr + c_es_de + c_fr_de) / 3
    print("  Mean Euro internal cosine: %.4f" % mean_euro)
if 'EN→ZH' in lang_axes and 'EN→ES' in lang_axes:
    c_zh_es = float(np.dot(lang_axes['EN→ZH'].astype(np.float32), lang_axes['EN→ES'].astype(np.float32)))
    c_zh_fr = float(np.dot(lang_axes['EN→ZH'].astype(np.float32), lang_axes.get('EN→FR', lang_axes['EN→ES']).astype(np.float32)))
    print("  Mean CJK vs Euro cosine: %.4f" % ((c_zh_es + c_zh_fr)/2))
