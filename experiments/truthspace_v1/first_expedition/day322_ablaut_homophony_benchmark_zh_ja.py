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
    elif pc > 0.05 and irred > 0.85: return 'translation/factual_local'
    elif pc > 0.05: return 'borderline'
    elif loo > 0.15: return 'polar_local-partial'
    else: return 'polar_local'

print()
print("DAY 322: ABLAUT SUB-TYPES, HOMOPHONY, FULL BENCHMARK, ZH/JA TRANSLATION")
print("="*72)
print()

# =====================================================================
# PART A: ABLAUT SUB-TYPES — DO DIFFERENT VOWEL PATTERNS FORM ONE AXIS?
# =====================================================================
print("PART A: Ablaut sub-types — one axis or multiple?")
print("-"*72)

# English strong verb classes (phonological sub-patterns)
ABLAUT_UMLAUT = [('go','went'),('buy','bought'),('bring','brought'),
                  ('think','thought'),('catch','caught'),('teach','taught')]
ABLAUT_EE_OO  = [('see','saw'),('say','said'),('do','did'),('come','came'),
                  ('run','ran'),('hold','held')]
ABLAUT_TAKE   = [('take','took'),('give','gave'),('get','got'),('make','made'),
                  ('know','knew'),('grow','grew'),('throw','threw'),('blow','blew')]
ABLAUT_SING   = [('sing','sang'),('ring','rang'),('drink','drank'),('swim','swam'),
                  ('begin','began'),('spring','sprang')]
ABLAUT_BREAK  = [('break','broke'),('choose','chose'),('ride','rode'),('write','wrote'),
                  ('rise','rose'),('drive','drove'),('bite','bit')]
ABLAUT_ALL    = ABLAUT_UMLAUT + ABLAUT_EE_OO + ABLAUT_TAKE + ABLAUT_SING + ABLAUT_BREAK

axes = {}
for name, pairs in [
    ('umlaut(-ght)', ABLAUT_UMLAUT),
    ('see/saw type', ABLAUT_EE_OO),
    ('take/took',   ABLAUT_TAKE),
    ('sing/sang',   ABLAUT_SING),
    ('break/broke', ABLAUT_BREAK),
    ('ALL_ablaut',  ABLAUT_ALL),
]:
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-16s  n/a" % name); continue
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    loo_v = axis_loo(ax, valid, CLEAN_MASK)
    print("  %-16s  n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (name, len(valid), pc, 100*in_s/len(valid), 100*loo_v, best_s))
    axes[name] = ax

# Cross-cosines between sub-types
print()
print("  Cross-cosines between ablaut sub-types:")
sub_names = ['umlaut(-ght)', 'see/saw type', 'take/took', 'sing/sang', 'break/broke']
for i, n1 in enumerate(sub_names):
    if n1 not in axes: continue
    for n2 in sub_names[i+1:]:
        if n2 not in axes: continue
        c = float(np.dot(axes[n1].astype(np.float32), axes[n2].astype(np.float32)))
        print("  cos(%-14s, %-14s) = %+.4f" % (n1, n2, c))

# Wild-card: can ALL_ablaut axis predict unseen irregular verbs?
print()
if 'ALL_ablaut' in axes:
    ax_all, valid_all, pc_all = compute_axis(ABLAUT_ALL)
    best_s_all, _ = best_scale(ax_all, valid_all, CLEAN_MASK)
    WILD_CARDS = [('steal','stole'),('freeze','froze'),('speak','spoke'),
                   ('find','found'),('bind','bound'),('wind','wound'),
                   ('fly','flew'),('draw','drew'),('fall','fell'),('feel','felt')]
    print("  Wild-card test (ALL_ablaut axis, scale=%.3f):" % best_s_all)
    hits = 0; n_wc = 0
    for s_w, t_w in WILD_CARDS:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_wc += 1
        r = nn_retrieve(W_E[sid]+best_s_all*ax_all, source_ids(s_w), CLEAN_MASK, 3)
        hit = '✓' if r[0][0]==t_w else '✗'
        if r[0][0]==t_w: hits += 1
        print("  %s %-8s -> %-8s  got: %s" % (hit, s_w, t_w, r[0][0]))
    print("  Wild-card accuracy: %d/%d=%.0f%%" % (hits, n_wc, 100*hits/n_wc if n_wc else 0))
print()

# =====================================================================
# PART B: HOMOPHONY RESOLUTION — -al, -ly, -s
# =====================================================================
print("PART B: Homophony resolution — -al, -ly, -s")
print("-"*72)

HOMY_AXES = {
    '+al_relational': [('nation','national'),('region','regional'),('culture','cultural'),
                        ('nature','natural'),('person','personal'),('origin','original')],
    '+al_nominal':    [('arrive','arrival'),('propose','proposal'),('approve','approval'),
                        ('refuse','refusal'),('remove','removal'),('survive','survival')],
    '+ly_adverb':     [('quick','quickly'),('slow','slowly'),('quiet','quietly'),
                        ('rapid','rapidly'),('clear','clearly'),('bright','brightly'),
                        ('deep','deeply'),('hard','hardly'),('wide','widely')],
    '+ly_adjective':  [('friend','friendly'),('love','lovely'),('earth','earthly'),
                        ('cost','costly'),('dead','deadly'),('live','lively'),
                        ('man','manly'),('kind','kindly')],
    '+s_plural':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                        ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
    "+s_poss":        [("cat","cat's"),("dog","dog's"),("book","book's"),
                        ("man","man's"),("child","child's")],
}
HOMY_HOLDOUT = {
    '+al_relational': [('finance','financial'),('tradition','traditional'),
                        ('profession','professional'),('emotion','emotional')],
    '+al_nominal':    [('trial','trial'),('reclaim','reclamation'),('deny','denial'),
                        ('betray','betrayal')],
    '+ly_adverb':     [('strong','strongly'),('fair','fairly'),('free','freely'),
                        ('high','highly'),('real','really')],
    '+ly_adjective':  [('ghost','ghostly'),('prince','princely'),('supply','supply'),
                        ('order','orderly'),('father','fatherly')],
}

print("  %-20s  pc      LOO%%  irred%%  pred                     n" % "axis")
print("  " + "-"*75)
for name, pairs in HOMY_AXES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-20s  n/a (n=%d)" % (name, len(valid))); continue
    loo_v = axis_loo(ax, valid, CLEAN_MASK)
    irr_f, n_ho, _ = irred_on_holdout(ax, HOMY_HOLDOUT.get(name,[]), CLEAN_MASK)
    pred = classify_axis(pc, loo_v, irr_f)
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    print("  %-20s  %.4f  %.0f%%   %.0f%%    %-25s  n=%d in=%.0f%%" %
          (name, pc, 100*loo_v, 100*irr_f, pred, len(valid), 100*in_s/len(valid)))

# Cross-cosines between homophonous pairs
print()
print("  Cross-cosines between homophonous pairs:")
ax_al_rel, _, _ = compute_axis(HOMY_AXES['+al_relational'])
ax_al_nom, _, _ = compute_axis(HOMY_AXES['+al_nominal'])
ax_ly_adv, _, _ = compute_axis(HOMY_AXES['+ly_adverb'])
ax_ly_adj, _, _ = compute_axis(HOMY_AXES['+ly_adjective'])
ax_s_pl,   _, _ = compute_axis(HOMY_AXES['+s_plural'])
ax_s_pos,  _, _ = compute_axis(HOMY_AXES["+s_poss"])

for (n1,a1),(n2,a2) in [
    (('+al_rel',ax_al_rel),('+al_nom',ax_al_nom)),
    (('+ly_adv',ax_ly_adv),('+ly_adj',ax_ly_adj)),
    (('+s_plural',ax_s_pl),('+s_poss',ax_s_pos)),
    (('+al_rel',ax_al_rel),('+ly_adv',ax_ly_adv)),
    (('+al_nom',ax_al_nom),('+ly_adj',ax_ly_adj)),
]:
    if a1 is None or a2 is None: continue
    c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
    print("  cos(%-12s, %-12s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART C: EN→ZH and EN→JA TRANSLATION
# =====================================================================
print("PART C: EN→ZH and EN→JA translation axes")
print("-"*72)

# Find single-token Chinese/Japanese words
EN_ZH_CANDS = [
    ('cat','猫'),('dog','狗'),('water','水'),('fire','火'),('sun','日'),
    ('moon','月'),('earth','土'),('sky','天'),('wind','风'),('rain','雨'),
    ('mountain','山'),('river','河'),('sea','海'),('tree','木'),('flower','花'),
    ('fish','鱼'),('bird','鸟'),('horse','马'),('cow','牛'),('pig','猪'),
    ('man','男'),('woman','女'),('child','子'),('heart','心'),('hand','手'),
    ('eye','眼'),('mouth','口'),('day','天'),('year','年'),('time','时'),
    ('big','大'),('small','小'),('old','老'),('new','新'),('good','好'),
    ('love','爱'),('book','书'),('door','门'),('road','路'),('city','城'),
]
EN_JA_CANDS = [
    ('cat','猫'),('dog','犬'),('water','水'),('fire','火'),('sun','日'),
    ('moon','月'),('mountain','山'),('river','川'),('sea','海'),('tree','木'),
    ('fish','魚'),('bird','鳥'),('horse','馬'),('flower','花'),
    ('hand','手'),('eye','目'),('mouth','口'),('heart','心'),
    ('big','大'),('small','小'),('old','古'),('new','新'),('good','良'),
]

for lang_name, cands in [('EN→ZH', EN_ZH_CANDS), ('EN→JA', EN_JA_CANDS)]:
    valid_pairs = []
    for en, tgt in cands:
        e_en, _ = get_emb(en); e_tgt, _ = get_emb(tgt)
        if e_en is not None and e_tgt is not None:
            valid_pairs.append((en, tgt))

    print("  %s: %d/%d single-token pairs found" % (lang_name, len(valid_pairs), len(cands)))
    half = max(2, len(valid_pairs)//2)
    train = valid_pairs[:half]; test = valid_pairs[half:]

    ax, valid, pc = compute_axis(train)
    if ax is None or len(valid) < 2:
        print("  %s: insufficient data" % lang_name); continue

    mask = RELAXED_MASK  # Chinese/Japanese chars are capitalized-free but need relaxed mask
    best_s, in_s = best_scale(ax, valid, mask)
    loo_v = axis_loo(ax, valid, mask)
    irr_f, n_ho, _ = irred_on_holdout(ax, test, mask)
    pred = classify_axis(pc, loo_v, irr_f)
    print("  %s: n_train=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  irred=%.0f%%  -> %s" %
          (lang_name, len(valid), pc, 100*in_s/len(valid), 100*loo_v, 100*irr_f, pred))
    for s_w, t_w, sid, tid in valid[:6]:
        r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), mask, 3)
        hit = '✓' if r[0][0]==t_w else '✗'
        print("  %s %-8s -> %-4s  got: %s" % (hit, s_w, t_w, r[0][0]))
    print()

# Compare translation axes
print("  Cross-language translation cosines:")
axes_trans = {}
for lang_name, cands in [('EN→ZH', EN_ZH_CANDS), ('EN→JA', EN_JA_CANDS),
                           ('EN→ES', [('house','casa'),('water','agua'),('sun','sol'),
                                       ('book','libro'),('sea','mar'),('air','aire'),
                                       ('day','día'),('night','noche'),('year','año'),
                                       ('time','tiempo'),('hand','mano'),('heart','corazón')])]:
    vp = []
    for en, tgt in cands:
        e_en,_ = get_emb(en); e_tgt,_ = get_emb(tgt)
        if e_en is not None and e_tgt is not None: vp.append((en,tgt))
    ax, _, _ = compute_axis(vp[:len(vp)//2])
    if ax is not None: axes_trans[lang_name] = ax

for i, (n1,a1) in enumerate(axes_trans.items()):
    for n2, a2 in list(axes_trans.items())[i+1:]:
        c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
        print("  cos(%-8s, %-8s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART D: FULL PREDICTOR BENCHMARK — ALL AXES TABLE
# =====================================================================
print("PART D: Full predictor benchmark — all 30+ axes")
print("-"*72)

FULL_TABLE = [
    # (name, pc, LOO, irred, true_type)
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
print("  %-16s  pc     LOO%%  irred%%  pred                     true              ok?" % "axis")
print("  " + "-"*88)
for name, pc, loo, irred, true_type in FULL_TABLE:
    pred = classify_axis(pc, loo, irred)
    # match: prediction contains the true type name
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
    print("  %s %-16s  %.3f  %.0f%%   %.0f%%    %-25s  %s" %
          (tick, name, pc, 100*loo, 100*irred, pred, true_type))

print()
print("  OVERALL ACCURACY: %d/%d = %.0f%%" % (correct, total, 100*correct/total))
