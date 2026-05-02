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
              word[0].upper()+word[1:], word.upper(), ' '+word.upper(),
              '-'+word, '_'+word, ' -'+word]:
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

def nn_retrieve(pred_emb, excl_ids, mask, top_n=8):
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

print()
print("DAY 319: ANTONYM LOCAL GEOMETRY, FACTUAL LOCAL, TRANSLATION, CROSS-SUBSPACE")
print("="*72)
print()

# =====================================================================
# PART A: ANTONYM LOCAL GEOMETRY — WHY DOES in=100% WITH pc~0?
# =====================================================================
print("PART A: Antonym local geometry — nearest neighbors of antonym targets")
print("-"*72)

VERB_ANTONYMS = [('win','lose'),('rise','fall'),('push','pull'),('enter','exit'),
                 ('buy','sell'),('love','hate'),('open','close'),('start','stop')]
NOUN_ANTONYMS = [('war','peace'),('day','night'),('summer','winter'),('life','death'),
                 ('friend','enemy'),('truth','lie'),('good','evil'),('joy','sorrow')]

for set_name, pairs in [('VERB_ANT', VERB_ANTONYMS), ('NOUN_ANT', NOUN_ANTONYMS)]:
    ax, valid, pc = compute_axis(pairs)
    if ax is None: continue
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    print("  %s: pc=%.4f  in=%d/%d  scale=%.3f" % (set_name, pc, in_s, len(valid), best_s))
    for s_w, t_w, sid, tid in valid:
        pred = W_E[sid] + best_s * ax
        r = nn_retrieve(pred, source_ids(s_w), CLEAN_MASK, 5)
        target_cos = float(W_n[tid] @ normed(pred).astype(np.float32))
        nearest_clean_cos = r[0][1]
        hit = '✓' if r[0][0]==t_w else '✗'
        # show rank of actual target
        target_rank = next((i for i,(w,_,_) in enumerate(r) if w==t_w), -1)
        print("  %s %-8s -> %-8s  tgt_cos=%.3f  top1=%-8s(%.3f)  tgt_rank=%d" %
              (hit, s_w, t_w, target_cos, r[0][0], nearest_clean_cos,
               target_rank if target_rank>=0 else 99))
    print()

# Also check: are antonym targets NATURALLY nearest neighbors of source?
print("  Are antonyms naturally close (without axis)?")
for s_w, t_w in VERB_ANTONYMS[:4]:
    es, sid = get_emb(s_w); et, tid = get_emb(t_w)
    if es is None or et is None: continue
    base_sim = float(W_n[tid] @ W_n[sid])
    # rank of t_w in clean neighbors of s_w without any axis
    r = nn_retrieve(W_E[sid], source_ids(s_w), CLEAN_MASK, 20)
    rank = next((i for i,(w,_,_) in enumerate(r) if w==t_w), -1)
    print("  %-8s -> %-8s  cos=%.3f  baseline_rank=%s" %
          (s_w, t_w, base_sim, rank if rank>=0 else '>20'))
print()

# =====================================================================
# PART B: pc FULL CONTINUUM — ALL AXES ON ONE SCALE
# =====================================================================
print("PART B: pc continuum — all axis types ordered by pc")
print("-"*72)

ALL_AXES = {
    'er→est':  [('faster','fastest'),('slower','slowest'),('taller','tallest'),
                 ('shorter','shortest'),('brighter','brightest'),('darker','darkest')],
    '+er':     [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                 ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner')],
    'cc':      [('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
                 ('china','Beijing'),('canada','Ottawa'),('australia','Canberra'),
                 ('india','Delhi'),('russia','Moscow')],
    'cl':      [('france','French'),('germany','German'),('japan','Japanese'),
                 ('china','Chinese'),('turkey','Turkish'),('brazil','Portuguese')],
    'capl':    [('Paris','French'),('Berlin','German'),('Tokyo','Japanese'),
                 ('Beijing','Chinese'),('Moscow','Russian'),('Rome','Italian'),
                 ('Madrid','Spanish'),('Athens','Greek'),('Warsaw','Polish')],
    '+s':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
    '+ed':     [('walk','walked'),('jump','jumped'),('play','played'),('start','started'),
                 ('help','helped'),('work','worked'),('turn','turned'),('push','pushed')],
    'un-':     [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('safe','unsafe'),
                 ('wise','unwise'),('true','untrue'),('sure','unsure'),('clear','unclear')],
    '+ness':   [('happy','happiness'),('sad','sadness'),('kind','kindness'),
                 ('dark','darkness'),('warm','warmth'),('hard','hardness'),
                 ('soft','softness'),('weak','weakness')],
    '+ful':    [('hope','hopeful'),('care','careful'),('harm','harmful'),
                 ('use','useful'),('help','helpful'),('play','playful')],
    '+less':   [('hope','hopeless'),('help','helpless'),('use','useless'),
                 ('care','careless'),('harm','harmless'),('thought','thoughtless')],
    '+able':   [('break','breakable'),('wash','washable'),('read','readable'),
                 ('use','usable'),('move','movable'),('adjust','adjustable')],
    '+tion':   [('act','action'),('direct','direction'),('collect','collection'),
                 ('connect','connection'),('protect','protection'),('select','selection'),
                 ('inject','injection'),('reject','rejection')],
    'pres':    [('france','Macron'),('usa','Biden'),('russia','Putin'),
                 ('china','Xi'),('india','Modi'),('turkey','Erdogan')],
    'adj_ant': [('hot','cold'),('big','small'),('fast','slow'),('hard','soft'),
                 ('dark','light'),('loud','quiet'),('rough','smooth'),('sharp','dull'),
                 ('strong','weak'),('thick','thin')],
    'verb_ant': VERB_ANTONYMS,
    'noun_ant': NOUN_ANTONYMS,
}

results = []
for name, pairs in ALL_AXES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None: continue
    # use appropriate mask
    mask = RELAXED_MASK if name in ('cc','cl','capl','pres') else CLEAN_MASK
    best_s, in_s = best_scale(ax, valid, mask)
    loo = axis_loo(ax, valid, mask)
    results.append((pc, name, len(valid), in_s, loo))

results.sort(reverse=True)
print("  %-12s  pc      n    in%%   LOO%%  type" % "axis")
print("  " + "-"*62)
for pc, name, n, in_s, loo in results:
    # classify
    if pc >= 0.35 and loo >= 0.65:    t = 'morph_uniform'
    elif pc >= 0.35:                   t = 'morph_moderate-high'
    elif pc >= 0.20 and loo >= 0.30:  t = 'morph_moderate'
    elif pc >= 0.20:                   t = 'morph_moderate-low'
    elif in_s/n >= 0.85 and loo >= 0.30: t = 'phonol_scatter?'
    elif loo >= 0.30:                   t = 'morph_moderate?'
    elif pc < 0.05:                    t = 'antonym'
    else:                              t = 'semantic_diverse'
    print("  %-12s  %.4f  %d    %.0f%%   %.0f%%   %s" %
          (name, pc, n, 100*in_s/n, 100*loo, t))
print()

# =====================================================================
# PART C: FACTUAL LOCAL AXES
# =====================================================================
print("PART C: Factual local axes — scientist→discovery, author→character")
print("-"*72)

FACTUAL_AXES = {
    'scientist→discovery': [
        ('einstein','relativity'),('darwin','evolution'),('newton','gravity'),
        ('turing','computing'),('curie','radioactivity'),('mendel','genetics'),
        ('freud','psychoanalysis'),('bohr','quantum'),
    ],
    'author→character': [
        ('shakespeare','Hamlet'),('dickens','Oliver'),('cervantes','Quixote'),
        ('tolstoy','Anna'),('kafka','Gregor'),('doyle','Holmes'),
    ],
    'instrument→music': [
        ('piano','music'),('guitar','rock'),('violin','classical'),
        ('drum','jazz'),('trumpet','brass'),('flute','wind'),
    ],
    'sport→equipment': [
        ('tennis','racket'),('golf','club'),('cricket','bat'),
        ('baseball','glove'),('boxing','gloves'),('fencing','sword'),
    ],
}

print("  %-22s  pc      n  in%%   LOO%%  irred%%  mask" % "axis")
print("  " + "-"*68)
for name, pairs in FACTUAL_AXES.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-22s  n/a" % name); continue
    # Try both masks
    _, in_c = best_scale(ax, valid, CLEAN_MASK)
    _, in_r = best_scale(ax, valid, RELAXED_MASK)
    if in_r > in_c:
        mask = RELAXED_MASK; mask_name = 'relax'
        in_s = in_r
    else:
        mask = CLEAN_MASK; mask_name = 'clean'
        in_s = in_c
    loo = axis_loo(ax, valid, mask)
    # quick irred estimate: use full sweep on leave-2-out holdout
    irred = 0; ho_n = 0
    for i in range(min(4, len(valid))):
        ho_s, ho_t, ho_sid, ho_tid = valid[i]
        if get_emb(ho_t)[0] is None: continue
        ho_n += 1; found = False
        for s_test in np.linspace(0.02, 6.0, 60):
            r = nn_retrieve(W_E[ho_sid]+s_test*ax, source_ids(ho_s), mask, 1)
            if r[0][0] == ho_t: found=True; break
        if not found: irred += 1
    irred_frac = irred/ho_n if ho_n else 0
    print("  %-22s  %.4f  %d  %.0f%%   %.0f%%   %.0f%%    %s" %
          (name, pc, len(valid), 100*in_s/len(valid), 100*loo, 100*irred_frac, mask_name))
print()

# =====================================================================
# PART D: TRANSLATION AXIS — ENGLISH→SPANISH
# =====================================================================
print("PART D: Translation axis — English→Spanish")
print("-"*72)

EN_ES_TRAIN = [
    ('cat','gato'),('dog','perro'),('house','casa'),('water','agua'),
    ('fire','fuego'),('sun','sol'),('moon','luna'),('star','estrella'),
    ('book','libro'),('car','coche'),('tree','árbol'),('door','puerta'),
]
EN_ES_HOLDOUT = [
    ('fish','pez'),('bird','pájaro'),('table','mesa'),('chair','silla'),
    ('milk','leche'),('bread','pan'),('window','ventana'),('key','llave'),
]

# Also FR, DE, IT
EN_FR = [('cat','chat'),('dog','chien'),('house','maison'),('water','eau'),
          ('fire','feu'),('sun','soleil'),('book','livre'),('door','porte')]
EN_DE = [('cat','Katze'),('dog','Hund'),('house','Haus'),('water','Wasser'),
          ('fire','Feuer'),('sun','Sonne'),('book','Buch'),('door','Tür')]

for lang_name, pairs, holdout in [
    ('EN→ES', EN_ES_TRAIN, EN_ES_HOLDOUT),
    ('EN→FR', EN_FR, []),
    ('EN→DE', EN_DE, []),
]:
    ax, valid, pc = compute_axis(pairs)
    if ax is None: continue
    best_s_c, in_c = best_scale(ax, valid, CLEAN_MASK)
    best_s_r, in_r = best_scale(ax, valid, RELAXED_MASK)
    # Spanish words are clean (no caps), French/German have caps for nouns
    mask = CLEAN_MASK if in_c >= in_r else RELAXED_MASK
    mask_name = 'clean' if in_c >= in_r else 'relax'
    in_s = max(in_c, in_r); best_s = best_s_c if in_c >= in_r else best_s_r
    loo = axis_loo(ax, valid, mask)
    print("  %s: pc=%.4f  n=%d  in=%.0f%%  LOO=%.0f%%  scale=%.3f  [%s]" %
          (lang_name, pc, len(valid), 100*in_s/len(valid), 100*loo, best_s, mask_name))
    # Per-pair
    for s_w, t_w, sid, tid in valid:
        r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), mask, 3)
        hit = '✓' if r[0][0]==t_w else '✗'
        print("  %s %-10s -> %-10s  got: %s" % (hit, s_w, t_w, r[0][0]))
    # Holdout
    if holdout:
        ho_hits = 0; ho_n = 0
        for s_w, t_w in holdout:
            es, sid = get_emb(s_w); et, tid = get_emb(t_w)
            if es is None: continue
            ho_n += 1
            r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), mask, 1)
            if r[0][0] == t_w: ho_hits += 1
        print("  Holdout: %d/%d=%.0f%%" % (ho_hits, ho_n, 100*ho_hits/ho_n if ho_n else 0))
    print()

# cos between translation axes and other axes
ax_es, _, pc_es = compute_axis(EN_ES_TRAIN)
ax_fr, _, pc_fr = compute_axis(EN_FR)
ax_de, _, pc_de = compute_axis(EN_DE)

OTHER_AXES = {}
for nm, pairs in [
    ('cc', [('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
            ('china','Beijing'),('canada','Ottawa'),('australia','Canberra')]),
    ('+er',[('fast','faster'),('slow','slower'),('tall','taller'),
            ('short','shorter'),('bright','brighter'),('dark','darker')]),
    ('+s', [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
            ('tree','trees'),('book','books')]),
]:
    a, _, _ = compute_axis(pairs)
    if a is not None: OTHER_AXES[nm] = a

if ax_es is not None and ax_fr is not None:
    print("  Translation axis cosines:")
    for n1, a1 in [('EN→ES', ax_es), ('EN→FR', ax_fr), ('EN→DE', ax_de)]:
        if a1 is None: continue
        for n2, a2 in [('EN→ES', ax_es), ('EN→FR', ax_fr), ('EN→DE', ax_de)] + list(OTHER_AXES.items()):
            if a2 is None: continue
            c = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
            if n1 != n2: print("  cos(%-7s, %-7s) = %+.4f" % (n1, n2, c))
print()

# =====================================================================
# PART E: CROSS-SUBSPACE NAVIGATION
# =====================================================================
print("PART E: Cross-subspace navigation — what happens applying wrong axis type?")
print("-"*72)

# Build fresh axes
ax_cc, valid_cc, _ = compute_axis([('france','Paris'),('germany','Berlin'),
    ('japan','Tokyo'),('china','Beijing'),('canada','Ottawa'),('australia','Canberra'),
    ('india','Delhi'),('russia','Moscow')])
ax_er, valid_er, _ = compute_axis([('fast','faster'),('slow','slower'),
    ('tall','taller'),('short','shorter'),('bright','brighter'),('dark','darker'),
    ('deep','deeper'),('clean','cleaner')])
ax_ant, valid_ant, _ = compute_axis(VERB_ANTONYMS)

best_s_cc, _ = best_scale(ax_cc, valid_cc, RELAXED_MASK)
best_s_er, _ = best_scale(ax_er, valid_er, CLEAN_MASK)
best_s_ant, _ = best_scale(ax_ant, valid_ant, CLEAN_MASK)

# Apply cc_axis to morphological words
print("  Applying country->capital axis to adjectives (wrong subspace):")
for s_w in ['fast','slow','happy','clean','old','young']:
    es, sid = get_emb(s_w)
    if es is None: continue
    r = nn_retrieve(W_E[sid]+best_s_cc*ax_cc, source_ids(s_w), RELAXED_MASK, 3)
    print("  %-8s + cc_axis -> %s" % (s_w, ', '.join(w for w,_,_ in r[:3])))
print()

# Apply +er_axis to country names
print("  Applying +er axis to country names (wrong subspace):")
for s_w in ['france','germany','japan','china']:
    es, sid = get_emb(s_w)
    if es is None: continue
    r = nn_retrieve(W_E[sid]+best_s_er*ax_er, source_ids(s_w), CLEAN_MASK, 3)
    print("  %-10s + er_axis -> %s" % (s_w, ', '.join(w for w,_,_ in r[:3])))
print()

# Apply antonym axis to country names and adj
print("  Applying antonym axis to country names:")
for s_w in ['france','germany','war','day']:
    es, sid = get_emb(s_w)
    if es is None: continue
    r = nn_retrieve(W_E[sid]+best_s_ant*ax_ant, source_ids(s_w), CLEAN_MASK, 3)
    print("  %-10s + ant_axis -> %s" % (s_w, ', '.join(w for w,_,_ in r[:3])))
