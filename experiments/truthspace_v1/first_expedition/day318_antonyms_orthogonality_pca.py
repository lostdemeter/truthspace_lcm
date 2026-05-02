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
              '-'+word, '_'+word, ' -'+word, ' ']:
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
    if len(chords) < 2: return None, 0.0, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, 0.0, valid, pc

def best_scale(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask, 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

def axis_loo(valid, mask):
    if len(valid) < 3: return 0.0, 0
    chords_full = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    ax_full = normed(np.mean(chords_full, axis=0))
    global_s, _ = best_scale(ax_full, valid, mask)
    hits = 0
    for i in range(len(valid)):
        test_s, test_t, test_sid, _ = valid[i]
        train_v = [valid[j] for j in range(len(valid)) if j != i]
        ax_loo = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in train_v], axis=0))
        r = nn_retrieve(W_E[test_sid]+global_s*ax_loo, source_ids(test_s), mask, 1)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid), len(valid)

def irred_sweep(axis, holdout, mask, lo=0.02, hi=6.0, n=100):
    irred = 0; n_ho = 0; details = []
    for src, tgt in holdout:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        n_ho += 1; found_at = None
        for s in np.linspace(lo, hi, n):
            r = nn_retrieve(W_E[sid]+s*axis, source_ids(src), mask, 1)
            if r[0][0] == tgt: found_at = s; break
        if found_at is None:
            irred += 1; details.append((src, tgt, False, None))
        else:
            details.append((src, tgt, True, found_at))
    return irred/n_ho if n_ho else 0.0, n_ho, details

print()
print("DAY 318: ANTONYMS, EXTENDED ORTHOGONALITY, MULTI-HOP, PCA SUBSPACE")
print("="*72)
print()

# =====================================================================
# PART A: country→president LOO AND CLASSIFICATION
# =====================================================================
print("PART A: country→president — full classification")
print("-"*72)

PRES_TRAIN = [
    ('france','Macron'),('usa','Biden'),('russia','Putin'),
    ('china','Xi'),('brazil','Lula'),('india','Modi'),
    ('turkey','Erdogan'),('argentina','Milei'),('south','Ramaphosa'),
]
PRES_HOLDOUT = [
    ('germany','Scholz'),('japan','Kishida'),('canada','Trudeau'),
    ('australia','Albanese'),('mexico','Obrador'),
]

ax_pres, _, valid_pres, pc_pres = compute_axis(PRES_TRAIN)
if ax_pres is not None:
    best_s, in_s = best_scale(ax_pres, valid_pres, RELAXED_MASK)
    loo, _ = axis_loo(valid_pres, RELAXED_MASK)
    print("  country->president: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_pres), pc_pres, 100*in_s/len(valid_pres), 100*loo, best_s))
    irred_frac, n_ho, details = irred_sweep(ax_pres, PRES_HOLDOUT, RELAXED_MASK)
    print("  Holdout irred: %.0f%%  (%d pairs)" % (100*irred_frac, n_ho))
    for src, tgt, found, s_val in details:
        if found: print("  ✓ %-12s -> %-12s  at scale %.3f" % (src, tgt, s_val))
        else:
            r = nn_retrieve(W_E[next(iter(source_ids(src)|{list(source_ids(src))[0]}))]+best_s*ax_pres, source_ids(src), RELAXED_MASK, 1) if source_ids(src) else [('?',0,0)]
            es, sid = get_emb(src)
            if es is not None:
                r = nn_retrieve(W_E[sid]+best_s*ax_pres, source_ids(src), RELAXED_MASK, 1)
                print("  ✗ %-12s -> %-12s  got: %s" % (src, tgt, r[0][0]))
    # Type classification
    t = 'morph_uniform' if loo >= 0.65 else 'morph_moderate' if loo >= 0.30 else \
        'phonol_scatter' if in_s/len(valid_pres) >= 0.85 else 'semantic_diverse'
    print("  Classification: %s" % t)
print()

# =====================================================================
# PART B: EXTENDED ORTHOGONALITY MATRIX
# =====================================================================
print("PART B: Extended axis orthogonality matrix")
print("-"*72)

# Build all axes
ALL_AXIS_PAIRS = {
    'cc':    [('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
               ('china','Beijing'),('canada','Ottawa'),('australia','Canberra'),
               ('india','Delhi'),('russia','Moscow')],
    'cl':    [('france','French'),('germany','German'),('japan','Japanese'),
               ('china','Chinese'),('turkey','Turkish'),('brazil','Portuguese'),
               ('sweden','Swedish'),('norway','Norwegian'),('poland','Polish')],
    'capl':  [('Paris','French'),('Berlin','German'),('Tokyo','Japanese'),
               ('Beijing','Chinese'),('Moscow','Russian'),('Rome','Italian'),
               ('Madrid','Spanish'),('Athens','Greek'),('Warsaw','Polish')],
    'pres':  [('france','Macron'),('usa','Biden'),('russia','Putin'),
               ('china','Xi'),('brazil','Lula'),('india','Modi'),('turkey','Erdogan')],
    '+er':   [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
               ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner')],
    '+s':    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
               ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
    '+ed':   [('walk','walked'),('jump','jumped'),('play','played'),('start','started'),
               ('help','helped'),('work','worked'),('turn','turned'),('push','pushed')],
    '+tion': [('act','action'),('direct','direction'),('collect','collection'),
               ('connect','connection'),('protect','protection'),('select','selection'),
               ('inject','injection'),('reject','rejection')],
    'un-':   [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('safe','unsafe'),
               ('wise','unwise'),('true','untrue'),('sure','unsure'),('clear','unclear')],
}

axes = {}
pcs  = {}
for name, pairs in ALL_AXIS_PAIRS.items():
    ax, _, v, pc = compute_axis(pairs)
    if ax is not None: axes[name] = ax; pcs[name] = pc

names = list(axes.keys())
print("  pc values:")
for n in names:
    print("    %-8s  pc=%.4f" % (n, pcs[n]))
print()

print("  Pairwise cosines (upper triangle):")
header = "  %-8s" % "" + "".join(" %-7s" % n for n in names)
print(header)
for na in names:
    row = "  %-8s" % na
    for nb in names:
        c = float(np.dot(axes[na].astype(np.float32), axes[nb].astype(np.float32)))
        row += " %+6.3f" % c
    print(row)
print()

# =====================================================================
# PART C: ANTONYM AXIS — TRUE NON-GEOMETRIC?
# =====================================================================
print("PART C: Antonym axis — is opposition geometric?")
print("-"*72)

ANTONYM_SETS = {
    'adjective_ant': [
        ('hot','cold'),('big','small'),('fast','slow'),('hard','soft'),
        ('dark','light'),('loud','quiet'),('rough','smooth'),('sharp','dull'),
        ('strong','weak'),('thick','thin'),('clean','dirty'),('heavy','light'),
    ],
    'adverb_ant': [
        ('quickly','slowly'),('loudly','quietly'),('softly','roughly'),
        ('early','late'),('often','rarely'),('always','never'),
    ],
    'verb_ant': [
        ('win','lose'),('rise','fall'),('push','pull'),('enter','exit'),
        ('buy','sell'),('love','hate'),('open','close'),('start','stop'),
    ],
    'noun_ant': [
        ('war','peace'),('day','night'),('summer','winter'),('life','death'),
        ('friend','enemy'),('truth','lie'),('good','evil'),('joy','sorrow'),
    ],
}

print("  %-16s  pc      n  in%%   LOO%%  notes" % "axis")
print("  " + "-"*62)
for name, pairs in ANTONYM_SETS.items():
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None: print("  %-16s  n/a" % name); continue
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    loo, _ = axis_loo(valid, CLEAN_MASK)
    print("  %-16s  %.4f  %d  %.0f%%   %.0f%%" % (name, pc, len(valid), 100*in_s/len(valid), 100*loo))

    # Show in-sample hits
    found = []
    for s_w, t_w, sid, _ in valid:
        r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), CLEAN_MASK, 1)
        if r[0][0] == t_w: found.append("%s->%s" % (s_w, t_w))
    if found: print("    found: " + ', '.join(found))
print()

# Focus on adjective antonyms for deeper analysis
ax_adj_ant, _, valid_adj_ant, pc_adj_ant = compute_axis(ANTONYM_SETS['adjective_ant'])
if ax_adj_ant is not None:
    # Are antonym chords consistently pointing toward each other?
    # i.e., is hot->cold similar direction to big->small?
    # If antonyms are in opposite semantic clusters, all chords should have same direction

    # Compare pc to "expected random" (0.0) vs structured (+)
    print("  Deeper analysis of adjective antonyms:")
    # Chord magnitudes and directions
    chords = []
    for s_w, t_w, sid, tid in valid_adj_ant:
        chord = W_E[tid] - W_E[sid]
        chords.append(normed(chord).astype(np.float32))
    if chords:
        # What is the mean pc?
        pcs_pairs = [float(np.dot(chords[i], chords[j]))
                     for i in range(len(chords)) for j in range(i+1, len(chords))]
        print("  Mean pc=%.4f  min=%.4f  max=%.4f  n_pairs=%d" %
              (np.mean(pcs_pairs), min(pcs_pairs), max(pcs_pairs), len(pcs_pairs)))
        print("  (positive pc = all antonym chords point in same direction)")

    # Test reverse: does cold->hot work with -1x axis?
    best_s_fwd, in_fwd = best_scale(ax_adj_ant, valid_adj_ant, CLEAN_MASK)
    ax_rev = -ax_adj_ant
    best_s_rev, in_rev = best_scale(ax_rev, valid_adj_ant, CLEAN_MASK)
    print("  Forward (hot->cold): in=%.0f%%  scale=%.3f" % (100*in_fwd/len(valid_adj_ant), best_s_fwd))
    print("  Reverse (cold->hot): in=%.0f%%  scale=%.3f (using -axis)" % (100*in_rev/len(valid_adj_ant), best_s_rev))
    print()

    # cos between adj_ant axis and other axes
    print("  cos(adj_ant, other axes):")
    for name, ax in axes.items():
        c = float(np.dot(ax_adj_ant.astype(np.float32), ax.astype(np.float32)))
        print("    adj_ant vs %-8s  cos=%+.4f" % (name, c))
print()

# =====================================================================
# PART D: MULTI-HOP CHAIN — HOW DEEP CAN WE GO?
# =====================================================================
print("PART D: Multi-hop chain depth test")
print("-"*72)

# Build complete chain: country -> capital -> language -> ???
# We have: ax_cc, ax_capl
# What axes can we chain from language?

# Build language->country_adj axis (French->France-like?)
# Actually: let's chain to adjective form
# language name -> native speaker demonym? French -> Frenchman?
# Or: language -> continent?

# Try: language -> [native adjective] (French -> french = already is adjective)
# Instead: test continuing beyond language with known axis

ax_cc, _, valid_cc, _   = compute_axis([('france','Paris'),('germany','Berlin'),
    ('japan','Tokyo'),('china','Beijing'),('canada','Ottawa'),('australia','Canberra'),
    ('india','Delhi'),('russia','Moscow')])
ax_capl, _, valid_capl, _ = compute_axis([('Paris','French'),('Berlin','German'),
    ('Tokyo','Japanese'),('Beijing','Chinese'),('Moscow','Russian'),('Rome','Italian'),
    ('Madrid','Spanish'),('Athens','Greek'),('Warsaw','Polish')])

best_s_cc,   _ = best_scale(ax_cc,   valid_cc,   RELAXED_MASK)
best_s_capl, _ = best_scale(ax_capl, valid_capl, RELAXED_MASK)

print("  3-hop: country → capital → language → ???")
print("  (applying country→capital axis once more from language position)")
test_cases = [
    ('france', 'Paris', 'French'),
    ('germany', 'Berlin', 'German'),
    ('japan', 'Tokyo', 'Japanese'),
    ('china', 'Beijing', 'Chinese'),
    ('russia', 'Moscow', 'Russian'),
]
for country, capital, language in test_cases:
    es, sid = get_emb(country)
    if es is None: continue
    # Step 1: country -> capital
    r1 = nn_retrieve(W_E[sid]+best_s_cc*ax_cc, source_ids(country), RELAXED_MASK, 1)
    city = r1[0][0]
    # Step 2: capital -> language
    ec, cid = get_emb(city)
    if ec is None: print("  %-10s -> %-10s [multi-token] -> n/a" % (country, city)); continue
    r2 = nn_retrieve(W_E[cid]+best_s_capl*ax_capl, source_ids(city), RELAXED_MASK, 1)
    lang = r2[0][0]
    # Step 3: language -> ??? (apply cc axis again)
    el, lid = get_emb(lang)
    if el is None: print("  %-10s -> %-10s -> %-12s -> [multi-token]" % (country, city, lang)); continue
    r3 = nn_retrieve(W_E[lid]+best_s_cc*ax_cc, source_ids(lang), RELAXED_MASK, 3)
    r3_capl = nn_retrieve(W_E[lid]+best_s_capl*ax_capl, source_ids(lang), RELAXED_MASK, 3)
    print("  %-10s -> %-10s -> %-12s -> cc: %s | capl: %s" %
          (country, city, lang,
           ', '.join(w for w,_,_ in r3[:2]),
           ', '.join(w for w,_,_ in r3_capl[:2])))
print()

# =====================================================================
# PART E: PCA OF AXIS VECTORS
# =====================================================================
print("PART E: PCA of axis vectors — 2D visualization")
print("-"*72)

# Stack all axis vectors into a matrix
all_ax_names = list(axes.keys())
# Also include antonym axes
for ant_name, pairs in ANTONYM_SETS.items():
    ax, _, v, pc = compute_axis(pairs)
    if ax is not None:
        axes[ant_name] = ax
        pcs[ant_name] = pc
        all_ax_names.append(ant_name)

ax_matrix = np.array([axes[n].astype(np.float32) for n in all_ax_names])  # (n_axes, dim)
print("  Axis matrix shape: %s" % str(ax_matrix.shape))

# PCA
cov = ax_matrix @ ax_matrix.T  # (n_axes, n_axes) -- work in axis space
U, S, Vt = np.linalg.svd(cov)
print("  Top singular values of axis covariance: %s" % str([float('%.3f'%s) for s in S[:6]]))

# Project each axis onto top-2 PCs (= top-2 left singular vectors of ax_matrix)
U_ax, S_ax, Vt_ax = np.linalg.svd(ax_matrix, full_matrices=False)
# U_ax: (n_axes, n_axes), S_ax: (n_axes,), Vt_ax: (n_axes, dim)
coords_2d = U_ax[:, :2] * S_ax[:2]  # (n_axes, 2)
print("  Variance explained by top-2 PCs: %.1f%%" % (100*S_ax[:2].sum()/S_ax.sum()))
print()
print("  2D axis positions (PC1, PC2):")
for i, name in enumerate(all_ax_names):
    x, y = coords_2d[i]
    print("  %-16s  PC1=%+.4f  PC2=%+.4f  pc=%.3f" % (name, x, y, pcs.get(name, 0)))
print()

# Group analysis
relational = [n for n in all_ax_names if n in ('cc','cl','capl','pres')]
morphol    = [n for n in all_ax_names if n in ('+er','+s','+ed','+tion','un-')]
antonym    = [n for n in all_ax_names if n in ANTONYM_SETS]

for group_name, group in [('relational', relational), ('morphological', morphol), ('antonym', antonym)]:
    if not group: continue
    idxs = [all_ax_names.index(n) for n in group if n in all_ax_names]
    if not idxs: continue
    mean_pc1 = np.mean([coords_2d[i][0] for i in idxs])
    mean_pc2 = np.mean([coords_2d[i][1] for i in idxs])
    print("  %-16s  mean_PC1=%+.4f  mean_PC2=%+.4f" % (group_name, mean_pc1, mean_pc2))
