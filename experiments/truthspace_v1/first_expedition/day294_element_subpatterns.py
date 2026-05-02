import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

def normed(v): return v/(np.linalg.norm(v)+1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None
def compute_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, 0.0, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    coh = float(np.mean([np.dot(c, md.astype(np.float32)) for c in cn]))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, coh, valid, pc
def nn_retrieve(pred_emb, exclude_ids, top_n=3):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=80):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def eval_pairs(axis, scale, pairs):
    results = []
    for s, t in pairs:
        es, sid = get_emb(s)
        if es is None: results.append((s, t, None, '?', False)); continue
        r = nn_retrieve(W_E[sid]+scale*axis, [sid])
        got = r[0][0] if r else '?'
        results.append((s, t, sid, got, got==t))
    return results
def chord_pairwise_report(pairs, label):
    chords = []
    valid_pairs = []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(normed(et-es).astype(np.float32))
        valid_pairs.append((s, t, sid, tid))
    if len(chords) < 2: return
    sims = [float(np.dot(chords[i], chords[j]))
            for i in range(len(chords)) for j in range(i+1, len(chords))]
    md = normed(np.mean([c for c in chords], axis=0))
    coh = float(np.mean([np.dot(c, md.astype(np.float32)) for c in chords]))
    print("  %-30s  n=%2d  pc=%.4f  coh=%.4f" % (label, len(chords), np.mean(sims), coh))

print("DAY 294: ELEMENT->SYMBOL SUB-PATTERN ANALYSIS")
print("="*65)
print("Parallel to Day 290 past-tense sub-pattern split.")
print("Hypothesis: alphabetic sub-axis has higher pc and generalises")
print("better than the combined axis (pc=0.1394).")
print()

# ====================================================================
# DEFINE SUB-PATTERNS
# ====================================================================

# Rule A: Symbol = first letter of English name
SINGLE_LETTER = [
    ('hydrogen','H'),('carbon','C'),('nitrogen','N'),('oxygen','O'),
    ('sulfur','S'),('potassium','K'),('fluorine','F'),('iodine','I'),
    ('uranium','U'),('boron','B'),('vanadium','V'),('tungsten','W'),
    ('yttrium','Y'),('phosphorus','P'),
]

# Rule B: Symbol = first two letters of English name
DOUBLE_LETTER = [
    ('helium','He'),('lithium','Li'),('calcium','Ca'),('cobalt','Co'),
    ('copper','Cu'),('silicon','Si'),('aluminum','Al'),('magnesium','Mg'),
    ('chlorine','Cl'),('chromium','Cr'),('neon','Ne'),('argon','Ar'),
    ('nickel','Ni'),('titanium','Ti'),('manganese','Mn'),('barium','Ba'),
    ('beryllium','Be'),
]

# Rule C: Latin-derived (symbol from Latin name, no English relation)
LATIN_DERIVED = [
    ('iron','Fe'),('gold','Au'),('silver','Ag'),('lead','Pb'),
    ('tin','Sn'),('sodium','Na'),('potassium','K'),('tungsten','W'),
    ('mercury','Hg'),('antimony','Sb'),
]
# Note: sodium->Na and potassium->K are odd: potassium is from 'kalium',
# sodium is from 'natrium' — these are BOTH Latin and common knowledge.

# Latin-derived excluding the common Na/K which are well-known
LATIN_PURE = [
    ('iron','Fe'),('gold','Au'),('silver','Ag'),('lead','Pb'),
    ('tin','Sn'),('mercury','Hg'),('antimony','Sb'),
]

# ====================================================================
# PART A: PAIRWISE COSINE FOR EACH SUB-PATTERN
# ====================================================================
print("PART A: Sub-pattern linearity comparison")
print("-"*65)
chord_pairwise_report(SINGLE_LETTER, "single-letter (C, N, O, S...)")
chord_pairwise_report(DOUBLE_LETTER, "double-letter (He, Li, Ca...)")
chord_pairwise_report(LATIN_DERIVED, "latin-derived (Fe, Au, Ag...)")
chord_pairwise_report(LATIN_PURE,    "latin-pure (Fe,Au,Ag,Pb,Sn)")
all_available = [p for label, pairs in [
    ('s', SINGLE_LETTER), ('d', DOUBLE_LETTER), ('l', LATIN_PURE)
] for p in pairs]
chord_pairwise_report(all_available, "combined")
print()

# ====================================================================
# PART B: TRAIN/HOLDOUT FOR EACH SUB-AXIS
# ====================================================================
print("PART B: Sub-axis train/holdout generalisation")
print("-"*65)

for label, pairs in [
    ("single-letter", SINGLE_LETTER),
    ("double-letter", DOUBLE_LETTER),
    ("latin-derived", LATIN_PURE),
]:
    if len(pairs) < 4:
        print("  %-16s SKIP (too few)" % label); continue
    n_tr = max(2, len(pairs)*2//3)
    train = pairs[:n_tr]; hold = pairs[n_tr:]
    ax, coh, valid, pc = compute_axis(train)
    if ax is None or not hold: continue
    s_opt, acc_tr = best_scale(ax, valid)
    hold_r = eval_pairs(ax, s_opt, hold)
    acc_h = sum(1 for _,_,sid,_,hit in hold_r if hit and sid is not None)
    n_h   = sum(1 for _,_,sid,_,_ in hold_r if sid is not None)
    print("  %-16s  pc=%.4f  train=%d/%d  holdout=%d/%d (%.0f%%)  scale=%.2f" % (
        label, pc, acc_tr, len(valid), acc_h, n_h, 100*acc_h/max(1,n_h), s_opt))
    for s, t, sid, got, hit in hold_r:
        if sid is None: continue
        print("    %-12s -> %-6s  got=%-8s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    print()

# ====================================================================
# PART C: INTER-SUB-AXIS COSINE (are they pointing same direction?)
# ====================================================================
print("PART C: Inter-sub-axis cosine")
print("-"*65)

ax_s, _, _, _ = compute_axis(SINGLE_LETTER)
ax_d, _, _, _ = compute_axis(DOUBLE_LETTER)
ax_l, _, _, _ = compute_axis(LATIN_PURE)
ax_all, _, _, _ = compute_axis(all_available)

axes_named = [('single', ax_s), ('double', ax_d), ('latin', ax_l), ('combined', ax_all)]
for i, (n1, a1) in enumerate(axes_named):
    for j, (n2, a2) in enumerate(axes_named):
        if j <= i or a1 is None or a2 is None: continue
        sim = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
        print("  %-10s <-> %-10s  cos=%.4f" % (n1, n2, sim))
print()

# ====================================================================
# PART D: CROSS-PATTERN GENERALISATION
# Does single-letter axis predict double-letter symbols?
# Does combined axis predict within each sub-pattern?
# ====================================================================
print("PART D: Cross-sub-axis generalisation")
print("-"*65)

dl_available = [(s,t) for s,t in DOUBLE_LETTER
                if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
sl_available = [(s,t) for s,t in SINGLE_LETTER
                if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
lat_available = [(s,t) for s,t in LATIN_PURE
                 if get_emb(s)[0] is not None and get_emb(t)[0] is not None]

for train_label, train_pairs, test_label, test_pairs in [
    ("single-letter", sl_available, "double-letter",  dl_available),
    ("double-letter", dl_available, "single-letter",  sl_available),
    ("single-letter", sl_available, "latin-derived",  lat_available),
    ("combined",      all_available, "single-letter", sl_available[:4]),
    ("combined",      all_available, "double-letter", dl_available[:4]),
    ("combined",      all_available, "latin-derived", lat_available),
]:
    ax_t, _, valid_t, pc_t = compute_axis(train_pairs)
    if ax_t is None or not test_pairs: continue
    s_t, _ = best_scale(ax_t, valid_t)
    results = eval_pairs(ax_t, s_t, test_pairs)
    acc = sum(1 for _,_,sid,_,hit in results if hit and sid is not None)
    n   = sum(1 for _,_,sid,_,_ in results if sid is not None)
    print("  %-16s -> %-16s  acc=%d/%d (%.0f%%)" % (
        train_label, test_label, acc, n, 100*acc/max(1,n)))
print()

# ====================================================================
# PART E: FULL BEST-POSSIBLE SINGLE-LETTER AXIS (only clean pairs)
# ====================================================================
print("PART E: Clean single-letter sub-axis (full train)")
print("-"*65)

CLEAN_SL = [p for p in SINGLE_LETTER
            if get_emb(p[0])[0] is not None and get_emb(p[1])[0] is not None]
CLEAN_DL = [p for p in DOUBLE_LETTER
            if get_emb(p[0])[0] is not None and get_emb(p[1])[0] is not None]

ax_csl, coh_csl, valid_csl, pc_csl = compute_axis(CLEAN_SL)
ax_cdl, coh_cdl, valid_cdl, pc_cdl = compute_axis(CLEAN_DL)

for label, ax, coh, valid, pc in [
    ("single-letter", ax_csl, coh_csl, valid_csl, pc_csl),
    ("double-letter", ax_cdl, coh_cdl, valid_cdl, pc_cdl),
]:
    if ax is None: continue
    s_opt, acc = best_scale(ax, valid)
    print("  %-16s  pc=%.4f  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        label, pc, coh, s_opt, acc, len(valid), 100*acc/max(1,len(valid))))
    for s, t, sid, tid in valid:
        r = nn_retrieve(W_E[sid]+s_opt*ax, [sid])
        got = r[0][0] if r else '?'
        print("    %-12s -> %-6s  got=%-8s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()

# Sub-axis for Latin-derived
for label, pairs in [("latin-derived", LATIN_PURE)]:
    ax_l2, coh_l2, valid_l2, pc_l2 = compute_axis(pairs)
    if ax_l2 is None: continue
    s_l2, acc_l2 = best_scale(ax_l2, valid_l2)
    print("  %-16s  pc=%.4f  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        label, pc_l2, coh_l2, s_l2, acc_l2, len(valid_l2), 100*acc_l2/max(1,len(valid_l2))))
    for s, t, sid, tid in valid_l2:
        r = nn_retrieve(W_E[sid]+s_l2*ax_l2, [sid])
        got = r[0][0] if r else '?'
        print("    %-12s -> %-6s  got=%-8s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()

# ====================================================================
# PART F: LINEARITY SPECTRUM UPDATE
# Combine with previous results from Days 290-293
# ====================================================================
print("="*65)
print("UPDATED LINEARITY SPECTRUM (Days 290-294)")
print("="*65)
print()

all_axes = []
if ax_csl is not None:
    _, _, _, pc_s = compute_axis(CLEAN_SL)
    all_axes.append(("elem:single-letter", pc_s, "SEMANTIC"))
if ax_cdl is not None:
    _, _, _, pc_d = compute_axis(CLEAN_DL)
    all_axes.append(("elem:double-letter", pc_d, "SEMANTIC"))
ax_l3, _, _, pc_l3 = compute_axis(LATIN_PURE)
if ax_l3 is not None:
    all_axes.append(("elem:latin-derived", pc_l3, "SEMANTIC"))

# From previous days
PREV_RESULTS = [
    ("country->demonym", 0.563, "SEMANTIC"),
    ("country->lang",    0.474, "SEMANTIC*"),
    ("+est (sup)",       0.436, "MORPH"),
    ("+er (comp)",       0.393, "MORPH"),
    ("country->cap",     0.317, "SEMANTIC"),
    ("animal->class",    0.254, "SEMANTIC"),
    ("person->nat",      0.246, "SEMANTIC"),
    ("past_irr",         0.230, "MORPH"),
    ("gender",           0.213, "MORPH"),
    ("+ed (past_r)",     0.174, "MORPH"),
    ("element->sym",     0.139, "SEMANTIC"),
    ("+s plural",        0.155, "MORPH"),
    ("field->concept",   0.087, "SEMANTIC"),
    ("word->antonym",    0.020, "SEMANTIC"),
]
all_axes.extend(PREV_RESULTS)
all_axes.sort(key=lambda x: -x[1])

print("  %-28s  pc_cos   type" % "Axis")
print("  " + "-"*50)
for name, pc, atype in all_axes:
    mark = " *" if atype.endswith('*') else ""
    print("  %-28s  %.4f   %s%s" % (name, pc, atype.rstrip('*'), mark))
print()
print("  * = pc inflated by training data scope limitation")
print("  Sub-axes should rank higher/lower than their combined parent")
