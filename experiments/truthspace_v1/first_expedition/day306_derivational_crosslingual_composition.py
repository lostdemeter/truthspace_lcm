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
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 306: DERIVATIONAL RESIDUAL, CROSS-LINGUAL GENDER, 12x12 COMPOSITION")
print("="*70)
print()

# ====================================================================
# Build full morphological chord matrix (same as Day 305)
# ====================================================================
ALL_MORPH = {
    '+er':      [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                 ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                 ('light','lighter'),('strong','stronger'),('weak','weaker'),('soft','softer')],
    '+est':     [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                 ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest'),
                 ('light','lightest'),('strong','strongest'),('weak','weakest')],
    'er->est':  [('faster','fastest'),('slower','slowest'),('taller','tallest'),('shorter','shortest'),
                 ('brighter','brightest'),('darker','darkest'),('deeper','deepest'),('cleaner','cleanest'),
                 ('lighter','lightest'),('stronger','strongest'),('weaker','weakest')],
    'gender':   [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                 ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
    'past_irr': [('go','went'),('come','came'),('run','ran'),('see','saw'),
                 ('eat','ate'),('know','knew'),('take','took'),('make','made')],
    '+ed':      [('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                 ('end','ended'),('look','looked'),('call','called'),('help','helped')],
    '+ness':    [('sad','sadness'),('happy','happiness'),('dark','darkness'),('kind','kindness'),
                 ('bright','brightness'),('fit','fitness'),('mad','madness'),('glad','gladness')],
    '+ful':     [('hope','hopeful'),('care','careful'),('use','useful'),('power','powerful'),
                 ('peace','peaceful'),('harm','harmful'),('thank','thankful'),('help','helpful')],
    'un-':      [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('known','unknown'),
                 ('usual','unusual'),('clear','unclear'),('lock','unlock'),('wrap','unwrap')],
    '+ment':    [('achieve','achievement'),('manage','management'),('develop','development'),
                 ('move','movement'),('treat','treatment'),('argue','argument')],
    '+s':       [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
    '+tion':    [('act','action'),('direct','direction'),('collect','collection'),
                 ('connect','connection'),('protect','protection'),('select','selection')],
}

# Precompute all axes and valid pairs
named_axes, named_valid, named_pcs = {}, {}, {}
for nm, pairs in ALL_MORPH.items():
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, vp, pc = compute_axis(avail)
    if ax is None: continue
    named_axes[nm] = ax.astype(np.float64)
    named_valid[nm] = vp
    named_pcs[nm] = pc

# Build chord matrix for PCA
all_chords, all_labels = [], []
for nm, pairs in ALL_MORPH.items():
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        all_chords.append(normed(et-es).astype(np.float32))
        all_labels.append((s, t, nm))

M_mat = np.array(all_chords)
M_mean = M_mat.mean(axis=0)
M_c = (M_mat - M_mean).astype(np.float32)
rng = np.random.default_rng(0)
M_pcs = []
M_deflated = M_c.copy()
for k in range(8):
    vk = rng.standard_normal(M_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(200):
        vk = M_deflated.T @ (M_deflated @ vk)
        vk /= np.linalg.norm(vk)
    proj = M_deflated @ vk
    M_deflated = M_deflated - np.outer(proj, vk)
    var_k = float(np.var(M_c @ vk))
    M_pcs.append((vk.astype(np.float64), var_k))

# ====================================================================
# PART A: mPC6-8 — DERIVATIONAL RESIDUAL
# ====================================================================
print("PART A: mPC6-8 token atlas (derivational residual)")
print("-"*70)

for k in range(5, 8):
    mpc = M_pcs[k][0].astype(np.float32)
    projs = W_n @ mpc
    top_pos = np.argsort(projs)[-15:][::-1]
    top_neg = np.argsort(projs)[:15]

    # Alignment with known axes
    aligns = []
    for nm, ax in named_axes.items():
        c = float(np.dot(ax, M_pcs[k][0]))
        aligns.append((abs(c), c, nm))
    aligns.sort(reverse=True)

    print("  mPC%d  axis alignments (|cos|>0.10):" % (k+1))
    for _, c, nm in aligns:
        if abs(c) > 0.10: print("    %-14s  %+.4f" % (nm, c))
    print("  mPC%d  top 10 tokens:" % (k+1))
    for tid in top_pos[:10]:
        w = tok.decode([tid]).strip()
        print("    cos=%+.4f  '%s'  (id=%d)" % (float(projs[tid]), w, tid))
    print("  mPC%d  bottom 10 tokens:" % (k+1))
    for tid in top_neg[:10]:
        w = tok.decode([tid]).strip()
        print("    cos=%+.4f  '%s'  (id=%d)" % (float(projs[tid]), w, tid))
    print()

# ====================================================================
# PART B: CROSS-LINGUAL GENDER mPC5 VERIFICATION
# ====================================================================
print("PART B: Cross-lingual gender — mPC5 projections by language")
print("-"*70)

mpc5 = M_pcs[4][0].astype(np.float32)

# Feminine tokens in multiple languages
FEMININE_TOKENS = {
    'English': ['woman','women','girl','girls','mother','daughter','sister',
                'aunt','wife','actress','queen','lady','female','bride','nun'],
    'Spanish': ['mujer','mujeres','chica','madre','hija','hermana','tia',
                'esposa','actriz','reina','dama','novia'],
    'French':  ['femme','fille','mère','soeur','tante','épouse','reine',
                'dame','fille','madame'],
    'German':  ['frau','mädchen','mutter','tochter','schwester','tante',
                'ehefrau','königin','dame','braut'],
    'Chinese': ['女','妇','母','女儿','姐','妹','姑','嫂','娘','婆'],
    'Japanese':['女','母','姉','妹','娘','嬢'],
}
MASCULINE_TOKENS = {
    'English': ['man','men','boy','boys','father','son','brother','uncle',
                'husband','actor','king','lord','male','groom','monk'],
    'Spanish': ['hombre','hombres','chico','padre','hijo','hermano','tio',
                'esposo','actor','rey','señor','novio'],
    'French':  ['homme','garçon','père','frère','oncle','époux','roi',
                'monsieur','mari'],
    'German':  ['mann','junge','vater','sohn','bruder','onkel','ehemann',
                'könig','herr','bräutigam'],
    'Chinese': ['男','父','子','兄','弟','叔','舅','郎','夫','爷'],
}

print("  %-12s  FEMININE  MASCULINE  gap    n_f  n_m" % "language")
print("  " + "-"*52)
lang_results = {}
for lang in FEMININE_TOKENS:
    f_scores, m_scores = [], []
    for w in FEMININE_TOKENS[lang]:
        e, _ = get_emb(w)
        if e is None: continue
        en = normed(e).astype(np.float32)
        f_scores.append(float(np.dot(en, mpc5)))
    for w in MASCULINE_TOKENS.get(lang, []):
        e, _ = get_emb(w)
        if e is None: continue
        en = normed(e).astype(np.float32)
        m_scores.append(float(np.dot(en, mpc5)))
    if f_scores and m_scores:
        f_mean = np.mean(f_scores); m_mean = np.mean(m_scores)
        gap = f_mean - m_mean
        lang_results[lang] = (f_mean, m_mean, gap)
        print("  %-12s  %+.4f    %+.4f     %+.4f  %-3d  %d" %
              (lang, f_mean, m_mean, gap, len(f_scores), len(m_scores)))

print()
print("  Per-word examples (English):")
for w in ['woman','man','girl','boy','mother','father','queen','king']:
    e, _ = get_emb(w)
    if e is None: continue
    en = normed(e).astype(np.float32)
    s = float(np.dot(en, mpc5))
    print("    %-12s  mPC5=%+.4f" % (w, s))
print()

# ====================================================================
# PART C: 12×12 COMPOSITION MATRIX
# ====================================================================
print("PART C: 12×12 axis composition pairwise cosine matrix")
print("-"*70)

AXIS_NAMES = list(named_axes.keys())
n_axes = len(AXIS_NAMES)

# For each pair (A, B): compose A+B, measure cos with each named axis
# Also apply to 3 test words and check if result is sensible

print("  Composition: cos(A+B, A) and cos(A+B, B) for all pairs")
print("  Format: A+B -> cos_A  cos_B  [dominant]")
print()

# Representative source words for each axis
AXIS_SOURCES = {
    '+er':      [('fast', 'faster'),  ('tall', 'taller')],
    '+est':     [('fast', 'fastest'), ('slow', 'slowest')],
    'er->est':  [('faster', 'fastest'), ('taller', 'tallest')],
    'gender':   [('king', 'queen'),   ('man', 'woman')],
    'past_irr': [('go', 'went'),      ('run', 'ran')],
    '+ed':      [('walk', 'walked'),  ('jump', 'jumped')],
    '+ness':    [('sad', 'sadness'),  ('happy', 'happiness')],
    '+ful':     [('hope', 'hopeful'), ('care', 'careful')],
    'un-':      [('happy', 'unhappy'),('kind', 'unkind')],
    '+ment':    [('achieve', 'achievement'), ('manage', 'management')],
    '+s':       [('cat', 'cats'),     ('dog', 'dogs')],
    '+tion':    [('act', 'action'),   ('connect', 'connection')],
}

# Build raw (unnormalised) mean axis vectors
def raw_axis_vec(nm):
    pairs = ALL_MORPH[nm]
    chords = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        chords.append(normed(et-es).astype(np.float64))
    return np.mean(chords, axis=0) if chords else None

raw_axes = {nm: raw_axis_vec(nm) for nm in AXIS_NAMES}

# Compute pairwise cosines
print("  Mutual cosine table (cosine between each pair of axes):")
print("  %-10s" % "" + "".join("  %-8s" % nm[:8] for nm in AXIS_NAMES))
print("  " + "-"*(10 + 10*n_axes))
for nm_a in AXIS_NAMES:
    row = "  %-10s" % nm_a
    for nm_b in AXIS_NAMES:
        if nm_a == nm_b:
            row += "  [  1.00]"
        else:
            c = float(np.dot(named_axes[nm_a], named_axes[nm_b]))
            row += "  %+.3f  " % c
    print(row)
print()

# ====================================================================
# PART D: COMPOSITION RETRIEVAL TEST (selected pairs)
# ====================================================================
print("PART D: Composition retrieval tests (selected pairs)")
print("-"*70)

INTERESTING_PAIRS = [
    ('+er', 'er->est', '+est'),   # should compose to +est (already known)
    ('+s',  'gender',  None),     # cats + gender -> ???
    ('gender', '+s',   None),     # woman + plural -> women (!)
    ('+ed', 'gender',  None),     # walked + gender -> ???
    ('un-',  '+ness',  None),     # un- + ness -> ???
    ('+er', '+s',      None),     # faster + plural -> ???
    ('past_irr', '+ness', None),  # went + ness -> ???
    ('+tion', '+s',    None),     # actions + ??? -> ???
]

for pair in INTERESTING_PAIRS:
    nm_a, nm_b = pair[0], pair[1]
    expected = pair[2]
    if nm_a not in raw_axes or nm_b not in raw_axes: continue
    ra, rb = raw_axes[nm_a], raw_axes[nm_b]
    if ra is None or rb is None: continue

    composed = normed(ra + rb)
    cos_a = float(np.dot(composed, named_axes[nm_a]))
    cos_b = float(np.dot(composed, named_axes[nm_b]))
    cos_exp = None
    if expected and expected in named_axes:
        cos_exp = float(np.dot(composed, named_axes[expected]))

    # Apply to source word from nm_a
    src_pairs = AXIS_SOURCES.get(nm_a, [])
    hits = []
    for src, tgt in src_pairs[:2]:
        es, sid = get_emb(src)
        if es is None: continue
        s_a, _ = best_scale(named_axes[nm_a].astype(np.float32), named_valid[nm_a],
                            lo=0.5, hi=4.0, n=40)
        s_b, _ = best_scale(named_axes[nm_b].astype(np.float32), named_valid[nm_b],
                            lo=0.5, hi=4.0, n=40)
        pred = W_E[sid] + s_a * ra + s_b * rb
        r = nn_retrieve(pred, [sid], top_n=3)
        hits.append("%s->%s" % (src, r[0][0]))

    msg = "cos_a=%+.3f  cos_b=%+.3f" % (cos_a, cos_b)
    if cos_exp is not None: msg += "  cos_%s=%+.3f" % (expected, cos_exp)
    print("  %-8s + %-8s:  %s" % (nm_a, nm_b, msg))
    if hits: print("    examples: %s" % "  |  ".join(hits))
print()

# ====================================================================
# PART E: ENCODE=DECODE MORPHOLOGICAL TEST
# ====================================================================
print("PART E: ENCODE=DECODE — can +s axis predict plural in context?")
print("-"*70)

# Test: given a word in context (its W_E embedding), add +s axis,
# check if the nearest neighbour is the plural form.
# Focus on +s since it's the most isolated (pure mPC4).

ax_s = named_axes['+s']
_, _, valid_s, _ = compute_axis(ALL_MORPH['+s'])
s_scale, s_acc = best_scale(ax_s.astype(np.float32), valid_s)
print("  +s axis: scale=%.3f  training acc=%d/%d" % (s_scale, s_acc, len(valid_s)))
print()

# Holdout test: words NOT in training pairs
HOLDOUT_S = [
    ('flower', 'flowers'), ('city', 'cities'), ('country', 'countries'),
    ('window', 'windows'), ('chair', 'chairs'), ('table', 'tables'),
    ('river', 'rivers'),   ('mountain', 'mountains'), ('ocean', 'oceans'),
    ('planet', 'planets'), ('star', 'stars'), ('cloud', 'clouds'),
    ('island', 'islands'), ('forest', 'forests'), ('street', 'streets'),
    ('train', 'trains'),   ('plane', 'planes'), ('boat', 'boats'),
]

hits, total = 0, 0
print("  %-12s  predicted      correct?" % "source")
for src, tgt in HOLDOUT_S:
    es, sid = get_emb(src)
    et, tid = get_emb(tgt)
    if es is None or et is None: continue
    pred = W_E[sid] + s_scale * ax_s
    r = nn_retrieve(pred, [sid], top_n=3)
    top_word = r[0][0]
    ok = (top_word == tgt)
    if ok: hits += 1
    total += 1
    print("  %-12s  %-14s %s  [%s]" % (src, top_word, '✓' if ok else '✗', tgt))

print()
print("  +s holdout accuracy: %d/%d (%.1f%%)" % (hits, total, 100*hits/total if total else 0))
print()

# Also test +er holdout
ax_er = named_axes['+er']
_, _, valid_er, _ = compute_axis(ALL_MORPH['+er'])
er_scale, er_acc = best_scale(ax_er.astype(np.float32), valid_er)
HOLDOUT_ER = [
    ('loud','louder'), ('quiet','quieter'), ('warm','warmer'),('cold','colder'),
    ('old','older'), ('young','younger'), ('new','newer'), ('cheap','cheaper'),
    ('rich','richer'), ('poor','poorer'), ('wide','wider'), ('narrow','narrower'),
]
hits_er, total_er = 0, 0
print("  +er holdout test:")
print("  %-12s  predicted      correct?" % "source")
for src, tgt in HOLDOUT_ER:
    es, sid = get_emb(src)
    et, tid = get_emb(tgt)
    if es is None or et is None: continue
    pred = W_E[sid] + er_scale * ax_er
    r = nn_retrieve(pred, [sid], top_n=3)
    top_word = r[0][0]
    ok = (top_word == tgt)
    if ok: hits_er += 1
    total_er += 1
    print("  %-12s  %-14s %s  [%s]" % (src, top_word, '✓' if ok else '✗', tgt))

print()
print("  +er holdout accuracy: %d/%d (%.1f%%)" % (hits_er, total_er, 100*hits_er/total_er if total_er else 0))
