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
    if not chords: return None, 0.0, valid
    md = normed(np.mean(chords, axis=0))
    return md, float(np.mean([np.dot(normed(c), md) for c in chords])), valid
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=4.0, n=50):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for src,tgt,sid,tid in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==tgt)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

# ====================================================================
# Build all primitive axes
# ====================================================================
GENDER = [('king','queen'),('man','woman'),('boy','girl'),('son','daughter'),
          ('brother','sister'),('father','mother'),('uncle','aunt'),
          ('prince','princess'),('hero','heroine'),('actor','actress'),
          ('waiter','waitress'),('god','goddess'),('duke','duchess')]
PLURAL = [('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),
          ('book','books'),('car','cars'),('hand','hands'),('eye','eyes'),
          ('word','words'),('day','days'),('year','years'),('house','houses'),
          ('arm','arms'),('leg','legs'),('door','doors'),('line','lines'),
          ('way','ways'),('part','parts'),('name','names'),('place','places'),
          ('king','kings'),('queen','queens'),('boy','boys'),('girl','girls'),
          ('man','men'),('woman','women'),('god','gods'),('son','sons')]
COMP = [('fast','faster'),('slow','slower'),('tall','taller'),('small','smaller'),
        ('large','larger'),('hard','harder'),('soft','softer'),('warm','warmer'),
        ('dark','darker'),('clean','cleaner'),('sharp','sharper'),('deep','deeper'),
        ('wide','wider'),('strong','stronger'),('long','longer'),('old','older')]
SUP  = [('fast','fastest'),('slow','slowest'),('tall','tallest'),('small','smallest'),
        ('large','largest'),('hard','hardest'),('soft','softest'),('warm','warmest'),
        ('dark','darkest'),('clean','cleanest'),('sharp','sharpest'),('deep','deepest'),
        ('wide','widest'),('strong','strongest'),('long','longest'),('old','oldest')]
PAST = [('walk','walked'),('talk','talked'),('work','worked'),('play','played'),
        ('call','called'),('feel','felt'),('run','ran'),('go','went'),
        ('get','got'),('say','said'),('make','made'),('take','took'),
        ('see','saw'),('know','knew'),('come','came'),('give','gave')]

ax_g,  coh_g,  vg  = compute_axis(GENDER)
ax_pl, coh_pl, vpl = compute_axis(PLURAL)
ax_c,  coh_c,  vc  = compute_axis(COMP)
ax_s,  coh_s,  vs  = compute_axis(SUP)
ax_p,  coh_p,  vp  = compute_axis(PAST)

sg, _  = best_scale(ax_g,  vg)
spl, _ = best_scale(ax_pl, vpl)
sc, _  = best_scale(ax_c,  vc)
ss, _  = best_scale(ax_s,  vs)
sp, _  = best_scale(ax_p,  vp)

print("DAY 288: SIMULTANEOUS MULTI-AXIS TRANSFORMATION")
print("="*65)
print("Testing whether two axes can be applied simultaneously.")
print("Classic word2vec: king - man + woman = queen.")
print("Our test: king + gender_axis + plural_axis = queens?")
print()

# ====================================================================
# PART A: INTER-AXIS ORTHOGONALITY
# ====================================================================
print("PART A: Inter-axis cosine similarities")
print("-"*65)
axes = [("gender",ax_g),("plural",ax_pl),("comp",ax_c),("sup",ax_s),("past",ax_p)]
print("       " + "  ".join("%-8s" % n for n,_ in axes))
for n1,a1 in axes:
    row = "  %-7s" % n1
    for n2,a2 in axes:
        sim = float(np.dot(a1.astype(np.float32),a2.astype(np.float32)))
        row += " %+.4f " % sim
    print(row)
print()
print("If axes are orthogonal (cos~0): applying both simultaneously")
print("should work without interference.")
print()

# ====================================================================
# PART B: SIMULTANEOUS GENDER + PLURAL (king -> queens)
# ====================================================================
print("PART B: Simultaneous gender + plural (masc_sing -> fem_plural)")
print("-"*65)
print("Target: king->queens, man->women, boy->girls, son->daughters")
print()

# Sequential: gender first, then plural
print("  Sequential (gender then plural):")
SEQ_TESTS = [('king','queens'),('man','women'),('boy','girls'),
             ('son','daughters'),('brother','sisters'),('god','goddesses')]
seq_correct = 0
for src, tgt in SEQ_TESTS:
    es, sid = get_emb(src)
    if es is None: print("  %-12s SKIP" % src); continue
    # Step 1: gender
    r1 = nn_retrieve(es + sg * ax_g, [sid])
    fem = r1[0][0] if r1 else None
    ef, fid = get_emb(fem) if fem else (None, None)
    if ef is None:
        print("  %-10s -> %-12s -> ? -> ?" % (src, fem if fem else '?'))
        continue
    # Step 2: plural
    r2 = nn_retrieve(ef + spl * ax_pl, [fid])
    got = r2[0][0] if r2 else None
    hit = (got == tgt)
    if hit: seq_correct += 1
    print("  %-10s -> %-12s -> %-12s [%s] exp=%s" % (src, fem, got if got else '?', 'HIT' if hit else '---', tgt))
print()

# Simultaneous: add both axes at once
print("  Simultaneous (gender + plural in one step):")
sim_correct = 0
# Try different scale combinations
best_sim_acc = 0; best_sg2 = sg; best_spl2 = spl
for sg2 in np.linspace(0.1, 1.0, 15):
    for spl2 in np.linspace(0.1, 1.5, 15):
        c = sum(1 for src,tgt in SEQ_TESTS
                if (get_emb(src)[0] is not None) and
                nn_retrieve(get_emb(src)[0] + sg2*ax_g + spl2*ax_pl,
                             [get_emb(src)[1]])[0][0] == tgt)
        if c > best_sim_acc:
            best_sim_acc = c; best_sg2 = sg2; best_spl2 = spl2

for src, tgt in SEQ_TESTS:
    es, sid = get_emb(src)
    if es is None: continue
    r = nn_retrieve(es + best_sg2*ax_g + best_spl2*ax_pl, [sid])
    got = r[0][0] if r else None
    hit = (got == tgt)
    if hit: sim_correct += 1
    print("  %-10s -> %-12s [%s] exp=%s  (scale_g=%.2f, scale_pl=%.2f)" % (
        src, got if got else '?', 'HIT' if hit else '---', tgt, best_sg2, best_spl2))
print()
print("  Sequential: %d/%d   Simultaneous: %d/%d" % (
    seq_correct, len(SEQ_TESTS), sim_correct, len(SEQ_TESTS)))
print()

# ====================================================================
# PART C: CLASSIC WORD2VEC ANALOGY (king - man + woman)
# ====================================================================
print("PART C: Classic word2vec analogy (axis-free version)")
print("-"*65)
print("  king - man + woman = ? (expected: queen)")
print()

def analogy(a, b, c, top_n=5):
    ea, aid = get_emb(a); eb, bid = get_emb(b); ec, cid = get_emb(c)
    if any(x is None for x in [ea,eb,ec]): return []
    pred = ea - eb + ec
    return nn_retrieve(pred, [aid, bid, cid], top_n=top_n)

ANALOGIES = [
    ('king','man','woman','queen'),
    ('kings','men','women','queens'),
    ('brother','man','woman','sister'),
    ('father','man','woman','mother'),
    ('sons','men','women','daughters'),
    ('actor','man','woman','actress'),
    ('god','man','woman','goddess'),
    ('faster','fast','slow','slower'),
    ('fastest','fast','slow','slowest'),
    ('walked','walk','run','ran'),
    ('said','say','go','went'),
    ('German','Germany','France','French'),
    ('english','Britain','France','french'),
    ('queen','woman','man','king'),
]
correct_analogy = 0
for a, b, c, exp in ANALOGIES:
    r = analogy(a, b, c)
    got = r[0][0] if r else '?'
    top3 = [x[0] for x in r[:3]] if r else []
    hit = (exp in top3)
    if hit: correct_analogy += 1
    print("  %-10s - %-10s + %-10s = %-12s  got=%-12s [%s]" % (
        a, b, c, exp, got, 'HIT' if got==exp else ('IN3' if hit else '---')))
print()
print("  Classic analogy accuracy: %d/%d top-1 (%.0f%%)" % (
    sum(1 for a,b,c,exp in ANALOGIES if (r:=analogy(a,b,c)) and r[0][0]==exp),
    len(ANALOGIES),
    100*sum(1 for a,b,c,exp in ANALOGIES if (r:=analogy(a,b,c)) and r[0][0]==exp)/max(1,len(ANALOGIES))))
print()

# ====================================================================
# PART D: AXIS-BASED vs ANALOGY-BASED: which is better?
# ====================================================================
print("PART D: Axis-based vs analogy-based comparison on same test set")
print("-"*65)
print("  %-26s  %-10s  %-10s  correct?" % ("test", "axis", "analogy"))
print("  " + "-"*52)

TEST_GENDER = [('king','queen'),('man','woman'),('boy','girl'),
               ('son','daughter'),('actor','actress'),('god','goddess')]
for src, tgt in TEST_GENDER:
    es, sid = get_emb(src)
    if es is None: continue
    # Axis method
    r_ax = nn_retrieve(es + sg * ax_g, [sid])
    got_ax = r_ax[0][0] if r_ax else '?'
    # Analogy method (using the most canonical pair: man/woman)
    r_an = analogy(src, 'man', 'woman')
    got_an = r_an[0][0] if r_an else '?'
    hit_ax = (got_ax == tgt); hit_an = (got_an == tgt)
    print("  %-12s -> %-12s  %-10s[%s]  %-10s[%s]" % (
        src, tgt, got_ax, 'HIT' if hit_ax else '---',
        got_an, 'HIT' if hit_an else '---'))
print()

# ====================================================================
# PART E: NOVEL AXIS FROM COMPOSITION
# Test: can d_past - d_comparative produce a meaningful new axis?
# "walked" - "faster" direction?
# ====================================================================
print("PART E: Novel axis from cross-domain subtraction (d_past - d_comp)")
print("-"*65)
print("Testing if d_past - d_comp produces any meaningful retrieval...")
print()

ax_novel = normed(ax_p - ax_c)
# What does this axis do to adjectives?
print("  Applied to BASE adjectives (should it give... walked-fast??):")
test_adj = ['fast','slow','tall','dark','bright']
for w in test_adj:
    e, wid = get_emb(w)
    if e is None: continue
    r = nn_retrieve(e + ax_novel, [wid])
    print("  %-10s -> %s" % (w, [x[0] for x in r[:4]]))
print()
print("  Applied to BASE verbs:")
test_verb = ['walk','run','go','say','make']
for w in test_verb:
    e, wid = get_emb(w)
    if e is None: continue
    r = nn_retrieve(e + ax_novel, [wid])
    print("  %-10s -> %s" % (w, [x[0] for x in r[:4]]))
print()

# ====================================================================
# PART F: AXIS INTERFERENCE MEASUREMENT
# How much does applying one axis disturb another?
# ====================================================================
print("PART F: Axis interference — does gender axis disturb plural?")
print("-"*65)

DOUBLE_TRANSFORM_TEST = [
    # (source, after_gender, after_plural)
    ('king',   'queen',    'queens'),
    ('man',    'woman',    'women'),
    ('boy',    'girl',     'girls'),
    ('dog',    None,       'dogs'),    # no gender, only plural
    ('cat',    None,       'cats'),
]

for src, gender_tgt, plural_tgt in DOUBLE_TRANSFORM_TEST:
    es, sid = get_emb(src)
    if es is None: continue

    # Single-axis plural
    r_pl = nn_retrieve(es + spl * ax_pl, [sid])
    got_pl = r_pl[0][0] if r_pl else '?'

    # Gender then plural
    if gender_tgt is not None:
        r_g = nn_retrieve(es + sg * ax_g, [sid])
        fem = r_g[0][0] if r_g else None
        ef, fid = get_emb(fem) if fem else (None, None)
        if ef is not None:
            r_gpl = nn_retrieve(ef + spl * ax_pl, [fid])
            got_gpl = r_gpl[0][0] if r_gpl else '?'
        else:
            got_gpl = '?'
    else:
        got_gpl = 'n/a'

    print("  %-6s  plural-only: %-10s  gender+plural: %-12s  exp=%s" % (
        src, got_pl, got_gpl, plural_tgt if plural_tgt else ''))
print()

# ====================================================================
# SUMMARY
# ====================================================================
print("MULTI-AXIS TRANSFORMATION SUMMARY:")
print("="*65)
print()
print("  Inter-axis orthogonality:")
for i, (n1, a1) in enumerate(axes):
    for j, (n2, a2) in enumerate(axes):
        if j <= i: continue
        sim = float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
        print("    cos(%s, %s) = %+.4f  %s" % (n1, n2, sim,
            "ORTHOGONAL" if abs(sim) < 0.15 else
            "WEAKLY_RELATED" if abs(sim) < 0.30 else "RELATED"))
print()
print("  Simultaneous transformation accuracy: %d/%d" % (sim_correct, len(SEQ_TESTS)))
print("  Sequential transformation accuracy:   %d/%d" % (seq_correct, len(SEQ_TESTS)))
