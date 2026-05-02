import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and len(w) > 1 and not w.startswith('-') and not w.startswith('_'):
        RELAXED_MASK[i] = True

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

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    return tok.decode([int(np.argmax(sims))]).strip()

def nn_ret_with_id(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    i = int(np.argmax(sims))
    return tok.decode([i]).strip(), i

def build_axis(pairs):
    """Compute mean offset direction and per-pair details."""
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es)
        valid.append((s, t, sid, tid, et - es))
    if not chords: return None, []
    mean_dir = normed(np.mean(chords, axis=0))
    return mean_dir, valid

def best_scale_for_axis(ax_dir, valid, mask):
    """Find the scale that maximises training retrieval."""
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = sum(1 for _,t,sid,_,_ in valid
                if nn_ret(W_E[sid] + s*ax_dir, source_ids(tok.decode([sid]).strip()), mask) == t)
        if c > best_a: best_a=c; best_s=s
    return best_s, best_a

# ============================================================
# AXIS DEFINITIONS
# ============================================================

GENDER = [('king','queen'),('man','woman'),('boy','girl'),
          ('father','mother'),('son','daughter'),('husband','wife'),
          ('uncle','aunt'),('prince','princess'),('actor','actress'),
          ('waiter','waitress')]

ER_COMP = [('big','bigger'),('fast','faster'),('tall','taller'),
           ('clean','cleaner'),('bright','brighter'),('warm','warmer'),
           ('long','longer'),('cold','colder'),('old','older'),
           ('smart','smarter'),('strong','stronger'),('light','lighter')]

ER_SUP  = [('big','biggest'),('fast','fastest'),('tall','tallest'),
           ('clean','cleanest'),('bright','brightest'),('warm','warmest'),
           ('long','longest'),('cold','coldest'),('old','oldest'),
           ('smart','smartest'),('strong','strongest'),('light','lightest')]

PLURAL  = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
           ('tree','trees'),('book','books'),('bird','birds'),('door','doors'),
           ('hand','hands'),('arm','arms'),('eye','eyes'),('leg','legs')]

UN_NEG  = [('happy','unhappy'),('clear','unclear'),('fair','unfair'),
           ('likely','unlikely'),('known','unknown'),('safe','unsafe'),
           ('usual','unusual'),('equal','unequal'),('stable','unstable'),
           ('real','unreal'),('true','untrue'),('lock','unlock')]

EN_FR   = [('house','maison'),('water','eau'),('sun','soleil'),('book','livre'),
           ('day','jour'),('night','nuit'),('cat','chat'),('dog','chien'),
           ('fire','feu'),('moon','lune'),('sea','mer')]

# ============================================================
# COMPOSITION TESTS — what we're testing
# ============================================================
# Test philosophy: three ways to compose axis A then axis B
#
#  (1) CHAIN:  apply A scaled, take nn_ret to get intermediate word,
#              then apply B scaled from that word's embedding.
#              Tests: does stepping through the vocabulary work?
#
#  (2) DIRECT: apply A_dir * s_A + B_dir * s_B in one step (no intermediate nn_ret).
#              Tests: are axis offsets geometrically additive?
#
#  (3) AXIS:   compute mean(target_B - source_A) directly as a new "composed axis"
#              and test it via LOO on held-out pairs.
#              Tests: does the composed axis generalise?
#
# For each test we measure:
#   - hit rate on training pairs
#   - hit rate on held-out pairs (not used to build either constituent axis)
# ============================================================

def compose_chain(src_word, ax_A, s_A, ax_B, s_B, mask):
    """Two-step chain: src →[A]→ intermediate(nn) →[B]→ target"""
    es, sid = get_emb(src_word)
    if es is None: return None, None
    mid_emb = es + s_A * ax_A
    mid_word, mid_id = nn_ret_with_id(mid_emb, source_ids(src_word), mask)
    mid_raw = W_E[mid_id].copy()
    final_word = nn_ret(mid_raw + s_B * ax_B, source_ids(mid_word) | source_ids(src_word), mask)
    return mid_word, final_word

def compose_direct(src_word, ax_A, s_A, ax_B, s_B, mask):
    """Single step: src →[A+B]→ target (no intermediate snap)"""
    es, sid = get_emb(src_word)
    if es is None: return None
    return nn_ret(es + s_A * ax_A + s_B * ax_B, source_ids(src_word), mask)

def test_composition_pair(label, pairs_AB, ax_A, s_A, ax_B, s_B):
    """
    pairs_AB: list of (source, expected_final_target) tuples.
    ax_A, s_A: first axis direction and scale.
    ax_B, s_B: second axis direction and scale (applied from intermediate).
    """
    chain_hits = 0; direct_hits = 0; n = 0
    rows = []
    for src, tgt in pairs_AB:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        mid, chain_result = compose_chain(src, ax_A, s_A, ax_B, s_B, RELAXED_MASK)
        direct_result     = compose_direct(src, ax_A, s_A, ax_B, s_B, RELAXED_MASK)
        ch_ok  = (chain_result == tgt)
        di_ok  = (direct_result == tgt)
        if ch_ok:  chain_hits  += 1
        if di_ok:  direct_hits += 1
        rows.append((src, mid, tgt, chain_result, direct_result, ch_ok, di_ok))
    return rows, chain_hits, direct_hits, n

def print_composition_result(label, rows, chain_hits, direct_hits, n):
    print("  %-28s  chain %d/%d=%.0f%%  direct %d/%d=%.0f%%" % (
        label, chain_hits, n, 100*chain_hits/max(n,1),
        direct_hits, n, 100*direct_hits/max(n,1)))
    for src, mid, tgt, ch_r, di_r, ch_ok, di_ok in rows:
        ch_sym = '✓' if ch_ok else '✗'
        di_sym = '✓' if di_ok else '✗'
        print("    %-8s →[chain]→ %-8s → %-8s %s  |  [direct]→ %-8s %s  (expected: %s)" % (
            src, mid or '?', ch_r or '?', ch_sym, di_r or '?', di_sym, tgt))

# ============================================================
# PHASE 1: Build all axes
# ============================================================
print("\nDAY 348: Axis Composition — chaining geometric offsets")
print("="*70)

print("\nPhase 1: Building axes...", flush=True)
gender_dir, gender_valid = build_axis(GENDER)
comp_dir,   comp_valid   = build_axis(ER_COMP)
sup_dir,    sup_valid    = build_axis(ER_SUP)
plural_dir, plural_valid = build_axis(PLURAL)
un_dir,     un_valid     = build_axis(UN_NEG)
fr_dir,     fr_valid     = build_axis(EN_FR)

s_gender, a_gender = best_scale_for_axis(gender_dir, gender_valid, RELAXED_MASK)
s_comp,   a_comp   = best_scale_for_axis(comp_dir,   comp_valid,   RELAXED_MASK)
s_sup,    a_sup    = best_scale_for_axis(sup_dir,     sup_valid,    RELAXED_MASK)
s_plural, a_plural = best_scale_for_axis(plural_dir, plural_valid, RELAXED_MASK)
s_un,     a_un     = best_scale_for_axis(un_dir,     un_valid,     RELAXED_MASK)

print("  Axis         scale    train_acc")
print("  gender       %.2f     %d/%d" % (s_gender, a_gender, len(gender_valid)))
print("  er_comp      %.2f     %d/%d" % (s_comp,   a_comp,   len(comp_valid)))
print("  er_sup       %.2f     %d/%d" % (s_sup,    a_sup,    len(sup_valid)))
print("  plural       %.2f     %d/%d" % (s_plural, a_plural, len(plural_valid)))
print("  un_neg       %.2f     %d/%d" % (s_un,     a_un,     len(un_valid)))

# ============================================================
# PHASE 2: Axis angle matrix — how similar are axes to each other?
# ============================================================
print("\nPhase 2: Axis cosine similarity matrix")
axes = [('gender', gender_dir), ('er_comp', comp_dir), ('er_sup', sup_dir),
        ('plural', plural_dir), ('un_neg', un_dir), ('en_fr', fr_dir)]
header = "  %8s" + "  %8s" * len(axes)
print(header % tuple([''] + [a[0] for a in axes]))
for n1, d1 in axes:
    row = "  %8s" % n1
    for n2, d2 in axes:
        c = np.dot(d1.astype(np.float32), d2.astype(np.float32))
        row += "  %+8.3f" % c
    print(row)

# Also check: is er_sup ≈ er_comp + (comp→sup delta)?
comp_to_sup_delta = normed(np.mean([W_E[tid]-W_E[sid]
    for _,_,sid,tid,_ in comp_valid[:8]], axis=0) if comp_valid else [])

# ============================================================
# PHASE 3: The key composition tests
# ============================================================
print("\nPhase 3: Composition tests")
print("  Questions: (A) does chaining two axis steps work?")
print("             (B) is direct summing (no intermediate snap) equivalent?")
print("             (C) which composition mode is more accurate?\n")

# --- Test 1: COMP then SUP-from-COMP (base → comparative → superlative)
# The er_sup axis is base→superlative.
# Delta(comp→sup) = er_sup_dir * s_sup - er_comp_dir * s_comp (as offset)
# We test: source +[comp]→ comparative +[sup-from-comp]→ superlative
# "sup-from-comp" direction = er_sup_dir * s_sup - er_comp_dir * s_comp (as direction)
# But for CHAIN we need the direction FROM the comparative, not from the base.
# Direction from comparative to superlative = Δ_sup - Δ_comp (as vectors)
# Use s_sup for scale of this step.
comp_to_sup_dir = normed(sup_dir * s_sup - comp_dir * s_comp)
s_comp_to_sup = np.linalg.norm(sup_dir * s_sup - comp_dir * s_comp)

COMP_SUP_PAIRS = [
    ('big','biggest'), ('fast','fastest'), ('tall','tallest'),
    ('clean','cleanest'), ('warm','warmest'), ('long','longest'),
    ('old','oldest'), ('smart','smartest'), ('strong','strongest'),
    ('cold','coldest'), ('bright','brightest'), ('light','lightest'),
]
print("  Test 1: er_comp → (comp-to-sup)")
rows1, ch1, di1, n1 = test_composition_pair(
    'comp→sup', COMP_SUP_PAIRS,
    comp_dir, s_comp,
    comp_to_sup_dir, s_comp_to_sup)
print_composition_result('comp → sup (via comparative)', rows1, ch1, di1, n1)

# --- Test 2: GENDER then PLURAL (man → women, king → queens)
GENDER_PLURAL_PAIRS = [
    ('man','women'), ('king','queens'), ('boy','girls'),
    ('father','mothers'), ('son','daughters'), ('husband','wives'),
    ('uncle','aunts'), ('prince','princesses'),
    ('actor','actresses'), ('waiter','waitresses'),
]
print("\n  Test 2: gender → plural")
rows2, ch2, di2, n2 = test_composition_pair(
    'gender+plural', GENDER_PLURAL_PAIRS,
    gender_dir, s_gender,
    plural_dir, s_plural)
print_composition_result('gender → plural', rows2, ch2, di2, n2)

# Also try reversed order: plural then gender
PLURAL_GENDER_PAIRS = [(s,t) for s,t in GENDER_PLURAL_PAIRS]
# Build source as plural first: "men" → "women", "kings" → "queens", etc.
# But these aren't the same pairs. Let's test: does order matter?
# Test: king +[plural]→ kings +[gender]→ queens
PLURAL_THEN_GENDER = [
    ('king','queens'), ('man','women'), ('boy','girls'),
    ('father','mothers'), ('son','daughters'),
]
print("\n  Test 2b: plural → gender (reversed order)")
rows2b, ch2b, di2b, n2b = test_composition_pair(
    'plural+gender', PLURAL_THEN_GENDER,
    plural_dir, s_plural,
    gender_dir, s_gender)
print_composition_result('plural → gender (reversed)', rows2b, ch2b, di2b, n2b)

# --- Test 3: UN_NEG then ER_COMP (happy → unhappier)
UN_COMP_PAIRS = [
    ('happy','unhappier'), ('clear','unclearer'), ('fair','unfairer'),
    ('safe','unsafer'), ('usual','unusualer'),
    ('equal','unequaler'), ('real','unrealer'),
]
# Note: some of these (unclearer, unsafer) are unusual but grammatically valid
print("\n  Test 3: un_neg → er_comp (happy → unhappy → unhappier)")
rows3, ch3, di3, n3 = test_composition_pair(
    'un+comp', UN_COMP_PAIRS,
    un_dir, s_un,
    comp_dir, s_comp)
print_composition_result('un_neg → er_comp', rows3, ch3, di3, n3)

# --- Test 4: SELF-COMPOSITION — apply same axis twice
# If axes are truly geometric, applying er_comp twice should overshoot or produce something meaningful
DOUBLE_COMP_PAIRS = [
    ('big','biggest'), ('fast','fastest'), ('tall','tallest'),
    ('old','oldest'), ('smart','smartest'),
]
print("\n  Test 4: er_comp applied TWICE (does double-step = superlative?)")
rows4, ch4, di4, n4 = test_composition_pair(
    'comp+comp', DOUBLE_COMP_PAIRS,
    comp_dir, s_comp,
    comp_dir, s_comp)
print_composition_result('er_comp × 2', rows4, ch4, di4, n4)

# ============================================================
# PHASE 4: Composed axis as a new geometric object
# ============================================================
print("\nPhase 4: Composed axis as first-class geometric object")
print("  Build Δ_gender+plural by averaging (female_plural - male_sing) chords")
print("  Test LOO generalization on held-out pairs not in either constituent axis\n")

# Build composed axis directly: source=singular_male, target=plural_female
COMPOSED_TRAIN = [
    ('king','queens'), ('man','women'), ('boy','girls'),
    ('father','mothers'), ('son','daughters'), ('husband','wives'),
    ('uncle','aunts'), ('prince','princesses'),
]
COMPOSED_HOLD = [
    ('actor','actresses'), ('waiter','waitresses'), ('lion','lionesses'),
    ('god','goddesses'), ('hero','heroines'),
]

composed_dir_gp, composed_valid_gp = build_axis(COMPOSED_TRAIN)
s_comp_gp, a_comp_gp = best_scale_for_axis(composed_dir_gp, composed_valid_gp, RELAXED_MASK)

# LOO on training
loo_hits = 0
for i in range(len(COMPOSED_TRAIN)):
    tv = [COMPOSED_TRAIN[j] for j in range(len(COMPOSED_TRAIN)) if j != i]
    cv, vv = build_axis(tv)
    sv, _ = best_scale_for_axis(cv, vv, RELAXED_MASK)
    s, t = COMPOSED_TRAIN[i]
    es, sid = get_emb(s)
    if es is None: continue
    pred = nn_ret(es + sv * cv, source_ids(s), RELAXED_MASK)
    if pred == t: loo_hits += 1

print("  Composed gender+plural axis:")
print("    Training pairs: %d  best_scale=%.2f  train_acc=%d/%d=%.0f%%" % (
    len(COMPOSED_TRAIN), s_comp_gp, a_comp_gp,
    len(COMPOSED_TRAIN), 100*a_comp_gp/max(len(COMPOSED_TRAIN),1)))
print("    LOO accuracy: %d/%d = %.0f%%" % (
    loo_hits, len(COMPOSED_TRAIN), 100*loo_hits/max(len(COMPOSED_TRAIN),1)))

# Held-out test
ho_hits = 0
for s, t in COMPOSED_HOLD:
    es, sid = get_emb(s)
    if es is None: continue
    pred = nn_ret(es + s_comp_gp * composed_dir_gp, source_ids(s), RELAXED_MASK)
    ok = (pred == t)
    if ok: ho_hits += 1
    print("    holdout: %-10s → %-12s  expected=%-12s  %s" % (
        s, pred, t, '✓' if ok else '✗'))

print("    Held-out: %d/%d = %.0f%%" % (ho_hits, len(COMPOSED_HOLD), 100*ho_hits/max(len(COMPOSED_HOLD),1)))

# Compare composed direction to sum of individual directions
composed_sum = normed(gender_dir * s_gender + plural_dir * s_plural)
cos_align = np.dot(composed_dir_gp.astype(np.float32), composed_sum.astype(np.float32))
print("\n  Alignment: composed_direct vs (gender_dir×s_g + plural_dir×s_p) normalized")
print("    cos(composed, sum) = %.4f" % cos_align)
print("  (1.0 = perfect: direct chord average IS the sum of parts)")
print("  (< 1.0 = interaction effect: composition is not purely additive)")

# ============================================================
# PHASE 5: Scale analysis — how do composition scales relate?
# ============================================================
print("\nPhase 5: Scale analysis")
print("  How does the composed axis scale compare to constituent scales?")

PHI = (1 + np.sqrt(5)) / 2

print("  --- gender + plural ---")
print("  gender scale:        %.3f" % s_gender)
print("  plural scale:        %.3f" % s_plural)
print("  s_g + s_p            %.3f  (linear sum)" % (s_gender + s_plural))
print("  sqrt(s_g^2 + s_p^2)  %.3f  (vector norm)" % np.sqrt(s_gender**2 + s_plural**2))
print("  phi * max(s_g, s_p)  %.3f  (phi-scaled)" % (PHI * max(s_gender, s_plural)))
print("  composed direct-chord scale: %.3f" % s_comp_gp)

print("\n  --- er_comp + comp-to-sup ---")
print("  er_comp scale:       %.3f" % s_comp)
print("  comp-to-sup scale:   %.3f" % s_comp_to_sup)
print("  s_c + s_cs           %.3f  (linear sum)" % (s_comp + s_comp_to_sup))
print("  sqrt(s_c^2+s_cs^2)   %.3f  (vector norm)" % np.sqrt(s_comp**2 + s_comp_to_sup**2))
print("  phi * max            %.3f  (phi-scaled)" % (PHI * max(s_comp, s_comp_to_sup)))

print("\n  --- un_neg + er_comp (token check for targets) ---")
for s, t in UN_COMP_PAIRS:
    tc = len(tok(' '+t, add_special_tokens=False)['input_ids'])
    print("  target '%-14s' token_count=%d  %s" % (
        t, tc, '' if tc == 1 else '<-- MULTI-TOKEN: vocabulary gap, not geometry failure'))

# ============================================================
# PHASE 6: Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Axis Composition Day 348")
print("="*70)
print("  Composition mode A (chain, snap at intermediate):")
print("    comp→sup:     %d/%d = %.0f%%" % (ch1, n1, 100*ch1/max(n1,1)))
print("    gender+plural: %d/%d = %.0f%%" % (ch2, n2, 100*ch2/max(n2,1)))
print("    un_neg+comp:  %d/%d = %.0f%%" % (ch3, n3, 100*ch3/max(n3,1)))
print("    comp×2:       %d/%d = %.0f%%" % (ch4, n4, 100*ch4/max(n4,1)))
print("")
print("  Composition mode B (direct sum, one nn_ret step):")
print("    comp→sup:     %d/%d = %.0f%%" % (di1, n1, 100*di1/max(n1,1)))
print("    gender+plural: %d/%d = %.0f%%" % (di2, n2, 100*di2/max(n2,1)))
print("    un_neg+comp:  %d/%d = %.0f%%" % (di3, n3, 100*di3/max(n3,1)))
print("    comp×2:       %d/%d = %.0f%%" % (di4, n4, 100*di4/max(n4,1)))
print("")
print("  Composed axis (direct chord) LOO:    %d/%d = %.0f%%" % (
    loo_hits, len(COMPOSED_TRAIN), 100*loo_hits/max(len(COMPOSED_TRAIN),1)))
print("  Composed axis holdout (unseen pairs): %d/%d = %.0f%%" % (
    ho_hits, len(COMPOSED_HOLD), 100*ho_hits/max(len(COMPOSED_HOLD),1)))
print("")
print("  Axis addivity: cos(composed_direct, sum_of_parts) = %.4f" % cos_align)
print("  Order matters? chain(g→p) vs chain(p→g):")
print("    gender→plural: %d/%d | plural→gender: %d/%d" % (ch2, n2, ch2b, n2b))
