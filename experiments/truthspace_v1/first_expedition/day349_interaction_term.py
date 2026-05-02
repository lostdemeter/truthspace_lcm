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

def build_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es)
        valid.append((s, t, sid, tid, et - es))
    if not chords: return None, []
    return normed(np.mean(chords, axis=0)), valid

def best_scale(ax_dir, valid, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = sum(1 for _,t,sid,_,_ in valid
                if nn_ret(W_E[sid] + s*ax_dir, source_ids(tok.decode([sid]).strip()), mask) == t)
        if c > best_a: best_a=c; best_s=s
    return best_s

def nearest_vocab(direction, n=12):
    d = normed(direction).astype(np.float32)
    sims = W_n @ d
    top_ids = np.argsort(sims)[::-1][:n*3]
    words = []
    for i in top_ids:
        w = tok.decode([i]).strip()
        if w and len(w) > 1 and w.isalpha(): words.append((w, float(sims[i])))
        if len(words) >= n: break
    return words

# ============================================================
# INTERACTION TERM ANALYSIS
# ============================================================
# For a composition "axis A then axis B":
#   chord_AB(src, tgt) = W_E[tgt] - W_E[src]  (direct vector: no steps through vocab)
#   sum_AB             = s_A * dir_A + s_B * dir_B  (naive sum prediction)
#   interaction(src,tgt) = chord_AB - sum_AB  (the unexplained residual)
#
# Questions:
#   Q1. Within crossing type: how consistent is interaction() across word pairs?
#       (measured by avg cos between each residual and the mean residual)
#   Q2. Across crossing types: do different family crossings produce different interaction directions?
#   Q3. Does adding mean_interaction as a correction improve direct composition?
#   Q4. What vocabulary region does the interaction vector point toward?
# ============================================================

def compute_interactions(pairs_AB, dir_A, s_A, dir_B, s_B):
    """
    Returns list of (src, tgt, chord, sum_pred, interaction, |interaction|/|chord|)
    for each valid pair where both src and tgt are single-token.
    """
    results = []
    sum_pred = s_A * dir_A + s_B * dir_B
    for src, tgt in pairs_AB:
        es, _ = get_emb(src)
        et, _ = get_emb(tgt)
        if es is None or et is None: continue
        chord = et - es
        interaction = chord - sum_pred
        rel_mag = np.linalg.norm(interaction) / (np.linalg.norm(chord) + 1e-8)
        results.append((src, tgt, chord, sum_pred, interaction, rel_mag))
    return results

def consistency_score(interactions_list):
    """
    Given a list of interaction vectors, compute:
    - mean direction
    - avg cos(each, mean)
    - std of cosines
    """
    vecs = [r[4] for r in interactions_list]
    if not vecs: return None, 0.0, 0.0
    mean_dir = normed(np.mean(vecs, axis=0))
    coss = [float(np.dot(normed(v).astype(np.float32), mean_dir.astype(np.float32))) for v in vecs]
    return mean_dir, float(np.mean(coss)), float(np.std(coss))

def test_with_correction(pairs_AB, dir_A, s_A, dir_B, s_B, correction):
    """Apply direct composition + correction vector, measure hit rate."""
    hits = 0; n = 0
    for src, tgt in pairs_AB:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred = es + s_A * dir_A + s_B * dir_B + correction
        result = nn_ret(pred, source_ids(src), RELAXED_MASK)
        if result == tgt: hits += 1
    return hits, n

# ============================================================
# PHASE 1: Build axes (same as Day 348)
# ============================================================
print("\nDAY 349: The Interaction Term")
print("="*70)

GENDER = [('king','queen'),('man','woman'),('boy','girl'),
          ('father','mother'),('son','daughter'),('husband','wife'),
          ('uncle','aunt'),('prince','princess'),('actor','actress'),
          ('waiter','waitress')]
PLURAL = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
          ('tree','trees'),('book','books'),('bird','birds'),('door','doors'),
          ('hand','hands'),('arm','arms'),('eye','eyes'),('leg','legs')]
ER_COMP= [('big','bigger'),('fast','faster'),('tall','taller'),
          ('clean','cleaner'),('bright','brighter'),('warm','warmer'),
          ('long','longer'),('cold','colder'),('old','older'),
          ('smart','smarter'),('strong','stronger'),('light','lighter')]
ER_SUP = [('big','biggest'),('fast','fastest'),('tall','tallest'),
          ('clean','cleanest'),('bright','brightest'),('warm','warmest'),
          ('long','longest'),('cold','coldest'),('old','oldest'),
          ('smart','smartest'),('strong','strongest'),('light','lightest')]
UN_NEG = [('happy','unhappy'),('clear','unclear'),('fair','unfair'),
          ('likely','unlikely'),('known','unknown'),('safe','unsafe'),
          ('usual','unusual'),('equal','unequal'),('stable','unstable'),
          ('real','unreal')]
EN_FR  = [('house','maison'),('sun','soleil'),('book','livre'),
          ('day','jour'),('night','nuit'),('cat','chat'),('dog','chien'),
          ('fire','feu'),('moon','lune'),('sea','mer'),('sky','ciel')]

print("\nPhase 1: Building axes...")
gender_dir, gender_v = build_axis(GENDER)
plural_dir, plural_v = build_axis(PLURAL)
comp_dir,   comp_v   = build_axis(ER_COMP)
sup_dir,    sup_v    = build_axis(ER_SUP)
un_dir,     un_v     = build_axis(UN_NEG)
fr_dir,     fr_v     = build_axis(EN_FR)

s_g  = best_scale(gender_dir, gender_v, RELAXED_MASK)
s_p  = best_scale(plural_dir, plural_v, RELAXED_MASK)
s_c  = best_scale(comp_dir,   comp_v,   RELAXED_MASK)
s_s  = best_scale(sup_dir,    sup_v,    RELAXED_MASK)
s_u  = best_scale(un_dir,     un_v,     RELAXED_MASK)
s_f  = best_scale(fr_dir,     fr_v,     RELAXED_MASK)

# comp-to-sup direction: sup - comp (as offset vectors)
comp_to_sup_dir = normed(sup_dir * s_s - comp_dir * s_c)
s_cs = np.linalg.norm(sup_dir * s_s - comp_dir * s_c)

print("  gender=%.3f  plural=%.3f  comp=%.3f  sup=%.3f  un=%.3f  fr=%.3f" % (
    s_g, s_p, s_c, s_s, s_u, s_f))

# ============================================================
# PHASE 2: Define all composition pair sets (src, final_tgt)
# ============================================================

# Cross-family compositions (single-token targets verified or assumed)
GENDER_PLURAL = [
    ('man','women'), ('king','queens'), ('boy','girls'),
    ('son','daughters'), ('actor','actresses'),
]
PLURAL_GENDER = [
    ('man','women'), ('boy','girls'), ('son','daughters'),
]
COMP_SUP_CHAIN = [
    ('big','biggest'), ('fast','fastest'), ('tall','tallest'),
    ('long','longest'), ('old','oldest'), ('cold','coldest'),
    ('bright','brightest'), ('warm','warmest'), ('clean','cleanest'),
]

# en_fr + plural: English word → French plural form
EN_FR_PLURAL = [
    ('cat','chats'), ('dog','chiens'), ('book','livres'),
    ('day','jours'), ('night','nuits'), ('fire','feux'),
]

# gender + en_fr: male English → female French equivalent
GENDER_FR = [
    ('man','femme'), ('boy','fille'), ('king','reine'),
    ('son','fille'),
]

# un_neg + er_comp: even though targets are multi-token, compute interaction on valid src/tgt
# We use base vs un-base+comp: test interaction purely on vector geometry, not vocab retrieval
UN_COMP_PAIRS = [
    ('happy','unhappy'), ('clear','unclear'), ('fair','unfair'),
    ('safe','unsafe'), ('equal','unequal'),
]

# ============================================================
# PHASE 3: Compute interaction terms for each crossing type
# ============================================================
print("\nPhase 3: Interaction term analysis")
print("-"*70)

crossing_types = [
    ("gender → plural    [cross-family]",
     GENDER_PLURAL, gender_dir, s_g, plural_dir, s_p),
    ("plural → gender    [cross-family, reversed]",
     PLURAL_GENDER, plural_dir, s_p, gender_dir, s_g),
    ("comp → comp-to-sup [same-family]",
     COMP_SUP_CHAIN, comp_dir, s_c, comp_to_sup_dir, s_cs),
    ("en_fr → plural     [cross-family]",
     EN_FR_PLURAL, fr_dir, s_f, plural_dir, s_p),
    ("gender → en_fr     [cross-family]",
     GENDER_FR, gender_dir, s_g, fr_dir, s_f),
]

mean_interactions = {}

for label, pairs, dA, sA, dB, sB in crossing_types:
    ilist = compute_interactions(pairs, dA, sA, dB, sB)
    if not ilist:
        print("  %-38s  no valid pairs" % label); continue

    mean_dir, avg_cos, std_cos = consistency_score(ilist)
    rel_mags = [r[5] for r in ilist]
    mean_rel = np.mean(rel_mags)

    print("\n  %s" % label)
    print("    pairs=%d  avg_cos=%.4f (±%.4f)  rel_magnitude=%.3f" % (
        len(ilist), avg_cos, std_cos, mean_rel))
    print("    per-pair cos(interaction_i, mean_interaction):")
    for src, tgt, chord, _, inter, rel in ilist:
        c = float(np.dot(normed(inter).astype(np.float32), mean_dir.astype(np.float32)))
        chord_mag = np.linalg.norm(chord)
        inter_mag = np.linalg.norm(inter)
        print("      %-8s → %-10s  cos=%.3f  |inter|=%.3f  |chord|=%.3f  rel=%.2f" % (
            src, tgt, c, inter_mag, chord_mag, rel))

    mean_interactions[label] = mean_dir

    # What vocabulary region does the interaction point toward?
    top_words = nearest_vocab(mean_dir, n=8)
    print("    interaction direction → vocab: %s" % ', '.join(
        '%s(%.3f)' % (w, s) for w, s in top_words[:6]))

# ============================================================
# PHASE 4: Cross-type interaction similarity matrix
# ============================================================
print("\n" + "-"*70)
print("Phase 4: Interaction direction similarity matrix")
print("  (How similar are the interaction vectors across different crossing types?)")
print("  1.0 = same interaction; 0.0 = orthogonal; -1.0 = opposite")

labels_short = {
    "gender → plural    [cross-family]":  "g→p",
    "plural → gender    [cross-family, reversed]": "p→g",
    "comp → comp-to-sup [same-family]":   "c→s",
    "en_fr → plural     [cross-family]":  "f→p",
    "gender → en_fr     [cross-family]":  "g→f",
}

keys = list(mean_interactions.keys())
print("\n  %6s" % '' + ''.join("  %6s" % labels_short.get(k,'?') for k in keys))
for k1 in keys:
    row = "  %6s" % labels_short.get(k1,'?')
    for k2 in keys:
        c = np.dot(mean_interactions[k1].astype(np.float32),
                   mean_interactions[k2].astype(np.float32))
        row += "  %+6.3f" % c
    print(row)

# ============================================================
# PHASE 5: Does the correction help?
# ============================================================
print("\n" + "-"*70)
print("Phase 5: Correction test")
print("  Apply direct composition WITH mean_interaction as correction term.")
print("  Does accuracy improve beyond the 0/10 baseline from Day 348?\n")

gp_key = "gender → plural    [cross-family]"
if gp_key in mean_interactions:
    corr = mean_interactions[gp_key]

    # Sweep correction scale
    print("  gender+plural, sweep correction scale:")
    print("  %-8s  hits/n" % "scale")
    best_corr_hits = 0; best_corr_s = 0.0
    for cscale in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        correction = cscale * corr
        hits, n = test_with_correction(GENDER_PLURAL, gender_dir, s_g, plural_dir, s_p, correction)
        print("  %-8.2f  %d/%d = %.0f%%" % (cscale, hits, n, 100*hits/max(n,1)))
        if hits > best_corr_hits: best_corr_hits=hits; best_corr_s=cscale

    print("\n  Best corrected: %.0f/%d = %.0f%% at scale=%.2f" % (
        best_corr_hits, len(GENDER_PLURAL), 100*best_corr_hits/max(len(GENDER_PLURAL),1), best_corr_s))

# Also try comp→sup with correction (same-family, already works — correction should make no difference)
cs_key = "comp → comp-to-sup [same-family]"
if cs_key in mean_interactions:
    corr_cs = mean_interactions[cs_key]
    print("\n  comp→sup with its own correction (same-family, should be stable):")
    for cscale in [0.0, 0.5, 1.0]:
        correction = cscale * corr_cs
        hits, n = test_with_correction(COMP_SUP_CHAIN, comp_dir, s_c, comp_to_sup_dir, s_cs, correction)
        print("  %-8.2f  %d/%d = %.0f%%" % (cscale, hits, n, 100*hits/max(n,1)))

# ============================================================
# PHASE 6: Is the interaction term in the same semantic family as
#          either axis, or in a completely different region?
# ============================================================
print("\n" + "-"*70)
print("Phase 6: Interaction term geometry — where does it sit?")
print("  Measure cos(interaction, each axis) to see if it's in any known direction.\n")

axis_dirs = [
    ('gender',  gender_dir),
    ('plural',  plural_dir),
    ('er_comp', comp_dir),
    ('er_sup',  sup_dir),
    ('un_neg',  un_dir),
    ('en_fr',   fr_dir),
]

for label, inter_dir in mean_interactions.items():
    short = labels_short.get(label, label[:10])
    row = "  inter[%4s] vs:" % short
    for aname, adir in axis_dirs:
        c = np.dot(inter_dir.astype(np.float32), adir.astype(np.float32))
        row += "  %s=%+.3f" % (aname, c)
    print(row)

# Also check: is the interaction direction close to ANY of the pair chords themselves?
print("\n  Checking: is interaction a residual of chord_A, chord_B, or truly novel?")
gp_ilist = compute_interactions(GENDER_PLURAL, gender_dir, s_g, plural_dir, s_p)
if gp_ilist:
    mean_chord_AB  = normed(np.mean([r[2] for r in gp_ilist], axis=0))
    mean_inter     = mean_interactions.get(gp_key)
    if mean_inter is not None:
        c_chord = np.dot(mean_inter.astype(np.float32), mean_chord_AB.astype(np.float32))
        c_g     = np.dot(mean_inter.astype(np.float32), gender_dir.astype(np.float32))
        c_p     = np.dot(mean_inter.astype(np.float32), plural_dir.astype(np.float32))
        print("  gender+plural interaction vs:")
        print("    mean chord_AB:    cos=%.4f" % c_chord)
        print("    gender axis dir:  cos=%.4f" % c_g)
        print("    plural axis dir:  cos=%.4f" % c_p)

# ============================================================
# PHASE 7: Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Day 349 Interaction Term")
print("="*70)
print("  Key question: is the interaction term consistent within a crossing type?")
print("  If avg_cos > 0.7 → consistent → learnable correction")
print("  If avg_cos < 0.3 → idiosyncratic → no universal correction exists")
print()
for label, pairs, dA, sA, dB, sB in crossing_types:
    ilist = compute_interactions(pairs, dA, sA, dB, sB)
    if not ilist: continue
    _, avg_cos, _ = consistency_score(ilist)
    verdict = ("CONSISTENT (learnable)" if avg_cos > 0.7
               else "MODERATE" if avg_cos > 0.4
               else "IDIOSYNCRATIC (not learnable)")
    short = labels_short.get(label, label[:12])
    print("  %-6s  avg_cos=%.3f  %s" % (short, avg_cos, verdict))
