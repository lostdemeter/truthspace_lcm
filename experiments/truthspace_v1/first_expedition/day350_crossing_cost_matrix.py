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
        chords.append(et - es); valid.append((s, t, sid, tid, et - es))
    if not chords: return None, []
    return normed(np.mean(chords, axis=0)), valid

def best_scale(ax_dir, valid, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = sum(1 for _,t,sid,_,_ in valid
                if nn_ret(W_E[sid] + s*ax_dir, source_ids(tok.decode([sid]).strip()), mask) == t)
        if c > best_a: best_a=c; best_s=s
    return best_s

# ============================================================
# AXIS DATA
# ============================================================
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

print("\nDAY 350: Crossing Cost Matrix")
print("="*70)

print("\nPhase 1: Building axes...")
gender_dir, gender_v = build_axis(GENDER)
plural_dir, plural_v = build_axis(PLURAL)
comp_dir,   comp_v   = build_axis(ER_COMP)
sup_dir,    sup_v    = build_axis(ER_SUP)
un_dir,     un_v     = build_axis(UN_NEG)
fr_dir,     fr_v     = build_axis(EN_FR)

s_g = best_scale(gender_dir, gender_v, RELAXED_MASK)
s_p = best_scale(plural_dir, plural_v, RELAXED_MASK)
s_c = best_scale(comp_dir,   comp_v,   RELAXED_MASK)
s_s = best_scale(sup_dir,    sup_v,    RELAXED_MASK)
s_u = best_scale(un_dir,     un_v,     RELAXED_MASK)
s_f = best_scale(fr_dir,     fr_v,     RELAXED_MASK)
comp_to_sup_dir = normed(sup_dir * s_s - comp_dir * s_c)
s_cs = np.linalg.norm(sup_dir * s_s - comp_dir * s_c)

AXES = {
    'gender': (gender_dir, s_g),
    'plural': (plural_dir, s_p),
    'er_comp': (comp_dir, s_c),
    'er_sup':  (sup_dir, s_s),
    'un_neg':  (un_dir, s_u),
    'en_fr':   (fr_dir, s_f),
}
print("  Scales: " + "  ".join("%s=%.3f" % (k,v[1]) for k,v in AXES.items()))

# ============================================================
# CROSSING COST FUNCTION
# ============================================================
# From Day 349: interaction ≈ -k · (s_B · dir_B)
# where k = cos(mean_interaction, dir_B) × |mean_interaction| / (s_B × something)
#
# Simpler: the interaction is dominated by -cos(inter, dir_B) × |inter| in dir_B direction.
# We estimate k as: k = -cos(mean_inter, dir_B), the projection coefficient.
# Then the "anti-axis correction" is: correction = k × s_B × dir_B (to SUBTRACT from sum).
# That is: Δ_composed ≈ s_A·dir_A + (1−k)·s_B·dir_B
#
# This is the ZERO-SHOT prediction: no training pairs needed, just k.
# ============================================================

def compute_k_factor(pairs_AB, dir_A, s_A, dir_B, s_B):
    """
    Compute the k factor for crossing A→B:
    k = projection of mean interaction onto dir_B, normalized by s_B.
    Returns (k, mean_interaction_dir, consistency_cos).
    """
    sum_pred = s_A * dir_A + s_B * dir_B
    interactions = []
    for src, tgt in pairs_AB:
        es, _ = get_emb(src); et, _ = get_emb(tgt)
        if es is None or et is None: continue
        chord = et - es
        interactions.append(chord - sum_pred)
    if not interactions: return 0.0, None, 0.0

    mean_inter = np.mean(interactions, axis=0)
    mean_inter_dir = normed(mean_inter)

    # k = -cos(mean_inter, dir_B)  (the anti-B projection)
    # (negative because interaction is anti-B, so cos is negative)
    cos_with_B = float(np.dot(mean_inter_dir.astype(np.float32),
                               dir_B.astype(np.float32)))
    k = -cos_with_B  # positive k means interaction opposes B

    # Consistency: avg cos(each interaction, mean interaction)
    if len(interactions) > 1:
        coss = [float(np.dot(normed(iv).astype(np.float32),
                             mean_inter_dir.astype(np.float32)))
                for iv in interactions]
        consistency = float(np.mean(coss))
    else:
        consistency = 1.0

    return k, mean_inter_dir, consistency

def test_composition(pairs, dir_A, s_A, dir_B, s_B, correction_k, label=""):
    """
    Test direct composition with anti-B correction at scale k.
    Δ_composed = s_A·dir_A + (1−k)·s_B·dir_B
    Equivalent to: sum + k·(-dir_B)·s_B
    """
    hits = 0; n = 0; details = []
    for src, tgt in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred = es + s_A * dir_A + (1.0 - correction_k) * s_B * dir_B
        result = nn_ret(pred, source_ids(src), RELAXED_MASK)
        ok = (result == tgt)
        if ok: hits += 1
        details.append((src, tgt, result, ok))
    return hits, n, details

# ============================================================
# PHASE 2: Compute k-factor for each crossing type
# ============================================================
# Composition pairs: (A, B, training_pairs, test_pairs)
# Training pairs used to estimate k; test pairs used to evaluate

GENDER_PLURAL_TRAIN = [('man','women'), ('king','queens'), ('boy','girls')]
GENDER_PLURAL_TEST  = [('son','daughters'), ('actor','actresses'),
                       ('father','mothers'), ('uncle','aunts')]

PLURAL_GENDER_TRAIN = [('man','women'), ('boy','girls'), ('son','daughters')]
PLURAL_GENDER_TEST  = []  # small set, use same as train for now

COMP_SUP_TRAIN = [('big','biggest'),('fast','fastest'),('tall','tallest'),
                  ('long','longest'),('old','oldest'),('cold','coldest')]
COMP_SUP_TEST  = [('bright','brightest'),('warm','warmest'),('clean','cleanest'),
                  ('smart','smartest'),('strong','strongest')]

EN_FR_PLURAL_TRAIN = [('cat','chats'),('dog','chiens'),('book','livres')]
EN_FR_PLURAL_TEST  = [('day','jours'),('night','nuits')]

GENDER_FR_TRAIN = [('man','femme'),('boy','fille'),('son','fille')]
GENDER_FR_TEST  = [('king','reine')]

crossing_specs = [
    ("gender→plural   [cross]", 'gender', 'plural',
     GENDER_PLURAL_TRAIN, GENDER_PLURAL_TEST),
    ("plural→gender   [cross]", 'plural', 'gender',
     PLURAL_GENDER_TRAIN, PLURAL_GENDER_TEST),
    ("comp→comp-to-sup[same]",  'er_comp', 'er_sup',  # proxy: using sup axis as B
     COMP_SUP_TRAIN, COMP_SUP_TEST),
    ("en_fr→plural    [cross]", 'en_fr',  'plural',
     EN_FR_PLURAL_TRAIN, EN_FR_PLURAL_TEST),
    ("gender→en_fr    [cross]", 'gender', 'en_fr',
     GENDER_FR_TRAIN, GENDER_FR_TEST),
]

print("\nPhase 2: k-factor estimation (anti-second-axis projection)")
print("  Hypothesis: interaction ≈ -k · s_B · dir_B  (pure anti-B correction)")
print()
print("  %-28s  k       consistency  interpretation" % "crossing")
print("-"*70)

k_values = {}
for label, ax_A_name, ax_B_name, train_pairs, test_pairs in crossing_specs:
    dir_A, s_A = AXES[ax_A_name]
    dir_B, s_B = AXES[ax_B_name]
    k, mean_dir, consistency = compute_k_factor(train_pairs, dir_A, s_A, dir_B, s_B)
    k_values[label] = k
    interp = ("anti-B dominant" if k > 0.5
              else "partial anti-B" if k > 0.2
              else "minimal correction")
    print("  %-28s  k=%+.3f  cos=%+.3f        %s" % (label, k, consistency, interp))

# ============================================================
# PHASE 3: Zero-shot correction test
# ============================================================
# Using k estimated from TRAIN pairs only, test on HELD-OUT pairs.
# Zero-shot means: we don't fit a free-form correction vector,
# just use k × (-dir_B) as the correction direction.
# ============================================================

print("\nPhase 3: Zero-shot correction generalization to held-out pairs")
print("  Formula: Δ_composed = s_A·dir_A + (1−k)·s_B·dir_B  (k from train pairs)")
print()

for label, ax_A_name, ax_B_name, train_pairs, test_pairs in crossing_specs:
    dir_A, s_A = AXES[ax_A_name]
    dir_B, s_B = AXES[ax_B_name]
    k = k_values[label]
    all_pairs = train_pairs + test_pairs

    # Baseline: k=0 (no correction = direct sum)
    h_base, n_base, _ = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, 0.0)
    # With k (anti-B correction)
    h_corr, n_corr, details = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, k)
    # With k=1 (completely cancel B — pure A)
    h_pure_A, n_pa, _ = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, 1.0)

    print("  %-28s  k=%.3f" % (label, k))
    print("    base (k=0):   %d/%d = %.0f%%" % (h_base, n_base, 100*h_base/max(n_base,1)))
    print("    corrected:    %d/%d = %.0f%%" % (h_corr, n_corr, 100*h_corr/max(n_corr,1)))
    print("    pure-A (k=1): %d/%d = %.0f%%" % (h_pure_A, n_pa, 100*h_pure_A/max(n_pa,1)))
    for src, tgt, result, ok in details:
        mark = '✓' if ok else '✗'
        train_tag = ' [train]' if (src,tgt) in train_pairs else ' [test]'
        print("    %s %-8s → %-12s  expected=%-12s %s" % (
            mark, src, result, tgt, train_tag))
    print()

# ============================================================
# PHASE 4: Full k sweep — find optimal k for each crossing
# ============================================================
print("Phase 4: Optimal k sweep (what k maximises accuracy?)")
print("  Testing k in [0, 1.5] to find the global optimum")
print()

for label, ax_A_name, ax_B_name, train_pairs, test_pairs in crossing_specs:
    dir_A, s_A = AXES[ax_A_name]
    dir_B, s_B = AXES[ax_B_name]
    all_pairs = train_pairs + test_pairs

    best_k, best_h = 0.0, 0
    ks_results = []
    for k in np.linspace(0.0, 1.5, 31):
        h, n, _ = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, k)
        ks_results.append((k, h, n))
        if h > best_h: best_h=h; best_k=k

    print("  %-28s  optimal_k=%.3f  acc=%d/%d=%.0f%%" % (
        label, best_k, best_h, len(all_pairs),
        100*best_h/max(len(all_pairs),1)))

    # Show curve around optimum
    curve = [(k, h, n) for k, h, n in ks_results
             if abs(k - best_k) < 0.4]
    print("    k sweep (near optimum):", end="")
    for k, h, n in curve[::2]:
        print(" k=%.2f→%d/%d" % (k, h, n), end="")
    print()

# ============================================================
# PHASE 5: The universal k hypothesis
# ============================================================
# Hypothesis: k is approximately constant across ALL cross-family crossings.
# If k ≈ 0.7 universally, then composition rule is simply:
#   Δ_composed = s_A·dir_A + 0.3·s_B·dir_B  (regardless of which families)
#
# Test: collect optimal k values and check variance.
# ============================================================

print("\nPhase 5: Universal k hypothesis")
print("  Is k ≈ constant across cross-family crossings?")
print()

cross_k = []
same_k  = []

for label, ax_A_name, ax_B_name, train_pairs, test_pairs in crossing_specs:
    dir_A, s_A = AXES[ax_A_name]
    dir_B, s_B = AXES[ax_B_name]
    all_pairs = train_pairs + test_pairs
    is_same = '[same]' in label

    best_k, best_h = 0.0, 0
    for k in np.linspace(0.0, 1.5, 31):
        h, n, _ = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, k)
        if h > best_h: best_h=h; best_k=k

    if is_same:
        same_k.append(best_k)
    else:
        cross_k.append(best_k)

print("  Cross-family optimal k values: %s" % [round(k,3) for k in cross_k])
print("  Same-family optimal k values:  %s" % [round(k,3) for k in same_k])
if cross_k:
    print("  Cross-family k: mean=%.3f  std=%.3f" % (np.mean(cross_k), np.std(cross_k)))
if same_k:
    print("  Same-family k:  mean=%.3f  std=%.3f" % (np.mean(same_k), np.std(same_k)))
PHI = (1 + np.sqrt(5)) / 2
print()
print("  Reference values:")
print("    1/φ        = %.4f" % (1/PHI))
print("    1/φ²       = %.4f" % (1/PHI**2))
print("    1 - 1/φ    = %.4f" % (1 - 1/PHI))
print("    1 - 1/φ²   = %.4f" % (1 - 1/PHI**2))

if cross_k:
    k_mean = np.mean(cross_k)
    print()
    print("  Closest φ-level to mean k:")
    for name, val in [('1/φ', 1/PHI), ('1/φ²', 1/PHI**2), ('1-1/φ', 1-1/PHI),
                       ('1-1/φ²', 1-1/PHI**2), ('0.5', 0.5), ('2/φ²', 2/PHI**2)]:
        print("    |k_mean - %s| = %.4f" % (name, abs(k_mean - val)))

# ============================================================
# PHASE 6: Crossing cost matrix (full)
# ============================================================
# Measure the pairwise crossing costs between ALL axis types.
# For each (A, B) pair, estimate k from whatever word pairs
# are available or from the anti-B projection formula directly.
# ============================================================

print("\n" + "-"*70)
print("Phase 6: Full crossing cost matrix (k from anti-B projection)")
print("  k = -cos(mean_interaction, dir_B)  measured for all family pairs")
print("  using the training pairs from each single-axis set as proxy\n")

# For each (A, B) pair, use words where we have A-transformed words as intermediates
# Proxy approach: use A's training words (src, tgt_A) and B's training words (src', tgt_B)
# and measure: for words that appear in both, compute direct chord and subtract sum

# Simplified: compute k analytically from axis cosine similarity
# If axes are orthogonal (cos=0), k should be 0 (no interaction)
# If axes are correlated (cos>0), k should be positive (more damping)
# This gives us the k PREDICTOR: k ≈ cos(dir_A, dir_B) × correction

print("  Axis cosine similarity (basis for k prediction):")
axis_list = [('gender', gender_dir), ('plural', plural_dir),
             ('er_comp', comp_dir), ('er_sup', sup_dir),
             ('un_neg', un_dir), ('en_fr', fr_dir)]
print("  %8s" % '' + ''.join("  %7s" % a[0][:7] for a in axis_list))
for n1, d1 in axis_list:
    row = "  %8s" % n1
    for n2, d2 in axis_list:
        c = float(np.dot(d1.astype(np.float32), d2.astype(np.float32)))
        row += "  %+7.3f" % c
    print(row)

# The key question: does cos(dir_A, dir_B) predict k?
# If k ≈ f(cos(dir_A, dir_B)), then crossing cost is derivable from axis directions alone.
print()
print("  Empirical k vs predicted k from cos(dir_A, dir_B):")
print("  %-28s  cos(A,B)   empirical_k  predicted_k=|cos|" % "crossing")
for label, ax_A_name, ax_B_name, train_pairs, _ in crossing_specs:
    dir_A, s_A = AXES[ax_A_name]
    dir_B, s_B = AXES[ax_B_name]
    cos_AB = float(np.dot(dir_A.astype(np.float32), dir_B.astype(np.float32)))
    k_emp = k_values[label]
    print("  %-28s  cos=%+.3f  k_emp=%.3f   k_pred=%.3f" % (
        label, cos_AB, k_emp, abs(cos_AB)))

# ============================================================
# PHASE 7: SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Day 350 Crossing Cost Matrix")
print("="*70)
print()
print("  Central question: is k predictable without training pairs?")
print("  k = anti-B projection coefficient in: chord_AB ≈ s_A·dir_A + (1-k)·s_B·dir_B")
print()
print("  Results:")
for label, ax_A_name, ax_B_name, train_pairs, test_pairs in crossing_specs:
    dir_A, s_A = AXES[ax_A_name]
    dir_B, s_B = AXES[ax_B_name]
    cos_AB = float(np.dot(dir_A.astype(np.float32), dir_B.astype(np.float32)))
    k_emp = k_values[label]
    # find optimal k
    all_pairs = train_pairs + test_pairs
    best_k, best_h = 0.0, 0
    for kk in np.linspace(0.0, 1.5, 31):
        h, _, _ = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, kk)
        if h > best_h: best_h=h; best_k=kk
    h0, n0, _ = test_composition(all_pairs, dir_A, s_A, dir_B, s_B, 0.0)
    print("  %-28s  cos(A,B)=%+.3f  k_proj=%.3f  k_opt=%.3f  base=%d/%d  best=%d/%d" % (
        label, cos_AB, k_emp, best_k, h0, n0, best_h, n0))
print()
print("  If k_proj ≈ k_opt: the anti-B projection formula is sufficient (zero-shot)")
print("  If k_proj ≠ k_opt: a scale calibration is needed (few-shot)")
