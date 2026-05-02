import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
ENGLISH_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) < 2: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
    if w.isalpha() and w.isascii(): ENGLISH_MASK[i] = True

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

def tokenize_word(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if ids: return ids, [tok.decode([i]) for i in ids]
    return [], []

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def nn_ret_top(pred_emb, excl_ids, mask, top_k=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    top = np.argsort(sims)[::-1][:top_k*2]
    out = []
    for idx in top:
        if len(out) >= top_k: break
        out.append((tok.decode([int(idx)]).strip(), float(sims[idx]), int(idx)))
    return out

def build_axis(pairs):
    chords = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es)
    return normed(np.mean(chords, axis=0))

def best_scale(ax_dir, pairs, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = sum(1 for sr,tg in pairs
                if (lambda es,_: es is not None and
                    nn_ret(es + s*ax_dir, source_ids(sr), mask)[0]==tg
                )(*get_emb(sr)))
        if c > best_a: best_a=c; best_s=s
    return best_s

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

GP_PAIRS = [
    ('man',    'woman',    'women'),
    ('king',   'queen',    'queens'),
    ('boy',    'girl',     'girls'),
    ('son',    'daughter', 'daughters'),
    ('actor',  'actress',  'actresses'),
    ('father', 'mother',   'mothers'),
    ('uncle',  'aunt',     'aunts'),
    ('prince', 'princess', 'princesses'),
    ('husband','wife',     'wives'),
    ('waiter', 'waitress', 'waitresses'),
]
CS_PAIRS = [
    ('big',    'bigger',   'biggest'),
    ('fast',   'faster',   'fastest'),
    ('tall',   'taller',   'tallest'),
    ('long',   'longer',   'longest'),
    ('old',    'older',    'oldest'),
    ('cold',   'colder',   'coldest'),
    ('bright', 'brighter', 'brightest'),
    ('warm',   'warmer',   'warmest'),
    ('clean',  'cleaner',  'cleanest'),
    ('smart',  'smarter',  'smartest'),
    ('strong', 'stronger', 'strongest'),
    ('light',  'lighter',  'lightest'),
]

print("\nDAY 353: Axis-Residual Decoding")
print("="*70)

print("\nPhase 1: Building axes...")
gender_dir = build_axis(GENDER)
plural_dir = build_axis(PLURAL)
comp_dir   = build_axis(ER_COMP)
sup_dir    = build_axis(ER_SUP)

s_g = best_scale(gender_dir, GENDER, RELAXED_MASK)
s_p = best_scale(plural_dir, PLURAL, RELAXED_MASK)
s_c = best_scale(comp_dir,   ER_COMP, RELAXED_MASK)
s_s = best_scale(sup_dir,    ER_SUP,  RELAXED_MASK)
comp_to_sup_dir = normed(sup_dir * s_s - comp_dir * s_c)
s_cs = float(np.linalg.norm(sup_dir * s_s - comp_dir * s_c))

print("  scales: g=%.3f  p=%.3f  c=%.3f  s=%.3f  c->s=%.3f" % (
    s_g, s_p, s_c, s_s, s_cs))

# ============================================================
# PHASE 2: Residual discriminator for M1 failures
# ============================================================
# Key prediction: for M1 failures (cold→cold instead of coldest),
# the residual (pred - snap_emb) has high cos with comp_to_sup_dir.
# For successes (big→biggest), residual has LOW cos with axis_dir.
#
# Why? When snap succeeds: pred ≈ snap_emb → residual ≈ 0 (random direction)
# When snap fails (M1): pred = colder + s*axis, snap="cold" ≠ target,
#                        pred - snap ≈ s*axis → cos(residual, axis) ≈ high

print("\nPhase 2: Residual discriminator — comp→sup chain")
print("  cos(residual, comp_to_sup_dir) for each pair")
print("  Prediction: HIGH for M1 failures, LOW for successes\n")

resid_sims_success = []
resid_sims_failure = []

for src, inter, final in CS_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    # Step 1: adj → comparative
    pred1 = es + s_c * comp_dir
    inter_word, _, inter_idx = nn_ret(pred1, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    if ei is None: continue
    # Step 2: comparative → superlative
    pred2 = ei + s_cs * comp_to_sup_dir
    snap_word, snap_sim, snap_idx = nn_ret(pred2, source_ids(inter_word), RELAXED_MASK)
    snap_emb = W_E[snap_idx].copy()
    step_ok = (snap_word == final)

    # Residual
    residual = pred2 - snap_emb
    resid_norm = np.linalg.norm(residual) + 1e-8
    resid_dir  = residual / resid_norm
    resid_cos  = float(np.dot(resid_dir.astype(np.float32), comp_to_sup_dir.astype(np.float32)))

    # Sub-tokens of final
    ids_f, toks_f = tokenize_word(final)
    n_f = len(ids_f)
    ftype = 'SUCCESS' if step_ok else ('F3-M1' if n_f > 1 else 'F2')

    mark = '✓' if step_ok else '✗'
    print("  %s %-8s → %-10s → %-12s  expected=%-12s  resid_cos=%.4f  resid_mag=%.4f  [%s]" % (
        mark, src, inter_word, snap_word, final, resid_cos, resid_norm, ftype))

    if step_ok: resid_sims_success.append(resid_cos)
    else:       resid_sims_failure.append(resid_cos)

print()
print("  SUCCESS residual cos: mean=%.4f  min=%.4f  max=%.4f  (n=%d)" % (
    np.mean(resid_sims_success), np.min(resid_sims_success),
    np.max(resid_sims_success), len(resid_sims_success)))
print("  FAILURE residual cos: mean=%.4f  min=%.4f  max=%.4f  (n=%d)" % (
    np.mean(resid_sims_failure), np.min(resid_sims_failure),
    np.max(resid_sims_failure), len(resid_sims_failure)))

# Find threshold that separates them
all_vals = [(c, 1) for c in resid_sims_success] + [(c, 0) for c in resid_sims_failure]
best_thresh, best_acc = 0.0, 0.0
for thresh in np.linspace(-0.5, 1.0, 30):
    acc = sum(1 for c,ok in all_vals if (c < thresh) == ok) / len(all_vals)
    if acc > best_acc: best_acc=acc; best_thresh=thresh
print("  Best discriminant threshold: %.4f  (accuracy=%.0f%%)" % (
    best_thresh, 100*best_acc))

# ============================================================
# PHASE 3: Geometric morpheme finding
# ============================================================
# For M1 failures: the residual vector (pred2 - snap_emb) points along axis_dir.
# Can we recover the morpheme "est" by doing NN(residual) in the suffix vocabulary?
# Suffix tokens: tokens that look like morphemes (short, common suffixes)

print("\nPhase 3: Geometric morpheme finding via NN(residual)")
print("  What word/suffix does NN(residual) find for M1 failure cases?\n")

# Build a suffix-biased vocabulary: short tokens (1-5 chars), includes suffixes
SUFFIX_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i])
    if w.startswith(' '): continue  # exclude space-prefixed (full words)
    w = w.strip()
    if 1 <= len(w) <= 5 and w.isalpha() and w.isascii(): SUFFIX_MASK[i] = True

for src, inter, final in CS_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    pred1 = es + s_c * comp_dir
    inter_word, _, _ = nn_ret(pred1, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    if ei is None: continue
    pred2 = ei + s_cs * comp_to_sup_dir
    snap_word, _, snap_idx = nn_ret(pred2, source_ids(inter_word), RELAXED_MASK)
    if snap_word == final: continue  # skip successes

    snap_emb = W_E[snap_idx].copy()
    residual = pred2 - snap_emb

    # NN of the RESIDUAL in SUFFIX_MASK
    top_suffixes = nn_ret_top(residual, set(), SUFFIX_MASK, top_k=5)
    # NN of the RESIDUAL in ENGLISH_MASK (full words)
    top_words = nn_ret_top(residual, set(), ENGLISH_MASK, top_k=5)
    # NN of axis direction itself (what morpheme does comp_to_sup_dir point toward?)
    top_axis  = nn_ret_top(s_cs * comp_to_sup_dir, set(), SUFFIX_MASK, top_k=5)

    print("  %s → %s (FAIL→%s):" % (src, final, snap_word))
    print("    residual NN [suffix]: %s" % [(w,round(s,4)) for w,s,_ in top_suffixes])
    print("    residual NN [words]:  %s" % [(w,round(s,4)) for w,s,_ in top_words])

# What does the axis direction itself look like as a morpheme?
top_axis_any = nn_ret_top(s_cs * comp_to_sup_dir, set(), SUFFIX_MASK, top_k=10)
print()
print("  comp_to_sup_dir × s_cs NN [suffix]: %s" % [(w,round(s,4)) for w,s,_ in top_axis_any])

# Same for plural direction
top_plural_sfx = nn_ret_top(s_p * plural_dir, set(), SUFFIX_MASK, top_k=10)
print("  plural_dir × s_p NN [suffix]:       %s" % [(w,round(s,4)) for w,s,_ in top_plural_sfx])

# ============================================================
# PHASE 4: Full axis-residual decoding pipeline
# ============================================================
# Algorithm:
#   1. Snap to nearest single-token: snap_word
#   2. Compute residual = pred - snap_emb
#   3. cos(residual, axis_dir) > THRESHOLD → need morpheme extension
#   4. NN(residual, SUFFIX_MASK) → morpheme_token
#   5. Output = snap_word + morpheme_token
#
# Test on comp→sup chain with this enhanced decoding

print("\nPhase 4: Full axis-residual decoding pipeline (comp→sup)")
print()

# Use best_thresh from Phase 2, and also try tuned thresholds
for THRESHOLD in [best_thresh, 0.3, 0.5, 0.7]:
    hits = 0; n = 0
    details = []
    for src, inter, final in CS_PAIRS:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred1 = es + s_c * comp_dir
        inter_word, _, _ = nn_ret(pred1, source_ids(src), RELAXED_MASK)
        ei, _ = get_emb(inter_word)
        if ei is None: continue
        pred2 = ei + s_cs * comp_to_sup_dir
        snap_word, snap_sim, snap_idx = nn_ret(pred2, source_ids(inter_word), RELAXED_MASK)
        snap_emb = W_E[snap_idx].copy()

        residual = pred2 - snap_emb
        resid_cos = float(np.dot(normed(residual).astype(np.float32),
                                  comp_to_sup_dir.astype(np.float32)))

        if resid_cos > THRESHOLD:
            # Extend: find morpheme
            morpheme_hits = nn_ret_top(residual, set(), SUFFIX_MASK, top_k=3)
            morpheme = morpheme_hits[0][0].strip() if morpheme_hits else ''
            output = snap_word + morpheme
        else:
            output = snap_word

        ok = (output == final)
        if ok: hits += 1
        details.append((src, final, output, ok, resid_cos, THRESHOLD))

    print("  THRESHOLD=%.2f: %d/%d = %.0f%%" % (THRESHOLD, hits, n, 100*hits/max(n,1)))

# Verbose output for optimal threshold
print("\n  Detailed: THRESHOLD=%.2f" % best_thresh)
for src, inter, final in CS_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    pred1 = es + s_c * comp_dir
    inter_word, _, _ = nn_ret(pred1, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    if ei is None: continue
    pred2 = ei + s_cs * comp_to_sup_dir
    snap_word, _, snap_idx = nn_ret(pred2, source_ids(inter_word), RELAXED_MASK)
    snap_emb = W_E[snap_idx].copy()
    residual = pred2 - snap_emb
    resid_cos = float(np.dot(normed(residual).astype(np.float32),
                              comp_to_sup_dir.astype(np.float32)))
    if resid_cos > best_thresh:
        morph_hits = nn_ret_top(residual, set(), SUFFIX_MASK, top_k=1)
        morph = morph_hits[0][0].strip() if morph_hits else ''
        output = snap_word + morph
    else:
        output = snap_word
    ok = (output == final)
    mark = '✓' if ok else '✗'
    print("  %s %-8s → %-10s → %-14s  expected=%-12s  cos=%.4f  ext=%s" % (
        mark, src, inter_word, output, final, resid_cos,
        '✓' if resid_cos > best_thresh else ''))

# ============================================================
# PHASE 5: Residual decoding applied to gender→plural (M2 fix)
# ============================================================
# M2 case: princesses = ['princess', 'es']
# When we step from "princess" + plural_dir, we're moving to a region near "princesses".
# But "princesses" first token IS "princess" (excluded).
# Fix: allow "princess" as a valid snap target, then detect residual.

print("\nPhase 5: Axis-residual decoding — gender→plural chain")
print()

# Residual analysis for all GP_PAIRS
resid_sims_gp_success = []
resid_sims_gp_failure = []

print("  Residual cos(residual, plural_dir) for each pair:")
for src, inter, final in GP_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    pred1 = es + s_g * gender_dir
    inter_word, _, _ = nn_ret(pred1, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    if ei is None: continue
    pred2 = ei + s_p * plural_dir
    snap_word, snap_sim, snap_idx = nn_ret(pred2, source_ids(inter_word), ENGLISH_MASK)
    snap_emb = W_E[snap_idx].copy()
    step_ok = (snap_word == final)

    residual = pred2 - snap_emb
    resid_cos = float(np.dot(normed(residual).astype(np.float32),
                              plural_dir.astype(np.float32)))
    resid_mag = float(np.linalg.norm(residual))

    ids_f, toks_f = tokenize_word(final)
    n_f = len(ids_f)
    ftype = 'S' if step_ok else ('F3-M%s'%('1' if (toks_f and toks_f[0].strip()==inter_word) else '3') if n_f>1 else 'F2')

    mark = '✓' if step_ok else '✗'
    print("  %s %-9s → %-12s → %-14s  expected=%-14s  cos=%.4f  mag=%.4f  [%s]" % (
        mark, src, inter_word, snap_word, final, resid_cos, resid_mag, ftype))

    if step_ok: resid_sims_gp_success.append(resid_cos)
    else:       resid_sims_gp_failure.append(resid_cos)

print()
print("  SUCCESS cos: mean=%.4f (n=%d)" % (np.mean(resid_sims_gp_success) if resid_sims_gp_success else 0,
                                             len(resid_sims_gp_success)))
print("  FAILURE cos: mean=%.4f (n=%d)" % (np.mean(resid_sims_gp_failure) if resid_sims_gp_failure else 0,
                                             len(resid_sims_gp_failure)))

# Find threshold for GP
all_gp = [(c,1) for c in resid_sims_gp_success] + [(c,0) for c in resid_sims_gp_failure]
best_thresh_gp, best_acc_gp = 0.0, 0.0
for thresh in np.linspace(-0.5, 1.0, 30):
    acc = sum(1 for c,ok in all_gp if (c < thresh)==ok) / len(all_gp)
    if acc > best_acc_gp: best_acc_gp=acc; best_thresh_gp=thresh
print("  Best GP threshold: %.4f  (accuracy=%.0f%%)" % (best_thresh_gp, 100*best_acc_gp))

# M2 fix: allow intermediate word in search, then detect
print("\n  M2 fix for princesses: allow intermediate in NN search")
for src, inter, final in [('prince', 'princess', 'princesses')]:
    es, _ = get_emb(src)
    pred1 = es + s_g * gender_dir
    inter_word, _, _ = nn_ret(pred1, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)

    # Normal NN (excludes princess)
    snap_normal, sim_normal, _ = nn_ret(ei + s_p * plural_dir, source_ids(inter_word), ENGLISH_MASK)
    # NN WITHOUT exclusion
    snap_noxcl, sim_noxcl, snap_idx = nn_ret(ei + s_p * plural_dir, set(), ENGLISH_MASK)

    pred2 = ei + s_p * plural_dir
    snap_emb = W_E[snap_idx].copy()
    residual = pred2 - snap_emb
    morph_hits = nn_ret_top(residual, set(), SUFFIX_MASK, top_k=5)
    morph = morph_hits[0][0].strip() if morph_hits else ''
    output_M2 = snap_noxcl + morph

    print("  %s → %s (expected %s)" % (src, inter_word, final))
    print("    normal snap (excl inter): %s (%.4f)" % (snap_normal, sim_normal))
    print("    snap no-exclusion:        %s (%.4f)" % (snap_noxcl, sim_noxcl))
    print("    residual morpheme NN:     %s" % [(w,round(s,4)) for w,s,_ in morph_hits])
    print("    M2 output:                %s  %s" % (output_M2, '✓' if output_M2==final else '✗'))

# ============================================================
# PHASE 6: Full accuracy comparison (baseline vs residual decoding)
# ============================================================

def chain_residual(pairs, dir_A, s_A, dir_B, s_B, mask_A, mask_B,
                   axis_B_dir, threshold, allow_intermediate_in_B=False):
    """Chain with axis-residual extension at step 2."""
    hits = 0; n = 0; decoded = []
    for src, inter, final in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred1 = es + s_A * dir_A
        inter_word, _, _ = nn_ret(pred1, source_ids(src), mask_A)
        ei, _ = get_emb(inter_word)
        if ei is None: continue
        pred2 = ei + s_B * dir_B

        excl = set() if allow_intermediate_in_B else source_ids(inter_word)
        snap_word, _, snap_idx = nn_ret(pred2, excl, mask_B)
        snap_emb = W_E[snap_idx].copy()
        residual = pred2 - snap_emb
        resid_cos = float(np.dot(normed(residual).astype(np.float32),
                                  axis_B_dir.astype(np.float32)))
        if resid_cos > threshold:
            morph_hits = nn_ret_top(residual, set(), SUFFIX_MASK, top_k=1)
            morph = morph_hits[0][0].strip() if morph_hits else ''
            output = snap_word + morph
        else:
            output = snap_word

        ok = (output == final)
        if ok: hits += 1
        decoded.append((src, final, output, ok))
    return hits, n, decoded

print("\nPhase 6: Full accuracy — baseline vs residual decoding")
print()

# Comp→sup
h_base_cs, n_cs, _ = chain_residual(
    CS_PAIRS, comp_dir, s_c, comp_to_sup_dir, s_cs,
    RELAXED_MASK, RELAXED_MASK, comp_to_sup_dir, threshold=999.0)
print("  comp→sup baseline:                %d/%d = %.0f%%" % (h_base_cs, n_cs, 100*h_base_cs/max(n_cs,1)))

for thresh in [0.3, best_thresh, 0.5, 0.7]:
    h, n, _ = chain_residual(
        CS_PAIRS, comp_dir, s_c, comp_to_sup_dir, s_cs,
        RELAXED_MASK, RELAXED_MASK, comp_to_sup_dir, threshold=thresh)
    print("  comp→sup residual (t=%.2f):      %d/%d = %.0f%%" % (thresh, h, n, 100*h/max(n,1)))

# Gender→plural
h_base_gp, n_gp, _ = chain_residual(
    GP_PAIRS, gender_dir, s_g, plural_dir, s_p,
    RELAXED_MASK, ENGLISH_MASK, plural_dir, threshold=999.0)
print("  gender→plural baseline (EN step2): %d/%d = %.0f%%" % (h_base_gp, n_gp, 100*h_base_gp/max(n_gp,1)))

for thresh in [0.3, best_thresh_gp, 0.5, 0.7]:
    h, n, _ = chain_residual(
        GP_PAIRS, gender_dir, s_g, plural_dir, s_p,
        RELAXED_MASK, ENGLISH_MASK, plural_dir, threshold=thresh)
    print("  gender→plural residual (t=%.2f):  %d/%d = %.0f%%" % (thresh, h, n, 100*h/max(n,1)))

# With M2 fix (allow intermediate in B search) + residual
h_m2, n_m2, dec_m2 = chain_residual(
    GP_PAIRS, gender_dir, s_g, plural_dir, s_p,
    RELAXED_MASK, ENGLISH_MASK, plural_dir, threshold=best_thresh_gp,
    allow_intermediate_in_B=True)
print("  gender→plural + M2-noexcl:         %d/%d = %.0f%%" % (h_m2, n_m2, 100*h_m2/max(n_m2,1)))

# ============================================================
# PHASE 7: Residual as a GENERAL signal — does cos(residual, axis) work across axes?
# ============================================================
# If the residual signal generalises, it means: after any axis navigation step,
# the residual vector is the "unaccounted-for axis energy" left in the prediction.
# This would be a general principle: the snap residual encodes HOW MUCH the axis was applied.

print("\nPhase 7: Residual generalisation — is residual∝axis_dir across all steps?")
print("  Measuring cos(pred - snap, axis_dir) for single-step navigation\n")

for label, pairs, ax_dir, s, mask in [
    ("gender",  GENDER,  gender_dir, s_g, RELAXED_MASK),
    ("plural",  PLURAL,  plural_dir, s_p, RELAXED_MASK),
    ("comp",    ER_COMP, comp_dir,   s_c, RELAXED_MASK),
    ("sup",     ER_SUP,  sup_dir,    s_s, RELAXED_MASK),
]:
    correct_cos = []; wrong_cos = []
    for src, tgt in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        pred = es + s * ax_dir
        snap_w, _, snap_idx = nn_ret(pred, source_ids(src), mask)
        snap_emb = W_E[snap_idx].copy()
        residual = pred - snap_emb
        cos = float(np.dot(normed(residual).astype(np.float32), ax_dir.astype(np.float32)))
        if snap_w == tgt: correct_cos.append(cos)
        else:             wrong_cos.append(cos)
    print("  %-8s: correct_cos=%.4f±%.4f (n=%d)  wrong_cos=%.4f±%.4f (n=%d)" % (
        label,
        np.mean(correct_cos) if correct_cos else 0,
        np.std(correct_cos)  if correct_cos else 0, len(correct_cos),
        np.mean(wrong_cos)   if wrong_cos  else 0,
        np.std(wrong_cos)    if wrong_cos  else 0, len(wrong_cos)))

# ============================================================
# PHASE 8: Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Day 353 Axis-Residual Decoding")
print("="*70)
print()
print("  Residual discriminator: cos(pred - snap, axis_dir)")
print("    HIGH → snap is wrong (M1: first token = base, more needs to be emitted)")
print("    LOW  → snap is correct")
print()
print("  Final accuracies with residual decoding:")
h_cs_best = 0
for thresh in np.linspace(0.0, 0.9, 19):
    h, n, _ = chain_residual(CS_PAIRS, comp_dir, s_c, comp_to_sup_dir, s_cs,
                              RELAXED_MASK, RELAXED_MASK, comp_to_sup_dir, threshold=thresh)
    if h > h_cs_best: h_cs_best = h; best_t_cs = thresh
h_gp_best = 0
for thresh in np.linspace(0.0, 0.9, 19):
    h, n, _ = chain_residual(GP_PAIRS, gender_dir, s_g, plural_dir, s_p,
                              RELAXED_MASK, ENGLISH_MASK, plural_dir, threshold=thresh)
    if h > h_gp_best: h_gp_best = h; best_t_gp = thresh

print("  comp→sup:     best=%d/12=%.0f%% (t=%.2f)" % (h_cs_best, 100*h_cs_best/12, best_t_cs))
print("  gender→plural: best=%d/10=%.0f%% (t=%.2f)" % (h_gp_best, 100*h_gp_best/10, best_t_gp))
