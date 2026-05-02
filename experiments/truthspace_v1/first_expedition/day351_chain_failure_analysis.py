import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

# ---- vocabulary masks ----
RELAXED_MASK  = np.zeros(len(W_E), dtype=bool)  # any single-token word > 1 char
ENGLISH_MASK  = np.zeros(len(W_E), dtype=bool)  # purely ASCII alphabetic
TOP5K_MASK    = np.zeros(len(W_E), dtype=bool)  # first 5000 token IDs (roughly highest freq in Qwen)
TOP10K_MASK   = np.zeros(len(W_E), dtype=bool)

for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) < 2: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
    if w.isalpha() and w.isascii(): ENGLISH_MASK[i] = True

# Qwen2 vocabulary: tokens 0..N are roughly sorted by frequency in training.
# We use index-based cutoffs as a frequency proxy.
for i in range(min(5000, len(W_E))):
    w = tok.decode([i]).strip()
    if w and len(w) >= 2: TOP5K_MASK[i] = True
for i in range(min(10000, len(W_E))):
    w = tok.decode([i]).strip()
    if w and len(w) >= 2: TOP10K_MASK[i] = True

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

def nn_ret_top(pred_emb, excl_ids, mask, top_k=1):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    top_ids = np.argsort(sims)[::-1][:top_k*3]
    results = []
    for idx in top_ids:
        if len(results) >= top_k: break
        results.append((tok.decode([int(idx)]).strip(), float(sims[idx])))
    return results

def nn_ret(pred_emb, excl_ids, mask):
    return nn_ret_top(pred_emb, excl_ids, mask, 1)[0][0]

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
# AXIS DATA (same as Days 348-350)
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

print("\nDAY 351: Chain Failure Analysis")
print("="*70)

print("\nPhase 1: Building axes...")
gender_dir, gender_v = build_axis(GENDER)
plural_dir, plural_v = build_axis(PLURAL)
comp_dir,   comp_v   = build_axis(ER_COMP)
sup_dir,    sup_v    = build_axis(ER_SUP)

s_g = best_scale(gender_dir, gender_v, RELAXED_MASK)
s_p = best_scale(plural_dir, plural_v, RELAXED_MASK)
s_c = best_scale(comp_dir,   comp_v,   RELAXED_MASK)
s_s = best_scale(sup_dir,    sup_v,    RELAXED_MASK)
comp_to_sup_dir = normed(sup_dir * s_s - comp_dir * s_c)
s_cs = np.linalg.norm(sup_dir * s_s - comp_dir * s_c)

print("  scales: gender=%.3f  plural=%.3f  comp=%.3f  sup=%.3f" % (s_g, s_p, s_c, s_s))

# ============================================================
# PHASE 2: Detailed chain step diagnostics
# ============================================================
# For each (src, final_tgt) pair in a composition:
#   Step 1: predicted_intermediate = src_emb + s_A * dir_A
#           actual_intermediate   = nn_ret(predicted_intermediate, ...)
#           step1_ok = (actual_intermediate == true_intermediate)
#           step1_sim = sim(predicted, actual)   confidence
#           true_sim  = sim(predicted, W_E[true_intermediate])  if true is single-token
#
#   Step 2: predicted_final = actual_intermediate_emb + s_B * dir_B
#           actual_final    = nn_ret(predicted_final, ...)
#           step2_ok = (actual_final == final_tgt)
#
# Failure types:
#   F1: Step 1 wrong (wrong intermediate)
#   F2: Step 1 right but Step 2 wrong
#   F3: Step 1 right but final_tgt is multi-token (can't succeed)
#   S:  Both steps correct

def chain_detail(src, true_intermediate, final_tgt, dir_A, s_A, dir_B, s_B,
                 mask_A, mask_B):
    """
    Returns dict with full chain step diagnostics.
    true_intermediate: expected word after step A (None if it's fine to snap anywhere)
    """
    es, src_id = get_emb(src)
    if es is None: return None

    # Step 1
    pred_inter = es + s_A * dir_A
    top1_inter = nn_ret_top(pred_inter, source_ids(src), mask_A, top_k=5)
    actual_inter_word = top1_inter[0][0]
    actual_inter_sim  = top1_inter[0][1]

    ea, inter_id = get_emb(actual_inter_word)
    step1_ok = (true_intermediate is None or actual_inter_word == true_intermediate)

    # Sim to true intermediate (if known and single-token)
    true_inter_sim = None
    if true_intermediate is not None:
        et_inter, _ = get_emb(true_intermediate)
        if et_inter is not None:
            true_inter_sim = float(np.dot(normed(pred_inter).astype(np.float32),
                                          normed(et_inter).astype(np.float32)))

    # Step 2 (from actual intermediate)
    step2_ok = False
    actual_final = None
    step2_sim = None
    if ea is not None:
        pred_final = ea + s_B * dir_B
        top1_final = nn_ret_top(pred_final, source_ids(actual_inter_word), mask_B, top_k=5)
        actual_final = top1_final[0][0]
        step2_sim    = top1_final[0][1]
        step2_ok     = (actual_final == final_tgt)

    # Is final_tgt single-token?
    final_single = get_emb(final_tgt)[0] is not None

    # Failure type
    if step1_ok and step2_ok:
        ftype = 'SUCCESS'
    elif not final_single:
        ftype = 'F3-multi-token-target'
    elif not step1_ok:
        ftype = 'F1-wrong-intermediate'
    else:
        ftype = 'F2-wrong-final'

    return {
        'src': src, 'true_inter': true_intermediate, 'final_tgt': final_tgt,
        'actual_inter': actual_inter_word, 'inter_sim': actual_inter_sim,
        'true_inter_sim': true_inter_sim,
        'actual_final': actual_final, 'final_sim': step2_sim,
        'step1_ok': step1_ok, 'step2_ok': step2_ok,
        'final_single': final_single, 'ftype': ftype,
        'top5_inter': [w for w,s in top1_inter],
    }

# ============================================================
# COMPOSITION TEST SETS WITH TRUE INTERMEDIATES
# ============================================================

# gender then plural: true intermediate = gender-flipped word
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

# plural then gender: true intermediate = pluralised word
PG_PAIRS = [
    ('man',  'men',   'women'),
    ('boy',  'boys',  'girls'),
    ('son',  'sons',  'daughters'),
    ('king', 'kings', 'queens'),
]

# comp then comp-to-sup: true intermediate = comparative form
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

print("\nPhase 2: Per-pair chain diagnostics")
print("-"*70)

for test_name, pairs, dir_A, s_A, dir_B, s_B, mask_A, mask_B in [
    ("gender → plural [cross-family]",
     GP_PAIRS, gender_dir, s_g, plural_dir, s_p, RELAXED_MASK, RELAXED_MASK),
    ("plural → gender [cross-family]",
     PG_PAIRS, plural_dir, s_p, gender_dir, s_g, RELAXED_MASK, RELAXED_MASK),
    ("comp → comp-to-sup [same-family]",
     CS_PAIRS, comp_dir, s_c, comp_to_sup_dir, s_cs, RELAXED_MASK, RELAXED_MASK),
]:
    print("\n  %s" % test_name)
    print("  %-9s %-12s %-14s %-14s %-6s %-6s  %-28s" % (
        "src", "true_inter", "actual_inter", "final_tgt", "s1", "s2", "type"))
    type_counts = {}
    for src, true_inter, final_tgt in pairs:
        d = chain_detail(src, true_inter, final_tgt, dir_A, s_A, dir_B, s_B,
                         mask_A, mask_B)
        if d is None: continue
        s1 = '✓' if d['step1_ok'] else '✗'
        s2 = '✓' if d['step2_ok'] else '✗'
        print("  %-9s %-12s %-14s %-14s %-6s %-6s  %-28s" % (
            src, true_inter, d['actual_inter']+'(%.2f)'%d['inter_sim'],
            d['actual_final'] or '?', s1, s2, d['ftype']))
        type_counts[d['ftype']] = type_counts.get(d['ftype'], 0) + 1
    print("  Failure type summary: %s" % type_counts)

# ============================================================
# PHASE 3: Vocabulary mask experiments
# ============================================================
# Test whether restricting the intermediate retrieval vocabulary
# improves chain accuracy by avoiding the Chinese cluster.

def chain_with_masks(pairs, dir_A, s_A, dir_B, s_B, mask_A, mask_B):
    hits = 0; n = 0
    for src, true_inter, final_tgt in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred_inter = es + s_A * dir_A
        inter_word = nn_ret(pred_inter, source_ids(src), mask_A)
        ei, _ = get_emb(inter_word)
        if ei is None: continue
        pred_final = ei + s_B * dir_B
        final_word = nn_ret(pred_final, source_ids(inter_word), mask_B)
        if final_word == final_tgt: hits += 1
    return hits, n

print("\nPhase 3: Vocabulary mask experiments on gender→plural chain")
print("  Testing different masks for the intermediate (step 1) retrieval\n")

masks = [
    ("RELAXED × RELAXED   (baseline)",    RELAXED_MASK,  RELAXED_MASK),
    ("ENGLISH × RELAXED   (en step1)",    ENGLISH_MASK,  RELAXED_MASK),
    ("ENGLISH × ENGLISH   (en both)",     ENGLISH_MASK,  ENGLISH_MASK),
    ("TOP5K   × RELAXED   (freq step1)",  TOP5K_MASK,    RELAXED_MASK),
    ("TOP10K  × RELAXED   (freq step1)",  TOP10K_MASK,   RELAXED_MASK),
    ("RELAXED × ENGLISH   (en step2)",    RELAXED_MASK,  ENGLISH_MASK),
]

for label, m1, m2 in masks:
    h, n = chain_with_masks(GP_PAIRS, gender_dir, s_g, plural_dir, s_p, m1, m2)
    print("  %-40s  %d/%d = %.0f%%" % (label, h, n, 100*h/max(n,1)))

print("\n  Gender→plural detail with ENGLISH mask (step 1):")
for src, true_inter, final_tgt in GP_PAIRS:
    d = chain_detail(src, true_inter, final_tgt,
                     gender_dir, s_g, plural_dir, s_p,
                     ENGLISH_MASK, RELAXED_MASK)
    if d is None: continue
    mark = '✓' if d['step2_ok'] else '✗'
    print("  %s %-9s → %-12s → %-14s (expected %s)" % (
        mark, src, d['actual_inter'], d['actual_final'], final_tgt))

print("\n  Comp→comp-to-sup with ENGLISH mask (step 1):")
h_cs_eng, n_cs_eng = chain_with_masks(CS_PAIRS, comp_dir, s_c, comp_to_sup_dir, s_cs,
                                       ENGLISH_MASK, RELAXED_MASK)
h_cs_rel, n_cs_rel = chain_with_masks(CS_PAIRS, comp_dir, s_c, comp_to_sup_dir, s_cs,
                                       RELAXED_MASK, RELAXED_MASK)
print("  RELAXED: %d/%d = %.0f%%  |  ENGLISH step1: %d/%d = %.0f%%" % (
    h_cs_rel, n_cs_rel, 100*h_cs_rel/max(n_cs_rel,1),
    h_cs_eng, n_cs_eng, 100*h_cs_eng/max(n_cs_eng,1)))

# ============================================================
# PHASE 4: Intermediate confidence scoring
# ============================================================
# Key question: can we PREDICT before running whether chain step 1 will succeed?
# Confidence = sim(pred_inter, nn_inter) vs sim(pred_inter, true_inter)
# Margin = sim(nn_inter) - sim(true_inter)
# If margin > 0: NN beats true → step 1 WRONG
# If margin < 0: true beats NN → step 1 RIGHT (but can only check if true is single-token)

print("\nPhase 4: Intermediate confidence scoring")
print("  Margin = sim(pred, nn_best) − sim(pred, true_intermediate)")
print("  Negative margin → true intermediate is closest → step 1 CORRECT\n")

for test_name, pairs, dir_A, s_A in [
    ("gender→plural", GP_PAIRS, gender_dir, s_g),
    ("plural→gender", PG_PAIRS, plural_dir, s_p),
    ("comp→sup",      CS_PAIRS, comp_dir,   s_c),
]:
    print("  %s:" % test_name)
    correct_margins = []
    wrong_margins   = []
    for src, true_inter, _ in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        pred = es + s_A * dir_A
        top = nn_ret_top(pred, source_ids(src), RELAXED_MASK, top_k=1)
        nn_sim = top[0][1]

        et, _ = get_emb(true_inter)
        if et is None: continue
        true_sim = float(np.dot(normed(pred).astype(np.float32),
                                normed(et).astype(np.float32)))
        margin = nn_sim - true_sim
        step1_ok = (top[0][0] == true_inter)
        if step1_ok:
            correct_margins.append(margin)
        else:
            wrong_margins.append(margin)

        print("    %-9s → %-12s  nn=%-12s  nn_sim=%.4f  true_sim=%.4f  margin=%+.4f  %s" % (
            src, true_inter, top[0][0], nn_sim, true_sim, margin,
            '✓' if step1_ok else '✗'))
    print("    Correct step1 margins: mean=%+.4f (n=%d)" % (
        np.mean(correct_margins) if correct_margins else 0, len(correct_margins)))
    print("    Wrong step1 margins:   mean=%+.4f (n=%d)" % (
        np.mean(wrong_margins) if wrong_margins else 0, len(wrong_margins)))
    print()

# ============================================================
# PHASE 5: Top-K chain (beam search)
# ============================================================
# Instead of committing to top-1 intermediate, take top-K,
# apply step B to each, then pick the best final by highest sim.

def chain_topK(pairs, dir_A, s_A, dir_B, s_B, mask_A, mask_B, K=5):
    hits = 0; n = 0; details = []
    for src, true_inter, final_tgt in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred_inter = es + s_A * dir_A
        top_inters = nn_ret_top(pred_inter, source_ids(src), mask_A, top_k=K)

        best_final = None; best_sim = -1.0
        for inter_word, _ in top_inters:
            ei, _ = get_emb(inter_word)
            if ei is None: continue
            pred_final = ei + s_B * dir_B
            top_finals = nn_ret_top(pred_final, source_ids(inter_word), mask_B, top_k=1)
            if top_finals[0][1] > best_sim:
                best_sim   = top_finals[0][1]
                best_final = top_finals[0][0]

        ok = (best_final == final_tgt)
        if ok: hits += 1
        details.append((src, final_tgt, best_final, ok))
    return hits, n, details

print("Phase 5: Top-K chain (beam search over top-K intermediates)")
print()
for test_name, pairs, dir_A, s_A, dir_B, s_B in [
    ("gender→plural", GP_PAIRS, gender_dir, s_g, plural_dir, s_p),
    ("plural→gender", PG_PAIRS, plural_dir, s_p, gender_dir, s_g),
    ("comp→sup",      CS_PAIRS, comp_dir,   s_c, comp_to_sup_dir, s_cs),
]:
    print("  %s:" % test_name)
    for K in [1, 3, 5, 10]:
        h, n, _ = chain_topK(pairs, dir_A, s_A, dir_B, s_B,
                              RELAXED_MASK, RELAXED_MASK, K)
        print("    K=%-3d  %d/%d = %.0f%%" % (K, h, n, 100*h/max(n,1)))
    # Also test english-restricted top-K
    h_en, n_en, _ = chain_topK(pairs, dir_A, s_A, dir_B, s_B,
                                ENGLISH_MASK, RELAXED_MASK, K=5)
    print("    K=5 (ENGLISH intermediate): %d/%d = %.0f%%" % (
        h_en, n_en, 100*h_en/max(n_en,1)))
    print()

# ============================================================
# PHASE 6: Semantic zone of failures
# ============================================================
# Do failures cluster in specific semantic categories?
# For gender→plural failures: are they kinship, royalty, profession, or other?

print("Phase 6: Semantic zone of chain failures (gender→plural)")
print("  Checking where the intermediate (step 1) lands for failure cases\n")

categories = {
    'kinship':   ['father','mother','son','daughter','husband','wife','brother','sister'],
    'royalty':   ['king','queen','prince','princess','duke','duchess','lord','lady'],
    'profession':['actor','actress','waiter','waitress','doctor','nurse','teacher'],
    'basic':     ['man','woman','boy','girl','uncle','aunt'],
}

def categorise(word):
    for cat, words in categories.items():
        if word in words: return cat
    return 'other'

print("  %-9s %-12s %-14s %-12s %-12s" % (
    "src", "true_inter", "actual_inter", "src_cat", "type"))
for src, true_inter, final_tgt in GP_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    pred_inter = es + s_g * gender_dir
    top = nn_ret_top(pred_inter, source_ids(src), RELAXED_MASK, top_k=1)
    actual_inter = top[0][0]
    step1_ok = (actual_inter == true_inter)

    # Is intermediate Chinese?
    is_chinese = not actual_inter.isascii() if not step1_ok else False
    note = ' [Chinese]' if is_chinese else ''

    d = chain_detail(src, true_inter, final_tgt,
                     gender_dir, s_g, plural_dir, s_p, RELAXED_MASK, RELAXED_MASK)
    print("  %-9s %-12s %-14s %-12s %s%s" % (
        src, true_inter, actual_inter, categorise(src), d['ftype'] if d else '?', note))

# ============================================================
# PHASE 7: Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Day 351 Chain Failure Analysis")
print("="*70)

print("\n  Base chain accuracy (RELAXED mask):")
for test_name, pairs, dir_A, s_A, dir_B, s_B in [
    ("gender→plural", GP_PAIRS, gender_dir, s_g, plural_dir, s_p),
    ("plural→gender", PG_PAIRS, plural_dir, s_p, gender_dir, s_g),
    ("comp→sup",      CS_PAIRS, comp_dir,   s_c, comp_to_sup_dir, s_cs),
]:
    h, n = chain_with_masks(pairs, dir_A, s_A, dir_B, s_B, RELAXED_MASK, RELAXED_MASK)
    print("    %-20s  %d/%d = %.0f%%" % (test_name, h, n, 100*h/max(n,1)))

print("\n  Best improved chain accuracy:")
for test_name, pairs, dir_A, s_A, dir_B, s_B in [
    ("gender→plural", GP_PAIRS, gender_dir, s_g, plural_dir, s_p),
    ("plural→gender", PG_PAIRS, plural_dir, s_p, gender_dir, s_g),
    ("comp→sup",      CS_PAIRS, comp_dir,   s_c, comp_to_sup_dir, s_cs),
]:
    h_en, n_en = chain_with_masks(pairs, dir_A, s_A, dir_B, s_B, ENGLISH_MASK, RELAXED_MASK)
    h_k5, _, _ = chain_topK(pairs, dir_A, s_A, dir_B, s_B, RELAXED_MASK, RELAXED_MASK, K=5)
    h_k5e, _, _= chain_topK(pairs, dir_A, s_A, dir_B, s_B, ENGLISH_MASK, RELAXED_MASK, K=5)
    print("    %-20s  ENGLISH_step1=%d/%d=%.0f%%  top5=%d/%d=%.0f%%  top5+EN=%d/%d=%.0f%%" % (
        test_name, h_en, n_en, 100*h_en/max(n_en,1),
        h_k5, n_en, 100*h_k5/max(n_en,1),
        h_k5e, n_en, 100*h_k5e/max(n_en,1)))
