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

def build_axis(pairs):
    chords = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es)
    if not chords: return None
    return normed(np.mean(chords, axis=0))

def best_scale(ax_dir, pairs, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = 0
        for src, tgt in pairs:
            es, _ = get_emb(src)
            if es is None: continue
            w, _, _ = nn_ret(es + s*ax_dir, source_ids(src), mask)
            if w == tgt: c += 1
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

print("\nDAY 352: Multi-Token Barrier Investigation")
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

print("  scales: gender=%.3f  plural=%.3f  comp=%.3f  sup=%.3f  c->s=%.3f" % (
    s_g, s_p, s_c, s_s, s_cs))

# ============================================================
# PHASE 2: Tokenization audit of all composition target words
# ============================================================
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

print("\nPhase 2: Tokenization of all target words")
print("  [gender→plural chain targets]")
print("  %-12s  %-10s  %-8s  tokens" % ("intermediate", "final", "#toks"))
for src, inter, final in GP_PAIRS:
    ids_i, toks_i = tokenize_word(inter)
    ids_f, toks_f = tokenize_word(final)
    n_i = len(ids_i); n_f = len(ids_f)
    barrier = ' ← BARRIER' if n_f > 1 else ''
    print("  %-12s  %-10s  %-3d→%-3d  inter=%s  final=%s%s" % (
        inter, final, n_i, n_f, toks_i, toks_f, barrier))

print("\n  [comp→sup chain targets]")
for src, inter, final in CS_PAIRS:
    ids_i, toks_i = tokenize_word(inter)
    ids_f, toks_f = tokenize_word(final)
    n_f = len(ids_f)
    barrier = ' ← BARRIER' if n_f > 1 else ''
    print("  %-10s  %-10s  %-3d→%-3d  final=%s%s" % (
        inter, final, len(ids_i), n_f, toks_f, barrier))

# ============================================================
# PHASE 3: Where does the chain prediction land for F3 pairs?
# ============================================================
# For pairs where the final target is multi-token:
# - What single-token word does the chain predict?
# - How close is the prediction to the first sub-token of the target?
# - How close to a proxy embedding (mean of sub-tokens)?

print("\nPhase 3: Chain prediction proximity to multi-token targets")
print("  For F3 failures: measuring distance to first sub-token vs proxy embedding")
print()

F3_PAIRS = [
    ('aunt',     'aunts'),
    ('princess', 'princesses'),
    ('waitress', 'waitresses'),
    ('wife',     'wives'),
    ('mother',   'mothers'),
]

for inter_word, final_tgt in F3_PAIRS:
    ei, _ = get_emb(inter_word)
    if ei is None:
        print("  %s → (not single-token, skip)" % inter_word)
        continue

    pred = ei + s_p * plural_dir

    # NN with RELAXED
    nn_relax, sim_relax, _ = nn_ret(pred, source_ids(inter_word), RELAXED_MASK)
    # NN with ENGLISH
    nn_eng,   sim_eng,   _ = nn_ret(pred, source_ids(inter_word), ENGLISH_MASK)

    # Sub-tokens of target
    ids_f, toks_f = tokenize_word(final_tgt)
    n_f = len(ids_f)

    pred_n = normed(pred).astype(np.float32)

    # Sim to each sub-token
    sub_sims = []
    for tid in ids_f:
        sub_sims.append(float(np.dot(pred_n, W_n[tid])))

    # Proxy embedding: mean of sub-token embeddings
    proxy_emb = np.mean([W_E[tid] for tid in ids_f], axis=0)
    proxy_sim = float(np.dot(pred_n, normed(proxy_emb).astype(np.float32)))

    # Sim to whole word if single-token
    exact_sim = None
    if n_f == 1:
        exact_sim = float(np.dot(pred_n, W_n[ids_f[0]]))

    print("  %s → %s  (%d tokens: %s)" % (inter_word, final_tgt, n_f, toks_f))
    print("    chain→RELAXED: %-14s (sim=%.4f)" % (nn_relax, sim_relax))
    print("    chain→ENGLISH: %-14s (sim=%.4f)" % (nn_eng, sim_eng))
    print("    sim to sub-tokens: %s" % ["tok%d=%s(%.4f)" % (i, toks_f[i], s)
                                          for i, s in enumerate(sub_sims)])
    print("    sim to proxy_mean: %.4f  (vs best NN sim=%.4f)" % (proxy_sim, sim_relax))
    if exact_sim is not None:
        print("    sim to target (single-token): %.4f" % exact_sim)
    print()

# ============================================================
# PHASE 4: Proxy-extended vocabulary
# ============================================================
# Build a supplementary lookup table: for each multi-token word in our test set,
# compute proxy_emb = mean(sub-token embeddings) and check if chain prediction
# is closer to proxy_emb than to any single-token word.

print("Phase 4: Proxy embedding lookup for multi-token targets")
print("  Strategy: can the chain predict multi-token targets via proxy=mean(sub-tokens)?")
print()

MULTI_TOKEN_TARGETS = [
    'aunts', 'princesses', 'waitresses',
    'daughters', 'actresses', 'mothers', 'queens', 'girls', 'women', 'wives',
    'grandmothers', 'goddesses',
]

proxy_table = {}
for word in MULTI_TOKEN_TARGETS:
    ids, toks = tokenize_word(word)
    if len(ids) <= 1: continue  # already single-token
    proxy_emb = np.mean([W_E[tid] for tid in ids], axis=0)
    proxy_table[word] = (normed(proxy_emb).astype(np.float32), ids, toks)

print("  Multi-token words found: %s" % [w for w in proxy_table])

def nn_ret_with_proxy(pred_emb, excl_ids, mask, proxy_table):
    """NN retrieval extended with proxy embeddings for multi-token words."""
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    best_single_idx = int(np.argmax(sims))
    best_single_sim = float(sims[best_single_idx])
    best_single_word = tok.decode([best_single_idx]).strip()

    best_proxy_word = None; best_proxy_sim = -1.0
    for word, (proxy_n, ids, toks) in proxy_table.items():
        s = float(np.dot(pred_n, proxy_n))
        if s > best_proxy_sim:
            best_proxy_sim = s; best_proxy_word = word

    if best_proxy_sim > best_single_sim:
        return best_proxy_word, best_proxy_sim, 'proxy'
    else:
        return best_single_word, best_single_sim, 'single'

print("\n  gender→plural chain WITH proxy extension:")
for src, inter, final in GP_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    pred_inter = es + s_g * gender_dir
    inter_word, _, _ = nn_ret(pred_inter, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    if ei is None: continue
    pred_final = ei + s_p * plural_dir
    result, sim, rtype = nn_ret_with_proxy(pred_final, source_ids(inter_word),
                                            ENGLISH_MASK, proxy_table)
    ok = (result == final)
    mark = '✓' if ok else '✗'
    print("  %s %-9s → %-12s → %-16s [%s, sim=%.4f]  expected=%s" % (
        mark, src, inter_word, result, rtype, sim, final))

# ============================================================
# PHASE 5: comp→sup chain failure diagnosis
# ============================================================
# 7/12=58%% succeed. What are the 5 failures?
# Are they multi-token? Wrong intermediate? Wrong final?

print("\nPhase 5: comp→sup chain failure diagnosis")
print()
for src, inter, final in CS_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    # Step 1
    pred_inter = es + s_c * comp_dir
    inter_word, inter_sim, _ = nn_ret(pred_inter, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    step1_ok = (inter_word == inter)
    # Step 2
    step2_ok = False; final_word = '?'
    if ei is not None:
        pred_final = ei + s_cs * comp_to_sup_dir
        final_word, final_sim, _ = nn_ret(pred_final, source_ids(inter_word), RELAXED_MASK)
        step2_ok = (final_word == final)

    # Tokenization of final
    ids_f, toks_f = tokenize_word(final)
    n_f = len(ids_f)

    s1 = '✓' if step1_ok else '✗'
    s2 = '✓' if step2_ok else '✗'
    ftype = 'SUCCESS' if (step1_ok and step2_ok) else (
        'F3-multi' if n_f > 1 else
        'F1-wrong-inter' if not step1_ok else
        'F2-wrong-final')
    print("  %s%s %-9s → %-12s → %-14s  expected=%-14s  [%s, final_ntok=%d]" % (
        s1, s2, src, inter_word, final_word, final, ftype, n_f))

# ============================================================
# PHASE 6: Direct plural axis applied to female forms
# ============================================================
# Do the F3 targets (aunts, princesses, waitresses) exist in a reachable
# sub-region if we apply the PLURAL axis at a DIFFERENT scale?
# The scale s_p=0.429 was calibrated on common nouns.
# Maybe the scale for less common derived forms (waitress→waitresses) is different.

print("\nPhase 6: Scale sweep for F3 targets (direct plural from female word)")
print("  Finding optimal scale for each F3 female word → target")
print()

F3_SCALE_TESTS = [
    ('aunt',     'aunts'),
    ('princess', 'princesses'),
    ('waitress', 'waitresses'),
    ('wife',     'wives'),
    ('mother',   'mothers'),
    ('daughter', 'daughters'),
]

for inter_word, final_tgt in F3_SCALE_TESTS:
    ei, _ = get_emb(inter_word)
    if ei is None:
        print("  %s → skip (not single-token)" % inter_word)
        continue

    ids_f, toks_f = tokenize_word(final_tgt)
    n_f = len(ids_f)

    # For single-token targets: find optimal scale
    if n_f == 1:
        best_s = None; best_sim = -1.0
        for s in np.linspace(0.02, 10.0, 50):
            pred = ei + s * plural_dir
            pred_n = normed(pred).astype(np.float32)
            sim = float(np.dot(pred_n, W_n[ids_f[0]]))
            if sim > best_sim: best_sim = sim; best_s = s
        # What does NN give at optimal scale?
        pred_opt = ei + best_s * plural_dir
        nn_opt, sim_opt, _ = nn_ret(pred_opt, source_ids(inter_word), ENGLISH_MASK)
        nn_relax, sim_relax, _ = nn_ret(pred_opt, source_ids(inter_word), RELAXED_MASK)
        print("  %-12s → %-14s  optimal_s=%.3f (base_s=%.3f)  "
              "best_sim_to_target=%.4f  nn@opt=%s(%s)" % (
            inter_word, final_tgt, best_s, s_p, best_sim,
            nn_opt, nn_relax))
    else:
        # For multi-token: check sim to first sub-token and proxy at each scale
        best_s_proxy = None; best_proxy_sim = -1.0
        best_s_first = None; best_first_sim = -1.0
        first_id = ids_f[0]
        for s in np.linspace(0.02, 10.0, 50):
            pred = ei + s * plural_dir
            pred_n = normed(pred).astype(np.float32)
            sim_first = float(np.dot(pred_n, W_n[first_id]))
            proxy_emb = np.mean([W_E[tid] for tid in ids_f], axis=0)
            sim_proxy = float(np.dot(pred_n, normed(proxy_emb).astype(np.float32)))
            if sim_first > best_first_sim: best_first_sim = sim_first; best_s_first = s
            if sim_proxy > best_proxy_sim: best_proxy_sim = sim_proxy; best_s_proxy = s

        # NN at s_p
        pred_base = ei + s_p * plural_dir
        nn_base, sim_base, _ = nn_ret(pred_base, source_ids(inter_word), ENGLISH_MASK)
        print("  %-12s → %-14s [%d toks: %s]" % (inter_word, final_tgt, n_f, toks_f))
        print("    @s_p=%.3f: nn=%s sim=%.4f" % (s_p, nn_base, sim_base))
        print("    best s for first_token sim: s=%.3f → sim_first=%.4f  (tok=%s)" % (
            best_s_first, best_first_sim, tok.decode([first_id])))
        print("    best s for proxy_mean sim:  s=%.3f → sim_proxy=%.4f" % (
            best_s_proxy, best_proxy_sim))

# ============================================================
# PHASE 7: Can we predict the FIRST sub-token of multi-token targets?
# ============================================================
# Test: is the first sub-token of "aunts", "princesses", "waitresses"
# predictable from aunt_emb + s_p * plural_dir?
# i.e., if we just limit to single-token retrieval but consider the first
# sub-token as a valid "match", does the chain succeed?

print("\nPhase 7: First-token prediction accuracy")
print("  Testing: chain → first_sub_token of multi-token target")
print()

hits_first = 0; n_total = 0
for src, inter, final in GP_PAIRS:
    es, _ = get_emb(src)
    if es is None: continue
    n_total += 1
    # Step 1: gender
    pred_inter = es + s_g * gender_dir
    inter_word, _, _ = nn_ret(pred_inter, source_ids(src), RELAXED_MASK)
    ei, _ = get_emb(inter_word)
    if ei is None: continue
    # Step 2: plural
    pred_final = ei + s_p * plural_dir
    nn_word, _, _ = nn_ret(pred_final, source_ids(inter_word), ENGLISH_MASK)

    # Sub-tokens of final target
    ids_f, toks_f = tokenize_word(final)
    first_subtok = toks_f[0].strip() if toks_f else ''
    exact_match = (nn_word == final)
    first_match = (nn_word == first_subtok and not exact_match)
    full_match = exact_match or first_match

    if exact_match: hits_first += 1

    mark_exact = '✓' if exact_match else ('≈' if first_match else '✗')
    print("  %s %-9s → %-12s → %-14s  target=%-14s  first_tok=%s  %s" % (
        mark_exact, src, inter_word, nn_word, final, first_subtok,
        '[first-tok match]' if first_match else ''))

# ============================================================
# PHASE 8: Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Day 352 Multi-Token Barrier")
print("="*70)
print()
print("  F3 failures arise because target words are multi-token in Qwen2.")
print("  The chain can perfectly retrieve the female intermediate (100%),")
print("  but cannot snap to a multi-token target.")
print()
print("  Key questions answered:")
print("  1. Which targets are multi-token? (Phase 2)")
print("  2. Does proxy embedding help? (Phase 4)")
print("  3. Are comp→sup failures also multi-token? (Phase 5)")
print("  4. Is there an optimal scale for reaching F3 targets? (Phase 6)")
print("  5. Can we predict the first sub-token? (Phase 7)")
