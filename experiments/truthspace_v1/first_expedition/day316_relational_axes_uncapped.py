import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building token masks...", flush=True)
# Masks: clean (no caps, no compounds, len>1) and relaxed (no compounds, len>1)
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

def get_emb(word, allow_caps=False):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            w = tok.decode([ids[0]]).strip()
            if allow_caps or not w[0].isupper():
                return W_E[ids[0]].copy(), ids[0]
    # If caps not found with strict filter, try anyway
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

def compute_axis(pairs, allow_caps=False):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s, allow_caps=True)
        et, tid = get_emb(t, allow_caps=True)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, 0.0, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    coh = float(np.mean([np.dot(c, md.astype(np.float32)) for c in cn]))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, coh, valid, pc

def best_scale_mask(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask, 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

def axis_loo_mask(valid, mask):
    if len(valid) < 3: return 0.0, 0
    chords_full = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    ax_full = normed(np.mean(chords_full, axis=0))
    global_s, _ = best_scale_mask(ax_full, valid, mask)
    hits = 0
    for i in range(len(valid)):
        test_s, test_t, test_sid, _ = valid[i]
        train_v = [valid[j] for j in range(len(valid)) if j != i]
        ax_loo = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in train_v], axis=0))
        r = nn_retrieve(W_E[test_sid]+global_s*ax_loo, source_ids(test_s), mask, 1)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid), len(valid)

print()
print("DAY 316: RELATIONAL AXES WITH RELAXED FILTER, +able SWEEP, LOO GAP")
print("="*70)
print()

# ====================================================================
# PART A: COUNTRY→LANGUAGE WITHOUT CAPS FILTER
# ====================================================================
print("PART A: country→language with RELAXED filter (caps allowed)")
print("-"*70)

COUNTRY_LANG = [
    ('france','French'),('germany','German'),('japan','Japanese'),
    ('spain','Spanish'),('italy','Italian'),('russia','Russian'),
    ('china','Chinese'),('greece','Greek'),('poland','Polish'),
    ('sweden','Swedish'),('norway','Norwegian'),('turkey','Turkish'),
    ('brazil','Portuguese'),('vietnam','Vietnamese'),('thailand','Thai'),
]
COUNTRY_LANG_HOLDOUT = [
    ('india','Hindi'),('egypt','Arabic'),('iran','Persian'),
    ('mexico','Spanish'),('korea','Korean'),('ukraine','Ukrainian'),
]

ax_cl, _, valid_cl, pc_cl = compute_axis(COUNTRY_LANG)
if ax_cl is not None:
    # Test with clean mask (original)
    best_s_clean, in_clean = best_scale_mask(ax_cl, valid_cl, CLEAN_MASK)
    # Test with relaxed mask (allow caps)
    best_s_relax, in_relax = best_scale_mask(ax_cl, valid_cl, RELAXED_MASK)
    loo_relax, _ = axis_loo_mask(valid_cl, RELAXED_MASK)

    print("  country->language: n=%d  pc=%.4f" % (len(valid_cl), pc_cl))
    print("  Clean   mask: in-sample=%d/%d=%.0f%%  scale=%.3f" %
          (in_clean, len(valid_cl), 100*in_clean/len(valid_cl), best_s_clean))
    print("  Relaxed mask: in-sample=%d/%d=%.0f%%  scale=%.3f" %
          (in_relax, len(valid_cl), 100*in_relax/len(valid_cl), best_s_relax))
    print("  LOO (relaxed): %.0f%%" % (100*loo_relax))
    print()

    # Per-pair results at best relaxed scale
    print("  Per-pair (relaxed, scale=%.3f):" % best_s_relax)
    for s_w, t_w, sid, tid in valid_cl:
        pred = W_E[sid] + best_s_relax * ax_cl
        r = nn_retrieve(pred, source_ids(s_w), RELAXED_MASK, 3)
        hit = '✓' if r[0][0] == t_w else '✗'
        print("  %s %-10s -> %-12s  got: %s" % (hit, s_w, t_w, r[0][0]))
    print()

    # Holdout test
    print("  Holdout (relaxed, scale=%.3f):" % best_s_relax)
    ho_hits = 0; ho_n = 0
    for s_w, t_w in COUNTRY_LANG_HOLDOUT:
        es, sid = get_emb(s_w, allow_caps=True)
        et, tid = get_emb(t_w, allow_caps=True)
        if es is None: print("  ? %-10s [not single token]" % s_w); continue
        ho_n += 1
        pred = W_E[sid] + best_s_relax * ax_cl
        r = nn_retrieve(pred, source_ids(s_w), RELAXED_MASK, 3)
        hit = '✓' if r[0][0] == t_w else '✗'
        if r[0][0] == t_w: ho_hits += 1
        print("  %s %-10s -> %-12s  got: %s (cos %.3f)" %
              (hit, s_w, t_w, r[0][0], r[0][1]))
    print("  Holdout: %d/%d=%.0f%%" % (ho_hits, ho_n, 100*ho_hits/ho_n if ho_n else 0))
print()

# ====================================================================
# PART B: ELEMENT→SYMBOL WITHOUT CAPS / LENGTH FILTER
# ====================================================================
print("PART B: element→symbol — relax caps AND length filter")
print("-"*70)

# Build super-relaxed mask: allow any non-compound, non-empty token
SUPER_RELAXED = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w: continue
    if w.startswith('-') or w.startswith('_'): continue
    SUPER_RELAXED[i] = True
print("  Super-relaxed: %d tokens" % SUPER_RELAXED.sum())

ELEM_SYM = [
    ('hydrogen','H'),('helium','He'),('carbon','C'),('nitrogen','N'),
    ('oxygen','O'),('sodium','Na'),('iron','Fe'),('gold','Au'),
    ('silver','Ag'),('copper','Cu'),('potassium','K'),('calcium','Ca'),
]

ax_es, _, valid_es, pc_es = compute_axis(ELEM_SYM)
if ax_es is not None:
    print("  element->symbol: n=%d  pc=%.4f" % (len(valid_es), pc_es))
    for mask_name, mask in [('clean', CLEAN_MASK), ('relaxed', RELAXED_MASK), ('super', SUPER_RELAXED)]:
        best_s, in_s = best_scale_mask(ax_es, valid_es, mask)
        print("  %-8s mask: in-sample=%d/%d=%.0f%%  scale=%.3f" %
              (mask_name, in_s, len(valid_es), 100*in_s/len(valid_es), best_s))
    # Per-pair with super-relaxed
    best_s_sr, _ = best_scale_mask(ax_es, valid_es, SUPER_RELAXED)
    print("  Per-pair (super-relaxed, scale=%.3f):" % best_s_sr)
    for s_w, t_w, sid, tid in valid_es:
        pred = W_E[sid] + best_s_sr * ax_es
        r = nn_retrieve(pred, source_ids(s_w), SUPER_RELAXED, 3)
        hit = '✓' if r[0][0] == t_w else '✗'
        print("  %s %-12s -> %-5s  got: %s" % (hit, s_w, t_w, r[0][0]))
print()

# ====================================================================
# PART C: +able HOLDOUT SWEEP (verify phonol_scatter vs morph_moderate)
# ====================================================================
print("PART C: +able full scale sweep — phonol_scatter or morph_moderate?")
print("-"*70)

ABLE_TRAIN = [
    ('break','breakable'),('wash','washable'),('read','readable'),
    ('use','usable'),('move','movable'),('adjust','adjustable'),
    ('adapt','adaptable'),('accept','acceptable'),('avoid','avoidable'),
    ('change','changeable'),
]
ABLE_HOLDOUT = [
    ('manage','manageable'),('agree','agreeable'),('debate','debatable'),
    ('comfort','comfortable'),('reason','reasonable'),('achieve','achievable'),
    ('predict','predictable'),('replace','replaceable'),('rely','reliable'),
    ('trust','trustworthy'),
]

ax_ab, _, valid_ab, pc_ab = compute_axis(ABLE_TRAIN)
if ax_ab is not None:
    best_s, in_s = best_scale_mask(ax_ab, valid_ab, CLEAN_MASK)
    loo, _ = axis_loo_mask(valid_ab, CLEAN_MASK)
    print("  +able train: n=%d  pc=%.4f  in-sample=%.0f%%  LOO=%.0f%%  best_s=%.3f" %
          (len(valid_ab), pc_ab, 100*in_s/len(valid_ab), 100*loo, best_s))
    # Full holdout sweep (find IRREDUCIBLE words across all scales)
    irred_count = 0; ho_total = 0
    irred_words = []
    print("  Holdout sweep (full scale 0.02-6.0):")
    for src, tgt in ABLE_HOLDOUT:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: print("  ? %s [not single token]" % src); continue
        ho_total += 1
        found_at = None
        for s_test in np.linspace(0.02, 6.0, 120):
            pred = W_E[sid] + s_test * ax_ab
            r = nn_retrieve(pred, source_ids(src), CLEAN_MASK, 1)
            if tid is not None and r[0][0] == tgt:
                found_at = s_test; break
        if found_at is not None:
            print("  ✓ %-16s -> %-16s  at scale %.3f" % (src, tgt, found_at))
        else:
            irred_count += 1; irred_words.append(src)
            pred = W_E[sid] + best_s * ax_ab
            r = nn_retrieve(pred, source_ids(src), CLEAN_MASK, 3)
            print("  ✗ %-16s -> %-16s  got: %s" % (src, tgt, r[0][0]))
    print()
    print("  +able irred: %d/%d=%.0f%%  → type=%s" %
          (irred_count, ho_total, 100*irred_count/ho_total if ho_total else 0,
           'phonol_scatter' if irred_count/ho_total < 0.3 else
           'morph_moderate' if irred_count/ho_total < 0.6 else 'semantic_diverse'))
print()

# ====================================================================
# PART D: un- PER-FOLD vs GLOBAL SCALE LOO GAP
# ====================================================================
print("PART D: un- LOO gap — per-fold scale vs global scale")
print("-"*70)

UN_ADJ = [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('safe','unsafe'),
          ('wise','unwise'),('true','untrue'),('sure','unsure'),('clear','unclear'),('fit','unfit')]

ax_ua, _, valid_ua, pc_ua = compute_axis(UN_ADJ)
if ax_ua is not None:
    # Global scale LOO (fast)
    loo_global, _ = axis_loo_mask(valid_ua, CLEAN_MASK)
    # Per-fold scale LOO (accurate, but small n so fast enough)
    hits_perfold = 0
    for i in range(len(valid_ua)):
        test_s, test_t, test_sid, _ = valid_ua[i]
        train_v = [valid_ua[j] for j in range(len(valid_ua)) if j != i]
        ax_loo = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in train_v], axis=0))
        # Per-fold best scale on training
        best_s_fold, _ = best_scale_mask(ax_loo, train_v, CLEAN_MASK)
        r = nn_retrieve(W_E[test_sid]+best_s_fold*ax_loo, source_ids(test_s), CLEAN_MASK, 1)
        if r[0][0] == test_t: hits_perfold += 1
    loo_perfold = hits_perfold / len(valid_ua)
    print("  un-ADJ: pc=%.4f  LOO_global=%.0f%%  LOO_perfold=%.0f%%  gap=%.0f%%" %
          (pc_ua, 100*loo_global, 100*loo_perfold, 100*(loo_global-loo_perfold)))

# Also for +tion -ct
TION_CT = [('act','action'),('direct','direction'),('collect','collection'),
           ('connect','connection'),('protect','protection'),('select','selection'),
           ('inject','injection'),('reject','rejection'),('detect','detection'),
           ('infect','infection'),('inspect','inspection'),('correct','correction')]
ax_tc, _, valid_tc, pc_tc = compute_axis(TION_CT)
if ax_tc is not None:
    loo_global_tc, _ = axis_loo_mask(valid_tc, CLEAN_MASK)
    hits_pf = 0
    for i in range(len(valid_tc)):
        test_s, test_t, test_sid, _ = valid_tc[i]
        train_v = [valid_tc[j] for j in range(len(valid_tc)) if j != i]
        ax_loo = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in train_v], axis=0))
        best_s_fold, _ = best_scale_mask(ax_loo, train_v, CLEAN_MASK)
        r = nn_retrieve(W_E[test_sid]+best_s_fold*ax_loo, source_ids(test_s), CLEAN_MASK, 1)
        if r[0][0] == test_t: hits_pf += 1
    loo_pf_tc = hits_pf / len(valid_tc)
    print("  +tion -ct: pc=%.4f  LOO_global=%.0f%%  LOO_perfold=%.0f%%  gap=%.0f%%" %
          (pc_tc, 100*loo_global_tc, 100*loo_pf_tc, 100*(loo_global_tc-loo_pf_tc)))

# Comparative (+er) for reference
ER_PAIRS = [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
            ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
            ('hard','harder'),('warm','warmer'),('cool','cooler'),('sweet','sweeter')]
ax_er, _, valid_er, pc_er = compute_axis(ER_PAIRS)
if ax_er is not None:
    loo_global_er, _ = axis_loo_mask(valid_er, CLEAN_MASK)
    hits_pf_er = 0
    for i in range(len(valid_er)):
        test_s, test_t, test_sid, _ = valid_er[i]
        train_v = [valid_er[j] for j in range(len(valid_er)) if j != i]
        ax_loo = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in train_v], axis=0))
        best_s_fold, _ = best_scale_mask(ax_loo, train_v, CLEAN_MASK)
        r = nn_retrieve(W_E[test_sid]+best_s_fold*ax_loo, source_ids(test_s), CLEAN_MASK, 1)
        if r[0][0] == test_t: hits_pf_er += 1
    loo_pf_er = hits_pf_er / len(valid_er)
    print("  +er:      pc=%.4f  LOO_global=%.0f%%  LOO_perfold=%.0f%%  gap=%.0f%%" %
          (pc_er, 100*loo_global_er, 100*loo_pf_er, 100*(loo_global_er-loo_pf_er)))
print()

# ====================================================================
# PART E: COUNTRY→LANGUAGE COMPOSABILITY
# ====================================================================
print("PART E: Relational axis composability")
print("-"*70)

# Chain: country → language → capital
# Step 1: country -> language (using relaxed mask)
# Step 2: language -> (language_adj -> ?) — test if language tokens are navigable
# Step 3: Also test country -> capital with relaxed mask

COUNTRY_CAPITAL = [
    ('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
    ('spain','Madrid'),('italy','Rome'),('china','Beijing'),
    ('russia','Moscow'),('japan','Tokyo'),('canada','Ottawa'),
    ('australia','Canberra'),('brazil','Brasilia'),('india','Delhi'),
]

ax_cc, _, valid_cc, pc_cc = compute_axis(COUNTRY_CAPITAL)
if ax_cc is not None:
    print("  country->capital: n=%d  pc=%.4f" % (len(valid_cc), pc_cc))
    for mask_name, mask in [('clean', CLEAN_MASK), ('relaxed', RELAXED_MASK)]:
        best_s, in_s = best_scale_mask(ax_cc, valid_cc, mask)
        print("  %-8s: in-sample=%d/%d=%.0f%%  scale=%.3f" %
              (mask_name, in_s, len(valid_cc), 100*in_s/len(valid_cc), best_s))
    # Per-pair relaxed
    best_s_r, in_r = best_scale_mask(ax_cc, valid_cc, RELAXED_MASK)
    if in_r > 0:
        print("  Per-pair (relaxed, scale=%.3f):" % best_s_r)
        for s_w, t_w, sid, tid in valid_cc:
            pred = W_E[sid] + best_s_r * ax_cc
            r = nn_retrieve(pred, source_ids(s_w), RELAXED_MASK, 3)
            hit = '✓' if r[0][0] == t_w else '✗'
            print("  %s %-12s -> %-12s  got: %s" % (hit, s_w, t_w, r[0][0]))
print()

# Composition: if country→language works, try chaining
# country -> [country_lang_axis] -> language_name
# language_name -> [language_lang_axis] -> ? (what is the nearest neighbor of 'French' displaced?)
if ax_cl is not None:
    best_s_cl, _ = best_scale_mask(ax_cl, valid_cl, RELAXED_MASK)
    print("  Composition test: country -> language -> ???")
    print("  Apply country->language twice:")
    for s_w, t_w, sid, tid in valid_cl[:5]:
        # Step 1: country -> language
        pred1 = W_E[sid] + best_s_cl * ax_cl
        r1 = nn_retrieve(pred1, source_ids(s_w), RELAXED_MASK, 1)
        lang_token = r1[0][0]
        # Step 2: Apply same axis again from language position
        lang_emb, lang_id = get_emb(lang_token, allow_caps=True)
        if lang_emb is None: continue
        pred2 = lang_emb + best_s_cl * ax_cl
        r2 = nn_retrieve(pred2, source_ids(lang_token), RELAXED_MASK, 3)
        print("  %-10s -> %-12s -> %s" % (s_w, lang_token, ', '.join(w for w,_,_ in r2)))
    print()

# ====================================================================
# PART F: pc TABLE FOR ALL RELATIONAL AXES TESTED
# ====================================================================
print("PART F: pc comparison — morphological vs relational axes")
print("-"*70)

RELATIONAL_TESTS = {
    'country->lang':  COUNTRY_LANG,
    'country->capital': COUNTRY_CAPITAL,
    '+er':    [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
               ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner')],
    'er->est':[('faster','fastest'),('slower','slowest'),('taller','tallest'),
               ('shorter','shortest'),('brighter','brightest'),('darker','darkest')],
    '+s':     [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
               ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
}

print("  %-18s  pc      n  notes" % "axis")
print("  " + "-"*50)
for nm, pairs in RELATIONAL_TESTS.items():
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None: print("  %-18s  n/a" % nm); continue
    print("  %-18s  %.4f  %d" % (nm, pc, len(valid)))
