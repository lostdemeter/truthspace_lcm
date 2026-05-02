import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building masks...", flush=True)
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

def get_emb(word):
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

def compute_axis(pairs, mask=None):
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

def best_scale(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask, 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

def axis_loo(valid, mask):
    if len(valid) < 3: return 0.0, 0
    chords_full = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    ax_full = normed(np.mean(chords_full, axis=0))
    global_s, _ = best_scale(ax_full, valid, mask)
    hits = 0
    for i in range(len(valid)):
        test_s, test_t, test_sid, _ = valid[i]
        train_v = [valid[j] for j in range(len(valid)) if j != i]
        ax_loo = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in train_v], axis=0))
        r = nn_retrieve(W_E[test_sid]+global_s*ax_loo, source_ids(test_s), mask, 1)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid), len(valid)

print()
print("DAY 317: RELATIONAL AXES — HOLDOUT, REVERSE, CHAIN, NON-GEOMETRIC SEARCH, +less")
print("="*72)
print()

# Re-build axes from Day 316
COUNTRY_CAPITAL_TRAIN = [
    ('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
    ('china','Beijing'),('canada','Ottawa'),('australia','Canberra'),
    ('india','Delhi'),('russia','Moscow'),
]
COUNTRY_CAPITAL_HOLDOUT = [
    ('sweden','Stockholm'),('greece','Athens'),('portugal','Lisbon'),
    ('argentina','Buenos'),('norway','Oslo'),('denmark','Copenhagen'),
    ('finland','Helsinki'),('austria','Vienna'),
]

COUNTRY_LANG_TRAIN = [
    ('france','French'),('germany','German'),('japan','Japanese'),
    ('china','Chinese'),('turkey','Turkish'),('brazil','Portuguese'),
    ('sweden','Swedish'),('norway','Norwegian'),('poland','Polish'),
]

ax_cc, _, valid_cc, pc_cc = compute_axis(COUNTRY_CAPITAL_TRAIN)
ax_cl, _, valid_cl, pc_cl = compute_axis(COUNTRY_LANG_TRAIN)

# ====================================================================
# PART A: COUNTRY→CAPITAL HOLDOUT AND LOO
# ====================================================================
print("PART A: country→capital — holdout and LOO")
print("-"*72)

if ax_cc is not None:
    best_s_cc, in_cc = best_scale(ax_cc, valid_cc, RELAXED_MASK)
    loo_cc, _ = axis_loo(valid_cc, RELAXED_MASK)
    print("  Train: n=%d  pc=%.4f  in-sample=%d/n=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_cc), pc_cc, in_cc, 100*in_cc/len(valid_cc), 100*loo_cc, best_s_cc))

    # Holdout
    ho_hits = 0; ho_n = 0
    print("  Holdout pairs:")
    for s_w, t_w in COUNTRY_CAPITAL_HOLDOUT:
        es, sid = get_emb(s_w); et, tid = get_emb(t_w)
        if es is None: print("  ? %-12s  [not single token]" % s_w); continue
        ho_n += 1
        r = nn_retrieve(W_E[sid]+best_s_cc*ax_cc, source_ids(s_w), RELAXED_MASK, 3)
        hit = '✓' if r[0][0] == t_w else '✗'
        if r[0][0] == t_w: ho_hits += 1
        print("  %s %-12s -> %-14s  got: %s" % (hit, s_w, t_w, r[0][0]))
    print("  Holdout: %d/%d=%.0f%%" % (ho_hits, ho_n, 100*ho_hits/ho_n if ho_n else 0))
print()

# ====================================================================
# PART B: REVERSE AXIS — CAPITAL→COUNTRY
# ====================================================================
print("PART B: Reverse axis — capital→country")
print("-"*72)

CAPITAL_COUNTRY_TRAIN = [(t,s) for s,t in COUNTRY_CAPITAL_TRAIN]
ax_rev, _, valid_rev, pc_rev = compute_axis(CAPITAL_COUNTRY_TRAIN)

if ax_rev is not None:
    best_s_rev, in_rev = best_scale(ax_rev, valid_rev, CLEAN_MASK)
    loo_rev, _ = axis_loo(valid_rev, CLEAN_MASK)
    print("  capital->country: n=%d  pc=%.4f  in-sample=%d/n=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_rev), pc_rev, in_rev, 100*in_rev/len(valid_rev), 100*loo_rev, best_s_rev))
    print("  (clean mask because country names are lowercase)")
    # Per-pair
    for s_w, t_w, sid, tid in valid_rev:
        r = nn_retrieve(W_E[sid]+best_s_rev*ax_rev, source_ids(s_w), CLEAN_MASK, 3)
        hit = '✓' if r[0][0] == t_w else '✗'
        print("  %s %-14s -> %-12s  got: %s" % (hit, s_w, t_w, r[0][0]))

    # Reversibility: cos(forward_axis, -reverse_axis)
    if ax_cc is not None:
        cos_rev = float(np.dot(ax_cc.astype(np.float32), ax_rev.astype(np.float32)))
        print("  cos(cc_axis, reverse_axis) = %.4f  (perfect reversal = -1.0)" % cos_rev)
print()

# ====================================================================
# PART C: CHAIN COMPOSITION — capital→language (via two axes)
# ====================================================================
print("PART C: Chain composition — country→capital→language")
print("-"*72)

# Approach 1: Build capital->language axis directly
CAPITAL_LANG_TRAIN = [
    ('Paris','French'),('Berlin','German'),('Tokyo','Japanese'),
    ('Beijing','Chinese'),('Moscow','Russian'),('Rome','Italian'),
    ('Madrid','Spanish'),('Athens','Greek'),('Warsaw','Polish'),
]
ax_capl, _, valid_capl, pc_capl = compute_axis(CAPITAL_LANG_TRAIN)

if ax_capl is not None:
    best_s_capl, in_capl = best_scale(ax_capl, valid_capl, RELAXED_MASK)
    loo_capl, _ = axis_loo(valid_capl, RELAXED_MASK)
    print("  capital->language axis: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_capl), pc_capl, 100*in_capl/len(valid_capl), 100*loo_capl, best_s_capl))
    print()

# Approach 2: Chain country→capital then capital→language
print("  Chain test: country →[country_capital]→ city →[capital_lang]→ language")
if ax_cc is not None and ax_capl is not None:
    test_countries = [
        ('france','Paris','French'),
        ('germany','Berlin','German'),
        ('japan','Tokyo','Japanese'),
        ('china','Beijing','Chinese'),
        ('russia','Moscow','Russian'),
    ]
    chain_hits = 0; chain_n = 0
    for country, capital, language in test_countries:
        es, sid = get_emb(country)
        if es is None: continue
        chain_n += 1
        # Step 1: country → capital
        pred1 = W_E[sid] + best_s_cc * ax_cc
        r1 = nn_retrieve(pred1, source_ids(country), RELAXED_MASK, 1)
        city_got = r1[0][0]
        # Step 2: predicted capital → language
        ec, cid = get_emb(city_got)
        if ec is None:
            print("  %-10s -> %-10s [multi-token] -> n/a" % (country, city_got))
            continue
        pred2 = W_E[cid] + best_s_capl * ax_capl
        r2 = nn_retrieve(pred2, source_ids(city_got), RELAXED_MASK, 1)
        lang_got = r2[0][0]
        hit_cap = '✓' if city_got == capital else '✗'
        hit_lang = '✓' if lang_got == language else '✗'
        both = '✓✓' if city_got == capital and lang_got == language else ('✓✗' if city_got == capital else '✗?')
        if city_got == capital and lang_got == language: chain_hits += 1
        print("  %s %-10s →%s %-10s →%s %-12s (want: %s→%s)" %
              (both, country, hit_cap, city_got, hit_lang, lang_got, capital, language))
    print("  Both-correct: %d/%d=%.0f%%" % (chain_hits, chain_n, 100*chain_hits/chain_n if chain_n else 0))

    # Direct composition: country→language via ax_cc + ax_capl
    print()
    print("  Direct composition: country + (ax_cc + ax_capl) at combined scale")
    combined = normed(ax_cc + ax_capl)
    _, direct_in = best_scale(combined, valid_cl, RELAXED_MASK)
    print("  Combined axis in-sample on country_lang pairs: %d/%d=%.0f%%" %
          (direct_in, len(valid_cl), 100*direct_in/len(valid_cl)))
print()

# ====================================================================
# PART D: SEARCH FOR TRUE NON-GEOMETRIC RELATIONS
# ====================================================================
print("PART D: True non-geometric relations — can we find any?")
print("-"*72)

NON_GEO_CANDIDATES = {
    'country->president': [
        ('france','Macron'),('germany','Scholz'),('usa','Biden'),
        ('russia','Putin'),('china','Xi'),('brazil','Lula'),
        ('india','Modi'),('turkey','Erdogan'),('japan','Kishida'),
    ],
    'author->work': [
        ('Shakespeare','Hamlet'),('Tolstoy','Anna'),('Dickens','Oliver'),
        ('Kafka','Metamorphosis'),('Orwell','Nineteen'),('Hemingway','Farewell'),
    ],
    'scientist->field': [
        ('einstein','physics'),('darwin','biology'),('newton','mechanics'),
        ('turing','computing'),('curie','chemistry'),('freud','psychology'),
        ('darwin','evolution'),('mendel','genetics'),
    ],
    'color->fruit': [
        ('red','apple'),('yellow','banana'),('orange','orange'),
        ('purple','grape'),('green','lime'),('blue','blueberry'),
    ],
    'element->state': [
        ('hydrogen','gas'),('oxygen','gas'),('iron','solid'),
        ('mercury','liquid'),('nitrogen','gas'),('gold','solid'),
    ],
}

print("  %-22s  pc      n  in%%  (relaxed)  notes" % "relation")
print("  " + "-"*66)
for name, pairs in NON_GEO_CANDIDATES.items():
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-22s  n/a  <%d valid pairs" % (name, 2)); continue
    # Test with both masks
    _, in_clean = best_scale(ax, valid, CLEAN_MASK)
    _, in_relax = best_scale(ax, valid, RELAXED_MASK)
    in_best = max(in_clean, in_relax)
    mask_used = 'clean' if in_clean >= in_relax else 'relaxed'
    print("  %-22s  %.4f  %d  %.0f%%  [%s]" %
          (name, pc, len(valid), 100*in_best/len(valid), mask_used))
    # If any non-zero hits, show which ones
    best_s = (best_scale(ax, valid, CLEAN_MASK if in_clean >= in_relax else RELAXED_MASK)[0])
    mask = CLEAN_MASK if in_clean >= in_relax else RELAXED_MASK
    found = [(s,t) for s,t,sid,_ in valid
             if nn_retrieve(W_E[sid]+best_s*ax, source_ids(s), mask, 1)[0][0]==t
             for _,t_,_,_ in [(s,t,None,None)] if t_==t]
    # simpler:
    found_pairs = []
    for s_w, t_w, sid, tid in valid:
        r = nn_retrieve(W_E[sid]+best_s*ax, source_ids(s_w), mask, 1)
        if r[0][0] == t_w: found_pairs.append("%s->%s" % (s_w, t_w))
    if found_pairs:
        print("    found: " + ', '.join(found_pairs))
print()

# ====================================================================
# PART E: +less FULL SWEEP
# ====================================================================
print("PART E: +less full scale sweep — semantic_diverse confirmed?")
print("-"*72)

LESS_TRAIN = [('hope','hopeless'),('help','helpless'),('use','useless'),
              ('care','careless'),('harm','harmless'),('thought','thoughtless'),
              ('power','powerless')]
LESS_HOLDOUT = [('worth','worthless'),('home','homeless'),('job','jobless'),
                ('friend','friendless'),('speech','speechless'),('god','godless'),
                ('breath','breathless'),('tooth','toothless'),('bone','boneless'),
                ('arm','armless')]

ax_less, _, valid_less, pc_less = compute_axis(LESS_TRAIN)
if ax_less is not None:
    best_s_less, in_less = best_scale(ax_less, valid_less, CLEAN_MASK)
    loo_less, _ = axis_loo(valid_less, CLEAN_MASK)
    print("  +less train: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_less), pc_less, 100*in_less/len(valid_less), 100*loo_less, best_s_less))
    # Full sweep
    irred = 0; ho_n = 0
    for src, tgt in LESS_HOLDOUT:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: print("  ? %s [multi-token]" % src); continue
        ho_n += 1
        found_at = None
        for s_test in np.linspace(0.02, 6.0, 120):
            r = nn_retrieve(W_E[sid]+s_test*ax_less, source_ids(src), CLEAN_MASK, 1)
            if r[0][0] == tgt: found_at = s_test; break
        if found_at is not None:
            print("  ✓ %-12s -> %-14s  at scale %.3f" % (src, tgt, found_at))
        else:
            irred += 1
            r = nn_retrieve(W_E[sid]+best_s_less*ax_less, source_ids(src), CLEAN_MASK, 3)
            print("  ✗ %-12s -> %-14s  got: %s" % (src, tgt, r[0][0]))
    print()
    print("  +less irred: %d/%d=%.0f%%  → type=%s" %
          (irred, ho_n, 100*irred/ho_n if ho_n else 0,
           'phonol_scatter' if irred/ho_n < 0.3 else
           'morph_moderate' if irred/ho_n < 0.6 else 'semantic_diverse'))
print()

# ====================================================================
# PART F: AXIS ORTHOGONALITY — DO RELATIONAL AND MORPHOLOGICAL AXES DIFFER?
# ====================================================================
print("PART F: Axis orthogonality — relational vs morphological")
print("-"*72)

ER_PAIRS  = [('fast','faster'),('slow','slower'),('tall','taller'),
             ('short','shorter'),('bright','brighter'),('dark','darker'),
             ('deep','deeper'),('clean','cleaner')]
S_PAIRS   = [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
             ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')]
ED_PAIRS  = [('walk','walked'),('jump','jumped'),('play','played'),
             ('start','started'),('help','helped'),('work','worked'),
             ('turn','turned'),('push','pushed')]

ax_er, _, _, _ = compute_axis(ER_PAIRS)
ax_s,  _, _, _ = compute_axis(S_PAIRS)
ax_ed, _, _, _ = compute_axis(ED_PAIRS)

axes = {'cc': ax_cc, 'cl': ax_cl, '+er': ax_er, '+s': ax_s, '+ed': ax_ed}
names = list(axes.keys())

print("  Pairwise cosines between axes:")
print("  " + "  ".join("%-8s" % n for n in names))
for na in names:
    row = []
    for nb in names:
        if axes[na] is None or axes[nb] is None: row.append('   n/a  ')
        else:
            c = float(np.dot(axes[na].astype(np.float32), axes[nb].astype(np.float32)))
            row.append(" %+.3f " % c)
    print("  %-6s  %s" % (na, ' '.join(row)))
