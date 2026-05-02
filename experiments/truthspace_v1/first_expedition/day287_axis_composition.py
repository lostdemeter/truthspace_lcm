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
def nn_retrieve(pred_emb, exclude_ids, top_n=3):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, n_scales=50, lo=0.02, hi=4.0, reverse=False):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n_scales):
        if not reverse:
            correct = sum(1 for src,tgt,sid,tid in valid_pairs
                          if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==tgt)
        else:
            correct = sum(1 for src,tgt,sid,tid in valid_pairs
                          if nn_retrieve(W_E[tid]-s*axis,[tid])[0][0]==src)
        if correct > best_acc: best_acc=correct; best_s=s
    return best_s, best_acc

print("DAY 287: AXIS COMPOSITION")
print("="*65)
print("Testing whether derived axes can be computed from primitive ones.")
print("Key test: comp->sup = d_sup - d_comp (scaled appropriately)")
print()

# ====================================================================
# BUILD PRIMITIVE AXES
# ====================================================================
COMP_PAIRS = [
    ('fast','faster'),('slow','slower'),('tall','taller'),('small','smaller'),
    ('large','larger'),('hard','harder'),('soft','softer'),('warm','warmer'),
    ('cool','cooler'),('bright','brighter'),('dark','darker'),('clean','cleaner'),
    ('sharp','sharper'),('deep','deeper'),('wide','wider'),('strong','stronger'),
    ('long','longer'),('short','shorter'),('cheap','cheaper'),('fresh','fresher'),
    ('old','older'),('young','younger'),('cold','colder'),('thick','thicker'),
]
SUP_PAIRS = [
    ('fast','fastest'),('slow','slowest'),('tall','tallest'),('small','smallest'),
    ('large','largest'),('hard','hardest'),('soft','softest'),('warm','warmest'),
    ('cool','coolest'),('bright','brightest'),('dark','darkest'),('clean','cleanest'),
    ('sharp','sharpest'),('deep','deepest'),('wide','widest'),('strong','strongest'),
    ('long','longest'),('short','shortest'),('cheap','cheapest'),('fresh','freshest'),
    ('old','oldest'),('young','youngest'),('cold','coldest'),('thick','thickest'),
]
PLURAL_PAIRS = [
    ('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),('book','books'),
    ('car','cars'),('hand','hands'),('eye','eyes'),('word','words'),('day','days'),
    ('year','years'),('name','names'),('place','places'),('thing','things'),
    ('way','ways'),('part','parts'),('house','houses'),('arm','arms'),('leg','legs'),
    ('door','doors'),('line','lines'),('man','men'),('child','children'),
]
PAST_PAIRS = [
    ('walk','walked'),('talk','talked'),('work','worked'),('play','played'),
    ('call','called'),('turn','turned'),('start','started'),('move','moved'),
    ('live','lived'),('love','loved'),('use','used'),('ask','asked'),
    ('seem','seemed'),('help','helped'),('feel','felt'),('run','ran'),
    ('go','went'),('get','got'),('say','said'),('make','made'),
    ('take','took'),('see','saw'),('know','knew'),('come','came'),
    ('give','gave'),('think','thought'),('find','found'),('tell','told'),
]

ax_comp, coh_comp, valid_comp = compute_axis(COMP_PAIRS)
ax_sup,  coh_sup,  valid_sup  = compute_axis(SUP_PAIRS)
ax_pl,   coh_pl,   valid_pl   = compute_axis(PLURAL_PAIRS)
ax_past, coh_past, valid_past = compute_axis(PAST_PAIRS)

sf_comp, af_comp = best_scale(ax_comp, valid_comp)
sf_sup,  af_sup  = best_scale(ax_sup,  valid_sup)
sf_pl,   af_pl   = best_scale(ax_pl,   valid_pl)
sf_past, af_past = best_scale(ax_past, valid_past)

print("Primitive axes:")
for nm, ax, coh, sf, af, vl in [
    ("base->comp",  ax_comp, coh_comp, sf_comp, af_comp, valid_comp),
    ("base->sup",   ax_sup,  coh_sup,  sf_sup,  af_sup,  valid_sup),
    ("base->plural",ax_pl,   coh_pl,   sf_pl,   af_pl,   valid_pl),
    ("base->past",  ax_past, coh_past, sf_past, af_past, valid_past),
]:
    n = len(vl)
    print("  %-14s coh=%.4f scale=%.2f acc=%d/%d (%.0f%%)" % (
        nm, coh, sf, af, n, 100*af/max(1,n)))
print()

# ====================================================================
# PART A: DIRECT comparative->superlative AXIS
# ====================================================================
print("PART A: Direct comparative->superlative axis")
print("-"*65)

COMP_SUP_PAIRS = [(c,s) for (b,c),(b2,s) in zip(COMP_PAIRS,SUP_PAIRS) if b==b2]
ax_cs, coh_cs, valid_cs = compute_axis(COMP_SUP_PAIRS)
sf_cs, af_cs = best_scale(ax_cs, valid_cs) if ax_cs is not None else (1.0, 0)
sr_cs, ar_cs = best_scale(ax_cs, valid_cs, reverse=True) if ax_cs is not None else (1.0, 0)
n_cs = len(valid_cs)
print("  Direct comp->sup axis: coh=%.4f scale=%.2f acc=%d/%d (%.0f%%) rev_acc=%d/%d (%.0f%%)" % (
    coh_cs if ax_cs is not None else 0, sf_cs, af_cs, n_cs, 100*af_cs/max(1,n_cs),
    ar_cs, n_cs, 100*ar_cs/max(1,n_cs)))
print()
if ax_cs is not None:
    for s, t, sid, tid in valid_cs[:15]:
        r = nn_retrieve(W_E[sid] + sf_cs * ax_cs, [sid])
        got = r[0][0] if r else '?'
        print("  %-12s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
print()

# ====================================================================
# PART B: COMPOSED comparative->superlative = d_sup - d_comp
# ====================================================================
print("PART B: Composed comp->sup axis = d_sup - d_comp")
print("-"*65)

if ax_comp is not None and ax_sup is not None:
    ax_composed = normed(ax_sup - ax_comp)
    # Coherence of composed axis on comp->sup pairs
    coh_composed = float(np.mean([
        np.dot(normed(W_E[tid]-W_E[sid]).astype(np.float32),
               ax_composed.astype(np.float32))
        for s,t,sid,tid in valid_cs
    ])) if valid_cs else 0.0
    sf_comp_ax, af_comp_ax = best_scale(ax_composed, valid_cs)
    sr_comp_ax, ar_comp_ax = best_scale(ax_composed, valid_cs, reverse=True)
    print("  Composed axis coh=%.4f scale=%.2f acc=%d/%d (%.0f%%) rev=%d/%d (%.0f%%)" % (
        coh_composed, sf_comp_ax, af_comp_ax, n_cs, 100*af_comp_ax/max(1,n_cs),
        ar_comp_ax, n_cs, 100*ar_comp_ax/max(1,n_cs)))
    # Cosine similarity between direct and composed
    sim_dc = float(np.dot(ax_cs.astype(np.float32), ax_composed.astype(np.float32))) if ax_cs is not None else 0
    print("  cos(direct, composed) = %.4f" % sim_dc)
    print()
    for s, t, sid, tid in valid_cs[:15]:
        r = nn_retrieve(W_E[sid] + sf_comp_ax * ax_composed, [sid])
        got = r[0][0] if r else '?'
        print("  %-12s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()

# ====================================================================
# PART C: superlative->comparative = -d_comp + d_sup reversed
# ====================================================================
print("PART C: Superlative -> comparative (reverse composed axis)")
print("-"*65)

if ax_comp is not None and ax_sup is not None:
    print("  Using -composed axis (sup->comp at reverse scale %.2f):" % sr_comp_ax)
    for s, t, sid, tid in valid_cs[:15]:
        r = nn_retrieve(W_E[tid] - sr_comp_ax * ax_composed, [tid])
        got = r[0][0] if r else '?'
        print("  %-12s -> %-14s  got=%-14s [%s]" % (t, s, got, 'HIT' if got==s else '---'))
    print()

# ====================================================================
# PART D: PLURAL->PAST composition test
# Can we compute base->past from base->plural and plural->past?
# ====================================================================
print("PART D: Axis composition — base->past via plural")
print("-"*65)

# Build plural->past axis
PLURAL_PAST_PAIRS = []
past_map = {b:p for b,p in [(s,t) for s,t,_,_ in valid_past]}
plural_map = {b:pl for b,pl in [(s,t) for s,t,_,_ in valid_pl]}
for base in past_map:
    if base in plural_map:
        PLURAL_PAST_PAIRS.append((plural_map[base], past_map[base]))

ax_pp, coh_pp, valid_pp = compute_axis(PLURAL_PAST_PAIRS)
sf_pp, af_pp = best_scale(ax_pp, valid_pp) if ax_pp is not None else (1.0, 0)
n_pp = len(valid_pp)
if ax_pp is not None:
    print("  Direct plural->past axis: coh=%.4f scale=%.2f acc=%d/%d (%.0f%%)" % (
        coh_pp, sf_pp, af_pp, n_pp, 100*af_pp/max(1,n_pp)))
    print()

# Composed: base->past = (base->plural) + (plural->past)
# i.e., d_past_composed = d_plural + d_plural->past, using matching scales
if ax_pp is not None:
    # Reconstruct base->past via the two-hop chain
    chain_correct = 0; chain_total = 0
    for base, past in past_map.items():
        if base not in plural_map: continue
        eb, bid = get_emb(base)
        if eb is None: continue
        plural = plural_map[base]
        # Hop 1: base -> plural
        r1 = nn_retrieve(eb + sf_pl * ax_pl, [bid])
        plural_got = r1[0][0] if r1 else None
        e1, i1 = get_emb(plural_got) if plural_got else (None, None)
        if e1 is None: continue
        # Hop 2: plural -> past
        r2 = nn_retrieve(e1 + sf_pp * ax_pp, [i1])
        past_got = r2[0][0] if r2 else None
        chain_total += 1
        if past_got == past: chain_correct += 1
    if chain_total > 0:
        print("  2-hop chain (base->plural->past): %d/%d (%.0f%%)" % (
            chain_correct, chain_total, 100*chain_correct/max(1,chain_total)))
    print()

# ====================================================================
# PART E: AXIS ALGEBRA — test whether d_A + d_B = d_C for related axes
# ====================================================================
print("PART E: Axis algebra — d_plural + d_past = d_plural_past?")
print("-"*65)

# Test: can we predict plural past form (dogs walked) from base (dog)?
# Alg: base + sf_pl*d_pl + sf_past*d_past = plural_past?
# (e.g., dog -> dogs -> walked? No -- this should give walked only)
# Real test: base + d_pl -> plural; plural + d_past -> plural_past?
# Let's test additive composition of axes directly:
#   d_combined = normed(sf_pl*d_pl + sf_past*d_past)
if ax_pl is not None and ax_past is not None:
    ax_combined = normed(sf_pl * ax_pl + sf_past * ax_past)
    # What does base + combined give us?
    test_words = [('dog','dogs'),('walk','walks'),('cat','cats'),('run','runs'),('go','goes')]
    print("  Additive axis combination (d_pl + d_past):")
    print("  For base forms, what does d_pl+d_past retrieve?")
    for base, _ in test_words:
        eb, bid = get_emb(base)
        if eb is None: continue
        r = nn_retrieve(eb + ax_combined, [bid])
        print("  %-10s -> %s" % (base, [x[0] for x in r[:3]]))
    print()

# ====================================================================
# PART F: COMPOSITION SUMMARY
# ====================================================================
print("COMPOSITION SUMMARY:")
print("="*65)
print()
print("  comp->sup direct axis:    coh=%.4f  acc=%d/%d (%.0f%%)" % (
    coh_cs if ax_cs is not None else 0, af_cs, n_cs, 100*af_cs/max(1,n_cs)))
if ax_comp is not None and ax_sup is not None:
    print("  comp->sup composed:       coh=%.4f  acc=%d/%d (%.0f%%)" % (
        coh_composed, af_comp_ax, n_cs, 100*af_comp_ax/max(1,n_cs)))
    print("  cos(direct, composed):    %.4f" % sim_dc)
    print()
    print("  Primitive axis scales:")
    print("    base->comp:  %.2f    base->sup:  %.2f" % (sf_comp, sf_sup))
    print("    Scale delta: %.2f" % (sf_sup - sf_comp))
    print()
    print("  Direct comp->sup scale: %.2f" % sf_cs)
    print("  Composed comp->sup scale: %.2f" % sf_comp_ax)
    print()
    print("  The composed axis d_sup - d_comp represents the DIFFERENCE")
    print("  in morphological transformation between superlative and comparative.")
    print("  If this equals the direct comp->sup axis, axis subtraction is valid.")
