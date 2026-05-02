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
def best_scale_fwd(axis, valid_pairs, n_scales=40):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.02, 4.0, n_scales):
        correct = sum(1 for src, tgt, sid, tid in valid_pairs
                      if nn_retrieve(W_E[sid] + s * axis, [sid])[0][0] == tgt)
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc
def best_scale_rev(axis, valid_pairs, n_scales=40):
    """Best scale for REVERSE: tgt - s*axis -> src"""
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.02, 4.0, n_scales):
        correct = sum(1 for src, tgt, sid, tid in valid_pairs
                      if nn_retrieve(W_E[tid] - s * axis, [tid])[0][0] == src)
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

print("DAY 286: MORPHOLOGICAL AXIS REVERSIBILITY (ENCODE=DECODE TEST)")
print("="*65)
print("Prediction: morphological axes are bijective -> reversible")
print("Expected: reverse scale ~ forward scale (ratio ~ 1.0)")
print()

# ====================================================================
# DEFINE MORPHOLOGICAL AXES
# ====================================================================
AXES = {
    'singular->plural': [
        ('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),
        ('book','books'),('car','cars'),('house','houses'),('hand','hands'),
        ('eye','eyes'),('arm','arms'),('leg','legs'),('door','doors'),
        ('word','words'),('line','lines'),('day','days'),('year','years'),
        ('name','names'),('place','places'),('thing','things'),('time','times'),
        ('way','ways'),('part','parts'),('world','worlds'),('man','men'),
        ('city','cities'),('child','children'),('foot','feet'),('tooth','teeth'),
    ],
    'plural->singular': [  # reversed training pairs for inverse axis
        ('cats','cat'),('dogs','dog'),('birds','bird'),('trees','tree'),
        ('books','book'),('cars','car'),('hands','hand'),('eyes','eye'),
        ('arms','arm'),('legs','leg'),('doors','door'),('words','word'),
        ('lines','line'),('days','day'),('years','year'),('names','name'),
        ('places','place'),('things','thing'),('ways','way'),('parts','part'),
        ('worlds','world'),
    ],
    'base->comparative': [
        ('fast','faster'),('slow','slower'),('tall','taller'),('small','smaller'),
        ('large','larger'),('hard','harder'),('soft','softer'),('warm','warmer'),
        ('cool','cooler'),('bright','brighter'),('dark','darker'),('clean','cleaner'),
        ('sharp','sharper'),('deep','deeper'),('wide','wider'),('strong','stronger'),
        ('long','longer'),('short','shorter'),('cheap','cheaper'),('fresh','fresher'),
    ],
    'base->superlative': [
        ('fast','fastest'),('slow','slowest'),('tall','tallest'),('small','smallest'),
        ('large','largest'),('hard','hardest'),('soft','softest'),('warm','warmest'),
        ('cool','coolest'),('bright','brightest'),('dark','darkest'),('clean','cleanest'),
        ('sharp','sharpest'),('deep','deepest'),('wide','widest'),('strong','strongest'),
        ('long','longest'),('short','shortest'),('cheap','cheapest'),('fresh','freshest'),
    ],
    'masc->fem': [
        ('king','queen'),('man','woman'),('boy','girl'),('son','daughter'),
        ('brother','sister'),('father','mother'),('uncle','aunt'),('prince','princess'),
        ('hero','heroine'),('actor','actress'),('waiter','waitress'),('host','hostess'),
        ('lion','lioness'),('tiger','tigress'),('god','goddess'),('duke','duchess'),
    ],
    'base->past': [
        ('walk','walked'),('talk','talked'),('work','worked'),('play','played'),
        ('call','called'),('turn','turned'),('start','started'),('move','moved'),
        ('live','lived'),('love','loved'),('use','used'),('ask','asked'),
        ('seem','seemed'),('help','helped'),('want','wanted'),('need','needed'),
        ('feel','felt'),('run','ran'),('go','went'),('get','got'),
        ('say','said'),('make','made'),('take','took'),('see','saw'),
        ('know','knew'),('come','came'),('give','gave'),('think','thought'),
        ('find','found'),('tell','told'),
    ],
}

results = {}
for axis_name, pairs in AXES.items():
    ax, coh, valid = compute_axis(pairs)
    if ax is None or len(valid) < 3:
        print("  %-22s  SKIP (n=%d)" % (axis_name, len(valid))); continue
    scale_fwd, acc_fwd = best_scale_fwd(ax, valid)
    scale_rev, acc_rev = best_scale_rev(ax, valid)
    ratio = scale_rev / scale_fwd if scale_fwd > 0 else 0
    results[axis_name] = (ax, coh, scale_fwd, acc_fwd, scale_rev, acc_rev, ratio, valid)

print("%-22s  %5s  %6s %5s  %6s %5s  %5s" % (
    "Axis", "coh", "s_fwd", "fwd%", "s_rev", "rev%", "ratio"))
print("-"*65)
for name, (ax, coh, sf, af, sr, ar, ratio, valid) in results.items():
    n = len(valid)
    pf = 100*af/max(1,n); pr = 100*ar/max(1,n)
    print("  %-22s %.3f  %5.2f %4.0f%%  %5.2f %4.0f%%  %5.2f" % (
        name, coh, sf, pf, sr, pr, ratio))
print()

# ====================================================================
# DETAILED PAIR-BY-PAIR: singular<->plural (most interesting)
# ====================================================================
print("DETAILED: singular->plural forward and plural<-singular reverse")
print("-"*65)

if 'singular->plural' in results:
    ax_pl, coh_pl, sf_pl, af_pl, sr_pl, ar_pl, ratio_pl, valid_pl = results['singular->plural']
    print("  Forward (singular->plural)  scale=%.2f:" % sf_pl)
    fwd_correct = 0
    for s, t, sid, tid in valid_pl[:20]:
        r = nn_retrieve(W_E[sid] + sf_pl * ax_pl, [sid])
        got = r[0][0] if r else '?'
        hit = (got == t)
        if hit: fwd_correct += 1
        print("    %-12s -> %-12s  got=%-12s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    print()
    print("  Reverse (plural->singular)  scale=%.2f:" % sr_pl)
    rev_correct = 0
    for s, t, sid, tid in valid_pl[:20]:
        r = nn_retrieve(W_E[tid] - sr_pl * ax_pl, [tid])
        got = r[0][0] if r else '?'
        hit = (got == s)
        if rev_correct < 25: rev_correct += (1 if hit else 0)
        print("    %-12s -> %-12s  got=%-12s [%s]" % (t, s, got, 'HIT' if hit else '---'))
    print()

# ====================================================================
# DETAILED: masculine<->feminine (gender axis)
# ====================================================================
print("DETAILED: masculine->feminine forward and reverse")
print("-"*65)

if 'masc->fem' in results:
    ax_g, coh_g, sf_g, af_g, sr_g, ar_g, ratio_g, valid_g = results['masc->fem']
    print("  Forward (masc->fem)  scale=%.2f:" % sf_g)
    for s, t, sid, tid in valid_g:
        r = nn_retrieve(W_E[sid] + sf_g * ax_g, [sid])
        got = r[0][0] if r else '?'
        print("    %-12s -> %-12s  got=%-12s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()
    print("  Reverse (fem->masc)  scale=%.2f:" % sr_g)
    for s, t, sid, tid in valid_g:
        r = nn_retrieve(W_E[tid] - sr_g * ax_g, [tid])
        got = r[0][0] if r else '?'
        print("    %-12s -> %-12s  got=%-12s [%s]" % (t, s, got, 'HIT' if got==s else '---'))
    print()

# ====================================================================
# DETAILED: base<->comparative
# ====================================================================
print("DETAILED: base->comparative forward and reverse")
print("-"*65)
if 'base->comparative' in results:
    ax_c, coh_c, sf_c, af_c, sr_c, ar_c, ratio_c, valid_c = results['base->comparative']
    print("  Forward (base->comparative)  scale=%.2f:" % sf_c)
    for s, t, sid, tid in valid_c[:15]:
        r = nn_retrieve(W_E[sid] + sf_c * ax_c, [sid])
        got = r[0][0] if r else '?'
        print("    %-10s -> %-12s  got=%-12s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()
    print("  Reverse (comparative->base)  scale=%.2f:" % sr_c)
    for s, t, sid, tid in valid_c[:15]:
        r = nn_retrieve(W_E[tid] - sr_c * ax_c, [tid])
        got = r[0][0] if r else '?'
        print("    %-10s -> %-12s  got=%-12s [%s]" % (t, s, got, 'HIT' if got==s else '---'))
    print()

# ====================================================================
# SYMMETRY TABLE (test with inverted training pairs)
# ====================================================================
print("CROSS-CHECK: build inverse axis directly vs use negative forward axis")
print("-"*65)
print("(Does training plural->singular give same results as -1*(singular->plural) axis?)")
print()

if 'singular->plural' in results and 'plural->singular' in results:
    ax_fwd = results['singular->plural'][0]
    ax_inv_direct = results['plural->singular'][0]
    # Cosine similarity between -ax_fwd and ax_inv_direct
    sim = float(np.dot(normed(-ax_fwd).astype(np.float32), normed(ax_inv_direct).astype(np.float32)))
    print("  cos(-forward_axis, direct_inverse_axis) = %.4f" % sim)
    print("  (1.0 = perfectly equivalent; -1.0 = perfectly opposite)")
    print()
    _, _, sf_inv, af_inv, _, _, _, valid_inv = results['plural->singular']
    n_inv = len(valid_inv)
    print("  Direct inverse axis (plural->singular): acc=%d/%d (%.0f%%) scale=%.2f" % (
        af_inv, n_inv, 100*af_inv/max(1,n_inv), sf_inv))

# ====================================================================
# SYMMETRY SUMMARY
# ====================================================================
print()
print("ENCODE=DECODE SYMMETRY SUMMARY:")
print("="*65)
print("%-22s  %5s  %5s  %5s  SYMMETRIC?" % ("Axis", "fwd%", "rev%", "ratio"))
print("-"*65)
SYM_THRESHOLD = 0.85  # ratio 0.85-1.15 = symmetric
for name, (ax, coh, sf, af, sr, ar, ratio, valid) in results.items():
    n = len(valid)
    pf = 100*af/max(1,n); pr = 100*ar/max(1,n)
    sym = "YES" if abs(ratio - 1.0) < 0.20 and pr >= pf * 0.70 else "NO"
    print("  %-22s %4.0f%%  %4.0f%%  %5.2f  %s" % (name, pf, pr, ratio, sym))

print()
print("Prediction from DC 420: all bijective morphological axes -> YES")
print("If nat->lang is symmetric (ratio=1.00), morphology should be too.")
