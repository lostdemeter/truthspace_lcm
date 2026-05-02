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
    chords = [get_emb(t)[0]-get_emb(s)[0] for s,t in pairs
              if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if not chords: return None, 0.0
    md = normed(np.mean(chords, axis=0))
    return md, float(np.mean([np.dot(normed(c), md) for c in chords]))

# Scale-based retrieval (Day 277 approach)
def nn_scaled(source_emb, axis, scale, exclude_ids, top_n=1):
    pred = source_emb + scale * axis
    pred_n = normed(pred).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), i) for i in top]

# Scale-free: retrieve by cosine of normed(source + alpha*axis) for several alpha
# The key insight: normed(source + alpha*axis) sweeps a great-circle arc as alpha increases
# For large alpha it converges to the axis direction; for small alpha it stays near source
# We test a range and pick the best alpha per-pair at train time, or scan at test time
def nn_scalefree_scan(source_emb, axis, exclude_ids, alphas, top_n=1):
    """Scan over multiple alpha values, return all top-1 results."""
    results = []
    for a in alphas:
        pred_n = normed(source_emb + a * axis).astype(np.float32)
        sims = W_n @ pred_n
        for eid in exclude_ids: sims[eid] = -1.0
        idx = np.argmax(sims)
        results.append((a, tok.decode([idx]).strip(), float(sims[idx]), int(idx)))
    return results

def nn_scalefree_best(source_emb, axis, exclude_ids, alphas=None):
    """Find the alpha that gives the most common top-1 result (consensus vote)."""
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0]
    scan = nn_scalefree_scan(source_emb, axis, exclude_ids, alphas)
    from collections import Counter
    votes = Counter(word for _, word, _, _ in scan)
    best_word = votes.most_common(1)[0][0]
    best_alpha = next(a for a, w, _, _ in scan if w == best_word)
    best_sim   = next(s for a, w, s, _ in scan if w == best_word)
    best_id    = next(i for a, w, _, i in scan if w == best_word)
    return best_word, best_alpha, best_sim, best_id

def scale_for_axis(axis, pairs, n_scales=25):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = 0
        for src, tgt in pairs:
            es, sid = get_emb(src); et, tid = get_emb(tgt)
            if es is None or et is None: continue
            res = nn_scaled(es, axis, s, [sid])
            if res and res[0][0] == tgt: correct += 1
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

def normalise_token(word, targets_lower_set):
    if word.lower() in targets_lower_set: return word.lower()
    return word

# --- Build axes (Day 277 versions: clean, unextended) ---
SCIENTIST_NAT = [('Einstein','German'),('Newton','British'),('Darwin','British'),('Kepler','German'),('Euler','Swiss'),('Gauss','German'),('Turing','British'),('Tesla','Serbian'),('Napoleon','French'),('Churchill','British'),('Lincoln','American'),('Gandhi','Indian'),('Caesar','Roman'),('Aristotle','Greek'),('Plato','Greek'),('Shakespeare','British'),('Mozart','Austrian'),('Marx','German'),('Freud','Austrian'),('Kant','German')]
DEMONYM_COUNTRY=[('German','germany'),('British','britain'),('French','france'),('Japanese','japan'),('Chinese','china'),('Italian','italy'),('Spanish','spain'),('Russian','russia'),('Greek','greece'),('Polish','poland'),('Swedish','sweden'),('Norwegian','norway'),('Austrian','austria'),('American','america'),('Swiss','switzerland')]
LANGUAGE_PAIRS =[('france','french'),('germany','german'),('spain','spanish'),('russia','russian'),('japan','japanese'),('china','chinese'),('italy','italian'),('portugal','portuguese'),('poland','polish'),('sweden','swedish'),('norway','norwegian'),('denmark','danish'),('greece','greek'),('turkey','turkish'),('finland','finnish'),('hungary','hungarian')]

ax_nat, coh_nat = compute_axis(SCIENTIST_NAT)
ax_dem, coh_dem = compute_axis(DEMONYM_COUNTRY)
ax_lan, coh_lan = compute_axis(LANGUAGE_PAIRS)

scale_nat, acc_nat = scale_for_axis(ax_nat, SCIENTIST_NAT)
scale_dem, acc_dem = scale_for_axis(ax_dem, DEMONYM_COUNTRY)
scale_lan, acc_lan = scale_for_axis(ax_lan, LANGUAGE_PAIRS)

n_nat = sum(1 for s,t in SCIENTIST_NAT if get_emb(s)[0] is not None)
n_dem = sum(1 for s,t in DEMONYM_COUNTRY if get_emb(s)[0] is not None)
n_lan = sum(1 for s,t in LANGUAGE_PAIRS if get_emb(s)[0] is not None)

DEM_TARGETS_LOWER = set(t.lower() for _, t in DEMONYM_COUNTRY)

print("DAY 279: SCALE-FREE COSINE-DIRECTION NN RETRIEVAL")
print("="*60)
print()
print("Axes (Day 277 originals, unextended):")
print("  nat:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_nat, scale_nat, acc_nat, n_nat, 100*acc_nat/n_nat))
print("  dem:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_dem, scale_dem, acc_dem, n_dem, 100*acc_dem/n_dem))
print("  lan:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_lan, scale_lan, acc_lan, n_lan, 100*acc_lan/n_lan))
print()

# --- SINGLE-HOP: scale-free vs scaled ---
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0, 3.0]

print("SINGLE-HOP COMPARISON: scaled vs scale-free (voting)")
print("-"*60)
for axis_name, axis, scale, pairs in [
    ('nat',  ax_nat, scale_nat, SCIENTIST_NAT),
    ('dem',  ax_dem, scale_dem, DEMONYM_COUNTRY),
    ('lan',  ax_lan, scale_lan, LANGUAGE_PAIRS),
]:
    sc_correct = 0; sf_correct = 0; n = 0
    for src, tgt in pairs:
        es, sid = get_emb(src)
        if es is None: continue
        n += 1
        # scaled
        r_sc = nn_scaled(es, axis, scale, [sid])
        if r_sc and r_sc[0][0] == tgt: sc_correct += 1
        # scale-free voting
        word_sf, alpha_sf, _, _ = nn_scalefree_best(es, axis, [sid], ALPHAS)
        if word_sf == tgt: sf_correct += 1
    print("  %-5s  scaled: %d/%d (%.0f%%)  scale-free: %d/%d (%.0f%%)" % (
        axis_name, sc_correct, n, 100*sc_correct/n, sf_correct, n, 100*sf_correct/n))
print()

# --- 3-HOP SEQUENTIAL: scale-free ---
PERSON_LANGUAGE = [
    ('Einstein',  'german',  'German',  'germany'),
    ('Newton',    'english', 'British', 'britain'),
    ('Darwin',    'english', 'British', 'britain'),
    ('Kepler',    'german',  'German',  'germany'),
    ('Gauss',     'german',  'German',  'germany'),
    ('Napoleon',  'french',  'French',  'france'),
    ('Aristotle', 'greek',   'Greek',   'greece'),
    ('Plato',     'greek',   'Greek',   'greece'),
    ('Mozart',    'german',  'Austrian','austria'),
    ('Marx',      'german',  'German',  'germany'),
]

print("SEQUENTIAL 3-HOP: scale-free vs scaled (Day 277 baseline)")
print("-"*60)

correct_sc = 0   # Day 277 scaled (recomputed for same test set)
correct_sf = 0   # scale-free

for person, lang_exp, nat_exp, cty_exp in PERSON_LANGUAGE:
    ep, pid = get_emb(person)
    if ep is None: continue

    # SCALED (Day 277 method, + capitalisation normalisation)
    r1 = nn_scaled(ep, ax_nat, scale_nat, [pid])
    nat_sc = r1[0][0] if r1 else None
    e1, i1 = get_emb(nat_sc) if nat_sc else (None, None)
    if e1 is not None:
        r2 = nn_scaled(e1, ax_dem, scale_dem, [i1])
        cty_raw = r2[0][0] if r2 else None
        cty_sc = normalise_token(cty_raw, DEM_TARGETS_LOWER) if cty_raw else None
        e2, i2 = get_emb(cty_sc) if cty_sc else (None, None)
        if e2 is not None:
            r3 = nn_scaled(e2, ax_lan, scale_lan, [i2])
            lan_sc = r3[0][0] if r3 else None
        else: lan_sc = None
    else: cty_sc = None; lan_sc = None
    hit_sc = (lan_sc == lang_exp)
    if hit_sc: correct_sc += 1

    # SCALE-FREE (voting)
    nat_sf, _, _, nid_sf = nn_scalefree_best(ep, ax_nat, [pid], ALPHAS)
    e1sf, i1sf = get_emb(nat_sf) if nat_sf else (None, None)
    if e1sf is not None:
        cty_raw_sf, _, _, ctid_raw_sf = nn_scalefree_best(e1sf, ax_dem, [i1sf], ALPHAS)
        cty_sf = normalise_token(cty_raw_sf, DEM_TARGETS_LOWER) if cty_raw_sf else None
        e2sf, i2sf = get_emb(cty_sf) if cty_sf else (None, None)
        if e2sf is not None:
            lan_sf, _, _, _ = nn_scalefree_best(e2sf, ax_lan, [i2sf], ALPHAS)
        else: lan_sf = None
    else: cty_sf = None; lan_sf = None
    hit_sf = (lan_sf == lang_exp)
    if hit_sf: correct_sf += 1

    print("  %-12s nat: %-10s/%-10s  cty: %-10s/%-10s  lang: %-8s [sc:%s sf:%s]" % (
        person,
        nat_sc if nat_sc else '?', nat_sf if nat_sf else '?',
        cty_sc if cty_sc else '?', cty_sf if cty_sf else '?',
        lang_exp,
        'HIT' if hit_sc else '---', 'HIT' if hit_sf else '---'))

n_valid = len(PERSON_LANGUAGE)
print()
print("  Scaled 3-hop (+ normalisation):  %d/%d (%.0f%%)" % (correct_sc, n_valid, 100*correct_sc/n_valid))
print("  Scale-free 3-hop (+ normalisation): %d/%d (%.0f%%)" % (correct_sf, n_valid, 100*correct_sf/n_valid))
print()

# --- ALPHA SWEEP ANALYSIS: what alpha wins for each axis ---
print("ALPHA SWEEP: most common winning alpha per axis (single-hop)")
print("-"*60)
for axis_name, axis, pairs in [('nat', ax_nat, SCIENTIST_NAT), ('dem', ax_dem, DEMONYM_COUNTRY), ('lan', ax_lan, LANGUAGE_PAIRS)]:
    alpha_counts = {a: 0 for a in ALPHAS}
    n = 0
    for src, tgt in pairs:
        es, sid = get_emb(src)
        if es is None: continue
        n += 1
        scan = nn_scalefree_scan(es, axis, [sid], ALPHAS)
        for a, word, sim, idx in scan:
            if word == tgt:
                alpha_counts[a] += 1
                break  # count first alpha that hits
    print("  %s  (n=%d): alpha_hits=%s" % (axis_name, n, [(a, alpha_counts[a]) for a in ALPHAS if alpha_counts[a] > 0]))
