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
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def scale_and_acc(axis, valid_pairs, n_scales=30):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = sum(1 for src, tgt, sid, tid in valid_pairs
                      if nn_retrieve(W_E[sid] + s * axis, [sid])[0][0] == tgt)
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

print("DAY 285: ENCODE=DECODE SYMMETRY TEST")
print("="*65)
print("The TruthSpace hypothesis: encoding and decoding are the SAME")
print("operation in opposite directions (phi and 1/phi).")
print("Test: can we reverse person->nat->lang to lang->nat->person")
print("using negative axis directions at appropriate scales?")
print()

# ====================================================================
# Build all axes (replicating Day 282 architecture)
# ====================================================================
CLUSTERS = {
    'German':   [('Einstein','German'),('Marx','German'),('Kepler','German'),('Gauss','German'),('Kant','German'),('Planck','German'),('Schiller','German')],
    'Austrian': [('Freud','Austrian'),('Mozart','Austrian'),('Schubert','Austrian'),('Haydn','Austrian')],
    'British':  [('Newton','British'),('Darwin','British'),('Turing','British'),('Churchill','British'),('Shakespeare','British'),('Faraday','British'),('Dickens','British'),('Austen','British'),('Locke','British'),('Hobbes','British')],
    'French':   [('Napoleon','French'),('Curie','French'),('Descartes','French'),('Voltaire','French'),('Rousseau','French'),('Hugo','French'),('Pasteur','French')],
    'Greek':    [('Aristotle','Greek'),('Plato','Greek'),('Pythagoras','Greek'),('Socrates','Greek'),('Homer','Greek'),('Euclid','Greek'),('Archimedes','Greek')],
    'American': [('Lincoln','American'),('Franklin','American'),('Edison','American'),('Washington','American'),('Jefferson','American'),('Thoreau','American'),('Whitman','American')],
    'Italian':  [('Galileo','Italian'),('Leonardo','Italian'),('Dante','Italian'),('Michelangelo','Italian'),('Machiavelli','Italian'),('Vivaldi','Italian')],
    'Russian':  [('Lenin','Russian'),('Tolstoy','Russian'),('Dostoevsky','Russian'),('Pushkin','Russian'),('Tchaikovsky','Russian'),('Stalin','Russian')],
}
NAT_LANG = [
    ('German','german'),('Austrian','german'),('British','english'),('American','english'),
    ('French','french'),('Spanish','spanish'),('Russian','russian'),('Italian','italian'),
    ('Japanese','japanese'),('Chinese','chinese'),('Polish','polish'),('Swiss','german'),
    ('Roman','latin'),('Indian','hindi'),('Brazilian','portuguese'),
]

cluster_axes, cluster_centroids = {}, {}
for cname, pairs in CLUSTERS.items():
    ax, coh, valid = compute_axis(pairs)
    if ax is None or len(valid) < 2: continue
    scale, acc = scale_and_acc(ax, valid)
    cluster_axes[cname] = (ax, coh, scale, valid)
    embs = [W_E[sid] for _,_,sid,_ in valid]
    cluster_centroids[cname] = normed(np.mean(embs, axis=0))

ax_nl, coh_nl, valid_nl = compute_axis(NAT_LANG)
scale_nl, acc_nl = scale_and_acc(ax_nl, valid_nl)

print("Axes built:")
for cname, (ax, coh, scale, valid) in cluster_axes.items():
    print("  %-12s coh=%.3f scale=%.2f n=%d" % (cname, coh, scale, len(valid)))
print("  %-12s coh=%.3f scale=%.2f n=%d" % ("nat->lang", coh_nl, scale_nl, len(valid_nl)))
print()

# ====================================================================
# PART A: FORWARD CHAIN (baseline) person -> nat -> lang
# ====================================================================
print("PART A: FORWARD CHAIN  person → nat → lang")
print("-"*65)

FORWARD_TESTS = [
    ('Einstein', 'German',   'german'),
    ('Newton',   'British',  'english'),
    ('Napoleon', 'French',   'french'),
    ('Aristotle','Greek',    'greek'),
    ('Lenin',    'Russian',  'russian'),
    ('Marx',     'German',   'german'),
    ('Freud',    'Austrian', 'german'),
    ('Galileo',  'Italian',  'italian'),
    ('Lincoln',  'American', 'english'),
    ('Kant',     'German',   'german'),
]

def cluster_hop1(person_emb, person_id):
    en = normed(person_emb).astype(np.float32)
    best_c, best_sim = None, -1.0
    for cname, cen in cluster_centroids.items():
        sim = float(np.dot(en, cen.astype(np.float32)))
        if sim > best_sim: best_sim = sim; best_c = cname
    if best_c and best_c in cluster_axes:
        ax_c, _, scale_c, _ = cluster_axes[best_c]
        r = nn_retrieve(person_emb + scale_c * ax_c, [person_id])
        return r[0][0] if r else None, best_c
    return None, None

fwd_results = []
correct_fwd = 0
for person, nat_exp, lang_exp in FORWARD_TESTS:
    ep, pid = get_emb(person)
    if ep is None: print("  %-12s SKIP" % person); continue
    nat_got, cname = cluster_hop1(ep, pid)
    e1, i1 = get_emb(nat_got) if nat_got else (None, None)
    lang_got = None
    if e1 is not None:
        r2 = nn_retrieve(e1 + scale_nl * ax_nl, [i1])
        lang_got = r2[0][0] if r2 else None
    hit = (lang_got == lang_exp)
    if hit: correct_fwd += 1
    fwd_results.append((person, nat_got, lang_got, nat_exp, lang_exp, hit))
    print("  %-12s -> %-10s -> %-10s  [%s]  exp=%s" % (
        person, nat_got if nat_got else '?', lang_got if lang_got else '?',
        'HIT' if hit else '---', lang_exp))
print()
print("  Forward chain:  %d/%d (%.0f%%)" % (correct_fwd, len(fwd_results), 100*correct_fwd/max(1,len(fwd_results))))
print()

# ====================================================================
# PART B: REVERSE CHAIN (ENCODE=DECODE test) lang -> nat -> person
# Using NEGATIVE axis directions
# ====================================================================
print("PART B: REVERSE CHAIN  lang → nat → person  (negative axes)")
print("-"*65)
print("Using -scale_nl * ax_nl  and  -scale_c * ax_c")
print()

# Build nat->lang INVERSE axis scale (scan negative direction)
def scale_and_acc_reverse(axis, valid_pairs, n_scales=30):
    """Find best scale for NEGATIVE axis direction."""
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = sum(1 for src, tgt, sid, tid in valid_pairs
                      if nn_retrieve(W_E[tid] - s * axis, [tid])[0][0] == src)
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

# Find reverse scale for nat->lang (inverse: lang -> nat)
scale_nl_rev, acc_nl_rev = scale_and_acc_reverse(ax_nl, valid_nl)
print("  nat->lang axis reverse scale:  %.2f  (acc=%d/%d on training pairs)" % (
    scale_nl_rev, acc_nl_rev, len(valid_nl)))
print("  (positive scale: %.2f)" % scale_nl)
print()

# Test lang -> nat using negative axis
print("  Single-hop lang->nat (negative nat->lang axis):")
LANG_NAT_TEST = [
    ('german',   'German'),  ('english', 'British'),  ('french',  'French'),
    ('russian',  'Russian'), ('italian', 'Italian'),  ('spanish', 'Spanish'),
    ('japanese', 'Japanese'),('chinese', 'Chinese'),
]
lang_nat_correct = 0
for lang, nat_exp in LANG_NAT_TEST:
    el, lid = get_emb(lang)
    if el is None: print("  %-12s SKIP" % lang); continue
    r = nn_retrieve(el - scale_nl_rev * ax_nl, [lid])
    got = r[0][0] if r else '?'
    hit = (got == nat_exp)
    if hit: lang_nat_correct += 1
    top3 = [x[0] for x in r[:3]]
    print("    %-10s -> %-14s  got=%-14s [%s]  top3=%s" % (
        lang, nat_exp, got, 'HIT' if hit else '---', top3))
print()
print("  lang->nat (reverse): %d/%d (%.0f%%)" % (lang_nat_correct, len(LANG_NAT_TEST), 100*lang_nat_correct/max(1,len(LANG_NAT_TEST))))
print()

# Now nat -> person using negative person->nat axis
# Use the cluster centroid as the 'landing zone' for person retrieval
print("  Single-hop nat->person (negative cluster nat axes):")
NAT_PERSON_TEST = [
    ('German',   ['Einstein','Marx','Kepler','Gauss','Kant']),
    ('British',  ['Newton','Darwin','Turing','Churchill','Shakespeare']),
    ('French',   ['Napoleon','Voltaire','Rousseau','Hugo','Curie']),
    ('Russian',  ['Lenin','Tolstoy','Dostoevsky','Pushkin']),
    ('American', ['Lincoln','Franklin','Edison','Washington']),
]

for nat, expected_persons in NAT_PERSON_TEST:
    en_nat, nid = get_emb(nat)
    if en_nat is None: continue
    # Find the cluster for this nationality
    matching_cluster = None
    for cname, (ax_c, coh_c, scale_c, valid_c) in cluster_axes.items():
        # Check if this cluster targets this nat
        targets = [t for _, t, _, _ in valid_c]
        if nat in targets:
            matching_cluster = cname
            break
    if matching_cluster is None: print("  No cluster for %s" % nat); continue
    ax_c, coh_c, scale_c, valid_c = cluster_axes[matching_cluster]
    # Reverse: nat - scale * axis should point toward person cluster
    r = nn_retrieve(en_nat - scale_c * ax_c, [nid])
    top5 = [x[0] for x in r[:5]]
    hit = any(p in top5 for p in expected_persons)
    print("  %-12s -> top5=%s  [%s]" % (nat, top5, 'HIT' if hit else '---'))
print()

# ====================================================================
# PART C: FULL REVERSE 2-hop: lang -> nat -> person
# ====================================================================
print("PART C: FULL REVERSE 2-hop  lang -> nat -> person")
print("-"*65)

REVERSE_TESTS = [
    ('german',  'German',   'Einstein'),
    ('german',  'German',   'Marx'),
    ('english', 'British',  'Newton'),
    ('english', 'British',  'Darwin'),
    ('french',  'French',   'Napoleon'),
    ('russian', 'Russian',  'Lenin'),
    ('italian', 'Italian',  'Galileo'),
    ('english', 'American', 'Lincoln'),
]

correct_rev = 0
print("  %-10s  nat_got      person_top5                    expected   ok?" % "lang")
print("  " + "-"*68)
for lang, nat_exp, person_exp in REVERSE_TESTS:
    el, lid = get_emb(lang)
    if el is None: continue

    # Hop R1: lang -> nat (negative nat->lang axis)
    r1 = nn_retrieve(el - scale_nl_rev * ax_nl, [lid])
    nat_got = r1[0][0] if r1 else None

    # Hop R2: nat -> person (negative cluster axis)
    matching_cluster = None
    for cname, (ax_c, coh_c, scale_c, valid_c) in cluster_axes.items():
        targets = [t for _, t, _, _ in valid_c]
        if nat_got and nat_got in targets:
            matching_cluster = cname
            break

    if matching_cluster:
        ax_c, coh_c, scale_c, valid_c = cluster_axes[matching_cluster]
        en_nat, nid = get_emb(nat_got)
        if en_nat is not None:
            r2 = nn_retrieve(en_nat - scale_c * ax_c, [nid])
            top5 = [x[0] for x in r2[:5]]
        else:
            top5 = []
    else:
        top5 = []

    hit = (person_exp in top5)
    if hit: correct_rev += 1
    print("  %-10s  %-12s %-34s %-10s [%s]" % (
        lang,
        nat_got if nat_got else '?',
        str(top5[:5]),
        person_exp,
        'HIT' if hit else '---'))

n_rev = len(REVERSE_TESTS)
print()
print("  Reverse 2-hop (lang->nat->person): %d/%d (%.0f%%)" % (correct_rev, n_rev, 100*correct_rev/max(1,n_rev)))
print()

# ====================================================================
# PART D: ENCODE=DECODE SYMMETRY ANALYSIS
# ====================================================================
print("PART D: SYMMETRY ANALYSIS")
print("-"*65)
print()
print("  Forward chain  (person->nat->lang):  %d/%d (%.0f%%)" % (
    correct_fwd, len(fwd_results), 100*correct_fwd/max(1,len(fwd_results))))
print("  Reverse chain  (lang->nat->person):  %d/%d (%.0f%%)" % (
    correct_rev, n_rev, 100*correct_rev/max(1,n_rev)))
print()
print("  Single-hop forward  nat->lang:       %d/%d (%.0f%%)" % (acc_nl, len(valid_nl), 100*acc_nl/max(1,len(valid_nl))))
print("  Single-hop reverse  lang->nat:       %d/%d (%.0f%%)" % (lang_nat_correct, len(LANG_NAT_TEST), 100*lang_nat_correct/max(1,len(LANG_NAT_TEST))))
print()
print("  If ENCODE=DECODE: forward acc ≈ reverse acc")
print("  If asymmetric:    forward >> reverse  OR  reverse >> forward")
print()
print("  nat->lang scale (forward):  %.2f" % scale_nl)
print("  lang->nat scale (reverse):  %.2f  (ratio %.2f)" % (scale_nl_rev, scale_nl_rev/scale_nl))
print()

# ====================================================================
# PART E: AXIS SELF-INVERSE TEST
# Check if applying axis twice returns to origin
# phi * phi = phi^2 (not phi) -- the golden ratio property
# ====================================================================
print("PART E: AXIS DOUBLE-APPLICATION (phi^2 test)")
print("-"*65)
print("Does source + 2*scale*axis return to a meaningful location?")
print()

for word, nat_exp, lang_exp in [('Einstein','German','german'),('Newton','British','english'),('Napoleon','French','french')]:
    ep, pid = get_emb(word)
    if ep is None: continue
    cname, _ = max(cluster_centroids.items(), key=lambda x: float(np.dot(normed(ep).astype(np.float32), x[1].astype(np.float32))))
    ax_c, _, scale_c, _ = cluster_axes[cname]

    # Single hop: person -> nat
    r1 = nn_retrieve(ep + scale_c * ax_c, [pid])
    nat_got = r1[0][0] if r1 else '?'

    # Double hop: person + 2*scale = ?
    r2 = nn_retrieve(ep + 2*scale_c * ax_c, [pid])
    double_got = r2[0][0] if r2 else '?'

    # Triple hop: person + 3*scale = ?
    r3 = nn_retrieve(ep + 3*scale_c * ax_c, [pid])
    triple_got = r3[0][0] if r3 else '?'

    print("  %-12s  x1->%-12s  x2->%-12s  x3->%-12s  (exp: %s)" % (
        word, nat_got, double_got, triple_got, nat_exp))
print()

# nat->lang double application
for nat, lang_exp in [('German','german'),('British','english'),('French','french'),('Russian','russian')]:
    en, nid = get_emb(nat)
    if en is None: continue
    r1 = nn_retrieve(en + scale_nl * ax_nl, [nid])
    lang_got = r1[0][0] if r1 else '?'
    r2 = nn_retrieve(en + 2*scale_nl * ax_nl, [nid])
    double_got = r2[0][0] if r2 else '?'
    r3 = nn_retrieve(en + 3*scale_nl * ax_nl, [nid])
    triple_got = r3[0][0] if r3 else '?'
    print("  %-12s  x1->%-12s  x2->%-12s  x3->%-12s  (exp: %s)" % (
        nat, lang_got, double_got, triple_got, lang_exp))
