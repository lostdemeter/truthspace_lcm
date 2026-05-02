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
def scale_and_acc(axis, valid_pairs, n_scales=25):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = sum(1 for src, tgt, sid, tid in valid_pairs
                      if nn_retrieve(W_E[sid] + s * axis, [sid])[0][0] == tgt)
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

print("DAY 281: SOURCE-TYPE CLUSTERING FOR NATIONALITY AXIS")
print("="*65)
print()

# Full nat axis (Day 277 baseline)
ALL_NAT = [
    ('Einstein','German'), ('Marx','German'), ('Kepler','German'),
    ('Gauss','German'),    ('Kant','German'),  ('Freud','Austrian'),
    ('Mozart','Austrian'), ('Euler','Swiss'),
    ('Newton','British'),  ('Darwin','British'), ('Turing','British'),
    ('Churchill','British'),('Shakespeare','British'),
    ('Napoleon','French'), ('Curie','French'),
    ('Aristotle','Greek'), ('Plato','Greek'), ('Pythagoras','Greek'),
    ('Lincoln','American'),('Franklin','American'), ('Edison','American'),
    ('Gandhi','Indian'),   ('Tesla','Serbian'),
    ('Caesar','Roman'),    ('Cicero','Roman'),
]

ax_all, coh_all, valid_all = compute_axis(ALL_NAT)
scale_all, acc_all = scale_and_acc(ax_all, valid_all)
print("GLOBAL axis: coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
    coh_all, scale_all, acc_all, len(valid_all), 100*acc_all/len(valid_all)))
print()

# PER-CLUSTER AXES
CLUSTERS = {
    'German/Austrian': [('Einstein','German'),('Marx','German'),('Kepler','German'),('Gauss','German'),('Kant','German'),('Planck','German'),('Schiller','German'),('Freud','Austrian'),('Mozart','Austrian'),('Euler','Swiss')],
    'British':        [('Newton','British'),('Darwin','British'),('Turing','British'),('Churchill','British'),('Shakespeare','British'),('Faraday','British'),('Dickens','British'),('Austen','British'),('Locke','British'),('Hobbes','British')],
    'French':         [('Napoleon','French'),('Curie','French'),('Descartes','French'),('Voltaire','French'),('Rousseau','French'),('Hugo','French'),('Pasteur','French'),('Moliere','French')],
    'Greek':          [('Aristotle','Greek'),('Plato','Greek'),('Pythagoras','Greek'),('Socrates','Greek'),('Homer','Greek'),('Euclid','Greek'),('Archimedes','Greek'),('Herodotus','Greek')],
    'American':       [('Lincoln','American'),('Franklin','American'),('Edison','American'),('Washington','American'),('Jefferson','American'),('Thoreau','American'),('Whitman','American')],
    'Italian':        [('Galileo','Italian'),('Leonardo','Italian'),('Dante','Italian'),('Michelangelo','Italian'),('Machiavelli','Italian'),('Columbus','Italian'),('Vivaldi','Italian')],
    'Russian':        [('Lenin','Russian'),('Tolstoy','Russian'),('Dostoevsky','Russian'),('Pushkin','Russian'),('Tchaikovsky','Russian'),('Stalin','Russian')],
}

cluster_axes = {}
print("PER-CLUSTER AXES:")
print("-"*65)
for cluster_name, pairs in CLUSTERS.items():
    axis, coh, valid = compute_axis(pairs)
    if axis is None or len(valid) < 2:
        print("  %-22s SKIP (n=%d valid)" % (cluster_name, len(valid)))
        continue
    scale, acc = scale_and_acc(axis, valid)
    n = len(valid)
    print("  %-22s coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)  [%d valid]" % (
        cluster_name, coh, scale, acc, n, 100*acc/n, n))
    cluster_axes[cluster_name] = (axis, coh, scale, valid, pairs)
print()

# CLUSTER CENTROID CLASSIFIER
# For each query person, find which cluster centroid they're closest to
cluster_centroids = {}
for cname, (axis, coh, scale, valid, pairs) in cluster_axes.items():
    embs = [W_E[sid] for _, _, sid, _ in valid]
    centroid = normed(np.mean(embs, axis=0))
    cluster_centroids[cname] = centroid

def assign_cluster(word):
    e, wid = get_emb(word)
    if e is None: return None
    en = normed(e).astype(np.float32)
    best_c, best_sim = None, -1.0
    for cname, cen in cluster_centroids.items():
        sim = float(np.dot(en, cen.astype(np.float32)))
        if sim > best_sim: best_sim = sim; best_c = cname
    return best_c, best_sim

print("CLUSTER ASSIGNMENT TEST:")
print("-"*65)
TEST_PERSONS = [
    ('Einstein',  'German',   'German/Austrian'),
    ('Kepler',    'German',   'German/Austrian'),
    ('Newton',    'British',  'British'),
    ('Darwin',    'British',  'British'),
    ('Turing',    'British',  'British'),
    ('Napoleon',  'French',   'French'),
    ('Aristotle', 'Greek',    'Greek'),
    ('Plato',     'Greek',    'Greek'),
    ('Mozart',    'Austrian', 'German/Austrian'),
    ('Marx',      'German',   'German/Austrian'),
    ('Gandhi',    'Indian',   None),  # no Indian cluster
    ('Caesar',    'Roman',    None),  # no Roman cluster
    ('Galileo',   'Italian',  'Italian'),
    ('Lenin',     'Russian',  'Russian'),
    ('Lincoln',   'American', 'American'),
    ('Franklin',  'American', 'American'),
    ('Edison',    'American', 'American'),
]

assign_correct = 0; n_assign = 0
for person, nat_exp, cluster_exp in TEST_PERSONS:
    e, eid = get_emb(person)
    if e is None: print("  %-14s SKIP" % person); continue
    result = assign_cluster(person)
    if result is None: continue
    assigned, sim = result
    correct_assign = (assigned == cluster_exp) if cluster_exp else None
    if cluster_exp: 
        n_assign += 1
        if correct_assign: assign_correct += 1
    print("  %-14s  assigned=%-22s [%s]  sim=%.4f  expected=%s" % (
        person, assigned, 
        'OK ' if correct_assign else ('---' if cluster_exp else 'N/A'),
        sim, cluster_exp if cluster_exp else 'none'))

print()
print("  Cluster assignment accuracy: %d/%d (%.0f%%)" % (
    assign_correct, n_assign, 100*assign_correct/n_assign))
print()

# CLUSTER-AWARE RETRIEVAL: assign cluster, then apply that cluster's axis
print("CLUSTER-AWARE HOP 1: person -> nationality")
print("-"*65)

DEMONYM_COUNTRY=[('German','germany'),('British','britain'),('French','france'),('Japanese','japan'),('Chinese','china'),('Italian','italy'),('Spanish','spain'),('Russian','russia'),('Greek','greece'),('Polish','poland'),('Swedish','sweden'),('Norwegian','norway'),('Austrian','austria'),('American','america'),('Swiss','switzerland')]
LANGUAGE_PAIRS =[('france','french'),('germany','german'),('spain','spanish'),('russia','russian'),('japan','japanese'),('china','chinese'),('italy','italian'),('portugal','portuguese'),('poland','polish'),('sweden','swedish'),('norway','norwegian'),('denmark','danish'),('greece','greek'),('turkey','turkish'),('finland','finnish'),('hungary','hungarian')]
ax_dem, coh_dem, valid_dem = compute_axis(DEMONYM_COUNTRY)
ax_lan, coh_lan, valid_lan = compute_axis(LANGUAGE_PAIRS)
scale_dem, _ = scale_and_acc(ax_dem, valid_dem)
scale_lan, _ = scale_and_acc(ax_lan, valid_lan)
DEM_TARGETS_LOWER = set(t.lower() for _, t in DEMONYM_COUNTRY)

def normalise_token(word, targets_lower_set):
    if word.lower() in targets_lower_set: return word.lower()
    return word

correct_global = 0; correct_cluster = 0; n_test = 0
for person, nat_exp, cluster_exp in TEST_PERSONS:
    ep, pid = get_emb(person)
    if ep is None: continue
    n_test += 1

    # Global axis
    r_global = nn_retrieve(ep + scale_all * ax_all, [pid])
    nat_global = r_global[0][0] if r_global else None
    hit_global = (nat_global == nat_exp)
    if hit_global: correct_global += 1

    # Cluster-aware axis
    result = assign_cluster(person)
    if result and result[0] in cluster_axes:
        cname = result[0]
        ax_c, coh_c, scale_c, _, _ = cluster_axes[cname]
        r_cluster = nn_retrieve(ep + scale_c * ax_c, [pid])
        nat_cluster = r_cluster[0][0] if r_cluster else None
    else:
        nat_cluster = nat_global  # fallback to global if no cluster
    hit_cluster = (nat_cluster == nat_exp)
    if hit_cluster: correct_cluster += 1

    print("  %-14s  global=%-12s [%s]  cluster=%-12s [%s]  expected=%s" % (
        person,
        nat_global  if nat_global  else '?', 'HIT' if hit_global  else '---',
        nat_cluster if nat_cluster else '?', 'HIT' if hit_cluster else '---',
        nat_exp))

print()
print("  Global axis hop 1:  %d/%d (%.0f%%)" % (correct_global, n_test, 100*correct_global/n_test))
print("  Cluster-aware hop 1: %d/%d (%.0f%%)" % (correct_cluster, n_test, 100*correct_cluster/n_test))
print()

# FULL 3-HOP CHAIN WITH CLUSTER-AWARE HOP 1
PERSON_LANGUAGE_3HOP = [
    ('Einstein', 'german',  'German',  'germany'),
    ('Kepler',   'german',  'German',  'germany'),
    ('Newton',   'english', 'British', 'britain'),
    ('Darwin',   'english', 'British', 'britain'),
    ('Napoleon', 'french',  'French',  'france'),
    ('Aristotle','greek',   'Greek',   'greece'),
    ('Plato',    'greek',   'Greek',   'greece'),
    ('Marx',     'german',  'German',  'germany'),
    ('Galileo',  'italian', 'Italian', 'italy'),
    ('Lenin',    'russian', 'Russian', 'russia'),
    ('Lincoln',  'english', 'American','america'),
]

print("SEQUENTIAL 3-HOP WITH CLUSTER-AWARE HOP 1:")
print("-"*65)
correct_3c = 0; correct_3g = 0
for person, lang_exp, nat_exp, cty_exp in PERSON_LANGUAGE_3HOP:
    ep, pid = get_emb(person)
    if ep is None: continue

    # --- CLUSTER-AWARE 3-hop ---
    result = assign_cluster(person)
    if result and result[0] in cluster_axes:
        ax_c, _, scale_c, _, _ = cluster_axes[result[0]]
        r1c = nn_retrieve(ep + scale_c * ax_c, [pid])
    else:
        r1c = nn_retrieve(ep + scale_all * ax_all, [pid])
    nat_c = r1c[0][0] if r1c else None
    e1c, i1c = get_emb(nat_c) if nat_c else (None, None)
    if e1c is not None:
        r2c = nn_retrieve(e1c + scale_dem * ax_dem, [i1c])
        cty_c = normalise_token(r2c[0][0], DEM_TARGETS_LOWER) if r2c else None
        e2c, i2c = get_emb(cty_c) if cty_c else (None, None)
        if e2c is not None:
            r3c = nn_retrieve(e2c + scale_lan * ax_lan, [i2c])
            lan_c = r3c[0][0] if r3c else None
        else: lan_c = None
    else: cty_c = None; lan_c = None
    hit_c = (lan_c == lang_exp)
    if hit_c: correct_3c += 1

    # --- GLOBAL 3-hop ---
    r1g = nn_retrieve(ep + scale_all * ax_all, [pid])
    nat_g = r1g[0][0] if r1g else None
    e1g, i1g = get_emb(nat_g) if nat_g else (None, None)
    if e1g is not None:
        r2g = nn_retrieve(e1g + scale_dem * ax_dem, [i1g])
        cty_g = normalise_token(r2g[0][0], DEM_TARGETS_LOWER) if r2g else None
        e2g, i2g = get_emb(cty_g) if cty_g else (None, None)
        if e2g is not None:
            r3g = nn_retrieve(e2g + scale_lan * ax_lan, [i2g])
            lan_g = r3g[0][0] if r3g else None
        else: lan_g = None
    else: cty_g = None; lan_g = None
    hit_g = (lan_g == lang_exp)
    if hit_g: correct_3g += 1

    print("  %-14s  nat_c=%-10s  cty_c=%-10s  lang_c=%-10s [c:%s g:%s]" % (
        person,
        nat_c if nat_c else '?',
        cty_c if cty_c else '?',
        lan_c if lan_c else '?',
        'HIT' if hit_c else '---',
        'HIT' if hit_g else '---'))

n = len(PERSON_LANGUAGE_3HOP)
print()
print("  Global 3-hop:        %d/%d (%.0f%%)" % (correct_3g, n, 100*correct_3g/n))
print("  Cluster-aware 3-hop: %d/%d (%.0f%%)" % (correct_3c, n, 100*correct_3c/n))
