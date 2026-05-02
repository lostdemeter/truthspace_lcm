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
def scale_and_acc(axis, valid_pairs, n_scales=30):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = sum(1 for src, tgt, sid, tid in valid_pairs
                      if nn_retrieve(W_E[sid] + s * axis, [sid])[0][0] == tgt)
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

print("DAY 282: DIRECT NATIONALITY -> LANGUAGE AXIS")
print("="*65)
print()

# DIRECT nat->language pairs (bypasses country intermediate entirely)
# This collapses hops 2+3 into a single hop
NAT_LANG = [
    ('German',   'german'),    ('Austrian', 'german'),
    ('British',  'english'),   ('American', 'english'),
    ('French',   'french'),    ('Spanish',  'spanish'),
    ('Russian',  'russian'),   ('Greek',    'greek'),
    ('Italian',  'italian'),   ('Japanese', 'japanese'),
    ('Chinese',  'chinese'),   ('Polish',   'polish'),
    ('Swedish',  'swedish'),   ('Norwegian','norwegian'),
    ('Danish',   'danish'),    ('Finnish',  'finnish'),
    ('Hungarian','hungarian'), ('Portuguese','portuguese'),
    ('Turkish',  'turkish'),   ('Dutch',    'dutch'),
    ('Swiss',    'german'),    ('Serbian',  'serbian'),
    ('Roman',    'latin'),     ('Indian',   'hindi'),
    ('Brazilian','portuguese'),
]

ax_nl, coh_nl, valid_nl = compute_axis(NAT_LANG)
scale_nl, acc_nl = scale_and_acc(ax_nl, valid_nl)
print("Direct nat->lang axis:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
    coh_nl, scale_nl, acc_nl, len(valid_nl), 100*acc_nl/len(valid_nl)))
print("  Valid pairs (%d): %s" % (len(valid_nl), [s for s,t,_,_ in valid_nl]))
print()

# Compare with the old 3-hop: dem->country + country->lang
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

# Cluster axes from Day 281
CLUSTERS = {
    'German':    [('Einstein','German'),('Marx','German'),('Kepler','German'),('Gauss','German'),('Kant','German'),('Planck','German'),('Schiller','German')],
    'Austrian':  [('Freud','Austrian'),('Mozart','Austrian'),('Schubert','Austrian'),('Haydn','Austrian')],
    'British':   [('Newton','British'),('Darwin','British'),('Turing','British'),('Churchill','British'),('Shakespeare','British'),('Faraday','British'),('Dickens','British'),('Austen','British'),('Locke','British'),('Hobbes','British')],
    'French':    [('Napoleon','French'),('Curie','French'),('Descartes','French'),('Voltaire','French'),('Rousseau','French'),('Hugo','French'),('Pasteur','French')],
    'Greek':     [('Aristotle','Greek'),('Plato','Greek'),('Pythagoras','Greek'),('Socrates','Greek'),('Homer','Greek'),('Euclid','Greek'),('Archimedes','Greek')],
    'American':  [('Lincoln','American'),('Franklin','American'),('Edison','American'),('Washington','American'),('Jefferson','American'),('Thoreau','American'),('Whitman','American')],
    'Italian':   [('Galileo','Italian'),('Leonardo','Italian'),('Dante','Italian'),('Michelangelo','Italian'),('Machiavelli','Italian'),('Vivaldi','Italian')],
    'Russian':   [('Lenin','Russian'),('Tolstoy','Russian'),('Dostoevsky','Russian'),('Pushkin','Russian'),('Tchaikovsky','Russian'),('Stalin','Russian')],
}
cluster_axes = {}
cluster_centroids = {}
for cname, pairs in CLUSTERS.items():
    ax, coh, valid = compute_axis(pairs)
    if ax is None or len(valid) < 2: continue
    scale, acc = scale_and_acc(ax, valid)
    cluster_axes[cname] = (ax, coh, scale, valid)
    embs = [W_E[sid] for _, _, sid, _ in valid]
    cluster_centroids[cname] = normed(np.mean(embs, axis=0))

ALL_NAT_PAIRS = [p for pairs in CLUSTERS.values() for p in pairs]
ax_all, coh_all, valid_all = compute_axis(ALL_NAT_PAIRS)
scale_all, _ = scale_and_acc(ax_all, valid_all)

def assign_cluster(word):
    e, wid = get_emb(word)
    if e is None: return None, 0.0
    en = normed(e).astype(np.float32)
    best_c, best_sim = None, -1.0
    for cname, cen in cluster_centroids.items():
        sim = float(np.dot(en, cen.astype(np.float32)))
        if sim > best_sim: best_sim = sim; best_c = cname
    return best_c, best_sim

# Show per-cluster coherence
print("Per-cluster nat axes (split German/Austrian this time):")
for cname, (ax, coh, scale, valid) in cluster_axes.items():
    acc = sum(1 for s, t, sid, _ in valid
              if nn_retrieve(W_E[sid] + scale * ax, [sid])[0][0] == t)
    print("  %-12s coh=%.4f scale=%.2f acc=%d/%d" % (cname, coh, scale, acc, len(valid)))
print()

# FULL TEST: 2-hop (cluster nat + direct nat->lang) vs 3-hop
TEST = [
    ('Einstein',   'german',     'German'),
    ('Kepler',     'german',     'German'),
    ('Newton',     'english',    'British'),
    ('Darwin',     'english',    'British'),
    ('Turing',     'english',    'British'),
    ('Napoleon',   'french',     'French'),
    ('Aristotle',  'greek',      'Greek'),
    ('Plato',      'greek',      'Greek'),
    ('Mozart',     'german',     'Austrian'),
    ('Marx',       'german',     'German'),
    ('Galileo',    'italian',    'Italian'),
    ('Lenin',      'russian',    'Russian'),
    ('Lincoln',    'english',    'American'),
    ('Franklin',   'english',    'American'),
    ('Edison',     'english',    'American'),
    ('Freud',      'german',     'Austrian'),
    ('Dostoevsky', 'russian',    'Russian'),
    ('Descartes',  'french',     'French'),
    ('Pythagoras', 'greek',      'Greek'),
]

print("COMPARISON: 2-hop cluster+nat->lang  vs  3-hop cluster+dem+lang")
print("-"*65)
correct_2h = 0; correct_3h = 0; n_test = 0

for person, lang_exp, nat_exp in TEST:
    ep, pid = get_emb(person)
    if ep is None: continue
    n_test += 1

    # Hop 1: cluster-aware person -> nationality
    cname, csim = assign_cluster(person)
    if cname and cname in cluster_axes:
        ax_c, _, scale_c, _ = cluster_axes[cname]
        r1 = nn_retrieve(ep + scale_c * ax_c, [pid])
    else:
        r1 = nn_retrieve(ep + scale_all * ax_all, [pid])
    nat = r1[0][0] if r1 else None
    e1, i1 = get_emb(nat) if nat else (None, None)

    # 2-HOP: nat -> language (direct)
    if e1 is not None:
        r2_2h = nn_retrieve(e1 + scale_nl * ax_nl, [i1])
        lang_2h = r2_2h[0][0] if r2_2h else None
    else:
        lang_2h = None
    hit_2h = (lang_2h == lang_exp)
    if hit_2h: correct_2h += 1

    # 3-HOP: nat -> country -> language
    if e1 is not None:
        r2_3h = nn_retrieve(e1 + scale_dem * ax_dem, [i1])
        cty_raw = r2_3h[0][0] if r2_3h else None
        cty = normalise_token(cty_raw, DEM_TARGETS_LOWER) if cty_raw else None
        e2, i2 = get_emb(cty) if cty else (None, None)
        if e2 is not None:
            r3 = nn_retrieve(e2 + scale_lan * ax_lan, [i2])
            lang_3h = r3[0][0] if r3 else None
        else: lang_3h = None
    else:
        cty = None; lang_3h = None
    hit_3h = (lang_3h == lang_exp)
    if hit_3h: correct_3h += 1

    print("  %-14s nat=%-10s  2h:%-10s[%s]  3h:%-10s[%s]  exp=%s" % (
        person,
        nat if nat else '?',
        lang_2h if lang_2h else '?', 'HIT' if hit_2h else '---',
        lang_3h if lang_3h else '?', 'HIT' if hit_3h else '---',
        lang_exp))

print()
print("  2-hop (cluster nat + nat->lang): %d/%d (%.0f%%)" % (correct_2h, n_test, 100*correct_2h/n_test))
print("  3-hop (cluster nat + dem + lan): %d/%d (%.0f%%)" % (correct_3h, n_test, 100*correct_3h/n_test))
print()

# Also test the nat->lang axis alone (without the nat clustering step)
# to see if it provides value as a direct axis
print("DIRECT nat->lang SINGLE-HOP TEST (nationality -> language):")
print("-"*65)
direct_correct = 0; direct_n = 0
for nat_word, lang_word in [('German','german'),('British','english'),('French','french'),('Greek','greek'),('Russian','russian'),('American','english'),('Italian','italian'),('Austrian','german'),('Spanish','spanish'),('Japanese','japanese'),('Chinese','chinese'),('Polish','polish')]:
    e, eid = get_emb(nat_word)
    if e is None: continue
    direct_n += 1
    r = nn_retrieve(e + scale_nl * ax_nl, [eid])
    got = r[0][0] if r else None
    hit = (got == lang_word)
    if hit: direct_correct += 1
    print("  %-12s -> %-12s  got=%-12s [%s]" % (nat_word, lang_word, got if got else '?', 'HIT' if hit else '---'))
print()
print("  Nat->lang direct: %d/%d (%.0f%%)" % (direct_correct, direct_n, 100*direct_correct/direct_n))
