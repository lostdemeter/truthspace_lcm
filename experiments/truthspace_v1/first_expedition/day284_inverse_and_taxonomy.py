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

print("DAY 284: INVERSE AXIS + BIOLOGICAL TAXONOMY")
print("="*65)
print()

# ====================================================================
# PART A: concept->field INVERSE axis (predicted many-to-one, high coh)
# ====================================================================
print("PART A: concept->field INVERSE axis (reversed from Day 283)")
print("-"*65)

CONCEPT_FIELD = [
    ('gravity',   'physics'),    ('motion',   'physics'),
    ('force',     'physics'),    ('energy',   'physics'),
    ('reaction',  'chemistry'),  ('element',  'chemistry'),
    ('evolution', 'biology'),    ('cell',     'biology'),
    ('proof',     'mathematics'),('number',   'mathematics'),
    ('logic',     'philosophy'), ('ethics',   'philosophy'),
    ('mind',      'psychology'), ('behavior', 'psychology'),
    ('market',    'economics'),  ('trade',    'economics'),
    ('orbit',     'astronomy'),  ('star',     'astronomy'),
    ('algorithm', 'computing'),  ('code',     'computing'),
    ('gene',      'genetics'),   ('mutation', 'genetics'),
]

ax_cf, coh_cf, valid_cf = compute_axis(CONCEPT_FIELD)
if ax_cf is not None:
    scale_cf, acc_cf = scale_and_acc(ax_cf, valid_cf)
    n = len(valid_cf)
    print("concept->field axis:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        coh_cf, scale_cf, acc_cf, n, 100*acc_cf/max(1,n)))
    print()
    for s, t in CONCEPT_FIELD:
        es, sid = get_emb(s)
        if es is None: print("  %-14s [SKIP]" % s); continue
        r = nn_retrieve(es + scale_cf * ax_cf, [sid])
        got = r[0][0] if r else '?'
        hit = 'HIT' if got == t else '---'
        print("  %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, hit))
else:
    print("  insufficient valid pairs")
print()

# Compare forward vs inverse
FIELD_CONCEPT_FWD = [
    ('physics','gravity'),('chemistry','reaction'),('biology','evolution'),
    ('mathematics','proof'),('philosophy','logic'),('psychology','mind'),
    ('economics','market'),('astronomy','orbit'),('computing','algorithm'),('genetics','gene'),
]
ax_fc_fwd, coh_fc_fwd, valid_fc_fwd = compute_axis(FIELD_CONCEPT_FWD)
if ax_fc_fwd is not None:
    scale_fc_fwd, acc_fc_fwd = scale_and_acc(ax_fc_fwd, valid_fc_fwd)
    nf = len(valid_fc_fwd)
    print("COMPARISON:")
    print("  field->concept (forward):  coh=%.4f  acc=%d/%d (%.0f%%)" % (
        coh_fc_fwd, acc_fc_fwd, nf, 100*acc_fc_fwd/max(1,nf)))
    if ax_cf is not None:
        ni = len(valid_cf)
        print("  concept->field (inverse):  coh=%.4f  acc=%d/%d (%.0f%%)" % (
            coh_cf, acc_cf, ni, 100*acc_cf/max(1,ni)))
    print()
    print("  Inverse coh vs forward coh: %.4f vs %.4f (delta=%.4f)" % (
        coh_cf if ax_cf is not None else 0, coh_fc_fwd,
        (coh_cf - coh_fc_fwd) if ax_cf is not None else 0))

print()

# ====================================================================
# PART B: animal->class TAXONOMY axis (new domain, one-to-one)
# ====================================================================
print("PART B: animal->class taxonomy axis")
print("-"*65)

ANIMAL_CLASS = [
    ('dog',     'mammal'),   ('cat',    'mammal'),   ('horse',  'mammal'),
    ('wolf',    'mammal'),   ('whale',  'mammal'),   ('dolphin','mammal'),
    ('lion',    'mammal'),   ('bear',   'mammal'),   ('deer',   'mammal'),
    ('rabbit',  'mammal'),   ('monkey', 'mammal'),   ('tiger',  'mammal'),
    ('eagle',   'bird'),     ('hawk',   'bird'),     ('owl',    'bird'),
    ('robin',   'bird'),     ('crow',   'bird'),     ('swan',   'bird'),
    ('parrot',  'bird'),     ('duck',   'bird'),
    ('salmon',  'fish'),     ('shark',  'fish'),     ('tuna',   'fish'),
    ('cod',     'fish'),     ('trout',  'fish'),
    ('frog',    'amphibian'),('toad',   'amphibian'),('newt',   'amphibian'),
    ('snake',   'reptile'),  ('lizard', 'reptile'),  ('turtle', 'reptile'),
    ('rose',    'plant'),    ('oak',    'plant'),    ('pine',   'plant'),
    ('fern',    'plant'),    ('ivy',    'plant'),
]

ax_ac, coh_ac, valid_ac = compute_axis(ANIMAL_CLASS)
if ax_ac is None: print("  insufficient valid pairs"); print()
else:
    scale_ac, acc_ac = scale_and_acc(ax_ac, valid_ac)
    n = len(valid_ac)
    print("animal->class axis:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        coh_ac, scale_ac, acc_ac, n, 100*acc_ac/max(1,n)))
    print()

    # Per-class breakdown
    for cls in ['mammal','bird','fish','amphibian','reptile','plant']:
        pairs_cls = [(s,t,sid,tid) for s,t,sid,tid in valid_ac if t == cls]
        if not pairs_cls: continue
        hits = sum(1 for s,t,sid,tid in pairs_cls
                   if nn_retrieve(W_E[sid]+scale_ac*ax_ac,[sid])[0][0]==t)
        print("  Class %-12s  %d/%d (%.0f%%)" % (cls, hits, len(pairs_cls), 100*hits/max(1,len(pairs_cls))))
    print()

    # Full pair results
    for s, t in ANIMAL_CLASS:
        es, sid = get_emb(s)
        if es is None: print("  %-12s [SKIP]" % s); continue
        if not any(x==s for x,_,_,_ in valid_ac): continue
        r = nn_retrieve(es + scale_ac * ax_ac, [sid])
        got = r[0][0] if r else '?'
        hit = 'HIT' if got == t else '---'
        print("  %-12s -> %-12s  got=%-12s [%s]" % (s, t, got, hit))
    print()

# ====================================================================
# PART C: animal CLUSTER axes (per-class)
# ====================================================================
print("PART C: Per-class cluster axes for animal->class")
print("-"*65)

CLASS_PAIRS = {
    'mammal':    [(s,'mammal') for s in ['dog','cat','horse','wolf','whale','dolphin','lion','bear','deer','rabbit','monkey','tiger','elephant','fox','sheep']],
    'bird':      [(s,'bird')   for s in ['eagle','hawk','owl','robin','crow','swan','parrot','duck','pigeon','sparrow','falcon','heron']],
    'fish':      [(s,'fish')   for s in ['salmon','shark','tuna','cod','trout','bass','carp','herring','pike','perch']],
    'reptile':   [(s,'reptile')for s in ['snake','lizard','turtle','crocodile','gecko','iguana']],
    'amphibian': [(s,'amphibian')for s in ['frog','toad','newt','salamander']],
    'plant':     [(s,'plant')  for s in ['rose','oak','pine','fern','ivy','maple','elm','cedar','tulip','daisy']],
}

cluster_axes = {}
cluster_centroids = {}
for cls, pairs in CLASS_PAIRS.items():
    ax, coh, valid = compute_axis(pairs)
    if ax is None or len(valid) < 2: continue
    scale, acc = scale_and_acc(ax, valid)
    cluster_axes[cls] = (ax, coh, scale, valid)
    embs = [W_E[sid] for _,_,sid,_ in valid]
    cluster_centroids[cls] = normed(np.mean(embs, axis=0))
    n = len(valid)
    print("  %-12s coh=%.4f scale=%.2f acc=%d/%d (%.0f%%)" % (
        cls, coh, scale, acc, n, 100*acc/max(1,n)))

print()

# Cluster assignment
def assign_class(word):
    e, wid = get_emb(word)
    if e is None: return None, 0.0
    en = normed(e).astype(np.float32)
    best_c, best_sim = None, -1.0
    for cname, cen in cluster_centroids.items():
        sim = float(np.dot(en, cen.astype(np.float32)))
        if sim > best_sim: best_sim = sim; best_c = cname
    return best_c, best_sim

print("Cluster-aware retrieval vs global axis:")
print("  %-12s  %-12s  global       cluster      expected" % ("Animal","True class"))
cluster_correct = 0; global_correct = 0; n_test = 0
TEST_ANIMALS = [
    ('dog','mammal'),('eagle','bird'),('salmon','fish'),('frog','amphibian'),
    ('snake','reptile'),('rose','plant'),('cat','mammal'),('hawk','bird'),
    ('shark','fish'),('oak','plant'),('lion','mammal'),('owl','bird'),
    ('toad','amphibian'),('lizard','reptile'),('wolf','mammal'),
]
for animal, cls_exp in TEST_ANIMALS:
    ea, aid = get_emb(animal)
    if ea is None: continue
    n_test += 1
    # Global
    if ax_ac is not None:
        rg = nn_retrieve(ea + scale_ac * ax_ac, [aid])
        got_g = rg[0][0] if rg else None
    else: got_g = None
    hit_g = (got_g == cls_exp)
    if hit_g: global_correct += 1
    # Cluster-aware
    cname, csim = assign_class(animal)
    if cname and cname in cluster_axes:
        ax_c, _, scale_c, _ = cluster_axes[cname]
        rc = nn_retrieve(ea + scale_c * ax_c, [aid])
        got_c = rc[0][0] if rc else None
    else: got_c = got_g
    hit_c = (got_c == cls_exp)
    if hit_c: cluster_correct += 1
    print("  %-12s  %-12s  %-10s[%s]  %-10s[%s]" % (
        animal, cls_exp,
        got_g if got_g else '?', 'HIT' if hit_g else '---',
        got_c if got_c else '?', 'HIT' if hit_c else '---'))

print()
print("  Global axis:         %d/%d (%.0f%%)" % (global_correct, n_test, 100*global_correct/max(1,n_test)))
print("  Cluster-aware axis:  %d/%d (%.0f%%)" % (cluster_correct, n_test, 100*cluster_correct/max(1,n_test)))
print()

# ====================================================================
# PART D: SUMMARY — reversibility and domain generalisation
# ====================================================================
print("SUMMARY TABLE — all axes tested Days 276-284:")
print("="*65)
print("%-30s  %6s  %6s  %s" % ("Axis", "coh", "acc%", "relation_type"))
print("-"*65)
rows = [
    ("person->nat (global)",      0.491, 41,  "mixed (attractor)"),
    ("person->nat (cluster)",     0.852, 100, "many-to-one"),
    ("nat->lang",                 0.583, 83,  "one-to-one"),
    ("2-hop person->lang",        "---", 87,  "chain (2 hops)"),
    ("person->field (global)",    0.497, 60,  "mixed (attractor)"),
    ("field->concept (fwd)",      0.401, 15,  "one-to-many FAILS"),
    ("concept->field (inv)",      coh_cf if ax_cf is not None else 0,
                                  int(100*acc_cf/max(1,len(valid_cf))) if ax_cf is not None else 0,
                                  "many-to-one (inverse)"),
    ("animal->class (global)",    coh_ac if ax_ac is not None else 0,
                                  int(100*acc_ac/max(1,len(valid_ac))) if ax_ac is not None else 0,
                                  "many-to-one"),
]
for name, coh, acc, rtype in rows:
    if isinstance(coh, float):
        print("  %-30s  %.3f  %4d%%  %s" % (name, coh, acc, rtype))
    else:
        print("  %-30s  %5s  %4d%%  %s" % (name, coh, acc, rtype))
