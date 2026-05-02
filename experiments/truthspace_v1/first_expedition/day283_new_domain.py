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

print("DAY 283: GENERALISATION — NEW KNOWLEDGE DOMAIN")
print("="*65)
print()
print("Testing whether the 2-hop cluster+axis architecture generalises")
print("beyond nationality/language to entirely new domains.")
print()

# ---- PART A: FIX Greek->english with cluster-specific nat->lang ----
print("PART A: Fix Greek/Polish via cluster-specific nat->lang axis")
print("-"*65)

GREEK_NAT_LANG = [('Greek','greek'),('Hellenic','greek'),('Ancient','greek')]
POLISH_NAT_LANG = [('Polish','polish'),('Slavic','polish')]  # test if axis helps
ax_gr_nl, coh_gr_nl, valid_gr_nl = compute_axis(GREEK_NAT_LANG)
if ax_gr_nl is not None and len(valid_gr_nl) > 0:
    scale_gr, _ = scale_and_acc(ax_gr_nl, valid_gr_nl)
    eg, gid = get_emb('Greek')
    if eg is not None:
        r = nn_retrieve(eg + scale_gr * ax_gr_nl, [gid])
        print("  Greek-specific nat->lang (scale=%.2f): Greek -> %s" % (scale_gr, r[0][0] if r else '?'))
else:
    print("  Greek-specific: insufficient valid pairs")

# Test: scan scales for Greek directly
eg, gid = get_emb('Greek')
if eg is not None:
    # Load the global nat->lang axis
    NAT_LANG = [('German','german'),('Austrian','german'),('British','english'),('American','english'),('French','french'),('Spanish','spanish'),('Russian','russian'),('Italian','italian'),('Japanese','japanese'),('Chinese','chinese'),('Polish','polish'),('Swiss','german'),('Roman','latin'),('Indian','hindi'),('Brazilian','portuguese')]
    ax_nl, coh_nl, valid_nl = compute_axis(NAT_LANG)
    print()
    print("  Scale scan for Greek with global nat->lang axis:")
    for s in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0, 1.17, 1.5, 2.0]:
        r = nn_retrieve(eg + s * ax_nl, [gid])
        hit = 'HIT' if (r and r[0][0] == 'greek') else '---'
        print("    scale=%.2f -> %-12s [%s]" % (s, r[0][0] if r else '?', hit))
    print()

# ---- PART B: NEW DOMAIN — scientist -> field of study ----
print("PART B: New domain — person -> field of study")
print("-"*65)

PERSON_FIELD = [
    ('Einstein','physics'),   ('Newton','physics'),    ('Kepler','astronomy'),
    ('Curie','chemistry'),    ('Darwin','biology'),     ('Mendel','genetics'),
    ('Euler','mathematics'),  ('Gauss','mathematics'), ('Pythagoras','mathematics'),
    ('Aristotle','philosophy'),('Plato','philosophy'), ('Kant','philosophy'),
    ('Freud','psychology'),   ('Jung','psychology'),   ('Pavlov','psychology'),
    ('Adam','economics'),     ('Marx','economics'),    ('Keynes','economics'),
    ('Turing','computing'),   ('Babbage','computing'),
]

ax_pf, coh_pf, valid_pf = compute_axis(PERSON_FIELD)
scale_pf, acc_pf = scale_and_acc(ax_pf, valid_pf) if ax_pf is not None else (1.0, 0)
print("person->field axis:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
    coh_pf, scale_pf, acc_pf, len(valid_pf), 100*acc_pf/max(1,len(valid_pf))))
print()

# Show which pairs are valid and check retrieval
print("  Per-pair results:")
for s, t in PERSON_FIELD:
    es, sid = get_emb(s)
    if es is None: print("  %-14s [SKIP - multi-token]" % s); continue
    r = nn_retrieve(es + scale_pf * ax_pf, [sid])
    hit = 'HIT' if (r and r[0][0] == t) else '---'
    got = r[0][0] if r else '?'
    print("  %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, hit))
print()

# ---- PART C: NEW DOMAIN — field -> key concept ----
print("PART C: New domain — field -> key concept")
print("-"*65)

FIELD_CONCEPT = [
    ('physics','gravity'),    ('physics','motion'),
    ('chemistry','reaction'), ('chemistry','element'),
    ('biology','evolution'),  ('biology','cell'),
    ('mathematics','proof'),  ('mathematics','number'),
    ('philosophy','logic'),   ('philosophy','ethics'),
    ('psychology','mind'),    ('psychology','behavior'),
    ('economics','market'),   ('economics','trade'),
    ('astronomy','orbit'),    ('astronomy','star'),
    ('computing','algorithm'),('computing','code'),
    ('genetics','gene'),      ('genetics','mutation'),
]

ax_fc, coh_fc, valid_fc = compute_axis(FIELD_CONCEPT)
if ax_fc is not None:
    scale_fc, acc_fc = scale_and_acc(ax_fc, valid_fc)
    print("field->concept axis:  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
        coh_fc, scale_fc, acc_fc, len(valid_fc), 100*acc_fc/max(1,len(valid_fc))))
    print()
    for s, t in FIELD_CONCEPT:
        es, sid = get_emb(s)
        if es is None: print("  %-16s [SKIP]" % s); continue
        r = nn_retrieve(es + scale_fc * ax_fc, [sid])
        hit = 'HIT' if (r and r[0][0] == t) else '---'
        got = r[0][0] if r else '?'
        print("  %-16s -> %-14s  got=%-14s [%s]" % (s, t, got, hit))
    print()
else:
    print("  field->concept: insufficient valid pairs")
    print()

# ---- PART D: 2-hop CHAIN: person -> field -> concept ----
print("PART D: 2-hop chain — person -> field -> concept")
print("-"*65)

PERSON_FIELD_CONCEPT = [
    ('Einstein',  'physics',     'gravity'),
    ('Newton',    'physics',     'gravity'),
    ('Darwin',    'biology',     'evolution'),
    ('Mendel',    'genetics',    'gene'),
    ('Curie',     'chemistry',   'element'),
    ('Euler',     'mathematics', 'proof'),
    ('Gauss',     'mathematics', 'proof'),
    ('Aristotle', 'philosophy',  'logic'),
    ('Freud',     'psychology',  'mind'),
    ('Marx',      'economics',   'market'),
    ('Turing',    'computing',   'algorithm'),
]

if ax_pf is not None and ax_fc is not None:
    correct_2h = 0; n_2h = 0
    print("  %-14s  field           concept         expected   ok?" % "Person")
    print("  " + "-"*62)
    for person, field_exp, concept_exp in PERSON_FIELD_CONCEPT:
        ep, pid = get_emb(person)
        if ep is None: continue
        n_2h += 1
        r1 = nn_retrieve(ep + scale_pf * ax_pf, [pid])
        field_got = r1[0][0] if r1 else None
        e1, i1 = get_emb(field_got) if field_got else (None, None)
        if e1 is not None:
            r2 = nn_retrieve(e1 + scale_fc * ax_fc, [i1])
            concept_got = r2[0][0] if r2 else None
        else:
            concept_got = None
        hit = (concept_got == concept_exp)
        if hit: correct_2h += 1
        print("  %-14s  %-14s  %-14s  %-10s [%s]" % (
            person,
            field_got if field_got else '?',
            concept_got if concept_got else '?',
            concept_exp,
            'HIT' if hit else '---'))
    print()
    print("  2-hop (person->field->concept): %d/%d (%.0f%%)" % (
        correct_2h, n_2h, 100*correct_2h/max(1,n_2h)))
else:
    print("  Insufficient axes for 2-hop test")

# ---- PART E: DOMAIN COMPARISON TABLE ----
print()
print("DOMAIN COMPARISON SUMMARY:")
print("="*65)
print("%-30s  %6s  %5s  %7s" % ("Axis", "coh", "scale", "acc%"))
print("-"*65)
for ax_name, ax, coh, valid in [
    ("nat->lang (Day 282)",   ax_nl  if ax_nl is not None else None,  coh_nl  if ax_nl is not None else 0,  valid_nl),
    ("person->field (D283)",  ax_pf  if ax_pf is not None else None,  coh_pf  if ax_pf is not None else 0,  valid_pf),
    ("field->concept (D283)", ax_fc  if ax_fc is not None else None,  coh_fc  if ax_fc is not None else 0,  valid_fc),
]:
    if ax is None:
        print("  %-30s  SKIP" % ax_name); continue
    scale, acc = scale_and_acc(ax, valid)
    n = len(valid)
    pct = 100*acc/max(1,n)
    print("  %-30s  %.3f  %5.2f  %5.0f%% (%d/%d)" % (ax_name, coh, scale, pct, acc, n))
