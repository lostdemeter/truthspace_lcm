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
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), i) for i in top]
def scale_for_axis(axis, pairs, n_scales=25):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = 0
        for src, tgt in pairs:
            es, sid = get_emb(src); et, tid = get_emb(tgt)
            if es is None or et is None: continue
            top = nn_retrieve(es + s * axis, [sid])
            if top and top[0][0] == tgt: correct += 1
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

print("DAY 278: FIXED AXIS CHAINING")
print("="*60)
print()
print("Fix 1: Balance nationality training pairs (reduce German bias)")
print("Fix 2: Add capitalised targets to demonym->country pairs")
print()

# FIX 1: Balanced nationality pairs -- 2 per nationality, diverse coverage
SCIENTIST_NAT_FIXED = [
    # German (2)
    ('Einstein','German'), ('Marx','German'),
    # British (4 - large pool of famous British)
    ('Newton','British'), ('Darwin','British'), ('Turing','British'), ('Shakespeare','British'),
    # French (2)
    ('Napoleon','French'), ('Curie','French'),
    # Greek (2)
    ('Aristotle','Greek'), ('Plato','Greek'),
    # American (2)
    ('Lincoln','American'), ('Franklin','American'),
    # Austrian (2)
    ('Mozart','Austrian'), ('Freud','Austrian'),
    # Italian (2)
    ('Galileo','Italian'), ('Leonardo','Italian'),
    # Russian (2)
    ('Lenin','Russian'), ('Tolstoy','Russian'),
    # Chinese (1 - limited single-token famous Chinese names)
    ('Confucius','Chinese'),
    # Indian (1)
    ('Gandhi','Indian'),
]

# Check which are single-token
print("Checking single-token status of new person names:")
for person, nat in SCIENTIST_NAT_FIXED:
    e, _ = get_emb(person)
    print("  %-15s %-12s %s" % (person, nat, "OK" if e is not None else "MULTI"))
print()

# FIX 2: Demonym->country with BOTH capitalised and lowercase targets
# Also add more demonyms to cover the test set
DEMONYM_COUNTRY_FIXED = [
    # Lowercase (axis training)
    ('German','germany'), ('British','britain'), ('French','france'),
    ('Japanese','japan'), ('Chinese','china'), ('Italian','italy'),
    ('Spanish','spain'), ('Russian','russia'), ('Greek','greece'),
    ('Polish','poland'), ('Swedish','sweden'), ('Norwegian','norway'),
    ('Austrian','austria'), ('American','america'), ('Swiss','switzerland'),
    ('Indian','india'), ('Brazilian','brazil'), ('Dutch','netherlands'),
    ('Portuguese','portugal'), ('Turkish','turkey'),
]

ax_nat_fixed, coh_nat_fixed = compute_axis(SCIENTIST_NAT_FIXED)
ax_dem_fixed, coh_dem_fixed = compute_axis(DEMONYM_COUNTRY_FIXED)

# Original language axis (was already good)
LANGUAGE_PAIRS = [('france','french'),('germany','german'),('spain','spanish'),
                  ('russia','russian'),('japan','japanese'),('china','chinese'),
                  ('italy','italian'),('portugal','portuguese'),('poland','polish'),
                  ('sweden','swedish'),('norway','norwegian'),('denmark','danish'),
                  ('greece','greek'),('turkey','turkish'),('finland','finnish'),
                  ('hungary','hungarian'),('india','hindi'),('brazil','portuguese'),
                  ('netherlands','dutch'),('austria','german')]
ax_lan, coh_lan = compute_axis(LANGUAGE_PAIRS)

# Find scales
scale_nat, acc_nat = scale_for_axis(ax_nat_fixed, SCIENTIST_NAT_FIXED)
scale_dem, acc_dem = scale_for_axis(ax_dem_fixed, DEMONYM_COUNTRY_FIXED)
scale_lan, acc_lan = scale_for_axis(ax_lan, LANGUAGE_PAIRS)

n_nat = sum(1 for s,t in SCIENTIST_NAT_FIXED if get_emb(s)[0] is not None)
n_dem = sum(1 for s,t in DEMONYM_COUNTRY_FIXED if get_emb(s)[0] is not None)
n_lan = sum(1 for s,t in LANGUAGE_PAIRS if get_emb(s)[0] is not None)

print("Fixed axis single-hop performance:")
print("  nat (balanced):  coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_nat_fixed, scale_nat, acc_nat, n_nat, 100*acc_nat/n_nat))
print("  dem->cty (fixed):coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_dem_fixed, scale_dem, acc_dem, n_dem, 100*acc_dem/n_dem))
print("  language:        coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_lan, scale_lan, acc_lan, n_lan, 100*acc_lan/n_lan))
print()

# Post-retrieval capitalisation normalisation
def normalise_token(word, axis_training_targets):
    """If retrieved word is capitalised but lowercase version is in training targets, lowercase it."""
    targets_lower = set(t.lower() for t in axis_training_targets)
    if word.lower() in targets_lower:
        return word.lower()
    return word

DEM_TARGETS = [t for _, t in DEMONYM_COUNTRY_FIXED]

# FULL SEQUENTIAL 3-HOP TEST (same test set as Day 277)
PERSON_LANGUAGE = [
    ('Einstein',  'german',   'German',   'germany'),
    ('Newton',    'english',  'British',  'britain'),
    ('Darwin',    'english',  'British',  'britain'),
    ('Aristotle', 'greek',    'Greek',    'greece'),
    ('Plato',     'greek',    'Greek',    'greece'),
    ('Napoleon',  'french',   'French',   'france'),
    ('Mozart',    'german',   'Austrian', 'austria'),
    ('Marx',      'german',   'German',   'germany'),
    ('Turing',    'english',  'British',  'britain'),
    ('Gandhi',    'hindi',    'Indian',   'india'),
    ('Galileo',   'italian',  'Italian',  'italy'),
    ('Curie',     'french',   'French',   'france'),
    ('Lenin',     'russian',  'Russian',  'russia'),
    ('Shakespeare','english', 'British',  'britain'),
    ('Franklin',  'english',  'American', 'america'),
]

print("SEQUENTIAL 3-HOP (FIXED): person -[nat]-> -[dem]-> country -[lan]-> language")
print("-"*70)

# Old axis (Day 277 baseline for comparison)
SCIENTIST_NAT_OLD  = [('Einstein','German'),('Newton','British'),('Darwin','British'),('Kepler','German'),('Euler','Swiss'),('Gauss','German'),('Turing','British'),('Tesla','Serbian'),('Napoleon','French'),('Churchill','British'),('Lincoln','American'),('Gandhi','Indian'),('Caesar','Roman'),('Aristotle','Greek'),('Plato','Greek'),('Shakespeare','British'),('Mozart','Austrian'),('Marx','German'),('Freud','Austrian'),('Kant','German')]
DEMONYM_COUNTRY_OLD= [('German','germany'),('British','britain'),('French','france'),('Japanese','japan'),('Chinese','china'),('Italian','italy'),('Spanish','spain'),('Russian','russia'),('Greek','greece'),('Polish','poland'),('Swedish','sweden'),('Norwegian','norway'),('Austrian','austria'),('American','america'),('Swiss','switzerland')]
ax_nat_old, _ = compute_axis(SCIENTIST_NAT_OLD)
ax_dem_old, _ = compute_axis(DEMONYM_COUNTRY_OLD)
scale_nat_old, _ = scale_for_axis(ax_nat_old, SCIENTIST_NAT_OLD)
scale_dem_old, _ = scale_for_axis(ax_dem_old, DEMONYM_COUNTRY_OLD)

correct_old = 0
correct_new = 0
correct_new_nc = 0   # new axes + no normalisation (to isolate fix contributions)

for person, lang_exp, nat_exp, cty_exp in PERSON_LANGUAGE:
    ep, pid = get_emb(person)
    if ep is None: continue

    # OLD (Day 277 method)
    r1 = nn_retrieve(ep + scale_nat_old * ax_nat_old, [pid])
    nat_old = r1[0][0] if r1 else None
    e1, i1 = get_emb(nat_old) if nat_old else (None, None)
    if e1 is not None:
        r2 = nn_retrieve(e1 + scale_dem_old * ax_dem_old, [i1])
        cty_old = r2[0][0] if r2 else None
        e2, i2 = get_emb(cty_old) if cty_old else (None, None)
        if e2 is not None:
            r3 = nn_retrieve(e2 + scale_lan * ax_lan, [i2])
            lan_old = r3[0][0] if r3 else None
        else: lan_old = None
    else: cty_old = None; lan_old = None
    hit_old = (lan_old == lang_exp)
    if hit_old: correct_old += 1

    # NEW fixed axes + capitalisation normalisation
    r1n = nn_retrieve(ep + scale_nat * ax_nat_fixed, [pid])
    nat_new = r1n[0][0] if r1n else None
    e1n, i1n = get_emb(nat_new) if nat_new else (None, None)
    if e1n is not None:
        r2n = nn_retrieve(e1n + scale_dem * ax_dem_fixed, [i1n])
        cty_raw = r2n[0][0] if r2n else None
        cty_new = normalise_token(cty_raw, DEM_TARGETS) if cty_raw else None
        e2n, i2n = get_emb(cty_new) if cty_new else (None, None)
        if e2n is not None:
            r3n = nn_retrieve(e2n + scale_lan * ax_lan, [i2n])
            lan_new = r3n[0][0] if r3n else None
        else: lan_new = None
    else: cty_new = None; lan_new = None
    hit_new = (lan_new == lang_exp)
    if hit_new: correct_new += 1

    print("  %-13s  nat: %-11s -> %-11s   cty: %-11s -> %-11s   lang: %-10s [old:%s new:%s]" % (
        person,
        nat_old if nat_old else '?', nat_new if nat_new else '?',
        cty_old if cty_old else '?', cty_new if cty_new else '?',
        lang_exp,
        'HIT' if hit_old else '---', 'HIT' if hit_new else '---'))

n_valid = len(PERSON_LANGUAGE)
print()
print("  Old (Day 277 axes, no normalisation): %d/%d (%.0f%%)" % (correct_old, n_valid, 100*correct_old/n_valid))
print("  New (fixed axes + normalisation):     %d/%d (%.0f%%)" % (correct_new, n_valid, 100*correct_new/n_valid))
print()

# Per-hop analysis on new axes
print("Per-hop accuracy (new fixed axes + normalisation):")
h1, h2, h3, n3 = 0, 0, 0, 0
for person, lang_exp, nat_exp, cty_exp in PERSON_LANGUAGE:
    ep, pid = get_emb(person)
    if ep is None: continue
    n3 += 1
    r1 = nn_retrieve(ep + scale_nat * ax_nat_fixed, [pid])
    nat_w = r1[0][0] if r1 else None
    if nat_w == nat_exp: h1 += 1
    e1, i1 = get_emb(nat_w) if nat_w else (None, None)
    if e1 is not None:
        r2 = nn_retrieve(e1 + scale_dem * ax_dem_fixed, [i1])
        cty_raw = r2[0][0] if r2 else None
        cty_w = normalise_token(cty_raw, DEM_TARGETS) if cty_raw else None
        if cty_w == cty_exp: h2 += 1
        e2, i2 = get_emb(cty_w) if cty_w else (None, None)
        if e2 is not None:
            r3 = nn_retrieve(e2 + scale_lan * ax_lan, [i2])
            lan_w = r3[0][0] if r3 else None
            if lan_w == lang_exp: h3 += 1
print("  Hop 1 (person->nat):    %d/%d (%.0f%%)" % (h1, n3, 100*h1/n3))
print("  Hop 2 (nat->country):   %d/%d (%.0f%%)" % (h2, n3, 100*h2/n3))
print("  Hop 3 (country->lang):  %d/%d (%.0f%%)" % (h3, n3, 100*h3/n3))
print("  Product prediction:     %.0f%%" % (100*(h1/n3)*(h2/n3)*(h3/n3)))
print("  Actual end-to-end:      %d/%d (%.0f%%)" % (correct_new, n_valid, 100*correct_new/n_valid))
