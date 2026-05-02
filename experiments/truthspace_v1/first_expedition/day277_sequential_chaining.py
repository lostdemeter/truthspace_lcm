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
def scale_for_axis(axis, pairs, n_scales=20):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.1, 3.0, n_scales):
        correct = 0
        for src, tgt in pairs:
            es, sid = get_emb(src); et, tid = get_emb(tgt)
            if es is None or et is None: continue
            top = nn_retrieve(es + s * axis, [sid])
            if top and top[0][0] == tgt: correct += 1
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

# --- Build axes ---
CAPITAL_PAIRS  = [('france','paris'),('germany','berlin'),('japan','tokyo'),('china','beijing'),('italy','rome'),('spain','madrid'),('russia','moscow'),('india','delhi'),('brazil','brasilia'),('canada','ottawa'),('egypt','cairo'),('greece','athens'),('turkey','ankara'),('poland','warsaw'),('sweden','stockholm'),('austria','vienna')]
LANGUAGE_PAIRS = [('france','french'),('germany','german'),('spain','spanish'),('russia','russian'),('japan','japanese'),('china','chinese'),('italy','italian'),('portugal','portuguese'),('poland','polish'),('sweden','swedish'),('norway','norwegian'),('denmark','danish'),('greece','greek'),('turkey','turkish'),('finland','finnish'),('hungary','hungarian')]
SCIENTIST_NAT  = [('Einstein','German'),('Newton','British'),('Darwin','British'),('Kepler','German'),('Euler','Swiss'),('Gauss','German'),('Turing','British'),('Tesla','Serbian'),('Napoleon','French'),('Churchill','British'),('Lincoln','American'),('Gandhi','Indian'),('Caesar','Roman'),('Aristotle','Greek'),('Plato','Greek'),('Shakespeare','British'),('Mozart','Austrian'),('Marx','German'),('Freud','Austrian'),('Kant','German')]
DEMONYM_COUNTRY= [('German','germany'),('British','britain'),('French','france'),('Japanese','japan'),('Chinese','china'),('Italian','italy'),('Spanish','spain'),('Russian','russia'),('Greek','greece'),('Polish','poland'),('Swedish','sweden'),('Norwegian','norway'),('Austrian','austria'),('American','america'),('Swiss','switzerland')]

ax_cap, coh_cap = compute_axis(CAPITAL_PAIRS)
ax_lan, coh_lan = compute_axis(LANGUAGE_PAIRS)
ax_nat, coh_nat = compute_axis(SCIENTIST_NAT)
ax_dem, coh_dem = compute_axis(DEMONYM_COUNTRY)

scale_cap, acc_cap = scale_for_axis(ax_cap, CAPITAL_PAIRS)
scale_lan, acc_lan = scale_for_axis(ax_lan, LANGUAGE_PAIRS)
scale_nat, acc_nat = scale_for_axis(ax_nat, SCIENTIST_NAT)
scale_dem, acc_dem = scale_for_axis(ax_dem, DEMONYM_COUNTRY)

n_cap = sum(1 for s,t in CAPITAL_PAIRS if get_emb(s)[0] is not None)
n_lan = sum(1 for s,t in LANGUAGE_PAIRS if get_emb(s)[0] is not None)
n_nat = sum(1 for s,t in SCIENTIST_NAT if get_emb(s)[0] is not None)
n_dem = sum(1 for s,t in DEMONYM_COUNTRY if get_emb(s)[0] is not None)

print("DAY 277: SEQUENTIAL AXIS CHAINING")
print("="*60)
print()
print("Single-hop baselines:")
print("  capital:     coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_cap, scale_cap, acc_cap, n_cap, 100*acc_cap/n_cap))
print("  language:    coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_lan, scale_lan, acc_lan, n_lan, 100*acc_lan/n_lan))
print("  person->nat: coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_nat, scale_nat, acc_nat, n_nat, 100*acc_nat/n_nat))
print("  demonym->cty:coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (coh_dem, scale_dem, acc_dem, n_dem, 100*acc_dem/n_dem))
print()

# --- SEQUENTIAL 2-HOP: country -> capital -> ??? ---
# We'll test country -[cap]-> city -[city_to_?]-> ?
# But first test: does sequential chaining through capital actually improve
# on just direct language?
# country -[lan]-> language (direct, 1 hop)
# country -[cap]-> capital (nn_grounded) -[city_to_country]-> country (back??)
# Better: test sequential chain that we KNOW should work if each step works

# MAIN TEST A: Sequential 2-hop: person -[nat]-> nationality -[dem_cty]-> country
# Compare: additive (Day 276 approach) vs sequential (NN grounding at each step)
print("SEQUENTIAL vs ADDITIVE: person -[nat]-> nationality -[dem_cty]-> country")
print("-"*60)
PERSON_COUNTRY = [
    ('Einstein', 'Germany',   'German'),
    ('Newton',   'Britain',   'British'),
    ('Darwin',   'Britain',   'British'),
    ('Kepler',   'Germany',   'German'),
    ('Gauss',    'Germany',   'German'),
    ('Turing',   'Britain',   'British'),
    ('Napoleon', 'France',    'French'),
    ('Caesar',   'Rome',      'Roman'),
    ('Aristotle','Greece',    'Greek'),
    ('Plato',    'Greece',    'Greek'),
    ('Mozart',   'Austria',   'Austrian'),
    ('Marx',     'Germany',   'German'),
]
# Note: DEMONYM_COUNTRY uses lowercase countries. Let's check.
print("  (expected country targets, lowercase form used in axis training)")

correct_seq = 0
correct_add = 0
correct_1hop= 0
for person, country_expected, nat_expected in PERSON_COUNTRY:
    ep, pid = get_emb(person)
    if ep is None: continue
    country_lower = country_expected.lower()
    ec_nat, nat_id = get_emb(nat_expected)
    ec_cty, cty_id = get_emb(country_lower)

    # Step 1: person -> nationality
    pred_nat = ep + scale_nat * ax_nat
    top_nat  = nn_retrieve(pred_nat, [pid])
    nat_word = top_nat[0][0] if top_nat else None
    nat_hit  = (nat_word == nat_expected)

    # Sequential: use retrieved nationality word's actual embedding
    e_nat_ret, nat_ret_id = get_emb(nat_word) if nat_word else (None, None)
    if e_nat_ret is not None:
        pred_cty_seq = e_nat_ret + scale_dem * ax_dem
        top_cty_seq  = nn_retrieve(pred_cty_seq, [nat_ret_id])
        cty_seq = top_cty_seq[0][0] if top_cty_seq else None
    else:
        cty_seq = None
    hit_seq = (cty_seq == country_lower)
    if hit_seq: correct_seq += 1

    # Additive: sum both axes at once
    pred_cty_add = ep + scale_nat * ax_nat + scale_dem * ax_dem
    top_cty_add  = nn_retrieve(pred_cty_add, [pid])
    cty_add = top_cty_add[0][0] if top_cty_add else None
    hit_add = (cty_add == country_lower)
    if hit_add: correct_add += 1

    # Direct single-hop for nat (already has scale)
    if nat_hit: correct_1hop += 1

    print("  %-12s nat->%-10s[%s]  seq_cty=%-12s[%s]  add_cty=%-12s[%s]" % (
        person,
        nat_word if nat_word else '?', 'HIT' if nat_hit else '---',
        cty_seq  if cty_seq  else '?', 'HIT' if hit_seq  else '---',
        cty_add  if cty_add  else '?', 'HIT' if hit_add  else '---'))

n_valid = len(PERSON_COUNTRY)
print()
print("  Nat retrieval (step 1):     %d/%d (%.0f%%)" % (correct_1hop, n_valid, 100*correct_1hop/n_valid))
print("  Sequential 2-hop accuracy: %d/%d (%.0f%%)" % (correct_seq, n_valid, 100*correct_seq/n_valid))
print("  Additive 2-hop accuracy:   %d/%d (%.0f%%)" % (correct_add, n_valid, 100*correct_add/n_valid))
print()

# --- SEQUENTIAL 3-HOP: person -> nationality -> country -> language ---
print("SEQUENTIAL 3-HOP: person -[nat]-> nationality -[dem]-> country -[lan]-> language")
print("-"*60)
PERSON_LANGUAGE = [
    ('Einstein', 'german',   'German',  'germany'),
    ('Newton',   'english',  'British', 'britain'),
    ('Darwin',   'english',  'British', 'britain'),
    ('Kepler',   'german',   'German',  'germany'),
    ('Gauss',    'german',   'German',  'germany'),
    ('Napoleon', 'french',   'French',  'france'),
    ('Aristotle','greek',    'Greek',   'greece'),
    ('Plato',    'greek',    'Greek',   'greece'),
    ('Mozart',   'german',   'Austrian','austria'),
    ('Marx',     'german',   'German',  'germany'),
]
correct_3seq = 0
correct_3add = 0
for person, lang_expected, nat_exp, cty_exp in PERSON_LANGUAGE:
    ep, pid = get_emb(person)
    if ep is None: continue

    # Sequential 3-hop
    # hop 1: person -> nationality
    top1 = nn_retrieve(ep + scale_nat * ax_nat, [pid])
    nat_w = top1[0][0] if top1 else None
    e_nat, nid = get_emb(nat_w) if nat_w else (None, None)

    # hop 2: nationality -> country
    if e_nat is not None:
        top2 = nn_retrieve(e_nat + scale_dem * ax_dem, [nid])
        cty_w = top2[0][0] if top2 else None
        e_cty, ctid = get_emb(cty_w) if cty_w else (None, None)
    else:
        cty_w, e_cty, ctid = None, None, None

    # hop 3: country -> language
    if e_cty is not None:
        top3 = nn_retrieve(e_cty + scale_lan * ax_lan, [ctid])
        lan_w = top3[0][0] if top3 else None
    else:
        lan_w = None
    hit_3seq = (lan_w == lang_expected)
    if hit_3seq: correct_3seq += 1

    # Additive 3-hop (baseline from Day 276)
    pred_add = ep + scale_nat*ax_nat + scale_dem*ax_dem + scale_lan*ax_lan
    top_add  = nn_retrieve(pred_add, [pid])
    lan_add  = top_add[0][0] if top_add else None
    hit_add  = (lan_add == lang_expected)
    if hit_add: correct_3add += 1

    print("  %-12s  nat=%-10s  cty=%-10s  seq_lan=%-10s[%s]  add=%-10s[%s]" % (
        person,
        nat_w  if nat_w  else '?',
        cty_w  if cty_w  else '?',
        lan_w  if lan_w  else '?', 'HIT' if hit_3seq else '---',
        lan_add if lan_add else '?', 'HIT' if hit_add  else '---'))

n_valid3 = len(PERSON_LANGUAGE)
print()
print("  Sequential 3-hop accuracy: %d/%d (%.0f%%)" % (correct_3seq, n_valid3, 100*correct_3seq/n_valid3))
print("  Additive   3-hop accuracy: %d/%d (%.0f%%)" % (correct_3add, n_valid3, 100*correct_3add/n_valid3))
predicted = (acc_nat/n_nat) * (acc_dem/n_dem) * (acc_lan/n_lan)
print("  Predicted (product of singles): %.0f%%" % (100*predicted))
print()

# --- ERROR ANALYSIS: where does sequential fail? ---
print("ERROR ANALYSIS: intermediate hop success rates in sequential 3-hop")
correct_h1, correct_h2, correct_h3 = 0, 0, 0
n3 = 0
for person, lang_expected, nat_exp, cty_exp in PERSON_LANGUAGE:
    ep, pid = get_emb(person)
    if ep is None: continue
    n3 += 1
    top1 = nn_retrieve(ep + scale_nat * ax_nat, [pid])
    nat_w = top1[0][0] if top1 else None
    if nat_w == nat_exp: correct_h1 += 1
    e_nat, nid = get_emb(nat_w) if nat_w else (None, None)
    if e_nat is not None:
        top2 = nn_retrieve(e_nat + scale_dem * ax_dem, [nid])
        cty_w = top2[0][0] if top2 else None
        if cty_w == cty_exp: correct_h2 += 1
        e_cty, ctid = get_emb(cty_w) if cty_w else (None, None)
        if e_cty is not None:
            top3 = nn_retrieve(e_cty + scale_lan * ax_lan, [ctid])
            lan_w = top3[0][0] if top3 else None
            if lan_w == lang_expected: correct_h3 += 1
print("  Hop 1 (person->nat):    %d/%d (%.0f%%)" % (correct_h1, n3, 100*correct_h1/n3))
print("  Hop 2 (nat->country):   %d/%d (%.0f%%)" % (correct_h2, n3, 100*correct_h2/n3))
print("  Hop 3 (country->lang):  %d/%d (%.0f%%)" % (correct_h3, n3, 100*correct_h3/n3))
print()

# Expected if independent: h1*h2*h3 (where h_i is per-hop accuracy on this specific chain)
# Note: hop 2 uses whatever hop 1 returned, not the gold intermediate
print("  Note: hop 2 uses retrieved nat (not gold), hop 3 uses retrieved country (not gold).")
print("  Error propagation: mistakes at hop 1 cascade through hops 2 and 3.")
