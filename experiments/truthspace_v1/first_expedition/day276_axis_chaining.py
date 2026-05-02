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
    coh = float(np.mean([np.dot(normed(c), md) for c in chords]))
    return md, coh
def nn_top5(pred_emb, exclude_ids):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids:
        sims[eid] = -1.0
    top = np.argsort(sims)[-5:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i])) for i in top]
def scale_for_axis(axis, pairs, n_scales=20):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.1, 3.0, n_scales):
        correct = 0
        for src, tgt in pairs:
            es, sid = get_emb(src); et, tid = get_emb(tgt)
            if es is None or et is None: continue
            pred = es + s * axis
            top = nn_top5(pred, [sid])
            if top and top[0][0] == tgt: correct += 1
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc

# --- Build relation axes with calibrated scales ---
CAPITAL_PAIRS  = [('france','paris'),('germany','berlin'),('japan','tokyo'),('china','beijing'),('italy','rome'),('spain','madrid'),('russia','moscow'),('india','delhi'),('brazil','brasilia'),('canada','ottawa'),('egypt','cairo'),('greece','athens'),('turkey','ankara'),('poland','warsaw'),('sweden','stockholm'),('austria','vienna')]
LANGUAGE_PAIRS = [('france','french'),('germany','german'),('spain','spanish'),('russia','russian'),('japan','japanese'),('china','chinese'),('italy','italian'),('portugal','portuguese'),('poland','polish'),('sweden','swedish'),('norway','norwegian'),('denmark','danish'),('greece','greek'),('turkey','turkish'),('finland','finnish'),('hungary','hungarian')]
SCIENTIST_NAT  = [('Einstein','German'),('Newton','British'),('Darwin','British'),('Kepler','German'),('Euler','Swiss'),('Gauss','German'),('Turing','British'),('Tesla','Serbian'),('Napoleon','French'),('Churchill','British'),('Lincoln','American'),('Gandhi','Indian'),('Caesar','Roman'),('Aristotle','Greek'),('Plato','Greek'),('Shakespeare','British'),('Mozart','Austrian'),('Marx','German'),('Freud','Austrian'),('Kant','German')]

ax_cap, coh_cap = compute_axis(CAPITAL_PAIRS)
ax_lan, coh_lan = compute_axis(LANGUAGE_PAIRS)
ax_nat, coh_nat = compute_axis(SCIENTIST_NAT)

print("DAY 276: COMPOSITIONAL AXIS CHAINING")
print("="*60)
print()
print("Axis coherences:")
print("  capital:     %.4f" % coh_cap)
print("  language:    %.4f" % coh_lan)
print("  person->nat: %.4f" % coh_nat)
print()

# Find optimal scales
scale_cap, acc_cap = scale_for_axis(ax_cap, CAPITAL_PAIRS)
scale_lan, acc_lan = scale_for_axis(ax_lan, LANGUAGE_PAIRS)
scale_nat, acc_nat = scale_for_axis(ax_nat, SCIENTIST_NAT)
n_cap = sum(1 for s,t in CAPITAL_PAIRS if get_emb(s)[0] is not None)
n_lan = sum(1 for s,t in LANGUAGE_PAIRS if get_emb(s)[0] is not None)
n_nat = sum(1 for s,t in SCIENTIST_NAT if get_emb(s)[0] is not None)
print("Optimal scales and single-hop accuracy:")
print("  capital:     scale=%.2f  acc=%d/%d (%.0f%%)" % (scale_cap, acc_cap, n_cap, 100*acc_cap/n_cap))
print("  language:    scale=%.2f  acc=%d/%d (%.0f%%)" % (scale_lan, acc_lan, n_lan, 100*acc_lan/n_lan))
print("  person->nat: scale=%.2f  acc=%d/%d (%.0f%%)" % (scale_nat, acc_nat, n_nat, 100*acc_nat/n_nat))
print()

# --- TWO-HOP CHAINS ---
print("TWO-HOP CHAINING TESTS")
print("-"*60)

# Chain A: person -> nationality -> capital
# e.g. Einstein -> German -> Berlin?
# Step 1: person + nat_axis -> nationality (intermediate)
# Step 2: nationality_emb + capital_axis (country->capital) ... but capital axis goes country->capital
# nationality word is a demonym (German), not country name (Germany)
# We need nat->country axis or just test person -> [nat_axis] -> demonym
# Then demonym -> country (country_adj -> country) or use directly?

# More natural: country -> language -> speakers  (2-hop)
# country + cap_axis -> capital  (1-hop works)
# country + lan_axis -> language (1-hop works)
# Then: language + ??? 

# Best natural 2-hop: person -> nat -> country  (using REVERSE nat axis then something)
# OR: country -> capital (then capital -> ???)

# Let's test: person -> nat -> language
# Step 1: emb(Einstein) + scale_nat * ax_nat -> German (demonym)
# Step 2: emb(German) + scale_lan * ax_lan -> ??? (but ax_lan goes FROM country TO language)
# Issue: step 2 needs country embedding, not demonym.

# Better chain: country -> language, then language -> script?
# OR: just direct two-hop: country -> capital, THEN use country's capital in another hop

# Chain type 1: country -[capital]-> capital_city -[nat_language_of_city]-> ?
# Chain type 2: person -[nat]-> nationality_adj -[adj_to_country]-> country name

# Let's test: country + ax_cap + ax_lan (double-hop: France->Paris? No, that's same country)
# Actually, let's test the most natural:
# country -[cap_axis]-> capital, then check: capital -[language_axis]-> ???

# EXPERIMENT 1: Does emb(paris) + scale_lan*ax_lan retrieve 'french'?
# (Paris is a city, ax_lan was built from country->language. Does it generalise?)
print("EXPERIMENT 1: Does language axis generalise from country to city?")
print("  (Do city embeddings + language_axis -> city's language?)")
city_to_language = [('paris','french'),('berlin','german'),('tokyo','japanese'),
                    ('beijing','chinese'),('rome','italian'),('madrid','spanish'),
                    ('moscow','russian'),('athens','greek'),('warsaw','polish'),
                    ('stockholm','swedish'),('vienna','german'),('oslo','norwegian')]
correct = 0
for city, lang in city_to_language:
    ec, cid = get_emb(city)
    if ec is None: continue
    pred = ec + scale_lan * ax_lan
    top5 = nn_top5(pred, [cid])
    hit = (top5[0][0] == lang) if top5 else False
    if hit: correct += 1
    print("  %-12s + lan_axis -> %-12s  got: %-12s %s" % (
        city, lang, top5[0][0] if top5 else '?', 'HIT' if hit else ''))
print("  Accuracy: %d/%d (%.0f%%)" % (correct, len(city_to_language), 100*correct/len(city_to_language)))
print()

# EXPERIMENT 2: Two-hop chain: country -[cap]-> city then city -[lan]-> language?
print("EXPERIMENT 2: Two-hop chain: country -[cap_axis+lan_axis]-> language?")
print("  (Combined axis: does sum of two axes retrieve language directly from country?)")
country_to_lang_via_cap = [('france','french'),('germany','german'),('japan','japanese'),
                            ('china','chinese'),('italy','italian'),('spain','spanish'),
                            ('russia','russian'),('greece','greek'),('poland','polish'),
                            ('sweden','swedish'),('norway','norwegian'),('austria','german')]
correct_direct = 0
correct_chain  = 0
for country, lang in country_to_lang_via_cap:
    ec, cid = get_emb(country)
    if ec is None: continue
    # Direct language axis (single hop)
    pred_direct = ec + scale_lan * ax_lan
    top_direct  = nn_top5(pred_direct, [cid])
    hit_direct  = (top_direct[0][0] == lang) if top_direct else False
    if hit_direct: correct_direct += 1
    # Two-hop: capital axis then language axis
    pred_chain = ec + scale_cap * ax_cap + scale_lan * ax_lan
    top_chain  = nn_top5(pred_chain, [cid])
    hit_chain  = (top_chain[0][0] == lang) if top_chain else False
    if hit_chain: correct_chain += 1
    print("  %-10s  direct->%-10s  chain->%-10s" % (
        country, top_direct[0][0] if top_direct else '?',
        top_chain[0][0] if top_chain else '?'))
print("  Direct acc:  %d/%d (%.0f%%)" % (correct_direct, len(country_to_lang_via_cap), 100*correct_direct/len(country_to_lang_via_cap)))
print("  2-hop chain: %d/%d (%.0f%%)" % (correct_chain, len(country_to_lang_via_cap), 100*correct_chain/len(country_to_lang_via_cap)))
print()

# EXPERIMENT 3: person -> nationality -> language (3-hop geometric chain)
print("EXPERIMENT 3: Person -[nat]-> demonym -[???]-> language")
print("  Testing: emb(person) + nat_axis -> nationality, then project that onto language axis")
# Build reverse axis: demonym -> country (e.g. German -> germany)
DEMONYM_COUNTRY = [('German','germany'),('British','britain'),('French','france'),
                   ('Japanese','japan'),('Chinese','china'),('Italian','italy'),
                   ('Spanish','spain'),('Russian','russia'),('Greek','greece'),
                   ('Polish','poland'),('Swedish','sweden'),('Norwegian','norway'),
                   ('Austrian','austria'),('American','america'),('Swiss','switzerland')]
ax_dem_cty, coh_dem = compute_axis(DEMONYM_COUNTRY)
scale_dem, acc_dem = scale_for_axis(ax_dem_cty, DEMONYM_COUNTRY)
n_dem = sum(1 for s,t in DEMONYM_COUNTRY if get_emb(s)[0] is not None)
print("  demonym->country axis coh=%.4f  scale=%.2f  acc=%d/%d (%.0f%%)" % (
    coh_dem, scale_dem, acc_dem, n_dem, 100*acc_dem/n_dem))
print()

# Full 3-hop person -> demonym -> country -> language
PERSON_TO_LANG = [('Einstein','german'),('Newton','english'),('Darwin','english'),
                  ('Kepler','german'),('Euler','german'),('Gauss','german'),
                  ('Turing','english'),('Napoleon','french'),('Caesar','latin'),
                  ('Aristotle','greek'),('Plato','greek'),('Mozart','german')]
print("  3-hop: person -[nat]-> demonym -[dem_cty]-> country -[lan]-> language")
correct_3 = 0
for person, lang in PERSON_TO_LANG:
    ep, pid = get_emb(person)
    if ep is None: continue
    # Combined 3 axes at once
    pred = ep + scale_nat * ax_nat + scale_dem * ax_dem_cty + scale_lan * ax_lan
    top5 = nn_top5(pred, [pid])
    hit = (top5[0][0] == lang) if top5 else False
    if hit: correct_3 += 1
    print("  %-12s -> expected %-10s  got: %s %s" % (
        person, lang, top5[0][0] if top5 else '?', 'HIT' if hit else ''))
print("  3-hop accuracy: %d/%d (%.0f%%)" % (correct_3, len(PERSON_TO_LANG), 100*correct_3/len(PERSON_TO_LANG)))
print()

# EXPERIMENT 4: Analogy: Einstein:German::Newton:?  (axis arithmetic verification)
print("EXPERIMENT 4: Analogy arithmetic (classic word2vec style)")
print("  A:B::C:?  =>  C + (B_emb - A_emb) = ?")
analogies = [
    ('france','paris','germany','berlin'),
    ('france','paris','japan','tokyo'),
    ('france','paris','spain','madrid'),
    ('france','french','germany','german'),
    ('france','french','japan','japanese'),
    ('france','french','spain','spanish'),
    ('Einstein','German','Newton','British'),
    ('Einstein','German','Darwin','British'),
    ('Einstein','German','Kepler','German'),
    ('Einstein','physicist','Darwin','biologist'),
    ('Einstein','physicist','Newton','physicist'),
]
correct_ana = 0
for a, b, c, d in analogies:
    ea, aid = get_emb(a); eb, bid = get_emb(b)
    ec2, cid2 = get_emb(c); ed, did = get_emb(d)
    if any(x is None for x in [ea, eb, ec2, ed]): continue
    pred = ec2 + (eb - ea)
    top5 = nn_top5(pred, [aid, bid, cid2])
    hit = (top5[0][0] == d) if top5 else False
    if hit: correct_ana += 1
    print("  %-10s:%-12s :: %-10s:%-12s -> got %-12s %s" % (
        a, b, c, d, top5[0][0] if top5 else '?', 'HIT' if hit else ''))
print("  Analogy accuracy: %d/%d (%.0f%%)" % (correct_ana, len(analogies), 100*correct_ana/len(analogies)))
