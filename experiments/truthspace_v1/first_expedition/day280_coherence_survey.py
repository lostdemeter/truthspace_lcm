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
def compute_axis_full(pairs):
    chords = []
    valid  = []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es)
        valid.append((s, t, sid, tid))
    if not chords: return None, 0.0, 0.0, []
    md = normed(np.mean(chords, axis=0))
    sims = [float(np.dot(normed(c), md)) for c in chords]
    mean_coh = float(np.mean(sims))
    std_coh  = float(np.std(sims))
    return md, mean_coh, std_coh, valid
def compute_voronoi_size(axis, valid_pairs, scale, n_neighbors=50):
    """Estimate target Voronoi cell size as fraction of vocab closer than runner-up."""
    sizes = []
    for s, t, sid, tid in valid_pairs[:20]:  # sample 20
        et = W_E[tid].copy()
        et_n = normed(et).astype(np.float32)
        sims = W_n @ et_n
        sorted_sims = np.sort(sims)[::-1]
        # Voronoi radius = gap between rank-0 (itself) and rank-1
        gap = float(sorted_sims[0] - sorted_sims[1])
        sizes.append(gap)
    return float(np.mean(sizes)) if sizes else 0.0
def scale_and_acc(axis, valid_pairs, n_scales=25):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(0.05, 3.0, n_scales):
        correct = 0
        for src, tgt, sid, tid in valid_pairs:
            es = W_E[sid].copy()
            pred_n = normed(es + s * axis).astype(np.float32)
            sims = W_n @ pred_n
            sims[sid] = -1.0
            if np.argmax(sims) == tid: correct += 1
        if correct > best_acc: best_acc = correct; best_s = s
    return best_s, best_acc, len(valid_pairs)

# =========================================================
# ALL AXES TESTED IN DAYS 262-279
# =========================================================
AXES = {}

# MORPHOLOGICAL AXES (Day 262-265)
AXES['plural']    = [('cat','cats'),('dog','dogs'),('book','books'),('car','cars'),('tree','trees'),('bird','birds'),('house','houses'),('chair','chairs'),('table','tables'),('door','doors'),('window','windows'),('flower','flowers'),('stone','stones'),('river','rivers'),('cloud','clouds'),('mountain','mountains')]
AXES['past_tense']= [('walk','walked'),('talk','talked'),('jump','jumped'),('play','played'),('work','worked'),('help','helped'),('clean','cleaned'),('open','opened'),('close','closed'),('start','started'),('finish','finished'),('follow','followed'),('turn','turned'),('reach','reached'),('watch','watched'),('learn','learned')]
AXES['comparative']=[('big','bigger'),('small','smaller'),('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),('long','longer'),('bright','brighter'),('dark','darker'),('old','older'),('young','younger'),('clean','cleaner'),('warm','warmer'),('cool','cooler'),('hard','harder'),('soft','softer')]
AXES['superlative']=[('big','biggest'),('small','smallest'),('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),('long','longest'),('bright','brightest'),('dark','darkest'),('old','oldest'),('young','youngest'),('clean','cleanest'),('warm','warmest'),('cool','coolest'),('hard','hardest'),('soft','softest')]
AXES['gender']    = [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),('son','daughter'),('brother','sister'),('husband','wife'),('uncle','aunt'),('grandfather','grandmother'),('actor','actress'),('prince','princess'),('lion','lioness'),('wizard','witch'),('monk','nun')]

# SEMANTIC AXES (Day 266-273)
AXES['capital']   = [('france','paris'),('germany','berlin'),('japan','tokyo'),('china','beijing'),('italy','rome'),('spain','madrid'),('russia','moscow'),('india','delhi'),('brazil','brasilia'),('canada','ottawa'),('egypt','cairo'),('greece','athens'),('turkey','ankara'),('poland','warsaw'),('sweden','stockholm'),('austria','vienna')]
AXES['language']  = [('france','french'),('germany','german'),('spain','spanish'),('russia','russian'),('japan','japanese'),('china','chinese'),('italy','italian'),('portugal','portuguese'),('poland','polish'),('sweden','swedish'),('norway','norwegian'),('denmark','danish'),('greece','greek'),('turkey','turkish'),('finland','finnish'),('hungary','hungarian')]
AXES['currency']  = [('japan','yen'),('china','yuan'),('russia','ruble'),('india','rupee'),('brazil','real'),('korea','won'),('sweden','krona'),('norway','krone'),('denmark','krone'),('mexico','peso'),('turkey','lira'),('poland','zloty')]
AXES['antonym']   = [('hot','cold'),('big','small'),('fast','slow'),('light','dark'),('hard','soft'),('old','new'),('high','low'),('long','short'),('strong','weak'),('rich','poor'),('happy','sad'),('good','bad'),('love','hate'),('open','close'),('start','end'),('enter','exit')]
AXES['hypernym']  = [('dog','animal'),('cat','animal'),('rose','flower'),('oak','tree'),('salmon','fish'),('eagle','bird'),('diamond','gem'),('gold','metal'),('iron','metal'),('copper','metal'),('ruby','gem'),('pine','tree'),('shark','fish'),('wolf','animal'),('daisy','flower')]
AXES['meronym']   = [('car','wheel'),('house','roof'),('tree','branch'),('face','eye'),('hand','finger'),('clock','hand'),('book','page'),('door','handle'),('bicycle','wheel'),('airplane','wing'),('computer','screen')]

# SECOND-ORDER AXES (Day 269-273)
AXES['person_nat']= [('Einstein','German'),('Newton','British'),('Darwin','British'),('Kepler','German'),('Euler','Swiss'),('Gauss','German'),('Turing','British'),('Tesla','Serbian'),('Napoleon','French'),('Churchill','British'),('Lincoln','American'),('Gandhi','Indian'),('Caesar','Roman'),('Aristotle','Greek'),('Plato','Greek'),('Shakespeare','British'),('Mozart','Austrian'),('Marx','German'),('Freud','Austrian'),('Kant','German')]
AXES['country_axis']=[('England','British'),('France','French'),('Germany','German'),('Spain','Spanish'),('Russia','Russian'),('Japan','Japanese'),('China','Chinese'),('Italy','Italian'),('Greece','Greek'),('Poland','Polish'),('Sweden','Swedish'),('Norway','Norwegian'),('Austria','Austrian'),('America','American'),('Switzerland','Swiss')]
AXES['dem_country']=[('German','germany'),('British','britain'),('French','france'),('Japanese','japan'),('Chinese','china'),('Italian','italy'),('Spanish','spain'),('Russian','russia'),('Greek','greece'),('Polish','poland'),('Swedish','sweden'),('Norwegian','norway'),('Austrian','austria'),('American','america'),('Swiss','switzerland')]
AXES['city_country']=[('paris','france'),('berlin','germany'),('tokyo','japan'),('beijing','china'),('rome','italy'),('madrid','spain'),('moscow','russia'),('athens','greece'),('warsaw','poland'),('stockholm','sweden'),('vienna','austria'),('oslo','norway'),('lisbon','portugal'),('ankara','turkey'),('helsinki','finland')]

# =========================================================
print("DAY 280: COHERENCE vs ACCURACY SURVEY — ALL 14 AXES")
print("="*70)
print()
print("%-16s  %5s %5s %5s  %7s  %5s  %3s  %4s" % (
    "Axis", "n", "coh", "std", "scale", "acc%", "n/a", "V_gap"))
print("-"*70)

results = []
for name, pairs in AXES.items():
    axis, coh, std, valid = compute_axis_full(pairs)
    if axis is None:
        print("  %-16s  SKIP (no valid pairs)" % name)
        continue
    scale, acc, n_valid = scale_and_acc(axis, valid)
    acc_pct = 100.0 * acc / n_valid if n_valid > 0 else 0.0
    vgap = compute_voronoi_size(axis, valid, scale)
    print("  %-16s  %3d  %.3f %.3f  %5.2f  %4.0f%%  %3d  %.4f" % (
        name, len(pairs), coh, std, scale, acc_pct, n_valid, vgap))
    results.append((name, len(pairs), coh, std, scale, acc_pct, n_valid, vgap))

print()
print("SORTED BY COHERENCE:")
print("-"*70)
for row in sorted(results, key=lambda x: -x[2]):
    name, npairs, coh, std, scale, acc_pct, n_valid, vgap = row
    bar = '#' * int(acc_pct / 5)
    print("  %-16s  coh=%.3f  acc=%5.1f%%  |%s" % (name, coh, acc_pct, bar))

print()
print("SORTED BY ACCURACY:")
print("-"*70)
for row in sorted(results, key=lambda x: -x[5]):
    name, npairs, coh, std, scale, acc_pct, n_valid, vgap = row
    bar = '#' * int(acc_pct / 5)
    print("  %-16s  coh=%.3f  acc=%5.1f%%  vgap=%.4f  |%s" % (name, coh, acc_pct, vgap, bar))

print()
print("CORRELATION ANALYSIS:")
print("-"*70)
cohs  = np.array([r[2] for r in results])
accs  = np.array([r[5] for r in results])
vgaps = np.array([r[7] for r in results])
stds  = np.array([r[3] for r in results])

def pearson(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    return float(np.dot(xm, ym) / (np.linalg.norm(xm) * np.linalg.norm(ym) + 1e-10))

print("  Pearson r(coherence, accuracy):     %.4f" % pearson(cohs, accs))
print("  Pearson r(voronoi_gap, accuracy):   %.4f" % pearson(vgaps, accs))
print("  Pearson r(coh_std, accuracy):       %.4f" % pearson(stds, accs))
print("  Pearson r(coh*vgap, accuracy):      %.4f" % pearson(cohs*vgaps, accs))
print()

print("THRESHOLD TEST: coh >= 0.65 -> acc >= 60%?")
print("-"*70)
for row in results:
    name, npairs, coh, std, scale, acc_pct, n_valid, vgap = row
    pred = "HIGH" if coh >= 0.65 else "LOW"
    actual = "HIGH" if acc_pct >= 60.0 else "LOW"
    match = "OK" if pred == actual else "WRONG"
    print("  %-16s  coh=%.3f [%s]  acc=%5.1f%% [%s]  %s" % (
        name, coh, pred, acc_pct, actual, match))

print()
print("VORONOI GAP ANALYSIS: does large gap -> high accuracy?")
print("-"*70)
for row in sorted(results, key=lambda x: -x[7]):
    name, npairs, coh, std, scale, acc_pct, n_valid, vgap = row
    print("  %-16s  vgap=%.4f  acc=%5.1f%%" % (name, vgap, acc_pct))
