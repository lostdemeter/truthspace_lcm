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
    if len(chords) < 2: return None, 0.0, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    coh = float(np.mean([np.dot(c, md.astype(np.float32)) for c in cn]))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, coh, valid, pc
def nn_retrieve(pred_emb, exclude_ids, top_n=3):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=6.0, n=60):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def eval_holdout(axis, scale, holdout_pairs):
    results = []
    for s, t in holdout_pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None:
            results.append((s, t, None, '?', False, 'SKIP'))
            continue
        top3 = nn_retrieve(W_E[sid] + scale * axis, [sid], top_n=3)
        got  = top3[0][0] if top3 else '?'
        hit  = (got == t)
        top3_words = [r[0] for r in top3]
        rank = top3_words.index(t)+1 if t in top3_words else 0
        results.append((s, t, sid, got, hit, rank))
    return results

print("DAY 292: SEMANTIC AXIS GENERALISATION HOLDOUT")
print("="*65)
print("Does pc > 0.35 => generalises for semantic axes same as morph?")
print("Training on 10 countries, testing on 5-10 unseen countries.")
print()

# ====================================================================
# TRAINING AND HOLDOUT POOLS
# ====================================================================

CAP_TRAIN = [
    ('France','Paris'),('Germany','Berlin'),('Spain','Madrid'),('Italy','Rome'),
    ('Japan','Tokyo'),('China','Beijing'),('Russia','Moscow'),('Egypt','Cairo'),
    ('India','Delhi'),('Turkey','Ankara'),
]
CAP_HOLD = [
    ('Greece','Athens'),('Poland','Warsaw'),('Sweden','Stockholm'),('Norway','Oslo'),
    ('Brazil','Brasilia'),('Argentina','Buenos'),('Mexico','Mexico'),
    ('Canada','Ottawa'),('Australia','Canberra'),('Netherlands','Amsterdam'),
    ('Switzerland','Bern'),('Portugal','Lisbon'),('Ireland','Dublin'),
    ('Austria','Vienna'),('Denmark','Copenhagen'),
]

DEM_TRAIN = [
    ('France','French'),('Germany','German'),('Spain','Spanish'),
    ('Italy','Italian'),('Japan','Japanese'),('China','Chinese'),
    ('Russia','Russian'),('Egypt','Egyptian'),('Brazil','Brazilian'),
    ('Portugal','Portuguese'),
]
DEM_HOLD = [
    ('Sweden','Swedish'),('Greece','Greek'),('Poland','Polish'),
    ('Turkey','Turkish'),('Korea','Korean'),('India','Indian'),
    ('America','American'),('Britain','British'),('Ireland','Irish'),
    ('Netherlands','Dutch'),('Denmark','Danish'),('Norway','Norwegian'),
    ('Austria','Austrian'),('Canada','Canadian'),('Australia','Australian'),
]

LANG_TRAIN = [
    ('France','French'),('Germany','German'),('Spain','Spanish'),
    ('Italy','Italian'),('Portugal','Portuguese'),('Japan','Japanese'),
    ('China','Chinese'),('Russia','Russian'),('Egypt','Arabic'),
    ('Netherlands','Dutch'),
]
LANG_HOLD = [
    ('Sweden','Swedish'),('Korea','Korean'),('Turkey','Turkish'),
    ('Greece','Greek'),('Poland','Polish'),('Britain','English'),
    ('Brazil','Portuguese'),('Mexico','Spanish'),('India','Hindi'),
    ('Vietnam','Vietnamese'),('Iran','Persian'),('Israel','Hebrew'),
]

ANIMAL_TRAIN = [
    ('cat','mammal'),('dog','mammal'),('horse','mammal'),('whale','mammal'),
    ('eagle','bird'),('robin','bird'),('sparrow','bird'),('salmon','fish'),
    ('shark','fish'),('frog','amphibian'),
]
ANIMAL_HOLD = [
    ('wolf','mammal'),('bear','mammal'),('dolphin','mammal'),('bat','mammal'),
    ('hawk','bird'),('dove','bird'),('pigeon','bird'),('tuna','fish'),
    ('cod','fish'),('cobra','reptile'),('lizard','reptile'),('turtle','reptile'),
    ('salamander','amphibian'),
]

# ====================================================================
# PART A: MINI-TRAIN CURVES (same approach as Day 289 for plural/comp)
# ====================================================================
print("PART A: Mini-train curves — how few pairs needed to generalise?")
print("-"*65)
print()

for axis_name, all_pairs, full_holdout in [
    ('country->cap',     CAP_TRAIN + CAP_HOLD[:5],   CAP_HOLD),
    ('country->demonym', DEM_TRAIN + DEM_HOLD[:5],   DEM_HOLD),
    ('country->lang',    LANG_TRAIN + LANG_HOLD[:5], LANG_HOLD),
]:
    print("  %s:" % axis_name)
    # Compute full axis for reference scale
    ax_full, _, vf, _ = compute_axis(all_pairs)
    s_full, _ = best_scale(ax_full, vf) if ax_full is not None else (1.0, 0)
    for n_train in [2, 5, 10, len(all_pairs)//2 if len(all_pairs) > 10 else len(all_pairs)]:
        if n_train > len(all_pairs): break
        train_sub = all_pairs[:n_train]
        hold_sub  = [p for p in all_pairs if p not in train_sub]
        ax, _, vt, pc = compute_axis(train_sub)
        if ax is None: continue
        hold_results = eval_holdout(ax, s_full, hold_sub)
        acc_h = sum(1 for _,_,_,_,hit,_ in hold_results if hit)
        n_h   = sum(1 for _,_,sid,_,_,_ in hold_results if sid is not None)
        print("    n_train=%2d  pc=%.3f  holdout=%d/%d (%.0f%%)" % (
            n_train, pc, acc_h, n_h, 100*acc_h/max(1,n_h)))
    print()

# ====================================================================
# PART B: FULL HOLDOUT TEST FOR EACH AXIS
# ====================================================================
print("PART B: Full holdout tests (train on pool, test on holdout)")
print("-"*65)

for axis_name, train_pairs, holdout_pairs in [
    ('country->cap',     CAP_TRAIN,  CAP_HOLD),
    ('country->demonym', DEM_TRAIN,  DEM_HOLD),
    ('country->lang',    LANG_TRAIN, LANG_HOLD),
    ('animal->class',    ANIMAL_TRAIN, ANIMAL_HOLD),
]:
    ax, coh, valid, pc = compute_axis(train_pairs)
    if ax is None: print("  %s SKIP" % axis_name); continue
    s_opt, acc_tr = best_scale(ax, valid)
    results = eval_holdout(ax, s_opt, holdout_pairs)
    acc_h = sum(1 for _,_,_,_,hit,_ in results if hit)
    n_h   = sum(1 for _,_,sid,_,_,_ in results if sid is not None)
    print("  %-22s  pc=%.3f  coh=%.3f  scale=%.2f  train=%d/%d  hold=%d/%d (%.0f%%)" % (
        axis_name, pc, coh, s_opt, acc_tr, len(valid), acc_h, n_h, 100*acc_h/max(1,n_h)))
    for s, t, sid, got, hit, rank in results:
        if sid is None: continue
        flag = 'HIT' if hit else ('TOP3' if isinstance(rank,int) and 0<rank<=3 else '---')
        print("    %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, flag))
    print()

# ====================================================================
# PART C: COMPARISON TABLE — semantic vs morphological holdout
# ====================================================================
print("PART C: Holdout comparison — semantic vs morphological axes")
print("-"*65)

# Morphological holdout results from Day 289
MORPH_RESULTS = [
    ('+est (sup)',     0.436, '?',   '?',    1,  1),
    ('+er (comp)',     0.393, '?',   '?',    6,  6),
    ('+s (plural)',    0.155, '?',   '?',   27, 29),
    ('gender',         0.213, '?',   '?',    2,  5),
    ('+ed (past_r)',   0.174, '?',   '?',    7, 12),
    ('past_irr',       0.230, '?',   '?',    5, 12),
]

print("  %-22s  pc_cos  hold_acc  hold_pct  generalises?" % "Axis")
print("  " + "-"*60)

# Print morphological
for name, pc, _, __, acc_h, n_h in MORPH_RESULTS:
    ho_pct = 100*acc_h/max(1,n_h)
    gen = "YES" if ho_pct >= 70 else "NO"
    print("  %-22s  %.3f   %2d/%-2d     %3.0f%%      %s  [MORPH]" % (
        name, pc, acc_h, n_h, ho_pct, gen))

# Print semantic (will compute above, placeholder for now)
print("  (Semantic holdout results printed in PART B above)")
print()

# ====================================================================
# PART D: ZERO-SHOT RARE COUNTRIES
# Test on countries outside common training distributions
# ====================================================================
print("PART D: Zero-shot — rare/less-common countries")
print("-"*65)

RARE_CAPS = [
    ('Finland','Helsinki'),('Belgium','Brussels'),('Czechia','Prague'),
    ('Hungary','Budapest'),('Romania','Bucharest'),('Croatia','Zagreb'),
    ('Slovakia','Bratislava'),('Slovenia','Ljubljana'),('Estonia','Tallinn'),
    ('Latvia','Riga'),('Lithuania','Vilnius'),('Serbia','Belgrade'),
    ('Albania','Tirana'),('Iceland','Reykjavik'),('Luxembourg','Luxembourg'),
]
RARE_DEMS = [
    ('Finland','Finnish'),('Belgium','Belgian'),('Czechia','Czech'),
    ('Hungary','Hungarian'),('Romania','Romanian'),('Croatia','Croatian'),
    ('Serbia','Serbian'),('Albania','Albanian'),('Iceland','Icelandic'),
    ('Wales','Welsh'),('Scotland','Scottish'),('Ireland','Irish'),
]

# Reuse axes trained on CAP_TRAIN / DEM_TRAIN
ax_cap, coh_cap, valid_cap, pc_cap = compute_axis(CAP_TRAIN)
ax_dem, coh_dem, valid_dem, pc_dem = compute_axis(DEM_TRAIN)
s_cap, acc_cap = best_scale(ax_cap, valid_cap) if ax_cap is not None else (0.53, 0)
s_dem, acc_dem = best_scale(ax_dem, valid_dem) if ax_dem is not None else (0.22, 0)

for axis_name, ax, scale, test_pairs in [
    ('country->cap  (rare)', ax_cap, s_cap, RARE_CAPS),
    ('country->dem  (rare)', ax_dem, s_dem, RARE_DEMS),
]:
    if ax is None: continue
    results = eval_holdout(ax, scale, test_pairs)
    acc = sum(1 for _,_,sid,_,hit,_ in results if hit and sid is not None)
    n   = sum(1 for _,_,sid,_,_,_ in results if sid is not None)
    print("  %-28s %d/%d (%.0f%%)" % (axis_name, acc, n, 100*acc/max(1,n)))
    for s, t, sid, got, hit, rank in results:
        if sid is None: continue
        print("    %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    print()

# ====================================================================
# PART E: ANTONYM CLUSTER ANALYSIS
# If antonyms have no global axis, do local/cluster axes work?
# ====================================================================
print("PART E: Antonym cluster decomposition")
print("-"*65)
print("Testing: can we build DIMENSION-SPECIFIC antonym axes?")
print("(temperature, size, speed, morality, time) separately")
print()

ANT_CLUSTERS = {
    'temperature': [('hot','cold'),('warm','cool'),('boiling','freezing'),
                    ('scorching','icy')],
    'size':        [('big','small'),('large','tiny'),('huge','minuscule'),
                    ('giant','dwarf'),('tall','short'),('wide','narrow')],
    'speed':       [('fast','slow'),('quick','slow'),('rapid','gradual'),
                    ('swift','sluggish')],
    'morality':    [('good','bad'),('right','wrong'),('honest','dishonest'),
                    ('kind','cruel'),('brave','coward')],
    'quantity':    [('many','few'),('much','little'),('full','empty'),
                    ('rich','poor'),('heavy','light')],
    'time':        [('early','late'),('old','young'),('new','old'),
                    ('start','end'),('first','last')],
}

cluster_axes = {}
for dim, pairs in ANT_CLUSTERS.items():
    ax_c, coh_c, valid_c, pc_c = compute_axis(pairs)
    if ax_c is None or len(valid_c) < 2: continue
    s_c, acc_c = best_scale(ax_c, valid_c, hi=8.0)
    cluster_axes[dim] = (ax_c, coh_c, pc_c, s_c, acc_c, len(valid_c))
    print("  %-14s  pc=%.3f  coh=%.3f  acc=%d/%d  scale=%.2f" % (
        dim, pc_c, coh_c, acc_c, len(valid_c), s_c))

print()

# Cross-cluster cosine
print("  Inter-cluster cosine similarities:")
dims = list(cluster_axes.keys())
for i in range(len(dims)):
    for j in range(i+1, len(dims)):
        sim = float(np.dot(cluster_axes[dims[i]][0].astype(np.float32),
                           cluster_axes[dims[j]][0].astype(np.float32)))
        print("    %-12s <-> %-12s  %.4f" % (dims[i], dims[j], sim))
print()

# Test cross-cluster generalisation within each cluster
print("  Cluster-specific antonym generalisation:")
for dim, pairs in ANT_CLUSTERS.items():
    if dim not in cluster_axes or len(pairs) < 3: continue
    ax_c, _, pc_c, s_c, _, _ = cluster_axes[dim]
    train_p = pairs[:2]; hold_p = pairs[2:]
    ax_mini, _, _, pc_mini = compute_axis(train_p)
    if ax_mini is None or not hold_p: continue
    hold_results = eval_holdout(ax_mini, s_c, hold_p)
    acc_h = sum(1 for _,_,sid,_,hit,_ in hold_results if hit and sid is not None)
    n_h   = sum(1 for _,_,sid,_,_,_ in hold_results if sid is not None)
    print("    %-12s  train=2  pc_mini=%.3f  hold=%d/%d" % (
        dim, pc_mini, acc_h, n_h))
print()

# ====================================================================
# PART F: FINAL SUMMARY — UNIFIED LINEARITY PRINCIPLE COMPLETE
# ====================================================================
print("="*65)
print("FINAL SUMMARY: UNIFIED LINEARITY PRINCIPLE")
print("="*65)
print()
print("Morphological holdout (from Day 289):")
print("  +est/+er: >70% holdout (HIGH pc)")
print("  plural:   93% holdout with 20 pairs (LOW-MED pc)")
print("  gender:   40% holdout (MED pc, suppletive)")
print("  past_reg: 58% holdout (MED pc, verb diversity)")
print("  past_irr: 42% holdout (MED pc, closed class helps train)")
print()
print("Semantic holdout (Day 292):")

# Print final tallies for semantic axes
for axis_name, train_pairs, holdout_pairs in [
    ('country->cap',     CAP_TRAIN,  CAP_HOLD),
    ('country->demonym', DEM_TRAIN,  DEM_HOLD),
    ('country->lang',    LANG_TRAIN, LANG_HOLD),
    ('animal->class',    ANIMAL_TRAIN, ANIMAL_HOLD),
]:
    ax, coh, valid, pc = compute_axis(train_pairs)
    if ax is None: continue
    s_opt, acc_tr = best_scale(ax, valid)
    results = eval_holdout(ax, s_opt, holdout_pairs)
    acc_h = sum(1 for _,_,sid,_,hit,_ in results if hit and sid is not None)
    n_h   = sum(1 for _,_,sid,_,_,_ in results if sid is not None)
    ho_pct = 100*acc_h/max(1,n_h)
    gen = "YES" if ho_pct >= 70 else "NO"
    print("  %-22s  pc=%.3f  hold=%d/%d (%.0f%%)  [%s]" % (
        axis_name, pc, acc_h, n_h, ho_pct, gen))

print()
print("Prediction: pc > 0.35 => generalises for semantic axes too.")
print("Result:")
print("  country->cap (0.317):  predicted YES, result = ?")
print("  country->dem (0.563):  predicted YES, result = ?")
print("  (see above results)")
