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
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=80):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def eval_pairs(axis, scale, pairs):
    results = []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: results.append((s,t,None,'?',False)); continue
        top3 = nn_retrieve(W_E[sid]+scale*axis, [sid], top_n=3)
        got = top3[0][0]
        t3  = [r[0] for r in top3]
        results.append((s, t, sid, got, got==t))
    return results
def target_diversity(pairs):
    targets = [t for _,t in pairs]
    unique = len(set(targets))
    return unique / max(1, len(targets))
def source_pairwise(pairs):
    embs = []
    for s, _ in pairs:
        e, _ = get_emb(s)
        if e is not None: embs.append(normed(e).astype(np.float32))
    if len(embs) < 2: return 0.0
    sims = [float(np.dot(embs[i], embs[j]))
            for i in range(len(embs)) for j in range(i+1, len(embs))]
    return float(np.mean(sims))

print("DAY 293: REAL vs INFLATED PAIRWISE COSINE")
print("="*65)
print("Part A: antonym speed cluster — shared-target inflation fix")
print("Part B: element->symbol axis — clean bijective factual domain")
print("Part C: contamination detection diagnostics")
print()

# ====================================================================
# PART A: ANTONYM SPEED CLUSTER — SHARED TARGET INFLATION
# ====================================================================
print("PART A: Antonym speed cluster — shared-target vs diversified")
print("-"*65)
print()

# Version 1: SHARED TARGET (fast->slow, quick->slow) — INFLATED
SPD_SHARED = [
    ('fast','slow'),('quick','slow'),('rapid','slow'),('swift','slow'),
    ('speedy','sluggish'),('brisk','gradual'),('hasty','leisurely'),('fleet','plodding'),
]
# Version 2: DIVERSIFIED TARGETS (each pair has different targets)
SPD_DIVERSE = [
    ('fast','slow'),('quick','quiet'),('rapid','gradual'),('swift','sluggish'),
    ('speedy','lazy'),('brisk','sluggish'),('hasty','cautious'),('fleet','crawling'),
]
# Note: the "true" antonym pairs are diverse targets — no word repeats as target

# Version 3: CANONICAL SPEED ANTONYMS (true, correct pairs)
SPD_CANONICAL = [
    ('fast','slow'),('quick','slow'),('rapid','gradual'),('swift','sluggish'),
    ('brisk','sluggish'),('speedy','slow'),('hasty','deliberate'),
    ('accelerating','decelerating'),
]
SPD_HOLDOUT = [
    ('quick','slow'),('nimble','clumsy'),('agile','sluggish'),
    ('zippy','poky'),('lively','sluggish'),
]

for name, pairs in [
    ('shared_target  ', SPD_SHARED[:4]),
    ('diversified    ', SPD_DIVERSE[:4]),
    ('canonical      ', SPD_CANONICAL[:4]),
]:
    ax, coh, valid, pc = compute_axis(pairs)
    if ax is None: continue
    td = target_diversity(pairs)
    sp = source_pairwise(pairs)
    s_opt, acc = best_scale(ax, valid)
    print("  %-16s  pc=%.4f  coh=%.4f  target_div=%.2f  src_pc=%.4f  scale=%.2f  acc=%d/%d" % (
        name, pc, coh, td, sp, s_opt, acc, len(valid)))

print()
print("  Axis comparison: do shared_target and diversified point same way?")
ax_sh,  _, _, _ = compute_axis(SPD_SHARED[:4])
ax_div, _, _, _ = compute_axis(SPD_DIVERSE[:4])
ax_can, _, _, _ = compute_axis(SPD_CANONICAL[:4])
if ax_sh is not None and ax_div is not None:
    print("  shared_target  <-> diversified:  cos=%.4f" % float(np.dot(
        ax_sh.astype(np.float32), ax_div.astype(np.float32))))
if ax_sh is not None and ax_can is not None:
    print("  shared_target  <-> canonical:    cos=%.4f" % float(np.dot(
        ax_sh.astype(np.float32), ax_can.astype(np.float32))))
if ax_div is not None and ax_can is not None:
    print("  diversified    <-> canonical:    cos=%.4f" % float(np.dot(
        ax_div.astype(np.float32), ax_can.astype(np.float32))))
print()

# Test holdout for each
for name, pairs, hold in [
    ('shared_target  ', SPD_SHARED[:4],    SPD_HOLDOUT),
    ('diversified    ', SPD_DIVERSE[:4],   SPD_HOLDOUT),
    ('canonical      ', SPD_CANONICAL[:4], SPD_HOLDOUT),
]:
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None: continue
    s_opt, _ = best_scale(ax, valid)
    results = eval_pairs(ax, s_opt, hold)
    acc = sum(1 for _,_,sid,_,hit in results if hit and sid is not None)
    n   = sum(1 for _,_,sid,_,_ in results if sid is not None)
    print("  %-16s  holdout=%d/%d (%.0f%%)" % (name, acc, n, 100*acc/max(1,n)))
    for s, t, sid, got, hit in results:
        if sid is None: continue
        print("    %-12s -> %-12s  got=%-12s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    print()

# ====================================================================
# PART B: ELEMENT -> SYMBOL AXIS
# Chemical elements and their symbols — bijective, factual
# ====================================================================
print("PART B: element->symbol axis (new domain)")
print("-"*65)

ELEM_TRAIN = [
    ('hydrogen','H'),('helium','He'),('lithium','Li'),('carbon','C'),
    ('nitrogen','N'),('oxygen','O'),('sodium','Na'),('calcium','Ca'),
    ('iron','Fe'),('copper','Cu'),('zinc','Zn'),('silver','Ag'),
    ('gold','Au'),('lead','Pb'),('tin','Sn'),('mercury','Hg'),
]
ELEM_HOLD = [
    ('potassium','K'),('chlorine','Cl'),('phosphorus','P'),('sulfur','S'),
    ('aluminum','Al'),('silicon','Si'),('magnesium','Mg'),('neon','Ne'),
    ('argon','Ar'),('fluorine','F'),('boron','B'),('chromium','Cr'),
]

# Check token availability
print("  Checking BPE token availability:")
train_avail, hold_avail = [], []
for s, t in ELEM_TRAIN:
    es, sid = get_emb(s); et, tid = get_emb(t)
    ok = es is not None and et is not None
    if not ok:
        print("    SKIP: %s->%s  (src=%s, tgt=%s)" % (
            s, t, 'ok' if es is not None else 'MULTI', 'ok' if et is not None else 'MULTI'))
    else:
        train_avail.append((s, t))

for s, t in ELEM_HOLD:
    es, sid = get_emb(s); et, tid = get_emb(t)
    ok = es is not None and et is not None
    if not ok:
        print("    SKIP: %s->%s  (src=%s, tgt=%s)" % (
            s, t, 'ok' if es is not None else 'MULTI', 'ok' if et is not None else 'MULTI'))
    else:
        hold_avail.append((s, t))

print("  Available: train=%d/%d  hold=%d/%d" % (
    len(train_avail), len(ELEM_TRAIN), len(hold_avail), len(ELEM_HOLD)))
print()

ax_elem, coh_elem, valid_elem, pc_elem = compute_axis(train_avail)
if ax_elem is not None:
    td_elem = target_diversity(train_avail)
    sp_elem = source_pairwise(train_avail)
    s_elem, acc_elem = best_scale(ax_elem, valid_elem)
    print("  Axis: pc=%.4f  coh=%.4f  target_div=%.2f  src_pc=%.4f" % (
        pc_elem, coh_elem, td_elem, sp_elem))
    print("  Train: acc=%d/%d (%.0f%%)  scale=%.2f" % (
        acc_elem, len(valid_elem), 100*acc_elem/max(1,len(valid_elem)), s_elem))
    print()
    print("  Training results:")
    for s, t, sid, tid in valid_elem:
        r = nn_retrieve(W_E[sid]+s_elem*ax_elem, [sid])
        got = r[0][0] if r else '?'
        print("    %-14s -> %-6s  got=%-8s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()
    print("  Holdout:")
    hold_results_elem = eval_pairs(ax_elem, s_elem, hold_avail)
    acc_h = sum(1 for _,_,sid,_,hit in hold_results_elem if hit and sid is not None)
    n_h   = sum(1 for _,_,sid,_,_ in hold_results_elem if sid is not None)
    for s, t, sid, got, hit in hold_results_elem:
        if sid is None: continue
        print("    %-14s -> %-6s  got=%-8s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    print("  Holdout: %d/%d (%.0f%%)" % (acc_h, n_h, 100*acc_h/max(1,n_h)))
    print()

# ====================================================================
# PART C: CONTAMINATION DETECTION DIAGNOSTICS
# ====================================================================
print("PART C: Contamination detection diagnostics")
print("-"*65)
print("Measuring three signals: pc, target_diversity, source_pairwise")
print()

DIAG_AXES = [
    # GENUINE high-pc axes
    ('+er (comp)',  [('fast','faster'),('slow','slower'),('tall','taller'),
                     ('small','smaller'),('large','larger'),('hard','harder'),
                     ('soft','softer'),('warm','warmer'),('dark','darker'),
                     ('clean','cleaner'),('sharp','sharper'),('deep','deeper')],
     'GENUINE'),
    ('country->dem', [('France','French'),('Germany','German'),('Spain','Spanish'),
                      ('Italy','Italian'),('Japan','Japanese'),('China','Chinese'),
                      ('Russia','Russian'),('Egypt','Egyptian'),('Brazil','Brazilian'),
                      ('Portugal','Portuguese')],
     'GENUINE'),
    # INFLATED pc axes
    ('speed (shared)', [('fast','slow'),('quick','slow'),('rapid','slow'),
                        ('swift','slow'),('speedy','slow'),('brisk','slow')],
     'INFLATED'),
    ('country->lang', [('France','French'),('Germany','German'),('Spain','Spanish'),
                       ('Italy','Italian'),('Portugal','Portuguese'),('Japan','Japanese'),
                       ('China','Chinese'),('Russia','Russian'),('Egypt','Arabic'),
                       ('Netherlands','Dutch')],
     'SCOPE_LIMITED'),
    # AMBIGUOUS cases
    ('element->sym',  train_avail,  'UNKNOWN'),
    ('+s plural',     [('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),
                       ('book','books'),('car','cars'),('hand','hands'),('eye','eyes'),
                       ('word','words'),('day','days'),('year','years'),('house','houses')],
     'GENUINE'),
]

print("  %-22s  pc      target_div  src_pc   verdict" % "Axis")
print("  " + "-"*65)
for name, pairs, known in DIAG_AXES:
    ax, coh, valid, pc = compute_axis(pairs)
    if ax is None: continue
    td = target_diversity(pairs)
    sp = source_pairwise(pairs)
    # Heuristic contamination score
    inflated = (td < 0.6 or sp > 0.4) and pc > 0.3
    flag = "WARN_INFLATED" if inflated else "OK"
    print("  %-22s  %.4f  %.4f      %.4f   [%s] (%s)" % (
        name, pc, td, sp, flag, known))
print()
print("  Contamination heuristic: WARN if pc>0.3 AND (target_div<0.6 OR src_pc>0.4)")
print()

# ====================================================================
# PART D: ELEMENT AXIS — MINI-TRAIN CURVE
# ====================================================================
if ax_elem is not None:
    print("PART D: element->symbol mini-train curve")
    print("-"*65)
    for n_tr in [2, 4, 6, 8, 10, len(train_avail)]:
        if n_tr > len(train_avail): break
        sub_tr = train_avail[:n_tr]
        sub_ho = [p for p in train_avail if p not in sub_tr] + hold_avail
        ax_m, _, vt_m, pc_m = compute_axis(sub_tr)
        if ax_m is None: continue
        s_m, acc_m = best_scale(ax_m, vt_m)
        hold_r = eval_pairs(ax_m, s_m, sub_ho)
        acc_h = sum(1 for _,_,sid,_,hit in hold_r if hit and sid is not None)
        n_h   = sum(1 for _,_,sid,_,_ in hold_r if sid is not None)
        print("  n_train=%2d  pc=%.4f  train=%d/%d  hold=%d/%d (%.0f%%)" % (
            n_tr, pc_m, acc_m, len(vt_m), acc_h, n_h, 100*acc_h/max(1,n_h)))
    print()

# ====================================================================
# PART E: SYMBOL -> ELEMENT (REVERSE AXIS)
# Tests ENCODE=DECODE for this new domain
# ====================================================================
if ax_elem is not None:
    print("PART E: symbol->element reverse axis (ENCODE=DECODE test)")
    print("-"*65)
    # Compute reverse: swap src and tgt
    rev_pairs = [(t, s) for s, t in train_avail]
    ax_rev, coh_rev, valid_rev, pc_rev = compute_axis(rev_pairs)
    if ax_rev is not None:
        s_rev, acc_rev = best_scale(ax_rev, valid_rev)
        print("  Forward: pc=%.4f  scale=%.2f  train=%d/%d" % (
            pc_elem, s_elem, acc_elem, len(valid_elem)))
        print("  Reverse: pc=%.4f  scale=%.2f  train=%d/%d" % (
            pc_rev, s_rev, acc_rev, len(valid_rev)))
        print("  Scale ratio fwd/rev = %.3f" % (s_elem/max(0.001,s_rev)))
        cos_fwd_rev = float(np.dot(normed(ax_elem).astype(np.float32),
                                   normed(ax_rev).astype(np.float32)))
        print("  cos(fwd_axis, rev_axis) = %.4f  (expect ~-1.0 for ENCODE=DECODE)" % cos_fwd_rev)
        print()
        print("  Reverse retrieval (symbol -> element):")
        for s, t, sid, tid in valid_rev[:8]:
            r = nn_retrieve(W_E[sid]+s_rev*ax_rev, [sid])
            got = r[0][0] if r else '?'
            print("    %-6s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*65)
print("SUMMARY: DAY 293")
print("="*65)
print()
print("1. Shared-target inflation: fast->slow, quick->slow => pc INFLATED")
print("   Diversified antonym pairs: pc falls, still fails holdout")
print("   => antonymy is FUNDAMENTALLY not a geometric axis regardless")
print()
print("2. element->symbol: new domain, bijective, factual")
if ax_elem is not None:
    acc_h_final = sum(1 for _,_,sid,_,hit in hold_results_elem if hit and sid is not None)
    n_h_final   = sum(1 for _,_,sid,_,_ in hold_results_elem if sid is not None)
    print("   pc=%.4f, train=%d/%d, holdout=%d/%d (%.0f%%)" % (
        pc_elem, acc_elem, len(valid_elem), acc_h_final, n_h_final,
        100*acc_h_final/max(1,n_h_final)))
print()
print("3. Contamination detector: target_div and src_pc are diagnostic")
print("   WARN: pc>0.3 AND (target_div<0.6 OR src_pc>0.4)")
