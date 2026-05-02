import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

# ── precompute clean-token mask once ──────────────────────────────────
print("Building clean token mask...", flush=True)
CLEAN_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and len(w) > 1 and not w[0].isupper() and not w.startswith('-') and not w.startswith('_'):
        CLEAN_MASK[i] = True
print("  %d clean tokens" % CLEAN_MASK.sum())

# ── memoised source-ID lookup ──────────────────────────────────────────
_src_cache = {}
def source_ids(word):
    if word in _src_cache: return _src_cache[word]
    ids = set()
    for p in [' '+word, word,
              ' '+word[0].upper()+word[1:], word[0].upper()+word[1:],
              word.upper(), ' '+word.upper(),
              '-'+word, '_'+word, ' -'+word, ' ']:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    _src_cache[word] = ids
    return ids

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def nn_fast(pred_emb, excl_ids, top_n=3):
    """Fast retrieval: pre-computed mask, pre-computed exclusion set."""
    pred_n = normed(pred_emb).astype(np.float32)
    sims   = W_n @ pred_n
    sims[~CLEAN_MASK] = -1.0
    for eid in excl_ids: sims[eid] = -1.0
    top = np.argpartition(sims, -top_n)[-top_n:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

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

def best_scale(axis, valid, lo=0.02, hi=6.0, n=30):
    """Find best scale using 30-step sweep."""
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _, t, sid, _ in valid
                if nn_fast(W_E[sid] + s*axis, source_ids(tok.decode([sid]).strip()), 1)[0][0] == t)
        if c > best_acc: best_acc = c; best_s = s
    return best_s, best_acc

def axis_loo_fast(valid):
    """
    Fast LOO: for each fold, train on N-1 pairs using the GLOBAL best scale
    (already found), test the held-out pair.  One global scale search.
    """
    if len(valid) < 3: return 0.0, 0
    # 1. Find global scale on full axis
    chords_full = [W_E[tid] - W_E[sid] for _,_,sid,tid in valid]
    ax_full = normed(np.mean(chords_full, axis=0))
    global_s, _ = best_scale(ax_full, valid)
    # 2. LOO at that scale
    hits = 0
    for i in range(len(valid)):
        test_s, test_t, test_sid, _ = valid[i]
        train_v = [valid[j] for j in range(len(valid)) if j != i]
        chords_loo = [W_E[tid] - W_E[sid] for _,_,sid,tid in train_v]
        ax_loo = normed(np.mean(chords_loo, axis=0))
        pred = W_E[test_sid] + global_s * ax_loo
        r = nn_fast(pred, source_ids(test_s), 1)
        if r[0][0] == test_t: hits += 1
    return hits / len(valid), len(valid)

print("DAY 315: +tion LOO, AXIS CLASSIFICATION PROTOCOL, NAMED ENTITIES, INVARIANT PLURALS")
print("="*72)
print()

# ====================================================================
# PART A: +tion WITHIN-DOMAIN LOO
# ====================================================================
print("PART A: +tion within-domain LOO — confirm phonological scatter")
print("-"*72)

TION_CT = [('act','action'),('direct','direction'),('collect','collection'),
           ('connect','connection'),('protect','protection'),('select','selection'),
           ('inject','injection'),('reject','rejection'),('detect','detection'),
           ('infect','infection'),('inspect','inspection'),('correct','correction')]

TION_OBS = [('observe','observation'),('describe','description'),('produce','production'),
            ('resolve','resolution'),('evolve','evolution'),('revolve','revolution'),
            ('solve','solution'),('dissolve','dissolution')]

TION_ATE = [('communicate','communication'),('investigate','investigation'),
            ('appreciate','appreciation'),('evaluate','evaluation'),
            ('participate','participation'),('generate','generation'),
            ('create','creation'),('educate','education'),
            ('indicate','indication'),('locate','location')]

for dom_name, pairs in [('-ct', TION_CT), ('-serve/-scribe', TION_OBS), ('-ate', TION_ATE)]:
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None: print("  %s: no valid pairs" % dom_name); continue
    loo_frac, _ = axis_loo_fast(valid)
    best_s, in_s = best_scale(ax, valid)
    print("  %-18s  n=%d  pc=%.4f  LOO=%.0f%%  in-sample=%.0f%%  scale=%.3f" %
          (dom_name, len(valid), pc, 100*loo_frac, 100*in_s/len(valid), best_s))
print()

UN_ADJ = [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('safe','unsafe'),
          ('wise','unwise'),('true','untrue'),('sure','unsure'),('clear','unclear'),('fit','unfit')]
ax_ua, _, valid_ua, pc_ua = compute_axis(UN_ADJ)
if ax_ua is not None:
    loo_ua, _ = axis_loo_fast(valid_ua)
    best_s_ua, in_s_ua = best_scale(ax_ua, valid_ua)
    print("  %-18s  n=%d  pc=%.4f  LOO=%.0f%%  in-sample=%.0f%%  scale=%.3f" %
          ('un-ADJ (compare)', len(valid_ua), pc_ua, 100*loo_ua, 100*in_s_ua/len(valid_ua), best_s_ua))
print()

# ====================================================================
# PART B: AXIS CLASSIFICATION PROTOCOL
# ====================================================================
print("PART B: Axis classification protocol — 5 new axes")
print("-"*72)

def classify_axis(name, train_pairs, test_pairs):
    ax, _, valid, pc = compute_axis(train_pairs)
    if ax is None or len(valid) < 2:
        return {'name': name, 'type': 'insufficient', 'pc': 0.0,
                'in_sample': 0.0, 'loo': 0.0, 'irred': 0.0, 'n_ho': 0}
    best_s, in_s = best_scale(ax, valid)
    in_frac = in_s / len(valid)
    loo_frac, _ = axis_loo_fast(valid)
    # Holdout: test at best_s (fast, no re-sweep)
    irred = 0; n_ho = 0
    for src, tgt in test_pairs:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        n_ho += 1
        r = nn_fast(W_E[sid] + best_s * ax, source_ids(src), 1)
        if tid is None or r[0][0] != tgt: irred += 1
    irred_frac = irred / n_ho if n_ho > 0 else 0.0
    if in_frac < 0.15:               t = 'named_entity'
    elif loo_frac >= 0.50:            t = 'morph_uniform'
    elif in_frac >= 0.85 and loo_frac < 0.30: t = 'phonol_scatter'
    elif in_frac >= 0.85:             t = 'morph_moderate'
    else:                             t = 'semantic_diverse'
    return {'name': name, 'type': t, 'pc': pc, 'n_train': len(valid),
            'in_sample': in_frac, 'loo': loo_frac, 'irred': irred_frac, 'n_ho': n_ho}

NEW_AXES = {
    '+er_adj':    ([('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                    ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                    ('hard','harder'),('warm','warmer'),('cool','cooler'),('sweet','sweeter')],
                   [('kind','kinder'),('old','older'),('new','newer'),('long','longer')]),
    '+ing':       ([('run','running'),('jump','jumping'),('walk','walking'),('talk','talking'),
                    ('play','playing'),('start','starting'),('help','helping'),('work','working'),
                    ('turn','turning'),('push','pushing'),('call','calling'),('look','looking')],
                   [('move','moving'),('pull','pulling'),('open','opening'),('think','thinking')]),
    '+less':      ([('hope','hopeless'),('help','helpless'),('use','useless'),('care','careless'),
                    ('harm','harmless'),('thought','thoughtless'),('power','powerless')],
                   [('worth','worthless'),('home','homeless'),('job','jobless'),
                    ('friend','friendless'),('speech','speechless')]),
    'country_lang':([('france','French'),('germany','German'),('spain','Spanish'),
                     ('russia','Russian'),('china','Chinese'),('japan','Japanese'),
                     ('italy','Italian'),('greece','Greek'),('poland','Polish')],
                    [('brazil','Portuguese'),('mexico','Spanish'),('egypt','Arabic')]),
    '+able':      ([('break','breakable'),('wash','washable'),('read','readable'),
                    ('use','usable'),('move','movable'),('adjust','adjustable'),
                    ('adapt','adaptable'),('accept','acceptable'),('avoid','avoidable'),
                    ('change','changeable')],
                   [('manage','manageable'),('agree','agreeable'),('debate','debatable'),
                    ('comfort','comfortable'),('reason','reasonable')]),
}

print("  %-16s  pc     in%%   LOO%%  irred%%  type" % "axis")
print("  " + "-"*62)
for nm, (train, test) in NEW_AXES.items():
    r = classify_axis(nm, train, test)
    print("  %-16s  %.3f  %.0f%%   %.0f%%   %.0f%%     %s" %
          (nm, r['pc'], 100*r['in_sample'], 100*r['loo'], 100*r['irred'], r['type']))
print()

# ====================================================================
# PART C: NAMED ENTITY RELATIONS — IN-SAMPLE CHECK ONLY
# ====================================================================
print("PART C: Named entity relations — in-sample check")
print("-"*72)

NE_TESTS = {
    'country->capital':  [('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
                           ('spain','Madrid'),('italy','Rome'),('china','Beijing'),
                           ('russia','Moscow'),('egypt','Cairo'),('canada','Ottawa')],
    'country->language': [('france','French'),('germany','German'),('japan','Japanese'),
                           ('spain','Spanish'),('italy','Italian'),('russia','Russian')],
    'element->symbol':   [('hydrogen','H'),('helium','He'),('carbon','C'),
                           ('nitrogen','N'),('oxygen','O'),('sodium','Na')],
}

print("  %-22s  pc     n  in_sample%%  notes" % "relation")
print("  " + "-"*60)
for name, pairs in NE_TESTS.items():
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None or len(valid) < 2:
        print("  %-22s  n/a   <2 valid" % name); continue
    best_s, in_s = best_scale(ax, valid)
    found = []
    for s_w, t_w, sid, _ in valid:
        pred = W_E[sid] + best_s * ax
        r = nn_fast(pred, source_ids(s_w), 1)
        if r[0][0] == t_w: found.append(s_w+'->'+t_w)
    print("  %-22s  %.3f  %d  %.0f%%         %s" %
          (name, pc, len(valid), 100*in_s/len(valid),
           ', '.join(found) if found else 'none'))
print()

# ====================================================================
# PART D: INVARIANT PLURAL GEOMETRY
# ====================================================================
print("PART D: Invariant plural geometry — deer, fish, sheep etc.")
print("-"*72)

BODYPART_TRAIN = [('head','heads'),('foot','feet'),('ear','ears'),('knee','knees'),
                  ('toe','toes'),('lip','lips'),('hip','hips'),('rib','ribs'),
                  ('thumb','thumbs'),('wrist','wrists'),('elbow','elbows'),('heel','heels')]
ax_bp, _, _, _ = compute_axis(BODYPART_TRAIN)
ax_s, _, _, _  = compute_axis([('cat','cats'),('dog','dogs'),('house','houses'),
                                ('car','cars'),('tree','trees'),('book','books'),
                                ('bird','birds'),('ship','ships')])

INV_PLURALS = ['deer','fish','sheep','moose','bison','salmon','trout','cod','elk','swine']
print("  word     +s_axis_top1      bp_axis_top1")
for word in INV_PLURALS:
    es, sid = get_emb(word)
    if es is None: print("  %-8s  [multi-token]" % word); continue
    excl = source_ids(word)
    r_s  = nn_fast(W_E[sid] + 0.181 * ax_s,  excl, 1) if ax_s  is not None else [('n/a',0,0)]
    r_bp = nn_fast(W_E[sid] + 0.342 * ax_bp, excl, 1) if ax_bp is not None else [('n/a',0,0)]
    print("  %-8s  %-18s  %s" % (word, r_s[0][0], r_bp[0][0]))
print()

# Top-10 clean neighbors of 'deer' with no axis
es_deer, sid_deer = get_emb('deer')
if es_deer is not None:
    sims = W_n @ normed(es_deer).astype(np.float32)
    sims[~CLEAN_MASK] = -1.0
    for eid in source_ids('deer'): sims[eid] = -1.0
    top = np.argpartition(sims, -12)[-12:]
    top = top[np.argsort(sims[top])[::-1]]
    print("  Top-10 clean neighbors of 'deer' (no axis):")
    for i in top[:10]:
        print("    %-14s  cos=%.4f" % (tok.decode([i]).strip(), float(sims[i])))
print()

# ====================================================================
# PART E: SUMMARY TABLE — ALL KNOWN AXES CLASSIFIED
# ====================================================================
print("PART E: Final axis classification table")
print("-"*72)

ALL_KNOWN = [
    # name,         pc,    in%,  LOO%,  irred%
    ('+er',         0.394, 1.00, 0.80,  0.12),
    ('+est',        0.401, 1.00, 0.75,  0.25),
    ('er->est',     0.436, 1.00, 0.90,  0.00),
    ('+s',          0.297, 1.00, 0.87,  0.12),
    ('+ed',         0.227, 1.00, 0.87,  0.12),
    ('past_irr',    0.284, 1.00, 0.75,  0.25),
    ('gender',      0.241, 1.00, 0.50,  0.33),
    ('+tion',       0.116, 1.00, 0.50,  0.00),
    ('+ly',         0.142, 1.00, 0.50,  0.17),
    ('+er_noun',    0.130, 1.00, 0.50,  0.33),
    ('+ment',       0.124, 1.00, 0.33,  0.33),
    ('+ness',       0.169, 1.00, 0.25,  0.75),
    ('+ful',        0.112, 1.00, 0.25,  0.75),
    ('un-',         0.103, 1.00, 0.06,  0.86),
    ('capital',     0.317, 0.00, 0.00,  1.00),
]
print("  %-12s  pc     in%%   LOO%%  irred%%  type" % "axis")
print("  " + "-"*62)
for nm, pc, ins, loo, irred in ALL_KNOWN:
    if ins < 0.15:                          t = 'named_entity'
    elif loo >= 0.60:                       t = 'morph_uniform'
    elif ins >= 0.85 and loo < 0.30:        t = 'phonol_scatter'
    elif ins >= 0.85 and loo >= 0.30:       t = 'morph_moderate'
    else:                                   t = 'semantic_diverse'
    print("  %-12s  %.3f  %.0f%%    %.0f%%   %.0f%%     %s" %
          (nm, pc, 100*ins, 100*loo, 100*irred, t))
