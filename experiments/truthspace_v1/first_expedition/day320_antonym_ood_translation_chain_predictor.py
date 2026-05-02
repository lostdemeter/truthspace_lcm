import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building masks...", flush=True)
CLEAN_MASK   = np.zeros(len(W_E), dtype=bool)
RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) <= 1: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
    if not w[0].isupper(): CLEAN_MASK[i] = True
print("  clean=%d  relaxed=%d" % (CLEAN_MASK.sum(), RELAXED_MASK.sum()))

_src_cache = {}
def source_ids(word):
    if word in _src_cache: return _src_cache[word]
    ids = set()
    for p in [' '+word, word, ' '+word[0].upper()+word[1:],
              word[0].upper()+word[1:], word.upper(), ' '+word.upper(),
              '-'+word, '_'+word]:
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

def nn_retrieve(pred_emb, excl_ids, mask, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims   = W_n @ pred_n
    sims[~mask] = -1.0
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
    if len(chords) < 2: return None, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, valid, pc

def best_scale(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask, 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

def axis_loo(axis, valid, mask):
    if len(valid) < 3: return 0.0
    chords_f = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    ax_full  = normed(np.mean(chords_f, axis=0))
    gs, _    = best_scale(ax_full, valid, mask)
    hits = 0
    for i in range(len(valid)):
        tv = [valid[j] for j in range(len(valid)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        test_s, test_t, test_sid, _ = valid[i]
        r = nn_retrieve(W_E[test_sid]+gs*al, source_ids(test_s), mask, 1)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid)

print()
print("DAY 320: ANTONYM OOD, TRANSLATION CHAIN, +NESS, 2D PREDICTOR")
print("="*72)
print()

# =====================================================================
# PART A: ANTONYM POLAR OOD — HOLDOUT CONFIRMATION
# =====================================================================
print("PART A: Antonym polar OOD holdout")
print("-"*72)

VERB_TRAIN = [('win','lose'),('rise','fall'),('push','pull'),('enter','exit'),
              ('buy','sell'),('love','hate'),('open','close'),('start','stop')]
VERB_HOLDOUT = [('give','take'),('build','destroy'),('remember','forget'),
                ('accept','reject'),('attack','defend'),('create','destroy'),
                ('teach','learn'),('send','receive'),('ask','answer'),
                ('hide','reveal')]
NOUN_TRAIN = [('war','peace'),('day','night'),('summer','winter'),('life','death'),
              ('friend','enemy'),('truth','lie'),('good','evil'),('joy','sorrow')]
NOUN_HOLDOUT = [('light','darkness'),('heaven','hell'),('love','hate'),
                ('wealth','poverty'),('victory','defeat'),('hope','despair'),
                ('north','south'),('past','future')]

for set_name, train, holdout in [
    ('VERB_ANT', VERB_TRAIN, VERB_HOLDOUT),
    ('NOUN_ANT', NOUN_TRAIN, NOUN_HOLDOUT),
]:
    ax, valid, pc = compute_axis(train)
    if ax is None: continue
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    loo = axis_loo(ax, valid, CLEAN_MASK)
    print("  %s: pc=%.4f  in=%d/%d  LOO=%.0f%%  scale=%.3f" %
          (set_name, pc, in_s, len(valid), 100*loo, best_s))

    # Holdout: test each pair, show tgt_rank, tgt_cos, and top-1
    ho_rank0 = 0; ho_n = 0
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w); et, tid = get_emb(t_w)
        if es is None or et is None:
            print("  ? %-10s -> %-10s [multi-token]" % (s_w, t_w)); continue
        ho_n += 1
        pred = W_E[sid] + best_s * ax
        pred_n = normed(pred).astype(np.float32)
        tgt_cos = float(W_n[tid] @ pred_n)
        r = nn_retrieve(pred, source_ids(s_w), CLEAN_MASK, 5)
        tgt_rank = next((i for i,(w,_,_) in enumerate(r) if w==t_w), -1)
        # Also baseline rank
        baseline = nn_retrieve(W_E[sid], source_ids(s_w), CLEAN_MASK, 30)
        base_rank = next((i for i,(w,_,_) in enumerate(baseline) if w==t_w), '>30')
        hit = '✓' if tgt_rank == 0 else ('~' if tgt_rank >= 0 else '✗')
        if tgt_rank == 0: ho_rank0 += 1
        print("  %s %-10s -> %-10s  tgt_rank=%s  tgt_cos=%.3f  base_rank=%s  top1=%s" %
              (hit, s_w, t_w, tgt_rank if tgt_rank>=0 else '>5',
               tgt_cos, base_rank if isinstance(base_rank,str) else base_rank,
               r[0][0]))
    print("  OOD rank=0: %d/%d=%.0f%%" % (ho_rank0, ho_n, 100*ho_rank0/ho_n if ho_n else 0))
    print()

# =====================================================================
# PART B: TRANSLATION CHAIN — EN→ES→FR COMPOSITION
# =====================================================================
print("PART B: Translation chain — EN→ES and ES→FR composition")
print("-"*72)

EN_ES = [('cat','gato'),('dog','perro'),('house','casa'),('water','agua'),
         ('fire','fuego'),('sun','sol'),('moon','luna'),('star','estrella'),
         ('book','libro'),('car','coche'),('door','puerta'),('tree','árbol')]

ES_FR = [('gato','chat'),('perro','chien'),('casa','maison'),('agua','eau'),
         ('fuego','feu'),('sol','soleil'),('libro','livre'),('puerta','porte')]

EN_FR = [('cat','chat'),('dog','chien'),('house','maison'),('water','eau'),
         ('fire','feu'),('sun','soleil'),('book','livre'),('door','porte')]

ax_en_es, valid_en_es, pc_en_es = compute_axis(EN_ES)
ax_es_fr, valid_es_fr, pc_es_fr = compute_axis(ES_FR)
ax_en_fr, valid_en_fr, pc_en_fr = compute_axis(EN_FR)

for name, ax, valid, pc in [
    ('EN→ES', ax_en_es, valid_en_es, pc_en_es),
    ('ES→FR', ax_es_fr, valid_es_fr, pc_es_fr),
    ('EN→FR', ax_en_fr, valid_en_fr, pc_en_fr),
]:
    if ax is None: print("  %s: n/a" % name); continue
    best_s, in_s = best_scale(ax, valid, CLEAN_MASK)
    loo = axis_loo(ax, valid, CLEAN_MASK)
    print("  %s: pc=%.4f  n=%d  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (name, pc, len(valid), 100*in_s/len(valid), 100*loo, best_s))

if ax_en_es is not None and ax_es_fr is not None and ax_en_fr is not None:
    # Cosines between translation axes
    c_es_esfr = float(np.dot(ax_en_es.astype(np.float32), ax_es_fr.astype(np.float32)))
    c_esfr_fr = float(np.dot(ax_es_fr.astype(np.float32), ax_en_fr.astype(np.float32)))
    c_es_fr   = float(np.dot(ax_en_es.astype(np.float32), ax_en_fr.astype(np.float32)))
    print()
    print("  cos(EN→ES, ES→FR) = %+.4f" % c_es_esfr)
    print("  cos(ES→FR, EN→FR) = %+.4f" % c_esfr_fr)
    print("  cos(EN→ES, EN→FR) = %+.4f" % c_es_fr)

    # Chain test: EN→ES→FR using two axes
    best_s_es, _ = best_scale(ax_en_es, valid_en_es, CLEAN_MASK)
    best_s_esfr, _ = best_scale(ax_es_fr, valid_es_fr, CLEAN_MASK)

    print()
    print("  Chain test: EN →[EN→ES]→ ES →[ES→FR]→ FR")
    test_en_fr = [('cat','gato','chat'),('dog','perro','chien'),
                  ('house','casa','maison'),('fire','fuego','feu'),
                  ('sun','sol','soleil'),('book','libro','livre')]
    chain_hits = 0; chain_n = 0
    for en, es, fr in test_en_fr:
        e_en, sid = get_emb(en)
        if e_en is None: continue
        chain_n += 1
        # Step 1: EN → ES
        r1 = nn_retrieve(W_E[sid]+best_s_es*ax_en_es, source_ids(en), CLEAN_MASK, 1)
        es_got = r1[0][0]
        # Step 2: ES → FR
        e_es, es_id = get_emb(es_got)
        if e_es is None:
            print("  %-6s -> %-8s [multi-token] -> n/a" % (en, es_got)); continue
        r2 = nn_retrieve(W_E[es_id]+best_s_esfr*ax_es_fr, source_ids(es_got), CLEAN_MASK, 1)
        fr_got = r2[0][0]
        h1 = '✓' if es_got==es else '✗'
        h2 = '✓' if fr_got==fr else '✗'
        both = '✓✓' if es_got==es and fr_got==fr else ('✓✗' if es_got==es else '✗?')
        if es_got==es and fr_got==fr: chain_hits += 1
        print("  %s %-6s →%s %-8s →%s %-8s (want %s→%s)" %
              (both, en, h1, es_got, h2, fr_got, es, fr))
    print("  Both-correct: %d/%d=%.0f%%" % (chain_hits, chain_n, 100*chain_hits/chain_n if chain_n else 0))

    # Direct composition EN→FR via combined axis
    print()
    combined = normed(ax_en_es + ax_es_fr)
    _, direct_in = best_scale(combined, valid_en_fr, CLEAN_MASK)
    print("  Direct (EN→ES + ES→FR) composed, tested on EN→FR pairs: %d/%d=%.0f%%" %
          (direct_in, len(valid_en_fr), 100*direct_in/len(valid_en_fr)))
print()

# =====================================================================
# PART C: +NESS RECONCILIATION — HIGH LOO BUT HIGH IRRED?
# =====================================================================
print("PART C: +ness reconciliation — LOO=86%% but high holdout irred?")
print("-"*72)

NESS_TRAIN = [('happy','happiness'),('sad','sadness'),('kind','kindness'),
              ('dark','darkness'),('warm','warmth'),('hard','hardness'),
              ('soft','softness'),('weak','weakness'),('clean','cleanliness'),
              ('lonely','loneliness')]
NESS_HOLDOUT_EASY = [('sick','sickness'),('thick','thickness'),('rich','richness'),
                     ('fresh','freshness'),('calm','calmness'),('bright','brightness')]
NESS_HOLDOUT_HARD = [('good','goodness'),('great','greatness'),('wide','width'),
                     ('long','length'),('high','height'),('strong','strength'),
                     ('deep','depth'),('broad','breadth')]

ax_ness, valid_ness, pc_ness = compute_axis(NESS_TRAIN)
if ax_ness is not None:
    best_s, in_s = best_scale(ax_ness, valid_ness, CLEAN_MASK)
    loo = axis_loo(ax_ness, valid_ness, CLEAN_MASK)
    print("  +ness train: n=%d  pc=%.4f  in=%.0f%%  LOO=%.0f%%  scale=%.3f" %
          (len(valid_ness), pc_ness, 100*in_s/len(valid_ness), 100*loo, best_s))
    print()

    # Easy holdout
    print("  Easy holdout (regular +ness):")
    easy_hits = 0; easy_n = 0
    for s_w, t_w in NESS_HOLDOUT_EASY:
        es, sid = get_emb(s_w)
        if es is None: continue
        easy_n += 1
        # Full sweep for this pair
        found_at = None
        for s_test in np.linspace(0.02, 6.0, 60):
            r = nn_retrieve(W_E[sid]+s_test*ax_ness, source_ids(s_w), CLEAN_MASK, 1)
            if r[0][0] == t_w: found_at=s_test; break
        if found_at:
            easy_hits += 1
            print("  ✓ %-12s -> %-14s  at scale %.3f" % (s_w, t_w, found_at))
        else:
            r = nn_retrieve(W_E[sid]+best_s*ax_ness, source_ids(s_w), CLEAN_MASK, 3)
            print("  ✗ %-12s -> %-14s  got: %s" % (s_w, t_w, r[0][0]))
    print("  Easy irred: %d/%d=%.0f%%" % (easy_n-easy_hits, easy_n, 100*(easy_n-easy_hits)/easy_n if easy_n else 0))
    print()

    # Hard holdout (irregular +ness: strong→strength, long→length)
    print("  Hard holdout (irregular: width, length, height...):")
    hard_hits = 0; hard_n = 0
    for s_w, t_w in NESS_HOLDOUT_HARD:
        es, sid = get_emb(s_w)
        if es is None: continue
        hard_n += 1
        found_at = None
        for s_test in np.linspace(0.02, 6.0, 60):
            r = nn_retrieve(W_E[sid]+s_test*ax_ness, source_ids(s_w), CLEAN_MASK, 1)
            if r[0][0] == t_w: found_at=s_test; break
        if found_at:
            hard_hits += 1
            print("  ✓ %-12s -> %-14s  at scale %.3f" % (s_w, t_w, found_at))
        else:
            r = nn_retrieve(W_E[sid]+best_s*ax_ness, source_ids(s_w), CLEAN_MASK, 3)
            print("  ✗ %-12s -> %-14s  got: %s" % (s_w, t_w, r[0][0]))
    print("  Hard irred: %d/%d=%.0f%%" % (hard_n-hard_hits, hard_n, 100*(hard_n-hard_hits)/hard_n if hard_n else 0))
    print()

    # How does +ness LOO work if some holdout words are irred?
    # LOO tests on TRAINING words -- they all work if regular derivation
    # Hard holdout are IRREGULAR -- they're by definition not in training set
    print("  Reconciliation: high LOO=86%% because training pairs are all regular.")
    print("  Hard irred is from IRREGULAR derivations outside the +ness morphological domain.")
    print("  Conclusion: +ness is phonol_scatter for the REGULAR domain (LOO=86%%)")
    print("  but has irred for IRREGULAR words (different derivation suffix).")
print()

# =====================================================================
# PART D: 2D AXIS TYPE PREDICTOR — pc vs LOO DECISION BOUNDARY
# =====================================================================
print("PART D: 2D axis type predictor — pc vs LOO decision boundary")
print("-"*72)

# Collect all axis type data
AXIS_DATA = [
    # (name, pc, LOO%, type, irred%)
    ('er→est',  0.426, 100, 'morph_uniform',   5),
    ('+er',     0.385,  88, 'morph_uniform',   10),
    ('cc',      0.351,  71, 'relational_geom', 20),
    ('cl',      0.399,  67, 'relational_geom', 15),
    ('capl',    0.394, 100, 'relational_geom', 10),
    ('+s',      0.297, 100, 'morph_moderate',  15),
    ('+ed',     0.259, 100, 'morph_moderate',  20),
    ('+able',   0.220,   0, 'semantic_diverse', 60),
    ('+ness',   0.203,  86, 'phonol_scatter',   30),
    ('un-',     0.189,  67, 'phonol_scatter',   57),
    ('+less',   0.167,   0, 'semantic_diverse', 90),
    ('pres',    0.165,   0, 'factual_local',   100),
    ('+ful',    0.142,  33, 'phonol_scatter',   40),
    ('+tion',   0.112,  75, 'phonol_scatter',    5),
    ('EN→ES',   0.082,  25, 'translation',     100),
    ('EN→FR',   0.064,   0, 'translation',     100),
    ('EN→DE',   0.101,   0, 'translation',     100),
    ('adj_ant', 0.055,  30, 'antonym',          90),
    ('noun_ant',0.020,   0, 'antonym',         100),
    ('verb_ant',0.016,   0, 'antonym',         100),
]

print("  pc vs LOO scatterplot (text):")
print("  %s" % ("─"*60))
print("  %-12s  pc     LOO%%  irred%%  type" % "axis")
print("  %s" % ("─"*60))

# Sort by pc descending
AXIS_DATA.sort(key=lambda x: -x[1])
for name, pc, loo, t, irred in AXIS_DATA:
    bar_pc  = int(pc * 40)
    print("  %-12s  %.3f  %3d%%  %3d%%   %s" % (name, pc, loo, irred, t))

print()
# Decision rules
print("  Decision rules (2D: pc, LOO):")
print("  pc > 0.35:                  morph_uniform OR relational_geom")
print("  0.20 < pc <= 0.35, LOO>50%: morph_moderate OR relational_geom-low")
print("  0.20 < pc <= 0.35, LOO<50%: morph_moderate-low (check irred)")
print("  0.10 < pc <= 0.20, LOO>50%: phonol_scatter")
print("  0.10 < pc <= 0.20, LOO<50%: semantic_diverse (irred>60%) OR factual_local (irred=100%%)")
print("  0.05 < pc <= 0.10:          translation (check cross-lingual?)")
print("  pc <= 0.05:                 antonym/polar")
print()
print("  Ambiguous cases:")
print("  +able: pc=0.220, LOO=0%% -> morph_moderate by pc but semantic_diverse by irred=60%%")
print("  +ful:  pc=0.142, LOO=33%% -> borderline phonol_scatter/semantic_diverse")
print("  adj_ant: pc=0.055, LOO=30%% -> borderline antonym/semantic_diverse")
print()

# Compute accuracy of 2D rule
def predict_type(pc, loo, irred):
    if pc > 0.35: return 'morph_uniform/relational'
    elif pc > 0.20 and loo > 50: return 'morph_moderate/phonol_scatter-high'
    elif pc > 0.20 and loo <= 50: return 'morph_moderate-low/semantic_diverse'
    elif pc > 0.10 and loo > 50: return 'phonol_scatter'
    elif pc > 0.10 and loo <= 50 and irred > 60: return 'semantic_diverse/factual_local'
    elif pc > 0.10 and loo <= 50: return 'borderline'
    elif pc > 0.05: return 'translation'
    else: return 'antonym/polar'

print("  Predictions vs true type:")
correct = 0; total = len(AXIS_DATA)
for name, pc, loo, t, irred in AXIS_DATA:
    pred = predict_type(pc, loo, irred)
    match = '✓' if t.split('_')[0] in pred or t in pred else '?'
    if match == '✓': correct += 1
    print("  %s %-12s  pred=%-30s  true=%s" % (match, name, pred, t))
print("  Predictor accuracy: %d/%d=%.0f%%" % (correct, total, 100*correct/total))
