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
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, 0.0, valid, 0.0
    chord_norms = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    coh = float(np.mean([np.dot(cn, md.astype(np.float32)) for cn in chord_norms]))
    sims = [float(np.dot(chord_norms[i], chord_norms[j]))
            for i in range(len(chord_norms)) for j in range(i+1, len(chord_norms))]
    pc = float(np.mean(sims)) if sims else 0.0
    return md, coh, valid, pc
def nn_retrieve(pred_emb, exclude_ids, top_n=1):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale_acc(axis, valid_pairs, lo=0.02, hi=6.0, n=60):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for s_,t_,sid,tid in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t_)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def source_homogeneity(pairs):
    src_embs = []
    for s, t in pairs:
        es, sid = get_emb(s)
        if es is not None: src_embs.append(normed(es).astype(np.float32))
    if len(src_embs) < 2: return 0.0
    sims = [float(np.dot(src_embs[i], src_embs[j]))
            for i in range(len(src_embs)) for j in range(i+1, len(src_embs))]
    return float(np.mean(sims)) if sims else 0.0

print("DAY 291: UNIFIED LINEARITY TEST — SEMANTIC AXES")
print("="*65)
print("Does pairwise chord cosine predict accuracy for semantic axes")
print("the same way it predicts accuracy for morphological axes?")
print()

# ====================================================================
# MORPHOLOGICAL AXES (from previous days)
# ====================================================================
MORPH_AXES = {
    '+est (sup)': [('fast','fastest'),('slow','slowest'),('tall','tallest'),
                   ('small','smallest'),('large','largest'),('hard','hardest'),
                   ('soft','softest'),('warm','warmest'),('dark','darkest'),
                   ('clean','cleanest'),('sharp','sharpest'),('deep','deepest'),
                   ('wide','widest'),('strong','strongest'),('long','longest'),('old','oldest')],
    '+er (comp)': [('fast','faster'),('slow','slower'),('tall','taller'),
                   ('small','smaller'),('large','larger'),('hard','harder'),
                   ('soft','softer'),('warm','warmer'),('dark','darker'),
                   ('clean','cleaner'),('sharp','sharper'),('deep','deeper'),
                   ('wide','wider'),('strong','stronger'),('long','longer'),('old','older')],
    '+s (plural)': [('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),
                    ('book','books'),('car','cars'),('hand','hands'),('eye','eyes'),
                    ('word','words'),('day','days'),('year','years'),('house','houses'),
                    ('arm','arms'),('leg','legs'),('door','doors'),('line','lines')],
    'gender':      [('king','queen'),('man','woman'),('boy','girl'),('son','daughter'),
                    ('brother','sister'),('father','mother'),('uncle','aunt'),('prince','princess'),
                    ('hero','heroine'),('actor','actress'),('waiter','waitress'),('god','goddess')],
    '+ed (past_r)':[('walk','walked'),('talk','talked'),('work','worked'),('play','played'),
                    ('call','called'),('turn','turned'),('start','started'),('move','moved'),
                    ('live','lived'),('love','loved'),('use','used'),('ask','asked'),
                    ('seem','seemed'),('help','helped'),('want','wanted'),('need','needed')],
    'past_irr':    [('feel','felt'),('run','ran'),('go','went'),('get','got'),
                    ('say','said'),('make','made'),('take','took'),('see','saw'),
                    ('know','knew'),('come','came'),('give','gave'),('think','thought'),
                    ('find','found'),('tell','told'),('keep','kept'),('leave','left')],
}

# ====================================================================
# SEMANTIC AXES (from Days 281-285)
# ====================================================================
SEMANTIC_AXES = {
    'nat->lang':   [('French','French'),('German','German'),('Spanish','Spanish'),
                    ('Italian','Italian'),('Portuguese','Portuguese'),('Japanese','Japanese'),
                    ('Chinese','Chinese'),('Russian','Russian'),('Arabic','Arabic'),
                    ('Dutch','Dutch'),('Swedish','Swedish'),('Polish','Polish'),
                    ('Korean','Korean'),('Turkish','Turkish'),('Greek','Greek'),
                    ('English','English')],
    # nat->lang axis: nationality word -> language word
    # These are the same word! Need the actual transformation pairs
    'person->nat': [('Einstein','German'),('Shakespeare','English'),('Napoleon','French'),
                    ('Goethe','German'),('Tolstoy','Russian'),('Cervantes','Spanish'),
                    ('Dante','Italian'),('Confucius','Chinese'),('Gandhi','Indian'),
                    ('Newton','English'),('Darwin','English'),('Pasteur','French'),
                    ('Mozart','Austrian'),('Bach','German'),('Beethoven','German'),
                    ('Voltaire','French')],
    'animal->class':[('cat','mammal'),('dog','mammal'),('horse','mammal'),('whale','mammal'),
                     ('eagle','bird'),('robin','bird'),('sparrow','bird'),('hawk','bird'),
                     ('salmon','fish'),('shark','fish'),('tuna','fish'),('trout','fish'),
                     ('frog','amphibian'),('cobra','reptile'),('lizard','reptile')],
    'country->cap': [('France','Paris'),('Germany','Berlin'),('Spain','Madrid'),
                     ('Italy','Rome'),('Japan','Tokyo'),('China','Beijing'),
                     ('Russia','Moscow'),('Egypt','Cairo'),('Brazil','Brasilia'),
                     ('India','Delhi'),('Turkey','Ankara'),('Greece','Athens'),
                     ('Poland','Warsaw'),('Sweden','Stockholm'),('Norway','Oslo')],
    'word->antonym':[('hot','cold'),('fast','slow'),('big','small'),('strong','weak'),
                     ('light','dark'),('high','low'),('old','young'),('rich','poor'),
                     ('happy','sad'),('love','hate'),('war','peace'),('good','bad'),
                     ('start','end'),('open','close'),('push','pull')],
    'country->demonym': [('France','French'),('Germany','German'),('Spain','Spanish'),
                         ('Italy','Italian'),('Japan','Japanese'),('China','Chinese'),
                         ('Russia','Russian'),('Egypt','Egyptian'),('Brazil','Brazilian'),
                         ('Portugal','Portuguese'),('Sweden','Swedish'),('Greece','Greek'),
                         ('Poland','Polish'),('Turkey','Turkish'),('Korea','Korean')],
    'field->concept':  [('physics','gravity'),('chemistry','atom'),('biology','cell'),
                        ('mathematics','equation'),('psychology','behavior'),
                        ('economics','market'),('linguistics','grammar'),
                        ('philosophy','logic'),('medicine','diagnosis'),
                        ('astronomy','telescope'),('geology','erosion'),
                        ('sociology','culture'),('history','event'),('music','melody')],
}

# Fix nat->lang with actual pairs (nationality->language)
# These are the demonym->language pairs
SEMANTIC_AXES['nat->lang'] = [
    ('French','French'),('German','German'),('Spanish','Spanish'),
    ('Italian','Italian'),('Portuguese','Portuguese'),('Japanese','Japanese'),
    ('Chinese','Chinese'),('Russian','Russian'),('Arabic','Arabic'),
    ('Dutch','Dutch'),('Swedish','Swedish'),('Korean','Korean'),
    ('Turkish','Turkish'),('Greek','Greek'),('Polish','Polish'),
]
# This is circular -- need actual nationality to language transformation
# The true nat->lang axis: nationality word -> language name (often same!)
# Let's use the actual data from Day 282 — nationality adjective to language noun
# In practice French->French is the same token, so let's use country->language
SEMANTIC_AXES['country->lang'] = [
    ('France','French'),('Germany','German'),('Spain','Spanish'),
    ('Italy','Italian'),('Portugal','Portuguese'),('Japan','Japanese'),
    ('China','Chinese'),('Russia','Russian'),('Egypt','Arabic'),
    ('Netherlands','Dutch'),('Sweden','Swedish'),('Korea','Korean'),
    ('Turkey','Turkish'),('Greece','Greek'),('Poland','Polish'),
    ('Britain','English'),('Brazil','Portuguese'),('Mexico','Spanish'),
]
del SEMANTIC_AXES['nat->lang']

# ====================================================================
# COMPUTE LINEARITY AND ACCURACY FOR ALL AXES
# ====================================================================
print("Computing pairwise chord cosine, coherence, source homogeneity, accuracy...")
print()

all_results = {}
for name, pairs in {**MORPH_AXES, **SEMANTIC_AXES}.items():
    ax, coh, valid, pc = compute_axis_full(pairs)
    if ax is None or len(valid) < 4:
        print("  %-22s SKIP (only %d valid)" % (name, len(valid))); continue
    s_opt, acc = best_scale_acc(ax, valid)
    src_hom = source_homogeneity(pairs)
    all_results[name] = (ax, coh, pc, src_hom, s_opt, acc, len(valid))

# ====================================================================
# PRINT UNIFIED COMPARISON TABLE
# ====================================================================
print("UNIFIED LINEARITY TABLE (sorted by pairwise chord cosine):")
print("-"*80)
print("  %-22s  pc_cos  coh    src_hom  scale  acc%%   n    type" % "Axis")
print("  " + "-"*76)

sorted_results = sorted(all_results.items(), key=lambda x: -x[1][2])
for name, (ax, coh, pc, src_hom, s_opt, acc, n) in sorted_results:
    atype = "MORPH" if name in MORPH_AXES else "SEMANTIC"
    print("  %-22s  %.4f  %.4f  %.4f   %.2f  %3.0f%%  %2d  %s" % (
        name, pc, coh, src_hom, s_opt, 100*acc/max(1,n), n, atype))
print()

# ====================================================================
# CORRELATION: does pairwise_cos predict accuracy?
# ====================================================================
print("CORRELATION ANALYSIS: pairwise_cos vs accuracy")
print("-"*65)
pcs  = np.array([r[2] for r in all_results.values()])
accs = np.array([r[5]/max(1,r[6]) for r in all_results.values()])
homs = np.array([r[3] for r in all_results.values()])

corr_pc_acc  = float(np.corrcoef(pcs, accs)[0,1])
corr_hom_acc = float(np.corrcoef(homs, accs)[0,1])
corr_hom_pc  = float(np.corrcoef(homs, pcs)[0,1])

print("  corr(pairwise_cos, accuracy)    = %+.4f" % corr_pc_acc)
print("  corr(src_homogeneity, accuracy) = %+.4f" % corr_hom_acc)
print("  corr(src_homogeneity, pairwise) = %+.4f" % corr_hom_pc)
print()

# ====================================================================
# SEMANTIC AXIS DETAILS
# ====================================================================
print("SEMANTIC AXIS DETAIL: country->cap (expected high accuracy)")
print("-"*65)
name = 'country->cap'
if name in all_results:
    ax, coh, pc, src_hom, s_opt, acc, n = all_results[name]
    for s, t in SEMANTIC_AXES[name]:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        r = nn_retrieve(W_E[sid] + s_opt * ax, [sid])
        got = r[0][0] if r else '?'
        print("  %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
print()

print("SEMANTIC AXIS DETAIL: word->antonym")
print("-"*65)
name = 'word->antonym'
if name in all_results:
    ax, coh, pc, src_hom, s_opt, acc, n = all_results[name]
    for s, t in SEMANTIC_AXES[name]:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        r = nn_retrieve(W_E[sid] + s_opt * ax, [sid])
        got = r[0][0] if r else '?'
        print("  %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
print()

# ====================================================================
# HOLDOUT TEST: country->demonym (trained on 10, tested on 5)
# ====================================================================
print("HOLDOUT TEST: country->demonym (trained on 10, tested on 5)")
print("-"*65)
dem_pairs = SEMANTIC_AXES['country->demonym']
dem_train = dem_pairs[:10]; dem_hold = dem_pairs[10:]
ax_dem, coh_dem, valid_dem, pc_dem = compute_axis_full(dem_train)
if ax_dem is not None:
    s_dem, acc_dem = best_scale_acc(ax_dem, valid_dem)
    print("  Train: axis coh=%.4f  pc=%.4f  acc=%d/%d  scale=%.2f" % (
        coh_dem, pc_dem, acc_dem, len(valid_dem), s_dem))
    print("  Holdout:")
    for s, t in dem_hold:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        r = nn_retrieve(W_E[sid] + s_dem * ax_dem, [sid])
        got = r[0][0] if r else '?'
        print("    %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
print()

# ====================================================================
# THE KEY QUESTION: Do pairwise_cos thresholds hold across domains?
# ====================================================================
print("LINEARITY THRESHOLD ANALYSIS:")
print("-"*65)
print("  Does pairwise_cos > 0.3 => high accuracy hold for semantic axes?")
print()

for name, (ax, coh, pc, src_hom, s_opt, acc, n) in sorted_results:
    cat = "HIGH" if pc > 0.30 else ("MED" if pc > 0.15 else "LOW")
    acc_pct = 100*acc/max(1,n)
    expected = ">80%" if cat=="HIGH" else ("50-80%" if cat=="MED" else "<60%")
    actual = ">80%" if acc_pct > 80 else ("50-80%" if acc_pct >= 50 else "<50%")
    match = "OK" if expected == actual else "MISMATCH"
    print("  %-22s  pc=%.3f [%s]  acc=%3.0f%%  exp=%s  %s" % (
        name, pc, cat, acc_pct, expected, match))
print()

# ====================================================================
# ANTONYM AXIS: unique test
# Antonyms are OPPOSITIONAL — axis should reverse polarity
# ====================================================================
print("ANTONYM AXIS DEEP DIVE: oppositional semantics")
print("-"*65)
print("Antonym axis should point OPPOSITE to word meaning.")
print("Testing: does hot->cold axis predict warm->cool?")
print()

# Test antonym generalisation on holdout adjectives
ANT_TRAIN = [('hot','cold'),('fast','slow'),('big','small'),('strong','weak'),
             ('light','dark'),('high','low'),('old','young'),('rich','poor')]
ANT_HOLD  = [('warm','cool'),('loud','quiet'),('rough','smooth'),('deep','shallow'),
             ('wide','narrow'),('hard','soft'),('sharp','blunt'),('long','short'),
             ('near','far'),('early','late'),('hard','easy'),('thick','thin')]

ax_ant, coh_ant, valid_ant, pc_ant = compute_axis_full(ANT_TRAIN)
if ax_ant is not None:
    s_ant, acc_ant = best_scale_acc(ax_ant, valid_ant, hi=8.0)
    valid_hold_ant = [(s,t,sid,tid) for s,t in ANT_HOLD
                      for es,sid in [get_emb(s)] for et,tid in [get_emb(t)]
                      if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    valid_hold_ant = []
    for s, t in ANT_HOLD:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        valid_hold_ant.append((s, t, sid, tid))

    print("  Train: pc=%.4f  coh=%.4f  acc=%d/%d  scale=%.2f" % (
        pc_ant, coh_ant, acc_ant, len(valid_ant), s_ant))
    acc_h_ant = 0
    for s_, t_, sid, tid in valid_hold_ant:
        r = nn_retrieve(W_E[sid] + s_ant * ax_ant, [sid])
        got = r[0][0] if r else '?'
        hit = (got == t_)
        if hit: acc_h_ant += 1
        print("  %-12s -> %-12s  got=%-12s [%s]" % (s_, t_, got, 'HIT' if hit else '---'))
    print()
    print("  Holdout: %d/%d (%.0f%%)" % (acc_h_ant, len(valid_hold_ant),
        100*acc_h_ant/max(1,len(valid_hold_ant))))
print()

# ====================================================================
# SUMMARY: THE UNIFIED PRINCIPLE
# ====================================================================
print("="*65)
print("UNIFIED LINEARITY PRINCIPLE SUMMARY")
print("="*65)
print()
print("  pairwise chord cosine => predicted accuracy:")
print("  pc > 0.35  => HIGH    (>85%)")
print("  pc 0.15-0.35 => MEDIUM (50-80%)")
print("  pc < 0.15  => LOW     (<60% even with 20+ pairs)")
print()
print("  Observed:")
for name, (ax, coh, pc, src_hom, s_opt, acc, n) in sorted_results:
    print("    %-22s  pc=%.3f  acc=%3.0f%%  src_hom=%.3f" % (
        name, pc, 100*acc/max(1,n), src_hom))
print()
print("  Correlation pairwise_cos->accuracy:", round(corr_pc_acc, 4))
print("  Correlation src_homogeneity->accuracy:", round(corr_hom_acc, 4))
print("  Correlation src_homogeneity->pairwise:", round(corr_hom_pc, 4))
