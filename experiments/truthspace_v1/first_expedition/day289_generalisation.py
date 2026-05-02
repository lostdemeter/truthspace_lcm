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
def nn_retrieve(pred_emb, exclude_ids, top_n=1):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def eval_axis(axis, scale, test_pairs, fwd=True):
    correct = 0
    for s, t, sid, tid in test_pairs:
        if fwd:
            r = nn_retrieve(W_E[sid] + scale * axis, [sid])
        else:
            r = nn_retrieve(W_E[tid] - scale * axis, [tid])
        if r and r[0][0] == (t if fwd else s): correct += 1
    return correct, len(test_pairs)
def best_scale_on(axis, valid_pairs, lo=0.02, hi=4.0, n=50, fwd=True):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c, _ = eval_axis(axis, s, valid_pairs, fwd=fwd)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 289: GENERALISATION TEST")
print("="*65)
print("Do morphological axes transfer to UNSEEN words?")
print("Method: train axis on subset, test on holdout.")
print("'Structure IS information' hypothesis predicts: YES.")
print()

# ====================================================================
# DEFINE LARGE WORD POOLS
# ====================================================================

# PLURAL — 60+ pairs, split train/holdout
ALL_PLURAL = [
    # very common nouns (train set)
    ('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),
    ('book','books'),('car','cars'),('hand','hands'),('eye','eyes'),
    ('word','words'),('day','days'),('year','years'),('house','houses'),
    ('arm','arms'),('leg','legs'),('door','doors'),('line','lines'),
    ('way','ways'),('part','parts'),('name','names'),('place','places'),
    # less common nouns (holdout)
    ('lamp','lamps'),('desk','desks'),('chair','chairs'),('cup','cups'),
    ('plate','plates'),('spoon','spoons'),('fork','forks'),('knife','knives'),
    ('river','rivers'),('mountain','mountains'),('cloud','clouds'),('star','stars'),
    ('stone','stones'),('flower','flowers'),('garden','gardens'),('bridge','bridges'),
    ('road','roads'),('street','streets'),('town','towns'),('village','villages'),
    ('teacher','teachers'),('student','students'),('doctor','doctors'),('nurse','nurses'),
    ('soldier','soldiers'),('farmer','farmers'),('artist','artists'),('writer','writers'),
    ('fish','fish'),('sheep','sheep'),  # zero-plural (should fail gracefully)
]

# GENDER — 16 pairs, split train/holdout
ALL_GENDER = [
    # train
    ('king','queen'),('man','woman'),('boy','girl'),('son','daughter'),
    ('brother','sister'),('father','mother'),('uncle','aunt'),('prince','princess'),
    # holdout
    ('hero','heroine'),('actor','actress'),('waiter','waitress'),('god','goddess'),
    ('duke','duchess'),('lion','lioness'),('tiger','tigress'),('monk','nun'),
]

# COMPARATIVE — 30 pairs, split train/holdout
ALL_COMP = [
    # train (20)
    ('fast','faster'),('slow','slower'),('tall','taller'),('small','smaller'),
    ('large','larger'),('hard','harder'),('soft','softer'),('warm','warmer'),
    ('cool','cooler'),('bright','brighter'),('dark','darker'),('clean','cleaner'),
    ('sharp','sharper'),('deep','deeper'),('wide','wider'),('strong','stronger'),
    ('long','longer'),('short','shorter'),('cheap','cheaper'),('fresh','fresher'),
    # holdout (10)
    ('thick','thicker'),('thin','thinner'),('rough','rougher'),('smooth','smoother'),
    ('sweet','sweeter'),('bitter','bitterer'),('quiet','quieter'),('loud','louder'),
    ('safe','safer'),('brave','braver'),
]

# SUPERLATIVE — same split
ALL_SUP = [
    # train (20)
    ('fast','fastest'),('slow','slowest'),('tall','tallest'),('small','smallest'),
    ('large','largest'),('hard','hardest'),('soft','softest'),('warm','warmest'),
    ('cool','coolest'),('bright','brightest'),('dark','darkest'),('clean','cleanest'),
    ('sharp','sharpest'),('deep','deepest'),('wide','widest'),('strong','strongest'),
    ('long','longest'),('short','shortest'),('cheap','cheapest'),('fresh','freshest'),
    # holdout (10)
    ('thick','thickest'),('thin','thinnest'),('rough','roughest'),('smooth','smoothest'),
    ('sweet','sweetest'),('quiet','quietest'),('loud','loudest'),('safe','safest'),
    ('brave','bravest'),('cold','coldest'),
]

# PAST TENSE — split into regular vs irregular
ALL_PAST_REG = [
    # train (regular)
    ('walk','walked'),('talk','talked'),('work','worked'),('play','played'),
    ('call','called'),('turn','turned'),('start','started'),('move','moved'),
    ('live','lived'),('love','loved'),('use','used'),('ask','asked'),
    ('seem','seemed'),('help','helped'),('want','wanted'),('need','needed'),
    # holdout (regular)
    ('jump','jumped'),('push','pushed'),('pull','pulled'),('kick','kicked'),
    ('open','opened'),('close','closed'),('clean','cleaned'),('wash','washed'),
    ('cook','cooked'),('print','printed'),('paint','painted'),('count','counted'),
]
ALL_PAST_IRR = [
    # train (irregular)
    ('feel','felt'),('run','ran'),('go','went'),('get','got'),
    ('say','said'),('make','made'),('take','took'),('see','saw'),
    # holdout (irregular)
    ('know','knew'),('come','came'),('give','gave'),('think','thought'),
    ('find','found'),('tell','told'),('keep','kept'),('leave','left'),
    ('stand','stood'),('lose','lost'),('hold','held'),('read','read'),
]

AXES_CONFIG = {
    'plural':  (ALL_PLURAL[:20], ALL_PLURAL[20:]),
    'gender':  (ALL_GENDER[:8],  ALL_GENDER[8:]),
    'comp':    (ALL_COMP[:20],   ALL_COMP[20:]),
    'sup':     (ALL_SUP[:20],    ALL_SUP[20:]),
    'past_reg':(ALL_PAST_REG[:16], ALL_PAST_REG[16:]),
    'past_irr':(ALL_PAST_IRR[:8],  ALL_PAST_IRR[8:]),
}

print("GENERALISATION RESULTS:")
print("%-12s  %5s  %6s  %5s  %6s  %6s  %6s  GEN?" % (
    "Axis", "coh", "s_tr", "tr%", "train", "hold", "ratio"))
print("-"*70)

all_results = {}
for name, (train_pairs, holdout_pairs) in AXES_CONFIG.items():
    ax, coh, valid_tr = compute_axis(train_pairs)
    if ax is None or len(valid_tr) < 3:
        print("  %-12s SKIP" % name); continue
    scale_tr, acc_tr = best_scale_on(ax, valid_tr)
    acc_hold, n_hold = eval_axis(ax, scale_tr, [v for v in
        [(*((get_emb(s)[0], get_emb(s)[1], get_emb(t)[0], get_emb(t)[1]) if get_emb(s)[0] is not None and get_emb(t)[0] is not None else (None,None,None,None)), s, t)
         for s,t in holdout_pairs] if v[0] is not None], fwd=True) if False else (0,0)
    # Re-compute holdout validity properly
    valid_hold = []
    for s, t in holdout_pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        valid_hold.append((s, t, sid, tid))
    acc_hold_c, n_hold = eval_axis(ax, scale_tr, valid_hold)
    n_tr = len(valid_tr)
    all_results[name] = (ax, coh, scale_tr, acc_tr, n_tr, acc_hold_c, n_hold, valid_tr, valid_hold)
    gen = "YES" if n_hold > 0 and acc_hold_c/n_hold >= 0.70 else "NO"
    print("  %-12s %.3f  %5.2f  %4.0f%%  %d/%d     %d/%d     %.2f  %s" % (
        name, coh, scale_tr, 100*acc_tr/max(1,n_tr), acc_tr, n_tr,
        acc_hold_c, n_hold, acc_hold_c/max(1,n_hold), gen))
print()

# ====================================================================
# DETAIL: show holdout pair results for best axes
# ====================================================================
for name in ['plural','gender','comp','sup']:
    if name not in all_results: continue
    ax, coh, scale_tr, acc_tr, n_tr, acc_hold_c, n_hold, valid_tr, valid_hold = all_results[name]
    print("HOLDOUT pairs for %-10s (axis trained on %d words, tested on %d):" % (
        name, n_tr, n_hold))
    for s, t, sid, tid in valid_hold:
        r = nn_retrieve(W_E[sid] + scale_tr * ax, [sid])
        got = r[0][0] if r else '?'
        print("  %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()

# ====================================================================
# CROSS-AXIS GENERALISATION: train on 5 pairs, test on 15
# ====================================================================
print("MINI-TRAIN GENERALISATION: train axis on just 5 pairs")
print("-"*65)
print("Testing: how few training pairs are needed for the axis to generalise?")
print()

for name, (train_pairs, holdout_pairs) in [
    ('plural', (ALL_PLURAL[:20], ALL_PLURAL[20:])),
    ('comp',   (ALL_COMP[:20],   ALL_COMP[20:])),
    ('gender', (ALL_GENDER[:8],  ALL_GENDER[8:])),
]:
    print("  %s:" % name)
    all_pairs = train_pairs + holdout_pairs
    np.random.seed(42)
    full_ax, _, full_valid = compute_axis(all_pairs)
    full_s, _ = best_scale_on(full_ax, full_valid) if full_ax is not None else (1.0, 0)
    for n_train in [2, 5, 10, len(train_pairs)]:
        if n_train > len(train_pairs): break
        mini_train = train_pairs[:n_train]
        mini_holdout = [p for p in all_pairs if p not in mini_train]
        ax_mini, _, valid_mini = compute_axis(mini_train)
        if ax_mini is None: continue
        # Evaluate at the FULL axis's optimal scale
        valid_hold = [(s,t,sid,tid) for s,t in mini_holdout
                      for es,sid in [get_emb(s)] for et,tid in [get_emb(t)]
                      if es is not None and et is not None]
        valid_hold = []
        for s, t in mini_holdout:
            es, sid = get_emb(s); et, tid = get_emb(t)
            if es is None or et is None: continue
            valid_hold.append((s, t, sid, tid))
        acc_h, n_h = eval_axis(ax_mini, full_s, valid_hold)
        coh_mini = float(np.mean([np.dot(normed(W_E[tid]-W_E[sid]).astype(np.float32),
                                         ax_mini.astype(np.float32))
                                  for s,t,sid,tid in valid_hold])) if valid_hold else 0
        print("    n_train=%2d  holdout_acc=%d/%d (%.0f%%)  coh_on_holdout=%.3f" % (
            n_train, acc_h, n_h, 100*acc_h/max(1,n_h), coh_mini))
    print()

# ====================================================================
# CROSS-DOMAIN TRANSFER: can the plural axis from common nouns
# generalise to completely different noun categories?
# ====================================================================
print("CROSS-DOMAIN TRANSFER:")
print("-"*65)
print("Axis trained on common nouns. Tested on domain-specific nouns.")
print()

ax_pl, coh_pl, valid_pl = compute_axis(ALL_PLURAL[:20])
spl, _ = best_scale_on(ax_pl, valid_pl)

DOMAIN_TESTS = {
    'body_parts': [('finger','fingers'),('shoulder','shoulders'),('knee','knees'),
                   ('elbow','elbows'),('cheek','cheeks'),('chin','chins')],
    'animals':    [('horse','horses'),('cow','cows'),('pig','pigs'),
                   ('wolf','wolves'),('fox','foxes'),('deer','deer')],
    'foods':      [('apple','apples'),('orange','oranges'),('grape','grapes'),
                   ('carrot','carrots'),('potato','potatoes'),('tomato','tomatoes')],
    'countries':  [('nation','nations'),('country','countries'),('state','states'),
                   ('region','regions'),('province','provinces')],
    'abstract':   [('idea','ideas'),('concept','concepts'),('theory','theories'),
                   ('problem','problems'),('solution','solutions'),('method','methods')],
    'verbs_as_nouns': [('dream','dreams'),('plan','plans'),('act','acts'),
                        ('form','forms'),('use','uses'),('work','works')],
}

for domain, test_pairs in DOMAIN_TESTS.items():
    valid_d = []
    for s, t in test_pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        valid_d.append((s, t, sid, tid))
    if not valid_d: continue
    acc, n = eval_axis(ax_pl, spl, valid_d)
    detail = []
    for s, t, sid, tid in valid_d:
        r = nn_retrieve(W_E[sid] + spl * ax_pl, [sid])
        got = r[0][0] if r else '?'
        detail.append('%s->%s[%s]' % (s, got, 'OK' if got==t else 'X'))
    print("  %-16s %d/%d (%.0f%%)  %s" % (domain, acc, n, 100*acc/max(1,n), '  '.join(detail)))
print()

# ====================================================================
# ZERO-SHOT: test on words that COULD NOT appear in training
# (very rare or technical words that are single-token in Qwen2)
# ====================================================================
print("ZERO-SHOT: rare/technical words (single-token, not in training):")
print("-"*65)

RARE_PAIRS = {
    'plural_rare': [
        ('fjord','fjords'),('glacier','glaciers'),('quasar','quasars'),
        ('photon','photons'),('neuron','neurons'),('enzyme','enzymes'),
        ('isotope','isotopes'),('comet','comets'),('asteroid','asteroids'),
        ('senator','senators'),('parliament','parliaments'),('diplomat','diplomats'),
    ],
    'comp_rare': [
        ('blunt','blunter'),('steep','steeper'),('slim','slimmer'),
        ('dense','denser'),('crude','cruder'),('bold','bolder'),
        ('mild','milder'),('vague','vaguer'),('stark','starker'),
    ],
}

for test_name, test_pairs in RARE_PAIRS.items():
    nm = test_name.split('_')[0]
    if nm == 'plural': ax_t, s_t = ax_pl, spl
    else:
        ax_t, _, vc = compute_axis(ALL_COMP[:20])
        s_t, _ = best_scale_on(ax_t, vc)
    valid_r = []
    for s, t in test_pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        valid_r.append((s, t, sid, tid))
    if not valid_r: continue
    acc, n = eval_axis(ax_t, s_t, valid_r)
    print("  %-16s %d/%d (%.0f%%)" % (test_name, acc, n, 100*acc/max(1,n)))
    for s, t, sid, tid in valid_r:
        r = nn_retrieve(W_E[sid] + s_t * ax_t, [sid])
        got = r[0][0] if r else '?'
        print("    %-14s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if got==t else '---'))
    print()

# ====================================================================
# SUMMARY
# ====================================================================
print("GENERALISATION SUMMARY:")
print("="*65)
for name, (ax, coh, scale_tr, acc_tr, n_tr, acc_hold_c, n_hold, valid_tr, valid_hold) in all_results.items():
    tr_pct = 100*acc_tr/max(1,n_tr); ho_pct = 100*acc_hold_c/max(1,n_hold)
    delta = ho_pct - tr_pct
    gen = "GENERALISES" if ho_pct >= 70 else "FAILS"
    print("  %-12s  train=%d/%d(%.0f%%)  hold=%d/%d(%.0f%%)  delta=%+.0f%%  %s" % (
        name, acc_tr, n_tr, tr_pct, acc_hold_c, n_hold, ho_pct, delta, gen))
print()
print("Prediction: all axes generalise (structure IS information).")
print("If generalisation holds, the axis IS the morphological rule,")
print("not just a memorisation of the training pairs.")
