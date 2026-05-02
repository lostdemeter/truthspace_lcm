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

def get_all_source_ids(word):
    """Return ALL single-token IDs for all prefix variants of word (FIXED)."""
    ids = set()
    for p in [' ' + word,           # space+lowercase (PRIMARY)
              word,                  # no-space lowercase (SECONDARY)
              ' ' + word[0].upper() + word[1:],  # space+capitalized
              word[0].upper() + word[1:],         # capitalized
              word.upper(),          # all-caps
              ' ' + word.upper(),    # space+all-caps
              '-' + word,            # compound prefix -
              '_' + word,            # compound prefix _
              ' -' + word,           # space+compound
              ' ',                   # space alone
              ]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

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

def nn_retrieve_clean(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    for i in range(len(sims)):
        w = tok.decode([i]).strip()
        if not w or len(w) <= 1: sims[i] = -1.0; continue
        if w[0].isupper(): sims[i] = -1.0; continue
        if w.startswith('-') or w.startswith('_'): sims[i] = -1.0; continue
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

def nn_retrieve_exact(pred_emb, source_word, top_n=5):
    """FIXED: exclude ALL token variants of source word."""
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    exclude = get_all_source_ids(source_word)
    for eid in exclude: sims[eid] = -1.0
    for i in range(len(sims)):
        w = tok.decode([i]).strip()
        if not w or len(w) <= 1: sims[i] = -1.0; continue
        if w[0].isupper(): sims[i] = -1.0; continue
        if w.startswith('-') or w.startswith('_'): sims[i] = -1.0; continue
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

# Body-part axis
BODYPART_TRAIN = [('head','heads'),('foot','feet'),('ear','ears'),('knee','knees'),
                  ('toe','toes'),('lip','lips'),('hip','hips'),('rib','ribs'),
                  ('thumb','thumbs'),('wrist','wrists'),('elbow','elbows'),('heel','heels'),
                  ('shoulder','shoulders'),('chin','chins'),('neck','necks'),('jaw','jaws')]
ax_bp, _, valid_bp, pc_bp = compute_axis(BODYPART_TRAIN)
scale_bp = 0.342

print("DAY 314: EXACT RETRIEVAL FIX, hand->hands, +tion DOMAINS, pc PREDICTION")
print("="*70)
print()

# ====================================================================
# PART A: FIXED nn_retrieve_exact — verify source exclusion
# ====================================================================
print("PART A: Fixed nn_retrieve_exact — source exclusion verification")
print("-"*70)

print("  All token IDs excluded for 'hand' (FIXED):")
hand_ids = get_all_source_ids('hand')
for tid in sorted(hand_ids):
    decoded = repr(tok.decode([tid]))
    print("    id=%6d  repr=%s" % (tid, decoded))
print()

print("  All token IDs excluded for 'eye' (FIXED):")
eye_ids = get_all_source_ids('eye')
for tid in sorted(eye_ids):
    decoded = repr(tok.decode([tid]))
    print("    id=%6d  repr=%s" % (tid, decoded))
print()

# Now test with fixed exact retrieval
FULL_TEST = [
    ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
    ('flower','flowers'),('star','stars'),('cup','cups'),('door','doors'),
    ('fire','fires'),('train','trains'),('head','heads'),('foot','feet'),
    ('ear','ears'),('knee','knees'),('tooth','teeth'),('wolf','wolves'),
    ('deer','deer'),('fish','fish'),
]

print("  Body-part axis (scale=%.3f) with FIXED exact exclusion:" % scale_bp)
print("  %-10s -> %-10s  exact_nn      clean_nn      diff?" % ("src", "tgt"))
hits_exact, hits_clean = 0, 0
for src, tgt in FULL_TEST:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    pred = W_E[sid] + scale_bp * ax_bp
    r_exact = nn_retrieve_exact(pred, src, top_n=3)
    r_clean = nn_retrieve_clean(pred, [sid], top_n=3)
    hit_e = (tid is not None and r_exact[0][0] == tgt)
    hit_c = (tid is not None and r_clean[0][0] == tgt)
    if hit_e: hits_exact += 1
    if hit_c: hits_clean += 1
    differ = ' *' if r_exact[0][0] != r_clean[0][0] else ''
    print("  %-10s -> %-10s  %-12s  %-12s  %s%s" %
          (src, tgt, r_exact[0][0], r_clean[0][0],
           '✓' if hit_e else '✗', differ))
print()
n = len(FULL_TEST)
print("  Exact (fixed): %d/≤%d   Clean: %d/≤%d" % (hits_exact, n, hits_clean, n))
print()

# Specifically test hand at multiple scales with exact exclusion
print("  hand->hands path with FIXED exact exclusion:")
print("  scale  exact_top1          sim     clean_top1          sim")
es_hand, sid_hand = get_emb('hand')
for s in [0.1, 0.2, 0.3, 0.342, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    pred = W_E[sid_hand] + s * ax_bp
    r_e = nn_retrieve_exact(pred, 'hand', 1)
    r_c = nn_retrieve_clean(pred, [sid_hand], 1)
    print("  %.3f  %-20s %.4f  %-20s %.4f" %
          (s, r_e[0][0], r_e[0][1], r_c[0][0], r_c[0][1]))
print()

# ====================================================================
# PART B: +tion DOMAIN SPLIT
# ====================================================================
print("PART B: +tion domain analysis — why does it have 22% irreducibility?")
print("-"*70)

TION_CT_SIMPLE = [
    ('act','action'),('direct','direction'),('collect','collection'),
    ('connect','connection'),('protect','protection'),('select','selection'),
    ('inject','injection'),('reject','rejection'),('detect','detection'),
    ('infect','infection'),('inspect','inspection'),('correct','correction'),
]
TION_OBSERVE = [
    ('observe','observation'),('describe','description'),('produce','production'),
    ('resolve','resolution'),('evolve','evolution'),('revolve','revolution'),
    ('solve','solution'),('dissolve','dissolution'),
]
TION_ATE_VERB = [
    ('communicate','communication'),('investigate','investigation'),
    ('appreciate','appreciation'),('evaluate','evaluation'),
    ('participate','participation'),('generate','generation'),
    ('create','creation'),('educate','education'),
    ('indicate','indication'),('locate','location'),
]

for dom_name, pairs in [
    ('-ct simple', TION_CT_SIMPLE),
    ('-serve/-scribe', TION_OBSERVE),
    ('-ate verbs', TION_ATE_VERB),
]:
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None:
        print("  %s: insufficient single-token pairs" % dom_name)
        continue
    # Best scale
    best_s, best_acc = 0.5, 0
    for s_test in np.linspace(0.02, 6.0, 100):
        c = sum(1 for _,t,s,_ in valid
                if nn_retrieve_exact(W_E[s]+s_test*ax, tok.decode([s]).strip().lstrip(), 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s_test
    print("  %s: n=%d  pc=%.4f  in-sample=%d/%d=%.0f%%  scale=%.3f" %
          (dom_name, len(valid), pc, best_acc, len(valid),
           100*best_acc/len(valid) if valid else 0, best_s))
    # Cross-axis tests
    print()

# Cross-domain transfer: ct_simple axis on observe domain
ax_ct, _, valid_ct, pc_ct = compute_axis(TION_CT_SIMPLE)
ax_obs, _, valid_obs, pc_obs = compute_axis(TION_OBSERVE)
ax_ate, _, valid_ate, pc_ate = compute_axis(TION_ATE_VERB)

if ax_ct is not None and ax_obs is not None:
    print("  cos(ct_axis, observe_axis) = %.4f" % float(np.dot(ax_ct, ax_obs)))
if ax_ct is not None and ax_ate is not None:
    print("  cos(ct_axis, ate_axis) = %.4f" % float(np.dot(ax_ct, ax_ate)))
if ax_obs is not None and ax_ate is not None:
    print("  cos(obs_axis, ate_axis) = %.4f" % float(np.dot(ax_obs, ax_ate)))
print()

# Cross-domain: ct_axis on observe pairs
if ax_ct is not None and valid_obs:
    best_s, best_hits = 0.5, 0
    for s_test in np.linspace(0.02, 4.0, 80):
        c = sum(1 for _,t,s,_ in valid_obs
                if nn_retrieve_exact(W_E[s]+s_test*ax_ct, tok.decode([s]).strip().lstrip(), 1)[0][0]==t)
        if c > best_hits: best_hits=c; best_s=s_test
    print("  ct_axis -> observe holdout: %d/%d=%.0f%%" %
          (best_hits, len(valid_obs), 100*best_hits/len(valid_obs)))

if ax_ct is not None and valid_ate:
    best_s, best_hits = 0.5, 0
    for s_test in np.linspace(0.02, 4.0, 80):
        c = sum(1 for _,t,s,_ in valid_ate
                if nn_retrieve_exact(W_E[s]+s_test*ax_ct, tok.decode([s]).strip().lstrip(), 1)[0][0]==t)
        if c > best_hits: best_hits=c; best_s=s_test
    print("  ct_axis -> ate holdout: %d/%d=%.0f%%" %
          (best_hits, len(valid_ate), 100*best_hits/len(valid_ate)))

if ax_ate is not None and valid_ct:
    best_s, best_hits = 0.5, 0
    for s_test in np.linspace(0.02, 4.0, 80):
        c = sum(1 for _,t,s,_ in valid_ct
                if nn_retrieve_exact(W_E[s]+s_test*ax_ate, tok.decode([s]).strip().lstrip(), 1)[0][0]==t)
        if c > best_hits: best_hits=c; best_s=s_test
    print("  ate_axis -> ct holdout: %d/%d=%.0f%%" %
          (best_hits, len(valid_ct), 100*best_hits/len(valid_ct)))
print()

# Show irreducible words per domain
print("  Irreducible words under ct_axis (all three domains as holdout):")
all_tion_holdout = TION_OBSERVE + TION_ATE_VERB
if ax_ct is not None:
    for src, tgt in all_tion_holdout:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        found = False
        for s_test in np.linspace(0.02, 6.0, 150):
            pred = W_E[sid] + s_test * ax_ct
            r = nn_retrieve_exact(pred, src, 1)
            if tid is not None and r[0][0] == tgt: found = True; break
        if not found:
            pred = W_E[sid] + 0.5 * ax_ct
            r = nn_retrieve_exact(pred, src, 3)
            print("  IRRED: %-16s -> %-16s  got: %s" %
                  (src, tgt, ', '.join(w for w,_,_ in r)))
print()

# ====================================================================
# PART C: pc THRESHOLD PREDICTION TEST
# ====================================================================
print("PART C: pc threshold prediction — new axes and prediction accuracy")
print("-"*70)

LINEAR_SLOPE  = -1.601
LINEAR_INTCPT =  0.74

def predict_irreducibility(pc_val):
    return max(0.0, min(1.0, LINEAR_SLOPE * pc_val + LINEAR_INTCPT))

# New axes to test
NEW_AXES = {
    '+ly':    ([('quick','quickly'),('slow','slowly'),('clear','clearly'),
                 ('near','nearly'),('deep','deeply'),('short','shortly'),
                 ('hard','hardly'),('bright','brightly'),('dark','darkly'),
                 ('sharp','sharply'),('calm','calmly'),('wide','widely')],
                [('fair','fairly'),('clean','cleanly'),('warm','warmly'),
                 ('bold','boldly'),('soft','softly'),('firm','firmly')]),
    '+ing':   ([('run','running'),('jump','jumping'),('walk','walking'),
                 ('talk','talking'),('play','playing'),('start','starting'),
                 ('help','helping'),('work','working'),('turn','turning'),
                 ('push','pushing'),('call','calling'),('look','looking')],
                [('move','moving'),('pull','pulling'),('end','ending'),
                 ('open','opening'),('close','closing'),('think','thinking')]),
    '+er_noun':([('teach','teacher'),('farm','farmer'),('build','builder'),
                  ('lead','leader'),('read','reader'),('write','writer'),
                  ('fight','fighter'),('drive','driver'),('hunt','hunter'),
                  ('manage','manager'),('speak','speaker'),('train','trainer')],
                 [('work','worker'),('play','player'),('own','owner'),
                  ('follow','follower'),('report','reporter'),('sing','singer')]),
    'capital': ([('france','Paris'),('germany','Berlin'),('japan','Tokyo'),
                  ('china','Beijing'),('india','Delhi'),('egypt','Cairo'),
                  ('spain','Madrid'),('brazil','Brasilia'),('canada','Ottawa'),
                  ('russia','Moscow'),('mexico','Mexico'),('italy','Rome')],
                 [('australia','Canberra'),('peru','Lima'),('chile','Santiago'),
                  ('sweden','Stockholm'),('portugal','Lisbon'),('greece','Athens')]),
}

print("  Axis         pc      pred_irred  actual_irred  diff")
print("  " + "-"*56)
for nm, (train_pairs, test_pairs) in NEW_AXES.items():
    ax, _, valid, pc = compute_axis(train_pairs)
    if ax is None: continue
    pred_irred = predict_irreducibility(pc)
    # Find best scale
    best_s, best_acc = 0.5, 0
    for s_test in np.linspace(0.02, 6.0, 100):
        c = sum(1 for _,t,s,_ in valid
                if nn_retrieve_exact(W_E[s]+s_test*ax, tok.decode([s]).strip().lstrip(), 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s_test
    # Holdout irreducibility
    irred_count = 0; total_test = 0
    for src, tgt in test_pairs:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        total_test += 1
        found = False
        for s_test in np.linspace(0.02, 6.0, 150):
            pred = W_E[sid] + s_test * ax
            r = nn_retrieve_exact(pred, src, 1)
            if tid is not None and r[0][0] == tgt: found = True; break
        if not found: irred_count += 1
    actual_irred = irred_count/total_test if total_test > 0 else 0.0
    diff = actual_irred - pred_irred
    print("  %-12s  %.4f  %.0f%%          %d/%d=%.0f%%        %+.0f%%" %
          (nm, pc, 100*pred_irred, irred_count, total_test,
           100*actual_irred, 100*diff))
    print("         in-sample=%d/%d scale=%.3f" % (best_acc, len(valid), best_s))
print()

# ====================================================================
# PART D: FULL AXIS QUALITY TABLE WITH FIXED EXACT
# ====================================================================
print("PART D: Complete axis quality table with fixed exact retrieval")
print("-"*70)

ALL_AXES = {
    '+er':     ([('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                  ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                  ('light','lighter'),('strong','stronger'),('weak','weaker'),('soft','softer'),
                  ('hard','harder'),('sharp','sharper'),('warm','warmer'),('cool','cooler')],
                 [('kind','kinder'),('young','younger'),('old','older'),('new','newer'),
                  ('long','longer'),('high','higher'),('thick','thicker'),('thin','thinner')]),
    '+s':      ([('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                  ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
                 [('flower','flowers'),('star','stars'),('boat','boats'),('cup','cups'),
                  ('door','doors'),('road','roads'),('wall','walls'),('arm','arms'),
                  ('leg','legs'),('fire','fires'),('train','trains'),('room','rooms'),
                  ('hand','hands'),('eye','eyes'),('head','heads'),('foot','feet')]),
    'past_irr':([('go','went'),('come','came'),('run','ran'),('see','saw'),
                  ('eat','ate'),('know','knew'),('take','took'),('make','made'),
                  ('give','gave'),('find','found'),('buy','bought'),('bring','brought')],
                 [('say','said'),('get','got'),('do','did'),('think','thought'),
                  ('speak','spoke'),('ride','rode'),('write','wrote'),('grow','grew')]),
    '+tion_ct':([('act','action'),('direct','direction'),('collect','collection'),
                  ('connect','connection'),('protect','protection'),('select','selection'),
                  ('inject','injection'),('reject','rejection'),('detect','detection'),
                  ('infect','infection'),('inspect','inspection'),('correct','correction')],
                 [('observe','observation'),('describe','description'),('produce','production'),
                  ('create','creation'),('generate','generation'),('educate','education')]),
}

print("  %-12s  pc    train  train%%  holdout  hold%%  irred%%  pred_irred%%" % "axis")
print("  " + "-"*70)
for nm, (train_pairs, test_pairs) in ALL_AXES.items():
    ax, _, valid, pc = compute_axis(train_pairs)
    if ax is None: continue
    # In-sample
    best_s, tr_acc = 0.5, 0
    for s_test in np.linspace(0.02, 6.0, 100):
        c = sum(1 for _,t,s,_ in valid
                if nn_retrieve_exact(W_E[s]+s_test*ax, tok.decode([s]).strip().lstrip(), 1)[0][0]==t)
        if c > tr_acc: tr_acc=c; best_s=s_test
    # Holdout
    ho_hits, irred_count, ho_total = 0, 0, 0
    for src, tgt in test_pairs:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        ho_total += 1
        found = False
        for s_test in np.linspace(0.02, 6.0, 150):
            pred = W_E[sid] + s_test * ax
            r = nn_retrieve_exact(pred, src, 1)
            if tid is not None and r[0][0] == tgt: found = True; break
        if found: ho_hits += 1
        else: irred_count += 1
    actual_irred = irred_count/ho_total if ho_total > 0 else 0.0
    pred_irred = predict_irreducibility(pc)
    print("  %-12s  %.3f  %d     %.0f%%     %d/%d     %.0f%%    %.0f%%    %.0f%%" %
          (nm, pc, len(valid), 100*tr_acc/len(valid) if valid else 0,
           ho_hits, ho_total, 100*ho_hits/ho_total if ho_total else 0,
           100*actual_irred, 100*pred_irred))
