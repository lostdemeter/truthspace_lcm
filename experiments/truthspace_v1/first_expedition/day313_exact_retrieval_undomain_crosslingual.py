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

def get_all_token_ids(word):
    """Return ALL single-token IDs for all prefix variants of word."""
    ids = set()
    for p in [' ', '', ' ' + word[0].upper() + word[1:],
              word[0].upper() + word[1:], word.upper(),
              ' ' + word.upper(), '-' + word, '_' + word, ' -' + word]:
        toks = tok(p, add_special_tokens=False)['input_ids']
        if len(toks) == 1: ids.add(toks[0])
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
    """Exclude ALL token variants of source word (space, no-space, caps, compound)."""
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    # Get all token IDs for source_word variants
    exclude = get_all_token_ids(source_word)
    for eid in exclude: sims[eid] = -1.0
    # Also apply clean filter (no caps, compounds, single chars)
    for i in range(len(sims)):
        w = tok.decode([i]).strip()
        if not w or len(w) <= 1: sims[i] = -1.0; continue
        if w[0].isupper(): sims[i] = -1.0; continue
        if w.startswith('-') or w.startswith('_'): sims[i] = -1.0; continue
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

def best_scale_exact(axis, pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = 0
        for _, t, sid, tid in pairs:
            es = W_E[sid]
            _, src_word = None, None
            r = nn_retrieve_exact(es + s*axis, tok.decode([sid]).strip())
            if tid is not None and r[0][0] == t: c += 1
        if c > best_acc: best_acc = c; best_s = s
    return best_s, best_acc

print("DAY 313: EXACT RETRIEVAL, un- SUBDOMAINS, CROSS-LINGUAL MAP, pc-IRREDUCIBILITY FIT")
print("="*72)
print()

# Build body-part axis
BODYPART_TRAIN = [('head','heads'),('foot','feet'),('ear','ears'),('knee','knees'),
                  ('toe','toes'),('lip','lips'),('hip','hips'),('rib','ribs'),
                  ('thumb','thumbs'),('wrist','wrists'),('elbow','elbows'),('heel','heels'),
                  ('shoulder','shoulders'),('chin','chins'),('neck','necks'),('jaw','jaws')]
ax_bp, _, valid_bp, pc_bp = compute_axis(BODYPART_TRAIN)

# ====================================================================
# PART A: nn_retrieve_exact — does it fix hand->hands?
# ====================================================================
print("PART A: nn_retrieve_exact — excluding ALL word variants")
print("-"*72)

scale_bp = 0.342
FULL_S_TEST = [
    ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
    ('flower','flowers'),('star','stars'),('cup','cups'),('door','doors'),
    ('fire','fires'),('train','trains'),('head','heads'),('foot','feet'),
]

print("  Testing body-part axis (scale=%.3f) with exact exclusion:" % scale_bp)
print("  %-10s -> %-10s  exact_nn      clean_nn      same?" % ("src", "tgt"))
hits_exact, hits_clean, total = 0, 0, 0
for src, tgt in FULL_S_TEST:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    total += 1
    pred = W_E[sid] + scale_bp * ax_bp
    r_exact = nn_retrieve_exact(pred, src, top_n=3)
    r_clean = nn_retrieve_clean(pred, [sid], top_n=3)
    hit_exact = (tid is not None and r_exact[0][0] == tgt)
    hit_clean = (tid is not None and r_clean[0][0] == tgt)
    if hit_exact: hits_exact += 1
    if hit_clean: hits_clean += 1
    same = (r_exact[0][0] == r_clean[0][0])
    diff_marker = '' if same else ' **DIFFER**'
    print("  %-10s -> %-10s  %-12s  %-12s  %s%s" %
          (src, tgt, r_exact[0][0], r_clean[0][0],
           'exact=✓' if hit_exact else 'exact=✗', diff_marker))
print()
print("  Exact:  %d/%d=%.0f%%   Clean:  %d/%d=%.0f%%" %
      (hits_exact, total, 100*hits_exact/total,
       hits_clean, total, 100*hits_clean/total))
print()

# Check what exact exclusion removes for 'hand' specifically
print("  All token IDs excluded for 'hand' by exact exclusion:")
hand_ids = get_all_token_ids('hand')
for tid in sorted(hand_ids):
    decoded = tok.decode([tid])
    print("    id=%6d  repr=%r" % (tid, decoded))
print()

# ====================================================================
# PART B: un- SUBDOMAIN SPLIT
# ====================================================================
print("PART B: un- subdomain split — adj vs verb vs state")
print("-"*72)

UN_ADJ   = [('happy','unhappy'),('kind','unkind'),('fair','unfair'),
             ('safe','unsafe'),('wise','unwise'),('true','untrue'),
             ('sure','unsure'),('clear','unclear'),('fit','unfit'),
             ('just','unjust'),('real','unreal'),('clean','unclean')]

UN_VERB  = [('lock','unlock'),('wrap','unwrap'),('tie','untie'),
             ('fold','unfold'),('pack','unpack'),('cover','uncover'),
             ('load','unload'),('zip','unzip'),('plug','unplug'),
             ('do','undo'),('make','unmake'),('dress','undress')]

UN_STATE = [('known','unknown'),('usual','unusual'),('likely','unlikely'),
             ('wanted','unwanted'),('expected','unexpected'),('planned','unplanned'),
             ('treated','untreated'),('finished','unfinished')]

for dom_name, un_pairs in [('un-ADJ', UN_ADJ), ('un-VERB', UN_VERB), ('un-STATE', UN_STATE)]:
    ax, _, valid, pc = compute_axis(un_pairs)
    if ax is None:
        print("  %s: not enough single-token pairs" % dom_name)
        continue
    # Quick holdout test: leave one out style
    hits = 0
    for i in range(len(valid)):
        test_s, test_t, test_sid, test_tid = valid[i]
        # Train on all others
        train_pairs = [(valid[j][0], valid[j][1]) for j in range(len(valid)) if j != i]
        ax_loo, _, vloo, _ = compute_axis(train_pairs)
        if ax_loo is None: continue
        # Find best scale on training set
        best_s, _ = 0.5, 0
        best_hits = 0
        for s_test in np.linspace(0.02, 4.0, 80):
            c = sum(1 for _,t,s,_ in vloo
                    if nn_retrieve_exact(W_E[s]+s_test*ax_loo, tok.decode([s]).strip(), 1)[0][0]==t)
            if c > best_hits: best_hits=c; best_s=s_test
        pred = W_E[test_sid] + best_s * ax_loo
        r = nn_retrieve_exact(pred, test_s, top_n=1)
        if r[0][0] == test_t: hits += 1
    print("  %s: n=%d valid  pc=%.4f  LOO accuracy=%d/%d=%.0f%%" %
          (dom_name, len(valid), pc, hits, len(valid), 100*hits/len(valid) if valid else 0))
    # Show per-pair retrieval at best full-set scale
    if ax is not None and valid:
        best_s_all, best_acc = 0.5, 0
        for s_test in np.linspace(0.02, 4.0, 80):
            c = sum(1 for _,t,s,_ in valid
                    if nn_retrieve_exact(W_E[s]+s_test*ax, tok.decode([s]).strip(), 1)[0][0]==t)
            if c > best_acc: best_acc=c; best_s_all=s_test
        print("  %s in-sample: scale=%.3f  acc=%d/%d" % (dom_name, best_s_all, best_acc, len(valid)))
    print()

# Cross-domain: train on un-ADJ, test on un-VERB, and vice versa
print("  Cross-domain transfer for un-:")
ax_adj, _, valid_adj, pc_adj = compute_axis(UN_ADJ)
ax_verb, _, valid_verb, pc_verb = compute_axis(UN_VERB)
ax_state, _, valid_state, pc_state = compute_axis(UN_STATE)

if ax_adj is not None and ax_verb is not None:
    print("  cos(adj_axis, verb_axis) = %.4f" % float(np.dot(ax_adj, ax_verb)))
if ax_adj is not None and ax_state is not None:
    print("  cos(adj_axis, state_axis) = %.4f" % float(np.dot(ax_adj, ax_state)))
if ax_verb is not None and ax_state is not None:
    print("  cos(verb_axis, state_axis) = %.4f" % float(np.dot(ax_verb, ax_state)))

if ax_adj is not None and valid_verb:
    best_s, best_hits = 0.5, 0
    for s_test in np.linspace(0.02, 4.0, 80):
        c = sum(1 for _,t,s,_ in valid_verb
                if nn_retrieve_exact(W_E[s]+s_test*ax_adj, tok.decode([s]).strip(), 1)[0][0]==t)
        if c > best_hits: best_hits=c; best_s=s_test
    print("  adj_axis -> verb holdout: %d/%d=%.0f%% (scale=%.3f)" %
          (best_hits, len(valid_verb), 100*best_hits/len(valid_verb), best_s))

if ax_verb is not None and valid_adj:
    best_s, best_hits = 0.5, 0
    for s_test in np.linspace(0.02, 4.0, 80):
        c = sum(1 for _,t,s,_ in valid_adj
                if nn_retrieve_exact(W_E[s]+s_test*ax_verb, tok.decode([s]).strip(), 1)[0][0]==t)
        if c > best_hits: best_hits=c; best_s=s_test
    print("  verb_axis -> adj holdout: %d/%d=%.0f%% (scale=%.3f)" %
          (best_hits, len(valid_adj), 100*best_hits/len(valid_adj), best_s))
print()

# ====================================================================
# PART C: CROSS-LINGUAL INTERFERENCE MAP
# ====================================================================
print("PART C: Cross-lingual interference map")
print("-"*72)
# For a range of common words, check if the top clean NN of their near-plural
# axis endpoint is a cross-lingual token

# Use object-noun axis
ax_obj, _, valid_obj, _ = compute_axis([('cat','cats'),('dog','dogs'),('house','houses'),
                                          ('car','cars'),('tree','trees'),('book','books'),
                                          ('bird','birds'),('ship','ships')])
scale_obj = 0.181

TEST_WORDS = [
    # Body parts
    'hand','eye','arm','leg','head','foot','ear','nose','cheek','knee','lip','neck',
    # Objects
    'cup','door','car','tree','book','ship','star','flower','wall','room',
    # Animals
    'cat','dog','bird','wolf','bear','fish','deer',
    # Abstract
    'fire','train','road','forest','boat',
]

print("  Word       after_obj_axis_top1   xling?   xling_token")
xling_count = 0
total_words = 0
for word in TEST_WORDS:
    es, sid = get_emb(word)
    if es is None: continue
    total_words += 1
    pred = W_E[sid] + scale_obj * ax_obj
    r = nn_retrieve_exact(pred, word, top_n=5)
    # Check if top-1 is cross-lingual
    top_word = r[0][0]
    is_xling = False
    xling_tok = ''
    # Check if the top-1 token contains non-ASCII characters
    for result_word, _, _ in r[:3]:
        try:
            result_word.encode('ascii')
        except UnicodeEncodeError:
            if result_word != top_word:
                is_xling = True
                xling_tok = result_word
                break
    # Is the top-1 itself cross-lingual?
    top_is_xling = False
    try:
        top_word.encode('ascii')
    except UnicodeEncodeError:
        top_is_xling = True
        xling_count += 1
    top3 = r[0][0]
    print("  %-10s %-22s %-8s %s" %
          (word, top3, 'TOP_XLING' if top_is_xling else ('xling_near' if is_xling else 'ok'),
           xling_tok if is_xling and not top_is_xling else ''))

print()
print("  Cross-lingual TOP-1 count: %d/%d = %.0f%%" % (xling_count, total_words, 100*xling_count/total_words))

# Specifically check body-part words
print()
print("  Body-part cross-lingual proximity (distance from word to Chinese equivalent):")
body_xling_pairs = [('hand','手'),('eye','眼睛'),('arm','手臂'),('leg','腿'),
                    ('head','头'),('foot','脚'),('ear','耳'),('nose','鼻'),
                    ('cup','杯'),('car','车'),('tree','树'),('book','书')]
for eng, chi in body_xling_pairs:
    es, sid = get_emb(eng)
    ec, cid = get_emb(chi)
    if es is None or ec is None: continue
    cos_xling = float(np.dot(normed(es).astype(np.float32), normed(ec).astype(np.float32)))
    print("  %-8s / %-6s  cos=%.4f" % (eng, chi, cos_xling))
print()

# ====================================================================
# PART D: pc vs IRREDUCIBILITY LINEAR FIT
# ====================================================================
print("PART D: pc vs irreducibility — linear relationship")
print("-"*72)

ALL_AXES_TEST = {
    '+er':     ([('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                  ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                  ('light','lighter'),('strong','stronger'),('weak','weaker'),('soft','softer'),
                  ('hard','harder'),('sharp','sharper'),('warm','warmer'),('cool','cooler')],
                 [('kind','kinder'),('young','younger'),('old','older'),('new','newer'),
                  ('long','longer'),('high','higher'),('thick','thicker'),('thin','thinner')]),
    '+est':    ([('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                  ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest'),
                  ('hard','hardest'),('warm','warmest'),('cool','coolest'),('sweet','sweetest')],
                 [('kind','kindest'),('young','youngest'),('old','oldest'),('long','longest')]),
    'gender':  ([('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                  ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
                 [('grandfather','grandmother'),('nephew','niece'),('groom','bride'),
                  ('actor','actress'),('waiter','waitress'),('host','hostess')]),
    'past_irr': ([('go','went'),('come','came'),('run','ran'),('see','saw'),
                   ('eat','ate'),('know','knew'),('take','took'),('make','made'),
                   ('give','gave'),('find','found'),('buy','bought'),('bring','brought')],
                  [('say','said'),('get','got'),('do','did'),('think','thought'),
                   ('speak','spoke'),('ride','rode'),('write','wrote'),('grow','grew')]),
    '+ed':     ([('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                  ('end','ended'),('look','looked'),('call','called'),('help','helped')],
                 [('play','played'),('work','worked'),('turn','turned'),('push','pushed'),
                  ('pull','pulled'),('open','opened'),('close','closed'),('move','moved')]),
    '+ness':   ([('sad','sadness'),('happy','happiness'),('dark','darkness'),('kind','kindness'),
                  ('bright','brightness'),('mad','madness'),('sick','sickness'),('weak','weakness')],
                 [('bold','boldness'),('cold','coldness'),('soft','softness'),('hard','hardness')]),
    '+ful':    ([('hope','hopeful'),('care','careful'),('use','useful'),('power','powerful'),
                  ('peace','peaceful'),('harm','harmful'),('thank','thankful'),('help','helpful')],
                 [('play','playful'),('wonder','wonderful'),('color','colorful'),('grace','graceful')]),
    'un-':     ([('happy','unhappy'),('kind','unkind'),('fair','unfair'),('known','unknown'),
                  ('usual','unusual'),('clear','unclear'),('lock','unlock'),('wrap','unwrap')],
                 [('tie','untie'),('fold','unfold'),('pack','unpack'),('cover','uncover'),
                  ('safe','unsafe'),('wise','unwise'),('true','untrue')]),
    '+ment':   ([('achieve','achievement'),('manage','management'),('develop','development'),
                  ('move','movement'),('treat','treatment'),('argue','argument'),
                  ('judge','judgment'),('employ','employment')],
                 [('invest','investment'),('punish','punishment'),('amuse','amusement'),
                  ('amaze','amazement'),('excite','excitement'),('refresh','refreshment')]),
    '+s':      ([('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                  ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
                 [('flower','flowers'),('star','stars'),('forest','forests'),('boat','boats'),
                  ('cup','cups'),('door','doors'),('road','roads'),('wall','walls'),
                  ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
                  ('fire','fires'),('train','trains'),('room','rooms')]),
    '+tion':   ([('act','action'),('direct','direction'),('collect','collection'),
                  ('connect','connection'),('protect','protection'),('select','selection')],
                 [('inject','injection'),('reject','rejection'),('detect','detection'),
                  ('infect','infection'),('inspect','inspection'),('correct','correction'),
                  ('observe','observation'),('describe','description'),('produce','production')]),
}

print("  %-12s  pc      n_test  irred_clean  irred_exact  frac_clean  frac_exact" % "axis")
print("  " + "-"*74)
pc_vals, irred_clean_fracs, irred_exact_fracs = [], [], []

for nm, (train_pairs, test_pairs) in ALL_AXES_TEST.items():
    ax, _, valid, pc = compute_axis(train_pairs)
    if ax is None: continue
    # Find best clean scale
    best_s_clean = 0.5
    best_acc = 0
    for s_test in np.linspace(0.02, 6.0, 100):
        c = sum(1 for _,t,s,_ in valid
                if nn_retrieve_clean(W_E[s]+s_test*ax, [s], 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s_clean=s_test

    irred_clean, irred_exact = [], []
    total = 0
    for src, tgt in test_pairs:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        total += 1
        found_clean, found_exact = False, False
        for s_test in np.linspace(0.02, 6.0, 150):
            pred = W_E[sid] + s_test * ax
            if not found_clean:
                r = nn_retrieve_clean(pred, [sid], 1)
                if tid is not None and r[0][0] == tgt: found_clean = True
            if not found_exact:
                r = nn_retrieve_exact(pred, src, 1)
                if tid is not None and r[0][0] == tgt: found_exact = True
            if found_clean and found_exact: break
        if not found_clean: irred_clean.append(src)
        if not found_exact: irred_exact.append(src)

    fc = len(irred_clean)/total if total > 0 else 0.0
    fe = len(irred_exact)/total if total > 0 else 0.0
    pc_vals.append(pc); irred_clean_fracs.append(fc); irred_exact_fracs.append(fe)
    print("  %-12s  %.4f  %d      %d           %d           %.0f%%        %.0f%%" %
          (nm, pc, total, len(irred_clean), len(irred_exact), 100*fc, 100*fe))

print()
# Linear fit: pc vs irreducibility
if len(pc_vals) >= 3:
    x = np.array(pc_vals)
    y_c = np.array(irred_clean_fracs)
    y_e = np.array(irred_exact_fracs)
    # Pearson correlation
    r_c = float(np.corrcoef(x, y_c)[0,1])
    r_e = float(np.corrcoef(x, y_e)[0,1])
    # Linear regression
    slope_c = float(np.polyfit(x, y_c, 1)[0])
    slope_e = float(np.polyfit(x, y_e, 1)[0])
    print("  pc vs irred_clean:  r=%.4f  slope=%.3f (per unit pc)" % (r_c, slope_c))
    print("  pc vs irred_exact:  r=%.4f  slope=%.3f (per unit pc)" % (r_e, slope_e))
    print()
    print("  Interpretation:")
    print("  For every +0.1 increase in pc, irreducibility changes by %.0f%% (clean)" %
          (100*slope_c*0.1))
    print("  For every +0.1 increase in pc, irreducibility changes by %.0f%% (exact)" %
          (100*slope_e*0.1))
    print()
    # Predicted irreducibility at pc thresholds
    intercept_c = float(np.polyfit(x, y_c, 1)[1])
    for pc_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pred_irred = slope_c * pc_thresh + intercept_c
        pred_irred = max(0, min(1, pred_irred))
        print("  Predicted irred at pc=%.1f: %.0f%% (clean)" % (pc_thresh, 100*pred_irred))
