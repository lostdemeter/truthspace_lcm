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
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
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
def best_scale_clean(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve_clean(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

# Build body-part axis
BODYPART_TRAIN = [
    ('head','heads'),('foot','feet'),('ear','ears'),('knee','knees'),
    ('toe','toes'),('lip','lips'),('hip','hips'),('rib','ribs'),
    ('thumb','thumbs'),('wrist','wrists'),('elbow','elbows'),('heel','heels'),
    ('shoulder','shoulders'),('chin','chins'),('neck','necks'),('jaw','jaws'),
]
ax_bp, _, valid_bp, pc_bp = compute_axis(BODYPART_TRAIN)

print("DAY 312: HAND PATH ANALYSIS, IRREDUCIBLE MAP, IRREGULAR PLURALS, gPC5")
print("="*70)
print()

# ====================================================================
# PART A: hand->hands PATH ANALYSIS
# ====================================================================
print("PART A: hand->hands — full path analysis along body-part axis")
print("-"*70)
es_hand, sid_hand = get_emb('hand')
_, tid_hands = get_emb('hands')
scale_bp, _ = best_scale_clean(ax_bp.astype(np.float32), valid_bp)
print("  Body-part axis scale: %.4f   pc: %.4f" % (scale_bp, pc_bp))
print()

print("  Path from 'hand' along body-part axis:")
print("  scale   top-1 clean         sim     rank_of_hands  sim_hands  sim_hand")
for s in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
    pred = W_E[sid_hand] + s * ax_bp
    pred_n = normed(pred).astype(np.float32)
    sims = W_n @ pred_n
    # Top clean neighbor
    sims_c = sims.copy()
    sims_c[sid_hand] = -1.0
    for i in range(len(sims_c)):
        w = tok.decode([i]).strip()
        if not w or len(w) <= 1: sims_c[i] = -1.0; continue
        if w[0].isupper(): sims_c[i] = -1.0; continue
        if w.startswith('-') or w.startswith('_'): sims_c[i] = -1.0; continue
    top_clean = np.argsort(sims_c)[-1]
    top_word = tok.decode([top_clean]).strip()
    top_sim = float(sims_c[top_clean])
    # Rank of 'hands' in clean search
    if tid_hands is not None:
        sim_hands = float(sims[tid_hands])
        # Count how many clean tokens score higher
        clean_mask = np.zeros(len(sims_c), dtype=bool)
        for i in range(len(sims_c)):
            if sims_c[i] > sim_hands: clean_mask[i] = True
        rank_hands = int(np.sum(clean_mask)) + 1
    else:
        sim_hands, rank_hands = 0.0, -1
    sim_hand = float(sims[sid_hand])
    print("  %.3f   %-20s %.4f  rank=%4d  sim_hands=%.4f  sim_hand=%.4f" %
          (s, top_word, top_sim, rank_hands, sim_hands, sim_hand))

print()
# What IS at scale=0.5 and 1.0 when we don't exclude 'hand'?
print("  Top-5 tokens (including self) at scale=0.5:")
pred = W_E[sid_hand] + 0.5 * ax_bp
r = nn_retrieve(pred, [], top_n=10)
for w, sim, i in r[:10]:
    print("    %-20s  %.4f" % (w, sim))
print()

# ====================================================================
# PART B: IRREDUCIBLE WORDS ACROSS ALL 12 AXES
# ====================================================================
print("PART B: Irreducible words — fraction that can NEVER be retrieved")
print("-"*70)

ALL_AXES_DATA = {
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

print("  %-12s  n_train  n_test  optscale  irreducible  frac_irred" % "axis")
print("  " + "-"*62)
for nm, (train_pairs, test_pairs) in ALL_AXES_DATA.items():
    ax, _, valid, pc = compute_axis(train_pairs)
    if ax is None: continue
    scale, tr_acc = best_scale_clean(ax.astype(np.float32), valid)
    irreducible = []
    total_test = 0
    for src, tgt in test_pairs:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        total_test += 1
        found = False
        for s_test in np.linspace(0.02, 6.0, 150):
            pred = W_E[sid] + s_test * ax
            r = nn_retrieve_clean(pred, [sid], top_n=1)
            if tid is not None and r[0][0] == tgt:
                found = True; break
        if not found:
            irreducible.append(src)
    frac = len(irreducible) / total_test if total_test > 0 else 0.0
    irred_str = ','.join(irreducible[:4]) if irreducible else 'none'
    print("  %-12s  %d       %d      %.3f     %-30s  %.1f%%" %
          (nm, len(valid), total_test, scale, irred_str, 100*frac))

print()

# ====================================================================
# PART C: CROSS-SEMANTIC IRREGULAR PLURALS
# ====================================================================
print("PART C: Irregular plurals — does body-part axis generalize?")
print("-"*70)

IRREGULAR_PLURALS = [
    # Body parts
    ('foot','feet'),('tooth','teeth'),('goose','geese'),
    # Non-body-part irregular
    ('mouse','mice'),('louse','lice'),('man','men'),('woman','women'),
    ('child','children'),('ox','oxen'),('leaf','leaves'),('knife','knives'),
    ('life','lives'),('wolf','wolves'),('calf','calves'),('half','halves'),
    # Unchanged plurals
    ('sheep','sheep'),('deer','deer'),('fish','fish'),('moose','moose'),
]

print("  Testing body-part axis (scale=%.3f) on irregular plurals:" % scale_bp)
print("  %-12s -> %-12s  clean_nn           hit?" % ("source", "target"))
for src, tgt in IRREGULAR_PLURALS:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None:
        print("  %-12s  SKIP (multi-token)" % src)
        continue
    pred = W_E[sid] + scale_bp * ax_bp
    r = nn_retrieve_clean(pred, [sid], top_n=3)
    hit = (tid is not None and r[0][0] == tgt)
    top3 = ', '.join(w for w,_,_ in r[:3])
    print("  %-12s -> %-12s  %-24s  %s" % (src, tgt, top3, '✓' if hit else '✗'))
print()

# Also test object-noun axis and check foot/tooth
ax_obj, _, valid_obj, _ = compute_axis([('cat','cats'),('dog','dogs'),('house','houses'),
                                          ('car','cars'),('tree','trees'),('book','books'),
                                          ('bird','birds'),('ship','ships')])
scale_obj, _ = best_scale_clean(ax_obj.astype(np.float32), valid_obj)
print("  Testing object-noun axis (scale=%.3f) on irregular plurals:" % scale_obj)
print("  %-12s -> %-12s  clean_nn           hit?" % ("source", "target"))
for src, tgt in IRREGULAR_PLURALS[:8]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    pred = W_E[sid] + scale_obj * ax_obj
    r = nn_retrieve_clean(pred, [sid], top_n=3)
    hit = (tid is not None and r[0][0] == tgt)
    top3 = ', '.join(w for w,_,_ in r[:3])
    print("  %-12s -> %-12s  %-24s  %s" % (src, tgt, top3, '✓' if hit else '✗'))
print()

# ====================================================================
# PART D: gPC5 ROYAL AXIS — WHAT TOKENS ARE AT POLES?
# ====================================================================
print("PART D: gPC5 royal axis — pole tokens and navigability")
print("-"*70)

ALL_GENDER_EXTENDED = [
    ('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
    ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife'),
    ('grandfather','grandmother'),('nephew','niece'),('groom','bride'),
    ('lord','lady'),('prince','princess'),('knight','dame'),
    ('actor','actress'),('waiter','waitress'),('hero','heroine'),
    ('host','hostess'),('heir','heiress'),
]
gender_chords = []
for s, t in ALL_GENDER_EXTENDED:
    es, _ = get_emb(s); et, _ = get_emb(t)
    if es is None or et is None: continue
    gender_chords.append(normed(et-es).astype(np.float32))

M_g = np.array(gender_chords)
M_gc = (M_g - M_g.mean(axis=0)).astype(np.float32)

rng = np.random.default_rng(42)
gPCs_vecs = []
M_defl = M_gc.copy()
for k in range(10):
    vk = rng.standard_normal(M_gc.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(300):
        vk = M_defl.T @ (M_defl @ vk)
        n = np.linalg.norm(vk)
        if n < 1e-10: break
        vk /= n
    proj = M_defl @ vk
    M_defl = M_defl - np.outer(proj, vk)
    gPCs_vecs.append(vk.astype(np.float64))

gPC5 = gPCs_vecs[4].astype(np.float32)
print("  Top tokens at gPC5 positive pole:")
scores = W_n @ gPC5
top_pos = np.argsort(scores)[-20:][::-1]
for i in top_pos:
    print("    %-22s  %.4f" % (tok.decode([i]).strip(), float(scores[i])))
print()
print("  Top tokens at gPC5 negative pole:")
top_neg = np.argsort(scores)[:20]
for i in top_neg:
    print("    %-22s  %.4f" % (tok.decode([i]).strip(), float(scores[i])))
print()

# Navigation test: can we use gPC5 to retrieve 'queen' from 'king'?
print("  gPC5 navigability test:")
for src, tgt in [('king','queen'),('emperor','empress'),('prince','princess'),
                 ('duke','duchess'),('tsar','tsarina'),('lord','lady'),
                 ('man','woman'),('boy','girl'),('hero','heroine')]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    # Find scale that maximizes similarity to target
    best_s, best_hit = 1.0, False
    for s_test in np.linspace(0.01, 4.0, 200):
        pred = W_E[sid] + s_test * gPC5
        r = nn_retrieve_clean(pred, [sid], 1)
        if tid is not None and r[0][0] == tgt:
            best_s = s_test; best_hit = True; break
    if not best_hit:
        pred = W_E[sid] + best_s * gPC5
        r = nn_retrieve_clean(pred, [sid], 3)
        top3 = ', '.join(w for w,_,_ in r[:3])
        print("  %-8s -> %-12s  scale=??  got: %-20s  ✗" % (src, tgt, top3))
    else:
        pred = W_E[sid] + best_s * gPC5
        r = nn_retrieve_clean(pred, [sid], 3)
        print("  %-8s -> %-12s  scale=%.2f  got: %-20s  ✓" % (src, tgt, best_s, r[0][0]))
print()

# Also compare gPC5 with gPC1 on 'man->woman' type pairs
print("  gPC1 navigability (for comparison):")
gPC1 = gPCs_vecs[0].astype(np.float32)
for src, tgt in [('man','woman'),('boy','girl'),('king','queen'),
                 ('father','mother'),('son','daughter')]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    best_s, best_hit = 1.0, False
    for s_test in np.linspace(0.01, 4.0, 200):
        pred = W_E[sid] + s_test * gPC1
        r = nn_retrieve_clean(pred, [sid], 1)
        if tid is not None and r[0][0] == tgt:
            best_s = s_test; best_hit = True; break
    if not best_hit:
        pred = W_E[sid] + best_s * gPC1
        r = nn_retrieve_clean(pred, [sid], 3)
        top3 = ', '.join(w for w,_,_ in r[:3])
        print("  %-8s -> %-12s  scale=??  got: %-20s  ✗" % (src, tgt, top3))
    else:
        pred = W_E[sid] + best_s * gPC1
        r = nn_retrieve_clean(pred, [sid], 3)
        print("  %-8s -> %-12s  scale=%.2f  got: %-20s  ✓" % (src, tgt, best_s, r[0][0]))
print()

# ====================================================================
# PART E: WHAT IS IN THE hand NEIGHBORHOOD GEOMETRICALLY?
# ====================================================================
print("PART E: Why is hand->hands irreducible? Detailed neighborhood")
print("-"*70)
es_hand, sid_hand = get_emb('hand')
es_hands, sid_hands = get_emb('hands')
es_arm, sid_arm = get_emb('arm')
es_arms, sid_arms = get_emb('arms')

if es_hand is not None and es_hands is not None:
    # How much of the space between 'hand' and 'hands' is occupied by other tokens?
    direction_hh = normed(es_hands - es_hand).astype(np.float32)
    print("  Tokens along direction hand->hands:")
    scores_hh = W_n @ direction_hh
    # Project onto normalized vector from hand
    hand_n = normed(es_hand).astype(np.float32)
    hands_n = normed(es_hands).astype(np.float32)
    # Find tokens that are "between" hand and hands
    # i.e., cos(tok, direction) > 0 AND cos(tok, hand_n) > cos(hand, hands_n)
    cos_hh = float(np.dot(hand_n, hands_n))
    print("  cos(hand, hands) = %.4f" % cos_hh)
    print()

    # Show what happens at each interpolation step from hand to hands
    print("  Interpolation steps from hand to hands:")
    print("  t     top-1 clean token         sim")
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        interp = (1-t) * es_hand + t * es_hands
        r = nn_retrieve_clean(interp, [], top_n=1)
        print("  %.1f   %-22s  %.4f" % (t, r[0][0], r[0][1]))
    print()

    # Compare with arm->arms interpolation
    print("  Interpolation steps from arm to arms:")
    print("  t     top-1 clean token         sim")
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        interp = (1-t) * es_arm + t * es_arms
        r = nn_retrieve_clean(interp, [], top_n=1)
        print("  %.1f   %-22s  %.4f" % (t, r[0][0], r[0][1]))
    print()
    
    # What is the cos(hand_path_halfway, hands)?
    halfway = normed(0.5 * es_hand + 0.5 * es_hands).astype(np.float32)
    print("  cos(halfway hand->hands, hands) = %.4f" % float(np.dot(halfway, W_n[sid_hands])))
    print("  cos(halfway arm->arms, arms)    = %.4f" % float(np.dot(
        normed(0.5*es_arm + 0.5*es_arms).astype(np.float32), W_n[sid_arms])))
    print()

    # Check the region near the body-part axis endpoint for 'hand'
    print("  Tokens near body-part axis endpoint for 'hand' (scale=%.3f):" % scale_bp)
    pred = W_E[sid_hand] + scale_bp * ax_bp
    r_all = nn_retrieve(pred, [], top_n=15)
    for w, s, i in r_all:
        is_cap = w and w[0].isupper()
        is_cmp = w.startswith('-') or w.startswith('_')
        marker = ' [CAP]' if is_cap else (' [CMP]' if is_cmp else '')
        print("    %-22s  %.4f%s" % (w, s, marker))
