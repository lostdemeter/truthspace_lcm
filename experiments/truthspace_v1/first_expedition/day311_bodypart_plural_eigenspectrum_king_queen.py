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
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve_clean(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def eigenspectrum(pairs, n_components=5):
    """Compute top-n eigenvalues of the chord matrix PCA."""
    chords = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        chords.append(normed(et - es).astype(np.float32))
    if len(chords) < 3: return []
    M = np.array(chords)
    M_c = (M - M.mean(axis=0)).astype(np.float32)
    rng = np.random.default_rng(42)
    lambdas = []
    M_d = M_c.copy()
    for k in range(min(n_components, len(chords)-1)):
        vk = rng.standard_normal(M_c.shape[1]).astype(np.float32)
        vk /= np.linalg.norm(vk)
        for _ in range(200):
            vk = M_d.T @ (M_d @ vk)
            n = np.linalg.norm(vk)
            if n < 1e-10: break
            vk /= n
        proj = M_d @ vk
        lam = float(np.var(M_d @ vk))
        M_d = M_d - np.outer(proj, vk)
        lambdas.append(lam)
    total = sum(lambdas) if sum(lambdas) > 0 else 1.0
    return lambdas, [l/total for l in lambdas]

print("DAY 311: BODY-PART PLURAL, EIGENSPECTRUM, king->queen ANOMALY")
print("="*70)
print()

# ====================================================================
# PART A: BODY-PART PLURAL AXIS
# ====================================================================
print("PART A: Body-part plural axis")
print("-"*70)

BODYPART_TRAIN = [
    ('head','heads'),('foot','feet'),('ear','ears'),('knee','knees'),
    ('toe','toes'),('lip','lips'),('hip','hips'),('rib','ribs'),
    ('thumb','thumbs'),('wrist','wrists'),('elbow','elbows'),('heel','heels'),
    ('shoulder','shoulders'),('chin','chins'),('neck','necks'),('jaw','jaws'),
]
BODYPART_HOLDOUT = [
    ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
    ('finger','fingers'),('ankle','ankles'),('chest','chests'),('back','backs'),
    ('nose','noses'),('cheek','cheeks'),('forehead','foreheads'),
]

print("  Training set — checking single-token availability:")
for s, t in BODYPART_TRAIN:
    _, sid = get_emb(s); _, tid = get_emb(t)
    print("  %-14s -> %-14s  %s" % (s, t, 'both' if sid and tid else
          ('src' if sid else ('tgt' if tid else 'neither'))))

ax_bp, _, valid_bp, pc_bp = compute_axis(BODYPART_TRAIN)
print()
if ax_bp is not None:
    scale_bp, tr_acc = best_scale(ax_bp.astype(np.float32), valid_bp)
    print("  Body-part +s axis: pc=%.4f  scale=%.3f  train=%d/%d" %
          (pc_bp, scale_bp, tr_acc, len(valid_bp)))

    # Object-noun axis (standard training)
    ax_obj, _, valid_obj, pc_obj = compute_axis([
        ('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
        ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')])
    cos_bp_obj = float(np.dot(ax_bp, ax_obj)) if ax_obj is not None else 0.0
    print("  cos(bodypart_axis, object_axis) = %.4f" % cos_bp_obj)
    print()

    print("  Holdout results with body-part axis:")
    print("  %-12s -> %-12s  got (clean)    hit?" % ("source", "target"))
    for src, tgt in BODYPART_HOLDOUT:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: print("  %-12s  SKIP (multi-token)" % src); continue
        pred = W_E[sid] + scale_bp * ax_bp
        r = nn_retrieve_clean(pred, [sid], top_n=3)
        hit = (tid is not None and r[0][0] == tgt)
        top3 = ', '.join(w for w,_,_ in r[:3])
        print("  %-12s -> %-12s  %-22s  %s" % (src, tgt, top3, '✓' if hit else '✗'))

    print()
    # Object-noun axis on body-part holdout for comparison
    if ax_obj is not None:
        scale_obj, _ = best_scale(ax_obj.astype(np.float32), valid_obj)
        print("  Object-noun axis on same holdout (scale=%.3f, pc=%.4f):" % (scale_obj, pc_obj))
        hits_bp, hits_obj, total = 0, 0, 0
        for src, tgt in BODYPART_HOLDOUT:
            es, sid = get_emb(src); et, tid = get_emb(tgt)
            if es is None or tid is None: continue
            total += 1
            pred_bp  = W_E[sid] + scale_bp  * ax_bp
            pred_obj = W_E[sid] + scale_obj * ax_obj
            r_bp  = nn_retrieve_clean(pred_bp,  [sid], 1)
            r_obj = nn_retrieve_clean(pred_obj, [sid], 1)
            if r_bp[0][0]  == tgt: hits_bp  += 1
            if r_obj[0][0] == tgt: hits_obj += 1
        print("  Body-part axis: %d/%d=%.0f%%  Object axis: %d/%d=%.0f%%" %
              (hits_bp, total, 100*hits_bp/total, hits_obj, total, 100*hits_obj/total))
print()

# ====================================================================
# PART B: EIGENVALUE SPECTRUM FOR ALL 12 AXES
# ====================================================================
print("PART B: Eigenvalue spectrum for all 12 morphological axes")
print("-"*70)

ALL_AXES = {
    '+er':     [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                ('light','lighter'),('strong','stronger'),('weak','weaker'),('soft','softer'),
                ('hard','harder'),('sharp','sharper'),('warm','warmer'),('cool','cooler')],
    '+est':    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest'),
                ('hard','hardest'),('warm','warmest'),('cool','coolest'),('sweet','sweetest')],
    'er->est': [('faster','fastest'),('slower','slowest'),('taller','tallest'),
                ('shorter','shortest'),('brighter','brightest'),('darker','darkest'),
                ('deeper','deepest'),('harder','hardest'),('warmer','warmest')],
    'gender':  [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife'),
                ('grandfather','grandmother'),('nephew','niece'),('groom','bride')],
    'past_irr':[('go','went'),('come','came'),('run','ran'),('see','saw'),
                ('eat','ate'),('know','knew'),('take','took'),('make','made'),
                ('give','gave'),('find','found'),('buy','bought'),('bring','brought')],
    '+ed':     [('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                ('end','ended'),('look','looked'),('call','called'),('help','helped'),
                ('play','played'),('work','worked'),('turn','turned'),('push','pushed')],
    '+ness':   [('sad','sadness'),('happy','happiness'),('dark','darkness'),('kind','kindness'),
                ('bright','brightness'),('mad','madness'),('sick','sickness'),('weak','weakness'),
                ('bold','boldness'),('cold','coldness')],
    '+ful':    [('hope','hopeful'),('care','careful'),('use','useful'),('power','powerful'),
                ('peace','peaceful'),('harm','harmful'),('thank','thankful'),('help','helpful'),
                ('play','playful'),('wonder','wonderful'),('color','colorful'),('grace','graceful')],
    'un-':     [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('known','unknown'),
                ('usual','unusual'),('clear','unclear'),('lock','unlock'),('wrap','unwrap'),
                ('tie','untie'),('fold','unfold'),('pack','unpack'),('cover','uncover')],
    '+ment':   [('achieve','achievement'),('manage','management'),('develop','development'),
                ('move','movement'),('treat','treatment'),('argue','argument'),
                ('judge','judgment'),('employ','employment'),('invest','investment')],
    '+s':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                ('tree','trees'),('book','books'),('bird','birds'),('ship','ships'),
                ('flower','flowers'),('star','stars'),('boat','boats'),('cup','cups')],
    '+tion':   [('act','action'),('direct','direction'),('collect','collection'),
                ('connect','connection'),('protect','protection'),('select','selection'),
                ('inject','injection'),('reject','rejection'),('infect','infection'),
                ('inspect','inspection'),('detect','detection'),('correct','correction')],
}

print("  %-12s  n    pc      λ1/Σλ   λ2/Σλ   λ3/Σλ   isotropy" % "axis")
print("  " + "-"*66)
axis_isotropy = {}
for nm, pairs in ALL_AXES.items():
    ax, _, valid, pc = compute_axis(pairs)
    if ax is None: continue
    result = eigenspectrum(pairs, n_components=5)
    if not result: continue
    lambdas, fracs = result
    # Isotropy = 1 - (λ1 - λ2)/λ1 (0=totally directional, 1=perfectly isotropic)
    isotropy = 1.0 - (fracs[0] - fracs[1]) if len(fracs) >= 2 else 0.0
    axis_isotropy[nm] = isotropy
    print("  %-12s  %d   %.4f  %.4f  %.4f  %.4f  %.4f" %
          (nm, len(valid), pc, fracs[0], fracs[1] if len(fracs)>1 else 0,
           fracs[2] if len(fracs)>2 else 0, isotropy))

print()
print("  Most isotropic axes (no dominant direction):")
for nm, iso in sorted(axis_isotropy.items(), key=lambda x: -x[1])[:5]:
    print("    %-12s  isotropy=%.4f" % (nm, iso))
print()
print("  Most directional axes (clear dominant direction):")
for nm, iso in sorted(axis_isotropy.items(), key=lambda x: x[1])[:5]:
    print("    %-12s  isotropy=%.4f" % (nm, iso))
print()

# ====================================================================
# PART C: king->queen ANOMALY — WHAT PC DOES IT LIVE IN?
# ====================================================================
print("PART C: king->queen anomaly — projecting onto all gender PCs")
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
gender_labels = []
for s, t in ALL_GENDER_EXTENDED:
    es, _ = get_emb(s); et, _ = get_emb(t)
    if es is None or et is None: continue
    gender_chords.append(normed(et-es).astype(np.float32))
    gender_labels.append((s, t))

M_g = np.array(gender_chords)
M_gc = (M_g - M_g.mean(axis=0)).astype(np.float32)

# Compute top-10 gender PCs
rng = np.random.default_rng(42)
gPCs = []
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
    lam = float(np.var(M_gc @ vk))
    M_defl = M_defl - np.outer(proj, vk)
    gPCs.append((vk.astype(np.float64), lam))

total_lam = sum(l for _, l in gPCs)
print("  Gender PCA eigenvalue spectrum (top 10):")
for k, (v, lam) in enumerate(gPCs):
    print("  gPC%2d  λ=%.6f  frac=%.4f" % (k+1, lam, lam/total_lam if total_lam else 0))
print()

# Project each gender pair chord onto all 10 gPCs
print("  Projections of king->queen chord onto all 10 gPCs:")
es_k, sid_k = get_emb('king'); et_q, _ = get_emb('queen')
if es_k is not None and et_q is not None:
    chord_kq = normed(et_q - es_k).astype(np.float32)
    for k, (v, lam) in enumerate(gPCs):
        proj = float(np.dot(chord_kq, v.astype(np.float32)))
        print("  gPC%2d  projection=%+.4f" % (k+1, proj))

print()
# Reconstruct king->queen from top-N gPCs
print("  King->queen reconstruction accuracy using top-N gPCs:")
if es_k is not None and et_q is not None:
    chord_kq = (et_q - es_k).astype(np.float64)
    chord_kq_n = normed(chord_kq)
    for n_pcs in [1, 2, 3, 5, 8, 10]:
        recon = np.zeros(W_E.shape[1])
        for k in range(n_pcs):
            v, _ = gPCs[k]
            c = normed((et_q-es_k).astype(np.float32))
            proj = float(np.dot(c, v.astype(np.float32)))
            recon += proj * v
        cos_recon = float(np.dot(normed(recon), chord_kq_n))
        print("  Using top-%d gPCs: cos(recon, actual)=%.4f" % (n_pcs, cos_recon))

print()
# Compare king->queen vs man->woman across gPCs
print("  Top-5 gPC projections: king->queen vs man->woman vs boy->girl:")
for word_pair in [('king','queen'),('man','woman'),('boy','girl'),
                  ('father','mother'),('uncle','aunt'),('groom','bride')]:
    s, t = word_pair
    es, _ = get_emb(s); et, _ = get_emb(t)
    if es is None or et is None: continue
    c = normed(et-es).astype(np.float32)
    projs = [float(np.dot(c, gPCs[k][0].astype(np.float32))) for k in range(5)]
    norm_top5 = np.sqrt(sum(p**2 for p in projs))
    print("  %-10s->%-10s  " % (s, t) +
          "  ".join("%+.4f" % p for p in projs) +
          "  ||top5||=%.4f" % norm_top5)

print()

# ====================================================================
# PART D: SEMANTIC FIELD OF IRREDUCIBLE PLURAL FAILURES
# ====================================================================
print("PART D: Semantic field analysis — hand vs arm (why different?)")
print("-"*70)

# Find nearest neighbors of 'hands' and 'eyes' in W_E
for target_word in ['hands', 'eyes', 'arms', 'legs']:
    _, tid = get_emb(target_word)
    if tid is None:
        print("  %s: MULTI-TOKEN" % target_word)
        continue
    pred_n = W_n[tid]
    sims = W_n @ pred_n
    sims[tid] = -1.0
    top = np.argsort(sims)[-10:][::-1]
    print("  Nearest neighbors of '%s':" % target_word)
    for i in top:
        w = tok.decode([i]).strip()
        print("    %-20s  %.4f" % (w, float(sims[i])))
    print()

# Check what direction the +s axis points for hand vs arm
print("  +s axis score for hand, eye, arm, leg (does axis point away from body parts?):")
ax_obj, _, _, _ = compute_axis([('cat','cats'),('dog','dogs'),('house','houses'),
                                  ('car','cars'),('tree','trees'),('book','books'),
                                  ('bird','birds'),('ship','ships')])
for word in ['hand', 'eye', 'arm', 'leg', 'foot', 'head', 'ear', 'nose',
             'cat', 'dog', 'car', 'tree', 'cup', 'door']:
    es, sid = get_emb(word)
    if es is None: continue
    # How much does the +s axis point from word toward its plural?
    _, tid = get_emb(word+'s') if word not in ['foot','eye'] else \
             (get_emb('feet') if word=='foot' else get_emb('eyes'))
    if tid is None: continue
    chord = normed(W_E[tid] - W_E[sid]).astype(np.float32)
    score = float(np.dot(chord, ax_obj.astype(np.float32)))
    print("  %-10s  cos(word_plural, obj_axis)=%+.4f" % (word, score))

print()

# ====================================================================
# PART E: BODY-PART +s AXIS vs OBJECT NOUN +s AXIS
# ====================================================================
print("PART E: Comparing body-part and object-noun +s axes")
print("-"*70)

if ax_bp is not None and ax_obj is not None:
    cos_axes = float(np.dot(ax_bp, ax_obj))
    print("  cos(body-part axis, object-noun axis) = %.4f" % cos_axes)
    print()

    # Cross-test
    FULL_NOUNS = [('flower','flowers'),('star','stars'),('boat','boats'),
                  ('cup','cups'),('door','doors'),('road','roads'),
                  ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
                  ('head','heads'),('ear','ears'),('knee','knees'),('foot','feet')]

    scale_bp2, _ = best_scale(ax_bp.astype(np.float32), valid_bp)
    scale_obj2, _ = best_scale(ax_obj.astype(np.float32), valid_obj)

    print("  %-12s -> %-12s  obj_axis   bp_axis   target" % ("source", "target"))
    hits_obj2, hits_bp2, total2 = 0, 0, 0
    for src, tgt in FULL_NOUNS:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        total2 += 1
        pred_obj = W_E[sid] + scale_obj2 * ax_obj
        pred_bp  = W_E[sid] + scale_bp2  * ax_bp
        r_obj = nn_retrieve_clean(pred_obj, [sid], 1)
        r_bp  = nn_retrieve_clean(pred_bp,  [sid], 1)
        hit_obj = (tid is not None and r_obj[0][0] == tgt)
        hit_bp  = (tid is not None and r_bp[0][0]  == tgt)
        if hit_obj: hits_obj2 += 1
        if hit_bp:  hits_bp2  += 1
        marker = '↑' if (hit_bp and not hit_obj) else \
                 ('↓' if (hit_obj and not hit_bp) else '=')
        print("  %-12s -> %-12s  %-10s %-10s %-10s  %s" %
              (src, tgt, r_obj[0][0], r_bp[0][0], tgt, marker))

    print()
    print("  Object axis: %d/%d=%.0f%%" % (hits_obj2, total2, 100*hits_obj2/total2))
    print("  Body-part axis: %d/%d=%.0f%%" % (hits_bp2, total2, 100*hits_bp2/total2))
