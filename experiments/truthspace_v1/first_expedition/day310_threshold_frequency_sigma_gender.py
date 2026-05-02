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
def get_all_ids(word):
    """Return all token-ids for all prefix variants."""
    ids_all = []
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        ids_all.append((p+word, ids))
    return ids_all
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
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 310: DISPLACEMENT THRESHOLD, FREQUENCY, SIGMA-GENDER")
print("="*70)
print()

# ====================================================================
# PART A: TITLES HOLDOUT TOKENIZATION CHECK
# ====================================================================
print("PART A: Titles holdout — tokenization deep-dive")
print("-"*70)
TITLES_PAIRS = [
    ('lord','lady'),('duke','duchess'),('prince','princess'),
    ('emperor','empress'),('king','queen'),
    ('count','countess'),('baron','baroness'),('marquis','marchioness'),
    ('viscount','viscountess'),('tsar','tsarina'),
    ('sir','madam'),('knight','dame'),('earl','countess'),('sultan','sultana'),
]
print("  %-14s  src_tok  %-14s  tgt_tok  single?" % ("source", "target"))
print("  " + "-"*58)
for src, tgt in TITLES_PAIRS:
    _, sid = get_emb(src)
    _, tid = get_emb(tgt)
    # Get raw tokenisation
    src_ids = tok(' '+src, add_special_tokens=False)['input_ids']
    tgt_ids = tok(' '+tgt, add_special_tokens=False)['input_ids']
    single = 'both' if (sid is not None and tid is not None) else \
             'src' if (sid is not None) else \
             'tgt' if (tid is not None) else 'neither'
    print("  %-14s  [%s]    %-14s  [%s]    %s" %
          (src, ','.join(str(x) for x in src_ids[:2]),
           tgt, ','.join(str(x) for x in tgt_ids[:2]),
           single))
print()

# Build titles axis from single-token pairs only
titles_single = [(s,t) for s,t in TITLES_PAIRS if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Single-token titles pairs:", len(titles_single))
for s,t in titles_single:
    print("    %s -> %s" % (s, t))
ax_titles, _, valid_titles, pc_titles = compute_axis(titles_single)
if ax_titles is not None:
    scale_titles, tr_acc = best_scale(ax_titles.astype(np.float32), valid_titles)
    print("  Titles axis: pc=%.4f  scale=%.3f  train=%d/%d" % (pc_titles, scale_titles, tr_acc, len(valid_titles)))
    # Holdout: only single-token pairs not in training
    for s, t in TITLES_PAIRS:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        if (s,t) in titles_single[:len(valid_titles)]: continue
        pred = W_E[sid] + scale_titles * ax_titles
        r = nn_retrieve_clean(pred, [sid], top_n=3)
        hit = (r[0][0] == t)
        print("  holdout: %-10s -> %-12s  got: %s  %s" % (s, t, r[0][0], '✓' if hit else '✗'))
print()

# ====================================================================
# PART B: +s DISPLACEMENT THRESHOLD vs TOKEN FREQUENCY
# ====================================================================
print("PART B: +s displacement threshold — escape distance from source")
print("-"*70)
ax_s, _, valid_s, _ = compute_axis([('cat','cats'),('dog','dogs'),('house','houses'),
                                     ('car','cars'),('tree','trees'),('book','books'),
                                     ('bird','birds'),('ship','ships')])
scale_s = 0.181  # clean-optimal scale

HOLDOUT_S_ALL = [
    ('flower','flowers'),('star','stars'),('forest','forests'),('train','trains'),
    ('boat','boats'),('cup','cups'),('door','doors'),('road','roads'),
    ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
    ('wall','walls'),('room','rooms'),('fire','fires'),
    # Extra probes for threshold analysis
    ('bird','birds'),('cat','cats'),('dog','dogs'),('car','cars'),('book','books'),
]

print("  %-10s  cos(src,tgt)  delta_scale  clean_nn   hit?  freq_rank" % "source")
print("  " + "-"*62)
# Estimate "frequency rank" by norm of embedding (proxy — higher norm ~ more frequent)
for src, tgt in HOLDOUT_S_ALL[:15]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    cos_st = float(np.dot(normed(es).astype(np.float32),
                          normed(et).astype(np.float32))) if et is not None else 0.0
    # Find minimum scale needed to get target as top-1 clean NN
    min_scale_hit = None
    for s_test in np.linspace(0.05, 4.0, 200):
        pred = W_E[sid] + s_test * ax_s
        r = nn_retrieve_clean(pred, [sid], top_n=1)
        if et is not None and r[0][0] == tgt:
            min_scale_hit = s_test
            break
    # Norm of source embedding as frequency proxy (larger norm = more common in vocab)
    src_norm = float(np.linalg.norm(W_E[sid]))
    # Top clean NN at scale_s=0.181
    pred_def = W_E[sid] + scale_s * ax_s
    r_def = nn_retrieve_clean(pred_def, [sid], top_n=1)
    hit_def = (et is not None and r_def[0][0] == tgt)
    print("  %-10s  %.4f       %s           %-10s %s   %.1f" %
          (src, cos_st,
           "%.3f" % min_scale_hit if min_scale_hit else "NEVER",
           r_def[0][0], '✓' if hit_def else '✗', src_norm))

print()
# Separate hits from misses, compare mean min_scale
hits_scales, miss_scales = [], []
for src, tgt in HOLDOUT_S_ALL[:15]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None or et is None: continue
    for s_test in np.linspace(0.05, 4.0, 200):
        pred = W_E[sid] + s_test * ax_s
        r = nn_retrieve_clean(pred, [sid], top_n=1)
        if r[0][0] == tgt:
            (hits_scales if True else miss_scales).append(s_test)  # all reachable
            break
print("  Escape scales for each +s holdout word:")
for src, tgt in HOLDOUT_S_ALL[:15]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None or et is None: continue
    found = None
    for s_test in np.linspace(0.02, 6.0, 300):
        pred = W_E[sid] + s_test * ax_s
        r = nn_retrieve_clean(pred, [sid], top_n=1)
        if r[0][0] == tgt:
            found = s_test; break
    print("  %-10s -> %-10s  min_scale=%s" % (src, tgt, "%.3f" % found if found else "NEVER"))
print()

# ====================================================================
# PART C: TOKEN FREQUENCY PROXY ANALYSIS
# ====================================================================
print("PART C: Token frequency proxy (embedding norm) vs +s success")
print("-"*70)
# For each holdout word, compute embedding norm and whether it can be hit
print("  %-10s  emb_norm  can_hit?  min_scale" % "word")
for src, tgt in HOLDOUT_S_ALL[:15]:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None or et is None: continue
    src_norm = float(np.linalg.norm(W_E[sid]))
    tgt_norm = float(np.linalg.norm(W_E[tid]))
    found = None
    for s_test in np.linspace(0.02, 6.0, 300):
        pred = W_E[sid] + s_test * ax_s
        r = nn_retrieve_clean(pred, [sid], top_n=1)
        if r[0][0] == tgt:
            found = s_test; break
    print("  %-10s  src=%.2f  tgt=%.2f  %s  %s" %
          (src, src_norm, tgt_norm,
           'YES' if found else ' NO',
           "%.3f" % found if found else "-----"))
print()

# ====================================================================
# PART D: σ-GENDER — SECOND PRINCIPAL COMPONENT OF GENDER SPACE
# ====================================================================
print("PART D: σ-gender — two gender dimensions in W_E")
print("-"*70)

# Gather all gender pairs across all domains
ALL_GENDER_PAIRS = [
    # kin
    ('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
    ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife'),
    # kin extended
    ('grandfather','grandmother'),('nephew','niece'),('groom','bride'),
    # titles
    ('lord','lady'),('duke','duchess'),('prince','princess'),
    ('emperor','empress'),
    # occupation
    ('actor','actress'),('waiter','waitress'),('host','hostess'),
    ('heir','heiress'),('hero','heroine'),
    # animals
    ('lion','lioness'),('tiger','tigress'),('stallion','mare'),
]

chords_all = []
labels_all = []
for s, t in ALL_GENDER_PAIRS:
    es, sid = get_emb(s); et, tid = get_emb(t)
    if es is None or et is None: continue
    chords_all.append(normed(et-es).astype(np.float32))
    labels_all.append((s, t))

M = np.array(chords_all)
M_c = (M - M.mean(axis=0)).astype(np.float32)

# Compute top-2 PCs of the gender chord matrix
rng = np.random.default_rng(42)
gPCs = []
M_defl = M_c.copy()
for k in range(3):
    vk = rng.standard_normal(M_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(300):
        vk = M_defl.T @ (M_defl @ vk)
        vk /= np.linalg.norm(vk)
    proj = M_defl @ vk
    lam = float(np.var(M_c @ vk))
    M_defl = M_defl - np.outer(proj, vk)
    gPCs.append((vk.astype(np.float64), lam))
    print("  gPC%d: eigenvalue=%.6f  (var explained=%.1f%%)" %
          (k+1, lam, 100*lam/np.var(M_c)*M_c.shape[1]))

print()
gPC1, _ = gPCs[0]
gPC2, _ = gPCs[1]
print("  cos(gPC1, gPC2) = %.6f" % float(np.dot(gPC1, gPC2)))

# Compare kin vs animals on gPC1 and gPC2
print()
print("  Per-pair gPC1 and gPC2 projections:")
print("  %-14s -> %-14s  gPC1     gPC2" % ("src", "tgt"))
for (s, t), c in zip(labels_all, chords_all):
    p1 = float(np.dot(c, gPC1.astype(np.float32)))
    p2 = float(np.dot(c, gPC2.astype(np.float32)))
    # Classify domain
    dom = 'kin' if s in ['king','man','boy','father','son','brother','uncle','husband',
                          'grandfather','nephew','groom'] \
          else 'titles' if s in ['lord','duke','prince','emperor'] \
          else 'occ' if s in ['actor','waiter','host','heir','hero'] \
          else 'animal'
    print("  %-14s -> %-14s  %+.4f   %+.4f   [%s]" % (s, t, p1, p2, dom))

print()
# Compute per-domain mean gPC1 and gPC2
for dom, members in [('kin', ['king','man','boy','father','son','brother','uncle','husband',
                               'grandfather','nephew','groom']),
                      ('titles', ['lord','duke','prince','emperor']),
                      ('occ', ['actor','waiter','host','heir','hero']),
                      ('animal', ['lion','tiger','stallion'])]:
    dom_chords = [c for (s,t), c in zip(labels_all, chords_all) if s in members]
    if not dom_chords: continue
    m1 = float(np.mean([np.dot(c, gPC1.astype(np.float32)) for c in dom_chords]))
    m2 = float(np.mean([np.dot(c, gPC2.astype(np.float32)) for c in dom_chords]))
    print("  %-10s  mean_gPC1=%+.4f  mean_gPC2=%+.4f  n=%d" % (dom, m1, m2, len(dom_chords)))

print()
# Check: do animals cluster on gPC2 but not gPC1?
kin_chords = [c for (s,t), c in zip(labels_all, chords_all)
              if s in ['king','man','boy','father','son','brother','uncle','husband']]
animal_chords = [c for (s,t), c in zip(labels_all, chords_all)
                 if s in ['lion','tiger','stallion']]

if kin_chords and animal_chords:
    k1 = np.mean([np.dot(c, gPC1.astype(np.float32)) for c in kin_chords])
    k2 = np.mean([np.dot(c, gPC2.astype(np.float32)) for c in kin_chords])
    a1 = np.mean([np.dot(c, gPC1.astype(np.float32)) for c in animal_chords])
    a2 = np.mean([np.dot(c, gPC2.astype(np.float32)) for c in animal_chords])
    print("  KIN   axis: gPC1=%+.4f  gPC2=%+.4f" % (k1, k2))
    print("  ANIMAL axis: gPC1=%+.4f  gPC2=%+.4f" % (a1, a2))
    print()
    print("  Separation: kin vs animal on gPC2: |%.4f - %.4f| = %.4f" %
          (k2, a2, abs(k2-a2)))

print()

# ====================================================================
# PART E: WHAT TOKENS ARE NEAR gPC2?
# ====================================================================
print("PART E: Top tokens along gPC2 (the σ-gender axis)")
print("-"*70)
gpc2_f = gPC2.astype(np.float32)
scores = W_n @ gpc2_f
top_pos = np.argsort(scores)[-20:][::-1]
top_neg = np.argsort(scores)[:20]

print("  Positive pole (gPC2+):")
for i in top_pos:
    print("    %-20s  %.4f" % (tok.decode([i]).strip(), float(scores[i])))
print()
print("  Negative pole (gPC2-):")
for i in top_neg:
    print("    %-20s  %.4f" % (tok.decode([i]).strip(), float(scores[i])))
print()

# Also show where known gender pairs land
print("  Known gender pairs on gPC2:")
for s, t in [('king','queen'),('lion','lioness'),('stallion','mare'),
             ('duke','duchess'),('actor','actress'),('lord','lady')]:
    es, sid = get_emb(s); et, tid = get_emb(t)
    if es is None or et is None: continue
    s_score = float(np.dot(normed(es).astype(np.float32), gpc2_f))
    t_score = float(np.dot(normed(et).astype(np.float32), gpc2_f))
    print("    %-10s %+.4f  ->  %-10s %+.4f  (Δ=%+.4f)" %
          (s, s_score, t, t_score, t_score - s_score))
