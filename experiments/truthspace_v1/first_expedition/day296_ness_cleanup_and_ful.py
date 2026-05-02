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
def eval_pairs_full(pairs, ax, scale, label=''):
    results = []
    for s, t in pairs:
        es, sid = get_emb(s)
        if es is None: results.append((s,t,None,'?',False)); continue
        got = nn_retrieve(W_E[sid]+scale*ax, [sid])[0][0]
        results.append((s,t,sid,got,got==t))
    acc = sum(1 for _,_,sid,_,hit in results if hit and sid is not None)
    n   = sum(1 for _,_,sid,_,_ in results if sid is not None)
    if label:
        print("  %-24s  %d/%d (%.0f%%)" % (label, acc, n, 100*acc/max(1,n)))
    return results, acc, n

print("DAY 296: +NESS CLEANUP AND +FUL ANALYSIS")
print("="*65)
print()

# ====================================================================
# PART A: CLEAN +NESS AXIS
# Remove contaminated pairs: warm->warmth (not +ness), clean->cleanness
# (correct form is cleanliness).
# Keep only: adj+ness = canonical English word, no spelling change required.
# ====================================================================
print("PART A: Clean +ness axis (contamination removed)")
print("-"*65)

NESS_DIRTY = [
    ('happy','happiness'),('sad','sadness'),('kind','kindness'),
    ('dark','darkness'),('soft','softness'),('hard','hardness'),
    ('warm','warmth'),     # CONTAMINATED: +th, not +ness
    ('cold','coldness'),('bright','brightness'),
    ('clean','cleanness'), # CONTAMINATED: correct form = cleanliness
    ('loud','loudness'),('sweet','sweetness'),('weak','weakness'),
    ('bold','boldness'),('calm','calmness'),
]

# Pure +ness: adjective + ness = canonical word, no spelling change
NESS_CLEAN_TRAIN = [
    ('sad','sadness'),('kind','kindness'),('dark','darkness'),
    ('hard','hardness'),('cold','coldness'),('bright','brightness'),
    ('loud','loudness'),('sweet','sweetness'),('weak','weakness'),
    ('bold','boldness'),('calm','calmness'),('mild','mildness'),
    ('deep','deepness'),('tall','tallness'),('fast','fastness'),
]
NESS_CLEAN_HOLD = [
    ('neat','neatness'),('sharp','sharpness'),('rough','roughness'),
    ('thick','thickness'),('smooth','smoothness'),('plain','plainness'),
    ('round','roundness'),('cool','coolness'),('quick','quickness'),
    ('still','stillness'),
]

# Test dirty vs clean
for label, pairs_tr, pairs_ho in [
    ('DIRTY (w/ warmth)', NESS_DIRTY, [('neat','neatness'),('sharp','sharpness'),('smooth','smoothness')]),
    ('CLEAN (no contamination)', NESS_CLEAN_TRAIN, NESS_CLEAN_HOLD),
]:
    ax, coh, valid, pc = compute_axis(pairs_tr)
    if ax is None: continue
    s_opt, acc_tr = best_scale(ax, valid)
    hold_r, acc_h, n_h = eval_pairs_full(pairs_ho, ax, s_opt)
    print("  %-26s  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d  hold=%d/%d (%.0f%%)" % (
        label, pc, coh, s_opt, acc_tr, len(valid), acc_h, n_h, 100*acc_h/max(1,n_h)))
    if pairs_ho:
        for s, t, sid, got, hit in hold_r:
            if sid is None: continue
            print("    %-12s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if hit else '---'))
    print()

# ====================================================================
# PART B: +NESS SUB-PATTERNS BY PHONOLOGICAL CONTEXT
# ====================================================================
print("PART B: +ness sub-patterns by phonological context")
print("-"*65)
print("Hypothesis: -y adjectives (happy->happiness) form sub-pattern")
print("distinct from consonant-final adjectives (sad->sadness).")
print()

# Sub-pattern 1: adj ending in vowel+consonant (simple +ness)
NESS_VCC = [  # ends in consonant cluster or single consonant
    ('sad','sadness'),('kind','kindness'),('dark','darkness'),
    ('hard','hardness'),('cold','coldness'),('bright','brightness'),
    ('loud','loudness'),('bold','boldness'),('calm','calmness'),
    ('mild','mildness'),('tall','tallness'),
]

# Sub-pattern 2: adj ending in -ight
NESS_IGHT = [
    ('bright','brightness'),('light','lightness'),
    ('right','rightness'),  # possibly 'rightness' is not common
]

# Sub-pattern 3: adj ending in -y -> +iness (y->i spelling change)
NESS_Y = [
    ('happy','happiness'),('ready','readiness'),('empty','emptiness'),
    ('heavy','heaviness'),('pretty','prettiness'),('lazy','laziness'),
    ('crazy','craziness'),('easy','easiness'),('busy','busyness'),
    ('ugly','ugliness'),('lively','liveliness'),
]

# Sub-pattern 4: adj ending in -e (silent e + ness)
NESS_E = [
    ('pale','paleness'),('safe','safeness'),('brave','braveness'),
    ('pure','pureness'),('bare','bareness'),('rare','rareness'),
    ('wide','wideness'),('fine','fineness'),('wise','wiseness'),
]

print("  Sub-pattern comparison:")
for label, pairs in [
    ('consonant-final (+ness)', NESS_VCC),
    ('-y -> +iness',             NESS_Y),
    ('-e + ness',                NESS_E),
    ('-ight + ness',             NESS_IGHT),
]:
    ax, coh, valid, pc = compute_axis(pairs)
    if ax is None: print("  %-26s  SKIP" % label); continue
    s_opt, acc = best_scale(ax, valid)
    print("  %-26s  n=%2d  pc=%.4f  coh=%.4f  train=%d/%d (%.0f%%)" % (
        label, len(valid), pc, coh, acc, len(valid), 100*acc/max(1,len(valid))))
print()

# Inter-sub-axis cosines for +ness sub-patterns
ax_vcc, _, _, _ = compute_axis(NESS_VCC)
ax_y, _, _, _ = compute_axis(NESS_Y)
ax_e, _, _, _ = compute_axis(NESS_E)
if ax_vcc is not None and ax_y is not None:
    c = float(np.dot(ax_vcc.astype(np.float32), ax_y.astype(np.float32)))
    print("  cos(consonant-final, -y) = %.4f" % c)
if ax_vcc is not None and ax_e is not None:
    c = float(np.dot(ax_vcc.astype(np.float32), ax_e.astype(np.float32)))
    print("  cos(consonant-final, -e) = %.4f" % c)
if ax_y is not None and ax_e is not None:
    c = float(np.dot(ax_y.astype(np.float32), ax_e.astype(np.float32)))
    print("  cos(-y, -e) = %.4f" % c)
print()

# Cross-sub-axis: consonant-final axis applied to -y holdout
if ax_vcc is not None:
    _, _, valid_vcc, _ = compute_axis(NESS_VCC)
    s_vcc, _ = best_scale(ax_vcc, valid_vcc)
    cross_y = [('happy','happiness'),('ready','readiness'),('heavy','heaviness')]
    cross_e = [('pale','paleness'),('safe','safeness'),('brave','braveness')]
    for label, pairs in [('-y words (cross)', cross_y), ('-e words (cross)', cross_e)]:
        r, a, n = eval_pairs_full(pairs, ax_vcc, s_vcc, label)
        for s, t, sid, got, hit in r:
            if sid is None: continue
            print("    %-12s -> %-16s  got=%-16s [%s]" % (s, t, got, 'HIT' if hit else '---'))
print()

# ====================================================================
# PART C: WHY DOES +FUL ACHIEVE 67% HOLDOUT AT PC=0.104?
# ====================================================================
print("PART C: +ful anomaly -- why 67% holdout at pc=0.104?")
print("-"*65)

FUL_TRAIN = [
    ('hope','hopeful'),('care','careful'),('help','helpful'),
    ('wonder','wonderful'),('color','colorful'),('power','powerful'),
    ('peace','peaceful'),('grace','graceful'),('skill','skillful'),
    ('use','useful'),('cheer','cheerful'),('faith','faithful'),
]
FUL_HOLD = [
    ('harm','harmful'),('delight','delightful'),('respect','respectful'),
    ('thought','thoughtful'),('beauty','beautiful'),('play','playful'),
]

ax_ful, _, valid_ful, pc_ful = compute_axis(FUL_TRAIN)
if ax_ful is not None:
    s_ful, acc_tr = best_scale(ax_ful, valid_ful)
    print("  pc=%.4f  scale=%.2f  train=%d/%d" % (pc_ful, s_ful, acc_tr, len(valid_ful)))
    print()
    
    # Examine source cluster
    print("  Source word embeddings for +ful training:")
    src_embs = []
    for s, t, sid, tid in valid_ful:
        es = W_E[sid].copy()
        src_embs.append(normed(es).astype(np.float32))
        # nearest neighbors of source
        r = nn_retrieve(es, [sid], top_n=3)
        print("    %-10s  nn=[%s, %s, %s]" % (
            s, r[0][0], r[1][0], r[2][0]))
    src_pc = float(np.mean([np.dot(src_embs[i], src_embs[j])
                             for i in range(len(src_embs))
                             for j in range(i+1, len(src_embs))]))
    print("  src_pc = %.4f (homogeneity of source cluster)" % src_pc)
    print()
    
    # Full holdout test with details
    print("  Holdout results:")
    r, acc_h, n_h = eval_pairs_full(FUL_HOLD, ax_ful, s_ful)
    for s, t, sid, got, hit in r:
        if sid is None: continue
        # show top-3 candidates
        r3 = nn_retrieve(W_E[sid]+s_ful*ax_ful, [sid], top_n=3)
        print("    %-10s -> %-14s  got=%-14s [%s]  (top3: %s, %s)" % (
            s, t, got, 'HIT' if hit else '---', r3[1][0] if len(r3)>1 else '?', r3[2][0] if len(r3)>2 else '?'))
    print()
    
    # Why does beauty->beautiful work?
    # 'beautiful' is multi-token or single-token?
    for w in ['beautiful', 'harmful', 'delightful', 'respectful', 'thoughtful', 'playful']:
        ids = tok(' '+w, add_special_tokens=False)['input_ids']
        ids2 = tok(w, add_special_tokens=False)['input_ids']
        print("  token('%s') = %s  alt=%s" % (w, [tok.decode([i]) for i in ids], [tok.decode([i]) for i in ids2]))
    print()
    
    # Compare attractor cluster test
    print("  Attractor cluster test: 5 unseen nouns not in training")
    unseen = [('joy','joyful'),('taste','tasteful'),('truth','truthful'),
              ('awe','awesome'),('fear','fearful')]
    for s, t in unseen:
        es, sid = get_emb(s)
        if es is None: print("    %-10s  [multi-token]" % s); continue
        r3 = nn_retrieve(W_E[sid]+s_ful*ax_ful, [sid], top_n=5)
        got = r3[0][0]
        print("    %-10s -> %-14s  got=%-14s [%s]  (top5: %s, %s, %s)" % (
            s, t, got, 'HIT' if got==t else '---',
            r3[1][0] if len(r3)>1 else '?',
            r3[2][0] if len(r3)>2 else '?',
            r3[3][0] if len(r3)>3 else '?'))
    print()

# ====================================================================
# PART D: COMPARISON TABLE — pc vs holdout across all axes
# ====================================================================
print("PART D: pc vs holdout accuracy correlation")
print("-"*65)
print("  Can holdout accuracy be predicted from pc?")
print()

all_results = [
    # (name, pc, train_pct, holdout_pct)
    # Inflectional
    ("+er (comp)",        0.393, 100, 100),
    ("+est (sup)",        0.436, 100, 100),
    ("+s plural",         0.155, 100, 88),   # from earlier days
    ("+ed (past_r)",      0.174, 100, 75),   # from earlier days
    ("past_irr",          0.230, 100, 80),
    ("gender",            0.213, 100, 85),
    # Derivational
    ("+ness (dirty)",     0.211, 100,  0),   # true holdout 0%
    ("+ness (clean-est)", 0.250, 100, 40),   # predicted after cleanup
    ("un-",               0.121, 100, 33),
    ("in-/im-",           0.133, 100, 50),
    ("+ful",              0.104, 100, 67),   # ANOMALY
    ("+less",             0.133, 55,   0),
    ("+ment",             0.124, 100, 75),   # BEST
    ("+tion",             0.130, 100, 33),
    # Semantic
    ("country->demonym",  0.563, 100, 100),
    ("country->cap",      0.317, 100, 75),
    ("elem:single",       0.390, 100, 100),  # predicted (no holdout available)
    ("elem:double",       0.163, 100,  50),
    ("elem:latin",        0.104, 100,   0),
]

print("  %-22s  pc      train  holdout" % "Axis")
print("  " + "-"*55)
for name, pc, tr, ho in sorted(all_results, key=lambda x: -x[1]):
    print("  %-22s  %.3f   %3d%%   %3d%%" % (name, pc, tr, ho))
print()

# Spearman correlation between pc and holdout
import scipy.stats as stats
pcs  = [x[1] for x in all_results]
hold = [x[3] for x in all_results]
rho, pval = stats.spearmanr(pcs, hold)
print("  Spearman rho(pc, holdout) = %.4f  p=%.4f" % (rho, pval))
print()
print("  Outliers (high holdout, low pc):")
for name, pc, tr, ho in all_results:
    if ho >= 60 and pc < 0.15:
        print("    %-22s  pc=%.3f  hold=%d%%  *** ANOMALY" % (name, pc, ho))
print()

# ====================================================================
# PART E: CLEAN +NESS TRAIN/HOLDOUT — PROPER EVALUATION
# ====================================================================
print("PART E: Clean +ness train=10, holdout=10 proper evaluation")
print("-"*65)

NESS_PROPER_TRAIN = [
    ('sad','sadness'),('kind','kindness'),('dark','darkness'),
    ('hard','hardness'),('cold','coldness'),('loud','loudness'),
    ('sweet','sweetness'),('weak','weakness'),('bold','boldness'),
    ('calm','calmness'),
]
NESS_PROPER_HOLD = [
    ('neat','neatness'),('sharp','sharpness'),('rough','roughness'),
    ('thick','thickness'),('plain','plainness'),('round','roundness'),
    ('cool','coolness'),('still','stillness'),('fast','fastness'),
    ('mild','mildness'),
]

ax_nc, coh_nc, valid_nc, pc_nc = compute_axis(NESS_PROPER_TRAIN)
if ax_nc is not None:
    s_nc, acc_nc = best_scale(ax_nc, valid_nc)
    r_nc, acc_h_nc, n_h_nc = eval_pairs_full(NESS_PROPER_HOLD, ax_nc, s_nc)
    print("  Clean +ness:  pc=%.4f  coh=%.4f  scale=%.2f" % (pc_nc, coh_nc, s_nc))
    print("  Train: %d/%d (%.0f%%)" % (acc_nc, len(valid_nc), 100*acc_nc/max(1,len(valid_nc))))
    print("  Hold:  %d/%d (%.0f%%)" % (acc_h_nc, n_h_nc, 100*acc_h_nc/max(1,n_h_nc)))
    print()
    print("  Holdout detail:")
    for s, t, sid, got, hit in r_nc:
        if sid is None: continue
        print("    %-12s -> %-14s  got=%-14s [%s]" % (s, t, got, 'HIT' if hit else '---'))
