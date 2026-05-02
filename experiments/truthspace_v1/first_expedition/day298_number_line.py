import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr

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
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 298: NUMBER LINE, WEEKDAY->NUMBER, MONTH->NUMBER HOLDOUT")
print("="*65)
print()

# ====================================================================
# DIGIT / NUMBER TOKEN AVAILABILITY
# ====================================================================
print("Token availability:")
CARDINALS_STR = ['1','2','3','4','5','6','7','8','9',
                 '10','11','12','13','14','15','16','17','18','19','20']
CARDINALS_WORD = ['one','two','three','four','five','six','seven','eight',
                  'nine','ten','eleven','twelve','thirteen','fourteen','fifteen']
ORDINALS_WORD  = ['first','second','third','fourth','fifth','sixth','seventh',
                  'eighth','ninth','tenth','eleventh','twelfth']

for group, words in [('Digit strings', CARDINALS_STR),
                     ('Cardinal words', CARDINALS_WORD),
                     ('Ordinal words', ORDINALS_WORD)]:
    avail = [(w, get_emb(w)[1] is not None) for w in words]
    print("  %s: %d/%d single-token" % (group, sum(a for _,a in avail), len(avail)))
    print("    " + "  ".join("%s=%s" % (w, 'Y' if a else 'N') for w,a in avail))
print()

# ====================================================================
# PART A: NUMBER LINE AXIS (1->2, 2->3, ..., 8->9)
# ====================================================================
print("PART A: Number line axis (consecutive digit strings)")
print("-"*65)

digits = [str(i) for i in range(1, 20) if get_emb(str(i))[0] is not None]
print("  Single-token digit strings: %s" % ', '.join(digits))
print()

# Consecutive pairs from available single-token digits
digit_pairs_consec = [(digits[i], digits[i+1]) for i in range(len(digits)-1)]

ax_d, coh_d, valid_d, pc_d = compute_axis(digit_pairs_consec)
if ax_d is not None:
    s_d, acc_d = best_scale(ax_d, valid_d)
    print("  Consecutive digit axis:")
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_d, coh_d, s_d, acc_d, len(valid_d), 100*acc_d/max(1,len(valid_d))))
    print()
    print("  Full results:")
    for s, t in digit_pairs_consec:
        es, sid = get_emb(s)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+s_d*ax_d, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-4s -> %-4s  got=%-6s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# Cardinal word consecutive pairs
word_digits = [w for w in CARDINALS_WORD if get_emb(w)[0] is not None]
print("  Single-token cardinal words: %s" % ', '.join(word_digits))
word_pairs_consec = [(word_digits[i], word_digits[i+1]) for i in range(len(word_digits)-1)]

ax_wc, coh_wc, valid_wc, pc_wc = compute_axis(word_pairs_consec)
if ax_wc is not None:
    s_wc, acc_wc = best_scale(ax_wc, valid_wc)
    print("  Consecutive cardinal-word axis:")
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_wc, coh_wc, s_wc, acc_wc, len(valid_wc), 100*acc_wc/max(1,len(valid_wc))))
    print()

# SVD of digit embeddings — does PC1 correlate with numeric order?
if len(digits) >= 5:
    D_mat = np.array([get_emb(d)[0] for d in digits])
    D_c = D_mat - D_mat.mean(axis=0)
    U, S, Vt = np.linalg.svd(D_c, full_matrices=False)
    proj_d = U[:, :3] * S[:3]
    idx_d = list(range(len(digits)))
    r_d, p_d = pearsonr(idx_d, proj_d[:,0])
    rho_d, p_rho_d = spearmanr(idx_d, proj_d[:,0])
    print("  Digit SVD: Pearson r(digit_idx, PC1) = %.4f  p=%.4f" % (r_d, p_d))
    print("  Digit SVD: Spearman ρ(digit_idx, PC1) = %.4f  p=%.4f" % (rho_d, p_rho_d))
    print()
    print("  Digit projections onto PC1:")
    for i, d in enumerate(digits):
        print("    %3s  PC1=%+.4f" % (d, proj_d[i,0]))
    print()

# src_pc for digit embeddings
if len(digits) >= 3:
    src_vecs_d = [normed(get_emb(d)[0]).astype(np.float32) for d in digits]
    n = len(src_vecs_d)
    src_pc_d = float(np.mean([np.dot(src_vecs_d[i], src_vecs_d[j])
                               for i in range(n) for j in range(i+1,n)]))
    print("  src_pc (digit embeddings) = %.4f" % src_pc_d)
print()

# ====================================================================
# PART B: WEEKDAY->NUMBER AXIS
# ====================================================================
print("PART B: Weekday->number axis")
print("-"*65)

WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
WEEKDAY_NUM_PAIRS = [(WEEKDAYS[i], str(i+1)) for i in range(7)]

wd_avail = [(s,t) for s,t in WEEKDAY_NUM_PAIRS
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available weekday->number pairs: %d/%d" % (len(wd_avail), 7))
for s, t in wd_avail:
    print("    %-12s -> %s" % (s, t))
print()

ax_wdn, coh_wdn, valid_wdn, pc_wdn = compute_axis(wd_avail)
if ax_wdn is not None:
    s_wdn, acc_wdn = best_scale(ax_wdn, valid_wdn)
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_wdn, coh_wdn, s_wdn, acc_wdn, len(valid_wdn), 100*acc_wdn/max(1,len(valid_wdn))))
    print()
    for s, t, sid, tid in valid_wdn:
        r = nn_retrieve(W_E[sid]+s_wdn*ax_wdn, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-12s -> %s  got=%-6s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# ====================================================================
# PART C: MONTH->NUMBER HOLDOUT (train Jan-Jun, hold Jul-Sep)
# ====================================================================
print("PART C: Month->number holdout (train Jan-Jun, hold Jul-Sep)")
print("-"*65)

MONTHS = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']

MONTH_NUM_TRAIN = [(MONTHS[i], str(i+1)) for i in range(6)]   # Jan-Jun -> 1-6
MONTH_NUM_HOLD  = [(MONTHS[i], str(i+1)) for i in range(6,9)] # Jul-Sep -> 7-9

tr_avail = [(s,t) for s,t in MONTH_NUM_TRAIN
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
ho_avail = [(s,t) for s,t in MONTH_NUM_HOLD
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]

print("  Train: %s" % ', '.join('%s->%s' % (s,t) for s,t in tr_avail))
print("  Hold:  %s" % ', '.join('%s->%s' % (s,t) for s,t in ho_avail))
print()

ax_mnh, coh_mnh, valid_mnh, pc_mnh = compute_axis(tr_avail)
if ax_mnh is not None:
    s_mnh, acc_mnh = best_scale(ax_mnh, valid_mnh)
    print("  Train axis: pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_mnh, coh_mnh, s_mnh, acc_mnh, len(valid_mnh), 100*acc_mnh/max(1,len(valid_mnh))))
    print()
    
    hold_hits = 0
    print("  Holdout results:")
    for s, t in ho_avail:
        es, sid = get_emb(s)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+s_mnh*ax_mnh, [sid], top_n=5)
        got = r[0][0]
        hit = got == t
        if hit: hold_hits += 1
        rank_t = next((i+1 for i, (w,_,_) in enumerate(r) if w==t), None)
        print("    %-12s -> %s  got=%-6s [%s]  top5: %s" % (
            s, t, got, 'HIT' if hit else '---',
            ', '.join(w for w,_,_ in r)))
    print("  Holdout: %d/%d (%.0f%%)" % (hold_hits, len(ho_avail), 100*hold_hits/max(1,len(ho_avail))))
    print()
    
    # Full training set (all 9 months with single-token numbers)
    MONTH_NUM_ALL = [(MONTHS[i], str(i+1)) for i in range(9)]
    all_avail = [(s,t) for s,t in MONTH_NUM_ALL
                 if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    ax_mnf, _, valid_mnf, pc_mnf = compute_axis(all_avail)
    if ax_mnf is not None:
        s_mnf, acc_mnf = best_scale(ax_mnf, valid_mnf)
        print("  Full month->number (all 9): pc=%.4f  coh=%.4f  scale=%.2f  %d/%d (%.0f%%)" % (
            pc_mnf, coh_mnf if False else 0.0, s_mnf, acc_mnf, len(valid_mnf),
            100*acc_mnf/max(1,len(valid_mnf))))
    # Compute separately
    _, coh_mnf2, _, _ = compute_axis(all_avail)
    print()

# ====================================================================
# PART D: ORDINAL->CARDINAL AXIS (first->one, second->two, ...)
# ====================================================================
print("PART D: Ordinal->cardinal axis (first->one, second->two, ...)")
print("-"*65)

ORD_CARD = [
    ('first','one'),('second','two'),('third','three'),
    ('fourth','four'),('fifth','five'),('sixth','six'),
    ('seventh','seven'),('eighth','eight'),('ninth','nine'),
    ('tenth','ten'),
]
oc_avail = [(s,t) for s,t in ORD_CARD
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available pairs: %d/%d" % (len(oc_avail), len(ORD_CARD)))

ax_oc, coh_oc, valid_oc, pc_oc = compute_axis(oc_avail)
if ax_oc is not None:
    s_oc, acc_oc = best_scale(ax_oc, valid_oc)
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_oc, coh_oc, s_oc, acc_oc, len(valid_oc), 100*acc_oc/max(1,len(valid_oc))))
    print()
    for s, t, sid, tid in valid_oc:
        r = nn_retrieve(W_E[sid]+s_oc*ax_oc, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-10s -> %-8s  got=%-8s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# ====================================================================
# PART E: CARDINAL STRING -> CARDINAL WORD (1->one, 2->two, ...)
# ====================================================================
print("PART E: Digit string->word axis (1->one, 2->two, ...)")
print("-"*65)

DIGIT_WORD = [(str(i+1), CARDINALS_WORD[i]) for i in range(len(CARDINALS_WORD))]
dw_avail = [(s,t) for s,t in DIGIT_WORD
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available pairs: %d/%d" % (len(dw_avail), len(DIGIT_WORD)))

ax_dw, coh_dw, valid_dw, pc_dw = compute_axis(dw_avail)
if ax_dw is not None:
    s_dw, acc_dw = best_scale(ax_dw, valid_dw)
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_dw, coh_dw, s_dw, acc_dw, len(valid_dw), 100*acc_dw/max(1,len(valid_dw))))
    for s, t, sid, tid in valid_dw:
        r = nn_retrieve(W_E[sid]+s_dw*ax_dw, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-4s -> %-8s  got=%-8s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# ====================================================================
# PART F: NUMBER CLUSTER ANALYSIS
# SVD of digit embeddings to see if they form a line or circle
# ====================================================================
print("PART F: Number cluster analysis (SVD)")
print("-"*65)

all_nums = {d: get_emb(d)[0] for d in digits if get_emb(d)[0] is not None}
all_num_words = {w: get_emb(w)[0] for w in word_digits if get_emb(w)[0] is not None}

for group_name, emb_dict, idx_map in [
    ('digit strings', all_nums, {d: int(d)-1 for d in digits}),
    ('cardinal words', all_num_words, {w: CARDINALS_WORD.index(w) for w in word_digits}),
]:
    if len(emb_dict) < 4: continue
    keys = list(emb_dict.keys())
    M = np.array([emb_dict[k] for k in keys])
    M_c = M - M.mean(axis=0)
    U, S, Vt = np.linalg.svd(M_c, full_matrices=False)
    var_ratio = S**2 / (S**2).sum()
    proj = U[:, :2] * S[:2]
    idxs = [idx_map[k] for k in keys]
    r1, p1 = pearsonr(idxs, proj[:,0])
    r2, p2 = pearsonr(idxs, proj[:,1])
    print("  %s:" % group_name)
    print("  Var explained: PC1=%.3f  PC2=%.3f  PC3=%.3f" % (var_ratio[0], var_ratio[1], var_ratio[2]))
    print("  Pearson r(idx, PC1) = %.4f  p=%.4f" % (r1, p1))
    print("  Pearson r(idx, PC2) = %.4f  p=%.4f" % (r2, p2))
    print("  PC1 projections:")
    for k, idx in sorted(zip(keys, idxs)):
        print("    %-8s  PC1=%+.4f" % (k, proj[idxs.index(idx),0]))
    print()

# ====================================================================
# PART G: CROSS-DOMAIN AXES (month->number vs weekday->number vs ...)
# Do they share the same axis direction?
# ====================================================================
print("PART G: Cross-domain labelling axes (month->num, weekday->num, ordinal->cardinal)")
print("-"*65)

axes_info = {}
for label, pairs in [
    ('month->num',  [(MONTHS[i], str(i+1)) for i in range(9)]),
    ('weekday->num', WEEKDAY_NUM_PAIRS),
    ('ordinal->card', ORD_CARD),
    ('digit->word',  DIGIT_WORD),
]:
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    ax, _, _, pc = compute_axis(avail)
    axes_info[label] = (ax, pc, avail)
    print("  %-18s  pc=%.4f  n=%d" % (label, pc, len(avail)))
print()

# Pairwise cosines between labelling axes
labels = list(axes_info.keys())
print("  Inter-axis cosines:")
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        ax_i = axes_info[labels[i]][0]
        ax_j = axes_info[labels[j]][0]
        if ax_i is None or ax_j is None: continue
        c = float(np.dot(ax_i.astype(np.float32), ax_j.astype(np.float32)))
        print("  %-18s <-> %-18s  cos=%.4f" % (labels[i], labels[j], c))
print()

# ====================================================================
# PART H: UPDATED LINEARITY SPECTRUM
# ====================================================================
print("="*65)
print("UPDATED LINEARITY SPECTRUM (Day 298)")
print("="*65)

new_axes = []
for label, pairs in [
    ('digit consec', digit_pairs_consec),
    ('card-word consec', word_pairs_consec),
    ('digit->word', DIGIT_WORD),
    ('ordinal->card', ORD_CARD),
    ('weekday->num', WEEKDAY_NUM_PAIRS),
]:
    avail = [(s,t) for s,t in pairs
             if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if len(avail) < 2: continue
    _, _, _, pc = compute_axis(avail)
    new_axes.append((label, pc, 'NUMERIC'))

MONTH_NUM_ALL = [(MONTHS[i], str(i+1)) for i in range(9)]
mn_avail = [(s,t) for s,t in MONTH_NUM_ALL
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
_, _, _, pc_mn = compute_axis(mn_avail)
new_axes.append(('month->num', pc_mn, 'TEMPORAL'))

PREV = [
    ("country->demonym", 0.563, "SEMANTIC"),
    ("+est (sup)",       0.436, "INFL"),
    ("+er (comp)",       0.393, "INFL"),
    ("elem:single-lett", 0.390, "SEMANTIC"),
    ("country->cap",     0.317, "SEMANTIC"),
    ("past_irr",         0.230, "INFL"),
    ("gender",           0.213, "INFL"),
    ("+ness",            0.211, "DERIV"),
    ("+ed (past_r)",     0.174, "INFL"),
    ("+s plural",        0.155, "INFL"),
    ("+ment",            0.124, "DERIV"),
    ("un-",              0.121, "DERIV"),
    ("+ful",             0.104, "DERIV"),
    ("word->antonym",    0.020, "SEMANTIC"),
    ("month (consec)",  -0.090, "TEMPORAL"),
    ("weekday (consec)",-0.153, "TEMPORAL"),
]

all_axes = new_axes + PREV
all_axes.sort(key=lambda x: -(x[1] if x[1] is not None else -99))
print()
print("  %-28s  pc_cos   type" % "Axis")
print("  " + "-"*52)
for name, pc, atype in all_axes:
    if pc is None or (isinstance(pc, float) and pc != pc): continue
    print("  %-28s  %+.4f   %s" % (name, pc, atype))
