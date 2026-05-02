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
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=80):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 297: TEMPORAL SEQUENCE AXES")
print("="*65)
print("Testing: month ordering, weekday ordering, ordinal numbers,")
print("century ordering. Hypothesis: ordinal sequences have very")
print("high pc (>0.40) because each step is semantically equidistant.")
print()

# ====================================================================
# TOKEN AVAILABILITY CHECK
# ====================================================================
print("Token availability check:")
MONTHS  = ['January','February','March','April','May','June',
           'July','August','September','October','November','December']
WEEKDAYS= ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
ORDINALS= ['first','second','third','fourth','fifth','sixth',
           'seventh','eighth','ninth','tenth']
NUMBERS = ['one','two','three','four','five','six','seven','eight','nine','ten',
           'eleven','twelve']

for group_name, words in [('Months', MONTHS), ('Weekdays', WEEKDAYS),
                           ('Ordinals', ORDINALS), ('Numbers', NUMBERS)]:
    available = []
    for w in words:
        _, sid = get_emb(w)
        tok_raw = tok(' '+w, add_special_tokens=False)['input_ids']
        available.append((w, sid is not None, len(tok_raw)))
    avail_count = sum(1 for _,a,_ in available if a)
    print("  %-10s  %d/%d single-token" % (group_name, avail_count, len(words)))
    for w, a, n in available:
        print("    %-12s  %s  (%d tokens)" % (w, 'OK' if a else '--', n))
    print()

# ====================================================================
# PART A: MONTH SEQUENCE AXIS (Jan->Feb, Feb->Mar, etc.)
# ====================================================================
print("PART A: Month sequence axis")
print("-"*65)

# Build consecutive month pairs
month_pairs_consec = [(MONTHS[i], MONTHS[i+1]) for i in range(len(MONTHS)-1)]
# Add wrap-around
month_pair_wrap = (MONTHS[-1], MONTHS[0])  # December -> January

# Only use pairs where both are single-token
month_pairs_avail = [(s,t) for s,t in month_pairs_consec
                     if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available consecutive pairs: %d/%d" % (len(month_pairs_avail), len(month_pairs_consec)))

ax_m, coh_m, valid_m, pc_m = compute_axis(month_pairs_avail)
if ax_m is not None:
    s_m, acc_m = best_scale(ax_m, valid_m)
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_m, coh_m, s_m, acc_m, len(valid_m), 100*acc_m/max(1,len(valid_m))))
    print()
    print("  Full month sequence results:")
    for s, t in month_pairs_avail:
        es, sid = get_emb(s)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+s_m*ax_m, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-12s -> %-12s  got=%-12s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()
    
    # Test wrap-around: December -> January
    es_dec, sid_dec = get_emb('December')
    et_jan, tid_jan = get_emb('January')
    if es_dec is not None and et_jan is not None:
        r_wrap = nn_retrieve(W_E[sid_dec]+s_m*ax_m, [sid_dec], top_n=3)
        chord_wrap = normed(W_E[tid_jan] - W_E[sid_dec]).astype(np.float32)
        cos_wrap = float(np.dot(chord_wrap, ax_m.astype(np.float32)))
        print("  Wrap-around test (Dec->Jan):")
        print("    Prediction: got=%s  (target=January)" % r_wrap[0][0])
        print("    cos(Dec->Jan chord, axis) = %.4f  (1.0=consistent, -1.0=reversed)")
        print()

# ====================================================================
# PART B: WEEKDAY SEQUENCE AXIS (Mon->Tue, etc.)
# ====================================================================
print("PART B: Weekday sequence axis")
print("-"*65)

weekday_pairs_consec = [(WEEKDAYS[i], WEEKDAYS[i+1]) for i in range(len(WEEKDAYS)-1)]
weekday_pairs_avail = [(s,t) for s,t in weekday_pairs_consec
                       if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available consecutive pairs: %d/%d" % (len(weekday_pairs_avail), len(weekday_pairs_consec)))

ax_w, coh_w, valid_w, pc_w = compute_axis(weekday_pairs_avail)
if ax_w is not None:
    s_w, acc_w = best_scale(ax_w, valid_w)
    print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
        pc_w, coh_w, s_w, acc_w, len(valid_w), 100*acc_w/max(1,len(valid_w))))
    print()
    print("  Full weekday sequence results:")
    for s, t in weekday_pairs_avail:
        es, sid = get_emb(s)
        if es is None: continue
        r = nn_retrieve(W_E[sid]+s_w*ax_w, [sid], top_n=3)
        hit = r[0][0] == t
        print("    %-12s -> %-12s  got=%-12s [%s]  (also: %s, %s)" % (
            s, t, r[0][0], 'HIT' if hit else '---', r[1][0], r[2][0]))
    print()

# ====================================================================
# PART C: MONTH ABBREVIATION PAIRS (Jan, Feb, Mar...)
# ====================================================================
print("PART C: Month abbreviations (Jan->Feb sequence)")
print("-"*65)

MONTHS_ABBR = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
abbr_pairs = [(MONTHS_ABBR[i], MONTHS_ABBR[i+1]) for i in range(len(MONTHS_ABBR)-1)]
abbr_avail = [(s,t) for s,t in abbr_pairs
              if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available abbreviated pairs: %d/%d" % (len(abbr_avail), len(abbr_pairs)))

if abbr_avail:
    ax_a, coh_a, valid_a, pc_a = compute_axis(abbr_avail)
    if ax_a is not None:
        s_a, acc_a = best_scale(ax_a, valid_a)
        print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
            pc_a, coh_a, s_a, acc_a, len(valid_a), 100*acc_a/max(1,len(valid_a))))
        for s, t in abbr_avail:
            es, sid = get_emb(s)
            if es is None: continue
            r = nn_retrieve(W_E[sid]+s_a*ax_a, [sid], top_n=3)
            hit = r[0][0] == t
            print("    %-6s -> %-6s  got=%-6s [%s]" % (
                s, t, r[0][0], 'HIT' if hit else '---'))
print()

# ====================================================================
# PART D: MONTH NUMBER AXIS (January=1, February=2, ...)
# Cross-domain: month_name -> ordinal_number
# ====================================================================
print("PART D: Month-to-number axis (semantic, non-consecutive)")
print("-"*65)

MONTH_NUM_PAIRS = [
    ('January','1'),('February','2'),('March','3'),
    ('April','4'),('May','5'),('June','6'),
    ('July','7'),('August','8'),('September','9'),
    ('October','10'),('November','11'),('December','12'),
]
# Check number token availability
for w in ['1','2','3','4','5','6','7','8','9','10','11','12']:
    _, sid = get_emb(w)
    print("  token('%s') = %s" % (w, 'OK' if sid is not None else '--'))
print()

mn_avail = [(s,t) for s,t in MONTH_NUM_PAIRS
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available month->number pairs: %d/%d" % (len(mn_avail), len(MONTH_NUM_PAIRS)))

if mn_avail:
    ax_mn, coh_mn, valid_mn, pc_mn = compute_axis(mn_avail)
    if ax_mn is not None:
        s_mn, acc_mn = best_scale(ax_mn, valid_mn)
        print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
            pc_mn, coh_mn, s_mn, acc_mn, len(valid_mn), 100*acc_mn/max(1,len(valid_mn))))
print()

# ====================================================================
# PART E: SKIP-ONE MONTH AXIS (Jan->Mar, Feb->Apr, etc.)
# Tests whether the sequence axis is linear at different step sizes
# ====================================================================
print("PART E: Skip-one month axis (Jan->Mar, Feb->Apr, ...)")
print("-"*65)

skip_pairs = [(MONTHS[i], MONTHS[i+2]) for i in range(len(MONTHS)-2)]
skip_avail = [(s,t) for s,t in skip_pairs
              if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
print("  Available skip-1 pairs: %d/%d" % (len(skip_avail), len(skip_pairs)))

if skip_avail:
    ax_sk, coh_sk, valid_sk, pc_sk = compute_axis(skip_avail)
    if ax_sk is not None and ax_m is not None:
        s_sk, acc_sk = best_scale(ax_sk, valid_sk)
        cos_ms = float(np.dot(ax_m.astype(np.float32), ax_sk.astype(np.float32)))
        print("  pc=%.4f  coh=%.4f  scale=%.2f  train=%d/%d (%.0f%%)" % (
            pc_sk, coh_sk, s_sk, acc_sk, len(valid_sk), 100*acc_sk/max(1,len(valid_sk))))
        print("  cos(consec axis, skip-1 axis) = %.4f" % cos_ms)
        print("  scale_skip / scale_consec = %.3f  (expect ~2.0 if linear)" % (s_sk/max(0.001,s_m)))
        print()
        
        # If linear, skip-1 axis should point in SAME direction as consec axis
        # and scale should be 2x
        if abs(cos_ms) > 0.9:
            print("  CONFIRMED: skip-1 axis aligns with consec axis (cos=%.3f)" % cos_ms)
            print("  Temporal sequence IS geometrically linear in W_E")
        else:
            print("  WARNING: skip-1 axis does NOT align with consec axis")
            print("  Temporal sequence may NOT be geometrically linear")
print()

# ====================================================================
# PART F: MONTH CLUSTER ANALYSIS
# Is there a "temporal order" dimension in month embeddings?
# ====================================================================
print("PART F: Month embedding cluster analysis")
print("-"*65)

month_embs = {}
for m in MONTHS:
    e, sid = get_emb(m)
    if e is not None:
        month_embs[m] = (e, sid)

avail_months = [m for m in MONTHS if m in month_embs]
print("  Available months: %s" % ', '.join(avail_months))

if len(month_embs) >= 6:
    # Pairwise cosine matrix between month embeddings (source words)
    src_vecs = [normed(month_embs[m][0]).astype(np.float32) for m in avail_months]
    n = len(src_vecs)
    src_pc_months = float(np.mean([np.dot(src_vecs[i], src_vecs[j])
                                   for i in range(n) for j in range(i+1,n)]))
    print("  Source pairwise cosine (month embeddings) = %.4f" % src_pc_months)
    print()
    
    # SVD of month embedding matrix — first PC should correlate with month order
    M = np.array([month_embs[m][0] for m in avail_months])
    M_c = M - M.mean(axis=0)
    U, S, Vt = np.linalg.svd(M_c, full_matrices=False)
    
    # Project each month onto first 3 PCs
    print("  Month projections onto first 3 principal components:")
    print("  (If PC1 correlates with month order, the sequence is linear in W_E)")
    proj = U[:, :3] * S[:3]  # shape (n_months, 3)
    for i, m in enumerate(avail_months):
        print("    %-12s  PC1=%+.3f  PC2=%+.3f  PC3=%+.3f" % (
            m, proj[i,0], proj[i,1], proj[i,2]))
    print()
    
    # Compute correlation between month index and PC1
    month_idx = [MONTHS.index(m) for m in avail_months]
    from scipy.stats import pearsonr, spearmanr
    r_pc1, p_pc1 = pearsonr(month_idx, proj[:,0])
    r_pc2, p_pc2 = pearsonr(month_idx, proj[:,1])
    rho_pc1, p_rho = spearmanr(month_idx, proj[:,0])
    print("  Pearson r(month_idx, PC1) = %.4f  p=%.4f" % (r_pc1, p_pc1))
    print("  Pearson r(month_idx, PC2) = %.4f  p=%.4f" % (r_pc2, p_pc2))
    print("  Spearman ρ(month_idx, PC1) = %.4f  p=%.4f" % (rho_pc1, p_rho))
    print()
    
    # Is there a circular (cyclic) structure?
    # Project onto first 2 PCs and see if they trace a circle
    if len(avail_months) >= 8:
        angles = np.arctan2(proj[:,1], proj[:,0])
        print("  Polar angles (PC1, PC2 plane):")
        for i, m in enumerate(avail_months):
            print("    %-12s  θ=%+.2f rad (%+.0f°)" % (m, angles[i], np.degrees(angles[i])))

# ====================================================================
# PART G: WEEKDAY CLUSTER ANALYSIS
# ====================================================================
print()
print("PART G: Weekday embedding cluster analysis")
print("-"*65)

weekday_embs = {}
for w in WEEKDAYS:
    e, sid = get_emb(w)
    if e is not None:
        weekday_embs[w] = (e, sid)

avail_days = [w for w in WEEKDAYS if w in weekday_embs]
print("  Available weekdays: %s" % ', '.join(avail_days))

if len(weekday_embs) >= 5:
    src_vecs_w = [normed(weekday_embs[w][0]).astype(np.float32) for w in avail_days]
    n = len(src_vecs_w)
    src_pc_days = float(np.mean([np.dot(src_vecs_w[i], src_vecs_w[j])
                                  for i in range(n) for j in range(i+1,n)]))
    print("  Source pairwise cosine (weekday embeddings) = %.4f" % src_pc_days)
    print()
    
    W_days = np.array([weekday_embs[w][0] for w in avail_days])
    W_c = W_days - W_days.mean(axis=0)
    U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
    proj_w = U[:, :3] * S[:3]
    
    print("  Weekday projections onto first 3 PCs:")
    for i, w in enumerate(avail_days):
        print("    %-12s  PC1=%+.3f  PC2=%+.3f  PC3=%+.3f" % (
            w, proj_w[i,0], proj_w[i,1], proj_w[i,2]))
    print()
    
    day_idx = [WEEKDAYS.index(w) for w in avail_days]
    from scipy.stats import pearsonr, spearmanr
    r_pc1_w, p_pc1_w = pearsonr(day_idx, proj_w[:,0])
    rho_pc1_w, p_rho_w = spearmanr(day_idx, proj_w[:,0])
    print("  Pearson r(day_idx, PC1) = %.4f  p=%.4f" % (r_pc1_w, p_pc1_w))
    print("  Spearman ρ(day_idx, PC1) = %.4f  p=%.4f" % (rho_pc1_w, p_rho_w))

# ====================================================================
# PART H: UPDATED LINEARITY SPECTRUM
# ====================================================================
print()
print("="*65)
print("UPDATED LINEARITY SPECTRUM (with temporal sequence axes)")
print("="*65)

new_vals = []
for label, pairs in [
    ('month (consec)', month_pairs_avail),
    ('weekday (consec)', weekday_pairs_avail),
]:
    _, _, _, pc = compute_axis(pairs)
    new_vals.append((label, pc, 'TEMPORAL'))

PREV = [
    ("country->demonym", 0.563, "SEMANTIC"),
    ("country->lang*",   0.474, "SEMANTIC"),
    ("+est (sup)",       0.436, "INFL"),
    ("+er (comp)",       0.393, "INFL"),
    ("elem:single-lett", 0.390, "SEMANTIC"),
    ("country->cap",     0.317, "SEMANTIC"),
    ("animal->class",    0.254, "SEMANTIC"),
    ("person->nat",      0.246, "SEMANTIC"),
    ("past_irr",         0.230, "INFL"),
    ("gender",           0.213, "INFL"),
    ("+ness",            0.211, "DERIV"),
    ("+ed (past_r)",     0.174, "INFL"),
    ("elem:double-lett", 0.163, "SEMANTIC"),
    ("+s plural",        0.155, "INFL"),
    ("element->sym",     0.139, "SEMANTIC"),
    ("in-/im-",          0.133, "DERIV"),
    ("+less",            0.133, "DERIV"),
    ("+tion",            0.130, "DERIV"),
    ("+ment",            0.124, "DERIV"),
    ("un-",              0.121, "DERIV"),
    ("elem:latin-deriv", 0.104, "SEMANTIC"),
    ("+ful",             0.104, "DERIV"),
    ("field->concept",   0.087, "SEMANTIC"),
    ("word->antonym",    0.020, "SEMANTIC"),
]
all_axes = new_vals + PREV
all_axes.sort(key=lambda x: -x[1])
print()
print("  %-28s  pc_cos   type" % "Axis")
print("  " + "-"*52)
for name, pc, atype in all_axes:
    if pc is None or pc != pc: continue  # skip NaN
    print("  %-28s  %.4f   %s" % (name, pc, atype))
