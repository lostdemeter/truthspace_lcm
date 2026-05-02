#!/usr/bin/env python3
"""
Expedition Day 26 — The Frequency Law

Hypothesis from Day 25:
  The 'gender_pair' gravitational body is not 'gendered words'. It is the
  high-frequency short concrete noun attractor. fox, wolf, bear, bread, lung,
  heart, train all show φ_cos ≈ 0.995 to the gender_pair centroid — the same
  as man, woman, king, queen.

  Day 26 tests whether this is a quantitative law:
      φ_cos(word, gender_pair) ≈ f(token_id, word_length, syllables)

Proxy variables (no external corpus required):
  1. token_id  — BPE assigns lower IDs to higher-frequency tokens; log(token_id)
                  is therefore an inverse-frequency proxy (lower = more frequent)
  2. word_length — character count; shorter words are typically more frequent
  3. n_syllables  — vowel-group count; fewer syllables → more frequent

Key predictions:
  P1: φ_cos(word, gender_pair) correlates NEGATIVELY with log(token_id)
      (more frequent → higher attraction to common-noun body)
  P2: φ_cos(word, gender_pair) correlates NEGATIVELY with word_length
  P3: The distribution of φ_cos(word, gender_pair) is BIMODAL, not continuous
      (either ~0.995 for common words OR ~0.5-0.7 for specialized words)
  P4: A simple threshold (token_id < K OR word_length < L) predicts cluster
      membership with high accuracy
  P5: The same frequency law holds for OTHER attractors:
      φ_cos(word, city_asia) correlates with token_id in the OTHER direction
      (rare proper nouns like city names are attracted to city_asia)

If P1–P3 hold, this is a quantitative law: the model's φ-space is a
frequency-ordered manifold. The 'gravitational body' a word belongs to is
largely determined by how often and in what contexts it appears in training.
"""

import sys, os, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
MID_COMB     = 14
CRYST_LAYER  = 2

SEEDS = {
    'city_europe':   ['berlin', 'paris', 'madrid', 'vienna', 'london', 'rome'],
    'city_asia':     ['tokyo', 'beijing', 'seoul', 'mumbai', 'bangkok'],
    'city_other':    ['cairo', 'sydney', 'nairobi'],
    'animal_large':  ['elephant', 'rhinoceros', 'hippopotamus', 'giraffe'],
    'animal_primate':['chimpanzee', 'gorilla', 'orangutan'],
    'animal_marine': ['dolphin', 'whale', 'octopus'],
    'animal_bird':   ['penguin', 'eagle', 'parrot'],
    'animal_reptile':['crocodile', 'python', 'iguana'],
    'elem_noble':    ['helium', 'neon', 'argon'],
    'elem_atm':      ['nitrogen', 'oxygen'],
    'elem_solid':    ['carbon', 'silicon', 'sulfur'],
    'elem_metal':    ['iron', 'copper', 'gold', 'silver'],
    'elem_reactive': ['hydrogen', 'sodium', 'potassium'],
    'plural':        ['cats', 'dogs', 'trees', 'birds', 'houses'],
    'gender_pair':   ['man', 'woman', 'king', 'queen', 'boy', 'girl'],
    'comparative':   ['bigger', 'faster', 'older'],
}

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# Extended word set for frequency law testing (Day 23 + Day 25 words + extras)
EXTENDED_WORDS = [
    # Very common (token_id < 2000)
    'man','woman','king','queen','boy','girl','cat','dog','bird','tree','house',
    'fish','fox','wolf','bear','lion','eel','bread','gold','silver','iron',
    'heart','brain','lung','bone','skin','hand','foot','eye','ear','nose',
    'fire','water','earth','wind','sun','moon','star','sky','sea','land',
    'red','blue','green','black','white','long','short','hard','soft','hot',
    'big','fast','old','new','good','bad','high','low','rich','poor',
    # Moderately common (2000–10000)
    'train','tractor','bread','wolf','bear','fox','lion','eel',
    'copper','sulfur','carbon','silicon','oxygen','nitrogen','hydrogen',
    'paris','london','berlin','tokyo','beijing','rome','cairo','sydney',
    'eagle','whale','dolphin','penguin','parrot','octopus','crocodile',
    'elephant','giraffe','gorilla','orangutan','python','iguana',
    'piano','violin','guitar','trumpet','flute','cello',
    'bicycle','motorcycle','submarine','helicopter','canoe','yacht',
    'bread','pasta','cheese','butter','tomato','potato','garlic','pepper',
    'liver','kidney','stomach','muscle','spine','artery',
    'freedom','justice','courage','wisdom','mercy','tyranny',
    # Rare (token_id > 10000)
    'hippopotamus','rhinoceros','orangutan','chimpanzee','mongoose',
    'saxophone','trombone','bassoon','clarinet','oboe','mandolin',
    'catamaran','zeppelin','gondola','trolley',
    'trachea','pancreas','thyroid','bladder','intestine',
    'tsunami','monsoon','avalanche','blizzard','cyclone',
    'magnesium','lithium','calcium','uranium','manganese','fluorine',
    'bromine','iodine','plutonium',
    'amsterdam','lisbon','budapest','warsaw','brussels','oslo','zurich',
    'istanbul','dubai','singapore','jakarta','manila','taipei',
    'toronto','chicago','houston','miami','seattle','bogota','havana',
    'leopard','cheetah','jaguar','koala','panda',
    'surgeon','dentist','architect','economist','geologist','biologist',
    'historian','diplomat','astronomer','philosopher','linguist',
    'democracy','sovereignty','solidarity','equality','liberty','dignity',
    'volcano','glacier','hurricane','tornado','drought','wildfire','earthquake',
    'algorithm','parliament','archipelago','metabolism','symphony',
    'cathedral','chromosome','constellation','renaissance','monastery',
]


def count_syllables(word):
    """Approximate syllable count via vowel groups."""
    word = word.lower()
    count = len(re.findall(r'[aeiou]+', word))
    return max(1, count)


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20: return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_hidden_states(model, tok, word):
    import torch
    for variant in (' ' + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if ids:
            target_id = ids[0]; break
    else:
        return None
    inputs = tok(word, return_tensors='pt')
    id_list = inputs['input_ids'][0]
    pos = next((i for i, t in enumerate(id_list) if t.item() == target_id),
               len(id_list) - 1)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return np.stack([hs[0, pos, :].numpy() for hs in out.hidden_states])


def phi_vec(h, z2_axis):
    hn   = h / (np.linalg.norm(h) + 1e-20)
    z2v  = float(np.dot(hn, z2_axis))
    perp = hn - z2v * z2_axis
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20), pm, z2v


if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import pearsonr, spearmanr

    print(f"  Loading {SMALL_MODEL}...")
    tok   = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    n_layers = model.config.num_hidden_layers

    # Deduplicate extended words
    all_words = list(dict.fromkeys(
        w for ws in SEEDS.values() for w in ws
    ) | dict.fromkeys(EXTENDED_WORDS).keys())
    all_words = list(dict.fromkeys(all_words))  # preserve order, deduplicate

    print(f"  Caching {len(all_words)} words...")
    cache = {}
    for w in sorted(set(all_words)):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs
    print(f"  Cached {len(cache)} words.\n")

    # Build Z2 axis
    comb_deltas = []
    for a, b in KILLING_PAIRS:
        for L in range(CRYST_LAYER, n_layers - 2):
            if a in cache and b in cache:
                d = cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                comb_deltas.append(d / (np.linalg.norm(d) + 1e-20))
    _, sv, Vt = np.linalg.svd(np.stack(comb_deltas), full_matrices=False)
    z2_axis = Vt[0]
    print(f"  Z2: {100*sv[0]**2/np.sum(sv**2):.2f}% variance\n")

    # Seed centroids
    phi_cache = {w: phi_vec(cache[w][MID_COMB].astype(np.float64), z2_axis)
                 for w in cache}
    seed_centroids = {}
    for sname, words in SEEDS.items():
        vecs = [phi_cache[w][0] for w in words if w in phi_cache]
        if vecs:
            c = np.mean(vecs, axis=0)
            seed_centroids[sname] = c / (np.linalg.norm(c) + 1e-20)

    # Collect proxy variables for all cached words
    print(f"{'='*70}")
    print(f"DAY 26 — The Frequency Law")
    print(f"{'='*70}")

    rows = []
    for w in sorted(cache.keys()):
        ids = tok.encode(' ' + w, add_special_tokens=False)
        if not ids: continue
        token_id  = ids[0]
        wlen      = len(w)
        sylls     = count_syllables(w)
        log_tid   = float(np.log1p(token_id))

        phi_w = phi_cache[w][0]
        sims  = {s: cos_sim(phi_w, c) for s, c in seed_centroids.items()}

        rows.append({
            'word':        w,
            'token_id':    token_id,
            'log_tid':     log_tid,
            'wlen':        wlen,
            'sylls':       sylls,
            **{f'phi_{s}': v for s, v in sims.items()},
        })

    words_list     = [r['word']      for r in rows]
    token_ids      = np.array([r['token_id']  for r in rows])
    log_tids       = np.array([r['log_tid']   for r in rows])
    wlens          = np.array([r['wlen']      for r in rows])
    syllsarr       = np.array([r['sylls']     for r in rows])
    phi_gp         = np.array([r['phi_gender_pair']   for r in rows])
    phi_ca         = np.array([r['phi_city_asia']      for r in rows])
    phi_ce         = np.array([r['phi_city_europe']    for r in rows])
    phi_am         = np.array([r['phi_animal_marine']  for r in rows])
    phi_er         = np.array([r['phi_elem_reactive']  for r in rows])
    phi_al         = np.array([r['phi_animal_large']   for r in rows])
    phi_ab         = np.array([r['phi_animal_bird']    for r in rows])

    # ── Section 1: Distribution of φ_cos to gender_pair ──────────────────────
    print(f"\n── Section 1: Distribution of φ_cos(word, gender_pair) ──────────────")
    print(f"  Is it bimodal? Histogram:\n")

    bins = np.linspace(0, 1.05, 22)
    hist, edges = np.histogram(phi_gp, bins=bins)
    print(f"  φ_cos range  count  bar")
    print("  " + "─" * 50)
    for i in range(len(hist)):
        bar = '█' * (hist[i] // 2)
        lo, hi = edges[i], edges[i+1]
        if hist[i] > 0:
            print(f"  [{lo:.2f}-{hi:.2f}]  {hist[i]:5d}  {bar}")
    print(f"\n  Mean: {phi_gp.mean():.4f}   Std: {phi_gp.std():.4f}")
    print(f"  Fraction with φ_cos > 0.90: {(phi_gp > 0.90).mean():.1%}")
    print(f"  Fraction with φ_cos > 0.95: {(phi_gp > 0.95).mean():.1%}")
    print(f"  Fraction with φ_cos < 0.80: {(phi_gp < 0.80).mean():.1%}")

    # Gap analysis
    high_gp = phi_gp > 0.90
    low_gp  = phi_gp < 0.80
    mid_gp  = ~high_gp & ~low_gp
    print(f"\n  HIGH group (φ_cos>0.90): {high_gp.sum()} words")
    print(f"  MID  group (0.80-0.90):  {mid_gp.sum()} words")
    print(f"  LOW  group (φ_cos<0.80): {low_gp.sum()} words")
    if high_gp.sum() > 0 and low_gp.sum() > 0:
        gap = phi_gp[high_gp].min() - phi_gp[low_gp].max()
        print(f"  Gap between HIGH min and LOW max: {gap:.4f}")

    # ── Section 2: Correlations ────────────────────────────────────────────────
    print(f"\n── Section 2: Correlation — frequency proxies vs φ attraction ────────")
    print(f"\n  Target: φ_cos(word, gender_pair)  [prediction: high-freq words → high φ_cos]")
    print(f"  {'Variable':<22} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} Direction")
    print("  " + "─"*68)

    for varname, arr in [('log(token_id)', log_tids), ('word_length', wlens),
                         ('syllables', syllsarr), ('token_id', token_ids)]:
        r_p, p_p = pearsonr(arr, phi_gp)
        r_s, p_s = spearmanr(arr, phi_gp)
        direction = '← high freq→high φ' if r_p < -0.3 else \
                    ('→ high freq→low φ' if r_p > 0.3 else '~ no clear direction')
        print(f"  {varname:<22} {r_p:+.4f}      {p_p:<12.4g} {r_s:+.4f}      {direction}")

    print(f"\n  Target: φ_cos(word, city_asia)  [prediction: rare proper nouns → high]")
    print(f"  {'Variable':<22} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ'}")
    print("  " + "─"*58)
    for varname, arr in [('log(token_id)', log_tids), ('word_length', wlens),
                         ('syllables', syllsarr)]:
        r_p, p_p = pearsonr(arr, phi_ca)
        r_s, p_s = spearmanr(arr, phi_ca)
        print(f"  {varname:<22} {r_p:+.4f}      {p_p:<12.4g} {r_s:+.4f}")

    print(f"\n  Target: φ_cos(word, elem_reactive)  [prediction: intermediate freq]")
    print(f"  {'Variable':<22} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ'}")
    print("  " + "─"*58)
    for varname, arr in [('log(token_id)', log_tids), ('word_length', wlens),
                         ('syllables', syllsarr)]:
        r_p, p_p = pearsonr(arr, phi_er)
        r_s, p_s = spearmanr(arr, phi_er)
        print(f"  {varname:<22} {r_p:+.4f}      {p_p:<12.4g} {r_s:+.4f}")

    # ── Section 3: HIGH group vs LOW group characterisation ───────────────────
    print(f"\n── Section 3: HIGH vs LOW gender_pair group profiles ────────────────")
    for label, mask in [('HIGH (φ_cos>0.90)', high_gp), ('LOW (φ_cos<0.80)', low_gp)]:
        if mask.sum() == 0: continue
        print(f"\n  {label}  (n={mask.sum()})")
        print(f"    mean token_id:  {token_ids[mask].mean():.0f}  "
              f"(median {np.median(token_ids[mask]):.0f})")
        print(f"    mean word_len:  {wlens[mask].mean():.1f}  "
              f"(median {np.median(wlens[mask]):.1f})")
        print(f"    mean syllables: {syllsarr[mask].mean():.1f}")
        print(f"    top-20 words: "
              f"{', '.join(sorted([words_list[i] for i in np.where(mask)[0]], key=lambda w: token_ids[words_list.index(w)])[:20])}")

    # ── Section 4: Threshold discovery ────────────────────────────────────────
    print(f"\n── Section 4: Threshold discovery ────────────────────────────────────")
    print(f"  Find the token_id and word_length thresholds that best separate")
    print(f"  HIGH (φ_cos>0.90) from LOW (φ_cos<0.80) groups.\n")

    true_labels = (phi_gp > 0.90).astype(int)  # 1=HIGH, 0=LOW (ignoring MID)
    mask_not_mid = ~mid_gp

    best_tid_thresh = None; best_tid_acc = 0
    for thr in range(500, 20000, 200):
        pred = (token_ids < thr).astype(int)
        acc  = (pred[mask_not_mid] == true_labels[mask_not_mid]).mean()
        if acc > best_tid_acc:
            best_tid_acc = acc; best_tid_thresh = thr

    best_len_thresh = None; best_len_acc = 0
    for thr in range(3, 16):
        pred = (wlens <= thr).astype(int)
        acc  = (pred[mask_not_mid] == true_labels[mask_not_mid]).mean()
        if acc > best_len_acc:
            best_len_acc = acc; best_len_thresh = thr

    best_syl_thresh = None; best_syl_acc = 0
    for thr in range(1, 8):
        pred = (syllsarr <= thr).astype(int)
        acc  = (pred[mask_not_mid] == true_labels[mask_not_mid]).mean()
        if acc > best_syl_acc:
            best_syl_acc = acc; best_syl_thresh = thr

    print(f"  Best token_id threshold:  token_id < {best_tid_thresh}  "
          f"accuracy={100*best_tid_acc:.1f}%")
    print(f"  Best word_length thresh:  len ≤ {best_len_thresh}        "
          f"accuracy={100*best_len_acc:.1f}%")
    print(f"  Best syllable threshold:  sylls ≤ {best_syl_thresh}      "
          f"accuracy={100*best_syl_acc:.1f}%")

    # Show misclassified words at each threshold
    for thresh_name, pred_mask, thresh_val, acc in [
        (f'token_id<{best_tid_thresh}', token_ids < best_tid_thresh, best_tid_thresh, best_tid_acc),
        (f'len≤{best_len_thresh}', wlens <= best_len_thresh, best_len_thresh, best_len_acc),
        (f'sylls≤{best_syl_thresh}', syllsarr <= best_syl_thresh, best_syl_thresh, best_syl_acc),
    ]:
        false_pos = [words_list[i] for i in np.where(pred_mask & ~mask_not_mid == False)[0]
                     if pred_mask[i] and phi_gp[i] < 0.80]
        false_neg = [words_list[i] for i in np.where(~pred_mask & mask_not_mid)[0]
                     if not pred_mask[i] and phi_gp[i] > 0.90]
        if false_pos or false_neg:
            print(f"\n  [{thresh_name}] Misclassified HIGH words (false negatives): "
                  f"{', '.join(false_neg[:10])}")
            print(f"  [{thresh_name}] Misclassified LOW words (false positives): "
                  f"{', '.join(false_pos[:10])}")

    # ── Section 5: The quantitative law ────────────────────────────────────────
    print(f"\n── Section 5: The quantitative law ──────────────────────────────────")
    print(f"  Fit: φ_cos(gender_pair) = a·log(token_id) + b·word_len + c·syllables + d\n")

    from scipy.stats import linregress

    X = np.column_stack([log_tids, wlens, syllsarr, np.ones(len(log_tids))])
    y = phi_gp
    # Ordinary least squares
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    R2 = 1 - ss_res / ss_tot
    print(f"  φ_cos(gender_pair) = "
          f"{beta[0]:.4f}·log(token_id) + "
          f"{beta[1]:.4f}·word_len + "
          f"{beta[2]:.4f}·syllables + "
          f"{beta[3]:.4f}")
    print(f"  R² = {R2:.4f}")
    if R2 > 0.5:
        print(f"  ✓ STRONG FIT — frequency proxies explain {100*R2:.1f}% of φ-attraction variance")
    elif R2 > 0.25:
        print(f"  ~ MODERATE FIT — {100*R2:.1f}% explained")
    else:
        print(f"  ✗ WEAK FIT — frequency proxies alone don't explain φ-attraction")

    # Which predictor dominates?
    # Standardise
    Xs = (X[:, :3] - X[:, :3].mean(0)) / (X[:, :3].std(0) + 1e-8)
    beta_std = np.linalg.pinv(Xs.T @ Xs) @ Xs.T @ y
    print(f"\n  Standardised coefficients (dominance ordering):")
    for name, b in zip(['log(token_id)', 'word_length', 'syllables'], beta_std):
        bar = '█' * int(abs(b) * 30)
        print(f"    {name:<20} β={b:+.4f}  {bar}")

    # ── Section 6: Generalisation to other attractors ─────────────────────────
    print(f"\n── Section 6: Frequency law generalisation ───────────────────────────")
    print(f"  Does log(token_id) predict attraction to OTHER gravitational bodies?\n")
    print(f"  {'Body':<22} {'r(log_tid,φ)':<15} {'r(wlen,φ)':<15} {'dominant predictor'}")
    print("  " + "─"*65)

    for body_name, phi_arr in [
        ('gender_pair',    phi_gp),
        ('city_asia',      phi_ca),
        ('city_europe',    phi_ce),
        ('animal_marine',  phi_am),
        ('elem_reactive',  phi_er),
        ('animal_large',   phi_al),
        ('animal_bird',    phi_ab),
    ]:
        r_tid, _ = pearsonr(log_tids, phi_arr)
        r_len, _ = pearsonr(wlens,    phi_arr)
        if abs(r_tid) > abs(r_len):
            dom = f'log(token_id) r={r_tid:+.3f}'
        else:
            dom = f'word_length  r={r_len:+.3f}'
        print(f"  {body_name:<22} {r_tid:+.4f}          {r_len:+.4f}          {dom}")

    # ── Section 7: Sample table — sorted by token_id ──────────────────────────
    print(f"\n── Section 7: Sample — sorted by token_id with φ-attraction ─────────")
    print(f"  {'Word':<16} {'tok_id':<8} {'len':<5} {'syl':<5} φ_gp   φ_city  φ_elem  "
          f"body")
    print("  " + "─"*78)

    sorted_rows = sorted(rows, key=lambda r: r['token_id'])
    show_indices = list(range(0, min(25, len(sorted_rows)))) + \
                   list(range(max(0, len(sorted_rows)//2 - 3),
                              min(len(sorted_rows)//2 + 3, len(sorted_rows)))) + \
                   list(range(max(0, len(sorted_rows) - 10), len(sorted_rows)))
    seen = set()
    prev_tid = -1
    for idx in show_indices:
        r = sorted_rows[idx]
        if r['word'] in seen: continue
        seen.add(r['word'])
        if prev_tid > 0 and r['token_id'] - prev_tid > 3000:
            print("  ...")
        prev_tid = r['token_id']
        gp   = r['phi_gender_pair']
        ca   = r['phi_city_asia']
        er   = r['phi_elem_reactive']
        best = max(r, key=lambda k: r[k] if k.startswith('phi_') else -1)
        body = best.replace('phi_', '') if best.startswith('phi_') else '?'
        hi   = '★' if gp > 0.90 else ' '
        print(f"  {hi}{r['word']:<15} {r['token_id']:<8} {r['wlen']:<5} "
              f"{r['sylls']:<5} {gp:.3f}  {ca:.3f}  {er:.3f}  {body}")

    # ── Section 8: Summary ────────────────────────────────────────────────────
    print(f"\n── Section 8: Summary ────────────────────────────────────────────────")
    print(f"""
  Words analysed:        {len(rows)}
  HIGH group (φ_gp>0.90): {high_gp.sum()} words
  LOW group (φ_gp<0.80):  {low_gp.sum()} words
  MID group (0.80-0.90):  {mid_gp.sum()} words
  R² (3-predictor fit):   {R2:.4f}
  Best single threshold:  {'token_id' if best_tid_acc >= best_len_acc else 'word_len'} = {best_tid_thresh if best_tid_acc >= best_len_acc else best_len_thresh}
  Best threshold accuracy:{100*max(best_tid_acc, best_len_acc):.1f}%

  The frequency law:
    High-frequency short words → attracted to gender_pair body (φ_cos~0.99)
    Low-frequency long words   → attracted to specific semantic bodies
    The dividing line is approximately token_id={best_tid_thresh} or word_len≤{best_len_thresh}

  This means the model's φ-space has a HIERARCHICAL structure:
    Level 1: frequency class (common vs specialized)
    Level 2: semantic body (for specialized words only)
    """)

    print(f"{'='*70}")
    print(f"Day 26 complete.")
    print(f"{'='*70}")
