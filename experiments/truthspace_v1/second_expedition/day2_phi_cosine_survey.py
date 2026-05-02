"""
SECOND EXPEDITION — DAY 2
=========================
The φ-Cosine Survey

Day 1 found that the mean rotation angle for several semantic axes falls
within 1° of arccos(1/φⁿ). Translated into cosine space, this means:

  cos(e_norm(src), e_norm(tgt)) ≈ 1/φⁿ  for integer n

If this is a real law and not small-sample noise, it implies the inner
product between semantically related word pairs is QUANTIZED at Fibonacci
fractions of unity:

  n=1:  cos ≈ 0.618  (= 1/φ,  the golden ratio conjugate)
  n=2:  cos ≈ 0.382  (= 1/φ²)
  n=3:  cos ≈ 0.236  (= 1/φ³)
  n=4:  cos ≈ 0.146  (= 1/φ⁴)
  n→∞: cos ≈ 0      (random/orthogonal)

Questions for this day:
  1. Does the φ-quantization hold with 30-50 pairs per axis? (Phase 1)
  2. What is the full vocabulary cosine distribution — any peaks at φ-levels? (Phase 2)
  3. What φ-level do antonyms, synonyms, hypernyms occupy? (Phase 3)
  4. Can we IDENTIFY semantically related pairs by their φ-level alone? (Phase 4)
  5. Do the φ-levels predict something about retrieval accuracy? (Phase 5)

Darwin's rule: follow the most surprising finding wherever it leads.

Script: second_expedition/day2_phi_cosine_survey.py
"""

import torch, numpy as np, sys

print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n = (W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
print(f"  shape={W_E.shape}")

EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(len(W_E))], dtype=bool)

ZH_MASK = np.array([
    any('\u4e00' <= c <= '\u9fff' for c in tok.decode([i]).strip())
    for i in range(len(W_E))], dtype=bool)

# ── utilities ──────────────────────────────────────────────────────────────────
PHI = (1 + 5**0.5) / 2
PHI_LEVELS = {n: 1.0 / PHI**n for n in range(0, 9)}  # 1/φⁿ for n=0..8

def normed(v): return v / (np.linalg.norm(v) + 1e-12)

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def get_emb_any(word):
    ids = tok(word, add_special_tokens=False)['input_ids']
    if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def cos_pair(w1, w2, gf1=get_emb, gf2=get_emb):
    e1, _ = gf1(w1);  e2, _ = gf2(w2)
    if e1 is None or e2 is None: return None
    return float(np.dot(normed(e1), normed(e2)))

def phi_level(cos_val):
    """Return the best-fit φ-level n such that 1/φⁿ ≈ cos_val."""
    if cos_val <= 0: return None
    n = -np.log(cos_val) / np.log(PHI)
    return float(n)

def nearest_phi_n(cos_val):
    """Return (n, residual) where 1/φⁿ is nearest to cos_val."""
    if cos_val <= 0: return (None, abs(cos_val))
    n_float = -np.log(max(cos_val, 1e-9)) / np.log(PHI)
    n_round = round(n_float)
    residual = abs(cos_val - PHI_LEVELS.get(n_round, 0))
    return (n_round, residual)

def print_pair_table(pairs, get_fn1=get_emb, get_fn2=get_emb, label=""):
    cosines = []
    print(f"\n  {label}:")
    print(f"    {'pair':>28}  {'cos':>7}  {'θ°':>6}  {'φ-level n':>9}  {'residual':>9}")
    print(f"    {'─'*28}  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*9}")
    for s, t in pairs:
        c = cos_pair(s, t, get_fn1, get_fn2)
        if c is None:
            print(f"    {s:>14}→{t:<14}  (not single-token)")
            continue
        th = np.degrees(np.arccos(np.clip(c, -1, 1)))
        n_round, resid = nearest_phi_n(c)
        n_str = f"n={n_round}" if n_round is not None else "  neg"
        flag = " ◀" if resid < 0.03 else ""
        print(f"    {s:>14}→{t:<14}  {c:>7.4f}  {th:>6.2f}°  {n_str:>9}  {resid:>9.4f}{flag}")
        cosines.append(c)
    if cosines:
        arr = np.array(cosines)
        n_mean, _ = nearest_phi_n(arr.mean())
        print(f"    {'─'*60}")
        print(f"    n={len(cosines)} pairs  mean_cos={arr.mean():.4f}  std={arr.std():.4f}  "
              f"mean_θ={np.degrees(np.arccos(arr.mean())):.2f}°  "
              f"nearest_φ_level: n={n_mean}  (1/φⁿ={PHI_LEVELS.get(n_mean,0):.4f})")
    return cosines

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Extended curated pairs — 30+ per axis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 1 — Extended φ-Cosine Test (30+ pairs per axis)")
print(f"  φ-levels: " + "  ".join(f"1/φ^{n}={v:.4f}" for n,v in list(PHI_LEVELS.items())[1:6]))
print("═"*72)

# Large gender set
GENDER_EN = [
    ('man','woman'),('king','queen'),('father','mother'),('son','daughter'),
    ('boy','girl'),('husband','wife'),('uncle','aunt'),('prince','princess'),
    ('brother','sister'),('actor','actress'),('hero','heroine'),('waiter','waitress'),
    ('monk','nun'),('wizard','witch'),('emperor','empress'),('duke','duchess'),
    ('lord','lady'),('god','goddess'),('male','female'),('he','she'),
    ('his','her'),('him','her'),('grandfather','grandmother'),('nephew','niece'),
    ('groom','bride'),('bull','cow'),('cock','hen'),('stallion','mare'),
    ('ram','ewe'),('lion','lioness'),
]

SIZE_EN = [
    ('big','small'),('large','tiny'),('huge','little'),('tall','short'),
    ('long','brief'),('fat','thin'),('wide','narrow'),('heavy','light'),
    ('strong','weak'),('hot','cold'),('fast','slow'),('hard','soft'),
    ('loud','quiet'),('deep','shallow'),('thick','thin'),('rich','poor'),
    ('old','young'),('high','low'),('full','empty'),('bright','dim'),
    ('sharp','blunt'),('rough','smooth'),('wet','dry'),('clean','dirty'),
    ('warm','cool'),('dark','pale'),
]

SENT_EN = [
    ('good','bad'),('happy','sad'),('love','hate'),('beautiful','ugly'),
    ('right','wrong'),('best','worst'),('kind','cruel'),('brave','cowardly'),
    ('honest','dishonest'),('wise','foolish'),('gentle','harsh'),
    ('generous','selfish'),('loyal','treacherous'),('peaceful','violent'),
    ('healthy','sick'),('success','failure'),('hope','despair'),('joy','grief'),
    ('pleasure','pain'),('truth','lie'),('friend','enemy'),('hero','villain'),
    ('angel','devil'),('heaven','hell'),('light','dark'),('life','death'),
]

ANTONYMS_STRICT = [
    ('up','down'),('left','right'),('north','south'),('east','west'),
    ('yes','no'),('true','false'),('on','off'),('open','close'),
    ('start','stop'),('come','go'),('rise','fall'),('buy','sell'),
    ('push','pull'),('give','take'),('win','lose'),('add','subtract'),
    ('enter','exit'),('begin','end'),('create','destroy'),('build','destroy'),
]

SYNONYMS = [
    ('happy','joyful'),('fast','quick'),('big','large'),('smart','intelligent'),
    ('angry','furious'),('sad','unhappy'),('cold','chilly'),('hot','warm'),
    ('begin','start'),('end','finish'),('show','display'),('help','assist'),
    ('talk','speak'),('walk','stroll'),('look','see'),('want','desire'),
    ('house','home'),('road','street'),('child','kid'),('gift','present'),
]

HYPERNYMS = [
    ('dog','animal'),('cat','animal'),('eagle','bird'),('salmon','fish'),
    ('rose','flower'),('oak','tree'),('iron','metal'),('ruby','stone'),
    ('Paris','city'),('London','city'),('jazz','music'),('chess','game'),
    ('sword','weapon'),('knife','weapon'),('surgeon','doctor'),('poet','artist'),
    ('anger','emotion'),('red','color'),('circle','shape'),('piano','instrument'),
]

all_phase1_cos = {}

all_phase1_cos['gender_EN'] = print_pair_table(GENDER_EN, label="EN Gender (30 pairs)")
all_phase1_cos['size_EN']   = print_pair_table(SIZE_EN,   label="EN Size/Polarity (26 pairs)")
all_phase1_cos['sent_EN']   = print_pair_table(SENT_EN,   label="EN Sentiment (26 pairs)")
all_phase1_cos['antonym']   = print_pair_table(ANTONYMS_STRICT, label="Strict Antonyms (directional)")
all_phase1_cos['synonym']   = print_pair_table(SYNONYMS,  label="Synonyms (near-identical meaning)")
all_phase1_cos['hypernym']  = print_pair_table(HYPERNYMS, label="Hypernyms (X is-a Y)")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Full vocabulary cosine distribution
# Sample 600 EN words, compute all pairwise cosines, build histogram
# Look for peaks at φ-levels
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 2 — Vocabulary-Wide Cosine Distribution")
print("  600 English content words × all pairwise cosines")
print("  Looking for peaks at φ-levels in the histogram")
print("═"*72)

rng = np.random.default_rng(42)
en_ids_all = np.where(EN_MASK)[0]
# Sample 600 tokens
sample_ids = rng.choice(en_ids_all, size=600, replace=False)
sample_embs = W_n[sample_ids].astype(np.float64)  # [600, 1536]
# Full pairwise cosine matrix
cos_matrix = sample_embs @ sample_embs.T  # [600, 600]
# Upper triangle (excluding diagonal)
tri_idx = np.triu_indices(600, k=1)
all_cos = cos_matrix[tri_idx]   # 179,700 pairs

print(f"\n  Full distribution ({len(all_cos):,} pairs):")
print(f"  mean={all_cos.mean():.4f}  std={all_cos.std():.4f}  "
      f"median={np.median(all_cos):.4f}")
print(f"  min={all_cos.min():.4f}  max={all_cos.max():.4f}")

# Histogram in bins of 0.02
bins = np.arange(-0.3, 1.01, 0.02)
hist, edges = np.histogram(all_cos, bins=bins)
total = hist.sum()

print(f"\n  Cosine histogram (bin width=0.02):")
print(f"  {'range':>14}  {'count':>8}  {'%':>6}  φ-ref?")
for i, (lo, hi, count) in enumerate(zip(edges[:-1], edges[1:], hist)):
    mid = (lo + hi) / 2
    pct = 100 * count / total
    if pct < 0.1: continue   # skip nearly empty bins
    phi_flag = ""
    for n in range(1, 7):
        pn = PHI_LEVELS[n]
        if lo <= pn < hi:
            phi_flag = f" ← 1/φ^{n}={pn:.4f}"
            break
    bar = "█" * int(pct * 1.5)
    print(f"  [{lo:+.2f},{hi:+.2f})  {count:>8,}  {pct:>6.2f}%  {bar}{phi_flag}")

# Peaks: find local maxima in the cosine distribution
from scipy.signal import find_peaks as _find_peaks
peaks, props = _find_peaks(hist, height=total*0.01, distance=3)
print(f"\n  Detected histogram peaks at cosine bins:")
for p in peaks:
    mid = (edges[p] + edges[p+1]) / 2
    n, resid = nearest_phi_n(mid)
    pct = 100 * hist[p] / total
    print(f"    cos≈{mid:+.3f}  ({pct:.2f}%)  nearest φ-level: 1/φ^{n}={PHI_LEVELS.get(n,0):.4f}  Δ={resid:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: φ-level neighbor identification
# For a set of seed words, find all vocabulary words at each φ-level
# and inspect: are φ-1 neighbors semantically related?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 3 — φ-Level Neighbor Identification")
print("  For each seed word: who are its φ-level-1, φ-level-2, φ-level-3 neighbors?")
print("  φ-level-1: cos ∈ [1/φ ± 0.05] = [0.568, 0.668]")
print("  φ-level-2: cos ∈ [1/φ² ± 0.04] = [0.342, 0.422]")
print("  φ-level-3: cos ∈ [1/φ³ ± 0.03] = [0.206, 0.266]")
print("═"*72)

SEEDS = ['king', 'man', 'good', 'large', 'dog', 'run', 'Paris', 'red', 'love', 'water']

BANDS = {
    1: (PHI_LEVELS[1] - 0.05, PHI_LEVELS[1] + 0.05),  # 0.568-0.668
    2: (PHI_LEVELS[2] - 0.04, PHI_LEVELS[2] + 0.04),  # 0.342-0.422
    3: (PHI_LEVELS[3] - 0.03, PHI_LEVELS[3] + 0.03),  # 0.206-0.266
}

for seed in SEEDS:
    e_seed, idx_seed = get_emb(seed)
    if e_seed is None:
        print(f"\n  {seed}: not single-token")
        continue
    e_seed_n = normed(e_seed).astype(np.float32)
    sims = W_n @ e_seed_n   # cosines with all vocabulary
    sims[idx_seed] = -2.0   # exclude self

    print(f"\n  SEED: '{seed}'")
    for level, (lo, hi) in BANDS.items():
        band_ids = np.where((sims >= lo) & (sims < hi) & EN_MASK)[0]
        # Sort by cosine and show top-15
        band_ids = band_ids[np.argsort(sims[band_ids])[::-1]][:15]
        words = [tok.decode([int(i)]).strip() for i in band_ids]
        cos_vals = [f"{sims[i]:.3f}" for i in band_ids]
        print(f"    φ-{level} (cos∈[{lo:.3f},{hi:.3f}]): "
              f"{', '.join(f'{w}({c})' for w,c in zip(words,cos_vals))}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: φ-level classification test
# For each semantic relationship type, measure what φ-level most pairs fall in.
# Are the levels consistent within a category?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 4 — φ-Level Classification of Semantic Categories")
print("  Does each semantic category occupy a distinct φ-level?")
print("═"*72)

CATEGORIES = {
    'Gender pairs':      (GENDER_EN,        get_emb, get_emb),
    'Size/polarity':     (SIZE_EN,           get_emb, get_emb),
    'Sentiment':         (SENT_EN,           get_emb, get_emb),
    'Strict antonyms':   (ANTONYMS_STRICT,   get_emb, get_emb),
    'Synonyms':          (SYNONYMS,          get_emb, get_emb),
    'Hypernyms':         (HYPERNYMS,         get_emb, get_emb),
}

print(f"\n{'Category':<20}  {'n':>4}  {'mean_cos':>9}  {'std':>6}  "
      f"{'mean_θ°':>8}  {'modal_n':>8}  {'pct@modal':>10}  "
      f"{'φ_ref':>8}  {'Δ':>6}")
print("─"*90)

for cat_name, (pairs, gf1, gf2) in CATEGORIES.items():
    cosines = []
    for s, t in pairs:
        c = cos_pair(s, t, gf1, gf2)
        if c is not None: cosines.append(c)
    if not cosines: continue
    arr = np.array(cosines)

    # Find modal φ-level: n that minimizes mean |cos - 1/φⁿ|
    best_n, best_err = None, 1e9
    for n in range(1, 8):
        err = float(np.mean(np.abs(arr - PHI_LEVELS[n])))
        if err < best_err: best_err = err; best_n = n

    # Fraction of pairs within 0.05 of that level
    pct_modal = 100 * np.mean(np.abs(arr - PHI_LEVELS[best_n]) < 0.05)
    ref = PHI_LEVELS[best_n]
    delta = abs(arr.mean() - ref)
    th = np.degrees(np.arccos(np.clip(arr.mean(), -1, 1)))
    print(f"{cat_name:<20}  {len(cosines):>4}  {arr.mean():>9.4f}  "
          f"{arr.std():>6.4f}  {th:>8.2f}°  {best_n:>8}  "
          f"{pct_modal:>10.1f}%  {ref:>8.4f}  {delta:>6.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: The cos(1/φ) retrieval test
# For each seed word, find nearest words at EXACTLY cos ≈ 1/φ.
# Is the nearest φ-1 neighbor the word's semantic opposite/complement?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 5 — The φ-1 Nearest Neighbor Test")
print("  For each word: who is its nearest neighbor at cos ≈ 1/φ ≈ 0.618?")
print("  Hypothesis: these are the word's natural 'gender-type' complement.")
print("═"*72)

PROBE_WORDS = [
    'king','man','father','son','brother','actor','husband',
    'good','happy','love','beautiful','light','fast','strong',
    'big','long','hot','heavy','deep','bright',
    'dog','cat','bird','fish','tree','flower',
    'Paris','Berlin','Tokyo','London',
]

TARGET_PHI1 = PHI_LEVELS[1]   # 0.618
print(f"\n  Target cosine: 1/φ = {TARGET_PHI1:.4f}")
print(f"  Method: find EN word with cos closest to 1/φ (from above)\n")
print(f"  {'seed':<12}  {'φ-1 neighbor':<15}  {'cos':>7}  {'Δ from 1/φ':>11}  θ°")
print(f"  {'─'*12}  {'─'*15}  {'─'*7}  {'─'*11}  {'─'*6}")

for seed in PROBE_WORDS:
    e_seed, idx_seed = get_emb(seed)
    if e_seed is None: continue
    e_seed_n = normed(e_seed).astype(np.float32)
    sims = W_n @ e_seed_n
    sims[idx_seed] = -2.0

    # Among EN words only, find the one whose cosine is nearest to TARGET_PHI1
    dists_from_target = np.abs(sims - TARGET_PHI1)
    dists_from_target[~EN_MASK] = 1e9
    best_idx = int(np.argmin(dists_from_target))
    best_cos = float(sims[best_idx])
    best_word = tok.decode([best_idx]).strip()
    delta = abs(best_cos - TARGET_PHI1)
    theta = np.degrees(np.arccos(np.clip(best_cos, -1, 1)))
    print(f"  {seed:<12}  {best_word:<15}  {best_cos:>7.4f}  {delta:>11.4f}  {theta:.2f}°")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: Negative φ-levels — antonyms and the other side of the sphere
# True opposites might have cos ≈ -1/φⁿ (negative inner product)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 6 — Negative φ-Levels: The Dark Side of the Sphere")
print("  True semantic opposites: is cos(src,tgt) ≈ -1/φⁿ?")
print(f"  -1/φ = {-PHI_LEVELS[1]:.4f}  -1/φ² = {-PHI_LEVELS[2]:.4f}  "
      f"-1/φ³ = {-PHI_LEVELS[3]:.4f}")
print("═"*72)

# Pairs expected to be true opposites
NEG_PAIRS = [
    # Strict antonyms
    ('up','down'),('left','right'),('north','south'),('yes','no'),
    ('true','false'),('on','off'),('open','close'),('start','stop'),
    ('add','subtract'),('win','lose'),('create','destroy'),
    # Maximum contrast
    ('black','white'),('day','night'),('summer','winter'),('land','sea'),
    ('war','peace'),('birth','death'),('fire','water'),('earth','sky'),
    # Valence extremes
    ('heaven','hell'),('angel','devil'),('god','devil'),('love','hate'),
    ('hero','villain'),('king','peasant'),('wealth','poverty'),
]

print(f"\n  {'pair':>28}  {'cos':>8}  {'θ°':>7}  nearest neg-φ  Δ")
print(f"  {'─'*28}  {'─'*8}  {'─'*7}  {'─'*13}  {'─'*6}")
neg_cosines = []
for s, t in NEG_PAIRS:
    c = cos_pair(s, t)
    if c is None: continue
    th = np.degrees(np.arccos(np.clip(c, -1, 1)))
    neg_cosines.append(c)
    # Find nearest -1/φⁿ
    best_n, best_resid = None, 1e9
    for n in range(1, 7):
        ref = -PHI_LEVELS[n]
        if abs(c - ref) < best_resid: best_resid = abs(c - ref); best_n = n
    flag = " ◀◀" if best_resid < 0.03 else " ◀" if best_resid < 0.06 else ""
    print(f"  {s:>14}→{t:<14}  {c:>8.4f}  {th:>7.2f}°  "
          f"-1/φ^{best_n}={-PHI_LEVELS[best_n]:.4f}  {best_resid:.4f}{flag}")

if neg_cosines:
    arr = np.array(neg_cosines)
    print(f"\n  Summary: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"positive_frac={np.mean(arr>0)*100:.0f}%  negative_frac={np.mean(arr<0)*100:.0f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7: The φ-quantization law — statistical summary
# Across all pairs measured: how well does 1/φⁿ fit the cosine distribution?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 7 — Statistical Summary of φ-Quantization")
print("═"*72)

all_measured = []
for cat_name, (pairs, gf1, gf2) in CATEGORIES.items():
    for s, t in pairs:
        c = cos_pair(s, t, gf1, gf2)
        if c is not None: all_measured.append((cat_name, s, t, c))

# For each pair, what is the residual from the nearest φ-level?
residuals = []
assignments = {n: [] for n in range(0, 8)}
for cat, s, t, c in all_measured:
    n_best, resid = nearest_phi_n(c)
    if n_best is not None and n_best <= 7:
        residuals.append(resid)
        assignments[n_best].append((cat, s, t, c))

print(f"\n  Total pairs measured: {len(all_measured)}")
print(f"  Mean residual from nearest 1/φⁿ: {np.mean(residuals):.4f}")
print(f"  Pairs within 0.03 of a φ-level:  {np.mean(np.array(residuals)<0.03)*100:.1f}%")
print(f"  Pairs within 0.05 of a φ-level:  {np.mean(np.array(residuals)<0.05)*100:.1f}%")
print(f"  Pairs within 0.10 of a φ-level:  {np.mean(np.array(residuals)<0.10)*100:.1f}%")

print(f"\n  Assignment to φ-levels:")
for n in range(1, 8):
    grp = assignments[n]
    if not grp: continue
    cats = list(set(c for c,_,_,_ in grp))
    cosvals = [c for _,_,_,c in grp]
    print(f"    n={n} (1/φ^{n}={PHI_LEVELS[n]:.4f}): {len(grp):>4} pairs  "
          f"cos={np.mean(cosvals):.4f}±{np.std(cosvals):.4f}  "
          f"categories: {', '.join(cats[:3])}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8: φ-level and navigation accuracy
# The founding hypothesis: if pairs at cos=1/φ are semantically strongest,
# they should have the highest navigation accuracy (first expedition's finding).
# Test: does axis retrieval accuracy correlate with φ-level?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("PHASE 8 — φ-Level and Navigation Accuracy")
print("  Do pairs closer to 1/φ (n=1) have better retrieval accuracy?")
print("  Prediction: lower n = smaller angle = better navigation precision.")
print("═"*72)

from collections import defaultdict

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def source_ids(word):
    ids = set()
    for p in [word, ' '+word, word[0].upper()+word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

# Build gender axis (as before) and test accuracy per pair, binned by cos level
# Use a single mean tangent axis from the gender pairs
gender_tangents = []
gender_pair_data = []
for s, t in GENDER_EN:
    es, _ = get_emb(s);  et, _ = get_emb(t)
    if es is None or et is None: continue
    en_s = normed(es);  en_t = normed(et)
    cos_th = float(np.clip(np.dot(en_s, en_t), -1, 1))
    c = float(np.dot(en_s, en_t))
    sin_th = float(np.sqrt(max(0, 1 - cos_th**2)))
    tangent = (en_t - cos_th * en_s) / (sin_th + 1e-12)
    gender_tangents.append(tangent)
    gender_pair_data.append((s, t, cos_th, c))

gender_axis = normed(np.mean(gender_tangents, axis=0))
mean_theta = np.mean([np.degrees(np.arccos(np.clip(c, -1, 1)))
                      for _, _, c, _ in gender_pair_data])

# Find optimal navigation theta
best_th, best_acc = mean_theta, 0
for th_deg in np.linspace(10, 70, 120):
    th = np.radians(th_deg)
    acc = 0
    for s, t, _, _ in gender_pair_data:
        es, _ = get_emb(s)
        if es is None: continue
        pred = np.cos(th) * normed(es) + np.sin(th) * gender_axis
        w, _, _ = nn_ret(pred, source_ids(s), EN_MASK)
        if w == t: acc += 1
    if acc > best_acc: best_acc = acc; best_th = th_deg

print(f"\n  Gender axis: optimal θ={best_th:.1f}°  accuracy={best_acc}/{len(gender_pair_data)}")
print(f"\n  Per-pair breakdown (sorted by cos value):")
print(f"    {'pair':>26}  {'cos':>7}  {'φ-level':>8}  correct?")
print(f"    {'─'*26}  {'─'*7}  {'─'*8}  {'─'*8}")

th_opt = np.radians(best_th)
pair_results = []
for s, t, cos_th, c in sorted(gender_pair_data, key=lambda x: -x[3]):
    es, _ = get_emb(s)
    if es is None: continue
    pred = np.cos(th_opt) * normed(es) + np.sin(th_opt) * gender_axis
    w, sim, _ = nn_ret(pred, source_ids(s), EN_MASK)
    ok = (w == t)
    n_best, resid = nearest_phi_n(c)
    pair_results.append((s, t, c, n_best, ok))
    print(f"    {s:>12}→{t:<14}  {c:>7.4f}  n={n_best}({PHI_LEVELS.get(n_best,0):.3f})  "
          f"{'✓' if ok else '✗'} (got: {w})")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 2 SUMMARY")
print("═"*72)
print(f"""
φ-level reference table:
  n=1: 1/φ  = {PHI_LEVELS[1]:.6f}  θ={np.degrees(np.arccos(PHI_LEVELS[1])):.3f}°
  n=2: 1/φ² = {PHI_LEVELS[2]:.6f}  θ={np.degrees(np.arccos(PHI_LEVELS[2])):.3f}°
  n=3: 1/φ³ = {PHI_LEVELS[3]:.6f}  θ={np.degrees(np.arccos(PHI_LEVELS[3])):.3f}°
  n=4: 1/φ⁴ = {PHI_LEVELS[4]:.6f}  θ={np.degrees(np.arccos(PHI_LEVELS[4])):.3f}°

Questions answered today:
  Phase 1: Extended pairs — does φ-quantization hold at scale?
  Phase 2: Vocabulary distribution — peaks at φ-levels?
  Phase 3: φ-neighbor identification — who lives at each level?
  Phase 4: Category classification — does each type occupy a distinct level?
  Phase 5: φ-1 nearest neighbor — semantic complement identification
  Phase 6: Negative φ-levels — antonyms on the other side
  Phase 7: Statistical summary of quantization quality
  Phase 8: φ-level vs navigation accuracy

Record findings in second_expedition/expedition_log.md
""")
