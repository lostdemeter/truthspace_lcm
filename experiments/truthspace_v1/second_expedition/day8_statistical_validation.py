"""
SECOND EXPEDITION — DAY 8
=========================
Statistical Validation: Are the φ-Level Peaks Exact?

Days 1-7 established the φ-quantization law at all scales.
But is it statistically significant? Are the peaks at EXACTLY 1/φⁿ,
or merely "approximately" there by coincidence?

Day 8 tests:
  1. Random word-pair cosine distribution — does the φ-structure appear
     in random pairs, or only in curated semantic pairs?
  2. Fractional part test — if cosines are quantized at 1/φⁿ, then
     log(cos)/log(1/φ) should have fractional parts concentrated near 0.
     Rayleigh test for this circular concentration.
  3. Peak position fitting — fit Gaussians to the semantic cosine
     histogram; are the peak centers exactly at 1/φⁿ?
  4. Level width scaling — does σ_n scale as 1/φⁿ (fractal self-similarity)?
  5. Nearest-neighbor φ-level — for each common word, what is the
     φ-level of its top-k nearest neighbors? Does every word have
     a clear n=1 nearest neighbor?
  6. The "quantization residual" — define Δ_n = |cos - 1/φⁿ| / (1/φⁿ).
     Is Δ_n the same at all levels? (Tests scale invariance of the law.)

Script: second_expedition/day8_statistical_validation.py
"""

import torch, numpy as np
from collections import defaultdict

print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n = (W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
V = len(W_E)
print(f"  shape={W_E.shape}")

PHI = (1 + 5**0.5) / 2
PHI_L = {n: 1.0/PHI**n for n in range(0, 12)}

def normed(v): return v / (np.linalg.norm(v) + 1e-12)

EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(V)], dtype=bool)
EN_IDS = np.where(EN_MASK)[0]
print(f"  EN vocab size: {len(EN_IDS)}")

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def phi_level(c):
    if c <= 0: return None
    return -np.log(max(c, 1e-12)) / np.log(PHI)  # continuous, not rounded

def phi_n(c):
    if c <= 0: return None
    return round(-np.log(max(c, 1e-12)) / np.log(PHI))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Random vs semantic cosine distributions
# Sample 5000 random pairs from the top-3000 common English tokens
# Compare to Day 5's curated 207 semantic pairs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — Random vs Semantic Cosine Distributions")
print("  Sampling 5000 random EN word pairs; comparing to 207 curated pairs")
print("═"*72)

# Get a common English word list (frequent tokens)
# Select words with length 3-10 that are common (lower token IDs tend to be more frequent)
common_ids = [i for i in EN_IDS if 2 <= len(tok.decode([i]).strip()) <= 10
              and i < 50000]  # use lower IDs (more frequent)
print(f"\n  Common EN tokens (len 2-10, id<50k): {len(common_ids)}")

# Sample 5000 random pairs
rng = np.random.default_rng(42)
N_RANDOM = 5000
pairs_random = []
for _ in range(N_RANDOM):
    i, j = rng.choice(len(common_ids), 2, replace=False)
    ia, ib = common_ids[i], common_ids[j]
    ea = W_n[ia]; eb = W_n[ib]
    c = float(np.dot(ea, eb))
    if c > 0: pairs_random.append(c)

print(f"  Random pairs collected: {len(pairs_random)}")

# Day 5's curated pairs
CURATED = [
    ('north','south'),('east','west'),('northeast','southwest'),('northwest','southeast'),
    ('above','below'),('inside','outside'),('Sunday','Saturday'),('Monday','Friday'),
    ('morning','evening'),('summer','winter'),('yesterday','tomorrow'),('January','July'),
    ('true','false'),('positive','negative'),('correct','incorrect'),('valid','invalid'),
    ('two','three'),('second','third'),('hundred','thousand'),
    ('senior','junior'),('major','minor'),('strong','weak'),('high','low'),
    ('Korea','Korean'),('China','Chinese'),('Japan','Japanese'),('Russia','Russian'),
    ('France','French'),('Germany','German'),('Italy','Italian'),
    ('month','week'),('year','month'),('encode','decode'),('early','late'),
    ('son','daughter'),('brother','sister'),('boy','girl'),('uncle','aunt'),
    ('mother','father'),('king','queen'),('husband','wife'),('grandfather','grandmother'),
    ('actor','actress'),('prince','princess'),
    ('good','bad'),('love','hate'),('beautiful','ugly'),('best','worst'),
    ('happy','sad'),('right','wrong'),('wise','foolish'),('honest','dishonest'),
    ('big','small'),('large','tiny'),('tall','short'),('heavy','light'),('fast','slow'),
    ('red','blue'),('black','white'),('red','green'),('dark','light'),
    ('dog','cat'),('lion','tiger'),('bird','fish'),('fire','water'),
    ('give','take'),('buy','sell'),('push','pull'),('win','lose'),
]

curated_cos = []
for s, t in CURATED:
    es, _ = get_emb(s); et, _ = get_emb(t)
    if es is None or et is None: continue
    c = float(np.dot(normed(es), normed(et)))
    if c > 0: curated_cos.append(c)

print(f"  Curated pairs measured: {len(curated_cos)}")

# Histogram comparison
BINS = np.linspace(0.0, 0.80, 33)  # 32 bins from 0 to 0.8
print(f"\n  Cosine histogram comparison (bins={len(BINS)-1}):")
print(f"  {'cos range':>14}  {'φ-center':>10}  {'random%':>8}  {'curated%':>9}  ratio  peaks?")
print(f"  {'─'*14}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*8}")

hist_r, _ = np.histogram(pairs_random, bins=BINS, density=False)
hist_c, _ = np.histogram(curated_cos, bins=BINS, density=False)
hist_r_frac = hist_r / len(pairs_random)
hist_c_frac = hist_c / len(curated_cos)

# Find φ-level centers
phi_centers = {n: 1.0/PHI**n for n in range(1, 8)}

for k in range(len(BINS)-1):
    lo, hi = BINS[k], BINS[k+1]
    mid = (lo + hi) / 2
    phi_str = ""
    for n, c in phi_centers.items():
        if lo <= c < hi: phi_str = f"←1/φ^{n}={c:.3f}"
    if hist_c_frac[k] > 0.02 or phi_str:
        ratio = hist_c_frac[k] / (hist_r_frac[k] + 1e-6)
        print(f"  [{lo:.3f},{hi:.3f})  {phi_str:>10}  {hist_r_frac[k]:>8.3f}  "
              f"{hist_c_frac[k]:>9.3f}  {ratio:>5.1f}x  {'▲' if phi_str else ''}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Fractional part test (Rayleigh-type)
# If cosines are at 1/φⁿ, then φ-level = log(cos)/log(1/φ) should be integer
# → fractional part f = φ-level mod 1 should cluster near 0
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — Fractional Part Test: Is φ-Quantization Statistically Significant?")
print("  φ-level(cos) = -log(cos)/log(φ) should be near-integer if quantized")
print("═"*72)

def frac_part(x): return x - np.floor(x)

frac_random   = [frac_part(phi_level(c)) for c in pairs_random if c > 0.01]
frac_curated  = [frac_part(phi_level(c)) for c in curated_cos   if c > 0.01]

# Histogram of fractional parts
BINS_F = np.linspace(0, 1, 21)  # 20 bins
hist_fr, _ = np.histogram(frac_random,  bins=BINS_F, density=False)
hist_fc, _ = np.histogram(frac_curated, bins=BINS_F, density=False)

print(f"\n  Fractional part distribution (0=quantized, 0.5=anti-quantized):")
print(f"  {'frac range':>14}  {'random':>8}  {'curated':>9}  bar")
print(f"  {'─'*14}  {'─'*8}  {'─'*9}  {'─'*30}")

expected_r = len(frac_random) / 20
expected_c = len(frac_curated) / 20

for k in range(len(BINS_F)-1):
    lo, hi = BINS_F[k], BINS_F[k+1]
    bar_r = "░" * int(hist_fr[k] / expected_r * 5)
    bar_c = "█" * int(hist_fc[k] / expected_c * 5)
    near_zero = " ◀" if lo < 0.1 or hi > 0.95 else ""
    print(f"  [{lo:.2f},{hi:.2f})  {hist_fr[k]:>8}  {hist_fc[k]:>9}  "
          f"{bar_r:>8}|{bar_c:<8}{near_zero}")

# Rayleigh test equivalent: measure concentration near 0
# Map frac to circular angle θ = 2π·frac, then measure |mean(exp(iθ))|
import cmath
def rayleigh_R(fracs):
    angles = [2*np.pi*f for f in fracs]
    z = sum(cmath.exp(1j*a) for a in angles) / len(angles)
    return abs(z)

R_random  = rayleigh_R(frac_random)
R_curated = rayleigh_R(frac_curated)
print(f"\n  Rayleigh concentration R (0=uniform, 1=perfect quantization):")
print(f"    Random pairs:  R = {R_random:.4f}")
print(f"    Curated pairs: R = {R_curated:.4f}")
print(f"    Ratio: {R_curated/R_random:.2f}x more concentrated for semantic pairs")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Peak position fitting
# For the CURATED pairs, fit Gaussian peaks and measure how close they are to 1/φⁿ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — Peak Position Fitting: How Exact Are the φ-Level Centers?")
print("  Fit Gaussian to each φ-level band; compare center to 1/φⁿ")
print("═"*72)

# Group curated cosines by phi-level
by_level = defaultdict(list)
for c in curated_cos:
    n = phi_n(c)
    if n is not None: by_level[n].append(c)

print(f"\n  Per-level statistics (all curated pairs):")
print(f"  {'n':>3}  {'1/φⁿ':>8}  {'count':>6}  {'mean_cos':>9}  {'std_cos':>8}  "
      f"{'mean-1/φⁿ':>10}  {'std/(1/φⁿ)':>11}  {'relative_std':>12}")
print(f"  {'─'*3}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*12}")

level_data = {}
for n in sorted(by_level.keys()):
    vals = by_level[n]
    phi_center = PHI_L[n]
    mean_c = np.mean(vals)
    std_c  = np.std(vals)
    bias   = mean_c - phi_center
    rel_std = std_c / phi_center  # σ / center
    level_data[n] = (phi_center, len(vals), mean_c, std_c, bias, rel_std)
    print(f"  {n:>3}  {phi_center:>8.4f}  {len(vals):>6}  {mean_c:>9.4f}  {std_c:>8.4f}  "
          f"{bias:>+10.4f}  {std_c/phi_center:>11.4f}  {rel_std:>12.4f}")

# Test: does σ_n scale as 1/φⁿ? (i.e., is the relative std constant?)
rel_stds = [level_data[n][5] for n in sorted(level_data.keys()) if level_data[n][1] >= 3]
if len(rel_stds) >= 2:
    print(f"\n  Relative std (σ/center) across levels:")
    for n in sorted(level_data.keys()):
        if level_data[n][1] >= 3:
            print(f"    n={n}: {level_data[n][5]:.4f}")
    print(f"  Mean relative std: {np.mean(rel_stds):.4f}  std: {np.std(rel_stds):.4f}")
    print(f"  Is σ ∝ 1/φⁿ (constant relative std)? {'YES' if np.std(rel_stds) < 0.1 else 'PARTIAL'}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Nearest-neighbor φ-level scan
# For 200 common English words, find their top-10 nearest neighbors
# and record the φ-level of each neighbor
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — Nearest-Neighbor φ-Level Scan")
print("  For 200 common EN words: what φ-level are their nearest neighbors?")
print("═"*72)

# Sample 200 random common words
seed_ids = rng.choice(common_ids[:3000], 200, replace=False)
nn_phi_levels = defaultdict(int)  # n → count of nn at φ-level n

for seed_id in seed_ids:
    e_seed = W_n[seed_id]
    sims = W_n[EN_MASK] @ e_seed
    # top-10 excluding self
    top_k = np.argsort(sims)[-11:][::-1]
    for rank, nn_local_id in enumerate(top_k):
        nn_global_id = EN_IDS[nn_local_id]
        if nn_global_id == seed_id: continue
        c = float(sims[nn_local_id])
        n = phi_n(c)
        if n is not None: nn_phi_levels[n] += 1

total_nn = sum(nn_phi_levels.values())
print(f"\n  φ-level distribution of top-10 neighbors (200 seed words, ~2000 total neighbors):")
print(f"  {'n':>3}  {'1/φⁿ':>8}  {'count':>7}  {'fraction':>9}  {'expected_uniform':>17}")
expected_per_bin = total_nn / 10  # 10 bins (n=0..9)
for n in range(0, 10):
    count = nn_phi_levels.get(n, 0)
    frac = count / total_nn
    exp = count / expected_per_bin
    bar = "█" * int(frac * 100) + "░" * max(0, int(expected_per_bin/total_nn * 100) - int(frac*100))
    print(f"  {n:>3}  {PHI_L.get(n,0):>8.4f}  {count:>7}  {frac:>9.3%}  "
          f"(exp={expected_per_bin:.0f})  {bar[:40]}")

# Nearest neighbor (rank 1) specifically
print(f"\n  For the NEAREST neighbor (rank 1) only:")
nn1_levels = defaultdict(int)
for seed_id in seed_ids:
    e_seed = W_n[seed_id]
    sims = W_n[EN_MASK] @ e_seed
    sims[EN_IDS == seed_id] = -1  # exclude self
    nn1_local = int(np.argmax(sims))
    c = float(sims[nn1_local])
    n = phi_n(c)
    word = tok.decode([seed_id]).strip()
    nn_word = tok.decode([EN_IDS[nn1_local]]).strip()
    if n is not None: nn1_levels[n] += 1
total_nn1 = sum(nn1_levels.values())
for n in sorted(nn1_levels.keys()):
    print(f"  n={n}: {nn1_levels[n]:>4}/{total_nn1}={nn1_levels[n]/total_nn1:.1%}  "
          f"(1/φ^{n}={PHI_L.get(n,0):.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Scale-invariance test — the quantization residual
# Define residual Δ_n = (cos - 1/φⁿ) / (1/φⁿ - 1/φⁿ⁺¹)  — normalized deviation
# Test: is Δ_n distributed the same at all levels? (Scale invariance)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — Scale-Invariance Test: Is the Residual Distribution the Same at All Levels?")
print("  Δ_n = (cos - 1/φⁿ) / (1/φⁿ⁺¹ - 1/φⁿ)  (normalized to band width)")
print("═"*72)

# Get a larger sample of semantic pairs by combining all Day 5 domains
ALL_SEMANTIC = {
    'compass':     [('north','south'),('east','west'),('northeast','southwest'),
                    ('northwest','southeast'),('above','below'),('inside','outside')],
    'calendar':    [('Sunday','Saturday'),('Monday','Friday'),('morning','evening'),
                    ('summer','winter'),('yesterday','tomorrow'),('January','July')],
    'boolean':     [('true','false'),('positive','negative'),('correct','incorrect'),('valid','invalid')],
    'numbers':     [('two','three'),('second','third'),('hundred','thousand')],
    'rank':        [('senior','junior'),('major','minor'),('strong','weak'),('high','low')],
    'nation_lang': [('Korea','Korean'),('China','Chinese'),('Japan','Japanese'),
                    ('Russia','Russian'),('France','French'),('Germany','German'),
                    ('Italy','Italian'),('Greece','Greek'),('Spain','Spanish')],
    'kinship':     [('son','daughter'),('brother','sister'),('boy','girl'),('uncle','aunt'),
                    ('mother','father'),('king','queen'),('husband','wife'),
                    ('grandfather','grandmother'),('actor','actress'),('prince','princess')],
    'sentiment':   [('good','bad'),('love','hate'),('beautiful','ugly'),('best','worst'),
                    ('happy','sad'),('right','wrong'),('wise','foolish'),('honest','dishonest')],
    'size':        [('big','small'),('large','tiny'),('tall','short'),('heavy','light'),('fast','slow')],
    'color':       [('red','blue'),('black','white'),('red','green'),('dark','light')],
    'animals':     [('dog','cat'),('lion','tiger'),('horse','cow'),('bird','fish')],
    'nature':      [('fire','water'),('sun','moon'),('land','sea'),('river','mountain')],
    'actions':     [('give','take'),('buy','sell'),('push','pull'),('win','lose')],
}

all_cos_by_level = defaultdict(list)
for domain, pairs in ALL_SEMANTIC.items():
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        c = float(np.dot(normed(es), normed(et)))
        if c > 0.01:
            n = phi_n(c)
            if n is not None: all_cos_by_level[n].append((s, t, c, domain))

# Compute normalized residual for each pair
residuals_by_level = defaultdict(list)
for n, items in all_cos_by_level.items():
    c_n   = PHI_L.get(n, 0)
    c_n1  = PHI_L.get(n+1, 0)
    bandwidth = abs(c_n - c_n1)
    for s, t, c, dom in items:
        delta = (c - c_n) / bandwidth if bandwidth > 0 else 0
        residuals_by_level[n].append(delta)

print(f"\n  Normalized residual Δ_n distribution per level:")
print(f"  {'n':>3}  {'1/φⁿ':>8}  {'count':>6}  {'mean_Δ':>8}  {'std_Δ':>8}  "
      f"{'%in[-0.5,0.5]':>13}  interpretation")
print(f"  {'─'*3}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*13}  {'─'*25}")

for n in sorted(residuals_by_level.keys()):
    res = residuals_by_level[n]
    if len(res) < 2: continue
    mean_d = np.mean(res); std_d = np.std(res)
    frac_in = sum(1 for d in res if -0.5 <= d <= 0.5) / len(res)
    centered = "centered" if abs(mean_d) < 0.2 else "biased"
    narrow = "narrow" if std_d < 0.3 else "wide"
    interp = f"{centered}, {narrow}"
    print(f"  {n:>3}  {PHI_L.get(n,0):>8.4f}  {len(res):>6}  {mean_d:>8.3f}  {std_d:>8.3f}  "
          f"{frac_in:>13.0%}  {interp}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: The null model comparison
# If we replace 1/φⁿ with RANDOM positions (same spacing), does the quantization disappear?
# This tests whether the SPECIFIC φ-positions matter, or just any evenly-spaced grid
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — Null Model: Does the φ-Specific Grid Matter?")
print("  Compare: φ-grid residuals vs alternative grids")
print("  If φ is special, residuals should be smaller than for other grids")
print("═"*72)

all_cos = [c for items in all_cos_by_level.values() for _, _, c, _ in items]

def mean_residual_to_grid(cosines, grid_points):
    """Mean absolute distance from each cosine to its nearest grid point."""
    residuals = []
    for c in cosines:
        dists = [abs(c - g) for g in grid_points]
        residuals.append(min(dists))
    return np.mean(residuals)

# φ-grid
phi_grid = [PHI_L[n] for n in range(1, 8)]

# Alternative grids with same number of points and same range
# Grid 1: Arithmetic series from 1/φ^1 to 1/φ^7
arith_lo, arith_hi = min(phi_grid), max(phi_grid)
arith_grid = list(np.linspace(arith_hi, arith_lo, len(phi_grid)))

# Grid 2: Geometric series with ratio r = 0.75 (instead of 1/φ≈0.618)
grid_075 = [0.75**n for n in range(1, len(phi_grid)+1)]

# Grid 3: Geometric series with ratio r = 0.50 (binary fractions)
grid_050 = [0.50**n for n in range(1, len(phi_grid)+1)]

# Grid 4: Geometric series with ratio r = 0.80
grid_080 = [0.80**n for n in range(1, len(phi_grid)+1)]

# Grid 5: Random grid (shuffled grid)
random_grid = sorted(rng.uniform(arith_lo, arith_hi, len(phi_grid)), reverse=True)

grids = {
    f'φ-grid (r=1/φ=0.618)': phi_grid,
    'arithmetic':             arith_grid,
    'r=0.750 geometric':     grid_075,
    'r=0.500 geometric':     grid_050,
    'r=0.800 geometric':     grid_080,
    'random grid':             random_grid,
}

print(f"\n  Grid comparison (mean absolute distance from cosine to nearest grid point):")
print(f"  {'grid':<28}  {'points':>40}  {'mean_resid':>10}")
print(f"  {'─'*28}  {'─'*40}  {'─'*10}")

results = {}
for name, grid in grids.items():
    mean_res = mean_residual_to_grid(all_cos, grid)
    results[name] = mean_res
    pts_str = "  ".join(f"{g:.3f}" for g in grid[:5])
    print(f"  {name:<28}  {pts_str:>40}  {mean_res:>10.4f}")

phi_res = results[f'φ-grid (r=1/φ=0.618)']
print(f"\n  φ-grid residual vs alternatives (lower = better fit):")
for name, res in sorted(results.items(), key=lambda x: x[1]):
    ratio = res / phi_res
    flag = " ← BEST" if ratio == 1.0 else (f" {ratio:.2f}x worse" if ratio > 1 else "")
    print(f"    {name:<28}  {res:.4f}  {flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 8 SUMMARY")
print("═"*72)
print("""
Day 8 provides statistical validation of the φ-quantization law:
  Phase 1: Random vs semantic cosine distributions
  Phase 2: Fractional part / Rayleigh concentration test
  Phase 3: Gaussian peak fitting — centers vs 1/φⁿ
  Phase 4: Nearest-neighbor φ-level scan across 200 words
  Phase 5: Scale-invariance of the normalized residual
  Phase 6: Null model — does the specific φ-grid outperform alternatives?

Record in second_expedition/expedition_log.md
""")
