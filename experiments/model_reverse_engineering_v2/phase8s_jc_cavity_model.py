#!/usr/bin/env python3
"""
Phase 8s: Jaynes-Cummings Cavity QED Model of Gate Content
============================================================

Inspired by MPQ343 (Kubanek, "Two-photon gateway and real-time
feedback control of a single atom in a cavity"), we test whether
cavity QED concepts improve our geometric model of gate content.

The atom-cavity system:
  - Atom (two-level) ↔ Hidden state (3584-D)
  - Cavity mode ↔ Gate space (18944-D)
  - Coupling g ↔ W_gate
  - Dressed states ↔ SVD modes of W_gate
  - Rabi splitting Ω_n = √(Δ² + 4g²n) ↔ singular values

Five tests:
1. NULL-SPACE PAIRS: Do null-space dimensions show two-photon-gateway-
   like paired correlations? Can pairs predict tokens?
2. √n LADDER: Does the singular value spectrum follow JC scaling
   (Ω_n ∝ √n)? Is our 2× norm ratio = first Rabi splitting Ω₁ = 2g?
3. DRESSED-STATE ROUTING: In SVD basis, is content routing simpler
   (sparser, more predictable)?
4. FEEDBACK DYNAMICS: Does layer-by-layer correction follow JC master
   equation dynamics (exponential decay × oscillation)?
5. PAIRED-GATE CORRECTION: Can null-space pair correlations build
   an additive correction that improves token prediction?

Goal: find actionable leads for improving model accuracy.
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os

PHI = (1 + np.sqrt(5)) / 2
SQRT_PHI = np.sqrt(PHI)
LOG_PHI = np.log(PHI)

COMB_START = 6
COMB_END = 23

print("=" * 80)
print("  PHASE 8s: JAYNES-CUMMINGS CAVITY QED MODEL")
print("  Testing whether cavity QED concepts improve gate prediction")
print("=" * 80)
print()

print("Loading Qwen2-7B...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

N_LAYERS = len(model.model.layers)
GATE_DIM = model.config.intermediate_size   # 18944
HIDDEN_DIM = model.config.hidden_size       # 3584


# ================================================================
# STEP 0: Extract W_gate SVDs for COMB layers
# ================================================================
print("-" * 80)
print("  STEP 0: Extract W_gate SVDs (this takes a while)")
print("-" * 80)

W_gates = {}
W_gate_svds = {}

# Focus on layer 14 for detailed analysis, sample others for cross-layer
FOCUS_LAYER = 14
SAMPLE_LAYERS = [6, 10, 14, 18, 22]

for layer in SAMPLE_LAYERS:
    print(f"  SVD of layer {layer}...", end=" ", flush=True)
    W = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    W_gates[layer] = W
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_gate_svds[layer] = (U, S, Vt)
    print(f"done. rank={np.sum(S > 1e-6)}, cond={S[0]/S[-1]:.1f}")

print()


# ================================================================
# STEP 1: Build scaffold and collect token data
# ================================================================
print("-" * 80)
print("  STEP 1: Collect token data")
print("-" * 80)

TRAIN_WORDS = [
    "king", "queen", "man", "woman", "boy", "girl",
    "hot", "cold", "fast", "slow", "big", "small",
    "love", "hate", "light", "dark", "true", "false",
    "cat", "dog", "tree", "water", "fire", "earth",
    "happy", "sad", "strong", "weak", "old", "young",
    "the", "is", "and", "of", "to", "in",
    "zero", "one", "two", "three", "four", "five",
    "red", "blue", "green", "black", "white", "yellow",
    "algorithm", "quantum", "geometry", "neural", "vector", "matrix",
    "Paris", "London", "Tokyo", "Einstein", "Newton", "Euler",
    "hello", "world", "computer", "science", "language", "model",
]

single_gates = {}
single_hs = {}

for word in TRAIN_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    decoded = tokenizer.decode([token_id]).strip()
    if decoded in single_gates:
        continue

    gate_storage = {}
    hs_storage = {}
    hooks = []

    def make_gate_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hs_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(N_LAYERS):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gate_hook(gate_storage, layer)
        )
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook(hs_storage, layer)
        )
        hooks.extend([h1, h2])

    with torch.no_grad():
        model(torch.tensor([[token_id]], device="cuda"))

    for h in hooks:
        h.remove()

    single_gates[decoded] = np.stack([gate_storage[l] for l in range(N_LAYERS)])
    single_hs[decoded] = np.stack([hs_storage[l] for l in range(N_LAYERS)])

train_words = sorted(single_gates.keys())
all_gates_single = np.stack([single_gates[w] for w in train_words])
all_hs_single = np.stack([single_hs[w] for w in train_words])

scaffold_gate = all_gates_single.mean(axis=0)
scaffold_hidden = all_hs_single.mean(axis=0)

print(f"  Crystal: {len(train_words)} tokens")
print()


# ================================================================
# TEST 1: NULL-SPACE PAIR DETECTION
# Two-photon gateway analog: do null-space dimensions show
# correlated pairs that carry token information?
# ================================================================
print("=" * 80)
print("  TEST 1: NULL-SPACE PAIR DETECTION (Two-Photon Gateway)")
print("  Do null-space dimensions show paired correlations?")
print("=" * 80)
print()

layer = FOCUS_LAYER
U, S, Vt = W_gate_svds[layer]
W = W_gates[layer]

# Column space projector: P_col = W @ W_pinv
# Null space projector: P_null = I - P_col
# But we can compute null-space projections more efficiently via SVD
# The column space of W (18944×3584) has rank 3584
# Null space has dim 18944 - 3584 = 15360

# For each token, project gate vector into null space
# g_null = g - W @ W_pinv @ g = g - U @ U^T @ g (since U spans column space)

print(f"  Layer {layer}: computing null-space projections...")

null_projs = []
col_projs = []
for word in train_words:
    g = single_gates[word][layer]
    g_resid = g - scaffold_gate[layer]

    # Column-space projection
    g_col = U @ (U.T @ g_resid)
    # Null-space projection
    g_null = g_resid - g_col

    col_projs.append(g_col)
    null_projs.append(g_null)

null_projs = np.array(null_projs)   # (n_tokens, 18944)
col_projs = np.array(col_projs)

null_norms = np.linalg.norm(null_projs, axis=1)
col_norms = np.linalg.norm(col_projs, axis=1)
total_norms = np.linalg.norm(null_projs + col_projs, axis=1)

print(f"  Null-space energy fraction: {np.mean(null_norms**2 / (total_norms**2 + 1e-10)):.4f}")
print(f"  Column-space energy fraction: {np.mean(col_norms**2 / (total_norms**2 + 1e-10)):.4f}")
print()

# SVD of null-space projections to find structure
print(f"  SVD of null-space projections ({len(train_words)} tokens)...")
U_null, S_null, Vt_null = np.linalg.svd(null_projs, full_matrices=False)

# Look for paired structure: check if singular values come in near-degenerate pairs
print(f"  Top 20 null-space singular values:")
print(f"  {'i':>3s}  {'S[i]':>10s}  {'S[i]/S[i+1]':>12s}  {'Pair gap':>10s}")
print("  " + "-" * 40)

pair_gaps = []
for i in range(20):
    ratio = S_null[i] / S_null[i+1] if S_null[i+1] > 1e-10 else float('inf')
    # Pair gap: |S[2k] - S[2k+1]| / S[2k]
    if i % 2 == 0 and i+1 < len(S_null):
        pair_gap = abs(S_null[i] - S_null[i+1]) / (S_null[i] + 1e-10)
        pair_gaps.append(pair_gap)
        print(f"  {i:3d}  {S_null[i]:10.4f}  {ratio:12.6f}  {pair_gap:10.6f} {'← PAIR' if pair_gap < 0.05 else ''}")
    else:
        print(f"  {i:3d}  {S_null[i]:10.4f}  {ratio:12.6f}")

print()
mean_pair_gap = np.mean(pair_gaps[:10])
print(f"  Mean pair gap (top 10 pairs): {mean_pair_gap:.4f}")
print(f"  {'>> PAIRED structure detected' if mean_pair_gap < 0.1 else '>> No clear pairing'}")
print()

# Do null-space pair coordinates correlate with token identity?
# Project tokens onto top null-space directions
null_coords = null_projs @ Vt_null[:20].T   # (n_tokens, 20)

# Check if pairs of null coordinates jointly predict better than singles
# Compute mutual information proxy: can we separate token classes?
from scipy.spatial.distance import pdist, squareform
pair_dists = squareform(pdist(null_coords[:, :2]))   # using pair (0,1)
single_dists = squareform(pdist(null_coords[:, :1]))  # using single 0

# Compare with actual token identity (use hidden-state distances as ground truth)
h_resids = np.array([single_hs[w][layer] - scaffold_hidden[layer] for w in train_words])
h_dists = squareform(pdist(h_resids))

from scipy.stats import spearmanr
corr_pair, _ = spearmanr(pair_dists.flatten(), h_dists.flatten())
corr_single, _ = spearmanr(single_dists.flatten(), h_dists.flatten())

print(f"  Null-space → token identity correlation:")
print(f"    Single dim (dim 0): Spearman r = {corr_single:.4f}")
print(f"    Paired dims (0,1):  Spearman r = {corr_pair:.4f}")

# Expand: check all pairs vs all singles
pair_corrs = []
single_corrs = []
for i in range(0, 20, 2):
    pair_d = squareform(pdist(null_coords[:, i:i+2]))
    sing_d = squareform(pdist(null_coords[:, i:i+1]))
    r_pair, _ = spearmanr(pair_d.flatten(), h_dists.flatten())
    r_sing, _ = spearmanr(sing_d.flatten(), h_dists.flatten())
    pair_corrs.append(r_pair)
    single_corrs.append(r_sing)

print(f"    Mean pair correlation (10 pairs): {np.mean(pair_corrs):.4f}")
print(f"    Mean single correlation (10 dims): {np.mean(single_corrs):.4f}")
print(f"    Pair advantage: {np.mean(pair_corrs) - np.mean(single_corrs):.4f}")
print()

test1_results = {
    'null_energy_fraction': float(np.mean(null_norms**2 / (total_norms**2 + 1e-10))),
    'mean_pair_gap': float(mean_pair_gap),
    'pair_corr_mean': float(np.mean(pair_corrs)),
    'single_corr_mean': float(np.mean(single_corrs)),
    'pair_advantage': float(np.mean(pair_corrs) - np.mean(single_corrs)),
    'null_sv_top10': [float(s) for s in S_null[:10]],
    'paired_structure': bool(mean_pair_gap < 0.1),
}


# ================================================================
# TEST 2: √n LADDER TEST
# JC model: Ω_n = √(Δ² + 4g²n) → at resonance Ω_n = 2g√n
# Does our singular value spectrum follow this?
# ================================================================
print("=" * 80)
print("  TEST 2: √n LADDER (Jaynes-Cummings Scaling)")
print("  Do singular values follow Ω_n = 2g√n?")
print("=" * 80)
print()

layer = FOCUS_LAYER
_, S, _ = W_gate_svds[layer]

# Test 1: Does S[i] ∝ 1/√(i+1) (inverse JC scaling)?
# The singular values of a coupling matrix in JC would show
# the Rabi frequencies at different excitation levels
n_test = min(100, len(S))
indices = np.arange(1, n_test + 1)

# Fit: S[i] = A / √(i + offset)
# Try several models
from scipy.optimize import curve_fit

def sqrt_n_model(n, A, offset):
    return A / np.sqrt(n + offset)

def power_law_model(n, A, alpha):
    return A * n**(-alpha)

def jc_model(n, g, delta):
    """JC Rabi frequency: Ω_n = √(Δ² + 4g²n)"""
    return np.sqrt(delta**2 + 4 * g**2 * n)

# Normalize S for fitting
S_norm = S[:n_test] / S[0]

try:
    popt_sqrt, _ = curve_fit(sqrt_n_model, indices, S_norm, p0=[1.0, 0.0], maxfev=5000)
    S_pred_sqrt = sqrt_n_model(indices, *popt_sqrt)
    rmse_sqrt = np.sqrt(np.mean((S_norm - S_pred_sqrt)**2))
except:
    rmse_sqrt = float('inf')
    popt_sqrt = [0, 0]

try:
    popt_power, _ = curve_fit(power_law_model, indices, S_norm, p0=[1.0, 0.5], maxfev=5000)
    S_pred_power = power_law_model(indices, *popt_power)
    rmse_power = np.sqrt(np.mean((S_norm - S_pred_power)**2))
except:
    rmse_power = float('inf')
    popt_power = [0, 0]

# Also try JC model on raw S (not inverted)
# In JC, higher n → higher Ω_n, but our S is decreasing
# So try: S[i] maps to Ω_{N-i} where N is total modes
# Or: the coupling matrix eigenvalues might go as g/√n

print(f"  Fitting first {n_test} singular values (normalized to S[0]=1):")
print(f"    1/√n model: RMSE = {rmse_sqrt:.6f}, params = A={popt_sqrt[0]:.4f}, offset={popt_sqrt[1]:.4f}")
print(f"    Power law:  RMSE = {rmse_power:.6f}, params = A={popt_power[0]:.4f}, α={popt_power[1]:.4f}")
print(f"    α vs 1/2 (√n): error = {abs(popt_power[1] - 0.5):.4f}")
print()

# Check consecutive ratios for JC pattern
# In JC: Ω_n/Ω_{n-1} = √(n/(n-1)) → approaches 1 from above
# Singular value ratios:
print(f"  Consecutive SV ratios vs JC prediction:")
print(f"  {'n':>3s}  {'S[n]/S[n-1]':>12s}  {'JC √(n/(n+1))':>14s}  {'Error':>8s}")
print("  " + "-" * 42)

jc_errors = []
for n in range(1, 20):
    obs_ratio = S[n] / S[n-1]
    jc_ratio = np.sqrt(n / (n + 1))  # predicted: S decays as 1/√n
    err = abs(obs_ratio - jc_ratio)
    jc_errors.append(err)
    print(f"  {n:3d}  {obs_ratio:12.6f}  {jc_ratio:14.6f}  {err:8.4f}")

print()
print(f"  Mean JC ratio error (first 20): {np.mean(jc_errors):.4f}")
print()

# Check the 2× norm ratio = first Rabi splitting
# Ω₁ = 2g at resonance. Our norm ratio was 2.019.
# The coupling constant g would be S[0]/2 in some normalization
norm_ratios = []
for word in train_words[:30]:
    g_resid = single_gates[word][layer] - scaffold_gate[layer]
    h_resid = single_hs[word][layer] - scaffold_hidden[layer]
    ratio = np.linalg.norm(g_resid) / (np.linalg.norm(h_resid) + 1e-10)
    norm_ratios.append(ratio)

mean_norm_ratio = np.mean(norm_ratios)
print(f"  Norm ratio (gate/hidden) = {mean_norm_ratio:.4f}")
print(f"  If this = Ω₁ = 2g, then g = {mean_norm_ratio/2:.4f}")
print(f"  S[0] = {S[0]:.4f}, S[0]/mean_norm_ratio = {S[0]/mean_norm_ratio:.4f}")
print()

# Cross-layer: does the norm ratio track S[0]?
print(f"  Cross-layer: norm ratio vs S[0]")
cross_norm_ratios = []
cross_s0 = []
for l in SAMPLE_LAYERS:
    _, Sl, _ = W_gate_svds[l]
    ratios_l = []
    for word in train_words[:30]:
        g_r = single_gates[word][l] - scaffold_gate[l]
        h_r = single_hs[word][l] - scaffold_hidden[l]
        ratios_l.append(np.linalg.norm(g_r) / (np.linalg.norm(h_r) + 1e-10))
    cross_norm_ratios.append(np.mean(ratios_l))
    cross_s0.append(Sl[0])
    print(f"    Layer {l}: norm_ratio={np.mean(ratios_l):.4f}, S[0]={Sl[0]:.4f}, ratio/S[0]={np.mean(ratios_l)/Sl[0]:.6f}")

corr_ratio_s0, _ = spearmanr(cross_norm_ratios, cross_s0)
print(f"  Correlation(norm_ratio, S[0]): {corr_ratio_s0:.4f}")
print()

test2_results = {
    'sqrt_n_rmse': float(rmse_sqrt),
    'power_law_rmse': float(rmse_power),
    'power_law_alpha': float(popt_power[1]),
    'mean_jc_ratio_error': float(np.mean(jc_errors)),
    'norm_ratio': float(mean_norm_ratio),
    'coupling_g': float(mean_norm_ratio / 2),
    'cross_layer_corr_ratio_s0': float(corr_ratio_s0),
}


# ================================================================
# TEST 3: DRESSED-STATE ROUTING
# Transform to SVD basis of W_gate. Is content sparser/simpler?
# ================================================================
print("=" * 80)
print("  TEST 3: DRESSED-STATE ROUTING")
print("  In the SVD basis of W_gate, is content routing simpler?")
print("=" * 80)
print()

layer = FOCUS_LAYER
U, S, Vt = W_gate_svds[layer]

# Transform gate residuals to dressed-state basis
# g_resid in dressed basis = U^T @ g_resid (project onto left singular vectors)
# This gives coordinates in the column space of W_gate

dressed_coords = []
raw_resids = []
for word in train_words:
    g_resid = single_gates[word][layer] - scaffold_gate[layer]
    raw_resids.append(g_resid)
    dressed = U.T @ g_resid  # (3584,) — coordinates in dressed basis
    dressed_coords.append(dressed)

dressed_coords = np.array(dressed_coords)  # (n_tokens, 3584)
raw_resids = np.array(raw_resids)          # (n_tokens, 18944)

# Test 1: Sparsity — are dressed coordinates sparser?
# Gini coefficient as sparsity measure
def gini(x):
    x = np.abs(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x) + 1e-10)) - (n + 1) / n

dressed_ginis = [gini(dc) for dc in dressed_coords]
raw_ginis = [gini(rr) for rr in raw_resids]

print(f"  Sparsity (Gini coefficient, higher = sparser):")
print(f"    Dressed basis (3584-D): {np.mean(dressed_ginis):.4f} ± {np.std(dressed_ginis):.4f}")
print(f"    Raw gate space (18944-D): {np.mean(raw_ginis):.4f} ± {np.std(raw_ginis):.4f}")
print()

# Test 2: Effective dimensionality in dressed basis
# How many dressed coordinates needed for 90% energy?
dressed_energy = dressed_coords ** 2
for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
    d_stars = []
    for i in range(len(train_words)):
        sorted_e = np.sort(dressed_energy[i])[::-1]
        cum = np.cumsum(sorted_e) / (np.sum(sorted_e) + 1e-10)
        d_star = int(np.searchsorted(cum, threshold) + 1)
        d_stars.append(d_star)
    print(f"  D* at {threshold*100:.0f}% energy (dressed): mean={np.mean(d_stars):.1f}, median={np.median(d_stars):.0f}")

print()

# Test 3: Token discrimination in dressed basis
# Do fewer dressed dimensions separate tokens?
# Compare: top-k dressed coords vs top-k raw coords
from sklearn.metrics import pairwise_distances

print(f"  Token discrimination (nearest-neighbor accuracy):")
for k in [1, 3, 7, 20, 50, 100]:
    # Dressed: top k coords
    dressed_k = dressed_coords[:, :k]
    d_dists = pairwise_distances(dressed_k)
    np.fill_diagonal(d_dists, np.inf)
    nn_dressed = np.argmin(d_dists, axis=1)

    # Raw: top k SVD coords of raw residuals
    U_raw, S_raw, Vt_raw = np.linalg.svd(raw_resids, full_matrices=False)
    raw_k = raw_resids @ Vt_raw[:k].T
    r_dists = pairwise_distances(raw_k)
    np.fill_diagonal(r_dists, np.inf)
    nn_raw = np.argmin(r_dists, axis=1)

    # Hidden: top k SVD coords
    h_resids = np.array([single_hs[w][layer] - scaffold_hidden[layer] for w in train_words])
    U_h, S_h, Vt_h = np.linalg.svd(h_resids, full_matrices=False)
    h_k = h_resids @ Vt_h[:k].T
    hh_dists = pairwise_distances(h_k)
    np.fill_diagonal(hh_dists, np.inf)
    nn_hidden = np.argmin(hh_dists, axis=1)

    # Use full-dimensional hidden as ground truth for NN identity
    full_h_dists = pairwise_distances(h_resids)
    np.fill_diagonal(full_h_dists, np.inf)
    nn_truth = np.argmin(full_h_dists, axis=1)

    acc_dressed = np.mean(nn_dressed == nn_truth)
    acc_raw = np.mean(nn_raw == nn_truth)
    acc_hidden = np.mean(nn_hidden == nn_truth)

    print(f"    k={k:3d}: dressed={acc_dressed:.3f}, raw_svd={acc_raw:.3f}, hidden_svd={acc_hidden:.3f}")

print()

# Test 4: Do dressed coordinates cluster by semantic category?
# Weight the dressed coords by singular values (energy-weighted)
weighted_dressed = dressed_coords * S[np.newaxis, :]  # weight by SV

# Token-token cosine similarity in weighted dressed basis vs raw
def cos_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X @ X.T) / (norms @ norms.T + 1e-10)

cos_dressed = cos_matrix(weighted_dressed)
cos_raw = cos_matrix(raw_resids)

# Are semantically related pairs (king/queen, man/woman, etc.) more similar in dressed basis?
semantic_pairs = [
    ("king", "queen"), ("man", "woman"), ("boy", "girl"),
    ("hot", "cold"), ("fast", "slow"), ("big", "small"),
    ("love", "hate"), ("light", "dark"), ("true", "false"),
    ("cat", "dog"), ("red", "blue"), ("old", "young"),
]

pair_cos_dressed = []
pair_cos_raw = []
for w1, w2 in semantic_pairs:
    if w1 in train_words and w2 in train_words:
        i1, i2 = train_words.index(w1), train_words.index(w2)
        pair_cos_dressed.append(abs(cos_dressed[i1, i2]))
        pair_cos_raw.append(abs(cos_raw[i1, i2]))

print(f"  Semantic pair |cosine| (higher = better separation of pairs):")
print(f"    Dressed (SV-weighted): {np.mean(pair_cos_dressed):.4f} ± {np.std(pair_cos_dressed):.4f}")
print(f"    Raw gate space:        {np.mean(pair_cos_raw):.4f} ± {np.std(pair_cos_raw):.4f}")
print()

test3_results = {
    'dressed_gini': float(np.mean(dressed_ginis)),
    'raw_gini': float(np.mean(raw_ginis)),
    'semantic_pair_cos_dressed': float(np.mean(pair_cos_dressed)),
    'semantic_pair_cos_raw': float(np.mean(pair_cos_raw)),
}


# ================================================================
# TEST 4: FEEDBACK DYNAMICS
# JC master equation: state evolves as exp(-κt) × cos(Ωt)
# Does layer-by-layer correction follow this pattern?
# ================================================================
print("=" * 80)
print("  TEST 4: FEEDBACK DYNAMICS")
print("  Does layer-by-layer gate correction follow JC dynamics?")
print("  JC: ρ(t) ~ exp(-κt) × cos(Ωt + φ)")
print("=" * 80)
print()

# For each token, measure "correction strength" at each COMB layer
# Correction = how much the gate residual changes from scaffold prediction
# This is the |α| value (rank-1 scalar) across layers

layer_corrections = []
for word in train_words:
    corrections = []
    for layer in range(COMB_START, COMB_END):
        g_resid = single_gates[word][layer] - scaffold_gate[layer]
        corrections.append(np.linalg.norm(g_resid))
    layer_corrections.append(corrections)

layer_corrections = np.array(layer_corrections)  # (n_tokens, 17 layers)
mean_corrections = layer_corrections.mean(axis=0)

# Normalize to first COMB layer
norm_corrections = mean_corrections / (mean_corrections[0] + 1e-10)

layers_comb = np.arange(COMB_START, COMB_END)

# Fit JC master equation: A * exp(-κ*t) * cos(Ω*t + φ₀) + C
def jc_decay(t, A, kappa, omega, phi0, C):
    return A * np.exp(-kappa * t) * np.cos(omega * t + phi0) + C

def exp_decay(t, A, kappa, C):
    return A * np.exp(-kappa * t) + C

def linear_model(t, A, B):
    return A * t + B

t_norm = layers_comb - COMB_START  # 0, 1, 2, ..., 16

try:
    popt_jc, _ = curve_fit(jc_decay, t_norm, norm_corrections, 
                           p0=[1.0, 0.1, 1.0, 0.0, 0.5], maxfev=10000)
    pred_jc = jc_decay(t_norm, *popt_jc)
    rmse_jc = np.sqrt(np.mean((norm_corrections - pred_jc)**2))
except:
    rmse_jc = float('inf')
    popt_jc = [0, 0, 0, 0, 0]

try:
    popt_exp, _ = curve_fit(exp_decay, t_norm, norm_corrections, 
                            p0=[1.0, 0.1, 0.5], maxfev=10000)
    pred_exp = exp_decay(t_norm, *popt_exp)
    rmse_exp = np.sqrt(np.mean((norm_corrections - pred_exp)**2))
except:
    rmse_exp = float('inf')
    popt_exp = [0, 0, 0]

try:
    popt_lin, _ = curve_fit(linear_model, t_norm, norm_corrections, maxfev=5000)
    pred_lin = linear_model(t_norm, *popt_lin)
    rmse_lin = np.sqrt(np.mean((norm_corrections - pred_lin)**2))
except:
    rmse_lin = float('inf')
    popt_lin = [0, 0]

print(f"  Layer-by-layer correction magnitude (normalized):")
for i, l in enumerate(layers_comb):
    print(f"    Layer {l:2d}: {norm_corrections[i]:.4f}")

print()
print(f"  Model fits (RMSE):")
print(f"    JC (exp×cos):   {rmse_jc:.6f}  κ={popt_jc[1]:.4f}, Ω={popt_jc[2]:.4f}")
print(f"    Pure exp decay: {rmse_exp:.6f}  κ={popt_exp[1]:.4f}")
print(f"    Linear:         {rmse_lin:.6f}  slope={popt_lin[0]:.4f}")
print()

# Check if Ω relates to φ
if rmse_jc < float('inf'):
    omega = abs(popt_jc[2])
    print(f"  Oscillation frequency Ω = {omega:.4f}")
    print(f"    vs π/φ = {np.pi/PHI:.4f} (err = {abs(omega - np.pi/PHI):.4f})")
    print(f"    vs φ = {PHI:.4f} (err = {abs(omega - PHI):.4f})")
    print(f"    vs 2π/17 = {2*np.pi/17:.4f} (err = {abs(omega - 2*np.pi/17):.4f})")
    print(f"  Decay rate κ = {abs(popt_jc[1]):.4f}")
    print(f"    vs log(φ)/17 = {LOG_PHI/17:.4f} (err = {abs(abs(popt_jc[1]) - LOG_PHI/17):.4f})")
    print()

# Per-token feedback: does each token follow the same dynamics?
# Check if the SHAPE of the correction curve varies
correction_shapes = layer_corrections / (layer_corrections[:, 0:1] + 1e-10)
shape_std = correction_shapes.std(axis=0)
print(f"  Per-token correction shape std (lower = more universal):")
print(f"    Mean std across layers: {np.mean(shape_std):.4f}")
print(f"    Max std: {np.max(shape_std):.4f} (layer {layers_comb[np.argmax(shape_std)]})")
print()

test4_results = {
    'jc_rmse': float(rmse_jc),
    'exp_rmse': float(rmse_exp),
    'linear_rmse': float(rmse_lin),
    'jc_kappa': float(abs(popt_jc[1])) if rmse_jc < float('inf') else None,
    'jc_omega': float(abs(popt_jc[2])) if rmse_jc < float('inf') else None,
    'correction_shape_mean_std': float(np.mean(shape_std)),
}


# ================================================================
# TEST 5: PAIRED-GATE CORRECTION
# Use null-space pair structure to build additive correction
# for multi-token gate prediction
# ================================================================
print("=" * 80)
print("  TEST 5: PAIRED-GATE CORRECTION")
print("  Can null-space pair correlations improve prediction?")
print("=" * 80)
print()

TEST_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "Albert Einstein developed the theory of",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
    "The chemical symbol for gold is",
    "The color of the sky is usually",
    "A triangle has three sides and three",
    "The square root of 144 is",
]

layer = FOCUS_LAYER
U, S, Vt = W_gate_svds[layer]

# Build a null-space pair model from training tokens
# For each pair of null-space dimensions, compute the cross-correlation
# across training tokens
null_coords_train = null_projs @ Vt_null[:20].T  # (n_train, 20)

# Pair covariance matrix: do even/odd null dims co-vary?
pair_covs = []
for i in range(0, 20, 2):
    cov = np.cov(null_coords_train[:, i], null_coords_train[:, i+1])[0, 1]
    pair_covs.append(cov)
    
print(f"  Null-space pair covariances (training data):")
for i, c in enumerate(pair_covs):
    print(f"    Pair ({2*i}, {2*i+1}): cov = {c:.4f}")
print()

# Now test on multi-token prompts
# For each prompt:
# 1. Get actual gate output and prediction via scaffold + W_gate(h_resid)
# 2. Add null-space correction based on pair correlations
# 3. Compare accuracy with and without correction

results_prompts = []

for prompt in TEST_PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tokens = input_ids.shape[1]

    gate_storage = {}
    hs_storage = {}
    hooks = []

    def make_gh(storage, li):
        def hook_fn(module, input, output):
            storage[li] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hh(storage, li):
        def hook_fn(module, input, output):
            storage[li] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for l in range(N_LAYERS):
        h1 = model.model.layers[l].mlp.gate_proj.register_forward_hook(make_gh(gate_storage, l))
        h2 = model.model.layers[l].mlp.register_forward_hook(make_hh(hs_storage, l))
        hooks.extend([h1, h2])

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :]
        pred_token = tokenizer.decode([torch.argmax(logits).item()])

    for h in hooks:
        h.remove()

    gates = gate_storage[layer]  # (n_tokens, 18944)
    hs = hs_storage[layer]       # (n_tokens, 3584)

    # Last-position gate (what we're trying to predict)
    g_last = gates[-1]
    h_last = hs[-1]

    # Baseline prediction: scaffold + W_gate(h_last - scaffold_h)
    h_resid_last = h_last - scaffold_hidden[layer]
    g_pred_baseline = scaffold_gate[layer] + W_gates[layer] @ h_resid_last

    # This is the column-space prediction — perfect for what W_gate can represent
    # The "error" is in the null space

    # Actual error
    error = g_last - g_pred_baseline
    error_null = error - U @ (U.T @ error)  # null-space component of error
    error_col = U @ (U.T @ error)           # column-space component

    error_null_norm = np.linalg.norm(error_null)
    error_col_norm = np.linalg.norm(error_col)
    error_total_norm = np.linalg.norm(error)

    # Can we predict the null-space error from context?
    # Use the pair structure: for each context position, extract null-space coords
    # and correlate with last-position null-space error
    context_null_coords = []
    for pos in range(n_tokens - 1):
        g_ctx = gates[pos] - scaffold_gate[layer]
        g_ctx_null = g_ctx - U @ (U.T @ g_ctx)
        ctx_null_coord = Vt_null[:20] @ g_ctx_null
        context_null_coords.append(ctx_null_coord)

    if len(context_null_coords) > 0:
        context_null_coords = np.array(context_null_coords)

        # Last-position null-space coords
        last_null_coord = Vt_null[:20] @ error_null

        # Mean context null coords
        mean_ctx_null = context_null_coords.mean(axis=0)

        # Correlation between context mean and last-position null error
        corr_ctx_last = np.dot(mean_ctx_null, last_null_coord) / (
            np.linalg.norm(mean_ctx_null) * np.linalg.norm(last_null_coord) + 1e-10)

        # Build correction: project mean context null coords back to gate space
        # and scale by pair covariance structure
        correction_coords = mean_ctx_null.copy()
        for i in range(0, 20, 2):
            # Scale by pair covariance ratio
            if abs(pair_covs[i//2]) > 1e-6:
                scale = pair_covs[i//2] / (np.var(null_coords_train[:, i]) + 1e-10)
                correction_coords[i] *= scale
                correction_coords[i+1] *= scale

        null_correction = Vt_null[:20].T @ correction_coords

        # Add correction to baseline
        g_pred_corrected = g_pred_baseline + null_correction

        cos_baseline = np.dot(g_last, g_pred_baseline) / (
            np.linalg.norm(g_last) * np.linalg.norm(g_pred_baseline) + 1e-10)
        cos_corrected = np.dot(g_last, g_pred_corrected) / (
            np.linalg.norm(g_last) * np.linalg.norm(g_pred_corrected) + 1e-10)
        err_baseline = np.linalg.norm(g_last - g_pred_baseline) / (np.linalg.norm(g_last) + 1e-10)
        err_corrected = np.linalg.norm(g_last - g_pred_corrected) / (np.linalg.norm(g_last) + 1e-10)
    else:
        cos_baseline = cos_corrected = err_baseline = err_corrected = 0.0
        corr_ctx_last = 0.0
        error_null_norm = error_col_norm = error_total_norm = 0.0

    results_prompts.append({
        'prompt': prompt,
        'pred_token': pred_token.strip(),
        'cos_baseline': float(cos_baseline),
        'cos_corrected': float(cos_corrected),
        'err_baseline': float(err_baseline),
        'err_corrected': float(err_corrected),
        'error_null_frac': float(error_null_norm / (error_total_norm + 1e-10)),
        'error_col_frac': float(error_col_norm / (error_total_norm + 1e-10)),
        'ctx_last_null_corr': float(corr_ctx_last),
    })

    print(f"  '{prompt}' -> '{pred_token.strip()}'")
    print(f"    Baseline cos: {cos_baseline:.6f}, error: {err_baseline:.6f}")
    print(f"    Corrected cos: {cos_corrected:.6f}, error: {err_corrected:.6f}")
    print(f"    Error breakdown: col={error_col_norm/(error_total_norm+1e-10):.3f}, null={error_null_norm/(error_total_norm+1e-10):.3f}")
    print(f"    Context↔last null corr: {corr_ctx_last:.4f}")
    print()

# Summary
mean_improvement = np.mean([r['cos_corrected'] - r['cos_baseline'] for r in results_prompts])
mean_null_frac = np.mean([r['error_null_frac'] for r in results_prompts])
mean_ctx_corr = np.mean([r['ctx_last_null_corr'] for r in results_prompts])

print(f"  SUMMARY:")
print(f"    Mean cosine improvement: {mean_improvement:+.6f}")
print(f"    Mean error in null space: {mean_null_frac:.3f}")
print(f"    Mean context↔last null correlation: {mean_ctx_corr:.4f}")
print()

test5_results = {
    'prompts': results_prompts,
    'mean_cosine_improvement': float(mean_improvement),
    'mean_null_fraction': float(mean_null_frac),
    'mean_ctx_null_corr': float(mean_ctx_corr),
}


# ================================================================
# OVERALL SUMMARY
# ================================================================
print()
print("=" * 80)
print("  OVERALL SUMMARY: JC CAVITY QED MODEL")
print("=" * 80)
print()

print("  TEST 1: NULL-SPACE PAIRS")
print(f"    Paired structure: {'YES' if test1_results['paired_structure'] else 'NO'} (gap={test1_results['mean_pair_gap']:.4f})")
print(f"    Pair advantage for token ID: {test1_results['pair_advantage']:+.4f}")
print(f"    VERDICT: {'PROMISING' if test1_results['pair_advantage'] > 0.01 else 'WEAK'}")
print()

print("  TEST 2: √n LADDER")
print(f"    Power law α = {test2_results['power_law_alpha']:.4f} (JC predicts 0.5)")
print(f"    Mean JC ratio error: {test2_results['mean_jc_ratio_error']:.4f}")
print(f"    VERDICT: {'PROMISING' if abs(test2_results['power_law_alpha'] - 0.5) < 0.1 else 'WEAK'}")
print()

print("  TEST 3: DRESSED-STATE ROUTING")
print(f"    Dressed Gini: {test3_results['dressed_gini']:.4f}")
print(f"    Raw Gini: {test3_results['raw_gini']:.4f}")
print(f"    Semantic pair cos (dressed vs raw): {test3_results['semantic_pair_cos_dressed']:.4f} vs {test3_results['semantic_pair_cos_raw']:.4f}")
print(f"    VERDICT: {'PROMISING' if test3_results['dressed_gini'] > test3_results['raw_gini'] + 0.01 else 'WEAK'}")
print()

print("  TEST 4: FEEDBACK DYNAMICS")
print(f"    JC (exp×cos) RMSE: {test4_results['jc_rmse']:.6f}")
print(f"    Pure exp RMSE: {test4_results['exp_rmse']:.6f}")
print(f"    Linear RMSE: {test4_results['linear_rmse']:.6f}")
best_model = min(['JC', 'exp', 'linear'], key=lambda m: {'JC': test4_results['jc_rmse'], 'exp': test4_results['exp_rmse'], 'linear': test4_results['linear_rmse']}[m])
print(f"    Best fit: {best_model}")
print(f"    VERDICT: {'PROMISING' if best_model == 'JC' else 'WEAK'}")
print()

print("  TEST 5: PAIRED-GATE CORRECTION")
print(f"    Mean cosine improvement: {test5_results['mean_cosine_improvement']:+.6f}")
print(f"    Mean null-space error fraction: {test5_results['mean_null_fraction']:.3f}")
print(f"    VERDICT: {'PROMISING' if test5_results['mean_cosine_improvement'] > 0.0001 else 'WEAK'}")
print()


# ================================================================
# Save results
# ================================================================
del model
torch.cuda.empty_cache()

results = {
    'test1_null_space_pairs': test1_results,
    'test2_sqrt_n_ladder': test2_results,
    'test3_dressed_state_routing': test3_results,
    'test4_feedback_dynamics': test4_results,
    'test5_paired_gate_correction': test5_results,
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8s_jc_cavity.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
