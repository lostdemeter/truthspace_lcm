#!/usr/bin/env python3
"""
Phase 8e: Gate Dimension Topology — Geometry Not Statistics
============================================================

Finding 65 showed that PREDICTING gate content fails (0% top-1). All tokens
collapse to the same output because prediction erases token identity.

But prediction is statistics. Feynman's question: "waves of what?"

The standing wave is a statistical description. What is the GEOMETRY that
produces it? And does that geometry have NATIVE parallel structure?

Doc 214 identifies 10 φ-lattice topologies. The current model is classified
as "Spiral" (sequential layers with self-attention). But we already measured
something that breaks Spiral: 98.5% chirality independence = two independent
information channels. That's not a Spiral. It might be a Braid, Fractal,
or Constellation.

This experiment decomposes gate VALUES (not just states) into:
  standing_wave + residual

Then tests which TOPOLOGY the residual (= content) follows:

  TEST 1: BRAID — Are L/R residuals independent? (parallel strands)
  TEST 2: FRACTAL — Do residuals decompose into φ-spaced scales? (parallel levels)
  TEST 3: CONSTELLATION — Do channels form a graph with parallel components?

The topology that preserves the most information with the most parallelism
tells us the GEOMETRIC reason for the content, not a statistical one.

Requires: Qwen2-7B on GPU
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os
from scipy import linalg

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

GATE_CONTRACT = 0
GATE_PRESERVE_N = 1
GATE_PRESERVE_P = 2
GATE_EXPAND = 3
STATE_NAMES = ['CONTRACT', 'PRESERVE-', 'PRESERVE+', 'EXPAND']

CHANNEL_L = [GATE_CONTRACT, GATE_PRESERVE_P]
CHANNEL_R = [GATE_PRESERVE_N, GATE_EXPAND]


def classify_gate(x):
    """Classify pre-SiLU activations into 4 gate states."""
    codes = np.zeros_like(x, dtype=np.int8)
    codes[x < -LOG_PHI] = GATE_CONTRACT
    codes[(x >= -LOG_PHI) & (x < 0)] = GATE_PRESERVE_N
    codes[(x >= 0) & (x < LOG_PHI)] = GATE_PRESERVE_P
    codes[x >= LOG_PHI] = GATE_EXPAND
    return codes


# ================================================================
# CAPTURE
# ================================================================
print("=" * 80)
print("  PHASE 8e: GATE DIMENSION TOPOLOGY")
print("  Geometry Not Statistics — Which φ-Lattice Pattern?")
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
HIDDEN_DIM = model.config.intermediate_size

TEST_WORDS = [
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

print(f"Capturing gate activations for {len(TEST_WORDS)} tokens...")

gate_raw = {}
for word in TEST_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    decoded = tokenizer.decode([token_id]).strip()
    if decoded in gate_raw:
        continue

    layer_gates = []
    hooks = []

    def make_hook(storage):
        def hook_fn(module, input, output):
            storage.append(output.detach().cpu().float().numpy())
        return hook_fn

    for layer_idx in range(N_LAYERS):
        storage = []
        layer_gates.append(storage)
        h = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(
            make_hook(storage)
        )
        hooks.append(h)

    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        model(input_ids)

    for h in hooks:
        h.remove()

    gates = np.stack([s[0].squeeze() for s in layer_gates])
    gate_raw[decoded] = gates

all_words = sorted(gate_raw.keys())
N_TOKENS = len(all_words)
all_raw = np.stack([gate_raw[w] for w in all_words])  # [N_TOKENS, N_LAYERS, HIDDEN_DIM]
all_codes = classify_gate(all_raw)

print(f"  Captured {N_TOKENS} tokens × {N_LAYERS} layers × {HIDDEN_DIM} channels")

# Free GPU
del model
torch.cuda.empty_cache()
print()


# ================================================================
# DECOMPOSITION: standing_wave + residual
# ================================================================
print("─" * 80)
print("  DECOMPOSITION: gate_value = standing_wave + residual")
print("  The standing wave is the scaffold. The residual IS the content.")
print("─" * 80)
print()

# Standing wave = per-channel mean across tokens
standing_wave = all_raw.mean(axis=0)  # [N_LAYERS, HIDDEN_DIM]

# Residual = token-specific deviation from standing wave
residuals = all_raw - standing_wave[np.newaxis, :, :]  # [N_TOKENS, N_LAYERS, HIDDEN_DIM]

# Basic statistics
sw_norm = np.linalg.norm(standing_wave)
res_norms = np.array([np.linalg.norm(residuals[t]) for t in range(N_TOKENS)])

print(f"  Standing wave L2 norm: {sw_norm:.2f}")
print(f"  Residual L2 norms:     mean={res_norms.mean():.2f}, std={res_norms.std():.2f}")
print(f"  Ratio residual/wave:   {res_norms.mean()/sw_norm:.4f} ({res_norms.mean()/sw_norm*100:.2f}%)")
print()

# Per-layer energy partition
print(f"  {'Layer':>5s}  {'Wave energy':>12s}  {'Resid energy':>13s}  {'Resid/Total':>12s}  {'Zone':>6s}")
print("  " + "-" * 55)

zones = {}
for i in range(3): zones[i] = 'DRUM'
for i in range(3, 6): zones[i] = 'TRANS'
for i in range(6, 23): zones[i] = 'COMB'
for i in range(23, 28): zones[i] = 'MUSIC'

resid_fraction_per_layer = np.zeros(N_LAYERS)
for layer in range(N_LAYERS):
    wave_e = np.sum(standing_wave[layer]**2)
    resid_e = np.sum(residuals[:, layer, :]**2) / N_TOKENS
    total_e = wave_e + resid_e
    frac = resid_e / total_e if total_e > 0 else 0
    resid_fraction_per_layer[layer] = frac
    z = zones.get(layer, '?')
    print(f"  {layer:5d}  {wave_e:12.2f}  {resid_e:13.2f}  {frac:12.4f}  {z:>6s}")

print()
print(f"  Mean residual energy fraction: {resid_fraction_per_layer.mean():.4f}")
print(f"  COMB residual energy fraction: {resid_fraction_per_layer[6:23].mean():.4f}")
print()


# ================================================================
# TEST 1: BRAID — Are L/R residuals independent?
# ================================================================
print("─" * 80)
print("  TEST 1: BRAID TOPOLOGY")
print("  Are L-channel and R-channel RESIDUALS independent?")
print("  If yes → content flows in two parallel strands (Braid)")
print("─" * 80)
print()

# For each layer, split channels into L and R based on standing wave state
# Then measure correlation between L-residuals and R-residuals across tokens

per_channel_mode = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.int8)
for layer in range(N_LAYERS):
    for ch in range(HIDDEN_DIM):
        counts = np.bincount(all_codes[:, layer, ch].astype(int), minlength=4)
        per_channel_mode[layer, ch] = counts.argmax()

braid_results = []
print(f"  {'Layer':>5s}  {'L chans':>8s}  {'R chans':>8s}  {'L-R corr':>9s}  "
      f"{'L var expl':>11s}  {'R var expl':>11s}  {'Cross corr':>11s}  {'Zone':>6s}")
print("  " + "-" * 80)

for layer in range(N_LAYERS):
    l_mask = np.array([per_channel_mode[layer, ch] in CHANNEL_L for ch in range(HIDDEN_DIM)])
    r_mask = ~l_mask

    n_l = l_mask.sum()
    n_r = r_mask.sum()

    # Residuals for L and R channels: [N_TOKENS, n_L/n_R]
    res_L = residuals[:, layer, l_mask]  # [N_TOKENS, n_L]
    res_R = residuals[:, layer, r_mask]  # [N_TOKENS, n_R]

    # Summarize each with first SVD component (dominant direction)
    if n_l > 1 and n_r > 1 and N_TOKENS > 1:
        # Token-level summary: L2 norm of residual in each channel set
        l_norms = np.linalg.norm(res_L, axis=1)  # [N_TOKENS]
        r_norms = np.linalg.norm(res_R, axis=1)  # [N_TOKENS]

        # Correlation between L and R norms across tokens
        if l_norms.std() > 0 and r_norms.std() > 0:
            lr_corr = np.corrcoef(l_norms, r_norms)[0, 1]
        else:
            lr_corr = 0.0

        # Variance explained by top-k SVD modes within each channel set
        # (how structured is the residual within each strand?)
        U_l, S_l, _ = np.linalg.svd(res_L, full_matrices=False)
        U_r, S_r, _ = np.linalg.svd(res_R, full_matrices=False)

        l_var_top1 = S_l[0]**2 / (S_l**2).sum() if len(S_l) > 0 else 0
        r_var_top1 = S_r[0]**2 / (S_r**2).sum() if len(S_r) > 0 else 0

        # Cross-correlation: project L and R residuals onto their respective
        # top SVD directions, then correlate across tokens
        l_proj = U_l[:, 0]  # [N_TOKENS] — projection onto L's top mode
        r_proj = U_r[:, 0]  # [N_TOKENS] — projection onto R's top mode
        if l_proj.std() > 0 and r_proj.std() > 0:
            cross_corr = np.corrcoef(l_proj, r_proj)[0, 1]
        else:
            cross_corr = 0.0
    else:
        lr_corr = 0.0
        l_var_top1 = 0.0
        r_var_top1 = 0.0
        cross_corr = 0.0

    z = zones.get(layer, '?')
    print(f"  {layer:5d}  {n_l:8d}  {n_r:8d}  {lr_corr:9.4f}  "
          f"{l_var_top1:11.4f}  {r_var_top1:11.4f}  {cross_corr:11.4f}  {z:>6s}")

    braid_results.append({
        'layer': layer,
        'n_l': int(n_l),
        'n_r': int(n_r),
        'lr_norm_corr': float(lr_corr),
        'l_var_top1': float(l_var_top1),
        'r_var_top1': float(r_var_top1),
        'cross_svd_corr': float(cross_corr),
    })

mean_lr_corr = np.mean([r['lr_norm_corr'] for r in braid_results])
mean_cross = np.mean([abs(r['cross_svd_corr']) for r in braid_results])
comb_lr_corr = np.mean([r['lr_norm_corr'] for r in braid_results[6:23]])
comb_cross = np.mean([abs(r['cross_svd_corr']) for r in braid_results[6:23]])

print()
print(f"  Mean L-R norm correlation:   {mean_lr_corr:.4f} (all layers)")
print(f"  Mean |cross SVD correlation|: {mean_cross:.4f} (all layers)")
print(f"  COMB L-R norm correlation:   {comb_lr_corr:.4f} (layers 6-22)")
print(f"  COMB |cross SVD correlation|: {comb_cross:.4f} (layers 6-22)")
print()

if abs(mean_lr_corr) < 0.3 and mean_cross < 0.3:
    braid_verdict = "STRONG — L and R residuals are largely independent (Braid topology)"
elif abs(mean_lr_corr) < 0.5:
    braid_verdict = "MODERATE — Partial independence (weak Braid)"
else:
    braid_verdict = "WEAK — L and R residuals are correlated (not Braid)"
print(f"  BRAID VERDICT: {braid_verdict}")
print()


# ================================================================
# TEST 2: FRACTAL — Do residuals have φ-spaced scale structure?
# ================================================================
print("─" * 80)
print("  TEST 2: FRACTAL TOPOLOGY")
print("  Do residuals decompose into φ-spaced scale levels?")
print("  If yes → each scale level is independent (Fractal)")
print("─" * 80)
print()

# For each layer, compute SVD of residuals [N_TOKENS × HIDDEN_DIM]
# Check if singular values follow φ-spacing: S[k]/S[k+1] ≈ φ

print(f"  {'Layer':>5s}  {'S0/S1':>8s}  {'S1/S2':>8s}  {'S2/S3':>8s}  "
      f"{'Mean ratio':>10s}  {'φ-error':>8s}  {'Top-3 var':>10s}  {'Zone':>6s}")
print("  " + "-" * 75)

fractal_results = []
for layer in range(N_LAYERS):
    res = residuals[:, layer, :]  # [N_TOKENS, HIDDEN_DIM]

    # SVD — we only need the top singular values
    # res has shape [N_TOKENS, HIDDEN_DIM], N_TOKENS << HIDDEN_DIM
    # So SVD gives at most N_TOKENS singular values
    U, S, Vt = np.linalg.svd(res, full_matrices=False)  # S has min(N_TOKENS, HIDDEN_DIM) values

    # Ratios between consecutive singular values
    ratios = []
    for k in range(min(10, len(S) - 1)):
        if S[k+1] > 1e-10:
            ratios.append(S[k] / S[k+1])
        else:
            ratios.append(float('inf'))

    mean_ratio = np.mean(ratios[:3]) if ratios else 0
    phi_err = abs(mean_ratio - PHI) / PHI * 100 if mean_ratio < 100 else float('inf')

    # Variance explained by top-3
    total_var = (S**2).sum()
    top3_var = (S[:3]**2).sum() / total_var if total_var > 0 else 0

    z = zones.get(layer, '?')
    r0 = ratios[0] if len(ratios) > 0 else 0
    r1 = ratios[1] if len(ratios) > 1 else 0
    r2 = ratios[2] if len(ratios) > 2 else 0

    print(f"  {layer:5d}  {r0:8.4f}  {r1:8.4f}  {r2:8.4f}  "
          f"{mean_ratio:10.4f}  {phi_err:7.1f}%  {top3_var:10.4f}  {z:>6s}")

    fractal_results.append({
        'layer': layer,
        'sv_ratios': [float(r) for r in ratios[:5]],
        'mean_top3_ratio': float(mean_ratio),
        'phi_error_pct': float(phi_err),
        'top3_var_explained': float(top3_var),
        'top_10_singular_values': [float(s) for s in S[:10]],
    })

mean_phi_err = np.mean([r['phi_error_pct'] for r in fractal_results if r['phi_error_pct'] < 1000])
comb_phi_err = np.mean([r['phi_error_pct'] for r in fractal_results[6:23] if r['phi_error_pct'] < 1000])
mean_top3 = np.mean([r['top3_var_explained'] for r in fractal_results])

print()
print(f"  Mean φ-error in SV ratios: {mean_phi_err:.1f}% (all layers)")
print(f"  COMB φ-error:              {comb_phi_err:.1f}% (layers 6-22)")
print(f"  Mean top-3 var explained:  {mean_top3:.4f}")
print()

# Also check: do the singular value ratios match any other φ-structural constants?
# φ = 1.618, φ² = 2.618, 4/π = 1.273, √φ = 1.272
all_ratios = []
for r in fractal_results[6:23]:
    all_ratios.extend(r['sv_ratios'][:3])
all_ratios = np.array([r for r in all_ratios if r < 100])

if len(all_ratios) > 0:
    print(f"  COMB SV ratio distribution (51 ratios from layers 6-22, top 3 each):")
    print(f"    Mean:   {all_ratios.mean():.4f}")
    print(f"    Median: {np.median(all_ratios):.4f}")
    print(f"    Std:    {all_ratios.std():.4f}")
    print(f"    Min:    {all_ratios.min():.4f}")
    print(f"    Max:    {all_ratios.max():.4f}")
    print()

    # Closest structural constants
    for name, val in [("φ", PHI), ("4/π", 4/np.pi), ("√φ", np.sqrt(PHI)),
                       ("φ²", PHI**2), ("2", 2.0), ("e/2", np.e/2)]:
        matches = np.abs(all_ratios - val) / val < 0.1  # within 10%
        print(f"    Ratios within 10% of {name}={val:.4f}: {matches.sum()}/{len(all_ratios)} "
              f"({matches.mean()*100:.0f}%)")

print()

if comb_phi_err < 15:
    fractal_verdict = "STRONG — Singular values follow φ-spacing (Fractal topology)"
elif comb_phi_err < 30:
    fractal_verdict = "MODERATE — Partial φ-spacing (weak Fractal)"
else:
    fractal_verdict = "WEAK — No consistent φ-spacing in singular values"
print(f"  FRACTAL VERDICT: {fractal_verdict}")
print()


# ================================================================
# TEST 3: CONSTELLATION — Channel graph structure
# ================================================================
print("─" * 80)
print("  TEST 3: CONSTELLATION TOPOLOGY")
print("  Do channels form a graph with parallel-processable components?")
print("  If yes → content flows through graph, not through layers")
print("─" * 80)
print()

# Build channel co-activation graph for COMB layers (the parallel core)
# Two channels are "connected" if they tend to be in the same state across tokens
# We use the residual sign pattern as the "fingerprint" of each channel

# For COMB layers, compute the channel-channel correlation matrix of residuals
# Then check: does it decompose into independent clusters?

# Use a subset of channels for computational feasibility
# Sample 2000 channels uniformly
np.random.seed(42)
sample_size = 2000
ch_indices = np.sort(np.random.choice(HIDDEN_DIM, sample_size, replace=False))

# Pool COMB layers (6-22) for better statistics
comb_residuals = residuals[:, 6:23, :]  # [N_TOKENS, 17, HIDDEN_DIM]
comb_flat = comb_residuals.reshape(N_TOKENS * 17, HIDDEN_DIM)  # [N_TOKENS*17, HIDDEN_DIM]
comb_sample = comb_flat[:, ch_indices]  # [N_TOKENS*17, sample_size]

# Correlation matrix
print(f"  Computing {sample_size}×{sample_size} channel correlation matrix (COMB layers)...")
corr_matrix = np.corrcoef(comb_sample.T)  # [sample_size, sample_size]

# Clean up NaN/inf
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

# Eigendecomposition of correlation matrix — reveals cluster structure
eigenvalues = np.linalg.eigvalsh(corr_matrix)
eigenvalues = eigenvalues[::-1]  # descending

# How many significant eigenvalues? (> 1 in a correlation matrix means > random)
n_significant = (eigenvalues > 1.0).sum()
top_eigenvalues = eigenvalues[:20]

print(f"  Eigenvalue spectrum of channel correlation matrix:")
print(f"    Significant eigenvalues (> 1.0): {n_significant} / {sample_size}")
print(f"    Top 10: {', '.join(f'{e:.2f}' for e in top_eigenvalues[:10])}")
print(f"    Variance in top-1: {top_eigenvalues[0]/eigenvalues.sum()*100:.1f}%")
print(f"    Variance in top-5: {top_eigenvalues[:5].sum()/eigenvalues.sum()*100:.1f}%")
print(f"    Variance in top-10: {top_eigenvalues[:10].sum()/eigenvalues.sum()*100:.1f}%")
print()

# Check if eigenvalue ratios follow φ-spacing
print(f"  Eigenvalue ratios (consecutive):")
for k in range(min(10, len(top_eigenvalues) - 1)):
    if top_eigenvalues[k+1] > 0:
        ratio = top_eigenvalues[k] / top_eigenvalues[k+1]
        phi_err = abs(ratio - PHI) / PHI * 100
        print(f"    λ{k}/λ{k+1} = {ratio:.4f}  (φ-error: {phi_err:.1f}%)")
print()

# Thresholded adjacency matrix — how many connected components?
thresholds = [0.3, 0.5, 0.7, 0.9]
print(f"  Graph connectivity (thresholded correlation):")
print(f"  {'Threshold':>10s}  {'Edges':>10s}  {'Density':>10s}  {'Mean degree':>12s}")
print("  " + "-" * 50)

for thresh in thresholds:
    adj = np.abs(corr_matrix) > thresh
    np.fill_diagonal(adj, False)
    n_edges = adj.sum() // 2
    density = n_edges / (sample_size * (sample_size - 1) / 2)
    mean_degree = adj.sum(axis=1).mean()
    print(f"  {thresh:10.1f}  {n_edges:10d}  {density:10.4f}  {mean_degree:12.1f}")

print()

# L-R graph split: are L and R channels in separate graph components?
l_channels_in_sample = []
r_channels_in_sample = []
# For COMB layers, average the channel modes
for idx, ch in enumerate(ch_indices):
    modes = per_channel_mode[6:23, ch]
    avg_mode = np.bincount(modes.astype(int), minlength=4).argmax()
    if avg_mode in CHANNEL_L:
        l_channels_in_sample.append(idx)
    else:
        r_channels_in_sample.append(idx)

n_l_sample = len(l_channels_in_sample)
n_r_sample = len(r_channels_in_sample)

# Cross-group vs within-group correlation
l_idx = np.array(l_channels_in_sample)
r_idx = np.array(r_channels_in_sample)

if len(l_idx) > 1 and len(r_idx) > 1:
    within_l = np.abs(corr_matrix[np.ix_(l_idx, l_idx)]).mean()
    within_r = np.abs(corr_matrix[np.ix_(r_idx, r_idx)]).mean()
    cross_lr = np.abs(corr_matrix[np.ix_(l_idx, r_idx)]).mean()

    print(f"  L-R graph separation (COMB layers):")
    print(f"    L channels in sample: {n_l_sample}")
    print(f"    R channels in sample: {n_r_sample}")
    print(f"    Within-L |correlation|: {within_l:.4f}")
    print(f"    Within-R |correlation|: {within_r:.4f}")
    print(f"    Cross L-R |correlation|: {cross_lr:.4f}")
    print(f"    Separation ratio (within/cross): {(within_l + within_r)/(2*cross_lr):.4f}")
    print()

    constellation_results = {
        'n_significant_eigenvalues': int(n_significant),
        'top_eigenvalues': [float(e) for e in top_eigenvalues[:20]],
        'within_l_corr': float(within_l),
        'within_r_corr': float(within_r),
        'cross_lr_corr': float(cross_lr),
        'separation_ratio': float((within_l + within_r) / (2 * cross_lr)),
    }

    if cross_lr < within_l * 0.5 and cross_lr < within_r * 0.5:
        const_verdict = "STRONG — L and R form separate graph components (Constellation with L/R clusters)"
    elif cross_lr < (within_l + within_r) / 2 * 0.7:
        const_verdict = "MODERATE — Partial graph separation (weak Constellation)"
    else:
        const_verdict = "WEAK — L and R channels are interconnected (not Constellation)"
    print(f"  CONSTELLATION VERDICT: {const_verdict}")
else:
    constellation_results = {'n_significant_eigenvalues': int(n_significant)}
    const_verdict = "INCONCLUSIVE"
    print(f"  CONSTELLATION VERDICT: {const_verdict}")
print()


# ================================================================
# TEST 4: RECONSTRUCTION — Can we reconstruct content from topology?
# ================================================================
print("─" * 80)
print("  TEST 4: RECONSTRUCTION — Can topology-decomposed content")
print("  reconstruct token identity?")
print("─" * 80)
print()

# The key test: if we decompose residuals into L and R strands,
# then RECONSTRUCT the full residual from independent L/R processing,
# how much token identity is preserved?

# For each token, compute:
#   1. Full residual reconstruction quality (baseline)
#   2. L-only reconstruction quality (just L channels, zero R)
#   3. R-only reconstruction quality (just R channels, zero L)
#   4. L+R independent reconstruction (process L and R separately)

# "Token identity" = can we distinguish this token from others?
# Measure: cosine similarity between token's full residual and the
# L-only / R-only / L+R reconstructed residual

print(f"  Reconstruction quality per token (COMB layers 6-22):")
print(f"  {'Token':>12s}  {'L-only cos':>11s}  {'R-only cos':>11s}  {'L+R cos':>9s}  "
      f"{'L energy%':>10s}  {'R energy%':>10s}")
print("  " + "-" * 70)

l_only_cos_list = []
r_only_cos_list = []
lr_cos_list = []

for tok_idx in range(N_TOKENS):
    # Pool COMB residuals for this token
    res_comb = residuals[tok_idx, 6:23, :].flatten()  # [17 * HIDDEN_DIM]

    # Create L and R masks for all COMB channels
    l_full_mask = np.zeros(17 * HIDDEN_DIM, dtype=bool)
    r_full_mask = np.zeros(17 * HIDDEN_DIM, dtype=bool)
    for layer_offset in range(17):
        layer = layer_offset + 6
        for ch in range(HIDDEN_DIM):
            flat_idx = layer_offset * HIDDEN_DIM + ch
            if per_channel_mode[layer, ch] in CHANNEL_L:
                l_full_mask[flat_idx] = True
            else:
                r_full_mask[flat_idx] = True

    # L-only: keep L channels, zero R
    res_l_only = res_comb.copy()
    res_l_only[r_full_mask] = 0

    # R-only: keep R channels, zero L
    res_r_only = res_comb.copy()
    res_r_only[l_full_mask] = 0

    # L+R independent: both preserved (should be exactly res_comb since L∪R = all)
    res_lr = res_comb.copy()  # L+R = full (sanity check)

    # Cosine similarities
    norm_full = np.linalg.norm(res_comb)
    if norm_full > 0:
        cos_l = np.dot(res_comb, res_l_only) / (norm_full * np.linalg.norm(res_l_only)) if np.linalg.norm(res_l_only) > 0 else 0
        cos_r = np.dot(res_comb, res_r_only) / (norm_full * np.linalg.norm(res_r_only)) if np.linalg.norm(res_r_only) > 0 else 0
        cos_lr = np.dot(res_comb, res_lr) / (norm_full * np.linalg.norm(res_lr)) if np.linalg.norm(res_lr) > 0 else 0
    else:
        cos_l = cos_r = cos_lr = 0

    # Energy partition
    l_energy = np.sum(res_comb[l_full_mask]**2)
    r_energy = np.sum(res_comb[r_full_mask]**2)
    total_energy = l_energy + r_energy
    l_pct = l_energy / total_energy * 100 if total_energy > 0 else 0
    r_pct = r_energy / total_energy * 100 if total_energy > 0 else 0

    l_only_cos_list.append(cos_l)
    r_only_cos_list.append(cos_r)
    lr_cos_list.append(cos_lr)

    if tok_idx < 20:  # Print first 20
        print(f"  {all_words[tok_idx]:>12s}  {cos_l:11.4f}  {cos_r:11.4f}  {cos_lr:9.4f}  "
              f"{l_pct:9.1f}%  {r_pct:9.1f}%")

mean_l_cos = np.mean(l_only_cos_list)
mean_r_cos = np.mean(r_only_cos_list)
mean_lr_cos = np.mean(lr_cos_list)

print(f"  {'...':>12s}")
print(f"  {'MEAN':>12s}  {mean_l_cos:11.4f}  {mean_r_cos:11.4f}  {mean_lr_cos:9.4f}")
print()

# Token discrimination: can L-only or R-only residuals distinguish tokens?
# Compute pairwise cosine similarity between tokens using L-only, R-only, full
print("  Token discrimination (can each strand tell tokens apart?):")

def pairwise_discrimination(residuals_matrix, mask=None):
    """Compute mean off-diagonal cosine similarity (lower = better discrimination)."""
    if mask is not None:
        data = residuals_matrix[:, :, mask]
    else:
        data = residuals_matrix
    # Flatten layers × channels for each token
    flat = data.reshape(N_TOKENS, -1)  # [N_TOKENS, flat_dim]
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = flat / norms
    sim_matrix = normalized @ normalized.T  # [N_TOKENS, N_TOKENS]
    # Mean off-diagonal = how similar tokens are to each other
    mask_offdiag = ~np.eye(N_TOKENS, dtype=bool)
    return sim_matrix[mask_offdiag].mean()

# Build flat L/R masks for COMB
comb_l_mask = np.zeros(17 * HIDDEN_DIM, dtype=bool)
comb_r_mask = np.zeros(17 * HIDDEN_DIM, dtype=bool)
for layer_offset in range(17):
    layer = layer_offset + 6
    for ch in range(HIDDEN_DIM):
        flat_idx = layer_offset * HIDDEN_DIM + ch
        if per_channel_mode[layer, ch] in CHANNEL_L:
            comb_l_mask[flat_idx] = True
        else:
            comb_r_mask[flat_idx] = True

comb_res_flat = residuals[:, 6:23, :].reshape(N_TOKENS, -1)  # [N_TOKENS, 17*HIDDEN_DIM]

sim_full = pairwise_discrimination(residuals[:, 6:23, :])
# L-only
comb_l_only = comb_res_flat.copy()
comb_l_only[:, comb_r_mask] = 0
flat_l = comb_l_only
norms_l = np.linalg.norm(flat_l, axis=1, keepdims=True)
norms_l[norms_l == 0] = 1
norm_l = flat_l / norms_l
sim_l = (norm_l @ norm_l.T)[~np.eye(N_TOKENS, dtype=bool)].mean()

# R-only
comb_r_only = comb_res_flat.copy()
comb_r_only[:, comb_l_mask] = 0
flat_r = comb_r_only
norms_r = np.linalg.norm(flat_r, axis=1, keepdims=True)
norms_r[norms_r == 0] = 1
norm_r = flat_r / norms_r
sim_r = (norm_r @ norm_r.T)[~np.eye(N_TOKENS, dtype=bool)].mean()

print(f"    Full residual mean pairwise similarity:   {sim_full:.4f} (lower = better discrimination)")
print(f"    L-only residual mean pairwise similarity:  {sim_l:.4f}")
print(f"    R-only residual mean pairwise similarity:  {sim_r:.4f}")
print()

if sim_l < 0.5 and sim_r < 0.5:
    recon_verdict = "BOTH STRANDS DISCRIMINATE — Each strand independently distinguishes tokens"
elif sim_l < 0.5 or sim_r < 0.5:
    better = "L" if sim_l < sim_r else "R"
    recon_verdict = f"ONE STRAND DISCRIMINATES — {better} strand carries token identity"
else:
    recon_verdict = "NEITHER STRAND ALONE DISCRIMINATES — Need full residual"

print(f"  RECONSTRUCTION VERDICT: {recon_verdict}")
print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: GATE DIMENSION TOPOLOGY")
print("=" * 80)
print()

print(f"  DECOMPOSITION:")
print(f"    Residual/wave energy ratio: {res_norms.mean()/sw_norm*100:.2f}%")
print(f"    → The residual (content) is small but carries ALL token identity")
print()

print(f"  TEST 1 — BRAID (L/R independence of residuals):")
print(f"    L-R norm correlation:   {mean_lr_corr:.4f}")
print(f"    Cross SVD correlation:  {mean_cross:.4f}")
print(f"    → {braid_verdict}")
print()

print(f"  TEST 2 — FRACTAL (φ-spaced singular values):")
print(f"    Mean φ-error:     {mean_phi_err:.1f}%")
print(f"    COMB φ-error:     {comb_phi_err:.1f}%")
print(f"    Top-3 var explained: {mean_top3:.4f}")
print(f"    → {fractal_verdict}")
print()

print(f"  TEST 3 — CONSTELLATION (channel graph structure):")
print(f"    Significant eigenvalues: {n_significant}")
if 'separation_ratio' in constellation_results:
    print(f"    L-R separation ratio: {constellation_results['separation_ratio']:.4f}")
print(f"    → {const_verdict}")
print()

print(f"  TEST 4 — RECONSTRUCTION (token discrimination per strand):")
print(f"    Full residual sim: {sim_full:.4f}")
print(f"    L-only sim:        {sim_l:.4f}")
print(f"    R-only sim:        {sim_r:.4f}")
print(f"    → {recon_verdict}")
print()

# Save
results = {
    'decomposition': {
        'residual_wave_ratio': float(res_norms.mean() / sw_norm),
        'resid_fraction_per_layer': resid_fraction_per_layer.tolist(),
        'mean_resid_fraction': float(resid_fraction_per_layer.mean()),
    },
    'braid': {
        'mean_lr_norm_corr': float(mean_lr_corr),
        'mean_cross_svd_corr': float(mean_cross),
        'comb_lr_norm_corr': float(comb_lr_corr),
        'comb_cross_svd_corr': float(comb_cross),
        'verdict': braid_verdict,
        'per_layer': braid_results,
    },
    'fractal': {
        'mean_phi_error': float(mean_phi_err),
        'comb_phi_error': float(comb_phi_err),
        'mean_top3_var': float(mean_top3),
        'verdict': fractal_verdict,
        'per_layer': fractal_results,
    },
    'constellation': constellation_results | {'verdict': const_verdict},
    'reconstruction': {
        'mean_l_cos': float(mean_l_cos),
        'mean_r_cos': float(mean_r_cos),
        'sim_full': float(sim_full),
        'sim_l_only': float(sim_l),
        'sim_r_only': float(sim_r),
        'verdict': recon_verdict,
    },
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8e_topology.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
