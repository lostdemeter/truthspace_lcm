#!/usr/bin/env python3
"""
Phase 8t: Dressed-State Shape Filter
======================================

Following the lead from Finding 80: W_gate's SVD modes are a more
natural coordinate system. The user's intuition: W_gate is an
"atomic-level shape filter" that only allows certain shapes through.

Key questions:
1. MODE SELECTIVITY: Which SVD modes discriminate tokens vs which
   are universal (scaffold-like)?
2. XOR SHAPES: Do pairs of modes jointly discriminate better than
   individual modes? (Combinatorial/XOR logic in the filter)
3. SHAPE CATALOG: What are the distinct "shapes" that pass through
   W_gate? Do semantic clusters emerge?
4. FILTER BANDWIDTH: How many modes carry signal per token?
   Does bandwidth vary by token type?
5. MINIMUM SHAPE: What's the smallest mode set that preserves
   token identity? Same for all tokens or token-specific?

The XOR hypothesis: token identity isn't in individual modes but
in COMBINATIONS — like XOR needs two bits to define a pattern.
If true, this explains why rank-1 gets 100% for single tokens
(one shape suffices) but multi-token needs 7 dimensions (XOR
combinations of shapes).
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os
from itertools import combinations

PHI = (1 + np.sqrt(5)) / 2
SQRT_PHI = np.sqrt(PHI)
LOG_PHI = np.log(PHI)

COMB_START = 6
COMB_END = 23
FOCUS_LAYER = 14

print("=" * 80)
print("  PHASE 8t: DRESSED-STATE SHAPE FILTER")
print("  W_gate as atomic shape filter — what shapes pass through?")
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
# STEP 0: Extract W_gate SVD for focus layer
# ================================================================
print("-" * 80)
print("  STEP 0: Extract W_gate SVD")
print("-" * 80)

W = model.model.layers[FOCUS_LAYER].mlp.gate_proj.weight.data.float().cpu().numpy()
print(f"  W_gate shape: {W.shape}")  # (18944, 3584)

print(f"  Computing full SVD...", end=" ", flush=True)
U, S, Vt = np.linalg.svd(W, full_matrices=False)
print(f"done. rank={np.sum(S > 1e-6)}")
print(f"  S[0]={S[0]:.4f}, S[-1]={S[-1]:.6f}, cond={S[0]/S[-1]:.1f}")
print()

# The "shapes" are the right singular vectors (Vt rows) — they define
# the hidden-space patterns that W_gate selects for.
# U columns define what those patterns look like in gate space.
# S values define the filter gain (selectivity) for each shape.


# ================================================================
# STEP 1: Collect token data in dressed-state basis
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
n_tokens = len(train_words)

# Build scaffold
all_gates = np.stack([single_gates[w] for w in train_words])
all_hs = np.stack([single_hs[w] for w in train_words])
scaffold_gate = all_gates.mean(axis=0)
scaffold_hidden = all_hs.mean(axis=0)

print(f"  Crystal: {n_tokens} tokens")
print()

# Compute dressed-state coordinates for all tokens at focus layer
# dressed[i] = U^T @ (g_i - scaffold_g) = S * (Vt @ (h_i - scaffold_h))
# These are the coordinates in the SVD basis, weighted by singular values

gate_resids = np.array([single_gates[w][FOCUS_LAYER] - scaffold_gate[FOCUS_LAYER] for w in train_words])
hidden_resids = np.array([single_hs[w][FOCUS_LAYER] - scaffold_hidden[FOCUS_LAYER] for w in train_words])

# Dressed coordinates in gate space (project onto left singular vectors)
dressed_gate = gate_resids @ U  # (n_tokens, 3584) — coordinates along each SVD mode

# Dressed coordinates in hidden space (project onto right singular vectors)
dressed_hidden = hidden_resids @ Vt.T  # (n_tokens, 3584)

# Verify: dressed_gate[i,k] = S[k] * dressed_hidden[i,k]
ratio_check = dressed_gate[:, :10] / (dressed_hidden[:, :10] * S[:10] + 1e-10)
print(f"  Verification: dressed_gate / (S * dressed_hidden) = {np.mean(ratio_check):.6f} (should be ~1.0)")
print()


# ================================================================
# TEST 1: MODE SELECTIVITY MAP
# Which modes discriminate tokens? Which are universal?
# ================================================================
print("=" * 80)
print("  TEST 1: MODE SELECTIVITY MAP")
print("  Which SVD modes discriminate tokens vs are universal?")
print("=" * 80)
print()

# For each mode k, measure:
# - Mean coordinate (should be ~0 after scaffold subtraction)
# - Std across tokens (higher = more discriminative)
# - Max/min range
# - F-ratio: between-token variance / within-token variance

mode_stds = np.std(dressed_gate, axis=0)       # (3584,)
mode_means = np.mean(dressed_gate, axis=0)      # (3584,)
mode_ranges = np.ptp(dressed_gate, axis=0)      # peak-to-peak range
mode_energy = np.mean(dressed_gate**2, axis=0)  # mean squared coordinate

# Selectivity = how well each mode separates tokens
# Use coefficient of variation (std/mean_abs) or just raw variance
# Normalized selectivity: mode_std / S[k] (variance relative to filter gain)
mode_selectivity = mode_stds / (S + 1e-10)

print(f"  Top 20 most selective modes (highest std/S ratio):")
print(f"  {'Mode':>5s}  {'S[k]':>8s}  {'Std':>10s}  {'Selectivity':>12s}  {'Range':>10s}  {'Energy':>10s}")
print("  " + "-" * 60)

top_selective = np.argsort(mode_selectivity)[::-1][:20]
for k in top_selective:
    print(f"  {k:5d}  {S[k]:8.4f}  {mode_stds[k]:10.4f}  {mode_selectivity[k]:12.6f}  {mode_ranges[k]:10.4f}  {mode_energy[k]:10.4f}")

print()

# Do the most discriminative modes cluster at high or low singular values?
# Partition modes into bands
n_modes = len(S)
band_size = n_modes // 10
print(f"  Selectivity by SV band (mode index range → mean selectivity):")
band_selectivities = []
for b in range(10):
    start = b * band_size
    end = (b + 1) * band_size if b < 9 else n_modes
    band_sel = np.mean(mode_selectivity[start:end])
    band_std = np.mean(mode_stds[start:end])
    band_s = np.mean(S[start:end])
    band_selectivities.append(band_sel)
    print(f"    Modes {start:4d}-{end:4d}: selectivity={band_sel:.6f}, raw_std={band_std:.4f}, mean_S={band_s:.4f}")

print()

# Key question: does token info concentrate in specific modes?
# Cumulative selectivity explained
sorted_sel_idx = np.argsort(mode_stds)[::-1]
cum_var = np.cumsum(mode_stds[sorted_sel_idx]**2) / np.sum(mode_stds**2)
for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
    n_needed = int(np.searchsorted(cum_var, thresh) + 1)
    print(f"  {thresh*100:.0f}% of token discrimination: {n_needed} modes ({n_needed/n_modes*100:.1f}%)")

print()


# ================================================================
# TEST 2: XOR SHAPE DETECTION
# Do mode PAIRS discriminate better than individual modes?
# ================================================================
print("=" * 80)
print("  TEST 2: XOR SHAPE DETECTION")
print("  Do mode pairs discriminate better than singles?")
print("=" * 80)
print()

# For efficiency, test the top 30 most selective modes
top_k = 30
top_modes = np.argsort(mode_stds)[::-1][:top_k]

# Metric: nearest-neighbor accuracy for token identity
# Ground truth: each token's NN in full hidden space
from sklearn.metrics import pairwise_distances

full_h_dists = pairwise_distances(hidden_resids)
np.fill_diagonal(full_h_dists, np.inf)
nn_truth = np.argmin(full_h_dists, axis=1)

# Single-mode NN accuracy
single_accs = {}
for k in top_modes:
    coords = dressed_gate[:, k:k+1]
    dists = pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    acc = np.mean(nn == nn_truth)
    single_accs[k] = acc

# Pair-mode NN accuracy — test all pairs of top modes
pair_accs = {}
xor_gains = []  # (pair_acc - max(single_acc_a, single_acc_b))

print(f"  Testing {top_k}C2 = {top_k*(top_k-1)//2} mode pairs...")

best_pairs = []
for i, (a, b) in enumerate(combinations(top_modes, 2)):
    coords = dressed_gate[:, [a, b]]
    dists = pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    acc = np.mean(nn == nn_truth)
    pair_accs[(a, b)] = acc

    max_single = max(single_accs[a], single_accs[b])
    xor_gain = acc - max_single
    xor_gains.append((xor_gain, acc, a, b, single_accs[a], single_accs[b]))

# Sort by XOR gain (pair advantage over best single)
xor_gains.sort(key=lambda x: -x[0])

print()
print(f"  Top 20 XOR pairs (biggest gain from pairing):")
print(f"  {'Mode A':>7s}  {'Mode B':>7s}  {'Single A':>9s}  {'Single B':>9s}  {'Pair':>7s}  {'XOR Gain':>9s}")
print("  " + "-" * 55)
for gain, acc, a, b, sa, sb in xor_gains[:20]:
    marker = " ← XOR!" if gain > 0.05 else ""
    print(f"  {a:7d}  {b:7d}  {sa:9.3f}  {sb:9.3f}  {acc:7.3f}  {gain:+9.3f}{marker}")

print()

# Statistics
xor_gain_values = [x[0] for x in xor_gains]
print(f"  XOR gain statistics:")
print(f"    Mean: {np.mean(xor_gain_values):+.4f}")
print(f"    Max:  {np.max(xor_gain_values):+.4f}")
print(f"    Pairs with gain > 0.05: {sum(1 for g in xor_gain_values if g > 0.05)}/{len(xor_gain_values)}")
print(f"    Pairs with gain > 0.10: {sum(1 for g in xor_gain_values if g > 0.10)}/{len(xor_gain_values)}")
print()

# Compare: best single mode vs best pair vs best triple
best_single_k = max(single_accs, key=single_accs.get)
best_single_acc = single_accs[best_single_k]

best_pair = xor_gains[0]
best_pair_acc = best_pair[1]
best_pair_modes = (best_pair[2], best_pair[3])

# Test triples of the top 10 modes
top_10 = top_modes[:10]
best_triple_acc = 0
best_triple_modes = None
for a, b, c in combinations(top_10, 3):
    coords = dressed_gate[:, [a, b, c]]
    dists = pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    acc = np.mean(nn == nn_truth)
    if acc > best_triple_acc:
        best_triple_acc = acc
        best_triple_modes = (a, b, c)

print(f"  Scaling: modes → NN accuracy")
print(f"    Best single mode {best_single_k}: {best_single_acc:.3f}")
print(f"    Best pair {best_pair_modes}: {best_pair_acc:.3f}")
print(f"    Best triple {best_triple_modes}: {best_triple_acc:.3f}")

# Also test cumulative: top-1, top-2, ..., top-20 modes by selectivity
print()
print(f"  Cumulative top-k modes (by selectivity):")
for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200]:
    if k > n_modes:
        break
    sel_modes = np.argsort(mode_stds)[::-1][:k]
    coords = dressed_gate[:, sel_modes]
    dists = pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    acc = np.mean(nn == nn_truth)
    print(f"    Top {k:3d} modes: acc = {acc:.3f}")

print()


# ================================================================
# TEST 3: SHAPE CATALOG
# What distinct "shapes" does W_gate select for?
# ================================================================
print("=" * 80)
print("  TEST 3: SHAPE CATALOG")
print("  What are the distinct shapes that pass through W_gate?")
print("=" * 80)
print()

# Each token's "shape" is its profile across the top selective modes
# Cluster tokens by shape similarity
top_shape_modes = np.argsort(mode_stds)[::-1][:20]
shape_profiles = dressed_gate[:, top_shape_modes]  # (n_tokens, 20)

# Normalize to unit vectors (shape = direction, not magnitude)
shape_norms = np.linalg.norm(shape_profiles, axis=1, keepdims=True)
shape_dirs = shape_profiles / (shape_norms + 1e-10)

# Cosine similarity matrix
cos_sim = shape_dirs @ shape_dirs.T

# Find clusters via hierarchical structure
# Which tokens have most similar shapes?
print(f"  Token shape similarity (top 20 modes, cosine):")
print(f"  Most similar pairs:")

# Extract upper triangle
n = len(train_words)
pair_sims = []
for i in range(n):
    for j in range(i+1, n):
        pair_sims.append((cos_sim[i, j], train_words[i], train_words[j]))

pair_sims.sort(key=lambda x: -abs(x[0]))

print(f"  {'Cos':>8s}  {'Token A':>10s}  {'Token B':>10s}")
print("  " + "-" * 32)
for sim, wa, wb in pair_sims[:20]:
    print(f"  {sim:8.4f}  {wa:>10s}  {wb:>10s}")

print()

# Anti-correlated pairs (opposite shapes)
print(f"  Most anti-correlated pairs (opposite shapes):")
pair_sims_neg = sorted(pair_sims, key=lambda x: x[0])
for sim, wa, wb in pair_sims_neg[:10]:
    print(f"  {sim:8.4f}  {wa:>10s}  {wb:>10s}")

print()

# Semantic categories — do they cluster?
categories = {
    'royalty': ['king', 'queen'],
    'gender_m': ['man', 'boy', 'king'],
    'gender_f': ['woman', 'girl', 'queen'],
    'temp': ['hot', 'cold'],
    'speed': ['fast', 'slow'],
    'size': ['big', 'small'],
    'emotion': ['love', 'hate', 'happy', 'sad'],
    'light': ['light', 'dark'],
    'truth': ['true', 'false'],
    'animal': ['cat', 'dog'],
    'nature': ['tree', 'water', 'fire', 'earth'],
    'color': ['red', 'blue', 'green', 'black', 'white', 'yellow'],
    'number': ['zero', 'one', 'two', 'three', 'four', 'five'],
    'function': ['the', 'is', 'and', 'of', 'to', 'in'],
    'science': ['algorithm', 'quantum', 'geometry', 'neural', 'vector', 'matrix'],
    'city': ['Paris', 'London', 'Tokyo'],
    'scientist': ['Einstein', 'Newton', 'Euler'],
}

print(f"  Within-category vs between-category shape similarity:")
within_sims = []
between_sims = []

for cat_name, cat_words in categories.items():
    cat_indices = [train_words.index(w) for w in cat_words if w in train_words]
    if len(cat_indices) < 2:
        continue

    # Within-category
    for i, j in combinations(cat_indices, 2):
        within_sims.append(cos_sim[i, j])

# Between categories (sample)
np.random.seed(42)
for _ in range(500):
    i, j = np.random.choice(n, 2, replace=False)
    between_sims.append(cos_sim[i, j])

print(f"    Within category:  mean={np.mean(within_sims):.4f} ± {np.std(within_sims):.4f}")
print(f"    Between category: mean={np.mean(between_sims):.4f} ± {np.std(between_sims):.4f}")
print(f"    Separation: {np.mean(within_sims) - np.mean(between_sims):.4f}")
print()

# Shape magnitude: which tokens have strongest shapes (most signal)?
print(f"  Shape magnitude (||profile||, top 20 modes):")
shape_mags = [(float(shape_norms[i, 0]), train_words[i]) for i in range(n)]
shape_mags.sort(key=lambda x: -x[0])
for mag, w in shape_mags[:10]:
    print(f"    {mag:8.4f}  {w}")
print(f"    ...")
for mag, w in shape_mags[-5:]:
    print(f"    {mag:8.4f}  {w}")

print()


# ================================================================
# TEST 4: FILTER BANDWIDTH
# How many modes carry signal per token?
# ================================================================
print("=" * 80)
print("  TEST 4: FILTER BANDWIDTH")
print("  How many modes carry signal per token?")
print("=" * 80)
print()

# For each token, compute effective dimensionality of its dressed profile
# D_eff = (sum |c_k|)^2 / sum c_k^2 (participation ratio)

bandwidths = []
d90s = []
for i, word in enumerate(train_words):
    coords = dressed_gate[i]
    abs_coords = np.abs(coords)

    # Participation ratio
    pr = (np.sum(abs_coords))**2 / (np.sum(coords**2) + 1e-10)
    bandwidths.append(pr)

    # D90: number of modes for 90% energy
    sorted_e = np.sort(coords**2)[::-1]
    cum = np.cumsum(sorted_e) / (np.sum(sorted_e) + 1e-10)
    d90 = int(np.searchsorted(cum, 0.9) + 1)
    d90s.append(d90)

print(f"  Participation ratio (effective # of active modes):")
print(f"    Mean: {np.mean(bandwidths):.1f}")
print(f"    Std:  {np.std(bandwidths):.1f}")
print(f"    Min:  {np.min(bandwidths):.1f} ({train_words[np.argmin(bandwidths)]})")
print(f"    Max:  {np.max(bandwidths):.1f} ({train_words[np.argmax(bandwidths)]})")
print()

print(f"  D90 (modes for 90% energy):")
print(f"    Mean: {np.mean(d90s):.1f}")
print(f"    Std:  {np.std(d90s):.1f}")
print(f"    Min:  {np.min(d90s)} ({train_words[np.argmin(d90s)]})")
print(f"    Max:  {np.max(d90s)} ({train_words[np.argmax(d90s)]})")
print()

# Does bandwidth correlate with token type?
# Function words vs content words
func_words = {'the', 'is', 'and', 'of', 'to', 'in'}
func_bw = [bandwidths[i] for i, w in enumerate(train_words) if w in func_words]
content_bw = [bandwidths[i] for i, w in enumerate(train_words) if w not in func_words]

print(f"  Function words bandwidth: {np.mean(func_bw):.1f} ± {np.std(func_bw):.1f}")
print(f"  Content words bandwidth:  {np.mean(content_bw):.1f} ± {np.std(content_bw):.1f}")
print()

# Bandwidth vs shape magnitude
from scipy.stats import spearmanr
corr_bw_mag, _ = spearmanr(bandwidths, [float(shape_norms[i, 0]) for i in range(n)])
print(f"  Correlation(bandwidth, shape_magnitude): {corr_bw_mag:.4f}")
print()


# ================================================================
# TEST 5: MINIMUM SHAPE FOR IDENTITY
# What's the smallest mode set that preserves token identity?
# And critically: is it the SAME modes for all tokens?
# ================================================================
print("=" * 80)
print("  TEST 5: MINIMUM SHAPE FOR IDENTITY")
print("  What's the smallest mode set for token ID?")
print("=" * 80)
print()

# For each token, find its "signature modes" — the modes where it
# deviates most from the population
# Signature strength = |coord - mean| / std

z_scores = np.abs(dressed_gate - mode_means) / (mode_stds + 1e-10)  # (n_tokens, 3584)

# For each token, which modes are its strongest signatures?
print(f"  Token signature modes (highest z-score):")
print(f"  {'Token':>10s}  {'Mode 1':>8s}  {'Mode 2':>8s}  {'Mode 3':>8s}  {'Overlap':>8s}")
print("  " + "-" * 40)

token_sig_modes = {}
for i, word in enumerate(train_words[:30]):
    top3 = np.argsort(z_scores[i])[::-1][:3]
    token_sig_modes[word] = set(top3[:5].tolist())

    # How much overlap with top-3 of other tokens?
    top3_set = set(top3.tolist())
    mean_overlap = np.mean([len(top3_set &
                           set(np.argsort(z_scores[j])[::-1][:3].tolist()))
                           for j in range(n) if j != i])

    print(f"  {word:>10s}  {top3[0]:8d}  {top3[1]:8d}  {top3[2]:8d}  {mean_overlap:8.2f}")

print()

# Universal vs token-specific modes
# Count how often each mode appears as a top-5 signature across all tokens
mode_signature_counts = np.zeros(n_modes)
for i in range(n):
    top5 = np.argsort(z_scores[i])[::-1][:5]
    mode_signature_counts[top5] += 1

# Most universal signature modes (appear for many tokens)
universal_modes = np.argsort(mode_signature_counts)[::-1][:20]
print(f"  Most universal signature modes (appear in most tokens' top-5):")
for k in universal_modes[:10]:
    print(f"    Mode {k}: used by {int(mode_signature_counts[k])}/{n} tokens ({mode_signature_counts[k]/n*100:.0f}%)")

print()

# Token-specific modes (appear for very few tokens)
specific_modes = [k for k in range(n_modes) if 0 < mode_signature_counts[k] <= 2]
print(f"  Token-specific modes (used by ≤2 tokens): {len(specific_modes)}")
print()

# GREEDY MODE SELECTION: find minimum mode set for 100% NN accuracy
# Start with most selective mode, greedily add modes that improve accuracy
print(f"  Greedy mode selection for token identity:")
selected = []
remaining = set(range(n_modes))
current_acc = 0.0

for step in range(50):
    best_mode = None
    best_acc = current_acc

    # Try each remaining mode
    candidates = list(np.argsort(mode_stds)[::-1][:100])  # only search top 100
    candidates = [c for c in candidates if c not in selected]

    for k in candidates:
        test_modes = selected + [k]
        coords = dressed_gate[:, test_modes]
        dists = pairwise_distances(coords)
        np.fill_diagonal(dists, np.inf)
        nn = np.argmin(dists, axis=1)
        acc = np.mean(nn == nn_truth)
        if acc > best_acc:
            best_acc = acc
            best_mode = k

    if best_mode is None:
        break

    selected.append(best_mode)
    current_acc = best_acc
    print(f"    Step {step+1}: +mode {best_mode} (S={S[best_mode]:.4f}), acc = {best_acc:.3f}")

    if best_acc >= 1.0:
        break

print()
print(f"  Minimum modes for 100% NN accuracy: {len(selected) if current_acc >= 1.0 else '>50'}")
print(f"  Selected modes: {selected}")
print(f"  Their singular values: {[f'{S[k]:.4f}' for k in selected]}")
print()

# Are these modes contiguous or scattered?
if selected:
    mode_indices = sorted(selected)
    gaps = [mode_indices[i+1] - mode_indices[i] for i in range(len(mode_indices)-1)]
    print(f"  Mode index range: {min(selected)}-{max(selected)}")
    print(f"  Mean gap between selected modes: {np.mean(gaps):.1f}" if gaps else "  Single mode")
    print()


# ================================================================
# TEST 6: XOR BINARY SHAPES
# The user's key question: can we find XOR-like patterns?
# Look for sign patterns in dressed coordinates
# ================================================================
print("=" * 80)
print("  TEST 6: XOR BINARY SHAPE PATTERNS")
print("  Do sign patterns (±) in mode coordinates act like XOR gates?")
print("=" * 80)
print()

# For each token, binarize its dressed profile: +1 if positive, -1 if negative
# This is the "XOR shape" — the sign pattern across modes

top_disc_modes = np.argsort(mode_stds)[::-1][:20]
binary_shapes = np.sign(dressed_gate[:, top_disc_modes])  # (n_tokens, 20)

# How many unique binary shapes are there?
unique_shapes = set()
for i in range(n):
    shape_tuple = tuple(binary_shapes[i].astype(int).tolist())
    unique_shapes.add(shape_tuple)

print(f"  Binary shapes from top 20 modes:")
print(f"    Total tokens: {n}")
print(f"    Unique shapes: {len(unique_shapes)}")
print(f"    Uniqueness ratio: {len(unique_shapes)/n:.3f}")
print()

# Hamming distance between binary shapes
def hamming(a, b):
    return np.sum(a != b)

# Do semantically related tokens share binary shapes?
print(f"  Hamming distances (20-bit binary shapes):")
semantic_pairs = [
    ("king", "queen"), ("man", "woman"), ("boy", "girl"),
    ("hot", "cold"), ("fast", "slow"), ("big", "small"),
    ("cat", "dog"), ("red", "blue"), ("true", "false"),
    ("love", "hate"), ("old", "young"), ("happy", "sad"),
]

pair_hammings = []
for w1, w2 in semantic_pairs:
    if w1 in train_words and w2 in train_words:
        i1, i2 = train_words.index(w1), train_words.index(w2)
        h = hamming(binary_shapes[i1], binary_shapes[i2])
        pair_hammings.append(h)
        print(f"    {w1:>8s} ↔ {w2:<8s}: Hamming = {h:2d}/20  {'(same shape!)' if h <= 2 else '(different)' if h >= 10 else ''}")

# Random pairs for comparison
rand_hammings = []
np.random.seed(42)
for _ in range(200):
    i, j = np.random.choice(n, 2, replace=False)
    rand_hammings.append(hamming(binary_shapes[i], binary_shapes[j]))

print()
print(f"  Semantic pair mean Hamming: {np.mean(pair_hammings):.1f}")
print(f"  Random pair mean Hamming:   {np.mean(rand_hammings):.1f}")
print(f"  Expected random (20 bits):  10.0")
print()

# Antonym test: do antonyms differ by exactly 1-2 bits? (minimal XOR flip)
antonym_pairs = [("hot", "cold"), ("fast", "slow"), ("big", "small"),
                 ("love", "hate"), ("light", "dark"), ("true", "false"),
                 ("happy", "sad"), ("strong", "weak"), ("old", "young")]

print(f"  Antonym bit flips:")
antonym_flips = []
for w1, w2 in antonym_pairs:
    if w1 in train_words and w2 in train_words:
        i1, i2 = train_words.index(w1), train_words.index(w2)
        flip_positions = np.where(binary_shapes[i1] != binary_shapes[i2])[0]
        antonym_flips.append(len(flip_positions))
        mode_ids = top_disc_modes[flip_positions]
        print(f"    {w1:>8s} → {w2:<8s}: {len(flip_positions)} flips at modes {mode_ids.tolist()}")

print()
if antonym_flips:
    print(f"  Mean antonym flips: {np.mean(antonym_flips):.1f}")
    print(f"  If antonyms = 1-bit XOR flip: {'YES' if np.mean(antonym_flips) <= 3 else 'NO'}")
print()

# Gender test: does male→female have a consistent bit flip pattern?
gender_pairs = [("king", "queen"), ("man", "woman"), ("boy", "girl")]
print(f"  Gender bit flips:")
gender_flip_modes = []
for w1, w2 in gender_pairs:
    if w1 in train_words and w2 in train_words:
        i1, i2 = train_words.index(w1), train_words.index(w2)
        flip_positions = np.where(binary_shapes[i1] != binary_shapes[i2])[0]
        flip_mode_ids = set(top_disc_modes[flip_positions].tolist())
        gender_flip_modes.append(flip_mode_ids)
        print(f"    {w1:>8s} → {w2:<8s}: flips at modes {sorted(flip_mode_ids)}")

if len(gender_flip_modes) >= 2:
    common_gender_modes = gender_flip_modes[0]
    for s in gender_flip_modes[1:]:
        common_gender_modes = common_gender_modes & s
    print(f"    Common gender flip modes: {sorted(common_gender_modes) if common_gender_modes else 'NONE'}")
print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  OVERALL SUMMARY")
print("=" * 80)
print()

test1_summary = {
    'most_selective_mode': int(top_selective[0]),
    'selectivity_concentration': {
        f'{int(t*100)}pct': int(np.searchsorted(cum_var, t) + 1)
        for t in [0.5, 0.8, 0.9, 0.95, 0.99]
    },
    'band_selectivities': [float(b) for b in band_selectivities],
}

test2_summary = {
    'best_single_mode': int(best_single_k),
    'best_single_acc': float(best_single_acc),
    'best_pair_modes': [int(m) for m in best_pair_modes],
    'best_pair_acc': float(best_pair_acc),
    'best_triple_modes': [int(m) for m in best_triple_modes] if best_triple_modes else None,
    'best_triple_acc': float(best_triple_acc),
    'mean_xor_gain': float(np.mean(xor_gain_values)),
    'max_xor_gain': float(np.max(xor_gain_values)),
    'xor_pairs_gt_005': int(sum(1 for g in xor_gain_values if g > 0.05)),
    'xor_pairs_gt_010': int(sum(1 for g in xor_gain_values if g > 0.10)),
}

test3_summary = {
    'within_category_cos': float(np.mean(within_sims)),
    'between_category_cos': float(np.mean(between_sims)),
    'separation': float(np.mean(within_sims) - np.mean(between_sims)),
}

test4_summary = {
    'mean_bandwidth': float(np.mean(bandwidths)),
    'std_bandwidth': float(np.std(bandwidths)),
    'mean_d90': float(np.mean(d90s)),
    'func_word_bandwidth': float(np.mean(func_bw)),
    'content_word_bandwidth': float(np.mean(content_bw)),
    'bandwidth_magnitude_corr': float(corr_bw_mag),
}

test5_summary = {
    'min_modes_for_100pct': len(selected) if current_acc >= 1.0 else None,
    'selected_modes': selected,
    'selected_svs': [float(S[k]) for k in selected],
    'final_acc': float(current_acc),
}

test6_summary = {
    'unique_binary_shapes': len(unique_shapes),
    'uniqueness_ratio': float(len(unique_shapes) / n),
    'semantic_pair_hamming': float(np.mean(pair_hammings)) if pair_hammings else None,
    'random_pair_hamming': float(np.mean(rand_hammings)),
    'mean_antonym_flips': float(np.mean(antonym_flips)) if antonym_flips else None,
    'common_gender_flip_modes': sorted(common_gender_modes) if gender_flip_modes and common_gender_modes else [],
}

results = {
    'test1_mode_selectivity': test1_summary,
    'test2_xor_shapes': test2_summary,
    'test3_shape_catalog': test3_summary,
    'test4_filter_bandwidth': test4_summary,
    'test5_minimum_shape': test5_summary,
    'test6_xor_binary': test6_summary,
}

print(f"  TEST 1: MODE SELECTIVITY")
print(f"    90% discrimination in {test1_summary['selectivity_concentration']['90pct']} modes")
print(f"    99% in {test1_summary['selectivity_concentration']['99pct']} modes")
print()
print(f"  TEST 2: XOR SHAPES")
print(f"    Best single: {best_single_acc:.3f}")
print(f"    Best pair:   {best_pair_acc:.3f} (XOR gain: {best_pair_acc - best_single_acc:+.3f})")
print(f"    Best triple: {best_triple_acc:.3f}")
print(f"    XOR pairs > 0.05 gain: {test2_summary['xor_pairs_gt_005']}")
print()
print(f"  TEST 3: SHAPE CATALOG")
print(f"    Within-cat cos: {test3_summary['within_category_cos']:.4f}")
print(f"    Between-cat cos: {test3_summary['between_category_cos']:.4f}")
print(f"    Separation: {test3_summary['separation']:.4f}")
print()
print(f"  TEST 4: FILTER BANDWIDTH")
print(f"    Mean bandwidth: {np.mean(bandwidths):.1f} modes")
print(f"    D90: {np.mean(d90s):.1f} modes")
print(f"    Function vs content: {np.mean(func_bw):.0f} vs {np.mean(content_bw):.0f}")
print()
print(f"  TEST 5: MINIMUM SHAPE")
print(f"    Modes for 100% NN: {len(selected) if current_acc >= 1.0 else f'>{len(selected)} (best={current_acc:.3f})'}")
print()
print(f"  TEST 6: XOR BINARY")
print(f"    Unique shapes: {len(unique_shapes)}/{n}")
print(f"    Semantic Hamming: {np.mean(pair_hammings):.1f} vs random {np.mean(rand_hammings):.1f}")
print(f"    Mean antonym flips: {np.mean(antonym_flips):.1f}" if antonym_flips else "    No antonym data")
print(f"    Common gender flip modes: {sorted(common_gender_modes) if common_gender_modes else 'NONE'}")
print()


# ================================================================
# Save results
# ================================================================
del model
torch.cuda.empty_cache()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8t_shape_filter.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"  Results saved to {results_path}")
