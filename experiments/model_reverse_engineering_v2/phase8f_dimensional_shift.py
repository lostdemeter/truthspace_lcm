#!/usr/bin/env python3
"""
Phase 8f: Dimensional Shift — Intrinsic Dimensionality of the Residual
========================================================================

Finding 66 showed the gate dimension's content is a 0.17% perturbation
on a 99.83% scaffold, with L/R channels carrying the same content (echo).

The DSS (Dimensional Shift Solver) principle: embed in the dimension where
structure becomes maximally visible. Doc 197/198: the standing wave IS the
"perspective-invariant analog", and the residual IS the "delta".

Questions:
  1. What is the intrinsic rank of the residual? (How many SVD dimensions
     preserve token identity?)
  2. Does the echo (L/R mirror) emerge naturally at the right rank?
     (If so, it's structural overlap, not designed redundancy)
  3. Does the critical rank relate to φ-structural constants?
     (φ, 4/π, log(3)/log(2), etc.)
  4. Can we INTERVENE with the low-rank residual and preserve token output?
     (The real test — does dimensional shift preserve content?)

Requires: Qwen2-7B on GPU
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
SIERPINSKI_D = np.log(3) / np.log(2)  # 1.585

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
print("  PHASE 8f: DIMENSIONAL SHIFT — Intrinsic Dimensionality")
print("  DSS + Perspective-Invariant Analog + Exploiting Structure")
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
print()


# ================================================================
# DECOMPOSITION
# ================================================================
standing_wave = all_raw.mean(axis=0)  # [N_LAYERS, HIDDEN_DIM]
residuals = all_raw - standing_wave[np.newaxis, :, :]  # [N_TOKENS, N_LAYERS, HIDDEN_DIM]

per_channel_mode = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.int8)
for layer in range(N_LAYERS):
    for ch in range(HIDDEN_DIM):
        counts = np.bincount(all_codes[:, layer, ch].astype(int), minlength=4)
        per_channel_mode[layer, ch] = counts.argmax()


# ================================================================
# TEST 1: INTRINSIC RANK — How many dimensions preserve token identity?
# ================================================================
print("─" * 80)
print("  TEST 1: INTRINSIC RANK OF RESIDUAL")
print("  At what rank k does the residual preserve token discrimination?")
print("  This is the 'natural dimension' of the content.")
print("─" * 80)
print()

# Per-layer SVD: decompose [N_TOKENS × HIDDEN_DIM] residual matrix
# Max possible rank = N_TOKENS (since N_TOKENS << HIDDEN_DIM)
# Test reconstruction at rank k = 1, 2, 3, ..., N_TOKENS

COMB_START = 6
COMB_END = 23

# Compute per-layer SVD for COMB layers
print(f"  Computing per-layer SVD for COMB layers ({COMB_START}-{COMB_END-1})...")

svd_per_layer = {}
for layer in range(COMB_START, COMB_END):
    res = residuals[:, layer, :]  # [N_TOKENS, HIDDEN_DIM]
    U, S, Vt = np.linalg.svd(res, full_matrices=False)
    svd_per_layer[layer] = (U, S, Vt)

# Test 1a: Per-layer rank analysis — variance explained curve
print()
print(f"  Variance explained by rank k (COMB layers, averaged):")
test_ranks = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, N_TOKENS]
print(f"  {'Rank k':>8s}  {'Var explained':>14s}  {'Cumulative':>12s}  {'φ-relation':>14s}")
print("  " + "-" * 55)

rank_results = []
for k in test_ranks:
    if k > N_TOKENS:
        continue
    var_explained_list = []
    for layer in range(COMB_START, COMB_END):
        U, S, Vt = svd_per_layer[layer]
        total_var = (S**2).sum()
        topk_var = (S[:k]**2).sum()
        var_explained_list.append(topk_var / total_var if total_var > 0 else 0)

    mean_var = np.mean(var_explained_list)

    # Check φ-relations for the rank number itself
    phi_note = ""
    for name, val in [("φ", PHI), ("φ²", PHI**2), ("4φ⁴/N", 4*PHI**4/N_TOKENS),
                       ("D_S", SIERPINSKI_D), ("4/π", 4/np.pi),
                       ("N/φ", N_TOKENS/PHI), ("N/φ²", N_TOKENS/PHI**2)]:
        if abs(k - val) / max(val, 1) < 0.1:  # within 10%
            phi_note = f"≈ {name}={val:.2f}"
            break

    print(f"  {k:8d}  {mean_var:14.4f}  {mean_var*100:11.1f}%  {phi_note:>14s}")
    rank_results.append({
        'rank': k,
        'mean_var_explained': float(mean_var),
        'phi_note': phi_note,
    })

print()

# Test 1b: Token discrimination at each rank
# Reconstruct residuals at rank k, then check if tokens are distinguishable
print(f"  Token discrimination at each rank (COMB, pairwise cosine sim):")
print(f"  {'Rank k':>8s}  {'Mean pairwise':>14s}  {'Min pairwise':>13s}  {'Discriminates?':>15s}")
print("  " + "-" * 58)

discrimination_results = []
for k in test_ranks:
    if k > N_TOKENS:
        continue
    # Reconstruct residuals at rank k for all COMB layers
    reconstructed = np.zeros((N_TOKENS, COMB_END - COMB_START, HIDDEN_DIM))
    for li, layer in enumerate(range(COMB_START, COMB_END)):
        U, S, Vt = svd_per_layer[layer]
        # Rank-k reconstruction: U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
        reconstructed[:, li, :] = (U[:, :k] * S[:k]) @ Vt[:k, :]

    # Flatten and compute pairwise cosine similarity
    flat = reconstructed.reshape(N_TOKENS, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = flat / norms
    sim_matrix = normalized @ normalized.T
    mask_offdiag = ~np.eye(N_TOKENS, dtype=bool)
    mean_sim = sim_matrix[mask_offdiag].mean()
    min_sim = sim_matrix[mask_offdiag].min()

    # Discriminates if mean pairwise sim is low (tokens look different)
    discriminates = "YES" if mean_sim < 0.3 else ("PARTIAL" if mean_sim < 0.6 else "NO")

    print(f"  {k:8d}  {mean_sim:14.4f}  {min_sim:13.4f}  {discriminates:>15s}")

    discrimination_results.append({
        'rank': k,
        'mean_pairwise_sim': float(mean_sim),
        'min_pairwise_sim': float(min_sim),
        'discriminates': discriminates,
    })

print()


# ================================================================
# TEST 2: ECHO EMERGENCE — Does L/R mirror appear at each rank?
# ================================================================
print("─" * 80)
print("  TEST 2: ECHO EMERGENCE")
print("  Does the L/R mirror (echo) appear naturally at each rank?")
print("  If yes at low rank → echo IS the structure, not an add-on")
print("─" * 80)
print()

# For each rank k, reconstruct L-only and R-only residuals,
# then measure how similar they are (high correlation = echo present)

print(f"  L/R echo correlation at each rank (COMB layers, averaged):")
print(f"  {'Rank k':>8s}  {'L-R norm corr':>14s}  {'L discrim':>11s}  {'R discrim':>11s}  {'Echo?':>8s}")
print("  " + "-" * 60)

echo_results = []
for k in test_ranks:
    if k > N_TOKENS:
        continue

    lr_corrs = []
    l_sims = []
    r_sims = []

    for li, layer in enumerate(range(COMB_START, COMB_END)):
        U, S, Vt = svd_per_layer[layer]
        recon = (U[:, :k] * S[:k]) @ Vt[:k, :]  # [N_TOKENS, HIDDEN_DIM]

        # Split into L and R channels
        l_mask = np.array([per_channel_mode[layer, ch] in CHANNEL_L for ch in range(HIDDEN_DIM)])
        r_mask = ~l_mask

        res_L = recon[:, l_mask]  # [N_TOKENS, n_L]
        res_R = recon[:, r_mask]  # [N_TOKENS, n_R]

        # L/R norm correlation across tokens
        l_norms = np.linalg.norm(res_L, axis=1)
        r_norms = np.linalg.norm(res_R, axis=1)

        if l_norms.std() > 1e-10 and r_norms.std() > 1e-10:
            corr = np.corrcoef(l_norms, r_norms)[0, 1]
            lr_corrs.append(corr)

    mean_lr_corr = np.mean(lr_corrs) if lr_corrs else 0

    # L-only and R-only discrimination at this rank
    # Pool across layers
    l_flat = np.zeros((N_TOKENS, 0))
    r_flat = np.zeros((N_TOKENS, 0))
    for li, layer in enumerate(range(COMB_START, COMB_END)):
        U, S, Vt = svd_per_layer[layer]
        recon = (U[:, :k] * S[:k]) @ Vt[:k, :]
        l_mask = np.array([per_channel_mode[layer, ch] in CHANNEL_L for ch in range(HIDDEN_DIM)])
        r_mask = ~l_mask
        l_flat = np.hstack([l_flat, recon[:, l_mask]])
        r_flat = np.hstack([r_flat, recon[:, r_mask]])

    # L discrimination
    norms = np.linalg.norm(l_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    norm_l = l_flat / norms
    sim_l = (norm_l @ norm_l.T)[~np.eye(N_TOKENS, dtype=bool)].mean()

    # R discrimination
    norms = np.linalg.norm(r_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    norm_r = r_flat / norms
    sim_r = (norm_r @ norm_r.T)[~np.eye(N_TOKENS, dtype=bool)].mean()

    echo = "YES" if mean_lr_corr > 0.8 else ("PARTIAL" if mean_lr_corr > 0.5 else "NO")

    print(f"  {k:8d}  {mean_lr_corr:14.4f}  {sim_l:11.4f}  {sim_r:11.4f}  {echo:>8s}")

    echo_results.append({
        'rank': k,
        'mean_lr_corr': float(mean_lr_corr),
        'l_pairwise_sim': float(sim_l),
        'r_pairwise_sim': float(sim_r),
        'echo': echo,
    })

print()


# ================================================================
# TEST 3: SINGULAR VALUE STRUCTURE — What do the SVs themselves tell us?
# ================================================================
print("─" * 80)
print("  TEST 3: SINGULAR VALUE STRUCTURE")
print("  Do the singular values relate to φ-structural constants?")
print("  DSS principle: the natural dimension reveals maximum structure")
print("─" * 80)
print()

# Collect all COMB layer singular values
all_sv_profiles = []
for layer in range(COMB_START, COMB_END):
    U, S, Vt = svd_per_layer[layer]
    all_sv_profiles.append(S)

all_svs = np.array(all_sv_profiles)  # [17, N_TOKENS]
mean_sv_profile = all_svs.mean(axis=0)  # [N_TOKENS] — averaged SV curve

print(f"  Mean SV profile across COMB layers (N_TOKENS = {N_TOKENS} max rank):")
print(f"  {'Index':>7s}  {'Mean SV':>10s}  {'Cumul var%':>11s}  {'Ratio S[i]/S[i+1]':>18s}  {'φ-match':>12s}")
print("  " + "-" * 65)

cumul_var = 0
total_var_mean = (mean_sv_profile**2).sum()

sv_structure = []
for i in range(min(N_TOKENS, 30)):
    sv_val = mean_sv_profile[i]
    cumul_var += sv_val**2
    cumul_pct = cumul_var / total_var_mean * 100

    if i < len(mean_sv_profile) - 1 and mean_sv_profile[i+1] > 1e-10:
        ratio = mean_sv_profile[i] / mean_sv_profile[i+1]
    else:
        ratio = float('inf')

    # Check ratio against φ-constants
    phi_match = ""
    for name, val in [("φ", PHI), ("√φ", np.sqrt(PHI)), ("4/π", 4/np.pi),
                       ("φ²", PHI**2), ("1/φ", 1/PHI), ("D_S", SIERPINSKI_D),
                       ("2", 2.0), ("e/2", np.e/2)]:
        if abs(ratio - val) / val < 0.05:  # within 5%
            phi_match = f"≈ {name}"
            break

    print(f"  {i:7d}  {sv_val:10.4f}  {cumul_pct:10.1f}%  {ratio:18.4f}  {phi_match:>12s}")

    sv_structure.append({
        'index': i,
        'mean_sv': float(sv_val),
        'cumulative_var_pct': float(cumul_pct),
        'ratio_to_next': float(ratio) if ratio != float('inf') else None,
        'phi_match': phi_match,
    })

print()

# Find the "elbow" — where does cumulative variance reach key thresholds?
print(f"  Cumulative variance thresholds:")
for threshold in [50, 75, 90, 95, 99]:
    cumul = 0
    for i in range(len(mean_sv_profile)):
        cumul += mean_sv_profile[i]**2
        if cumul / total_var_mean * 100 >= threshold:
            print(f"    {threshold}% variance at rank k={i+1}")

            # Check if k+1 relates to φ-constants
            k_val = i + 1
            for name, val in [("φ", PHI), ("φ²", PHI**2), ("4", 4),
                               ("4/π", 4/np.pi), ("D_S", SIERPINSKI_D),
                               ("N/φ", N_TOKENS/PHI), ("N/φ²", N_TOKENS/PHI**2),
                               ("φ³", PHI**3), ("4φ", 4*PHI)]:
                if abs(k_val - val) / max(val, 1) < 0.15:
                    print(f"      → k={k_val} ≈ {name} = {val:.2f}")
            break

print()


# ================================================================
# TEST 4: INTERVENTION — Does low-rank residual preserve token output?
# ================================================================
print("─" * 80)
print("  TEST 4: INTERVENTION — Can we reconstruct gates from")
print("  standing_wave + low_rank_residual and preserve token identity?")
print("  The ultimate test: does dimensional shift WORK?")
print("─" * 80)
print()

# Use the already-loaded model for intervention
print("  Getting baseline outputs...")
baseline_logits = {}
for word in all_words[:10]:  # Test on 10 tokens
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        out = model(input_ids)
        baseline_logits[word] = out.logits[0, -1, :].cpu().float().numpy()

test_tokens = list(baseline_logits.keys())
print(f"  Baseline captured for {len(test_tokens)} tokens")

# For each rank k, intervene with standing_wave + rank-k residual
intervention_ranks = [1, 2, 3, 5, 10, 20, N_TOKENS]

print()
print(f"  Intervention results (COMB layers {COMB_START}-{COMB_END-1}):")
print(f"  {'Rank k':>8s}  {'Cos sim':>9s}  {'Top-1 agree':>12s}  "
      f"{'Top-5 overlap':>14s}  {'KL div':>9s}  {'Verdict':>10s}")
print("  " + "-" * 70)

intervention_results = []
for k in intervention_ranks:
    if k > N_TOKENS:
        continue

    cos_sims = []
    top1_matches = 0
    top5_overlaps = []
    kl_divs = []

    for tok_word in test_tokens:
        tok_idx = all_words.index(tok_word)
        ids = tokenizer.encode(tok_word, add_special_tokens=False)
        token_id = ids[0]

        # Precompute intervened gate values for this token
        # gate = standing_wave + rank_k_residual
        intervened_gates = {}
        for layer in range(COMB_START, COMB_END):
            U, S, Vt = svd_per_layer[layer]
            # This token's rank-k residual
            rank_k_resid = (U[tok_idx, :k] * S[:k]) @ Vt[:k, :]
            intervened_gates[layer] = standing_wave[layer] + rank_k_resid

        # Run model with hooks that replace gate activations in COMB layers
        hooks = []

        def make_replace_hook(replacement):
            def hook_fn(module, input, output):
                rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
                return rep.reshape(output.shape)
            return hook_fn

        for layer in range(COMB_START, COMB_END):
            h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
                make_replace_hook(intervened_gates[layer])
            )
            hooks.append(h)

        with torch.no_grad():
            input_ids = torch.tensor([[token_id]], device="cuda")
            out = model(input_ids)
            int_logits = out.logits[0, -1, :].cpu().float().numpy()

        for h in hooks:
            h.remove()

        # Compare to baseline
        base_l = baseline_logits[tok_word]

        # Cosine similarity
        cos = np.dot(base_l, int_logits) / (np.linalg.norm(base_l) * np.linalg.norm(int_logits))
        cos_sims.append(cos)

        # Top-1 agreement
        if np.argmax(base_l) == np.argmax(int_logits):
            top1_matches += 1

        # Top-5 overlap
        base_top5 = set(np.argsort(base_l)[-5:])
        int_top5 = set(np.argsort(int_logits)[-5:])
        top5_overlaps.append(len(base_top5 & int_top5) / 5)

        # KL divergence (approximate, on top-1000 for stability)
        top_k = 1000
        base_probs = np.exp(base_l - base_l.max())
        base_probs = base_probs / base_probs.sum()
        int_probs = np.exp(int_logits - int_logits.max())
        int_probs = int_probs / int_probs.sum()
        # Clip for numerical stability
        base_probs = np.clip(base_probs, 1e-10, None)
        int_probs = np.clip(int_probs, 1e-10, None)
        kl = np.sum(base_probs * np.log(base_probs / int_probs))
        kl_divs.append(kl)

    mean_cos = np.mean(cos_sims)
    top1_rate = top1_matches / len(test_tokens)
    mean_top5 = np.mean(top5_overlaps)
    mean_kl = np.mean(kl_divs)

    verdict = "PERFECT" if top1_rate == 1.0 else (
        "GOOD" if top1_rate >= 0.8 else (
        "PARTIAL" if top1_rate >= 0.3 else "FAIL"))

    print(f"  {k:8d}  {mean_cos:9.4f}  {top1_rate:11.0%}  "
          f"{mean_top5:13.0%}  {mean_kl:9.3f}  {verdict:>10s}")

    intervention_results.append({
        'rank': k,
        'mean_cosine_sim': float(mean_cos),
        'top1_agreement': float(top1_rate),
        'top5_overlap': float(mean_top5),
        'mean_kl_divergence': float(mean_kl),
        'verdict': verdict,
        'per_token_cos': [float(c) for c in cos_sims],
    })

print()

# Show per-token detail for key ranks
for result in intervention_results:
    k = result['rank']
    if k in [1, 5, N_TOKENS]:
        print(f"  Per-token detail at rank {k}:")
        for i, word in enumerate(test_tokens):
            ids = tokenizer.encode(word, add_special_tokens=False)
            token_id = ids[0]

            base_top1_id = np.argmax(baseline_logits[word])
            base_top1 = tokenizer.decode([base_top1_id]).strip()

            # Re-run for this specific token to get intervened top-1
            tok_idx = all_words.index(word)
            intervened_gates = {}
            for layer in range(COMB_START, COMB_END):
                U, S, Vt = svd_per_layer[layer]
                rank_k_resid = (U[tok_idx, :k] * S[:k]) @ Vt[:k, :]
                intervened_gates[layer] = standing_wave[layer] + rank_k_resid

            hooks = []
            for layer in range(COMB_START, COMB_END):
                h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
                    make_replace_hook(intervened_gates[layer])
                )
                hooks.append(h)

            with torch.no_grad():
                input_ids_t = torch.tensor([[token_id]], device="cuda")
                out = model(input_ids_t)
                int_logits = out.logits[0, -1, :].cpu().float().numpy()

            for h in hooks:
                h.remove()

            int_top1_id = np.argmax(int_logits)
            int_top1 = tokenizer.decode([int_top1_id]).strip()
            match = "✓" if base_top1_id == int_top1_id else "✗"

            print(f"    {word:>12s}: base='{base_top1}' → rank-{k}='{int_top1}' {match}")
        print()


# ================================================================
# TEST 5: DIMENSIONAL STRUCTURE METRIC (DSS)
# ================================================================
print("─" * 80)
print("  TEST 5: DIMENSIONAL STRUCTURE METRIC (DSS)")
print("  Does the residual have maximum structure at a specific")
print("  dimensionality? DSS: S(D) = σ(d_ij) / μ(d_ij)")
print("─" * 80)
print()

# Embed tokens using rank-k residuals and compute structure metric S
# S = std(pairwise distances) / mean(pairwise distances)
# Higher S = more structure visible at that rank

print(f"  Structure metric S(k) at each rank (COMB layers):")
print(f"  {'Rank k':>8s}  {'S(k)':>10s}  {'Mean dist':>10s}  {'Std dist':>10s}  {'Max/min ratio':>14s}")
print("  " + "-" * 60)

structure_results = []
for k in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, N_TOKENS]:
    if k > N_TOKENS:
        continue

    # Project tokens into rank-k subspace (use U[:,:k] * S[:k] as coordinates)
    # Pool across COMB layers
    coords = np.zeros((N_TOKENS, k * (COMB_END - COMB_START)))
    for li, layer in enumerate(range(COMB_START, COMB_END)):
        U, S, Vt = svd_per_layer[layer]
        coords[:, li*k:(li+1)*k] = U[:, :k] * S[:k]

    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(coords, metric='euclidean')

    mean_d = dists.mean()
    std_d = dists.std()
    S_metric = std_d / mean_d if mean_d > 0 else 0
    max_min = dists.max() / dists.min() if dists.min() > 0 else float('inf')

    print(f"  {k:8d}  {S_metric:10.4f}  {mean_d:10.4f}  {std_d:10.4f}  {max_min:14.4f}")

    structure_results.append({
        'rank': k,
        'S_metric': float(S_metric),
        'mean_distance': float(mean_d),
        'std_distance': float(std_d),
        'max_min_ratio': float(max_min) if max_min != float('inf') else None,
    })

# Find optimal k (maximum S)
best_k = max(structure_results, key=lambda x: x['S_metric'])
print()
print(f"  Maximum structure at rank k = {best_k['rank']} (S = {best_k['S_metric']:.4f})")

# Check if best rank relates to φ-constants
for name, val in [("φ", PHI), ("φ²", PHI**2), ("4/π", 4/np.pi),
                   ("D_S", SIERPINSKI_D), ("φ³", PHI**3), ("4φ", 4*PHI),
                   ("2φ", 2*PHI), ("N/φ", N_TOKENS/PHI)]:
    if abs(best_k['rank'] - val) / max(val, 1) < 0.15:
        print(f"  → k={best_k['rank']} ≈ {name} = {val:.2f} ({abs(best_k['rank'] - val)/val*100:.1f}% error)")

print()

# Free model after all tests
del model
torch.cuda.empty_cache()

# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: DIMENSIONAL SHIFT ANALYSIS")
print("=" * 80)
print()

# Find minimum rank for discrimination
min_discrim_rank = N_TOKENS
for d in discrimination_results:
    if d['discriminates'] == 'YES':
        min_discrim_rank = d['rank']
        break

# Find minimum rank for intervention success
min_intervention_rank = N_TOKENS
for ir in intervention_results:
    if ir['top1_agreement'] >= 0.8:
        min_intervention_rank = ir['rank']
        break

print(f"  INTRINSIC RANK:")
print(f"    Min rank for token discrimination: k = {min_discrim_rank}")
print(f"    Min rank for intervention success (≥80% top-1): k = {min_intervention_rank}")
print(f"    Structure metric peak at: k = {best_k['rank']}")
print()

print(f"  ECHO EMERGENCE:")
for er in echo_results:
    if er['rank'] <= 5:
        print(f"    Rank {er['rank']}: L/R corr = {er['mean_lr_corr']:.4f} — Echo {'present' if er['echo'] == 'YES' else 'absent'}")
print()

print(f"  INTERVENTION RESULTS:")
for ir in intervention_results:
    print(f"    Rank {ir['rank']:3d}: cos={ir['mean_cosine_sim']:.4f}, "
          f"top1={ir['top1_agreement']:.0%}, top5={ir['top5_overlap']:.0%}, "
          f"KL={ir['mean_kl_divergence']:.3f} → {ir['verdict']}")
print()

# The key question: does the echo appear at the discrimination rank?
echo_at_discrim = None
for er in echo_results:
    if er['rank'] == min_discrim_rank:
        echo_at_discrim = er
        break

if echo_at_discrim:
    print(f"  ECHO AT DISCRIMINATION RANK (k={min_discrim_rank}):")
    print(f"    L/R correlation: {echo_at_discrim['mean_lr_corr']:.4f}")
    print(f"    L discrimination: {echo_at_discrim['l_pairwise_sim']:.4f}")
    print(f"    R discrimination: {echo_at_discrim['r_pairwise_sim']:.4f}")
    if echo_at_discrim['echo'] == 'YES':
        print(f"    → The echo EMERGES NATURALLY at the discrimination rank")
        print(f"      This confirms the echo is structural overlap, not error correction")
    else:
        print(f"    → The echo does NOT appear at this rank")
        print(f"      Echo requires more dimensions → it's a higher-order structure")
print()

# Compression ratio
if min_intervention_rank < N_TOKENS:
    compression = N_TOKENS / min_intervention_rank
    dim_reduction = HIDDEN_DIM / min_intervention_rank
    print(f"  DIMENSIONAL SHIFT ACHIEVED:")
    print(f"    Original: {N_TOKENS} tokens × {HIDDEN_DIM} channels per layer")
    print(f"    Shifted:  {min_intervention_rank} dimensions per layer")
    print(f"    Token rank compression: {compression:.1f}×")
    print(f"    Channel dim reduction:  {dim_reduction:.0f}× (from {HIDDEN_DIM} to {min_intervention_rank})")
    print(f"    Content representation: {min_intervention_rank} numbers instead of {HIDDEN_DIM}")

print()

# Save results
results = {
    'rank_analysis': rank_results,
    'discrimination': discrimination_results,
    'echo_emergence': echo_results,
    'sv_structure': sv_structure,
    'intervention': intervention_results,
    'structure_metric': structure_results,
    'summary': {
        'min_discrimination_rank': min_discrim_rank,
        'min_intervention_rank': min_intervention_rank,
        'structure_peak_rank': best_k['rank'],
        'n_tokens': N_TOKENS,
        'n_layers': N_LAYERS,
        'hidden_dim': HIDDEN_DIM,
    }
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8f_dimensional_shift.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
