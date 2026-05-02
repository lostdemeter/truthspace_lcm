#!/usr/bin/env python3
"""
Phase 8m: The Marble Geometry
==============================

The user's marble jar analogy:
- Each marble = a layer's representation. Centroid = scaffold (universal).
- Path drawn on marble surfaces = content routing (the 7 directions).
- Earthquake (new prompt) = centroids stay, marbles rotate.
- The AI reconstructs a path through rotated marbles within a cone of possibilities.

This explains:
- Scaffold generalizes (centroids don't move)
- 7 directions don't generalize (path orientation is prompt-specific)
- D*=7 might be consistent (path always needs ~7 DOF, just rotated)

Tests:
1. MARBLE SIZE: Is the residual norm (radius) consistent across prompts per layer?
   If yes → marbles are same size, only rotation changes.
2. MARBLE D*: Does each prompt independently need ~7 dims? Or does D* vary?
   If consistent → the "path shape" is universal, only orientation varies.
3. SINGULAR VALUE SPECTRUM: Do per-prompt SVD spectra match across prompts?
   If yes → the path's internal structure is universal (same curvature).
4. PATH CURVATURE: Angles between consecutive positions along the path.
   If consistent → the path bends the same way regardless of marble rotation.
5. ROTATION CHARACTERIZATION: What maps one prompt's subspace to another's?
   Is it a simple rotation? Related to attention?

Requires: Qwen2-7B on GPU
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
print("  PHASE 8m: THE MARBLE GEOMETRY")
print("  What's universal about the path? What's prompt-specific?")
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
HIDDEN_STATE_DIM = model.config.hidden_size


# ================================================================
# STEP 1: Build crystal from single tokens
# ================================================================
print("-" * 80)
print("  STEP 1: Build crystal from single tokens")
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
N_SINGLE = len(train_words)
all_gates_single = np.stack([single_gates[w] for w in train_words])
all_hs_single = np.stack([single_hs[w] for w in train_words])

scaffold_single = all_gates_single.mean(axis=0)
h_mean_single = all_hs_single.mean(axis=0)

print(f"  Crystal: {N_SINGLE} tokens")
print()


# ================================================================
# STEP 2: Capture diverse prompts
# ================================================================
print("-" * 80)
print("  STEP 2: Capture diverse prompts")
print("-" * 80)

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
    "The chemical symbol for gold is",
    "One plus one equals",
    "The color of the sky is",
    "To solve a quadratic equation you can use the",
    "Albert Einstein developed the theory of",
    "The currency used in Japan is the",
    "The boiling point of water is",
    "The largest ocean on Earth is the",
    "In binary, the number ten is",
    "The first president of the United States was",
    "Photosynthesis converts sunlight into",
    "The atomic number of carbon is",
    "The fastest land animal is the",
    "The Pythagorean theorem states that",
    "DNA stands for deoxyribonucleic",
    "The tallest mountain in the world is",
    "The square root of 144 is",
    "Shakespeare wrote the play Romeo and",
    "The human body has a total of",
    "The chemical formula for water is",
]

prompt_data = []
for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tok = input_ids.shape[1]

    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out.logits[0, -1, :].cpu().float().numpy()

    gate_storage = {}
    hs_storage = {}
    hooks = []

    def make_gate_hook2(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hs_hook2(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(N_LAYERS):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gate_hook2(gate_storage, layer)
        )
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook2(hs_storage, layer)
        )
        hooks.extend([h1, h2])

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    prompt_data.append({
        'prompt': prompt,
        'n_tokens': n_tok,
        'base_logits': base_logits,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
    })

N_PROMPTS = len(prompt_data)
print(f"  Captured {N_PROMPTS} prompts")
print()


# ================================================================
# Precompute stereo scaffolds
# ================================================================
corrected_scaffolds = {}
for pi, pd in enumerate(prompt_data):
    corrected_scaffolds[pi] = {}
    for layer in range(COMB_START, COMB_END):
        hs_all_pos = pd['hs'][layer]
        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        corrected_scaffolds[pi][layer] = scaffold_single[layer] + W_gate @ h_shift


# ================================================================
# TEST 1: MARBLE SIZE -- residual norms across prompts
# ================================================================
print("=" * 80)
print("  TEST 1: MARBLE SIZE")
print("  Is the residual norm (marble radius) consistent across prompts?")
print("=" * 80)
print()

# For each prompt, compute |gate_last - scaffold| at each layer
for layer in [6, 10, 14, 18, 22]:
    norms = []
    for pi, pd in enumerate(prompt_data):
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        last_pos = gates.shape[0] - 1
        residual = gates[last_pos] - scaffold
        norms.append(np.linalg.norm(residual))

    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    cv = std_norm / mean_norm  # coefficient of variation

    print(f"  Layer {layer:2d}: |residual| = {mean_norm:.1f} +/- {std_norm:.1f}  "
          f"(CV={cv:.3f})  range=[{min(norms):.1f}, {max(norms):.1f}]")

print()
print("  CV < 0.1 = very consistent (same marble size)")
print("  CV < 0.3 = moderately consistent")
print("  CV > 0.5 = highly variable")
print()


# ================================================================
# TEST 2: PER-PROMPT D* -- Does each prompt need ~7 dims independently?
# ================================================================
print("=" * 80)
print("  TEST 2: PER-PROMPT D*")
print("  Does each prompt need ~7 dims for 90%+ variance capture?")
print("=" * 80)
print()

# For each prompt, compute SVD of its per-position residuals at each layer
# Find how many dims capture 90% of the per-prompt residual variance

layer = 14  # Representative COMB layer
print(f"  Layer {layer} analysis:")
print()

prompt_spectra = []    # Store per-prompt singular value spectra
prompt_d_stars = []    # D* for 90% variance
prompt_d_stars_95 = [] # D* for 95% variance

for pi, pd in enumerate(prompt_data):
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = gates.shape[0]

    # Per-position residuals
    residuals = gates - scaffold[np.newaxis, :]  # [n_pos, HIDDEN_DIM]

    # SVD
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)

    # Variance captured
    total_var = np.sum(S ** 2)
    cum_var = np.cumsum(S ** 2) / total_var

    # D* at 90% and 95%
    d90 = int(np.searchsorted(cum_var, 0.90) + 1)
    d95 = int(np.searchsorted(cum_var, 0.95) + 1)
    prompt_d_stars.append(d90)
    prompt_d_stars_95.append(d95)

    # Normalize spectrum
    if S[0] > 0:
        normed_spectrum = S[:min(15, len(S))] / S[0]
        prompt_spectra.append(normed_spectrum)

    print(f"  {pd['prompt'][:40]:>40s}  n_pos={n_pos:2d}  "
          f"D*(90%)={d90:2d}  D*(95%)={d95:2d}  "
          f"S0/S1={S[0]/S[1]:.3f}  S1/S2={S[1]/S[2]:.3f}")

print()
print(f"  D*(90%): mean={np.mean(prompt_d_stars):.1f} +/- {np.std(prompt_d_stars):.1f}  "
      f"range=[{min(prompt_d_stars)}, {max(prompt_d_stars)}]")
print(f"  D*(95%): mean={np.mean(prompt_d_stars_95):.1f} +/- {np.std(prompt_d_stars_95):.1f}  "
      f"range=[{min(prompt_d_stars_95)}, {max(prompt_d_stars_95)}]")
print()


# ================================================================
# TEST 3: SINGULAR VALUE SPECTRUM UNIVERSALITY
# ================================================================
print("=" * 80)
print("  TEST 3: SINGULAR VALUE SPECTRUM UNIVERSALITY")
print("  Do normalized spectra match across prompts? (Same path shape?)")
print("=" * 80)
print()

# Compare normalized singular value spectra across prompts
# If the marble analogy holds: spectra should be similar (same path curvature)
# even though directions differ (different rotation)

# Stack all normalized spectra and compute mean + std
max_k = min(len(s) for s in prompt_spectra)
spectrum_matrix = np.stack([s[:max_k] for s in prompt_spectra])  # [N_PROMPTS, max_k]

mean_spectrum = spectrum_matrix.mean(axis=0)
std_spectrum = spectrum_matrix.std(axis=0)
cv_spectrum = std_spectrum / (mean_spectrum + 1e-10)

print(f"  Layer {layer} normalized spectrum (S_k / S_0):")
print(f"  {'k':>5s}  {'Mean':>8s}  {'Std':>8s}  {'CV':>8s}  {'Shape':>30s}")
print("  " + "-" * 65)

for k in range(min(10, max_k)):
    bar = "#" * int(mean_spectrum[k] * 30)
    print(f"  {k+1:5d}  {mean_spectrum[k]:8.4f}  {std_spectrum[k]:8.4f}  "
          f"{cv_spectrum[k]:8.4f}  {bar}")

print()
print("  Low CV = spectrum is universal (same marble curvature)")
print("  High CV = spectrum varies (different marble shapes)")
print()

# Are the spectral gaps consistent?
gaps = []
for s in prompt_spectra:
    if len(s) >= 2 and s[1] > 0:
        gaps.append(s[0] / s[1])  # This is always 1/spectrum[1] since spectrum[0]=1

print(f"  S0/S1 across prompts: mean={1/mean_spectrum[1]:.3f}  "
      f"(range=[{min(1/s[1] for s in prompt_spectra if s[1]>0):.3f}, "
      f"{max(1/s[1] for s in prompt_spectra if s[1]>0):.3f}])")
print()


# ================================================================
# TEST 4: PATH CURVATURE -- angles between consecutive positions
# ================================================================
print("=" * 80)
print("  TEST 4: PATH CURVATURE")
print("  Do consecutive positions bend the same way across prompts?")
print("=" * 80)
print()

# For each prompt, compute angles between consecutive residuals
# This traces the "path on the marble surface"

for layer in [6, 14, 22]:
    all_angles = []
    per_prompt_angles = []

    for pi, pd in enumerate(prompt_data):
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        n_pos = gates.shape[0]

        residuals = gates - scaffold[np.newaxis, :]

        angles = []
        for p in range(n_pos - 1):
            r1 = residuals[p]
            r2 = residuals[p + 1]
            n1 = np.linalg.norm(r1)
            n2 = np.linalg.norm(r2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_angle = np.dot(r1, r2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
                all_angles.append(angle)

        per_prompt_angles.append(angles)

    if all_angles:
        print(f"  Layer {layer}: consecutive angle = {np.mean(all_angles):.1f}° "
              f"+/- {np.std(all_angles):.1f}°  "
              f"(range=[{min(all_angles):.1f}°, {max(all_angles):.1f}°])")

        # Are the per-prompt MEAN angles consistent?
        prompt_means = [np.mean(a) if a else 0 for a in per_prompt_angles]
        print(f"          per-prompt mean angle CV = {np.std(prompt_means)/np.mean(prompt_means):.3f}")

print()
print("  Consistent angles = path bends the same way on every marble")
print("  Variable angles = each marble's surface path is different")
print()


# ================================================================
# TEST 5: SUBSPACE OVERLAP -- How rotated are the per-prompt subspaces?
# ================================================================
print("=" * 80)
print("  TEST 5: SUBSPACE OVERLAP (The Rotation)")
print("  How much do per-prompt 7D subspaces overlap?")
print("=" * 80)
print()

# For each pair of prompts, compute principal angles between their top-k subspaces
# This tells us how "rotated" the marbles are relative to each other

layer = 14
D_SUB = 7  # Compare 7D subspaces

# First, extract per-prompt top-D subspaces
prompt_subspaces = []
for pi, pd in enumerate(prompt_data):
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    residuals = gates - scaffold[np.newaxis, :]
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    d = min(D_SUB, Vt.shape[0])
    prompt_subspaces.append(Vt[:d].copy())  # [d, HIDDEN_DIM]

# Compute pairwise principal angles
# Principal angle between subspaces A, B: cos(theta) = SVD of A @ B^T
n = len(prompt_subspaces)
overlap_matrix = np.zeros((n, n))
min_angle_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        # Grassmann distance via principal angles
        M = prompt_subspaces[i] @ prompt_subspaces[j].T  # [d, d]
        svals = np.linalg.svd(M, compute_uv=False)
        svals = np.clip(svals, 0, 1)
        angles = np.degrees(np.arccos(svals))

        # Mean cosine = subspace overlap
        overlap = np.mean(svals)
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap

        # Smallest principal angle (most aligned direction)
        min_angle_matrix[i, j] = angles.min()
        min_angle_matrix[j, i] = angles.min()

# Off-diagonal statistics
offdiag_overlap = []
offdiag_min_angle = []
for i in range(n):
    for j in range(i + 1, n):
        offdiag_overlap.append(overlap_matrix[i, j])
        offdiag_min_angle.append(min_angle_matrix[i, j])

print(f"  Layer {layer}, D={D_SUB} subspace overlap (principal angles):")
print(f"    Mean subspace overlap (mean cos): {np.mean(offdiag_overlap):.4f} "
      f"+/- {np.std(offdiag_overlap):.4f}")
print(f"    Range: [{min(offdiag_overlap):.4f}, {max(offdiag_overlap):.4f}]")
print()
print(f"    Min principal angle (most aligned dir): {np.mean(offdiag_min_angle):.1f}° "
      f"+/- {np.std(offdiag_min_angle):.1f}°")
print(f"    Range: [{min(offdiag_min_angle):.1f}°, {max(offdiag_min_angle):.1f}°]")
print()

# Interpretation
mean_overlap = np.mean(offdiag_overlap)
if mean_overlap > 0.7:
    print("  >> HIGH OVERLAP: subspaces share most directions (marbles barely rotated)")
elif mean_overlap > 0.3:
    print("  >> MODERATE OVERLAP: subspaces partially shared (marbles partially rotated)")
elif mean_overlap > 0.1:
    print("  >> LOW OVERLAP: subspaces mostly different (marbles heavily rotated)")
else:
    print("  >> NEAR-ZERO OVERLAP: subspaces nearly orthogonal (each prompt's marble rotated independently)")
print()

# Do this across layers
print(f"  Subspace overlap across layers (D={D_SUB}):")
for layer in range(COMB_START, COMB_END):
    subspaces = []
    for pi, pd in enumerate(prompt_data):
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        residuals = gates - scaffold[np.newaxis, :]
        U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
        d = min(D_SUB, Vt.shape[0])
        subspaces.append(Vt[:d].copy())

    overlaps = []
    for i in range(len(subspaces)):
        for j in range(i + 1, len(subspaces)):
            M = subspaces[i] @ subspaces[j].T
            svals = np.linalg.svd(M, compute_uv=False)
            overlaps.append(np.mean(np.clip(svals, 0, 1)))

    print(f"    Layer {layer:2d}: mean overlap = {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f}")
print()


# ================================================================
# TEST 6: THE CONE -- What's the angular width of the "cone of possibilities"?
# ================================================================
print("=" * 80)
print("  TEST 6: THE CONE OF POSSIBILITIES")
print("  How wide is the cone? Does its width correlate with model confidence?")
print("=" * 80)
print()

# The "cone" for each prompt at each layer is the spread of residuals
# across positions. The singular values define the cone's shape.
# A narrow cone = tight path, wide cone = many possibilities.

layer = 14
print(f"  Layer {layer} cone analysis:")
print()

cone_widths = []
confidences = []

for pi, pd in enumerate(prompt_data):
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = gates.shape[0]

    residuals = gates - scaffold[np.newaxis, :]
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)

    # Cone width: ratio of S1/S0 (how spread is the cone beyond the main axis?)
    if S[0] > 0:
        cone_width = S[1] / S[0] if len(S) > 1 else 0
    else:
        cone_width = 0

    # Model confidence: how peaked is the logit distribution?
    logits = pd['base_logits']
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    top_prob = probs.max()
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    cone_widths.append(cone_width)
    confidences.append(top_prob)

    print(f"  {pd['prompt'][:40]:>40s}  cone={cone_width:.4f}  "
          f"conf={top_prob:.4f}  entropy={entropy:.2f}  n_pos={n_pos}")

# Correlation between cone width and confidence
from numpy import corrcoef
if len(cone_widths) > 2:
    corr = corrcoef(cone_widths, confidences)[0, 1]
    print()
    print(f"  Correlation(cone_width, confidence) = {corr:.4f}")
    if abs(corr) > 0.5:
        print(f"  >> SIGNIFICANT: wider cone = {'less' if corr < 0 else 'more'} confident")
    else:
        print(f"  >> WEAK: cone width and confidence are not strongly related")
print()


# ================================================================
# TEST 7: RADIUS CONSISTENCY PER POSITION (not just last)
# ================================================================
print("=" * 80)
print("  TEST 7: MARBLE RADIUS PER POSITION")
print("  Is the 'marble size' consistent at each position in the path?")
print("=" * 80)
print()

# For prompts of similar length, compare residual norms at each position
layer = 14

# Group prompts by length
from collections import defaultdict
len_groups = defaultdict(list)
for pi, pd in enumerate(prompt_data):
    n_pos = pd['gates'][layer].shape[0]
    len_groups[n_pos].append(pi)

for n_pos, indices in sorted(len_groups.items()):
    if len(indices) < 3:
        continue

    print(f"  Prompts with {n_pos} tokens ({len(indices)} prompts):")

    # Per-position residual norms
    pos_norms = np.zeros((len(indices), n_pos))
    for i, pi in enumerate(indices):
        pd = prompt_data[pi]
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        residuals = gates - scaffold[np.newaxis, :]
        for p in range(n_pos):
            pos_norms[i, p] = np.linalg.norm(residuals[p])

    # Per-position statistics
    for p in range(n_pos):
        norms = pos_norms[:, p]
        cv = np.std(norms) / np.mean(norms) if np.mean(norms) > 0 else 0
        bar = "#" * int(np.mean(norms) / 5)
        print(f"    pos {p:2d}: |residual| = {np.mean(norms):6.1f} +/- {np.std(norms):5.1f}  CV={cv:.3f}  {bar}")

    print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: THE MARBLE GEOMETRY")
print("=" * 80)
print()
print("  The marble jar analogy predicts:")
print("    1. Marble SIZE (residual norm) should be consistent across prompts")
print("    2. D* should be consistent (path needs same number of DOF)")
print("    3. Singular value spectrum should be universal (same path shape)")
print("    4. Path curvature should be consistent (same bending)")
print("    5. Subspace DIRECTIONS should differ (marbles rotated differently)")
print()

# Free model
del model
torch.cuda.empty_cache()

# Save results
results = {
    'n_prompts': N_PROMPTS,
    'prompts': PROMPTS,
    'd_stars_90': [int(d) for d in prompt_d_stars],
    'd_stars_95': [int(d) for d in prompt_d_stars_95],
    'cone_widths': [float(c) for c in cone_widths],
    'confidences': [float(c) for c in confidences],
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8m_marble_geometry.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to {results_path}")
