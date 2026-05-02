#!/usr/bin/env python3
"""
Phase 8L: Predict α₁..α₇ from Hidden State (No Oracle)
========================================================

Finding 72 showed D*=7: seven scalars per token per layer give 100% top-1.
But those scalars were computed using the TRUE gate (oracle).

Key insight: prediction is EXACT via linear algebra:
  α_k = dir_k · (gate - scaffold) = dir_k · W_gate @ (h - h_ref)
      = (W_gate^T @ dir_k) · (h - h_ref)
      = w_k · (h - h_ref)

where w_k = W_gate^T @ dir_k is a precomputable 3584-dim vector.

So there's no "learning" — it's exact. The REAL question:
Do the 7 directions generalize to UNSEEN prompts?

Plan:
1. Expanded prompt set (25 diverse prompts)
2. Train/test split (15 train, 10 test)
3. Extract directions from TRAIN set only
4. Predict α using exact linear formula
5. Test on HELD-OUT prompts with intervention
6. Sweep D (1..14) to find generalized D*
7. Compare oracle vs predicted reconstruction

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
print("  PHASE 8L: PREDICT α FROM HIDDEN STATE (NO ORACLE)")
print("  Can 7 directions generalize to unseen prompts?")
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
HIDDEN_DIM = model.config.intermediate_size      # 18944
HIDDEN_STATE_DIM = model.config.hidden_size       # 3584


# ================================================================
# STEP 1: Build crystal from single tokens
# ================================================================
print("-" * 80)
print("  STEP 1: Build crystal from single tokens")
print("-" * 80)
print()

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

scaffold_single = all_gates_single.mean(axis=0)   # [N_LAYERS, HIDDEN_DIM]
h_mean_single = all_hs_single.mean(axis=0)         # [N_LAYERS, HIDDEN_STATE_DIM]

# Dir1 from single-token SVD
residuals_single = all_gates_single - scaffold_single[np.newaxis, :, :]
svd_single = {}
for layer in range(COMB_START, COMB_END):
    res = residuals_single[:, layer, :]
    U, S, Vt = np.linalg.svd(res, full_matrices=False)
    svd_single[layer] = {'U': U, 'S': S, 'Vt': Vt}

print(f"  Crystal: {N_SINGLE} tokens, {COMB_END - COMB_START} COMB layers")
print()


# ================================================================
# STEP 2: Expanded prompt set with train/test split
# ================================================================
print("-" * 80)
print("  STEP 2: Expanded prompt set (train/test split)")
print("-" * 80)
print()

# TRAIN prompts: used to extract directions
TRAIN_PROMPTS = [
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
    "The tallest mountain in the world is",
    "The square root of 144 is",
    "Shakespeare wrote the play Romeo and",
    "The human body has a total of",
    "The chemical formula for water is",
]

# TEST prompts: NEVER seen during direction extraction
TEST_PROMPTS = [
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
]

ALL_PROMPTS = TRAIN_PROMPTS + TEST_PROMPTS

print(f"  Train prompts: {len(TRAIN_PROMPTS)}")
print(f"  Test prompts:  {len(TEST_PROMPTS)}")
print()


# ================================================================
# STEP 3: Capture all prompts
# ================================================================
print("-" * 80)
print("  STEP 3: Capture gates and hidden states for all prompts")
print("-" * 80)
print()

prompt_data = []
for prompt in ALL_PROMPTS:
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

    is_train = prompt in TRAIN_PROMPTS
    prompt_data.append({
        'prompt': prompt,
        'is_train': is_train,
        'n_tokens': n_tok,
        'base_logits': base_logits,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
    })

train_data = [pd for pd in prompt_data if pd['is_train']]
test_data = [pd for pd in prompt_data if not pd['is_train']]

print(f"  Captured {len(prompt_data)} prompts ({len(train_data)} train, {len(test_data)} test)")
print()


# ================================================================
# STEP 4: Extract directions from TRAIN set only
# ================================================================
print("-" * 80)
print("  STEP 4: Extract directions from TRAIN set (iterative peel)")
print("-" * 80)
print()

# Precompute stereo scaffolds for ALL prompts
corrected_scaffolds = {}
for pi, pd in enumerate(prompt_data):
    corrected_scaffolds[pi] = {}
    for layer in range(COMB_START, COMB_END):
        hs_all_pos = pd['hs'][layer]
        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        corrected_scaffolds[pi][layer] = scaffold_single[layer] + W_gate @ h_shift

MAX_DIMS = 14  # Up to 14 directions (limited by train set size = 15)

# Initialize with dir1 from single-token SVD
directions = {}  # layer -> list of directions
for layer in range(COMB_START, COMB_END):
    directions[layer] = [svd_single[layer]['Vt'][0].copy()]

# Iteratively peel using TRAIN set only
train_indices = [i for i, pd in enumerate(prompt_data) if pd['is_train']]

for dim_k in range(2, MAX_DIMS + 1):
    for layer in range(COMB_START, COMB_END):
        existing_dirs = np.stack(directions[layer])  # [k-1, HIDDEN_DIM]
        residual_stack = []

        for pi in train_indices:
            pd = prompt_data[pi]
            gates_all_pos = pd['gates'][layer]
            scaffold = corrected_scaffolds[pi][layer]
            last_pos = gates_all_pos.shape[0] - 1

            residual = gates_all_pos[last_pos] - scaffold
            projections = residual @ existing_dirs.T
            projected = projections @ existing_dirs
            remaining = residual - projected
            residual_stack.append(remaining)

        residual_matrix = np.stack(residual_stack)
        U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
        directions[layer].append(Vt[0].copy())

print(f"  Extracted up to {MAX_DIMS} directions per layer from {len(train_indices)} train prompts")
print()


# ================================================================
# STEP 5: Precompute w_k = W_gate^T @ dir_k (the prediction vectors)
# ================================================================
print("-" * 80)
print("  STEP 5: Precompute prediction vectors w_k = W_gate^T @ dir_k")
print("-" * 80)
print()

# w_k is a 3584-dim vector: α_k = w_k · (h - h_ref)
prediction_vectors = {}  # layer -> [w_1, w_2, ..., w_D]
for layer in range(COMB_START, COMB_END):
    W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    # W_gate is [18944, 3584], dir_k is [18944]
    # w_k = W_gate^T @ dir_k = [3584]
    pvecs = []
    for d in range(len(directions[layer])):
        w_k = W_gate.T @ directions[layer][d]
        pvecs.append(w_k)
    prediction_vectors[layer] = pvecs

print(f"  Precomputed {MAX_DIMS} prediction vectors per layer")
print(f"  Each w_k is {HIDDEN_STATE_DIM}-dim (vs gate W_gate = {HIDDEN_DIM}x{HIDDEN_STATE_DIM})")
print(f"  Storage: {MAX_DIMS} x {HIDDEN_STATE_DIM} = {MAX_DIMS * HIDDEN_STATE_DIM:,} params per layer")
print(f"  vs W_gate: {HIDDEN_DIM * HIDDEN_STATE_DIM:,} params per layer")
print(f"  Ratio: {HIDDEN_DIM * HIDDEN_STATE_DIM / (MAX_DIMS * HIDDEN_STATE_DIM):.0f}x smaller")
print()


# ================================================================
# STEP 6: Verify exact prediction (oracle vs hidden-state predicted)
# ================================================================
print("=" * 80)
print("  STEP 6: VERIFY EXACT PREDICTION")
print("  α_k from oracle (dir_k · residual) vs predicted (w_k · (h - h_ref))")
print("=" * 80)
print()

# For a few prompts, compare oracle alpha vs predicted alpha
for pi in [0, len(TRAIN_PROMPTS)]:  # One train, one test
    pd = prompt_data[pi]
    label = "TRAIN" if pd['is_train'] else "TEST"
    print(f"  [{label}] {pd['prompt']}")

    for layer in [6, 14, 22]:
        gates_all_pos = pd['gates'][layer]
        hs_all_pos = pd['hs'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        last_pos = gates_all_pos.shape[0] - 1

        # Oracle: project true gate residual
        residual_true = gates_all_pos[last_pos] - scaffold
        oracle_alphas = []
        for d in range(min(7, len(directions[layer]))):
            alpha_oracle = np.dot(residual_true, directions[layer][d])
            oracle_alphas.append(alpha_oracle)

        # Predicted: w_k · (h - h_ref)
        h_last = hs_all_pos[last_pos]
        h_mean_prompt = hs_all_pos.mean(axis=0)
        # h_ref for stereo correction: the "expected" h for this scaffold
        # scaffold_corrected = scaffold_single + W_gate @ (h_mean_prompt - h_mean_single)
        # So gate - scaffold_corrected = W_gate @ h - scaffold_single - W_gate @ h_shift
        #                               = W_gate @ (h - h_mean_single - h_shift)
        #                               = W_gate @ (h - h_mean_prompt)
        # Therefore α_k = dir_k · W_gate @ (h - h_mean_prompt) = w_k · (h - h_mean_prompt)
        predicted_alphas = []
        for d in range(min(7, len(prediction_vectors[layer]))):
            alpha_pred = np.dot(prediction_vectors[layer][d], h_last - h_mean_prompt)
            predicted_alphas.append(alpha_pred)

        max_err = max(abs(o - p) for o, p in zip(oracle_alphas, predicted_alphas))
        rel_errs = [abs(o - p) / (abs(o) + 1e-10) for o, p in zip(oracle_alphas, predicted_alphas)]
        max_rel = max(rel_errs)

        print(f"    Layer {layer}: max|oracle-pred| = {max_err:.6f}, max_rel = {max_rel:.4%}")

    print()

print("  (Errors should be near zero -- prediction is exact up to float precision)")
print()


# ================================================================
# STEP 7: Test generalization -- intervention on HELD-OUT prompts
# ================================================================
print("=" * 80)
print("  STEP 7: GENERALIZATION TEST")
print("  Intervene on HELD-OUT test prompts using predicted alphas")
print("  Directions extracted from TRAIN set only")
print("=" * 80)
print()

def make_replace_hook(replacement):
    def hook_fn(module, input, output):
        rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
        return rep.reshape(output.shape)
    return hook_fn

# Sweep number of dimensions
for D in [1, 2, 3, 5, 7, 9, 11, 14]:
    # Test on TRAIN set
    train_matches = 0
    train_cos = []

    for pi in train_indices:
        pd = prompt_data[pi]
        input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

        # Predict alphas from hidden states and reconstruct gate
        intervened_gates = {}
        for layer in range(COMB_START, COMB_END):
            hs_all_pos = pd['hs'][layer]
            scaffold = corrected_scaffolds[pi][layer]
            n_pos = hs_all_pos.shape[0]
            h_mean_prompt = hs_all_pos.mean(axis=0)

            dirs = np.stack(directions[layer][:D])
            pvecs = np.stack(prediction_vectors[layer][:D])

            # Predict alphas for ALL positions from hidden states
            h_centered = hs_all_pos - h_mean_prompt[np.newaxis, :]  # [n_pos, 3584]
            alphas_pred = h_centered @ pvecs.T                       # [n_pos, D]
            reconstruction = alphas_pred @ dirs                      # [n_pos, HIDDEN_DIM]
            intervened_gates[layer] = scaffold + reconstruction

        hooks = []
        for layer in range(COMB_START, COMB_END):
            h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
                make_replace_hook(intervened_gates[layer])
            )
            hooks.append(h)

        with torch.no_grad():
            int_out = model(input_ids)
            int_logits = int_out.logits[0, -1, :].cpu().float().numpy()

        for h in hooks:
            h.remove()

        base_l = pd['base_logits']
        cos = np.dot(base_l, int_logits) / (np.linalg.norm(base_l) * np.linalg.norm(int_logits) + 1e-10)
        train_cos.append(cos)
        if np.argmax(base_l) == np.argmax(int_logits):
            train_matches += 1

    # Test on TEST set (held-out)
    test_matches = 0
    test_cos = []
    test_results = []

    test_indices = [i for i, pd in enumerate(prompt_data) if not pd['is_train']]
    for pi in test_indices:
        pd = prompt_data[pi]
        input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

        intervened_gates = {}
        for layer in range(COMB_START, COMB_END):
            hs_all_pos = pd['hs'][layer]
            scaffold = corrected_scaffolds[pi][layer]
            n_pos = hs_all_pos.shape[0]
            h_mean_prompt = hs_all_pos.mean(axis=0)

            dirs = np.stack(directions[layer][:D])
            pvecs = np.stack(prediction_vectors[layer][:D])

            h_centered = hs_all_pos - h_mean_prompt[np.newaxis, :]
            alphas_pred = h_centered @ pvecs.T
            reconstruction = alphas_pred @ dirs
            intervened_gates[layer] = scaffold + reconstruction

        hooks = []
        for layer in range(COMB_START, COMB_END):
            h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
                make_replace_hook(intervened_gates[layer])
            )
            hooks.append(h)

        with torch.no_grad():
            int_out = model(input_ids)
            int_logits = int_out.logits[0, -1, :].cpu().float().numpy()

        for h in hooks:
            h.remove()

        base_l = pd['base_logits']
        cos = np.dot(base_l, int_logits) / (np.linalg.norm(base_l) * np.linalg.norm(int_logits) + 1e-10)
        test_cos.append(cos)

        base_top1 = np.argmax(base_l)
        int_top1 = np.argmax(int_logits)
        match = base_top1 == int_top1
        if match:
            test_matches += 1

        base_tok = tokenizer.decode([base_top1]).strip()
        int_tok = tokenizer.decode([int_top1]).strip()
        test_results.append({
            'prompt': pd['prompt'],
            'match': bool(match),
            'cos': float(cos),
            'base': base_tok,
            'int': int_tok,
        })

    train_rate = train_matches / len(train_indices) * 100
    test_rate = test_matches / len(test_indices) * 100
    train_cos_mean = np.mean(train_cos)
    test_cos_mean = np.mean(test_cos)

    print(f"  D={D:2d}:  TRAIN top-1={train_rate:5.1f}% cos={train_cos_mean:.4f}  |  "
          f"TEST top-1={test_rate:5.1f}% cos={test_cos_mean:.4f}")

    # Show per-prompt details for key D values
    if D in [7, 14]:
        print(f"         TEST per-prompt (D={D}):")
        for r in test_results:
            mark = "Y" if r['match'] else "N"
            print(f"           {mark} {r['prompt'][:45]:>45s}  -> {r['int']:>15s} (want {r['base']:>15s})")
        print()


# ================================================================
# STEP 8: The speedup calculation
# ================================================================
print()
print("=" * 80)
print("  STEP 8: THE SPEEDUP")
print("=" * 80)
print()

# Original gate computation
original_ops = HIDDEN_DIM * HIDDEN_STATE_DIM
print(f"  Original W_gate matmul: {HIDDEN_DIM} x {HIDDEN_STATE_DIM} = {original_ops:,} multiply-adds")
print()

for D in [7, 14]:
    # New computation:
    # Step A: α = h_centered @ pvecs^T  (n_pos x 3584) @ (3584 x D) = n_pos x D multiply-adds per pos: 3584 * D
    # Step B: gate = scaffold + α @ dirs  (n_pos x D) @ (D x 18944) = n_pos x 18944 multiply-adds per pos: D * 18944
    # Total per position: 3584*D + D*18944 = D * (3584 + 18944)
    new_ops_per_pos = D * (HIDDEN_STATE_DIM + HIDDEN_DIM)
    speedup = original_ops / new_ops_per_pos

    print(f"  D={D}:")
    print(f"    Predict α: {D} x {HIDDEN_STATE_DIM} = {D * HIDDEN_STATE_DIM:,} ops")
    print(f"    Reconstruct gate: {D} x {HIDDEN_DIM} = {D * HIDDEN_DIM:,} ops")
    print(f"    Total: {new_ops_per_pos:,} ops")
    print(f"    Speedup: {speedup:.1f}x")
    print(f"    Parameters: {D * HIDDEN_STATE_DIM:,} (w vectors) + {D * HIDDEN_DIM:,} (dirs) + {HIDDEN_DIM:,} (scaffold)")
    print(f"    vs original: {original_ops:,}")
    print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: PREDICTING FROM HIDDEN STATE")
print("=" * 80)
print()
print("  1. Prediction is EXACT: α_k = w_k · (h - h_mean_prompt)")
print(f"     where w_k = W_gate^T @ dir_k ({HIDDEN_STATE_DIM}-dim, precomputed)")
print()
print("  2. No learning or fitting needed — pure linear algebra")
print()
print("  3. The question is DIRECTION GENERALIZATION:")
print("     Do directions from 15 train prompts work on 10 test prompts?")
print()

# Free model
del model
torch.cuda.empty_cache()

# Save results
results = {
    'n_train_prompts': len(TRAIN_PROMPTS),
    'n_test_prompts': len(TEST_PROMPTS),
    'n_single_tokens': N_SINGLE,
    'hidden_dim': HIDDEN_DIM,
    'hidden_state_dim': HIDDEN_STATE_DIM,
    'train_prompts': TRAIN_PROMPTS,
    'test_prompts': TEST_PROMPTS,
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8l_predict_from_hidden.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
