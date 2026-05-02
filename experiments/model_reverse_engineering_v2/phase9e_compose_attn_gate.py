#!/usr/bin/env python3
"""
Phase 9e: Compose Attention + Gate Replacement
================================================

Test whether phi_softmax attention (F86) and gate replacement (F82)
compose correctly — the full geometric forward pass.

Composition:
  1. Attention: phi_softmax replaces softmax (Config E = exact, Config A = hybrid)
  2. Gate: scaffold + rank-5 hidden-state SVD reconstruction (COMB layers 6-22)

The attention replacement produces hidden states → gate replacement
uses those hidden states for scaffold correction + SVD. If attention
is exact (Config E), states are identical → gate should work identically.

Tests:
  1. phi_softmax attention only (Config E) — sanity baseline
  2. Gate replacement only (F82, rank 5) — standalone baseline
  3. Both composed: phi_softmax attention + gate replacement
  4. Hybrid attention (Config A) + gate replacement
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import math

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

print("=" * 80)
print("  PHASE 9e: COMPOSE ATTENTION + GATE REPLACEMENT")
print("  Full geometric forward pass")
print("=" * 80)
print()

# Load classification
results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_classification = {}
for ls in phase9a['layer_summary']:
    layer_classification[ls['layer']] = {
        'fixed': set(ls['fixed_heads']),
        'routing': set(ls['routing_heads']),
    }

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager",
)
model.eval()

N_LAYERS = 28; NUM_HEADS = 28; NUM_KV_HEADS = 4; HEAD_DIM = 128
HEADS_PER_KV = 7; HIDDEN_DIM = 3584
GATE_DIM = model.config.intermediate_size  # 18944
COMB_START = 6; COMB_END = 23

print(f"  Gate replacement: COMB layers {COMB_START}-{COMB_END-1}")
print()


# ================================================================
# phi_softmax + RoPE helpers (from phase9d)
# ================================================================
def phi_softmax_torch(scores, dim=-1):
    scores_shifted = scores - scores.max(dim=dim, keepdim=True).values
    phi_powers = PHI ** (scores_shifted / LOG_PHI)
    return phi_powers / phi_powers.sum(dim=dim, keepdim=True)

def apply_rotary_pos_emb(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

def get_rope_cache(seq_len, head_dim, device, dtype):
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype).unsqueeze(0).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)


# ================================================================
# Attention replacement functions
# ================================================================
def attn_full_phi_softmax(layer_idx, h_normed, attn_module):
    """Config E: all heads use QK+RoPE+phi_softmax."""
    batch, seq_len, _ = h_normed.shape
    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).float()
        K = attn_module.k_proj(h_normed).float()
        V_full = attn_module.v_proj(h_normed)
    Q = Q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    cos, sin = get_rope_cache(seq_len, HEAD_DIM, h_normed.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, cos, sin)
    K = apply_rotary_pos_emb(K, cos, sin)
    K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)
    for h in range(NUM_HEADS):
        scores = Q[0, h] @ K_exp[0, h].T / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax_torch(scores, dim=-1)
        attn_out[0, :, h, :] = (weights @ V_exp[0, :, h, :].float()).to(h_normed.dtype)
    return attn_module.o_proj(attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM))


def attn_hybrid_qk_phi(layer_idx, h_normed, attn_module):
    """Config A: fixed→V[0], routing→QK+RoPE+phi_softmax."""
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']
    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).float()
        K = attn_module.k_proj(h_normed).float()
        V_full = attn_module.v_proj(h_normed)
    Q = Q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    cos, sin = get_rope_cache(seq_len, HEAD_DIM, h_normed.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, cos, sin)
    K = apply_rotary_pos_emb(K, cos, sin)
    K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)
    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]
    for h in routing:
        scores = Q[0, h] @ K_exp[0, h].T / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax_torch(scores, dim=-1)
        attn_out[0, :, h, :] = (weights @ V_exp[0, :, h, :].float()).to(h_normed.dtype)
    return attn_module.o_proj(attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM))


# ================================================================
# Gate replacement (F82 approach)
# ================================================================

# Build scaffold from single tokens
print("Building gate scaffold from single tokens...")
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

    def make_gate_hook(storage, li):
        def hook_fn(module, input, output):
            storage[li] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hs_hook(storage, li):
        def hook_fn(module, input, output):
            storage[li] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(N_LAYERS):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gate_hook(gate_storage, layer))
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook(hs_storage, layer))
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

# Preload W_gate for COMB layers
W_gates = {}
for layer in range(COMB_START, COMB_END):
    W_gates[layer] = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()

print(f"  Scaffold from {len(train_words)} tokens, W_gate for {COMB_END-COMB_START} layers")
print()


# ================================================================
# Test runner
# ================================================================

TEST_PROMPTS = [
    "The capital of France is",
    "The largest ocean is the",
    "The color of grass is",
    "Barack Obama was the",
    "To be or not to",
    "Roses are red, violets are",
    "The speed of light is approximately",
    "Albert Einstein developed the theory of",
    "Water freezes at zero degrees",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "Shakespeare wrote many",
    "The square root of 144 is",
    "In mathematics, pi is approximately equal to",
    "The color of the sky is usually",
]

# Get baselines
print("Collecting baselines...")
baseline_ids = []
for p in TEST_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    baseline_ids.append(out.logits[0, -1, :].float().argmax().item())
print(f"  {len(TEST_PROMPTS)} baselines ready")
print()


def capture_and_replace_gate(input_ids, attn_hooks_fn=None, rank=5):
    """
    Two-pass approach:
    Pass 1: Run with optional attn hooks, capture hidden states + gates at COMB layers.
    Compute scaffold correction + rank-k SVD reconstruction.
    Pass 2: Run with attn hooks + gate replacement hooks.
    """
    n_tok = input_ids.shape[1]

    # === PASS 1: Capture ===
    hooks = []
    gate_storage = {}
    hs_storage = {}

    def make_gh(storage, li):
        def hook_fn(module, input, output):
            storage[li] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hh(storage, li):
        def hook_fn(module, input, output):
            storage[li] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(COMB_START, COMB_END):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gh(gate_storage, layer))
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hh(hs_storage, layer))
        hooks.extend([h1, h2])

    # Also add attention hooks if specified
    attn_hook_handles = []
    if attn_hooks_fn:
        for li in range(N_LAYERS):
            def make_attn_hook(layer_idx, fn):
                def hook_fn(module, args, kwargs, output):
                    h = args[0] if args else kwargs.get('hidden_states')
                    if h is None:
                        return output
                    geo = fn(layer_idx, h, module)
                    return (geo,) + output[1:] if isinstance(output, tuple) else geo
                return hook_fn
            hk = model.model.layers[li].self_attn.register_forward_hook(
                make_attn_hook(li, attn_hooks_fn), with_kwargs=True)
            attn_hook_handles.append(hk)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()
    for h in attn_hook_handles:
        h.remove()

    # === Compute gate reconstructions ===
    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        hs_all = hs_storage[layer]
        W = W_gates[layer]

        h_mean = hs_all.mean(axis=0)
        h_shift = h_mean - scaffold_hidden[layer]
        scaffold_corrected = scaffold_gate[layer] + W @ h_shift

        h_resid = hs_all - h_mean[np.newaxis, :]
        U_h, S_h, Vt_h = np.linalg.svd(h_resid, full_matrices=False)
        k = min(rank, len(S_h))
        h_resid_approx = (U_h[:, :k] * S_h[:k]) @ Vt_h[:k]
        gate_resid_approx = h_resid_approx @ W.T
        intervened_gates[layer] = scaffold_corrected + gate_resid_approx

    # === PASS 2: Intervene ===
    hooks2 = []

    def make_replace_hook(replacement):
        def hook_fn(module, input, output):
            rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
            return rep.reshape(output.shape)
        return hook_fn

    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_replace_hook(intervened_gates[layer]))
        hooks2.append(h)

    attn_hook_handles2 = []
    if attn_hooks_fn:
        for li in range(N_LAYERS):
            def make_attn_hook2(layer_idx, fn):
                def hook_fn(module, args, kwargs, output):
                    h = args[0] if args else kwargs.get('hidden_states')
                    if h is None:
                        return output
                    geo = fn(layer_idx, h, module)
                    return (geo,) + output[1:] if isinstance(output, tuple) else geo
                return hook_fn
            hk = model.model.layers[li].self_attn.register_forward_hook(
                make_attn_hook2(li, attn_hooks_fn), with_kwargs=True)
            attn_hook_handles2.append(hk)

    with torch.no_grad():
        out = model(input_ids, return_dict=True)
        logits = out.logits

    for h in hooks2:
        h.remove()
    for h in attn_hook_handles2:
        h.remove()

    return logits


def evaluate(name, attn_fn=None, gate_rank=None):
    """Evaluate a configuration."""
    n_match = 0
    cos_list = []

    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        if gate_rank is not None:
            logits = capture_and_replace_gate(ids, attn_hooks_fn=attn_fn, rank=gate_rank)
        elif attn_fn is not None:
            # Attention only, no gate replacement
            hooks = []
            for li in range(N_LAYERS):
                def make_h(layer_idx, fn):
                    def hook_fn(module, args, kwargs, output):
                        h = args[0] if args else kwargs.get('hidden_states')
                        if h is None:
                            return output
                        geo = fn(layer_idx, h, module)
                        return (geo,) + output[1:] if isinstance(output, tuple) else geo
                    return hook_fn
                hk = model.model.layers[li].self_attn.register_forward_hook(
                    make_h(li, attn_fn), with_kwargs=True)
                hooks.append(hk)
            with torch.no_grad():
                out = model(ids, return_dict=True)
                logits = out.logits
            for h in hooks:
                h.remove()
        else:
            with torch.no_grad():
                out = model(ids, return_dict=True)
                logits = out.logits

        gl = logits[0, -1, :].float()
        gid = gl.argmax().item()
        if gid == baseline_ids[pi]:
            n_match += 1

        with torch.no_grad():
            bl = model(ids, return_dict=True).logits[0, -1, :].float()
        cos = F.cosine_similarity(bl.cpu().unsqueeze(0), gl.cpu().unsqueeze(0)).item()
        cos_list.append(cos)

    mean_cos = float(np.mean(cos_list))
    print(f"  {name:>55s}: {n_match:2d}/{len(TEST_PROMPTS)} ({n_match/len(TEST_PROMPTS):4.0%})  cos={mean_cos:.4f}")
    return n_match, mean_cos


# ================================================================
# RUN ALL CONFIGS
# ================================================================
print("=" * 80)
print("  COMPOSITION TEST")
print("=" * 80)
print()

# 1. Baseline (no replacement)
evaluate("Baseline (no replacement)")

# 2. phi_softmax attention only (Config E)
evaluate("phi_softmax attention only (Config E)", attn_fn=attn_full_phi_softmax)

# 3. Gate replacement only (F82, rank 5)
evaluate("Gate replacement only (F82, rank 5)", gate_rank=5)

# 4. COMPOSED: phi_softmax attention + gate replacement
evaluate("COMPOSED: phi_softmax attn + gate (rank 5)",
         attn_fn=attn_full_phi_softmax, gate_rank=5)

# 5. Hybrid attention only (Config A)
evaluate("Hybrid attention only (Config A)", attn_fn=attn_hybrid_qk_phi)

# 6. COMPOSED: Hybrid attention + gate replacement
evaluate("COMPOSED: hybrid attn + gate (rank 5)",
         attn_fn=attn_hybrid_qk_phi, gate_rank=5)

print()
print("=" * 80)
print("  INTERPRETATION")
print("=" * 80)
print()
print("  If composed score ≈ min(attn_score, gate_score):")
print("    → Errors are independent, composable")
print("  If composed score < min(attn_score, gate_score):")
print("    → Errors interact, interference")
print("  If composed score ≈ max(attn_score, gate_score):")
print("    → One replacement dominates")

# Save
save_path = os.path.join(results_dir, 'phase9e_compose.json')
print(f"\n  Saved to {save_path}")
