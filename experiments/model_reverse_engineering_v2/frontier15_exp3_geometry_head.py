#!/usr/bin/env python3
"""
Frontier 15 Experiment 3: Geometry Head Prototype (standalone)
==============================================================

How few SVD dimensions of lm_head are needed to identify the
correct output token from a hidden state?

Memory-efficient: loads only what's needed, frees aggressively.
Uses float32 for lm_head to halve memory vs float64.

DC 289 §6
"""

import numpy as np
import os
import sys
import json
import gc

PHI = (1 + np.sqrt(5)) / 2
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')
GRID = 128
EPS = 1e-6


def decode_phi(path, dtype=np.float64):
    """Decode φ-encoded weight matrix."""
    d = np.load(path)
    signs = d['signs'].astype(dtype)
    exponents = d['exponents'].astype(dtype)
    return signs * (dtype(PHI) ** (exponents / dtype(GRID)))


def rms_norm(x, weight):
    """RMSNorm."""
    rms = np.sqrt(np.mean(x ** 2) + EPS)
    return (x / rms) * weight.astype(x.dtype)


def silu(x):
    """SiLU activation."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def load_tokenizer():
    """Load tokenizer vocabulary."""
    for candidate in [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
    ]:
        if os.path.exists(candidate):
            snapshots = os.listdir(candidate)
            if snapshots:
                vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                if os.path.exists(vocab_file):
                    with open(vocab_file, 'r') as f:
                        tokenizer_data = json.load(f)
                    vocab = tokenizer_data.get('model', {}).get('vocab', {})
                    id_to_token = {idx: tok for tok, idx in vocab.items()}
                    token_to_id = {}
                    for tok, idx in vocab.items():
                        token_to_id[tok] = idx
                        token_to_id[tok.lower()] = idx
                    return id_to_token, token_to_id
    return None, None


def find_token_id(word, token_to_id):
    for c in [word, word.lower(), word.capitalize(), word.upper(),
              f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
              f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}"]:
        if c in token_to_id:
            return token_to_id[c], c
    return None, None


def single_token_forward(token_id):
    """Minimal forward pass for one token. Returns final normed hidden state."""
    # Load embedding for this one token only
    d = np.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    x = d['signs'][token_id].astype(np.float64) * (PHI ** (d['exponents'][token_id].astype(np.float64) / GRID))
    del d; gc.collect()

    config = json.load(open(os.path.join(MODEL_DIR, 'config.json')))
    num_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    head_dim = config['head_dim']
    heads_per_kv = num_heads // num_kv_heads

    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')

        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        input_ln = norms['input_layernorm'].astype(np.float64)
        post_attn_ln = norms['post_attention_layernorm'].astype(np.float64)

        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        v_bias = biases['v_proj_bias'].astype(np.float64)

        # Attention (single token: weight=1, only V matters)
        x_normed = rms_norm(x, input_ln)

        v_proj = decode_phi(os.path.join(layer_dir, 'v_proj.npz'))
        v = v_proj @ x_normed + v_bias
        del v_proj; gc.collect()

        attn_out = np.zeros(num_heads * head_dim)
        for h in range(num_heads):
            kv_idx = h // heads_per_kv
            attn_out[h * head_dim:(h + 1) * head_dim] = v[kv_idx * head_dim:(kv_idx + 1) * head_dim]

        o_proj = decode_phi(os.path.join(layer_dir, 'o_proj.npz'))
        x = x + o_proj @ attn_out
        del o_proj; gc.collect()

        # MLP
        x_normed = rms_norm(x, post_attn_ln)

        gate_proj = decode_phi(os.path.join(layer_dir, 'gate_proj.npz'))
        gate = gate_proj @ x_normed
        del gate_proj; gc.collect()

        up_proj = decode_phi(os.path.join(layer_dir, 'up_proj.npz'))
        up = up_proj @ x_normed
        del up_proj; gc.collect()

        intermediate = silu(gate) * up

        down_proj = decode_phi(os.path.join(layer_dir, 'down_proj.npz'))
        x = x + down_proj @ intermediate
        del down_proj; gc.collect()

        print(f"    Layer {layer_idx:2d}: ||x||={np.linalg.norm(x):.2f}")

    final_norm_w = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))['weight'].astype(np.float64)
    return rms_norm(x, final_norm_w)


def main():
    print()
    print("=" * 80)
    print("  Experiment 3: Geometry Head Prototype")
    print("  How few SVD dimensions to identify the correct token?")
    print("=" * 80)
    print()

    id_to_token, token_to_id = load_tokenizer()
    if id_to_token is None:
        print("  ERROR: Could not load tokenizer")
        return

    # Use "dragon" as test token
    dragon_id, dragon_tok = find_token_id("dragon", token_to_id)
    print(f"  Test token: '{dragon_tok}' (id={dragon_id})")
    print(f"  Running forward pass through 28 layers...")
    print()

    x_final = single_token_forward(dragon_id)
    print(f"\n  Final hidden state: ||x||={np.linalg.norm(x_final):.4f}")
    gc.collect()

    # Load lm_head in float32 to save memory (4GB → 2GB)
    print("\n  Loading lm_head (float32 to save memory)...")
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'), dtype=np.float32)
    print(f"  lm_head shape: {lm_head.shape}, dtype: {lm_head.dtype}")

    # Full logits
    x_f32 = x_final.astype(np.float32)
    full_logits = lm_head @ x_f32
    full_top10_idx = np.argsort(full_logits)[-10:][::-1]
    print(f"\n  Full lm_head top 10:")
    for i, idx in enumerate(full_top10_idx):
        tok = id_to_token.get(idx, f"tok_{idx}")
        print(f"    {i}: {tok!r:>25s}  logit={full_logits[idx]:.4f}")

    full_rank = int(np.sum(full_logits > full_logits[dragon_id]))
    print(f"\n  'dragon' token rank: {full_rank}")
    # Use the ACTUAL top predicted token as the "correct" one for geometry head test
    top_token_id = full_top10_idx[0]
    top_token = id_to_token.get(top_token_id, '?')
    print(f"  Top predicted token: '{top_token}' (id={top_token_id})")

    # SVD via eigendecomposition of lm_head^T @ lm_head
    # Use float32 throughout to halve memory
    print("\n  Computing covariance matrix (3584 x 3584, float32)...")
    cov = lm_head.T @ lm_head  # (3584, 3584) float32
    print("  Eigendecomposition...")
    eigenvalues, V = np.linalg.eigh(cov.astype(np.float64))
    V = V.astype(np.float32)
    del cov; gc.collect()

    # Sort descending
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    V = V[:, idx_sort]

    # Project lm_head and x into SVD basis
    print("  Projecting into SVD basis...")
    x_in_basis = (V.T @ x_f32)  # (3584,)
    projected_lm = lm_head @ V  # (152064, 3584)
    del lm_head; gc.collect()

    # Test increasing dimensions
    test_dims = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 3584]

    print(f"\n  {'Dims':>6s}  {'Top rank':>8s}  {'Dragon rank':>12s}  {'Top token':>25s}  {'Var %':>8s}")
    print("  " + "-" * 70)

    total_var = float(np.sum(eigenvalues))
    first_rank1_top = None
    first_rank1_dragon = None

    for k in test_dims:
        k = min(k, 3584)
        logits_k = projected_lm[:, :k] @ x_in_basis[:k]

        rank_top = int(np.sum(logits_k > logits_k[top_token_id]))
        rank_dragon = int(np.sum(logits_k > logits_k[dragon_id]))
        top_idx = np.argmax(logits_k)
        top_tok = id_to_token.get(top_idx, f"tok_{top_idx}")
        var_pct = float(np.sum(eigenvalues[:k])) / total_var * 100

        markers = ""
        if rank_top == 0: markers += " ★top"
        if rank_dragon == 0: markers += " ★dragon"

        print(f"  {k:6d}  {rank_top:8d}  {rank_dragon:12d}  {top_tok!r:>25s}  {var_pct:7.1f}%{markers}")

        if rank_top == 0 and first_rank1_top is None:
            first_rank1_top = k
        if rank_dragon == 0 and first_rank1_dragon is None:
            first_rank1_dragon = k

    print()
    if first_rank1_top:
        print(f"  ★ Top predicted token becomes rank 1 at {first_rank1_top} dimensions")
        print(f"    ({first_rank1_top}/3584 = {first_rank1_top/3584*100:.1f}% of total)")
    if first_rank1_dragon:
        print(f"  ★ 'dragon' becomes rank 1 at {first_rank1_dragon} dimensions")
    else:
        print(f"  'dragon' never reached rank 1 (full rank = {full_rank})")
    print()


if __name__ == '__main__':
    main()
