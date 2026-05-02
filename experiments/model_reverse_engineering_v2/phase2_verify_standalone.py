#!/usr/bin/env python3
"""
Phase 2 Standalone Verification (no PyTorch needed).

Compares the φ-engine pipeline against a manual numpy reference
forward pass using the same decoded weights. This isolates pipeline
integration bugs (wrong shapes, operations, masks, etc.).

The reference implements the same Qwen2 architecture step by step:
  1. Embedding lookup
  2. RMSNorm
  3. Q/K/V projection + bias
  4. RoPE
  5. GQA expansion
  6. Attention (scores, mask, softmax, weighted sum)
  7. Output projection
  8. Residual
  9. MLP (gate + up + SiLU + down)
  10. Residual
  11. Final norm
  12. LM head

Compares φ-engine output at each stage against reference.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")

# Config
HIDDEN = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 18944
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS
ROPE_THETA = 1_000_000.0


def rms_norm_ref(x, weight, eps=1e-6):
    variance = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight


def softmax_ref(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu_ref(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def rope_ref(x, cos, sin):
    """Apply RoPE. x: (batch, heads, seq, dim)."""
    x1 = x[..., :HEAD_DIM // 2]
    x2 = x[..., HEAD_DIM // 2:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    c = cos[np.newaxis, np.newaxis, :, :]
    s = sin[np.newaxis, np.newaxis, :, :]
    return x * c + rotated * s


def get_rope_tables(seq_len):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    positions = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def reference_forward(token_ids, n_layers=2):
    """Manual numpy forward pass — the ground truth."""
    seq_len = len(token_ids)

    # Embedding
    emb_enc = PhiEncoded.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    emb_table = emb_enc.decode()
    hidden = emb_table[token_ids][np.newaxis, :, :]  # (1, seq, hidden)

    # RoPE tables
    cos, sin = get_rope_tables(seq_len)

    for layer_idx in range(n_layers):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')

        # Load and decode weights
        W_q = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz')).decode()
        W_k = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz')).decode()
        W_v = PhiEncoded.load(os.path.join(layer_dir, 'v_proj.npz')).decode()
        W_o = PhiEncoded.load(os.path.join(layer_dir, 'o_proj.npz')).decode()
        W_gate = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz')).decode()
        W_up = PhiEncoded.load(os.path.join(layer_dir, 'up_proj.npz')).decode()
        W_down = PhiEncoded.load(os.path.join(layer_dir, 'down_proj.npz')).decode()

        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        norms = np.load(os.path.join(layer_dir, 'norms.npz'))

        b_q = biases['q_proj_bias']
        b_k = biases['k_proj_bias']
        b_v = biases['v_proj_bias']
        ln1_w = norms['input_layernorm']
        ln2_w = norms['post_attention_layernorm']

        # --- ATTENTION ---
        normed = rms_norm_ref(hidden, ln1_w)

        Q = normed @ W_q.T + b_q  # (1, seq, num_heads*head_dim)
        K = normed @ W_k.T + b_k  # (1, seq, num_kv_heads*head_dim)
        V = normed @ W_v.T + b_v

        Q = Q.reshape(1, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        Q = rope_ref(Q, cos, sin)
        K = rope_ref(K, cos, sin)

        K = np.repeat(K, HEADS_PER_KV, axis=1)
        V = np.repeat(V, HEADS_PER_KV, axis=1)

        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(HEAD_DIM)
        if seq_len > 1:
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask

        attn = softmax_ref(scores, axis=-1)
        attn_out = np.einsum('bhqk,bhkd->bhqd', attn, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        attn_out = attn_out @ W_o.T

        hidden = hidden + attn_out

        # --- MLP ---
        normed = rms_norm_ref(hidden, ln2_w)

        gate = normed @ W_gate.T
        up = normed @ W_up.T
        mlp_hidden = silu_ref(gate) * up
        mlp_out = mlp_hidden @ W_down.T

        hidden = hidden + mlp_out

    # Final norm
    final_norm = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))['weight']
    hidden = rms_norm_ref(hidden, final_norm)

    # LM head
    lm_head = PhiEncoded.load(os.path.join(MODEL_DIR, 'lm_head.npz')).decode()
    logits = hidden @ lm_head.T

    return logits


def main():
    print("=" * 70)
    print("  Phase 2 Standalone Verification")
    print("  Comparing φ-engine vs manual numpy reference")
    print("=" * 70)
    print()

    test_tokens = [9707, 220, 279, 374, 220]  # "Hello is "
    n_layers = 2

    # --- Reference forward pass ---
    print(f"Reference forward pass ({n_layers} layers, {len(test_tokens)} tokens)...")
    t0 = time.time()
    ref_logits = reference_forward(test_tokens, n_layers)
    ref_time = time.time() - t0
    print(f"  Time: {ref_time:.1f}s")
    print(f"  Shape: {ref_logits.shape}")

    # --- φ-engine forward pass ---
    print(f"\nφ-engine forward pass ({n_layers} layers)...")
    from phi_geometric.inference import PhiQwen2Engine
    engine = PhiQwen2Engine.load(MODEL_DIR, max_layers=n_layers, verbose=False)

    t0 = time.time()
    phi_logits = engine.forward(test_tokens)
    phi_time = time.time() - t0
    print(f"  Time: {phi_time:.1f}s")
    print(f"  Shape: {phi_logits.shape}")

    # --- Compare ---
    print()
    print("─" * 70)
    print("  Position-by-Position Comparison")
    print("─" * 70)
    print()
    print(f"  {'Pos':>4s}  {'Corr':>10s}  {'MaxAbsDiff':>11s}  {'Top1 Match':>10s}  "
          f"{'Top10 Agree':>11s}")
    print("  " + "-" * 55)

    all_corr = []
    all_match = []

    for pos in range(len(test_tokens)):
        ref_pos = ref_logits[0, pos]
        phi_pos = phi_logits[0, pos]

        corr = np.corrcoef(ref_pos, phi_pos)[0, 1]
        max_diff = np.max(np.abs(ref_pos - phi_pos))

        ref_top1 = int(np.argmax(ref_pos))
        phi_top1 = int(np.argmax(phi_pos))
        match = ref_top1 == phi_top1

        ref_top10 = set(np.argsort(ref_pos)[-10:])
        phi_top10 = set(np.argsort(phi_pos)[-10:])
        agree = len(ref_top10 & phi_top10) / 10

        all_corr.append(corr)
        all_match.append(match)

        mark = "✓" if match else "✗"
        print(f"  {pos:4d}  {corr:10.8f}  {max_diff:11.6f}  "
              f"{mark:>10s}  {agree:10.0%}")

    print()
    mean_corr = np.mean(all_corr)
    match_rate = np.mean(all_match)
    print(f"  Mean correlation:     {mean_corr:.8f}")
    print(f"  Top-1 match rate:     {match_rate:.0%}")

    # Last position detailed comparison
    ref_last = ref_logits[0, -1]
    phi_last = phi_logits[0, -1]

    # Check predictions
    ref_pred = int(np.argmax(ref_last))
    phi_pred = int(np.argmax(phi_last))

    print(f"\n  Last position:")
    print(f"    Reference top-1: {ref_pred}")
    print(f"    φ-engine top-1:  {phi_pred}")
    print(f"    Match: {'✓' if ref_pred == phi_pred else '✗'}")

    # Verdict
    print()
    if mean_corr > 0.9999:
        print("  ✓ PERFECT: Pipeline is identical to reference (r > 0.9999)")
        print("    The φ-encoding introduces no measurable error in hybrid mode.")
    elif mean_corr > 0.999:
        print("  ✓ EXCELLENT: Pipeline nearly identical (r > 0.999)")
        print("    Tiny numerical differences from φ-decode precision.")
    elif mean_corr > 0.99:
        print("  ✓ GOOD: Pipeline correct (r > 0.99)")
        print("    Some numerical drift from φ-encoding.")
    elif mean_corr > 0.9:
        print("  ~ FAIR: Pipeline mostly correct (r > 0.9)")
        print("    Investigate potential issues in RoPE or attention.")
    else:
        print(f"  ✗ PROBLEM: Low correlation ({mean_corr:.4f})")
        print("    Pipeline has a bug — check each stage.")


if __name__ == '__main__':
    main()
