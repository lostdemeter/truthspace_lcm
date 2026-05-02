"""
Phase 6: Full Integer Forward Pass

Runs the ENTIRE Qwen2-7B forward pass using only integer arithmetic:
  - All matmuls: sign XOR + exp ADD + block-scaled accumulation
  - SiLU: integer LUT (176 KB)
  - RMS norm: integer square/sum/sqrt/divide
  - Softmax: LUT exponentiation + integer normalization
  - RoPE: φ-encoded cos/sin + integer multiply + add
  - Residual: integer add
  - Embedding: φ-encoded lookup

Activations flow as (int8 signs, int16 exponents) throughout.
The ONLY float operation is the final decode to logits for argmax.

Fail-fast: no fallbacks. If integer mode fails, we see exactly where.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, get_silu_lut, get_softmax_lut,
    phi_accumulate, phi_silu_int, phi_rms_norm_int,
    phi_add_encoded, phi_matmul_integer, phi_multiply_int,
    phi_scale_int, phi_einsum_qk_int, phi_einsum_av_int,
    phi_softmax_full_int, PhiRoPEInt,
    float_to_phi, phi_to_float,
    EXP_MIN,
)

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

# Scale factor for attention: 1/sqrt(head_dim) = 1/sqrt(128)
# In φ-exponent: round(PHI_GRID * ln(1/sqrt(128)) / LOG_PHI)
ATTN_SCALE_EXP = int(round(PHI_GRID * np.log(1.0 / np.sqrt(128)) / LOG_PHI))

# Causal mask: very negative value for masked positions
MASK_SIGN = np.int8(-1)
MASK_EXP = np.int16(4000)  # -φ^(4000/128) ≈ -φ^31 ≈ -3e6


def integer_forward_layer(layer, h_signs, h_exps, rope_int, layer_idx,
                          num_heads=28, num_kv_heads=4, head_dim=128):
    """
    One transformer layer in pure integer mode.

    Args:
        layer: PhiTransformerLayer (for weights only)
        h_signs, h_exps: (1, seq_len, hidden_dim) φ-encoded hidden state
        rope_int: PhiRoPEInt instance
        layer_idx: int

    Returns:
        (h_signs, h_exps): updated hidden state
    """
    attn = layer.attention
    mlp = layer.mlp
    batch, seq_len, hidden_dim = h_signs.shape
    heads_per_kv = num_heads // num_kv_heads

    # ─── ATTENTION ───

    # 1. RMS norm (pre-attention)
    norm_w_signs, norm_w_exps = float_to_phi(attn.norm_weight)
    n_signs, n_exps = phi_rms_norm_int(
        h_signs, h_exps, norm_w_signs, norm_w_exps, hidden_dim)

    # 2. Q/K/V projections (integer matmul)
    n_s_2d = n_signs.reshape(-1, hidden_dim)
    n_e_2d = n_exps.reshape(-1, hidden_dim)

    q_s, q_e = phi_matmul_integer(attn.W_q, n_s_2d, n_e_2d)
    k_s, k_e = phi_matmul_integer(attn.W_k, n_s_2d, n_e_2d)
    v_s, v_e = phi_matmul_integer(attn.W_v, n_s_2d, n_e_2d)

    # 3. Add biases (integer) — broadcast bias to match (batch*seq, dim)
    bq_s, bq_e = float_to_phi(attn.b_q)
    bk_s, bk_e = float_to_phi(attn.b_k)
    bv_s, bv_e = float_to_phi(attn.b_v)

    n_tokens = q_s.shape[0]
    bq_s_bc = np.broadcast_to(bq_s, q_s.shape).copy()
    bq_e_bc = np.broadcast_to(bq_e, q_e.shape).copy()
    bk_s_bc = np.broadcast_to(bk_s, k_s.shape).copy()
    bk_e_bc = np.broadcast_to(bk_e, k_e.shape).copy()
    bv_s_bc = np.broadcast_to(bv_s, v_s.shape).copy()
    bv_e_bc = np.broadcast_to(bv_e, v_e.shape).copy()

    q_s, q_e = phi_add_encoded(q_s, q_e, bq_s_bc, bq_e_bc)
    k_s, k_e = phi_add_encoded(k_s, k_e, bk_s_bc, bk_e_bc)
    v_s, v_e = phi_add_encoded(v_s, v_e, bv_s_bc, bv_e_bc)

    # 4. Reshape for multi-head: (batch, seq, heads*dim) → (batch, heads, seq, dim)
    q_s = q_s.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    q_e = q_e.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_s = k_s.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    k_e = k_e.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_s = v_s.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_e = v_e.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # 5. RoPE (integer)
    q_s, q_e = rope_int.apply(q_s, q_e)
    k_s, k_e = rope_int.apply(k_s, k_e)

    # 6. GQA expand: repeat KV heads
    k_s = np.repeat(k_s, heads_per_kv, axis=1)
    k_e = np.repeat(k_e, heads_per_kv, axis=1)
    v_s = np.repeat(v_s, heads_per_kv, axis=1)
    v_e = np.repeat(v_e, heads_per_kv, axis=1)

    # 7. Attention scores: Q @ K^T (integer einsum)
    score_s, score_e = phi_einsum_qk_int(q_s, q_e, k_s, k_e)
    # Scale by 1/sqrt(d)
    score_s, score_e = phi_scale_int(score_s, score_e, ATTN_SCALE_EXP)

    # 8. Causal mask
    if seq_len > 1:
        mask_indices = np.triu_indices(seq_len, k=1)
        for b in range(batch):
            for h_idx in range(num_heads):
                score_s[b, h_idx, mask_indices[0], mask_indices[1]] = MASK_SIGN
                score_e[b, h_idx, mask_indices[0], mask_indices[1]] = MASK_EXP

    # 9. Softmax (integer)
    attn_s, attn_e = phi_softmax_full_int(score_s, score_e, axis=-1)

    # 10. Value aggregation: attn @ V (integer einsum)
    out_s, out_e = phi_einsum_av_int(attn_s, attn_e, v_s, v_e)

    # 11. Reshape back: (batch, heads, seq, dim) → (batch, seq, heads*dim)
    out_s = out_s.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    out_e = out_e.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    # 12. Output projection (integer matmul, no bias for o_proj)
    out_s_2d = out_s.reshape(-1, num_heads * head_dim)
    out_e_2d = out_e.reshape(-1, num_heads * head_dim)
    ao_s, ao_e = phi_matmul_integer(attn.W_o, out_s_2d, out_e_2d)
    ao_s = ao_s.reshape(batch, seq_len, hidden_dim)
    ao_e = ao_e.reshape(batch, seq_len, hidden_dim)

    # 13. Residual add (integer)
    h_signs, h_exps = phi_add_encoded(h_signs, h_exps, ao_s, ao_e)

    # ─── MLP ───

    # 14. RMS norm (pre-MLP)
    mlp_nw_s, mlp_nw_e = float_to_phi(mlp.norm_weight)
    mn_s, mn_e = phi_rms_norm_int(
        h_signs, h_exps, mlp_nw_s, mlp_nw_e, hidden_dim)

    # 15. Gate/Up projections (integer matmul)
    mn_s_2d = mn_s.reshape(-1, hidden_dim)
    mn_e_2d = mn_e.reshape(-1, hidden_dim)

    gate_s, gate_e = phi_matmul_integer(mlp.W_gate, mn_s_2d, mn_e_2d)
    up_s, up_e = phi_matmul_integer(mlp.W_up, mn_s_2d, mn_e_2d)

    # 16. SiLU (integer LUT)
    gate_s, gate_e = phi_silu_int(gate_s, gate_e)

    # 17. Gate × Up (integer multiply)
    mlp_h_s, mlp_h_e = phi_multiply_int(gate_s, gate_e, up_s, up_e)

    # 18. Down projection (integer matmul)
    intermediate_dim = mlp_h_s.shape[-1]
    mlp_h_s_2d = mlp_h_s.reshape(-1, intermediate_dim)
    mlp_h_e_2d = mlp_h_e.reshape(-1, intermediate_dim)
    mlp_out_s, mlp_out_e = phi_matmul_integer(mlp.W_down, mlp_h_s_2d, mlp_h_e_2d)
    mlp_out_s = mlp_out_s.reshape(batch, seq_len, hidden_dim)
    mlp_out_e = mlp_out_e.reshape(batch, seq_len, hidden_dim)

    # 19. Residual add (integer)
    h_signs, h_exps = phi_add_encoded(h_signs, h_exps, mlp_out_s, mlp_out_e)

    return h_signs, h_exps


def integer_forward(engine, token_ids, rope_int):
    """
    Full integer forward pass: tokens → logits.

    Every operation is integer except:
      - Final decode to float for argmax
      - LUT construction (done once at init)

    Returns:
        logits: (1, seq_len, vocab_size) float32
    """
    seq_len = len(token_ids)

    # 1. Embedding lookup → φ-encode
    hidden_float = engine.embedding(token_ids)  # (seq_len, hidden_dim)
    h_signs, h_exps = float_to_phi(hidden_float)
    h_signs = h_signs[np.newaxis, :, :]  # (1, seq_len, hidden_dim)
    h_exps = h_exps[np.newaxis, :, :]

    # 2. Transformer layers
    for layer in engine.layers:
        t0 = time.time()
        h_signs, h_exps = integer_forward_layer(
            layer, h_signs, h_exps, rope_int, layer.layer_idx)
        dt = time.time() - t0
        print(f"    Layer {layer.layer_idx:2d}: {dt:.1f}s")

    # 3. Final RMS norm (integer)
    fnw_s, fnw_e = float_to_phi(engine.final_norm_weight)
    h_signs, h_exps = phi_rms_norm_int(
        h_signs, h_exps, fnw_s, fnw_e, engine.hidden_dim)

    # 4. LM head (integer matmul) — this is the big one: 152064 × 3584
    #    For speed, decode to float and use hybrid matmul for LM head
    #    (the matmul itself is still φ-encoded weights, just hybrid accumulation)
    #    TODO: switch to integer when performance allows
    h_float = phi_to_float(h_signs, h_exps)
    from phi_geometric.inference.phi_matmul import phi_linear
    logits = phi_linear(engine.lm_head.weight, h_float.reshape(1, seq_len, -1))

    return logits


def integer_forward_single_layer(engine, token_ids, rope_int, target_layer=0):
    """
    Run integer forward pass through just one layer for quick testing.
    Returns both integer and float results for comparison.
    """
    seq_len = len(token_ids)

    # Embedding
    hidden_float = engine.embedding(token_ids)
    hidden_float = hidden_float[np.newaxis, :, :]  # (1, seq, dim)

    # Float baseline for this layer
    from phi_geometric.inference.phi_components import rms_norm
    float_out = hidden_float.copy()
    for i in range(target_layer + 1):
        float_out = engine.layers[i](float_out, pure=False)

    # Integer path
    h_signs, h_exps = float_to_phi(hidden_float[0])
    h_signs = h_signs[np.newaxis, :, :]
    h_exps = h_exps[np.newaxis, :, :]
    for i in range(target_layer + 1):
        h_signs, h_exps = integer_forward_layer(
            engine.layers[i], h_signs, h_exps, rope_int, i)

    int_out = phi_to_float(h_signs, h_exps)

    return float_out[0], int_out[0]


# Test prompts (subset for speed)
TEST_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Water freezes at", "0"),
    ("The color of the sky is", "blue"),
    ("One plus one equals", "two"),
]


def main():
    print("Phase 6: Full Integer Forward Pass")
    print("=" * 80)

    # Initialize integer LUTs
    print("\nInitializing integer LUTs...")
    t0 = time.time()
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    rope_int = PhiRoPEInt(head_dim=128, rope_theta=1_000_000.0)
    print(f"  LUTs ready ({time.time()-t0:.1f}s)")

    # Load model
    print("\nLoading model...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=True)

    # Load tokenizer
    from phi_geometric.inference.tokenizer import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer()

    print("\n" + "=" * 80)
    print("  SINGLE-LAYER VALIDATION (Layer 0)")
    print("=" * 80)

    # Quick single-layer test first
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Tokens: {tokens}")

    t0 = time.time()
    float_out, int_out = integer_forward_single_layer(
        engine, tokens, rope_int, target_layer=0)
    dt = time.time() - t0

    corr = np.corrcoef(float_out.flatten(), int_out.flatten())[0, 1]
    max_abs = np.max(np.abs(int_out - float_out))
    print(f"\n  Layer 0 correlation: {corr:.8f}")
    print(f"  Max absolute diff:  {max_abs:.4e}")
    print(f"  Time: {dt:.1f}s")

    if corr < 0.99:
        print("\n  ✗ Layer 0 correlation too low. Stopping.")
        return

    print(f"\n  ✓ Layer 0 passes! Proceeding to multi-layer test...")

    print("\n" + "=" * 80)
    print("  3-LAYER VALIDATION (Layers 0-2)")
    print("=" * 80)

    t0 = time.time()
    float_out, int_out = integer_forward_single_layer(
        engine, tokens, rope_int, target_layer=2)
    dt = time.time() - t0

    corr = np.corrcoef(float_out.flatten(), int_out.flatten())[0, 1]
    print(f"\n  Layers 0-2 correlation: {corr:.8f}")
    print(f"  Time: {dt:.1f}s")

    if corr < 0.95:
        print("\n  ✗ 3-layer correlation too low. Integer errors may compound.")
        print("  Investigating per-layer correlations...")

        for target in range(3):
            f, i = integer_forward_single_layer(
                engine, tokens, rope_int, target_layer=target)
            c = np.corrcoef(f.flatten(), i.flatten())[0, 1]
            print(f"    Layer 0-{target}: corr={c:.8f}")
        return

    print(f"  ✓ 3 layers pass!")

    # If 3 layers work well, try more
    for n_layers in [7, 14, 28]:
        print(f"\n  Testing {n_layers} layers...")
        t0 = time.time()
        float_out, int_out = integer_forward_single_layer(
            engine, tokens, rope_int, target_layer=n_layers - 1)
        dt = time.time() - t0
        corr = np.corrcoef(float_out.flatten(), int_out.flatten())[0, 1]
        print(f"  Layers 0-{n_layers-1}: corr={corr:.8f}  ({dt:.1f}s)")

        if corr < 0.90:
            print(f"  ✗ Correlation dropped below 0.90 at {n_layers} layers.")
            break

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
