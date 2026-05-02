"""
Phase 6 Diagnostic: Find where integer precision is lost in Layer 0.

Runs each step of the integer forward pass side-by-side with float,
comparing correlation at every stage.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, get_silu_lut, get_softmax_lut,
    phi_accumulate, phi_silu_int, phi_rms_norm_int,
    phi_add_encoded, phi_matmul_integer, phi_multiply_int,
    phi_scale_int, phi_einsum_qk_int, phi_einsum_av_int,
    phi_softmax_full_int, PhiRoPEInt,
    float_to_phi, phi_to_float,
    FIXED_SCALE_BITS, EXP_MIN,
)

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
ATTN_SCALE_EXP = int(round(PHI_GRID * np.log(1.0 / np.sqrt(128)) / LOG_PHI))


def corr(a, b):
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def report(name, float_val, int_signs, int_exps):
    int_val = phi_to_float(int_signs, int_exps)
    # Match shapes
    fv = float_val.reshape(int_val.shape) if float_val.shape != int_val.shape else float_val
    c = corr(fv, int_val)
    max_rel = np.max(np.abs(int_val - fv) / (np.abs(fv) + 1e-10))
    mean_rel = np.mean(np.abs(int_val - fv) / (np.abs(fv) + 1e-10))
    print(f"  {name:40s}  corr={c:.8f}  mean_rel={mean_rel:.4e}  max_rel={max_rel:.4e}")
    return c


def main():
    print(f"Phase 6 Diagnostic: Integer Precision (FIXED_SCALE_BITS={FIXED_SCALE_BITS})")
    print("=" * 100)

    # Init LUTs
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    rope_int = PhiRoPEInt(head_dim=128, rope_theta=1_000_000.0)

    # Load model (only need layer 0)
    print("Loading model...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    from phi_geometric.inference.tokenizer import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer()

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}' → {len(tokens)} tokens\n")

    layer = engine.layers[0]
    attn = layer.attention
    mlp = layer.mlp

    num_heads, num_kv_heads, head_dim = 28, 4, 128
    heads_per_kv = num_heads // num_kv_heads

    # ─── EMBEDDING ───
    hidden_float = engine.embedding(tokens)[np.newaxis, :, :]  # (1, seq, 3584)
    h_s, h_e = float_to_phi(hidden_float[0])
    h_s = h_s[np.newaxis]; h_e = h_e[np.newaxis]
    report("Embedding (encode→decode roundtrip)", hidden_float, h_s, h_e)

    batch, seq_len, hidden_dim = hidden_float.shape

    # ─── ATTENTION: RMS NORM ───
    normed_float = rms_norm(hidden_float, attn.norm_weight)
    nw_s, nw_e = float_to_phi(attn.norm_weight)
    n_s, n_e = phi_rms_norm_int(h_s, h_e, nw_s, nw_e, hidden_dim)
    report("RMS norm (pre-attention)", normed_float, n_s, n_e)

    # ─── Q/K/V PROJECTIONS ───
    Q_float = phi_linear(attn.W_q, normed_float, attn.b_q)
    K_float = phi_linear(attn.W_k, normed_float, attn.b_k)
    V_float = phi_linear(attn.W_v, normed_float, attn.b_v)

    # Integer: matmul
    n_s_2d = n_s.reshape(-1, hidden_dim)
    n_e_2d = n_e.reshape(-1, hidden_dim)
    q_s, q_e = phi_matmul_integer(attn.W_q, n_s_2d, n_e_2d)
    k_s, k_e = phi_matmul_integer(attn.W_k, n_s_2d, n_e_2d)
    v_s, v_e = phi_matmul_integer(attn.W_v, n_s_2d, n_e_2d)

    report("Q projection (before bias)", phi_linear(attn.W_q, normed_float), q_s, q_e)

    # Bias add
    bq_s, bq_e = float_to_phi(attn.b_q)
    bk_s, bk_e = float_to_phi(attn.b_k)
    bv_s, bv_e = float_to_phi(attn.b_v)
    q_s, q_e = phi_add_encoded(q_s, q_e,
                                np.broadcast_to(bq_s, q_s.shape).copy(),
                                np.broadcast_to(bq_e, q_e.shape).copy())
    k_s, k_e = phi_add_encoded(k_s, k_e,
                                np.broadcast_to(bk_s, k_s.shape).copy(),
                                np.broadcast_to(bk_e, k_e.shape).copy())
    v_s, v_e = phi_add_encoded(v_s, v_e,
                                np.broadcast_to(bv_s, v_s.shape).copy(),
                                np.broadcast_to(bv_e, v_e.shape).copy())

    report("Q projection (after bias)", Q_float.reshape(-1, Q_float.shape[-1]), q_s, q_e)
    report("K projection (after bias)", K_float.reshape(-1, K_float.shape[-1]), k_s, k_e)
    report("V projection (after bias)", V_float.reshape(-1, V_float.shape[-1]), v_s, v_e)

    # ─── RESHAPE + ROPE ───
    Q_f = Q_float.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_f = K_float.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    V_f = V_float.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    q_s = q_s.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    q_e = q_e.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_s = k_s.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    k_e = k_e.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_s = v_s.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_e = v_e.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    Q_f = attn.rope.apply(Q_f)
    K_f = attn.rope.apply(K_f)

    q_s, q_e = rope_int.apply(q_s, q_e)
    k_s, k_e = rope_int.apply(k_s, k_e)

    report("Q after RoPE", Q_f, q_s, q_e)
    report("K after RoPE", K_f, k_s, k_e)

    # ─── GQA + ATTENTION SCORES ───
    K_exp_f = np.repeat(K_f, heads_per_kv, axis=1)
    V_exp_f = np.repeat(V_f, heads_per_kv, axis=1)
    k_s = np.repeat(k_s, heads_per_kv, axis=1)
    k_e = np.repeat(k_e, heads_per_kv, axis=1)
    v_s = np.repeat(v_s, heads_per_kv, axis=1)
    v_e = np.repeat(v_e, heads_per_kv, axis=1)

    scores_f = np.einsum('bhqd,bhkd->bhqk', Q_f, K_exp_f) * (1.0 / np.sqrt(128))
    score_s, score_e = phi_einsum_qk_int(q_s, q_e, k_s, k_e)
    score_s, score_e = phi_scale_int(score_s, score_e, ATTN_SCALE_EXP)

    report("Attention scores (Q@K^T/sqrt(d))", scores_f, score_s, score_e)

    # ─── CAUSAL MASK + SOFTMAX ───
    if seq_len > 1:
        causal = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        scores_f = scores_f + causal
        mask_idx = np.triu_indices(seq_len, k=1)
        for b in range(batch):
            for h in range(num_heads):
                score_s[b, h, mask_idx[0], mask_idx[1]] = np.int8(-1)
                score_e[b, h, mask_idx[0], mask_idx[1]] = np.int16(4000)

    attn_w_f = phi_softmax(scores_f, axis=-1)
    attn_s, attn_e = phi_softmax_full_int(score_s, score_e, axis=-1)

    report("Softmax weights", attn_w_f, attn_s, attn_e)

    # ─── VALUE AGGREGATION ───
    attn_out_f = np.einsum('bhqk,bhkd->bhqd', attn_w_f, V_exp_f)
    out_s, out_e = phi_einsum_av_int(attn_s, attn_e, v_s, v_e)

    report("Attention output (attn@V)", attn_out_f, out_s, out_e)

    # ─── OUTPUT PROJECTION + RESIDUAL ───
    attn_out_f_flat = attn_out_f.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    o_float = phi_linear(attn.W_o, attn_out_f_flat)
    post_attn_f = hidden_float + o_float

    out_s_flat = out_s.transpose(0, 2, 1, 3).reshape(batch * seq_len, -1)
    out_e_flat = out_e.transpose(0, 2, 1, 3).reshape(batch * seq_len, -1)
    ao_s, ao_e = phi_matmul_integer(attn.W_o, out_s_flat, out_e_flat)
    ao_s = ao_s.reshape(batch, seq_len, hidden_dim)
    ao_e = ao_e.reshape(batch, seq_len, hidden_dim)

    report("Output projection (W_o)", o_float, ao_s, ao_e)

    pa_s, pa_e = phi_add_encoded(h_s, h_e, ao_s, ao_e)
    report("Post-attention (hidden + attn_out)", post_attn_f, pa_s, pa_e)

    # ─── MLP ───
    mlp_normed_f = rms_norm(post_attn_f, mlp.norm_weight)
    mnw_s, mnw_e = float_to_phi(mlp.norm_weight)
    mn_s, mn_e = phi_rms_norm_int(pa_s, pa_e, mnw_s, mnw_e, hidden_dim)
    report("RMS norm (pre-MLP)", mlp_normed_f, mn_s, mn_e)

    gate_f = phi_linear(mlp.W_gate, mlp_normed_f)
    up_f = phi_linear(mlp.W_up, mlp_normed_f)

    mn_s_2d = mn_s.reshape(-1, hidden_dim)
    mn_e_2d = mn_e.reshape(-1, hidden_dim)
    gate_s, gate_e = phi_matmul_integer(mlp.W_gate, mn_s_2d, mn_e_2d)
    up_s, up_e = phi_matmul_integer(mlp.W_up, mn_s_2d, mn_e_2d)

    report("Gate projection", gate_f.reshape(-1, gate_f.shape[-1]), gate_s, gate_e)
    report("Up projection", up_f.reshape(-1, up_f.shape[-1]), up_s, up_e)

    silu_f = phi_silu(gate_f)
    gate_s, gate_e = phi_silu_int(gate_s, gate_e)
    report("SiLU(gate)", silu_f.reshape(-1, silu_f.shape[-1]), gate_s, gate_e)

    mlp_h_f = silu_f * up_f
    mlp_h_s, mlp_h_e = phi_multiply_int(gate_s, gate_e, up_s, up_e)
    report("SiLU(gate) * up", mlp_h_f.reshape(-1, mlp_h_f.shape[-1]), mlp_h_s, mlp_h_e)

    down_f = phi_linear(mlp.W_down, mlp_h_f)
    mlp_h_s_2d = mlp_h_s.reshape(-1, mlp_h_s.shape[-1])
    mlp_h_e_2d = mlp_h_e.reshape(-1, mlp_h_e.shape[-1])
    down_s, down_e = phi_matmul_integer(mlp.W_down, mlp_h_s_2d, mlp_h_e_2d)
    down_s = down_s.reshape(batch, seq_len, hidden_dim)
    down_e = down_e.reshape(batch, seq_len, hidden_dim)

    report("Down projection", down_f, down_s, down_e)

    final_f = post_attn_f + down_f
    final_s, final_e = phi_add_encoded(pa_s, pa_e, down_s, down_e)
    report("LAYER OUTPUT (hidden + mlp_out)", final_f, final_s, final_e)

    print("\nDone.")


if __name__ == '__main__':
    main()
