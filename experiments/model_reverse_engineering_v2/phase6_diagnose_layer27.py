"""
Phase 6: Diagnose the layer 27 cliff.

Run 27 layers in both float and integer to get the input to layer 27,
then step through layer 27 operation by operation to find the divergence.
"""

import sys, time
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
    EXP_MIN,
)
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

sys.path.insert(0, 'experiments/model_reverse_engineering_v2')
from phase6_integer_forward_pass import integer_forward_layer

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
ATTN_SCALE_EXP = int(round(PHI_GRID * np.log(1.0 / np.sqrt(128)) / LOG_PHI))


def corr(a, b):
    af, bf = a.flatten(), b.flatten()
    mask = np.isfinite(af) & np.isfinite(bf)
    if mask.sum() < 2:
        return float('nan')
    return float(np.corrcoef(af[mask], bf[mask])[0, 1])


def report(name, float_val, int_signs, int_exps):
    int_val = phi_to_float(int_signs, int_exps)
    fv = float_val.reshape(int_val.shape) if float_val.shape != int_val.shape else float_val
    c = corr(fv, int_val)
    
    # Check for extreme exponents
    min_e = int(np.min(int_exps))
    max_e = int(np.max(int_exps))
    
    # Check for clamped values
    n_clamped = int(np.sum(int_exps == EXP_MIN))
    pct_clamped = 100.0 * n_clamped / int_exps.size
    
    fv_range = f"[{float_val.min():.3e}, {float_val.max():.3e}]"
    print(f"  {name:40s}  corr={c:.8f}  exp=[{min_e},{max_e}]  clamped={pct_clamped:.1f}%  float_range={fv_range}")
    return c


def main():
    print("Diagnosing Layer 27 Cliff")
    print("=" * 120)

    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    rope_int = PhiRoPEInt(head_dim=128, rope_theta=1_000_000.0)

    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    tokens = tokenizer.encode("The capital of France is")

    # Run 27 layers to get input to layer 27
    print(f"\nRunning layers 0-26 (both float & integer)...")
    hidden_float = engine.embedding(tokens)[np.newaxis, :, :]
    h_s, h_e = float_to_phi(hidden_float[0])
    h_s = h_s[np.newaxis]; h_e = h_e[np.newaxis]

    float_out = hidden_float.copy()
    for i in range(27):
        float_out = engine.layers[i](float_out, pure=False)
        h_s, h_e = integer_forward_layer(engine.layers[i], h_s, h_e, rope_int, i)

    c = corr(float_out.flatten(), phi_to_float(h_s, h_e).flatten())
    print(f"  After layer 26: corr={c:.8f}")

    # Now step through layer 27 operation by operation
    print(f"\n{'='*120}")
    print(f"  LAYER 27 STEP-BY-STEP")
    print(f"{'='*120}")

    layer = engine.layers[27]
    attn = layer.attention
    mlp = layer.mlp
    num_heads, num_kv_heads, head_dim = 28, 4, 128
    heads_per_kv = num_heads // num_kv_heads
    batch, seq_len, hidden_dim = float_out.shape

    # Input stats
    print(f"\n  Input hidden: float range [{float_out.min():.3e}, {float_out.max():.3e}]")
    print(f"  Input hidden: int exp range [{h_e.min()}, {h_e.max()}]")
    print(f"  Input hidden: int clamped = {100*np.mean(h_e == EXP_MIN):.1f}%")

    # ─── ATTENTION: RMS NORM ───
    normed_float = rms_norm(float_out, attn.norm_weight)
    nw_s, nw_e = float_to_phi(attn.norm_weight)
    n_s, n_e = phi_rms_norm_int(h_s, h_e, nw_s, nw_e, hidden_dim)
    report("RMS norm (pre-attention)", normed_float, n_s, n_e)

    # ─── Q/K/V PROJECTIONS ───
    Q_float = phi_linear(attn.W_q, normed_float, attn.b_q)
    K_float = phi_linear(attn.W_k, normed_float, attn.b_k)
    V_float = phi_linear(attn.W_v, normed_float, attn.b_v)

    n_s_2d = n_s.reshape(-1, hidden_dim)
    n_e_2d = n_e.reshape(-1, hidden_dim)
    q_s, q_e = phi_matmul_integer(attn.W_q, n_s_2d, n_e_2d)
    k_s, k_e = phi_matmul_integer(attn.W_k, n_s_2d, n_e_2d)
    v_s, v_e = phi_matmul_integer(attn.W_v, n_s_2d, n_e_2d)

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

    report("Q (after bias)", Q_float.reshape(-1, Q_float.shape[-1]), q_s, q_e)
    report("K (after bias)", K_float.reshape(-1, K_float.shape[-1]), k_s, k_e)
    report("V (after bias)", V_float.reshape(-1, V_float.shape[-1]), v_s, v_e)

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

    # ─── GQA + SCORES + SOFTMAX ───
    K_exp_f = np.repeat(K_f, heads_per_kv, axis=1)
    V_exp_f = np.repeat(V_f, heads_per_kv, axis=1)
    k_s = np.repeat(k_s, heads_per_kv, axis=1)
    k_e = np.repeat(k_e, heads_per_kv, axis=1)
    v_s = np.repeat(v_s, heads_per_kv, axis=1)
    v_e = np.repeat(v_e, heads_per_kv, axis=1)

    scores_f = np.einsum('bhqd,bhkd->bhqk', Q_f, K_exp_f) * (1.0 / np.sqrt(128))
    score_s, score_e = phi_einsum_qk_int(q_s, q_e, k_s, k_e)
    score_s, score_e = phi_scale_int(score_s, score_e, ATTN_SCALE_EXP)
    report("Attention scores", scores_f, score_s, score_e)

    if seq_len > 1:
        causal = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        scores_f = scores_f + causal
        mask_idx = np.triu_indices(seq_len, k=1)
        for b in range(batch):
            for h in range(num_heads):
                score_s[b, h, mask_idx[0], mask_idx[1]] = np.int8(-1)
                score_e[b, h, mask_idx[0], mask_idx[1]] = np.int16(4000)

    attn_w_f = phi_softmax(scores_f, axis=-1)
    attn_ws, attn_we = phi_softmax_full_int(score_s, score_e, axis=-1)
    report("Softmax weights", attn_w_f, attn_ws, attn_we)

    # ─── VALUE AGG + OUTPUT PROJ + RESIDUAL ───
    attn_out_f = np.einsum('bhqk,bhkd->bhqd', attn_w_f, V_exp_f)
    out_s, out_e = phi_einsum_av_int(attn_ws, attn_we, v_s, v_e)
    report("Attention output", attn_out_f, out_s, out_e)

    attn_out_f_flat = attn_out_f.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    o_float = phi_linear(attn.W_o, attn_out_f_flat)

    out_s_flat = out_s.transpose(0, 2, 1, 3).reshape(batch * seq_len, -1)
    out_e_flat = out_e.transpose(0, 2, 1, 3).reshape(batch * seq_len, -1)
    ao_s, ao_e = phi_matmul_integer(attn.W_o, out_s_flat, out_e_flat)
    ao_s = ao_s.reshape(batch, seq_len, hidden_dim)
    ao_e = ao_e.reshape(batch, seq_len, hidden_dim)
    report("Output proj (W_o)", o_float, ao_s, ao_e)

    post_attn_f = float_out + o_float
    pa_s, pa_e = phi_add_encoded(h_s, h_e, ao_s, ao_e)
    report("Post-attention residual", post_attn_f, pa_s, pa_e)

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
    report("LAYER 27 OUTPUT", final_f, final_s, final_e)

    print("\nDone.")


if __name__ == '__main__':
    main()
