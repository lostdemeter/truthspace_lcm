"""
Phase 4: Close the Gap — d_k(bias) + φ-quant VO

The fix script revealed:
  cos(d_k_bias, d_k_nobias) = 0.005 — essentially ORTHOGONAL
  d_k(bias) + phi_linear V + Wo = 6/6 (margin=0.152)
  d_k(nobias) + anything = 5/6

This script tests the critical missing combination:
  d_k(bias) + φ-quant VO → does correct routing + geometric VO = 6/6?

Also tests:
  - sign(d_k_bias) = all -1s + φ-quant VO
  - φ-quant(d_k_bias) + φ-quant VO  
  - Fully geometric pipeline with all components
  - Why d_k(bias) vs d_k(nobias) are so different
"""

import sys, numpy as np, time, gc
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def finish_forward(engine, hidden_start, start_layer):
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    s = np.sort(logits[0, -1, :])[::-1]
    return idx, tok, s[0] - s[1]


def phi_quant(M):
    return np.sign(M) * PHI ** np.round(np.log(np.abs(M) + 1e-20) / LOG_PHI)


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    target_layer = 23
    head_idx = 6
    attn = engine.layers[target_layer].attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    kv_group = head_idx // heads_per_kv
    hidden_dim = engine.hidden_dim

    prompts = [
        'The capital of France is',
        'The largest ocean is the',
        'The color of grass is',
        'Barack Obama was the',
        'To be or not to',
        'Roses are red, violets are',
    ]

    # === Extract weights ===
    print("\nExtracting weights...", flush=True)
    I = np.eye(hidden_dim, dtype=np.float32)

    # WITH bias (correct d_k)
    Wk_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wv_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    # WITHOUT bias (for clean VO)
    Wv_nb = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wk_nb = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq_nb = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]
        qo_b = phi_linear(attn.W_q, c, attn.b_q)[0].reshape(-1, num_heads, head_dim)
        ko_b = phi_linear(attn.W_k, c, attn.b_k)[0].reshape(-1, num_kv_heads, head_dim)
        vo_b = phi_linear(attn.W_v, c, attn.b_v)[0].reshape(-1, num_kv_heads, head_dim)
        Wq_b[:, s:e] = qo_b[:, head_idx, :].T
        Wk_b[:, s:e] = ko_b[:, kv_group, :].T
        Wv_b[:, s:e] = vo_b[:, kv_group, :].T

        qo = phi_linear(attn.W_q, c)[0].reshape(-1, num_heads, head_dim)
        ko = phi_linear(attn.W_k, c)[0].reshape(-1, num_kv_heads, head_dim)
        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        Wq_nb[:, s:e] = qo[:, head_idx, :].T
        Wk_nb[:, s:e] = ko[:, kv_group, :].T
        Wv_nb[:, s:e] = vo[:, kv_group, :].T

        if s % 1024 == 0: print(f"  {e}/{hidden_dim}...", flush=True)

    # V bias
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
    bv_full = bv_full.reshape(num_kv_heads, head_dim)
    bv_group = bv_full[kv_group]

    # W_o for head 6
    h6in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h6in[0, 0, :] = 0.0
        h6in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h6in)[0, 0, :]

    # d_k variants
    MESH_b = Wq_b @ Wk_b.T
    U_b, S_b, Vt_b = np.linalg.svd(MESH_b)
    d_k_bias = Wk_b.T @ Vt_b[0, :]

    MESH_nb = Wq_nb @ Wk_nb.T
    U_nb, S_nb, Vt_nb = np.linalg.svd(MESH_nb)
    d_k_nobias = Wk_nb.T @ Vt_nb[0, :]

    dk_sign_bias = np.sign(d_k_bias)       # all -1s
    dk_sign_nobias = np.sign(d_k_nobias)   # mixed signs
    dk_phi_bias = phi_quant(d_k_bias)

    # VO matrices
    VO_full = Wo @ Wv_nb
    Uvo, Svo, Vtvo = np.linalg.svd(VO_full, full_matrices=False)
    S128 = Svo[:128]
    U_phi = phi_quant(Uvo[:, :128])
    Vt_phi = phi_quant(Vtvo[:128, :])
    S_phi = PHI ** np.round(np.log(S128 + 1e-20) / LOG_PHI)
    VO_phiq = (U_phi * S_phi[np.newaxis, :]) @ Vt_phi
    bias_out = Wo @ bv_group

    # Also build VO from bias-included Wv
    VO_full_b = Wo @ Wv_b
    Uvo_b, Svo_b, Vtvo_b = np.linalg.svd(VO_full_b, full_matrices=False)
    S128_b = Svo_b[:128]
    U_phi_b = phi_quant(Uvo_b[:, :128])
    Vt_phi_b = phi_quant(Vtvo_b[:128, :])
    S_phi_b = PHI ** np.round(np.log(S128_b + 1e-20) / LOG_PHI)
    VO_phiq_b = (U_phi_b * S_phi_b[np.newaxis, :]) @ Vt_phi_b

    print(f"Weights extracted.", flush=True)

    # =========================================================================
    #   Part A: Understand WHY d_k(bias) and d_k(nobias) differ
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: Why d_k(bias) ≠ d_k(nobias)")
    print("=" * 80)

    print(f"\n  d_k(bias):   all_neg={( d_k_bias < 0).all()}, ||d_k||={np.linalg.norm(d_k_bias):.4f}")
    print(f"  d_k(nobias): all_neg={(d_k_nobias < 0).all()}, ||d_k||={np.linalg.norm(d_k_nobias):.4f}")
    print(f"  cos(bias, nobias) = {np.dot(d_k_bias, d_k_nobias)/(np.linalg.norm(d_k_bias)*np.linalg.norm(d_k_nobias)):.6f}")

    # The bias absorbed into W_q and W_k changes the MESH fundamentally
    # MESH_bias = (W_q + b_q)^T @ (W_k + b_k)  vs  MESH_nobias = W_q^T @ W_k
    # The cross terms (b_q^T @ W_k + W_q^T @ b_k + b_q^T @ b_k) dominate
    print(f"\n  MESH(bias) S[0]={S_b[0]:.1f}, S[1]={S_b[1]:.4f}, ratio={S_b[0]/S_b[1]:.0f}:1")
    print(f"  MESH(nobias) S[0]={S_nb[0]:.1f}, S[1]={S_nb[1]:.4f}, ratio={S_nb[0]/S_nb[1]:.0f}:1")

    # What does the bias actually contribute?
    # Wq_b = Wq_nb + bias_contribution_q
    # The "with bias" extraction: phi_linear(W, I, b) = W @ I + b = W + b
    # So Wq_b[:, i] = Wq_nb[:, i] + bq_i  (bias added to each column)
    bias_contrib_q = Wq_b - Wq_nb  # (head_dim, hidden_dim)
    bias_contrib_k = Wk_b - Wk_nb
    print(f"\n  Bias contribution to Q: ||bias||={np.linalg.norm(bias_contrib_q):.4f}, ||Wq||={np.linalg.norm(Wq_nb):.4f}")
    print(f"  Bias contribution to K: ||bias||={np.linalg.norm(bias_contrib_k):.4f}, ||Wk||={np.linalg.norm(Wk_nb):.4f}")
    print(f"  Bias/Weight ratio Q: {np.linalg.norm(bias_contrib_q)/np.linalg.norm(Wq_nb)*100:.1f}%")
    print(f"  Bias/Weight ratio K: {np.linalg.norm(bias_contrib_k)/np.linalg.norm(Wk_nb)*100:.1f}%")

    # Is the bias constant across columns? (it should be — it's input-independent)
    # If phi_linear(W, e_i, b) = W @ e_i + b, the bias column is constant
    bq_col0 = bias_contrib_q[:, 0]
    bq_var = np.std([np.linalg.norm(bias_contrib_q[:, i] - bq_col0) for i in range(hidden_dim)])
    print(f"  Q bias column variation: {bq_var:.6f} (should be ~0 if constant)")

    # The bias is a rank-1 perturbation to MESH!
    # MESH_bias = Wq_nb @ Wk_nb.T + bq @ Wk_nb.T + Wq_nb @ bk.T + bq @ bk.T
    # where bq, bk are the constant bias vectors (replicated across columns)
    bq = bq_col0  # (head_dim,)
    bk = bias_contrib_k[:, 0]
    print(f"\n  Q bias vector: ||bq||={np.linalg.norm(bq):.4f}")
    print(f"  K bias vector: ||bk||={np.linalg.norm(bk):.4f}")

    # The MESH perturbation terms:
    # bq @ Wk_nb.T → rank-1 (outer product of bq with all k projections)
    # But wait — it's bq repeated for each input dim, so:
    # MESH_bias[i,j] = sum_d (Wq[i,d] + bq[i]) * (Wk[j,d] + bk[j])
    # = Wq[i,:] @ Wk[j,:].T + bq[i] * sum_d Wk[j,d] + Wq[i,:] @ bk * 1s + bq[i]*bk[j]*D
    # Actually let me think more carefully...
    # phi_linear(W_q, e_i, b_q) gives Q projection for input e_i
    # With bias: the output for input x is W_q @ x + b_q (broadcast)
    # So Wq_b extracted via identity = W_q + b_q repeated? No...
    # phi_linear(W_q, I[i:i+1], b_q) = W_q @ e_i + b_q
    # So column i of Wq_b = W_q[:, i] + b_q  (the bias is added to EVERY column)
    # Therefore: Wq_b = Wq_nb + bq @ ones(1, D)  (rank-1 update!)
    # Similarly: Wk_b = Wk_nb + bk @ ones(1, D)

    # MESH_b = Wq_b @ Wk_b.T
    #        = (Wq_nb + bq·1^T) @ (Wk_nb + bk·1^T)^T
    #        = Wq_nb @ Wk_nb^T + bq·1^T @ Wk_nb^T + Wq_nb @ 1·bk^T + bq·1^T @ 1·bk^T
    #        = MESH_nb + bq @ (Wk_nb @ 1)^T + (Wq_nb @ 1) @ bk^T + D * bq @ bk^T

    ones = np.ones(hidden_dim, dtype=np.float32)
    term1 = MESH_nb
    term2 = np.outer(bq, Wk_nb @ ones)   # bq × (Wk_nb summed over input dims)
    term3 = np.outer(Wq_nb @ ones, bk)   # (Wq_nb summed over input dims) × bk
    term4 = hidden_dim * np.outer(bq, bk) # D × bq × bk (rank-1)

    MESH_reconstructed = term1 + term2 + term3 + term4
    recon_error = np.linalg.norm(MESH_b - MESH_reconstructed)
    print(f"\n  MESH decomposition:")
    print(f"    ||MESH_nb||      = {np.linalg.norm(term1):.1f}")
    print(f"    ||bq×(Wk@1)^T|| = {np.linalg.norm(term2):.1f}")
    print(f"    ||(Wq@1)×bk^T|| = {np.linalg.norm(term3):.1f}")
    print(f"    ||D×bq×bk^T||   = {np.linalg.norm(term4):.1f}")
    print(f"    Reconstruction error: {recon_error:.6f}")
    print(f"    Bias terms / total: {(np.linalg.norm(term2) + np.linalg.norm(term3) + np.linalg.norm(term4))/np.linalg.norm(MESH_b)*100:.1f}%")

    # =========================================================================
    #   Part B: Test d_k(bias) with φ-quant VO — the critical test
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: d_k(bias) + φ-quant VO (the critical combination)")
    print("=" * 80)

    def run_test(label, d_k_vec, vo_fn):
        """Hard argmax routing + custom VO."""
        layer = engine.layers[target_layer]
        n_pass = 0; fm = None; fp = False

        for prompt in prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    full_out = lo(h.copy())
                    break
                h = lo(h)

            normed = rms_norm(h, attn.norm_weight)
            kf = normed[0] @ d_k_vec
            sp = int(np.argmax(kf))
            h_sel = normed[0, sp, :]

            attn_c = vo_fn(h_sel)

            pa = h.copy()
            pa[0, -1, :] += attn_c

            mlp = layer.mlp
            nm = rms_norm(pa, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mo = phi_linear(mlp.W_down, phi_silu(g) * u)
            so = pa + mo

            lf = finish_forward(engine, full_out, target_layer)
            ls = finish_forward(engine, so, target_layer)
            fi, ft, _ = get_top1(lf, tokenizer)
            si, st, sm = get_top1(ls, tokenizer)
            if si == fi: n_pass += 1
            if 'France' in prompt:
                fm = sm; fp = si == fi
                # Also track what was selected
                if not fp:
                    # Get correct token's logit
                    correct_logit = float(ls[0, -1, fi])
                    wrong_logit = float(ls[0, -1, si])

        fs = "✓" if fp else "✗"
        ms = f"margin={fm:.4f}" if fm is not None else ""
        print(f"  {label:>60s}: {n_pass}/6  France={fs} {ms}", flush=True)
        return n_pass, fp, fm

    # The critical tests: d_k(bias) + various VO
    print("\n  --- d_k(bias) routing + different VO ---", flush=True)
    run_test("d_k(bias) + full float32 VO + bias",
             d_k_bias, lambda h: VO_full @ h + bias_out)
    run_test("d_k(bias) + φ-quant VO + bias",
             d_k_bias, lambda h: VO_phiq @ h + bias_out)
    run_test("d_k(bias) + φ-quant VO(bias-included) [no sep bias]",
             d_k_bias, lambda h: VO_phiq_b @ h)
    run_test("d_k(bias) + phi_linear V + extracted Wo",
             d_k_bias, lambda h: Wo @ (Wv_nb @ h + bv_group))
    run_test("d_k(bias) + phi_linear V (direct)",
             d_k_bias,
             lambda h: Wo @ (phi_linear(attn.W_v,
                 h[np.newaxis, np.newaxis, :], attn.b_v).reshape(
                 num_kv_heads, head_dim)[kv_group]))

    # sign(d_k_bias) = all -1s
    print("\n  --- sign(d_k_bias) = all -1s + different VO ---", flush=True)
    run_test("sign(d_k_bias)=all-1s + full float32 VO + bias",
             dk_sign_bias, lambda h: VO_full @ h + bias_out)
    run_test("sign(d_k_bias)=all-1s + φ-quant VO + bias",
             dk_sign_bias, lambda h: VO_phiq @ h + bias_out)
    run_test("sign(d_k_bias)=all-1s + φ-quant VO(bias-incl)",
             dk_sign_bias, lambda h: VO_phiq_b @ h)
    run_test("sign(d_k_bias)=all-1s + phi_linear V + Wo",
             dk_sign_bias,
             lambda h: Wo @ (phi_linear(attn.W_v,
                 h[np.newaxis, np.newaxis, :], attn.b_v).reshape(
                 num_kv_heads, head_dim)[kv_group]))

    # φ-quant(d_k_bias)
    print("\n  --- φ-quant(d_k_bias) + different VO ---", flush=True)
    run_test("φ-quant(d_k_bias) + full float32 VO + bias",
             dk_phi_bias, lambda h: VO_full @ h + bias_out)
    run_test("φ-quant(d_k_bias) + φ-quant VO + bias",
             dk_phi_bias, lambda h: VO_phiq @ h + bias_out)
    run_test("φ-quant(d_k_bias) + φ-quant VO(bias-incl)",
             dk_phi_bias, lambda h: VO_phiq_b @ h)

    # =========================================================================
    #   Part C: Understand the bias absorption into VO
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Bias absorption analysis")
    print("=" * 80)

    # When we use d_k(bias), routing is correct. The remaining question is
    # whether V also needs bias.
    # The V bias creates a constant offset: for any input h,
    #   V_with_bias(h) = Wv @ h + bv
    #   O(V_with_bias(h)) = Wo @ (Wv @ h + bv) = Wo @ Wv @ h + Wo @ bv
    #                      = VO @ h + bias_out
    # So the V bias is just a constant additive term in the output.
    # Can this constant be absorbed into the φ-quant VO?

    # Compare VO_phiq_b (bias absorbed into Wv_b) vs VO_phiq + bias_out
    # For a given input h:
    # VO_phiq_b @ h  vs  VO_phiq @ h + bias_out
    # These are different because VO_phiq_b = φ-quant(Wo @ Wv_b) where Wv_b already has bias

    print(f"\n  V bias analysis:")
    print(f"    ||bv_group|| = {np.linalg.norm(bv_group):.6f}")
    print(f"    ||bias_out|| = ||Wo @ bv|| = {np.linalg.norm(bias_out):.6f}")
    print(f"    ||VO_full||  = {np.linalg.norm(VO_full):.4f}")
    print(f"    ||bias_out|| / ||VO_full|| = {np.linalg.norm(bias_out)/np.linalg.norm(VO_full)*100:.1f}%")

    # Is the bias itself φ-structured?
    bv_abs = np.abs(bv_group[bv_group != 0])
    if len(bv_abs) > 0:
        phi_lvl = np.log(bv_abs) / LOG_PHI
        phi_rnd = np.round(phi_lvl)
        resid = phi_lvl - phi_rnd
        print(f"    V bias φ-levels: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
        print(f"    V bias mean |φ-residual|: {np.abs(resid).mean():.4f}")

    bo_abs = np.abs(bias_out[bias_out != 0])
    if len(bo_abs) > 0:
        phi_lvl = np.log(bo_abs) / LOG_PHI
        phi_rnd = np.round(phi_lvl)
        resid = phi_lvl - phi_rnd
        print(f"    Output bias φ-levels: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
        print(f"    Output bias mean |φ-residual|: {np.abs(resid).mean():.4f}")

    # φ-quantize the bias
    bias_out_phi = phi_quant(bias_out)
    print(f"\n  --- d_k(bias) variants + φ-quant VO + φ-quant bias ---", flush=True)
    run_test("d_k(bias) + φ-quant VO + φ-quant bias",
             d_k_bias, lambda h: VO_phiq @ h + bias_out_phi)
    run_test("sign(d_k_bias) + φ-quant VO + φ-quant bias",
             dk_sign_bias, lambda h: VO_phiq @ h + bias_out_phi)
    run_test("φ-quant(d_k_bias) + φ-quant VO + φ-quant bias",
             dk_phi_bias, lambda h: VO_phiq @ h + bias_out_phi)

    # =========================================================================
    #   Part D: The fully geometric resonator — best config
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Fully geometric resonator configurations")
    print("=" * 80)

    configs = [
        # (label, d_k, vo_fn, description)
        ("BASELINE: phi_linear everything",
         d_k_bias,
         lambda h: Wo @ (phi_linear(attn.W_v,
             h[np.newaxis, np.newaxis, :], attn.b_v).reshape(
             num_kv_heads, head_dim)[kv_group]),
         "Reference: 6/6"),
        ("sign(d_k_bias) + phi_linear V + Wo",
         dk_sign_bias,
         lambda h: Wo @ (phi_linear(attn.W_v,
             h[np.newaxis, np.newaxis, :], attn.b_v).reshape(
             num_kv_heads, head_dim)[kv_group]),
         "Sign routing + phi_linear V/O"),
        ("sign(d_k_bias) + full VO + bias",
         dk_sign_bias,
         lambda h: VO_full @ h + bias_out,
         "Sign routing + extracted V/O"),
        ("sign(d_k_bias) + φ-quant VO + bias",
         dk_sign_bias,
         lambda h: VO_phiq @ h + bias_out,
         "Sign routing + φ-quant V/O"),
        ("sign(d_k_bias) + φ-quant VO + φ-quant bias",
         dk_sign_bias,
         lambda h: VO_phiq @ h + bias_out_phi,
         "FULLY GEOMETRIC (all φ)"),
        ("sign(d_k_bias) + φ-quant VO(bias-absorbed)",
         dk_sign_bias,
         lambda h: VO_phiq_b @ h,
         "FULLY GEOMETRIC (bias in VO)"),
    ]

    print(flush=True)
    for label, dk, vofn, desc in configs:
        n, fp, fm = run_test(label, dk, vofn)
        if fp: print(f"    ^^ {desc}: WORKS! ^^")

    # =========================================================================
    #   Part E: Parameter accounting for the winning config
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Parameter accounting")
    print("=" * 80)

    print(f"\n  Routing:")
    print(f"    d_k = sign(d_k_bias) = all -1s → 1 bit total")
    print(f"    Operation: argmax(-sum(h[pos])) per position")
    print(f"\n  V/O projection (if φ-quant works):")
    print(f"    U_φ: {128} × {hidden_dim} × 7 bits = {128*hidden_dim*7/8/1024:.1f} KB")
    print(f"    V_φ: {128} × {hidden_dim} × 7 bits = {128*hidden_dim*7/8/1024:.1f} KB")
    print(f"    S_φ: {128} × 6 bits = {128*6/8:.0f} bytes (or formula: 0 params)")
    print(f"    bias_out_φ: {hidden_dim} × 7 bits = {hidden_dim*7/8/1024:.1f} KB")
    total_bits = 1 + 2*128*hidden_dim*7 + hidden_dim*7
    print(f"\n  Total: {total_bits/8/1024:.1f} KB")
    print(f"  Full attention: {hidden_dim * head_dim * 4 * 4 / 1024:.1f} KB")
    print(f"  Compression: {hidden_dim * head_dim * 4 * 4 / (total_bits/8):.0f}×")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
