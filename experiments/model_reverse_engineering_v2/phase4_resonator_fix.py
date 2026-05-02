"""
Phase 4: Close the 5/6 → 6/6 Gap

Finding 44 showed the France failure is a routing issue (hard argmax picks
position 0 instead of 3), NOT a V/O issue. Two approaches known to work:

  A. Soft selection (weighted blend of positions) — Finding 40: 6/6
  B. phi_linear V/O direct — Finding 40: 6/6

This script:
  1. Diagnoses the exact routing scores for France (per-position d_k features)
  2. Tests soft selection + φ-quant VO
  3. Tests phi_linear V + φ-quant O directions
  4. Tests the full geometric pipeline: soft routing + φ-quant everything
  5. Identifies root cause of the hard argmax misroute
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

    # Extract WITH bias (like Finding 40) for comparison
    Wv_with_bias = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wk_with_bias = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq_with_bias = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    # Extract WITHOUT bias (clean matrices for φ-quantization)
    Wv_no_bias = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wk_no_bias = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq_no_bias = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]

        # With bias
        qo_b = phi_linear(attn.W_q, c, attn.b_q)[0].reshape(-1, num_heads, head_dim)
        ko_b = phi_linear(attn.W_k, c, attn.b_k)[0].reshape(-1, num_kv_heads, head_dim)
        vo_b = phi_linear(attn.W_v, c, attn.b_v)[0].reshape(-1, num_kv_heads, head_dim)
        Wq_with_bias[:, s:e] = qo_b[:, head_idx, :].T
        Wk_with_bias[:, s:e] = ko_b[:, kv_group, :].T
        Wv_with_bias[:, s:e] = vo_b[:, kv_group, :].T

        # Without bias
        qo = phi_linear(attn.W_q, c)[0].reshape(-1, num_heads, head_dim)
        ko = phi_linear(attn.W_k, c)[0].reshape(-1, num_kv_heads, head_dim)
        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        Wq_no_bias[:, s:e] = qo[:, head_idx, :].T
        Wk_no_bias[:, s:e] = ko[:, kv_group, :].T
        Wv_no_bias[:, s:e] = vo[:, kv_group, :].T

        if s % 1024 == 0: print(f"  {e}/{hidden_dim}...", flush=True)

    # V bias
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    if attn.b_v is not None:
        bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
        bv_full = bv_full.reshape(num_kv_heads, head_dim)
        bv_group = bv_full[kv_group]
    else:
        bv_group = np.zeros(head_dim, dtype=np.float32)

    # W_o for head 6
    h6in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h6in[0, 0, :] = 0.0
        h6in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h6in)[0, 0, :]

    # d_k from MESH SVD — two versions
    MESH_with_bias = Wq_with_bias @ Wk_with_bias.T
    U_b, S_b, Vt_b = np.linalg.svd(MESH_with_bias)
    d_k_with_bias = Wk_with_bias.T @ Vt_b[0, :]

    MESH_no_bias = Wq_no_bias @ Wk_no_bias.T
    U_nb, S_nb, Vt_nb = np.linalg.svd(MESH_no_bias)
    d_k_no_bias = Wk_no_bias.T @ Vt_nb[0, :]

    dk_sign = np.sign(d_k_no_bias)

    # VO SVD for φ-quantized version
    VO = Wo @ Wv_no_bias
    Uvo, Svo, Vtvo = np.linalg.svd(VO, full_matrices=False)
    S128 = Svo[:128]
    U_phi = phi_quant(Uvo[:, :128])
    Vt_phi = phi_quant(Vtvo[:128, :])
    S_phi = PHI ** np.round(np.log(S128 + 1e-20) / LOG_PHI)
    VO_phiq = (U_phi * S_phi[np.newaxis, :]) @ Vt_phi
    bias_out = Wo @ bv_group

    print(f"Weights extracted.", flush=True)
    print(f"  d_k (with bias) all neg: {(d_k_with_bias < 0).all()}")
    print(f"  d_k (no bias) all neg:   {(d_k_no_bias < 0).all()}")
    print(f"  cos(d_k_bias, d_k_nobias) = {np.dot(d_k_with_bias, d_k_no_bias)/(np.linalg.norm(d_k_with_bias)*np.linalg.norm(d_k_no_bias)):.6f}")

    # =========================================================================
    #   Part A: Diagnose the France routing — per-position d_k scores
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: France routing diagnosis")
    print("=" * 80)

    for prompt in prompts:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == target_layer:
                break
            h = lo(h)

        normed = rms_norm(h, attn.norm_weight)

        # Full attention routing
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        Q = Q.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        K_exp = np.repeat(K, heads_per_kv, axis=1)
        full_scores = Q[0, head_idx, -1, :] @ K_exp[0, head_idx, :, :].T
        full_argmax = int(np.argmax(full_scores))

        # d_k routing (with bias — Finding 40 style)
        dk_scores_b = normed[0] @ d_k_with_bias
        dk_argmax_b = int(np.argmax(dk_scores_b))

        # d_k routing (no bias — our φ-quant style)
        dk_scores_nb = normed[0] @ d_k_no_bias
        dk_argmax_nb = int(np.argmax(dk_scores_nb))

        # sign-only d_k routing
        dk_scores_sign = normed[0] @ dk_sign
        dk_argmax_sign = int(np.argmax(dk_scores_sign))

        is_france = 'France' in prompt
        marker = " *** FRANCE ***" if is_france else ""
        print(f"\n  \"{prompt}\"{marker}")
        print(f"    Full attention argmax: pos {full_argmax} = '{tokens[full_argmax]}'")
        print(f"    d_k (with bias):       pos {dk_argmax_b} = '{tokens[dk_argmax_b]}' {'✓' if dk_argmax_b == full_argmax else '✗'}")
        print(f"    d_k (no bias):         pos {dk_argmax_nb} = '{tokens[dk_argmax_nb]}' {'✓' if dk_argmax_nb == full_argmax else '✗'}")
        print(f"    sign d_k:              pos {dk_argmax_sign} = '{tokens[dk_argmax_sign]}' {'✓' if dk_argmax_sign == full_argmax else '✗'}")

        if is_france:
            print(f"\n    Per-position scores (France prompt):")
            print(f"    {'pos':>4s}  {'token':>10s}  {'full_attn':>10s}  {'dk_bias':>10s}  {'dk_nobias':>10s}  {'dk_sign':>10s}")
            for pos in range(len(tokens)):
                fa_marker = " ←full" if pos == full_argmax else ""
                db_marker = " ←dk_b" if pos == dk_argmax_b else ""
                nb_marker = " ←dk_nb" if pos == dk_argmax_nb else ""
                sg_marker = " ←sign" if pos == dk_argmax_sign else ""
                print(f"    {pos:4d}  {tokens[pos]:>10s}  {full_scores[pos]:+10.3f}  {dk_scores_b[pos]:+10.4f}"
                      f"  {dk_scores_nb[pos]:+10.4f}  {dk_scores_sign[pos]:+10.4f}"
                      f"{fa_marker}{db_marker}{nb_marker}{sg_marker}")
            # Gap between top-2
            sorted_scores_b = np.sort(dk_scores_b)[::-1]
            sorted_scores_nb = np.sort(dk_scores_nb)[::-1]
            print(f"\n    d_k (with bias) gap: {sorted_scores_b[0] - sorted_scores_b[1]:.6f}")
            print(f"    d_k (no bias) gap:   {sorted_scores_nb[0] - sorted_scores_nb[1]:.6f}")

    # =========================================================================
    #   Part B: Test function for all configurations
    # =========================================================================
    def run_config(label, routing_fn, vo_fn):
        """
        routing_fn(normed) -> (head_dim,) value vector to project through O
        vo_fn(v_head) -> (hidden_dim,) output contribution
        """
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
            v_head = routing_fn(normed)  # (head_dim,)
            attn_c = vo_fn(v_head)       # (hidden_dim,)

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
            if 'France' in prompt: fm = sm; fp = si == fi

        fs = "✓" if fp else "✗"
        ms = f"margin={fm:.3f}" if fm is not None else ""
        print(f"  {label:>60s}: {n_pass}/6  France={fs} {ms}", flush=True)
        return n_pass

    # =========================================================================
    #   Part C: Hard argmax routing — which d_k works?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Hard argmax with different d_k + different V/O")
    print("=" * 80)

    # Routing: hard argmax with d_k_with_bias (Finding 40 style)
    def hard_route_bias(normed):
        kf = normed[0] @ d_k_with_bias
        sp = int(np.argmax(kf))
        V = phi_linear(attn.W_v, normed[:, sp:sp+1, :], attn.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        return V[0, 0, kv_group, :]

    # Routing: hard argmax with d_k_no_bias
    def hard_route_nobias(normed):
        kf = normed[0] @ d_k_no_bias
        sp = int(np.argmax(kf))
        V = phi_linear(attn.W_v, normed[:, sp:sp+1, :], attn.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        return V[0, 0, kv_group, :]

    # Routing: hard argmax with sign d_k
    def hard_route_sign(normed):
        kf = normed[0] @ dk_sign
        sp = int(np.argmax(kf))
        V = phi_linear(attn.W_v, normed[:, sp:sp+1, :], attn.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        return V[0, 0, kv_group, :]

    print("\n  --- phi_linear V + extracted Wo ---")
    run_config("hard d_k(bias) + phi_linear V + Wo",
               hard_route_bias, lambda v: Wo @ v)
    run_config("hard d_k(nobias) + phi_linear V + Wo",
               hard_route_nobias, lambda v: Wo @ v)
    run_config("hard sign_dk + phi_linear V + Wo",
               hard_route_sign, lambda v: Wo @ v)

    # =========================================================================
    #   Part D: Soft selection — weighted blend
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Soft selection (weighted blend)")
    print("=" * 80)

    def make_soft_route(d_k_vec, temp):
        def soft_route(normed):
            kf = normed[0] @ d_k_vec  # (seq_len,)
            weights = phi_softmax(kf[np.newaxis, :] * temp, axis=-1)  # (1, seq_len)
            V = phi_linear(attn.W_v, normed, attn.b_v)
            V = V.reshape(1, -1, num_kv_heads, head_dim)
            V_group = V[0, :, kv_group, :]  # (seq_len, head_dim)
            return (weights[0, :, np.newaxis] * V_group).sum(axis=0)  # (head_dim,)
        return soft_route

    print("\n  --- Soft routing + extracted Wo ---")
    for temp in [1.0, 5.0, 10.0, 50.0, 100.0]:
        run_config(f"soft d_k(bias) T={temp:.0f} + phi_linear V + Wo",
                   make_soft_route(d_k_with_bias, temp), lambda v: Wo @ v)

    print()
    for temp in [1.0, 5.0, 10.0, 50.0, 100.0]:
        run_config(f"soft sign_dk T={temp:.0f} + phi_linear V + Wo",
                   make_soft_route(dk_sign, temp), lambda v: Wo @ v)

    # =========================================================================
    #   Part E: Soft selection + φ-quant VO (fully geometric)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Soft selection + φ-quant VO (fully geometric)")
    print("=" * 80)

    def make_soft_route_extracted(d_k_vec, temp, Wv_mat, bv):
        """Soft routing with extracted (non-phi_linear) V."""
        def soft_route(normed):
            kf = normed[0] @ d_k_vec  # (seq_len,)
            weights = phi_softmax(kf[np.newaxis, :] * temp, axis=-1)
            # Extracted V for each position: Wv @ normed + bias
            seq_len = normed.shape[1]
            V_all = np.zeros((seq_len, head_dim), dtype=np.float32)
            for pos in range(seq_len):
                V_all[pos, :] = Wv_mat @ normed[0, pos, :] + bv
            return (weights[0, :, np.newaxis] * V_all).sum(axis=0)
        return soft_route

    def make_soft_route_phiquant_vo(d_k_vec, temp, VO_mat, bias_o):
        """Soft routing with φ-quant VO applied directly."""
        def soft_route(normed):
            kf = normed[0] @ d_k_vec
            weights = phi_softmax(kf[np.newaxis, :] * temp, axis=-1)
            # VO directly on each position's hidden state
            seq_len = normed.shape[1]
            contributions = np.zeros((seq_len, hidden_dim), dtype=np.float32)
            for pos in range(seq_len):
                contributions[pos, :] = VO_mat @ normed[0, pos, :] + bias_o
            return (weights[0, :, np.newaxis] * contributions).sum(axis=0)
        return soft_route

    # φ-quant VO with soft routing — this is the FULLY geometric pipeline
    print("\n  --- Soft sign_dk + φ-quant VO (fully geometric) ---")
    for temp in [1.0, 5.0, 10.0, 50.0, 100.0]:
        run_config(f"soft sign_dk T={temp:.0f} + φ-quant VO",
                   make_soft_route_phiquant_vo(dk_sign, temp, VO_phiq, bias_out),
                   lambda v: v)  # VO already applied in routing_fn

    # φ-Zipf spectrum (S = S[0] × (i+1)^(-1/φ))
    ranks = np.arange(1, 129, dtype=np.float32)
    S_zipf = S128[0] * ranks ** (-1.0 / PHI)
    VO_zipf = (U_phi * S_zipf[np.newaxis, :].astype(np.float32)) @ Vt_phi

    print("\n  --- Soft sign_dk + φ-quant VO (φ-Zipf spectrum) ---")
    for temp in [1.0, 5.0, 10.0, 50.0, 100.0]:
        run_config(f"soft sign_dk T={temp:.0f} + φ-quant VO (φ-Zipf S)",
                   make_soft_route_phiquant_vo(dk_sign, temp, VO_zipf, bias_out),
                   lambda v: v)

    # =========================================================================
    #   Part F: Hybrid — phi_linear V + soft routing + φ-quant O directions
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part F: Hybrid phi_linear V + φ-quant O directions")
    print("=" * 80)

    # φ-quantize just Wo
    Wo_phi = phi_quant(Wo)

    def make_soft_philinear_phiO(d_k_vec, temp, Wo_mat):
        """Soft routing with phi_linear V and φ-quant Wo."""
        def soft_route(normed):
            kf = normed[0] @ d_k_vec
            weights = phi_softmax(kf[np.newaxis, :] * temp, axis=-1)
            V = phi_linear(attn.W_v, normed, attn.b_v)
            V = V.reshape(1, -1, num_kv_heads, head_dim)
            V_group = V[0, :, kv_group, :]
            v_head = (weights[0, :, np.newaxis] * V_group).sum(axis=0)
            return Wo_mat @ v_head
        return soft_route

    print("\n  --- Soft sign_dk + phi_linear V + φ-quant Wo ---")
    for temp in [1.0, 5.0, 10.0, 50.0]:
        run_config(f"soft sign_dk T={temp:.0f} + phi_linear V + φ-quant Wo",
                   make_soft_philinear_phiO(dk_sign, temp, Wo_phi),
                   lambda v: v)

    print("\n  --- Soft sign_dk + phi_linear V + full Wo ---")
    for temp in [1.0, 5.0, 10.0, 50.0]:
        run_config(f"soft sign_dk T={temp:.0f} + phi_linear V + full Wo",
                   make_soft_philinear_phiO(dk_sign, temp, Wo),
                   lambda v: v)

    # =========================================================================
    #   Part G: The temperature = what φ-power?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part G: Temperature as φ-power")
    print("=" * 80)

    # Finding 40 mentions φ-softmax with T = ln(φ). Let's test φ-power temps.
    phi_temps = [PHI**(-2), PHI**(-1), 1.0, PHI, PHI**2, PHI**3, PHI**4, PHI**5]
    print("\n  --- Soft sign_dk at φ-power temperatures + phi_linear V + Wo ---")
    for t in phi_temps:
        phi_lvl = np.log(t) / LOG_PHI
        run_config(f"soft sign_dk T=φ^{phi_lvl:.1f}={t:.3f} + phi_linear V + Wo",
                   make_soft_route(dk_sign, t), lambda v: Wo @ v)

    print("\n  --- Same but fully geometric (φ-quant VO) ---")
    for t in phi_temps:
        phi_lvl = np.log(t) / LOG_PHI
        run_config(f"soft sign_dk T=φ^{phi_lvl:.1f}={t:.3f} + φ-quant VO",
                   make_soft_route_phiquant_vo(dk_sign, t, VO_phiq, bias_out),
                   lambda v: v)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
