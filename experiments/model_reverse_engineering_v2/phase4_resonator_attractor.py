"""
Phase 4: Closing the 5/6 → 6/6 Gap with Geometric Attractors

The fully geometric resonator (φ-quant directions + formula spectrum) achieves
5/6 with France failing by only 0.018 logits. Can we close this gap?

Approaches:
  A. Characterize the France error precisely (what tokens compete, exact gap)
  B. Sweep the spectrum formula parameter c to find optimum
  C. Attractor correction: a small geometric nudge in VO output space
  D. Bias φ-tuning: adjust the V bias geometrically
  E. Direction refinement: can the φ-quantized directions be nudged?
"""

import sys, numpy as np, time, gc
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_silu
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


def get_topk(logits, tokenizer, k=5):
    """Return top-k tokens with their logits."""
    last = logits[0, -1, :]
    idx = np.argsort(last)[::-1][:k]
    results = []
    for i in idx:
        results.append((int(i), tokenizer.decode_token(int(i)), float(last[i])))
    return results


def run_resonator_detailed(engine, tokenizer, target_layer, d_k_vec,
                           vo_fn, prompts, label=""):
    """Test resonator with combined VO function. Returns per-prompt details."""
    layer = engine.layers[target_layer]
    attn = layer.attention
    results = []

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

        full_topk = get_topk(lf, tokenizer)
        simp_topk = get_topk(ls, tokenizer)

        fi = full_topk[0][0]
        si = simp_topk[0][0]
        match = si == fi

        # Find the correct token's rank and logit in simplified output
        correct_logit = float(ls[0, -1, fi])
        wrong_logit = float(ls[0, -1, si]) if not match else None
        gap = (correct_logit - simp_topk[0][2]) if not match else simp_topk[0][2] - simp_topk[1][2]

        results.append({
            'prompt': prompt,
            'match': match,
            'full_top': full_topk[0],
            'simp_top': simp_topk[0],
            'gap': gap,  # positive = correct wins, negative = wrong wins
            'h_sel': h_sel.copy(),
            'attn_c': attn_c.copy(),
            'h_pre_attn': h[0, -1, :].copy(),
            'pa': pa[0, -1, :].copy(),
            'selected_pos': sp,
        })

    n_pass = sum(1 for r in results if r['match'])
    france = [r for r in results if 'France' in r['prompt']][0]
    fs = "✓" if france['match'] else "✗"
    gs = f"gap={france['gap']:+.4f}"
    print(f"  {label:>55s}: {n_pass}/6  France={fs} {gs}", flush=True)
    return results


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

    # === Extract weights (bias-free) ===
    print("\nExtracting weights (bias-free)...", flush=True)
    I = np.eye(hidden_dim, dtype=np.float32)
    Wk = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wv = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]
        qo = phi_linear(attn.W_q, c)[0].reshape(-1, num_heads, head_dim)
        ko = phi_linear(attn.W_k, c)[0].reshape(-1, num_kv_heads, head_dim)
        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        Wq[:, s:e] = qo[:, head_idx, :].T
        Wk[:, s:e] = ko[:, kv_group, :].T
        Wv[:, s:e] = vo[:, kv_group, :].T
        if s % 1024 == 0: print(f"  {e}/{hidden_dim}...", flush=True)

    # Extract V bias
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    if attn.b_v is not None:
        bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
        bv_full = bv_full.reshape(num_kv_heads, head_dim)
        bv_group = bv_full[kv_group]
    else:
        bv_group = np.zeros(head_dim, dtype=np.float32)

    # Extract W_o for head 6
    h6in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h6in[0, 0, :] = 0.0
        h6in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h6in)[0, 0, :]

    # d_k from MESH SVD
    MESH = Wq @ Wk.T
    Um, Sm, Vtm = np.linalg.svd(MESH)
    d_k = Wk.T @ Vtm[0, :]
    dk_sign = np.sign(d_k)

    # VO SVD
    VO = Wo @ Wv
    Uvo, Svo, Vtvo = np.linalg.svd(VO, full_matrices=False)
    S128 = Svo[:128]
    bias_out = Wo @ bv_group  # pre-compute the output bias

    print(f"Weights extracted. VO SVD done.", flush=True)

    # =========================================================================
    #   Part A: Characterize the France error precisely
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: France error characterization")
    print("=" * 80)

    # Full float32 baseline
    print("\n  --- Full float32 VO ---", flush=True)
    res_full = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: VO @ h + bias_out,
        prompts, "Full VO (float32)")

    france_full = [r for r in res_full if 'France' in r['prompt']][0]
    print(f"\n  France prompt details (full VO):")
    print(f"    Selected position: {france_full['selected_pos']}")
    print(f"    Full model top-1: '{france_full['full_top'][1]}' (logit={france_full['full_top'][2]:.4f})")
    print(f"    Simplified top-1: '{france_full['simp_top'][1]}' (logit={france_full['simp_top'][2]:.4f})")
    print(f"    Gap (correct - wrong): {france_full['gap']:+.4f}")

    # φ-quantized baseline
    def phi_quant(M):
        return np.sign(M) * PHI ** np.round(np.log(np.abs(M) + 1e-20) / LOG_PHI)

    U_phi = phi_quant(Uvo[:, :128])
    Vt_phi = phi_quant(Vtvo[:128, :])
    S_phi = PHI ** np.round(np.log(S128 + 1e-20) / LOG_PHI)

    VO_phiq = (U_phi * S_phi[np.newaxis, :]) @ Vt_phi

    print("\n  --- φ-quantized VO ---", flush=True)
    res_phiq = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: VO_phiq @ h + bias_out,
        prompts, "φ-quant U,V,S")

    france_phiq = [r for r in res_phiq if 'France' in r['prompt']][0]
    print(f"\n  France prompt details (φ-quant):")
    print(f"    Full model top-1: '{france_phiq['full_top'][1]}' (logit={france_phiq['full_top'][2]:.4f})")
    print(f"    Simplified top-1: '{france_phiq['simp_top'][1]}' (logit={france_phiq['simp_top'][2]:.4f})")
    print(f"    Gap: {france_phiq['gap']:+.4f}")

    # =========================================================================
    #   Part B: Spectrum formula sweep — optimize the single constant
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Spectrum formula optimization")
    print("=" * 80)

    # φ-Zipf: S[i] = c * (i+1)^(-1/φ)
    # Try different c values and α values
    ranks = np.arange(1, 129, dtype=np.float32)

    print("\n  Sweeping α in S[i] = S[0] × (i+1)^(-α):", flush=True)
    best_gap = -999
    best_alpha = None
    for alpha_val in [0.0, 0.1, 0.2, 0.3, 1.0/PHI, 0.7, 0.8, 1.0, 1.0/PHI**2, 2.0/PHI]:
        S_test = S128[0] * ranks ** (-alpha_val)
        VO_test = (U_phi * S_test[np.newaxis, :].astype(np.float32)) @ Vt_phi
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, vo=VO_test: vo @ h + bias_out,
            prompts, f"α={alpha_val:.4f}")
        fr = [r for r in res if 'France' in r['prompt']][0]
        if fr['gap'] > best_gap:
            best_gap = fr['gap']
            best_alpha = alpha_val

    print(f"\n  Best α = {best_alpha:.4f}, France gap = {best_gap:+.4f}")

    # Fine-tune around best α
    print(f"\n  Fine-tuning around α={best_alpha:.4f}:", flush=True)
    for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
        alpha_val = best_alpha + delta
        if alpha_val < 0: continue
        S_test = S128[0] * ranks ** (-alpha_val)
        VO_test = (U_phi * S_test[np.newaxis, :].astype(np.float32)) @ Vt_phi
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, vo=VO_test: vo @ h + bias_out,
            prompts, f"α={alpha_val:.4f}")
        fr = [r for r in res if 'France' in r['prompt']][0]
        if fr['gap'] > best_gap:
            best_gap = fr['gap']
            best_alpha = alpha_val

    print(f"\n  Final best α = {best_alpha:.4f}, France gap = {best_gap:+.4f}")

    # Also sweep the overall scale c
    print(f"\n  Sweeping overall scale c (with best α={best_alpha:.4f}):", flush=True)
    S_base = S128[0] * ranks ** (-best_alpha)
    for c_mult in [0.5, 0.618, 0.8, 0.9, 1.0, 1.1, 1.2, 1/0.618, 1.618, 2.0]:
        S_test = (c_mult * S_base).astype(np.float32)
        VO_test = (U_phi * S_test[np.newaxis, :]) @ Vt_phi
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, vo=VO_test: vo @ h + bias_out,
            prompts, f"c={c_mult:.3f}")
        fr = [r for r in res if 'France' in r['prompt']][0]
        if fr['gap'] > best_gap:
            best_gap = fr['gap']

    # =========================================================================
    #   Part C: Attractor correction — geometric nudge in output space
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Attractor correction")
    print("=" * 80)

    # Compute the error vector for France between full VO and φ-quant VO
    france_h_sel = france_full['h_sel']
    attn_c_full = VO @ france_h_sel + bias_out
    attn_c_phiq = VO_phiq @ france_h_sel + bias_out
    error_vec = attn_c_full - attn_c_phiq

    print(f"\n  Error vector (full VO - φ-quant VO) for France:")
    print(f"    ||error|| = {np.linalg.norm(error_vec):.6f}")
    print(f"    ||attn_c_full|| = {np.linalg.norm(attn_c_full):.6f}")
    print(f"    Relative error: {np.linalg.norm(error_vec)/np.linalg.norm(attn_c_full)*100:.2f}%")

    # Is the error itself φ-structured?
    err_abs = np.abs(error_vec)
    nz = err_abs[err_abs > 1e-12]
    if len(nz) > 0:
        phi_lvl = np.log(nz) / LOG_PHI
        phi_rnd = np.round(phi_lvl)
        resid = phi_lvl - phi_rnd
        print(f"    Error φ-level range: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
        print(f"    Mean |residual|: {np.abs(resid).mean():.4f}")

    # Approach C1: Add the error vector as a fixed correction
    print("\n  --- C1: Fixed error correction ---", flush=True)
    res = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: VO_phiq @ h + bias_out + error_vec,
        prompts, "φ-quant VO + France error vec")

    # Approach C2: Project error onto VO's singular directions
    # This tells us WHICH directions need correction
    error_in_svd = Uvo[:, :128].T @ error_vec  # project onto VO output directions
    print(f"\n  Error projected onto VO singular directions:")
    print(f"    ||error in SVD basis|| = {np.linalg.norm(error_in_svd):.6f}")
    top_err_idx = np.argsort(np.abs(error_in_svd))[::-1][:10]
    for i, idx in enumerate(top_err_idx):
        print(f"    SVD dir {idx:3d}: error component = {error_in_svd[idx]:+.6f}, S[{idx}] = {S128[idx]:.4f}")

    # Approach C3: Attractor = error projected onto d_k direction
    # Since d_k is the routing direction, an attractor along d_k is geometric
    dk_normed = dk_sign / np.linalg.norm(dk_sign)
    error_along_dk = np.dot(error_vec, dk_normed) * dk_normed
    print(f"\n  Error component along d_k: {np.dot(error_vec, dk_normed):.6f}")
    print(f"    (= {np.dot(error_vec, dk_normed)/np.linalg.norm(error_vec)*100:.1f}% of total error)")

    res = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: VO_phiq @ h + bias_out + error_along_dk,
        prompts, "φ-quant VO + error along d_k")

    # Approach C4: Attractor along each of the top VO singular directions
    print("\n  --- C4: Attractor along top VO singular directions ---", flush=True)
    for n_dirs in [1, 2, 3, 5, 10, 20, 50, 128]:
        correction = np.zeros(hidden_dim, dtype=np.float32)
        for idx in top_err_idx[:n_dirs]:
            correction += error_in_svd[idx] * Uvo[:, idx]
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, c=correction: VO_phiq @ h + bias_out + c,
            prompts, f"φ-quant VO + top-{n_dirs} SVD error dirs")

    # =========================================================================
    #   Part D: Bias φ-tuning
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Bias φ-tuning")
    print("=" * 80)

    # Is the bias itself φ-structured?
    bv_abs = np.abs(bv_group)
    nz = bv_abs[bv_abs > 1e-12]
    phi_lvl = np.log(nz) / LOG_PHI
    phi_rnd = np.round(phi_lvl)
    resid = phi_lvl - phi_rnd
    print(f"\n  V bias φ-structure:")
    print(f"    ||bv|| = {np.linalg.norm(bv_group):.4f}")
    print(f"    φ-level range: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
    print(f"    Mean |residual|: {np.abs(resid).mean():.4f}")
    print(f"    Within 0.1 of int: {(np.abs(resid)<0.1).mean()*100:.1f}%")

    # φ-quantize the bias
    bv_phi = np.sign(bv_group) * PHI ** np.round(np.log(np.abs(bv_group) + 1e-20) / LOG_PHI)
    bias_out_phi = Wo @ bv_phi

    print("\n  --- φ-quantized bias ---", flush=True)
    res = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: VO_phiq @ h + bias_out_phi,
        prompts, "φ-quant VO + φ-quant bias")

    # Scale the bias by φ-powers
    print("\n  --- Bias scale sweep ---", flush=True)
    for scale in [1/PHI**2, 1/PHI, 1.0, PHI, PHI**2]:
        phi_lvl_s = np.log(scale) / LOG_PHI
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, s=scale: VO_phiq @ h + s * bias_out,
            prompts, f"φ-quant VO + bias×φ^{phi_lvl_s:.2f}")

    # =========================================================================
    #   Part E: The nuclear option — use phi_linear internally for V/O
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: phi_linear internal path (ground truth)")
    print("=" * 80)

    # This is what gets 6/6 with margin=0.152 — the phi_linear internal path
    # Let's confirm and measure the exact gap
    def vo_phi_linear(h_sel):
        """Use phi_linear for V projection, then extracted Wo for O."""
        h_in = h_sel[np.newaxis, np.newaxis, :]  # (1, 1, hidden)
        V = phi_linear(attn.W_v, h_in, attn.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        vs = V[0, 0, kv_group, :]
        return Wo @ vs

    print("\n  --- phi_linear V + extracted Wo ---", flush=True)
    res_pli = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        vo_phi_linear,
        prompts, "phi_linear V + extracted Wo")

    france_pli = [r for r in res_pli if 'France' in r['prompt']][0]
    print(f"\n  France with phi_linear V:")
    print(f"    Top-1: '{france_pli['simp_top'][1]}' gap={france_pli['gap']:+.4f}")

    # What's the difference between phi_linear V and extracted Wv?
    attn_c_pli = vo_phi_linear(france_h_sel)
    diff_pli = attn_c_pli - attn_c_full
    print(f"    ||phi_linear - extracted|| = {np.linalg.norm(diff_pli):.6f}")
    print(f"    ||phi_linear - φ_quant||   = {np.linalg.norm(attn_c_pli - attn_c_phiq):.6f}")

    # Can we capture this phi_linear "magic" as a geometric correction?
    # The difference is the quantization residual from phi_linear's internal representation
    diff_phi_levels = np.abs(diff_pli)
    nz = diff_phi_levels[diff_phi_levels > 1e-12]
    if len(nz) > 0:
        pl = np.log(nz) / LOG_PHI
        pr = np.round(pl)
        re = pl - pr
        print(f"    Diff φ-level range: [{pl.min():.2f}, {pl.max():.2f}]")
        print(f"    Mean |residual|: {np.abs(re).mean():.4f}")

    # Apply the phi_linear correction to the φ-quant pipeline
    pli_correction = attn_c_pli - attn_c_phiq
    print(f"\n  --- φ-quant VO + phi_linear correction ---", flush=True)
    res = run_resonator_detailed(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: VO_phiq @ h + bias_out + pli_correction,
        prompts, "φ-quant VO + phi_linear correction (France-specific)")

    # =========================================================================
    #   Part F: The attractor idea — can we derive the correction geometrically?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part F: Geometric attractor derivation")
    print("=" * 80)

    # The φ-quantization error per SVD direction should be predictable:
    # Each direction was φ-quantized independently.
    # The reconstruction error = (real_U × real_S × real_V - phi_U × phi_S × phi_V) @ h
    # This is a DETERMINISTIC function of h — not content-dependent in an unpredictable way.

    # Compute the VO error matrix once
    VO_error = VO - VO_phiq  # (hidden, hidden)
    print(f"\n  VO quantization error matrix:")
    print(f"    ||VO_error|| = {np.linalg.norm(VO_error):.6f}")
    print(f"    ||VO||       = {np.linalg.norm(VO):.6f}")
    print(f"    Relative:     {np.linalg.norm(VO_error)/np.linalg.norm(VO)*100:.2f}%")

    # SVD of the error matrix itself — what rank is the correction?
    Ue, Se, Vte = np.linalg.svd(VO_error, full_matrices=False)
    Se128 = Se[:128]
    print(f"\n  VO error SVD:")
    print(f"    S[0] = {Se128[0]:.6f}")
    print(f"    S[0]/S[1] = {Se128[0]/Se128[1]:.2f}")
    eff_rank = (Se128 > Se128[0] * 0.01).sum()
    print(f"    Effective rank (S > 1% of S[0]): {eff_rank}")

    # Energy in top-k directions
    cumvar = np.cumsum(Se128**2) / np.sum(Se128**2)
    for k in [1, 2, 5, 10, 20, 50]:
        print(f"    Top-{k:2d} captures {cumvar[k-1]*100:.1f}% of error energy")

    # Test: add rank-k correction from error SVD
    print(f"\n  --- Rank-k error correction (from error SVD) ---", flush=True)
    for k in [1, 2, 5, 10, 20, 50, 128]:
        VO_corrected = VO_phiq + (Ue[:, :k] * Se[:k][np.newaxis, :]) @ Vte[:k, :]
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, vo=VO_corrected: vo @ h + bias_out,
            prompts, f"φ-quant VO + rank-{k} error correction")

    # Can the error correction itself be φ-quantized?
    print(f"\n  --- φ-quantized error correction ---", flush=True)
    for k in [5, 10, 20, 50]:
        Ue_phi = phi_quant(Ue[:, :k])
        Vte_phi = phi_quant(Vte[:k, :])
        Se_phi = PHI ** np.round(np.log(Se[:k] + 1e-20) / LOG_PHI)
        VO_corrected = VO_phiq + (Ue_phi * Se_phi[np.newaxis, :]) @ Vte_phi
        res = run_resonator_detailed(
            engine, tokenizer, target_layer, dk_sign,
            lambda h, vo=VO_corrected: vo @ h + bias_out,
            prompts, f"φ-quant VO + φ-quant rank-{k} correction")

    # =========================================================================
    #   Part G: Summary — what's the minimal geometric intervention for 6/6?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part G: Parameter count for correction approaches")
    print("=" * 80)

    configs = [
        ("φ-quant VO (no correction)", 0),
        ("+ rank-1 error (2 vecs × 3584 × 32b + 1 scalar)", 2 * 3584 * 32 + 32),
        ("+ rank-5 error (2 × 5 vecs + 5 scalars)", 2 * 5 * 3584 * 32 + 5 * 32),
        ("+ rank-5 φ-quant error (2 × 5 vecs × 7b + 5 × 6b)", 2 * 5 * 3584 * 7 + 5 * 6),
        ("+ rank-10 φ-quant error", 2 * 10 * 3584 * 7 + 10 * 6),
        ("+ single bias scale (1 float)", 32),
    ]

    base_bits = (hidden_dim * 128 * 7) * 2 + 128 * 6  # φ-quant U,V + φ-quant S
    for name, extra_bits in configs:
        total = base_bits + extra_bits
        kb = total / 8 / 1024
        print(f"    {name:>55s}: +{extra_bits:>8d} bits = {kb:.1f} KB total")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
