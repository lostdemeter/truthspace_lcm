"""
Phase 4: V/O as Geometric Downcasting

Finding 42 concluded V/O weights are "NOT on the φ-lattice" — but that analysis
looked at element-level φ-residuals. This was the WRONG level of analysis.

From Doc 209: "Attention IS dimensional downcasting."
From Doc 152: MLP weights = sign × φ^level at 97.5% correlation.
From Doc 144: The zeta critical line is the balance point; attention finds it.

The V/O combined projection is a rank-128 operation:
  VO = W_o @ W_v  (3584×3584, rank 128)

This IS the downcasting lens. The right questions are:
  A. Does the SVD spectrum of VO follow φ-Zipf (S[i] ∝ i^(-1/φ))?
  B. Do the singular vectors align with known geometric structures (d_k, etc.)?
  C. Can we reconstruct VO from φ-geometric components?
  D. Does the VO projection have zeta-like symmetry structure?
  E. Do the INDIVIDUAL Wv/Wo matrices have φ-lattice structure in their SVD?
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


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    s = np.sort(logits[0, -1, :])[::-1]
    return idx, tok, s[0] - s[1]


def run_resonator_custom(engine, tokenizer, target_layer, d_k_vec,
                         v_fn, o_fn, prompts, label=""):
    """Test resonator with custom V/O functions."""
    layer = engine.layers[target_layer]
    attn = layer.attention
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

        v_out = v_fn(h_sel)
        attn_c = o_fn(v_out)

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
    print(f"  {label:>55s}: {n_pass}/6  France={fs} {ms}", flush=True)
    return n_pass


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

    print(f"\nWv: {Wv.shape}, Wo: {Wo.shape}")
    print(f"V bias norm: {np.linalg.norm(bv_group):.4f}")

    # =========================================================================
    #   Part A: SVD of VO — Is the spectrum φ-Zipf?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: SVD spectrum of V/O — Is it φ-Zipf?")
    print("=" * 80)

    VO = Wo @ Wv  # (hidden_dim, hidden_dim), rank 128
    Uvo, Svo, Vtvo = np.linalg.svd(VO, full_matrices=False)
    # Only first 128 are non-trivial
    S128 = Svo[:128]
    print(f"\n  VO SVD: top-10 singular values:")
    for i in range(10):
        print(f"    S[{i:3d}] = {S128[i]:.6f}")
    print(f"    S[127] = {S128[127]:.6f}")
    print(f"    S[0]/S[127] = {S128[0]/S128[127]:.1f}")
    print(f"    Effective rank (S > S[0]*0.01): {(S128 > S128[0]*0.01).sum()}")

    # Fit φ-Zipf: S[i] ∝ (i+1)^(-alpha)
    log_rank = np.log(np.arange(1, 129))
    log_S = np.log(S128 + 1e-20)
    # Linear fit in log-log space
    A = np.vstack([log_rank, np.ones(128)]).T
    alpha_fit, log_c = np.linalg.lstsq(A, log_S, rcond=None)[0]
    alpha = -alpha_fit  # negate because it's a decay
    r_zipf = np.corrcoef(log_rank, log_S)[0, 1]

    print(f"\n  Zipf fit: S[i] ∝ i^(-α)")
    print(f"    α = {alpha:.4f}")
    print(f"    1/φ = {1/PHI:.4f}")
    print(f"    |α - 1/φ| = {abs(alpha - 1/PHI):.4f}")
    print(f"    log-log R² = {r_zipf**2:.4f}")

    # Also check power-of-φ decay: S[i] = S[0] × φ^(-β×i)
    ratios = S128[1:] / S128[:-1]
    mean_ratio = np.mean(ratios)
    log_ratio = np.log(mean_ratio) / LOG_PHI
    print(f"\n  Geometric decay check:")
    print(f"    Mean S[i+1]/S[i] = {mean_ratio:.6f}")
    print(f"    = φ^{log_ratio:.4f}")
    print(f"    Std of ratio: {np.std(ratios):.6f}")

    # Check if singular values themselves sit on φ-levels
    sv_phi_levels = np.log(S128) / LOG_PHI
    sv_phi_rnd = np.round(sv_phi_levels)
    sv_resid = sv_phi_levels - sv_phi_rnd
    print(f"\n  Singular values on φ-lattice:")
    print(f"    φ-level range: [{sv_phi_levels[0]:.2f}, {sv_phi_levels[-1]:.2f}]")
    print(f"    Mean |residual| from int φ-level: {np.abs(sv_resid).mean():.4f}")
    print(f"    Within 0.1 of int: {(np.abs(sv_resid)<0.1).mean()*100:.1f}%")
    for i in range(0, 128, 16):
        print(f"    S[{i:3d}] = {S128[i]:.6f}  φ-level = {sv_phi_levels[i]:.3f}"
              f"  resid = {sv_resid[i]:+.3f}")

    # =========================================================================
    #   Part B: Do the VO singular vectors align with d_k?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Singular vector alignment with d_k and embedding geometry")
    print("=" * 80)

    # Left singular vectors of VO (output directions)
    # Right singular vectors of VO (input directions)
    d_k_normed = d_k / np.linalg.norm(d_k)
    dk_sign_normed = dk_sign / np.linalg.norm(dk_sign)

    print("\n  cos(d_k, VO output directions U[:, i]):")
    for i in range(min(10, 128)):
        c = abs(np.dot(d_k_normed, Uvo[:, i]))
        print(f"    cos(d_k, U[{i}]) = {c:.6f}")

    print("\n  cos(d_k, VO input directions V[i, :]):")
    for i in range(min(10, 128)):
        c = abs(np.dot(d_k_normed, Vtvo[i, :]))
        print(f"    cos(d_k, V[{i}]) = {c:.6f}")

    # Check if VO directions are related to uniform vector (since d_k ≈ all -1s)
    uniform = np.ones(hidden_dim, dtype=np.float32) / np.sqrt(hidden_dim)
    print(f"\n  cos(uniform, VO output U[:, i]):")
    for i in range(min(10, 128)):
        c = abs(np.dot(uniform, Uvo[:, i]))
        print(f"    cos(1/√N, U[{i}]) = {c:.6f}")

    # =========================================================================
    #   Part C: φ-Zipf reconstruction — replace S with φ-Zipf, keep directions
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: φ-Zipf spectral replacement")
    print("=" * 80)

    # Test: keep real U, V directions but replace S with φ-Zipf
    def make_phi_zipf_S(S_orig, alpha_val):
        """Replace S with φ-Zipf: S[i] = S[0] × (i+1)^(-alpha)"""
        n = len(S_orig)
        ranks = np.arange(1, n + 1, dtype=np.float32)
        S_zipf = S_orig[0] * ranks ** (-alpha_val)
        return S_zipf

    def make_vo_from_svd(U, S, Vt, hidden_dim):
        """Reconstruct VO from truncated SVD components."""
        return (U[:, :len(S)] * S[np.newaxis, :]) @ Vt[:len(S), :]

    # Baseline with extracted matrices
    print("\n  Testing with sign-only d_k:", flush=True)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv @ h + bv_group, lambda v: Wo @ v,
        prompts, "Full Wv + Wo (baseline)")

    # Test: VO via SVD reconstruction (sanity check)
    VO_recon = make_vo_from_svd(Uvo, S128, Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_recon @ v + Wo @ bv_group,
        prompts, "VO SVD reconstruction (sanity)")

    # Test: replace S with φ-Zipf using fitted alpha
    S_zipf_fitted = make_phi_zipf_S(S128, alpha)
    VO_zipf = make_vo_from_svd(Uvo, S_zipf_fitted, Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_zipf @ v + Wo @ bv_group,
        prompts, f"φ-Zipf S (α={alpha:.3f}, fitted)")

    # Test: replace S with φ-Zipf using α = 1/φ
    S_zipf_phi = make_phi_zipf_S(S128, 1.0/PHI)
    VO_zipf_phi = make_vo_from_svd(Uvo, S_zipf_phi, Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_zipf_phi @ v + Wo @ bv_group,
        prompts, "φ-Zipf S (α=1/φ)")

    # Test: replace S with φ^(-i) geometric decay
    S_geo = S128[0] * PHI ** (-np.arange(128) * abs(log_ratio))
    VO_geo = make_vo_from_svd(Uvo, S_geo.astype(np.float32), Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_geo @ v + Wo @ bv_group,
        prompts, f"Geometric S (φ^({log_ratio:.3f}×i))")

    # Test: replace S with φ^(-i/φ) — self-similar geometric decay
    S_ss = S128[0] * PHI ** (-np.arange(128) / PHI)
    VO_ss = make_vo_from_svd(Uvo, S_ss.astype(np.float32), Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_ss @ v + Wo @ bv_group,
        prompts, "Self-similar S (φ^(-i/φ))")

    # Test: UNIFORM S (all singular values = mean)
    S_uniform = np.full(128, S128.mean(), dtype=np.float32)
    VO_uni = make_vo_from_svd(Uvo, S_uniform, Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_uni @ v + Wo @ bv_group,
        prompts, "Uniform S (all = mean)")

    # Test: BINARY S (S[0] or 0)
    S_binary = np.where(S128 > S128[0]*0.01, S128[0], 0.0).astype(np.float32)
    VO_bin = make_vo_from_svd(Uvo, S_binary, Vtvo[:128, :], hidden_dim)
    n_active = (S_binary > 0).sum()
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_bin @ v + Wo @ bv_group,
        prompts, f"Binary S ({n_active} active, rest 0)")

    # Test: φ-quantized S (snap each S to nearest φ^level)
    S_phi_quant = PHI ** np.round(np.log(S128 + 1e-20) / LOG_PHI)
    VO_phiq = make_vo_from_svd(Uvo, S_phi_quant.astype(np.float32), Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_phiq @ v + Wo @ bv_group,
        prompts, "φ-quantized S (each S → nearest φ^n)")

    # =========================================================================
    #   Part D: SVD of Wv and Wo individually — φ-structure in their spectra?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: SVD of W_v and W_o individually")
    print("=" * 80)

    # Wv is (128, 3584) — SVD gives 128 singular values
    Uv, Sv, Vtv = np.linalg.svd(Wv, full_matrices=False)
    # Wo is (3584, 128) — SVD gives 128 singular values
    Uo, So, Vto = np.linalg.svd(Wo, full_matrices=False)

    for name, S_mat in [("W_v", Sv), ("W_o", So)]:
        print(f"\n  {name} SVD spectrum ({len(S_mat)} singular values):")
        print(f"    S[0] = {S_mat[0]:.6f}, S[-1] = {S_mat[-1]:.6f}")
        print(f"    S[0]/S[-1] = {S_mat[0]/S_mat[-1]:.1f}")

        # Zipf fit
        lr = np.log(np.arange(1, len(S_mat)+1))
        ls = np.log(S_mat + 1e-20)
        A = np.vstack([lr, np.ones(len(S_mat))]).T
        af, _ = np.linalg.lstsq(A, ls, rcond=None)[0]
        al = -af
        rc = np.corrcoef(lr, ls)[0, 1]
        print(f"    Zipf α = {al:.4f} (1/φ = {1/PHI:.4f}, |diff| = {abs(al-1/PHI):.4f})")
        print(f"    log-log R² = {rc**2:.4f}")

        # φ-level of singular values
        sv_levels = np.log(S_mat) / LOG_PHI
        sv_rnd = np.round(sv_levels)
        sv_res = sv_levels - sv_rnd
        print(f"    SV φ-level range: [{sv_levels[0]:.2f}, {sv_levels[-1]:.2f}]")
        print(f"    Mean |residual|: {np.abs(sv_res).mean():.4f}")
        print(f"    Within 0.1 of int: {(np.abs(sv_res)<0.1).mean()*100:.1f}%")

        # Geometric ratio
        ratios = S_mat[1:] / S_mat[:-1]
        mr = np.mean(ratios)
        print(f"    Mean S[i+1]/S[i] = {mr:.6f} = φ^{np.log(mr)/LOG_PHI:.4f}")

    # =========================================================================
    #   Part E: Zeta symmetry — does VO have σ=1/2 structure?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Zeta-like symmetry in VO")
    print("=" * 80)

    # Check: is VO symmetric, antisymmetric, or neither?
    VO_sym = (VO + VO.T) / 2
    VO_anti = (VO - VO.T) / 2
    sym_norm = np.linalg.norm(VO_sym)
    anti_norm = np.linalg.norm(VO_anti)
    print(f"\n  VO = symmetric + antisymmetric:")
    print(f"    ||VO_sym||  = {sym_norm:.4f} ({sym_norm/(sym_norm+anti_norm)*100:.1f}%)")
    print(f"    ||VO_anti|| = {anti_norm:.4f} ({anti_norm/(sym_norm+anti_norm)*100:.1f}%)")

    # Check: trace (sum of eigenvalues)
    trace_VO = np.trace(VO)
    print(f"    trace(VO) = {trace_VO:.6f}")
    print(f"    sum(S) = {S128.sum():.6f}")

    # Eigenvalue analysis (VO is not symmetric, so use general eig)
    # For the symmetric part:
    eig_sym = np.linalg.eigvalsh(VO_sym)
    eig_sym_sorted = np.sort(eig_sym)[::-1]
    n_pos_eig = (eig_sym > 1e-10).sum()
    n_neg_eig = (eig_sym < -1e-10).sum()
    print(f"\n  Symmetric part eigenvalues:")
    print(f"    Positive: {n_pos_eig}, Negative: {n_neg_eig}, Zero: {len(eig_sym)-n_pos_eig-n_neg_eig}")
    if n_pos_eig > 0 and n_neg_eig > 0:
        print(f"    Largest positive: {eig_sym_sorted[0]:.6f}")
        print(f"    Largest negative: {eig_sym_sorted[-1]:.6f}")
        print(f"    Ratio |pos/neg|: {abs(eig_sym_sorted[0]/eig_sym_sorted[-1]):.2f}")

    # Check: does VO @ d_k produce a φ-structured direction?
    vo_dk = VO @ d_k_normed
    vo_dk_norm = np.linalg.norm(vo_dk)
    vo_dk_dir = vo_dk / vo_dk_norm
    # φ-level analysis of this output
    vo_dk_abs = np.abs(vo_dk)
    nz = vo_dk_abs[vo_dk_abs > 1e-10]
    phi_lvl = np.log(nz) / LOG_PHI
    phi_rnd = np.round(phi_lvl)
    resid = phi_lvl - phi_rnd
    print(f"\n  VO @ d_k direction:")
    print(f"    ||VO @ d_k|| = {vo_dk_norm:.6f}")
    print(f"    φ-level range: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
    print(f"    Mean |residual| from int φ-level: {np.abs(resid).mean():.4f}")
    print(f"    Within 0.1 of int: {(np.abs(resid)<0.1).mean()*100:.1f}%")

    # =========================================================================
    #   Part F: The ENCODE=DECODE test — does V invert O (or vice versa)?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part F: ENCODE=DECODE — V/O inversion structure")
    print("=" * 80)

    # If V and O are geometric inverses: Wv @ Wo ≈ I (in head space)
    VtimesO = Wv @ Wo  # (128, 128) — should this be ~identity?
    ident_128 = np.eye(head_dim, dtype=np.float32)
    diff_from_I = np.linalg.norm(VtimesO - ident_128) / np.linalg.norm(ident_128)
    print(f"\n  W_v @ W_o (128×128) — is it identity?")
    print(f"    ||W_v@W_o - I|| / ||I|| = {diff_from_I:.4f}")

    # Check if it's a scalar multiple of identity
    diag = np.diag(VtimesO)
    off_diag = VtimesO - np.diag(diag)
    print(f"    Mean diagonal: {diag.mean():.6f}")
    print(f"    Std diagonal: {diag.std():.6f}")
    print(f"    Mean |off-diagonal|: {np.abs(off_diag).mean():.6f}")
    print(f"    diag/off-diag ratio: {abs(diag.mean())/np.abs(off_diag).mean():.1f}")

    # Is it a scaled identity? VtimesO ≈ α × I
    alpha_scale = np.trace(VtimesO) / head_dim
    diff_scaled = np.linalg.norm(VtimesO - alpha_scale * ident_128) / np.linalg.norm(alpha_scale * ident_128)
    print(f"    Best scalar: α = {alpha_scale:.6f}")
    print(f"    φ-level of α: {np.log(abs(alpha_scale))/LOG_PHI:.3f}")
    print(f"    ||W_v@W_o - α×I|| / ||α×I|| = {diff_scaled:.4f}")

    # Is α close to a φ-power?
    alpha_phi_level = np.log(abs(alpha_scale)) / LOG_PHI
    alpha_phi_round = round(alpha_phi_level)
    alpha_phi_resid = alpha_phi_level - alpha_phi_round
    print(f"    α ≈ φ^{alpha_phi_level:.3f}, nearest int: φ^{alpha_phi_round}")
    print(f"    Residual: {alpha_phi_resid:.4f}")

    # =========================================================================
    #   Part G: Can the φ-quantized SVD beat element-level φ-quantization?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part G: SVD-level φ-quantization (directions + φ-quantized S)")
    print("=" * 80)

    # φ-quantize the singular values only, keep real directions
    # This stores: 128 integers (6 bits each = 96 bytes) + directions
    S_phi = PHI ** np.round(np.log(S128 + 1e-20) / LOG_PHI)

    # Now: can we also φ-quantize the directions?
    # The U columns and V rows are unit vectors. Their components are real-valued.
    # φ-quantize the direction components
    def phi_quant_matrix(M):
        return np.sign(M) * PHI ** np.round(np.log(np.abs(M) + 1e-20) / LOG_PHI)

    U_phi = phi_quant_matrix(Uvo[:, :128])
    Vt_phi = phi_quant_matrix(Vtvo[:128, :])

    # Test: real directions + φ-quantized S
    VO_realdir_phiS = make_vo_from_svd(Uvo, S_phi.astype(np.float32), Vtvo[:128, :], hidden_dim)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_realdir_phiS @ v + Wo @ bv_group,
        prompts, "Real U,V + φ-quant S")

    # Test: φ-quantized everything (directions + S)
    VO_allphi = (U_phi * S_phi[np.newaxis, :]) @ Vt_phi
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_allphi @ v + Wo @ bv_group,
        prompts, "φ-quant U,V + φ-quant S (all geometric)")

    # Test: sign-only directions + φ-quantized S
    U_sign = np.sign(Uvo[:, :128])
    Vt_sign = np.sign(Vtvo[:128, :])
    VO_signdir_phiS = (U_sign * S_phi[np.newaxis, :]) @ Vt_sign
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_signdir_phiS @ v + Wo @ bv_group,
        prompts, "Sign U,V + φ-quant S")

    # Test: sign-only directions + real S
    VO_signdir_realS = (U_sign * S128[np.newaxis, :]) @ Vt_sign
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: h,
        lambda v: VO_signdir_realS @ v + Wo @ bv_group,
        prompts, "Sign U,V + real S")

    # =========================================================================
    #   Part H: Parameter count comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part H: Parameter counts for geometric VO representations")
    print("=" * 80)

    full_attn_bits = 51_380_224 * 32  # 51.4M params × 32 bits

    configs = [
        ("Full VO (float32)", (hidden_dim * head_dim * 2) * 32),
        ("SVD: real U,V + φ-quant S",
         (hidden_dim * 128 * 32) * 2 + 128 * 6),  # two direction matrices + 128 levels
        ("SVD: φ-quant U,V,S",
         (hidden_dim * 128 * 7) * 2 + 128 * 6),  # 7 bits per component (sign+6bit level)
        ("SVD: sign U,V + real S",
         (hidden_dim * 128) * 2 + 128 * 32),  # 1 bit per direction component + 32-bit S
        ("SVD: sign U,V + φ-quant S",
         (hidden_dim * 128) * 2 + 128 * 6),  # 1 bit + 6 bits
        ("Sign d_k + sign U,V + φ-quant S (total resonator)",
         hidden_dim + (hidden_dim * 128) * 2 + 128 * 6 + head_dim * 32),  # +bias
    ]

    for name, bits in configs:
        kb = bits / 8 / 1024
        compression = full_attn_bits / bits
        print(f"    {name:>55s}: {bits:>12,d} bits ({kb:>8.1f} KB)  {compression:>6.0f}× vs full attn")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
