#!/usr/bin/env python3
"""
Phase 10z5: Dirichlet Series Processor Hypothesis
===================================================

Finding 108 + 10z4 correction established:
- Processor is NOT Newton iteration (ratio 0.99 vs 0.44)
- Processor IS spectral accumulation (back-loaded, answer crystallizes late)
- SVD power-law α=2/φ² matches Dirichlet-like decay

HYPOTHESIS: The Processor computes a truncated Dirichlet series in SVD space:
    answer ≈ Σ_{k=1}^{17} σ_k · v_k

where σ_k ~ k^(-2/φ²) and the 17 effective components correspond
to the rank-17 SVD from F107.

TESTS:
1. CUMULATIVE SVD RECONSTRUCTION: How much of the "answer" does each
   additional SVD component capture? Does it match Dirichlet partial sums?

2. DIRICHLET CONVERGENCE RATE: In ζ(s), partial sums Σ_{n=1}^{N} n^{-s}
   converge at a specific rate. Does the SVD partial sum match?

3. BACK-LOADING TEST: In ζ(s), later terms matter LESS (n^{-s} shrinks).
   But in the transformer, later layers matter MORE. How does the SVD
   reconcile this? (Answer: layers ≠ SVD components)

4. φ IN THE SVD: Are the singular vectors themselves φ-structured?
   Do they have 3×5=15 fold symmetry? Pentagonal angles?

5. THE CRYSTALLIZATION POINT: At what SVD rank does the answer "crystallize"?
   Is it at rank ≈ φ⁴ ≈ 7? Or rank ≈ 15 (3×5)?
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mpmath import zeta, mp, mpc, fabs
import json
import os

mp.dps = 15

PHI = (1 + np.sqrt(5)) / 2
TWO_PI = 2 * np.pi

MODEL_NAME = "Qwen/Qwen2.5-7B"
N_LAYERS = 28

PROMPTS = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The speed of light is approximately",
    "In mathematics, pi equals",
    "The largest planet in our solar system is",
    "Photosynthesis converts sunlight into",
    "The human heart has four",
    "DNA stands for deoxyribonucleic",
    "Shakespeare wrote Romeo and",
    "The chemical symbol for gold is",
    "Gravity pulls objects toward the",
    "The Pacific Ocean is the",
    "Electrons orbit around the",
    "The Pythagorean theorem states that",
    "Carbon dioxide is composed of",
    "The mitochondria is the powerhouse of the",
    "Newton discovered that gravity",
    "The boiling point of water is",
    "Hydrogen is the lightest",
    "The Great Wall of China was built to",
    "Sound travels faster through water than through",
    "The moon orbits the",
    "Einstein developed the theory of",
    "Diamonds are made of",
    "The Amazon River flows through",
    "Oxygen makes up about twenty percent of",
    "The Earth rotates on its",
    "Antibiotics are used to treat bacterial",
    "The speed of sound is approximately",
    "Volcanoes erupt when magma reaches the",
]


def extract_additions_and_target(model, tokenizer):
    """
    Extract per-layer additions (what each layer ADDS to the residual stream)
    and the prediction direction, for all prompts.
    """
    print("  Extracting layer additions and targets...")

    all_additions = []  # (n_prompts, n_layers, hidden_dim)
    all_pred_dirs = []  # (n_prompts, hidden_dim)
    all_trajectories = []  # (n_prompts, n_layers+1) cumulative projection

    for pi, prompt in enumerate(PROMPTS):
        print(f"\r    Prompt {pi+1}/{len(PROMPTS)}", end="", flush=True)

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        # Hooks to capture layer inputs
        layer_inputs = {}
        hooks = []

        for li in range(N_LAYERS):
            def make_hook(idx):
                def fn(mod, args):
                    layer_inputs[idx] = args[0].detach().clone()
                return fn
            hooks.append(model.model.layers[li].register_forward_pre_hook(make_hook(li)))

        def out_hook(mod, args, output):
            layer_inputs["final_out"] = output.detach().clone()
        hooks.append(model.model.layers[N_LAYERS-1].register_forward_hook(out_hook))

        with torch.no_grad():
            outputs = model(**tokens)
        for hk in hooks:
            hk.remove()

        # Prediction direction
        logits = outputs.logits[0, -1, :]
        pred_token = logits.argmax()
        pred_dir = model.lm_head.weight[pred_token].detach().float()
        pred_dir = pred_dir / pred_dir.norm()
        all_pred_dirs.append(pred_dir.numpy())

        # Precompute attention infrastructure
        with torch.no_grad():
            h_embed = model.model.embed_tokens(input_ids)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_embed, position_ids)
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Extract additions layer by layer
        additions = []
        trajectory = []

        h_emb_lt = h_embed[0, -1, :].float()
        trajectory.append(float(torch.dot(h_emb_lt, pred_dir)))

        with torch.no_grad():
            for li in range(N_LAYERS):
                layer = model.model.layers[li]
                h_in = layer_inputs[li]
                h_in_lt = h_in[0, -1].float()

                s1 = layer.input_layernorm(h_in)
                s2, _ = layer.self_attn(
                    hidden_states=s1,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                s3 = h_in + s2
                s4 = layer.post_attention_layernorm(s3)
                s5 = layer.mlp(s4)
                s6 = s3 + s5
                s6_lt = s6[0, -1].float()

                addition = (s6_lt - h_in_lt).numpy()
                additions.append(addition)
                trajectory.append(float(torch.dot(s6_lt, pred_dir)))

        all_additions.append(np.array(additions))  # (28, 3584)
        all_trajectories.append(trajectory)

    print()
    return all_additions, all_pred_dirs, all_trajectories


def test_svd_dirichlet(all_additions, all_pred_dirs, all_trajectories):
    """
    Test 1-5: SVD structure matches Dirichlet series.
    """
    n_prompts = len(all_additions)
    hidden_dim = all_additions[0].shape[1]

    # ================================================================
    # TEST 1: Cumulative SVD reconstruction
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Cumulative SVD Reconstruction")
    print("=" * 80)

    # Average addition matrix across prompts
    mean_additions = np.mean(all_additions, axis=0)  # (28, 3584)
    mean_pred_dir = np.mean(all_pred_dirs, axis=0)
    mean_pred_dir /= np.linalg.norm(mean_pred_dir)

    # SVD
    U, S, Vt = np.linalg.svd(mean_additions, full_matrices=False)
    print(f"\n  SVD: {mean_additions.shape} → U({U.shape}), S({S.shape}), Vt({Vt.shape})")
    print(f"  Top 5 singular values: {S[:5]}")

    # How much of the PROJECTION onto pred_dir does each SV component capture?
    # The full projection: additions @ pred_dir = (U S Vt) @ pred_dir
    # = U @ diag(S) @ (Vt @ pred_dir)
    Vt_pred = Vt @ mean_pred_dir  # (28,) — how much each right SV aligns with pred
    US = U * S[None, :]  # (28, 28)

    # Full trajectory projection
    full_proj = mean_additions @ mean_pred_dir  # (28,)
    cumsum_full = np.cumsum(full_proj)

    # Per-SV-component projection
    sv_projections = []
    for k in range(len(S)):
        # Component k: σ_k * u_k * (v_k · pred_dir)
        comp_k = S[k] * U[:, k] * Vt_pred[k]  # (28,) per-layer contribution of SV k
        sv_projections.append(comp_k)

    sv_projections = np.array(sv_projections)  # (28, 28) — [sv_idx, layer_idx]

    # Cumulative reconstruction: how well do first K SVs reconstruct the trajectory?
    print(f"\n  Cumulative SVD reconstruction of prediction trajectory:")
    print(f"  {'Rank':>6}  {'Explained%':>10}  {'Residual':>10}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*40}")

    crystallization_rank = None
    total_var = np.sum(full_proj**2)

    for K in range(1, min(len(S) + 1, 29)):
        # Reconstruct with first K components
        recon = np.sum(sv_projections[:K, :], axis=0)  # (28,)
        explained = 1 - np.sum((full_proj - recon)**2) / total_var
        residual = np.sqrt(np.sum((full_proj - recon)**2))
        bar = "█" * int(explained * 40) if explained > 0 else ""

        marker = ""
        if K == int(PHI**4):
            marker = f" ← φ⁴={PHI**4:.1f}"
        elif K == 15:
            marker = " ← 3×5"
        elif K == 17:
            marker = " ← rank-17 (F107)"

        if explained > 0.99 and crystallization_rank is None:
            crystallization_rank = K
            marker += " ★ CRYSTALLIZATION"

        print(f"  K={K:3d}  {explained*100:9.2f}%  {residual:10.4f}  {bar}{marker}")

    if crystallization_rank:
        print(f"\n  Crystallization at rank {crystallization_rank}")
        for name, val in [("φ", PHI), ("φ²", PHI**2), ("φ³", PHI**3),
                          ("φ⁴", PHI**4), ("3×5", 15), ("φ^7/4", PHI**7/4)]:
            match = (1 - abs(crystallization_rank - val) / val) * 100
            if match > 50:
                print(f"    ≈ {name} = {val:.2f} ({match:.1f}% match)")

    # ================================================================
    # TEST 2: Dirichlet convergence rate comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Dirichlet Convergence Rate")
    print("=" * 80)

    # For ζ(s), partial sum error ≈ N^(1-s) / (s-1) for Re(s) > 1
    # On the critical line s=1/2+it, it diverges — need Abel summation
    # But for comparison: the SV decay rate α gives convergence rate

    # SV power law fit: S_k ~ A * k^(-α)
    k_vals = np.arange(1, len(S) + 1)
    log_k = np.log(k_vals)
    log_S = np.log(S + 1e-20)
    valid = S > S[0] * 1e-6
    if np.sum(valid) > 2:
        coeffs = np.polyfit(log_k[valid], log_S[valid], 1)
        alpha_sv = -coeffs[0]
        A_sv = np.exp(coeffs[1])
        R2 = 1 - np.var(log_S[valid] - np.polyval(coeffs, log_k[valid])) / np.var(log_S[valid])

        print(f"\n  SV power law: σ_k ~ {A_sv:.1f} × k^(-{alpha_sv:.4f})")
        print(f"  R² = {R2:.4f}")
        print(f"\n  φ-expression match for α = {alpha_sv:.4f}:")
        for name, val in [("1/φ", 1/PHI), ("2/φ²", 2/PHI**2), ("1/φ²", 1/PHI**2),
                          ("1/(φ-1)", 1/(PHI-1)), ("φ/2", PHI/2)]:
            match = (1 - abs(alpha_sv - val) / val) * 100
            print(f"    {name:8s} = {val:.4f}: {match:.1f}% match")

    # Dirichlet partial sum convergence
    print(f"\n  Dirichlet ζ(s) partial sum convergence at s=1/2+14i:")
    s = complex(0.5, 14.134725)  # Near first zero
    partial = 0
    for n in range(1, 29):
        partial += n**(-s)
        if n <= 28:
            err = abs(partial - complex(float(zeta(mpc(0.5, 14.134725)).real),
                                        float(zeta(mpc(0.5, 14.134725)).imag)))
            if n in [1, 2, 3, 5, 7, 10, 15, 17, 20, 25, 28]:
                print(f"    N={n:3d}: |partial - ζ| = {err:.6f}")

    # ================================================================
    # TEST 3: Back-loading reconciliation
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Back-Loading — Layers vs SVD Components")
    print("=" * 80)

    # In layer space, later layers contribute MORE (back-loaded)
    # In SVD space, later components contribute LESS (front-loaded, like Dirichlet)
    # The SVD rotation RECONCILES these

    print(f"\n  Per-layer projection onto pred_dir:")
    print(f"  {'Layer':>6}  {'Projection':>10}  {'Cumulative':>10}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*40}")

    cumsum = 0
    max_abs_proj = max(abs(p) for p in full_proj)
    for li in range(N_LAYERS):
        cumsum += full_proj[li]
        bar_len = int(abs(full_proj[li]) / max_abs_proj * 30) if max_abs_proj > 0 else 0
        sign = "+" if full_proj[li] >= 0 else "-"
        zone = ""
        if li <= 3: zone = " [COMP]"
        elif li <= 25: zone = " [PROC]"
        else: zone = " [TARG]"
        print(f"  L{li:02d}    {full_proj[li]:+10.4f}  {cumsum:10.4f}  "
              f"{'█' * bar_len}{zone}")

    # Compare: per-SV projection
    print(f"\n  Per-SV-component projection onto pred_dir:")
    print(f"  {'SV':>6}  {'|σ·v·p|':>10}  {'Cumul%':>10}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*40}")

    cumul_var = 0
    for k in range(min(len(S), 28)):
        contrib = S[k] * abs(Vt_pred[k])
        cumul_var += contrib**2
        explained_pct = cumul_var / (np.sum((S * np.abs(Vt_pred))**2) + 1e-20) * 100
        bar = "█" * int(contrib / (S[0] * abs(Vt_pred[0]) + 1e-20) * 30)
        print(f"  SV{k:02d}  {contrib:10.4f}  {explained_pct:9.1f}%  {bar}")

    # ================================================================
    # TEST 4: φ in the SVD — singular vector structure
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 4: φ-Structure in Singular Vectors")
    print("=" * 80)

    # Check angles between consecutive left singular vectors
    print(f"\n  Angles between consecutive left singular vectors (U columns):")
    sv_angles = []
    for k in range(min(len(S) - 1, 20)):
        cos_a = np.dot(U[:, k], U[:, k+1])
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        sv_angles.append(angle)

    for k, angle in enumerate(sv_angles):
        marker = ""
        for name, val in [("90° ortho", 90), ("72° penta", 72),
                          ("70.5° tetra", 70.53), ("60° hexa", 60)]:
            if abs(angle - val) < 5:
                marker = f" ≈ {name}"
                break
        print(f"    SV{k:02d}↔SV{k+1:02d}: {angle:.1f}°{marker}")

    mean_sv_angle = np.mean(sv_angles)
    print(f"\n  Mean angle: {mean_sv_angle:.1f}°")
    print(f"  Expected if orthogonal: 90.0°")
    print(f"  Expected if pentagonal: 72.0° (arccos(1/2φ))")
    print(f"  Expected if tetrahedral: 70.5° (arccos(1/3))")

    # Check FFT of first singular vector for 15-fold structure
    print(f"\n  FFT of first left singular vector (U[:,0]):")
    u0 = U[:, 0]
    fft_u0 = np.abs(np.fft.fft(u0))
    freqs = np.fft.fftfreq(len(u0))
    half = len(fft_u0) // 2
    top_k = np.argsort(fft_u0[1:half])[-5:][::-1] + 1
    for k in top_k:
        period = 1.0 / freqs[k] if freqs[k] != 0 else float('inf')
        energy = fft_u0[k]**2 / np.sum(fft_u0[1:half]**2) * 100
        marker = ""
        for name, val in [("φ^7/4", PHI**7/4), ("φ⁴", PHI**4), ("15", 15),
                          ("φ³", PHI**3), ("7", 7), ("14", 14), ("28", 28)]:
            if abs(period - val) / val < 0.15:
                marker = f" ≈ {name}"
                break
        print(f"    k={k}: period={period:.1f}, energy={energy:.1f}%{marker}")

    # ================================================================
    # TEST 5: Crystallization point
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Crystallization — When Does the Answer Emerge?")
    print("=" * 80)

    # Per-prompt: reconstruct trajectory from SVD, find crystallization rank
    cryst_ranks = []
    for pi in range(n_prompts):
        add_matrix = all_additions[pi]  # (28, 3584)
        p_dir = all_pred_dirs[pi]
        p_dir /= np.linalg.norm(p_dir)

        proj = add_matrix @ p_dir  # (28,)
        total_var_p = np.sum(proj**2)
        if total_var_p < 1e-10:
            continue

        U_p, S_p, Vt_p = np.linalg.svd(add_matrix, full_matrices=False)
        Vt_pred_p = Vt_p @ p_dir
        sv_proj_p = []
        for k in range(len(S_p)):
            sv_proj_p.append(S_p[k] * U_p[:, k] * Vt_pred_p[k])
        sv_proj_p = np.array(sv_proj_p)

        for K in range(1, len(S_p) + 1):
            recon = np.sum(sv_proj_p[:K, :], axis=0)
            explained = 1 - np.sum((proj - recon)**2) / total_var_p
            if explained > 0.95:
                cryst_ranks.append(K)
                break

    if cryst_ranks:
        mean_cryst = np.mean(cryst_ranks)
        med_cryst = np.median(cryst_ranks)
        print(f"\n  Per-prompt crystallization rank (95% explained):")
        print(f"    Mean: {mean_cryst:.1f}")
        print(f"    Median: {med_cryst:.1f}")
        print(f"    Min: {min(cryst_ranks)}, Max: {max(cryst_ranks)}")

        print(f"\n  φ-expression matches:")
        for name, val in [("1", 1), ("φ", PHI), ("φ²", PHI**2), ("3", 3),
                          ("φ³", PHI**3), ("5", 5), ("φ⁴", PHI**4),
                          ("7", 7), ("15", 15), ("17", 17)]:
            match = (1 - abs(mean_cryst - val) / max(val, 1)) * 100
            if match > 50:
                print(f"    {name:5s} = {val:.2f}: {match:.1f}% match")

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("\n" + "=" * 80)
    print("SYNTHESIS: Is the Processor a Dirichlet Series Computer?")
    print("=" * 80)

    evidence_for = []
    evidence_against = []

    # SV decay
    if 'alpha_sv' in dir():
        alpha_match = (1 - abs(alpha_sv - 2/PHI**2) / (2/PHI**2)) * 100
        if alpha_match > 90:
            evidence_for.append(f"SV decay α={alpha_sv:.3f} matches 2/φ²={2/PHI**2:.3f} "
                              f"({alpha_match:.1f}%)")
        else:
            evidence_against.append(f"SV decay α={alpha_sv:.3f} doesn't match 2/φ²")

    # Crystallization
    if cryst_ranks:
        if abs(mean_cryst - 1) < 1:
            evidence_for.append(f"Crystallization at rank {mean_cryst:.0f} — "
                              f"RANK-1 DOMINATED (like Targeter)")
        elif abs(mean_cryst - PHI**2) < 2:
            evidence_for.append(f"Crystallization at rank {mean_cryst:.1f} ≈ φ²={PHI**2:.1f}")
        elif abs(mean_cryst - 17) < 3:
            evidence_for.append(f"Crystallization at rank {mean_cryst:.1f} ≈ 17 (F107)")

    # Back-loading
    proc_proj = full_proj[4:26]
    if np.sum(proc_proj[-6:]**2) > np.sum(proc_proj[:6]**2):
        evidence_for.append("Back-loaded: last 6 Processor layers contribute more than first 6")
    else:
        evidence_against.append("NOT back-loaded in projection space")

    print(f"\n  Evidence FOR Dirichlet interpretation:")
    for e in evidence_for:
        print(f"    ✅ {e}")

    print(f"\n  Evidence AGAINST:")
    for e in evidence_against:
        print(f"    ❌ {e}")

    if not evidence_against:
        print(f"    (none)")

    # Save
    save_data = {
        "sv_decay_alpha": float(alpha_sv) if 'alpha_sv' in dir() else None,
        "sv_decay_R2": float(R2) if 'R2' in dir() else None,
        "crystallization_ranks": cryst_ranks if cryst_ranks else [],
        "mean_crystallization": float(mean_cryst) if cryst_ranks else None,
        "sv_angles": [float(a) for a in sv_angles],
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/phase10z5_dirichlet_processor.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to results/phase10z5_dirichlet_processor.json")


def main():
    print("=" * 80)
    print("PHASE 10z5: DIRICHLET SERIES PROCESSOR HYPOTHESIS")
    print("=" * 80)
    print(f"\nHypothesis: Processor (L4-25) computes truncated Dirichlet series")
    print(f"in SVD space: answer ≈ Σ_k σ_k · v_k with σ_k ~ k^(-2/φ²)")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    print(f"  Loaded. {N_LAYERS} layers, hidden_dim={model.config.hidden_size}")

    # Extract data
    all_additions, all_pred_dirs, all_trajectories = extract_additions_and_target(
        model, tokenizer
    )

    # Run tests
    test_svd_dirichlet(all_additions, all_pred_dirs, all_trajectories)


if __name__ == "__main__":
    main()
