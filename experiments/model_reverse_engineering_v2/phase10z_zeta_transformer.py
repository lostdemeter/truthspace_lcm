#!/usr/bin/env python3
"""
Phase 10z: Transformer as Zeta Function
========================================

THE QUESTION: A transformer shares the same geometric structure as the
Riemann zeta function (3-fold x 5-fold, phi-governed, holofractal).
Can we "crunch" transformers using the same mathematical machinery?

TESTS:
------
1. Power Law Decay: Do addition norms decay as l^(-alpha)? Compare with 1/phi^2.
2. Harmonic Decomposition: FFT the layer trajectory. Do zeta harmonics (3,6,9,12) dominate?
3. Ramanujan Fit: Can we fit trajectory to base + harmonics + spiral?
4. Transformer "Zeros": Sign changes in additions — GUE or Poisson spacing?
5. Zeta Prediction: Predict final output from partial layers using zeta-like formulas.
6. Dirichlet Polynomial: Construct T(s) from layer additions, find zeros on Re(s)=1/2.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from scipy.fft import fft
from scipy.optimize import curve_fit
from scipy.special import lambertw

MODEL_NAME = "Qwen/Qwen2.5-7B"
PHI = (1 + np.sqrt(5)) / 2
N_LAYERS = 28

PROMPTS = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The speed of light is approximately",
    "In mathematics, pi equals",
    "The largest planet in our solar system is",
    "Photosynthesis converts sunlight into",
    "The chemical formula for water is",
    "Shakespeare wrote Romeo and",
    "The square root of 144 is",
    "Electrons orbit the nucleus of an",
    "The theory of relativity was developed by",
    "DNA stands for deoxyribonucleic",
    "The boiling point of water is 100 degrees",
    "Gravity pulls objects toward the",
    "The periodic table organizes chemical",
    "Neural networks are inspired by the",
    "The Pythagorean theorem states that",
    "Oxygen is essential for",
    "The Renaissance began in",
    "Quantum mechanics describes behavior at the",
    "The mitochondria is the powerhouse of the",
    "Binary code uses zeros and",
    "The human genome contains approximately",
    "Plate tectonics explains the movement of",
    "The Fibonacci sequence starts with",
    "Entropy measures the disorder of a",
    "The Milky Way is a spiral",
    "Ohm's law relates voltage and",
    "The double helix structure of DNA was",
    "Machine learning algorithms improve through",
]


def extract_layer_data(model, tokenizer):
    """Extract per-layer additions and trajectories for all prompts.
    Uses hooks + rotary_emb pattern (from phase10t_crossroads_tests.py)."""
    all_trajectories = []
    all_addition_norms = []
    all_addition_angles = []
    all_addition_projs = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\r  Processing prompt {pi+1}/{len(PROMPTS)}", end="", flush=True)

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Get prediction direction from full forward pass
            outputs = model(**tokens)
            logits = outputs.logits[0, -1, :]
            pred_token = logits.argmax()
            pred_dir = model.lm_head.weight[pred_token].detach().float()
            pred_dir = pred_dir / pred_dir.norm()

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
            model(input_ids)
        for hk in hooks:
            hk.remove()

        # Precompute attention infrastructure
        with torch.no_grad():
            h_embed = model.model.embed_tokens(input_ids)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_embed, position_ids)
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Dissect all layers
        trajectory = []
        addition_norms = []
        addition_angles = []
        addition_projs = []

        # Initial embedding projection
        h_emb_lt = h_embed[0, -1, :]
        trajectory.append(float(torch.dot(h_emb_lt, pred_dir)))

        with torch.no_grad():
            for li in range(N_LAYERS):
                layer = model.model.layers[li]
                h_in = layer_inputs[li]
                h_in_lt = h_in[0, -1]

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
                s6_lt = s6[0, -1]

                # Total addition this layer
                total_add = s6_lt - h_in_lt

                # Norm
                addition_norms.append(total_add.norm().item())

                # Angle vs residual
                cos_a = torch.dot(total_add, h_in_lt) / (total_add.norm() * h_in_lt.norm() + 1e-10)
                addition_angles.append(np.degrees(np.arccos(np.clip(cos_a.item(), -1, 1))))

                # Projection onto prediction direction
                addition_projs.append(float(torch.dot(total_add, pred_dir)))

                # Cumulative trajectory
                trajectory.append(float(torch.dot(s6_lt, pred_dir)))

        all_trajectories.append(trajectory)
        all_addition_norms.append(addition_norms)
        all_addition_angles.append(addition_angles)
        all_addition_projs.append(addition_projs)

    print()
    return (np.array(all_trajectories), np.array(all_addition_norms),
            np.array(all_addition_angles), np.array(all_addition_projs))


def test1_power_law(add_norms):
    """TEST 1: Power Law Decay of Addition Norms."""
    print("\n" + "=" * 80)
    print("TEST 1: Power Law Decay of Addition Norms")
    print("=" * 80)

    mean_norms = np.mean(add_norms, axis=0)

    # Fit on L2-L27 (skip L0-L1 creator zone)
    layers_fit = np.arange(2, N_LAYERS)
    norms_fit = mean_norms[2:]

    log_l = np.log(layers_fit)
    log_n = np.log(norms_fit + 1e-10)

    valid = np.isfinite(log_n) & np.isfinite(log_l)
    z = np.polyfit(log_l[valid], log_n[valid], 1)
    alpha_fit = -z[0]
    A_fit = np.exp(z[1])

    predicted = z[0] * log_l[valid] + z[1]
    ss_res = np.sum((log_n[valid] - predicted) ** 2)
    ss_tot = np.sum((log_n[valid] - np.mean(log_n[valid])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Power law fit: norm ~ {A_fit:.4f} * l^(-{alpha_fit:.4f})")
    print(f"  R² = {r2:.4f}")
    print(f"  1/φ² = {1/PHI**2:.4f} (expected from KK-1)")
    print(f"  Ratio α/(1/φ²) = {alpha_fit / (1/PHI**2):.4f}")
    print(f"  If Dirichlet series, Re(s) ≈ {alpha_fit:.4f} (critical line = 0.5)")

    print(f"\n  Per-layer mean norms:")
    for li in range(N_LAYERS):
        bar = "█" * int(mean_norms[li] / max(mean_norms) * 40)
        print(f"    L{li:2d}: {mean_norms[li]:8.4f} {bar}")

    return {"alpha": float(alpha_fit), "r2": float(r2),
            "expected": float(1/PHI**2), "norms": mean_norms.tolist()}


def test2_harmonics(add_projs):
    """TEST 2: Harmonic Decomposition of Layer Additions."""
    print("\n" + "=" * 80)
    print("TEST 2: Harmonic Decomposition of Layer Trajectory")
    print("=" * 80)

    # Per-prompt FFT, then average magnitudes
    all_mags = []
    for pi in range(len(PROMPTS)):
        fft_p = fft(add_projs[pi])
        all_mags.append(np.abs(fft_p))
    mean_mags = np.mean(all_mags, axis=0)

    print(f"\n  Mean FFT magnitudes (averaged across {len(PROMPTS)} prompts):")
    max_m = max(mean_mags[1:N_LAYERS//2+1])
    for k in range(N_LAYERS//2 + 1):
        rel = mean_mags[k] / max_m if max_m > 0 else 0
        bar = "█" * int(rel * 30)
        note = ""
        if k in [3, 6, 9, 12]:
            note = " ← ZETA"
        if k == 5:
            note = " ← 5-fold"
        if k == 13:
            note = " ← alias of 15"
        print(f"    k={k:2d}: {mean_mags[k]:10.4f} ({rel:5.2f}) {bar}{note}")

    # Note: k=15 aliases to k=28-15=13 with 28 samples
    zeta_ks = [3, 6, 9, 12, 13]  # 13 = alias of 15
    zeta_e = sum(mean_mags[k]**2 for k in zeta_ks)
    total_e = sum(mean_mags[k]**2 for k in range(1, N_LAYERS//2+1))
    print(f"\n  Zeta harmonic energy (3,6,9,12,13≡15): {zeta_e/total_e*100:.1f}%")

    return {"magnitudes": mean_mags[:N_LAYERS//2+1].tolist(),
            "zeta_energy_pct": float(zeta_e/total_e*100)}


def test3_ramanujan_fit(trajectories):
    """TEST 3: Ramanujan Predictor Fit to Layer Trajectory."""
    print("\n" + "=" * 80)
    print("TEST 3: Ramanujan Predictor Fit to Layer Trajectory")
    print("=" * 80)

    mean_traj = np.mean(trajectories, axis=0)
    layers = np.arange(N_LAYERS + 1)
    ss_tot = np.sum((mean_traj - np.mean(mean_traj))**2)

    # Model 1: Log base only
    def log_model(l, a, b, c):
        return a * np.log(l + 1) + b + c * l

    r2_log = 0
    try:
        popt, _ = curve_fit(log_model, layers, mean_traj, p0=[1, 0, 0])
        pred = log_model(layers, *popt)
        r2_log = 1 - np.sum((mean_traj - pred)**2) / ss_tot
        print(f"\n  Model 1 (log base): R² = {r2_log:.4f}")
    except Exception as e:
        print(f"\n  Model 1: FIT FAILED ({e})")

    # Model 2: Log base + zeta harmonics (3,6,9,12)
    def ram_model(l, a, b, c, h3, h6, h9, h12):
        base = a * np.log(l + 1) + b + c * l
        theta = 2 * np.pi * l / 28
        return base + h3*np.sin(3*theta) + h6*np.sin(6*theta) + h9*np.sin(9*theta) + h12*np.sin(12*theta)

    r2_ram = 0
    try:
        popt, _ = curve_fit(ram_model, layers, mean_traj, p0=[1,0,0,0,0,0,0], maxfev=10000)
        pred = ram_model(layers, *popt)
        r2_ram = 1 - np.sum((mean_traj - pred)**2) / ss_tot
        h_str = {3: abs(popt[3]), 6: abs(popt[4]), 9: abs(popt[5]), 12: abs(popt[6])}
        strongest = max(h_str, key=h_str.get)
        print(f"  Model 2 (Ramanujan): R² = {r2_ram:.4f}")
        print(f"    h3={popt[3]:.4f}, h6={popt[4]:.4f}, h9={popt[5]:.4f}, h12={popt[6]:.4f}")
        print(f"    Strongest: k={strongest} (|h|={h_str[strongest]:.4f})")
        print(f"    Harmonic improvement: +{r2_ram - r2_log:.4f}")
    except Exception as e:
        print(f"  Model 2: FIT FAILED ({e})")

    # Model 3: All harmonics 1-13
    def free_model(l, *params):
        a, b, c = params[:3]
        base = a * np.log(l + 1) + b + c * l
        theta = 2 * np.pi * l / 28
        harmonics = sum(params[3+k] * np.sin((k+1)*theta) for k in range(13))
        return base + harmonics

    r2_free = 0
    try:
        p0 = [1, 0, 0] + [0]*13
        popt, _ = curve_fit(free_model, layers, mean_traj, p0=p0, maxfev=20000)
        pred = free_model(layers, *popt)
        r2_free = 1 - np.sum((mean_traj - pred)**2) / ss_tot
        print(f"  Model 3 (all harmonics): R² = {r2_free:.4f}")
        h_all = {k+1: abs(popt[3+k]) for k in range(13)}
        max_h = max(h_all.values()) if max(h_all.values()) > 0 else 1
        print(f"    Harmonic strengths:")
        for k in range(1, 14):
            bar = "█" * int(h_all[k] / max_h * 30)
            note = " ← ZETA" if k in [3, 6, 9, 12] else ""
            note = " ← 5-fold" if k == 5 else note
            note = " ← alias(15)" if k == 13 else note
            print(f"      k={k:2d}: {h_all[k]:8.4f} {bar}{note}")

        zeta_e = sum(h_all[k]**2 for k in [3, 6, 9, 12])
        total_e = sum(v**2 for v in h_all.values())
        print(f"    Zeta harmonic energy: {zeta_e/total_e*100:.1f}%")
    except Exception as e:
        print(f"  Model 3: FIT FAILED ({e})")

    return {"r2_log": float(r2_log), "r2_ramanujan": float(r2_ram),
            "r2_free": float(r2_free)}


def test4_zeros(add_projs):
    """TEST 4: Transformer 'Zeros' — Sign Changes in Addition Projections."""
    print("\n" + "=" * 80)
    print("TEST 4: Transformer 'Zeros' — Sign Changes in Additions")
    print("=" * 80)

    all_zero_positions = []
    all_spacings = []

    for pi in range(len(PROMPTS)):
        projs = add_projs[pi]
        signs = np.sign(projs)
        zeros = []
        for li in range(len(signs) - 1):
            if signs[li] != signs[li+1] and signs[li] != 0 and signs[li+1] != 0:
                t = projs[li] / (projs[li] - projs[li+1])
                zeros.append(li + t)
        all_zero_positions.append(zeros)
        if len(zeros) > 1:
            all_spacings.extend(np.diff(zeros))

    n_zeros = [len(z) for z in all_zero_positions]
    print(f"\n  Mean zeros per prompt: {np.mean(n_zeros):.1f} ± {np.std(n_zeros):.1f}")
    print(f"  Range: [{min(n_zeros)}, {max(n_zeros)}]")

    result = {"mean_zeros": float(np.mean(n_zeros))}

    if len(all_spacings) > 5:
        spacings = np.array(all_spacings)
        norm_sp = spacings / np.mean(spacings)
        measured_var = float(np.var(norm_sp))

        print(f"\n  Normalized spacing variance: {measured_var:.4f}")
        print(f"  GUE prediction:    0.286")
        print(f"  Poisson prediction: 1.000")

        if abs(measured_var - 0.286) < abs(measured_var - 1.0):
            print(f"  → Closer to GUE (repulsive, like zeta zeros)")
        else:
            print(f"  → Closer to Poisson (uncorrelated)")

        # Histogram of zero positions
        all_zp = [z for zeros in all_zero_positions for z in zeros]
        hist, _ = np.histogram(all_zp, bins=N_LAYERS, range=(0, N_LAYERS))
        print(f"\n  Zero position histogram:")
        max_h = max(hist) if max(hist) > 0 else 1
        for i in range(len(hist)):
            bar = "█" * int(hist[i] / max_h * 20)
            print(f"    L{i:2d}-{i+1:2d}: {hist[i]:3d} {bar}")

        result["spacing_var"] = measured_var
    else:
        print(f"  Not enough zeros for spacing analysis")

    return result


def test5_prediction(trajectories, add_projs):
    """TEST 5: Predict Final Output from Partial Layers."""
    print("\n" + "=" * 80)
    print("TEST 5: Zeta Prediction — Final from Partial Layers")
    print("=" * 80)

    final_vals = trajectories[:, -1]
    range_val = np.std(final_vals)

    methods = {}

    # A: Linear extrapolation from L0-3
    pred_A = []
    for pi in range(len(PROMPTS)):
        z = np.polyfit([0,1,2,3], trajectories[pi, :4], 1)
        pred_A.append(z[0] * 28 + z[1])
    err_A = np.abs(np.array(pred_A) - final_vals)
    methods["A_linear_4layers"] = float(np.mean(err_A))

    # B: Log extrapolation from L0-3
    pred_B = []
    for pi in range(len(PROMPTS)):
        try:
            popt, _ = curve_fit(lambda l, a, b: a*np.log(l+1)+b,
                               [0,1,2,3], trajectories[pi, :4], p0=[1,0])
            pred_B.append(popt[0] * np.log(29) + popt[1])
        except:
            pred_B.append(pred_A[pi])
    err_B = np.abs(np.array(pred_B) - final_vals)
    methods["B_log_4layers"] = float(np.mean(err_B))

    # C: Log+linear from 5 strategic layers (like Newton sample points)
    sample_layers = [0, 3, 7, 14, 27]
    pred_C = []
    for pi in range(len(PROMPTS)):
        sv = [trajectories[pi, l] for l in sample_layers]
        try:
            popt, _ = curve_fit(lambda l, a, b, c: a*np.log(l+1)+b+c*l,
                               sample_layers, sv, p0=[1,0,0])
            pred_C.append(popt[0]*np.log(29) + popt[1] + popt[2]*28)
        except:
            pred_C.append(sv[-1])
    err_C = np.abs(np.array(pred_C) - final_vals)
    methods["C_strategic_5layers"] = float(np.mean(err_C))

    # D: Power-law calibrated from first 5 layers
    pred_D = []
    for pi in range(len(PROMPTS)):
        try:
            projs_5 = add_projs[pi][:5]
            z = np.polyfit(np.log(np.arange(1, 6)), np.log(np.abs(projs_5)+1e-10), 1)
            alpha_e = -z[0]
            A_e = np.exp(z[1])
            total = trajectories[pi, 0]
            for l in range(N_LAYERS):
                if l < 5:
                    total += add_projs[pi][l]
                else:
                    sign = np.sign(np.mean(projs_5))
                    total += sign * A_e * (l+1)**(-alpha_e)
            pred_D.append(total)
        except:
            pred_D.append(trajectories[pi, 5])
    err_D = np.abs(np.array(pred_D) - final_vals)
    methods["D_powerlaw_5layers"] = float(np.mean(err_D))

    print(f"\n  Final value range (std): {range_val:.4f}")
    print(f"\n  {'Method':<40} {'MeanErr':>8} {'Relative':>8} {'Layers':>6}")
    print(f"  {'-'*65}")
    print(f"  {'A: Linear extrap from L0-3':<40} {np.mean(err_A):8.4f} {np.mean(err_A)/range_val:8.4f} {'4':>6}")
    print(f"  {'B: Log extrap from L0-3':<40} {np.mean(err_B):8.4f} {np.mean(err_B)/range_val:8.4f} {'4':>6}")
    print(f"  {'C: Log+lin from 5 strategic layers':<40} {np.mean(err_C):8.4f} {np.mean(err_C)/range_val:8.4f} {'5':>6}")
    print(f"  {'D: Power-law from 5 calibration':<40} {np.mean(err_D):8.4f} {np.mean(err_D)/range_val:8.4f} {'5':>6}")

    methods["range"] = float(range_val)
    return methods


def test6_dirichlet(add_projs, add_norms, add_angles):
    """TEST 6: Construct Dirichlet polynomial, find zeros on Re(s)=1/2."""
    print("\n" + "=" * 80)
    print("TEST 6: Dirichlet Polynomial T(s) from Layer Additions")
    print("=" * 80)

    # Average complex coefficients: c_l = proj_l + i * norm_l * sin(angle_l)
    mean_projs = np.mean(add_projs, axis=0)
    mean_norms = np.mean(add_norms, axis=0)
    mean_angles = np.mean(add_angles, axis=0)

    # Construct complex coefficients
    # Use projection as real part, and orthogonal component as imaginary
    c_real = mean_projs
    c_imag = mean_norms * np.sin(np.radians(mean_angles))
    coeffs = c_real + 1j * c_imag

    print(f"\n  Dirichlet coefficients (layer, |c|, arg(c)):")
    for l in range(N_LAYERS):
        print(f"    L{l:2d}: |c| = {abs(coeffs[l]):8.4f}, arg = {np.degrees(np.angle(coeffs[l])):7.1f}°")

    # T(s) = Σ_{l=1}^{28} c_l * l^{-s}
    # Evaluate on critical line: s = 1/2 + it
    t_range = np.linspace(-20, 20, 2000)
    T_vals = np.zeros(len(t_range), dtype=complex)

    for l in range(N_LAYERS):
        n = l + 1  # 1-indexed
        for ti, t in enumerate(t_range):
            s = 0.5 + 1j * t
            T_vals[ti] += coeffs[l] * n ** (-s)

    T_abs = np.abs(T_vals)

    # Find zeros: local minima of |T(s)|
    zeros_t = []
    for i in range(1, len(T_abs) - 1):
        if T_abs[i] < T_abs[i-1] and T_abs[i] < T_abs[i+1] and T_abs[i] < np.median(T_abs) * 0.1:
            zeros_t.append(t_range[i])

    print(f"\n  Zeros of T(1/2 + it) found: {len(zeros_t)}")
    if zeros_t:
        for i, zt in enumerate(zeros_t[:15]):
            print(f"    t_{i+1} = {zt:.4f} (|T| = {np.interp(zt, t_range, T_abs):.6f})")

    # Zero spacing statistics
    result = {"n_zeros": len(zeros_t)}
    if len(zeros_t) > 2:
        spacings = np.diff(zeros_t)
        norm_sp = spacings / np.mean(spacings)
        sp_var = float(np.var(norm_sp))
        print(f"\n  Zero spacing statistics:")
        print(f"    Mean spacing: {np.mean(spacings):.4f}")
        print(f"    Normalized variance: {sp_var:.4f}")
        print(f"    GUE: 0.286, Poisson: 1.000")
        if abs(sp_var - 0.286) < abs(sp_var - 1.0):
            print(f"    → Closer to GUE")
        else:
            print(f"    → Closer to Poisson")
        result["spacing_var"] = sp_var

    return result


def main():
    print("=" * 80)
    print("PHASE 10z: TRANSFORMER AS ZETA FUNCTION")
    print("=" * 80)
    print(f"\nHypothesis: The transformer's layer computation is a Dirichlet-like")
    print(f"series with the same structure as the Riemann zeta function.")
    print(f"N_prompts = {len(PROMPTS)}, N_layers = {N_LAYERS}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    print("\nExtracting layer data...")
    trajectories, add_norms, add_angles, add_projs = extract_layer_data(model, tokenizer)

    r1 = test1_power_law(add_norms)
    r2 = test2_harmonics(add_projs)
    r3 = test3_ramanujan_fit(trajectories)
    r4 = test4_zeros(add_projs)
    r5 = test5_prediction(trajectories, add_projs)
    r6 = test6_dirichlet(add_projs, add_norms, add_angles)

    # Summary
    print("\n" + "=" * 80)
    print("VERDICT: IS THE TRANSFORMER A ZETA FUNCTION?")
    print("=" * 80)

    print(f"\n  T1 Power Law:   α = {r1['alpha']:.4f} vs 1/φ² = {r1['expected']:.4f} (R²={r1['r2']:.3f})")
    print(f"  T2 Harmonics:   Zeta energy = {r2['zeta_energy_pct']:.1f}%")
    print(f"  T3 Ramanujan:   R² log={r3['r2_log']:.3f} → ram={r3['r2_ramanujan']:.3f} → free={r3['r2_free']:.3f}")
    sp_var = r4.get('spacing_var', None)
    print(f"  T4 Zeros:       {r4['mean_zeros']:.1f}/prompt, spacing var={sp_var if sp_var else 'N/A'}")
    best_err = min(v for k,v in r5.items() if k != 'range')
    print(f"  T5 Prediction:  Best relative error = {best_err/r5['range']:.4f} from 5 layers")
    print(f"  T6 Dirichlet:   {r6['n_zeros']} zeros on critical line")

    results = {"test1": r1, "test2": r2, "test3": r3, "test4": r4, "test5": r5, "test6": r6}
    os.makedirs("results", exist_ok=True)
    with open("results/phase10z_zeta_transformer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/phase10z_zeta_transformer.json")


if __name__ == "__main__":
    main()
