#!/usr/bin/env python3
"""
Phase 10z2: Spectral Zeta — The Right Mapping
==============================================

Phase 10z showed the direct mapping (layer→term) fails: norms GROW.
But KK-1 showed singular values of the addition matrix DECAY as n^(-1/phi^2).

THE INSIGHT: The zeta connection is SPECTRAL, not term-by-term.
- Each layer adds MORE (norms grow)
- But the INDEPENDENT DIRECTIONS decay (SVs decay)
- The zeta function's convergence is spectral too (individual primes grow,
  but the series converges because of cancellation)

NEW TESTS:
----------
1. SVD of addition matrix → Dirichlet coefficients from singular values
2. Harmonic analysis of SINGULAR VECTORS (not raw norms)
3. Zeta function of the SV spectrum: Z(s) = Σ σ_k * k^(-s)
4. GUE test on SV-space zeros (not layer-space sign changes)
5. Ramanujan predictor for SV trajectory
6. Can the zeta solver predict the SVD structure from partial information?
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import math
from scipy.fft import fft
from scipy.optimize import curve_fit

MODEL_NAME = "Qwen/Qwen2.5-7B"
PHI = (1 + math.sqrt(5)) / 2
N_LAYERS = 28

PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the wave function",
    "Once upon a time in a land far away",
    "The derivative of sin(x) is",
    "def fibonacci(n):",
    "The relationship between energy and mass",
    "Yesterday I went to the store and",
    "According to the theory of relativity",
    "She opened the door and saw",
    "The fundamental theorem of calculus states",
    "import torch\nmodel = ",
    "To be or not to be, that is",
    "The mitochondria is the powerhouse of",
    "When the temperature drops below freezing",
    "In the beginning, there was",
    "The speed of light in a vacuum is approximately",
    "Water boils at 100 degrees Celsius at",
    "The largest planet in our solar system is",
    "Machine learning algorithms can be classified as",
    "The human body contains approximately",
    "In 1969, the first humans walked on",
    "The chemical formula for water is",
    "A binary search tree has the property that",
    "The periodic table organizes elements by their",
    "Shakespeare wrote the famous play",
    "The area of a circle is calculated by",
    "DNA stands for deoxyribonucleic",
    "The Pythagorean theorem states that",
    "Photosynthesis converts sunlight into",
    "The Great Wall of China was built to",
    "Neural networks consist of layers of",
    "The pH scale measures the acidity of",
    "In Python, a list comprehension is",
    "The Amazon rainforest is located in",
    "Gravity accelerates objects at approximately",
    "The Fourier transform converts signals from",
    "Antibiotics are used to treat infections caused by",
    "The Magna Carta was signed in the year",
    "A recursive function calls",
    "The boiling point of nitrogen is",
    "Entropy in thermodynamics measures the",
    "The Renaissance began in",
    "TCP/IP is the fundamental protocol of",
    "The Hubble telescope orbits the Earth at",
    "Plate tectonics explains how the Earth's",
    "A hash table provides average-case",
    "The human genome contains approximately",
    "Mozart composed his first symphony at",
    "The Fibonacci sequence begins with",
    "Dark matter makes up approximately",
]


def angle_deg(a, b):
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def main():
    print("=" * 80)
    print("PHASE 10z2: SPECTRAL ZETA — THE RIGHT MAPPING")
    print("=" * 80)
    print(f"\nInsight: The zeta connection is SPECTRAL, not term-by-term.")
    print(f"Layer norms grow, but singular values of the addition matrix decay.")
    print(f"N_prompts = {len(PROMPTS)}, N_layers = {N_LAYERS}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    # ================================================================
    # EXTRACT: Per-layer addition vectors (full 3584-dim)
    # ================================================================
    print("\nExtracting layer additions...")
    all_addition_matrices = []  # (N_prompts, N_layers, hidden_dim)
    all_attn_additions = []     # separate attn additions
    all_ffn_additions = []      # separate FFN additions

    for pi, prompt in enumerate(PROMPTS):
        print(f"\r  Prompt {pi+1}/{len(PROMPTS)}", end="", flush=True)

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        # Hooks
        layer_inputs = {}
        hooks = []
        for li in range(N_LAYERS):
            def make_hook(idx):
                def fn(mod, args):
                    layer_inputs[idx] = args[0].detach().clone()
                return fn
            hooks.append(model.model.layers[li].register_forward_pre_hook(make_hook(li)))

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

        additions = []
        attn_adds = []
        ffn_adds = []

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
                s2_lt = s2[0, -1]
                s3 = h_in + s2

                s4 = layer.post_attention_layernorm(s3)
                s5 = layer.mlp(s4)
                s5_lt = s5[0, -1]
                s6 = s3 + s5
                s6_lt = s6[0, -1]

                total_add = s6_lt - h_in_lt
                additions.append(total_add.numpy())
                attn_adds.append(s2_lt.numpy())
                ffn_adds.append(s5_lt.numpy())

        all_addition_matrices.append(np.array(additions))    # (28, 3584)
        all_attn_additions.append(np.array(attn_adds))
        all_ffn_additions.append(np.array(ffn_adds))

    print()

    # ================================================================
    # TEST 1: SVD of Addition Matrix — The Dirichlet Spectrum
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 1: SVD of Addition Matrix — The Dirichlet Spectrum")
    print("=" * 80)

    all_sv_spectra = []
    all_rank99 = []

    for pi in range(len(PROMPTS)):
        A = all_addition_matrices[pi]  # (28, 3584)
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        all_sv_spectra.append(S)

        # Effective rank at 99%
        total = np.sum(S**2)
        cumsum = np.cumsum(S**2)
        rank99 = np.searchsorted(cumsum, 0.99 * total) + 1
        all_rank99.append(rank99)

    sv_mean = np.mean(all_sv_spectra, axis=0)  # (28,)
    rank99_mean = np.mean(all_rank99)

    print(f"\n  Mean effective rank (99%): {rank99_mean:.1f} out of {N_LAYERS}")

    # Power law fit: σ_k ~ A * k^(-alpha)
    ks = np.arange(1, len(sv_mean) + 1)
    log_k = np.log(ks)
    log_sv = np.log(sv_mean + 1e-10)

    valid = np.isfinite(log_sv)
    z = np.polyfit(log_k[valid], log_sv[valid], 1)
    alpha_sv = -z[0]
    A_sv = np.exp(z[1])
    predicted = z[0] * log_k[valid] + z[1]
    ss_res = np.sum((log_sv[valid] - predicted)**2)
    ss_tot = np.sum((log_sv[valid] - np.mean(log_sv[valid]))**2)
    r2_sv = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  SV power law: σ_k ~ {A_sv:.2f} * k^(-{alpha_sv:.4f})")
    print(f"  R² = {r2_sv:.4f}")
    print(f"  1/φ² = {1/PHI**2:.4f} (expected from KK-1)")
    print(f"  Ratio α/(1/φ²) = {alpha_sv / (1/PHI**2):.4f}")

    print(f"\n  Singular value spectrum:")
    for k in range(min(N_LAYERS, 28)):
        bar = "█" * int(sv_mean[k] / sv_mean[0] * 40)
        print(f"    σ_{k+1:2d} = {sv_mean[k]:10.2f} {bar}")

    # ================================================================
    # TEST 2: Harmonic Analysis of Left Singular Vectors
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Harmonic Content of Singular Vectors (Layer Patterns)")
    print("=" * 80)

    # The left singular vectors U[:, k] are 28-dim vectors showing
    # how each layer contributes to the k-th mode.
    # FFT of these vectors reveals the harmonic structure.
    all_U = []
    for pi in range(len(PROMPTS)):
        A = all_addition_matrices[pi]
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        all_U.append(U)

    mean_U = np.mean(all_U, axis=0)  # (28, 28)

    print(f"\n  FFT of first 5 left singular vectors (layer patterns):")
    zeta_energy_per_mode = []
    for mode in range(5):
        u_k = mean_U[:, mode]
        fft_u = np.abs(fft(u_k))
        max_f = max(fft_u[1:N_LAYERS//2+1])

        zeta_ks = [3, 6, 9, 12, 13]  # 13 = alias of 15
        zeta_e = sum(fft_u[k]**2 for k in zeta_ks if k <= N_LAYERS//2)
        total_e = sum(fft_u[k]**2 for k in range(1, N_LAYERS//2+1))
        zeta_pct = zeta_e / total_e * 100 if total_e > 0 else 0
        zeta_energy_per_mode.append(zeta_pct)

        print(f"\n    Mode {mode+1} (σ={sv_mean[mode]:.1f}):")
        dominant_k = np.argmax(fft_u[1:N_LAYERS//2+1]) + 1
        print(f"      Dominant harmonic: k={dominant_k}")
        print(f"      Zeta harmonic energy: {zeta_pct:.1f}%")
        for k in range(N_LAYERS//2 + 1):
            rel = fft_u[k] / max_f if max_f > 0 else 0
            bar = "█" * int(rel * 25)
            note = " ← Z" if k in [3,6,9,12,13] else ""
            if rel > 0.3:
                print(f"        k={k:2d}: {fft_u[k]:6.3f} ({rel:4.2f}) {bar}{note}")

    # ================================================================
    # TEST 3: Zeta Function of the SV Spectrum
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Zeta Function of SV Spectrum Z(s) = Σ σ_k * k^(-s)")
    print("=" * 80)

    # Construct Z(s) using SINGULAR VALUES as Dirichlet coefficients
    # These decay, so the series converges for Re(s) > 0
    t_range = np.linspace(-15, 15, 2000)
    Z_vals = np.zeros(len(t_range), dtype=complex)

    for k in range(N_LAYERS):
        n = k + 1
        for ti, t in enumerate(t_range):
            s = 0.5 + 1j * t
            Z_vals[ti] += sv_mean[k] * n**(-s)

    Z_abs = np.abs(Z_vals)

    # Find zeros (local minima below threshold)
    threshold = np.median(Z_abs) * 0.05
    zeros_t = []
    for i in range(1, len(Z_abs) - 1):
        if Z_abs[i] < Z_abs[i-1] and Z_abs[i] < Z_abs[i+1] and Z_abs[i] < threshold:
            # Refine with parabolic interpolation
            a, b, c = Z_abs[i-1], Z_abs[i], Z_abs[i+1]
            dt = t_range[1] - t_range[0]
            offset = 0.5 * (a - c) / (a - 2*b + c + 1e-20)
            zeros_t.append(t_range[i] + offset * dt)

    print(f"\n  Z(1/2 + it) zeros found: {len(zeros_t)}")
    print(f"  (Threshold: |Z| < {threshold:.2f})")

    sv_zero_spacings = None
    if zeros_t:
        print(f"\n  Zeros:")
        for i, zt in enumerate(zeros_t[:20]):
            z_val = np.interp(zt, t_range, Z_abs)
            print(f"    t_{i+1:2d} = {zt:8.4f}  |Z| = {z_val:.6f}")

        if len(zeros_t) > 2:
            spacings = np.diff(zeros_t)
            norm_sp = spacings / np.mean(spacings)
            sv_zero_var = float(np.var(norm_sp))
            sv_zero_spacings = spacings.tolist()

            print(f"\n  Spacing statistics:")
            print(f"    Mean spacing: {np.mean(spacings):.4f}")
            print(f"    Normalized variance: {sv_zero_var:.4f}")
            print(f"    GUE: 0.286, Poisson: 1.000")
            if abs(sv_zero_var - 0.286) < abs(sv_zero_var - 1.0):
                print(f"    → CLOSER TO GUE (zero repulsion, like zeta!)")
            else:
                print(f"    → Closer to Poisson")

    # ================================================================
    # TEST 4: The Three-Zone Harmonic Structure
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Three-Zone SVD — Compressor / Processor / Targeter")
    print("=" * 80)

    # Decompose by zone and compare SV spectra
    zones = {
        "Compressor (L0-3)": slice(0, 4),
        "Processor (L4-25)": slice(4, 26),
        "Targeter (L26-27)": slice(26, 28),
    }

    zone_spectra = {}
    for name, slc in zones.items():
        zone_svs = []
        for pi in range(len(PROMPTS)):
            A_zone = all_addition_matrices[pi][slc, :]
            _, S_z, _ = np.linalg.svd(A_zone, full_matrices=False)
            zone_svs.append(S_z)

        mean_svs = np.mean(zone_svs, axis=0)
        zone_spectra[name] = mean_svs

        # Power law fit
        n_svs = len(mean_svs)
        ks_z = np.arange(1, n_svs + 1)
        log_sv_z = np.log(mean_svs + 1e-10)
        valid_z = np.isfinite(log_sv_z)
        if np.sum(valid_z) > 2:
            z_fit = np.polyfit(np.log(ks_z[valid_z]), log_sv_z[valid_z], 1)
            alpha_z = -z_fit[0]
            pred_z = z_fit[0] * np.log(ks_z[valid_z]) + z_fit[1]
            ss_r = np.sum((log_sv_z[valid_z] - pred_z)**2)
            ss_t = np.sum((log_sv_z[valid_z] - np.mean(log_sv_z[valid_z]))**2)
            r2_z = 1 - ss_r / ss_t if ss_t > 0 else 0
        else:
            alpha_z = 0
            r2_z = 0

        print(f"\n  {name}: {n_svs} SVs")
        print(f"    Power law: α = {alpha_z:.4f} (R² = {r2_z:.4f})")
        print(f"    Top 5 SVs: {', '.join(f'{s:.1f}' for s in mean_svs[:5])}")
        print(f"    Energy concentration (σ1/Σ): {mean_svs[0]**2 / np.sum(mean_svs**2) * 100:.1f}%")

    # ================================================================
    # TEST 5: Cross-Zone Angle Structure
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Cross-Zone Principal Directions — Do Zones Share Geometry?")
    print("=" * 80)

    # Get top singular vector from each zone for each prompt
    all_zone_dirs = {name: [] for name in zones}
    for pi in range(len(PROMPTS)):
        for name, slc in zones.items():
            A_zone = all_addition_matrices[pi][slc, :]
            _, _, Vh_z = np.linalg.svd(A_zone, full_matrices=False)
            all_zone_dirs[name].append(Vh_z[0])  # Top right SV

    # Mean directions
    mean_dirs = {}
    for name in zones:
        dirs = np.array(all_zone_dirs[name])
        # Sign-align before averaging
        ref = dirs[0]
        for i in range(len(dirs)):
            if np.dot(dirs[i], ref) < 0:
                dirs[i] = -dirs[i]
        mean_dirs[name] = np.mean(dirs, axis=0)
        mean_dirs[name] /= np.linalg.norm(mean_dirs[name])

    zone_names = list(zones.keys())
    print(f"\n  Angles between top singular vectors of each zone:")
    for i in range(len(zone_names)):
        for j in range(i+1, len(zone_names)):
            cos = np.dot(mean_dirs[zone_names[i]], mean_dirs[zone_names[j]])
            ang = np.degrees(np.arccos(np.clip(abs(cos), 0, 1)))
            print(f"    {zone_names[i]} ↔ {zone_names[j]}: {ang:.1f}°")

    # ================================================================
    # TEST 6: Ramanujan Predictor for SV Trajectory
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 6: Ramanujan Predictor for SV Spectrum")
    print("=" * 80)

    # Can we fit σ_k to the Ramanujan form?
    # σ_k ≈ A * k^(-alpha) + harmonic corrections
    ks_fit = np.arange(1, N_LAYERS + 1).astype(float)

    # Model 1: Pure power law
    def power_model(k, A, alpha):
        return A * k**(-alpha)

    try:
        popt, _ = curve_fit(power_model, ks_fit, sv_mean, p0=[sv_mean[0], 0.5])
        pred_pow = power_model(ks_fit, *popt)
        ss_res = np.sum((sv_mean - pred_pow)**2)
        ss_tot = np.sum((sv_mean - np.mean(sv_mean))**2)
        r2_pow = 1 - ss_res / ss_tot
        print(f"\n  Model 1 (power law): σ_k ~ {popt[0]:.1f} * k^(-{popt[1]:.4f}), R² = {r2_pow:.4f}")
    except Exception as e:
        r2_pow = 0
        print(f"\n  Model 1: FIT FAILED ({e})")

    # Model 2: Power law + zeta harmonics
    def ram_sv_model(k, A, alpha, h3, h6, h9, h12):
        base = A * k**(-alpha)
        theta = 2 * np.pi * k / N_LAYERS
        harmonics = h3*np.sin(3*theta) + h6*np.sin(6*theta) + h9*np.sin(9*theta) + h12*np.sin(12*theta)
        return base + harmonics

    try:
        popt2, _ = curve_fit(ram_sv_model, ks_fit, sv_mean,
                            p0=[sv_mean[0], 0.5, 0, 0, 0, 0], maxfev=10000)
        pred_ram = ram_sv_model(ks_fit, *popt2)
        ss_res = np.sum((sv_mean - pred_ram)**2)
        r2_ram = 1 - ss_res / ss_tot
        print(f"  Model 2 (Ramanujan): R² = {r2_ram:.4f}")
        print(f"    α = {popt2[1]:.4f}")
        print(f"    h3={popt2[2]:.2f}, h6={popt2[3]:.2f}, h9={popt2[4]:.2f}, h12={popt2[5]:.2f}")
        h_str = {3: abs(popt2[2]), 6: abs(popt2[3]), 9: abs(popt2[4]), 12: abs(popt2[5])}
        strongest = max(h_str, key=h_str.get)
        print(f"    Strongest harmonic: k={strongest} (|h|={h_str[strongest]:.2f})")
        print(f"    Harmonic improvement: +{r2_ram - r2_pow:.4f}")
    except Exception as e:
        r2_ram = 0
        print(f"  Model 2: FIT FAILED ({e})")

    # Model 3: Power law + ALL harmonics
    def free_sv_model(k, *params):
        A, alpha = params[:2]
        base = A * k**(-alpha)
        theta = 2 * np.pi * k / N_LAYERS
        harmonics = sum(params[2+i] * np.sin((i+1)*theta) for i in range(min(13, len(params)-2)))
        return base + harmonics

    try:
        p0 = [sv_mean[0], 0.5] + [0]*13
        popt3, _ = curve_fit(free_sv_model, ks_fit, sv_mean, p0=p0, maxfev=20000)
        pred_free = free_sv_model(ks_fit, *popt3)
        ss_res = np.sum((sv_mean - pred_free)**2)
        r2_free = 1 - ss_res / ss_tot
        print(f"  Model 3 (all harmonics): R² = {r2_free:.4f}")

        h_all = {i+1: abs(popt3[2+i]) for i in range(13)}
        max_h = max(h_all.values()) if max(h_all.values()) > 0 else 1
        for k in range(1, 14):
            bar = "█" * int(h_all[k] / max_h * 25)
            note = " ← ZETA" if k in [3,6,9,12] else ""
            note = " ← alias(15)" if k == 13 else note
            print(f"    k={k:2d}: {h_all[k]:8.2f} {bar}{note}")

        zeta_e = sum(h_all[k]**2 for k in [3, 6, 9, 12])
        total_e = sum(v**2 for v in h_all.values())
        print(f"    Zeta harmonic energy: {zeta_e/total_e*100:.1f}%")
    except Exception as e:
        r2_free = 0
        print(f"  Model 3: FIT FAILED ({e})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("VERDICT: SPECTRAL ZETA CONNECTION")
    print("=" * 80)

    print(f"\n  T1 SV Power Law:  α = {alpha_sv:.4f}, 1/φ² = {1/PHI**2:.4f}, R² = {r2_sv:.4f}")
    print(f"     Rank(99%) = {rank99_mean:.1f} out of {N_LAYERS}")
    print(f"  T2 SV Harmonics:  Zeta energy per mode: {', '.join(f'{z:.1f}%' for z in zeta_energy_per_mode)}")
    if zeros_t:
        print(f"  T3 SV Zeta Zeros: {len(zeros_t)} zeros found")
        if sv_zero_spacings and len(sv_zero_spacings) > 1:
            print(f"     Spacing var = {np.var(np.array(sv_zero_spacings)/np.mean(sv_zero_spacings)):.4f}")
    else:
        print(f"  T3 SV Zeta Zeros: No zeros found")
    print(f"  T4 Three Zones:   Separate SV spectra computed")
    print(f"  T5 Cross-Zone:    Direction angles computed")
    print(f"  T6 Ramanujan:     Power R²={r2_pow:.3f} → +zeta={r2_ram:.3f} → +all={r2_free:.3f}")

    # Save
    results = {
        "test1_sv_power_law": {
            "alpha": float(alpha_sv),
            "r2": float(r2_sv),
            "rank99_mean": float(rank99_mean),
            "sv_spectrum": sv_mean.tolist(),
        },
        "test2_sv_harmonics": {
            "zeta_energy_per_mode": [float(z) for z in zeta_energy_per_mode],
        },
        "test3_sv_zeta": {
            "n_zeros": len(zeros_t),
            "zero_positions": zeros_t[:20] if zeros_t else [],
            "spacings": sv_zero_spacings if sv_zero_spacings else [],
        },
        "test6_ramanujan": {
            "r2_power": float(r2_pow) if r2_pow else 0,
            "r2_ramanujan": float(r2_ram) if r2_ram else 0,
            "r2_free": float(r2_free) if r2_free else 0,
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/phase10z2_spectral_zeta.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/phase10z2_spectral_zeta.json")


if __name__ == "__main__":
    main()
