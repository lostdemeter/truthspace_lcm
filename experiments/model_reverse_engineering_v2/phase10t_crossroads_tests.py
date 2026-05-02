"""
Phase 10t-Crossroads: Six Hyperdimensional Tests

Tests derived from comparative analysis with Kaluza-Klein theory,
Hoagland's hyperdimensional physics, and Haramein's hypergeometry.

TEST KK-1: Singular value decay law — does SV spectrum follow power law?
TEST KK-2: Spring softness ∝ information content?
TEST H-1:  Is 70.53° (arccos(1/3), tetrahedral face angle) stable?
TEST H-2:  Does L0's output form a tetrahedral frame with its components?
TEST NH-1: Holographic vs volumetric rank scaling?
TEST NH-2: Self-similar sub-zones in Refiner?
"""

import torch
import json
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
REFINER_LAYERS = list(range(4, 18))
ALL_LAYERS = list(range(28))

# 50 prompts for statistical power on H-1
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

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
TETRAHEDRAL_FACE = math.degrees(math.acos(1/3))  # 70.53°
TETRAHEDRAL_BOND = 180 - TETRAHEDRAL_FACE          # 109.47°
TETRAHEDRAL_LAT = math.degrees(math.asin(1/3))    # 19.47°


def angle(a, b):
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def effective_rank(singular_values, threshold=0.99):
    total = (singular_values ** 2).sum()
    cumsum = (singular_values ** 2).cumsum(0)
    return (cumsum < threshold * total).sum().item() + 1


def main():
    print("=" * 80)
    print("  PHASE 10t-CROSSROADS: SIX HYPERDIMENSIONAL TESTS")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    # Storage
    all_attn_successive = []  # for H-1
    all_ffn_successive = []
    all_cross_angles = []
    all_l0_tetra = []         # for H-2
    all_refiner_additions = {n: {"attn": [], "ffn": []} for n in [4, 7, 10, 14]}  # for NH-1
    per_layer_sigs = {li: [] for li in REFINER_LAYERS}  # for NH-2
    per_layer_spring = {li: [] for li in ALL_LAYERS}  # for KK-2
    per_layer_info = {li: [] for li in ALL_LAYERS}  # for KK-2
    sv_spectra = []  # for KK-1

    for pi, prompt in enumerate(PROMPTS):
        if pi % 10 == 0:
            print(f"  Prompt {pi}/{len(PROMPTS)}")

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        # Hooks to capture layer inputs
        layer_inputs = {}
        hooks = []
        for li in ALL_LAYERS:
            def make_hook(idx):
                def fn(mod, args):
                    layer_inputs[idx] = args[0].detach().clone()
                return fn
            hooks.append(model.model.layers[li].register_forward_pre_hook(make_hook(li)))

        # Capture final layer output
        def out_hook(mod, args, output):
            layer_inputs["final_out"] = output.detach().clone()
        hooks.append(model.model.layers[27].register_forward_hook(out_hook))

        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        # Precompute attention infrastructure
        with torch.no_grad():
            h_embed = model.model.embed_tokens(input_ids)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_embed, position_ids)
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # ============================================================
        # DISSECT ALL LAYERS
        # ============================================================
        prev_attn_add = None
        prev_ffn_add = None
        refiner_attn_adds = []
        refiner_ffn_adds = []

        with torch.no_grad():
            for li in ALL_LAYERS:
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
                s3_lt = s3[0, -1]

                s4 = layer.post_attention_layernorm(s3)
                s5 = layer.mlp(s4)
                s5_lt = s5[0, -1]
                s6 = s3 + s5
                s6_lt = s6[0, -1]

                # Spring stiffness
                k1 = h_in_lt.norm().item() / s3_lt.norm().item() if s3_lt.norm().item() > 0 else 1.0
                per_layer_spring[li].append(k1)

                # Info content: norm of addition relative to residual
                attn_info = s2_lt.norm().item() / h_in_lt.norm().item() if h_in_lt.norm().item() > 0 else 0
                ffn_info = s5_lt.norm().item() / s3_lt.norm().item() if s3_lt.norm().item() > 0 else 0
                per_layer_info[li].append(attn_info + ffn_info)

                if li in REFINER_LAYERS:
                    refiner_attn_adds.append(s2_lt)
                    refiner_ffn_adds.append(s5_lt)

                    idx = li - 4  # 0-indexed within refiner

                    # NH-2: Per-layer signature
                    sig = {
                        "cos_in_a": cos_sim(h_in_lt, s2_lt),
                        "cos_in_f": cos_sim(h_in_lt, s5_lt),
                        "cos_a_f": cos_sim(s2_lt, s5_lt),
                        "k1": k1,
                        "attn_norm_ratio": s2_lt.norm().item() / h_in_lt.norm().item(),
                        "ffn_norm_ratio": s5_lt.norm().item() / h_in_lt.norm().item(),
                        "total_rotation": angle(h_in_lt, s6_lt),
                    }
                    per_layer_sigs[li].append(sig)

                    # H-1: Successive angles
                    if prev_attn_add is not None and li in REFINER_LAYERS:
                        all_attn_successive.append(angle(s2_lt, prev_attn_add))
                        all_ffn_successive.append(angle(s5_lt, prev_ffn_add))
                    if li in REFINER_LAYERS:
                        all_cross_angles.append(angle(s2_lt, s5_lt))

                    prev_attn_add = s2_lt
                    prev_ffn_add = s5_lt

                # H-2: L0 tetrahedral test
                if li == 0:
                    accumulated = s6_lt  # h_in + attn + ffn
                    ang_acc_in = angle(accumulated, h_in_lt)
                    ang_acc_attn = angle(accumulated, s2_lt)
                    ang_acc_ffn = angle(accumulated, s5_lt)
                    ang_in_attn = angle(h_in_lt, s2_lt)
                    ang_in_ffn = angle(h_in_lt, s5_lt)
                    ang_attn_ffn = angle(s2_lt, s5_lt)
                    all_l0_tetra.append({
                        "acc_in": ang_acc_in, "acc_attn": ang_acc_attn,
                        "acc_ffn": ang_acc_ffn, "in_attn": ang_in_attn,
                        "in_ffn": ang_in_ffn, "attn_ffn": ang_attn_ffn,
                    })

        # NH-1: Rank for subsets
        for n_layers in [4, 7, 10, 14]:
            attn_mat = torch.stack(refiner_attn_adds[:n_layers])
            ffn_mat = torch.stack(refiner_ffn_adds[:n_layers])
            all_mat = torch.cat([attn_mat, ffn_mat], dim=0)
            _, S, _ = torch.linalg.svd(all_mat, full_matrices=False)
            all_refiner_additions[n_layers]["rank99"] = effective_rank(S, 0.99)
            all_refiner_additions[n_layers]["rank95"] = effective_rank(S, 0.95)
            all_refiner_additions[n_layers]["n_vectors"] = all_mat.shape[0]

        # KK-1: SV spectrum of full refiner additions
        all_mat = torch.cat([torch.stack(refiner_attn_adds), torch.stack(refiner_ffn_adds)], dim=0)
        _, S, _ = torch.linalg.svd(all_mat, full_matrices=False)
        sv_spectra.append(S.numpy())

    # ================================================================
    # ANALYSIS
    # ================================================================

    print("\n" + "=" * 60)
    print("  TEST KK-1: SINGULAR VALUE DECAY LAW")
    print("=" * 60)

    mean_sv = np.mean(sv_spectra, axis=0)
    mean_sv_norm = mean_sv / mean_sv[0]  # normalize to first SV

    print(f"\n  Reference: arccos(1/3) = {TETRAHEDRAL_FACE:.2f}°")
    print(f"\n  Normalized SV spectrum (mean across {len(PROMPTS)} prompts):")
    for i in range(min(15, len(mean_sv_norm))):
        bar = "█" * int(mean_sv_norm[i] * 40)
        print(f"    SV[{i+1:2d}]: {mean_sv_norm[i]:.4f}  {bar}")

    # Fit power law: S_n = S_1 * n^(-alpha)
    indices = np.arange(1, len(mean_sv_norm) + 1, dtype=float)
    # Log-log fit (exclude first point)
    log_n = np.log(indices[1:15])
    log_s = np.log(mean_sv_norm[1:15])
    alpha, log_c = np.polyfit(log_n, log_s, 1)
    c = np.exp(log_c)
    print(f"\n  Power law fit (SV 2-15): S_n = {c:.3f} * n^({alpha:.3f})")
    print(f"  Decay exponent α = {-alpha:.3f}")

    # Check if α relates to φ
    print(f"  α vs 1/φ = {1/PHI:.3f}")
    print(f"  α vs 1/2 = 0.500")
    print(f"  α vs 1   = 1.000")

    # Fit quality
    predicted = c * indices[1:15] ** alpha
    residuals = log_s - np.log(predicted)
    r_squared = 1 - np.sum(residuals**2) / np.sum((log_s - log_s.mean())**2)
    print(f"  R² = {r_squared:.4f}")

    print("\n" + "=" * 60)
    print("  TEST KK-2: SPRING SOFTNESS ∝ INFORMATION CONTENT")
    print("=" * 60)

    layers_k = []
    layers_info = []
    layers_softness = []
    for li in ALL_LAYERS:
        mean_k = np.mean(per_layer_spring[li])
        mean_info = np.mean(per_layer_info[li])
        softness = 1 - mean_k
        layers_k.append(mean_k)
        layers_info.append(mean_info)
        layers_softness.append(softness)

    # Correlation between softness and info
    corr = np.corrcoef(layers_softness, layers_info)[0, 1]
    print(f"\n  Correlation(1-k₁, info_content): {corr:.4f}")

    print(f"\n  Per-layer:")
    print(f"  {'Layer':>5} | {'k₁':>6} | {'1-k₁':>6} | {'Info':>8}")
    print("  " + "-" * 32)
    for li in ALL_LAYERS:
        print(f"  L{li:>3} | {layers_k[li]:>6.3f} | {layers_softness[li]:>6.3f} | {layers_info[li]:>8.4f}")

    print("\n" + "=" * 60)
    print(f"  TEST H-1: IS {TETRAHEDRAL_FACE:.2f}° A STABLE CONSTANT?")
    print("=" * 60)

    attn_arr = np.array(all_attn_successive)
    ffn_arr = np.array(all_ffn_successive)
    cross_arr = np.array(all_cross_angles)

    print(f"\n  Attention successive angles (N={len(attn_arr)}):")
    print(f"    Mean:   {attn_arr.mean():.2f}°")
    print(f"    Median: {np.median(attn_arr):.2f}°")
    print(f"    Std:    {attn_arr.std():.2f}°")
    print(f"    SEM:    {attn_arr.std() / np.sqrt(len(attn_arr)):.2f}°")
    print(f"    Target: arccos(1/3) = {TETRAHEDRAL_FACE:.2f}°")
    diff = abs(attn_arr.mean() - TETRAHEDRAL_FACE)
    sem = attn_arr.std() / np.sqrt(len(attn_arr))
    z_score = diff / sem if sem > 0 else float('inf')
    print(f"    |Δ| = {diff:.2f}°, z-score = {z_score:.1f}")
    if z_score < 2:
        print(f"    → CONSISTENT with arccos(1/3) (z < 2)")
    else:
        print(f"    → NOT arccos(1/3) (z > 2, statistically different)")

    print(f"\n  FFN successive angles (N={len(ffn_arr)}):")
    print(f"    Mean: {ffn_arr.mean():.2f}°, Std: {ffn_arr.std():.2f}°")
    print(f"    Target: 90° (orthogonal)")
    print(f"    |Δ| from 90° = {abs(ffn_arr.mean() - 90):.2f}°")

    print(f"\n  Cross angles attn↔FFN (N={len(cross_arr)}):")
    print(f"    Mean: {cross_arr.mean():.2f}°, Std: {cross_arr.std():.2f}°")
    print(f"    Target: 109.47° (tetrahedral bond)")
    diff2 = abs(cross_arr.mean() - TETRAHEDRAL_BOND)
    sem2 = cross_arr.std() / np.sqrt(len(cross_arr))
    z2 = diff2 / sem2 if sem2 > 0 else float('inf')
    print(f"    |Δ| = {diff2:.2f}°, z = {z2:.1f}")

    print("\n" + "=" * 60)
    print("  TEST H-2: TETRAHEDRAL FRAME AT L0?")
    print("=" * 60)

    # For a tetrahedral frame, all 6 pairwise angles should be ~109.47°
    # (or the angles between 4 directions pointing from center to tetrahedron vertices)
    # Actually for 4 directions from center of tetrahedron: arccos(-1/3) = 109.47°
    print(f"\n  Tetrahedral reference: arccos(-1/3) = {TETRAHEDRAL_BOND:.2f}°")
    print(f"  Orthogonal reference: 90.00°")

    angle_keys = ["acc_in", "acc_attn", "acc_ffn", "in_attn", "in_ffn", "attn_ffn"]
    angle_labels = ["Accumulated↔Input", "Accumulated↔Attn", "Accumulated↔FFN",
                    "Input↔Attn", "Input↔FFN", "Attn↔FFN"]

    print(f"\n  L0 pairwise angles (4 vectors: input, attn, ffn, accumulated):")
    for key, label in zip(angle_keys, angle_labels):
        vals = [t[key] for t in all_l0_tetra]
        mean_ang = np.mean(vals)
        std_ang = np.std(vals)
        diff_tetra = abs(mean_ang - TETRAHEDRAL_BOND)
        diff_ortho = abs(mean_ang - 90)
        closer = "TETRAHEDRAL" if diff_tetra < diff_ortho else "ORTHOGONAL"
        print(f"    {label:>22}: {mean_ang:>6.1f}° ± {std_ang:.1f}°  → closer to {closer}")

    print("\n" + "=" * 60)
    print("  TEST NH-1: HOLOGRAPHIC vs VOLUMETRIC SCALING")
    print("=" * 60)

    print(f"\n  Rank(99%) vs number of Refiner layers included:")
    print(f"  {'N_layers':>8} | {'N_vectors':>9} | {'Rank(99%)':>9} | {'Rank(95%)':>9} | {'Rank/N_vec':>10}")
    print("  " + "-" * 52)

    n_layers_list = [4, 7, 10, 14]
    ranks_99 = []
    n_vecs = []
    for n in n_layers_list:
        r99 = all_refiner_additions[n]["rank99"]
        r95 = all_refiner_additions[n]["rank95"]
        nv = all_refiner_additions[n]["n_vectors"]
        ratio = r99 / nv
        ranks_99.append(r99)
        n_vecs.append(nv)
        print(f"  {n:>8} | {nv:>9} | {r99:>9.1f} | {r95:>9.1f} | {ratio:>10.3f}")

    # Fit: Rank = a * N^b
    log_nv = np.log(n_vecs)
    log_r = np.log(ranks_99)
    b_fit, log_a = np.polyfit(log_nv, log_r, 1)
    print(f"\n  Power law fit: Rank(99%) = {np.exp(log_a):.2f} × N^{b_fit:.3f}")
    print(f"    b = 1.0 → VOLUMETRIC (each vector adds 1 rank)")
    print(f"    b < 1.0 → HOLOGRAPHIC (sub-linear growth)")
    print(f"    b > 1.0 → SUPER-LINEAR (unlikely)")
    print(f"    Measured b = {b_fit:.3f}")

    if b_fit < 0.95:
        print(f"    → HOLOGRAPHIC scaling (b = {b_fit:.3f} < 1)")
    elif b_fit > 1.05:
        print(f"    → SUPER-LINEAR scaling (b = {b_fit:.3f} > 1)")
    else:
        print(f"    → VOLUMETRIC scaling (b = {b_fit:.3f} ≈ 1)")

    print("\n" + "=" * 60)
    print("  TEST NH-2: SELF-SIMILAR SUB-ZONES IN REFINER?")
    print("=" * 60)

    print(f"\n  Per-layer mechanical signatures (mean across {len(PROMPTS)} prompts):")
    print(f"  {'Layer':>5} | {'cos(i,a)':>8} | {'cos(i,f)':>8} | {'cos(a,f)':>8} | {'k₁':>6} | {'rot°':>6} | Zone?")
    print("  " + "-" * 62)

    for li in REFINER_LAYERS:
        sigs = per_layer_sigs[li]
        m_ia = np.mean([s["cos_in_a"] for s in sigs])
        m_if = np.mean([s["cos_in_f"] for s in sigs])
        m_af = np.mean([s["cos_a_f"] for s in sigs])
        m_k = np.mean([s["k1"] for s in sigs])
        m_rot = np.mean([s["total_rotation"] for s in sigs])

        # Classify sub-zone
        zone = ""
        if abs(m_if) > 0.15:
            zone = "AIM?" if m_if > 0 else "CORRECT?"
        elif m_af < -0.3:
            zone = "FIRE?"
        elif m_k > 0.87:
            zone = "REFINE"
        else:
            zone = "CREATE?" if li == 4 else "REFINE"

        print(f"  L{li:>3} | {m_ia:>+8.3f} | {m_if:>+8.3f} | {m_af:>+8.3f} | {m_k:>6.3f} | {m_rot:>5.1f}° | {zone}")

    # ================================================================
    # SUMMARY
    # ================================================================

    print("\n" + "=" * 60)
    print("  SUMMARY OF SIX TESTS")
    print("=" * 60)

    print(f"""
  KK-1 (SV decay):         α = {-alpha:.3f} (power law exponent)
  KK-2 (Spring↔Info):      correlation = {corr:.3f}
  H-1  (70.53° test):      mean = {attn_arr.mean():.2f}°, z = {z_score:.1f}
  H-2  (L0 tetrahedron):   see angles above
  NH-1 (Holographic):      b = {b_fit:.3f} scaling exponent
  NH-2 (Self-similarity):  see sub-zone table above
""")

    # Save
    save_data = {
        "kk1_alpha": float(-alpha),
        "kk1_r_squared": float(r_squared),
        "kk2_correlation": float(corr),
        "h1_attn_mean": float(attn_arr.mean()),
        "h1_attn_std": float(attn_arr.std()),
        "h1_attn_z_vs_70_53": float(z_score),
        "h1_cross_mean": float(cross_arr.mean()),
        "h1_cross_z_vs_109_47": float(z2),
        "nh1_scaling_exponent": float(b_fit),
        "tetrahedral_face": TETRAHEDRAL_FACE,
        "tetrahedral_bond": TETRAHEDRAL_BOND,
        "sv_spectrum_normalized": mean_sv_norm.tolist(),
    }
    out_path = "results/phase10t_crossroads.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {out_path}")

    print("\n" + "=" * 80)
    print("  PHASE 10t-CROSSROADS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
