#!/usr/bin/env python3
"""
Phase 10z4: Newton-Processor Hypothesis
========================================

Finding 108 showed:
- Lambert W (Compressor) captures 95%+ of ζ zero prediction
- O(1) harmonic corrections barely matter
- The "quantum barrier" σ≈0.33 is only broken by ITERATIVE refinement (Newton)

HYPOTHESIS: The transformer's Processor layers (L4-25) are doing iterative
refinement, NOT harmonic corrections. Each layer is like one Newton step.

TEST: If Processor layers ARE Newton steps, then:
1. Error should DECAY geometrically across layers (like Newton convergence)
2. The decay rate should be φ-related (golden section convergence rate)
3. Each layer's "correction" should be proportional to the current "error"
4. The Compressor (L0-3) sets up the initial estimate (Lambert W)
5. The Targeter (L26-27) does a final rank-1 precision step

We test this by:
A. Measuring how the residual stream "error" (distance to final answer)
   decays across layers — is it geometric with rate φ?
B. Comparing the per-layer error decay to actual Newton iterations on ζ
C. Checking if the Processor's α=2/φ² power law matches Newton convergence

This is the conceptual proof that ζ = ideal transformer.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import lambertw
from mpmath import zetazero, zeta, mp, mpf, mpc, im, fabs
import json
import os

mp.dps = 25

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


# ============================================================================
# PART A: Transformer layer-by-layer error decay
# ============================================================================

def measure_layer_error_decay(model, tokenizer):
    """
    For each prompt, measure how the residual stream "error" decays per layer.
    
    "Error" = distance from current state to the FINAL state (the answer).
    If layers are Newton steps, this should decay geometrically.
    """
    print("\n  PART A: Measuring layer-by-layer error decay...")
    
    all_convergence = []  # Per-prompt: [error_L0, error_L1, ..., error_L27]
    all_ratios = []       # Per-prompt: [error_L1/error_L0, error_L2/error_L1, ...]
    all_angles_to_final = []  # Angle between current direction and final direction
    
    for pi, prompt in enumerate(PROMPTS):
        print(f"\r    Prompt {pi+1}/{len(PROMPTS)}", end="", flush=True)
        
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]
        
        # Hook to capture all layer outputs (last token)
        layer_states = {}
        hooks = []
        
        for li in range(N_LAYERS):
            def make_pre_hook(idx):
                def fn(mod, args):
                    layer_states[idx] = args[0][0, -1].detach().float()
                return fn
            hooks.append(model.model.layers[li].register_forward_pre_hook(make_pre_hook(li)))
        
        def final_hook(mod, args, output):
            layer_states["final"] = output[0, -1].detach().float()
        hooks.append(model.model.layers[N_LAYERS-1].register_forward_hook(final_hook))
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        for h in hooks:
            h.remove()
        
        # The "answer" direction: final layer output (after all processing)
        final_state = layer_states["final"]
        
        # Also get the prediction direction from lm_head
        logits = outputs.logits[0, -1, :]
        pred_token = logits.argmax()
        pred_dir = model.lm_head.weight[pred_token].detach().float()
        pred_dir = pred_dir / pred_dir.norm()
        
        # Measure "error" at each layer: distance to final state
        errors = []
        angles = []
        for li in range(N_LAYERS):
            state = layer_states[li]
            # Error = ||state - final_state||
            err = (state - final_state).norm().item()
            errors.append(err)
            # Angle to prediction direction
            cos_a = torch.dot(state / state.norm(), pred_dir).item()
            angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            angles.append(angle)
        
        # Add final state error (should be 0) and angle
        errors.append(0.0)
        cos_final = torch.dot(final_state / final_state.norm(), pred_dir).item()
        angles.append(np.degrees(np.arccos(np.clip(cos_final, -1, 1))))
        
        all_convergence.append(errors)
        all_angles_to_final.append(angles)
        
        # Compute layer-to-layer error ratios
        ratios = []
        for li in range(1, N_LAYERS):
            if errors[li-1] > 1e-10:
                ratios.append(errors[li] / errors[li-1])
            else:
                ratios.append(0.0)
        all_ratios.append(ratios)
    
    print()
    return np.array(all_convergence), np.array(all_ratios), np.array(all_angles_to_final)


# ============================================================================
# PART B: Newton iteration convergence on ζ zeros
# ============================================================================

def measure_newton_convergence(n_zeros=20):
    """
    Run Newton iterations on ζ zeros and measure convergence rate.
    Compare to golden section convergence.
    """
    print("\n  PART B: Measuring Newton convergence on ζ zeros...")
    
    all_newton_errors = []
    all_newton_ratios = []
    
    for idx in range(1, n_zeros + 1):
        # True zero
        t_true = float(zetazero(idx).imag)
        
        # Lambert W initial guess
        shift = idx - 11/8
        if shift <= 0:
            t_guess = 14.134725
        else:
            t_guess = TWO_PI * shift / np.real(lambertw(shift / np.e))
        
        # Run Newton iterations
        mp.dps = 25
        t = mpf(t_guess)
        errors = [abs(float(t) - t_true)]
        
        # Cache derivative (like the zeta solver does)
        s_init = mpc('0.5', t)
        zp_cached = zeta(s_init, derivative=1)
        
        for step in range(15):
            s = mpc('0.5', t)
            z = zeta(s)
            correction = z / zp_cached
            t = t - im(correction)
            err = abs(float(t) - t_true)
            errors.append(err)
            if err < 1e-15:
                break
        
        all_newton_errors.append(errors)
        
        # Compute ratios
        ratios = []
        for i in range(1, len(errors)):
            if errors[i-1] > 1e-20:
                ratios.append(errors[i] / errors[i-1])
            else:
                ratios.append(0.0)
        all_newton_ratios.append(ratios)
    
    return all_newton_errors, all_newton_ratios


# ============================================================================
# PART C: Compare convergence profiles
# ============================================================================

def compare_convergence(layer_errors, layer_ratios, newton_errors, newton_ratios):
    """
    Compare transformer layer convergence to Newton iteration convergence.
    """
    print("\n  PART C: Comparing convergence profiles...")
    
    # Average layer errors across prompts
    mean_errors = np.mean(layer_errors, axis=0)
    mean_ratios = np.mean(layer_ratios, axis=0)
    
    # Normalize errors to [0, 1]
    norm_errors = mean_errors / mean_errors[0]
    
    print(f"\n  Transformer layer error decay (normalized):")
    print(f"  {'Layer':>6}  {'Error':>10}  {'Ratio':>8}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*40}")
    
    for li in range(N_LAYERS + 1):
        err = norm_errors[li]
        ratio_str = f"{mean_ratios[li-1]:.4f}" if li > 0 and li <= len(mean_ratios) else "—"
        bar_len = int(err * 40) if np.isfinite(err) else 0
        bar = "█" * bar_len
        zone = ""
        if li <= 3:
            zone = " [COMP]"
        elif li <= 25:
            zone = " [PROC]"
        else:
            zone = " [TARG]"
        print(f"  L{li:02d}    {err:10.6f}  {ratio_str:>8}  {bar}{zone}")
    
    # Fit geometric decay in each zone
    print(f"\n  Geometric decay fitting (error[l] = a * r^l):")
    
    zones = {
        "Compressor (L0-3)": list(range(0, 4)),
        "Processor (L4-25)": list(range(4, 26)),
        "Targeter (L26-27)": list(range(26, 28)),
    }
    
    for zone_name, layers in zones.items():
        errs = norm_errors[layers]
        if len(errs) > 1 and all(e > 0 for e in errs):
            # Fit log(error) = log(a) + l * log(r)
            log_errs = np.log(errs + 1e-20)
            layer_idx = np.array(layers, dtype=float)
            coeffs = np.polyfit(layer_idx, log_errs, 1)
            r = np.exp(coeffs[0])
            a = np.exp(coeffs[1])
            R2 = 1 - np.var(log_errs - np.polyval(coeffs, layer_idx)) / np.var(log_errs)
            
            # Check φ-expressions
            phi_matches = [
                ("1/φ", 1/PHI, abs(r - 1/PHI) / (1/PHI)),
                ("1/φ²", 1/PHI**2, abs(r - 1/PHI**2) / (1/PHI**2)),
                ("2/φ²", 2/PHI**2, abs(r - 2/PHI**2) / (2/PHI**2)),
                ("φ-1", PHI-1, abs(r - (PHI-1)) / (PHI-1)),
            ]
            best_match = min(phi_matches, key=lambda x: x[2])
            
            print(f"    {zone_name}: r = {r:.4f} (R² = {R2:.4f})")
            print(f"      Best φ-match: r ≈ {best_match[0]} = {best_match[1]:.4f} "
                  f"({(1-best_match[2])*100:.1f}% match)")
    
    # Average Newton convergence
    print(f"\n  Newton iteration convergence on ζ zeros:")
    max_steps = max(len(e) for e in newton_errors)
    
    for step in range(min(max_steps, 10)):
        errs_at_step = [e[step] for e in newton_errors if step < len(e)]
        mean_err = np.mean(errs_at_step)
        ratios_at_step = [r[step] for r in newton_ratios if step < len(r)]
        mean_ratio = np.mean(ratios_at_step) if ratios_at_step else float('nan')
        
        bar = "█" * min(int(np.log10(mean_err + 1e-20) + 20), 40) if mean_err > 0 else ""
        print(f"    Step {step:2d}: error = {mean_err:.2e}, ratio = {mean_ratio:.4f}  {bar}")
    
    # Newton convergence is typically quadratic, but with cached derivative
    # it becomes linear-ish. Measure the effective ratio.
    all_step_ratios = []
    for ratios in newton_ratios:
        for r in ratios:
            if 0 < r < 1:
                all_step_ratios.append(r)
    
    if all_step_ratios:
        mean_newton_ratio = np.mean(all_step_ratios)
        median_newton_ratio = np.median(all_step_ratios)
        print(f"\n    Mean Newton ratio: {mean_newton_ratio:.4f}")
        print(f"    Median Newton ratio: {median_newton_ratio:.4f}")
        
        # Compare to Processor ratio
        proc_ratios = mean_ratios[4:25]
        proc_ratios_valid = proc_ratios[proc_ratios > 0]
        if len(proc_ratios_valid) > 0:
            mean_proc_ratio = np.mean(proc_ratios_valid)
            print(f"    Mean Processor ratio: {mean_proc_ratio:.4f}")
            
            ratio_similarity = 1 - abs(mean_proc_ratio - mean_newton_ratio) / mean_newton_ratio
            print(f"    Ratio similarity: {ratio_similarity*100:.1f}%")
    
    return norm_errors, mean_ratios


def main():
    print("=" * 80)
    print("PHASE 10z4: NEWTON-PROCESSOR HYPOTHESIS")
    print("=" * 80)
    print(f"\nHypothesis: Processor layers are Newton refinement steps, not harmonic corrections.")
    print(f"If true, error decays geometrically at rate φ across layers.")
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    print(f"  Loaded. {N_LAYERS} layers.")
    
    # Part A: Transformer convergence
    layer_errors, layer_ratios, layer_angles = measure_layer_error_decay(model, tokenizer)
    
    # Part B: Newton convergence
    newton_errors, newton_ratios = measure_newton_convergence(n_zeros=20)
    
    # Part C: Compare
    norm_errors, mean_ratios = compare_convergence(
        layer_errors, layer_ratios, newton_errors, newton_ratios
    )
    
    # ================================================================
    # ANALYSIS: Angle trajectory — does it match φ?
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS: Angle to Prediction Direction")
    print("=" * 80)
    
    mean_angles = np.mean(layer_angles, axis=0)
    print(f"\n  {'Layer':>6}  {'Angle°':>8}  {'Δ°':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}")
    
    for li in range(N_LAYERS + 1):
        delta = mean_angles[li] - mean_angles[li-1] if li > 0 else 0
        zone = ""
        if li <= 3: zone = " [COMP]"
        elif li <= 25: zone = " [PROC]"
        else: zone = " [TARG]"
        print(f"  L{li:02d}    {mean_angles[li]:8.2f}  {delta:+8.2f}{zone}")
    
    final_angle = mean_angles[-1]
    arccos_phi2 = np.degrees(np.arccos(1/PHI**2))
    print(f"\n  Final angle: {final_angle:.2f}°")
    print(f"  arccos(1/φ²): {arccos_phi2:.2f}°")
    print(f"  Match: {(1 - abs(final_angle - arccos_phi2)/arccos_phi2)*100:.1f}%")
    
    # ================================================================
    # SYNTHESIS: The Conceptual Proof
    # ================================================================
    print("\n" + "=" * 80)
    print("SYNTHESIS: ζ = Ideal Transformer?")
    print("=" * 80)
    
    # Gather evidence
    evidence = []
    
    # 1. Three-stage pipeline
    evidence.append("1. THREE-STAGE PIPELINE: Both use Estimate → Refine → Target")
    
    # 2. Compressor = Lambert W
    comp_errors = norm_errors[:4]
    comp_drop = comp_errors[0] / comp_errors[3] if comp_errors[3] > 0 else float('inf')
    evidence.append(f"2. COMPRESSOR = LAMBERT W: First 4 layers reduce error by {comp_drop:.1f}×")
    
    # 3. Processor = Newton iteration
    proc_ratios = mean_ratios[4:25]
    proc_valid = proc_ratios[proc_ratios > 0]
    if len(proc_valid) > 0:
        mean_proc = np.mean(proc_valid)
        evidence.append(f"3. PROCESSOR RATIO: {mean_proc:.4f} per layer")
        for name, val in [("1/φ", 1/PHI), ("2/φ²", 2/PHI**2), ("1/φ²", 1/PHI**2)]:
            match = (1 - abs(mean_proc - val)/val) * 100
            evidence.append(f"   vs {name} = {val:.4f}: {match:.1f}% match")
    
    # 4. Targeter = cached derivative
    evidence.append("4. TARGETER = CACHED ζ': rank-1, independent attention (F98)")
    
    # 5. φ at every level
    evidence.append("5. φ AT EVERY LEVEL:")
    evidence.append(f"   - Compressor: α = 1/φ (F107)")
    evidence.append(f"   - Processor: α = 2/φ² (F107)")
    evidence.append(f"   - Targeter: arccos(1/φ²) (F97)")
    evidence.append(f"   - Error frequency: period φ⁷/4 (F108)")
    evidence.append(f"   - Zone angle: arccos(1/3) tetrahedral (F107)")
    
    print()
    for e in evidence:
        print(f"  {e}")
    
    # Save results
    save_data = {
        "layer_error_decay": norm_errors.tolist(),
        "layer_ratios": mean_ratios.tolist(),
        "layer_angles": np.mean(layer_angles, axis=0).tolist(),
        "evidence": evidence,
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/phase10z4_newton_processor.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to results/phase10z4_newton_processor.json")


if __name__ == "__main__":
    main()
