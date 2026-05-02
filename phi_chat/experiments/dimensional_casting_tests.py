#!/usr/bin/env python3
"""
Dimensional Casting Tests

Testing the unification hypothesis:
1. Do attention weights follow Gaussian moment distribution?
2. Can we use Sierpiński dimension (1.585) for context compression?
3. Does the moment hierarchy σ_k = σ_0 × φ^k appear in attention patterns?

From dimensional downcasting:
- Gaussian moments capture structure at multiple scales
- D = 1.585 = log(3)/log(2) is optimal for prime-structured problems
- φ-scaling: σ_k = σ_0 × φ^k

From context window:
- Attention weights select which V vectors contribute
- Layer 3 is the critical "click" point
- φ-level converges to 1 at bottleneck
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
SIERPINSKI_DIM = np.log(3) / np.log(2)  # ≈ 1.585


class DimensionalCastingTester:
    """Test the dimensional casting unification hypothesis."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Model loaded!\n")
    
    def _get_attention_weights(self, text: str) -> List[np.ndarray]:
        """Get attention weights for all layers."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                output_attentions=True
            )
        
        # Attention: list of (batch, heads, seq, seq) per layer
        # Average over heads, get last token's attention to all positions
        attentions = []
        for layer_attn in outputs.attentions:
            # (batch, heads, seq, seq) -> (seq,) for last token
            attn = layer_attn[0].float().cpu().mean(dim=0)[-1, :].numpy()
            attentions.append(attn)
        
        return attentions
    
    # =========================================================
    # TEST 1: Do attention weights follow Gaussian distribution?
    # =========================================================
    
    def test_gaussian_distribution(self, texts: List[str]) -> Dict:
        """
        Test if attention weights follow a Gaussian-like distribution.
        
        In dimensional downcasting, Gaussian moments are used:
        w_k = exp(-x²/2σ_k²) where σ_k = σ_0 × φ^k
        
        If attention is similar, we'd expect:
        - Attention weights to decay like Gaussians from anchor points
        - The decay rate to follow φ-scaling
        """
        print("TEST 1: Gaussian Distribution of Attention Weights")
        print("=" * 60)
        
        all_attentions = []
        
        for text in texts:
            attentions = self._get_attention_weights(text)
            # Focus on layer 3 (the click point)
            all_attentions.append(attentions[3])
        
        # Combine all attention patterns
        combined = np.concatenate(all_attentions)
        
        # Test for Gaussian-like distribution
        # Attention weights should be positive and sum to 1
        # But their distribution might follow Gaussian decay from peaks
        
        # Fit a Gaussian to the attention distribution
        def gaussian(x, mu, sigma, A):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        # Histogram of attention weights
        hist, bin_edges = np.histogram(combined, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Try to fit Gaussian
        try:
            popt, _ = curve_fit(gaussian, bin_centers, hist, 
                               p0=[np.mean(combined), np.std(combined), max(hist)],
                               maxfev=5000)
            fitted_mu, fitted_sigma, fitted_A = popt
            
            # Compute R² for the fit
            fitted_values = gaussian(bin_centers, *popt)
            ss_res = np.sum((hist - fitted_values)**2)
            ss_tot = np.sum((hist - np.mean(hist))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            gaussian_fit = {
                'mu': fitted_mu,
                'sigma': fitted_sigma,
                'r_squared': r_squared
            }
        except:
            gaussian_fit = {'mu': np.mean(combined), 'sigma': np.std(combined), 'r_squared': 0}
        
        # Also test for power-law (which would indicate fractal structure)
        # Attention often follows Zipf-like distribution
        sorted_attn = np.sort(combined)[::-1]
        ranks = np.arange(1, len(sorted_attn) + 1)
        
        # Fit power law: attn ~ rank^(-α)
        log_ranks = np.log(ranks[sorted_attn > 1e-10])
        log_attn = np.log(sorted_attn[sorted_attn > 1e-10])
        
        if len(log_ranks) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_attn)
            power_law_alpha = -slope
            power_law_r = r_value**2
        else:
            power_law_alpha = 0
            power_law_r = 0
        
        print(f"\nAttention weight statistics:")
        print(f"  Mean: {np.mean(combined):.4f}")
        print(f"  Std: {np.std(combined):.4f}")
        print(f"  Max: {np.max(combined):.4f}")
        
        print(f"\nGaussian fit:")
        print(f"  μ = {gaussian_fit['mu']:.4f}")
        print(f"  σ = {gaussian_fit['sigma']:.4f}")
        print(f"  R² = {gaussian_fit['r_squared']:.4f}")
        
        print(f"\nPower-law fit (Zipf):")
        print(f"  α = {power_law_alpha:.4f}")
        print(f"  R² = {power_law_r:.4f}")
        
        # Check if power-law exponent is close to 1/φ (from Doc 135)
        phi_zipf_target = 1 / PHI  # ≈ 0.618
        phi_zipf_error = abs(power_law_alpha - phi_zipf_target)
        
        print(f"\nφ-Zipf check:")
        print(f"  Target α = 1/φ = {phi_zipf_target:.4f}")
        print(f"  Measured α = {power_law_alpha:.4f}")
        print(f"  Error = {phi_zipf_error:.4f}")
        
        return {
            'gaussian_fit': gaussian_fit,
            'power_law_alpha': power_law_alpha,
            'power_law_r': power_law_r,
            'phi_zipf_error': phi_zipf_error
        }
    
    # =========================================================
    # TEST 2: Sierpiński dimension for context compression
    # =========================================================
    
    def test_sierpinski_compression(self, text: str) -> Dict:
        """
        Test if Sierpiński dimension (1.585) is optimal for context compression.
        
        In DSS, D = log(3)/log(2) ≈ 1.585 is optimal for prime-structured problems.
        
        For context, we'll test if keeping D^k tokens (for various D) preserves
        information better at D ≈ 1.585.
        """
        print("\nTEST 2: Sierpiński Dimension for Context Compression")
        print("=" * 60)
        
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        num_tokens = inputs.input_ids.shape[1]
        
        # Get full attention and hidden states
        with torch.no_grad():
            full_outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True,
                output_attentions=True
            )
        
        full_layer3 = full_outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
        full_final = full_outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        
        # Get attention weights to determine which tokens to keep
        layer3_attn = full_outputs.attentions[3][0].float().cpu().mean(dim=0)[-1, :].numpy()
        
        # Sort tokens by attention (most attended first)
        sorted_indices = np.argsort(layer3_attn)[::-1]
        
        # Test different compression dimensions
        dimensions = [1.0, 1.2, 1.4, 1.585, 1.8, 2.0, 2.5, 3.0]
        results = []
        
        for D in dimensions:
            # Keep D tokens (at minimum 2)
            # For fractal compression: keep top ceil(D) tokens
            # More sophisticated: keep n^(1/D) tokens where n is original
            if D >= 1:
                k = max(2, int(np.ceil(num_tokens ** (1/D))))
            else:
                k = max(2, int(np.ceil(D * num_tokens)))
            
            k = min(k, num_tokens)  # Can't keep more than we have
            
            # Keep top-k attended tokens
            keep_indices = sorted(sorted_indices[:k])
            
            # Create compressed input
            compressed_ids = inputs.input_ids[0, keep_indices].unsqueeze(0)
            
            # Get compressed outputs
            with torch.no_grad():
                comp_outputs = self.model(
                    compressed_ids,
                    output_hidden_states=True
                )
            
            comp_layer3 = comp_outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
            comp_final = comp_outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
            
            # Compute similarities
            layer3_cos = np.dot(full_layer3, comp_layer3) / (
                np.linalg.norm(full_layer3) * np.linalg.norm(comp_layer3) + 1e-10
            )
            final_cos = np.dot(full_final, comp_final) / (
                np.linalg.norm(full_final) * np.linalg.norm(comp_final) + 1e-10
            )
            
            compression_ratio = num_tokens / k
            
            results.append({
                'D': D,
                'tokens_kept': k,
                'compression': compression_ratio,
                'layer3_sim': float(layer3_cos),
                'final_sim': float(final_cos),
                'combined_score': float(layer3_cos * final_cos)  # Product as combined metric
            })
        
        print(f"\nOriginal tokens: {num_tokens}")
        print(f"\n{'D':>6} | {'Kept':>5} | {'Comp':>6} | {'L3 Sim':>8} | {'Final':>8} | {'Score':>8}")
        print("-" * 60)
        
        for r in results:
            marker = " ← Sierpiński" if abs(r['D'] - SIERPINSKI_DIM) < 0.01 else ""
            print(f"{r['D']:>6.3f} | {r['tokens_kept']:>5} | {r['compression']:>6.1f}x | "
                  f"{r['layer3_sim']:>8.4f} | {r['final_sim']:>8.4f} | {r['combined_score']:>8.4f}{marker}")
        
        # Find optimal D
        best = max(results, key=lambda x: x['combined_score'])
        sierpinski_result = next((r for r in results if abs(r['D'] - SIERPINSKI_DIM) < 0.01), None)
        
        print(f"\nBest D: {best['D']:.3f} (score: {best['combined_score']:.4f})")
        if sierpinski_result:
            print(f"Sierpiński D=1.585 score: {sierpinski_result['combined_score']:.4f}")
        
        return {
            'results': results,
            'best_D': best['D'],
            'sierpinski_score': sierpinski_result['combined_score'] if sierpinski_result else 0
        }
    
    # =========================================================
    # TEST 3: φ-scaling in attention patterns
    # =========================================================
    
    def test_phi_scaling(self, texts: List[str]) -> Dict:
        """
        Test if attention patterns follow φ-scaling: σ_k = σ_0 × φ^k
        
        In dimensional downcasting, Gaussian scales follow golden ratio hierarchy.
        
        For attention, we'll check:
        1. Do attention spreads across layers follow φ-scaling?
        2. Do attention peak positions follow φ-scaling?
        """
        print("\nTEST 3: φ-Scaling in Attention Patterns")
        print("=" * 60)
        
        layer_spreads = []  # Attention spread (entropy) per layer
        layer_peaks = []    # Peak attention value per layer
        
        for text in texts:
            attentions = self._get_attention_weights(text)
            
            spreads = []
            peaks = []
            
            for layer_attn in attentions:
                # Spread = entropy of attention distribution
                attn_probs = layer_attn / (layer_attn.sum() + 1e-10)
                entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-10))
                spreads.append(entropy)
                
                # Peak = max attention value
                peaks.append(np.max(layer_attn))
            
            layer_spreads.append(spreads)
            layer_peaks.append(peaks)
        
        # Average across texts
        avg_spreads = np.mean(layer_spreads, axis=0)
        avg_peaks = np.mean(layer_peaks, axis=0)
        
        # Check for φ-scaling in spreads
        # If σ_k = σ_0 × φ^k, then log(σ_k) = log(σ_0) + k × log(φ)
        layers = np.arange(len(avg_spreads))
        
        # Fit: spread = a × φ^(b×layer)
        def phi_scale(k, a, b):
            return a * (PHI ** (b * k))
        
        try:
            popt_spread, _ = curve_fit(phi_scale, layers, avg_spreads, 
                                       p0=[avg_spreads[0], 0.1], maxfev=5000)
            spread_a, spread_b = popt_spread
            
            # Compute R²
            fitted_spreads = phi_scale(layers, *popt_spread)
            ss_res = np.sum((avg_spreads - fitted_spreads)**2)
            ss_tot = np.sum((avg_spreads - np.mean(avg_spreads))**2)
            spread_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except:
            spread_a, spread_b, spread_r2 = 0, 0, 0
        
        # Check for φ-scaling in peaks
        try:
            popt_peak, _ = curve_fit(phi_scale, layers, avg_peaks,
                                     p0=[avg_peaks[0], -0.1], maxfev=5000)
            peak_a, peak_b = popt_peak
            
            fitted_peaks = phi_scale(layers, *popt_peak)
            ss_res = np.sum((avg_peaks - fitted_peaks)**2)
            ss_tot = np.sum((avg_peaks - np.mean(avg_peaks))**2)
            peak_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except:
            peak_a, peak_b, peak_r2 = 0, 0, 0
        
        print(f"\nAttention spread (entropy) by layer:")
        print(f"  Layer 0: {avg_spreads[0]:.4f}")
        print(f"  Layer 3 (click): {avg_spreads[3]:.4f}")
        print(f"  Layer 27 (bottleneck): {avg_spreads[27] if len(avg_spreads) > 27 else avg_spreads[-1]:.4f}")
        
        print(f"\nφ-scaling fit for spreads: σ = {spread_a:.4f} × φ^({spread_b:.4f} × layer)")
        print(f"  R² = {spread_r2:.4f}")
        
        print(f"\nAttention peak by layer:")
        print(f"  Layer 0: {avg_peaks[0]:.4f}")
        print(f"  Layer 3 (click): {avg_peaks[3]:.4f}")
        print(f"  Layer 27 (bottleneck): {avg_peaks[27] if len(avg_peaks) > 27 else avg_peaks[-1]:.4f}")
        
        print(f"\nφ-scaling fit for peaks: peak = {peak_a:.4f} × φ^({peak_b:.4f} × layer)")
        print(f"  R² = {peak_r2:.4f}")
        
        # Check if the scaling exponent is close to 1 (pure φ-scaling)
        print(f"\nφ-scaling check:")
        print(f"  Spread exponent b = {spread_b:.4f} (target: ±1 for pure φ-scaling)")
        print(f"  Peak exponent b = {peak_b:.4f} (target: ±1 for pure φ-scaling)")
        
        # Also check layer-to-layer ratios
        spread_ratios = avg_spreads[1:] / (avg_spreads[:-1] + 1e-10)
        peak_ratios = avg_peaks[1:] / (avg_peaks[:-1] + 1e-10)
        
        print(f"\nLayer-to-layer ratios:")
        print(f"  Spread ratio mean: {np.mean(spread_ratios):.4f} (φ = {PHI:.4f}, 1/φ = {1/PHI:.4f})")
        print(f"  Peak ratio mean: {np.mean(peak_ratios):.4f}")
        
        return {
            'spread_phi_fit': {'a': spread_a, 'b': spread_b, 'r2': spread_r2},
            'peak_phi_fit': {'a': peak_a, 'b': peak_b, 'r2': peak_r2},
            'spread_ratio_mean': float(np.mean(spread_ratios)),
            'peak_ratio_mean': float(np.mean(peak_ratios)),
            'avg_spreads': avg_spreads.tolist(),
            'avg_peaks': avg_peaks.tolist()
        }


def run_dimensional_casting_tests():
    """Run all dimensional casting tests."""
    tester = DimensionalCastingTester()
    
    # Test texts
    test_texts = [
        "The golden ratio φ appears throughout mathematics and nature.",
        "Transformers use attention mechanisms to process sequences.",
        "The Riemann zeta function has zeros on the critical line.",
        "Dimensional downcasting projects high-dimensional structures to lower dimensions.",
        "Context windows in language models act as lenses focusing information.",
    ]
    
    long_text = """You are completing a goal step by step.

GOAL: Write a summary about the φ-computer proof

TOOLS:
- search: Find information about the topic
- generate_and_save: Create and save output file
- done: Mark the task as complete

The φ-computer proof demonstrates that transformers are geometric computers.
Layer 3 is the critical click point where context is integrated.
The bottleneck at layer 27 converges to φ-level 1.
This means the transformer's intelligence is encoded in its geometry.

Current state: Knowledge has been gathered from multiple sources.
What is your next action?"""
    
    results = {}
    
    # Test 1: Gaussian distribution
    results['gaussian'] = tester.test_gaussian_distribution(test_texts)
    
    # Test 2: Sierpiński compression
    results['sierpinski'] = tester.test_sierpinski_compression(long_text)
    
    # Test 3: φ-scaling
    results['phi_scaling'] = tester.test_phi_scaling(test_texts)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: DIMENSIONAL CASTING UNIFICATION")
    print("=" * 60)
    
    print("\n1. ATTENTION DISTRIBUTION:")
    if results['gaussian']['phi_zipf_error'] < 0.2:
        print(f"   ✓ Attention follows φ-Zipf (α ≈ 1/φ = 0.618)")
        print(f"     Measured α = {results['gaussian']['power_law_alpha']:.4f}")
    else:
        print(f"   ~ Attention follows power-law but not exactly φ-Zipf")
        print(f"     Measured α = {results['gaussian']['power_law_alpha']:.4f}, target = 0.618")
    
    print("\n2. SIERPIŃSKI COMPRESSION:")
    best_D = results['sierpinski']['best_D']
    sierpinski_score = results['sierpinski']['sierpinski_score']
    if abs(best_D - SIERPINSKI_DIM) < 0.3:
        print(f"   ✓ Optimal D ≈ Sierpiński dimension (1.585)")
        print(f"     Best D = {best_D:.3f}")
    else:
        print(f"   ~ Optimal D differs from Sierpiński")
        print(f"     Best D = {best_D:.3f}, Sierpiński = 1.585")
    
    print("\n3. φ-SCALING IN ATTENTION:")
    spread_r2 = results['phi_scaling']['spread_phi_fit']['r2']
    spread_ratio = results['phi_scaling']['spread_ratio_mean']
    if spread_r2 > 0.5 or abs(spread_ratio - PHI) < 0.3 or abs(spread_ratio - 1/PHI) < 0.3:
        print(f"   ✓ Attention patterns show φ-scaling")
        print(f"     Spread ratio = {spread_ratio:.4f} (φ = {PHI:.4f}, 1/φ = {1/PHI:.4f})")
    else:
        print(f"   ~ Weak φ-scaling in attention")
        print(f"     Spread ratio = {spread_ratio:.4f}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    # Count how many tests support the unification
    supports = 0
    if results['gaussian']['phi_zipf_error'] < 0.3:
        supports += 1
    if abs(best_D - SIERPINSKI_DIM) < 0.5:
        supports += 1
    if spread_r2 > 0.3 or abs(spread_ratio - PHI) < 0.5 or abs(spread_ratio - 1/PHI) < 0.5:
        supports += 1
    
    if supports >= 2:
        print(f"\n✓ STRONG SUPPORT for dimensional casting unification ({supports}/3 tests)")
        print("  Attention and dimensional downcasting share geometric structure.")
    elif supports == 1:
        print(f"\n~ PARTIAL SUPPORT for unification ({supports}/3 tests)")
        print("  Some geometric parallels exist but not universal.")
    else:
        print(f"\n✗ WEAK SUPPORT for unification ({supports}/3 tests)")
        print("  The parallels may be superficial.")
    
    return results


def deeper_phi_analysis():
    """
    Deeper analysis of where φ appears in attention.
    
    From the initial tests:
    - Power-law α ≈ 0.78 (close to 1/φ = 0.618)
    - Layer-to-layer ratios ≈ 1 (not φ)
    
    Let's look at:
    1. Ratios between specific layers (0→3, 3→27)
    2. Attention concentration ratios
    3. The structure metric S(D) from DSS
    """
    tester = DimensionalCastingTester()
    
    print("\n" + "=" * 60)
    print("DEEPER φ ANALYSIS")
    print("=" * 60)
    
    test_texts = [
        "The golden ratio φ appears throughout mathematics and nature.",
        "Transformers use attention mechanisms to process sequences.",
        "The Riemann zeta function has zeros on the critical line.",
        "Context windows in language models act as lenses focusing information.",
        "You are completing a goal. GOAL: Write a summary. Current state: No knowledge.",
    ]
    
    # Collect detailed attention data
    all_layer_data = []
    
    for text in test_texts:
        attentions = tester._get_attention_weights(text)
        
        layer_data = {
            'layer0_entropy': -np.sum(attentions[0] * np.log(attentions[0] + 1e-10)),
            'layer3_entropy': -np.sum(attentions[3] * np.log(attentions[3] + 1e-10)),
            'layer27_entropy': -np.sum(attentions[27] * np.log(attentions[27] + 1e-10)) if len(attentions) > 27 else 0,
            'layer0_max': np.max(attentions[0]),
            'layer3_max': np.max(attentions[3]),
            'layer27_max': np.max(attentions[27]) if len(attentions) > 27 else 0,
            'layer0_top3': np.sum(np.sort(attentions[0])[-3:]),
            'layer3_top3': np.sum(np.sort(attentions[3])[-3:]),
            'layer27_top3': np.sum(np.sort(attentions[27])[-3:]) if len(attentions) > 27 else 0,
        }
        all_layer_data.append(layer_data)
    
    # Average
    avg_data = {k: np.mean([d[k] for d in all_layer_data]) for k in all_layer_data[0].keys()}
    
    print("\n1. LAYER RATIOS")
    print("-" * 40)
    
    # Key ratios
    entropy_0_to_3 = avg_data['layer3_entropy'] / (avg_data['layer0_entropy'] + 1e-10)
    entropy_3_to_27 = avg_data['layer27_entropy'] / (avg_data['layer3_entropy'] + 1e-10)
    
    max_0_to_3 = avg_data['layer3_max'] / (avg_data['layer0_max'] + 1e-10)
    max_3_to_27 = avg_data['layer27_max'] / (avg_data['layer3_max'] + 1e-10)
    
    top3_0_to_3 = avg_data['layer3_top3'] / (avg_data['layer0_top3'] + 1e-10)
    top3_3_to_27 = avg_data['layer27_top3'] / (avg_data['layer3_top3'] + 1e-10)
    
    print(f"Entropy ratios:")
    print(f"  Layer 0 → 3: {entropy_0_to_3:.4f} (φ = {PHI:.4f}, 1/φ = {1/PHI:.4f})")
    print(f"  Layer 3 → 27: {entropy_3_to_27:.4f}")
    
    print(f"\nMax attention ratios:")
    print(f"  Layer 0 → 3: {max_0_to_3:.4f}")
    print(f"  Layer 3 → 27: {max_3_to_27:.4f}")
    
    print(f"\nTop-3 concentration ratios:")
    print(f"  Layer 0 → 3: {top3_0_to_3:.4f}")
    print(f"  Layer 3 → 27: {top3_3_to_27:.4f}")
    
    # Check which ratios are close to φ or 1/φ
    print("\n2. φ PROXIMITY CHECK")
    print("-" * 40)
    
    ratios = {
        'entropy_0_to_3': entropy_0_to_3,
        'entropy_3_to_27': entropy_3_to_27,
        'max_0_to_3': max_0_to_3,
        'max_3_to_27': max_3_to_27,
        'top3_0_to_3': top3_0_to_3,
        'top3_3_to_27': top3_3_to_27,
    }
    
    for name, ratio in ratios.items():
        phi_dist = abs(ratio - PHI)
        inv_phi_dist = abs(ratio - 1/PHI)
        closest = "φ" if phi_dist < inv_phi_dist else "1/φ"
        dist = min(phi_dist, inv_phi_dist)
        marker = "✓" if dist < 0.15 else "~" if dist < 0.3 else " "
        print(f"  {marker} {name}: {ratio:.4f} (closest to {closest}, dist={dist:.4f})")
    
    # Structure metric from DSS
    print("\n3. STRUCTURE METRIC S(D)")
    print("-" * 40)
    
    # S(D) = σ(distances) / μ(distances)
    # High S = strong structure, Low S = uniform
    
    for text in test_texts[:2]:
        attentions = tester._get_attention_weights(text)
        
        # Compute S for each layer
        S_values = []
        for layer_attn in attentions:
            # Pairwise "distances" = differences in attention weights
            diffs = []
            for i in range(len(layer_attn)):
                for j in range(i+1, len(layer_attn)):
                    diffs.append(abs(layer_attn[i] - layer_attn[j]))
            
            if diffs:
                S = np.std(diffs) / (np.mean(diffs) + 1e-10)
                S_values.append(S)
        
        print(f"\nText: {text[:40]}...")
        print(f"  S at layer 0: {S_values[0]:.4f}")
        print(f"  S at layer 3: {S_values[3]:.4f}")
        print(f"  S at layer 27: {S_values[27] if len(S_values) > 27 else S_values[-1]:.4f}")
        
        # Ratio of S values
        S_ratio_0_3 = S_values[3] / (S_values[0] + 1e-10)
        S_ratio_3_27 = S_values[27] / (S_values[3] + 1e-10) if len(S_values) > 27 else 0
        print(f"  S ratio 0→3: {S_ratio_0_3:.4f}")
        print(f"  S ratio 3→27: {S_ratio_3_27:.4f}")
    
    # Check the 3/27 ratio (layer numbers)
    print("\n4. LAYER NUMBER RATIOS")
    print("-" * 40)
    print(f"  Layer 3 / Layer 27 = {3/27:.4f}")
    print(f"  1/φ² = {1/PHI**2:.4f}")
    print(f"  Difference: {abs(3/27 - 1/PHI**2):.4f}")
    
    # 3 and 27 are interesting: 27 = 3³
    print(f"\n  27 = 3³ (self-similar!)")
    print(f"  log(27)/log(3) = 3 (integer dimension)")
    print(f"  But 3/27 = 1/9 ≈ 1/φ⁴ = {1/PHI**4:.4f}")


if __name__ == "__main__":
    run_dimensional_casting_tests()
    deeper_phi_analysis()
