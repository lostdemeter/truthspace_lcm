#!/usr/bin/env python3
"""
Protocol Analysis: Examining the Irreducible Sign Representation
=================================================================

Applying all four protocols to determine if sign patterns are truly irreducible:

1. GOP (Gushurst Optimization Protocol):
   - Fractal Peel: Extract recursive structure from sign patterns
   - Formalize Parameters: Identify mathematical meaning
   - Time Affinity: Use computation as fitness signal

2. MGOP (Multifold Gushurst Optimization Protocol):
   - Multiple projections: spatial, frequency, fractal, zeta
   - Check for holographic bound
   - Determine if all projections converge

3. EDP (Equation Discovery Protocol):
   - Search for closed-form patterns in sign structure
   - Use anchor coordinates (φ, sierpinski, etc.)
   - Error-as-signal analysis

4. PEP (Probe Extraction Protocol):
   - Can we extract sign structure via probing?
   - Is there a measurement-based approach?
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.fft import fft, fftfreq

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)


class ProtocolAnalyzer:
    """Apply all four protocols to sign pattern analysis."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach().float()
        self.hidden_dim = self.all_embeds.shape[1]
        self.vocab_size = self.all_embeds.shape[0]
        
        # Precompute signs
        self.all_signs = torch.sign(self.all_embeds).cpu()
        self.all_signs[self.all_signs == 0] = 1
    
    def get_sign_pattern(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_signs[ids[0]]
    
    # =========================================================================
    # GOP: GUSHURST OPTIMIZATION PROTOCOL
    # =========================================================================
    
    def gop_fractal_peel(self, pairs: List[Tuple[str, str]]):
        """
        GOP Phase 1: Fractal Peel
        
        Extract recursive structure from sign flip patterns.
        """
        print("\n" + "="*70)
        print("GOP PHASE 1: FRACTAL PEEL")
        print("="*70)
        
        # Collect all flip patterns
        flip_patterns = []
        for neg, pos in pairs:
            s_neg = self.get_sign_pattern(neg)
            s_pos = self.get_sign_pattern(pos)
            if s_neg is not None and s_pos is not None:
                flips = (s_neg != s_pos).float()
                flip_patterns.append(flips)
        
        if not flip_patterns:
            print("  No valid pairs found")
            return
        
        # Stack: [n_pairs, hidden_dim]
        F = torch.stack(flip_patterns)
        
        # 1. Compute autocorrelation of flip patterns
        mean_pattern = F.mean(dim=0)
        autocorr = np.correlate(mean_pattern.numpy(), mean_pattern.numpy(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        print(f"\n  Autocorrelation analysis:")
        print(f"    Lag 0: {autocorr[0]:.4f}")
        print(f"    Lag 1: {autocorr[1]:.4f}")
        print(f"    Lag φ (≈1.618): {autocorr[2]:.4f}")
        print(f"    Lag 10: {autocorr[10]:.4f}")
        print(f"    Lag 100: {autocorr[100]:.4f}")
        
        # 2. FFT to find dominant frequencies
        fft_result = np.abs(fft(mean_pattern.numpy()))
        freqs = fftfreq(len(mean_pattern), 1.0)
        
        # Find top 10 frequencies
        top_indices = np.argsort(fft_result)[-20:][::-1]
        print(f"\n  Top FFT frequencies:")
        for i, idx in enumerate(top_indices[:10]):
            freq = freqs[idx]
            power = fft_result[idx]
            # Check if frequency relates to φ
            if abs(freq) > 0.001:
                phi_relation = abs(freq) / (1/PHI)
                print(f"    {i+1}. freq={abs(freq):.4f}, power={power:.2f}, φ-ratio={phi_relation:.3f}")
        
        # 3. Check for Fibonacci periods (from Doc 147)
        fib_periods = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        print(f"\n  Fibonacci period analysis:")
        for period in fib_periods[:8]:
            if period < len(autocorr):
                print(f"    Period {period}: autocorr={autocorr[period]:.4f}")
        
        # 4. Resfrac score (predictability)
        # High resfrac = random, Low resfrac = structured
        resfrac = 1 - np.abs(autocorr[1:100]).mean()
        print(f"\n  Resfrac score: {resfrac:.4f}")
        if resfrac > 0.5:
            print("    → ERGODIC (appears random)")
        else:
            print("    → STRUCTURED (exploitable pattern)")
        
        return {
            'autocorr': autocorr,
            'fft': fft_result,
            'resfrac': resfrac,
            'mean_pattern': mean_pattern
        }
    
    def gop_formalize_parameters(self, peel_result):
        """
        GOP Phase 2: Formalize Parameters
        
        Identify mathematical meaning of discovered patterns.
        """
        print("\n" + "="*70)
        print("GOP PHASE 2: FORMALIZE PARAMETERS")
        print("="*70)
        
        mean_pattern = peel_result['mean_pattern']
        
        # 1. Analyze flip probability distribution
        flip_prob = mean_pattern.numpy()
        
        print(f"\n  Flip probability statistics:")
        print(f"    Mean: {flip_prob.mean():.4f}")
        print(f"    Std:  {flip_prob.std():.4f}")
        print(f"    Min:  {flip_prob.min():.4f}")
        print(f"    Max:  {flip_prob.max():.4f}")
        
        # 2. Check if distribution follows φ-Zipf
        sorted_probs = np.sort(flip_prob)[::-1]
        ranks = np.arange(1, len(sorted_probs) + 1)
        
        # Zipf: prob ∝ rank^(-α)
        # φ-Zipf: prob ∝ φ^(-rank)
        
        # Fit power law
        log_ranks = np.log(ranks[:100])
        log_probs = np.log(sorted_probs[:100] + 1e-10)
        slope = np.polyfit(log_ranks, log_probs, 1)[0]
        
        print(f"\n  Power law fit:")
        print(f"    Zipf exponent α: {-slope:.4f}")
        print(f"    Expected for φ-Zipf: {1/LOG_PHI:.4f}")
        
        # 3. Check for level structure (from Doc 128)
        # Weights live on φ-lattice with peak at φ^-9
        levels = np.round(np.log(np.abs(self.all_embeds[0].cpu().numpy()) + 1e-10) / LOG_PHI)
        level_counts = np.bincount(levels.astype(int) + 50, minlength=100)
        peak_level = np.argmax(level_counts) - 50
        
        print(f"\n  φ-level structure:")
        print(f"    Peak level: φ^{peak_level}")
        print(f"    Expected (Doc 128): φ^-9")
        
        return {
            'zipf_exponent': -slope,
            'peak_level': peak_level,
            'flip_prob_stats': {
                'mean': flip_prob.mean(),
                'std': flip_prob.std()
            }
        }
    
    # =========================================================================
    # MGOP: MULTIFOLD GUSHURST OPTIMIZATION PROTOCOL
    # =========================================================================
    
    def mgop_projection_synthesis(self, pairs: List[Tuple[str, str]]):
        """
        MGOP Phase 5: Projection Synthesis
        
        Check if all projections converge to the same value (holographic bound).
        """
        print("\n" + "="*70)
        print("MGOP PHASE 5: PROJECTION SYNTHESIS")
        print("="*70)
        
        # Collect flip patterns
        flip_patterns = []
        for neg, pos in pairs:
            s_neg = self.get_sign_pattern(neg)
            s_pos = self.get_sign_pattern(pos)
            if s_neg is not None and s_pos is not None:
                flips = (s_neg != s_pos).float()
                flip_patterns.append(flips)
        
        F = torch.stack(flip_patterns)
        
        # Projection 1: Spatial (dimension-wise flip probability)
        spatial_score = F.mean(dim=0).mean().item()
        
        # Projection 2: Frequency (FFT of mean pattern)
        fft_result = np.abs(fft(F.mean(dim=0).numpy()))
        freq_score = fft_result.mean()
        
        # Projection 3: Fractal (variance across pairs)
        fractal_score = F.var(dim=0).mean().item()
        
        # Projection 4: SVD (low-rank structure)
        U, S, Vh = torch.linalg.svd(F, full_matrices=False)
        # How much variance is captured by top-k components?
        total_var = (S ** 2).sum().item()
        top10_var = (S[:10] ** 2).sum().item()
        svd_score = top10_var / total_var
        
        print(f"\n  Projection scores:")
        print(f"    Spatial (mean flip prob):  {spatial_score:.4f}")
        print(f"    Frequency (FFT mean):      {freq_score:.4f}")
        print(f"    Fractal (variance):        {fractal_score:.4f}")
        print(f"    SVD (top-10 variance):     {svd_score:.4f}")
        
        # Check convergence
        scores = [spatial_score, fractal_score, svd_score]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        convergence_ratio = std_score / mean_score if mean_score > 0 else float('inf')
        
        print(f"\n  Convergence analysis:")
        print(f"    Mean score: {mean_score:.4f}")
        print(f"    Std score:  {std_score:.4f}")
        print(f"    Convergence ratio: {convergence_ratio:.4f}")
        
        if convergence_ratio < 0.1:
            print("    → HOLOGRAPHIC BOUND DETECTED")
        else:
            print("    → Projections diverge - more structure to extract")
        
        # Singular value analysis
        print(f"\n  Singular value analysis:")
        print(f"    Top 10 singular values: {S[:10].tolist()}")
        print(f"    Effective rank (90% var): {(S.cumsum(0) / S.sum() < 0.9).sum().item() + 1}")
        
        return {
            'spatial': spatial_score,
            'frequency': freq_score,
            'fractal': fractal_score,
            'svd': svd_score,
            'convergence_ratio': convergence_ratio,
            'singular_values': S[:20].tolist()
        }
    
    # =========================================================================
    # EDP: EQUATION DISCOVERY PROTOCOL
    # =========================================================================
    
    def edp_pattern_detection(self, pairs: List[Tuple[str, str]]):
        """
        EDP Phase 5: Pattern Detection
        
        Search for closed-form patterns in sign structure.
        """
        print("\n" + "="*70)
        print("EDP PHASE 5: PATTERN DETECTION")
        print("="*70)
        
        # Collect flip counts per dimension
        flip_counts = torch.zeros(self.hidden_dim)
        n_pairs = 0
        
        for neg, pos in pairs:
            s_neg = self.get_sign_pattern(neg)
            s_pos = self.get_sign_pattern(pos)
            if s_neg is not None and s_pos is not None:
                flips = (s_neg != s_pos).float()
                flip_counts += flips
                n_pairs += 1
        
        flip_prob = flip_counts / n_pairs
        
        # Search for φ-patterns in flip probability
        print(f"\n  Searching for φ-patterns in flip probability...")
        
        # Check if flip_prob[i] ≈ (n/d) × φ^k for small n, d, k
        phi_patterns = []
        for i in range(min(100, self.hidden_dim)):
            prob = flip_prob[i].item()
            best_pattern = self._find_phi_pattern(prob)
            if best_pattern and best_pattern['error'] < 0.01:
                phi_patterns.append((i, best_pattern))
        
        print(f"    Found {len(phi_patterns)} clean φ-patterns in first 100 dims")
        if phi_patterns:
            print(f"    Examples:")
            for i, (dim, pattern) in enumerate(phi_patterns[:5]):
                print(f"      dim {dim}: {pattern['n']}/{pattern['d']} × φ^{pattern['k']} (err={pattern['error']:.6f})")
        
        # Check for arctan(1/φ) and log(φ) patterns
        arctan_phi = math.atan(1/PHI)
        log_phi = math.log(PHI)
        
        # Total flip probability
        total_flip = flip_prob.sum().item()
        mean_flip = flip_prob.mean().item()
        
        print(f"\n  Total flip probability: {total_flip:.4f}")
        print(f"  Mean flip probability: {mean_flip:.4f}")
        
        # Check if mean relates to φ
        print(f"\n  Checking φ-relations:")
        print(f"    mean / (1/φ): {mean_flip / (1/PHI):.4f}")
        print(f"    mean / (1/φ²): {mean_flip / (1/PHI**2):.4f}")
        print(f"    mean × φ: {mean_flip * PHI:.4f}")
        print(f"    mean × 2: {mean_flip * 2:.4f}")
        
        # Check if mean ≈ 0.5 (random) or has structure
        if abs(mean_flip - 0.5) < 0.05:
            print(f"    → Mean ≈ 0.5 (appears random)")
        else:
            print(f"    → Mean ≠ 0.5 (structured!)")
        
        return {
            'phi_patterns': phi_patterns,
            'total_flip': total_flip,
            'mean_flip': mean_flip
        }
    
    def _find_phi_pattern(self, value, max_n=20, max_d=20, max_k=10):
        """Find (n/d) × φ^k approximation for a value."""
        best = None
        
        for k in range(-max_k, max_k + 1):
            phi_k = PHI ** k
            for d in range(1, max_d + 1):
                for n in range(-max_n, max_n + 1):
                    if n == 0:
                        continue
                    approx = (n / d) * phi_k
                    err = abs(value - approx)
                    if best is None or err < best['error']:
                        best = {'n': n, 'd': d, 'k': k, 'approx': approx, 'error': err}
        
        return best
    
    # =========================================================================
    # PEP: PROBE EXTRACTION PROTOCOL
    # =========================================================================
    
    def pep_probe_analysis(self, pairs: List[Tuple[str, str]]):
        """
        PEP: Probe Extraction Protocol
        
        Can we extract sign structure via probing?
        """
        print("\n" + "="*70)
        print("PEP: PROBE EXTRACTION PROTOCOL")
        print("="*70)
        
        # The key question: Can we MEASURE the sign pattern directly
        # rather than APPROXIMATE it?
        
        # For embeddings, the signs ARE directly measurable:
        # sign[i] = sign(embedding[i])
        
        # The question is: Is there a SIMPLER representation?
        
        # Collect flip patterns
        flip_patterns = []
        for neg, pos in pairs:
            s_neg = self.get_sign_pattern(neg)
            s_pos = self.get_sign_pattern(pos)
            if s_neg is not None and s_pos is not None:
                flips = (s_neg != s_pos).float()
                flip_patterns.append(flips)
        
        F = torch.stack(flip_patterns)
        
        # SVD to find low-rank structure
        U, S, Vh = torch.linalg.svd(F, full_matrices=False)
        
        print(f"\n  SVD analysis of flip patterns:")
        print(f"    Shape: {F.shape}")
        print(f"    Rank: {(S > 1e-6).sum().item()}")
        
        # How many components needed for 95% variance?
        cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
        n_95 = (cumvar < 0.95).sum().item() + 1
        n_99 = (cumvar < 0.99).sum().item() + 1
        
        print(f"    Components for 95% variance: {n_95}")
        print(f"    Components for 99% variance: {n_99}")
        
        # Can we reconstruct flip patterns from low-rank approximation?
        for k in [1, 5, 10, 20]:
            if k > len(S):
                continue
            F_approx = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            reconstruction_error = (F - F_approx).abs().mean().item()
            accuracy = ((F_approx > 0.5) == (F > 0.5)).float().mean().item()
            print(f"    Rank-{k} reconstruction: error={reconstruction_error:.4f}, accuracy={accuracy*100:.1f}%")
        
        # The key insight from PEP:
        # If we can't reduce the representation, the signs ARE the irreducible form
        print(f"\n  PEP Conclusion:")
        if n_95 > F.shape[0] * 0.5:
            print("    → Signs have HIGH effective rank")
            print("    → Cannot be compressed significantly")
            print("    → Signs ARE the irreducible representation")
        else:
            print("    → Signs have LOW effective rank")
            print("    → Can be compressed via SVD")
            print("    → There IS hidden structure to exploit")
        
        return {
            'rank': (S > 1e-6).sum().item(),
            'n_95': n_95,
            'n_99': n_99,
            'singular_values': S[:20].tolist()
        }


def run_full_protocol_analysis(model, tokenizer):
    """Run all four protocols on sign pattern analysis."""
    print("="*70)
    print("FULL PROTOCOL ANALYSIS: EXAMINING IRREDUCIBLE REPRESENTATION")
    print("="*70)
    print("""
Applying all four protocols to determine if sign patterns are truly irreducible:
  1. GOP: Fractal Peel + Formalize Parameters
  2. MGOP: Projection Synthesis (holographic bound check)
  3. EDP: Pattern Detection (φ-patterns)
  4. PEP: Probe Extraction (low-rank structure)
""")
    
    analyzer = ProtocolAnalyzer(model, tokenizer)
    
    # Collect opposite pairs
    pairs = [
        ("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery"),
        ("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant"),
        ("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"),
        ("short", "tall"), ("low", "high"), ("squat", "towering"),
        ("dark", "bright"), ("dim", "light"), ("gloomy", "radiant"),
        ("young", "old"), ("new", "ancient"), ("fresh", "stale"),
        ("bad", "good"), ("sad", "happy"), ("negative", "positive"),
        ("light", "heavy"), ("weightless", "weighty"),
        ("soft", "hard"), ("tender", "tough"), ("gentle", "harsh"),
        ("dry", "wet"), ("arid", "damp"), ("parched", "moist"),
    ]
    
    # Run all protocols
    results = {}
    
    # GOP
    peel_result = analyzer.gop_fractal_peel(pairs)
    results['gop_peel'] = peel_result
    
    if peel_result:
        formal_result = analyzer.gop_formalize_parameters(peel_result)
        results['gop_formal'] = formal_result
    
    # MGOP
    mgop_result = analyzer.mgop_projection_synthesis(pairs)
    results['mgop'] = mgop_result
    
    # EDP
    edp_result = analyzer.edp_pattern_detection(pairs)
    results['edp'] = edp_result
    
    # PEP
    pep_result = analyzer.pep_probe_analysis(pairs)
    results['pep'] = pep_result
    
    # Final synthesis
    print("\n" + "="*70)
    print("FINAL SYNTHESIS: IS THE SIGN REPRESENTATION IRREDUCIBLE?")
    print("="*70)
    
    # Collect evidence
    evidence_for_irreducible = []
    evidence_against_irreducible = []
    
    # GOP evidence
    if peel_result and peel_result['resfrac'] > 0.5:
        evidence_for_irreducible.append("GOP: Resfrac > 0.5 (ergodic/random)")
    else:
        evidence_against_irreducible.append("GOP: Resfrac < 0.5 (structured)")
    
    # MGOP evidence
    if mgop_result['convergence_ratio'] < 0.1:
        evidence_for_irreducible.append("MGOP: Holographic bound detected")
    else:
        evidence_against_irreducible.append("MGOP: Projections diverge")
    
    # EDP evidence
    if len(edp_result['phi_patterns']) < 10:
        evidence_for_irreducible.append("EDP: Few clean φ-patterns")
    else:
        evidence_against_irreducible.append("EDP: Many clean φ-patterns")
    
    # PEP evidence
    if pep_result['n_95'] > len(pairs) * 0.5:
        evidence_for_irreducible.append("PEP: High effective rank")
    else:
        evidence_against_irreducible.append("PEP: Low effective rank (compressible)")
    
    print(f"\n  Evidence FOR irreducibility ({len(evidence_for_irreducible)}):")
    for e in evidence_for_irreducible:
        print(f"    ✓ {e}")
    
    print(f"\n  Evidence AGAINST irreducibility ({len(evidence_against_irreducible)}):")
    for e in evidence_against_irreducible:
        print(f"    ✗ {e}")
    
    # Verdict
    print("\n" + "-"*70)
    if len(evidence_for_irreducible) > len(evidence_against_irreducible):
        print("  VERDICT: Signs ARE the irreducible representation")
        print("  The sign bit encodes learned semantic content that cannot be simplified.")
    else:
        print("  VERDICT: Signs have HIDDEN STRUCTURE")
        print("  There may be a simpler representation waiting to be discovered.")
    print("-"*70)
    
    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    run_full_protocol_analysis(model, tokenizer)


if __name__ == "__main__":
    main()
