#!/usr/bin/env python3
"""
ConvNeXt Weight Analysis

Analyze ConvNeXt weights for geometric patterns:
- SVD singular value distribution (φ-Zipf?)
- Weight magnitude distribution
- Correlation between layers

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

PHI = (1 + np.sqrt(5)) / 2


def analyze_svd_pattern(W, name):
    """Analyze SVD singular values for φ-Zipf pattern."""
    if W.ndim == 4:
        # Conv weight: [out, in, H, W] -> [out, in*H*W]
        W_2d = W.reshape(W.shape[0], -1)
    elif W.ndim == 2:
        W_2d = W
    else:
        return None
    
    U, S, Vt = np.linalg.svd(W_2d, full_matrices=False)
    
    # Fit Zipf: S[i] ∝ 1/i^α
    # log(S[i]) = log(S[0]) - α * log(i)
    ranks = np.arange(1, len(S) + 1)
    log_ranks = np.log(ranks)
    log_S = np.log(S + 1e-10)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_S)
    alpha = -slope  # Zipf exponent
    
    # Check ratio of first two singular values
    ratio = S[0] / S[1] if S[1] > 1e-10 else float('inf')
    
    return {
        'name': name,
        'shape': W.shape,
        'S': S,
        'alpha': alpha,
        'r_squared': r_value**2,
        'S0_S1_ratio': ratio,
        'effective_rank': np.sum(S > S[0] * 0.01),  # Rank at 1% threshold
    }


def analyze_convnext_weights():
    """Analyze all ConvNeXt weights."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    convnext = model.encoder.arch
    
    print("=" * 70)
    print("CONVNEXT WEIGHT ANALYSIS")
    print("=" * 70)
    
    # Collect all weight matrices
    results = []
    
    for name, param in convnext.named_parameters():
        if 'weight' in name and param.ndim >= 2:
            W = param.detach().cpu().numpy()
            result = analyze_svd_pattern(W, name)
            if result:
                results.append(result)
    
    print("\n1. SVD ANALYSIS (Zipf Exponent)")
    print("-" * 70)
    print(f"{'Layer':<45} {'Shape':<20} {'α':<8} {'R²':<8} {'S0/S1':<8}")
    print("-" * 70)
    
    alphas_dwconv = []
    alphas_pwconv = []
    
    for r in results:
        layer_type = 'dwconv' if 'dwconv' in r['name'] else 'pwconv'
        shape_str = str(r['shape'])
        print(f"{r['name']:<45} {shape_str:<20} {r['alpha']:.4f}  {r['r_squared']:.4f}  {r['S0_S1_ratio']:.4f}")
        
        if 'dwconv' in r['name']:
            alphas_dwconv.append(r['alpha'])
        elif 'pwconv' in r['name']:
            alphas_pwconv.append(r['alpha'])
    
    print("\n2. SUMMARY BY LAYER TYPE")
    print("-" * 70)
    print(f"Target φ-Zipf exponent: 1/φ = {1/PHI:.4f}")
    print()
    
    if alphas_dwconv:
        print(f"Depthwise Conv (7x7):")
        print(f"  Mean α: {np.mean(alphas_dwconv):.4f}")
        print(f"  Std α:  {np.std(alphas_dwconv):.4f}")
        print(f"  Range:  [{min(alphas_dwconv):.4f}, {max(alphas_dwconv):.4f}]")
    
    if alphas_pwconv:
        print(f"\nPointwise Conv (Linear):")
        print(f"  Mean α: {np.mean(alphas_pwconv):.4f}")
        print(f"  Std α:  {np.std(alphas_pwconv):.4f}")
        print(f"  Range:  [{min(alphas_pwconv):.4f}, {max(alphas_pwconv):.4f}]")
    
    print("\n3. COMPARISON TO KNOWN PATTERNS")
    print("-" * 70)
    print(f"""
    Pattern          | Expected α | ConvNeXt DWConv | ConvNeXt PWConv
    -----------------|------------|-----------------|----------------
    φ-Zipf (ideal)   | 0.618      | {np.mean(alphas_dwconv):.3f}           | {np.mean(alphas_pwconv):.3f}
    Attention heads  | ~0.65      |                 |
    MLP layers       | ~0.12      |                 |
    """)
    
    print("\n4. INTERPRETATION")
    print("-" * 70)
    
    dwconv_mean = np.mean(alphas_dwconv) if alphas_dwconv else 0
    pwconv_mean = np.mean(alphas_pwconv) if alphas_pwconv else 0
    
    if abs(dwconv_mean - 1/PHI) < 0.1:
        print("  ✓ Depthwise convs follow φ-Zipf pattern!")
    else:
        print(f"  ✗ Depthwise convs do NOT follow φ-Zipf (α={dwconv_mean:.3f} vs {1/PHI:.3f})")
    
    if abs(pwconv_mean - 1/PHI) < 0.1:
        print("  ✓ Pointwise convs follow φ-Zipf pattern!")
    else:
        print(f"  ✗ Pointwise convs do NOT follow φ-Zipf (α={pwconv_mean:.3f} vs {1/PHI:.3f})")
    
    return results


def analyze_weight_distribution():
    """Analyze weight value distributions."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    convnext = model.encoder.arch
    
    print("\n" + "=" * 70)
    print("WEIGHT VALUE DISTRIBUTION")
    print("=" * 70)
    
    all_weights = []
    for name, param in convnext.named_parameters():
        if 'weight' in name:
            all_weights.append(param.detach().cpu().numpy().flatten())
    
    all_weights = np.concatenate(all_weights)
    
    print(f"\nTotal weights: {len(all_weights):,}")
    print(f"Mean: {all_weights.mean():.6f}")
    print(f"Std:  {all_weights.std():.6f}")
    print(f"Min:  {all_weights.min():.6f}")
    print(f"Max:  {all_weights.max():.6f}")
    
    # Check for sparsity
    near_zero = np.sum(np.abs(all_weights) < 0.01) / len(all_weights)
    print(f"\nSparsity (|w| < 0.01): {near_zero*100:.1f}%")
    
    # Check for quantization potential
    unique_vals = len(np.unique(np.round(all_weights, 3)))
    print(f"Unique values (3 decimal): {unique_vals:,}")


if __name__ == "__main__":
    results = analyze_convnext_weights()
    analyze_weight_distribution()
