#!/usr/bin/env python3
"""
Polarization Analysis: What Fundamental Structure Are We Missing?

Current state:
- φ-only: 0.88 correlation
- Optimal: 0.94 correlation
- Gap: 6.2%

What could we be missing?
1. SIGN structure - are there opposing dimension pairs?
2. PHASE - do dimensions have angular relationships?
3. POLARIZATION - positive/negative depth encoding?
4. QUADRATURE - 90° phase relationships?

Key insight from holographic work:
- Complex numbers (magnitude + phase) doubled information density
- Phase encodes WHAT KIND of concept
- Magnitude encodes HOW MUCH

Maybe DA2's dimensions have a similar structure:
- Some dimensions encode "near" (positive polarization)
- Some dimensions encode "far" (negative polarization)
- The DIFFERENCE between them is the depth signal

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def load_da2():
    """Load DA2 model."""
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    
    return model, processor


def extract_structure(model, processor, rgb: np.ndarray):
    """Extract DA2's backbone structure."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
        
        full_output = model(inputs['pixel_values'])
        da2_depth = full_output.predicted_depth.squeeze().numpy()
    
    return structure, _normalize(da2_depth)


def collect_data(model, processor, n_images: int = 25):
    """Collect patch-level data."""
    print("\n  Collecting data...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, da2_depth = extract_structure(model, processor, rgb)
        
        structure = structure[1:]
        N, C = structure.shape
        
        depth_h, depth_w = da2_depth.shape
        H_s, W_s = depth_h // 14, depth_w // 14
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        for y in range(H_s):
            for x in range(W_s):
                all_features.append(struct_spatial[y, x])
                all_depths.append(depth_small[y, x])
    
    return np.array(all_features), np.array(all_depths)


def analyze_polarization(features: np.ndarray, depths: np.ndarray):
    """
    Analyze if dimensions have polarization structure.
    
    Hypothesis: Some dimensions encode "near", others encode "far".
    The depth signal is the DIFFERENCE between polarized groups.
    """
    print("\n" + "=" * 70)
    print("POLARIZATION ANALYSIS")
    print("=" * 70)
    
    # Get correlations
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append(corr)
    correlations = np.array(correlations)
    
    # Split into positive and negative correlation dimensions
    pos_dims = np.where(correlations > 0.1)[0]
    neg_dims = np.where(correlations < -0.1)[0]
    neutral_dims = np.where(np.abs(correlations) <= 0.1)[0]
    
    print(f"\n  Dimension polarization:")
    print(f"    Positive (near): {len(pos_dims)} dims")
    print(f"    Negative (far): {len(neg_dims)} dims")
    print(f"    Neutral: {len(neutral_dims)} dims")
    
    # Test polarization hypothesis: depth = pos_sum - neg_sum
    pos_sum = features[:, pos_dims].sum(axis=1) if len(pos_dims) > 0 else np.zeros(len(features))
    neg_sum = features[:, neg_dims].sum(axis=1) if len(neg_dims) > 0 else np.zeros(len(features))
    
    polarized_depth = pos_sum - neg_sum
    polarized_depth = _normalize(polarized_depth)
    
    polar_corr = np.corrcoef(polarized_depth, depths)[0, 1]
    
    print(f"\n  Polarization test (pos - neg):")
    print(f"    Correlation: {polar_corr:.4f}")
    
    # Test weighted polarization
    pos_weights = correlations[pos_dims] if len(pos_dims) > 0 else np.array([])
    neg_weights = -correlations[neg_dims] if len(neg_dims) > 0 else np.array([])
    
    weighted_pos = (features[:, pos_dims] * pos_weights).sum(axis=1) if len(pos_dims) > 0 else np.zeros(len(features))
    weighted_neg = (features[:, neg_dims] * neg_weights).sum(axis=1) if len(neg_dims) > 0 else np.zeros(len(features))
    
    weighted_polar = weighted_pos + weighted_neg
    weighted_polar = _normalize(weighted_polar)
    
    weighted_corr = np.corrcoef(weighted_polar, depths)[0, 1]
    
    print(f"\n  Weighted polarization:")
    print(f"    Correlation: {weighted_corr:.4f}")
    
    return {
        'pos_dims': pos_dims,
        'neg_dims': neg_dims,
        'neutral_dims': neutral_dims,
        'polar_corr': polar_corr,
        'weighted_corr': weighted_corr,
        'correlations': correlations
    }


def analyze_quadrature(features: np.ndarray, depths: np.ndarray, correlations: np.ndarray):
    """
    Analyze if dimensions have quadrature (90° phase) relationships.
    
    In signal processing, quadrature components capture different aspects:
    - I (in-phase): direct correlation
    - Q (quadrature): 90° shifted, captures rate of change
    
    Maybe some dimensions encode depth, others encode depth GRADIENT.
    """
    print("\n" + "=" * 70)
    print("QUADRATURE ANALYSIS")
    print("=" * 70)
    
    # Compute depth gradient (what Q-component might encode)
    # We'll use the variance of depth as a proxy for gradient
    depth_var = np.var(depths)
    
    # For each dimension, check if it correlates with depth variance
    # (i.e., is it encoding "how much depth changes" rather than "what depth is")
    
    # Split features into high/low depth groups
    median_depth = np.median(depths)
    high_depth_mask = depths > median_depth
    low_depth_mask = depths <= median_depth
    
    # Compute dimension variance in each group
    high_var = features[high_depth_mask].var(axis=0)
    low_var = features[low_depth_mask].var(axis=0)
    
    # Dimensions where variance differs between depth groups
    # might be encoding depth-dependent information
    var_ratio = high_var / (low_var + 1e-10)
    
    # Find dimensions with strong variance asymmetry
    asymmetric_dims = np.where(np.abs(np.log(var_ratio)) > 0.5)[0]
    
    print(f"\n  Variance asymmetry analysis:")
    print(f"    Asymmetric dimensions: {len(asymmetric_dims)}")
    print(f"    Top asymmetric dims: {asymmetric_dims[:10]}")
    
    # Test: combine I (correlation) and Q (variance asymmetry)
    # I-component: standard correlation-weighted sum
    sorted_idx = np.argsort(np.abs(correlations))[::-1]
    top_50 = sorted_idx[:50]
    
    I_component = (features[:, top_50] * correlations[top_50]).sum(axis=1)
    
    # Q-component: variance-asymmetric dimensions
    Q_dims = asymmetric_dims[:25]  # Top 25 asymmetric
    Q_weights = np.log(var_ratio[Q_dims])  # Weight by asymmetry
    Q_component = (features[:, Q_dims] * Q_weights).sum(axis=1)
    
    # Combine I and Q
    I_norm = _normalize(I_component)
    Q_norm = _normalize(Q_component)
    
    # Test different combinations
    results = {}
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
        combined = I_norm + alpha * Q_norm
        combined = _normalize(combined)
        corr = np.corrcoef(combined, depths)[0, 1]
        results[f'I+{alpha}Q'] = corr
    
    print(f"\n  I/Q combination results:")
    for name, corr in results.items():
        print(f"    {name}: {corr:.4f}")
    
    return results, var_ratio, asymmetric_dims


def analyze_sign_pairs(features: np.ndarray, depths: np.ndarray, correlations: np.ndarray):
    """
    Analyze if dimensions come in opposing pairs.
    
    Hypothesis: DA2 might encode depth as DIFFERENCE between paired dimensions.
    Like stereoscopy: left - right = disparity.
    """
    print("\n" + "=" * 70)
    print("SIGN PAIR ANALYSIS")
    print("=" * 70)
    
    n_dims = features.shape[1]
    
    # Find dimension pairs with opposite correlations
    pairs = []
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            # Check if correlations are opposite sign and similar magnitude
            if correlations[i] * correlations[j] < 0:  # Opposite signs
                mag_ratio = abs(correlations[i]) / (abs(correlations[j]) + 1e-10)
                if 0.5 < mag_ratio < 2.0:  # Similar magnitude
                    pairs.append((i, j, correlations[i], correlations[j]))
    
    print(f"\n  Found {len(pairs)} opposing pairs")
    
    if len(pairs) > 0:
        # Sort by combined correlation magnitude
        pairs.sort(key=lambda x: abs(x[2]) + abs(x[3]), reverse=True)
        
        print(f"  Top 5 pairs:")
        for i, (d1, d2, c1, c2) in enumerate(pairs[:5]):
            print(f"    Pair {i+1}: dim {d1} ({c1:+.3f}) vs dim {d2} ({c2:+.3f})")
        
        # Test pair-difference encoding
        # depth = Σ (dim_pos - dim_neg) for each pair
        pair_depth = np.zeros(len(features))
        for d1, d2, c1, c2 in pairs[:25]:  # Use top 25 pairs
            if c1 > 0:
                pair_depth += features[:, d1] - features[:, d2]
            else:
                pair_depth += features[:, d2] - features[:, d1]
        
        pair_depth = _normalize(pair_depth)
        pair_corr = np.corrcoef(pair_depth, depths)[0, 1]
        
        print(f"\n  Pair-difference encoding:")
        print(f"    Correlation: {pair_corr:.4f}")
        
        return pairs, pair_corr
    
    return [], 0.0


def test_complex_encoding(features: np.ndarray, depths: np.ndarray, correlations: np.ndarray):
    """
    Test complex number encoding (magnitude + phase).
    
    From holographic work: complex encoding doubled information density.
    
    Idea: Treat pairs of dimensions as complex numbers:
    - dim[2i] = real part
    - dim[2i+1] = imaginary part
    - magnitude = sqrt(real² + imag²)
    - phase = atan2(imag, real)
    """
    print("\n" + "=" * 70)
    print("COMPLEX ENCODING TEST")
    print("=" * 70)
    
    n_dims = features.shape[1]
    n_complex = n_dims // 2
    
    # Sort dimensions by correlation
    sorted_idx = np.argsort(np.abs(correlations))[::-1]
    
    # Pair adjacent sorted dimensions as complex
    magnitudes = []
    phases = []
    
    for i in range(0, min(100, n_dims), 2):
        d1, d2 = sorted_idx[i], sorted_idx[i+1]
        real = features[:, d1]
        imag = features[:, d2]
        
        mag = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
        
        magnitudes.append(mag)
        phases.append(phase)
    
    magnitudes = np.array(magnitudes).T  # (n_samples, n_complex)
    phases = np.array(phases).T
    
    # Test magnitude-only encoding
    mag_weights = np.array([PHI ** (-i/5) for i in range(magnitudes.shape[1])])
    mag_depth = (magnitudes * mag_weights).sum(axis=1)
    mag_depth = _normalize(mag_depth)
    mag_corr = np.corrcoef(mag_depth, depths)[0, 1]
    
    print(f"\n  Magnitude-only encoding:")
    print(f"    Correlation: {mag_corr:.4f}")
    
    # Test phase-weighted encoding
    # Phase might encode "type" of depth (near vs far)
    phase_weights = np.cos(phases)  # Convert phase to weight
    phase_depth = (magnitudes * phase_weights * mag_weights).sum(axis=1)
    phase_depth = _normalize(phase_depth)
    phase_corr = np.corrcoef(phase_depth, depths)[0, 1]
    
    print(f"\n  Phase-weighted encoding:")
    print(f"    Correlation: {phase_corr:.4f}")
    
    # Test complex inner product
    # <query|feature> = Σ conj(q) × f = Σ (real - i*imag) × (real + i*imag)
    # Real part = cos(phase_diff) = agreement
    
    return mag_corr, phase_corr


def summarize_findings(polar_results: dict, quad_results: dict, 
                      pairs: list, pair_corr: float,
                      mag_corr: float, phase_corr: float):
    """Summarize what we found about the missing structure."""
    
    print("\n" + "=" * 70)
    print("POLARIZATION/PHASE SUMMARY")
    print("=" * 70)
    
    print(f"""
    WHAT WE TESTED:
    
    1. POLARIZATION (pos vs neg correlation dims)
       - Simple: {polar_results['polar_corr']:.4f}
       - Weighted: {polar_results['weighted_corr']:.4f}
    
    2. QUADRATURE (I + αQ)
       - Best: {max(quad_results.values()):.4f}
    
    3. SIGN PAIRS (opposing dimension pairs)
       - Found {len(pairs)} pairs
       - Pair-difference: {pair_corr:.4f}
    
    4. COMPLEX ENCODING (magnitude + phase)
       - Magnitude: {mag_corr:.4f}
       - Phase-weighted: {phase_corr:.4f}
    
    BASELINE COMPARISON:
       - φ-only: 0.88
       - Optimal: 0.94
    """)
    
    # Find best approach
    all_results = {
        'polarization': polar_results['weighted_corr'],
        'quadrature': max(quad_results.values()),
        'sign_pairs': pair_corr,
        'complex_mag': mag_corr,
        'complex_phase': phase_corr
    }
    
    best = max(all_results.items(), key=lambda x: x[1])
    
    print(f"    BEST APPROACH: {best[0]} ({best[1]:.4f})")
    
    if best[1] > 0.88:
        print(f"    → This IMPROVES on φ-only!")
    else:
        print(f"    → No improvement over φ-only")
    
    return all_results


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect data
    features, depths = collect_data(model, processor, n_images=25)
    print(f"  Collected {len(features)} patches")
    
    # Analyze polarization
    polar_results = analyze_polarization(features, depths)
    
    # Analyze quadrature
    quad_results, var_ratio, asymmetric_dims = analyze_quadrature(
        features, depths, polar_results['correlations']
    )
    
    # Analyze sign pairs
    pairs, pair_corr = analyze_sign_pairs(
        features, depths, polar_results['correlations']
    )
    
    # Test complex encoding
    mag_corr, phase_corr = test_complex_encoding(
        features, depths, polar_results['correlations']
    )
    
    # Summarize
    all_results = summarize_findings(
        polar_results, quad_results, pairs, pair_corr, mag_corr, phase_corr
    )
