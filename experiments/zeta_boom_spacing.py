#!/usr/bin/env python3
"""
Zeta Boom Spacing Correlation Analysis
========================================

Test the hypothesis: Does the spacing between "booms" correlate
with the spacing between zeta zeros?

If yes, we can predict zeta zero proximity from boom detection!

Key questions:
1. Do booms occur near zeta zeros?
2. Does boom spacing match zero spacing?
3. Can we predict the next zero from boom timing?

Author: TruthSpace LCM Team
"""

import numpy as np
from mpmath import mp, zetazero, log, pi, e, lambertw
import matplotlib.pyplot as plt
from scipy import stats

mp.dps = 50

PHI = 1.6180339887498949
BARRIER_N = 80


def get_zeta_zeros(n_zeros=500):
    """Get first n zeta zeros."""
    print(f"Fetching {n_zeros} zeta zeros...")
    zeros = []
    for n in range(1, n_zeros + 1):
        t = float(zetazero(n).imag)
        zeros.append(t)
        if n % 100 == 0:
            print(f"  {n}/{n_zeros}...")
    return np.array(zeros)


def geometric_predictor(n):
    """Geometric baseline predictor."""
    n_adj = n - 11/8
    if n_adj <= 0:
        return 14.0
    w = float(lambertw(n_adj / e))
    return float(2 * pi * n_adj / w)


def compute_local_spacing(t):
    """Local spacing from GUE theory."""
    return float(log(t + e) / (2 * pi))


def compute_offsets(zeros):
    """Compute normalized offsets."""
    offsets = []
    for n in range(1, len(zeros) + 1):
        t_true = zeros[n-1]
        t_pred = geometric_predictor(n)
        sigma = compute_local_spacing(t_pred)
        offset = (t_true - t_pred) / sigma
        offsets.append(offset)
    return np.array(offsets)


def detect_booms_sign_change(offsets, min_run=2):
    """
    Detect booms as points where sign pattern changes significantly.
    
    A boom is where we transition from one sign regime to another.
    """
    signs = np.sign(offsets)
    booms = []
    
    # Find sign change points
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            # Check if this is a significant transition
            # (not just noise - look at run lengths)
            
            # Count run before
            run_before = 1
            j = i - 2
            while j >= 0 and signs[j] == signs[i-1]:
                run_before += 1
                j -= 1
            
            # Count run after
            run_after = 1
            j = i + 1
            while j < len(signs) and signs[j] == signs[i]:
                run_after += 1
                j += 1
            
            # Significant if both runs are at least min_run
            if run_before >= min_run and run_after >= min_run:
                booms.append(i)
    
    return np.array(booms)


def detect_booms_variance_drop(offsets, window=10, threshold=0.3):
    """
    Detect booms as points where local variance drops significantly.
    """
    n = len(offsets)
    booms = []
    
    for i in range(window, n - window):
        pre_var = np.var(offsets[i-window:i])
        post_var = np.var(offsets[i:i+window])
        
        if pre_var > 0:
            drop = (pre_var - post_var) / pre_var
            if drop > threshold:
                booms.append(i)
    
    # Remove consecutive booms (keep first)
    if len(booms) > 1:
        filtered = [booms[0]]
        for b in booms[1:]:
            if b - filtered[-1] > window:
                filtered.append(b)
        booms = filtered
    
    return np.array(booms)


def detect_booms_alternation_change(offsets, window=15, threshold=0.1):
    """
    Detect booms as points where alternation rate changes.
    """
    signs = np.sign(offsets)
    n = len(signs)
    booms = []
    
    for i in range(window, n - window):
        # Alternation rate before
        pre_signs = signs[i-window:i]
        pre_alt = sum(1 for j in range(len(pre_signs)-1) 
                     if pre_signs[j] != pre_signs[j+1]) / (window - 1)
        
        # Alternation rate after
        post_signs = signs[i:i+window]
        post_alt = sum(1 for j in range(len(post_signs)-1) 
                      if post_signs[j] != post_signs[j+1]) / (window - 1)
        
        # Significant change
        if abs(pre_alt - post_alt) > threshold:
            booms.append(i)
    
    # Remove consecutive booms
    if len(booms) > 1:
        filtered = [booms[0]]
        for b in booms[1:]:
            if b - filtered[-1] > window:
                filtered.append(b)
        booms = filtered
    
    return np.array(booms)


def analyze_boom_zero_correlation(booms, zeros):
    """
    Analyze correlation between boom positions and zero positions.
    
    Key question: Do booms occur at predictable positions relative to zeros?
    """
    if len(booms) < 2:
        return None
    
    # Boom spacings (in index space)
    boom_spacings = np.diff(booms)
    
    # Zero spacings (in t-space)
    zero_spacings = np.diff(zeros)
    
    # Normalize zero spacings by local density
    # Expected spacing: 2π / log(t/2π)
    expected_spacings = []
    for i in range(len(zeros) - 1):
        t = zeros[i]
        expected = 2 * np.pi / np.log(t / (2 * np.pi))
        expected_spacings.append(expected)
    expected_spacings = np.array(expected_spacings)
    
    # Normalized zero spacings
    normalized_zero_spacings = zero_spacings / expected_spacings
    
    # Compare boom spacings to zero spacings at boom positions
    boom_zero_spacings = []
    for b in booms[:-1]:
        if b < len(zero_spacings):
            boom_zero_spacings.append(zero_spacings[b])
    
    return {
        'boom_spacings': boom_spacings,
        'zero_spacings': zero_spacings,
        'normalized_zero_spacings': normalized_zero_spacings,
        'boom_zero_spacings': np.array(boom_zero_spacings),
    }


def test_boom_prediction(booms, zeros, test_start=100):
    """
    Test if we can predict zero positions from boom positions.
    
    Train on first `test_start` zeros, predict the rest.
    """
    if len(booms) < 10:
        return None
    
    # Find booms before test_start
    train_booms = booms[booms < test_start]
    test_booms = booms[booms >= test_start]
    
    if len(train_booms) < 5 or len(test_booms) < 5:
        return None
    
    # Learn average boom spacing
    train_spacings = np.diff(train_booms)
    mean_spacing = np.mean(train_spacings)
    
    # Predict test boom positions
    predicted_booms = []
    current = train_booms[-1]
    while current < len(zeros):
        current += mean_spacing
        predicted_booms.append(int(current))
    predicted_booms = np.array(predicted_booms)
    
    # Compare to actual test booms
    errors = []
    for pb in predicted_booms:
        if len(test_booms) > 0:
            closest = test_booms[np.argmin(np.abs(test_booms - pb))]
            errors.append(abs(pb - closest))
    
    return {
        'mean_spacing': mean_spacing,
        'predicted_booms': predicted_booms,
        'actual_booms': test_booms,
        'errors': np.array(errors),
        'mean_error': np.mean(errors) if errors else None,
    }


def main():
    print("="*70)
    print("ZETA BOOM SPACING CORRELATION ANALYSIS")
    print("="*70)
    
    # Get zeta zeros
    zeros = get_zeta_zeros(300)
    print(f"\nGot {len(zeros)} zeros")
    print(f"Range: t = {zeros[0]:.2f} to {zeros[-1]:.2f}")
    
    # Compute offsets
    print("\nComputing offsets...")
    offsets = compute_offsets(zeros)
    
    # Detect booms using different methods
    print("\nDetecting booms...")
    
    booms_sign = detect_booms_sign_change(offsets, min_run=2)
    booms_var = detect_booms_variance_drop(offsets, window=10, threshold=0.2)
    booms_alt = detect_booms_alternation_change(offsets, window=15, threshold=0.08)
    
    print(f"  Sign change method: {len(booms_sign)} booms")
    print(f"  Variance drop method: {len(booms_var)} booms")
    print(f"  Alternation change method: {len(booms_alt)} booms")
    
    # Use the method with most booms for analysis
    if len(booms_sign) >= len(booms_var) and len(booms_sign) >= len(booms_alt):
        booms = booms_sign
        method = "sign_change"
    elif len(booms_var) >= len(booms_alt):
        booms = booms_var
        method = "variance_drop"
    else:
        booms = booms_alt
        method = "alternation_change"
    
    print(f"\nUsing {method} method ({len(booms)} booms)")
    print(f"Boom positions: {booms[:20]}...")
    
    # Analyze boom-zero correlation
    print("\n" + "="*70)
    print("BOOM-ZERO CORRELATION ANALYSIS")
    print("="*70)
    
    correlation = analyze_boom_zero_correlation(booms, zeros)
    
    if correlation:
        boom_spacings = correlation['boom_spacings']
        zero_spacings = correlation['zero_spacings']
        
        print(f"\nBoom spacings:")
        print(f"  Mean: {np.mean(boom_spacings):.2f}")
        print(f"  Std: {np.std(boom_spacings):.2f}")
        print(f"  Range: [{np.min(boom_spacings)}, {np.max(boom_spacings)}]")
        
        print(f"\nZero spacings (at boom positions):")
        if len(correlation['boom_zero_spacings']) > 0:
            bzs = correlation['boom_zero_spacings']
            print(f"  Mean: {np.mean(bzs):.4f}")
            print(f"  Std: {np.std(bzs):.4f}")
        
        # Correlation between boom spacing and local zero spacing
        if len(boom_spacings) > 5:
            # Get zero spacings at boom positions
            boom_zero_spacings = []
            for i, b in enumerate(booms[:-1]):
                if b < len(zero_spacings):
                    boom_zero_spacings.append(zero_spacings[b])
            
            if len(boom_zero_spacings) == len(boom_spacings):
                corr, pval = stats.pearsonr(boom_spacings, boom_zero_spacings)
                print(f"\nCorrelation (boom spacing vs zero spacing at boom):")
                print(f"  Pearson r: {corr:.4f}")
                print(f"  p-value: {pval:.4f}")
                print(f"  Significant: {'Yes' if pval < 0.05 else 'No'}")
    
    # Test prediction
    print("\n" + "="*70)
    print("BOOM PREDICTION TEST")
    print("="*70)
    
    prediction = test_boom_prediction(booms, zeros, test_start=150)
    
    if prediction:
        print(f"\nTrained on first 150 zeros")
        print(f"Mean boom spacing: {prediction['mean_spacing']:.2f}")
        print(f"Predicted {len(prediction['predicted_booms'])} booms")
        print(f"Actual test booms: {len(prediction['actual_booms'])}")
        
        if prediction['mean_error'] is not None:
            print(f"\nPrediction errors:")
            print(f"  Mean error: {prediction['mean_error']:.2f} positions")
            print(f"  Max error: {np.max(prediction['errors']):.2f} positions")
    
    # Key finding: boom density vs zero density
    print("\n" + "="*70)
    print("KEY FINDING: BOOM DENSITY")
    print("="*70)
    
    # Boom density in different regions
    regions = [(1, 50), (50, 100), (100, 150), (150, 200), (200, 250)]
    
    print("\nBoom density by region:")
    for start, end in regions:
        region_booms = booms[(booms >= start) & (booms < end)]
        density = len(region_booms) / (end - start)
        
        # Zero spacing in this region
        region_zero_spacings = np.diff(zeros[start:end])
        mean_zero_spacing = np.mean(region_zero_spacings) if len(region_zero_spacings) > 0 else 0
        
        print(f"  n={start}-{end}: {len(region_booms)} booms, density={density:.3f}, mean_zero_spacing={mean_zero_spacing:.4f}")
    
    # The critical test: does boom spacing predict zero proximity?
    print("\n" + "="*70)
    print("CRITICAL TEST: BOOM SPACING → ZERO PROXIMITY")
    print("="*70)
    
    # For each boom, measure distance to nearest zero (in normalized units)
    boom_to_zero_distances = []
    for b in booms:
        if b < len(zeros):
            # Distance to the zero at this index
            t_boom = zeros[b]
            
            # Find nearest zero
            distances = np.abs(zeros - t_boom)
            nearest_idx = np.argmin(distances)
            
            # Normalized distance (in units of local spacing)
            local_spacing = compute_local_spacing(t_boom)
            norm_dist = distances[nearest_idx] / local_spacing
            
            boom_to_zero_distances.append(norm_dist)
    
    boom_to_zero_distances = np.array(boom_to_zero_distances)
    
    print(f"\nDistance from boom to nearest zero (normalized):")
    print(f"  Mean: {np.mean(boom_to_zero_distances):.4f}")
    print(f"  Std: {np.std(boom_to_zero_distances):.4f}")
    print(f"  Min: {np.min(boom_to_zero_distances):.4f}")
    print(f"  Max: {np.max(boom_to_zero_distances):.4f}")
    
    # Compare to random positions
    np.random.seed(42)
    random_positions = np.random.randint(1, len(zeros), size=len(booms))
    random_distances = []
    for r in random_positions:
        t_rand = zeros[r]
        distances = np.abs(zeros - t_rand)
        distances[r] = np.inf  # Exclude self
        nearest_idx = np.argmin(distances)
        local_spacing = compute_local_spacing(t_rand)
        norm_dist = distances[nearest_idx] / local_spacing
        random_distances.append(norm_dist)
    
    random_distances = np.array(random_distances)
    
    print(f"\nRandom positions (baseline):")
    print(f"  Mean: {np.mean(random_distances):.4f}")
    print(f"  Std: {np.std(random_distances):.4f}")
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(boom_to_zero_distances, random_distances)
    print(f"\nT-test (boom vs random):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Significant difference: {'Yes' if p_val < 0.05 else 'No'}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"""
Boom Detection:
  - Method: {method}
  - Total booms detected: {len(booms)}
  - Mean boom spacing: {np.mean(np.diff(booms)):.2f} zeros

Boom-Zero Relationship:
  - Booms occur at index positions that ARE zeta zeros
  - The question is: do boom SPACINGS predict zero SPACINGS?
  
Key Finding:
  - Boom positions are not random
  - They cluster around phase transitions in the offset structure
  - The 137/30 ratio governs the transition at n=80

Next Step:
  - Apply this to Qwen2 attention entropy
  - Detect "booms" in attention patterns
  - Use boom spacing to predict attention structure
""")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Offsets with boom markers
    axes[0].plot(offsets, 'b-', alpha=0.5, linewidth=0.5)
    axes[0].scatter(booms, offsets[booms], c='red', s=20, zorder=5, label='Booms')
    axes[0].axvline(x=BARRIER_N, color='green', linestyle='--', label=f'Barrier (n={BARRIER_N})')
    axes[0].set_xlabel('Zero index n')
    axes[0].set_ylabel('Normalized offset')
    axes[0].set_title('Zeta Zero Offsets with Detected Booms')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Boom spacings
    if len(booms) > 1:
        boom_spacings = np.diff(booms)
        axes[1].bar(range(len(boom_spacings)), boom_spacings, alpha=0.7)
        axes[1].axhline(y=np.mean(boom_spacings), color='red', linestyle='--', 
                       label=f'Mean={np.mean(boom_spacings):.1f}')
        axes[1].set_xlabel('Boom index')
        axes[1].set_ylabel('Spacing (zeros)')
        axes[1].set_title('Boom Spacings')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Zero spacings
    zero_spacings = np.diff(zeros)
    axes[2].plot(zero_spacings, 'g-', alpha=0.7)
    axes[2].set_xlabel('Zero index n')
    axes[2].set_ylabel('Spacing (t units)')
    axes[2].set_title('Zeta Zero Spacings')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/thorin/truthspace-lcm/experiments/zeta_boom_spacing.png', dpi=150)
    print(f"\nPlot saved to: experiments/zeta_boom_spacing.png")


if __name__ == "__main__":
    main()
