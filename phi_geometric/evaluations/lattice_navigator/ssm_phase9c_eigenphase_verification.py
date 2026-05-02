"""
Phase 9C: Rigorous Verification of Eigenvalue Phase Clustering

Phase 9B found eigenvalue phases of W2@W1 clustering at φ-lattice positions.
This script tests whether that clustering is statistically significant.

Method:
  1. Compute eigenvalue phases for ALL blocks
  2. Bootstrap test: compare to random matrices with same SV spectrum
  3. KS test against uniform distribution
  4. Nearest-neighbor test against φ-lattice
  5. If confirmed: quantify HOW MUCH information the φ-lattice captures
"""
import numpy as np
import sys
from scipy.stats import kstest, chisquare
from scipy.optimize import curve_fit

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')

from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
v16 = V16GeometricColorizer()
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]


# ================================================================
# STEP 1: Collect ALL eigenvalue phases from ALL blocks
# ================================================================
print('=' * 70)
print('STEP 1: Collecting Eigenvalue Phases from All Blocks')
print('=' * 70)
print()

all_phases_deg = []
all_phases_by_block = {}

for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w1 = v16._get_weight(f'{prefix}.pwconv1.weight')
        w2 = v16._get_weight(f'{prefix}.pwconv2.weight')
        if w1 is None or w2 is None:
            continue
        
        W_eff = w2.numpy() @ w1.numpy()
        eigvals = np.linalg.eigvals(W_eff)
        
        # Get phases of complex eigenvalues
        complex_mask = np.abs(eigvals.imag) > 0.01
        complex_eigs = eigvals[complex_mask]
        
        if len(complex_eigs) > 0:
            phases = np.abs(np.degrees(np.angle(complex_eigs)))
            # Only keep phases in [0, 180] (symmetric)
            phases = phases[phases <= 180]
            all_phases_deg.extend(phases)
            all_phases_by_block[(stage_idx, block_idx)] = phases

all_phases_deg = np.array(all_phases_deg)
print(f'Total complex eigenvalue phases: {len(all_phases_deg)}')
print(f'Mean: {all_phases_deg.mean():.1f}°, Std: {all_phases_deg.std():.1f}°')


# ================================================================
# STEP 2: φ-lattice definition
# ================================================================
print()
print('=' * 70)
print('STEP 2: φ-Lattice Positions')
print('=' * 70)
print()

# All φ-lattice angles in [0, 180]
phi_lattice = sorted(set(
    [180.0 / PHI**n for n in range(1, 8)] +
    [360.0 / PHI**n for n in range(1, 8) if 360.0/PHI**n <= 180]
))
phi_lattice = [a for a in phi_lattice if 1 < a < 179]

print(f'φ-lattice angles ({len(phi_lattice)}):')
for a in phi_lattice:
    # Find which formula
    for base in [180, 360]:
        for n in range(1, 8):
            if abs(a - base/PHI**n) < 0.01:
                print(f'  {a:>7.2f}° = {base}/φ^{n}')
                break


# ================================================================
# STEP 3: Nearest-neighbor distance to φ-lattice
# ================================================================
print()
print('=' * 70)
print('STEP 3: Nearest-Neighbor Distance to φ-Lattice')
print('=' * 70)
print()

def nearest_lattice_distance(phases, lattice):
    """Mean distance from each phase to nearest lattice point."""
    dists = []
    for p in phases:
        min_dist = min(abs(p - l) for l in lattice)
        dists.append(min_dist)
    return np.array(dists)

real_dists = nearest_lattice_distance(all_phases_deg, phi_lattice)
real_mean = real_dists.mean()

# Bootstrap: generate random phases with same distribution shape
# Test 1: uniform random in [0, 180]
n_bootstrap = 10000
random_means = []
for _ in range(n_bootstrap):
    random_phases = np.random.uniform(0, 180, len(all_phases_deg))
    random_dists = nearest_lattice_distance(random_phases, phi_lattice)
    random_means.append(random_dists.mean())

random_means = np.array(random_means)
p_value_nn = np.mean(random_means <= real_mean)

print(f'Real data mean nearest-neighbor distance: {real_mean:.3f}°')
print(f'Random uniform mean: {np.mean(random_means):.3f}° ± {np.std(random_means):.3f}°')
print(f'p-value (one-sided, real ≤ random): {p_value_nn:.6f}')
print(f'→ {"SIGNIFICANT" if p_value_nn < 0.05 else "NOT SIGNIFICANT"} at α=0.05')

# Test 2: compare to random matrices with SAME singular value spectrum
print()
print('Bootstrap with matched SV spectrum:')

# Use one representative block's SVs to generate random matrices
prefix = 'encoder.arch.stages.2.4'
w1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
w2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
W_eff_ref = w2 @ w1
_, S_ref, _ = np.linalg.svd(W_eff_ref)

random_matched_means = []
for _ in range(1000):
    # Random orthogonal matrices
    Q1, _ = np.linalg.qr(np.random.randn(W_eff_ref.shape[0], W_eff_ref.shape[0]))
    Q2, _ = np.linalg.qr(np.random.randn(W_eff_ref.shape[0], W_eff_ref.shape[0]))
    # Random matrix with same SVs
    W_rand = Q1 @ np.diag(S_ref) @ Q2
    eigvals_rand = np.linalg.eigvals(W_rand)
    complex_mask = np.abs(eigvals_rand.imag) > 0.01
    if complex_mask.sum() > 0:
        phases_rand = np.abs(np.degrees(np.angle(eigvals_rand[complex_mask])))
        phases_rand = phases_rand[phases_rand <= 180]
        if len(phases_rand) > 0:
            dists_rand = nearest_lattice_distance(phases_rand, phi_lattice)
            random_matched_means.append(dists_rand.mean())

if random_matched_means:
    random_matched_means = np.array(random_matched_means)
    # Compare block 2.4 specifically
    block_phases = all_phases_by_block.get((2, 4), np.array([]))
    if len(block_phases) > 0:
        block_mean = nearest_lattice_distance(block_phases, phi_lattice).mean()
        p_matched = np.mean(random_matched_means <= block_mean)
        print(f'Block 2.4 mean NN distance: {block_mean:.3f}°')
        print(f'Matched-SV random mean: {np.mean(random_matched_means):.3f}° ± {np.std(random_matched_means):.3f}°')
        print(f'p-value (matched): {p_matched:.6f}')
        print(f'→ {"SIGNIFICANT" if p_matched < 0.05 else "NOT SIGNIFICANT"}')


# ================================================================
# STEP 4: Histogram test — binned comparison
# ================================================================
print()
print('=' * 70)
print('STEP 4: Phase Histogram Analysis')
print('=' * 70)
print()

bins = np.arange(0, 185, 5)
hist, edges = np.histogram(all_phases_deg, bins=bins)
hist_frac = hist / hist.sum()

# Expected from uniform
expected_frac = np.ones_like(hist_frac) / len(hist_frac)

# KS test against uniform
ks_stat, ks_p = kstest(all_phases_deg / 180, 'uniform')
print(f'KS test against uniform: stat={ks_stat:.4f}, p={ks_p:.6f}')
print(f'→ {"NOT uniform" if ks_p < 0.05 else "Consistent with uniform"}')

# Chi-squared test
chi2_stat, chi2_p = chisquare(hist, f_exp=np.ones_like(hist) * hist.sum() / len(hist))
print(f'Chi-squared test: stat={chi2_stat:.2f}, p={chi2_p:.6f}')
print(f'→ {"NOT uniform" if chi2_p < 0.05 else "Consistent with uniform"}')

# Show histogram with φ-lattice markers
print(f'\nPhase histogram (5° bins):')
for i in range(len(hist)):
    center = (edges[i] + edges[i+1]) / 2
    bar = '█' * int(hist_frac[i] * 200)
    
    # Check if near φ-lattice
    lattice_mark = ""
    for a in phi_lattice:
        if abs(center - a) < 2.5:
            for base in [180, 360]:
                for n in range(1, 8):
                    if abs(a - base/PHI**n) < 0.01:
                        lattice_mark = f" ← {base}/φ^{n}"
    
    density_mark = ""
    if hist_frac[i] > np.median(hist_frac) * 1.5:
        density_mark = " [PEAK]"
    elif hist_frac[i] < np.median(hist_frac) * 0.5:
        density_mark = " [GAP]"
    
    print(f'  {edges[i]:>5.0f}-{edges[i+1]:>3.0f}°: {hist_frac[i]:.4f} {bar}{lattice_mark}{density_mark}')


# ================================================================
# STEP 5: Per-block consistency
# ================================================================
print()
print('=' * 70)
print('STEP 5: Per-Block Phase Distribution Consistency')
print('=' * 70)
print()

# Are the phase distributions consistent across blocks?
# (If φ-lattice is universal, all blocks should show same pattern)

block_hists = []
for key in sorted(all_phases_by_block.keys()):
    phases = all_phases_by_block[key]
    if len(phases) < 20:
        continue
    h, _ = np.histogram(phases, bins=bins)
    h_frac = h / h.sum()
    block_hists.append(h_frac)

if len(block_hists) > 1:
    block_hists = np.array(block_hists)
    # Cross-block correlation
    from itertools import combinations
    cors = []
    for i, j in combinations(range(len(block_hists)), 2):
        cors.append(np.corrcoef(block_hists[i], block_hists[j])[0, 1])
    
    print(f'Cross-block histogram correlation: {np.mean(cors):.4f} ± {np.std(cors):.4f}')
    print(f'  (1.0 = identical across blocks, 0.0 = independent)')
    
    if np.mean(cors) > 0.7:
        print('  → The phase distribution is CONSISTENT across blocks (universal)')
    elif np.mean(cors) > 0.4:
        print('  → The phase distribution is MODERATELY consistent')
    else:
        print('  → The phase distribution VARIES across blocks')


# ================================================================
# STEP 6: What are the phases — peaks and gaps analysis
# ================================================================
print()
print('=' * 70)
print('STEP 6: Peak and Gap Structure')
print('=' * 70)
print()

# Find peaks and gaps in the histogram
from scipy.signal import find_peaks

# Smooth histogram slightly
from scipy.ndimage import gaussian_filter1d
smoothed = gaussian_filter1d(hist_frac, sigma=1)

peak_indices, peak_props = find_peaks(smoothed, height=np.median(smoothed) * 1.2)
gap_indices, _ = find_peaks(-smoothed, height=-np.median(smoothed) * 0.8)

peak_angles = [(edges[i] + edges[i+1]) / 2 for i in peak_indices]
gap_angles = [(edges[i] + edges[i+1]) / 2 for i in gap_indices]

print(f'Peaks at: {[f"{a:.0f}°" for a in peak_angles]}')
print(f'Gaps at:  {[f"{a:.0f}°" for a in gap_angles]}')

# Do peaks/gaps align with φ-lattice?
print(f'\nPeak alignment with φ-lattice:')
for pa in peak_angles:
    nearest = min(phi_lattice, key=lambda l: abs(pa - l))
    dist = abs(pa - nearest)
    for base in [180, 360]:
        for n in range(1, 8):
            if abs(nearest - base/PHI**n) < 0.01:
                print(f'  Peak {pa:.0f}° → nearest lattice: {nearest:.1f}° ({base}/φ^{n}), dist={dist:.1f}°')

print(f'\nGap alignment with φ-lattice:')
for ga in gap_angles:
    nearest = min(phi_lattice, key=lambda l: abs(ga - l))
    dist = abs(ga - nearest)
    for base in [180, 360]:
        for n in range(1, 8):
            if abs(nearest - base/PHI**n) < 0.01:
                print(f'  Gap {ga:.0f}° → nearest lattice: {nearest:.1f}° ({base}/φ^{n}), dist={dist:.1f}°')


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 9C SUMMARY: Eigenvalue Phase Verification')
print('=' * 70)
print()
print(f'Total eigenvalue phases analyzed: {len(all_phases_deg)}')
print(f'KS test against uniform: p={ks_p:.6f} → {"NOT uniform" if ks_p < 0.05 else "uniform"}')
print(f'Nearest-neighbor to φ-lattice: {real_mean:.3f}° (random: {np.mean(random_means):.3f}°)')
print(f'NN p-value: {p_value_nn:.6f} → {"SIGNIFICANT" if p_value_nn < 0.05 else "NOT significant"}')
if len(block_hists) > 1:
    print(f'Cross-block consistency: {np.mean(cors):.4f}')
print()

if ks_p < 0.05:
    print('The eigenvalue phases are NOT uniformly distributed.')
    if p_value_nn < 0.05:
        print('They cluster CLOSER to the φ-lattice than random → φ-structure confirmed.')
    else:
        print('But they do NOT cluster at φ-lattice positions specifically.')
        print('The non-uniformity may be due to the SV spectrum, not φ.')
else:
    print('The eigenvalue phases are consistent with uniform distribution.')
    print('The apparent clustering in Phase 9B was likely an artifact of small sample size.')
