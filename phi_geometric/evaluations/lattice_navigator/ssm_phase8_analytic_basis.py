"""
Phase 8: Analytic φ-Radial Basis Functions — The Ultimate Hypothesis Test

Phase 7 discovered:
  - Depthwise conv spatial basis functions decay as φ^(-αd) from center
  - α progresses: early blocks α≈2-5, middle α≈φ, late α≈1
  - Basis 1 center = -0.618 = -1/φ
  - Only 3 basis functions needed for 90% variance

THE TEST: Can we replace LEARNED depthwise conv kernels with ANALYTIC
φ-radial basis functions constructed from first principles?

If YES → φ-geometry is sufficient for spatial mixing (hypothesis confirmed)
If NO  → learned kernels contain information beyond φ-geometry (hypothesis limits)

Construction:
  ψ_k(x,y) = φ^(-α_k × d(x,y)) × cos(2π × f_k × θ(x,y) + phase_k)
  
  where:
    d(x,y) = distance from center of 7×7 grid
    θ(x,y) = angle from center
    α_k = radial decay rate (from {1/φ, 1, φ, 2})
    f_k = angular frequency (from {0, 1, 2, 3})
    phase_k = phase offset

  kernel = Σ c_k × ψ_k(x,y)   where c_k are FIT to match learned kernels
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
import time
from scipy.optimize import least_squares
from scipy.stats import wilcoxon

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from geometric_colorizer_v16_convnext import V16GeometricColorizer
from geometric_colorizer_v17_minimal import V17MinimalColorizer

PHI = (1 + np.sqrt(5)) / 2
SZ = 256

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]


def compute_rmse(pred_tensor, gt_ab):
    pred_ab = pred_tensor[0, :2].permute(1, 2, 0).numpy()
    pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    return np.sqrt(np.mean((pred_r - gt_ab)**2))


# Load test data
test_data = []
for idx in range(300, 400):
    if len(test_data) >= 30:
        break
    im = cv2.imread(all_imgs[idx])
    if im is None:
        continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append({'tensor': t, 'gt_ab': gt_ab})

print(f'Test set: {len(test_data)} images')


# ================================================================
# STEP 1: Construct analytic φ-radial basis functions
# ================================================================
print()
print('=' * 70)
print('STEP 1: Constructing Analytic φ-Radial Basis Functions')
print('=' * 70)
print()

# Create the 7×7 spatial grid
ys, xs = np.mgrid[0:7, 0:7]
center_y, center_x = 3.0, 3.0
dy = ys - center_y
dx = xs - center_x
dist = np.sqrt(dy**2 + dx**2)
theta = np.arctan2(dy, dx)  # [-π, π]

# Normalize distance to [0, 1] range (max dist ≈ 4.24)
dist_norm = dist / dist.max()

# Define basis function families
# Family 1: Pure radial — φ^(-α×d) for various α
# Family 2: Radial × angular — φ^(-α×d) × cos(f×θ + phase)
# Family 3: Separable — φ^(-α×|x|) × φ^(-β×|y|)

def phi_radial(alpha):
    """Pure radial: φ^(-α×d)"""
    return PHI ** (-alpha * dist)

def phi_radial_angular(alpha, freq, phase=0):
    """Radial × angular: φ^(-α×d) × cos(f×θ + phase)"""
    return PHI ** (-alpha * dist) * np.cos(freq * theta + phase)

def phi_separable(alpha_x, alpha_y):
    """Separable: φ^(-α×|x|) × φ^(-β×|y|)"""
    return PHI ** (-alpha_x * np.abs(dx)) * PHI ** (-alpha_y * np.abs(dy))

# Generate a rich set of basis functions
basis_functions = []
basis_labels = []

# Pure radial with various decay rates
for alpha in [1/PHI, 1.0, PHI, 2.0, 3.0]:
    b = phi_radial(alpha)
    b = b / (np.linalg.norm(b) + 1e-10)
    basis_functions.append(b.flatten())
    basis_labels.append(f'R(α={alpha:.3f})')

# Radial × angular with various frequencies and phases
for alpha in [1/PHI, 1.0, PHI]:
    for freq in [1, 2, 3, 4]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            b = phi_radial_angular(alpha, freq, phase)
            b = b / (np.linalg.norm(b) + 1e-10)
            basis_functions.append(b.flatten())
            basis_labels.append(f'RA(α={alpha:.2f},f={freq},φ={phase:.2f})')

# Separable with φ-related rates
for ax in [1/PHI, 1.0, PHI]:
    for ay in [1/PHI, 1.0, PHI]:
        b = phi_separable(ax, ay)
        b = b / (np.linalg.norm(b) + 1e-10)
        basis_functions.append(b.flatten())
        basis_labels.append(f'S(αx={ax:.2f},αy={ay:.2f})')

# Add constant (DC) basis
basis_functions.append(np.ones(49) / np.sqrt(49))
basis_labels.append('DC')

# φ-BBP inspired: arctan(1/φ) angle basis
phi_bbp_angle = np.arctan(1/PHI)  # ≈ 0.5536 rad ≈ 31.7°
for n in range(1, 5):
    b = np.cos(n * phi_bbp_angle * dist)
    b = b / (np.linalg.norm(b) + 1e-10)
    basis_functions.append(b.flatten())
    basis_labels.append(f'BBP(n={n})')

basis_matrix = np.array(basis_functions)  # [N_basis, 49]
N_BASIS = len(basis_functions)

print(f'Total analytic basis functions: {N_BASIS}')
print(f'  Pure radial: 5')
print(f'  Radial×angular: {3*4*4}')
print(f'  Separable: {3*3}')
print(f'  DC + BBP: {1 + 4}')


# ================================================================
# STEP 2: How well do analytic bases represent learned kernels?
# ================================================================
print()
print('=' * 70)
print('STEP 2: Analytic Basis Fit to Learned Kernels')
print('=' * 70)
print()

# For each block, project learned kernels onto analytic basis
# and measure reconstruction error

print(f"{'Block':<10} {'Channels':<10} {'Fit R²':<10} {'RMSE fit':<10} "
      f"{'Top basis':<30}")
print("-" * 70)

all_fit_r2 = []
all_analytic_weights = {}  # Store for later use

for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w = v16._get_weight(f'{prefix}.dwconv.weight')
        if w is None:
            continue
        
        C = w.shape[0]
        w_flat = w.view(C, 49).numpy()  # [C, 49]
        
        # Project each channel onto the analytic basis
        # w_flat ≈ coeffs @ basis_matrix
        # coeffs = w_flat @ basis_matrix.T @ (basis_matrix @ basis_matrix.T)^-1
        # Or use least squares
        
        # Regularized least squares: coeffs = w_flat @ pinv(basis_matrix)
        basis_pinv = np.linalg.pinv(basis_matrix)  # [49, N_basis]
        coeffs = w_flat @ basis_pinv  # [C, N_basis]
        
        # Reconstruct
        w_recon = coeffs @ basis_matrix  # [C, 49]
        
        # R² per channel
        ss_res = np.sum((w_flat - w_recon)**2, axis=1)
        ss_tot = np.sum((w_flat - w_flat.mean(axis=1, keepdims=True))**2, axis=1)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        mean_r2 = r2.mean()
        
        rmse_fit = np.sqrt(np.mean((w_flat - w_recon)**2))
        
        # Top contributing basis functions (by mean |coefficient|)
        mean_abs_coeffs = np.abs(coeffs).mean(axis=0)
        top_indices = np.argsort(mean_abs_coeffs)[::-1][:3]
        top_labels = [f'{basis_labels[i]}' for i in top_indices]
        
        all_fit_r2.append(mean_r2)
        all_analytic_weights[(stage_idx, block_idx)] = {
            'coeffs': coeffs,
            'recon': w_recon,
            'shape': w.shape,
            'r2': mean_r2,
        }
        
        print(f"  {stage_idx}.{block_idx:<7} {C:<10} {mean_r2:<10.4f} {rmse_fit:<10.6f} "
              f"{', '.join(top_labels)}")

print(f"\nOverall fit R²: {np.mean(all_fit_r2):.4f} ± {np.std(all_fit_r2):.4f}")


# ================================================================
# STEP 3: Test analytic kernels in the full pipeline
# ================================================================
print()
print('=' * 70)
print('STEP 3: Analytic Kernels in Full Pipeline')
print('=' * 70)
print()

# Replace all depthwise conv kernels with their analytic approximations
# and measure RMSE

# Baseline
baseline_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    baseline_rmses.append(compute_rmse(pred, img_d['gt_ab']))
baseline = np.mean(baseline_rmses)
print(f'Baseline RMSE: {baseline:.3f}')

# Analytic replacement
original_get = v16._get_weight

def analytic_get(name, weights=all_analytic_weights, orig=original_get):
    for (si, bi), data in weights.items():
        if name == f'encoder.arch.stages.{si}.{bi}.dwconv.weight':
            return torch.from_numpy(data['recon'].reshape(data['shape'])).float()
    return orig(name)

v16._get_weight = analytic_get
analytic_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    analytic_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get

analytic_rmses = np.array(analytic_rmses)
baseline_rmses = np.array(baseline_rmses)

_, p_val = wilcoxon(baseline_rmses, analytic_rmses)
delta = (analytic_rmses.mean() - baseline) / baseline * 100
corr = np.corrcoef(baseline_rmses, analytic_rmses)[0, 1]

print(f'Analytic RMSE: {analytic_rmses.mean():.3f} ({delta:+.2f}%)')
print(f'p-value: {p_val:.4f} ({"SIG" if p_val < 0.05 else "NOT SIG"})')
print(f'Correlation: {corr:.4f}')


# ================================================================
# STEP 4: How few analytic bases do we need?
# ================================================================
print()
print('=' * 70)
print('STEP 4: Minimum Analytic Bases Needed')
print('=' * 70)
print()

# Try with fewer basis functions
# Rank the basis functions by their overall importance

# Global importance: sum of |coefficients| across all blocks and channels
global_importance = np.zeros(N_BASIS)
for (si, bi), data in all_analytic_weights.items():
    global_importance += np.abs(data['coeffs']).sum(axis=0)

importance_order = np.argsort(global_importance)[::-1]

print(f"Top 15 most important analytic basis functions:")
for i, idx in enumerate(importance_order[:15]):
    print(f"  {i+1:>2}. {basis_labels[idx]:<30} importance={global_importance[idx]:.2f}")

# Test with top-K bases
for K in [3, 5, 10, 15, 20, N_BASIS]:
    # Keep only top-K bases
    keep = importance_order[:K]
    sub_basis = basis_matrix[keep]  # [K, 49]
    sub_pinv = np.linalg.pinv(sub_basis)
    
    # Refit all blocks with reduced basis
    reduced_weights = {}
    for (si, bi), data in all_analytic_weights.items():
        w = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.dwconv.weight')
        C = w.shape[0]
        w_flat = w.view(C, 49).numpy()
        
        sub_coeffs = w_flat @ sub_pinv  # [C, K]
        w_recon = sub_coeffs @ sub_basis  # [C, 49]
        reduced_weights[(si, bi)] = {
            'recon': w_recon, 'shape': w.shape
        }
    
    def reduced_get(name, weights=reduced_weights, orig=original_get):
        for (si, bi), data in weights.items():
            if name == f'encoder.arch.stages.{si}.{bi}.dwconv.weight':
                return torch.from_numpy(data['recon'].reshape(data['shape'])).float()
        return orig(name)
    
    v16._get_weight = reduced_get
    rmses = []
    for img_d in test_data[:15]:
        with torch.no_grad():
            pred = v16.forward(img_d['tensor'])
        rmses.append(compute_rmse(pred, img_d['gt_ab']))
    v16._get_weight = original_get
    
    mean_rmse = np.mean(rmses)
    sub_baseline = np.mean(baseline_rmses[:15])
    delta_k = (mean_rmse - sub_baseline) / sub_baseline * 100
    
    # Params: K coefficients per channel per block
    total_coeffs = sum(dims[si] * K for (si, bi) in all_analytic_weights.keys())
    orig_params = sum(np.prod(data['shape']) for data in all_analytic_weights.values())
    
    print(f'  K={K:>3}: RMSE={mean_rmse:.3f} ({delta_k:+.2f}%), '
          f'params: {orig_params:,} → {total_coeffs:,} ({total_coeffs/orig_params*100:.1f}%)')


# ================================================================
# STEP 5: The PURE φ test — can 5 φ-radial bases do it?
# ================================================================
print()
print('=' * 70)
print('STEP 5: Pure φ-Radial Test (No Angular Components)')
print('=' * 70)
print()

# Use ONLY the 5 pure radial basis functions + DC
# This is the purest test of whether φ-radial decay is sufficient

pure_phi_indices = []
for i, label in enumerate(basis_labels):
    if label.startswith('R(') or label == 'DC':
        pure_phi_indices.append(i)

print(f"Pure φ-radial bases ({len(pure_phi_indices)}):")
for i in pure_phi_indices:
    print(f"  {basis_labels[i]}")

pure_basis = basis_matrix[pure_phi_indices]
pure_pinv = np.linalg.pinv(pure_basis)

pure_weights = {}
for (si, bi), data in all_analytic_weights.items():
    w = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.dwconv.weight')
    C = w.shape[0]
    w_flat = w.view(C, 49).numpy()
    
    coeffs = w_flat @ pure_pinv
    w_recon = coeffs @ pure_basis
    pure_weights[(si, bi)] = {'recon': w_recon, 'shape': w.shape}

def pure_get(name, weights=pure_weights, orig=original_get):
    for (si, bi), data in weights.items():
        if name == f'encoder.arch.stages.{si}.{bi}.dwconv.weight':
            return torch.from_numpy(data['recon'].reshape(data['shape'])).float()
    return orig(name)

v16._get_weight = pure_get
pure_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    pure_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get

pure_rmses = np.array(pure_rmses)
_, p_pure = wilcoxon(baseline_rmses, pure_rmses)
delta_pure = (pure_rmses.mean() - baseline) / baseline * 100

print(f'\nPure φ-radial RMSE: {pure_rmses.mean():.3f} ({delta_pure:+.2f}%)')
print(f'p-value: {p_pure:.4f} ({"SIG" if p_pure < 0.05 else "NOT SIG"})')
print(f'Correlation: {np.corrcoef(baseline_rmses, pure_rmses)[0,1]:.4f}')

# Compare: how much does the radial-only capture?
print(f'\n  Full analytic ({N_BASIS} bases): {analytic_rmses.mean():.3f} ({(analytic_rmses.mean()-baseline)/baseline*100:+.2f}%)')
print(f'  Pure φ-radial ({len(pure_phi_indices)} bases):  {pure_rmses.mean():.3f} ({delta_pure:+.2f}%)')
print(f'  The angular component adds: {(analytic_rmses.mean()-pure_rmses.mean())/pure_rmses.mean()*100:+.2f}% RMSE')


# ================================================================
# STEP 6: V19 — FULLY ANALYTIC colorizer
# ================================================================
print()
print('=' * 70)
print('STEP 6: V19 — Fully Analytic Colorizer')
print('=' * 70)
print()

# V19 = analytic φ-basis encoder + no transformer decoder
# This is the ULTIMATE test: can φ-geometry alone colorize images?

v17 = V17MinimalColorizer()
original_get_v17 = v17._get_weight

def v19_get(name, weights=all_analytic_weights, orig=original_get_v17):
    for (si, bi), data in weights.items():
        if name == f'encoder.arch.stages.{si}.{bi}.dwconv.weight':
            return torch.from_numpy(data['recon'].reshape(data['shape'])).float()
    return orig(name)

v17._get_weight = v19_get
v19_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v17.forward(img_d['tensor'])
    v19_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v17._get_weight = original_get_v17

v19_rmses = np.array(v19_rmses)
_, p_19 = wilcoxon(baseline_rmses, v19_rmses)

# V17 baseline
v17_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v17.forward(img_d['tensor'])
    v17_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v17_rmses = np.array(v17_rmses)

print(f"{'Version':<25} {'RMSE':<10} {'Δ vs V16':<12} {'p vs V16':<10} {'Corr':<8}")
print("-" * 65)
print(f"  {'V16 (full)':<23} {baseline:.3f}    {'—':<12} {'—':<10} {'—':<8}")
print(f"  {'V17 (no xfmr)':<23} {v17_rmses.mean():.3f}    {(v17_rmses.mean()-baseline)/baseline*100:+.2f}%      "
      f"{wilcoxon(baseline_rmses, v17_rmses)[1]:.4f}    {np.corrcoef(baseline_rmses, v17_rmses)[0,1]:.4f}")
print(f"  {'V19 (analytic+no xfmr)':<23} {v19_rmses.mean():.3f}    {(v19_rmses.mean()-baseline)/baseline*100:+.2f}%      "
      f"{p_19:.4f}    {np.corrcoef(baseline_rmses, v19_rmses)[0,1]:.4f}")

# What information does V19 KEEP vs LOSE?
print(f"\n  V19 vs V16 per-image comparison:")
v19_wins = np.sum(v19_rmses < baseline_rmses)
v16_wins = np.sum(baseline_rmses < v19_rmses)
print(f"    V19 wins: {v19_wins}/{len(baseline_rmses)}")
print(f"    V16 wins: {v16_wins}/{len(baseline_rmses)}")

# Parameter count for V19
# DW conv: N_BASIS coefficients per channel instead of 49 per channel
v19_dw_params = sum(dims[si] * N_BASIS for (si, bi) in all_analytic_weights.keys())
v16_dw_params = sum(np.prod(data['shape']) for data in all_analytic_weights.values())
v19_total = 55020784 - 14787072 + 25600 - v16_dw_params + v19_dw_params

print(f"\n  V19 DW conv params: {v16_dw_params:,} → {v19_dw_params:,} "
      f"({v19_dw_params/v16_dw_params*100:.1f}%)")
print(f"  V19 total params: {v19_total:,} (vs V16 {55020784:,})")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 8 SUMMARY: Analytic φ-Basis Functions')
print('=' * 70)
print()
print(f'Fit quality: R² = {np.mean(all_fit_r2):.4f} ({N_BASIS} analytic bases)')
print(f'Full analytic encoder: {analytic_rmses.mean():.3f} RMSE ({(analytic_rmses.mean()-baseline)/baseline*100:+.2f}%)')
print(f'Pure φ-radial only:    {pure_rmses.mean():.3f} RMSE ({delta_pure:+.2f}%)')
print(f'V19 (analytic+no xfmr): {v19_rmses.mean():.3f} RMSE ({(v19_rmses.mean()-baseline)/baseline*100:+.2f}%)')
print()

if abs((analytic_rmses.mean()-baseline)/baseline*100) < 5.0:
    print('✓ HYPOTHESIS SUPPORTED: φ-radial analytic basis functions')
    print('  can approximate learned spatial mixing with <5% quality loss.')
    print('  The depthwise conv learned φ-geometric structure.')
else:
    print('✗ HYPOTHESIS PARTIALLY REFUTED: learned kernels contain information')
    print('  beyond what φ-radial basis functions capture.')
    print(f'  Gap: {(analytic_rmses.mean()-baseline)/baseline*100:.1f}% quality loss.')

print()
if abs(delta_pure) < abs((analytic_rmses.mean()-baseline)/baseline*100) * 2:
    print('  The angular component is NOT essential — pure radial is nearly')
    print('  as good as radial+angular. The decay shape IS the information.')
else:
    print('  The angular component IS important — spatial orientation matters')
    print('  beyond just radial decay.')
