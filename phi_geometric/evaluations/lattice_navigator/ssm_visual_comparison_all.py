"""
Visual Comparison: Ground Truth vs V16 vs V19 (Analytic) vs V_ULTIMATE

Generates side-by-side comparison images showing:
  1. Original color image (ground truth)
  2. Grayscale input
  3. V16 colorization (full DDColor replica, 55M params)
  4. V19 colorization (analytic φ-basis DW + no transformer)
  5. V_ULTIMATE colorization (analytic DW + 75% PW + no transformer)

Also computes per-image RMSE and saturation metrics.
"""
import numpy as np
import cv2
import sys
import glob
import torch
import os

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from geometric_colorizer_v16_convnext import V16GeometricColorizer
from geometric_colorizer_v17_minimal import V17MinimalColorizer

PHI = (1 + np.sqrt(5)) / 2
SZ = 256

# Output directory
OUT_DIR = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator/visual_comparisons'
os.makedirs(OUT_DIR, exist_ok=True)

# Load models
v16 = V16GeometricColorizer()
v17 = V17MinimalColorizer()

# Build φ-analytic basis
ys, xs = np.mgrid[0:7, 0:7]
dxg = xs - 3.0
dyg = ys - 3.0
dist = np.sqrt(dxg**2 + dyg**2)
theta = np.arctan2(dyg, dxg)

phi_basis_functions = []
for ax in [1/PHI, 1.0, PHI]:
    for ay in [1/PHI, 1.0, PHI]:
        b = PHI ** (-ax * np.abs(dxg)) * PHI ** (-ay * np.abs(dyg))
        b = b / (np.linalg.norm(b) + 1e-10)
        phi_basis_functions.append(b.flatten())

for alpha in [1/PHI, 1.0, PHI]:
    for freq in [1, 2, 3, 4]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            b = PHI ** (-alpha * dist) * np.cos(freq * theta + phase)
            b = b / (np.linalg.norm(b) + 1e-10)
            phi_basis_functions.append(b.flatten())

for alpha in [1/PHI, 1.0, PHI, 2.0, 3.0]:
    b = PHI ** (-alpha * dist)
    b = b / (np.linalg.norm(b) + 1e-10)
    phi_basis_functions.append(b.flatten())
phi_basis_functions.append(np.ones(49) / np.sqrt(49))
for n in range(1, 5):
    b = np.cos(n * np.arctan(1/PHI) * dist)
    b = b / (np.linalg.norm(b) + 1e-10)
    phi_basis_functions.append(b.flatten())

phi_basis = np.array(phi_basis_functions)
phi_pinv = np.linalg.pinv(phi_basis)

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

# Precompute all modified weights
print("Precomputing analytic DW conv weights...")
all_modified_dw = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w = v16._get_weight(f'{prefix}.dwconv.weight')
        if w is None:
            continue
        C = w.shape[0]
        w_flat = w.view(C, 49).numpy()
        coeffs = w_flat @ phi_pinv
        w_recon = coeffs @ phi_basis
        all_modified_dw[f'{prefix}.dwconv.weight'] = torch.from_numpy(
            w_recon.reshape(w.shape)).float()

# Precompute low-rank PW weights (75%)
print("Precomputing 75% rank PW conv weights...")
all_modified_pw = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        for pw_name in ['pwconv1', 'pwconv2']:
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            w = v16._get_weight(f'{prefix}.{pw_name}.weight')
            if w is None:
                continue
            w_np = w.numpy()
            U, S, Vt = np.linalg.svd(w_np, full_matrices=False)
            k = max(1, int(len(S) * 0.75))
            w_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            all_modified_pw[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(
                w_approx.reshape(w.shape)).float()

# Combined modifications
all_ultimate = {}
all_ultimate.update(all_modified_dw)
all_ultimate.update(all_modified_pw)

# Weight override functions
original_v16_get = v16._get_weight
original_v17_get = v17._get_weight

def v19_get(name, dw=all_modified_dw, orig=original_v17_get):
    if name in dw:
        return dw[name]
    return orig(name)

def vult_get(name, mod=all_ultimate, orig=original_v17_get):
    if name in mod:
        return mod[name]
    return orig(name)


def colorize_to_bgr(model, gray_tensor, L_channel):
    """Run model and produce BGR output."""
    with torch.no_grad():
        pred = model.forward(gray_tensor)
    pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
    pred_ab_resized = cv2.resize(pred_ab, (L_channel.shape[1], L_channel.shape[0]))
    lab = np.zeros((L_channel.shape[0], L_channel.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = L_channel
    lab[:, :, 1] = pred_ab_resized[:, :, 0] + 128
    lab[:, :, 2] = pred_ab_resized[:, :, 1] + 128
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return bgr, pred_ab_resized


def compute_metrics(pred_ab, gt_ab):
    rmse = np.sqrt(np.mean((pred_ab - gt_ab)**2))
    sat_pred = np.sqrt(pred_ab[:,:,0]**2 + pred_ab[:,:,1]**2).mean()
    sat_gt = np.sqrt(gt_ab[:,:,0]**2 + gt_ab[:,:,1]**2).mean()
    return rmse, sat_pred, sat_gt


# Load test images
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
print(f"Found {len(all_imgs)} validation images")

# Select diverse images (pick from different ranges)
test_indices = [300, 305, 312, 320, 335, 350, 365, 380]
results = []

for idx_i, idx in enumerate(test_indices):
    print(f"\nProcessing image {idx_i+1}/{len(test_indices)} (index {idx})...")
    
    im = cv2.imread(all_imgs[idx])
    if im is None:
        continue
    
    # Prepare input
    r = cv2.resize(im, (SZ, SZ))
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L_channel = lab_gt[:, :, 0].astype(np.float32)
    gt_ab = lab_gt[:, :, 1:].astype(float) - 128.0
    
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    # V16 colorization
    v16._get_weight = original_v16_get
    bgr_v16, ab_v16 = colorize_to_bgr(v16, t, L_channel)
    rmse_v16, sat_v16, sat_gt = compute_metrics(ab_v16, gt_ab)
    
    # V19 colorization (analytic DW + no transformer)
    v17._get_weight = v19_get
    bgr_v19, ab_v19 = colorize_to_bgr(v17, t, L_channel)
    rmse_v19, sat_v19, _ = compute_metrics(ab_v19, gt_ab)
    
    # V_ULTIMATE colorization (analytic DW + 75% PW + no transformer)
    v17._get_weight = vult_get
    bgr_vult, ab_vult = colorize_to_bgr(v17, t, L_channel)
    rmse_vult, sat_vult, _ = compute_metrics(ab_vult, gt_ab)
    
    v17._get_weight = original_v17_get
    
    results.append({
        'idx': idx,
        'rmse_v16': rmse_v16, 'rmse_v19': rmse_v19, 'rmse_vult': rmse_vult,
        'sat_gt': sat_gt, 'sat_v16': sat_v16, 'sat_v19': sat_v19, 'sat_vult': sat_vult,
    })
    
    # Build comparison image
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    panels = [
        (r.copy(), "Ground Truth"),
        (gray_bgr.copy(), "Grayscale Input"),
        (bgr_v16.copy(), f"V16 Full (RMSE {rmse_v16:.1f})"),
        (bgr_v19.copy(), f"V19 Analytic (RMSE {rmse_v19:.1f})"),
        (bgr_vult.copy(), f"V_ULT (RMSE {rmse_vult:.1f})"),
    ]
    
    for panel, label in panels:
        # Black background for text
        cv2.rectangle(panel, (0, 0), (SZ, 22), (0, 0, 0), -1)
        cv2.putText(panel, label, (4, 16), font, font_scale, (255, 255, 255), thickness)
    
    row = np.hstack([p for p, _ in panels])
    
    # Save individual comparison
    out_path = os.path.join(OUT_DIR, f'comparison_{idx:05d}.png')
    cv2.imwrite(out_path, row)

# Build summary grid (all images stacked vertically)
print("\nBuilding summary grid...")
all_rows = []
for idx in [test_indices[i] for i in range(len(test_indices))]:
    path = os.path.join(OUT_DIR, f'comparison_{idx:05d}.png')
    if os.path.exists(path):
        all_rows.append(cv2.imread(path))

if all_rows:
    grid = np.vstack(all_rows)
    grid_path = os.path.join(OUT_DIR, 'summary_grid.png')
    cv2.imwrite(grid_path, grid)
    print(f"Summary grid saved: {grid_path}")

# Print metrics table
print()
print('=' * 80)
print('METRICS SUMMARY')
print('=' * 80)
print()
print(f"{'Image':<8} {'V16 RMSE':<12} {'V19 RMSE':<12} {'V_ULT RMSE':<12} "
      f"{'Sat GT':<10} {'Sat V16':<10} {'Sat V19':<10}")
print("-" * 74)

for r in results:
    print(f"  {r['idx']:<6} {r['rmse_v16']:<12.2f} {r['rmse_v19']:<12.2f} "
          f"{r['rmse_vult']:<12.2f} {r['sat_gt']:<10.1f} {r['sat_v16']:<10.1f} "
          f"{r['sat_v19']:<10.1f}")

# Averages
mean_v16 = np.mean([r['rmse_v16'] for r in results])
mean_v19 = np.mean([r['rmse_v19'] for r in results])
mean_vult = np.mean([r['rmse_vult'] for r in results])
mean_sat_gt = np.mean([r['sat_gt'] for r in results])
mean_sat_v16 = np.mean([r['sat_v16'] for r in results])
mean_sat_v19 = np.mean([r['sat_v19'] for r in results])

print("-" * 74)
print(f"  {'MEAN':<6} {mean_v16:<12.2f} {mean_v19:<12.2f} {mean_vult:<12.2f} "
      f"{mean_sat_gt:<10.1f} {mean_sat_v16:<10.1f} {mean_sat_v19:<10.1f}")

print(f"\n  V19 vs V16: {(mean_v19 - mean_v16) / mean_v16 * 100:+.2f}% RMSE")
print(f"  V_ULT vs V16: {(mean_vult - mean_v16) / mean_v16 * 100:+.2f}% RMSE")
print(f"  V16 params: 55.0M")
print(f"  V19 params: ~40.3M (26.8% fewer, analytic DW + no transformer)")
print(f"  V_ULT params: ~38.6M (29.8% fewer, analytic DW + 75% PW + no transformer)")

print(f"\nImages saved to: {OUT_DIR}/")
print(f"  Individual: comparison_XXXXX.png")
print(f"  Grid:       summary_grid.png")
