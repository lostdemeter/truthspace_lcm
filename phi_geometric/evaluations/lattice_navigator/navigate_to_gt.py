"""
Navigate to Ground Truth

Every colorization is a configuration of 100 activation maps.
DDColor's output is one point. Ground truth is another.
The path between them reveals how shape stores information.

Plan:
1. Get DDColor's activation maps (forward pass)
2. Compute the CORRECTION activation maps that would produce GT colors
3. Study the correction: is it low-rank? Where does it concentrate?
4. Navigate the path: interpolate DDColor → GT and watch what happens
5. Can we steer at intermediate layers to reach GT more efficiently?
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer
from territory_mapper import get_ddcolor_territories

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def maps_to_ab(color_maps, color_wheel, input_weights, refine_b, img_tensor):
    """Convert activation maps to ab output using the 208 fixed numbers."""
    H, W = color_maps.shape[1], color_maps.shape[2]
    cm_flat = color_maps.reshape(100, -1)
    ab_flat = color_wheel.T @ cm_flat  # [2, H*W]
    
    inp = img_tensor.squeeze(0).numpy()
    inp_flat = inp.reshape(3, -1)
    ab_input = input_weights @ inp_flat
    
    ab_total = ab_flat + ab_input
    ab_out = ab_total.reshape(2, H, W).transpose(1, 2, 0)
    ab_out[:,:,0] += refine_b[0]
    ab_out[:,:,1] += refine_b[1]
    return ab_out


def compute_gt_correction(color_maps_dd, ab_gt, ab_dd, color_wheel):
    """
    Find the minimum-norm correction to activation maps that produces GT colors.
    
    We need: Σ_q (act_dd[q] + Δact[q]) × [w_a[q], w_b[q]] = ab_gt
    So:      Σ_q Δact[q] × [w_a[q], w_b[q]] = ab_gt - ab_dd = Δab
    
    This is: W @ Δact = Δab  (per pixel)
    Where W = color_wheel.T [2, 100]
    
    Minimum norm solution: Δact = W.T @ (W @ W.T)^{-1} @ Δab
    This correction lives in the 2D subspace spanned by W's rows.
    """
    H, W_img = color_maps_dd.shape[1], color_maps_dd.shape[2]
    
    # Color wheel: [100, 2], its transpose: [2, 100]
    Wt = color_wheel.T  # [2, 100]
    
    # Pseudo-inverse: W.T @ (W @ W.T)^{-1}
    WWT = Wt @ Wt.T  # [2, 2]
    WWT_inv = np.linalg.inv(WWT)
    pinv = Wt.T @ WWT_inv  # [100, 2]
    
    # Per-pixel correction
    delta_ab = (ab_gt - ab_dd).reshape(-1, 2)  # [H*W, 2]
    delta_act = delta_ab @ pinv.T  # [H*W, 100]
    delta_act = delta_act.T.reshape(100, H, W_img)  # [100, H, W]
    
    return delta_act


print('=== NAVIGATE TO GROUND TRUTH ===')
print()

v16 = V16GeometricColorizer()

refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()
color_wheel = refine_w[:, :100].T  # [100, 2]
input_weights = refine_w[:, 100:]  # [2, 3]

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in range(50, 66)]

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/navigation'
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# PART 1: DDColor → GT correction in activation-map space
# ============================================================
print('=== PART 1: DDColor → GT Correction Analysis ===\n')

all_corrections = []
all_results = []

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    
    # Ground truth ab (centered)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    
    # DDColor's activation maps and ab output
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    color_maps_dd, ab_dd = get_ddcolor_territories(v16, img_tensor)
    
    # Compute correction
    delta_act = compute_gt_correction(color_maps_dd, ab_gt, ab_dd, color_wheel)
    
    # Verify: corrected maps should produce GT
    maps_corrected = color_maps_dd + delta_act
    ab_corrected = maps_to_ab(maps_corrected, color_wheel, input_weights, refine_b, img_tensor)
    verify_err = np.sqrt(np.mean((ab_corrected - ab_gt)**2))
    
    # Analyze the correction
    delta_norm = np.sqrt(np.sum(delta_act**2, axis=0))  # [H, W] per-pixel correction magnitude
    maps_norm = np.sqrt(np.sum(color_maps_dd**2, axis=0))  # [H, W] original magnitude
    relative_correction = delta_norm / (maps_norm + 1e-8)
    
    # SVD of the correction maps
    delta_flat = delta_act.reshape(100, -1)  # [100, H*W]
    U, S, Vt = np.linalg.svd(delta_flat, full_matrices=False)
    total_var = (S**2).sum()
    cumvar = np.cumsum(S**2) / total_var
    rank1_pct = cumvar[0] * 100
    rank2_pct = cumvar[1] * 100
    rank5_pct = cumvar[4] * 100 if len(cumvar) > 4 else 100
    
    # DDColor error to GT
    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    
    print(f'  {name}:')
    print(f'    DDColor→GT error: {err_dd:.2f}')
    print(f'    Correction verify: {verify_err:.4f}')
    print(f'    Correction magnitude: mean={delta_norm.mean():.2f}, max={delta_norm.max():.2f}')
    print(f'    Relative correction: mean={relative_correction.mean():.2%}')
    print(f'    Correction SVD: rank1={rank1_pct:.1f}%, rank2={rank2_pct:.1f}%, rank5={rank5_pct:.1f}%')
    print(f'    Top 5 singular values: {S[:5].round(1)}')
    
    # Which queries need the biggest corrections?
    query_correction = np.sqrt(np.mean(delta_act**2, axis=(1,2)))  # [100]
    top5_queries = np.argsort(query_correction)[-5:][::-1]
    print(f'    Top corrected queries: {top5_queries} '
          f'(corrections: {query_correction[top5_queries].round(2)})')
    
    all_corrections.append(delta_act)
    all_results.append({
        'name': name, 'err_dd': err_dd, 'rank1': rank1_pct, 'rank2': rank2_pct,
        'delta_norm_mean': delta_norm.mean(), 'relative_mean': relative_correction.mean(),
    })

# ============================================================
# PART 2: Navigation — interpolate DDColor → GT
# ============================================================
print('\n=== PART 2: Navigation DDColor → GT (morph) ===\n')

# Pick a few representative images
for img_path in test_paths[:4]:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    color_maps_dd, ab_dd = get_ddcolor_territories(v16, img_tensor)
    
    delta_act = compute_gt_correction(color_maps_dd, ab_gt, ab_dd, color_wheel)
    
    # Interpolate: α=0 (DDColor) → α=1 (GT)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    frames = []
    for alpha in alphas:
        maps_interp = color_maps_dd + alpha * delta_act
        ab_interp = maps_to_ab(maps_interp, color_wheel, input_weights, refine_b, img_tensor)
        bgr_interp = ab_to_bgr(ab_interp, L)
        
        label = f'a={alpha:.2f}'
        cv2.putText(bgr_interp, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
        cv2.putText(bgr_interp, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        frames.append(bgr_interp)
    
    # Add GT for comparison
    bgr_gt = r.copy()
    cv2.putText(bgr_gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
    cv2.putText(bgr_gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    frames.append(bgr_gt)
    
    strip = np.hstack(frames)
    cv2.imwrite(os.path.join(out_dir, f'nav_{name}.jpg'), strip)
    print(f'  {name}: navigation strip saved (DDColor → GT in 5 steps)')

# ============================================================
# PART 3: Correction universality — do the same queries need
#         fixing across different images?
# ============================================================
print('\n=== PART 3: Cross-Image Correction Patterns ===\n')

# For each image, get the per-query correction magnitude
query_corrections = []
for delta in all_corrections:
    qc = np.sqrt(np.mean(delta**2, axis=(1,2)))
    query_corrections.append(qc)

qc_matrix = np.array(query_corrections)  # [N_images, 100]

# Which queries are CONSISTENTLY corrected across images?
mean_correction = qc_matrix.mean(axis=0)  # [100]
std_correction = qc_matrix.std(axis=0)    # [100]

top10 = np.argsort(mean_correction)[-10:][::-1]
print(f'Top 10 most-corrected queries (across all images):')
for q in top10:
    color_dir = np.degrees(np.arctan2(color_wheel[q, 1], color_wheel[q, 0]))
    mag = np.linalg.norm(color_wheel[q])
    print(f'  Query {q:3d}: mean_corr={mean_correction[q]:.3f} ± {std_correction[q]:.3f}, '
          f'color_angle={color_dir:.1f}°, color_mag={mag:.3f}')

# Correlation between images' correction patterns
corr_matrix = np.corrcoef(qc_matrix)
print(f'\nCorrection pattern correlation across images:')
print(f'  Mean: {corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean():.3f}')
print(f'  Min:  {corr_matrix[np.triu_indices_from(corr_matrix, k=1)].min():.3f}')

# ============================================================
# PART 4: Can we steer with just a FEW query corrections?
# ============================================================
print('\n=== PART 4: Low-Rank Steering ===\n')

for img_path in test_paths[:4]:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    color_maps_dd, ab_dd = get_ddcolor_territories(v16, img_tensor)
    
    delta_act = compute_gt_correction(color_maps_dd, ab_gt, ab_dd, color_wheel)
    err_full = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    
    # SVD of the correction
    delta_flat = delta_act.reshape(100, -1)
    U, S, Vt = np.linalg.svd(delta_flat, full_matrices=False)
    
    # Reconstruct with k modes
    frames = []
    print(f'  {name} (DDColor err={err_full:.2f}):')
    for k in [1, 2, 5, 10, 20, 100]:
        delta_k = (U[:, :k] * S[:k]) @ Vt[:k, :]
        delta_k = delta_k.reshape(100, SZ, SZ)
        maps_k = color_maps_dd + delta_k
        ab_k = maps_to_ab(maps_k, color_wheel, input_weights, refine_b, img_tensor)
        err_k = np.sqrt(np.mean((ab_k - ab_gt)**2))
        gap_closed = (1 - err_k / err_full) * 100
        print(f'    rank-{k:3d}: err={err_k:.2f}, gap_closed={gap_closed:.1f}%')
        
        if k in [1, 2, 5, 100]:
            bgr_k = ab_to_bgr(ab_k, L)
            label = f'rank-{k} {gap_closed:.0f}%'
            cv2.putText(bgr_k, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
            cv2.putText(bgr_k, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
            frames.append(bgr_k)
    
    bgr_dd = ab_to_bgr(ab_dd, L)
    cv2.putText(bgr_dd, f'DDColor', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
    cv2.putText(bgr_dd, f'DDColor', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    
    bgr_gt = r.copy()
    cv2.putText(bgr_gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
    cv2.putText(bgr_gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    
    strip = np.hstack([bgr_dd] + frames + [bgr_gt])
    cv2.imwrite(os.path.join(out_dir, f'steer_{name}.jpg'), strip)

# ============================================================
# PART 5: The geometry of the correction
# ============================================================
print('\n=== PART 5: Geometry of the Correction ===\n')

# Stack all corrections across images
all_delta_flat = []
for delta in all_corrections:
    all_delta_flat.append(delta.reshape(100, -1))

stacked = np.concatenate(all_delta_flat, axis=1)  # [100, total_pixels]
U_all, S_all, Vt_all = np.linalg.svd(stacked, full_matrices=False)

cumvar_all = np.cumsum(S_all**2) / (S_all**2).sum()
print(f'Cross-image correction SVD:')
for k in [1, 2, 3, 5, 10, 20]:
    if k <= len(cumvar_all):
        print(f'  rank-{k:2d}: {cumvar_all[k-1]*100:.1f}% variance')

# Singular value ratios
print(f'\nSingular value ratios:')
for i in range(min(10, len(S_all)-1)):
    ratio = S_all[i] / S_all[i+1]
    phi_err = abs(ratio - PHI) / PHI * 100
    print(f'  S[{i}]/S[{i+1}] = {ratio:.4f} (phi err: {phi_err:.1f}%)')

# The correction basis vectors — what do they look like in query space?
print(f'\nCorrection basis vectors (top 5 modes, query-space):')
for mode in range(min(5, U_all.shape[1])):
    u = U_all[:, mode]
    top3 = np.argsort(np.abs(u))[-3:][::-1]
    angles = [np.degrees(np.arctan2(color_wheel[q, 1], color_wheel[q, 0])) for q in top3]
    print(f'  Mode {mode}: queries [{top3[0]}, {top3[1]}, {top3[2]}] '
          f'(angles: {angles[0]:.0f}°, {angles[1]:.0f}°, {angles[2]:.0f}°), '
          f'S={S_all[mode]:.1f}')

# ============================================================
# Summary
# ============================================================
print('\n=== SUMMARY ===\n')

mean_err = np.mean([r['err_dd'] for r in all_results])
mean_rank1 = np.mean([r['rank1'] for r in all_results])
mean_rank2 = np.mean([r['rank2'] for r in all_results])
mean_rel = np.mean([r['relative_mean'] for r in all_results])

print(f'DDColor → GT:')
print(f'  Mean error: {mean_err:.2f}')
print(f'  Mean relative correction: {mean_rel:.2%}')
print(f'  Correction rank structure: rank-1={mean_rank1:.1f}%, rank-2={mean_rank2:.1f}%')
print(f'\nThe correction from DDColor to GT is a NAVIGABLE path in activation-map space.')
print(f'Low-rank approximations show how few dimensions of steering are needed.')
print(f'\nOutput saved to: {out_dir}/')
print('Done!')
