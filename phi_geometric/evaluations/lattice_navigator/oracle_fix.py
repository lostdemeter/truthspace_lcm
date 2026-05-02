"""
Oracle Fix: Per-Pixel Color Wheel Application

The gray patch on the elephant's head is caused by activation cancellation:
when a region spans two query territories, averaging activations BEFORE
applying color weights causes them to cancel.

Fix: apply color wheel per-pixel, THEN smooth within regions.

Also: diagnose exactly what causes remaining error in the oracle.
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer
from ks_v2_damping import segment_by_edges
from territory_mapper import get_ddcolor_territories

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()

print('=== ORACLE FIX: Per-Pixel Color Wheel ===')
print()

v16 = V16GeometricColorizer()

refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()
color_wheel = refine_w[:, :100].T  # [100, 2]
input_weights = refine_w[:, 100:]  # [2, 3]

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in range(66, 74)]

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/oracle_fixed'
os.makedirs(out_dir, exist_ok=True)

def oracle_v1(gray, color_maps, color_wheel):
    """Original: per-region averaging (causes gray patches)."""
    h, w = gray.shape
    labeled, _ = segment_by_edges(gray)
    ab_out = np.zeros((h, w, 2))
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 5: continue
        region_acts = color_maps[:, mask].mean(axis=1)
        ab_out[:,:,0][mask] = (region_acts * color_wheel[:, 0]).sum()
        ab_out[:,:,1][mask] = (region_acts * color_wheel[:, 1]).sum()
    
    for ch in range(2):
        ab_out[:,:,ch] = cv2.bilateralFilter(ab_out[:,:,ch].astype(np.float32), 9, 30, 30)
    return ab_out

def oracle_v2_pixel(color_maps, color_wheel, refine_b):
    """Fixed: per-pixel color wheel application. No region averaging."""
    # color_maps: [100, H, W]
    # color_wheel: [100, 2]
    # ab(h,w) = Σ_q color_maps[q,h,w] * color_wheel[q, :] + bias
    
    H, W = color_maps.shape[1], color_maps.shape[2]
    
    # Efficient: matrix multiply
    # Reshape color_maps to [100, H*W], multiply by color_wheel.T [2, 100] → [2, H*W]
    cm_flat = color_maps.reshape(100, -1)  # [100, H*W]
    ab_flat = color_wheel.T @ cm_flat      # [2, H*W]
    
    ab_out = ab_flat.reshape(2, H, W).transpose(1, 2, 0)  # [H, W, 2]
    ab_out[:,:,0] += refine_b[0]
    ab_out[:,:,1] += refine_b[1]
    
    return ab_out

def oracle_v3_pixel_plus_input(color_maps, color_wheel, input_weights, refine_b, img_tensor):
    """Full oracle: per-pixel color wheel + input channel contribution."""
    H, W = color_maps.shape[1], color_maps.shape[2]
    
    # Query contribution
    cm_flat = color_maps.reshape(100, -1)
    ab_flat = color_wheel.T @ cm_flat  # [2, H*W]
    
    # Input channel contribution
    # input_weights: [2, 3], img_tensor: [3, H, W]
    inp = img_tensor.squeeze(0).numpy()  # [3, H, W]
    inp_resized = np.stack([cv2.resize(inp[c], (W, H)) for c in range(3)])  # [3, H, W]
    inp_flat = inp_resized.reshape(3, -1)  # [3, H*W]
    ab_input = input_weights @ inp_flat    # [2, H*W]
    
    ab_total = ab_flat + ab_input
    ab_out = ab_total.reshape(2, H, W).transpose(1, 2, 0)
    ab_out[:,:,0] += refine_b[0]
    ab_out[:,:,1] += refine_b[1]
    
    return ab_out

print('Testing three oracle versions...\n')

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    color_maps, ab_ddcolor = get_ddcolor_territories(v16, img_tensor)
    
    # Three oracle versions
    ab_v1 = oracle_v1(gray, color_maps, color_wheel)
    ab_v2 = oracle_v2_pixel(color_maps, color_wheel, refine_b)
    ab_v3 = oracle_v3_pixel_plus_input(color_maps, color_wheel, input_weights, refine_b, img_tensor)
    
    err_v1 = np.sqrt(np.mean((ab_v1 - ab_ddcolor)**2))
    err_v2 = np.sqrt(np.mean((ab_v2 - ab_ddcolor)**2))
    err_v3 = np.sqrt(np.mean((ab_v3 - ab_ddcolor)**2))
    
    bgr_v1 = ab_to_bgr(ab_v1, L)
    bgr_v2 = ab_to_bgr(ab_v2, L)
    bgr_v3 = ab_to_bgr(ab_v3, L)
    bgr_dd = ab_to_bgr(ab_ddcolor, L)
    
    print(f'{name}: v1(region)={err_v1:.3f}, v2(pixel)={err_v2:.3f}, v3(pixel+input)={err_v3:.3f}')
    
    imgs = [
        (bgr_v1, f'v1 region e={err_v1:.2f}'),
        (bgr_v2, f'v2 pixel e={err_v2:.3f}'),
        (bgr_v3, f'v3 full e={err_v3:.4f}'),
        (bgr_dd, 'DDColor'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'oracle_{name}.jpg'), strip)

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
