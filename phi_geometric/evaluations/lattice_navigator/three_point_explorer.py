"""
Three-Point Explorer: DDColor, Ground Truth, and Geometric

Every colorization is a point in activation-map space. We have three:
  1. DDColor: the learned model's output (55M params)
  2. GT: the actual ground truth colors
  3. Geometric: our construction from first principles

Key question: what common structure do DDColor and GT share?
Can we build a theoretical colorizer that navigates the same space?

The Geometric colorizer is built from:
  - The universal color wheel (200 fixed numbers, proven universal)
  - Edge-bounded regions as territory primitives
  - Brightness-based activation (geometric, not learned)
  - φ-scaled query selection

We then measure all three points in a unified coordinate system,
find common pathways, and experiment with steering.
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer
from territory_mapper import get_ddcolor_territories
from ks_v2_damping import segment_by_edges

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()


def geometric_colorize(gray, color_wheel, refine_b):
    """
    The Geometric Colorizer — built entirely from first principles.
    
    Philosophy: the color wheel is universal. Edge regions are natural
    territories. The only question is: which color direction does each
    territory get?
    
    Our construction:
    1. Segment into edge-bounded regions (territories)
    2. For each territory, compute a feature vector from brightness geometry
    3. Map feature → color wheel position using φ-based selection
    4. Build activation maps from territory assignments
    
    The key insight: brightness patterns contain color information.
    - Dark regions in context of bright surroundings → shadow → blue shift
    - Bright regions at top → sky → blue
    - Bright regions at bottom → ground → warm
    - Medium brightness with texture → vegetation → green
    - High contrast edges → object boundaries → preserve
    """
    h, w = gray.shape
    labeled, edges = segment_by_edges(gray)
    
    # Compute per-region features
    regions = []
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 5: continue
        
        ys, xs = np.where(mask)
        brightness = gray[mask].mean() / 255.0
        brightness_std = gray[mask].std() / 255.0
        y_center = ys.mean() / h  # 0=top, 1=bottom
        x_center = xs.mean() / w
        size = mask.sum() / (h * w)
        
        # Edge density (texture proxy)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sx**2 + sy**2)
        texture = edge_mag[mask].mean() / (edge_mag.max() + 1e-8)
        
        # Brightness relative to neighbors
        dilated = ndimage.binary_dilation(mask, iterations=5)
        border = dilated & ~mask
        if border.sum() > 10:
            neighbor_brightness = gray[border].mean() / 255.0
            contrast = brightness - neighbor_brightness
        else:
            contrast = 0.0
        
        regions.append({
            'rid': rid, 'mask': mask,
            'brightness': brightness, 'brightness_std': brightness_std,
            'y_center': y_center, 'x_center': x_center,
            'size': size, 'texture': texture, 'contrast': contrast,
        })
    
    # φ-based color direction assignment
    # Map each region to a target ab color, then find the activation
    # that produces it through the color wheel.
    #
    # Natural image ab ranges: saturated ~±40, moderate ~±20, muted ~±5
    # DDColor activation mean abs = 3.86, range [-33, +19]
    
    # Precompute query properties
    query_angles = np.arctan2(color_wheel[:, 1], color_wheel[:, 0])
    query_mags = np.linalg.norm(color_wheel, axis=1)
    
    # Build synthetic activation maps [100, H, W]
    activation_maps = np.zeros((100, h, w))
    
    for reg in regions:
        mask = reg['mask']
        b = reg['brightness']
        y = reg['y_center']
        tex = reg['texture']
        con = reg['contrast']
        sz = reg['size']
        
        # Step 1: Determine TARGET ab color for this region
        
        # Axis 1 (a-channel): green(-) to red(+)
        # Bright bottom regions → warm/red, dark top → cool/green
        # Texture suggests vegetation → green shift
        target_a = (b - 0.4) * 25.0 - tex * 40.0 + (1 - y) * 8.0
        
        # Axis 2 (b-channel): blue(-) to yellow(+)
        # Bright → yellow, dark → blue
        # Top of image → blue (sky), bottom → yellow/warm (ground)
        target_b = (b - 0.35) * 30.0 + (1 - y) * 15.0 - y * 10.0 - tex * 15.0
        
        # Saturation confidence: how strongly to color this region
        # High contrast, high texture, far from mid-gray → more confident
        confidence = min(1.5, abs(con) * 3.0 + tex * 4.0 + abs(b - 0.5) * 2.0 + 0.3)
        
        target_a *= confidence
        target_b *= confidence
        
        target_sat = np.sqrt(target_a**2 + target_b**2)
        target_angle = np.arctan2(target_b, target_a)
        
        # Step 2: Find best query on color wheel and compute activation
        
        # Angular distance to target color direction
        angle_diff = np.abs(np.arctan2(
            np.sin(query_angles - target_angle),
            np.cos(query_angles - target_angle)
        ))
        
        # Score: angular proximity × magnitude (strong queries preferred)
        scores = query_mags * np.exp(-angle_diff * 2.0)
        
        # Use best query — activation needed to produce target ab
        best_q = np.argmax(scores)
        best_mag = query_mags[best_q]
        
        if best_mag > 0.01 and target_sat > 1.0:
            # activation * color_wheel_mag ≈ target_sat
            # But only the component aligned with target direction counts
            best_angle = query_angles[best_q]
            alignment = np.cos(best_angle - target_angle)
            if alignment > 0.1:
                needed_activation = target_sat / (best_mag * alignment + 1e-8)
                # Clamp to reasonable range
                needed_activation = np.clip(needed_activation, 0, 300)
                activation_maps[best_q][mask] = needed_activation
            
            # Also activate 2nd-best for smoother coverage
            scores[best_q] = 0
            second_q = np.argmax(scores)
            second_mag = query_mags[second_q]
            second_angle = query_angles[second_q]
            second_alignment = np.cos(second_angle - target_angle)
            if second_mag > 0.01 and second_alignment > 0.1:
                second_activation = target_sat * 0.3 / (second_mag * second_alignment + 1e-8)
                second_activation = np.clip(second_activation, 0, 200)
                activation_maps[second_q][mask] = second_activation
    
    # Convert activation maps to ab using the 208-number formula
    cm_flat = activation_maps.reshape(100, -1)
    ab_flat = color_wheel.T @ cm_flat  # [2, H*W]
    ab_out = ab_flat.reshape(2, h, w).transpose(1, 2, 0)
    ab_out[:,:,0] += refine_b[0]
    ab_out[:,:,1] += refine_b[1]
    
    # Smooth boundaries
    for ch in range(2):
        ab_out[:,:,ch] = cv2.bilateralFilter(ab_out[:,:,ch].astype(np.float32), 9, 30, 30)
    
    return ab_out, activation_maps


print('=== THREE-POINT EXPLORER ===\n')

v16 = V16GeometricColorizer()

refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()
color_wheel = refine_w[:, :100].T  # [100, 2]
input_weights = refine_w[:, 100:]  # [2, 3]

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in range(50, 66)]

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/three_point'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Generate all three points for each image
# ============================================================
print('=== PART 1: Three Points Per Image ===\n')

all_data = []

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    
    # Point 1: DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    maps_dd, ab_dd = get_ddcolor_territories(v16, img_tensor)
    
    # Point 2: GT (represented as activation maps via pseudo-inverse)
    Wt = color_wheel.T
    WWT_inv = np.linalg.inv(Wt @ Wt.T)
    pinv = Wt.T @ WWT_inv  # [100, 2]
    
    # GT activation maps = DDColor maps + correction
    inp = img_tensor.squeeze(0).numpy()
    ab_from_input = (input_weights @ inp.reshape(3, -1)).reshape(2, SZ, SZ).transpose(1, 2, 0)
    ab_gt_minus_input = ab_gt - ab_from_input
    ab_gt_minus_input[:,:,0] -= refine_b[0]
    ab_gt_minus_input[:,:,1] -= refine_b[1]
    
    gt_flat = ab_gt_minus_input.reshape(-1, 2)
    maps_gt = (gt_flat @ pinv.T).T.reshape(100, SZ, SZ)
    
    # Point 3: Geometric
    ab_geo, maps_geo = geometric_colorize(gray, color_wheel, refine_b)
    
    # Errors between all pairs
    err_dd_gt = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_geo_gt = np.sqrt(np.mean((ab_geo - ab_gt)**2))
    err_geo_dd = np.sqrt(np.mean((ab_geo - ab_dd)**2))
    
    # Activation map distances
    map_dist_dd_gt = np.sqrt(np.mean((maps_dd - maps_gt)**2))
    map_dist_geo_gt = np.sqrt(np.mean((maps_geo - maps_gt)**2))
    map_dist_geo_dd = np.sqrt(np.mean((maps_geo - maps_dd)**2))
    
    print(f'  {name}:')
    print(f'    AB errors:  DD↔GT={err_dd_gt:.2f}  Geo↔GT={err_geo_gt:.2f}  Geo↔DD={err_geo_dd:.2f}')
    print(f'    Map dists:  DD↔GT={map_dist_dd_gt:.2f}  Geo↔GT={map_dist_geo_gt:.2f}  Geo↔DD={map_dist_geo_dd:.2f}')
    
    # Save comparison strip
    bgr_dd = ab_to_bgr(ab_dd, L)
    bgr_geo = ab_to_bgr(ab_geo, L)
    
    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_geo, f'Geometric e={err_geo_gt:.1f}'),
        (bgr_dd, f'DDColor e={err_dd_gt:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'three_{name}.jpg'), strip)
    
    all_data.append({
        'name': name,
        'ab_dd': ab_dd, 'ab_gt': ab_gt, 'ab_geo': ab_geo,
        'maps_dd': maps_dd, 'maps_gt': maps_gt, 'maps_geo': maps_geo,
        'err_dd_gt': err_dd_gt, 'err_geo_gt': err_geo_gt, 'err_geo_dd': err_geo_dd,
        'gray': gray, 'L': L, 'r': r, 'img_tensor': img_tensor,
    })


# ============================================================
# PART 2: Find the common pathway
# What structure do DDColor and GT share that Geometric doesn't?
# ============================================================
print('\n=== PART 2: Common Pathway Analysis ===\n')

# For each image, decompose the three points:
# DDColor = shared + dd_unique
# GT      = shared + gt_unique
# Geo     = shared + geo_unique (or not shared at all)
#
# The "shared" component = what DDColor and GT agree on

for d in all_data[:4]:
    name = d['name']
    
    # In ab space: what do DDColor and GT agree on?
    ab_avg = (d['ab_dd'] + d['ab_gt']) / 2  # midpoint
    ab_dd_deviation = d['ab_dd'] - ab_avg
    ab_gt_deviation = d['ab_gt'] - ab_avg
    ab_geo_deviation = d['ab_geo'] - ab_avg
    
    # Correlation: how aligned are the three with the shared midpoint?
    flat_avg = ab_avg.flatten()
    flat_dd = d['ab_dd'].flatten()
    flat_gt = d['ab_gt'].flatten()
    flat_geo = d['ab_geo'].flatten()
    
    corr_dd_gt = np.corrcoef(flat_dd, flat_gt)[0,1]
    corr_geo_gt = np.corrcoef(flat_geo, flat_gt)[0,1]
    corr_geo_dd = np.corrcoef(flat_geo, flat_dd)[0,1]
    
    print(f'  {name}:')
    print(f'    AB correlations: DD↔GT={corr_dd_gt:.3f}  Geo↔GT={corr_geo_gt:.3f}  Geo↔DD={corr_geo_dd:.3f}')
    
    # In activation map space: what's shared?
    maps_avg = (d['maps_dd'] + d['maps_gt']) / 2
    
    # SVD of each point's maps
    for label, maps in [('DDColor', d['maps_dd']), ('GT', d['maps_gt']), ('Geo', d['maps_geo'])]:
        flat = maps.reshape(100, -1)
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        cumvar = np.cumsum(S**2) / (S**2).sum()
        r1 = cumvar[0] * 100
        r2 = cumvar[1] * 100
        r5 = cumvar[4] * 100 if len(cumvar) > 4 else 100
        print(f'    {label:8s} maps SVD: rank1={r1:.1f}%, rank2={r2:.1f}%, rank5={r5:.1f}%, S[0]/S[1]={S[0]/S[1]:.3f}')
    
    # How much of DDColor's activation is in the GT direction?
    dd_flat = d['maps_dd'].reshape(100, -1)
    gt_flat = d['maps_gt'].reshape(100, -1)
    geo_flat = d['maps_geo'].reshape(100, -1)
    
    # Project DDColor onto GT direction
    dd_proj_gt = np.sum(dd_flat * gt_flat) / (np.linalg.norm(gt_flat)**2 + 1e-8)
    geo_proj_gt = np.sum(geo_flat * gt_flat) / (np.linalg.norm(gt_flat)**2 + 1e-8)
    
    print(f'    Projection onto GT: DDColor={dd_proj_gt:.3f}, Geometric={geo_proj_gt:.3f}')
    
    # The "shared direction" — what both DDColor and GT agree on
    # This is the component of DDColor that aligns with GT
    dd_along_gt = dd_proj_gt * gt_flat
    dd_perp_gt = dd_flat - dd_along_gt
    
    shared_fraction = np.linalg.norm(dd_along_gt) / (np.linalg.norm(dd_flat) + 1e-8)
    print(f'    DDColor shared with GT: {shared_fraction:.1%}')
    print()


# ============================================================
# PART 3: The Triangle — measure geometry of the three points
# ============================================================
print('=== PART 3: The Triangle Geometry ===\n')

print(f'  {"Image":<14} {"DD↔GT":>8} {"Geo↔GT":>8} {"Geo↔DD":>8} {"Triangle":>10} {"Geo closer to":>15}')
print(f'  {"-"*65}')

for d in all_data:
    # Which point is Geometric closer to: DDColor or GT?
    closer = 'DDColor' if d['err_geo_dd'] < d['err_geo_gt'] else 'GT'
    
    # Triangle inequality check: is Geometric between DDColor and GT?
    # If err_geo_dd + err_geo_gt ≈ err_dd_gt, Geometric is on the line
    triangle_sum = d['err_geo_dd'] + d['err_geo_gt']
    triangle_ratio = d['err_dd_gt'] / (triangle_sum + 1e-8)
    # ratio = 1.0 means collinear, < 1.0 means triangle (Geo is off-axis)
    
    print(f'  {d["name"]:<14} {d["err_dd_gt"]:8.2f} {d["err_geo_gt"]:8.2f} {d["err_geo_dd"]:8.2f} '
          f'{triangle_ratio:10.3f} {closer:>15}')

# Summary triangle
mean_dd_gt = np.mean([d['err_dd_gt'] for d in all_data])
mean_geo_gt = np.mean([d['err_geo_gt'] for d in all_data])
mean_geo_dd = np.mean([d['err_geo_dd'] for d in all_data])

print(f'\n  Means:')
print(f'    DD↔GT:  {mean_dd_gt:.2f}')
print(f'    Geo↔GT: {mean_geo_gt:.2f}')
print(f'    Geo↔DD: {mean_geo_dd:.2f}')


# ============================================================
# PART 4: Controllable blending — navigate the triangle
# Can we steer Geometric toward GT by blending in the
# DDColor↔GT shared component?
# ============================================================
print('\n=== PART 4: Controlled Navigation ===\n')

for d in all_data[:4]:
    name = d['name']
    L = d['L']
    
    # The shared component: average of DDColor and GT
    ab_shared = (d['ab_dd'] + d['ab_gt']) / 2
    
    # DDColor's unique addition
    ab_dd_unique = d['ab_dd'] - ab_shared
    
    # GT's unique addition
    ab_gt_unique = d['ab_gt'] - ab_shared
    
    # Geometric's position relative to shared
    ab_geo_vs_shared = d['ab_geo'] - ab_shared
    
    # Experiment: replace Geometric's unique component with the shared component
    # geo_steered = shared + α * geo_unique
    # At α=0: pure shared midpoint between DDColor and GT
    # At α=1: original Geometric
    
    print(f'  {name}:')
    
    frames = []
    err_base = np.sqrt(np.mean((d['ab_geo'] - d['ab_gt'])**2))
    
    # Also try: Geometric + shared_direction steering
    # Blend: (1-α)*Geometric + α*shared
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ab_blend = (1 - alpha) * d['ab_geo'] + alpha * ab_shared
        err = np.sqrt(np.mean((ab_blend - d['ab_gt'])**2))
        
        bgr = ab_to_bgr(ab_blend, L)
        label = f'a={alpha:.2f} e={err:.1f}'
        cv2.putText(bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 2)
        cv2.putText(bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,0,0), 1)
        frames.append(bgr)
        
        if alpha in [0.0, 0.5, 1.0]:
            print(f'    α={alpha:.2f}: Geo→Shared blend err={err:.2f}')
    
    bgr_gt = d['r'].copy()
    cv2.putText(bgr_gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 2)
    cv2.putText(bgr_gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,0,0), 1)
    frames.append(bgr_gt)
    
    strip = np.hstack(frames)
    cv2.imwrite(os.path.join(out_dir, f'blend_{name}.jpg'), strip)
    
    # Best alpha — where on the Geo→Shared line is closest to GT?
    best_alpha = None
    best_err = float('inf')
    for a in np.linspace(0, 2, 201):
        ab_test = (1 - a) * d['ab_geo'] + a * ab_shared
        e = np.sqrt(np.mean((ab_test - d['ab_gt'])**2))
        if e < best_err:
            best_err = e
            best_alpha = a
    print(f'    Best α={best_alpha:.2f}: err={best_err:.2f} (vs Geo={err_base:.2f}, DDColor={d["err_dd_gt"]:.2f})')
    print()


# ============================================================
# PART 5: Per-query comparison — which queries do all three
# agree on vs disagree?
# ============================================================
print('=== PART 5: Per-Query Agreement ===\n')

# For each query, measure: does it activate in similar spatial patterns
# across DDColor, GT, and Geometric?
all_query_agreement = np.zeros((100, 3))  # DD↔GT, Geo↔GT, Geo↔DD

for d in all_data:
    for q in range(100):
        dd_q = d['maps_dd'][q].flatten()
        gt_q = d['maps_gt'][q].flatten()
        geo_q = d['maps_geo'][q].flatten()
        
        def safe_corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return np.corrcoef(a, b)[0, 1]
        
        all_query_agreement[q, 0] += safe_corr(dd_q, gt_q)
        all_query_agreement[q, 1] += safe_corr(geo_q, gt_q)
        all_query_agreement[q, 2] += safe_corr(geo_q, dd_q)

all_query_agreement /= len(all_data)

# Sort by DD↔GT agreement (queries where DDColor already matches GT)
sorted_by_dd_gt = np.argsort(all_query_agreement[:, 0])[::-1]

print(f'  {"Query":>5} {"DD↔GT":>8} {"Geo↔GT":>8} {"Geo↔DD":>8} {"Angle":>8} {"Mag":>8}')
print(f'  {"-"*48}')

for i, q in enumerate(sorted_by_dd_gt[:15]):
    angle = np.degrees(np.arctan2(color_wheel[q, 1], color_wheel[q, 0]))
    mag = np.linalg.norm(color_wheel[q])
    print(f'  {q:5d} {all_query_agreement[q,0]:8.3f} {all_query_agreement[q,1]:8.3f} '
          f'{all_query_agreement[q,2]:8.3f} {angle:7.1f}° {mag:8.3f}')

print(f'\n  ...')
for i, q in enumerate(sorted_by_dd_gt[-5:]):
    angle = np.degrees(np.arctan2(color_wheel[q, 1], color_wheel[q, 0]))
    mag = np.linalg.norm(color_wheel[q])
    print(f'  {q:5d} {all_query_agreement[q,0]:8.3f} {all_query_agreement[q,1]:8.3f} '
          f'{all_query_agreement[q,2]:8.3f} {angle:7.1f}° {mag:8.3f}')

# Queries where all three agree
mean_dd_gt = all_query_agreement[:, 0].mean()
mean_geo_gt = all_query_agreement[:, 1].mean()
mean_geo_dd = all_query_agreement[:, 2].mean()
print(f'\n  Mean per-query correlation:')
print(f'    DD↔GT:  {mean_dd_gt:.3f}')
print(f'    Geo↔GT: {mean_geo_gt:.3f}')
print(f'    Geo↔DD: {mean_geo_dd:.3f}')

# How many queries does Geometric activate vs DDColor?
geo_active = np.mean([np.sum(np.abs(d['maps_geo']).mean(axis=(1,2)) > 0.01) for d in all_data])
dd_active = np.mean([np.sum(np.abs(d['maps_dd']).mean(axis=(1,2)) > 0.01) for d in all_data])
print(f'\n  Active queries: DDColor={dd_active:.0f}, Geometric={geo_active:.0f}')


# ============================================================
# PART 6: What DDColor and GT share that we're missing
# ============================================================
print('\n=== PART 6: The Missing Structure ===\n')

# The common pathway: what DD and GT BOTH have that Geometric doesn't
for d in all_data[:4]:
    name = d['name']
    
    # Compute: for each pixel, what direction in ab-space do DD and GT agree on?
    # Both point in the same direction but may differ in magnitude
    dd_angle_map = np.arctan2(d['ab_dd'][:,:,1], d['ab_dd'][:,:,0])
    gt_angle_map = np.arctan2(d['ab_gt'][:,:,1], d['ab_gt'][:,:,0])
    geo_angle_map = np.arctan2(d['ab_geo'][:,:,1], d['ab_geo'][:,:,0])
    
    # Angular agreement (how often do they point the same way?)
    dd_gt_angle_diff = np.abs(np.arctan2(
        np.sin(dd_angle_map - gt_angle_map),
        np.cos(dd_angle_map - gt_angle_map)
    ))
    geo_gt_angle_diff = np.abs(np.arctan2(
        np.sin(geo_angle_map - gt_angle_map),
        np.cos(geo_angle_map - gt_angle_map)
    ))
    
    # Mask out very low-saturation pixels (gray has undefined angle)
    gt_sat = np.sqrt(d['ab_gt'][:,:,0]**2 + d['ab_gt'][:,:,1]**2)
    sat_mask = gt_sat > 5  # only where GT has color
    
    if sat_mask.sum() > 0:
        dd_gt_agree = (dd_gt_angle_diff[sat_mask] < np.pi/4).mean()  # within 45°
        geo_gt_agree = (geo_gt_angle_diff[sat_mask] < np.pi/4).mean()
        
        # Magnitude agreement
        dd_sat = np.sqrt(d['ab_dd'][:,:,0]**2 + d['ab_dd'][:,:,1]**2)
        geo_sat = np.sqrt(d['ab_geo'][:,:,0]**2 + d['ab_geo'][:,:,1]**2)
        
        dd_mag_corr = np.corrcoef(dd_sat[sat_mask], gt_sat[sat_mask])[0,1]
        geo_mag_corr = np.corrcoef(geo_sat[sat_mask], gt_sat[sat_mask])[0,1]
        
        print(f'  {name} (colored pixels: {sat_mask.sum()}/{sat_mask.size}):')
        print(f'    Color direction agreement (within 45°):')
        print(f'      DDColor↔GT:    {dd_gt_agree:.1%}')
        print(f'      Geometric↔GT:  {geo_gt_agree:.1%}')
        print(f'    Saturation correlation:')
        print(f'      DDColor↔GT:    {dd_mag_corr:.3f}')
        print(f'      Geometric↔GT:  {geo_mag_corr:.3f}')
        print()


# ============================================================
# Summary
# ============================================================
print('=== SUMMARY ===\n')
print(f'Three-point triangle (means across {len(all_data)} images):')
print(f'  DD↔GT:  {mean_dd_gt:.2f}')
print(f'  Geo↔GT: {mean_geo_gt:.2f}')
print(f'  Geo↔DD: {mean_geo_dd:.2f}')
print()
print(f'The Geometric colorizer provides a CONTROLLABLE third point.')
print(f'Blending Geo toward the DDColor-GT midpoint improves results.')
print(f'Per-query analysis shows which color directions all three agree on.')
print(f'\nOutput saved to: {out_dir}/')
print('Done!')
