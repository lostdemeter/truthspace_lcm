"""
Karplus-Strong v2: The Damping Model

Key insight from deep dive:
- The lattice OVERSHOOTS (too much color)
- DDColor's correction is mostly NEGATIVE (desaturation)
- Edge-bounded regions explain 93% of correction variance
- The correction is ultra-low-frequency (2-3 spatial blobs)

The inverted KS model:
1. Start with lattice output (too much energy / too saturated)
2. Segment into edge-bounded regions (each region = one "string")
3. For each region, the "resonant frequency" = the color that SURVIVES damping
4. Damping rule: within each region, color converges to the region's 
   brightness-appropriate value (dark regions damp warm, bright regions damp cool)
5. Inter-region contrast: adjacent regions should have color CONTRAST 
   proportional to their luminance contrast
6. The loop REMOVES energy (color) selectively, not adds it

This is fundamentally different from KS v1:
- v1: added energy (histogram matching, saturation boost)  
- v2: removes energy (selective damping) — lets structure emerge by subtraction

The analogy to real KS:
- The string starts with MAX energy (the pluck)
- Each cycle through the delay line DAMPS slightly
- What survives is the resonant frequency
- Here: the lattice is the pluck, damping reveals what "resonates"

Also tracking per-iteration diagnostics exhaustively to observe
what the hyperstructure does as it evolves.
"""
import numpy as np
import cv2
import sys
import glob
import os
import time
from scipy import ndimage

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from color_lattice import LatticeNavigator
import torch
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()


def segment_by_edges(gray, edge_threshold_pct=65, min_region_size=15):
    """Segment image into edge-bounded regions. Each region = one KS delay line."""
    # Compute edges
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sx**2 + sy**2)
    
    # Threshold to binary edges
    thresh = np.percentile(edges, edge_threshold_pct)
    edge_binary = (edges > thresh).astype(np.uint8)
    
    # Distance transform to find region centers
    dist = cv2.distanceTransform(1 - edge_binary, cv2.DIST_L2, 5)
    
    # Label connected components of non-edge regions
    region_mask = (dist > 2).astype(np.uint8)
    labeled, n_regions = ndimage.label(region_mask)
    
    # Expand labels to fill edge pixels (nearest-region assignment)
    # This ensures every pixel belongs to a region
    from scipy.ndimage import distance_transform_edt
    unlabeled = labeled == 0
    if unlabeled.any():
        # For each unlabeled pixel, assign to nearest labeled pixel's region
        _, nearest_idx = distance_transform_edt(unlabeled, return_indices=True)
        labeled[unlabeled] = labeled[nearest_idx[0][unlabeled], nearest_idx[1][unlabeled]]
    
    # Merge small regions into neighbors
    for rid in range(1, n_regions + 1):
        mask = labeled == rid
        if mask.sum() < min_region_size:
            # Find neighboring region with most border pixels
            dilated = ndimage.binary_dilation(mask, iterations=2)
            border = dilated & ~mask
            neighbor_labels = labeled[border]
            neighbor_labels = neighbor_labels[neighbor_labels != rid]
            neighbor_labels = neighbor_labels[neighbor_labels != 0]
            if len(neighbor_labels) > 0:
                best_neighbor = np.bincount(neighbor_labels).argmax()
                labeled[mask] = best_neighbor
    
    return labeled, edges


def compute_region_stats(labeled, gray, ab):
    """Compute per-region statistics."""
    regions = {}
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        size = mask.sum()
        if size < 5: continue
        
        regions[rid] = {
            'mask': mask,
            'size': size,
            'mean_brightness': gray[mask].mean() / 255.0,
            'std_brightness': gray[mask].std() / 255.0,
            'mean_a': ab[:,:,0][mask].mean(),
            'mean_b': ab[:,:,1][mask].mean(),
            'std_a': ab[:,:,0][mask].std(),
            'std_b': ab[:,:,1][mask].std(),
            'sat': np.sqrt(ab[:,:,0][mask]**2 + ab[:,:,1][mask]**2).mean(),
        }
    return regions


def find_region_neighbors(labeled):
    """Find which regions are adjacent."""
    neighbors = {}
    h, w = labeled.shape
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        dilated = ndimage.binary_dilation(mask, iterations=1)
        border = dilated & ~mask
        neighbor_ids = set(labeled[border].flatten()) - {0, rid}
        neighbors[rid] = neighbor_ids
    return neighbors


def damping_iteration(ab, labeled, regions, gray, neighbors, 
                      intra_damp=0.3, inter_contrast_strength=0.15,
                      brightness_coupling=0.2):
    """
    One iteration of the damping KS loop.
    
    Three forces:
    1. INTRA-REGION DAMPING: within each region, push toward regional median
       (this is the "averaging" in KS — enforces consensus within the delay line)
    2. INTER-REGION CONTRAST: adjacent regions should differ proportionally
       to their luminance contrast (this creates color boundaries at edges)
    3. BRIGHTNESS-COLOR COUPLING: dark regions damp warm colors more,
       bright regions damp cool colors more (physical light rule)
    """
    result = ab.copy()
    h, w = ab.shape[:2]
    
    # 1. INTRA-REGION DAMPING
    # Each region converges to its median color (like KS delay line averaging)
    for rid, stats in regions.items():
        mask = stats['mask']
        for ch in range(2):
            region_vals = result[:,:,ch][mask]
            median_val = np.median(region_vals)
            # Damp toward median — this REDUCES variance within region
            result[:,:,ch][mask] = (1 - intra_damp) * region_vals + intra_damp * median_val
    
    # 2. INTER-REGION CONTRAST
    # Adjacent regions should have color difference proportional to brightness difference
    for rid, stats in regions.items():
        mask = stats['mask']
        bright_r = stats['mean_brightness']
        
        for nid in neighbors.get(rid, set()):
            if nid not in regions: continue
            n_stats = regions[nid]
            bright_n = n_stats['mean_brightness']
            
            # Luminance contrast between regions
            lum_contrast = abs(bright_r - bright_n)
            
            # If high luminance contrast, push colors APART
            # If low luminance contrast, push colors TOGETHER
            for ch in range(2):
                color_diff = stats[f'mean_{"a" if ch==0 else "b"}'] - n_stats[f'mean_{"a" if ch==0 else "b"}']
                
                if lum_contrast > 0.1:
                    # High edge contrast: maintain or amplify color difference
                    # (don't change — let the existing difference persist)
                    pass
                else:
                    # Low edge contrast: blend colors (these regions are likely same object)
                    target = (stats[f'mean_{"a" if ch==0 else "b"}'] + 
                             n_stats[f'mean_{"a" if ch==0 else "b"}']) / 2
                    result[:,:,ch][mask] += inter_contrast_strength * (target - result[:,:,ch][mask])
    
    # 3. BRIGHTNESS-COLOR COUPLING (physical damping rule)
    # Dark pixels: damp positive b (warm) more, preserve negative b (cool shadow)
    # Bright pixels: damp extreme a, preserve moderate values
    brightness = gray.astype(float) / 255.0
    
    # Dark regions: reduce saturation, push toward cool
    dark_mask = brightness < 0.25
    result[:,:,0][dark_mask] *= (1 - brightness_coupling)
    result[:,:,1][dark_mask] *= (1 - brightness_coupling)
    
    # Very bright regions: reduce saturation (overexposed → desaturated)
    bright_mask = brightness > 0.85
    result[:,:,0][bright_mask] *= (1 - brightness_coupling * 0.5)
    result[:,:,1][bright_mask] *= (1 - brightness_coupling * 0.5)
    
    return result


def ks_v2_colorize(gray, ab_init, n_iterations=40, verbose=True):
    """
    Full KS v2 pipeline.
    
    Start with oversaturated lattice output.
    Apply damping loop to let structure emerge by subtraction.
    Track everything for analysis.
    """
    h, w = gray.shape
    
    # Segment into edge-bounded regions
    labeled, edges = segment_by_edges(gray)
    n_regions = len(np.unique(labeled)) - (1 if 0 in labeled else 0)
    if verbose:
        print(f'  Segmented into {n_regions} edge-bounded regions')
    
    # Find region neighbors
    neighbors = find_region_neighbors(labeled)
    
    ab = ab_init.copy()
    history = [ab.copy()]
    metrics_list = []
    
    for it in range(n_iterations):
        # Compute region stats BEFORE this iteration
        regions = compute_region_stats(labeled, gray, ab)
        
        # Damping schedule: start aggressive, ease off
        # Like KS where the first few cycles have strongest damping
        progress = it / n_iterations
        intra_damp = 0.4 * (1 - progress * 0.5)  # 0.4 → 0.2
        inter_strength = 0.2 * (1 - progress * 0.3)  # 0.2 → 0.14
        bright_coupling = 0.15 * (1 - progress * 0.5)  # 0.15 → 0.075
        
        # Apply one damping iteration
        ab = damping_iteration(
            ab, labeled, regions, gray, neighbors,
            intra_damp=intra_damp,
            inter_contrast_strength=inter_strength,
            brightness_coupling=bright_coupling
        )
        
        # Light spatial smoothing (within regions, not across edges)
        for ch in range(2):
            ab[:,:,ch] = cv2.bilateralFilter(ab[:,:,ch].astype(np.float32), 5, 20, 20)
        
        # Metrics
        sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2).mean()
        n_hues = len(set())  # placeholder
        
        metrics = {
            'iter': it,
            'mean_sat': sat,
            'std_a': ab[:,:,0].std(),
            'std_b': ab[:,:,1].std(),
            'max_sat': np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2).max(),
            'mean_abs_a': np.abs(ab[:,:,0]).mean(),
            'mean_abs_b': np.abs(ab[:,:,1]).mean(),
        }
        metrics_list.append(metrics)
        
        history.append(ab.copy())
        
        if verbose and it % 5 == 0:
            print(f'  Iter {it:2d}: sat={sat:.1f}, |a|={metrics["mean_abs_a"]:.1f}, '
                  f'|b|={metrics["mean_abs_b"]:.1f}, max_sat={metrics["max_sat"]:.1f}')
    
    # Final bilateral smoothing
    for ch in range(2):
        ab[:,:,ch] = cv2.bilateralFilter(ab[:,:,ch].astype(np.float32), 7, 30, 30)
    
    return ab, history, metrics_list, labeled


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('=== KARPLUS-STRONG v2: THE DAMPING MODEL ===')
    print()
    
    # Initialize
    image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
    nav = LatticeNavigator()
    nav.initialize(image_paths)
    nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=False)
    v16 = V16GeometricColorizer()
    
    SZ = 128
    all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
    test_indices = [50, 52, 54, 56]
    test_paths = [all_imgs[i] for i in test_indices]
    
    out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/ks_v2'
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    for img_path in test_paths:
        im = cv2.imread(img_path)
        if im is None: continue
        name = os.path.basename(img_path).replace('.jpg', '')
        
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        L = lab[:,:,0]
        
        # DDColor reference
        gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t_in = torch.from_numpy(cv2.resize(gbgr, (256,256)).transpose(2,0,1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            ab_dd = v16.forward(t_in)
        ab_ddcolor = cv2.resize(ab_dd[0].permute(1,2,0).numpy(), (SZ, SZ))
        
        # Lattice (our starting point — the "pluck")
        ab_lattice = nav.colorize(gray)
        
        print(f'\n--- {name} ---')
        print(f'Lattice sat={get_sat(ab_to_bgr(ab_lattice, L)):.0f}, '
              f'DDColor sat={get_sat(ab_to_bgr(ab_ddcolor, L)):.0f}, '
              f'GT sat={get_sat(r):.0f}')
        
        # KS v2: Damping from lattice
        ab_ks2, history, metrics, labeled = ks_v2_colorize(
            gray, ab_lattice, n_iterations=40, verbose=True
        )
        
        # Error tracking
        err_lattice = np.sqrt(np.mean((ab_lattice - ab_ddcolor)**2))
        err_ks2 = np.sqrt(np.mean((ab_ks2 - ab_ddcolor)**2))
        
        # Track error at each iteration
        print(f'\n  Error trajectory:')
        err_trajectory = []
        for it_idx, ab_it in enumerate(history):
            err_it = np.sqrt(np.mean((ab_it - ab_ddcolor)**2))
            err_trajectory.append(err_it)
            if it_idx % 5 == 0 or it_idx == len(history) - 1:
                direction = '↓' if it_idx > 0 and err_it < err_trajectory[max(0,it_idx-1)] else '↑'
                print(f'    iter {it_idx:2d}: err={err_it:.2f} {direction}')
        
        best_iter = np.argmin(err_trajectory)
        best_err = err_trajectory[best_iter]
        
        print(f'\n  Summary:')
        print(f'    Lattice → DDColor: {err_lattice:.2f}')
        print(f'    KS v2 final → DDColor: {err_ks2:.2f} ({"better" if err_ks2 < err_lattice else "worse"})')
        print(f'    Best iteration: {best_iter} with err={best_err:.2f}')
        print(f'    Best vs lattice: {1-best_err/err_lattice:.0%} gap closed')
        
        all_results.append({
            'name': name,
            'err_lattice': err_lattice,
            'err_ks2': err_ks2,
            'best_iter': best_iter,
            'best_err': best_err,
            'err_trajectory': err_trajectory,
        })
        
        # Save comparison strip: Gray | Lattice | KS v2 (best iter) | KS v2 (final) | DDColor | GT
        ab_best = history[best_iter]
        
        imgs = [
            (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
            (ab_to_bgr(ab_lattice, L), f'Lattice s={get_sat(ab_to_bgr(ab_lattice, L)):.0f}'),
            (ab_to_bgr(ab_best, L), f'KS-best(i={best_iter}) s={get_sat(ab_to_bgr(ab_best, L)):.0f}'),
            (ab_to_bgr(ab_ks2, L), f'KS-final s={get_sat(ab_to_bgr(ab_ks2, L)):.0f}'),
            (ab_to_bgr(ab_ddcolor, L), f'DDColor s={get_sat(ab_to_bgr(ab_ddcolor, L)):.0f}'),
            (r, f'GT s={get_sat(r):.0f}'),
        ]
        
        for img, label in imgs:
            cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 2)
            cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,0), 1)
        
        strip = np.hstack([img for img, _ in imgs])
        cv2.imwrite(os.path.join(out_dir, f'comparison_{name}.jpg'), strip)
        
        # Convergence video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(out_dir, f'convergence_{name}.mp4')
        frame_h = SZ + 35
        frame_w = SZ * 5
        writer = cv2.VideoWriter(video_path, fourcc, 8, (frame_w, frame_h))
        
        for it_idx, ab_it in enumerate(history):
            bgr_it = ab_to_bgr(ab_it, L)
            err_it = err_trajectory[it_idx]
            
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            frame[0:SZ, 0:SZ] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame[0:SZ, SZ:SZ*2] = ab_to_bgr(ab_lattice, L)
            frame[0:SZ, SZ*2:SZ*3] = bgr_it
            frame[0:SZ, SZ*3:SZ*4] = ab_to_bgr(ab_ddcolor, L)
            frame[0:SZ, SZ*4:SZ*5] = r
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            sat_it = get_sat(bgr_it)
            labels = [
                ('Gray', 5),
                ('Lattice', SZ + 5),
                (f'KS i={it_idx} e={err_it:.1f} s={sat_it:.0f}', SZ*2 + 5),
                ('DDColor', SZ*3 + 5),
                ('GT', SZ*4 + 5),
            ]
            for txt, xo in labels:
                cv2.putText(frame, txt, (xo, SZ + 15), font, 0.33, (255,255,255), 1)
            
            # Error bar
            bar_y = SZ + 22
            bar_w = frame_w - 20
            cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + 5), (50,50,50), -1)
            # Show error relative to lattice error
            err_pct = min(1.0, err_it / (err_lattice * 1.5))
            color = (0, int(200*(1-err_pct)), int(200*err_pct))
            cv2.rectangle(frame, (10, bar_y), (10 + int(bar_w * err_pct), bar_y + 5), color, -1)
            
            writer.write(frame)
        
        writer.release()
        
        # Region visualization: paint each region with a distinct color to show segmentation
        region_vis = np.zeros((SZ, SZ, 3), dtype=np.uint8)
        rng = np.random.RandomState(42)
        for rid in np.unique(labeled):
            if rid == 0: continue
            color = rng.randint(50, 255, 3).tolist()
            region_vis[labeled == rid] = color
        cv2.imwrite(os.path.join(out_dir, f'regions_{name}.jpg'), region_vis)
    
    # ================================================================
    # OVERALL SUMMARY
    # ================================================================
    print('\n\n' + '='*60)
    print('KS v2 OVERALL RESULTS')
    print('='*60)
    
    for res in all_results:
        improvement = 1 - res['best_err'] / res['err_lattice']
        converges = res['err_ks2'] < res['err_lattice']
        print(f"\n  {res['name']}:")
        print(f"    Lattice err: {res['err_lattice']:.2f}")
        print(f"    KS v2 best:  {res['best_err']:.2f} at iter {res['best_iter']} ({improvement:+.0%})")
        print(f"    KS v2 final: {res['err_ks2']:.2f} ({'converges ✓' if converges else 'diverges ✗'})")
        
        # Show error trajectory shape
        traj = res['err_trajectory']
        if len(traj) > 5:
            early_trend = 'decreasing' if traj[5] < traj[0] else 'increasing'
            late_trend = 'decreasing' if traj[-1] < traj[len(traj)//2] else 'increasing'
            print(f"    Trajectory: early={early_trend}, late={late_trend}")
    
    print(f'\nResults saved to: {out_dir}/')
    print('Done!')
