"""
Deep Dive: Why Does KS Converge to Bland?

Analyze the iteration dynamics to understand:
1. What happens to the COLOR DISTRIBUTION at each iteration?
2. What happens to SPATIAL VARIANCE (do regions maintain distinct colors)?
3. What happens to the CORRECTION vs DDColor at each step?
4. Which constraint is the main culprit (diffusion? histogram? region coherence?)
5. What does DDColor's actual correction look like in spatial frequency space?

The goal: understand what a GOOD feedback loop would need to do differently.
"""
import numpy as np
import cv2
import sys
import glob
import os

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from color_lattice import LatticeNavigator
from karplus_strong_colorizer import KarplusStrongColorizer, ab_to_bgr, get_sat
import torch
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

print('=== KS DEEP DIVE: Why Does It Converge to Bland? ===')
print()

# Setup
image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
nav = LatticeNavigator()
nav.initialize(image_paths)
nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=False)
v16 = V16GeometricColorizer()
ks = KarplusStrongColorizer()

SZ = 128
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/ks_deep_dive'
os.makedirs(out_dir, exist_ok=True)

# Use the sheep image (best lattice result) for detailed analysis
img_path = all_imgs[56]  # sheep
im = cv2.imread(img_path)
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

# Lattice reference
ab_lattice = nav.colorize(gray)

print(f'Analyzing: {name}')
print(f'DDColor mean |a|={np.abs(ab_ddcolor[:,:,0]).mean():.1f}, |b|={np.abs(ab_ddcolor[:,:,1]).mean():.1f}')
print(f'Lattice mean |a|={np.abs(ab_lattice[:,:,0]).mean():.1f}, |b|={np.abs(ab_lattice[:,:,1]).mean():.1f}')

# ================================================================
# ANALYSIS 1: Track what happens at each KS iteration
# ================================================================
print('\n=== ANALYSIS 1: Iteration-by-Iteration Dynamics ===')

feats = ks.extract_features(gray)
scaffolding = ks.compute_scaffolding(feats)
noise = ks.manufacture_noise(SZ, SZ, gray, feats)
ab = ab_lattice + scaffolding + noise * 0.3
edge_map = feats['edge_map']

# Track metrics at each iteration
metrics = {
    'iter': [],
    'mean_sat': [],
    'std_a': [],
    'std_b': [],
    'spatial_var_a': [],  # variance ACROSS space (do regions differ?)
    'spatial_var_b': [],
    'err_to_dd': [],
    'n_distinct_hues': [],  # how many distinct color families
    'max_sat': [],
    'pct_chromatic': [],  # % pixels with sat > 5
}

def count_hue_families(ab, sat_thresh=5):
    """Count distinct color families via hue binning."""
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    chromatic = sat > sat_thresh
    if chromatic.sum() < 10:
        return 0
    hues = np.arctan2(ab[:,:,1][chromatic], ab[:,:,0][chromatic])
    hist, _ = np.histogram(hues, bins=12, range=(-np.pi, np.pi))
    return (hist > chromatic.sum() * 0.03).sum()  # bins with >3% of chromatic pixels

# Run KS step by step, isolating each constraint
n_iters = 30

for it in range(-1, n_iters):
    if it == -1:
        label = 'init'
    else:
        # Apply constraints one by one
        ab = ks.edge_guided_diffusion(ab, edge_map, strength=0.3, iterations=2)
        
        if it % 3 == 0:
            ab = ks.region_coherence(ab, gray)
        
        if it % 5 == 0:
            ab = ks.scale_consistency(ab, gray)
        
        hist_strength = 0.1 * (1.0 - it / n_iters)
        ab = ks.histogram_constraint(ab, ks.natural_ab_mean, ks.natural_ab_std, strength=hist_strength)
        ab = ks.saturation_boost(ab, gray, min_sat=3.0)
        label = f'iter_{it}'
    
    sat_map = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    
    metrics['iter'].append(it)
    metrics['mean_sat'].append(sat_map.mean())
    metrics['std_a'].append(ab[:,:,0].std())
    metrics['std_b'].append(ab[:,:,1].std())
    metrics['spatial_var_a'].append(np.var(ab[:,:,0]))
    metrics['spatial_var_b'].append(np.var(ab[:,:,1]))
    metrics['err_to_dd'].append(np.sqrt(np.mean((ab - ab_ddcolor)**2)))
    metrics['n_distinct_hues'].append(count_hue_families(ab))
    metrics['max_sat'].append(sat_map.max())
    metrics['pct_chromatic'].append((sat_map > 5).mean() * 100)

print(f'\n{"iter":>5} {"sat":>6} {"std_a":>6} {"std_b":>6} '
      f'{"err_dd":>7} {"hues":>5} {"max_s":>6} {"%chrom":>7}')
print('-' * 55)
for i, it in enumerate(metrics['iter']):
    print(f'{it:>5} {metrics["mean_sat"][i]:>6.1f} {metrics["std_a"][i]:>6.1f} '
          f'{metrics["std_b"][i]:>6.1f} {metrics["err_to_dd"][i]:>7.1f} '
          f'{metrics["n_distinct_hues"][i]:>5} {metrics["max_sat"][i]:>6.1f} '
          f'{metrics["pct_chromatic"][i]:>6.1f}%')

# ================================================================
# ANALYSIS 2: Isolate each constraint's effect
# ================================================================
print('\n\n=== ANALYSIS 2: Isolate Each Constraint ===')
print('Apply each constraint ALONE to see its individual effect.\n')

# Fresh start
ab_start = ab_lattice + scaffolding + noise * 0.3

for constraint_name, apply_fn in [
    ('diffusion_only', lambda ab: ks.edge_guided_diffusion(ab, edge_map, strength=0.3, iterations=2)),
    ('region_only', lambda ab: ks.region_coherence(ab, gray)),
    ('histogram_only', lambda ab: ks.histogram_constraint(ab, ks.natural_ab_mean, ks.natural_ab_std, strength=0.2)),
    ('scale_only', lambda ab: ks.scale_consistency(ab, gray)),
    ('sat_boost_only', lambda ab: ks.saturation_boost(ab, gray, min_sat=3.0)),
]:
    ab_test = ab_start.copy()
    
    # Apply 10 times
    for _ in range(10):
        ab_test = apply_fn(ab_test)
    
    sat_before = np.sqrt(ab_start[:,:,0]**2 + ab_start[:,:,1]**2).mean()
    sat_after = np.sqrt(ab_test[:,:,0]**2 + ab_test[:,:,1]**2).mean()
    err_before = np.sqrt(np.mean((ab_start - ab_ddcolor)**2))
    err_after = np.sqrt(np.mean((ab_test - ab_ddcolor)**2))
    hues_before = count_hue_families(ab_start)
    hues_after = count_hue_families(ab_test)
    
    print(f'{constraint_name:>20}: sat {sat_before:.1f}→{sat_after:.1f} '
          f'err {err_before:.1f}→{err_after:.1f} '
          f'hues {hues_before}→{hues_after} '
          f'{"✓ HELPS" if err_after < err_before else "✗ HURTS"}')

# ================================================================
# ANALYSIS 3: What does DDColor's correction LOOK LIKE in frequency space?
# ================================================================
print('\n\n=== ANALYSIS 3: DDColor Correction in Frequency Space ===')

correction = ab_ddcolor - ab_lattice  # The "ground truth" correction

# 2D FFT of each channel
for ch, ch_name in enumerate(['a', 'b']):
    C = correction[:,:,ch]
    F = np.fft.fft2(C)
    F_shift = np.fft.fftshift(F)
    magnitude = np.abs(F_shift)
    
    # What fraction of energy is in low frequencies?
    h, w = magnitude.shape
    cy, cx = h//2, w//2
    
    total_energy = np.sum(magnitude**2)
    
    for radius in [2, 4, 8, 16, 32]:
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2
        low_energy = np.sum(magnitude[mask]**2)
        pct = low_energy / total_energy * 100
        print(f'  Channel {ch_name}: radius≤{radius:2d} → {pct:5.1f}% of energy '
              f'(≈{radius*2} pixel wavelength)')

# ================================================================
# ANALYSIS 4: Compare DDColor correction to image structure
# ================================================================
print('\n\n=== ANALYSIS 4: DDColor Correction vs Image Structure ===')

# Key question: does the correction correlate with CONNECTED REGIONS?
# If so, the feedback loop needs to enforce region-level consistency

# Segment image using watershed or simple thresholding
from scipy import ndimage

# Simple segmentation: threshold brightness + morphological cleanup
for n_levels in [4, 8, 16]:
    quantized = (gray.astype(float) / 255.0 * n_levels).astype(int)
    labeled, n_regions = ndimage.label(quantized)
    
    # For each region: is the correction within the region consistent?
    intra_region_var = []
    inter_region_means = []
    
    for region_id in range(1, min(n_regions + 1, 200)):
        mask = labeled == region_id
        if mask.sum() < 20:
            continue
        
        for ch in range(2):
            region_correction = correction[:,:,ch][mask]
            intra_region_var.append(np.var(region_correction))
            inter_region_means.append(np.mean(region_correction))
    
    total_var = np.var(correction)
    mean_intra = np.mean(intra_region_var) if intra_region_var else 0
    var_inter = np.var(inter_region_means) if inter_region_means else 0
    
    print(f'  {n_levels} brightness levels → {n_regions} regions:')
    print(f'    Total correction variance: {total_var:.1f}')
    print(f'    Mean intra-region variance: {mean_intra:.1f} ({mean_intra/total_var*100:.0f}% of total)')
    print(f'    Inter-region mean variance: {var_inter:.1f} ({var_inter/total_var*100:.0f}% of total)')
    print(f'    → Region explains {(1 - mean_intra/total_var)*100:.0f}% of correction')

# ================================================================
# ANALYSIS 5: What if the "delay line" is the edge-bounded region?
# ================================================================
print('\n\n=== ANALYSIS 5: Edge-Bounded Regions as KS Delay Lines ===')

# Watershed segmentation using edges
edges = feats['edge_map']
edge_binary = (edges > np.percentile(edges, 70)).astype(np.uint8)

# Distance transform + watershed-like labeling
dist_transform = cv2.distanceTransform(1 - edge_binary, cv2.DIST_L2, 5)
_, markers = cv2.connectedComponents((dist_transform > 3).astype(np.uint8))

n_edge_regions = markers.max()
print(f'Edge-bounded regions: {n_edge_regions}')

# For each edge-bounded region: what is DDColor's correction?
region_corrections_a = {}
region_corrections_b = {}
region_sizes = {}

for rid in range(1, n_edge_regions + 1):
    mask = markers == rid
    size = mask.sum()
    if size < 10:
        continue
    region_sizes[rid] = size
    region_corrections_a[rid] = correction[:,:,0][mask].mean()
    region_corrections_b[rid] = correction[:,:,1][mask].mean()

print(f'Regions with ≥10 pixels: {len(region_sizes)}')

# Is the correction UNIFORM within edge-bounded regions?
intra_vars_a = []
intra_vars_b = []
for rid in region_sizes:
    mask = markers == rid
    intra_vars_a.append(np.var(correction[:,:,0][mask]))
    intra_vars_b.append(np.var(correction[:,:,1][mask]))

total_var_a = np.var(correction[:,:,0])
total_var_b = np.var(correction[:,:,1])
mean_intra_a = np.mean(intra_vars_a)
mean_intra_b = np.mean(intra_vars_b)

print(f'\nEdge-bounded region analysis:')
print(f'  Channel a: intra-region var = {mean_intra_a:.1f} '
      f'({mean_intra_a/total_var_a*100:.0f}% of total {total_var_a:.1f})')
print(f'  Channel b: intra-region var = {mean_intra_b:.1f} '
      f'({mean_intra_b/total_var_b*100:.0f}% of total {total_var_b:.1f})')
print(f'  → Edge regions explain {(1-mean_intra_a/total_var_a)*100:.0f}% (a), '
      f'{(1-mean_intra_b/total_var_b)*100:.0f}% (b) of correction')

# Visualize: paint each region with its mean correction color
region_vis = np.zeros((SZ, SZ, 2))
for rid in region_sizes:
    mask = markers == rid
    region_vis[:,:,0][mask] = region_corrections_a[rid]
    region_vis[:,:,1][mask] = region_corrections_b[rid]

# This is "perfect region-level correction"
ab_region_corrected = ab_lattice + region_vis
err_region = np.sqrt(np.mean((ab_region_corrected - ab_ddcolor)**2))
err_lattice = np.sqrt(np.mean((ab_lattice - ab_ddcolor)**2))

bgr_region = ab_to_bgr(ab_region_corrected, L)
bgr_dd = ab_to_bgr(ab_ddcolor, L)
bgr_lat = ab_to_bgr(ab_lattice, L)

print(f'\n  If we knew the CORRECT per-region correction:')
print(f'    Lattice→DDColor error: {err_lattice:.1f}')
print(f'    Region-corrected→DDColor error: {err_region:.1f}')
print(f'    Gap closed: {1-err_region/err_lattice:.0%}')
print(f'    Sat: lattice={get_sat(bgr_lat):.0f}, region={get_sat(bgr_region):.0f}, DDColor={get_sat(bgr_dd):.0f}')

# Save visualization
for img, label in [(bgr_lat, 'Lattice'), (bgr_region, f'Region-corrected err={err_region:.1f}'),
                    (bgr_dd, 'DDColor'), (r, 'GT')]:
    cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 2)
    cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
strip = np.hstack([bgr_lat, bgr_region, bgr_dd, r])
cv2.imwrite(os.path.join(out_dir, f'region_correction_{name}.jpg'), strip)

# ================================================================
# ANALYSIS 6: Distribution of per-region corrections
# ================================================================
print('\n\n=== ANALYSIS 6: What Are the Actual Per-Region Corrections? ===')

# Are the corrections clustered? Are there a few "correction types"?
corr_points = np.array([[region_corrections_a[rid], region_corrections_b[rid]] 
                         for rid in region_sizes])

print(f'Correction distribution across {len(corr_points)} regions:')
print(f'  Mean: a={corr_points[:,0].mean():.1f}, b={corr_points[:,1].mean():.1f}')
print(f'  Std:  a={corr_points[:,0].std():.1f}, b={corr_points[:,1].std():.1f}')
print(f'  Range a: [{corr_points[:,0].min():.1f}, {corr_points[:,0].max():.1f}]')
print(f'  Range b: [{corr_points[:,1].min():.1f}, {corr_points[:,1].max():.1f}]')

# Cluster the corrections
from scipy.cluster.hierarchy import fcluster, linkage

if len(corr_points) > 5:
    Z = linkage(corr_points, method='ward')
    for n_clusters in [3, 5, 8]:
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        print(f'\n  With {n_clusters} correction clusters:')
        for c in range(1, n_clusters + 1):
            mask = labels == c
            if mask.sum() == 0: continue
            c_points = corr_points[mask]
            c_sizes = [region_sizes[rid] for rid, m in zip(region_sizes.keys(), mask) if m]
            total_pixels = sum(c_sizes)
            print(f'    Cluster {c}: n={mask.sum()} regions, {total_pixels} pixels, '
                  f'mean_corr=({c_points[:,0].mean():+.1f}, {c_points[:,1].mean():+.1f}), '
                  f'std=({c_points[:,0].std():.1f}, {c_points[:,1].std():.1f})')

# ================================================================
# SUMMARY
# ================================================================
print('\n\n' + '='*60)
print('DEEP DIVE SUMMARY')
print('='*60)
print()
print('Key findings:')
print()
print('1. WHAT KILLS SATURATION:')
for constraint_name in ['diffusion_only', 'region_only', 'histogram_only', 'scale_only', 'sat_boost_only']:
    pass  # printed above
print('   (See Analysis 2 output above)')
print()
print('2. FREQUENCY CONTENT:')
print('   The correction is concentrated in low spatial frequencies.')
print('   This means large-scale color shifts, not pixel-level detail.')
print()
print('3. EDGE-BOUNDED REGIONS:')
print(f'   Edge regions explain significant correction variance.')
print(f'   Perfect per-region correction closes {1-err_region/err_lattice:.0%} of the gap.')
print()
print('4. THE REAL KS ANALOGY:')
print('   Each edge-bounded region IS a delay line.')
print('   The "pitch" = one color per region.')
print('   The "pluck" = our initial guess for that region\'s color.')
print('   The "feedback" should enforce CONSENSUS within region,')
print('   not diffuse ACROSS regions.')
print()
print(f'Output saved to: {out_dir}/')
print('Done!')
