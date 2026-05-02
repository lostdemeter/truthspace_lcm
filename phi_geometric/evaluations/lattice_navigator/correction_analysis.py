"""
Correction Field Analysis: The Shape of Knowledge

The DIFFERENCE between our lattice output and DDColor's output IS the knowledge.
This script:
1. Computes the correction field: C = DDColor_ab - Lattice_ab
2. SVDs the correction field to find its rank (how many basis corrections?)
3. Reconstructs with increasing rank to see when it "clicks"
4. Extrapolates BEYOND 100% (t=1.5, 2.0) to reveal the direction of knowledge
5. Generates video showing rank-1, rank-2, ... rank-N reconstructions

Key question: Is the correction field low-rank like the bulge (10 dims = 87.5%)?
If yes, the "knowledge gap" is a small geometric shape, not 55M parameters.
"""
import numpy as np
import cv2
import sys
import glob
import os

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

print('=== CORRECTION FIELD ANALYSIS: The Shape of Knowledge ===')
print()

# Initialize
image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
nav = LatticeNavigator()
nav.initialize(image_paths)
nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=False)
v16 = V16GeometricColorizer()

# Use more test images for robust SVD
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_indices = [50, 52, 54, 56, 58, 60, 62, 64]
test_paths = [all_imgs[i] for i in test_indices if i < len(all_imgs)]

SZ = 128  # Work at 128 for speed
out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/correction_analysis'
os.makedirs(out_dir, exist_ok=True)

# Collect correction fields across images
all_corrections = []  # Each is [SZ*SZ, 2] = flattened ab correction
all_lattice = []
all_ddcolor = []
all_L = []
all_names = []

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    # DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t_in = torch.from_numpy(cv2.resize(gbgr, (256,256)).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        ab_dd = v16.forward(t_in)
    ab_ddcolor = cv2.resize(ab_dd[0].permute(1,2,0).numpy(), (SZ, SZ))
    
    # Lattice
    ab_lattice = nav.colorize(gray)
    
    # Correction field
    correction = ab_ddcolor - ab_lattice  # [SZ, SZ, 2]
    
    all_corrections.append(correction.reshape(-1, 2))
    all_lattice.append(ab_lattice)
    all_ddcolor.append(ab_ddcolor)
    all_L.append(L)
    all_names.append(name)
    
    print(f'  {name}: correction mean={np.abs(correction).mean():.1f}, '
          f'max={np.abs(correction).max():.1f}')

print(f'\nCollected {len(all_corrections)} correction fields')

# ================================================================
# PART 1: SVD of the correction field
# ================================================================
print('\n=== PART 1: SVD of Correction Field ===')

# Stack all corrections: [N_images * SZ*SZ, 2]
C_all = np.vstack(all_corrections)  # [total_pixels, 2]
print(f'Correction matrix shape: {C_all.shape}')
print(f'Correction stats: mean_a={C_all[:,0].mean():.2f}, mean_b={C_all[:,1].mean():.2f}')
print(f'  std_a={C_all[:,0].std():.2f}, std_b={C_all[:,1].std():.2f}')

# SVD of the 2D correction (this is trivial since it's 2D)
# But let's think bigger: what if we include SPATIAL structure?

# For each image, the correction is [SZ, SZ, 2]. 
# Reshape to [SZ*SZ, 2] and stack across images: [N_img, SZ*SZ*2]
# SVD of THIS matrix tells us how many "correction patterns" exist

C_matrix = np.array([c.flatten() for c in all_corrections])  # [N_img, SZ*SZ*2]
print(f'\nCorrection pattern matrix: {C_matrix.shape}')

U, S, Vt = np.linalg.svd(C_matrix, full_matrices=False)
total_var = np.sum(S**2)
cumvar = np.cumsum(S**2) / total_var

print(f'\nSingular values: {S}')
print(f'Cumulative variance explained:')
for i, (s, cv) in enumerate(zip(S, cumvar)):
    phi_ratio = S[0] / s if s > 0 else float('inf')
    print(f'  Rank {i+1}: S={s:.2f}, cumvar={cv*100:.1f}%, S[0]/S[{i+1}]={phi_ratio:.3f}')

# Check for phi patterns in singular value ratios
print(f'\nPhi pattern check:')
for i in range(len(S)-1):
    if S[i+1] > 0:
        ratio = S[i] / S[i+1]
        phi_err = abs(ratio - PHI) / PHI
        print(f'  S[{i}]/S[{i+1}] = {ratio:.4f} (phi={PHI:.4f}, err={phi_err*100:.1f}%)')

# ================================================================
# PART 2: Per-image correction SVD (spatial structure)
# ================================================================
print('\n=== PART 2: Per-Image Correction Spatial Structure ===')

for idx in range(min(4, len(all_corrections))):
    name = all_names[idx]
    correction = all_corrections[idx].reshape(SZ, SZ, 2)
    
    # Treat correction as 2-channel image, SVD each channel
    for ch, ch_name in enumerate(['a', 'b']):
        C_ch = correction[:, :, ch]  # [SZ, SZ]
        U_ch, S_ch, Vt_ch = np.linalg.svd(C_ch, full_matrices=False)
        total = np.sum(S_ch**2)
        cumvar_ch = np.cumsum(S_ch**2) / total
        
        rank_90 = np.searchsorted(cumvar_ch, 0.9) + 1
        rank_95 = np.searchsorted(cumvar_ch, 0.95) + 1
        rank_99 = np.searchsorted(cumvar_ch, 0.99) + 1
        
        print(f'  {name} channel {ch_name}: '
              f'rank@90%={rank_90}, rank@95%={rank_95}, rank@99%={rank_99} '
              f'(out of {SZ})')
    
    # Reconstruct with increasing rank and save frames
    correction_a = correction[:, :, 0]
    correction_b = correction[:, :, 1]
    
    U_a, S_a, Vt_a = np.linalg.svd(correction_a, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(correction_b, full_matrices=False)
    
    L = all_L[idx]
    ab_lat = all_lattice[idx]
    ab_dd = all_ddcolor[idx]
    
    rank_frames = []
    for rank in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, SZ]:
        if rank == 0:
            ab_recon = ab_lat.copy()
        else:
            r_a = U_a[:, :rank] @ np.diag(S_a[:rank]) @ Vt_a[:rank, :]
            r_b = U_b[:, :rank] @ np.diag(S_b[:rank]) @ Vt_b[:rank, :]
            ab_recon = ab_lat + np.stack([r_a, r_b], axis=-1)
        
        bgr = ab_to_bgr(ab_recon, L)
        
        # Label
        if rank == 0:
            label = 'Lattice'
        elif rank == SZ:
            label = f'Full (r={SZ})'
        else:
            var_a = np.sum(S_a[:rank]**2) / np.sum(S_a**2) * 100
            var_b = np.sum(S_b[:rank]**2) / np.sum(S_b**2) * 100
            label = f'r={rank} ({(var_a+var_b)/2:.0f}%)'
        
        cv2.putText(bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
        cv2.putText(bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        rank_frames.append(bgr)
    
    # Add GT
    gt = cv2.resize(cv2.imread(test_paths[idx]), (SZ, SZ))
    cv2.putText(gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
    cv2.putText(gt, 'GT', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    rank_frames.append(gt)
    
    # Make strip (2 rows of frames)
    n = len(rank_frames)
    half = (n + 1) // 2
    row1 = np.hstack(rank_frames[:half])
    row2_frames = rank_frames[half:]
    while len(row2_frames) < half:
        row2_frames.append(np.zeros_like(rank_frames[0]))
    row2 = np.hstack(row2_frames)
    strip = np.vstack([row1, row2])
    
    strip_path = os.path.join(out_dir, f'rank_reconstruction_{name}.jpg')
    cv2.imwrite(strip_path, strip)
    print(f'  Saved: {strip_path}')

# ================================================================
# PART 3: Extrapolation beyond 100%
# ================================================================
print('\n=== PART 3: Extrapolation Beyond 100% ===')

for idx in range(min(4, len(all_corrections))):
    name = all_names[idx]
    L = all_L[idx]
    ab_lat = all_lattice[idx]
    ab_dd = all_ddcolor[idx]
    correction = ab_dd - ab_lat
    
    extrap_frames = []
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    for t in t_values:
        ab_t = ab_lat + t * correction
        bgr = ab_to_bgr(ab_t, L)
        sat = get_sat(bgr)
        
        label = f't={t:.2f} s={sat:.0f}'
        cv2.putText(bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
        cv2.putText(bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        extrap_frames.append(bgr)
    
    strip = np.hstack(extrap_frames)
    strip_path = os.path.join(out_dir, f'extrapolation_{name}.jpg')
    cv2.imwrite(strip_path, strip)
    print(f'  {name}: saved extrapolation strip')
    
    # Print t vs saturation
    print(f'    t=0.0→{get_sat(ab_to_bgr(ab_lat, L)):.0f}, '
          f't=1.0→{get_sat(ab_to_bgr(ab_dd, L)):.0f}, '
          f't=1.5→{get_sat(ab_to_bgr(ab_lat + 1.5*correction, L)):.0f}, '
          f't=2.0→{get_sat(ab_to_bgr(ab_lat + 2.0*correction, L)):.0f}')

# ================================================================
# PART 4: Make extrapolation video
# ================================================================
print('\n=== PART 4: Extrapolation Video (0% → 200%) ===')

N_FRAMES = 120
FPS = 20

for idx in range(min(4, len(all_corrections))):
    name = all_names[idx]
    L = all_L[idx]
    ab_lat = all_lattice[idx]
    ab_dd = all_ddcolor[idx]
    correction = ab_dd - ab_lat
    gt = cv2.resize(cv2.imread(test_paths[idx]), (SZ, SZ))
    
    video_path = os.path.join(out_dir, f'extrap_video_{name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_h = SZ + 35
    frame_w = SZ * 4
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame_w, frame_h))
    
    t_values = np.linspace(0, 2.0, N_FRAMES)
    
    for t in t_values:
        ab_t = ab_lat + t * correction
        bgr_t = ab_to_bgr(ab_t, L)
        bgr_lat = ab_to_bgr(ab_lat, L)
        bgr_dd = ab_to_bgr(ab_dd, L)
        
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame[0:SZ, 0:SZ] = bgr_lat
        frame[0:SZ, SZ:SZ*2] = bgr_t
        frame[0:SZ, SZ*2:SZ*3] = bgr_dd
        frame[0:SZ, SZ*3:SZ*4] = gt
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        pct = int(t * 100)
        sat = get_sat(bgr_t)
        labels = [
            ('Lattice (0%)', 5),
            (f't={pct}% sat={sat:.0f}', SZ + 5),
            ('DDColor (100%)', SZ*2 + 5),
            ('Ground Truth', SZ*3 + 5),
        ]
        for txt, xo in labels:
            cv2.putText(frame, txt, (xo, SZ + 18), font, 0.4, (255,255,255), 1)
        
        # Progress bar (marks at 0%, 100%, 200%)
        bar_y = SZ + 24
        bar_w = frame_w - 20
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + 5), (50,50,50), -1)
        pos = int(bar_w * t / 2.0)
        # Color: green before 100%, red after
        color = (0, 200, 0) if t <= 1.0 else (0, 0, 200)
        cv2.rectangle(frame, (10, bar_y), (10 + pos, bar_y + 5), color, -1)
        # Mark 100%
        mark_100 = int(bar_w * 0.5)
        cv2.line(frame, (10 + mark_100, bar_y - 2), (10 + mark_100, bar_y + 7), (255,255,255), 1)
        
        writer.write(frame)
    
    writer.release()
    print(f'  Saved: {video_path}')

# ================================================================
# PART 5: Summary statistics
# ================================================================
print('\n=== SUMMARY ===')
print(f'Across {len(all_corrections)} images:')

# Average rank needed for 90% reconstruction
avg_rank_90 = []
for idx in range(len(all_corrections)):
    correction = all_corrections[idx].reshape(SZ, SZ, 2)
    for ch in range(2):
        _, S_ch, _ = np.linalg.svd(correction[:,:,ch], full_matrices=False)
        cumvar = np.cumsum(S_ch**2) / np.sum(S_ch**2)
        avg_rank_90.append(np.searchsorted(cumvar, 0.9) + 1)

print(f'Average rank for 90% correction: {np.mean(avg_rank_90):.1f} (out of {SZ})')
print(f'  → The correction IS {"LOW-RANK" if np.mean(avg_rank_90) < SZ/4 else "HIGH-RANK"}')
print(f'  → {np.mean(avg_rank_90)/SZ*100:.0f}% of dimensions needed for 90% of correction')

# This is the key number: if it's low (like 10-20 out of 128), then
# the "knowledge" is a small geometric shape, not 55M parameters
print(f'\nImplication: The knowledge gap between lattice navigation and DDColor')
print(f'requires ~{int(np.mean(avg_rank_90))} spatial frequencies to express.')
print(f'This is the RANK of the knowledge shape.')

print('\nDone!')
