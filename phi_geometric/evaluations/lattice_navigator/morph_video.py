"""
Morph Video: Animate the transition from Lattice colorization to DDColor.

For each test image:
1. Get lattice output (ab channels)
2. Get DDColor output (ab channels)  
3. Interpolate: ab(t) = (1-t) * lattice + t * ddcolor for t in [0..1]
4. Save frames and compile to video

The frames where the biggest visual jumps happen reveal
exactly WHAT knowledge DDColor has that the lattice doesn't.
"""
import numpy as np
import cv2
import sys
import glob
import time
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

def make_error_heatmap(ab1, ab2, scale=3.0):
    """Pixel-wise color error as a heatmap."""
    diff = np.sqrt(np.sum((ab1 - ab2)**2, axis=-1))
    norm = np.clip(diff / (diff.max() + 1e-8), 0, 1)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

print('=== MORPH VIDEO: Lattice → DDColor ===')
print()

# Initialize lattice
image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
nav = LatticeNavigator()
nav.initialize(image_paths)
nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=True)

# Load DDColor
v16 = V16GeometricColorizer()

# Test images
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in [50, 52, 54, 56]]

SZ = 256  # Output size for video
N_FRAMES = 60  # frames per image
FPS = 15

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/morph_frames'
os.makedirs(out_dir, exist_ok=True)

# Process each image
for img_idx, img_path in enumerate(test_paths):
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    # Prepare grayscale
    r = cv2.resize(im, (SZ, SZ))
    gt_bgr = r.copy()
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    # DDColor output
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t_in = torch.from_numpy(gbgr.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        ab_dd = v16.forward(t_in)
    ab_ddcolor = ab_dd[0].permute(1,2,0).numpy()  # [256, 256, 2]
    
    # Lattice output (at 128, upscale to 256)
    gray128 = cv2.resize(gray, (128, 128))
    print(f'\nColorizing {name} with lattice...')
    ab_lat_128 = nav.colorize(gray128)
    ab_lattice = cv2.resize(ab_lat_128.astype(np.float32), (SZ, SZ))  # [256, 256, 2]
    
    # Error heatmap
    error_map = make_error_heatmap(ab_lattice, ab_ddcolor)
    
    # Compute per-pixel error magnitude for analysis
    pixel_error = np.sqrt(np.sum((ab_lattice - ab_ddcolor)**2, axis=-1))
    mean_err = pixel_error.mean()
    max_err = pixel_error.max()
    
    print(f'  {name}: mean_error={mean_err:.1f}, max_error={max_err:.1f}')
    
    # Generate morph frames
    print(f'  Generating {N_FRAMES} morph frames...')
    
    # Use phi-spaced t values for non-linear morph (more frames near 0 and 1)
    t_values = np.linspace(0, 1, N_FRAMES)
    
    video_path = os.path.join(out_dir, f'morph_{name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Layout: Gray | Morph | DDColor | GT | Error
    frame_w = SZ * 4
    frame_h = SZ + 40  # room for labels
    
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame_w, frame_h))
    
    for fi, t in enumerate(t_values):
        # Interpolate ab channels
        ab_morph = (1.0 - t) * ab_lattice + t * ab_ddcolor
        
        # Convert to BGR
        bgr_morph = ab_to_bgr(ab_morph, L)
        bgr_lattice = ab_to_bgr(ab_lattice, L)
        bgr_ddcolor = ab_to_bgr(ab_ddcolor, L)
        
        # Build frame
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        # Images row
        frame[0:SZ, 0:SZ] = bgr_lattice
        frame[0:SZ, SZ:SZ*2] = bgr_morph
        frame[0:SZ, SZ*2:SZ*3] = bgr_ddcolor
        frame[0:SZ, SZ*3:SZ*4] = gt_bgr
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_label = SZ + 25
        
        sat_m = get_sat(bgr_morph)
        pct = int(t * 100)
        
        labels = [
            (f'Lattice (0%)', 10),
            (f'Morph t={pct}% sat={sat_m:.0f}', SZ + 10),
            (f'DDColor (100%)', SZ*2 + 10),
            (f'Ground Truth', SZ*3 + 10),
        ]
        for txt, xo in labels:
            cv2.putText(frame, txt, (xo, y_label), font, 0.5, (255,255,255), 1)
        
        # Progress bar
        bar_y = SZ + 32
        bar_w = frame_w - 20
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + 4), (50,50,50), -1)
        cv2.rectangle(frame, (10, bar_y), (10 + int(bar_w * t), bar_y + 4), (0,200,255), -1)
        
        writer.write(frame)
    
    writer.release()
    print(f'  Saved: {video_path}')
    
    # Also save key frames as stills
    for t_key in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ab_key = (1.0 - t_key) * ab_lattice + t_key * ab_ddcolor
        bgr_key = ab_to_bgr(ab_key, L)
        still_path = os.path.join(out_dir, f'{name}_t{int(t_key*100):03d}.jpg')
        cv2.imwrite(still_path, bgr_key)

# Also make a combined comparison strip for each image
print('\nGenerating comparison strips...')
for img_idx, img_path in enumerate(test_paths):
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    stills = []
    for t_key in [0, 25, 50, 75, 100]:
        still_path = os.path.join(out_dir, f'{name}_t{t_key:03d}.jpg')
        s = cv2.imread(still_path)
        if s is not None:
            # Add label
            cv2.putText(s, f't={t_key}%', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(s, f't={t_key}%', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            stills.append(s)
    
    if stills:
        strip = np.hstack(stills)
        strip_path = os.path.join(out_dir, f'strip_{name}.jpg')
        cv2.imwrite(strip_path, strip)
        print(f'  Strip: {strip_path}')

print('\nDone! Videos and strips saved to:')
print(f'  {out_dir}/')
