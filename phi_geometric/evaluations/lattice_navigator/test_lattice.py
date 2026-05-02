"""
Test the Lattice Navigator: Generate synthetic color knowledge and colorize.
"""
import numpy as np
import cv2
import sys
import glob
import time

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from color_lattice import LatticeNavigator

import torch
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

def ab_to_bgr(ab, L):
    ab_u = np.clip(ab + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.stack([L, ab_u[:,:,0], ab_u[:,:,1]], axis=-1), cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()

print('=== LATTICE NAVIGATOR: Synthetic Knowledge Test ===')
print()

# Initialize
image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
nav = LatticeNavigator()
nav.initialize(image_paths)

# Navigate - generate synthetic knowledge
nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=True)
nav.report()

# Load DDColor for comparison
v16 = V16GeometricColorizer()

# Test on images
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in [50, 52, 54, 56]]

rows = []
for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    r = cv2.resize(im, (128, 128))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    # DDColor
    t = torch.from_numpy(cv2.resize(im, (256,256)).transpose(2,0,1)[:,:,::-1].copy()).float().unsqueeze(0)/255.0
    gbgr256 = cv2.cvtColor(cv2.cvtColor(cv2.resize(im, (256,256)), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gbgr256.transpose(2,0,1)).float().unsqueeze(0)/255.0
    with torch.no_grad(): ab_dd = v16.forward(t)
    ab_dd_np = ab_dd[0].permute(1,2,0).numpy()
    ab_dd_128 = cv2.resize(ab_dd_np, (128, 128))
    bgr_dd = ab_to_bgr(ab_dd_128, L)
    
    # Lattice colorization
    name = img_path.split('/')[-1]
    print(f'\nColorizing {name} with lattice...')
    t0 = time.time()
    ab_lat = nav.colorize(g)
    elapsed = time.time() - t0
    bgr_lat = ab_to_bgr(ab_lat, L)
    print(f'  Done in {elapsed:.1f}s')
    
    # Row: Gray | Lattice | DDColor | GT
    row = np.hstack([gbgr, bgr_lat, bgr_dd, r])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ('Gray', 5),
        (f'Lattice s={get_sat(bgr_lat):.0f}', 133),
        (f'DDColor s={get_sat(bgr_dd):.0f}', 261),
        (f'GT s={get_sat(r):.0f}', 389),
    ]
    for txt, xo in labels:
        cv2.putText(row, txt, (xo, 14), font, 0.35, (255,255,255), 2)
        cv2.putText(row, txt, (xo, 14), font, 0.35, (0,0,0), 1)
    
    rows.append(row)
    print(f'  {name}: Lattice sat={get_sat(bgr_lat):.0f}, DDColor sat={get_sat(bgr_dd):.0f}, GT sat={get_sat(r):.0f}')

full = np.vstack(rows)
out = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/lattice_v3.jpg'
cv2.imwrite(out, full)
print(f'\nSaved: {out}')

# Also save the lattice itself for analysis
print(f'\nLattice summary:')
print(f'  Total nodes: {len(nav.nodes)}')
print(f'  Seed nodes: {sum(1 for n in nav.nodes if n.source == "seed")}')
print(f'  Navigated nodes: {sum(1 for n in nav.nodes if n.source == "navigated")}')

# Show some example navigated nodes
print(f'\nExample navigated nodes:')
navigated = [n for n in nav.nodes if n.source == 'navigated' and n.confidence > 0.2]
navigated.sort(key=lambda n: -n.confidence)
for n in navigated[:15]:
    print(f'  gen={n.generation} ctx={n.context:30s} '
          f'a={n.color_a:+6.1f} b={n.color_b:+6.1f} '
          f'sat={n.saturation:5.1f} conf={n.confidence:.3f} '
          f'bright={n.brightness:.2f}')

print('\nDone!')
