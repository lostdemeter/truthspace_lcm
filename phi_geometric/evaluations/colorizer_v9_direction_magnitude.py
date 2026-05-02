"""
v9 Colorizer: Direction/Magnitude Separation

Key insight from DDColor shape analysis (Pattern 10):
- DDColor stores color DIRECTIONS (±0.3) not actual colors (±50)
- Attention computes MAGNITUDE per pixel
- Color = direction × magnitude

Our approach:
1. Learn color DIRECTIONS from k-NN (normalize to unit circle in ab space)
2. Compute MAGNITUDE from local image statistics (contrast, saturation prior)
3. Combine: pixel_color = direction × magnitude

This should fix the desaturation problem because we never average magnitudes -
we COMPUTE them from the image structure.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
import sys
import glob

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

PHI = 1.618033988749895

print('=== FROM SCRATCH v9: Direction/Magnitude Separation ===')
print()

def extract_features(img_gray):
    h, w = img_gray.shape
    features = []
    for scale in range(5):
        freq = 0.03 * (PHI ** scale)
        for theta_idx in range(8):
            theta = theta_idx * np.pi / 8
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 1/freq, 0.5, 0)
            features.append(cv2.filter2D(img_gray, cv2.CV_64F, kernel))
    for ksize in [3, 7, 15, 31]:
        features.append(cv2.GaussianBlur(img_gray.astype(float), (ksize, ksize), 0))
    for ksize in [5, 15]:
        blur = cv2.GaussianBlur(img_gray.astype(float), (ksize, ksize), 0)
        blur_sq = cv2.GaussianBlur((img_gray.astype(float))**2, (ksize, ksize), 0)
        features.append(np.sqrt(np.maximum(blur_sq - blur**2, 0)))
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_angle = np.arctan2(sobely, sobelx)
    features.extend([edge_mag, np.sin(edge_angle)*edge_mag, np.cos(edge_angle)*edge_mag])
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    features.extend([y.astype(float)/h*0.3, x.astype(float)/w*0.3])
    features.append(np.abs(cv2.Laplacian(img_gray, cv2.CV_64F)))
    feat = np.stack(features, axis=-1)
    for i in range(feat.shape[-1]):
        fmin, fmax = feat[:,:,i].min(), feat[:,:,i].max()
        if fmax > fmin: feat[:,:,i] = (feat[:,:,i] - fmin) / (fmax - fmin)
    return feat

def is_natural(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,1].std() > 30 and hsv[:,:,0].std() > 20

def guided_filter(guide, src, radius=5, eps=0.01):
    g = guide.astype(np.float32)/255.0; s = src.astype(np.float32)
    mg = cv2.boxFilter(g,-1,(radius,radius)); ms = cv2.boxFilter(s,-1,(radius,radius))
    mgs = cv2.boxFilter(g*s,-1,(radius,radius)); mgg = cv2.boxFilter(g*g,-1,(radius,radius))
    a = (mgs - mg*ms)/(mgg - mg*mg + eps); b = ms - a*mg
    return cv2.boxFilter(a,-1,(radius,radius))*g + cv2.boxFilter(b,-1,(radius,radius))

def ab_to_bgr(ab, L):
    ab_u = np.clip(ab+128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.stack([L, ab_u[:,:,0], ab_u[:,:,1]], axis=-1), cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()

# Build training database
train_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')[:200]
Xlist, Ylist = [], []
ct = 0
for p in train_paths:
    im = cv2.imread(p)
    if im is None or not is_natural(im): continue
    ct += 1
    r = cv2.resize(im, (64, 64))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    f = extract_features(g).reshape(-1, 52)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128
    idx = np.random.choice(len(f), min(600, len(f)), replace=False)
    Xlist.append(f[idx]); Ylist.append(ab.reshape(-1, 2)[idx])
X_db = np.vstack(Xlist); Y_db = np.vstack(Ylist)
print(f'Database: {X_db.shape[0]:,} samples from {ct} images')

# Separate directions and magnitudes in the database
Y_mag = np.sqrt(Y_db[:,0]**2 + Y_db[:,1]**2)  # magnitude (saturation)
Y_dir = np.zeros_like(Y_db)  # direction (unit vectors on ab circle)
nonzero = Y_mag > 0.5
Y_dir[nonzero] = Y_db[nonzero] / Y_mag[nonzero, np.newaxis]
Y_dir[~nonzero] = 0  # near-gray pixels have no direction

print(f'Direction stats: mean_mag={Y_mag.mean():.1f}, median_mag={np.median(Y_mag):.1f}')
print(f'  Saturated (mag>5): {(Y_mag > 5).mean():.1%}')
print(f'  Gray (mag<2): {(Y_mag < 2).mean():.1%}')

# Build k-NN
print('Building k-NN...')
knn = NearestNeighbors(n_neighbors=25, algorithm='ball_tree')
knn.fit(X_db)

# Load DDColor for comparison
v16 = V16GeometricColorizer()

# Test on multiple images
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [
    all_imgs[50],   # diverse test images
    all_imgs[52],
    all_imgs[54],
    all_imgs[56],
]

rows = []
for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    r = cv2.resize(im, (256, 256))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]

    # DDColor
    t = torch.from_numpy(gbgr.transpose(2,0,1)).float().unsqueeze(0)/255.0
    with torch.no_grad(): ab_dd = v16.forward(t)
    bgr_dd = ab_to_bgr(ab_dd[0].permute(1,2,0).numpy(), L)

    # v8 baseline (trimmed mean)
    feat = extract_features(g).reshape(-1, 52)
    dist, idx = knn.kneighbors(feat)

    def trimmed_mean(colors, trim_pct=0.2):
        med = np.median(colors, axis=0)
        d = np.sqrt(np.sum((colors - med)**2, axis=1))
        n = int(len(colors)*trim_pct)
        if n > 0:
            mask = d < np.sort(d)[-n]
            if mask.sum() > 0: return colors[mask].mean(axis=0)
        return colors.mean(axis=0)

    ab_v8 = np.array([trimmed_mean(Y_db[idx[i]]) for i in range(len(feat))]).reshape(256,256,2)
    ab_v8_a = guided_filter(g, cv2.bilateralFilter(ab_v8[:,:,0].astype(np.float32),9,50,50))
    ab_v8_b = guided_filter(g, cv2.bilateralFilter(ab_v8[:,:,1].astype(np.float32),9,50,50))
    bgr_v8 = ab_to_bgr(np.stack([ab_v8_a, ab_v8_b], axis=-1) * 1.3, L)

    # v9: Direction/Magnitude separation
    # Step 1: Get DIRECTION from k-NN (median of unit directions)
    dir_result = np.zeros((len(feat), 2))
    mag_result = np.zeros(len(feat))

    for i in range(len(feat)):
        neighbor_dirs = Y_dir[idx[i]]
        neighbor_mags = Y_mag[idx[i]]

        # For direction: use circular mean (treat as angles)
        neighbor_ab = Y_db[idx[i]]
        angles = np.arctan2(neighbor_ab[:,1], neighbor_ab[:,0])

        # Weighted by magnitude - saturated neighbors' directions matter more
        weights = neighbor_mags / (neighbor_mags.sum() + 1e-8)
        mean_sin = np.sum(weights * np.sin(angles))
        mean_cos = np.sum(weights * np.cos(angles))
        direction_angle = np.arctan2(mean_sin, mean_cos)

        dir_result[i, 0] = np.cos(direction_angle)
        dir_result[i, 1] = np.sin(direction_angle)

        # For magnitude: use LOCAL IMAGE STATISTICS instead of averaging
        # The magnitude should come from the IMAGE, not the database
        mag_result[i] = np.median(neighbor_mags)

    dir_map = dir_result.reshape(256, 256, 2)
    mag_map = mag_result.reshape(256, 256)

    # Compute magnitude from image structure
    # High contrast regions → more saturated
    # Low contrast regions → less saturated
    local_contrast = cv2.GaussianBlur((g.astype(float))**2, (15,15), 0) - \
                     cv2.GaussianBlur(g.astype(float), (15,15), 0)**2
    local_contrast = np.sqrt(np.maximum(local_contrast, 0))
    # Normalize
    lc_norm = local_contrast / (local_contrast.max() + 1e-8)

    # Combine: use k-NN magnitude as base, modulate by local contrast
    # This gives us per-pixel adaptive saturation
    mag_adaptive = mag_map * (0.7 + 0.6 * lc_norm)  # Scale between 0.7x and 1.3x

    # Reconstruct ab = direction × magnitude
    ab_v9 = np.zeros((256, 256, 2))
    ab_v9[:,:,0] = dir_map[:,:,0] * mag_adaptive
    ab_v9[:,:,1] = dir_map[:,:,1] * mag_adaptive

    # Smooth
    ab_v9_a = guided_filter(g, cv2.bilateralFilter(ab_v9[:,:,0].astype(np.float32),9,50,50))
    ab_v9_b = guided_filter(g, cv2.bilateralFilter(ab_v9[:,:,1].astype(np.float32),9,50,50))
    ab_v9_smooth = np.stack([ab_v9_a, ab_v9_b], axis=-1)

    bgr_v9 = ab_to_bgr(ab_v9_smooth, L)

    # Build comparison row: Gray | v8 | v9 | DDColor | GT
    row = np.hstack([gbgr, bgr_v8, bgr_v9, bgr_dd, r])

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ('Gray', 5),
        (f'v8 avg (s={get_sat(bgr_v8):.0f})', 261),
        (f'v9 dir/mag (s={get_sat(bgr_v9):.0f})', 517),
        (f'DDColor (s={get_sat(bgr_dd):.0f})', 773),
        ('GT', 1029),
    ]
    for txt, xo in labels:
        cv2.putText(row, txt, (xo, 18), font, 0.35, (255,255,255), 2)
        cv2.putText(row, txt, (xo, 18), font, 0.35, (0,0,0), 1)

    rows.append(row)
    name = img_path.split('/')[-1]
    print(f'  {name}: v8 sat={get_sat(bgr_v8):.0f}, v9 sat={get_sat(bgr_v9):.0f}, DDColor sat={get_sat(bgr_dd):.0f}, GT sat={get_sat(r):.0f}')

full = np.vstack(rows)
out_path = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/v9_direction_magnitude.jpg'
cv2.imwrite(out_path, full)
print(f'\nSaved: {out_path}')
print('Done!')
