"""
v11 Colorizer: k-NN Content + Selection (not Averaging)

The key insight from v8-v10:
- v8 has right CONTENT (k-NN from real data) but wrong OPERATION (averaging → gray)
- v10 has right OPERATION (selection via softmax) but wrong CONTENT (random → gray)
- v11: combine both. Use k-NN data but SELECT instead of AVERAGE.

Selection mechanism:
For each pixel, find 25 nearest neighbors. Their colors disagree.
Instead of averaging (which kills saturation), find the DOMINANT color direction
among the neighbors. Use THAT direction's magnitude.

This is like a vote: blue neighbors vote blue, yellow vote yellow.
The majority wins. No compromise. No green from blue+yellow.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cv2
import sys
import glob

sys.path.insert(0, '/home/thorin/truthspace-lcm')

from sklearn.neighbors import NearestNeighbors
import torch

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

print('=== v11: k-NN Content + Selection ===')
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

def select_dominant_color(neighbor_colors):
    """
    SELECTION not averaging.
    
    Among k neighbor colors, find the dominant cluster and return its center.
    Uses angular binning in ab-space: bin by hue angle, pick largest bin.
    """
    mags = np.sqrt(neighbor_colors[:,0]**2 + neighbor_colors[:,1]**2)
    
    # Split into "gray" and "chromatic" neighbors
    gray_mask = mags < 3
    chrom_mask = ~gray_mask
    
    if chrom_mask.sum() == 0:
        # All gray - return gray (low saturation)
        return np.array([0.0, 0.0])
    
    chrom_colors = neighbor_colors[chrom_mask]
    chrom_mags = mags[chrom_mask]
    
    # Bin by hue angle (6 bins = 60° each)
    angles = np.arctan2(chrom_colors[:,1], chrom_colors[:,0])
    n_bins = 6
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_ids = np.digitize(angles, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    
    # Find the bin with the most votes, weighted by magnitude
    bin_weights = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_weights[b] = chrom_mags[mask].sum()
    
    best_bin = np.argmax(bin_weights)
    winners = bin_ids == best_bin
    
    if winners.sum() == 0:
        return chrom_colors.mean(axis=0)
    
    # Return the MEDIAN of the winning bin (preserves magnitude better than mean)
    return np.median(chrom_colors[winners], axis=0)

# Build database
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
print(f'Database: {X_db.shape[0]:,} from {ct} images')

knn = NearestNeighbors(n_neighbors=25, algorithm='ball_tree')
knn.fit(X_db)

v16 = V16GeometricColorizer()

# Test images
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in [50, 52, 54, 56]]

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
    
    # v8 (trimmed mean baseline)
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
    for c in range(2):
        ab_v8[:,:,c] = guided_filter(g, cv2.bilateralFilter(ab_v8[:,:,c].astype(np.float32),9,50,50))
    bgr_v8 = ab_to_bgr(ab_v8 * 1.3, L)
    
    # v11 (selection)
    ab_v11 = np.array([select_dominant_color(Y_db[idx[i]]) for i in range(len(feat))]).reshape(256,256,2)
    for c in range(2):
        ab_v11[:,:,c] = guided_filter(g, cv2.bilateralFilter(ab_v11[:,:,c].astype(np.float32),9,50,50))
    bgr_v11 = ab_to_bgr(ab_v11, L)
    
    # Row: Gray | v8 (avg) | v11 (select) | DDColor | GT
    row = np.hstack([gbgr, bgr_v8, bgr_v11, bgr_dd, r])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ('Gray', 5),
        (f'v8 avg s={get_sat(bgr_v8):.0f}', 261),
        (f'v11 select s={get_sat(bgr_v11):.0f}', 517),
        (f'DDColor s={get_sat(bgr_dd):.0f}', 773),
        ('GT', 1029),
    ]
    for txt, xo in labels:
        cv2.putText(row, txt, (xo, 18), font, 0.32, (255,255,255), 2)
        cv2.putText(row, txt, (xo, 18), font, 0.32, (0,0,0), 1)
    
    rows.append(row)
    name = img_path.split('/')[-1]
    print(f'  {name}: v8={get_sat(bgr_v8):.0f}, v11={get_sat(bgr_v11):.0f}, DDColor={get_sat(bgr_dd):.0f}, GT={get_sat(r):.0f}')

full = np.vstack(rows)
out = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/v11_knn_selection.jpg'
cv2.imwrite(out, full)
print(f'\nSaved: {out}')
print('Done!')
