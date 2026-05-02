"""
v12 Colorizer: Region-Based Voting

Problem in v11: each pixel votes independently → competing colors within surfaces
Fix: segment image into regions, then ALL pixels in a region vote together

Pattern 13: Region Voting
- Segment grayscale into superpixels (connected regions of similar texture)
- For each region, collect ALL k-NN votes from ALL pixels in that region
- The region gets ONE color from the collective dominant vote
- This enforces: one surface = one color

This is what DDColor's attention does implicitly - it shares information
across spatially connected regions. We're adding this explicitly.
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

print('=== v12: Region-Based Voting ===')
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

def select_dominant_color(colors):
    """Vote by hue bin, pick the winner's median."""
    mags = np.sqrt(colors[:,0]**2 + colors[:,1]**2)
    chrom_mask = mags > 3
    if chrom_mask.sum() == 0:
        return np.array([0.0, 0.0])
    
    chrom = colors[chrom_mask]
    chrom_mags = mags[chrom_mask]
    angles = np.arctan2(chrom[:,1], chrom[:,0])
    
    n_bins = 6
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_ids = np.clip(np.digitize(angles, bin_edges) - 1, 0, n_bins - 1)
    
    bin_weights = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_weights[b] = chrom_mags[mask].sum()
    
    best_bin = np.argmax(bin_weights)
    winners = bin_ids == best_bin
    
    if winners.sum() == 0:
        return chrom.mean(axis=0)
    return np.median(chrom[winners], axis=0)

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
    
    feat = extract_features(g).reshape(-1, 52)
    dist, idx = knn.kneighbors(feat)
    
    # ======== SUPERPIXEL SEGMENTATION ========
    # Use watershed on gradient magnitude for proper oversegmentation
    blur = cv2.GaussianBlur(g, (5,5), 1.5)
    
    # Create grid of seed points (roughly 16px apart → ~256 seeds for 256x256)
    step = 16
    markers = np.zeros_like(g, dtype=np.int32)
    marker_id = 1
    for yi in range(step//2, 256, step):
        for xi in range(step//2, 256, step):
            markers[yi, xi] = marker_id
            marker_id += 1
    
    # Watershed needs 3-channel image
    watershed_input = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input, markers)
    
    # Watershed sets boundaries to -1, relabel
    labels = markers.copy()
    labels[labels == -1] = 0
    
    # Merge tiny regions (< 20 pixels) into nearest larger neighbor
    for rid in np.unique(labels):
        if rid == 0: continue
        rmask = labels == rid
        if rmask.sum() < 20 and rmask.sum() > 0:
            dilated = cv2.dilate(rmask.astype(np.uint8), np.ones((3,3), np.uint8))
            border = (dilated > 0) & ~rmask
            if border.sum() > 0:
                neighbor_labels = labels[border]
                neighbor_labels = neighbor_labels[(neighbor_labels != rid) & (neighbor_labels != 0)]
                if len(neighbor_labels) > 0:
                    vals, counts = np.unique(neighbor_labels, return_counts=True)
                    labels[rmask] = vals[np.argmax(counts)]
    
    # Handle boundary pixels (label 0) - assign to nearest non-zero neighbor
    boundary = labels == 0
    if boundary.sum() > 0:
        dilated_labels = cv2.dilate(labels.astype(np.float32), np.ones((3,3), np.uint8)).astype(np.int32)
        labels[boundary] = dilated_labels[boundary]
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.vectorize(lambda x: remap.get(x, 0))(labels)
    n_regions = len(unique_labels)
    
    print(f'  {img_path.split("/")[-1]}: {n_regions} superpixels')
    
    # ======== v11: Per-pixel selection (baseline) ========
    ab_v11 = np.array([select_dominant_color(Y_db[idx[i]]) for i in range(len(feat))]).reshape(256,256,2)
    for c in range(2):
        ab_v11[:,:,c] = guided_filter(g, cv2.bilateralFilter(ab_v11[:,:,c].astype(np.float32),9,50,50))
    bgr_v11 = ab_to_bgr(ab_v11, L)
    
    # ======== v12: Region voting ========
    # For each superpixel region, collect ALL neighbor colors from ALL pixels in that region
    # Then do ONE dominant color vote for the entire region
    ab_v12 = np.zeros((256, 256, 2))
    
    for region_id in range(n_regions):
        region_mask = (labels == region_id)
        region_pixels = np.where(region_mask.flatten())[0]
        
        if len(region_pixels) == 0:
            continue
        
        # Collect ALL k-NN colors for ALL pixels in this region
        all_neighbor_colors = []
        for px in region_pixels:
            all_neighbor_colors.append(Y_db[idx[px]])
        all_neighbor_colors = np.vstack(all_neighbor_colors)
        
        # ONE vote for the whole region
        region_color = select_dominant_color(all_neighbor_colors)
        
        # Assign to all pixels in region
        ab_v12[region_mask] = region_color
    
    # Light smoothing to blend superpixel boundaries
    for c in range(2):
        ab_v12[:,:,c] = guided_filter(g, ab_v12[:,:,c].astype(np.float32), radius=3, eps=0.005)
    bgr_v12 = ab_to_bgr(ab_v12, L)
    
    # Row: Gray | v11 (pixel) | v12 (region) | DDColor | GT
    row = np.hstack([gbgr, bgr_v11, bgr_v12, bgr_dd, r])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels_txt = [
        ('Gray', 5),
        (f'v11 pixel s={get_sat(bgr_v11):.0f}', 261),
        (f'v12 region s={get_sat(bgr_v12):.0f}', 517),
        (f'DDColor s={get_sat(bgr_dd):.0f}', 773),
        ('GT', 1029),
    ]
    for txt, xo in labels_txt:
        cv2.putText(row, txt, (xo, 18), font, 0.32, (255,255,255), 2)
        cv2.putText(row, txt, (xo, 18), font, 0.32, (0,0,0), 1)
    
    rows.append(row)
    name = img_path.split('/')[-1]
    print(f'    v11={get_sat(bgr_v11):.0f}, v12={get_sat(bgr_v12):.0f}, DDColor={get_sat(bgr_dd):.0f}, GT={get_sat(r):.0f}')

full = np.vstack(rows)
out = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/v12_region_voting.jpg'
cv2.imwrite(out, full)
print(f'\nSaved: {out}')
print('Done!')
