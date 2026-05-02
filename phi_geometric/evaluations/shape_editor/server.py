"""
Live Shape Editor - Backend
Real-time interactive colorization parameter tuning.
"""
import numpy as np
import cv2
import glob
import sys
import os
import json
import time
import base64
from io import BytesIO
from flask import Flask, send_from_directory, request, jsonify

sys.path.insert(0, '/home/thorin/truthspace-lcm')

from sklearn.neighbors import NearestNeighbors

PHI = 1.618033988749895

app = Flask(__name__)

# ========== GLOBAL STATE ==========
DB = {}       # Pre-computed database
IMAGES = {}   # Pre-loaded test images
CACHE = {}    # Feature/kNN cache per image

def extract_features(img_gray, n_gabor_scales=5, position_weight=0.3):
    h, w = img_gray.shape
    features = []
    for scale in range(n_gabor_scales):
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
    features.extend([y.astype(float)/h * position_weight, x.astype(float)/w * position_weight])
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

def select_dominant_color(colors, n_bins=6):
    mags = np.sqrt(colors[:,0]**2 + colors[:,1]**2)
    chrom_mask = mags > 3
    if chrom_mask.sum() == 0:
        return np.array([0.0, 0.0])
    chrom = colors[chrom_mask]
    chrom_mags = mags[chrom_mask]
    angles = np.arctan2(chrom[:,1], chrom[:,0])
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

def trimmed_mean(colors, trim_pct=0.2):
    med = np.median(colors, axis=0)
    d = np.sqrt(np.sum((colors - med)**2, axis=1))
    n = int(len(colors)*trim_pct)
    if n > 0:
        mask = d < np.sort(d)[-n]
        if mask.sum() > 0: return colors[mask].mean(axis=0)
    return colors.mean(axis=0)

def segment_image(gray, region_step=16, min_region_size=20):
    blur = cv2.GaussianBlur(gray, (5,5), 1.5)
    markers = np.zeros_like(gray, dtype=np.int32)
    marker_id = 1
    h, w = gray.shape
    for yi in range(region_step//2, h, region_step):
        for xi in range(region_step//2, w, region_step):
            markers[yi, xi] = marker_id
            marker_id += 1
    watershed_input = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input, markers)
    labels = markers.copy()
    labels[labels == -1] = 0
    for rid in np.unique(labels):
        if rid == 0: continue
        rmask = labels == rid
        if rmask.sum() < min_region_size and rmask.sum() > 0:
            dilated = cv2.dilate(rmask.astype(np.uint8), np.ones((3,3), np.uint8))
            border = (dilated > 0) & ~rmask
            if border.sum() > 0:
                nl = labels[border]
                nl = nl[(nl != rid) & (nl != 0)]
                if len(nl) > 0:
                    vals, counts = np.unique(nl, return_counts=True)
                    labels[rmask] = vals[np.argmax(counts)]
    boundary = labels == 0
    if boundary.sum() > 0:
        dl = cv2.dilate(labels.astype(np.float32), np.ones((3,3), np.uint8)).astype(np.int32)
        labels[boundary] = dl[boundary]
    return labels

def colorize(img_name, params):
    """Main colorization with all tunable parameters."""
    t0 = time.time()
    
    info = IMAGES[img_name]
    gray = info['gray']
    L = info['L']
    h, w = gray.shape
    
    # Get cached features and kNN results
    feat = info['feat_flat']
    idx = info['knn_idx']
    
    method = params.get('method', 'region_select')
    k_neighbors = min(params.get('k_neighbors', 25), idx.shape[1])
    n_hue_bins = params.get('n_hue_bins', 6)
    region_step = params.get('region_step', 16)
    min_region = params.get('min_region', 20)
    sat_boost = params.get('sat_boost', 1.0)
    bilateral_d = params.get('bilateral_d', 9)
    bilateral_sigma = params.get('bilateral_sigma', 50)
    guided_radius = params.get('guided_radius', 5)
    guided_eps = params.get('guided_eps', 0.01)
    chroma_threshold = params.get('chroma_threshold', 3)
    
    Y_db = DB['Y']
    
    # Trim k-NN results to k_neighbors
    idx_k = idx[:, :k_neighbors]
    
    if method == 'pixel_avg':
        # v8-style trimmed mean per pixel
        ab = np.array([trimmed_mean(Y_db[idx_k[i]]) for i in range(len(feat))])
        ab = ab.reshape(h, w, 2)
        
    elif method == 'pixel_select':
        # v11-style per-pixel selection
        ab = np.array([select_dominant_color(Y_db[idx_k[i]], n_bins=n_hue_bins) 
                       for i in range(len(feat))])
        ab = ab.reshape(h, w, 2)
        
    elif method == 'region_select':
        # v12-style region voting
        labels = segment_image(gray, region_step=region_step, min_region_size=min_region)
        ab = np.zeros((h, w, 2))
        for rid in np.unique(labels):
            rmask = labels == rid
            rpx = np.where(rmask.flatten())[0]
            if len(rpx) == 0: continue
            all_colors = np.vstack([Y_db[idx_k[px]] for px in rpx])
            color = select_dominant_color(all_colors, n_bins=n_hue_bins)
            ab[rmask] = color
            
    elif method == 'region_avg':
        # Region-based but with averaging instead of selection
        labels = segment_image(gray, region_step=region_step, min_region_size=min_region)
        ab = np.zeros((h, w, 2))
        for rid in np.unique(labels):
            rmask = labels == rid
            rpx = np.where(rmask.flatten())[0]
            if len(rpx) == 0: continue
            all_colors = np.vstack([Y_db[idx_k[px]] for px in rpx])
            color = trimmed_mean(all_colors)
            ab[rmask] = color
    
    # Smoothing
    if bilateral_d > 0:
        for c in range(2):
            ab[:,:,c] = cv2.bilateralFilter(ab[:,:,c].astype(np.float32), 
                                             bilateral_d, bilateral_sigma, bilateral_sigma)
    if guided_radius > 0:
        for c in range(2):
            ab[:,:,c] = guided_filter(gray, ab[:,:,c].astype(np.float32), 
                                      radius=guided_radius, eps=guided_eps)
    
    # Saturation boost
    ab = ab * sat_boost
    
    # Convert to BGR
    ab_u = np.clip(ab + 128, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(np.stack([L, ab_u[:,:,0], ab_u[:,:,1]], axis=-1), cv2.COLOR_Lab2BGR)
    
    elapsed = time.time() - t0
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()
    
    return bgr, {'time': f'{elapsed:.2f}s', 'saturation': f'{sat:.0f}'}

def img_to_base64(bgr):
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

# ========== ROUTES ==========

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app.route('/api/images')
def list_images():
    return jsonify(list(IMAGES.keys()))

@app.route('/api/colorize', methods=['POST'])
def api_colorize():
    data = request.json
    img_name = data.get('image', list(IMAGES.keys())[0])
    params = data.get('params', {})
    
    if img_name not in IMAGES:
        return jsonify({'error': 'Image not found'}), 404
    
    bgr, stats = colorize(img_name, params)
    
    return jsonify({
        'image': img_to_base64(bgr),
        'gray': img_to_base64(IMAGES[img_name]['gray_bgr']),
        'gt': img_to_base64(IMAGES[img_name]['bgr']),
        'stats': stats,
    })

# ========== INIT ==========

def init():
    print('=== Shape Editor: Initializing ===')
    
    # Build training database
    print('Building k-NN database...')
    train_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')[:100]
    Xlist, Ylist = [], []
    ct = 0
    for p in train_paths:
        im = cv2.imread(p)
        if im is None or not is_natural(im): continue
        ct += 1
        r = cv2.resize(im, (48, 48))
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        f = extract_features(g).reshape(-1, 52)
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        ab = lab[:,:,1:].astype(float) - 128
        idx = np.random.choice(len(f), min(300, len(f)), replace=False)
        Xlist.append(f[idx]); Ylist.append(ab.reshape(-1, 2)[idx])
    
    X_db = np.vstack(Xlist); Y_db = np.vstack(Ylist)
    print(f'  Database: {X_db.shape[0]:,} from {ct} images')
    
    knn = NearestNeighbors(n_neighbors=35, algorithm='ball_tree')
    knn.fit(X_db)
    DB['X'] = X_db
    DB['Y'] = Y_db
    DB['knn'] = knn
    
    # Load test images and pre-compute features + kNN
    all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
    test_indices = [50, 52, 54, 56]
    
    print('Pre-computing features for test images...')
    for ti in test_indices:
        if ti >= len(all_imgs): continue
        path = all_imgs[ti]
        name = os.path.basename(path)
        im = cv2.imread(path)
        if im is None: continue
        
        SZ = 128
        bgr = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        
        print(f'    Extracting features for {name}...')
        feat = extract_features(gray)
        feat_flat = feat.reshape(-1, 52)
        print(f'    Running kNN for {name}...')
        dist, idx = knn.kneighbors(feat_flat)
        
        IMAGES[name] = {
            'bgr': bgr,
            'gray': gray,
            'gray_bgr': cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            'L': lab[:,:,0],
            'feat_flat': feat_flat,
            'knn_idx': idx,
            'knn_dist': dist,
        }
        print(f'  Loaded: {name}')
    
    print(f'Ready! {len(IMAGES)} images loaded.')

if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=8899, debug=False)
