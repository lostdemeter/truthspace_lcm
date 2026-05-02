"""
v10 Colorizer: Test the Shape Hypothesis

HYPOTHESIS: If knowledge is a shape, then a RANDOM rotation with the
correct SHAPE properties (Zipf-1/φ singular values) should produce
plausible colorization. The specific content of the rotation shouldn't matter -
only its geometric properties.

We test 3 conditions:
A) Random rotation with FLAT singular values (no structure)
B) Random rotation with ZIPF-1/φ singular values (DDColor's shape)  
C) Random rotation with STEEP singular values (too much structure)

If B is better than A and C → shape matters, and φ-shape is special
If all are similar → shape doesn't matter (content does)
If none work → we need more than just the rotation shape

The pipeline (from Doc 230):
1. PROJECT: Gabor features (φ-scaled) → feature space
2. ROTATE: Random matrix with controlled shape → semantic space  
3. PROJECT: Softmax over 100 queries → selection
4. DILATE: Local image statistics → magnitude
5. LOOKUP: Natural color centers → ab direction
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import sys
import glob

sys.path.insert(0, '/home/thorin/truthspace-lcm')

from sklearn.cluster import KMeans
from scipy.special import softmax

PHI = 1.618033988749895

print('=== v10: TESTING THE SHAPE HYPOTHESIS ===')
print()
print('If knowledge is a shape, random rotations with the RIGHT shape')
print('should produce plausible colors.')
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

# ======== Step 1: Build color vocabulary from natural images ========
print('Building natural color vocabulary...')
train_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')[:100]
all_ab = []
for p in train_paths:
    im = cv2.imread(p)
    if im is None: continue
    r = cv2.resize(im, (64, 64))
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128
    all_ab.append(ab.reshape(-1, 2))
all_ab = np.vstack(all_ab)
sat = np.sqrt(all_ab[:,0]**2 + all_ab[:,1]**2)
saturated_ab = all_ab[sat > 5]

N_QUERIES = 100
km = KMeans(n_clusters=N_QUERIES, random_state=42, n_init=10)
km.fit(saturated_ab)
color_vocab = km.cluster_centers_  # [100, 2] - the 100 natural color directions
print(f'  {N_QUERIES} color centers from {len(saturated_ab):,} saturated pixels')

# ======== Step 2: Build random rotations with controlled shapes ========
print('Building shaped rotations...')

N_FEAT = 52  # our feature dimension

def make_shaped_rotation(n_in, n_out, alpha, scale=1.0, seed=42):
    """Create a random matrix with Zipf-alpha singular value spectrum."""
    rng = np.random.RandomState(seed)
    # Random orthogonal bases
    U, _ = np.linalg.qr(rng.randn(n_out, n_out))
    V, _ = np.linalg.qr(rng.randn(n_in, n_in))
    # Shaped singular values
    k = min(n_in, n_out)
    if alpha == 0:  # flat
        S = np.ones(k) * scale
    else:
        S = scale / np.arange(1, k+1)**alpha
    # Build matrix
    Sigma = np.zeros((n_out, n_in))
    for i in range(k):
        Sigma[i, i] = S[i]
    return U @ Sigma @ V[:n_in, :n_in]

# Condition A: FLAT spectrum (alpha=0) - no structure
W_flat = make_shaped_rotation(N_FEAT, N_QUERIES, alpha=0.0, scale=1.0, seed=42)

# Condition B: ZIPF-1/φ spectrum - DDColor's shape
W_phi = make_shaped_rotation(N_FEAT, N_QUERIES, alpha=1/PHI, scale=3.0, seed=42)

# Condition C: STEEP spectrum (alpha=2) - too much structure
W_steep = make_shaped_rotation(N_FEAT, N_QUERIES, alpha=2.0, scale=10.0, seed=42)

# Verify shapes
for name, W in [('Flat', W_flat), ('φ-Zipf', W_phi), ('Steep', W_steep)]:
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    print(f'  {name}: S[0]={S[0]:.2f}, S[-1]={S[-1]:.4f}, condition={S[0]/S[-1]:.0f}')

# ======== Step 3: Colorize ========
print()
print('Colorizing test images...')

def colorize_with_shape(img_gray, W, color_vocab, temperature=0.5):
    """
    Pipeline: feature → rotate → softmax → select color → dilate
    
    This implements the shape program:
    PROJECT (Gabor) → ROTATE (W) → PROJECT (softmax) → DILATE (contrast) → LOOKUP (vocab)
    """
    h, w = img_gray.shape
    feat = extract_features(img_gray)  # [h, w, 52]
    feat_flat = feat.reshape(-1, N_FEAT)  # [h*w, 52]
    
    # ROTATE: apply shaped rotation to get query scores
    scores = feat_flat @ W.T  # [h*w, 100]
    
    # PROJECT: softmax to get query weights (selection, not averaging)
    weights = softmax(scores / temperature, axis=1)  # [h*w, 100]
    
    # LOOKUP: weighted combination of color directions
    ab_dirs = weights @ color_vocab  # [h*w, 2]
    
    # DILATE: magnitude from local image statistics
    local_contrast = cv2.GaussianBlur((img_gray.astype(float))**2, (15,15), 0) - \
                     cv2.GaussianBlur(img_gray.astype(float), (15,15), 0)**2
    local_contrast = np.sqrt(np.maximum(local_contrast, 0))
    lc_flat = local_contrast.reshape(-1)
    lc_norm = lc_flat / (lc_flat.max() + 1e-8)
    
    # Scale: higher contrast regions get more saturation
    magnitude = 0.8 + 0.5 * lc_norm  # range [0.8, 1.3]
    
    ab = ab_dirs * magnitude[:, np.newaxis]
    ab = ab.reshape(h, w, 2)
    
    # Smooth
    ab[:,:,0] = guided_filter(img_gray, cv2.bilateralFilter(ab[:,:,0].astype(np.float32), 9, 50, 50))
    ab[:,:,1] = guided_filter(img_gray, cv2.bilateralFilter(ab[:,:,1].astype(np.float32), 9, 50, 50))
    
    return ab

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
    
    results = {}
    for name, W in [('Flat (α=0)', W_flat), ('φ-Zipf (α=1/φ)', W_phi), ('Steep (α=2)', W_steep)]:
        ab = colorize_with_shape(g, W, color_vocab, temperature=0.5)
        bgr = ab_to_bgr(ab, L)
        results[name] = bgr
    
    # Row: Gray | Flat | φ-Zipf | Steep | GT
    row = np.hstack([gbgr] + [results[n] for n in results] + [r])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [('Gray', 5)]
    for i, name in enumerate(results):
        bgr = results[name]
        s = get_sat(bgr)
        labels.append((f'{name} s={s:.0f}', 261 + i*256))
    labels.append(('GT', 261 + len(results)*256))
    
    for txt, xo in labels:
        cv2.putText(row, txt, (xo, 18), font, 0.32, (255,255,255), 2)
        cv2.putText(row, txt, (xo, 18), font, 0.32, (0,0,0), 1)
    
    rows.append(row)
    name = img_path.split('/')[-1]
    sats = [f'{n}: {get_sat(results[n]):.0f}' for n in results]
    print(f'  {name} | {" | ".join(sats)} | GT: {get_sat(r):.0f}')

full = np.vstack(rows)
out = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/v10_shape_hypothesis.jpg'
cv2.imwrite(out, full)
print(f'\nSaved: {out}')

# ======== Also test: does the SHAPE affect spatial coherence? ========
print()
print('=== SPATIAL COHERENCE TEST ===')
# For each condition, measure how much color changes between adjacent pixels
# relative to how much grayscale changes (should be low = edge-respecting)

im = cv2.imread(test_paths[0])
r = cv2.resize(im, (256, 256))
g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
L = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)[:,:,0]

for name, W in [('Flat', W_flat), ('φ-Zipf', W_phi), ('Steep', W_steep)]:
    ab = colorize_with_shape(g, W, color_vocab)
    
    # Color gradient magnitude
    color_dx = np.diff(ab, axis=1)
    color_dy = np.diff(ab, axis=0)
    color_grad = np.sqrt(color_dx[:255,:,0]**2 + color_dx[:255,:,1]**2 + 
                         color_dy[:,:255,0]**2 + color_dy[:,:255,1]**2)
    
    # Gray gradient magnitude
    gray_dx = np.abs(np.diff(g.astype(float), axis=1))
    gray_dy = np.abs(np.diff(g.astype(float), axis=0))
    gray_grad = gray_dx[:255,:] + gray_dy[:,:255]
    
    # Correlation: do color changes happen at gray edges?
    corr = np.corrcoef(color_grad.flatten(), gray_grad.flatten())[0,1]
    
    # Smoothness: average color gradient in FLAT regions (gray_grad < 5)
    flat_mask = gray_grad < 5
    smoothness = color_grad[flat_mask].mean() if flat_mask.sum() > 0 else -1
    
    print(f'  {name:8s}: edge_corr={corr:.3f}, flat_smoothness={smoothness:.2f}')

print()
print('Higher edge_corr = colors change at edges (good)')
print('Lower flat_smoothness = colors are uniform in flat regions (good)')
print()
print('Done!')
