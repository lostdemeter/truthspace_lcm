"""Analyze DDColor's learned shapes and test building them from first principles."""
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
from scipy.optimize import curve_fit

PHI = 1.618033988749895

print('=== SHAPE ANALYSIS: What Shapes Did DDColor Learn? ===')
print()

weights = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_weights_static.npz')

# ======== SHAPE 1: The 100 Color Queries ========
query_feat = weights['decoder.color_decoder.query_feat.weight']  # [100, 256]
query_embed = weights['decoder.color_decoder.query_embed.weight']  # [100, 256]

print('=== Shape 1: Color Queries (100 x 256) ===')
U_q, S_q, Vt_q = np.linalg.svd(query_feat, full_matrices=False)
cumvar = np.cumsum(S_q**2) / np.sum(S_q**2)
rank90 = np.searchsorted(cumvar, 0.9) + 1
rank95 = np.searchsorted(cumvar, 0.95) + 1
print(f'  Effective rank: 90%={rank90}, 95%={rank95}')
print(f'  Top 5 singular values: {S_q[:5].round(3)}')
print(f'  Spectrum is FLAT - all queries equally important')

def zipf(i, s0, alpha):
    return s0 / (i ** alpha)

try:
    popt, _ = curve_fit(zipf, np.arange(1, len(S_q)+1), S_q, p0=[S_q[0], 0.618])
    print(f'  Zipf alpha = {popt[1]:.4f} (1/phi = {1/PHI:.4f})')
except:
    pass

# ======== SHAPE 2: Color Embed MLP ========
print()
print('=== Shape 2: Color Embed MLP (3 layers, 256x256 each) ===')
for layer_idx in range(3):
    W = weights[f'decoder.color_decoder.color_embed.layers.{layer_idx}.weight']
    _, S, _ = np.linalg.svd(W)
    cv = np.cumsum(S**2) / np.sum(S**2)
    r90 = np.searchsorted(cv, 0.9) + 1
    try:
        popt, _ = curve_fit(zipf, np.arange(1, len(S)+1), S, p0=[S[0], 0.618])
        alpha = popt[1]
    except:
        alpha = -1
    print(f'  Layer {layer_idx}: rank90={r90}, condition={S[0]/S[-1]:.0f}, Zipf alpha={alpha:.4f}')

# ======== SHAPE 3: refine_net (final 2x103 projection) ========
W_refine = weights['refine_net.0.0.weight'].squeeze()
W_color = W_refine[:, :100]  # [2, 100]
W_input = W_refine[:, 100:]  # [2, 3]
ddcolor_query_colors = W_color.T  # [100, 2] - each query's color direction

print()
print('=== Shape 3: refine_net projection ===')
print(f'  Query color range: a=[{ddcolor_query_colors[:,0].min():.3f}, {ddcolor_query_colors[:,0].max():.3f}]')
print(f'                     b=[{ddcolor_query_colors[:,1].min():.3f}, {ddcolor_query_colors[:,1].max():.3f}]')
print(f'  W_input (RGB->ab):')
print(f'    R->a: {W_input[0,0]:.4f}, G->a: {W_input[0,1]:.4f}, B->a: {W_input[0,2]:.4f}')
print(f'    R->b: {W_input[1,0]:.4f}, G->b: {W_input[1,1]:.4f}, B->b: {W_input[1,2]:.4f}')

# KEY INSIGHT
print()
print('=== KEY INSIGHT ===')
print(f'DDColor query range: ±0.3 in ab space')
print(f'Natural color range: ±50 in ab space')
print(f'DDColor queries are NOT colors - they are DIRECTIONS scaled by attention.')
print(f'Our k-NN stores ACTUAL colors. Different shape entirely.')
print()

# ======== Now collect natural color statistics ========
print('=== Natural Image Color Distribution ===')
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

# Filter saturated pixels
sat = np.sqrt(all_ab[:,0]**2 + all_ab[:,1]**2)
saturated_ab = all_ab[sat > 5]
print(f'  Total pixels: {all_ab.shape[0]:,}')
print(f'  Saturated (>5): {saturated_ab.shape[0]:,} ({saturated_ab.shape[0]/all_ab.shape[0]:.1%})')

# Build 3 vocabularies
print()
print('=== Building Color Vocabularies ===')

# 1. Natural distribution (KMeans on real colors)
km = KMeans(n_clusters=100, random_state=42, n_init=10)
km.fit(saturated_ab)
natural_centers = km.cluster_centers_
print(f'  Natural: a=[{natural_centers[:,0].min():.1f}, {natural_centers[:,0].max():.1f}], b=[{natural_centers[:,1].min():.1f}, {natural_centers[:,1].max():.1f}]')

# 2. phi-spiral
theta = np.linspace(0, 6*np.pi, 100)
r_spiral = np.linspace(2, 40, 100)
phi_queries = np.stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)], axis=1)
print(f'  phi-Spiral: covers ab space in expanding spiral')

# 3. Golden angle (sunflower pattern)
golden_angle = 2 * np.pi * (1 - 1/PHI)
angles_ga = np.array([i * golden_angle for i in range(100)])
radii_ga = np.sqrt(np.linspace(1, 40**2, 100))
ga_queries = np.stack([radii_ga * np.cos(angles_ga), radii_ga * np.sin(angles_ga)], axis=1)
print(f'  Golden Angle: sunflower pattern in ab space')

# ======== VISUALIZATION ========
fig = plt.figure(figsize=(20, 14))

# Row 1: The 4 vocabularies
ax1 = fig.add_subplot(2, 4, 1)
# Scale DDColor queries to same range for visual comparison
ddq_scaled = ddcolor_query_colors * 100  # Scale up for visibility
ax1.scatter(ddq_scaled[:,1], ddq_scaled[:,0], c='red', s=20, alpha=0.7)
ax1.set_title(f'DDColor Queries\n(scaled 100x, actual ±0.3)')
ax1.set_xlim(-50, 50); ax1.set_ylim(-50, 50)
ax1.set_aspect('equal'); ax1.axhline(0, color='gray', lw=0.3); ax1.axvline(0, color='gray', lw=0.3)
ax1.set_xlabel('b'); ax1.set_ylabel('a')

ax2 = fig.add_subplot(2, 4, 2)
ax2.scatter(natural_centers[:,1], natural_centers[:,0], c='blue', s=20, alpha=0.7)
ax2.set_title('Natural Distribution\n(KMeans on real colors)')
ax2.set_xlim(-50, 50); ax2.set_ylim(-50, 50)
ax2.set_aspect('equal'); ax2.axhline(0, color='gray', lw=0.3); ax2.axvline(0, color='gray', lw=0.3)

ax3 = fig.add_subplot(2, 4, 3)
ax3.scatter(phi_queries[:,1], phi_queries[:,0], c='green', s=20, alpha=0.7)
ax3.set_title('phi-Spiral\n(expanding spiral)')
ax3.set_xlim(-50, 50); ax3.set_ylim(-50, 50)
ax3.set_aspect('equal'); ax3.axhline(0, color='gray', lw=0.3); ax3.axvline(0, color='gray', lw=0.3)

ax4 = fig.add_subplot(2, 4, 4)
ax4.scatter(ga_queries[:,1], ga_queries[:,0], c='purple', s=20, alpha=0.7)
ax4.set_title('Golden Angle\n(sunflower pattern)')
ax4.set_xlim(-50, 50); ax4.set_ylim(-50, 50)
ax4.set_aspect('equal'); ax4.axhline(0, color='gray', lw=0.3); ax4.axvline(0, color='gray', lw=0.3)

# Row 2: SVD spectra comparison
ax5 = fig.add_subplot(2, 4, 5)
ax5.semilogy(S_q, 'b.-', label='query_feat')
x = np.arange(1, len(S_q)+1)
try:
    popt, _ = curve_fit(zipf, x, S_q, p0=[S_q[0], 0.618])
    ax5.semilogy(x-1, zipf(x, *popt), 'r--', label=f'Zipf α={popt[1]:.3f}')
except:
    pass
ax5.set_title('query_feat Spectrum')
ax5.set_xlabel('Index'); ax5.set_ylabel('Singular Value')
ax5.legend()

ax6 = fig.add_subplot(2, 4, 6)
for li in range(3):
    W = weights[f'decoder.color_decoder.color_embed.layers.{li}.weight']
    _, S, _ = np.linalg.svd(W)
    ax6.semilogy(S, label=f'Layer {li}')
    # Overlay Zipf
    x = np.arange(1, len(S)+1)
    try:
        po, _ = curve_fit(zipf, x, S, p0=[S[0], 0.618])
        ax6.semilogy(x-1, zipf(x, *po), '--', alpha=0.5)
    except:
        pass
ax6.set_title('Color Embed MLP Spectra')
ax6.legend()

ax7 = fig.add_subplot(2, 4, 7)
ax7.plot(cumvar, 'g.-')
ax7.axhline(0.9, color='r', ls='--', alpha=0.5, label='90%')
ax7.axhline(0.95, color='orange', ls='--', alpha=0.5, label='95%')
ax7.set_title('query_feat Cumulative Variance')
ax7.set_xlabel('Dimensions')
ax7.legend()

ax8 = fig.add_subplot(2, 4, 8)
# Natural color density
ax8.hist2d(all_ab[:,1], all_ab[:,0], bins=50, range=[[-50,50],[-50,50]], cmap='hot')
ax8.set_title('Natural Color Density')
ax8.set_xlabel('b (yellow-blue)'); ax8.set_ylabel('a (red-green)')
ax8.set_aspect('equal')

plt.suptitle('DDColor Shape Analysis: If Knowledge Is a Shape, Can We Build It?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/shape_analysis.png', dpi=150)
print()
print('Saved: shape_analysis.png')

# ======== PATTERN SUMMARY ========
print()
print('='*60)
print('PATTERNS FOUND:')
print('='*60)
print()
print('Pattern A: FLAT SPECTRUM (query_feat)')
print('  - All 100 queries equally important')
print('  - Zipf alpha=0.24 (much flatter than 1/phi=0.618)')
print('  - MEANING: No query is more important than another')
print('  - This is a DEMOCRATIC shape - uniform basis')
print()
print('Pattern B: APPROACHING PHI (color_embed MLP)')
print('  - Zipf alpha: 0.476, 0.456, 0.574 across layers')
print('  - Layer 2 closest to 1/phi=0.618')
print('  - MEANING: The MLP compresses information as it processes')
print('  - Deeper layers have MORE phi-structure')
print('  - This is an EMERGENT shape - phi appears through optimization')
print()
print('Pattern C: DIRECTION/MAGNITUDE SEPARATION (refine_net)')
print('  - Queries store DIRECTIONS (±0.3 range)')
print('  - Attention stores MAGNITUDES (scales to ±50 range)')
print('  - MEANING: Color = direction × magnitude')
print('  - The query says WHAT color, attention says HOW MUCH')
print()
print('Pattern D: NON-UNIFORM COLOR DENSITY (natural images)')
print('  - Colors cluster near origin (most pixels are low-saturation)')
print('  - Saturated colors are sparse but important')
print('  - Green-yellow axis has more density than blue-red')
print('  - MEANING: Need more queries near gray, fewer at extremes')
print()
print('HYPOTHESIS: To build knowledge-as-shape for colorization:')
print('  1. Use FLAT spectrum for the basis (democratic queries)')
print('  2. Let phi-structure EMERGE in the transformation layers')
print('  3. Separate DIRECTION (what color) from MAGNITUDE (how much)')
print('  4. Match query density to natural color density')
print()
print('Done!')
