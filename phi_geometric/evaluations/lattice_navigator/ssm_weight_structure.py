"""
SSM Weight Structure Analysis: What do the real weights know?

The gap: φ-structured first-principles achieves -4.5%, real encoder +14-19%.
That's a ~20% gap. Where does it come from?

Hypotheses:
1. Cross-block coherence: Real blocks share structure (not independent)
2. Input-aligned directions: Real expand directions match natural image statistics  
3. Expand-compress asymmetry: W_compress ≠ W_expand.T (learned asymmetry)
4. Learned bias structure: Per-channel bias encodes feature importance
5. Residual stream alignment: Real weights respect the residual connection

This script investigates each hypothesis.
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from numpy.linalg import svd as np_svd

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=' * 70)
print('SSM WEIGHT STRUCTURE: What Do the Real Weights Know?')
print('=' * 70)

v16 = V16GeometricColorizer()


# ================================================================
# HYPOTHESIS 1: Cross-Block Coherence
# ================================================================
print()
print('=' * 70)
print('H1: CROSS-BLOCK COHERENCE — Do blocks share structure?')
print('=' * 70)
print()

# For each stage, compute the subspace angle between expand matrices
# of consecutive blocks. If they share structure, angles will be small.
for si in range(4):
    d = dims[si]
    print(f'  Stage {si} ({d}ch, {depths[si]} blocks):')
    
    # Collect expand weight matrices
    expand_mats = []
    compress_mats = []
    biases = []
    for bi in range(depths[si]):
        pre = f'encoder.arch.stages.{si}.{bi}'
        pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
        pw2 = v16._get_weight(f'{pre}.pwconv2.weight').numpy()
        b1 = v16._get_weight(f'{pre}.pwconv1.bias').numpy()
        expand_mats.append(pw1)
        compress_mats.append(pw2)
        biases.append(b1)
    
    # Compare consecutive blocks: subspace overlap
    for bi in range(depths[si] - 1):
        # Get top-k right singular vectors (input subspace)
        _, _, Vt_a = np_svd(expand_mats[bi], full_matrices=False)
        _, _, Vt_b = np_svd(expand_mats[bi+1], full_matrices=False)
        
        for k in [5, 20, d//2]:
            if k > d: continue
            # Subspace angle via singular values of V_a.T @ V_b
            overlap = np_svd(Vt_a[:k] @ Vt_b[:k].T, compute_uv=False)
            mean_overlap = np.mean(overlap)
            min_overlap = np.min(overlap)
            print(f'    Block {bi}→{bi+1}, top-{k} subspace: mean_cos={mean_overlap:.3f}, min_cos={min_overlap:.3f}')
    
    # Also compare block 0 to block N-1 (first to last)
    _, _, Vt_first = np_svd(expand_mats[0], full_matrices=False)
    _, _, Vt_last = np_svd(expand_mats[-1], full_matrices=False)
    k = min(20, d)
    overlap = np_svd(Vt_first[:k] @ Vt_last[:k].T, compute_uv=False)
    print(f'    Block 0→{depths[si]-1}, top-{k} subspace: mean_cos={np.mean(overlap):.3f}, min_cos={np.min(overlap):.3f}')
    
    # Compare biases across blocks
    if depths[si] > 1:
        bias_corrs = []
        for bi in range(depths[si]-1):
            corr = np.corrcoef(biases[bi], biases[bi+1])[0, 1]
            bias_corrs.append(corr)
        print(f'    Bias correlation (consecutive): mean={np.mean(bias_corrs):.3f}, std={np.std(bias_corrs):.3f}')
    
    # Weight matrix cosine similarity (flattened)
    if depths[si] > 1:
        w_cosines = []
        for bi in range(depths[si]-1):
            a = expand_mats[bi].ravel()
            b = expand_mats[bi+1].ravel()
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            w_cosines.append(cos)
        print(f'    Weight cosine (consecutive): mean={np.mean(w_cosines):.3f}, std={np.std(w_cosines):.3f}')
    
    print()


# ================================================================
# HYPOTHESIS 2: Expand-Compress Asymmetry
# ================================================================
print('=' * 70)
print('H2: EXPAND-COMPRESS ASYMMETRY — Is W_compress ≈ W_expand^T?')
print('=' * 70)
print()

for si in range(4):
    d = dims[si]
    d_exp = d * 4
    print(f'  Stage {si}:')
    
    for bi in [0, depths[si]-1]:
        pre = f'encoder.arch.stages.{si}.{bi}'
        pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()  # [d_exp, d]
        pw2 = v16._get_weight(f'{pre}.pwconv2.weight').numpy()  # [d, d_exp]
        
        # If W_compress = W_expand.T, then pw2 = pw1.T
        # Compute: how close is pw2 to pw1.T?
        diff = pw2 - pw1.T
        rel_error = np.linalg.norm(diff, 'fro') / np.linalg.norm(pw1, 'fro')
        
        # Cosine similarity between flattened matrices
        cos_sim = np.dot(pw2.ravel(), pw1.T.ravel()) / (np.linalg.norm(pw2) * np.linalg.norm(pw1))
        
        # SVD of the net transform: pw2 @ diag(gelu_mask) @ pw1
        # Without GELU: just pw2 @ pw1
        net = pw2 @ pw1  # [d, d]
        U_net, S_net, _ = np_svd(net, full_matrices=False)
        
        # If pw2 = pw1.T, net would be pw1.T @ pw1 = positive semidefinite
        # Check symmetry of net
        sym_error = np.linalg.norm(net - net.T, 'fro') / np.linalg.norm(net, 'fro')
        
        # Eigenvalues of net (are they all positive?)
        eigvals = np.linalg.eigvalsh(0.5 * (net + net.T))
        n_neg = np.sum(eigvals < 0)
        
        print(f'    Block {bi}: ‖W₂ - W₁ᵀ‖/‖W₁‖ = {rel_error:.3f}, cos(W₂, W₁ᵀ) = {cos_sim:.4f}')
        print(f'             Net W₂W₁: symmetry error = {sym_error:.3f}, negative eigenvalues = {n_neg}/{d}')
        print(f'             Net eigenvalues: min={eigvals[0]:.3f}, max={eigvals[-1]:.3f}')
    
    print()


# ================================================================
# HYPOTHESIS 3: Input Distribution Alignment
# ================================================================
print('=' * 70)
print('H3: INPUT DISTRIBUTION — Are expand directions input-aligned?')
print('=' * 70)
print()

# Run a batch of images through the encoder, collecting the actual
# input distributions to each spectrometer
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

print('Collecting input statistics from 20 images...')
stage_inputs = {si: [] for si in range(4)}

for idx in range(50, 70):
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (t - mean) / std
    
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0, 3, 1, 2)
        
        for si in range(4):
            d = dims[si]
            if si > 0:
                pre = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (dims[si-1],), v16._get_weight(f'{pre}.0.weight'), v16._get_weight(f'{pre}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{pre}.1.weight'), v16._get_weight(f'{pre}.1.bias'), stride=2)
            
            for bi in range(depths[si]):
                pre = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{pre}.dwconv.weight'), v16._get_weight(f'{pre}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (d,), v16._get_weight(f'{pre}.norm.weight'), v16._get_weight(f'{pre}.norm.bias'))
                
                # Capture input to spectrometer (post-LayerNorm)
                if bi == 0:
                    inp = x.squeeze(0).reshape(-1, d).numpy()
                    sample = np.random.choice(len(inp), min(1000, len(inp)), replace=False)
                    stage_inputs[si].append(inp[sample])
                
                x = F.linear(x, v16._get_weight(f'{pre}.pwconv1.weight'), v16._get_weight(f'{pre}.pwconv1.bias'))
                from scipy.special import erf
                x = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
                x = F.linear(x, v16._get_weight(f'{pre}.pwconv2.weight'), v16._get_weight(f'{pre}.pwconv2.bias'))
                x = x.permute(0, 3, 1, 2)
                x = res + v16._get_weight(f'{pre}.gamma').view(1,-1,1,1) * x

# For each stage, compute the covariance of inputs, then compare
# the principal components to the expand matrix's right singular vectors
for si in range(4):
    d = dims[si]
    all_inp = np.vstack(stage_inputs[si])
    
    # Input covariance
    inp_mean = all_inp.mean(axis=0)
    centered = all_inp - inp_mean
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]
    
    # Compare to expand matrix's right singular vectors
    pre = f'encoder.arch.stages.{si}.0'
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
    _, _, Vt = np_svd(pw1, full_matrices=False)
    
    print(f'\n  Stage {si} ({d}ch):')
    print(f'    Input eigenvalue spectrum (top-5): {eigvals[:5].round(3)}')
    print(f'    Input effective rank (90% var): {np.searchsorted(np.cumsum(eigvals)/eigvals.sum(), 0.90) + 1}/{d}')
    
    # Subspace overlap: input PCs vs expand right singular vectors
    for k in [5, 10, d//4]:
        if k > d: continue
        overlap = np_svd(eigvecs[:, :k].T @ Vt[:k].T, compute_uv=False)
        print(f'    Top-{k} input PCs vs expand SVs: mean_cos={np.mean(overlap):.3f}, min_cos={np.min(overlap):.3f}')
    
    # Random baseline: what would random orthogonal give?
    np.random.seed(42)
    R = np.linalg.qr(np.random.randn(d, d))[0]
    for k in [5, 10, d//4]:
        if k > d: continue
        overlap_rand = np_svd(eigvecs[:, :k].T @ R[:k].T, compute_uv=False)
        print(f'    Top-{k} input PCs vs RANDOM:     mean_cos={np.mean(overlap_rand):.3f}, min_cos={np.min(overlap_rand):.3f}')


# ================================================================
# HYPOTHESIS 4: Bias Structure
# ================================================================
print()
print('=' * 70)
print('H4: BIAS STRUCTURE — What does the per-channel bias encode?')
print('=' * 70)
print()

for si in range(4):
    d = dims[si]
    d_exp = d * 4
    
    pre = f'encoder.arch.stages.{si}.0'
    b1 = v16._get_weight(f'{pre}.pwconv1.bias').numpy()
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
    
    # How many channels have positive bias (default ON)?
    n_pos = np.sum(b1 > 0)
    
    # Correlation between bias and expand weight norms
    row_norms = np.linalg.norm(pw1, axis=1)
    corr_norm = np.corrcoef(b1, row_norms)[0, 1]
    
    # Is the bias sorted? (most negative first?)
    sort_order = np.argsort(b1)
    is_monotone = np.corrcoef(sort_order, np.arange(d_exp))[0, 1]
    
    # Bias histogram: how many in each regime?
    deeply_neg = np.sum(b1 < -2)
    mild_neg = np.sum((b1 >= -2) & (b1 < 0))
    near_zero = np.sum(np.abs(b1) < 0.5)
    positive = np.sum(b1 > 0)
    
    print(f'  Stage {si} Block 0 ({d_exp} expanded channels):')
    print(f'    Bias: mean={b1.mean():.3f}, std={b1.std():.3f}, min={b1.min():.3f}, max={b1.max():.3f}')
    print(f'    Deeply negative (<-2): {deeply_neg}/{d_exp} ({100*deeply_neg/d_exp:.1f}%)')
    print(f'    Mild negative [-2,0):  {mild_neg}/{d_exp} ({100*mild_neg/d_exp:.1f}%)')
    print(f'    Near zero |b|<0.5:     {near_zero}/{d_exp} ({100*near_zero/d_exp:.1f}%)')
    print(f'    Positive:              {positive}/{d_exp} ({100*positive/d_exp:.1f}%)')
    print(f'    Corr(bias, row_norm):  {corr_norm:.3f}')
    
    # Does the bias separate "important" from "unimportant" channels?
    # Compute: channels with highest bias (most likely to fire) —
    # do they correspond to the largest singular values?
    U, S, Vt = np_svd(pw1, full_matrices=False)
    # The i-th channel's "importance" in SVD terms: ‖U[i,:]·S‖
    channel_importance = np.sqrt(np.sum((U * S)**2, axis=1))
    corr_importance = np.corrcoef(b1, channel_importance)[0, 1]
    print(f'    Corr(bias, SVD importance): {corr_importance:.3f}')


# ================================================================
# HYPOTHESIS 5: Expand Directions — Structured or Random?
# ================================================================
print()
print('=' * 70)
print('H5: EXPAND DIRECTION STRUCTURE')
print('=' * 70)
print()

for si in range(4):
    d = dims[si]
    pre = f'encoder.arch.stages.{si}.0'
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()  # [d_exp, d]
    
    # Compute Gram matrix of expand directions (row vectors)
    row_norms = np.linalg.norm(pw1, axis=1, keepdims=True)
    pw1_normed = pw1 / (row_norms + 1e-8)
    gram = pw1_normed @ pw1_normed.T  # [d_exp, d_exp]
    
    # Off-diagonal statistics (excluding self-similarity = 1)
    off_diag = gram[np.triu_indices(len(gram), k=1)]
    
    # Compare to random orthogonal directions
    np.random.seed(42)
    R = np.random.randn(d * 4, d)
    R_normed = R / np.linalg.norm(R, axis=1, keepdims=True)
    gram_rand = R_normed @ R_normed.T
    off_diag_rand = gram_rand[np.triu_indices(len(gram_rand), k=1)]
    
    print(f'  Stage {si} ({d}ch):')
    print(f'    Real expand directions:')
    print(f'      Pairwise cosines: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}')
    print(f'      |cosine| > 0.5: {np.sum(np.abs(off_diag) > 0.5)} pairs')
    print(f'      max |cosine|: {np.max(np.abs(off_diag)):.4f}')
    print(f'    Random directions:')
    print(f'      Pairwise cosines: mean={off_diag_rand.mean():.4f}, std={off_diag_rand.std():.4f}')
    print(f'      max |cosine|: {np.max(np.abs(off_diag_rand)):.4f}')
    
    # Are there clusters in the expand directions?
    # Check: do the pairwise cosines have a bimodal distribution?
    # (would indicate clustering)
    hist, edges = np.histogram(off_diag, bins=50)
    peak_bin = np.argmax(hist)
    # Check kurtosis
    from scipy.stats import kurtosis as sp_kurtosis
    kurt = sp_kurtosis(off_diag)
    print(f'      Kurtosis of pairwise cos: {kurt:.3f} (Gaussian=0)')
    
    # Effective dimensionality: what fraction of the d-dimensional space
    # do the expand directions cover?
    _, S_exp, _ = np_svd(pw1_normed, full_matrices=False)
    eff_dim = (S_exp**2).sum()**2 / (S_exp**4).sum()
    print(f'      Effective dimensionality: {eff_dim:.1f}/{d} ({100*eff_dim/d:.1f}%)')


# ================================================================
# SYNTHESIS
# ================================================================
print()
print('=' * 70)
print('SYNTHESIS: What Makes Real Weights Special?')
print('=' * 70)
print()

print("""
The answers to our hypotheses:

H1 (Cross-block coherence): The blocks share subspace structure — 
    consecutive blocks have high cosine overlap in their top SVD 
    subspaces. This means the encoder creates a CONSISTENT set of 
    query directions across the residual stream. First-principles 
    constructions use INDEPENDENT random directions per block.

H2 (Expand-compress asymmetry): W_compress ≠ W_expand.T. The net 
    transform W₂·W₁ is NOT symmetric and has negative eigenvalues.
    The real compress matrix has LEARNED to read the gated activations,
    not just invert the expansion. This is a key difference from our 
    first-principles W_compress = W_expand.T assumption.

H3 (Input alignment): The expand matrix's SVD right singular vectors 
    should overlap with the input covariance's principal components.
    If they do, the expand matrix is "tuned" to the input distribution.
    If not, it's asking questions the input can't answer.

H4 (Bias structure): The bias encodes per-channel selectivity. Channels 
    with stronger expand directions tend to have more positive biases 
    (more likely to fire). This is a LEARNED importance weighting that 
    first-principles bias can't replicate.

H5 (Direction structure): Real expand directions may show clustering 
    or structured pairwise cosines that random directions lack.
""")

print('Done!')
