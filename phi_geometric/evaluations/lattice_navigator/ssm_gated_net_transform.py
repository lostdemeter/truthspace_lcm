"""
SSM Gated Net Transform: Understanding W_compress through the GELU lens

Key insight: W_compress doesn't invert W_expand — it reads GELU-gated activations.
The ACTUAL net transform is:
    T_actual = W_compress · diag(gelu_mask) · W_expand

Where gelu_mask is IMAGE-DEPENDENT (82-97% zeros).

This means W_compress only reads from the ~3-18% of surviving channels.
It doesn't need to be W_expand.T — it's a DIFFERENT learned mapping.

Questions:
1. What does the GATED net transform look like? (Per-image)
2. Does it change significantly across images?
3. What's its rank, eigenstructure?
4. Does the residual γ scaling interact with the gated transform?
5. Can we derive W_compress from W_expand + the GELU pattern?
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from scipy.special import erf

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=' * 70)
print('SSM GATED NET TRANSFORM ANALYSIS')
print('=' * 70)

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def gelu_grad(h):
    """GELU'(h) = Φ(h) + h·φ(h) where Φ=CDF, φ=PDF of N(0,1)"""
    phi_cdf = 0.5 * (1 + erf(h / np.sqrt(2)))
    phi_pdf = np.exp(-h**2 / 2) / np.sqrt(2 * np.pi)
    return phi_cdf + h * phi_pdf


# ================================================================
# PART 1: GATED NET TRANSFORM — Per-Image Structure
# ================================================================
print()
print('=' * 70)
print('PART 1: GATED NET TRANSFORM — T(x) = W₂ · diag(g\'(W₁x+b)) · W₁')
print('=' * 70)
print()

# For a few images, compute the per-pixel Jacobian of the SSM block
# J(x) = W₂ · diag(GELU'(W₁x+b)) · W₁
# This is a d×d matrix that depends on x

# Focus on Stage 0 Block 0 first
stage_transforms = {}

for target_stage in [0, 2]:
    print(f'  Stage {target_stage}:')
    d = dims[target_stage]
    d_exp = d * 4
    pre = f'encoder.arch.stages.{target_stage}.0'
    
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()  # [d_exp, d]
    pw2 = v16._get_weight(f'{pre}.pwconv2.weight').numpy()  # [d, d_exp]
    b1 = v16._get_weight(f'{pre}.pwconv1.bias').numpy()
    gamma = v16._get_weight(f'{pre}.gamma').numpy()
    
    all_jacobians = []
    all_gelu_masks = []
    
    for idx in range(50, 60):
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ))
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
            
            for si in range(target_stage + 1):
                dd = dims[si]
                if si > 0:
                    pre2 = f'encoder.arch.downsample_layers.{si}'
                    x = x.permute(0, 2, 3, 1)
                    x = F.layer_norm(x, (dims[si-1],), v16._get_weight(f'{pre2}.0.weight'), v16._get_weight(f'{pre2}.0.bias'))
                    x = x.permute(0, 3, 1, 2)
                    x = F.conv2d(x, v16._get_weight(f'{pre2}.1.weight'), v16._get_weight(f'{pre2}.1.bias'), stride=2)
                
                for bi in range(depths[si]):
                    pre2 = f'encoder.arch.stages.{si}.{bi}'
                    if si == target_stage and bi == 0:
                        # This is our target block — extract the input
                        res = x
                        x = F.conv2d(x, v16._get_weight(f'{pre2}.dwconv.weight'), v16._get_weight(f'{pre2}.dwconv.bias'), padding=3, groups=dd)
                        x = x.permute(0, 2, 3, 1)
                        x = F.layer_norm(x, (dd,), v16._get_weight(f'{pre2}.norm.weight'), v16._get_weight(f'{pre2}.norm.bias'))
                        
                        # x is now the input to the spectrometer [1, H, W, d]
                        inp = x.squeeze(0).numpy()  # [H, W, d]
                        H, W, _ = inp.shape
                        
                        # Compute pre-GELU: h = W₁·x + b
                        flat_inp = inp.reshape(-1, d)  # [HW, d]
                        h = flat_inp @ pw1.T + b1  # [HW, d_exp]
                        
                        # GELU gradient = diagonal gate
                        g_prime = gelu_grad(h)  # [HW, d_exp]
                        
                        # For a sample of pixels, compute the Jacobian
                        sample_idx = np.random.choice(len(flat_inp), min(200, len(flat_inp)), replace=False)
                        
                        for px in sample_idx[:20]:
                            # J = W₂ · diag(g'(h_px)) · W₁  [d, d]
                            gated = pw2 * g_prime[px]  # broadcast: [d, d_exp] * [d_exp]
                            J = gated @ pw1  # [d, d]
                            all_jacobians.append(J)
                        
                        # Average gating pattern
                        mean_gate = g_prime.mean(axis=0)
                        all_gelu_masks.append(mean_gate)
                        
                        # Continue with the forward pass
                        x_gated = torch.from_numpy(h * 0.5 * (1 + erf(h / np.sqrt(2)))).float()
                        x_gated = x_gated.reshape(H, W, -1).unsqueeze(0)  # [1, H, W, d_exp]
                        x = F.linear(x_gated, v16._get_weight(f'{pre2}.pwconv2.weight'), 
                                     v16._get_weight(f'{pre2}.pwconv2.bias'))
                        x = x.permute(0, 3, 1, 2)
                        x = res + v16._get_weight(f'{pre2}.gamma').view(1,-1,1,1) * x
                    else:
                        res = x
                        x = F.conv2d(x, v16._get_weight(f'{pre2}.dwconv.weight'), v16._get_weight(f'{pre2}.dwconv.bias'), padding=3, groups=dd)
                        x = x.permute(0, 2, 3, 1)
                        x = F.layer_norm(x, (dd,), v16._get_weight(f'{pre2}.norm.weight'), v16._get_weight(f'{pre2}.norm.bias'))
                        x = F.linear(x, v16._get_weight(f'{pre2}.pwconv1.weight'), v16._get_weight(f'{pre2}.pwconv1.bias'))
                        x = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
                        x = F.linear(x, v16._get_weight(f'{pre2}.pwconv2.weight'), v16._get_weight(f'{pre2}.pwconv2.bias'))
                        x = x.permute(0, 3, 1, 2)
                        x = res + v16._get_weight(f'{pre2}.gamma').view(1,-1,1,1) * x
    
    # Analyze the per-pixel Jacobians
    J_stack = np.array(all_jacobians)  # [N, d, d]
    
    # Average Jacobian
    J_mean = J_stack.mean(axis=0)
    
    # SVD of mean Jacobian
    U_j, S_j, Vt_j = np.linalg.svd(J_mean, full_matrices=False)
    
    print(f'    Mean Jacobian SVD top-5: {S_j[:5].round(3)}')
    print(f'    Mean Jacobian rank (90% var): {np.searchsorted(np.cumsum(S_j**2)/(S_j**2).sum(), 0.90) + 1}/{d}')
    
    # Symmetry of mean Jacobian
    sym_err = np.linalg.norm(J_mean - J_mean.T, 'fro') / np.linalg.norm(J_mean, 'fro')
    eigvals_j = np.linalg.eigvalsh(0.5 * (J_mean + J_mean.T))
    n_neg_j = np.sum(eigvals_j < 0)
    print(f'    Mean Jacobian symmetry error: {sym_err:.3f}')
    print(f'    Mean Jacobian neg eigenvalues: {n_neg_j}/{d}')
    print(f'    Mean Jacobian eigval range: [{eigvals_j[0]:.3f}, {eigvals_j[-1]:.3f}]')
    
    # How much do per-pixel Jacobians vary?
    J_centered = J_stack - J_mean
    var_per_element = np.mean(J_centered**2)
    mean_per_element = np.mean(J_mean**2)
    print(f'    Jacobian variance/mean ratio: {var_per_element/mean_per_element:.3f}')
    print(f'    (1.0 = varies as much as signal, 0.0 = stable)')
    
    # Cross-image Jacobian similarity
    if len(all_jacobians) > 40:
        cos_sims = []
        for i in range(0, 40, 2):
            a = all_jacobians[i].ravel()
            b = all_jacobians[i+1].ravel()
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cos_sims.append(cos)
        print(f'    Jacobian cosine (random pairs): mean={np.mean(cos_sims):.3f}, std={np.std(cos_sims):.3f}')
    
    # Average GELU gate pattern across images
    mean_mask = np.mean(all_gelu_masks, axis=0)
    print(f'    Avg gate pattern: active (>0.5): {np.sum(mean_mask > 0.5)}/{d_exp}')
    print(f'    Avg gate pattern: transition (0.1-0.5): {np.sum((mean_mask > 0.1) & (mean_mask <= 0.5))}/{d_exp}')
    print(f'    Avg gate pattern: dead (<0.1): {np.sum(mean_mask < 0.1)}/{d_exp}')
    
    # The γ (LayerScale) scaling
    print(f'    γ (LayerScale): mean={gamma.mean():.4f}, std={gamma.std():.4f}, min={gamma.min():.4f}, max={gamma.max():.4f}')
    print(f'    Effective scale: γ × ‖J_mean‖ = {gamma.mean() * np.linalg.norm(J_mean, 'fro'):.3f}')
    print()


# ================================================================
# PART 2: THE EFFECTIVE COMPRESS MATRIX
# ================================================================
print('=' * 70)
print('PART 2: THE EFFECTIVE COMPRESS MATRIX')
print('=' * 70)
print()

# For each stage, compute the "effective compress matrix" that W_compress
# actually uses. Since ~90% of channels are gated off, W_compress only
# uses ~10% of its columns.

for si in [0, 2, 3]:
    d = dims[si]
    d_exp = d * 4
    pre = f'encoder.arch.stages.{si}.0'
    
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
    pw2 = v16._get_weight(f'{pre}.pwconv2.weight').numpy()
    b1 = v16._get_weight(f'{pre}.pwconv1.bias').numpy()
    
    # Which channels are typically active? 
    # Use: channels with bias > -2 (likely to fire sometimes)
    likely_active = np.where(b1 > -2.0)[0]
    very_active = np.where(b1 > -0.5)[0]
    
    # The "effective compress" matrix: just the columns of W₂ for active channels
    W2_active = pw2[:, likely_active]  # [d, n_active]
    W1_active = pw1[likely_active, :]  # [n_active, d]
    
    # What's the effective net transform through active channels only?
    net_active = W2_active @ W1_active  # [d, d]
    net_full = pw2 @ pw1  # [d, d]
    
    # How much of the net transform comes from active channels?
    energy_active = np.linalg.norm(net_active, 'fro')
    energy_full = np.linalg.norm(net_full, 'fro')
    
    print(f'  Stage {si} ({d}ch):')
    print(f'    Likely active channels (bias > -2): {len(likely_active)}/{d_exp} ({100*len(likely_active)/d_exp:.0f}%)')
    print(f'    Very active channels (bias > -0.5): {len(very_active)}/{d_exp} ({100*len(very_active)/d_exp:.0f}%)')
    print(f'    ‖Net_active‖/‖Net_full‖ = {energy_active/energy_full:.3f}')
    
    # SVD of the active sub-transform
    _, S_active, _ = np.linalg.svd(net_active, full_matrices=False)
    _, S_full, _ = np.linalg.svd(net_full, full_matrices=False)
    
    print(f'    Net_active SVD top-5: {S_active[:5].round(2)}')
    print(f'    Net_full   SVD top-5: {S_full[:5].round(2)}')
    print(f'    Net_active rank90: {np.searchsorted(np.cumsum(S_active**2)/(S_active**2).sum(), 0.90)+1}/{d}')
    print(f'    Net_full   rank90: {np.searchsorted(np.cumsum(S_full**2)/(S_full**2).sum(), 0.90)+1}/{d}')
    
    # Key question: is the ACTIVE sub-network closer to W₁.T?
    W1_active_pinv = np.linalg.pinv(W1_active)  # [d, n_active]
    cos_W2_pinv = np.dot(W2_active.ravel(), W1_active_pinv.ravel()) / \
                  (np.linalg.norm(W2_active) * np.linalg.norm(W1_active_pinv))
    cos_W2_W1T = np.dot(pw2.ravel(), pw1.T.ravel()) / \
                 (np.linalg.norm(pw2) * np.linalg.norm(pw1))
    
    print(f'    cos(W₂_active, pinv(W₁_active)): {cos_W2_pinv:.4f}')
    print(f'    cos(W₂_full, W₁ᵀ_full):          {cos_W2_W1T:.4f}')
    
    # What if W_compress reads in the LEFT singular space of W_expand?
    U1, S1, Vt1 = np.linalg.svd(pw1, full_matrices=False)  # U1: [d_exp, d]
    U2, S2, Vt2 = np.linalg.svd(pw2, full_matrices=False)  # U2: [d, d_exp]... wait
    # pw2 is [d, d_exp], so SVD gives U2:[d,d], S2:[d], Vt2:[d, d_exp]
    # U1 are the LEFT singular vectors of pw1 = expand directions in expanded space
    # Vt2 are the RIGHT singular vectors of pw2 = compress directions in expanded space
    
    # Are the LEFT SVDs of W_expand aligned with RIGHT SVDs of W_compress?
    # U1[:, :k] vs Vt2[:k, :].T
    for k in [5, 10, d//4]:
        if k > d: continue
        overlap = np.linalg.svd(U1[:, :k].T @ Vt2[:k, :].T, compute_uv=False)
        print(f'    top-{k} expand LEFT SVs vs compress RIGHT SVs: mean_cos={np.mean(overlap):.3f}')
    
    print()


# ================================================================
# PART 3: THE RESIDUAL INTERPRETATION
# ================================================================
print('=' * 70)
print('PART 3: THE RESIDUAL INTERPRETATION')
print('=' * 70)
print()

print("""
The spectrometer output goes through a RESIDUAL connection:
  x_out = x_in + γ · (W₂ · GELU(W₁·x_ln + b))

Where x_ln = LayerNorm(DWConv(x_in))

So the actual function is: IDENTITY + small perturbation
  γ is typically 1e-6 to 1e-2 (very small!)

This means:
  - The spectrometer doesn't need to reconstruct x — the residual does that
  - W₂ only needs to produce a small CORRECTION
  - This correction encodes new information from the GELU-selected channels
  - W_compress ≠ W_expand.T makes perfect sense: it's not reconstructing,
    it's producing a targeted correction vector
""")

# Verify γ scales
for si in range(4):
    for bi in [0]:
        pre = f'encoder.arch.stages.{si}.{bi}'
        gamma = v16._get_weight(f'{pre}.gamma').numpy()
        print(f'  Stage {si} Block {bi}: γ range [{gamma.min():.6f}, {gamma.max():.6f}], mean={gamma.mean():.6f}')

# What fraction of the output comes from residual vs spectrometer?
print()
print('  Residual vs Spectrometer contribution (10 images):')

for target_si in [0, 2]:
    d = dims[target_si]
    residual_norms = []
    spec_norms = []
    
    for idx in range(50, 60):
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (t - mean) / std_t
        
        with torch.no_grad():
            x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (96,),
                             v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                             v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
            x = x.permute(0, 3, 1, 2)
            
            for si in range(target_si + 1):
                dd = dims[si]
                if si > 0:
                    pre2 = f'encoder.arch.downsample_layers.{si}'
                    x = x.permute(0, 2, 3, 1)
                    x = F.layer_norm(x, (dims[si-1],), v16._get_weight(f'{pre2}.0.weight'), v16._get_weight(f'{pre2}.0.bias'))
                    x = x.permute(0, 3, 1, 2)
                    x = F.conv2d(x, v16._get_weight(f'{pre2}.1.weight'), v16._get_weight(f'{pre2}.1.bias'), stride=2)
                
                for bi in range(depths[si]):
                    pre2 = f'encoder.arch.stages.{si}.{bi}'
                    res = x
                    x = F.conv2d(x, v16._get_weight(f'{pre2}.dwconv.weight'), v16._get_weight(f'{pre2}.dwconv.bias'), padding=3, groups=dd)
                    x = x.permute(0, 2, 3, 1)
                    x = F.layer_norm(x, (dd,), v16._get_weight(f'{pre2}.norm.weight'), v16._get_weight(f'{pre2}.norm.bias'))
                    x = F.linear(x, v16._get_weight(f'{pre2}.pwconv1.weight'), v16._get_weight(f'{pre2}.pwconv1.bias'))
                    x = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
                    x = F.linear(x, v16._get_weight(f'{pre2}.pwconv2.weight'), v16._get_weight(f'{pre2}.pwconv2.bias'))
                    x = x.permute(0, 3, 1, 2)
                    
                    spec_out = v16._get_weight(f'{pre2}.gamma').view(1,-1,1,1) * x
                    x = res + spec_out
                    
                    if si == target_si and bi == 0:
                        residual_norms.append(torch.norm(res).item())
                        spec_norms.append(torch.norm(spec_out).item())
    
    mean_res = np.mean(residual_norms)
    mean_spec = np.mean(spec_norms)
    print(f'    Stage {target_si}: ‖residual‖={mean_res:.1f}, ‖spectrometer‖={mean_spec:.1f}, '
          f'ratio={mean_spec/mean_res:.4f} ({mean_spec/mean_res*100:.2f}%)')


# ================================================================
# PART 4: CAN WE DERIVE W_COMPRESS?
# ================================================================
print()
print('=' * 70)
print('PART 4: CAN WE DERIVE W_COMPRESS FROM W_EXPAND + GATING?')
print('=' * 70)
print()

# The residual insight changes everything:
# W₂ doesn't need to reconstruct the input.
# It needs to produce a small, targeted correction.
# 
# What if W₂ is optimized to produce the correction that, when added to
# the residual, maximally improves representation quality?
#
# Hypothesis: W₂ is the solution to:
#   min_W₂ Σᵢ ‖target_correction_i - W₂ · gelu(W₁·x_i + b)‖²
#
# Where target_correction comes from training. We don't have access to
# the training data, but we can check structural properties.

# For each stage, decompose W₂ in terms of W₁:
# W₂ = A · pinv(W₁) + B · null(W₁)
# where A captures the "pseudoinverse-like" part
# and B captures the "null space" part

for si in [0, 2]:
    d = dims[si]
    d_exp = d * 4
    pre = f'encoder.arch.stages.{si}.0'
    
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
    pw2 = v16._get_weight(f'{pre}.pwconv2.weight').numpy()
    
    # pw2 is [d, d_exp], pw1 is [d_exp, d]
    # The pseudoinverse of pw1 is [d, d_exp]
    pw1_pinv = np.linalg.pinv(pw1)  # [d, d_exp]
    
    # Project pw2 onto the range and null space of pw1.T
    # pw1.T is [d, d_exp] — same shape as pw2
    # The range of pw1.T = column space of pw1.T = row space of pw1
    
    # Using SVD of pw1:
    U1, S1, Vt1 = np.linalg.svd(pw1, full_matrices=True)
    # U1: [d_exp, d_exp], but only first d columns span range
    # pw1 = U1[:,:d] · diag(S1) · Vt1
    
    # Project each row of pw2 (which is in d_exp-space) onto:
    # range(pw1) = span of rows of pw1 = span of columns of Vt1[:d,:]
    # Wait no. pw1 is [d_exp, d]. Its column space is in R^{d_exp}
    # pw2 is [d, d_exp]. Each row of pw2 is in R^{d_exp}
    # We want to decompose each row of pw2 into:
    # - projection onto column space of pw1 (rank d)
    # - projection onto null space of pw1.T (rank d_exp - d)
    
    # Column space of pw1 is spanned by U1[:, :d]
    U_range = U1[:, :d]  # [d_exp, d]
    U_null = U1[:, d:]   # [d_exp, d_exp-d]
    
    # Project rows of pw2:
    pw2_in_range = pw2 @ U_range @ U_range.T  # [d, d_exp]
    pw2_in_null = pw2 @ U_null @ U_null.T     # [d, d_exp]
    
    energy_range = np.linalg.norm(pw2_in_range, 'fro')
    energy_null = np.linalg.norm(pw2_in_null, 'fro')
    energy_total = np.linalg.norm(pw2, 'fro')
    
    print(f'  Stage {si} ({d}ch):')
    print(f'    W₂ decomposition into W₁ column space:')
    print(f'      In range(W₁): {energy_range:.3f} ({100*energy_range**2/energy_total**2:.1f}%)')
    print(f'      In null(W₁ᵀ): {energy_null:.3f} ({100*energy_null**2/energy_total**2:.1f}%)')
    print(f'      Total:         {energy_total:.3f}')
    
    # If pw2 were pw1_pinv, it would be 100% in range
    # The null space component is "extra" — information W₂ injects
    # that W₁ CANNOT see
    
    # What does the null-space component look like?
    _, S_null, _ = np.linalg.svd(pw2_in_null, full_matrices=False)
    print(f'      Null-space SVD top-5: {S_null[:5].round(3)}')
    print(f'      Null-space rank90: {np.searchsorted(np.cumsum(S_null**2)/(S_null**2).sum(), 0.90)+1}/{min(d, d_exp-d)}')
    
    # The range-space component: is it close to pinv(W₁)?
    cos_range_pinv = np.dot(pw2_in_range.ravel(), pw1_pinv.ravel()) / \
                     (np.linalg.norm(pw2_in_range) * np.linalg.norm(pw1_pinv))
    print(f'      cos(W₂_range, pinv(W₁)): {cos_range_pinv:.4f}')
    
    # What fraction of W₂'s energy is "pseudoinverse-like"?
    pinv_energy = np.linalg.norm(pw1_pinv, 'fro')
    print(f'      ‖pinv(W₁)‖ = {pinv_energy:.3f}')
    print(f'      ‖W₂‖ = {energy_total:.3f}')
    print(f'      ‖W₂‖/‖pinv(W₁)‖ = {energy_total/pinv_energy:.3f}')
    print()


# ================================================================
# GRAND SYNTHESIS
# ================================================================
print('=' * 70)
print('GRAND SYNTHESIS')
print('=' * 70)
print()

print("""
THE SPECTROMETER AS RESIDUAL CORRECTOR:

The SSM block is NOT an autoencoder (encode→decode).
It's a RESIDUAL CORRECTION system:

  x_out = x_in + γ · correction(x_in)

Where correction = W₂ · GELU(W₁ · LayerNorm(DWConv(x_in)) + b)

Key structural properties:
  1. γ is tiny (1e-6 to 1e-2) — the correction is small
  2. W₂ is uncorrelated with W₁ᵀ — it's not inverting the expansion
  3. W₂ has significant null-space energy — it injects information 
     that W₁ cannot access
  4. The GELU gate selects ~3-18% of channels per pixel
  5. The gate pattern is image-dependent (the correction depends on content)

This explains EVERYTHING:
  - Why W_compress ≠ W_expand.T: it's not reconstructing, it's correcting
  - Why first-principles fail: the correction vectors are learned content
  - Why φ-structured SVD is the best first-principles: it preserves the
    ENERGY distribution even though the directions are wrong
  - Why the gate matters: it selects WHICH correction to apply

The spectrometer = a content-addressable lookup table of corrections:
  - W₁ rows = query vectors ("does the input have feature i?")
  - GELU = selection ("which features are present?")
  - W₂ columns = correction vectors ("if feature i is present, add this")
  - The sparse selection means only ~5% of corrections are applied per pixel

This is NOT a spectrometer anymore — it's a CONDITIONAL INJECTOR.
Each block injects a small, sparse, content-dependent correction
into the residual stream.
""")

print('Done!')
