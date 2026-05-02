"""
SSM φ-Attractor: Force trained weights toward the φ-manifold

Instead of building from scratch (losing 18.6%), take trained weights
and ATTRACT them toward φ-structure. The idea:

  TRAINED WEIGHTS = φ-SCAFFOLD + LEARNED CONTENT

If we can separate these, we can:
1. Verify the scaffold IS φ-structured
2. Quantify exactly what the "content" is
3. Find the minimum perturbation to impose φ-structure

Approaches:
A. Keep learned U,V directions, replace S with φ-decay
B. Interpolate: S_new = (1-α)·S_real + α·S_φ  (sweep α)
C. Null-space scaling: keep null(W₁) content but scale it
D. Per-stage attractor strength (some stages need more/less attraction)

The boom hypothesis: if there's a phase transition in α, that's the
"boom" — the point where continuous (learned) snaps to discrete (φ).
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from numpy.linalg import lstsq

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=' * 70)
print('SSM φ-ATTRACTOR: Trained Weights → φ-Manifold')
print('=' * 70)

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


# ================================================================
# INFRASTRUCTURE
# ================================================================

def run_encoder_custom(v16, img_tensor, weight_mutations=None, gate_fn=None):
    if weight_mutations is None:
        weight_mutations = {}
    if gate_fn is None:
        gate_fn = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    features = []
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
                pw1_w = weight_mutations.get(f'{pre}.pwconv1.weight', v16._get_weight(f'{pre}.pwconv1.weight'))
                pw1_b = weight_mutations.get(f'{pre}.pwconv1.bias', v16._get_weight(f'{pre}.pwconv1.bias'))
                pw2_w = weight_mutations.get(f'{pre}.pwconv2.weight', v16._get_weight(f'{pre}.pwconv2.weight'))
                pw2_b = weight_mutations.get(f'{pre}.pwconv2.bias', v16._get_weight(f'{pre}.pwconv2.bias'))
                x = F.linear(x, pw1_w, pw1_b)
                x = gate_fn(x)
                x = F.linear(x, pw2_w, pw2_b)
                x = x.permute(0, 3, 1, 2)
                x = res + v16._get_weight(f'{pre}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0, 2, 3, 1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'), v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0, 3, 1, 2))
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()


print('\nBuilding color basis...')
train_indices = list(range(50, 70))
all_enc = []
all_gt = []
for idx in train_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    if np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean() < 2: continue
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = run_encoder_custom(v16, t)
    flat = enc.reshape(256, -1).T
    sample = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    all_enc.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc_arr = np.vstack(all_enc)
all_gt_arr = np.vstack(all_gt)
enc_mean = all_enc_arr.mean(axis=0)
C = (all_enc_arr - enc_mean).T @ all_gt_arr / len(all_enc_arr)
U_color, S_color, _ = np.linalg.svd(C, full_matrices=False)
color_dir_1, color_dir_2 = U_color[:, 0], U_color[:, 1]
proj_1 = (all_enc_arr - enc_mean) @ color_dir_1
proj_2 = (all_enc_arr - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt_arr[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt_arr[:, 1], rcond=None)


def evaluate(weight_mutations=None, gate_fn=None, n_images=15):
    if gate_fn is None:
        gate_fn = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    gaps = []
    for idx in range(80, 300):
        if len(gaps) >= n_images: break
        if idx >= len(all_imgs): break
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
        err_z = np.sqrt(np.mean(ab_gt**2))
        if err_z < 2: continue
        gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        enc = run_encoder_custom(v16, t, weight_mutations, gate_fn)
        flat = (enc.reshape(256, -1).T - enc_mean)
        fields = np.column_stack([flat @ color_dir_1, flat @ color_dir_2, np.ones(SZ*SZ)])
        ab_pred = np.stack([
            np.clip(fields @ W_a, -50, 50).reshape(SZ, SZ),
            np.clip(fields @ W_b, -50, 50).reshape(SZ, SZ)
        ], axis=2)
        err = np.sqrt(np.mean((ab_pred - ab_gt)**2))
        gaps.append((1 - err/err_z) * 100)
    return np.mean(gaps), np.std(gaps)


# ================================================================
# PART 0: How close are real singular values to φ-decay already?
# ================================================================
print()
print('=' * 70)
print('PART 0: Are real singular values ALREADY φ-structured?')
print('=' * 70)
print()

for si in [0, 1, 2, 3]:
    d = dims[si]
    d_exp = d * 4
    pre = f'encoder.arch.stages.{si}.0'
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
    _, S_real, _ = np.linalg.svd(pw1, full_matrices=False)
    
    k = len(S_real)
    S_phi = np.array([S_real[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
    
    # Fit: S_real[i] ≈ A · φ^(-i·β/k) 
    # log(S_real[i]) = log(A) - i·β·log(φ)/k
    log_S = np.log(S_real + 1e-10)
    idx_arr = np.arange(k)
    # Linear fit: log(S) = a + b·i
    coeffs = np.polyfit(idx_arr, log_S, 1)
    # b = -β·log(φ)/k → β = -b·k/log(φ)
    beta_fit = -coeffs[0] * k / np.log(PHI)
    
    residual = np.linalg.norm(S_real - S_phi) / np.linalg.norm(S_real)
    
    # Also check: does S_real follow power law? S[i] ∝ i^(-α)
    log_i = np.log(idx_arr[1:] + 1)
    log_s = np.log(S_real[1:])
    alpha_fit = -np.polyfit(log_i, log_s, 1)[0]
    
    print(f'  Stage {si} ({d}ch):')
    print(f'    S_real top-5: {S_real[:5].round(3)}')
    print(f'    S_φ   top-5: {S_phi[:5].round(3)}')
    print(f'    ‖S_real - S_φ‖/‖S_real‖ = {residual:.3f} ({100*residual:.1f}% deviation)')
    print(f'    Best-fit φ-exponent β = {beta_fit:.3f} (theoretical: 0.5)')
    print(f'    Power law α = {alpha_fit:.3f} (Zipf: 1.0, 1/φ: 0.618)')
    print(f'    S[0]/S[1] = {S_real[0]/S_real[1]:.3f} (φ = {PHI:.3f})')
    print()


# ================================================================
# PART 1: ATTRACTOR — Replace S with φ-decay, keep U,V
# ================================================================
print('=' * 70)
print('PART 1: φ-ATTRACTOR — Keep learned directions, impose φ-structure')
print('=' * 70)
print()

def make_attractor_mutations(alpha, mode='sv_only'):
    """
    Attract trained weights toward φ-manifold.
    
    alpha: attraction strength (0 = pure real, 1 = pure φ)
    mode: 
      'sv_only' — interpolate singular values only
      'sv_and_compress' — also attract W₂ toward pinv(W₁_new)
      'sv_compress_bias' — also attract bias toward φ-structure
    """
    muts = {}
    
    for si in range(4):
        d = dims[si]
        d_exp = d * 4
        for bi in range(depths[si]):
            prefix = f'encoder.arch.stages.{si}.{bi}'
            
            pw1_real = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
            pw2_real = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
            b1_real = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
            
            # SVD of W₁
            U1, S1, Vt1 = np.linalg.svd(pw1_real, full_matrices=False)
            k = len(S1)
            
            # φ-structured singular values (matching scale of real)
            S_phi = np.array([S1[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
            
            # Interpolate: S_new = (1-α)·S_real + α·S_φ
            S_new = (1 - alpha) * S1 + alpha * S_phi
            
            # Reconstruct W₁ with new singular values, old directions
            pw1_new = (U1 * S_new) @ Vt1
            
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(pw1_new).float()
            muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b1_real).float()
            
            if mode in ('sv_and_compress', 'sv_compress_bias'):
                # Attract W₂ toward pinv(W₁_new)
                # pinv(W₁_new) = Vt1.T · diag(1/S_new) · U1.T
                pinv_new = (Vt1.T * (1.0 / (S_new + 1e-6))) @ U1.T  # [d, d_exp]
                
                # Scale pinv to match real W₂ norm
                pinv_scaled = pinv_new * (np.linalg.norm(pw2_real) / np.linalg.norm(pinv_new))
                
                # Interpolate W₂
                pw2_new = (1 - alpha) * pw2_real + alpha * pinv_scaled
                muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(pw2_new).float()
            else:
                muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(pw2_real).float()
            
            if mode == 'sv_compress_bias':
                # Attract bias toward stage mean
                bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
                b_target = np.full(d_exp, bias_means[si])
                b_new = (1 - alpha) * b1_real + alpha * b_target
                muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b_new).float()
    
    return muts


# Baseline
print('Computing baseline...')
mean_base, std_base = evaluate()
print(f'  Full encoder: {mean_base:+.1f}% ± {std_base:.1f}%')
print()

# Dense sweep of attraction strength
print('Sweeping attraction strength α (sv_only mode):')
print(f'  {"α":<8} {"Gap%":<10} {"Δ from real":<15} {"Note"}')
print('-' * 55)

alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sv_results = {}

for a in alphas:
    muts = make_attractor_mutations(a, mode='sv_only')
    mg, sg = evaluate(muts)
    sv_results[a] = mg
    delta = mg - mean_base
    note = ''
    if a == 0: note = '← pure real'
    elif a == 1.0: note = '← pure φ (learned dirs)'
    elif abs(a - 0.382) < 0.02: note = '← 1/φ²'
    elif abs(a - 0.618) < 0.02: note = '← 1/φ'
    print(f'  {a:<8.2f} {mg:+6.1f}%    {delta:+5.1f}%          {note}')

# Find the transition point (where does it start degrading?)
print()

# Detailed sweep near the transition
print('Fine sweep near transition:')
fine_alphas = np.arange(0.0, 0.35, 0.025)
for a in fine_alphas:
    if a in sv_results: continue
    muts = make_attractor_mutations(float(a), mode='sv_only')
    mg, _ = evaluate(muts)
    sv_results[float(a)] = mg
    delta = mg - mean_base
    print(f'  {a:<8.3f} {mg:+6.1f}%    {delta:+5.1f}%')


# ================================================================
# PART 2: FULL ATTRACTOR (SV + Compress + Bias)
# ================================================================
print()
print('=' * 70)
print('PART 2: FULL ATTRACTOR — SV + W₂ attraction + Bias attraction')
print('=' * 70)
print()

modes = ['sv_only', 'sv_and_compress', 'sv_compress_bias']
mode_names = {
    'sv_only': 'SV only (keep real W₂)',
    'sv_and_compress': 'SV + attract W₂→pinv',
    'sv_compress_bias': 'SV + W₂→pinv + bias→mean',
}

for a in [0.1, 0.3, 0.5, 1.0]:
    print(f'  α = {a}:')
    for mode in modes:
        muts = make_attractor_mutations(a, mode=mode)
        mg, sg = evaluate(muts)
        delta = mg - mean_base
        print(f'    {mode_names[mode]:<35} {mg:+6.1f}% (Δ={delta:+5.1f}%)')
    print()


# ================================================================
# PART 3: PER-STAGE ATTRACTION — Different α per stage
# ================================================================
print('=' * 70)
print('PART 3: PER-STAGE ATTRACTION — Which stages tolerate φ-structure?')
print('=' * 70)
print()

def make_per_stage_attractor(stage_alphas):
    """stage_alphas = {0: α₀, 1: α₁, 2: α₂, 3: α₃}"""
    muts = {}
    for si in range(4):
        alpha = stage_alphas.get(si, 0.0)
        d = dims[si]
        for bi in range(depths[si]):
            prefix = f'encoder.arch.stages.{si}.{bi}'
            pw1_real = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
            pw2_real = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
            b1_real = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
            
            U1, S1, Vt1 = np.linalg.svd(pw1_real, full_matrices=False)
            k = len(S1)
            S_phi = np.array([S1[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
            S_new = (1 - alpha) * S1 + alpha * S_phi
            pw1_new = (U1 * S_new) @ Vt1
            
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(pw1_new).float()
            muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(pw2_real).float()
            muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b1_real).float()
    return muts

# Test: fully attract each stage individually
print('  Attract one stage at a time (α=1.0):')
for target_si in range(4):
    alphas_dict = {si: (1.0 if si == target_si else 0.0) for si in range(4)}
    muts = make_per_stage_attractor(alphas_dict)
    mg, sg = evaluate(muts)
    delta = mg - mean_base
    print(f'    Stage {target_si} only: {mg:+6.1f}% (Δ={delta:+5.1f}%)')

print()

# Test: attract all EXCEPT one stage
print('  Attract all except one (α=1.0 for rest):')
for skip_si in range(4):
    alphas_dict = {si: (0.0 if si == skip_si else 1.0) for si in range(4)}
    muts = make_per_stage_attractor(alphas_dict)
    mg, sg = evaluate(muts)
    delta = mg - mean_base
    print(f'    Skip Stage {skip_si}: {mg:+6.1f}% (Δ={delta:+5.1f}%)')


# ================================================================
# PART 4: SINGULAR VALUE SPECTRUM — Is there a "boom"?
# ================================================================
print()
print('=' * 70)
print('PART 4: SINGULAR VALUE TRANSITION — Looking for the boom')
print('=' * 70)
print()

# For each stage, plot how performance changes as we sweep alpha
# If there's a boom, we'll see a sharp transition
for si in [0, 2]:
    d = dims[si]
    pre = f'encoder.arch.stages.{si}.0'
    pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
    _, S_real, _ = np.linalg.svd(pw1, full_matrices=False)
    k = len(S_real)
    S_phi = np.array([S_real[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
    
    # How do real and φ SVs differ?
    ratio = S_real / S_phi
    print(f'  Stage {si}: S_real/S_φ ratio')
    print(f'    Top modes (1-5):   {ratio[:5].round(3)}')
    print(f'    Mid modes:         {ratio[k//4:k//4+5].round(3)}')
    print(f'    Low modes:         {ratio[-5:].round(3)}')
    print(f'    Mean ratio: {ratio.mean():.3f}, Std: {ratio.std():.3f}')
    
    # Where does φ-decay diverge from real?
    # The "boom point" is where the ratio changes sharply
    diff_ratio = np.diff(ratio)
    boom_idx = np.argmax(np.abs(diff_ratio))
    print(f'    Max ratio change at mode {boom_idx}: Δ={diff_ratio[boom_idx]:.4f}')
    print(f'    Ratio at boom: {ratio[boom_idx]:.3f} → {ratio[boom_idx+1]:.3f}')
    print()


# ================================================================
# PART 5: QUANTIFY THE GAP — What exactly is in the 18.6%?
# ================================================================
print('=' * 70)
print('PART 5: QUANTIFYING THE GAP')
print('=' * 70)
print()

# The gap has 3 components:
# 1. Singular value structure (measured by attractor sweep)
# 2. Direction structure (U, V contain learned content)
# 3. Null-space injection (W₂ null content)

# Component 1: SV structure
sv_only_full = sv_results.get(1.0, None)
if sv_only_full is not None:
    print(f'  Component analysis:')
    print(f'    Full encoder:                        {mean_base:+6.1f}%')
    print(f'    Real dirs + real SVs (α=0):          {sv_results[0.0]:+6.1f}%')
    print(f'    Real dirs + φ SVs (α=1):             {sv_only_full:+6.1f}%')
    sv_cost = sv_results[0.0] - sv_only_full
    print(f'    → Cost of φ-structuring SVs:         {sv_cost:+5.1f}%')
    print()
    
    # The remaining gap after α=0 is from W₂ mutation (which is zero here)
    # So α=0 = pure real encoder performance
    print(f'  Gap decomposition:')
    print(f'    Total gap (full → φ-SVD+pinv):       ~18.6%')
    print(f'    From SVs alone (α=1, real dirs):     {sv_cost:.1f}%')
    print(f'    From directions (random vs learned):  ~{18.6 - sv_cost:.1f}% (estimated)')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()

# Sort results
sorted_sv = sorted(sv_results.items(), key=lambda x: x[0])
best_a = max(sv_results, key=sv_results.get)
print(f'Best α for SV-only attraction: {best_a} ({sv_results[best_a]:+.1f}%)')
print(f'Full encoder baseline: {mean_base:+.1f}%')
print(f'Gap at best α: {mean_base - sv_results[best_a]:.1f}%')

print('\nDone!')
