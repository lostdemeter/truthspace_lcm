"""
SSM Singular Value Spectrum: Finding the Correct Decay Law

The ENTIRE 18.8% gap is in the SV distribution. If we find the right
decay law, directions can be random and it still works.

Current knowledge:
- φ-decay S[i] = S[0]·φ^(-i·0.5/k) is WAY too flat
- Real SVs follow power law α ≈ 0.39-0.75 per stage  
- S[0]/S[1] ≈ φ for Stage 0

Questions:
1. What decay law fits the real spectrum? (power, exp, stretched exp, φ-based)
2. Is S[0]/S[1] ≈ φ universal across blocks?
3. Do consecutive SV ratios S[i]/S[i+1] have structure?
4. Is there a "boom" in the SV spectrum (sharp transition)?
5. Can we derive the correct spectrum from dimensional analysis?
"""
import numpy as np
import sys
import glob
from scipy.optimize import curve_fit

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=' * 70)
print('SSM SINGULAR VALUE SPECTRUM ANALYSIS')
print('=' * 70)

v16 = V16GeometricColorizer()


# ================================================================
# PART 1: FULL SV CATALOG — Every block
# ================================================================
print()
print('=' * 70)
print('PART 1: FULL SV CATALOG — All 18 blocks')
print('=' * 70)
print()

all_svs = {}  # (stage, block) → S array
all_ratios = {}  # (stage, block) → S[i]/S[i+1] array

for si in range(4):
    d = dims[si]
    for bi in range(depths[si]):
        pre = f'encoder.arch.stages.{si}.{bi}'
        pw1 = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
        _, S, _ = np.linalg.svd(pw1, full_matrices=False)
        all_svs[(si, bi)] = S
        ratios = S[:-1] / S[1:]
        all_ratios[(si, bi)] = ratios

# Print S[0]/S[1] for every block
print(f'  {"Block":<15} {"S[0]/S[1]":<10} {"S[1]/S[2]":<10} {"S[2]/S[3]":<10} {"Δ from φ":<10}')
print('-' * 60)
for si in range(4):
    for bi in range(depths[si]):
        r = all_ratios[(si, bi)]
        delta_phi = abs(r[0] - PHI) / PHI * 100
        print(f'  S{si}.B{bi:<10} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f} {delta_phi:5.1f}%')


# ================================================================
# PART 2: CONSECUTIVE RATIO ANALYSIS
# ================================================================
print()
print('=' * 70)
print('PART 2: CONSECUTIVE SV RATIOS — Is there structure?')
print('=' * 70)
print()

for si in [0, 1, 2, 3]:
    S = all_svs[(si, 0)]
    k = len(S)
    ratios = S[:-1] / S[1:]
    
    print(f'  Stage {si} Block 0 ({k} SVs):')
    print(f'    First 10 ratios:  {ratios[:10].round(4)}')
    print(f'    Mid ratios [{k//4}:{k//4+5}]: {ratios[k//4:k//4+5].round(4)}')
    print(f'    Last 5 ratios:    {ratios[-5:].round(4)}')
    print(f'    Mean ratio: {ratios.mean():.4f}, Std: {ratios.std():.4f}')
    print(f'    Ratio of ratios (r[0]/r[1]): {ratios[0]/ratios[1]:.4f}')
    
    # Check: does the ratio converge to something?
    # For a pure power law S[i] ∝ i^(-α), consecutive ratio = (i/(i+1))^α → 1
    # For exponential S[i] ∝ exp(-λi), consecutive ratio = exp(λ) = constant
    # For the real spectrum, it starts high and drops toward ~1
    
    # Fit an exponential to the ratio sequence
    # r[i] = a * exp(-b*i) + c
    try:
        def ratio_model(i, a, b, c):
            return a * np.exp(-b * i) + c
        i_arr = np.arange(len(ratios))
        popt, _ = curve_fit(ratio_model, i_arr, ratios, p0=[0.5, 0.1, 1.0], maxfev=10000)
        pred = ratio_model(i_arr, *popt)
        residual = np.sqrt(np.mean((ratios - pred)**2))
        print(f'    Ratio fit: r[i] = {popt[0]:.3f}·exp(-{popt[1]:.3f}·i) + {popt[2]:.3f} (RMSE={residual:.4f})')
        print(f'    Asymptotic ratio: {popt[2]:.4f} (log → decay rate = {-np.log(popt[2]):.4f})')
    except Exception as e:
        print(f'    Ratio fit failed: {e}')
    print()


# ================================================================
# PART 3: FIT DECAY MODELS
# ================================================================
print('=' * 70)
print('PART 3: FITTING DECAY MODELS TO THE SV SPECTRUM')
print('=' * 70)
print()

def power_law(i, A, alpha):
    return A * (i + 1.0) ** (-alpha)

def exponential(i, A, lam):
    return A * np.exp(-lam * i)

def stretched_exp(i, A, lam, beta):
    return A * np.exp(-lam * (i ** beta))

def phi_power(i, A, beta):
    return A * PHI ** (-beta * i)

def marcenko_pastur_edge(i, A, gamma, sigma):
    """Inspired by random matrix theory — MP distribution edge"""
    k = len(i) if hasattr(i, '__len__') else 1
    return A * (1 + np.sqrt(gamma)) * np.exp(-sigma * i)

def phi_ratio_decay(i, S0, r0, decay):
    """S[0] = S0, S[i+1] = S[i] / (r0 * exp(-decay*i) + 1)"""
    S = np.zeros_like(i, dtype=float)
    S[0] = S0
    for j in range(1, len(i)):
        ratio = r0 * np.exp(-decay * j) + 1.0
        S[j] = S[j-1] / ratio
    return S

models = {
    'Power law: A·(i+1)^(-α)': (power_law, [1.0, 0.5], 2),
    'Exponential: A·exp(-λi)': (exponential, [1.0, 0.01], 2),
    'Stretched exp: A·exp(-λ·i^β)': (stretched_exp, [1.0, 0.01, 0.5], 3),
    'φ-power: A·φ^(-βi)': (phi_power, [1.0, 0.01], 2),
}

for si in range(4):
    S = all_svs[(si, 0)]
    k = len(S)
    i_arr = np.arange(k, dtype=float)
    
    print(f'  Stage {si} ({k} SVs, S[0]={S[0]:.3f}, S[-1]={S[-1]:.3f}):')
    
    best_rmse = float('inf')
    best_model = None
    
    for name, (func, p0, nparams) in models.items():
        try:
            popt, _ = curve_fit(func, i_arr, S, p0=p0[:nparams], maxfev=50000)
            pred = func(i_arr, *popt)
            rmse = np.sqrt(np.mean((S - pred)**2))
            rel_err = rmse / np.mean(S) * 100
            
            params_str = ', '.join([f'{p:.4f}' for p in popt])
            print(f'    {name:<35} RMSE={rmse:.4f} ({rel_err:.1f}%)  [{params_str}]')
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = name
        except Exception as e:
            print(f'    {name:<35} FAILED: {e}')
    
    print(f'    → Best fit: {best_model} (RMSE={best_rmse:.4f})')
    
    # Also try: S[i] = S[0] · (1 + i/k)^(-α·k) — normalized power law
    try:
        def norm_power(i, alpha_k):
            return S[0] * (1 + i / k) ** (-alpha_k)
        popt_np, _ = curve_fit(norm_power, i_arr, S, p0=[1.0])
        pred_np = norm_power(i_arr, *popt_np)
        rmse_np = np.sqrt(np.mean((S - pred_np)**2))
        print(f'    Normalized power: S[0]·(1+i/k)^(-{popt_np[0]:.3f})  RMSE={rmse_np:.4f}')
    except:
        pass
    
    # Key test: does S[i] = S[0] · φ^(-i^β) work? (stretched φ)
    try:
        def stretched_phi(i, beta):
            return S[0] * PHI ** (-(i ** beta))
        popt_sp, _ = curve_fit(stretched_phi, i_arr[1:], S[1:], p0=[0.5])
        pred_sp = np.concatenate([[S[0]], stretched_phi(i_arr[1:], *popt_sp)])
        rmse_sp = np.sqrt(np.mean((S - pred_sp)**2))
        print(f'    Stretched φ: S[0]·φ^(-i^{popt_sp[0]:.4f})  RMSE={rmse_sp:.4f}')
    except:
        pass
    
    print()


# ================================================================
# PART 4: THE FIRST RATIO — Is S[0]/S[1] ≈ φ universal?
# ================================================================
print('=' * 70)
print('PART 4: THE FIRST RATIO S[0]/S[1] — φ connection')
print('=' * 70)
print()

first_ratios = []
for si in range(4):
    stage_ratios = []
    for bi in range(depths[si]):
        r = all_ratios[(si, bi)][0]
        stage_ratios.append(r)
        first_ratios.append((si, bi, r))
    
    mean_r = np.mean(stage_ratios)
    std_r = np.std(stage_ratios)
    delta_phi = abs(mean_r - PHI) / PHI * 100
    print(f'  Stage {si}: mean S[0]/S[1] = {mean_r:.4f} ± {std_r:.4f} (Δφ = {delta_phi:.1f}%)')

print()
all_r01 = [r for _, _, r in first_ratios]
print(f'  Overall: mean = {np.mean(all_r01):.4f}, std = {np.std(all_r01):.4f}')
print(f'  φ = {PHI:.4f}, Δ = {abs(np.mean(all_r01) - PHI):.4f} ({abs(np.mean(all_r01) - PHI)/PHI*100:.1f}%)')

# What about S[0]/S[2], S[0]/S[3]?
print()
print(f'  {"Block":<10} {"S[0]/S[1]":<10} {"S[0]/S[2]":<10} {"S[0]/S[3]":<10} {"S[0]/S[4]":<10}')
for si in [0, 2]:
    S = all_svs[(si, 0)]
    print(f'  S{si}.B0     {S[0]/S[1]:<10.4f} {S[0]/S[2]:<10.4f} {S[0]/S[3]:<10.4f} {S[0]/S[4]:<10.4f}')
    # Check: S[0]/S[k] ≈ φ^(k^β) for some β?
    for j in [1, 2, 3, 4, 5, 10, 20]:
        if j < len(S):
            ratio = S[0] / S[j]
            log_phi_ratio = np.log(ratio) / np.log(PHI)
            print(f'         S[0]/S[{j}] = {ratio:.4f} = φ^{log_phi_ratio:.3f}')


# ================================================================
# PART 5: SPECTRAL SHAPE — Normalized comparison across stages
# ================================================================
print()
print('=' * 70)
print('PART 5: NORMALIZED SPECTRAL SHAPE — Do all stages share a shape?')
print('=' * 70)
print()

# Normalize: S_norm[i] = S[i] / S[0], i_norm = i / (k-1)
for si in range(4):
    S = all_svs[(si, 0)]
    k = len(S)
    S_norm = S / S[0]
    i_norm = np.arange(k) / (k - 1)
    
    # Percentiles
    p25 = np.searchsorted(-S_norm, -S_norm[0] * 0.25) 
    p50 = np.searchsorted(-S_norm, -S_norm[0] * 0.50)
    p75 = np.searchsorted(-S_norm, -S_norm[0] * 0.75)
    
    # What fraction of S[0] is S at 25%, 50%, 75% of the way through?
    q25 = S_norm[k//4]
    q50 = S_norm[k//2]
    q75 = S_norm[3*k//4]
    
    print(f'  Stage {si} ({k} SVs):')
    print(f'    At 25% of modes: S/S[0] = {q25:.4f}')
    print(f'    At 50% of modes: S/S[0] = {q50:.4f}')
    print(f'    At 75% of modes: S/S[0] = {q75:.4f}')
    print(f'    At 100%:         S/S[0] = {S_norm[-1]:.4f}')
    
    # Effective rank ratios
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    r50 = np.searchsorted(cumvar, 0.50) + 1
    r80 = np.searchsorted(cumvar, 0.80) + 1
    r90 = np.searchsorted(cumvar, 0.90) + 1
    r99 = np.searchsorted(cumvar, 0.99) + 1
    print(f'    Rank50={r50}/{k} ({100*r50/k:.0f}%), Rank80={r80}/{k} ({100*r80/k:.0f}%), '
          f'Rank90={r90}/{k} ({100*r90/k:.0f}%), Rank99={r99}/{k} ({100*r99/k:.0f}%)')
    print()


# ================================================================
# PART 6: TEST THE CORRECT DECAY — What if we use REAL power law?
# ================================================================
print('=' * 70)
print('PART 6: TEST — Use real power law instead of φ-decay')
print('=' * 70)
print()

# From Part 3, we fit power law α for each stage. 
# Can we use that fitted power law as the SV spectrum?

import torch
import torch.nn.functional as F
import cv2
from numpy.linalg import lstsq

all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
SZ = 256

def run_encoder_custom(v16, img_tensor, weight_mutations=None):
    if weight_mutations is None:
        weight_mutations = {}
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

# Build color basis
print('Building color basis...')
all_enc, all_gt = [], []
for idx in range(50, 70):
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


def evaluate(weight_mutations=None, n_images=15):
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
        enc = run_encoder_custom(v16, t, weight_mutations)
        flat = (enc.reshape(256, -1).T - enc_mean)
        fields = np.column_stack([flat @ color_dir_1, flat @ color_dir_2, np.ones(SZ*SZ)])
        ab_pred = np.stack([
            np.clip(fields @ W_a, -50, 50).reshape(SZ, SZ),
            np.clip(fields @ W_b, -50, 50).reshape(SZ, SZ)
        ], axis=2)
        err = np.sqrt(np.mean((ab_pred - ab_gt)**2))
        gaps.append((1 - err/err_z) * 100)
    return np.mean(gaps), np.std(gaps)


def make_sv_attractor(sv_func, keep_real_w2=True):
    """Replace SVs using sv_func(S_real, stage_idx) → S_new, keep learned dirs."""
    muts = {}
    bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
    bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
    
    for si in range(4):
        d = dims[si]
        for bi in range(depths[si]):
            prefix = f'encoder.arch.stages.{si}.{bi}'
            pw1_real = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
            pw2_real = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
            b1_real = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
            
            U1, S1, Vt1 = np.linalg.svd(pw1_real, full_matrices=False)
            S_new = sv_func(S1, si)
            pw1_new = (U1 * S_new) @ Vt1
            
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(pw1_new).float()
            muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b1_real).float()
            
            if keep_real_w2:
                muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(pw2_real).float()
            else:
                # Adjust W₂ to compensate for SV change
                # New pinv: Vt1.T · diag(1/S_new) · U1.T
                pinv_new = (Vt1.T * (1.0 / (S_new + 1e-6))) @ U1.T
                pinv_scaled = pinv_new * (np.linalg.norm(pw2_real) / np.linalg.norm(pinv_new))
                # Mix: 70% real + 30% pinv_new (moderate attraction)
                pw2_mix = 0.7 * pw2_real + 0.3 * pinv_scaled
                muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(pw2_mix).float()
    
    return muts


# First: baseline and pure φ for comparison
print('\nBaseline evaluation...')
base_mean, base_std = evaluate()
print(f'  Full encoder: {base_mean:+.1f}% ± {base_std:.1f}%')

# Fitted power law parameters per stage (from Part 3)
# We'll fit them now
stage_alphas = {}
stage_S0 = {}
for si in range(4):
    S = all_svs[(si, 0)]
    k = len(S)
    i_arr = np.arange(k, dtype=float)
    try:
        popt, _ = curve_fit(power_law, i_arr, S, p0=[S[0], 0.5], maxfev=50000)
        stage_alphas[si] = popt[1]
        stage_S0[si] = popt[0]
    except:
        stage_alphas[si] = 0.5
        stage_S0[si] = S[0]

print(f'\nFitted power law exponents:')
for si in range(4):
    print(f'  Stage {si}: α = {stage_alphas[si]:.4f}, A = {stage_S0[si]:.3f}')

# Test various SV replacement strategies
print()
print(f'  {"Method":<50} {"Gap%":<10} {"Std":<8}')
print('-' * 70)

# 1. Keep real SVs (baseline via attractor α=0)
def sv_real(S, si): return S
mg, sg = evaluate(make_sv_attractor(sv_real))
print(f'  {"Keep real SVs (sanity check)":<50} {mg:+6.1f}%    {sg:.1f}%')

# 2. Old φ-decay 
def sv_phi_flat(S, si):
    k = len(S)
    return np.array([S[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
mg, sg = evaluate(make_sv_attractor(sv_phi_flat))
print(f'  {"φ-decay: S[0]·φ^(-i·0.5/k) (old, too flat)":<50} {mg:+6.1f}%    {sg:.1f}%')

# 3. Fitted power law per stage
def sv_power_law(S, si):
    k = len(S)
    i_arr = np.arange(k, dtype=float)
    return power_law(i_arr, stage_S0[si], stage_alphas[si])
mg, sg = evaluate(make_sv_attractor(sv_power_law))
print(f'  {"Fitted power law per stage":<50} {mg:+6.1f}%    {sg:.1f}%')

# 4. Power law with α = 1/φ for all stages
def sv_inv_phi_power(S, si):
    k = len(S)
    i_arr = np.arange(k, dtype=float)
    return power_law(i_arr, S[0], 1.0/PHI)
mg, sg = evaluate(make_sv_attractor(sv_inv_phi_power))
print(f'  {"Power law α=1/φ=0.618 (universal)":<50} {mg:+6.1f}%    {sg:.1f}%')

# 5. Steeper φ-decay: S[0]·φ^(-i·β/k) with β from fit
def sv_phi_steep(S, si):
    k = len(S)
    beta = stage_alphas[si] * np.log(k) / np.log(PHI)  # match overall decay
    return np.array([S[0] * PHI ** (-i * beta / k) for i in range(k)])
mg, sg = evaluate(make_sv_attractor(sv_phi_steep))
print(f'  {"Steep φ-decay: matched β per stage":<50} {mg:+6.1f}%    {sg:.1f}%')

# 6. Stretched exponential (φ-based)
def sv_stretched_phi(S, si):
    k = len(S)
    # S[i] = S[0] · φ^(-i^0.5) — stretched by √i
    return np.array([S[0] * PHI ** (-(i ** 0.5)) for i in range(k)])
mg, sg = evaluate(make_sv_attractor(sv_stretched_phi))
print(f'  {"Stretched φ: S[0]·φ^(-√i)":<50} {mg:+6.1f}%    {sg:.1f}%')

# 7. S[0]/S[1] = φ, then constant decay  
def sv_phi_first_then_flat(S, si):
    k = len(S)
    result = np.zeros(k)
    result[0] = S[0]
    result[1] = S[0] / PHI
    # Constant ratio after that
    ratio = (S[-1] / result[1]) ** (1.0 / (k-2))
    for i in range(2, k):
        result[i] = result[1] * ratio ** (i-1)
    return result
mg, sg = evaluate(make_sv_attractor(sv_phi_first_then_flat))
print(f'  {"φ first ratio + geometric tail":<50} {mg:+6.1f}%    {sg:.1f}%')

# 8. Exact real SV shape, but scale matched (this tests if SHAPE matters vs SCALE)
def sv_real_shape_prescaled(S, si):
    # Keep the exact shape but normalize so total energy matches φ-decay total energy
    k = len(S)
    S_phi = np.array([S[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
    # Match total energy: scale S so sum(S²) = sum(S_phi²)
    scale = np.sqrt(np.sum(S_phi**2) / np.sum(S**2))
    return S * scale
mg, sg = evaluate(make_sv_attractor(sv_real_shape_prescaled))
print(f'  {"Real shape, φ-matched energy":<50} {mg:+6.1f}%    {sg:.1f}%')

# 9. Also test with W₂ adjustment for the best methods
print()
print('  With W₂ attraction (30% toward pinv):')

mg, sg = evaluate(make_sv_attractor(sv_power_law, keep_real_w2=False))
print(f'  {"Fitted power law + W₂ adj":<50} {mg:+6.1f}%    {sg:.1f}%')

mg, sg = evaluate(make_sv_attractor(sv_phi_steep, keep_real_w2=False))
print(f'  {"Steep φ-decay + W₂ adj":<50} {mg:+6.1f}%    {sg:.1f}%')

mg, sg = evaluate(make_sv_attractor(sv_stretched_phi, keep_real_w2=False))
print(f'  {"Stretched φ + W₂ adj":<50} {mg:+6.1f}%    {sg:.1f}%')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('Done!')
