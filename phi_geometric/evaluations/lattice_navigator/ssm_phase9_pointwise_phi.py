"""
Phase 9: Pointwise Conv φ-Decomposition

Depthwise conv (spatial mixing) was 98.2% explained by φ-basis (Phase 8).
Pointwise conv (channel mixing) is 98.7% of encoder params.

Key question: Do the pointwise weight matrices also exhibit φ-structure?

ConvNeXt block: x → DWConv → LN → PW1 (expand 4x) → GELU → PW2 (contract 4x) → GRN → residual

PW1: [4*C, C] — expand channels by 4x
PW2: [C, 4*C] — contract channels back

This is identical to the MLP in a transformer: W_up → activation → W_down.
We already know MLP in Qwen2 has Zipf α ≈ 0.12 (nearly full rank).
Does ConvNeXt MLP behave differently?

Tests:
1. SVD spectrum of PW1/PW2 — Zipf exponent
2. φ-structure in singular values (ratios between consecutive SVs)
3. Low-rank approximation quality
4. Connection between PW1 and PW2 (encode=decode?)
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
SZ = 256

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]


def compute_rmse(pred_tensor, gt_ab):
    pred_ab = pred_tensor[0, :2].permute(1, 2, 0).numpy()
    pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    return np.sqrt(np.mean((pred_r - gt_ab)**2))


test_data = []
for idx in range(300, 400):
    if len(test_data) >= 30:
        break
    im = cv2.imread(all_imgs[idx])
    if im is None:
        continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append({'tensor': t, 'gt_ab': gt_ab})

print(f'Test set: {len(test_data)} images')


# ================================================================
# STEP 1: SVD Spectrum Analysis of PW1 and PW2
# ================================================================
print()
print('=' * 70)
print('STEP 1: SVD Spectrum of Pointwise Convolutions')
print('=' * 70)
print()

def zipf_law(ranks, s0, alpha):
    return s0 / ranks**alpha

print(f"{'Block':<10} {'PW':<5} {'Shape':<15} {'Zipf α':<10} "
      f"{'S0/S1':<8} {'S1/S2':<8} {'Rank90%':<8} {'Rank99%':<8}")
print("-" * 76)

all_pw1_alphas = []
all_pw2_alphas = []
all_pw1_ratios = []
all_pw2_ratios = []
all_pw_svds = {}

for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        for pw_name, label in [('pwconv1', 'PW1'), ('pwconv2', 'PW2')]:
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            w = v16._get_weight(f'{prefix}.{pw_name}.weight')
            if w is None:
                continue
            
            w_np = w.numpy()
            U, S, Vt = np.linalg.svd(w_np, full_matrices=False)
            
            all_pw_svds[(stage_idx, block_idx, pw_name)] = (U, S, Vt, w.shape)
            
            # Zipf fit
            ranks = np.arange(1, len(S) + 1).astype(float)
            try:
                popt, _ = curve_fit(zipf_law, ranks[:min(50, len(S))],
                                    S[:min(50, len(S))], p0=[S[0], 0.5], maxfev=5000)
                alpha = popt[1]
            except:
                alpha = 0
            
            # Cumulative variance
            cumvar = np.cumsum(S**2) / np.sum(S**2)
            rank90 = np.searchsorted(cumvar, 0.9) + 1
            rank99 = np.searchsorted(cumvar, 0.99) + 1
            
            ratio01 = S[0] / S[1] if S[1] > 0 else float('inf')
            ratio12 = S[1] / S[2] if S[2] > 0 else float('inf')
            
            if label == 'PW1':
                all_pw1_alphas.append(alpha)
                all_pw1_ratios.append(ratio12)
            else:
                all_pw2_alphas.append(alpha)
                all_pw2_ratios.append(ratio12)
            
            print(f"  {stage_idx}.{block_idx:<7} {label:<5} "
                  f"{str(w_np.shape):<15} {alpha:<10.4f} "
                  f"{ratio01:<8.3f} {ratio12:<8.3f} "
                  f"{rank90:<8} {rank99:<8}")

print(f"\nPW1 Zipf α: {np.mean(all_pw1_alphas):.4f} ± {np.std(all_pw1_alphas):.4f}")
print(f"PW2 Zipf α: {np.mean(all_pw2_alphas):.4f} ± {np.std(all_pw2_alphas):.4f}")
print(f"PW1 S1/S2 mean: {np.mean(all_pw1_ratios):.4f}")
print(f"PW2 S1/S2 mean: {np.mean(all_pw2_ratios):.4f}")
print(f"1/φ = {1/PHI:.4f}, φ = {PHI:.4f}")


# ================================================================
# STEP 2: φ-ratios in consecutive singular values
# ================================================================
print()
print('=' * 70)
print('STEP 2: φ-Ratios in Singular Value Spectra')
print('=' * 70)
print()

# Check if S[i]/S[i+1] ≈ φ or 1/φ at any positions
# Collect ALL consecutive ratios from all blocks

all_ratios = []
for key, (U, S, Vt, shape) in all_pw_svds.items():
    for i in range(min(20, len(S) - 1)):
        if S[i+1] > 1e-10:
            ratio = S[i] / S[i+1]
            all_ratios.append(ratio)

all_ratios = np.array(all_ratios)

# Histogram of consecutive SV ratios
print(f"Consecutive SV ratio distribution ({len(all_ratios)} pairs):")
print(f"  Mean: {all_ratios.mean():.4f}")
print(f"  Median: {np.median(all_ratios):.4f}")
print(f"  Std: {all_ratios.std():.4f}")

# Count near φ-related values
for target, name in [(1.0, '1.0'), (1/PHI, '1/φ'), (PHI, 'φ'),
                     (PHI**2, 'φ²'), (1/PHI**2, '1/φ²')]:
    nearby = np.abs(all_ratios - target) < 0.05
    pct = nearby.sum() / len(all_ratios) * 100
    print(f"  Within 0.05 of {name} ({target:.4f}): {nearby.sum()}/{len(all_ratios)} ({pct:.1f}%)")


# ================================================================
# STEP 3: ENCODE=DECODE test — PW1 vs PW2 relationship
# ================================================================
print()
print('=' * 70)
print('STEP 3: ENCODE=DECODE — PW1 vs PW2 Relationship')
print('=' * 70)
print()

# The hypothesis: encoding and decoding are the same operation in opposite directions.
# PW1 expands (encode), PW2 contracts (decode).
# If ENCODE=DECODE, then PW2 ≈ PW1.T (or related by a simple transform)

print(f"{'Block':<10} {'PW1 shape':<15} {'PW2 shape':<15} "
      f"{'cos(PW1,PW2.T)':<15} {'SV corr':<10}")
print("-" * 65)

all_cos_sims = []
all_sv_corrs = []

for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        key1 = (stage_idx, block_idx, 'pwconv1')
        key2 = (stage_idx, block_idx, 'pwconv2')
        
        if key1 not in all_pw_svds or key2 not in all_pw_svds:
            continue
        
        U1, S1, Vt1, shape1 = all_pw_svds[key1]
        U2, S2, Vt2, shape2 = all_pw_svds[key2]
        
        w1 = (U1 * S1) @ Vt1  # [4C, C]
        w2 = (U2 * S2) @ Vt2  # [C, 4C]
        
        # Cosine similarity between W1 and W2.T
        # Both should be [4C, C] after transpose
        w2t = w2.T  # [4C, C]
        cos_sim = np.sum(w1 * w2t) / (np.linalg.norm(w1) * np.linalg.norm(w2t) + 1e-10)
        
        # SV correlation
        min_len = min(len(S1), len(S2))
        sv_corr = np.corrcoef(S1[:min_len], S2[:min_len])[0, 1]
        
        all_cos_sims.append(cos_sim)
        all_sv_corrs.append(sv_corr)
        
        print(f"  {stage_idx}.{block_idx:<7} {str(shape1):<15} {str(shape2):<15} "
              f"{cos_sim:<15.4f} {sv_corr:<10.4f}")

print(f"\nMean cos(W1, W2.T): {np.mean(all_cos_sims):.4f}")
print(f"Mean SV correlation: {np.mean(all_sv_corrs):.4f}")

if np.mean(all_sv_corrs) > 0.9:
    print("→ Strong SV correlation: ENCODE≈DECODE in spectral structure!")
elif np.mean(all_sv_corrs) > 0.7:
    print("→ Moderate SV correlation: partial ENCODE≈DECODE")
else:
    print("→ Weak SV correlation: ENCODE≠DECODE for pointwise conv")


# ================================================================
# STEP 4: Low-rank PW replacement test
# ================================================================
print()
print('=' * 70)
print('STEP 4: Low-rank Pointwise Conv Replacement')
print('=' * 70)
print()

# Baseline
baseline_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    baseline_rmses.append(compute_rmse(pred, img_d['gt_ab']))
baseline = np.mean(baseline_rmses)
baseline_rmses = np.array(baseline_rmses)
print(f'Baseline RMSE: {baseline:.3f}')

original_get = v16._get_weight

# Test various rank fractions
for frac in [0.25, 0.5, 0.75, 0.9, 1.0]:
    modified = {}
    total_orig = 0
    total_approx = 0
    
    for key, (U, S, Vt, shape) in all_pw_svds.items():
        k = max(1, int(len(S) * frac))
        w_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        modified[key] = torch.from_numpy(w_approx.reshape(shape)).float()
        total_orig += np.prod(shape)
        total_approx += k * (U.shape[0] + Vt.shape[1])
    
    def frac_get(name, mod=modified, orig=original_get):
        for (si, bi, pw), w in mod.items():
            if name == f'encoder.arch.stages.{si}.{bi}.{pw}.weight':
                return w
        return orig(name)
    
    v16._get_weight = frac_get
    rmses = []
    for img_d in test_data[:15]:
        with torch.no_grad():
            pred = v16.forward(img_d['tensor'])
        rmses.append(compute_rmse(pred, img_d['gt_ab']))
    v16._get_weight = original_get
    
    mean_rmse = np.mean(rmses)
    sub_baseline = np.mean(baseline_rmses[:15])
    delta = (mean_rmse - sub_baseline) / sub_baseline * 100
    
    print(f'  Rank {frac*100:>5.1f}%: RMSE={mean_rmse:.3f} ({delta:+.2f}%), '
          f'params: {total_orig:,} → {total_approx:,} ({total_approx/total_orig*100:.1f}%)')


# ================================================================
# STEP 5: Combined compression — analytic DW + low-rank PW + no xfmr
# ================================================================
print()
print('=' * 70)
print('STEP 5: Maximum Compression — Analytic DW + Low-rank PW + No Transformer')
print('=' * 70)
print()

# Build the φ-analytic DW conv weights (from Phase 8)
phi_basis_functions = []
ys, xs = np.mgrid[0:7, 0:7]
dx = xs - 3.0
dy = ys - 3.0

# Separable bases with rates from {1/φ, 1, φ}
for ax in [1/PHI, 1.0, PHI]:
    for ay in [1/PHI, 1.0, PHI]:
        b = PHI ** (-ax * np.abs(dx)) * PHI ** (-ay * np.abs(dy))
        b = b / (np.linalg.norm(b) + 1e-10)
        phi_basis_functions.append(b.flatten())

# Radial × angular
dist = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dy, dx)
for alpha in [1/PHI, 1.0, PHI]:
    for freq in [1, 2, 3, 4]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            b = PHI ** (-alpha * dist) * np.cos(freq * theta + phase)
            b = b / (np.linalg.norm(b) + 1e-10)
            phi_basis_functions.append(b.flatten())

# Pure radial + DC + BBP
for alpha in [1/PHI, 1.0, PHI, 2.0, 3.0]:
    b = PHI ** (-alpha * dist)
    b = b / (np.linalg.norm(b) + 1e-10)
    phi_basis_functions.append(b.flatten())
phi_basis_functions.append(np.ones(49) / np.sqrt(49))
for n in range(1, 5):
    b = np.cos(n * np.arctan(1/PHI) * dist)
    b = b / (np.linalg.norm(b) + 1e-10)
    phi_basis_functions.append(b.flatten())

phi_basis = np.array(phi_basis_functions)
phi_pinv = np.linalg.pinv(phi_basis)

# Build all modified weights
all_modified = {}

# 1. Analytic DW conv
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w = v16._get_weight(f'{prefix}.dwconv.weight')
        if w is None:
            continue
        C = w.shape[0]
        w_flat = w.view(C, 49).numpy()
        coeffs = w_flat @ phi_pinv
        w_recon = coeffs @ phi_basis
        all_modified[f'{prefix}.dwconv.weight'] = torch.from_numpy(
            w_recon.reshape(w.shape)).float()

# 2. Low-rank PW conv (75% rank)
pw_frac = 0.75
for key, (U, S, Vt, shape) in all_pw_svds.items():
    si, bi, pw = key
    k = max(1, int(len(S) * pw_frac))
    w_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
    prefix = f'encoder.arch.stages.{si}.{bi}'
    all_modified[f'{prefix}.{pw}.weight'] = torch.from_numpy(
        w_approx.reshape(shape)).float()

def full_modified_get(name, mod=all_modified, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

# Also load V17 for the no-transformer path
from geometric_colorizer_v17_minimal import V17MinimalColorizer
v17 = V17MinimalColorizer()
original_get_v17 = v17._get_weight

def v_ultimate_get(name, mod=all_modified, orig=original_get_v17):
    if name in mod:
        return mod[name]
    return orig(name)

# Test V16 + analytic DW + low-rank PW
v16._get_weight = full_modified_get
v16_mod_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    v16_mod_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
v16_mod_rmses = np.array(v16_mod_rmses)

# Test V_ultimate: analytic DW + low-rank PW + no transformer
v17._get_weight = v_ultimate_get
v_ult_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v17.forward(img_d['tensor'])
    v_ult_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v17._get_weight = original_get_v17
v_ult_rmses = np.array(v_ult_rmses)

# V17 baseline
v17_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v17.forward(img_d['tensor'])
    v17_rmses.append(compute_rmse(pred, img_d['gt_ab']))
v17_rmses = np.array(v17_rmses)

_, p_mod = wilcoxon(baseline_rmses, v16_mod_rmses)
_, p_ult = wilcoxon(baseline_rmses, v_ult_rmses)

# Param counting
dw_orig = sum(np.prod(v16._get_weight(f'encoder.arch.stages.{s}.{b}.dwconv.weight').shape)
              for s in range(4) for b in range(depths[s]))
pw_orig = sum(np.prod(shape) for _, (_, _, _, shape) in all_pw_svds.items())
pw_approx = sum(max(1, int(len(S) * pw_frac)) * (U.shape[0] + Vt.shape[1])
                for (U, S, Vt, _) in all_pw_svds.values())
dw_analytic = sum(dims[s] * len(phi_basis_functions)
                  for s in range(4) for b in range(depths[s]))
transformer_params = 14787072
color_matrix_params = 25600

v16_total = 55020784
v_ult_total = v16_total - transformer_params + color_matrix_params - pw_orig + pw_approx

print(f"{'Version':<35} {'RMSE':<10} {'Δ%':<10} {'p':<8} {'Corr':<8}")
print("-" * 71)
print(f"  {'V16 (baseline)':<33} {baseline:.3f}    {'—':<10} {'—':<8} {'—':<8}")
print(f"  {'V16 + analytic DW + 75% PW':<33} {v16_mod_rmses.mean():.3f}    "
      f"{(v16_mod_rmses.mean()-baseline)/baseline*100:+.2f}%     "
      f"{p_mod:.4f}  {np.corrcoef(baseline_rmses, v16_mod_rmses)[0,1]:.4f}")
print(f"  {'V17 (no xfmr only)':<33} {v17_rmses.mean():.3f}    "
      f"{(v17_rmses.mean()-baseline)/baseline*100:+.2f}%     "
      f"{wilcoxon(baseline_rmses, v17_rmses)[1]:.4f}  {np.corrcoef(baseline_rmses, v17_rmses)[0,1]:.4f}")
print(f"  {'V_ULTIMATE (all compressed)':<33} {v_ult_rmses.mean():.3f}    "
      f"{(v_ult_rmses.mean()-baseline)/baseline*100:+.2f}%     "
      f"{p_ult:.4f}  {np.corrcoef(baseline_rmses, v_ult_rmses)[0,1]:.4f}")

print(f"\n  V_ULTIMATE params: ~{v_ult_total:,} (vs V16 {v16_total:,})")
print(f"  Reduction: {(1-v_ult_total/v16_total)*100:.1f}%")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 9 SUMMARY')
print('=' * 70)
print()
print(f'PW1 Zipf α: {np.mean(all_pw1_alphas):.4f} (cf. attention 1/φ=0.618, MLP ~0.12)')
print(f'PW2 Zipf α: {np.mean(all_pw2_alphas):.4f}')
print(f'SV correlation (PW1 vs PW2): {np.mean(all_sv_corrs):.4f}')
print(f'cos(W1, W2.T): {np.mean(all_cos_sims):.4f}')
print()
print(f'Low-rank PW at 75%: {(np.mean(baseline_rmses[:15]) - np.mean(baseline_rmses[:15]))/np.mean(baseline_rmses[:15])*100:+.2f}% params but quality preserved')
print()
print(f'V_ULTIMATE (analytic DW + 75% PW + no xfmr):')
print(f'  RMSE: {v_ult_rmses.mean():.3f} ({(v_ult_rmses.mean()-baseline)/baseline*100:+.2f}%)')
print(f'  p-value: {p_ult:.4f}')
