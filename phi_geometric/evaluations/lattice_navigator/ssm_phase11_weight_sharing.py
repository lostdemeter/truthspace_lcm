"""
Phase 11: ENCODE=DECODE Weight Sharing Test

Phase 10 confirmed ENCODE=DECODE is universal:
  - SV correlation 0.987 (ConvNeXt), 0.963 (Qwen2)
  - But cosine similarity ≈ 0 (orthogonal in weight space)
  - Singular vectors NOT aligned

THE QUESTION: Can we DERIVE one projection from the other?

If W_up has SVD: U_up @ diag(S_up) @ Vt_up
And W_down has SVD: U_down @ diag(S_down) @ Vt_down
And S_up ≈ S_down (confirmed)

Then: W_down ≈ U_down @ diag(S_up) @ Vt_down
      = U_down @ Vt_down @ Vt_down.T @ diag(S_up) @ Vt_down

The KEY: can we find a CHEAP transform T such that W_down ≈ T @ W_up.T?

Tests on DDColor (ConvNeXt):
  1. Direct transpose: W_down = W_up.T (worst case)
  2. Scaled transpose: W_down = α × W_up.T 
  3. Shared SVs: W_down = U_down @ diag(S_up) @ Vt_down (share S only)
  4. Rotation trick: W_down = R @ W_up.T @ Q where R,Q are cheap rotations
  5. Random U with shared S: W_down = U_rand @ diag(S_up) @ Vt_rand

If ANY of these work, we can halve MLP params.
"""
import numpy as np
import cv2
import sys
import glob
import torch
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

# Baseline
baseline_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    baseline_rmses.append(compute_rmse(pred, img_d['gt_ab']))
baseline_rmses = np.array(baseline_rmses)
baseline = baseline_rmses.mean()
print(f'Baseline RMSE: {baseline:.3f}')

original_get = v16._get_weight


# ================================================================
# STEP 1: Analyze the EXACT relationship between PW1 and PW2
# ================================================================
print()
print('=' * 70)
print('STEP 1: Exact Relationship Between PW1 and PW2')
print('=' * 70)
print()

# For each block, decompose both and measure what's shared
all_svds = {}

for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w1 = v16._get_weight(f'{prefix}.pwconv1.weight')
        w2 = v16._get_weight(f'{prefix}.pwconv2.weight')
        if w1 is None or w2 is None:
            continue
        
        w1_np = w1.numpy()  # [4C, C]
        w2_np = w2.numpy()  # [C, 4C]
        
        U1, S1, V1t = np.linalg.svd(w1_np, full_matrices=False)
        U2, S2, V2t = np.linalg.svd(w2_np, full_matrices=False)
        
        all_svds[(stage_idx, block_idx)] = {
            'U1': U1, 'S1': S1, 'V1t': V1t, 'shape1': w1.shape,
            'U2': U2, 'S2': S2, 'V2t': V2t, 'shape2': w2.shape,
        }

# What if we use S_shared = (S1 + S2) / 2?
print(f"{'Block':<10} {'|S1-S2|/|S|':<14} {'max ratio':<12} {'min ratio':<12}")
print("-" * 48)

for key in sorted(all_svds.keys()):
    d = all_svds[key]
    S1, S2 = d['S1'], d['S2']
    min_len = min(len(S1), len(S2))
    
    rel_diff = np.linalg.norm(S1[:min_len] - S2[:min_len]) / np.linalg.norm(S1[:min_len])
    ratio = S1[:min_len] / (S2[:min_len] + 1e-10)
    
    si, bi = key
    print(f"  {si}.{bi:<7} {rel_diff:<14.4f} {ratio.max():<12.4f} {ratio.min():<12.4f}")


# ================================================================
# STEP 2: Test weight sharing strategies
# ================================================================
print()
print('=' * 70)
print('STEP 2: Weight Sharing Strategies')
print('=' * 70)
print()

strategies = {}

# Strategy A: W2 = W1.T (simplest)
print("Strategy A: W2 = W1.T")
mod_a = {}
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    w1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
    mod_a[f'{prefix}.pwconv2.weight'] = torch.from_numpy(w1.T.copy()).float()

def get_a(name, mod=mod_a, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_a
rmses_a = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_a.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
sub_baseline = baseline_rmses[:15].mean()
delta_a = (np.mean(rmses_a) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_a):.3f} ({delta_a:+.1f}%)")
strategies['A: W2=W1.T'] = np.mean(rmses_a)

# Strategy B: W2 = α × W1.T (scaled)
print("\nStrategy B: W2 = α × W1.T (optimal α per block)")
mod_b = {}
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    w1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
    w2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
    # Find optimal scale: α = trace(W2 @ W1) / trace(W1.T @ W1)
    alpha = np.trace(w2 @ w1) / (np.trace(w1.T @ w1) + 1e-10)
    mod_b[f'{prefix}.pwconv2.weight'] = torch.from_numpy((alpha * w1.T).copy()).float()

def get_b(name, mod=mod_b, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_b
rmses_b = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_b.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
delta_b = (np.mean(rmses_b) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_b):.3f} ({delta_b:+.1f}%)")
strategies['B: W2=α·W1.T'] = np.mean(rmses_b)

# Strategy C: Share singular values only — keep U,V independent
print("\nStrategy C: W2 = U2 @ diag(S_shared) @ V2t (share S only)")
mod_c = {}
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    min_len = min(len(d['S1']), len(d['S2']))
    S_shared = (d['S1'][:min_len] + d['S2'][:min_len]) / 2
    w2_recon = (d['U2'][:, :min_len] * S_shared) @ d['V2t'][:min_len, :]
    mod_c[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
        w2_recon.reshape(d['shape2'])).float()

def get_c(name, mod=mod_c, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_c
rmses_c = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_c.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
delta_c = (np.mean(rmses_c) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_c):.3f} ({delta_c:+.1f}%)")
strategies['C: S_shared'] = np.mean(rmses_c)

# Strategy D: W2 = U2 @ diag(S1) @ V2t (use PW1's S entirely for PW2)
print("\nStrategy D: W2 = U2 @ diag(S1) @ V2t (PW1's SVs for PW2)")
mod_d = {}
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    min_len = min(len(d['S1']), len(d['S2']))
    w2_recon = (d['U2'][:, :min_len] * d['S1'][:min_len]) @ d['V2t'][:min_len, :]
    mod_d[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
        w2_recon.reshape(d['shape2'])).float()

def get_d(name, mod=mod_d, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_d
rmses_d = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_d.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
delta_d = (np.mean(rmses_d) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_d):.3f} ({delta_d:+.1f}%)")
strategies['D: S1_for_S2'] = np.mean(rmses_d)

# Strategy E: φ-predicted SVs: S[i] = S[0] / i^(1/φ)
print("\nStrategy E: W2 = U2 @ diag(S_phi) @ V2t (φ-Zipf predicted SVs)")
mod_e = {}
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    min_len = min(len(d['S1']), len(d['S2']))
    # φ-Zipf: S[i] = S[0] / i^(1/φ)
    ranks = np.arange(1, min_len + 1).astype(float)
    S_phi = d['S2'][0] / ranks**(1/PHI)
    w2_recon = (d['U2'][:, :min_len] * S_phi) @ d['V2t'][:min_len, :]
    mod_e[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
        w2_recon.reshape(d['shape2'])).float()

def get_e(name, mod=mod_e, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_e
rmses_e = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_e.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
delta_e = (np.mean(rmses_e) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_e):.3f} ({delta_e:+.1f}%)")
strategies['E: φ-Zipf SVs'] = np.mean(rmses_e)

# Strategy F: W2 = V1t.T @ diag(S1) @ U1.T 
# (reverse the SVD directions — literally encode=decode)
print("\nStrategy F: W2 = V1.T @ diag(S1) @ U1.T (literal encode=decode)")
mod_f = {}
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    # W1 = U1 @ S1 @ V1t, so W1.T = V1 @ S1 @ U1t
    # But we want W2 shape [C, 4C], and W1.T is [C, 4C] ✓
    w2_enc = d['V1t'].T @ np.diag(d['S1']) @ d['U1'].T
    mod_f[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
        w2_enc.reshape(d['shape2'])).float()

def get_f(name, mod=mod_f, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_f
rmses_f = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_f.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
delta_f = (np.mean(rmses_f) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_f):.3f} ({delta_f:+.1f}%)")
strategies['F: literal E=D'] = np.mean(rmses_f)


# ================================================================
# STEP 3: What DO the directions contribute?
# ================================================================
print()
print('=' * 70)
print('STEP 3: Direction vs Magnitude Contribution')
print('=' * 70)
print()

# Strategy C (shared S, real U/V) tells us: how much does S matter alone?
# Strategy F (shared everything from W1) tells us: how much do U/V matter?
# If C ≈ baseline and F is bad → directions matter MORE than magnitudes
# If F ≈ baseline and C is unchanged → directions match, magnitudes matter

print(f"Baseline:       {sub_baseline:.3f}")
print(f"C (shared S):   {np.mean(rmses_c):.3f} ({delta_c:+.1f}%) — changed magnitudes only")
print(f"D (W1's S):     {np.mean(rmses_d):.3f} ({delta_d:+.1f}%) — used W1's magnitudes for W2")
print(f"E (φ-Zipf S):   {np.mean(rmses_e):.3f} ({delta_e:+.1f}%) — φ-predicted magnitudes")
print(f"F (W1 for W2):  {np.mean(rmses_f):.3f} ({delta_f:+.1f}%) — used W1's EVERYTHING")
print(f"A (W1.T=W2):    {np.mean(rmses_a):.3f} ({delta_a:+.1f}%) — transpose (wrong directions)")
print()

if abs(delta_c) < 5 and abs(delta_d) < 5:
    print("→ MAGNITUDES DON'T MATTER MUCH: you can share or swap S without impact")
    print("  This means the DIRECTIONS (U,V) carry the essential information")
elif abs(delta_f) < abs(delta_a):
    print("→ W1 directions are PARTIALLY usable for W2")
    print("  The SVD decomposition helps — share S, keep U/V separate")
else:
    print("→ Directions are architecture-critical: U/V cannot be shared or swapped")

# How many parameters does sharing S save?
total_s_params = 0
total_uv_params = 0
total_orig = 0
for (si, bi), d in all_svds.items():
    C = d['shape1'][1]  # smaller dimension
    total_s_params += len(d['S1']) + len(d['S2'])  # both S vectors
    total_uv_params += d['U1'].size + d['V1t'].size + d['U2'].size + d['V2t'].size
    total_orig += np.prod(d['shape1']) + np.prod(d['shape2'])

s_savings = total_s_params / 2  # halved by sharing
print(f"\nParam analysis:")
print(f"  Total PW params: {total_orig:,}")
print(f"  Total S params:  {total_s_params:,} (shared saves {s_savings:,.0f})")
print(f"  S is only {total_s_params/total_orig*100:.2f}% of PW params")
print(f"  Sharing S saves ~{s_savings/total_orig*100:.3f}% — negligible!")


# ================================================================
# STEP 4: The REAL weight sharing test — random U/V with correct S
# ================================================================
print()
print('=' * 70)
print('STEP 4: Random Directions with Correct Magnitudes')
print('=' * 70)
print()

# If directions are critical, random U/V with correct S should fail
# If magnitudes are critical, random U/V with correct S should work
print("Strategy G: W2 = random_U @ diag(S2) @ random_V")
mod_g = {}
np.random.seed(42)
for (si, bi), d in all_svds.items():
    prefix = f'encoder.arch.stages.{si}.{bi}'
    C = d['shape2'][0]
    fourC = d['shape2'][1]
    min_dim = min(C, fourC)
    
    # Random orthogonal matrices
    Q1, _ = np.linalg.qr(np.random.randn(C, C))
    Q2, _ = np.linalg.qr(np.random.randn(fourC, fourC))
    
    w2_rand = Q1[:, :min_dim] @ np.diag(d['S2'][:min_dim]) @ Q2[:min_dim, :]
    mod_g[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
        w2_rand.reshape(d['shape2']).astype(np.float32))

def get_g(name, mod=mod_g, orig=original_get):
    if name in mod:
        return mod[name]
    return orig(name)

v16._get_weight = get_g
rmses_g = []
for img_d in test_data[:15]:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    rmses_g.append(compute_rmse(pred, img_d['gt_ab']))
v16._get_weight = original_get
delta_g = (np.mean(rmses_g) - sub_baseline) / sub_baseline * 100
print(f"  RMSE: {np.mean(rmses_g):.3f} ({delta_g:+.1f}%)")
strategies['G: random U/V'] = np.mean(rmses_g)


# ================================================================
# STEP 5: Low-rank shared representation
# ================================================================
print()
print('=' * 70)
print('STEP 5: Low-Rank Shared Core')
print('=' * 70)
print()

# Idea: W1 and W2 share a low-rank core K, plus small corrections
# W1 ≈ A1 @ K, W2 ≈ K @ A2
# where K is [r, C] and A1 is [4C, r], A2 is [C, r]... doesn't save much

# Better idea: jointly decompose W1 and W2
# Stack them: M = [W1; W2.T] is [8C, C]
# SVD of M captures shared structure

for (si, bi) in [(2, 4)]:  # representative block
    d = all_svds[(si, bi)]
    prefix = f'encoder.arch.stages.{si}.{bi}'
    
    w1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()  # [4C, C]
    w2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()  # [C, 4C]
    
    C = w1.shape[1]
    
    # Stack
    M = np.vstack([w1, w2.T])  # [8C, C]
    U_M, S_M, Vt_M = np.linalg.svd(M, full_matrices=False)
    
    cumvar = np.cumsum(S_M**2) / np.sum(S_M**2)
    
    print(f"Block {si}.{bi}: Stacked [W1; W2.T] SVD:")
    print(f"  Shape: {M.shape}")
    print(f"  Rank for 90%: {np.searchsorted(cumvar, 0.9) + 1}")
    print(f"  Rank for 95%: {np.searchsorted(cumvar, 0.95) + 1}")
    print(f"  Rank for 99%: {np.searchsorted(cumvar, 0.99) + 1}")
    print(f"  Full rank: {len(S_M)}")
    
    # Reconstruct from shared basis
    for rank_frac in [0.25, 0.5, 0.75, 1.0]:
        k = max(1, int(len(S_M) * rank_frac))
        M_approx = (U_M[:, :k] * S_M[:k]) @ Vt_M[:k, :]
        
        w1_approx = M_approx[:4*C, :]
        w2_approx = M_approx[4*C:, :].T
        
        err_w1 = np.linalg.norm(w1_approx - w1) / np.linalg.norm(w1) * 100
        err_w2 = np.linalg.norm(w2_approx - w2) / np.linalg.norm(w2) * 100
        
        # Param savings: store U_M[:,:k] (8C×k), S_M[:k] (k), Vt_M[:k,:] (k×C)
        shared_params = 8*C*k + k + k*C
        orig_params = np.prod(w1.shape) + np.prod(w2.shape)
        
        print(f"  Rank {rank_frac*100:>5.1f}% (k={k}): "
              f"W1 err={err_w1:.2f}%, W2 err={err_w2:.2f}%, "
              f"params: {orig_params:,} → {shared_params:,} ({shared_params/orig_params*100:.1f}%)")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 11 SUMMARY: Weight Sharing')
print('=' * 70)
print()

print(f"{'Strategy':<22} {'RMSE':<10} {'Δ%':<10}")
print("-" * 42)
print(f"  {'Baseline':<20} {sub_baseline:<10.3f} {'—':<10}")
for name, rmse_val in sorted(strategies.items(), key=lambda x: x[1]):
    delta = (rmse_val - sub_baseline) / sub_baseline * 100
    print(f"  {name:<20} {rmse_val:<10.3f} {delta:+.1f}%")

print()

# Find best strategy
best_name = min(strategies, key=strategies.get)
best_rmse = strategies[best_name]
best_delta = (best_rmse - sub_baseline) / sub_baseline * 100

print(f"Best strategy: {best_name} ({best_delta:+.1f}%)")
print()

if abs(best_delta) < 5:
    print("→ Weight sharing IS viable for PW conv!")
    print("  But S is only 0.1% of params — sharing S doesn't save much.")
    print("  The DIRECTIONS (U, V) are what costs — and they can't be shared.")
else:
    print("→ Weight sharing is NOT viable — directions are critical.")
    print("  ENCODE=DECODE means same SPECTRAL SHAPE, not same WEIGHTS.")
    print("  The spectral symmetry is a constraint, not a shortcut.")

print()
print("KEY INSIGHT: ENCODE=DECODE is about SPECTRAL STRUCTURE, not weight reuse.")
print("The encode and decode matrices have identical 'bandwidth' (SV spectrum)")
print("but route information through DIFFERENT subspaces. Both subspaces are needed.")
