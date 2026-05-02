"""
Phase 12: What Do the Singular Vector Directions Encode?

Phase 11 proved: directions (U,V) carry ALL the information in PW conv.
Magnitudes (S) are freely interchangeable. Random directions +39% RMSE.

THE QUESTION: Are the U/V directions structured or random?

Tests:
  1. Cosine similarity structure across blocks — do later blocks build on earlier?
  2. Discrete Cosine Transform (DCT) structure — are U/V close to DCT basis?
  3. Cross-block alignment — do singular vectors progress systematically?
  4. Sparsity — are U/V sparse or dense?
  5. φ-structure in vector entries — do entries follow φ-lattice?
"""
import numpy as np
import sys
from scipy.optimize import curve_fit

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')

from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
v16 = V16GeometricColorizer()
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]


# Collect all SVDs
all_svds = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw in ['pwconv1', 'pwconv2']:
            w = v16._get_weight(f'{prefix}.{pw}.weight')
            if w is None:
                continue
            w_np = w.numpy()
            U, S, Vt = np.linalg.svd(w_np, full_matrices=False)
            all_svds[(stage_idx, block_idx, pw)] = (U, S, Vt, w.shape)


# ================================================================
# STEP 1: Cross-block singular vector alignment
# ================================================================
print('=' * 70)
print('STEP 1: Cross-Block Singular Vector Alignment')
print('=' * 70)
print()

# Within a stage, do consecutive blocks share singular vector structure?
# If yes: there's a "thread" through the blocks (progressive refinement)
# If no: each block discovers independent structure

print("V1t alignment between consecutive blocks (same stage):")
print(f"{'Pair':<15} {'Top-1 cos':<12} {'Top-5 cos':<12} {'Top-10 cos':<12}")
print("-" * 51)

for stage_idx in range(4):
    for b1 in range(depths[stage_idx] - 1):
        b2 = b1 + 1
        key1 = (stage_idx, b1, 'pwconv1')
        key2 = (stage_idx, b2, 'pwconv1')
        if key1 not in all_svds or key2 not in all_svds:
            continue
        
        _, _, V1t_a, _ = all_svds[key1]
        _, _, V1t_b, _ = all_svds[key2]
        
        # Top-k cosine similarity (take max alignment per vector)
        for k_top, label in [(1, 'Top-1'), (5, 'Top-5'), (10, 'Top-10')]:
            alignment = np.abs(V1t_a[:k_top] @ V1t_b[:k_top].T)
            max_align = alignment.max(axis=1).mean()
            if label == 'Top-1':
                cos1 = max_align
            elif label == 'Top-5':
                cos5 = max_align
            else:
                cos10 = max_align
        
        print(f"  {stage_idx}.{b1}→{b2:<8} {cos1:<12.4f} {cos5:<12.4f} {cos10:<12.4f}")

# Cross-stage alignment (using first block of each stage)
print(f"\nCross-stage alignment (first blocks):")
print(f"  (Not directly comparable — different dimensions)")


# ================================================================
# STEP 2: DCT/DFT structure test
# ================================================================
print()
print('=' * 70)
print('STEP 2: DCT/DFT Structure in Singular Vectors')
print('=' * 70)
print()

# The DCT basis is the "natural" basis for many signal processing tasks.
# If U/V are close to DCT, the network learned a Fourier-like decomposition.

from scipy.fft import dct

print(f"{'Block':<10} {'PW':<5} {'V·DCT align':<15} {'U sparsity':<12} {'V sparsity':<12}")
print("-" * 54)

all_dct_aligns = []
all_u_sparsities = []
all_v_sparsities = []

for stage_idx in [0, 2, 3]:
    for block_idx in [0, depths[stage_idx]-1]:
        for pw in ['pwconv1']:
            key = (stage_idx, block_idx, pw)
            if key not in all_svds:
                continue
            
            U, S, Vt, shape = all_svds[key]
            C = Vt.shape[0]  # min dimension
            
            # Build DCT basis for C dimensions
            dct_basis = np.zeros((C, C))
            for i in range(C):
                e = np.zeros(C)
                e[i] = 1.0
                dct_basis[i] = dct(e, type=2, norm='ortho')
            
            # Alignment between V and DCT
            align = np.abs(Vt[:min(20, C)] @ dct_basis[:min(20, C)].T)
            max_per_row = align.max(axis=1).mean()
            all_dct_aligns.append(max_per_row)
            
            # Sparsity: what fraction of entries are "small"?
            u_sparse = (np.abs(U) < 0.05).mean()
            v_sparse = (np.abs(Vt) < 0.05).mean()
            all_u_sparsities.append(u_sparse)
            all_v_sparsities.append(v_sparse)
            
            print(f"  {stage_idx}.{block_idx:<7} {pw:<5} {max_per_row:<15.4f} "
                  f"{u_sparse:<12.4f} {v_sparse:<12.4f}")

print(f"\nMean DCT alignment: {np.mean(all_dct_aligns):.4f} (1.0=perfect, random≈{1/np.sqrt(dims[0]):.4f})")
print(f"Mean U sparsity: {np.mean(all_u_sparsities)*100:.1f}%")
print(f"Mean V sparsity: {np.mean(all_v_sparsities)*100:.1f}%")


# ================================================================
# STEP 3: Entry distribution — φ-structure?
# ================================================================
print()
print('=' * 70)
print('STEP 3: Entry Distribution in Singular Vectors')
print('=' * 70)
print()

# Collect all entries from V1t (right singular vectors of PW1)
all_v_entries = []
for (si, bi, pw), (U, S, Vt, shape) in all_svds.items():
    if pw == 'pwconv1':
        all_v_entries.extend(Vt[:10].flatten().tolist())

all_v_entries = np.array(all_v_entries)

print(f"V entries (top 10 SVs, all PW1 blocks): {len(all_v_entries)} total")
print(f"  Mean: {all_v_entries.mean():.6f}")
print(f"  Std:  {all_v_entries.std():.6f}")
print(f"  Kurtosis: {((all_v_entries - all_v_entries.mean())**4).mean() / all_v_entries.std()**4:.4f}")
print(f"  (Gaussian=3.0, heavy-tailed>3, peaked<3)")

# Do entries cluster at specific values?
abs_entries = np.abs(all_v_entries)
print(f"\n  |V| distribution:")
print(f"    Mean |V|: {abs_entries.mean():.6f}")
print(f"    Expected for random orthogonal (1/√C ≈ {1/np.sqrt(96):.6f} to {1/np.sqrt(768):.6f})")

# Check if entries follow φ-lattice
# In φ-encoding: values ≈ sign × φ^(k/scale) for integer k
log_phi_entries = np.log(abs_entries[abs_entries > 1e-8] + 1e-10) / np.log(PHI)
log_phi_frac = log_phi_entries - np.round(log_phi_entries)

print(f"\n  φ-lattice test (log_φ(|V|) modulo 1):")
print(f"    Mean fractional part: {np.abs(log_phi_frac).mean():.4f}")
print(f"    (0.0 = on φ-lattice, 0.25 = random)")

from scipy.stats import kstest
ks_stat, ks_p = kstest(log_phi_frac, 'uniform', args=(-0.5, 1.0))
print(f"    KS test vs uniform: stat={ks_stat:.4f}, p={ks_p:.6f}")
print(f"    → {'NOT on φ-lattice' if ks_p > 0.05 else 'φ-lattice structure detected!'}")


# ================================================================
# STEP 4: Progressive rotation through blocks
# ================================================================
print()
print('=' * 70)
print('STEP 4: Progressive Rotation Through Blocks')
print('=' * 70)
print()

# Within stage 2 (9 blocks), track how V1t rotates from block to block
# This tests whether the network builds a progressive representation

print("Stage 2 (9 blocks): Angular distance between V1t of consecutive blocks")
print(f"{'Pair':<12} {'Angle (°)':<12} {'Cumulative':<12}")
print("-" * 36)

cumulative = 0
for b in range(depths[2] - 1):
    key_a = (2, b, 'pwconv1')
    key_b = (2, b+1, 'pwconv1')
    if key_a not in all_svds or key_b not in all_svds:
        continue
    
    Va = all_svds[key_a][2][:20]  # Top 20 singular vectors
    Vb = all_svds[key_b][2][:20]
    
    # Procrustes alignment to find rotation angle
    M = Va @ Vb.T  # [20, 20]
    U_p, S_p, Vt_p = np.linalg.svd(M)
    
    # The rotation angle is related to the trace of the optimal rotation
    # For a rotation in n-d: trace(R) = Σ cos(θ_i)
    # Average rotation angle: arccos(trace/n)
    trace_val = S_p.sum() / len(S_p)
    angle = np.degrees(np.arccos(np.clip(trace_val, -1, 1)))
    cumulative += angle
    
    print(f"  {b}→{b+1:<8} {angle:<12.2f} {cumulative:<12.2f}")

print(f"\n  Total rotation across 9 blocks: {cumulative:.1f}°")
if 85 < cumulative < 95:
    print(f"  → Close to 90°! The 9 blocks perform a QUARTER rotation.")
elif 170 < cumulative < 190:
    print(f"  → Close to 180°! The 9 blocks perform a HALF rotation.")
elif cumulative < 45:
    print(f"  → Small total rotation — blocks refine rather than rotate.")
else:
    print(f"  → {cumulative/9:.1f}° per block average.")


# ================================================================
# STEP 5: Effective dimensionality of V space
# ================================================================
print()
print('=' * 70)
print('STEP 5: Effective Dimensionality of Singular Vector Space')
print('=' * 70)
print()

# If we stack all V1t vectors from all blocks, what's the effective
# dimensionality? If low → there's a shared subspace. If high → independent.

# Stage 2 has constant dimension (384), so we can stack
stage2_vecs = []
for b in range(depths[2]):
    key = (2, b, 'pwconv1')
    if key in all_svds:
        Vt = all_svds[key][2][:20]  # Top 20 per block
        stage2_vecs.append(Vt)

if stage2_vecs:
    stacked = np.vstack(stage2_vecs)  # [9*20, 384]
    _, S_stacked, _ = np.linalg.svd(stacked, full_matrices=False)
    cumvar = np.cumsum(S_stacked**2) / np.sum(S_stacked**2)
    
    rank90 = np.searchsorted(cumvar, 0.9) + 1
    rank99 = np.searchsorted(cumvar, 0.99) + 1
    
    print(f"Stage 2: Stacked V1t (top 20 per block, 9 blocks)")
    print(f"  Shape: {stacked.shape}")
    print(f"  Rank for 90%: {rank90} (out of {stacked.shape[0]})")
    print(f"  Rank for 99%: {rank99}")
    print(f"  Effective dim fraction: {rank90/stacked.shape[0]*100:.1f}%")
    
    if rank90 < stacked.shape[0] * 0.5:
        print(f"  → SHARED SUBSPACE: blocks reuse ~{rank90} directions")
        print(f"     This means we could compress V using a shared basis!")
    else:
        print(f"  → INDEPENDENT: each block uses different directions")


# ================================================================
# STEP 6: Can we predict V from block position?
# ================================================================
print()
print('=' * 70)
print('STEP 6: Block Position → Singular Vector Prediction')
print('=' * 70)
print()

# If there's a progression, we should be able to interpolate V from position
# Test: use blocks 0,2,4,6,8 to predict blocks 1,3,5,7

print("Stage 2: interpolation test (predict odd blocks from even blocks)")
print(f"{'Target':<10} {'Interp align':<15} {'Random align':<15}")
print("-" * 40)

for target_b in [1, 3, 5, 7]:
    prev_b = target_b - 1
    next_b = target_b + 1
    
    key_prev = (2, prev_b, 'pwconv1')
    key_next = (2, next_b, 'pwconv1')
    key_target = (2, target_b, 'pwconv1')
    
    if not all(k in all_svds for k in [key_prev, key_next, key_target]):
        continue
    
    V_prev = all_svds[key_prev][2][:10]
    V_next = all_svds[key_next][2][:10]
    V_target = all_svds[key_target][2][:10]
    
    # Linear interpolation
    V_interp = (V_prev + V_next) / 2
    # Re-orthogonalize via SVD: V_interp is [10, C], we want orthonormal rows
    _, _, Vt_orth = np.linalg.svd(V_interp, full_matrices=False)
    V_interp_orth = Vt_orth[:10]  # [10, C]
    
    # Alignment
    align_interp = np.abs(V_target @ V_interp_orth.T).max(axis=1).mean()
    
    # Random baseline
    Q_rand, _ = np.linalg.qr(np.random.randn(V_target.shape[1], 10))
    align_rand = np.abs(V_target @ Q_rand).max(axis=1).mean()
    
    print(f"  Block {target_b:<6} {align_interp:<15.4f} {align_rand:<15.4f}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 12 SUMMARY: Direction Structure')
print('=' * 70)
print()
print(f'Cross-block V alignment (consecutive): varies by stage')
print(f'DCT alignment: {np.mean(all_dct_aligns):.4f}')
print(f'V entry kurtosis: {((all_v_entries - all_v_entries.mean())**4).mean() / all_v_entries.std()**4:.2f}')
print(f'φ-lattice in entries: {"No" if ks_p > 0.05 else "Yes"} (p={ks_p:.4f})')

if stage2_vecs:
    print(f'Shared subspace: rank {rank90} for 90% ({rank90/stacked.shape[0]*100:.0f}% of max)')
    
print()
print('The directions are:')
if np.mean(all_dct_aligns) > 0.5:
    print('  - DCT-like (frequency decomposition)')
elif rank90 < stacked.shape[0] * 0.5:
    print('  - Shared subspace across blocks (compressible!)')
else:
    print('  - Block-specific (each block discovers its own directions)')
print(f'  - Dense (sparsity U={np.mean(all_u_sparsities)*100:.0f}%, V={np.mean(all_v_sparsities)*100:.0f}%)')
print(f'  - Progressive rotation: {cumulative:.0f}° across 9 blocks')
