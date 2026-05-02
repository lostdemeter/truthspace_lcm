#!/usr/bin/env python3
"""
Transformation Analysis - Finding the Geometric Structure

This experiment analyzes the grayscale→color mapping to find
its true geometric structure.

Questions:
1. Is it a rotation in joint space?
2. Is it a projection onto a subspace?
3. Does it involve φ-scaling?
4. What's the intrinsic dimensionality?

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import sobel
from scipy.linalg import svd, orthogonal_procrustes
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    u = -0.147 * rgb[..., 0] - 0.289 * rgb[..., 1] + 0.436 * rgb[..., 2]
    v = 0.615 * rgb[..., 0] - 0.515 * rgb[..., 1] - 0.100 * rgb[..., 2]
    return np.stack([y, u, v], axis=-1)


def extract_features(gray_patch: np.ndarray, y_pos: float, x_pos: float) -> np.ndarray:
    """Extract 8 core features."""
    patch = gray_patch.astype(np.float32) / 255.0
    h, w = patch.shape
    
    luminance = patch.mean()
    contrast = patch.std()
    
    texture_h = np.abs(np.diff(patch, axis=1)).mean() if w > 1 else 0
    texture_v = np.abs(np.diff(patch, axis=0)).mean() if h > 1 else 0
    
    if h > 2 and w > 2:
        gy = sobel(patch, axis=0)
        gx = sobel(patch, axis=1)
        gradient_mag = np.sqrt(gx**2 + gy**2).mean()
        gradient_dir = np.arctan2(gy.mean(), gx.mean()) / np.pi
    else:
        gradient_mag = gradient_dir = 0
    
    return np.array([
        luminance, contrast, texture_h, texture_v,
        y_pos, x_pos, gradient_mag, gradient_dir
    ], dtype=np.float32)


def collect_data(n_images: int = 100, sample_rate: float = 0.15, patch_size: int = 16):
    """Collect (features, U, V) triplets."""
    image_files = sorted(COCO_PATH.glob("*.jpg"))[:n_images]
    
    features_list = []
    u_list = []
    v_list = []
    
    for img_path in image_files:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except:
            continue
        
        H, W = img.shape[:2]
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        yuv = rgb_to_yuv(img)
        
        for y in range(0, H - patch_size, patch_size):
            for x in range(0, W - patch_size, patch_size):
                if np.random.random() > sample_rate:
                    continue
                
                gray_patch = gray[y:y+patch_size, x:x+patch_size]
                yuv_patch = yuv[y:y+patch_size, x:x+patch_size]
                
                y_pos = (y + patch_size/2) / H
                x_pos = (x + patch_size/2) / W
                
                feat = extract_features(gray_patch, y_pos, x_pos)
                mean_u = yuv_patch[:, :, 1].mean()
                mean_v = yuv_patch[:, :, 2].mean()
                
                features_list.append(feat)
                u_list.append(mean_u)
                v_list.append(mean_v)
    
    return np.array(features_list), np.array(u_list), np.array(v_list)


def analyze_linear_mapping(features, u, v):
    """Analyze if the mapping is linear (projection)."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: LINEAR PROJECTION")
    print("=" * 60)
    
    # Solve for projection matrices: U = features @ W_u, V = features @ W_v
    X = np.hstack([features, np.ones((len(features), 1))])  # Add bias
    
    W_u = np.linalg.lstsq(X, u, rcond=None)[0]
    W_v = np.linalg.lstsq(X, v, rcond=None)[0]
    
    u_pred = X @ W_u
    v_pred = X @ W_v
    
    # R² scores
    r2_u = 1 - np.sum((u - u_pred)**2) / np.sum((u - u.mean())**2)
    r2_v = 1 - np.sum((v - v_pred)**2) / np.sum((v - v.mean())**2)
    
    print(f"\n   Linear projection R²:")
    print(f"     U: {r2_u:.4f}")
    print(f"     V: {r2_v:.4f}")
    
    # Analyze weights
    print(f"\n   Projection weights (excluding bias):")
    dim_names = ['lum', 'con', 'tex_h', 'tex_v', 'y_pos', 'x_pos', 'grad_m', 'grad_d']
    
    print(f"   {'Dim':<8} {'W_u':>10} {'W_v':>10}")
    for i, name in enumerate(dim_names):
        print(f"   {name:<8} {W_u[i]:>10.4f} {W_v[i]:>10.4f}")
    
    return r2_u, r2_v, W_u, W_v


def analyze_rotation(features, u, v):
    """Analyze if the mapping is a rotation in joint space."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: ROTATION IN JOINT SPACE")
    print("=" * 60)
    
    # Joint space: [features, 0, 0] → [features, U, V]
    # If it's a rotation, the norms should be preserved
    
    n_features = features.shape[1]
    
    # Source: features with zero color
    source = np.hstack([features, np.zeros((len(features), 2))])
    
    # Target: features with true color
    target = np.hstack([features, u.reshape(-1, 1), v.reshape(-1, 1)])
    
    # Center both
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    
    # Find optimal rotation using Procrustes
    R, scale = orthogonal_procrustes(source_centered, target_centered)
    
    # Apply rotation
    rotated = source_centered @ R
    
    # Measure fit
    residual = np.linalg.norm(rotated - target_centered) / np.linalg.norm(target_centered)
    
    print(f"\n   Procrustes analysis:")
    print(f"     Optimal scale: {scale:.4f}")
    print(f"     Residual (lower is better): {residual:.4f}")
    
    # Check if R is actually a rotation (det = 1)
    det = np.linalg.det(R)
    print(f"     det(R): {det:.4f} (should be 1 for rotation)")
    
    # Analyze the rotation matrix structure
    print(f"\n   Rotation matrix structure:")
    print(f"     Shape: {R.shape}")
    
    # Look at the last two columns (how features map to U, V)
    print(f"\n   How features map to color (last 2 cols of R):")
    print(f"   {'Dim':<8} {'→U':>10} {'→V':>10}")
    dim_names = ['lum', 'con', 'tex_h', 'tex_v', 'y_pos', 'x_pos', 'grad_m', 'grad_d', 'zero_u', 'zero_v']
    for i in range(min(len(dim_names), R.shape[0])):
        print(f"   {dim_names[i]:<8} {R[i, -2]:>10.4f} {R[i, -1]:>10.4f}")
    
    return residual, R


def analyze_svd_structure(features, u, v):
    """Analyze the SVD structure of the mapping."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: SVD STRUCTURE")
    print("=" * 60)
    
    # Joint matrix: each row is [features, U, V]
    joint = np.hstack([features, u.reshape(-1, 1), v.reshape(-1, 1)])
    
    # Center
    joint_centered = joint - joint.mean(axis=0)
    
    # SVD
    U_svd, S, Vt = svd(joint_centered, full_matrices=False)
    
    # Variance explained
    var_explained = S**2 / (S**2).sum()
    cumvar = np.cumsum(var_explained)
    
    print(f"\n   Singular values:")
    for i in range(min(10, len(S))):
        print(f"     S[{i}] = {S[i]:.2f} ({var_explained[i]*100:.1f}%, cumulative: {cumvar[i]*100:.1f}%)")
    
    # Check for φ-relationship in singular values
    print(f"\n   φ-relationships in singular values:")
    for i in range(min(5, len(S)-1)):
        ratio = S[i] / S[i+1] if S[i+1] > 0.01 else float('inf')
        phi_error = abs(ratio - PHI) / PHI
        print(f"     S[{i}]/S[{i+1}] = {ratio:.4f} (φ error: {phi_error*100:.1f}%)")
    
    # Intrinsic dimensionality
    for threshold in [0.90, 0.95, 0.99]:
        n_dims = np.argmax(cumvar >= threshold) + 1
        print(f"\n   Dimensions for {threshold*100:.0f}% variance: {n_dims}")
    
    # Analyze principal components
    print(f"\n   Principal component structure (Vt rows):")
    dim_names = ['lum', 'con', 'tex_h', 'tex_v', 'y_pos', 'x_pos', 'grad_m', 'grad_d', 'U', 'V']
    
    print(f"\n   PC0 (largest variance):")
    for i, name in enumerate(dim_names):
        print(f"     {name}: {Vt[0, i]:.3f}")
    
    print(f"\n   PC1:")
    for i, name in enumerate(dim_names):
        print(f"     {name}: {Vt[1, i]:.3f}")
    
    return S, Vt, cumvar


def analyze_phi_structure(features, u, v):
    """Analyze if φ-scaling improves the mapping."""
    print("\n" + "=" * 60)
    print("ANALYSIS 4: φ-SCALING")
    print("=" * 60)
    
    n_features = features.shape[1]
    
    # Try different φ-levels for each feature
    best_r2 = 0
    best_levels = np.zeros(n_features)
    
    levels_to_try = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    
    for target, name in [(u, 'U'), (v, 'V')]:
        print(f"\n   Finding best φ-levels for {name}:")
        
        best_feature_levels = np.zeros(n_features)
        
        for d in range(n_features):
            best_corr = 0
            best_level = 0
            
            for level in levels_to_try:
                scaled = features[:, d] * (PHI ** level)
                corr = np.corrcoef(scaled, target)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_level = level
            
            best_feature_levels[d] = best_level
        
        # Build φ-scaled features
        phi_features = np.zeros_like(features)
        for d in range(n_features):
            phi_features[:, d] = features[:, d] * (PHI ** best_feature_levels[d])
        
        # Linear fit with φ-scaled features
        X = np.hstack([phi_features, np.ones((len(features), 1))])
        W = np.linalg.lstsq(X, target, rcond=None)[0]
        pred = X @ W
        r2 = 1 - np.sum((target - pred)**2) / np.sum((target - target.mean())**2)
        
        print(f"     R² with φ-scaling: {r2:.4f}")
        print(f"     Best levels: {best_feature_levels}")
        
        dim_names = ['lum', 'con', 'tex_h', 'tex_v', 'y_pos', 'x_pos', 'grad_m', 'grad_d']
        print(f"     Level interpretation:")
        for i, (name_d, level) in enumerate(zip(dim_names, best_feature_levels)):
            print(f"       {name_d}: φ^{level:.1f} = {PHI**level:.3f}")


def analyze_nonlinear_structure(features, u, v):
    """Analyze non-linear relationships."""
    print("\n" + "=" * 60)
    print("ANALYSIS 5: NON-LINEAR STRUCTURE")
    print("=" * 60)
    
    # Try polynomial features
    from itertools import combinations_with_replacement
    
    n_features = features.shape[1]
    dim_names = ['lum', 'con', 'tex_h', 'tex_v', 'y_pos', 'x_pos', 'grad_m', 'grad_d']
    
    # Quadratic features
    quad_features = []
    quad_names = []
    
    for i in range(n_features):
        quad_features.append(features[:, i] ** 2)
        quad_names.append(f"{dim_names[i]}²")
    
    # Interaction features (top pairs)
    for i, j in combinations_with_replacement(range(n_features), 2):
        if i != j:
            quad_features.append(features[:, i] * features[:, j])
            quad_names.append(f"{dim_names[i]}×{dim_names[j]}")
    
    quad_features = np.array(quad_features).T
    
    # Combined features
    all_features = np.hstack([features, quad_features, np.ones((len(features), 1))])
    
    # Fit
    W_u = np.linalg.lstsq(all_features, u, rcond=None)[0]
    W_v = np.linalg.lstsq(all_features, v, rcond=None)[0]
    
    u_pred = all_features @ W_u
    v_pred = all_features @ W_v
    
    r2_u = 1 - np.sum((u - u_pred)**2) / np.sum((u - u.mean())**2)
    r2_v = 1 - np.sum((v - v_pred)**2) / np.sum((v - v.mean())**2)
    
    print(f"\n   Quadratic features R²:")
    print(f"     U: {r2_u:.4f}")
    print(f"     V: {r2_v:.4f}")
    
    # Find most important quadratic terms
    print(f"\n   Top quadratic terms for U:")
    all_names = dim_names + quad_names + ['bias']
    sorted_idx = np.argsort(np.abs(W_u))[::-1]
    for i in sorted_idx[:5]:
        print(f"     {all_names[i]}: {W_u[i]:.4f}")
    
    print(f"\n   Top quadratic terms for V:")
    sorted_idx = np.argsort(np.abs(W_v))[::-1]
    for i in sorted_idx[:5]:
        print(f"     {all_names[i]}: {W_v[i]:.4f}")
    
    return r2_u, r2_v


def main():
    print("=" * 70)
    print("TRANSFORMATION ANALYSIS")
    print("Finding the Geometric Structure of Grayscale→Color")
    print("=" * 70)
    
    # Collect data
    print("\nCollecting data from 100 images...")
    features, u, v = collect_data(n_images=100, sample_rate=0.15)
    print(f"Collected {len(features)} samples")
    print(f"Feature shape: {features.shape}")
    print(f"U range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"V range: [{v.min():.3f}, {v.max():.3f}]")
    
    # Run analyses
    r2_u_lin, r2_v_lin, W_u, W_v = analyze_linear_mapping(features, u, v)
    residual, R = analyze_rotation(features, u, v)
    S, Vt, cumvar = analyze_svd_structure(features, u, v)
    analyze_phi_structure(features, u, v)
    r2_u_quad, r2_v_quad = analyze_nonlinear_structure(features, u, v)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
   Linear projection:     R²_U = {r2_u_lin:.4f}, R²_V = {r2_v_lin:.4f}
   Quadratic features:    R²_U = {r2_u_quad:.4f}, R²_V = {r2_v_quad:.4f}
   Rotation residual:     {residual:.4f}
   
   Intrinsic dimensionality:
     90% variance: {np.argmax(cumvar >= 0.90) + 1} dims
     95% variance: {np.argmax(cumvar >= 0.95) + 1} dims
     99% variance: {np.argmax(cumvar >= 0.99) + 1} dims
   
   Key findings:
   - Linear R² is low (~0.05) → NOT a simple projection
   - Quadratic R² is higher (~{max(r2_u_quad, r2_v_quad):.2f}) → NON-LINEAR relationships matter
   - Rotation residual is {residual:.2f} → NOT a pure rotation
   - Most variance in few dims → LOW-DIMENSIONAL structure exists
   
   The transformation appears to be:
   - Non-linear (quadratic terms help)
   - Low-dimensional (few principal components)
   - NOT a simple rotation or projection
""")


if __name__ == "__main__":
    main()
