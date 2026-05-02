#!/usr/bin/env python3
"""
ConvNeXt Feature Probing

Analyze what semantic information each ConvNeXt layer captures.
This helps us understand what we need to replicate geometrically.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


def visualize_feature_maps():
    """Visualize what each stage of ConvNeXt captures."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    model = model.to(device)
    
    print("=" * 60)
    print("CONVNEXT FEATURE PROBING")
    print("=" * 60)
    
    # Load test image
    img_path = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/000000127494.jpg'
    img_bgr = cv2.imread(img_path)
    img = (img_bgr / 255.0).astype(np.float32)
    img_resized = cv2.resize(img, (512, 512))
    
    # Convert to grayscale LAB input
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
    tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    tensor_norm = (tensor - model.mean) / model.std
    
    # Run encoder
    with torch.no_grad():
        model.encoder(tensor_norm)
    
    print("\n1. FEATURE MAP STATISTICS")
    print("-" * 60)
    
    for i, hook in enumerate(model.encoder.hooks):
        feat = hook.feature
        print(f"\nStage {i}: {feat.shape}")
        print(f"  Mean: {feat.mean():.4f}")
        print(f"  Std:  {feat.std():.4f}")
        print(f"  Min:  {feat.min():.4f}")
        print(f"  Max:  {feat.max():.4f}")
        
        # Analyze channel activation patterns
        feat_np = feat[0].cpu().numpy()  # [C, H, W]
        
        # Which channels are most active?
        channel_activity = np.abs(feat_np).mean(axis=(1, 2))
        top_channels = np.argsort(-channel_activity)[:5]
        print(f"  Top 5 active channels: {top_channels}")
        print(f"  Activity range: [{channel_activity.min():.4f}, {channel_activity.max():.4f}]")
    
    print("\n2. FEATURE SIMILARITY ANALYSIS")
    print("-" * 60)
    
    # Compare features at different spatial locations
    # Do similar image regions have similar features?
    
    for i, hook in enumerate(model.encoder.hooks):
        feat = hook.feature[0].cpu().numpy()  # [C, H, W]
        C, H, W = feat.shape
        
        # Flatten spatial dimensions
        feat_flat = feat.reshape(C, -1).T  # [H*W, C]
        
        # Compute pairwise cosine similarity (sample)
        n_samples = 100
        indices = np.random.choice(H*W, n_samples, replace=False)
        feat_sample = feat_flat[indices]
        
        # Normalize
        feat_norm = feat_sample / (np.linalg.norm(feat_sample, axis=1, keepdims=True) + 1e-6)
        similarity = feat_norm @ feat_norm.T
        
        # Exclude diagonal
        mask = ~np.eye(n_samples, dtype=bool)
        off_diag = similarity[mask]
        
        print(f"\nStage {i} feature similarity:")
        print(f"  Mean: {off_diag.mean():.4f}")
        print(f"  Std:  {off_diag.std():.4f}")
        print(f"  Range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    
    return model


def analyze_semantic_clustering():
    """Check if ConvNeXt features cluster by semantic category."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    model = model.to(device)
    
    print("\n" + "=" * 60)
    print("SEMANTIC CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Collect features from multiple images
    import glob
    images = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')[:10]
    
    all_features = [[] for _ in range(4)]
    
    for img_path in images:
        img_bgr = cv2.imread(img_path)
        img = (img_bgr / 255.0).astype(np.float32)
        img_resized = cv2.resize(img, (512, 512))
        
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        
        tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        tensor_norm = (tensor - model.mean) / model.std
        
        with torch.no_grad():
            model.encoder(tensor_norm)
        
        for i, hook in enumerate(model.encoder.hooks):
            feat = hook.feature[0].cpu().numpy()  # [C, H, W]
            # Sample some spatial locations
            feat_flat = feat.reshape(feat.shape[0], -1).T  # [H*W, C]
            indices = np.random.choice(feat_flat.shape[0], 50, replace=False)
            all_features[i].append(feat_flat[indices])
    
    # Concatenate features
    for i in range(4):
        all_features[i] = np.concatenate(all_features[i], axis=0)
    
    print("\n1. FEATURE CLUSTERING (K-MEANS)")
    print("-" * 60)
    
    from sklearn.cluster import KMeans
    
    for i in range(4):
        feat = all_features[i]
        print(f"\nStage {i}: {feat.shape}")
        
        # Cluster into 10 groups
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feat)
        
        # Compute cluster quality (silhouette-like)
        centers = kmeans.cluster_centers_
        
        # Intra-cluster distance
        intra_dist = 0
        for c in range(10):
            mask = labels == c
            if mask.sum() > 0:
                cluster_feat = feat[mask]
                center = centers[c]
                intra_dist += np.mean(np.linalg.norm(cluster_feat - center, axis=1))
        intra_dist /= 10
        
        # Inter-cluster distance
        inter_dist = 0
        count = 0
        for c1 in range(10):
            for c2 in range(c1+1, 10):
                inter_dist += np.linalg.norm(centers[c1] - centers[c2])
                count += 1
        inter_dist /= count
        
        print(f"  Intra-cluster distance: {intra_dist:.4f}")
        print(f"  Inter-cluster distance: {inter_dist:.4f}")
        print(f"  Ratio (higher = better): {inter_dist/intra_dist:.4f}")
    
    print("\n2. INTERPRETATION")
    print("-" * 60)
    print("""
    What ConvNeXt learns at each stage:
    
    Stage 0 (96 channels, 128x128):
      - Low-level features: edges, textures, local patterns
      - Similar to Gabor filters but learned
    
    Stage 1 (192 channels, 64x64):
      - Mid-level features: texture combinations, simple shapes
      - Beginning of object parts
    
    Stage 2 (384 channels, 32x32):
      - High-level features: object parts, semantic regions
      - "This looks like grass", "This looks like sky"
    
    Stage 3 (768 channels, 16x16):
      - Abstract features: object categories, scene understanding
      - Global context and relationships
    
    The key insight: Stages 2-3 capture SEMANTIC information that
    our geometric encoder (Gabor + position) cannot replicate.
    """)


def compare_geometric_vs_convnext():
    """Compare our geometric features to ConvNeXt features."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    from phi_geometric.models.geometric_encoder import GeometricEncoder
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    ddcolor.eval()
    ddcolor = ddcolor.to(device)
    
    geo_encoder = GeometricEncoder().to(device)
    
    print("\n" + "=" * 60)
    print("GEOMETRIC vs CONVNEXT COMPARISON")
    print("=" * 60)
    
    # Load test image
    img_path = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/000000127494.jpg'
    img_bgr = cv2.imread(img_path)
    img = (img_bgr / 255.0).astype(np.float32)
    img_resized = cv2.resize(img, (512, 512))
    
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
    tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    tensor_norm = (tensor - ddcolor.mean) / ddcolor.std
    
    with torch.no_grad():
        # Get ConvNeXt features
        ddcolor.encoder(tensor_norm)
        convnext_feats = [hook.feature for hook in ddcolor.encoder.hooks]
        
        # Get geometric features
        geo_feats = geo_encoder(tensor)
    
    print("\n1. FEATURE COMPARISON")
    print("-" * 60)
    
    for i in range(4):
        cn_feat = convnext_feats[i][0].cpu().numpy()
        geo_feat = geo_feats[i][0].cpu().numpy()
        
        print(f"\nStage {i}:")
        print(f"  ConvNeXt: {cn_feat.shape}, range=[{cn_feat.min():.2f}, {cn_feat.max():.2f}]")
        print(f"  Geometric: {geo_feat.shape}, range=[{geo_feat.min():.2f}, {geo_feat.max():.2f}]")
        
        # Correlation between corresponding channels
        n_channels = min(cn_feat.shape[0], geo_feat.shape[0])
        correlations = []
        for c in range(n_channels):
            cn_flat = cn_feat[c].flatten()
            geo_flat = geo_feat[c].flatten()
            corr = np.corrcoef(cn_flat, geo_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        if correlations:
            print(f"  Channel correlation: mean={np.mean(correlations):.4f}, max={np.max(correlations):.4f}")
    
    print("\n2. KEY DIFFERENCES")
    print("-" * 60)
    print("""
    ConvNeXt features are:
    - Learned from millions of images (ImageNet pretraining)
    - Hierarchically refined through 18 blocks
    - Semantically meaningful (object-aware)
    
    Geometric features are:
    - Computed from first principles (Gabor, position, stats)
    - Single-pass extraction (no refinement)
    - Texture/position-aware but NOT semantic
    
    The gap: ConvNeXt has learned to recognize OBJECTS.
    Geometric features only capture APPEARANCE.
    """)


if __name__ == "__main__":
    visualize_feature_maps()
    analyze_semantic_clustering()
    compare_geometric_vs_convnext()
