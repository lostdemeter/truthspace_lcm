#!/usr/bin/env python3
"""
Extract DDColor's actual query→color mapping by observing outputs.

Strategy:
1. Run DDColor on many images
2. For each pixel, record which query had highest attention
3. Record the ab color at that pixel
4. Build a mapping: query_id → average (a, b) color

This gives us the ACTUAL learned color vocabulary.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
import sys
from collections import defaultdict
import json

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


def extract_query_colors(n_images: int = 50):
    """Extract the color mapping from DDColor."""
    from ddcolor import DDColor
    from ddcolor.pipeline import ColorizationPipeline
    from huggingface_hub import PyTorchModelHubMixin
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading DDColor...")
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    model = model.to(device)
    
    # Get query embeddings
    query_embed = model.decoder.color_decoder.query_embed.weight.detach()
    
    # Storage for colors per query
    query_colors = {i: {'a': [], 'b': []} for i in range(100)}
    
    # Load COCO images
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    image_paths = list(coco_path.glob("*.jpg"))[:n_images]
    
    print(f"Processing {len(image_paths)} images...")
    
    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(image_paths)}")
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        # Preprocess
        input_size = 512
        img_resized = cv2.resize(img_bgr, (input_size, input_size))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        
        tensor = torch.from_numpy(gray_3ch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
        
        # Hook to capture features and attention
        captured = {}
        
        def hook_fn(module, input, output):
            captured['features'] = input[1].detach()  # [1, 256, H, W]
        
        hook = model.decoder.color_decoder.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            # Run model
            ab_output = model(tensor)  # [1, 2, H, W]
            
            # Get features
            features = captured['features']  # [1, 256, H, W]
            
            # Compute attention manually
            B, C, H, W = features.shape
            feat_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            # Attention scores
            scores = torch.matmul(feat_flat, query_embed.T)  # [1, H*W, 100]
            attention = F.softmax(scores / 0.07, dim=-1)
            
            # Get dominant query per pixel
            dominant_query = attention.argmax(dim=-1)  # [1, H*W]
            
            # Get ab values
            ab = ab_output[0].cpu().numpy()  # [2, H, W]
            a_channel = ab[0].flatten()  # [H*W]
            b_channel = ab[1].flatten()
            
            # Record colors for each query
            for pixel_idx in range(H * W):
                q = dominant_query[0, pixel_idx].item()
                query_colors[q]['a'].append(float(a_channel[pixel_idx]))
                query_colors[q]['b'].append(float(b_channel[pixel_idx]))
        
        hook.remove()
    
    # Compute statistics
    print("\nComputing query color statistics...")
    results = {}
    
    for q in range(100):
        a_vals = query_colors[q]['a']
        b_vals = query_colors[q]['b']
        
        if len(a_vals) > 0:
            results[q] = {
                'mean_a': float(np.mean(a_vals)),
                'mean_b': float(np.mean(b_vals)),
                'std_a': float(np.std(a_vals)),
                'std_b': float(np.std(b_vals)),
                'n_samples': len(a_vals),
            }
        else:
            results[q] = {
                'mean_a': 0.0,
                'mean_b': 0.0,
                'std_a': 0.0,
                'std_b': 0.0,
                'n_samples': 0,
            }
    
    return results


def visualize_query_colors(results: dict):
    """Visualize the extracted color vocabulary."""
    print("\n" + "=" * 60)
    print("EXTRACTED QUERY COLOR VOCABULARY")
    print("=" * 60)
    
    # Sort by saturation (distance from neutral)
    def saturation(r):
        return np.sqrt(r['mean_a']**2 + r['mean_b']**2)
    
    sorted_queries = sorted(results.items(), key=lambda x: saturation(x[1]), reverse=True)
    
    print("\nTop 20 most saturated queries:")
    for q, r in sorted_queries[:20]:
        sat = saturation(r)
        # Determine color name
        a, b = r['mean_a'], r['mean_b']
        if sat < 0.05:
            color = "gray"
        elif a > 0.1 and b > 0.1:
            color = "orange/yellow"
        elif a > 0.1 and b < -0.1:
            color = "magenta/pink"
        elif a < -0.1 and b > 0.1:
            color = "lime/yellow-green"
        elif a < -0.1 and b < -0.1:
            color = "cyan/teal"
        elif a > 0.1:
            color = "red"
        elif a < -0.1:
            color = "green"
        elif b > 0.1:
            color = "yellow"
        elif b < -0.1:
            color = "blue"
        else:
            color = "neutral"
        
        print(f"  Query {q:2d}: a={a:+.3f}, b={b:+.3f}, sat={sat:.3f}, n={r['n_samples']:6d} → {color}")
    
    print("\nBottom 10 (most neutral):")
    for q, r in sorted_queries[-10:]:
        sat = saturation(r)
        print(f"  Query {q:2d}: a={r['mean_a']:+.3f}, b={r['mean_b']:+.3f}, sat={sat:.3f}, n={r['n_samples']:6d}")
    
    # Create color tensor for use in V5
    color_tensor = torch.zeros(100, 2)
    for q, r in results.items():
        color_tensor[q, 0] = r['mean_a']
        color_tensor[q, 1] = r['mean_b']
    
    return color_tensor


def main():
    results = extract_query_colors(n_images=50)
    
    color_tensor = visualize_query_colors(results)
    
    # Save results
    output_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/extracted_query_colors.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Save tensor
    tensor_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/query_color_tensor.pt")
    torch.save(color_tensor, tensor_path)
    print(f"Saved tensor to: {tensor_path}")
    
    return results, color_tensor


if __name__ == "__main__":
    results, color_tensor = main()
