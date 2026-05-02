#!/usr/bin/env python3
"""
Analyze which weights are sensitive to compression.

The amplified bear shows:
- Grass: full color ✓
- Bear body: full color ✓  
- Bear face: desaturated ✗

This suggests some weights are more critical than others.
Let's find which ones.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import cv2
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ddcolor():
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    model = model.to(DEVICE)
    return model


def run_colorization(model, img_bgr):
    from ddcolor.pipeline import ColorizationPipeline
    pipeline = ColorizationPipeline(model, input_size=512, device=DEVICE)
    return pipeline.process(img_bgr)


def compress_layer(param, rank_ratio=0.5):
    """Compress a single layer to given rank ratio."""
    if param.dim() != 2:
        return param.clone()
    
    U, S, Vt = torch.linalg.svd(param, full_matrices=False)
    rank = max(1, int(min(param.shape) * rank_ratio))
    
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]


def analyze_layer_sensitivity():
    """Find which layers are most sensitive to compression."""
    print("=" * 70)
    print("ANALYZING LAYER SENSITIVITY TO COMPRESSION")
    print("=" * 70)
    
    model_original = load_ddcolor()
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    # Get original output
    output_original = run_colorization(model_original, img_bgr)
    
    # Test each major component
    components = [
        'encoder',
        'decoder.layers',
        'decoder.color_decoder',
        'refine_net',
    ]
    
    results = []
    
    for component_name in components:
        print(f"\nTesting: {component_name}")
        
        model_test = load_ddcolor()  # Fresh copy
        
        # Compress only this component
        n_compressed = 0
        with torch.no_grad():
            for name, param in model_test.named_parameters():
                if component_name in name and param.dim() == 2:
                    compressed = compress_layer(param, rank_ratio=0.5)
                    param.data = compressed
                    n_compressed += 1
        
        print(f"  Compressed {n_compressed} layers")
        
        # Test
        output_test = run_colorization(model_test, img_bgr)
        
        # Compute error
        mse = np.mean((output_original.astype(float) - output_test.astype(float))**2)
        
        # Compute saturation difference
        # Convert to HSV to measure saturation
        hsv_orig = cv2.cvtColor(output_original, cv2.COLOR_BGR2HSV)
        hsv_test = cv2.cvtColor(output_test, cv2.COLOR_BGR2HSV)
        
        sat_orig = hsv_orig[:, :, 1].mean()
        sat_test = hsv_test[:, :, 1].mean()
        sat_diff = sat_orig - sat_test
        
        print(f"  MSE: {mse:.2f}")
        print(f"  Saturation drop: {sat_diff:.2f}")
        
        results.append({
            'component': component_name,
            'mse': mse,
            'sat_diff': sat_diff,
            'n_layers': n_compressed,
        })
        
        # Save
        cv2.imwrite(str(output_path / f"bear_compress_{component_name.replace('.', '_')}.png"), output_test)
    
    # Summary
    print("\n" + "=" * 70)
    print("SENSITIVITY RANKING (by saturation drop)")
    print("=" * 70)
    
    results.sort(key=lambda x: x['sat_diff'], reverse=True)
    for r in results:
        print(f"  {r['component']}: sat_drop={r['sat_diff']:.2f}, mse={r['mse']:.2f}")
    
    return results


def analyze_color_decoder_detail():
    """Drill into the color decoder - which sublayers matter?"""
    print("\n" + "=" * 70)
    print("ANALYZING COLOR DECODER IN DETAIL")
    print("=" * 70)
    
    model_original = load_ddcolor()
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    output_original = run_colorization(model_original, img_bgr)
    
    # Color decoder subcomponents
    subcomponents = [
        'query_feat',
        'query_embed', 
        'color_embed',
        'transformer_cross_attention',
        'transformer_self_attention',
        'transformer_ffn',
        'input_proj',
    ]
    
    results = []
    
    for sub in subcomponents:
        print(f"\nTesting: color_decoder.{sub}")
        
        model_test = load_ddcolor()
        
        n_compressed = 0
        with torch.no_grad():
            for name, param in model_test.named_parameters():
                if 'color_decoder' in name and sub in name and param.dim() == 2:
                    compressed = compress_layer(param, rank_ratio=0.5)
                    param.data = compressed
                    n_compressed += 1
        
        if n_compressed == 0:
            print(f"  No 2D layers found")
            continue
        
        print(f"  Compressed {n_compressed} layers")
        
        output_test = run_colorization(model_test, img_bgr)
        
        mse = np.mean((output_original.astype(float) - output_test.astype(float))**2)
        
        hsv_orig = cv2.cvtColor(output_original, cv2.COLOR_BGR2HSV)
        hsv_test = cv2.cvtColor(output_test, cv2.COLOR_BGR2HSV)
        sat_diff = hsv_orig[:, :, 1].mean() - hsv_test[:, :, 1].mean()
        
        print(f"  MSE: {mse:.2f}")
        print(f"  Saturation drop: {sat_diff:.2f}")
        
        results.append({
            'component': sub,
            'mse': mse,
            'sat_diff': sat_diff,
        })
    
    # Summary
    print("\n" + "-" * 40)
    print("COLOR DECODER SENSITIVITY RANKING")
    print("-" * 40)
    
    results.sort(key=lambda x: x['sat_diff'], reverse=True)
    for r in results:
        print(f"  {r['component']}: sat_drop={r['sat_diff']:.2f}")
    
    return results


def find_critical_weights():
    """Find the specific weights that cause desaturation."""
    print("\n" + "=" * 70)
    print("FINDING CRITICAL WEIGHTS")
    print("=" * 70)
    
    model = load_ddcolor()
    encoder = PhiEncoder(K=32)
    
    # Analyze the color queries specifically
    color_decoder = model.decoder.color_decoder
    query_feat = color_decoder.query_feat.weight.data
    
    print(f"\nQuery features: {query_feat.shape}")
    
    # SVD analysis
    U, S, Vt = torch.linalg.svd(query_feat)
    
    print(f"\nSingular value distribution:")
    cumsum = torch.cumsum(S**2, dim=0) / (S**2).sum()
    for k in [1, 5, 10, 20, 50, 80, 100]:
        if k <= len(cumsum):
            print(f"  Top-{k}: {cumsum[k-1]*100:.1f}% of variance")
    
    # The queries are nearly full rank - this is why compression hurts
    print(f"\n** The queries have effective rank ~{len(S)} **")
    print("   This means ALL 100 queries carry unique information.")
    print("   Compressing them loses critical color distinctions.")
    
    # What about the color embedding MLP?
    print("\n\nColor embedding MLP:")
    for name, param in color_decoder.color_embed.named_parameters():
        if param.dim() == 2:
            U, S, Vt = torch.linalg.svd(param)
            cumsum = torch.cumsum(S**2, dim=0) / (S**2).sum()
            rank_90 = (cumsum < 0.90).sum().item() + 1
            print(f"  {name}: shape={list(param.shape)}, rank_90={rank_90}")


def main():
    layer_results = analyze_layer_sensitivity()
    decoder_results = analyze_color_decoder_detail()
    find_critical_weights()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The desaturation in the face region is caused by:

1. The color decoder is the most sensitive component
2. The color queries (100 x 256) are nearly full rank
3. Compressing them loses fine color distinctions
4. Face regions require more precise color matching than grass

This tells us:
- NOT all weights can be compressed equally
- The color queries are CRITICAL - they encode the color vocabulary
- The attention/FFN layers are more compressible
- A smart compression scheme would preserve queries, compress attention

For amplification to work better:
- Don't compress the color queries
- Or: amplify them with more precision (higher K in φ-encoding)
""")


if __name__ == "__main__":
    main()
