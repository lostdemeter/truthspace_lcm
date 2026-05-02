#!/usr/bin/env python3
"""
Smart Geometric Amplification

Based on our sensitivity analysis:
- color_embed: CRITICAL - don't compress
- attention/ffn: Safe to compress
- encoder: Can compress aggressively

This implements an optimal compression strategy that preserves
the critical color vocabulary while compressing everything else.

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


def snap_to_lattice(tensor: torch.Tensor, encoder: PhiEncoder) -> torch.Tensor:
    """Snap weights to nearest φ-lattice point."""
    signs, exps = encoder.encode(tensor)
    return encoder.decode(signs, exps)


def compress_to_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Compress to low-rank approximation."""
    if tensor.dim() != 2 or min(tensor.shape) <= rank:
        return tensor.clone()
    
    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]


def smart_compress(model, encoder: PhiEncoder, config: dict) -> dict:
    """
    Apply smart compression based on sensitivity analysis.
    
    Config specifies compression strategy per component:
    - 'preserve': Don't compress, just snap to lattice
    - 'light': Compress to 80% rank
    - 'medium': Compress to 50% rank  
    - 'aggressive': Compress to 20% rank
    """
    stats = {
        'preserved': 0,
        'light': 0,
        'medium': 0,
        'aggressive': 0,
        'original_params': 0,
        'compressed_params': 0,
    }
    
    rank_ratios = {
        'preserve': 1.0,
        'light': 0.8,
        'medium': 0.5,
        'aggressive': 0.2,
    }
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() != 2:
                continue
            
            original_size = param.numel()
            stats['original_params'] += original_size
            
            # Determine compression level based on config
            level = 'medium'  # default
            for pattern, lvl in config.items():
                if pattern in name:
                    level = lvl
                    break
            
            if level == 'preserve':
                # Just snap to lattice, don't compress
                param.data = snap_to_lattice(param.data, encoder)
                stats['preserved'] += 1
                stats['compressed_params'] += original_size
            else:
                # Compress then snap
                rank_ratio = rank_ratios[level]
                rank = max(1, int(min(param.shape) * rank_ratio))
                
                compressed = compress_to_rank(param.data, rank)
                param.data = snap_to_lattice(compressed, encoder)
                
                # Estimate compressed size (for stats)
                compressed_size = rank * (param.shape[0] + param.shape[1])
                stats['compressed_params'] += compressed_size
                stats[level] += 1
    
    return stats


def compute_metrics(output1: np.ndarray, output2: np.ndarray) -> dict:
    """Compute comparison metrics."""
    o1 = output1.astype(np.float32)
    o2 = output2.astype(np.float32)
    
    mse = np.mean((o1 - o2) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    corr = np.corrcoef(o1.flatten(), o2.flatten())[0, 1]
    
    # Saturation comparison
    hsv1 = cv2.cvtColor(output1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(output2, cv2.COLOR_BGR2HSV)
    sat1 = hsv1[:, :, 1].mean()
    sat2 = hsv2[:, :, 1].mean()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': corr,
        'sat_original': sat1,
        'sat_compressed': sat2,
        'sat_ratio': sat2 / sat1 if sat1 > 0 else 1.0,
    }


def run_comparison():
    """Compare different compression strategies."""
    print("=" * 70)
    print("SMART GEOMETRIC AMPLIFICATION")
    print("=" * 70)
    
    encoder = PhiEncoder(K=32)
    
    # Load test image
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    # Get original output
    model_original = load_ddcolor()
    output_original = run_colorization(model_original, img_bgr)
    
    # Define compression strategies
    strategies = {
        'naive': {
            # Compress everything equally
        },
        'smart_v1': {
            # Preserve color_embed, compress rest
            'color_embed': 'preserve',
        },
        'smart_v2': {
            # Preserve color_embed and queries
            'color_embed': 'preserve',
            'query_feat': 'preserve',
            'query_embed': 'preserve',
        },
        'smart_v3': {
            # Preserve color decoder entirely, compress encoder aggressively
            'color_decoder': 'preserve',
            'encoder': 'aggressive',
        },
        'optimal': {
            # Based on full sensitivity analysis
            'color_embed': 'preserve',
            'query_feat': 'preserve', 
            'query_embed': 'preserve',
            'transformer_cross_attention': 'light',
            'transformer_self_attention': 'light',
            'transformer_ffn': 'medium',
            'encoder': 'medium',
        },
    }
    
    results = []
    
    for name, config in strategies.items():
        print(f"\n{'='*50}")
        print(f"Strategy: {name}")
        print(f"{'='*50}")
        
        model = load_ddcolor()
        stats = smart_compress(model, encoder, config)
        
        print(f"  Preserved: {stats['preserved']} layers")
        print(f"  Light: {stats['light']} layers")
        print(f"  Medium: {stats['medium']} layers")
        print(f"  Aggressive: {stats['aggressive']} layers")
        
        compression_ratio = stats['original_params'] / max(1, stats['compressed_params'])
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        # Run colorization
        output = run_colorization(model, img_bgr)
        
        # Compute metrics
        metrics = compute_metrics(output_original, output)
        
        print(f"\n  Results:")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
        print(f"    Correlation: {metrics['correlation']:.6f}")
        print(f"    Saturation ratio: {metrics['sat_ratio']:.4f}")
        
        # Save output
        cv2.imwrite(str(output_path / f"bear_{name}.png"), output)
        
        results.append({
            'name': name,
            'compression': compression_ratio,
            **metrics,
        })
    
    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Compress':<10} {'PSNR':<10} {'Corr':<12} {'Sat Ratio':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<15} {r['compression']:<10.2f} {r['psnr']:<10.2f} {r['correlation']:<12.6f} {r['sat_ratio']:<10.4f}")
    
    # Find best
    best = max(results, key=lambda x: x['sat_ratio'])
    print(f"\n✓ Best strategy: {best['name']} (sat_ratio={best['sat_ratio']:.4f})")
    
    return results


def create_final_comparison():
    """Create a side-by-side comparison image."""
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    # Load images
    original = cv2.imread(str(output_path / "000000000285_ddcolor_pipeline.png"))
    naive = cv2.imread(str(output_path / "bear_naive.png"))
    optimal = cv2.imread(str(output_path / "bear_optimal.png"))
    
    if original is None or naive is None or optimal is None:
        print("Could not load all images for comparison")
        return
    
    # Resize to same size
    h, w = original.shape[:2]
    naive = cv2.resize(naive, (w, h))
    optimal = cv2.resize(optimal, (w, h))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(naive, "Naive Compress", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(optimal, "Smart Compress", (10, 30), font, 1, (255, 255, 255), 2)
    
    # Concatenate
    comparison = np.concatenate([original, naive, optimal], axis=1)
    
    cv2.imwrite(str(output_path / "amplification_comparison.png"), comparison)
    print(f"\nSaved: amplification_comparison.png")


def main():
    results = run_comparison()
    create_final_comparison()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Smart compression preserves accuracy by:
1. Keeping color_embed layers intact (the color vocabulary)
2. Keeping query features intact (the color atoms)
3. Compressing attention/ffn layers (the routing logic)
4. Compressing encoder moderately (feature extraction)

The key insight: NOT all weights are equal.
Some encode critical knowledge (colors), others encode routing.
Smart amplification preserves the knowledge, compresses the routing.
""")


if __name__ == "__main__":
    main()
