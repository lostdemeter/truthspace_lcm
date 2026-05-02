#!/usr/bin/env python3
"""
Analyze the Remaining 3% Saturation Error

The optimal compression achieves sat_ratio=1.03, meaning 3% saturation difference.
Where is this error coming from?

1. Which pixels/regions have the most error?
2. Which compressed layers contribute most?
3. What's the minimum compression for 100% saturation match?

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


def snap_to_lattice(tensor, encoder):
    signs, exps = encoder.encode(tensor)
    return encoder.decode(signs, exps)


def compress_to_rank(tensor, rank):
    if tensor.dim() != 2 or min(tensor.shape) <= rank:
        return tensor.clone()
    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]


# ============================================================================
# ANALYSIS 1: Spatial Error Distribution
# ============================================================================

def analyze_spatial_error():
    """Where in the image is the 3% error?"""
    print("=" * 70)
    print("ANALYSIS 1: SPATIAL ERROR DISTRIBUTION")
    print("=" * 70)
    
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    # Load the outputs
    original = cv2.imread(str(output_path / "000000000285_ddcolor_pipeline.png"))
    optimal = cv2.imread(str(output_path / "bear_optimal.png"))
    
    # Resize to match
    h, w = original.shape[:2]
    optimal = cv2.resize(optimal, (w, h))
    
    # Convert to HSV
    hsv_orig = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_opt = cv2.cvtColor(optimal, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Saturation difference
    sat_diff = hsv_opt[:, :, 1] - hsv_orig[:, :, 1]
    
    print(f"\nSaturation difference statistics:")
    print(f"  Mean: {sat_diff.mean():.2f}")
    print(f"  Std: {sat_diff.std():.2f}")
    print(f"  Min: {sat_diff.min():.2f}")
    print(f"  Max: {sat_diff.max():.2f}")
    
    # Where is the error concentrated?
    abs_diff = np.abs(sat_diff)
    
    # Divide into regions
    h3, w3 = h // 3, w // 3
    regions = {
        'top-left': abs_diff[:h3, :w3],
        'top-center': abs_diff[:h3, w3:2*w3],
        'top-right': abs_diff[:h3, 2*w3:],
        'mid-left': abs_diff[h3:2*h3, :w3],
        'mid-center': abs_diff[h3:2*h3, w3:2*w3],
        'mid-right': abs_diff[h3:2*h3, 2*w3:],
        'bot-left': abs_diff[2*h3:, :w3],
        'bot-center': abs_diff[2*h3:, w3:2*w3],
        'bot-right': abs_diff[2*h3:, 2*w3:],
    }
    
    print(f"\nError by region (mean abs saturation diff):")
    for name, region in sorted(regions.items(), key=lambda x: x[1].mean(), reverse=True):
        print(f"  {name}: {region.mean():.2f}")
    
    # Create error heatmap
    error_map = (abs_diff / abs_diff.max() * 255).astype(np.uint8)
    error_colored = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path / "saturation_error_heatmap.png"), error_colored)
    print(f"\nSaved: saturation_error_heatmap.png")
    
    # Also analyze by luminance
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Bin by luminance
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 255)]
    print(f"\nError by luminance:")
    for low, high in bins:
        mask = (gray >= low) & (gray < high)
        if mask.sum() > 0:
            mean_err = abs_diff[mask].mean()
            print(f"  L={low}-{high}: {mean_err:.2f} (n={mask.sum()})")
    
    return sat_diff


# ============================================================================
# ANALYSIS 2: Per-Layer Error Contribution
# ============================================================================

def analyze_layer_contribution():
    """Which compressed layers contribute most to the 3% error?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: PER-LAYER ERROR CONTRIBUTION")
    print("=" * 70)
    
    encoder = PhiEncoder(K=32)
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    # Get original output
    model_original = load_ddcolor()
    output_original = run_colorization(model_original, img_bgr)
    hsv_orig = cv2.cvtColor(output_original, cv2.COLOR_BGR2HSV)
    sat_orig = hsv_orig[:, :, 1].mean()
    
    # The optimal config
    optimal_config = {
        'color_embed': 'preserve',
        'query_feat': 'preserve',
        'query_embed': 'preserve',
        'transformer_cross_attention': 'light',
        'transformer_self_attention': 'light',
        'transformer_ffn': 'medium',
        'encoder': 'medium',
    }
    
    # Test removing each compression one at a time
    components_to_test = [
        'transformer_cross_attention',
        'transformer_self_attention', 
        'transformer_ffn',
        'encoder',
    ]
    
    results = []
    
    for component in components_to_test:
        print(f"\nTesting: preserve {component} (instead of compress)")
        
        # Create config with this component preserved
        test_config = optimal_config.copy()
        test_config[component] = 'preserve'
        
        model = load_ddcolor()
        
        # Apply compression
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() != 2:
                    continue
                
                level = 'medium'  # default
                for pattern, lvl in test_config.items():
                    if pattern in name:
                        level = lvl
                        break
                
                if level == 'preserve':
                    param.data = snap_to_lattice(param.data, encoder)
                elif level == 'light':
                    rank = max(1, int(min(param.shape) * 0.8))
                    compressed = compress_to_rank(param.data, rank)
                    param.data = snap_to_lattice(compressed, encoder)
                else:  # medium
                    rank = max(1, int(min(param.shape) * 0.5))
                    compressed = compress_to_rank(param.data, rank)
                    param.data = snap_to_lattice(compressed, encoder)
        
        output = run_colorization(model, img_bgr)
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        
        sat_ratio = sat / sat_orig
        print(f"  Saturation ratio: {sat_ratio:.4f}")
        
        results.append({
            'component': component,
            'sat_ratio': sat_ratio,
            'improvement': abs(1.0 - sat_ratio),
        })
    
    # Summary
    print("\n" + "-" * 40)
    print("ERROR CONTRIBUTION RANKING")
    print("-" * 40)
    
    results.sort(key=lambda x: x['improvement'])
    for r in results:
        contribution = (1.03 - r['sat_ratio']) / 0.03 * 100 if r['sat_ratio'] < 1.03 else 0
        print(f"  {r['component']}: sat_ratio={r['sat_ratio']:.4f}, contributes ~{contribution:.0f}% of error")
    
    return results


# ============================================================================
# ANALYSIS 3: Find Minimum Compression for 100% Match
# ============================================================================

def find_minimum_compression():
    """What's the minimum compression that achieves 100% saturation match?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: MINIMUM COMPRESSION FOR 100% MATCH")
    print("=" * 70)
    
    encoder = PhiEncoder(K=32)
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    model_original = load_ddcolor()
    output_original = run_colorization(model_original, img_bgr)
    hsv_orig = cv2.cvtColor(output_original, cv2.COLOR_BGR2HSV)
    sat_orig = hsv_orig[:, :, 1].mean()
    
    # Test different rank ratios
    rank_ratios = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    
    print("\nTesting rank ratios for attention/ffn layers:")
    
    for ratio in rank_ratios:
        model = load_ddcolor()
        
        n_compressed = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() != 2:
                    continue
                
                # Preserve color vocabulary
                if any(p in name for p in ['color_embed', 'query_feat', 'query_embed']):
                    param.data = snap_to_lattice(param.data, encoder)
                else:
                    # Compress with given ratio
                    rank = max(1, int(min(param.shape) * ratio))
                    if rank < min(param.shape):
                        compressed = compress_to_rank(param.data, rank)
                        param.data = snap_to_lattice(compressed, encoder)
                        n_compressed += 1
        
        output = run_colorization(model, img_bgr)
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        sat_ratio = sat / sat_orig
        
        status = "✓" if abs(1.0 - sat_ratio) < 0.01 else " "
        print(f"  {status} ratio={ratio:.2f}: sat_ratio={sat_ratio:.4f}, compressed={n_compressed} layers")
        
        if abs(1.0 - sat_ratio) < 0.005:
            print(f"\n  Found: ratio={ratio} achieves <0.5% error!")
            break
    
    # Also test: what if we ONLY snap to lattice, no rank compression?
    print("\n\nTest: Lattice snap only (no rank compression):")
    model = load_ddcolor()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() == 2:
                param.data = snap_to_lattice(param.data, encoder)
    
    output = run_colorization(model, img_bgr)
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean()
    sat_ratio = sat / sat_orig
    
    print(f"  Lattice snap only: sat_ratio={sat_ratio:.4f}")
    
    if abs(1.0 - sat_ratio) < 0.01:
        print("  ✓ Lattice snapping alone achieves <1% error!")
        print("  The 3% error comes from RANK COMPRESSION, not lattice snapping.")


# ============================================================================
# ANALYSIS 4: Exact Error Source
# ============================================================================

def analyze_exact_error_source():
    """Pinpoint exactly where the 3% comes from."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: EXACT ERROR SOURCE")
    print("=" * 70)
    
    encoder = PhiEncoder(K=32)
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    model_original = load_ddcolor()
    output_original = run_colorization(model_original, img_bgr)
    hsv_orig = cv2.cvtColor(output_original, cv2.COLOR_BGR2HSV)
    sat_orig = hsv_orig[:, :, 1].mean()
    
    # Test each layer group individually
    layer_groups = [
        ('encoder.stages', 'Encoder stages'),
        ('encoder.downsample', 'Encoder downsample'),
        ('decoder.layers', 'Decoder upsample'),
        ('color_decoder.transformer_cross', 'Cross attention'),
        ('color_decoder.transformer_self', 'Self attention'),
        ('color_decoder.transformer_ffn', 'FFN layers'),
        ('color_decoder.input_proj', 'Input projection'),
        ('refine_net', 'Refine net'),
    ]
    
    print("\nCompressing each layer group individually (50% rank):")
    
    for pattern, name in layer_groups:
        model = load_ddcolor()
        
        n_compressed = 0
        with torch.no_grad():
            for pname, param in model.named_parameters():
                if param.dim() != 2:
                    continue
                
                if pattern in pname:
                    rank = max(1, int(min(param.shape) * 0.5))
                    if rank < min(param.shape):
                        compressed = compress_to_rank(param.data, rank)
                        param.data = snap_to_lattice(compressed, encoder)
                        n_compressed += 1
                else:
                    # Just snap, don't compress
                    param.data = snap_to_lattice(param.data, encoder)
        
        if n_compressed == 0:
            continue
        
        output = run_colorization(model, img_bgr)
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        sat_ratio = sat / sat_orig
        
        error_pct = abs(1.0 - sat_ratio) * 100
        print(f"  {name}: sat_ratio={sat_ratio:.4f}, error={error_pct:.2f}%, layers={n_compressed}")


def main():
    sat_diff = analyze_spatial_error()
    layer_results = analyze_layer_contribution()
    find_minimum_compression()
    analyze_exact_error_source()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
