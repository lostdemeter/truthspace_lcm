#!/usr/bin/env python3
"""
Examine Dedicated Colorization Models

Goal: Understand what makes colorization work by examining models built for it.

We'll look at:
1. DDColor (ICCV 2023) - State of the art
2. Architecture differences from our approach
3. What "color queries" are and how they work
4. φ-structure in their weights (if any)

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


def examine_ddcolor():
    """
    Examine DDColor architecture and weights.
    """
    print("=" * 70)
    print("EXAMINING DDCOLOR")
    print("State-of-the-art colorization model")
    print("=" * 70)
    
    try:
        from huggingface_hub import hf_hub_download
        import torch
        
        print("\n1. DOWNLOADING MODEL")
        print("-" * 50)
        
        # DDColor is available via HuggingFace
        # Try the artistic model first
        model_path = hf_hub_download(
            repo_id="piddnad/ddcolor_paper_tiny",
            filename="ddcolor_paper_tiny.pth"
        )
        print(f"   Downloaded: {model_path}")
        
        print("\n2. LOADING WEIGHTS")
        print("-" * 50)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"   Keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'raw state_dict'}")
        print(f"   Number of parameters: {len(state_dict)}")
        
        # Analyze architecture from weight names
        print("\n3. ARCHITECTURE ANALYSIS")
        print("-" * 50)
        
        # Group by module
        modules = {}
        for key in state_dict.keys():
            parts = key.split('.')
            module = parts[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(key)
        
        print("   Modules found:")
        for module, keys in sorted(modules.items()):
            total_params = sum(state_dict[k].numel() for k in keys)
            print(f"     {module}: {len(keys)} layers, {total_params:,} params")
        
        # Look for color-specific components
        print("\n4. COLOR-SPECIFIC COMPONENTS")
        print("-" * 50)
        
        color_keys = [k for k in state_dict.keys() if 'color' in k.lower() or 'query' in k.lower() or 'decoder' in k.lower()]
        print(f"   Found {len(color_keys)} color-related keys:")
        for key in color_keys[:20]:
            shape = state_dict[key].shape
            print(f"     {key}: {shape}")
        
        # Analyze weight distributions
        print("\n5. WEIGHT DISTRIBUTION ANALYSIS")
        print("-" * 50)
        
        all_weights = []
        for key, tensor in state_dict.items():
            if tensor.dtype in [torch.float32, torch.float16]:
                all_weights.extend(tensor.flatten().numpy().tolist())
        
        all_weights = np.array(all_weights)
        print(f"   Total weights: {len(all_weights):,}")
        print(f"   Mean: {all_weights.mean():.6f}")
        print(f"   Std: {all_weights.std():.6f}")
        print(f"   Min: {all_weights.min():.6f}")
        print(f"   Max: {all_weights.max():.6f}")
        
        # Check φ-structure
        print("\n6. φ-STRUCTURE ANALYSIS")
        print("-" * 50)
        
        # Quantize to φ-levels
        def to_phi_level(v, k=32):
            if abs(v) < 1e-10:
                return 0
            return int(round(k * np.log(abs(v)) / LN_PHI))
        
        # Sample weights for analysis
        sample_size = min(1000000, len(all_weights))
        sample = np.random.choice(all_weights, sample_size, replace=False)
        
        levels = np.array([to_phi_level(w) for w in sample])
        
        unique_levels, counts = np.unique(levels, return_counts=True)
        print(f"   Unique φ-levels: {len(unique_levels)}")
        print(f"   Level range: [{levels.min()}, {levels.max()}]")
        
        # Most common levels
        top_idx = np.argsort(counts)[-10:]
        print(f"   Most common levels:")
        for idx in top_idx[::-1]:
            level = unique_levels[idx]
            count = counts[idx]
            phi_exp = level / 32
            print(f"     φ^{phi_exp:.2f}: {count} ({100*count/len(levels):.1f}%)")
        
        # Check if level differences follow Fibonacci
        FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        # Sample pairs and check differences
        n_pairs = 100000
        idx1 = np.random.choice(len(levels), n_pairs)
        idx2 = np.random.choice(len(levels), n_pairs)
        diffs = np.abs(levels[idx1] - levels[idx2])
        
        fib_exact = sum(1 for d in diffs if d in FIBONACCI)
        fib_near = sum(1 for d in diffs if any(abs(d - f) <= 1 for f in FIBONACCI))
        
        print(f"\n   Level difference analysis ({n_pairs} pairs):")
        print(f"     Exact Fibonacci: {fib_exact} ({100*fib_exact/n_pairs:.1f}%)")
        print(f"     Near Fibonacci (±1): {fib_near} ({100*fib_near/n_pairs:.1f}%)")
        
        return state_dict, all_weights
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def examine_deoldify():
    """
    Examine DeOldify - another popular colorization model.
    """
    print("\n" + "=" * 70)
    print("EXAMINING DEOLDIFY")
    print("Popular colorization model")
    print("=" * 70)
    
    try:
        from huggingface_hub import hf_hub_download
        import torch
        
        print("\n1. DOWNLOADING MODEL")
        print("-" * 50)
        
        # Try to find DeOldify on HuggingFace
        model_path = hf_hub_download(
            repo_id="leonelhs/deoldify",
            filename="ColorizeArtistic_gen.pth"
        )
        print(f"   Downloaded: {model_path}")
        
        print("\n2. LOADING WEIGHTS")
        print("-" * 50)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else {}
        
        print(f"   Number of parameters: {len(state_dict)}")
        
        # Analyze architecture
        print("\n3. ARCHITECTURE ANALYSIS")
        print("-" * 50)
        
        modules = {}
        for key in state_dict.keys():
            parts = key.split('.')
            module = parts[0] if parts else 'unknown'
            if module not in modules:
                modules[module] = []
            modules[module].append(key)
        
        print("   Modules found:")
        for module, keys in sorted(modules.items())[:10]:
            total_params = sum(state_dict[k].numel() for k in keys if k in state_dict)
            print(f"     {module}: {len(keys)} layers")
        
        return state_dict
        
    except Exception as e:
        print(f"   Error: {e}")
        return None


def compare_architectures():
    """
    Compare what colorization models do vs what we're doing.
    """
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    print("""
   OUR APPROACH (DA2-based):
   -------------------------
   1. Extract features from DA2 backbone (384 dims)
   2. Linear regression: features → U, V
   3. Upsample U, V to full resolution
   4. Combine with grayscale Y
   
   Problem: Each image uses different dimensions
   Result: MAE ~12-14 (cross-image), ~4 (single-image)
   
   DDCOLOR APPROACH:
   -----------------
   1. Encoder: Extract multi-scale features
   2. Color Decoder: Learnable "color queries" (tokens)
   3. Cross-attention: Queries attend to image features
   4. Pixel Decoder: Upsample to full resolution
   
   Key insight: COLOR QUERIES are learned embeddings that
   represent "what colors look like" in feature space.
   
   DEOLDIFY APPROACH:
   ------------------
   1. U-Net with ResNet backbone
   2. Self-attention at multiple scales
   3. Perceptual loss (not just pixel loss)
   4. GAN discriminator for realism
   
   Key insight: PERCEPTUAL LOSS captures "looks right"
   rather than "matches exactly".
   
   WHAT WE'RE MISSING:
   -------------------
   1. Learnable color representations (queries/embeddings)
   2. Attention mechanism to match features to colors
   3. Multi-scale processing
   4. Perceptual/adversarial training
   
   The key difference: They LEARN what colors look like.
   We're trying to DERIVE colors from features that weren't
   designed to encode color.
""")


def run_examination():
    """Run full examination of colorization models."""
    
    # Examine DDColor
    ddcolor_weights, ddcolor_all = examine_ddcolor()
    
    # Examine DeOldify
    deoldify_weights = examine_deoldify()
    
    # Compare architectures
    compare_architectures()
    
    return {
        'ddcolor': ddcolor_weights,
        'deoldify': deoldify_weights
    }


if __name__ == "__main__":
    results = run_examination()
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    print("""
   WHAT MAKES COLORIZATION WORK:
   
   1. LEARNABLE COLOR QUERIES
      - DDColor uses "color tokens" that learn to represent colors
      - These are NOT derived from image features
      - They're learned embeddings that encode "what red looks like"
   
   2. CROSS-ATTENTION
      - Image features attend to color queries
      - This finds "which color query matches this region"
      - Attention is the key mechanism, not linear regression
   
   3. MULTI-SCALE FEATURES
      - Low-res: semantic understanding (sky, grass, skin)
      - High-res: edge preservation
      - Both are needed for good colorization
   
   4. PERCEPTUAL TRAINING
      - Not just "match the pixels"
      - "Does this LOOK like a plausible colorization?"
      - GAN discriminators help with realism
   
   WHAT WE NEED TO DO:
   
   Option A: Build a geometric version of color queries
      - φ-positioned color embeddings
      - Geometric attention (distance-based, not learned)
      - This would be a TRUE geometric colorizer
   
   Option B: Use existing colorizer, analyze its φ-structure
      - Run DDColor, examine intermediate representations
      - See if φ-structure emerges in learned color queries
      - Validate hypothesis that "learning = finding φ-structure"
   
   Option C: Hybrid approach
      - Use DDColor's color queries as our "drum"
      - Apply φ-arithmetic to the attention mechanism
      - See if we can replicate results with pure geometry
""")
