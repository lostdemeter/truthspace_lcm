#!/usr/bin/env python3
"""
ConvNeXt Architecture Analysis

Understand the structure of ConvNeXt to determine what we need to replace.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


def analyze_convnext():
    """Analyze ConvNeXt architecture from DDColor."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    
    print("=" * 60)
    print("CONVNEXT ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    # Get the ConvNeXt encoder
    encoder = model.encoder
    convnext = encoder.arch
    
    print("\n1. TOP-LEVEL STRUCTURE")
    print("-" * 40)
    for name, child in convnext.named_children():
        print(f"  {name}: {type(child).__name__}")
    
    print("\n2. DETAILED LAYER BREAKDOWN")
    print("-" * 40)
    
    # Count parameters
    total_params = 0
    layer_params = {}
    
    for name, module in convnext.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                total_params += params
                # Group by top-level
                top_level = name.split('.')[0] if '.' in name else name
                layer_params[top_level] = layer_params.get(top_level, 0) + params
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"\n  Parameters by section:")
    for name, params in sorted(layer_params.items(), key=lambda x: -x[1]):
        pct = 100 * params / total_params
        print(f"    {name}: {params:,} ({pct:.1f}%)")
    
    print("\n3. CONVNEXT BLOCK STRUCTURE")
    print("-" * 40)
    
    # Find a ConvNeXt block
    for name, module in convnext.named_modules():
        if 'Block' in type(module).__name__:
            print(f"\n  Block type: {type(module).__name__}")
            print(f"  Location: {name}")
            print(f"  Components:")
            for child_name, child in module.named_children():
                print(f"    {child_name}: {type(child).__name__}")
                if hasattr(child, 'weight'):
                    print(f"      weight: {child.weight.shape}")
            break
    
    print("\n4. STAGE STRUCTURE")
    print("-" * 40)
    
    # Analyze stages
    if hasattr(convnext, 'stages'):
        for i, stage in enumerate(convnext.stages):
            print(f"\n  Stage {i}:")
            print(f"    Type: {type(stage).__name__}")
            n_blocks = len(list(stage.children()))
            print(f"    Blocks: {n_blocks}")
            
            # Get first block's dimensions
            for block in stage.children():
                if hasattr(block, 'dwconv'):
                    print(f"    DWConv: {block.dwconv.weight.shape}")
                if hasattr(block, 'pwconv1'):
                    print(f"    PWConv1: {block.pwconv1.weight.shape}")
                if hasattr(block, 'pwconv2'):
                    print(f"    PWConv2: {block.pwconv2.weight.shape}")
                break
    
    print("\n5. FEATURE DIMENSIONS AT EACH STAGE")
    print("-" * 40)
    
    # Run a test image to see feature dimensions
    import cv2
    img = np.random.rand(512, 512, 3).astype(np.float32)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    
    # Normalize
    tensor = (tensor - model.mean) / model.std
    
    with torch.no_grad():
        encoder(tensor)
    
    for i, hook in enumerate(encoder.hooks):
        if hook.feature is not None:
            print(f"  Stage {i} output: {hook.feature.shape}")
    
    print("\n6. KEY OPERATIONS IN CONVNEXT")
    print("-" * 40)
    print("""
    ConvNeXt Block consists of:
    1. Depthwise Conv (7x7) - spatial mixing within each channel
    2. LayerNorm - normalization
    3. Pointwise Conv 1 (1x1) - channel expansion (4x)
    4. GELU activation - non-linearity
    5. Pointwise Conv 2 (1x1) - channel reduction
    6. Residual connection
    
    Key insight: This is essentially an inverted bottleneck with depthwise convolution.
    Similar to MobileNet but with larger kernels and different normalization.
    """)
    
    print("\n7. WHAT MAKES CONVNEXT WORK?")
    print("-" * 40)
    print("""
    ConvNeXt modernizes ResNet with ideas from Vision Transformers:
    
    1. LARGER KERNELS (7x7 vs 3x3)
       - Captures more context per layer
       - Similar receptive field to attention
    
    2. FEWER ACTIVATIONS
       - Only one GELU per block (vs multiple ReLUs)
       - Preserves more information
    
    3. LAYERNORM (vs BatchNorm)
       - More stable training
       - Works better with varying batch sizes
    
    4. INVERTED BOTTLENECK
       - Expand channels, process, contract
       - More expressive transformations
    
    5. STAGE RATIO [3, 3, 9, 3]
       - Most computation in stage 3
       - Matches ViT's layer distribution
    """)
    
    return convnext


def analyze_single_block():
    """Deep dive into a single ConvNeXt block."""
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    convnext = model.encoder.arch
    
    print("\n" + "=" * 60)
    print("SINGLE BLOCK ANALYSIS")
    print("=" * 60)
    
    # Get first block from stage 0
    block = None
    for name, module in convnext.named_modules():
        if 'Block' in type(module).__name__:
            block = module
            block_name = name
            break
    
    if block is None:
        print("Could not find ConvNeXt block")
        return
    
    print(f"\nBlock: {block_name}")
    print(f"Type: {type(block).__name__}")
    
    print("\nWeights:")
    for name, param in block.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\nForward pass structure:")
    print("""
    x = input
    x = dwconv(x)           # Depthwise 7x7 conv
    x = x.permute(...)      # NCHW -> NHWC
    x = norm(x)             # LayerNorm
    x = pwconv1(x)          # Linear expansion (C -> 4C)
    x = act(x)              # GELU
    x = pwconv2(x)          # Linear reduction (4C -> C)
    x = x.permute(...)      # NHWC -> NCHW
    x = input + x * scale   # Residual with layer scale
    """)


if __name__ == "__main__":
    convnext = analyze_convnext()
    analyze_single_block()
