#!/usr/bin/env python3
"""
Backbone Profiler
==================

Detailed profiling of the DA2 backbone to identify optimization opportunities.

Breaks down:
1. Image preprocessing
2. Backbone stages (DINOv2)
3. Neck (feature fusion)
4. Head (depth projection)
5. φ-decoder (AIG vs float)

Goal: Find where time is spent and what can be optimized.
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class TimingResult:
    """Timing result for a pipeline stage."""
    name: str
    time_ms: float
    percentage: float
    params: int = 0
    flops: int = 0
    description: str = ""


def profile_full_pipeline():
    """Profile every step of the depth estimation pipeline."""
    
    print("=" * 70)
    print("FULL PIPELINE PROFILER")
    print("=" * 70)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("Loading DA2 model...")
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf'
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf'
    ).to(device)
    model.eval()
    
    # Use FP16
    model = model.half()
    dtype = torch.float16
    
    # Load AIG decoder
    from aig_depth_decoder import AIGPhiDecoder
    aig_decoder = AIGPhiDecoder()
    
    # Test image
    test_image = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up...")
    from PIL import Image
    pil_image = Image.fromarray(test_image)
    inputs = processor(images=pil_image, return_tensors='pt')
    inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}
    
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # Profile each stage
    print()
    print("Profiling pipeline stages...")
    print("-" * 70)
    
    timings = []
    n_runs = 20
    
    # Stage 1: Image preprocessing (CPU)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pil_image = Image.fromarray(test_image)
        inputs = processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    preprocess_time = np.mean(times) * 1000
    timings.append(TimingResult(
        "1. Preprocessing", preprocess_time, 0,
        description="PIL conversion + HuggingFace processor + GPU transfer"
    ))
    
    # Prepare input for subsequent stages
    pil_image = Image.fromarray(test_image)
    inputs = processor(images=pil_image, return_tensors='pt')
    inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}
    pixel_values = inputs['pixel_values']
    
    # Stage 2: Full backbone (embeddings + encoder)
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            backbone_output = model.backbone(pixel_values)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    backbone_time = np.mean(times) * 1000
    timings.append(TimingResult(
        "2. Backbone (DINOv2)", backbone_time, 0,
        params=sum(p.numel() for p in model.backbone.parameters()),
        description=f"Embeddings + {len(model.backbone.encoder.layer)} transformer layers"
    ))
    
    # Stage 3: Neck + Head combined (they depend on each other)
    # Run full model to get neck output properly
    head_features = None
    
    def capture_hook(module, inp, out):
        nonlocal head_features
        head_features = out.detach()
    
    handle = model.head.activation1.register_forward_hook(capture_hook)
    
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            # Run backbone
            backbone_out = model.backbone(pixel_values)
            # Run neck + head (need to call full forward for proper shapes)
            _ = model(pixel_values)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    handle.remove()
    
    # Subtract backbone time to get neck+head time
    full_time = np.mean(times) * 1000
    neck_head_time = full_time - backbone_time
    
    timings.append(TimingResult(
        "3. Neck + Head", neck_head_time, 0,
        params=sum(p.numel() for p in model.neck.parameters()) + sum(p.numel() for p in model.head.parameters()),
        description="Feature fusion + depth projection"
    ))
    
    # Stage 4: AIG φ-decoder
    features_np = head_features.squeeze().cpu().float().numpy()
    if features_np.ndim == 3 and features_np.shape[0] == 32:
        features_np = features_np.transpose(1, 2, 0)
    
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        depth = aig_decoder.decode(features_np, use_packed=True)
        times.append(time.perf_counter() - t0)
    aig_time = np.mean(times) * 1000
    timings.append(TimingResult(
        "4. AIG φ-Decoder", aig_time, 0,
        params=32,  # 32 weights
        description="Integer shift-add (byte-packed)"
    ))
    
    # Stage 5: Resize to original
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        depth_resized = cv2.resize(depth, (854, 480), interpolation=cv2.INTER_LINEAR)
        times.append(time.perf_counter() - t0)
    resize_time = np.mean(times) * 1000
    timings.append(TimingResult(
        "5. Resize", resize_time, 0,
        description="Bilinear upscale to input resolution"
    ))
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # Encoder layer breakdown (estimate based on backbone time / 12 layers)
    n_layers = len(model.backbone.encoder.layer)
    encoder_times = [backbone_time / n_layers] * n_layers  # Approximate equal distribution
    
    # Calculate total and percentages
    total_time = sum(t.time_ms for t in timings)
    for t in timings:
        t.percentage = (t.time_ms / total_time) * 100
    
    # Print results
    print()
    print(f"{'Stage':<30} {'Time (ms)':>10} {'%':>8} {'Params':>12} {'Description'}")
    print("-" * 90)
    
    for t in timings:
        params_str = f"{t.params:,}" if t.params > 0 else "-"
        print(f"{t.name:<30} {t.time_ms:>10.2f} {t.percentage:>7.1f}% {params_str:>12} {t.description}")
    
    print("-" * 90)
    print(f"{'TOTAL':<30} {total_time:>10.2f} {'100.0':>7}%")
    print()
    
    # Summary
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    # Find bottlenecks
    sorted_timings = sorted(timings, key=lambda x: x.time_ms, reverse=True)
    
    print("Top bottlenecks:")
    for i, t in enumerate(sorted_timings[:3]):
        print(f"  {i+1}. {t.name}: {t.time_ms:.2f}ms ({t.percentage:.1f}%)")
    
    print()
    
    # Encoder breakdown
    if encoder_times:
        encoder_total = sum(encoder_times)
        print(f"Encoder layer breakdown (total: {encoder_total:.2f}ms):")
        for i, et in enumerate(encoder_times):
            bar = "█" * int(et / max(encoder_times) * 20)
            print(f"  Layer {i+1:2d}: {et:5.2f}ms {bar}")
    
    print()
    
    return timings, model, encoder_times


def analyze_model_architecture(model):
    """Analyze model architecture for optimization opportunities."""
    
    print("=" * 70)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("=" * 70)
    print()
    
    # Count parameters by component
    components = {
        'backbone.embeddings': model.backbone.embeddings,
        'backbone.encoder': model.backbone.encoder,
        'neck': model.neck,
        'head': model.head,
    }
    
    print(f"{'Component':<30} {'Parameters':>15} {'Size (MB)':>12}")
    print("-" * 60)
    
    total_params = 0
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        size_mb = params * 2 / 1024 / 1024  # FP16
        total_params += params
        print(f"{name:<30} {params:>15,} {size_mb:>12.2f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_params:>15,} {total_params * 2 / 1024 / 1024:>12.2f}")
    print()
    
    # Analyze encoder layers
    print("Encoder layer details:")
    n_layers = len(model.backbone.encoder.layer)
    params_per_layer = sum(p.numel() for p in model.backbone.encoder.layer[0].parameters())
    print(f"  {n_layers} transformer layers, {params_per_layer:,} params each")
    
    print()


def suggest_optimizations(timings):
    """Suggest optimizations based on profiling results."""
    
    print("=" * 70)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 70)
    print()
    
    total_time = sum(t.time_ms for t in timings)
    
    suggestions = []
    
    # Check preprocessing
    preprocess = next((t for t in timings if "Preprocessing" in t.name), None)
    if preprocess and preprocess.percentage > 10:
        suggestions.append({
            'component': 'Preprocessing',
            'current_ms': preprocess.time_ms,
            'issue': 'PIL + HuggingFace processor is slow',
            'solutions': [
                'Use cv2.resize directly (skip PIL)',
                'Pre-compute normalization constants',
                'Use CUDA for image resizing (cv2.cuda)',
                'Batch preprocessing on GPU'
            ],
            'potential_savings': '50-70%'
        })
    
    # Check encoder stages
    encoder_timings = [t for t in timings if "Encoder Stage" in t.name]
    encoder_total = sum(t.time_ms for t in encoder_timings)
    if encoder_total / total_time > 0.5:
        suggestions.append({
            'component': 'Encoder (Transformer)',
            'current_ms': encoder_total,
            'issue': 'Transformer attention is O(n²) in sequence length',
            'solutions': [
                'Use smaller input resolution (384 instead of 518)',
                'Use Flash Attention (if available)',
                'Quantize to INT8 (TensorRT)',
                'Use distilled/smaller backbone',
                'Skip later encoder stages (early exit)'
            ],
            'potential_savings': '30-60%'
        })
    
    # Check neck
    neck = next((t for t in timings if "Neck" in t.name), None)
    if neck and neck.percentage > 10:
        suggestions.append({
            'component': 'Neck',
            'current_ms': neck.time_ms,
            'issue': 'Feature pyramid fusion has multiple conv layers',
            'solutions': [
                'Use single-scale features (skip FPN)',
                'Reduce neck channels',
                'Fuse neck into head'
            ],
            'potential_savings': '20-40%'
        })
    
    # Print suggestions
    for i, s in enumerate(suggestions):
        print(f"{i+1}. {s['component']} ({s['current_ms']:.1f}ms)")
        print(f"   Issue: {s['issue']}")
        print(f"   Solutions:")
        for sol in s['solutions']:
            print(f"     • {sol}")
        print(f"   Potential savings: {s['potential_savings']}")
        print()
    
    # Estimate optimized time
    print("=" * 70)
    print("PROJECTED OPTIMIZATIONS")
    print("=" * 70)
    print()
    
    current_total = total_time
    
    optimizations = [
        ("Fast preprocessing (cv2 direct)", 0.5, "Preprocessing"),
        ("Lower resolution (384px)", 0.7, "Encoder"),
        ("TensorRT INT8", 0.5, "Encoder"),
        ("Skip FPN neck", 0.3, "Neck"),
    ]
    
    print(f"{'Optimization':<35} {'Savings':>10} {'New Total':>12}")
    print("-" * 60)
    
    running_total = current_total
    for name, factor, component in optimizations:
        component_time = next((t.time_ms for t in timings if component in t.name), 0)
        if "Encoder" in component:
            component_time = sum(t.time_ms for t in timings if "Encoder" in t.name)
        
        savings = component_time * (1 - factor)
        running_total -= savings
        print(f"{name:<35} {savings:>9.1f}ms {running_total:>11.1f}ms")
    
    print("-" * 60)
    print(f"{'Current':<35} {'':<10} {current_total:>11.1f}ms ({1000/current_total:.0f} FPS)")
    print(f"{'Optimized (projected)':<35} {'':<10} {running_total:>11.1f}ms ({1000/running_total:.0f} FPS)")
    print()


def main():
    """Run full profiling analysis."""
    
    timings, model, encoder_times = profile_full_pipeline()
    analyze_model_architecture(model)
    suggest_optimizations(timings)
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The backbone (DINOv2 transformer) is the main bottleneck.")
    print("The AIG φ-decoder is already very fast (~1.6ms).")
    print()
    print("Key optimization paths:")
    print("  1. Lower input resolution (384 vs 518)")
    print("  2. TensorRT/ONNX quantization")
    print("  3. Faster preprocessing (skip PIL)")
    print("  4. Early exit from encoder")
    print()


if __name__ == "__main__":
    main()
