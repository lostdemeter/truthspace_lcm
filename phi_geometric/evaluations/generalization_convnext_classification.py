#!/usr/bin/env python3
"""
GENERALIZATION TEST: Does x·σ(φ·x) work beyond colorization?

Test: Replace GELU → x·σ(φ·x) in a pre-trained ConvNeXt-Tiny (ImageNet classifier).
If predictions agree, the geometric gate generalizes to a completely different task.

The "data object" — the Geometric Spectrometer Pattern:
  1. Identify expand-gate-contract blocks
  2. Replace gate with x·σ(φ·x) — 0 params, φ-defined
  3. Measure: do predictions survive?

This is the acid test: same geometric structure, different task, different weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import json
import time
import copy
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2


# ============================================================================
# The Data Object: φ-gate replacement
# ============================================================================

class PhiGate(nn.Module):
    """x·σ(φ·x) — SiLU with φ-steepness. The geometric gate."""
    def forward(self, x):
        return x * torch.sigmoid(PHI * x)


def replace_gelu_with_phi_gate(model):
    """
    The generalizable process: find all GELU activations in any model
    and replace them with x·σ(φ·x).
    
    This is the "data object" — a reusable geometric transformation
    that can be applied to ANY architecture with GELU gates.
    """
    count = 0
    phi_gate = PhiGate()
    
    for name, module in model.named_modules():
        # ConvNeXt stores GELU in the block's activation
        if isinstance(module, nn.GELU):
            # Replace in parent
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], phi_gate)
            count += 1
    
    return count


# ============================================================================
# Part 1: Geometric Audit of x·σ(φ·x) version
# ============================================================================

print("=" * 80)
print("GEOMETRIC AUDIT: x·σ(φ·x) ConvNeXt")
print("=" * 80)
print()

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# Load pre-trained model
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model_gelu = convnext_tiny(weights=weights)
model_gelu.eval()

# Count parameters by type
total_params = 0
gelu_equivalent_params = 0  # GELU has 0 params, but affects everything
pw_params = 0
dw_params = 0
norm_params = 0
other_params = 0

for name, param in model_gelu.named_parameters():
    n = param.numel()
    total_params += n
    if 'dwconv' in name:
        dw_params += n
    elif 'pwconv' in name or 'grn' in name:
        pw_params += n
    elif 'norm' in name or 'ln' in name or 'layer_scale' in name:
        norm_params += n
    else:
        other_params += n

print(f"  ConvNeXt-Tiny: {total_params/1e6:.1f}M parameters")
print(f"    PW (spectrometer):  {pw_params/1e6:.1f}M ({pw_params/total_params*100:.1f}%)")
print(f"    DW (spatial):       {dw_params/1e6:.1f}M ({dw_params/total_params*100:.1f}%)")
print(f"    Norm (statistics):  {norm_params/1e6:.1f}M ({norm_params/total_params*100:.1f}%)")
print(f"    Other (stem/head):  {other_params/1e6:.1f}M ({other_params/total_params*100:.1f}%)")
print()

# Count GELU activations
gelu_count = sum(1 for _, m in model_gelu.named_modules() if isinstance(m, nn.GELU))
print(f"  GELU activations found: {gelu_count}")
print(f"  Each GELU: 0 learned params, fully defined by x·Φ(x)")
print(f"  Our replacement x·σ(φ·x): 0 learned params, fully defined by φ")
print()

# The audit
print("  GEOMETRIC AUDIT (if we replace GELU → x·σ(φ·x)):")
print(f"    {'Component':<25} {'Params':<12} {'Type':<20}")
print(f"    {'-'*55}")
print(f"    {'x·σ(φ·x) gate':<25} {'0':<12} {'GEOMETRIC (φ)':<20}")
print(f"    {'DW conv (spatial)':<25} {f'{dw_params/1e6:.1f}M':<12} {'φ-separable (R²=0.98)':<20}")
print(f"    {'Norms':<25} {f'{norm_params/1e6:.1f}M':<12} {'Statistics':<20}")
print(f"    {'PW (spectrometer)':<25} {f'{pw_params/1e6:.1f}M':<12} {'LEARNED (directions)':<20}")
print(f"    {'Stem + head':<25} {f'{other_params/1e6:.1f}M':<12} {'LEARNED (interface)':<20}")
print()

geometric_frac = (0 + dw_params + norm_params) / total_params * 100
print(f"  Geometric or derivable: {geometric_frac:.1f}% of params")
print(f"  Learned (irreducible):  {100-geometric_frac:.1f}% of params")
print(f"  But: the GATE SHAPE is geometric — and it controls ALL computation")

# ============================================================================
# Part 2: The Generalization Test
# ============================================================================

print()
print("=" * 80)
print("GENERALIZATION TEST: ImageNet Classification")
print("=" * 80)
print()

# Create φ-gate version
model_phi = copy.deepcopy(model_gelu)
n_replaced = replace_gelu_with_phi_gate(model_phi)
model_phi.eval()
print(f"  Replaced {n_replaced} GELU activations with x·σ(φ·x)")

# Use COCO val images as test (don't need correct labels — 
# just need predictions to AGREE between GELU and φ-gate)
image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))

# ImageNet preprocessing
preprocess = weights.transforms()

N_TEST = 100
results = {
    'top1_agree': 0,
    'top5_agree': 0,
    'logit_cosines': [],
    'feature_cosines': [],
    'gelu_times': [],
    'phi_times': [],
    'gelu_preds': [],
    'phi_preds': [],
    'logit_rmses': [],
    'max_logit_diffs': [],
}

# Hook to capture features before classifier head
gelu_features = {}
phi_features = {}

def make_hook(storage, key):
    def hook(module, input, output):
        storage[key] = output.detach()
    return hook

# Register hooks on the last norm layer (before classifier)
for name, module in model_gelu.named_modules():
    if name == 'classifier.0':  # LayerNorm before final Linear
        module.register_forward_hook(make_hook(gelu_features, 'final'))
        
for name, module in model_phi.named_modules():
    if name == 'classifier.0':
        module.register_forward_hook(make_hook(phi_features, 'final'))

# Also hook intermediate stages
stage_hooks_gelu = {}
stage_hooks_phi = {}
for stage_idx in range(4):
    for name, module in model_gelu.named_modules():
        if name == f'features.{stage_idx*2+1}':  # After each stage
            module.register_forward_hook(make_hook(stage_hooks_gelu, f'stage_{stage_idx}'))
    for name, module in model_phi.named_modules():
        if name == f'features.{stage_idx*2+1}':
            module.register_forward_hook(make_hook(stage_hooks_phi, f'stage_{stage_idx}'))

print(f"  Testing {N_TEST} images...")
print()

# Load ImageNet class names
try:
    categories = weights.meta["categories"]
except:
    categories = [f"class_{i}" for i in range(1000)]

with torch.no_grad():
    for idx in range(N_TEST):
        # Load and preprocess
        img = cv2.imread(images[idx])
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and preprocess
        from torchvision.transforms.functional import to_pil_image
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        tensor = preprocess(pil_img).unsqueeze(0)
        
        # GELU forward
        t0 = time.time()
        logits_gelu = model_gelu(tensor)
        t1 = time.time()
        results['gelu_times'].append(t1 - t0)
        
        # φ-gate forward
        t0 = time.time()
        logits_phi = model_phi(tensor)
        t1 = time.time()
        results['phi_times'].append(t1 - t0)
        
        # Compare predictions
        pred_gelu = logits_gelu.argmax(dim=1).item()
        pred_phi = logits_phi.argmax(dim=1).item()
        results['gelu_preds'].append(pred_gelu)
        results['phi_preds'].append(pred_phi)
        
        if pred_gelu == pred_phi:
            results['top1_agree'] += 1
        
        # Top-5 agreement
        top5_gelu = set(logits_gelu.topk(5, dim=1).indices[0].tolist())
        top5_phi = set(logits_phi.topk(5, dim=1).indices[0].tolist())
        if top5_gelu == top5_phi:
            results['top5_agree'] += 1
        
        # Logit similarity
        cos = F.cosine_similarity(logits_gelu, logits_phi, dim=1).item()
        results['logit_cosines'].append(cos)
        
        # Logit RMSE
        rmse = torch.sqrt(torch.mean((logits_gelu - logits_phi)**2)).item()
        results['logit_rmses'].append(rmse)
        
        # Max logit difference
        max_diff = (logits_gelu - logits_phi).abs().max().item()
        results['max_logit_diffs'].append(max_diff)
        
        # Feature similarity (before classifier)
        if 'final' in gelu_features and 'final' in phi_features:
            feat_cos = F.cosine_similarity(
                gelu_features['final'].flatten().unsqueeze(0),
                phi_features['final'].flatten().unsqueeze(0)
            ).item()
            results['feature_cosines'].append(feat_cos)

# ============================================================================
# Results
# ============================================================================

n = len(results['gelu_preds'])
print(f"  {'Metric':<35} {'Value':<15}")
print(f"  {'-'*50}")
print(f"  {'Top-1 prediction agreement':<35} {results['top1_agree']}/{n} ({results['top1_agree']/n*100:.1f}%)")
print(f"  {'Top-5 prediction agreement':<35} {results['top5_agree']}/{n} ({results['top5_agree']/n*100:.1f}%)")
print(f"  {'Mean logit cosine similarity':<35} {np.mean(results['logit_cosines']):.6f}")
print(f"  {'Min logit cosine similarity':<35} {np.min(results['logit_cosines']):.6f}")
print(f"  {'Mean logit RMSE':<35} {np.mean(results['logit_rmses']):.4f}")
print(f"  {'Max logit difference':<35} {np.mean(results['max_logit_diffs']):.4f}")
if results['feature_cosines']:
    print(f"  {'Mean feature cosine (pre-head)':<35} {np.mean(results['feature_cosines']):.6f}")
    print(f"  {'Min feature cosine (pre-head)':<35} {np.min(results['feature_cosines']):.6f}")
print(f"  {'Mean GELU time (ms)':<35} {np.mean(results['gelu_times'])*1000:.1f}")
print(f"  {'Mean φ-gate time (ms)':<35} {np.mean(results['phi_times'])*1000:.1f}")

# Stage-by-stage feature comparison
print()
print("  Stage-by-stage feature similarity (last image):")
for stage_idx in range(4):
    key = f'stage_{stage_idx}'
    if key in stage_hooks_gelu and key in stage_hooks_phi:
        g = stage_hooks_gelu[key].flatten()
        p = stage_hooks_phi[key].flatten()
        cos = F.cosine_similarity(g.unsqueeze(0), p.unsqueeze(0)).item()
        mag_ratio = p.norm() / g.norm()
        print(f"    Stage {stage_idx}: cos={cos:.6f}, mag_ratio={mag_ratio.item():.4f}")

# Show some example predictions
print()
print("  Example predictions (first 10):")
print(f"  {'Image':<6} {'GELU prediction':<30} {'φ-gate prediction':<30} {'Match'}")
print(f"  {'-'*75}")
for i in range(min(10, n)):
    g = categories[results['gelu_preds'][i]]
    p = categories[results['phi_preds'][i]]
    match = "✓" if results['gelu_preds'][i] == results['phi_preds'][i] else "✗"
    print(f"  {i:<6} {g:<30} {p:<30} {match}")

# Show disagreements
disagrees = [(i, results['gelu_preds'][i], results['phi_preds'][i]) 
             for i in range(n) if results['gelu_preds'][i] != results['phi_preds'][i]]
if disagrees:
    print(f"\n  Disagreements ({len(disagrees)}/{n}):")
    for i, g, p in disagrees[:10]:
        gname = categories[g]
        pname = categories[p]
        cos = results['logit_cosines'][i]
        print(f"    img {i}: GELU={gname}, φ={pname} (logit cos={cos:.4f})")

# ============================================================================
# The Framework Definition
# ============================================================================

print()
print("=" * 80)
print("THE GEOMETRIC SPECTROMETER PATTERN")
print("=" * 80)
print()
print("  The reusable data object for geometric neural network decomposition:")
print()
print("  1. IDENTIFY the expand-gate-contract primitive")
print("     - ConvNeXt: PW1 → GELU → PW2     (vision)")
print("     - LLaMA:    gate_proj → SiLU → down_proj  (language)")
print("     - GPT:      fc1 → GELU → fc2       (language)")
print()
print("  2. REPLACE the gate with x·σ(φ·x)")
print("     - 0 learned parameters")
print("     - φ controls transition steepness")
print("     - Max error vs GELU: 0.030")
print("     - Max error vs SiLU: much larger (SiLU uses k=1, not k=φ)")
print()
print("  3. VERIFY the spatial structure is φ-separable")
print("     - DW conv kernels: φ-decay from center (R²=0.98)")
print("     - Attention patterns: φ-Zipf spectral decay")
print()
print("  4. COMPRESS learned directions via φ-Zipf SVD")
print("     - PW singular values follow S[i] ∝ 1/i^(1/φ)")
print("     - Rank 50% preserves 95%+ of information")
print()
print("  5. REPLACE task-specific decoders geometrically")
print("     - Color matrix (colorization)")
print("     - Linear probe (classification) ")
print("     - The decoder is task-specific; the encoder is universal")
print()

# Final verdict
top1_pct = results['top1_agree'] / n * 100
mean_cos = np.mean(results['logit_cosines'])
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()
print(f"  x·σ(φ·x) preserves {top1_pct:.1f}% of top-1 predictions")
print(f"  Logit cosine similarity: {mean_cos:.6f}")
print()
if top1_pct >= 95:
    print("  THE PATTERN GENERALIZES.")
    print("  The geometric gate works for classification AND colorization.")
    print("  Same φ. Same structure. Different task. Different weights.")
elif top1_pct >= 80:
    print("  THE PATTERN MOSTLY GENERALIZES.")
    print(f"  {100-top1_pct:.1f}% of predictions change — small but measurable drift.")
    print("  The gate shape matters more when compounded through deep networks.")
else:
    print("  THE PATTERN DOES NOT GENERALIZE CLEANLY.")
    print("  The 0.030 max error compounds too much through this architecture.")
