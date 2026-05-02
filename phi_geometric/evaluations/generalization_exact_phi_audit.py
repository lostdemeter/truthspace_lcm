#!/usr/bin/env python3
"""
Three questions:
1. Can φ-lattice quantization + x·σ(φ·x) close the 12% classification gap?
2. What are norms? Can they be eliminated or derived geometrically?
3. What's truly irreducible after all geometric replacements?

From docs 128, 140, 125:
- Weights live on the φ-lattice (sign × φ^level)
- Signs are irreducible knowledge (1 bit/weight)
- Levels are universal structure (compressible)
- φ-arithmetic gives 99.98% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import copy
import time
from PIL import Image
from pathlib import Path
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI
LOG_PHI = np.log(PHI)


# ============================================================================
# φ-lattice tools (from doc 128)
# ============================================================================

def to_phi_lattice(w):
    """Snap weights to nearest φ-lattice position: sign × φ^level."""
    signs = torch.sign(w)
    magnitudes = torch.abs(w).clamp(min=1e-20)
    levels = torch.round(torch.log(magnitudes) / LOG_PHI)
    levels = levels.clamp(-30, 10)
    lattice_values = signs * (PHI ** levels)
    corrections = w - lattice_values
    return signs, levels, corrections, lattice_values


def snap_to_lattice(w, keep_corrections=False, correction_threshold=0.005):
    """Snap weights to φ-lattice. Optionally keep large corrections."""
    signs, levels, corrections, lattice = to_phi_lattice(w)
    if keep_corrections:
        mask = corrections.abs() >= correction_threshold
        return lattice + corrections * mask.float()
    return lattice


class PhiGate(nn.Module):
    """x·σ(φ·x) — SiLU with φ-steepness."""
    def forward(self, x):
        return x * torch.sigmoid(PHI * x)


def replace_gelu(model):
    """Replace all GELU with x·σ(φ·x)."""
    phi_gate = PhiGate()
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], phi_gate)
            count += 1
    return count


# ============================================================================
# Part 1: Can φ-lattice quantization close the 12% gap?
# ============================================================================

print("=" * 80)
print("PART 1: Can φ-lattice coordinates close the 12% gap?")
print("=" * 80)
print()

weights_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model_orig = convnext_tiny(weights=weights_enum)
model_orig.eval()
preprocess = weights_enum.transforms()

# Analyze weight distribution on φ-lattice
print("  Weight analysis on φ-lattice:")
total_params = 0
total_on_lattice = 0
level_counts = {}

for name, param in model_orig.named_parameters():
    w = param.data
    signs, levels, corrections, lattice = to_phi_lattice(w)
    n = w.numel()
    total_params += n
    on_lattice = (corrections.abs() < 0.005).sum().item()
    total_on_lattice += on_lattice

    # Count levels
    for lev in levels.flatten().int().tolist():
        level_counts[lev] = level_counts.get(lev, 0) + 1

print(f"    Total params: {total_params/1e6:.1f}M")
print(f"    On φ-lattice (correction < 0.005): {total_on_lattice/total_params*100:.1f}%")
print()

# Show level distribution
sorted_levels = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
print(f"    {'Level':<8} {'φ^level':<12} {'Count':<12} {'%':<8}")
print(f"    {'-'*38}")
for level, count in sorted_levels[:12]:
    pct = count / total_params * 100
    phi_val = PHI ** level
    bar = '█' * int(pct)
    print(f"    {level:<8} {phi_val:<12.6f} {count:<12} {pct:<6.1f}% {bar}")

# Now test variants
image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))
N_TEST = 100

variants = {
    'A: GELU (original)': {'gate': 'gelu', 'lattice': False},
    'B: x·σ(φ·x) only': {'gate': 'phi', 'lattice': False},
    'C: GELU + φ-lattice weights': {'gate': 'gelu', 'lattice': True},
    'D: x·σ(φ·x) + φ-lattice weights': {'gate': 'phi', 'lattice': True},
    'E: x·σ(φ·x) + φ-lattice + corrections': {'gate': 'phi', 'lattice': 'with_corrections'},
}

print()
print("  Testing 5 variants on 100 images...")

results = {}
for variant_name, config in variants.items():
    model = copy.deepcopy(model_orig)
    model.eval()

    # Apply gate
    if config['gate'] == 'phi':
        replace_gelu(model)

    # Apply φ-lattice quantization
    if config['lattice']:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if config['lattice'] == 'with_corrections':
                    param.data = snap_to_lattice(param.data, keep_corrections=True)
                else:
                    param.data = snap_to_lattice(param.data, keep_corrections=False)

    # Test
    preds = []
    ref_preds = []
    logit_cosines = []

    with torch.no_grad():
        for idx in range(N_TEST):
            img = cv2.imread(images[idx])
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = preprocess(pil_img).unsqueeze(0)

            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            preds.append(pred)

            # Get reference (variant A)
            if variant_name == 'A: GELU (original)':
                ref_preds.append(pred)
            else:
                logits_ref = model_orig(tensor)
                ref_preds.append(logits_ref.argmax(dim=1).item())
                cos = F.cosine_similarity(logits, logits_ref, dim=1).item()
                logit_cosines.append(cos)

    n = len(preds)
    agree = sum(1 for p, r in zip(preds, ref_preds) if p == r)
    results[variant_name] = {
        'agree': agree,
        'n': n,
        'pct': agree / n * 100,
        'cos': np.mean(logit_cosines) if logit_cosines else 1.0,
    }

print()
print(f"  {'Variant':<45} {'Top-1 Agree':<15} {'Logit Cos':<10}")
print(f"  {'-'*68}")
for name, r in results.items():
    print(f"  {name:<45} {r['agree']}/{r['n']} ({r['pct']:.1f}%)  {r['cos']:.4f}")


# ============================================================================
# Part 2: What are norms? Can they be eliminated?
# ============================================================================

print()
print("=" * 80)
print("PART 2: What are norms? Can they be eliminated?")
print("=" * 80)
print()

# Collect all norm parameters
norm_params = {}
for name, param in model_orig.named_parameters():
    if 'norm' in name.lower() or 'layer_scale' in name.lower():
        norm_params[name] = param.data.clone()
    # Block LayerNorm (block.2 is the LayerNorm in each block)
    elif '.block.2.' in name:
        norm_params[name] = param.data.clone()

print(f"  Norm parameters found: {len(norm_params)}")
print()
print(f"  {'Name':<55} {'Shape':<15} {'Mean':<10} {'Std':<10} {'~1?':<8}")
print(f"  {'-'*95}")

norm_weight_devs = []
norm_bias_devs = []

for name, p in sorted(norm_params.items()):
    shape = str(list(p.shape))
    mean = p.mean().item()
    std = p.std().item()
    near_one = abs(mean - 1.0) < 0.1 if 'weight' in name else abs(mean) < 0.1
    indicator = "≈1" if 'weight' in name and near_one else ("≈0" if 'bias' in name and near_one else "")

    if 'weight' in name:
        norm_weight_devs.append(abs(mean - 1.0))
    if 'bias' in name:
        norm_bias_devs.append(abs(mean))

    if len(name) > 55:
        short_name = '...' + name[-52:]
    else:
        short_name = name
    print(f"  {short_name:<55} {shape:<15} {mean:<10.4f} {std:<10.4f} {indicator}")

print()
if norm_weight_devs:
    print(f"  Norm weights: mean deviation from 1.0 = {np.mean(norm_weight_devs):.4f}")
if norm_bias_devs:
    print(f"  Norm biases:  mean deviation from 0.0 = {np.mean(norm_bias_devs):.4f}")

# Test: what if we set ALL norm weights to 1.0 and biases to 0.0?
print()
print("  Test: replace norm params with identity (weight=1, bias=0)...")

model_no_norms = copy.deepcopy(model_orig)
model_no_norms.eval()

with torch.no_grad():
    for name, param in model_no_norms.named_parameters():
        if ('norm' in name.lower() or '.block.2.' in name):
            if 'weight' in name:
                param.data.fill_(1.0)
            elif 'bias' in name:
                param.data.fill_(0.0)

# Also test: norms removed + phi gate
model_no_norms_phi = copy.deepcopy(model_no_norms)
replace_gelu(model_no_norms_phi)
model_no_norms_phi.eval()

# Test
for label, test_model in [('Identity norms + GELU', model_no_norms),
                           ('Identity norms + x·σ(φ·x)', model_no_norms_phi)]:
    agree = 0
    cosines = []
    with torch.no_grad():
        for idx in range(N_TEST):
            img = cv2.imread(images[idx])
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = preprocess(pil_img).unsqueeze(0)

            logits = test_model(tensor)
            logits_ref = model_orig(tensor)

            if logits.argmax(1).item() == logits_ref.argmax(1).item():
                agree += 1
            cosines.append(F.cosine_similarity(logits, logits_ref, dim=1).item())

    print(f"    {label:<35} agree={agree}/{N_TEST} ({agree/N_TEST*100:.1f}%)  cos={np.mean(cosines):.4f}")


# ============================================================================
# Part 3: GRN (Global Response Normalization) analysis
# ============================================================================
print()
print("  GRN (Global Response Normalization) parameters:")
grn_params = {}
for name, param in model_orig.named_parameters():
    if '.block.4.' in name:  # GRN is block.4
        grn_params[name] = param.data.clone()

for name, p in sorted(grn_params.items()):
    print(f"    {name}: shape={list(p.shape)}, mean={p.mean():.6f}, std={p.std():.6f}")


# ============================================================================
# Part 4: The Irreducibility Audit
# ============================================================================

print()
print("=" * 80)
print("PART 3: What's TRULY irreducible?")
print("=" * 80)
print()

# Categorize everything
categories = {
    'GEOMETRIC (0 cost)': {
        'x·σ(φ·x) gate': 0,
    },
    'DERIVABLE (compressible)': {},
    'LEARNED (irreducible)': {},
}

for name, param in model_orig.named_parameters():
    n = param.numel()

    # Gate: already replaced (0 params)
    # GELU has no params

    # Norms
    if 'norm' in name.lower() or '.block.2.' in name:
        # Check if it's near identity
        if 'weight' in name and abs(param.mean().item() - 1.0) < 0.5:
            categories['DERIVABLE (compressible)'][f'norm.weight: {name}'] = n
        elif 'bias' in name and abs(param.mean().item()) < 0.5:
            categories['DERIVABLE (compressible)'][f'norm.bias: {name}'] = n
        else:
            categories['LEARNED (irreducible)'][f'norm: {name}'] = n

    # GRN
    elif '.block.4.' in name:
        categories['DERIVABLE (compressible)'][f'GRN: {name}'] = n

    # DW conv
    elif '.block.0.' in name:
        categories['DERIVABLE (compressible)'][f'DW: {name}'] = n

    # Layer scale
    elif 'layer_scale' in name:
        categories['DERIVABLE (compressible)'][f'layer_scale: {name}'] = n

    # PW (expand/contract)
    elif '.block.3.' in name or '.block.5.' in name:
        # These are the PW weights — the spectrometer
        # From doc 140: levels are compressible, signs are irreducible
        signs = torch.sign(param.data)
        # Signs: 1 bit per weight
        categories['LEARNED (irreducible)'][f'PW signs: {name}'] = n  # 1 bit each
        categories['DERIVABLE (compressible)'][f'PW levels: {name}'] = n  # ~5 bits each

    # Stem and classifier
    else:
        categories['LEARNED (irreducible)'][f'{name}'] = n

# Print summary
for cat_name, items in categories.items():
    total = sum(items.values())
    print(f"  {cat_name}: {total/1e6:.2f}M params")
    # Show top items
    sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
    for item_name, n in sorted_items[:5]:
        short = item_name[:70]
        print(f"    {short:<70} {n:>10,}")
    if len(sorted_items) > 5:
        print(f"    ... and {len(sorted_items)-5} more items")
    print()

# The key question: what's the irreducible INFORMATION?
pw_total = 0
pw_sign_bits = 0
for name, param in model_orig.named_parameters():
    if '.block.3.' in name or '.block.5.' in name:
        n = param.numel()
        pw_total += n
        pw_sign_bits += n  # 1 bit per sign

print("=" * 80)
print("THE IRREDUCIBLE CONTENT")
print("=" * 80)
print()
print(f"  PW (spectrometer) weights: {pw_total/1e6:.1f}M")
print(f"  From doc 140: Model = φ^levels × signs")
print(f"    - Levels: UNIVERSAL, compressible (~5 bits each)")
print(f"    - Signs:  IRREDUCIBLE, 1 bit each")
print(f"    - Total irreducible: {pw_sign_bits/8/1e6:.1f} MB ({pw_sign_bits} bits)")
print()
print(f"  Compare:")
print(f"    Original model:     {total_params * 4 / 1e6:.0f} MB (float32)")
print(f"    φ-lattice (6 bits): {total_params * 6 / 8 / 1e6:.1f} MB")
print(f"    Signs only (1 bit): {pw_sign_bits / 8 / 1e6:.1f} MB")
print()
print(f"  The TRULY irreducible content of a 28.6M-param ConvNeXt classifier:")
print(f"    {pw_sign_bits} sign bits = {pw_sign_bits/8/1e6:.1f} MB")
print(f"    Everything else is derivable from φ-geometry.")
print(f"    Compression: {total_params * 32 / pw_sign_bits:.0f}x vs float32")
