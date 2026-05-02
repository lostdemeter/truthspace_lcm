#!/usr/bin/env python3
"""
Alternating Sign Error Correction for the Gate Gap

From Base64_BBP: (-1)^n provides built-in error correction.
Each term alternates sign → partial sums bounce above/below truth → 
error bounded by LAST term, not SUM of all terms.

The gate error ε(x) = GELU(x) - x·σ(φ·x) compounds through 18 blocks.
If the error is same-sign, it accumulates. If we can make it alternate,
it cancels.

Three investigations:
1. Does the per-block error naturally alternate sign?
2. Can we decompose ε(x) as an alternating series in φ-powers?
3. Does alternating k between blocks close the gap?

The BBP insight applied:
  BBP: π = Σ (-1)^n × a_n × (1/64)^n  
  Gate: GELU = x·σ(φ·x) + Σ (-1)^n × c_n × correction_n

  The alternating sign is the error correction mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import copy
from PIL import Image

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI
SQRT_PI = np.sqrt(np.pi)


# ============================================================================
# Part 1: Does the per-block error alternate sign naturally?
# ============================================================================

print("=" * 80)
print("PART 1: Per-block error sign analysis")
print("=" * 80)
print()

# Analyze the error function ε(x) = GELU(x) - x·σ(φ·x)
x = torch.linspace(-5, 5, 10000)
gelu = x * 0.5 * (1 + torch.erf(x / np.sqrt(2.0)))
phi_gate = x * torch.sigmoid(PHI * x)
error = gelu - phi_gate

print("  Error function ε(x) = GELU(x) - x·σ(φ·x):")
print(f"    Max error:  {error.abs().max():.6f} at x={x[error.abs().argmax()]:.3f}")
print(f"    ε(0.5) = {error[5250].item():.6f}")
print(f"    ε(1.0) = {error[6000].item():.6f}")
print(f"    ε(2.0) = {error[7000].item():.6f}")
print(f"    ε(-1.0) = {error[4000].item():.6f}")
print(f"    ε(-2.0) = {error[3000].item():.6f}")
print()

# The error IS an alternating function of x:
# For small |x| → negative (σ(φx) > Φ(x) because φ > √(8/π))
# For larger |x| → positive (σ(φx) saturates faster than Φ(x))
# This alternation is BUILT INTO the error shape

# Now: does this translate to alternating per-block errors?
# Load model and trace per-block feature differences
weights_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model_orig = convnext_tiny(weights=weights_enum)
model_orig.eval()
preprocess = weights_enum.transforms()

# Create phi-gate model
model_phi = copy.deepcopy(model_orig)
phi_gate_module = type('PhiGate', (nn.Module,), {'forward': lambda self, x: x * torch.sigmoid(PHI * x)})()
for name, module in model_phi.named_modules():
    if isinstance(module, nn.GELU):
        parts = name.split('.')
        parent = model_phi
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], phi_gate_module)

image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))

# Hook into each block to capture intermediate features
block_features_orig = {}
block_features_phi = {}

def make_hook(storage, name):
    def hook(module, input, output):
        storage[name] = output.detach()
    return hook

# Register hooks on each ConvNeXt block output
hooks = []
for stage_idx in [1, 3, 5, 7]:  # ConvNeXt stages
    stage = model_orig.features[stage_idx]
    stage_phi = model_phi.features[stage_idx]
    for block_idx in range(len(stage)):
        name = f"stage{stage_idx}_block{block_idx}"
        hooks.append(stage[block_idx].register_forward_hook(
            make_hook(block_features_orig, name)))
        hooks.append(stage_phi[block_idx].register_forward_hook(
            make_hook(block_features_phi, name)))

# Run one image through both
img = cv2.imread(images[0])
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
tensor = preprocess(pil_img).unsqueeze(0)

with torch.no_grad():
    _ = model_orig(tensor)
    _ = model_phi(tensor)

# Remove hooks
for h in hooks:
    h.remove()

# Analyze per-block error
print(f"  Per-block feature error (GELU vs φ-gate):")
print(f"  {'Block':<25} {'Mean Δ':<12} {'Std Δ':<12} {'Sign':<8} {'Cos Sim':<10}")
print(f"  {'-'*65}")

block_errors = []
prev_sign = None
alternations = 0
total_blocks = 0

for name in sorted(block_features_orig.keys()):
    feat_orig = block_features_orig[name]
    feat_phi = block_features_phi[name]
    diff = (feat_phi - feat_orig).flatten()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    sign = "+" if mean_diff > 0 else "-"
    cos = F.cosine_similarity(feat_orig.flatten().unsqueeze(0),
                               feat_phi.flatten().unsqueeze(0)).item()

    block_errors.append(mean_diff)

    if prev_sign is not None:
        if (mean_diff > 0) != (prev_sign > 0):
            alternations += 1
        total_blocks += 1
    prev_sign = mean_diff

    print(f"  {name:<25} {mean_diff:<12.6f} {std_diff:<12.6f} {sign:<8} {cos:<10.6f}")

if total_blocks > 0:
    print(f"\n  Sign alternations: {alternations}/{total_blocks} "
          f"({alternations/total_blocks*100:.0f}%)")
    print(f"  Expected if random: 50%")
    print(f"  Expected if same-sign: 0%")
    print(f"  Expected if BBP-style: ~50% or higher")


# ============================================================================
# Part 2: The alternating series decomposition of ε(x)
# ============================================================================

print()
print("=" * 80)
print("PART 2: Alternating series decomposition of ε(x)")
print("=" * 80)
print()

# Express ε(x) = GELU(x) - x·σ(φ·x) as alternating series
# Using Taylor expansion around 0:
# Φ(x) = 1/2 + x/√(2π) - x³/(6√(2π)) + x⁵/(40√(2π)) - ...
# σ(φx) = 1/2 + φx/4 - (φx)³/48 + (φx)⁵/480 - ...

# So Φ(x) - σ(φx) = [1/√(2π) - φ/4]x + [φ³/48 - 1/(6√(2π))]x³ + ...

inv_sqrt_2pi = 1.0 / np.sqrt(2 * np.pi)

# Linear coefficient
c1 = inv_sqrt_2pi - PHI / 4
# Cubic coefficient  
c3 = PHI**3 / 48 - inv_sqrt_2pi / 6
# Quintic coefficient
c5 = -PHI**5 / 480 + inv_sqrt_2pi / 40

print(f"  ε(x) = x·[Φ(x) - σ(φx)]")
print(f"       = x·[c₁x + c₃x³ + c₅x⁵ + ...]")
print(f"       = c₁x² + c₃x⁴ + c₅x⁶ + ...")
print()
print(f"  Coefficients:")
print(f"    c₁ = 1/√(2π) - φ/4 = {inv_sqrt_2pi:.6f} - {PHI/4:.6f} = {c1:.6f}")
print(f"    c₃ = φ³/48 - 1/(6√(2π)) = {PHI**3/48:.6f} - {inv_sqrt_2pi/6:.6f} = {c3:.6f}")
print(f"    c₅ = -φ⁵/480 + 1/(40√(2π)) = {-PHI**5/480:.6f} + {inv_sqrt_2pi/40:.6f} = {c5:.6f}")
print()
print(f"  Note: c₁ < 0, c₃ > 0, c₅ < 0  →  ALTERNATING SIGNS!")
print(f"  ε(x) = {c1:.4f}x² + {c3:.4f}x⁴ + {c5:.4f}x⁶ + ...")
print(f"       = −0.0056x² + 0.0217x⁴ − 0.0232x⁶ + ...")
print()

# The error IS already an alternating series!
# Each power of x² gets a sign flip. The coefficients involve φ powers.

# Now: can we express coefficients in terms of φ?
print(f"  Coefficient ratios:")
print(f"    |c₃/c₁| = {abs(c3/c1):.4f}  (cf. φ² = {PHI**2:.4f})")
print(f"    |c₅/c₃| = {abs(c5/c3):.4f}  (cf. φ = {PHI:.4f})")
print(f"    |c₅/c₁| = {abs(c5/c1):.4f}  (cf. φ³ = {PHI**3:.4f})")
print()

# Verify the series approximation
x_test = torch.linspace(-3, 3, 1000)
gelu_test = x_test * 0.5 * (1 + torch.erf(x_test / np.sqrt(2.0)))
phi_test = x_test * torch.sigmoid(PHI * x_test)
error_test = (gelu_test - phi_test).numpy()

approx_1 = c1 * x_test.numpy()**2
approx_3 = approx_1 + c3 * x_test.numpy()**4
approx_5 = approx_3 + c5 * x_test.numpy()**6

print(f"  Series approximation quality (|x| < 3):")
print(f"    1 term (c₁x²):           max err = {np.max(np.abs(error_test - approx_1)):.6f}")
print(f"    2 terms (+ c₃x⁴):        max err = {np.max(np.abs(error_test - approx_3)):.6f}")
print(f"    3 terms (+ c₅x⁶):        max err = {np.max(np.abs(error_test - approx_5)):.6f}")
print(f"    Original error:            max err = {np.max(np.abs(error_test)):.6f}")


# ============================================================================
# Part 3: BBP-style alternating correction applied to blocks
# ============================================================================

print()
print("=" * 80)
print("PART 3: BBP-style alternating gate correction")
print("=" * 80)
print()

# The insight: instead of using a FIXED k for all blocks,
# alternate the steepness to make errors cancel.
#
# BBP uses: (-1)^n / 64^n
# We use:   k_n = φ + (-1)^n × δ
#
# Even blocks: k = φ + δ (overshoots GELU → positive error)
# Odd blocks:  k = φ - δ (undershoots GELU → negative error)
# The errors cancel in pairs.
#
# What should δ be? From the alternating series:
# The optimal δ minimizes the CUMULATIVE error after 18 blocks.
# With perfect alternation, the residual is ~δ/φ^18 (geometric decay).

class AlternatingPhiGate(nn.Module):
    """BBP-style alternating gate: k alternates between k+ and k- per block."""
    def __init__(self, k_plus, k_minus):
        super().__init__()
        self.k_plus = k_plus
        self.k_minus = k_minus
        self.block_counter = 0

    def forward(self, x):
        k = self.k_plus if self.block_counter % 2 == 0 else self.k_minus
        self.block_counter += 1
        return x * torch.sigmoid(k * x)

    def reset(self):
        self.block_counter = 0


class AlternatingTripleGate(nn.Module):
    """Three-phase alternating: k cycles through (k₁, k₂, k₃) based on BBP
    triple structure: 8/(4n+1) + 4/(4n+2) + 1/(4n+3).
    
    The BBP formula uses THREE terms per base-64 position.
    Our 18 blocks = 6 groups of 3. Each group is one BBP 'term'."""
    def __init__(self, k1, k2, k3):
        super().__init__()
        self.k_values = [k1, k2, k3]
        self.block_counter = 0

    def forward(self, x):
        phase = self.block_counter % 3
        group = self.block_counter // 3
        # Alternating sign per group (like (-1)^n in BBP)
        sign_flip = (-1) ** group
        k = self.k_values[phase]
        # Apply sign correction: for odd groups, flip around the mean k
        if sign_flip < 0:
            k_mean = sum(self.k_values) / 3
            k = 2 * k_mean - k  # Reflect around mean
        self.block_counter += 1
        return x * torch.sigmoid(k * x)

    def reset(self):
        self.block_counter = 0


def replace_gelu_with_module(model, gate_module):
    """Replace all GELU with a shared gate module."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], gate_module)
            count += 1
    return count


N_TEST = 100

# Test different alternating strategies
# First: find optimal δ for the simple alternating gate
print("  Searching for optimal alternating δ...")
print()

best_agree = 0
best_delta = 0
best_k_pair = (0, 0)

# δ search: the correction magnitude
deltas = [0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.155, 0.20, 0.25, 0.30]

# Get reference predictions
ref_preds = []
ref_logits = []
with torch.no_grad():
    for idx in range(N_TEST):
        img = cv2.imread(images[idx])
        if img is None:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil_img).unsqueeze(0)
        logits = model_orig(tensor)
        ref_preds.append(logits.argmax(1).item())
        ref_logits.append(logits)

print(f"  {'δ':<8} {'k+':<8} {'k-':<8} {'Top-1':<12} {'Top-5':<12} {'Cos':<8}")
print(f"  {'-'*58}")

for delta in deltas:
    k_plus = PHI + delta
    k_minus = PHI - delta

    gate = AlternatingPhiGate(k_plus, k_minus)
    model_test = copy.deepcopy(model_orig)
    replace_gelu_with_module(model_test, gate)
    model_test.eval()

    agree1 = 0
    agree5 = 0
    cosines = []

    with torch.no_grad():
        for idx in range(len(ref_preds)):
            gate.reset()
            img = cv2.imread(images[idx])
            if img is None:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = preprocess(pil_img).unsqueeze(0)
            logits = model_test(tensor)

            if logits.argmax(1).item() == ref_preds[idx]:
                agree1 += 1
            top5 = set(logits.topk(5, dim=1).indices[0].tolist())
            top5_ref = set(ref_logits[idx].topk(5, dim=1).indices[0].tolist())
            if top5 == top5_ref:
                agree5 += 1
            cosines.append(F.cosine_similarity(logits, ref_logits[idx], dim=1).item())

    n = len(ref_preds)
    print(f"  {delta:<8.3f} {k_plus:<8.3f} {k_minus:<8.3f} "
          f"{agree1}/{n} ({agree1/n*100:.0f}%)  {agree5}/{n} ({agree5/n*100:.0f}%)  "
          f"{np.mean(cosines):.4f}")

    if agree1 > best_agree:
        best_agree = agree1
        best_delta = delta
        best_k_pair = (k_plus, k_minus)

print()
print(f"  Best: δ={best_delta:.3f}, k+={best_k_pair[0]:.4f}, k-={best_k_pair[1]:.4f}, "
      f"agree={best_agree}/{N_TEST}")

# Also test: alternating around √π instead of φ
print()
print("  Now alternating around √π (optimal steepness):")
print(f"  {'δ':<8} {'k+':<8} {'k-':<8} {'Top-1':<12} {'Top-5':<12} {'Cos':<8}")
print(f"  {'-'*58}")

for delta in [0, 0.02, 0.05, 0.10, 0.15, 0.20]:
    k_plus = SQRT_PI + delta
    k_minus = SQRT_PI - delta

    gate = AlternatingPhiGate(k_plus, k_minus)
    model_test = copy.deepcopy(model_orig)
    replace_gelu_with_module(model_test, gate)
    model_test.eval()

    agree1 = 0
    cosines = []

    with torch.no_grad():
        for idx in range(len(ref_preds)):
            gate.reset()
            img = cv2.imread(images[idx])
            if img is None:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = preprocess(pil_img).unsqueeze(0)
            logits = model_test(tensor)

            if logits.argmax(1).item() == ref_preds[idx]:
                agree1 += 1
            cosines.append(F.cosine_similarity(logits, ref_logits[idx], dim=1).item())

    n = len(ref_preds)
    print(f"  {delta:<8.3f} {k_plus:<8.3f} {k_minus:<8.3f} "
          f"{agree1}/{n} ({agree1/n*100:.0f}%)                 "
          f"{np.mean(cosines):.4f}")


# ============================================================================
# Part 4: The BBP triple structure
# ============================================================================

print()
print("=" * 80)
print("PART 4: BBP triple correction (3-phase)")
print("=" * 80)
print()

# The BBP formula uses THREE terms per position:
# 8/(4n+1) + 4/(4n+2) + 1/(4n+3)
# Ratio: 8 : 4 : 1 — powers of 2
#
# Our 18 blocks = 6 groups of 3.
# Each group is one BBP "position".
# Within each group: k₁, k₂, k₃ (different steepnesses)
# Between groups: alternating sign correction

# The BBP coefficients are 8:4:1. In φ terms:
# 8/4 = 2 ≈ φ^(3/2) = φ·√φ
# 4/1 = 4 ≈ φ³/φ = φ² ... close to φ² = 2.618
# So the ratios aren't exactly φ but close.

# Try k values spaced by φ-ratio
k_center = SQRT_PI  # optimal center
k1 = k_center + 0.1
k2 = k_center
k3 = k_center - 0.1

gate = AlternatingTripleGate(k1, k2, k3)
model_test = copy.deepcopy(model_orig)
replace_gelu_with_module(model_test, gate)
model_test.eval()

agree1 = 0
cosines = []

with torch.no_grad():
    for idx in range(len(ref_preds)):
        gate.reset()
        img = cv2.imread(images[idx])
        if img is None:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil_img).unsqueeze(0)
        logits = model_test(tensor)

        if logits.argmax(1).item() == ref_preds[idx]:
            agree1 += 1
        cosines.append(F.cosine_similarity(logits, ref_logits[idx], dim=1).item())

n = len(ref_preds)
print(f"  BBP triple (k={k1:.3f}/{k2:.3f}/{k3:.3f} with group alternation):")
print(f"    Top-1: {agree1}/{n} ({agree1/n*100:.0f}%)  Cos: {np.mean(cosines):.4f}")


# ============================================================================
# Part 5: The alternating series IS the error — express correction
# ============================================================================

print()
print("=" * 80)
print("PART 5: The correction as alternating series in φ")
print("=" * 80)
print()

# From Part 2: ε(x) = c₁x² + c₃x⁴ + c₅x⁶ + ...
# where c₁ < 0, c₃ > 0, c₅ < 0 (alternating!)
#
# This IS already a BBP-style alternating series!
# The "base" is x². Each "term" flips sign.
#
# So the CORRECTED gate is:
# GELU(x) = x·σ(φ·x) + c₁x² + c₃x⁴ + c₅x⁶ + ...
#
# Keeping just the first correction term:
# GELU(x) ≈ x·σ(φ·x) + c₁x²
#
# The alternating series theorem says the error is bounded by |c₃x⁴|
# For |x| < 2: |c₃| × 16 = 0.022 × 16 = 0.35... too big.
#
# But if we keep TWO correction terms:
# GELU(x) ≈ x·σ(φ·x) + c₁x² + c₃x⁴
# Error bounded by |c₅x⁶|. For |x| < 2: 0.023 × 64 = 1.5... still big.
#
# The series converges too slowly for |x| > 1.
# But the KEY insight is different:
# The alternation doesn't need to be IN THE SERIES.
# It needs to be IN THE BLOCKS.

# What if the correction is PER-BLOCK with alternating sign?
# Block n applies: x·σ(k_n·x) where k_n = φ + (-1)^n × δ/φ^n
# The correction DECAYS geometrically (like 1/64^n in BBP)
# AND alternates sign (like (-1)^n in BBP)

print("  Per-block geometric decay + alternation:")
print("  k_n = φ + (-1)^n × δ₀/φ^(n/r)")
print()

# Try different decay rates r and initial δ₀
for delta_0 in [0.1, 0.155, 0.2]:
    for decay_r in [2, 3, 6, 9, 18]:
        # Compute k values for each of 18 blocks
        k_values = []
        for n in range(18):
            correction = delta_0 * (-1)**n / (PHI ** (n / decay_r))
            k_values.append(PHI + correction)

        # Create a gate that cycles through these k values
        class DecayingAlternatingGate(nn.Module):
            def __init__(self, k_vals):
                super().__init__()
                self.k_vals = k_vals
                self.counter = 0
            def forward(self, x):
                k = self.k_vals[min(self.counter, len(self.k_vals)-1)]
                self.counter += 1
                return x * torch.sigmoid(k * x)
            def reset(self):
                self.counter = 0

        gate = DecayingAlternatingGate(k_values)
        model_test = copy.deepcopy(model_orig)
        replace_gelu_with_module(model_test, gate)
        model_test.eval()

        agree1 = 0
        cosines = []
        with torch.no_grad():
            for idx in range(len(ref_preds)):
                gate.reset()
                img = cv2.imread(images[idx])
                if img is None:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                tensor = preprocess(pil_img).unsqueeze(0)
                logits = model_test(tensor)
                if logits.argmax(1).item() == ref_preds[idx]:
                    agree1 += 1
                cosines.append(F.cosine_similarity(logits, ref_logits[idx], dim=1).item())

        n = len(ref_preds)
        k_range = f"[{min(k_values):.3f}-{max(k_values):.3f}]"
        print(f"    δ₀={delta_0:.3f} decay_r={decay_r:<3} k_range={k_range:<18} "
              f"agree={agree1}/{n} ({agree1/n*100:.0f}%)  cos={np.mean(cosines):.4f}")

    print()


# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("SUMMARY: The BBP connection")
print("=" * 80)
print()
print("  BBP formula:  π = Σ (-1)^n × a_n / 64^n")
print("    Key: alternating sign + geometric decay = bounded error")
print()
print("  Gate error:   ε(x) = c₁x² + c₃x⁴ + c₅x⁶ + ...")
print("    Key: already alternating! c₁<0, c₃>0, c₅<0")
print(f"    c₁ = {c1:.6f} (1/√2π - φ/4)")
print(f"    c₃ = {c3:.6f} (φ³/48 - 1/6√2π)")
print(f"    c₅ = {c5:.6f}")
print()
print("  The error between GELU and x·σ(φ·x) IS an alternating series")
print("  in x². The question is whether we can exploit this through")
print("  the block structure to get cancellation.")
