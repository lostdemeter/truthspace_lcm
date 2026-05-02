#!/usr/bin/env python3
"""
Geometric Uppercase → Lowercase Converter

Build a neural network from GEOMETRY, not training.
Derive every weight from the problem structure.
Use the Ideal Gate as the only nonlinearity.

The problem:
  - Input: ASCII code (0-127)
  - Output: if uppercase (65-90), add 32. Otherwise pass through.
  - f(x) = x + 32 · rect(x; 65, 90)

The geometry:
  rect(x; a, b) = step(x - a) - step(x - b)

  A smooth step centered at c (between integers) with sharpness s:
    step_s(x, c) = [gate(s(x - (c-0.5))) - gate(s(x - (c+0.5)))] / s

  This works because gate(s·z) ≈ s·z for z > 0 and ≈ 0 for z < 0.
  So gate(s(x-a)) - gate(s(x-b)) ≈ s for x > b, ≈ 0 for x < a.

  KEY: Transitions must happen BETWEEN integers, so all integers
  are either fully inside or fully outside the rectangle.

Architecture:
  Residual block: output = x + W₂ · gate(W₁ · x + b₁)

  4 hidden neurons construct 2 smooth steps:
    h₁ = gate(s · (x - 64))   ┐
    h₂ = gate(s · (x - 65))   ┘ → step centered at 64.5
    h₃ = gate(s · (x - 90))   ┐
    h₄ = gate(s · (x - 91))   ┘ → step centered at 90.5

  Output weights:
    W₂ = [32/s, -32/s, -32/s, 32/s]

  output = x + (32/s) · (h₁ - h₂ - h₃ + h₄)
         = x + 32 · (step_64.5 - step_90.5)
         = x + 32 · rect(x; 65, 90)

Every weight is derived. Nothing is trained. Fail fast.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PHI = (1 + np.sqrt(5)) / 2
SQRT_8_OVER_PI = np.sqrt(8.0 / np.pi)
C_GEOMETRIC = (4 - np.pi) / (6 * np.pi)


def ideal_gate(x):
    """The Ideal Gate: gate(x) = x · σ(√(8/π) · x · (1 + C·x²))"""
    f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
    return x * torch.sigmoid(f)


class GeometricBlock(nn.Module):
    """A single geometric residual block with derived weights."""

    def __init__(self, W1, b1, W2, b2=None):
        super().__init__()
        self.W1 = nn.Parameter(torch.tensor(W1, dtype=torch.float32), requires_grad=False)
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32), requires_grad=False)
        self.W2 = nn.Parameter(torch.tensor(W2, dtype=torch.float32), requires_grad=False)
        if b2 is not None:
            self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32), requires_grad=False)
        else:
            self.b2 = None

    def forward(self, x):
        h = x @ self.W1.T + self.b1  # [batch, hidden]
        h = ideal_gate(h)             # Apply Ideal Gate
        out = h @ self.W2.T           # [batch, output]
        if self.b2 is not None:
            out = out + self.b2
        return x + out                # Residual connection


def build_uppercase_converter(sharpness):
    """
    Build the converter by deriving weights from problem geometry.

    The rectangle function rect(x; 65, 91) is constructed from
    two smooth steps, each made from a pair of shifted ramps.

    Args:
        sharpness: s, controls transition sharpness. Higher = sharper.
    """
    s = sharpness

    # Layer 1: [input_dim=1] → [hidden_dim=4]
    # Each neuron detects a threshold crossing
    W1 = [[s],      # neuron 1: s · x
           [s],      # neuron 2: s · x
           [s],      # neuron 3: s · x
           [s]]      # neuron 4: s · x

    b1 = [-s * 64.0,   # ramp at 64 (lower step, left edge)
           -s * 65.0,   # ramp at 65 (lower step, right edge)
           -s * 90.0,   # ramp at 90 (upper step, left edge)
           -s * 91.0]   # ramp at 91 (upper step, right edge)

    # Layer 2: [hidden_dim=4] → [output_dim=1]
    # Combine to make rectangle × 32
    W2 = [[32.0 / s,    # +step_low_rise
           -32.0 / s,   # -step_low_fall  → step at 65
           -32.0 / s,   # -step_high_rise → inverted
            32.0 / s]]  # +step_high_fall → step at 91

    return GeometricBlock(W1, b1, W2)


# ============================================================================
# Test the converter
# ============================================================================

print("=" * 70)
print("GEOMETRIC UPPERCASE → LOWERCASE CONVERTER")
print("All weights derived from problem geometry. Nothing trained.")
print("=" * 70)
print()

# Test across different sharpness values
sharpness_values = [1.0, PHI, PHI**2, PHI**3, 5.0, 10.0, PHI**5, 50.0, 100.0]
results = {}

for s in sharpness_values:
    model = build_uppercase_converter(s)

    # Test all 128 ASCII values
    inputs = torch.arange(128).float().unsqueeze(1)  # [128, 1]
    with torch.no_grad():
        outputs = model(inputs)

    outputs_rounded = outputs.squeeze().numpy()

    # Ground truth
    gt = np.arange(128, dtype=float)
    for i in range(65, 91):
        gt[i] += 32

    # Exact match (after rounding to nearest int)
    predicted = np.round(outputs_rounded)
    exact_match = (predicted == gt).sum()

    # Raw error
    raw_error = np.abs(outputs_rounded - gt)
    max_error = raw_error.max()
    mean_error = raw_error.mean()

    # Error specifically on uppercase letters
    upper_error = raw_error[65:91]
    upper_max = upper_error.max()
    upper_mean = upper_error.mean()

    # Error on non-uppercase (should be identity)
    non_upper_mask = np.ones(128, dtype=bool)
    non_upper_mask[65:91] = False
    non_upper_error = raw_error[non_upper_mask]
    non_upper_max = non_upper_error.max()

    results[s] = {
        'exact': exact_match,
        'max_error': max_error,
        'mean_error': mean_error,
        'upper_max': upper_max,
        'upper_mean': upper_mean,
        'non_upper_max': non_upper_max,
        'outputs': outputs_rounded.copy(),
        'raw_error': raw_error.copy(),
    }

    phi_label = ""
    for k in range(-2, 10):
        if abs(s - PHI**k) < 0.001:
            phi_label = f" (φ^{k})"
            break

    print(f"  s={s:8.3f}{phi_label:8s}: exact={exact_match}/128  "
          f"max_err={max_error:.4f}  upper_max_err={upper_max:.4f}  "
          f"passthru_max_err={non_upper_max:.6f}")

# Find the minimum sharpness for perfect conversion
print()
print("-" * 70)
print("Finding minimum sharpness for 128/128 exact match...")
for s_test in np.arange(0.5, 20.0, 0.1):
    model = build_uppercase_converter(s_test)
    inputs = torch.arange(128).float().unsqueeze(1)
    with torch.no_grad():
        outputs = model(inputs)
    gt = np.arange(128, dtype=float)
    for i in range(65, 91):
        gt[i] += 32
    predicted = np.round(outputs.squeeze().numpy())
    if (predicted == gt).all():
        phi_power = np.log(s_test) / np.log(PHI)
        print(f"  Minimum sharpness for perfect: s = {s_test:.1f} "
              f"(φ^{phi_power:.2f})")
        break


# ============================================================================
# Detailed analysis at φ² sharpness
# ============================================================================

print()
print("=" * 70)
print(f"DETAILED ANALYSIS at s = φ² = {PHI**2:.4f}")
print("=" * 70)
print()

s_detail = PHI ** 2
model = build_uppercase_converter(s_detail)

# Print the actual weights
print("  DERIVED WEIGHTS (from geometry):")
print(f"    W1 = {model.W1.data.numpy().flatten()}")
print(f"    b1 = {model.b1.data.numpy()}")
print(f"    W2 = {model.W2.data.numpy().flatten()}")
print()

# Show conversion for all printable ASCII
print("  CONVERSION TABLE (printable ASCII):")
inputs = torch.arange(128).float().unsqueeze(1)
with torch.no_grad():
    outputs = model(inputs)

gt = np.arange(128, dtype=float)
for i in range(65, 91):
    gt[i] += 32

for i in range(32, 128):
    inp_char = chr(i)
    out_val = outputs[i].item()
    out_rounded = int(round(out_val))
    out_char = chr(out_rounded) if 32 <= out_rounded < 128 else '?'
    expected = int(gt[i])
    exp_char = chr(expected) if 32 <= expected < 128 else '?'
    error = abs(out_val - gt[i])
    marker = " ★" if error > 0.1 else ""
    if 65 <= i <= 90 or error > 0.01:
        print(f"    {i:3d} '{inp_char}' → {out_val:7.3f} → '{out_char}' "
              f"(expected '{exp_char}' = {expected})  err={error:.4f}{marker}")


# ============================================================================
# φ-structure analysis of the weights
# ============================================================================

print()
print("=" * 70)
print("φ-STRUCTURE OF THE WEIGHTS")
print("=" * 70)
print()

print("  The weights encode the problem geometry:")
print(f"    Sharpness s = φ² = {PHI**2:.6f}")
print(f"    Step thresholds: 64.5, 65.5, 90.5, 91.5")
print(f"    Step width: 1.0 (transition zone)")
print(f"    Rectangle width: 91 - 65 = 26")
print(f"    Offset: 32 = 2⁵")
print()
print("  Geometric ratios:")
print(f"    Offset / Rectangle width = 32 / 26 = {32/26:.6f}")
print(f"    φ² / φ = φ = {PHI:.6f}")
print(f"    26 = 2 × 13")
print(f"    32 = 2⁵")
print()
print("  Weight magnitudes at s=φ²:")
print(f"    W1 entries: {PHI**2:.4f}")
print(f"    b1 entries: {-PHI**2 * 64:.2f}, {-PHI**2 * 65:.2f}, "
      f"{-PHI**2 * 90:.2f}, {-PHI**2 * 91:.2f}")
print(f"    W2 entries: ±{32/PHI**2:.4f} = ±32/φ² = ±{32/PHI**2:.4f}")
print(f"    32/φ² = {32/PHI**2:.6f}")
print(f"    32×φ⁻² = {32 * PHI**(-2):.6f}")
print()


# ============================================================================
# Can we do it WITHOUT the skip connection?
# ============================================================================

print("=" * 70)
print("WITHOUT SKIP CONNECTION: Can the gate do identity?")
print("=" * 70)
print()

class NoSkipBlock(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = nn.Parameter(torch.tensor(W1, dtype=torch.float32), requires_grad=False)
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32), requires_grad=False)
        self.W2 = nn.Parameter(torch.tensor(W2, dtype=torch.float32), requires_grad=False)
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        h = x @ self.W1.T + self.b1
        h = ideal_gate(h)
        return h @ self.W2.T + self.b2


# 5 neurons: 1 for identity pass-through, 4 for step detection
s = PHI ** 2
W1_ns = [[1.0],   # identity neuron
          [s],     [s],     [s],     [s]]
b1_ns = [0.0,     -s*64.0, -s*65.0, -s*90.0, -s*91.0]
W2_ns = [[1.0, 32.0/s, -32.0/s, -32.0/s, 32.0/s]]
b2_ns = [0.0]

model_ns = NoSkipBlock(W1_ns, b1_ns, W2_ns, b2_ns)

inputs = torch.arange(128).float().unsqueeze(1)
with torch.no_grad():
    outputs_ns = model_ns(inputs)

out_ns = outputs_ns.squeeze().numpy()
gt = np.arange(128, dtype=float)
for i in range(65, 91):
    gt[i] += 32

error_ns = np.abs(out_ns - gt)
predicted_ns = np.round(out_ns)
exact_ns = (predicted_ns == gt).sum()

print(f"  No-skip (5 neurons, s=φ²):")
print(f"    Exact match: {exact_ns}/128")
print(f"    Max error: {error_ns.max():.6f}")
print(f"    gate(x) ≈ x error for ASCII 0: {abs(ideal_gate(torch.tensor(0.0)).item() - 0.0):.6f}")
print(f"    gate(x) ≈ x error for ASCII 1: {abs(ideal_gate(torch.tensor(1.0)).item() - 1.0):.6f}")
print(f"    gate(x) ≈ x error for ASCII 10: {abs(ideal_gate(torch.tensor(10.0)).item() - 10.0):.6f}")
print(f"    gate(x) ≈ x error for ASCII 65: {abs(ideal_gate(torch.tensor(65.0)).item() - 65.0):.6f}")
print()
print(f"  Identity neuron error at small values:")
for val in [0, 1, 2, 3, 5, 10, 32]:
    gate_val = ideal_gate(torch.tensor(float(val))).item()
    print(f"    gate({val}) = {gate_val:.6f}  (error: {abs(gate_val - val):.6f})")


# ============================================================================
# SCALING TEST: Can this approach scale?
# ============================================================================

print()
print("=" * 70)
print("SCALING TEST: ROT13 — a more complex mapping")
print("=" * 70)
print()

# ROT13: shift by 13 in the alphabet, wrapping
# A(65) → N(78), ..., M(77) → Z(90), N(78) → A(65), ..., Z(90) → M(77)
# a(97) → n(110), etc.
# This requires TWO rectangles per case (upper/lower)

def build_rot13_converter(s):
    """
    ROT13 = shift by 13, wrapping at boundaries.
    Uppercase A-M (65-77) → +13
    Uppercase N-Z (78-90) → -13
    Lowercase a-m (97-109) → +13
    Lowercase n-z (110-122) → -13

    4 rectangles, 2 neurons each = 8 step-detection neurons.
    """
    W1 = [[s]] * 16  # 16 neurons (2 per step, 8 steps)

    # 4 rectangles:
    # rect(65, 77): A-M → +13, step between 64/65 and 77/78
    # rect(78, 90): N-Z → -13, step between 77/78 and 90/91
    # rect(97, 109): a-m → +13, step between 96/97 and 109/110
    # rect(110, 122): n-z → -13, step between 109/110 and 122/123

    b1 = [
        -s * 64.0, -s * 65.0,   # step centered at 64.5
        -s * 77.0, -s * 78.0,   # step centered at 77.5
        -s * 77.0, -s * 78.0,   # step centered at 77.5 (reused)
        -s * 90.0, -s * 91.0,   # step centered at 90.5
        -s * 96.0, -s * 97.0,   # step centered at 96.5
        -s * 109.0, -s * 110.0, # step centered at 109.5
        -s * 109.0, -s * 110.0, # step centered at 109.5 (reused)
        -s * 122.0, -s * 123.0, # step centered at 122.5
    ]

    # Output: +13 for rect(65,78), -13 for rect(78,91),
    #         +13 for rect(97,110), -13 for rect(110,123)
    c = 1.0 / s
    W2 = [[
        13*c, -13*c,   -13*c, 13*c,   # rect(65,78) → +13
        -13*c, 13*c,    13*c, -13*c,  # rect(78,91) → -13
        13*c, -13*c,   -13*c, 13*c,   # rect(97,110) → +13
        -13*c, 13*c,    13*c, -13*c,  # rect(110,123) → -13
    ]]

    return GeometricBlock(W1, b1, W2)


# Ground truth ROT13
gt_rot13 = np.arange(128, dtype=float)
for i in range(65, 91):
    gt_rot13[i] = 65 + (i - 65 + 13) % 26
for i in range(97, 123):
    gt_rot13[i] = 97 + (i - 97 + 13) % 26

for s in [PHI**2, PHI**3, 10.0, PHI**5, 50.0]:
    model_rot = build_rot13_converter(s)
    inputs = torch.arange(128).float().unsqueeze(1)
    with torch.no_grad():
        outputs_rot = model_rot(inputs)
    out_rot = outputs_rot.squeeze().numpy()
    predicted_rot = np.round(out_rot)
    exact_rot = (predicted_rot == gt_rot13).sum()
    error_rot = np.abs(out_rot - gt_rot13)

    phi_label = ""
    for k in range(-2, 10):
        if abs(s - PHI**k) < 0.001:
            phi_label = f" (φ^{k})"
            break

    print(f"  ROT13 s={s:8.3f}{phi_label:8s}: exact={exact_rot}/128  "
          f"max_err={error_rot.max():.4f}")

# Verify ROT13 at best sharpness
s_best = PHI ** 5
model_rot = build_rot13_converter(s_best)
inputs = torch.arange(128).float().unsqueeze(1)
with torch.no_grad():
    outputs_rot = model_rot(inputs)
out_rot = np.round(outputs_rot.squeeze().numpy())

print(f"\n  ROT13 verification at s=φ⁵:")
test_str = "HELLO WORLD"
encoded = ""
for ch in test_str:
    i = ord(ch)
    o = int(out_rot[i])
    encoded += chr(o) if 0 <= o < 128 else '?'
print(f"    '{test_str}' → '{encoded}'")

# Double-encode should return original
encoded2 = ""
for ch in encoded:
    i = ord(ch)
    o = int(out_rot[i])
    encoded2 += chr(o) if 0 <= o < 128 else '?'
print(f"    '{encoded}' → '{encoded2}'  (should match original)")
print(f"    Round-trip correct: {encoded2 == test_str}")


# ============================================================================
# WEIGHT ANALYSIS: Structure of derived weights
# ============================================================================

print()
print("=" * 70)
print("WEIGHT STRUCTURE ANALYSIS")
print("=" * 70)
print()

s = PHI ** 2
model = build_uppercase_converter(s)

W1_vals = model.W1.data.numpy()
b1_vals = model.b1.data.numpy()
W2_vals = model.W2.data.numpy()

print("  Uppercase converter at s=φ²:")
print(f"    W1: all entries = φ² = {PHI**2:.6f}")
print(f"    b1: thresholds × (-φ²)")
print(f"    W2: all entries = ±32/φ² = ±{32/PHI**2:.6f}")
print()
print("  The weight magnitudes form a φ-hierarchy:")
print(f"    |W1| = φ²  = {PHI**2:.4f}")
print(f"    |W2| = 32/φ² = {32/PHI**2:.4f} = 32 × φ⁻²")
print(f"    |W1| × |W2| = 32 (the offset!) ← geometry preserved")
print(f"    |W1| / |W2| = φ⁴ / 32 = {PHI**4/32:.6f}")
print()
print("  Number of parameters:")
print(f"    W1: 4 weights (all identical = φ²)")
print(f"    b1: 4 biases (thresholds)")
print(f"    W2: 4 weights (all ±32/φ²)")
print(f"    Total: 12 parameters")
print(f"    Unique values: 3 (φ², 32/φ², and thresholds)")
print()

total_params = 4 + 4 + 4  # W1, b1, W2
print(f"  Bits of information in the weights:")
print(f"    4 threshold values: log2(128) × 4 = {np.log2(128)*4:.0f} bits")
print(f"    1 sharpness value: ~7 bits")
print(f"    1 offset value (32): 5 bits")
print(f"    4 signs in W2: 4 bits")
print(f"    Total: ~{7*4 + 7 + 5 + 4:.0f} bits of structured information")
print(f"    This IS the geometry of the problem, encoded in weights.")


# ============================================================================
# Visualization
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1: Input→Output mapping at different sharpness
ax1 = fig.add_subplot(gs[0, 0])
x_range = np.arange(128)
gt = np.arange(128, dtype=float)
for i in range(65, 91):
    gt[i] += 32
ax1.plot(x_range, gt, 'k--', linewidth=2, label='Ground truth', alpha=0.5)
for s_key in [1.0, PHI**2, 10.0, 100.0]:
    if s_key in results:
        ax1.plot(x_range, results[s_key]['outputs'], '-', linewidth=1.5,
                 label=f's={s_key:.1f}', alpha=0.8)
ax1.set_xlabel('Input ASCII')
ax1.set_ylabel('Output ASCII')
ax1.set_title('Geometric Converter\nInput → Output')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Panel 2: Error vs sharpness
ax2 = fig.add_subplot(gs[0, 1])
s_list = sorted(results.keys())
exact_list = [results[s]['exact'] for s in s_list]
max_err_list = [results[s]['max_error'] for s in s_list]
ax2.semilogx(s_list, exact_list, 'go-', linewidth=2, markersize=8)
ax2.axhline(y=128, color='green', linestyle='--', alpha=0.5)
ax2.set_xlabel('Sharpness (s)')
ax2.set_ylabel('Exact matches / 128')
ax2.set_title('Accuracy vs Sharpness')
# Mark φ powers
for k in range(0, 6):
    pk = PHI ** k
    if pk <= max(s_list):
        ax2.axvline(x=pk, color='gold', alpha=0.3, linestyle=':')
        ax2.text(pk, 126, f'φ^{k}', fontsize=7, ha='center', color='goldenrod')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(100, 130)

# Panel 3: Zoomed error around uppercase range
ax3 = fig.add_subplot(gs[0, 2])
s_detail = PHI ** 2
model_d = build_uppercase_converter(s_detail)
x_fine = torch.linspace(60, 95, 500).unsqueeze(1)
with torch.no_grad():
    y_fine = model_d(x_fine).squeeze().numpy()
x_fine_np = x_fine.squeeze().numpy()
gt_fine = x_fine_np.copy()
gt_fine[(x_fine_np >= 65) & (x_fine_np <= 90)] += 32
ax3.plot(x_fine_np, gt_fine, 'k--', linewidth=2, label='Ground truth')
ax3.plot(x_fine_np, y_fine, 'b-', linewidth=2, label=f's=φ²={PHI**2:.2f}')
# Mark integer positions
for i in range(60, 96):
    marker_col = 'red' if 65 <= i <= 90 else 'gray'
    ax3.plot(i, gt[i], 'o', color=marker_col, markersize=4, alpha=0.5)
ax3.set_xlabel('ASCII code')
ax3.set_ylabel('Output')
ax3.set_title(f'Transition Detail (s=φ²)\nA(65)→a(97), Z(90)→z(122)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Hidden neuron activations
ax4 = fig.add_subplot(gs[0, 3])
s_vis = PHI ** 2
model_v = build_uppercase_converter(s_vis)
x_vis = torch.linspace(55, 100, 500).unsqueeze(1)
with torch.no_grad():
    h_vis = x_vis @ model_v.W1.T + model_v.b1
    h_gated = ideal_gate(h_vis)
x_vis_np = x_vis.squeeze().numpy()
colors_h = ['blue', 'cyan', 'red', 'orange']
labels_h = ['gate(s(x-64.5))', 'gate(s(x-65.5))', 'gate(s(x-90.5))', 'gate(s(x-91.5))']
for i in range(4):
    ax4.plot(x_vis_np, h_gated[:, i].numpy(), '-', color=colors_h[i],
             linewidth=1.5, label=labels_h[i])
ax4.set_xlabel('Input ASCII')
ax4.set_ylabel('Gated activation')
ax4.set_title('Hidden Neurons\n(4 threshold detectors)')
ax4.legend(fontsize=6)
ax4.grid(True, alpha=0.3)
ax4.axvline(x=65, color='gray', linestyle=':', alpha=0.5)
ax4.axvline(x=91, color='gray', linestyle=':', alpha=0.5)

# Panel 5: The constructed rectangle function
ax5 = fig.add_subplot(gs[1, 0])
with torch.no_grad():
    correction = (h_gated @ model_v.W2.T).squeeze().numpy()
ax5.plot(x_vis_np, correction, 'green', linewidth=2)
ax5.axhline(y=32, color='red', linestyle='--', alpha=0.5, label='Target = 32')
ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax5.axvline(x=65, color='gray', linestyle=':', alpha=0.5)
ax5.axvline(x=91, color='gray', linestyle=':', alpha=0.5)
ax5.set_xlabel('Input ASCII')
ax5.set_ylabel('Correction (added to input)')
ax5.set_title('Constructed Rectangle\n32 × rect(x; 65, 91)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Panel 6: Per-character error
ax6 = fig.add_subplot(gs[1, 1])
error_detail = results[PHI**2]['raw_error']
colors_err = ['red' if 65 <= i <= 90 else 'steelblue' for i in range(128)]
ax6.bar(range(128), error_detail, color=colors_err, alpha=0.7)
ax6.set_xlabel('ASCII code')
ax6.set_ylabel('Absolute error')
ax6.set_title(f'Per-Character Error (s=φ²)\nRed = uppercase target')
ax6.grid(True, alpha=0.3)

# Panel 7: ROT13 mapping
ax7 = fig.add_subplot(gs[1, 2])
model_rot_vis = build_rot13_converter(PHI**5)
inputs_rot = torch.arange(128).float().unsqueeze(1)
with torch.no_grad():
    out_rot_vis = model_rot_vis(inputs_rot).squeeze().numpy()
ax7.plot(range(128), gt_rot13, 'k--', linewidth=2, label='ROT13 truth', alpha=0.5)
ax7.plot(range(128), out_rot_vis, 'r-', linewidth=1.5, label=f's=φ⁵', alpha=0.8)
ax7.set_xlabel('Input ASCII')
ax7.set_ylabel('Output ASCII')
ax7.set_title('ROT13 (scaled up!)\n16 neurons, same architecture')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Panel 8: Weight structure diagram
ax8 = fig.add_subplot(gs[1, 3])
ax8.axis('off')
arch_text = (
    "ARCHITECTURE\n"
    "═══════════════════════════════\n\n"
    "Input (1D: ASCII code)\n"
    "       │\n"
    "       ├─── skip connection ──┐\n"
    "       │                      │\n"
    "   [W₁ · x + b₁]            │\n"
    "   4 neurons:                 │\n"
    "   • gate(s(x-64.5))         │\n"
    "   • gate(s(x-65.5))         │\n"
    "   • gate(s(x-90.5))         │\n"
    "   • gate(s(x-91.5))         │\n"
    "       │                      │\n"
    "   [Ideal Gate]               │\n"
    "       │                      │\n"
    "   [W₂ · h]                   │\n"
    "   = 32·rect(x;65,91)        │\n"
    "       │                      │\n"
    "       └──── + ───────────────┘\n"
    "       │\n"
    "Output (1D: converted)\n\n"
    "12 parameters. 0 trained."
)
ax8.text(0.05, 0.95, arch_text, transform=ax8.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 9-10: String conversion demo
ax9 = fig.add_subplot(gs[2, 0:2])
ax9.axis('off')

# Demo string conversion
s_demo = PHI ** 3
model_demo = build_uppercase_converter(s_demo)
demo_strings = [
    "HELLO WORLD",
    "The Quick Brown Fox",
    "ASCII 2025",
    "GEOMETRIC STRUCTURE",
    "φ-GATE CONVERTER",
]
demo_text = "STRING CONVERSION DEMO (s=φ³)\n" + "─" * 50 + "\n\n"
for ds in demo_strings:
    inputs_str = torch.tensor([ord(c) for c in ds], dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        outputs_str = model_demo(inputs_str)
    out_chars = ""
    for val in outputs_str:
        c = int(round(val.item()))
        out_chars += chr(c) if 0 <= c < 128 else '?'
    demo_text += f"  '{ds}'\n  → '{out_chars}'\n\n"

ax9.text(0.05, 0.95, demo_text, transform=ax9.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

# Panel 11-12: Summary
ax10 = fig.add_subplot(gs[2, 2:4])
ax10.axis('off')
summary_text = (
    "GEOMETRIC NETWORK SUMMARY\n"
    "═════════════════════════════════════\n\n"
    "WHAT: Uppercase → Lowercase converter\n"
    "HOW:  All weights derived from geometry\n"
    "GATE: The Ideal Gate (only nonlinearity)\n\n"
    "KEY RESULTS:\n"
    f"  • 128/128 exact at s ≥ ~3.5 (φ² = {PHI**2:.2f})\n"
    "  • 12 parameters, 0 trained\n"
    "  • Skip + 4 hidden neurons\n"
    "  • Scales to ROT13 (16 neurons)\n\n"
    "WEIGHT STRUCTURE:\n"
    f"  W₁ = φ² = {PHI**2:.4f} (sharpness)\n"
    f"  W₂ = 32/φ² = {32/PHI**2:.4f} (offset/sharpness)\n"
    "  W₁ × W₂ = 32 (the offset)\n\n"
    "WHY IT WORKS:\n"
    "  • Gate ≈ ReLU for large inputs (identity)\n"
    "  • Gate ≈ 0 for negative inputs (blocking)\n"
    "  • Ramp pairs → smooth steps\n"
    "  • Steps → rectangles → offsets\n"
    "  • Skip connection → identity passthrough\n\n"
    "NEXT: Scale to word embeddings,\n"
    "      multi-character sequences,\n"
    "      semantic operations."
)
ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Geometric Uppercase→Lowercase: Structure IS Information\n'
             'Every weight derived from problem geometry. Nothing trained.',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('/tmp/geometric_uppercase.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print()
print("Saved: /tmp/geometric_uppercase.png")
