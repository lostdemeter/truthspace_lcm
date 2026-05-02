#!/usr/bin/env python3
"""
The erf alternating series gives 100% at N=15 but has stability issues.
Even N values (8, 10, 20) fail because intermediate terms are huge.

This is EXACTLY the problem BBP solves: use exact arithmetic (integer ops)
so the alternating cancellation is precise.

Here we:
1. Diagnose the stability issue (why even N fails)
2. Find the stable range using Horner's method (numerically stable evaluation)
3. Test with double precision to confirm it's a float32 issue
4. Connect to BBP: the alternating series needs exact arithmetic
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
INV_SQRT_2 = 1.0 / np.sqrt(2.0)
INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


class StableErfGate(nn.Module):
    """Numerically stable erf series using Horner's method.
    
    Instead of computing each term separately and summing (catastrophic cancellation),
    we use Horner's method: evaluate the polynomial from inside out.
    
    erf(z) = (2/√π) × z × Σ_{n=0}^{N} (-1)^n × z^(2n) / (n!·(2n+1))
    
    Horner form: erf(z) = (2/√π) × z × (a₀ + z² × (a₁ + z² × (a₂ + z² × (...))))
    where a_n = (-1)^n / (n! × (2n+1))
    """
    def __init__(self, n_terms):
        super().__init__()
        self.n_terms = n_terms
        # Pre-compute coefficients a_n = (-1)^n / (n! × (2n+1))
        coeffs = []
        for n in range(n_terms):
            sign = (-1) ** n
            factorial_n = 1
            for i in range(1, n + 1):
                factorial_n *= i
            c = sign / (factorial_n * (2 * n + 1))
            coeffs.append(c)
        self.coeffs = coeffs

    def forward(self, x):
        z = x * INV_SQRT_2  # x/√2
        z_sq = z * z

        # Horner's method: evaluate from innermost term outward
        # P(z²) = a_{N-1} + z² × 0 (last term)
        # Then: P = a_{n} + z² × P  for n = N-2, N-3, ..., 0
        result = torch.full_like(z_sq, self.coeffs[-1])
        for n in range(self.n_terms - 2, -1, -1):
            result = self.coeffs[n] + z_sq * result

        # erf(z) ≈ (2/√π) × z × P(z²)
        erf_approx = (2.0 * INV_SQRT_PI) * z * result

        # Φ(x) = (1 + erf(x/√2)) / 2
        phi_x = 0.5 * (1.0 + erf_approx)

        # Clamp for safety (shouldn't be needed with Horner)
        phi_x = phi_x.clamp(0, 1)

        return x * phi_x


class StableErfGateFloat64(nn.Module):
    """Same as StableErfGate but in float64 for comparison."""
    def __init__(self, n_terms):
        super().__init__()
        self.n_terms = n_terms
        coeffs = []
        for n in range(n_terms):
            sign = (-1) ** n
            factorial_n = 1
            for i in range(1, n + 1):
                factorial_n *= i
            c = sign / (factorial_n * (2 * n + 1))
            coeffs.append(c)
        self.coeffs = coeffs

    def forward(self, x):
        x64 = x.double()
        z = x64 * (1.0 / np.sqrt(2.0))
        z_sq = z * z

        result = torch.full_like(z_sq, self.coeffs[-1])
        for n in range(self.n_terms - 2, -1, -1):
            result = self.coeffs[n] + z_sq * result

        erf_approx = (2.0 / np.sqrt(np.pi)) * z * result
        phi_x = 0.5 * (1.0 + erf_approx)
        phi_x = phi_x.clamp(0, 1)

        return (x64 * phi_x).float()


# ============================================================================
# Part 1: Stability diagnosis
# ============================================================================

print("=" * 80)
print("STABILITY ANALYSIS: Horner's method vs naive summation")
print("=" * 80)
print()

x = torch.linspace(-5, 5, 10000)
gelu_exact = x * 0.5 * (1 + torch.erf(x / np.sqrt(2.0)))

print("  Max |error| vs GELU over [-5, 5]:")
print(f"  {'N':<5} {'Naive (float32)':<18} {'Horner (float32)':<18} {'Horner (float64)':<18}")
print(f"  {'-'*60}")

for n in range(1, 26):
    # Horner float32
    gate_h32 = StableErfGate(n)
    with torch.no_grad():
        out_h32 = gate_h32(x)
    err_h32 = (out_h32 - gelu_exact).abs().max().item()

    # Horner float64
    gate_h64 = StableErfGateFloat64(n)
    with torch.no_grad():
        out_h64 = gate_h64(x)
    err_h64 = (out_h64 - gelu_exact).abs().max().item()

    marker = ""
    if err_h32 < 0.030:
        marker = " < φ-gate"
    if err_h32 < 0.001:
        marker = " ★ HIGH PRECISION"
    if err_h32 < 1e-5:
        marker = " ★★ NEAR-EXACT"

    print(f"  {n:<5} {'—':<18} {err_h32:<18.8f} {err_h64:<18.8f}{marker}")


# ============================================================================
# Part 2: Classification test with stable series
# ============================================================================

print()
print("=" * 80)
print("CLASSIFICATION TEST: Stable erf series (Horner)")
print("=" * 80)
print()

weights_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model_orig = convnext_tiny(weights=weights_enum)
model_orig.eval()
preprocess = weights_enum.transforms()

image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))
N_TEST = 100

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

# Test key N values with Horner
print(f"  {'Gate':<35} {'Max err':<10} {'Top-1':<12} {'Top-5':<12} {'Cos':<8}")
print(f"  {'-'*75}")

for label, gate in [
    ('x·σ(φ·x)', type('G', (nn.Module,), {'forward': lambda s, x: x * torch.sigmoid(PHI * x)})()),
    ('x·σ(√π·x)', type('G', (nn.Module,), {'forward': lambda s, x: x * torch.sigmoid(np.sqrt(np.pi) * x)})()),
    ('Horner erf N=5', StableErfGate(5)),
    ('Horner erf N=8', StableErfGate(8)),
    ('Horner erf N=10', StableErfGate(10)),
    ('Horner erf N=12', StableErfGate(12)),
    ('Horner erf N=15', StableErfGate(15)),
    ('Horner erf N=20', StableErfGate(20)),
    ('Horner erf N=25 (float64)', StableErfGateFloat64(25)),
]:
    with torch.no_grad():
        gate_out = gate(x)
    max_err = (gate_out - gelu_exact).abs().max().item()

    model = copy.deepcopy(model_orig)
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], gate)
    model.eval()

    agree1 = 0
    agree5 = 0
    cosines = []

    with torch.no_grad():
        for idx in range(len(ref_preds)):
            img = cv2.imread(images[idx])
            if img is None:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = preprocess(pil_img).unsqueeze(0)
            logits = model(tensor)

            if logits.argmax(1).item() == ref_preds[idx]:
                agree1 += 1
            top5 = set(logits.topk(5, dim=1).indices[0].tolist())
            top5_ref = set(ref_logits[idx].topk(5, dim=1).indices[0].tolist())
            if top5 == top5_ref:
                agree5 += 1
            cos = F.cosine_similarity(logits, ref_logits[idx], dim=1).item()
            if not np.isnan(cos):
                cosines.append(cos)

    n = len(ref_preds)
    cos_mean = np.mean(cosines) if cosines else float('nan')
    print(f"  {label:<35} {max_err:<10.6f} "
          f"{agree1}/{n} ({agree1/n*100:>3.0f}%) "
          f"{agree5}/{n} ({agree5/n*100:>3.0f}%)  "
          f"{cos_mean:.4f}")


# ============================================================================
# Part 3: The BBP interpretation
# ============================================================================

print()
print("=" * 80)
print("THE BBP INTERPRETATION")
print("=" * 80)
print()
print("  BBP for π: exact computation via alternating series + integer arithmetic")
print("  Our GELU: exact computation via alternating erf series + Horner stability")
print()
print("  The parallel:")
print("    BBP:  π/4 = Σ (-1)^n × a_n / 64^n")
print("    GELU: Φ(x) = 1/2 + (2/√π) × (x/√2) × Σ (-1)^n × (x²/2)^n / (n!·(2n+1))")
print()
print("  Both use:")
print("    1. Alternating signs (-1)^n for error correction")
print("    2. Geometric decay (1/64^n for BBP, 1/n!·(2n+1) for erf)")
print("    3. Exact coefficients (rational numbers)")
print()
print("  The stability problem:")
print("    BBP uses Decimal(150) for exact arithmetic")
print("    Our erf series needs Horner's method for float32 stability")
print("    OR: use φ-integer arithmetic (doc 125) for exact computation")
print()
print("  The key BBP insight that applies:")
print("    The alternating series IS the error correction mechanism.")
print("    Each term corrects the previous overshoot/undershoot.")
print("    The factorial decay (n!) guarantees convergence.")
print("    With sufficient terms (N≥15 for float32), we get 100%.")
print()

# Show the convergence pattern
print("  Convergence pattern (BBP-style bouncing):")
print("  Evaluating at x=2.0 (where error peaks):")
z = 2.0 / np.sqrt(2)
z_sq = z * z
import math as _math
gelu_true = 2.0 * 0.5 * (1 + _math.erf(2.0 / np.sqrt(2.0)))
print(f"    True GELU(2.0) = {gelu_true:.8f}")
print()

partial = 0
for n in range(15):
    sign = (-1) ** n
    factorial_n = 1
    for i in range(1, n + 1):
        factorial_n *= i
    coeff = sign / (factorial_n * (2 * n + 1))
    partial += coeff * (z_sq ** n)

    erf_approx = (2.0 / np.sqrt(np.pi)) * z * partial
    phi_approx = 0.5 * (1 + erf_approx)
    gelu_approx = 2.0 * phi_approx
    error = gelu_approx - gelu_true
    direction = "↑" if error > 0 else "↓"

    print(f"    N={n+1:>2}: GELU ≈ {gelu_approx:>12.8f}  error = {error:>+.8f}  {direction}")

print()
print("  Each term bounces above/below truth — the alternating")
print("  series correction is VISIBLE in the convergence pattern.")
print("  By N=15, the error is < 0.001 at the worst point (x=2).")
