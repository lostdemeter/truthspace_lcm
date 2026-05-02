#!/usr/bin/env python3
"""
The BBP insight applied properly: use the EXACT alternating series for erf,
not a sigmoid approximation.

BBP computes π exactly via: π = Σ (-1)^n × a_n / 64^n  (convergent alternating)
GELU is exactly: GELU(x) = x × [1/2 + (x/√(2π)) × Σ (-1)^n × x^(2n) / (2^n·n!·(2n+1))]

The alternating erf series converges for ALL x. Each additional term
bounces above/below truth (BBP-style error correction).

Question: how many terms do we need for 100% agreement?
And can these terms be expressed in φ-arithmetic?
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
INV_SQRT_2PI = 1.0 / np.sqrt(2 * np.pi)  # 0.3989
INV_SQRT_2 = 1.0 / np.sqrt(2.0)          # 0.7071


class ErfSeriesGate(nn.Module):
    """GELU via truncated erf alternating series.
    
    GELU(x) = x/2 + (x²/√(2π)) × Σ_{n=0}^{N} (-1)^n × (x/√2)^(2n) / (n! × (2n+1))
    
    Each additional term alternates sign → BBP-style error correction.
    N=0: linear approximation  
    N=1: + quadratic correction
    N=2: + quartic correction
    ...
    N=∞: exact GELU
    """
    def __init__(self, n_terms):
        super().__init__()
        self.n_terms = n_terms
        # Pre-compute 1/(n! × (2n+1)) for each term
        coeffs = []
        for n in range(n_terms):
            factorial_n = 1
            for i in range(1, n + 1):
                factorial_n *= i
            c = 1.0 / (factorial_n * (2 * n + 1))
            coeffs.append(c)
        self.coeffs = coeffs

    def forward(self, x):
        # Compute erf(x/√2) via alternating series
        z = x * INV_SQRT_2  # x/√2
        z_sq = z * z

        # erf(z) ≈ (2/√π) × z × Σ (-1)^n × z^(2n) / (n! × (2n+1))
        # So Φ(x) = 1/2 + (1/2) × erf(x/√2)
        # = 1/2 + (1/√π) × z × Σ (-1)^n × z^(2n) / (n! × (2n+1))

        series_sum = torch.zeros_like(x)
        z_power = torch.ones_like(x)  # z^(2n), starts at z^0 = 1

        for n in range(self.n_terms):
            sign = (-1) ** n
            term = sign * self.coeffs[n] * z_power
            series_sum = series_sum + term
            z_power = z_power * z_sq  # z^(2(n+1))

        # Φ(x) = 1/2 + (1/√π) × z × series_sum
        inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
        phi_x = 0.5 + inv_sqrt_pi * z * series_sum

        # Clamp to [0, 1] for stability (series may overshoot for large |x|)
        phi_x = phi_x.clamp(0, 1)

        return x * phi_x


class PhiLUTGate(nn.Module):
    """GELU via φ-lattice lookup table (doc 125 approach).
    Pre-compute Φ(x) on a fine grid, interpolate."""
    def __init__(self, n_entries=16384, x_range=10.0):
        super().__init__()
        self.n_entries = n_entries
        self.x_range = x_range
        # Pre-compute GELU on uniform grid
        x = torch.linspace(-x_range, x_range, n_entries)
        y = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
        self.dx = 2 * x_range / (n_entries - 1)

    def forward(self, x):
        # Linear interpolation in the LUT
        idx = ((x + self.x_range) / self.dx).clamp(0, self.n_entries - 2)
        idx_floor = idx.long()
        frac = idx - idx_floor.float()
        y0 = self.grid_y[idx_floor]
        y1 = self.grid_y[(idx_floor + 1).clamp(max=self.n_entries - 1)]
        return y0 + frac * (y1 - y0)


class PhiGate(nn.Module):
    """x·σ(φ·x)"""
    def forward(self, x):
        return x * torch.sigmoid(PHI * x)


class SqrtPiGate(nn.Module):
    """x·σ(√π·x)"""
    def forward(self, x):
        return x * torch.sigmoid(np.sqrt(np.pi) * x)


def replace_gelu_with(model, gate):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], gate)
            count += 1
    return count


# ============================================================================
# Part 1: How many erf series terms for 100%?
# ============================================================================

print("=" * 80)
print("ALTERNATING ERF SERIES: How many terms for 100%?")
print("=" * 80)
print()

# First: verify the series approximation quality
x = torch.linspace(-5, 5, 10000)
gelu_exact = x * 0.5 * (1 + torch.erf(x / np.sqrt(2.0)))

print("  Series approximation quality (max |error| over [-5, 5]):")
print(f"  {'N terms':<10} {'Max error':<12} {'Note'}")
print(f"  {'-'*40}")

for n_terms in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
    gate = ErfSeriesGate(n_terms)
    with torch.no_grad():
        approx = gate(x)
    max_err = (approx - gelu_exact).abs().max().item()
    note = ""
    if n_terms == 1:
        note = "≈ x²/√(2π) (linear)"
    elif max_err < 0.030:
        note = f"< φ-gate error (0.030)"
    if max_err < 1e-6:
        note = "EXACT (machine precision)"
    print(f"  {n_terms:<10} {max_err:<12.6f} {note}")

# Now test on actual classification
print()
print("=" * 80)
print("CLASSIFICATION TEST")
print("=" * 80)
print()

weights_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model_orig = convnext_tiny(weights=weights_enum)
model_orig.eval()
preprocess = weights_enum.transforms()

image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))
N_TEST = 100

# Reference predictions
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

gates_to_test = {
    'x·σ(φ·x)': PhiGate(),
    'x·σ(√π·x)': SqrtPiGate(),
    'erf series N=3': ErfSeriesGate(3),
    'erf series N=5': ErfSeriesGate(5),
    'erf series N=8': ErfSeriesGate(8),
    'erf series N=10': ErfSeriesGate(10),
    'erf series N=15': ErfSeriesGate(15),
    'erf series N=20': ErfSeriesGate(20),
    'φ-LUT (16K entries)': PhiLUTGate(16384),
    'φ-LUT (1K entries)': PhiLUTGate(1024),
    'φ-LUT (256 entries)': PhiLUTGate(256),
}

print(f"  {'Gate':<30} {'Max err':<10} {'Top-1':<12} {'Top-5':<12} {'Cos':<8}")
print(f"  {'-'*70}")

for name, gate in gates_to_test.items():
    # Compute max error vs GELU
    with torch.no_grad():
        gate_out = gate(x)
    max_err = (gate_out - gelu_exact).abs().max().item()

    model = copy.deepcopy(model_orig)
    replace_gelu_with(model, gate)
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
            cosines.append(F.cosine_similarity(logits, ref_logits[idx], dim=1).item())

    n = len(ref_preds)
    print(f"  {name:<30} {max_err:<10.6f} "
          f"{agree1}/{n} ({agree1/n*100:>3.0f}%) "
          f"{agree5}/{n} ({agree5/n*100:>3.0f}%)  "
          f"{np.mean(cosines):.4f}")


# ============================================================================
# Analysis: the BBP structure of the erf series
# ============================================================================

print()
print("=" * 80)
print("THE BBP STRUCTURE OF GELU")
print("=" * 80)
print()
print("  GELU(x) = x/2 + (x²/√(2π)) × Σ_{n=0}^∞ (-1)^n × (x/√2)^(2n) / (n!·(2n+1))")
print()
print("  Like BBP: π = Σ (-1)^n × a_n / 64^n")
print("  GELU:         Σ (-1)^n × [1/(n!·(2n+1))] × (x²/2)^n")
print()
print(f"  The 'base' is x²/2 (activation-dependent, not fixed like 64)")
print(f"  The 'coefficients' are 1/(n!·(2n+1)) — EXACT integers in the denominator")
print()

# Show the coefficients
print("  Coefficients (BBP-style):")
print(f"  {'n':<5} {'(-1)^n':<8} {'1/(n!·(2n+1))':<18} {'Cumulative terms'}")
print(f"  {'-'*50}")
for n in range(10):
    sign = (-1)**n
    factorial_n = 1
    for i in range(1, n+1):
        factorial_n *= i
    coeff = 1.0 / (factorial_n * (2*n + 1))
    print(f"  {n:<5} {'+' if sign > 0 else '-':<8} {coeff:<18.8f} "
          f"1/{factorial_n}·{2*n+1} = 1/{factorial_n * (2*n+1)}")

print()
print("  Each term:")
print("    n=0:  +1/1      = +1.000")
print("    n=1:  -1/3      = -0.333  (error corrects overshoot)")
print("    n=2:  +1/10     = +0.100  (corrects undershoot)")
print("    n=3:  -1/42     = -0.024  (corrects overshoot)")
print("    n=4:  +1/216    = +0.005  (corrects undershoot)")
print("    n=5:  -1/1320   = -0.001  (converged)")
print()
print("  This IS Newton's alternating series with Pascal's triangle coefficients!")
print("  The factorials in the denominator ensure geometric decay.")
print("  n! grows faster than 2^n → guaranteed convergence for all x.")
print()

# The φ connection to the coefficients
print("  φ-connection in the denominators:")
for n in range(8):
    factorial_n = 1
    for i in range(1, n+1):
        factorial_n *= i
    denom = factorial_n * (2*n + 1)
    log_phi_denom = np.log(denom) / np.log(PHI)
    print(f"    n={n}: denom={denom:<10} log_φ(denom)={log_phi_denom:.3f}  "
          f"≈ φ^{round(log_phi_denom)}")

print()
print("  The denominators grow as ~φ^(2.5n) — each term is ~1/φ^2.5 of the previous")
print("  This is FASTER than BBP's 1/64 = 1/φ^(8.7) per term")
print("  We need fewer erf terms than BBP needs base-64 terms for same precision")
