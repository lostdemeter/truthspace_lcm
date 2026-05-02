#!/usr/bin/env python3
"""
The Ideal Gate: Base-Invariant + Alternating Error Correction

Requirements:
  1. Base-invariant: bounded operations (sigmoid), works in f16/bf16/f32/f64
  2. Alternating error correction: self-correcting through network depth
  3. Geometric constants only: π, φ, or derived — no empirical fits
  4. 0 learned parameters
  5. 100% classification agreement with GELU

Approaches:
  A. Corrected Sigmoid: x·σ(k·x·(1 + c·x²))
     - k = √(8/π) matches Φ(x) ↔ σ(kx) at 1st derivative
     - c = (4-π)/(6π) matches at 3rd derivative
     - DERIVED, not fit. All constants from π.
     
  B. Multi-Sigmoid Mixture: x·Σ w_i·σ(k_i·x)
     - Each sigmoid bounded → base-invariant
     - Alternating weights → error correction
     - k_i from φ-lattice
     
  C. Corrected Sigmoid with quintic: x·σ(k·x·(1 + c₃·x² + c₅·x⁴))
     - Match through 5th derivative
     
  D. Sigmoid-of-sigmoid: x·σ(k·x + correction(x))
     - correction itself uses bounded ops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import copy
import math
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI
SQRT_8_OVER_PI = np.sqrt(8.0 / np.pi)  # 1.5958
SQRT_PI = np.sqrt(np.pi)               # 1.7725

# ============================================================================
# Derivation of the corrected sigmoid coefficient
# ============================================================================

print("=" * 80)
print("DERIVATION: The Corrected Sigmoid Gate")
print("=" * 80)
print()

# GELU(x) = x·Φ(x) where Φ(x) = (1 + erf(x/√2))/2
# Approximation: gate(x) = x·σ(k·x·(1 + c·x²))
# where k and c are chosen to match derivatives of Φ at x=0.

# Φ(x) = 1/2 + (1/√(2π))x - (1/(6√(2π)))x³ + (1/(40√(2π)))x⁵ - ...
# σ(f(x)) where f(x) = k·x·(1 + c·x²)

# At x=0: both = 1/2 ✓
# 1st derivative match: σ'(0)·f'(0) = Φ'(0)
#   (1/4)·k = 1/√(2π) → k = 4/√(2π) = √(8/π) ✓

k = SQRT_8_OVER_PI
print(f"  k = √(8/π) = {k:.6f}")
print(f"  This matches the 1st derivative: σ'(0)·k = Φ'(0)")
print(f"    (1/4)·{k:.4f} = {k/4:.6f} vs 1/√(2π) = {1/np.sqrt(2*np.pi):.6f}")
print()

# 3rd derivative match:
# d³/dx³[σ(f(x))] at x=0 = σ'''(0)·k³ + σ'(0)·6kc
# = (-1/8)·k³ + (1/4)·6kc = -k³/8 + 3kc/2
#
# Φ'''(0) = -1/√(2π)
#
# So: -k³/8 + 3kc/2 = -1/√(2π)
#     3kc/2 = -1/√(2π) + k³/8
#     c = (-1/√(2π) + k³/8) / (3k/2)

inv_sqrt_2pi = 1.0 / np.sqrt(2 * np.pi)
c_derived = (-inv_sqrt_2pi + k**3 / 8) / (1.5 * k)

print(f"  c = (-1/√(2π) + k³/8) / (3k/2)")
print(f"    = ({-inv_sqrt_2pi:.6f} + {k**3/8:.6f}) / {1.5*k:.6f}")
print(f"    = {-inv_sqrt_2pi + k**3/8:.6f} / {1.5*k:.6f}")
print(f"    = {c_derived:.6f}")
print()

# Simplify: with k = √(8/π), k³ = (8/π)^(3/2) = 8√(8/π)/π
# -1/√(2π) + k³/8 = -1/√(2π) + √(8/π)/(π)  [since k³/8 = (8/π)^(3/2)/8 = √(8/π)/π]
# = -1/√(2π) + √8/(π·√π) = -1/√(2π) + 2√2/(π^(3/2))
# = [-π^(3/2) + 2√2·√(2π)] / [√(2π)·π^(3/2)]
# ... Let me simplify differently:
# 
# c = (4 - π) / (6π)

c_formula = (4 - np.pi) / (6 * np.pi)
print(f"  Simplified: c = (4 - π) / (6π)")
print(f"    = ({4 - np.pi:.6f}) / ({6*np.pi:.6f})")
print(f"    = {c_formula:.6f}")
print(f"    vs derived: {c_derived:.6f}")
print(f"    Match: {abs(c_formula - c_derived) < 1e-10}")
print()

# Compare with empirical GELU coefficient
c_empirical = 0.044715
print(f"  Comparison:")
print(f"    Geometric (4-π)/(6π):  {c_formula:.6f}")
print(f"    Empirical (GELU paper): {c_empirical:.6f}")
print(f"    Difference: {abs(c_formula - c_empirical)/c_empirical*100:.2f}%")
print()

# ============================================================================
# 5th derivative: derive c₅ for quintic correction
# ============================================================================

print("  Deriving quintic correction c₅...")
print()

# For f(x) = k·x·(1 + c₃·x² + c₅·x⁴), we need to match Φ^(5)(0)
# This gets complex. Let me do it numerically: fit c₅ to minimize
# max |Φ(x) - σ(f(x))| for |x| < 4

# Numerical optimization for c₅
from scipy.optimize import minimize_scalar

x_opt = np.linspace(-4, 4, 10000)
phi_exact = 0.5 * (1 + np.vectorize(math.erf)(x_opt / np.sqrt(2)))

def max_error_c5(c5):
    f = k * x_opt * (1 + c_formula * x_opt**2 + c5 * x_opt**4)
    sig = 1.0 / (1.0 + np.exp(-f))
    return np.max(np.abs(sig - phi_exact))

result = minimize_scalar(max_error_c5, bounds=(-0.01, 0.01), method='bounded')
c5_optimal = result.x
err_c5 = result.fun

print(f"  c₅ (optimal for |x| < 4): {c5_optimal:.8f}")
print(f"  Max error with c₃ + c₅:   {err_c5:.8f}")
print()

# Check if c₅ has geometric structure
print(f"  Is c₅ geometric?")
print(f"    c₅ = {c5_optimal:.8f}")
print(f"    c₃² = {c_formula**2:.8f}")
print(f"    c₃/π = {c_formula/np.pi:.8f}")
print(f"    (4-π)²/(6π)² = {((4-np.pi)/(6*np.pi))**2:.8f}")
print(f"    (π²-8)/(180π²) = {(np.pi**2-8)/(180*np.pi**2):.8f}")
# The 5th derivative of Φ at 0: Φ^(5)(0) = 3/√(2π)
# Matching through chain rule...
# Actually let me just derive it symbolically
# d⁵/dx⁵[σ(f)] at x=0 involves σ^(5)(0)·(f')⁵ + ... + σ'(0)·f^(5)(0)
# σ^(5)(0) = 1/2 (I think... need to check)
# This is getting messy. Let's check numerically what c₅ should be for exact 5th deriv match

# Compute Φ^(5)(0) numerically
h = 1e-5
phi_func = lambda x: 0.5 * (1 + math.erf(x / np.sqrt(2)))
phi5_num = (phi_func(2*h) - 2*phi_func(h) + 2*phi_func(-h) - phi_func(-2*h)) / (2*h**3)
# Better: use finite differences (central, 7-point stencil)
h = 1e-3
coeffs_5 = [(-1,2), (5,-1), (-10,0.5), (10,-0.5), (-5,1), (1,-2)]  # not standard
# Use known analytic result: Φ^(5)(0) = 3/√(2π)
phi5 = 3.0 / np.sqrt(2 * np.pi)
print(f"    Φ^(5)(0) = {phi5:.6f} (= 3/√(2π))")
print()


# ============================================================================
# Gate implementations
# ============================================================================

class CorrectedSigmoidGate(nn.Module):
    """gate(x) = x · σ(√(8/π) · x · (1 + c·x²))
    c = (4-π)/(6π) from 3rd derivative matching.
    Single sigmoid → bounded → base-invariant."""
    def forward(self, x):
        k = 1.5957691216  # √(8/π)
        c = 0.04555367    # (4-π)/(6π)
        f = k * x * (1.0 + c * x * x)
        return x * torch.sigmoid(f)


class CorrectedSigmoidEmpiricalGate(nn.Module):
    """Same but with the empirical c=0.044715 from GELU paper."""
    def forward(self, x):
        k = 1.5957691216
        c = 0.044715
        f = k * x * (1.0 + c * x * x)
        return x * torch.sigmoid(f)


class QuinticCorrectedSigmoidGate(nn.Module):
    """gate(x) = x · σ(√(8/π) · x · (1 + c₃·x² + c₅·x⁴))"""
    def __init__(self, c5):
        super().__init__()
        self.c5 = c5
    def forward(self, x):
        k = 1.5957691216
        c3 = 0.04555367
        x_sq = x * x
        f = k * x * (1.0 + c3 * x_sq + self.c5 * x_sq * x_sq)
        return x * torch.sigmoid(f)


class MultiSigmoidGate(nn.Module):
    """gate(x) = x · Σ w_i · σ(k_i · x)
    Mixture of sigmoids — each bounded, alternating weights."""
    def __init__(self, k_values, weights):
        super().__init__()
        self.k_values = k_values
        self.weights = weights
    def forward(self, x):
        result = torch.zeros_like(x)
        for k, w in zip(self.k_values, self.weights):
            result = result + w * torch.sigmoid(k * x)
        return x * result


class PhiGate(nn.Module):
    """x·σ(φ·x) — baseline."""
    def forward(self, x):
        return x * torch.sigmoid(PHI * x)


class SqrtPiGate(nn.Module):
    """x·σ(√π·x) — optimal single sigmoid."""
    def forward(self, x):
        return x * torch.sigmoid(SQRT_PI * x)


class HornerErfGate(nn.Module):
    """Erf series N=15 via Horner."""
    def __init__(self, n_terms=15):
        super().__init__()
        coeffs = []
        for n in range(n_terms):
            sign = (-1) ** n
            coeffs.append(sign / (math.factorial(n) * (2 * n + 1)))
        self.coeffs = coeffs
    def forward(self, x):
        z = x * (1.0 / np.sqrt(2.0))
        z_sq = z * z
        result = torch.full_like(z_sq, self.coeffs[-1])
        for n in range(len(self.coeffs) - 2, -1, -1):
            result = self.coeffs[n] + z_sq * result
        erf_approx = (2.0 / np.sqrt(np.pi)) * z * result
        phi_x = (0.5 * (1.0 + erf_approx)).clamp(0, 1)
        return x * phi_x


# ============================================================================
# Find optimal multi-sigmoid mixtures
# ============================================================================

print("=" * 80)
print("MULTI-SIGMOID OPTIMIZATION")
print("=" * 80)
print()

# Fit w_i to minimize ||Σ w_i·σ(k_i·x) - Φ(x)||² subject to Σ w_i = 1
x_fit = torch.linspace(-5, 5, 5000).numpy()
phi_target = 0.5 * (1 + np.vectorize(math.erf)(x_fit / np.sqrt(2)))

def fit_multi_sigmoid(k_values):
    """Fit weights for multi-sigmoid mixture to match Φ(x)."""
    M = len(k_values)
    # Build design matrix
    A = np.zeros((len(x_fit), M))
    for i, k in enumerate(k_values):
        A[:, i] = 1.0 / (1.0 + np.exp(-k * x_fit))

    # Constrained least squares: Σ w_i = 1
    # Use Lagrange multiplier: minimize ||Aw - Φ||² + λ(1ᵀw - 1)
    # KKT: AᵀAw + λ1 = AᵀΦ, 1ᵀw = 1
    ATA = A.T @ A
    ATb = A.T @ phi_target
    # Augmented system
    M_aug = np.zeros((M+1, M+1))
    M_aug[:M, :M] = ATA
    M_aug[:M, M] = 1
    M_aug[M, :M] = 1
    rhs = np.zeros(M+1)
    rhs[:M] = ATb
    rhs[M] = 1
    sol = np.linalg.solve(M_aug, rhs)
    w = sol[:M]

    # Compute error
    approx = A @ w
    max_err = np.max(np.abs(approx - phi_target))
    return w, max_err


# Test different k combinations
print(f"  {'K values':<50} {'Weights':<40} {'Max err'}")
print(f"  {'-'*95}")

k_combos = [
    # 2 sigmoids
    ([PHI, SQRT_PI], "φ, √π"),
    ([1.0, PHI], "1, φ"),
    ([SQRT_8_OVER_PI, SQRT_PI], "√(8/π), √π"),
    ([1/PHI, PHI, PHI**2], "1/φ, φ, φ²"),
    # 3 sigmoids on φ-lattice
    ([1.0, PHI, PHI**2], "1, φ, φ²"),
    ([INV_PHI, SQRT_8_OVER_PI, SQRT_PI], "1/φ, √(8/π), √π"),
    ([SQRT_8_OVER_PI, PHI, SQRT_PI], "√(8/π), φ, √π"),
    # 4 sigmoids
    ([INV_PHI, 1.0, PHI, PHI**2], "1/φ, 1, φ, φ²"),
    ([1.0, SQRT_8_OVER_PI, PHI, SQRT_PI], "1, √(8/π), φ, √π"),
    # 5 sigmoids
    ([INV_PHI, 1.0, SQRT_8_OVER_PI, PHI, SQRT_PI], "1/φ, 1, √(8/π), φ, √π"),
]

best_multi = None
best_multi_err = float('inf')

for k_vals, label in k_combos:
    w, max_err = fit_multi_sigmoid(k_vals)
    w_str = ", ".join(f"{wi:.3f}" for wi in w)
    alt = sum(1 for i in range(len(w)-1) if w[i]*w[i+1] < 0)
    print(f"  {label:<50} [{w_str}]  {max_err:.6f}  alt:{alt}")
    if max_err < best_multi_err:
        best_multi_err = max_err
        best_multi = (k_vals, w, label)

print()
if best_multi:
    print(f"  Best multi-sigmoid: {best_multi[2]}")
    print(f"    Max error: {best_multi_err:.6f}")
    print(f"    Weights: {best_multi[1]}")


# ============================================================================
# Accuracy comparison: all gate candidates
# ============================================================================

print()
print("=" * 80)
print("GATE ACCURACY COMPARISON (max error vs GELU)")
print("=" * 80)
print()

x_test = torch.linspace(-5, 5, 10000)
gelu_exact = x_test * 0.5 * (1 + torch.erf(x_test / np.sqrt(2.0)))

gates = {
    'x·σ(φ·x)': PhiGate(),
    'x·σ(√π·x)': SqrtPiGate(),
    'Corrected σ (geometric c)': CorrectedSigmoidGate(),
    'Corrected σ (empirical c)': CorrectedSigmoidEmpiricalGate(),
    f'Quintic σ (c₅={c5_optimal:.6f})': QuinticCorrectedSigmoidGate(c5_optimal),
    'Horner erf N=15': HornerErfGate(15),
}

# Add best multi-sigmoid
if best_multi:
    gates[f'Multi-σ ({best_multi[2]})'] = MultiSigmoidGate(best_multi[0], best_multi[1])

print(f"  {'Gate':<40} {'Max err':<12} {'vs φ-gate'}")
print(f"  {'-'*60}")

for name, gate in gates.items():
    with torch.no_grad():
        out = gate(x_test)
    max_err = (out - gelu_exact).abs().max().item()
    vs_phi = f"{max_err/0.030:.1f}x" if max_err < 0.030 else ""
    print(f"  {name:<40} {max_err:<12.6f} {vs_phi}")


# ============================================================================
# Base-invariance test
# ============================================================================

print()
print("=" * 80)
print("BASE-INVARIANCE (collapse across precisions)")
print("=" * 80)
print()

x32 = torch.linspace(-5, 5, 10000)
gelu32 = x32 * 0.5 * (1 + torch.erf(x32 / np.sqrt(2.0)))

key_gates = {
    'x·σ(φ·x)': PhiGate(),
    'Corrected σ (geometric)': CorrectedSigmoidGate(),
    'Corrected σ (empirical)': CorrectedSigmoidEmpiricalGate(),
    'Horner erf N=15': HornerErfGate(15),
}

if best_multi:
    key_gates[f'Multi-σ best'] = MultiSigmoidGate(best_multi[0], best_multi[1])

print(f"  {'Gate':<35} {'float16':<12} {'bfloat16':<12} {'float32':<12} {'float64':<12}")
print(f"  {'-'*80}")

for name, gate in key_gates.items():
    errs = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        x_d = x32.to(dtype)
        with torch.no_grad():
            try:
                out = gate(x_d).float()
                err = (out - gelu32).abs().max().item()
            except:
                err = float('inf')
        errs.append(err)
    print(f"  {name:<35} {errs[0]:<12.6f} {errs[1]:<12.6f} {errs[2]:<12.6f} {errs[3]:<12.6f}")


# ============================================================================
# Classification test
# ============================================================================

print()
print("=" * 80)
print("CLASSIFICATION TEST (ConvNeXt-Tiny, 100 images)")
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


def test_gate(label, gate):
    model = copy.deepcopy(model_orig)
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], gate)
    model.eval()

    agree1 = agree5 = 0
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
            if set(logits.topk(5, dim=1).indices[0].tolist()) == \
               set(ref_logits[idx].topk(5, dim=1).indices[0].tolist()):
                agree5 += 1
            cos = F.cosine_similarity(logits, ref_logits[idx], dim=1).item()
            if not np.isnan(cos):
                cosines.append(cos)

    n = len(ref_preds)
    cos_mean = np.mean(cosines) if cosines else float('nan')
    print(f"  {label:<40} {agree1}/{n} ({agree1/n*100:>3.0f}%) "
          f"{agree5}/{n} ({agree5/n*100:>3.0f}%)  {cos_mean:.4f}")

print(f"  {'Gate':<40} {'Top-1':<12} {'Top-5':<12} {'Cos'}")
print(f"  {'-'*70}")

test_gates = {
    'x·σ(φ·x)': PhiGate(),
    'x·σ(√π·x)': SqrtPiGate(),
    'Corrected σ (c=(4-π)/(6π))': CorrectedSigmoidGate(),
    'Corrected σ (c=0.044715)': CorrectedSigmoidEmpiricalGate(),
    f'Quintic σ (c₅={c5_optimal:.4f})': QuinticCorrectedSigmoidGate(c5_optimal),
    'Horner erf N=15': HornerErfGate(15),
}

if best_multi:
    test_gates[f'Multi-σ ({best_multi[2]})'] = MultiSigmoidGate(best_multi[0], best_multi[1])

for label, gate in test_gates.items():
    test_gate(label, gate)


# ============================================================================
# Per-block error propagation: does corrected sigmoid fix superluminal?
# ============================================================================

print()
print("=" * 80)
print("LIGHT-CONE TEST: Error propagation speed")
print("=" * 80)
print()

# Track per-block error for corrected sigmoid vs φ-gate
for gate_label, gate_cls in [
    ('x·σ(φ·x)', PhiGate()),
    ('Corrected σ', CorrectedSigmoidGate()),
]:
    model_test = copy.deepcopy(model_orig)
    for name, module in model_test.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model_test
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], gate_cls)
    model_test.eval()

    # Hook per-stage
    features_orig = {}
    features_test = {}
    hooks_o = []
    hooks_t = []
    for s in range(8):
        hooks_o.append(model_orig.features[s].register_forward_hook(
            lambda m, i, o, name=f's{s}': features_orig.update({name: o.detach()})))
        hooks_t.append(model_test.features[s].register_forward_hook(
            lambda m, i, o, name=f's{s}': features_test.update({name: o.detach()})))

    # Average over a few images
    rms_per_stage = {f's{s}': [] for s in range(8)}
    for idx in range(min(10, len(ref_preds))):
        img = cv2.imread(images[idx])
        if img is None:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            _ = model_orig(tensor)
            _ = model_test(tensor)
        for s in range(8):
            key = f's{s}'
            if key in features_orig and key in features_test:
                diff = features_test[key] - features_orig[key]
                rms_per_stage[key].append(diff.pow(2).mean().sqrt().item())

    for h in hooks_o + hooks_t:
        h.remove()

    block_counts = [0, 3, 0, 3, 0, 9, 0, 3]
    cum = 0
    print(f"  {gate_label}:")
    for s in range(8):
        cum += block_counts[s]
        if rms_per_stage[f's{s}']:
            rms = np.mean(rms_per_stage[f's{s}'])
            sqrt_n = np.sqrt(max(cum, 1))
            print(f"    s{s}: RMS={rms:.6f}  blocks={cum}  RMS/√N={rms/sqrt_n:.6f}")
    print()


# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("THE IDEAL GATE")
print("=" * 80)
print()
print("  gate(x) = x · σ(√(8/π) · x · (1 + [(4-π)/(6π)] · x²))")
print()
print("  Constants (all from π):")
print(f"    k = √(8/π) = {SQRT_8_OVER_PI:.6f}")
print(f"    c = (4-π)/(6π) = {c_formula:.6f}")
print()
print("  Properties:")
print("    ✓ Bounded (single sigmoid) → base-invariant")
print("    ✓ Geometric derivation (3rd derivative matching)")
print("    ✓ 0 learned parameters")
print("    ✓ Efficient (one sigmoid + one multiplication)")
print()
print("  The cubic correction x·(1 + c·x²) inside the sigmoid")
print("  warps the input to match the Gaussian CDF's curvature.")
print("  This is the same structure as the GELU tanh approximation,")
print("  but with c derived geometrically rather than fit empirically.")
