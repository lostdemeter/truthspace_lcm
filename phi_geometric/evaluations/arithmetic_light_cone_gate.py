#!/usr/bin/env python3
"""
The Arithmetic Light Cone Applied to the Gate Problem

From rharithmeticlight:
  F(t) = raw fluctuation → GROWS
  G(t) = e^{-t/2} · F(t) → BOUNDED (light-cone normalization)
  H(t) = G(t)/t² → STABLE

The even-N instability in our erf series is a "tachyonic mode":
  Raw terms z^(2n) grow exponentially → OVERFLOW in float32
  But z^(2n)/n! is bounded → factorial "speed limit" tames growth

The fix from the paper: compute in multiplicative time (log-space)
where the light-cone constraint is automatic.

Three investigations:
1. Light-cone normalization: compute erf in log-space → fix even-N
2. Base-collapse: same results across float32/float64/φ-integer
3. Equidistribution horizon: at what depth do gate errors equidistribute?
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


# ============================================================================
# Part 1: The light-cone normalization
# ============================================================================

print("=" * 80)
print("PART 1: The Light-Cone Structure of the Erf Series")
print("=" * 80)
print()

# The erf series: erf(z) = (2/√π) × z × Σ (-1)^n × z^(2n) / (n!·(2n+1))
#
# Raw term magnitude: |z^(2n)|
# Normalized term:     |z^(2n) / (n!·(2n+1))|
# 
# The paper's analogy:
#   F(t) ↔ z^(2n)              (raw, grows)
#   G(t) ↔ z^(2n) / n!         (light-cone bounded)
#   H(t) ↔ z^(2n) / (n!·(2n+1))  (stable)

print("  Raw vs normalized term magnitudes at different z:")
print(f"  {'z':<6} {'n':<5} {'|z^(2n)|':<15} {'|z^(2n)/n!|':<15} {'|z^(2n)/(n!·(2n+1))|':<20} {'Bounded?'}")
print(f"  {'-'*70}")

for z_val in [1.0, 2.0, 3.54]:  # 3.54 = 5/√2, the worst case
    for n in [0, 5, 10, 15, 20]:
        raw = z_val ** (2*n)
        normalized = raw / math.factorial(n)
        stable = normalized / (2*n + 1)
        bounded = "YES" if stable < 1.0 else ("marginal" if stable < 10 else "NO → overflow")
        print(f"  {z_val:<6.2f} {n:<5} {raw:<15.2e} {normalized:<15.2e} {stable:<20.2e} {bounded}")
    print()

# The "light cone" is the boundary where factorial decay overtakes exponential growth
# For z = x/√2: the critical point is where z² ≈ n (Stirling's approx)
# n! ≈ (n/e)^n → z^(2n)/n! ≈ (ez²/n)^n → bounded when n > ez²

print("  LIGHT-CONE BOUNDARY: n* where factorial overtakes exponential")
print(f"  {'z=x/√2':<10} {'x':<8} {'n* = e·z²':<12} {'Meaning'}")
print(f"  {'-'*50}")
for x_val in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    z = x_val / np.sqrt(2)
    n_star = np.e * z**2
    print(f"  {z:<10.3f} {x_val:<8.1f} {n_star:<12.1f} {'← typical activations' if x_val <= 3 else ('← tail' if x_val <= 5 else '← extreme')}")

print()
print("  For typical activations |x| < 3: n* ≈ 12 → 15 terms is well within light cone")
print("  For tail |x| = 5: n* ≈ 34 → 15 terms is OUTSIDE light cone → unstable!")
print("  This explains even-N failure: at |x|=5, we're beyond the causal horizon")


# ============================================================================
# Part 2: Log-space computation (respecting the light cone)
# ============================================================================

print()
print("=" * 80)
print("PART 2: Log-Space Erf (Light-Cone Respecting)")
print("=" * 80)
print()

class LogSpaceErfGate(nn.Module):
    """Compute erf series in log-space to respect the light cone.
    
    Instead of forming z^(2n) (overflow) then dividing by n! (underflow),
    compute log(|term|) = 2n·log|z| - log(n!) - log(2n+1)
    then recover with exp and apply sign.
    
    This is the multiplicative-time reparameterization from the paper:
    t = log(x) makes multiplication → addition.
    """
    def __init__(self, n_terms):
        super().__init__()
        self.n_terms = n_terms
        # Pre-compute log(n!) and log(2n+1)
        self.log_factorials = []
        self.log_odds = []
        for n in range(n_terms):
            lf = sum(np.log(i) for i in range(1, n+1)) if n > 0 else 0.0
            self.log_factorials.append(lf)
            self.log_odds.append(np.log(2*n + 1))

    def forward(self, x):
        z = x / np.sqrt(2.0)
        z_sq = z * z

        # Compute log|z²| for the log-space terms
        log_z_sq = torch.log(z_sq.clamp(min=1e-30))

        # Sum terms using log-sum-exp for stability
        # Each term: (-1)^n × z^(2n) / (n!·(2n+1))
        # log|term_n| = n·log(z²) - log(n!) - log(2n+1)

        # We'll compute the positive and negative partial sums separately
        pos_terms = []  # n even
        neg_terms = []  # n odd

        for n in range(self.n_terms):
            log_abs_term = n * log_z_sq - self.log_factorials[n] - self.log_odds[n]

            if n % 2 == 0:
                pos_terms.append(log_abs_term)
            else:
                neg_terms.append(log_abs_term)

        # Log-sum-exp for positive terms
        if pos_terms:
            pos_stack = torch.stack(pos_terms, dim=0)
            pos_max = pos_stack.max(dim=0).values
            pos_sum = pos_max + torch.log(torch.exp(pos_stack - pos_max).sum(dim=0))
            pos_val = torch.exp(pos_sum)
        else:
            pos_val = torch.zeros_like(x)

        # Log-sum-exp for negative terms
        if neg_terms:
            neg_stack = torch.stack(neg_terms, dim=0)
            neg_max = neg_stack.max(dim=0).values
            neg_sum = neg_max + torch.log(torch.exp(neg_stack - neg_max).sum(dim=0))
            neg_val = torch.exp(neg_sum)
        else:
            neg_val = torch.zeros_like(x)

        # series_sum = positive_total - negative_total
        series_sum = pos_val - neg_val

        # Handle z=0 (where log is -inf)
        series_sum = torch.where(z_sq < 1e-20, torch.ones_like(series_sum), series_sum)

        # erf(z) ≈ (2/√π) × z × series_sum
        erf_approx = (2.0 / np.sqrt(np.pi)) * z * series_sum

        # Φ(x) = (1 + erf(x/√2)) / 2
        phi_x = 0.5 * (1.0 + erf_approx)
        phi_x = phi_x.clamp(0, 1)

        return x * phi_x


class HornerErfGate(nn.Module):
    """Horner's method erf gate (from previous experiment)."""
    def __init__(self, n_terms):
        super().__init__()
        self.n_terms = n_terms
        coeffs = []
        for n in range(n_terms):
            sign = (-1) ** n
            factorial_n = math.factorial(n)
            c = sign / (factorial_n * (2 * n + 1))
            coeffs.append(c)
        self.coeffs = coeffs

    def forward(self, x):
        z = x / np.sqrt(2.0)
        z_sq = z * z
        result = torch.full_like(z_sq, self.coeffs[-1])
        for n in range(self.n_terms - 2, -1, -1):
            result = self.coeffs[n] + z_sq * result
        erf_approx = (2.0 / np.sqrt(np.pi)) * z * result
        phi_x = 0.5 * (1.0 + erf_approx)
        phi_x = phi_x.clamp(0, 1)
        return x * phi_x


# Test: does log-space fix the even-N instability?
x = torch.linspace(-5, 5, 10000)
gelu_exact = x * 0.5 * (1 + torch.erf(x / np.sqrt(2.0)))

print("  Log-space vs Horner: max error over [-5, 5]")
print(f"  {'N':<5} {'Horner':<15} {'Log-space':<15} {'Fixed?'}")
print(f"  {'-'*45}")

for n in range(1, 26):
    gate_h = HornerErfGate(n)
    gate_l = LogSpaceErfGate(n)
    with torch.no_grad():
        out_h = gate_h(x)
        out_l = gate_l(x)
    err_h = (out_h - gelu_exact).abs().max().item()
    err_l = (out_l - gelu_exact).abs().max().item()
    fixed = ""
    if err_h > 1.0 and err_l < 0.1:
        fixed = "★ FIXED"
    elif err_l < 0.030:
        fixed = "< φ-gate"
    if err_l < 0.001:
        fixed = "★ HIGH PRECISION"
    print(f"  {n:<5} {err_h:<15.6f} {err_l:<15.6f} {fixed}")


# ============================================================================
# Part 3: Classification test with log-space gate
# ============================================================================

print()
print("=" * 80)
print("PART 3: Classification with Light-Cone Gate")
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
    print(f"  {label:<35} {agree1}/{n} ({agree1/n*100:>3.0f}%) "
          f"{agree5}/{n} ({agree5/n*100:>3.0f}%)  {cos_mean:.4f}")

print(f"  {'Gate':<35} {'Top-1':<12} {'Top-5':<12} {'Cos':<8}")
print(f"  {'-'*65}")

# Test the log-space gate at even N values that failed before
for n_terms in [8, 10, 12, 14, 15, 16, 20]:
    test_gate(f'Log-space erf N={n_terms}', LogSpaceErfGate(n_terms))

# Compare with Horner odd-N
for n_terms in [15, 17, 19, 21]:
    test_gate(f'Horner erf N={n_terms}', HornerErfGate(n_terms))


# ============================================================================
# Part 4: The equidistribution horizon
# ============================================================================

print()
print("=" * 80)
print("PART 4: The Equidistribution Horizon")
print("=" * 80)
print()

# From the paper: equidistribution occurs beyond 2·log(q)
# For our network: when does the gate error "equidistribute"
# (become random rather than systematic)?
#
# We measure: at each block, does the error grow like √N (light cone)
# or like N (superluminal)?

# Hook into every block to track cumulative error
model_phi = copy.deepcopy(model_orig)
phi_gate = type('PhiGate', (nn.Module,), {'forward': lambda self, x: x * torch.sigmoid(PHI * x)})()
for name, module in model_phi.named_modules():
    if isinstance(module, nn.GELU):
        parts = name.split('.')
        parent = model_phi
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], phi_gate)
model_phi.eval()

# Track per-stage outputs
stage_features_orig = {}
stage_features_phi = {}

def make_hook(storage, name):
    def hook(module, input, output):
        storage[name] = output.detach()
    return hook

hooks = []
for stage_idx in [0, 1, 2, 3, 4, 5, 6, 7]:
    stage_o = model_orig.features[stage_idx]
    stage_p = model_phi.features[stage_idx]
    hooks.append(stage_o.register_forward_hook(make_hook(stage_features_orig, f's{stage_idx}')))
    hooks.append(stage_p.register_forward_hook(make_hook(stage_features_phi, f's{stage_idx}')))

# Process multiple images and track error growth
n_horizon = min(20, len(ref_preds))
all_errors = {f's{i}': [] for i in range(8)}

for idx in range(n_horizon):
    img = cv2.imread(images[idx])
    if img is None:
        continue
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        _ = model_orig(tensor)
        _ = model_phi(tensor)

    for s in range(8):
        key = f's{s}'
        if key in stage_features_orig and key in stage_features_phi:
            diff = (stage_features_phi[key] - stage_features_orig[key])
            rms = diff.pow(2).mean().sqrt().item()
            all_errors[key].append(rms)

for h in hooks:
    h.remove()

# Analyze error growth pattern
print("  Per-stage RMS error (GELU vs φ-gate):")
print(f"  {'Stage':<10} {'Mean RMS':<12} {'Cumulative blocks':<20} {'√N scaling':<12} {'Ratio'}")
print(f"  {'-'*65}")

cum_blocks = 0
block_counts = [0, 3, 0, 3, 0, 9, 0, 3]  # ConvNeXt-Tiny block counts per stage
prev_rms = None

for s in range(8):
    key = f's{s}'
    cum_blocks += block_counts[s]
    if all_errors[key]:
        mean_rms = np.mean(all_errors[key])
        sqrt_n = np.sqrt(max(cum_blocks, 1))
        ratio = mean_rms / sqrt_n if sqrt_n > 0 else 0
        growth = ""
        if prev_rms and prev_rms > 0 and mean_rms > 0:
            actual_growth = mean_rms / prev_rms
            expected_sqrt = np.sqrt(cum_blocks / max(cum_blocks - block_counts[s], 1))
            if actual_growth > expected_sqrt * 1.5:
                growth = "SUPERLUMINAL"
            elif actual_growth < expected_sqrt * 0.5:
                growth = "subluminal"
            else:
                growth = "≈ light-cone"
        print(f"  s{s:<9} {mean_rms:<12.6f} {cum_blocks:<20} {sqrt_n:<12.3f} {ratio:.6f}  {growth}")
        if mean_rms > 0:
            prev_rms = mean_rms

# The horizon from the paper: 2·log(q)
# For our network: q = "modulus" of the gate error = per-block error contribution
# Horizon = depth beyond which errors equidistribute
print()
print("  Paper's horizon formula: t_horizon = 2·log(q)")
print("  Our analog: block_horizon = 2·log(per_block_error_contribution)")
print()

# The per-block error is ~0.030 for x·σ(φ·x)
# Horizon = 2·log(1/0.030) = 2·3.51 = 7.0 blocks
# After ~7 blocks, the gate error should equidistribute
# ConvNeXt has 18 blocks → we're FAR beyond the horizon
# This means the error has been "equidistributed" for the last 11 blocks
# BUT: equidistribution means RANDOM, not zero → cumulative ~√N

horizon_phi = 2 * np.log(1/0.030)
horizon_sqrt_pi = 2 * np.log(1/0.014)
horizon_erf15 = 2 * np.log(1/0.002)

print(f"  φ-gate (err=0.030):   horizon ≈ {horizon_phi:.1f} blocks")
print(f"  √π-gate (err=0.014):  horizon ≈ {horizon_sqrt_pi:.1f} blocks")
print(f"  erf N=15 (err=0.002): horizon ≈ {horizon_erf15:.1f} blocks")
print(f"  ConvNeXt depth: 18 blocks")
print()
print("  For φ-gate: 18 > 7 → beyond horizon → errors have equidistributed")
print("  For erf N=15: 18 > 12 → barely beyond → errors still partly coherent")
print("  This may explain why erf N=15 achieves 100% — it's near the horizon")
print("  where errors haven't fully randomized yet (still structured/alternating)")


# ============================================================================
# Part 5: Base-collapse — does the gate give same results regardless of
# numerical representation?
# ============================================================================

print()
print("=" * 80)
print("PART 5: Base-Collapse — Representation Invariance")
print("=" * 80)
print()

# The paper shows prime distributions collapse across bases 3,8,10,12,30,60,101
# Our analog: does the gate give same results in float16, float32, float64,
# bfloat16? If the computation is "on the critical line" (properly normalized),
# it should be base-invariant.

print("  Testing gate representation invariance:")
print(f"  {'Gate × Precision':<40} {'Max err vs GELU (f32)':<22} {'Δ vs f32 gate'}")
print(f"  {'-'*70}")

x32 = torch.linspace(-5, 5, 10000)
gelu32 = x32 * 0.5 * (1 + torch.erf(x32 / np.sqrt(2.0)))

for gate_name, gate_fn in [
    ('x·σ(φ·x)', lambda x: x * torch.sigmoid(PHI * x)),
    ('Horner erf N=15', lambda x: HornerErfGate(15)(x)),
    ('Log-space erf N=15', lambda x: LogSpaceErfGate(15)(x)),
]:
    for dtype_name, dtype in [('float16', torch.float16), ('bfloat16', torch.bfloat16),
                              ('float32', torch.float32), ('float64', torch.float64)]:
        x_d = x32.to(dtype)
        with torch.no_grad():
            out = gate_fn(x_d).float()
        # Error vs GELU in float32
        err_vs_gelu = (out - gelu32).abs().max().item()
        # Error vs same gate in float32
        with torch.no_grad():
            out_f32 = gate_fn(x32)
        err_vs_self = (out - out_f32).abs().max().item()
        
        collapse = ""
        if err_vs_self < 0.001:
            collapse = "COLLAPSED (base-invariant)"
        elif err_vs_self < 0.01:
            collapse = "near-collapsed"
        else:
            collapse = f"DIVERGED (Δ={err_vs_self:.4f})"
        
        print(f"  {gate_name + ' ' + dtype_name:<40} {err_vs_gelu:<22.6f} {collapse}")
    print()


# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("THE ARITHMETIC LIGHT CONE OF GELU")
print("=" * 80)
print()
print("  From rharithmeticlight:")
print("    F(t) grows → G(t) = e^{-t/2}·F(t) bounded → H(t) = G(t)/t² stable")
print()
print("  For the erf series:")
print("    z^(2n) grows → z^(2n)/n! bounded → z^(2n)/(n!·(2n+1)) stable")
print("    The factorial IS the light-cone speed limit")
print()
print("  The light-cone boundary: n* = e·z² = e·x²/2")
print("    Within cone (n < n*): terms grow → must sum precisely")
print("    Beyond cone (n > n*): terms decay → safe to truncate")
print("    Even-N fails when the LAST term is within the cone (pushing up)")
print("    Odd-N works when the LAST term pulls down (self-correcting)")
print()
print("  The equidistribution horizon: 2·log(1/ε)")
print("    φ-gate: horizon ≈ 7 blocks (18 > 7 → errors randomized → 88%)")
print("    erf N=15: horizon ≈ 12 blocks (18 ≈ 12 → errors structured → 100%)")
print()
print("  The key insight:")
print("    Alternating signs keep errors WITHIN the light cone.")
print("    Without alternation, errors go superluminal (same-sign accumulation).")
print("    The erf series' (-1)^n IS the light-cone constraint.")
print("    This is why N=15 (alternating) beats the LUT (non-alternating).")
