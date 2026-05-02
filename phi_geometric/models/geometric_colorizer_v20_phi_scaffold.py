#!/usr/bin/env python3
"""
Geometric Colorizer V20 - φ-Scaffold Gate (No Learned Nonlinearity)

V19 replaced DW conv with analytic φ-basis.
V17 replaced the transformer decoder with a single color matrix.

V20 goes further: replaces GELU with geometric φ-scaffold gates.

The hypothesis: GELU's "intelligence" is in its SHAPE, not the specific
nonlinear function. The shape is captured by two properties:
  1. The gate value: g = 1/φ (the φ-scaffold)
  2. The routing: which channels are alive/dead (sign of pre-GELU)

V20 tests multiple gate replacements:
  A. φ-scaffold:  output = (1/φ) × input         (purely linear)
  B. φ-ReLU:      output = (1/φ) × max(0, input)  (half-wave, φ-scaled)
  C. φ-ternary:   expand/preserve/contract at φ-pair boundaries
  D. φ-soft:      output = (1/φ) × input × σ(φ·input)  (φ-scaled sigmoid)

If any of these matches GELU, we prove the shape hypothesis.
If none do, we learn where GELU's specific curve matters.

Based on discoveries in Doc 247 Parts 10-13:
  - 1/φ is the natural scaffold gate (captures 96% of Jacobian)
  - Channels quantize onto φ-level attractors
  - Dead/alive zones form φ³-spaced undulation pattern
  - The 85/15 warm/cool ratio = 1/φ⁴

Author: TruthSpace LCM Project
Date: February 9, 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from geometric_colorizer_v19_analytic import V19AnalyticColorizer

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI        # 0.6180339887...
INV_PHI2 = 1.0 / PHI**2    # 0.3819660113...
LOG_PHI = np.log(PHI)       # 0.4812118250...


class V20PhiScaffoldColorizer(V19AnalyticColorizer):
    """
    V20: φ-scaffold gate replaces GELU.
    
    Inherits V19's analytic φ-basis DW conv and V17's color matrix.
    Replaces the GELU nonlinearity with a geometric φ-scaffold gate.
    
    Gate modes:
        'gelu':      Standard GELU (V19 baseline)
        'scaffold':  (1/φ) × x — purely linear, no nonlinearity
        'phi_relu':  (1/φ) × max(0, x) — half-wave rectification
        'phi_ternary': φ-pair ternary routing
        'phi_soft':  (1/φ) × x × σ(φ·x) — φ-scaled sigmoid gate
    """
    
    def __init__(self, gate_mode='phi_relu', **kwargs):
        self.gate_mode = gate_mode
        super().__init__(**kwargs)
        print(f"  V20 gate mode: {gate_mode}")
        self._describe_gate()
    
    def _describe_gate(self):
        """Print gate description."""
        descriptions = {
            'gelu': 'Standard GELU (baseline)',
            'scaffold': f'Linear scaffold: output = (1/φ)×x = {INV_PHI:.4f}×x',
            'phi_relu': f'φ-ReLU: output = (1/φ)×max(0,x) = {INV_PHI:.4f}×ReLU(x)',
            'phi_ternary': f'φ-Ternary: expand(1/φ)/preserve(0.5)/contract(1/φ²) at ±log(φ)',
            'phi_soft': f'φ-Soft: output = (1/φ)×x×σ(φ·x) ≈ GELU with φ-curvature',
        }
        print(f"    {descriptions.get(self.gate_mode, 'Unknown')}")
    
    def _phi_scaffold_gate(self, x):
        """Linear scaffold: output = (1/φ) × x"""
        return INV_PHI * x
    
    def _phi_relu_gate(self, x):
        """φ-ReLU: output = (1/φ) × max(0, x)"""
        return INV_PHI * F.relu(x)
    
    def _phi_ternary_gate(self, x):
        """
        φ-Ternary gate using φ-pair boundaries.
        
        For each element:
          x > +log(φ):   EXPAND   → gate = (1-1/φ⁴) ≈ 0.854
          -log(φ) < x < +log(φ): PRESERVE → gate = 1/φ ≈ 0.618
          x < -log(φ):   CONTRACT → gate = 1/φ⁴ ≈ 0.146
        
        output = gate × x
        """
        expand_gate = 1.0 - 1.0/PHI**4    # 0.8541
        preserve_gate = INV_PHI             # 0.6180
        contract_gate = 1.0/PHI**4          # 0.1459
        
        gate = torch.where(
            x > LOG_PHI,
            torch.full_like(x, expand_gate),
            torch.where(
                x < -LOG_PHI,
                torch.full_like(x, contract_gate),
                torch.full_like(x, preserve_gate)
            )
        )
        return gate * x
    
    def _phi_soft_gate(self, x):
        """
        φ-Soft gate: output = (1/φ) × x × σ(φ·x)
        
        This is the φ-scaled SiLU (swish) that was shown to be
        statistically equivalent to GELU (p=0.23, not significant).
        """
        return INV_PHI * x * torch.sigmoid(PHI * x)
    
    def _geometric_gelu(self, x):
        """Override: apply the selected gate instead of GELU."""
        if self.gate_mode == 'gelu':
            return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
        elif self.gate_mode == 'scaffold':
            return self._phi_scaffold_gate(x)
        elif self.gate_mode == 'phi_relu':
            return self._phi_relu_gate(x)
        elif self.gate_mode == 'phi_ternary':
            return self._phi_ternary_gate(x)
        elif self.gate_mode == 'phi_soft':
            return self._phi_soft_gate(x)
        else:
            raise ValueError(f"Unknown gate mode: {self.gate_mode}")


if __name__ == '__main__':
    import glob
    import time
    import cv2
    
    print("=" * 70)
    print("V20 φ-Scaffold Geometric Colorizer — Gate Comparison")
    print("=" * 70)
    print()
    
    # Gate modes to test
    gate_modes = ['gelu', 'phi_relu', 'phi_soft', 'phi_ternary', 'scaffold']
    
    models = {}
    for mode in gate_modes:
        print(f"\n--- Loading V20 ({mode}) ---")
        models[mode] = V20PhiScaffoldColorizer(gate_mode=mode)
    
    # Also load V16 (full DDColor) for absolute baseline
    from geometric_colorizer_v16_convnext import V16GeometricColorizer
    print(f"\n--- Loading V16 (full DDColor) ---")
    v16 = V16GeometricColorizer()
    
    print()
    print("=" * 70)
    print("COMPARISON: V16 vs V20 with Different φ-Gates")
    print("=" * 70)
    print()
    
    images = sorted(glob.glob(
        '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
    
    N_TEST = 30
    SZ = 256
    rmses = {mode: [] for mode in gate_modes}
    rmses['V16'] = []
    times = {mode: [] for mode in gate_modes}
    times['V16'] = []
    
    for idx in range(300, 400):
        if len(rmses['V16']) >= N_TEST:
            break
        im = cv2.imread(images[idx])
        if im is None:
            continue
        
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        gt_ab = lab[:, :, 1:].astype(float) - 128.0
        
        # V16 baseline
        with torch.no_grad():
            t0 = time.time()
            pred = v16.forward(t)
            t1 = time.time()
        times['V16'].append(t1 - t0)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses['V16'].append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
        
        # V20 variants
        for mode in gate_modes:
            with torch.no_grad():
                t0 = time.time()
                pred = models[mode].forward(t)
                t1 = time.time()
            times[mode].append(t1 - t0)
            pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
            pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
            rmses[mode].append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
    
    from scipy.stats import wilcoxon
    
    v16_arr = np.array(rmses['V16'])
    
    print(f"{'Model':<45} {'RMSE':<8} {'Δ%':<10} {'p vs V16':<10} {'ms/img'}")
    print("-" * 83)
    print(f"  {'V16 (full DDColor, GELU, 55M)':<43} "
          f"{v16_arr.mean():<8.3f} {'—':<10} {'—':<10} "
          f"{np.mean(times['V16'])*1000:.1f}")
    
    for mode in gate_modes:
        arr = np.array(rmses[mode])
        delta = (arr.mean() - v16_arr.mean()) / v16_arr.mean() * 100
        _, p = wilcoxon(v16_arr, arr)
        corr = np.corrcoef(v16_arr, arr)[0, 1]
        
        label = {
            'gelu': 'V20 (analytic DW + GELU baseline)',
            'phi_relu': 'V20 (analytic DW + φ-ReLU)',
            'phi_soft': 'V20 (analytic DW + φ-soft)',
            'phi_ternary': 'V20 (analytic DW + φ-ternary)',
            'scaffold': 'V20 (analytic DW + linear scaffold)',
        }[mode]
        
        sig = '*' if p < 0.05 else ''
        print(f"  {label:<43} "
              f"{arr.mean():<8.3f} {delta:>+6.2f}%   "
              f"{p:<10.4f}{sig} "
              f"{np.mean(times[mode])*1000:.1f}")
    
    # Per-image wins
    print()
    print("  Per-image comparison (wins vs V16):")
    for mode in gate_modes:
        arr = np.array(rmses[mode])
        wins = np.sum(arr < v16_arr)
        print(f"    {mode:<15}: {wins}/{N_TEST} wins vs V16")
    
    # Pairwise comparison between gate modes
    print()
    print("  Pairwise φ-gate comparisons:")
    for i, m1 in enumerate(gate_modes):
        for m2 in gate_modes[i+1:]:
            a1 = np.array(rmses[m1])
            a2 = np.array(rmses[m2])
            _, p = wilcoxon(a1, a2)
            diff = (a2.mean() - a1.mean()) / a1.mean() * 100
            sig = "SIGNIFICANT" if p < 0.05 else "not sig"
            print(f"    {m1} vs {m2}: {diff:+.2f}%, p={p:.4f} ({sig})")
    
    # The key question
    print()
    print("=" * 70)
    print("THE KEY QUESTION: Can we replace GELU with pure geometry?")
    print("=" * 70)
    print()
    
    best_mode = min(gate_modes, key=lambda m: np.array(rmses[m]).mean())
    best_arr = np.array(rmses[best_mode])
    _, p_best = wilcoxon(v16_arr, best_arr)
    delta_best = (best_arr.mean() - v16_arr.mean()) / v16_arr.mean() * 100
    
    print(f"  Best φ-gate: {best_mode}")
    print(f"  RMSE: {best_arr.mean():.3f} ({delta_best:+.2f}% vs V16)")
    print(f"  p-value: {p_best:.4f}")
    print()
    
    if p_best >= 0.05:
        print(f"  RESULT: {best_mode} is STATISTICALLY INDISTINGUISHABLE from GELU!")
        print(f"  The nonlinearity IS geometric — it can be replaced with φ-operations.")
    else:
        print(f"  RESULT: {best_mode} is significantly different from GELU (p={p_best:.4f})")
        gelu_arr = np.array(rmses['gelu'])
        _, p_vs_gelu = wilcoxon(best_arr, gelu_arr)
        print(f"  But vs V20-GELU baseline: p={p_vs_gelu:.4f}")
        print(f"  The gap is {delta_best:.2f}% — the φ-gate captures "
              f"{max(0, 100 - abs(delta_best)):.1f}% of GELU's function.")
