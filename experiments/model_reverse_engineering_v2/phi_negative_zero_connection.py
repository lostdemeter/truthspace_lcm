#!/usr/bin/env python3
"""
Negative Zero Connection — DC 253 meets the rank-1 discovery.

DC 253: Gate activations have 4 states (+1, +0, -0, -1). Sign > magnitude.
Today:  Weight matrix = sign × φ^(rank1 + residual). Sign carries 75%.

Hypothesis: The weight matrix's 4 quadrants (sign±, residual high/low)
ARE the holographic fringe structure. The sign is the fringe (bright/dark),
the magnitude envelope is rank-1, and the residual encodes the fine
fringe spacing — the "negative zero" information.

Tests:
  1. Map weights into 4 states: (sign, |residual| vs threshold)
  2. Energy contribution of each state to the matmul
  3. Does the "negative zero" quadrant (sign=-, |res|≈0) carry
     disproportionate information like DC 253 predicted?
  4. For gate_proj specifically: do the 4 weight states predict
     the 4 gate states when x is presented?
"""

import os, sys, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
LOG_PHI = np.log(PHI)


def levels(W):
    return W.exponents.astype(np.int32) // PHI_GRID

def extract_rank1(W):
    lvl = levels(W).astype(np.float32)
    U, sigma, Vt = np.linalg.svd(lvl, full_matrices=False)
    u = U[:, 0] * sigma[0]
    v = Vt[0, :]
    return u, v, sigma, lvl


# ============================================================================
# PART 1: The 4 Quadrants of the Weight Matrix
# ============================================================================

def weight_4_states(W, name=""):
    """
    Map each weight element to one of 4 states based on:
      - Sign: +1 or -1
      - Residual magnitude: |level - rank1| > threshold or not
    
    This mirrors DC 253's 4 gate states:
      +1 EXPAND  ↔ sign=+, |res| high (strong positive weight)
      +0 PRESERVE+ ↔ sign=+, |res| low (weak positive — "bright fringe, low intensity")
      -0 PRESERVE- ↔ sign=-, |res| low (weak negative — "NEGATIVE ZERO")
      -1 CONTRACT ↔ sign=-, |res| high (strong negative weight)
    """
    u, v, sigma, lvl = extract_rank1(W)
    out_f, in_f = lvl.shape
    residual = lvl - np.outer(u, v)
    sgn = W.signs  # int8
    
    # Threshold: median |residual|
    abs_res = np.abs(residual)
    threshold = np.median(abs_res)
    
    print(f"\n{'='*70}")
    print(f"  WEIGHT 4-STATE CLASSIFICATION ({name})")
    print(f"{'='*70}")
    print(f"  Residual threshold (median): {threshold:.3f}")
    
    # Classify
    high_mag = abs_res > threshold
    pos_sign = sgn > 0
    
    expand     = pos_sign & high_mag    # +1
    preserve_p = pos_sign & ~high_mag   # +0
    preserve_n = ~pos_sign & ~high_mag  # -0 (NEGATIVE ZERO)
    contract   = ~pos_sign & high_mag   # -1
    
    n_total = out_f * in_f
    print(f"\n  State distribution:")
    print(f"    +1 EXPAND    (sign=+, |res| high): {np.sum(expand)/n_total:.1%}")
    print(f"    +0 PRESERVE+ (sign=+, |res| low):  {np.sum(preserve_p)/n_total:.1%}")
    print(f"    -0 PRESERVE- (sign=-, |res| low):  {np.sum(preserve_n)/n_total:.1%}")
    print(f"    -1 CONTRACT  (sign=-, |res| high): {np.sum(contract)/n_total:.1%}")
    
    return {'+1': expand, '+0': preserve_p, '-0': preserve_n, '-1': contract}, residual


# ============================================================================
# PART 2: Energy Contribution per State
# ============================================================================

def energy_per_state(W, states, name=""):
    """
    How much does each state contribute to the matmul output?
    Like DC 253's finding that CONTRACT carries 3.6-42.4% of energy.
    """
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    
    n_samples = 50
    X = np.random.randn(n_samples, in_f).astype(np.float32) * 0.02
    Y_full = X @ W_dec.T  # (n_samples, out_f)
    full_energy = np.mean(Y_full ** 2)
    
    print(f"\n  Energy contribution per state ({name}):")
    print(f"    Full output energy: {full_energy:.8f}")
    
    for state_name, mask in states.items():
        # Zero out all weights NOT in this state
        W_state = np.where(mask, W_dec, 0)
        Y_state = X @ W_state.T
        state_energy = np.mean(Y_state ** 2)
        
        # Correlation with full output
        corr = np.corrcoef(Y_full.flatten(), Y_state.flatten())[0, 1]
        
        # Cross-energy (signed contribution)
        cross = np.mean(Y_full * Y_state)
        
        frac = state_energy / full_energy
        cross_frac = cross / full_energy
        
        print(f"    {state_name:>2s}: energy={frac:>6.1%}, "
              f"cross={cross_frac:>7.1%}, corr={corr:.4f}")
    
    # Now test: sign-only vs magnitude-only (like DC 253's ablation)
    sgn = W.signs.astype(np.float32)
    Y_sign = X @ sgn.T
    corr_sign = np.corrcoef(Y_full.flatten(), Y_sign.flatten())[0, 1]
    
    # Magnitude only (absolute value)
    W_abs = np.abs(W_dec)
    Y_abs = X @ W_abs.T
    corr_abs = np.corrcoef(Y_full.flatten(), Y_abs.flatten())[0, 1]
    
    print(f"\n    Sign-only matmul corr:      {corr_sign:.4f}")
    print(f"    |Magnitude|-only matmul corr: {corr_abs:.4f}")
    print(f"    Sign advantage: {corr_sign/max(abs(corr_abs), 1e-10):.1f}×")


# ============================================================================
# PART 3: Negative Zero — the weight's "dark fringe at low intensity"
# ============================================================================

def negative_zero_test(W, states, name=""):
    """
    DC 253: negative zero (sign=-, level≈0) carries essential info.
    Here: weights near the rank-1 surface but with negative sign.
    
    Test: remove ONLY the negative-zero weights. Does accuracy drop
    more than removing the same NUMBER of expand weights?
    """
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    
    n_samples = 100
    X = np.random.randn(n_samples, in_f).astype(np.float32) * 0.02
    Y_full = X @ W_dec.T
    
    print(f"\n  NEGATIVE ZERO ABLATION ({name}):")
    
    for state_name in ['+1', '+0', '-0', '-1']:
        mask = states[state_name]
        # Remove this state
        W_ablated = np.where(mask, 0, W_dec)
        Y_ablated = X @ W_ablated.T
        corr = np.corrcoef(Y_full.flatten(), Y_ablated.flatten())[0, 1]
        
        n_removed = np.sum(mask)
        frac_removed = n_removed / (out_f * in_f)
        
        print(f"    Remove {state_name:>2s} ({frac_removed:.1%} of weights): "
              f"remaining corr={corr:.6f}")
    
    # Pairs
    for pair_name, keys in [("sign=+ only", ['+1', '+0']),
                             ("sign=- only", ['-1', '-0']),
                             ("|res| high only", ['+1', '-1']),
                             ("|res| low only", ['+0', '-0'])]:
        combined_mask = states[keys[0]] | states[keys[1]]
        W_keep = np.where(combined_mask, W_dec, 0)
        Y_keep = X @ W_keep.T
        corr = np.corrcoef(Y_full.flatten(), Y_keep.flatten())[0, 1]
        print(f"    Keep {pair_name:>20s}: corr={corr:.6f}")


# ============================================================================
# PART 4: Do weight states predict gate states?
# ============================================================================

def weight_predicts_gate(name=""):
    """
    For gate_proj: the weight's 4 states determine WHERE the gate
    fires/leaks. Test if the weight sign predicts the gate sign.
    """
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    
    W_gate = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
    W_up = PhiEncoded.load(os.path.join(layer_dir, 'up_proj.npz'))
    
    W_gate_dec = W_gate.decode_cached()
    W_up_dec = W_up.decode_cached()
    
    out_f, in_f = W_gate.shape
    
    n_samples = 50
    X = np.random.randn(n_samples, in_f).astype(np.float32) * 0.02
    
    # Gate activation: SiLU(gate_proj @ x) = gate_proj @ x * sigmoid(gate_proj @ x)
    gate_pre = X @ W_gate_dec.T  # (n_samples, 18944)
    gate_sign = np.sign(gate_pre)  # sign of gate activation
    
    # Weight sign per output channel (majority sign across inputs, weighted by |x|)
    # Actually, each gate output g[j] = Σ_i W[j,i] × x[i]
    # The SIGN of g[j] depends on the sign pattern of row j
    
    # Classify gate activations into 4 states (DC 253 thresholds at ±log(φ))
    expand     = gate_pre > LOG_PHI
    preserve_p = (gate_pre > 0) & (gate_pre <= LOG_PHI)
    preserve_n = (gate_pre <= 0) & (gate_pre >= -LOG_PHI)
    contract   = gate_pre < -LOG_PHI
    
    print(f"\n{'='*70}")
    print(f"  WEIGHT → GATE STATE PREDICTION")
    print(f"{'='*70}")
    
    print(f"\n  Gate state distribution (across {n_samples} samples):")
    print(f"    +1 EXPAND:    {np.mean(expand):.1%}")
    print(f"    +0 PRESERVE+: {np.mean(preserve_p):.1%}")
    print(f"    -0 PRESERVE-: {np.mean(preserve_n):.1%}")
    print(f"    -1 CONTRACT:  {np.mean(contract):.1%}")
    
    # Weight sign predicts gate sign
    w_sgn = W_gate.signs  # (out_f, in_f)
    
    # For each gate output row, what's the "bias" — the mean sign?
    row_sign_bias = np.mean(w_sgn.astype(np.float32), axis=1)  # (out_f,)
    
    # Does the weight row's sign bias predict the gate output sign?
    # gate_sign is (n_samples, out_f), row_sign_bias is (out_f,)
    # For each channel, does sign(bias) predict sign(gate)?
    bias_sign = np.sign(row_sign_bias)
    
    agreement = np.mean(gate_sign == bias_sign[None, :], axis=0)  # per channel
    print(f"\n  Weight sign bias predicts gate sign:")
    print(f"    Mean agreement: {np.mean(agreement):.1%}")
    print(f"    (50% = random)")
    
    # More direct: extract rank-1 of gate weights
    u_gate, v_gate, sigma_gate, lvl_gate = extract_rank1(W_gate)
    
    # The rank-1 predicts: gate ≈ sign_row ⊙ φ^(u⊗v) @ x
    # The sign of gate output depends on sign pattern + x
    # But the MAGNITUDE of gate output depends on |u[j]| × |v| ⊙ |x|
    
    # Channel-level analysis: which channels are always CONTRACT?
    always_contract = np.all(contract, axis=0)  # channel is CONTRACT for all samples
    mostly_contract = np.mean(contract, axis=0) > 0.9
    
    print(f"\n  Channel stability:")
    print(f"    Always CONTRACT: {np.mean(always_contract):.1%}")
    print(f"    >90% CONTRACT: {np.mean(mostly_contract):.1%}")
    
    # For CONTRACT channels, what does their weight look like?
    contract_channels = np.where(mostly_contract)[0]
    active_channels = np.where(~mostly_contract)[0]
    
    if len(contract_channels) > 0 and len(active_channels) > 0:
        # Weight row norms
        w_norms_contract = np.linalg.norm(W_gate_dec[contract_channels], axis=1)
        w_norms_active = np.linalg.norm(W_gate_dec[active_channels], axis=1)
        
        print(f"\n  Weight row norms:")
        print(f"    CONTRACT channels: mean={np.mean(w_norms_contract):.4f}")
        print(f"    Active channels:   mean={np.mean(w_norms_active):.4f}")
        
        # Row-level sign statistics for CONTRACT vs active
        sgn_f = w_sgn.astype(np.float32)
        bias_contract = np.abs(np.mean(sgn_f[contract_channels], axis=1))
        bias_active = np.abs(np.mean(sgn_f[active_channels], axis=1))
        
        print(f"    CONTRACT sign |bias|: mean={np.mean(bias_contract):.4f}")
        print(f"    Active sign |bias|:   mean={np.mean(bias_active):.4f}")
        
        # Rank-1 u values
        u_contract = np.abs(u_gate[contract_channels])
        u_active = np.abs(u_gate[active_channels])
        
        print(f"    CONTRACT |u| (row scale): mean={np.mean(u_contract):.2f}")
        print(f"    Active |u| (row scale):   mean={np.mean(u_active):.2f}")
    
    W_gate.clear_cache()
    W_up.clear_cache()


# ============================================================================
# PART 5: The Holographic Fringe Pattern
# ============================================================================

def holographic_fringe_analysis(W, residual, name=""):
    """
    If the sign matrix IS a holographic fringe pattern:
    - It should have spatial frequency structure
    - Bright and dark fringes should alternate with characteristic spacing
    - The fringe spacing relates to the "object" being encoded
    
    Test: autocorrelation of sign pattern along rows and columns.
    """
    sgn = W.signs.astype(np.float32)
    out_f, in_f = sgn.shape
    
    print(f"\n{'='*70}")
    print(f"  HOLOGRAPHIC FRINGE ANALYSIS ({name})")
    print(f"{'='*70}")
    
    # Sign run lengths: how many consecutive same-sign elements in rows?
    sample_rows = np.random.choice(out_f, 20, replace=False)
    
    all_runs = []
    for r in sample_rows:
        row = sgn[r]
        runs = []
        current_run = 1
        for i in range(1, len(row)):
            if row[i] == row[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        all_runs.extend(runs)
    
    all_runs = np.array(all_runs)
    print(f"\n  Sign run lengths (along rows):")
    print(f"    Mean: {np.mean(all_runs):.2f}")
    print(f"    Median: {np.median(all_runs):.0f}")
    print(f"    Max: {np.max(all_runs)}")
    print(f"    (Random binary: mean=2.0, median=1)")
    
    # Autocorrelation of sign along rows
    print(f"\n  Sign autocorrelation (row-wise):")
    lags = [1, 2, 4, 8, 16, 32, 64, 128]
    for lag in lags:
        if lag >= in_f:
            break
        corrs = []
        for r in sample_rows[:10]:
            c = np.corrcoef(sgn[r, :in_f-lag], sgn[r, lag:])[0, 1]
            corrs.append(c)
        print(f"    Lag {lag:>4d}: mean_corr={np.mean(corrs):.4f}")
    
    # Same for columns
    sample_cols = np.random.choice(in_f, 20, replace=False)
    print(f"\n  Sign autocorrelation (column-wise):")
    for lag in lags:
        if lag >= out_f:
            break
        corrs = []
        for c in sample_cols[:10]:
            cc = np.corrcoef(sgn[:out_f-lag, c], sgn[lag:, c])[0, 1]
            corrs.append(cc)
        print(f"    Lag {lag:>4d}: mean_corr={np.mean(corrs):.4f}")
    
    # Frequency content: FFT of sign pattern rows
    print(f"\n  Sign frequency content (FFT of rows):")
    spectra = np.zeros(in_f // 2 + 1)
    for r in sample_rows[:10]:
        fft = np.abs(np.fft.rfft(sgn[r]))
        spectra += fft
    spectra /= 10
    
    # Is spectrum flat (white noise) or peaked (structured)?
    peak_freq = np.argmax(spectra[1:]) + 1  # skip DC
    print(f"    DC component: {spectra[0]:.1f}")
    print(f"    Peak non-DC freq: {peak_freq} (period={in_f/peak_freq:.0f})")
    print(f"    Peak/mean ratio: {spectra[peak_freq]/np.mean(spectra[1:]):.2f}")
    print(f"    Spectrum flatness: {np.exp(np.mean(np.log(spectra[1:]+1e-10)))/np.mean(spectra[1:]):.4f}")
    print(f"    (1.0 = perfectly flat/white, <1 = peaked/structured)")


def run():
    print("=" * 70)
    print("  NEGATIVE ZERO CONNECTION")
    print("  DC 253 + Rank-1 Discovery = Same Structure")
    print("=" * 70)

    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)

    for wname in ['q_proj', 'gate_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        print(f"\n{'='*70}\n  {wname} ({W.shape[0]}×{W.shape[1]})\n{'='*70}")

        states, residual = weight_4_states(W, wname)
        energy_per_state(W, states, wname)
        negative_zero_test(W, states, wname)
        holographic_fringe_analysis(W, residual, wname)
        W.clear_cache()

    weight_predicts_gate()

    print(f"\n{'='*70}")
    print(f"  SYNTHESIS")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
