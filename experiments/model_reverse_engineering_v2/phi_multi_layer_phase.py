#!/usr/bin/env python3
"""
Multi-Layer Phase Shift: Does concept/detail separation emerge from gating?

Single-layer phase shift = 12° deflection, macro/full ratio = 1.000.
But the SiLU gate (x * σ(x)) is NONLINEAR. Through multiple layers:

  x → gate_proj → SiLU → ×up_proj → down_proj → residual → next layer

The gate amplifies large signals and kills small ones. After N layers:
- Macro groups (loud) survive the gate → concept persists
- Detail groups (quiet) get gated out → detail vanishes

Test: Run MLP forward pass through layers 0..N with:
  (a) FULL weights (all ε-groups)
  (b) MACRO weights (top K ε-groups only)
  (c) PHASE-SHIFTED weights (top group × φ^Δ)

Measure divergence after each layer. If the gate creates separation,
macro will CONVERGE to full (concept captured) while detail diverges.
"""

import os, sys, time, gc
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')

def levels(W): return W.exponents.astype(np.int32) // PHI_GRID

def extract_rank1(W):
    lvl = levels(W).astype(np.float32)
    U, s, Vt = np.linalg.svd(lvl, full_matrices=False)
    return U[:, 0] * s[0], Vt[0, :], lvl

def cos_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)

def angular_deflection(a, b):
    cs = np.clip(cos_sim(a, b), -1, 1)
    return np.degrees(np.arccos(cs))

def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def build_macro_weight(W_dec, eps_int, top_k_groups):
    """Zero out all ε-groups except the top K (by element count)."""
    unique_eps = np.unique(eps_int)
    counts = {int(k): np.sum(eps_int == k) for k in unique_eps}
    sorted_groups = sorted(counts.items(), key=lambda x: -x[1])
    keep = set(int(g[0]) for g in sorted_groups[:top_k_groups])
    mask = np.zeros_like(eps_int, dtype=bool)
    for k in keep:
        mask |= (eps_int == k)
    return W_dec * mask, keep


def build_shifted_weight(W_dec, eps_int, target_eps, delta):
    """Apply φ^delta phase shift to a specific ε-group."""
    mask = (eps_int == target_eps)
    W_out = W_dec.copy()
    W_out[mask] *= PHI ** delta
    return W_out


def load_mlp_layer(layer_idx):
    """Load and decode MLP weights for one layer. Returns (gate, up, down, eps_dicts)."""
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    
    result = {}
    for wname in ['gate_proj', 'up_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        u, v, lvl = extract_rank1(W)
        lvl_r1 = np.round(np.outer(u, v)).astype(np.int32)
        lvl_true = lvl.astype(np.int32)
        eps_int = lvl_true - lvl_r1
        result[wname] = {'W': W_dec, 'eps': eps_int}
        W.clear_cache()
        del W
    
    # Also load norms for RMSNorm
    norms_path = os.path.join(layer_dir, 'norms.npz')
    norms = np.load(norms_path)
    result['post_attn_norm'] = norms['post_attention_layernorm'].astype(np.float32)
    
    return result


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def mlp_forward(x, gate_W, up_W, down_W):
    """MLP forward: down(SiLU(gate(x)) * up(x))"""
    gate_out = x @ gate_W.T        # (batch, hidden_dim)
    up_out = x @ up_W.T            # (batch, hidden_dim)
    hidden = silu(gate_out) * up_out  # THE NONLINEAR GATE
    return hidden @ down_W.T       # (batch, model_dim)


def run():
    print("=" * 70)
    print("  MULTI-LAYER PHASE SHIFT: Gating Creates Concept Separation?")
    print("=" * 70)
    
    # Use layers 0-7 (8 layers = enough to see trend, manageable memory)
    N_LAYERS = 8
    N_MACRO_GROUPS = 5
    N_INPUTS = 20
    PHASE_DELTA = 1.0  # φ^1 shift
    
    np.random.seed(42)
    hidden_dim = 3584  # Qwen2-7B hidden size
    
    # Generate test inputs
    inputs = [np.random.randn(1, hidden_dim).astype(np.float32) * 0.1 
              for _ in range(N_INPUTS)]
    
    # Track metrics per layer
    print(f"\n  Config: {N_LAYERS} layers, {N_MACRO_GROUPS} macro groups, "
          f"{N_INPUTS} inputs, φ^{PHASE_DELTA} shift")
    
    # ================================================================
    # EXPERIMENT 1: Macro vs Full through layers
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1: Macro ({N_MACRO_GROUPS} groups) vs Full through MLP layers")
    print(f"{'='*70}")
    
    # Initialize state vectors for each input
    states_full = [x.copy() for x in inputs]
    states_macro = [x.copy() for x in inputs]
    states_shifted = [x.copy() for x in inputs]
    
    layer_metrics = []
    
    for layer_idx in range(N_LAYERS):
        t0 = time.time()
        print(f"\n  Layer {layer_idx}:", end=' ', flush=True)
        
        mlp = load_mlp_layer(layer_idx)
        norm_w = mlp['post_attn_norm']
        
        gate_W = mlp['gate_proj']['W']
        up_W = mlp['up_proj']['W']
        down_W = mlp['down_proj']['W']
        gate_eps = mlp['gate_proj']['eps']
        up_eps = mlp['up_proj']['eps']
        down_eps = mlp['down_proj']['eps']
        
        # Build macro versions (top K groups only)
        gate_macro, gate_keep = build_macro_weight(gate_W, gate_eps, N_MACRO_GROUPS)
        up_macro, up_keep = build_macro_weight(up_W, up_eps, N_MACRO_GROUPS)
        down_macro, down_keep = build_macro_weight(down_W, down_eps, N_MACRO_GROUPS)
        
        # Build phase-shifted versions (shift top group by φ^Δ)
        # Find the globally dominant ε-group for this layer's gate
        unique_eps = np.unique(gate_eps)
        counts = {int(k): np.sum(gate_eps == k) for k in unique_eps}
        top_eps = max(counts, key=counts.get)
        
        gate_shifted = build_shifted_weight(gate_W, gate_eps, top_eps, PHASE_DELTA)
        up_shifted = build_shifted_weight(up_W, up_eps, top_eps, PHASE_DELTA)
        down_shifted = build_shifted_weight(down_W, down_eps, top_eps, PHASE_DELTA)
        
        # Forward pass for each input
        angles_macro = []
        angles_shifted = []
        cos_sims_macro = []
        cos_sims_shifted = []
        mag_ratios_macro = []
        mag_ratios_shifted = []
        
        for i in range(N_INPUTS):
            # RMSNorm before MLP
            x_full = rms_norm(states_full[i], norm_w)
            x_macro = rms_norm(states_macro[i], norm_w)
            x_shifted = rms_norm(states_shifted[i], norm_w)
            
            # MLP forward
            mlp_full = mlp_forward(x_full, gate_W, up_W, down_W)
            mlp_macro = mlp_forward(x_macro, gate_macro, up_macro, down_macro)
            mlp_shifted = mlp_forward(x_shifted, gate_shifted, up_shifted, down_shifted)
            
            # Residual connection
            states_full[i] = states_full[i] + mlp_full
            states_macro[i] = states_macro[i] + mlp_macro
            states_shifted[i] = states_shifted[i] + mlp_shifted
            
            # Metrics: compare macro/shifted to full
            a_macro = angular_deflection(states_macro[i], states_full[i])
            a_shifted = angular_deflection(states_shifted[i], states_full[i])
            cs_macro = cos_sim(states_macro[i], states_full[i])
            cs_shifted = cos_sim(states_shifted[i], states_full[i])
            mr_macro = np.linalg.norm(states_macro[i]) / (np.linalg.norm(states_full[i]) + 1e-30)
            mr_shifted = np.linalg.norm(states_shifted[i]) / (np.linalg.norm(states_full[i]) + 1e-30)
            
            angles_macro.append(a_macro)
            angles_shifted.append(a_shifted)
            cos_sims_macro.append(cs_macro)
            cos_sims_shifted.append(cs_shifted)
            mag_ratios_macro.append(mr_macro)
            mag_ratios_shifted.append(mr_shifted)
        
        metrics = {
            'layer': layer_idx,
            'top_eps': top_eps,
            'angle_macro_mean': np.mean(angles_macro),
            'angle_macro_std': np.std(angles_macro),
            'angle_shifted_mean': np.mean(angles_shifted),
            'angle_shifted_std': np.std(angles_shifted),
            'cos_macro_mean': np.mean(cos_sims_macro),
            'cos_shifted_mean': np.mean(cos_sims_shifted),
            'mag_macro_mean': np.mean(mag_ratios_macro),
            'mag_shifted_mean': np.mean(mag_ratios_shifted),
        }
        layer_metrics.append(metrics)
        
        elapsed = time.time() - t0
        print(f"top_ε={top_eps}, "
              f"macro_angle={metrics['angle_macro_mean']:.1f}°±{metrics['angle_macro_std']:.1f}, "
              f"shift_angle={metrics['angle_shifted_mean']:.1f}°±{metrics['angle_shifted_std']:.1f} "
              f"({elapsed:.1f}s)")
        
        # Free memory
        del mlp, gate_W, up_W, down_W, gate_macro, up_macro, down_macro
        del gate_shifted, up_shifted, down_shifted
        gc.collect()
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"  LAYER-BY-LAYER SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Layer':>5s}  {'ε':>3s}  {'Macro angle°':>12s}  {'Shift angle°':>12s}  "
          f"{'Macro cos':>10s}  {'Shift cos':>10s}  {'Macro |y|':>10s}  {'Shift |y|':>10s}")
    
    for m in layer_metrics:
        print(f"  {m['layer']:>5d}  {m['top_eps']:>3d}  "
              f"{m['angle_macro_mean']:>12.2f}  {m['angle_shifted_mean']:>12.2f}  "
              f"{m['cos_macro_mean']:>10.6f}  {m['cos_shifted_mean']:>10.6f}  "
              f"{m['mag_macro_mean']:>10.4f}  {m['mag_shifted_mean']:>10.4f}")
    
    # Trend analysis
    print(f"\n{'='*70}")
    print(f"  TREND ANALYSIS")
    print(f"{'='*70}")
    
    macro_angles = [m['angle_macro_mean'] for m in layer_metrics]
    shift_angles = [m['angle_shifted_mean'] for m in layer_metrics]
    
    # Is the angle growing, shrinking, or stable?
    if len(macro_angles) >= 3:
        # Linear fit
        layers = np.arange(len(macro_angles))
        
        macro_fit = np.polyfit(layers, macro_angles, 1)
        shift_fit = np.polyfit(layers, shift_angles, 1)
        
        print(f"\n  Macro angle trend: {macro_fit[0]:+.2f}°/layer "
              f"(start={macro_angles[0]:.1f}°, end={macro_angles[-1]:.1f}°)")
        print(f"  Shift angle trend: {shift_fit[0]:+.2f}°/layer "
              f"(start={shift_angles[0]:.1f}°, end={shift_angles[-1]:.1f}°)")
        
        if macro_fit[0] < -0.5:
            print(f"\n  → MACRO CONVERGES: The gate is filtering detail!")
            print(f"    After {N_LAYERS} layers, macro is {macro_angles[-1]:.1f}° from full")
            print(f"    The nonlinearity IS creating concept/detail separation")
        elif macro_fit[0] > 0.5:
            print(f"\n  → MACRO DIVERGES: The gate amplifies differences!")
            print(f"    After {N_LAYERS} layers, macro is {macro_angles[-1]:.1f}° from full")
            print(f"    The missing groups accumulate error through gating")
        else:
            print(f"\n  → MACRO STABLE: angle stays ~constant through layers")
            print(f"    The gate neither converges nor diverges macro")
        
        if shift_fit[0] > 0.5:
            print(f"\n  → PHASE SHIFT COMPOUNDS: {shift_angles[0]:.1f}° → {shift_angles[-1]:.1f}°")
            print(f"    Each layer amplifies the shift by ~{shift_fit[0]:.1f}°")
            print(f"    At 28 layers: projected {shift_angles[0] + shift_fit[0]*27:.0f}° deflection")
        elif shift_fit[0] < -0.5:
            print(f"\n  → PHASE SHIFT DAMPENS: the funnel absorbs the perturbation")
        else:
            print(f"\n  → PHASE SHIFT STABLE: ~{np.mean(shift_angles):.1f}° regardless of depth")
    
    # ================================================================
    # EXPERIMENT 2: Rank evolution through layers
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 2: Output rank evolution (macro vs full)")
    print(f"{'='*70}")
    
    # Collect all full/macro states across inputs
    full_mat = np.array([s.flatten() for s in states_full])
    macro_mat = np.array([s.flatten() for s in states_macro])
    shifted_mat = np.array([s.flatten() for s in states_shifted])
    
    def effective_rank(mat, threshold=0.99):
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        cumvar = np.cumsum(s**2) / np.sum(s**2)
        return np.searchsorted(cumvar, threshold) + 1
    
    rank_full = effective_rank(full_mat)
    rank_macro = effective_rank(macro_mat)
    rank_shifted = effective_rank(shifted_mat)
    
    print(f"\n  After {N_LAYERS} layers ({N_INPUTS} inputs):")
    print(f"    Full effective rank (99%):    {rank_full}")
    print(f"    Macro effective rank (99%):   {rank_macro}")
    print(f"    Shifted effective rank (99%): {rank_shifted}")
    
    if rank_macro < rank_full:
        print(f"\n    → Macro output lives in LOWER-dimensional subspace!")
        print(f"      Ratio: {rank_macro}/{rank_full} = {rank_macro/rank_full:.2f}")
        print(f"      The nonlinearity HAS separated concept from detail")
    else:
        print(f"\n    → Same dimensionality — separation not yet visible at {N_LAYERS} layers")
    
    # ================================================================
    # EXPERIMENT 3: Gate selectivity analysis
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 3: Gate selectivity (what does SiLU kill?)")
    print(f"{'='*70}")
    
    # Reload layer 0 to analyze gate behavior
    mlp = load_mlp_layer(0)
    gate_W = mlp['gate_proj']['W']
    gate_eps = mlp['gate_proj']['eps']
    norm_w = mlp['post_attn_norm']
    
    x_test = rms_norm(inputs[0], norm_w)
    gate_out = x_test @ gate_W.T
    gate_activated = silu(gate_out)
    
    # What fraction of gate output is killed (< 10% of max)?
    gate_max = np.max(np.abs(gate_activated))
    alive_frac = np.mean(np.abs(gate_activated) > 0.1 * gate_max)
    dead_frac = 1 - alive_frac
    
    print(f"\n  Gate output statistics (layer 0, single input):")
    print(f"    Range: [{gate_activated.min():.4f}, {gate_activated.max():.4f}]")
    print(f"    Alive (>10% of max): {alive_frac:.1%}")
    print(f"    Dead (<10% of max):  {dead_frac:.1%}")
    
    # Now compare: macro gate vs full gate — which elements survive?
    gate_macro, _ = build_macro_weight(gate_W, gate_eps, N_MACRO_GROUPS)
    gate_out_macro = x_test @ gate_macro.T
    gate_act_macro = silu(gate_out_macro)
    
    # Correlation between full and macro gate activations
    gate_corr = cos_sim(gate_activated, gate_act_macro)
    
    # Binary agreement: do they kill the same elements?
    full_alive = np.abs(gate_activated.flatten()) > 0.1 * gate_max
    macro_alive = np.abs(gate_act_macro.flatten()) > 0.1 * np.max(np.abs(gate_act_macro))
    agreement = np.mean(full_alive == macro_alive)
    
    print(f"\n  Macro vs Full gate comparison:")
    print(f"    Activation cos_sim: {gate_corr:.6f}")
    print(f"    Binary gate agreement: {agreement:.1%}")
    
    if agreement > 0.9:
        print(f"    → Macro gate makes SAME kill decisions as full gate")
        print(f"      The top {N_MACRO_GROUPS} groups determine the binary routing!")
    
    # Which gate elements are DIFFERENTLY activated?
    disagree = full_alive != macro_alive
    n_disagree = np.sum(disagree)
    print(f"    Elements with different gate decision: {n_disagree} "
          f"({n_disagree/len(full_alive.flatten()):.1%})")
    
    del mlp, gate_W, gate_macro
    gc.collect()
    
    # ================================================================
    # SYNTHESIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS: Multi-Layer Phase Shift")
    print(f"{'='*70}")
    print(f"""
  Through {N_LAYERS} MLP layers with SiLU gating and residual connections:
  
  MACRO ({N_MACRO_GROUPS} groups):
    Start angle: {macro_angles[0]:.1f}°, End angle: {macro_angles[-1]:.1f}°
    Trend: {macro_fit[0]:+.2f}°/layer
    
  PHASE SHIFT (φ^{PHASE_DELTA}):
    Start angle: {shift_angles[0]:.1f}°, End angle: {shift_angles[-1]:.1f}°
    Trend: {shift_fit[0]:+.2f}°/layer
    
  KEY QUESTION ANSWERS:
    1. Can we process at concept scale (macro)?
       → After {N_LAYERS} layers: macro at {macro_angles[-1]:.1f}° from full
    2. Do phase shifts compound through layers?
       → Trend: {shift_fit[0]:+.2f}°/layer
    3. Does the gate create concept/detail separation?
       → Gate agreement: {{measured above}}
    4. Is the funnel manipulable?
       → Phase shift is {'compounding' if shift_fit[0] > 0.5 else 'stable' if abs(shift_fit[0]) < 0.5 else 'dampening'}
""")


if __name__ == '__main__':
    run()
