#!/usr/bin/env python3
"""
Phase Shift Sweep: Find the controllable regime.

Previous results: φ^1 shift blows up (7.5× mag, 68° angle after 8 layers).
This sweep finds the sweet spot where shift compounds meaningfully without exploding.

MEMORY-EFFICIENT: loads one layer at a time, processes all experiments for that layer,
then frees the weights before loading the next layer.
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
    return np.degrees(np.arccos(np.clip(cos_sim(a, b), -1, 1)))

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

def rms_norm(x, weight, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * weight

def mlp_forward(x, gate_W, up_W, down_W):
    return (silu(x @ gate_W.T) * (x @ up_W.T)) @ down_W.T

def load_mlp_layer(layer_idx):
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    result = {}
    for wname in ['gate_proj', 'up_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        u, v, lvl = extract_rank1(W)
        lvl_r1 = np.round(np.outer(u, v)).astype(np.int32)
        eps_int = lvl.astype(np.int32) - lvl_r1
        result[wname] = {'W': W_dec, 'eps': eps_int}
        W.clear_cache(); del W
    norms = np.load(os.path.join(layer_dir, 'norms.npz'))
    result['norm'] = norms['post_attention_layernorm'].astype(np.float32)
    return result

def get_top_eps(eps_int):
    unique, counts = np.unique(eps_int, return_counts=True)
    return int(unique[np.argmax(counts)])

def build_shifted(W, eps, target_eps, delta):
    out = W.copy(); out[eps == target_eps] *= PHI ** delta; return out

def build_macro(W, eps, n_groups):
    unique, counts = np.unique(eps, return_counts=True)
    top_k = set(int(unique[i]) for i in np.argsort(-counts)[:n_groups])
    mask = np.zeros_like(eps, dtype=bool)
    for k in top_k: mask |= (eps == k)
    return W * mask


def run():
    print("=" * 70)
    print("  PHASE SHIFT SWEEP (memory-efficient)")
    print("=" * 70)
    sys.stdout.flush()

    N_LAYERS = 8
    N_INPUTS = 10
    HD = 3584

    deltas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    group_counts = [3, 5, 7, 10, 15, 20, 30]

    np.random.seed(42)
    base_inputs = [np.random.randn(1, HD).astype(np.float32) * 0.1 for _ in range(N_INPUTS)]

    # State arrays: one set per delta, one set per group count
    # EXP1: phase shift states
    shift_states = {d: [x.copy() for x in base_inputs] for d in deltas}
    full_states_exp1 = [x.copy() for x in base_inputs]

    # EXP3: macro group count states (4 layers only)
    macro_states = {ng: [x.copy() for x in base_inputs] for ng in group_counts}
    full_states_exp3 = [x.copy() for x in base_inputs]

    # EXP4: where to shift (4 layers, δ=0.1)
    shift_configs = [('gate', True, False, False), ('up', False, True, False),
                     ('down', False, False, True), ('gate+up', True, True, False),
                     ('all', True, True, True)]
    config_states = {name: [x.copy() for x in base_inputs] for name, *_ in shift_configs}
    full_states_exp4 = [x.copy() for x in base_inputs]

    # Accumulators for results
    exp1_angles = {d: [] for d in deltas}   # per delta, per layer
    exp1_mags = {d: [] for d in deltas}
    exp2_data = {}  # filled at layer 0 only
    exp3_angles = {ng: [] for ng in group_counts}  # per ng, per layer (0-3)
    exp4_angles = {name: [] for name, *_ in shift_configs}

    for l_idx in range(N_LAYERS):
        t0 = time.time()
        print(f"\n  Loading layer {l_idx}...", end=' ', flush=True)
        mlp = load_mlp_layer(l_idx)
        nw = mlp['norm']
        gW = mlp['gate_proj']['W']; ge = mlp['gate_proj']['eps']
        uW = mlp['up_proj']['W'];  ue = mlp['up_proj']['eps']
        dW = mlp['down_proj']['W']; de = mlp['down_proj']['eps']
        te = get_top_eps(ge)
        print(f"done ({time.time()-t0:.1f}s), top_ε={te}", flush=True)

        # ── EXP 1: Phase magnitude sweep ──
        # Advance full states
        for i in range(N_INPUTS):
            xn = rms_norm(full_states_exp1[i], nw)
            full_states_exp1[i] = full_states_exp1[i] + mlp_forward(xn, gW, uW, dW)

        for delta in deltas:
            gs = build_shifted(gW, ge, te, delta)
            us = build_shifted(uW, ue, te, delta)
            ds = build_shifted(dW, de, te, delta)
            angles = []
            for i in range(N_INPUTS):
                xn = rms_norm(shift_states[delta][i], nw)
                shift_states[delta][i] = shift_states[delta][i] + mlp_forward(xn, gs, us, ds)
                angles.append(angular_deflection(shift_states[delta][i], full_states_exp1[i]))
            exp1_angles[delta].append(np.mean(angles))
            exp1_mags[delta].append(np.mean([
                np.linalg.norm(shift_states[delta][i]) / (np.linalg.norm(full_states_exp1[i]) + 1e-30)
                for i in range(N_INPUTS)]))
            del gs, us, ds

        # ── EXP 2: Gate agreement (layer 0 only) ──
        if l_idx == 0:
            for ng in group_counts:
                gm = build_macro(gW, ge, ng)
                agrees, cs_list = [], []
                for i in range(N_INPUTS):
                    x = rms_norm(base_inputs[i], nw)
                    gf = silu(x @ gW.T); gmc = silu(x @ gm.T)
                    tf = 0.1 * np.max(np.abs(gf)); tm = 0.1 * np.max(np.abs(gmc))
                    agrees.append(np.mean((np.abs(gf.flatten()) > tf) == (np.abs(gmc.flatten()) > tm)))
                    cs_list.append(cos_sim(gf, gmc))
                exp2_data[ng] = (np.mean(agrees), np.mean(cs_list))
                del gm

        # ── EXP 3: Macro group count (first 4 layers) ──
        if l_idx < 4:
            for i in range(N_INPUTS):
                xn = rms_norm(full_states_exp3[i], nw)
                full_states_exp3[i] = full_states_exp3[i] + mlp_forward(xn, gW, uW, dW)

            for ng in group_counts:
                gm = build_macro(gW, ge, ng)
                um = build_macro(uW, ue, ng)
                dm = build_macro(dW, de, ng)
                angles = []
                for i in range(N_INPUTS):
                    xn = rms_norm(macro_states[ng][i], nw)
                    macro_states[ng][i] = macro_states[ng][i] + mlp_forward(xn, gm, um, dm)
                    angles.append(angular_deflection(macro_states[ng][i], full_states_exp3[i]))
                exp3_angles[ng].append(np.mean(angles))
                del gm, um, dm

        # ── EXP 4: Where to shift (first 4 layers, δ=0.1) ──
        if l_idx < 4:
            for i in range(N_INPUTS):
                xn = rms_norm(full_states_exp4[i], nw)
                full_states_exp4[i] = full_states_exp4[i] + mlp_forward(xn, gW, uW, dW)

            for name, sg, su, sd in shift_configs:
                g_ = build_shifted(gW, ge, te, 0.1) if sg else gW
                u_ = build_shifted(uW, ue, te, 0.1) if su else uW
                d_ = build_shifted(dW, de, te, 0.1) if sd else dW
                angles = []
                for i in range(N_INPUTS):
                    xn = rms_norm(config_states[name][i], nw)
                    config_states[name][i] = config_states[name][i] + mlp_forward(xn, g_, u_, d_)
                    angles.append(angular_deflection(config_states[name][i], full_states_exp4[i]))
                exp4_angles[name].append(np.mean(angles))
                if sg: del g_
                if su: del u_
                if sd: del d_

        # Free layer weights
        del mlp, gW, uW, dW, ge, ue, de, nw
        gc.collect()

        elapsed = time.time() - t0
        print(f"  Layer {l_idx} processed ({elapsed:.1f}s total)", flush=True)

    # ════════════════════════════════════════════════════════════════════
    # PRINT ALL RESULTS
    # ════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  EXP 1: Phase Magnitude Sweep ({N_LAYERS} layers)")
    print(f"{'='*70}")
    print(f"  {'δ':>6s}  {'φ^δ':>7s}", end='')
    for l in range(N_LAYERS): print(f"  {'L'+str(l):>7s}", end='')
    print(f"  {'Mag':>7s}")
    for delta in deltas:
        print(f"  {delta:>6.2f}  {PHI**delta:>7.4f}", end='')
        for a in exp1_angles[delta]: print(f"  {a:>6.1f}°", end='')
        print(f"  {exp1_mags[delta][-1]:>7.3f}")

    # Sweet spot analysis
    print(f"\n  SWEET SPOT: largest δ with mag < 1.5 after {N_LAYERS} layers:")
    for delta in reversed(deltas):
        if exp1_mags[delta][-1] < 1.5:
            print(f"  → δ = {delta:.2f} (angle={exp1_angles[delta][-1]:.1f}°, "
                  f"mag={exp1_mags[delta][-1]:.3f})")
            break

    print(f"\n{'='*70}")
    print(f"  EXP 2: Gate Agreement vs ε-Group Count (Layer 0)")
    print(f"{'='*70}")
    print(f"  {'Groups':>6s}  {'Gate agree':>10s}  {'cos_sim':>8s}")
    for ng in group_counts:
        if ng in exp2_data:
            a, c = exp2_data[ng]
            print(f"  {ng:>6d}  {a:>9.1%}  {c:>8.4f}")

    print(f"\n{'='*70}")
    print(f"  EXP 3: Macro Groups vs Multi-Layer Divergence (4 layers)")
    print(f"{'='*70}")
    print(f"  {'Groups':>6s}  {'L0':>8s}  {'L1':>8s}  {'L2':>8s}  {'L3':>8s}")
    for ng in group_counts:
        angles = exp3_angles[ng]
        if len(angles) == 4:
            print(f"  {ng:>6d}  {angles[0]:>7.1f}°  {angles[1]:>7.1f}°  "
                  f"{angles[2]:>7.1f}°  {angles[3]:>7.1f}°")

    print(f"\n{'='*70}")
    print(f"  EXP 4: Where to Apply Phase Shift (δ=0.1, 4 layers)")
    print(f"{'='*70}")
    for name, *_ in shift_configs:
        angles = exp4_angles[name]
        if len(angles) == 4:
            print(f"  {name:>12s}: L0={angles[0]:.2f}° L1={angles[1]:.2f}° "
                  f"L2={angles[2]:.2f}° L3={angles[3]:.2f}°")

    # Final synthesis
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS")
    print(f"{'='*70}")
    print(f"\n  Phase shift through {N_LAYERS} MLP layers with SiLU gating:")
    for delta in deltas:
        a0, aL = exp1_angles[delta][0], exp1_angles[delta][-1]
        m = exp1_mags[delta][-1]
        label = "CONTROLLABLE" if m < 1.5 else "MODERATE" if m < 3.0 else "EXPLOSIVE"
        print(f"    δ={delta:.2f}: {a0:.1f}° → {aL:.1f}° (mag {m:.2f}×) [{label}]")

    print(f"\n  Done.")


if __name__ == '__main__':
    run()
