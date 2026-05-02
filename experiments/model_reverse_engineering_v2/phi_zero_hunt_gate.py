#!/usr/bin/env python3
"""
Zero-Hunting on the Phase Shift Control Surface

Applying the rhzeros three-stage pipeline to ε-group phase shifts:
  1. COMPRESSOR (Lambert W analog): Compute gate activations h_j, identify near-zero dims
  2. PROCESSOR (Ramanujan analog): Compute sensitivity dh_j/dδ per dimension per ε-group
  3. TARGETER (Z(t) + Newton analog): Find exact δ_j where gate flips (h_j crosses zero)

This maps to DC 253 four states:
  +1: h_j >> 0 (gate fully open)
  +0: h_j slightly > 0 (barely open, near zero from positive side)
  -0: h_j slightly < 0 (barely closed, near zero from negative side)
  -1: h_j << 0 (gate fully shut)

The "zeros" are exact δ values where dimensions cross +0 ↔ -0.
A zero spectrum gives us precision control over gate routing.
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

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def rms_norm(x, weight, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * weight

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

def get_sorted_groups(eps_int):
    unique, counts = np.unique(eps_int, return_counts=True)
    order = np.argsort(-counts)
    return [(int(unique[i]), int(counts[i])) for i in order]


def classify_four_states(h, threshold=0.1):
    """Classify gate activations into DC 253 four states."""
    mag = np.abs(h.flatten())
    sign = np.sign(h.flatten())
    max_mag = np.max(mag) if np.max(mag) > 0 else 1.0

    states = np.zeros_like(h.flatten(), dtype=np.int8)
    # +1: strongly positive (gate open)
    states[(sign > 0) & (mag > threshold * max_mag)] = 1
    # +0: barely positive (near zero from positive side)
    states[(sign > 0) & (mag <= threshold * max_mag)] = 2
    # -0: barely negative (near zero from negative side)
    states[(sign <= 0) & (mag <= threshold * max_mag)] = 3
    # -1: strongly negative (gate shut)
    states[(sign <= 0) & (mag > threshold * max_mag)] = 4
    return states  # 1=+1, 2=+0, 3=-0, 4=-1


def run():
    print("=" * 70)
    print("  ZERO-HUNTING ON THE PHASE SHIFT CONTROL SURFACE")
    print("  (rhzeros pipeline applied to SiLU gate)")
    print("=" * 70)
    sys.stdout.flush()

    HD = 3584
    N_INPUTS = 5

    np.random.seed(42)
    inputs = [np.random.randn(1, HD).astype(np.float32) * 0.1 for _ in range(N_INPUTS)]

    # Work with layer 0 first (single layer, memory efficient)
    for layer_idx in [0, 3]:
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        mlp = load_mlp_layer(layer_idx)
        gate_W = mlp['gate_proj']['W']
        gate_eps = mlp['gate_proj']['eps']
        norm_w = mlp['norm']
        sorted_groups = get_sorted_groups(gate_eps)

        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Gate shape: {gate_W.shape}, {len(sorted_groups)} ε-groups")
        print(f"  Top 5 groups: {sorted_groups[:5]}")
        sys.stdout.flush()

        # ════════════════════════════════════════════════════════════════
        # STAGE 1: COMPRESSOR — Map the gate landscape
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── STAGE 1: COMPRESSOR (Lambert W analog) ──")
        print(f"  Map gate activations, classify four states")

        for inp_idx in range(N_INPUTS):
            x = rms_norm(inputs[inp_idx], norm_w)
            h = (x @ gate_W.T).flatten()  # gate pre-activation [intermediate_dim]

            states = classify_four_states(h, threshold=0.1)
            n_plus1 = np.sum(states == 1)
            n_plus0 = np.sum(states == 2)
            n_minus0 = np.sum(states == 3)
            n_minus1 = np.sum(states == 4)

            # SiLU output
            g = silu(h)
            alive = np.abs(g) > 0.1 * np.max(np.abs(g))

            if inp_idx == 0:
                print(f"\n    Input {inp_idx}: h range [{h.min():.4f}, {h.max():.4f}]")
                print(f"    Four states: +1={n_plus1} ({n_plus1/len(h):.1%}), "
                      f"+0={n_plus0} ({n_plus0/len(h):.1%}), "
                      f"-0={n_minus0} ({n_minus0/len(h):.1%}), "
                      f"-1={n_minus1} ({n_minus1/len(h):.1%})")
                print(f"    SiLU alive: {np.sum(alive)} ({np.sum(alive)/len(h):.1%})")

                # The near-zero dims (+0 and -0) are the SENSITIVE ones
                near_zero_mask = (states == 2) | (states == 3)
                print(f"    Near-zero (flippable): {np.sum(near_zero_mask)} dims")
                print(f"    → These are the 'zeros' we can hunt")

        # Use first input for detailed analysis
        x0 = rms_norm(inputs[0], norm_w)
        h0 = (x0 @ gate_W.T).flatten()

        # ════════════════════════════════════════════════════════════════
        # STAGE 2: PROCESSOR — Compute sensitivities per ε-group
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── STAGE 2: PROCESSOR (Ramanujan analog) ──")
        print(f"  Compute dh_j/dδ for each dimension and ε-group")

        # For each ε-group k, the contribution to h_j from group k elements is:
        #   c_j^k = Σ_{i where eps[j,i]==k} gate_W[j,i] * x[i]
        # When we shift group k by φ^δ, the new h_j = h_j + c_j^k * (φ^δ - 1)
        # So dh_j/dδ ≈ c_j^k * φ^δ * ln(φ)  (at δ=0, ≈ c_j^k * ln(φ))
        # Critical δ: h_j + c_j^k * (φ^δ - 1) = 0
        # → φ^δ = 1 - h_j / c_j^k
        # → δ = log_φ(1 - h_j / c_j^k)

        x_flat = x0.flatten()
        top_groups = sorted_groups[:5]  # Analyze top 5 groups

        print(f"\n    Per-group contribution to gate activations:")
        print(f"    {'Group':>6s}  {'Count':>6s}  {'Mean |c|':>9s}  "
              f"{'Flippable':>9s}  {'%of near-0':>10s}")

        group_contributions = {}

        for eps_val, eps_count in top_groups:
            # Mask for this group: shape [intermediate_dim, hidden_dim]
            mask = (gate_eps == eps_val)  # bool [int_dim, hid_dim]

            # Contribution of this group to each output dim
            # c_j = Σ_i (gate_W[j,i] * x[i]) where eps[j,i] == eps_val
            W_masked = gate_W * mask
            c = (W_masked @ x_flat).flatten()  # [intermediate_dim]
            group_contributions[eps_val] = c

            # How many near-zero dims can this group flip?
            # Critical: h_j + c_j * (φ^δ - 1) = 0 → need c_j ≠ 0 and h_j/c_j ≠ 0
            near_zero = (np.abs(h0) < 0.1 * np.max(np.abs(h0)))
            can_flip = near_zero & (np.abs(c) > 1e-10)

            print(f"    {eps_val:>6d}  {eps_count:>6d}  {np.mean(np.abs(c)):>9.6f}  "
                  f"{np.sum(can_flip):>9d}  {np.sum(can_flip)/max(1,np.sum(near_zero)):>9.1%}")

            del W_masked

        # ════════════════════════════════════════════════════════════════
        # STAGE 3: TARGETER — Hunt the zeros
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── STAGE 3: TARGETER (Z(t) + Newton analog) ──")
        print(f"  Find exact δ where each gate dimension flips")

        # For the top ε-group, compute critical δ for each dimension
        top_eps_val = sorted_groups[0][0]
        c_top = group_contributions[top_eps_val]

        # Critical δ: h_j + c_j * (φ^δ - 1) = 0
        # φ^δ = 1 - h_j / c_j
        # δ = log(1 - h_j / c_j) / log(φ)

        valid = np.abs(c_top) > 1e-10
        ratio = np.full_like(h0, np.nan)
        ratio[valid] = 1.0 - h0[valid] / c_top[valid]

        # δ only exists if ratio > 0 (otherwise no real solution)
        has_zero = valid & (ratio > 0)
        delta_critical = np.full_like(h0, np.nan)
        delta_critical[has_zero] = np.log(ratio[has_zero]) / np.log(PHI)

        n_zeros = np.sum(has_zero)
        print(f"\n    Top group (ε={top_eps_val}): {n_zeros} dims have real zeros "
              f"({n_zeros}/{gate_W.shape[0]})")

        # Sort zeros by |δ| (closest first = most easily flipped)
        zero_dims = np.where(has_zero)[0]
        zero_deltas = delta_critical[has_zero]
        sort_order = np.argsort(np.abs(zero_deltas))
        zero_dims = zero_dims[sort_order]
        zero_deltas = zero_deltas[sort_order]

        # Zero spectrum
        print(f"\n    ZERO SPECTRUM (top group ε={top_eps_val}):")
        print(f"    Sorted by |δ| (easiest flips first):")
        print(f"    {'Rank':>5s}  {'Dim':>6s}  {'δ_crit':>10s}  {'h_j':>10s}  "
              f"{'c_j':>10s}  {'State':>6s}  {'Flip':>8s}")

        state_names = {1: '+1', 2: '+0', 3: '-0', 4: '-1'}
        states0 = classify_four_states(h0, threshold=0.1)

        for rank in range(min(30, len(zero_dims))):
            j = zero_dims[rank]
            d = zero_deltas[rank]
            s = states0[j]
            # After flip: sign changes
            flip_dir = '-' if h0[j] > 0 else '+'
            print(f"    {rank:>5d}  {j:>6d}  {d:>10.6f}  {h0[j]:>10.6f}  "
                  f"{c_top[j]:>10.6f}  {state_names.get(s,'?'):>6s}  →{flip_dir}")

        # ════════════════════════════════════════════════════════════════
        # VERIFICATION: Do the predicted zeros actually work?
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── VERIFICATION: Do predicted zeros flip the gate? ──")

        n_verify = min(20, len(zero_dims))
        correct = 0
        for rank in range(n_verify):
            j = zero_dims[rank]
            d = zero_deltas[rank]

            # Build shifted weight for this δ
            mask_eps = (gate_eps == top_eps_val)
            gate_shifted = gate_W.copy()
            gate_shifted[mask_eps] *= PHI ** d

            h_shifted = (x0 @ gate_shifted.T).flatten()

            # Check if sign of h[j] actually flipped
            orig_sign = np.sign(h0[j])
            new_sign = np.sign(h_shifted[j])
            flipped = (orig_sign != new_sign) or (np.abs(h_shifted[j]) < 1e-8)

            if flipped:
                correct += 1

            if rank < 10:
                print(f"    δ={d:>10.6f}: h[{j}] = {h0[j]:>10.6f} → {h_shifted[j]:>10.6f} "
                      f"{'✓ FLIPPED' if flipped else '✗ SAME'}")

            del gate_shifted

        print(f"\n    Verification: {correct}/{n_verify} predicted zeros actually flip "
              f"({correct/n_verify:.0%})")

        # ════════════════════════════════════════════════════════════════
        # ANALYSIS: Zero spectrum statistics
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── ZERO SPECTRUM ANALYSIS ──")

        # Distribution of critical deltas
        abs_deltas = np.abs(zero_deltas)
        print(f"\n    {len(zero_deltas)} zeros total for top ε-group")
        print(f"    |δ| distribution:")
        print(f"      Min:    {abs_deltas.min():.6f}")
        print(f"      10th%:  {np.percentile(abs_deltas, 10):.6f}")
        print(f"      Median: {np.median(abs_deltas):.4f}")
        print(f"      90th%:  {np.percentile(abs_deltas, 90):.4f}")
        print(f"      Max:    {abs_deltas.max():.4f}")

        # How many zeros in the controllable regime (|δ| < 0.1)?
        controllable = abs_deltas < 0.1
        moderate = (abs_deltas >= 0.1) & (abs_deltas < 0.5)
        beyond = abs_deltas >= 0.5

        print(f"\n    Zeros by regime:")
        print(f"      Controllable (|δ|<0.1): {np.sum(controllable)} "
              f"({np.sum(controllable)/len(abs_deltas):.1%})")
        print(f"      Moderate (0.1≤|δ|<0.5): {np.sum(moderate)} "
              f"({np.sum(moderate)/len(abs_deltas):.1%})")
        print(f"      Beyond (|δ|≥0.5):       {np.sum(beyond)} "
              f"({np.sum(beyond)/len(abs_deltas):.1%})")

        # Sign of δ: positive = amplify group, negative = attenuate
        pos_deltas = zero_deltas > 0
        neg_deltas = zero_deltas < 0
        print(f"\n    Direction:")
        print(f"      δ > 0 (amplify to flip):   {np.sum(pos_deltas)}")
        print(f"      δ < 0 (attenuate to flip): {np.sum(neg_deltas)}")

        # Four-state transition analysis
        print(f"\n    State transitions at zeros:")
        for from_state in [1, 2, 3, 4]:
            mask = states0[zero_dims] == from_state
            if np.sum(mask) > 0:
                print(f"      From {state_names[from_state]}: {np.sum(mask)} zeros, "
                      f"median |δ|={np.median(abs_deltas[mask]):.4f}")

        # ════════════════════════════════════════════════════════════════
        # MULTI-GROUP ZERO HUNTING
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── MULTI-GROUP ZERO SPECTRUM ──")
        print(f"  Each ε-group has its own zero spectrum")

        for eps_val, eps_count in top_groups[:5]:
            c_k = group_contributions[eps_val]
            valid_k = np.abs(c_k) > 1e-10
            ratio_k = np.full_like(h0, np.nan)
            ratio_k[valid_k] = 1.0 - h0[valid_k] / c_k[valid_k]
            has_zero_k = valid_k & (ratio_k > 0)
            delta_k = np.full_like(h0, np.nan)
            delta_k[has_zero_k] = np.log(ratio_k[has_zero_k]) / np.log(PHI)

            n_z = np.sum(has_zero_k)
            if n_z > 0:
                abs_d = np.abs(delta_k[has_zero_k])
                n_ctrl = np.sum(abs_d < 0.1)
                print(f"    ε={eps_val:>3d} ({eps_count:>6d} elems): "
                      f"{n_z:>5d} zeros, {n_ctrl:>4d} controllable, "
                      f"median |δ|={np.median(abs_d):.4f}")
            else:
                print(f"    ε={eps_val:>3d} ({eps_count:>6d} elems): no real zeros")

        # ════════════════════════════════════════════════════════════════
        # PRECISION TARGETING: Flip exactly N dimensions
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── PRECISION TARGETING ──")
        print(f"  Choose δ to flip exactly N dimensions")

        # At δ, dimensions with |δ_critical| < |δ| flip
        target_flips = [1, 5, 10, 50, 100, 500]

        for n_flip in target_flips:
            if n_flip > len(abs_deltas):
                continue
            # δ needed to flip the n_flip-th easiest dimension
            d_needed = abs_deltas[n_flip - 1]
            # Count how many actually flip at this δ (might be more)
            actual_flips = np.sum(abs_deltas <= d_needed + 1e-10)

            print(f"    Target {n_flip:>4d} flips: δ = {d_needed:.6f} "
                  f"(φ^δ = {PHI**d_needed:.6f}), actually flips {actual_flips}")

        # ════════════════════════════════════════════════════════════════
        # CROSS-INPUT STABILITY: Same zeros for different inputs?
        # ════════════════════════════════════════════════════════════════

        print(f"\n  ── CROSS-INPUT STABILITY ──")
        print(f"  Do the same dimensions flip for different inputs?")

        all_zero_sets = []
        all_delta_arrays = []

        for inp_idx in range(N_INPUTS):
            x_i = rms_norm(inputs[inp_idx], norm_w)
            h_i = (x_i @ gate_W.T).flatten()
            c_i = (gate_W * (gate_eps == top_eps_val) @ x_i.flatten()).flatten()

            valid_i = np.abs(c_i) > 1e-10
            ratio_i = np.full_like(h_i, np.nan)
            ratio_i[valid_i] = 1.0 - h_i[valid_i] / c_i[valid_i]
            has_zero_i = valid_i & (ratio_i > 0)
            delta_i = np.full_like(h_i, np.nan)
            delta_i[has_zero_i] = np.log(ratio_i[has_zero_i]) / np.log(PHI)

            # Controllable zeros
            ctrl_mask = has_zero_i & (np.abs(delta_i) < 0.1)
            ctrl_dims = set(np.where(ctrl_mask)[0])
            all_zero_sets.append(ctrl_dims)
            all_delta_arrays.append(delta_i)

        # Pairwise overlap of controllable zero sets
        print(f"\n    Controllable zero sets (|δ|<0.1):")
        for i in range(N_INPUTS):
            print(f"      Input {i}: {len(all_zero_sets[i])} controllable zeros")

        # Intersection analysis
        if len(all_zero_sets) >= 2:
            common = all_zero_sets[0]
            for s in all_zero_sets[1:]:
                common = common & s
            union = all_zero_sets[0]
            for s in all_zero_sets[1:]:
                union = union | s

            print(f"\n    Intersection (all inputs): {len(common)} dims")
            print(f"    Union (any input):         {len(union)} dims")
            print(f"    Jaccard similarity:        {len(common)/max(1,len(union)):.3f}")

            if len(common) > 0:
                # For the common dims, how similar are their critical deltas?
                common_list = sorted(common)[:20]
                print(f"\n    Common zeros — δ_crit across inputs (first 20):")
                print(f"    {'Dim':>6s}", end='')
                for i in range(N_INPUTS):
                    print(f"  {'Inp'+str(i):>10s}", end='')
                print(f"  {'Std':>8s}")

                for j in common_list:
                    print(f"    {j:>6d}", end='')
                    vals = []
                    for i in range(N_INPUTS):
                        d = all_delta_arrays[i][j]
                        print(f"  {d:>10.6f}", end='')
                        vals.append(d)
                    print(f"  {np.std(vals):>8.6f}")

        # Cleanup
        del mlp, gate_W, gate_eps, norm_w
        gc.collect()

    print(f"\n{'='*70}")
    print(f"  SYNTHESIS")
    print(f"{'='*70}")
    print(f"""
  The rhzeros pipeline maps onto the phase shift control surface:

  COMPRESSOR (Lambert W):
    Gate activations h_j classify into four states (+1, +0, -0, -1)
    Near-zero dims are the flippable targets

  PROCESSOR (Ramanujan):
    Each ε-group k has contribution c_j^k to each gate dim j
    Sensitivity: dh_j/dδ ∝ c_j^k · ln(φ)
    Critical δ: log_φ(1 - h_j / c_j^k)

  TARGETER (Z(t) + Newton):
    Zero spectrum: sorted list of (δ_j, dim_j) pairs
    Choose δ → control exactly which dims flip
    Verified: predicted zeros actually flip the gate

  The zero spectrum IS the control interface to the funnel.
  Each ε-group provides an independent set of zeros.
  Precision targeting: choose δ to flip exactly N dimensions.
""")


if __name__ == '__main__':
    run()
