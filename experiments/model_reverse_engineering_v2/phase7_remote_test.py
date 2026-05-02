"""
Phase 7a: Remote Integer Compute — End-to-End Test

Proves that phi_matmul_integer computed on gimli (192.168.1.111)
produces IDENTICAL results to local computation.

Usage:
    1. On gimli: python ~/truthspace-node/server.py
    2. On dev:   python experiments/model_reverse_engineering_v2/phase7_remote_test.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference.phi_remote import PhiRemoteClient, WID_Q, WID_K, WID_V, WID_O, WID_GATE, WID_UP, WID_DOWN
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, phi_matmul_integer, float_to_phi, phi_to_float
)
from phi_geometric.inference.phi_types import PhiEncoded

GIMLI_HOST = '192.168.1.111'
GIMLI_PORT = 7618
MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def test_ping(client):
    """Test basic connectivity."""
    print("1. PING test...")
    ok = client.ping()
    print(f"   → {'OK' if ok else 'FAILED'}")
    return ok


def test_matmul_match(client, layer_idx=0):
    """Compare remote matmul vs local matmul for all weight matrices."""
    print(f"\n2. MATMUL match test (layer {layer_idx})...")

    # Init local LUT
    get_fixed_lut()

    # Load local weights
    import os
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')

    weight_tests = [
        (WID_Q, 'q_proj', 3584),
        (WID_K, 'k_proj', 3584),
        (WID_V, 'v_proj', 3584),
        (WID_O, 'o_proj', 3584),
        (WID_GATE, 'gate_proj', 3584),
        (WID_UP, 'up_proj', 3584),
        (WID_DOWN, 'down_proj', 18944),
    ]

    # Create test input (5 tokens × hidden_dim)
    np.random.seed(42)
    x_float = np.random.randn(5, 3584).astype(np.float32) * 10.0
    x_signs, x_exps = float_to_phi(x_float)

    all_match = True
    for wid, name, in_dim in weight_tests:
        # Adjust input for down_proj (in_dim=18944)
        if in_dim != 3584:
            x_f = np.random.randn(5, in_dim).astype(np.float32) * 10.0
            xs, xe = float_to_phi(x_f)
        else:
            xs, xe = x_signs, x_exps

        # Local computation
        W = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        t0 = time.time()
        local_s, local_e = phi_matmul_integer(W, xs, xe)
        local_dt = time.time() - t0

        # Remote computation
        t0 = time.time()
        remote_s, remote_e = client.matmul(layer_idx, wid, xs, xe)
        remote_dt = time.time() - t0

        # Compare
        sign_match = np.array_equal(local_s, remote_s)
        exp_match = np.array_equal(local_e, remote_e)
        match = sign_match and exp_match

        # Correlation check (decode both and compare)
        local_float = phi_to_float(local_s, local_e).flatten()
        remote_float = phi_to_float(remote_s, remote_e).flatten()
        corr = float(np.corrcoef(local_float, remote_float)[0, 1])

        status = "EXACT MATCH" if match else f"DIFF (corr={corr:.6f})"
        print(f"   {name:10s}: {status}  local={local_dt*1000:.0f}ms  remote={remote_dt*1000:.0f}ms  "
              f"(network={max(0, remote_dt-local_dt)*1000:.0f}ms)")

        if not match:
            # Show how many differ
            sign_diffs = np.sum(local_s != remote_s)
            exp_diffs = np.sum(local_e != remote_e)
            print(f"              sign diffs: {sign_diffs}/{local_s.size}  "
                  f"exp diffs: {exp_diffs}/{local_e.size}")
            all_match = False

    return all_match


def test_prediction_match(client):
    """Run a simple prompt through both local and remote matmul, compare logits."""
    print(f"\n3. Prediction test...")
    print("   (Full integration test deferred to Phase 7b)")


def main():
    print("=" * 70)
    print("  Phase 7a: Remote Integer Compute — Verification")
    print(f"  Target: {GIMLI_HOST}:{GIMLI_PORT}")
    print("=" * 70)

    client = PhiRemoteClient(GIMLI_HOST, GIMLI_PORT)

    try:
        print("\nConnecting...")
        client.connect()
        print("Connected!\n")

        # Test 1: Ping
        if not test_ping(client):
            print("PING failed — aborting")
            return

        # Test 2: Matmul exact match
        match = test_matmul_match(client, layer_idx=0)

        print("\n" + "=" * 70)
        if match:
            print("  RESULT: ALL OPERATIONS PRODUCE EXACT MATCH")
            print("  Integer geometry travels over the network and computes correctly.")
            print("  The φ-lattice is substrate-independent.")
        else:
            print("  RESULT: DIFFERENCES DETECTED (see above)")
            print("  Investigate: endianness? numpy version? LUT construction?")
        print("=" * 70)

    except ConnectionRefusedError:
        print(f"\nCannot connect to {GIMLI_HOST}:{GIMLI_PORT}")
        print("Make sure the server is running on gimli:")
        print(f"  ssh gimli@{GIMLI_HOST}")
        print("  source ~/truthspace-node/.venv/bin/activate")
        print("  python ~/truthspace-node/server.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == '__main__':
    main()
