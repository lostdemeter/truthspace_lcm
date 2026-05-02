"""
Phase 7b: Full Layer Remote — End-to-End Test

Proves that an ENTIRE transformer layer computed on gimli produces
IDENTICAL results to local computation. Then runs a full prompt
with all 28 layers on gimli.

Usage:
    1. On gimli: python ~/truthspace-node/server.py
    2. On dev:   python experiments/model_reverse_engineering_v2/phase7b_full_layer_test.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference.phi_remote import PhiRemoteClient
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, get_silu_lut, get_softmax_lut,
    PhiRoPEInt, float_to_phi, phi_to_float, phi_rms_norm_int,
)
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

sys.path.insert(0, 'experiments/model_reverse_engineering_v2')
from phase6_integer_forward_pass import integer_forward_layer

GIMLI_HOST = '192.168.1.111'
GIMLI_PORT = 7618
MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def test_single_layer_match(client, engine, rope_int, tokens, layer_idx=0):
    """Compare remote full-layer vs local full-layer for one layer."""
    print(f"\n  Layer {layer_idx}: local vs remote...")

    # Create input hidden state
    hidden_float = engine.embedding(tokens)
    h_s, h_e = float_to_phi(hidden_float)
    h_s = h_s[np.newaxis, :, :]
    h_e = h_e[np.newaxis, :, :]

    # Run preceding layers locally (both paths start from same state)
    for i in range(layer_idx):
        h_s, h_e = integer_forward_layer(
            engine.layers[i], h_s, h_e, rope_int, i)

    # Local: run target layer
    t0 = time.time()
    local_s, local_e = integer_forward_layer(
        engine.layers[layer_idx], h_s.copy(), h_e.copy(), rope_int, layer_idx)
    local_dt = time.time() - t0

    # Remote: run target layer on gimli
    t0 = time.time()
    remote_s, remote_e = client.full_layer(layer_idx, h_s, h_e)
    remote_dt = time.time() - t0

    # Compare
    sign_match = np.array_equal(local_s, remote_s)
    exp_match = np.array_equal(local_e, remote_e)
    exact = sign_match and exp_match

    local_f = phi_to_float(local_s, local_e).flatten()
    remote_f = phi_to_float(remote_s, remote_e).flatten()
    corr = float(np.corrcoef(local_f, remote_f)[0, 1])

    status = "EXACT MATCH" if exact else f"DIFF (corr={corr:.6f})"
    print(f"    {status}  local={local_dt:.1f}s  remote={remote_dt:.1f}s")

    if not exact:
        sign_diffs = np.sum(local_s != remote_s)
        exp_diffs = np.sum(local_e != remote_e)
        print(f"    sign diffs: {sign_diffs}/{local_s.size}  exp diffs: {exp_diffs}/{local_e.size}")

    return exact


def test_full_forward_remote(client, engine, rope_int, tokenizer):
    """Run a full prompt with ALL 28 layers computed on gimli."""
    print("\n" + "=" * 70)
    print("  FULL FORWARD PASS — ALL 28 LAYERS ON GIMLI")
    print("=" * 70)

    prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("The color of the sky is", "blue"),
        ("One plus one equals", "two"),
        ("The chemical symbol for gold is", "Au"),
    ]

    correct = 0
    match = 0

    for prompt, expected in prompts:
        tokens = tokenizer.encode(prompt)
        t0 = time.time()

        # 1. Embedding (local — just a lookup)
        hidden_float = engine.embedding(tokens)
        h_s, h_e = float_to_phi(hidden_float)
        h_s = h_s[np.newaxis, :, :]
        h_e = h_e[np.newaxis, :, :]

        # 2. All 28 layers on gimli
        for layer_idx in range(28):
            h_s, h_e = client.full_layer(layer_idx, h_s, h_e)

        # 3. Final norm (local — tiny operation)
        fnw_s, fnw_e = float_to_phi(engine.final_norm_weight)
        h_s, h_e = phi_rms_norm_int(h_s, h_e, fnw_s, fnw_e, engine.hidden_dim)

        # 4. LM head (local — decode to float for argmax)
        h_float = phi_to_float(h_s, h_e)
        from phi_geometric.inference.phi_matmul import phi_linear
        logits = phi_linear(engine.lm_head.weight,
                            h_float.reshape(1, len(tokens), -1))

        top_id = int(np.argmax(logits[0, -1, :]))
        top_tok = tokenizer.decode_token(top_id)

        # Also get float baseline
        float_logits = engine.forward(tokens, pure=False)
        float_top_id = int(np.argmax(float_logits[0, -1, :]))
        float_tok = tokenizer.decode_token(float_top_id)

        dt = time.time() - t0
        tok_match = (top_id == float_top_id)
        is_correct = expected.lower() in top_tok.lower().strip()

        match += tok_match
        correct += is_correct

        m_str = "MATCH" if tok_match else "DIFF"
        c_str = "✓" if is_correct else "✗"
        print(f"  {c_str} [{m_str}] '{prompt}' → '{top_tok.strip()}'  "
              f"(float:'{float_tok.strip()}')  {dt:.1f}s")

    print(f"\n  Token match with float: {match}/{len(prompts)}")
    print(f"  Correct predictions:    {correct}/{len(prompts)}")
    return match, len(prompts)


def main():
    print("=" * 70)
    print("  Phase 7b: Full Layer Remote — Verification")
    print(f"  Target: {GIMLI_HOST}:{GIMLI_PORT}")
    print("=" * 70)

    # Init
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    rope_int = PhiRoPEInt(head_dim=128, rope_theta=1_000_000.0)
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()

    tokens = tokenizer.encode("The capital of France is")

    client = PhiRemoteClient(GIMLI_HOST, GIMLI_PORT)

    try:
        print("\nConnecting...")
        client.connect()
        print("Connected!")

        # Test 1: Single layer exact match (layer 0)
        print("\n" + "=" * 70)
        print("  TEST 1: Single Layer Exact Match")
        print("=" * 70)

        layers_to_test = [0, 1, 13, 27]
        all_match = True
        for l in layers_to_test:
            if not test_single_layer_match(client, engine, rope_int, tokens, l):
                all_match = False

        if all_match:
            print(f"\n  ✓ All {len(layers_to_test)} layers: EXACT MATCH")
        else:
            print(f"\n  Some layers differ — investigating...")

        # Test 2: Full forward pass with all layers on gimli
        test_full_forward_remote(client, engine, rope_int, tokenizer)

        print("\n" + "=" * 70)
        print("  PHASE 7b COMPLETE")
        print("=" * 70)

    except ConnectionRefusedError:
        print(f"\nCannot connect to {GIMLI_HOST}:{GIMLI_PORT}")
        print("Start the server on gimli first:")
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
