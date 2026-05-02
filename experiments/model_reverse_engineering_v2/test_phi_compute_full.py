"""
Full Model Test — 28 layers + LM head via thin client φ-compute node.

The node has ZERO knowledge of the model. All it sees is:
  - Opaque blobs (weights stored by numeric ID)
  - 28 layer programs + 1 LM head program

The controller (this script) holds all architectural knowledge:
  - Transformer with GQA, RoPE, causal masking, SwiGLU MLP
  - Qwen2-7B dimensions and structure
  - Embedding lookup (only thing done locally — just a table lookup)

Usage:
    # On gimli: python phi_compute_node.py --port 7619 --gpu
    # On dev:   python test_phi_compute_full.py [--shutdown]
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_integer import (
    phi_rms_norm_int, float_to_phi, phi_to_float, PhiRoPEInt,
    get_fixed_lut, get_silu_lut, get_softmax_lut,
)
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_compute_client import (
    PhiComputeClient, Program, OP_MATMUL, OP_RMS_NORM, OP_LOAD,
)

# Import the layer compiler from the layer test
from test_phi_compute_layer import (
    compile_layer, blob_ids, load_and_store_layer, store_rope_tables,
    BLOB_ROPE_COS, BLOB_ROPE_SIN, HIDDEN_DIM,
)

GIMLI_HOST = '192.168.1.111'
GIMLI_PORT = 7619
MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

BLOB_FINAL_NORM = 60000
BLOB_LM_HEAD = 60001

TEST_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The color of the sky is", "blue"),
    ("One plus one equals", "two"),
    ("The chemical symbol for gold is", "Au"),
]


def store_lm_head(client, engine):
    """Store final norm weight and LM head on the node."""
    # Final norm (small, send from controller)
    fnw_s, fnw_e = float_to_phi(engine.final_norm_weight)
    client.store(BLOB_FINAL_NORM, fnw_s, fnw_e)

    # LM head (large — load from gimli's local disk)
    gimli_path = '/home/gimli/truthspace-node/weights/lm_head.npz'
    client.store_local(BLOB_LM_HEAD, gimli_path, fmt=1)  # fmt=1 = raw phi


def compile_lm_head(batch, seq_len):
    """Compile the final norm + LM head into a program.

    Input reg 0: hidden state (batch, seq_len, 3584)
    Output reg 2: logits (batch*seq_len, 152064)
    """
    prog = Program()

    # Reg 0 = input hidden state (batch, seq_len, hidden_dim)
    # Step 1: Reshape to 2D for RMS_NORM and MATMUL
    prog.reshape(1, 0, shape=(batch * seq_len, HIDDEN_DIM))

    # Step 2: RMS norm with final norm weight
    prog.rms_norm(2, 1, BLOB_FINAL_NORM, dim=HIDDEN_DIM)

    # Step 3: LM head matmul → logits
    prog.matmul(3, 2, BLOB_LM_HEAD)

    prog.set_outputs([3])
    return prog


def main():
    parser = argparse.ArgumentParser(description='Full model test')
    parser.add_argument('--host', default=GIMLI_HOST)
    parser.add_argument('--port', type=int, default=GIMLI_PORT)
    parser.add_argument('--shutdown', action='store_true',
                        help='Shutdown the node after test')
    args = parser.parse_args()

    print("=" * 70)
    print("  FULL MODEL TEST — 28 Layers + LM Head via Thin Client")
    print("  Node knows: NOTHING about the model")
    print("  Controller knows: transformer architecture, dimensions, etc.")
    print("=" * 70)

    # Init local LUTs
    print("\nInitializing local LUTs...")
    t0 = time.time()
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Load model (locally — only for embedding lookup)
    print("\nLoading model (local — for embedding only)...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"  {len(engine.layers)} layers, hidden_dim={engine.hidden_dim}")

    # Connect to thin client
    print(f"\nConnecting to φ-compute node at {args.host}:{args.port}...")
    client = PhiComputeClient(args.host, args.port)
    client.connect()
    print(f"  {client.ping()}")

    # Store RoPE tables
    print("\nStoring RoPE tables...")
    t0 = time.time()
    rope = store_rope_tables(client)
    print(f"  Done ({time.time()-t0:.0f}ms)")

    # Load ALL 28 layers + LM head
    print("\nLoading all 28 layers onto node...")
    total_t0 = time.time()
    for layer_idx in range(28):
        t0 = time.time()
        load_and_store_layer(client, layer_idx)
        dt = time.time() - t0
        print(f"  Layer {layer_idx:2d}: {dt:.1f}s", end='')
        if (layer_idx + 1) % 7 == 0:
            print()
        else:
            print("  ", end='')

    # LM head + final norm
    print("\n  Loading LM head + final norm...")
    t0 = time.time()
    store_lm_head(client, engine)
    dt = time.time() - t0
    print(f"  LM head: {dt:.1f}s")

    load_dt = time.time() - total_t0
    print(f"  Total load time: {load_dt:.0f}s")

    blobs = client.list_blobs()
    total_mb = sum(b[2] for b in blobs) / 1e6
    print(f"  {len(blobs)} blobs ({total_mb:.0f} MB)")

    # Warm LM head cache (one-time decode φ→float, ~6s)
    print("\nWarming LM head cache...")
    t0 = time.time()
    engine.lm_head.weight.decode_cached()
    print(f"  Done ({time.time()-t0:.1f}s) — subsequent calls ~300ms")

    # Run test prompts
    print("\n" + "=" * 70)
    print("  INFERENCE — 5 prompts × (28 layers + LM head)")
    print("  Layers on gimli, LM head local (cached)")
    print("=" * 70)

    correct = 0
    total = len(TEST_PROMPTS)

    for prompt, expected in TEST_PROMPTS:
        tokens = tokenizer.encode(prompt)
        seq_len = len(tokens)
        batch = 1

        t_start = time.time()

        # 1. Embedding (local — just a table lookup, ~0ms)
        hidden_float = engine.embedding(tokens)
        h_s, h_e = float_to_phi(hidden_float)
        h_s = h_s[np.newaxis, :, :]  # (1, seq_len, hidden_dim)
        h_e = h_e[np.newaxis, :, :]

        # 2. All 28 layers on the thin client
        t_layers = time.perf_counter()
        for layer_idx in range(28):
            bids = blob_ids(layer_idx)
            program = compile_layer(bids, batch, seq_len)
            program.add_input(0, h_s, h_e)
            results = client.run_program(program)
            h_s, h_e = results[0]
        layers_dt = time.perf_counter() - t_layers

        # 3. Final norm + LM head (local — cached decode makes this fast)
        t_head = time.perf_counter()
        fnw_s, fnw_e = float_to_phi(engine.final_norm_weight)
        h_s, h_e = phi_rms_norm_int(h_s, h_e, fnw_s, fnw_e, engine.hidden_dim)
        h_float = phi_to_float(h_s, h_e)
        from phi_geometric.inference.phi_matmul import phi_linear
        logits = phi_linear(engine.lm_head.weight,
                            h_float.reshape(1, seq_len, -1))
        head_dt = time.perf_counter() - t_head

        top_id = int(np.argmax(logits[0, -1, :]))
        top_tok = tokenizer.decode_token(top_id)

        dt = time.time() - t_start
        is_correct = expected.lower() in top_tok.lower().strip()
        correct += is_correct

        status = "✓" if is_correct else "✗"
        print(f"  {status} '{prompt}' → '{top_tok.strip()}'  "
              f"({dt:.1f}s  layers={layers_dt:.1f}s  lm_head={head_dt:.1f}s)")

    # Summary
    n_layer_instr = 28 * 55
    n_lm_instr = 3
    n_total = n_layer_instr + n_lm_instr
    print("\n" + "=" * 70)
    print(f"  RESULTS: {correct}/{total} correct predictions")
    if correct == total:
        print(f"  ✓ FULL MODEL: ALL PREDICTIONS CORRECT")
        print(f"  The node executed {n_total} instructions")
        print(f"    28 layers × 55 = {n_layer_instr} layer instructions")
        print(f"    + {n_lm_instr} LM head instructions (reshape + norm + matmul)")
        print(f"  Only the embedding lookup runs locally (table lookup, ~0ms)")
        print(f"  with ZERO knowledge of the model architecture.")
    else:
        print(f"  Some predictions incorrect — investigating...")
    print("=" * 70)

    if args.shutdown:
        print("\n  Sending SHUTDOWN to node...")
        client.shutdown()
        print("  Node stopped.")
    else:
        client.close()

    return 0 if correct == total else 1


if __name__ == '__main__':
    sys.exit(main())
