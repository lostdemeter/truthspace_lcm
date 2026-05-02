"""
Profile the φ-compute thin client pipeline.

Measures:
  1. Per-opcode server-side timing (via instrumented PROGRAM handler)
  2. Network overhead (serialization, transfer, deserialization)
  3. Memory usage (RAM for blobs, per-layer footprint)
  4. Per-layer timing breakdown across all 28 layers
  5. GPU vs CPU split (MATMUL is GPU, rest is CPU)

Usage:
    # On gimli: python phi_compute_node.py --port 7619 --gpu
    # On dev:   python profile_phi_compute.py [--shutdown]
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_integer import (
    float_to_phi, phi_to_float, phi_rms_norm_int,
    get_fixed_lut, get_silu_lut, get_softmax_lut,
)
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_compute_client import PhiComputeClient, Program

from test_phi_compute_layer import (
    compile_layer, blob_ids, load_and_store_layer, store_rope_tables,
    HIDDEN_DIM,
)

GIMLI_HOST = '192.168.1.111'
GIMLI_PORT = 7619
MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def profile_network_overhead(client):
    """Measure serialization + round-trip overhead for programs."""
    print("\n" + "=" * 70)
    print("  PROFILE 1: Network Overhead")
    print("=" * 70)

    # Generate test input
    seq_len = 5
    batch = 1
    x_s = np.random.choice([-1, 1], size=(batch, seq_len, HIDDEN_DIM)).astype(np.int8)
    x_e = np.random.randint(0, 20, size=(batch, seq_len, HIDDEN_DIM)).astype(np.int16)

    bids = blob_ids(0)
    program = compile_layer(bids, batch, seq_len)

    # Measure serialization time
    t0 = time.perf_counter()
    for _ in range(10):
        payload = program.encode()
    serialize_dt = (time.perf_counter() - t0) / 10

    payload_bytes = len(payload)
    input_bytes = x_s.nbytes + x_e.nbytes

    # Measure round-trip with actual execution
    timings = []
    for _ in range(3):
        program = compile_layer(bids, batch, seq_len)
        program.add_input(0, x_s, x_e)

        t0 = time.perf_counter()
        results = client.run_program(program)
        dt = time.perf_counter() - t0
        timings.append(dt)

    avg_rt = np.mean(timings)
    output_bytes = results[0][0].nbytes + results[0][1].nbytes

    print(f"  Program payload:     {payload_bytes:,} bytes ({payload_bytes/1024:.1f} KB)")
    print(f"  Input data:          {input_bytes:,} bytes ({input_bytes/1024:.1f} KB)")
    print(f"  Output data:         {output_bytes:,} bytes ({output_bytes/1024:.1f} KB)")
    print(f"  Total wire bytes:    {payload_bytes + input_bytes + output_bytes:,}")
    print(f"  Serialization:       {serialize_dt*1000:.2f} ms")
    print(f"  Round-trip (avg 3):  {avg_rt*1000:.1f} ms")
    print(f"  Network fraction:    estimated from wire bytes @ ~300 MB/s → "
          f"{(payload_bytes + input_bytes + output_bytes) / 3e8 * 1000:.1f} ms")

    return {
        'payload_bytes': payload_bytes,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'serialize_ms': serialize_dt * 1000,
        'roundtrip_ms': avg_rt * 1000,
    }


def profile_memory(client):
    """Measure blob storage footprint."""
    print("\n" + "=" * 70)
    print("  PROFILE 2: Memory Usage")
    print("=" * 70)

    blobs = client.list_blobs()
    total_bytes = sum(b[2] for b in blobs)

    # Group by layer
    layer_bytes = {}
    rope_bytes = 0
    for bid, shape, nbytes in blobs:
        layer = bid // 100
        if layer >= 100:  # RoPE tables
            rope_bytes += nbytes
        else:
            if layer not in layer_bytes:
                layer_bytes[layer] = 0
            layer_bytes[layer] += nbytes

    per_layer = np.mean(list(layer_bytes.values())) if layer_bytes else 0

    print(f"  Total blobs:         {len(blobs)}")
    print(f"  Total storage:       {total_bytes / 1e9:.2f} GB")
    print(f"  RoPE tables:         {rope_bytes / 1e6:.1f} MB")
    print(f"  Per-layer (avg):     {per_layer / 1e6:.1f} MB")
    print(f"  Layers loaded:       {len(layer_bytes)}")

    # Show per-weight breakdown for layer 0
    print(f"\n  Layer 0 blob breakdown:")
    weight_names = {
        1: 'W_q', 2: 'W_k', 3: 'W_v', 4: 'W_o',
        5: 'W_gate', 6: 'W_up', 7: 'W_down',
        8: 'attn_norm', 9: 'mlp_norm',
        10: 'b_q', 11: 'b_k', 12: 'b_v',
    }
    for bid, shape, nbytes in blobs:
        if bid // 100 == 0:
            wid = bid % 100
            name = weight_names.get(wid, f'id={wid}')
            print(f"    {name:12s} {str(shape):20s} {nbytes/1e6:8.2f} MB")

    return {
        'total_blobs': len(blobs),
        'total_gb': total_bytes / 1e9,
        'rope_mb': rope_bytes / 1e6,
        'per_layer_mb': per_layer / 1e6,
    }


def profile_per_layer(client, engine, tokenizer):
    """Profile each layer individually for one prompt."""
    print("\n" + "=" * 70)
    print("  PROFILE 3: Per-Layer Timing")
    print("=" * 70)

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    seq_len = len(tokens)
    batch = 1

    # Embedding
    hidden_float = engine.embedding(tokens)
    h_s, h_e = float_to_phi(hidden_float)
    h_s = h_s[np.newaxis, :, :]
    h_e = h_e[np.newaxis, :, :]

    layer_times = []
    total_t0 = time.perf_counter()

    for layer_idx in range(28):
        bids = blob_ids(layer_idx)
        program = compile_layer(bids, batch, seq_len)
        program.add_input(0, h_s, h_e)

        t0 = time.perf_counter()
        results = client.run_program(program)
        dt = time.perf_counter() - t0

        h_s, h_e = results[0]
        layer_times.append(dt)

    total_dt = time.perf_counter() - total_t0

    # Final norm + LM head
    t0 = time.perf_counter()
    fnw_s, fnw_e = float_to_phi(engine.final_norm_weight)
    h_s, h_e = phi_rms_norm_int(h_s, h_e, fnw_s, fnw_e, engine.hidden_dim)
    h_float = phi_to_float(h_s, h_e)
    from phi_geometric.inference.phi_matmul import phi_linear
    logits = phi_linear(engine.lm_head.weight,
                        h_float.reshape(1, seq_len, -1))
    head_dt = time.perf_counter() - t0

    top_id = int(np.argmax(logits[0, -1, :]))
    top_tok = tokenizer.decode_token(top_id)

    # Summary
    times = np.array(layer_times) * 1000
    print(f"  Prompt: '{prompt}' → '{top_tok.strip()}'")
    print(f"  Tokens: {seq_len}")
    print(f"\n  Per-layer times (ms):")
    for i, t in enumerate(layer_times):
        bar = '█' * int(t * 1000 / 50)
        print(f"    Layer {i:2d}: {t*1000:7.1f} ms  {bar}")

    print(f"\n  Layer stats:")
    print(f"    Min:     {times.min():.1f} ms (layer {times.argmin()})")
    print(f"    Max:     {times.max():.1f} ms (layer {times.argmax()})")
    print(f"    Mean:    {times.mean():.1f} ms")
    print(f"    Std:     {times.std():.1f} ms")
    print(f"    Total:   {total_dt:.1f}s (28 layers)")
    print(f"    LM head: {head_dt*1000:.0f} ms (local)")
    print(f"    End-to-end: {total_dt + head_dt:.1f}s")

    return {
        'prompt': prompt,
        'prediction': top_tok.strip(),
        'seq_len': seq_len,
        'layer_times_ms': [round(t * 1000, 1) for t in layer_times],
        'layer_min_ms': round(float(times.min()), 1),
        'layer_max_ms': round(float(times.max()), 1),
        'layer_mean_ms': round(float(times.mean()), 1),
        'layer_std_ms': round(float(times.std()), 1),
        'total_layers_s': round(total_dt, 1),
        'lm_head_ms': round(head_dt * 1000, 0),
    }


def profile_seq_scaling(client, engine, tokenizer):
    """Measure how inference time scales with sequence length."""
    print("\n" + "=" * 70)
    print("  PROFILE 4: Sequence Length Scaling")
    print("=" * 70)

    # Different prompts of varying length
    prompts = [
        "Hi",                                          # ~2 tokens
        "The capital of France is",                    # ~5 tokens
        "The quick brown fox jumps over the lazy dog", # ~9 tokens
        "In a galaxy far far away there lived a young wizard who discovered "
        "that the ancient prophecy was actually about",  # ~20 tokens
    ]

    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        seq_len = len(tokens)
        batch = 1

        hidden_float = engine.embedding(tokens)
        h_s, h_e = float_to_phi(hidden_float)
        h_s = h_s[np.newaxis, :, :]
        h_e = h_e[np.newaxis, :, :]

        # Time just 3 layers (enough to see scaling pattern)
        layer_times = []
        for layer_idx in range(3):
            bids = blob_ids(layer_idx)
            program = compile_layer(bids, batch, seq_len)
            program.add_input(0, h_s, h_e)

            t0 = time.perf_counter()
            res = client.run_program(program)
            dt = time.perf_counter() - t0

            h_s, h_e = res[0]
            layer_times.append(dt)

        avg_ms = np.mean(layer_times) * 1000
        results.append((seq_len, avg_ms))
        print(f"  seq_len={seq_len:3d}  avg_layer={avg_ms:.0f}ms  "
              f"(projected 28 layers: {avg_ms * 28 / 1000:.1f}s)")

    # Check scaling
    if len(results) >= 2:
        lens = [r[0] for r in results]
        times = [r[1] for r in results]
        # Rough O(n) vs O(n^2) check
        ratio_len = lens[-1] / lens[0]
        ratio_time = times[-1] / times[0]
        print(f"\n  Scaling analysis:")
        print(f"    Sequence ratio: {ratio_len:.1f}×")
        print(f"    Time ratio:     {ratio_time:.1f}×")
        if ratio_time < ratio_len * 1.5:
            print(f"    → Approximately O(n) — dominated by matmuls")
        elif ratio_time < ratio_len ** 2 * 0.8:
            print(f"    → Between O(n) and O(n²)")
        else:
            print(f"    → Approximately O(n²) — dominated by attention")

    return {
        'seq_lengths': [r[0] for r in results],
        'avg_layer_ms': [round(r[1], 1) for r in results],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Profile φ-compute pipeline')
    parser.add_argument('--host', default=GIMLI_HOST)
    parser.add_argument('--port', type=int, default=GIMLI_PORT)
    parser.add_argument('--shutdown', action='store_true',
                        help='Shutdown the node after profiling')
    args = parser.parse_args()

    print("=" * 70)
    print("  φ-Compute Pipeline Profile")
    print("=" * 70)

    # Init
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()

    # Connect
    client = PhiComputeClient(args.host, args.port)
    client.connect()
    info = client.ping()
    print(f"  Connected: {info}")

    # Check if weights are loaded
    blobs = client.list_blobs()
    if len(blobs) < 338:
        print(f"\n  Only {len(blobs)} blobs loaded — loading all 28 layers...")
        rope = store_rope_tables(client)
        for layer_idx in range(28):
            t0 = time.time()
            load_and_store_layer(client, layer_idx)
            print(f"    Layer {layer_idx:2d} ({time.time()-t0:.1f}s)", end='')
            if (layer_idx + 1) % 7 == 0:
                print()
            else:
                print("  ", end='')
        print()

    # Run profiles
    results = {}
    results['network'] = profile_network_overhead(client)
    results['memory'] = profile_memory(client)
    results['per_layer'] = profile_per_layer(client, engine, tokenizer)
    results['scaling'] = profile_seq_scaling(client, engine, tokenizer)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'profile_phi_compute.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    net = results['network']
    mem = results['memory']
    pl = results['per_layer']
    print(f"  Memory:      {mem['total_gb']:.2f} GB across {mem['total_blobs']} blobs")
    print(f"  Per layer:   {pl['layer_mean_ms']:.0f} ms avg "
          f"({pl['layer_min_ms']:.0f}-{pl['layer_max_ms']:.0f} ms range)")
    print(f"  28 layers:   {pl['total_layers_s']}s")
    print(f"  LM head:     {pl['lm_head_ms']:.0f} ms (local)")
    print(f"  Wire data:   {net['payload_bytes'] + net['input_bytes']:.0f} bytes/program")
    print(f"  Serialize:   {net['serialize_ms']:.2f} ms")
    print(f"  (Server-side per-opcode breakdown printed on gimli terminal)")
    print("=" * 70)

    if args.shutdown:
        print("\n  Sending SHUTDOWN to node...")
        client.shutdown()
        print("  Node stopped.")
    else:
        client.close()
        print(f"\n  Node still running. Use --shutdown to stop it.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
