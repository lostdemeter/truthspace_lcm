"""
Test a full transformer layer compiled as a φ-compute PROGRAM.

The controller (this script) has ALL model knowledge:
  - Architecture: transformer with GQA
  - Dimensions: 3584 hidden, 28 heads, 4 KV heads, 128 head_dim
  - Layer structure: RMS norm → attn → residual → RMS norm → MLP → residual

The node has NONE of this. It just executes instructions on blobs.

Usage:
    # Restart thin client on gimli with new opcodes:
    #   python phi_compute_node.py --port 7619 --gpu
    # Then:
    python test_phi_compute_layer.py [--host 192.168.1.111] [--port 7619] [--layer 0]
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_integer import (
    phi_matmul_integer, phi_add_encoded, phi_multiply_int,
    phi_rms_norm_int, phi_silu_int, phi_softmax_full_int,
    phi_einsum_qk_int, phi_einsum_av_int, phi_scale_int,
    PhiRoPEInt, PhiEncoded, float_to_phi,
)
from phi_geometric.inference.phi_compute_client import (
    PhiComputeClient, Program,
)

# ---------------------------------------------------------------------------
# Model constants (controller knowledge — node knows NONE of this)
# ---------------------------------------------------------------------------
HIDDEN_DIM = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 7
INTERMEDIATE_DIM = 18944
ROPE_THETA = 1_000_000.0
ATTN_SCALE_EXP = int(round(128 * np.log(1.0 / np.sqrt(HEAD_DIM)) / np.log((1 + np.sqrt(5)) / 2)))
MASK_EXP = 4000

WEIGHTS_DIR = '/home/thorin/gimli/truthspace-node/weights'  # sshfs mount


# ---------------------------------------------------------------------------
# Blob ID scheme (controller assigns IDs — node doesn't care about naming)
# ---------------------------------------------------------------------------
def blob_ids(layer_idx):
    """Return blob ID map for a given layer."""
    base = layer_idx * 20
    return {
        'W_q': base + 1, 'W_k': base + 2, 'W_v': base + 3, 'W_o': base + 4,
        'W_gate': base + 5, 'W_up': base + 6, 'W_down': base + 7,
        'norm_input': base + 10, 'norm_post': base + 11,
        'bias_q': base + 12, 'bias_k': base + 13, 'bias_v': base + 14,
    }

BLOB_ROPE_COS = 50000
BLOB_ROPE_SIN = 50001


# ---------------------------------------------------------------------------
# Weight loading (controller reads files, sends to node)
# ---------------------------------------------------------------------------
def load_and_store_layer(client, layer_idx):
    """Load layer weights from disk and store as blobs on the node."""
    bids = blob_ids(layer_idx)
    layer_dir = os.path.join(WEIGHTS_DIR, f'layer_{layer_idx:02d}')

    # Weight projections (compressed format)
    weight_files = {
        'W_q': 'q_proj.npz', 'W_k': 'k_proj.npz', 'W_v': 'v_proj.npz',
        'W_o': 'o_proj.npz', 'W_gate': 'gate_proj.npz',
        'W_up': 'up_proj.npz', 'W_down': 'down_proj.npz',
    }
    for name, fname in weight_files.items():
        path = os.path.join(layer_dir, fname)
        # Use STORE_LOCAL — node loads from its own disk
        # Path on gimli: ~/truthspace-node/weights/layer_XX/name.npz
        gimli_path = f'/home/gimli/truthspace-node/weights/layer_{layer_idx:02d}/{fname}'
        client.store_local(bids[name], gimli_path, fmt=0)

    # Norms (float → φ-encode on controller side)
    norms = np.load(os.path.join(layer_dir, 'norms.npz'))
    norm_in_s, norm_in_e = float_to_phi(norms['input_layernorm'])
    norm_post_s, norm_post_e = float_to_phi(norms['post_attention_layernorm'])
    client.store(bids['norm_input'], norm_in_s, norm_in_e)
    client.store(bids['norm_post'], norm_post_s, norm_post_e)

    # Biases (float → φ-encode on controller side)
    biases = np.load(os.path.join(layer_dir, 'biases.npz'))
    for key, file_key in [('bias_q', 'q_proj_bias'), ('bias_k', 'k_proj_bias'),
                           ('bias_v', 'v_proj_bias')]:
        b_s, b_e = float_to_phi(biases[file_key])
        client.store(bids[key], b_s, b_e)

    return bids


def store_rope_tables(client):
    """Precompute and store RoPE cos/sin as blobs."""
    rope = PhiRoPEInt(HEAD_DIM, ROPE_THETA, max_seq_len=4096)
    client.store(BLOB_ROPE_COS, rope.cos_signs, rope.cos_exps)
    client.store(BLOB_ROPE_SIN, rope.sin_signs, rope.sin_exps)
    return rope


# ---------------------------------------------------------------------------
# Layer compiler — THE key piece
# ---------------------------------------------------------------------------
def compile_layer(bids, batch, seq_len):
    """
    Compile one transformer layer into a φ-compute Program.

    This function encodes ALL architectural knowledge:
      - Attention with GQA, RoPE, causal masking
      - SwiGLU MLP

    The node executing this program has NO idea it's a transformer.
    It just sees opcodes operating on registers and blobs.

    Returns: Program with R0 as input and output (hidden state)
    """
    p = Program()
    hd2 = HEAD_DIM // 2

    # Input: R0 = hidden state (batch, seq_len, HIDDEN_DIM)
    # We'll use registers R0-R50

    # ═══════════════════════════════════════════════════════
    # ATTENTION BLOCK
    # ═══════════════════════════════════════════════════════

    # 1. Pre-attention RMS norm
    p.rms_norm(1, 0, bids['norm_input'], HIDDEN_DIM)

    # 2. Flatten to 2D for matmul: (batch*seq, hidden)
    p.reshape(2, 1, (batch * seq_len, HIDDEN_DIM))

    # 3. Q/K/V projections
    p.matmul(3, 2, bids['W_q'])    # (batch*seq, num_heads*head_dim)
    p.matmul(4, 2, bids['W_k'])    # (batch*seq, num_kv_heads*head_dim)
    p.matmul(5, 2, bids['W_v'])

    # 4. Add biases
    p.broadcast_add(3, 3, bids['bias_q'])
    p.broadcast_add(4, 4, bids['bias_k'])
    p.broadcast_add(5, 5, bids['bias_v'])

    # 5. Reshape to multi-head: (batch, seq, heads, dim)
    p.reshape(3, 3, (batch, seq_len, NUM_HEADS, HEAD_DIM))
    p.reshape(4, 4, (batch, seq_len, NUM_KV_HEADS, HEAD_DIM))
    p.reshape(5, 5, (batch, seq_len, NUM_KV_HEADS, HEAD_DIM))

    # 6. Transpose to (batch, heads, seq, dim)
    p.transpose(3, 3, (0, 2, 1, 3))
    p.transpose(4, 4, (0, 2, 1, 3))
    p.transpose(5, 5, (0, 2, 1, 3))

    # 7. RoPE on Q (R3) — decomposed into primitives
    #    Load cos/sin tables, slice to seq_len, reshape for broadcast
    p.load_blob(10, BLOB_ROPE_COS)                             # (max_seq, head_dim)
    p.slice_op(11, 10, axis=0, start=0, end=seq_len)           # (seq_len, head_dim)
    p.reshape(12, 11, (1, 1, seq_len, HEAD_DIM))               # broadcastable

    p.load_blob(13, BLOB_ROPE_SIN)
    p.slice_op(14, 13, axis=0, start=0, end=seq_len)
    p.reshape(15, 14, (1, 1, seq_len, HEAD_DIM))

    # Q rotation: x_rot = [-x[..., hd2:], x[..., :hd2]]
    p.slice_op(16, 3, axis=3, start=0, end=hd2)                # Q first half
    p.slice_op(17, 3, axis=3, start=hd2, end=HEAD_DIM)         # Q second half
    p.negate(18, 17)                                             # -Q second half
    p.concat(19, 18, 16, axis=3)                                 # Q_rot = [-q2, q1]
    p.mul(20, 3, 12)                                             # t1 = Q * cos
    p.mul(21, 19, 15)                                            # t2 = Q_rot * sin
    p.add(3, 20, 21)                                             # Q = t1 + t2

    # 8. RoPE on K (R4) — reuse cos/sin from R12, R15
    p.slice_op(16, 4, axis=3, start=0, end=hd2)                # K first half
    p.slice_op(17, 4, axis=3, start=hd2, end=HEAD_DIM)         # K second half
    p.negate(18, 17)
    p.concat(19, 18, 16, axis=3)                                 # K_rot
    p.mul(20, 4, 12)
    p.mul(21, 19, 15)
    p.add(4, 20, 21)                                             # K = roped

    # 9. GQA expand: repeat KV heads
    p.repeat(4, 4, axis=1, count=HEADS_PER_KV)    # K: (b, 28, s, d)
    p.repeat(5, 5, axis=1, count=HEADS_PER_KV)    # V: (b, 28, s, d)

    # 10. Attention scores: Q @ K^T
    p.einsum_qk(22, 3, 4)                         # (b, heads, seq, seq)

    # 11. Scale by 1/sqrt(d)
    p.scale(22, 22, ATTN_SCALE_EXP)

    # 12. Causal mask (only if seq > 1)
    if seq_len > 1:
        p.causal_mask(22, 22, mask_exp=MASK_EXP)

    # 13. Softmax
    p.softmax(23, 22, axis=-1)

    # 14. Value aggregation: attn @ V
    p.einsum_av(24, 23, 5)                         # (b, heads, seq, dim)

    # 15. Transpose back: (b, seq, heads, dim)
    p.transpose(24, 24, (0, 2, 1, 3))

    # 16. Reshape: (b*seq, heads*dim)
    p.reshape(25, 24, (batch * seq_len, NUM_HEADS * HEAD_DIM))

    # 17. O projection
    p.matmul(26, 25, bids['W_o'])

    # 18. Reshape back to 3D
    p.reshape(26, 26, (batch, seq_len, HIDDEN_DIM))

    # 19. Residual add
    p.add(0, 0, 26)

    # ═══════════════════════════════════════════════════════
    # MLP BLOCK
    # ═══════════════════════════════════════════════════════

    # 20. Pre-MLP RMS norm
    p.rms_norm(30, 0, bids['norm_post'], HIDDEN_DIM)

    # 21. Flatten to 2D
    p.reshape(31, 30, (batch * seq_len, HIDDEN_DIM))

    # 22. Gate and Up projections
    p.matmul(32, 31, bids['W_gate'])    # (b*s, intermediate)
    p.matmul(33, 31, bids['W_up'])

    # 23. SiLU on gate
    p.silu(32, 32)

    # 24. Gate × Up
    p.mul(34, 32, 33)

    # 25. Down projection
    p.matmul(35, 34, bids['W_down'])

    # 26. Reshape to 3D
    p.reshape(35, 35, (batch, seq_len, HIDDEN_DIM))

    # 27. Residual add
    p.add(0, 0, 35)

    # Output: R0
    p.set_outputs([0])

    return p


# ---------------------------------------------------------------------------
# Local reference computation
# ---------------------------------------------------------------------------
def decompress_weight(path):
    """Load weight from npz and return as (signs, exps_int16) — decompressed."""
    data = np.load(path)
    if 'exponents' in data:
        return data['signs'], data['exponents']
    elif 'quant_exps' in data:
        signs = data['signs']
        quant = data['quant_exps']
        row_min = data['row_min'].astype(np.int32)
        row_max = data['row_max'].astype(np.int32)
        rng = np.maximum(row_max - row_min, 1)
        exps = row_min[:, np.newaxis] + quant.astype(np.int32) * rng[:, np.newaxis] // 255
        return signs, exps.astype(np.int16)
    else:
        raise ValueError(f"Unknown format: {list(data.keys())}")


def compute_reference_layer(layer_dir, h_signs, h_exps, rope):
    """Compute one layer locally using phi_integer primitives. Returns (h_signs, h_exps)."""
    batch, seq_len, hidden_dim = h_signs.shape

    # Load weights
    def load_w(name):
        s, e = decompress_weight(os.path.join(layer_dir, f'{name}.npz'))
        return PhiEncoded(signs=s, exponents=e)

    W_q = load_w('q_proj')
    W_k = load_w('k_proj')
    W_v = load_w('v_proj')
    W_o = load_w('o_proj')
    W_gate = load_w('gate_proj')
    W_up = load_w('up_proj')
    W_down = load_w('down_proj')

    # Norms
    norms = np.load(os.path.join(layer_dir, 'norms.npz'))
    nw_in_s, nw_in_e = float_to_phi(norms['input_layernorm'])
    nw_post_s, nw_post_e = float_to_phi(norms['post_attention_layernorm'])

    # Biases
    biases = np.load(os.path.join(layer_dir, 'biases.npz'))
    bq_s, bq_e = float_to_phi(biases['q_proj_bias'])
    bk_s, bk_e = float_to_phi(biases['k_proj_bias'])
    bv_s, bv_e = float_to_phi(biases['v_proj_bias'])

    # --- Attention ---
    n_s, n_e = phi_rms_norm_int(h_signs, h_exps, nw_in_s, nw_in_e, hidden_dim)
    n_s_2d = n_s.reshape(-1, hidden_dim)
    n_e_2d = n_e.reshape(-1, hidden_dim)

    q_s, q_e = phi_matmul_integer(W_q, n_s_2d, n_e_2d)
    k_s, k_e = phi_matmul_integer(W_k, n_s_2d, n_e_2d)
    v_s, v_e = phi_matmul_integer(W_v, n_s_2d, n_e_2d)

    # Biases
    q_s, q_e = phi_add_encoded(q_s, q_e,
        np.broadcast_to(bq_s, q_s.shape).copy(),
        np.broadcast_to(bq_e, q_e.shape).copy())
    k_s, k_e = phi_add_encoded(k_s, k_e,
        np.broadcast_to(bk_s, k_s.shape).copy(),
        np.broadcast_to(bk_e, k_e.shape).copy())
    v_s, v_e = phi_add_encoded(v_s, v_e,
        np.broadcast_to(bv_s, v_s.shape).copy(),
        np.broadcast_to(bv_e, v_e.shape).copy())

    # Reshape + transpose
    q_s = q_s.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    q_e = q_e.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k_s = k_s.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k_e = k_e.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v_s = v_s.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v_e = v_e.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    # RoPE
    q_s, q_e = rope.apply(q_s, q_e)
    k_s, k_e = rope.apply(k_s, k_e)

    # GQA expand
    k_s = np.repeat(k_s, HEADS_PER_KV, axis=1)
    k_e = np.repeat(k_e, HEADS_PER_KV, axis=1)
    v_s = np.repeat(v_s, HEADS_PER_KV, axis=1)
    v_e = np.repeat(v_e, HEADS_PER_KV, axis=1)

    # Attention
    score_s, score_e = phi_einsum_qk_int(q_s, q_e, k_s, k_e)
    score_s, score_e = phi_scale_int(score_s, score_e, ATTN_SCALE_EXP)

    if seq_len > 1:
        mask_r, mask_c = np.triu_indices(seq_len, k=1)
        score_s[:, :, mask_r, mask_c] = np.int8(-1)
        score_e[:, :, mask_r, mask_c] = np.int16(MASK_EXP)

    attn_s, attn_e = phi_softmax_full_int(score_s, score_e, axis=-1)
    ctx_s, ctx_e = phi_einsum_av_int(attn_s, attn_e, v_s, v_e)

    ctx_s = ctx_s.transpose(0, 2, 1, 3).reshape(batch * seq_len, NUM_HEADS * HEAD_DIM)
    ctx_e = ctx_e.transpose(0, 2, 1, 3).reshape(batch * seq_len, NUM_HEADS * HEAD_DIM)

    o_s, o_e = phi_matmul_integer(W_o, ctx_s, ctx_e)
    o_s = o_s.reshape(batch, seq_len, hidden_dim)
    o_e = o_e.reshape(batch, seq_len, hidden_dim)

    h_signs, h_exps = phi_add_encoded(h_signs, h_exps, o_s, o_e)

    # --- MLP ---
    mn_s, mn_e = phi_rms_norm_int(h_signs, h_exps, nw_post_s, nw_post_e, hidden_dim)
    mn_s_2d = mn_s.reshape(-1, hidden_dim)
    mn_e_2d = mn_e.reshape(-1, hidden_dim)

    gate_s, gate_e = phi_matmul_integer(W_gate, mn_s_2d, mn_e_2d)
    up_s, up_e = phi_matmul_integer(W_up, mn_s_2d, mn_e_2d)

    gate_s, gate_e = phi_silu_int(gate_s, gate_e)
    mlp_s, mlp_e = phi_multiply_int(gate_s, gate_e, up_s, up_e)

    down_s, down_e = phi_matmul_integer(W_down, mlp_s, mlp_e)
    down_s = down_s.reshape(batch, seq_len, hidden_dim)
    down_e = down_e.reshape(batch, seq_len, hidden_dim)

    h_signs, h_exps = phi_add_encoded(h_signs, h_exps, down_s, down_e)

    return h_signs, h_exps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test full layer via thin client')
    parser.add_argument('--host', default='192.168.1.111')
    parser.add_argument('--port', type=int, default=7619)
    parser.add_argument('--layer', type=int, default=0)
    args = parser.parse_args()

    layer_idx = args.layer
    batch, seq_len = 1, 5

    print("=" * 60)
    print(f"  Full Layer Test — Layer {layer_idx}")
    print(f"  batch={batch}, seq_len={seq_len}")
    print("=" * 60)

    # Connect
    print(f"\nConnecting to {args.host}:{args.port}...")
    client = PhiComputeClient(args.host, args.port)
    client.connect()
    print(f"  {client.ping()}")

    # 1. Store RoPE tables
    print("\nStoring RoPE tables...")
    t0 = time.perf_counter()
    rope = store_rope_tables(client)
    print(f"  Done ({(time.perf_counter()-t0)*1000:.0f}ms)")

    # 2. Store layer weights
    print(f"\nLoading layer {layer_idx} weights onto node...")
    t0 = time.perf_counter()
    bids = load_and_store_layer(client, layer_idx)
    dt = time.perf_counter() - t0
    print(f"  Done ({dt:.1f}s)")
    blobs = client.list_blobs()
    total_mb = sum(b[2] for b in blobs) / 1e6
    print(f"  {len(blobs)} blobs stored ({total_mb:.0f} MB)")

    # 3. Generate test input
    print("\nGenerating test input...")
    np.random.seed(42)
    h_s = np.random.choice([-1, 1], size=(batch, seq_len, HIDDEN_DIM)).astype(np.int8)
    h_e = np.random.randint(-1000, 1000, size=(batch, seq_len, HIDDEN_DIM)).astype(np.int16)

    # 4. Compile layer program
    print("\nCompiling layer program...")
    program = compile_layer(bids, batch, seq_len)
    n_instr = len(program.instructions)
    n_shapes = len(program.shape_table)
    print(f"  {n_instr} instructions, {n_shapes} shape entries")

    # 5. Execute on thin client
    print("\nExecuting on thin client...")
    program.add_input(0, h_s, h_e)
    t0 = time.perf_counter()
    results = client.run_program(program)
    remote_time = time.perf_counter() - t0
    remote_s, remote_e = results[0]
    print(f"  Done ({remote_time:.1f}s)")
    print(f"  Output shape: {remote_s.shape}")

    # 6. Compute local reference
    print("\nComputing local reference...")
    layer_dir = os.path.join(WEIGHTS_DIR, f'layer_{layer_idx:02d}')
    t0 = time.perf_counter()
    ref_s, ref_e = compute_reference_layer(layer_dir, h_s.copy(), h_e.copy(), rope)
    local_time = time.perf_counter() - t0
    print(f"  Done ({local_time:.1f}s)")

    # 7. Compare
    print("\n" + "=" * 60)
    s_match = np.array_equal(ref_s, remote_s)
    e_match = np.array_equal(ref_e, remote_e)

    if s_match and e_match:
        print("  ✓ LAYER TEST: BIT-IDENTICAL MATCH")
        print(f"    Remote: {remote_time:.1f}s  Local: {local_time:.1f}s")
        print(f"    Speedup: {local_time/remote_time:.1f}×")
    else:
        s_diff = np.sum(ref_s != remote_s)
        e_diff = np.sum(ref_e != remote_e)
        total = ref_s.size
        print(f"  ✗ LAYER TEST: MISMATCH")
        print(f"    Signs:  {s_diff}/{total} differ ({100*s_diff/total:.2f}%)")
        print(f"    Exps:   {e_diff}/{total} differ ({100*e_diff/total:.2f}%)")

        if e_diff > 0:
            diffs = np.abs(ref_e.astype(np.int32) - remote_e.astype(np.int32))
            print(f"    Max exp diff:  {np.max(diffs)}")
            print(f"    Mean exp diff: {np.mean(diffs[diffs > 0]):.1f}")

    print("=" * 60)

    client.close()
    return 0 if (s_match and e_match) else 1


if __name__ == '__main__':
    sys.exit(main())
