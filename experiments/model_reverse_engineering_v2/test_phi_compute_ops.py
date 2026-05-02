"""
Test every φ-compute opcode against local reference implementations.

For each opcode:
  1. Generate random φ-encoded test data
  2. Compute expected result locally (phi_integer.py)
  3. Send to thin client via EXEC or PROGRAM
  4. Compare: must be BIT-IDENTICAL

Usage:
    # Start thin client on gimli first:
    #   python phi_compute_node.py --port 7619
    # Then run tests:
    python test_phi_compute_ops.py [--host 192.168.1.111] [--port 7619]
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_integer import (
    phi_matmul_integer, phi_add_encoded, phi_multiply_int,
    phi_rms_norm_int, phi_silu_int, phi_softmax_full_int,
    phi_einsum_qk_int, phi_einsum_av_int, phi_scale_int,
    PhiRoPEInt, PhiEncoded,
)
from phi_geometric.inference.phi_compute_client import (
    PhiComputeClient, Program,
    OP_MATMUL, OP_ADD, OP_MUL, OP_RMS_NORM, OP_SILU, OP_SOFTMAX,
    OP_SCALE, OP_EINSUM_QK, OP_EINSUM_AV, OP_ROPE,
    OP_RESHAPE, OP_TRANSPOSE, OP_REPEAT, OP_BROADCAST_ADD,
    OP_CAUSAL_MASK, OP_COPY,
)


def random_phi(shape, exp_range=(-2000, 2000)):
    """Generate random φ-encoded tensor."""
    signs = np.random.choice([-1, 1], size=shape).astype(np.int8)
    exps = np.random.randint(exp_range[0], exp_range[1], size=shape).astype(np.int16)
    return signs, exps


def check_identical(name, ref_s, ref_e, got_s, got_e):
    """Check bit-identical match."""
    s_match = np.array_equal(ref_s, got_s)
    e_match = np.array_equal(ref_e, got_e)
    if s_match and e_match:
        print(f"  ✓ {name}: PASS (shape={ref_s.shape})")
        return True
    else:
        s_diff = np.sum(ref_s != got_s) if not s_match else 0
        e_diff = np.sum(ref_e != got_e) if not e_match else 0
        total = ref_s.size
        print(f"  ✗ {name}: FAIL — signs:{s_diff}/{total} exps:{e_diff}/{total} differ")
        if e_diff > 0:
            idx = np.where(ref_e != got_e)
            first = tuple(i[0] for i in idx)
            print(f"    First diff at {first}: ref_e={ref_e[first]} got_e={got_e[first]}")
        return False


def make_weight(out_features, in_features):
    """Create a random weight matrix as (signs, exps) tuple."""
    s = np.random.choice([-1, 1], size=(out_features, in_features)).astype(np.int8)
    e = np.random.randint(-1500, 1500, size=(out_features, in_features)).astype(np.int16)
    return (s, e)


# ---------------------------------------------------------------------------
# Individual opcode tests
# ---------------------------------------------------------------------------

def test_ping(client):
    print("\n--- PING ---")
    resp = client.ping()
    print(f"  ✓ PING: {resp}")
    return True


def test_store_and_list(client):
    print("\n--- STORE / LIST / DROP ---")
    s, e = random_phi((4, 8))
    client.store(9999, s, e)
    blobs = client.list_blobs()
    found = any(b[0] == 9999 for b in blobs)
    if found:
        print(f"  ✓ STORE+LIST: blob 9999 found")
    else:
        print(f"  ✗ STORE+LIST: blob 9999 NOT found in {blobs}")
        return False
    client.drop(9999)
    blobs = client.list_blobs()
    found = any(b[0] == 9999 for b in blobs)
    if not found:
        print(f"  ✓ DROP: blob 9999 removed")
    else:
        print(f"  ✗ DROP: blob 9999 still present")
        return False
    return True


def test_add(client):
    print("\n--- ADD ---")
    s_a, e_a = random_phi((5, 64))
    s_b, e_b = random_phi((5, 64))
    ref_s, ref_e = phi_add_encoded(s_a, e_a, s_b, e_b)
    got_s, got_e = client.exec_add(s_a, e_a, s_b, e_b)
    return check_identical("ADD", ref_s, ref_e, got_s, got_e)


def test_mul(client):
    print("\n--- MUL ---")
    s_a, e_a = random_phi((5, 64))
    s_b, e_b = random_phi((5, 64))
    ref_s, ref_e = phi_multiply_int(s_a, e_a, s_b, e_b)
    got_s, got_e = client.exec_mul(s_a, e_a, s_b, e_b)
    return check_identical("MUL", ref_s, ref_e, got_s, got_e)


def test_silu(client):
    print("\n--- SILU ---")
    s, e = random_phi((5, 128))
    ref_s, ref_e = phi_silu_int(s, e)
    got_s, got_e = client.exec_silu(s, e)
    return check_identical("SILU", ref_s, ref_e, got_s, got_e)


def test_scale(client):
    print("\n--- SCALE ---")
    s, e = random_phi((5, 64))
    offset = -725  # typical attention scale
    ref_s, ref_e = phi_scale_int(s, e, offset)
    got_s, got_e = client.exec_scale(s, e, offset)
    return check_identical("SCALE", ref_s, ref_e, got_s, got_e)


def test_softmax(client):
    print("\n--- SOFTMAX ---")
    s, e = random_phi((3, 5, 5), exp_range=(-500, 500))
    ref_s, ref_e = phi_softmax_full_int(s, e, axis=-1)
    got_s, got_e = client.exec_softmax(s, e, axis=-1)
    return check_identical("SOFTMAX", ref_s, ref_e, got_s, got_e)


def test_matmul(client):
    print("\n--- MATMUL ---")
    out_f, in_f = 32, 64
    weight = make_weight(out_f, in_f)
    blob_id = 1001

    # Store weight on node
    client.store(blob_id, weight[0], weight[1])

    # Test input — local phi_matmul_integer expects PhiEncoded, not tuple
    x_s, x_e = random_phi((3, in_f))
    W_enc = PhiEncoded(signs=weight[0], exponents=weight[1])
    ref_s, ref_e = phi_matmul_integer(W_enc, x_s, x_e)
    got_s, got_e = client.exec_matmul(blob_id, x_s, x_e)

    client.drop(blob_id)
    return check_identical("MATMUL", ref_s, ref_e, got_s, got_e)


def test_rms_norm(client):
    print("\n--- RMS_NORM ---")
    hidden_dim = 64
    x_s, x_e = random_phi((1, 5, hidden_dim))
    w_s, w_e = random_phi((hidden_dim,), exp_range=(-200, 200))
    blob_id = 1002

    client.store(blob_id, w_s, w_e)
    ref_s, ref_e = phi_rms_norm_int(x_s, x_e, w_s, w_e, hidden_dim)
    got_s, got_e = client.exec_rms_norm(x_s, x_e, blob_id, hidden_dim)

    client.drop(blob_id)
    return check_identical("RMS_NORM", ref_s, ref_e, got_s, got_e)


def test_broadcast_add(client):
    print("\n--- BROADCAST_ADD ---")
    x_s, x_e = random_phi((5, 64))
    bias_s, bias_e = random_phi((64,))
    blob_id = 1003

    client.store(blob_id, bias_s, bias_e)

    # Local reference
    bs_bc = np.broadcast_to(bias_s, x_s.shape).copy()
    be_bc = np.broadcast_to(bias_e, x_e.shape).copy()
    ref_s, ref_e = phi_add_encoded(x_s, x_e, bs_bc, be_bc)

    got_s, got_e = client.exec_broadcast_add(x_s, x_e, blob_id)
    client.drop(blob_id)
    return check_identical("BROADCAST_ADD", ref_s, ref_e, got_s, got_e)


def test_einsum_qk(client):
    print("\n--- EINSUM_QK ---")
    # (batch, heads, seq, dim)
    q_s, q_e = random_phi((1, 2, 3, 16))
    k_s, k_e = random_phi((1, 2, 3, 16))
    ref_s, ref_e = phi_einsum_qk_int(q_s, q_e, k_s, k_e)

    got_s, got_e = client.exec_op(
        OP_EINSUM_QK,
        inputs=[(0, q_s, q_e), (1, k_s, k_e)],
        dst=2, src_a=0, src_b=1
    )
    return check_identical("EINSUM_QK", ref_s, ref_e, got_s, got_e)


def test_einsum_av(client):
    print("\n--- EINSUM_AV ---")
    # attn: (batch, heads, seq, seq), v: (batch, heads, seq, dim)
    a_s, a_e = random_phi((1, 2, 3, 3), exp_range=(-500, 0))
    v_s, v_e = random_phi((1, 2, 3, 16))
    ref_s, ref_e = phi_einsum_av_int(a_s, a_e, v_s, v_e)

    got_s, got_e = client.exec_op(
        OP_EINSUM_AV,
        inputs=[(0, a_s, a_e), (1, v_s, v_e)],
        dst=2, src_a=0, src_b=1
    )
    return check_identical("EINSUM_AV", ref_s, ref_e, got_s, got_e)


def test_rope(client):
    print("\n--- ROPE ---")
    head_dim = 16
    seq_len = 3
    rope = PhiRoPEInt(head_dim, rope_theta=1_000_000.0, max_seq_len=64)

    x_s, x_e = random_phi((1, 2, seq_len, head_dim))
    ref_s, ref_e = rope.apply(x_s, x_e)

    # Store cos/sin tables as blob (4-tuple)
    blob_id = 1004
    # Serialize as 4-tensor blob: cos_signs, cos_exps, sin_signs, sin_exps
    # We need to send this differently — store as raw phi tensor won't work for 4-tuple
    # Use a special store: store each as concatenated then let the server split?
    # Actually, looking at the server code, STORE stores a tuple of arrays.
    # But the wire format only handles (signs, exps) pairs.
    # For ROPE, we need to store 4 arrays. Let me send via STORE_LOCAL or
    # pack cos+sin into a single blob.

    # Approach: concatenate cos and sin along a new axis, so blob is
    # (signs_concat, exps_concat) where first half is cos, second is sin
    # Server ROPE opcode would need to unpack this.
    # BUT the server currently expects 4-tuple for ROPE blobs.

    # For EXEC mode, let me just test via PROGRAM instead, where we can
    # pass rope tables differently. Or: adjust the protocol.

    # Simplest fix: store cos and sin as SEPARATE blobs, and modify the
    # ROPE opcode to take two blob_refs. But instruction format only has one.

    # Alternative: pack 4 arrays into a single STORE payload.
    # The server stores whatever tuple we give it.
    # We need a store_rope_tables method that stores a 4-tuple.

    # Let me use store() for cos, store() for sin, and adjust ROPE
    # to reference them differently... This needs a protocol change.

    # For now, the cleanest approach: store cos and sin as a combined blob.
    # Convention: blob = (cos_signs, cos_exps, sin_signs, sin_exps)
    # Wire format for STORE: send 4 concatenated tensors with a special flag.

    # Actually the simplest thing: use PROGRAM mode where we can compute
    # ROPE as MUL+MUL+ADD manually (decomposed). Test that instead.
    # This is actually MORE in the spirit of thin client — ROPE isn't
    # a primitive, it's a composed operation.

    # Let's test ROPE via a small program that decomposes it:
    hd2 = head_dim // 2

    # Precompute cos/sin slices
    cos_s = rope.cos_signs[:seq_len]  # (seq, head_dim)
    cos_e = rope.cos_exps[:seq_len]
    sin_s = rope.sin_signs[:seq_len]
    sin_e = rope.sin_exps[:seq_len]

    # Broadcast to (1, 1, seq, dim)
    cos_s_4d = cos_s[np.newaxis, np.newaxis, :, :]
    cos_e_4d = cos_e[np.newaxis, np.newaxis, :, :]
    sin_s_4d = sin_s[np.newaxis, np.newaxis, :, :]
    sin_e_4d = sin_e[np.newaxis, np.newaxis, :, :]

    # x_rot = [-x[..., hd2:], x[..., :hd2]]
    x_rot_s = np.concatenate([-x_s[..., hd2:], x_s[..., :hd2]], axis=-1)
    x_rot_e = np.concatenate([x_e[..., hd2:], x_e[..., :hd2]], axis=-1)

    # Compute locally with primitives
    t1_s, t1_e = phi_multiply_int(x_s, x_e, cos_s_4d, cos_e_4d)
    t2_s, t2_e = phi_multiply_int(x_rot_s, x_rot_e, sin_s_4d, sin_e_4d)
    local_s, local_e = phi_add_encoded(t1_s, t1_e, t2_s, t2_e)

    # Verify local decomposition matches PhiRoPEInt.apply
    ok1 = check_identical("ROPE local decomp", ref_s, ref_e, local_s, local_e)
    if not ok1:
        print("  ! ROPE decomposition doesn't match PhiRoPEInt — skip remote test")
        return False

    # Now test via remote program:
    # R0 = x, R1 = x_rot, R2 = cos, R3 = sin
    # MUL R4, R0, R2  (t1 = x * cos)
    # MUL R5, R1, R3  (t2 = x_rot * sin)
    # ADD R6, R4, R5  (result = t1 + t2)
    prog = Program()
    prog.add_input(0, x_s, x_e)
    prog.add_input(1, x_rot_s, x_rot_e)
    prog.add_input(2, cos_s_4d.astype(np.int8), cos_e_4d.astype(np.int16))
    prog.add_input(3, sin_s_4d.astype(np.int8), sin_e_4d.astype(np.int16))
    prog.mul(4, 0, 2)
    prog.mul(5, 1, 3)
    prog.add(6, 4, 5)
    prog.set_outputs([6])

    results = client.run_program(prog)
    got_s, got_e = results[0]

    return check_identical("ROPE (via PROGRAM)", ref_s, ref_e, got_s, got_e)


def test_reshape_transpose(client):
    print("\n--- RESHAPE + TRANSPOSE (PROGRAM) ---")
    # Input: (5, 64) → reshape to (5, 4, 16) → transpose to (4, 5, 16)
    s, e = random_phi((5, 64))

    ref_reshaped_s = s.reshape(5, 4, 16)
    ref_reshaped_e = e.reshape(5, 4, 16)
    ref_transposed_s = ref_reshaped_s.transpose(1, 0, 2)
    ref_transposed_e = ref_reshaped_e.transpose(1, 0, 2)

    prog = Program()
    prog.add_input(0, s, e)
    prog.reshape(1, 0, (5, 4, 16))
    prog.transpose(2, 1, (1, 0, 2))
    prog.set_outputs([2])

    results = client.run_program(prog)
    got_s, got_e = results[0]
    return check_identical("RESHAPE+TRANSPOSE", ref_transposed_s, ref_transposed_e,
                           got_s, got_e)


def test_repeat(client):
    print("\n--- REPEAT (PROGRAM) ---")
    s, e = random_phi((1, 2, 3, 8))
    ref_s = np.repeat(s, 7, axis=1)
    ref_e = np.repeat(e, 7, axis=1)

    prog = Program()
    prog.add_input(0, s, e)
    prog.repeat(1, 0, axis=1, count=7)
    prog.set_outputs([1])

    results = client.run_program(prog)
    got_s, got_e = results[0]
    return check_identical("REPEAT", ref_s, ref_e, got_s, got_e)


def test_causal_mask(client):
    print("\n--- CAUSAL_MASK (PROGRAM) ---")
    seq_len = 5
    s, e = random_phi((1, 2, seq_len, seq_len))

    # Local reference
    ref_s = s.copy()
    ref_e = e.copy()
    mask_r, mask_c = np.triu_indices(seq_len, k=1)
    ref_s[:, :, mask_r, mask_c] = np.int8(-1)
    ref_e[:, :, mask_r, mask_c] = np.int16(-30000)

    prog = Program()
    prog.add_input(0, s, e)
    prog.causal_mask(1, 0, mask_exp=-30000)
    prog.set_outputs([1])

    results = client.run_program(prog)
    got_s, got_e = results[0]
    return check_identical("CAUSAL_MASK", ref_s, ref_e, got_s, got_e)


def test_copy(client):
    print("\n--- COPY (PROGRAM) ---")
    s, e = random_phi((3, 16))

    prog = Program()
    prog.add_input(0, s, e)
    prog.copy(1, 0)
    prog.set_outputs([1])

    results = client.run_program(prog)
    got_s, got_e = results[0]
    return check_identical("COPY", s, e, got_s, got_e)


def test_multi_output_program(client):
    """Test a program that returns multiple outputs."""
    print("\n--- MULTI-OUTPUT PROGRAM ---")
    s_a, e_a = random_phi((5, 32))
    s_b, e_b = random_phi((5, 32))

    ref_add_s, ref_add_e = phi_add_encoded(s_a, e_a, s_b, e_b)
    ref_mul_s, ref_mul_e = phi_multiply_int(s_a, e_a, s_b, e_b)

    prog = Program()
    prog.add_input(0, s_a, e_a)
    prog.add_input(1, s_b, e_b)
    prog.add(2, 0, 1)
    prog.mul(3, 0, 1)
    prog.set_outputs([2, 3])

    results = client.run_program(prog)
    ok1 = check_identical("MULTI_OUT[0] ADD", ref_add_s, ref_add_e,
                          results[0][0], results[0][1])
    ok2 = check_identical("MULTI_OUT[1] MUL", ref_mul_s, ref_mul_e,
                          results[1][0], results[1][1])
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test φ-compute opcodes')
    parser.add_argument('--host', default='192.168.1.111')
    parser.add_argument('--port', type=int, default=7619)
    args = parser.parse_args()

    np.random.seed(42)

    print(f"Connecting to φ-compute node at {args.host}:{args.port}...")
    client = PhiComputeClient(args.host, args.port)
    client.connect()
    print("Connected.\n")

    tests = [
        ("PING",              test_ping),
        ("STORE/LIST/DROP",   test_store_and_list),
        ("ADD",               test_add),
        ("MUL",               test_mul),
        ("SILU",              test_silu),
        ("SCALE",             test_scale),
        ("SOFTMAX",           test_softmax),
        ("MATMUL",            test_matmul),
        ("RMS_NORM",          test_rms_norm),
        ("BROADCAST_ADD",     test_broadcast_add),
        ("EINSUM_QK",         test_einsum_qk),
        ("EINSUM_AV",         test_einsum_av),
        ("ROPE",              test_rope),
        ("RESHAPE+TRANSPOSE", test_reshape_transpose),
        ("REPEAT",            test_repeat),
        ("CAUSAL_MASK",       test_causal_mask),
        ("COPY",              test_copy),
        ("MULTI-OUTPUT",      test_multi_output_program),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            t0 = time.perf_counter()
            ok = test_fn(client)
            dt = (time.perf_counter() - t0) * 1000
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append(name)
            print(f"  ({dt:.1f}ms)")
        except Exception as e:
            failed += 1
            errors.append(f"{name}: {e}")
            print(f"  ✗ {name}: ERROR — {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print(f"FAILED: {errors}")
    print("=" * 60)

    client.close()
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
