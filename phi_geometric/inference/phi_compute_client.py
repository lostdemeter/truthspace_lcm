"""
φ-Compute Client — Controller-side client for the thin compute node.

Sends operations and programs to a model-agnostic φ-compute node.
All model knowledge lives HERE (the controller), not on the node.

Usage:
    client = PhiComputeClient('192.168.1.111', 7619)
    client.connect()
    client.store(blob_id=1, signs=s, exps=e)
    result = client.exec_matmul(blob_id=1, input_signs=x_s, input_exps=x_e)
"""

import socket
import struct
import time
import numpy as np

# Protocol constants (must match phi_compute_node.py)
MAGIC_REQ = b'PHC\x00'
MAGIC_RSP = b'PHR\x00'
HEADER_SIZE = 20

MSG_STORE       = 0x01
MSG_DROP        = 0x02
MSG_LIST        = 0x03
MSG_EXEC        = 0x04
MSG_PROGRAM     = 0x05
MSG_STORE_LOCAL = 0x06
MSG_SHUTDOWN    = 0xFE
MSG_PING        = 0xFF

OP_MATMUL        = 0x01
OP_ADD           = 0x02
OP_MUL           = 0x03
OP_RMS_NORM      = 0x04
OP_SILU          = 0x05
OP_SOFTMAX       = 0x06
OP_SCALE         = 0x07
OP_EINSUM_QK     = 0x08
OP_EINSUM_AV     = 0x09
OP_ROPE          = 0x0A
OP_RESHAPE       = 0x10
OP_TRANSPOSE     = 0x11
OP_REPEAT        = 0x12
OP_SLICE         = 0x13
OP_BROADCAST_ADD = 0x14
OP_CAUSAL_MASK   = 0x15
OP_NEGATE        = 0x16
OP_CONCAT        = 0x17
OP_LOAD          = 0x18
OP_COPY          = 0x1F

INSTRUCTION_SIZE = 16
STATUS_OK  = 0
STATUS_ERR = 1


# ---------------------------------------------------------------------------
# Wire format helpers
# ---------------------------------------------------------------------------
def encode_phi_tensor(signs, exps):
    """Serialize φ-tensor: n_dims(u8) + shape(u32[]) + signs(int8) + exps(int16)."""
    shape = signs.shape
    buf = struct.pack('<B', len(shape))
    for d in shape:
        buf += struct.pack('<I', d)
    buf += np.ascontiguousarray(signs).tobytes()
    buf += np.ascontiguousarray(exps.astype(np.int16)).tobytes()
    return buf


def decode_phi_tensor(data, offset=0):
    """Deserialize φ-tensor. Returns (signs, exps, new_offset)."""
    n_dims = data[offset]; offset += 1
    shape = []
    for _ in range(n_dims):
        d = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        shape.append(d)
    shape = tuple(shape)
    n_el = 1
    for d in shape:
        n_el *= d
    signs = np.frombuffer(data[offset:offset+n_el], dtype=np.int8).reshape(shape).copy()
    offset += n_el
    exps = np.frombuffer(data[offset:offset+n_el*2], dtype=np.int16).reshape(shape).copy()
    offset += n_el * 2
    return signs, exps, offset


def encode_instruction(opcode, dst=0, src_a=0xFF, src_b=0xFF, blob_ref=0, param=0):
    """Encode a 16-byte instruction."""
    return struct.pack('<BBBB Q i', opcode, dst, src_a, src_b, blob_ref, param)


def encode_shape_entry(values):
    """Encode a shape table entry: n_values(u8) + values(i32[])."""
    buf = struct.pack('<B', len(values))
    for v in values:
        buf += struct.pack('<i', v)
    return buf


# ---------------------------------------------------------------------------
# Instruction builder
# ---------------------------------------------------------------------------
class Instruction:
    """Represents a single opcode in a program."""
    __slots__ = ('opcode', 'dst', 'src_a', 'src_b', 'blob_ref', 'param')

    def __init__(self, opcode, dst=0, src_a=0xFF, src_b=0xFF, blob_ref=0, param=0):
        self.opcode = opcode
        self.dst = dst
        self.src_a = src_a
        self.src_b = src_b
        self.blob_ref = blob_ref
        self.param = param

    def encode(self):
        return encode_instruction(self.opcode, self.dst, self.src_a, self.src_b,
                                  self.blob_ref, self.param)


# ---------------------------------------------------------------------------
# Program builder
# ---------------------------------------------------------------------------
class Program:
    """Builds a program: a sequence of instructions with shape table."""

    def __init__(self):
        self.instructions = []
        self.shape_table = []
        self.inputs = []      # (reg_id, signs, exps)
        self.output_regs = []

    def add_shape(self, values):
        """Add a shape table entry, return its index."""
        idx = len(self.shape_table)
        self.shape_table.append(values)
        return idx

    def add_input(self, reg_id, signs, exps):
        """Register an input that will be loaded into a register."""
        self.inputs.append((reg_id, signs, exps))

    def set_outputs(self, reg_ids):
        """Specify which registers to return."""
        self.output_regs = list(reg_ids)

    # --- Instruction helpers ---

    def matmul(self, dst, src, blob_id):
        self.instructions.append(Instruction(OP_MATMUL, dst, src, 0xFF, blob_id))

    def add(self, dst, src_a, src_b):
        self.instructions.append(Instruction(OP_ADD, dst, src_a, src_b))

    def mul(self, dst, src_a, src_b):
        self.instructions.append(Instruction(OP_MUL, dst, src_a, src_b))

    def rms_norm(self, dst, src, blob_id, dim):
        self.instructions.append(Instruction(OP_RMS_NORM, dst, src, 0xFF, blob_id, dim))

    def silu(self, dst, src):
        self.instructions.append(Instruction(OP_SILU, dst, src))

    def softmax(self, dst, src, axis=-1):
        self.instructions.append(Instruction(OP_SOFTMAX, dst, src, param=axis))

    def scale(self, dst, src, exp_offset):
        self.instructions.append(Instruction(OP_SCALE, dst, src, param=exp_offset))

    def einsum_qk(self, dst, src_q, src_k):
        self.instructions.append(Instruction(OP_EINSUM_QK, dst, src_q, src_k))

    def einsum_av(self, dst, src_attn, src_v):
        self.instructions.append(Instruction(OP_EINSUM_AV, dst, src_attn, src_v))

    def rope(self, dst, src, freq_blob_id):
        self.instructions.append(Instruction(OP_ROPE, dst, src, blob_ref=freq_blob_id))

    def reshape(self, dst, src, shape):
        idx = self.add_shape(list(shape))
        self.instructions.append(Instruction(OP_RESHAPE, dst, src, param=idx))

    def transpose(self, dst, src, axes):
        idx = self.add_shape(list(axes))
        self.instructions.append(Instruction(OP_TRANSPOSE, dst, src, param=idx))

    def repeat(self, dst, src, axis, count):
        param = (axis & 0xFF) | ((count & 0xFFFFFF) << 8)
        self.instructions.append(Instruction(OP_REPEAT, dst, src, param=param))

    def slice_op(self, dst, src, axis, start, end):
        idx = self.add_shape([axis, start, end])
        self.instructions.append(Instruction(OP_SLICE, dst, src, param=idx))

    def broadcast_add(self, dst, src, blob_id):
        self.instructions.append(Instruction(OP_BROADCAST_ADD, dst, src, blob_ref=blob_id))

    def causal_mask(self, dst, src, mask_exp=-30000):
        self.instructions.append(Instruction(OP_CAUSAL_MASK, dst, src, param=mask_exp))

    def negate(self, dst, src):
        self.instructions.append(Instruction(OP_NEGATE, dst, src))

    def concat(self, dst, src_a, src_b, axis=-1):
        self.instructions.append(Instruction(OP_CONCAT, dst, src_a, src_b, param=axis))

    def load_blob(self, dst, blob_id):
        self.instructions.append(Instruction(OP_LOAD, dst, blob_ref=blob_id))

    def copy(self, dst, src):
        self.instructions.append(Instruction(OP_COPY, dst, src))

    def encode(self):
        """Serialize the full program payload."""
        parts = []
        # Header
        parts.append(struct.pack('<H', len(self.instructions)))
        parts.append(struct.pack('<B', len(self.inputs)))
        parts.append(struct.pack('<B', len(self.output_regs)))
        parts.append(struct.pack('<B', len(self.shape_table)))

        # Shape table
        for shape in self.shape_table:
            parts.append(encode_shape_entry(shape))

        # Instructions
        for instr in self.instructions:
            parts.append(instr.encode())

        # Inputs
        for reg_id, signs, exps in self.inputs:
            parts.append(struct.pack('<B', reg_id))
            parts.append(encode_phi_tensor(signs, exps))

        # Output register list
        for reg_id in self.output_regs:
            parts.append(struct.pack('<B', reg_id))

        return b''.join(parts)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class PhiComputeClient:
    """Client for the φ-compute node."""

    def __init__(self, host, port=7619):
        self.host = host
        self.port = port
        self.sock = None
        self._req_id = 0

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def _next_id(self):
        self._req_id += 1
        return self._req_id

    def _send(self, msg_type, payload=b''):
        req_id = self._next_id()
        header = MAGIC_REQ + struct.pack('<BBH I Q',
            1,         # version
            msg_type,
            0,         # flags
            len(payload),
            req_id)
        self.sock.sendall(header + payload)
        return req_id

    def _recv(self):
        """Receive response. Returns (status, payload)."""
        raw = b''
        while len(raw) < HEADER_SIZE:
            chunk = self.sock.recv(HEADER_SIZE - len(raw))
            if not chunk:
                raise ConnectionError("Connection closed")
            raw += chunk
        magic = raw[0:4]
        if magic != MAGIC_RSP:
            raise ValueError(f"Bad response magic: {magic!r}")
        status = raw[5]
        payload_len = struct.unpack('<I', raw[8:12])[0]
        request_id = struct.unpack('<Q', raw[12:20])[0]
        payload = b''
        while len(payload) < payload_len:
            chunk = self.sock.recv(payload_len - len(payload))
            if not chunk:
                raise ConnectionError("Connection closed")
            payload += chunk
        if status == STATUS_ERR:
            raise RuntimeError(f"Node error: {payload.decode('utf-8', errors='replace')}")
        return status, payload

    # --- High-level operations ---

    def ping(self):
        self._send(MSG_PING)
        _, payload = self._recv()
        return payload.decode('utf-8')

    def store(self, blob_id, signs, exps):
        """Store a raw φ-tensor blob on the node."""
        payload = struct.pack('<Q', blob_id)
        payload += struct.pack('<B', 0)  # not compressed
        payload += encode_phi_tensor(signs, exps)
        self._send(MSG_STORE, payload)
        self._recv()

    def store_compressed(self, blob_id, signs, quant, row_min, row_max):
        """Store a compressed weight blob on the node."""
        payload = struct.pack('<Q', blob_id)
        payload += struct.pack('<B', 1)  # compressed
        shape = signs.shape
        payload += struct.pack('<B', len(shape))
        for d in shape:
            payload += struct.pack('<I', d)
        payload += np.ascontiguousarray(signs).tobytes()
        payload += np.ascontiguousarray(quant).tobytes()
        payload += np.ascontiguousarray(row_min.astype(np.int16)).tobytes()
        payload += np.ascontiguousarray(row_max.astype(np.int16)).tobytes()
        self._send(MSG_STORE, payload)
        self._recv()

    def store_local(self, blob_id, file_path, fmt=0):
        """Tell the node to load a blob from its local filesystem."""
        path_bytes = file_path.encode('utf-8')
        payload = struct.pack('<Q', blob_id)
        payload += struct.pack('<H', len(path_bytes))
        payload += path_bytes
        payload += struct.pack('<B', fmt)  # 0=compressed_weight, 1=raw_phi
        self._send(MSG_STORE_LOCAL, payload)
        self._recv()

    def drop(self, blob_id):
        self._send(MSG_DROP, struct.pack('<Q', blob_id))
        self._recv()

    def list_blobs(self):
        self._send(MSG_LIST)
        _, payload = self._recv()
        offset = 0
        n_blobs = struct.unpack('<I', payload[offset:offset+4])[0]; offset += 4
        entries = []
        for _ in range(n_blobs):
            bid = struct.unpack('<Q', payload[offset:offset+8])[0]; offset += 8
            n_dims = payload[offset]; offset += 1
            shape = []
            for _ in range(n_dims):
                d = struct.unpack('<I', payload[offset:offset+4])[0]; offset += 4
                shape.append(d)
            nbytes = struct.unpack('<Q', payload[offset:offset+8])[0]; offset += 8
            entries.append((bid, tuple(shape), nbytes))
        return entries

    # --- Single-op execution (EXEC) ---

    def exec_op(self, opcode, inputs, blob_ref=0, param=0, dst=0, src_a=0, src_b=0xFF):
        """
        Execute a single operation.
        inputs: list of (reg_id, signs, exps) tuples
        Returns (signs, exps) of the destination register.
        """
        payload = encode_instruction(opcode, dst, src_a, src_b, blob_ref, param)
        payload += struct.pack('<B', len(inputs))
        for reg_id, signs, exps in inputs:
            payload += struct.pack('<B', reg_id)
            payload += encode_phi_tensor(signs, exps)
        self._send(MSG_EXEC, payload)
        _, resp = self._recv()
        signs, exps, _ = decode_phi_tensor(resp, 0)
        return signs, exps

    def exec_matmul(self, blob_id, in_signs, in_exps):
        """MATMUL: dst = blob @ input."""
        return self.exec_op(OP_MATMUL, [(0, in_signs, in_exps)],
                            blob_ref=blob_id, dst=1, src_a=0)

    def exec_add(self, s_a, e_a, s_b, e_b):
        return self.exec_op(OP_ADD, [(0, s_a, e_a), (1, s_b, e_b)],
                            dst=2, src_a=0, src_b=1)

    def exec_mul(self, s_a, e_a, s_b, e_b):
        return self.exec_op(OP_MUL, [(0, s_a, e_a), (1, s_b, e_b)],
                            dst=2, src_a=0, src_b=1)

    def exec_rms_norm(self, in_signs, in_exps, blob_id, dim):
        return self.exec_op(OP_RMS_NORM, [(0, in_signs, in_exps)],
                            blob_ref=blob_id, param=dim, dst=1, src_a=0)

    def exec_silu(self, in_signs, in_exps):
        return self.exec_op(OP_SILU, [(0, in_signs, in_exps)],
                            dst=1, src_a=0)

    def exec_softmax(self, in_signs, in_exps, axis=-1):
        return self.exec_op(OP_SOFTMAX, [(0, in_signs, in_exps)],
                            param=axis, dst=1, src_a=0)

    def exec_scale(self, in_signs, in_exps, exp_offset):
        return self.exec_op(OP_SCALE, [(0, in_signs, in_exps)],
                            param=exp_offset, dst=1, src_a=0)

    def exec_broadcast_add(self, in_signs, in_exps, blob_id):
        return self.exec_op(OP_BROADCAST_ADD, [(0, in_signs, in_exps)],
                            blob_ref=blob_id, dst=1, src_a=0)

    def shutdown(self):
        """Send SHUTDOWN to the node, causing it to exit cleanly."""
        self._send(MSG_SHUTDOWN)
        self._recv()
        self.close()

    # --- Program execution ---

    def run_program(self, program):
        """Execute a Program and return list of (signs, exps) outputs."""
        payload = program.encode()
        self._send(MSG_PROGRAM, payload)
        _, resp = self._recv()
        results = []
        offset = 0
        for _ in program.output_regs:
            signs, exps, offset = decode_phi_tensor(resp, offset)
            results.append((signs, exps))
        return results
