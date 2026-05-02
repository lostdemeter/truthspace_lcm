"""
TruthSpace Remote Integer Compute — Client

Dispatches φ-encoded integer operations to remote compute nodes
over TCP. Drop-in replacement for local integer functions.

Usage:
    client = PhiRemoteClient('192.168.1.111', 7618)
    client.connect()
    out_s, out_e = client.matmul(layer_idx=0, weight_id=WID_Q, x_signs, x_exps)
    client.close()
"""

import socket
import struct
import time
import numpy as np

# ---------------------------------------------------------------------------
# Protocol Constants (must match server.py)
# ---------------------------------------------------------------------------
MAGIC_REQ = b'PHI\x00'
MAGIC_RSP = b'PHR\x00'
HEADER_SIZE = 16

OP_PING       = 0xFF
OP_MATMUL     = 0x01
OP_RMS_NORM   = 0x02
OP_SILU       = 0x03
OP_MULTIPLY   = 0x04
OP_ADD        = 0x05
OP_SOFTMAX    = 0x06
OP_FULL_LAYER = 0x0A

WID_Q    = 0
WID_K    = 1
WID_V    = 2
WID_O    = 3
WID_GATE = 4
WID_UP   = 5
WID_DOWN = 6

STATUS_OK  = 0
STATUS_ERR = 1


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class PhiRemoteClient:
    """Client for dispatching integer operations to a remote compute node."""

    def __init__(self, host: str, port: int = 7618):
        self.host = host
        self.port = port
        self.sock = None
        self._connected = False

    def connect(self):
        """Establish TCP connection to the compute node."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.sock.connect((self.host, self.port))
        self._connected = True

    def close(self):
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # -----------------------------------------------------------------------
    # Low-level I/O
    # -----------------------------------------------------------------------
    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def _send_request(self, op_type: int, layer_idx: int, weight_id: int,
                      signs: np.ndarray, exps: np.ndarray, flags: int = 0):
        """Send a request packet with single payload."""
        if signs is None:
            # PING — no payload
            header = MAGIC_REQ + struct.pack('<BBBxII', op_type, layer_idx, weight_id, 0, 0)
            self.sock.sendall(header)
            return

        n_rows, n_cols = signs.shape
        header = MAGIC_REQ + struct.pack('<BBBBII',
                                         op_type, layer_idx, weight_id, flags,
                                         n_rows, n_cols)
        self.sock.sendall(header)
        self.sock.sendall(signs.astype(np.int8).tobytes())
        self.sock.sendall(exps.astype(np.int16).tobytes())

    def _send_dual_request(self, op_type: int, layer_idx: int, weight_id: int,
                           a_signs, a_exps, b_signs, b_exps):
        """Send a request with two payloads (for ADD, MULTIPLY)."""
        n_rows, n_cols = a_signs.shape
        header = MAGIC_REQ + struct.pack('<BBBBII',
                                         op_type, layer_idx, weight_id, 0,
                                         n_rows, n_cols)
        self.sock.sendall(header)
        self.sock.sendall(a_signs.astype(np.int8).tobytes())
        self.sock.sendall(a_exps.astype(np.int16).tobytes())
        self.sock.sendall(b_signs.astype(np.int8).tobytes())
        self.sock.sendall(b_exps.astype(np.int16).tobytes())

    def _recv_response(self):
        """Receive a response packet. Returns (status, out_signs, out_exps)."""
        header = self._recv_exact(HEADER_SIZE)
        magic = header[0:4]
        if magic != MAGIC_RSP:
            raise ValueError(f"Bad response magic: {magic!r}")

        status = header[4]
        n_rows = struct.unpack('<I', header[8:12])[0]
        n_cols = struct.unpack('<I', header[12:16])[0]

        if n_rows == 0 and n_cols == 0:
            return status, None, None

        n_elements = n_rows * n_cols
        signs_bytes = self._recv_exact(n_elements)
        exps_bytes = self._recv_exact(n_elements * 2)

        out_signs = np.frombuffer(signs_bytes, dtype=np.int8).reshape(n_rows, n_cols).copy()
        out_exps = np.frombuffer(exps_bytes, dtype=np.int16).reshape(n_rows, n_cols).copy()

        if status != STATUS_OK:
            raise RuntimeError(f"Remote compute error (status={status})")

        return status, out_signs, out_exps

    # -----------------------------------------------------------------------
    # High-level operations
    # -----------------------------------------------------------------------
    def ping(self) -> bool:
        """Ping the compute node. Returns True if alive."""
        self._send_request(OP_PING, 0, 0, None, None)
        status, _, _ = self._recv_response()
        return status == STATUS_OK

    def matmul(self, layer_idx: int, weight_id: int,
               x_signs: np.ndarray, x_exps: np.ndarray) -> tuple:
        """
        Remote matmul: W[layer][weight] @ x

        Args:
            layer_idx: which layer (0-27)
            weight_id: which weight (WID_Q, WID_K, etc.)
            x_signs: int8, shape (batch, in_features)
            x_exps: int16, shape (batch, in_features)

        Returns:
            (out_signs, out_exps): shape (batch, out_features)
        """
        self._send_request(OP_MATMUL, layer_idx, weight_id, x_signs, x_exps)
        _, out_s, out_e = self._recv_response()
        return out_s, out_e

    def rms_norm(self, layer_idx: int, norm_id: int,
                 x_signs: np.ndarray, x_exps: np.ndarray) -> tuple:
        """
        Remote RMS norm.
        norm_id: 0=input_layernorm, 1=post_attention_layernorm
        """
        self._send_request(OP_RMS_NORM, layer_idx, norm_id, x_signs, x_exps)
        _, out_s, out_e = self._recv_response()
        return out_s, out_e

    def silu(self, x_signs: np.ndarray, x_exps: np.ndarray) -> tuple:
        """Remote SiLU via integer LUT."""
        self._send_request(OP_SILU, 0, 0, x_signs, x_exps)
        _, out_s, out_e = self._recv_response()
        return out_s, out_e

    def multiply(self, a_signs, a_exps, b_signs, b_exps) -> tuple:
        """Remote element-wise multiply."""
        self._send_dual_request(OP_MULTIPLY, 0, 0,
                                a_signs, a_exps, b_signs, b_exps)
        _, out_s, out_e = self._recv_response()
        return out_s, out_e

    def add(self, a_signs, a_exps, b_signs, b_exps) -> tuple:
        """Remote residual add."""
        self._send_dual_request(OP_ADD, 0, 0,
                                a_signs, a_exps, b_signs, b_exps)
        _, out_s, out_e = self._recv_response()
        return out_s, out_e

    def softmax(self, x_signs: np.ndarray, x_exps: np.ndarray) -> tuple:
        """Remote softmax."""
        self._send_request(OP_SOFTMAX, 0, 0, x_signs, x_exps)
        _, out_s, out_e = self._recv_response()
        return out_s, out_e

    def full_layer(self, layer_idx: int,
                   h_signs: np.ndarray, h_exps: np.ndarray) -> tuple:
        """
        Run a complete transformer layer on the remote node.

        Args:
            layer_idx: which layer (0-27)
            h_signs: int8, shape (1, seq_len, hidden_dim) or (seq_len, hidden_dim)
            h_exps: int16, same shape

        Returns:
            (out_signs, out_exps): same shape as input
        """
        # Strip batch dim if present — protocol sends (seq_len, hidden_dim)
        squeeze = False
        if h_signs.ndim == 3:
            squeeze = True
            h_signs = h_signs[0]
            h_exps = h_exps[0]

        self._send_request(OP_FULL_LAYER, layer_idx, 0, h_signs, h_exps)
        _, out_s, out_e = self._recv_response()

        if squeeze:
            out_s = out_s[np.newaxis, :, :]
            out_e = out_e[np.newaxis, :, :]

        return out_s, out_e
