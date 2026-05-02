# Doc 252: φ-Compute Protocol — Model-Agnostic Integer Coprocessor

**Date:** February 18, 2026
**Status:** Design
**Prerequisites:** Doc 251 (Distributed Integer Compute), Finding 53 (GPU Acceleration)

## Motivation

The current compute node (gimli) embeds the full transformer architecture:
`integer_forward_layer` with hardcoded head counts, RoPE parameters, causal
masking, and a weight store organized by layer/projection name. The node
**knows** it's running Qwen2-7B.

But profiling (gpu_profile.py) shows that network overhead is <0.1% of compute
time — even with per-operation round trips, it would be only 2.7%. This means
we can decompose `FULL_LAYER` into individual operations with negligible cost.

More importantly: since the φ-coordinate system converts ANY model's weights
and activations into integer arithmetic, the processing node should be
**model-agnostic**. A node that only knows how to do φ-integer math can
process any φ-encoded model — transformers, convnets, diffusion models,
mixture-of-experts, or architectures that don't exist yet.

> **Design principle: The node is a calculator. The controller is the brain.**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CONTROLLER (dev machine, or any orchestrator)                  │
│                                                                 │
│  Knows:                                                         │
│    - Model architecture (transformer, convnet, etc.)            │
│    - Layer structure, head counts, dimensions                   │
│    - Operation sequencing (what feeds into what)                │
│    - Which blobs are weights, which are activations             │
│                                                                 │
│  Compiles model forward pass into PROGRAMS:                     │
│    sequences of φ-integer instructions                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ TCP
                           │ Programs + Data
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  COMPUTE NODE (gimli, or any node)                              │
│                                                                 │
│  Knows:                                                         │
│    - φ-integer arithmetic (matmul, add, multiply, etc.)         │
│    - How to store and retrieve named data blobs                 │
│    - How to execute a sequence of instructions                  │
│                                                                 │
│  Does NOT know:                                                 │
│    - What a "transformer" is                                    │
│    - What a "layer" or "attention head" is                      │
│    - What model is being run                                    │
│    - What the data means                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Representation

All data in the protocol is φ-encoded: pairs of `(signs, exponents)`.

```
φ-TENSOR:
  shape:    tuple of uint32 dimensions
  signs:    int8[]   — flattened, row-major
  exponents: int16[] — flattened, row-major, little-endian
```

No floats cross the wire. No floats exist on the node.

## Protocol

### Packet Format

```
REQUEST HEADER (20 bytes):
┌─────────────────────────────────────────────────┐
│ magic:      4 bytes  "PHC\x00" (phi compute)    │
│ version:    uint8    (protocol version, 1)       │
│ msg_type:   uint8    (see message types)         │
│ flags:      uint16   (reserved)                  │
│ payload_len: uint32  (bytes following header)     │
│ request_id: uint64   (for async/pipelining)       │
└─────────────────────────────────────────────────┘

RESPONSE HEADER (20 bytes):
┌─────────────────────────────────────────────────┐
│ magic:      4 bytes  "PHR\x00" (phi response)   │
│ version:    uint8    (protocol version, 1)       │
│ status:     uint8    (0=OK, 1=ERR)               │
│ flags:      uint16   (reserved)                  │
│ payload_len: uint32  (bytes following header)     │
│ request_id: uint64   (echo from request)          │
└─────────────────────────────────────────────────┘
```

### Message Types

| Code | Name | Description |
|:----:|------|-------------|
| 0x01 | STORE | Store a named data blob on the node |
| 0x02 | DROP | Free a stored blob |
| 0x03 | LIST | List stored blobs (names + shapes) |
| 0x04 | EXEC | Execute a single operation |
| 0x05 | PROGRAM | Execute a sequence of operations |
| 0xFF | PING | Health check / capabilities query |

### Storage Operations

Blobs are identified by a **blob_id** (uint64). The controller assigns IDs.
The node doesn't know or care what the data represents.

```
STORE payload:
  blob_id:    uint64
  n_dims:     uint8
  shape:      uint32[n_dims]
  compressed: uint8    (0=raw signs+exps, 1=compressed)
  data:       [signs + exps] or [signs + quant + row_min + row_max]

DROP payload:
  blob_id:    uint64

LIST response payload:
  n_blobs:    uint32
  entries:    [blob_id(u64) + n_dims(u8) + shape(u32[]) + bytes(u64)]...
```

### Instruction Set

Each instruction is a fixed-size record:

```
INSTRUCTION (16 bytes):
┌────────────────────────────────────────────────┐
│ opcode:   uint8                                │
│ dst:      uint8    (destination register 0-63) │
│ src_a:    uint8    (source register or 0xFF)   │
│ src_b:    uint8    (source register or 0xFF)   │
│ blob_ref: uint64   (blob_id, or 0)             │
│ param:    int32    (op-specific parameter)      │
└────────────────────────────────────────────────┘
```

#### Opcodes

| Code | Mnemonic | Operation | src_a | src_b | blob_ref | param |
|:----:|----------|-----------|-------|-------|----------|-------|
| 0x01 | MATMUL | dst = blob @ src_a | input | — | weight blob | — |
| 0x02 | ADD | dst = src_a + src_b | operand A | operand B | — | — |
| 0x03 | MUL | dst = src_a × src_b | operand A | operand B | — | — |
| 0x04 | RMS_NORM | dst = rms_norm(src_a, blob) | input | — | norm weight | dim |
| 0x05 | SILU | dst = silu(src_a) | input | — | — | — |
| 0x06 | SOFTMAX | dst = softmax(src_a, axis) | input | — | — | axis |
| 0x07 | SCALE | dst = scale(src_a, exp) | input | — | — | exp_offset |
| 0x08 | EINSUM_QK | dst = batched Q@K^T | Q | K | — | — |
| 0x09 | EINSUM_AV | dst = batched attn@V | attn | V | — | — |
| 0x0A | ROPE | dst = rope(src_a, freqs) | input | — | freq blob | — |
| 0x10 | RESHAPE | dst = reshape(src_a) | input | — | — | shape_ref |
| 0x11 | TRANSPOSE | dst = transpose(src_a) | input | — | — | axes_ref |
| 0x12 | REPEAT | dst = repeat(src_a) | input | — | — | packed(axis,n) |
| 0x13 | SLICE | dst = src_a[start:end] | input | — | — | packed params |
| 0x14 | BROADCAST_ADD | dst = src_a + blob (broadcast) | input | — | bias blob | — |
| 0x15 | CAUSAL_MASK | dst = apply_mask(src_a) | input | — | — | mask_val |
| 0x1F | COPY | dst = src_a | input | — | — | — |

**Note:** Reshape, transpose, and repeat parameters that exceed 4 bytes
use a **shape descriptor** appended after the instruction block (referenced
by `param` as an index into the shape table).

### Single Operation (EXEC)

For simple cases or debugging — one instruction, input data on the wire,
result returned immediately:

```
EXEC payload:
  instruction:  16 bytes (one instruction record)
  n_inputs:     uint8 (0, 1, or 2 — data loaded into src registers)
  for each input:
    register:   uint8
    n_dims:     uint8
    shape:      uint32[n_dims]
    data:       signs + exps
```

Response: the destination register contents.

### Program Execution (PROGRAM)

A batch of instructions with register management. Intermediates stay on
the node — only requested outputs travel back.

```
PROGRAM payload:
  n_instructions: uint16
  n_inputs:       uint8
  n_outputs:      uint8
  n_shapes:       uint8  (shape descriptors for reshape/transpose)

  shape_table:    [n_dims(u8) + dims(u32[])]... (n_shapes entries)
  instructions:   [16 bytes]... (n_instructions entries)

  inputs:         for each input:
                    register: uint8
                    n_dims:   uint8
                    shape:    uint32[n_dims]
                    data:     signs + exps

  output_regs:    uint8[n_outputs] (which registers to return)
```

Response: the requested output registers, in order.

### Registers

The node provides **64 registers** (R0–R63). Each holds a φ-tensor of
arbitrary shape. Registers are:
- Allocated on first write
- Freed when overwritten or at program end
- Scoped to the current PROGRAM (not persistent across calls)

## Example: One Transformer Layer as a Program

The controller compiles a transformer layer into ~25 instructions. The node
executes them without knowing what a transformer is.

```
# Blob IDs (pre-stored by controller during setup):
#   100 = norm_weight_input    104 = W_o        108 = W_down
#   101 = W_q                  105 = W_gate     109 = bias_q
#   102 = W_k                  106 = W_up       110 = bias_k
#   103 = W_v                  107 = norm_post   111 = bias_v
#                                                112 = rope_freqs

# Input: R0 = hidden state (seq_len, hidden_dim)
# Instructions:
RMS_NORM     R1,  R0,  —,   blob=100,  dim=3584    # pre-attn norm
MATMUL       R2,  R1,  —,   blob=101               # Q projection
MATMUL       R3,  R1,  —,   blob=102               # K projection
MATMUL       R4,  R1,  —,   blob=103               # V projection
BROADCAST_ADD R2, R2,  —,   blob=109               # Q + bias
BROADCAST_ADD R3, R3,  —,   blob=110               # K + bias
BROADCAST_ADD R4, R4,  —,   blob=111               # V + bias
RESHAPE      R2,  R2,  —,   shape=0                # (seq, heads, dim)
RESHAPE      R3,  R3,  —,   shape=1                # (seq, kv_heads, dim)
RESHAPE      R4,  R4,  —,   shape=1
TRANSPOSE    R2,  R2,  —,   axes=0                 # (heads, seq, dim)
TRANSPOSE    R3,  R3,  —,   axes=1
TRANSPOSE    R4,  R4,  —,   axes=1
ROPE         R2,  R2,  —,   blob=112               # RoPE on Q
ROPE         R3,  R3,  —,   blob=112               # RoPE on K
REPEAT       R3,  R3,  —,   param=(axis=1,n=7)     # GQA expand K
REPEAT       R4,  R4,  —,   param=(axis=1,n=7)     # GQA expand V
EINSUM_QK    R5,  R2,  R3                           # scores = Q @ K^T
SCALE        R5,  R5,  —,   param=-725              # / sqrt(d)
CAUSAL_MASK  R5,  R5,  —,   param=-30000            # mask future
SOFTMAX      R5,  R5,  —,   param=-1                # softmax
EINSUM_AV    R6,  R5,  R4                           # context = attn @ V
TRANSPOSE    R6,  R6,  —,   axes=2                  # (seq, heads, dim)
RESHAPE      R6,  R6,  —,   shape=2                 # (seq, hidden_dim)
MATMUL       R7,  R6,  —,   blob=104               # O projection
ADD          R0,  R0,  R7                           # residual
RMS_NORM     R1,  R0,  —,   blob=107,  dim=3584    # pre-MLP norm
MATMUL       R2,  R1,  —,   blob=105               # gate projection
MATMUL       R3,  R1,  —,   blob=106               # up projection
SILU         R2,  R2                                # silu(gate)
MUL          R4,  R2,  R3                           # gate × up
MATMUL       R5,  R4,  —,   blob=108               # down projection
ADD          R0,  R0,  R5                           # residual
# Output: R0
```

**32 instructions. The node sees only opcodes and register numbers.**
It has no idea this is attention + MLP. It would execute a convolution
or a diffusion step with exactly the same instruction set.

## Controller Compilation

The controller holds the model graph and compiles it to programs:

```python
class PhiProgramCompiler:
    """Compiles a model's forward pass into φ-compute programs."""

    def compile_layer(self, layer_config, blob_map):
        """
        Args:
            layer_config: model-specific layer description
            blob_map: {semantic_name: blob_id} mapping
        Returns:
            Program (list of instructions + I/O spec)
        """
        # Example for transformer layer:
        program = Program()
        r = program.allocator  # register allocator

        h = r.input(0)  # R0 = hidden state
        normed = r.alloc()
        program.add(RMS_NORM, dst=normed, src=h,
                     blob=blob_map['norm_input'], param=layer_config.hidden_dim)
        # ... etc
        return program
```

The compiler is the ONLY place that knows model architecture.
Different models → different compilers → same nodes.

## Blob Lifecycle

```
SETUP PHASE (once per model load):
  Controller assigns blob_ids (sequential uint64)
  Controller sends STORE for each weight matrix
  Node stores blobs in RAM (compressed ok)
  ~13 GB for Qwen2-7B (28 layers × 466 MB)

INFERENCE PHASE (per request):
  Controller sends PROGRAM with input data
  Node executes instructions, referencing stored blobs
  Node returns output registers
  Node frees all registers

TEARDOWN (optional):
  Controller sends DROP for each blob
  Or: node clears all blobs on disconnect
```

## GPU Acceleration (Transparent)

The node decides internally whether to use CPU or GPU for each operation.
The protocol doesn't mention GPU. This is an implementation detail:

- Node with GPU: transfers blobs to VRAM as needed, caches hot blobs
- Node without GPU: runs on CPU (slower but correct)
- Node with FPGA: could implement the same opcodes in hardware
- Multiple nodes: controller sends different programs to different nodes

The controller doesn't know or care about the node's hardware.

## Performance Budget

Based on profiling (gpu_profile.py, batch=5, seq=5):

```
PROGRAM mode (one round trip per layer):
  Network:          ~0.4 ms   (108 KB activations, 2.5GbE)
  GPU matmul:     ~1089 ms   (7 projections)
  Weight transfer:  ~176 ms   (466 MB CPU→GPU)
  Non-matmul:       ~144 ms   (CPU: SiLU 38ms, softmax 45ms, etc.)
  Overhead:        ~1409 ms   total

EXEC mode (one round trip per operation):
  Network:         ~34 ms    (20 round trips × ~1ms + ~3 MB data)
  GPU matmul:     ~1089 ms
  Weight transfer:  ~176 ms
  Non-matmul:       ~144 ms
  Overhead:        ~1443 ms   total (+2.4% vs PROGRAM)
```

PROGRAM mode is preferred for production (saves ~34ms/layer), but EXEC
mode is valuable for debugging and interactive exploration.

## Connection to the Hypothesis

> **"Structure IS information. Geometry IS computation."**

The φ-compute protocol makes this concrete:

- The **structure** (model architecture) lives in the controller as a program
- The **geometry** (φ-encoded weights and activations) lives on the node as blobs
- The **computation** is a sequence of integer operations on geometric data

The node doesn't need to understand the structure to compute with the geometry.
Any φ-encoded structure can be computed by the same node. The protocol is the
interface between structure and geometry.

This is substrate independence taken to its logical conclusion: not just
"any hardware" but "any model, any architecture, any structure" — as long
as it can be expressed in φ-integer arithmetic.

## Future Extensions

### Streaming Programs
For autoregressive generation, the controller sends a program template
with KV-cache slots. The node executes it repeatedly, updating cache
registers between iterations.

### Node Discovery
Nodes advertise capabilities (VRAM, compute speed, storage) via PING.
Controller routes programs to appropriate nodes.

### Instruction Fusion
Node-side optimization: fuse MATMUL + BROADCAST_ADD into a single
kernel. The protocol stays clean; the optimization is internal.

### Custom Opcodes
Nodes can register custom opcodes (0x80-0xFE) for hardware-specific
optimizations. Controller probes capabilities via PING and uses
custom opcodes when available, falling back to standard ones otherwise.
