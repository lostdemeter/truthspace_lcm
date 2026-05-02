# Doc 251: Distributed Integer Compute — Phase 7

**Date:** February 18, 2026  
**Status:** Phase 7c Complete  
**Prerequisites:** Doc 250 (Integer Geometric Pipeline), Finding 50 (23/23 ops integer)

## Motivation

Finding 50 proved that the entire Qwen2-7B forward pass can be computed using
only integer arithmetic: sign XOR, exponent ADD, LUT lookup, int64 accumulation.
No IEEE float multiply or divide.

This means the compute can run on **anything that does integer math** — including
remote machines with no GPU, FPGAs, microcontrollers, or clusters of commodity
hardware. Phase 7 proves this by offloading the integer operations to a remote
machine over the network.

## Architecture

### Principle: Weights Stay, Activations Travel

The φ-encoded model weights (12.75 GB) are **static**. They are pre-loaded on
each compute node at startup. Only activations (signs + exponents) travel over
the network per inference request.

For a typical prompt (seq_len=5, hidden_dim=3584):
- Input activations: 5 × 3584 × 3 bytes (int8 sign + int16 exp) = **53.8 KB**
- Output activations: same size
- Round-trip: **~108 KB** at 2.5GbE ≈ 0.4 ms

Network overhead is negligible compared to compute time.

### Network Protocol

Raw TCP. No external dependencies. Binary packets:

```
REQUEST PACKET:
┌──────────────────────────────────────────────────┐
│ Header (16 bytes)                                │
│   magic:      4 bytes  "PHI\x00"                 │
│   op_type:    uint8    (see operation table)      │
│   layer_idx:  uint8    (0-27)                     │
│   weight_id:  uint8    (0=Wq,1=Wk,...6=Wdown)    │
│   flags:      uint8    (reserved)                 │
│   n_rows:     uint32   (rows in activation)       │
│   n_cols:     uint32   (cols in activation)       │
├──────────────────────────────────────────────────┤
│ Payload                                          │
│   signs:      int8[n_rows × n_cols]               │
│   exponents:  int16[n_rows × n_cols]  (LE)        │
└──────────────────────────────────────────────────┘

RESPONSE PACKET:
┌──────────────────────────────────────────────────┐
│ Header (16 bytes)                                │
│   magic:      4 bytes  "PHR\x00"  (phi response) │
│   status:     uint8    (0=OK, 1=ERR)              │
│   reserved:   3 bytes                             │
│   n_rows:     uint32   (rows in result)           │
│   n_cols:     uint32   (cols in result)            │
├──────────────────────────────────────────────────┤
│ Payload                                          │
│   signs:      int8[n_rows × n_cols]               │
│   exponents:  int16[n_rows × n_cols]  (LE)        │
└──────────────────────────────────────────────────┘
```

### Operation Types

| Code | Operation | Inputs | What node does |
|:----:|-----------|--------|----------------|
| 0x01 | MATMUL | activations + weight_id | phi_matmul_integer(W[layer][weight_id], x) |
| 0x02 | RMS_NORM | activations + weight_id | phi_rms_norm_int(x, norm_weight[layer]) |
| 0x03 | SILU | activations | phi_silu_int(x) via LUT |
| 0x04 | MULTIPLY | two activation arrays | phi_multiply_int(a, b) |
| 0x05 | ADD | two activation arrays | phi_add_encoded(a, b) |
| 0x06 | SOFTMAX | activations | phi_softmax_full_int(x) |
| 0x07 | ROPE | activations + position | phi_rope_int(x, pos) |
| 0x08 | EINSUM_QK | Q + K activations | phi_einsum_qk_int(Q, K) |
| 0x09 | EINSUM_AV | attn + V activations | phi_einsum_av_int(attn, V) |
| 0x0A | FULL_LAYER | activations + layer_idx | Run entire layer (all ops) |
| 0xFF | PING | (empty) | Return status + node info |

Phase 7a implements 0x01 (MATMUL) and 0xFF (PING) only.
Phase 7b adds 0x0A (FULL_LAYER).
All operation types implemented on gimli.

## Compute Nodes

### Node: gimli (192.168.1.111)

| Spec | Value |
|------|-------|
| CPU | Intel Core i7-6700 @ 3.40GHz (4C/8T) |
| GPU | NVIDIA RTX 3050 (6 GB VRAM) |
| RAM | 16 GB |
| Storage | 1 TB NVMe |
| Network | 2.5 GbE |
| OS | Linux Mint |

Model weights compressed to 13.06 GB via per-row uint8 quantization (Finding 52).
GPU acceleration via CuPy: 9.1× matmul speedup, ~12× full-layer speedup (Finding 53).

### Node: development machine (controller)

Runs the inference orchestrator. Holds the model in memory. Dispatches
operations to compute nodes. Collects results and assembles the output.

## Phased Implementation

### Phase 7a: Single Operation Proof ✓ COMPLETE (Finding 51)

**Result:** 7/7 operations EXACT MATCH (bit-identical to local).
Network overhead: 0.4 ms vs 2.5 s compute = 0.02%.

### Phase 7b: Full Layer Remote ✓ COMPLETE (Finding 52)

**Result:** All 28 layers on gimli, 5/5 predictions match float baseline.
Required weight compression: per-row uint8 quantization (1.50×, 0.99991 corr).
13.06 GB in RAM on 16 GB gimli.

### Phase 7c: GPU Acceleration ✓ COMPLETE (Finding 53)

**Result:** CuPy-based GPU acceleration of φ-integer matmul on gimli's RTX 3050.
9.1× isolated matmul speedup, ~12× full-layer speedup (1.1s vs ~13s per layer).
100% bit-identical to CPU. 5/5 prompts correct with GPU enabled.

Key implementation details:
- Custom CuPy kernel (not standard GEMM): sign XOR + exp ADD + LUT + int64 accumulation
- Per-call weight transfer (13 GB weights won't fit 6 GB VRAM)
- Dynamic chunk sizing to keep peak VRAM under 1.5 GB
- Aggressive `del` of intermediates + `free_all_blocks()` to prevent OOM
- Files: `gimli:~/truthspace-node/phi_gpu.py`, `gpu_benchmark.py`

### Phase 7d: Multi-Node Pipeline Parallel (FUTURE)

**Goal:** Split layers across multiple machines.

- Node 1: layers 0-13 (~6.5 GB weights)
- Node 2: layers 14-27 (~6.5 GB weights)
- Controller: embedding, final norm, argmax, orchestration
- Hidden state flows: controller → node 1 → node 2 → controller

### Phase 7e: Bitpacking

**Goal:** Minimize network bandwidth.

- Signs: 1 bit each (pack 8 per byte) → 8× compression
- Exponents: variable-width encoding (most are in a narrow range)
- Target: ~1 byte per parameter instead of 3

## Connection to the Hypothesis

> **"Structure IS information. Geometry IS computation."**

If integer geometric computation can travel over a network and produce correct
results on a remote machine, we demonstrate that the geometric structure is
**substrate-independent**. The shape computes regardless of WHERE the integers
are crunched. This is a step toward proving that the φ-lattice is a universal
computational substrate — not tied to any particular hardware, precision format,
or physical location.
