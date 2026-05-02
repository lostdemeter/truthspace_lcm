#!/usr/bin/env python3
"""
Phase 3: Verify KV cache correctness.

Tests:
  1. Run full forward (no cache) for a 5-token sequence → get logits
  2. Run prefill (4 tokens with cache) + decode (1 token with cache) → get logits
  3. Compare: the last-position logits must be identical

This validates that:
  - KV cache stores the correct K/V tensors
  - RoPE positions are correct with seq_offset
  - Incremental decode matches full forward
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.phi_attention import KVCache

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")
N_LAYERS = 2


def main():
    print("=" * 70)
    print("  Phase 3: KV Cache Correctness Verification")
    print("=" * 70)
    print()

    # Load engine
    engine = PhiQwen2Engine.load(MODEL_DIR, max_layers=N_LAYERS, verbose=False)
    print(f"Loaded {N_LAYERS}-layer engine\n")

    tokens = [9707, 220, 279, 374, 220]  # 5 tokens

    # ─── Test 1: Full forward (no cache) ───
    print("Test 1: Full forward (no cache), 5 tokens")
    logits_full = engine.forward(tokens)
    print(f"  Shape: {logits_full.shape}")
    full_last = logits_full[0, -1, :]

    # ─── Test 2: Prefill 4 + decode 1 (with cache) ───
    print("\nTest 2: Prefill 4 tokens + decode 1 token (with KV cache)")
    kv_cache = KVCache(N_LAYERS, engine.num_kv_heads, engine.head_dim)

    # Prefill: first 4 tokens
    logits_prefill = engine.forward(tokens[:4], kv_cache=kv_cache)
    print(f"  Prefill shape: {logits_prefill.shape}")
    print(f"  KV cache seq_len after prefill: {kv_cache.seq_len}")

    # Decode: 5th token only
    logits_decode = engine.forward([tokens[4]], kv_cache=kv_cache)
    print(f"  Decode shape: {logits_decode.shape}")
    print(f"  KV cache seq_len after decode: {kv_cache.seq_len}")
    cached_last = logits_decode[0, -1, :]

    # ─── Compare ───
    print("\n" + "─" * 70)
    print("  Comparison: full forward vs prefill+decode")
    print("─" * 70)

    corr = np.corrcoef(full_last, cached_last)[0, 1]
    max_diff = np.max(np.abs(full_last - cached_last))
    mean_diff = np.mean(np.abs(full_last - cached_last))

    full_top1 = int(np.argmax(full_last))
    cached_top1 = int(np.argmax(cached_last))
    match = full_top1 == cached_top1

    full_top10 = set(np.argsort(full_last)[-10:])
    cached_top10 = set(np.argsort(cached_last)[-10:])
    top10_agree = len(full_top10 & cached_top10) / 10

    print(f"  Correlation:      {corr:.10f}")
    print(f"  Max abs diff:     {max_diff:.2e}")
    print(f"  Mean abs diff:    {mean_diff:.2e}")
    print(f"  Top-1 match:      {'✓' if match else '✗'} (full={full_top1}, cached={cached_top1})")
    print(f"  Top-10 agreement: {top10_agree:.0%}")

    # ─── Test 3: Multi-step decode (simulate generation) ───
    print("\n" + "─" * 70)
    print("  Test 3: Multi-step incremental decode")
    print("─" * 70)

    # Full forward for all 5 tokens → logits at each position
    logits_all = engine.forward(tokens)

    # Incremental: add tokens one by one
    kv_cache2 = KVCache(N_LAYERS, engine.num_kv_heads, engine.head_dim)
    all_ok = True

    for i in range(len(tokens)):
        logits_inc = engine.forward([tokens[i]], kv_cache=kv_cache2)
        inc_last = logits_inc[0, -1, :]
        ref_last = logits_all[0, i, :]

        corr_i = np.corrcoef(inc_last, ref_last)[0, 1]
        max_diff_i = np.max(np.abs(inc_last - ref_last))
        match_i = int(np.argmax(inc_last)) == int(np.argmax(ref_last))

        status = "✓" if corr_i > 0.9999 and match_i else "✗"
        print(f"  Step {i}: corr={corr_i:.10f}  max_diff={max_diff_i:.2e}  "
              f"top1_match={'✓' if match_i else '✗'}  {status}")

        if not match_i or corr_i < 0.999:
            all_ok = False

    # ─── Verdict ───
    print()
    if corr > 0.9999 and match and all_ok:
        print("  ✓ KV CACHE VERIFIED: Incremental decode matches full forward")
    elif corr > 0.99:
        print("  ~ KV CACHE CLOSE: Minor numerical differences (investigate)")
    else:
        print("  ✗ KV CACHE MISMATCH: Significant divergence detected")


if __name__ == '__main__':
    main()
