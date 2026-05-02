#!/usr/bin/env python3
"""
Phase 2 Verification: Compare PhiQwen2Engine output against HuggingFace.

Tests:
  1. Load φ-encoded model (N layers)
  2. Run forward pass on a test prompt
  3. Compare logits against HuggingFace reference
  4. Report correlation, top-k agreement, prediction match

This script requires the HuggingFace model for comparison.
If not available, it runs a standalone smoke test.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")

# Test with fewer layers first for speed
TEST_LAYERS = 2


def smoke_test():
    """Basic pipeline test — no reference model needed."""
    print("=" * 70)
    print("  Phase 2 Verification: Smoke Test")
    print("=" * 70)
    print()

    # Load engine
    engine = PhiQwen2Engine.load(MODEL_DIR, max_layers=TEST_LAYERS)
    print()

    # Test forward pass with a few tokens
    # Token 9707 = "Hello", 220 = " "  (approximate — we just need any valid IDs)
    test_tokens = [9707, 220, 279]
    print(f"Test tokens: {test_tokens}")
    print(f"Running forward pass ({TEST_LAYERS} layers)...")

    t0 = time.time()
    logits = engine.forward(test_tokens, verbose=True)
    dt = time.time() - t0

    print(f"\nLogits shape: {logits.shape}")
    print(f"Expected:     (1, {len(test_tokens)}, {engine.vocab_size})")
    assert logits.shape == (1, len(test_tokens), engine.vocab_size), \
        f"Shape mismatch: {logits.shape}"

    # Basic sanity checks
    last_logits = logits[0, -1, :]
    print(f"\nLast-token logits stats:")
    print(f"  min={last_logits.min():.4f}  max={last_logits.max():.4f}  "
          f"mean={last_logits.mean():.4f}  std={last_logits.std():.4f}")

    # Check for NaN/Inf
    assert not np.any(np.isnan(logits)), "NaN in logits!"
    assert not np.any(np.isinf(logits)), "Inf in logits!"

    # Check that logits are not all the same (would indicate broken pipeline)
    assert last_logits.std() > 0.01, f"Logits have near-zero variance: {last_logits.std()}"

    # Top-5 predictions
    top5_idx = np.argsort(last_logits)[-5:][::-1]
    print(f"\nTop-5 predicted tokens: {top5_idx.tolist()}")
    print(f"Top-5 logit values:    {[f'{last_logits[i]:.3f}' for i in top5_idx]}")

    print(f"\nForward pass time: {dt:.2f}s for {len(test_tokens)} tokens, "
          f"{TEST_LAYERS} layers")
    print(f"Per-layer time: {dt/TEST_LAYERS:.2f}s")

    print("\n✓ Smoke test PASSED — pipeline produces valid logits")
    return engine


def reference_comparison(engine):
    """Compare against HuggingFace reference model."""
    print()
    print("=" * 70)
    print("  Phase 2 Verification: HuggingFace Comparison")
    print("=" * 70)
    print()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  HuggingFace transformers not available — skipping reference test")
        return

    # Load tokenizer
    model_name = "Qwen/Qwen2-7B"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors='pt')
    token_ids = inputs['input_ids'][0].tolist()
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {token_ids}")
    print()

    # φ-engine forward pass
    print(f"φ-engine forward pass ({TEST_LAYERS} layers)...")
    t0 = time.time()
    phi_logits = engine.forward(token_ids, verbose=True)
    phi_time = time.time() - t0
    print(f"  Time: {phi_time:.2f}s")

    # HuggingFace forward pass (same number of layers)
    print(f"\nHuggingFace forward pass ({TEST_LAYERS} layers)...")
    print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='cpu',
        num_hidden_layers=TEST_LAYERS,
    )
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        hf_outputs = model(inputs['input_ids'])
    hf_time = time.time() - t0
    hf_logits = hf_outputs.logits[0].numpy()  # (seq_len, vocab_size)
    print(f"  Time: {hf_time:.2f}s")

    # Compare logits for each position
    print(f"\n{'Pos':>4s}  {'Corr':>8s}  {'Top1 Match':>10s}  {'Top10 Agree':>11s}  "
          f"{'φ-top1':>8s}  {'HF-top1':>8s}")
    print("  " + "-" * 65)

    phi_last = phi_logits[0]  # (seq_len, vocab_size)

    for pos in range(len(token_ids)):
        phi_pos = phi_last[pos]
        hf_pos = hf_logits[pos]

        # Correlation
        corr = np.corrcoef(phi_pos, hf_pos)[0, 1]

        # Top-1
        phi_top1 = int(np.argmax(phi_pos))
        hf_top1 = int(np.argmax(hf_pos))
        match = "✓" if phi_top1 == hf_top1 else "✗"

        # Top-10 agreement
        phi_top10 = set(np.argsort(phi_pos)[-10:])
        hf_top10 = set(np.argsort(hf_pos)[-10:])
        agree = len(phi_top10 & hf_top10) / 10

        print(f"  {pos:3d}  {corr:8.6f}  {match:>10s}  {agree:10.0%}  "
              f"{phi_top1:>8d}  {hf_top1:>8d}")

    # Summary for last position
    phi_final = phi_last[-1]
    hf_final = hf_logits[-1]
    final_corr = np.corrcoef(phi_final, hf_final)[0, 1]

    print(f"\nLast-position logits correlation: {final_corr:.6f}")

    # Decode predictions
    phi_pred = int(np.argmax(phi_final))
    hf_pred = int(np.argmax(hf_final))
    print(f"φ-engine prediction: '{tokenizer.decode([phi_pred])}' (id={phi_pred})")
    print(f"HF prediction:       '{tokenizer.decode([hf_pred])}' (id={hf_pred})")
    print(f"Match: {'✓ YES' if phi_pred == hf_pred else '✗ NO'}")

    # Clean up
    del model

    if final_corr > 0.99:
        print(f"\n✓ EXCELLENT: {final_corr:.6f} correlation (>0.99)")
    elif final_corr > 0.95:
        print(f"\n✓ GOOD: {final_corr:.6f} correlation (>0.95)")
    elif final_corr > 0.90:
        print(f"\n~ FAIR: {final_corr:.6f} correlation (>0.90)")
    else:
        print(f"\n✗ LOW: {final_corr:.6f} correlation — investigate!")


def main():
    engine = smoke_test()
    reference_comparison(engine)


if __name__ == '__main__':
    main()
