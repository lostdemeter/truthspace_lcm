#!/usr/bin/env python3
"""
Phase 2: Full 28-layer forward pass through φ-encoded Qwen2-7B.

This is the definitive test: run ALL 28 layers on a real prompt
and see what the model predicts. No GPU. No PyTorch. Pure NumPy
with φ-encoded weights.

Expected timing: ~2-4 minutes for a short prompt on CPU.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")


def main():
    print("=" * 70)
    print("  Phase 2: Full 28-Layer Forward Pass")
    print("  φ-Encoded Qwen2-7B — No GPU, No PyTorch")
    print("=" * 70)
    print()

    # Load all 28 layers
    t_load_start = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, max_layers=28)
    t_load = time.time() - t_load_start
    print(f"\nTotal load time: {t_load:.1f}s")

    # Try to load tokenizer for readable output
    tokenizer = None
    try:
        import json
        for candidate in [
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots"),
        ]:
            if os.path.exists(candidate):
                snapshots = os.listdir(candidate)
                if snapshots:
                    vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                    if os.path.exists(vocab_file):
                        with open(vocab_file, 'r') as f:
                            tokenizer_data = json.load(f)
                        vocab = tokenizer_data.get('model', {}).get('vocab', {})
                        # Build id→token map
                        id_to_token = {}
                        for tok, idx in vocab.items():
                            id_to_token[idx] = tok
                        print(f"  Loaded tokenizer ({len(id_to_token)} tokens)")
                        break
    except Exception as e:
        print(f"  Tokenizer not available: {e}")

    def decode_token(tok_id):
        if id_to_token:
            raw = id_to_token.get(tok_id, f"[{tok_id}]")
            # Clean up Ġ (space prefix) for readability
            return raw.replace('Ġ', ' ').replace('Ċ', '\n')
        return f"[{tok_id}]"

    # Test prompts — use raw token IDs that we know work
    # These are approximate but any valid token IDs will work
    test_prompts = [
        # "The capital of France is" (approximate token IDs for Qwen2)
        {"name": "Geography", "tokens": [785, 6864, 315, 9822, 374]},
        # "1 + 1 =" (simple math)
        {"name": "Math", "tokens": [16, 488, 220, 16, 284]},
    ]

    for prompt_info in test_prompts:
        name = prompt_info["name"]
        token_ids = prompt_info["tokens"]

        prompt_text = ''.join(decode_token(t) for t in token_ids)
        print(f"\n{'─' * 70}")
        print(f"  Prompt: '{prompt_text}'")
        print(f"  Tokens: {token_ids}")
        print(f"{'─' * 70}")

        # Forward pass
        print(f"\n  Running 28-layer forward pass...")
        t0 = time.time()
        logits = engine.forward(token_ids, verbose=True)
        total_time = time.time() - t0

        # Results
        last_logits = logits[0, -1, :]
        print(f"\n  Total time: {total_time:.1f}s")
        print(f"  Per-layer: {total_time / 28:.2f}s")
        print(f"  Logits: min={last_logits.min():.2f}  max={last_logits.max():.2f}  "
              f"std={last_logits.std():.2f}")

        # Top-10 predictions
        top10_idx = np.argsort(last_logits)[-10:][::-1]
        print(f"\n  Top-10 predictions:")
        for rank, tok_id in enumerate(top10_idx):
            tok_str = decode_token(tok_id)
            print(f"    {rank+1:2d}. '{tok_str}' (id={tok_id}, logit={last_logits[tok_id]:.3f})")

        # Sanity checks
        assert not np.any(np.isnan(logits)), "NaN detected!"
        assert not np.any(np.isinf(logits)), "Inf detected!"
        assert last_logits.std() > 0.1, "Near-zero variance!"

    print(f"\n{'=' * 70}")
    print(f"  ✓ Full 28-layer forward pass COMPLETE")
    print(f"  No GPU. No PyTorch. Pure φ-encoded NumPy inference.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
