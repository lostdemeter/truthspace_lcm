#!/usr/bin/env python3
"""
phi-generate: Text generation with phi-encoded Qwen2-7B.

No GPU. No PyTorch. Pure NumPy with phi-integer weights.

Usage:
    python phi_generate.py "The capital of France is"
    python phi_generate.py "1 + 1 =" --max-tokens 10
    python phi_generate.py "Hello" --layers 2 --verbose
"""

import argparse
import sys
import time
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments", "model_reverse_engineering_v2", "phi_model"
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with phi-encoded Qwen2-7B (CPU-only, no GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "The capital of France is"
  %(prog)s "Once upon a time" --max-tokens 50
  %(prog)s "Hello" --layers 2 --verbose
  %(prog)s "1 + 1 =" --max-tokens 5
        """,
    )
    parser.add_argument("prompt", help="Text prompt to complete")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate (default: 20)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers to use (default: all 28)")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to phi_model directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-token timing")
    parser.add_argument("--warm", action="store_true",
                        help="Pre-decode all weights (faster generation, +28GB RAM)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Print all output at once instead of streaming")

    args = parser.parse_args()

    # Load tokenizer
    try:
        tokenizer = Qwen2Tokenizer()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        print("Error: prompt encoded to empty token list", file=sys.stderr)
        sys.exit(1)

    # Load engine
    n_layers = args.layers if args.layers else None
    engine = PhiQwen2Engine.load(args.model_dir, max_layers=n_layers)

    # Optional: pre-decode weights for faster generation
    if args.warm:
        engine.warm_weights()

    print()
    layer_desc = f"{n_layers}" if n_layers else "28"
    print(f"  Prompt:     \"{args.prompt}\"")
    print(f"  Tokens:     {prompt_ids} ({len(prompt_ids)} tokens)")
    print(f"  Layers:     {layer_desc}")
    print(f"  Max new:    {args.max_tokens}")
    print()

    # Stream or batch output
    if args.no_stream:
        # Batch mode: generate all, then print
        t0 = time.time()
        output_ids = engine.generate(
            prompt_ids,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        total_time = time.time() - t0

        generated_ids = output_ids[len(prompt_ids):]
        generated_text = tokenizer.decode(generated_ids)
        full_text = args.prompt + generated_text

        print(f"  Output: {full_text}")
        print()
        print(f"  Generated {len(generated_ids)} tokens in {total_time:.1f}s")
        if len(generated_ids) > 0:
            prefill_est = total_time * len(prompt_ids) / (len(prompt_ids) + len(generated_ids))
            decode_est = total_time - prefill_est
            if len(generated_ids) > 1:
                print(f"  Decode speed: ~{decode_est / len(generated_ids):.1f}s/token")

    else:
        # Streaming mode: print tokens as they arrive
        sys.stdout.write(f"  {args.prompt}")
        sys.stdout.flush()

        token_times = []

        def on_token(step, token_id):
            token_str = tokenizer.decode_token(token_id)
            sys.stdout.write(token_str)
            sys.stdout.flush()
            token_times.append(time.time())

        t0 = time.time()
        output_ids = engine.generate(
            prompt_ids,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose,
            token_callback=on_token,
        )
        total_time = time.time() - t0

        sys.stdout.write("\n\n")

        generated_ids = output_ids[len(prompt_ids):]
        n_gen = len(generated_ids)

        print(f"  Prompt: {len(prompt_ids)} tokens")
        print(f"  Generated: {n_gen} tokens in {total_time:.1f}s")

        if n_gen > 0 and len(token_times) > 0:
            prefill_time = token_times[0] - t0
            print(f"  Prefill: {prefill_time:.1f}s ({prefill_time/len(prompt_ids):.2f}s/tok)")
            if n_gen > 1:
                decode_time = total_time - prefill_time
                print(f"  Decode:  {decode_time:.1f}s ({decode_time/(n_gen-1):.2f}s/tok)")

    # Cleanup
    engine.clear_weight_cache()


if __name__ == '__main__':
    main()
