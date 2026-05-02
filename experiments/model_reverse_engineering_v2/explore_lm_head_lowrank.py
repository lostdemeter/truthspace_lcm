"""
LM Head Low-Rank Decomposition — Connecting Doc 198 + Doc 244 to LM Head

The DDColor Jacobian finding (Doc 244) proved that low-rank approximation
of weight matrices can be BETTER than full-rank because it denoises.

Doc 198 showed weight matrices have low effective rank:
  rank 1000 = 81.7% energy, 2× speedup

The LM head is (152064, 3584). SVD decomposition:
  W ≈ U @ diag(S) @ V.T  where U(152064,k), S(k), V(3584,k)

For inference:  logits = h @ W.T = h @ V @ diag(S) @ U.T
  Full:    3584 × 152064 = 545M ops
  k=256:   3584×256 + 256×152064 = 39.8M ops (13.7× fewer)
  k=512:   3584×512 + 512×152064 = 79.7M ops (6.8× fewer)

Critical question: does low-rank preserve argmax?
If yes, we can also offload to gimli (each factor fits in 6GB VRAM).

Usage:
    python explore_lm_head_lowrank.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID
from phi_geometric.inference.phi_integer import float_to_phi, phi_to_float
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

TEST_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The color of the sky is", "blue"),
    ("One plus one equals", "two"),
    ("The chemical symbol for gold is", "Au"),
    ("Water freezes at zero degrees", "Celsius"),
    ("The speed of light is approximately 300000", "km"),
    ("In Python, you print with the", "print"),
    ("The opposite of hot is", "cold"),
    ("The square root of 144 is", "12"),
]


def get_hidden_states(engine, tokenizer, prompts):
    """Run model to get hidden states at last position for each prompt."""
    states = []
    for prompt, _ in prompts:
        tokens = tokenizer.encode(prompt)
        hidden = engine.embedding(tokens)
        hidden = hidden[np.newaxis, :, :]
        for layer in engine.layers:
            hidden = layer(hidden, pure=False)
        hidden = rms_norm(hidden, engine.final_norm_weight)
        states.append(hidden[0, -1, :])  # last position
    return states


def analyze_svd_spectrum(W_decoded):
    """Analyze the singular value spectrum of the LM head."""
    print("=" * 70)
    print("  Part 1: SVD Spectrum of LM Head (152064 × 3584)")
    print("=" * 70)

    # W is (152064, 3584) — more rows than columns
    # SVD: W = U @ diag(S) @ V.T
    # U: (152064, 3584), S: (3584,), V: (3584, 3584)
    print("\n  Computing SVD (this may take a moment)...")
    t0 = time.time()
    # Use economy SVD — only need up to rank 3584
    U, S, Vt = np.linalg.svd(W_decoded, full_matrices=False)
    dt = time.time() - t0
    print(f"  SVD computed in {dt:.1f}s")
    print(f"  U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

    # Energy spectrum
    total_energy = np.sum(S ** 2)
    cumulative_energy = np.cumsum(S ** 2) / total_energy

    print(f"\n  Singular value spectrum:")
    print(f"    S[0] = {S[0]:.1f}")
    print(f"    S[1] = {S[1]:.1f}")
    print(f"    S[0]/S[1] = {S[0]/S[1]:.1f}")
    print(f"    S[-1] = {S[-1]:.4f}")

    ranks = [32, 64, 128, 256, 512, 1024, 2048, 3584]
    print(f"\n  Cumulative energy by rank:")
    for k in ranks:
        if k <= len(S):
            print(f"    rank {k:5d}: {cumulative_energy[k-1]*100:.2f}% energy")

    return U, S, Vt


def test_lowrank_accuracy(W_decoded, U, S, Vt, hidden_states, tokenizer, prompts):
    """Test if low-rank preserves argmax."""
    print("\n" + "=" * 70)
    print("  Part 2: Low-Rank Argmax Preservation")
    print("=" * 70)

    # Ground truth
    true_tops = []
    for h in hidden_states:
        logits = h @ W_decoded.T
        true_tops.append(int(np.argmax(logits)))

    ranks = [32, 64, 128, 256, 512, 1024, 2048]

    for k in ranks:
        # Low-rank approximation: W_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
        # For inference: logits = h @ W_k.T = h @ Vt[:k,:].T @ diag(S[:k]) @ U[:,:k].T
        # = (h @ V[:k].T) @ diag(S[:k]) @ U[:,:k].T
        # Step 1: h @ V[:k].T → (k,)   [cheap: 3584 × k]
        # Step 2: × S[:k]    → (k,)   [trivial]
        # Step 3: @ U[:,:k].T → (152064,) [medium: k × 152064]

        correct = 0
        top5_correct = 0
        max_logit_error = 0
        t0 = time.perf_counter()

        for i, h in enumerate(hidden_states):
            # Low-rank forward
            projected = h @ Vt[:k, :].T  # (k,)
            scaled = projected * S[:k]    # (k,)
            logits_lr = scaled @ U[:, :k].T  # (152064,)

            lr_top = int(np.argmax(logits_lr))
            if lr_top == true_tops[i]:
                correct += 1

            # Top-5 check
            lr_top5 = set(np.argsort(logits_lr)[-5:])
            if true_tops[i] in lr_top5:
                top5_correct += 1

            # Logit correlation
            full_logits = h @ W_decoded.T
            err = np.abs(logits_lr - full_logits).max()
            max_logit_error = max(max_logit_error, err)

        dt = (time.perf_counter() - t0) * 1000 / len(hidden_states)

        n = len(prompts)
        ops_full = 3584 * 152064
        ops_lr = 3584 * k + k * 152064
        speedup = ops_full / ops_lr

        print(f"  rank {k:5d}: top1={correct}/{n}  top5={top5_correct}/{n}  "
              f"max_err={max_logit_error:.2f}  {dt:.0f}ms/prompt  "
              f"ops={ops_lr/1e6:.0f}M ({speedup:.1f}×)")

        # Show mismatches
        if correct < n:
            for i, h in enumerate(hidden_states):
                projected = h @ Vt[:k, :].T
                scaled = projected * S[:k]
                logits_lr = scaled @ U[:, :k].T
                lr_top = int(np.argmax(logits_lr))
                if lr_top != true_tops[i]:
                    true_tok = tokenizer.decode_token(true_tops[i]).strip()
                    lr_tok = tokenizer.decode_token(lr_top).strip()
                    print(f"    ✗ '{prompts[i][0]}': "
                          f"full='{true_tok}' lr='{lr_tok}'")


def test_phi_lowrank(W_phi, U, S, Vt, hidden_states, tokenizer, prompts):
    """Test low-rank with φ-encoded factors."""
    print("\n" + "=" * 70)
    print("  Part 3: φ-Encoded Low-Rank Factors")
    print("=" * 70)

    # Can we φ-encode U and V separately?
    # This is the key to offloading: store φ-encoded factors on gimli
    # The matmul opcode handles φ-encoded weights natively

    true_tops = []
    W_decoded = W_phi.decode_cached()
    for h in hidden_states:
        logits = h @ W_decoded.T
        true_tops.append(int(np.argmax(logits)))

    for k in [256, 512, 1024]:
        # Factor 1: V_k = Vt[:k, :].T → (3584, k)
        # Factor 2: US_k = U[:, :k] @ diag(S[:k]) → (152064, k)
        # Combined: W_k.T = V_k @ US_k.T
        # Inference: logits = (h @ V_k) @ US_k.T

        V_k = Vt[:k, :].T  # (3584, k)
        US_k = U[:, :k] * S[:k]  # (152064, k) — absorb S into U

        # φ-encode the factors
        V_k_phi = PhiEncoded.encode(V_k)
        US_k_phi = PhiEncoded.encode(US_k)

        # Storage comparison
        full_bytes = W_phi.signs.nbytes + W_phi.exponents.nbytes
        lr_bytes = (V_k_phi.signs.nbytes + V_k_phi.exponents.nbytes +
                    US_k_phi.signs.nbytes + US_k_phi.exponents.nbytes)

        # Decode and test
        V_k_dec = V_k_phi.decode()
        US_k_dec = US_k_phi.decode()

        correct = 0
        t0 = time.perf_counter()
        for i, h in enumerate(hidden_states):
            projected = h @ V_k_dec  # (k,)
            logits_lr = projected @ US_k_dec.T  # (152064,)
            lr_top = int(np.argmax(logits_lr))
            if lr_top == true_tops[i]:
                correct += 1
        dt = (time.perf_counter() - t0) * 1000 / len(hidden_states)

        n = len(prompts)
        print(f"  φ-encoded rank {k:5d}: {correct}/{n} correct  "
              f"{dt:.0f}ms/prompt  "
              f"storage={lr_bytes/1e6:.0f}MB (was {full_bytes/1e6:.0f}MB, "
              f"{full_bytes/lr_bytes:.1f}× smaller)")

        if correct < n:
            for i, h in enumerate(hidden_states):
                projected = h @ V_k_dec
                logits_lr = projected @ US_k_dec.T
                lr_top = int(np.argmax(logits_lr))
                if lr_top != true_tops[i]:
                    true_tok = tokenizer.decode_token(true_tops[i]).strip()
                    lr_tok = tokenizer.decode_token(lr_top).strip()
                    print(f"    ✗ '{prompts[i][0]}': "
                          f"full='{true_tok}' lr='{lr_tok}'")


def timing_comparison(W_decoded, U, S, Vt, hidden_states):
    """Compare timing of full vs low-rank matmul."""
    print("\n" + "=" * 70)
    print("  Part 4: Timing — Full vs Low-Rank")
    print("=" * 70)

    h = hidden_states[0]

    # Full matmul (cached)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = h @ W_decoded.T
        times.append(time.perf_counter() - t0)
    full_ms = np.median(times) * 1000
    print(f"  Full matmul (cached):     {full_ms:.1f} ms")

    for k in [128, 256, 512, 1024]:
        V_k = Vt[:k, :].T
        US_k = U[:, :k] * S[:k]

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            projected = h @ V_k
            logits = projected @ US_k.T
            times.append(time.perf_counter() - t0)
        lr_ms = np.median(times) * 1000
        print(f"  Low-rank k={k:4d}:          {lr_ms:.1f} ms  "
              f"({full_ms/lr_ms:.1f}× speedup)")


def main():
    print("=" * 70)
    print("  LM Head Low-Rank Exploration")
    print("  Connecting Doc 198 (Low-Rank) + Doc 244 (Jacobian) to LM Head")
    print("=" * 70)

    print("\nLoading model...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()

    # Warm cache
    W_decoded = engine.lm_head.weight.decode_cached()
    print(f"  LM head: {W_decoded.shape}, {W_decoded.nbytes/1e6:.0f} MB float32")

    # Get hidden states
    print("\nComputing hidden states for test prompts...")
    t0 = time.time()
    hidden_states = get_hidden_states(engine, tokenizer, TEST_PROMPTS)
    print(f"  {len(hidden_states)} prompts in {time.time()-t0:.1f}s")

    # SVD analysis
    U, S, Vt = analyze_svd_spectrum(W_decoded)

    # Accuracy tests
    test_lowrank_accuracy(W_decoded, U, S, Vt, hidden_states, tokenizer, TEST_PROMPTS)

    # φ-encoded low-rank
    test_phi_lowrank(engine.lm_head.weight, U, S, Vt, hidden_states, tokenizer, TEST_PROMPTS)

    # Timing
    timing_comparison(W_decoded, U, S, Vt, hidden_states)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  If low-rank preserves argmax at k=256-512:
    → 7-14× fewer ops
    → Each factor fits in gimli's 6GB VRAM
    → Can offload as two MATMUL instructions
    → May IMPROVE accuracy (Doc 244 Jacobian precedent)

  Connection to prior work:
    Doc 198: Weight matrices have low effective rank
    Doc 244: Low-rank Jacobian BETTER than full MLP (-1.64%)
    Doc 169: Grouped φ-matmul for integer path
    Doc 199: φ-complete computation substrate
""")


if __name__ == '__main__':
    main()
