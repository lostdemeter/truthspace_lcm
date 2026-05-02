"""
Frontier 3b: Scope of Content-Independence
=============================================
F138 showed: same structure, different entity → same attention.
But does this hold for COMPLETELY different token sequences?

Test: at the same N, compare attention for prompts with
completely different tokens at every position.

If attention depends ONLY on RoPE position indices (not token content),
then ANY two N=5 prompts should have identical attention.
If it depends on token content, we need per-structure caching.
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def get_full_attention(engine, h, li):
    """Extract full attention [nh, seq, seq]."""
    layer = engine.layers[li]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]
    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if seq_len > 1:
        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    return phi_softmax(scores, axis=-1)[0]


def main():
    print("=" * 80)
    print("  Frontier 3b: Scope of Content-Independence")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    nh = 28
    print(f" done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Test 1: Same N, same structure, different entity (F138 recap)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Test 1: Same N=5, same structure, different entity")
    print("=" * 80)

    same_struct = [
        'The capital of France is',   # N=5
        'The capital of Germany is',   # N=5
        'The capital of Japan is',     # N=5
    ]

    attn_same = {}
    for prompt in same_struct:
        tids = tokenizer.encode(prompt)
        assert len(tids) == 5, f"Expected 5 tokens, got {len(tids)} for '{prompt}'"
        h = engine.embedding(tids)[np.newaxis, :, :]
        layers_attn = {}
        for li in range(n_layers):
            layers_attn[li] = get_full_attention(engine, h, li)
            h = engine.layers[li](h)
        attn_same[prompt] = layers_attn

    # Compare pairwise
    prompts = list(attn_same.keys())
    for li in [0, 3, 10, 23, 27]:
        cos_vals = []
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                a = attn_same[prompts[i]][li].ravel()
                b = attn_same[prompts[j]][li].ravel()
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                cos_vals.append(cos)
        print(f"  L{li:2d}: cos = {np.mean(cos_vals):.6f} (min={min(cos_vals):.6f})")

    # ═══════════════════════════════════════════════════════════
    # Test 2: Same N, DIFFERENT structure
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Test 2: Same N=5, COMPLETELY different structure")
    print("=" * 80)

    diff_struct_5 = [
        'The capital of France is',    # N=5
        'I really love eating pizza',  # should be N=5
        'Please help me find this',    # should be N=5
        'Once upon a time there',      # should be N=5
        'How does the engine work',    # should be N=5
    ]

    # Verify lengths and find working ones
    working_5 = []
    for prompt in diff_struct_5:
        tids = tokenizer.encode(prompt)
        print(f"  '{prompt}' → {len(tids)} tokens: {tids}")
        if len(tids) == 5:
            working_5.append(prompt)

    # If some aren't N=5, try alternatives
    if len(working_5) < 3:
        alt_prompts = [
            'This is a test sentence',
            'My dog loves to run',
            'She went to the store',
            'They built a new house',
            'We need more time now',
            'He wrote a long book',
            'It was very cold today',
            'Can you see the light',
            'What time does it start',
            'I have three red cats',
        ]
        for prompt in alt_prompts:
            tids = tokenizer.encode(prompt)
            if len(tids) == 5 and prompt not in working_5:
                working_5.append(prompt)
                print(f"  + '{prompt}' → {len(tids)} tokens")
            if len(working_5) >= 5:
                break

    print(f"\n  Using {len(working_5)} prompts with N=5")

    if len(working_5) >= 2:
        attn_diff = {}
        for prompt in working_5:
            tids = tokenizer.encode(prompt)
            h = engine.embedding(tids)[np.newaxis, :, :]
            layers_attn = {}
            for li in range(n_layers):
                layers_attn[li] = get_full_attention(engine, h, li)
                h = engine.layers[li](h)
            attn_diff[prompt] = layers_attn

        # Compare: how similar is attention for completely different content?
        prompts_d = list(attn_diff.keys())
        print(f"\n  Cosine similarity of attention (different structure, same N=5):")
        for li in [0, 3, 10, 20, 23, 27]:
            cos_vals = []
            for i in range(len(prompts_d)):
                for j in range(i + 1, len(prompts_d)):
                    a = attn_diff[prompts_d[i]][li].ravel()
                    b = attn_diff[prompts_d[j]][li].ravel()
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    cos_vals.append(cos)
            print(f"    L{li:2d}: mean cos = {np.mean(cos_vals):.4f}, "
                  f"min = {min(cos_vals):.4f}, max = {max(cos_vals):.4f}")

        # Detailed: per-position comparison for L0 (most sensitive layer)
        print(f"\n  Per-position BOS fraction at L0 (head 0):")
        for prompt in prompts_d[:4]:
            A = attn_diff[prompt][0]  # [28, 5, 5]
            bos = [f"{float(A[0, q, 0]):.4f}" for q in range(5)]
            short = prompt[:35].ljust(35)
            print(f"    {short} BOS: [{', '.join(bos)}]")

        # Per-row comparison
        print(f"\n  Per-row cosine similarity at L0 (France vs each other):")
        ref = attn_diff[working_5[0]]
        for prompt in working_5[1:]:
            other = attn_diff[prompt]
            row_cos = []
            for q in range(5):
                a = ref[0][:, q, :].ravel()
                b = other[0][:, q, :].ravel()
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                row_cos.append(cos)
            short = prompt[:35].ljust(35)
            print(f"    {short} rows: " + " ".join(f"q{i}={c:.3f}" for i, c in enumerate(row_cos)))

    # ═══════════════════════════════════════════════════════════
    # Test 3: Same N, same structure, VERY different entity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Test 3: Same structure but wildly different entities")
    print("=" * 80)

    wild_entities = [
        'The capital of France is',
        'The capital of Narnia is',     # fictional
        'The meaning of purple is',     # abstract
        'The inverse of matrix is',     # math
    ]

    working_wild = []
    for prompt in wild_entities:
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            working_wild.append(prompt)
            print(f"  '{prompt}' → N={len(tids)} ✓")
        else:
            print(f"  '{prompt}' → N={len(tids)} ✗")

    if len(working_wild) >= 2:
        attn_wild = {}
        for prompt in working_wild:
            tids = tokenizer.encode(prompt)
            h = engine.embedding(tids)[np.newaxis, :, :]
            layers_attn = {}
            for li in range(n_layers):
                layers_attn[li] = get_full_attention(engine, h, li)
                h = engine.layers[li](h)
            attn_wild[prompt] = layers_attn

        prompts_w = list(attn_wild.keys())
        print(f"\n  Cosine similarity (same structure, wild entities):")
        for li in [0, 3, 10, 23, 27]:
            cos_vals = []
            for i in range(len(prompts_w)):
                for j in range(i + 1, len(prompts_w)):
                    a = attn_wild[prompts_w[i]][li].ravel()
                    b = attn_wild[prompts_w[j]][li].ravel()
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    cos_vals.append(cos)
            print(f"    L{li:2d}: cos = {np.mean(cos_vals):.6f}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  Test 1: Same structure, different entity (F138 recap)
    → Expected: cos ≈ 1.000 (proven in F138)
    
  Test 2: DIFFERENT structure, same N
    → If cos ≈ 1.000: attention depends ONLY on N (RoPE position)
      → One cache per N. Full Q/K elimination with ~8.5 MB template bank.
    → If cos << 1.000: attention depends on token content
      → Need per-structure cache. Still works for fixed-template tasks.

  Test 3: Same structure, wildly different entity
    → Stress test of content-independence beyond capital cities.
""")


if __name__ == '__main__':
    main()
