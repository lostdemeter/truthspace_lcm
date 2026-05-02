"""
Frontier 1: All-Position Templates
====================================
Can we extend T(N) from last-token-only to T(N,q) for ALL query positions?

Investigations:
  1. Extract full attention matrices [nh, seq, seq] for multiple prompts
  2. Check content-independence at EVERY position (not just last)
  3. Characterize the attention pattern structure per position
  4. Fit parametric T(N,q) if patterns emerge
  5. Test all-position template replacement
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  'The capital of France is',
    'Japan':   'The capital of Japan is',
    'Germany': 'The capital of Germany is',
    'Italy':   'The capital of Italy is',
    'Spain':   'The capital of Spain is',
    'Egypt':   'The capital of Egypt is',
}

ANSWERS = {
    'France': ' Paris', 'Japan': ' Tokyo', 'Germany': ' Berlin',
    'Italy': ' Rome', 'Spain': ' Madrid', 'Egypt': ' Cairo',
}


def get_full_attention(engine, h, li):
    """Extract FULL attention weights [nh, seq, seq] at layer li."""
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
    return phi_softmax(scores, axis=-1)[0]  # [nh, seq, seq]


def run_layer_with_full_template(engine, h, li, template):
    """Run layer replacing ALL attention positions with fixed template."""
    layer = engine.layers[li]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    normed = rms_norm(h, attn.norm_weight)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Ve = np.repeat(V, hpk, axis=1)

    # Use full template for ALL positions
    w = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
    ts = template.shape[1]  # template seq len
    if seq_len == ts:
        w[0] = template
    elif seq_len < ts:
        w[0] = template[:, :seq_len, :seq_len]
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
    else:
        w[0, :, :ts, :ts] = template
        # Extend: new positions attend to self
        for p in range(ts, seq_len):
            w[0, :, p, p] = 1.0
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    h = h + phi_linear(attn.W_o, ao)

    mlp = layer.mlp
    nm = rms_norm(h, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    h = h + phi_linear(mlp.W_down, phi_silu(g) * u)
    return h


def predict(engine, tokenizer, h, answer):
    normed = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = engine.lm_head(normed)[0, 0, :]
    top_tid = int(np.argmax(logits))
    ans_tid = tokenizer.encode(answer)[0]
    rank = int(np.sum(logits > logits[ans_tid]))
    return tokenizer.decode([top_tid]), rank


def main():
    print("=" * 80)
    print("  Frontier 1: All-Position Templates")
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
    # Investigation 1: Content-independence at ALL positions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 1: Content-Independence at All Positions")
    print("=" * 80)

    # Extract full attention for all 6 prompts
    all_attn = {}  # {country: {layer: [nh, seq, seq]}}
    for country, prompt in FACTS.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        all_attn[country] = {}
        for li in range(n_layers):
            all_attn[country][li] = get_full_attention(engine, h, li)
            h = engine.layers[li](h)
        print(f"    {country} extracted (N={len(tids)})")

    # Compare cross-prompt at each position
    countries = list(FACTS.keys())
    ref = countries[0]  # France as reference

    print(f"\n  Cross-prompt cosine similarity (ref={ref}):")
    print(f"  Per-position mean cos across all layers and heads:\n")

    # For a few sample layers, show per-position similarity
    sample_layers = [0, 3, 10, 20, 23, 27]
    seq_len = all_attn[ref][0].shape[1]

    print(f"  {'Layer':<8}", end="")
    for q in range(seq_len):
        print(f"  pos={q}", end="")
    print(f"  {'mean':>8}")
    print(f"  {'─'*(8 + seq_len*7 + 8)}")

    for li in sample_layers:
        pos_sims = []
        for q in range(seq_len):
            sims = []
            for c in countries[1:]:
                ref_row = all_attn[ref][li][:, q, :].ravel()
                cmp_row = all_attn[c][li][:, q, :].ravel()
                cos = float(np.dot(ref_row, cmp_row) /
                           (np.linalg.norm(ref_row) * np.linalg.norm(cmp_row) + 1e-12))
                sims.append(cos)
            pos_sims.append(np.mean(sims))
        print(f"  L{li:<6}", end="")
        for s in pos_sims:
            print(f"  {s:.3f}", end="")
        print(f"  {np.mean(pos_sims):>7.3f}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: Attention pattern structure per position
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 2: Attention Pattern Structure")
    print("=" * 80)

    # For France, show what each position attends to
    ref_attn = all_attn[ref]
    print(f"\n  France attention patterns (head-averaged):")
    print(f"  Tokens: {[tokenizer.decode([t]) for t in tokenizer.encode(FACTS[ref])]}")

    for li in sample_layers:
        w = ref_attn[li]  # [nh, seq, seq]
        w_avg = w.mean(axis=0)  # [seq, seq] head-averaged
        print(f"\n  Layer {li} (head-averaged attention matrix):")
        print(f"  {'query↓\\key→':<12}", end="")
        for k in range(seq_len):
            print(f"  k={k:>3}", end="")
        print()
        for q in range(seq_len):
            print(f"  q={q:<9}", end="")
            for k in range(seq_len):
                v = w_avg[q, k]
                if v > 0.01:
                    print(f"  {v:>.3f}", end="")
                else:
                    print(f"  {'·':>5}", end="")
            print()

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Per-position entropy (how peaked is attention)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 3: Per-Position Attention Entropy")
    print("=" * 80)

    print(f"\n  Entropy per position (bits, head-averaged, France):")
    print(f"  {'Layer':<8}", end="")
    for q in range(seq_len):
        print(f"  pos={q}", end="")
    print()
    print(f"  {'─'*(8 + seq_len*7)}")

    for li in sample_layers:
        w = ref_attn[li]  # [nh, seq, seq]
        entropies = []
        for q in range(seq_len):
            h_ent = []
            for hi in range(nh):
                row = w[hi, q, :q+1]  # only valid positions (causal)
                row = row + 1e-12
                ent = -np.sum(row * np.log2(row))
                h_ent.append(ent)
            entropies.append(np.mean(h_ent))
        print(f"  L{li:<6}", end="")
        for e in entropies:
            print(f"  {e:>.2f} ", end="")
        print()

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: Full-template replacement test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 4: Full-Template Replacement (France→all)")
    print("=" * 80)

    # Extract France's full templates
    tids_ref = tokenizer.encode(FACTS[ref])
    h = engine.embedding(tids_ref)[np.newaxis, :, :]
    france_templates = {}
    for li in range(n_layers):
        france_templates[li] = get_full_attention(engine, h, li)
        h = engine.layers[li](h)

    # Test: replace ALL attention (all positions) with France's templates
    correct = 0
    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_full_template(engine, h, li, france_templates[li])
        top, rank = predict(engine, tokenizer, h, ANSWERS[country])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0: correct += 1
        print(f"    {country:>8}: '{top}' {ok}")
    print(f"\n  Full-template (France→all): {correct}/6")

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Per-position similarity summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 5: Cross-Prompt Similarity — All Layers")
    print("=" * 80)

    # For each layer, compute mean cross-prompt cos for each position
    print(f"\n  Mean cross-prompt cos (France vs others), all layers:")
    print(f"  Position 0 = BOS, position {seq_len-1} = last (prediction)")
    print()

    pos_means_all = np.zeros((n_layers, seq_len))
    for li in range(n_layers):
        for q in range(seq_len):
            sims = []
            for c in countries[1:]:
                r = all_attn[ref][li][:, q, :].ravel()
                o = all_attn[c][li][:, q, :].ravel()
                cos = float(np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-12))
                sims.append(cos)
            pos_means_all[li, q] = np.mean(sims)

    # Show heatmap-style summary
    print(f"  {'Layer':<8}", end="")
    for q in range(seq_len):
        print(f"  p{q}", end="")
    print()
    print(f"  {'─'*(8 + seq_len*4)}")
    for li in range(n_layers):
        print(f"  L{li:<5}", end="")
        for q in range(seq_len):
            v = pos_means_all[li, q]
            if v > 0.999:
                ch = "█"
            elif v > 0.99:
                ch = "▓"
            elif v > 0.95:
                ch = "▒"
            elif v > 0.9:
                ch = "░"
            else:
                ch = "·"
            print(f"  {ch} ", end="")
        print(f"  {pos_means_all[li].mean():.3f}")

    print(f"\n  Legend: █>0.999  ▓>0.99  ▒>0.95  ░>0.9  ·<0.9")

    # Overall per-position summary
    print(f"\n  Per-position mean across ALL layers:")
    for q in range(seq_len):
        m = pos_means_all[:, q].mean()
        low = pos_means_all[:, q].min()
        print(f"    Position {q}: mean={m:.4f}, min={low:.4f}")

    print()


if __name__ == '__main__':
    main()
