"""
Frontier 7: Weight Shape Translation
======================================
The weights are already φ-encoded as signs + exponents.
Signs = the SHAPE (hyperplane arrangement).
Exponents = the MAGNITUDE (how strongly each hyperplane pushes).

Key question: Do the weight SIGNS alone carry the computation?
If yes, the COMB zone is a binary sorting machine — like the
IPA converter's RECT pairs — and we can design our own shapes.

Tests:
  1. Sign-only MLP: replace all exponents with uniform → does output
     direction match? (cosine similarity)
  2. Per-matrix ablation: which weight's signs matter most?
     (W_gate signs vs W_up signs vs W_down signs)
  3. Sign structure: rank, sparsity, clustering of sign matrices
  4. Cross-layer similarity: are sign patterns shared across COMB layers?
  5. SVD of sign matrices: what are the principal "shapes"?
  6. Binary forward pass: sign(W) @ sign(x) → does it predict output direction?
"""

import sys, os, time
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI_CONST = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI_CONST)


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-20:
        return 0.0
    return float(np.dot(a, b) / d)


def sign_only_weight(W_phi: PhiEncoded, uniform_exp: int = 0) -> PhiEncoded:
    """Replace all exponents with a uniform value, keeping signs."""
    return PhiEncoded(
        signs=W_phi.signs.copy(),
        exponents=np.full_like(W_phi.exponents, uniform_exp),
    )


def run_mlp_custom(normed, W_gate, W_up, W_down):
    """Run MLP with custom weight matrices (PhiEncoded)."""
    gate = phi_linear(W_gate, normed)
    up = phi_linear(W_up, normed)
    return phi_linear(W_down, phi_silu(gate) * up)


def run_attn_custom(hidden, attn, W_q=None, W_k=None, W_v=None, W_o=None):
    """Run attention with optionally replaced weights."""
    batch, seq_len, hidden_dim = hidden.shape
    normed = rms_norm(hidden, attn.norm_weight)

    Q = phi_linear(W_q or attn.W_q, normed, attn.b_q)
    K = phi_linear(W_k or attn.W_k, normed, attn.b_k)
    V = phi_linear(W_v or attn.W_v, normed, attn.b_v)

    Q = Q.reshape(batch, seq_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, attn.heads_per_kv, axis=1)
    Ve = np.repeat(V, attn.heads_per_kv, axis=1)

    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if seq_len > 1:
        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    w = phi_softmax(scores, axis=-1)
    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return hidden + phi_linear(W_o or attn.W_o, ao)


def run_layer_custom(hidden, layer, mlp_gate=None, mlp_up=None, mlp_down=None,
                     attn_q=None, attn_k=None, attn_v=None, attn_o=None):
    """Run one layer with optionally replaced weights."""
    attn = layer.attention
    mlp = layer.mlp

    # Attention
    h_pa = run_attn_custom(hidden, attn, attn_q, attn_k, attn_v, attn_o)

    # MLP
    normed = rms_norm(h_pa, mlp.norm_weight)
    gate_w = mlp_gate or mlp.W_gate
    up_w = mlp_up or mlp.W_up
    down_w = mlp_down or mlp.W_down
    mlp_out = run_mlp_custom(normed, gate_w, up_w, down_w)
    return h_pa + mlp_out


def run_layers(engine, h, start, end):
    for li in range(start, end):
        layer = engine.layers[li]
        attn, mlp = layer.attention, layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        sl = h.shape[1]
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        h_pa = h + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)
        gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
        h = h_pa + phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))
    return h


def predict_token(engine, tokenizer, h):
    for attr in ['final_norm_weight', 'norm_weight', 'ln_f_weight']:
        if hasattr(engine, attr):
            final_norm_w = getattr(engine, attr)
            break
    else:
        final_norm_w = engine.final_norm.weight
    h_last = rms_norm(h[:, -1:, :], final_norm_w)
    lm_w = engine.lm_head_weight if hasattr(engine, 'lm_head_weight') else engine.lm_head.weight
    logits = phi_linear(lm_w, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def main():
    print("=" * 80)
    print("  Frontier 7: Weight Shape Translation")
    print("=" * 80)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    prompts = [
        ('The capital of France is', ' Paris'),
        ('The capital of Germany is', ' Berlin'),
    ]

    # ═══════════════════════════════════════════════════════════
    # Baseline: normal forward pass through COMB zone
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Baseline: Normal Forward Pass")
    print("=" * 80)

    baselines = {}
    for prompt, expected in prompts:
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 28)
        _, tok, _ = predict_token(engine, tokenizer, h)
        print(f"  '{prompt}' → {tok[0]!r} (expected {expected!r})")
        baselines[prompt] = h.copy()

    # ═══════════════════════════════════════════════════════════
    # Inv 1: Weight Sign Structure Analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 1: Weight Sign Structure (L15-L20)")
    print("=" * 80)

    for li in range(15, 21):
        layer = engine.layers[li]
        mlp = layer.mlp
        attn = layer.attention

        print(f"\n  Layer {li}:")
        for name, W in [('W_gate', mlp.W_gate), ('W_up', mlp.W_up),
                         ('W_down', mlp.W_down), ('W_o', attn.W_o)]:
            signs = W.signs.astype(np.float32)
            exps = W.exponents.astype(np.float32)

            # Sign balance: what fraction is +1?
            frac_pos = float(np.mean(signs > 0))

            # Exponent stats
            exp_mean = float(np.mean(exps))
            exp_std = float(np.std(exps))

            # Sign matrix rank (sample — full SVD too expensive for 18944×3584)
            # Use a smaller sample to estimate structure
            n_sample = min(512, signs.shape[0])
            idx = np.random.choice(signs.shape[0], n_sample, replace=False)
            signs_sample = signs[idx]
            _, S_s, _ = np.linalg.svd(signs_sample, full_matrices=False)
            energy = np.cumsum(S_s ** 2) / np.sum(S_s ** 2)
            rank_90 = int(np.searchsorted(energy, 0.90) + 1)
            rank_99 = int(np.searchsorted(energy, 0.99) + 1)

            print(f"    {name:8s}: {W.signs.shape}  frac+={frac_pos:.3f}  "
                  f"exp μ={exp_mean:.0f} σ={exp_std:.0f}  "
                  f"sign_rank@90%={rank_90} @99%={rank_99}")

    # ═══════════════════════════════════════════════════════════
    # Inv 2: Sign-Only Layer Test — does output direction match?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: Sign-Only Layer Test (One Layer at a Time)")
    print("=" * 80)

    prompt = 'The capital of France is'
    tids = tokenizer.encode(prompt)

    # Compute median exponent per weight matrix to use as uniform scale
    for li in [15, 17, 19]:
        layer = engine.layers[li]
        mlp = layer.mlp

        # Get hidden state at this layer
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, li)
        h_before = h.copy()

        # Normal layer output
        h_normal = run_layer_custom(h_before.copy(), layer)
        normal_delta = h_normal - h_before

        # Sign-only MLP (keep attention normal)
        # Use median exponent as uniform magnitude
        for name, orig_w in [('gate', mlp.W_gate), ('up', mlp.W_up), ('down', mlp.W_down)]:
            median_exp = int(np.median(orig_w.exponents))

            # Build sign-only version of just this one weight
            so_w = sign_only_weight(orig_w, median_exp)

            kwargs = {}
            if name == 'gate':
                kwargs['mlp_gate'] = so_w
            elif name == 'up':
                kwargs['mlp_up'] = so_w
            elif name == 'down':
                kwargs['mlp_down'] = so_w

            h_so = run_layer_custom(h_before.copy(), layer, **kwargs)
            so_delta = h_so - h_before

            cos = cosine(normal_delta[0, -1], so_delta[0, -1])
            print(f"  L{li} sign-only {name:5s}: cos(delta)={cos:.4f}  "
                  f"median_exp={median_exp}")

        # ALL MLP weights sign-only
        so_gate = sign_only_weight(mlp.W_gate, int(np.median(mlp.W_gate.exponents)))
        so_up = sign_only_weight(mlp.W_up, int(np.median(mlp.W_up.exponents)))
        so_down = sign_only_weight(mlp.W_down, int(np.median(mlp.W_down.exponents)))

        h_all_so = run_layer_custom(h_before.copy(), layer,
                                     mlp_gate=so_gate, mlp_up=so_up, mlp_down=so_down)
        all_so_delta = h_all_so - h_before
        cos_all = cosine(normal_delta[0, -1], all_so_delta[0, -1])

        # Continue forward to see if prediction survives
        h_rest = run_layers(engine, h_all_so, li + 1, 28)
        _, tok, _ = predict_token(engine, tokenizer, h_rest)
        print(f"  L{li} ALL MLP sign-only: cos(delta)={cos_all:.4f} → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    # Inv 3: Pure Binary Forward — sign(W) @ sign(x)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: Pure Binary MLP — sign(W) @ sign(x)")
    print("=" * 80)

    for li in [15, 17, 19]:
        layer = engine.layers[li]
        mlp = layer.mlp

        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, li)
        h_before = h.copy()

        # Normal output
        h_normal = run_layer_custom(h_before.copy(), layer)
        normal_delta = h_normal - h_before

        # Binary forward: sign(W) @ sign(x) for each MLP step
        normed = rms_norm(h_before, mlp.norm_weight)
        x_last = normed[0, -1].astype(np.float64)  # (3584,)
        x_sign = np.sign(x_last).astype(np.float64)
        x_sign[x_sign == 0] = 1.0

        # Binary gate = sign(W_gate) @ sign(x)
        W_gate_s = mlp.W_gate.signs.astype(np.float64)  # (18944, 3584)
        W_up_s = mlp.W_up.signs.astype(np.float64)
        W_down_s = mlp.W_down.signs.astype(np.float64)

        bin_gate = W_gate_s @ x_sign  # (18944,) — each entry is sum of ±1
        bin_up = W_up_s @ x_sign

        # The gate output is a "vote" — majority of ±1 products
        # Positive = this hyperplane is on the input's side
        # Negative = opposite side
        # SiLU(bin_gate) selects which neurons fire

        # Apply SiLU to the binary gate vote
        gate_activated = phi_silu(bin_gate.astype(np.float32))
        bin_hidden = gate_activated * bin_up.astype(np.float32)

        # Project back
        bin_out = W_down_s @ bin_hidden.astype(np.float64)  # (3584,)

        cos_bin = cosine(normal_delta[0, -1], bin_out)
        print(f"  L{li} binary MLP: cos(normal_delta, binary_out)={cos_bin:.4f}")

        # What about just the gate pattern?
        gate_normal = phi_linear(mlp.W_gate, normed)[0, -1]  # (18944,)
        gate_sign_match = float(np.mean(np.sign(gate_normal) == np.sign(bin_gate)))
        print(f"  L{li} gate sign agreement: {gate_sign_match:.3f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 4: Cross-Layer Sign Similarity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: Cross-Layer Sign Pattern Similarity")
    print("=" * 80)

    # Compare W_gate sign patterns across COMB layers
    gate_signs = {}
    for li in range(15, 21):
        gate_signs[li] = engine.layers[li].mlp.W_gate.signs.astype(np.float32)

    print("\n  W_gate sign cosine (sampled 512 rows):")
    print(f"        ", end="")
    for lj in range(15, 21):
        print(f"  L{lj:2d}", end="")
    print()

    for li in range(15, 21):
        print(f"  L{li:2d}: ", end="")
        for lj in range(15, 21):
            if lj < li:
                print("      ", end="")
                continue
            # Sample rows for tractable comparison
            n = min(256, gate_signs[li].shape[0])
            idx = np.arange(0, gate_signs[li].shape[0], gate_signs[li].shape[0] // n)[:n]
            a = gate_signs[li][idx].ravel()
            b = gate_signs[lj][idx].ravel()
            c = cosine(a, b)
            print(f"  {c:.3f}" if li != lj else "  1.000", end="")
        print()

    # ═══════════════════════════════════════════════════════════
    # Inv 5: Sign-Only Full COMB Zone
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Sign-Only Full COMB Zone (L15-L20)")
    print("=" * 80)

    for prompt, expected in prompts:
        tids_p = tokenizer.encode(prompt)
        h = engine.embedding(tids_p)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)  # Normal L0-L14

        # Sign-only L15-L20
        for li in range(15, 21):
            layer = engine.layers[li]
            mlp = layer.mlp

            so_gate = sign_only_weight(mlp.W_gate, int(np.median(mlp.W_gate.exponents)))
            so_up = sign_only_weight(mlp.W_up, int(np.median(mlp.W_up.exponents)))
            so_down = sign_only_weight(mlp.W_down, int(np.median(mlp.W_down.exponents)))

            h = run_layer_custom(h, layer,
                                  mlp_gate=so_gate, mlp_up=so_up, mlp_down=so_down)

        # Normal L21-L27
        h = run_layers(engine, h, 21, 28)
        _, tok, _ = predict_token(engine, tokenizer, h)
        cos_vs_bl = cosine(h[0, -1], baselines[prompt][0, -1])
        print(f"  '{prompt}' sign-only COMB: → {tok[0]!r} "
              f"(expected {expected!r})  cos_vs_baseline={cos_vs_bl:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 6: Exponent Distribution — Is it really separable?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 6: Exponent Distribution Analysis")
    print("=" * 80)

    for li in [15, 17, 20]:
        mlp = engine.layers[li].mlp
        for name, W in [('W_gate', mlp.W_gate), ('W_up', mlp.W_up), ('W_down', mlp.W_down)]:
            exps = W.exponents.ravel().astype(np.float64)
            # Histogram of exponents
            p5, p25, p50, p75, p95 = np.percentile(exps, [5, 25, 50, 75, 95])
            unique_exps = len(np.unique(exps))
            print(f"  L{li} {name:8s}: p5={p5:.0f} p25={p25:.0f} p50={p50:.0f} "
                  f"p75={p75:.0f} p95={p95:.0f}  unique={unique_exps}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  Weight Shape Translation results:
  - If sign-only cos ≈ 1: shapes carry the computation → translatable
  - If sign-only cos ≈ 0: magnitudes essential → shapes necessary but insufficient
  - Gate sign agreement shows whether binary hyperplane decisions match
  - Cross-layer similarity shows whether COMB uses shared or unique shapes
""")


if __name__ == '__main__':
    main()
