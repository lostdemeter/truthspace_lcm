"""
Frontier 2: MLP Geometry Survey
=================================
Survey the weight geometry of all 28 layers' MLPs.

Investigations:
  1. SVD of W_down: rank-1 ratio S[0]/S[1] per layer
  2. SVD of W_gate and W_up: structure analysis
  3. Gate-Up alignment per position (cos of gate/up projections)
  4. MLP output norms per position per layer
  5. MLP output direction similarity (cross-prompt and cross-position)
  6. Identify layers with special structure (like L3's rank-1 BOS pump)
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  'The capital of France is',
    'Germany': 'The capital of Germany is',
}

ANSWERS = {'France': ' Paris', 'Germany': ' Berlin'}


def decode_weight(w):
    if isinstance(w, PhiEncoded):
        return w.decode()
    return w


def main():
    print("=" * 80)
    print("  Frontier 2: MLP Geometry Survey")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Investigation 1: W_down SVD — rank-1 ratio per layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 1: W_down SVD Survey")
    print("=" * 80)

    print(f"\n  {'Layer':<8} {'Shape':<18} {'S[0]/S[1]':>10} {'S[0]/S[2]':>10} {'S[0]/S[-1]':>11} {'top-1%':>8}")
    print(f"  {'─'*68}")

    sv_ratios = []
    for li in range(n_layers):
        W_down = decode_weight(engine.layers[li].mlp.W_down)
        # Only compute first few singular values (faster)
        # Full SVD is too slow for 28 layers; use partial
        try:
            from scipy.sparse.linalg import svds
            # svds gives largest k singular values
            k = min(10, min(W_down.shape) - 1)
            U, S, Vt = svds(W_down.astype(np.float64), k=k)
            S = np.sort(S)[::-1]  # svds returns ascending
        except ImportError:
            # Fallback: compute full SVD (slower)
            S = np.linalg.svd(W_down, compute_uv=False)

        ratio_01 = S[0] / S[1] if len(S) > 1 else float('inf')
        ratio_02 = S[0] / S[2] if len(S) > 2 else float('inf')
        ratio_0last = S[0] / S[-1] if S[-1] > 0 else float('inf')
        top1_pct = 100.0 * S[0]**2 / (np.sum(S**2) + 1e-12)
        sv_ratios.append(ratio_01)

        marker = " ◄◄◄" if ratio_01 > 2.0 else ""
        print(f"  L{li:<6} {str(W_down.shape):<18} {ratio_01:>10.3f} {ratio_02:>10.3f} "
              f"{ratio_0last:>11.3f} {top1_pct:>7.1f}%{marker}")
        gc.collect()

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: MLP output norms per position per layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 2: MLP Output Norms Per Position")
    print("=" * 80)

    for country, prompt in FACTS.items():
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        h = engine.embedding(tids)[np.newaxis, :, :]
        tokens = [tokenizer.decode([t]) for t in tids]

        print(f"\n  {country}: {tokens}")
        print(f"  {'Layer':<8}", end="")
        for p in range(seq_len):
            print(f"  {'p'+str(p):>8}", end="")
        print(f"  {'BOS/avg':>8}")
        print(f"  {'─'*(8 + seq_len*10 + 10)}")

        for li in range(n_layers):
            layer = engine.layers[li]
            attn = layer.attention
            mlp = layer.mlp

            # Run attention sublayer
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            normed = rms_norm(h, attn.norm_weight)
            Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
            K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if seq_len > 1:
                scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
            w = phi_softmax(scores, axis=-1)
            ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
            ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
            h_post_attn = h + phi_linear(attn.W_o, ao)

            # MLP sublayer
            nm = rms_norm(h_post_attn, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)

            norms = [float(np.linalg.norm(mlp_out[0, p, :])) for p in range(seq_len)]
            bos_ratio = norms[0] / (np.mean(norms[1:]) + 1e-12) if seq_len > 1 else 0

            h = h_post_attn + mlp_out

            marker = " ◄" if norms[0] > 100 and bos_ratio > 10 else ""
            print(f"  L{li:<6}", end="")
            for n in norms:
                if n > 100:
                    print(f"  {n:>8.0f}", end="")
                else:
                    print(f"  {n:>8.1f}", end="")
            print(f"  {bos_ratio:>7.1f}x{marker}")

        # Only do this for the first country to save time
        break

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Gate-Up alignment per position
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 3: Gate-Up Alignment (cos) Per Position")
    print("=" * 80)

    tids = tokenizer.encode(FACTS['France'])
    seq_len = len(tids)
    h = engine.embedding(tids)[np.newaxis, :, :]

    print(f"\n  France: cos(gate_proj, up_proj) per position")
    print(f"  {'Layer':<8}", end="")
    for p in range(seq_len):
        print(f"  {'p'+str(p):>7}", end="")
    print()
    print(f"  {'─'*(8 + seq_len*9)}")

    for li in range(n_layers):
        layer = engine.layers[li]
        h = layer(h)  # fast — don't need to decompose
        # But we need pre-MLP state. Let me redo properly.
    # Reset and redo with decomposition
    h = engine.embedding(tids)[np.newaxis, :, :]
    for li in range(n_layers):
        layer = engine.layers[li]
        attn = layer.attention
        mlp = layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim

        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if seq_len > 1:
            scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
        wts = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', wts, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        h_post_attn = h + phi_linear(attn.W_o, ao)

        nm = rms_norm(h_post_attn, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
        h = h_post_attn + mlp_out

        # Compute cos(gate, up) per position
        cos_vals = []
        for p in range(seq_len):
            gv = g[0, p, :]
            uv = u[0, p, :]
            cos = float(np.dot(gv, uv) / (np.linalg.norm(gv) * np.linalg.norm(uv) + 1e-12))
            cos_vals.append(cos)

        marker = " ◄" if cos_vals[0] > 0.5 else ""
        print(f"  L{li:<6}", end="")
        for c in cos_vals:
            print(f"  {c:>7.3f}", end="")
        print(marker)

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: Cross-prompt MLP output direction
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 4: Cross-Prompt MLP Output Direction (BOS)")
    print("=" * 80)

    mlp_bos = {}  # {country: {layer: mlp_out_at_bos}}
    for country, prompt in FACTS.items():
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        h = engine.embedding(tids)[np.newaxis, :, :]
        mlp_bos[country] = {}

        for li in range(n_layers):
            layer = engine.layers[li]
            attn = layer.attention
            mlp = layer.mlp
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim

            normed = rms_norm(h, attn.norm_weight)
            Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
            K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if seq_len > 1:
                scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
            wts = phi_softmax(scores, axis=-1)
            ao = np.einsum('bhqk,bhkd->bhqd', wts, Ve)
            ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
            h_post_attn = h + phi_linear(attn.W_o, ao)

            nm = rms_norm(h_post_attn, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
            mlp_bos[country][li] = mlp_out[0, 0, :].copy()
            h = h_post_attn + mlp_out

    print(f"\n  cos(France_BOS, Germany_BOS) of MLP output per layer:")
    print(f"  {'Layer':<8} {'cos':>8} {'||Fr||':>10} {'||De||':>10} {'ratio':>8}")
    print(f"  {'─'*48}")

    high_cos_layers = []
    for li in range(n_layers):
        f = mlp_bos['France'][li]
        g = mlp_bos['Germany'][li]
        cos = float(np.dot(f, g) / (np.linalg.norm(f) * np.linalg.norm(g) + 1e-12))
        nf = float(np.linalg.norm(f))
        ng = float(np.linalg.norm(g))
        ratio = nf / (ng + 1e-12)
        marker = " ◄" if cos > 0.99 and nf > 50 else ""
        if cos > 0.99 and nf > 50:
            high_cos_layers.append(li)
        print(f"  L{li:<6} {cos:>8.4f} {nf:>10.1f} {ng:>10.1f} {ratio:>7.3f}{marker}")

    if high_cos_layers:
        print(f"\n  Layers with cos > 0.99 AND ||out|| > 50: {high_cos_layers}")
        print(f"  These are candidates for synthetic replacement (like L3 BOS pump)")

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print(f"\n  W_down rank-1 ratios (S[0]/S[1]):")
    special = [(li, r) for li, r in enumerate(sv_ratios) if r > 1.5]
    typical = [r for r in sv_ratios if r <= 1.5]
    print(f"    Typical range: {min(typical):.3f} – {max(typical):.3f}" if typical else "    No typical layers")
    for li, r in special:
        print(f"    L{li}: S[0]/S[1] = {r:.3f} ◄ ELEVATED")

    print(f"\n  High-cos BOS MLP layers (cross-prompt direction stable):")
    for li in high_cos_layers:
        f = mlp_bos['France'][li]
        print(f"    L{li}: ||out|| = {np.linalg.norm(f):.1f}")

    print()


if __name__ == '__main__':
    main()
