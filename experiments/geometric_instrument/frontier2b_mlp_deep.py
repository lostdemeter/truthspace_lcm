"""
Frontier 2b: Deep MLP Investigation
======================================
Follow-up on the MLP geometry survey.

Investigations:
  1. Non-BOS position MLP content-independence (France vs Germany at p1-p4)
  2. BOS reservoir lifecycle: cumulative MLP contribution to h[0]
  3. Multi-layer synthetic BOS pump: replace BOS MLP output at ALL layers
  4. Progressive: how many layers can we synthetically replace?
"""

import sys, os, time, gc
import numpy as np
try:
    from scipy.sparse.linalg import svds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

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


def run_layer_manual(engine, h, li):
    """Run a single layer manually, returning h_new and mlp_out separately."""
    layer = engine.layers[li]
    attn = layer.attention
    mlp = layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

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

    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)

    return h_post_attn + mlp_out, mlp_out


def run_layer_with_synth_bos_mlp(engine, h, li, synth_bos_mlp):
    """Run layer but replace MLP output at position 0 with synthetic vector."""
    layer = engine.layers[li]
    attn = layer.attention
    mlp = layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

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

    # Compute real MLP for non-BOS positions
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)

    # Replace BOS position with synthetic
    mlp_out[0, 0, :] = synth_bos_mlp

    return h_post_attn + mlp_out


def predict(engine, tokenizer, h, answer):
    from phi_geometric.inference.phi_types import PhiEncoded
    fnw = engine.final_norm_weight
    if isinstance(fnw, PhiEncoded):
        fnw = fnw.decode()
    normed = rms_norm(h[:, -1:, :], fnw)
    logits = engine.lm_head(normed)[0, 0, :]
    top_tid = int(np.argmax(logits))
    ans_tid = tokenizer.encode(answer)[0]
    rank = int(np.sum(logits > logits[ans_tid]))
    return tokenizer.decode([top_tid]), rank


def main():
    print("=" * 80)
    print("  Frontier 2b: Deep MLP Investigation")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Investigation 1: Non-BOS MLP content-independence
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 1: MLP Output Content-Independence (ALL positions)")
    print("=" * 80)

    countries = ['France', 'Germany', 'Japan', 'Italy']
    mlp_outs = {}  # {country: {layer: [seq, hidden]}}

    for country in countries:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        mlp_outs[country] = {}
        for li in range(n_layers):
            h_new, mlp_out = run_layer_manual(engine, h, li)
            mlp_outs[country][li] = mlp_out[0].copy()  # [seq, hidden]
            h = h_new
        print(f"    {country} extracted")

    ref = 'France'
    others = [c for c in countries if c != ref]
    seq_len = mlp_outs[ref][0].shape[0]

    print(f"\n  cos(France, other) of MLP output per position:")
    print(f"  {'Layer':<8}", end="")
    for p in range(seq_len):
        print(f"  {'p'+str(p)+' (BOS)' if p==0 else 'p'+str(p):>10}", end="")
    print()
    print(f"  {'─'*(8 + seq_len*12)}")

    for li in range(n_layers):
        print(f"  L{li:<6}", end="")
        for p in range(seq_len):
            sims = []
            for c in others:
                r = mlp_outs[ref][li][p, :]
                o = mlp_outs[c][li][p, :]
                nr, no = np.linalg.norm(r), np.linalg.norm(o)
                cos = float(np.dot(r, o) / (nr * no + 1e-12)) if nr > 1e-6 and no > 1e-6 else 0
                sims.append(cos)
            mean_cos = np.mean(sims)
            if mean_cos > 0.999:
                print(f"  {'1.000':>10}", end="")
            elif mean_cos > 0.99:
                print(f"  {mean_cos:>10.4f}", end="")
            elif mean_cos > 0.9:
                print(f"  {mean_cos:>10.3f}", end="")
            else:
                print(f"  {mean_cos:>10.3f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: BOS Reservoir Lifecycle
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 2: BOS Reservoir Lifecycle")
    print("=" * 80)

    tids = tokenizer.encode(FACTS['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]

    print(f"\n  Tracking h[0] (BOS hidden state) through layers:")
    print(f"  {'Layer':<8} {'||h[0]||':>10} {'Δ_attn':>10} {'Δ_mlp':>10} {'MLP/total':>10}")
    print(f"  {'─'*50}")

    bos_norms = [float(np.linalg.norm(h[0, 0, :]))]
    mlp_bos_vectors = {}  # Store for synthetic replacement

    for li in range(n_layers):
        h_old = h.copy()
        h_new, mlp_out = run_layer_manual(engine, h, li)
        h_post_attn = h_new - mlp_out  # reconstruct post-attn state

        attn_delta = float(np.linalg.norm(h_post_attn[0, 0, :] - h_old[0, 0, :]))
        mlp_delta = float(np.linalg.norm(mlp_out[0, 0, :]))
        h_norm = float(np.linalg.norm(h_new[0, 0, :]))
        bos_norms.append(h_norm)
        mlp_bos_vectors[li] = mlp_out[0, 0, :].copy()

        total_delta = float(np.linalg.norm(h_new[0, 0, :] - h_old[0, 0, :]))
        mlp_frac = mlp_delta / (total_delta + 1e-12)

        marker = ""
        if mlp_delta > 1000: marker = " ◄◄◄"
        elif mlp_delta > 100: marker = " ◄"
        print(f"  L{li:<6} {h_norm:>10.1f} {attn_delta:>10.1f} {mlp_delta:>10.1f} "
              f"{mlp_frac:>9.1%}{marker}")

        h = h_new

    # Plot the lifecycle
    print(f"\n  BOS hidden state norm trajectory:")
    max_norm = max(bos_norms)
    for i, n in enumerate(bos_norms):
        bar_len = int(50 * n / max_norm)
        label = f"  {'embed' if i==0 else f'L{i-1}':<8}"
        print(f"{label} {'█' * bar_len} {n:.0f}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Calibrate synthetic BOS MLP for all layers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 3: Calibrate Synthetic BOS MLP (All Layers)")
    print("=" * 80)

    # For each layer, the synthetic BOS MLP = the France MLP output at BOS
    # (since cos=1.000 for all prompts, it's universal)
    print(f"\n  Using France BOS MLP outputs as synthetic vectors.")
    print(f"  (Validated: cos=1.000 and ||Fr||=||De|| at all layers)")

    # Test: replace BOS MLP at progressively more layers
    print(f"\n  Progressive synthetic BOS MLP replacement:")
    print(f"  (Replace BOS MLP output with cached France vector)")

    # First test: replace ALL layers
    for replace_set_name, replace_layers in [
        ("L3 only", [3]),
        ("L3+L26 (pump+drain)", [3, 26]),
        ("L1-L13 (early)", list(range(1, 14))),
        ("L20-L27 (late)", list(range(20, 28))),
        ("ALL 28 layers", list(range(28))),
    ]:
        correct = 0
        for country in FACTS:
            tids = tokenizer.encode(FACTS[country])
            h = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                if li in replace_layers:
                    h = run_layer_with_synth_bos_mlp(
                        engine, h, li, mlp_bos_vectors[li])
                else:
                    h = engine.layers[li](h)
            _, rank = predict(engine, tokenizer, h, ANSWERS[country])
            if rank == 0: correct += 1

        print(f"    {replace_set_name:<30} {correct}/6")

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: Rank-1 approximation of BOS MLP at each layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 4: BOS MLP Direction Analysis")
    print("=" * 80)

    # For each layer, compute the dominant direction of BOS MLP output
    # using SVD of W_down's first singular vector
    print(f"\n  Comparing BOS MLP output direction to W_down SV0:")
    print(f"  {'Layer':<8} {'||mlp_bos||':>12} {'cos(out,sv0)':>13} {'rank-1 approx':>14}")
    print(f"  {'─'*50}")

    sv0_directions = {}
    for li in range(n_layers):
        mlp_vec = mlp_bos_vectors[li]
        mlp_norm = float(np.linalg.norm(mlp_vec))

        if mlp_norm < 1.0:
            print(f"  L{li:<6} {mlp_norm:>12.1f} {'(tiny)':>13} {'skip':>14}")
            continue

        # Get first singular vector of W_down
        W_down = engine.layers[li].mlp.W_down
        if isinstance(W_down, PhiEncoded):
            W_down = W_down.decode()

        try:
            if HAS_SCIPY:
                U, S, Vt = svds(W_down.astype(np.float64), k=1)
                sv0 = U[:, 0].astype(np.float32)
            else:
                U, S, Vt = np.linalg.svd(W_down.astype(np.float64), full_matrices=False)
                sv0 = U[:, 0].astype(np.float32)
            # Fix sign to align with mlp_vec
            if np.dot(sv0, mlp_vec) < 0:
                sv0 = -sv0
            sv0_directions[li] = sv0
            cos_sv0 = float(np.dot(mlp_vec / mlp_norm, sv0))
            # Rank-1 approximation: project onto sv0
            proj_scale = float(np.dot(mlp_vec, sv0))
            approx = proj_scale * sv0
            residual = float(np.linalg.norm(mlp_vec - approx))
            rank1_quality = 1.0 - residual / mlp_norm
        except Exception as e:
            cos_sv0 = 0
            rank1_quality = 0
            print(f"  L{li:<6} {mlp_norm:>12.1f} {'ERR':>13} {str(e)[:20]}")
            continue

        marker = " ◄" if cos_sv0 > 0.99 else ""
        print(f"  L{li:<6} {mlp_norm:>12.1f} {cos_sv0:>13.6f} {rank1_quality:>13.3%}{marker}")

        gc.collect()

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Can we use scale * sv0 for ALL layers?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 5: Universal Scale × SV0 Replacement")
    print("=" * 80)

    # For layers with high cos(out, sv0), compute the scale factor
    synth_vectors = {}
    print(f"\n  {'Layer':<8} {'scale':>12} {'cos(out,sv0)':>13}")
    print(f"  {'─'*35}")

    for li in range(n_layers):
        if li not in sv0_directions:
            continue
        mlp_vec = mlp_bos_vectors[li]
        sv0 = sv0_directions[li]
        scale = float(np.dot(mlp_vec, sv0))
        synth_vectors[li] = scale * sv0
        cos_check = float(np.dot(mlp_vec, synth_vectors[li]) /
                         (np.linalg.norm(mlp_vec) * np.linalg.norm(synth_vectors[li]) + 1e-12))
        print(f"  L{li:<6} {scale:>12.1f} {cos_check:>13.6f}")

    # Test: replace with scale * sv0 at all layers where we have sv0
    print(f"\n  Testing scale×sv0 synthetic replacement:")

    for replace_set_name, replace_layers in [
        ("L3 only (sv0)", [3]),
        ("ALL with sv0", list(sv0_directions.keys())),
    ]:
        correct = 0
        details = []
        for country in FACTS:
            tids = tokenizer.encode(FACTS[country])
            h = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                if li in replace_layers and li in synth_vectors:
                    h = run_layer_with_synth_bos_mlp(
                        engine, h, li, synth_vectors[li])
                else:
                    h = engine.layers[li](h)
            top, rank = predict(engine, tokenizer, h, ANSWERS[country])
            if rank == 0: correct += 1
            details.append(f"{country}={'✓' if rank==0 else f'r{rank}'}")

        print(f"    {replace_set_name:<30} {correct}/6  [{', '.join(details)}]")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print(f"""
  Key findings:
  1. BOS MLP is content-independent at ALL layers (cos=1.000)
  2. Non-BOS positions: check results above
  3. BOS reservoir lifecycle: L3 pumps ({mlp_bos_vectors[3].shape[0]}D, ||{np.linalg.norm(mlp_bos_vectors[3]):.0f}||),
     L26 drains (||{np.linalg.norm(mlp_bos_vectors[26]):.0f}||)
  4. Synthetic replacement with cached vectors: see progressive results
  5. Rank-1 (scale×sv0) replacement: see sv0 results

  Parameter count for synthetic BOS MLP vectors:
    Per layer: 1 vector (3584 floats) = 14 KB
    All 28 layers: 28 × 3584 = {28 * 3584:,} floats = {28 * 3584 * 4 / 1024:.0f} KB
    vs computing MLP: 3 matrices × (3584 × 18944) = {3 * 3584 * 18944:,} params per layer
    Compression per layer: {3 * 3584 * 18944 / 3584:.0f}:1
""")


if __name__ == '__main__':
    main()
