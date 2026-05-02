"""
Combined Frontier Test: All-Position Templates + All-Layer BOS SV0
====================================================================
Combines:
  - F138: Full attention matrix replacement (all positions)
  - F139: BOS MLP replacement via scale × sv0 at all 28 layers

This represents the maximum geometric replacement we can achieve.
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

# Additional prompts for different lengths
LENGTH_PROMPTS = [
    ('N=5',  'The capital of France is'),
    ('N=7',  'The official capital city of France is'),
    ('N=9',  'The official capital city of the country France is'),
]


def decode_weight(w):
    return w.decode() if isinstance(w, PhiEncoded) else w


def get_full_attention(engine, h, li):
    """Extract FULL attention weights [nh, seq, seq]."""
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


def get_sv0_direction(engine, li):
    """Get first left singular vector of W_down via power iteration."""
    W_down = decode_weight(engine.layers[li].mlp.W_down)
    rng = np.random.RandomState(42)
    v = rng.randn(W_down.shape[1]).astype(np.float64)
    for _ in range(20):
        u = W_down.astype(np.float64) @ v
        u /= np.linalg.norm(u)
        v = W_down.astype(np.float64).T @ u
        v /= np.linalg.norm(v)
    return u.astype(np.float32)


def calibrate_bos_mlp(engine, tokenizer, prompt):
    """Extract BOS MLP output vectors for all layers."""
    tids = tokenizer.encode(prompt)
    h = engine.embedding(tids)[np.newaxis, :, :]
    bos_mlp = {}
    for li in range(len(engine.layers)):
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
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        h_pa = h + phi_linear(attn.W_o, ao)

        nm = rms_norm(h_pa, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
        bos_mlp[li] = mlp_out[0, 0, :].copy()
        h = h_pa + mlp_out
    return bos_mlp


def run_combined_geometric(engine, h, li, template, synth_bos_vec):
    """Run layer with:
    - ALL attention positions replaced by fixed template
    - BOS MLP output replaced by synthetic vector
    """
    layer = engine.layers[li]
    attn = layer.attention
    mlp = layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    # V projection only (no Q/K needed!)
    normed = rms_norm(h, attn.norm_weight)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Ve = np.repeat(V, hpk, axis=1)

    # Apply full template
    w = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
    ts = template.shape[1]
    if seq_len == ts:
        w[0] = template
    elif seq_len < ts:
        w[0] = template[:, :seq_len, :seq_len]
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
    else:
        w[0, :, :ts, :ts] = template
        for p in range(ts, seq_len):
            w[0, :, p, p] = 1.0
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    h_pa = h + phi_linear(attn.W_o, ao)

    # MLP with synthetic BOS
    nm = rms_norm(h_pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
    mlp_out[0, 0, :] = synth_bos_vec  # Replace BOS MLP
    return h_pa + mlp_out


def predict(engine, tokenizer, h, answer):
    fnw = decode_weight(engine.final_norm_weight)
    normed = rms_norm(h[:, -1:, :], fnw)
    logits = engine.lm_head(normed)[0, 0, :]
    top_tid = int(np.argmax(logits))
    ans_tid = tokenizer.encode(answer)[0]
    rank = int(np.sum(logits > logits[ans_tid]))
    return tokenizer.decode([top_tid]), rank


def main():
    print("=" * 80)
    print("  Combined Frontier: All-Position Templates + All-Layer BOS SV0")
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
    # Step 1: Calibrate — extract templates and BOS MLP vectors
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 1: Calibrate (extract templates + BOS MLP from France)")
    print("=" * 80)

    # Extract full attention templates from France
    tids_ref = tokenizer.encode(FACTS['France'])
    h = engine.embedding(tids_ref)[np.newaxis, :, :]
    france_templates = {}
    for li in range(n_layers):
        france_templates[li] = get_full_attention(engine, h, li)
        h = engine.layers[li](h)
    print(f"  Full attention templates extracted (N={len(tids_ref)})")

    # Extract BOS MLP vectors
    bos_mlp = calibrate_bos_mlp(engine, tokenizer, FACTS['France'])
    print(f"  BOS MLP vectors extracted (28 layers)")

    # Compute sv0 directions and scales for all layers
    print(f"  Computing SV0 directions...", end="", flush=True)
    sv0_scales = {}
    sv0_dirs = {}
    for li in range(n_layers):
        sv0 = get_sv0_direction(engine, li)
        if np.dot(sv0, bos_mlp[li]) < 0:
            sv0 = -sv0
        sv0_dirs[li] = sv0
        sv0_scales[li] = float(np.dot(bos_mlp[li], sv0))
        gc.collect()
    print(f" done")

    synth_sv0 = {li: sv0_scales[li] * sv0_dirs[li] for li in range(n_layers)}

    # ═══════════════════════════════════════════════════════════
    # Step 2: Baseline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 2: Baseline (normal model)")
    print("=" * 80)

    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = engine.layers[li](h)
        top, rank = predict(engine, tokenizer, h, ANSWERS[country])
        print(f"    {country:>8}: '{top}' {'✓' if rank==0 else f'rank={rank}'}")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Combined — templates + sv0 BOS MLP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 3: Combined (all-position templates + all-layer BOS sv0)")
    print("=" * 80)

    print(f"\n  What's replaced:")
    print(f"    Attention: FULL matrix replaced by fixed France template")
    print(f"               (Q/K not computed — V only)")
    print(f"    BOS MLP:   scale × sv0 at all 28 layers")
    print(f"    Remaining: V/O projections + non-BOS MLP (neural)")

    correct = 0
    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_combined_geometric(engine, h, li,
                                       france_templates[li], synth_sv0[li])
        top, rank = predict(engine, tokenizer, h, ANSWERS[country])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0: correct += 1
        print(f"    {country:>8}: '{top}' {ok}")
    print(f"\n  Combined (templates + sv0): {correct}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Combined with exact cached BOS MLP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 4: Combined (all-position templates + exact cached BOS MLP)")
    print("=" * 80)

    correct2 = 0
    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_combined_geometric(engine, h, li,
                                       france_templates[li], bos_mlp[li])
        top, rank = predict(engine, tokenizer, h, ANSWERS[country])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0: correct2 += 1
        print(f"    {country:>8}: '{top}' {ok}")
    print(f"\n  Combined (templates + exact cached): {correct2}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 5: Ablation — each replacement independently
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 5: Ablation — independent and combined")
    print("=" * 80)

    configs = [
        ("Templates only (no BOS replace)", True, False),
        ("BOS sv0 only (no template replace)", False, True),
        ("Both combined", True, True),
    ]

    for name, use_templates, use_bos_sv0 in configs:
        correct_a = 0
        for country in FACTS:
            tids = tokenizer.encode(FACTS[country])
            h = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                if use_templates and use_bos_sv0:
                    h = run_combined_geometric(engine, h, li,
                                               france_templates[li], synth_sv0[li])
                elif use_templates:
                    # Template only, real MLP
                    layer = engine.layers[li]
                    attn = layer.attention
                    mlp = layer.mlp
                    seq_len = h.shape[1]
                    normed = rms_norm(h, attn.norm_weight)
                    V = phi_linear(attn.W_v, normed, attn.b_v)
                    V = V.reshape(1, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
                    Ve = np.repeat(V, attn.num_heads // attn.num_kv_heads, axis=1)
                    w = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
                    ts = france_templates[li].shape[1]
                    if seq_len == ts:
                        w[0] = france_templates[li]
                    w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
                    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
                    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
                    h_pa = h + phi_linear(attn.W_o, ao)
                    nm = rms_norm(h_pa, mlp.norm_weight)
                    g = phi_linear(mlp.W_gate, nm)
                    u = phi_linear(mlp.W_up, nm)
                    h = h_pa + phi_linear(mlp.W_down, phi_silu(g) * u)
                elif use_bos_sv0:
                    # Real attention, synthetic BOS MLP
                    layer = engine.layers[li]
                    attn = layer.attention
                    mlp = layer.mlp
                    seq_len = h.shape[1]
                    nh_l, nkv = attn.num_heads, attn.num_kv_heads
                    hpk, hd = nh_l // nkv, attn.head_dim
                    normed = rms_norm(h, attn.norm_weight)
                    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, seq_len, nh_l, hd).transpose(0, 2, 1, 3)
                    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
                    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
                    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
                    Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
                    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
                    if seq_len > 1:
                        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
                    wts = phi_softmax(scores, axis=-1)
                    ao = np.einsum('bhqk,bhkd->bhqd', wts, Ve).transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
                    h_pa = h + phi_linear(attn.W_o, ao)
                    nm = rms_norm(h_pa, mlp.norm_weight)
                    g = phi_linear(mlp.W_gate, nm)
                    u = phi_linear(mlp.W_up, nm)
                    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
                    mlp_out[0, 0, :] = synth_sv0[li]
                    h = h_pa + mlp_out

            _, rank = predict(engine, tokenizer, h, ANSWERS[country])
            if rank == 0: correct_a += 1
        print(f"    {name:<45} {correct_a}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 6: Parameter inventory
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 6: Parameter Inventory")
    print("=" * 80)

    # What we've replaced geometrically
    qk_params = 0
    vo_params = 0
    mlp_params = 0
    for li in range(n_layers):
        attn = engine.layers[li].attention
        mlp_l = engine.layers[li].mlp
        qk_params += decode_weight(attn.W_q).size + decode_weight(attn.W_k).size
        if attn.b_q is not None:
            qk_params += attn.b_q.size + attn.b_k.size
        vo_params += decode_weight(attn.W_v).size + decode_weight(attn.W_o).size
        if attn.b_v is not None:
            vo_params += attn.b_v.size
        mlp_params += (decode_weight(mlp_l.W_gate).size +
                       decode_weight(mlp_l.W_up).size +
                       decode_weight(mlp_l.W_down).size)

    total_params = qk_params + vo_params + mlp_params
    total_params += engine.embedding.table.size + decode_weight(engine.lm_head.weight).size
    total_params += decode_weight(engine.final_norm_weight).size

    # Geometric constants
    template_floats = sum(france_templates[li].size for li in range(n_layers))
    sv0_floats = 28  # just scale factors (sv0 derived from W_down)
    bos_cached_floats = 28 * 3584  # if using cached vectors instead

    print(f"""
  WHAT'S GEOMETRIC (replaced):
    Q/K attention routing (ALL positions):  {qk_params:>14,} params
    BOS MLP output (all 28 layers):         28 scale factors

  WHAT'S NEURAL (still needed):
    V/O projections:                        {vo_params:>14,} params
    MLP (non-BOS positions):                {mlp_params:>14,} params  (minus BOS savings)
    Embeddings + LM head:                   {engine.embedding.table.size + decode_weight(engine.lm_head.weight).size:>14,} params

  GEOMETRIC CONSTANTS:
    Full-matrix templates (France):         {template_floats:>10,} floats ({template_floats * 4 / 1024:.1f} KB)
    BOS sv0 scale factors:                  {sv0_floats:>10,} floats ({sv0_floats * 4:,} bytes)
    Total geometric constants:              {template_floats + sv0_floats:>10,} floats ({(template_floats + sv0_floats) * 4 / 1024:.1f} KB)

  COMPRESSION:
    Q/K replaced:                           {qk_params:>14,} params → {template_floats:,} template floats
    Ratio:                                  {qk_params / template_floats:>14,.0f}:1
    BOS MLP replaced:                       28 × full MLP → 28 scalars
    """)

    # ═══════════════════════════════════════════════════════════
    # Step 7: Cross-length test
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  Step 7: Cross-Length Generalization")
    print("=" * 80)

    for label, prompt in LENGTH_PROMPTS:
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_combined_geometric(engine, h, li,
                                       france_templates[li], synth_sv0[li])
        top, rank = predict(engine, tokenizer, h, ' Paris')
        print(f"    {label} ({len(tids)} tokens): '{top}' {'✓' if rank==0 else f'rank={rank}'}")

    print()


if __name__ == '__main__':
    main()
