"""
Frontier Optimal: Best Combination of All Geometric Replacements
=================================================================
Combines the PROVEN approaches:
  - F136: Parametric T(N) for last-token attention row (5 params/head)
  - F139: scale × sv0 BOS MLP replacement at all 28 layers

This keeps Q/K for non-last positions (real softmax) but eliminates:
  1. All Q/K computation at the last position (replaced by T(N))
  2. All MLP computation at BOS (replaced by scale × sv0)
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

LENGTH_PROMPTS = [
    ('N=5', 'The capital of France is'),
    ('N=7', 'The official capital city of France is'),
    ('N=9', 'The official capital city of the country France is'),
]


def decode_weight(w):
    return w.decode() if isinstance(w, PhiEncoded) else w


def get_last_token_attention(engine, h, li):
    """Extract last-token attention [nh, seq]."""
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
    return phi_softmax(scores, axis=-1)[0, :, -1, :]  # [nh, seq]


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


def run_layer_hybrid(engine, h, li, last_row_template, synth_bos_vec):
    """Run layer with:
    - Real softmax attention for non-last positions
    - Fixed template for last-token attention row
    - Synthetic BOS MLP output
    """
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
    weights = phi_softmax(scores, axis=-1)

    # Replace ONLY last-token row with parametric template
    fw = last_row_template  # [nh, fw_seq]
    fw_seq = fw.shape[1]
    if seq_len == fw_seq:
        weights[0, :, -1, :] = fw
    elif seq_len < fw_seq:
        trimmed = fw[:, :seq_len]
        weights[0, :, -1, :] = trimmed / (trimmed.sum(axis=1, keepdims=True) + 1e-12)
    else:
        padded = np.zeros((nh, seq_len), dtype=np.float32)
        padded[:, :fw_seq] = fw
        weights[0, :, -1, :] = padded / (padded.sum(axis=1, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    h_pa = h + phi_linear(attn.W_o, ao)

    # MLP with synthetic BOS
    nm = rms_norm(h_pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
    mlp_out[0, 0, :] = synth_bos_vec
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
    print("  Frontier Optimal: Parametric T(N) Last-Row + All-Layer BOS SV0")
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
    # Step 1: Calibrate parametric T(N) (from F136 approach)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 1: Calibrate parametric T(N) per head (F136)")
    print("=" * 80)

    real_templates = {}  # {seq_len: [layer_templates]}
    for _, prompt in LENGTH_PROMPTS:
        tids = tokenizer.encode(prompt)
        sl = len(tids)
        h = engine.embedding(tids)[np.newaxis, :, :]
        layer_t = []
        for li in range(n_layers):
            w = get_last_token_attention(engine, h, li)
            layer_t.append(w.copy())
            h = engine.layers[li](h)
        real_templates[sl] = layer_t
        print(f"    N={sl} extracted")

    # Fit per-head parametric model: {BOS(N), subj(N), last(N), mid(N)}
    # BOS(N) = a/(1+b*N), subj = mean, last = la/N+lb
    seq_lens = np.array(sorted(real_templates.keys()), dtype=float)
    layer_params = {}

    for li in range(n_layers):
        for hi in range(nh):
            bos_vals, subj_vals, last_vals = [], [], []
            for sl in sorted(real_templates.keys()):
                t = real_templates[int(sl)][li][hi]  # [seq_len]
                bos_vals.append(float(t[0]))
                last_vals.append(float(t[-1]))
                subj_vals.append(float(t[-2]))

            bos_vals = np.array(bos_vals)
            inv_bos = 1.0 / (bos_vals + 1e-12)
            A = np.column_stack([np.ones_like(seq_lens), seq_lens])
            c = np.linalg.lstsq(A, inv_bos, rcond=None)[0]
            a_bos = 1.0 / max(c[0], 0.01)
            b_bos = c[1] / max(c[0], 0.01)

            subj_mean = float(np.mean(subj_vals))
            last_fit = np.polyfit(1.0 / seq_lens, last_vals, 1)

            layer_params[(li, hi)] = {
                'a_bos': a_bos, 'b_bos': b_bos,
                'subj_mean': subj_mean,
                'last_a': last_fit[0], 'last_b': last_fit[1],
            }

    total_param_floats = n_layers * nh * 5
    print(f"  Fitted {n_layers * nh} (layer, head) combos × 5 params = {total_param_floats} floats")

    def generate_last_row_template(li, seq_len):
        """Generate parametric last-token attention row."""
        template = np.zeros((nh, seq_len), dtype=np.float32)
        for hi in range(nh):
            p = layer_params[(li, hi)]
            N = float(seq_len)
            bos = max(0.01, min(0.99, p['a_bos'] / (1 + p['b_bos'] * N)))
            subj = max(0.001, min(0.5, p['subj_mean']))
            last = max(0.001, min(0.5, p['last_a'] / N + p['last_b']))
            remaining = max(0.0, 1.0 - bos - subj - last)
            n_mid = max(seq_len - 3, 0)
            mid = remaining / n_mid if n_mid > 0 else 0.0
            template[hi, 0] = bos
            if n_mid > 0:
                template[hi, 1:-2] = mid
            template[hi, -2] = subj
            template[hi, -1] = last
            row_sum = template[hi].sum()
            if row_sum > 0:
                template[hi] /= row_sum
        return template

    # ═══════════════════════════════════════════════════════════
    # Step 2: Calibrate BOS sv0 pump (F139)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 2: Calibrate BOS sv0 pump (all 28 layers)")
    print("=" * 80)

    # Extract BOS MLP from France
    tids = tokenizer.encode(FACTS['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    bos_mlp = {}
    for li in range(n_layers):
        layer = engine.layers[li]
        attn = layer.attention
        mlp = layer.mlp
        nhl, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nhl // nkv, attn.head_dim
        sl = h.shape[1]
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nhl, hd).transpose(0, 2, 1, 3)
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
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
        bos_mlp[li] = mlp_out[0, 0, :].copy()
        h = h_pa + mlp_out

    # Compute sv0 and scales
    print(f"  Computing sv0 directions...", end="", flush=True)
    synth_sv0 = {}
    sv0_scale_list = []
    for li in range(n_layers):
        sv0 = get_sv0_direction(engine, li)
        if np.dot(sv0, bos_mlp[li]) < 0:
            sv0 = -sv0
        scale = float(np.dot(bos_mlp[li], sv0))
        synth_sv0[li] = scale * sv0
        sv0_scale_list.append(scale)
        gc.collect()
    print(f" done")
    print(f"  28 scale factors computed")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Test all configurations
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 3: All Configurations (N=5)")
    print("=" * 80)

    def test_config(name, use_param_last_row, use_bos_sv0):
        correct = 0
        for country in FACTS:
            tids = tokenizer.encode(FACTS[country])
            sl = len(tids)
            h = engine.embedding(tids)[np.newaxis, :, :]

            if use_param_last_row and use_bos_sv0:
                template = generate_last_row_template
                for li in range(n_layers):
                    h = run_layer_hybrid(engine, h, li,
                                        template(li, sl), synth_sv0[li])
            elif use_param_last_row:
                # Param last-row, real MLP
                for li in range(n_layers):
                    layer = engine.layers[li]
                    attn = layer.attention
                    mlp = layer.mlp
                    nhl, nkv = attn.num_heads, attn.num_kv_heads
                    hpk, hd = nhl // nkv, attn.head_dim
                    normed = rms_norm(h, attn.norm_weight)
                    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nhl, hd).transpose(0, 2, 1, 3)
                    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
                    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
                    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
                    Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
                    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
                    if sl > 1:
                        scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
                    wts = phi_softmax(scores, axis=-1)
                    fw = generate_last_row_template(li, sl)
                    wts[0, :, -1, :] = fw
                    ao = np.einsum('bhqk,bhkd->bhqd', wts, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
                    h = h + phi_linear(attn.W_o, ao)
                    nm = rms_norm(h, mlp.norm_weight)
                    g = phi_linear(mlp.W_gate, nm)
                    u = phi_linear(mlp.W_up, nm)
                    h = h + phi_linear(mlp.W_down, phi_silu(g) * u)
            elif use_bos_sv0:
                # Real attention, sv0 BOS MLP
                for li in range(n_layers):
                    layer = engine.layers[li]
                    attn = layer.attention
                    mlp = layer.mlp
                    nhl, nkv = attn.num_heads, attn.num_kv_heads
                    hpk, hd = nhl // nkv, attn.head_dim
                    normed = rms_norm(h, attn.norm_weight)
                    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nhl, hd).transpose(0, 2, 1, 3)
                    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
                    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
                    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
                    Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
                    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
                    if sl > 1:
                        scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
                    wts = phi_softmax(scores, axis=-1)
                    ao = np.einsum('bhqk,bhkd->bhqd', wts, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
                    h_pa = h + phi_linear(attn.W_o, ao)
                    nm = rms_norm(h_pa, mlp.norm_weight)
                    g = phi_linear(mlp.W_gate, nm)
                    u = phi_linear(mlp.W_up, nm)
                    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
                    mlp_out[0, 0, :] = synth_sv0[li]
                    h = h_pa + mlp_out
            else:
                for li in range(n_layers):
                    h = engine.layers[li](h)

            _, rank = predict(engine, tokenizer, h, ANSWERS[country])
            if rank == 0: correct += 1
        return correct

    configs = [
        ("Baseline",                     False, False),
        ("Parametric T(N) last-row only", True,  False),
        ("BOS sv0 only (all layers)",    False, True),
        ("T(N) + BOS sv0 COMBINED",      True,  True),
    ]

    for name, plr, bsv in configs:
        c = test_config(name, plr, bsv)
        marker = " ◄◄◄" if c == 6 else ""
        print(f"    {name:<40} {c}/6{marker}")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Cross-length generalization (combined)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 4: Cross-Length (T(N) + BOS sv0)")
    print("=" * 80)

    test_prompts = [
        ('N=5', 'The capital of France is', ' Paris'),
        ('N=5', 'The capital of Japan is', ' Tokyo'),
        ('N=5', 'The capital of Germany is', ' Berlin'),
        ('N=7', 'The official capital city of France is', ' Paris'),
        ('N=9', 'The official capital city of the country France is', ' Paris'),
        ('N=6', 'The main capital of France is', ' Paris'),
    ]

    for label, prompt, answer in test_prompts:
        tids = tokenizer.encode(prompt)
        sl = len(tids)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_hybrid(engine, h, li,
                                generate_last_row_template(li, sl), synth_sv0[li])
        top, rank = predict(engine, tokenizer, h, answer)
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {label} (N={sl}): '{top}' {ok}")

    # ═══════════════════════════════════════════════════════════
    # Step 5: Parameter inventory
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 5: Parameter Inventory")
    print("=" * 80)

    print(f"""
  GEOMETRIC CONSTANTS:
    Parametric T(N) last-row:     {total_param_floats} floats ({total_param_floats * 4 / 1024:.1f} KB)
    BOS sv0 scale factors:        28 floats (112 bytes)
    Total:                        {total_param_floats + 28} floats ({(total_param_floats + 28) * 4 / 1024:.1f} KB)

  WHAT'S REPLACED:
    Last-token Q/K routing:       Parametric formula (no Q/K at last pos)
    BOS MLP at all 28 layers:     scale × sv0 (no MLP compute at BOS)

  WHAT'S STILL NEURAL:
    Non-last Q/K routing:         Still needs Q/K weights
    V/O projections:              Still needed (all positions)
    MLP (non-BOS positions):      Still needed
    Embeddings + LM head:         Still needed

  vs F137 (previous best):
    F137: {3920 + 3072} geometric constants, 5/6
    Now:  {total_param_floats + 28} geometric constants
    Improvement: BOS MLP replaced at ALL layers (not just L3)
""")


if __name__ == '__main__':
    main()
