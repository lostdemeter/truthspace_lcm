"""
Frontier 1c: Parametric All-Position Templates T(N, q)
========================================================
Extend parametric template generation from last-token-only (F136)
to ALL query positions.

Key insight from F138: attention at all positions follows:
  - Heavy BOS sink (varies by layer and position)
  - Self-attention component
  - Small spread to intermediate positions

Model: For each (layer, head, query_position q):
  w(q, 0) = BOS_frac(layer, head, q, N)
  w(q, q) = self_frac(layer, head, q, N)
  w(q, k) = spread for 0 < k < q, k != q

We fit these from multiple sequence lengths.
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

# Prompts at different lengths for fitting
CALIBRATION_PROMPTS = [
    'The capital of France is',                              # N=5
    'The official capital city of France is',                # N=7
    'The official capital city of the country France is',    # N=9
]


def decode_weight(w):
    return w.decode() if isinstance(w, PhiEncoded) else w


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


def run_layer_with_full_template(engine, h, li, template):
    """Run layer replacing ALL attention with template + BOS sv0 MLP."""
    layer = engine.layers[li]
    attn = layer.attention
    mlp = layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    normed = rms_norm(h, attn.norm_weight)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Ve = np.repeat(V, hpk, axis=1)

    w = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
    ts = template.shape[1]
    if seq_len == ts:
        w[0] = template
    elif seq_len < ts:
        w[0] = template[:, :seq_len, :seq_len]
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
    else:
        # Extend template for longer sequences
        w[0, :, :ts, :ts] = template
        for p in range(ts, seq_len):
            w[0, :, p, p] = 1.0
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    h_pa = h + phi_linear(attn.W_o, ao)

    nm = rms_norm(h_pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    h = h_pa + phi_linear(mlp.W_down, phi_silu(g) * u)
    return h


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
    print("  Frontier 1c: Parametric All-Position Templates T(N, q)")
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
    # Step 1: Extract full attention at multiple lengths
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 1: Extract full attention at multiple lengths")
    print("=" * 80)

    real_attn = {}  # {seq_len: {layer: [nh, seq, seq]}}
    for prompt in CALIBRATION_PROMPTS:
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        h = engine.embedding(tids)[np.newaxis, :, :]
        real_attn[seq_len] = {}
        for li in range(n_layers):
            real_attn[seq_len][li] = get_full_attention(engine, h, li)
            h = engine.layers[li](h)
        print(f"    N={seq_len} extracted")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Analyze per-position structure across lengths
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 2: Per-position attention structure across lengths")
    print("=" * 80)

    # For each layer, each head, each relative position:
    # w(q, 0) = BOS fraction
    # w(q, q) = self fraction
    # How do these vary with q and N?

    sample_layers = [0, 3, 10, 20, 23, 27]
    for li in sample_layers:
        print(f"\n  Layer {li} — BOS fraction by position (head-averaged):")
        print(f"  {'q\\N':<6}", end="")
        for sl in sorted(real_attn.keys()):
            print(f"  {'N='+str(sl):>7}", end="")
        print()

        max_pos = max(real_attn.keys())
        for q in range(max_pos):
            print(f"  q={q:<4}", end="")
            for sl in sorted(real_attn.keys()):
                if q < sl:
                    bos_frac = float(real_attn[sl][li][:, q, 0].mean())
                    print(f"  {bos_frac:>7.4f}", end="")
                else:
                    print(f"  {'—':>7}", end="")
            print()

    # ═══════════════════════════════════════════════════════════
    # Step 3: Fit parametric model per layer per head
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 3: Fit parametric T(N, q) per layer per head")
    print("=" * 80)

    # Model: for query position q in a sequence of length N:
    #   BOS(q, N) = a_bos / (1 + b_bos * q)  [decays with position]
    #   self(q, N) = a_self * (q / N)         [grows with relative position]
    #   spread = uniform over remaining positions
    #
    # Per-head parameters: a_bos, b_bos, a_self per (layer, head)

    seq_lens = sorted(real_attn.keys())

    # Collect all (q, N, bos_frac, self_frac) observations per (layer, head)
    layer_head_params = {}  # {(li, hi): {a_bos, b_bos, a_self}}

    for li in range(n_layers):
        for hi in range(nh):
            obs_bos = []  # (q, N, bos_frac)
            obs_self = []  # (q, N, self_frac)

            for sl in seq_lens:
                attn_mat = real_attn[sl][li][hi]  # [seq, seq]
                for q in range(sl):
                    bos_frac = float(attn_mat[q, 0])
                    self_frac = float(attn_mat[q, q])
                    obs_bos.append((q, sl, bos_frac))
                    obs_self.append((q, sl, self_frac))

            # Fit BOS: BOS(q, N) = a / (1 + b*q)
            # Linearize: 1/BOS = 1/a + (b/a)*q
            qs = np.array([o[0] for o in obs_bos], dtype=float)
            bos_vals = np.array([o[2] for o in obs_bos], dtype=float)
            bos_vals = np.clip(bos_vals, 1e-6, 1.0)

            # Simple least squares: 1/BOS = c0 + c1*q
            A = np.column_stack([np.ones_like(qs), qs])
            inv_bos = 1.0 / bos_vals
            try:
                c = np.linalg.lstsq(A, inv_bos, rcond=None)[0]
                a_bos = 1.0 / max(c[0], 0.01)
                b_bos = c[1] / max(c[0], 0.01)
            except:
                a_bos = float(bos_vals.mean())
                b_bos = 0.0

            # Fit self: self(q, N) ~ a_self for q > 0 (constant-ish)
            self_vals = np.array([o[2] for o in obs_self if o[0] > 0])
            a_self = float(self_vals.mean()) if len(self_vals) > 0 else 0.1

            layer_head_params[(li, hi)] = {
                'a_bos': a_bos, 'b_bos': b_bos, 'a_self': a_self
            }

    # Print sample
    for li in [0, 10, 23, 27]:
        p = layer_head_params[(li, 0)]
        print(f"  L{li} h0: BOS={p['a_bos']:.4f}/(1+{p['b_bos']:.4f}*q)  self={p['a_self']:.4f}")

    # Count total parameters
    total_params = n_layers * nh * 3
    print(f"\n  Total parametric constants: {total_params} "
          f"({total_params * 4 / 1024:.1f} KB)")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Generate parametric templates and compare to real
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 4: Parametric vs real template comparison")
    print("=" * 80)

    def generate_full_template(li, seq_len):
        """Generate [nh, seq, seq] parametric template."""
        template = np.zeros((nh, seq_len, seq_len), dtype=np.float32)
        for hi in range(nh):
            p = layer_head_params[(li, hi)]
            for q in range(seq_len):
                if q == 0:
                    template[hi, 0, 0] = 1.0
                else:
                    bos = p['a_bos'] / (1 + p['b_bos'] * q)
                    bos = max(0.01, min(0.99, bos))
                    self_w = p['a_self']
                    self_w = max(0.01, min(0.5, self_w))

                    remaining = max(0.0, 1.0 - bos - self_w)
                    n_mid = max(q - 1, 0)  # positions between BOS and self
                    mid = remaining / n_mid if n_mid > 0 else 0.0

                    template[hi, q, 0] = bos
                    if n_mid > 0:
                        template[hi, q, 1:q] = mid
                    template[hi, q, q] = self_w

                    # Renormalize
                    row_sum = template[hi, q, :q+1].sum()
                    if row_sum > 0:
                        template[hi, q, :q+1] /= row_sum
        return template

    # Compare at each calibration length
    for sl in seq_lens:
        cos_layers = []
        for li in range(n_layers):
            real = real_attn[sl][li]  # [nh, seq, seq]
            synth = generate_full_template(li, sl)
            # Flatten and compare
            cos = float(np.dot(real.ravel(), synth.ravel()) /
                       (np.linalg.norm(real.ravel()) * np.linalg.norm(synth.ravel()) + 1e-12))
            cos_layers.append(cos)
        mean_cos = np.mean(cos_layers)
        min_cos = min(cos_layers)
        print(f"  N={sl}: mean cos = {mean_cos:.4f}, min = {min_cos:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Step 5: Test parametric templates at calibration length
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 5: Parametric all-position templates → prediction (N=5)")
    print("=" * 80)

    param_templates = {li: generate_full_template(li, 5) for li in range(n_layers)}

    correct = 0
    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_full_template(engine, h, li, param_templates[li])
        top, rank = predict(engine, tokenizer, h, ANSWERS[country])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0: correct += 1
        print(f"    {country:>8}: '{top}' {ok}")
    print(f"\n  Parametric T(N,q) at N=5: {correct}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 6: Cross-length generalization
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 6: Cross-length generalization with T(N, q)")
    print("=" * 80)

    test_prompts = [
        ('N=5',  'The capital of France is', ' Paris'),
        ('N=5',  'The capital of Japan is', ' Tokyo'),
        ('N=5',  'The capital of Germany is', ' Berlin'),
        ('N=7',  'The official capital city of France is', ' Paris'),
        ('N=9',  'The official capital city of the country France is', ' Paris'),
        ('N=6',  'The main capital of France is', ' Paris'),   # unseen length
        ('N=8',  'The official main capital city of France is', ' Paris'),  # unseen length
    ]

    for label, prompt, answer in test_prompts:
        tids = tokenizer.encode(prompt)
        sl = len(tids)
        templates_sl = {li: generate_full_template(li, sl) for li in range(n_layers)}
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_full_template(engine, h, li, templates_sl[li])
        top, rank = predict(engine, tokenizer, h, answer)
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {label} (N={sl}): '{top}' {ok}  [{prompt}]")

    # ═══════════════════════════════════════════════════════════
    # Step 7: Parameter summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)

    print(f"""
  Parametric T(N, q) model:
    Parameters per (layer, head): 3 (a_bos, b_bos, a_self)
    Total: {n_layers} layers × {nh} heads × 3 = {total_params} floats ({total_params * 4 / 1024:.1f} KB)
    Generates full [N × N] attention matrix for any N

  This replaces:
    Q weights: {n_layers} × (3584 × 3584) = {n_layers * 3584 * 3584:,} params
    K weights: {n_layers} × (3584 × 3584) = {n_layers * 3584 * 3584:,} params
    Q/K biases: {n_layers} × 2 × 3584 = {n_layers * 2 * 3584:,} params
    Total Q/K: ~411M params
    Compression: {411_000_000 // total_params:,}:1
""")


if __name__ == '__main__':
    main()
