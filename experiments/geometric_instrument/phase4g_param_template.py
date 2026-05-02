"""
Phase 4g: Parametric Template Generator
=========================================

Correct implementation: compute real softmax for all positions,
only replace the LAST-TOKEN ROW with parametric template.

Uses Phase 4c's proven approach (F132: 5/6).

Tests:
  1. Control: Real France templates → all 6 prompts (should match F132)
  2. Parametric templates at known lengths (5, 7, 9, 11 tok)
  3. Per-layer parametric templates (different formula per layer)
  4. Parametric templates at held-out length
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
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}

LENGTH_PROMPTS = [
    ('5tok', 'The capital of France is'),
    ('7tok', 'I know the capital of France is'),
    ('9tok', 'Can you tell me the capital of France is'),
    ('11tok', 'Please can you tell me what the capital of France is'),
]


def cos_sim(a, b):
    return float(np.dot(a.ravel(), b.ravel()) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def get_last_token_attention(engine, h, layer_idx):
    """Extract last-token attention weights [nh, seq_len]."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if seq_len > 1:
        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    weights = phi_softmax(scores, axis=-1)
    return weights[0, :, -1, :]  # [nh, seq_len]


def run_layer_with_fixed_last_row(engine, h, layer_idx, fixed_weights):
    """Run a layer replacing ONLY last-token attention with fixed weights.
    Real softmax for all other positions. (Phase 4c approach)"""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    V = phi_linear(attn.W_v, normed, attn.b_v)
    Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    Ve = np.repeat(V, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if seq_len > 1:
        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    weights = phi_softmax(scores, axis=-1)

    # Replace ONLY last-token row with fixed template
    fw = fixed_weights  # [nh, fw_seq]
    cur_seq, fw_seq = seq_len, fw.shape[1]
    if cur_seq == fw_seq:
        weights[0, :, -1, :] = fw
    elif cur_seq < fw_seq:
        trimmed = fw[:, :cur_seq]
        weights[0, :, -1, :] = trimmed / (trimmed.sum(axis=1, keepdims=True) + 1e-12)
    else:
        padded = np.zeros((nh, cur_seq), dtype=np.float32)
        padded[:, :fw_seq] = fw
        weights[0, :, -1, :] = padded / (padded.sum(axis=1, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    ao = phi_linear(attn.W_o, ao)
    h_post_attn = h + ao

    mlp = layer.mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    h_out = h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)
    return h_out


def run_with_templates(engine, token_ids, templates):
    """Run model with fixed last-token-row templates at all layers."""
    h = engine.embedding(token_ids)[np.newaxis, :, :]
    for li in range(len(engine.layers)):
        h = run_layer_with_fixed_last_row(engine, h, li, templates[li])
    return h


def predict(engine, tokenizer, h, answer):
    """Get prediction and rank of expected answer."""
    normed = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = engine.lm_head(normed)[0, 0, :]
    top_tid = int(np.argmax(logits))
    answer_tid = tokenizer.encode(answer)[0]
    rank = int(np.sum(logits > logits[answer_tid]))
    top_tok = tokenizer.decode([top_tid])
    return top_tok, rank


def main():
    print("=" * 80)
    print("  Phase 4g: Parametric Template Generator")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    n_layers = len(engine.layers)
    nh = 28

    # ═══════════════════════════════════════════════════════════
    # Step 1: Extract real templates at multiple lengths
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 1: Extract real templates at multiple lengths")
    print("=" * 80)

    real_templates = {}  # {seq_len: [templates_per_layer]}

    for label, prompt in LENGTH_PROMPTS:
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        print(f"    Extracting {label} (N={seq_len})...", end="", flush=True)
        t0 = time.time()

        h = engine.embedding(tids)[np.newaxis, :, :]
        layer_templates = []
        for li in range(n_layers):
            w_lt = get_last_token_attention(engine, h, li)
            layer_templates.append(w_lt.copy())
            h = engine.layers[li](h)
        real_templates[seq_len] = layer_templates
        print(f" {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Control — real France templates → all prompts (F132 check)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 2: Control — Real France templates (N=5) → all prompts")
    print("=" * 80)

    templates_5 = real_templates[5]
    correct = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = run_with_templates(engine, tids, templates_5)
        top, rank = predict(engine, tokenizer, h, info['answer'])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0:
            correct += 1
        print(f"    {country}: '{top}' {ok}")
    print(f"\n  Control: {correct}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Analyze per-layer template structure
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 3: Per-layer template structure (BOS fraction)")
    print("=" * 80)

    print(f"\n  {'Layer':>5}  ", end="")
    for sl in sorted(real_templates.keys()):
        print(f"{'N='+str(sl):>8}", end="")
    print()
    print("  " + "─" * 50)

    # Store per-layer BOS fractions for fitting
    layer_bos = {li: {} for li in range(n_layers)}  # {layer: {seq_len: bos_frac}}

    for li in [0, 3, 5, 10, 15, 20, 23, 27]:
        print(f"  L{li:>3}:  ", end="")
        for sl in sorted(real_templates.keys()):
            t = real_templates[sl][li]  # [nh, seq_len]
            bos = float(t[:, 0].mean())
            layer_bos[li][sl] = bos
            print(f"  {bos:>6.4f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════
    # Step 4: Fit parametric model per layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 4: Fit parametric templates per layer")
    print("=" * 80)

    # For each layer, fit: last-token template = {BOS(N), mid(N), subj(N), last(N)}
    # Use all 4 lengths for fitting
    layer_params = {}  # {layer: {a_bos, b_bos, subj_mean, last_a, last_b}}

    for li in range(n_layers):
        lens = np.array(sorted(real_templates.keys()), dtype=float)
        bos_vals = []
        subj_vals = []
        last_vals = []
        mid_vals = []

        for sl in sorted(real_templates.keys()):
            t = real_templates[int(sl)][li]  # [nh, seq_len]
            avg = t.mean(axis=0)  # [seq_len]
            bos_vals.append(float(avg[0]))
            last_vals.append(float(avg[-1]))
            subj_vals.append(float(avg[-2]))
            if int(sl) > 3:
                mid_vals.append(float(avg[1:-2].mean()))
            else:
                mid_vals.append(0.0)

        bos_vals = np.array(bos_vals)
        subj_vals = np.array(subj_vals)
        last_vals = np.array(last_vals)

        # BOS(N) = a / (1 + b*N) → 1/BOS = 1/a + (b/a)*N
        inv_bos = 1.0 / (bos_vals + 1e-12)
        A = np.column_stack([np.ones_like(lens), lens])
        c0, c1 = np.linalg.lstsq(A, inv_bos, rcond=None)[0]
        a_bos = 1.0 / (c0 + 1e-12)
        b_bos = c1 / (c0 + 1e-12)

        subj_mean = float(subj_vals.mean())

        # last(N) = la/N + lb
        last_fit = np.polyfit(1.0 / lens, last_vals, 1)

        layer_params[li] = {
            'a_bos': a_bos, 'b_bos': b_bos,
            'subj_mean': subj_mean,
            'last_a': last_fit[0], 'last_b': last_fit[1],
        }

    # Print sample layers
    for li in [0, 3, 10, 23, 27]:
        p = layer_params[li]
        print(f"  L{li:>2}: BOS={p['a_bos']:.3f}/(1+{p['b_bos']:.4f}*N)  "
              f"subj={p['subj_mean']:.4f}  last={p['last_a']:.4f}/N+{p['last_b']:.4f}")

    def generate_parametric_template(layer_idx, seq_len, n_heads=28):
        """Generate a parametric template for one layer at given seq_len."""
        p = layer_params[layer_idx]
        N = float(seq_len)
        bos = p['a_bos'] / (1 + p['b_bos'] * N)
        subj = p['subj_mean']
        last = p['last_a'] / N + p['last_b']

        # Clamp to valid range
        bos = max(0.01, min(0.99, bos))
        subj = max(0.001, min(0.5, subj))
        last = max(0.001, min(0.5, last))

        remaining = max(0.0, 1.0 - bos - subj - last)
        n_mid = max(seq_len - 3, 0)
        mid = remaining / n_mid if n_mid > 0 else 0.0

        template = np.zeros((n_heads, seq_len), dtype=np.float32)
        template[:, 0] = bos
        if n_mid > 0:
            template[:, 1:-2] = mid
        template[:, -2] = subj
        template[:, -1] = last

        # Renormalize
        row_sums = template.sum(axis=1, keepdims=True)
        template = template / (row_sums + 1e-12)
        return template

    # ═══════════════════════════════════════════════════════════
    # Step 5: Compare parametric vs real templates
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 5: Parametric vs real template similarity")
    print("=" * 80)

    print(f"\n  {'Layer':>5}  ", end="")
    for sl in sorted(real_templates.keys()):
        print(f"  {'N='+str(sl):>6}", end="")
    print()
    print("  " + "─" * 45)

    for li in [0, 3, 10, 15, 20, 23, 27]:
        print(f"  L{li:>3}:  ", end="")
        for sl in sorted(real_templates.keys()):
            real = real_templates[sl][li].mean(axis=0)
            synth = generate_parametric_template(li, sl).mean(axis=0)
            cos = cos_sim(real, synth)
            print(f"  {cos:>6.4f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════
    # Step 6: Test parametric templates for prediction (same length)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 6: Parametric templates → predictions (same length, N=5)")
    print("=" * 80)

    # Generate parametric templates for N=5
    param_templates_5 = [generate_parametric_template(li, 5) for li in range(n_layers)]

    correct_param = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = run_with_templates(engine, tids, param_templates_5)
        top, rank = predict(engine, tokenizer, h, info['answer'])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0:
            correct_param += 1
        print(f"    {country}: '{top}' {ok}")
    print(f"\n  Parametric N=5: {correct_param}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 7: Test per-head parametric (not averaged)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 7: Per-head parametric templates")
    print("=" * 80)

    # Fit per-head parameters
    head_params = {}  # {(layer, head): {a_bos, b_bos, ...}}

    for li in range(n_layers):
        for hi in range(nh):
            lens = np.array(sorted(real_templates.keys()), dtype=float)
            bos_vals = []
            subj_vals = []
            last_vals = []

            for sl in sorted(real_templates.keys()):
                t = real_templates[int(sl)][li][hi, :]  # [seq_len] for this head
                bos_vals.append(float(t[0]))
                last_vals.append(float(t[-1]))
                subj_vals.append(float(t[-2]))

            bos_vals = np.array(bos_vals)
            subj_vals = np.array(subj_vals)
            last_vals = np.array(last_vals)

            inv_bos = 1.0 / (bos_vals + 1e-12)
            A = np.column_stack([np.ones_like(lens), lens])
            c0, c1 = np.linalg.lstsq(A, inv_bos, rcond=None)[0]
            a_bos = 1.0 / (c0 + 1e-12)
            b_bos = c1 / (c0 + 1e-12)
            subj_mean = float(subj_vals.mean())
            last_fit = np.polyfit(1.0 / lens, last_vals, 1)

            head_params[(li, hi)] = {
                'a_bos': a_bos, 'b_bos': b_bos,
                'subj_mean': subj_mean,
                'last_a': last_fit[0], 'last_b': last_fit[1],
            }

    def generate_perhead_template(layer_idx, seq_len):
        """Generate per-head parametric template."""
        template = np.zeros((nh, seq_len), dtype=np.float32)
        for hi in range(nh):
            p = head_params[(layer_idx, hi)]
            N = float(seq_len)
            bos = p['a_bos'] / (1 + p['b_bos'] * N)
            subj = p['subj_mean']
            last = p['last_a'] / N + p['last_b']
            bos = max(0.001, min(0.999, bos))
            subj = max(0.001, min(0.5, subj))
            last = max(0.001, min(0.5, last))
            remaining = max(0.0, 1.0 - bos - subj - last)
            n_mid = max(seq_len - 3, 0)
            mid = remaining / n_mid if n_mid > 0 else 0.0

            template[hi, 0] = bos
            if n_mid > 0:
                template[hi, 1:-2] = mid
            template[hi, -2] = subj
            template[hi, -1] = last

        # Renormalize per head
        row_sums = template.sum(axis=1, keepdims=True)
        template = template / (row_sums + 1e-12)
        return template

    # Test per-head parametric at N=5
    perhead_templates_5 = [generate_perhead_template(li, 5) for li in range(n_layers)]

    correct_ph = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = run_with_templates(engine, tids, perhead_templates_5)
        top, rank = predict(engine, tokenizer, h, info['answer'])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0:
            correct_ph += 1
        print(f"    {country}: '{top}' {ok}")
    print(f"\n  Per-head parametric N=5: {correct_ph}/6")

    # Test per-head parametric at different lengths (France only)
    print(f"\n  Per-head parametric → France at different lengths:")
    for label, prompt in LENGTH_PROMPTS:
        tids = tokenizer.encode(prompt)
        sl = len(tids)
        ph_templates = [generate_perhead_template(li, sl) for li in range(n_layers)]
        h = run_with_templates(engine, tids, ph_templates)
        top, rank = predict(engine, tokenizer, h, ' Paris')
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {label} (N={sl}): '{top}' {ok}")

    # ═══════════════════════════════════════════════════════════
    # Step 8: Hybrid — parametric at L0-L21, real at L22-L27
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 8: Hybrid — parametric L0-L21, real L22-L27")
    print("=" * 80)

    correct_hyb = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        sl = len(tids)

        # Build hybrid: parametric for decomposition, real for extraction
        hybrid_templates = []
        for li in range(n_layers):
            if li < 22:
                hybrid_templates.append(generate_perhead_template(li, sl))
            else:
                hybrid_templates.append(real_templates[sl][li])
        
        h = run_with_templates(engine, tids, hybrid_templates)
        top, rank = predict(engine, tokenizer, h, info['answer'])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0:
            correct_hyb += 1
        print(f"    {country}: '{top}' {ok}")
    print(f"\n  Hybrid (parametric L0-21, real L22-27): {correct_hyb}/6")

    # ═══════════════════════════════════════════════════════════
    # Step 9: Interpolation test — generate for N=6 (unseen)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 9: Interpolation — parametric for unseen length N=6")
    print("=" * 80)

    # "I think the capital of France is" — should tokenize to 6 tokens
    # Let's find a prompt that tokenizes to 6
    test_prompts_6 = [
        'Tell me the capital of France is',
        'I think the capital of France is',
        'The main capital of France is',
    ]
    for prompt in test_prompts_6:
        tids = tokenizer.encode(prompt)
        sl = len(tids)
        if sl == 6:
            print(f"  Found N=6 prompt: '{prompt}'")
            ph6 = [generate_perhead_template(li, 6) for li in range(n_layers)]
            h = run_with_templates(engine, tids, ph6)
            top, rank = predict(engine, tokenizer, h, ' Paris')
            ok = "✓" if rank == 0 else f"rank={rank}"
            print(f"    Parametric N=6: '{top}' {ok}")

            # Also test with real softmax (baseline)
            h_base = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                h_base = engine.layers[li](h_base)
            top_base, rank_base = predict(engine, tokenizer, h_base, ' Paris')
            ok_base = "✓" if rank_base == 0 else f"rank={rank_base}"
            print(f"    Real softmax:    '{top_base}' {ok_base}")
            break
        else:
            print(f"  '{prompt}' → {sl} tokens (not 6)")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print(f"\n  Control (real France templates):       {correct}/6")
    print(f"  Parametric (avg heads, N=5):           {correct_param}/6")
    print(f"  Per-head parametric (N=5):             {correct_ph}/6")
    print(f"  Hybrid (param L0-21, real L22-27):     {correct_hyb}/6")

    print(f"\n  Template formula per layer (sample):")
    for li in [0, 23, 27]:
        p = layer_params[li]
        print(f"    L{li}: BOS = {p['a_bos']:.3f} / (1 + {p['b_bos']:.4f} * N)")

    print(f"\n  Total parameters for T(N) per-head: "
          f"{n_layers * nh * 5} scalars = {n_layers * nh * 5 * 4} bytes")
    print()


if __name__ == '__main__':
    main()
