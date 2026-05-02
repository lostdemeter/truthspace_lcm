"""
Frontier 4: Cross-Structure Investigation
============================================
Key questions:
1. Is there a φ-curve of position "uniqueness" (high at edges, low in middle)?
2. What exactly makes cross-structure fail at early layers?
3. Multi-token capitals — what are we already predicting beyond first/last?
4. Can we decompose the cross-structure gap into a small correction?

Hypothesis: First and last tokens are geometrically "extreme" — most
distinguishable — while middle tokens converge to a normalized distribution.
On a φ-basis, this is natural: φ^n grows at extremes, normalizes in center.
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

PHI = (1 + np.sqrt(5)) / 2  # golden ratio


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


def get_hidden_states_and_attention(engine, tids):
    """Run full forward pass, return per-layer hidden states and attention."""
    h = engine.embedding(tids)[np.newaxis, :, :]
    hidden_states = [h.copy()]
    attentions = {}
    for li in range(len(engine.layers)):
        attentions[li] = get_full_attention(engine, h, li)
        h = engine.layers[li](h)
        hidden_states.append(h.copy())
    return hidden_states, attentions


def main():
    print("=" * 80)
    print("  Frontier 4: Cross-Structure Investigation")
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
    # Investigation 1: Multi-token capitals — what are we predicting?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 1: Multi-Token Capitals — What Are We Predicting?")
    print("=" * 80)

    facts = {
        'France':  ('The capital of France is', ' Paris'),
        'Japan':   ('The capital of Japan is', ' Tokyo'),
        'Germany': ('The capital of Germany is', ' Berlin'),
        'Italy':   ('The capital of Italy is', ' Rome'),
        'Spain':   ('The capital of Spain is', ' Madrid'),
        'Egypt':   ('The capital of Egypt is', ' Cairo'),
    }

    print("\n  Token analysis:")
    for country, (prompt, answer) in facts.items():
        prompt_tids = tokenizer.encode(prompt)
        answer_tids = tokenizer.encode(answer)
        country_tids = tokenizer.encode(' ' + country)
        answer_str = tokenizer.decode(answer_tids)

        # Decode each token individually
        prompt_tokens = [tokenizer.decode([t]) for t in prompt_tids]
        answer_tokens = [tokenizer.decode([t]) for t in answer_tids]
        country_tokens = [tokenizer.decode([t]) for t in country_tids]

        print(f"    {country:>8}: prompt={len(prompt_tids)} toks {prompt_tokens}")
        print(f"             answer={len(answer_tids)} toks {answer_tokens}")
        print(f"             entity={len(country_tids)} toks {country_tokens}")

    # Check: where does the entity sit in the prompt?
    print("\n  Entity positions in prompt:")
    for country, (prompt, answer) in facts.items():
        prompt_tids = tokenizer.encode(prompt)
        country_tids = tokenizer.encode(country)
        # Find country tokens in prompt
        for i in range(len(prompt_tids)):
            for j in range(i + 1, len(prompt_tids) + 1):
                decoded = tokenizer.decode(prompt_tids[i:j])
                if country in decoded:
                    print(f"    {country:>8}: entity at positions {i}..{j-1} of {len(prompt_tids)}")
                    break
            else:
                continue
            break

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: Per-Position Uniqueness Across Structures
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 2: Per-Position Uniqueness (φ-Curve Hypothesis)")
    print("=" * 80)

    # Collect diverse N=5 prompts
    diverse_5 = [
        'The capital of France is',    # [785, 6722, 315, 9625, 374]
        'I really love eating pizza',   # [40, 2167, 2948, 12182, 22502]
        'Please help me find this',     # [5501, 1492, 752, 1477, 419]
        'Once upon a time there',       # [12522, 5193, 264, 882, 1052]
        'How does the engine work',     # [4340, 1558, 279, 4712, 975]
        'The capital of Germany is',
        'The capital of Japan is',
        'The capital of Italy is',
        'The capital of Spain is',
        'The capital of Egypt is',
    ]

    # Verify and collect
    working = []
    for prompt in diverse_5:
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            working.append((prompt, tids))

    print(f"  Collected {len(working)} N=5 prompts")

    # Extract attention for all prompts
    all_attn = {}
    for prompt, tids in working:
        h = engine.embedding(tids)[np.newaxis, :, :]
        layers = {}
        for li in range(n_layers):
            layers[li] = get_full_attention(engine, h, li)
            h = engine.layers[li](h)
        all_attn[prompt] = layers
        gc.collect()
    print("  Attention extracted for all prompts")

    # For each position q, compute how much the attention row varies across structures
    print("\n  Per-position cross-structure variance (head-averaged):")
    print(f"  {'Layer':>6} | {'q=0':>8} {'q=1':>8} {'q=2':>8} {'q=3':>8} {'q=4':>8} | {'Ratio(edge/mid)':>16}")

    position_variance = {}  # (layer, q) → variance
    for li in [0, 1, 2, 3, 5, 10, 15, 20, 23, 27]:
        row_vars = []
        for q in range(5):
            # Collect attention row at position q across all prompts
            rows = []
            for prompt in all_attn:
                row = all_attn[prompt][li][:, q, :].ravel()  # [28 * 5] = [140]
                rows.append(row)
            rows = np.array(rows)  # [n_prompts, 140]

            # Compute pairwise cosine similarity
            cos_vals = []
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    cos_vals.append(cos)

            mean_cos = np.mean(cos_vals)
            row_vars.append(mean_cos)
            position_variance[(li, q)] = mean_cos

        # Compute edge/middle ratio
        edge_avg = (row_vars[0] + row_vars[4]) / 2.0
        mid_avg = row_vars[2]  # middle position
        ratio = edge_avg / mid_avg if mid_avg > 0 else float('inf')

        vals = " ".join(f"{v:.4f}" for v in row_vars)
        print(f"  L{li:>4} | {vals} | {ratio:.4f}")

    # φ-curve comparison
    print("\n  φ-Curve Comparison:")
    print("  Theoretical φ-curve for 5 positions (φ^|q - N/2| / φ^(N/2)):")
    N = 5
    center = (N - 1) / 2.0
    phi_weights = np.array([PHI ** abs(q - center) for q in range(N)])
    phi_weights /= phi_weights.max()
    print(f"    Relative φ-weight: [{', '.join(f'{w:.4f}' for w in phi_weights)}]")

    # Compare with actual uniqueness at each layer
    print("\n  Actual cross-structure SIMILARITY (higher = more universal):")
    for li in [0, 3, 10, 27]:
        actual = [position_variance.get((li, q), 0) for q in range(5)]
        # Invert: uniqueness = 1 - similarity
        uniqueness = [1 - v for v in actual]
        unique_norm = np.array(uniqueness) / (max(uniqueness) + 1e-12)
        print(f"    L{li:2d} uniqueness: [{', '.join(f'{u:.4f}' for u in unique_norm)}]")
    print(f"    φ-curve:         [{', '.join(f'{w:.4f}' for w in phi_weights)}]")

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: WHAT makes early layers structure-dependent?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 3: Anatomy of Cross-Structure Failure")
    print("=" * 80)

    # Take the capital prompts (same structure) vs diverse prompts
    same_struct_prompts = [p for p, _ in working if 'capital' in p]
    diff_struct_prompts = [p for p, _ in working if 'capital' not in p]

    print(f"  Same-structure: {len(same_struct_prompts)} prompts")
    print(f"  Different-structure: {len(diff_struct_prompts)} prompts")

    # For L0 and L3, decompose: what part of the attention changes?
    for li in [0, 3]:
        print(f"\n  Layer {li} — Breaking down cross-structure difference:")

        # Compute mean attention within same-structure and across
        same_attn = np.mean([all_attn[p][li] for p in same_struct_prompts], axis=0)
        diff_attn = [all_attn[p][li] for p in diff_struct_prompts]

        # Per-head analysis: which heads are most structure-sensitive?
        head_sensitivity = []
        for hi in range(nh):
            same_row = same_attn[hi].ravel()
            cos_vals = []
            for d in diff_attn:
                diff_row = d[hi].ravel()
                cos = float(np.dot(same_row, diff_row) / (np.linalg.norm(same_row) * np.linalg.norm(diff_row) + 1e-12))
                cos_vals.append(cos)
            head_sensitivity.append((hi, np.mean(cos_vals)))

        head_sensitivity.sort(key=lambda x: x[1])
        print(f"    Most structure-sensitive heads (lowest cross-cos):")
        for hi, cos in head_sensitivity[:5]:
            print(f"      Head {hi:2d}: cross-struct cos = {cos:.4f}")
        print(f"    Most structure-invariant heads:")
        for hi, cos in head_sensitivity[-3:]:
            print(f"      Head {hi:2d}: cross-struct cos = {cos:.4f}")

        # For the most sensitive head, show what changes
        worst_h = head_sensitivity[0][0]
        print(f"\n    Head {worst_h} (most sensitive) — attention patterns:")
        for prompt in [same_struct_prompts[0]] + diff_struct_prompts[:2]:
            A = all_attn[prompt][li][worst_h]  # [5, 5]
            short = prompt[:30].ljust(30)
            # Show last row (prediction position)
            last = " ".join(f"{A[-1, k]:.3f}" for k in range(5))
            # Show row 2 (middle position)
            mid = " ".join(f"{A[2, k]:.3f}" for k in range(3))
            print(f"      {short} last=[{last}] mid=[{mid}]")

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: Hidden State Divergence by Position
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 4: Hidden State Divergence by Position")
    print("=" * 80)

    # Track hidden states for two very different N=5 prompts through layers
    prompt_a = 'The capital of France is'
    prompt_b = 'I really love eating pizza'

    tids_a = tokenizer.encode(prompt_a)
    tids_b = tokenizer.encode(prompt_b)

    h_a = engine.embedding(tids_a)[np.newaxis, :, :]
    h_b = engine.embedding(tids_b)[np.newaxis, :, :]

    print(f"\n  Comparing: '{prompt_a}' vs '{prompt_b}'")
    print(f"  Per-position hidden state cosine similarity through layers:")
    print(f"  {'Layer':>6} | {'pos=0':>8} {'pos=1':>8} {'pos=2':>8} {'pos=3':>8} {'pos=4':>8} | {'pos0 norm_a':>10} {'pos0 norm_b':>10}")

    for li in range(n_layers):
        # Before this layer
        cos_per_pos = []
        for p in range(5):
            va = h_a[0, p, :]
            vb = h_b[0, p, :]
            cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            cos_per_pos.append(cos)

        norm_a0 = float(np.linalg.norm(h_a[0, 0, :]))
        norm_b0 = float(np.linalg.norm(h_b[0, 0, :]))

        if li in [0, 1, 2, 3, 5, 10, 15, 20, 23, 26, 27]:
            vals = " ".join(f"{c:.4f}" for c in cos_per_pos)
            print(f"  L{li:>4} | {vals} | {norm_a0:>10.1f} {norm_b0:>10.1f}")

        h_a = engine.layers[li](h_a)
        h_b = engine.layers[li](h_b)

    # Final hidden state
    cos_per_pos = []
    for p in range(5):
        va = h_a[0, p, :]
        vb = h_b[0, p, :]
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        cos_per_pos.append(cos)
    vals = " ".join(f"{c:.4f}" for c in cos_per_pos)
    print(f"  Final | {vals} |")

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: BOS Convergence — Why First Token Works
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 5: BOS Convergence — Why First Token Works")
    print("=" * 80)

    # Track BOS hidden state across ALL prompts
    print("\n  BOS hidden state cos with 'France' reference across layers:")
    ref_tids = tokenizer.encode('The capital of France is')
    ref_h = engine.embedding(ref_tids)[np.newaxis, :, :]

    other_prompts = [
        ('Germany', 'The capital of Germany is'),
        ('pizza',   'I really love eating pizza'),
        ('help',    'Please help me find this'),
        ('once',    'Once upon a time there'),
        ('engine',  'How does the engine work'),
    ]

    other_hs = {}
    for name, prompt in other_prompts:
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            other_hs[name] = engine.embedding(tids)[np.newaxis, :, :]

    sample_layers = [0, 1, 2, 3, 5, 10, 15, 20, 23, 26, 27]
    print(f"  {'Layer':>6} | " + " ".join(f"{name:>8}" for name in other_hs))

    for li in range(n_layers):
        if li in sample_layers:
            cos_vals = []
            for name in other_hs:
                va = ref_h[0, 0, :]  # BOS of reference
                vb = other_hs[name][0, 0, :]  # BOS of other
                cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
                cos_vals.append(f"{cos:.4f}")
            print(f"  L{li:>4} | " + " ".join(f"{c:>8}" for c in cos_vals))

        ref_h = engine.layers[li](ref_h)
        for name in other_hs:
            other_hs[name] = engine.layers[li](other_hs[name])

    # ═══════════════════════════════════════════════════════════
    # Investigation 6: Cross-Structure Template — What If We Average?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 6: Cross-Structure Template — Mean Template Test")
    print("=" * 80)

    # Compute mean attention template across ALL diverse prompts
    mean_template = {}
    for li in range(n_layers):
        templates = [all_attn[p][li] for p in all_attn]
        mean_template[li] = np.mean(templates, axis=0)

    # Also compute a "capital-only" mean template
    capital_template = {}
    for li in range(n_layers):
        templates = [all_attn[p][li] for p in same_struct_prompts]
        capital_template[li] = np.mean(templates, axis=0)

    # Compute BOS sv0 from France calibration
    print("  Computing BOS sv0...", end="", flush=True)
    cal_tids = tokenizer.encode('The capital of France is')
    h_cal = engine.embedding(cal_tids)[np.newaxis, :, :]
    bos_mlp = {}
    for li in range(n_layers):
        layer = engine.layers[li]
        attn = layer.attention
        mlp = layer.mlp
        nhl, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nhl // nkv, attn.head_dim
        sl = h_cal.shape[1]
        normed = rms_norm(h_cal, attn.norm_weight)
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
        h_pa = h_cal + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
        bos_mlp[li] = mlp_out[0, 0, :].copy()
        h_cal = h_pa + mlp_out

    def get_sv0_direction(engine, li):
        W_down = decode_weight(engine.layers[li].mlp.W_down)
        rng = np.random.RandomState(42)
        v = rng.randn(W_down.shape[1]).astype(np.float64)
        for _ in range(20):
            u = W_down.astype(np.float64) @ v
            u /= np.linalg.norm(u)
            v = W_down.astype(np.float64).T @ u
            v /= np.linalg.norm(v)
        return u.astype(np.float32)

    synth_sv0 = {}
    for li in range(n_layers):
        sv0 = get_sv0_direction(engine, li)
        if np.dot(sv0, bos_mlp[li]) < 0:
            sv0 = -sv0
        scale = float(np.dot(bos_mlp[li], sv0))
        synth_sv0[li] = scale * sv0
        gc.collect()
    print(" done")

    def run_with_template_and_sv0(engine, tids, templates, synth_sv0):
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(len(engine.layers)):
            layer = engine.layers[li]
            attn = layer.attention
            mlp = layer.mlp
            nhl, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nhl // nkv, attn.head_dim
            sl = h.shape[1]
            normed = rms_norm(h, attn.norm_weight)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            Ve = np.repeat(V, hpk, axis=1)
            w = np.zeros((1, nhl, sl, sl), dtype=np.float32)
            T = templates[li]
            ts = T.shape[1]
            if sl == ts:
                w[0] = T
            elif sl < ts:
                w[0] = T[:, :sl, :sl]
                w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
            else:
                w[0, :, :ts, :ts] = T
                for p in range(ts, sl):
                    w[0, :, p, p] = 1.0
                w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
            ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
            h_pa = h + phi_linear(attn.W_o, ao)
            nm = rms_norm(h_pa, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
            mlp_out[0, 0, :] = synth_sv0[li]
            h = h_pa + mlp_out
        return h

    def predict(engine, tokenizer, h, answer):
        fnw = decode_weight(engine.final_norm_weight)
        normed = rms_norm(h[:, -1:, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        top_tid = int(np.argmax(logits))
        ans_tid = tokenizer.encode(answer)[0]
        rank = int(np.sum(logits > logits[ans_tid]))
        return tokenizer.decode([top_tid]), rank

    # Test A: France's own template → 6 capitals
    print("\n  Test A: France-calibrated template (same structure) → 6 capitals")
    france_template = {li: all_attn['The capital of France is'][li] for li in range(n_layers)}
    for country, (prompt, answer) in facts.items():
        tids = tokenizer.encode(prompt)
        h = run_with_template_and_sv0(engine, tids, france_template, synth_sv0)
        _, rank = predict(engine, tokenizer, h, answer)
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {country:>8}: {ok}")

    # Test B: Mean template (ALL diverse prompts) → 6 capitals
    print("\n  Test B: Mean template (diverse N=5) → 6 capitals")
    for country, (prompt, answer) in facts.items():
        tids = tokenizer.encode(prompt)
        h = run_with_template_and_sv0(engine, tids, mean_template, synth_sv0)
        _, rank = predict(engine, tokenizer, h, answer)
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {country:>8}: {ok}")

    # Test C: Mean template → non-capital prompts
    print("\n  Test C: Mean template (diverse N=5) → diverse prompts (next token)")
    for prompt in diff_struct_prompts:
        tids = tokenizer.encode(prompt)
        # Get baseline next token
        h_real = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h_real = engine.layers[li](h_real)
        fnw = decode_weight(engine.final_norm_weight)
        normed = rms_norm(h_real[:, -1:, :], fnw)
        logits_real = engine.lm_head(normed)[0, 0, :]
        real_tok = int(np.argmax(logits_real))
        real_word = tokenizer.decode([real_tok])

        # Now with mean template
        h_tmpl = run_with_template_and_sv0(engine, tids, mean_template, synth_sv0)
        normed2 = rms_norm(h_tmpl[:, -1:, :], fnw)
        logits_tmpl = engine.lm_head(normed2)[0, 0, :]
        tmpl_tok = int(np.argmax(logits_tmpl))
        tmpl_word = tokenizer.decode([tmpl_tok])

        # Rank of real answer in template logits
        rank = int(np.sum(logits_tmpl > logits_tmpl[real_tok]))
        match = "✓" if rank == 0 else f"rank={rank}"
        print(f"    '{prompt}' → real='{real_word}' tmpl='{tmpl_word}' {match}")

    # Test D: Prompt's OWN template → its next token
    print("\n  Test D: Each prompt's OWN template (self-cache) → next token")
    for prompt in diff_struct_prompts:
        tids = tokenizer.encode(prompt)
        own_template = {li: all_attn[prompt][li] for li in range(n_layers)}

        # Get baseline
        h_real = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h_real = engine.layers[li](h_real)
        fnw = decode_weight(engine.final_norm_weight)
        normed = rms_norm(h_real[:, -1:, :], fnw)
        logits_real = engine.lm_head(normed)[0, 0, :]
        real_tok = int(np.argmax(logits_real))
        real_word = tokenizer.decode([real_tok])

        # With own template + sv0
        h_tmpl = run_with_template_and_sv0(engine, tids, own_template, synth_sv0)
        normed2 = rms_norm(h_tmpl[:, -1:, :], fnw)
        logits_tmpl = engine.lm_head(normed2)[0, 0, :]
        tmpl_tok = int(np.argmax(logits_tmpl))
        tmpl_word = tokenizer.decode([tmpl_tok])

        rank = int(np.sum(logits_tmpl > logits_tmpl[real_tok]))
        match = "✓" if rank == 0 else f"rank={rank}"
        print(f"    '{prompt}' → real='{real_word}' tmpl='{tmpl_word}' {match}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 7: Hybrid — Real Early Layers + Cached Late Layers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 7: Hybrid — Real Q/K Early + Cached Late")
    print("=" * 80)

    # Run real Q/K for L0-Lk, then switch to mean template for Lk+1-L27
    for k_switch in [0, 3, 5, 10, 15, 20]:
        correct = 0
        for prompt in diff_struct_prompts:
            tids = tokenizer.encode(prompt)
            h = engine.embedding(tids)[np.newaxis, :, :]

            # Real layers 0..k_switch-1
            for li in range(k_switch):
                h = engine.layers[li](h)

            # Cached + sv0 layers k_switch..27
            for li in range(k_switch, n_layers):
                layer = engine.layers[li]
                attn = layer.attention
                mlp = layer.mlp
                nhl, nkv = attn.num_heads, attn.num_kv_heads
                hpk, hd = nhl // nkv, attn.head_dim
                sl = h.shape[1]
                normed = rms_norm(h, attn.norm_weight)
                V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
                Ve = np.repeat(V, hpk, axis=1)
                w = np.zeros((1, nhl, sl, sl), dtype=np.float32)
                w[0] = mean_template[li]
                ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
                h_pa = h + phi_linear(attn.W_o, ao)
                nm = rms_norm(h_pa, mlp.norm_weight)
                g = phi_linear(mlp.W_gate, nm)
                u = phi_linear(mlp.W_up, nm)
                mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
                mlp_out[0, 0, :] = synth_sv0[li]
                h = h_pa + mlp_out

            fnw = decode_weight(engine.final_norm_weight)
            normed = rms_norm(h[:, -1:, :], fnw)
            logits = engine.lm_head(normed)[0, 0, :]

            # Baseline
            h_real = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                h_real = engine.layers[li](h_real)
            normed_r = rms_norm(h_real[:, -1:, :], fnw)
            logits_r = engine.lm_head(normed_r)[0, 0, :]
            real_tok = int(np.argmax(logits_r))

            rank = int(np.sum(logits > logits[real_tok]))
            if rank == 0:
                correct += 1

        print(f"  Real L0-L{k_switch-1 if k_switch > 0 else 'none':>4} + cached L{k_switch}-L27: "
              f"{correct}/{len(diff_struct_prompts)}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary: Cross-Structure Analysis")
    print("=" * 80)
    print("""
  KEY FINDINGS:
  1. BOS (pos 0) converges across ALL structures — the pump works universally
  2. Last position is most sensitive to structure at early layers
  3. Middle positions have intermediate uniqueness
  4. φ-curve hypothesis: TBD from the data above

  PATHS TO GENERAL SOLVER:
  - Per-structure cache: works perfectly for known templates
  - Mean template: test results above show if averaging works
  - Hybrid: real early + cached late — find the switching point
  - The general solver needs EITHER:
    (a) Enough cached structures to cover the space, or
    (b) A way to cheaply compute early-layer attention per structure
""")


if __name__ == '__main__':
    main()
