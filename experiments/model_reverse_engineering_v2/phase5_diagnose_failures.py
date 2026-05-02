"""
Phase 5: Diagnose the 4 Failures

The geometric Resonator (Head 6 only) fails on 4/35 prompts.
Prior work (Docs 055, 123, 135, 161) suggests two hypotheses:

  H1: Multi-head — Head 6 alone lacks the information; other routing heads
      (10, 16, 22, 23, 24, 25, 27) carry the missing semantic dimensions.
      (Doc 135: heads specialize by semantic dimension)

  H2: Temporal navigation — "nine→ten" and "Monday→Tuesday" require forward
      tachyon navigation (Doc 055), not content retrieval. The Resonator is
      a retriever, not a temporal navigator.

Diagnostic tests:
  1. For each failing prompt, what does EACH of the 8 routing heads attend to?
  2. Does adding more routing heads (multi-head geometric Resonator) fix failures?
  3. What is the per-head V/O contribution to the correct vs incorrect token?
  4. How much attention mass does Head 6 carry vs other heads for these prompts?
"""

import sys, numpy as np, time, gc
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

ROUTING_HEADS = [6, 10, 16, 22, 23, 24, 25, 27]

FAILING_PROMPTS = [
    'The capital of Germany is',
    'The capital of Australia is',
    'The number after nine is',
    'If today is Monday, tomorrow is',
]

PASSING_PROMPTS = [
    'The capital of France is',
    'The capital of Japan is',
    'The capital of Italy is',
]


def phi_quant(M):
    return np.sign(M) * PHI ** np.round(np.log(np.abs(M) + 1e-20) / LOG_PHI)


def finish_forward(engine, hidden_start, start_layer):
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    s = np.sort(logits[0, -1, :])[::-1]
    return idx, tok, s[0] - s[1]


def extract_head_weights(attn, head_idx, hidden_dim):
    """Extract d_k (with bias), Wo, Wv (no bias), bv for one head."""
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    kv_group = head_idx // heads_per_kv

    I = np.eye(hidden_dim, dtype=np.float32)

    Wk_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wv_nb = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]

        qo_b = phi_linear(attn.W_q, c, attn.b_q)[0].reshape(-1, num_heads, head_dim)
        ko_b = phi_linear(attn.W_k, c, attn.b_k)[0].reshape(-1, num_kv_heads, head_dim)
        Wq_b[:, s:e] = qo_b[:, head_idx, :].T
        Wk_b[:, s:e] = ko_b[:, kv_group, :].T

        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        Wv_nb[:, s:e] = vo[:, kv_group, :].T

    # V bias
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
    bv_full = bv_full.reshape(num_kv_heads, head_dim)
    bv_group = bv_full[kv_group]

    # W_o
    h_in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h_in[0, 0, :] = 0.0
        h_in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h_in)[0, 0, :]

    # d_k from MESH with bias
    MESH_b = Wq_b @ Wk_b.T
    _, _, Vt_b = np.linalg.svd(MESH_b)
    d_k_bias = Wk_b.T @ Vt_b[0, :]
    dk_sign = np.sign(d_k_bias)

    # VO
    VO_full = Wo @ Wv_nb
    bias_out = Wo @ bv_group

    return {
        'dk_sign': dk_sign,
        'd_k_bias': d_k_bias,
        'VO_full': VO_full,
        'bias_out': bias_out,
        'all_neg': bool((d_k_bias < 0).all()),
        'head_idx': head_idx,
    }


def analyze_prompt(engine, tokenizer, all_head_weights, prompt, target_layer=23):
    """Full analysis of one prompt across all routing heads."""
    attn = engine.layers[target_layer].attention
    layer = engine.layers[target_layer]

    p_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode_token(t) for t in p_ids]
    h = engine.embedding(p_ids)[np.newaxis, :, :]

    for lo in engine.layers:
        if lo.layer_idx == target_layer:
            full_out = lo(h.copy())
            break
        h = lo(h)

    # Baseline
    logits_base = finish_forward(engine, full_out, target_layer)
    base_idx, base_tok, base_margin = get_top1(logits_base, tokenizer)

    normed = rms_norm(h, attn.norm_weight)

    result = {
        'prompt': prompt,
        'tokens': tokens,
        'n_tokens': len(p_ids),
        'baseline_top1': base_tok,
        'baseline_top1_idx': base_idx,
        'baseline_margin': base_margin,
        'per_head': {},
    }

    # Per-head analysis
    for hi, hw in all_head_weights.items():
        # Routing: what position does this head select?
        kf = normed[0] @ hw['dk_sign']
        selected_pos = int(np.argmax(kf))

        # Also get raw routing scores per position
        routing_scores = kf.copy()

        h_sel = normed[0, selected_pos, :]
        attn_out = hw['VO_full'] @ h_sel + hw['bias_out']

        # Test this single head's contribution
        pa = h.copy()
        pa[0, -1, :] += attn_out

        mlp = layer.mlp
        nm = rms_norm(pa, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mo = phi_linear(mlp.W_down, phi_silu(g) * u)
        single_out = pa + mo

        logits_single = finish_forward(engine, single_out, target_layer)
        s_idx, s_tok, s_margin = get_top1(logits_single, tokenizer)

        result['per_head'][hi] = {
            'selected_pos': selected_pos,
            'selected_token': tokens[selected_pos] if selected_pos < len(tokens) else '?',
            'routing_scores': routing_scores,
            'top1': s_tok,
            'top1_idx': s_idx,
            'match': s_idx == base_idx,
            'margin': s_margin,
            'all_neg': hw['all_neg'],
            'attn_out_norm': float(np.linalg.norm(attn_out)),
        }

    # Multi-head test: combine all routing heads
    combined_out = np.zeros_like(h[0, -1, :])
    for hi, hw in all_head_weights.items():
        kf = normed[0] @ hw['dk_sign']
        sp = int(np.argmax(kf))
        h_sel = normed[0, sp, :]
        combined_out += hw['VO_full'] @ h_sel + hw['bias_out']

    pa_multi = h.copy()
    pa_multi[0, -1, :] += combined_out
    nm_m = rms_norm(pa_multi, layer.mlp.norm_weight)
    g_m = phi_linear(layer.mlp.W_gate, nm_m)
    u_m = phi_linear(layer.mlp.W_up, nm_m)
    mo_m = phi_linear(layer.mlp.W_down, phi_silu(g_m) * u_m)
    multi_out = pa_multi + mo_m

    logits_multi = finish_forward(engine, multi_out, target_layer)
    m_idx, m_tok, m_margin = get_top1(logits_multi, tokenizer)

    result['multi_head'] = {
        'top1': m_tok,
        'top1_idx': m_idx,
        'match': m_idx == base_idx,
        'margin': m_margin,
    }

    return result


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

    target_layer = 23
    attn = engine.layers[target_layer].attention
    hidden_dim = engine.hidden_dim

    # Extract weights for all 8 routing heads
    all_head_weights = {}
    for hi in ROUTING_HEADS:
        print(f"Extracting head {hi}...", flush=True)
        all_head_weights[hi] = extract_head_weights(attn, hi, hidden_dim)
        print(f"  all_neg={all_head_weights[hi]['all_neg']}", flush=True)

    print(f"\nWeights extracted in {time.time()-t0:.1f}s", flush=True)

    # Analyze failing prompts
    print("\n" + "=" * 100)
    print("  FAILING PROMPTS — MULTI-HEAD DIAGNOSIS")
    print("=" * 100)

    for prompt in FAILING_PROMPTS:
        r = analyze_prompt(engine, tokenizer, all_head_weights, prompt)

        print(f"\n{'─' * 100}")
        print(f"  Prompt: \"{r['prompt']}\"")
        print(f"  Tokens: {r['tokens']}")
        print(f"  Baseline: {r['baseline_top1']} (margin={r['baseline_margin']:.4f})")
        print(f"  Multi-head ({len(ROUTING_HEADS)} heads): {r['multi_head']['top1']} "
              f"{'✓' if r['multi_head']['match'] else '✗'} (margin={r['multi_head']['margin']:.4f})")
        print(f"{'─' * 100}")

        print(f"  {'Head':>4s} {'sel_pos':>7s} {'sel_token':>12s} {'top1':>12s} {'match':>5s} "
              f"{'margin':>8s} {'||out||':>8s} {'all_neg':>7s}")
        for hi in ROUTING_HEADS:
            ph = r['per_head'][hi]
            m = "✓" if ph['match'] else "✗"
            print(f"  {hi:4d} {ph['selected_pos']:7d} {ph['selected_token']:>12s} "
                  f"{ph['top1']:>12s} {m:>5s} {ph['margin']:8.4f} "
                  f"{ph['attn_out_norm']:8.2f} {str(ph['all_neg']):>7s}")

        # Show routing scores for Head 6
        h6 = r['per_head'][6]
        print(f"\n  Head 6 routing scores per position:")
        for pos in range(r['n_tokens']):
            score = h6['routing_scores'][pos]
            marker = " ← SELECTED" if pos == h6['selected_pos'] else ""
            print(f"    pos {pos} ({r['tokens'][pos]:>12s}): {score:10.2f}{marker}")

    # Analyze passing prompts for comparison
    print("\n\n" + "=" * 100)
    print("  PASSING PROMPTS — COMPARISON")
    print("=" * 100)

    for prompt in PASSING_PROMPTS:
        r = analyze_prompt(engine, tokenizer, all_head_weights, prompt)

        print(f"\n{'─' * 100}")
        print(f"  Prompt: \"{r['prompt']}\"")
        print(f"  Baseline: {r['baseline_top1']} (margin={r['baseline_margin']:.4f})")
        print(f"  Multi-head ({len(ROUTING_HEADS)} heads): {r['multi_head']['top1']} "
              f"{'✓' if r['multi_head']['match'] else '✗'} (margin={r['multi_head']['margin']:.4f})")

        print(f"  {'Head':>4s} {'sel_pos':>7s} {'sel_token':>12s} {'top1':>12s} {'match':>5s} "
              f"{'margin':>8s} {'||out||':>8s}")
        for hi in ROUTING_HEADS:
            ph = r['per_head'][hi]
            m = "✓" if ph['match'] else "✗"
            print(f"  {hi:4d} {ph['selected_pos']:7d} {ph['selected_token']:>12s} "
                  f"{ph['top1']:>12s} {m:>5s} {ph['margin']:8.4f} "
                  f"{ph['attn_out_norm']:8.2f}")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print("\n  Key question: Does multi-head geometric Resonator fix the failures?")
    for prompt in FAILING_PROMPTS:
        r = analyze_prompt(engine, tokenizer, all_head_weights, prompt)
        m = "✓ FIXED" if r['multi_head']['match'] else "✗ STILL FAILS"
        print(f"    {prompt:45s} → {r['multi_head']['top1']:>12s} {m}")

    print("\n  Does multi-head break any passing prompts?")
    for prompt in PASSING_PROMPTS:
        r = analyze_prompt(engine, tokenizer, all_head_weights, prompt)
        m = "✓ STILL PASSES" if r['multi_head']['match'] else "✗ BROKEN"
        print(f"    {prompt:45s} → {r['multi_head']['top1']:>12s} {m}")

    print("\n" + "=" * 100, flush=True)


if __name__ == '__main__':
    main()
