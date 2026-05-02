"""
Phase 5: Resonator Validation on Broad Prompt Set

Tests the fully geometric Resonator (Finding 45):
  sign(d_k_bias) = all -1s  →  argmax(-Σ h[pos])
  φ-quant VO + φ-quant bias  →  U_φ @ diag(S_φ) @ V_φ @ h_sel + bias_φ

Compares against full φ-engine baseline on 30+ prompts across categories:
  - Factual recall (capitals, geography, science)
  - Completion (idioms, poetry, common phrases)
  - Entity retrieval (people, organizations)
  - Logical/arithmetic
  - Longer multi-token contexts

Reports: per-prompt match, margin, aggregate accuracy, category breakdown.
"""

import sys, numpy as np, time, gc, json
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# ─────────────────────────────────────────────────────────────────────────────
#  Prompt categories
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    'factual_capitals': [
        'The capital of France is',
        'The capital of Japan is',
        'The capital of Germany is',
        'The capital of Italy is',
        'The capital of Australia is',
    ],
    'factual_geography': [
        'The largest ocean is the',
        'The longest river in the world is the',
        'The tallest mountain in the world is',
        'The smallest continent is',
        'The driest desert in the world is the',
    ],
    'factual_science': [
        'The color of grass is',
        'Water freezes at zero degrees',
        'The speed of light is approximately',
        'The chemical symbol for gold is',
        'The nearest star to Earth is the',
    ],
    'completion_idioms': [
        'To be or not to',
        'Roses are red, violets are',
        'An apple a day keeps the',
        'The early bird catches the',
        'All that glitters is not',
    ],
    'entity_people': [
        'Barack Obama was the',
        'Albert Einstein developed the theory of',
        'The first person to walk on the moon was',
        'William Shakespeare wrote',
        'Isaac Newton discovered the law of',
    ],
    'logical_arithmetic': [
        '2 + 2 =',
        'The opposite of hot is',
        'The number after nine is',
        'If today is Monday, tomorrow is',
        'The square root of 144 is',
    ],
    'longer_context': [
        'In the year 1969, the United States successfully landed astronauts on the',
        'The theory of evolution by natural selection was proposed by Charles',
        'According to Einstein, energy equals mass times the speed of light',
        'The Great Wall of China was built to protect against',
        'In computer science, the time complexity of binary search is',
    ],
}


def phi_quant(M):
    return np.sign(M) * PHI ** np.round(np.log(np.abs(M) + 1e-20) / LOG_PHI)


def finish_forward(engine, hidden_start, start_layer):
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def get_topk(logits, tokenizer, k=5):
    """Return top-k (idx, token, logit) tuples and margin."""
    last = logits[0, -1, :]
    top_idx = np.argsort(last)[::-1][:k]
    margin = float(last[top_idx[0]] - last[top_idx[1]])
    results = []
    for i in top_idx:
        results.append((int(i), tokenizer.decode_token(int(i)), float(last[i])))
    return results, margin


def extract_resonator_weights(engine, target_layer=23, head_idx=6):
    """Extract all geometric Resonator components: d_k, VO, bias."""
    attn = engine.layers[target_layer].attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    kv_group = head_idx // heads_per_kv
    hidden_dim = engine.hidden_dim

    I = np.eye(hidden_dim, dtype=np.float32)

    # Extract W_k with bias (for correct d_k) and W_v without bias (for clean VO)
    Wk_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq_b = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wv_nb = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]

        # With bias (for routing)
        qo_b = phi_linear(attn.W_q, c, attn.b_q)[0].reshape(-1, num_heads, head_dim)
        ko_b = phi_linear(attn.W_k, c, attn.b_k)[0].reshape(-1, num_kv_heads, head_dim)
        Wq_b[:, s:e] = qo_b[:, head_idx, :].T
        Wk_b[:, s:e] = ko_b[:, kv_group, :].T

        # Without bias (for V/O)
        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        Wv_nb[:, s:e] = vo[:, kv_group, :].T

        if s % 1024 == 0:
            print(f"  {e}/{hidden_dim}...", flush=True)

    # V bias
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
    bv_full = bv_full.reshape(num_kv_heads, head_dim)
    bv_group = bv_full[kv_group]

    # W_o for head 6
    h6in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h6in[0, 0, :] = 0.0
        h6in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h6in)[0, 0, :]

    # d_k from bias-included MESH
    MESH_b = Wq_b @ Wk_b.T
    _, _, Vt_b = np.linalg.svd(MESH_b)
    d_k_bias = Wk_b.T @ Vt_b[0, :]
    dk_sign = np.sign(d_k_bias)  # all -1s

    # φ-quant VO
    VO_full = Wo @ Wv_nb
    Uvo, Svo, Vtvo = np.linalg.svd(VO_full, full_matrices=False)
    S128 = Svo[:128]
    U_phi = phi_quant(Uvo[:, :128])
    Vt_phi = phi_quant(Vtvo[:128, :])
    S_phi = PHI ** np.round(np.log(S128 + 1e-20) / LOG_PHI)
    VO_phiq = (U_phi * S_phi[np.newaxis, :]) @ Vt_phi

    # φ-quant bias
    bias_out = Wo @ bv_group
    bias_out_phi = phi_quant(bias_out)

    return {
        'dk_sign': dk_sign,
        'VO_phiq': VO_phiq,
        'bias_out_phi': bias_out_phi,
        'VO_full': VO_full,
        'bias_out': bias_out,
        'd_k_bias': d_k_bias,
        'all_neg': bool((d_k_bias < 0).all()),
    }


def run_resonator_test(engine, tokenizer, weights, prompt, target_layer=23):
    """Run one prompt through baseline and geometric Resonator, return results."""
    attn = engine.layers[target_layer].attention
    layer = engine.layers[target_layer]

    p_ids = tokenizer.encode(prompt)
    h = engine.embedding(p_ids)[np.newaxis, :, :]

    # Forward through layers up to target
    for lo in engine.layers:
        if lo.layer_idx == target_layer:
            full_out = lo(h.copy())
            break
        h = lo(h)

    # ── Baseline: full model from target layer ──
    logits_base = finish_forward(engine, full_out, target_layer)
    topk_base, margin_base = get_topk(logits_base, tokenizer)

    # ── Geometric Resonator: sign(d_k_bias) + φ-quant VO + φ-quant bias ──
    normed = rms_norm(h, attn.norm_weight)

    # Routing: sign(d_k_bias) = all -1s → argmax(-Σ h_dim)
    kf = normed[0] @ weights['dk_sign']
    selected_pos = int(np.argmax(kf))

    h_sel = normed[0, selected_pos, :]
    attn_out = weights['VO_phiq'] @ h_sel + weights['bias_out_phi']

    # Rebuild from geometric output
    pa = h.copy()
    pa[0, -1, :] += attn_out

    mlp = layer.mlp
    nm = rms_norm(pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mo = phi_linear(mlp.W_down, phi_silu(g) * u)
    geo_out = pa + mo

    logits_geo = finish_forward(engine, geo_out, target_layer)
    topk_geo, margin_geo = get_topk(logits_geo, tokenizer)

    # Also test with float32 VO + float32 bias (to separate routing from VO)
    attn_out_f32 = weights['VO_full'] @ h_sel + weights['bias_out']
    pa2 = h.copy()
    pa2[0, -1, :] += attn_out_f32
    nm2 = rms_norm(pa2, mlp.norm_weight)
    g2 = phi_linear(mlp.W_gate, nm2)
    u2 = phi_linear(mlp.W_up, nm2)
    mo2 = phi_linear(mlp.W_down, phi_silu(g2) * u2)
    f32_out = pa2 + mo2
    logits_f32 = finish_forward(engine, f32_out, target_layer)
    topk_f32, margin_f32 = get_topk(logits_f32, tokenizer)

    # Logit correlation
    base_last = logits_base[0, -1, :]
    geo_last = logits_geo[0, -1, :]
    corr = np.corrcoef(base_last, geo_last)[0, 1]

    return {
        'prompt': prompt,
        'n_tokens': len(p_ids),
        'selected_pos': selected_pos,
        'selected_token': tokenizer.decode_token(p_ids[selected_pos]),
        'baseline_top1': topk_base[0][1],
        'baseline_top1_idx': topk_base[0][0],
        'baseline_margin': margin_base,
        'geo_phiq_top1': topk_geo[0][1],
        'geo_phiq_top1_idx': topk_geo[0][0],
        'geo_phiq_margin': margin_geo,
        'geo_f32_top1': topk_f32[0][1],
        'geo_f32_top1_idx': topk_f32[0][0],
        'geo_f32_margin': margin_f32,
        'match_phiq': topk_geo[0][0] == topk_base[0][0],
        'match_f32': topk_f32[0][0] == topk_base[0][0],
        'logit_corr': float(corr),
    }


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

    # ── Extract Resonator weights ──
    print("Extracting Resonator weights...", flush=True)
    t1 = time.time()
    weights = extract_resonator_weights(engine)
    print(f"Extracted in {time.time()-t1:.1f}s", flush=True)
    print(f"  d_k all negative: {weights['all_neg']}")
    print(f"  VO_phiq shape: {weights['VO_phiq'].shape}")

    # ── Run all prompts ──
    print("\n" + "=" * 100)
    print("  PHASE 5: RESONATOR VALIDATION")
    print("=" * 100)

    all_results = []
    category_stats = {}

    for cat, prompts in PROMPTS.items():
        print(f"\n{'─' * 100}")
        print(f"  Category: {cat} ({len(prompts)} prompts)")
        print(f"{'─' * 100}")

        cat_match_phiq = 0
        cat_match_f32 = 0

        for prompt in prompts:
            r = run_resonator_test(engine, tokenizer, weights, prompt)
            all_results.append({**r, 'category': cat})

            m_phiq = "✓" if r['match_phiq'] else "✗"
            m_f32 = "✓" if r['match_f32'] else "✗"

            print(f"  {prompt:55s} base={r['baseline_top1']:>12s}  "
                  f"φ-geo={r['geo_phiq_top1']:>12s} {m_phiq}  "
                  f"f32-geo={r['geo_f32_top1']:>12s} {m_f32}  "
                  f"r={r['logit_corr']:.4f}  pos={r['selected_pos']}→'{r['selected_token']}'",
                  flush=True)

            if r['match_phiq']: cat_match_phiq += 1
            if r['match_f32']: cat_match_f32 += 1

        cat_n = len(prompts)
        category_stats[cat] = {
            'n': cat_n,
            'match_phiq': cat_match_phiq,
            'match_f32': cat_match_f32,
            'pct_phiq': cat_match_phiq / cat_n * 100,
            'pct_f32': cat_match_f32 / cat_n * 100,
        }

    # ── Summary ──
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    total = len(all_results)
    total_phiq = sum(1 for r in all_results if r['match_phiq'])
    total_f32 = sum(1 for r in all_results if r['match_f32'])
    avg_corr = np.mean([r['logit_corr'] for r in all_results])
    avg_margin_phiq = np.mean([r['geo_phiq_margin'] for r in all_results if r['match_phiq']] or [0])
    avg_margin_base = np.mean([r['baseline_margin'] for r in all_results])

    print(f"\n  Total prompts:        {total}")
    print(f"  φ-quant Resonator:    {total_phiq}/{total} = {total_phiq/total*100:.1f}%")
    print(f"  float32 Resonator:    {total_f32}/{total} = {total_f32/total*100:.1f}%")
    print(f"  Avg logit correlation: {avg_corr:.4f}")
    print(f"  Avg baseline margin:   {avg_margin_base:.4f}")
    print(f"  Avg φ-geo margin (matched): {avg_margin_phiq:.4f}")

    print(f"\n  Per-category breakdown:")
    print(f"  {'Category':25s} {'φ-quant':>10s} {'float32':>10s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10}")
    for cat, s in category_stats.items():
        print(f"  {cat:25s} {s['match_phiq']}/{s['n']} ({s['pct_phiq']:5.1f}%)  "
              f"{s['match_f32']}/{s['n']} ({s['pct_f32']:5.1f}%)")

    # ── Failures analysis ──
    failures_phiq = [r for r in all_results if not r['match_phiq']]
    if failures_phiq:
        print(f"\n  φ-quant Resonator FAILURES ({len(failures_phiq)}):")
        for r in failures_phiq:
            print(f"    {r['prompt']:55s} base={r['baseline_top1']:>12s}  "
                  f"geo={r['geo_phiq_top1']:>12s}  r={r['logit_corr']:.4f}  "
                  f"pos={r['selected_pos']}→'{r['selected_token']}'")

    failures_f32 = [r for r in all_results if not r['match_f32']]
    if failures_f32:
        print(f"\n  float32 Resonator FAILURES ({len(failures_f32)}):")
        for r in failures_f32:
            print(f"    {r['prompt']:55s} base={r['baseline_top1']:>12s}  "
                  f"geo={r['geo_f32_top1']:>12s}  r={r['logit_corr']:.4f}  "
                  f"pos={r['selected_pos']}→'{r['selected_token']}'")

    # ── Save results ──
    results_path = 'experiments/model_reverse_engineering_v2/results/phase5_resonator_validation.json'
    import os
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'total': total,
            'match_phiq': total_phiq,
            'match_f32': total_f32,
            'pct_phiq': total_phiq / total * 100,
            'pct_f32': total_f32 / total * 100,
            'avg_logit_corr': float(avg_corr),
            'category_stats': category_stats,
            'per_prompt': all_results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    print("\n" + "=" * 100)
    print("  DONE")
    print("=" * 100, flush=True)


if __name__ == '__main__':
    main()
