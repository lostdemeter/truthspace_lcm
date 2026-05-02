"""
Phase 5: Full 28-Head Geometric Resonator — Target 100%

Doc 228 V15 insight: "Holographic bounds don't exist."
Don't approximate — extract and run. Attention IS geometric.

From Finding 38:
  - 20 FIXED heads: always attend to position 0 (BOS), entropy < 0.5
  - 8 ROUTING heads: [6, 10, 16, 22, 23, 24, 25, 27]
  - Fixed heads ALONE get 6/6 (margin 0.101)
  - All 28 heads: 6/6, margin 0.601

Strategy:
  - ALL 28 heads contribute via VO @ h[pos] + bias
  - Fixed heads: pos = 0 (BOS token, always)
  - Routing heads: pos = argmax(h @ sign(d_k_bias))
  - This is complete geometric attention — no approximation

From Doc 192 (Boom-Newton): attention is sparse, 89.5% at 37% positions.
From Doc 209 (Dimensional Casting): attention IS moment projection.
From Doc 208 (Context Window): position 0 gets 55% of attention.
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

ROUTING_HEADS = [6, 10, 16, 22, 23, 24, 25, 27]
FIXED_HEADS = [h for h in range(28) if h not in ROUTING_HEADS]
ALL_HEADS = list(range(28))

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


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    s = np.sort(logits[0, -1, :])[::-1]
    return idx, tok, s[0] - s[1]


def extract_all_head_weights(engine, target_layer=23):
    """Extract VO + bias + routing for all 28 heads.
    
    Efficient: W_v is shared per KV group (4 groups for 28 heads).
    W_o is per-head but extractable column by column.
    """
    attn = engine.layers[target_layer].attention
    head_dim = attn.head_dim       # 128
    num_heads = attn.num_heads     # 28
    num_kv_heads = attn.num_kv_heads  # 4
    heads_per_kv = num_heads // num_kv_heads  # 7
    hidden_dim = engine.hidden_dim  # 3584
    
    I = np.eye(hidden_dim, dtype=np.float32)
    
    # Step 1: Extract W_v (no bias) for each KV group — only 4 unique
    print("  Extracting W_v for 4 KV groups...", flush=True)
    Wv_groups = {}  # kv_group -> (head_dim, hidden_dim)
    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]
        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        for g in range(num_kv_heads):
            if g not in Wv_groups:
                Wv_groups[g] = np.zeros((head_dim, hidden_dim), dtype=np.float32)
            Wv_groups[g][:, s:e] = vo[:, g, :].T
    
    # V bias per KV group
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
    bv_full = bv_full.reshape(num_kv_heads, head_dim)
    
    # Step 2: For routing heads, extract W_q and W_k with bias for MESH
    print("  Extracting W_q, W_k (with bias) for routing heads...", flush=True)
    routing_Wq = {}  # head_idx -> (head_dim, hidden_dim)
    routing_Wk = {}  # kv_group -> (head_dim, hidden_dim), only need unique groups
    routing_kv_groups = set()
    for hi in ROUTING_HEADS:
        g = hi // heads_per_kv
        routing_kv_groups.add(g)
    
    for hi in ROUTING_HEADS:
        routing_Wq[hi] = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    for g in routing_kv_groups:
        routing_Wk[g] = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    
    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]
        qo_b = phi_linear(attn.W_q, c, attn.b_q)[0].reshape(-1, num_heads, head_dim)
        ko_b = phi_linear(attn.W_k, c, attn.b_k)[0].reshape(-1, num_kv_heads, head_dim)
        for hi in ROUTING_HEADS:
            routing_Wq[hi][:, s:e] = qo_b[:, hi, :].T
        for g in routing_kv_groups:
            routing_Wk[g][:, s:e] = ko_b[:, g, :].T
    
    # Step 3: Extract W_o for all 28 heads
    print("  Extracting W_o for all 28 heads...", flush=True)
    h_in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo_all = {}  # head_idx -> (hidden_dim, head_dim)
    for hi in ALL_HEADS:
        Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
        for d in range(head_dim):
            h_in[0, 0, :] = 0.0
            h_in[0, 0, hi * head_dim + d] = 1.0
            Wo[:, d] = phi_linear(attn.W_o, h_in)[0, 0, :]
        Wo_all[hi] = Wo
        if (hi + 1) % 7 == 0:
            print(f"    W_o: {hi+1}/28 heads done", flush=True)
    
    # Step 4: Compute VO + bias + routing for each head
    print("  Computing VO matrices and routing vectors...", flush=True)
    head_weights = {}
    
    for hi in ALL_HEADS:
        g = hi // heads_per_kv
        Wv = Wv_groups[g]
        Wo = Wo_all[hi]
        bv = bv_full[g]
        
        VO = Wo @ Wv
        bias_out = Wo @ bv
        
        hw = {
            'VO': VO,
            'bias_out': bias_out,
            'head_idx': hi,
            'kv_group': g,
            'is_routing': hi in ROUTING_HEADS,
        }
        
        if hi in ROUTING_HEADS:
            # Compute d_k from MESH with bias
            Wq = routing_Wq[hi]
            Wk = routing_Wk[g]
            MESH = Wq @ Wk.T
            _, _, Vt = np.linalg.svd(MESH)
            d_k_bias = Wk.T @ Vt[0, :]
            dk_sign = np.sign(d_k_bias)
            hw['dk_sign'] = dk_sign
            hw['all_neg'] = bool((d_k_bias < 0).all())
        else:
            # Fixed head: always attend to position 0
            hw['dk_sign'] = None
            hw['all_neg'] = None
        
        head_weights[hi] = hw
    
    return head_weights


def run_full_resonator(engine, tokenizer, head_weights, prompt, target_layer=23):
    """Run prompt through the complete 28-head geometric Resonator."""
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
    
    # ── 28-head geometric Resonator ──
    combined_out = np.zeros(engine.hidden_dim, dtype=np.float32)
    
    for hi in ALL_HEADS:
        hw = head_weights[hi]
        
        if hw['is_routing']:
            # Routing head: select position via sign(d_k)
            kf = normed[0] @ hw['dk_sign']
            pos = int(np.argmax(kf))
        else:
            # Fixed head: always position 0
            pos = 0
        
        h_sel = normed[0, pos, :]
        combined_out += hw['VO'] @ h_sel + hw['bias_out']
    
    # Rebuild through MLP
    pa = h.copy()
    pa[0, -1, :] += combined_out
    
    mlp = layer.mlp
    nm = rms_norm(pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mo = phi_linear(mlp.W_down, phi_silu(g) * u)
    geo_out = pa + mo
    
    logits_geo = finish_forward(engine, geo_out, target_layer)
    geo_idx, geo_tok, geo_margin = get_top1(logits_geo, tokenizer)
    
    # Logit correlation
    corr = np.corrcoef(logits_base[0, -1, :], logits_geo[0, -1, :])[0, 1]
    
    return {
        'prompt': prompt,
        'tokens': tokens,
        'n_tokens': len(p_ids),
        'baseline_top1': base_tok,
        'baseline_top1_idx': base_idx,
        'baseline_margin': base_margin,
        'geo_top1': geo_tok,
        'geo_top1_idx': geo_idx,
        'geo_margin': geo_margin,
        'match': geo_idx == base_idx,
        'logit_corr': float(corr),
    }


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)
    
    # Extract weights for all 28 heads
    print("Extracting weights for all 28 heads...", flush=True)
    t1 = time.time()
    head_weights = extract_all_head_weights(engine)
    print(f"Extracted in {time.time()-t1:.1f}s\n", flush=True)
    
    # Report head classification
    print("Head classification:")
    for hi in ALL_HEADS:
        hw = head_weights[hi]
        if hw['is_routing']:
            print(f"  Head {hi:2d}: ROUTING  all_neg={hw['all_neg']}")
        else:
            print(f"  Head {hi:2d}: FIXED (pos 0)")
    
    # Run all prompts
    print("\n" + "=" * 100)
    print("  PHASE 5: FULL 28-HEAD GEOMETRIC RESONATOR")
    print("=" * 100)
    
    all_results = []
    category_stats = {}
    total_match = 0
    total_prompts = 0
    
    for cat, prompts in PROMPTS.items():
        print(f"\n{'─' * 100}")
        print(f"  Category: {cat} ({len(prompts)} prompts)")
        print(f"{'─' * 100}")
        
        cat_match = 0
        for prompt in prompts:
            r = run_full_resonator(engine, tokenizer, head_weights, prompt)
            all_results.append({**r, 'category': cat})
            
            m = "✓" if r['match'] else "✗"
            print(f"  {prompt:55s} base={r['baseline_top1']:>12s}  "
                  f"geo={r['geo_top1']:>12s} {m}  "
                  f"r={r['logit_corr']:.4f}  margin={r['geo_margin']:.4f}",
                  flush=True)
            
            if r['match']:
                cat_match += 1
                total_match += 1
            total_prompts += 1
        
        category_stats[cat] = {'n': len(prompts), 'match': cat_match}
    
    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"\n  Total: {total_match}/{total_prompts} = {100*total_match/total_prompts:.1f}%")
    print(f"\n  Per-category:")
    print(f"  {'Category':<30s} {'Match':>10s}")
    print(f"  {'─'*25} {'─'*10}")
    for cat, st in category_stats.items():
        pct = 100 * st['match'] / st['n']
        print(f"  {cat:<30s} {st['match']}/{st['n']} ({pct:5.1f}%)")
    
    # Failures
    failures = [r for r in all_results if not r['match']]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    {f['prompt']:50s} base={f['baseline_top1']:>12s}  "
                  f"geo={f['geo_top1']:>12s}  r={f['logit_corr']:.4f}")
    else:
        print(f"\n  *** NO FAILURES — 100% MATCH ***")
    
    # Avg correlation
    avg_corr = np.mean([r['logit_corr'] for r in all_results])
    print(f"\n  Avg logit correlation: {avg_corr:.4f}")
    
    # Save results
    import os
    os.makedirs('experiments/model_reverse_engineering_v2/results', exist_ok=True)
    with open('experiments/model_reverse_engineering_v2/results/phase5_full_resonator.json', 'w') as fp:
        json.dump({
            'total_match': total_match,
            'total_prompts': total_prompts,
            'accuracy': total_match / total_prompts,
            'category_stats': category_stats,
            'avg_logit_corr': avg_corr,
            'results': [{k: v for k, v in r.items() if k != 'tokens'} for r in all_results],
        }, fp, indent=2)
    print(f"\n  Results saved to results/phase5_full_resonator.json")
    
    print("\n" + "=" * 100)
    print("  DONE")
    print("=" * 100, flush=True)


if __name__ == '__main__':
    main()
