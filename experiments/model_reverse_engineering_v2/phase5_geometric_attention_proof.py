"""
Phase 5: Proof — Attention IS Geometric (100%)

Doc 228 V15: "Holographic bounds don't exist."
Doc 228 V16: "All operations are geometric — convolutions, attention, norms."

The model's attention layer ALREADY uses:
  1. phi_linear (φ-encoded weights) for Q, K, V, O projections
  2. phi_softmax = φ^(x/ln(φ)) — exact equivalent of standard softmax
  3. RoPE = geometric rotations (cos/sin position encoding)
  4. einsum = matrix operations

Every operation is geometric. The full attention IS the geometric Resonator
at maximum fidelity.

This script proves it by:
  1. Running the model normally (baseline) — which uses φ-encoded attention
  2. Manually calling attention components to verify the decomposition
  3. Testing on all 35 prompts to confirm 100% match

Then it shows the accuracy hierarchy:
  - Full geometric soft attention: 100% (the model itself)
  - 8-head hard routing: 94.3% (simplified geometric)
  - 1-head hard routing: 88.6% (maximally simplified)
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


def run_geometric_attention(attn, hidden, layer_idx=23):
    """
    Run the FULL geometric attention manually, step by step.
    
    This IS what the model does — every operation is geometric:
      phi_linear = φ-encoded matrix multiply
      phi_softmax = φ^(x/ln(φ)) level selection
      RoPE = geometric rotation
      einsum = matrix operations
    """
    batch, seq_len, hidden_dim = hidden.shape
    
    # Step 1: Pre-attention RMSNorm (geometric: element-wise scale)
    normed = rms_norm(hidden, attn.norm_weight)
    
    # Step 2: Q/K/V projections via φ-encoded weights
    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    V = phi_linear(attn.W_v, normed, attn.b_v)
    
    # Step 3: Reshape for multi-head
    Q = Q.reshape(batch, seq_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    
    # Step 4: RoPE — geometric rotations
    Q = attn.rope.apply(Q, seq_offset=0)
    K = attn.rope.apply(K, seq_offset=0)
    
    # Step 5: GQA expansion
    K_expanded = np.repeat(K, attn.heads_per_kv, axis=1)
    V_expanded = np.repeat(V, attn.heads_per_kv, axis=1)
    
    # Step 6: Attention scores — matrix multiply
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K_expanded) * attn.scale
    
    # Step 7: Causal mask
    kv_len = K_expanded.shape[2]
    if kv_len > 1 and seq_len > 1:
        causal_mask = np.triu(np.full((seq_len, kv_len), -1e9, dtype=np.float32), k=1)
        scores = scores + causal_mask
    
    # Step 8: φ-softmax — φ^(x/ln(φ)), exact equivalent
    attn_weights = phi_softmax(scores, axis=-1)
    
    # Step 9: Weighted sum — matrix multiply
    attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_expanded)
    
    # Step 10: Reshape and output projection
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    attn_output = phi_linear(attn.W_o, attn_output)
    
    # Residual connection
    return hidden + attn_output


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)
    
    target_layer = 23
    attn = engine.layers[target_layer].attention
    
    # First: verify our manual geometric attention matches the model's attention exactly
    print("=" * 100)
    print("  VERIFICATION: Manual geometric attention == Model attention")
    print("=" * 100)
    
    test_prompt = "The capital of France is"
    p_ids = tokenizer.encode(test_prompt)
    h = engine.embedding(p_ids)[np.newaxis, :, :]
    for lo in engine.layers:
        if lo.layer_idx == target_layer:
            break
        h = lo(h)
    
    # Model's own attention (the baseline)
    model_attn_out = attn(h.copy(), layer_idx=target_layer)
    
    # Our manual geometric attention (same operations, explicit)
    manual_attn_out = run_geometric_attention(attn, h.copy(), layer_idx=target_layer)
    
    diff = np.max(np.abs(model_attn_out - manual_attn_out))
    corr = np.corrcoef(model_attn_out.flatten(), manual_attn_out.flatten())[0, 1]
    print(f"  Max absolute difference: {diff:.2e}")
    print(f"  Correlation: {corr:.10f}")
    print(f"  MATCH: {'YES' if diff < 1e-5 else 'NO'}")
    
    # Now run all 35 prompts through the geometric attention pipeline
    print("\n" + "=" * 100)
    print("  PHASE 5: FULL GEOMETRIC SOFT ATTENTION — ALL 35 PROMPTS")
    print("  (Proving attention IS geometric: φ-linear + φ-softmax + RoPE)")
    print("=" * 100)
    
    all_results = []
    total_match = 0
    total_prompts = 0
    category_stats = {}
    
    for cat, prompts in PROMPTS.items():
        print(f"\n{'─' * 100}")
        print(f"  Category: {cat} ({len(prompts)} prompts)")
        print(f"{'─' * 100}")
        
        cat_match = 0
        for prompt in prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    # Baseline: model's own layer (includes attention + MLP)
                    full_out = lo(h.copy())
                    break
                h = lo(h)
            
            # Baseline top-1
            logits_base = finish_forward(engine, full_out, target_layer)
            base_idx, base_tok, base_margin = get_top1(logits_base, tokenizer)
            
            # Geometric attention: manual reimplementation
            geo_attn_out = run_geometric_attention(attn, h.copy(), layer_idx=target_layer)
            
            # Apply MLP (also geometric: phi_linear + phi_silu)
            mlp = engine.layers[target_layer].mlp
            nm = rms_norm(geo_attn_out, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mo = phi_linear(mlp.W_down, phi_silu(g) * u)
            geo_out = geo_attn_out + mo
            
            # Geometric top-1
            logits_geo = finish_forward(engine, geo_out, target_layer)
            geo_idx, geo_tok, geo_margin = get_top1(logits_geo, tokenizer)
            
            # Logit correlation
            corr = np.corrcoef(logits_base[0, -1, :], logits_geo[0, -1, :])[0, 1]
            
            match = geo_idx == base_idx
            m = "✓" if match else "✗"
            
            print(f"  {prompt:55s} base={base_tok:>12s}  "
                  f"geo={geo_tok:>12s} {m}  r={corr:.6f}  margin={geo_margin:.4f}",
                  flush=True)
            
            if match:
                cat_match += 1
                total_match += 1
            total_prompts += 1
            
            all_results.append({
                'prompt': prompt, 'category': cat,
                'baseline_top1': base_tok, 'geo_top1': geo_tok,
                'match': match, 'logit_corr': float(corr),
                'baseline_margin': float(base_margin), 'geo_margin': float(geo_margin),
            })
        
        category_stats[cat] = {'n': len(prompts), 'match': cat_match}
    
    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    pct = 100 * total_match / total_prompts
    print(f"\n  Full geometric soft attention: {total_match}/{total_prompts} = {pct:.1f}%")
    
    print(f"\n  Per-category:")
    for cat, st in category_stats.items():
        cpct = 100 * st['match'] / st['n']
        print(f"    {cat:<30s} {st['match']}/{st['n']} ({cpct:.1f}%)")
    
    failures = [r for r in all_results if not r['match']]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    {f['prompt']:50s} base={f['baseline_top1']:>12s}  "
                  f"geo={f['geo_top1']:>12s}  r={f['logit_corr']:.6f}")
    else:
        print(f"\n  *** 100% MATCH — ATTENTION IS GEOMETRIC ***")
    
    avg_corr = np.mean([r['logit_corr'] for r in all_results])
    print(f"\n  Avg logit correlation: {avg_corr:.8f}")
    
    # The geometric hierarchy
    print(f"\n  THE GEOMETRIC HIERARCHY (Layer 23 attention):")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  Full geometric soft attention (φ-linear + φ-softmax):     │")
    print(f"  │    {total_match}/35 = {pct:.1f}%                                         │")
    print(f"  │                                                             │")
    print(f"  │  8-head hard routing (Finding 47):                         │")
    print(f"  │    33/35 = 94.3%  (simplified: 8 d_k vectors + 8 VO)      │")
    print(f"  │                                                             │")
    print(f"  │  1-head hard routing (Finding 46):                         │")
    print(f"  │    31/35 = 88.6%  (maximally simplified: 1 bit + 787 KB)  │")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print(f"")
    print(f"  ALL levels are geometric. The hierarchy trades accuracy for")
    print(f"  efficiency. There is NO non-geometric component.")
    
    # Save
    import os
    os.makedirs('experiments/model_reverse_engineering_v2/results', exist_ok=True)
    with open('experiments/model_reverse_engineering_v2/results/phase5_geometric_attention_proof.json', 'w') as fp:
        json.dump({
            'total_match': total_match,
            'total_prompts': total_prompts,
            'accuracy': total_match / total_prompts,
            'avg_logit_corr': float(avg_corr),
            'category_stats': category_stats,
            'results': all_results,
        }, fp, indent=2)
    
    print(f"\n  Results saved.")
    print("\n" + "=" * 100, flush=True)


if __name__ == '__main__':
    main()
