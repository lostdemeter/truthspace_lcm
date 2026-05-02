"""
Phase 4, Step 20: All-Layer Geometric Routing Test
====================================================

Replace softmax attention routing with geometric selectors (argmax(h·d_k))
at EVERY layer simultaneously. Run a full forward pass and compare top-1
predictions with the real model.

Step 19 showed 112/112 KV groups are rank-1 (>100K) with pure polarity.
This test determines if the model still works without softmax.

Approach:
  - Pre-extract selectors for all 28 layers × 4 KV groups
  - For each prompt, run modified forward pass:
    * At each layer, replace last-token softmax routing with geometric selectors
    * Keep V, W_o, MLP, and non-last-token routing as real
  - Compare top-1 predictions

Depends on: F129 (Phase 3 extraction layer), Step 19 (all layers geometric)
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_integer import phi_to_float

from experiments.geometric_instrument.components.selector import Selector

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}


def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids:
        return None, None
    tid = tids[0]
    return int(np.sum(logits > logits[tid])), float(logits[tid])


def extract_all_selectors(engine):
    """Extract geometric selectors for all 28 layers × 28 heads.
    
    Returns: list of 28 lists, each containing 28 Selector objects.
    Due to GQA sharing, heads in the same KV group share d_k.
    """
    n_layers = len(engine.layers)
    all_layer_selectors = []
    
    for li in range(n_layers):
        attn = engine.layers[li].attention
        nh = attn.num_heads
        nkv = attn.num_kv_heads
        hpk = nh // nkv
        
        # Extract one selector per KV group, share across heads
        kv_selectors = []
        for kv in range(nkv):
            sel = Selector.from_model(engine, li, kv * hpk)
            kv_selectors.append(sel)
        
        # Map each head to its KV group's selector
        layer_selectors = []
        for hi in range(nh):
            kv = hi // hpk
            layer_selectors.append(kv_selectors[kv])
        
        all_layer_selectors.append(layer_selectors)
    
    return all_layer_selectors


def geometric_routing_layer(engine, h, layer_idx, selectors):
    """Run one layer with geometric routing (last-token only).
    
    Replaces softmax attention weights for the last token with
    hard selection via geometric selectors. All other tokens and
    the MLP use real computation.
    
    Args:
        engine: PhiQwen2Engine
        h: [1, seq_len, d_model] hidden states
        layer_idx: which layer
        selectors: list of 28 Selector objects for this layer
    
    Returns:
        h_out: [1, seq_len, d_model], sel_positions: list of 28 ints
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]
    
    normed = rms_norm(h, attn.norm_weight)
    
    # Full QKV computation
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
    
    # Replace last-token routing with geometric selectors
    normed_2d = normed[0]
    sel_positions = []
    for hi in range(nh):
        sel_pos = selectors[hi].select(normed_2d)
        sel_positions.append(sel_pos)
        weights[0, hi, -1, :] = 0.0
        weights[0, hi, -1, sel_pos] = 1.0
    
    # Attention output
    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    ao = phi_linear(attn.W_o, ao)
    h_post_attn = h + ao
    
    # Real MLP
    mlp = layer.mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    h_post_mlp = h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)
    
    return h_post_mlp, sel_positions


def main():
    print("=" * 80)
    print("  PHASE 4, STEP 20: All-Layer Geometric Routing Test")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    
    # ── Pre-extract all selectors ──
    print(f"  Extracting selectors for {n_layers} layers × 4 KV groups...",
          end="", flush=True)
    t0 = time.time()
    all_selectors = extract_all_selectors(engine)
    print(f" done in {time.time()-t0:.1f}s")
    
    # ── Test 1: Real model (ground truth) ──
    print("\n" + "─" * 70)
    print("  Ground truth: Real model forward pass")
    print("─" * 70)
    
    ground_truth = {}
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = engine.layers[li](h)
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        ground_truth[country] = {'top': top_tok, 'rank': rank}
        mark = "✓" if rank == 0 else "✗"
        print(f"    {country:>8}: top='{top_tok}', answer rank={rank} {mark}")
    
    # ── Test 2: Geometric routing at ALL layers ──
    print("\n" + "─" * 70)
    print("  ALL-LAYER geometric routing (no softmax for last-token at any layer)")
    print("─" * 70)
    
    geo_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        
        layer_sels = {}
        for li in range(n_layers):
            h, sel_positions = geometric_routing_layer(
                engine, h, li, all_selectors[li])
            layer_sels[li] = sel_positions
        
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        
        match = (top_tok == ground_truth[country]['top'])
        if match:
            geo_match += 1
        mark = "✓" if match else "✗"
        
        # Show extraction layer (L23) selection
        l23_sels = layer_sels[23]
        print(f"    {country:>8}: top='{top_tok}', rank={rank} {mark}  "
              f"L23 sels={sorted(set(l23_sels))}")
    
    print(f"\n  All-layer geometric routing: {geo_match}/6 match ground truth")
    
    # ── Test 3: Layer-by-layer ablation ──
    # Replace one layer at a time with geometric routing to find which layers matter
    print("\n" + "─" * 70)
    print("  Layer-by-layer ablation: geometric routing at ONE layer, rest real")
    print("─" * 70)
    
    # Use France as the canonical test case
    france_tids = tokenizer.encode(FACTS['France']['prompt'])
    
    layer_results = []
    for target_li in range(n_layers):
        h = engine.embedding(france_tids)[np.newaxis, :, :]
        
        for li in range(n_layers):
            if li == target_li:
                h, _ = geometric_routing_layer(
                    engine, h, li, all_selectors[li])
            else:
                h = engine.layers[li](h)
        
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, score = get_rank(logits, ' Paris', tokenizer)
        
        mark = "✓" if rank == 0 else "✗"
        layer_results.append({'layer': target_li, 'top': top_tok, 'rank': rank})
        
        if rank != 0:
            print(f"    L{target_li:>2}: top='{top_tok}', Paris rank={rank} {mark}")
    
    n_ok = sum(1 for r in layer_results if r['rank'] == 0)
    n_fail = sum(1 for r in layer_results if r['rank'] != 0)
    print(f"\n  Single-layer ablation (France): {n_ok}/{n_layers} layers OK, "
          f"{n_fail} cause failure")
    if n_fail == 0:
        print("  → Every layer works individually! Failures must be from interactions.")
    
    # ── Test 4: Progressive — geometric routing for layers 0..N ──
    print("\n" + "─" * 70)
    print("  Progressive: geometric routing for layers 0..N, rest real")
    print("  (France only — finding the breaking point)")
    print("─" * 70)
    
    for n_geo in range(n_layers + 1):
        h = engine.embedding(france_tids)[np.newaxis, :, :]
        
        for li in range(n_layers):
            if li < n_geo:
                h, _ = geometric_routing_layer(
                    engine, h, li, all_selectors[li])
            else:
                h = engine.layers[li](h)
        
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, ' Paris', tokenizer)
        
        mark = "✓" if rank == 0 else "✗"
        status = f"geo L0-L{n_geo-1}" if n_geo > 0 else "all real"
        if n_geo == n_layers:
            status = "ALL GEO"
        print(f"    {status:>15}: top='{top_tok}', Paris rank={rank} {mark}")
    
    # ── Test 5: All prompts with all-layer geometric routing ──
    print("\n" + "─" * 70)
    print("  Full results: all 6 prompts × all-layer geometric routing")
    print("─" * 70)
    
    full_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        
        for li in range(n_layers):
            h, _ = geometric_routing_layer(
                engine, h, li, all_selectors[li])
        
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        
        gt_top = ground_truth[country]['top']
        match = (top_tok == gt_top)
        if match:
            full_match += 1
        mark = "✓" if match else "✗"
        
        print(f"    {country:>8}: top='{top_tok}' (gt='{gt_top}'), "
              f"answer rank={rank} {mark}")
    
    # ── Summary ──
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n  All-layer geometric routing: {full_match}/6 match ground truth")
    print(f"  Single-layer ablation (France): {n_ok}/{n_layers} layers safe individually")
    
    if full_match == 6:
        print("\n  ★ SOFTMAX IS FULLY REPLACEABLE ACROSS THE ENTIRE MODEL ★")
    elif full_match >= 5:
        print(f"\n  Near-complete: {full_match}/6. Minor edge cases remain.")
    else:
        print(f"\n  Partial: {full_match}/6. Layer interactions cause degradation.")
    
    print()


if __name__ == '__main__':
    main()
