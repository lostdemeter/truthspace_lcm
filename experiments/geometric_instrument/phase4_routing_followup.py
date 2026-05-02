"""
Phase 4, Step 20b: Targeted Follow-up Tests
=============================================

Step 20 showed: all-layer geo routing = 0/6, single-layer = 22/28 OK.
6 failing layers: L0, L4, L6, L11, L16, L27.
Progressive from L0 breaks immediately (Paris rank 0→7).

This script investigates:
  1. Reverse progressive (geo from end backward) — how many layers 
     from the top can be replaced?
  2. Skip failing layers — replace 22/28, keep 6 real.
  3. L0 attention pattern analysis — WHY does L0 fail? Is attention
     distributed rather than hard-selective?
  4. Attention entropy per layer — measure how "hard" vs "soft" the
     real attention is at each layer.
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

FAILING_LAYERS = {0, 4, 6, 11, 16, 27}


def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids:
        return None, None
    tid = tids[0]
    return int(np.sum(logits > logits[tid])), float(logits[tid])


def extract_all_selectors(engine):
    """Extract geometric selectors for all 28 layers × 28 heads."""
    n_layers = len(engine.layers)
    all_layer_selectors = []
    for li in range(n_layers):
        attn = engine.layers[li].attention
        nh = attn.num_heads
        nkv = attn.num_kv_heads
        hpk = nh // nkv
        kv_selectors = []
        for kv in range(nkv):
            sel = Selector.from_model(engine, li, kv * hpk)
            kv_selectors.append(sel)
        layer_selectors = []
        for hi in range(nh):
            kv = hi // hpk
            layer_selectors.append(kv_selectors[kv])
        all_layer_selectors.append(layer_selectors)
    return all_layer_selectors


def geometric_routing_layer(engine, h, layer_idx, selectors):
    """Run one layer with geometric routing (last-token only)."""
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
    
    normed_2d = normed[0]
    sel_positions = []
    for hi in range(nh):
        sel_pos = selectors[hi].select(normed_2d)
        sel_positions.append(sel_pos)
        weights[0, hi, -1, :] = 0.0
        weights[0, hi, -1, sel_pos] = 1.0
    
    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    ao = phi_linear(attn.W_o, ao)
    h_post_attn = h + ao
    
    mlp = layer.mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    h_post_mlp = h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)
    
    return h_post_mlp, sel_positions


def get_attention_weights(engine, h, layer_idx):
    """Get the real softmax attention weights for a layer (last-token row only)."""
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
    
    # Return last-token attention weights: [nh, seq_len]
    return weights[0, :, -1, :]


def run_forward_with_geo_set(engine, tids, all_selectors, geo_layers):
    """Forward pass replacing the specified set of layers with geometric routing."""
    h = engine.embedding(tids)[np.newaxis, :, :]
    for li in range(len(engine.layers)):
        if li in geo_layers:
            h, _ = geometric_routing_layer(engine, h, li, all_selectors[li])
        else:
            h = engine.layers[li](h)
    normed = rms_norm(h, engine.final_norm_weight)
    logits = engine.lm_head(normed)[0, -1, :]
    return logits


def main():
    print("=" * 80)
    print("  PHASE 4, STEP 20b: Targeted Follow-up Tests")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    
    print(f"  Extracting selectors...", end="", flush=True)
    t0 = time.time()
    all_selectors = extract_all_selectors(engine)
    print(f" done in {time.time()-t0:.1f}s")
    
    france_tids = tokenizer.encode(FACTS['France']['prompt'])
    
    # ═══════════════════════════════════════════════════════════
    # Test 1: Attention entropy analysis — how hard vs soft is each layer?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Test 1: Attention entropy per layer (France, last token)")
    print("  Low entropy = hard selection, High entropy = distributed")
    print("─" * 70)
    
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    seq_len = h.shape[1]
    max_entropy = np.log(seq_len)  # uniform distribution entropy
    
    layer_entropies = []
    layer_max_weights = []
    layer_argmax_agree = []  # does argmax match geometric selector?
    
    for li in range(n_layers):
        weights_lt = get_attention_weights(engine, h, li)  # [nh, seq_len]
        
        # Entropy per head
        eps = 1e-12
        ent = -np.sum(weights_lt * np.log(weights_lt + eps), axis=-1)  # [nh]
        mean_ent = float(np.mean(ent))
        
        # Max weight per head (how much does the top position dominate?)
        max_w = np.max(weights_lt, axis=-1)  # [nh]
        mean_max_w = float(np.mean(max_w))
        
        # Does softmax argmax match geometric selector argmax?
        normed = rms_norm(h, engine.layers[li].attention.norm_weight)
        normed_2d = normed[0]
        n_agree = 0
        for hi in range(28):
            softmax_pick = int(np.argmax(weights_lt[hi]))
            geo_pick = all_selectors[li][hi].select(normed_2d)
            if softmax_pick == geo_pick:
                n_agree += 1
        
        layer_entropies.append(mean_ent)
        layer_max_weights.append(mean_max_w)
        layer_argmax_agree.append(n_agree)
        
        fail_mark = " ← FAIL" if li in FAILING_LAYERS else ""
        print(f"    L{li:>2}: entropy={mean_ent:.3f}/{max_entropy:.3f} "
              f"({mean_ent/max_entropy*100:.0f}%), "
              f"max_w={mean_max_w:.3f}, "
              f"argmax agree={n_agree}/28{fail_mark}")
        
        # Run real layer for next iteration
        h = engine.layers[li](h)
    
    # Stats
    fail_ent = [layer_entropies[li] for li in FAILING_LAYERS]
    ok_ent = [layer_entropies[li] for li in range(n_layers) if li not in FAILING_LAYERS]
    fail_maxw = [layer_max_weights[li] for li in FAILING_LAYERS]
    ok_maxw = [layer_max_weights[li] for li in range(n_layers) if li not in FAILING_LAYERS]
    fail_agree = [layer_argmax_agree[li] for li in FAILING_LAYERS]
    ok_agree = [layer_argmax_agree[li] for li in range(n_layers) if li not in FAILING_LAYERS]
    
    print(f"\n  Failing layers (L{sorted(FAILING_LAYERS)}):")
    print(f"    Mean entropy: {np.mean(fail_ent):.3f}, mean max_w: {np.mean(fail_maxw):.3f}, "
          f"mean argmax agree: {np.mean(fail_agree):.1f}/28")
    print(f"  OK layers:")
    print(f"    Mean entropy: {np.mean(ok_ent):.3f}, mean max_w: {np.mean(ok_maxw):.3f}, "
          f"mean argmax agree: {np.mean(ok_agree):.1f}/28")
    
    # ═══════════════════════════════════════════════════════════
    # Test 2: Reverse progressive (geo from end backward)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Test 2: Reverse progressive — geo routing for layers N..27")
    print("  (France only)")
    print("─" * 70)
    
    for start_geo in range(n_layers, -1, -1):
        geo_set = set(range(start_geo, n_layers))
        logits = run_forward_with_geo_set(engine, france_tids, all_selectors, geo_set)
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, ' Paris', tokenizer)
        
        mark = "✓" if rank == 0 else "✗"
        if start_geo == n_layers:
            status = "all real"
        elif start_geo == 0:
            status = "ALL GEO"
        else:
            status = f"geo L{start_geo}-L27"
        print(f"    {status:>15}: top='{top_tok}', Paris rank={rank} {mark}")
    
    # ═══════════════════════════════════════════════════════════
    # Test 3: Skip failing layers — replace 22, keep 6 real
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print(f"  Test 3: Skip failing layers {sorted(FAILING_LAYERS)}")
    print(f"  Replace {n_layers - len(FAILING_LAYERS)} layers with geo routing, keep 6 real")
    print("─" * 70)
    
    safe_geo_set = set(range(n_layers)) - FAILING_LAYERS
    
    skip_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        logits = run_forward_with_geo_set(engine, tids, all_selectors, safe_geo_set)
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        
        mark = "✓" if rank == 0 else "✗"
        if rank == 0:
            skip_match += 1
        print(f"    {country:>8}: top='{top_tok}', rank={rank} {mark}")
    
    print(f"\n  Skip-failing geo routing: {skip_match}/6")
    
    # ═══════════════════════════════════════════════════════════
    # Test 4: Extraction region only (L23-L27) — known to work from Phase 3
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Test 4: Extraction+amplification region only (L23-L27)")
    print("─" * 70)
    
    extraction_geo_set = {23, 24, 25, 26, 27}
    
    ext_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        logits = run_forward_with_geo_set(engine, tids, all_selectors, extraction_geo_set)
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        
        mark = "✓" if rank == 0 else "✗"
        if rank == 0:
            ext_match += 1
        print(f"    {country:>8}: top='{top_tok}', rank={rank} {mark}")
    
    print(f"\n  Extraction region geo routing: {ext_match}/6")
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n  All-layer geo routing (Step 20):     0/6")
    print(f"  Skip 6 failing layers:               {skip_match}/6")
    print(f"  Extraction region only (L23-L27):    {ext_match}/6")
    
    # Key metric: what fraction of layers have hard attention?
    hard_layers = sum(1 for mw in layer_max_weights if mw > 0.8)
    soft_layers = sum(1 for mw in layer_max_weights if mw < 0.5)
    print(f"\n  Hard attention (max_w > 0.8): {hard_layers}/{n_layers} layers")
    print(f"  Soft attention (max_w < 0.5): {soft_layers}/{n_layers} layers")
    print(f"  Argmax agreement (geo = softmax): "
          f"{sum(a == 28 for a in layer_argmax_agree)}/{n_layers} layers with 28/28")
    
    print()


if __name__ == '__main__':
    main()
