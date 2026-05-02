"""
Phase 4: Fixed vs Routing Heads at Layer 23

Key finding from phase4_attn_pattern_lut.py:
  - 28 per-head entropies → error R² = 0.7976 (80%!)
  - Most heads ALWAYS attend to position 0 (BOS/first token)
  - Only ~9 "routing" heads vary their attention target per prompt
  - Hybrid (real attn + real MLP) = 6/6

This script answers:
  1. Which heads are "fixed" (always attend to pos 0) vs "routing" (vary)?
  2. For France prompt, what do the routing heads attend to?
  3. If we run ONLY the routing heads with real matmuls and use
     spectrometer rules for everything else, does France work?
  4. Can we approximate the routing decision with simpler φ-level operations?

The hypothesis: most of the attention computation is REDUNDANT at layer 23.
The spectrometer error comes from a handful of routing heads that need to
know which token to attend to — this is the φ-softmax routing decision.
"""

import sys
import numpy as np
import time
import gc

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_spectrometer import SpectrometerLayer, load_all_rules

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
RULES_DIR = 'experiments/model_reverse_engineering_v2/results/phase4_rules_full'

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def finish_forward(engine, hidden_start, start_layer):
    """Run remaining layers + final norm + LM head."""
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    sorted_l = np.sort(logits[0, -1, :])[::-1]
    margin = sorted_l[0] - sorted_l[1]
    return idx, tok, margin


def extract_attention_details(engine, layer_idx, hidden):
    """
    Run attention and return per-head decomposition.
    Returns attention weights AND per-head output contributions.
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    
    batch, seq_len, hidden_dim = hidden.shape
    
    normed = rms_norm(hidden, attn.norm_weight)
    
    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    V = phi_linear(attn.W_v, normed, attn.b_v)
    
    Q = Q.reshape(batch, seq_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    
    Q = attn.rope.apply(Q)
    K = attn.rope.apply(K)
    
    K_expanded = np.repeat(K, attn.heads_per_kv, axis=1)
    V_expanded = np.repeat(V, attn.heads_per_kv, axis=1)
    
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K_expanded) * attn.scale
    
    kv_len = K_expanded.shape[2]
    if kv_len > 1 and seq_len > 1:
        causal_mask = np.triu(np.full((seq_len, kv_len), -1e9, dtype=np.float32), k=1)
        scores = scores + causal_mask
    
    attn_weights = phi_softmax(scores, axis=-1)
    
    # Per-head weighted V (before output projection)
    per_head_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_expanded)
    # Shape: (batch, num_heads, seq_len, head_dim)
    
    # Combined output (all heads concatenated, projected)
    combined = per_head_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    attn_proj = phi_linear(attn.W_o, combined)
    
    return attn_weights, per_head_output, attn_proj, scores


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
    
    target_layer = 23
    rules = all_rules[target_layer]
    spec_layer = SpectrometerLayer(
        rules=rules, full_layer=engine.layers[target_layer],
        r2_threshold=0.7, mode='rules_only',
    )
    
    cal_prompts = [
        '1 + 1 =', '2 + 2 =', 'The sky is', 'Water is made of',
        'The sun rises in the', 'Gravity makes things fall',
        'Once upon a time', 'She walked into the room and',
        'He said that he would', 'They decided to go to the',
        'The old man sat on the', 'After the rain stopped',
        'The quick brown fox', 'In machine learning',
        'Python is a programming', 'The function returns',
        'An algorithm that sorts', 'The largest planet is',
        'Albert Einstein developed the', 'Shakespeare wrote many',
        'The speed of light is', 'DNA stands for',
        'The Pacific Ocean is', 'I think that we should',
        'She said that she would', 'It is important to note that',
        'The reason for this is', 'According to the latest',
        'In 2024, the world', 'If you want to learn',
        'The best way to', 'One of the most important',
        'As a result of the', 'Between the two options',
        'Despite the challenges', 'For example, consider',
        'However, it is worth', 'In conclusion, the',
    ]
    
    test_prompts = [
        'The capital of France is',
        'The largest ocean is the',
        'The color of grass is',
        'Barack Obama was the',
        'To be or not to',
        'Roses are red, violets are',
    ]
    
    # =========================================================================
    #   Part A: Classify heads as fixed vs routing
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: Fixed vs routing heads")
    print("=" * 80)
    
    head_argmaxes = {hi: [] for hi in range(28)}
    head_entropies = {hi: [] for hi in range(28)}
    
    for pi, p in enumerate(cal_prompts):
        p_ids = tokenizer.encode(p)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for layer_obj in engine.layers:
            if layer_obj.layer_idx == target_layer:
                break
            h = layer_obj(h)
        
        aw, pho, ap, sc = extract_attention_details(engine, target_layer, h)
        
        for hi in range(28):
            w = aw[0, hi, -1, :]
            am = int(np.argmax(w))
            ent = float(-np.sum(w * np.log(w + 1e-20)))
            head_argmaxes[hi].append(am)
            head_entropies[hi].append(ent)
        
        if pi % 10 == 0:
            print(f"    Collecting: {pi}/{len(cal_prompts)}...", flush=True)
        gc.collect()
    
    print(f"\n  Head classification (38 calibration prompts):")
    print(f"  {'Head':>6s}  {'Unique':>6s}  {'Always0':>7s}  {'Mean H':>6s}  {'Std H':>6s}  {'Type':>10s}")
    
    fixed_heads = []
    routing_heads = []
    
    for hi in range(28):
        ams = head_argmaxes[hi]
        unique = len(set(ams))
        always_0 = sum(1 for a in ams if a == 0) / len(ams)
        mean_h = np.mean(head_entropies[hi])
        std_h = np.std(head_entropies[hi])
        
        # Classification: fixed if always attends to pos 0 (>90% of time)
        # and has low entropy (mean < 0.5)
        if always_0 > 0.90 and mean_h < 0.5:
            htype = "FIXED"
            fixed_heads.append(hi)
        elif unique <= 2 and always_0 > 0.80:
            htype = "MOSTLY_FIX"
            fixed_heads.append(hi)
        else:
            htype = "ROUTING"
            routing_heads.append(hi)
        
        print(f"  {hi:6d}  {unique:6d}  {always_0:7.2f}  {mean_h:6.3f}  {std_h:6.3f}  {htype:>10s}")
    
    print(f"\n  Fixed heads ({len(fixed_heads)}): {fixed_heads}")
    print(f"  Routing heads ({len(routing_heads)}): {routing_heads}")
    
    # =========================================================================
    #   Part B: What do routing heads attend to for France?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Routing heads on the France prompt")
    print("=" * 80)
    
    prompt = "The capital of France is"
    p_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode_token(i) for i in p_ids]
    
    h_france = engine.embedding(p_ids)[np.newaxis, :, :]
    for layer_obj in engine.layers:
        if layer_obj.layer_idx == target_layer:
            full_out_france = layer_obj(h_france.copy())
            spec_out_france = spec_layer(h_france.copy())
            break
        h_france = layer_obj(h_france)
    
    aw, per_head_out, attn_proj, scores = extract_attention_details(engine, target_layer, h_france)
    
    print(f"\n  Tokens: {tokens}")
    print(f"\n  Routing heads attention pattern (last token):")
    for hi in routing_heads:
        w = aw[0, hi, -1, :]
        print(f"    Head {hi:2d}: ", end="")
        for pos in range(len(tokens)):
            marker = " ←" if pos == np.argmax(w) else ""
            print(f"  {tokens[pos]:>10s}={w[pos]:.3f}{marker}", end="")
        print()
    
    print(f"\n  Fixed heads attention pattern (last token):")
    for hi in fixed_heads[:5]:  # show first 5
        w = aw[0, hi, -1, :]
        print(f"    Head {hi:2d}: ", end="")
        for pos in range(len(tokens)):
            marker = " ←" if pos == np.argmax(w) else ""
            print(f"  {tokens[pos]:>10s}={w[pos]:.3f}{marker}", end="")
        print()
    print(f"    ... ({len(fixed_heads)} fixed heads total, all similar)")
    
    # =========================================================================
    #   Part C: Per-head contribution to the error
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Which heads contribute most to the spectrometer error?")
    print("=" * 80)
    
    error = (full_out_france - spec_out_france)[0, -1, :]
    
    # The output projection W_o mixes all heads. To see per-head contribution,
    # we need to project each head through W_o separately.
    layer_obj = engine.layers[target_layer]
    attn = layer_obj.attention
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    
    print(f"\n  Per-head output norms and alignment with error:")
    print(f"  (Each head's contribution = W_o @ [0...head_i_output...0])")
    
    head_contributions = []
    for hi in range(num_heads):
        # Create a vector with only this head's output
        single_head = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
        single_head[0, 0, hi*head_dim:(hi+1)*head_dim] = per_head_out[0, hi, -1, :]
        
        # Project through W_o
        head_proj = phi_linear(attn.W_o, single_head)[0, 0, :]
        head_contributions.append(head_proj)
        
        cos_err = np.dot(head_proj, error) / (np.linalg.norm(head_proj) * np.linalg.norm(error) + 1e-20)
        htype = "ROUTING" if hi in routing_heads else "fixed"
        print(f"    Head {hi:2d} [{htype:>7s}]: ||proj||={np.linalg.norm(head_proj):8.2f}  "
              f"cos(proj,error)={cos_err:+.4f}")
    
    # Sum of routing heads vs fixed heads contribution
    routing_contrib = sum(head_contributions[hi] for hi in routing_heads)
    fixed_contrib = sum(head_contributions[hi] for hi in fixed_heads)
    
    cos_routing = np.dot(routing_contrib, error) / (np.linalg.norm(routing_contrib) * np.linalg.norm(error) + 1e-20)
    cos_fixed = np.dot(fixed_contrib, error) / (np.linalg.norm(fixed_contrib) * np.linalg.norm(error) + 1e-20)
    
    print(f"\n  Routing heads combined:")
    print(f"    ||routing_contrib||: {np.linalg.norm(routing_contrib):.2f}")
    print(f"    cos(routing, error): {cos_routing:.4f}")
    print(f"  Fixed heads combined:")
    print(f"    ||fixed_contrib||: {np.linalg.norm(fixed_contrib):.2f}")
    print(f"    cos(fixed, error): {cos_fixed:.4f}")
    
    # =========================================================================
    #   Part D: Ablation — what happens if we zero out specific heads?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Head ablation tests on ALL test prompts")
    print("=" * 80)
    print("  Testing: which combination of heads must be computed correctly?")
    
    # For each test, we run the full layer but zero out certain heads
    # to see which ones matter
    
    def run_with_head_mask(engine, layer_idx, hidden, head_mask, tokenizer, prompt):
        """Run attention with only certain heads active (others zeroed)."""
        layer = engine.layers[layer_idx]
        attn = layer.attention
        batch, seq_len, hidden_dim = hidden.shape
        
        normed = rms_norm(hidden, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        V = phi_linear(attn.W_v, normed, attn.b_v)
        
        Q = Q.reshape(batch, seq_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        
        K_expanded = np.repeat(K, attn.heads_per_kv, axis=1)
        V_expanded = np.repeat(V, attn.heads_per_kv, axis=1)
        
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K_expanded) * attn.scale
        
        if seq_len > 1:
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask
        
        attn_weights = phi_softmax(scores, axis=-1)
        per_head_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_expanded)
        
        # Apply head mask — zero out heads not in mask
        for hi in range(attn.num_heads):
            if hi not in head_mask:
                per_head_output[0, hi, :, :] = 0.0
        
        combined = per_head_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        attn_proj = phi_linear(attn.W_o, combined)
        
        post_attn = hidden + attn_proj
        
        # Real MLP
        mlp = layer.mlp
        normed_mlp = rms_norm(post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_hidden = phi_silu(gate) * up
        mlp_out = phi_linear(mlp.W_down, mlp_hidden)
        
        return post_attn + mlp_out
    
    all_heads = set(range(28))
    
    configs = [
        ("All 28 heads", all_heads),
        ("Routing heads only", set(routing_heads)),
        ("Fixed heads only", set(fixed_heads)),
        ("No heads (just residual+MLP)", set()),
    ]
    
    for config_name, head_set in configs:
        print(f"\n  Config: {config_name} ({len(head_set)} heads)")
        n_pass = 0
        for prompt in test_prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    full_out = lo(h.copy())
                    break
                h = lo(h)
            
            ablated_out = run_with_head_mask(engine, target_layer, h, head_set, tokenizer, prompt)
            
            logits_full = finish_forward(engine, full_out, target_layer)
            logits_abl = finish_forward(engine, ablated_out, target_layer)
            
            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            abl_id, abl_tok, abl_margin = get_top1(logits_abl, tokenizer)
            
            match = '✓' if abl_id == full_id else '✗'
            if abl_id == full_id: n_pass += 1
            
            extra = f" margin={abl_margin:.3f}" if 'France' in prompt else ""
            print(f"    {match} {prompt:>35s} → {abl_tok:>8s} (want {full_tok}){extra}")
        
        print(f"    Score: {n_pass}/{len(test_prompts)}")
    
    # =========================================================================
    #   Part E: Single-head ablation — which individual head fixes France?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Single routing head ablation for France")
    print("=" * 80)
    print("  Which individual routing head, when added, fixes France?")
    
    prompt = "The capital of France is"
    p_ids = tokenizer.encode(prompt)
    h = engine.embedding(p_ids)[np.newaxis, :, :]
    for lo in engine.layers:
        if lo.layer_idx == target_layer:
            full_out = lo(h.copy())
            break
        h = lo(h)
    
    logits_full = finish_forward(engine, full_out, target_layer)
    full_id, full_tok, _ = get_top1(logits_full, tokenizer)
    
    # Test: fixed heads + ONE routing head at a time
    for rh in routing_heads:
        head_set = set(fixed_heads) | {rh}
        ablated_out = run_with_head_mask(engine, target_layer, h, head_set, tokenizer, prompt)
        logits_abl = finish_forward(engine, ablated_out, target_layer)
        abl_id, abl_tok, abl_margin = get_top1(logits_abl, tokenizer)
        
        match = '✓' if abl_id == full_id else '✗'
        print(f"    Fixed + head {rh:2d}: {match} → {abl_tok:>8s} (want {full_tok}) margin={abl_margin:.3f}")
    
    # Test: all routing heads EXCEPT one
    print(f"\n  All heads EXCEPT one routing head:")
    for rh in routing_heads:
        head_set = all_heads - {rh}
        ablated_out = run_with_head_mask(engine, target_layer, h, head_set, tokenizer, prompt)
        logits_abl = finish_forward(engine, ablated_out, target_layer)
        abl_id, abl_tok, abl_margin = get_top1(logits_abl, tokenizer)
        
        match = '✓' if abl_id == full_id else '✗'
        print(f"    All except head {rh:2d}: {match} → {abl_tok:>8s} (want {full_tok}) margin={abl_margin:.3f}")
    
    # =========================================================================
    #   Part F: Minimum head set — greedy search
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part F: Minimum head set for 6/6 (greedy search)")
    print("=" * 80)
    
    def test_head_set(head_set):
        """Return (n_pass, france_margin)"""
        n_pass = 0
        france_margin = None
        for prompt in test_prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    full_out = lo(h.copy())
                    break
                h = lo(h)
            
            ablated_out = run_with_head_mask(engine, target_layer, h, head_set, tokenizer, prompt)
            logits_full = finish_forward(engine, full_out, target_layer)
            logits_abl = finish_forward(engine, ablated_out, target_layer)
            
            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            abl_id, abl_tok, abl_margin = get_top1(logits_abl, tokenizer)
            
            if abl_id == full_id:
                n_pass += 1
            if 'France' in prompt:
                france_margin = abl_margin
                france_pass = abl_id == full_id
        
        return n_pass, france_margin, france_pass
    
    # Start with no heads, greedily add the most impactful
    current_set = set()
    remaining = set(range(28))
    
    print(f"  Greedy head addition:")
    while remaining and len(current_set) < 28:
        best_head = None
        best_score = -1
        best_margin = -999
        
        for candidate in remaining:
            trial = current_set | {candidate}
            score, margin, fp = test_head_set(trial)
            if score > best_score or (score == best_score and (margin or 0) > best_margin):
                best_head = candidate
                best_score = score
                best_margin = margin or 0
                best_fp = fp
        
        current_set.add(best_head)
        remaining.remove(best_head)
        
        htype = "ROUTING" if best_head in routing_heads else "fixed"
        france_mark = '✓' if best_fp else '✗'
        print(f"    +head {best_head:2d} [{htype:>7s}]: {best_score}/6  "
              f"France={france_mark} margin={best_margin:.3f}  "
              f"({len(current_set)} heads total)", flush=True)
        
        if best_score == 6:
            print(f"\n  *** 6/6 achieved with {len(current_set)} heads! ***")
            print(f"  Head set: {sorted(current_set)}")
            n_routing = len(current_set & set(routing_heads))
            n_fixed = len(current_set & set(fixed_heads))
            print(f"  Routing: {n_routing}, Fixed: {n_fixed}")
            break
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
