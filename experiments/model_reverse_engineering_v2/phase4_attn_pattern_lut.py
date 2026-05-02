"""
Phase 4: Attention Pattern LUT for Layer 23

Previous approach FAILED: tried to LUT the hidden state features.
The error is content-dependent (attention routing), not state-dependent.

NEW insight: phi_softmax already expresses attention as φ-level selection:
  softmax(x) = φ^(x/T) / Σ φ^(x/T)   where T = ln(φ)

So the attention weights ARE already φ-levels. The routing pattern after
softmax is a DISCRETE structure in φ-space. We should LUT the attention
pattern itself, not the hidden state.

The plan:
  Part A: Extract attention patterns at layer 23 for calibration prompts
          - What do the softmax weights look like?
          - How many unique φ-level patterns are there?
          - Do they cluster?

  Part B: Map attention patterns → φ-level signatures
          - Discretize each attention weight to its φ-level
          - Hash the pattern for LUT lookup

  Part C: Build LUT: attention φ-pattern → correction vector
          - For each unique pattern, average the error
          - Test on held-out prompts

  Part D: If the full attention pattern is too high-dim, try:
          - Per-head attention entropy (scalar per head)
          - Per-head argmax position (which token gets most attention)
          - These are natural φ-level features of the routing
"""

import sys
import json
import numpy as np
import time
import gc

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax
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


def extract_attention_pattern(engine, layer_idx, hidden):
    """
    Run JUST the attention portion of the layer and return the attention weights.
    
    Returns:
        attn_weights: (batch, num_heads, q_len, kv_len) after softmax
        attn_output_before_residual: the attention contribution (before adding residual)
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    
    batch, seq_len, hidden_dim = hidden.shape
    
    # Pre-attention RMSNorm
    normed = rms_norm(hidden, attn.norm_weight)
    
    # Q/K/V projections
    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    V = phi_linear(attn.W_v, normed, attn.b_v)
    
    # Reshape for multi-head
    Q = Q.reshape(batch, seq_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    
    # RoPE
    Q = attn.rope.apply(Q)
    K = attn.rope.apply(K)
    
    # GQA expansion
    K_expanded = np.repeat(K, attn.heads_per_kv, axis=1)
    V_expanded = np.repeat(V, attn.heads_per_kv, axis=1)
    
    # Attention scores
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K_expanded) * attn.scale
    
    # Causal mask
    kv_len = K_expanded.shape[2]
    if kv_len > 1 and seq_len > 1:
        causal_mask = np.triu(np.full((seq_len, kv_len), -1e9, dtype=np.float32), k=1)
        scores = scores + causal_mask
    
    # Softmax — THIS is where the φ-level selection happens
    attn_weights = phi_softmax(scores, axis=-1)
    
    # Weighted sum
    attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_expanded)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    # Output projection
    attn_proj = phi_linear(attn.W_o, attn_output)
    
    return attn_weights, scores, attn_proj


def attn_weights_to_phi_levels(attn_weights):
    """Convert attention weights to φ-levels.
    
    attn_weight ∈ (0, 1], so log_φ(w) is ≤ 0.
    Level = round(64 × log_φ(w)) gives integer φ-levels.
    """
    w = np.clip(attn_weights, 1e-20, 1.0)
    levels = np.round(64.0 * np.log(w) / LOG_PHI).astype(int)
    return levels


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
    #   Part A: Extract attention patterns at layer 23
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: Attention patterns at layer 23")
    print("=" * 80)
    
    # First, look at the France prompt in detail
    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode_token(i) for i in ids]
    
    h = engine.embedding(ids)[np.newaxis, :, :]
    for layer in engine.layers:
        if layer.layer_idx == target_layer:
            break
        h = layer(h)
    
    attn_weights, scores, attn_proj = extract_attention_pattern(engine, target_layer, h)
    
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Tokens: {tokens}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"    = (batch={attn_weights.shape[0]}, heads={attn_weights.shape[1]}, "
          f"q={attn_weights.shape[2]}, kv={attn_weights.shape[3]})", flush=True)
    
    # Focus on last token (the one that determines prediction)
    last_attn = attn_weights[0, :, -1, :]  # (28 heads, seq_len)
    
    print(f"\n  Last-token attention distribution (28 heads × {last_attn.shape[1]} positions):")
    for h_idx in range(28):
        w = last_attn[h_idx]
        argmax_pos = int(np.argmax(w))
        max_w = w[argmax_pos]
        entropy = -np.sum(w * np.log(w + 1e-20))
        # φ-level of the max weight
        phi_level = int(np.round(64.0 * np.log(max_w + 1e-20) / LOG_PHI))
        print(f"    Head {h_idx:2d}: argmax=pos {argmax_pos} ({tokens[argmax_pos]:>12s}) "
              f"w={max_w:.4f} φ-level={phi_level:4d}  H={entropy:.3f}")
    
    # Convert to φ-levels
    phi_levels = attn_weights_to_phi_levels(last_attn)
    print(f"\n  φ-level range of attention weights: [{phi_levels.min()}, {phi_levels.max()}]")
    unique_levels = np.unique(phi_levels)
    print(f"  Unique φ-levels: {len(unique_levels)}  ({unique_levels[:10]}...)")
    
    # =========================================================================
    #   Part A2: Collect attention patterns across all calibration prompts
    # =========================================================================
    print(f"\n  Collecting attention patterns for {len(cal_prompts)} calibration prompts...")
    
    cal_data = []
    for pi, p in enumerate(cal_prompts):
        p_ids = tokenizer.encode(p)
        p_tokens = [tokenizer.decode_token(i) for i in p_ids]
        
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for layer_obj in engine.layers:
            if layer_obj.layer_idx == target_layer:
                full_out = layer_obj(h.copy())
                spec_out = spec_layer(h.copy())
                break
            h = layer_obj(h)
        
        aw, sc, ap = extract_attention_pattern(engine, target_layer, h)
        
        # Last token attention and error
        last_attn = aw[0, :, -1, :]  # (28, seq_len)
        err_vec = (full_out - spec_out)[0, -1, :]  # (3584,)
        
        # Per-head features of the attention pattern
        head_features = []
        for hi in range(28):
            w = last_attn[hi]
            argmax_pos = int(np.argmax(w))
            max_w = float(w[argmax_pos])
            entropy = float(-np.sum(w * np.log(w + 1e-20)))
            head_features.append({
                'argmax': argmax_pos,
                'max_w': max_w,
                'entropy': entropy,
                'phi_level': int(np.round(64.0 * np.log(max_w + 1e-20) / LOG_PHI)),
            })
        
        cal_data.append({
            'prompt': p,
            'tokens': p_tokens,
            'seq_len': len(p_ids),
            'last_attn': last_attn,
            'error': err_vec,
            'head_features': head_features,
        })
        
        if pi % 10 == 0:
            print(f"    {pi}/{len(cal_prompts)}...", flush=True)
        gc.collect()
    
    print(f"  Done. Collected {len(cal_data)} patterns.")
    
    # =========================================================================
    #   Part B: Analyze attention pattern structure
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Attention pattern structure in φ-space")
    print("=" * 80)
    
    # Per-head entropy statistics
    print(f"\n  Per-head entropy across prompts:")
    for hi in range(28):
        entropies = [d['head_features'][hi]['entropy'] for d in cal_data]
        print(f"    Head {hi:2d}: mean={np.mean(entropies):.3f}  "
              f"std={np.std(entropies):.3f}  "
              f"range=[{np.min(entropies):.3f}, {np.max(entropies):.3f}]")
    
    # Per-head argmax position — is each head consistently attending to the same position?
    print(f"\n  Per-head argmax position (which token gets most attention):")
    for hi in range(28):
        argmaxes = [d['head_features'][hi]['argmax'] for d in cal_data]
        # Relative position: argmax / seq_len
        rel_positions = [d['head_features'][hi]['argmax'] / d['seq_len'] for d in cal_data]
        # How many unique positions?
        unique_abs = len(set(argmaxes))
        mean_rel = np.mean(rel_positions)
        std_rel = np.std(rel_positions)
        print(f"    Head {hi:2d}: {unique_abs:2d} unique positions  "
              f"rel_pos={mean_rel:.2f}±{std_rel:.2f}")
    
    # Per-head max φ-level
    print(f"\n  Per-head max weight φ-level:")
    for hi in range(28):
        levels = [d['head_features'][hi]['phi_level'] for d in cal_data]
        print(f"    Head {hi:2d}: mean={np.mean(levels):.1f}  "
              f"std={np.std(levels):.1f}  "
              f"unique={len(set(levels))}")
    
    # =========================================================================
    #   Part B2: Can we build a signature from the attention pattern?
    # =========================================================================
    print(f"\n  Building attention signatures...")
    
    # Signature: per-head (argmax_relative_pos, entropy_quantized, max_phi_level)
    # This captures WHAT each head is routing (argmax), HOW focused it is (entropy),
    # and the φ-level of the routing weight
    
    all_signatures = []
    all_errors = []
    all_attn_flat = []
    
    for d in cal_data:
        sig = []
        for hi in range(28):
            hf = d['head_features'][hi]
            # Quantize: relative argmax pos to bins, entropy to φ-levels
            rel_pos = hf['argmax'] / d['seq_len']
            rel_pos_bin = int(rel_pos * 4)  # 4 bins: [0, 0.25, 0.5, 0.75, 1.0]
            entropy_level = int(np.round(hf['entropy'] * 4))  # quantize entropy
            sig.extend([rel_pos_bin, entropy_level, hf['phi_level']])
        
        all_signatures.append(tuple(sig))
        all_errors.append(d['error'])
        
        # Also flatten the full last-token attention for comparison
        # Pad to max seq len for consistent shape
        padded = np.zeros((28, 20))  # max 20 tokens
        sl = d['last_attn'].shape[1]
        padded[:, :sl] = d['last_attn']
        all_attn_flat.append(padded.flatten())
    
    all_errors = np.array(all_errors)
    all_attn_flat = np.array(all_attn_flat)
    
    n_unique_sigs = len(set(all_signatures))
    print(f"  Unique attention signatures: {n_unique_sigs} / {len(all_signatures)}")
    
    # =========================================================================
    #   Part B3: How well does the full attention pattern predict the error?
    # =========================================================================
    print(f"\n  Attention pattern → error prediction:")
    
    # SVD of attention patterns
    A_centered = all_attn_flat - all_attn_flat.mean(axis=0)
    U_a, S_a, Vt_a = np.linalg.svd(A_centered, full_matrices=False)
    
    total_a = (S_a ** 2).sum()
    print(f"  Attention SVD:")
    for k in [1, 3, 5, 10, 20]:
        cap = (S_a[:k] ** 2).sum() / total_a * 100
        print(f"    Rank-{k:2d}: {cap:.1f}% of attention variance")
    
    # Project attention onto top PCs
    A_proj = A_centered @ Vt_a[:20].T
    
    # Regress: attention PCs → error
    for n_pcs in [3, 5, 10, 20]:
        A_feat = A_proj[:, :n_pcs]
        W, _, _, _ = np.linalg.lstsq(A_feat, all_errors, rcond=1e-3)
        E_pred = A_feat @ W
        r2 = 1.0 - np.sum((all_errors - E_pred) ** 2) / np.sum(all_errors ** 2)
        print(f"    {n_pcs:2d} attn PCs → error R²: {r2:.4f}")
    
    # =========================================================================
    #   Part B4: Compact attention features
    # =========================================================================
    print(f"\n  Compact attention features → error prediction:")
    
    # Build compact feature matrix from per-head scalars
    compact_features = []
    for d in cal_data:
        feats = []
        for hi in range(28):
            hf = d['head_features'][hi]
            feats.append(hf['entropy'])
            feats.append(hf['max_w'])
            feats.append(hf['argmax'] / d['seq_len'])
        compact_features.append(feats)
    
    compact_features = np.array(compact_features)  # (38, 84) — 3 features × 28 heads
    print(f"  Compact feature matrix: {compact_features.shape}")
    
    W_c, _, _, _ = np.linalg.lstsq(compact_features, all_errors, rcond=1e-3)
    E_pred_c = compact_features @ W_c
    r2_compact = 1.0 - np.sum((all_errors - E_pred_c) ** 2) / np.sum(all_errors ** 2)
    print(f"  84 compact attn features → error R²: {r2_compact:.4f}")
    
    # Per-head entropy only (28 features)
    entropy_features = np.array([[d['head_features'][hi]['entropy'] 
                                   for hi in range(28)] for d in cal_data])
    W_e, _, _, _ = np.linalg.lstsq(entropy_features, all_errors, rcond=1e-3)
    E_pred_e = entropy_features @ W_e
    r2_entropy = 1.0 - np.sum((all_errors - E_pred_e) ** 2) / np.sum(all_errors ** 2)
    print(f"  28 per-head entropies → error R²: {r2_entropy:.4f}")
    
    # Per-head argmax position only (28 features)
    argmax_features = np.array([[d['head_features'][hi]['argmax'] / d['seq_len']
                                  for hi in range(28)] for d in cal_data])
    W_am, _, _, _ = np.linalg.lstsq(argmax_features, all_errors, rcond=1e-3)
    E_pred_am = argmax_features @ W_am
    r2_argmax = 1.0 - np.sum((all_errors - E_pred_am) ** 2) / np.sum(all_errors ** 2)
    print(f"  28 per-head argmax rel-pos → error R²: {r2_argmax:.4f}")
    
    # =========================================================================
    #   Part C: Build attention-based LUT and test
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Attention-pattern LUT correction")
    print("=" * 80)
    
    # Strategy: use compact attention features for clustering, then LUT
    def simple_kmeans(X, k, n_iter=20):
        n = X.shape[0]
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=min(k, n), replace=False)
        centroids = X[idx].copy()
        for _ in range(n_iter):
            dists = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = X[mask].mean(axis=0)
        return centroids, labels
    
    # Train LUTs with different feature sets and cluster counts
    feature_sets = {
        'entropy_28': entropy_features,
        'argmax_28': argmax_features,
        'compact_84': compact_features,
    }
    
    for feat_name, feat_matrix in feature_sets.items():
        print(f"\n  Feature set: {feat_name} ({feat_matrix.shape[1]} dims)")
        
        for n_clusters in [4, 8, 16]:
            centroids, labels = simple_kmeans(feat_matrix, n_clusters)
            
            # Build correction LUT
            lut = {}
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    lut[c] = all_errors[mask].mean(axis=0)
                else:
                    lut[c] = np.zeros(3584)
            
            # Train R²
            E_corr = np.zeros_like(all_errors)
            for i in range(len(all_errors)):
                E_corr[i] = all_errors[i] - lut[labels[i]]
            train_r2 = 1.0 - np.sum(E_corr ** 2) / np.sum(all_errors ** 2)
            
            # Test on held-out prompts
            n_pass = 0
            results = []
            for prompt in test_prompts:
                p_ids = tokenizer.encode(prompt)
                h = engine.embedding(p_ids)[np.newaxis, :, :]
                for layer_obj in engine.layers:
                    if layer_obj.layer_idx == target_layer:
                        full_out = layer_obj(h.copy())
                        spec_out = spec_layer(h.copy())
                        break
                    h = layer_obj(h)
                
                # Extract attention features for the test prompt
                aw, sc, ap = extract_attention_pattern(engine, target_layer, h)
                test_feats = []
                for hi in range(28):
                    w = aw[0, hi, -1, :]
                    am = int(np.argmax(w))
                    mw = float(w[am])
                    ent = float(-np.sum(w * np.log(w + 1e-20)))
                    
                    if feat_name == 'entropy_28':
                        test_feats.append(ent)
                    elif feat_name == 'argmax_28':
                        test_feats.append(am / len(p_ids))
                    else:  # compact_84
                        test_feats.extend([ent, mw, am / len(p_ids)])
                
                test_feats = np.array(test_feats)
                
                # Find nearest cluster
                dists = np.linalg.norm(centroids - test_feats[np.newaxis, :], axis=1)
                nearest = int(np.argmin(dists))
                
                # Apply correction to spectrometer output
                corrected = spec_out.copy()
                for t in range(corrected.shape[1]):
                    corrected[0, t] += lut[nearest]
                
                logits_full = finish_forward(engine, full_out, target_layer)
                logits_corr = finish_forward(engine, corrected, target_layer)
                
                full_id, full_tok, _ = get_top1(logits_full, tokenizer)
                corr_id, corr_tok, corr_margin = get_top1(logits_corr, tokenizer)
                
                match = corr_id == full_id
                if match: n_pass += 1
                results.append((prompt, match, corr_tok, full_tok, corr_margin))
            
            # Print summary
            france_result = [r for r in results if 'France' in r[0]][0]
            france_mark = '✓' if france_result[1] else '✗'
            france_margin = france_result[4]
            print(f"    {n_clusters:2d} clusters: train R²={train_r2:.4f}  "
                  f"test={n_pass}/{len(test_prompts)}  "
                  f"France={france_mark} margin={france_margin:.3f}")
    
    # =========================================================================
    #   Part D: Linear regression correction (attention features → error)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Linear correction (attention features → error directly)")
    print("=" * 80)
    print("  Instead of clustering, directly regress attention features → correction")
    
    for feat_name, feat_matrix in feature_sets.items():
        print(f"\n  Feature set: {feat_name} ({feat_matrix.shape[1]} dims)")
        
        # Learn W such that feat_matrix @ W ≈ all_errors
        W_reg, _, _, _ = np.linalg.lstsq(feat_matrix, all_errors, rcond=1e-3)
        
        # Test on held-out
        n_pass = 0
        for prompt in test_prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for layer_obj in engine.layers:
                if layer_obj.layer_idx == target_layer:
                    full_out = layer_obj(h.copy())
                    spec_out = spec_layer(h.copy())
                    break
                h = layer_obj(h)
            
            aw, sc, ap = extract_attention_pattern(engine, target_layer, h)
            test_feats = []
            for hi in range(28):
                w = aw[0, hi, -1, :]
                am = int(np.argmax(w))
                mw = float(w[am])
                ent = float(-np.sum(w * np.log(w + 1e-20)))
                
                if feat_name == 'entropy_28':
                    test_feats.append(ent)
                elif feat_name == 'argmax_28':
                    test_feats.append(am / len(p_ids))
                else:
                    test_feats.extend([ent, mw, am / len(p_ids)])
            
            test_feats = np.array(test_feats)
            correction = test_feats @ W_reg
            
            corrected = spec_out.copy()
            for t in range(corrected.shape[1]):
                corrected[0, t] += correction
            
            logits_full = finish_forward(engine, full_out, target_layer)
            logits_corr = finish_forward(engine, corrected, target_layer)
            
            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            corr_id, corr_tok, corr_margin = get_top1(logits_corr, tokenizer)
            
            match = '✓' if corr_id == full_id else '✗'
            if corr_id == full_id: n_pass += 1
            
            extra = f" margin={corr_margin:.3f}" if 'France' in prompt else ""
            print(f"    {match} {prompt:>35s} → {corr_tok:>8s} (want {full_tok}){extra}")
        
        print(f"    Score: {n_pass}/{len(test_prompts)}")
    
    # =========================================================================
    #   Part E: The nuclear option — use ACTUAL attention output as correction
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Direct attention output correction")
    print("=" * 80)
    print("  If we run the REAL attention (which we already did to get patterns),")
    print("  can we use its output directly to correct the spectrometer?")
    print("  This would mean: spectrometer rules for MLP, real matmuls for attention.")
    
    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for layer_obj in engine.layers:
            if layer_obj.layer_idx == target_layer:
                full_out = layer_obj(h.copy())
                spec_out = spec_layer(h.copy())
                break
            h = layer_obj(h)
        
        # Get real attention output
        aw, sc, attn_proj = extract_attention_pattern(engine, target_layer, h)
        
        # The full layer does: hidden + attn_proj + mlp_output
        # The spectrometer does: rules(hidden) ≈ hidden + attn_proj + mlp_output
        # 
        # What if we do: spectrometer_rules(hidden) + (real_attn_proj - estimated_attn_proj)?
        # Or simpler: replace the spectrometer output with: hidden + real_attn_proj + spec_mlp_part
        #
        # But we don't have the spec_mlp_part separately...
        # Instead, let's try: spec_out + (real_attn_proj - spec_attn_estimate)
        # where spec_attn_estimate is what the spectrometer "thinks" the attention contrib is
        #
        # Actually the cleanest test: run real attention, then spectrometer MLP on the result
        
        # Real attention step
        post_attn = h + attn_proj  # hidden + attention_output
        
        # Now run MLP with spectrometer rules on the post-attention state
        # But SpectrometerLayer doesn't separate attn/MLP...
        # 
        # Simpler test: just use the full layer output as ground truth
        # and compare: does the error come from attention or MLP?
        
        # Error decomposition: 
        # full_out = h + attn_proj + mlp_out
        # spec_out = spec_rules(h)
        # error = full_out - spec_out = (h + attn_proj + mlp_out) - spec_rules(h)
        
        # If we had spec_rules separately for attn and MLP contribution,
        # we could identify which part is wrong. But we don't.
        # 
        # However: we CAN measure how much of the error is explained by
        # the difference between real attn_proj and what "should" be there
        
        error = (full_out - spec_out)[0, -1, :]
        
        # The attention projection is added to hidden, so if spectrometer
        # gets the attn part wrong, the error should correlate with attn_proj
        attn_contrib = attn_proj[0, -1, :]
        
        if 'France' in prompt:
            cos_sim = np.dot(error, attn_contrib) / (
                np.linalg.norm(error) * np.linalg.norm(attn_contrib) + 1e-20)
            print(f"\n  France prompt error analysis:")
            print(f"    ||error||: {np.linalg.norm(error):.2f}")
            print(f"    ||attn_proj||: {np.linalg.norm(attn_contrib):.2f}")
            print(f"    cos(error, attn_proj): {cos_sim:.4f}")
            
            # Project error onto attn_proj direction
            proj_coeff = np.dot(error, attn_contrib) / (np.dot(attn_contrib, attn_contrib) + 1e-20)
            proj = proj_coeff * attn_contrib
            residual = error - proj
            print(f"    Error projected onto attn_proj: {np.linalg.norm(proj):.2f} "
                  f"({np.linalg.norm(proj)/np.linalg.norm(error)*100:.1f}%)")
            print(f"    Residual: {np.linalg.norm(residual):.2f}")
    
    # Now the hybrid test: run real attention, feed result to spectrometer MLP
    print(f"\n  Hybrid test: real attention + spectrometer MLP")
    print(f"  (Computing the MLP part of the layer manually)")
    
    layer_obj = engine.layers[target_layer]
    
    n_pass = 0
    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == target_layer:
                full_out = lo(h.copy())
                break
            h = lo(h)
        
        # Step 1: Real attention
        aw, sc, attn_proj = extract_attention_pattern(engine, target_layer, h)
        post_attn = h + attn_proj
        
        # Step 2: Real MLP (this is the part the spectrometer SHOULD handle)
        mlp = layer_obj.mlp
        normed_mlp = rms_norm(post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        from phi_geometric.inference.phi_components import phi_silu
        mlp_hidden = phi_silu(gate) * up
        mlp_out = phi_linear(mlp.W_down, mlp_hidden)
        hybrid_out = post_attn + mlp_out
        
        logits_full = finish_forward(engine, full_out, target_layer)
        logits_hybrid = finish_forward(engine, hybrid_out, target_layer)
        
        full_id, full_tok, _ = get_top1(logits_full, tokenizer)
        hyb_id, hyb_tok, hyb_margin = get_top1(logits_hybrid, tokenizer)
        
        match = '✓' if hyb_id == full_id else '✗'
        if hyb_id == full_id: n_pass += 1
        
        extra = f" margin={hyb_margin:.3f}" if 'France' in prompt else ""
        print(f"    {match} {prompt:>35s} → {hyb_tok:>8s} (want {full_tok}){extra}")
    
    print(f"  Score: {n_pass}/{len(test_prompts)}")
    print(f"  (This is the upper bound: real attn + real MLP = full layer)")
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
