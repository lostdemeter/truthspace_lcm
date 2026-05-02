"""
Phase 4b: Investigating Distributed Attention Geometry
=======================================================

F130 showed: MESH is rank-1 universal, but decomposition layers (L0-L21)
need distributed attention. The geometric selector disagrees with softmax
at ~25/28 heads per layer.

This script investigates:
  1. What do the actual attention patterns look like at each layer?
  2. Is there geometric structure in the distribution?
  3. What role does RoPE play in transforming the bias-dominated MESH?
  4. Are there identifiable attention "modes" across layers?

Prompt: "The capital of France is" (5 tokens)
Tokens: [BOS=0, "The"=1, " capital"=2, " of"=3, " France"=4, " is"=5]
(Note: BOS is token 0 if present, check actual tokenization)
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_integer import phi_to_float

from experiments.geometric_instrument.components.selector import Selector

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def get_full_attention_info(engine, h, layer_idx):
    """Get complete attention information for a layer.
    
    Returns:
        weights: [nh, seq_len, seq_len] full attention weights
        scores:  [nh, seq_len, seq_len] pre-softmax scores
        h_out:   [1, seq_len, d_model] output hidden states
    """
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
    
    # Also compute without RoPE for comparison
    Q_noR = phi_linear(attn.W_q, normed, attn.b_q)
    K_noR = phi_linear(attn.W_k, normed, attn.b_k)
    Q_noR = Q_noR.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K_noR = K_noR.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Ke_noR = np.repeat(K_noR, hpk, axis=1)
    scores_noR = np.einsum('bhqd,bhkd->bhqk', Q_noR, Ke_noR) * attn.scale
    if seq_len > 1:
        scores_noR += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    weights_noR = phi_softmax(scores_noR, axis=-1)
    
    # Run the actual layer for h_out
    h_out = layer(h)
    
    return {
        'weights': weights[0],          # [nh, seq, seq]
        'scores': scores[0],            # [nh, seq, seq]
        'weights_noR': weights_noR[0],  # [nh, seq, seq] without RoPE
        'scores_noR': scores_noR[0],    # [nh, seq, seq] without RoPE
        'h_out': h_out,
        'normed': normed,
    }


def entropy(weights):
    """Compute entropy of attention weights."""
    eps = 1e-12
    return -np.sum(weights * np.log(weights + eps), axis=-1)


def main():
    print("=" * 80)
    print("  Phase 4b: Investigating Distributed Attention Geometry")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    # Tokenize prompt
    prompt = "The capital of France is"
    tids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode([t]) for t in tids]
    seq_len = len(tids)
    
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Tokens ({seq_len}): {list(enumerate(tokens))}")
    
    n_layers = len(engine.layers)
    nh = engine.layers[0].attention.num_heads
    nkv = engine.layers[0].attention.num_kv_heads
    hpk = nh // nkv
    
    # ═══════════════════════════════════════════════════════════
    # Analysis 1: Last-token attention heatmap per layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  Analysis 1: Last-token attention weights per layer (averaged over heads)")
    print("  Each row = one layer, columns = positions attended to")
    print("─" * 80)
    
    # Header
    pos_labels = [f"p{i}({tokens[i].strip()[:6]})" for i in range(seq_len)]
    header = "Layer  " + "  ".join(f"{l:>10}" for l in pos_labels) + "  entropy  argmax"
    print(f"  {header}")
    print("  " + "─" * len(header))
    
    h = engine.embedding(tids)[np.newaxis, :, :]
    
    # Store per-layer data
    all_weights_lt = []       # last-token weights with RoPE
    all_weights_lt_noR = []   # last-token weights without RoPE
    all_argmax_with_rope = []
    all_argmax_without_rope = []
    
    for li in range(n_layers):
        info = get_full_attention_info(engine, h, li)
        
        # Last-token attention (averaged over all 28 heads)
        w_lt = info['weights'][:, -1, :]  # [nh, seq_len]
        w_lt_noR = info['weights_noR'][:, -1, :]  # [nh, seq_len]
        
        all_weights_lt.append(w_lt)
        all_weights_lt_noR.append(w_lt_noR)
        
        w_avg = w_lt.mean(axis=0)
        ent = float(entropy(w_avg))
        am = int(np.argmax(w_avg))
        
        vals = "  ".join(f"{w:>10.4f}" for w in w_avg)
        print(f"  L{li:>2}:  {vals}  {ent:.3f}   p{am}")
        
        # Track per-head argmax
        am_rope = np.argmax(w_lt, axis=-1)  # [nh]
        am_norope = np.argmax(w_lt_noR, axis=-1)  # [nh]
        all_argmax_with_rope.append(am_rope)
        all_argmax_without_rope.append(am_norope)
        
        h = info['h_out']
    
    # ═══════════════════════════════════════════════════════════
    # Analysis 2: RoPE's effect on routing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  Analysis 2: RoPE's effect on routing (per-head argmax comparison)")
    print("  For each layer: how many heads change their argmax when RoPE is removed?")
    print("─" * 80)
    
    for li in range(n_layers):
        am_r = all_argmax_with_rope[li]
        am_nr = all_argmax_without_rope[li]
        n_changed = int(np.sum(am_r != am_nr))
        
        # Show the distribution of where attention goes with vs without RoPE
        r_counts = np.bincount(am_r, minlength=seq_len)
        nr_counts = np.bincount(am_nr, minlength=seq_len)
        
        r_str = " ".join(f"p{i}:{c}" for i, c in enumerate(r_counts) if c > 0)
        nr_str = " ".join(f"p{i}:{c}" for i, c in enumerate(nr_counts) if c > 0)
        
        print(f"  L{li:>2}: {n_changed:>2}/28 changed  "
              f"with_RoPE=[{r_str}]  no_RoPE=[{nr_str}]")
    
    # ═══════════════════════════════════════════════════════════
    # Analysis 3: Per-KV-group attention patterns
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  Analysis 3: KV group attention patterns (last-token)")
    print("  Do heads within the same KV group attend to the same position?")
    print("─" * 80)
    
    for li in [0, 5, 10, 15, 20, 23, 27]:
        print(f"\n  Layer {li}:")
        w_lt = all_weights_lt[li]  # [nh, seq_len]
        
        for kv in range(nkv):
            head_start = kv * hpk
            head_end = (kv + 1) * hpk
            group_weights = w_lt[head_start:head_end]  # [hpk, seq_len]
            group_argmax = np.argmax(group_weights, axis=-1)
            
            # Intra-group agreement: do all heads in this KV group attend same position?
            unique_positions = np.unique(group_argmax)
            group_avg = group_weights.mean(axis=0)
            
            pos_str = ", ".join(f"p{p}" for p in group_argmax)
            avg_str = " ".join(f"{w:.3f}" for w in group_avg)
            agreement = f"{'AGREE' if len(unique_positions) == 1 else f'{len(unique_positions)} targets'}"
            
            print(f"    KV{kv} (H{head_start}-H{head_end-1}): argmax=[{pos_str}] "
                  f"{agreement}  avg=[{avg_str}]")
    
    # ═══════════════════════════════════════════════════════════
    # Analysis 4: Attention mode classification
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  Analysis 4: Attention modes per head (last-token)")
    print("  Classifying each head's attention as: BOS-focus, subject-focus,")
    print("  last-focus, distributed, or other")
    print("─" * 80)
    
    # For our prompt: p0=BOS-ish, p1=The, p2=capital, p3=of, p4=France, p5=is
    # Subject = France (p4), last = is (p5)
    subject_pos = seq_len - 2  # "France"
    last_pos = seq_len - 1     # "is"
    
    mode_counts = {}
    for li in range(n_layers):
        layer_modes = {'BOS': 0, 'subject': 0, 'last': 0, 'distributed': 0, 'other': 0}
        w_lt = all_weights_lt[li]  # [nh, seq_len]
        
        for hi in range(nh):
            w = w_lt[hi]
            am = int(np.argmax(w))
            max_w = float(w[am])
            ent = float(entropy(w))
            
            if max_w < 0.5:
                mode = 'distributed'
            elif am == 0:
                mode = 'BOS'
            elif am == subject_pos:
                mode = 'subject'
            elif am == last_pos:
                mode = 'last'
            else:
                mode = 'other'
            
            layer_modes[mode] += 1
        
        mode_counts[li] = layer_modes
        
        bar = (f"BOS={layer_modes['BOS']:>2} "
               f"subj={layer_modes['subject']:>2} "
               f"last={layer_modes['last']:>2} "
               f"dist={layer_modes['distributed']:>2} "
               f"other={layer_modes['other']:>2}")
        print(f"  L{li:>2}: {bar}")
    
    # ═══════════════════════════════════════════════════════════
    # Analysis 5: Score decomposition — bias vs RoPE-Q·K contribution
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  Analysis 5: Score decomposition at key layers")
    print("  For the last token's query, decompose score into:")
    print("    score = (bias_contribution) + (content_contribution) + (RoPE_effect)")
    print("─" * 80)
    
    h = engine.embedding(tids)[np.newaxis, :, :]
    
    for li in range(n_layers):
        layer = engine.layers[li]
        attn = layer.attention
        hd = attn.head_dim
        
        normed = rms_norm(h, attn.norm_weight)
        
        # Full Q, K with bias
        Q_full = phi_linear(attn.W_q, normed, attn.b_q)
        K_full = phi_linear(attn.W_k, normed, attn.b_k)
        
        # Q, K without bias
        Q_nobias = phi_linear(attn.W_q, normed)
        K_nobias = phi_linear(attn.W_k, normed)
        
        # Reshape for head 6 (the knowledge head at L23)
        # Use head 0 from each KV group as representative
        if li in [0, 5, 10, 15, 20, 23, 27]:
            print(f"\n  Layer {li}:")
            
            for kv in range(nkv):
                hi = kv * hpk
                
                # Get head's Q and K
                q_full = Q_full[0, -1, hi*hd:(hi+1)*hd]  # last token query
                q_nobias = Q_nobias[0, -1, hi*hd:(hi+1)*hd]
                
                k_full = K_full[0, :, kv*hd:(kv+1)*hd]  # all key positions
                k_nobias = K_nobias[0, :, kv*hd:(kv+1)*hd]
                
                # Bias vectors
                bq = attn.b_q[hi*hd:(hi+1)*hd]
                bk = attn.b_k[kv*hd:(kv+1)*hd]
                
                # Score = q·k = (q_content + bq)·(k_content + bk)
                #       = q_content·k_content + q_content·bk + bq·k_content + bq·bk
                
                scores_full = q_full @ k_full.T * attn.scale
                
                # Pure bias: bq · bk (same for all positions)
                bias_bias = float(bq @ bk) * attn.scale
                
                # Content-content: q_nobias · k_nobias
                content_content = q_nobias @ k_nobias.T * attn.scale
                
                # Cross terms
                q_content_bk = q_nobias @ bk * attn.scale  # scalar (same for all k)
                bq_k_content = bq @ k_nobias.T * attn.scale  # per position
                
                # NOTE: These are WITHOUT RoPE. With RoPE the decomposition is different.
                # But this shows the relative magnitudes.
                
                total_range = float(np.max(scores_full) - np.min(scores_full))
                cc_range = float(np.max(content_content) - np.min(content_content))
                bkc_range = float(np.max(bq_k_content) - np.min(bq_k_content))
                
                print(f"    KV{kv} H{hi}: bias²={bias_bias:>8.1f}  "
                      f"content_range={cc_range:>7.1f}  "
                      f"bq·k_range={bkc_range:>7.1f}  "
                      f"total_range={total_range:>7.1f}  "
                      f"bias_frac={abs(bias_bias)/max(abs(bias_bias)+total_range,1e-6)*100:.0f}%")
        
        h = layer(h)
    
    # ═══════════════════════════════════════════════════════════
    # Analysis 6: Does position determine attention, or content?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  Analysis 6: Position vs content — test with permuted input")
    print("  If we shuffle the token positions, do attention patterns change?")
    print("─" * 80)
    
    # Original hidden states
    h_orig = engine.embedding(tids)[np.newaxis, :, :]
    # Permuted: swap France and The (positions 1 and 4, or whichever)
    # Actually, let's try a simple test: reverse the order (except BOS if present)
    
    # First, run layer 0 with original and see attention
    info_orig = get_full_attention_info(engine, h_orig, 0)
    w_orig_lt = info_orig['weights'][:, -1, :]  # [nh, seq]
    
    # Now create a version where we swap embeddings of positions 1 and 3
    h_swap = h_orig.copy()
    h_swap[0, 1, :], h_swap[0, 3, :] = h_orig[0, 3, :].copy(), h_orig[0, 1, :].copy()
    
    info_swap = get_full_attention_info(engine, h_swap, 0)
    w_swap_lt = info_swap['weights'][:, -1, :]  # [nh, seq]
    
    # Compare
    max_diff = float(np.max(np.abs(w_orig_lt - w_swap_lt)))
    mean_diff = float(np.mean(np.abs(w_orig_lt - w_swap_lt)))
    
    # How many heads change their argmax?
    am_orig = np.argmax(w_orig_lt, axis=-1)
    am_swap = np.argmax(w_swap_lt, axis=-1)
    n_changed = int(np.sum(am_orig != am_swap))
    
    print(f"\n  L0: Swapped embeddings at pos 1 ('{tokens[1]}') and pos 3 ('{tokens[3]}')")
    print(f"  Max weight diff: {max_diff:.4f}")
    print(f"  Mean weight diff: {mean_diff:.4f}")
    print(f"  Heads that change argmax: {n_changed}/28")
    print(f"\n  Original argmax: {list(am_orig)}")
    print(f"  Swapped argmax:  {list(am_swap)}")
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    # Count dominant modes per region
    decomp_modes = {'BOS': 0, 'subject': 0, 'last': 0, 'distributed': 0, 'other': 0}
    extract_modes = {'BOS': 0, 'subject': 0, 'last': 0, 'distributed': 0, 'other': 0}
    
    for li in range(22):
        for k, v in mode_counts[li].items():
            decomp_modes[k] += v
    for li in range(22, 28):
        for k, v in mode_counts[li].items():
            extract_modes[k] += v
    
    print(f"\n  Decomposition layers (L0-L21) head modes ({22*28} total heads):")
    for k, v in decomp_modes.items():
        print(f"    {k:>12}: {v:>3} ({v/(22*28)*100:.0f}%)")
    
    print(f"\n  Extraction+amp layers (L22-L27) head modes ({6*28} total heads):")
    for k, v in extract_modes.items():
        print(f"    {k:>12}: {v:>3} ({v/(6*28)*100:.0f}%)")
    
    print()


if __name__ == '__main__':
    main()
