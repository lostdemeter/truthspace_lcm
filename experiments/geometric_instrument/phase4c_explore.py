"""
Phase 4c: Deep Exploration of Distributed Attention Geometry
==============================================================

Three investigations (optimized for speed):
  1. Multi-prompt template stability: BOS-sink across 6 prompts, sampled layers
  2. Fixed-template attention: Replace softmax at L0-L21 with extracted pattern
  3. BOS accumulation trace: p0 hidden state geometry across layers
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}

SAMPLE_LAYERS = [0, 5, 10, 15, 20, 23, 27]


def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids:
        return None, None
    tid = tids[0]
    return int(np.sum(logits > logits[tid])), float(logits[tid])


def get_last_token_attention(engine, h, layer_idx):
    """Get last-token attention weights [nh, seq_len] for a layer."""
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
    return weights[0, :, -1, :]  # [nh, seq_len]


def run_layer_with_fixed_attention(engine, h, layer_idx, fixed_weights):
    """Run a layer replacing last-token attention with fixed weights."""
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
    
    # Replace last-token weights with fixed template
    fw = fixed_weights
    cur_seq, fw_seq = seq_len, fw.shape[1]
    if cur_seq == fw_seq:
        weights[0, :, -1, :] = fw
    elif cur_seq < fw_seq:
        trimmed = fw[:, :cur_seq]
        weights[0, :, -1, :] = trimmed / (trimmed.sum(axis=1, keepdims=True) + 1e-12)
    else:
        padded = np.zeros((nh, cur_seq), dtype=np.float32)
        padded[:, :fw_seq] = fw
        weights[0, :, -1, :] = padded / (padded.sum(axis=1, keepdims=True) + 1e-12)
    
    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    ao = phi_linear(attn.W_o, ao)
    h_post_attn = h + ao
    
    mlp = layer.mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    return h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)


def main():
    print("=" * 80)
    print("  Phase 4c: Deep Exploration of Distributed Attention Geometry")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    nh = 28
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 1: Multi-prompt template stability (sampled layers only)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 1: Template Stability Across Prompts")
    print("  (sampling layers", SAMPLE_LAYERS, ")")
    print("=" * 80)
    
    print(f"\n  Per-prompt BOS attention fraction (avg over 28 heads):")
    print(f"  {'Prompt':>30}  {'tok':>3}  " +
          "  ".join(f"L{i:>2}" for i in SAMPLE_LAYERS))
    print("  " + "─" * 80)
    
    bos_fracs = {}  # {name: {layer: float}}
    subj_fracs = {}
    
    for name, info in FACTS.items():
        prompt = info['prompt']
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        
        h = engine.embedding(tids)[np.newaxis, :, :]
        bos_fracs[name] = {}
        subj_fracs[name] = {}
        
        layer_strs = []
        for li in range(n_layers):
            if li in SAMPLE_LAYERS:
                w_lt = get_last_token_attention(engine, h, li)
                bos_frac = float(w_lt[:, 0].mean())
                subj_frac = float(w_lt[:, -2].mean())
                bos_fracs[name][li] = bos_frac
                subj_fracs[name][li] = subj_frac
                layer_strs.append(f"{bos_frac:.3f}")
            h = engine.layers[li](h)
        
        print(f"  {prompt:>30}  {seq_len:>3}  " + "  ".join(f"{s:>5}" for s in layer_strs))
    
    # Stability stats
    print(f"\n  Cross-prompt std at each layer:")
    for li in SAMPLE_LAYERS:
        vals = [bos_fracs[n][li] for n in FACTS]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        print(f"    L{li:>2}: mean={mean_v:.3f}, std={std_v:.4f}")
    
    # Subject (second-to-last) fraction
    print(f"\n  Subject position fraction (second-to-last):")
    print(f"  {'Prompt':>30}  " +
          "  ".join(f"L{i:>2}" for i in SAMPLE_LAYERS))
    for name, info in FACTS.items():
        vals = [f"{subj_fracs[name][li]:.3f}" for li in SAMPLE_LAYERS]
        print(f"  {info['prompt']:>30}  " + "  ".join(f"{v:>5}" for v in vals))
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 2: Fixed-template attention tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 2: Fixed-Template Attention")
    print("=" * 80)
    
    france_tids = tokenizer.encode(FACTS['France']['prompt'])
    france_seq = len(france_tids)
    
    print(f"\n  Extracting templates from France ({france_seq} tokens)...", end="", flush=True)
    t0 = time.time()
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    templates = []
    for li in range(n_layers):
        w_lt = get_last_token_attention(engine, h, li)
        templates.append(w_lt.copy())
        h = engine.layers[li](h)
    print(f" done in {time.time()-t0:.1f}s")
    
    # Test A: France-specific template at L0-L21, real at L22-L27
    print("\n  Test A: France template at L0-L21, real L22-L27")
    test_a_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            if li < 22:
                h = run_layer_with_fixed_attention(engine, h, li, templates[li])
            else:
                h = engine.layers[li](h)
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        mark = "✓" if rank == 0 else "✗"
        if rank == 0: test_a_match += 1
        print(f"    {country:>8}: top='{top_tok}', rank={rank} {mark}")
    print(f"  Result: {test_a_match}/6")
    
    # Test B: France template at ALL layers
    print("\n  Test B: France template at ALL layers (L0-L27)")
    test_b_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_fixed_attention(engine, h, li, templates[li])
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        mark = "✓" if rank == 0 else "✗"
        if rank == 0: test_b_match += 1
        print(f"    {country:>8}: top='{top_tok}', rank={rank} {mark}")
    print(f"  Result: {test_b_match}/6")
    
    # Test C: Pure BOS (100% → p0) at L0-L21, real L22-L27
    print("\n  Test C: Pure BOS (100%→p0) at L0-L21, real L22-L27")
    bos_template = np.zeros((nh, france_seq), dtype=np.float32)
    bos_template[:, 0] = 1.0
    test_c_match = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            if li < 22:
                h = run_layer_with_fixed_attention(engine, h, li, bos_template)
            else:
                h = engine.layers[li](h)
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, info['answer'], tokenizer)
        mark = "✓" if rank == 0 else "✗"
        if rank == 0: test_c_match += 1
        print(f"    {country:>8}: top='{top_tok}', rank={rank} {mark}")
    print(f"  Result: {test_c_match}/6")
    
    # Test D: Progressive fixed template (France only)
    print("\n  Test D: Progressive (France only)")
    for n_fixed in [0, 5, 10, 15, 20, 22, 25, 28]:
        if n_fixed > n_layers: continue
        h = engine.embedding(france_tids)[np.newaxis, :, :]
        for li in range(n_layers):
            if li < n_fixed:
                h = run_layer_with_fixed_attention(engine, h, li, templates[li])
            else:
                h = engine.layers[li](h)
        normed = rms_norm(h, engine.final_norm_weight)
        logits = engine.lm_head(normed)[0, -1, :]
        top_tok = tokenizer.decode([int(np.argmax(logits))])
        rank, _ = get_rank(logits, ' Paris', tokenizer)
        mark = "✓" if rank == 0 else "✗"
        label = f"fixed L0-L{n_fixed-1}" if n_fixed > 0 else "all real"
        if n_fixed == n_layers: label = "ALL FIXED"
        print(f"    {label:>20}: top='{top_tok}', rank={rank} {mark}")
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 3: BOS accumulation trace (single forward pass)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 3: BOS Accumulation Trace")
    print("=" * 80)
    
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    
    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    
    print(f"\n  {'Stage':>5}  {'||h0||':>8}  {'||h_last||':>10}  {'ratio':>6}  "
          f"{'cos01':>7}  {'cos03':>7}  {'cos04':>7}")
    print("  " + "─" * 60)
    
    bos_states = []
    for li in range(n_layers + 1):
        h0 = h[0, 0, :]
        h_last = h[0, -1, :]
        norm0 = float(np.linalg.norm(h0))
        norm_l = float(np.linalg.norm(h_last))
        ratio = norm0 / norm_l if norm_l > 0 else 0
        cos01 = cos_sim(h0, h[0, 1, :])
        cos03 = cos_sim(h0, h[0, 3, :])
        cos04 = cos_sim(h0, h[0, 4, :])
        bos_states.append(h0.copy())
        
        label = "emb" if li == 0 else f"L{li-1}"
        print(f"  {label:>5}  {norm0:>8.1f}  {norm_l:>10.1f}  {ratio:>6.3f}  "
              f"{cos01:>7.4f}  {cos03:>7.4f}  {cos04:>7.4f}")
        
        if li < n_layers:
            h = engine.layers[li](h)
    
    # BOS change per layer
    print(f"\n  Per-layer BOS change:")
    print(f"  {'Layer':>5}  {'||Δ||':>8}  {'Δ/||h||':>8}  {'cos(prev,cur)':>14}")
    print("  " + "─" * 40)
    for i in range(1, len(bos_states)):
        delta = float(np.linalg.norm(bos_states[i] - bos_states[i-1]))
        rel = delta / float(np.linalg.norm(bos_states[i-1]))
        cos = cos_sim(bos_states[i], bos_states[i-1])
        print(f"  L{i-1:>3}  {delta:>8.2f}  {rel:>8.4f}  {cos:>14.6f}")
    
    # Convergence
    print(f"\n  BOS convergence: cos(h0_at_layer, h0_final)")
    final = bos_states[-1]
    for i in [0, 5, 10, 15, 20, 25, 28]:
        if i < len(bos_states):
            cos = cos_sim(bos_states[i], final)
            label = "emb" if i == 0 else f"L{i-1}"
            print(f"    {label:>5}: {cos:.6f}")
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    stds = []
    for li in SAMPLE_LAYERS:
        vals = [bos_fracs[n][li] for n in FACTS]
        stds.append(np.std(vals))
    mean_std = np.mean(stds)
    
    print(f"\n  Template stability (BOS frac σ across 6 prompts): {mean_std:.4f}")
    if mean_std < 0.05:
        print("  → BOS sink is STABLE across prompts")
    else:
        print("  → BOS sink VARIES across prompts")
    
    print(f"\n  Fixed-template results:")
    print(f"    A: France template L0-L21:   {test_a_match}/6")
    print(f"    B: France template ALL:      {test_b_match}/6")
    print(f"    C: Pure BOS L0-L21:          {test_c_match}/6")
    
    norm_growth = float(np.linalg.norm(bos_states[-1])) / float(np.linalg.norm(bos_states[0]))
    print(f"\n  BOS norm growth: {norm_growth:.2f}x (embedding → final)")
    
    print()


if __name__ == '__main__':
    main()
