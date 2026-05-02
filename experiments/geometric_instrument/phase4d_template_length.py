"""
Phase 4d: Template Length Generalization
=========================================

F132 proved fixed-template attention works for 5-token prompts.
The critical question: does this generalize to other lengths?

Investigations:
  1. Extract templates at different sequence lengths (3, 5, 7, 9+ tokens)
  2. Measure BOS-sink pattern vs length — is there a geometric rule?
  3. Cross-length transfer — can a 5-token template work for 7-token input?
  4. Relative position analysis — do templates follow a positional rule?
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

# Prompts of varying lengths, all asking for capitals
PROMPTS = {
    'France':  [
        ('3tok', 'France is'),
        ('4tok', 'France capital is'),
        ('5tok', 'The capital of France is'),
        ('7tok', 'I know the capital of France is'),
        ('9tok', 'Can you tell me the capital of France is'),
    ],
    'Germany': [
        ('3tok', 'Germany is'),
        ('4tok', 'Germany capital is'),
        ('5tok', 'The capital of Germany is'),
        ('7tok', 'I know the capital of Germany is'),
        ('9tok', 'Can you tell me the capital of Germany is'),
    ],
}

ANSWERS = {
    'France': ' Paris',
    'Germany': ' Berlin',
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
        # Pad: distribute extra positions' weight to p0 (BOS sink)
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


def run_with_template(engine, tids, templates, n_layers):
    """Full forward pass using fixed templates at all layers."""
    h = engine.embedding(tids)[np.newaxis, :, :]
    for li in range(n_layers):
        h = run_layer_with_fixed_attention(engine, h, li, templates[li])
    normed = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(normed)[0, -1, :]


def main():
    print("=" * 80)
    print("  Phase 4d: Template Length Generalization")
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
    # Step 1: Verify tokenization and baseline accuracy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 1: Tokenization & Baseline")
    print("=" * 80)
    
    all_templates = {}  # (country, label) -> [list of [nh, seq] per layer]
    
    for country in PROMPTS:
        for label, prompt in PROMPTS[country]:
            tids = tokenizer.encode(prompt)
            tokens = [tokenizer.decode([t]) for t in tids]
            actual_len = len(tids)
            
            # Baseline
            h = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                h = engine.layers[li](h)
            normed = rms_norm(h, engine.final_norm_weight)
            logits = engine.lm_head(normed)[0, -1, :]
            top_tok = tokenizer.decode([int(np.argmax(logits))])
            rank, _ = get_rank(logits, ANSWERS[country], tokenizer)
            mark = "✓" if rank == 0 else f"rank={rank}"
            
            print(f"  {country:>8} {label}: [{actual_len} tok] "
                  f"{tokens} → '{top_tok}' {mark}")
    
    # ═══════════════════════════════════════════════════════════
    # Step 2: Extract templates at each length
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 2: BOS-Sink Pattern vs Sequence Length")
    print("=" * 80)
    
    print(f"\n  BOS fraction (avg over 28 heads) at sampled layers:")
    print(f"  {'Country':>8} {'Label':>5} {'Len':>3}  " +
          "  ".join(f"L{i:>2}" for i in SAMPLE_LAYERS) +
          "  subject  last")
    print("  " + "─" * 90)
    
    for country in PROMPTS:
        for label, prompt in PROMPTS[country]:
            tids = tokenizer.encode(prompt)
            seq_len = len(tids)
            
            h = engine.embedding(tids)[np.newaxis, :, :]
            templates = []
            
            bos_strs = []
            subj_val = last_val = 0.0
            
            for li in range(n_layers):
                if li in SAMPLE_LAYERS:
                    w_lt = get_last_token_attention(engine, h, li)
                    bos_frac = float(w_lt[:, 0].mean())
                    bos_strs.append(f"{bos_frac:.3f}")
                    if li == 23:
                        subj_val = float(w_lt[:, -2].mean())
                        last_val = float(w_lt[:, -1].mean())
                    templates.append(w_lt.copy())
                else:
                    templates.append(None)
                h = engine.layers[li](h)
            
            # Store full templates (need to re-extract for non-sampled layers)
            all_templates[(country, label)] = {
                'tids': tids,
                'seq_len': seq_len,
                'sampled_templates': templates,
            }
            
            print(f"  {country:>8} {label:>5} {seq_len:>3}  " +
                  "  ".join(f"{s:>5}" for s in bos_strs) +
                  f"  {subj_val:.3f}   {last_val:.3f}")
    
    # ═══════════════════════════════════════════════════════════
    # Step 3: Extract FULL templates for France 5tok
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 3: Self-Template Test (same-length)")
    print("  Extract template from one prompt, test on same-length prompts")
    print("=" * 80)
    
    # Extract full templates for France 5tok
    france_5_tids = tokenizer.encode('The capital of France is')
    france_5_seq = len(france_5_tids)
    
    print(f"\n  Extracting full templates from France 5tok ({france_5_seq} tokens)...", end="", flush=True)
    t0 = time.time()
    h = engine.embedding(france_5_tids)[np.newaxis, :, :]
    france_5_templates = []
    for li in range(n_layers):
        w_lt = get_last_token_attention(engine, h, li)
        france_5_templates.append(w_lt.copy())
        h = engine.layers[li](h)
    print(f" done in {time.time()-t0:.1f}s")
    
    # Test: France 5tok template applied to Germany 5tok
    germany_5_tids = tokenizer.encode('The capital of Germany is')
    logits = run_with_template(engine, germany_5_tids, france_5_templates, n_layers)
    top_tok = tokenizer.decode([int(np.argmax(logits))])
    rank, _ = get_rank(logits, ' Berlin', tokenizer)
    print(f"\n  France_5 template → Germany_5: top='{top_tok}', rank={rank} {'✓' if rank==0 else '✗'}")
    
    # ═══════════════════════════════════════════════════════════
    # Step 4: Cross-length transfer tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 4: Cross-Length Transfer")
    print("  Can a template from one length work for a different length?")
    print("=" * 80)
    
    # Strategy A: Pad with zeros (extra positions get 0 weight, renormalize)
    # Strategy B: Place template positions relative to end (last, second-to-last, etc.)
    # Strategy C: Uniform padding (extra positions get equal share of remaining weight)
    
    # First extract templates at each available France length
    france_templates_by_len = {}
    
    for label, prompt in PROMPTS['France']:
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        
        print(f"\n  Extracting {label} ({seq_len} tokens)...", end="", flush=True)
        t0 = time.time()
        h = engine.embedding(tids)[np.newaxis, :, :]
        templates = []
        for li in range(n_layers):
            w_lt = get_last_token_attention(engine, h, li)
            templates.append(w_lt.copy())
            h = engine.layers[li](h)
        france_templates_by_len[seq_len] = templates
        print(f" done in {time.time()-t0:.1f}s")
    
    # Now test cross-length: apply each template to Germany at each length
    print(f"\n  Cross-length transfer (rows=template source, cols=target prompt):")
    
    germany_prompts = {label: tokenizer.encode(prompt) 
                       for label, prompt in PROMPTS['Germany']}
    
    # Header
    target_labels = [f"{l}({len(t)})" for l, t in germany_prompts.items()]
    print(f"  {'Template':>15}  " + "  ".join(f"{tl:>12}" for tl in target_labels))
    print("  " + "─" * (18 + 14 * len(target_labels)))
    
    for src_label, src_prompt in PROMPTS['France']:
        src_tids = tokenizer.encode(src_prompt)
        src_len = len(src_tids)
        templates = france_templates_by_len[src_len]
        
        results = []
        for tgt_label, tgt_tids in germany_prompts.items():
            logits = run_with_template(engine, tgt_tids, templates, n_layers)
            rank, _ = get_rank(logits, ' Berlin', tokenizer)
            top_tok = tokenizer.decode([int(np.argmax(logits))])
            mark = "✓" if rank == 0 else f"r={rank}"
            results.append(f"{mark:>5}({top_tok[:6]})")
        
        print(f"  Fr_{src_label:>10}  " + "  ".join(f"{r:>12}" for r in results))
    
    # ═══════════════════════════════════════════════════════════
    # Step 5: Relative position analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 5: Relative Position Analysis")
    print("  How does the attention template scale with position?")
    print("=" * 80)
    
    # At L23 (the key extraction layer), compare attention profiles:
    # - Position 0 (BOS/first): always high
    # - Position -2 (subject): the country token
    # - Position -1 (last): "is"
    # - Middle positions: everything else
    
    print(f"\n  L23 attention profile by relative position:")
    print(f"  {'Prompt':>35}  {'p0':>6}  {'mid_avg':>7}  {'p[-2]':>6}  {'p[-1]':>6}  {'sum':>5}")
    print("  " + "─" * 70)
    
    for country in PROMPTS:
        for label, prompt in PROMPTS[country]:
            tids = tokenizer.encode(prompt)
            seq_len = len(tids)
            
            h = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                if li == 23:
                    w_lt = get_last_token_attention(engine, h, li)
                    w_avg = w_lt.mean(axis=0)  # [seq_len]
                    
                    p0 = float(w_avg[0])
                    p_last = float(w_avg[-1])
                    p_subj = float(w_avg[-2])
                    mid = w_avg[1:-2]
                    mid_avg = float(mid.mean()) if len(mid) > 0 else 0.0
                    total = float(w_avg.sum())
                    
                    print(f"  {prompt:>35}  {p0:.4f}  {mid_avg:.5f}  {p_subj:.4f}  {p_last:.4f}  {total:.3f}")
                h = engine.layers[li](h)
    
    # Per-head analysis at L23 for different lengths
    print(f"\n  L23 per-head BOS fraction at different lengths (France only):")
    print(f"  {'Len':>3}  " + "  ".join(f"H{i:>2}" for i in range(nh)))
    print("  " + "─" * (6 + 5 * nh))
    
    for label, prompt in PROMPTS['France']:
        tids = tokenizer.encode(prompt)
        seq_len = len(tids)
        
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            if li == 23:
                w_lt = get_last_token_attention(engine, h, li)
                bos_per_head = w_lt[:, 0]  # [nh]
                head_str = "  ".join(f"{v:.2f}" for v in bos_per_head)
                print(f"  {seq_len:>3}  {head_str}")
            h = engine.layers[li](h)
    
    # ═══════════════════════════════════════════════════════════
    # Step 6: Right-aligned transfer test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 6: Right-Aligned Transfer")
    print("  Align template from END (last, subject, ..., BOS, zero-pad left)")
    print("=" * 80)
    
    def right_align_template(template, target_len):
        """Align template from the right side (end positions match)."""
        nh_t, src_len = template.shape
        if target_len == src_len:
            return template.copy()
        
        result = np.zeros((nh_t, target_len), dtype=np.float32)
        if target_len > src_len:
            # Target longer: right-align, pad left with zeros
            result[:, target_len - src_len:] = template
        else:
            # Target shorter: take rightmost positions
            result = template[:, src_len - target_len:]
        
        # Renormalize
        sums = result.sum(axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-12)
        return result / sums
    
    # Test right-aligned cross-length transfer
    print(f"\n  Right-aligned transfer (France template → Germany target):")
    
    for src_label, src_prompt in PROMPTS['France']:
        src_tids = tokenizer.encode(src_prompt)
        src_len = len(src_tids)
        templates = france_templates_by_len[src_len]
        
        for tgt_label, tgt_prompt in PROMPTS['Germany']:
            tgt_tids = tokenizer.encode(tgt_prompt)
            tgt_len = len(tgt_tids)
            
            if src_len == tgt_len:
                continue  # Already tested in Step 4
            
            # Right-align templates
            ra_templates = []
            for li in range(n_layers):
                ra_templates.append(right_align_template(templates[li], tgt_len))
            
            logits = run_with_template(engine, tgt_tids, ra_templates, n_layers)
            rank, _ = get_rank(logits, ' Berlin', tokenizer)
            top_tok = tokenizer.decode([int(np.argmax(logits))])
            mark = "✓" if rank == 0 else f"rank={rank}"
            
            print(f"    Fr_{src_label}({src_len}) → Ger_{tgt_label}({tgt_len}): "
                  f"top='{top_tok}' {mark}")
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    print()


if __name__ == '__main__':
    main()
