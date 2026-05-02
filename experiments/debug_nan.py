#!/usr/bin/env python3
"""
Debug NaN: Find where the computation goes wrong
=================================================
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
warnings.filterwarnings('ignore')


def main():
    print("Loading model...")
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    config._attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    device = next(model.parameters()).device
    
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = hidden_dim // n_heads
    n_layers = model.config.num_hidden_layers
    heads_per_kv = n_heads // n_kv_heads
    
    embeddings = model.model.embed_tokens.weight.data.float().cpu().numpy()
    final_ln = model.model.norm.weight.data.float().cpu().numpy()
    lm_head = model.lm_head.weight.data.float().cpu().numpy()
    
    # RoPE
    layer0 = model.model.layers[0]
    if hasattr(layer0.self_attn, 'rotary_emb') and hasattr(layer0.self_attn.rotary_emb, 'inv_freq'):
        inv_freq = layer0.self_attn.rotary_emb.inv_freq.float().cpu().numpy()
    else:
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    
    def rope_embed(seq_len):
        freqs = np.outer(np.arange(seq_len), inv_freq)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(x, cos, sin):
        x1, x2 = x[:head_dim//2], x[head_dim//2:]
        return x * cos + np.concatenate([-x2, x1]) * sin
    
    def rms_norm(x, w, eps=1e-6):
        return (x / np.sqrt(np.mean(x**2) + eps)) * w
    
    def silu(x):
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    # Test pair
    A, B = 100, 200
    
    # Get actual hidden states
    ids = torch.tensor([[A, B]]).to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    
    actual_h = [x[0].float().cpu().numpy() for x in out.hidden_states]
    
    print(f"\nActual hidden states:")
    for i, h in enumerate(actual_h):
        print(f"  Layer {i}: norm={np.linalg.norm(h[1]):.4f}, has_nan={np.isnan(h).any()}")
    
    # Now compute layer by layer and compare
    print(f"\n--- Computing layer by layer ---")
    
    h = np.stack([embeddings[A], embeddings[B]])
    cos, sin = rope_embed(2)
    
    print(f"Embeddings: norm={np.linalg.norm(h[1]):.4f}, has_nan={np.isnan(h).any()}")
    
    # Compare with actual
    cos_emb = np.dot(actual_h[0][1], h[1]) / (np.linalg.norm(actual_h[0][1]) * np.linalg.norm(h[1]) + 1e-10)
    print(f"  vs actual: cosine={cos_emb:.6f}")
    
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        
        # Extract weights
        W_q = attn.q_proj.weight.data.float().cpu().numpy()
        W_k = attn.k_proj.weight.data.float().cpu().numpy()
        W_v = attn.v_proj.weight.data.float().cpu().numpy()
        W_o = attn.o_proj.weight.data.float().cpu().numpy()
        b_q = attn.q_proj.bias.data.float().cpu().numpy() if attn.q_proj.bias is not None else None
        b_k = attn.k_proj.bias.data.float().cpu().numpy() if attn.k_proj.bias is not None else None
        b_v = attn.v_proj.bias.data.float().cpu().numpy() if attn.v_proj.bias is not None else None
        ln_attn = layer.input_layernorm.weight.data.float().cpu().numpy()
        ln_mlp = layer.post_attention_layernorm.weight.data.float().cpu().numpy()
        W_gate = layer.mlp.gate_proj.weight.data.float().cpu().numpy()
        W_up = layer.mlp.up_proj.weight.data.float().cpu().numpy()
        W_down = layer.mlp.down_proj.weight.data.float().cpu().numpy()
        
        W_q_heads = W_q.reshape(n_heads, head_dim, hidden_dim)
        W_k_heads = W_k.reshape(n_kv_heads, head_dim, hidden_dim)
        W_v_heads = W_v.reshape(n_kv_heads, head_dim, hidden_dim)
        b_q_heads = b_q.reshape(n_heads, head_dim) if b_q is not None else None
        b_k_heads = b_k.reshape(n_kv_heads, head_dim) if b_k is not None else None
        b_v_heads = b_v.reshape(n_kv_heads, head_dim) if b_v is not None else None
        
        h_A, h_B = h[0], h[1]
        h_A_n = rms_norm(h_A, ln_attn)
        h_B_n = rms_norm(h_B, ln_attn)
        
        attn_out = np.zeros((2, hidden_dim))
        
        # Position 0
        for hd in range(n_heads):
            kv = hd // heads_per_kv
            v_A = h_A_n @ W_v_heads[kv].T
            if b_v is not None:
                v_A += b_v_heads[kv]
            attn_out[0, hd*head_dim:(hd+1)*head_dim] = v_A
        
        # Position 1
        for hd in range(n_heads):
            kv = hd // heads_per_kv
            
            q_B = h_B_n @ W_q_heads[hd].T
            k_A = h_A_n @ W_k_heads[kv].T
            k_B = h_B_n @ W_k_heads[kv].T
            v_A = h_A_n @ W_v_heads[kv].T
            v_B = h_B_n @ W_v_heads[kv].T
            
            if b_q is not None:
                q_B += b_q_heads[hd]
            if b_k is not None:
                k_A += b_k_heads[kv]
                k_B += b_k_heads[kv]
            if b_v is not None:
                v_A += b_v_heads[kv]
                v_B += b_v_heads[kv]
            
            q_B_r = apply_rope(q_B, cos[1], sin[1])
            k_A_r = apply_rope(k_A, cos[0], sin[0])
            k_B_r = apply_rope(k_B, cos[1], sin[1])
            
            s_A = np.dot(q_B_r, k_A_r) / np.sqrt(head_dim)
            s_B = np.dot(q_B_r, k_B_r) / np.sqrt(head_dim)
            
            scores = np.array([s_A, s_B])
            attn = np.exp(scores - scores.max())
            attn = attn / attn.sum()
            
            attn_out[1, hd*head_dim:(hd+1)*head_dim] = attn[0]*v_A + attn[1]*v_B
        
        attn_out[0] = attn_out[0] @ W_o.T
        attn_out[1] = attn_out[1] @ W_o.T
        
        h_post = h + attn_out
        
        mlp_out = np.zeros((2, hidden_dim))
        for p in range(2):
            h_n = rms_norm(h_post[p], ln_mlp)
            mlp_out[p] = (silu(h_n @ W_gate.T) * (h_n @ W_up.T)) @ W_down.T
        
        h = h_post + mlp_out
        
        # Check for issues
        has_nan = np.isnan(h).any()
        has_inf = np.isinf(h).any()
        norm = np.linalg.norm(h[1])
        
        # Compare with actual
        actual = actual_h[layer_idx + 1][1]
        cos_sim = np.dot(actual, h[1]) / (np.linalg.norm(actual) * norm + 1e-10)
        
        if layer_idx < 5 or layer_idx >= n_layers - 3 or has_nan or has_inf:
            print(f"Layer {layer_idx:2d}: norm={norm:.4f}, nan={has_nan}, inf={has_inf}, cos={cos_sim:.6f}")
    
    # Final norm
    h_final = np.stack([rms_norm(h[0], final_ln), rms_norm(h[1], final_ln)])
    
    print(f"\nFinal (after norm): norm={np.linalg.norm(h_final[1]):.4f}")
    
    # Token prediction
    logits = lm_head @ h_final[1]
    pred_token = np.argmax(logits)
    actual_token = torch.argmax(out.logits[0, 1]).item()
    
    print(f"\nToken prediction:")
    print(f"  Actual: {actual_token} = '{tokenizer.decode([actual_token])}'")
    print(f"  Pred:   {pred_token} = '{tokenizer.decode([pred_token])}'")
    print(f"  Match:  {actual_token == pred_token}")


if __name__ == "__main__":
    main()
