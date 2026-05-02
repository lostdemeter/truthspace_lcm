#!/usr/bin/env python3
"""
Verify Exact Match: Compare our computation against model layer-by-layer
=========================================================================

The goal is to achieve 100% token accuracy by ensuring our computation
exactly matches the model at every step.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    print("=" * 70)
    print("EXACT MATCH VERIFICATION")
    print("=" * 70)
    
    # Load model
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    config._attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    device = next(model.parameters()).device
    
    # Test case
    A, B = 3656, 3990  # The mismatch case
    
    print(f"\nTest case: A={A} ('{tokenizer.decode([A])}'), B={B} ('{tokenizer.decode([B])}')")
    
    # Get model's prediction
    ids = torch.tensor([[A, B]]).to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    
    actual_token = torch.argmax(out.logits[0, 1]).item()
    print(f"Model prediction: {actual_token} ('{tokenizer.decode([actual_token])}')")
    
    # Get all hidden states
    hidden_states = out.hidden_states
    print(f"\nHidden states: {len(hidden_states)} (embeddings + 28 layers)")
    
    # Now let's manually compute using the model's weights but in float64
    # and compare at each layer
    
    print("\n--- Layer-by-layer comparison ---")
    
    # Extract weights to float64
    embeddings = model.model.embed_tokens.weight.data.float().cpu().numpy().astype(np.float64)
    final_ln = model.model.norm.weight.data.float().cpu().numpy().astype(np.float64)
    lm_head = model.lm_head.weight.data.float().cpu().numpy().astype(np.float64)
    
    hidden_dim = 3584
    n_heads = 28
    n_kv_heads = 4
    head_dim = 128
    heads_per_kv = 7
    
    # RoPE
    layer0 = model.model.layers[0]
    if hasattr(layer0.self_attn, 'rotary_emb') and hasattr(layer0.self_attn.rotary_emb, 'inv_freq'):
        inv_freq = layer0.self_attn.rotary_emb.inv_freq.float().cpu().numpy().astype(np.float64)
    else:
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    
    def rms_norm(x, weight, eps=1e-6):
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def rope_embed(seq_len):
        freqs = np.outer(np.arange(seq_len, dtype=np.float64), inv_freq)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(x, cos, sin):
        half = len(x) // 2
        x1, x2 = x[:half], x[half:]
        return x * cos + np.concatenate([-x2, x1]) * sin
    
    def silu(x):
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    # Start with embeddings
    h = np.stack([embeddings[A], embeddings[B]])
    cos, sin = rope_embed(2)
    
    # Compare embeddings
    actual_emb = hidden_states[0][0].float().cpu().numpy()
    emb_cos = np.dot(actual_emb[1], h[1]) / (np.linalg.norm(actual_emb[1]) * np.linalg.norm(h[1]))
    print(f"Embeddings: cosine={emb_cos:.6f}")
    
    # Process each layer
    for layer_idx in range(28):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        
        # Extract layer weights
        W_q = attn.q_proj.weight.data.float().cpu().numpy().astype(np.float64)
        W_k = attn.k_proj.weight.data.float().cpu().numpy().astype(np.float64)
        W_v = attn.v_proj.weight.data.float().cpu().numpy().astype(np.float64)
        W_o = attn.o_proj.weight.data.float().cpu().numpy().astype(np.float64)
        b_q = attn.q_proj.bias.data.float().cpu().numpy().astype(np.float64)
        b_k = attn.k_proj.bias.data.float().cpu().numpy().astype(np.float64)
        b_v = attn.v_proj.bias.data.float().cpu().numpy().astype(np.float64)
        ln_attn = layer.input_layernorm.weight.data.float().cpu().numpy().astype(np.float64)
        ln_mlp = layer.post_attention_layernorm.weight.data.float().cpu().numpy().astype(np.float64)
        W_gate = layer.mlp.gate_proj.weight.data.float().cpu().numpy().astype(np.float64)
        W_up = layer.mlp.up_proj.weight.data.float().cpu().numpy().astype(np.float64)
        W_down = layer.mlp.down_proj.weight.data.float().cpu().numpy().astype(np.float64)
        
        W_q_heads = W_q.reshape(n_heads, head_dim, hidden_dim)
        W_k_heads = W_k.reshape(n_kv_heads, head_dim, hidden_dim)
        W_v_heads = W_v.reshape(n_kv_heads, head_dim, hidden_dim)
        b_q_heads = b_q.reshape(n_heads, head_dim)
        b_k_heads = b_k.reshape(n_kv_heads, head_dim)
        b_v_heads = b_v.reshape(n_kv_heads, head_dim)
        
        h_A, h_B = h[0], h[1]
        h_A_n = rms_norm(h_A, ln_attn)
        h_B_n = rms_norm(h_B, ln_attn)
        
        attn_out = np.zeros((2, hidden_dim))
        
        # Position 0
        for head in range(n_heads):
            kv = head // heads_per_kv
            v_A = h_A_n @ W_v_heads[kv].T + b_v_heads[kv]
            attn_out[0, head*head_dim:(head+1)*head_dim] = v_A
        
        # Position 1
        for head in range(n_heads):
            kv = head // heads_per_kv
            
            q_B = h_B_n @ W_q_heads[head].T + b_q_heads[head]
            k_A = h_A_n @ W_k_heads[kv].T + b_k_heads[kv]
            k_B = h_B_n @ W_k_heads[kv].T + b_k_heads[kv]
            v_A = h_A_n @ W_v_heads[kv].T + b_v_heads[kv]
            v_B = h_B_n @ W_v_heads[kv].T + b_v_heads[kv]
            
            q_B_r = apply_rope(q_B, cos[1], sin[1])
            k_A_r = apply_rope(k_A, cos[0], sin[0])
            k_B_r = apply_rope(k_B, cos[1], sin[1])
            
            s_A = np.dot(q_B_r, k_A_r) / np.sqrt(head_dim)
            s_B = np.dot(q_B_r, k_B_r) / np.sqrt(head_dim)
            
            scores = np.array([s_A, s_B])
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            
            attn_out[1, head*head_dim:(head+1)*head_dim] = weights[0]*v_A + weights[1]*v_B
        
        attn_out[0] = attn_out[0] @ W_o.T
        attn_out[1] = attn_out[1] @ W_o.T
        
        h_post = h + attn_out
        
        mlp_out = np.zeros((2, hidden_dim))
        for p in range(2):
            h_n = rms_norm(h_post[p], ln_mlp)
            mlp_out[p] = (silu(h_n @ W_gate.T) * (h_n @ W_up.T)) @ W_down.T
        
        h = h_post + mlp_out
        
        # Compare with actual
        actual_h = hidden_states[layer_idx + 1][0].float().cpu().numpy()
        layer_cos = np.dot(actual_h[1], h[1]) / (np.linalg.norm(actual_h[1]) * np.linalg.norm(h[1]))
        
        if layer_idx < 3 or layer_idx >= 25:
            print(f"Layer {layer_idx:2d}: cosine={layer_cos:.6f}, our_norm={np.linalg.norm(h[1]):.2f}, actual_norm={np.linalg.norm(actual_h[1]):.2f}")
        elif layer_idx == 3:
            print("  ...")
    
    # Final prediction
    h_final = rms_norm(h[1], final_ln)
    logits = lm_head @ h_final
    pred_token = np.argmax(logits)
    
    print(f"\n--- Final Prediction ---")
    print(f"Our prediction: {pred_token} ('{tokenizer.decode([pred_token])}')")
    print(f"Model prediction: {actual_token} ('{tokenizer.decode([actual_token])}')")
    print(f"Match: {pred_token == actual_token}")
    
    # Check logit difference
    print(f"\nLogit for {actual_token}: {logits[actual_token]:.4f}")
    print(f"Logit for {pred_token}: {logits[pred_token]:.4f}")
    print(f"Difference: {logits[pred_token] - logits[actual_token]:.6f}")
    
    # Compare with model's logits
    model_logits = out.logits[0, 1].float().cpu().numpy()
    logits_cos = np.dot(model_logits, logits) / (np.linalg.norm(model_logits) * np.linalg.norm(logits))
    print(f"\nLogits cosine: {logits_cos:.6f}")


if __name__ == "__main__":
    main()
