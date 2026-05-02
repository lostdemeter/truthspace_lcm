#!/usr/bin/env python3
"""
Debug Layer 27: Why does norm explode?
======================================

Layer 26: our norm=303, actual norm=248, cosine=0.999
Layer 27: our norm=507, actual norm=302, cosine=NaN

Something is wrong with layer 27 specifically.
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
    heads_per_kv = n_heads // n_kv_heads
    
    # Test pair
    A, B = 100, 200
    ids = torch.tensor([[A, B]]).to(device)
    
    # Get actual hidden states
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    
    actual_h = [x[0].float().cpu().numpy() for x in out.hidden_states]
    
    # Get input to layer 27 (output of layer 26)
    h26_actual = actual_h[27]  # After layer 26
    h27_actual = actual_h[28]  # After layer 27
    
    print(f"Layer 26 output (input to layer 27):")
    print(f"  Norm: {np.linalg.norm(h26_actual[1]):.4f}")
    
    print(f"\nLayer 27 output (actual):")
    print(f"  Norm: {np.linalg.norm(h27_actual[1]):.4f}")
    
    # Now compute layer 27 manually from actual h26
    layer27 = model.model.layers[27]
    attn = layer27.self_attn
    
    W_q = attn.q_proj.weight.data.float().cpu().numpy()
    W_k = attn.k_proj.weight.data.float().cpu().numpy()
    W_v = attn.v_proj.weight.data.float().cpu().numpy()
    W_o = attn.o_proj.weight.data.float().cpu().numpy()
    b_q = attn.q_proj.bias.data.float().cpu().numpy() if attn.q_proj.bias is not None else None
    b_k = attn.k_proj.bias.data.float().cpu().numpy() if attn.k_proj.bias is not None else None
    b_v = attn.v_proj.bias.data.float().cpu().numpy() if attn.v_proj.bias is not None else None
    ln_attn = layer27.input_layernorm.weight.data.float().cpu().numpy()
    ln_mlp = layer27.post_attention_layernorm.weight.data.float().cpu().numpy()
    W_gate = layer27.mlp.gate_proj.weight.data.float().cpu().numpy()
    W_up = layer27.mlp.up_proj.weight.data.float().cpu().numpy()
    W_down = layer27.mlp.down_proj.weight.data.float().cpu().numpy()
    
    W_q_heads = W_q.reshape(n_heads, head_dim, hidden_dim)
    W_k_heads = W_k.reshape(n_kv_heads, head_dim, hidden_dim)
    W_v_heads = W_v.reshape(n_kv_heads, head_dim, hidden_dim)
    b_q_heads = b_q.reshape(n_heads, head_dim) if b_q is not None else None
    b_k_heads = b_k.reshape(n_kv_heads, head_dim) if b_k is not None else None
    b_v_heads = b_v.reshape(n_kv_heads, head_dim) if b_v is not None else None
    
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
    
    cos, sin = rope_embed(2)
    
    # Use ACTUAL h26 as input
    h = h26_actual.copy()
    h_A, h_B = h[0], h[1]
    
    h_A_n = rms_norm(h_A, ln_attn)
    h_B_n = rms_norm(h_B, ln_attn)
    
    print(f"\nLayer norm output:")
    print(f"  h_A_n norm: {np.linalg.norm(h_A_n):.4f}")
    print(f"  h_B_n norm: {np.linalg.norm(h_B_n):.4f}")
    
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
    
    print(f"\nAttention output:")
    print(f"  attn_out[1] norm: {np.linalg.norm(attn_out[1]):.4f}")
    
    h_post = h + attn_out
    
    print(f"\nPost-attention (h + attn_out):")
    print(f"  h_post[1] norm: {np.linalg.norm(h_post[1]):.4f}")
    
    # MLP
    h_n = rms_norm(h_post[1], ln_mlp)
    gate = h_n @ W_gate.T
    up = h_n @ W_up.T
    mlp_out = (silu(gate) * up) @ W_down.T
    
    print(f"\nMLP:")
    print(f"  gate norm: {np.linalg.norm(gate):.4f}")
    print(f"  up norm: {np.linalg.norm(up):.4f}")
    print(f"  mlp_out norm: {np.linalg.norm(mlp_out):.4f}")
    
    h_final = h_post[1] + mlp_out
    
    print(f"\nFinal (h_post + mlp_out):")
    print(f"  h_final norm: {np.linalg.norm(h_final):.4f}")
    print(f"  Actual h27 norm: {np.linalg.norm(h27_actual[1]):.4f}")
    
    # Compare
    cos_sim = np.dot(h_final, h27_actual[1]) / (np.linalg.norm(h_final) * np.linalg.norm(h27_actual[1]) + 1e-10)
    print(f"  Cosine: {cos_sim:.6f}")
    
    # Now let's capture the actual intermediate values
    print(f"\n--- Capturing actual layer 27 intermediates ---")
    
    captured = {}
    
    def capture_ln(module, input, output):
        captured['ln_input'] = input[0].detach().float().cpu().numpy()
        captured['ln_output'] = output.detach().float().cpu().numpy()
    
    def capture_attn(module, input, output):
        captured['attn_output'] = output[0].detach().float().cpu().numpy()
    
    def capture_mlp(module, input, output):
        captured['mlp_input'] = input[0].detach().float().cpu().numpy()
        captured['mlp_output'] = output.detach().float().cpu().numpy()
    
    h1 = layer27.input_layernorm.register_forward_hook(capture_ln)
    h2 = layer27.self_attn.register_forward_hook(capture_attn)
    h3 = layer27.mlp.register_forward_hook(capture_mlp)
    
    with torch.no_grad():
        model(ids, output_hidden_states=True)
    
    h1.remove()
    h2.remove()
    h3.remove()
    
    print(f"Actual layer norm output norm: {np.linalg.norm(captured['ln_output'][0, 1]):.4f}")
    print(f"Our layer norm output norm: {np.linalg.norm(h_B_n):.4f}")
    
    ln_cos = np.dot(captured['ln_output'][0, 1], h_B_n) / (
        np.linalg.norm(captured['ln_output'][0, 1]) * np.linalg.norm(h_B_n) + 1e-10)
    print(f"Layer norm cosine: {ln_cos:.6f}")
    
    print(f"\nActual attention output norm: {np.linalg.norm(captured['attn_output'][0, 1]):.4f}")
    print(f"Our attention output norm: {np.linalg.norm(attn_out[1]):.4f}")
    
    attn_cos = np.dot(captured['attn_output'][0, 1], attn_out[1]) / (
        np.linalg.norm(captured['attn_output'][0, 1]) * np.linalg.norm(attn_out[1]) + 1e-10)
    print(f"Attention output cosine: {attn_cos:.6f}")
    
    print(f"\nActual MLP output norm: {np.linalg.norm(captured['mlp_output'][0, 1]):.4f}")
    print(f"Our MLP output norm: {np.linalg.norm(mlp_out):.4f}")
    
    mlp_cos = np.dot(captured['mlp_output'][0, 1], mlp_out) / (
        np.linalg.norm(captured['mlp_output'][0, 1]) * np.linalg.norm(mlp_out) + 1e-10)
    print(f"MLP output cosine: {mlp_cos:.6f}")


if __name__ == "__main__":
    main()
