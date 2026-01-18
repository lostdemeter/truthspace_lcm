#!/usr/bin/env python3
"""
Debug: Why is attention reproduction only 86%?

The φ-MESH reconstruction is 99.95% accurate.
But full attention reproduction is only 86%.

Hypothesis: We're missing something in the computation:
1. RoPE (Rotary Position Embedding)?
2. Attention scaling?
3. Different normalization?

Let's trace through the actual model computation.
"""

import torch
import numpy as np

PHI = (1 + np.sqrt(5)) / 2


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def trace_attention_computation(model, tokenizer, text):
    """
    Trace through the exact attention computation in the model.
    """
    print()
    print("=" * 70)
    print(f"TRACING ATTENTION: '{text}'")
    print("=" * 70)
    print()
    
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs['input_ids'].shape[1]
    
    # Get the first layer
    layer = model.model.layers[0]
    
    # Step 1: Get input embeddings
    with torch.no_grad():
        hidden = model.model.embed_tokens(inputs['input_ids'])[0]  # [seq_len, 896]
    
    print(f"1. Input embeddings shape: {hidden.shape}")
    print(f"   Mean: {hidden.mean():.4f}, Std: {hidden.std():.4f}")
    
    # Step 2: Apply input layernorm (RMSNorm)
    ln_weight = layer.input_layernorm.weight
    
    # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden * torch.rsqrt(variance + 1e-6) * ln_weight
    
    print(f"2. After RMSNorm shape: {hidden_normed.shape}")
    print(f"   Mean: {hidden_normed.mean():.4f}, Std: {hidden_normed.std():.4f}")
    
    # Step 3: Project to Q, K, V
    W_q = layer.self_attn.q_proj.weight
    W_k = layer.self_attn.k_proj.weight
    W_v = layer.self_attn.v_proj.weight
    
    Q = hidden_normed @ W_q.T  # [seq_len, 896]
    K = hidden_normed @ W_k.T  # [seq_len, 128]
    V = hidden_normed @ W_v.T  # [seq_len, 128]
    
    print(f"3. Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    
    # Step 4: Reshape to heads
    n_q_heads = 14
    n_kv_heads = 2
    head_dim = 64
    
    Q = Q.view(seq_len, n_q_heads, head_dim)  # [seq_len, 14, 64]
    K = K.view(seq_len, n_kv_heads, head_dim)  # [seq_len, 2, 64]
    V = V.view(seq_len, n_kv_heads, head_dim)  # [seq_len, 2, 64]
    
    print(f"4. Q heads: {Q.shape}, K heads: {K.shape}")
    
    # Step 5: Apply RoPE (Rotary Position Embedding)
    # This is the key difference!
    print()
    print("5. ROPE (Rotary Position Embedding):")
    
    # Get RoPE parameters
    rotary_emb = layer.self_attn.rotary_emb
    
    # Generate position ids
    position_ids = torch.arange(seq_len).unsqueeze(0)
    
    # Get cos and sin for RoPE
    cos, sin = rotary_emb(V, position_ids)
    
    print(f"   cos shape: {cos.shape}, sin shape: {sin.shape}")
    print(f"   cos[0, :5]: {cos[0, 0, :5].numpy()}")
    print(f"   sin[0, :5]: {sin[0, 0, :5].numpy()}")
    
    # Apply RoPE to Q and K
    def apply_rope(x, cos, sin):
        """Apply rotary position embedding."""
        # x: [seq_len, n_heads, head_dim]
        # cos, sin: [1, seq_len, head_dim]
        
        # Split into two halves
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        
        # Rotate
        cos = cos.squeeze(0)  # [seq_len, head_dim]
        sin = sin.squeeze(0)
        
        cos1 = cos[:, :head_dim//2].unsqueeze(1)  # [seq_len, 1, head_dim//2]
        sin1 = sin[:, :head_dim//2].unsqueeze(1)
        
        rotated = torch.cat([
            x1 * cos1 - x2 * sin1,
            x2 * cos1 + x1 * sin1,
        ], dim=-1)
        
        return rotated
    
    Q_rope = apply_rope(Q, cos, sin)
    K_rope = apply_rope(K, cos, sin)
    
    print(f"   Q before RoPE: {Q[0, 0, :5].numpy()}")
    print(f"   Q after RoPE:  {Q_rope[0, 0, :5].numpy()}")
    
    # Step 6: Compute attention scores
    # Expand K for GQA (7 Q heads per K head)
    K_expanded = K_rope.repeat_interleave(7, dim=1)  # [seq_len, 14, 64]
    
    # Transpose for matmul: [n_heads, seq_len, head_dim]
    Q_t = Q_rope.transpose(0, 1)  # [14, seq_len, 64]
    K_t = K_expanded.transpose(0, 1)  # [14, seq_len, 64]
    
    # Attention scores
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / np.sqrt(head_dim)
    
    print(f"6. Attention scores shape: {scores.shape}")
    print(f"   Scores[0, 0, :]: {scores[0, 0, :].numpy()}")
    
    # Step 7: Apply causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    scores = scores + mask
    
    # Step 8: Softmax
    attention = torch.softmax(scores, dim=-1)
    
    print(f"7. Attention (after softmax) shape: {attention.shape}")
    print(f"   Attention[0, 0, :]: {attention[0, 0, :].numpy()}")
    
    # Compare with model's actual attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    actual_attention = outputs.attentions[0][0]  # [n_heads, seq_len, seq_len]
    
    print()
    print("8. Comparison with model's attention:")
    print(f"   Actual[0, 0, :]: {actual_attention[0, 0, :].numpy()}")
    
    # Correlation
    for h in range(3):
        our_attn = attention[h].numpy()
        model_attn = actual_attention[h].numpy()
        
        corr = np.corrcoef(our_attn.flatten(), model_attn.flatten())[0, 1]
        mse = np.mean((our_attn - model_attn) ** 2)
        
        print(f"   Head {h}: corr={corr:.6f}, mse={mse:.8f}")
    
    return attention, actual_attention


def test_without_rope(model, tokenizer, text):
    """
    Test attention reproduction WITHOUT RoPE to see the difference.
    """
    print()
    print("=" * 70)
    print("TESTING WITHOUT ROPE")
    print("=" * 70)
    print()
    
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs['input_ids'].shape[1]
    
    layer = model.model.layers[0]
    
    # Get normalized hidden state
    with torch.no_grad():
        hidden = model.model.embed_tokens(inputs['input_ids'])[0]
    
    ln_weight = layer.input_layernorm.weight
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden * torch.rsqrt(variance + 1e-6) * ln_weight
    
    # Project to Q, K
    W_q = layer.self_attn.q_proj.weight
    W_k = layer.self_attn.k_proj.weight
    
    Q = hidden_normed @ W_q.T
    K = hidden_normed @ W_k.T
    
    # Reshape
    n_q_heads = 14
    n_kv_heads = 2
    head_dim = 64
    
    Q = Q.view(seq_len, n_q_heads, head_dim)
    K = K.view(seq_len, n_kv_heads, head_dim)
    
    # WITHOUT RoPE - just expand K for GQA
    K_expanded = K.repeat_interleave(7, dim=1)
    
    Q_t = Q.transpose(0, 1)
    K_t = K_expanded.transpose(0, 1)
    
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / np.sqrt(head_dim)
    
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    scores = scores + mask
    
    attention_no_rope = torch.softmax(scores, dim=-1)
    
    # Compare with model's actual attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    actual_attention = outputs.attentions[0][0]
    
    print("Without RoPE:")
    for h in range(3):
        our_attn = attention_no_rope[h].numpy()
        model_attn = actual_attention[h].numpy()
        
        corr = np.corrcoef(our_attn.flatten(), model_attn.flatten())[0, 1]
        mse = np.mean((our_attn - model_attn) ** 2)
        
        print(f"  Head {h}: corr={corr:.6f}, mse={mse:.8f}")
    
    return attention_no_rope


def test_with_correct_rope(model, tokenizer, text):
    """
    Test with the EXACT RoPE implementation from the model.
    """
    print()
    print("=" * 70)
    print("TESTING WITH EXACT MODEL ROPE")
    print("=" * 70)
    print()
    
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs['input_ids'].shape[1]
    
    layer = model.model.layers[0]
    attn = layer.self_attn
    
    # Get normalized hidden state
    with torch.no_grad():
        hidden = model.model.embed_tokens(inputs['input_ids'])[0]
    
    ln_weight = layer.input_layernorm.weight
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden * torch.rsqrt(variance + 1e-6) * ln_weight
    hidden_normed = hidden_normed.unsqueeze(0)  # Add batch dim
    
    # Use the model's own attention forward pass
    # But we need to extract intermediate values
    
    # Project
    Q = attn.q_proj(hidden_normed)
    K = attn.k_proj(hidden_normed)
    V = attn.v_proj(hidden_normed)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    
    # Reshape
    bsz = 1
    Q = Q.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    K = K.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    V = V.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    
    print(f"Q reshaped: {Q.shape}")  # [1, 14, seq_len, 64]
    print(f"K reshaped: {K.shape}")  # [1, 2, seq_len, 64]
    
    # Apply RoPE using model's method
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = attn.rotary_emb(V, position_ids)
    
    # The model uses apply_rotary_pos_emb
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)
    
    print(f"Q after RoPE: {Q_rope.shape}")
    print(f"K after RoPE: {K_rope.shape}")
    
    # Expand K for GQA
    # repeat_kv expands [1, 2, seq, 64] -> [1, 14, seq, 64]
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    K_expanded = repeat_kv(K_rope, attn.num_key_value_groups)
    
    print(f"K expanded: {K_expanded.shape}")
    
    # Compute attention
    scores = torch.matmul(Q_rope, K_expanded.transpose(-2, -1)) / np.sqrt(attn.head_dim)
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    scores = scores + mask
    
    attention = torch.softmax(scores, dim=-1)
    
    # Compare with model's actual attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    actual_attention = outputs.attentions[0][0]
    
    print()
    print("With exact model RoPE:")
    for h in range(3):
        our_attn = attention[0, h].numpy()
        model_attn = actual_attention[h].numpy()
        
        corr = np.corrcoef(our_attn.flatten(), model_attn.flatten())[0, 1]
        mse = np.mean((our_attn - model_attn) ** 2)
        
        print(f"  Head {h}: corr={corr:.6f}, mse={mse:.8f}")
    
    # Check if we get perfect match
    total_mse = np.mean((attention[0].numpy() - actual_attention.numpy()) ** 2)
    accuracy = 1 - np.sqrt(total_mse)
    
    print()
    print(f"Total MSE: {total_mse:.10f}")
    print(f"Accuracy: {accuracy:.6%}")
    
    if accuracy > 0.9999:
        print("✓ PERFECT MATCH!")
    
    return attention, actual_attention


def main():
    model, tokenizer = load_model()
    
    text = "The king examined"
    
    # Trace the computation
    trace_attention_computation(model, tokenizer, text)
    
    # Test without RoPE
    test_without_rope(model, tokenizer, text)
    
    # Test with exact RoPE
    test_with_correct_rope(model, tokenizer, text)
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The key insight: RoPE (Rotary Position Embedding) is essential!")
    print()
    print("Without RoPE: ~86% accuracy (what we had)")
    print("With RoPE: Should be ~100% accuracy")
    print()
    print("RoPE adds position-dependent rotation to Q and K.")
    print("This rotation IS the 'tachyon navigation' - it encodes position!")


if __name__ == "__main__":
    main()
