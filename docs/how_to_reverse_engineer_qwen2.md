# How to Reverse Engineer Qwen2-7B: A Guide for AI Agents

**Purpose:** Self-contained guide for AI agents to reproduce Qwen2-7B-Instruct computations using explicit matrix operations.

**Key Result:** **100% token prediction accuracy** by manually computing all 28 layers.

---

## 1. Model Architecture

| Parameter | Value |
|-----------|-------|
| hidden_dim | 3584 |
| n_layers | 28 |
| n_heads | 28 |
| n_kv_heads | 4 (GQA) |
| head_dim | 128 |
| vocab_size | 152064 |

**Critical:** Qwen2 has **biases on Q, K, V projections** (unlike many transformers).

---

## 2. Core Math Operations

### RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

### RoPE
$$\text{RoPE}(x, pos) = x \odot \cos(\theta_{pos}) + \text{rotate}(x) \odot \sin(\theta_{pos})$$

Where rotate swaps halves: $[x_1, x_2] \to [-x_2, x_1]$

### Attention
$$\text{Attention}(q, K, V) = \text{softmax}\left(\frac{qK^T}{\sqrt{d_k}}\right) V$$

### Gated MLP
$$\text{MLP}(x) = W_{down} \cdot (\text{SiLU}(W_{gate} x) \odot W_{up} x)$$

---

## 3. Layer Computation

Each of 28 layers:
1. RMSNorm input
2. Q, K, V projections **with bias**
3. RoPE on Q and K
4. Attention scores + softmax
5. Output projection
6. Residual add
7. RMSNorm
8. Gated MLP (SiLU)
9. Residual add

---

## 4. GQA Indexing

```python
heads_per_kv = 28 // 4 = 7
kv_idx = head // heads_per_kv
```

---

## 5. Forward Pass

```python
def forward(token_A, token_B):
    h = stack([embeddings[token_A], embeddings[token_B]])
    cos, sin = rope_embed(2)
    
    for layer in range(28):
        h = compute_layer(layer, h, cos, sin)
    
    h_final = rms_norm(h[1], final_ln)
    logits = lm_head @ h_final
    return argmax(logits)
```

---

## 6. Validation Results

| Metric | Value |
|--------|-------|
| Per-layer cosine | 0.997-0.999 |
| Final cosine | ~0.97 |
| Token accuracy | **100%** |

---

## 7. Critical Issues

### Float16 Overflow
Model produces NaN at layer 27 in float16. **Use float64**.

### Missing Biases
Without Q/K/V biases: only 70% accuracy. With biases: 100%.

---

## 8. Weight Extraction

```python
# Biases are CRITICAL
b_q = attn.q_proj.bias.data.float().cpu().numpy()
b_k = attn.k_proj.bias.data.float().cpu().numpy()
b_v = attn.v_proj.bias.data.float().cpu().numpy()

# Reshape for heads
W_q_heads = W_q.reshape(28, 128, 3584)
W_k_heads = W_k.reshape(4, 128, 3584)  # Only 4 KV heads
b_q_heads = b_q.reshape(28, 128)
b_k_heads = b_k.reshape(4, 128)
```

---

## 9. Reference Files

- `experiments/test_unwound_final.py` - Complete working implementation
- `experiments/mesh_with_bias_rope.py` - Layer 3 detailed breakdown
- `docs/design_considerations/190_layer3_unwinding.md` - Full documentation

---

## 10. Key Insight

The transformer is **fully deterministic**. Given weights + input tokens, output is completely determined by explicit matrix operations. No black box - just:
- Matrix multiplications
- RoPE rotations  
- Softmax
- SiLU activation
- Residual connections

The "intelligence" is in the **shape** of the weight matrices.

---

## 11. Detailed Mathematical Formulations

### 11.1 RMSNorm (Root Mean Square Normalization)

Unlike LayerNorm, RMSNorm does not center the input:

$$\text{RMSNorm}(x; \gamma) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

Where:
- $x \in \mathbb{R}^d$ is input (d=3584)
- $\gamma \in \mathbb{R}^d$ is learnable scale
- $\epsilon = 10^{-6}$

```python
def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * weight
```

### 11.2 Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating pairs of dimensions:

$$\theta_i = 10000^{-2i/d_h}$$

For position $p$ and dimension pair $i$:
$$\text{RoPE}(x, p)_{2i} = x_{2i} \cos(p\theta_i) - x_{2i+1} \sin(p\theta_i)$$
$$\text{RoPE}(x, p)_{2i+1} = x_{2i} \sin(p\theta_i) + x_{2i+1} \cos(p\theta_i)$$

Vectorized form:
$$\text{RoPE}(x, p) = x \odot \cos(\Theta_p) + \text{rotate\_half}(x) \odot \sin(\Theta_p)$$

Where $\text{rotate\_half}([a, b]) = [-b, a]$ applied to each pair.

```python
def rope_embed(seq_len, inv_freq):
    freqs = np.outer(np.arange(seq_len), inv_freq)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(freqs), np.sin(freqs)

def apply_rope(x, cos, sin, head_dim):
    x1, x2 = x[:head_dim//2], x[head_dim//2:]
    return x * cos + np.concatenate([-x2, x1]) * sin
```

### 11.3 Scaled Dot-Product Attention

For query $q$, keys $K = [k_1, \ldots, k_n]$, values $V = [v_1, \ldots, v_n]$:

$$\alpha_i = \frac{\exp(q \cdot k_i / \sqrt{d_h})}{\sum_j \exp(q \cdot k_j / \sqrt{d_h})}$$

$$\text{Attention}(q, K, V) = \sum_i \alpha_i v_i$$

With causal masking, position $i$ only attends to positions $\leq i$.

```python
def attention(q, keys, values, head_dim):
    scores = np.array([np.dot(q, k) for k in keys]) / np.sqrt(head_dim)
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()
    return sum(w * v for w, v in zip(weights, values))
```

### 11.4 SiLU (Swish) Activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```python
def silu(x):
    return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
```

### 11.5 Gated MLP

$$\text{MLP}(x) = W_{down} \left( \text{SiLU}(xW_{gate}^T) \odot (xW_{up}^T) \right)$$

```python
def mlp(x, W_gate, W_up, W_down):
    gate = silu(x @ W_gate.T)
    up = x @ W_up.T
    return (gate * up) @ W_down.T
```

---

## 12. Complete Layer Computation

### 12.1 Single Layer Forward (2-token sequence)

```python
def compute_layer(layer_idx, h, cos, sin, layers, config):
    """
    Process one transformer layer for 2-token input.
    
    h: shape (2, 3584) - hidden states for tokens A and B
    Returns: shape (2, 3584) - updated hidden states
    """
    L = layers[layer_idx]
    h_A, h_B = h[0], h[1]
    
    # Pre-attention layer norm
    h_A_n = rms_norm(h_A, L['ln_attn'])
    h_B_n = rms_norm(h_B, L['ln_attn'])
    
    attn_out = np.zeros((2, 3584))
    
    # Position 0: self-attention only (causal)
    for head in range(28):
        kv = head // 7  # GQA indexing
        v_A = h_A_n @ L['W_v_heads'][kv].T + L['b_v_heads'][kv]
        attn_out[0, head*128:(head+1)*128] = v_A
    
    # Position 1: attends to both
    for head in range(28):
        kv = head // 7
        
        # Projections with bias
        q_B = h_B_n @ L['W_q_heads'][head].T + L['b_q_heads'][head]
        k_A = h_A_n @ L['W_k_heads'][kv].T + L['b_k_heads'][kv]
        k_B = h_B_n @ L['W_k_heads'][kv].T + L['b_k_heads'][kv]
        v_A = h_A_n @ L['W_v_heads'][kv].T + L['b_v_heads'][kv]
        v_B = h_B_n @ L['W_v_heads'][kv].T + L['b_v_heads'][kv]
        
        # RoPE
        q_B = apply_rope(q_B, cos[1], sin[1], 128)
        k_A = apply_rope(k_A, cos[0], sin[0], 128)
        k_B = apply_rope(k_B, cos[1], sin[1], 128)
        
        # Attention
        s_A = np.dot(q_B, k_A) / np.sqrt(128)
        s_B = np.dot(q_B, k_B) / np.sqrt(128)
        scores = np.array([s_A, s_B])
        weights = np.exp(scores - scores.max())
        weights = weights / weights.sum()
        
        attn_out[1, head*128:(head+1)*128] = weights[0]*v_A + weights[1]*v_B
    
    # Output projection
    attn_out[0] = attn_out[0] @ L['W_o'].T
    attn_out[1] = attn_out[1] @ L['W_o'].T
    
    # Residual
    h_post = h + attn_out
    
    # MLP
    mlp_out = np.zeros((2, 3584))
    for p in range(2):
        h_n = rms_norm(h_post[p], L['ln_mlp'])
        mlp_out[p] = (silu(h_n @ L['W_gate'].T) * (h_n @ L['W_up'].T)) @ L['W_down'].T
    
    return h_post + mlp_out
```

---

## 13. Complete Forward Pass

```python
def forward_unwound(token_A, token_B, embeddings, layers, final_ln, lm_head, inv_freq):
    """
    Complete forward pass through Qwen2-7B.
    
    Returns: predicted next token ID
    """
    # Embed
    h = np.stack([embeddings[token_A], embeddings[token_B]])
    
    # RoPE for 2 positions
    cos, sin = rope_embed(2, inv_freq)
    
    # 28 layers
    for i in range(28):
        h = compute_layer(i, h, cos, sin, layers, config)
    
    # Final norm + LM head
    h_final = rms_norm(h[1], final_ln)
    logits = lm_head @ h_final
    
    return np.argmax(logits)
```

---

## 14. Weight Shapes Reference

| Weight | Shape | Notes |
|--------|-------|-------|
| embeddings | (152064, 3584) | Token embeddings |
| W_q | (3584, 3584) | 28 heads × 128 dim |
| W_k | (512, 3584) | 4 KV heads × 128 dim |
| W_v | (512, 3584) | 4 KV heads × 128 dim |
| W_o | (3584, 3584) | Output projection |
| b_q | (3584,) | Q bias |
| b_k | (512,) | K bias |
| b_v | (512,) | V bias |
| ln_attn | (3584,) | Pre-attention norm |
| ln_mlp | (3584,) | Pre-MLP norm |
| W_gate | (18944, 3584) | MLP gate |
| W_up | (18944, 3584) | MLP up |
| W_down | (3584, 18944) | MLP down |
| final_ln | (3584,) | Final norm |
| lm_head | (152064, 3584) | Output projection |

---

## 15. Debugging Checklist

If your accuracy is low:

1. **Q/K/V biases included?** - Most common issue
2. **RoPE on both Q AND K?** - Not just Q
3. **Correct GQA indexing?** - `kv_idx = head // 7`
4. **Float64 precision?** - Float16 causes NaN
5. **Causal mask correct?** - Pos 0 only sees itself
6. **Residual connections?** - Two per layer
7. **Correct weight transpose?** - `x @ W.T` not `W @ x`

---

## 16. Validation Code

```python
def validate(model, tokenizer, forward_fn, n_samples=20):
    correct = 0
    for _ in range(n_samples):
        A = np.random.randint(1000, 10000)
        B = np.random.randint(1000, 10000)
        
        # Model prediction
        ids = torch.tensor([[A, B]]).to(model.device)
        with torch.no_grad():
            out = model(ids)
        actual = torch.argmax(out.logits[0, 1]).item()
        
        # Your prediction
        pred = forward_fn(A, B)
        
        if actual == pred:
            correct += 1
    
    print(f"Accuracy: {correct}/{n_samples} = {correct/n_samples*100:.1f}%")
```

Expected: **100% accuracy** with correct implementation.
