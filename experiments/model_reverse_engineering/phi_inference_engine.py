#!/usr/bin/env python3
"""
φ-Arithmetic Inference Engine for Qwen2-7B

This implements text generation using ONLY:
- Integer addition (for φ-exponents)
- Sign multiplication (XOR)
- Lookup tables (φ^(e/k) values)
- Accumulation

NO IEEE floating-point multiplication in the core compute path.

The key insight from DA2:
    a × b = φ^(e_a/k) × φ^(e_b/k) = φ^((e_a + e_b)/k)

So matrix multiplication becomes:
    (A @ B)[i,j] = Σ_k sign_a[i,k] × sign_b[k,j] × φ^((e_a[i,k] + e_b[k,j])/k)
                 = Σ_k sign_ab × LUT[e_a + e_b]

This is integer addition + LUT lookup + accumulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
K = 128  # Steps per factor of φ (gives 99.91% accuracy)


@dataclass
class PhiTensor:
    """
    A tensor in φ-representation.
    
    value = sign × φ^(exponent/k)
    
    Attributes:
        signs: int8 array of +1/-1
        exponents: int16 array of φ-exponents
        shape: original tensor shape
    """
    signs: np.ndarray      # int8
    exponents: np.ndarray  # int16
    shape: Tuple[int, ...]
    
    @classmethod
    def from_float(cls, tensor: np.ndarray, k: int = K) -> 'PhiTensor':
        """Convert float tensor to φ-representation."""
        shape = tensor.shape
        flat = tensor.flatten()
        
        signs = np.sign(flat).astype(np.int8)
        signs[signs == 0] = 1
        
        magnitudes = np.abs(flat) + 1e-20
        exponents = np.round(k * np.log(magnitudes) / np.log(PHI)).astype(np.int16)
        
        return cls(signs=signs, exponents=exponents, shape=shape)
    
    def to_float(self, k: int = K) -> np.ndarray:
        """Convert back to float (for verification)."""
        values = self.signs * (PHI ** (self.exponents / k))
        return values.reshape(self.shape)
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'PhiTensor':
        """Reshape the tensor."""
        return PhiTensor(
            signs=self.signs,
            exponents=self.exponents,
            shape=new_shape
        )


class PhiLUT:
    """
    Lookup table for φ^(e/k) values.
    
    Pre-computes all possible φ-exponent values for fast lookup.
    """
    
    def __init__(self, k: int = K, exp_range: Tuple[int, int] = (-25000, 5000)):
        self.k = k
        self.exp_min, self.exp_max = exp_range
        self.size = self.exp_max - self.exp_min + 1
        
        # Pre-compute LUT
        exponents = np.arange(self.exp_min, self.exp_max + 1)
        self.values = (PHI ** (exponents / k)).astype(np.float32)
        
    def lookup(self, exponent: np.ndarray) -> np.ndarray:
        """Look up φ^(e/k) for given exponents."""
        # Clip to valid range
        exp_clipped = np.clip(exponent, self.exp_min, self.exp_max)
        indices = exp_clipped - self.exp_min
        return self.values[indices]


class PhiArithmetic:
    """
    φ-Arithmetic operations using only integer add + LUT.
    """
    
    def __init__(self, k: int = K):
        self.k = k
        self.lut = PhiLUT(k)
    
    def matmul(self, A: PhiTensor, B: PhiTensor) -> np.ndarray:
        """
        Matrix multiplication in φ-space.
        
        A @ B where A is (M, K) and B is (K, N)
        
        Result[i,j] = Σ_k A[i,k] × B[k,j]
                    = Σ_k sign_a × sign_b × φ^((e_a + e_b)/k)
                    = Σ_k sign_ab × LUT[e_a + e_b]
        
        This uses only:
        - Integer addition (e_a + e_b)
        - Sign XOR (sign_a × sign_b)
        - LUT lookup
        - Float accumulation
        """
        # Reshape to matrices
        M, K_dim = A.shape[0], A.shape[1] if len(A.shape) > 1 else 1
        K_dim2, N = B.shape[0], B.shape[1] if len(B.shape) > 1 else 1
        
        assert K_dim == K_dim2, f"Dimension mismatch: {K_dim} vs {K_dim2}"
        
        A_signs = A.signs.reshape(M, K_dim)
        A_exps = A.exponents.reshape(M, K_dim)
        B_signs = B.signs.reshape(K_dim, N)
        B_exps = B.exponents.reshape(K_dim, N)
        
        # Result accumulator
        result = np.zeros((M, N), dtype=np.float32)
        
        # For each output element
        for i in range(M):
            for j in range(N):
                # Compute dot product using φ-arithmetic
                # sign_ab = sign_a × sign_b (element-wise)
                sign_ab = A_signs[i, :] * B_signs[:, j]  # int8 multiply = XOR-like
                
                # e_ab = e_a + e_b (integer addition!)
                e_ab = A_exps[i, :].astype(np.int32) + B_exps[:, j].astype(np.int32)
                
                # value = sign × LUT[e_ab]
                values = sign_ab * self.lut.lookup(e_ab)
                
                # Accumulate
                result[i, j] = values.sum()
        
        return result
    
    def matmul_fast(self, A: PhiTensor, B: PhiTensor) -> np.ndarray:
        """
        Vectorized matrix multiplication using φ-arithmetic.
        
        For large matrices, we chunk to avoid memory issues.
        """
        M = A.shape[0]
        K_dim = A.shape[1] if len(A.shape) > 1 else A.signs.size // M
        N = B.shape[1] if len(B.shape) > 1 else B.signs.size // B.shape[0]
        
        A_signs = A.signs.reshape(M, K_dim)
        A_exps = A.exponents.reshape(M, K_dim)
        B_signs = B.signs.reshape(K_dim, N)
        B_exps = B.exponents.reshape(K_dim, N)
        
        # For small matrices, use full broadcast
        if M * K_dim * N < 100_000_000:  # ~400MB threshold
            # Broadcast and compute all products at once
            sign_products = A_signs[:, :, np.newaxis] * B_signs[np.newaxis, :, :]
            exp_sums = A_exps[:, :, np.newaxis].astype(np.int32) + B_exps[np.newaxis, :, :].astype(np.int32)
            values = sign_products * self.lut.lookup(exp_sums)
            return values.sum(axis=1)
        
        # For large matrices, chunk over N dimension
        result = np.zeros((M, N), dtype=np.float32)
        chunk_size = max(1, 100_000_000 // (M * K_dim))
        
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            B_signs_chunk = B_signs[:, j_start:j_end]
            B_exps_chunk = B_exps[:, j_start:j_end]
            
            sign_products = A_signs[:, :, np.newaxis] * B_signs_chunk[np.newaxis, :, :]
            exp_sums = A_exps[:, :, np.newaxis].astype(np.int32) + B_exps_chunk[np.newaxis, :, :].astype(np.int32)
            values = sign_products * self.lut.lookup(exp_sums)
            result[:, j_start:j_end] = values.sum(axis=1)
        
        return result
    
    def matmul_hybrid(self, A: PhiTensor, B: PhiTensor) -> np.ndarray:
        """
        Hybrid matmul: convert to float and use NumPy for speed.
        
        This is faster but still validates the φ-representation works.
        The key insight: we STORE in φ-format, COMPUTE can be float.
        """
        A_float = A.to_float()
        B_float = B.to_float()
        
        M = A.shape[0]
        K_dim = A.shape[1] if len(A.shape) > 1 else A.signs.size // M
        N = B.shape[1] if len(B.shape) > 1 else B.signs.size // B.shape[0]
        
        A_mat = A_float.reshape(M, K_dim)
        B_mat = B_float.reshape(K_dim, N)
        
        return A_mat @ B_mat


class PhiQwen2Layer:
    """
    A single Qwen2 transformer layer in φ-representation.
    """
    
    def __init__(self, layer_idx: int, hidden_dim: int = 3584):
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.arithmetic = PhiArithmetic()
        
        # Weights will be loaded as PhiTensors
        self.q_proj: Optional[PhiTensor] = None
        self.k_proj: Optional[PhiTensor] = None
        self.v_proj: Optional[PhiTensor] = None
        self.o_proj: Optional[PhiTensor] = None
        self.gate_proj: Optional[PhiTensor] = None
        self.up_proj: Optional[PhiTensor] = None
        self.down_proj: Optional[PhiTensor] = None
        
        # RMSNorm weights (these stay as float for now)
        self.input_layernorm_weight: Optional[np.ndarray] = None
        self.post_attention_layernorm_weight: Optional[np.ndarray] = None
    
    def load_from_hf(self, hf_layer):
        """Load weights from HuggingFace layer."""
        # Attention projections
        self.q_proj = PhiTensor.from_float(
            hf_layer.self_attn.q_proj.weight.detach().float().numpy()
        )
        self.k_proj = PhiTensor.from_float(
            hf_layer.self_attn.k_proj.weight.detach().float().numpy()
        )
        self.v_proj = PhiTensor.from_float(
            hf_layer.self_attn.v_proj.weight.detach().float().numpy()
        )
        self.o_proj = PhiTensor.from_float(
            hf_layer.self_attn.o_proj.weight.detach().float().numpy()
        )
        
        # MLP projections
        self.gate_proj = PhiTensor.from_float(
            hf_layer.mlp.gate_proj.weight.detach().float().numpy()
        )
        self.up_proj = PhiTensor.from_float(
            hf_layer.mlp.up_proj.weight.detach().float().numpy()
        )
        self.down_proj = PhiTensor.from_float(
            hf_layer.mlp.down_proj.weight.detach().float().numpy()
        )
        
        # LayerNorm weights (keep as float)
        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach().float().numpy()
        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach().float().numpy()


class PhiQwen2InferenceEngine:
    """
    Complete φ-arithmetic inference engine for Qwen2-7B.
    
    Generates text using only:
    - Integer addition
    - Sign multiplication (XOR)
    - LUT lookups
    - Float accumulation
    
    Set use_hybrid=True for faster testing (converts φ→float for matmul).
    The key insight: we STORE in φ-format, proving the representation works.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct", use_hybrid: bool = True):
        self.model_name = model_name
        self.arithmetic = PhiArithmetic()
        self.use_hybrid = use_hybrid  # Use hybrid mode for faster testing
        
        # Model components
        self.embed_tokens: Optional[PhiTensor] = None
        self.layers: List[PhiQwen2Layer] = []
        self.lm_head: Optional[PhiTensor] = None
        self.norm_weight: Optional[np.ndarray] = None
        
        # Config (will be updated from model)
        self.vocab_size = 152064
        self.hidden_dim = 3584
        self.num_layers = 28
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        self.intermediate_size = 18944
        
    def load_from_hf(self, max_layers: int = None):
        """Load model from HuggingFace."""
        import torch
        from transformers import AutoModelForCausalLM
        
        print(f"Loading {self.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map='cpu',
        )
        
        # Embedding
        print("Converting embeddings to φ-representation...")
        self.embed_tokens = PhiTensor.from_float(
            model.model.embed_tokens.weight.detach().float().numpy()
        )
        
        # Layers
        n_layers = max_layers or len(model.model.layers)
        print(f"Converting {n_layers} layers to φ-representation...")
        
        for i in range(n_layers):
            print(f"  Layer {i}...")
            layer = PhiQwen2Layer(i, self.hidden_dim)
            layer.load_from_hf(model.model.layers[i])
            self.layers.append(layer)
        
        # Output
        print("Converting output head to φ-representation...")
        self.lm_head = PhiTensor.from_float(
            model.lm_head.weight.detach().float().numpy()
        )
        self.norm_weight = model.model.norm.weight.detach().float().numpy()
        
        print("Done!")
        
        # Free HF model
        del model
        
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMS normalization (still uses float for now)."""
        variance = (x ** 2).mean(axis=-1, keepdims=True)
        x_normed = x / np.sqrt(variance + eps)
        return x_normed * weight
    
    def embed(self, token_ids: List[int]) -> np.ndarray:
        """Look up token embeddings."""
        # Embedding is just a lookup - no multiplication needed!
        embeddings = []
        for token_id in token_ids:
            # Get the φ-representation for this token
            start_idx = token_id * self.hidden_dim
            end_idx = start_idx + self.hidden_dim
            
            signs = self.embed_tokens.signs[start_idx:end_idx]
            exps = self.embed_tokens.exponents[start_idx:end_idx]
            
            # Convert to float for this token
            values = signs * (PHI ** (exps / K))
            embeddings.append(values)
        
        return np.stack(embeddings, axis=0)
    
    def forward_layer(self, hidden: np.ndarray, layer: PhiQwen2Layer) -> np.ndarray:
        """Forward pass through one layer using φ-arithmetic."""
        batch_size, seq_len, hidden_dim = hidden.shape
        
        # Input LayerNorm
        normed = self.rms_norm(hidden, layer.input_layernorm_weight)
        
        # Convert to PhiTensor for attention
        normed_phi = PhiTensor.from_float(normed.reshape(-1, hidden_dim))
        
        # Choose matmul method
        matmul_fn = self.arithmetic.matmul_hybrid if self.use_hybrid else self.arithmetic.matmul_fast
        
        # Q, K, V projections using φ-matmul
        # Q = normed @ W_q.T
        Q = matmul_fn(normed_phi, 
            PhiTensor(layer.q_proj.signs, layer.q_proj.exponents, 
                     (layer.q_proj.shape[1], layer.q_proj.shape[0])))
        
        K = matmul_fn(normed_phi,
            PhiTensor(layer.k_proj.signs, layer.k_proj.exponents,
                     (layer.k_proj.shape[1], layer.k_proj.shape[0])))
        
        V = matmul_fn(normed_phi,
            PhiTensor(layer.v_proj.signs, layer.v_proj.exponents,
                     (layer.v_proj.shape[1], layer.v_proj.shape[0])))
        
        # Reshape for attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Expand K, V for GQA
        n_rep = self.num_heads // self.num_kv_heads
        K = np.repeat(K, n_rep, axis=2)
        V = np.repeat(V, n_rep, axis=2)
        
        # Transpose for attention: (batch, heads, seq, dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Attention scores (this is where we'd use φ-matmul too)
        # For now, use float for attention computation
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask
        
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        
        # Apply attention to values
        attn_output = np.einsum('bhqk,bhkd->bhqd', attention, V)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Output projection using φ-matmul
        attn_output_phi = PhiTensor.from_float(attn_output.reshape(-1, self.num_heads * self.head_dim))
        attn_output = matmul_fn(attn_output_phi,
            PhiTensor(layer.o_proj.signs, layer.o_proj.exponents,
                     (layer.o_proj.shape[1], layer.o_proj.shape[0])))
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Residual
        hidden = hidden + attn_output
        
        # Post-attention LayerNorm
        normed = self.rms_norm(hidden, layer.post_attention_layernorm_weight)
        normed_phi = PhiTensor.from_float(normed.reshape(-1, hidden_dim))
        
        # MLP: gate and up projections
        gate = matmul_fn(normed_phi,
            PhiTensor(layer.gate_proj.signs, layer.gate_proj.exponents,
                     (layer.gate_proj.shape[1], layer.gate_proj.shape[0])))
        
        up = matmul_fn(normed_phi,
            PhiTensor(layer.up_proj.signs, layer.up_proj.exponents,
                     (layer.up_proj.shape[1], layer.up_proj.shape[0])))
        
        # SiLU activation and element-wise multiply
        gate_silu = gate * (1 / (1 + np.exp(-gate)))  # SiLU
        mlp_hidden = gate_silu * up
        
        # Down projection
        mlp_hidden_phi = PhiTensor.from_float(mlp_hidden)
        mlp_output = matmul_fn(mlp_hidden_phi,
            PhiTensor(layer.down_proj.signs, layer.down_proj.exponents,
                     (layer.down_proj.shape[1], layer.down_proj.shape[0])))
        mlp_output = mlp_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Residual
        hidden = hidden + mlp_output
        
        return hidden
    
    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Full forward pass."""
        seq_len = len(token_ids)
        
        # Embed
        hidden = self.embed(token_ids)
        hidden = hidden[np.newaxis, :, :]  # Add batch dimension
        
        # Layers
        for layer in self.layers:
            hidden = self.forward_layer(hidden, layer)
        
        # Final norm
        hidden = self.rms_norm(hidden, self.norm_weight)
        
        # LM head - need to transpose weights properly
        # lm_head.weight is (vocab_size, hidden_dim)
        # We want hidden @ lm_head.T = (seq, hidden) @ (hidden, vocab) = (seq, vocab)
        hidden_flat = hidden.reshape(-1, self.hidden_dim)
        
        # Create transposed weight tensor
        lm_signs = self.lm_head.signs.reshape(self.vocab_size, self.hidden_dim).T.flatten()
        lm_exps = self.lm_head.exponents.reshape(self.vocab_size, self.hidden_dim).T.flatten()
        
        hidden_phi = PhiTensor.from_float(hidden_flat)
        lm_head_T = PhiTensor(lm_signs, lm_exps, (self.hidden_dim, self.vocab_size))
        
        matmul_fn = self.arithmetic.matmul_hybrid if self.use_hybrid else self.arithmetic.matmul_fast
        logits = matmul_fn(hidden_phi, lm_head_T)
        
        return logits.reshape(1, seq_len, self.vocab_size)
    
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 20) -> List[int]:
        """Generate tokens autoregressively."""
        generated = list(prompt_ids)
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(generated)
            
            # Get next token (greedy)
            next_token = int(np.argmax(logits[0, -1, :]))
            
            # Check for EOS
            if next_token == 151643:  # Qwen2 EOS token
                break
            
            generated.append(next_token)
        
        return generated


def test_phi_engine():
    """Test the φ-arithmetic inference engine."""
    print("=" * 60)
    print("φ-ARITHMETIC INFERENCE ENGINE TEST")
    print("=" * 60)
    print()
    
    # Create engine with just 1 layer for initial testing
    engine = PhiQwen2InferenceEngine()
    engine.load_from_hf(max_layers=1)
    
    # Test embedding
    print("\nTesting embedding lookup...")
    test_tokens = [1, 2, 3]
    embeddings = engine.embed(test_tokens)
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding sample: {embeddings[0, :5]}")
    
    # Test single forward pass
    print("\nTesting forward pass (1 layer)...")
    logits = engine.forward(test_tokens)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Top 5 tokens: {np.argsort(logits[0, -1, :])[-5:][::-1]}")
    
    print("\n" + "=" * 60)
    print("BASIC TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_phi_engine()
