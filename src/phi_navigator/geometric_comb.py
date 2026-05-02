"""
Geometric Comb - Layer operations as geometric transformations.

The Music Box Principle (Doc 112):
- Drum: Token embeddings (information)
- Comb: Layer operations (function)
- Music: Output (emergent from drum + comb interaction)

This module implements the "comb" - the geometric operations that
transform the drum (input) into music (output).

Each layer is a "tine" of the comb:
1. LayerNorm: Project to unit sphere
2. Attention (MESH): Project to 106-dim discriminant space, scale by φ-Zipf S
3. MLP: Bilinear transform (gate ≈ 0.5 in linear regime)
4. Residual: Add to running position

The 28 layers are 28 tines. The drum rotates through all 28 tines.
The music emerges from this geometric interaction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch

PHI = 1.6180339887498949
INV_PHI = 1.0 / PHI


@dataclass
class CombTine:
    """
    One tine of the comb (one transformer layer).
    
    Contains the geometric structure needed to transform the drum.
    """
    layer_idx: int
    
    # Attention components (full, for exact reproduction)
    attn_q_proj: np.ndarray     # (num_heads * head_dim, hidden_dim)
    attn_k_proj: np.ndarray     # (num_kv_heads * head_dim, hidden_dim)
    attn_v_proj: np.ndarray     # (num_kv_heads * head_dim, hidden_dim)
    attn_o_proj: np.ndarray     # (hidden_dim, num_heads * head_dim)
    attn_q_bias: np.ndarray     # (num_heads * head_dim,)
    attn_k_bias: np.ndarray     # (num_kv_heads * head_dim,)
    attn_v_bias: np.ndarray     # (num_kv_heads * head_dim,)
    
    # Attention (MESH) components for geometric approximation
    mesh_U: np.ndarray      # (hidden_dim, k) - left singular vectors
    mesh_S: np.ndarray      # (k,) - singular values (φ-Zipf scaled)
    mesh_Vt: np.ndarray     # (k, hidden_dim) - right singular vectors
    
    # MLP components
    mlp_gate_proj: np.ndarray    # (intermediate_dim, hidden_dim)
    mlp_up_proj: np.ndarray      # (intermediate_dim, hidden_dim)
    mlp_down_proj: np.ndarray    # (hidden_dim, intermediate_dim)
    
    # Layer norm parameters
    ln1_weight: np.ndarray   # (hidden_dim,) - input layer norm
    ln2_weight: np.ndarray   # (hidden_dim,) - post-attention layer norm
    
    @property
    def k(self) -> int:
        """Discriminant dimension."""
        return len(self.mesh_S)
    
    @property
    def hidden_dim(self) -> int:
        return self.mesh_U.shape[0]


class GeometricComb:
    """
    The geometric comb - transforms drum positions into music.
    
    Implements transformer layers as geometric operations:
    - Attention: Discriminant projection + φ-Zipf scaling
    - MLP: Bilinear transform (linearized SiLU)
    - Residual connections
    """
    
    def __init__(self, k: int = 106):
        """
        Args:
            k: Discriminant dimension (default 106 for 99.5% accuracy)
        """
        self.k = k
        self.tines: List[CombTine] = []
        self.tokenizer = None
        self.embeddings: Optional[np.ndarray] = None
        self.lm_head: Optional[np.ndarray] = None
        
        # Vocabulary for output
        self.vocab_size: int = 0
        self.hidden_dim: int = 0
        
        # Final layer norm (applied before LM head)
        self.final_norm_weight: Optional[np.ndarray] = None
    
    def load_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                        n_layers: int = None):
        """
        Extract geometric structure from model.
        
        Args:
            model_name: HuggingFace model name
            n_layers: Number of layers to load (default: all)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        n_layers = n_layers or model.config.num_hidden_layers
        
        # Get embeddings, LM head, and final norm
        self.embeddings = model.model.embed_tokens.weight.data.numpy()
        self.lm_head = model.lm_head.weight.data.numpy()  # (vocab_size, hidden_dim)
        self.final_norm_weight = model.model.norm.weight.data.numpy()  # (hidden_dim,)
        
        print(f"Extracting geometric structure from {n_layers} layers...")
        
        num_heads = model.config.num_attention_heads
        head_dim = self.hidden_dim // num_heads
        num_kv_heads = model.config.num_key_value_heads
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            
            # Extract full attention components
            W_q = layer.self_attn.q_proj.weight.data.numpy()
            W_k = layer.self_attn.k_proj.weight.data.numpy()
            W_v = layer.self_attn.v_proj.weight.data.numpy()
            W_o = layer.self_attn.o_proj.weight.data.numpy()
            
            b_q = layer.self_attn.q_proj.bias.data.numpy()
            b_k = layer.self_attn.k_proj.bias.data.numpy()
            b_v = layer.self_attn.v_proj.bias.data.numpy()
            
            # Compute MESH for head 0 (for geometric approximation)
            head_idx = 0
            q_start = head_idx * head_dim
            q_end = (head_idx + 1) * head_dim
            kv_head_idx = head_idx * num_kv_heads // num_heads
            k_start = kv_head_idx * head_dim
            k_end = (kv_head_idx + 1) * head_dim
            
            W_q_head = W_q[q_start:q_end, :]
            W_k_head = W_k[k_start:k_end, :]
            
            MESH = W_q_head.T @ W_k_head
            U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
            
            mesh_U = U[:, :self.k]
            mesh_S = S[:self.k]
            mesh_Vt = Vt[:self.k, :]
            
            # Extract MLP components
            gate_proj = layer.mlp.gate_proj.weight.data.numpy()
            up_proj = layer.mlp.up_proj.weight.data.numpy()
            down_proj = layer.mlp.down_proj.weight.data.numpy()
            
            # Layer norms
            ln1_weight = layer.input_layernorm.weight.data.numpy()
            ln2_weight = layer.post_attention_layernorm.weight.data.numpy()
            
            tine = CombTine(
                layer_idx=layer_idx,
                attn_q_proj=W_q,
                attn_k_proj=W_k,
                attn_v_proj=W_v,
                attn_o_proj=W_o,
                attn_q_bias=b_q,
                attn_k_bias=b_k,
                attn_v_bias=b_v,
                mesh_U=mesh_U,
                mesh_S=mesh_S,
                mesh_Vt=mesh_Vt,
                mlp_gate_proj=gate_proj,
                mlp_up_proj=up_proj,
                mlp_down_proj=down_proj,
                ln1_weight=ln1_weight,
                ln2_weight=ln2_weight,
            )
            
            self.tines.append(tine)
            
            if layer_idx % 7 == 0:
                print(f"  Layer {layer_idx}: MESH S[0]={mesh_S[0]:.3f}, "
                      f"k={self.k} dims capture {np.sum(mesh_S[:self.k]**2)/np.sum(S**2)*100:.1f}%")
        
        print(f"Loaded {len(self.tines)} tines (layers)")
        
        del model
    
    def layer_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        RMSNorm (used by Qwen2).
        
        Geometrically: project to unit sphere, then scale by weight.
        """
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        return (x / rms) * weight
    
    def attention_exact(self, x_norm: np.ndarray, tine: CombTine) -> np.ndarray:
        """
        Exact attention for single token.
        
        For single token, softmax is always 1, so attention = V @ O.
        This gives 100% correlation with the model.
        """
        # V projection with bias
        v = tine.attn_v_proj @ x_norm + tine.attn_v_bias
        
        # Expand for GQA (4 KV heads -> 28 Q heads)
        num_heads = 28
        num_kv_heads = 4
        head_dim = 128
        kv_per_q = num_heads // num_kv_heads
        
        v_heads = v.reshape(num_kv_heads, head_dim)
        v_expanded = np.repeat(v_heads, kv_per_q, axis=0).reshape(-1)
        
        # O projection (no bias)
        attn_output = tine.attn_o_proj @ v_expanded
        
        return attn_output
    
    def attention_geometric(self, x_norm: np.ndarray, tine: CombTine) -> np.ndarray:
        """
        Geometric attention using discriminant space.
        
        Approximation using MESH SVD - captures 99% with 106 dims.
        """
        # Project to discriminant space
        output = tine.mesh_U @ (tine.mesh_S * (tine.mesh_Vt @ x_norm))
        return output
    
    def mlp_exact(self, x_norm: np.ndarray, tine: CombTine) -> np.ndarray:
        """
        Exact MLP computation.
        
        gate * up with SiLU activation, then down projection.
        """
        gate = tine.mlp_gate_proj @ x_norm
        up = tine.mlp_up_proj @ x_norm
        
        # SiLU activation: x * sigmoid(x)
        silu_gate = gate / (1 + np.exp(-gate))
        hidden = silu_gate * up
        
        output = tine.mlp_down_proj @ hidden
        return output
    
    def mlp_bilinear(self, x_norm: np.ndarray, tine: CombTine) -> np.ndarray:
        """
        Bilinear MLP approximation.
        
        From Doc 132: In linear regime, SiLU(x) ≈ x/2.
        This gives 99.99% correlation with full MLP.
        """
        gate = tine.mlp_gate_proj @ x_norm
        up = tine.mlp_up_proj @ x_norm
        
        # Bilinear approximation
        hidden = (gate / 2) * up
        
        output = tine.mlp_down_proj @ hidden
        return output
    
    def forward_tine(self, x: np.ndarray, tine: CombTine, 
                     use_exact: bool = True) -> np.ndarray:
        """
        Pass input through one tine (layer).
        
        Args:
            x: Input hidden state
            tine: Layer parameters
            use_exact: If True, use exact computation. If False, use geometric approximation.
        """
        # Pre-attention norm
        x_norm = self.layer_norm(x, tine.ln1_weight)
        
        # Attention
        if use_exact:
            attn_out = self.attention_exact(x_norm, tine)
        else:
            attn_out = self.attention_geometric(x_norm, tine)
        
        # Residual
        x = x + attn_out
        
        # Post-attention norm
        x_norm = self.layer_norm(x, tine.ln2_weight)
        
        # MLP
        if use_exact:
            mlp_out = self.mlp_exact(x_norm, tine)
        else:
            mlp_out = self.mlp_bilinear(x_norm, tine)
        
        # Residual
        x = x + mlp_out
        
        return x
    
    def forward(self, token_id: int, n_layers: int = None, 
                use_exact: bool = True) -> np.ndarray:
        """
        Pass token through the comb (all layers).
        
        This is the drum rotating through all tines.
        
        Args:
            token_id: Token to process
            n_layers: Number of layers to use (default: all)
            use_exact: If True, use exact computation. If False, use geometric approximation.
        """
        n_layers = n_layers or len(self.tines)
        
        # Start with embedding (drum position)
        x = self.embeddings[token_id].copy()
        
        # Rotate through tines
        for i in range(min(n_layers, len(self.tines))):
            x = self.forward_tine(x, self.tines[i], use_exact=use_exact)
        
        return x
    
    def predict_next(self, token_id: int, n_layers: int = None, top_k: int = 5,
                     use_exact: bool = True) -> List[Tuple[str, float]]:
        """
        Predict next token using geometric comb.
        
        Returns top-k predictions with scores.
        """
        # Forward through comb
        hidden = self.forward(token_id, n_layers, use_exact=use_exact)
        
        # Apply final layer norm
        hidden = self.layer_norm(hidden, self.final_norm_weight)
        
        # Project to vocabulary (LM head)
        logits = self.lm_head @ hidden  # (vocab_size,)
        
        # Get top-k
        top_indices = np.argsort(-logits)[:top_k]
        
        results = []
        for idx in top_indices:
            token = self.tokenizer.decode([idx])
            score = logits[idx]
            results.append((token, score))
        
        return results
    
    def compare_with_model(self, token_id: int):
        """
        Compare geometric comb output with actual model output.
        """
        from transformers import AutoModelForCausalLM
        
        print(f"\nComparing geometric comb with model for token {token_id}...")
        
        # Geometric comb prediction
        geo_hidden = self.forward(token_id)
        geo_logits = self.lm_head @ geo_hidden
        geo_top = np.argsort(-geo_logits)[:5]
        
        print(f"\nGeometric comb predictions:")
        for idx in geo_top:
            token = self.tokenizer.decode([idx])
            print(f"  {token!r}: {geo_logits[idx]:.4f}")
        
        # Load model for comparison
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        # Get model prediction
        input_ids = torch.tensor([[token_id]])
        with torch.no_grad():
            outputs = model(input_ids)
            model_logits = outputs.logits[0, -1, :].numpy()
        
        model_top = np.argsort(-model_logits)[:5]
        
        print(f"\nModel predictions:")
        for idx in model_top:
            token = self.tokenizer.decode([idx])
            print(f"  {token!r}: {model_logits[idx]:.4f}")
        
        # Correlation
        corr = np.corrcoef(geo_logits, model_logits)[0, 1]
        print(f"\nLogit correlation: {corr:.4f}")
        
        del model
        
        return corr


def test_geometric_comb():
    """Test the geometric comb."""
    print("=" * 60)
    print("Testing Geometric Comb")
    print("=" * 60)
    
    # Start with just a few layers for speed
    comb = GeometricComb(k=106)
    comb.load_from_model(n_layers=4)  # Just first 4 layers for testing
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass")
    print("=" * 60)
    
    test_words = ["The", "king", "Hello"]
    
    for word in test_words:
        token_id = comb.tokenizer.encode(word, add_special_tokens=False)[0]
        
        print(f"\n{word} (id={token_id}):")
        
        # Get predictions
        predictions = comb.predict_next(token_id, top_k=5)
        
        print(f"  Top predictions:")
        for token, score in predictions:
            print(f"    {token!r}: {score:.4f}")


def test_exact_layer():
    """
    Test exact layer reproduction to understand the geometric structure.
    
    This uses the actual model computations to verify our understanding
    before simplifying to geometric approximations.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("=" * 60)
    print("Testing Exact Layer Reproduction")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    token_id = 785  # 'The'
    
    # Get model hidden states
    input_ids = torch.tensor([[token_id]])
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        model_hidden = [h[0, 0, :].numpy() for h in outputs.hidden_states]
    
    print(f"Token: {tokenizer.decode([token_id])!r}")
    
    # Now reproduce layer by layer using numpy
    emb = model.model.embed_tokens.weight.data.numpy()
    x = emb[token_id].copy()
    
    print(f"\nLayer-by-layer reproduction:")
    print(f"  Embedding: corr={np.corrcoef(x, model_hidden[0])[0,1]:.6f}")
    
    for layer_idx in range(4):
        layer = model.model.layers[layer_idx]
        
        # Extract weights
        ln1_weight = layer.input_layernorm.weight.data.numpy()
        ln2_weight = layer.post_attention_layernorm.weight.data.numpy()
        
        gate_proj = layer.mlp.gate_proj.weight.data.numpy()
        up_proj = layer.mlp.up_proj.weight.data.numpy()
        down_proj = layer.mlp.down_proj.weight.data.numpy()
        
        # RMSNorm
        def rms_norm(x, weight):
            rms = np.sqrt(np.mean(x ** 2) + 1e-6)
            return (x / rms) * weight
        
        # Pre-attention norm
        x_norm = rms_norm(x, ln1_weight)
        
        # SKIP ATTENTION for now (single token self-attention is identity-ish)
        # In practice, attention adds a learned bias
        attn_out = np.zeros_like(x)  # Placeholder
        
        # Residual
        x = x + attn_out
        
        # Post-attention norm
        x_norm = rms_norm(x, ln2_weight)
        
        # MLP (exact)
        gate = gate_proj @ x_norm
        up = up_proj @ x_norm
        
        # SiLU activation
        silu_gate = gate / (1 + np.exp(-gate))
        hidden = silu_gate * up
        
        mlp_out = down_proj @ hidden
        
        # Residual
        x = x + mlp_out
        
        # Compare with model
        actual = model_hidden[layer_idx + 1]
        corr = np.corrcoef(x, actual)[0, 1]
        
        print(f"  Layer {layer_idx + 1} (MLP only): corr={corr:.6f}")
    
    # Now let's see what happens if we use the model's attention output
    print(f"\nWith model attention outputs:")
    
    x = emb[token_id].copy()
    
    for layer_idx in range(4):
        layer = model.model.layers[layer_idx]
        
        ln1_weight = layer.input_layernorm.weight.data.numpy()
        ln2_weight = layer.post_attention_layernorm.weight.data.numpy()
        
        gate_proj = layer.mlp.gate_proj.weight.data.numpy()
        up_proj = layer.mlp.up_proj.weight.data.numpy()
        down_proj = layer.mlp.down_proj.weight.data.numpy()
        
        def rms_norm(x, weight):
            rms = np.sqrt(np.mean(x ** 2) + 1e-6)
            return (x / rms) * weight
        
        # Use the actual hidden state after attention (from model)
        # This tells us how much error comes from MLP vs attention
        x_after_attn = model_hidden[layer_idx + 1]  # Approximate
        
        # Actually, let's compute the difference
        # If we had perfect attention, x would match model_hidden[layer_idx]
        # after the residual
        
        # For now, just use model hidden state and apply MLP
        x_norm = rms_norm(model_hidden[layer_idx], ln2_weight)
        
        gate = gate_proj @ x_norm
        up = up_proj @ x_norm
        silu_gate = gate / (1 + np.exp(-gate))
        hidden = silu_gate * up
        mlp_out = down_proj @ hidden
        
        x = model_hidden[layer_idx] + mlp_out
        
        actual = model_hidden[layer_idx + 1]
        corr = np.corrcoef(x, actual)[0, 1]
        
        print(f"  Layer {layer_idx + 1} (MLP on model hidden): corr={corr:.6f}")
    
    del model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_geometric_comb()
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        comb = GeometricComb(k=106)
        comb.load_from_model(n_layers=4)
        token_id = comb.tokenizer.encode("The", add_special_tokens=False)[0]
        comb.compare_with_model(token_id)
    elif len(sys.argv) > 1 and sys.argv[1] == "--exact":
        test_exact_layer()
    else:
        print("Usage:")
        print("  python geometric_comb.py --test     # Test forward pass")
        print("  python geometric_comb.py --compare  # Compare with model")
