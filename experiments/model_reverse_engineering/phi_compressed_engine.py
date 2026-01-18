#!/usr/bin/env python3
"""
φ-Compressed Inference Engine

Uses low-rank decomposition for AIG-style compression:
- MESH matrices: 14× compression via rank-128 SVD
- MLP matrices: 12× compression via rank-256 SVD
- Total: 51.9 GB → 6.9 GB (7.5× compression)

Key insight: Learned weight matrices are low-rank because gradient descent
creates correlated patterns. We factor out this redundancy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import os

PHI = (1 + np.sqrt(5)) / 2
K = 128  # φ-grid resolution


@dataclass
class PhiEncoded:
    """A tensor encoded in φ-basis."""
    signs: np.ndarray      # int8
    exponents: np.ndarray  # int16
    shape: Tuple[int, ...]
    
    @classmethod
    def from_float(cls, tensor: np.ndarray) -> 'PhiEncoded':
        """Encode tensor in φ-basis."""
        shape = tensor.shape
        flat = tensor.flatten()
        
        signs = np.sign(flat).astype(np.int8)
        signs[signs == 0] = 1
        
        magnitudes = np.abs(flat) + 1e-20
        exponents = np.round(K * np.log(magnitudes) / np.log(PHI)).astype(np.int16)
        
        return cls(signs=signs, exponents=exponents, shape=shape)
    
    def to_float(self) -> np.ndarray:
        """Decode from φ-basis."""
        values = self.signs * (PHI ** (self.exponents / K))
        return values.reshape(self.shape).astype(np.float32)
    
    def storage_bytes(self) -> int:
        return self.signs.nbytes + self.exponents.nbytes
    
    def save(self, path: str):
        np.savez(path, signs=self.signs, exponents=self.exponents, shape=np.array(self.shape))
    
    @classmethod
    def load(cls, path: str) -> 'PhiEncoded':
        data = np.load(path)
        return cls(signs=data['signs'], exponents=data['exponents'], shape=tuple(data['shape']))


@dataclass
class LowRankPhi:
    """Low-rank matrix in φ-basis: M ≈ U @ diag(S) @ Vt"""
    U: PhiEncoded      # (m, rank)
    S: PhiEncoded      # (rank,)
    Vt: PhiEncoded     # (rank, n)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, rank: int) -> 'LowRankPhi':
        """Create low-rank φ-encoded representation."""
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        return cls(
            U=PhiEncoded.from_float(U_r),
            S=PhiEncoded.from_float(S_r),
            Vt=PhiEncoded.from_float(Vt_r)
        )
    
    def to_float(self) -> np.ndarray:
        """Reconstruct matrix from low-rank factors."""
        U = self.U.to_float()
        S = self.S.to_float()
        Vt = self.Vt.to_float()
        return U @ np.diag(S) @ Vt
    
    def matmul(self, x: np.ndarray) -> np.ndarray:
        """Efficient matrix-vector multiplication: M @ x = U @ (S * (Vt @ x))"""
        U = self.U.to_float()
        S = self.S.to_float()
        Vt = self.Vt.to_float()
        
        # Efficient order: O(n*r + r + m*r) instead of O(m*n)
        temp = Vt @ x.T  # (rank, seq)
        temp = S[:, np.newaxis] * temp  # (rank, seq) - broadcast
        result = U @ temp  # (m, seq)
        return result.T
    
    def storage_bytes(self) -> int:
        return self.U.storage_bytes() + self.S.storage_bytes() + self.Vt.storage_bytes()
    
    def save(self, prefix: str):
        self.U.save(f"{prefix}_U.npz")
        self.S.save(f"{prefix}_S.npz")
        self.Vt.save(f"{prefix}_Vt.npz")
    
    @classmethod
    def load(cls, prefix: str) -> 'LowRankPhi':
        return cls(
            U=PhiEncoded.load(f"{prefix}_U.npz"),
            S=PhiEncoded.load(f"{prefix}_S.npz"),
            Vt=PhiEncoded.load(f"{prefix}_Vt.npz")
        )


class CompressedLayer:
    """Transformer layer with low-rank compressed weights."""
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        self.hidden_dim = 3584
        
        # Low-rank MESH matrices (one per head)
        self.mesh_qk: List[LowRankPhi] = []
        
        # Bias cross-terms (kept as float for simplicity)
        self.cross_qk: List[np.ndarray] = []
        self.cross_kq: List[np.ndarray] = []
        self.bias_term: List[float] = []
        
        # V and O projections (low-rank)
        self.W_v: Optional[LowRankPhi] = None
        self.W_o: Optional[LowRankPhi] = None
        self.b_v: Optional[np.ndarray] = None
        self.b_o: Optional[np.ndarray] = None
        
        # MLP (low-rank)
        self.W_gate: Optional[LowRankPhi] = None
        self.W_up: Optional[LowRankPhi] = None
        self.W_down: Optional[LowRankPhi] = None
        
        # LayerNorm weights
        self.ln1_weight: Optional[np.ndarray] = None
        self.ln2_weight: Optional[np.ndarray] = None
    
    def load_from_hf(self, hf_layer, mesh_rank: int = 128, mlp_rank: int = 256):
        """Load and compress from HuggingFace layer."""
        import torch
        
        # Get weight matrices
        W_q = hf_layer.self_attn.q_proj.weight.detach().float().numpy()
        b_q = hf_layer.self_attn.q_proj.bias.detach().float().numpy()
        W_k = hf_layer.self_attn.k_proj.weight.detach().float().numpy()
        b_k = hf_layer.self_attn.k_proj.bias.detach().float().numpy()
        W_v = hf_layer.self_attn.v_proj.weight.detach().float().numpy()
        b_v = hf_layer.self_attn.v_proj.bias.detach().float().numpy()
        W_o = hf_layer.self_attn.o_proj.weight.detach().float().numpy()
        
        # Compute and compress MESH matrices per head
        for head_idx in range(self.num_heads):
            kv_idx = head_idx // (self.num_heads // self.num_kv_heads)
            
            q_start = head_idx * self.head_dim
            q_end = q_start + self.head_dim
            k_start = kv_idx * self.head_dim
            k_end = k_start + self.head_dim
            
            W_q_head = W_q[q_start:q_end, :]
            b_q_head = b_q[q_start:q_end]
            W_k_head = W_k[k_start:k_end, :]
            b_k_head = b_k[k_start:k_end]
            
            # MESH = W_q.T @ W_k
            mesh = W_q_head.T @ W_k_head
            self.mesh_qk.append(LowRankPhi.from_matrix(mesh, mesh_rank))
            
            # Bias cross-terms
            self.cross_qk.append(W_q_head.T @ b_k_head)
            self.cross_kq.append(b_q_head @ W_k_head)
            self.bias_term.append(float(b_q_head @ b_k_head))
        
        # V and O projections (low-rank)
        self.W_v = LowRankPhi.from_matrix(W_v, mesh_rank)
        self.W_o = LowRankPhi.from_matrix(W_o, mesh_rank)
        self.b_v = b_v
        self.b_o = None  # Qwen2 doesn't have o_proj bias
        
        # MLP (low-rank)
        W_gate = hf_layer.mlp.gate_proj.weight.detach().float().numpy()
        W_up = hf_layer.mlp.up_proj.weight.detach().float().numpy()
        W_down = hf_layer.mlp.down_proj.weight.detach().float().numpy()
        
        self.W_gate = LowRankPhi.from_matrix(W_gate, mlp_rank)
        self.W_up = LowRankPhi.from_matrix(W_up, mlp_rank)
        self.W_down = LowRankPhi.from_matrix(W_down, mlp_rank)
        
        # LayerNorm
        self.ln1_weight = hf_layer.input_layernorm.weight.detach().float().numpy()
        self.ln2_weight = hf_layer.post_attention_layernorm.weight.detach().float().numpy()
    
    def save(self, layer_dir: str):
        """Save compressed layer to directory."""
        os.makedirs(layer_dir, exist_ok=True)
        
        # Save MESH matrices
        for i, mesh in enumerate(self.mesh_qk):
            mesh.save(os.path.join(layer_dir, f'mesh_{i:02d}'))
        
        # Save cross-terms
        np.savez(os.path.join(layer_dir, 'cross_terms.npz'),
                cross_qk=np.array(self.cross_qk),
                cross_kq=np.array(self.cross_kq),
                bias_term=np.array(self.bias_term))
        
        # Save V, O projections
        self.W_v.save(os.path.join(layer_dir, 'W_v'))
        self.W_o.save(os.path.join(layer_dir, 'W_o'))
        
        # Save MLP
        self.W_gate.save(os.path.join(layer_dir, 'W_gate'))
        self.W_up.save(os.path.join(layer_dir, 'W_up'))
        self.W_down.save(os.path.join(layer_dir, 'W_down'))
        
        # Save biases and norms
        np.savez(os.path.join(layer_dir, 'biases.npz'),
                b_v=self.b_v,
                b_o=self.b_o if self.b_o is not None else np.array([]),
                ln1_weight=self.ln1_weight,
                ln2_weight=self.ln2_weight)
    
    @classmethod
    def load(cls, layer_dir: str, layer_idx: int) -> 'CompressedLayer':
        """Load compressed layer from directory."""
        layer = cls(layer_idx)
        
        # Load MESH matrices
        layer.mesh_qk = []
        for i in range(layer.num_heads):
            mesh = LowRankPhi.load(os.path.join(layer_dir, f'mesh_{i:02d}'))
            layer.mesh_qk.append(mesh)
        
        # Load cross-terms
        cross_data = np.load(os.path.join(layer_dir, 'cross_terms.npz'))
        layer.cross_qk = list(cross_data['cross_qk'])
        layer.cross_kq = list(cross_data['cross_kq'])
        layer.bias_term = list(cross_data['bias_term'])
        
        # Load V, O projections
        layer.W_v = LowRankPhi.load(os.path.join(layer_dir, 'W_v'))
        layer.W_o = LowRankPhi.load(os.path.join(layer_dir, 'W_o'))
        
        # Load MLP
        layer.W_gate = LowRankPhi.load(os.path.join(layer_dir, 'W_gate'))
        layer.W_up = LowRankPhi.load(os.path.join(layer_dir, 'W_up'))
        layer.W_down = LowRankPhi.load(os.path.join(layer_dir, 'W_down'))
        
        # Load biases and norms
        bias_data = np.load(os.path.join(layer_dir, 'biases.npz'))
        layer.b_v = bias_data['b_v']
        layer.b_o = bias_data['b_o'] if len(bias_data['b_o']) > 0 else None
        layer.ln1_weight = bias_data['ln1_weight']
        layer.ln2_weight = bias_data['ln2_weight']
        
        return layer
    
    def storage_bytes(self) -> int:
        total = sum(m.storage_bytes() for m in self.mesh_qk)
        total += self.W_v.storage_bytes() + self.W_o.storage_bytes()
        total += self.W_gate.storage_bytes() + self.W_up.storage_bytes() + self.W_down.storage_bytes()
        return total


class PhiCompressedEngine:
    """
    Compressed φ-arithmetic inference engine.
    
    Uses low-rank decomposition for 7.5× compression.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        self.model_name = model_name
        
        # Model components
        self.embed_tokens: Optional[PhiEncoded] = None
        self.layers: List[CompressedLayer] = []
        self.lm_head: Optional[PhiEncoded] = None
        self.norm_weight: Optional[np.ndarray] = None
        
        # Config
        self.vocab_size = 152064
        self.hidden_dim = 3584
        self.num_layers = 28
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        
        # Compression settings
        self.mesh_rank = 128
        self.mlp_rank = 256
    
    def load_from_hf(self, max_layers: int = None):
        """Load and compress model from HuggingFace."""
        import torch
        from transformers import AutoModelForCausalLM
        
        print(f"Loading {self.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map='cpu',
        )
        
        # Embedding (not compressed - used for lookup)
        print("Encoding embeddings...")
        self.embed_tokens = PhiEncoded.from_float(
            model.model.embed_tokens.weight.detach().float().numpy()
        )
        
        # Layers
        n_layers = max_layers or len(model.model.layers)
        print(f"Compressing {n_layers} layers (mesh_rank={self.mesh_rank}, mlp_rank={self.mlp_rank})...")
        
        for i in range(n_layers):
            print(f"  Layer {i}...")
            layer = CompressedLayer(i)
            layer.load_from_hf(model.model.layers[i], self.mesh_rank, self.mlp_rank)
            self.layers.append(layer)
        
        # Output head (not compressed - used for final projection)
        print("Encoding output head...")
        self.lm_head = PhiEncoded.from_float(
            model.lm_head.weight.detach().float().numpy()
        )
        self.norm_weight = model.model.norm.weight.detach().float().numpy()
        
        print("Done!")
        del model
    
    def save(self, save_dir: str):
        """Save compressed engine to directory."""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving compressed engine to {save_dir}...")
        
        # Save embeddings
        self.embed_tokens.save(os.path.join(save_dir, 'embed_tokens.npz'))
        
        # Save layers
        for i, layer in enumerate(self.layers):
            layer_dir = os.path.join(save_dir, f'layer_{i:02d}')
            layer.save(layer_dir)
            print(f"  Saved layer {i}")
        
        # Save output head
        self.lm_head.save(os.path.join(save_dir, 'lm_head.npz'))
        
        # Save config
        np.savez(os.path.join(save_dir, 'config.npz'),
                norm_weight=self.norm_weight,
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=len(self.layers),
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mesh_rank=self.mesh_rank,
                mlp_rank=self.mlp_rank)
        
        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(save_dir):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        
        print(f"Done! Total size: {total_size / 1e9:.2f} GB")
    
    @classmethod
    def load(cls, save_dir: str) -> 'PhiCompressedEngine':
        """Load compressed engine from directory."""
        print(f"Loading compressed engine from {save_dir}...")
        
        engine = cls()
        
        # Load config
        config = np.load(os.path.join(save_dir, 'config.npz'))
        engine.vocab_size = int(config['vocab_size'])
        engine.hidden_dim = int(config['hidden_dim'])
        engine.num_heads = int(config['num_heads'])
        engine.num_kv_heads = int(config['num_kv_heads'])
        engine.head_dim = int(config['head_dim'])
        engine.mesh_rank = int(config['mesh_rank'])
        engine.mlp_rank = int(config['mlp_rank'])
        engine.norm_weight = config['norm_weight']
        num_layers = int(config['num_layers'])
        
        # Load embeddings
        engine.embed_tokens = PhiEncoded.load(os.path.join(save_dir, 'embed_tokens.npz'))
        
        # Load layers
        engine.layers = []
        for i in range(num_layers):
            layer_dir = os.path.join(save_dir, f'layer_{i:02d}')
            layer = CompressedLayer.load(layer_dir, i)
            engine.layers.append(layer)
            print(f"  Loaded layer {i}")
        
        # Load output head
        engine.lm_head = PhiEncoded.load(os.path.join(save_dir, 'lm_head.npz'))
        
        print(f"Done! Loaded {len(engine.layers)} layers.")
        return engine
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMS normalization."""
        variance = (x ** 2).mean(axis=-1, keepdims=True)
        x_normed = x / np.sqrt(variance + eps)
        return x_normed * weight
    
    def embed(self, token_ids: List[int]) -> np.ndarray:
        """Look up token embeddings."""
        embed_float = self.embed_tokens.to_float()
        return embed_float[token_ids]
    
    def forward_attention(self, hidden: np.ndarray, layer: CompressedLayer) -> np.ndarray:
        """Forward pass through attention with low-rank MESH."""
        batch_size, seq_len, hidden_dim = hidden.shape
        
        # LayerNorm
        normed = self.rms_norm(hidden, layer.ln1_weight)
        
        # V projection (low-rank)
        V = layer.W_v.matmul(normed.reshape(-1, hidden_dim)).reshape(batch_size, seq_len, -1)
        V = V + layer.b_v
        V = V.reshape(batch_size, seq_len, layer.num_kv_heads, layer.head_dim)
        V = V.transpose(0, 2, 1, 3)  # (batch, kv_heads, seq, head_dim)
        V = np.repeat(V, layer.num_heads // layer.num_kv_heads, axis=1)
        
        # Compute attention scores using low-rank MESH
        all_scores = []
        
        for head_idx in range(layer.num_heads):
            mesh = layer.mesh_qk[head_idx]
            cross_qk = layer.cross_qk[head_idx]
            cross_kq = layer.cross_kq[head_idx]
            bias = layer.bias_term[head_idx]
            
            # Efficient low-rank: input @ MESH @ input.T
            # = input @ (U @ S @ Vt) @ input.T
            # = (input @ U @ S) @ (Vt @ input.T)
            
            U = mesh.U.to_float()
            S = mesh.S.to_float()
            Vt = mesh.Vt.to_float()
            
            # (batch, seq, hidden) @ (hidden, rank) = (batch, seq, rank)
            temp1 = normed @ U
            temp1 = temp1 * S  # broadcast
            
            # (rank, hidden) @ (hidden, seq) for each batch
            temp2 = Vt @ normed.transpose(0, 2, 1)  # (batch, rank, seq)
            
            # (batch, seq, rank) @ (batch, rank, seq) = (batch, seq, seq)
            term1 = np.einsum('bsr,brt->bst', temp1, temp2)
            
            # Bias terms
            term2 = normed @ cross_qk  # (batch, seq)
            term3 = cross_kq @ normed.transpose(0, 2, 1)  # (batch, seq)
            
            scores = term1 + term2[:, :, np.newaxis] + term3[:, np.newaxis, :] + bias
            all_scores.append(scores)
        
        # Stack scores
        scores = np.stack(all_scores, axis=1)  # (batch, heads, seq, seq)
        
        # Scale
        scores = scores / np.sqrt(layer.head_dim)
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask
        
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        
        # Apply attention to V
        attn_output = np.einsum('bhqk,bhkd->bhqd', attention, V)
        
        # Reshape and output projection (low-rank)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = layer.W_o.matmul(attn_output.reshape(-1, attn_output.shape[-1]))
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_dim)
        
        return hidden + attn_output
    
    def forward_mlp(self, hidden: np.ndarray, layer: CompressedLayer) -> np.ndarray:
        """Forward pass through MLP with low-rank weights."""
        batch_size, seq_len, hidden_dim = hidden.shape
        
        # LayerNorm
        normed = self.rms_norm(hidden, layer.ln2_weight)
        normed_flat = normed.reshape(-1, hidden_dim)
        
        # Gate and Up projections (low-rank)
        gate = layer.W_gate.matmul(normed_flat)
        up = layer.W_up.matmul(normed_flat)
        
        # SiLU activation
        gate_silu = gate * (1 / (1 + np.exp(-gate)))
        mlp_hidden = gate_silu * up
        
        # Down projection (low-rank)
        mlp_output = layer.W_down.matmul(mlp_hidden)
        mlp_output = mlp_output.reshape(batch_size, seq_len, hidden_dim)
        
        return hidden + mlp_output
    
    def forward_layer(self, hidden: np.ndarray, layer: CompressedLayer) -> np.ndarray:
        """Forward pass through one layer."""
        hidden = self.forward_attention(hidden, layer)
        hidden = self.forward_mlp(hidden, layer)
        return hidden
    
    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Full forward pass."""
        # Embed
        hidden = self.embed(token_ids)
        hidden = hidden[np.newaxis, :, :]  # Add batch dimension
        
        # Layers
        for layer in self.layers:
            hidden = self.forward_layer(hidden, layer)
        
        # Final norm
        hidden = self.rms_norm(hidden, self.norm_weight)
        
        # LM head
        lm_weight = self.lm_head.to_float()
        logits = hidden @ lm_weight.T
        
        return logits


def test_compressed_engine():
    """Test the compressed engine."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("φ-COMPRESSED INFERENCE ENGINE TEST")
    print("=" * 60)
    print()
    
    # Create and save compressed engine (2 layers for quick test)
    engine = PhiCompressedEngine()
    engine.load_from_hf(max_layers=2)
    
    # Test inference
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    prompt = 'Hi'
    inputs = tokenizer(prompt, return_tensors='pt')
    token_ids = inputs['input_ids'][0].tolist()
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Token IDs: {token_ids}")
    
    print("\nRunning compressed forward pass...")
    start = time.time()
    logits = engine.forward(token_ids)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    
    # Get prediction
    top_token = int(np.argmax(logits[0, -1, :]))
    print(f"Predicted: '{tokenizer.decode([top_token])}' (id={top_token})")
    
    # Storage comparison
    print("\n" + "=" * 60)
    print("STORAGE COMPARISON")
    print("=" * 60)
    
    layer_storage = engine.layers[0].storage_bytes()
    total_storage = layer_storage * len(engine.layers)
    total_storage += engine.embed_tokens.storage_bytes()
    total_storage += engine.lm_head.storage_bytes()
    
    print(f"Per layer: {layer_storage / 1e6:.1f} MB")
    print(f"Total (2 layers): {total_storage / 1e6:.1f} MB")
    print(f"Projected (28 layers): {(layer_storage * 28 + engine.embed_tokens.storage_bytes() + engine.lm_head.storage_bytes()) / 1e9:.2f} GB")


if __name__ == "__main__":
    test_compressed_engine()
