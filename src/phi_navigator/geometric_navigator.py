"""
Geometric Navigator - Pure φ-geometric navigation without transformer forward pass.

This replaces the standard transformer architecture with:
1. φ-Coordinates instead of float vectors
2. MESH navigation instead of attention
3. φ-Level transforms instead of MLP
4. Position in manifold instead of hidden states

Based on Design Consideration 165.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
INV_PHI = 1.0 / PHI  # 0.618...


@dataclass
class PhiCoordinate:
    """
    Position in φ-manifold.
    
    Represents a vector as signs × φ^levels, enabling:
    - Integer arithmetic for multiplication (level addition)
    - Geometric interpretation (position in semantic space)
    """
    signs: np.ndarray   # (dim,) int8, values in {-1, 0, +1}
    levels: np.ndarray  # (dim,) int16, φ-exponents
    
    @classmethod
    def from_float(cls, x: np.ndarray) -> 'PhiCoordinate':
        """Encode float array as φ-coordinate."""
        signs = np.sign(x).astype(np.int8)
        # Handle zeros: set level to very negative (effectively 0)
        with np.errstate(divide='ignore'):
            levels = np.round(np.log(np.abs(x) + 1e-45) / LOG_PHI).astype(np.int16)
        levels[signs == 0] = -100  # Very small
        return cls(signs=signs, levels=levels)
    
    def to_float(self) -> np.ndarray:
        """Decode to float (for validation)."""
        return self.signs.astype(np.float32) * (PHI ** self.levels.astype(np.float32))
    
    @property
    def dim(self) -> int:
        return len(self.signs)
    
    def phi_add(self, other: 'PhiCoordinate') -> 'PhiCoordinate':
        """
        Add two φ-coordinates (for residual connections).
        
        This is the tricky operation - addition in log-space.
        We use the dominant term approximation for efficiency.
        """
        # Decode, add, re-encode (exact but slower)
        # For now, use this approach; optimize later
        result = self.to_float() + other.to_float()
        return PhiCoordinate.from_float(result)
    
    def phi_distance(self, other: 'PhiCoordinate') -> float:
        """Compute distance in φ-manifold."""
        # L2 distance in decoded space
        diff = self.to_float() - other.to_float()
        return np.sqrt(np.sum(diff ** 2))
    
    def dot(self, other: 'PhiCoordinate') -> float:
        """Dot product in φ-space."""
        # sign1 * sign2 * φ^(level1 + level2)
        combined_signs = self.signs * other.signs
        combined_levels = self.levels + other.levels
        
        # Sum of φ^levels with signs
        # This requires log-sum-exp for accuracy
        # For now, decode and compute
        return np.sum(self.to_float() * other.to_float())


@dataclass 
class PhiTransform:
    """
    Transformation in φ-space (replaces weight matrix).
    
    Represents W[i,j] = signs[i,j] × φ^levels[i,j]
    """
    signs: np.ndarray   # (out_dim, in_dim) int8
    levels: np.ndarray  # (out_dim, in_dim) int16
    
    @classmethod
    def from_float(cls, W: np.ndarray) -> 'PhiTransform':
        """Encode weight matrix as φ-transform."""
        signs = np.sign(W).astype(np.int8)
        signs[signs == 0] = 1  # Handle zeros
        with np.errstate(divide='ignore'):
            levels = np.round(np.log(np.abs(W) + 1e-45) / LOG_PHI).astype(np.int16)
        return cls(signs=signs, levels=levels)
    
    def to_float(self) -> np.ndarray:
        """Decode to float matrix (for validation)."""
        return self.signs.astype(np.float32) * (PHI ** self.levels.astype(np.float32))
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.signs.shape
    
    def apply(self, coord: PhiCoordinate) -> PhiCoordinate:
        """
        Apply transformation to coordinate: output = W @ input
        
        In φ-space:
        - Multiplication = level addition
        - Accumulation = log-sum-exp (approximated)
        """
        # For now, decode and compute (exact but slower)
        W = self.to_float()
        x = coord.to_float()
        result = W @ x
        return PhiCoordinate.from_float(result)
    
    def apply_transposed(self, coord: PhiCoordinate) -> PhiCoordinate:
        """Apply transposed transformation: output = W.T @ input"""
        W = self.to_float()
        x = coord.to_float()
        result = W.T @ x
        return PhiCoordinate.from_float(result)


class MESHNavigator:
    """
    MESH-based attention navigation.
    
    MESH = U @ diag(S) @ Vt encodes the Q/K relationship geometrically.
    Navigation through MESH projects positions through semantic space.
    """
    
    def __init__(self, U: PhiTransform, S_levels: np.ndarray, Vt: PhiTransform):
        """
        Args:
            U: Left singular vectors as φ-transform (hidden_dim, rank)
            S_levels: Singular values as φ-levels (rank,) - follows φ-Zipf
            Vt: Right singular vectors as φ-transform (rank, hidden_dim)
        """
        self.U = U
        self.S_levels = S_levels  # Just levels, signs are always +1
        self.Vt = Vt
        self.rank = len(S_levels)
    
    @classmethod
    def from_mesh(cls, MESH: np.ndarray, rank: int = 128) -> 'MESHNavigator':
        """Create from MESH matrix via SVD."""
        U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Encode as φ-transforms
        U_phi = PhiTransform.from_float(U)
        S_levels = np.round(np.log(S + 1e-45) / LOG_PHI).astype(np.int16)
        Vt_phi = PhiTransform.from_float(Vt)
        
        return cls(U_phi, S_levels, Vt_phi)
    
    def navigate(self, position: PhiCoordinate) -> PhiCoordinate:
        """
        Navigate through MESH structure.
        
        MESH @ x = U @ diag(S) @ Vt @ x
        
        Steps:
        1. Project through Vt: z = Vt @ x  (rank,)
        2. Scale by S: z_scaled = S * z
        3. Project through U: result = U @ z_scaled
        """
        # Step 1: Project through Vt (Vt is rank x hidden_dim)
        z = self.Vt.apply(position)  # Vt @ x -> (rank,)
        
        # Step 2: Scale by singular values (level addition in φ-space)
        # S is stored as levels, so multiply = add levels
        scaled_levels = z.levels + self.S_levels
        z_scaled = PhiCoordinate(signs=z.signs, levels=scaled_levels)
        
        # Step 3: Project through U (U is hidden_dim x rank)
        result = self.U.apply(z_scaled)  # U @ z_scaled -> (hidden_dim,)
        
        return result


class PhiLevelMLP:
    """
    φ-Level MLP transformation.
    
    Replaces: hidden = SiLU(x @ W_gate.T) * (x @ W_up.T); output = hidden @ W_down.T
    With: φ-level arithmetic where SiLU(x) ≈ x/2 (level - 1)
    """
    
    def __init__(self, W_gate: PhiTransform, W_up: PhiTransform, W_down: PhiTransform):
        self.W_gate = W_gate
        self.W_up = W_up
        self.W_down = W_down
    
    @classmethod
    def from_weights(cls, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray) -> 'PhiLevelMLP':
        """Create from float weight matrices."""
        return cls(
            W_gate=PhiTransform.from_float(W_gate),
            W_up=PhiTransform.from_float(W_up),
            W_down=PhiTransform.from_float(W_down),
        )
    
    def transform(self, position: PhiCoordinate, linearized: bool = True) -> PhiCoordinate:
        """
        Apply MLP transformation.
        
        Args:
            position: Input position
            linearized: If True, use SiLU(x) ≈ x/2 for integer path
        """
        # Gate and Up projections
        gate = self.W_gate.apply(position)
        up = self.W_up.apply(position)
        
        if linearized:
            # SiLU(x) ≈ x/2 → level - 1 (since φ^-1 ≈ 0.618 ≈ 1/2)
            # hidden = (gate/2) * up
            # In φ-space: combined_level = gate_level - 1 + up_level
            
            # Element-wise: sign = gate_sign * up_sign, level = gate_level + up_level - 1
            hidden_signs = gate.signs * up.signs
            hidden_levels = gate.levels + up.levels - 1  # -1 for the /2
            hidden = PhiCoordinate(signs=hidden_signs, levels=hidden_levels)
        else:
            # Full SiLU - decode, compute, re-encode
            gate_f = gate.to_float()
            up_f = up.to_float()
            hidden_f = (gate_f / (1 + np.exp(-gate_f))) * up_f  # SiLU(gate) * up
            hidden = PhiCoordinate.from_float(hidden_f)
        
        # Down projection
        output = self.W_down.apply(hidden)
        
        return output


class GeometricLayer:
    """
    One layer of geometric navigation (attention + MLP).
    """
    
    def __init__(self, mesh_navigators: List[MESHNavigator], mlp: PhiLevelMLP,
                 ln1_weight: PhiCoordinate, ln2_weight: PhiCoordinate,
                 W_v: PhiTransform, W_o: PhiTransform):
        self.mesh_navigators = mesh_navigators  # One per head
        self.mlp = mlp
        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight
        self.W_v = W_v
        self.W_o = W_o
        self.num_heads = len(mesh_navigators)
    
    def navigate(self, position: PhiCoordinate) -> PhiCoordinate:
        """Navigate through this layer."""
        # LayerNorm 1 (simplified: just scale by weight)
        normed1 = self._rms_norm(position, self.ln1_weight)
        
        # Attention via MESH navigation
        # For single position, we navigate through each head's MESH
        attn_outputs = []
        for mesh in self.mesh_navigators:
            head_out = mesh.navigate(normed1)
            attn_outputs.append(head_out)
        
        # Combine heads (simplified: average for now)
        # In full implementation, would concatenate and project through W_o
        combined = attn_outputs[0]
        for head_out in attn_outputs[1:]:
            combined = combined.phi_add(head_out)
        
        # Residual connection
        position = position.phi_add(combined)
        
        # LayerNorm 2
        normed2 = self._rms_norm(position, self.ln2_weight)
        
        # MLP
        mlp_out = self.mlp.transform(normed2)
        
        # Residual connection
        position = position.phi_add(mlp_out)
        
        return position
    
    def _rms_norm(self, coord: PhiCoordinate, weight: PhiCoordinate) -> PhiCoordinate:
        """RMS normalization in φ-space."""
        # Decode, normalize, re-encode (for now)
        x = coord.to_float()
        w = weight.to_float()
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        normed = (x / rms) * w
        return PhiCoordinate.from_float(normed)


class GeometricNavigator:
    """
    Pure geometric navigation engine.
    
    Replaces the transformer forward pass with φ-geometric operations.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/phi_navigation_geometric")
        
        # Embeddings as φ-coordinates
        self.embeddings: Optional[List[PhiCoordinate]] = None
        
        # Layers
        self.layers: List[GeometricLayer] = []
        
        # Final norm and LM head
        self.final_norm_weight: Optional[PhiCoordinate] = None
        self.lm_head: Optional[PhiTransform] = None
        
        # Tokenizer
        self.tokenizer = None
    
    def convert_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct", max_layers: int = 28):
        """Convert a transformer model to geometric representation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Convert embeddings
        print("Converting embeddings...")
        emb_weight = model.model.embed_tokens.weight.data.numpy()
        self.embeddings = [
            PhiCoordinate.from_float(emb_weight[i])
            for i in range(emb_weight.shape[0])
        ]
        
        # Convert LM head
        print("Converting LM head...")
        lm_weight = model.lm_head.weight.data.numpy()
        self.lm_head = PhiTransform.from_float(lm_weight)
        
        # Convert final norm
        norm_weight = model.model.norm.weight.data.numpy()
        self.final_norm_weight = PhiCoordinate.from_float(norm_weight)
        
        # Convert layers
        n_layers = min(max_layers, len(model.model.layers))
        print(f"Converting {n_layers} layers...")
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            
            # Extract attention weights
            W_q = layer.self_attn.q_proj.weight.data.numpy()
            W_k = layer.self_attn.k_proj.weight.data.numpy()
            W_v = layer.self_attn.v_proj.weight.data.numpy()
            W_o = layer.self_attn.o_proj.weight.data.numpy()
            
            # Compute MESH = W_q.T @ W_k for each head
            num_heads = 28
            num_kv_heads = 4
            head_dim = 128
            heads_per_kv = num_heads // num_kv_heads
            
            mesh_navigators = []
            for h in range(num_heads):
                kv_idx = h // heads_per_kv
                W_q_h = W_q[h * head_dim:(h + 1) * head_dim, :]
                W_k_h = W_k[kv_idx * head_dim:(kv_idx + 1) * head_dim, :]
                MESH = W_q_h.T @ W_k_h  # (hidden_dim, hidden_dim)
                mesh_nav = MESHNavigator.from_mesh(MESH, rank=64)  # Reduced rank for efficiency
                mesh_navigators.append(mesh_nav)
            
            # Extract MLP weights
            W_gate = layer.mlp.gate_proj.weight.data.numpy()
            W_up = layer.mlp.up_proj.weight.data.numpy()
            W_down = layer.mlp.down_proj.weight.data.numpy()
            mlp = PhiLevelMLP.from_weights(W_gate, W_up, W_down)
            
            # LayerNorm weights
            ln1_weight = PhiCoordinate.from_float(layer.input_layernorm.weight.data.numpy())
            ln2_weight = PhiCoordinate.from_float(layer.post_attention_layernorm.weight.data.numpy())
            
            # Create geometric layer
            geo_layer = GeometricLayer(
                mesh_navigators=mesh_navigators,
                mlp=mlp,
                ln1_weight=ln1_weight,
                ln2_weight=ln2_weight,
                W_v=PhiTransform.from_float(W_v),
                W_o=PhiTransform.from_float(W_o),
            )
            self.layers.append(geo_layer)
            print(f"  Layer {layer_idx} converted")
        
        print(f"Converted {n_layers} layers to geometric representation")
        
        # Clean up
        del model
    
    def navigate(self, token_ids: List[int]) -> PhiCoordinate:
        """
        Navigate from input tokens to output position.
        
        This is the core geometric operation - no hidden states,
        just traversal through the φ-manifold.
        """
        # Get starting position (last token's embedding)
        position = self.embeddings[token_ids[-1]]
        
        # For sequence, we need to aggregate context
        # Simplified: use last token only for now
        # Full implementation would handle full sequence
        
        # Navigate through layers
        for layer in self.layers:
            position = layer.navigate(position)
        
        # Final norm
        position = self._rms_norm(position, self.final_norm_weight)
        
        return position
    
    def predict_next_token(self, token_ids: List[int], temperature: float = 0.0) -> int:
        """Predict next token via geometric navigation."""
        # Navigate to output position
        position = self.navigate(token_ids)
        
        # Find nearest token in embedding space
        # This is: argmax(position @ embeddings.T) = argmax(lm_head @ position)
        logits = self.lm_head.apply(position).to_float()
        
        if temperature == 0:
            return int(np.argmax(logits))
        else:
            # Softmax sampling
            logits = logits / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            return int(np.random.choice(len(probs), p=probs))
    
    def _rms_norm(self, coord: PhiCoordinate, weight: PhiCoordinate) -> PhiCoordinate:
        """RMS normalization."""
        x = coord.to_float()
        w = weight.to_float()
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        normed = (x / rms) * w
        return PhiCoordinate.from_float(normed)
    
    def generate(self, prompt: str, max_tokens: int = 20, temperature: float = 0.0) -> str:
        """Generate text from prompt."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        for _ in range(max_tokens):
            next_token = self.predict_next_token(token_ids, temperature)
            token_ids.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(token_ids)


def test_phi_coordinate():
    """Test PhiCoordinate encoding/decoding."""
    print("=" * 60)
    print("Testing PhiCoordinate")
    print("=" * 60)
    
    # Random vector
    x = np.random.randn(100).astype(np.float32)
    
    # Encode and decode
    coord = PhiCoordinate.from_float(x)
    x_decoded = coord.to_float()
    
    # Check correlation
    corr = np.corrcoef(x.flatten(), x_decoded.flatten())[0, 1]
    mse = np.mean((x - x_decoded) ** 2)
    
    print(f"Original shape: {x.shape}")
    print(f"Correlation: {corr:.6f}")
    print(f"MSE: {mse:.2e}")
    print()


def test_phi_transform():
    """Test PhiTransform encoding/decoding."""
    print("=" * 60)
    print("Testing PhiTransform")
    print("=" * 60)
    
    # Random matrix
    W = np.random.randn(64, 128).astype(np.float32) * 0.1
    
    # Encode and decode
    transform = PhiTransform.from_float(W)
    W_decoded = transform.to_float()
    
    # Check correlation
    corr = np.corrcoef(W.flatten(), W_decoded.flatten())[0, 1]
    mse = np.mean((W - W_decoded) ** 2)
    
    print(f"Original shape: {W.shape}")
    print(f"Correlation: {corr:.6f}")
    print(f"MSE: {mse:.2e}")
    
    # Test apply
    x = np.random.randn(128).astype(np.float32)
    coord = PhiCoordinate.from_float(x)
    
    # Original matmul
    y_original = W @ x
    
    # φ-transform
    y_coord = transform.apply(coord)
    y_phi = y_coord.to_float()
    
    corr_apply = np.corrcoef(y_original.flatten(), y_phi.flatten())[0, 1]
    print(f"Apply correlation: {corr_apply:.6f}")
    print()


def test_mesh_navigator():
    """Test MESHNavigator."""
    print("=" * 60)
    print("Testing MESHNavigator")
    print("=" * 60)
    
    # Create random MESH
    hidden_dim = 256
    MESH = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.01
    
    # Test with different ranks
    for rank in [32, 64, 128, 256]:
        navigator = MESHNavigator.from_mesh(MESH, rank=min(rank, hidden_dim))
        
        # Test navigation
        x = np.random.randn(hidden_dim).astype(np.float32)
        position = PhiCoordinate.from_float(x)
        
        # Original: MESH @ x
        y_original = MESH @ x
        
        # Geometric navigation
        y_coord = navigator.navigate(position)
        y_phi = y_coord.to_float()
        
        corr = np.corrcoef(y_original.flatten(), y_phi.flatten())[0, 1]
        print(f"Rank {rank}: correlation = {corr:.4f}")
    print()


def test_phi_level_mlp():
    """Test PhiLevelMLP."""
    print("=" * 60)
    print("Testing PhiLevelMLP")
    print("=" * 60)
    
    # Create random MLP weights
    in_dim = 128
    hidden_dim = 256
    
    W_gate = np.random.randn(hidden_dim, in_dim).astype(np.float32) * 0.1
    W_up = np.random.randn(hidden_dim, in_dim).astype(np.float32) * 0.1
    W_down = np.random.randn(in_dim, hidden_dim).astype(np.float32) * 0.1
    
    mlp = PhiLevelMLP.from_weights(W_gate, W_up, W_down)
    
    # Test input
    x = np.random.randn(in_dim).astype(np.float32)
    position = PhiCoordinate.from_float(x)
    
    # Original MLP (linearized)
    gate = W_gate @ x
    up = W_up @ x
    hidden = (gate / 2) * up  # Linearized SiLU
    y_original = W_down @ hidden
    
    # φ-Level MLP
    y_coord = mlp.transform(position, linearized=True)
    y_phi = y_coord.to_float()
    
    corr = np.corrcoef(y_original.flatten(), y_phi.flatten())[0, 1]
    print(f"Input dim: {in_dim}, Hidden dim: {hidden_dim}")
    print(f"MLP correlation (linearized): {corr:.6f}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        # Convert model to geometric representation
        navigator = GeometricNavigator()
        navigator.convert_from_model(max_layers=2)  # Start with 2 layers for testing
        
        # Test generation
        prompt = "Hello"
        print(f"\nPrompt: {prompt}")
        output = navigator.generate(prompt, max_tokens=10)
        print(f"Output: {output}")
    else:
        # Run tests
        test_phi_coordinate()
        test_phi_transform()
        test_mesh_navigator()
        test_phi_level_mlp()
