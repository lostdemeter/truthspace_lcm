"""
Normalized Geometric Navigator - Separates learned content (signs) from universal structure (levels).

Key insight from Doc 164: Signs ARE the signal.
Key insight from Doc 128: Levels follow φ-geometry (universal).

The separation:
    weight = sign × φ^level
            ↓         ↓
        LEARNED    UNIVERSAL
        (specific)  (geometric)

This navigator normalizes model data to fit the geometric paradigm:
1. Signs encode the learned semantic content (1 bit per weight)
2. Levels follow φ-geometry (can be predicted/computed)
3. Different layer types get appropriate normalization
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
INV_PHI = 1.0 / PHI


@dataclass
class SignMatrix:
    """
    Signs only - the learned semantic content.
    
    From Doc 164: Signs ARE the signal.
    - 100% accuracy on semantic navigation
    - 16x compression (1 bit vs 16 bits)
    - O(1) lookup for known relationships
    """
    signs: np.ndarray  # int8, values in {-1, +1}
    
    @classmethod
    def from_weights(cls, W: np.ndarray) -> 'SignMatrix':
        """Extract signs from weight matrix."""
        signs = np.sign(W).astype(np.int8)
        signs[signs == 0] = 1  # Handle exact zeros
        return cls(signs=signs)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.signs.shape
    
    def hamming_distance(self, other: 'SignMatrix') -> int:
        """Compute Hamming distance (number of sign disagreements)."""
        return np.sum(self.signs != other.signs)
    
    def agreement(self, other: 'SignMatrix') -> float:
        """Fraction of signs that agree."""
        return np.mean(self.signs == other.signs)
    
    def flip(self, mask: np.ndarray) -> 'SignMatrix':
        """Flip signs at positions where mask is True."""
        new_signs = self.signs.copy()
        new_signs[mask] *= -1
        return SignMatrix(signs=new_signs)
    
    def to_bytes(self) -> bytes:
        """Pack signs into bits for storage."""
        # Pack 8 signs into 1 byte
        flat = self.signs.flatten()
        # Convert -1 to 0, +1 to 1
        bits = ((flat + 1) // 2).astype(np.uint8)
        # Pack into bytes
        n_bytes = (len(bits) + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(len(bits)):
            packed[i // 8] |= (bits[i] << (i % 8))
        return packed.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes, shape: Tuple[int, ...]) -> 'SignMatrix':
        """Unpack signs from bytes."""
        packed = np.frombuffer(data, dtype=np.uint8)
        total = int(np.prod(shape))
        bits = np.zeros(total, dtype=np.int8)
        for i in range(total):
            bit_val = int((packed[i // 8] >> (i % 8)) & 1)
            bits[i] = bit_val * 2 - 1  # 0 -> -1, 1 -> +1
        return cls(signs=bits.reshape(shape))


@dataclass
class LevelDistribution:
    """
    Level distribution - the universal φ-geometric structure.
    
    From Doc 128: Weights peak at φ^-9 (17.8%), same across all layers.
    The distribution is UNIVERSAL - we don't need to store it per-layer.
    """
    peak_level: int = -9  # φ^-9 is the characteristic scale
    
    # Standard distribution (from Doc 128)
    STANDARD_DIST = {
        -12: 0.068, -11: 0.102, -10: 0.146, -9: 0.178,
        -8: 0.140, -7: 0.046, -6: 0.003
    }
    
    @classmethod
    def from_weights(cls, W: np.ndarray) -> 'LevelDistribution':
        """Analyze weight distribution to find peak level."""
        with np.errstate(divide='ignore', invalid='ignore'):
            levels = np.round(np.log(np.abs(W) + 1e-45) / LOG_PHI).astype(int)
        
        # Find peak level
        unique, counts = np.unique(levels, return_counts=True)
        peak_idx = np.argmax(counts)
        peak_level = unique[peak_idx]
        
        return cls(peak_level=peak_level)
    
    def sample_level(self, size: int) -> np.ndarray:
        """Sample levels from the standard distribution."""
        levels = list(self.STANDARD_DIST.keys())
        probs = list(self.STANDARD_DIST.values())
        probs = np.array(probs) / sum(probs)  # Normalize
        return np.random.choice(levels, size=size, p=probs)
    
    def expected_magnitude(self) -> float:
        """Expected magnitude based on distribution."""
        total = 0.0
        for level, prob in self.STANDARD_DIST.items():
            total += prob * (PHI ** level)
        return total


class NormalizedLayer:
    """
    A layer normalized to geometric paradigm.
    
    Stores:
    - Signs (learned content) - 1 bit per weight
    - Level distribution (universal) - shared across layers
    
    For forward pass:
    - Reconstruct weights as sign × φ^level
    - Or use sign-only navigation (Doc 164)
    """
    
    def __init__(self, name: str, signs: SignMatrix, 
                 level_dist: LevelDistribution,
                 original_shape: Tuple[int, ...]):
        self.name = name
        self.signs = signs
        self.level_dist = level_dist
        self.original_shape = original_shape
    
    @classmethod
    def from_weights(cls, name: str, W: np.ndarray) -> 'NormalizedLayer':
        """Create normalized layer from weight matrix."""
        signs = SignMatrix.from_weights(W)
        level_dist = LevelDistribution.from_weights(W)
        return cls(name=name, signs=signs, level_dist=level_dist, 
                   original_shape=W.shape)
    
    def reconstruct_weights(self, use_sampled_levels: bool = False) -> np.ndarray:
        """
        Reconstruct weights from signs and level distribution.
        
        If use_sampled_levels=True, sample levels from distribution.
        Otherwise, use peak level for all (simpler, often sufficient).
        """
        if use_sampled_levels:
            levels = self.level_dist.sample_level(self.signs.signs.size)
            levels = levels.reshape(self.original_shape)
        else:
            # Use peak level for all weights
            levels = np.full(self.original_shape, self.level_dist.peak_level)
        
        return self.signs.signs.astype(np.float32) * (PHI ** levels)
    
    def storage_bytes(self) -> int:
        """Storage size in bytes."""
        # Signs: 1 bit per weight
        sign_bytes = (self.signs.signs.size + 7) // 8
        # Level distribution: just the peak level (1 byte)
        return sign_bytes + 1
    
    def compression_ratio(self) -> float:
        """Compression vs float32."""
        original = self.signs.signs.size * 4  # float32
        compressed = self.storage_bytes()
        return original / compressed


class NormalizedNavigator:
    """
    Navigator using normalized geometric representation.
    
    Key principles:
    1. Signs ARE the signal (Doc 164)
    2. Levels follow φ-geometry (Doc 128)
    3. Different layer types get appropriate treatment
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/normalized_navigator")
        
        # Normalized layers
        self.embedding_signs: Optional[SignMatrix] = None
        self.lm_head_signs: Optional[SignMatrix] = None
        self.layers: List[Dict[str, NormalizedLayer]] = []
        
        # Universal level distribution (shared)
        self.level_dist = LevelDistribution()
        
        # Tokenizer
        self.tokenizer = None
        
        # Config
        self.hidden_dim = None
        self.vocab_size = None
    
    def convert_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct", 
                           max_layers: int = 28):
        """Convert model to normalized geometric representation."""
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
        
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Convert embeddings (signs only)
        print("Normalizing embeddings...")
        emb_weight = model.model.embed_tokens.weight.data.numpy()
        self.embedding_signs = SignMatrix.from_weights(emb_weight)
        emb_level_dist = LevelDistribution.from_weights(emb_weight)
        print(f"  Embeddings: peak level = φ^{emb_level_dist.peak_level}")
        print(f"  Compression: {emb_weight.nbytes / ((emb_weight.size + 7) // 8):.1f}x")
        
        # Convert LM head
        print("Normalizing LM head...")
        lm_weight = model.lm_head.weight.data.numpy()
        self.lm_head_signs = SignMatrix.from_weights(lm_weight)
        lm_level_dist = LevelDistribution.from_weights(lm_weight)
        print(f"  LM head: peak level = φ^{lm_level_dist.peak_level}")
        
        # Convert layers
        n_layers = min(max_layers, len(model.model.layers))
        print(f"Normalizing {n_layers} layers...")
        
        total_original = 0
        total_compressed = 0
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            layer_dict = {}
            
            # Attention weights
            for name, proj in [
                ('q_proj', layer.self_attn.q_proj),
                ('k_proj', layer.self_attn.k_proj),
                ('v_proj', layer.self_attn.v_proj),
                ('o_proj', layer.self_attn.o_proj),
            ]:
                W = proj.weight.data.numpy()
                normalized = NormalizedLayer.from_weights(f"layer{layer_idx}.{name}", W)
                layer_dict[name] = normalized
                total_original += W.nbytes
                total_compressed += normalized.storage_bytes()
            
            # MLP weights
            for name, proj in [
                ('gate_proj', layer.mlp.gate_proj),
                ('up_proj', layer.mlp.up_proj),
                ('down_proj', layer.mlp.down_proj),
            ]:
                W = proj.weight.data.numpy()
                normalized = NormalizedLayer.from_weights(f"layer{layer_idx}.{name}", W)
                layer_dict[name] = normalized
                total_original += W.nbytes
                total_compressed += normalized.storage_bytes()
            
            # LayerNorm weights (keep as float - they're small)
            layer_dict['input_layernorm'] = layer.input_layernorm.weight.data.numpy()
            layer_dict['post_attention_layernorm'] = layer.post_attention_layernorm.weight.data.numpy()
            
            self.layers.append(layer_dict)
            
            if layer_idx % 5 == 0:
                print(f"  Layer {layer_idx}: normalized")
        
        print(f"\nTotal compression: {total_original / total_compressed:.1f}x")
        print(f"  Original: {total_original / 1e9:.2f} GB")
        print(f"  Compressed: {total_compressed / 1e9:.2f} GB")
        
        del model
    
    def get_embedding_signs(self, token_id: int) -> SignMatrix:
        """Get sign pattern for a token."""
        return SignMatrix(signs=self.embedding_signs.signs[token_id].copy())
    
    def find_nearest_token_by_signs(self, target_signs: SignMatrix) -> int:
        """Find token with most similar sign pattern (Hamming distance)."""
        # Compute agreement with all tokens
        agreement = np.sum(self.embedding_signs.signs == target_signs.signs, axis=1)
        return int(np.argmax(agreement))
    
    def navigate_by_sign_flip(self, token_id: int, flip_mask: np.ndarray) -> int:
        """
        Navigate to new token by flipping signs.
        
        This is the core of Doc 164's sign-only navigation.
        """
        source_signs = self.get_embedding_signs(token_id)
        target_signs = source_signs.flip(flip_mask)
        return self.find_nearest_token_by_signs(target_signs)
    
    def forward_layer_signs(self, input_signs: SignMatrix, layer_idx: int) -> SignMatrix:
        """
        Forward pass through layer using sign-only computation.
        
        Key insight: For sign-only navigation, we only need to track
        which signs flip through the transformation.
        
        This is a simplified version - full implementation would
        properly handle the bilinear structure of MLP (Doc 132).
        
        Weight shapes:
        - gate_proj: (intermediate_size, hidden_size) e.g. (18944, 3584)
        - up_proj: (intermediate_size, hidden_size) e.g. (18944, 3584)
        - down_proj: (hidden_size, intermediate_size) e.g. (3584, 18944)
        """
        layer = self.layers[layer_idx]
        
        # Get weight signs - shapes are (out_dim, in_dim)
        gate_signs = layer['gate_proj'].signs.signs  # (18944, 3584)
        up_signs = layer['up_proj'].signs.signs      # (18944, 3584)
        down_signs = layer['down_proj'].signs.signs  # (3584, 18944)
        
        # Input: (3584,)
        # Simplified sign propagation:
        # output_sign[i] = majority vote of (input_sign[j] * weight_sign[i,j])
        
        # Gate projection: (18944, 3584) @ (3584,) -> (18944,)
        # For each output dim, sum(input_sign * weight_sign) and take sign
        gate_votes = input_signs.signs[np.newaxis, :] * gate_signs  # (18944, 3584)
        gate_output = np.sign(np.sum(gate_votes, axis=1)).astype(np.int8)  # (18944,)
        gate_output[gate_output == 0] = 1
        
        # Up projection: same shape
        up_votes = input_signs.signs[np.newaxis, :] * up_signs  # (18944, 3584)
        up_output = np.sign(np.sum(up_votes, axis=1)).astype(np.int8)  # (18944,)
        up_output[up_output == 0] = 1
        
        # Gate * Up (sign multiplication) - element-wise
        hidden_signs = gate_output * up_output  # (18944,)
        
        # Down projection: (3584, 18944) @ (18944,) -> (3584,)
        down_votes = hidden_signs[np.newaxis, :] * down_signs  # (3584, 18944)
        output_signs = np.sign(np.sum(down_votes, axis=1)).astype(np.int8)  # (3584,)
        output_signs[output_signs == 0] = 1
        
        return SignMatrix(signs=output_signs)
    
    def navigate(self, token_ids: List[int]) -> SignMatrix:
        """Navigate through model using sign-only computation."""
        # Start with last token's signs
        position_signs = self.get_embedding_signs(token_ids[-1])
        
        # Navigate through layers
        for layer_idx in range(len(self.layers)):
            layer_output = self.forward_layer_signs(position_signs, layer_idx)
            # Simplified residual: XOR-like combination
            # (In full implementation, would handle this more carefully)
            position_signs = SignMatrix(
                signs=np.sign(position_signs.signs.astype(np.int16) + 
                             layer_output.signs.astype(np.int16)).astype(np.int8)
            )
            position_signs.signs[position_signs.signs == 0] = 1
        
        return position_signs
    
    def predict_next_token(self, token_ids: List[int]) -> int:
        """Predict next token via sign-only navigation."""
        output_signs = self.navigate(token_ids)
        return self.find_nearest_token_by_signs(output_signs)
    
    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text using sign-only navigation."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        for _ in range(max_tokens):
            next_token = self.predict_next_token(token_ids)
            token_ids.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(token_ids)


def test_sign_matrix():
    """Test SignMatrix operations."""
    print("=" * 60)
    print("Testing SignMatrix")
    print("=" * 60)
    
    # Create from weights
    W = np.random.randn(100, 200).astype(np.float32)
    signs = SignMatrix.from_weights(W)
    
    print(f"Shape: {signs.shape}")
    print(f"Unique values: {np.unique(signs.signs)}")
    
    # Test packing/unpacking
    packed = signs.to_bytes()
    unpacked = SignMatrix.from_bytes(packed, signs.shape)
    
    print(f"Pack/unpack match: {np.all(signs.signs == unpacked.signs)}")
    print(f"Compression: {W.nbytes / len(packed):.1f}x")
    
    # Test Hamming distance
    signs2 = signs.flip(np.random.rand(*signs.shape) > 0.5)
    print(f"Hamming distance after random flip: {signs.hamming_distance(signs2)}")
    print(f"Agreement: {signs.agreement(signs2):.2%}")
    print()


def test_normalized_layer():
    """Test NormalizedLayer."""
    print("=" * 60)
    print("Testing NormalizedLayer")
    print("=" * 60)
    
    # Create weight matrix with φ^-9 peak (like real model)
    np.random.seed(42)
    levels = np.random.choice([-11, -10, -9, -8, -7], size=(256, 512),
                               p=[0.1, 0.15, 0.5, 0.15, 0.1])
    signs = np.random.choice([-1, 1], size=(256, 512))
    W = (signs * (PHI ** levels)).astype(np.float32)
    
    # Normalize
    layer = NormalizedLayer.from_weights("test_layer", W)
    
    print(f"Original shape: {W.shape}")
    print(f"Peak level: φ^{layer.level_dist.peak_level}")
    print(f"Compression: {layer.compression_ratio():.1f}x")
    
    # Reconstruct
    W_reconstructed = layer.reconstruct_weights(use_sampled_levels=False)
    
    # Check sign preservation
    sign_match = np.mean(np.sign(W) == np.sign(W_reconstructed))
    print(f"Sign preservation: {sign_match:.2%}")
    
    # The magnitudes won't match exactly (we're using peak level for all)
    # but the SIGNS are preserved perfectly
    print()


def test_sign_navigation():
    """Test sign-only navigation concept."""
    print("=" * 60)
    print("Testing Sign-Only Navigation Concept")
    print("=" * 60)
    
    # Simulate embeddings
    np.random.seed(42)
    vocab_size = 1000
    hidden_dim = 256
    
    # Create random embeddings
    embeddings = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    embedding_signs = SignMatrix.from_weights(embeddings)
    
    # Test: can we find a token by its sign pattern?
    target_token = 42
    target_signs = SignMatrix(signs=embedding_signs.signs[target_token].copy())
    
    # Find by Hamming distance
    agreement = np.sum(embedding_signs.signs == target_signs.signs, axis=1)
    found_token = np.argmax(agreement)
    
    print(f"Target token: {target_token}")
    print(f"Found token: {found_token}")
    print(f"Match: {target_token == found_token}")
    
    # Test: flip some signs and find nearest
    flip_mask = np.random.rand(hidden_dim) > 0.9  # Flip ~10% of signs
    flipped_signs = target_signs.flip(flip_mask)
    
    agreement_flipped = np.sum(embedding_signs.signs == flipped_signs.signs, axis=1)
    found_after_flip = np.argmax(agreement_flipped)
    
    print(f"\nAfter flipping {np.sum(flip_mask)} signs:")
    print(f"Found token: {found_after_flip}")
    print(f"Agreement with original: {agreement_flipped[target_token] / hidden_dim:.2%}")
    print(f"Agreement with found: {agreement_flipped[found_after_flip] / hidden_dim:.2%}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        navigator = NormalizedNavigator()
        navigator.convert_from_model(max_layers=2)
        
        prompt = "Hello"
        print(f"\nPrompt: {prompt}")
        output = navigator.generate(prompt, max_tokens=10)
        print(f"Output: {output}")
    else:
        test_sign_matrix()
        test_normalized_layer()
        test_sign_navigation()
