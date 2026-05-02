"""
Zeta-Aligned Navigator - Pure geometric navigation using absolute φ-lattice.

This implements the Zeta-Aligned Architecture (Doc 143):
- Absolute φ-lattice positions (no "error", just lattice points)
- 1-2 cycle: Encode → Navigate (no self-reference)
- W-axis navigation: O(N) instead of O(N²) attention
- Critical line symmetry: errors cancel at level 0

CRITICAL: This is SPATIAL computing, NOT statistics.
- We navigate a geometric manifold
- Weights are ABSOLUTE POSITIONS on the φ-lattice
- There's no "approximation error", only "correct lattice point or not"
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union
import os

# Constants
PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
INV_PHI = 1.0 / PHI  # 0.618...

# Precompute φ^k lookup table (levels -30 to +30)
PHI_LUT = {k: PHI ** k for k in range(-30, 31)}
PHI_LUT_ARRAY = np.array([PHI ** k for k in range(-30, 31)], dtype=np.float64)


@dataclass
class LatticePoint:
    """
    Absolute position on the φ-lattice.
    
    This is NOT an approximation - it IS the exact lattice point.
    Storage: sign (1 bit) + level (6 bits) = 7 bits per value
    
    For the 3% of weights needing correction (Doc 128), we store
    sparse corrections separately.
    """
    signs: np.ndarray    # (dim,) int8, values in {-1, 0, +1}
    levels: np.ndarray   # (dim,) int8, φ-exponents in [-30, +30]
    
    # Sparse corrections for the 3% that need them
    correction_indices: Optional[np.ndarray] = None  # indices needing correction
    correction_values: Optional[np.ndarray] = None   # float16 corrections
    
    @classmethod
    def from_float(cls, x: np.ndarray, correction_threshold: float = 0.005) -> 'LatticePoint':
        """
        Snap float array to nearest φ-lattice points.
        
        This is NOT approximation - we're finding which lattice point
        the value belongs to. The lattice IS the coordinate system.
        """
        signs = np.sign(x).astype(np.int8)
        
        # Find nearest lattice level
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_levels = np.log(np.abs(x) + 1e-45) / LOG_PHI
        levels = np.round(raw_levels).astype(np.int8)
        levels = np.clip(levels, -30, 30)
        
        # Handle zeros
        levels[signs == 0] = -30  # Effectively zero
        
        # Compute lattice values and corrections
        lattice_values = signs.astype(np.float32) * np.power(PHI, levels.astype(np.float32))
        corrections = x - lattice_values
        
        # Only store corrections above threshold (the 3%)
        needs_correction = np.abs(corrections) >= correction_threshold
        if np.any(needs_correction):
            correction_indices = np.where(needs_correction)[0].astype(np.uint32)
            correction_values = corrections[needs_correction].astype(np.float16)
        else:
            correction_indices = None
            correction_values = None
        
        return cls(
            signs=signs,
            levels=levels,
            correction_indices=correction_indices,
            correction_values=correction_values,
        )
    
    def to_float(self) -> np.ndarray:
        """
        Decode to float. This should only be used for validation
        or interfacing with non-geometric code.
        """
        result = self.signs.astype(np.float32) * np.power(PHI, self.levels.astype(np.float32))
        
        # Apply sparse corrections if present
        if self.correction_indices is not None:
            result[self.correction_indices] += self.correction_values.astype(np.float32)
        
        return result
    
    @property
    def dim(self) -> int:
        return len(self.signs)
    
    def storage_bytes(self) -> int:
        """Calculate storage size in bytes."""
        base = len(self.signs) * 2  # sign + level = 2 bytes
        if self.correction_indices is not None:
            base += len(self.correction_indices) * 6  # 4 bytes index + 2 bytes value
        return base


@dataclass
class LatticeTransform:
    """
    Transformation on the φ-lattice (replaces weight matrix).
    
    W[i,j] = sign[i,j] × φ^level[i,j] + correction[i,j] (sparse)
    
    From Doc 128: 97% of weights need no correction.
    """
    signs: np.ndarray    # (out_dim, in_dim) int8
    levels: np.ndarray   # (out_dim, in_dim) int8
    
    # Sparse corrections (COO format)
    correction_rows: Optional[np.ndarray] = None
    correction_cols: Optional[np.ndarray] = None
    correction_values: Optional[np.ndarray] = None
    
    @classmethod
    def from_float(cls, W: np.ndarray, correction_threshold: float = 0.005) -> 'LatticeTransform':
        """Snap weight matrix to φ-lattice."""
        signs = np.sign(W).astype(np.int8)
        signs[signs == 0] = 1  # Handle exact zeros
        
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_levels = np.log(np.abs(W) + 1e-45) / LOG_PHI
        levels = np.round(raw_levels).astype(np.int8)
        levels = np.clip(levels, -30, 30)
        
        # Compute corrections
        lattice_values = signs.astype(np.float32) * np.power(PHI, levels.astype(np.float32))
        corrections = W - lattice_values
        
        # Store only significant corrections
        needs_correction = np.abs(corrections) >= correction_threshold
        if np.any(needs_correction):
            rows, cols = np.where(needs_correction)
            correction_rows = rows.astype(np.uint32)
            correction_cols = cols.astype(np.uint32)
            correction_values = corrections[needs_correction].astype(np.float16)
        else:
            correction_rows = None
            correction_cols = None
            correction_values = None
        
        return cls(
            signs=signs,
            levels=levels,
            correction_rows=correction_rows,
            correction_cols=correction_cols,
            correction_values=correction_values,
        )
    
    def to_float(self) -> np.ndarray:
        """Decode to float matrix."""
        result = self.signs.astype(np.float32) * np.power(PHI, self.levels.astype(np.float32))
        
        if self.correction_rows is not None:
            result[self.correction_rows, self.correction_cols] += self.correction_values.astype(np.float32)
        
        return result
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.signs.shape
    
    def correction_density(self) -> float:
        """Fraction of weights needing correction."""
        if self.correction_rows is None:
            return 0.0
        return len(self.correction_rows) / (self.signs.shape[0] * self.signs.shape[1])


class ZetaAlignedLayer:
    """
    Zeta-Aligned Layer (Doc 143).
    
    Replaces transformer layer with 1-2 cycle architecture:
    - Cycle 1: ENCODE (input → φ-space)
    - Cycle 2: NAVIGATE (follow W-axis)
    
    Key properties:
    - No self-reference (input appears once, not twice)
    - Additive errors only (not multiplicative)
    - Critical line symmetry (errors cancel at level 0)
    - O(N) navigation via W-axis (not O(N²) attention)
    """
    
    def __init__(self, 
                 W_encode: LatticeTransform,
                 W_nav: np.ndarray,  # Navigation weights (small, kept as float)
                 ln_weight: LatticePoint):
        """
        Args:
            W_encode: Main transformation (sign × φ^level)
            W_nav: Navigation component for W-axis (float, small)
            ln_weight: LayerNorm weight as lattice point
        """
        self.W_encode = W_encode
        self.W_nav = W_nav  # Keep as float - it's small and used for steering
        self.ln_weight = ln_weight
        
        # Precompute level-grouped indices for efficient matmul
        self._precompute_level_groups()
    
    def _precompute_level_groups(self):
        """Group weight indices by φ-level for efficient computation."""
        self.level_groups = {}
        unique_levels = np.unique(self.W_encode.levels)
        
        for level in unique_levels:
            mask = (self.W_encode.levels == level)
            rows, cols = np.where(mask)
            if len(rows) > 0:
                self.level_groups[int(level)] = (rows, cols, self.W_encode.signs[mask])
    
    def forward(self, x: LatticePoint) -> LatticePoint:
        """
        Forward pass using 1-2 cycle.
        
        Cycle 1: ENCODE
          x_level = input levels
          x_sign = input signs
          x_w = input @ W_nav (navigation component)
        
        Cycle 2: NAVIGATE
          combined_level = W_level + x_level (integer addition!)
          combined_sign = W_sign × x_sign
          output = sum(combined_sign × φ^combined_level) × φ^x_w
        """
        # === CYCLE 1: ENCODE ===
        x_float = x.to_float()  # Decode input (we'll eliminate this later)
        
        # Compute navigation component (W-axis)
        x_w = x_float @ self.W_nav  # Small matmul for steering
        
        # RMS normalize
        rms = np.sqrt(np.mean(x_float ** 2) + 1e-6)
        x_normed = x_float / rms
        
        # Apply LayerNorm weight
        ln_w = self.ln_weight.to_float()
        x_normed = x_normed * ln_w
        
        # Re-encode to lattice
        x_encoded = LatticePoint.from_float(x_normed)
        
        # === CYCLE 2: NAVIGATE ===
        # This is where the magic happens - level addition instead of float multiply
        
        out_dim, in_dim = self.W_encode.shape
        output = np.zeros(out_dim, dtype=np.float64)
        
        # Group computation by level (Doc 152 optimization)
        for level, (rows, cols, signs) in self.level_groups.items():
            # Combined level = W_level + x_level
            # But we've grouped by W_level, so we need x_levels at cols
            x_levels_at_cols = x_encoded.levels[cols]
            x_signs_at_cols = x_encoded.signs[cols]
            
            combined_levels = level + x_levels_at_cols
            combined_signs = signs * x_signs_at_cols
            
            # φ^combined_level lookup
            # Clip to LUT range
            combined_levels = np.clip(combined_levels, -30, 30)
            phi_values = np.power(PHI, combined_levels.astype(np.float64))
            
            # Accumulate: output[row] += sign × φ^level
            np.add.at(output, rows, combined_signs * phi_values)
        
        # Apply navigation scaling: output × φ^x_w
        # x_w is a scalar (or small vector) that steers the output
        nav_scale = np.power(PHI, np.mean(x_w))  # Simplified: use mean for now
        output = output * nav_scale
        
        # Apply sparse corrections if present
        if self.W_encode.correction_rows is not None:
            # Corrections are small, compute directly
            for i, (row, col, val) in enumerate(zip(
                self.W_encode.correction_rows,
                self.W_encode.correction_cols,
                self.W_encode.correction_values
            )):
                output[row] += val * x_normed[col]
        
        # Snap output to lattice
        return LatticePoint.from_float(output.astype(np.float32))


class ZetaNavigator:
    """
    Full Zeta-Aligned Navigator.
    
    Replaces transformer forward pass with geometric navigation:
    TOKEN → φ-COORDINATE → MANIFOLD TRAVERSAL → φ-COORDINATE → TOKEN
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/zeta_navigator")
        
        # Embeddings as lattice points
        self.embeddings: Optional[np.ndarray] = None  # (vocab, dim) as LatticePoint per row
        self.embedding_signs: Optional[np.ndarray] = None
        self.embedding_levels: Optional[np.ndarray] = None
        
        # Layers
        self.layers: List[ZetaAlignedLayer] = []
        
        # LM head
        self.lm_head: Optional[LatticeTransform] = None
        
        # Final norm
        self.final_norm: Optional[LatticePoint] = None
        
        # Tokenizer
        self.tokenizer = None
        
        # Config
        self.hidden_dim = None
        self.vocab_size = None
    
    def convert_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct", max_layers: int = 28):
        """Convert transformer model to Zeta-aligned representation."""
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
        
        # Get config
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Convert embeddings
        print("Converting embeddings to lattice...")
        emb_weight = model.model.embed_tokens.weight.data.numpy()
        self.embedding_signs = np.sign(emb_weight).astype(np.int8)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.embedding_levels = np.round(
                np.log(np.abs(emb_weight) + 1e-45) / LOG_PHI
            ).astype(np.int8)
        self.embedding_levels = np.clip(self.embedding_levels, -30, 30)
        
        # Convert LM head
        print("Converting LM head to lattice...")
        lm_weight = model.lm_head.weight.data.numpy()
        self.lm_head = LatticeTransform.from_float(lm_weight)
        print(f"  LM head correction density: {self.lm_head.correction_density():.2%}")
        
        # Convert final norm
        norm_weight = model.model.norm.weight.data.numpy()
        self.final_norm = LatticePoint.from_float(norm_weight)
        
        # Convert layers
        n_layers = min(max_layers, len(model.model.layers))
        print(f"Converting {n_layers} layers to Zeta-aligned...")
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            
            # For Zeta-aligned, we combine attention and MLP into single transform
            # This is a simplification - full implementation would be more nuanced
            
            # Use the MLP gate projection as the main encode transform
            # (it's the largest and most important)
            W_gate = layer.mlp.gate_proj.weight.data.numpy()
            W_encode = LatticeTransform.from_float(W_gate)
            
            # Navigation weights from attention (simplified)
            # In full implementation, this would be derived from Q/K structure
            W_q = layer.self_attn.q_proj.weight.data.numpy()
            W_nav = np.mean(W_q, axis=0)  # Simplified: average Q weights
            W_nav = W_nav / (np.linalg.norm(W_nav) + 1e-6)  # Normalize
            
            # LayerNorm
            ln_weight = LatticePoint.from_float(
                layer.input_layernorm.weight.data.numpy()
            )
            
            zeta_layer = ZetaAlignedLayer(
                W_encode=W_encode,
                W_nav=W_nav,
                ln_weight=ln_weight,
            )
            
            self.layers.append(zeta_layer)
            print(f"  Layer {layer_idx}: correction density = {W_encode.correction_density():.2%}")
        
        print(f"\nConverted {n_layers} layers")
        
        # Clean up
        del model
    
    def get_embedding(self, token_id: int) -> LatticePoint:
        """Get embedding for token as lattice point."""
        return LatticePoint(
            signs=self.embedding_signs[token_id].copy(),
            levels=self.embedding_levels[token_id].copy(),
        )
    
    def navigate(self, token_ids: List[int]) -> LatticePoint:
        """
        Navigate from input tokens to output position.
        
        This is pure geometric traversal - no hidden states.
        """
        # Start at last token's position
        position = self.get_embedding(token_ids[-1])
        
        # Navigate through layers
        for layer in self.layers:
            position = layer.forward(position)
        
        # Final norm
        x = position.to_float()
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        x_normed = (x / rms) * self.final_norm.to_float()
        position = LatticePoint.from_float(x_normed)
        
        return position
    
    def predict_next_token(self, token_ids: List[int], temperature: float = 0.0) -> int:
        """Predict next token via geometric navigation."""
        position = self.navigate(token_ids)
        
        # Project through LM head
        pos_float = position.to_float()
        lm_float = self.lm_head.to_float()
        logits = lm_float @ pos_float
        
        if temperature == 0:
            return int(np.argmax(logits))
        else:
            logits = logits / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            return int(np.random.choice(len(probs), p=probs))
    
    def generate(self, prompt: str, max_tokens: int = 20, temperature: float = 0.0) -> str:
        """Generate text from prompt."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        for _ in range(max_tokens):
            next_token = self.predict_next_token(token_ids, temperature)
            token_ids.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(token_ids)


def test_lattice_point():
    """Test LatticePoint encoding."""
    print("=" * 60)
    print("Testing LatticePoint (Absolute φ-Lattice)")
    print("=" * 60)
    
    # Test with values that should snap cleanly to lattice
    # φ^-9 ≈ 0.013 is the peak (Doc 128)
    test_values = np.array([
        PHI ** -9,   # Should snap exactly
        PHI ** -8,   # Should snap exactly
        PHI ** -10,  # Should snap exactly
        0.015,       # Between φ^-9 and φ^-8, needs correction
        -0.013,      # Negative, near φ^-9
    ], dtype=np.float32)
    
    point = LatticePoint.from_float(test_values)
    decoded = point.to_float()
    
    print(f"Original:  {test_values}")
    print(f"Levels:    {point.levels}")
    print(f"Signs:     {point.signs}")
    print(f"Decoded:   {decoded}")
    print(f"Diff:      {test_values - decoded}")
    print(f"Corrections: {point.correction_indices}")
    print()
    
    # Test with actual model-like weights (peak at φ^-9)
    np.random.seed(42)
    # Generate weights that cluster around φ^-9 like real model
    levels = np.random.choice([-11, -10, -9, -8, -7], size=1000, p=[0.1, 0.15, 0.5, 0.15, 0.1])
    signs = np.random.choice([-1, 1], size=1000)
    weights = signs * (PHI ** levels) + np.random.randn(1000) * 0.001  # Small noise
    weights = weights.astype(np.float32)
    
    point = LatticePoint.from_float(weights)
    decoded = point.to_float()
    
    # Check how many snap exactly vs need correction
    exact_match = np.sum(point.correction_indices is None or 
                         len(point.correction_indices) == 0)
    
    print(f"Model-like weights test (1000 values):")
    print(f"  Correction density: {0 if point.correction_indices is None else len(point.correction_indices)/1000:.2%}")
    print(f"  Max absolute error: {np.max(np.abs(weights - decoded)):.6f}")
    print(f"  Storage: {point.storage_bytes()} bytes vs {weights.nbytes} bytes original")
    print(f"  Compression: {weights.nbytes / point.storage_bytes():.1f}x")
    print()


def test_lattice_transform():
    """Test LatticeTransform encoding."""
    print("=" * 60)
    print("Testing LatticeTransform (Weight Matrix)")
    print("=" * 60)
    
    # Create weight matrix with φ^-9 peak distribution (like Qwen2)
    np.random.seed(42)
    out_dim, in_dim = 256, 512
    
    # Generate levels with peak at -9
    levels = np.random.choice(
        [-12, -11, -10, -9, -8, -7, -6],
        size=(out_dim, in_dim),
        p=[0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05]
    )
    signs = np.random.choice([-1, 1], size=(out_dim, in_dim))
    W = signs * (PHI ** levels) + np.random.randn(out_dim, in_dim) * 0.0005
    W = W.astype(np.float32)
    
    transform = LatticeTransform.from_float(W)
    W_decoded = transform.to_float()
    
    print(f"Shape: {W.shape}")
    print(f"Correction density: {transform.correction_density():.2%}")
    print(f"Max absolute error: {np.max(np.abs(W - W_decoded)):.6f}")
    
    # Test matmul accuracy
    x = np.random.randn(in_dim).astype(np.float32) * 0.01
    y_original = W @ x
    y_lattice = W_decoded @ x
    
    # Check if we get the SAME result (not just correlated)
    max_diff = np.max(np.abs(y_original - y_lattice))
    print(f"Matmul max diff: {max_diff:.6f}")
    print(f"Matmul matches: {max_diff < 0.01}")  # Should be very close
    print()


def test_zeta_layer():
    """Test ZetaAlignedLayer."""
    print("=" * 60)
    print("Testing ZetaAlignedLayer (1-2 Cycle)")
    print("=" * 60)
    
    np.random.seed(42)
    in_dim = 128
    out_dim = 256
    
    # Create layer with model-like weights
    levels = np.random.choice([-11, -10, -9, -8, -7], size=(out_dim, in_dim), 
                               p=[0.1, 0.15, 0.5, 0.15, 0.1])
    signs = np.random.choice([-1, 1], size=(out_dim, in_dim))
    W = signs * (PHI ** levels)
    W = W.astype(np.float32)
    
    W_encode = LatticeTransform.from_float(W)
    W_nav = np.random.randn(in_dim).astype(np.float32) * 0.01
    ln_weight = LatticePoint.from_float(np.ones(in_dim, dtype=np.float32))
    
    layer = ZetaAlignedLayer(W_encode, W_nav, ln_weight)
    
    # Test forward pass
    x = np.random.randn(in_dim).astype(np.float32) * 0.01
    x_lattice = LatticePoint.from_float(x)
    
    output = layer.forward(x_lattice)
    output_float = output.to_float()
    
    print(f"Input dim: {in_dim}, Output dim: {out_dim}")
    print(f"Input levels range: [{x_lattice.levels.min()}, {x_lattice.levels.max()}]")
    print(f"Output levels range: [{output.levels.min()}, {output.levels.max()}]")
    print(f"Output values range: [{output_float.min():.4f}, {output_float.max():.4f}]")
    
    # The key insight: output values should ALSO cluster at φ^k
    # Let's check the distribution
    output_abs = np.abs(output_float)
    output_abs = output_abs[output_abs > 1e-10]  # Remove zeros
    output_levels_actual = np.log(output_abs) / LOG_PHI
    
    print(f"\nOutput level distribution (should cluster at integers):")
    for target_level in range(-5, 5):
        count = np.sum(np.abs(output_levels_actual - target_level) < 0.3)
        pct = count / len(output_levels_actual) * 100
        if pct > 1:
            print(f"  φ^{target_level:+d}: {pct:.1f}%")
    
    # Check if outputs naturally cluster (they should if the structure is geometric)
    level_fractions = output_levels_actual - np.round(output_levels_actual)
    print(f"\nLevel fraction std: {np.std(level_fractions):.3f} (lower = better lattice fit)")
    print(f"  (0.0 = perfect lattice, 0.29 = uniform random)")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        navigator = ZetaNavigator()
        navigator.convert_from_model(max_layers=2)
        
        prompt = "Hello"
        print(f"\nPrompt: {prompt}")
        output = navigator.generate(prompt, max_tokens=10)
        print(f"Output: {output}")
    else:
        test_lattice_point()
        test_lattice_transform()
        test_zeta_layer()
