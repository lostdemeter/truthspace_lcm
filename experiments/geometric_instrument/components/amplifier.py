"""
Module 7: Amplifier — Geometric Laser Gain Medium
===================================================

Coherent signal boosting via SiLU-gated MLP that operates
orthogonally to the attention output. Each amplification stage
approximately doubles the answer signal projection.

Optical analog: Laser gain medium (stimulated emission)
Characteristic dimensionality: 18944 intermediate → 3584 output

Specification:
    Architecture:      SiLU(x @ W_gate.T) ⊙ (x @ W_up.T) @ W_down.T
    Expansion:         d → ~5.3d intermediate → d
    Orthogonality:     cos(Δattn, Δmlp) ≈ 0
    Dominance:         ||Δmlp|| / ||Δattn|| > 2×
    Gain per stage:    ~2× signal projection increase
    Stages needed:     ~5 (L23–L27) to go from 13% to dominance
"""

import numpy as np


def _silu(x):
    """SiLU activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


class Amplifier:
    """Coherent signal boosting. Orthogonal to attention."""

    def __init__(self, W_gate, W_up, W_down, norm_weight):
        """Initialize with MLP weights and pre-MLP norm weights.
        
        Args:
            W_gate: Gate projection [intermediate, d_model]
            W_up:   Up projection [intermediate, d_model]
            W_down: Down projection [d_model, intermediate]
            norm_weight: RMSNorm weights for pre-MLP normalization [d_model]
        """
        self.W_gate = np.asarray(W_gate, dtype=np.float32)
        self.W_up = np.asarray(W_up, dtype=np.float32)
        self.W_down = np.asarray(W_down, dtype=np.float32)
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32)
        self.d_model = self.W_down.shape[0]
        self.intermediate_size = self.W_gate.shape[0]

    def _rms_norm(self, x):
        """RMS normalization."""
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        return (x / rms) * self.norm_weight

    def amplify(self, h):
        """Apply one stage of amplification.
        
        This is the full MLP operation:
            normed = rms_norm(h)
            gate = normed @ W_gate.T
            up = normed @ W_up.T
            hidden = SiLU(gate) * up
            output = hidden @ W_down.T
            return h + output  (residual connection)
        
        Args:
            h: Input hidden state [d_model]
            
        Returns:
            Amplified state [d_model] (h + MLP(norm(h)))
        """
        normed = self._rms_norm(h)
        gate = normed @ self.W_gate.T
        up = normed @ self.W_up.T
        hidden = _silu(gate) * up
        output = hidden @ self.W_down.T
        return h + output

    def delta(self, h):
        """Compute just the MLP's contribution (without residual).
        
        Args:
            h: Input hidden state [d_model]
            
        Returns:
            MLP output [d_model] (the Δmlp that gets added to h)
        """
        normed = self._rms_norm(h)
        gate = normed @ self.W_gate.T
        up = normed @ self.W_up.T
        hidden = _silu(gate) * up
        return hidden @ self.W_down.T

    def measure_gain(self, h_before, answer_dir):
        """Measure amplification gain along the answer direction.
        
        Args:
            h_before: State before amplification [d_model]
            answer_dir: Answer token direction (from LM head) [d_model]
            
        Returns:
            dict with projection_before, projection_after, gain,
            delta_norm, orthogonality info
        """
        ans_unit = answer_dir / np.linalg.norm(answer_dir)
        
        proj_before = float(np.dot(h_before, ans_unit))
        
        h_after = self.amplify(h_before)
        proj_after = float(np.dot(h_after, ans_unit))
        
        d_mlp = h_after - h_before
        d_mlp_norm = float(np.linalg.norm(d_mlp))
        
        return {
            'projection_before': proj_before,
            'projection_after': proj_after,
            'gain': proj_after / proj_before if abs(proj_before) > 1e-12 else float('inf'),
            'delta_norm': d_mlp_norm,
            'delta_cos_answer': float(np.dot(d_mlp, ans_unit) / d_mlp_norm) if d_mlp_norm > 1e-12 else 0.0,
        }

    @classmethod
    def from_model(cls, engine, layer_idx):
        """Extract Amplifier weights from a real model.
        
        Args:
            engine: PhiQwen2Engine instance
            layer_idx: Layer index (e.g. 23)
            
        Returns:
            Amplifier instance with MLP weights from the model.
        """
        from phi_geometric.inference.phi_integer import phi_to_float
        
        mlp = engine.layers[layer_idx].mlp
        
        W_gate = phi_to_float(mlp.W_gate.signs, mlp.W_gate.exponents)
        W_up = phi_to_float(mlp.W_up.signs, mlp.W_up.exponents)
        W_down = phi_to_float(mlp.W_down.signs, mlp.W_down.exponents)
        norm_weight = mlp.norm_weight.copy()
        
        return cls(W_gate, W_up, W_down, norm_weight)

    def spec(self):
        """Return specification measurements."""
        return {
            'd_model': self.d_model,
            'intermediate_size': self.intermediate_size,
            'expansion_ratio': self.intermediate_size / self.d_model,
        }
