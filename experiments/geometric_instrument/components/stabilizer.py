"""
Module 2: Stabilizer — Geometric Gyroscope
============================================

Self-correcting dynamics in the residual stream. Errors settle
into a stable displaced orbit rather than diverging. The stabilizer
is emergent from the residual + RMSNorm dynamics — we don't build
it, we verify it emerges.

Optical analog: Adaptive optics / vibration isolation
Characteristic dimensionality: 1 (a single angle)

Specification:
    Settling time:    N/2 layers (half the instrument depth)
    Steady-state:     arccos(1/φ²) ≈ 68.4°
    Drift ratio:      ||error|| / ||signal|| ≈ 1.30 (prompt-independent)
    Parameters:       0 (emergent from residual stream dynamics)
"""

import numpy as np


class Stabilizer:
    """Self-correcting dynamics. Errors → stable orbit."""

    def __init__(self, norm_weight):
        """Initialize with RMSNorm weights.
        
        The Stabilizer wraps the normalization that precedes each
        sub-layer. This normalization is the key mechanism that
        prevents error divergence.
        
        Args:
            norm_weight: RMSNorm weight vector [d_model]
        """
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32).copy()
        self.d_model = self.norm_weight.shape[0]

    def normalize(self, h):
        """Apply RMS normalization (the stabilizing operation).
        
        Args:
            h: Hidden state [d_model] or [batch, seq, d_model]
            
        Returns:
            Normalized state with same shape.
        """
        rms = np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
        return (h / rms) * self.norm_weight

    @staticmethod
    def measure_drift(h_true, h_approx):
        """Measure angular displacement between true and approximate trajectories.
        
        This is the core diagnostic: if the Gyroscope is working,
        this angle should settle to a constant (~68.4°) rather than
        growing without bound.
        
        Args:
            h_true: True hidden state [d_model]
            h_approx: Approximate hidden state [d_model]
            
        Returns:
            dict with angle_deg, drift_ratio, cos_similarity
        """
        n_true = np.linalg.norm(h_true)
        n_approx = np.linalg.norm(h_approx)
        
        if n_true < 1e-12 or n_approx < 1e-12:
            return {'angle_deg': 0.0, 'drift_ratio': 0.0, 'cos_similarity': 1.0}
        
        cos_sim = float(np.dot(h_true, h_approx) / (n_true * n_approx))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angle = float(np.arccos(cos_sim) * 180 / np.pi)
        
        error = h_approx - h_true
        drift_ratio = float(np.linalg.norm(error) / n_true)
        
        return {
            'angle_deg': angle,
            'drift_ratio': drift_ratio,
            'cos_similarity': cos_sim,
        }

    @staticmethod
    def measure_trajectory(true_states, approx_states):
        """Measure drift across a full trajectory (all layers).
        
        Args:
            true_states: list of [d_model] arrays, one per layer
            approx_states: list of [d_model] arrays, one per layer
            
        Returns:
            dict with per-layer measurements and settling analysis
        """
        assert len(true_states) == len(approx_states)
        n_layers = len(true_states)
        
        angles = []
        drift_ratios = []
        for i in range(n_layers):
            m = Stabilizer.measure_drift(true_states[i], approx_states[i])
            angles.append(m['angle_deg'])
            drift_ratios.append(m['drift_ratio'])
        
        # Settling analysis: find where drift ratio stabilizes
        # (changes by less than 5% between consecutive layers)
        settled_layer = n_layers
        for i in range(1, n_layers):
            if abs(drift_ratios[i] - drift_ratios[i-1]) < 0.05 * drift_ratios[i]:
                settled_layer = i
                break
        
        return {
            'angles': angles,
            'drift_ratios': drift_ratios,
            'settled_layer': settled_layer,
            'steady_state_angle': float(np.mean(angles[-5:])) if n_layers >= 5 else float(np.mean(angles)),
            'steady_state_drift': float(np.mean(drift_ratios[-5:])) if n_layers >= 5 else float(np.mean(drift_ratios)),
        }

    @classmethod
    def from_model(cls, engine, layer_idx, sublayer='attn'):
        """Extract Stabilizer norm weights from a real model.
        
        Args:
            engine: PhiQwen2Engine instance
            layer_idx: Layer index
            sublayer: 'attn' for pre-attention norm, 'mlp' for pre-MLP norm
            
        Returns:
            Stabilizer instance.
        """
        layer = engine.layers[layer_idx]
        if sublayer == 'attn':
            return cls(layer.attention.norm_weight)
        elif sublayer == 'mlp':
            return cls(layer.mlp.norm_weight)
        else:
            raise ValueError(f"Unknown sublayer: {sublayer}")
