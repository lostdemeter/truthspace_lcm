"""
Module 3: Decomposer — Geometric Spectrometer
===============================================

Spectral decomposition into independent channels. Each of the d
dimensions is processed by a simple 1-d rule (COMB, PRESERVE, or
FLIP), separating the broadband input into independent spectral
components.

Optical analog: Diffraction grating (spectral decomposition)
Characteristic dimensionality: 3584 (one rule per dimension)

Specification:
    Channels:          d independent spectral channels
    Rule per channel:  One of {COMB, PRESERVE, FLIP}
    Predictability:    > 95% of state from per-channel rule
    Layers active:     L0 through L_extraction
    Storage:           d sign values per layer (448 bytes per layer)
"""

import numpy as np


# Channel modes
COMB = 'COMB'           # Complement (flip sign each layer)
PRESERVE = 'PRESERVE'   # Maintain sign
FLIP = 'FLIP'           # One-time sign change


class Decomposer:
    """Spectral decomposition into independent channels."""

    def __init__(self, d_model, channel_rules=None):
        """Initialize the decomposer.
        
        Args:
            d_model: Number of spectral channels (e.g. 3584)
            channel_rules: dict mapping dim_index → rule string
                           ('COMB', 'PRESERVE', 'FLIP')
                           If None, all channels default to PRESERVE.
        """
        self.d_model = d_model
        self.channel_rules = {}
        if channel_rules is not None:
            self.channel_rules = dict(channel_rules)

    def predict_sign(self, dim, layer, initial_sign=1.0):
        """Predict the sign of dimension `dim` at layer `layer`.
        
        Args:
            dim: Dimension index
            layer: Layer index
            initial_sign: Sign at layer 0 (+1 or -1)
            
        Returns:
            Predicted sign (+1 or -1)
        """
        rule = self.channel_rules.get(dim, PRESERVE)
        
        if rule == COMB:
            # Alternates sign each layer
            return initial_sign * ((-1) ** layer)
        elif rule == PRESERVE:
            return initial_sign
        elif rule == FLIP:
            # Flips once (at some transition layer)
            return -initial_sign
        else:
            return initial_sign

    def predict_state(self, initial_signs, layer):
        """Predict the full sign state at a given layer.
        
        Args:
            initial_signs: [d_model] array of signs at layer 0
            layer: Layer index
            
        Returns:
            [d_model] array of predicted signs
        """
        predicted = initial_signs.copy()
        for dim in range(self.d_model):
            predicted[dim] = self.predict_sign(dim, layer, initial_signs[dim])
        return predicted

    def accuracy(self, actual_signs, predicted_signs):
        """Measure prediction accuracy.
        
        Args:
            actual_signs: [d_model] actual sign state
            predicted_signs: [d_model] predicted sign state
            
        Returns:
            Fraction of correct predictions.
        """
        return float(np.mean(np.sign(actual_signs) == np.sign(predicted_signs)))

    @classmethod
    def from_trajectory(cls, d_model, sign_trajectory):
        """Learn channel rules from an observed sign trajectory.
        
        Analyzes the sign pattern across layers for each dimension
        and classifies it as COMB, PRESERVE, or FLIP.
        
        Args:
            d_model: Number of dimensions
            sign_trajectory: [n_layers, d_model] array of sign values
                             (actual hidden state signs across layers)
            
        Returns:
            Decomposer with learned channel rules.
        """
        n_layers = sign_trajectory.shape[0]
        rules = {}
        
        for dim in range(d_model):
            signs = np.sign(sign_trajectory[:, dim])
            # Count sign flips
            flips = np.sum(signs[1:] != signs[:-1])
            flip_rate = flips / (n_layers - 1) if n_layers > 1 else 0
            
            if flip_rate > 0.8:
                rules[dim] = COMB
            elif flip_rate < 0.1:
                rules[dim] = PRESERVE
            else:
                rules[dim] = FLIP
        
        return cls(d_model, rules)

    def rule_distribution(self):
        """Count how many channels use each rule.
        
        Returns:
            dict with counts for COMB, PRESERVE, FLIP, and UNSET
        """
        counts = {COMB: 0, PRESERVE: 0, FLIP: 0, 'UNSET': 0}
        for dim in range(self.d_model):
            rule = self.channel_rules.get(dim, 'UNSET')
            counts[rule] = counts.get(rule, 0) + 1
        return counts

    def spec(self):
        """Return specification measurements."""
        dist = self.rule_distribution()
        return {
            'd_model': self.d_model,
            'n_rules_set': len(self.channel_rules),
            'distribution': dist,
        }
