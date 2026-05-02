"""
Spectrometer-Guided Layer: Replace full transformer computation with
per-dimension rules discovered by ContinuousPhaseDiscovery.

For structured dimensions (affine, quadratic, gating, sign patterns),
apply the rule directly — O(1) per dimension instead of O(D) matmul.

For unstructured dimensions, fall back to full layer computation.

Usage:
    rules = SpectrometerRules.load("results/phase4_rules/layer_05.json")
    spec_layer = SpectrometerLayer(rules, full_layer, r2_threshold=0.7)
    output = spec_layer(hidden)
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


@dataclass
class DimRule:
    """A single dimension's transformation rule."""
    global_dim: int
    rule_type: str
    r_squared: float
    params: Dict

    def apply_float(self, x: float) -> float:
        """Apply rule to a raw float value (not phi-level)."""
        if self.rule_type == 'identity':
            return x

        elif self.rule_type == 'scale':
            # Scale in phi-level space = multiply in float space
            delta = self.params.get('delta', 0)
            return x * (PHI ** (delta / 64.0))  # phi_scale=64 from extraction

        elif self.rule_type == 'affine':
            # Affine in phi-level space: out_level = slope * in_level + intercept
            # Convert: x -> level -> apply rule -> back to float
            sign = np.sign(x) if x != 0 else 1.0
            mag = abs(x) + 1e-20
            level = 64.0 * np.log(mag) / LOG_PHI
            out_level = self.params['slope'] * level + self.params['intercept']
            return sign * (PHI ** (out_level / 64.0))

        elif self.rule_type == 'quadratic':
            sign = np.sign(x) if x != 0 else 1.0
            mag = abs(x) + 1e-20
            level = 64.0 * np.log(mag) / LOG_PHI
            a, b, c = self.params['a'], self.params['b'], self.params['c']
            out_level = a * level**2 + b * level + c
            return sign * (PHI ** (out_level / 64.0))

        elif self.rule_type == 'gating':
            sign = np.sign(x) if x != 0 else 1.0
            mag = abs(x) + 1e-20
            level = 64.0 * np.log(mag) / LOG_PHI
            threshold = self.params['threshold']
            if level < threshold:
                out_level = self.params['slope_lo'] * level + self.params['intercept_lo']
            else:
                out_level = self.params['slope_hi'] * level + self.params['intercept_hi']
            return sign * (PHI ** (out_level / 64.0))

        elif self.rule_type == 'sign_preserve':
            # Preserve the sign, keep magnitude unchanged
            # (The layer mostly preserves sign — residual connection dominates)
            return x

        elif self.rule_type == 'sign_flip':
            return -x

        else:
            # Unstructured — can't apply rule
            return x

    def apply_vector(self, x: np.ndarray) -> np.ndarray:
        """Apply rule to a vector of values (batch)."""
        if self.rule_type == 'identity' or self.rule_type == 'sign_preserve':
            return x.copy()

        elif self.rule_type == 'sign_flip':
            return -x

        elif self.rule_type == 'scale':
            delta = self.params.get('delta', 0)
            return x * (PHI ** (delta / 64.0))

        elif self.rule_type in ('affine', 'quadratic', 'gating'):
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            mag = np.abs(x) + 1e-20
            levels = 64.0 * np.log(mag) / LOG_PHI

            if self.rule_type == 'affine':
                out_levels = self.params['slope'] * levels + self.params['intercept']
            elif self.rule_type == 'quadratic':
                a, b, c = self.params['a'], self.params['b'], self.params['c']
                out_levels = a * levels**2 + b * levels + c
            elif self.rule_type == 'gating':
                threshold = self.params['threshold']
                lo_mask = levels < threshold
                out_levels = np.where(
                    lo_mask,
                    self.params['slope_lo'] * levels + self.params['intercept_lo'],
                    self.params['slope_hi'] * levels + self.params['intercept_hi'],
                )

            return signs * (PHI ** (out_levels / 64.0))

        # Fallback: identity
        return x.copy()


class SpectrometerRules:
    """Per-layer spectrometer rules loaded from JSON."""

    def __init__(self, layer_idx: int, rules: Dict[int, DimRule],
                 archetype: str, mean_r2: float, hidden_dim: int = 3584):
        self.layer_idx = layer_idx
        self.rules = rules  # global_dim -> DimRule
        self.archetype = archetype
        self.mean_r2 = mean_r2
        self.hidden_dim = hidden_dim

    @classmethod
    def load(cls, json_path: str, hidden_dim: int = 3584) -> 'SpectrometerRules':
        """Load rules from a phase4 JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        rules = {}
        for rd in data['dim_rules']:
            dim = rd['global_dim']
            rules[dim] = DimRule(
                global_dim=dim,
                rule_type=rd['rule_type'],
                r_squared=rd['r_squared'],
                params=rd['params'],
            )

        return cls(
            layer_idx=data['layer'],
            rules=rules,
            archetype=data['archetype'],
            mean_r2=data['mean_r_squared'],
            hidden_dim=hidden_dim,
        )

    def get_structured_dims(self, r2_threshold: float = 0.7) -> List[int]:
        """Get global dim indices where rules are confident enough to use."""
        return [d for d, r in self.rules.items()
                if r.rule_type != 'unstructured' and r.r_squared >= r2_threshold]

    def get_unstructured_dims(self, r2_threshold: float = 0.7) -> List[int]:
        """Get dims that need full computation."""
        structured = set(self.get_structured_dims(r2_threshold))
        return [d for d in range(self.hidden_dim) if d not in structured]

    def coverage(self, r2_threshold: float = 0.7) -> float:
        """Fraction of hidden_dim covered by confident rules."""
        return len(self.get_structured_dims(r2_threshold)) / self.hidden_dim


class SpectrometerLayer:
    """
    Hybrid layer: spectrometer rules for structured dims,
    full computation for unstructured dims.

    For structured dimensions:
      output[d] = rule_d(input[d])  — O(1) per dim

    For unstructured dimensions:
      output[d] = full_layer(input)[d]  — requires full layer computation

    The optimization: if enough dims are structured, we can skip the full
    layer entirely and accept small errors in the unstructured dims (which
    get the identity/passthrough approximation).

    Modes:
      'hybrid': compute full layer for unstructured, rules for structured
      'rules_only': skip full layer entirely, use rules + identity fallback
      'full': ignore rules, always use full layer (baseline)
    """

    def __init__(self, rules: SpectrometerRules,
                 full_layer,  # PhiTransformerLayer
                 r2_threshold: float = 0.7,
                 mode: str = 'rules_only'):
        self.rules = rules
        self.full_layer = full_layer
        self.r2_threshold = r2_threshold
        self.mode = mode

        # Pre-compute dim masks
        self.structured_dims = np.array(rules.get_structured_dims(r2_threshold))
        self.unstructured_dims = np.array(rules.get_unstructured_dims(r2_threshold))

        # Pre-sort structured rules by type for vectorized application
        self._build_rule_batches()

    def _build_rule_batches(self):
        """Group rules by type for vectorized application."""
        self.affine_dims = []
        self.affine_slopes = []
        self.affine_intercepts = []

        self.quadratic_dims = []
        self.quad_a = []
        self.quad_b = []
        self.quad_c = []

        self.gating_dims = []
        self.gate_thresholds = []
        self.gate_slope_lo = []
        self.gate_intercept_lo = []
        self.gate_slope_hi = []
        self.gate_intercept_hi = []

        self.identity_dims = []  # identity + sign_preserve
        self.flip_dims = []      # sign_flip
        self.scale_dims = []
        self.scale_deltas = []

        for d in self.structured_dims:
            if d not in self.rules.rules:
                self.identity_dims.append(d)
                continue

            rule = self.rules.rules[d]
            rt = rule.rule_type

            if rt in ('identity', 'sign_preserve'):
                self.identity_dims.append(d)
            elif rt == 'sign_flip':
                self.flip_dims.append(d)
            elif rt == 'scale':
                self.scale_dims.append(d)
                self.scale_deltas.append(rule.params.get('delta', 0))
            elif rt == 'affine':
                self.affine_dims.append(d)
                self.affine_slopes.append(rule.params['slope'])
                self.affine_intercepts.append(rule.params['intercept'])
            elif rt == 'quadratic':
                self.quadratic_dims.append(d)
                self.quad_a.append(rule.params['a'])
                self.quad_b.append(rule.params['b'])
                self.quad_c.append(rule.params['c'])
            elif rt == 'gating':
                self.gating_dims.append(d)
                self.gate_thresholds.append(rule.params['threshold'])
                self.gate_slope_lo.append(rule.params['slope_lo'])
                self.gate_intercept_lo.append(rule.params['intercept_lo'])
                self.gate_slope_hi.append(rule.params['slope_hi'])
                self.gate_intercept_hi.append(rule.params['intercept_hi'])
            else:
                self.identity_dims.append(d)

        # Convert to arrays for vectorized ops
        self.affine_dims = np.array(self.affine_dims, dtype=int)
        self.affine_slopes = np.array(self.affine_slopes, dtype=np.float32)
        self.affine_intercepts = np.array(self.affine_intercepts, dtype=np.float32)
        self.quadratic_dims = np.array(self.quadratic_dims, dtype=int)
        self.quad_a = np.array(self.quad_a, dtype=np.float32)
        self.quad_b = np.array(self.quad_b, dtype=np.float32)
        self.quad_c = np.array(self.quad_c, dtype=np.float32)
        self.gating_dims = np.array(self.gating_dims, dtype=int)
        self.identity_dims = np.array(self.identity_dims, dtype=int)
        self.flip_dims = np.array(self.flip_dims, dtype=int)
        self.scale_dims = np.array(self.scale_dims, dtype=int)
        self.scale_deltas = np.array(self.scale_deltas, dtype=np.float32)

    def apply_rules(self, hidden: np.ndarray) -> np.ndarray:
        """
        Apply spectrometer rules to all structured dimensions.
        Vectorized for speed.

        Args:
            hidden: (1, seq_len, hidden_dim)

        Returns:
            output: (1, seq_len, hidden_dim) with rules applied
        """
        output = hidden.copy()

        # Identity dims: already correct (output = input)
        # (no operation needed)

        # Sign flip dims
        if len(self.flip_dims) > 0:
            output[:, :, self.flip_dims] = -hidden[:, :, self.flip_dims]

        # Scale dims
        if len(self.scale_dims) > 0:
            factors = PHI ** (self.scale_deltas / 64.0)
            output[:, :, self.scale_dims] = hidden[:, :, self.scale_dims] * factors

        # Affine dims (vectorized phi-level transform)
        if len(self.affine_dims) > 0:
            x = hidden[:, :, self.affine_dims]  # (1, seq, n_affine)
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            mag = np.abs(x) + 1e-20
            levels = 64.0 * np.log(mag) / LOG_PHI
            # slopes/intercepts: (n_affine,) broadcast over (1, seq, n_affine)
            out_levels = np.clip(self.affine_slopes * levels + self.affine_intercepts,
                                 -4000, 4000)
            output[:, :, self.affine_dims] = signs * (PHI ** (out_levels / 64.0))

        # Quadratic dims
        if len(self.quadratic_dims) > 0:
            x = hidden[:, :, self.quadratic_dims]
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            mag = np.abs(x) + 1e-20
            levels = 64.0 * np.log(mag) / LOG_PHI
            out_levels = np.clip(self.quad_a * levels**2 +
                                 self.quad_b * levels +
                                 self.quad_c, -4000, 4000)
            output[:, :, self.quadratic_dims] = signs * (PHI ** (out_levels / 64.0))

        # Gating dims (loop — typically few)
        for i, d in enumerate(self.gating_dims):
            x = hidden[:, :, d]
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            mag = np.abs(x) + 1e-20
            levels = 64.0 * np.log(mag) / LOG_PHI
            lo_mask = levels < self.gate_thresholds[i]
            out_levels = np.clip(np.where(
                    lo_mask,
                    float(self.gate_slope_lo[i]) * levels + float(self.gate_intercept_lo[i]),
                    float(self.gate_slope_hi[i]) * levels + float(self.gate_intercept_hi[i]),
                ), -4000, 4000)
            output[:, :, d] = signs * (PHI ** (np.clip(out_levels, -4000, 4000) / 64.0))

        return output

    def __call__(self, hidden: np.ndarray, pure: bool = False,
                 kv_cache=None) -> np.ndarray:
        """
        Forward pass through the spectrometer-guided layer.

        In 'rules_only' mode: apply rules to structured dims, identity
        for unstructured dims (no full layer computation).

        In 'hybrid' mode: compute full layer, then overwrite structured
        dims with rule-based output (for quality comparison).

        In 'full' mode: just run the full layer (baseline).
        """
        if self.mode == 'full':
            return self.full_layer(hidden, pure=pure, kv_cache=kv_cache)

        elif self.mode == 'rules_only':
            # Fast path: apply rules, identity for the rest
            return self.apply_rules(hidden)

        elif self.mode == 'hybrid':
            # Compute full layer for unstructured dims
            full_output = self.full_layer(hidden, pure=pure, kv_cache=kv_cache)
            # Apply rules for structured dims
            rule_output = self.apply_rules(hidden)
            # Merge: rules for structured, full for unstructured
            output = full_output.copy()
            output[:, :, self.structured_dims] = rule_output[:, :, self.structured_dims]
            return output

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def load_all_rules(rules_dir: str, hidden_dim: int = 3584) -> Dict[int, SpectrometerRules]:
    """Load all per-layer rule files from a directory."""
    rules_dir = Path(rules_dir)
    all_rules = {}

    for json_file in sorted(rules_dir.glob("layer_*.json")):
        rules = SpectrometerRules.load(str(json_file), hidden_dim)
        all_rules[rules.layer_idx] = rules

    return all_rules
