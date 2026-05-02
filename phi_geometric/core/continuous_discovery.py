"""
Continuous Phase Discovery: Find geometric structure in continuous data.

Extension of PhaseDiscovery from discrete tokens to φ-lattice encoded
continuous values. Instead of discrete mapping rules, discovers:

    LINEAR:
    - identity:       Δlevel = 0 (pass-through)
    - scale:          Δlevel = constant (uniform shift in φ-space)
    - affine:         Δlevel = a × level_in + b (linear in log-space)

    NON-LINEAR:
    - quadratic:      out = a×in² + b×in + c (curvature from MLP)
    - gating:         piecewise linear with threshold (SiLU/GELU)
    - sigmoid:        saturating map (softmax-like compression)

    CROSS-DIMENSIONAL:
    - cross_dim:      out_d = a×in_d + b×in_{d'} + c (attention mixing)
    - context:        Δlevel = f(neighbor_levels) (neighbor-dependent)

    STRUCTURAL:
    - collapse:       Many input levels → fewer output levels

The φ-lattice provides the bridge between continuous values and
discrete structure: values quantize to integer φ-levels, and
transformations become level deltas (integer arithmetic).

Key Principles:
    - Same fail-fast philosophy as PhaseDiscovery
    - No fallbacks — if geometric rules don't explain the data, we see the error
    - The φ-level IS the natural discrete unit for continuous values
    - Transformation rules operate on levels, not raw floats

Author: TruthSpace LCM Project
Date: February 2026
"""

import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# ============================================================================
# φ-LEVEL ENCODING
# ============================================================================

def to_phi_levels(values: np.ndarray, scale: int = 128) -> np.ndarray:
    """
    Encode continuous values as integer φ-levels.
    
    value = sign × φ^(level/scale)
    level = round(scale × log_φ(|value|))
    
    Returns integer array of same shape.
    """
    signs = np.sign(values)
    signs[signs == 0] = 1
    
    magnitudes = np.abs(values).astype(np.float64) + 1e-20
    levels = np.round(scale * np.log(magnitudes) / LOG_PHI).astype(np.int32)
    
    # Encode sign: positive levels stay positive, negative values get offset
    # Actually, keep sign separate — return (signs, levels) tuple
    return signs.astype(np.int8), levels


def from_phi_levels(signs: np.ndarray, levels: np.ndarray, scale: int = 128) -> np.ndarray:
    """Decode φ-levels back to continuous values."""
    return signs * (PHI ** (levels / scale))


# ============================================================================
# CONTINUOUS TRANSFORM RULES
# ============================================================================

@dataclass
class ContinuousRule:
    """A transformation rule discovered from continuous data.
    
    Operates on φ-levels: output_level = f(input_level, context_levels)
    """
    dim_idx: int          # Which dimension this rule applies to
    rule_type: str        # identity, scale, affine, context, collapse
    params: Dict          # Rule-specific parameters
    r_squared: float      # How well this rule fits (0-1)
    
    def apply(self, input_level: int, context: Optional[Dict[int, int]] = None) -> int:
        """Apply this rule to get predicted output level."""
        if self.rule_type == 'identity':
            return input_level
        
        elif self.rule_type == 'scale':
            return input_level + self.params['delta']
        
        elif self.rule_type == 'affine':
            a = self.params['slope']
            b = self.params['intercept']
            return round(a * input_level + b)
        
        elif self.rule_type == 'quadratic':
            a = self.params['a']
            b = self.params['b']
            c = self.params['c']
            return round(a * input_level**2 + b * input_level + c)
        
        elif self.rule_type == 'gating':
            threshold = self.params['threshold']
            if input_level < threshold:
                return round(self.params['slope_lo'] * input_level + self.params['intercept_lo'])
            else:
                return round(self.params['slope_hi'] * input_level + self.params['intercept_hi'])
        
        elif self.rule_type == 'sigmoid':
            L = self.params['L']
            k = self.params['k']
            x0 = self.params['x0']
            base = self.params['base']
            return round(base + L / (1 + np.exp(-k * (input_level - x0))))
        
        elif self.rule_type == 'cross_dim':
            a = self.params['coeff_self']
            b = self.params['coeff_other']
            c = self.params['intercept']
            ctx_dim = self.params['other_dim']
            other_val = context.get(ctx_dim, 0) if context else 0
            return round(a * input_level + b * other_val + c)
        
        elif self.rule_type == 'context':
            ctx_dim = self.params['context_dim']
            if context and ctx_dim in context:
                ctx_val = context[ctx_dim]
                slope = self.params['ctx_slope']
                intercept = self.params['ctx_intercept']
                return round(slope * ctx_val + intercept)
            return input_level + self.params.get('mean_delta', 0)
        
        elif self.rule_type == 'collapse':
            boundaries = self.params['boundaries']
            outputs = self.params['outputs']
            for i, b in enumerate(boundaries):
                if input_level < b:
                    return outputs[i]
            return outputs[-1]
        
        return input_level


@dataclass
class ContinuousPhase:
    """A phase of continuous transformation rules."""
    rules: List[ContinuousRule]
    name: str = ""
    
    def get_rule(self, dim_idx: int) -> Optional[ContinuousRule]:
        """Get rule for a specific dimension."""
        for rule in self.rules:
            if rule.dim_idx == dim_idx:
                return rule
        return None


# ============================================================================
# DISCOVERY RESULT
# ============================================================================

@dataclass
class ContinuousDiscoveryResult:
    """Result of continuous phase discovery."""
    phases: List[ContinuousPhase]
    archetype: str                    # Overall transformation archetype
    rule_distribution: Dict[str, int] # Count of each rule type
    mean_r_squared: float             # Average fit quality
    dim_results: Dict[int, ContinuousRule]  # Per-dimension rules
    
    def summary(self) -> str:
        lines = [
            f"Archetype: {self.archetype}",
            f"Phases: {len(self.phases)}",
            f"R²: {self.mean_r_squared:.4f}",
            f"Rules: {self.rule_distribution}",
        ]
        return "\n".join(lines)


# ============================================================================
# CONTINUOUS PHASE DISCOVERY ENGINE
# ============================================================================

class ContinuousPhaseDiscovery:
    """
    Discover transformation rules from continuous (input, output) pairs.
    
    Like PhaseDiscovery but for φ-lattice encoded continuous values.
    Each dimension is analyzed independently to find its transformation rule.
    Context-dependent rules look at neighboring dimensions.
    
    Usage:
        cpd = ContinuousPhaseDiscovery(phi_scale=128)
        cpd.add_pair(input_vector, output_vector)
        # ... add more pairs ...
        result = cpd.discover()
    """
    
    def __init__(self, phi_scale: int = 128, context_radius: int = 0,
                 identity_threshold: float = 0.5,
                 affine_threshold: float = 0.8):
        """
        Args:
            phi_scale: Resolution of φ-level encoding (higher = finer)
            context_radius: How many neighbor dims to consider for context rules
            identity_threshold: Max |mean_delta| in φ-levels to classify as identity
            affine_threshold: Min R² to accept an affine fit
        """
        self.phi_scale = phi_scale
        self.context_radius = context_radius
        self.identity_threshold = identity_threshold
        self.affine_threshold = affine_threshold
        
        self.input_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def add_pair(self, input_vec: np.ndarray, output_vec: np.ndarray):
        """Add an (input, output) observation pair."""
        self.input_pairs.append((
            np.asarray(input_vec, dtype=np.float64),
            np.asarray(output_vec, dtype=np.float64),
        ))
    
    def discover(self) -> ContinuousDiscoveryResult:
        """
        Analyze all pairs and discover per-dimension transformation rules.
        
        For each dimension d:
        1. Collect all (input[d], output[d]) values
        2. Convert to φ-levels
        3. Compute level deltas
        4. Classify: identity, scale, affine, context, or collapse
        """
        if not self.input_pairs:
            raise ValueError("No pairs added")
        
        # Stack into matrices
        inputs = np.array([p[0] for p in self.input_pairs])   # (N, D)
        outputs = np.array([p[1] for p in self.input_pairs])  # (N, D)
        
        N, D = inputs.shape
        
        # Convert to φ-levels
        in_signs, in_levels = to_phi_levels(inputs, self.phi_scale)
        out_signs, out_levels = to_phi_levels(outputs, self.phi_scale)
        
        # Also compute on raw deltas for residual analysis
        deltas = outputs - inputs
        delta_signs, delta_levels = to_phi_levels(deltas, self.phi_scale)
        
        # Analyze each dimension
        dim_rules = {}
        for d in range(D):
            rule = self._analyze_dimension(
                d, N, D,
                in_levels[:, d], out_levels[:, d],
                in_signs[:, d], out_signs[:, d],
                inputs[:, d], outputs[:, d],
                in_levels, out_levels,
                in_signs, out_signs,
            )
            dim_rules[d] = rule
        
        # Classify overall archetype
        rule_dist = Counter(r.rule_type for r in dim_rules.values())
        mean_r2 = np.mean([r.r_squared for r in dim_rules.values()])
        archetype = self._classify_archetype(rule_dist, D)
        
        # Group into phases based on rule types
        phases = self._build_phases(dim_rules)
        
        return ContinuousDiscoveryResult(
            phases=phases,
            archetype=archetype,
            rule_distribution=dict(rule_dist),
            mean_r_squared=float(mean_r2),
            dim_results=dim_rules,
        )
    
    # ------------------------------------------------------------------
    # Helper: compute R² from predicted vs actual
    # ------------------------------------------------------------------
    @staticmethod
    def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2) + 1e-20
        return float(max(0, 1 - ss_res / ss_tot))
    
    # ------------------------------------------------------------------
    # Individual rule fitters (each returns (r2, params) or None)
    # ------------------------------------------------------------------
    
    def _fit_identity(self, in_lvl, out_lvl, level_deltas, mean_delta, std_delta):
        if abs(mean_delta) < self.identity_threshold and std_delta < self.identity_threshold * 2:
            r2 = self._r_squared(out_lvl, in_lvl)
            return r2, {}
        return None
    
    def _fit_scale(self, in_lvl, out_lvl, level_deltas, mean_delta, std_delta):
        if std_delta < abs(mean_delta) * 0.3 + self.identity_threshold:
            delta_int = round(mean_delta)
            predicted = in_lvl + delta_int
            r2 = self._r_squared(out_lvl, predicted)
            if r2 > self.affine_threshold:
                return r2, {'delta': delta_int}
        return None
    
    def _fit_affine(self, in_lvl, out_lvl, N):
        if N < 3:
            return None
        try:
            coeffs = np.polyfit(in_lvl.astype(float), out_lvl.astype(float), 1)
            slope, intercept = coeffs
            predicted = slope * in_lvl + intercept
            r2 = self._r_squared(out_lvl, predicted)
            if r2 > self.affine_threshold:
                return r2, {'slope': float(slope), 'intercept': float(intercept)}
        except (np.linalg.LinAlgError, ValueError):
            pass
        return None
    
    def _fit_quadratic(self, in_lvl, out_lvl, N):
        """Fit out = a×in² + b×in + c. Captures MLP curvature."""
        if N < 5:
            return None
        try:
            x = in_lvl.astype(float)
            y = out_lvl.astype(float)
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs
            predicted = a * x**2 + b * x + c
            r2 = self._r_squared(out_lvl, predicted)
            # Only accept if meaningfully better than affine
            affine_coeffs = np.polyfit(x, y, 1)
            affine_pred = affine_coeffs[0] * x + affine_coeffs[1]
            affine_r2 = self._r_squared(out_lvl, affine_pred)
            if r2 > self.affine_threshold and r2 > affine_r2 + 0.05:
                return r2, {'a': float(a), 'b': float(b), 'c': float(c)}
        except (np.linalg.LinAlgError, ValueError):
            pass
        return None
    
    def _fit_gating(self, in_lvl, out_lvl, N):
        """
        Fit piecewise linear: two slopes with a threshold.
        Captures SiLU/GELU gating behavior.
        """
        if N < 8:
            return None
        
        x = in_lvl.astype(float)
        y = out_lvl.astype(float)
        
        # Try breakpoints at 20th, 35th, 50th, 65th, 80th percentiles
        best_r2 = 0
        best_params = None
        
        for pct in [20, 35, 50, 65, 80]:
            threshold = float(np.percentile(x, pct))
            lo_mask = x < threshold
            hi_mask = x >= threshold
            
            if np.sum(lo_mask) < 3 or np.sum(hi_mask) < 3:
                continue
            
            try:
                lo_coeffs = np.polyfit(x[lo_mask], y[lo_mask], 1)
                hi_coeffs = np.polyfit(x[hi_mask], y[hi_mask], 1)
                
                predicted = np.where(
                    x < threshold,
                    lo_coeffs[0] * x + lo_coeffs[1],
                    hi_coeffs[0] * x + hi_coeffs[1],
                )
                r2 = self._r_squared(out_lvl, predicted)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        'threshold': float(threshold),
                        'slope_lo': float(lo_coeffs[0]),
                        'intercept_lo': float(lo_coeffs[1]),
                        'slope_hi': float(hi_coeffs[0]),
                        'intercept_hi': float(hi_coeffs[1]),
                    }
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if best_params is not None and best_r2 > self.affine_threshold:
            # Only accept if meaningfully better than simple affine
            try:
                affine_coeffs = np.polyfit(x, y, 1)
                affine_r2 = self._r_squared(out_lvl, affine_coeffs[0]*x + affine_coeffs[1])
            except:
                affine_r2 = 0
            if best_r2 > affine_r2 + 0.05:
                return best_r2, best_params
        return None
    
    def _fit_sigmoid(self, in_lvl, out_lvl, N):
        """
        Fit saturating sigmoid: out = base + L / (1 + exp(-k*(in - x0)))
        Captures softmax-like compression / saturation.
        """
        if N < 8:
            return None
        
        x = in_lvl.astype(float)
        y = out_lvl.astype(float)
        
        # Quick check: does the output range suggest saturation?
        y_range = np.max(y) - np.min(y)
        x_range = np.max(x) - np.min(x)
        if y_range < 1 or x_range < 1:
            return None
        
        # Approximate sigmoid params from data
        y_min, y_max = np.min(y), np.max(y)
        L = float(y_max - y_min)
        base = float(y_min)
        x0 = float(np.median(x))
        
        # Estimate k from the slope at the midpoint
        mid_mask = (x > np.percentile(x, 35)) & (x < np.percentile(x, 65))
        if np.sum(mid_mask) < 3:
            return None
        
        try:
            mid_slope = np.polyfit(x[mid_mask], y[mid_mask], 1)[0]
            # For sigmoid: max slope = L*k/4 at x=x0
            k = float(4 * mid_slope / (L + 1e-10))
        except:
            return None
        
        if abs(k) < 1e-10:
            return None
        
        # Compute predictions
        z = -k * (x - x0)
        z = np.clip(z, -50, 50)  # Prevent overflow
        predicted = base + L / (1 + np.exp(z))
        r2 = self._r_squared(out_lvl, predicted)
        
        if r2 > self.affine_threshold:
            # Only accept if meaningfully better than affine
            try:
                affine_coeffs = np.polyfit(x, y, 1)
                affine_r2 = self._r_squared(out_lvl, affine_coeffs[0]*x + affine_coeffs[1])
            except:
                affine_r2 = 0
            if r2 > affine_r2 + 0.05:
                return r2, {'L': L, 'k': k, 'x0': x0, 'base': base}
        return None
    
    def _fit_cross_dim(self, d, in_lvl, out_lvl, all_in_lvl, N, D):
        """
        Fit out_d = a×in_d + b×in_{d'} + c for the best other dimension d'.
        Captures the simplest form of attention-like cross-dimensional mixing.
        """
        if N < 5:
            return None
        
        x_self = in_lvl.astype(float)
        y = out_lvl.astype(float)
        
        # Compute affine R² as baseline
        try:
            affine_coeffs = np.polyfit(x_self, y, 1)
            affine_r2 = self._r_squared(out_lvl, affine_coeffs[0]*x_self + affine_coeffs[1])
        except:
            affine_r2 = 0
        
        best_r2 = 0
        best_params = None
        
        # Search for best partner dimension
        # For efficiency, sample candidate dimensions
        search_radius = self.context_radius if self.context_radius > 0 else 5
        candidates = []
        for offset in range(-search_radius, search_radius + 1):
            if offset == 0:
                continue
            cd = d + offset
            if 0 <= cd < D:
                candidates.append(cd)
        # Also try some random dimensions for broader coverage
        if D > 2 * search_radius + 1:
            n_random = min(10, D - len(candidates))
            random_dims = np.random.choice(D, n_random, replace=False)
            candidates.extend([int(rd) for rd in random_dims if rd != d])
        
        for cd in candidates:
            x_other = all_in_lvl[:, cd].astype(float)
            
            try:
                # Least squares: y = a*x_self + b*x_other + c
                A = np.column_stack([x_self, x_other, np.ones(N)])
                result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = result
                predicted = a * x_self + b * x_other + c
                r2 = self._r_squared(out_lvl, predicted)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        'coeff_self': float(a),
                        'coeff_other': float(b),
                        'intercept': float(c),
                        'other_dim': cd,
                    }
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if best_params and best_r2 > self.affine_threshold and best_r2 > affine_r2 + 0.05:
            return best_r2, best_params
        return None
    
    def _fit_context(self, d, in_lvl, out_lvl, all_in_lvl, D):
        """Fit output as function of a single neighbor dimension (legacy)."""
        if self.context_radius <= 0:
            return None
        
        best_r2 = 0
        best_params = None
        
        for offset in range(-self.context_radius, self.context_radius + 1):
            if offset == 0:
                continue
            ctx_d = d + offset
            if ctx_d < 0 or ctx_d >= D:
                continue
            
            ctx_levels = all_in_lvl[:, ctx_d].astype(float)
            try:
                coeffs = np.polyfit(ctx_levels, out_lvl.astype(float), 1)
                predicted = coeffs[0] * ctx_levels + coeffs[1]
                r2 = self._r_squared(out_lvl, predicted)
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        'context_dim': ctx_d,
                        'ctx_slope': float(coeffs[0]),
                        'ctx_intercept': float(coeffs[1]),
                    }
            except (np.linalg.LinAlgError, ValueError):
                pass
        
        if best_params and best_r2 > self.affine_threshold:
            return best_r2, best_params
        return None
    
    # ------------------------------------------------------------------
    # SIGN-PATTERN RULES
    # Signs encode which side of hyperplane boundaries — the irreducible
    # 1-bit decisions. XOR of signs = boundary crossing computation.
    # ------------------------------------------------------------------
    
    def _fit_sign_preserve(self, in_sgn, out_sgn, in_raw, out_raw):
        """Check if output sign always matches input sign."""
        match = (in_sgn == out_sgn)
        accuracy = float(np.mean(match))
        if accuracy > 0.85:
            # Compute R² on full signed values using sign prediction
            r2 = self._r_squared(out_raw, np.abs(out_raw) * in_sgn)
            return max(r2, accuracy), {'sign_accuracy': accuracy}
        return None
    
    def _fit_sign_flip(self, in_sgn, out_sgn, in_raw, out_raw):
        """Check if output sign always opposite to input sign."""
        match = (in_sgn == -out_sgn)
        accuracy = float(np.mean(match))
        if accuracy > 0.85:
            r2 = self._r_squared(out_raw, np.abs(out_raw) * (-in_sgn))
            return max(r2, accuracy), {'sign_accuracy': accuracy}
        return None
    
    def _fit_sign_xor(self, d, in_sgn, out_sgn, all_in_sgn, in_raw, out_raw, N, D):
        """
        Check if output sign = XOR of input signs across dimensions.
        
        XOR in sign space: sign_out = sign_in[d] * sign_in[d']
        This captures boundary-crossing decisions that depend on
        relationships between dimensions — the core of attention routing.
        """
        if N < 5:
            return None
        
        best_accuracy = 0
        best_params = None
        
        # Search neighbor dimensions + random sample
        search_radius = self.context_radius if self.context_radius > 0 else 5
        candidates = []
        for offset in range(-search_radius, search_radius + 1):
            if offset == 0:
                continue
            cd = d + offset
            if 0 <= cd < D:
                candidates.append(cd)
        if D > 2 * search_radius + 1:
            n_random = min(10, D - len(candidates))
            random_dims = np.random.choice(D, n_random, replace=False)
            candidates.extend([int(rd) for rd in random_dims if rd != d])
        
        for cd in candidates:
            other_sgn = all_in_sgn[:, cd]
            # XOR in sign space: (+1)(+1)=+1, (+1)(-1)=-1, etc.
            xor_sign = in_sgn * other_sgn
            match = (xor_sign == out_sgn)
            accuracy = float(np.mean(match))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'xor_dim': cd,
                    'sign_accuracy': accuracy,
                }
            
            # Also try inverted XOR
            match_inv = (-xor_sign == out_sgn)
            accuracy_inv = float(np.mean(match_inv))
            if accuracy_inv > best_accuracy:
                best_accuracy = accuracy_inv
                best_params = {
                    'xor_dim': cd,
                    'inverted': True,
                    'sign_accuracy': accuracy_inv,
                }
        
        if best_params and best_accuracy > 0.85:
            return best_accuracy, best_params
        return None
    
    def _fit_sign_gate(self, in_sgn, out_sgn, in_lvl, in_raw, out_raw, N):
        """
        Check if sign behavior depends on input magnitude.
        Below threshold: preserve sign. Above threshold: flip sign.
        (Or vice versa.) Captures gating decisions.
        """
        if N < 8:
            return None
        
        x = in_lvl.astype(float)
        best_accuracy = 0
        best_params = None
        
        for pct in [20, 35, 50, 65, 80]:
            threshold = float(np.percentile(x, pct))
            lo_mask = x < threshold
            hi_mask = x >= threshold
            
            if np.sum(lo_mask) < 3 or np.sum(hi_mask) < 3:
                continue
            
            # Try: preserve below, flip above
            predicted_sgn = np.where(lo_mask, in_sgn, -in_sgn)
            acc1 = float(np.mean(predicted_sgn == out_sgn))
            
            # Try: flip below, preserve above
            predicted_sgn2 = np.where(lo_mask, -in_sgn, in_sgn)
            acc2 = float(np.mean(predicted_sgn2 == out_sgn))
            
            if acc1 > best_accuracy:
                best_accuracy = acc1
                best_params = {
                    'threshold': threshold,
                    'lo_action': 'preserve',
                    'hi_action': 'flip',
                    'sign_accuracy': acc1,
                }
            if acc2 > best_accuracy:
                best_accuracy = acc2
                best_params = {
                    'threshold': threshold,
                    'lo_action': 'flip',
                    'hi_action': 'preserve',
                    'sign_accuracy': acc2,
                }
        
        if best_params and best_accuracy > 0.85:
            return best_accuracy, best_params
        return None
    
    def _fit_collapse(self, in_lvl, out_lvl):
        """Check for many-to-few level mapping."""
        unique_in = len(np.unique(in_lvl))
        unique_out = len(np.unique(out_lvl))
        if not (unique_out < unique_in * 0.5 and unique_out <= 10):
            return None
        
        boundaries = []
        outputs_map = []
        for out_val in np.unique(out_lvl):
            mask = out_lvl == out_val
            in_vals = in_lvl[mask]
            boundaries.append(int(np.max(in_vals)) + 1)
            outputs_map.append(int(out_val))
        
        pairs = sorted(zip(boundaries, outputs_map))
        boundaries = [p[0] for p in pairs]
        outputs_map = [p[1] for p in pairs]
        
        predicted = np.zeros_like(out_lvl)
        for i, val in enumerate(in_lvl):
            for j, b in enumerate(boundaries):
                if val < b:
                    predicted[i] = outputs_map[j]
                    break
            else:
                predicted[i] = outputs_map[-1]
        
        r2 = self._r_squared(out_lvl, predicted)
        return r2, {
            'boundaries': boundaries,
            'outputs': outputs_map,
            'unique_in': unique_in,
            'unique_out': unique_out,
        }
    
    # ------------------------------------------------------------------
    # Main dimension analyzer — tries all rules in complexity order
    # ------------------------------------------------------------------
    
    def _analyze_dimension(
        self, d: int, N: int, D: int,
        in_lvl: np.ndarray, out_lvl: np.ndarray,
        in_sgn: np.ndarray, out_sgn: np.ndarray,
        in_raw: np.ndarray, out_raw: np.ndarray,
        all_in_lvl: np.ndarray, all_out_lvl: np.ndarray,
        all_in_sgn: np.ndarray = None, all_out_sgn: np.ndarray = None,
    ) -> ContinuousRule:
        """
        Analyze a single dimension's transformation.
        
        Tries rules in order of complexity, accepting the simplest
        that exceeds the threshold. This is the spectrometer's core
        classification logic.
        
        Rule hierarchy:
          LINEAR:      identity → scale → affine
          NON-LINEAR:  quadratic → gating → sigmoid
          CROSS-DIM:   cross_dim → context
          STRUCTURAL:  collapse
          SIGN:        sign_preserve → sign_flip → sign_xor → sign_gate
          FALLBACK:    unstructured
        """
        level_deltas = out_lvl - in_lvl
        mean_delta = float(np.mean(level_deltas))
        std_delta = float(np.std(level_deltas))
        
        # --- LINEAR RULES (simplest first) ---
        
        # 1. Identity
        result = self._fit_identity(in_lvl, out_lvl, level_deltas, mean_delta, std_delta)
        if result:
            return ContinuousRule(d, 'identity', result[1], result[0])
        
        # 2. Scale
        result = self._fit_scale(in_lvl, out_lvl, level_deltas, mean_delta, std_delta)
        if result:
            return ContinuousRule(d, 'scale', result[1], result[0])
        
        # 3. Affine
        result = self._fit_affine(in_lvl, out_lvl, N)
        if result:
            return ContinuousRule(d, 'affine', result[1], result[0])
        
        # --- NON-LINEAR RULES ---
        
        # 4. Quadratic (captures MLP curvature)
        result = self._fit_quadratic(in_lvl, out_lvl, N)
        if result:
            return ContinuousRule(d, 'quadratic', result[1], result[0])
        
        # 5. Gating / piecewise linear (captures SiLU/GELU)
        result = self._fit_gating(in_lvl, out_lvl, N)
        if result:
            return ContinuousRule(d, 'gating', result[1], result[0])
        
        # 6. Sigmoid / saturating (captures softmax compression)
        result = self._fit_sigmoid(in_lvl, out_lvl, N)
        if result:
            return ContinuousRule(d, 'sigmoid', result[1], result[0])
        
        # --- CROSS-DIMENSIONAL RULES ---
        
        # 7. Cross-dimensional linear (captures attention mixing)
        result = self._fit_cross_dim(d, in_lvl, out_lvl, all_in_lvl, N, D)
        if result:
            return ContinuousRule(d, 'cross_dim', result[1], result[0])
        
        # 8. Context (single neighbor)
        result = self._fit_context(d, in_lvl, out_lvl, all_in_lvl, D)
        if result:
            return ContinuousRule(d, 'context', result[1], result[0])
        
        # --- STRUCTURAL RULES ---
        
        # 9. Collapse
        result = self._fit_collapse(in_lvl, out_lvl)
        if result:
            return ContinuousRule(d, 'collapse', result[1], result[0])
        
        # --- SIGN-PATTERN RULES ---
        # Signs are the irreducible 1-bit decisions (doc 141).
        # XOR of signs = boundary crossing. Even when magnitudes are
        # chaotic, signs may follow structured patterns.
        
        # 10. Sign preserve
        result = self._fit_sign_preserve(in_sgn, out_sgn, in_raw, out_raw)
        if result:
            return ContinuousRule(d, 'sign_preserve', result[1], result[0])
        
        # 11. Sign flip
        result = self._fit_sign_flip(in_sgn, out_sgn, in_raw, out_raw)
        if result:
            return ContinuousRule(d, 'sign_flip', result[1], result[0])
        
        # 12. Sign XOR (cross-dimensional boundary crossing)
        if all_in_sgn is not None:
            result = self._fit_sign_xor(d, in_sgn, out_sgn, all_in_sgn,
                                        in_raw, out_raw, N, D)
            if result:
                return ContinuousRule(d, 'sign_xor', result[1], result[0])
        
        # 13. Sign gate (magnitude-dependent sign change)
        result = self._fit_sign_gate(in_sgn, out_sgn, in_lvl, in_raw, out_raw, N)
        if result:
            return ContinuousRule(d, 'sign_gate', result[1], result[0])
        
        # --- FALLBACK ---
        
        # 14. Unstructured — report the best we could do
        try:
            x = in_lvl.astype(float)
            y = out_lvl.astype(float)
            affine_coeffs = np.polyfit(x, y, 1)
            affine_r2 = self._r_squared(out_lvl, affine_coeffs[0]*x + affine_coeffs[1])
        except:
            affine_r2 = 0.0
            affine_coeffs = [1.0, 0.0]
        
        # Also report sign accuracy for the unstructured dimensions
        sign_match = float(np.mean(in_sgn == out_sgn))
        
        return ContinuousRule(d, 'unstructured', {
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            'best_affine_r2': affine_r2,
            'best_affine_slope': float(affine_coeffs[0]),
            'best_affine_intercept': float(affine_coeffs[1]),
            'sign_match_rate': sign_match,
        }, affine_r2)
    
    def _classify_archetype(self, rule_dist: Counter, D: int) -> str:
        """Classify overall transformation archetype from rule distribution."""
        total = sum(rule_dist.values())
        
        identity_frac = rule_dist.get('identity', 0) / total
        scale_frac = rule_dist.get('scale', 0) / total
        affine_frac = rule_dist.get('affine', 0) / total
        quadratic_frac = rule_dist.get('quadratic', 0) / total
        gating_frac = rule_dist.get('gating', 0) / total
        sigmoid_frac = rule_dist.get('sigmoid', 0) / total
        cross_dim_frac = rule_dist.get('cross_dim', 0) / total
        context_frac = rule_dist.get('context', 0) / total
        collapse_frac = rule_dist.get('collapse', 0) / total
        unstructured_frac = rule_dist.get('unstructured', 0) / total
        
        sign_preserve_frac = rule_dist.get('sign_preserve', 0) / total
        sign_flip_frac = rule_dist.get('sign_flip', 0) / total
        sign_xor_frac = rule_dist.get('sign_xor', 0) / total
        sign_gate_frac = rule_dist.get('sign_gate', 0) / total
        
        linear_frac = identity_frac + scale_frac + affine_frac
        nonlinear_frac = quadratic_frac + gating_frac + sigmoid_frac
        crossdim_frac = cross_dim_frac + context_frac
        sign_frac = sign_preserve_frac + sign_flip_frac + sign_xor_frac + sign_gate_frac
        structured_frac = linear_frac + nonlinear_frac + crossdim_frac + collapse_frac + sign_frac
        
        if identity_frac > 0.8:
            return "identity"
        elif scale_frac > 0.4:
            return "uniform_scale"
        elif affine_frac > 0.4:
            return "affine"
        elif gating_frac > 0.3:
            return "gating"
        elif quadratic_frac > 0.3:
            return "quadratic"
        elif sigmoid_frac > 0.3:
            return "sigmoid"
        elif nonlinear_frac > 0.3:
            return "nonlinear_mixed"
        elif cross_dim_frac > 0.3:
            return "cross_dimensional"
        elif context_frac > 0.3:
            return "context_dependent"
        elif sign_frac > 0.3:
            return "sign_pattern"
        elif collapse_frac > 0.3:
            return "collapse"
        elif linear_frac > 0.4:
            return "mostly_linear"
        elif structured_frac > 0.6:
            return "mixed_structured"
        elif unstructured_frac > 0.5:
            return "unstructured"
        else:
            return "mixed"
    
    def _build_phases(self, dim_rules: Dict[int, ContinuousRule]) -> List[ContinuousPhase]:
        """Group dimension rules into phases by type."""
        by_type = defaultdict(list)
        for d, rule in sorted(dim_rules.items()):
            by_type[rule.rule_type].append(rule)
        
        phases = []
        for rtype, rules in sorted(by_type.items(), key=lambda x: -len(x[1])):
            phases.append(ContinuousPhase(
                rules=rules,
                name=f"{rtype}_phase ({len(rules)} dims)",
            ))
        
        return phases
