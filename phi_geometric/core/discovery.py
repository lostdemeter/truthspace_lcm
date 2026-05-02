"""
Structure Discovery: Find geometric structure in observation data.

The spectrometer of the φ-Geometric Framework. Given observations of
(input → output) under varying contexts, discovers the minimal geometric
structure that explains the transformation:

    1. DETECT which inputs have inconsistent outputs (context-dependent)
    2. DISCOVER which context variable explains the inconsistency
    3. BUILD a gear train: coarse selector + fine selectors for ambiguities

This is the data-driven complement to ShapeProjector (top-down from ProblemSpec).
ShapeProjector says "what shape should the solution be?"
StructureDiscovery says "what structure IS in the data?"

Key Principles:
    - No gradient descent. No neural network.
    - Structure emerges from information-theoretic analysis
    - Gears are geometric selectors, not learned weights
    - The gear train IS the knowledge — not a model of knowledge

Ported from: evaluations/auto_context_detection.py (IPA demo)
Generalized from character transformations to arbitrary hashable types.

Author: TruthSpace LCM Project
Date: February 2026
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Hashable


# ============================================================================
# OBSERVATION TYPES
# ============================================================================

@dataclass
class Observation:
    """A single (output, context) observation.
    
    Attributes:
        output: The observed output value (any hashable type)
        context: Dict of context variable name → value
    """
    output: Hashable
    context: Dict[str, Hashable]


@dataclass
class DiscoveryResult:
    """Result of structure discovery on a set of observations.
    
    Attributes:
        input_value: The input being analyzed
        rule_type: 'identity', 'consistent', 'selector', 'geared'
        output: For identity/consistent, the single output value
        coarse_var: For selector/geared, the primary context variable
        selector_map: {context_value: output} for the coarse gear
        fine_gears: {coarse_value: FineGear} for ambiguous teeth
        default_output: Majority output (fallback)
        stats: Diagnostic information
    """
    input_value: Hashable
    rule_type: str
    output: Optional[Hashable] = None
    coarse_var: Optional[str] = None
    selector_map: Optional[Dict[Hashable, Hashable]] = None
    fine_gears: Optional[Dict] = None
    default_output: Optional[Hashable] = None
    stats: Dict = field(default_factory=dict)
    
    @property
    def is_identity(self) -> bool:
        return self.rule_type == 'identity'
    
    @property
    def is_context_dependent(self) -> bool:
        return self.rule_type in ('selector', 'geared')
    
    @property
    def gear_count(self) -> int:
        """Number of active gears (0 for identity/consistent, 1 for selector, 2 for geared)."""
        if self.rule_type == 'geared':
            n_fine = sum(1 for v in (self.fine_gears or {}).values()
                        if v[0] is not None)
            return 1 + (1 if n_fine > 0 else 0)
        elif self.rule_type == 'selector':
            return 1
        return 0


@dataclass
class FineGear:
    """A fine gear for one ambiguous tooth of the coarse gear.
    
    Attributes:
        variable: Context variable name for fine selection
        selector_map: {fine_value: output}
        channels: {output: set of fine_values}
        gain: Information gain of the fine gear
        zone_default: Majority output within this ambiguous zone
    """
    variable: Optional[str]
    selector_map: Dict[Hashable, Hashable]
    channels: Dict[Hashable, Set]
    gain: float
    zone_default: Hashable


# ============================================================================
# CORE ALGORITHMS
# ============================================================================

def compute_entropy(counts: Dict[Hashable, int], total: int) -> float:
    """Compute Shannon entropy from a count dict."""
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def detect_inconsistencies(
    observations_by_input: Dict[Hashable, List[Observation]]
) -> Tuple[Dict, Dict, Dict]:
    """Classify inputs by their output consistency.
    
    Args:
        observations_by_input: {input_value: [Observation, ...]}
    
    Returns:
        identity: {input: output} where input == output (pass-through)
        consistent: {input: output} where all observations agree on one output ≠ input
        inconsistent: {input: [Observation, ...]} where outputs vary by context
    """
    identity = {}
    consistent = {}
    inconsistent = {}
    
    for inp, obs_list in observations_by_input.items():
        output_set = set(ob.output for ob in obs_list)
        if len(output_set) == 1:
            out = output_set.pop()
            if out == inp:
                identity[inp] = out
            else:
                consistent[inp] = out
        else:
            inconsistent[inp] = obs_list
    
    return identity, consistent, inconsistent


def discover_selector(
    observations: List[Observation],
    candidate_vars: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[Dict], Optional[Dict], float]:
    """Find the context variable that best explains output variation.
    
    Uses information gain (entropy reduction) to rank candidates.
    
    Args:
        observations: List of (output, context) observations
        candidate_vars: Which context variables to test. If None, uses
                       all keys from the first observation's context.
    
    Returns:
        best_variable: Name of the best context variable
        selector_map: {context_value: output} mapping
        channels: {output: set(context_values)} reverse mapping
        gain: Information gain of the best variable
    """
    n_total = len(observations)
    if n_total == 0:
        return None, None, None, 0.0
    
    # Group by output
    output_groups = defaultdict(list)
    for ob in observations:
        output_groups[ob.output].append(ob.context)
    
    if len(output_groups) < 2:
        return None, None, None, 0.0
    
    # Base entropy
    base_counts = {out: len(ctxs) for out, ctxs in output_groups.items()}
    base_entropy = compute_entropy(base_counts, n_total)
    
    # Default to all context keys
    if candidate_vars is None:
        candidate_vars = list(observations[0].context.keys())
    
    results = []
    for var_name in candidate_vars:
        # Group by context value
        value_groups = defaultdict(lambda: defaultdict(int))
        for ob in observations:
            val = ob.context.get(var_name)
            value_groups[val][ob.output] += 1
        
        # Conditional entropy H(output | var)
        cond_entropy = 0.0
        for val, output_counts in value_groups.items():
            n_val = sum(output_counts.values())
            p_val = n_val / n_total
            val_entropy = compute_entropy(output_counts, n_val)
            cond_entropy += p_val * val_entropy
        
        gain = base_entropy - cond_entropy
        
        # Build selector map: majority output per context value
        selector_map = {}
        purity = 0
        for val, output_counts in value_groups.items():
            best_out = max(output_counts, key=output_counts.get)
            selector_map[val] = best_out
            purity += output_counts[best_out]
        
        accuracy = purity / n_total
        results.append((gain, accuracy, var_name, selector_map))
    
    if not results:
        return None, None, None, 0.0
    
    # Pick best by gain, break ties by accuracy
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_gain, _, best_var, best_map = results[0]
    
    # Build channels: reverse map
    channels = defaultdict(set)
    for val, out in best_map.items():
        channels[out].add(val)
    
    return best_var, best_map, dict(channels), best_gain


def discover_gears(
    observations: List[Observation],
    coarse_candidates: Optional[List[str]] = None,
    fine_candidates: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[Dict], Optional[Dict], Hashable, Dict]:
    """Find a gear train for context-dependent transformations.
    
    Gear 1 (coarse): The simplest selector that resolves the most cases.
                     Prefers high resolved-per-tooth ratio (coverage / cardinality).
    
    Gear 2 (fine):   For each ambiguous tooth in Gear 1, find a secondary
                     selector that resolves within that subset.
    
    This IS the geometric spectrometer: it discovers the minimal geometric
    structure that explains the observed transformations.
    
    Args:
        observations: List of Observation(output, context)
        coarse_candidates: Context vars for coarse gear (low cardinality preferred).
                          If None, uses all context keys.
        fine_candidates: Context vars for fine gear (can include high cardinality).
                        If None, uses all context keys.
    
    Returns:
        coarse_var: Name of the coarse gear variable (None if no structure found)
        pure_map: {coarse_value: output} for resolved teeth
        fine_gears: {coarse_value: (fine_var, fine_map, channels, gain, zone_default)}
        default_output: Majority output across all observations
        stats: Diagnostic info dict
    """
    n_total = len(observations)
    if n_total == 0:
        return None, None, None, None, {}
    
    # Overall default: majority output
    output_counts = defaultdict(int)
    for ob in observations:
        output_counts[ob.output] += 1
    default_output = max(output_counts, key=output_counts.get)
    
    # Default candidates from context keys
    all_keys = list(observations[0].context.keys())
    if coarse_candidates is None:
        coarse_candidates = all_keys
    if fine_candidates is None:
        fine_candidates = all_keys
    
    best_coarse = None
    best_score = -1
    
    for var_name in coarse_candidates:
        by_val = defaultdict(list)
        for ob in observations:
            by_val[ob.context.get(var_name)].append(ob)
        
        n_pure = 0
        n_resolved = 0
        n_ambiguous = 0
        pure_map = {}
        ambiguous_vals = {}
        
        for val, obs_group in by_val.items():
            outputs = set(ob.output for ob in obs_group)
            if len(outputs) == 1:
                n_pure += 1
                n_resolved += len(obs_group)
                pure_map[val] = next(iter(outputs))
            else:
                n_ambiguous += 1
                ambiguous_vals[val] = obs_group
        
        if n_pure == 0:
            continue
        
        # Score: resolved fraction / total teeth
        n_teeth = n_pure + n_ambiguous
        score = n_resolved / n_teeth
        
        if score > best_score:
            best_score = score
            best_coarse = {
                'var': var_name,
                'pure_map': pure_map,
                'ambiguous': ambiguous_vals,
                'n_resolved': n_resolved,
                'n_pure': n_pure,
                'n_ambiguous': n_ambiguous,
                'n_teeth': n_teeth,
            }
    
    if best_coarse is None:
        return None, None, None, default_output, {}
    
    # For each ambiguous tooth, find the fine gear
    fine_gears = {}
    total_fine_resolved = 0
    
    for val, obs_group in best_coarse['ambiguous'].items():
        # Zone default: majority within this subset
        zone_counts = defaultdict(int)
        for ob in obs_group:
            zone_counts[ob.output] += 1
        zone_default = max(zone_counts, key=zone_counts.get)
        
        # Try fine candidates
        fine_var, fine_map, fine_channels, fine_gain = discover_selector(
            obs_group, fine_candidates
        )
        if fine_var and fine_gain > 0.0:
            fine_gears[val] = (fine_var, fine_map, fine_channels, fine_gain,
                               zone_default)
            for ob in obs_group:
                if fine_map.get(ob.context.get(fine_var)) == ob.output:
                    total_fine_resolved += 1
        else:
            fine_gears[val] = (None, {}, {}, 0.0, zone_default)
    
    stats = {
        'coarse_resolved': best_coarse['n_resolved'],
        'coarse_teeth': best_coarse['n_teeth'],
        'coarse_pure': best_coarse['n_pure'],
        'coarse_ambiguous': best_coarse['n_ambiguous'],
        'fine_gears_active': sum(1 for v in fine_gears.values() if v[0] is not None),
        'fine_resolved': total_fine_resolved,
        'total': n_total,
    }
    
    return (best_coarse['var'], best_coarse['pure_map'],
            fine_gears, default_output, stats)


# ============================================================================
# TRANSFORM RULE — executable geometric structure
# ============================================================================

class TransformRule:
    """An executable transformation rule discovered from data.
    
    This is the geometric primitive: a rule that maps input → output
    given context. The rule type determines the geometric complexity:
    
        identity:   No transformation (pass-through)
        consistent: Fixed output regardless of context (RECT)
        selector:   Output depends on one context variable (RECT × SELECTOR)
        geared:     Coarse gear + fine gears for ambiguous teeth
                    (RECT × SELECTOR + fallthrough RECT × SELECTOR)
    """
    
    def __init__(self, input_value: Hashable, rule_type: str, **params):
        self.input_value = input_value
        self.rule_type = rule_type
        self.params = params
    
    def apply(self, value: Hashable, context: Optional[Dict] = None) -> Hashable:
        """Apply this rule to a value with optional context.
        
        Returns the transformed value, or the original if no match.
        """
        if value != self.input_value:
            return value
        
        if self.rule_type == 'identity':
            return value
        
        elif self.rule_type == 'consistent':
            return self.params['output']
        
        elif self.rule_type == 'selector':
            if context is None:
                return self.params.get('default_output', value)
            var = self.params['variable']
            ctx_val = context.get(var)
            return self.params['selector_map'].get(
                ctx_val, self.params.get('default_output', value))
        
        elif self.rule_type == 'geared':
            if context is None:
                return self.params.get('default_output', value)
            
            coarse_var = self.params['coarse_var']
            coarse_val = context.get(coarse_var)
            pure_map = self.params['pure_map']
            
            # Gear 1: coarse
            if coarse_val in pure_map:
                return pure_map[coarse_val]
            
            # Fallthrough → engage fine gear
            fine_gears = self.params.get('fine_gears', {})
            if coarse_val in fine_gears:
                fine_var, fine_map, _, _, zone_default = fine_gears[coarse_val]
                if fine_var is not None:
                    fine_val = context.get(fine_var)
                    if fine_val in fine_map:
                        return fine_map[fine_val]
                return zone_default
            
            return self.params.get('default_output', value)
        
        return value
    
    def describe(self) -> str:
        """Human-readable description of this rule."""
        inp = repr(self.input_value)
        
        if self.rule_type == 'identity':
            return f"  {inp} → {inp}  (identity)"
        
        elif self.rule_type == 'consistent':
            out = repr(self.params['output'])
            return f"  {inp} → {out}  (consistent)"
        
        elif self.rule_type == 'selector':
            var = self.params['variable']
            channels = self.params.get('channels', {})
            lines = [f"  {inp} → context-dependent on '{var}':"]
            for out, ctx_vals in sorted(channels.items(), key=str):
                vals_str = ', '.join(repr(v) for v in sorted(ctx_vals, key=str))
                lines.append(f"    → {repr(out)} when {var} ∈ {{{vals_str}}}")
            lines.append(f"    default: {repr(self.params.get('default_output'))}")
            return '\n'.join(lines)
        
        elif self.rule_type == 'geared':
            coarse_var = self.params['coarse_var']
            pure_map = self.params['pure_map']
            fine_gears = self.params.get('fine_gears', {})
            stats = self.params.get('stats', {})
            
            lines = [f"  {inp} → GEARED on '{coarse_var}':"]
            
            # Gear 1
            resolved_by_output = defaultdict(list)
            for val, out in sorted(pure_map.items(), key=str):
                resolved_by_output[out].append(val)
            
            lines.append(f"    Gear 1 (coarse): {stats.get('coarse_pure', '?')} pure, "
                        f"{stats.get('coarse_ambiguous', '?')} ambiguous")
            for out, vals in sorted(resolved_by_output.items(), key=str):
                vals_str = ', '.join(repr(v) for v in sorted(vals, key=str))
                lines.append(f"      → {repr(out)} when {coarse_var} ∈ {{{vals_str}}}")
            
            # Gear 2
            active = {k: v for k, v in fine_gears.items() if v[0] is not None}
            if active:
                lines.append(f"    Gear 2 (fine): {len(active)} fallthrough(s):")
                for cv, (fv, fm, _, fg, zd) in sorted(fine_gears.items(), key=str):
                    if fv is None:
                        lines.append(f"      {coarse_var}={repr(cv)} → default {repr(zd)}")
                    else:
                        lines.append(f"      {coarse_var}={repr(cv)} → engage '{fv}':")
                        for fval, out in sorted(fm.items(), key=str):
                            lines.append(f"        {fv}={repr(fval)} → {repr(out)}")
            
            lines.append(f"    default: {repr(self.params.get('default_output'))}")
            return '\n'.join(lines)
        
        return f"  {inp} → (unknown rule type: {self.rule_type})"


# ============================================================================
# STRUCTURE DISCOVERY — the spectrometer
# ============================================================================

class StructureDiscovery:
    """The geometric spectrometer: discovers structure from observation data.
    
    Feed it (input, output, context) observations. It discovers:
    - Which inputs have consistent outputs (simple rules)
    - Which inputs are context-dependent (need selectors)
    - What gear train resolves each context-dependent input
    
    This is the data-driven learning primitive of the φ-Geometric Framework.
    No gradients. No training loops. Pure information-theoretic structure detection.
    
    Example:
        discovery = StructureDiscovery()
        
        # Add observations
        discovery.observe('c', 'k', {'next_char': 'a'})
        discovery.observe('c', 's', {'next_char': 'i'})
        discovery.observe('c', 'k', {'next_char': 'o'})
        
        # Discover structure
        rules = discovery.discover()
        
        # Apply
        for rule in rules:
            print(rule.describe())
    """
    
    def __init__(
        self,
        coarse_candidates: Optional[List[str]] = None,
        fine_candidates: Optional[List[str]] = None
    ):
        """
        Args:
            coarse_candidates: Context vars for coarse gear selection.
                              Low-cardinality vars preferred.
            fine_candidates: Context vars for fine gear (can be higher cardinality).
        """
        self._observations: Dict[Hashable, List[Observation]] = defaultdict(list)
        self.coarse_candidates = coarse_candidates
        self.fine_candidates = fine_candidates
        self._rules: Optional[List[TransformRule]] = None
    
    def observe(self, input_value: Hashable, output_value: Hashable,
                context: Dict[str, Hashable]):
        """Record an observation: input → output under context.
        
        Args:
            input_value: The input (e.g., 'c', 42, (1,0,1))
            output_value: The observed output
            context: Dict of context variables and their values
        """
        self._observations[input_value].append(
            Observation(output=output_value, context=context)
        )
        self._rules = None  # invalidate cached rules
    
    def observe_batch(self, triples: List[Tuple[Hashable, Hashable, Dict]]):
        """Record multiple observations at once.
        
        Args:
            triples: List of (input, output, context) tuples
        """
        for inp, out, ctx in triples:
            self.observe(inp, out, ctx)
    
    @property
    def n_observations(self) -> int:
        return sum(len(obs) for obs in self._observations.values())
    
    @property
    def n_inputs(self) -> int:
        return len(self._observations)
    
    def discover(self) -> List[TransformRule]:
        """Run the spectrometer: discover geometric structure from observations.
        
        Returns:
            List of TransformRule objects, one per observed input value.
        """
        if self._rules is not None:
            return self._rules
        
        identity, consistent, inconsistent = detect_inconsistencies(
            self._observations
        )
        
        rules = []
        
        # Identity rules
        for inp in sorted(identity, key=str):
            rules.append(TransformRule(inp, 'identity'))
        
        # Consistent rules (fixed output)
        for inp, out in sorted(consistent.items(), key=str):
            rules.append(TransformRule(inp, 'consistent', output=out))
        
        # Context-dependent rules — engage the spectrometer
        for inp, obs_list in sorted(inconsistent.items(), key=str):
            coarse_var, pure_map, fine_gears, default, stats = discover_gears(
                obs_list,
                coarse_candidates=self.coarse_candidates,
                fine_candidates=self.fine_candidates
            )
            
            if coarse_var is None:
                # No structure found — majority vote
                rules.append(TransformRule(inp, 'consistent', output=default))
            elif not any(v[0] is not None for v in (fine_gears or {}).values()):
                # Coarse gear resolves everything — simple selector
                channels = defaultdict(set)
                for val, out in pure_map.items():
                    channels[out].add(val)
                rules.append(TransformRule(inp, 'selector',
                    variable=coarse_var,
                    selector_map=pure_map,
                    channels=dict(channels),
                    default_output=default,
                    stats=stats
                ))
            else:
                # Full gear train
                rules.append(TransformRule(inp, 'geared',
                    coarse_var=coarse_var,
                    pure_map=pure_map,
                    fine_gears=fine_gears,
                    default_output=default,
                    stats=stats
                ))
        
        self._rules = rules
        return rules
    
    def discover_for(self, input_value: Hashable) -> Optional[DiscoveryResult]:
        """Discover structure for a single input value.
        
        Returns DiscoveryResult or None if no observations exist.
        """
        obs_list = self._observations.get(input_value)
        if not obs_list:
            return None
        
        output_set = set(ob.output for ob in obs_list)
        
        if len(output_set) == 1:
            out = output_set.pop()
            if out == input_value:
                return DiscoveryResult(input_value, 'identity', output=out)
            return DiscoveryResult(input_value, 'consistent', output=out)
        
        coarse_var, pure_map, fine_gears, default, stats = discover_gears(
            obs_list,
            coarse_candidates=self.coarse_candidates,
            fine_candidates=self.fine_candidates
        )
        
        if coarse_var is None:
            return DiscoveryResult(input_value, 'consistent',
                                  output=default, default_output=default, stats=stats)
        
        has_fine = any(v[0] is not None for v in (fine_gears or {}).values())
        return DiscoveryResult(
            input_value=input_value,
            rule_type='geared' if has_fine else 'selector',
            coarse_var=coarse_var,
            selector_map=pure_map,
            fine_gears=fine_gears,
            default_output=default,
            stats=stats
        )
    
    def describe(self) -> str:
        """Human-readable summary of discovered structure."""
        rules = self.discover()
        
        n_identity = sum(1 for r in rules if r.rule_type == 'identity')
        n_consistent = sum(1 for r in rules if r.rule_type == 'consistent')
        n_selector = sum(1 for r in rules if r.rule_type == 'selector')
        n_geared = sum(1 for r in rules if r.rule_type == 'geared')
        
        lines = [
            f"StructureDiscovery: {self.n_observations} observations, "
            f"{self.n_inputs} inputs",
            f"  {n_identity} identity (pass-through)",
            f"  {n_consistent} consistent (fixed output)",
            f"  {n_selector} selector (1-gear context-dependent)",
            f"  {n_geared} geared (multi-gear context-dependent)",
            "",
        ]
        
        # Show non-identity rules
        for r in rules:
            if r.rule_type != 'identity':
                lines.append(r.describe())
        
        return '\n'.join(lines)
