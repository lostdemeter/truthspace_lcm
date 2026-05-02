"""
Phase Discovery: Automatically discover cascade phase structure from training data.

Strategy: INCONSISTENCY-DRIVEN DISCOVERY
    1. Start with naive 1→1 token mapping across all training pairs
    2. Detect inconsistencies (same input → different outputs)
    3. For each inconsistency, try resolving by:
       a) MULTI-TOKEN PATTERN: Does grouping adjacent tokens resolve it?
       b) CONTEXT DEPENDENCE: Does a neighbor variable explain it?
       c) RESIDUAL: Unresolvable → majority vote
    4. Phases EMERGE from the resolution layers:
       - Multi-token patterns → early phase (longest match first)
       - Context-dependent → middle phase
       - Consistent → final phase
    5. Freeze detection: if multi-token output collides with single-token input

This mirrors how StructureDiscovery works for individual rules, but
operates at the PHASE level — discovering the cascade structure itself.

Author: TruthSpace LCM Project
Date: February 2026
"""

import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)

from .discovery import (
    StructureDiscovery, TransformRule, Observation,
    detect_inconsistencies, discover_gears, discover_selector
)
from .cascade_navigator import (
    CascadeNavigator, Phase, default_context_extractor,
    geometric_context_extractor
)


# ============================================================================
# OBSERVATION RECORDS
# ============================================================================

@dataclass
class TokenMapping:
    """A single observed token mapping with full context.
    
    Attributes:
        input_token: The input token
        output_token: The observed output token
        position: Position in the input sequence
        input_seq: The full input sequence (for context extraction)
        output_seq: The full output sequence
        pair_index: Which training pair this came from
    """
    input_token: Hashable
    output_token: Hashable
    position: int
    input_seq: List[Hashable]
    output_seq: List[Hashable]
    pair_index: int
    
    @property
    def is_identity(self) -> bool:
        return self.input_token == self.output_token
    
    @property
    def changed(self) -> bool:
        return self.input_token != self.output_token


@dataclass
class MultiTokenPattern:
    """A discovered multi-token pattern that resolves an inconsistency.
    
    Attributes:
        input_tokens: Tuple of input tokens (e.g., ('s', 'h'))
        output_tokens: What they map to (e.g., ('ʃ',))
        evidence_count: How many training pairs show this pattern
        resolves: Set of single-token inconsistencies this pattern explains
    """
    input_tokens: Tuple[Hashable, ...]
    output_tokens: Tuple[Hashable, ...]
    evidence_count: int = 0
    resolves: Set[Hashable] = field(default_factory=set)
    
    @property
    def arity(self) -> int:
        return len(self.input_tokens)
    
    @property
    def key(self) -> Tuple[Hashable, ...]:
        return self.input_tokens
    
    def __repr__(self):
        inp = ''.join(str(t) for t in self.input_tokens)
        out = ''.join(str(t) for t in self.output_tokens)
        return f"Multi({inp}→{out} ×{self.evidence_count})"


@dataclass
class ExpandPattern:
    """A discovered 1→N expansion pattern (mirror of MultiTokenPattern).
    
    Attributes:
        input_token: Single input token (e.g., 'x')
        output_tokens: What it expands to (e.g., ('k', 's'))
        evidence_count: How many training pairs show this pattern
    """
    input_token: Hashable
    output_tokens: Tuple[Hashable, ...]
    evidence_count: int = 0
    
    @property
    def expansion(self) -> int:
        return len(self.output_tokens)
    
    def __repr__(self):
        out = ''.join(str(t) for t in self.output_tokens)
        return f"Expand({self.input_token}→{out} ×{self.evidence_count})"


# ============================================================================
# PHASE CANDIDATE
# ============================================================================

@dataclass
class PhaseCandidate:
    """A proposed phase in the cascade, with diagnostic reasoning.
    
    Attributes:
        name: Descriptive name
        reason: Why this is a separate phase
        order_priority: Higher = earlier in cascade
        freeze: Whether outputs should be frozen
        context_dependent: Whether rules need context
        multi_token_patterns: For multi-token phases (N→1 collapse)
        expand_patterns: For expand phases (1→N expansion)
        token_rules: For single-token phases (input → TransformRule)
        rule_observations: Raw observations for context-dependent rules
    """
    name: str
    reason: str
    order_priority: int = 0
    freeze: bool = False
    context_dependent: bool = False
    multi_token_patterns: List[MultiTokenPattern] = field(default_factory=list)
    expand_patterns: List[ExpandPattern] = field(default_factory=list)
    token_rules: Dict[Hashable, Tuple[Hashable, ...]] = field(default_factory=dict)
    rule_observations: Dict[Hashable, List[Tuple]] = field(default_factory=dict)
    
    @property
    def n_rules(self) -> int:
        return (len(self.multi_token_patterns) + len(self.expand_patterns) +
                len(self.token_rules) + len(self.rule_observations))
    
    def describe(self) -> str:
        lines = [f"Phase '{self.name}' (priority {self.order_priority}):"]
        lines.append(f"  Reason: {self.reason}")
        flags = []
        if self.freeze:
            flags.append("freeze")
        if self.context_dependent:
            flags.append("context-dependent")
        if flags:
            lines.append(f"  Flags: {', '.join(flags)}")
        
        if self.multi_token_patterns:
            lines.append(f"  Multi-token patterns ({len(self.multi_token_patterns)}):")
            for mp in self.multi_token_patterns[:10]:
                lines.append(f"    {mp}")
        
        if self.expand_patterns:
            lines.append(f"  Expand patterns ({len(self.expand_patterns)}):")
            for ep in self.expand_patterns[:10]:
                lines.append(f"    {ep}")
        
        if self.token_rules:
            lines.append(f"  Token rules ({len(self.token_rules)}):")
            for tok, out in sorted(self.token_rules.items(), key=str):
                out_str = ''.join(str(t) for t in out)
                lines.append(f"    {tok} → {out_str}")
        
        if self.rule_observations:
            lines.append(f"  Context-dependent rules ({len(self.rule_observations)}):")
            for tok, obs_list in sorted(self.rule_observations.items(), key=str):
                outputs = defaultdict(int)
                for out, _ in obs_list:
                    outputs[out] += 1
                out_str = ', '.join(f"{o}×{c}" for o, c in 
                                   sorted(outputs.items(), key=lambda x: -x[1]))
                lines.append(f"    {tok} → {{{out_str}}}")
        
        return '\n'.join(lines)


# ============================================================================
# PHASE DISCOVERY RESULT
# ============================================================================

@dataclass
class PhaseDiscoveryResult:
    """Result of automatic phase discovery."""
    phases: List[PhaseCandidate]
    training_pairs: List[Tuple[List, List]]
    diagnostics: Dict = field(default_factory=dict)
    geometric: bool = False
    
    @property
    def n_phases(self) -> int:
        return len(self.phases)
    
    @property
    def n_rules(self) -> int:
        return sum(p.n_rules for p in self.phases)
    
    @property
    def archetype(self) -> str:
        """Return the archetype signature as an underscore-joined string.
        
        Examples: 'map', 'context_map', 'collapse_context_map',
                  'expand_map', 'expand_collapse_map'
        """
        types = []
        for phase in self.phases:
            if phase.multi_token_patterns:
                types.append('collapse')
            elif phase.expand_patterns:
                types.append('expand')
            elif phase.context_dependent:
                types.append('context')
            else:
                types.append('map')
        return '_'.join(types)
    
    @property
    def phase_types(self) -> list:
        """Return ordered list of phase type strings."""
        return self.archetype.split('_')
    
    def describe(self) -> str:
        lines = [
            f"PhaseDiscovery: {self.n_phases} phases, {self.n_rules} rules "
            f"from {len(self.training_pairs)} training pairs",
            ""
        ]
        
        # Show diagnostics
        d = self.diagnostics
        if d:
            lines.append(f"  Step 1 - Naive 1→1 mappings: "
                        f"{d.get('n_identity', '?')} identity, "
                        f"{d.get('n_consistent', '?')} consistent, "
                        f"{d.get('n_inconsistent', '?')} inconsistent")
            if d.get('multi_resolved'):
                lines.append(f"  Step 2 - Multi-token resolution: "
                            f"{len(d['multi_resolved'])} inconsistencies resolved "
                            f"by {d.get('n_multi_patterns', '?')} multi-token patterns")
            if d.get('context_resolved'):
                lines.append(f"  Step 3 - Context resolution: "
                            f"{len(d['context_resolved'])} resolved by context")
            if d.get('collisions'):
                lines.append(f"  Step 4 - Freeze detection: "
                            f"{len(d['collisions'])} output-input collisions")
            lines.append("")
        
        for p in self.phases:
            lines.append(p.describe())
            lines.append("")
        
        return '\n'.join(lines)
    
    def to_navigator(
        self,
        context_extractor=None
    ) -> CascadeNavigator:
        """Build an executable CascadeNavigator from discovered phases.
        
        Multi-token phases become collapse patterns (pre-processing).
        Single-token phases become element-by-element Phase objects.
        """
        # Use geometric context extractor if discovery used geometric mode
        if context_extractor is None and self.geometric:
            context_extractor = geometric_context_extractor
        
        nav = CascadeNavigator()
        
        for pc in self.phases:
            if pc.multi_token_patterns:
                # Multi-token → collapse patterns (not a Phase)
                should_freeze = pc.freeze
                for mp in pc.multi_token_patterns:
                    nav.add_collapse(mp.input_tokens, mp.output_tokens,
                                    freeze=should_freeze)
            elif pc.expand_patterns:
                # Expand → expand patterns (pre-processing, mirror of collapse)
                for ep in pc.expand_patterns:
                    nav.add_expand(ep.input_token, ep.output_tokens)
            elif pc.context_dependent:
                phase = _build_context_phase(pc, context_extractor)
                nav.add_phase(phase)
            else:
                phase = _build_simple_phase(pc, context_extractor)
                nav.add_phase(phase)
        
        return nav
    
    def validate(self) -> Dict:
        """Validate discovered phases against training data."""
        nav = self.to_navigator()
        correct = 0
        total = 0
        errors = []
        
        for inp_seq, expected_seq in self.training_pairs:
            trace = nav.execute(inp_seq)
            actual = trace.output_elements
            
            max_len = max(len(expected_seq), len(actual))
            for i in range(max_len):
                total += 1
                exp = expected_seq[i] if i < len(expected_seq) else '<missing>'
                act = actual[i] if i < len(actual) else '<missing>'
                if exp == act:
                    correct += 1
                else:
                    errors.append({
                        'input': inp_seq,
                        'position': i,
                        'expected': exp,
                        'actual': act,
                    })
        
        return {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0.0,
            'errors': errors[:20],
        }


# ============================================================================
# PHASE BUILDING HELPERS
# ============================================================================

def _build_multi_token_phase(pc: PhaseCandidate, context_extractor=None) -> Phase:
    """Build a Phase for multi-token patterns (digraphs, etc.).
    
    Encodes multi-token patterns as context-dependent single-token rules:
    first_token when next=second_token → output.
    """
    phase = Phase(
        name=pc.name,
        freeze_outputs=pc.freeze,
        use_original_context=True,
        context_extractor=context_extractor
    )
    
    # Group by first token to handle conflicts (e.g., 's' starts both 'sh' and 'st')
    by_first = defaultdict(list)
    for mp in pc.multi_token_patterns:
        by_first[mp.input_tokens[0]].append(mp)
    
    for first_tok, patterns in by_first.items():
        if len(patterns) == 1:
            mp = patterns[0]
            out_str = mp.output_tokens[0] if len(mp.output_tokens) == 1 else \
                      ''.join(str(t) for t in mp.output_tokens)
            
            if mp.arity == 2:
                second = mp.input_tokens[1]
                rule = TransformRule(
                    first_tok, 'selector',
                    variable='next',
                    selector_map={second: out_str},
                    default_output=first_tok
                )
            else:
                rule = TransformRule(first_tok, 'consistent', output=out_str)
        else:
            # Multiple patterns starting with same token
            selector_map = {}
            for mp in patterns:
                out_str = mp.output_tokens[0] if len(mp.output_tokens) == 1 else \
                          ''.join(str(t) for t in mp.output_tokens)
                if mp.arity >= 2:
                    selector_map[mp.input_tokens[1]] = out_str
            
            rule = TransformRule(
                first_tok, 'selector',
                variable='next',
                selector_map=selector_map,
                default_output=first_tok
            )
        
        phase.add_rule(rule)
    
    return phase


def _build_context_phase(pc: PhaseCandidate, context_extractor=None) -> Phase:
    """Build a Phase for context-dependent single-token rules."""
    phase = Phase(
        name=pc.name,
        freeze_outputs=pc.freeze,
        use_original_context=True,
        context_extractor=context_extractor
    )
    
    for token, obs_list in pc.rule_observations.items():
        sd = StructureDiscovery()
        for out_val, ctx in obs_list:
            sd.observe(token, out_val, ctx)
        
        rules = sd.discover()
        for rule in rules:
            phase.add_rule(rule)
    
    return phase


def _build_simple_phase(pc: PhaseCandidate, context_extractor=None) -> Phase:
    """Build a Phase for simple consistent single-token rules."""
    phase = Phase(
        name=pc.name,
        freeze_outputs=pc.freeze,
        use_original_context=True,
        context_extractor=context_extractor
    )
    
    for token, out_tokens in pc.token_rules.items():
        out_str = out_tokens[0] if len(out_tokens) == 1 else \
                  ''.join(str(t) for t in out_tokens)
        phase.add_rule(TransformRule(token, 'consistent', output=out_str))
    
    return phase


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

def _extract_context(
    seq: List[Hashable],
    pos: int,
    window: int = 3
) -> Dict[str, Hashable]:
    """Extract context dict for a position in a sequence."""
    n = len(seq)
    ctx = {
        'position': pos,
        'length': n,
        'is_start': pos == 0,
        'is_end': pos == n - 1,
    }
    
    for offset in range(1, window + 1):
        prev_key = 'prev' if offset == 1 else f'prev_{offset}'
        next_key = 'next' if offset == 1 else f'next_{offset}'
        ctx[prev_key] = seq[pos - offset] if pos - offset >= 0 else None
        ctx[next_key] = seq[pos + offset] if pos + offset < n else None
    
    return ctx


# φ-level bins: which distances map to which geometric level
# Level 0: distance 1      (weight φ^0 = 1.000) — immediate neighbor
# Level 1: distance 2-3    (weight φ^-1 = 0.618) — near context
# Level 2: distance 4-7    (weight φ^-2 = 0.382) — medium context
# Level 3: distance 8-12   (weight φ^-3 = 0.236) — far context
_PHI_LEVEL_RANGES = [
    (1, 1),    # level 0: distance 1
    (2, 3),    # level 1: distance 2-3
    (4, 7),    # level 2: distance 4-7
    (8, 12),   # level 3: distance 8-12
]


def _distance_to_phi_level(distance: int) -> int:
    """Map an absolute distance to its φ-level bin."""
    for level, (lo, hi) in enumerate(_PHI_LEVEL_RANGES):
        if lo <= distance <= hi:
            return level
    return len(_PHI_LEVEL_RANGES)  # overflow level


def _extract_geometric_context(
    seq: List[Hashable],
    pos: int,
    max_levels: int = 4
) -> Dict[str, Hashable]:
    """Extract context using φ-level geometric binning.
    
    Instead of fixed prev_1, prev_2, ... prev_N (N features for N distances),
    we bin by φ-levels:
        phi_prev_0 = token at distance 1     (immediate)
        phi_prev_1 = token at distance 2-3   (near)
        phi_prev_2 = token at distance 4-7   (medium)
        phi_prev_3 = token at distance 8-12  (far)
    
    This covers distance 1-12 with just 4 features per direction,
    mirroring how attention decays: φ^(-level) weighting.
    
    For levels spanning multiple distances (level 1+), we provide both
    the NEAREST and FARTHEST tokens in the range. This mirrors how
    attention considers all keys within a range, not just the closest.
    """
    n = len(seq)
    ctx = {
        'position': pos,
        'length': n,
        'is_start': pos == 0,
        'is_end': pos == n - 1,
    }
    
    # Also include raw prev/next for backward compatibility
    ctx['prev'] = seq[pos - 1] if pos > 0 else None
    ctx['next'] = seq[pos + 1] if pos + 1 < n else None
    
    # φ-level binned context
    for level in range(min(max_levels, len(_PHI_LEVEL_RANGES))):
        lo, hi = _PHI_LEVEL_RANGES[level]
        
        # Previous direction: nearest and farthest in range
        prev_near = None
        prev_far = None
        for d in range(lo, hi + 1):
            idx = pos - d
            if 0 <= idx < n:
                if prev_near is None:
                    prev_near = seq[idx]
                prev_far = seq[idx]
        ctx[f'phi_prev_{level}'] = prev_near
        if hi > lo:  # multi-distance levels get a _far variant
            ctx[f'phi_prev_{level}_far'] = prev_far
        
        # Next direction: nearest and farthest in range
        next_near = None
        next_far = None
        for d in range(lo, hi + 1):
            idx = pos + d
            if 0 <= idx < n:
                if next_near is None:
                    next_near = seq[idx]
                next_far = seq[idx]
        ctx[f'phi_next_{level}'] = next_near
        if hi > lo:
            ctx[f'phi_next_{level}_far'] = next_far
    
    return ctx


# ============================================================================
# PHASE DISCOVERY ENGINE
# ============================================================================

class PhaseDiscovery:
    """Automatically discover cascade phase structure from training data.
    
    The approach is INCONSISTENCY-DRIVEN:
    1. Map all training pairs as naive 1→1 token correspondences
    2. Find tokens with inconsistent outputs
    3. Try resolving inconsistencies by multi-token grouping
    4. Try resolving remaining inconsistencies by context
    5. Phases emerge from the resolution layers
    
    Example:
        pd = PhaseDiscovery()
        
        pd.add_pair(list('ship'), list('ʃɪp'))
        pd.add_pair(list('sit'), list('sɪt'))
        pd.add_pair(list('cat'), list('kæt'))
        pd.add_pair(list('city'), list('sɪti'))
        
        result = pd.discover()
        print(result.describe())
        
        nav = result.to_navigator()
        trace = nav.execute(list('ship'))
    """
    
    def __init__(self, context_window: int = 3, max_multi: int = 4,
                 geometric: bool = False):
        """
        Args:
            context_window: Neighbor tokens to include in context
            max_multi: Maximum multi-token pattern size to try
            geometric: If True, use φ-level geometric context binning
                      instead of fixed-window context. Covers distance
                      1-12 with 4 features per direction via φ-decay
                      levels, mirroring how attention naturally works.
        """
        self.training_pairs: List[Tuple[List, List]] = []
        self.context_window = context_window
        self.max_multi = max_multi
        self.geometric = geometric
    
    def add_pair(self, input_seq: List[Hashable], output_seq: List[Hashable]):
        """Add a training pair."""
        self.training_pairs.append((list(input_seq), list(output_seq)))
    
    def add_pairs(self, pairs: List[Tuple[List, List]]):
        """Add multiple training pairs."""
        for inp, out in pairs:
            self.add_pair(inp, out)
    
    def discover(self) -> PhaseDiscoveryResult:
        """Run inconsistency-driven phase discovery.
        
        Returns:
            PhaseDiscoveryResult with proposed phases and diagnostics
        """
        diagnostics = {}
        
        # ============================================================
        # STEP 1: Naive 1→1 mapping — collect all token observations
        # ============================================================
        token_obs = self._collect_naive_mappings()
        
        # Classify: identity, consistent, inconsistent
        identity_tokens = set()
        consistent_tokens = {}       # token → output_token
        inconsistent_tokens = {}     # token → list of (output, context, pair_idx, pos)
        
        for token, obs_list in token_obs.items():
            outputs = set(o for o, _, _, _ in obs_list)
            non_identity = outputs - {token}
            
            if len(non_identity) == 0:
                identity_tokens.add(token)
            elif len(non_identity) == 1 and token not in non_identity:
                # Consistent: always maps to the same (non-identity) output
                # But check if it sometimes stays as identity too
                out_counts = defaultdict(int)
                for o, _, _, _ in obs_list:
                    out_counts[o] += 1
                if len(out_counts) == 1:
                    consistent_tokens[token] = next(iter(non_identity))
                else:
                    # Sometimes changes, sometimes doesn't → inconsistent
                    inconsistent_tokens[token] = obs_list
            else:
                inconsistent_tokens[token] = obs_list
        
        diagnostics['n_identity'] = len(identity_tokens)
        diagnostics['n_consistent'] = len(consistent_tokens)
        diagnostics['n_inconsistent'] = len(inconsistent_tokens)
        
        # ============================================================
        # STEP 2: Try resolving inconsistencies with multi-token patterns
        # ============================================================
        multi_patterns, multi_resolved = self._resolve_with_multi_token(
            inconsistent_tokens, known_consistent=consistent_tokens
        )
        
        diagnostics['multi_resolved'] = multi_resolved
        diagnostics['n_multi_patterns'] = len(multi_patterns)
        
        # Remove resolved inconsistencies and update consistent/inconsistent
        still_inconsistent = {}
        newly_consistent = {}
        
        for token, obs_list in inconsistent_tokens.items():
            if token in multi_resolved:
                # Filter out observations explained by multi-token patterns
                remaining = []
                for out, ctx, pair_idx, pos in obs_list:
                    if not self._explained_by_multi(token, pair_idx, pos, 
                                                    multi_patterns):
                        remaining.append((out, ctx, pair_idx, pos))
                
                if not remaining:
                    continue  # fully explained
                
                # Check if remaining are consistent
                rem_outputs = set(o for o, _, _, _ in remaining)
                non_id = rem_outputs - {token}
                if len(non_id) <= 1:
                    if non_id:
                        newly_consistent[token] = next(iter(non_id))
                    # else: all identity after removing multi-token cases
                else:
                    still_inconsistent[token] = remaining
            else:
                still_inconsistent[token] = obs_list
        
        consistent_tokens.update(newly_consistent)
        
        # ============================================================
        # STEP 2b: Try resolving with expand patterns (1→N)
        # ============================================================
        expand_patterns = self._resolve_with_expand(
            known_consistent=consistent_tokens
        )
        
        diagnostics['n_expand_patterns'] = len(expand_patterns)
        
        # Tokens explained by expand patterns should be removed from
        # inconsistent set (they appear only in length-increased pairs,
        # so they won't be in still_inconsistent, but mark for completeness)
        expand_tokens = {ep.input_token for ep in expand_patterns}
        
        # ============================================================
        # STEP 3: Resolve remaining inconsistencies with context
        # ============================================================
        context_rules = {}
        final_consistent = {}
        
        for token, obs_list in still_inconsistent.items():
            # Build observations for StructureDiscovery
            ctx_obs = []
            for out, ctx, pair_idx, pos in obs_list:
                if out != token:  # skip identity observations for cleaner signal
                    ctx_obs.append((out, ctx))
                else:
                    ctx_obs.append((out, ctx))
            
            # Check if context can resolve
            outputs = set(o for o, _ in ctx_obs)
            if len(outputs) > 1:
                context_rules[token] = ctx_obs
                diagnostics.setdefault('context_resolved', set()).add(token)
            elif len(outputs) == 1:
                out = next(iter(outputs))
                if out != token:
                    final_consistent[token] = out
        
        consistent_tokens.update(final_consistent)
        
        # ============================================================
        # STEP 4: Detect output-input collisions → freeze flags
        # ============================================================
        multi_outputs = set()
        for mp in multi_patterns:
            for tok in mp.output_tokens:
                multi_outputs.add(tok)
        
        single_inputs = set(consistent_tokens.keys()) | set(context_rules.keys())
        collisions = multi_outputs & single_inputs
        diagnostics['collisions'] = collisions
        
        should_freeze = len(collisions) > 0
        
        # ============================================================
        # STEP 5: Propose phases
        # ============================================================
        phases = []
        
        # Multi-token phases (grouped by arity, longest first)
        if multi_patterns:
            by_arity = defaultdict(list)
            for mp in multi_patterns:
                by_arity[mp.arity].append(mp)
            
            for arity in sorted(by_arity.keys(), reverse=True):
                patterns = by_arity[arity]
                name = {4: 'quadgraph', 3: 'trigraph', 2: 'digraph'}.get(
                    arity, f'group_{arity}')
                
                phases.append(PhaseCandidate(
                    name=f"{name}_collapse",
                    reason=(f"{len(patterns)} multi-token pattern(s) of arity {arity} "
                            f"discovered. Resolves {len(multi_resolved)} single-token "
                            f"inconsistencies." +
                            (f" Output-input collisions detected → freeze."
                             if should_freeze else "")),
                    order_priority=100 - arity * 10,
                    freeze=should_freeze,
                    multi_token_patterns=patterns,
                ))
        
        # Expand phases (1→N expansion, runs after collapse, before context)
        if expand_patterns:
            by_expansion = defaultdict(list)
            for ep in expand_patterns:
                by_expansion[ep.expansion].append(ep)
            
            for exp_size in sorted(by_expansion.keys(), reverse=True):
                patterns = by_expansion[exp_size]
                
                phases.append(PhaseCandidate(
                    name=f"expand_1to{exp_size}",
                    reason=(f"{len(patterns)} expand pattern(s) of width {exp_size} "
                            f"discovered (1→{exp_size} token expansion)."),
                    order_priority=90,
                    expand_patterns=patterns,
                ))
        
        # Context-dependent phase
        if context_rules:
            phases.append(PhaseCandidate(
                name='context_rules',
                reason=(f"{len(context_rules)} token(s) with context-dependent outputs. "
                        f"Resolved by neighbor analysis via StructureDiscovery."),
                order_priority=50,
                context_dependent=True,
                rule_observations=context_rules,
            ))
        
        # Simple consistent phase
        if consistent_tokens:
            phases.append(PhaseCandidate(
                name='token_map',
                reason=f"{len(consistent_tokens)} token(s) with consistent 1→1 mapping.",
                order_priority=10,
                token_rules={t: (o,) for t, o in consistent_tokens.items()},
            ))
        
        # Sort by priority (higher = earlier)
        phases.sort(key=lambda p: -p.order_priority)
        
        return PhaseDiscoveryResult(
            phases=phases,
            training_pairs=self.training_pairs,
            diagnostics=diagnostics,
            geometric=self.geometric,
        )
    
    # ================================================================
    # STEP 1 HELPER: Collect naive 1→1 mappings
    # ================================================================
    
    def _collect_naive_mappings(
        self
    ) -> Dict[Hashable, List[Tuple]]:
        """Collect 1→1 token mappings from EQUAL-LENGTH pairs only.
        
        Key insight: equal-length pairs have clean positional 1→1 mapping.
        Length-reduced pairs have multi-token collapses that corrupt naive
        1→1 alignment (e.g., bath→bæθ would give t→θ instead of th→θ).
        
        Unequal-length pairs are reserved for multi-token pattern verification
        in Step 2, where the length reduction itself is the signal.
        
        Returns:
            {token: [(output_token, context_dict, pair_index, position), ...]}
        """
        token_obs = defaultdict(list)
        
        for pair_idx, (inp_seq, out_seq) in enumerate(self.training_pairs):
            if len(inp_seq) != len(out_seq):
                continue  # skip — these are for multi-token verification
            
            for pos in range(len(inp_seq)):
                if self.geometric:
                    ctx = _extract_geometric_context(inp_seq, pos)
                else:
                    ctx = _extract_context(inp_seq, pos, self.context_window)
                token_obs[inp_seq[pos]].append(
                    (out_seq[pos], ctx, pair_idx, pos)
                )
        
        return token_obs
    
    def _dp_align_1to1(
        self,
        inp: List[Hashable],
        out: List[Hashable]
    ) -> List[Tuple[int, Hashable]]:
        """DP alignment that returns best 1→1 mapping for unequal sequences.
        
        Returns list of (input_position, output_token) pairs.
        Tokens without a match get mapped to themselves (identity).
        """
        n, m = len(inp), len(out)
        
        # Simple LCS-based alignment
        # dp[i][j] = best score for aligning inp[:i] with out[:j]
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if inp[i-1] == out[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 2  # match bonus
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1],
                                   dp[i-1][j-1] + 1)  # substitution
        
        # Backtrack to get alignment
        result = {}
        i, j = n, m
        while i > 0 and j > 0:
            if inp[i-1] == out[j-1]:
                result[i-1] = out[j-1]
                i -= 1
                j -= 1
            elif dp[i-1][j-1] + 1 == dp[i][j]:
                result[i-1] = out[j-1]
                i -= 1
                j -= 1
            elif dp[i-1][j] >= dp[i][j-1]:
                # Skip input token (maps to self)
                result[i-1] = inp[i-1]
                i -= 1
            else:
                j -= 1
        
        # Any remaining input tokens map to themselves
        while i > 0:
            result[i-1] = inp[i-1]
            i -= 1
        
        return [(pos, result[pos]) for pos in sorted(result.keys())]
    
    # ================================================================
    # STEP 2 HELPER: Discover multi-token patterns from length-reduced pairs
    # ================================================================
    
    def _resolve_with_multi_token(
        self,
        inconsistent: Dict[Hashable, List[Tuple]],
        known_consistent: Optional[Dict[Hashable, Hashable]] = None
    ) -> Tuple[List[MultiTokenPattern], Set[Hashable]]:
        """Discover multi-token collapse patterns from length-reduced pairs.
        
        Direct approach: for each length-reduced pair (len_in > len_out),
        try all possible N-gram collapses and verify which ones produce
        clean residual alignments with known consistent mappings.
        
        This works even when the first token of a digraph appears consistent
        in equal-length data (e.g., 's' is always 's' in equal-length pairs,
        but 'sh' collapses to 'ʃ' in length-reduced pairs).
        
        Returns:
            multi_patterns: Verified multi-token patterns
            resolved: Inconsistent tokens explained by these patterns
        """
        # Collect candidate patterns from length-reduced pairs
        # PARSIMONY: shortest match first, non-overlapping per pair
        candidate_counts = defaultdict(lambda: defaultdict(int))
        
        known = known_consistent or {}
        
        for pair_idx, (inp_seq, out_seq) in enumerate(self.training_pairs):
            n_collapse = len(inp_seq) - len(out_seq)
            if n_collapse <= 0:
                continue
            
            # Collect ALL candidate collapses for this pair with quality scores
            candidates = []  # (pos, width, ngram, out_tok, residual_score)
            
            for width in range(2, min(self.max_multi + 1, len(inp_seq) + 1)):
                for pos in range(len(inp_seq) - width + 1):
                    ngram = tuple(inp_seq[pos:pos + width])
                    
                    out_pos = self._estimate_output_position(inp_seq, out_seq, pos)
                    if out_pos is None or out_pos >= len(out_seq):
                        continue
                    
                    out_tok = out_seq[out_pos]
                    
                    if out_tok == ngram[0]:
                        continue
                    
                    # Score residuals: count how many match known_consistent
                    score = self._score_residuals(
                        inp_seq, out_seq, pos, width, out_pos, 1, known
                    )
                    if score < 0:
                        continue  # residual has contradictions
                    
                    candidates.append((pos, width, ngram, out_tok, score))
            
            # Greedy selection: pick best-scoring non-overlapping candidates
            # Prefer: highest score, then shortest width (parsimony)
            candidates.sort(key=lambda c: (-c[4], c[1]))
            covered = set()
            
            for pos, width, ngram, out_tok, score in candidates:
                if any(p in covered for p in range(pos, pos + width)):
                    continue
                for p in range(pos, pos + width):
                    covered.add(p)
                candidate_counts[ngram][out_tok] += 1
        
        # Filter: keep patterns with consistent output across multiple pairs
        # or with strong single-pair evidence + no contradictions
        multi_patterns = []
        seen_keys = set()
        
        for ngram, output_map in candidate_counts.items():
            if ngram in seen_keys:
                continue
            
            # Pick the majority output
            best_out = max(output_map, key=output_map.get)
            best_count = output_map[best_out]
            total = sum(output_map.values())
            
            # Require: majority output has >=70% of observations
            if best_count / total < 0.7:
                continue
            
            # Require: pattern observed in at least 2 training pairs.
            # Real multi-token patterns (sh→ʃ, th→θ) repeat consistently.
            # Spurious patterns from alignment noise appear only once.
            if best_count < 2:
                continue
            
            # NOVEL OUTPUT CHECK: A real collapse produces an output that
            # is NOT the individual mapping of any consumed token.
            # e.g., red+yellow→orange is novel (orange ≠ crimson, ≠ gold)
            # but gray+black→charcoal is spurious (charcoal = black's map)
            is_spurious = False
            for k in range(len(ngram)):
                tok = ngram[k]
                # Check if collapse output matches this token's known 1→1 map
                if tok in known and known[tok] == best_out:
                    is_spurious = True
                    break
                # Also check identity: if tok == best_out, this token just
                # passes through and the "collapse" doesn't really merge
                if tok == best_out:
                    is_spurious = True
                    break
            
            if is_spurious:
                continue
            
            # Cross-verify against equal-length pairs
            contradiction = False
            for inp_seq, out_seq in self.training_pairs:
                if len(out_seq) >= len(inp_seq):
                    for i in range(len(inp_seq) - len(ngram) + 1):
                        if all(inp_seq[i + k] == ngram[k] for k in range(len(ngram))):
                            if i < len(out_seq) and out_seq[i] == best_out:
                                if ngram[0] in known and known[ngram[0]] != best_out:
                                    contradiction = True
            
            if contradiction:
                continue
            
            mp = MultiTokenPattern(
                input_tokens=ngram,
                output_tokens=(best_out,),
                evidence_count=best_count,
                resolves={ngram[0]}
            )
            multi_patterns.append(mp)
            seen_keys.add(ngram)
        
        # Determine which inconsistent tokens are resolved
        resolved = set()
        for mp in multi_patterns:
            for tok in mp.input_tokens:
                if tok in inconsistent:
                    resolved.add(tok)
        
        return multi_patterns, resolved
    
    # ================================================================
    # STEP 2b HELPER: Discover expand patterns from length-increased pairs
    # ================================================================
    
    def _resolve_with_expand(
        self,
        known_consistent: Optional[Dict[Hashable, Hashable]] = None
    ) -> List[ExpandPattern]:
        """Discover 1→N expansion patterns from length-increased pairs.
        
        Mirror of _resolve_with_multi_token: scans pairs where
        len(output) > len(input) and finds single input tokens that
        consistently map to N output tokens.
        
        Returns:
            List of verified ExpandPatterns
        """
        known = known_consistent or {}
        
        # candidate_counts: {input_token: {output_tuple: count}}
        candidate_counts = defaultdict(lambda: defaultdict(int))
        
        for pair_idx, (inp_seq, out_seq) in enumerate(self.training_pairs):
            n_expand = len(out_seq) - len(inp_seq)
            if n_expand <= 0:
                continue
            
            # Try each input position as an expansion source
            candidates = []  # (inp_pos, width, out_tokens, score)
            
            for inp_pos in range(len(inp_seq)):
                tok = inp_seq[inp_pos]
                
                # Skip tokens with known consistent 1→1 mappings —
                # they don't expand
                if tok in known:
                    continue
                
                # Try expansion widths 2, 3, ...
                # Only consider widths consistent with this pair's total expansion.
                # A width-w expansion adds (w-1) tokens. For clean evidence,
                # the pair's n_expand should equal (w-1) — otherwise there are
                # multiple expansions and position estimates become unreliable.
                for width in range(2, min(self.max_multi + 1, len(out_seq) + 1)):
                    if width - 1 != n_expand:
                        continue  # skip — pair has different expansion count
                    # Estimate where this input token starts in the output
                    # For expansion: tokens before inp_pos map 1→1,
                    # so out_start ≈ inp_pos + accumulated_expansions_before
                    # For the simplest case (single expansion), out_start = inp_pos
                    out_start = self._estimate_expand_output_pos(
                        inp_seq, out_seq, inp_pos, known
                    )
                    if out_start is None:
                        continue
                    if out_start + width > len(out_seq):
                        continue
                    
                    out_tokens = tuple(out_seq[out_start:out_start + width])
                    
                    # Score residuals: remove this token from input and
                    # these tokens from output, check if rest aligns 1→1
                    score = self._score_residuals(
                        inp_seq, out_seq,
                        inp_pos, 1,        # 1 input token consumed
                        out_start, width,  # width output tokens produced
                        known
                    )
                    if score < 0:
                        continue  # residual contradictions
                    
                    candidates.append((inp_pos, width, out_tokens, score))
            
            # Greedy: pick best-scoring non-overlapping
            candidates.sort(key=lambda c: (-c[3], c[1]))
            covered_inp = set()
            
            for inp_pos, width, out_tokens, score in candidates:
                if inp_pos in covered_inp:
                    continue
                covered_inp.add(inp_pos)
                candidate_counts[inp_seq[inp_pos]][out_tokens] += 1
        
        # Filter: require evidence ≥2, majority ≥70%
        expand_patterns = []
        
        for tok, output_map in candidate_counts.items():
            best_out = max(output_map, key=output_map.get)
            best_count = output_map[best_out]
            total = sum(output_map.values())
            
            if best_count / total < 0.7:
                continue
            if best_count < 2:
                continue
            
            # NOVEL OUTPUT CHECK (expand version):
            # The expansion output should NOT just be the token's known map
            # repeated or trivially derived
            if len(best_out) == 1 and tok in known and known[tok] == best_out[0]:
                continue  # not really an expansion
            
            # Cross-verify: in equal-length pairs, this token should NOT
            # produce the expansion (it should have a different mapping or
            # not appear)
            contradiction = False
            for inp_seq, out_seq in self.training_pairs:
                if len(inp_seq) == len(out_seq):
                    for i, t in enumerate(inp_seq):
                        if t == tok and i < len(out_seq):
                            # In equal-length pair, token maps to single output
                            # If that single output equals expansion[0], suspicious
                            # but only flag if expansion is width > 1
                            pass  # equal-length pairs shouldn't have this token
                                  # since it only appears in expand pairs
            
            ep = ExpandPattern(
                input_token=tok,
                output_tokens=best_out,
                evidence_count=best_count,
            )
            expand_patterns.append(ep)
        
        return expand_patterns
    
    def _estimate_expand_output_pos(
        self,
        inp_seq: List, out_seq: List,
        inp_pos: int,
        known: Dict[Hashable, Hashable]
    ) -> Optional[int]:
        """Estimate where an input token's expansion starts in output.
        
        Walk through input positions before inp_pos. For each:
        - If token has a known 1→1 mapping, it consumes 1 output position
        - If token is unknown, assume 1 output position (might be wrong
          but residual scoring will catch misalignments)
        """
        out_idx = 0
        for i in range(inp_pos):
            if out_idx >= len(out_seq):
                return None
            tok = inp_seq[i]
            # Check if this might be another expand token
            # For now, assume 1→1 for tokens before our target
            out_idx += 1
        
        return out_idx if out_idx < len(out_seq) else None
    
    def _verify_multi_pattern(
        self,
        pattern: Tuple[Hashable, ...],
        expected_output: Hashable,
        known_consistent: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Tuple[Hashable, ...]]]:
        """Verify that a multi-token pattern is a REAL collapse.
        
        Two-part verification:
        1. LENGTH CHECK: Pattern must appear in pairs where output is shorter
           than input (proves actual token consumption).
        2. RESIDUAL CHECK: After assigning the pattern, all OTHER tokens in
           the pair must have clean 1→1 mappings. If the residual contains
           impossible mappings (e.g., h→θ), the pattern is spurious.
        
        Returns:
            (is_consistent, output_tokens) or (False, None)
        """
        arity = len(pattern)
        occurrences = 0
        matches = 0
        output_tokens = None
        
        for inp_seq, out_seq in self.training_pairs:
            for i in range(len(inp_seq) - arity + 1):
                if not all(inp_seq[i + k] == pattern[k] for k in range(arity)):
                    continue
                
                # Only count length-reduced pairs as evidence
                if len(out_seq) >= len(inp_seq):
                    continue
                
                occurrences += 1
                
                out_pos = self._estimate_output_position(inp_seq, out_seq, i)
                if out_pos is None or out_pos >= len(out_seq):
                    continue
                if out_seq[out_pos] != expected_output:
                    continue
                
                # RESIDUAL CHECK: if we assign pattern at positions i..i+arity-1
                # to output at out_pos, do the remaining tokens align cleanly?
                if known_consistent is not None:
                    residual_ok = self._check_residuals(
                        inp_seq, out_seq, i, arity, out_pos, 1,
                        known_consistent
                    )
                    if not residual_ok:
                        continue
                
                matches += 1
                if output_tokens is None:
                    output_tokens = (expected_output,)
        
        if occurrences == 0:
            return False, None
        
        is_consistent = matches / occurrences >= 0.7
        return is_consistent, output_tokens
    
    def _check_residuals(
        self,
        inp_seq: List, out_seq: List,
        pattern_start: int, pattern_arity: int,
        out_start: int, out_consumed: int,
        known_consistent: Dict[Hashable, Hashable]
    ) -> bool:
        """Check that non-pattern tokens have clean 1→1 mappings.
        
        After removing the pattern's input and output tokens, the remaining
        tokens should align as known consistent mappings or identity.
        
        Returns True if residuals are clean, False if suspicious.
        """
        # Build residual sequences (input and output without pattern)
        pattern_positions = set(range(pattern_start, pattern_start + pattern_arity))
        out_pattern_positions = set(range(out_start, out_start + out_consumed))
        
        res_inp = [inp_seq[i] for i in range(len(inp_seq)) if i not in pattern_positions]
        res_out = [out_seq[j] for j in range(len(out_seq)) if j not in out_pattern_positions]
        
        if len(res_inp) != len(res_out):
            # Residuals don't align 1→1 — might be other collapses
            # Can't verify, give benefit of doubt
            return True
        
        # Check each residual mapping
        suspicious = 0
        for inp_tok, out_tok in zip(res_inp, res_out):
            if inp_tok == out_tok:
                continue  # identity, always fine
            if inp_tok in known_consistent:
                if known_consistent[inp_tok] == out_tok:
                    continue  # matches known consistent mapping
                else:
                    suspicious += 1  # contradicts known mapping!
            # Unknown token — can't verify, allow it
        
        # Reject if more than 1 suspicious residual mapping
        return suspicious <= 1
    
    def _score_residuals(
        self,
        inp_seq: List, out_seq: List,
        pattern_start: int, pattern_arity: int,
        out_start: int, out_consumed: int,
        known_consistent: Dict[Hashable, Hashable]
    ) -> int:
        """Score how well residual tokens match known consistent mappings.
        
        Returns:
            score >= 0: number of residual tokens matching known mappings
            -1: contradiction found (residual contradicts known mapping)
        """
        pattern_positions = set(range(pattern_start, pattern_start + pattern_arity))
        out_pattern_positions = set(range(out_start, out_start + out_consumed))
        
        res_inp = [inp_seq[i] for i in range(len(inp_seq)) if i not in pattern_positions]
        res_out = [out_seq[j] for j in range(len(out_seq)) if j not in out_pattern_positions]
        
        if len(res_inp) != len(res_out):
            return 0  # can't score, other collapses present
        
        score = 0
        for inp_tok, out_tok in zip(res_inp, res_out):
            if inp_tok == out_tok:
                score += 1  # identity match
            elif inp_tok in known_consistent:
                if known_consistent[inp_tok] == out_tok:
                    score += 2  # matches known consistent — strong signal
                else:
                    return -1  # contradiction!
            # Unknown: neither helps nor hurts
        
        return score
    
    def _estimate_output_position(
        self,
        inp_seq: List, out_seq: List, inp_pos: int
    ) -> Optional[int]:
        """Estimate where input position maps to in output.
        
        Uses a simple forward scan matching tokens.
        """
        # Count matching tokens from start to estimate offset
        out_idx = 0
        for i in range(inp_pos):
            if out_idx >= len(out_seq):
                return None
            # If tokens match, advance both
            if i < len(inp_seq) and out_idx < len(out_seq):
                if inp_seq[i] == out_seq[out_idx]:
                    out_idx += 1
                else:
                    # Mismatch — could be a substitution or collapse
                    # Advance output by 1 (substitution assumption)
                    out_idx += 1
        
        return out_idx if out_idx < len(out_seq) else None
    
    def _explained_by_multi(
        self,
        token: Hashable,
        pair_idx: int,
        pos: int,
        multi_patterns: List[MultiTokenPattern]
    ) -> bool:
        """Check if a specific observation is explained by a multi-token pattern."""
        inp_seq = self.training_pairs[pair_idx][0]
        
        for mp in multi_patterns:
            if mp.input_tokens[0] != token:
                continue
            
            # Check if the pattern matches at this position
            if pos + mp.arity <= len(inp_seq):
                if all(inp_seq[pos + k] == mp.input_tokens[k]
                       for k in range(mp.arity)):
                    return True
        
        return False
