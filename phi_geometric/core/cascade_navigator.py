"""
Cascade Navigator: Execute ordered phase pipelines with flow control.

The discrete counterpart to Navigator (which handles continuous tensor ops).
CascadeNavigator executes Cascade topology patterns — ordered phases where
each phase selects and applies one TransformRule per input element.

Key flow-control primitives (ported from IPA geometric demo):

    1. FROZEN OUTPUTS: When a phase produces a frozen output, downstream
       phases skip that element. This prevents double-processing
       (e.g., digraph 'ee'→'iː' shouldn't be re-vowel-mapped).

    2. ORIGINAL CONTEXT: Context variables are always evaluated against
       the ORIGINAL input, not intermediate transformations. This ensures
       that phase ordering doesn't corrupt context signals.

    3. TRACE: Every transformation is recorded, producing a full trace
       of which rule fired at each phase for each element. This is the
       data that drives the tumbler disk visualization.

Author: TruthSpace LCM Project
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple
from collections import defaultdict

from .discovery import TransformRule, Observation


# ============================================================================
# PHASE DEFINITION
# ============================================================================

@dataclass
class Phase:
    """A single phase in a Cascade pipeline.
    
    Attributes:
        name: Human-readable phase name
        rules: List of TransformRules for this phase
        context_extractor: Function (elements, index, original_elements) → context dict.
                          If None, uses default positional context.
        freeze_outputs: If True, outputs from this phase are frozen
                       (downstream phases skip them)
        use_original_context: If True, context is extracted from original elements
                             rather than current (intermediate) elements
    """
    name: str
    rules: List[TransformRule] = field(default_factory=list)
    context_extractor: Optional[Callable] = None
    freeze_outputs: bool = False
    use_original_context: bool = True
    
    def add_rule(self, rule: TransformRule):
        """Add a rule to this phase."""
        self.rules.append(rule)
    
    @property
    def rule_index(self) -> Dict[Hashable, TransformRule]:
        """Index rules by input value for O(1) lookup."""
        return {r.input_value: r for r in self.rules}


# ============================================================================
# TRACE RECORDS
# ============================================================================

@dataclass
class PhaseTrace:
    """Record of what happened at one phase for one element.
    
    Attributes:
        phase_index: Which phase (0-based)
        phase_name: Phase name
        element_index: Which element in the sequence
        input_value: Value entering this phase
        output_value: Value leaving this phase
        rule_fired: The TransformRule that fired (None if no match)
        was_frozen: Whether this element was frozen (skipped)
        context: The context dict used for rule selection
    """
    phase_index: int
    phase_name: str
    element_index: int
    input_value: Hashable
    output_value: Hashable
    rule_fired: Optional[TransformRule] = None
    was_frozen: bool = False
    context: Optional[Dict] = None
    
    @property
    def changed(self) -> bool:
        return self.input_value != self.output_value


@dataclass
class ElementTrace:
    """Full trace for one element through all phases.
    
    Attributes:
        original: The original input value
        final: The final output value
        phases: List of PhaseTrace records
        frozen_at: Phase index where this element was frozen (-1 if never)
    """
    original: Hashable
    final: Hashable
    phases: List[PhaseTrace] = field(default_factory=list)
    frozen_at: int = -1
    
    @property
    def is_frozen(self) -> bool:
        return self.frozen_at >= 0
    
    @property
    def transformations(self) -> List[PhaseTrace]:
        """Only the phases where a change actually occurred."""
        return [p for p in self.phases if p.changed]


@dataclass
class CascadeTrace:
    """Full trace of a cascade execution.
    
    Attributes:
        input_elements: Original input sequence
        output_elements: Final output sequence
        elements: Per-element traces
        n_phases: Number of phases executed
    """
    input_elements: List[Hashable]
    output_elements: List[Hashable]
    elements: List[ElementTrace] = field(default_factory=list)
    n_phases: int = 0
    
    @property
    def n_transformations(self) -> int:
        """Total number of actual changes made."""
        return sum(len(e.transformations) for e in self.elements)
    
    @property
    def n_frozen(self) -> int:
        """Number of elements that got frozen."""
        return sum(1 for e in self.elements if e.is_frozen)
    
    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Cascade: {len(self.input_elements)} elements → "
            f"{len(self.output_elements)} outputs, "
            f"{self.n_phases} phases, "
            f"{self.n_transformations} transformations, "
            f"{self.n_frozen} frozen"
        ]
        for et in self.elements:
            arrow = f"{repr(et.original)} → {repr(et.final)}"
            if et.is_frozen:
                arrow += f" (frozen at phase {et.frozen_at})"
            changes = [f"p{p.phase_index}:{p.rule_fired.input_value if p.rule_fired else '—'}"
                      for p in et.phases if p.changed]
            if changes:
                arrow += f"  [{', '.join(changes)}]"
            lines.append(f"  {arrow}")
        return '\n'.join(lines)


# ============================================================================
# DEFAULT CONTEXT EXTRACTOR
# ============================================================================

def default_context_extractor(
    elements: List[Hashable],
    index: int,
    original_elements: List[Hashable]
) -> Dict[str, Hashable]:
    """Default context: positional + neighbor information.
    
    This mirrors the context used in the IPA demo but is generalized
    to work with any hashable element type.
    """
    n = len(elements)
    return {
        'prev': elements[index - 1] if index > 0 else None,
        'next': elements[index + 1] if index + 1 < n else None,
        'prev_prev': elements[index - 2] if index > 1 else None,
        'next_next': elements[index + 2] if index + 2 < n else None,
        'position': index,
        'length': n,
        'is_start': index == 0,
        'is_end': index == n - 1,
        'original': original_elements[index],
    }


# φ-level bins for geometric context
_PHI_LEVEL_RANGES = [
    (1, 1),    # level 0: distance 1
    (2, 3),    # level 1: distance 2-3
    (4, 7),    # level 2: distance 4-7
    (8, 12),   # level 3: distance 8-12
]


def geometric_context_extractor(
    elements: List[Hashable],
    index: int,
    original_elements: List[Hashable]
) -> Dict[str, Hashable]:
    """Geometric context using φ-level binning.
    
    Covers distance 1-12 with 4 features per direction (8 geometric
    features total), mirroring how attention decays with φ^(-level).
    
    Also includes raw prev/next for backward compatibility with
    rules that were discovered using those feature names.
    """
    n = len(elements)
    ctx = {
        'prev': elements[index - 1] if index > 0 else None,
        'next': elements[index + 1] if index + 1 < n else None,
        'position': index,
        'length': n,
        'is_start': index == 0,
        'is_end': index == n - 1,
        'original': original_elements[index],
    }
    
    for level, (lo, hi) in enumerate(_PHI_LEVEL_RANGES):
        prev_near = None
        prev_far = None
        for d in range(lo, hi + 1):
            idx = index - d
            if 0 <= idx < n:
                if prev_near is None:
                    prev_near = elements[idx]
                prev_far = elements[idx]
        ctx[f'phi_prev_{level}'] = prev_near
        if hi > lo:
            ctx[f'phi_prev_{level}_far'] = prev_far
        
        next_near = None
        next_far = None
        for d in range(lo, hi + 1):
            idx = index + d
            if 0 <= idx < n:
                if next_near is None:
                    next_near = elements[idx]
                next_far = elements[idx]
        ctx[f'phi_next_{level}'] = next_near
        if hi > lo:
            ctx[f'phi_next_{level}_far'] = next_far
    
    return ctx


# ============================================================================
# CASCADE NAVIGATOR
# ============================================================================

class CascadeNavigator:
    """Execute a Cascade topology: ordered phases with flow control.
    
    This is the discrete-domain navigator for sequential transformation
    pipelines. It implements three flow-control primitives:
    
    1. **Collapse patterns**: Multi-token patterns (e.g., sh→ʃ) that
       consume N input tokens and produce M output tokens. Applied as
       a pre-processing pass before element-by-element phases.
    
    2. **Frozen outputs**: Elements can be marked as frozen by a phase,
       preventing downstream phases from re-processing them.
    
    3. **Original context**: Context extraction uses the original input
       elements, not intermediate transformations.
    
    Example:
        nav = CascadeNavigator()
        
        # Add collapse patterns (digraphs)
        nav.add_collapse(('s', 'h'), ('ʃ',), freeze=True)
        nav.add_collapse(('t', 'h'), ('θ',), freeze=True)
        
        # Add element-by-element phases
        p0 = Phase('vowel_map')
        p0.add_rule(TransformRule('i', 'consistent', output='ɪ'))
        nav.add_phase(p0)
        
        # Execute
        trace = nav.execute(['s', 'h', 'i', 'p'])
        print(trace.output_elements)  # ['ʃ', 'ɪ', 'p']
    """
    
    def __init__(self):
        self.phases: List[Phase] = []
        self.collapse_patterns: List[Tuple[Tuple, Tuple, bool]] = []  # (input, output, freeze)
        self.expand_patterns: List[Tuple[Hashable, Tuple]] = []  # (input_token, output_tokens)
    
    def add_phase(self, phase: Phase):
        """Add a phase to the pipeline."""
        self.phases.append(phase)
    
    def add_collapse(self, input_tokens: Tuple, output_tokens: Tuple,
                     freeze: bool = False):
        """Add a multi-token collapse pattern.
        
        Collapses are applied longest-match-first before element-by-element
        phases. If freeze=True, the output tokens are frozen (skipped by
        downstream phases).
        
        Args:
            input_tokens: Tuple of input tokens to match (e.g., ('s', 'h'))
            output_tokens: Tuple of replacement tokens (e.g., ('ʃ',))
            freeze: Whether output tokens should be frozen
        """
        self.collapse_patterns.append((tuple(input_tokens), tuple(output_tokens), freeze))
    
    def add_expand(self, input_token: Hashable, output_tokens: Tuple):
        """Add a 1→N expansion pattern.
        
        Expansions are applied after collapses but before element-by-element
        phases. A single input token is replaced by multiple output tokens.
        
        Args:
            input_token: Single token to match (e.g., 'x')
            output_tokens: Tuple of replacement tokens (e.g., ('k', 's'))
        """
        self.expand_patterns.append((input_token, tuple(output_tokens)))
    
    def execute(
        self,
        elements: List[Hashable],
        trace: bool = True
    ) -> CascadeTrace:
        """Execute the cascade pipeline on a sequence of elements.
        
        Args:
            elements: Input sequence
            trace: Whether to record full execution trace
        
        Returns:
            CascadeTrace with input, output, and per-element traces
        """
        # Pre-processing: apply collapse patterns (longest match first)
        collapsed, collapse_frozen, orig_map = self._apply_collapses(elements)
        
        # Pre-processing: apply expand patterns (1→N)
        if self.expand_patterns:
            collapsed, collapse_frozen, orig_map = self._apply_expands(
                collapsed, collapse_frozen, orig_map
            )
        
        n = len(collapsed)
        current = list(collapsed)           # mutable working copy
        original = list(elements)           # immutable original (pre-collapse)
        frozen = list(collapse_frozen)      # frozen flags from collapses
        
        # Initialize per-element traces
        element_traces = [
            ElementTrace(original=collapsed[i], final=collapsed[i])
            for i in range(n)
        ]
        
        for pi, phase in enumerate(self.phases):
            rule_index = phase.rule_index
            extractor = phase.context_extractor or default_context_extractor
            
            for ei in range(n):
                # Skip frozen elements
                if frozen[ei]:
                    if trace:
                        element_traces[ei].phases.append(PhaseTrace(
                            phase_index=pi, phase_name=phase.name,
                            element_index=ei,
                            input_value=current[ei], output_value=current[ei],
                            was_frozen=True
                        ))
                    continue
                
                # Extract context from original or current elements
                if phase.use_original_context:
                    # Map collapsed index back to original position
                    ctx_pos = orig_map[ei]
                    ctx = extractor(original, ctx_pos, original)
                else:
                    ctx = extractor(current, ei, original)
                
                # Look up rule for this element's current value
                rule = rule_index.get(current[ei])
                
                if rule is not None:
                    old_val = current[ei]
                    new_val = rule.apply(current[ei], ctx)
                    current[ei] = new_val
                    
                    # Freeze if phase says so and a change occurred
                    if phase.freeze_outputs and new_val != old_val:
                        frozen[ei] = True
                        element_traces[ei].frozen_at = pi
                    
                    if trace:
                        element_traces[ei].phases.append(PhaseTrace(
                            phase_index=pi, phase_name=phase.name,
                            element_index=ei,
                            input_value=old_val, output_value=new_val,
                            rule_fired=rule if new_val != old_val else None,
                            context=ctx
                        ))
                else:
                    # No rule matched — pass through
                    if trace:
                        element_traces[ei].phases.append(PhaseTrace(
                            phase_index=pi, phase_name=phase.name,
                            element_index=ei,
                            input_value=current[ei], output_value=current[ei]
                        ))
        
        # Update final values
        for i in range(n):
            element_traces[i].final = current[i]
        
        return CascadeTrace(
            input_elements=list(elements),
            output_elements=current,
            elements=element_traces,
            n_phases=len(self.phases)
        )
    
    def _apply_collapses(
        self,
        elements: List[Hashable]
    ) -> Tuple[List[Hashable], List[bool], List[int]]:
        """Apply collapse patterns to the input sequence.
        
        Scans left-to-right, longest match first at each position.
        
        Returns:
            collapsed: New sequence after collapses
            frozen: Frozen flags for each element in collapsed
            orig_map: Maps each collapsed index → original input index
        """
        if not self.collapse_patterns:
            return (list(elements), [False] * len(elements),
                    list(range(len(elements))))
        
        # Sort patterns by input length descending (longest match first)
        sorted_patterns = sorted(self.collapse_patterns,
                                key=lambda p: len(p[0]), reverse=True)
        
        collapsed = []
        frozen_flags = []
        orig_map = []
        i = 0
        
        while i < len(elements):
            matched = False
            for inp_pat, out_pat, freeze in sorted_patterns:
                pat_len = len(inp_pat)
                if i + pat_len <= len(elements):
                    if all(elements[i + k] == inp_pat[k] for k in range(pat_len)):
                        # Match! Replace with output tokens
                        for tok in out_pat:
                            collapsed.append(tok)
                            frozen_flags.append(freeze)
                            orig_map.append(i)
                        i += pat_len
                        matched = True
                        break
            
            if not matched:
                collapsed.append(elements[i])
                frozen_flags.append(False)
                orig_map.append(i)
                i += 1
        
        return collapsed, frozen_flags, orig_map
    
    def _apply_expands(
        self,
        elements: List[Hashable],
        frozen: List[bool],
        orig_map: List[int]
    ) -> Tuple[List[Hashable], List[bool], List[int]]:
        """Apply expand patterns to the sequence.
        
        Scans left-to-right, replacing single tokens with their expansions.
        
        Returns:
            expanded: New sequence after expansions
            frozen: Updated frozen flags
            orig_map: Updated original index mapping
        """
        if not self.expand_patterns:
            return elements, frozen, orig_map
        
        # Build lookup: token → output_tokens (longest output first)
        expand_lookup = {}
        for inp_tok, out_toks in self.expand_patterns:
            expand_lookup[inp_tok] = out_toks
        
        expanded = []
        new_frozen = []
        new_orig_map = []
        
        for i, tok in enumerate(elements):
            if not frozen[i] and tok in expand_lookup:
                out_toks = expand_lookup[tok]
                for t in out_toks:
                    expanded.append(t)
                    new_frozen.append(False)
                    new_orig_map.append(orig_map[i])
            else:
                expanded.append(tok)
                new_frozen.append(frozen[i])
                new_orig_map.append(orig_map[i])
        
        return expanded, new_frozen, new_orig_map
    
    def describe(self) -> str:
        """Human-readable description of the pipeline."""
        lines = []
        n_pre = len(self.collapse_patterns) + len(self.expand_patterns)
        if n_pre > 0:
            parts = []
            if self.collapse_patterns:
                parts.append(f"{len(self.collapse_patterns)} collapse(s)")
            if self.expand_patterns:
                parts.append(f"{len(self.expand_patterns)} expand(s)")
            lines.append(f"CascadeNavigator: {', '.join(parts)}, "
                        f"{len(self.phases)} phases")
            for inp, out, freeze in self.collapse_patterns:
                inp_s = ''.join(str(t) for t in inp)
                out_s = ''.join(str(t) for t in out)
                f = ' [freeze]' if freeze else ''
                lines.append(f"  Collapse: {inp_s}→{out_s}{f}")
            for inp_tok, out_toks in self.expand_patterns:
                out_s = ''.join(str(t) for t in out_toks)
                lines.append(f"  Expand: {inp_tok}→{out_s}")
        else:
            lines.append(f"CascadeNavigator: {len(self.phases)} phases")
        
        for i, phase in enumerate(self.phases):
            flags = []
            if phase.freeze_outputs:
                flags.append("freeze")
            if phase.use_original_context:
                flags.append("orig_ctx")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  Phase {i}: {phase.name} "
                        f"({len(phase.rules)} rules){flag_str}")
        return '\n'.join(lines)
