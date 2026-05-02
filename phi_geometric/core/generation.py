"""
Generation: Reverse transformation, constrained generation, and lattice navigation.

Three layers of generation capability built on the proven transformation engine:

Layer 1 — Reverse Transformation:
    Given a desired output, find inputs that produce it.
    "What English spelling gives pronunciation [ʃɪp]?" → "ship"

Layer 2 — Constrained Generation:
    Given a partial sequence with wildcards, find completions
    consistent with discovered rules.

Layer 3 — Lattice Navigation (Ribbon Math pattern):
    Starting from known valid pairs, generate NEW valid pairs
    by perturbation + verification. Discovers sequences the
    training data never contained.

Usage:
    from phi_geometric import PhaseDiscovery
    from phi_geometric.core.generation import ReverseEngine

    pd = PhaseDiscovery()
    pd.add_pair(list('ship'), list('ʃɪp'))
    pd.add_pair(list('cat'),  list('kæt'))
    result = pd.discover()
    nav = result.to_navigator()

    engine = ReverseEngine(nav)

    # Layer 1: Reverse
    inputs = engine.reverse(list('ʃɪp'))
    # → [['s', 'h', 'ɪ', 'p']]  (found valid input)

    # Layer 2: Constrained generation
    completions = engine.complete(['?', '?', 'a', 't'], target_output=list('kæt'))
    # → [['c', 'a', 't']]

    # Layer 3: Lattice navigation
    novel = engine.navigate(seed_pairs=result.training_pairs, steps=100)
    # → new (input, output) pairs never in training

Author: TruthSpace LCM Project
Date: February 2026
"""

from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Set, Tuple
from itertools import product

from .cascade_navigator import CascadeNavigator
from .discovery import TransformRule


# =========================================================================
# LAYER 1: REVERSE TRANSFORMATION
# =========================================================================

class ReverseEngine:
    """Generate inputs from outputs by reversing a CascadeNavigator.

    The navigator encodes a forward transformation: input → output.
    This engine inverts it: given a desired output, find valid inputs.
    """

    def __init__(self, nav: CascadeNavigator):
        self.nav = nav
        self._build_reverse_tables()

    def _build_reverse_tables(self):
        """Extract reverse mappings from the navigator's rules."""
        # Reverse map: output_token → set of possible input_tokens
        self.reverse_map: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        # Context-dependent reverse: output_token → [(input, variable, ctx_val)]
        self.reverse_context: Dict[Hashable, List[Tuple]] = defaultdict(list)
        # Reverse collapses: output_tuple → input_tuple
        self.reverse_collapse: Dict[Tuple, Tuple] = {}
        # Reverse expands: output_tuple → input_token
        self.reverse_expand: Dict[Tuple, Hashable] = {}
        # Identity tokens (pass through unchanged)
        self.identity_tokens: Set[Hashable] = set()
        # All known input tokens
        self.known_inputs: Set[Hashable] = set()
        # All known output tokens
        self.known_outputs: Set[Hashable] = set()

        # Reverse collapse patterns: output → input
        for inp_toks, out_toks, freeze in self.nav.collapse_patterns:
            self.reverse_collapse[out_toks] = inp_toks
            for t in inp_toks:
                self.known_inputs.add(t)
            for t in out_toks:
                self.known_outputs.add(t)

        # Reverse expand patterns: output → input
        for inp_tok, out_toks in self.nav.expand_patterns:
            self.reverse_expand[out_toks] = inp_tok
            self.known_inputs.add(inp_tok)
            for t in out_toks:
                self.known_outputs.add(t)

        # Reverse phase rules
        for phase in self.nav.phases:
            for rule in phase.rules:
                self.known_inputs.add(rule.input_value)
                if rule.rule_type == 'identity':
                    self.identity_tokens.add(rule.input_value)
                elif rule.rule_type == 'consistent':
                    out = rule.params['output']
                    self.reverse_map[out].add(rule.input_value)
                    self.known_outputs.add(out)
                elif rule.rule_type == 'selector':
                    sel_map = rule.params.get('selector_map', {})
                    for ctx_val, out in sel_map.items():
                        self.reverse_map[out].add(rule.input_value)
                        self.reverse_context[out].append(
                            (rule.input_value, rule.params['variable'], ctx_val)
                        )
                        self.known_outputs.add(out)
                    default = rule.params.get('default_output')
                    if default:
                        self.reverse_map[default].add(rule.input_value)
                        self.known_outputs.add(default)
                elif rule.rule_type == 'geared':
                    pure_map = rule.params.get('pure_map', {})
                    for ctx_val, out in pure_map.items():
                        self.reverse_map[out].add(rule.input_value)
                        self.reverse_context[out].append(
                            (rule.input_value, rule.params['coarse_var'], ctx_val)
                        )
                        self.known_outputs.add(out)
                    fine_gears = rule.params.get('fine_gears', {})
                    for coarse_val, fg in fine_gears.items():
                        fine_var, fine_map, _, _, zone_default = fg
                        if fine_map:
                            for fv, out in fine_map.items():
                                self.reverse_map[out].add(rule.input_value)
                                self.known_outputs.add(out)
                        if zone_default:
                            self.reverse_map[zone_default].add(rule.input_value)
                            self.known_outputs.add(zone_default)
                    default = rule.params.get('default_output')
                    if default:
                        self.reverse_map[default].add(rule.input_value)
                        self.known_outputs.add(default)

    # -----------------------------------------------------------------
    # Layer 1: Reverse Transformation
    # -----------------------------------------------------------------

    def reverse(
        self,
        target_output: List[Hashable],
        max_results: int = 10,
        verify: bool = True,
    ) -> List[List[Hashable]]:
        """Find input sequences that produce the target output.

        Works by:
        1. Reverse collapses/expands to get pre-phase tokens
        2. Reverse phase rules to get candidate input tokens
        3. Combine candidates and verify via forward execution

        Args:
            target_output: Desired output sequence
            max_results: Maximum number of valid inputs to return
            verify: If True, verify each candidate via forward execution

        Returns:
            List of valid input sequences that produce target_output
        """
        # Step 1: Reverse collapses — expand output tokens back
        pre_phase = self._reverse_pre_processing(target_output)

        # Step 2: For each position, find candidate input tokens
        candidates_per_pos = []
        for tok in pre_phase:
            candidates = set()
            # Identity: token passes through unchanged
            if tok in self.identity_tokens or tok not in self.known_outputs:
                candidates.add(tok)
            # Reverse map: which input tokens produce this output?
            if tok in self.reverse_map:
                candidates.update(self.reverse_map[tok])
            # If nothing found, assume identity
            if not candidates:
                candidates.add(tok)
            candidates_per_pos.append(sorted(candidates))

        # Step 3: Generate combinations (with pruning)
        results = []
        total_combos = 1
        for c in candidates_per_pos:
            total_combos *= len(c)

        if total_combos <= 10000:
            # Small enough to enumerate
            for combo in product(*candidates_per_pos):
                inp = list(combo)
                if verify:
                    trace = self.nav.execute(inp)
                    if trace.output_elements == target_output:
                        results.append(inp)
                        if len(results) >= max_results:
                            break
                else:
                    results.append(inp)
                    if len(results) >= max_results:
                        break
        else:
            # Too many combos — use greedy search
            results = self._greedy_reverse(
                pre_phase, candidates_per_pos, target_output, max_results
            )

        return results

    def _reverse_pre_processing(
        self, output_tokens: List[Hashable]
    ) -> List[Hashable]:
        """Reverse collapse and expand patterns to get pre-phase tokens.

        Collapse (forward): [s, h] → [ʃ]  →  Reverse: [ʃ] → [s, h]
        Expand (forward): [x] → [k, s]    →  Reverse: [k, s] → [x]
        """
        tokens = list(output_tokens)

        # Reverse expands first (they were applied after collapses)
        if self.reverse_expand:
            # Sort by output length descending (longest match first)
            sorted_rev = sorted(self.reverse_expand.items(),
                                key=lambda x: len(x[0]), reverse=True)
            new_tokens = []
            i = 0
            while i < len(tokens):
                matched = False
                for out_toks, inp_tok in sorted_rev:
                    pat_len = len(out_toks)
                    if i + pat_len <= len(tokens):
                        if all(tokens[i + k] == out_toks[k]
                               for k in range(pat_len)):
                            new_tokens.append(inp_tok)
                            i += pat_len
                            matched = True
                            break
                if not matched:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Reverse collapses (they were applied before expands)
        if self.reverse_collapse:
            sorted_rev = sorted(self.reverse_collapse.items(),
                                key=lambda x: len(x[0]), reverse=True)
            new_tokens = []
            i = 0
            while i < len(tokens):
                matched = False
                for out_toks, inp_toks in sorted_rev:
                    pat_len = len(out_toks)
                    if i + pat_len <= len(tokens):
                        if all(tokens[i + k] == out_toks[k]
                               for k in range(pat_len)):
                            new_tokens.extend(inp_toks)
                            i += pat_len
                            matched = True
                            break
                if not matched:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def _greedy_reverse(
        self,
        pre_phase: List[Hashable],
        candidates_per_pos: List[List[Hashable]],
        target_output: List[Hashable],
        max_results: int,
    ) -> List[List[Hashable]]:
        """Greedy reverse search for large candidate spaces."""
        results = []

        # Try each candidate at each position, keeping best
        best = list(pre_phase)  # start with identity guess
        for pos in range(len(best)):
            for candidate in candidates_per_pos[pos]:
                test = list(best)
                test[pos] = candidate
                trace = self.nav.execute(test)
                if trace.output_elements == target_output:
                    best = test
                    break

        trace = self.nav.execute(best)
        if trace.output_elements == target_output:
            results.append(best)

        return results

    # -----------------------------------------------------------------
    # Layer 2: Constrained Generation (Completion)
    # -----------------------------------------------------------------

    WILDCARD = '?'

    def complete(
        self,
        partial_input: List[Hashable],
        target_output: Optional[List[Hashable]] = None,
        max_results: int = 10,
    ) -> List[List[Hashable]]:
        """Fill in wildcards ('?') in a partial input sequence.

        If target_output is provided, only return completions that
        produce that exact output. Otherwise, return any completion
        where all wildcards are replaced with known input tokens.

        Args:
            partial_input: Sequence with '?' wildcards
            target_output: Optional desired output to match
            max_results: Maximum results to return

        Returns:
            List of completed input sequences
        """
        wildcard_positions = [
            i for i, t in enumerate(partial_input) if t == self.WILDCARD
        ]

        if not wildcard_positions:
            return [list(partial_input)]

        # Determine candidate tokens for each wildcard position
        candidates_per_wild = []
        for pos in wildcard_positions:
            if target_output is not None and pos < len(target_output):
                # Use reverse mapping from target output at this position
                out_tok = target_output[pos]
                candidates = set()
                if out_tok in self.reverse_map:
                    candidates.update(self.reverse_map[out_tok])
                if out_tok in self.identity_tokens or out_tok not in self.known_outputs:
                    candidates.add(out_tok)
                if not candidates:
                    candidates = set(self.known_inputs)
            else:
                candidates = set(self.known_inputs)
            candidates_per_wild.append(sorted(candidates))

        # Enumerate and verify
        results = []
        total = 1
        for c in candidates_per_wild:
            total *= len(c)

        combos = product(*candidates_per_wild) if total <= 50000 else []

        for combo in combos:
            inp = list(partial_input)
            for i, pos in enumerate(wildcard_positions):
                inp[pos] = combo[i]

            if target_output is not None:
                trace = self.nav.execute(inp)
                if trace.output_elements == target_output:
                    results.append(inp)
            else:
                results.append(inp)

            if len(results) >= max_results:
                break

        return results

    # -----------------------------------------------------------------
    # Layer 3: Lattice Navigation (Ribbon Math Pattern)
    # -----------------------------------------------------------------

    def navigate(
        self,
        seed_pairs: List[Tuple[List[Hashable], List[Hashable]]],
        steps: int = 100,
        max_novel: int = 50,
    ) -> List[Tuple[List[Hashable], List[Hashable]]]:
        """Generate novel valid (input, output) pairs by lattice navigation.

        The Ribbon Math pattern:
            1. Start from seed pairs (known valid)
            2. Perturb: substitute tokens within their rule domains
            3. Verify: run forward and check consistency
            4. Accept novel valid pairs into the lattice
            5. Repeat from expanded lattice

        Args:
            seed_pairs: Known valid (input, output) pairs
            steps: Maximum perturbation attempts
            max_novel: Maximum novel pairs to generate

        Returns:
            List of novel (input, output) pairs not in seeds
        """
        import random

        # Build token classes: tokens that share the same rule behavior
        token_classes = self._build_token_classes()

        # Track known pairs (as frozen sets for dedup)
        known = set()
        for inp, out in seed_pairs:
            known.add((tuple(inp), tuple(out)))

        novel_pairs = []
        attempts = 0

        while attempts < steps and len(novel_pairs) < max_novel:
            attempts += 1

            # Pick a random seed
            seed_inp, seed_out = random.choice(seed_pairs)

            # Pick a random position to perturb
            if not seed_inp:
                continue
            pos = random.randint(0, len(seed_inp) - 1)
            original_token = seed_inp[pos]

            # Find tokens in the same class
            token_class = token_classes.get(original_token, {original_token})
            substitutes = token_class - {original_token}
            if not substitutes:
                continue

            # Perturb
            new_inp = list(seed_inp)
            new_inp[pos] = random.choice(sorted(substitutes))

            # Verify
            trace = self.nav.execute(new_inp)
            new_out = trace.output_elements

            # Check novelty
            key = (tuple(new_inp), tuple(new_out))
            if key not in known:
                known.add(key)
                novel_pairs.append((new_inp, new_out))

                # Expand the seed pool (the lattice grows)
                seed_pairs = list(seed_pairs) + [(new_inp, new_out)]

        return novel_pairs

    def _build_token_classes(self) -> Dict[Hashable, Set[Hashable]]:
        """Group tokens that share similar transformation behavior.

        Tokens in the same class can be substituted for each other
        while preserving the structural archetype. Classes:

        - All identity tokens (pass-through)
        - All consistent-map tokens (1→1 substitution, same role)
        - All context-dependent tokens (selector/geared)
        - All tokens appearing in collapse patterns

        This is broader than "same output" — consonants that each map
        to different outputs still share the "consistent map" role.
        """
        classes: Dict[Hashable, Set[Hashable]] = {}

        identity_group: Set[Hashable] = set()
        consistent_group: Set[Hashable] = set()
        context_group: Set[Hashable] = set()

        for phase in self.nav.phases:
            for rule in phase.rules:
                if rule.rule_type == 'identity':
                    identity_group.add(rule.input_value)
                elif rule.rule_type == 'consistent':
                    consistent_group.add(rule.input_value)
                elif rule.rule_type in ('selector', 'geared'):
                    context_group.add(rule.input_value)

        # Identity tokens can substitute for each other
        for tok in identity_group:
            classes[tok] = set(identity_group)

        # All consistent-map tokens form a class (same structural role)
        for tok in consistent_group:
            classes[tok] = set(consistent_group)

        # Context-dependent tokens can substitute for each other
        for tok in context_group:
            classes[tok] = set(context_group)

        # Collapse input tokens form a class
        collapse_inputs: Set[Hashable] = set()
        for inp_toks, _, _ in self.nav.collapse_patterns:
            collapse_inputs.update(inp_toks)
        for tok in collapse_inputs:
            classes.setdefault(tok, set()).update(collapse_inputs)

        return classes

    # -----------------------------------------------------------------
    # Convenience: describe the reverse engine
    # -----------------------------------------------------------------

    def describe(self) -> str:
        """Human-readable summary of reverse capabilities."""
        lines = ["ReverseEngine:"]
        lines.append(f"  Known input tokens:  {len(self.known_inputs)}")
        lines.append(f"  Known output tokens: {len(self.known_outputs)}")
        lines.append(f"  Identity tokens:     {len(self.identity_tokens)}")
        lines.append(f"  Reverse map entries: {sum(len(v) for v in self.reverse_map.values())}")
        lines.append(f"  Reverse collapses:   {len(self.reverse_collapse)}")
        lines.append(f"  Reverse expands:     {len(self.reverse_expand)}")

        if self.reverse_map:
            lines.append("  Sample reverse mappings:")
            for out, inputs in sorted(self.reverse_map.items(), key=str)[:8]:
                inp_str = ', '.join(str(i) for i in sorted(inputs, key=str))
                lines.append(f"    {out!r} ← {{{inp_str}}}")

        return '\n'.join(lines)
