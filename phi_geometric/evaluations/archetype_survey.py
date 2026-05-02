#!/usr/bin/env python3
"""
Transformation Archetype Survey
================================

Tests PhaseDiscovery on four toy domains, each representing a different
transformation archetype. The question: does PhaseDiscovery correctly
identify the STRUCTURE of the transformation, even when the domains
are completely different?

Archetypes:
  A. MAP-ONLY          [map]                 — Pure substitution cipher
  B. CONTEXT→MAP       [context, map]        — Neighbor-influenced encoding
  C. COLLAPSE→MAP      [collapse, map]       — Token merging + substitution
  D. COLLAPSE→CTX→MAP  [collapse, ctx, map]  — Full cascade (IPA/Pixel)
"""

import sys
import os
import importlib.util
import types
from collections import defaultdict

# ============================================================================
# Load PhaseDiscovery without torch dependency
# ============================================================================
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_core_path = os.path.join(_project_root, 'phi_geometric', 'core')

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pkg = types.ModuleType('phi_geometric')
_pkg.__path__ = [os.path.join(_project_root, 'phi_geometric')]
sys.modules['phi_geometric'] = _pkg
_core = types.ModuleType('phi_geometric.core')
_core.__path__ = [_core_path]
sys.modules['phi_geometric.core'] = _core
_load_module('phi_geometric.core.discovery', os.path.join(_core_path, 'discovery.py'))
_load_module('phi_geometric.core.cascade_navigator', os.path.join(_core_path, 'cascade_navigator.py'))
_pd_mod = _load_module('phi_geometric.core.phase_discovery', os.path.join(_core_path, 'phase_discovery.py'))
PhaseDiscovery = _pd_mod.PhaseDiscovery


# ============================================================================
# ARCHETYPE A: MAP-ONLY — "Elvish Cipher"
# ============================================================================
# A simple substitution cipher on a fantasy alphabet.
# Every rune maps to exactly one other rune. No context, no collapses.
# Domain: sequence of rune names → sequence of cipher names.

ELVISH_MAP = {
    'ash':  'mir',
    'oak':  'tel',
    'elm':  'ven',
    'yew':  'dor',
    'ivy':  'sal',
    'bay':  'nul',
    'fig':  'por',
    'rue':  'kef',
}

def apply_elvish(seq):
    return [ELVISH_MAP.get(t, t) for t in seq]

ELVISH_TRAINING = [
    ['ash', 'oak', 'elm'],
    ['yew', 'ivy', 'bay'],
    ['fig', 'rue', 'ash'],
    ['oak', 'elm', 'yew', 'ivy'],
    ['bay', 'fig', 'rue', 'ash'],
    ['elm', 'yew', 'fig'],
    ['ash', 'ivy', 'rue'],
    ['oak', 'bay', 'elm', 'fig'],
    ['yew', 'ash', 'oak'],
    ['rue', 'bay', 'ivy', 'yew'],
    ['fig', 'ash', 'bay'],
    ['ivy', 'elm', 'oak', 'rue'],
    ['ash', 'fig', 'yew', 'bay'],
    ['oak', 'rue', 'ivy'],
    ['elm', 'bay', 'ash', 'oak'],
]


# ============================================================================
# ARCHETYPE B: CONTEXT→MAP — "Traffic Signal Encoding"
# ============================================================================
# Sensors in a sequence report colors. The encoding depends on what
# the NEXT sensor reports (anticipatory encoding for a control system).
# No collapses — always same length. Some tokens are context-dependent.
#
# Rules:
#   yellow → caution  if next is red (slow down!)
#   yellow → proceed  if next is green (keep going)
#   yellow → yellow   otherwise (no change)
#   red → stop        always (consistent map)
#   green → go        always (consistent map)
#   blue → info       always (consistent map)

def apply_traffic(seq):
    result = []
    for i, tok in enumerate(seq):
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if tok == 'yellow':
            if nxt == 'red':
                result.append('caution')
            elif nxt == 'green':
                result.append('proceed')
            else:
                result.append('yellow')
        elif tok == 'red':
            result.append('stop')
        elif tok == 'green':
            result.append('go')
        elif tok == 'blue':
            result.append('info')
        else:
            result.append(tok)
    return result

TRAFFIC_TRAINING = [
    ['red', 'green', 'blue'],
    ['green', 'red', 'yellow'],
    ['yellow', 'red', 'green'],
    ['yellow', 'green', 'red'],
    ['red', 'yellow', 'red'],
    ['green', 'yellow', 'green'],
    ['blue', 'yellow', 'red', 'green'],
    ['yellow', 'red', 'yellow', 'green'],
    ['red', 'yellow', 'green', 'blue'],
    ['green', 'blue', 'yellow', 'red'],
    ['yellow', 'blue', 'red'],
    ['blue', 'red', 'yellow', 'green'],
    ['red', 'green', 'yellow', 'red'],
    ['green', 'yellow', 'red', 'blue'],
    ['blue', 'yellow', 'green', 'red'],
    ['red', 'blue', 'green'],
    ['green', 'red', 'blue'],
    ['yellow', 'green', 'blue'],
    ['blue', 'green', 'yellow', 'red'],
    ['red', 'yellow', 'blue', 'green'],
]


# ============================================================================
# ARCHETYPE C: COLLAPSE→MAP — "Musical Chord Notation"
# ============================================================================
# A sequence of notes. Adjacent notes that form a known interval
# collapse into a chord name. Remaining notes get their standard
# notation name. No context dependence — purely structural.
#
# Collapses:
#   C + E → Cmaj   (major third)
#   D + F → Dmin   (minor third)
#   E + G → Emin   (minor third)
#   G + B → Gmaj   (major third)
#
# Simple maps:
#   A → La
#   F → Fa
#   C → Do  (when not part of chord)

CHORD_COLLAPSES = {
    ('C', 'E'): ('Cmaj',),
    ('D', 'F'): ('Dmin',),
    ('E', 'G'): ('Emin',),
    ('G', 'B'): ('Gmaj',),
}

CHORD_MAP = {
    'A': 'La',
    'F': 'Fa',
    'C': 'Do',
}

def apply_chords(seq):
    # Phase 1: Collapse intervals
    collapsed = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq):
            pair = (seq[i], seq[i + 1])
            if pair in CHORD_COLLAPSES:
                collapsed.extend(CHORD_COLLAPSES[pair])
                i += 2
                continue
        collapsed.append(seq[i])
        i += 1
    # Phase 2: Simple name map
    return [CHORD_MAP.get(t, t) for t in collapsed]

CHORD_TRAINING = [
    # Collapse examples
    ['C', 'E', 'A'],
    ['D', 'F', 'A'],
    ['E', 'G', 'A'],
    ['G', 'B', 'A'],
    ['C', 'E', 'D', 'F'],
    ['G', 'B', 'C', 'E'],
    ['A', 'C', 'E'],
    ['A', 'D', 'F'],
    ['F', 'G', 'B'],
    ['C', 'E', 'G', 'B'],
    ['A', 'E', 'G', 'F'],
    ['D', 'F', 'G', 'B', 'A'],
    # Simple map examples (no collapses, equal length)
    ['A', 'F', 'C'],
    ['C', 'A', 'F'],
    ['F', 'A', 'C'],
    ['A', 'C', 'F', 'A'],
    ['C', 'F', 'A'],
    ['F', 'C', 'A', 'F'],
    # More collapse evidence
    ['C', 'E', 'F'],
    ['D', 'F', 'C'],
    ['E', 'G', 'F'],
    ['G', 'B', 'C'],
    ['A', 'C', 'E', 'A'],
    ['F', 'D', 'F', 'A'],
]


# ============================================================================
# ARCHETYPE D: COLLAPSE→CONTEXT→MAP — Already tested (IPA/Pixel)
# Included here for completeness in the archetype table.
# ============================================================================
# Using a simplified version: "Alien Language"
#
# Collapses:
#   zz → Z  (geminate simplification)
#   kh → X  (fricative merge)
#
# Context:
#   q → qw  before 'a' (labialization)
#   q → q   otherwise
#   ... wait, that would expand. Let me keep it simple.
#
# Actually, let's use:
#   v → f  before voiceless (p, t, k)    (devoicing)
#   v → v  otherwise
#
# Simple maps:
#   p → b   (voicing)
#   t → d
#   k → g

def apply_alien(seq):
    # Phase 1: Collapse geminates
    collapsed = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq):
            if seq[i] == 'z' and seq[i+1] == 'z':
                collapsed.append('Z')
                i += 2
                continue
            if seq[i] == 'k' and seq[i+1] == 'h':
                collapsed.append('X')
                i += 2
                continue
        collapsed.append(seq[i])
        i += 1

    # Phase 2: Context — devoicing (use original for context)
    voiced = list(collapsed)
    for i, tok in enumerate(collapsed):
        if tok == 'v':
            # Find original position for context
            orig_nxt = _alien_orig_next(seq, collapsed, i)
            if orig_nxt in ('p', 't', 'k'):
                voiced[i] = 'f'

    # Phase 3: Simple voicing map
    voicing = {'p': 'b', 't': 'd', 'k': 'g'}
    return [voicing.get(t, t) for t in voiced]

def _alien_orig_next(original, collapsed, collapsed_idx):
    """Get the next token from original sequence for context."""
    oi, ci = 0, 0
    while oi < len(original) and ci < len(collapsed):
        if ci == collapsed_idx:
            # Found the original position, return next original token
            # Skip past current token(s) consumed by collapse
            remaining_oi = oi
            # Advance past whatever this collapsed token consumed
            if remaining_oi + 1 < len(original):
                if (original[remaining_oi] == 'z' and
                    remaining_oi + 1 < len(original) and
                    original[remaining_oi + 1] == 'z'):
                    remaining_oi += 2
                elif (original[remaining_oi] == 'k' and
                      remaining_oi + 1 < len(original) and
                      original[remaining_oi + 1] == 'h'):
                    remaining_oi += 2
                else:
                    remaining_oi += 1
            else:
                remaining_oi += 1
            if remaining_oi < len(original):
                return original[remaining_oi]
            return None
        # Advance
        if oi + 1 < len(original):
            if original[oi] == 'z' and original[oi+1] == 'z':
                oi += 2; ci += 1; continue
            if original[oi] == 'k' and original[oi+1] == 'h':
                oi += 2; ci += 1; continue
        oi += 1; ci += 1
    return None

ALIEN_TRAINING = [
    # Collapse evidence
    ['z', 'z', 'a'],
    ['a', 'z', 'z'],
    ['k', 'h', 'a'],
    ['a', 'k', 'h'],
    ['z', 'z', 'k', 'h'],
    ['k', 'h', 'z', 'z'],
    ['a', 'z', 'z', 'a'],
    ['a', 'k', 'h', 'a'],
    # Context: v before voiceless → f
    ['v', 'p', 'a'],
    ['v', 't', 'a'],
    ['v', 'k', 'a'],
    ['a', 'v', 'p'],
    ['a', 'v', 't'],
    ['a', 'v', 'k'],
    # Context: v before other → v
    ['v', 'a', 'p'],
    ['a', 'v', 'a'],
    ['v', 'v', 'a'],
    # Simple voicing maps
    ['p', 'a', 't'],
    ['t', 'a', 'k'],
    ['k', 'a', 'p'],
    ['a', 'p', 'a'],
    ['a', 't', 'a'],
    ['a', 'k', 'a'],
    ['p', 't', 'k'],
    ['a', 'p', 't', 'k'],
    # Mixed
    ['z', 'z', 'v', 'p'],
    ['k', 'h', 'v', 't'],
    ['a', 'z', 'z', 'v', 'k', 'a'],
    ['v', 'p', 'z', 'z', 'a'],
    ['a', 'v', 'a', 'p', 't'],
]


# ============================================================================
# ARCHETYPE E: EXPAND→MAP — "Phonetic Spelling"
# ============================================================================
# Single tokens that expand into multiple phonetic tokens.
# The mirror of collapse: 1 input token → N output tokens.
#
# Expand rules:
#   x → k, s   (x is always "ks")
#   q → k, w   (q is always "kw")
#
# Simple maps:
#   a → A, b → B, c → C, d → D, e → E

def apply_phonetic(seq):
    result = []
    for tok in seq:
        if tok == 'x':
            result.extend(['k', 's'])
        elif tok == 'q':
            result.extend(['k', 'w'])
        else:
            MAP = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E'}
            result.append(MAP.get(tok, tok))
    return result

PHONETIC_TRAINING = [
    # Expand evidence (output longer than input)
    ['a', 'x', 'b'],
    ['c', 'x', 'd'],
    ['e', 'x', 'a'],
    ['b', 'x', 'c'],
    ['a', 'q', 'b'],
    ['c', 'q', 'd'],
    ['e', 'q', 'a'],
    ['b', 'q', 'c'],
    ['x', 'a', 'q'],
    ['q', 'x', 'a'],
    # Equal-length (simple maps only)
    ['a', 'b', 'c'],
    ['d', 'e', 'a'],
    ['b', 'c', 'd'],
    ['c', 'a', 'e'],
    ['e', 'd', 'b'],
    ['a', 'd', 'c'],
]


# ============================================================================
# ARCHETYPE F: COLLAPSE→EXPAND→MAP — "Chemical Reaction Notation"
# ============================================================================
# A chemical sequence where:
#   Collapse: Adjacent identical atoms merge (H+H → H2, O+O → O2)
#   Expand:   Water notation splits (W → H2, O)
#   Map:      Charge annotation (Na → Na+, Cl → Cl-)
#
# This tests the most complex archetype: collapse AND expand in same transform.

CHEM_COLLAPSES = {
    ('H', 'H'): ('H2',),
    ('O', 'O'): ('O2',),
}

CHEM_EXPANDS = {
    'W': ['H2', 'O'],  # water shorthand
}

CHEM_MAP = {
    'Na': 'Na+',
    'Cl': 'Cl-',
    'K':  'K+',
}

def apply_chem(seq):
    # Phase 1: Expand W → H2, O
    expanded = []
    for tok in seq:
        if tok in CHEM_EXPANDS:
            expanded.extend(CHEM_EXPANDS[tok])
        else:
            expanded.append(tok)
    # Phase 2: Collapse identical pairs
    collapsed = []
    i = 0
    while i < len(expanded):
        if i + 1 < len(expanded):
            pair = (expanded[i], expanded[i + 1])
            if pair in CHEM_COLLAPSES:
                collapsed.extend(CHEM_COLLAPSES[pair])
                i += 2
                continue
        collapsed.append(expanded[i])
        i += 1
    # Phase 3: Charge map
    return [CHEM_MAP.get(t, t) for t in collapsed]

CHEM_TRAINING = [
    # Expand evidence (W → H2, O)
    ['Na', 'W'],           # → Na+, H2, O
    ['W', 'Cl'],           # → H2, O, Cl-
    ['K', 'W'],            # → K+, H2, O
    ['W', 'Na'],           # → H2, O, Na+
    ['Na', 'W', 'Cl'],     # → Na+, H2, O, Cl-
    ['K', 'W', 'Na'],      # → K+, H2, O, Na+
    # Collapse evidence (H+H → H2, O+O → O2)
    ['H', 'H', 'Na'],      # → H2, Na+
    ['Na', 'H', 'H'],      # → Na+, H2
    ['O', 'O', 'Cl'],      # → O2, Cl-
    ['Cl', 'O', 'O'],      # → Cl-, O2
    ['H', 'H', 'O', 'O'],  # → H2, O2
    ['K', 'H', 'H'],       # → K+, H2
    ['H', 'H', 'K'],       # → H2, K+
    ['O', 'O', 'Na'],      # → O2, Na+
    # Equal-length (simple maps only)
    ['Na', 'Cl', 'K'],     # → Na+, Cl-, K+
    ['K', 'Na', 'Cl'],     # → K+, Na+, Cl-
    ['Cl', 'K', 'Na'],     # → Cl-, K+, Na+
    ['Na', 'K', 'Cl'],     # → Na+, K+, Cl-
    ['Cl', 'Na', 'K'],     # → Cl-, Na+, K+
    ['K', 'Cl', 'Na'],     # → K+, Cl-, Na+
]


# ============================================================================
# ARCHETYPE G: EXPAND→CONTEXT→MAP — "Morse-like Encoding"
# ============================================================================
# A simplified encoding where:
#   Expand:   X → d, d  (double-dot)
#   Context:  s → S if next is d (stressed), s → s otherwise
#   Map:      d → D, a → A, b → B
#
# Tests expand + context (no collapse).

def apply_morse(seq):
    # Phase 1: Expand X → d, d
    expanded = []
    for tok in seq:
        if tok == 'X':
            expanded.extend(['d', 'd'])
        else:
            expanded.append(tok)
    # Phase 2: Context — s before d → S
    result = []
    for i, tok in enumerate(expanded):
        if tok == 's':
            nxt = expanded[i + 1] if i + 1 < len(expanded) else None
            if nxt == 'd':
                result.append('S')
            else:
                result.append('s')
        else:
            result.append(tok)
    # Phase 3: Simple map
    MAP = {'d': 'D', 'a': 'A', 'b': 'B'}
    return [MAP.get(t, t) for t in result]

MORSE_TRAINING = [
    # Expand evidence (X → d, d)
    ['a', 'X', 'b'],       # → A, D, D, B
    ['b', 'X', 'a'],       # → B, D, D, A
    ['a', 'X', 'a'],       # → A, D, D, A
    ['b', 'X', 'b'],       # → B, D, D, B
    # Context evidence (s before d → S)
    ['s', 'd', 'a'],       # → S, D, A
    ['a', 's', 'd'],       # → A, S, D
    ['s', 'd', 'b'],       # → S, D, B
    ['b', 's', 'd'],       # → B, S, D
    # Context: s NOT before d → s
    ['s', 'a', 'd'],       # → s, A, D
    ['s', 'b', 'a'],       # → s, B, A
    ['a', 's', 'a'],       # → A, s, A
    ['a', 's', 'b'],       # → A, s, B
    # Equal-length (simple maps)
    ['a', 'b', 'd'],       # → A, B, D
    ['d', 'a', 'b'],       # → D, A, B
    ['b', 'd', 'a'],       # → B, D, A
    ['d', 'b', 'a'],       # → D, B, A
    ['a', 'd', 'b'],       # → A, D, B
    ['b', 'a', 'd'],       # → B, A, D
]


# ============================================================================
# ARCHETYPE H: LONG-RANGE CONTEXT→MAP — "Vowel Harmony"
# ============================================================================
# Linguistic vowel harmony: 'a' changes based on nearest PRECEDING vowel.
#   'a' → 'æ' if nearest preceding vowel is 'e' (front harmony)
#   'a' → 'ɑ' if nearest preceding vowel is 'o' (back harmony)
#   'a' → 'a' if no preceding vowel (default)
# Consonants: simple maps c→C, d→D, f→F, g→G
#
# The twist: consonants separate the harmony trigger from 'a' by 2-5 positions.
# Fixed context_window=1 can't see the trigger. Geometric context can.
# This archetype REQUIRES geometric=True to achieve 100%.

HARMONY_CMAP = {'c': 'C', 'd': 'D', 'f': 'F', 'g': 'G', 'e': 'e', 'o': 'o'}

def _nearest_prev_vowel(seq, pos):
    for i in range(pos - 1, -1, -1):
        if seq[i] in ('e', 'o'):
            return seq[i]
    return None

def apply_harmony(seq):
    result = []
    for i, tok in enumerate(seq):
        if tok == 'a':
            v = _nearest_prev_vowel(seq, i)
            if v == 'e':
                result.append('æ')
            elif v == 'o':
                result.append('ɑ')
            else:
                result.append('a')
        else:
            result.append(HARMONY_CMAP.get(tok, tok))
    return result

HARMONY_TRAINING = [
    # Distance 2: vowel + 1 consonant + a
    ['e', 'c', 'a'], ['o', 'c', 'a'], ['e', 'd', 'a'], ['o', 'd', 'a'],
    ['e', 'f', 'a'], ['o', 'f', 'a'],
    # Distance 3: vowel + 2 consonants + a
    ['e', 'c', 'd', 'a'], ['o', 'c', 'd', 'a'],
    ['e', 'f', 'g', 'a'], ['o', 'f', 'g', 'a'],
    # Distance 4: vowel + 3 consonants + a
    ['e', 'c', 'd', 'f', 'a'], ['o', 'c', 'd', 'f', 'a'],
    # No preceding vowel: a stays a
    ['c', 'a', 'd'], ['d', 'a', 'f'], ['f', 'a', 'g'], ['c', 'd', 'a'],
    # Pure consonant (equal length, for map rules)
    ['c', 'd', 'f', 'g'], ['g', 'f', 'd', 'c'],
    ['c', 'f', 'c', 'd'], ['d', 'g', 'f', 'c'],
]


# ============================================================================
# RUN ALL ARCHETYPES
# ============================================================================

def classify_phases(result):
    """Extract the archetype signature from a PhaseDiscoveryResult."""
    types = []
    for phase in result.phases:
        if phase.multi_token_patterns:
            types.append('collapse')
        elif phase.expand_patterns:
            types.append('expand')
        elif phase.context_dependent:
            types.append('context')
        else:
            types.append('map')
    return types


def test_archetype(name, expected_type, apply_fn, training_data, test_data=None,
                   geometric=False):
    """Run PhaseDiscovery on a toy domain and report results."""
    print(f"\n{'='*70}")
    print(f"  ARCHETYPE {name}")
    print(f"  Expected: [{' → '.join(expected_type)}]")
    if geometric:
        print(f"  Mode: geometric φ-context")
    print(f"{'='*70}\n")

    pd = PhaseDiscovery(context_window=1, geometric=geometric)
    pairs = []
    for seq in training_data:
        out = apply_fn(seq)
        pairs.append((seq, out))
        pd.add_pair(seq, out)

    result = pd.discover()

    discovered = classify_phases(result)
    match = discovered == expected_type
    match_str = '✓ MATCH' if match else '✗ MISMATCH'

    print(f"  Discovered: [{' → '.join(discovered)}]  {match_str}")
    print()

    # Show phase details
    for phase in result.phases:
        if phase.multi_token_patterns:
            print(f"  COLLAPSE: {len(phase.multi_token_patterns)} patterns")
            for mp in phase.multi_token_patterns[:6]:
                inp = '+'.join(str(t) for t in mp.input_tokens)
                out = '+'.join(str(t) for t in mp.output_tokens)
                print(f"    {inp} → {out}  (×{mp.evidence_count})")
        elif phase.expand_patterns:
            print(f"  EXPAND: {len(phase.expand_patterns)} patterns")
            for ep in phase.expand_patterns[:6]:
                out = '+'.join(str(t) for t in ep.output_tokens)
                print(f"    {ep.input_token} → {out}  (×{ep.evidence_count})")
        elif phase.context_dependent:
            print(f"  CONTEXT: {len(phase.rule_observations)} token(s)")
            for tok, obs in phase.rule_observations.items():
                outputs = defaultdict(int)
                for o, _ in obs:
                    outputs[o] += 1
                out_str = ', '.join(f'{o}×{c}' for o, c in
                                   sorted(outputs.items(), key=lambda x: -x[1]))
                print(f"    {tok} → {{{out_str}}}")
        else:
            print(f"  MAP: {len(phase.token_rules)} rules")
            for k, v in sorted(phase.token_rules.items(), key=str):
                print(f"    {k} → {''.join(str(t) for t in v)}")
    print()

    # Validate
    nav = result.to_navigator()
    correct = 0
    for seq in training_data:
        expected = apply_fn(seq)
        actual = nav.execute(seq).output_elements
        if actual == expected:
            correct += 1

    print(f"  Training accuracy: {correct}/{len(training_data)}")

    # Test data
    if test_data:
        test_correct = 0
        for seq in test_data:
            expected = apply_fn(seq)
            actual = nav.execute(seq).output_elements
            if actual == expected:
                test_correct += 1
            else:
                pass  # silent on test failures
        print(f"  Generalization:    {test_correct}/{len(test_data)}")

    return discovered, match, correct, len(training_data)


def main():
    print("=" * 70)
    print("  TRANSFORMATION ARCHETYPE SURVEY")
    print("  Can PhaseDiscovery identify different cascade structures?")
    print("=" * 70)

    results = []

    # A: Map-only
    d, m, c, t = test_archetype(
        'A: MAP-ONLY — "Elvish Cipher"',
        ['map'],
        apply_elvish,
        ELVISH_TRAINING,
        test_data=[
            ['ash', 'elm', 'ivy', 'fig'],
            ['yew', 'bay', 'oak', 'rue'],
            ['fig', 'ivy', 'ash'],
        ]
    )
    results.append(('A: map-only', d, m, c, t))

    # B: Context→Map
    d, m, c, t = test_archetype(
        'B: CONTEXT→MAP — "Traffic Signals"',
        ['context', 'map'],
        apply_traffic,
        TRAFFIC_TRAINING,
        test_data=[
            ['yellow', 'red', 'blue', 'green'],
            ['green', 'yellow', 'green', 'red'],
            ['blue', 'yellow', 'green', 'yellow', 'red'],
        ]
    )
    results.append(('B: context→map', d, m, c, t))

    # C: Collapse→Map
    d, m, c, t = test_archetype(
        'C: COLLAPSE→MAP — "Musical Chords"',
        ['collapse', 'map'],
        apply_chords,
        CHORD_TRAINING,
        test_data=[
            ['C', 'E', 'A', 'D', 'F'],
            ['G', 'B', 'F', 'A'],
            ['A', 'E', 'G', 'C'],
        ]
    )
    results.append(('C: collapse→map', d, m, c, t))

    # D: Collapse→Context→Map
    d, m, c, t = test_archetype(
        'D: COLLAPSE→CONTEXT→MAP — "Alien Language"',
        ['collapse', 'context', 'map'],
        apply_alien,
        ALIEN_TRAINING,
        test_data=[
            ['z', 'z', 'v', 't', 'a'],
            ['a', 'k', 'h', 'v', 'p'],
            ['v', 'a', 'k', 'z', 'z'],
        ]
    )
    results.append(('D: collapse→ctx→map', d, m, c, t))

    # E: Expand→Map
    d, m, c, t = test_archetype(
        'E: EXPAND→MAP — "Phonetic Spelling"',
        ['expand', 'map'],
        apply_phonetic,
        PHONETIC_TRAINING,
        test_data=[
            ['a', 'x', 'e'],
            ['d', 'q', 'b'],
            ['x', 'q', 'c'],
        ]
    )
    results.append(('E: expand→map', d, m, c, t))

    # F: Expand+Collapse+Map (expand runs first at priority 90, collapse at 80)
    d, m, c, t = test_archetype(
        'F: EXPAND→COLLAPSE→MAP — "Chemical Notation"',
        ['expand', 'collapse', 'map'],
        apply_chem,
        CHEM_TRAINING,
        test_data=[
            ['Na', 'W', 'K'],
            ['H', 'H', 'Cl'],
            ['W', 'K'],
        ]
    )
    results.append(('F: exp→col→map', d, m, c, t))

    # G: Expand+Context+Map
    d, m, c, t = test_archetype(
        'G: EXPAND→CONTEXT→MAP — "Morse-like Encoding"',
        ['expand', 'context', 'map'],
        apply_morse,
        MORSE_TRAINING,
        test_data=[
            ['s', 'X', 'a'],
            ['a', 's', 'X'],
            ['s', 'a', 'X', 'b'],
        ]
    )
    results.append(('G: exp→ctx→map', d, m, c, t))

    # H: Long-range Context (geometric required)
    d, m, c, t = test_archetype(
        'H: LONG-RANGE CONTEXT→MAP — "Vowel Harmony"',
        ['context', 'map'],
        apply_harmony,
        HARMONY_TRAINING,
        test_data=[
            ['e', 'g', 'a'],
            ['o', 'g', 'a'],
            ['e', 'd', 'f', 'a'],
            ['o', 'g', 'c', 'a'],
            ['g', 'a', 'c'],
        ],
        geometric=True,
    )
    results.append(('H: φ-ctx→map', d, m, c, t))

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  ARCHETYPE SURVEY SUMMARY")
    print("=" * 70 + "\n")

    print(f"  {'Archetype':<25s} {'Expected':>20s} {'Discovered':>20s} {'Match':>7s} {'Acc':>8s}")
    print(f"  {'─'*25} {'─'*20} {'─'*20} {'─'*7} {'─'*8}")

    all_match = True
    archetype_expected = {
        'A: map-only':          ['map'],
        'B: context→map':       ['ctx','map'],
        'C: collapse→map':      ['col','map'],
        'D: collapse→ctx→map':  ['col','ctx','map'],
        'E: expand→map':        ['exp','map'],
        'F: exp→col→map':       ['exp','col','map'],
        'G: exp→ctx→map':       ['exp','ctx','map'],
        'H: φ-ctx→map':         ['ctx','map'],
    }
    for name, discovered, match, correct, total in results:
        exp_str = '→'.join(archetype_expected.get(name, ['?']))
        disc_str = '→'.join(d[:3] for d in discovered)
        m_str = '✓' if match else '✗'
        acc_str = f'{correct}/{total}'
        print(f"  {name:<25s} {exp_str:>20s} {disc_str:>20s} {m_str:>7s} {acc_str:>8s}")
        if not match:
            all_match = False

    print()
    if all_match:
        print("  ALL ARCHETYPES CORRECTLY IDENTIFIED!")
        print("  PhaseDiscovery is archetype-agnostic — it discovers the")
        print("  transformation structure regardless of domain or complexity.")
    else:
        print("  Some archetypes were misidentified. See details above.")

    print()


if __name__ == '__main__':
    main()
