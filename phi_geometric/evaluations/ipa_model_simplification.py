#!/usr/bin/env python3
"""
IPA Model Simplification: Finding the Least Common Denominator for Shapes
=========================================================================

Given the trained IPA model (29 rules, 159 gate_step primitives),
explore whether the geometric structure can be simplified.

Three perspectives:
1. AIG: What is the minimal universal basis for geometric operations?
2. Mesh: Can we reduce primitives while preserving behavior?
3. Information: How many bits does the model actually contain?

The hypothesis: Just as AIGs reduce any Boolean circuit to AND+NOT,
our geometric model reduces to COMPARE+AND+SCALE — and then we can
find shared structure (like AIG structural hashing) to simplify.
"""

import sys
import math
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from phi_geometric.evaluations.ipa_geometric_demo import (
    GeometricProgram, GeometricRule, make_examples, LESSONS,
    learn_magic_e_rules, VOWELS, CONSONANTS, LONG_VOWELS,
)
from phi_geometric.evaluations.auto_context_detection import (
    build_rules as auto_build_rules,
)


# ============================================================================
# Part 1: Atomic Operations — The "Least Common Denominator"
# ============================================================================

@dataclass
class Atom:
    """An atomic geometric operation — the smallest indivisible unit.
    
    Every rule in the IPA model decomposes into these atoms:
    - COMPARE: Does variable X equal value V?  → {0, 1}
    - AND:     Are both inputs true?           → {0, 1}
    - OR:      Is either input true?           → {0, 1}
    - NOT:     Invert                          → {0, 1}
    - EMIT:    Produce output string           → str
    - MUX:     Select output based on control  → str
    
    This is the geometric equivalent of AIG's AND+NOT basis,
    extended with COMPARE (character-level test) and EMIT (output).
    """
    op: str           # 'compare', 'and', 'or', 'not', 'emit', 'mux'
    variable: str     # What is being tested (e.g., 'input_char', 'next_char')
    value: Any        # What it's compared to (e.g., 'a', 'e', True)
    output: Any       # For emit: the output string
    source_rule: str  # Which rule this came from
    atom_id: int = 0  # Unique ID for sharing analysis
    
    def signature(self):
        """Canonical signature for structural hashing."""
        return (self.op, self.variable, self.value)


@dataclass 
class AtomicRule:
    """A rule decomposed into its atoms."""
    name: str
    rule_type: str
    atoms: List[Atom] = field(default_factory=list)
    comparisons: int = 0
    logic_ops: int = 0
    outputs: int = 0


def decompose_char_rule(rule: GeometricRule) -> AtomicRule:
    """Decompose a character RECT rule into atoms.
    
    a→æ becomes:
      COMPARE(input_char, 'a') → if true, EMIT('æ')
    
    This is 1 comparison + 1 output = 2 atoms.
    """
    ar = AtomicRule(
        name=f"{rule.input_char}→{rule.output_char}",
        rule_type='char_rect',
    )
    ar.atoms.append(Atom(
        op='compare', variable='input_char',
        value=rule.input_char, output=None,
        source_rule=ar.name,
    ))
    ar.atoms.append(Atom(
        op='emit', variable='output',
        value=None, output=rule.output_char,
        source_rule=ar.name,
    ))
    ar.comparisons = 1
    ar.outputs = 1
    return ar


def decompose_digraph_rule(c1, c2, replacement, frozen=False) -> AtomicRule:
    """Decompose a digraph rule into atoms.
    
    sh→ʃ becomes:
      COMPARE(char[i], 's') AND COMPARE(char[i+1], 'h') → EMIT('ʃ')
    
    This is 2 comparisons + 1 AND + 1 output = 4 atoms.
    """
    name = f"{c1}{c2}→{replacement}" + (" [frozen]" if frozen else "")
    ar = AtomicRule(name=name, rule_type='digraph')
    
    ar.atoms.append(Atom(
        op='compare', variable='input_char',
        value=c1, output=None, source_rule=name,
    ))
    ar.atoms.append(Atom(
        op='compare', variable='next_char',
        value=c2, output=None, source_rule=name,
    ))
    ar.atoms.append(Atom(
        op='and', variable='pair_match',
        value=(c1, c2), output=None, source_rule=name,
    ))
    for ch in (replacement if replacement else ['∅']):
        ar.atoms.append(Atom(
            op='emit', variable='output',
            value=None, output=ch, source_rule=name,
        ))
    
    ar.comparisons = 2
    ar.logic_ops = 1
    ar.outputs = max(1, len(replacement))
    return ar


def decompose_context_rule(target_char, rule_obj) -> AtomicRule:
    """Decompose a context/geared rule into atoms."""
    name = f"{target_char}→context"
    ar = AtomicRule(name=name, rule_type=rule_obj.rule_type)
    
    # Input comparison
    ar.atoms.append(Atom(
        op='compare', variable='input_char',
        value=target_char, output=None, source_rule=name,
    ))
    ar.comparisons = 1
    
    if rule_obj.rule_type == 'context':
        # Simple selector: compare one context variable
        var = rule_obj.params['selector_variable']
        sel_map = rule_obj.params['selector_map']
        for val, out in sel_map.items():
            ar.atoms.append(Atom(
                op='compare', variable=var,
                value=val, output=None, source_rule=name,
            ))
            ar.atoms.append(Atom(
                op='emit', variable='output',
                value=val, output=out, source_rule=name,
            ))
            ar.comparisons += 1
            ar.outputs += 1
    
    elif rule_obj.rule_type == 'geared':
        # Geared: coarse selector + fine gears for ambiguous teeth
        coarse_var = rule_obj.params['coarse_var']
        pure_map = rule_obj.params['pure_map']
        fine_gears = rule_obj.params.get('fine_gears', {})
        
        # Pure teeth
        for val, out in pure_map.items():
            ar.atoms.append(Atom(
                op='compare', variable=coarse_var,
                value=val, output=None, source_rule=name,
            ))
            ar.atoms.append(Atom(
                op='emit', variable='output',
                value=val, output=out, source_rule=name,
            ))
            ar.comparisons += 1
            ar.outputs += 1
        
        # Fine gears
        for coarse_val, fine_data in fine_gears.items():
            fine_var, fine_map, _, _, zone_default = fine_data
            ar.atoms.append(Atom(
                op='compare', variable=coarse_var,
                value=coarse_val, output=None, source_rule=name,
            ))
            ar.comparisons += 1
            ar.logic_ops += 1  # AND with coarse
            
            if fine_var and fine_map:
                for fval, fout in fine_map.items():
                    ar.atoms.append(Atom(
                        op='compare', variable=fine_var,
                        value=fval, output=None, source_rule=name,
                    ))
                    ar.atoms.append(Atom(
                        op='and', variable='fine_match',
                        value=(coarse_val, fval), output=None, source_rule=name,
                    ))
                    ar.atoms.append(Atom(
                        op='emit', variable='output',
                        value=(coarse_val, fval), output=fout, source_rule=name,
                    ))
                    ar.comparisons += 1
                    ar.logic_ops += 1
                    ar.outputs += 1
            
            # Zone default
            ar.atoms.append(Atom(
                op='emit', variable='zone_default',
                value=coarse_val, output=zone_default, source_rule=name,
            ))
            ar.outputs += 1
    
    return ar


def decompose_magic_e_rules(rules: dict) -> List[AtomicRule]:
    """Decompose the trained magic-e rules into atoms."""
    result = []
    
    for vowel, (rule_type, data) in rules.items():
        name = f"magic_e_{vowel}"
        ar = AtomicRule(name=name, rule_type=f'magic_e_{rule_type}')
        
        # Compare: is this a magic-e vowel position?
        ar.atoms.append(Atom(
            op='compare', variable='is_magic_e',
            value=True, output=None, source_rule=name,
        ))
        # Compare: which vowel?
        ar.atoms.append(Atom(
            op='compare', variable='vowel',
            value=vowel, output=None, source_rule=name,
        ))
        ar.comparisons = 2
        
        if rule_type == 'simple':
            ar.atoms.append(Atom(
                op='emit', variable='output',
                value=None, output=data, source_rule=name,
            ))
            ar.outputs = 1
        
        elif rule_type == 'geared':
            coarse_var = data['coarse_var']
            for val, out in data['pure_map'].items():
                ar.atoms.append(Atom(
                    op='compare', variable=coarse_var,
                    value=val, output=None, source_rule=name,
                ))
                ar.atoms.append(Atom(
                    op='emit', variable='output',
                    value=val, output=out, source_rule=name,
                ))
                ar.comparisons += 1
                ar.outputs += 1
            
            for coarse_val, fg in data.get('fine_gears', {}).items():
                fine_var, fine_map, _, _, zone_default = fg
                ar.atoms.append(Atom(
                    op='compare', variable=coarse_var,
                    value=coarse_val, output=None, source_rule=name,
                ))
                ar.comparisons += 1
                ar.logic_ops += 1
                
                if fine_var and fine_map:
                    for fval, fout in fine_map.items():
                        ar.atoms.append(Atom(
                            op='compare', variable=fine_var,
                            value=fval, output=None, source_rule=name,
                        ))
                        ar.atoms.append(Atom(
                            op='and', variable='fine_match',
                            value=(coarse_val, fval), output=None,
                            source_rule=name,
                        ))
                        ar.atoms.append(Atom(
                            op='emit', variable='output',
                            value=(coarse_val, fval), output=fout,
                            source_rule=name,
                        ))
                        ar.comparisons += 1
                        ar.logic_ops += 1
                        ar.outputs += 1
                
                ar.atoms.append(Atom(
                    op='emit', variable='zone_default',
                    value=coarse_val, output=zone_default,
                    source_rule=name,
                ))
                ar.outputs += 1
        
        result.append(ar)
    
    return result


# ============================================================================
# Part 2: Sharing Analysis — The "Structural Hashing"
# ============================================================================

def find_shared_comparisons(all_rules: List[AtomicRule]) -> Dict[tuple, List[str]]:
    """Find comparisons that appear in multiple rules.
    
    Like AIG structural hashing — identical sub-circuits are shared.
    """
    comparison_users = defaultdict(list)
    
    for rule in all_rules:
        for atom in rule.atoms:
            if atom.op == 'compare':
                sig = atom.signature()
                comparison_users[sig].append(rule.name)
    
    # Filter to shared (appears in 2+ rules)
    shared = {sig: users for sig, users in comparison_users.items()
              if len(users) > 1}
    
    return shared


# ============================================================================
# Part 3: Information Content Analysis
# ============================================================================

def compute_information_content(program, all_rules):
    """Compute the model's information content in bits.
    
    The key question: how many bits does the model ACTUALLY contain?
    159 gate_steps is the ENCODING; the information content is smaller.
    """
    info = {}
    
    # Character rules: each needs input_char (5 bits for 26 letters)
    # + output_cp (8 bits for IPA range) = 13 bits each
    n_char = len(program.rules)
    info['char_rules'] = n_char * 13
    
    # Digraph rules: each needs 2 input chars (10 bits)
    # + output string (avg 1.5 chars × 8 bits) + frozen flag (1 bit) = 23 bits
    n_digraph = len(program.digraph_rules)
    avg_output_len = sum(len(v) for v in program.digraph_rules.values()) / max(1, n_digraph)
    info['digraph_rules'] = int(n_digraph * (10 + avg_output_len * 8 + 1))
    
    # Context rules: selector variable (3 bits for ~8 options)
    # + branch table (N entries × (5 bits key + 8 bits output))
    ctx_bits = 0
    for char, rule in program.context_rules.items():
        ctx_bits += 3  # selector variable
        p = rule.params if hasattr(rule, 'params') else {}
        sel_map = p.get('selector_map', {})
        if sel_map:
            ctx_bits += len(sel_map) * 13
        pure_map = p.get('pure_map', {})
        if pure_map:
            ctx_bits += len(pure_map) * 13
        fine_gears = p.get('fine_gears', {})
        for _, fg in fine_gears.items():
            fine_var, fine_map, _, _, _ = fg
            if fine_map:
                ctx_bits += 3 + len(fine_map) * 13
    info['context_rules'] = ctx_bits
    
    # Magic-e rules: similar structure
    me_bits = 0
    if program.magic_e_rules:
        for vowel, (rtype, data) in program.magic_e_rules.items():
            me_bits += 5  # vowel identity
            if rtype == 'simple':
                me_bits += 8  # output
            elif rtype == 'geared':
                me_bits += 3  # coarse var
                if 'pure_map' in data:
                    me_bits += len(data['pure_map']) * 13
                if 'fine_gears' in data:
                    for _, fg in data['fine_gears'].items():
                        fv, fm, _, _, _ = fg
                        if fm:
                            me_bits += 3 + len(fm) * 13
    info['magic_e_rules'] = me_bits
    
    # Phase 0 detectors (detect_magic_e, detect_igh, detect_silent_final_e)
    # These are fixed algorithms, not parameterized — count as "code" not "data"
    info['phase0_detectors'] = 0  # algorithmic, no parameters
    
    info['total'] = sum(info.values())
    info['total_bytes'] = math.ceil(info['total'] / 8)
    
    return info


# ============================================================================
# Part 4: Template Vocabulary — The "Mesh Faces"
# ============================================================================

@dataclass
class ShapeTemplate:
    """A template shape — the "least common denominator" face type.
    
    Like mesh faces: a few face TYPES, each instantiated many times
    with different parameters.
    """
    name: str
    description: str
    parameter_schema: List[str]  # what parameters it needs
    instances: int = 0
    atoms_per_instance: float = 0.0


def identify_templates(all_rules: List[AtomicRule]) -> List[ShapeTemplate]:
    """Identify the minimal set of template shapes.
    
    Every rule in the model is an INSTANCE of one of these templates.
    """
    templates = []
    
    # Group rules by type
    by_type = defaultdict(list)
    for rule in all_rules:
        by_type[rule.rule_type].append(rule)
    
    for rtype, rules in sorted(by_type.items()):
        avg_atoms = sum(len(r.atoms) for r in rules) / len(rules)
        avg_compare = sum(r.comparisons for r in rules) / len(rules)
        avg_outputs = sum(r.outputs for r in rules) / len(rules)
        
        if rtype == 'char_rect':
            schema = ['input_char', 'output_char']
        elif rtype == 'digraph':
            schema = ['char1', 'char2', 'replacement', 'frozen']
        elif rtype == 'context':
            schema = ['target_char', 'selector_var', 'branch_map']
        elif rtype == 'geared':
            schema = ['target_char', 'coarse_var', 'pure_map', 'fine_gears']
        elif rtype.startswith('magic_e_'):
            schema = ['vowel', 'rule_type', 'gear_tree']
        else:
            schema = ['params']
        
        templates.append(ShapeTemplate(
            name=rtype,
            description=f"{len(rules)} instances, avg {avg_atoms:.1f} atoms each",
            parameter_schema=schema,
            instances=len(rules),
            atoms_per_instance=avg_atoms,
        ))
    
    return templates


# ============================================================================
# Part 5: Simplified Execution Model
# ============================================================================

# Phase 0 detectors — pure Python, no numpy
# (imported from ipa_geometric_demo: detect_magic_e, detect_igh,
#  detect_silent_final_e, VOWELS, CONSONANTS)
# Context extraction — pure Python
# (imported from auto_context_detection: extract_context_at)

class SimplifiedExecutor:
    """A simplified execution model using ONLY lookup tables.
    
    No numpy. No gate_step. No GeometricRule objects. No floating-point.
    Just Python dicts implementing the same 4-phase pipeline.
    
    This IS mesh simplification: the continuous gate_step surfaces
    are replaced by flat lookup faces. On discrete character inputs,
    the behavior is IDENTICAL.
    
    Build from a trained GeometricProgram via:
        executor = SimplifiedExecutor.simplify(program)
        result = executor.apply_text("hello")
    """
    
    def __init__(self):
        # Phase 0: algorithmic detectors (no parameters to store)
        self.magic_e_enabled = False
        
        # Phase 1: Digraph table
        # {(c1, c2): replacement_str}
        self.digraphs = {}
        self.frozen_digraphs = set()  # set of (c1, c2) keys
        
        # Phase 2: Context rules — flattened to pure dicts
        # {input_char: {
        #     'type': 'context' | 'geared',
        #     'selector_var': str,
        #     'selector_map': {value: output},  # for 'context' type
        #     'default': str,
        #     'coarse_var': str,       # for 'geared' type
        #     'pure_map': {val: out},
        #     'fine_gears': {coarse_val: (fine_var, {fine_val: out}, zone_default)},
        # }}
        self.context_tables = {}
        
        # Phase 3: Character map — single flat dict
        # {input_char: output_char}
        self.char_map = {}
        
        # Magic-e: flattened gear tables per vowel
        # {vowel: {
        #     'type': 'simple' | 'geared',
        #     'output': str,           # for 'simple'
        #     'coarse_var': str,       # for 'geared'
        #     'pure_map': {val: out},
        #     'fine_gears': {coarse_val: (fine_var, {fine_val: out}, zone_default)},
        #     'default': str,
        # }}
        self.magic_e_tables = {}
    
    @classmethod
    def simplify(cls, program):
        """Extract pure lookup tables from a trained GeometricProgram.
        
        This is the mesh simplification: strip away the continuous
        gate_step machinery, keep only the discrete structure.
        """
        exe = cls()
        exe.magic_e_enabled = program.magic_e_enabled
        
        # Phase 1: Digraphs (already a dict — just copy)
        exe.digraphs = dict(program.digraph_rules)
        exe.frozen_digraphs = set(program.frozen_digraphs)
        
        # Phase 3: Character rules → flat map
        for rule in program.rules:
            exe.char_map[rule.input_char] = rule.output_char
        
        # Phase 2: Context rules → flat tables
        for char, rule in program.context_rules.items():
            p = rule.params
            if rule.rule_type == 'context':
                exe.context_tables[char] = {
                    'type': 'context',
                    'selector_var': p['selector_variable'],
                    'selector_map': dict(p['selector_map']),
                    'default': p.get('default_output', char),
                }
            elif rule.rule_type == 'geared':
                fg_flat = {}
                for cv, (fv, fm, _, _, zd) in p.get('fine_gears', {}).items():
                    fg_flat[cv] = (fv, dict(fm) if fm else {}, zd)
                exe.context_tables[char] = {
                    'type': 'geared',
                    'coarse_var': p['coarse_var'],
                    'pure_map': dict(p['pure_map']),
                    'fine_gears': fg_flat,
                    'default': p.get('default_output', char),
                }
        
        # Magic-e rules → flat tables
        if program.magic_e_rules:
            for vowel, (rtype, data) in program.magic_e_rules.items():
                if rtype == 'simple':
                    exe.magic_e_tables[vowel] = {
                        'type': 'simple',
                        'output': data,
                    }
                elif rtype == 'geared':
                    fg_flat = {}
                    for cv, (fv, fm, _, _, zd) in data.get('fine_gears', {}).items():
                        fg_flat[cv] = (fv, dict(fm) if fm else {}, zd)
                    exe.magic_e_tables[vowel] = {
                        'type': 'geared',
                        'coarse_var': data['coarse_var'],
                        'pure_map': dict(data['pure_map']),
                        'fine_gears': fg_flat,
                        'default': data.get('default', LONG_VOWELS.get(vowel, vowel)),
                    }
        
        return exe
    
    def _lookup_context(self, char, ctx):
        """Phase 2: Look up context-dependent rule via flat table."""
        table = self.context_tables[char]
        
        if table['type'] == 'context':
            val = ctx.get(table['selector_var'])
            return table['selector_map'].get(val, table['default'])
        
        elif table['type'] == 'geared':
            coarse_val = ctx.get(table['coarse_var'])
            # Pure map
            if coarse_val in table['pure_map']:
                return table['pure_map'][coarse_val]
            # Fine gears
            if coarse_val in table['fine_gears']:
                fine_var, fine_map, zone_default = table['fine_gears'][coarse_val]
                if fine_var:
                    fine_val = ctx.get(fine_var)
                    if fine_val in fine_map:
                        return fine_map[fine_val]
                return zone_default
            return table['default']
        
        return char
    
    def _lookup_magic_e(self, vowel, ctx):
        """Look up magic-e rule via flat table."""
        if vowel not in self.magic_e_tables:
            return LONG_VOWELS.get(vowel, vowel)
        
        table = self.magic_e_tables[vowel]
        
        if table['type'] == 'simple':
            return table['output']
        
        elif table['type'] == 'geared':
            coarse_val = ctx.get(table['coarse_var'])
            if coarse_val in table['pure_map']:
                return table['pure_map'][coarse_val]
            if coarse_val in table['fine_gears']:
                fine_var, fine_map, zone_default = table['fine_gears'][coarse_val]
                if fine_var:
                    fine_val = ctx.get(fine_var)
                    if fine_val in fine_map:
                        return fine_map[fine_val]
                return zone_default
            return table['default']
        
        return LONG_VOWELS.get(vowel, vowel)
    
    def apply_text(self, text):
        """Apply the simplified model using the same 4-phase pipeline.
        
        Identical logic to GeometricProgram.apply_text, but using
        only dict lookups — no gate_step, no numpy, no float arithmetic.
        """
        from phi_geometric.evaluations.ipa_geometric_demo import (
            detect_magic_e, detect_igh, detect_silent_final_e,
        )
        from phi_geometric.evaluations.auto_context_detection import (
            extract_context_at,
        )
        
        chars = list(text)
        chars_lc = [c.lower() for c in chars]
        
        # Phase 0: Feature extraction (same algorithm, pure Python)
        magic_e_ctx = {}
        igh_vowels = set()
        if self.magic_e_enabled:
            magic_vowels, silent_e_positions = detect_magic_e(chars)
            for mi in magic_vowels:
                magic_e_ctx[mi] = extract_context_at(chars_lc, mi)
            igh_v, igh_s = detect_igh(chars)
            igh_vowels = igh_v
            silent_e_positions |= igh_s
            magic_vowels -= igh_vowels
            silent_final = detect_silent_final_e(chars, silent_e_positions)
            silent_e_positions |= silent_final
        else:
            magic_vowels, silent_e_positions = set(), set()
        
        # Phase 1: Digraph collapse (dict lookups only)
        i = 0
        processed = []
        orig_map = []
        frozen = set()
        silent = set()
        magic_v_processed = {}
        
        while i < len(chars):
            if i in silent_e_positions:
                silent.add(len(processed))
                orig_map.append(i)
                processed.append(chars[i])
                i += 1
                continue
            
            if i + 1 < len(chars):
                pair = (chars[i].lower(), chars[i+1].lower())
                if pair in self.digraphs:
                    replacement = self.digraphs[pair]
                    start_idx = len(processed)
                    for rc in replacement:
                        orig_map.append(i)
                        processed.append(rc)
                    if pair in self.frozen_digraphs:
                        for j in range(start_idx, len(processed)):
                            frozen.add(j)
                    i += 2
                    continue
            
            if i in magic_vowels:
                magic_v_processed[len(processed)] = i
            if i in igh_vowels:
                magic_v_processed[len(processed)] = i
            
            orig_map.append(i)
            processed.append(chars[i])
            i += 1
        
        # Phase 2 + 3: Context + character lookups
        result = []
        for idx, ch in enumerate(processed):
            if idx in silent:
                continue
            
            if idx in frozen or ord(ch) > 127:
                result.append(ch)
                continue
            
            lc = ch.lower()
            
            # igh trigraph
            if idx in magic_v_processed:
                orig_idx = magic_v_processed[idx]
                if orig_idx in igh_vowels:
                    result.append('aɪ')
                    continue
            
            # Magic-e vowel
            if idx in magic_v_processed and (lc in self.magic_e_tables or lc in LONG_VOWELS):
                orig_idx = magic_v_processed[idx]
                ctx = magic_e_ctx.get(orig_idx, extract_context_at(processed, idx))
                result.append(self._lookup_magic_e(lc, ctx))
                continue
            
            # Context-dependent (dict lookup)
            if lc in self.context_tables:
                oi = orig_map[idx] if idx < len(orig_map) else idx
                ctx = extract_context_at(chars_lc, oi)
                result.append(self._lookup_context(lc, ctx))
                continue
            
            # Simple character map (dict lookup)
            if lc in self.char_map:
                result.append(self.char_map[lc])
                continue
            
            # Pass through unchanged
            result.append(lc)
        
        return ''.join(result)
    
    def stats(self):
        """Return statistics about the simplified model."""
        n_digraph = len(self.digraphs)
        n_frozen = len(self.frozen_digraphs)
        n_char = len(self.char_map)
        n_context = len(self.context_tables)
        n_magic_e = len(self.magic_e_tables)
        
        # Count total lookup entries
        n_entries = n_digraph + n_char
        for table in self.context_tables.values():
            if table['type'] == 'context':
                n_entries += len(table['selector_map'])
            elif table['type'] == 'geared':
                n_entries += len(table['pure_map'])
                for _, (fv, fm, _) in table['fine_gears'].items():
                    n_entries += len(fm)
        for table in self.magic_e_tables.values():
            if table['type'] == 'simple':
                n_entries += 1
            elif table['type'] == 'geared':
                n_entries += len(table['pure_map'])
                for _, (fv, fm, _) in table['fine_gears'].items():
                    n_entries += len(fm)
        
        return {
            'digraphs': n_digraph,
            'frozen': n_frozen,
            'char_map': n_char,
            'context_rules': n_context,
            'magic_e_rules': n_magic_e,
            'total_entries': n_entries,
        }
    
    def show(self):
        """Display the simplified model."""
        s = self.stats()
        lines = []
        lines.append("SimplifiedExecutor (pure lookup tables):")
        lines.append(f"  Phase 1: {s['digraphs']} digraphs ({s['frozen']} frozen)")
        for (c1, c2), repl in sorted(self.digraphs.items()):
            tag = 'F' if (c1, c2) in self.frozen_digraphs else 'D'
            lines.append(f"    [{tag}] {c1}{c2} → {repl if repl else '∅'}")
        
        lines.append(f"  Phase 2: {s['context_rules']} context rules")
        for char, table in sorted(self.context_tables.items()):
            if table['type'] == 'context':
                lines.append(f"    {char} → select on {table['selector_var']}: "
                           f"{dict(table['selector_map'])}")
            elif table['type'] == 'geared':
                lines.append(f"    {char} → geared on {table['coarse_var']}: "
                           f"{len(table['pure_map'])} pure + "
                           f"{len(table['fine_gears'])} fine")
        
        lines.append(f"  Phase 3: {s['char_map']} character substitutions")
        for ic, oc in sorted(self.char_map.items()):
            lines.append(f"    {ic} → {oc}")
        
        lines.append(f"  Magic-e: {s['magic_e_rules']} vowel rules")
        for vowel, table in sorted(self.magic_e_tables.items()):
            if table['type'] == 'simple':
                lines.append(f"    {vowel} → {table['output']}")
            elif table['type'] == 'geared':
                lines.append(f"    {vowel} → geared on {table['coarse_var']}: "
                           f"{len(table['pure_map'])} pure + "
                           f"{len(table['fine_gears'])} fine")
        
        lines.append(f"  Total lookup entries: {s['total_entries']}")
        return '\n'.join(lines)


# ============================================================================
# Part 6: Verification
# ============================================================================

TEST_WORDS = [
    'cat', 'bed', 'sit', 'hot', 'but', 'run', 'man', 'dog', 'pig', 'cup',
    'ship', 'thin', 'ring', 'chin', 'when', 'back', 'make', 'bite', 'code',
    'cute', 'these', 'come', 'love', 'have', 'give', 'gone', 'feet', 'moon',
    'rain', 'boat', 'city', 'cell', 'cold', 'game', 'gem', 'get', 'gift',
    'gym', 'yes', 'jumping', 'thinking', 'singing', 'fishing', 'the',
    'there', 'where', 'here', 'place', 'space', 'face', 'dance', 'chance',
    'prince', 'since', 'voice', 'choice', 'noise', 'light', 'night',
    'right', 'high', 'bright', 'sight', 'think', 'bank', 'drink',
    'world', 'work', 'word', 'store', 'more', 'shove', 'above', 'once',
    'large', 'bridge', 'edge',
]


# ============================================================================
# Main Analysis
# ============================================================================

def build_program():
    """Build the IPA program from lessons (same as demo)."""
    program = GeometricProgram()
    for lesson in LESSONS:
        if lesson['type'] == 'digraph':
            c1, c2 = lesson['chars']
            freeze = lesson.get('freeze', False)
            program.add_digraph(c1, c2, lesson['ipa'], freeze=freeze)
        elif lesson['type'] == 'char':
            rule = GeometricRule(lesson['input'], lesson['ipa'])
            examples = make_examples(ord(lesson['input']), ord(lesson['ipa']))
            rule.learn_from_examples(examples)
            program.add_rule(rule)
        elif lesson['type'] == 'context':
            rules = auto_build_rules(lesson['training'])
            for r in rules:
                if r.input_char == lesson['target_char']:
                    program.add_context_rule(r)
        elif lesson['type'] == 'magic_e_trained':
            program.magic_e_enabled = True
            program.magic_e_rules = learn_magic_e_rules(lesson['training'])
    return program


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " IPA MODEL SIMPLIFICATION: FINDING THE LCD OF SHAPES ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Build the model
    program = build_program()
    
    # Capture reference outputs
    reference = {word: program.apply_text(word) for word in TEST_WORDS}
    
    # ================================================================
    # Part 1: Atomic Decomposition
    # ================================================================
    print("=" * 70)
    print("  Part 1: ATOMIC DECOMPOSITION")
    print("  What is the 'least common denominator' for shapes?")
    print("=" * 70)
    print()
    
    all_rules = []
    
    # Char rules
    for rule in program.rules:
        all_rules.append(decompose_char_rule(rule))
    
    # Digraph rules
    for (c1, c2), replacement in program.digraph_rules.items():
        frozen = (c1, c2) in program.frozen_digraphs
        all_rules.append(decompose_digraph_rule(c1, c2, replacement, frozen))
    
    # Context rules
    for char, rule_obj in program.context_rules.items():
        all_rules.append(decompose_context_rule(char, rule_obj))
    
    # Magic-e rules
    if program.magic_e_rules:
        all_rules.extend(decompose_magic_e_rules(program.magic_e_rules))
    
    total_atoms = sum(len(r.atoms) for r in all_rules)
    total_compare = sum(r.comparisons for r in all_rules)
    total_logic = sum(r.logic_ops for r in all_rules)
    total_output = sum(r.outputs for r in all_rules)
    
    print(f"  The model decomposes into {total_atoms} atoms:")
    print(f"    COMPARE operations: {total_compare}")
    print(f"    AND/OR logic:       {total_logic}")
    print(f"    EMIT outputs:       {total_output}")
    print()
    
    print("  The ATOM is: COMPARE(variable, value) → {{0, 1}}")
    print("  Everything else is composition: AND, OR, EMIT.")
    print()
    print("  This is our geometric equivalent of AIG's AND+NOT basis.")
    print("  Where AIGs operate on BITS, our atoms operate on CHARACTERS.")
    print()
    
    # Show per-rule breakdown
    print("  Per-rule breakdown:")
    print(f"  {'Rule':<25} {'Type':<18} {'Atoms':>6} {'Cmp':>5} {'Logic':>6} {'Out':>5}")
    print("  " + "-" * 65)
    for rule in all_rules:
        print(f"  {rule.name:<25} {rule.rule_type:<18} "
              f"{len(rule.atoms):>6} {rule.comparisons:>5} "
              f"{rule.logic_ops:>6} {rule.outputs:>5}")
    print("  " + "-" * 65)
    print(f"  {'TOTAL':<25} {'':>18} {total_atoms:>6} {total_compare:>5} "
          f"{total_logic:>6} {total_output:>5}")
    print()
    
    # ================================================================
    # Part 2: Sharing Analysis (Structural Hashing)
    # ================================================================
    print("=" * 70)
    print("  Part 2: SHARING ANALYSIS (Structural Hashing)")
    print("  Which comparisons appear in multiple rules?")
    print("=" * 70)
    print()
    
    shared = find_shared_comparisons(all_rules)
    
    all_sigs = set()
    for rule in all_rules:
        for atom in rule.atoms:
            if atom.op == 'compare':
                all_sigs.add(atom.signature())
    
    print(f"  Total unique comparisons: {len(all_sigs)}")
    print(f"  Shared across rules:      {len(shared)}")
    print(f"  Rule-private:             {len(all_sigs) - len(shared)}")
    print()
    
    if shared:
        print("  Shared comparisons (like AIG structural hashing):")
        for sig, users in sorted(shared.items(), key=lambda x: -len(x[1])):
            op, var, val = sig
            user_str = ", ".join(users[:4])
            if len(users) > 4:
                user_str += f", ... ({len(users)} total)"
            print(f"    {var}=={val!r:>6} → used by: {user_str}")
        print()
        
        # Savings from sharing
        total_raw_compares = sum(r.comparisons for r in all_rules)
        shared_savings = sum(len(users) - 1 for users in shared.values())
        print(f"  Raw comparisons:  {total_raw_compares}")
        print(f"  After sharing:    {total_raw_compares - shared_savings}")
        print(f"  Savings:          {shared_savings} ({100*shared_savings/total_raw_compares:.0f}%)")
    print()
    
    # ================================================================
    # Part 3: Template Vocabulary (Mesh Face Types)
    # ================================================================
    print("=" * 70)
    print("  Part 3: TEMPLATE VOCABULARY (Mesh Face Types)")
    print("  How many distinct SHAPES does the model use?")
    print("=" * 70)
    print()
    
    templates = identify_templates(all_rules)
    
    print(f"  The model uses {len(templates)} shape templates:")
    print()
    for t in templates:
        print(f"    [{t.name}]")
        print(f"      {t.description}")
        print(f"      Parameters: {', '.join(t.parameter_schema)}")
        print(f"      Atoms per instance: {t.atoms_per_instance:.1f}")
        print()
    
    total_instances = sum(t.instances for t in templates)
    print(f"  Total: {len(templates)} templates × {total_instances} instances = {total_atoms} atoms")
    print()
    print("  INSIGHT: The 'mesh' has only a few FACE TYPES (templates).")
    print("  Simplification means: reducing to fewer face types,")
    print("  or fewer instances of existing types.")
    print()
    
    # ================================================================
    # Part 4: Information Content
    # ================================================================
    print("=" * 70)
    print("  Part 4: INFORMATION CONTENT")
    print("  How many bits does the model actually contain?")
    print("=" * 70)
    print()
    
    info = compute_information_content(program, all_rules)
    
    print(f"  Component breakdown:")
    for key, bits in info.items():
        if key in ('total', 'total_bytes'):
            continue
        if bits > 0:
            print(f"    {key:<20} {bits:>5} bits ({bits/8:.1f} bytes)")
    print(f"    {'':─<20} {'':─>5}")
    print(f"    {'TOTAL':<20} {info['total']:>5} bits ({info['total_bytes']} bytes)")
    print()
    
    # Compare to representation sizes
    gate_step_cost = 159  # each gate_step ≈ 3 floats (t, s, h) = 96 bits
    gate_step_bits = gate_step_cost * 96
    
    print(f"  Representation comparison:")
    print(f"    gate_step encoding: {gate_step_bits:>6} bits ({gate_step_bits//8} bytes)")
    print(f"    Information content: {info['total']:>6} bits ({info['total_bytes']} bytes)")
    print(f"    Compression ratio:  {gate_step_bits / info['total']:.1f}×")
    print()
    print(f"  The model's INFORMATION is {info['total_bytes']} bytes.")
    print(f"  The gate_step ENCODING uses {gate_step_bits//8} bytes.")
    print(f"  The encoding is {gate_step_bits / info['total']:.1f}× larger than the information.")
    print()
    
    # ================================================================
    # Part 5: Simplified Executor (The Actual Mesh Simplification)
    # ================================================================
    print("=" * 70)
    print("  Part 5: SIMPLIFIED EXECUTOR")
    print("  Build → Simplify → Verify (lossless mesh simplification)")
    print("=" * 70)
    print()
    
    # Build the simplified model
    t0 = time.perf_counter()
    executor = SimplifiedExecutor.simplify(program)
    simplify_time = time.perf_counter() - t0
    
    st = executor.stats()
    print(f"  Simplification took {simplify_time*1000:.2f}ms")
    print()
    print(executor.show())
    print()
    
    # Verify: identical outputs on ALL test words
    print("  VERIFICATION: Original vs Simplified")
    print("  " + "-" * 50)
    
    n_pass = 0
    n_fail = 0
    fails = []
    
    # Time both models
    t0 = time.perf_counter()
    for word in TEST_WORDS:
        _ = program.apply_text(word)
    original_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for word in TEST_WORDS:
        _ = executor.apply_text(word)
    simplified_time = time.perf_counter() - t0
    
    for word in TEST_WORDS:
        orig = reference[word]
        simp = executor.apply_text(word)
        if orig == simp:
            n_pass += 1
        else:
            n_fail += 1
            fails.append((word, orig, simp))
    
    total = n_pass + n_fail
    print(f"  Passed: {n_pass}/{total} ({100*n_pass/total:.0f}%)")
    
    if fails:
        print(f"  FAILURES ({n_fail}):")
        for word, orig, simp in fails[:10]:
            print(f"    {word:<15} original: {orig}")
            print(f"    {'':15} simplified: {simp}")
    else:
        print(f"  ✓ LOSSLESS: Every word produces identical output.")
    print()
    
    print(f"  Timing ({len(TEST_WORDS)} words):")
    print(f"    Original (gate_step):  {original_time*1000:.2f}ms")
    print(f"    Simplified (lookup):   {simplified_time*1000:.2f}ms")
    if original_time > 0:
        print(f"    Speedup:               {original_time/simplified_time:.1f}×")
    print()
    
    # Show some examples
    showcase = [
        "The bright light shone right there in the night.",
        "Some love to dance but none have a choice in the voice.",
        "I think the prince sat on the fence and drank his drink.",
    ]
    print("  Showcase (from simplified executor):")
    for text in showcase:
        result = executor.apply_text(text)
        print(f"    EN:  {text}")
        print(f"    IPA: {result}")
        print()
    
    # ================================================================
    # Part 6: The Universal Basis
    # ================================================================
    print("=" * 70)
    print("  Part 6: THE UNIVERSAL BASIS")
    print("  What is the 'AND gate' of geometric computation?")
    print("=" * 70)
    print()
    
    print("  AIG:  Any Boolean circuit → AND + NOT")
    print("  Ours: Any character transform → COMPARE + AND + EMIT")
    print()
    print("  The comparison:")
    print("  ┌─────────────────┬──────────────────────────────┐")
    print("  │ AIG              │ Geometric Model              │")
    print("  ├─────────────────┼──────────────────────────────┤")
    print("  │ Input: bits      │ Input: characters             │")
    print("  │ AND(a,b)         │ COMPARE(var, val)             │")
    print("  │ NOT(a)           │ AND(cmp1, cmp2)               │")
    print("  │ Output: bits     │ EMIT(ipa_string)              │")
    print("  │ Structural hash  │ Shared comparisons            │")
    print("  │ Rewriting rules  │ Template factorization        │")
    print("  │ Level reduction  │ Phase pipeline (0→1→2→3)      │")
    print("  └─────────────────┴──────────────────────────────┘")
    print()
    print("  Key insight: COMPARE is our UNIVERSAL GATE.")
    print()
    print(f"  The entire IPA model reduces to:")
    print(f"    {total_compare} comparisons")
    print(f"    + {total_logic} AND/OR gates")
    print(f"    + {total_output} output emitters")
    print(f"    = {total_atoms} total atoms")
    print()
    print(f"  Encoded in {info['total_bytes']} bytes of parameters.")
    print(f"  Executing in 4 pipeline phases.")
    print(f"  Zero floating-point arithmetic needed.")
    print()
    
    # ================================================================
    # Summary
    # ================================================================
    print("=" * 70)
    print("  SUMMARY: Three Levels of Simplification")
    print("=" * 70)
    print()
    print("  Level 0 (Original):   159 gate_step calls")
    print(f"                         {gate_step_bits//8} bytes (float params)")
    print(f"                         Continuous arithmetic")
    print()
    print(f"  Level 1 (Factored):   {len(templates)} templates × {total_instances} instances")
    print(f"                         {info['total_bytes']} bytes (discrete params)")
    print(f"                         Same behavior, structured")
    print()
    print(f"  Level 2 (Atomic):     {total_compare} COMPARE + {total_logic} AND + {total_output} EMIT")
    raw = total_raw_compares if 'total_raw_compares' in dir() else total_compare
    after_share = raw - shared_savings if shared else raw
    print(f"                         After sharing: {after_share} unique comparisons")
    print(f"                         Pure discrete logic")
    print()
    print(f"  Level 3 (Minimal):    {info['total_bytes']} bytes")
    print(f"                         Just the lookup tables")
    print(f"                         No computation — structure IS the answer")
    print()
    print("  The LCD of shapes is the COMPARISON.")
    print("  Every rule, every gear, every magic-e exception")
    print("  is just comparisons composed with AND and EMIT.")
    print()
    print("  The model doesn't COMPUTE the answer.")
    print("  The model IS the answer — a structured table")
    print(f"  that fits in {info['total_bytes']} bytes.")
    print()


if __name__ == "__main__":
    main()
