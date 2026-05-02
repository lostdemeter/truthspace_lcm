#!/usr/bin/env python3
"""
Auto-Detection of Context Dependence

Given training data as (input_word, output_word) pairs, automatically:

1. DETECT which input characters have inconsistent mappings
   (same input char -> different output chars in different contexts)

2. DISCOVER which context variable (prev_char, next_char, position, etc.)
   explains the inconsistency — i.e., find the selector

3. BUILD shader channels: one channel per distinct output, with a
   geometric selector that picks the right channel based on context

The algorithm:
  For each character position in each training pair:
    - Record (input_char, output_char, context) tuples
    - Group by input_char
    - If all outputs are the same → simple RECT rule
    - If outputs differ → context-dependent rule
      - For each candidate context variable, compute information gain
      - The variable with highest gain IS the selector
      - The values that predict each output class ARE the selector RECTs

No gradient descent. No neural network. Just geometric structure detection.
"""

import numpy as np
from collections import defaultdict
import time


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

def extract_context_at(chars, idx):
    """Extract context dict for a single position in a character sequence.
    
    Args:
        chars: list or string of characters
        idx: position index
    
    Returns:
        context dict with prev_char, next_char, next_next_char, etc.
    """
    n = len(chars)
    next_ch = chars[idx+1] if idx+1 < n else ' '
    next_next_ch = chars[idx+2] if idx+2 < n else ' '
    prev_ch = chars[idx-1] if idx > 0 else ' '
    prev_prev_ch = chars[idx-2] if idx > 1 else ' '
    return {
        'prev_char': prev_ch,
        'next_char': next_ch,
        'next_next_char': next_next_ch,
        'prev_prev_char': prev_prev_ch,
        'next_bigram': next_ch + next_next_ch,
        'position': idx,
        'word_len': n,
        'is_start': idx == 0 or prev_ch == ' ',
        'is_end': idx == n - 1 or next_ch == ' ',
    }


def extract_contexts(input_word, output_word):
    """Extract (input_char, output_char, context_dict) for each position.
    
    Context includes:
      - prev_char: character before current position (or ' ' at start)
      - next_char: character after current position (or ' ' at end)
      - next_next_char: character two positions ahead (or ' ')
      - prev_prev_char: character two positions before (or ' ')
      - next_bigram: next two characters as a string
      - position: index in word
      - word_len: length of word
      - is_start: position == 0
      - is_end: position == len-1
    """
    entries = []
    for i, (ic, oc) in enumerate(zip(input_word, output_word)):
        next_ch = input_word[i+1] if i+1 < len(input_word) else ' '
        next_next_ch = input_word[i+2] if i+2 < len(input_word) else ' '
        prev_ch = input_word[i-1] if i > 0 else ' '
        prev_prev_ch = input_word[i-2] if i > 1 else ' '
        ctx = {
            'prev_char': prev_ch,
            'next_char': next_ch,
            'next_next_char': next_next_ch,
            'prev_prev_char': prev_prev_ch,
            'next_bigram': next_ch + next_next_ch,
            'position': i,
            'word_len': len(input_word),
            'is_start': i == 0,
            'is_end': i == len(input_word) - 1,
        }
        entries.append((ic, oc, ctx))
    return entries


# ============================================================================
# INCONSISTENCY DETECTION
# ============================================================================

def detect_inconsistencies(training_pairs):
    """Find characters with multiple output mappings.
    
    Returns:
      consistent: {input_char: output_char} for 1:1 mappings
      inconsistent: {input_char: [(output_char, context_dict), ...]}
    """
    # Collect all (input_char, output_char, context) observations
    observations = defaultdict(list)
    
    for input_word, output_word in training_pairs:
        # Align characters — for now, require same length
        if len(input_word) != len(output_word):
            continue
        entries = extract_contexts(input_word, output_word)
        for ic, oc, ctx in entries:
            observations[ic].append((oc, ctx))
    
    consistent = {}
    inconsistent = {}
    identity = {}
    
    for ic, obs_list in observations.items():
        output_chars = set(oc for oc, _ in obs_list)
        if len(output_chars) == 1:
            oc = output_chars.pop()
            if oc == ic:
                identity[ic] = oc
            else:
                consistent[ic] = oc
        else:
            inconsistent[ic] = obs_list
    
    return identity, consistent, inconsistent


# ============================================================================
# SELECTOR DISCOVERY
# ============================================================================

def discover_selector(input_char, observations):
    """Given inconsistent observations for one input_char, find the
    context variable that best explains the output variation.
    
    Uses information gain (entropy reduction) to rank candidate
    context variables.
    
    Returns:
      best_variable: name of the context variable (e.g., 'next_char')
      selector_map: {context_value: output_char} mapping
      channels: {output_char: [context_values]} reverse mapping
      gain: information gain of the best variable
    """
    # Group observations by output character
    output_groups = defaultdict(list)
    for oc, ctx in observations:
        output_groups[oc].append(ctx)
    
    n_total = len(observations)
    n_outputs = len(output_groups)
    
    if n_outputs < 2:
        return None, None, None, 0.0
    
    # Compute base entropy
    base_entropy = 0.0
    for oc, contexts in output_groups.items():
        p = len(contexts) / n_total
        if p > 0:
            base_entropy -= p * np.log2(p)
    
    # Test each context variable
    candidate_vars = ['prev_char', 'next_char', 'next_next_char',
                      'prev_prev_char', 'next_bigram', 'is_start', 'is_end']
    
    results = []
    
    for var_name in candidate_vars:
        # Group by context value
        value_groups = defaultdict(lambda: defaultdict(int))
        for oc, ctx in observations:
            val = ctx[var_name]
            value_groups[val][oc] += 1
        
        # Compute conditional entropy H(output | var)
        cond_entropy = 0.0
        for val, output_counts in value_groups.items():
            n_val = sum(output_counts.values())
            p_val = n_val / n_total
            val_entropy = 0.0
            for oc, count in output_counts.items():
                p_oc = count / n_val
                if p_oc > 0:
                    val_entropy -= p_oc * np.log2(p_oc)
            cond_entropy += p_val * val_entropy
        
        gain = base_entropy - cond_entropy
        
        # Build selector map: for each context value, what's the majority output?
        selector_map = {}
        purity = 0
        for val, output_counts in value_groups.items():
            best_oc = max(output_counts, key=output_counts.get)
            selector_map[val] = best_oc
            purity += output_counts[best_oc]
        
        accuracy = purity / n_total
        
        results.append((gain, accuracy, var_name, selector_map))
    
    # Pick best variable by gain, break ties by accuracy
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_gain, best_acc, best_var, best_map = results[0]
    
    # Build channels: reverse map from output_char -> set of context values
    channels = defaultdict(set)
    for val, oc in best_map.items():
        channels[oc].add(val)
    
    return best_var, best_map, dict(channels), best_gain


def discover_gears(input_char, observations):
    """Find a gear train for context-dependent rules.
    
    Gear 1 (coarse): the simplest selector that resolves the most cases.
             Prefer high resolved-per-tooth ratio (coverage / cardinality).
    
    Gear 2 (fine): for each ambiguous tooth in Gear 1, find a secondary
             selector that resolves within that subset. Only engages when
             the fallthrough register has data.
    
    Returns:
      coarse_var: name of the coarse gear variable
      pure_map: {coarse_value: output_char} for resolved teeth
      fine_gears: {coarse_value: (fine_var, fine_map, fine_channels, fine_gain)}
      default_output: majority output across all observations
      stats: dict with diagnostic info
    """
    # Only consider single-char context vars for coarse gear
    # (bigrams are fine-gear territory — they have higher cardinality)
    coarse_candidates = ['prev_char', 'next_char', 'is_start', 'is_end']
    all_candidates = ['prev_char', 'next_char', 'next_next_char',
                      'prev_prev_char', 'next_bigram', 'is_start', 'is_end']
    
    n_total = len(observations)
    
    # Find overall default
    output_counts = defaultdict(int)
    for oc, _ in observations:
        output_counts[oc] += 1
    default_output = max(output_counts, key=output_counts.get)
    
    best_coarse = None
    best_score = -1
    
    for var_name in coarse_candidates:
        by_val = defaultdict(list)
        for oc, ctx in observations:
            by_val[ctx[var_name]].append((oc, ctx))
        
        n_pure = 0
        n_resolved = 0
        n_ambiguous = 0
        pure_map = {}
        ambiguous_vals = {}
        
        for val, obs_group in by_val.items():
            outputs = set(oc for oc, _ in obs_group)
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
        # Maximize: how many cases each tooth handles on average
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
        # Compute the ambiguous zone's default (majority within this subset)
        zone_counts = defaultdict(int)
        for oc, _ in obs_group:
            zone_counts[oc] += 1
        zone_default = max(zone_counts, key=zone_counts.get)
        
        # Try ALL candidate variables (including bigrams) for fine gear
        best_var2, best_map2, channels2, gain2 = _discover_selector_from(
            obs_group, all_candidates
        )
        if best_var2 and gain2 > 0.0:
            fine_gears[val] = (best_var2, best_map2, channels2, gain2,
                               zone_default)
            # Count how many the fine gear resolves
            for oc, ctx in obs_group:
                if best_map2.get(ctx[best_var2]) == oc:
                    total_fine_resolved += 1
        else:
            # No fine gear found, but store the zone default
            fine_gears[val] = (None, {}, {}, 0.0, zone_default)
    
    stats = {
        'coarse_resolved': best_coarse['n_resolved'],
        'coarse_teeth': best_coarse['n_teeth'],
        'coarse_pure': best_coarse['n_pure'],
        'coarse_ambiguous': best_coarse['n_ambiguous'],
        'fine_gears': len(fine_gears),
        'fine_resolved': total_fine_resolved,
        'total': n_total,
    }
    
    return (best_coarse['var'], best_coarse['pure_map'],
            fine_gears, default_output, stats)


def _discover_selector_from(observations, candidate_vars):
    """discover_selector but with an explicit candidate list."""
    n_total = len(observations)
    output_groups = defaultdict(list)
    for oc, ctx in observations:
        output_groups[oc].append(ctx)
    
    if len(output_groups) < 2:
        return None, None, None, 0.0
    
    base_entropy = 0.0
    for oc, contexts in output_groups.items():
        p = len(contexts) / n_total
        if p > 0:
            base_entropy -= p * np.log2(p)
    
    results = []
    for var_name in candidate_vars:
        value_groups = defaultdict(lambda: defaultdict(int))
        for oc, ctx in observations:
            value_groups[ctx[var_name]][oc] += 1
        
        cond_entropy = 0.0
        for val, oc_counts in value_groups.items():
            n_val = sum(oc_counts.values())
            p_val = n_val / n_total
            val_ent = 0.0
            for count in oc_counts.values():
                p_oc = count / n_val
                if p_oc > 0:
                    val_ent -= p_oc * np.log2(p_oc)
            cond_entropy += p_val * val_ent
        
        gain = base_entropy - cond_entropy
        
        selector_map = {}
        purity = 0
        for val, oc_counts in value_groups.items():
            best_oc = max(oc_counts, key=oc_counts.get)
            selector_map[val] = best_oc
            purity += oc_counts[best_oc]
        
        results.append((gain, purity / n_total, var_name, selector_map))
    
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_gain, _, best_var, best_map = results[0]
    
    channels = defaultdict(set)
    for val, oc in best_map.items():
        channels[oc].add(val)
    
    return best_var, best_map, dict(channels), best_gain


# ============================================================================
# RULE BUILDER
# ============================================================================

class GeometricRule:
    """A learned character transformation rule."""
    
    def __init__(self, input_char, rule_type, **kwargs):
        self.input_char = input_char
        self.rule_type = rule_type  # 'identity', 'simple', 'context'
        self.params = kwargs
    
    def apply(self, current_char, context=None):
        if current_char != self.input_char:
            return current_char
        
        if self.rule_type == 'identity':
            return current_char
        
        elif self.rule_type == 'simple':
            return self.params['output_char']
        
        elif self.rule_type == 'context':
            if context is None:
                return self.params.get('default_output', current_char)
            var_name = self.params['selector_variable']
            ctx_val = context.get(var_name)
            selector_map = self.params['selector_map']
            return selector_map.get(ctx_val, self.params.get('default_output', current_char))
        
        elif self.rule_type == 'geared':
            if context is None:
                return self.params.get('default_output', current_char)
            
            coarse_var = self.params['coarse_var']
            coarse_val = context.get(coarse_var)
            pure_map = self.params['pure_map']
            
            # Gear 1: coarse selector
            if coarse_val in pure_map:
                return pure_map[coarse_val]
            
            # Fallthrough register has data — engage fine gear
            fine_gears = self.params.get('fine_gears', {})
            if coarse_val in fine_gears:
                fine_var, fine_map, _, _, zone_default = fine_gears[coarse_val]
                if fine_var is not None:
                    fine_val = context.get(fine_var)
                    if fine_val in fine_map:
                        return fine_map[fine_val]
                # Fine gear didn't resolve — use ZONE default
                # (majority within the ambiguous subset, not global)
                return zone_default
            
            # No gear at all — use global default
            return self.params.get('default_output', current_char)
        
        return current_char
    
    def describe(self):
        if self.rule_type == 'identity':
            return f"  '{self.input_char}' → '{self.input_char}'  (identity)"
        
        elif self.rule_type == 'simple':
            oc = self.params['output_char']
            h = ord(oc) - ord(self.input_char)
            return (f"  '{self.input_char}' → '{oc}'  "
                    f"RECT[{ord(self.input_char)-0.5}, {ord(self.input_char)+0.5}] "
                    f"h={h:+d}")
        
        elif self.rule_type == 'context':
            var = self.params['selector_variable']
            channels = self.params['channels']
            gain = self.params.get('info_gain', 0)
            lines = [f"  '{self.input_char}' → context-dependent on {var} "
                    f"(gain={gain:.3f}):"]
            for oc, ctx_vals in channels.items():
                h = ord(oc) - ord(self.input_char)
                vals_str = ', '.join(repr(v) for v in sorted(ctx_vals, key=str))
                lines.append(f"    Channel '{oc}' (h={h:+d}): "
                           f"when {var} ∈ {{{vals_str}}}")
            lines.append(f"    Default: '{self.params.get('default_output', self.input_char)}'")
            lines.append(f"    Geometric: RECT(x,{ord(self.input_char)}) × "
                        f"SELECTOR({var})")
            return '\n'.join(lines)
        
        elif self.rule_type == 'geared':
            coarse_var = self.params['coarse_var']
            pure_map = self.params['pure_map']
            fine_gears = self.params.get('fine_gears', {})
            stats = self.params.get('stats', {})
            
            lines = [f"  '{self.input_char}' → GEARED on {coarse_var}:"]
            
            # Gear 1: resolved teeth
            resolved_by_output = defaultdict(list)
            for val, oc in sorted(pure_map.items(), key=str):
                resolved_by_output[oc].append(val)
            
            lines.append(f"    Gear 1 (coarse): {coarse_var} "
                        f"[{stats.get('coarse_pure', '?')} pure teeth, "
                        f"{stats.get('coarse_ambiguous', '?')} ambiguous]")
            for oc, vals in sorted(resolved_by_output.items()):
                h = ord(oc) - ord(self.input_char)
                vals_str = ', '.join(repr(v) for v in sorted(vals, key=str))
                lines.append(f"      → '{oc}' (h={h:+d}): "
                           f"when {coarse_var} ∈ {{{vals_str}}}")
            
            # Gear 2: fine gears for ambiguous teeth
            if fine_gears:
                active_fine = {k: v for k, v in fine_gears.items()
                              if v[0] is not None}
                lines.append(f"    Gear 2 (fine): {len(active_fine)} "
                           f"fallthrough register(s):")
                for coarse_val, (fine_var, fine_map, fine_ch, fine_gain,
                                 zone_default) \
                        in sorted(fine_gears.items(), key=str):
                    if fine_var is None:
                        lines.append(f"      When {coarse_var}='{coarse_val}' "
                                   f"→ zone default '{zone_default}'")
                        continue
                    lines.append(f"      When {coarse_var}='{coarse_val}' "
                               f"→ engage {fine_var} "
                               f"(zone default='{zone_default}'):")
                    for fine_val, oc in sorted(fine_map.items(), key=str):
                        lines.append(f"        {fine_var}='{fine_val}' → '{oc}'")
            
            lines.append(f"    Default: '{self.params.get('default_output', self.input_char)}'")
            n_active = sum(1 for v in fine_gears.values() if v[0] is not None)
            lines.append(f"    Geometric: RECT(x,{ord(self.input_char)}) × "
                        f"SELECTOR({coarse_var}) "
                        f"+ {n_active} fine RECT × SELECTOR")
            return '\n'.join(lines)


def build_rules(training_pairs):
    """Automatically build geometric rules from training pairs.
    
    Returns list of GeometricRule objects.
    """
    identity, consistent, inconsistent = detect_inconsistencies(training_pairs)
    
    rules = []
    
    # Identity rules (no-ops, but tracked for completeness)
    for ic in sorted(identity):
        rules.append(GeometricRule(ic, 'identity'))
    
    # Simple rules (consistent 1:1 mappings)
    for ic, oc in sorted(consistent.items()):
        rules.append(GeometricRule(ic, 'simple', output_char=oc))
    
    # Context-dependent rules — use gear discovery
    for ic, obs in sorted(inconsistent.items()):
        coarse_var, pure_map, fine_gears, default, stats = \
            discover_gears(ic, obs)
        
        if coarse_var is None:
            # No coarse gear found — majority vote
            rules.append(GeometricRule(ic, 'simple', output_char=default))
        elif not fine_gears:
            # Coarse gear resolves everything — simple context rule
            # (no fine gear needed, all teeth are pure)
            channels = defaultdict(set)
            for val, oc in pure_map.items():
                channels[oc].add(val)
            rules.append(GeometricRule(ic, 'context',
                                      selector_variable=coarse_var,
                                      selector_map=pure_map,
                                      channels=dict(channels),
                                      default_output=default,
                                      info_gain=stats.get('coarse_resolved', 0) /
                                                stats.get('total', 1)))
        else:
            # Gear train: coarse + fine
            rules.append(GeometricRule(ic, 'geared',
                                      coarse_var=coarse_var,
                                      pure_map=pure_map,
                                      fine_gears=fine_gears,
                                      default_output=default,
                                      stats=stats))
    
    return rules


def base_entropy_for(observations):
    """Compute base entropy of output distribution."""
    counts = defaultdict(int)
    for oc, _ in observations:
        counts[oc] += 1
    n = len(observations)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


# ============================================================================
# GEOMETRIC PROGRAM (from auto-detected rules)
# ============================================================================

class AutoGeometricProgram:
    """A program built from auto-detected rules."""
    
    def __init__(self, rules):
        self.rules = {r.input_char: r for r in rules}
        self.context_rules = [r for r in rules if r.rule_type == 'context']
        self.simple_rules = [r for r in rules if r.rule_type == 'simple']
    
    def apply_word(self, word):
        result = []
        for i, ch in enumerate(word):
            next_ch = word[i+1] if i+1 < len(word) else ' '
            next_next_ch = word[i+2] if i+2 < len(word) else ' '
            prev_ch = word[i-1] if i > 0 else ' '
            prev_prev_ch = word[i-2] if i > 1 else ' '
            ctx = {
                'prev_char': prev_ch,
                'next_char': next_ch,
                'next_next_char': next_next_ch,
                'prev_prev_char': prev_prev_ch,
                'next_bigram': next_ch + next_next_ch,
                'is_start': i == 0,
                'is_end': i == len(word) - 1,
            }
            rule = self.rules.get(ch)
            if rule:
                result.append(rule.apply(ch, ctx))
            else:
                result.append(ch)
        return ''.join(result)
    
    def describe(self):
        lines = []
        n_identity = sum(1 for r in self.rules.values() if r.rule_type == 'identity')
        n_simple = len(self.simple_rules)
        n_context = len(self.context_rules)
        
        lines.append(f"Auto-Geometric Program:")
        lines.append(f"  {n_identity} identity rules (no-op)")
        lines.append(f"  {n_simple} simple RECT rules")
        lines.append(f"  {n_context} context-dependent rules (shader channels)")
        lines.append("")
        
        if self.simple_rules:
            lines.append("Simple rules:")
            for r in self.simple_rules:
                lines.append(r.describe())
            lines.append("")
        
        if self.context_rules:
            lines.append("Context-dependent rules:")
            for r in self.context_rules:
                lines.append(r.describe())
            lines.append("")
        
        geared_in_rules = [r for r in self.rules.values()
                           if r.rule_type == 'geared']
        if geared_in_rules:
            lines.append("Geared rules:")
            for r in geared_in_rules:
                lines.append(r.describe())
            lines.append("")
        
        geared_rules = [r for r in self.rules.values() if r.rule_type == 'geared']
        n_geared = len(geared_rules)
        
        if geared_rules:
            lines.append(f"  {n_geared} geared rules (coarse + fine selectors)")
        
        # Gate count
        n_gates = n_simple * 2
        n_mux = n_context
        for r in self.context_rules:
            n_channels = len(r.params.get('channels', {}))
            n_selector_vals = sum(len(v) for v in r.params.get('channels', {}).values())
            n_gates += n_channels * 2
            n_gates += n_selector_vals * 2
        for r in geared_rules:
            n_pure = len(r.params.get('pure_map', {}))
            n_gates += n_pure * 2  # coarse gear RECTs
            n_mux += 1
            for _, fg in r.params.get('fine_gears', {}).items():
                if fg[0] is not None:  # has a fine variable
                    n_gates += len(fg[1]) * 2  # fine gear RECTs
                    n_mux += 1  # fine MUX
        
        lines.append(f"Geometric cost: {n_gates} gate_step + {n_mux} MUX")
        
        return '\n'.join(lines)


# ============================================================================
# TEST CASES
# ============================================================================

def test_c_rule():
    """Test auto-detection of the English 'c' rule."""
    print("=" * 60)
    print("  TEST 1: English 'c' rule (soft/hard c)")
    print("  Can we discover next_char as the selector?")
    print("=" * 60)
    
    # Training data: (input, output) word pairs
    # We use phonetic spellings as output
    training = [
        ("cat",    "kat"),
        ("city",   "sity"),
        ("cup",    "kup"),
        ("cent",   "sent"),
        ("code",   "kode"),
        ("clap",   "klap"),
        ("acid",   "asid"),
        ("cry",    "kry"),
        ("cell",   "sell"),
        ("clay",   "klay"),
        ("cite",   "site"),
        ("cold",   "kold"),
        ("ace",    "ase"),
        ("curl",   "kurl"),
    ]
    
    print(f"\n  Training pairs: {len(training)}")
    for inp, out in training[:5]:
        print(f"    {inp} → {out}")
    print(f"    ... ({len(training)-5} more)")
    
    t0 = time.perf_counter()
    identity, consistent, inconsistent = detect_inconsistencies(training)
    t_detect = time.perf_counter() - t0
    
    print(f"\n  Detection ({t_detect*1000:.2f}ms):")
    print(f"    Identity chars: {sorted(identity.keys())}")
    print(f"    Simple rules: {dict(sorted(consistent.items()))}")
    print(f"    Inconsistent chars: {sorted(inconsistent.keys())}")
    
    # Auto-discover selector for inconsistent chars
    for ic, obs in sorted(inconsistent.items()):
        print(f"\n  Analyzing '{ic}' ({len(obs)} observations):")
        outputs = defaultdict(int)
        for oc, _ in obs:
            outputs[oc] += 1
        print(f"    Outputs: {dict(outputs)}")
        
        best_var, selector_map, channels, gain = discover_selector(ic, obs)
        print(f"    Best selector: {best_var} (info gain = {gain:.3f} bits)")
        if channels:
            for oc, vals in sorted(channels.items()):
                print(f"      '{ic}' → '{oc}' when {best_var} ∈ {sorted(vals, key=str)}")
    
    # Build and test program
    rules = build_rules(training)
    program = AutoGeometricProgram(rules)
    
    print(f"\n{program.describe()}")
    
    # Test on training data
    print("\n  Training accuracy:")
    correct = 0
    for inp, expected in training:
        result = program.apply_word(inp)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"    {inp:8s} → {result:8s}  (expected {expected:8s}) {match}")
    print(f"  Accuracy: {correct}/{len(training)}")
    
    # Test on unseen words
    print("\n  Generalization (unseen words):")
    test_words = [
        ("cake",  "kake"),   # hard c before a
        ("mice",  "mise"),   # soft c before e
        ("click", "klick"),  # hard c before l
        ("cinch", "sinch"),  # soft c before i
        ("occur", "okkur"),  # hard c before c (unseen context!)
    ]
    correct = 0
    for inp, expected in test_words:
        result = program.apply_word(inp)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"    {inp:8s} → {result:8s}  (expected {expected:8s}) {match}")
    print(f"  Accuracy: {correct}/{len(test_words)}")
    
    return program


def test_g_rule():
    """Test auto-detection of the English 'g' rule with gear shift."""
    print("\n" + "=" * 60)
    print("  TEST 2: English 'g' rule (soft/hard g)")
    print("  Gear shift: coarse gear on next_char,")
    print("  fine gear engages on ambiguous 'i' tooth")
    print("=" * 60)
    
    training = [
        ("game",    "game"),    # hard g before a
        ("gem",     "jem"),     # soft g before e
        ("gift",    "gift"),    # hard g before i (exception!)
        ("gin",     "jin"),     # soft g before i
        ("go",      "go"),      # hard g before o
        ("giant",   "jiant"),   # soft g before i
        ("gust",    "gust"),    # hard g before u
        ("gel",     "jel"),     # soft g before e
        ("glad",    "glad"),    # hard g before l
        ("gym",     "jym"),     # soft g before y
        ("gist",    "jist"),    # soft g before i
        ("girl",    "girl"),    # hard g before i (exception!)
        ("gig",     "jig"),     # soft g before i (first g)
    ]
    
    print(f"\n  Training pairs: {len(training)}")
    for inp, out in training:
        flag = ""
        if len(inp) > 0 and inp[0] == 'g' and len(inp) > 1 and inp[1] == 'i':
            flag = "  ← g before i"
        print(f"    {inp:8s} → {out:8s}{flag}")
    
    t0 = time.perf_counter()
    rules = build_rules(training)
    t_build = time.perf_counter() - t0
    program = AutoGeometricProgram(rules)
    
    print(f"\n  Built in {t_build*1000:.2f}ms")
    print(f"\n{program.describe()}")
    
    # Training accuracy
    print("\n  Training accuracy:")
    correct = 0
    for inp, expected in training:
        result = program.apply_word(inp)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        else:
            print(f"    {inp:8s} → {result:8s}  (expected {expected:8s}) {match}")
    print(f"  {correct}/{len(training)}")
    
    # Generalization
    print("\n  Generalization (unseen words):")
    test_words = [
        ("gate",    "gate"),    # hard g before a (coarse gear)
        ("gene",    "jene"),    # soft g before e (coarse gear)
        ("gulp",    "gulp"),    # hard g before u (coarse gear)
        ("gild",    "jild"),    # soft g before i (fine gear: 'l' after i)
        ("give",    "give"),    # hard g before i (fine gear: 'v' after i)
    ]
    correct = 0
    for inp, expected in test_words:
        result = program.apply_word(inp)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"    {inp:8s} → {result:8s}  (expected {expected:8s}) {match}")
    print(f"  Accuracy: {correct}/{len(test_words)}")
    
    return program


def test_combined():
    """Test auto-detection with mixed c/g rules + simple substitutions."""
    print("\n" + "=" * 60)
    print("  TEST 3: Combined rules (c + g + vowels)")
    print("  Can the system auto-detect WHICH chars need context?")
    print("=" * 60)
    
    # Mix of simple and context-dependent rules
    training = [
        # c-rule examples
        ("cat",    "kæt"),
        ("city",   "sɪty"),
        ("cup",    "kʌp"),
        ("cent",   "sɛnt"),
        ("code",   "kɒdɛ"),
        ("clap",   "klæp"),
        ("acid",   "æsɪd"),
        ("cell",   "sɛll"),
        # g-rule examples
        ("game",   "gæmɛ"),
        ("gem",    "ʒɛm"),
        ("go",     "gɒ"),
        ("gin",    "ʒɪn"),
        # words without c or g (to learn vowel rules)
        ("bat",    "bæt"),
        ("bed",    "bɛd"),
        ("bit",    "bɪt"),
        ("bot",    "bɒt"),
        ("but",    "bʌt"),
    ]
    
    print(f"\n  Training pairs: {len(training)}")
    
    t0 = time.perf_counter()
    rules = build_rules(training)
    t_build = time.perf_counter() - t0
    
    program = AutoGeometricProgram(rules)
    print(f"\n  Built in {t_build*1000:.2f}ms")
    print(f"\n{program.describe()}")
    
    # Test
    print("\n  Training accuracy:")
    correct = 0
    for inp, expected in training:
        result = program.apply_word(inp)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        if result != expected:
            print(f"    {inp:8s} → {result:10s}  (expected {expected:10s}) {match}")
    print(f"  {correct}/{len(training)}")
    
    # Generalization
    print("\n  Generalization:")
    test = [
        ("cake",   "kækɛ"),
        ("mice",   "mɪsɛ"),
        ("huge",   "hʌʒɛ"),
        ("golf",   "gɒlf"),
    ]
    correct = 0
    for inp, expected in test:
        result = program.apply_word(inp)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"    {inp:8s} → {result:10s}  (expected {expected:10s}) {match}")
    print(f"  Accuracy: {correct}/{len(test)}")
    
    return program


if __name__ == "__main__":
    test_c_rule()
    test_g_rule()
    test_combined()
