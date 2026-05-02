#!/usr/bin/env python3
"""
Geometric IPA Demo: Learn phonetic rules by using them.

Each lesson teaches an IPA symbol by:
1. Explaining the sound (in progressively IPA-ified text)
2. Providing example codepoint mappings
3. Learning the rule geometrically via detection v5
4. Applying ALL learned rules to a demo sentence

Rules compose additively — each is a RECT pair (width-1 gate)
that maps one codepoint to another. The final program is just
the stack of all learned RECT pairs.

No training. No neural network. Just geometry.
"""

import numpy as np
import sys
import time
from collections import defaultdict

# Auto-detection machinery (gear-shift mechanism)
from phi_geometric.evaluations.auto_context_detection import (
    build_rules as auto_build_rules,
    extract_context_at,
)

# Gate infrastructure (from detection v5)
PHI = (1 + np.sqrt(5)) / 2
S8P = np.sqrt(8.0 / np.pi)
CGE = (4 - np.pi) / (6 * np.pi)

def ideal_gate(x):
    x = np.asarray(x, dtype=np.float64)
    f = S8P * x * (1.0 + CGE * x * x)
    f = np.clip(f, -500, 500)
    return x * (1.0 / (1.0 + np.exp(-f)))

def gate_step(x, t, s):
    return (ideal_gate(s * (x - (t - 0.5))) - ideal_gate(s * (x - (t + 0.5)))) / s

S = PHI ** 2  # sharpness


# ============================================================================
# GEOMETRIC RULE: A single learned character substitution
# ============================================================================

class GeometricRule:
    """A character substitution learned from examples.
    
    Internally, this is a RECT pair: two gate_step primitives that
    activate only at the target codepoint.
    
    The rule is LEARNED by providing example (input_char, output_char)
    pairs plus surrounding identity examples, then detecting the
    structure via the v5 pipeline.
    """
    
    def __init__(self, input_char, output_char, context=None):
        self.input_char = input_char
        self.output_char = output_char
        self.input_cp = ord(input_char)
        self.output_cp = ord(output_char)
        self.height = self.output_cp - self.input_cp
        self.context = context  # e.g., 'before_h' for digraphs
        
        # The geometric representation: RECT pair
        self.bp_open = self.input_cp - 0.5
        self.bp_close = self.input_cp + 0.5
        
        # Learn from examples
        self.training_examples = []
        self.detection_time = 0
        self.learned = False
    
    def learn_from_examples(self, examples):
        """Learn this rule from (input_cp, output_cp) examples.
        
        Uses detection v5 pipeline to verify the geometric structure.
        Returns True if the detected structure matches expectations.
        """
        self.training_examples = examples
        
        train_x = np.array([ex[0] for ex in examples], dtype=np.float64)
        train_y = np.array([ex[1] for ex in examples], dtype=np.float64)
        
        t0 = time.perf_counter()
        
        # The residual r = y - x should show a RECT at our target codepoint
        rs = train_y - train_x
        
        # Detect: find where residual is non-zero
        nonzero = np.where(np.abs(rs) > 0.5)[0]
        
        if len(nonzero) == 0:
            self.learned = False
            self.detection_time = time.perf_counter() - t0
            return False
        
        # The non-zero residual points should all be at our target codepoint
        target_xs = train_x[nonzero]
        detected_height = np.median(rs[nonzero])
        
        # Verify: detected height matches expected
        height_match = abs(detected_height - self.height) < 1.0
        
        # Verify: all non-zero points are at the target codepoint
        position_match = all(abs(x - self.input_cp) < 0.5 for x in target_xs)
        
        self.detected_height = detected_height
        self.detection_time = time.perf_counter() - t0
        self.learned = height_match and position_match
        
        return self.learned
    
    def apply_single(self, codepoint):
        """Apply this rule to a single codepoint using gate evaluation."""
        x = np.array([codepoint], dtype=np.float64)
        # RECT pair: step up at bp_open, step down at bp_close
        r = self.height * gate_step(x, self.input_cp, S)
        return float(x[0] + r[0])
    
    def __repr__(self):
        return (f"Rule('{self.input_char}'→'{self.output_char}', "
                f"cp {self.input_cp}→{self.output_cp}, "
                f"RECT at [{self.bp_open}, {self.bp_close}], "
                f"h={self.height:+d})")


# ============================================================================
# GEOMETRIC PROGRAM: Composed stack of rules
# ============================================================================

VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz')


def detect_magic_e(chars):
    """Detect magic-e positions: vowel + consonant + e + boundary.
    
    Returns a set of vowel indices that are affected by magic-e,
    and a set of e indices that should be silenced.
    
    Pattern: V + C + e + (space | end | consonant)
    Examples: make (a is magic), bite (i is magic), code (o is magic)
    
    This is Phase 0 feature extraction — computing a higher-level
    feature from the raw character sequence.
    """
    magic_vowels = set()  # indices of vowels affected by magic-e
    silent_e = set()      # indices of the silent e itself
    n = len(chars)
    
    for i in range(n - 2):
        v = chars[i].lower()
        c = chars[i+1].lower()
        e = chars[i+2].lower()
        
        if v not in VOWELS or c not in CONSONANTS or e != 'e':
            continue
        
        # Guard: if preceded by another vowel, this is likely a diphthong
        # (e.g., "oi" in "joined", "ou" in "house") — not magic-e
        if i > 0 and chars[i-1].lower() in VOWELS:
            continue
        
        # Check boundary after the e: end of word or another consonant
        if i + 3 >= n:
            # e is at end of text
            magic_vowels.add(i)
            silent_e.add(i + 2)
        elif chars[i+3] == ' ' or chars[i+3] in '.!?,;:':
            # e is at end of word
            magic_vowels.add(i)
            silent_e.add(i + 2)
        elif chars[i+3].lower() in CONSONANTS and (i + 4 >= n or chars[i+4] == ' ' or chars[i+4] in '.!?,;:'):
            # e + consonant at end (e.g., "makes", "bites")
            # Still magic-e if the consonant is just an inflection
            if chars[i+3].lower() in ('s', 'd'):
                magic_vowels.add(i)
                silent_e.add(i + 2)
    
    return magic_vowels, silent_e


def detect_igh(chars):
    """Detect 'igh' trigraph: i becomes /aɪ/, g+h become silent.
    
    Pattern: i + g + h (+ optional consonant like t)
    Examples: light, night, right, high, sigh, sight
    
    Returns: (magic_vowel_positions, silent_positions)
    """
    magic_vowels = set()
    silent_pos = set()
    n = len(chars)
    
    for i in range(n - 2):
        if (chars[i].lower() == 'i' and
            chars[i+1].lower() == 'g' and
            chars[i+2].lower() == 'h'):
            magic_vowels.add(i)
            silent_pos.add(i + 1)
            silent_pos.add(i + 2)
    
    return magic_vowels, silent_pos


def detect_silent_final_e(chars, already_silent):
    """Detect word-final 'e' that should be silent.
    
    In English, a final 'e' is almost always silent when:
    - The word has at least one other vowel
    - The word is 3+ characters long
    - The 'e' isn't already handled by magic-e
    
    Examples: dance, prince, voice, house, large, bridge
    Exceptions: me, he, be, we, the (short words / only vowel)
    """
    silent = set()
    n = len(chars)
    
    for i in range(n):
        if chars[i].lower() != 'e' or i in already_silent:
            continue
        
        # Check if word-final
        is_word_end = (i == n - 1 or chars[i+1] in ' .!?,;:\'"')
        if not is_word_end:
            continue
        
        # Find word start
        word_start = 0
        for j in range(i - 1, -1, -1):
            if chars[j] in ' .!?,;:\'"':
                word_start = j + 1
                break
        
        word_len = i - word_start + 1
        if word_len < 3:
            continue
        
        # Check for another vowel in the word (before this 'e')
        has_other_vowel = any(
            chars[k].lower() in VOWELS
            for k in range(word_start, i)
        )
        
        if has_other_vowel and i > 0 and chars[i-1].lower() in CONSONANTS:
            silent.add(i)
    
    return silent


# Long vowel IPA outputs (default — used before training)
LONG_VOWELS = {
    'a': 'eɪ',   # make, cake, rain, brain
    'e': 'iː',   # these, scene  (rare magic-e)
    'i': 'aɪ',   # bite, time, fine
    'o': 'oʊ',   # code, bone, boat, road
    'u': 'juː',  # cute, mute
}


def learn_magic_e_rules(training_pairs):
    """Learn context-dependent magic-e rules from (word, vowel_output) pairs.
    
    At each magic-e position (V+C+e+boundary), the existing context variables
    (prev_char, next_char=consonant, etc.) are extracted. The gear discovery
    machinery then finds rules that distinguish true magic-e (long vowel)
    from exceptions (short vowel).
    
    Returns: dict of vowel -> (rule_type, rule_data)
      - ('simple', output): all observations agree on this output
      - ('geared', gears_result): use discover_gears result
      - ('context', selector_result): use discover_selector result
    """
    from phi_geometric.evaluations.auto_context_detection import (
        discover_gears, discover_selector
    )
    
    # Extract observations per vowel
    vowel_obs = defaultdict(list)  # vowel -> [(output, context_dict), ...]
    
    for word, vowel_output in training_pairs:
        chars = list(word.lower())
        n = len(chars)
        for i in range(n - 2):
            v = chars[i]
            c = chars[i+1]
            e = chars[i+2]
            
            if v not in VOWELS or c not in CONSONANTS or e != 'e':
                continue
            
            # Diphthong guard
            if i > 0 and chars[i-1] in VOWELS:
                continue
            
            # Check boundary
            is_boundary = False
            if i + 3 >= n:
                is_boundary = True
            elif chars[i+3] in ' .!?,;:':
                is_boundary = True
            elif chars[i+3] in CONSONANTS and chars[i+3] in ('s', 'd'):
                if i + 4 >= n or chars[i+4] in ' .!?,;:':
                    is_boundary = True
            
            if is_boundary:
                # Build context dict (same as extract_context_at)
                prev_ch = chars[i-1] if i > 0 else ' '
                prev_prev_ch = chars[i-2] if i > 1 else ' '
                next_ch = chars[i+1] if i+1 < n else ' '
                next_next_ch = chars[i+2] if i+2 < n else ' '
                ctx = {
                    'prev_char': prev_ch,
                    'next_char': next_ch,  # = consonant
                    'next_next_char': next_next_ch,  # = 'e'
                    'prev_prev_char': prev_prev_ch,
                    'next_bigram': next_ch + next_next_ch,
                    'position': i,
                    'word_len': n,
                    'is_start': i == 0 or prev_ch == ' ',
                    'is_end': False,
                }
                vowel_obs[v].append((vowel_output, ctx))
                break  # Only first magic-e position per word
    
    # For each vowel, discover rules
    rules = {}
    for vowel, obs_list in vowel_obs.items():
        outputs = set(out for out, _ in obs_list)
        if len(outputs) == 1:
            rules[vowel] = ('simple', list(outputs)[0])
        else:
            # Try gear discovery
            coarse_var, pure_map, fine_gears, default_out, stats = \
                discover_gears(vowel, obs_list)
            if coarse_var:
                rules[vowel] = ('geared', {
                    'coarse_var': coarse_var,
                    'pure_map': pure_map,
                    'fine_gears': fine_gears,
                    'default': default_out,
                    'stats': stats,
                })
            else:
                # Fall back to simple selector
                best_var, sel_map, channels, gain = \
                    discover_selector(vowel, obs_list)
                if best_var:
                    rules[vowel] = ('context', {
                        'variable': best_var,
                        'selector_map': sel_map,
                        'channels': channels,
                        'gain': gain,
                        'default': LONG_VOWELS.get(vowel, vowel),
                    })
                else:
                    # Can't disambiguate — use majority
                    rules[vowel] = ('simple', default_out)
    
    return rules


def apply_magic_e_rule(vowel, ctx, rules):
    """Apply a learned magic-e rule at a specific position."""
    if vowel not in rules:
        return LONG_VOWELS.get(vowel, vowel)
    
    rule_type, data = rules[vowel]
    
    if rule_type == 'simple':
        return data
    
    elif rule_type == 'geared':
        coarse_val = ctx.get(data['coarse_var'])
        # Check pure map
        if coarse_val in data['pure_map']:
            return data['pure_map'][coarse_val]
        # Check fine gears
        if coarse_val in data.get('fine_gears', {}):
            fg = data['fine_gears'][coarse_val]
            fine_var, fine_map, _, _, zone_default = fg
            if fine_var:
                fine_val = ctx.get(fine_var)
                if fine_val in fine_map:
                    return fine_map[fine_val]
                return zone_default
            return zone_default
        return data['default']
    
    elif rule_type == 'context':
        val = ctx.get(data['variable'])
        return data['selector_map'].get(val, data['default'])
    
    return LONG_VOWELS.get(vowel, vowel)


class GeometricProgram:
    """A stack of geometric rules that compose additively.
    
    Four-phase architecture:
      Phase 0: FEATURE EXTRACTION (magic-e detection)
      Phase 1: DIGRAPH COLLAPSE (merge multi-char patterns)
      Phase 2: CONTEXT CHANNELS (auto-detected geared rules)
      Phase 3: CHARACTER RECTS (simple substitutions)
    """
    
    def __init__(self):
        self.rules = []
        self.digraph_rules = {}    # (char1, char2) -> replacement_str
        self.frozen_digraphs = set()  # digraph keys whose output is frozen
        self.context_rules = {}    # input_char -> auto-detected GeometricRule
        self.magic_e_enabled = False
        self.magic_e_rules = {}   # vowel -> learned rules from training
    
    def add_rule(self, rule):
        self.rules.append(rule)
    
    def add_digraph(self, char1, char2, replacement, freeze=False):
        """Add a digraph rule: two chars become one or more IPA symbols.
        
        If freeze=True, the output chars are marked as already-resolved
        and will skip Phases 2-3 (prevents re-processing).
        """
        self.digraph_rules[(char1, char2)] = replacement
        if freeze:
            self.frozen_digraphs.add((char1, char2))
    
    def add_context_rule(self, auto_rule):
        """Add a context-dependent rule from auto-detection."""
        self.context_rules[auto_rule.input_char] = auto_rule
    
    def apply_char(self, codepoint):
        """Apply all rules to a single codepoint.
        
        Uses exact integer evaluation (s→∞ limit of the gate).
        The geometric structure is the same RECT pairs — we evaluate
        at the natural integer resolution where gate_step becomes
        an exact indicator function. This avoids gate tail bleed
        that occurs with s=φ² on width-1 RECTs with large heights.
        """
        cp = int(round(codepoint))
        offset = 0
        for rule in self.rules:
            if cp == rule.input_cp:
                offset += rule.height
        return cp + offset
    
    def apply_text(self, text):
        """Apply all rules to a string using the 4-phase pipeline."""
        chars = list(text)
        chars_lc = [c.lower() for c in chars]  # lowercase for context extraction
        
        # Phase 0: Feature extraction (magic-e, igh, silent final e)
        # Store context from ORIGINAL chars (before digraph collapse)
        magic_e_ctx = {}  # original_index -> context_dict
        igh_vowels = set()  # positions where i→aɪ due to 'igh' trigraph
        if self.magic_e_enabled:
            magic_vowels, silent_e_positions = detect_magic_e(chars)
            for mi in magic_vowels:
                magic_e_ctx[mi] = extract_context_at(chars_lc, mi)
            # Also detect 'igh' trigraph (i→aɪ, g+h→silent)
            igh_v, igh_s = detect_igh(chars)
            igh_vowels = igh_v
            silent_e_positions |= igh_s
            # Remove igh vowels from magic_vowels if overlap
            magic_vowels -= igh_vowels
            # Detect silent word-final 'e' (dance, prince, voice, etc.)
            silent_final = detect_silent_final_e(chars, silent_e_positions)
            silent_e_positions |= silent_final
        else:
            magic_vowels, silent_e_positions = set(), set()
        
        # Phase 1: Digraph collapse (scan for 2-char patterns)
        # Track frozen (already-resolved) and silent (omit from output) positions
        # Also maintain orig_map: processed_idx -> original_char_idx
        # so that Phase 2 context rules can use ORIGINAL char context
        i = 0
        processed = []
        orig_map = []    # processed_idx -> original chars index
        frozen = set()
        silent = set()   # positions to omit from output (but keep for context!)
        magic_v_processed = {}  # processed_idx -> original_idx
        
        while i < len(chars):
            # Silent-e: KEEP in processed list (for context extraction)
            # but mark as silent so it's omitted from output
            if i in silent_e_positions:
                silent.add(len(processed))
                orig_map.append(i)
                processed.append(chars[i])
                i += 1
                continue
            
            if i + 1 < len(chars):
                pair = (chars[i].lower(), chars[i+1].lower())
                if pair in self.digraph_rules:
                    replacement = self.digraph_rules[pair]
                    start_idx = len(processed)
                    for rc in replacement:
                        orig_map.append(i)  # map all replacement chars to first original
                        processed.append(rc)
                    if pair in self.frozen_digraphs:
                        for j in range(start_idx, len(processed)):
                            frozen.add(j)
                    i += 2
                    continue
            
            # Track magic vowel positions in processed list
            if i in magic_vowels:
                magic_v_processed[len(processed)] = i
            if i in igh_vowels:
                magic_v_processed[len(processed)] = i  # igh vowels also tracked
            
            orig_map.append(i)
            processed.append(chars[i])
            i += 1
        
        # Phase 2: Context-dependent rules (gear shift)
        # Phase 3: Simple character-level rules
        result = []
        for idx, ch in enumerate(processed):
            # Silent positions: omit from output entirely
            if idx in silent:
                continue
            
            # Frozen or already IPA — pass through
            if idx in frozen or ord(ch) > 127:
                result.append(ch)
                continue
            
            lc = ch.lower()
            
            # 'igh' trigraph vowel: always aɪ
            if idx in magic_v_processed:
                orig_idx = magic_v_processed[idx]
                if orig_idx in igh_vowels:
                    result.append('aɪ')
                    continue
            
            # Magic-e vowel: use learned rules or default long vowel
            if idx in magic_v_processed and (lc in self.magic_e_rules or lc in LONG_VOWELS):
                if self.magic_e_rules:
                    # Use context from ORIGINAL chars (before digraph collapse)
                    orig_idx = magic_v_processed[idx]
                    ctx = magic_e_ctx.get(orig_idx, extract_context_at(processed, idx))
                    result.append(apply_magic_e_rule(lc, ctx, self.magic_e_rules))
                else:
                    result.append(LONG_VOWELS.get(lc, lc))
                continue
            
            # Context-dependent rules (use ORIGINAL char context, lowercased)
            if lc in self.context_rules:
                oi = orig_map[idx] if idx < len(orig_map) else idx
                ctx = extract_context_at(chars_lc, oi)
                output = self.context_rules[lc].apply(lc, ctx)
                result.append(output)
                continue
            
            # Simple character-level rules (IPA is lowercase)
            cp = self.apply_char(ord(lc))
            result.append(chr(cp))
        
        return ''.join(result)
    
    def show_program(self):
        """Display the current geometric program."""
        if not self.rules and not self.digraph_rules and not self.context_rules:
            return "  (empty program)"
        lines = []
        for i, rule in enumerate(self.rules):
            lines.append(f"  [{i+1}] {rule.input_char}→{rule.output_char}  "
                        f"RECT[{rule.bp_open:.1f}, {rule.bp_close:.1f}] "
                        f"h={rule.height:+d}")
        for (c1, c2), rep in self.digraph_rules.items():
            lines.append(f"  [D] {c1}{c2}→{rep}  (digraph pre-scan)")
        for ic, crule in self.context_rules.items():
            if crule.rule_type == 'geared':
                n_fine = sum(1 for v in crule.params.get('fine_gears', {}).values()
                           if v[0] is not None)
                lines.append(f"  [G] {ic}→GEARED on "
                           f"{crule.params['coarse_var']} "
                           f"({crule.params['stats'].get('coarse_pure', '?')} "
                           f"pure + {n_fine} fine gear)")
            elif crule.rule_type == 'context':
                lines.append(f"  [C] {ic}→context on "
                           f"{crule.params['selector_variable']}")
        return '\n'.join(lines)


# ============================================================================
# IPA LESSONS
# ============================================================================

def make_examples(input_cp, output_cp, n_context=3):
    """Generate training examples: the target mapping + identity context."""
    examples = [(input_cp, output_cp)]  # The rule itself
    # Add identity examples around the target
    for offset in range(-n_context, n_context + 1):
        cp = input_cp + offset
        if cp != input_cp and 32 <= cp <= 126:
            examples.append((cp, cp))
    return examples


LESSONS = [
    # --- CONSONANT DIGRAPHS (the visually distinctive IPA symbols) ---
    {
        'title': 'The "sh" sound',
        'explain': (
            'In English, we write "sh" for the sound in "ship" and "fish".\n'
            'In IPA, this is a single symbol: ʃ (called "esh").\n'
            'Let\'s teach the system: when we see "sh", replace with "ʃ".'
        ),
        'type': 'digraph',
        'chars': ('s', 'h'),
        'ipa': 'ʃ',
        'demo_sentence': 'She sells seashells by the seashore.',
    },
    {
        'title': 'The voiceless "th" sound',
        'explain': (
            'English has two "th" sounds. The voiceless one appears in\n'
            '"think", "thick", and "math". In IPA: θ (Greek theta).\n'
            'Let\'s teach: "th" at word starts → "θ".'
        ),
        'type': 'digraph',
        'chars': ('t', 'h'),
        'ipa': 'θ',
        'demo_sentence': 'Think about this thing three times.',
    },
    {
        'title': 'The "ng" sound',
        'explain': (
            'The sound at the end of "sing" and "thing" is one sound\n'
            'in IPA: ŋ (called "eng"). It\'s not "n" + "g" — it\'s\n'
            'a single nasal consonant made at the back of the mouth.'
        ),
        'type': 'digraph',
        'chars': ('n', 'g'),
        'ipa': 'ŋ',
        'demo_sentence': 'Singing songs all evening long.',
    },
    {
        'title': 'The "ch" sound',
        'explain': (
            'The "ch" in "church" and "chip" is an affricate — a stop\n'
            'followed by a fricative. In IPA: ʧ (t-esh ligature).\n'
            'Like "sh" but with a "t" onset: t + ʃ = ʧ.'
        ),
        'type': 'digraph',
        'chars': ('c', 'h'),
        'ipa': 'ʧ',
        'demo_sentence': 'The children chose chocolate chip cheesecake.',
    },
    {
        'title': 'The "wh" simplification',
        'explain': (
            'In most modern English dialects, "wh" is just pronounced\n'
            'as "w". "What" = "wat", "when" = "wen", "where" = "were".\n'
            'Historically it was a voiceless w, but that distinction is lost.'
        ),
        'type': 'digraph',
        'chars': ('w', 'h'),
        'ipa': 'w',
        'demo_sentence': 'What happens when the white whale whistles?',
    },
    {
        'title': 'The "ck" simplification',
        'explain': (
            'The digraph "ck" is just a way to spell /k/ after short vowels.\n'
            '"back" = "bak", "pick" = "pik", "duck" = "duk".\n'
            'Two letters, one sound: k.'
        ),
        'type': 'digraph',
        'chars': ('c', 'k'),
        'ipa': 'k',
        'demo_sentence': 'The duck picked a black rock from the deck.',
    },
    {
        'title': 'The "qu" combination',
        'explain': (
            'In English, "qu" is always pronounced /kw/.\n'
            '"queen" = "kween", "quick" = "kwik", "quest" = "kwest".\n'
            'Two letters become two sounds: k + w.'
        ),
        'type': 'digraph',
        'chars': ('q', 'u'),
        'ipa': 'kw',
        'demo_sentence': 'The queen made a quick request for quiet.',
    },
    
    # --- SILENT / NASAL DIGRAPHS ---
    {
        'title': 'The silent "gh"',
        'explain': (
            'The digraph "gh" is silent in modern English (it used to be\n'
            'a guttural /x/ in Old English). "through", "thought", "bought".\n'
            'Combined with Phase 0 "igh" detection: "light" = /laɪt/,\n'
            '"night" = /naɪt/, "right" = /ɹaɪt/.'
        ),
        'type': 'digraph',
        'chars': ('g', 'h'),
        'ipa': '',
        'demo_sentence': 'The bright light shone through the night sky.',
    },
    {
        'title': 'The "nk" nasal cluster',
        'explain': (
            'Before "k", the "n" becomes the velar nasal /ŋ/.\n'
            '"think" = /θɪŋk/, "bank" = /bæŋk/, "drink" = /dɹɪŋk/.\n'
            'Same assimilation as "ng" but the "k" is still pronounced.'
        ),
        'type': 'digraph',
        'chars': ('n', 'k'),
        'ipa': 'ŋk',
        'demo_sentence': 'I think the bank will sink if we drink the ink.',
    },
    
    # --- VOWEL DIGRAPHS (frozen output — skip further vowel processing) ---
    {
        'title': 'The "ee" long vowel',
        'explain': (
            'The digraph "ee" makes the long /iː/ sound: "see", "tree",\n'
            '"feet", "need". This is different from short "i" (as in "sit").\n'
            'The output is FROZEN — it won\'t be re-processed by vowel rules.'
        ),
        'type': 'digraph',
        'chars': ('e', 'e'),
        'ipa': 'iː',
        'freeze': True,
        'demo_sentence': 'We need to see the tree by the creek.',
    },
    {
        'title': 'The "oo" long vowel',
        'explain': (
            'The digraph "oo" usually makes the long /uː/ sound:\n'
            '"food", "moon", "boot", "cool". (In some words like "book"\n'
            'it\'s shorter, but we\'ll use the long form as default.)'
        ),
        'type': 'digraph',
        'chars': ('o', 'o'),
        'ipa': 'uː',
        'freeze': True,
        'demo_sentence': 'The cool moon shone over the pool.',
    },
    {
        'title': 'The "ai" diphthong',
        'explain': (
            'The digraph "ai" makes the /eɪ/ diphthong: "rain", "train",\n'
            '"brain", "main". This is the same sound as the "long a"\n'
            'in "make" — two different spellings for the same sound!'
        ),
        'type': 'digraph',
        'chars': ('a', 'i'),
        'ipa': 'eɪ',
        'freeze': True,
        'demo_sentence': 'The rain on the train was a pain.',
    },
    {
        'title': 'The "oa" diphthong',
        'explain': (
            'The digraph "oa" makes the /oʊ/ diphthong: "boat", "coat",\n'
            '"road", "load". Same sound as "long o" in "bone" — again,\n'
            'two spellings encoding the same geometric position!'
        ),
        'type': 'digraph',
        'chars': ('o', 'a'),
        'ipa': 'oʊ',
        'freeze': True,
        'demo_sentence': 'The boat floated down the road to the coast.',
    },
    
    # --- MAGIC-E (Phase 0 feature extraction + trained exceptions) ---
    {
        'title': 'The magic "e" — non-local context!',
        'explain': (
            'Silent "e" at the end of a word changes the preceding vowel\n'
            'from short to long AND deletes itself. "bit" → /bɪt/ but\n'
            '"bite" → /baɪt/. "cod" → /kɒd/ but "code" → /koʊd/.\n'
            'This is a NON-LOCAL effect: the e at position i+2\n'
            'affects the vowel at position i. This requires Phase 0\n'
            'feature extraction — scanning for patterns BEFORE processing.\n'
            'BUT some common words are exceptions: "come" ≠ /koʊm/,\n'
            '"love" ≠ /loʊv/. We train from examples and let the\n'
            'gear-shift discover what distinguishes them!'
        ),
        'type': 'magic_e_trained',
        'training': [
            # --- Vowel 'a': magic-e works ---
            ("make",  "eɪ"),
            ("cake",  "eɪ"),
            ("lake",  "eɪ"),
            ("wave",  "eɪ"),
            ("save",  "eɪ"),
            ("cave",  "eɪ"),
            ("name",  "eɪ"),
            ("same",  "eɪ"),
            ("late",  "eɪ"),
            ("fate",  "eɪ"),
            # --- Vowel 'a': exception ---
            ("have",  "æ"),
            # --- Vowel 'i': magic-e works ---
            ("bite",  "aɪ"),
            ("ride",  "aɪ"),
            ("fine",  "aɪ"),
            ("time",  "aɪ"),
            ("dive",  "aɪ"),
            ("wine",  "aɪ"),
            ("like",  "aɪ"),
            ("life",  "aɪ"),
            # --- Vowel 'i': exceptions ---
            ("give",  "ɪ"),
            ("live",  "ɪ"),
            # --- Vowel 'o': magic-e works ---
            ("hope",  "oʊ"),
            ("code",  "oʊ"),
            ("bone",  "oʊ"),
            ("home",  "oʊ"),
            ("nose",  "oʊ"),
            ("rope",  "oʊ"),
            ("note",  "oʊ"),
            ("woke",  "oʊ"),
            ("stove", "oʊ"),
            ("drove", "oʊ"),
            # --- Vowel 'o': exceptions (Germanic survivals) ---
            ("come",  "ʌ"),
            ("some",  "ʌ"),
            ("done",  "ʌ"),
            ("love",  "ʌ"),
            ("dove",  "ʌ"),
            ("gone",  "ɒ"),
            ("none",  "ʌ"),
            ("shove", "ʌ"),
            ("above", "ʌ"),
            # --- Vowel 'e': magic-e works ---
            ("these", "iː"),
            # --- Vowel 'e': exceptions (r-controlled) ---
            ("there", "ɛ"),
            ("where", "ɛ"),
            ("here",  "ɛ"),
            # --- Vowel 'u': magic-e works ---
            ("cute",  "juː"),
            ("mute",  "juː"),
        ],
        'demo_sentence': 'I hope to make a fine cake but some come from love not code.',
    },
    
    # --- VOWEL SUBSTITUTIONS (simplified: default SHORT pronunciation) ---
    {
        'title': 'The short "a" vowel',
        'explain': (
            'The short "a" as in "cat" is written æ in IPA (called "ash").\n'
            'This is one of the most distinctive IPA symbols.\n'
            'We\'ll map: a → æ (simplified — real IPA is context-dependent).'
        ),
        'type': 'char',
        'input': 'a',
        'ipa': 'æ',
        'demo_sentence': 'A black cat sat on a flat mat.',
    },
    {
        'title': 'The short "e" vowel',
        'explain': (
            'The short "e" as in "bed" is written ɛ in IPA ("epsilon").\n'
            'It looks like a reversed 3. Common in "get", "red", "help".'
        ),
        'type': 'char',
        'input': 'e',
        'ipa': 'ɛ',
        'demo_sentence': 'Ten men held red pens.',
    },
    {
        'title': 'The short "i" vowel',
        'explain': (
            'The short "i" as in "bit" is written ɪ in IPA ("small cap I").\n'
            'It\'s slightly different from regular "i" — more relaxed.'
        ),
        'type': 'char',
        'input': 'i',
        'ipa': 'ɪ',
        'demo_sentence': 'His fish swims in a big dish.',
    },
    {
        'title': 'The short "o" vowel',
        'explain': (
            'The short "o" as in "lot" is written ɒ in IPA.\n'
            'It\'s an open back rounded vowel — mouth wide open.'
        ),
        'type': 'char',
        'input': 'o',
        'ipa': 'ɒ',
        'demo_sentence': 'Bob got a pot from the shop.',
    },
    {
        'title': 'The short "u" vowel',
        'explain': (
            'The short "u" as in "cup" is written ʌ in IPA ("caret/wedge").\n'
            'It looks like an upside-down v. Common in "but", "run", "fun".'
        ),
        'type': 'char',
        'input': 'u',
        'ipa': 'ʌ',
        'demo_sentence': 'A duck must run up the dusty bus.',
    },
    
    # --- MORE CONSONANT SUBSTITUTIONS ---
    {
        'title': 'The "j" sound',
        'explain': (
            'The English "j" as in "jump" is written dʒ in IPA.\n'
            'For simplicity, we\'ll map j → ʒ (the voiced "sh" sound),\n'
            'which is the core fricative component.'
        ),
        'type': 'char',
        'input': 'j',
        'ipa': 'ʒ',
        'demo_sentence': 'Jack just joined the jazz jam.',
    },
    {
        'title': 'The "r" sound',
        'explain': (
            'The English "r" is written ɹ in IPA ("turned r").\n'
            'It\'s upside-down because IPA reserves "r" for the\n'
            'trilled r used in Spanish and Italian.'
        ),
        'type': 'char',
        'input': 'r',
        'ipa': 'ɹ',
        'demo_sentence': 'Run around the red river road.',
    },
    
    # --- CONTEXT-DEPENDENT RULES (auto-detected via gear shift) ---
    {
        'title': 'Soft and hard "c"',
        'explain': (
            'The letter "c" has TWO sounds in English:\n'
            '  - Hard c (= k) before a, o, u, consonants: cat, cup, clap\n'
            '  - Soft c (= s) before e, i: city, cent, ace\n'
            'This is a CONTEXT-DEPENDENT rule. The system must discover\n'
            'that next_char determines the output. No hard-coding!'
        ),
        'type': 'context',
        'training': [
            ("cat",   "kat"),
            ("city",  "sity"),
            ("cup",   "kup"),
            ("cent",  "sent"),
            ("code",  "kode"),
            ("clap",  "klap"),
            ("acid",  "asid"),
            ("cell",  "sell"),
            ("cold",  "kold"),
            ("cry",   "kry"),
            ("cite",  "site"),
            ("ace",   "ase"),
        ],
        'target_char': 'c',
        'demo_sentence': 'A nice cat can catch mice in the cold cellar.',
    },
    {
        'title': 'Soft and hard "g" (gear shift!)',
        'explain': (
            'The letter "g" is even trickier:\n'
            '  - Hard g before a, o, u, l: game, go, gust, glad\n'
            '  - Soft g (= j) before e, y: gem, gym\n'
            '  - Before "i": BOTH! gift=hard, gin=soft\n'
            'This needs a GEAR SHIFT: coarse gear on next_char,\n'
            'fine gear on next_next_char when next_char="i".'
        ),
        'type': 'context',
        'training': [
            # Hard g (before a, o, u, l, r)
            ("game",   "game"),
            ("go",     "go"),
            ("gust",   "gust"),
            ("glad",   "glad"),
            ("grab",   "grab"),
            ("gum",    "gum"),
            ("gap",    "gap"),
            # Soft g (before e — Romance/Latin origin)
            ("gem",    "jem"),
            ("gel",    "jel"),
            ("gene",   "jene"),
            ("gentle", "jentle"),
            ("germ",   "jerm"),
            # Hard g before e (Germanic exceptions!)
            ("get",    "get"),
            ("gear",   "gear"),
            ("geese",  "geese"),
            ("geld",   "geld"),
            # Soft g before i
            ("gin",    "jin"),
            ("giant",  "jiant"),
            ("gist",   "jist"),
            ("gig",    "jig"),
            # Hard g before i
            ("gift",   "gift"),
            ("girl",   "girl"),
            ("gild",   "gild"),
            # Soft g before y
            ("gym",    "jym"),
        ],
        'target_char': 'g',
        'demo_sentence': 'The gentle giant gave a gift to the girl and got the gem.',
    },
    {
        'title': 'The two faces of "y"',
        'explain': (
            'The letter "y" is a consonant at word starts but a vowel\n'
            'elsewhere. Word-initial: "yes" = /jes/, "you" = /jou/.\n'
            'Mid-word: "gym" = /gim/, "myth" = /miθ/.\n'
            'The system should discover is_start as the selector.'
        ),
        'type': 'context',
        'training': [
            ("yes",  "jes"),
            ("yet",  "jet"),
            ("yam",  "jam"),
            ("yap",  "jap"),
            ("yell", "jell"),
            ("gym",  "gim"),
            ("myth", "mith"),
            ("hymn", "himn"),
            ("lynx", "linx"),
            ("sync", "sinc"),
        ],
        'target_char': 'y',
        'demo_sentence': 'Yes, the young gym has many myths about yoga.',
    },
]


# ============================================================================
# DEMO RUNNER
# ============================================================================

def run_demo():
    program = GeometricProgram()
    
    print("=" * 70)
    print("  GEOMETRIC IPA: Learning Phonetics Through Geometry")
    print("  Each rule is a RECT pair — no training, no neural network.")
    print("=" * 70)
    print()
    print("The system will learn IPA rules one at a time from examples.")
    print("Each rule is a geometric primitive (gate_step pair).")
    print("Rules compose additively. Watch the text transform.\n")
    
    for lesson_num, lesson in enumerate(LESSONS, 1):
        print(f"{'─' * 70}")
        print(f"  LESSON {lesson_num}: {lesson['title']}")
        print(f"{'─' * 70}")
        
        # Show explanation — apply current rules to it!
        explanation = lesson['explain']
        if lesson_num > 1:
            ipa_explanation = program.apply_text(explanation)
            print(f"\n{ipa_explanation}")
        else:
            print(f"\n{explanation}")
        
        if lesson['type'] == 'digraph':
            c1, c2 = lesson['chars']
            ipa_str = lesson['ipa']
            
            # Register as digraph (pre-scan replaces both chars)
            input_cp = ord(c1)
            
            if len(ipa_str) == 1:
                # Single-char replacement — verify with geometric detection
                output_cp = ord(ipa_str)
                examples = make_examples(input_cp, output_cp)
                rule = GeometricRule(c1, ipa_str)
                
                print(f"\n  Training examples:")
                print(f"    Target: '{c1}{c2}' → "
                      f"'{ipa_str}' (U+{output_cp:04X})")
                print(f"    + {len(examples)-1} identity examples as context")
                
                t0 = time.perf_counter()
                success = rule.learn_from_examples(examples)
                t_learn = time.perf_counter() - t0
                
                print(f"\n  ⚡ Geometric detection ({t_learn*1000:.1f}ms):")
                print(f"    Structure: RECT pair (width 1)")
                print(f"    Breakpoints: [{rule.bp_open:.1f}, {rule.bp_close:.1f}]")
                print(f"    Height: {rule.height:+d}")
                print(f"    Learned: {'✓' if success else '✗'}")
            else:
                # Multi-char replacement (e.g., qu → kw)
                cps = ' + '.join(f"'{c}'" for c in ipa_str)
                print(f"\n  Digraph mapping:")
                print(f"    '{c1}{c2}' → {cps}  (2 → {len(ipa_str)} expansion)")
                print(f"\n  ⚡ Pattern registered (no geometric detection needed)")
                print(f"    Learned: ✓")
            
            freeze = lesson.get('freeze', False)
            program.add_digraph(c1, c2, ipa_str, freeze=freeze)
            if freeze:
                print(f"    Output frozen: ✓ (skips further vowel processing)")
            
        elif lesson['type'] == 'magic_e_trained':
            # Learn magic-e rules from training data
            training = lesson['training']
            program.magic_e_enabled = True
            
            print(f"\n  Phase 0 feature extraction: V+C+e+boundary detector")
            print(f"  Training from {len(training)} word examples:")
            
            # Show a sample
            for word, out in training[:4]:
                print(f"    {word:8s} → vowel becomes '{out}'")
            print(f"    ... ({len(training)-4} more)")
            
            t0 = time.perf_counter()
            rules = learn_magic_e_rules(training)
            t_learn = time.perf_counter() - t0
            
            program.magic_e_rules = rules
            
            print(f"\n  ⚡ Auto-detection ({t_learn*1000:.1f}ms):")
            for vowel in sorted(rules.keys()):
                rtype, rdata = rules[vowel]
                if rtype == 'simple':
                    print(f"    Vowel '{vowel}': always → '{rdata}'")
                elif rtype == 'geared':
                    stats = rdata.get('stats', {})
                    coarse = rdata['coarse_var']
                    n_pure = stats.get('coarse_pure', '?')
                    n_amb = stats.get('coarse_ambiguous', 0)
                    print(f"    Vowel '{vowel}': GEARED on {coarse} "
                          f"({n_pure} pure + {n_amb} ambiguous teeth)")
                    # Show pure teeth grouped by output
                    by_output = defaultdict(list)
                    for val, oc in rdata['pure_map'].items():
                        by_output[oc].append(val)
                    for oc, vals in sorted(by_output.items()):
                        vals_str = ', '.join(repr(v) for v in sorted(vals, key=str))
                        print(f"      → '{oc}' when {coarse} ∈ {{{vals_str}}}")
                    for cval, fg in sorted(rdata.get('fine_gears', {}).items(), key=str):
                        fine_var, fine_map, _, _, zone_def = fg
                        if fine_var:
                            print(f"      When {coarse}='{cval}' → fine gear on "
                                  f"{fine_var} (default='{zone_def}'):")
                            for fv, fo in sorted(fine_map.items(), key=str):
                                print(f"        {fine_var}='{fv}' → '{fo}'")
                        else:
                            print(f"      When {coarse}='{cval}' → default '{zone_def}'")
                elif rtype == 'context':
                    var = rdata['variable']
                    print(f"    Vowel '{vowel}': context on {var}")
                    for oc, vals in sorted(rdata['channels'].items()):
                        vals_str = ', '.join(repr(v) for v in sorted(vals, key=str))
                        print(f"      → '{oc}' when {var} ∈ {{{vals_str}}}")
            print(f"    Learned: ✓")
        
        elif lesson['type'] == 'char':
            input_char = lesson['input']
            ipa_char = lesson['ipa']
            input_cp = ord(input_char)
            output_cp = ord(ipa_char)
            
            examples = make_examples(input_cp, output_cp)
            
            rule = GeometricRule(input_char, ipa_char)
            
            print(f"\n  Training examples:")
            print(f"    Target: '{input_char}' (U+{input_cp:04X}) → "
                  f"'{ipa_char}' (U+{output_cp:04X})")
            print(f"    + {len(examples)-1} identity examples as context")
            
            t0 = time.perf_counter()
            success = rule.learn_from_examples(examples)
            t_learn = time.perf_counter() - t0
            
            print(f"\n  ⚡ Geometric detection ({t_learn*1000:.1f}ms):")
            print(f"    Structure: RECT pair (width 1)")
            print(f"    Breakpoints: [{rule.bp_open:.1f}, {rule.bp_close:.1f}]")
            print(f"    Height: {rule.height:+d}")
            print(f"    Learned: {'✓' if success else '✗'}")
            
            program.add_rule(rule)
        
        elif lesson['type'] == 'context':
            training = lesson['training']
            target_char = lesson['target_char']
            
            print(f"\n  Training word pairs: {len(training)}")
            for inp, out in training[:6]:
                print(f"    {inp:8s} → {out}")
            if len(training) > 6:
                print(f"    ... ({len(training)-6} more)")
            
            t0 = time.perf_counter()
            auto_rules = auto_build_rules(training)
            t_learn = time.perf_counter() - t0
            
            # Find the rule for our target character
            target_rule = None
            for ar in auto_rules:
                if ar.input_char == target_char:
                    target_rule = ar
                    break
            
            if target_rule is None:
                print(f"\n  ⚠ No rule found for '{target_char}'")
            else:
                print(f"\n  ⚡ Auto-detection ({t_learn*1000:.1f}ms):")
                print(f"    Rule type: {target_rule.rule_type}")
                
                if target_rule.rule_type == 'geared':
                    stats = target_rule.params.get('stats', {})
                    coarse_var = target_rule.params['coarse_var']
                    pure_map = target_rule.params['pure_map']
                    fine_gears = target_rule.params.get('fine_gears', {})
                    n_fine = sum(1 for v in fine_gears.values()
                               if v[0] is not None)
                    
                    print(f"    Coarse gear: {coarse_var} "
                          f"({stats.get('coarse_pure', '?')} pure teeth)")
                    
                    # Show pure teeth grouped by output
                    by_output = defaultdict(list)
                    for val, oc in pure_map.items():
                        by_output[oc].append(val)
                    for oc, vals in sorted(by_output.items()):
                        vals_str = ', '.join(repr(v) for v in sorted(vals, key=str))
                        print(f"      → '{oc}' when {coarse_var} ∈ {{{vals_str}}}")
                    
                    if n_fine > 0:
                        print(f"    Fine gear(s): {n_fine} fallthrough register(s)")
                        for cval, fg in sorted(fine_gears.items(), key=str):
                            if fg[0] is not None:
                                fine_var, fine_map, _, fine_gain, zone_def = fg
                                print(f"      When {coarse_var}='{cval}' "
                                      f"→ engage {fine_var} "
                                      f"(zone default='{zone_def}'):")
                                for fv, fo in sorted(fine_map.items(), key=str):
                                    print(f"        {fine_var}='{fv}' → '{fo}'")
                    
                    print(f"    Ambiguous teeth: "
                          f"{stats.get('coarse_ambiguous', 0)}")
                    
                elif target_rule.rule_type == 'context':
                    var = target_rule.params['selector_variable']
                    channels = target_rule.params.get('channels', {})
                    print(f"    Selector: {var}")
                    for oc, vals in sorted(channels.items()):
                        vals_str = ', '.join(repr(v) for v in sorted(vals, key=str))
                        print(f"      → '{oc}' when {var} ∈ {{{vals_str}}}")
                
                print(f"    Learned: ✓")
                program.add_context_rule(target_rule)
        
        # Apply to demo sentence
        original = lesson['demo_sentence']
        transformed = program.apply_text(original)
        
        print(f"\n  Demo:")
        print(f"    IN:  {original}")
        print(f"    OUT: {transformed}")
        
        # Show current program state
        print(f"\n  Geometric program ({len(program.rules)} char rules, "
              f"{len(program.digraph_rules)} digraph rules, "
              f"{len(program.context_rules)} context rules):")
        print(program.show_program())
        print()
    
    # Final showcase
    print(f"\n{'=' * 70}")
    print("  FINAL RESULT: Full Geometric IPA Program")
    print(f"{'=' * 70}")
    print()
    print("Program:")
    print(program.show_program())
    print()
    
    # Apply to a longer text — exercise all rule types including magic-e
    showcase_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She thinks singing is the best thing in the world.",
        "A thin man sat on a red bus eating fish and chips.",
        "A nice city cat can catch mice in the cold cellar.",
        "The gentle giant gave a gift to the girl and got the gem.",
        "I hope to make a fine cake and ride home in time.",
        "We need to see the boat float down the road in the rain.",
        "The bright light shone right there in the night.",
        "Some love to dance but none have a choice in the voice.",
        "I think the prince sat on the fence and drank his drink.",
    ]
    
    print("Showcase transformations:")
    print()
    for text in showcase_texts:
        transformed = program.apply_text(text)
        print(f"  EN:  {text}")
        print(f"  IPA: {transformed}")
        print()
    
    # Statistics
    n_char = len(program.rules)
    n_digraph = len(program.digraph_rules)
    n_frozen = len(program.frozen_digraphs)
    n_context = len(program.context_rules)
    n_geared = sum(1 for r in program.context_rules.values()
                   if r.rule_type == 'geared')
    has_magic_e = program.magic_e_enabled
    total_rules = n_char + n_digraph + n_context + (1 if has_magic_e else 0)
    total_primitives = n_char * 2 + n_digraph
    for r in program.context_rules.values():
        if r.rule_type == 'geared':
            total_primitives += len(r.params.get('pure_map', {})) * 2
            for fg in r.params.get('fine_gears', {}).values():
                if fg[0] is not None:
                    total_primitives += len(fg[1]) * 2
        elif r.rule_type == 'context':
            total_primitives += len(r.params.get('selector_map', {})) * 2
    n_magic_e_rules = len(program.magic_e_rules)
    n_magic_e_geared = sum(1 for _, (rt, _) in program.magic_e_rules.items() if rt == 'geared')
    if has_magic_e:
        for vowel, (rt, rd) in program.magic_e_rules.items():
            if rt == 'geared':
                total_primitives += len(rd.get('pure_map', {})) * 2
                for fg in rd.get('fine_gears', {}).values():
                    if fg[0] is not None:
                        total_primitives += len(fg[1]) * 2
            elif rt == 'simple':
                total_primitives += 2
            elif rt == 'context':
                total_primitives += len(rd.get('selector_map', {})) * 2
    
    print(f"{'─' * 70}")
    print(f"  Statistics:")
    print(f"    Simple char rules: {n_char}")
    print(f"    Digraph rules: {n_digraph} ({n_frozen} frozen)")
    print(f"    Context rules: {n_context} ({n_geared} geared)")
    print(f"    Magic-e: {n_magic_e_rules} trained vowel rules ({n_magic_e_geared} geared)")
    print(f"    Total rules: {total_rules + n_magic_e_rules}")
    print(f"    Geometric primitives: {total_primitives} gate_step calls")
    print(f"    Four-phase architecture:")
    print(f"      Phase 0: FEATURE EXTRACT   (magic-e: {n_magic_e_rules} trained vowel rules)")
    print(f"      Phase 1: DIGRAPH COLLAPSE  ({n_digraph} patterns, {n_frozen} frozen)")
    print(f"      Phase 2: CONTEXT CHANNELS  ({n_context} geared/context rules)")
    print(f"      Phase 3: CHARACTER RECTS   ({n_char} simple substitutions)")
    print(f"    Gradient descent: none")
    print(f"    Neural network: none")
    print(f"    Structure IS the computation.")
    print(f"{'─' * 70}")
    
    return program


if __name__ == "__main__":
    program = run_demo()
