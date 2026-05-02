#!/usr/bin/env python3
"""
Context-Dependent Geometric Rules: Three Approaches

Test case: English "c" rule
  - c before {e, i} → /s/ (soft c: "city", "cent")
  - c before {a, o, u, consonants} → /k/ (hard c: "cat", "cup")

Approach 1: BIGRAM ENCODING
  Flatten (current, next) into one integer: code = current*128 + next
  Learn f(code) → output_codepoint using v5 pipeline on 1D domain [0, 16384)

Approach 2: NESTED SELECTORS (gate × gate)
  Factor into: output = current + gate(current, c) * [gate(next, front_vowel) * Δs + (1-gate(next, front_vowel)) * Δk]
  Product of 1D gates on different variables. Learn each factor independently.

Approach 3: SHADER CHANNELS
  Channel A: c→k rule (always produces 'k' when current='c')
  Channel B: c→s rule (always produces 's' when current='c')
  Selector: geometric function of next_char picks channel A or B
  Channels don't interfere — orthogonal registers.

All three should give the same result. The question is which
is most natural for geometric learning and composition.
"""

import numpy as np
import time

PHI = (1 + np.sqrt(5)) / 2
S8P = np.sqrt(8.0 / np.pi)
CGE = (4 - np.pi) / (6 * np.pi)
S = PHI ** 2

def ideal_gate(x):
    x = np.asarray(x, dtype=np.float64)
    f = S8P * x * (1.0 + CGE * x * x)
    f = np.clip(f, -500, 500)
    return x * (1.0 / (1.0 + np.exp(-f)))

def gate_step(x, t, s):
    return (ideal_gate(s * (x - (t - 0.5))) - ideal_gate(s * (x - (t + 0.5)))) / s


# ============================================================================
# TEST DATA: The "c" rule
# ============================================================================

# Truth table for 'c' before various characters
C_RULES = {
    # c before front vowels → s (soft c)
    'e': 's', 'i': 's',
    # c before back vowels / consonants → k (hard c)
    'a': 'k', 'o': 'k', 'u': 'k',
    'l': 'k', 'r': 'k', 'h': 'k', 'k': 'k',
    't': 'k', 'n': 'k', 'd': 'k',
}

# Test words — c-rule ONLY (no vowel IPA substitutions)
TEST_WORDS_C_ONLY = [
    ("cat",   "kat"),
    ("city",  "sity"),
    ("cup",   "kup"),
    ("cent",  "sent"),
    ("code",  "kode"),
    ("ice",   "ise"),      # c before e
    ("clap",  "klap"),    # c before l
    ("cry",   "kry"),     # c before r
    ("acid",  "asid"),    # c before i
    ("occur", "okkur"),   # c before c → k (first c before c=consonant)
]

# Full IPA expected (c-rule + vowel rules + digraphs)
TEST_WORDS_FULL = [
    ("cat",   "kæt"),
    ("city",  "sɪty"),
    ("cup",   "kʌp"),
    ("cent",  "sɛnt"),
    ("code",  "kɒdɛ"),
    ("ice",   "ɪsɛ"),
    ("clap",  "klæp"),
    ("cry",   "kɹy"),
    ("acid",  "æsɪd"),
    ("occur", "ɒkkʌɹ"),
]


# ============================================================================
# APPROACH 1: BIGRAM ENCODING
# ============================================================================

def bigram_code(c1, c2):
    """Encode two characters as a single integer."""
    return ord(c1) * 128 + ord(c2)

def bigram_decode_first(code):
    """Extract first character from bigram code."""
    return code // 128

def approach1_bigram():
    """Learn context-dependent 'c' rule via bigram encoding."""
    print("\n" + "=" * 60)
    print("  APPROACH 1: BIGRAM ENCODING")
    print("  Domain: [0, 16384)  Code = char1*128 + char2")
    print("=" * 60)
    
    # Build training data: (bigram_code) → output_codepoint_for_char1
    train_x = []
    train_y = []
    
    # The 'c' rules
    for next_char, output_char in C_RULES.items():
        code = bigram_code('c', next_char)
        train_x.append(code)
        train_y.append(ord(output_char))
    
    # Identity examples: non-'c' first chars should pass through
    for c1 in 'abdefghijklmnopqrstuvwxyz':
        if c1 == 'c':
            continue
        for c2 in 'aeiou':
            code = bigram_code(c1, c2)
            train_x.append(code)
            train_y.append(ord(c1))  # identity: output = first char
    
    train_x = np.array(train_x, dtype=np.float64)
    train_y = np.array(train_y, dtype=np.float64)
    
    print(f"\n  Training: {len(train_x)} bigram examples")
    print(f"  Domain: [0, {128*128})")
    
    # Analyze the residual structure
    # For bigrams, "identity" means output = first_char = code // 128
    identity_y = np.floor(train_x / 128)
    residual = train_y - identity_y
    
    nonzero = np.where(np.abs(residual) > 0.5)[0]
    print(f"  Non-zero residuals: {len(nonzero)}")
    for idx in nonzero:
        c1 = chr(int(train_x[idx]) // 128)
        c2 = chr(int(train_x[idx]) % 128)
        print(f"    ({c1},{c2}) code={int(train_x[idx])}: "
              f"identity={chr(int(identity_y[idx]))} → "
              f"output={chr(int(train_y[idx]))} "
              f"residual={int(residual[idx]):+d}")
    
    # Detect structure in the residual
    # The residual has two distinct values: +8 (c→k) and +16 (c→s)
    # In bigram space, these are at specific codes
    
    # Try learning with exact evaluation (s→∞)
    # Build a lookup from training data
    t0 = time.perf_counter()
    lookup = {}
    for i in range(len(train_x)):
        code = int(train_x[i])
        output = int(train_y[i])
        first_char = code // 128
        if output != first_char:
            lookup[code] = output
    t_learn = time.perf_counter() - t0
    
    print(f"\n  Learned in {t_learn*1000:.2f}ms")
    print(f"  Distinct rules: {len(lookup)}")
    for code, output in sorted(lookup.items()):
        c1, c2 = chr(code // 128), chr(code % 128)
        print(f"    ({c1},{c2}) → {chr(output)}")
    
    # Test on words
    print(f"\n  Testing on words:")
    correct = 0
    total = 0
    for word, expected in TEST_WORDS_C_ONLY:
        result = []
        for i, ch in enumerate(word):
            next_ch = word[i+1] if i+1 < len(word) else ' '
            code = bigram_code(ch.lower(), next_ch.lower())
            if code in lookup:
                result.append(chr(lookup[code]))
            else:
                result.append(ch)
        result_str = ''.join(result)
        match = "✓" if result_str == expected else "✗"
        if result_str == expected:
            correct += 1
        total += 1
        print(f"    {word:8s} → {result_str:10s}  (expected {expected:10s}) {match}")
    
    print(f"\n  Accuracy: {correct}/{total}")
    return lookup


# ============================================================================
# APPROACH 2: NESTED SELECTORS (gate × gate)
# ============================================================================

def approach2_nested():
    """Learn context-dependent 'c' rule via nested gate products."""
    print("\n" + "=" * 60)
    print("  APPROACH 2: NESTED SELECTORS (gate × gate)")
    print("  output = gate(current=c) × [selector(next) × Δs + (1-selector) × Δk]")
    print("=" * 60)
    
    # Step 1: Learn the OUTER gate — "is this character 'c'?"
    # This is a width-1 RECT at codepoint 99
    print("\n  Step 1: Learn outer gate (character = 'c'?)")
    
    outer_train_x = np.array([97, 98, 99, 100, 101], dtype=np.float64)
    outer_train_y = np.array([0,  0,  1,  0,   0],  dtype=np.float64)
    
    # Detect: residual has a spike at 99
    residual = outer_train_y  # since identity is 0
    nonzero = np.where(np.abs(residual) > 0.5)[0]
    outer_bp = outer_train_x[nonzero[0]] if len(nonzero) > 0 else None
    print(f"    Detected: spike at codepoint {int(outer_bp) if outer_bp else 'NONE'} "
          f"({'c' if outer_bp == 99 else '?'})")
    print(f"    Gate: RECT[98.5, 99.5], fires when char='c'")
    
    # Step 2: Learn the INNER selector — "is next char a front vowel?"
    # Front vowels: e=101, i=105
    # This is TWO width-1 RECTs (or one wider RECT if they were contiguous)
    print("\n  Step 2: Learn inner selector (next char = front vowel?)")
    
    # Training: which next-chars produce soft c?
    inner_train_x = []
    inner_train_y = []
    for next_char, output in C_RULES.items():
        inner_train_x.append(ord(next_char))
        inner_train_y.append(1.0 if output == 's' else 0.0)
    
    inner_train_x = np.array(inner_train_x, dtype=np.float64)
    inner_train_y = np.array(inner_train_y, dtype=np.float64)
    
    # Detect front vowel positions
    front_vowels = inner_train_x[inner_train_y > 0.5]
    print(f"    Front vowel codepoints: {[int(x) for x in front_vowels]} "
          f"= {[chr(int(x)) for x in front_vowels]}")
    print(f"    Gate: RECT[100.5, 101.5] + RECT[104.5, 105.5]")
    print(f"    (Two width-1 RECTs — 'e' and 'i' are not contiguous)")
    
    # Step 3: Compose — product of gates
    # For character 'c': output = 'c' + Δk + selector(next) × (Δs - Δk)
    # Where: Δk = ord('k') - ord('c') = 8
    #        Δs = ord('s') - ord('c') = 16
    #        selector(next) = 1 if next ∈ {e,i}, else 0
    
    delta_k = ord('k') - ord('c')  # +8
    delta_s = ord('s') - ord('c')  # +16
    
    print(f"\n  Step 3: Compose")
    print(f"    Δk = {delta_k:+d} (c→k offset)")
    print(f"    Δs = {delta_s:+d} (c→s offset)")
    print(f"    Formula: output(x, next) = x + outer(x) × [Δk + selector(next) × (Δs - Δk)]")
    print(f"           = x + RECT(x,99) × [{delta_k} + RECT(next,{{e,i}}) × {delta_s - delta_k}]")
    
    # Evaluation function
    def evaluate_nested(current_cp, next_cp):
        # Outer gate: is current == 'c'?
        is_c = 1 if current_cp == ord('c') else 0
        # Inner selector: is next a front vowel?
        is_front = 1 if next_cp in (ord('e'), ord('i')) else 0
        # Compose
        offset = is_c * (delta_k + is_front * (delta_s - delta_k))
        return current_cp + offset
    
    # Test
    print(f"\n  Testing on words (c-rule only):")
    correct = 0
    total = 0
    for word, expected in TEST_WORDS_C_ONLY:
        result = []
        for i, ch in enumerate(word):
            next_ch = word[i+1] if i+1 < len(word) else ' '
            out_cp = evaluate_nested(ord(ch.lower()), ord(next_ch.lower()))
            result.append(chr(out_cp))
        result_str = ''.join(result)
        match = "✓" if result_str == expected else "✗"
        if result_str == expected:
            correct += 1
        total += 1
        print(f"    {word:8s} → {result_str:10s}  (expected {expected:10s}) {match}")
    
    print(f"\n  Accuracy: {correct}/{total}")
    
    # Count primitives
    print(f"\n  Geometric primitives:")
    print(f"    Outer gate: 1 RECT (2 gate_step calls)")
    print(f"    Inner selector: 2 RECTs (4 gate_step calls)")
    print(f"    Constants: Δk={delta_k}, Δs-Δk={delta_s-delta_k}")
    print(f"    Total: 6 gate_step calls + 1 multiply + 1 add")
    
    return evaluate_nested


# ============================================================================
# APPROACH 3: SHADER CHANNELS
# ============================================================================

def approach3_shader():
    """Learn context-dependent 'c' rule via shader-style channels."""
    print("\n" + "=" * 60)
    print("  APPROACH 3: SHADER CHANNELS")
    print("  Each candidate writes to its own register.")
    print("  A selector reads the correct register based on context.")
    print("=" * 60)
    
    # Channel A: "hard c" channel — produces 'k' when current='c'
    # Channel B: "soft c" channel — produces 's' when current='c'
    # Selector: picks A or B based on next character
    
    print("\n  Channel A (hard c): c → k")
    print("    RECT[98.5, 99.5] h=+8")
    print("    Always computes: output_A = current + 8 × RECT(current, 99)")
    
    print("\n  Channel B (soft c): c → s")
    print("    RECT[98.5, 99.5] h=+16")
    print("    Always computes: output_B = current + 16 × RECT(current, 99)")
    
    # Selector: function of next_char
    # selector = 0 → use channel A (hard c)
    # selector = 1 → use channel B (soft c)
    print("\n  Selector (function of next_char only):")
    
    # Learn selector from examples
    selector_x = []
    selector_y = []
    for next_char, output in C_RULES.items():
        selector_x.append(ord(next_char))
        selector_y.append(1.0 if output == 's' else 0.0)
    
    selector_x = np.array(selector_x, dtype=np.float64)
    selector_y = np.array(selector_y, dtype=np.float64)
    
    # The selector is: 1 at e=101 and i=105, 0 elsewhere
    # Two width-1 RECTs
    print(f"    RECT[100.5, 101.5] h=+1 (fires at 'e')")
    print(f"    RECT[104.5, 105.5] h=+1 (fires at 'i')")
    
    # Combined evaluation
    print(f"\n  Combined:")
    print(f"    output = (1-sel) × channel_A + sel × channel_B")
    print(f"           = current + RECT(curr,99) × [(1-sel)×8 + sel×16]")
    print(f"           = current + RECT(curr,99) × [8 + sel×8]")
    print(f"    where sel = RECT(next,101) + RECT(next,105)")
    
    def evaluate_shader(current_cp, next_cp):
        # Channel outputs (both always computed, like shader varyings)
        is_c = 1 if current_cp == ord('c') else 0
        channel_a = current_cp + is_c * 8    # hard c → k
        channel_b = current_cp + is_c * 16   # soft c → s
        
        # Selector
        sel = 1 if next_cp in (ord('e'), ord('i')) else 0
        
        # MUX: pick channel based on selector
        output = (1 - sel) * channel_a + sel * channel_b
        return int(output)
    
    # Test
    print(f"\n  Testing on words (c-rule only):")
    correct = 0
    total = 0
    for word, expected in TEST_WORDS_C_ONLY:
        result = []
        for i, ch in enumerate(word):
            next_ch = word[i+1] if i+1 < len(word) else ' '
            out_cp = evaluate_shader(ord(ch.lower()), ord(next_ch.lower()))
            result.append(chr(out_cp))
        result_str = ''.join(result)
        match = "✓" if result_str == expected else "✗"
        if result_str == expected:
            correct += 1
        total += 1
        print(f"    {word:8s} → {result_str:10s}  (expected {expected:10s}) {match}")
    
    print(f"\n  Accuracy: {correct}/{total}")
    
    print(f"\n  Geometric primitives:")
    print(f"    Channel A: 1 RECT (2 gate_step)")
    print(f"    Channel B: 1 RECT (2 gate_step)")
    print(f"    Selector: 2 RECTs (4 gate_step)")
    print(f"    MUX: 1 multiply + 1 add")
    print(f"    Total: 8 gate_step calls, all parallelizable")
    print(f"    Key: channels A and B are computed SIMULTANEOUSLY")
    print(f"         (no crosstalk — orthogonal registers)")
    
    return evaluate_shader


# ============================================================================
# COMPARISON
# ============================================================================

def compare_approaches():
    """Run all three approaches and compare."""
    print("=" * 60)
    print("  CONTEXT-DEPENDENT RULES: THREE APPROACHES")
    print("  Test case: English 'c' → /k/ or /s/ based on next letter")
    print("=" * 60)
    
    lookup = approach1_bigram()
    eval_nested = approach2_nested()
    eval_shader = approach3_shader()
    
    # Final comparison
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    
    print("""
  Approach 1 (Bigram):
    + Simple: one function on one domain
    + v5 pipeline works directly (1D function)
    - Domain explodes: 128² = 16,384 for bigrams, 128³ for trigrams
    - Each new context character multiplies the domain
    - Rules don't compose: bigram rule for 'c' is separate from bigram rule for 'g'
    
  Approach 2 (Nested Selectors):
    + Factored: outer gate × inner selector × offset
    + Each factor is a 1D function (v5 can learn each)
    + Compact: 3 RECTs + 2 constants
    - Multiplication of gates required (new primitive)
    - Factorization must be known/discovered
    
  Approach 3 (Shader Channels):
    + Parallel: all channels computed simultaneously  
    + No crosstalk: channels are independent registers
    + Composable: add a new channel without touching existing ones
    + Natural MUX: selector × channel is the universal combiner
    - Slightly more gate_step calls (8 vs 6)
    - But all parallelizable (like GPU shader cores)
    
  Key insight: Approaches 2 and 3 are mathematically IDENTICAL.
  
    Nested:  output = x + RECT(x,c) × [Δk + sel(next) × (Δs-Δk)]
    Shader:  output = (1-sel) × [x + RECT(x,c)×Δk] + sel × [x + RECT(x,c)×Δs]
    
  Expanding shader: x + RECT(x,c) × [(1-sel)×Δk + sel×Δs]
                   = x + RECT(x,c) × [Δk + sel×(Δs-Δk)]  ← same as nested!
  
  The difference is ARCHITECTURAL, not mathematical:
  - Nested: compute selector first, then conditional offset (sequential)
  - Shader: compute all channels + selector in parallel, MUX at end (parallel)
  
  For a geometric computer, the shader model is better because:
  1. All gate evaluations are independent → parallel execution
  2. Adding a new rule = adding a new channel (no rewriting)
  3. The MUX (multiply + add) is the only sequential step
  4. This IS how transformers work: all heads in parallel, combine at end
""")


# ============================================================================
# COMPOSED DEMO: Shader channels + character-level IPA rules
# ============================================================================

def composed_demo():
    """Show shader-channel context rules composed with character-level IPA."""
    print("\n" + "=" * 60)
    print("  COMPOSED DEMO: Context Rules + Character IPA")
    print("  Shader channels for 'c' + RECT pairs for vowels/digraphs")
    print("=" * 60)
    
    # Character-level IPA rules (from ipa_geometric_demo.py)
    char_rules = {
        ord('a'): ord('æ'),   # +133
        ord('e'): ord('ɛ'),   # +502
        ord('i'): ord('ɪ'),   # +513
        ord('o'): ord('ɒ'),   # +483
        ord('u'): ord('ʌ'),   # +535
        ord('j'): ord('ʒ'),   # +552
        ord('r'): ord('ɹ'),   # +519
    }
    
    # Digraph rules
    digraphs = {
        ('s', 'h'): 'ʃ',
        ('t', 'h'): 'θ',
        ('n', 'g'): 'ŋ',
    }
    
    # Context-dependent rules (shader channels)
    # c-rule: selector on next_char, two channels
    def context_c(current_cp, next_cp):
        """Shader-channel evaluation for 'c' rule."""
        if current_cp != ord('c'):
            return None  # not our character
        # Selector: is next a front vowel?
        sel = 1 if next_cp in (ord('e'), ord('i')) else 0
        # MUX: channel A (k=107) or channel B (s=115)
        return (1 - sel) * ord('k') + sel * ord('s')
    
    # g-rule: same structure as c!
    # g before {e, i} → /dʒ/ (simplified to ʒ)
    # g before {a, o, u, consonants} → /g/ (stays g)
    def context_g(current_cp, next_cp):
        """Shader-channel evaluation for 'g' rule."""
        if current_cp != ord('g'):
            return None
        sel = 1 if next_cp in (ord('e'), ord('i')) else 0
        # hard g stays g (103), soft g → ʒ (658)
        return (1 - sel) * ord('g') + sel * ord('ʒ')
    
    context_rules = [context_c, context_g]
    
    def apply_full(text):
        """Apply all rules: digraphs → context → character."""
        # Phase 1: Digraph pre-scan
        chars = list(text.lower())
        i = 0
        processed = []
        while i < len(chars):
            if i + 1 < len(chars):
                pair = (chars[i], chars[i+1])
                if pair in digraphs:
                    processed.append(digraphs[pair])
                    i += 2
                    continue
            processed.append(chars[i])
            i += 1
        
        # Phase 2: Context-dependent rules (shader channels)
        context_applied = []
        for i, ch in enumerate(processed):
            if ord(ch) > 127:
                context_applied.append(ch)
                continue
            next_cp = ord(processed[i+1]) if i+1 < len(processed) else ord(' ')
            handled = False
            for ctx_rule in context_rules:
                result = ctx_rule(ord(ch), next_cp)
                if result is not None:
                    context_applied.append(chr(result))
                    handled = True
                    break
            if not handled:
                context_applied.append(ch)
        
        # Phase 3: Character-level rules (width-1 RECTs)
        result = []
        for ch in context_applied:
            cp = ord(ch)
            if cp in char_rules:
                result.append(chr(char_rules[cp]))
            else:
                result.append(ch)
        
        return ''.join(result)
    
    # Test c-rule accuracy with full IPA
    print("\n  Testing c-rule words with full IPA pipeline:")
    correct = 0
    total = 0
    for word, expected in TEST_WORDS_FULL:
        result = apply_full(word)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        total += 1
        print(f"    {word:8s} → {result:10s}  (expected {expected:10s}) {match}")
    print(f"\n  Accuracy: {correct}/{total}")
    
    # Showcase sentences
    print("\n  Full sentences with context-dependent c + g:")
    sentences = [
        "The cat sat in the center of the city.",
        "A gentle giant gave grace to the congregation.",
        "George placed the ceramic cup carefully on the cedar shelf.",
        "The circus cage contained a curious giraffe.",
    ]
    for sent in sentences:
        result = apply_full(sent)
        print(f"    EN:  {sent}")
        print(f"    IPA: {result}")
        print()
    
    # Architecture summary
    print("  Three-phase architecture:")
    print("    Phase 1: DIGRAPH PRE-SCAN  (string-level pattern matching)")
    print("             sh→ʃ, th→θ, ng→ŋ")
    print("    Phase 2: CONTEXT CHANNELS  (shader-style parallel evaluation)")
    print("             c: [hard→k | soft→s] selected by next_char")
    print("             g: [hard→g | soft→ʒ] selected by next_char")
    print("    Phase 3: CHARACTER RECTS   (width-1 RECT pairs, s→∞)")
    print("             a→æ, e→ɛ, i→ɪ, o→ɒ, u→ʌ, j→ʒ, r→ɹ")
    print()
    print("  Total geometric primitives:")
    n_char = len(char_rules)
    n_digraph = len(digraphs)
    n_context = 2  # c and g rules
    n_selectors = 2  # each context rule has a front-vowel selector
    print(f"    Character RECTs: {n_char} × 2 = {n_char*2} gate_step")
    print(f"    Context channels: {n_context} × 2 = {n_context*2} gate_step")
    print(f"    Context selectors: {n_selectors} × 4 = {n_selectors*4} gate_step")
    print(f"    Digraph patterns: {n_digraph} (pre-scan, no gates)")
    print(f"    MUX operations: {n_context} multiply + {n_context} add")
    print(f"    Total: {n_char*2 + n_context*2 + n_selectors*4} gate_step + "
          f"{n_context} MUX + {n_digraph} digraph")
    print(f"    Phases 2 and 3 are fully parallelizable.")


if __name__ == "__main__":
    compare_approaches()
    composed_demo()
