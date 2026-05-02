# Plan: Geometric IPA Standalone GitHub Repository

## Overview

Extract the geometric IPA (International Phonetic Alphabet) converter from `truthspace-lcm` into a standalone, clone-and-run GitHub repository called **`geometric-ipa`**.

The project demonstrates that English-to-IPA conversion can be done with **pure geometry** — no neural network, no training, no gradient descent. Just RECT pairs (gate_step primitives) that compose additively through a four-phase pipeline.

## User Experience

```
$ git clone https://github.com/<user>/geometric-ipa.git
$ cd geometric-ipa
$ pip install -r requirements.txt
$ python ipa_demo.py
```

The demo runs 24 progressive lessons. Each lesson:
1. Explains an IPA concept (in progressively IPA-ified text — the explanation itself transforms as rules accumulate)
2. Provides training examples
3. Learns the rule geometrically (RECT pair detection in <1ms)
4. Applies ALL accumulated rules to a demo sentence

After all lessons, it showcases 10 full sentences transformed to IPA.

### Additional Modes

```
$ python ipa_demo.py                    # Full progressive lesson demo
$ python ipa_demo.py --interactive      # Type text, get IPA back (NEW)
$ python ipa_demo.py --test             # Run test suite (84/84) (NEW)
$ python auto_context_detection.py      # Run standalone context detection tests
```

## What Makes This Special

This is the **most accessible** of our standalone demos because:
- **Zero dependencies beyond numpy** — no torch, no models, no downloads
- **Instant startup** — no weight extraction, no GPU, runs in milliseconds
- **Self-teaching** — the demo teaches YOU IPA while demonstrating the geometry
- **Provably correct** — 84/84 test cases pass, every rule is transparent

## Architecture Summary

### The Four-Phase Pipeline

```
Input text: "The gentle giant gave a gift."
    │
    ▼
Phase 0: FEATURE EXTRACTION
    │  Detect magic-e (V+C+e+boundary), 'igh' trigraphs, silent final 'e'
    │  Non-local pattern scanning BEFORE character processing
    │  5 trained vowel rules (4 geared for Germanic exceptions)
    │
    ▼
Phase 1: DIGRAPH COLLAPSE
    │  13 patterns: sh→ʃ, th→θ, ng→ŋ, ch→ʧ, wh→w, ck→k, qu→kw,
    │               gh→∅, nk→ŋk, ee→iː, oo→uː, ai→eɪ, oa→oʊ
    │  4 frozen (vowel digraphs skip further processing)
    │
    ▼
Phase 2: CONTEXT CHANNELS
    │  3 auto-detected rules using gear-shift mechanism:
    │  c→k/s (selector: next_char)
    │  g→g/j (GEARED: next_char coarse + next_next_char fine, 24 examples)
    │  y→j/i (selector: is_start)
    │
    ▼
Phase 3: CHARACTER RECTS
    │  7 simple substitutions: a→æ, e→ɛ, i→ɪ, o→ɒ, u→ʌ, j→ʒ, r→ɹ
    │
    ▼
Output: "θɛ ʒɛntl ʒaɪænt geɪv æ gɪft."
```

### The Geometric Primitives

Every rule is built from **gate_step** — a width-1 RECT pair:

```python
PHI = (1 + √5) / 2
S = PHI²  # sharpness

def gate_step(x, t, s):
    """RECT pair centered at t with width 1."""
    return (ideal_gate(s*(x-(t-0.5))) - ideal_gate(s*(x-(t+0.5)))) / s
```

A character substitution a→æ is: `output = input + height × gate_step(input, ord('a'), S)`

At the integer codepoint resolution, gate_step becomes an exact indicator function — the geometry IS the computation.

### The Gear-Shift Mechanism

For context-dependent rules (like soft/hard g), the system auto-discovers:

- **Coarse gear**: Which context variable (prev_char, next_char, etc.) best explains the variation? Uses information gain (entropy reduction).
- **Fine gear**: For ambiguous teeth on the coarse gear, which secondary variable resolves them?

Example for 'g':
```
Coarse gear: next_char
  → 'g' when next_char ∈ {a, o, u, l, r, ...}  (hard g)
  → 'j' when next_char ∈ {e, y}                  (soft g)
  → AMBIGUOUS when next_char = 'i'               (gift vs gin)

Fine gear (engages when next_char='i'): next_next_char
  → 'g' when next_next_char ∈ {f, r, v}          (gift, girl, give)
  → 'j' when next_next_char ∈ {a, n, s, l, g}    (giant, gin, gist)
```

This is discovered automatically from 24 training word pairs. No hard-coding.

## Source Files from truthspace-lcm

### Core Files

#### 1. `ipa_demo.py` — Main demo (cleaned from ipa_geometric_demo.py)
- **Source:** `/home/thorin/truthspace-lcm/phi_geometric/evaluations/ipa_geometric_demo.py` (1362 lines)
- **What it provides:**
  - `GeometricRule` — single character substitution as RECT pair
  - `GeometricProgram` — four-phase pipeline compositor
  - `detect_magic_e()`, `detect_igh()`, `detect_silent_final_e()` — Phase 0 feature extractors
  - `learn_magic_e_rules()`, `apply_magic_e_rule()` — trained exception handling
  - `LESSONS` — 24 progressive IPA lessons with training data
  - `run_demo()` — the interactive lesson runner
- **Changes needed:**
  - Fix import: `from phi_geometric.evaluations.auto_context_detection import ...` → `from auto_context_detection import ...`
  - Add `--interactive` mode: user types text, gets IPA back in a loop
  - Add `--test` mode: run the 84/84 test suite from the memory
  - Remove any remaining `sys.path` manipulation if present

#### 2. `auto_context_detection.py` — Gear-shift framework
- **Source:** `/home/thorin/truthspace-lcm/phi_geometric/evaluations/auto_context_detection.py` (938 lines)
- **What it provides:**
  - `extract_context_at()` — context dict extraction for any character position
  - `extract_contexts()` — batch context extraction for word pairs
  - `detect_inconsistencies()` — find characters with multiple output mappings
  - `discover_selector()` — information-gain based selector discovery
  - `discover_gears()` — two-level gear train discovery (coarse + fine)
  - `build_rules()` — full auto-rule builder from training pairs
  - `GeometricRule` (auto version) — identity/simple/context/geared rule types
  - `AutoGeometricProgram` — program built from auto-detected rules
  - `test_c_rule()`, `test_g_rule()`, `test_combined()` — standalone tests
- **Changes needed:**
  - None structurally — this file is already self-contained
  - The tests in `__main__` serve as both documentation and verification

## Repository Structure

```
geometric-ipa/
├── README.md
├── LICENSE                          # MIT
├── requirements.txt
├── .gitignore
├── ipa_demo.py                      # Main demo (from ipa_geometric_demo.py)
├── auto_context_detection.py        # Gear-shift framework (unchanged)
└── tests/
    └── test_ipa.py                  # Test suite: 84/84 cases (NEW)
```

That's it. Two Python files + a test file. No weights, no models, no downloads.

## requirements.txt

```
numpy
```

That's the entire dependency list. The project uses only:
- `numpy` — for the gate function math (φ, sigmoid, clip)
- Python stdlib: `collections.defaultdict`, `time`, `sys`

## Test Suite

The test file should verify all 84 test cases. Based on the memory, these cover:

```python
# tests/test_ipa.py
test_cases = [
    # Germanic magic-e exceptions
    ("come", "kʌm"),   ("love", "lʌv"),   ("have", "hæv"),
    ("give", "gɪv"),   ("shove", "ʃʌv"),  ("above", "æbʌv"),
    
    # g-before-e exceptions  
    ("get", ...),      ("gear", ...),     ("geese", ...),
    
    # igh words
    ("light", ...),    ("night", ...),    ("right", ...),
    ("high", ...),     ("bright", ...),   ("sight", ...),
    
    # Silent final e
    ("dance", ...),    ("prince", ...),   ("voice", ...),
    ("choice", ...),   ("noise", ...),    ("once", ...),
    
    # Case-sensitive
    ("Some", "sʌm"),   ("Come", "kʌm"),   ("Light", "laɪt"),
    
    # r-controlled magic-e
    ("there", "θɛɹ"),  ("where", "wɛɹ"),  ("here", "hɛɹ"),
]
```

Run with: `python -m pytest tests/` or `python ipa_demo.py --test`

## README.md Outline

```markdown
# geometric-ipa: English to IPA via Pure Geometry

Converts English text to the International Phonetic Alphabet using
geometric primitives — no neural network, no training, no gradient descent.

## What is this?

A program that learns 29 phonetic rules from examples, each rule being
a geometric RECT pair (gate_step primitive). Rules compose through a
four-phase pipeline that handles digraphs, context-dependent sounds,
magic-e, and simple substitutions.

    EN:  The bright light shone right there in the night.
    IPA: θɛ bɹaɪt laɪt ʃoʊn ɹaɪt θɛɹ ɪn θɛ naɪt.

    EN:  Some love to dance but none have a choice in the voice.
    IPA: sʌm lʌv tɒ dæns bʌt nʌn hæv æ ʧɒɪs ɪn θɛ vɒɪs.

## Quick Start

    git clone https://github.com/<user>/geometric-ipa.git
    cd geometric-ipa
    pip install -r requirements.txt
    python ipa_demo.py

## Interactive Mode

    python ipa_demo.py --interactive

Type any English text and see IPA output in real-time.

## How It Works

### The Core Primitive

Every rule is a RECT pair — two gate_step calls that activate at
exactly one codepoint:

    rule(x) = height × gate_step(x, target_codepoint, φ²)

This is a geometric selector: it outputs `height` when x equals
the target, and 0 everywhere else. Character substitution = 
addition of these selectors.

### Four-Phase Pipeline

1. **Feature Extract** — scan for non-local patterns (magic-e, igh)
2. **Digraph Collapse** — merge character pairs (sh→ʃ, th→θ)
3. **Context Channels** — auto-detected gear-shift rules (c→k/s, g→g/j)
4. **Character Rects** — simple substitutions (a→æ, e→ɛ)

### The Gear-Shift Mechanism

When a character has context-dependent pronunciation (like 'g'), the
system automatically discovers which context variable explains the
variation using information gain:

- **Coarse gear**: next_char distinguishes hard g (before a,o,u) 
  from soft g (before e,y)
- **Fine gear**: next_next_char resolves the ambiguous 'i' case
  (gift=hard, gin=soft)

All discovered from 24 training examples. No hard-coded rules.

## Statistics

| Metric | Value |
|--------|-------|
| Rules | 29 |
| Geometric primitives | 159 gate_step calls |
| Test accuracy | 84/84 (100%) |
| External dependencies | numpy |
| Model weights | 0 bytes |
| Training time | <10ms total |
| Gradient descent | none |
| Neural network | none |

## The Golden Ratio Connection

The gate sharpness parameter is φ² (phi squared), where φ = (1+√5)/2.
The ideal_gate function uses:

    f(x) = √(8/π) · x · (1 + ((4-π)/(6π)) · x²)
    gate(x) = x · σ(f(x))

This is the same φ-soft gate used in our geometric colorizer and 
Qwen2-7B reverse engineering — the golden ratio appears as the 
optimal sharpness for geometric computation.

## Known Limitations

- 'ou' digraph too irregular (house ≠ through ≠ would)
- '-nge' words (change, strange — ng digraph fires before soft-g)
- Suffixes (making, nothing — need morphological decomposition)
- Voiced/voiceless th distinction (both map to θ)

## License

MIT

## Credits

Part of the [TruthSpace Geometric LCM](https://github.com/<user>/truthspace-lcm) project.
Theory documented in Parts 35-39 of the Geometric φ-Map (Doc 247).
```

## Key Differences from Other Standalone Repos

| Aspect | phi-depth | geometric-colorizer | **geometric-ipa** |
|--------|-----------|--------------------|--------------------|
| Dependencies | torch, transformers, cv2 | torch, cv2, huggingface | **numpy only** |
| Model download | ~400MB DA2 | ~200MB DDColor | **none** |
| Weight files | 125 bytes | ~204 MB | **0 bytes** |
| GPU needed | recommended | recommended | **no** |
| Startup time | ~10s | ~15s | **<1s** |
| Input | webcam | webcam | text |
| Output | depth map | colorized image | IPA transcription |
| Core insight | φ-decoder | φ-soft gate + color matrix | RECT pairs + gear-shift |

## Implementation Steps (for the new chat)

1. **Create repo structure** — `mkdir geometric-ipa && cd geometric-ipa`
2. **Copy and clean `ipa_demo.py`** from `ipa_geometric_demo.py`:
   - Fix import path to local `auto_context_detection`
   - Add `--interactive` mode (loop: input text → print IPA)
   - Add `--test` mode (run all 84 test cases, report results)
   - Add argparse for mode selection
3. **Copy `auto_context_detection.py`** — essentially unchanged
4. **Write `tests/test_ipa.py`** — 84 test cases covering all rule types
5. **Write `requirements.txt`** — just `numpy`
6. **Write `README.md`**
7. **Write `.gitignore`** — `__pycache__/`, `*.pyc`, `.pytest_cache/`
8. **Test** — `python ipa_demo.py` should produce full lesson output, `python ipa_demo.py --test` should pass 84/84

## Source File Reference

| File | Location in truthspace-lcm | Role |
|------|---------------------------|------|
| IPA demo | `phi_geometric/evaluations/ipa_geometric_demo.py` | Main demo (1362 lines) |
| Auto-context | `phi_geometric/evaluations/auto_context_detection.py` | Gear-shift framework (938 lines) |
| Theory (Parts 35-39) | `docs/design_considerations/247_geometric_phi_map.md` | Design rationale |
| Detection v5 | `phi_geometric/evaluations/detection_v5.py` | Gate infrastructure (referenced, not needed) |

## Interactive Mode Design

```python
def interactive_mode():
    """REPL for English → IPA conversion."""
    # Build the full program (all 24 lessons, silently)
    program = build_full_program()  # extracted from run_demo()
    
    print("Geometric IPA Converter")
    print("Type English text, get IPA back. Ctrl+C to quit.\n")
    
    while True:
        try:
            text = input("EN:  ")
            if not text.strip():
                continue
            ipa = program.apply_text(text)
            print(f"IPA: {ipa}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
```

This requires extracting a `build_full_program()` function from `run_demo()` that constructs all rules without printing the lesson output.
