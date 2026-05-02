#!/usr/bin/env python3
"""
Pixel Art Palette Styling — PhaseDiscovery in the Image Domain
==============================================================

Demonstrates that PhaseDiscovery generalizes beyond text/IPA to any
domain with discrete tokens. Here, tokens are COLOR NAMES representing
pixels in a scanline, and the transformation is a "style transfer"
that has the same phase structure as English→IPA:

  Phase 1 (collapse):  Adjacent color blending  (like sh→ʃ)
      red + yellow → orange    (warm blend, 2→1)
      blue + green → teal      (cool blend, 2→1)
      white + white → silver   (bright merge, 2→1)

  Phase 2 (context):   Neighbor-dependent shading  (like c→k/s)
      gray → dark_gray   when next to black
      gray → light_gray  when next to white/silver

  Phase 3 (simple):    Palette shift  (like a→æ)
      red → crimson
      blue → navy
      green → forest
      yellow → gold
      black → charcoal

The system discovers these phases automatically from training pairs,
with zero knowledge of color theory or image processing.
"""

import sys
import os
import importlib.util
import types

# Direct import to bypass top-level phi_geometric/__init__.py (requires torch)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_core_path = os.path.join(_project_root, 'phi_geometric', 'core')

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub the package paths so relative imports resolve
_pkg = types.ModuleType('phi_geometric')
_pkg.__path__ = [os.path.join(_project_root, 'phi_geometric')]
sys.modules['phi_geometric'] = _pkg

_core = types.ModuleType('phi_geometric.core')
_core.__path__ = [_core_path]
sys.modules['phi_geometric.core'] = _core

_load_module('phi_geometric.core.discovery',
             os.path.join(_core_path, 'discovery.py'))
_load_module('phi_geometric.core.cascade_navigator',
             os.path.join(_core_path, 'cascade_navigator.py'))
_pd_mod = _load_module('phi_geometric.core.phase_discovery',
                        os.path.join(_core_path, 'phase_discovery.py'))

PhaseDiscovery = _pd_mod.PhaseDiscovery


# ============================================================================
# GROUND TRUTH STYLE RULES (used to GENERATE training data, not seen by model)
# ============================================================================

# Phase 1: Adjacent color blending (multi-token collapse)
BLEND_RULES = {
    ('red', 'yellow'): ('orange',),
    ('yellow', 'red'): ('orange',),
    ('blue', 'green'): ('teal',),
    ('green', 'blue'): ('teal',),
    ('white', 'white'): ('silver',),
}

# Phase 2: Context-dependent shading
# Uses NEXT neighbor only — clean single-variable selector
# (parallels IPA c→k/s which uses next_char)
def shade_gray(pixel, ctx):
    """Gray shading depends on the NEXT pixel in the scanline."""
    nxt = ctx.get('next')
    if nxt in ('black', 'charcoal'):
        return 'dark_gray'
    if nxt in ('white', 'silver'):
        return 'light_gray'
    return 'gray'  # mid-tone or end-of-line — no change

# Phase 3: Simple palette shift
PALETTE_SHIFT = {
    'red': 'crimson',
    'blue': 'navy',
    'green': 'forest',
    'yellow': 'gold',
    'black': 'charcoal',
}


def _find_orig_pos(original, collapsed, collapsed_idx):
    """Map a collapsed-sequence index back to the original-sequence index."""
    # Walk through original, tracking where collapses happen
    oi = 0  # original index
    ci = 0  # collapsed index
    while oi < len(original) and ci < len(collapsed):
        if ci == collapsed_idx:
            return oi
        # Check if a blend starts here
        if oi + 1 < len(original):
            pair = (original[oi], original[oi + 1])
            if pair in BLEND_RULES:
                oi += 2
                ci += 1
                continue
        oi += 1
        ci += 1
    if ci == collapsed_idx and oi < len(original):
        return oi
    return None


def apply_ground_truth(scanline):
    """Apply the ground truth style transformation to a pixel scanline.
    
    This is the "oracle" — it knows the rules. PhaseDiscovery must
    rediscover them from input/output pairs alone.
    """
    # Phase 1: Collapse blends (scan left-to-right, longest match first)
    result = []
    orig = list(scanline)  # preserve original for context
    i = 0
    while i < len(scanline):
        if i + 1 < len(scanline):
            pair = (scanline[i], scanline[i + 1])
            if pair in BLEND_RULES:
                result.extend(BLEND_RULES[pair])
                i += 2
                continue
        result.append(scanline[i])
        i += 1
    
    # Phase 2: Context-dependent shading
    # Uses NEXT pixel as context (from pre-collapse original scanline)
    shaded = list(result)
    for i, pixel in enumerate(result):
        if pixel == 'gray':
            # Find this gray's position in the original scanline
            # and use the original NEXT pixel for context
            orig_pos = _find_orig_pos(scanline, result, i)
            ctx = {}
            if orig_pos is not None and orig_pos + 1 < len(scanline):
                ctx['next'] = scanline[orig_pos + 1]
            shaded[i] = shade_gray(pixel, ctx)
    
    # Phase 3: Simple palette shift
    final = []
    for pixel in shaded:
        final.append(PALETTE_SHIFT.get(pixel, pixel))
    
    return final


# ============================================================================
# GENERATE TRAINING DATA
# ============================================================================

# Scanlines representing pixel rows from various "images"
TRAINING_SCANLINES = [
    # Warm gradient with blend
    ['red', 'yellow', 'orange', 'red'],
    ['yellow', 'red', 'orange', 'yellow'],
    ['red', 'yellow', 'green', 'blue'],
    
    # Cool gradient with blend
    ['blue', 'green', 'white', 'black'],
    ['green', 'blue', 'red', 'yellow'],
    ['blue', 'green', 'red', 'yellow'],
    
    # White merge
    ['white', 'white', 'gray', 'black'],
    ['black', 'white', 'white', 'red'],
    ['white', 'white', 'white', 'white'],
    
    # Gray shading: gray before black → dark_gray  (like c before a/o/u → k)
    ['gray', 'black', 'red'],
    ['red', 'gray', 'black'],
    ['blue', 'gray', 'black', 'red'],
    ['gray', 'black', 'gray', 'white'],
    
    # Gray shading: gray before white → light_gray  (like c before i/e → s)
    ['gray', 'white', 'red'],
    ['red', 'gray', 'white'],
    ['blue', 'gray', 'white', 'red'],
    ['gray', 'white', 'gray', 'black'],
    
    # Gray with no relevant next → identity (gray stays gray)
    ['red', 'gray', 'blue'],
    ['gray', 'red', 'blue'],
    
    # Simple palette shifts (no blending, no context)
    ['red', 'blue', 'green'],
    ['blue', 'red', 'yellow'],
    ['green', 'yellow', 'black'],
    ['yellow', 'green', 'red'],
    ['black', 'red', 'blue'],
    ['red', 'green', 'black'],
    
    # Mixed: blending + context + palette
    ['red', 'yellow', 'gray', 'black'],
    ['blue', 'green', 'gray', 'white'],
    ['white', 'white', 'gray', 'black'],
    ['black', 'gray', 'black', 'white'],
    ['green', 'blue', 'gray', 'black', 'white'],
    
    # More blend evidence (need ≥2 per pattern)
    ['red', 'yellow', 'black'],
    ['yellow', 'red', 'white'],
    ['blue', 'green', 'yellow'],
    ['green', 'blue', 'red'],
    ['white', 'white', 'red'],
    ['white', 'white', 'blue'],
]


def main():
    print("=" * 70)
    print("  PIXEL ART PALETTE STYLING — PhaseDiscovery in the Image Domain")
    print("=" * 70)
    print()
    
    # Generate training pairs
    training_pairs = []
    for scanline in TRAINING_SCANLINES:
        output = apply_ground_truth(scanline)
        training_pairs.append((scanline, output))
    
    # Show some examples
    print("Sample training pairs:")
    print("-" * 50)
    for inp, out in training_pairs[:8]:
        inp_str = ' '.join(f'{c:>8s}' for c in inp)
        out_str = ' '.join(f'{c:>8s}' for c in out)
        print(f"  IN:  {inp_str}")
        print(f"  OUT: {out_str}")
        print()
    
    print(f"Total training pairs: {len(training_pairs)}")
    print()
    
    # ================================================================
    # Run PhaseDiscovery
    # ================================================================
    print("=" * 70)
    print("  RUNNING PHASE DISCOVERY")
    print("=" * 70)
    print()
    
    pd = PhaseDiscovery(context_window=1)
    for inp, out in training_pairs:
        pd.add_pair(inp, out)
    
    result = pd.discover()
    print(result.describe())
    
    # ================================================================
    # Build navigator and test
    # ================================================================
    print("=" * 70)
    print("  EXECUTABLE PIPELINE")
    print("=" * 70)
    print()
    
    nav = result.to_navigator()
    print(nav.describe())
    print()
    
    # ================================================================
    # Validate on training data
    # ================================================================
    print("=" * 70)
    print("  VALIDATION ON TRAINING DATA")
    print("=" * 70)
    print()
    
    correct = 0
    total = len(training_pairs)
    errors = []
    
    for inp, expected_out in training_pairs:
        trace = nav.execute(inp)
        actual = trace.output_elements
        if actual == expected_out:
            correct += 1
        else:
            errors.append((inp, expected_out, actual))
    
    print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print()
    
    if errors:
        print(f"  Errors ({len(errors)}):")
        for inp, expected, actual in errors[:10]:
            print(f"    IN:       {' '.join(inp)}")
            print(f"    EXPECTED: {' '.join(expected)}")
            print(f"    ACTUAL:   {' '.join(actual)}")
            print()
    
    # ================================================================
    # Test on UNSEEN scanlines
    # ================================================================
    print("=" * 70)
    print("  GENERALIZATION — UNSEEN SCANLINES")
    print("=" * 70)
    print()
    
    unseen = [
        ['red', 'yellow', 'blue', 'green'],       # two blends in one line
        ['white', 'white', 'white', 'white', 'white'],  # triple merge
        ['black', 'gray', 'gray', 'white'],        # context shading
        ['green', 'blue', 'gray', 'black', 'red', 'yellow'],  # everything
        ['yellow', 'red', 'gray', 'white', 'white'],
        ['blue', 'green', 'green', 'blue'],
    ]
    
    unseen_correct = 0
    for scanline in unseen:
        expected = apply_ground_truth(scanline)
        trace = nav.execute(scanline)
        actual = trace.output_elements
        match = actual == expected
        if match:
            unseen_correct += 1
        status = '✓' if match else '✗'
        
        print(f"  IN:       {' '.join(scanline)}")
        print(f"  EXPECTED: {' '.join(expected)}")
        print(f"  ACTUAL:   {' '.join(actual)}  {status}")
        print()
    
    print(f"  Generalization: {unseen_correct}/{len(unseen)}")
    print()
    
    # ================================================================
    # Summary
    # ================================================================
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print("  PhaseDiscovery was given pixel scanlines as (input, output) pairs")
    print("  with zero knowledge of color theory or image processing.")
    print()
    print("  Discovered structure:")
    for phase in result.phases:
        if phase.multi_token_patterns:
            print(f"    • Collapse phase: {len(phase.multi_token_patterns)} "
                  f"blend patterns (like digraphs)")
            for mp in phase.multi_token_patterns:
                inp_s = '+'.join(str(t) for t in mp.input_tokens)
                out_s = '+'.join(str(t) for t in mp.output_tokens)
                print(f"        {inp_s} → {out_s}")
        elif phase.context_dependent:
            n_ctx = len(phase.rule_observations)
            print(f"    • Context phase: {n_ctx} shading rules "
                  f"(like context-dependent c→k/s)")
            for tok, obs_list in phase.rule_observations.items():
                from collections import Counter
                counts = Counter(out for out, _ in obs_list)
                out_str = ', '.join(f'{o}×{c}' for o, c in counts.most_common())
                print(f"        {tok} → {{{out_str}}}")
        else:
            n_tok = len(phase.token_rules)
            print(f"    • Token map phase: {n_tok} palette shifts "
                  f"(like simple a→æ)")
            for inp, out in sorted(phase.token_rules.items(), key=str):
                out_str = ''.join(str(t) for t in out)
                print(f"        {inp} → {out_str}")
    
    print()
    print(f"  Training accuracy:      {correct}/{total}")
    print(f"  Generalization accuracy: {unseen_correct}/{len(unseen)}")


if __name__ == '__main__':
    main()
