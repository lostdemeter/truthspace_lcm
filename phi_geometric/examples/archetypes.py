"""
Transformation Archetype Examples
=================================

Eight ready-to-use archetype templates for PhaseDiscovery.
Each archetype is a function that returns (training_pairs, ground_truth_fn).

Usage:
    from phi_geometric.examples.archetypes import get_archetype

    pairs, apply_fn = get_archetype('expand_map')
    
    pd = PhaseDiscovery(context_window=1)
    for inp, out in pairs:
        pd.add_pair(inp, out)
    result = pd.discover()
    nav = result.to_navigator()

Or use the quick API:
    from phi_geometric.examples.archetypes import discover_archetype

    result, nav = discover_archetype('collapse_context_map')
    print(result.archetype)  # 'collapse_context_map'

Available archetypes:
    A: map                    - Pure substitution
    B: context_map            - Neighbor-dependent + substitution
    C: collapse_map           - Token merging + substitution
    D: collapse_context_map   - Full cascade (IPA-like)
    E: expand_map             - Token expansion + substitution
    F: expand_collapse_map    - Expansion + merging + substitution
    G: expand_context_map     - Expansion + neighbor-dependent + substitution
    H: geometric_context_map  - Long-range context via φ-decay (vowel harmony)
"""

from phi_geometric.core.phase_discovery import PhaseDiscovery


# ============================================================================
# A: MAP — Pure substitution
# ============================================================================

def _map_rules():
    """Elvish Cipher: every rune maps to exactly one other rune."""
    MAP = {
        'ash': 'mir', 'oak': 'tel', 'elm': 'ven', 'yew': 'dor',
        'ivy': 'sal', 'bay': 'nul', 'fig': 'por', 'rue': 'kef',
    }
    def apply(seq):
        return [MAP.get(t, t) for t in seq]

    training = [
        ['ash', 'oak', 'elm'], ['yew', 'ivy', 'bay'],
        ['fig', 'rue', 'ash'], ['oak', 'elm', 'yew', 'ivy'],
        ['bay', 'fig', 'rue', 'ash'], ['elm', 'yew', 'fig'],
        ['ash', 'ivy', 'rue'], ['oak', 'bay', 'elm', 'fig'],
        ['yew', 'ash', 'oak'], ['rue', 'bay', 'ivy', 'yew'],
        ['fig', 'ash', 'bay'], ['ivy', 'elm', 'oak', 'rue'],
        ['ash', 'fig', 'yew', 'bay'], ['oak', 'rue', 'ivy'],
        ['elm', 'bay', 'ash', 'oak'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# B: CONTEXT→MAP — Neighbor-dependent encoding
# ============================================================================

def _context_map_rules():
    """Traffic Signals: yellow depends on next sensor."""
    def apply(seq):
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

    training = [
        ['red', 'green', 'blue'], ['green', 'red', 'yellow'],
        ['yellow', 'red', 'green'], ['yellow', 'green', 'red'],
        ['red', 'yellow', 'red'], ['green', 'yellow', 'green'],
        ['blue', 'yellow', 'red', 'green'],
        ['yellow', 'red', 'yellow', 'green'],
        ['red', 'yellow', 'green', 'blue'],
        ['green', 'blue', 'yellow', 'red'],
        ['yellow', 'blue', 'red'], ['blue', 'red', 'yellow', 'green'],
        ['red', 'green', 'yellow', 'red'],
        ['green', 'yellow', 'red', 'blue'],
        ['blue', 'yellow', 'green', 'red'],
        ['red', 'blue', 'green'], ['green', 'red', 'blue'],
        ['yellow', 'green', 'blue'],
        ['blue', 'green', 'yellow', 'red'],
        ['red', 'yellow', 'blue', 'green'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# C: COLLAPSE→MAP — Token merging + substitution
# ============================================================================

def _collapse_map_rules():
    """Musical Chords: note pairs collapse to chord names."""
    CHORDS = {
        ('C', 'E'): ('Cmaj',), ('D', 'F'): ('Dmin',),
        ('E', 'G'): ('Emin',), ('G', 'B'): ('Gmaj',),
    }
    MAP = {'A': 'La', 'F': 'Fa', 'C': 'Do'}

    def apply(seq):
        collapsed = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq):
                pair = (seq[i], seq[i + 1])
                if pair in CHORDS:
                    collapsed.extend(CHORDS[pair])
                    i += 2
                    continue
            collapsed.append(seq[i])
            i += 1
        return [MAP.get(t, t) for t in collapsed]

    training = [
        ['C', 'E', 'A'], ['D', 'F', 'A'], ['E', 'G', 'A'],
        ['G', 'B', 'A'], ['C', 'E', 'D', 'F'], ['G', 'B', 'C', 'E'],
        ['A', 'C', 'E'], ['A', 'D', 'F'], ['F', 'G', 'B'],
        ['C', 'E', 'G', 'B'], ['A', 'E', 'G', 'F'],
        ['D', 'F', 'G', 'B', 'A'],
        ['A', 'F', 'C'], ['C', 'A', 'F'], ['F', 'A', 'C'],
        ['A', 'C', 'F', 'A'], ['C', 'F', 'A'], ['F', 'C', 'A', 'F'],
        ['C', 'E', 'F'], ['D', 'F', 'C'], ['E', 'G', 'F'],
        ['G', 'B', 'C'], ['A', 'C', 'E', 'A'], ['F', 'D', 'F', 'A'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# D: COLLAPSE→CONTEXT→MAP — Full cascade
# ============================================================================

def _collapse_context_map_rules():
    """Alien Language: geminates, devoicing, voicing shift."""
    def _orig_next(original, collapsed, ci):
        oi, c = 0, 0
        while oi < len(original) and c < len(collapsed):
            if c == ci:
                r = oi
                if r + 1 < len(original):
                    if original[r] == 'z' and original[r+1] == 'z':
                        r += 2
                    elif original[r] == 'k' and original[r+1] == 'h':
                        r += 2
                    else:
                        r += 1
                else:
                    r += 1
                return original[r] if r < len(original) else None
            if oi + 1 < len(original):
                if original[oi] == 'z' and original[oi+1] == 'z':
                    oi += 2; c += 1; continue
                if original[oi] == 'k' and original[oi+1] == 'h':
                    oi += 2; c += 1; continue
            oi += 1; c += 1
        return None

    def apply(seq):
        collapsed = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq):
                if seq[i] == 'z' and seq[i+1] == 'z':
                    collapsed.append('Z'); i += 2; continue
                if seq[i] == 'k' and seq[i+1] == 'h':
                    collapsed.append('X'); i += 2; continue
            collapsed.append(seq[i]); i += 1
        voiced = list(collapsed)
        for i, tok in enumerate(collapsed):
            if tok == 'v':
                nxt = _orig_next(seq, collapsed, i)
                if nxt in ('p', 't', 'k'):
                    voiced[i] = 'f'
        voicing = {'p': 'b', 't': 'd', 'k': 'g'}
        return [voicing.get(t, t) for t in voiced]

    training = [
        ['z', 'z', 'a'], ['a', 'z', 'z'], ['k', 'h', 'a'],
        ['a', 'k', 'h'], ['z', 'z', 'k', 'h'], ['k', 'h', 'z', 'z'],
        ['a', 'z', 'z', 'a'], ['a', 'k', 'h', 'a'],
        ['v', 'p', 'a'], ['v', 't', 'a'], ['v', 'k', 'a'],
        ['a', 'v', 'p'], ['a', 'v', 't'], ['a', 'v', 'k'],
        ['v', 'a', 'p'], ['a', 'v', 'a'], ['v', 'v', 'a'],
        ['p', 'a', 't'], ['t', 'a', 'k'], ['k', 'a', 'p'],
        ['a', 'p', 'a'], ['a', 't', 'a'], ['a', 'k', 'a'],
        ['p', 't', 'k'], ['a', 'p', 't', 'k'],
        ['z', 'z', 'v', 'p'], ['k', 'h', 'v', 't'],
        ['a', 'z', 'z', 'v', 'k', 'a'],
        ['v', 'p', 'z', 'z', 'a'], ['a', 'v', 'a', 'p', 't'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# E: EXPAND→MAP — Token expansion + substitution
# ============================================================================

def _expand_map_rules():
    """Phonetic Spelling: x→ks, q→kw."""
    MAP = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E'}

    def apply(seq):
        result = []
        for tok in seq:
            if tok == 'x':
                result.extend(['k', 's'])
            elif tok == 'q':
                result.extend(['k', 'w'])
            else:
                result.append(MAP.get(tok, tok))
        return result

    training = [
        ['a', 'x', 'b'], ['c', 'x', 'd'], ['e', 'x', 'a'],
        ['b', 'x', 'c'], ['a', 'q', 'b'], ['c', 'q', 'd'],
        ['e', 'q', 'a'], ['b', 'q', 'c'], ['x', 'a', 'q'],
        ['q', 'x', 'a'],
        ['a', 'b', 'c'], ['d', 'e', 'a'], ['b', 'c', 'd'],
        ['c', 'a', 'e'], ['e', 'd', 'b'], ['a', 'd', 'c'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# F: EXPAND→COLLAPSE→MAP — Expansion + merging + substitution
# ============================================================================

def _expand_collapse_map_rules():
    """Chemical Notation: W→H2+O, H+H→H2, charge annotation."""
    COLLAPSES = {('H', 'H'): ('H2',), ('O', 'O'): ('O2',)}
    EXPANDS = {'W': ['H2', 'O']}
    MAP = {'Na': 'Na+', 'Cl': 'Cl-', 'K': 'K+'}

    def apply(seq):
        expanded = []
        for tok in seq:
            if tok in EXPANDS:
                expanded.extend(EXPANDS[tok])
            else:
                expanded.append(tok)
        collapsed = []
        i = 0
        while i < len(expanded):
            if i + 1 < len(expanded):
                pair = (expanded[i], expanded[i + 1])
                if pair in COLLAPSES:
                    collapsed.extend(COLLAPSES[pair])
                    i += 2
                    continue
            collapsed.append(expanded[i])
            i += 1
        return [MAP.get(t, t) for t in collapsed]

    training = [
        ['Na', 'W'], ['W', 'Cl'], ['K', 'W'], ['W', 'Na'],
        ['Na', 'W', 'Cl'], ['K', 'W', 'Na'],
        ['H', 'H', 'Na'], ['Na', 'H', 'H'], ['O', 'O', 'Cl'],
        ['Cl', 'O', 'O'], ['H', 'H', 'O', 'O'], ['K', 'H', 'H'],
        ['H', 'H', 'K'], ['O', 'O', 'Na'],
        ['Na', 'Cl', 'K'], ['K', 'Na', 'Cl'], ['Cl', 'K', 'Na'],
        ['Na', 'K', 'Cl'], ['Cl', 'Na', 'K'], ['K', 'Cl', 'Na'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# G: EXPAND→CONTEXT→MAP — Expansion + neighbor-dependent + substitution
# ============================================================================

def _expand_context_map_rules():
    """Morse-like Encoding: X→dd, s stressed before d."""
    MAP = {'d': 'D', 'a': 'A', 'b': 'B'}

    def apply(seq):
        expanded = []
        for tok in seq:
            if tok == 'X':
                expanded.extend(['d', 'd'])
            else:
                expanded.append(tok)
        result = []
        for i, tok in enumerate(expanded):
            if tok == 's':
                nxt = expanded[i + 1] if i + 1 < len(expanded) else None
                result.append('S' if nxt == 'd' else 's')
            else:
                result.append(tok)
        return [MAP.get(t, t) for t in result]

    training = [
        ['a', 'X', 'b'], ['b', 'X', 'a'], ['a', 'X', 'a'],
        ['b', 'X', 'b'],
        ['s', 'd', 'a'], ['a', 's', 'd'], ['s', 'd', 'b'],
        ['b', 's', 'd'],
        ['s', 'a', 'd'], ['s', 'b', 'a'], ['a', 's', 'a'],
        ['a', 's', 'b'],
        ['a', 'b', 'd'], ['d', 'a', 'b'], ['b', 'd', 'a'],
        ['d', 'b', 'a'], ['a', 'd', 'b'], ['b', 'a', 'd'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# H: GEOMETRIC CONTEXT MAP — Long-range φ-decay context (vowel harmony)
# ============================================================================

def _geometric_context_map_rules():
    """Archetype H: Context dependency beyond immediate neighbors.
    
    Vowel harmony: 'a' changes based on nearest PRECEDING vowel,
    which can be 2-4 positions away (separated by consonants).
    Requires geometric=True for PhaseDiscovery.
    """
    cmap = {'c': 'C', 'd': 'D', 'f': 'F', 'g': 'G', 'e': 'e', 'o': 'o'}
    
    def nearest_prev_vowel(seq, pos):
        for i in range(pos - 1, -1, -1):
            if seq[i] in ('e', 'o'):
                return seq[i]
        return None
    
    def apply(seq):
        result = []
        for i, tok in enumerate(seq):
            if tok == 'a':
                v = nearest_prev_vowel(seq, i)
                if v == 'e':
                    result.append('æ')
                elif v == 'o':
                    result.append('ɑ')
                else:
                    result.append('a')
            else:
                result.append(cmap.get(tok, tok))
        return result
    
    training = [
        ['e', 'c', 'a'], ['o', 'c', 'a'], ['e', 'd', 'a'], ['o', 'd', 'a'],
        ['e', 'f', 'a'], ['o', 'f', 'a'],
        ['e', 'c', 'd', 'a'], ['o', 'c', 'd', 'a'],
        ['e', 'f', 'g', 'a'], ['o', 'f', 'g', 'a'],
        ['e', 'c', 'd', 'f', 'a'], ['o', 'c', 'd', 'f', 'a'],
        ['c', 'a', 'd'], ['d', 'a', 'f'], ['f', 'a', 'g'], ['c', 'd', 'a'],
        ['c', 'd', 'f', 'g'], ['g', 'f', 'd', 'c'],
        ['c', 'f', 'c', 'd'], ['d', 'g', 'f', 'c'],
    ]
    return [(s, apply(s)) for s in training], apply


# ============================================================================
# PUBLIC API
# ============================================================================

ARCHETYPES = {
    'map':                  _map_rules,
    'context_map':          _context_map_rules,
    'collapse_map':         _collapse_map_rules,
    'collapse_context_map': _collapse_context_map_rules,
    'expand_map':           _expand_map_rules,
    'expand_collapse_map':  _expand_collapse_map_rules,
    'expand_context_map':       _expand_context_map_rules,
    'geometric_context_map':    _geometric_context_map_rules,
}

ARCHETYPE_DESCRIPTIONS = {
    'map':                  'Pure 1→1 substitution (cipher, palette swap)',
    'context_map':          'Neighbor-dependent + substitution (highlighting, formatting)',
    'collapse_map':         'Token merging + substitution (BPE, chord recognition)',
    'collapse_context_map': 'Collapse + context + map (phonology, compiler front-end)',
    'expand_map':           'Token expansion + substitution (abbreviations, macros)',
    'expand_collapse_map':  'Expand + collapse + map (chemical notation, serialization)',
    'expand_context_map':   'Expand + context + map (encoding with stress rules)',
    'geometric_context_map': 'Long-range φ-decay context (vowel harmony, requires geometric=True)',
}

# Archetypes that require geometric=True for correct discovery
GEOMETRIC_ARCHETYPES = {'geometric_context_map'}


def get_archetype(name):
    """Get training pairs and ground truth function for a named archetype.
    
    Args:
        name: One of 'map', 'context_map', 'collapse_map',
              'collapse_context_map', 'expand_map',
              'expand_collapse_map', 'expand_context_map'
    
    Returns:
        (training_pairs, apply_fn) where:
            training_pairs: List of (input_seq, output_seq) tuples
            apply_fn: Ground truth function (input_seq → output_seq)
    """
    if name not in ARCHETYPES:
        available = ', '.join(sorted(ARCHETYPES.keys()))
        raise ValueError(f"Unknown archetype '{name}'. Available: {available}")
    return ARCHETYPES[name]()


def discover_archetype(name, context_window=1, geometric=None):
    """Run PhaseDiscovery on a named archetype and return result + navigator.
    
    Args:
        name: Archetype name (see get_archetype)
        context_window: Context window for PhaseDiscovery
        geometric: If True, use φ-level geometric context. If None,
                  auto-enables for archetypes that require it.
    
    Returns:
        (PhaseDiscoveryResult, CascadeNavigator)
    """
    pairs, _ = get_archetype(name)
    if geometric is None:
        geometric = name in GEOMETRIC_ARCHETYPES
    pd = PhaseDiscovery(context_window=context_window, geometric=geometric)
    for inp, out in pairs:
        pd.add_pair(inp, out)
    result = pd.discover()
    nav = result.to_navigator()
    return result, nav


def list_archetypes():
    """Print all available archetypes with descriptions."""
    print("Available Transformation Archetypes:")
    print()
    for name, desc in ARCHETYPE_DESCRIPTIONS.items():
        print(f"  {name:<25s}  {desc}")
    print()
    print("Usage: pairs, fn = get_archetype('collapse_map')")
