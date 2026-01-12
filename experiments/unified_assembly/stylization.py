#!/usr/bin/env python3
"""
Text Stylization Transforms

This module provides character-level stylization functions and detection.
Stylizations are transformations that operate at the CHARACTER scale.

Examples:
- vaporwave: "hello" → "h e l l o"
- mocking: "hello" → "hElLo"
- leetspeak: "hello" → "h3ll0"
- uppercase: "hello" → "HELLO"

Key insight: Stylizations are just another dimension type.
They use the same φ-geometry as content and pattern dimensions.

Author: TruthSpace LCM Project
License: GPLv3
"""

import random
import re
from typing import Dict, Callable, Optional


# =============================================================================
# STYLIZATION FUNCTIONS
# =============================================================================

def apply_vaporwave(text: str) -> str:
    """Vaporwave style: Add spaces between characters."""
    return ' '.join(text)


def apply_mocking(text: str) -> str:
    """Mocking/spongebob style: Random case alternation."""
    result = []
    for char in text:
        if random.random() > 0.5:
            result.append(char.upper())
        else:
            result.append(char.lower())
    return ''.join(result)


def apply_leetspeak(text: str) -> str:
    """Leetspeak: Character substitution."""
    leet_map = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5',
        't': '7', 'l': '1', 'b': '8', 'g': '9',
        'A': '4', 'E': '3', 'I': '1', 'O': '0', 'S': '5',
        'T': '7', 'L': '1', 'B': '8', 'G': '9',
    }
    return ''.join(leet_map.get(c, c) for c in text)


def apply_uppercase(text: str) -> str:
    """All uppercase (shouting)."""
    return text.upper()


def apply_lowercase(text: str) -> str:
    """All lowercase (quiet)."""
    return text.lower()


def apply_reverse(text: str) -> str:
    """Reverse the text."""
    return text[::-1]


def apply_stutter(text: str) -> str:
    """Stutter style: Repeat first letter of words."""
    words = text.split()
    stuttered = []
    for word in words:
        if word and word[0].isalpha():
            stuttered.append(f"{word[0].lower()}-{word}")
        else:
            stuttered.append(word)
    return ' '.join(stuttered)


def apply_uwu(text: str) -> str:
    """UwU style: Replace r/l with w, add emoticons."""
    result = text.lower()
    result = result.replace('r', 'w').replace('l', 'w')
    return result + " uwu"


def apply_plain(text: str) -> str:
    """No stylization (identity function)."""
    return text


# Registry of stylization functions
STYLIZATIONS: Dict[str, Callable[[str], str]] = {
    'plain': apply_plain,
    'vaporwave': apply_vaporwave,
    'mocking': apply_mocking,
    'leetspeak': apply_leetspeak,
    'uppercase': apply_uppercase,
    'lowercase': apply_lowercase,
    'reverse': apply_reverse,
    'stutter': apply_stutter,
    'uwu': apply_uwu,
}


# =============================================================================
# STYLIZATION DETECTION
# =============================================================================

def detect_stylization(text: str) -> str:
    """
    Detect which stylization was applied to text.
    
    This is the inverse operation - given styled text, identify the style.
    """
    # Check for vaporwave (spaces between every character)
    if len(text) > 2:
        chars = text.split(' ')
        if all(len(c) <= 1 for c in chars) and len(chars) > 2:
            return 'vaporwave'
    
    # Check for leetspeak
    leet_chars = set('43105789')
    leet_count = sum(1 for c in text if c in leet_chars)
    if leet_count > 0 and leet_count / max(len(text), 1) > 0.1:
        return 'leetspeak'
    
    # Check for all uppercase
    if text.isupper() and any(c.isalpha() for c in text):
        return 'uppercase'
    
    # Check for all lowercase
    if text.islower() and any(c.isalpha() for c in text):
        return 'lowercase'
    
    # Check for mocking (mixed case with many transitions)
    if any(c.isupper() for c in text) and any(c.islower() for c in text):
        transitions = sum(1 for i in range(1, len(text))
                        if text[i].isupper() != text[i-1].isupper()
                        and text[i].isalpha() and text[i-1].isalpha())
        if transitions > len(text) / 4:
            return 'mocking'
    
    # Check for uwu
    if 'uwu' in text.lower():
        return 'uwu'
    
    # Check for stutter
    if re.search(r'\b\w-\w', text):
        return 'stutter'
    
    return 'plain'


# =============================================================================
# UNSTYLIZATION (REVERSE TRANSFORMS)
# =============================================================================

def unstyle(text: str, style: str) -> str:
    """Attempt to reverse a stylization."""
    if style == 'vaporwave':
        return text.replace(' ', '')
    elif style == 'uppercase':
        return text.lower()
    elif style == 'lowercase':
        return text  # Can't recover original case
    elif style == 'leetspeak':
        unleet = {'4': 'a', '3': 'e', '1': 'i', '0': 'o', '5': 's',
                  '7': 't', '8': 'b', '9': 'g'}
        return ''.join(unleet.get(c, c) for c in text)
    elif style == 'uwu':
        return text.replace(' uwu', '').replace('w', 'r')
    elif style == 'stutter':
        return re.sub(r'\b(\w)-', '', text)
    elif style == 'reverse':
        return text[::-1]
    return text


# =============================================================================
# STYLIZATION MANAGER
# =============================================================================

class StylizationManager:
    """
    Manages stylization transforms and their geometric representation.
    """
    
    def __init__(self):
        self._stylizations = STYLIZATIONS.copy()
    
    def apply(self, text: str, style: str) -> str:
        """Apply a stylization to text."""
        if style in self._stylizations:
            return self._stylizations[style](text)
        return text
    
    def detect(self, text: str) -> str:
        """Detect the stylization of text."""
        return detect_stylization(text)
    
    def unstyle(self, text: str, style: str = None) -> str:
        """Remove stylization from text."""
        if style is None:
            style = self.detect(text)
        return unstyle(text, style)
    
    def traverse(self, text: str, from_style: str, to_style: str) -> str:
        """Traverse from one stylization to another."""
        # First unstyle
        plain = self.unstyle(text, from_style)
        # Then apply new style
        return self.apply(plain, to_style)
    
    def list_styles(self) -> list:
        """List available stylizations."""
        return list(self._stylizations.keys())
    
    def register_style(self, name: str, transform_fn: Callable[[str], str]):
        """Register a new stylization."""
        self._stylizations[name] = transform_fn


# =============================================================================
# DEMO
# =============================================================================

def demo_stylizations():
    """Demonstrate stylization transforms."""
    print("=" * 60)
    print("DEMO: Stylization Transforms")
    print("=" * 60)
    print()
    
    manager = StylizationManager()
    test_text = "Hello World"
    
    print(f"Original: '{test_text}'")
    print()
    print("Stylizations:")
    for style in manager.list_styles():
        styled = manager.apply(test_text, style)
        print(f"  {style:12} → '{styled}'")
    print()
    
    print("Detection:")
    for style in ['vaporwave', 'leetspeak', 'uppercase', 'mocking']:
        styled = manager.apply(test_text, style)
        detected = manager.detect(styled)
        match = "✓" if detected == style else f"✗ ({detected})"
        print(f"  {style:12} → {match}")
    print()
    
    print("Traversal:")
    print(f"  'H e l l o' (vaporwave) → (leetspeak):")
    result = manager.traverse("H e l l o", "vaporwave", "leetspeak")
    print(f"    → '{result}'")
    print()


if __name__ == "__main__":
    demo_stylizations()
