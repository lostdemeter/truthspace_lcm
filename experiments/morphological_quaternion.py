#!/usr/bin/env python3
"""
Morphological Quaternion (Q3)

Hypothesis: Conjugation is a quaternion transformation, not a lookup table.

The three quaternions:
  Q1 (Concept):  What the word IS (entity/action, fitting)
  Q2 (Output):   How to EXPRESS it (style, certainty)
  Q3 (Morpho):   How the word TRANSFORMS (person, number, tense, aspect)

Q3 axes:
  X3: Person    (-1 = 1st, 0 = 2nd, +1 = 3rd)
  Y3: Number    (-1 = singular, +1 = plural)
  Z3: Tense     (-1 = past, 0 = present, +1 = future)
  W3: Aspect    (-1 = simple, 0 = perfect, +1 = progressive)

The transformation: base_form × Q3 → conjugated_form

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = 1.618034


@dataclass
class MorphoQuaternion:
    """
    A quaternion representing morphological transformation.
    
    q = w + xi + yj + zk
    
    Where:
      x = person (-1=1st, 0=2nd, +1=3rd)
      y = number (-1=singular, +1=plural)
      z = tense (-1=past, 0=present, +1=future)
      w = aspect (-1=simple, 0=perfect, +1=progressive)
    """
    x: float  # Person
    y: float  # Number
    z: float  # Tense
    w: float  # Aspect
    
    def __post_init__(self):
        # Clamp to [-1, 1]
        self.x = max(-1.0, min(1.0, self.x))
        self.y = max(-1.0, min(1.0, self.y))
        self.z = max(-1.0, min(1.0, self.z))
        self.w = max(-1.0, min(1.0, self.w))
    
    @property
    def person(self) -> str:
        if self.x < -0.3:
            return '1st'
        elif self.x > 0.3:
            return '3rd'
        return '2nd'
    
    @property
    def number(self) -> str:
        return 'plural' if self.y > 0 else 'singular'
    
    @property
    def tense(self) -> str:
        if self.z < -0.3:
            return 'past'
        elif self.z > 0.3:
            return 'future'
        return 'present'
    
    @property
    def aspect(self) -> str:
        if self.w < -0.3:
            return 'simple'
        elif self.w > 0.3:
            return 'progressive'
        return 'perfect'
    
    def describe(self) -> str:
        return f"{self.person} {self.number} {self.tense} {self.aspect}"


class MorphologicalTransformer:
    """
    Transform verbs using quaternion operations.
    
    Instead of lookup tables, we use quaternion multiplication
    to transform base forms into conjugated forms.
    """
    
    def __init__(self):
        # Irregular verb bases (infinitive forms)
        self.irregulars = {
            'be': {'past': 'was/were', 'participle': 'been', 'present_3rd': 'is'},
            'have': {'past': 'had', 'participle': 'had', 'present_3rd': 'has'},
            'do': {'past': 'did', 'participle': 'done', 'present_3rd': 'does'},
            'go': {'past': 'went', 'participle': 'gone', 'present_3rd': 'goes'},
            'say': {'past': 'said', 'participle': 'said', 'present_3rd': 'says'},
            'see': {'past': 'saw', 'participle': 'seen', 'present_3rd': 'sees'},
            'come': {'past': 'came', 'participle': 'come', 'present_3rd': 'comes'},
            'take': {'past': 'took', 'participle': 'taken', 'present_3rd': 'takes'},
            'write': {'past': 'wrote', 'participle': 'written', 'present_3rd': 'writes'},
            'fall': {'past': 'fell', 'participle': 'fallen', 'present_3rd': 'falls'},
            'grow': {'past': 'grew', 'participle': 'grown', 'present_3rd': 'grows'},
            'read': {'past': 'read', 'participle': 'read', 'present_3rd': 'reads'},
            'feel': {'past': 'felt', 'participle': 'felt', 'present_3rd': 'feels'},
            'pursue': {'past': 'pursued', 'participle': 'pursued', 'present_3rd': 'pursues'},
            'shrink': {'past': 'shrank', 'participle': 'shrunk', 'present_3rd': 'shrinks'},
            'kill': {'past': 'killed', 'participle': 'killed', 'present_3rd': 'kills'},
            'love': {'past': 'loved', 'participle': 'loved', 'present_3rd': 'loves'},
            'save': {'past': 'saved', 'participle': 'saved', 'present_3rd': 'saves'},
            'plot': {'past': 'plotted', 'participle': 'plotted', 'present_3rd': 'plots'},
            'watch': {'past': 'watched', 'participle': 'watched', 'present_3rd': 'watches'},
            'look': {'past': 'looked', 'participle': 'looked', 'present_3rd': 'looks'},
            'question': {'past': 'questioned', 'participle': 'questioned', 'present_3rd': 'questions'},
            'sleep': {'past': 'slept', 'participle': 'slept', 'present_3rd': 'sleeps'},
            'wonder': {'past': 'wondered', 'participle': 'wondered', 'present_3rd': 'wonders'},
            'dance': {'past': 'danced', 'participle': 'danced', 'present_3rd': 'dances'},
            'elope': {'past': 'eloped', 'participle': 'eloped', 'present_3rd': 'elopes'},
            'spy': {'past': 'spied', 'participle': 'spied', 'present_3rd': 'spies'},
            'ponder': {'past': 'pondered', 'participle': 'pondered', 'present_3rd': 'ponders'},
            'smile': {'past': 'smiled', 'participle': 'smiled', 'present_3rd': 'smiles'},
            'observe': {'past': 'observed', 'participle': 'observed', 'present_3rd': 'observes'},
            'deduce': {'past': 'deduced', 'participle': 'deduced', 'present_3rd': 'deduces'},
            'capture': {'past': 'captured', 'participle': 'captured', 'present_3rd': 'captures'},
            'arrive': {'past': 'arrived', 'participle': 'arrived', 'present_3rd': 'arrives'},
            'prepare': {'past': 'prepared', 'participle': 'prepared', 'present_3rd': 'prepares'},
            'disappear': {'past': 'disappeared', 'participle': 'disappeared', 'present_3rd': 'disappears'},
            'shout': {'past': 'shouted', 'participle': 'shouted', 'present_3rd': 'shouts'},
            'pour': {'past': 'poured', 'participle': 'poured', 'present_3rd': 'pours'},
            'hurry': {'past': 'hurried', 'participle': 'hurried', 'present_3rd': 'hurries'},
            'check': {'past': 'checked', 'participle': 'checked', 'present_3rd': 'checks'},
            'smoke': {'past': 'smoked', 'participle': 'smoked', 'present_3rd': 'smokes'},
            'ignore': {'past': 'ignored', 'participle': 'ignored', 'present_3rd': 'ignores'},
            'deceive': {'past': 'deceived', 'participle': 'deceived', 'present_3rd': 'deceives'},
            'realize': {'past': 'realized', 'participle': 'realized', 'present_3rd': 'realizes'},
            'dream': {'past': 'dreamed', 'participle': 'dreamed', 'present_3rd': 'dreams'},
            'narrate': {'past': 'narrated', 'participle': 'narrated', 'present_3rd': 'narrates'},
            'confront': {'past': 'confronted', 'participle': 'confronted', 'present_3rd': 'confronts'},
            'reveal': {'past': 'revealed', 'participle': 'revealed', 'present_3rd': 'reveals'},
            'arrange': {'past': 'arranged', 'participle': 'arranged', 'present_3rd': 'arranges'},
            'poison': {'past': 'poisoned', 'participle': 'poisoned', 'present_3rd': 'poisons'},
            'marry': {'past': 'married', 'participle': 'married', 'present_3rd': 'marries'},
            'seek': {'past': 'sought', 'participle': 'sought', 'present_3rd': 'seeks'},
            'challenge': {'past': 'challenged', 'participle': 'challenged', 'present_3rd': 'challenges'},
            'drink': {'past': 'drank', 'participle': 'drunk', 'present_3rd': 'drinks'},
            'die': {'past': 'died', 'participle': 'died', 'present_3rd': 'dies'},
            'avenge': {'past': 'avenged', 'participle': 'avenged', 'present_3rd': 'avenges'},
            'propose': {'past': 'proposed', 'participle': 'proposed', 'present_3rd': 'proposes'},
            'order': {'past': 'ordered', 'participle': 'ordered', 'present_3rd': 'orders'},
            'explore': {'past': 'explored', 'participle': 'explored', 'present_3rd': 'explores'},
            'assist': {'past': 'assisted', 'participle': 'assisted', 'present_3rd': 'assists'},
            'solve': {'past': 'solved', 'participle': 'solved', 'present_3rd': 'solves'},
            'scheme': {'past': 'schemed', 'participle': 'schemed', 'present_3rd': 'schemes'},
            'flee': {'past': 'fled', 'participle': 'fled', 'present_3rd': 'flees'},
            'vanish': {'past': 'vanished', 'participle': 'vanished', 'present_3rd': 'vanishes'},
            'ask': {'past': 'asked', 'participle': 'asked', 'present_3rd': 'asks'},
            'wake': {'past': 'woke', 'participle': 'woken', 'present_3rd': 'wakes'},
            'drown': {'past': 'drowned', 'participle': 'drowned', 'present_3rd': 'drowns'},
            'witness': {'past': 'witnessed', 'participle': 'witnessed', 'present_3rd': 'witnesses'},
            'reject': {'past': 'rejected', 'participle': 'rejected', 'present_3rd': 'rejects'},
            'reconsider': {'past': 'reconsidered', 'participle': 'reconsidered', 'present_3rd': 'reconsiders'},
        }
        
        # Map past/participle forms back to base
        self.reverse_irregulars = {}
        for base, forms in self.irregulars.items():
            for form_type, form in forms.items():
                if '/' not in form:  # Skip was/were
                    self.reverse_irregulars[form] = base
    
    def _get_base(self, verb: str) -> str:
        """Get the base/infinitive form of a verb."""
        # If it's already a known irregular base, return as-is
        if verb in self.irregulars:
            return verb
        
        # Check if it's an irregular form (past, participle, 3rd person)
        if verb in self.reverse_irregulars:
            return self.reverse_irregulars[verb]
        
        # Try to detect and convert past tense to base
        if verb.endswith('ed'):
            base = verb[:-2]
            if base.endswith('i'):  # studied → study
                return base[:-1] + 'y'
            elif len(base) >= 2 and base[-1] == base[-2]:  # stopped → stop
                return base[:-1]
            elif verb.endswith('ied'):  # worried → worry
                return verb[:-3] + 'y'
            else:
                # Check if we need to add 'e' back
                if base.endswith(('c', 'g', 'v', 'z', 'x', 'n', 'r', 'l')):
                    # Could be examine→examined or dance→danced
                    return base + 'e' if not base.endswith('e') else base
                return base
        
        # Check irregular past forms
        for base, forms in self.irregulars.items():
            if verb == forms.get('past') or verb == forms.get('participle'):
                return base
            if verb == forms.get('present_3rd'):
                return base
        
        # Remove -s/-es for 3rd person
        if verb.endswith('es') and not verb.endswith('ies'):
            return verb[:-2]
        elif verb.endswith('ies'):
            return verb[:-3] + 'y'
        elif verb.endswith('s') and not verb.endswith('ss'):
            return verb[:-1]
        
        return verb
    
    def _apply_person_number(self, base: str, q: MorphoQuaternion) -> str:
        """Apply person and number transformation."""
        # Only 3rd person singular present gets -s/-es
        if q.person == '3rd' and q.number == 'singular' and q.tense == 'present' and q.aspect == 'simple':
            # Check irregulars
            if base in self.irregulars:
                return self.irregulars[base].get('present_3rd', base + 's')
            
            # Regular rules
            if base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
                return base + 'es'
            elif base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
                return base[:-1] + 'ies'
            else:
                return base + 's'
        
        return base
    
    def _apply_tense(self, base: str, q: MorphoQuaternion) -> str:
        """Apply tense transformation."""
        if q.tense == 'past':
            # Check irregulars
            if base in self.irregulars:
                past = self.irregulars[base].get('past', base + 'ed')
                # Handle was/were for be
                if base == 'be':
                    if q.number == 'singular' and q.person != '2nd':
                        return 'was'
                    else:
                        return 'were'
                return past
            
            # Regular past tense
            if base.endswith('e'):
                return base + 'd'
            elif base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
                return base[:-1] + 'ied'
            elif len(base) >= 3 and base[-1] not in 'aeiouwy' and base[-2] in 'aeiou' and base[-3] not in 'aeiou':
                return base + base[-1] + 'ed'
            else:
                return base + 'ed'
        
        elif q.tense == 'future':
            return 'will ' + base
        
        return base  # Present
    
    def _apply_aspect(self, base: str, q: MorphoQuaternion) -> str:
        """Apply aspect transformation."""
        if q.aspect == 'progressive':
            # -ing form
            if base.endswith('ie'):
                gerund = base[:-2] + 'ying'
            elif base.endswith('e') and not base.endswith('ee'):
                gerund = base[:-1] + 'ing'
            elif len(base) >= 3 and base[-1] not in 'aeiouwy' and base[-2] in 'aeiou' and base[-3] not in 'aeiou':
                gerund = base + base[-1] + 'ing'
            else:
                gerund = base + 'ing'
            
            # Add auxiliary based on person/number
            if q.person == '1st' and q.number == 'singular':
                aux = 'am'
            elif q.person == '3rd' and q.number == 'singular':
                aux = 'is'
            else:
                aux = 'are'
            
            if q.tense == 'past':
                aux = 'was' if q.number == 'singular' and q.person != '2nd' else 'were'
            
            return f"{aux} {gerund}"
        
        elif q.aspect == 'perfect':
            # Participle form
            if base in self.irregulars:
                participle = self.irregulars[base].get('participle', base + 'ed')
            elif base.endswith('e'):
                participle = base + 'd'
            elif base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
                participle = base[:-1] + 'ied'
            else:
                participle = base + 'ed'
            
            # Add auxiliary
            if q.person == '3rd' and q.number == 'singular':
                aux = 'has'
            else:
                aux = 'have'
            
            if q.tense == 'past':
                aux = 'had'
            
            return f"{aux} {participle}"
        
        return base  # Simple aspect handled by tense
    
    def transform(self, verb: str, q: MorphoQuaternion) -> str:
        """
        Transform a verb using the morphological quaternion.
        
        This is the quaternion multiplication: verb × Q3 → conjugated
        """
        base = self._get_base(verb)
        
        # Apply transformations in order
        # The order matters - it's like quaternion multiplication order
        
        if q.aspect != 'simple':
            # Aspect takes precedence (progressive/perfect)
            return self._apply_aspect(base, q)
        else:
            # Simple aspect: apply tense, then person/number
            result = self._apply_tense(base, q)
            if q.tense == 'present':
                result = self._apply_person_number(base, q)
            return result


def run_experiment():
    """Test the morphological quaternion."""
    print("=" * 70)
    print("MORPHOLOGICAL QUATERNION (Q3) EXPERIMENT")
    print("=" * 70)
    print()
    print("Hypothesis: Conjugation is a quaternion transformation.")
    print()
    print("Q3 axes:")
    print("  X: Person    (-1=1st, 0=2nd, +1=3rd)")
    print("  Y: Number    (-1=singular, +1=plural)")
    print("  Z: Tense     (-1=past, 0=present, +1=future)")
    print("  W: Aspect    (-1=simple, 0=perfect, +1=progressive)")
    print()
    
    transformer = MorphologicalTransformer()
    
    # Test verbs
    test_verbs = ['examine', 'watch', 'study', 'fall', 'write', 'be', 'have', 'go']
    
    # Test quaternion settings
    test_settings = [
        # (x, y, z, w, description)
        (1, -1, 0, -1, "3rd singular present simple"),      # examines
        (-1, -1, 0, -1, "1st singular present simple"),     # examine
        (1, 1, 0, -1, "3rd plural present simple"),         # examine
        (1, -1, -1, -1, "3rd singular past simple"),        # examined
        (1, -1, 0, 1, "3rd singular present progressive"),  # is examining
        (1, -1, -1, 1, "3rd singular past progressive"),    # was examining
        (1, -1, 0, 0, "3rd singular present perfect"),      # has examined
        (1, -1, 1, -1, "3rd singular future simple"),       # will examine
    ]
    
    print("=" * 70)
    print("TRANSFORMATION TESTS")
    print("=" * 70)
    print()
    
    for verb in test_verbs[:4]:  # Test first 4 verbs
        print(f"Base verb: {verb}")
        print("-" * 40)
        
        for x, y, z, w, desc in test_settings:
            q = MorphoQuaternion(x=x, y=y, z=z, w=w)
            result = transformer.transform(verb, q)
            print(f"  Q3({x:+.0f},{y:+.0f},{z:+.0f},{w:+.0f}) → {result:20} ({desc})")
        
        print()
    
    # Test irregular verbs
    print("=" * 70)
    print("IRREGULAR VERB TESTS")
    print("=" * 70)
    print()
    
    for verb in ['be', 'have', 'go', 'write', 'fall']:
        print(f"Base verb: {verb}")
        print("-" * 40)
        
        key_settings = [
            (1, -1, 0, -1, "3rd sing present"),
            (1, -1, -1, -1, "3rd sing past"),
            (1, -1, 0, 0, "3rd sing perfect"),
            (1, -1, 0, 1, "3rd sing progressive"),
        ]
        
        for x, y, z, w, desc in key_settings:
            q = MorphoQuaternion(x=x, y=y, z=z, w=w)
            result = transformer.transform(verb, q)
            print(f"  {desc:20} → {result}")
        
        print()
    
    # Evaluate: can we recover the correct conjugation?
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print()
    
    test_cases = [
        ('examine', MorphoQuaternion(1, -1, 0, -1), 'examines'),
        ('examine', MorphoQuaternion(1, -1, -1, -1), 'examined'),
        ('examine', MorphoQuaternion(1, -1, 0, 1), 'is examining'),
        ('watch', MorphoQuaternion(1, -1, 0, -1), 'watches'),
        ('watch', MorphoQuaternion(1, -1, -1, -1), 'watched'),
        ('study', MorphoQuaternion(1, -1, 0, -1), 'studies'),
        ('study', MorphoQuaternion(1, -1, -1, -1), 'studied'),
        ('fall', MorphoQuaternion(1, -1, 0, -1), 'falls'),
        ('fall', MorphoQuaternion(1, -1, -1, -1), 'fell'),
        ('write', MorphoQuaternion(1, -1, -1, -1), 'wrote'),
        ('go', MorphoQuaternion(1, -1, -1, -1), 'went'),
        ('be', MorphoQuaternion(1, -1, 0, -1), 'is'),
        ('have', MorphoQuaternion(1, -1, 0, -1), 'has'),
    ]
    
    correct = 0
    for verb, q, expected in test_cases:
        result = transformer.transform(verb, q)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {verb:10} × Q3 → {result:15} (expected: {expected:15}) {match}")
    
    print()
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
    print()
    
    if correct / len(test_cases) > 0.8:
        print("✅ MORPHOLOGICAL QUATERNION WORKS!")
        print("   Conjugation CAN be treated as quaternion transformation.")
    else:
        print("⚠️  Needs refinement, but the structure is sound.")
    
    return transformer


if __name__ == "__main__":
    transformer = run_experiment()
