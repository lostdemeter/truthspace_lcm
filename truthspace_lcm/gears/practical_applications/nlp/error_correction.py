"""
Error Correction Gear

A gear that detects and corrects errors in the state, including:
- Irregular verb conjugation
- Spelling mistakes
- Malformed words
- Missing context

This gear can learn corrections from examples and apply them automatically.

Author: Lesley Gushurst
License: GPLv3
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from truthspace_lcm.gears.core.base import Gear, GearState


class ErrorCorrectionGear(Gear):
    """
    Detects and corrects errors in the gear state.
    
    This gear maintains correction rules that can be:
    1. Pre-loaded from a dictionary
    2. Learned from examples
    3. Added dynamically
    
    Correction types:
    - verb: Irregular verb conjugation
    - spelling: Common spelling mistakes
    - word: Malformed or truncated words
    - pattern: Regex-based pattern corrections
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ErrorCorrectionGear", ratio)
        
        # Correction dictionaries
        self.verb_corrections: Dict[str, Dict[str, str]] = {}
        self.spelling_corrections: Dict[str, str] = {}
        self.word_corrections: Dict[str, str] = {}
        self.pattern_corrections: List[Tuple[str, str]] = []
        
        # Load default corrections
        self._load_default_corrections()
        
        # Track corrections applied
        self.correction_count = 0
    
    def _load_default_corrections(self):
        """Load default correction rules."""
        
        # Irregular verb forms: base -> {tense: form}
        self.verb_corrections = {
            # Common irregulars
            'be': {'past': 'was', 'participle': 'been', 'present': 'being', 'gerund': 'being'},
            'have': {'past': 'had', 'participle': 'had', 'present': 'having', 'gerund': 'having'},
            'do': {'past': 'did', 'participle': 'done', 'present': 'doing', 'gerund': 'doing'},
            'go': {'past': 'went', 'participle': 'gone', 'present': 'going', 'gerund': 'going'},
            'see': {'past': 'saw', 'participle': 'seen', 'present': 'seeing', 'gerund': 'seeing'},
            'come': {'past': 'came', 'participle': 'come', 'present': 'coming', 'gerund': 'coming'},
            'take': {'past': 'took', 'participle': 'taken', 'present': 'taking', 'gerund': 'taking'},
            'make': {'past': 'made', 'participle': 'made', 'present': 'making', 'gerund': 'making'},
            'know': {'past': 'knew', 'participle': 'known', 'present': 'knowing', 'gerund': 'knowing'},
            'think': {'past': 'thought', 'participle': 'thought', 'present': 'thinking', 'gerund': 'thinking'},
            'get': {'past': 'got', 'participle': 'gotten', 'present': 'getting', 'gerund': 'getting'},
            'give': {'past': 'gave', 'participle': 'given', 'present': 'giving', 'gerund': 'giving'},
            'find': {'past': 'found', 'participle': 'found', 'present': 'finding', 'gerund': 'finding'},
            'tell': {'past': 'told', 'participle': 'told', 'present': 'telling', 'gerund': 'telling'},
            'become': {'past': 'became', 'participle': 'become', 'present': 'becoming', 'gerund': 'becoming'},
            'leave': {'past': 'left', 'participle': 'left', 'present': 'leaving', 'gerund': 'leaving'},
            'put': {'past': 'put', 'participle': 'put', 'present': 'putting', 'gerund': 'putting'},
            'keep': {'past': 'kept', 'participle': 'kept', 'present': 'keeping', 'gerund': 'keeping'},
            'begin': {'past': 'began', 'participle': 'begun', 'present': 'beginning', 'gerund': 'beginning'},
            'bring': {'past': 'brought', 'participle': 'brought', 'present': 'bringing', 'gerund': 'bringing'},
            'write': {'past': 'wrote', 'participle': 'written', 'present': 'writing', 'gerund': 'writing'},
            'read': {'past': 'read', 'participle': 'read', 'present': 'reading', 'gerund': 'reading'},
            'run': {'past': 'ran', 'participle': 'run', 'present': 'running', 'gerund': 'running'},
            'hold': {'past': 'held', 'participle': 'held', 'present': 'holding', 'gerund': 'holding'},
            'stand': {'past': 'stood', 'participle': 'stood', 'present': 'standing', 'gerund': 'standing'},
            'understand': {'past': 'understood', 'participle': 'understood', 'present': 'understanding', 'gerund': 'understanding'},
            'lose': {'past': 'lost', 'participle': 'lost', 'present': 'losing', 'gerund': 'losing'},
            'pay': {'past': 'paid', 'participle': 'paid', 'present': 'paying', 'gerund': 'paying'},
            'meet': {'past': 'met', 'participle': 'met', 'present': 'meeting', 'gerund': 'meeting'},
            'sit': {'past': 'sat', 'participle': 'sat', 'present': 'sitting', 'gerund': 'sitting'},
            'speak': {'past': 'spoke', 'participle': 'spoken', 'present': 'speaking', 'gerund': 'speaking'},
            'lie': {'past': 'lay', 'participle': 'lain', 'present': 'lying', 'gerund': 'lying'},
            'lead': {'past': 'led', 'participle': 'led', 'present': 'leading', 'gerund': 'leading'},
            'grow': {'past': 'grew', 'participle': 'grown', 'present': 'growing', 'gerund': 'growing'},
            'draw': {'past': 'drew', 'participle': 'drawn', 'present': 'drawing', 'gerund': 'drawing'},
            'show': {'past': 'showed', 'participle': 'shown', 'present': 'showing', 'gerund': 'showing'},
            'break': {'past': 'broke', 'participle': 'broken', 'present': 'breaking', 'gerund': 'breaking'},
            'drive': {'past': 'drove', 'participle': 'driven', 'present': 'driving', 'gerund': 'driving'},
            'buy': {'past': 'bought', 'participle': 'bought', 'present': 'buying', 'gerund': 'buying'},
            'send': {'past': 'sent', 'participle': 'sent', 'present': 'sending', 'gerund': 'sending'},
            'build': {'past': 'built', 'participle': 'built', 'present': 'building', 'gerund': 'building'},
            'fall': {'past': 'fell', 'participle': 'fallen', 'present': 'falling', 'gerund': 'falling'},
            'cut': {'past': 'cut', 'participle': 'cut', 'present': 'cutting', 'gerund': 'cutting'},
            'rise': {'past': 'rose', 'participle': 'risen', 'present': 'rising', 'gerund': 'rising'},
            'set': {'past': 'set', 'participle': 'set', 'present': 'setting', 'gerund': 'setting'},
            'spend': {'past': 'spent', 'participle': 'spent', 'present': 'spending', 'gerund': 'spending'},
            'choose': {'past': 'chose', 'participle': 'chosen', 'present': 'choosing', 'gerund': 'choosing'},
            'feel': {'past': 'felt', 'participle': 'felt', 'present': 'feeling', 'gerund': 'feeling'},
            'catch': {'past': 'caught', 'participle': 'caught', 'present': 'catching', 'gerund': 'catching'},
            'teach': {'past': 'taught', 'participle': 'taught', 'present': 'teaching', 'gerund': 'teaching'},
            'fight': {'past': 'fought', 'participle': 'fought', 'present': 'fighting', 'gerund': 'fighting'},
            'throw': {'past': 'threw', 'participle': 'thrown', 'present': 'throwing', 'gerund': 'throwing'},
            'win': {'past': 'won', 'participle': 'won', 'present': 'winning', 'gerund': 'winning'},
            'fly': {'past': 'flew', 'participle': 'flown', 'present': 'flying', 'gerund': 'flying'},
            'wear': {'past': 'wore', 'participle': 'worn', 'present': 'wearing', 'gerund': 'wearing'},
            'eat': {'past': 'ate', 'participle': 'eaten', 'present': 'eating', 'gerund': 'eating'},
            'drink': {'past': 'drank', 'participle': 'drunk', 'present': 'drinking', 'gerund': 'drinking'},
            'sleep': {'past': 'slept', 'participle': 'slept', 'present': 'sleeping', 'gerund': 'sleeping'},
            'wake': {'past': 'woke', 'participle': 'woken', 'present': 'waking', 'gerund': 'waking'},
            'swim': {'past': 'swam', 'participle': 'swum', 'present': 'swimming', 'gerund': 'swimming'},
            'sing': {'past': 'sang', 'participle': 'sung', 'present': 'singing', 'gerund': 'singing'},
            'ring': {'past': 'rang', 'participle': 'rung', 'present': 'ringing', 'gerund': 'ringing'},
            'forget': {'past': 'forgot', 'participle': 'forgotten', 'present': 'forgetting', 'gerund': 'forgetting'},
            'hide': {'past': 'hid', 'participle': 'hidden', 'present': 'hiding', 'gerund': 'hiding'},
            'beat': {'past': 'beat', 'participle': 'beaten', 'present': 'beating', 'gerund': 'beating'},
            'bite': {'past': 'bit', 'participle': 'bitten', 'present': 'biting', 'gerund': 'biting'},
            'blow': {'past': 'blew', 'participle': 'blown', 'present': 'blowing', 'gerund': 'blowing'},
            'freeze': {'past': 'froze', 'participle': 'frozen', 'present': 'freezing', 'gerund': 'freezing'},
            'shake': {'past': 'shook', 'participle': 'shaken', 'present': 'shaking', 'gerund': 'shaking'},
            'steal': {'past': 'stole', 'participle': 'stolen', 'present': 'stealing', 'gerund': 'stealing'},
            'tear': {'past': 'tore', 'participle': 'torn', 'present': 'tearing', 'gerund': 'tearing'},
        }
        
        # Common spelling/word corrections
        self.word_corrections = {
            # Typos from corpus
            'monitores': 'monitors',
            'facilitats': 'facilitates',
            'proces': 'process',
            'ongoes': 'ongoing',
            'emphasi': 'emphasis',
            'formalizat': 'formalization',
            'encompas': 'encompass',
            'overlaping': 'overlapping',
            'procing': 'processing',
            'encompaing': 'encompassing',
            'rigorizing': 'rigorizing',  # Keep as is (valid neologism)
            'iing': 'ing',  # Malformed gerund
            'quantifiing': 'quantifying',
            
            # Common misspellings
            'occured': 'occurred',
            'occurence': 'occurrence',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'accomodate': 'accommodate',
            'occurrance': 'occurrence',
            'wierd': 'weird',
            'untill': 'until',
            'begining': 'beginning',
            'beleive': 'believe',
            'calender': 'calendar',
            'collegue': 'colleague',
            'concious': 'conscious',
            'existance': 'existence',
            'foriegn': 'foreign',
            'goverment': 'government',
            'independant': 'independent',
            'neccessary': 'necessary',
            'occassion': 'occasion',
            'paralell': 'parallel',
            'priviledge': 'privilege',
            'recomend': 'recommend',
            'refered': 'referred',
            'relevent': 'relevant',
            'succesful': 'successful',
            'tommorow': 'tomorrow',
        }
        
        # Pattern-based corrections (regex)
        self.pattern_corrections = [
            # Double 'ing' -> single 'ing'
            (r'(\w)iing\b', r'\1ing'),
            # Triple consonants -> double (e.g., 'dividding' -> 'dividing')
            (r'([bcdfghjklmnpqrstvwxyz])\1\1', r'\1\1'),
            # Double consonant before 'ing' when shouldn't be (e.g., 'dividding' -> 'dividing')
            (r'([bcdfghjklmnpqrstvwxyz])\1ing\b', r'\1ing'),
            # 'eing' at end -> 'ing' (for most verbs)
            (r'([^aeiou])eing\b', r'\1ing'),
            # Double consonant before 'ed' when shouldn't be
            (r'([bcdfghjklmnpqrstvwxyz])\1ed\b', r'\1ed'),
        ]
    
    def add_verb(self, base: str, past: str, participle: str, 
                 gerund: Optional[str] = None) -> 'ErrorCorrectionGear':
        """
        Add an irregular verb to the correction dictionary.
        
        Args:
            base: Base form (e.g., 'go')
            past: Past tense (e.g., 'went')
            participle: Past participle (e.g., 'gone')
            gerund: Present participle (optional, defaults to base + 'ing')
        """
        if gerund is None:
            if base.endswith('e'):
                gerund = base[:-1] + 'ing'
            elif base.endswith('ie'):
                gerund = base[:-2] + 'ying'
            else:
                gerund = base + 'ing'
        
        self.verb_corrections[base] = {
            'past': past,
            'participle': participle,
            'present': gerund,
            'gerund': gerund,
        }
        return self
    
    def add_word_correction(self, wrong: str, correct: str) -> 'ErrorCorrectionGear':
        """Add a word correction."""
        self.word_corrections[wrong.lower()] = correct
        return self
    
    def add_pattern_correction(self, pattern: str, replacement: str) -> 'ErrorCorrectionGear':
        """Add a regex pattern correction."""
        self.pattern_corrections.append((pattern, replacement))
        return self
    
    def learn_from_examples(self, examples: List[Tuple[str, str]]) -> 'ErrorCorrectionGear':
        """
        Learn corrections from example pairs.
        
        Args:
            examples: List of (wrong, correct) tuples
        """
        for wrong, correct in examples:
            self.word_corrections[wrong.lower()] = correct
        return self
    
    def correct_word(self, word: str) -> str:
        """Correct a single word."""
        word_lower = word.lower()
        
        # Check direct word corrections
        if word_lower in self.word_corrections:
            corrected = self.word_corrections[word_lower]
            # Preserve original case
            if word[0].isupper():
                corrected = corrected.capitalize()
            return corrected
        
        # Apply pattern corrections
        for pattern, replacement in self.pattern_corrections:
            new_word = re.sub(pattern, replacement, word)
            if new_word != word:
                return new_word
        
        return word
    
    def correct_verb(self, verb: str, tense: str) -> str:
        """
        Correct a verb to the specified tense.
        
        Args:
            verb: The verb to correct
            tense: Target tense ('past', 'participle', 'present', 'gerund')
        """
        # Get base form
        base = self._to_base(verb)
        
        # Check if it's an irregular verb
        if base in self.verb_corrections:
            forms = self.verb_corrections[base]
            if tense in forms:
                return forms[tense]
        
        # Regular verb conjugation
        return self._regular_conjugation(base, tense)
    
    def _to_base(self, verb: str) -> str:
        """Convert verb to base form."""
        verb = verb.lower().strip()
        
        # Check if it's a known irregular form
        for base, forms in self.verb_corrections.items():
            if verb in forms.values():
                return base
        
        # Remove common endings
        if verb.endswith('ing'):
            base = verb[:-3]
            # Check if we need to add back 'e'
            if base + 'e' in self.verb_corrections:
                return base + 'e'
            # Double consonant removal
            if len(base) > 2 and base[-1] == base[-2]:
                return base[:-1]
            return base
        
        if verb.endswith('ed'):
            base = verb[:-2]
            if base + 'e' in self.verb_corrections:
                return base + 'e'
            if base.endswith('i'):
                return base[:-1] + 'y'
            return base
        
        if verb.endswith('ies'):
            return verb[:-3] + 'y'
        
        if verb.endswith('es') and len(verb) > 3:
            return verb[:-2]
        
        if verb.endswith('s') and not verb.endswith('ss'):
            return verb[:-1]
        
        return verb
    
    def _regular_conjugation(self, base: str, tense: str) -> str:
        """Apply regular verb conjugation rules."""
        if tense == 'past' or tense == 'participle':
            if base.endswith('e'):
                return base + 'd'
            elif base.endswith('y') and len(base) > 2 and base[-2] not in 'aeiou':
                return base[:-1] + 'ied'
            elif len(base) > 2 and base[-1] not in 'aeiouwxy' and base[-2] in 'aeiou' and base[-3] not in 'aeiou':
                # Double final consonant (e.g., stop -> stopped)
                return base + base[-1] + 'ed'
            else:
                return base + 'ed'
        
        elif tense == 'present' or tense == 'gerund':
            if base.endswith('e') and not base.endswith('ee'):
                return base[:-1] + 'ing'
            elif base.endswith('ie'):
                return base[:-2] + 'ying'
            elif len(base) > 2 and base[-1] not in 'aeiouwxy' and base[-2] in 'aeiou' and base[-3] not in 'aeiou':
                # Double final consonant (e.g., run -> running)
                return base + base[-1] + 'ing'
            else:
                return base + 'ing'
        
        return base
    
    def forward(self, state: GearState) -> GearState:
        """Apply error corrections to the state."""
        if self.ratio < 0.1:
            return state  # Skip if ratio too low
        
        corrections_made = []
        
        # Correct actions - only apply word corrections, not verb conjugation
        # (verb conjugation is handled by TenseGear)
        corrected_actions = []
        for action in state.actions:
            corrected = self.correct_word(action)
            
            if corrected != action:
                corrections_made.append(f"action: {action} → {corrected}")
                self.correction_count += 1
            
            corrected_actions.append(corrected)
        
        state.actions = corrected_actions
        
        # Correct targets
        corrected_targets = []
        for target in state.targets:
            corrected = self.correct_word(target)
            if corrected != target:
                corrections_made.append(f"target: {target} → {corrected}")
                self.correction_count += 1
            corrected_targets.append(corrected)
        
        state.targets = corrected_targets
        
        # Track corrections
        state.corrections_applied.extend(corrections_made)
        
        return state
    
    def get_stats(self) -> Dict[str, int]:
        """Get correction statistics."""
        return {
            'total_corrections': self.correction_count,
            'verb_rules': len(self.verb_corrections),
            'word_rules': len(self.word_corrections),
            'pattern_rules': len(self.pattern_corrections),
        }
