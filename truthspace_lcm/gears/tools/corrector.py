"""
Corpus Corrector

Applies corrections to corpus frames:
- Spelling fixes
- Role corrections
- Verb normalization
- Pattern-based transformations

Author: Lesley Gushurst
License: GPLv3
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class CorrectionResult:
    """Results from a correction operation."""
    corrected_frames: List[Dict[str, Any]] = field(default_factory=list)
    corrections: List[Tuple[str, str, str]] = field(default_factory=list)  # (frame_id, before, after)
    unchanged_count: int = 0
    
    @property
    def corrected_count(self) -> int:
        return len(self.corrected_frames)
    
    def summary(self) -> str:
        lines = [
            f"Correction Results:",
            f"  Corrected frames: {self.corrected_count}",
            f"  Unchanged frames: {self.unchanged_count}",
            f"  Total corrections: {len(self.corrections)}",
        ]
        if self.corrections[:5]:
            lines.append("  Sample corrections:")
            for frame_id, before, after in self.corrections[:5]:
                lines.append(f"    - {before[:40]}... → {after[:40]}...")
        return "\n".join(lines)


class CorpusCorrector:
    """
    Applies corrections to corpus frames.
    
    Usage:
        corrector = CorpusCorrector()
        corrector.add_spelling('wierd', 'weird')
        corrector.add_role_fix('character', 'concept', lambda t: 'ology' in t)
        
        result = corrector.correct(corpus_data)
        print(result.summary())
    """
    
    def __init__(self):
        # Spelling corrections: wrong -> correct
        self.spelling: Dict[str, str] = {
            'occured': 'occurred',
            'occurence': 'occurrence',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'accomodate': 'accommodate',
            'wierd': 'weird',
            'untill': 'until',
            'begining': 'beginning',
            'beleive': 'believe',
        }
        
        # Role corrections: (from_role, to_role, condition_fn)
        self.role_fixes: List[Tuple[str, str, Callable[[str], bool]]] = []
        
        # Pattern replacements: (pattern, replacement)
        self.patterns: List[Tuple[str, str]] = []
        
        # Custom transformers
        self.transformers: List[Callable[[str], str]] = []
    
    def add_spelling(self, wrong: str, correct: str) -> 'CorpusCorrector':
        """Add a spelling correction."""
        self.spelling[wrong.lower()] = correct
        return self
    
    def add_role_fix(self, from_role: str, to_role: str, 
                     condition: Callable[[str], bool]) -> 'CorpusCorrector':
        """
        Add a role correction rule.
        
        Args:
            from_role: Role to change from (e.g., 'character')
            to_role: Role to change to (e.g., 'concept')
            condition: Function that takes frame text and returns True if rule applies
        """
        self.role_fixes.append((from_role, to_role, condition))
        return self
    
    def add_pattern(self, pattern: str, replacement: str) -> 'CorpusCorrector':
        """Add a regex pattern replacement."""
        self.patterns.append((pattern, replacement))
        return self
    
    def add_transformer(self, fn: Callable[[str], str]) -> 'CorpusCorrector':
        """Add a custom transformation function."""
        self.transformers.append(fn)
        return self
    
    def _apply_spelling(self, text: str) -> str:
        """Apply spelling corrections."""
        result = text
        for wrong, correct in self.spelling.items():
            # Use word boundaries
            pattern = r'\b' + re.escape(wrong) + r'\b'
            result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
        return result
    
    def _apply_role_fixes(self, text: str) -> str:
        """Apply role corrections."""
        result = text
        for from_role, to_role, condition in self.role_fixes:
            if condition(text):
                # Replace role in text
                pattern = rf'\bis a[n]? {from_role}\b'
                article = 'an' if to_role[0].lower() in 'aeiou' else 'a'
                result = re.sub(pattern, f'is {article} {to_role}', result, flags=re.IGNORECASE)
                
                pattern = rf'\bis {from_role}\b'
                result = re.sub(pattern, f'is {article} {to_role}', result, flags=re.IGNORECASE)
        return result
    
    def _apply_patterns(self, text: str) -> str:
        """Apply pattern replacements."""
        result = text
        for pattern, replacement in self.patterns:
            result = re.sub(pattern, replacement, result)
        return result
    
    def _apply_transformers(self, text: str) -> str:
        """Apply custom transformers."""
        result = text
        for transformer in self.transformers:
            result = transformer(result)
        return result
    
    def correct_frame(self, frame: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Correct a single frame.
        
        Returns:
            Tuple of (corrected frame, was_changed)
        """
        text = frame.get('text', '')
        original = text
        
        # Apply all corrections
        text = self._apply_spelling(text)
        text = self._apply_role_fixes(text)
        text = self._apply_patterns(text)
        text = self._apply_transformers(text)
        
        if text != original:
            corrected = frame.copy()
            corrected['text'] = text
            return corrected, True
        
        return frame, False
    
    def correct(self, corpus_data: Dict[str, Any]) -> Tuple[Dict[str, Any], CorrectionResult]:
        """
        Apply corrections to entire corpus.
        
        Returns:
            Tuple of (corrected corpus, correction result)
        """
        frames = corpus_data.get('frames', [])
        result = CorrectionResult()
        
        corrected_frames = []
        for i, frame in enumerate(frames):
            corrected, changed = self.correct_frame(frame)
            corrected_frames.append(corrected)
            
            if changed:
                result.corrected_frames.append(corrected)
                result.corrections.append((
                    str(i),
                    frame.get('text', '')[:50],
                    corrected.get('text', '')[:50]
                ))
            else:
                result.unchanged_count += 1
        
        # Create new corpus
        corrected_corpus = corpus_data.copy()
        corrected_corpus['frames'] = corrected_frames
        
        return corrected_corpus, result


def create_default_corrector() -> CorpusCorrector:
    """Create a corrector with default rules."""
    corrector = CorpusCorrector()
    
    # Add abstract concept role fixes
    abstract_suffixes = ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism']
    
    def is_abstract(text: str) -> bool:
        text_lower = text.lower()
        # Check if subject has abstract suffix
        match = re.search(r'^(\w+)\s+is', text_lower)
        if match:
            subject = match.group(1)
            return any(subject.endswith(s) for s in abstract_suffixes)
        return False
    
    def is_plural(text: str) -> bool:
        text_lower = text.lower()
        match = re.search(r'^(\w+)\s+is', text_lower)
        if match:
            subject = match.group(1)
            return subject.endswith('s') and not subject.endswith('ss') and len(subject) > 3
        return False
    
    corrector.add_role_fix('character', 'concept', is_abstract)
    corrector.add_role_fix('someone', 'concept', is_abstract)
    corrector.add_role_fix('character', 'concept', is_plural)
    corrector.add_role_fix('someone', 'entity', is_plural)
    
    return corrector
