"""
Corpus Pruner

Identifies and removes bad data from the corpus based on:
- Frame length (too short/long)
- Duplication
- Typos and malformed words
- Incorrect roles

Author: Lesley Gushurst
License: GPLv3
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class PruneResult:
    """Results from a pruning operation."""
    removed_frames: List[Dict[str, Any]] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=lambda: Counter())
    original_count: int = 0
    final_count: int = 0
    
    @property
    def removed_count(self) -> int:
        return len(self.removed_frames)
    
    def summary(self) -> str:
        lines = [
            f"Pruning Results:",
            f"  Original frames: {self.original_count}",
            f"  Removed frames: {self.removed_count}",
            f"  Final frames: {self.final_count}",
            f"  Reasons:"
        ]
        for reason, count in self.reasons.most_common():
            lines.append(f"    - {reason}: {count}")
        return "\n".join(lines)


class CorpusPruner:
    """
    Prunes bad data from the corpus.
    
    Usage:
        pruner = CorpusPruner()
        pruner.set_min_length(25)
        pruner.set_max_duplicates(10)
        pruner.add_typo_pattern(r'\\bwierd\\b', 'weird')
        
        result = pruner.prune(corpus_data)
        print(result.summary())
    """
    
    def __init__(self):
        self.min_length = 25
        self.max_length = 1000
        self.max_duplicates = 10
        
        # Typo patterns: (pattern, correction)
        self.typo_patterns: List[Tuple[str, str]] = [
            (r'\boccured\b', 'occurred'),
            (r'\boccurence\b', 'occurrence'),
            (r'\brecieve\b', 'receive'),
            (r'\bseperate\b', 'separate'),
            (r'\bdefinately\b', 'definitely'),
        ]
        
        # Words that indicate bad frames
        self.bad_indicators: Set[str] = {
            'undefined', 'null', 'error', 'nan', 'none',
        }
        
        # Role corrections
        self.abstract_suffixes = [
            'ology', 'ics', 'istry', 'tion', 'ment', 'ness',
            'ism', 'ure', 'ance', 'ence', 'ity', 'ty'
        ]
    
    def set_min_length(self, length: int) -> 'CorpusPruner':
        """Set minimum frame length."""
        self.min_length = length
        return self
    
    def set_max_length(self, length: int) -> 'CorpusPruner':
        """Set maximum frame length."""
        self.max_length = length
        return self
    
    def set_max_duplicates(self, count: int) -> 'CorpusPruner':
        """Set maximum allowed duplicates."""
        self.max_duplicates = count
        return self
    
    def add_typo_pattern(self, pattern: str, correction: str) -> 'CorpusPruner':
        """Add a typo pattern to check."""
        self.typo_patterns.append((pattern, correction))
        return self
    
    def add_bad_indicator(self, word: str) -> 'CorpusPruner':
        """Add a word that indicates a bad frame."""
        self.bad_indicators.add(word.lower())
        return self
    
    def _check_length(self, frame: Dict[str, Any]) -> Optional[str]:
        """Check if frame length is acceptable."""
        text = frame.get('text', '')
        if len(text) < self.min_length:
            return 'too_short'
        if len(text) > self.max_length:
            return 'too_long'
        return None
    
    def _check_typos(self, frame: Dict[str, Any]) -> Optional[str]:
        """Check for typos in frame."""
        text = frame.get('text', '').lower()
        for pattern, _ in self.typo_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'has_typo'
        return None
    
    def _check_bad_indicators(self, frame: Dict[str, Any]) -> Optional[str]:
        """Check for bad indicator words."""
        text = frame.get('text', '').lower()
        for indicator in self.bad_indicators:
            if indicator in text:
                return 'bad_indicator'
        return None
    
    def _check_role(self, frame: Dict[str, Any]) -> Optional[str]:
        """Check for incorrect roles."""
        text = frame.get('text', '').lower()
        
        # Check if abstract concept is labeled as character
        if 'is a character' in text or 'is someone' in text:
            # Extract the subject
            match = re.search(r'^(\w+)\s+is\s+a', text)
            if match:
                subject = match.group(1)
                # Check if it's an abstract concept
                if any(subject.endswith(s) for s in self.abstract_suffixes):
                    return 'wrong_role'
                # Check if it's plural
                if subject.endswith('s') and not subject.endswith('ss'):
                    return 'wrong_role'
        
        return None
    
    def analyze(self, corpus_data: Dict[str, Any]) -> PruneResult:
        """
        Analyze corpus without modifying it.
        
        Returns a PruneResult with frames that would be removed.
        """
        frames = corpus_data.get('frames', [])
        result = PruneResult(original_count=len(frames))
        
        # Track duplicates
        text_counts: Counter = Counter()
        for frame in frames:
            text_counts[frame.get('text', '')] += 1
        
        for frame in frames:
            text = frame.get('text', '')
            
            # Check length
            reason = self._check_length(frame)
            if reason:
                result.removed_frames.append(frame)
                result.reasons[reason] += 1
                continue
            
            # Check duplicates
            if text_counts[text] > self.max_duplicates:
                result.removed_frames.append(frame)
                result.reasons['duplicate'] += 1
                text_counts[text] -= 1  # Only flag excess duplicates
                continue
            
            # Check typos
            reason = self._check_typos(frame)
            if reason:
                result.removed_frames.append(frame)
                result.reasons[reason] += 1
                continue
            
            # Check bad indicators
            reason = self._check_bad_indicators(frame)
            if reason:
                result.removed_frames.append(frame)
                result.reasons[reason] += 1
                continue
            
            # Check role
            reason = self._check_role(frame)
            if reason:
                result.removed_frames.append(frame)
                result.reasons[reason] += 1
                continue
        
        result.final_count = result.original_count - result.removed_count
        return result
    
    def prune(self, corpus_data: Dict[str, Any], 
              dry_run: bool = False) -> Tuple[Dict[str, Any], PruneResult]:
        """
        Prune bad frames from corpus.
        
        Args:
            corpus_data: The corpus to prune
            dry_run: If True, don't modify the corpus
            
        Returns:
            Tuple of (pruned corpus, prune result)
        """
        result = self.analyze(corpus_data)
        
        if dry_run:
            return corpus_data, result
        
        # Build set of texts to remove
        remove_texts = {f.get('text', '') for f in result.removed_frames}
        
        # Filter frames
        pruned_frames = [
            f for f in corpus_data.get('frames', [])
            if f.get('text', '') not in remove_texts
        ]
        
        # Create new corpus
        pruned_corpus = corpus_data.copy()
        pruned_corpus['frames'] = pruned_frames
        
        return pruned_corpus, result
