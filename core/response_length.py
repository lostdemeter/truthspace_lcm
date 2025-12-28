"""
Response Length Control for GeometricLCM

Enforces token/word budgets on generated responses while maintaining
coherence by truncating at natural boundaries (sentences, clauses).

Key principles:
1. Never cut mid-sentence
2. Prefer complete thoughts over arbitrary truncation
3. Add continuation markers when truncated
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class LengthStats:
    """Statistics about response length."""
    words: int
    sentences: int
    estimated_tokens: int
    was_truncated: bool
    truncation_point: Optional[str] = None


class ResponseLengthController:
    """
    Controls response length by enforcing token/word budgets.
    
    Truncates at natural boundaries to maintain coherence.
    """
    
    # Approximate tokens per word (English average)
    TOKENS_PER_WORD = 1.3
    
    def __init__(self, max_tokens: int = 500, min_tokens: int = 10):
        """
        Initialize the length controller.
        
        Args:
            max_tokens: Maximum tokens allowed
            min_tokens: Minimum tokens before truncation is allowed
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
    
    @property
    def max_words(self) -> int:
        """Maximum words based on token budget."""
        return int(self.max_tokens / self.TOKENS_PER_WORD)
    
    @property
    def min_words(self) -> int:
        """Minimum words before truncation."""
        return int(self.min_tokens / self.TOKENS_PER_WORD)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        words = len(text.split())
        return int(words * self.TOKENS_PER_WORD)
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Filter empty strings
        return len([s for s in sentences if s.strip()])
    
    def truncate(self, text: str, add_continuation: bool = True) -> Tuple[str, LengthStats]:
        """
        Truncate text to fit within token budget.
        
        Args:
            text: Text to truncate
            add_continuation: Whether to add "..." when truncated
        
        Returns:
            Tuple of (truncated_text, stats)
        """
        words = text.split()
        original_word_count = len(words)
        
        # Check if truncation needed
        if len(words) <= self.max_words:
            return text, LengthStats(
                words=len(words),
                sentences=self.count_sentences(text),
                estimated_tokens=self.estimate_tokens(text),
                was_truncated=False,
            )
        
        # Find truncation point at sentence boundary
        truncated = self._truncate_at_sentence(text, self.max_words)
        
        # If sentence truncation didn't work, try clause boundary
        if len(truncated.split()) > self.max_words:
            truncated = self._truncate_at_clause(text, self.max_words)
        
        # Last resort: word boundary
        if len(truncated.split()) > self.max_words:
            truncated = ' '.join(words[:self.max_words])
            if add_continuation and not truncated.endswith(('...', '.', '!', '?')):
                truncated = truncated.rstrip('.,;:') + '...'
        
        return truncated, LengthStats(
            words=len(truncated.split()),
            sentences=self.count_sentences(truncated),
            estimated_tokens=self.estimate_tokens(truncated),
            was_truncated=True,
            truncation_point=f"word {len(truncated.split())} of {original_word_count}",
        )
    
    def _truncate_at_sentence(self, text: str, max_words: int) -> str:
        """Truncate at the last complete sentence within word limit."""
        # Find all sentence boundaries
        sentence_ends = []
        for match in re.finditer(r'[.!?]+\s*', text):
            end_pos = match.end()
            words_so_far = len(text[:end_pos].split())
            if words_so_far <= max_words:
                sentence_ends.append(end_pos)
        
        if sentence_ends:
            # Take text up to last valid sentence end
            return text[:sentence_ends[-1]].strip()
        
        # No complete sentence fits
        return text
    
    def _truncate_at_clause(self, text: str, max_words: int) -> str:
        """Truncate at clause boundary (comma, semicolon, etc.)."""
        # Find clause boundaries
        clause_ends = []
        for match in re.finditer(r'[,;:]\s*', text):
            end_pos = match.end()
            words_so_far = len(text[:end_pos].split())
            if words_so_far <= max_words:
                clause_ends.append(end_pos)
        
        if clause_ends:
            truncated = text[:clause_ends[-1]].rstrip(',;: ')
            return truncated + '...'
        
        return text
    
    def fits_budget(self, text: str) -> bool:
        """Check if text fits within token budget."""
        return self.estimate_tokens(text) <= self.max_tokens
    
    def remaining_budget(self, text: str) -> int:
        """Get remaining token budget after text."""
        used = self.estimate_tokens(text)
        return max(0, self.max_tokens - used)
    
    def can_add(self, current: str, addition: str) -> bool:
        """Check if addition can be added within budget."""
        combined = current + " " + addition if current else addition
        return self.fits_budget(combined)


class IncrementalBuilder:
    """
    Build responses incrementally while respecting length limits.
    
    Useful for streaming or building responses piece by piece.
    """
    
    def __init__(self, controller: ResponseLengthController):
        self.controller = controller
        self.parts: List[str] = []
        self.current_tokens = 0
    
    def add(self, text: str) -> bool:
        """
        Add text to response if it fits.
        
        Returns:
            True if text was added, False if it would exceed budget
        """
        tokens = self.controller.estimate_tokens(text)
        
        if self.current_tokens + tokens <= self.controller.max_tokens:
            self.parts.append(text)
            self.current_tokens += tokens
            return True
        
        return False
    
    def add_sentence(self, sentence: str) -> bool:
        """Add a complete sentence if it fits."""
        # Ensure sentence ends with punctuation
        if sentence and not sentence.endswith(('.', '!', '?')):
            sentence = sentence + '.'
        return self.add(sentence)
    
    def build(self) -> str:
        """Build the final response."""
        return ' '.join(self.parts)
    
    def remaining(self) -> int:
        """Get remaining token budget."""
        return self.controller.max_tokens - self.current_tokens
    
    def is_full(self) -> bool:
        """Check if budget is exhausted."""
        return self.remaining() < 10  # Less than ~7 words


def create_length_controller(max_tokens: int = 500, 
                             depth: float = 0.0) -> ResponseLengthController:
    """
    Create a length controller with depth-adjusted limits.
    
    Args:
        max_tokens: Base token limit
        depth: Depth dial value (-1 to +1)
    
    Returns:
        Configured ResponseLengthController
    """
    # Adjust max_tokens based on depth
    # Terse (depth=-1): reduce by 50%
    # Elaborate (depth=+1): increase by 50%
    adjustment = 1.0 + (depth * 0.5)
    adjusted_max = int(max_tokens * adjustment)
    
    # Ensure reasonable bounds
    adjusted_max = max(50, min(2000, adjusted_max))
    
    return ResponseLengthController(max_tokens=adjusted_max)
