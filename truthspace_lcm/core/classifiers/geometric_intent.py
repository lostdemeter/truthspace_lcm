"""
Geometric Intent Classifier

Fully geometric intent classification using the same principles as KnowledgeSpace:
- Bootstrap patterns define attractor basins (acceptable)
- Runtime matching is geometric (φ-importance, coverage)
- No hardcoded keyword sets or regex patterns

Follows the Emergent Gear Pattern (Design 086):
1. STRUCTURE - Intent categories with bootstrapped examples
2. BOOTSTRAP - Seed with initial examples (the ONLY hardcoding)
3. MATCH - Find intent via geometric similarity
4. LEARN - Refine from usage

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import sys

# Import from hypermapping
hypermapping_parent = Path(__file__).parent.parent.parent.parent
if str(hypermapping_parent) not in sys.path:
    sys.path.insert(0, str(hypermapping_parent))

from hypermapping import HyperMapping, Mapping, TextEncoder, CRITICAL_LINE

# Golden ratio for φ-weighting
PHI = (1 + np.sqrt(5)) / 2


class Intent(Enum):
    """Intent categories for query routing."""
    CODE_GENERATION = "code"
    TOOL_CALL = "tool"
    KNOWLEDGE = "knowledge"
    PLOT_GENERATION = "plot"
    CLARIFICATION = "clarify"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    reason: str = ""
    tool_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeometricIntentSpace(HyperMapping):
    """
    Geometric intent classification using HyperMapping.
    
    Key principles (same as KnowledgeSpace):
    - Bootstrap patterns are acceptable (they seed the structure)
    - Runtime matching uses φ-importance and coverage
    - No hardcoded keyword sets - words emerge from bootstrap data
    - Stop words detected geometrically via coverage > critical line
    """
    
    def __init__(self, dims: int = 8):
        encoder = TextEncoder(dims=dims)
        super().__init__(dims=dims, encoder=encoder, name="intent")
        
        # Word tracking for geometric detection
        self._word_counts: Dict[str, int] = {}
        self._total_patterns: int = 0
        
        # Entity tracking for φ-importance (like KnowledgeSpace)
        self._entities: Dict[str, Dict] = {}  # name -> {frequency, rank, intents}
        self._ranks_computed: bool = False
        
        # Bootstrap
        self._bootstrap_encoder()
        self._bootstrap_intents()
    
    def extract_words(self, text: str) -> Set[str]:
        """
        Extract content words from text.
        
        Geometric stop word detection:
        - Short words (< 3 chars) are structural
        - High coverage words (> critical line) are structural
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return {w for w in words if self._is_content_word(w)}
    
    def _is_content_word(self, word: str) -> bool:
        """
        Geometric stop word detection.
        
        No hardcoded stop word list - detection is emergent:
        1. Short words (< 3 chars) are structural
        2. High coverage (> critical line) = structural
        """
        if len(word) < 3:
            return False
        
        # Geometric detection via coverage
        if word in self._word_counts and self._total_patterns > 1:
            coverage = self._word_counts[word] / self._total_patterns
            if coverage > CRITICAL_LINE:
                return False
        
        return True
    
    def _compute_ranks(self) -> None:
        """Compute ranks based on frequency."""
        sorted_entities = sorted(
            self._entities.items(),
            key=lambda x: -x[1]['frequency']
        )
        for rank, (name, data) in enumerate(sorted_entities, 1):
            data['rank'] = rank
        self._ranks_computed = True
    
    def phi_weight(self, word: str) -> float:
        """φ-based weighting using normalized rank."""
        if not self._ranks_computed:
            self._compute_ranks()
        
        if word not in self._entities:
            return 0.0
        
        rank = self._entities[word]['rank']
        log_rank = np.log1p(rank)
        return PHI ** (-log_rank)
    
    def _add_pattern(self, text: str, intent: Intent, source: str = "bootstrap") -> None:
        """Add a pattern and track word statistics."""
        words = self.extract_words(text)
        
        # Update word counts
        for word in words:
            self._word_counts[word] = self._word_counts.get(word, 0) + 1
            
            # Track entity with intent association
            if word not in self._entities:
                self._entities[word] = {
                    'frequency': 0,
                    'rank': 0,
                    'intents': {},  # intent -> count
                }
            self._entities[word]['frequency'] += 1
            intent_name = intent.value
            self._entities[word]['intents'][intent_name] = \
                self._entities[word]['intents'].get(intent_name, 0) + 1
        
        self._total_patterns += 1
        self._ranks_computed = False
        
        # Create mapping
        position = self.encoder.encode_input(text)
        self.map(
            text,
            intent.value,
            position=position,
            metadata={
                'intent': intent.value,
                'words': list(words),
                'source': source,
            }
        )
    
    def _bootstrap_encoder(self) -> None:
        """Learn word positions from bootstrap patterns."""
        all_patterns = []
        
        # Knowledge queries
        all_patterns.extend([
            "what is", "who is", "how does", "why does",
            "tell me about", "explain", "describe",
            "what are", "when did", "where is",
        ])
        
        # Tool calls
        all_patterns.extend([
            "create file", "delete file", "run command", "execute",
            "list files", "read file", "show contents",
            "search for", "find in", "edit file",
        ])
        
        # Code generation
        all_patterns.extend([
            "write code", "python function", "implement",
            "code that", "program to", "script that",
        ])
        
        # Plot generation
        all_patterns.extend([
            "plot", "graph", "chart", "visualize",
            "sine wave", "cosine wave", "histogram",
            "scatter plot", "bar chart", "line graph",
        ])
        
        self.encoder.learn(all_patterns)
    
    def _bootstrap_intents(self) -> None:
        """
        Bootstrap intent patterns.
        
        These are the ONLY hardcoded patterns - they seed the structure.
        After bootstrap, all matching is geometric.
        """
        # Knowledge queries
        knowledge_patterns = [
            "what is python",
            "who is the author",
            "how does this work",
            "why is this important",
            "tell me about machine learning",
            "explain the concept",
            "describe the process",
            "what are the benefits",
            "when did this happen",
            "where is the file",
        ]
        for pattern in knowledge_patterns:
            self._add_pattern(pattern, Intent.KNOWLEDGE)
        
        # Tool calls - more patterns for better discrimination
        tool_patterns = [
            "create a new file",
            "delete the file",
            "run the command",
            "run the tests",
            "run pytest",
            "execute the script",
            "execute tests",
            "list files in directory",
            "list all files",
            "read the config file",
            "read readme",
            "show contents of readme",
            "search for TODO",
            "search in files",
            "find python files",
            "find all files",
            "edit main.py",
            "cat package.json",
            "ls current directory",
            "grep for pattern",
            "mkdir new folder",
            "rm old file",
        ]
        for pattern in tool_patterns:
            self._add_pattern(pattern, Intent.TOOL_CALL)
        
        # Code generation - more patterns for better discrimination
        code_patterns = [
            "write a python function",
            "write a function to sort",
            "write code to process",
            "implement a class",
            "implement sorting algorithm",
            "code that sorts a list",
            "code to calculate",
            "program to calculate",
            "program that processes",
            "script that processes",
            "function that returns",
            "function to sort a list",
            "algorithm to find",
            "method to compute",
        ]
        for pattern in code_patterns:
            self._add_pattern(pattern, Intent.CODE_GENERATION)
        
        # Plot generation
        plot_patterns = [
            "create a sine wave plot",
            "make a bar chart",
            "generate a histogram",
            "plot a scatter diagram",
            "draw a line graph",
            "visualize the data",
            "create a cosine wave",
            "make a pie chart",
        ]
        for pattern in plot_patterns:
            self._add_pattern(pattern, Intent.PLOT_GENERATION)
        
        # Reproject after bootstrap
        self.reproject()
    
    def text_importance(self, query_words: Set[str], pattern_words: Set[str]) -> float:
        """
        Compute importance between query and pattern using φ-weighting.
        
        Same formula as KnowledgeSpace:
        importance = Σ phi_weight(word)² for matching words
        """
        if not query_words or not pattern_words:
            return 0.0
        
        total = 0.0
        matching = query_words & pattern_words
        
        for word in matching:
            phi = self.phi_weight(word)
            total += phi * phi
        
        return total
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify query intent using geometric matching.
        
        Uses φ-importance to find best matching intent.
        """
        query_words = self.extract_words(query)
        
        if not query_words:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                reason="no content words extracted"
            )
        
        # Compute importance against each pattern
        intent_scores: Dict[str, float] = {}
        best_pattern = None
        best_score = 0.0
        
        for mapping in self._mappings:
            pattern_words = set(mapping.metadata.get('words', []))
            importance = self.text_importance(query_words, pattern_words)
            
            intent_name = mapping.metadata.get('intent', 'unknown')
            intent_scores[intent_name] = intent_scores.get(intent_name, 0.0) + importance
            
            if importance > best_score:
                best_score = importance
                best_pattern = mapping
        
        # Find best intent
        if not intent_scores:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                reason="no matching patterns"
            )
        
        best_intent_name = max(intent_scores, key=intent_scores.get)
        best_intent_score = intent_scores[best_intent_name]
        
        # Normalize confidence
        total_score = sum(intent_scores.values())
        confidence = best_intent_score / total_score if total_score > 0 else 0.0
        
        try:
            intent = Intent(best_intent_name)
        except ValueError:
            intent = Intent.UNKNOWN
        
        # Detect tool hint from query words
        tool_hint = self._detect_tool_hint(query_words)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            reason=f"geometric match (φ-importance={best_intent_score:.4f})",
            tool_hint=tool_hint,
            metadata={
                'scores': intent_scores,
                'query_words': list(query_words),
            }
        )
    
    def _detect_tool_hint(self, query_words: Set[str]) -> Optional[str]:
        """
        Detect which tool might be needed based on learned word-intent associations.
        
        Geometric approach: Find words that are strongly associated with TOOL_CALL
        intent (high frequency in tool patterns, low in others).
        """
        if not self._ranks_computed:
            self._compute_ranks()
        
        # Find words strongly associated with tool intent
        tool_associations = {}
        
        for word in query_words:
            if word not in self._entities:
                continue
            
            entity = self._entities[word]
            intents = entity.get('intents', {})
            tool_count = intents.get('tool', 0)
            total_count = sum(intents.values())
            
            if total_count > 0 and tool_count > 0:
                # Word's affinity for tool intent
                tool_affinity = tool_count / total_count
                if tool_affinity > 0.5:  # Majority tool-associated
                    tool_associations[word] = tool_affinity
        
        # Return the most tool-associated word as hint
        if tool_associations:
            best_word = max(tool_associations, key=tool_associations.get)
            return best_word  # Return the word itself as hint
        
        return None
    
    def learn(self, query: str, correct_intent: Intent, success: bool = True) -> None:
        """
        Learn from feedback.
        
        If classification was wrong, add the query as a new pattern.
        """
        if success:
            # Reinforce existing patterns (position adjustment)
            pass
        else:
            # Add as new pattern for correct intent
            self._add_pattern(query, correct_intent, source="learned")
            self.reproject()
