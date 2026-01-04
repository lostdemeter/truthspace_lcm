"""
Geometric Intent Classifier

Classifies user queries into intent categories using ONLY geometric matching.
No regex patterns, no keyword sets - just holographic projection.

Follows the pattern from Design 084/085:
1. Define similarity (word overlap)
2. Construct positions via eigendecomposition  
3. Match by proximity
4. Inject temporary for unknowns
5. Promote on success (learning)

The intent space is seeded with examples, then learns from usage.
JSON serialization allows persistence across sessions.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set


class Intent(Enum):
    """Intent categories for query routing."""
    CODE_GENERATION = "code"
    TOOL_CALL = "tool"
    KNOWLEDGE = "knowledge"
    CLARIFICATION = "clarify"
    UNSUPPORTED = "unsupported"


@dataclass
class IntentModule:
    """A module in the intent space with holographically projected position."""
    name: str
    text: str
    words: Set[str]
    intent: Intent
    examples: List[str] = field(default_factory=list)
    use_count: int = 0
    success_count: int = 0
    position: Optional[np.ndarray] = None
    temporary: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'text': self.text,
            'words': list(self.words),
            'intent': self.intent.value,
            'examples': self.examples,
            'use_count': self.use_count,
            'success_count': self.success_count,
            'position': self.position.tolist() if self.position is not None else None,
            'temporary': self.temporary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentModule':
        position = np.array(data['position']) if data.get('position') else None
        return cls(
            name=data['name'],
            text=data['text'],
            words=set(data.get('words', [])),
            intent=Intent(data['intent']),
            examples=data.get('examples', []),
            use_count=data.get('use_count', 0),
            success_count=data.get('success_count', 0),
            position=position,
            temporary=data.get('temporary', False),
        )


@dataclass
class IntentMatch:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    reason: str
    module: Optional[IntentModule] = None
    was_injected: bool = False


class GeometricIntentClassifier:
    """
    Classifies queries using holographic pattern space.
    
    NO regex patterns. NO keyword sets. Just geometry.
    
    The space is seeded with intent examples, then positions are
    constructed via eigendecomposition of the word overlap matrix.
    Queries are classified by finding the nearest module.
    
    Unknown queries trigger temporary module injection, which can
    be promoted to permanent on success (learning).
    """
    
    # Filler words to exclude from word extraction
    FILLER = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in',
              'that', 'this', 'is', 'are', 'it', 'be', 'can', 'you', 'i', 'me',
              'my', 'your', 'please', 'could', 'would', 'should'}
    
    def __init__(self, dims: int = 8):
        self.dims = dims
        self.modules: List[IntentModule] = []
        self.positions: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self._seeded = False
    
    @classmethod
    def extract_words(cls, text: str) -> Set[str]:
        """Extract content words from text."""
        words = text.lower().split()
        return {w for w in words if w not in cls.FILLER and len(w) > 1}
    
    @classmethod
    def word_overlap(cls, words1: Set[str], words2: Set[str]) -> float:
        """Jaccard similarity between word sets."""
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def _reproject(self):
        """Construct positions from similarity matrix via eigendecomposition."""
        n = len(self.modules)
        if n == 0:
            self.positions = None
            self.similarity_matrix = None
            return
        
        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = self.word_overlap(self.modules[i].words, self.modules[j].words)
        
        self.similarity_matrix = S
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Take top dims eigenvectors, scaled by sqrt(eigenvalue)
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)  # Ensure non-negative
        self.positions = eigenvectors[:, idx] * np.sqrt(valid_eigenvalues)
        
        # Update module positions
        for i, module in enumerate(self.modules):
            module.position = self.positions[i]
    
    def add_module(self, name: str, text: str, intent: Intent,
                   examples: List[str] = None, temporary: bool = False) -> IntentModule:
        """Add a module and reproject all positions."""
        words = self.extract_words(text)
        module = IntentModule(
            name=name,
            text=text,
            words=words,
            intent=intent,
            examples=examples or [],
            temporary=temporary,
        )
        self.modules.append(module)
        self._reproject()
        return module
    
    def seed(self):
        """Seed the space with intent examples."""
        if self._seeded:
            return
        
        # Code generation examples
        code_examples = [
            ("code_plot", "create a sine wave plot"),
            ("code_chart", "make a bar chart"),
            ("code_histogram", "generate a histogram"),
            ("code_scatter", "plot a scatter diagram"),
            ("code_graph", "draw a line graph"),
            ("code_cosine", "create a cosine wave"),
            ("code_viz", "make a visualization"),
            ("code_function", "write a python function"),
            ("code_script", "generate a script"),
        ]
        for name, text in code_examples:
            self.add_module(name, text, Intent.CODE_GENERATION)
        
        # Tool call examples
        tool_examples = [
            ("tool_list", "list files in directory"),
            ("tool_read", "read the file contents"),
            ("tool_show", "show contents of config"),
            ("tool_run", "run pytest"),
            ("tool_execute", "execute the tests"),
            ("tool_find", "find all python files"),
            ("tool_search", "search for TODO in code"),
            ("tool_edit", "edit the main file"),
            ("tool_cat", "cat package json"),
            ("tool_ls", "ls directory"),
            ("tool_create", "create a new file"),
            ("tool_delete", "delete the file"),
            ("tool_mkdir", "make a directory"),
        ]
        for name, text in tool_examples:
            self.add_module(name, text, Intent.TOOL_CALL)
        
        # Knowledge query examples
        knowledge_examples = [
            ("know_what", "what is a sine wave"),
            ("know_how", "how does matplotlib work"),
            ("know_explain", "explain the difference between bar and histogram"),
            ("know_params", "what are the parameters for scatter plot"),
            ("know_describe", "describe numpy arrays"),
            ("know_tell", "tell me about python decorators"),
            ("know_why", "why use virtual environments"),
            ("know_who", "who is george washington"),
            ("know_define", "define machine learning"),
            ("know_meaning", "what does recursion mean"),
        ]
        for name, text in knowledge_examples:
            self.add_module(name, text, Intent.KNOWLEDGE)
        
        self._seeded = True
    
    def classify(self, query: str, min_similarity: float = 0.15) -> IntentMatch:
        """
        Classify a query by finding the nearest module.
        
        Pure geometric matching - no regex, no keywords.
        """
        if not self._seeded:
            self.seed()
        
        query_words = self.extract_words(query)
        
        if not query_words:
            return IntentMatch(
                intent=Intent.UNSUPPORTED,
                confidence=0.0,
                reason="no content words in query"
            )
        
        # Compute similarity to each module
        similarities = []
        for module in self.modules:
            sim = self.word_overlap(query_words, module.words)
            similarities.append(sim)
        
        if not similarities or max(similarities) == 0:
            # No overlap with any module - inject temporary
            return self._handle_unknown(query, query_words)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        best_module = self.modules[best_idx]
        
        if best_sim < min_similarity:
            # Below threshold - might be unknown
            return self._handle_unknown(query, query_words, best_module, best_sim)
        
        # Good match
        best_module.use_count += 1
        
        return IntentMatch(
            intent=best_module.intent,
            confidence=best_sim,
            reason=f"matched {best_module.name} (overlap={best_sim:.2f})",
            module=best_module,
            was_injected=False,
        )
    
    def _handle_unknown(self, query: str, query_words: Set[str],
                        weak_match: IntentModule = None, 
                        weak_sim: float = 0.0) -> IntentMatch:
        """Handle queries with no good match."""
        # If there's a weak match, use it but with low confidence
        if weak_match and weak_sim > 0:
            return IntentMatch(
                intent=weak_match.intent,
                confidence=weak_sim,
                reason=f"weak match to {weak_match.name} (overlap={weak_sim:.2f})",
                module=weak_match,
                was_injected=False,
            )
        
        # No match at all - inject temporary module
        # Default to KNOWLEDGE for unknown queries (safest)
        temp_module = self.add_module(
            name=f"temp_{len(self.modules)}",
            text=query,
            intent=Intent.KNOWLEDGE,
            temporary=True,
        )
        
        return IntentMatch(
            intent=Intent.KNOWLEDGE,
            confidence=1.0,
            reason="injected temporary module (unknown query)",
            module=temp_module,
            was_injected=True,
        )
    
    def feedback(self, module: IntentModule, success: bool, 
                 correct_intent: Intent = None):
        """
        Provide feedback on a classification.
        
        If successful, increment success count and potentially promote.
        If failed with correct_intent, update the module's intent.
        """
        if success:
            module.success_count += 1
            
            # Promote temporary modules after success
            if module.temporary:
                module.temporary = False
        else:
            if correct_intent and correct_intent != module.intent:
                # Wrong intent - update module
                module.intent = correct_intent
    
    def prune_temporary(self):
        """Remove temporary modules that haven't been promoted."""
        self.modules = [m for m in self.modules if not m.temporary]
        if self.modules:
            self._reproject()
    
    def save(self, path: str):
        """Save the classifier to JSON."""
        data = {
            'version': '1.0',
            'type': 'geometric_intent_classifier',
            'dims': self.dims,
            'modules': [m.to_dict() for m in self.modules if not m.temporary],
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GeometricIntentClassifier':
        """Load a classifier from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        classifier = cls(dims=data.get('dims', 8))
        
        for module_data in data.get('modules', []):
            module = IntentModule.from_dict(module_data)
            classifier.modules.append(module)
        
        classifier._reproject()
        classifier._seeded = True
        
        return classifier


# Convenience function
def create_geometric_classifier() -> GeometricIntentClassifier:
    """Create and seed a geometric intent classifier."""
    classifier = GeometricIntentClassifier()
    classifier.seed()
    return classifier
