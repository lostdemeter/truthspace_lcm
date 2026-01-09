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
    tool_name: Optional[str] = None  # For TOOL_CALL: which tool type
    tool_args: Dict[str, Any] = field(default_factory=dict)  # Extracted arguments


# Golden ratio for φ-Zipf duality (Design 039)
PHI = (1 + np.sqrt(5)) / 2

# Bootstrap filler - minimal set used until modules are seeded
# After seeding, filler is derived via φ-Zipf duality
BOOTSTRAP_FILLER = {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'and', 'or', 'in'}

# φ-weight threshold: words with φ^(-rank) >= this are filler (most frequent)
# φ^(-1) = 0.618, φ^(-2) = 0.382, φ^(-3) = 0.236
# Only the top 2 most frequent words are filler (threshold > φ^(-3))
PHI_FILLER_WEIGHT_THRESHOLD = 0.25


class GeometricIntentClassifier:
    """
    Classifies queries using holographic pattern space.
    
    NO regex patterns. NO keyword sets. Just geometry.
    
    The space is seeded with intent examples, then positions are
    constructed via eigendecomposition of the word overlap matrix.
    Queries are classified by finding the nearest module.
    
    Filler words are derived via φ-Zipf duality (Design 039):
    - Words are ranked by frequency across modules
    - φ^(-rank) gives geometric weight
    - High weight (high frequency) words are filler
    - This follows the principle: The geometry IS the weighting
    
    Unknown queries trigger temporary module injection, which can
    be promoted to permanent on success (learning).
    """
    
    def __init__(self, dims: int = 8):
        self.dims = dims
        self.modules: List[IntentModule] = []
        self.positions: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self._seeded = False
        
        # Emergent filler - derived via φ-Zipf duality after seeding
        self._filler: Set[str] = BOOTSTRAP_FILLER.copy()
        self._word_phi_weight: Dict[str, float] = {}  # word -> φ^(-rank) weight
        self._word_frequency: Dict[str, int] = {}  # word -> raw frequency
    
    def _extract_all_words(self, text: str) -> Set[str]:
        """Extract ALL words from text (no filtering)."""
        return {w for w in text.lower().split() if len(w) > 1}
    
    def extract_words(self, text: str) -> Set[str]:
        """Extract content words from text (filler removed)."""
        words = self._extract_all_words(text)
        return words - self._filler
    
    def _derive_filler_words(self):
        """
        Derive filler words via φ-Zipf duality (Design 039).
        
        Instead of statistical analysis, we use geometric φ-weighting:
        1. Count word frequencies across all modules
        2. Rank words by frequency (most frequent = rank 1)
        3. Compute φ^(-rank) weight for each word
        4. Words with high φ-weight (high frequency) are filler
        
        This follows the principle: The geometry IS the weighting.
        φ^n for encoding (outward), φ^(-n) for weighting (inward).
        Same fractal, opposite directions.
        """
        from collections import Counter
        
        if not self.modules:
            return
        
        # Count total word occurrences across all modules
        word_freq: Counter = Counter()
        for module in self.modules:
            if module.temporary:
                continue
            words = self._extract_all_words(module.text)
            word_freq.update(words)
        
        if not word_freq:
            return
        
        # Store raw frequencies
        self._word_frequency = dict(word_freq)
        
        # Rank words by frequency (most frequent = rank 1)
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        
        # Compute φ^(-rank) weight and identify filler
        filler = set()
        for rank, (word, freq) in enumerate(sorted_words, 1):
            # φ-Zipf duality: weight = φ^(-rank)
            # Rank 1 (most frequent) → φ^(-1) = 0.618 (high weight)
            # Rank 100 (rare) → φ^(-100) ≈ 0 (low weight)
            phi_weight = PHI ** (-rank)
            self._word_phi_weight[word] = phi_weight
            
            # HIGH weight = HIGH frequency = filler
            if phi_weight >= PHI_FILLER_WEIGHT_THRESHOLD:
                filler.add(word)
        
        # Combine with bootstrap filler
        self._filler = filler | BOOTSTRAP_FILLER
    
    def get_filler_info(self) -> Dict[str, Any]:
        """Get information about filler words (for debugging)."""
        emergent = self._filler - BOOTSTRAP_FILLER
        return {
            "bootstrap": sorted(BOOTSTRAP_FILLER),
            "emergent": sorted(emergent),
            "all": sorted(self._filler),
            "total": len(self._filler),
            "phi_weight_threshold": PHI_FILLER_WEIGHT_THRESHOLD,
            "phi_weights": sorted(
                [(w, self._word_phi_weight.get(w, 0), self._word_frequency.get(w, 0))
                 for w in self._filler if w in self._word_phi_weight],
                key=lambda x: -x[2]  # Sort by frequency
            )[:20],
            "content_words": sorted(
                [(w, weight, self._word_frequency.get(w, 0))
                 for w, weight in self._word_phi_weight.items()
                 if weight < PHI_FILLER_WEIGHT_THRESHOLD],
                key=lambda x: -x[1]  # Sort by φ-weight
            )[:10],
        }
    
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
        
        # Derive filler words emergently from module statistics
        self._derive_filler_words()
        
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
        
        # Extract tool info if this is a TOOL_CALL intent
        tool_name, tool_args = self._extract_tool_info(query, best_module)
        
        return IntentMatch(
            intent=best_module.intent,
            confidence=best_sim,
            reason=f"matched {best_module.name} (overlap={best_sim:.2f})",
            module=best_module,
            was_injected=False,
            tool_name=tool_name,
            tool_args=tool_args,
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
    
    def _extract_tool_info(self, query: str, module: IntentModule) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extract tool name and arguments from query based on matched module.
        
        Uses geometric matching via module name patterns, not regex.
        The module name encodes the tool type (e.g., tool_list -> Glob, tool_read -> Read).
        Arguments are extracted by finding non-filler words not in the module template.
        """
        if module.intent != Intent.TOOL_CALL:
            return None, {}
        
        # Map module names to tool types (geometric: module name IS the tool type)
        MODULE_TO_TOOL = {
            'tool_list': 'Glob',
            'tool_ls': 'Glob',
            'tool_find': 'Glob',
            'tool_read': 'Read',
            'tool_show': 'Read',
            'tool_cat': 'Read',
            'tool_run': 'Bash',
            'tool_execute': 'Bash',
            'tool_search': 'Grep',
            'tool_edit': 'Edit',
            'tool_create': 'Write',
            'tool_delete': 'Bash',
            'tool_mkdir': 'Bash',
        }
        
        tool_name = MODULE_TO_TOOL.get(module.name, 'Bash')  # Default to Bash
        
        # Extract arguments geometrically: words in query but not in module template
        query_words = self.extract_words(query)
        template_words = module.words
        arg_words = query_words - template_words - self._filler
        
        # Build arguments based on tool type
        tool_args = {}
        if tool_name == 'Glob':
            # Path argument - join remaining words
            path = ' '.join(sorted(arg_words)) if arg_words else '.'
            tool_args = {'pattern': path}
        elif tool_name == 'Read':
            # File path argument
            path = ' '.join(sorted(arg_words)) if arg_words else ''
            tool_args = {'file_path': path}
        elif tool_name == 'Grep':
            # Search query
            search = ' '.join(sorted(arg_words)) if arg_words else ''
            tool_args = {'query': search}
        elif tool_name == 'Bash':
            # Command - use full query minus common words
            cmd = ' '.join(sorted(arg_words)) if arg_words else ''
            tool_args = {'command': cmd}
        elif tool_name in ('Write', 'Edit'):
            # File path
            path = ' '.join(sorted(arg_words)) if arg_words else ''
            tool_args = {'file_path': path}
        
        return tool_name, tool_args
    
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
