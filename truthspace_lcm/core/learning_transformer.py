"""
Learning Concept Transformer - Auto-learning geometric transformer

Extends ConceptTransformer with:
1. Auto-learning: When a concept is unknown, query LLM to learn it
2. Persistence: Save/load learned concepts to disk
3. Integration: Works with the chat API for on-the-fly learning

The flow:
1. User requests transformation for unknown concept
2. Transformer detects unknown concept
3. ConceptPairTrainer queries LLM for transformation pairs
4. Pairs are learned into the geometric space
5. Transformation succeeds
6. Learned concepts are persisted for future use

Author: Lesley Gushurst
License: GPLv3
"""

import json
import logging
import re
import requests
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from truthspace_lcm.core.concept_transformer import (
    ConceptTransformer, ConceptTransformResult, DIMENSION_LEVELS, PHI,
    load_concept_transformer
)

logger = logging.getLogger(__name__)


# =============================================================================
# DIMENSION PROMPTS FOR LLM
# =============================================================================

DIMENSION_PROMPTS = {
    # ==========================================================================
    # GRAMMATICAL DIMENSIONS
    # ==========================================================================
    "tense": {
        "values": ["past", "present", "future"],
        "prompt": """For the word/phrase "{concept}", provide its forms in different tenses.
Format your response as JSON only, no explanation:
{{"past": "<past tense>", "present": "<present tense>", "future": "<future tense>"}}

Example for "go": {{"past": "went", "present": "go", "future": "will go"}}

Now for "{concept}":""",
    },
    "voice": {
        "values": ["active", "passive"],
        "prompt": """For the verb "{concept}", provide active and passive forms.
Format as JSON only: {{"active": "<active>", "passive": "<passive>"}}

Example for "wrote": {{"active": "wrote", "passive": "was written"}}

Now for "{concept}":""",
    },
    "number": {
        "values": ["singular", "plural"],
        "prompt": """For the noun "{concept}", provide singular and plural forms.
Format as JSON only: {{"singular": "<singular>", "plural": "<plural>"}}

Example for "dog": {{"singular": "dog", "plural": "dogs"}}
Example for "child": {{"singular": "child", "plural": "children"}}

Now for "{concept}":""",
    },
    "degree": {
        "values": ["positive", "comparative", "superlative"],
        "prompt": """For the adjective/adverb "{concept}", provide degree forms.
Format as JSON only: {{"positive": "<base>", "comparative": "<more>", "superlative": "<most>"}}

Example for "fast": {{"positive": "fast", "comparative": "faster", "superlative": "fastest"}}
Example for "beautiful": {{"positive": "beautiful", "comparative": "more beautiful", "superlative": "most beautiful"}}

Now for "{concept}":""",
    },
    
    # ==========================================================================
    # SEMANTIC DIMENSIONS
    # ==========================================================================
    "formality": {
        "values": ["casual", "neutral", "formal"],
        "prompt": """For "{concept}", provide forms at different formality levels.
Format as JSON only: {{"casual": "<casual>", "neutral": "<neutral>", "formal": "<formal>"}}

Example for "hello": {{"casual": "hi", "neutral": "hello", "formal": "greetings"}}
Example for "child": {{"casual": "kid", "neutral": "child", "formal": "youth"}}

Now for "{concept}":""",
    },
    "regality": {
        "values": ["common", "noble", "royal"],
        "prompt": """For "{concept}", provide forms at different social status levels.
Format as JSON only: {{"common": "<common>", "noble": "<noble>", "royal": "<royal>"}}

Example for "house": {{"common": "house", "noble": "manor", "royal": "palace"}}
Example for "man": {{"common": "man", "noble": "gentleman", "royal": "king"}}

Now for "{concept}":""",
    },
    "intensity": {
        "values": ["mild", "moderate", "intense"],
        "prompt": """For "{concept}", provide forms at different intensity levels.
Format as JSON only: {{"mild": "<mild>", "moderate": "<moderate>", "intense": "<intense>"}}

Example for "hot": {{"mild": "warm", "moderate": "hot", "intense": "scorching"}}
Example for "happy": {{"mild": "content", "moderate": "happy", "intense": "ecstatic"}}

Now for "{concept}":""",
    },
    "polarity": {
        "values": ["negative", "neutral", "positive"],
        "prompt": """For the concept "{concept}", provide forms with different sentiment.
Format as JSON only: {{"negative": "<negative>", "neutral": "<neutral>", "positive": "<positive>"}}

Example for "quality": {{"negative": "terrible", "neutral": "okay", "positive": "excellent"}}
Example for "feeling": {{"negative": "hate", "neutral": "indifferent", "positive": "love"}}

Now for "{concept}":""",
    },
    "specificity": {
        "values": ["general", "specific", "precise"],
        "prompt": """For "{concept}", provide forms at different specificity levels.
Format as JSON only: {{"general": "<broad category>", "specific": "<specific>", "precise": "<very precise>"}}

Example for "dog": {{"general": "animal", "specific": "dog", "precise": "golden retriever"}}
Example for "car": {{"general": "vehicle", "specific": "car", "precise": "sedan"}}

Now for "{concept}":""",
    },
    "certainty": {
        "values": ["uncertain", "neutral", "certain"],
        "prompt": """For "{concept}", provide forms with different certainty levels.
Format as JSON only: {{"uncertain": "<uncertain>", "neutral": "<neutral>", "certain": "<certain>"}}

Example for "think": {{"uncertain": "might think", "neutral": "think", "certain": "know"}}
Example for "probably": {{"uncertain": "perhaps", "neutral": "probably", "certain": "definitely"}}

Now for "{concept}":""",
    },
    "emotion": {
        "values": ["sad", "neutral", "happy"],
        "prompt": """For "{concept}", provide emotionally varied forms.
Format as JSON only: {{"sad": "<sad version>", "neutral": "<neutral>", "happy": "<happy version>"}}

Example for "said": {{"sad": "sighed", "neutral": "said", "happy": "exclaimed"}}
Example for "expression": {{"sad": "frown", "neutral": "neutral face", "happy": "smile"}}

Now for "{concept}":""",
    },
    "size": {
        "values": ["small", "medium", "large"],
        "prompt": """For "{concept}", provide forms at different size scales.
Format as JSON only: {{"small": "<small>", "medium": "<medium>", "large": "<large>"}}

Example for "house": {{"small": "cottage", "medium": "house", "large": "mansion"}}
Example for "dog": {{"small": "puppy", "medium": "dog", "large": "hound"}}

Now for "{concept}":""",
    },
    "speed": {
        "values": ["slow", "medium", "fast"],
        "prompt": """For "{concept}", provide forms at different speed levels.
Format as JSON only: {{"slow": "<slow>", "medium": "<medium>", "fast": "<fast>"}}

Example for "walk": {{"slow": "stroll", "medium": "walk", "fast": "stride"}}
Example for "run": {{"slow": "jog", "medium": "run", "fast": "sprint"}}

Now for "{concept}":""",
    },
    "age": {
        "values": ["young", "adult", "old"],
        "prompt": """For "{concept}", provide forms at different life stages.
Format as JSON only: {{"young": "<young>", "adult": "<adult>", "old": "<old>"}}

Example for "person": {{"young": "child", "adult": "adult", "old": "elder"}}
Example for "dog": {{"young": "puppy", "adult": "dog", "old": "old dog"}}

Now for "{concept}":""",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LearnedConcept:
    """A concept that was learned via LLM."""
    concept: str
    dimension: str
    forms: Dict[str, str]  # value -> form (e.g., {"past": "went", "future": "will go"})
    learned_at: str
    source: str = "llm"  # "llm", "manual", "corpus"


@dataclass 
class LearningResult:
    """Result of attempting to learn a concept."""
    concept: str
    dimension: str
    success: bool
    pairs_learned: int = 0
    forms: Dict[str, str] = field(default_factory=dict)
    error: str = ""


# =============================================================================
# LEARNING CONCEPT TRANSFORMER
# =============================================================================

class LearningConceptTransformer(ConceptTransformer):
    """
    ConceptTransformer with auto-learning capabilities.
    
    When a transformation fails due to unknown concept:
    1. Query LLM for transformation pairs
    2. Learn pairs into geometric space
    3. Retry transformation
    4. Persist learned concepts
    
    Usage:
        transformer = LearningConceptTransformer()
        transformer.load_corpus(corpus_path)  # Load base vocabulary
        transformer.load_learned()  # Load previously learned concepts
        
        # This will auto-learn "jumped" if unknown
        result = transformer.transform_sentence(
            "The dog jumped", "tense", "past", "future",
            auto_learn=True
        )
        # Returns: "The dog will jump"
    """
    
    def __init__(self,
                 ollama_url: str = "http://127.0.0.1:11434/api/generate",
                 model: str = "qwen2.5:14b",
                 learned_path: Path = None,
                 auto_save: bool = True):
        """
        Initialize the learning transformer.
        
        Args:
            ollama_url: URL for Ollama API
            model: Model to use for learning
            learned_path: Path to save/load learned concepts
            auto_save: Whether to auto-save after learning
        """
        super().__init__()
        
        self.ollama_url = ollama_url
        self.model = model
        self.auto_save = auto_save
        
        # Default learned path
        if learned_path is None:
            learned_path = Path.home() / ".truthspace" / "learned_concepts.json"
        self.learned_path = Path(learned_path)
        
        # Track learned concepts
        self._learned_concepts: Dict[Tuple[str, str], LearnedConcept] = {}
        self._learning_attempts: Set[Tuple[str, str]] = set()  # Track failed attempts
    
    # =========================================================================
    # LLM INTERACTION
    # =========================================================================
    
    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query the local LLM."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, str]]:
        """Parse JSON from LLM response."""
        if not response:
            return None
        
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            return None
        
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try to fix common issues
            json_str = json_match.group()
            json_str = re.sub(r',\s*}', '}', json_str)
            try:
                return json.loads(json_str)
            except:
                return None
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available."""
        try:
            response = requests.get(
                self.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn_concept(self, 
                      concept: str, 
                      dimension: str) -> LearningResult:
        """
        Learn a concept's transformation forms via LLM.
        
        Args:
            concept: The concept to learn (word or phrase)
            dimension: The dimension to learn (e.g., "tense")
            
        Returns:
            LearningResult with success status and learned forms
        """
        result = LearningResult(concept=concept, dimension=dimension, success=False)
        
        # Check if we have a prompt for this dimension
        if dimension not in DIMENSION_PROMPTS:
            result.error = f"Unknown dimension: {dimension}"
            return result
        
        dim_config = DIMENSION_PROMPTS[dimension]
        prompt = dim_config["prompt"].format(concept=concept)
        values = dim_config["values"]
        
        # Query LLM
        logger.info(f"Learning concept '{concept}' for dimension '{dimension}'...")
        response = self._query_llm(prompt)
        
        if not response:
            result.error = "LLM query failed"
            return result
        
        # Parse response
        forms = self._parse_json_response(response)
        if not forms:
            result.error = f"Failed to parse LLM response: {response[:100]}"
            return result
        
        result.forms = forms
        
        # Generate and learn pairs
        pairs_learned = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                src_val = values[i]
                tgt_val = values[j]
                
                src_form = forms.get(src_val, "").lower().strip()
                tgt_form = forms.get(tgt_val, "").lower().strip()
                
                if src_form and tgt_form and src_form != tgt_form:
                    # Learn this pair
                    self._learn_pair(src_form, tgt_form, dimension, src_val, tgt_val)
                    pairs_learned += 1
        
        if pairs_learned > 0:
            result.success = True
            result.pairs_learned = pairs_learned
            
            # Record learned concept
            learned = LearnedConcept(
                concept=concept,
                dimension=dimension,
                forms=forms,
                learned_at=datetime.now().isoformat(),
                source="llm"
            )
            self._learned_concepts[(concept.lower(), dimension)] = learned
            
            # Recompute deltas
            self._compute_deltas()
            
            # Auto-save
            if self.auto_save:
                self.save_learned()
            
            logger.info(f"Learned {pairs_learned} pairs for '{concept}' ({dimension})")
        else:
            result.error = "No valid pairs generated"
        
        return result
    
    def _learn_pair(self, 
                    src_phrase: str, 
                    tgt_phrase: str,
                    dimension: str,
                    src_val: str,
                    tgt_val: str) -> None:
        """Learn a single transformation pair."""
        # Assign to same concept
        self._assign_concept(src_phrase, tgt_phrase)
        
        # Record pair
        self._pairs[dimension].append((src_phrase, tgt_phrase, src_val, tgt_val))
        
        # Compute positions
        key_src = (src_phrase, dimension, src_val)
        key_tgt = (tgt_phrase, dimension, tgt_val)
        
        if key_src not in self._positions:
            self._positions[key_src] = self._get_position(src_phrase, dimension, src_val)
        if key_tgt not in self._positions:
            self._positions[key_tgt] = self._get_position(tgt_phrase, dimension, tgt_val)
    
    def learn_manual(self,
                     forms: Dict[str, str],
                     dimension: str,
                     concept_name: str = None) -> LearningResult:
        """
        Manually learn a concept with provided forms.
        
        Args:
            forms: Dict of value -> form (e.g., {"past": "went", "future": "will go"})
            dimension: Dimension name
            concept_name: Optional name for the concept
            
        Returns:
            LearningResult
        """
        if dimension not in DIMENSION_PROMPTS:
            return LearningResult(
                concept=concept_name or "unknown",
                dimension=dimension,
                success=False,
                error=f"Unknown dimension: {dimension}"
            )
        
        values = DIMENSION_PROMPTS[dimension]["values"]
        pairs_learned = 0
        
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                src_val = values[i]
                tgt_val = values[j]
                
                src_form = forms.get(src_val, "").lower().strip()
                tgt_form = forms.get(tgt_val, "").lower().strip()
                
                if src_form and tgt_form and src_form != tgt_form:
                    self._learn_pair(src_form, tgt_form, dimension, src_val, tgt_val)
                    pairs_learned += 1
        
        if pairs_learned > 0:
            self._compute_deltas()
            
            # Record
            concept_key = concept_name or list(forms.values())[0]
            learned = LearnedConcept(
                concept=concept_key,
                dimension=dimension,
                forms=forms,
                learned_at=datetime.now().isoformat(),
                source="manual"
            )
            self._learned_concepts[(concept_key.lower(), dimension)] = learned
            
            if self.auto_save:
                self.save_learned()
        
        return LearningResult(
            concept=concept_name or "manual",
            dimension=dimension,
            success=pairs_learned > 0,
            pairs_learned=pairs_learned,
            forms=forms
        )
    
    def learn_concept_multi(self, 
                            concept: str, 
                            dimensions: List[str] = None) -> Dict[str, LearningResult]:
        """
        Learn a concept across multiple dimensions.
        
        Args:
            concept: The concept to learn
            dimensions: List of dimensions to learn (default: all applicable)
            
        Returns:
            Dict mapping dimension -> LearningResult
        """
        if dimensions is None:
            dimensions = list(DIMENSION_PROMPTS.keys())
        
        results = {}
        for dim in dimensions:
            if dim in DIMENSION_PROMPTS:
                results[dim] = self.learn_concept(concept, dim)
        
        return results
    
    @staticmethod
    def available_dimensions() -> Dict[str, List[str]]:
        """Get all available dimensions and their values."""
        return {
            dim: config["values"] 
            for dim, config in DIMENSION_PROMPTS.items()
        }
    
    @staticmethod
    def dimension_info(dimension: str) -> Optional[Dict]:
        """Get info about a specific dimension."""
        if dimension in DIMENSION_PROMPTS:
            return {
                "name": dimension,
                "values": DIMENSION_PROMPTS[dimension]["values"],
            }
        return None
    
    # =========================================================================
    # AUTO-LEARNING TRANSFORM
    # =========================================================================
    
    def transform_sentence_auto(self,
                                sentence: str,
                                dimension: str,
                                source_value: str,
                                target_value: str) -> ConceptTransformResult:
        """
        Transform sentence with auto-learning for unknown concepts.
        
        If transformation fails due to unknown concept, attempts to
        learn the concept via LLM and retry.
        """
        # First try normal transform
        result = self.transform_sentence(sentence, dimension, source_value, target_value)
        
        if result.success:
            return result
        
        # Check if failure was due to unknown phrase
        failure_lower = result.failure_reason.lower()
        if "no matching" not in failure_lower and "not found" not in failure_lower and "unknown" not in failure_lower:
            return result
        
        # Extract the phrase that wasn't found
        # The failure reason contains the phrase we tried to find
        unknown_phrase = self._extract_unknown_phrase(sentence, dimension, source_value)
        
        if not unknown_phrase:
            return result
        
        # Check if we already tried to learn this
        attempt_key = (unknown_phrase.lower(), dimension)
        if attempt_key in self._learning_attempts:
            return result
        
        self._learning_attempts.add(attempt_key)
        
        # Try to learn the concept
        logger.info(f"Auto-learning unknown concept: '{unknown_phrase}' ({dimension})")
        learn_result = self.learn_concept(unknown_phrase, dimension)
        
        if not learn_result.success:
            logger.warning(f"Failed to learn '{unknown_phrase}': {learn_result.error}")
            return result
        
        # Retry transformation
        return self.transform_sentence(sentence, dimension, source_value, target_value)
    
    def _extract_unknown_phrase(self, 
                                sentence: str, 
                                dimension: str,
                                source_value: str) -> Optional[str]:
        """Extract the phrase that needs to be learned."""
        # Get all known phrases for this dimension/value
        known_phrases = set()
        for (phrase, dim, val), _ in self._positions.items():
            if dim == dimension and val == source_value:
                known_phrases.add(phrase.lower())
        
        # Also get all known phrases regardless of dimension (to avoid common words)
        all_known = set()
        for (phrase, _, _), _ in self._positions.items():
            all_known.add(phrase.lower())
        
        # Common words to skip
        skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'and', 'or', 'but', 'not', 'no', 'yes', 'if', 'then',
                      'that', 'this', 'it', 'he', 'she', 'they', 'we', 'you', 'i',
                      'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        # Tokenize sentence and find unknown words
        words = self.tokenize(sentence)
        
        # For tense dimension, prioritize words that look like verbs
        if dimension == "tense":
            # First pass: look for past tense verbs (-ed ending)
            for word in words:
                word_lower = word.lower()
                if word_lower in skip_words or word_lower in all_known:
                    continue
                if word_lower.endswith('ed') and len(word) > 3:
                    return word
            
            # Second pass: look for -ing verbs
            for word in words:
                word_lower = word.lower()
                if word_lower in skip_words or word_lower in all_known:
                    continue
                if word_lower.endswith('ing') and len(word) > 4:
                    return word
            
            # Third pass: any unknown word that's not too short
            for word in words:
                word_lower = word.lower()
                if word_lower in skip_words or word_lower in all_known:
                    continue
                if len(word) > 3:
                    return word
        else:
            # For other dimensions, return first unknown content word
            for word in words:
                word_lower = word.lower()
                if word_lower in skip_words or word_lower in all_known:
                    continue
                if len(word) > 2:
                    return word
        
        return None
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_learned(self, path: Path = None) -> bool:
        """Save learned concepts to disk."""
        path = path or self.learned_path
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize learned concepts
            data = {
                "version": 1,
                "saved_at": datetime.now().isoformat(),
                "concepts": [
                    {
                        "concept": lc.concept,
                        "dimension": lc.dimension,
                        "forms": lc.forms,
                        "learned_at": lc.learned_at,
                        "source": lc.source,
                    }
                    for lc in self._learned_concepts.values()
                ]
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self._learned_concepts)} learned concepts to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save learned concepts: {e}")
            return False
    
    def load_learned(self, path: Path = None) -> int:
        """
        Load previously learned concepts from disk.
        
        Returns number of concepts loaded.
        """
        path = path or self.learned_path
        
        if not path.exists():
            return 0
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            count = 0
            for item in data.get("concepts", []):
                concept = item["concept"]
                dimension = item["dimension"]
                forms = item["forms"]
                
                # Re-learn the pairs - use DIMENSION_PROMPTS for values
                dim_config = DIMENSION_PROMPTS.get(dimension, {})
                values = dim_config.get("values", [])
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        src_val = values[i]
                        tgt_val = values[j]
                        
                        src_form = forms.get(src_val, "").lower().strip()
                        tgt_form = forms.get(tgt_val, "").lower().strip()
                        
                        if src_form and tgt_form and src_form != tgt_form:
                            self._learn_pair(src_form, tgt_form, dimension, src_val, tgt_val)
                
                # Record
                learned = LearnedConcept(
                    concept=concept,
                    dimension=dimension,
                    forms=forms,
                    learned_at=item.get("learned_at", ""),
                    source=item.get("source", "loaded")
                )
                self._learned_concepts[(concept.lower(), dimension)] = learned
                count += 1
            
            # Recompute deltas
            if count > 0:
                self._compute_deltas()
            
            logger.info(f"Loaded {count} learned concepts from {path}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load learned concepts: {e}")
            return 0
    
    def forget_concept(self, concept: str, dimension: str = None) -> bool:
        """
        Forget a learned concept.
        
        Args:
            concept: Concept to forget
            dimension: Specific dimension (or all if None)
            
        Returns:
            True if concept was forgotten
        """
        concept_lower = concept.lower()
        removed = False
        
        keys_to_remove = []
        for (c, d) in self._learned_concepts.keys():
            if c == concept_lower:
                if dimension is None or d == dimension:
                    keys_to_remove.append((c, d))
        
        for key in keys_to_remove:
            del self._learned_concepts[key]
            removed = True
        
        if removed and self.auto_save:
            self.save_learned()
        
        return removed
    
    def clear_learned(self) -> int:
        """Clear all learned concepts. Returns count cleared."""
        count = len(self._learned_concepts)
        self._learned_concepts.clear()
        self._learning_attempts.clear()
        
        if self.auto_save:
            self.save_learned()
        
        return count
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> Dict:
        """Get statistics including learned concepts."""
        base_stats = super().stats()
        base_stats["learned_concepts"] = len(self._learned_concepts)
        base_stats["learning_attempts"] = len(self._learning_attempts)
        base_stats["llm_available"] = self.is_llm_available()
        base_stats["available_dimensions"] = list(DIMENSION_PROMPTS.keys())
        base_stats["dimension_count"] = len(DIMENSION_PROMPTS)
        return base_stats
    
    def learned_concepts_list(self) -> List[Dict]:
        """Get list of learned concepts with details."""
        return [
            {
                "concept": lc.concept,
                "dimension": lc.dimension,
                "forms": lc.forms,
                "learned_at": lc.learned_at,
                "source": lc.source,
            }
            for lc in self._learned_concepts.values()
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_learning_transformer(
    corpus_path: Path = None,
    learned_path: Path = None,
    **kwargs
) -> LearningConceptTransformer:
    """
    Load a learning transformer with corpus and learned concepts.
    
    Args:
        corpus_path: Path to transformation corpus (default: built-in)
        learned_path: Path to learned concepts (default: ~/.truthspace/learned_concepts.json)
        **kwargs: Additional args for LearningConceptTransformer
        
    Returns:
        Configured LearningConceptTransformer
    """
    if corpus_path is None:
        corpus_path = Path(__file__).parent.parent / "corpus" / "transformation_corpus.json"
    
    transformer = LearningConceptTransformer(learned_path=learned_path, **kwargs)
    
    # Load base corpus
    if corpus_path.exists():
        transformer.load_corpus(corpus_path)
        logger.info(f"Loaded corpus: {transformer.stats()['phrases']} phrases")
    
    # Load learned concepts
    learned_count = transformer.load_learned()
    if learned_count > 0:
        logger.info(f"Loaded {learned_count} previously learned concepts")
    
    return transformer
