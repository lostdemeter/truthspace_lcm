"""
Linguistic Output Chain

A concrete implementation of EmergentDimensionChain for conditioning
output into natural language based on sentence structure patterns.

This chain discovers dimensions from how sentences are structured,
enabling style-aware output generation.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import hashlib
from typing import Dict, Any, List
from .base_chain import EmergentDimensionChain, DataItem


class LinguisticChain(EmergentDimensionChain):
    """
    Chain for conditioning output via sentence structure patterns.
    
    Features extracted:
    - Sentence length (short/medium/long)
    - Punctuation patterns
    - Verb tense indicators
    - Complexity markers (conjunctions, relative clauses)
    - Style indicators (adverbs, formality)
    
    Groups: Sentence structure types (clustered by features)
    """
    
    def __init__(self, name: str = "LinguisticChain"):
        super().__init__(name)
        self.text_key = 'text'
        self.min_group_count = 1  # Each sentence is its own group
    
    def extract_features(self, item: Any) -> Dict[str, float]:
        """Extract structural features from a sentence."""
        if isinstance(item, dict):
            text = item.get(self.text_key, '')
        else:
            text = str(item)
        
        if not text:
            return {}
        
        words = text.split()
        text_lower = text.lower()
        
        features = {
            # Length features
            'len_short': 1.0 if len(words) < 8 else 0.0,
            'len_medium': 1.0 if 8 <= len(words) < 15 else 0.0,
            'len_long': 1.0 if len(words) >= 15 else 0.0,
            
            # Punctuation
            'has_comma': 1.0 if ',' in text else 0.0,
            'has_question': 1.0 if '?' in text else 0.0,
            'has_exclaim': 1.0 if '!' in text else 0.0,
            
            # Structure
            'starts_name': 1.0 if text[0].isupper() and not text_lower.startswith(('the ', 'a ', 'an ')) else 0.0,
            'starts_the': 1.0 if text_lower.startswith('the ') else 0.0,
            
            # Tense indicators
            'has_past': 1.0 if re.search(r'\b\w+ed\b', text_lower) else 0.0,
            'has_present': 1.0 if re.search(r'\b\w+s\b', text_lower) else 0.0,
            'has_progressive': 1.0 if re.search(r'\b\w+ing\b', text_lower) else 0.0,
            
            # Complexity
            'has_conjunction': 1.0 if any(w in text_lower.split() for w in ['and', 'but', 'or', 'while']) else 0.0,
            'has_relative': 1.0 if any(w in text_lower.split() for w in ['who', 'which', 'that']) else 0.0,
            
            # Style
            'has_adverb': 1.0 if re.search(r'\b\w+ly\b', text_lower) else 0.0,
            'is_descriptive': 1.0 if any(w in text_lower for w in [' is ', ' are ', ' was ', ' were ']) else 0.0,
            'is_action': 1.0 if re.search(r'^[A-Z]\w+\s+\w+s?\b', text) else 0.0,
        }
        
        return features
    
    def get_item_id(self, item: Any) -> str:
        """Generate a unique ID for each sentence."""
        if isinstance(item, dict):
            text = item.get(self.text_key, '')
        else:
            text = str(item)
        
        if not text or len(text) < 10:
            return ''
        
        # Use hash of text as ID
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def get_item_content(self, item: Any) -> Any:
        """Store the text content."""
        if isinstance(item, dict):
            return item.get(self.text_key, '')
        return str(item)
    
    def find_similar_sentences(self, target_features: Dict[str, float], k: int = 5) -> List[str]:
        """
        Find sentences with similar structural features.
        
        Args:
            target_features: Feature dictionary to match
            k: Number of results
            
        Returns:
            List of similar sentence texts
        """
        if not self.items or self.Vt is None:
            return []
        
        # Build target feature vector
        import numpy as np
        target_vec = np.array([target_features.get(f, 0.0) for f in self.feature_names])
        target_vec = target_vec - 0.5  # Rough centering
        
        # Project to dimension space
        target_pos = target_vec @ self.Vt[:len(self.dimensions)].T
        
        # Find closest items
        items = self.find_items_near(target_pos, k)
        return [item.content for item in items]
    
    def get_templates_for_style(self, style: str, k: int = 5) -> List[str]:
        """
        Get sentence templates matching a style.
        
        Styles:
        - 'action': Agent-focused action sentences
        - 'descriptive': Descriptive sentences with 'is/are'
        - 'complex': Longer sentences with conjunctions
        - 'simple': Short, direct sentences
        """
        style_features = {
            'action': {'is_action': 1.0, 'starts_name': 1.0, 'len_short': 0.5},
            'descriptive': {'is_descriptive': 1.0, 'has_adverb': 0.5, 'len_medium': 1.0},
            'complex': {'len_long': 1.0, 'has_conjunction': 1.0, 'has_comma': 1.0},
            'simple': {'len_short': 1.0, 'has_comma': 0.0, 'has_conjunction': 0.0},
        }
        
        features = style_features.get(style, {})
        return self.find_similar_sentences(features, k)
    
    def adapt_sentence(self, template: str, substitutions: Dict[str, str]) -> str:
        """
        Adapt a template sentence with substitutions.
        
        Args:
            template: The template sentence
            substitutions: Dict of old_value -> new_value
            
        Returns:
            Adapted sentence
        """
        result = template
        for old, new in substitutions.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            result = pattern.sub(new, result)
        return result
