"""
Style System for Geometric Chat System

Implements style extraction, classification, and transfer as specified in the SDS.

Core formulas:
- Style Centroid: c = (1/n) Σᵢ enc(exemplarᵢ)
- Style Transfer: styled = (1-α)·content + α·centroid
- Style Classification: argmax_s cosine(enc(text), s.centroid)
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from .vocabulary import Vocabulary, tokenize, cosine_similarity


@dataclass
class Style:
    """
    A style extracted from exemplars.
    
    The centroid IS the style - it captures the average
    semantic position of all exemplars.
    """
    name: str
    centroid: np.ndarray
    exemplar_count: int
    characteristic_words: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def similarity(self, text_vec: np.ndarray) -> float:
        """Compute similarity of text to this style."""
        return cosine_similarity(text_vec, self.centroid)
    
    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            'name': self.name,
            'centroid': self.centroid.tolist(),
            'exemplar_count': self.exemplar_count,
            'characteristic_words': self.characteristic_words,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Style':
        """Deserialize from JSON dict."""
        return cls(
            name=data['name'],
            centroid=np.array(data['centroid']),
            exemplar_count=data['exemplar_count'],
            characteristic_words=data.get('characteristic_words', []),
            metadata=data.get('metadata', {})
        )


class StyleEngine:
    """
    Style extraction, classification, and transfer engine.
    
    Implements the centroid approach validated at 8/8 and 6/6 accuracy.
    """
    
    def __init__(self, vocab: Vocabulary = None):
        self.vocab = vocab or Vocabulary()
        self.styles: Dict[str, Style] = {}
    
    def extract_style(self, exemplars: List[str], name: str) -> Style:
        """
        Extract style from exemplars.
        
        Algorithm:
        1. Encode each exemplar
        2. Compute centroid (mean position)
        3. Find characteristic words
        
        The centroid IS the style.
        """
        if not exemplars:
            raise ValueError("Cannot extract style from empty exemplars")
        
        # Add exemplars to vocabulary
        for ex in exemplars:
            self.vocab.add_text(ex)
        
        # Encode all exemplars
        encodings = [self.vocab.encode(ex) for ex in exemplars]
        
        # Compute centroid
        centroid = np.mean(encodings, axis=0)
        
        # Find characteristic words (nearest to centroid)
        all_words = set()
        for ex in exemplars:
            all_words.update(tokenize(ex))
        
        char_words = self.vocab.nearest_words(centroid, k=15)
        
        style = Style(
            name=name,
            centroid=centroid,
            exemplar_count=len(exemplars),
            characteristic_words=char_words
        )
        
        self.styles[name] = style
        return style
    
    def extract_from_text(self, text: str, name: str, 
                          chunk_by: str = 'sentence') -> Style:
        """
        Extract style from raw text by chunking into exemplars.
        
        Args:
            text: Raw text to extract style from
            name: Name for the style
            chunk_by: How to split text ('sentence', 'paragraph', 'line')
        """
        if chunk_by == 'sentence':
            # Simple sentence splitting
            import re
            exemplars = re.split(r'[.!?]+', text)
        elif chunk_by == 'paragraph':
            exemplars = text.split('\n\n')
        else:  # line
            exemplars = text.split('\n')
        
        # Filter empty chunks
        exemplars = [e.strip() for e in exemplars if e.strip()]
        
        return self.extract_style(exemplars, name)
    
    def extract_from_qa_pairs(self, pairs: List[Tuple[str, str]], 
                               name: str) -> Style:
        """
        Extract style from Q&A pairs.
        
        Combines questions and answers as exemplars.
        """
        exemplars = []
        for q, a in pairs:
            exemplars.append(q)
            exemplars.append(a)
        
        return self.extract_style(exemplars, name)
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify text against known styles.
        
        Returns: [(style_name, similarity), ...] sorted descending
        """
        if not self.styles:
            return []
        
        text_vec = self.vocab.encode(text)
        
        results = []
        for name, style in self.styles.items():
            sim = cosine_similarity(text_vec, style.centroid)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])
    
    def transfer(self, content: str, style_name: str, 
                 strength: float = 0.5) -> Tuple[np.ndarray, List[str]]:
        """
        Transfer content toward target style.
        
        Formula:
            styled_vec = (1 - α) · content_vec + α · style_centroid
        
        Returns:
            (styled_vector, suggested_words)
        """
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        style = self.styles[style_name]
        content_vec = self.vocab.encode(content)
        
        # Interpolate toward style centroid
        styled_vec = (1 - strength) * content_vec + strength * style.centroid
        
        # Find words that characterize the styled result
        content_words = set(tokenize(content))
        nearest = self.vocab.nearest_words(styled_vec, k=15, exclude=content_words)
        
        return styled_vec, [word for word, sim in nearest]
    
    def style_difference(self, style_a: str, style_b: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute what makes style_b different from style_a.
        
        Formula:
            direction = centroid_b - centroid_a
        
        Returns:
            (direction_vector, words_toward_b, words_toward_a)
        """
        if style_a not in self.styles or style_b not in self.styles:
            raise ValueError(f"Unknown style(s)")
        
        a = self.styles[style_a]
        b = self.styles[style_b]
        
        direction = b.centroid - a.centroid
        
        toward_b = self.vocab.nearest_words(direction, k=10)
        toward_a = self.vocab.nearest_words(-direction, k=10)
        
        return direction, toward_b, toward_a
    
    def save_styles(self, filepath: str):
        """Save all styles to JSON file."""
        data = {name: style.to_dict() for name, style in self.styles.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_styles(self, filepath: str):
        """Load styles from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, style_data in data.items():
            self.styles[name] = Style.from_dict(style_data)
