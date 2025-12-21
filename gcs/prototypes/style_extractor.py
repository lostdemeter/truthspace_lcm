#!/usr/bin/env python3
"""
Style Extractor

Extract style from ANY data source and apply it on-the-fly.

Use cases:
1. Feed in Q&A pairs → extract "Q&A style" → apply to new content
2. Feed in an author's writing → extract their style → apply to new content
3. Feed in technical docs → extract "technical style" → apply to new content

The key insight: Style IS the centroid of exemplars.
To extract a style, just compute the centroid of the input text chunks.
"""

import sys
import os
import re
import math
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# VOCABULARY (same as style_centroid.py but standalone)
# ============================================================================

class Vocabulary:
    """Word embeddings with hash-based positions and style biasing."""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.word_positions: Dict[str, np.ndarray] = {}
        self.word_counts: Counter = Counter()
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _base_position(self, word: str) -> np.ndarray:
        """Deterministic position from hash."""
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim)
        np.random.seed(None)
        return pos
    
    def add_text(self, text: str, style_bias: Optional[np.ndarray] = None, bias_strength: float = 0.2):
        """Add text to vocabulary, optionally biasing toward a style."""
        words = self._tokenize(text)
        self.word_counts.update(words)
        
        for word in words:
            if word not in self.word_positions:
                self.word_positions[word] = self._base_position(word)
            
            if style_bias is not None:
                self.word_positions[word] = (
                    (1 - bias_strength) * self.word_positions[word] + 
                    bias_strength * style_bias
                )
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text as IDF-weighted average of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        positions = []
        weights = []
        
        for word in words:
            if word in self.word_positions:
                positions.append(self.word_positions[word])
                count = self.word_counts.get(word, 1)
                weight = 1.0 / math.log(1 + count)
                weights.append(weight)
            else:
                # Add new word on-the-fly
                self.word_positions[word] = self._base_position(word)
                self.word_counts[word] = 1
                positions.append(self.word_positions[word])
                weights.append(1.0)
        
        if not positions:
            return np.zeros(self.dim)
        
        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(positions, axis=0, weights=weights)
    
    def nearest_words(self, vector: np.ndarray, k: int = 10, exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """Find nearest words to a vector."""
        if exclude is None:
            exclude = set()
        
        vec_norm = np.linalg.norm(vector)
        if vec_norm < 1e-8:
            return []
        
        results = []
        for word, pos in self.word_positions.items():
            if word in exclude:
                continue
            pos_norm = np.linalg.norm(pos)
            if pos_norm < 1e-8:
                continue
            sim = np.dot(vector, pos) / (vec_norm * pos_norm)
            results.append((word, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]


# ============================================================================
# EXTRACTED STYLE
# ============================================================================

@dataclass
class ExtractedStyle:
    """A style extracted from data."""
    name: str
    centroid: np.ndarray
    exemplar_count: int
    characteristic_words: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'centroid': self.centroid.tolist(),
            'exemplar_count': self.exemplar_count,
            'characteristic_words': self.characteristic_words,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractedStyle':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            centroid=np.array(data['centroid']),
            exemplar_count=data['exemplar_count'],
            characteristic_words=data.get('characteristic_words', []),
            metadata=data.get('metadata', {}),
        )


# ============================================================================
# STYLE EXTRACTOR
# ============================================================================

class StyleExtractor:
    """
    Extract styles from any data source.
    
    Usage:
        extractor = StyleExtractor()
        
        # Extract from text
        style = extractor.extract_from_text(text, "AuthorStyle")
        
        # Extract from file
        style = extractor.extract_from_file("book.txt", "BookStyle")
        
        # Extract from Q&A pairs
        style = extractor.extract_from_qa_pairs(pairs, "QAStyle")
        
        # Apply style
        result = extractor.apply_style(content, style)
        
        # Classify content
        best_style = extractor.classify(content)
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab = Vocabulary(dim)
        self.styles: Dict[str, ExtractedStyle] = {}
    
    def _split_into_chunks(self, text: str, chunk_type: str = 'sentence') -> List[str]:
        """Split text into chunks for style extraction."""
        if chunk_type == 'sentence':
            # Split on sentence boundaries
            chunks = re.split(r'(?<=[.!?])\s+', text)
        elif chunk_type == 'paragraph':
            # Split on double newlines
            chunks = re.split(r'\n\n+', text)
        elif chunk_type == 'line':
            # Split on single newlines
            chunks = text.split('\n')
        else:
            # Treat whole text as one chunk
            chunks = [text]
        
        # Filter empty chunks and strip whitespace
        chunks = [c.strip() for c in chunks if c.strip()]
        return chunks
    
    def _compute_centroid(self, chunks: List[str]) -> np.ndarray:
        """Compute centroid from chunks."""
        if not chunks:
            return np.zeros(self.dim)
        
        # First pass: add all text to vocabulary
        for chunk in chunks:
            self.vocab.add_text(chunk)
        
        # Second pass: compute rough centroid
        positions = [self.vocab.encode(chunk) for chunk in chunks]
        rough_centroid = np.mean(positions, axis=0)
        
        # Third pass: re-add with style bias
        for chunk in chunks:
            self.vocab.add_text(chunk, style_bias=rough_centroid, bias_strength=0.15)
        
        # Final centroid
        positions = [self.vocab.encode(chunk) for chunk in chunks]
        centroid = np.mean(positions, axis=0)
        
        return centroid
    
    def extract_from_text(self, text: str, name: str, chunk_type: str = 'sentence') -> ExtractedStyle:
        """
        Extract style from raw text.
        
        Args:
            text: The source text
            name: Name for the extracted style
            chunk_type: How to split text ('sentence', 'paragraph', 'line', 'whole')
        
        Returns:
            ExtractedStyle object
        """
        chunks = self._split_into_chunks(text, chunk_type)
        centroid = self._compute_centroid(chunks)
        
        # Find characteristic words
        char_words = self.vocab.nearest_words(centroid, k=15)
        
        style = ExtractedStyle(
            name=name,
            centroid=centroid,
            exemplar_count=len(chunks),
            characteristic_words=char_words,
            metadata={'chunk_type': chunk_type, 'source': 'text'},
        )
        
        self.styles[name] = style
        return style
    
    def extract_from_file(self, filepath: str, name: str, chunk_type: str = 'sentence') -> ExtractedStyle:
        """Extract style from a file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        text = path.read_text(encoding='utf-8', errors='ignore')
        style = self.extract_from_text(text, name, chunk_type)
        style.metadata['source'] = str(filepath)
        
        return style
    
    def extract_from_qa_pairs(self, pairs: List[Tuple[str, str]], name: str) -> ExtractedStyle:
        """
        Extract style from Q&A pairs.
        
        Args:
            pairs: List of (question, answer) tuples
            name: Name for the extracted style
        
        Returns:
            ExtractedStyle object
        """
        # Combine Q&A pairs into chunks
        chunks = []
        for q, a in pairs:
            chunks.append(f"{q} {a}")
        
        centroid = self._compute_centroid(chunks)
        char_words = self.vocab.nearest_words(centroid, k=15)
        
        style = ExtractedStyle(
            name=name,
            centroid=centroid,
            exemplar_count=len(pairs),
            characteristic_words=char_words,
            metadata={'source': 'qa_pairs', 'pair_count': len(pairs)},
        )
        
        self.styles[name] = style
        return style
    
    def extract_from_exemplars(self, exemplars: List[str], name: str) -> ExtractedStyle:
        """Extract style from a list of exemplar strings."""
        centroid = self._compute_centroid(exemplars)
        char_words = self.vocab.nearest_words(centroid, k=15)
        
        style = ExtractedStyle(
            name=name,
            centroid=centroid,
            exemplar_count=len(exemplars),
            characteristic_words=char_words,
            metadata={'source': 'exemplars'},
        )
        
        self.styles[name] = style
        return style
    
    def similarity(self, text: str, style: Union[str, ExtractedStyle]) -> float:
        """Compute cosine similarity between text and style."""
        if isinstance(style, str):
            if style not in self.styles:
                raise ValueError(f"Unknown style: {style}")
            style = self.styles[style]
        
        vec = self.vocab.encode(text)
        vec_norm = np.linalg.norm(vec)
        cent_norm = np.linalg.norm(style.centroid)
        
        if vec_norm < 1e-8 or cent_norm < 1e-8:
            return 0.0
        
        return np.dot(vec, style.centroid) / (vec_norm * cent_norm)
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text against all known styles."""
        results = []
        for name, style in self.styles.items():
            sim = self.similarity(text, style)
            results.append((name, sim))
        
        results.sort(key=lambda x: -x[1])
        return results
    
    def apply_style(self, text: str, target_style: Union[str, ExtractedStyle], 
                    strength: float = 0.5) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Apply style to text.
        
        Returns:
            (styled_vector, nearest_words_to_styled_vector)
        """
        if isinstance(target_style, str):
            if target_style not in self.styles:
                raise ValueError(f"Unknown style: {target_style}")
            target_style = self.styles[target_style]
        
        vec = self.vocab.encode(text)
        styled_vec = (1 - strength) * vec + strength * target_style.centroid
        
        # Find nearest words (excluding words already in text)
        text_words = set(re.findall(r'\w+', text.lower()))
        nearest = self.vocab.nearest_words(styled_vec, k=15, exclude=text_words)
        
        return styled_vec, nearest
    
    def style_difference(self, style_a: Union[str, ExtractedStyle], 
                         style_b: Union[str, ExtractedStyle]) -> Tuple[np.ndarray, List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Compute difference between two styles.
        
        Returns:
            (direction_vector, words_toward_b, words_toward_a)
        """
        if isinstance(style_a, str):
            style_a = self.styles[style_a]
        if isinstance(style_b, str):
            style_b = self.styles[style_b]
        
        direction = style_b.centroid - style_a.centroid
        
        toward_b = self.vocab.nearest_words(direction, k=10)
        toward_a = self.vocab.nearest_words(-direction, k=10)
        
        return direction, toward_b, toward_a
    
    def save_styles(self, filepath: str):
        """Save all extracted styles to JSON."""
        data = {name: style.to_dict() for name, style in self.styles.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_styles(self, filepath: str):
        """Load styles from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, style_data in data.items():
            self.styles[name] = ExtractedStyle.from_dict(style_data)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  STYLE EXTRACTOR")
    print("  Extract style from ANY data source, apply on-the-fly")
    print("=" * 70)
    
    extractor = StyleExtractor(dim=64)
    
    # -------------------------------------------------------------------------
    # Example 1: Extract style from Q&A pairs
    # -------------------------------------------------------------------------
    print("\n[1] Extracting Q&A style from pairs...")
    print("-" * 70)
    
    qa_pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("When did World War II end?", "World War II ended in 1945."),
        ("Where is the Great Wall located?", "The Great Wall is located in China."),
        ("How does photosynthesis work?", "Photosynthesis converts sunlight into energy."),
        ("Why do leaves change color?", "Leaves change color due to chlorophyll breakdown."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 km/s."),
        ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
    ]
    
    qa_style = extractor.extract_from_qa_pairs(qa_pairs, "Q&A")
    print(f"  Extracted Q&A style from {qa_style.exemplar_count} pairs")
    print(f"  Characteristic words: {', '.join(w for w, s in qa_style.characteristic_words[:8])}")
    
    # -------------------------------------------------------------------------
    # Example 2: Extract style from author's writing
    # -------------------------------------------------------------------------
    print("\n\n[2] Extracting author styles from text samples...")
    print("-" * 70)
    
    # Hemingway-style (short, direct sentences)
    hemingway_text = """
    The old man was thin and gaunt. He had deep wrinkles in the back of his neck.
    The brown blotches ran well down the sides of his face. His hands had deep scars.
    But none of these scars were fresh. They were as old as erosions in a fishless desert.
    He was an old man who fished alone in a skiff in the Gulf Stream.
    He had gone eighty-four days now without taking a fish.
    The sail was patched with flour sacks. It looked like the flag of permanent defeat.
    """
    
    hemingway_style = extractor.extract_from_text(hemingway_text, "Hemingway", chunk_type='sentence')
    print(f"  Hemingway style: {hemingway_style.exemplar_count} sentences")
    print(f"  Characteristic: {', '.join(w for w, s in hemingway_style.characteristic_words[:8])}")
    
    # Lovecraft-style (cosmic horror, elaborate)
    lovecraft_text = """
    The most merciful thing in the world is the inability of the human mind to correlate all its contents.
    We live on a placid island of ignorance in the midst of black seas of infinity.
    It was not meant that we should voyage far beyond the boundaries of our own solar system.
    The sciences, each straining in its own direction, have hitherto harmed us little.
    But some day the piecing together of dissociated knowledge will open up such terrifying vistas.
    We shall either go mad from the revelation or flee from the deadly light into the peace of a new dark age.
    That is not dead which can eternal lie, and with strange aeons even death may die.
    """
    
    lovecraft_style = extractor.extract_from_text(lovecraft_text, "Lovecraft", chunk_type='sentence')
    print(f"  Lovecraft style: {lovecraft_style.exemplar_count} sentences")
    print(f"  Characteristic: {', '.join(w for w, s in lovecraft_style.characteristic_words[:8])}")
    
    # Technical documentation style
    technical_text = """
    The function accepts two parameters: an integer and a string.
    It returns a boolean value indicating success or failure.
    If the input is invalid, an exception will be raised.
    The time complexity of this algorithm is O(n log n).
    Memory usage scales linearly with input size.
    See the API reference for additional configuration options.
    The module must be imported before use.
    Initialize the client with your API key and endpoint URL.
    """
    
    technical_style = extractor.extract_from_text(technical_text, "Technical", chunk_type='sentence')
    print(f"  Technical style: {technical_style.exemplar_count} sentences")
    print(f"  Characteristic: {', '.join(w for w, s in technical_style.characteristic_words[:8])}")
    
    # -------------------------------------------------------------------------
    # Example 3: Classify new text
    # -------------------------------------------------------------------------
    print("\n\n[3] Classifying new text against extracted styles...")
    print("-" * 70)
    
    test_texts = [
        "What is the meaning of life?",
        "The man sat alone. He drank his coffee. It was cold.",
        "The function returns null if the parameter is undefined.",
        "In his house at R'lyeh, dead Cthulhu waits dreaming.",
        "Who discovered America?",
        "The API endpoint accepts POST requests with JSON body.",
    ]
    
    for text in test_texts:
        print(f"\n  \"{text[:50]}...\"" if len(text) > 50 else f"\n  \"{text}\"")
        scores = extractor.classify(text)
        top = scores[0]
        print(f"    Best: {top[0]} ({top[1]:.3f})")
        print(f"    All: {', '.join(f'{n}={s:.2f}' for n, s in scores)}")
    
    # -------------------------------------------------------------------------
    # Example 4: Apply style to content
    # -------------------------------------------------------------------------
    print("\n\n[4] Applying styles to neutral content...")
    print("-" * 70)
    
    neutral = "The man walked down the street and entered the building."
    print(f"\n  Neutral: \"{neutral}\"")
    
    for style_name in ["Hemingway", "Lovecraft", "Technical", "Q&A"]:
        _, nearest = extractor.apply_style(neutral, style_name, strength=0.6)
        words = [w for w, s in nearest[:6]]
        print(f"    + {style_name:12s} → {', '.join(words)}")
    
    # -------------------------------------------------------------------------
    # Example 5: Style differences
    # -------------------------------------------------------------------------
    print("\n\n[5] Style differences (what makes one style different from another)...")
    print("-" * 70)
    
    pairs = [
        ("Hemingway", "Lovecraft"),
        ("Technical", "Q&A"),
        ("Hemingway", "Technical"),
    ]
    
    for style_a, style_b in pairs:
        _, toward_b, toward_a = extractor.style_difference(style_a, style_b)
        print(f"\n  {style_a} → {style_b}:")
        print(f"    Toward {style_b}: {', '.join(w for w, s in toward_b[:5])}")
        print(f"    Toward {style_a}: {', '.join(w for w, s in toward_a[:5])}")
    
    # -------------------------------------------------------------------------
    # Key insight
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHT")
    print("=" * 70)
    print("""
  Style extraction is SIMPLE:
  
    1. Split input into chunks (sentences, paragraphs, Q&A pairs)
    2. Compute centroid = average(encode(chunk) for chunk in chunks)
    3. That centroid IS the style
  
  To apply on-the-fly:
  
    style = extractor.extract_from_text(author_writing, "AuthorStyle")
    result = extractor.apply_style(my_content, style)
  
  Works with ANY data source:
    - Raw text files
    - Q&A pairs
    - Code documentation
    - Chat logs
    - Any collection of text
  
  The style IS the centroid. Extraction IS averaging.
  No training, no neural networks - just geometry.
""")


if __name__ == "__main__":
    demo()
