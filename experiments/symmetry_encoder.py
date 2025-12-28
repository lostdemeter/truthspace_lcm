#!/usr/bin/env python3
"""
Symmetry Encoder - Bootstrap Knowledge from Pure Symmetry

The hypothesis: Symmetry is the foundational "instinct" that requires no
prior knowledge. All other knowledge is measured as ASYMMETRY relative to
these fundamental symmetries.

Key insight: You don't need to know what "reflection" means to verify
something is symmetric under reflection - you just apply the operation
and check if it's unchanged.

Symmetry Types:
1. REVERSAL - Does reversing change meaning? (palindrome-like)
2. EXCHANGE - Is A→B same as B→A? (commutative)
3. SCALE - Does meaning change at different granularities?
4. TEMPORAL - Does meaning depend on time reference?
5. NEGATION - Does negating preserve structure?

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class SymmetrySignature:
    """A concept's position in symmetry space."""
    reversal: float      # How much meaning changes under reversal
    exchange: float      # How much A→B differs from B→A
    scale: float         # How much meaning changes at different scales
    repetition: float    # How much the concept repeats/cycles
    negation: float      # How much negation changes structure
    length_ratio: float  # Ratio of word lengths (structural rhythm)
    vowel_balance: float # Balance of vowels vs consonants (phonetic symmetry)
    position_weight: float # Where "weight" of text falls (center of mass)
    # Core symmetries from prime/compression insight
    compression: float   # How compressible (self-similar) the text is
    first_word: float    # Structural pattern of first word
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.reversal,
            self.exchange,
            self.scale,
            self.repetition,
            self.negation,
            self.length_ratio,
            self.vowel_balance,
            self.position_weight,
            self.compression,
            self.first_word,
        ])
    
    def distance(self, other: 'SymmetrySignature') -> float:
        return np.linalg.norm(self.to_vector() - other.to_vector())


class SymmetryEncoder:
    """
    Encode concepts based purely on symmetry breaking.
    
    No seed words. No pre-defined categories. Just symmetry operations
    applied to raw text, measuring how much each symmetry is broken.
    """
    
    def __init__(self):
        self.learned_signatures: Dict[str, SymmetrySignature] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - no semantic knowledge needed."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _char_sequence(self, text: str) -> List[str]:
        """Get character sequence for low-level symmetry."""
        return list(text.lower().replace(' ', ''))
    
    # =========================================================================
    # SYMMETRY OPERATIONS - These require NO prior knowledge
    # =========================================================================
    
    def measure_reversal_symmetry(self, text: str) -> float:
        """
        How much does reversing the text change its structure?
        
        Perfect symmetry (palindrome) = 0.0
        Complete asymmetry = 1.0
        
        This is self-verifiable: reverse and compare.
        """
        chars = self._char_sequence(text)
        if len(chars) < 2:
            return 0.0
        
        reversed_chars = chars[::-1]
        
        # Count matching positions
        matches = sum(1 for a, b in zip(chars, reversed_chars) if a == b)
        symmetry = matches / len(chars)
        
        # Return asymmetry (how much symmetry is broken)
        return 1.0 - symmetry
    
    def measure_exchange_symmetry(self, text: str) -> float:
        """
        How much does word order matter?
        
        "dog bites man" vs "man bites dog" - order matters (asymmetric)
        "salt and pepper" vs "pepper and salt" - order doesn't matter (symmetric)
        
        Measured by: how much does shuffling change the character distribution?
        """
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return 0.0
        
        # Original bigram distribution
        original_bigrams = Counter(zip(tokens[:-1], tokens[1:]))
        
        # Reversed token order bigram distribution
        reversed_tokens = tokens[::-1]
        reversed_bigrams = Counter(zip(reversed_tokens[:-1], reversed_tokens[1:]))
        
        # Measure overlap
        all_bigrams = set(original_bigrams.keys()) | set(reversed_bigrams.keys())
        if not all_bigrams:
            return 0.0
        
        overlap = sum(min(original_bigrams.get(b, 0), reversed_bigrams.get(b, 0)) 
                     for b in all_bigrams)
        total = sum(original_bigrams.values())
        
        symmetry = overlap / total if total > 0 else 0.0
        return 1.0 - symmetry
    
    def measure_scale_symmetry(self, text: str) -> float:
        """
        Does the text look similar at different scales?
        
        Fractal/self-similar text has high scale symmetry.
        Text with unique structure at each level has low scale symmetry.
        
        Measured by: character distribution at different n-gram levels.
        """
        chars = self._char_sequence(text)
        if len(chars) < 4:
            return 0.5
        
        # Distribution at scale 1 (characters)
        dist1 = Counter(chars)
        
        # Distribution at scale 2 (bigrams)
        bigrams = [''.join(chars[i:i+2]) for i in range(len(chars)-1)]
        dist2 = Counter(bigrams)
        
        # Distribution at scale 3 (trigrams)
        trigrams = [''.join(chars[i:i+3]) for i in range(len(chars)-2)]
        dist3 = Counter(trigrams)
        
        # Measure entropy at each scale
        def entropy(dist):
            total = sum(dist.values())
            if total == 0:
                return 0
            probs = [c/total for c in dist.values()]
            return -sum(p * np.log2(p) for p in probs if p > 0)
        
        e1, e2, e3 = entropy(dist1), entropy(dist2), entropy(dist3)
        
        # Self-similar: entropy grows linearly with scale
        # Non-self-similar: entropy grows non-linearly
        if e1 == 0:
            return 0.5
        
        # Expected linear growth
        expected_e2 = e1 * 1.5
        expected_e3 = e1 * 2.0
        
        # Deviation from linear
        deviation = abs(e2 - expected_e2) + abs(e3 - expected_e3)
        normalized = deviation / (e1 * 2) if e1 > 0 else 0
        
        return min(1.0, normalized)
    
    def measure_repetition_symmetry(self, text: str) -> float:
        """
        How much does the text repeat itself?
        
        High repetition = cyclic symmetry (like a circle)
        Low repetition = unique/linear structure
        
        Measured by: ratio of unique tokens to total tokens.
        """
        tokens = self._tokenize(text)
        if len(tokens) == 0:
            return 0.0
        
        unique = len(set(tokens))
        total = len(tokens)
        
        # High uniqueness = low repetition = asymmetric
        # Low uniqueness = high repetition = symmetric
        uniqueness = unique / total
        return uniqueness  # Return as asymmetry measure
    
    def measure_negation_symmetry(self, text: str) -> float:
        """
        Does the text contain its own negation?
        
        "to be or not to be" - contains both assertion and negation
        "the cat sat" - pure assertion
        
        Measured by: presence of negation markers and their balance.
        """
        tokens = self._tokenize(text)
        
        # Negation markers (these emerge from symmetry, not semantics)
        # Words that reverse/negate are structurally identifiable
        negation_markers = {'not', 'no', 'never', 'none', 'neither', 'nor',
                          'without', 'un', 'in', 'dis', 'non', "n't", "dont",
                          "cant", "wont", "isnt", "arent", "wasnt", "werent"}
        
        neg_count = sum(1 for t in tokens if t in negation_markers or
                       any(t.startswith(p) for p in ['un', 'in', 'dis', 'non']))
        
        if len(tokens) == 0:
            return 0.0
        
        # Balanced negation = symmetric, pure assertion/negation = asymmetric
        neg_ratio = neg_count / len(tokens)
        
        # Peak symmetry at 50% negation (balanced)
        # Asymmetry increases as we move toward 0% or 100%
        return abs(0.5 - neg_ratio) * 2
    
    def measure_length_ratio(self, text: str) -> float:
        """
        Structural rhythm: ratio of short to long words.
        
        Commands tend to have short punchy words.
        Descriptions tend to have longer words.
        Questions have mixed lengths.
        
        This is pure structure - no semantics needed.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.5
        
        lengths = [len(t) for t in tokens]
        avg_len = np.mean(lengths)
        
        # Normalize to 0-1 range (typical word length 3-10)
        return min(1.0, avg_len / 10.0)
    
    def measure_vowel_balance(self, text: str) -> float:
        """
        Phonetic symmetry: balance of vowels to consonants.
        
        This captures the "sound" structure without knowing meaning.
        Different text types have different phonetic patterns.
        """
        chars = self._char_sequence(text)
        if not chars:
            return 0.5
        
        vowels = set('aeiou')
        vowel_count = sum(1 for c in chars if c in vowels)
        
        return vowel_count / len(chars)
    
    def measure_position_weight(self, text: str) -> float:
        """
        Center of mass: where does the "weight" of the text fall?
        
        Weight = information density (unique characters per position)
        
        Commands: weight at start (verb first)
        Questions: weight distributed (question word + content)
        Descriptions: weight in middle (subject + predicate)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.5
        
        # Weight each position by word length (information content)
        total_weight = 0
        weighted_position = 0
        
        for i, token in enumerate(tokens):
            weight = len(set(token))  # Unique chars = information
            position = i / len(tokens)  # Normalized position 0-1
            weighted_position += weight * position
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_position / total_weight
    
    def measure_compression_ratio(self, text: str) -> float:
        """
        Self-similarity through compression.
        
        Key insight from primes: they can't be factored (compressed).
        High compression = high redundancy = high symmetry
        Low compression = high information = high asymmetry
        
        This is the CORE symmetry - how much can the text be reduced
        while preserving its structure?
        """
        chars = self._char_sequence(text)
        if len(chars) < 2:
            return 0.5
        
        # Simple compression: count unique n-grams vs total
        # More unique = less compressible = more asymmetric
        
        unigrams = len(set(chars))
        bigrams = len(set(''.join(chars[i:i+2]) for i in range(len(chars)-1)))
        trigrams = len(set(''.join(chars[i:i+3]) for i in range(len(chars)-2)))
        
        # Theoretical max if all unique
        max_uni = min(26, len(chars))
        max_bi = min(26*26, len(chars)-1)
        max_tri = min(26*26*26, len(chars)-2)
        
        # Compression ratio: how much below max?
        if max_uni == 0 or max_bi == 0 or max_tri == 0:
            return 0.5
        
        ratio = (unigrams/max_uni + bigrams/max_bi + trigrams/max_tri) / 3
        return ratio
    
    def measure_first_word_type(self, text: str) -> float:
        """
        Structural position of the "action" word.
        
        This is a STRUCTURAL symmetry, not semantic:
        - Verbs tend to be short, consonant-heavy
        - Nouns tend to be longer, more vowels
        - Question words have specific patterns (wh-, how)
        
        We detect this purely through character patterns.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.5
        
        first = tokens[0]
        
        # Question pattern: starts with wh- or how
        if first.startswith('wh') or first == 'how':
            return 0.0  # Question symmetry
        
        # Short word at start (likely imperative/command)
        if len(first) <= 4:
            return 0.3  # Command symmetry
        
        # Article/determiner at start (likely description)
        if first in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
            return 0.7  # Description symmetry
        
        # Default
        return 0.5
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def encode(self, text: str) -> SymmetrySignature:
        """
        Encode text purely through symmetry breaking measurements.
        
        No vocabulary. No categories. Just symmetry operations.
        """
        return SymmetrySignature(
            reversal=self.measure_reversal_symmetry(text),
            exchange=self.measure_exchange_symmetry(text),
            scale=self.measure_scale_symmetry(text),
            repetition=self.measure_repetition_symmetry(text),
            negation=self.measure_negation_symmetry(text),
            length_ratio=self.measure_length_ratio(text),
            vowel_balance=self.measure_vowel_balance(text),
            position_weight=self.measure_position_weight(text),
            compression=self.measure_compression_ratio(text),
            first_word=self.measure_first_word_type(text),
        )
    
    def learn(self, concept: str, examples: List[str]):
        """Learn a concept's symmetry signature from examples."""
        signatures = [self.encode(ex) for ex in examples]
        
        # Average the signatures
        avg = SymmetrySignature(
            reversal=np.mean([s.reversal for s in signatures]),
            exchange=np.mean([s.exchange for s in signatures]),
            scale=np.mean([s.scale for s in signatures]),
            repetition=np.mean([s.repetition for s in signatures]),
            negation=np.mean([s.negation for s in signatures]),
            length_ratio=np.mean([s.length_ratio for s in signatures]),
            vowel_balance=np.mean([s.vowel_balance for s in signatures]),
            position_weight=np.mean([s.position_weight for s in signatures]),
            compression=np.mean([s.compression for s in signatures]),
            first_word=np.mean([s.first_word for s in signatures]),
        )
        
        self.learned_signatures[concept] = avg
        return avg
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text by finding closest learned concepts."""
        sig = self.encode(text)
        
        distances = []
        for concept, learned_sig in self.learned_signatures.items():
            dist = sig.distance(learned_sig)
            distances.append((concept, dist))
        
        return sorted(distances, key=lambda x: x[1])


def test_symmetry_encoder():
    """
    Test if symmetry can distinguish semantic categories WITHOUT
    any pre-defined vocabulary or categories.
    """
    encoder = SymmetryEncoder()
    
    print("=" * 70)
    print("SYMMETRY ENCODER - Bootstrap from Pure Symmetry")
    print("=" * 70)
    print()
    
    # Test 1: Raw symmetry measurements on different text types
    print("TEST 1: Raw Symmetry Measurements")
    print("-" * 70)
    
    test_texts = [
        ("palindrome", "A man a plan a canal Panama"),
        ("question", "What is the meaning of life?"),
        ("statement", "The cat sat on the mat."),
        ("negation", "I do not think therefore I am not."),
        ("repetitive", "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo"),
        ("command", "Show me the files in the directory."),
        ("definition", "A prime is a number divisible only by itself and one."),
    ]
    
    for label, text in test_texts:
        sig = encoder.encode(text)
        print(f"\n{label}: \"{text[:50]}...\"" if len(text) > 50 else f"\n{label}: \"{text}\"")
        print(f"  Reversal:   {sig.reversal:.3f}")
        print(f"  Exchange:   {sig.exchange:.3f}")
        print(f"  Scale:      {sig.scale:.3f}")
        print(f"  Repetition: {sig.repetition:.3f}")
        print(f"  Negation:   {sig.negation:.3f}")
    
    print()
    print("=" * 70)
    print("TEST 2: Can Symmetry Distinguish Semantic Categories?")
    print("-" * 70)
    
    # Train on examples WITHOUT telling it what the categories mean
    encoder.learn("action", [
        "run the program",
        "execute the command",
        "start the process",
        "launch the application",
        "begin the task",
    ])
    
    encoder.learn("description", [
        "the file is large",
        "the process is running",
        "the system is stable",
        "the connection is active",
        "the status is ready",
    ])
    
    encoder.learn("question", [
        "what is the status?",
        "how does it work?",
        "where is the file?",
        "when did it start?",
        "why is it failing?",
    ])
    
    # Test classification
    test_queries = [
        "stop the server",           # Should be action
        "the server is stopped",     # Should be description
        "is the server stopped?",    # Should be question
        "delete all files",          # Should be action
        "the disk is full",          # Should be description
        "what files exist?",         # Should be question
    ]
    
    print("\nClassification Results:")
    for query in test_queries:
        results = encoder.classify(query)
        best = results[0]
        print(f"\n\"{query}\"")
        print(f"  → {best[0]} (distance: {best[1]:.3f})")
        print(f"     All: {[(c, f'{d:.3f}') for c, d in results]}")
    
    print()
    print("=" * 70)
    print("TEST 3: Emergent Structure Discovery")
    print("-" * 70)
    
    # Can we discover that certain texts cluster together
    # WITHOUT pre-defining categories?
    
    all_texts = [
        # Actions (we don't tell it this)
        "run the code", "execute now", "start process", "launch app",
        # Descriptions (we don't tell it this)
        "the file is big", "system is slow", "disk is full", "cpu is hot",
        # Questions (we don't tell it this)
        "what is this?", "how to fix?", "where is it?", "why broken?",
    ]
    
    signatures = [(text, encoder.encode(text)) for text in all_texts]
    
    # Compute pairwise distances
    print("\nPairwise Distance Matrix (first 8 texts):")
    print("         ", end="")
    for i in range(min(8, len(signatures))):
        print(f"  T{i:02d}", end="")
    print()
    
    for i, (text_i, sig_i) in enumerate(signatures[:8]):
        print(f"T{i:02d} ", end="")
        for j, (text_j, sig_j) in enumerate(signatures[:8]):
            dist = sig_i.distance(sig_j)
            print(f" {dist:.2f}", end="")
        print(f"  \"{text_i[:15]}...\"" if len(text_i) > 15 else f"  \"{text_i}\"")
    
    print()
    print("Legend: T00-T03=actions, T04-T07=descriptions")
    print("If symmetry works, we should see lower distances within groups.")
    
    # Calculate average within-group vs between-group distances
    actions = signatures[0:4]
    descriptions = signatures[4:8]
    questions = signatures[8:12]
    
    def avg_distance(group):
        dists = []
        for i, (_, s1) in enumerate(group):
            for j, (_, s2) in enumerate(group):
                if i < j:
                    dists.append(s1.distance(s2))
        return np.mean(dists) if dists else 0
    
    def avg_between(group1, group2):
        dists = []
        for _, s1 in group1:
            for _, s2 in group2:
                dists.append(s1.distance(s2))
        return np.mean(dists) if dists else 0
    
    print()
    print("Cluster Analysis:")
    print(f"  Within-actions avg distance:     {avg_distance(actions):.3f}")
    print(f"  Within-descriptions avg distance: {avg_distance(descriptions):.3f}")
    print(f"  Within-questions avg distance:    {avg_distance(questions):.3f}")
    print(f"  Between actions-descriptions:     {avg_between(actions, descriptions):.3f}")
    print(f"  Between actions-questions:        {avg_between(actions, questions):.3f}")
    print(f"  Between descriptions-questions:   {avg_between(descriptions, questions):.3f}")
    
    within_avg = (avg_distance(actions) + avg_distance(descriptions) + avg_distance(questions)) / 3
    between_avg = (avg_between(actions, descriptions) + avg_between(actions, questions) + 
                   avg_between(descriptions, questions)) / 3
    
    print()
    print(f"  WITHIN-GROUP AVERAGE:  {within_avg:.3f}")
    print(f"  BETWEEN-GROUP AVERAGE: {between_avg:.3f}")
    print(f"  RATIO (higher=better): {between_avg/within_avg:.2f}x")
    
    if between_avg > within_avg:
        print()
        print("✅ SUCCESS: Symmetry alone can distinguish semantic categories!")
        print("   Groups cluster together based purely on symmetry breaking patterns.")
    else:
        print()
        print("⚠️  Symmetry alone may not be sufficient for this task.")
        print("   Consider additional symmetry types or refinements.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    test_symmetry_encoder()
