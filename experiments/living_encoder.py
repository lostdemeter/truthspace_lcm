#!/usr/bin/env python3
"""
Living Encoder

No named concepts. No fixed dimensions. No hardcoded structure.

Words start at random positions and self-organize through:
1. CO-OCCURRENCE → ATTRACTION (words that appear together converge)
2. NON-CO-OCCURRENCE → REPULSION (words in different contexts separate)
3. ERROR FEEDBACK → REFINEMENT (mistakes adjust positions)

The "concepts" emerge as attractor basins - regions where related words cluster.
We don't name them. We don't assign dimensions. The geometry discovers itself.

This is closer to how biological neural networks work:
- No neuron is "the death neuron"
- Concepts are distributed patterns of activation
- Structure emerges from Hebbian learning ("neurons that fire together wire together")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
import math

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Fact:
    text: str
    position: np.ndarray
    id: str


class LivingEncoder:
    """
    An encoder that discovers its own structure through use.
    
    No predefined concepts. No fixed dimensions.
    Words find their positions through attractor dynamics.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # Word positions - start random, evolve through use
        self.positions: Dict[str, np.ndarray] = {}
        
        # Co-occurrence tracking
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.total_words = 0
        
        # Stored facts
        self.facts: List[Fact] = []
        
        # Dynamics parameters
        self.attraction_rate = 0.1
        self.repulsion_rate = 0.01
        self.repulsion_threshold = 0.5
        self.error_learning_rate = 0.15
        
        # Statistics
        self.dynamics_steps = 0
        self.error_adjustments = 0
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_position(self, word: str) -> np.ndarray:
        """Get or create position for a word."""
        if word not in self.positions:
            # Initialize with deterministic random based on word
            np.random.seed(hash(word) % (2**32))
            self.positions[word] = np.random.randn(self.dim) * 0.3
            np.random.seed(None)
        return self.positions[word]
    
    def _update_cooccurrence(self, words: List[str], window: int = 10):
        """Track which words appear together."""
        for i, word in enumerate(words):
            self.word_counts[word] += 1
            self.total_words += 1
            
            # Co-occurrence within window
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if i != j:
                    self.cooccurrence[word][words[j]] += 1
    
    def _run_dynamics(self, iterations: int = 1):
        """
        Run attractor/repeller dynamics.
        
        Words that co-occur ATTRACT.
        Words that don't co-occur REPEL (if too close).
        """
        all_words = list(self.positions.keys())
        if len(all_words) < 2:
            return
        
        for _ in range(iterations):
            forces = {w: np.zeros(self.dim) for w in all_words}
            
            for word in all_words:
                pos1 = self.positions[word]
                cooccur = self.cooccurrence.get(word, {})
                
                for other in all_words:
                    if other == word:
                        continue
                    
                    pos2 = self.positions[other]
                    diff = pos2 - pos1
                    dist = np.linalg.norm(diff) + 1e-8
                    direction = diff / dist
                    
                    # Attraction: proportional to co-occurrence
                    cooccur_count = cooccur.get(other, 0)
                    if cooccur_count > 0:
                        # Log scale to prevent explosion
                        strength = self.attraction_rate * math.log1p(cooccur_count)
                        forces[word] += strength * direction
                    
                    # Repulsion: only if close AND not co-occurring
                    elif dist < self.repulsion_threshold:
                        strength = self.repulsion_rate / (dist ** 2)
                        forces[word] -= strength * direction
            
            # Apply forces
            for word in all_words:
                self.positions[word] += forces[word]
                
                # Soft normalization to prevent explosion
                norm = np.linalg.norm(self.positions[word])
                if norm > 3.0:
                    self.positions[word] *= 3.0 / norm
            
            self.dynamics_steps += 1
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text as weighted sum of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        # Weight by inverse frequency (rare words matter more)
        position = np.zeros(self.dim)
        total_weight = 0.0
        
        for word in words:
            pos = self._get_position(word)
            
            # IDF-like weighting
            freq = self.word_counts.get(word, 1) / max(self.total_words, 1)
            weight = -math.log(freq + 1e-8)
            weight = max(1.0, min(weight, 15.0))  # Clamp
            
            position += weight * pos
            total_weight += weight
        
        if total_weight > 0:
            position /= total_weight
        
        return position
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = np.linalg.norm(a - b)
        return 1.0 / (1.0 + dist)
    
    def ingest(self, text: str):
        """
        Ingest text to learn word relationships.
        
        This updates co-occurrence and runs dynamics.
        """
        words = self._tokenize(text)
        
        # Ensure all words have positions
        for word in words:
            self._get_position(word)
        
        # Update co-occurrence
        self._update_cooccurrence(words)
        
        # Run dynamics to let positions evolve
        self._run_dynamics(iterations=1)
    
    def store(self, text: str, fact_id: str) -> Fact:
        """Store a fact after ingesting its text."""
        self.ingest(text)
        position = self._encode(text)
        fact = Fact(text=text, position=position, id=fact_id)
        self.facts.append(fact)
        return fact
    
    def query(self, text: str) -> Tuple[Optional[Fact], float]:
        """Find closest fact to query."""
        if not self.facts:
            return None, 0.0
        
        # Ingest query to update dynamics
        self.ingest(text)
        query_pos = self._encode(text)
        
        # Recompute fact positions (they may have shifted)
        for fact in self.facts:
            fact.position = self._encode(fact.text)
        
        best_fact, best_sim = None, -float('inf')
        for fact in self.facts:
            sim = self._similarity(query_pos, fact.position)
            if sim > best_sim:
                best_sim = sim
                best_fact = fact
        
        return best_fact, best_sim
    
    def learn_from_error(self, query_text: str, correct_fact_id: str) -> bool:
        """
        Learn from a query-answer pair.
        
        If wrong, adjust word positions to fix the error.
        """
        matched, _ = self.query(query_text)
        if matched is None:
            return False
        
        correct = next((f for f in self.facts if f.id == correct_fact_id), None)
        if correct is None:
            return False
        
        if matched.id == correct_fact_id:
            return True
        
        # Error! Adjust positions
        query_words = set(self._tokenize(query_text))
        correct_words = set(self._tokenize(correct.text))
        incorrect_words = set(self._tokenize(matched.text))
        
        attract = correct_words - incorrect_words
        repel = incorrect_words - correct_words
        
        for qw in query_words:
            q_pos = self._get_position(qw)
            
            for aw in attract:
                a_pos = self._get_position(aw)
                self.positions[qw] = q_pos + self.error_learning_rate * (a_pos - q_pos)
                q_pos = self.positions[qw]
            
            for rw in repel:
                r_pos = self._get_position(rw)
                diff = q_pos - r_pos
                dist = np.linalg.norm(diff) + 0.1
                self.positions[qw] = q_pos + self.error_learning_rate * diff / dist
        
        self.error_adjustments += 1
        return False
    
    def train(self, qa_pairs: List[Tuple[str, str]], epochs: int = 10) -> Dict:
        """Train on query-answer pairs."""
        stats = {'accuracy_history': []}
        
        for _ in range(epochs):
            correct = sum(1 for q, fid in qa_pairs if self.learn_from_error(q, fid))
            acc = correct / len(qa_pairs) if qa_pairs else 0
            stats['accuracy_history'].append(acc)
            if acc == 1.0:
                break
        
        stats['final_accuracy'] = stats['accuracy_history'][-1] if stats['accuracy_history'] else 0
        stats['error_adjustments'] = self.error_adjustments
        stats['dynamics_steps'] = self.dynamics_steps
        return stats
    
    def find_clusters(self, n_clusters: int = 10) -> Dict[int, List[str]]:
        """
        Find emergent clusters in the word positions.
        
        Uses simple k-means-like clustering to identify
        the "concepts" that emerged from dynamics.
        """
        if len(self.positions) < n_clusters:
            return {}
        
        words = list(self.positions.keys())
        positions = np.array([self.positions[w] for w in words])
        
        # Simple k-means
        np.random.seed(42)
        centroids = positions[np.random.choice(len(positions), n_clusters, replace=False)]
        
        for _ in range(20):
            # Assign to nearest centroid
            assignments = []
            for pos in positions:
                dists = [np.linalg.norm(pos - c) for c in centroids]
                assignments.append(np.argmin(dists))
            
            # Update centroids
            for i in range(n_clusters):
                members = [positions[j] for j in range(len(positions)) if assignments[j] == i]
                if members:
                    centroids[i] = np.mean(members, axis=0)
        
        # Build clusters
        clusters = defaultdict(list)
        for i, word in enumerate(words):
            clusters[assignments[i]].append(word)
        
        return dict(clusters)


def main():
    print("=" * 70)
    print("LIVING ENCODER")
    print("No named concepts. Structure emerges from use.")
    print("=" * 70)
    
    enc = LivingEncoder(dim=32)
    
    # Store facts - the encoder learns structure as it goes
    print("\nStoring facts (structure emerges as we go)...")
    
    facts = [
        # History
        ("Washington born 1732 Virginia", "gw_birth"),
        ("Washington president United States", "gw_president"),
        ("Washington died 1799", "gw_death"),
        ("Washington commanded army war", "gw_war"),
        ("Lincoln born 1809", "lincoln_birth"),
        ("Lincoln president 1861", "lincoln_president"),
        ("Lincoln assassinated died", "lincoln_death"),
        ("Lincoln Civil War", "lincoln_war"),
        
        # Cooking
        ("boil pasta water", "pasta_boil"),
        ("bake bread oven", "bread_bake"),
        ("fry chicken oil", "chicken_fry"),
        ("grill steak heat", "steak_grill"),
        
        # Tech
        ("list files directory", "ls"),
        ("disk space storage", "df"),
        ("process running system", "ps"),
        ("search text file", "grep"),
        
        # Geography
        ("Paris capital France", "paris"),
        ("London capital England", "london"),
        ("Tokyo capital Japan", "tokyo"),
        ("Nile river Africa", "nile"),
    ]
    
    for text, fid in facts:
        enc.store(text, fid)
    
    print(f"  Stored {len(facts)} facts")
    print(f"  Vocabulary size: {len(enc.positions)} words")
    print(f"  Dynamics steps: {enc.dynamics_steps}")
    
    # Find emergent clusters
    print("\n" + "=" * 70)
    print("EMERGENT CLUSTERS (discovered, not named)")
    print("=" * 70)
    
    clusters = enc.find_clusters(n_clusters=8)
    for i, words in sorted(clusters.items()):
        # Show top 8 words per cluster
        sample = words[:8]
        print(f"  Cluster {i}: {', '.join(sample)}")
    
    # Test queries
    print("\n" + "=" * 70)
    print("TESTING QUERIES")
    print("=" * 70)
    
    queries = [
        ("when was Washington born", "gw_birth"),
        ("Washington president", "gw_president"),
        ("Washington died", "gw_death"),
        ("Washington war army", "gw_war"),
        ("Lincoln born", "lincoln_birth"),
        ("Lincoln president", "lincoln_president"),
        ("Lincoln died assassinated", "lincoln_death"),
        ("Lincoln Civil War", "lincoln_war"),
        ("cook pasta boil", "pasta_boil"),
        ("bake bread", "bread_bake"),
        ("fry chicken", "chicken_fry"),
        ("grill steak", "steak_grill"),
        ("list files", "ls"),
        ("disk space", "df"),
        ("running processes", "ps"),
        ("search text", "grep"),
        ("capital France", "paris"),
        ("capital England", "london"),
        ("capital Japan", "tokyo"),
        ("river Africa", "nile"),
    ]
    
    correct = 0
    for query, expected in queries:
        matched, sim = enc.query(query)
        is_correct = matched.id == expected
        correct += is_correct
        marker = '✓' if is_correct else '✗'
        print(f"  {marker} \"{query}\" -> {matched.id}")
    
    print(f"\nAccuracy: {correct}/{len(queries)} = {correct/len(queries):.1%}")
    
    # Train on failures
    failures = [(q, e) for q, e in queries if enc.query(q)[0].id != e]
    
    if failures:
        print(f"\n" + "=" * 70)
        print(f"LEARNING FROM ERRORS ({len(failures)} failures)")
        print("=" * 70)
        
        stats = enc.train(failures, epochs=10)
        print(f"Accuracy history: {[f'{a:.0%}' for a in stats['accuracy_history']]}")
        print(f"Error adjustments: {stats['error_adjustments']}")
        print(f"Total dynamics steps: {enc.dynamics_steps}")
        
        # Retest
        correct = sum(1 for q, e in queries if enc.query(q)[0].id == e)
        print(f"\nFinal accuracy: {correct}/{len(queries)} = {correct/len(queries):.1%}")
    
    # Show final clusters
    print("\n" + "=" * 70)
    print("FINAL EMERGENT CLUSTERS")
    print("=" * 70)
    
    clusters = enc.find_clusters(n_clusters=8)
    for i, words in sorted(clusters.items()):
        sample = words[:10]
        print(f"  Cluster {i}: {', '.join(sample)}")


if __name__ == "__main__":
    main()
