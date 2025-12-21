#!/usr/bin/env python3
"""
Tree Encoder - Living/Dead Architecture

Inspired by how trees actually work:
- CAMBIUM (living layer): Thin outer layer where growth happens
- HEARTWOOD (dead core): Frozen structure that provides stability

The encoder has two modes:
- LIVING: Dynamics run, positions shift, learning happens
- DEAD: Positions frozen, fast lookup, stable

Crystallization: Periodically freeze the living layer into the dead core.
Like growth rings - each crystallization is a snapshot of structure.

This solves the tension between:
- Emergence (living): Structure discovers itself
- Efficiency (dead): Fast, predictable inference
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re
import math
import json
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class GrowthRing:
    """A crystallized snapshot of the living layer."""
    timestamp: float
    vocabulary_size: int
    positions: Dict[str, List[float]]  # word -> position (as list for JSON)
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'vocabulary_size': self.vocabulary_size,
            'positions': self.positions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GrowthRing':
        return cls(
            timestamp=data['timestamp'],
            vocabulary_size=data['vocabulary_size'],
            positions=data['positions'],
        )


@dataclass
class Fact:
    text: str
    position: np.ndarray
    id: str


class TreeEncoder:
    """
    An encoder with living (cambium) and dead (heartwood) layers.
    
    Living layer: Where new words are added and dynamics run
    Dead layer: Frozen positions for fast, stable lookup
    
    Words start in the living layer. When crystallized, they move
    to the dead layer and stop changing.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # DEAD LAYER (heartwood) - frozen positions
        self.dead_positions: Dict[str, np.ndarray] = {}
        
        # LIVING LAYER (cambium) - evolving positions
        self.living_positions: Dict[str, np.ndarray] = {}
        
        # Co-occurrence (only tracked for living layer)
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.total_words = 0
        
        # Facts
        self.facts: List[Fact] = []
        
        # Growth rings (history of crystallizations)
        self.growth_rings: List[GrowthRing] = []
        
        # Mode
        self.is_living = True  # Start in living mode
        
        # Dynamics parameters
        self.attraction_rate = 0.1
        self.repulsion_rate = 0.01
        self.repulsion_threshold = 0.5
        self.error_learning_rate = 0.12
        
        # Statistics
        self.dynamics_steps = 0
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_position(self, word: str) -> np.ndarray:
        """
        Get position for a word.
        
        Priority:
        1. Dead layer (frozen, stable)
        2. Living layer (evolving)
        3. Create new in living layer
        """
        # Check dead layer first (frozen knowledge)
        if word in self.dead_positions:
            return self.dead_positions[word]
        
        # Check living layer
        if word in self.living_positions:
            return self.living_positions[word]
        
        # Create new position in living layer
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.3
        np.random.seed(None)
        
        if self.is_living:
            self.living_positions[word] = pos
        else:
            # In dead mode, new words go to dead layer immediately
            self.dead_positions[word] = pos
        
        return pos
    
    def _update_cooccurrence(self, words: List[str], window: int = 10):
        """Track co-occurrence (only in living mode)."""
        if not self.is_living:
            return
        
        for i, word in enumerate(words):
            self.word_counts[word] += 1
            self.total_words += 1
            
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if i != j:
                    self.cooccurrence[word][words[j]] += 1
    
    def _run_dynamics(self, iterations: int = 1):
        """
        Run attractor/repeller dynamics on LIVING layer only.
        Dead layer is frozen.
        """
        if not self.is_living:
            return
        
        living_words = list(self.living_positions.keys())
        if len(living_words) < 2:
            return
        
        for _ in range(iterations):
            forces = {w: np.zeros(self.dim) for w in living_words}
            
            for word in living_words:
                pos1 = self.living_positions[word]
                cooccur = self.cooccurrence.get(word, {})
                
                # Interact with other living words
                for other in living_words:
                    if other == word:
                        continue
                    
                    pos2 = self.living_positions[other]
                    diff = pos2 - pos1
                    dist = np.linalg.norm(diff) + 1e-8
                    direction = diff / dist
                    
                    cooccur_count = cooccur.get(other, 0)
                    if cooccur_count > 0:
                        strength = self.attraction_rate * math.log1p(cooccur_count)
                        forces[word] += strength * direction
                    elif dist < self.repulsion_threshold:
                        strength = self.repulsion_rate / (dist ** 2)
                        forces[word] -= strength * direction
                
                # Also attract toward related dead words (but don't move dead words)
                for dead_word, dead_pos in self.dead_positions.items():
                    cooccur_count = cooccur.get(dead_word, 0)
                    if cooccur_count > 0:
                        diff = dead_pos - pos1
                        dist = np.linalg.norm(diff) + 1e-8
                        direction = diff / dist
                        strength = self.attraction_rate * 0.5 * math.log1p(cooccur_count)
                        forces[word] += strength * direction
            
            # Apply forces to living layer only
            for word in living_words:
                self.living_positions[word] += forces[word]
                norm = np.linalg.norm(self.living_positions[word])
                if norm > 3.0:
                    self.living_positions[word] *= 3.0 / norm
            
            self.dynamics_steps += 1
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text using both dead and living positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        total_weight = 0.0
        
        for word in words:
            pos = self._get_position(word)
            
            # IDF-like weighting
            freq = self.word_counts.get(word, 1) / max(self.total_words, 1)
            weight = -math.log(freq + 1e-8)
            weight = max(1.0, min(weight, 15.0))
            
            position += weight * pos
            total_weight += weight
        
        if total_weight > 0:
            position /= total_weight
        
        return position
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = np.linalg.norm(a - b)
        return 1.0 / (1.0 + dist)
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def grow(self):
        """Switch to living mode (cambium active)."""
        self.is_living = True
    
    def freeze(self):
        """Switch to dead mode (heartwood only)."""
        self.is_living = False
    
    def crystallize(self):
        """
        Crystallize the living layer into the dead layer.
        
        Like a tree forming a growth ring:
        - Living positions become dead (frozen)
        - Living layer is cleared
        - A growth ring is recorded
        """
        import time
        
        # Record growth ring
        ring = GrowthRing(
            timestamp=time.time(),
            vocabulary_size=len(self.living_positions),
            positions={w: pos.tolist() for w, pos in self.living_positions.items()}
        )
        self.growth_rings.append(ring)
        
        # Move living to dead
        for word, pos in self.living_positions.items():
            self.dead_positions[word] = pos.copy()
        
        # Clear living layer
        self.living_positions.clear()
        
        # Clear co-occurrence (start fresh)
        self.cooccurrence.clear()
        
        return ring
    
    def ingest(self, text: str):
        """Ingest text (only learns in living mode)."""
        words = self._tokenize(text)
        
        for word in words:
            self._get_position(word)
        
        self._update_cooccurrence(words)
        
        if self.is_living:
            self._run_dynamics(iterations=1)
    
    def store(self, text: str, fact_id: str) -> Fact:
        """Store a fact."""
        self.ingest(text)
        position = self._encode(text)
        fact = Fact(text=text, position=position, id=fact_id)
        self.facts.append(fact)
        return fact
    
    def query(self, text: str) -> Tuple[Optional[Fact], float]:
        """Query for closest fact."""
        if not self.facts:
            return None, 0.0
        
        # Ingest query (learns if in living mode)
        self.ingest(text)
        query_pos = self._encode(text)
        
        # Recompute fact positions
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
        """Learn from error (only in living mode)."""
        matched, _ = self.query(query_text)
        if matched is None:
            return False
        
        correct = next((f for f in self.facts if f.id == correct_fact_id), None)
        if correct is None:
            return False
        
        if matched.id == correct_fact_id:
            return True
        
        if not self.is_living:
            return False  # Can't learn in dead mode
        
        # Adjust living positions
        query_words = set(self._tokenize(query_text))
        correct_words = set(self._tokenize(correct.text))
        incorrect_words = set(self._tokenize(matched.text))
        
        attract = correct_words - incorrect_words
        repel = incorrect_words - correct_words
        
        for qw in query_words:
            if qw not in self.living_positions:
                continue  # Can't adjust dead words
            
            q_pos = self.living_positions[qw]
            
            for aw in attract:
                a_pos = self._get_position(aw)
                self.living_positions[qw] = q_pos + self.error_learning_rate * (a_pos - q_pos)
                q_pos = self.living_positions[qw]
            
            for rw in repel:
                r_pos = self._get_position(rw)
                diff = q_pos - r_pos
                dist = np.linalg.norm(diff) + 0.1
                self.living_positions[qw] = q_pos + self.error_learning_rate * diff / dist
        
        return False
    
    def train(self, qa_pairs: List[Tuple[str, str]], epochs: int = 10) -> Dict:
        """Train on QA pairs (must be in living mode)."""
        if not self.is_living:
            return {'error': 'Cannot train in dead mode'}
        
        stats = {'accuracy_history': []}
        
        for _ in range(epochs):
            correct = sum(1 for q, fid in qa_pairs if self.learn_from_error(q, fid))
            acc = correct / len(qa_pairs) if qa_pairs else 0
            stats['accuracy_history'].append(acc)
            if acc == 1.0:
                break
        
        stats['final_accuracy'] = stats['accuracy_history'][-1] if stats['accuracy_history'] else 0
        return stats
    
    def save(self, path: str):
        """Save the tree (dead layer + growth rings)."""
        data = {
            'dim': self.dim,
            'dead_positions': {w: pos.tolist() for w, pos in self.dead_positions.items()},
            'growth_rings': [ring.to_dict() for ring in self.growth_rings],
            'facts': [{'text': f.text, 'id': f.id} for f in self.facts],
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    def load(self, path: str):
        """Load a saved tree."""
        data = json.loads(Path(path).read_text())
        self.dim = data['dim']
        self.dead_positions = {w: np.array(pos) for w, pos in data['dead_positions'].items()}
        self.growth_rings = [GrowthRing.from_dict(r) for r in data['growth_rings']]
        # Reload facts
        for f_data in data['facts']:
            pos = self._encode(f_data['text'])
            self.facts.append(Fact(text=f_data['text'], position=pos, id=f_data['id']))
    
    def stats(self) -> Dict:
        """Get encoder statistics."""
        return {
            'mode': 'living' if self.is_living else 'dead',
            'dead_vocabulary': len(self.dead_positions),
            'living_vocabulary': len(self.living_positions),
            'total_vocabulary': len(self.dead_positions) + len(self.living_positions),
            'growth_rings': len(self.growth_rings),
            'facts': len(self.facts),
            'dynamics_steps': self.dynamics_steps,
        }


def main():
    print("=" * 70)
    print("TREE ENCODER")
    print("Living (cambium) + Dead (heartwood) Architecture")
    print("=" * 70)
    
    enc = TreeEncoder(dim=32)
    
    # Phase 1: LIVING - Learn initial structure
    print("\n" + "=" * 70)
    print("PHASE 1: LIVING MODE (learning)")
    print("=" * 70)
    
    enc.grow()  # Ensure living mode
    
    facts = [
        ("Washington born 1732 Virginia", "gw_birth"),
        ("Washington president United States", "gw_president"),
        ("Washington died 1799", "gw_death"),
        ("Lincoln born 1809", "lincoln_birth"),
        ("Lincoln president 1861", "lincoln_president"),
        ("Lincoln died assassinated", "lincoln_death"),
        ("boil pasta water", "pasta"),
        ("bake bread oven", "bread"),
        ("Paris capital France", "paris"),
        ("Tokyo capital Japan", "tokyo"),
    ]
    
    for text, fid in facts:
        enc.store(text, fid)
    
    print(f"Stats: {enc.stats()}")
    
    # Test in living mode
    queries = [
        ("Washington born", "gw_birth"),
        ("Washington president", "gw_president"),
        ("Washington died", "gw_death"),
        ("Lincoln born", "lincoln_birth"),
        ("cook pasta", "pasta"),
        ("capital France", "paris"),
    ]
    
    print("\nTesting in LIVING mode:")
    correct = sum(1 for q, e in queries if enc.query(q)[0].id == e)
    print(f"  Accuracy: {correct}/{len(queries)} = {correct/len(queries):.0%}")
    
    # Train on failures
    failures = [(q, e) for q, e in queries if enc.query(q)[0].id != e]
    if failures:
        print(f"  Training on {len(failures)} failures...")
        enc.train(failures, epochs=5)
        correct = sum(1 for q, e in queries if enc.query(q)[0].id == e)
        print(f"  After training: {correct}/{len(queries)} = {correct/len(queries):.0%}")
    
    # Phase 2: CRYSTALLIZE - Freeze the learned structure
    print("\n" + "=" * 70)
    print("PHASE 2: CRYSTALLIZATION")
    print("=" * 70)
    
    ring = enc.crystallize()
    print(f"  Created growth ring with {ring.vocabulary_size} words")
    print(f"  Stats: {enc.stats()}")
    
    # Phase 3: DEAD MODE - Fast inference
    print("\n" + "=" * 70)
    print("PHASE 3: DEAD MODE (frozen)")
    print("=" * 70)
    
    enc.freeze()
    
    print("\nTesting in DEAD mode (no learning):")
    correct = sum(1 for q, e in queries if enc.query(q)[0].id == e)
    print(f"  Accuracy: {correct}/{len(queries)} = {correct/len(queries):.0%}")
    print(f"  Stats: {enc.stats()}")
    
    # Phase 4: NEW GROWTH - Add more knowledge
    print("\n" + "=" * 70)
    print("PHASE 4: NEW GROWTH (living again)")
    print("=" * 70)
    
    enc.grow()  # Back to living mode
    
    new_facts = [
        ("London capital England", "london"),
        ("Berlin capital Germany", "berlin"),
        ("fry chicken oil", "chicken"),
        ("grill steak heat", "steak"),
    ]
    
    for text, fid in new_facts:
        enc.store(text, fid)
    
    print(f"  Added {len(new_facts)} new facts")
    print(f"  Stats: {enc.stats()}")
    
    # Test new knowledge
    new_queries = [
        ("capital England", "london"),
        ("capital Germany", "berlin"),
        ("fry chicken", "chicken"),
    ]
    
    print("\nTesting new knowledge:")
    correct = sum(1 for q, e in new_queries if enc.query(q)[0].id == e)
    print(f"  Accuracy: {correct}/{len(new_queries)} = {correct/len(new_queries):.0%}")
    
    # Final crystallization
    print("\n" + "=" * 70)
    print("FINAL: Second crystallization")
    print("=" * 70)
    
    ring2 = enc.crystallize()
    print(f"  Created growth ring with {ring2.vocabulary_size} words")
    print(f"  Total growth rings: {len(enc.growth_rings)}")
    print(f"  Final stats: {enc.stats()}")


if __name__ == "__main__":
    main()
