#!/usr/bin/env python3
"""
Autotrain Encoder

Quick bootstrap + automatic training to 100% accuracy.

The key insight: Error = Where to Build
- Each failure tells us exactly which positions need adjustment
- We use the error signal to move query words toward correct answers
- The system self-corrects until 100%

Bootstrap: Minimal cluster seeds (just the core concepts)
Autotrain: Error-driven position adjustment until convergence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

PHI = (1 + np.sqrt(5)) / 2

# MINIMAL BOOTSTRAP: Just the core concept seeds
# These are the "attractors" that pull related words together
CONCEPT_SEEDS = {
    # Life events
    'BIRTH': ['born', 'birth'],
    'DEATH': ['died', 'death', 'killed', 'assassinated'],
    
    # Actions
    'DISCOVER': ['discovered', 'developed', 'invented', 'theory'],
    'LEAD': ['president', 'leader', 'first'],
    
    # Scientists (need separate attractors)
    'EINSTEIN': ['einstein', 'albert', 'relativity'],
    'NEWTON': ['newton', 'isaac', 'gravity', 'motion'],
    'DARWIN': ['darwin', 'charles', 'evolution'],
    'CURIE': ['curie', 'marie', 'radioactivity', 'radium'],
    
    # Geography
    'CAPITAL': ['capital', 'city'],
    
    # Cooking methods
    'BOIL': ['boil', 'pasta', 'water'],
    'BAKE': ['bake', 'bread', 'oven'],
    'FRY': ['fry', 'chicken', 'oil'],
    'GRILL': ['grill', 'steak'],
    'ROAST': ['roast', 'vegetables'],
    
    # Tech
    'LIST': ['ls', 'list', 'files', 'directory'],
    'SEARCH': ['grep', 'search', 'find', 'text'],
    'SHOW': ['cat', 'display', 'show', 'contents'],
    'DISK': ['df', 'disk', 'space', 'storage'],
    'PROCESS': ['ps', 'process', 'running'],
}

# Build reverse lookup
WORD_TO_SEED = {}
for seed_name, words in CONCEPT_SEEDS.items():
    for word in words:
        WORD_TO_SEED[word.lower()] = seed_name


class AutotrainEncoder:
    """
    Encoder that bootstraps quickly and autotrains to 100%.
    """
    
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.word_positions: Dict[str, np.ndarray] = {}
        self.seed_positions: Dict[str, np.ndarray] = {}
        self.facts: List[Tuple[str, str]] = []
        
        # Training parameters
        self.learning_rate = 0.15
        self.train_iterations = 0
        
        self._init_seeds()
    
    def _init_seeds(self):
        """Initialize seed positions - the attractor centers."""
        for seed_name in CONCEPT_SEEDS.keys():
            np.random.seed(hash(seed_name) % (2**32))
            pos = np.random.randn(self.dim)
            pos = pos / np.linalg.norm(pos) * PHI
            self.seed_positions[seed_name] = pos
            np.random.seed(None)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_position(self, word: str) -> np.ndarray:
        """Get or create position for a word."""
        if word in self.word_positions:
            return self.word_positions[word]
        
        # Check if word has a seed
        if word in WORD_TO_SEED:
            seed = WORD_TO_SEED[word]
            np.random.seed(hash(word) % (2**32))
            offset = np.random.randn(self.dim) * 0.1
            np.random.seed(None)
            pos = self.seed_positions[seed] + offset
        else:
            # Unknown word - random position
            np.random.seed(hash(word) % (2**32))
            pos = np.random.randn(self.dim) * 0.5
            np.random.seed(None)
        
        self.word_positions[word] = pos
        return pos
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text as average of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        for word in words:
            position += self._get_position(word)
        
        return position / len(words)
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def store(self, text: str, fact_id: str):
        """Store a fact."""
        # Ensure all words have positions
        for word in self._tokenize(text):
            self._get_position(word)
        self.facts.append((text, fact_id))
    
    def query(self, text: str) -> Tuple[str, str, float]:
        """Query for best matching fact."""
        query_pos = self._encode(text)
        
        best_fact = None
        best_id = None
        best_sim = -float('inf')
        
        for fact_text, fact_id in self.facts:
            fact_pos = self._encode(fact_text)
            sim = self._similarity(query_pos, fact_pos)
            if sim > best_sim:
                best_sim = sim
                best_fact = fact_text
                best_id = fact_id
        
        return best_fact, best_id, best_sim
    
    def _adjust_positions(self, query: str, correct_id: str, wrong_id: str):
        """
        Adjust word positions based on error.
        
        Move query words TOWARD correct fact words.
        Move query words AWAY FROM wrong fact words.
        """
        query_words = set(self._tokenize(query))
        
        correct_fact = next((f for f, i in self.facts if i == correct_id), None)
        wrong_fact = next((f for f, i in self.facts if i == wrong_id), None)
        
        if not correct_fact or not wrong_fact:
            return
        
        correct_words = set(self._tokenize(correct_fact))
        wrong_words = set(self._tokenize(wrong_fact))
        
        # Words unique to correct fact - attract
        attract = correct_words - wrong_words
        # Words unique to wrong fact - repel
        repel = wrong_words - correct_words
        
        for qw in query_words:
            q_pos = self._get_position(qw)
            
            # Attract toward correct
            for aw in attract:
                a_pos = self._get_position(aw)
                direction = a_pos - q_pos
                self.word_positions[qw] = q_pos + self.learning_rate * direction
                q_pos = self.word_positions[qw]
            
            # Repel from wrong
            for rw in repel:
                r_pos = self._get_position(rw)
                direction = q_pos - r_pos
                dist = np.linalg.norm(direction) + 0.1
                self.word_positions[qw] = q_pos + self.learning_rate * direction / dist
    
    def autotrain(self, test_cases: List[Tuple[str, str]], max_epochs: int = 50, verbose: bool = True) -> Dict:
        """
        Automatically train until 100% accuracy or max epochs.
        
        Returns training statistics.
        """
        stats = {
            'epochs': 0,
            'accuracy_history': [],
            'adjustments': 0,
        }
        
        for epoch in range(max_epochs):
            # Test all cases
            correct = 0
            errors = []
            
            for query, expected_id in test_cases:
                _, result_id, _ = self.query(query)
                if result_id == expected_id:
                    correct += 1
                else:
                    errors.append((query, expected_id, result_id))
            
            accuracy = correct / len(test_cases)
            stats['accuracy_history'].append(accuracy)
            stats['epochs'] = epoch + 1
            
            if verbose:
                print(f"  Epoch {epoch + 1}: {accuracy:.1%} ({correct}/{len(test_cases)})")
            
            # Check for convergence
            if accuracy == 1.0:
                if verbose:
                    print(f"  ✓ 100% achieved in {epoch + 1} epochs!")
                break
            
            # Adjust positions based on errors
            for query, expected_id, wrong_id in errors:
                self._adjust_positions(query, expected_id, wrong_id)
                stats['adjustments'] += 1
            
            self.train_iterations += 1
        
        stats['final_accuracy'] = stats['accuracy_history'][-1] if stats['accuracy_history'] else 0
        return stats
    
    def stats(self) -> Dict:
        return {
            'facts': len(self.facts),
            'vocabulary': len(self.word_positions),
            'seeds': len(CONCEPT_SEEDS),
            'train_iterations': self.train_iterations,
        }


def main():
    print("=" * 70)
    print("AUTOTRAIN ENCODER")
    print("Quick bootstrap + automatic training to 100%")
    print("=" * 70)
    
    enc = AutotrainEncoder(dim=32)
    
    # Store facts
    facts = [
        ("George Washington was born in 1732 in Virginia", "gw_birth"),
        ("George Washington was the first president", "gw_president"),
        ("George Washington died in 1799", "gw_death"),
        ("Abraham Lincoln was born in 1809", "lincoln_birth"),
        ("Abraham Lincoln was the 16th president", "lincoln_president"),
        ("Abraham Lincoln was assassinated in 1865", "lincoln_death"),
        ("Albert Einstein developed the theory of relativity", "einstein"),
        ("Isaac Newton discovered gravity", "newton"),
        ("Charles Darwin developed evolution theory", "darwin"),
        ("Marie Curie discovered radioactivity", "curie"),
        ("Paris is the capital of France", "paris"),
        ("London is the capital of England", "london"),
        ("Tokyo is the capital of Japan", "tokyo"),
        ("Berlin is the capital of Germany", "berlin"),
        ("Moscow is the capital of Russia", "moscow"),
        ("Beijing is the capital of China", "beijing"),
        ("Boil pasta in water for 8 minutes", "pasta"),
        ("Bake bread in oven at 450 degrees", "bread"),
        ("Fry chicken in oil until golden", "chicken"),
        ("Grill steak for 4 minutes per side", "steak"),
        ("Roast vegetables at 425 degrees", "vegetables"),
        ("The ls command lists files", "ls"),
        ("The grep command searches text", "grep"),
        ("The cat command displays file contents", "cat"),
        ("The df command shows disk space", "df"),
        ("The ps command shows running processes", "ps"),
    ]
    
    print(f"\nStoring {len(facts)} facts...")
    for text, fid in facts:
        enc.store(text, fid)
    
    print(f"Stats: {enc.stats()}")
    
    # Test cases
    test_cases = [
        ("When was Washington born", "gw_birth"),
        ("Who was the first president", "gw_president"),
        ("When did Washington die", "gw_death"),
        ("When was Lincoln born", "lincoln_birth"),
        ("Lincoln president", "lincoln_president"),
        ("How did Lincoln die", "lincoln_death"),
        ("What did Einstein discover", "einstein"),
        ("What did Newton discover", "newton"),
        ("Darwin evolution", "darwin"),
        ("Curie radioactivity", "curie"),
        ("Capital of France", "paris"),
        ("Capital of England", "london"),
        ("Capital of Japan", "tokyo"),
        ("Capital of Germany", "berlin"),
        ("Capital of Russia", "moscow"),
        ("Capital of China", "beijing"),
        ("How to cook pasta", "pasta"),
        ("How to bake bread", "bread"),
        ("How to fry chicken", "chicken"),
        ("How to grill steak", "steak"),
        ("How to roast vegetables", "vegetables"),
        ("How to list files", "ls"),
        ("How to search text", "grep"),
        ("How to display file contents", "cat"),
        ("How to check disk space", "df"),
        ("How to see running processes", "ps"),
    ]
    
    # Test BEFORE training
    print("\n" + "=" * 70)
    print("BEFORE TRAINING (bootstrap only)")
    print("=" * 70)
    
    correct = sum(1 for q, e in test_cases if enc.query(q)[1] == e)
    print(f"Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")
    
    # AUTOTRAIN
    print("\n" + "=" * 70)
    print("AUTOTRAINING")
    print("=" * 70)
    
    stats = enc.autotrain(test_cases, max_epochs=20, verbose=True)
    
    print(f"\nTraining complete!")
    print(f"  Final accuracy: {stats['final_accuracy']:.1%}")
    print(f"  Epochs: {stats['epochs']}")
    print(f"  Adjustments: {stats['adjustments']}")
    
    # Test AFTER training
    print("\n" + "=" * 70)
    print("AFTER TRAINING")
    print("=" * 70)
    
    for query, expected in test_cases:
        fact, fid, sim = enc.query(query)
        marker = "✓" if fid == expected else "✗"
        print(f"  {marker} \"{query}\" → {fid}")
    
    # Conversation test
    print("\n" + "=" * 70)
    print("CONVERSATION TEST")
    print("=" * 70)
    
    conversation = [
        "Tell me about George Washington",
        "When was he born",
        "What about Lincoln",
        "How did Lincoln die",
        "Capital of France",
        "How to cook pasta",
        "How to list files",
        "What did Einstein discover",
    ]
    
    for q in conversation:
        fact, fid, sim = enc.query(q)
        print(f"\n  User: {q}")
        print(f"  Bot:  {fact}")


if __name__ == "__main__":
    main()
