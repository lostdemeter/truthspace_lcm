#!/usr/bin/env python3
"""
Convergence Test

Hypothesis: As data grows, geometry-only matching converges to cluster-based matching.

Test:
1. Start with small dataset, measure geometry-only accuracy
2. Add more data incrementally
3. Watch geometry-only accuracy climb toward 100%

The clusters bootstrap the positions.
The geometry should become self-sufficient with enough data.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re

PHI = (1 + np.sqrt(5)) / 2

# Semantic clusters for position initialization
SEMANTIC_CLUSTERS = {
    'DEATH': ['die', 'died', 'dies', 'dying', 'death', 'dead', 'killed', 
              'assassinated', 'assassination', 'passed', 'perished'],
    'BIRTH': ['born', 'birth', 'birthday', 'birthplace'],
    'WHEN': ['when', 'year', 'date', 'time'],
    'DISCOVER': ['discover', 'discovered', 'discovery', 'invented', 'developed', 'theory'],
    'LEAD': ['president', 'leader', 'led', 'ruled', 'first', '16th', 'king'],
    'CAPITAL': ['capital', 'city'],
    'COOK': ['cook', 'cooking', 'recipe', 'prepare'],
    'BOIL': ['boil', 'boiling', 'water', 'pasta'],
    'BAKE': ['bake', 'baking', 'oven', 'bread', 'dough', 'degrees'],
    'FRY': ['fry', 'frying', 'oil', 'chicken', 'crispy', 'golden'],
    'GRILL': ['grill', 'grilling', 'steak', 'medium', 'rare'],
    'ROAST': ['roast', 'roasting', 'vegetables'],
    'LIST_CMD': ['ls', 'list', 'files', 'directory'],
    'SEARCH_CMD': ['grep', 'search', 'find', 'text', 'pattern'],
    'SHOW_CMD': ['cat', 'display', 'show', 'view', 'contents'],
    'DISK_CMD': ['df', 'disk', 'space', 'storage', 'usage'],
    'PROCESS_CMD': ['ps', 'process', 'processes', 'running'],
    'WASHINGTON': ['washington', 'george', 'mount', 'vernon', 'first'],
    'LINCOLN': ['lincoln', 'abraham', 'civil', 'emancipation', '16th'],
    'EINSTEIN': ['einstein', 'albert', 'relativity'],
    'NEWTON': ['newton', 'isaac', 'gravity', 'motion', 'calculus'],
    'DARWIN': ['darwin', 'charles', 'evolution', 'species'],
    'CURIE': ['curie', 'marie', 'radioactivity', 'radium'],
    'FRANCE': ['france', 'paris', 'french'],
    'ENGLAND': ['england', 'london', 'english', 'british'],
    'JAPAN': ['japan', 'tokyo', 'japanese'],
    'GERMANY': ['germany', 'berlin', 'german'],
    'RUSSIA': ['russia', 'moscow', 'russian'],
    'CHINA': ['china', 'beijing', 'chinese'],
}

WORD_TO_CLUSTER = {}
for cluster_name, words in SEMANTIC_CLUSTERS.items():
    for word in words:
        WORD_TO_CLUSTER[word.lower()] = cluster_name


class ConvergenceEncoder:
    """Encoder for testing convergence hypothesis."""
    
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.word_positions: Dict[str, np.ndarray] = {}
        self.cluster_positions: Dict[str, np.ndarray] = {}
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[str, int] = defaultdict(int)
        
        self._init_cluster_positions()
    
    def _init_cluster_positions(self):
        for cluster_name in SEMANTIC_CLUSTERS.keys():
            np.random.seed(hash(cluster_name) % (2**32))
            pos = np.random.randn(self.dim)
            pos = pos / np.linalg.norm(pos) * PHI
            self.cluster_positions[cluster_name] = pos
            np.random.seed(None)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_position(self, word: str) -> np.ndarray:
        if word in self.word_positions:
            return self.word_positions[word]
        
        if word in WORD_TO_CLUSTER:
            cluster = WORD_TO_CLUSTER[word]
            np.random.seed(hash(word) % (2**32))
            offset = np.random.randn(self.dim) * 0.1
            np.random.seed(None)
            pos = self.cluster_positions[cluster] + offset
        else:
            np.random.seed(hash(word) % (2**32))
            pos = np.random.randn(self.dim) * 0.3
            np.random.seed(None)
        
        self.word_positions[word] = pos
        return pos
    
    def _encode(self, text: str) -> np.ndarray:
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        for word in words:
            position += self._get_position(word)
        
        return position / len(words)
    
    def _update_cooccurrence(self, text: str, window: int = 5):
        words = self._tokenize(text)
        for i, word in enumerate(words):
            self.word_counts[word] += 1
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if i != j:
                    self.cooccurrence[word][words[j]] += 1
    
    def _run_dynamics(self, iterations: int = 1):
        """Run attractor/repeller dynamics based on co-occurrence."""
        words = list(self.word_positions.keys())
        if len(words) < 2:
            return
        
        for _ in range(iterations):
            for word in words:
                pos = self.word_positions[word]
                cooccur = self.cooccurrence.get(word, {})
                total_cooccur = sum(cooccur.values()) + 1
                
                force = np.zeros(self.dim)
                
                for other in words:
                    if other == word:
                        continue
                    
                    other_pos = self.word_positions[other]
                    diff = other_pos - pos
                    dist = np.linalg.norm(diff) + 1e-8
                    direction = diff / dist
                    
                    cooccur_count = cooccur.get(other, 0)
                    
                    if cooccur_count > 0:
                        # ATTRACTION: words that co-occur pull together
                        strength = 0.005 * np.log1p(cooccur_count)
                        force += strength * direction
                    elif dist < 1.0:
                        # REPULSION: words that don't co-occur push apart (if close)
                        strength = 0.002 / (dist ** 2)
                        force -= strength * direction
                
                # Apply force with damping to prevent oscillation
                self.word_positions[word] = pos + 0.5 * force
    
    def ingest(self, text: str):
        """Ingest text, update co-occurrence, run dynamics."""
        words = self._tokenize(text)
        for word in words:
            self._get_position(word)
        self._update_cooccurrence(text)
        self._run_dynamics(iterations=1)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def query_geometric(self, query: str, facts: List[Tuple[str, str]]) -> Tuple[str, float]:
        """Query using ONLY geometric similarity."""
        query_pos = self._encode(query)
        
        best_id = None
        best_sim = -float('inf')
        
        for fact_text, fact_id in facts:
            fact_pos = self._encode(fact_text)
            sim = self.cosine_similarity(query_pos, fact_pos)
            if sim > best_sim:
                best_sim = sim
                best_id = fact_id
        
        return best_id, best_sim


def main():
    print("=" * 70)
    print("CONVERGENCE TEST")
    print("Does geometry-only accuracy converge to 100% with more data?")
    print("=" * 70)
    
    # Core test cases (fixed)
    test_cases = [
        ("When was Washington born", "gw_birth"),
        ("Who was the first president", "gw_president"),
        ("When did Washington die", "gw_death"),
        ("When was Lincoln born", "lincoln_birth"),
        ("Lincoln president", "lincoln_president"),
        ("How did Lincoln die", "lincoln_death"),
        ("What did Einstein discover", "einstein_relativity"),
        ("What did Newton discover", "newton_gravity"),
        ("Darwin evolution", "darwin_evolution"),
        ("Curie radioactivity", "curie_radioactivity"),
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
        ("How to search text in files", "grep"),
        ("How to display file contents", "cat"),
        ("How to check disk space", "df"),
        ("How to see running processes", "ps"),
    ]
    
    # Data batches - increasing amounts of data
    data_batches = [
        # Batch 1: Minimal (26 facts)
        [
            ("George Washington was born in 1732 in Virginia", "gw_birth"),
            ("George Washington was the first president of the United States", "gw_president"),
            ("George Washington died in 1799 at Mount Vernon", "gw_death"),
            ("Abraham Lincoln was born in 1809 in Kentucky", "lincoln_birth"),
            ("Abraham Lincoln was the 16th president", "lincoln_president"),
            ("Abraham Lincoln was assassinated in 1865", "lincoln_death"),
            ("Albert Einstein developed the theory of relativity", "einstein_relativity"),
            ("Isaac Newton discovered the laws of gravity and motion", "newton_gravity"),
            ("Charles Darwin developed the theory of evolution", "darwin_evolution"),
            ("Marie Curie discovered radioactivity and radium", "curie_radioactivity"),
            ("Paris is the capital of France", "paris"),
            ("London is the capital of England", "london"),
            ("Tokyo is the capital of Japan", "tokyo"),
            ("Berlin is the capital of Germany", "berlin"),
            ("Moscow is the capital of Russia", "moscow"),
            ("Beijing is the capital of China", "beijing"),
            ("To boil pasta bring water to a boil and cook for 8 minutes", "pasta"),
            ("To bake bread mix dough and bake at 450 degrees", "bread"),
            ("To fry chicken coat in flour and fry in oil until golden", "chicken"),
            ("To grill steak season and grill 4 minutes per side for medium rare", "steak"),
            ("To roast vegetables toss with oil and roast at 425 degrees", "vegetables"),
            ("The ls command lists files and directories", "ls"),
            ("The grep command searches for text patterns in files", "grep"),
            ("The cat command displays file contents", "cat"),
            ("The df command shows disk space usage", "df"),
            ("The ps command shows running processes", "ps"),
        ],
        # Batch 2: More history context
        [
            ("Washington was born on February 22 1732", "gw_birth"),
            ("The birth of Washington occurred in Virginia colony", "gw_birth"),
            ("Lincoln was born on February 12 1809", "lincoln_birth"),
            ("The birth of Lincoln happened in a log cabin", "lincoln_birth"),
            ("Washington died on December 14 1799", "gw_death"),
            ("The death of Washington was from a throat infection", "gw_death"),
            ("Lincoln died from assassination by John Wilkes Booth", "lincoln_death"),
            ("The death of Lincoln shocked the nation", "lincoln_death"),
        ],
        # Batch 3: More science context
        [
            ("Einstein discovered that energy equals mass times speed of light squared", "einstein_relativity"),
            ("The theory of relativity was developed by Einstein", "einstein_relativity"),
            ("Newton discovered that objects fall due to gravity", "newton_gravity"),
            ("The laws of motion were discovered by Newton", "newton_gravity"),
            ("Darwin discovered natural selection drives evolution", "darwin_evolution"),
            ("The theory of evolution explains species change over time", "darwin_evolution"),
            ("Curie discovered that certain elements emit radiation", "curie_radioactivity"),
            ("Radioactivity was first studied by Marie Curie", "curie_radioactivity"),
        ],
        # Batch 4: More geography context
        [
            ("France has Paris as its capital city", "paris"),
            ("The capital city of France is Paris on the Seine", "paris"),
            ("England has London as its capital city", "london"),
            ("The capital city of England is London on the Thames", "london"),
            ("Japan has Tokyo as its capital city", "tokyo"),
            ("The capital city of Japan is Tokyo", "tokyo"),
            ("Germany has Berlin as its capital city", "berlin"),
            ("Russia has Moscow as its capital city", "moscow"),
            ("China has Beijing as its capital city", "beijing"),
        ],
        # Batch 5: More cooking context
        [
            ("To cook pasta you need to boil water first", "pasta"),
            ("Pasta is cooked by boiling in salted water", "pasta"),
            ("To cook bread you need to bake it in an oven", "bread"),
            ("Bread is baked at high temperature", "bread"),
            ("To cook chicken you can fry it in oil", "chicken"),
            ("Fried chicken is cooked until golden and crispy", "chicken"),
            ("To cook steak you can grill it over high heat", "steak"),
            ("Grilled steak should be cooked to desired doneness", "steak"),
            ("To cook vegetables you can roast them in the oven", "vegetables"),
            ("Roasted vegetables are cooked until caramelized", "vegetables"),
        ],
        # Batch 6: More Linux context
        [
            ("To list files use the ls command", "ls"),
            ("The ls command shows directory contents", "ls"),
            ("To search text use the grep command", "grep"),
            ("The grep command finds patterns in files", "grep"),
            ("To display file contents use the cat command", "cat"),
            ("The cat command shows what is in a file", "cat"),
            ("To check disk space use the df command", "df"),
            ("The df command shows storage usage", "df"),
            ("To see running processes use the ps command", "ps"),
            ("The ps command shows active tasks", "ps"),
        ],
    ]
    
    enc = ConvergenceEncoder(dim=32)
    all_facts = []
    
    print("\n" + "=" * 70)
    print("CONVERGENCE RESULTS")
    print("=" * 70)
    print(f"{'Batch':<8} {'Facts':<8} {'Geometry':<12} {'Dynamics':<10}")
    print("-" * 40)
    
    for batch_num, batch in enumerate(data_batches):
        # Ingest batch
        for text, fid in batch:
            enc.ingest(text)
            # Only add to facts list if not already there (avoid duplicates)
            if not any(f[1] == fid for f in all_facts):
                all_facts.append((text, fid))
            else:
                # Update existing fact text (more context)
                pass
        
        # Run extra dynamics to let positions settle
        for _ in range(10):
            enc._run_dynamics(iterations=5)
        
        # Test geometry-only accuracy
        correct = 0
        for query, expected in test_cases:
            result_id, sim = enc.query_geometric(query, all_facts)
            if result_id == expected:
                correct += 1
        
        accuracy = correct / len(test_cases)
        print(f"{batch_num + 1:<8} {len(all_facts):<8} {accuracy:.1%}{'':>6}")
    
    # Final detailed results
    print("\n" + "=" * 70)
    print("FINAL DETAILED RESULTS (Geometry-Only)")
    print("=" * 70)
    
    correct = 0
    failures = []
    for query, expected in test_cases:
        result_id, sim = enc.query_geometric(query, all_facts)
        is_correct = result_id == expected
        correct += is_correct
        if not is_correct:
            failures.append((query, expected, result_id))
    
    print(f"\nFinal accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")
    
    if failures:
        print(f"\nRemaining failures ({len(failures)}):")
        for query, expected, got in failures:
            print(f"  ✗ \"{query}\" → {got} (expected {expected})")
    else:
        print("\n✓ 100% CONVERGENCE ACHIEVED!")
    
    # Show word distances to verify geometry
    print("\n" + "=" * 70)
    print("WORD GEOMETRY (verifying semantic distances)")
    print("=" * 70)
    
    word_pairs = [
        ('born', 'birth'),
        ('born', 'died'),
        ('die', 'assassinated'),
        ('die', 'born'),
        ('paris', 'france'),
        ('paris', 'japan'),
        ('boil', 'cook'),
        ('boil', 'president'),
    ]
    
    print("\nWord distances (lower = more similar):")
    for w1, w2 in word_pairs:
        if w1 in enc.word_positions and w2 in enc.word_positions:
            dist = np.linalg.norm(enc.word_positions[w1] - enc.word_positions[w2])
            print(f"  {w1:12} ↔ {w2:12}: {dist:.3f}")


if __name__ == "__main__":
    main()
