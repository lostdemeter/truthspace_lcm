#!/usr/bin/env python3
"""
Bootstrapped Tree Encoder

Combines:
1. PARETO BOOTSTRAP: Initial structure from Zipf + semantic clusters
2. TREE ARCHITECTURE: Living/dead layers with crystallization
3. AUTOBALANCING: Error-driven refinement

The bootstrap provides the "seed" - like how a tree grows from a seed
that already contains genetic information about structure.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
import math

PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# BOOTSTRAP DATA: The genetic code of the tree
# =============================================================================

ZIPF_RANKS = {
    # Structural (low info)
    'the': 1, 'be': 2, 'to': 3, 'of': 4, 'and': 5, 'a': 6, 'in': 7,
    'is': 8, 'it': 9, 'for': 10, 'was': 15, 'on': 20, 'are': 25,
    'that': 12, 'have': 14, 'this': 18, 'from': 22, 'by': 28,
    'he': 30, 'she': 32, 'his': 35, 'her': 38, 'they': 40,
    'an': 45, 'as': 50, 'at': 55, 'with': 60,
    
    # Question words
    'when': 70, 'where': 75, 'who': 80, 'how': 85, 'why': 90,
    'what': 60, 'which': 95, 'did': 100,
    
    # Time words
    'year': 200, 'years': 210, 'day': 220, 'days': 230,
    'born': 1300, 'died': 1350, 'death': 750, 'birth': 1400,
    
    # Common verbs
    'served': 1500, 'was': 15, 'is': 8, 'are': 25, 'were': 30,
    'discovered': 1600, 'invented': 1700, 'developed': 1200,
    'wrote': 1100, 'founded': 1550, 'won': 900,
    
    # Geography
    'capital': 800, 'city': 700, 'country': 320, 'river': 1600,
    'population': 1400, 'largest': 1200, 'longest': 1300,
    
    # Cooking
    'cook': 1700, 'boil': 1900, 'bake': 1800, 'fry': 2000,
    'grill': 2100, 'roast': 2200, 'heat': 1000,
    'minutes': 500, 'degrees': 1000, 'temperature': 1100,
    'pasta': 2100, 'bread': 1700, 'chicken': 1900, 'steak': 2300,
    
    # Tech
    'command': 2000, 'file': 1500, 'files': 1550, 'directory': 2500,
    'list': 600, 'show': 500, 'display': 700, 'search': 800,
    'disk': 2200, 'space': 500, 'process': 1300, 'running': 1400,
    
    # Names (high info)
    'washington': 5000, 'george': 6000, 'lincoln': 5500, 'abraham': 6500,
    'jefferson': 5800, 'thomas': 4000, 'franklin': 5200, 'benjamin': 6000,
    'einstein': 6000, 'newton': 5800, 'darwin': 6200, 'curie': 6500,
    'paris': 4500, 'london': 4200, 'tokyo': 5000, 'berlin': 4800,
    'moscow': 5200, 'beijing': 5500,
    'france': 4000, 'england': 3800, 'japan': 4300, 'germany': 4100,
    'russia': 3900, 'china': 3500,
}

SEMANTIC_CLUSTERS = {
    # Life events
    'birth_event': ['born', 'birth', 'birthday'],
    'death_event': ['died', 'death', 'assassinated', 'killed'],
    
    # Historical figures
    'washington': ['washington', 'george', 'continental', 'revolutionary', 'vernon'],
    'lincoln': ['lincoln', 'abraham', 'civil', 'emancipation', 'gettysburg', 'booth'],
    'jefferson': ['jefferson', 'thomas', 'declaration', 'independence'],
    'franklin': ['franklin', 'benjamin', 'lightning', 'electricity'],
    
    # Scientists
    'einstein': ['einstein', 'albert', 'relativity', 'photoelectric'],
    'newton': ['newton', 'isaac', 'gravity', 'motion', 'calculus', 'principia'],
    'darwin': ['darwin', 'charles', 'evolution', 'selection', 'species', 'beagle'],
    'curie': ['curie', 'marie', 'radioactivity', 'radium', 'polonium'],
    
    # Countries and capitals
    'france': ['france', 'paris', 'french', 'eiffel', 'seine'],
    'england': ['england', 'london', 'english', 'british', 'thames', 'ben'],
    'japan': ['japan', 'tokyo', 'japanese', 'fuji'],
    'germany': ['germany', 'berlin', 'german', 'wall'],
    'russia': ['russia', 'moscow', 'russian', 'kremlin'],
    'china': ['china', 'beijing', 'chinese', 'mandarin', 'wall'],
    
    # Cooking methods
    'boiling': ['boil', 'boiling', 'water', 'pot', 'pasta'],
    'baking': ['bake', 'baking', 'oven', 'bread', 'dough', 'yeast'],
    'frying': ['fry', 'frying', 'oil', 'pan', 'chicken', 'crispy'],
    'grilling': ['grill', 'grilling', 'steak', 'heat', 'medium', 'rare'],
    'roasting': ['roast', 'roasting', 'vegetables', 'oven'],
    
    # Linux commands
    'ls_cmd': ['ls', 'list', 'files', 'directory', 'directories'],
    'grep_cmd': ['grep', 'search', 'text', 'pattern', 'find'],
    'cat_cmd': ['cat', 'display', 'contents', 'file'],
    'df_cmd': ['df', 'disk', 'space', 'storage', 'filesystem'],
    'ps_cmd': ['ps', 'process', 'processes', 'running', 'system'],
    'chmod_cmd': ['chmod', 'permissions', 'read', 'write', 'execute'],
    'mkdir_cmd': ['mkdir', 'create', 'directory', 'folder'],
    'rm_cmd': ['rm', 'remove', 'delete', 'files'],
}

DEFAULT_RANK = 8000


@dataclass
class Fact:
    text: str
    position: np.ndarray
    id: str


class BootstrappedTreeEncoder:
    """
    Tree encoder with Pareto bootstrap.
    
    The bootstrap provides initial structure (like DNA in a seed).
    The living layer refines through dynamics and error correction.
    Crystallization freezes learned structure into the dead layer.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # DEAD LAYER (heartwood) - frozen positions
        self.dead_positions: Dict[str, np.ndarray] = {}
        
        # LIVING LAYER (cambium) - evolving positions  
        self.living_positions: Dict[str, np.ndarray] = {}
        
        # SEED LAYER (genetic code) - bootstrap positions
        self.seed_positions: Dict[str, np.ndarray] = {}
        self.seed_weights: Dict[str, float] = {}
        
        # Co-occurrence
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.total_words = 0
        
        # Facts
        self.facts: List[Fact] = []
        
        # Mode
        self.is_living = True
        
        # Parameters
        self.attraction_rate = 0.05
        self.error_learning_rate = 0.1
        
        # Stats
        self.dynamics_steps = 0
        self.error_adjustments = 0
        
        # Bootstrap!
        self._bootstrap()
    
    def _bootstrap(self):
        """Initialize seed layer from Zipf + clusters."""
        # Create cluster centroids
        cluster_centroids = {}
        for i, (cluster_name, words) in enumerate(SEMANTIC_CLUSTERS.items()):
            np.random.seed(hash(cluster_name) % (2**32))
            centroid = np.random.randn(self.dim)
            centroid = centroid / np.linalg.norm(centroid) * 1.5
            cluster_centroids[cluster_name] = centroid
            np.random.seed(None)
            
            # Words in cluster near centroid
            for word in words:
                if word not in self.seed_positions:
                    np.random.seed(hash(word) % (2**32))
                    offset = np.random.randn(self.dim) * 0.15
                    np.random.seed(None)
                    self.seed_positions[word] = centroid + offset
                    self.seed_weights[word] = self._rank_to_weight(
                        ZIPF_RANKS.get(word, DEFAULT_RANK)
                    )
        
        # Other Zipf words
        for word, rank in ZIPF_RANKS.items():
            if word not in self.seed_positions:
                np.random.seed(hash(word) % (2**32))
                pos = np.random.randn(self.dim) * 0.5
                np.random.seed(None)
                self.seed_positions[word] = pos
                self.seed_weights[word] = self._rank_to_weight(rank)
    
    def _rank_to_weight(self, rank: int) -> float:
        return math.log2(rank + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_position(self, word: str) -> np.ndarray:
        """Get position with priority: dead > living > seed > new."""
        if word in self.dead_positions:
            return self.dead_positions[word]
        if word in self.living_positions:
            return self.living_positions[word]
        if word in self.seed_positions:
            # Copy seed to living layer
            self.living_positions[word] = self.seed_positions[word].copy()
            return self.living_positions[word]
        
        # New word - random position in living layer
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.3
        np.random.seed(None)
        self.living_positions[word] = pos
        return pos
    
    def _get_weight(self, word: str) -> float:
        if word in self.seed_weights:
            return self.seed_weights[word]
        return self._rank_to_weight(DEFAULT_RANK)
    
    def _update_cooccurrence(self, words: List[str], window: int = 5):
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
        """Run dynamics on living layer only."""
        if not self.is_living:
            return
        
        living_words = list(self.living_positions.keys())
        if len(living_words) < 2:
            return
        
        for _ in range(iterations):
            for word in living_words:
                pos = self.living_positions[word]
                cooccur = self.cooccurrence.get(word, {})
                
                force = np.zeros(self.dim)
                
                # Attract to co-occurring words
                for other, count in cooccur.items():
                    if other in self.living_positions:
                        other_pos = self.living_positions[other]
                    elif other in self.dead_positions:
                        other_pos = self.dead_positions[other]
                    elif other in self.seed_positions:
                        other_pos = self.seed_positions[other]
                    else:
                        continue
                    
                    diff = other_pos - pos
                    strength = self.attraction_rate * math.log1p(count) / (1 + np.linalg.norm(diff))
                    force += strength * diff
                
                self.living_positions[word] = pos + force
            
            self.dynamics_steps += 1
    
    def _encode(self, text: str) -> np.ndarray:
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        total_weight = 0.0
        
        for word in words:
            pos = self._get_position(word)
            weight = self._get_weight(word)
            position += weight * pos
            total_weight += weight
        
        if total_weight > 0:
            position /= total_weight
        
        return position
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = np.linalg.norm(a - b)
        return 1.0 / (1.0 + dist)
    
    def grow(self):
        self.is_living = True
    
    def freeze(self):
        self.is_living = False
    
    def crystallize(self):
        """Move living layer to dead layer."""
        for word, pos in self.living_positions.items():
            self.dead_positions[word] = pos.copy()
        self.living_positions.clear()
    
    def ingest(self, text: str):
        words = self._tokenize(text)
        for word in words:
            self._get_position(word)
        self._update_cooccurrence(words)
        if self.is_living:
            self._run_dynamics(iterations=1)
    
    def store(self, text: str, fact_id: str) -> Fact:
        self.ingest(text)
        position = self._encode(text)
        fact = Fact(text=text, position=position, id=fact_id)
        self.facts.append(fact)
        return fact
    
    def query(self, text: str) -> Tuple[Optional[Fact], float]:
        if not self.facts:
            return None, 0.0
        
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
        matched, _ = self.query(query_text)
        if matched is None:
            return False
        
        correct = next((f for f in self.facts if f.id == correct_fact_id), None)
        if correct is None:
            return False
        
        if matched.id == correct_fact_id:
            return True
        
        if not self.is_living:
            return False
        
        # Adjust living positions
        query_words = set(self._tokenize(query_text))
        correct_words = set(self._tokenize(correct.text))
        incorrect_words = set(self._tokenize(matched.text))
        
        attract = correct_words - incorrect_words
        repel = incorrect_words - correct_words
        
        for qw in query_words:
            if qw not in self.living_positions:
                continue
            
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
        
        self.error_adjustments += 1
        return False
    
    def train(self, qa_pairs: List[Tuple[str, str]], epochs: int = 10) -> Dict:
        stats = {'accuracy_history': []}
        
        for _ in range(epochs):
            correct = sum(1 for q, fid in qa_pairs if self.learn_from_error(q, fid))
            acc = correct / len(qa_pairs) if qa_pairs else 0
            stats['accuracy_history'].append(acc)
            if acc == 1.0:
                break
        
        stats['final_accuracy'] = stats['accuracy_history'][-1] if stats['accuracy_history'] else 0
        stats['error_adjustments'] = self.error_adjustments
        return stats
    
    def stats(self) -> Dict:
        return {
            'mode': 'living' if self.is_living else 'dead',
            'seed_vocabulary': len(self.seed_positions),
            'dead_vocabulary': len(self.dead_positions),
            'living_vocabulary': len(self.living_positions),
            'facts': len(self.facts),
            'dynamics_steps': self.dynamics_steps,
            'error_adjustments': self.error_adjustments,
        }


# =============================================================================
# STRESS TEST
# =============================================================================

CORPUS = '''
George Washington was born on February 22, 1732, in Westmoreland County, Virginia.
Washington served as the first President of the United States from 1789 to 1797.
He commanded the Continental Army during the American Revolutionary War.
Washington died on December 14, 1799, at his home in Mount Vernon, Virginia.

Abraham Lincoln was born on February 12, 1809, in Hodgenville, Kentucky.
Lincoln served as the 16th President of the United States from 1861 to 1865.
He led the nation through the Civil War and abolished slavery.
Lincoln was assassinated by John Wilkes Booth on April 14, 1865.

Thomas Jefferson was born on April 13, 1743, in Shadwell, Virginia.
Jefferson was the principal author of the Declaration of Independence in 1776.
He served as the third President of the United States from 1801 to 1809.

Benjamin Franklin was born on January 17, 1706, in Boston, Massachusetts.
Franklin discovered that lightning is electrical and invented the lightning rod.
He helped draft the Declaration of Independence and the Constitution.

Albert Einstein was born on March 14, 1879, in Ulm, Germany.
Einstein developed the theory of relativity, including E equals mc squared.
He won the Nobel Prize in Physics in 1921 for the photoelectric effect.

Isaac Newton was born on December 25, 1642, in Woolsthorpe, England.
Newton discovered the laws of motion and universal gravitation.
He invented calculus independently of Leibniz.

Charles Darwin was born on February 12, 1809, in Shrewsbury, England.
Darwin developed the theory of evolution by natural selection.
He published On the Origin of Species in 1859.

Marie Curie was born on November 7, 1867, in Warsaw, Poland.
Curie discovered radioactivity and the elements polonium and radium.
She was the first woman to win a Nobel Prize.

Paris is the capital of France and is located on the Seine River.
The Eiffel Tower in Paris was built in 1889 and is 330 meters tall.

London is the capital of England and the United Kingdom.
The River Thames flows through London.

Tokyo is the capital of Japan and the most populous city in the world.
Mount Fuji is the highest mountain in Japan at 3776 meters.

Berlin is the capital of Germany and its largest city.
The Berlin Wall divided the city from 1961 to 1989.

Moscow is the capital of Russia and its largest city.
The Kremlin is a historic fortress in Moscow.

Beijing is the capital of China and one of the oldest cities in the world.
The Great Wall of China is over 20000 kilometers long.

To boil pasta, bring a large pot of salted water to a rolling boil.
Add the pasta and cook for 8 to 12 minutes until al dente.

To bake bread, mix flour, water, yeast, and salt to form a dough.
Bake at 450 degrees Fahrenheit for 30 to 40 minutes.

To fry chicken, coat the pieces in seasoned flour or batter.
Fry the chicken for 12 to 15 minutes until golden brown.

To grill a steak, season with salt and pepper.
Grill for 4 to 5 minutes per side for medium rare.

To roast vegetables, cut them into uniform pieces.
Roast at 425 degrees Fahrenheit for 25 to 35 minutes.

The ls command lists files and directories in Linux.
Use ls -la to show hidden files and detailed information.

The grep command searches for text patterns in files.
Use grep -r to search recursively through directories.

The cat command displays the contents of a file.
The df command shows disk space usage on the filesystem.

The ps command shows running processes on the system.
The chmod command changes file permissions in Linux.

The mkdir command creates new directories.
The rm command removes files and directories.
'''


def main():
    print("=" * 70)
    print("BOOTSTRAPPED TREE ENCODER - STRESS TEST")
    print("Pareto seed + Living dynamics + Error correction")
    print("=" * 70)
    
    # Parse corpus
    facts = []
    for i, line in enumerate(CORPUS.strip().split('\n')):
        line = line.strip()
        if line:
            words = line.split()[:3]
            fid = '_'.join(w.lower() for w in words)
            fid = re.sub(r'[^a-z_]', '', fid) + f'_{i}'
            facts.append((line, fid))
    
    print(f"\nParsed {len(facts)} facts")
    
    # Create encoder
    enc = BootstrappedTreeEncoder(dim=64)
    
    # Store facts
    print("\nStoring facts...")
    for text, fid in facts:
        enc.store(text, fid)
    
    print(f"Stats: {enc.stats()}")
    
    # Create QA pairs
    qa_pairs = []
    for text, fid in facts:
        tl = text.lower()
        words = text.split()
        name = words[0] if words else ""
        
        if 'born' in tl:
            qa_pairs.append((f"when was {name} born", fid))
        if 'died' in tl or 'assassinated' in tl:
            qa_pairs.append((f"when did {name} die", fid))
        if 'capital' in tl:
            if 'france' in tl: qa_pairs.append(("capital of France", fid))
            if 'england' in tl: qa_pairs.append(("capital of England", fid))
            if 'japan' in tl: qa_pairs.append(("capital of Japan", fid))
            if 'germany' in tl: qa_pairs.append(("capital of Germany", fid))
            if 'russia' in tl: qa_pairs.append(("capital of Russia", fid))
            if 'china' in tl: qa_pairs.append(("capital of China", fid))
        if 'boil' in tl and 'pasta' in tl:
            qa_pairs.append(("how to cook pasta", fid))
        if 'bake' in tl and 'bread' in tl:
            qa_pairs.append(("how to bake bread", fid))
        if 'fry' in tl and 'chicken' in tl:
            qa_pairs.append(("how to fry chicken", fid))
        if 'grill' in tl and 'steak' in tl:
            qa_pairs.append(("how to grill steak", fid))
        if tl.startswith('the ls '):
            qa_pairs.append(("how to list files", fid))
        if tl.startswith('the grep '):
            qa_pairs.append(("how to search text", fid))
        if tl.startswith('the df '):
            qa_pairs.append(("how to check disk space", fid))
        if tl.startswith('the ps '):
            qa_pairs.append(("how to see processes", fid))
    
    print(f"Generated {len(qa_pairs)} QA pairs")
    
    # Test BEFORE training
    print("\n" + "=" * 70)
    print("BEFORE TRAINING (bootstrap only)")
    print("=" * 70)
    
    correct = sum(1 for q, e in qa_pairs if enc.query(q)[0].id == e)
    print(f"Accuracy: {correct}/{len(qa_pairs)} = {correct/len(qa_pairs):.1%}")
    
    # Sample queries
    samples = [
        "when was Washington born",
        "when did Lincoln die",
        "capital of France",
        "capital of Japan",
        "how to cook pasta",
        "how to grill steak",
        "how to list files",
        "how to check disk space",
    ]
    
    print("\nSample queries:")
    for q in samples:
        matched, _ = enc.query(q)
        answer = matched.text[:55] + "..." if len(matched.text) > 55 else matched.text
        expected = [fid for query, fid in qa_pairs if query == q]
        is_correct = matched.id in expected if expected else "?"
        marker = "✓" if is_correct == True else ("✗" if is_correct == False else "?")
        print(f"  {marker} {q}")
        print(f"     → {answer}")
    
    # Train on failures
    print("\n" + "=" * 70)
    print("TRAINING (error-driven refinement)")
    print("=" * 70)
    
    stats = enc.train(qa_pairs, epochs=15)
    print(f"Accuracy history: {[f'{a:.0%}' for a in stats['accuracy_history']]}")
    print(f"Error adjustments: {stats['error_adjustments']}")
    
    # Test AFTER training
    print("\n" + "=" * 70)
    print("AFTER TRAINING")
    print("=" * 70)
    
    correct = sum(1 for q, e in qa_pairs if enc.query(q)[0].id == e)
    print(f"Accuracy: {correct}/{len(qa_pairs)} = {correct/len(qa_pairs):.1%}")
    
    print("\nSample queries:")
    for q in samples:
        matched, _ = enc.query(q)
        answer = matched.text[:55] + "..." if len(matched.text) > 55 else matched.text
        expected = [fid for query, fid in qa_pairs if query == q]
        is_correct = matched.id in expected if expected else "?"
        marker = "✓" if is_correct == True else ("✗" if is_correct == False else "?")
        print(f"  {marker} {q}")
        print(f"     → {answer}")
    
    # Conversation test
    print("\n" + "=" * 70)
    print("CONVERSATION TEST")
    print("=" * 70)
    
    conversation = [
        "Tell me about George Washington",
        "When was Washington born",
        "What about Lincoln",
        "How did Lincoln die",
        "What is the capital of France",
        "What about Japan",
        "How do I cook pasta",
        "How do I grill a steak",
        "How do I list files in Linux",
        "What did Einstein discover",
    ]
    
    for q in conversation:
        matched, _ = enc.query(q)
        print(f"\nUser: {q}")
        print(f"Bot:  {matched.text}")
    
    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    print(f"  {enc.stats()}")


if __name__ == "__main__":
    main()
