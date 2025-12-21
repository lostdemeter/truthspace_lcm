#!/usr/bin/env python3
"""
Concept Space Encoder

Inspired by isolating languages (Chinese, Vietnamese, etc.) that don't conjugate.
The concept is the same - tense/aspect are separate modifiers.

Architecture:
- ROOT CONCEPTS: The core meaning (DIE, MOVE, CREATE, etc.)
- TEMPORAL: Past/Present/Future as orthogonal dimension
- ASPECT: Perfective/Progressive/Habitual as orthogonal dimension

"died" = DIE + PAST
"dying" = DIE + PROGRESSIVE
"will die" = DIE + FUTURE
"death" = DIE (noun, same concept)

This is more universal and should improve bootstrap accuracy.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import re

PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# LEMMATIZATION: Map conjugations to root concepts
# =============================================================================

# Map inflected forms to root concept
LEMMA_MAP = {
    # Death concept
    'die': 'DIE', 'died': 'DIE', 'dies': 'DIE', 'dying': 'DIE',
    'death': 'DIE', 'dead': 'DIE', 'deadly': 'DIE', 'deceased': 'DIE',
    'killed': 'DIE', 'kill': 'DIE', 'kills': 'DIE', 'killing': 'DIE',
    'passed': 'DIE', 'perished': 'DIE',
    
    # Birth concept
    'born': 'BIRTH', 'birth': 'BIRTH', 'births': 'BIRTH', 'bearing': 'BIRTH',
    'birthplace': 'BIRTH', 'birthday': 'BIRTH', 'natal': 'BIRTH',
    
    # Movement concept
    'go': 'MOVE', 'goes': 'MOVE', 'went': 'MOVE', 'going': 'MOVE', 'gone': 'MOVE',
    'come': 'MOVE', 'comes': 'MOVE', 'came': 'MOVE', 'coming': 'MOVE',
    'move': 'MOVE', 'moved': 'MOVE', 'moves': 'MOVE', 'moving': 'MOVE',
    'run': 'MOVE', 'ran': 'MOVE', 'runs': 'MOVE', 'running': 'MOVE',
    'walk': 'MOVE', 'walked': 'MOVE', 'walks': 'MOVE', 'walking': 'MOVE',
    
    # Creation concept
    'create': 'CREATE', 'created': 'CREATE', 'creates': 'CREATE', 'creating': 'CREATE',
    'make': 'CREATE', 'made': 'CREATE', 'makes': 'CREATE', 'making': 'CREATE',
    'build': 'CREATE', 'built': 'CREATE', 'builds': 'CREATE', 'building': 'CREATE',
    'write': 'CREATE', 'wrote': 'CREATE', 'writes': 'CREATE', 'writing': 'CREATE', 'written': 'CREATE',
    'found': 'CREATE', 'founded': 'CREATE', 'founds': 'CREATE', 'founding': 'CREATE',
    
    # Discovery concept
    'discover': 'DISCOVER', 'discovered': 'DISCOVER', 'discovers': 'DISCOVER', 'discovering': 'DISCOVER',
    'find': 'DISCOVER', 'found': 'DISCOVER', 'finds': 'DISCOVER', 'finding': 'DISCOVER',
    'invent': 'DISCOVER', 'invented': 'DISCOVER', 'invents': 'DISCOVER', 'inventing': 'DISCOVER',
    
    # Governance concept
    'govern': 'GOVERN', 'governed': 'GOVERN', 'governs': 'GOVERN', 'governing': 'GOVERN',
    'rule': 'GOVERN', 'ruled': 'GOVERN', 'rules': 'GOVERN', 'ruling': 'GOVERN',
    'lead': 'GOVERN', 'led': 'GOVERN', 'leads': 'GOVERN', 'leading': 'GOVERN',
    'elect': 'GOVERN', 'elected': 'GOVERN', 'elects': 'GOVERN', 'electing': 'GOVERN',
    'serve': 'GOVERN', 'served': 'GOVERN', 'serves': 'GOVERN', 'serving': 'GOVERN',
    
    # Combat concept
    'fight': 'COMBAT', 'fought': 'COMBAT', 'fights': 'COMBAT', 'fighting': 'COMBAT',
    'battle': 'COMBAT', 'battled': 'COMBAT', 'battles': 'COMBAT',
    'war': 'COMBAT', 'command': 'COMBAT', 'commanded': 'COMBAT', 'commands': 'COMBAT',
    
    # Cooking concepts
    'cook': 'COOK', 'cooked': 'COOK', 'cooks': 'COOK', 'cooking': 'COOK',
    'bake': 'BAKE', 'baked': 'BAKE', 'bakes': 'BAKE', 'baking': 'BAKE',
    'boil': 'BOIL', 'boiled': 'BOIL', 'boils': 'BOIL', 'boiling': 'BOIL',
    'fry': 'FRY', 'fried': 'FRY', 'fries': 'FRY', 'frying': 'FRY',
    'roast': 'ROAST', 'roasted': 'ROAST', 'roasts': 'ROAST', 'roasting': 'ROAST',
    'grill': 'GRILL', 'grilled': 'GRILL', 'grills': 'GRILL', 'grilling': 'GRILL',
    'simmer': 'SIMMER', 'simmered': 'SIMMER', 'simmers': 'SIMMER', 'simmering': 'SIMMER',
    'chop': 'CUT', 'chopped': 'CUT', 'chops': 'CUT', 'chopping': 'CUT',
    'slice': 'CUT', 'sliced': 'CUT', 'slices': 'CUT', 'slicing': 'CUT',
    'dice': 'CUT', 'diced': 'CUT', 'dices': 'CUT', 'dicing': 'CUT',
    'cut': 'CUT', 'cuts': 'CUT', 'cutting': 'CUT',
    'sauté': 'SAUTE', 'saute': 'SAUTE', 'sautéed': 'SAUTE', 'sauteed': 'SAUTE',
    
    # Tech concepts
    'list': 'LIST', 'listed': 'LIST', 'lists': 'LIST', 'listing': 'LIST',
    'show': 'SHOW', 'showed': 'SHOW', 'shows': 'SHOW', 'showing': 'SHOW', 'shown': 'SHOW',
    'display': 'SHOW', 'displayed': 'SHOW', 'displays': 'SHOW',
    'search': 'SEARCH', 'searched': 'SEARCH', 'searches': 'SEARCH', 'searching': 'SEARCH',
    'delete': 'DELETE', 'deleted': 'DELETE', 'deletes': 'DELETE', 'deleting': 'DELETE',
    'remove': 'DELETE', 'removed': 'DELETE', 'removes': 'DELETE', 'removing': 'DELETE',
}

# Temporal markers
PAST_MARKERS = {'was', 'were', 'did', 'had', 'ago', 'yesterday', 'last', 'before', 'previously'}
PRESENT_MARKERS = {'is', 'are', 'am', 'now', 'currently', 'today', 'being'}
FUTURE_MARKERS = {'will', 'shall', 'going', 'tomorrow', 'next', 'soon', 'future'}

# Aspect markers
PROGRESSIVE_MARKERS = {'ing'}  # Detected by suffix
PERFECTIVE_MARKERS = {'ed', 'done', 'finished', 'completed'}

# =============================================================================
# ROOT CONCEPTS - The universal meaning atoms
# =============================================================================

ROOT_CONCEPTS = {
    # Life events
    'DIE': {'dim': 0, 'keywords': set()},  # Populated from LEMMA_MAP
    'BIRTH': {'dim': 1, 'keywords': set()},
    
    # Actions
    'MOVE': {'dim': 2, 'keywords': set()},
    'CREATE': {'dim': 3, 'keywords': set()},
    'DISCOVER': {'dim': 4, 'keywords': set()},
    'GOVERN': {'dim': 5, 'keywords': set()},
    'COMBAT': {'dim': 6, 'keywords': set()},
    
    # Cooking
    'COOK': {'dim': 7, 'keywords': set()},
    'BAKE': {'dim': 8, 'keywords': set()},
    'BOIL': {'dim': 9, 'keywords': set()},
    'FRY': {'dim': 10, 'keywords': set()},
    'ROAST': {'dim': 11, 'keywords': set()},
    'GRILL': {'dim': 12, 'keywords': set()},
    'SIMMER': {'dim': 13, 'keywords': set()},
    'CUT': {'dim': 14, 'keywords': set()},
    'SAUTE': {'dim': 15, 'keywords': set()},
    
    # Tech
    'LIST': {'dim': 16, 'keywords': set()},
    'SHOW': {'dim': 17, 'keywords': set()},
    'SEARCH': {'dim': 18, 'keywords': set()},
    'DELETE': {'dim': 19, 'keywords': set()},
}

# Populate keywords from LEMMA_MAP
for word, concept in LEMMA_MAP.items():
    if concept in ROOT_CONCEPTS:
        ROOT_CONCEPTS[concept]['keywords'].add(word)

# =============================================================================
# ENTITY CLUSTERS - Named entities that should cluster
# =============================================================================

ENTITY_CLUSTERS = {
    # Historical figures
    'WASHINGTON': ['washington', 'george', 'mount', 'vernon', 'martha', 'continental'],
    'LINCOLN': ['lincoln', 'abraham', 'ford', 'theatre', 'emancipation', 'gettysburg'],
    'JEFFERSON': ['jefferson', 'thomas', 'declaration', 'monticello', 'louisiana'],
    
    # Scientists
    'EINSTEIN': ['einstein', 'albert', 'relativity', 'e=mc2'],
    'NEWTON': ['newton', 'isaac', 'gravity', 'apple', 'calculus'],
    'DARWIN': ['darwin', 'charles', 'evolution', 'galapagos', 'species'],
    'CURIE': ['curie', 'marie', 'radioactivity', 'radium', 'polonium'],
    
    # Countries/Capitals
    'FRANCE': ['france', 'paris', 'french', 'eiffel'],
    'ENGLAND': ['england', 'london', 'english', 'british', 'britain', 'uk'],
    'JAPAN': ['japan', 'tokyo', 'japanese'],
    'GERMANY': ['germany', 'berlin', 'german'],
    'RUSSIA': ['russia', 'moscow', 'russian', 'kremlin'],
    'CHINA': ['china', 'beijing', 'chinese'],
    
    # Rivers
    'NILE': ['nile', 'egypt', 'african'],
    'AMAZON': ['amazon', 'brazil', 'rainforest'],
}

# =============================================================================
# ZIPF WEIGHTS
# =============================================================================

ZIPF_RANKS = {
    # Structural (low info)
    'the': 1, 'be': 2, 'to': 3, 'of': 4, 'and': 5, 'a': 6, 'in': 7,
    'is': 8, 'it': 9, 'for': 10, 'was': 15, 'on': 20, 'are': 25,
    'that': 12, 'have': 14, 'this': 18, 'from': 22, 'by': 28,
    'when': 70, 'where': 75, 'who': 80, 'how': 85, 'why': 90,
    'what': 60, 'which': 95, 'did': 100,
    
    # Domain words
    'president': 600, 'capital': 800, 'country': 320, 'city': 700,
    'river': 1600, 'largest': 1200, 'longest': 1300,
    'war': 550, 'army': 1100, 'military': 1200,
    'theory': 1300, 'discovered': 1500, 'invented': 1700,
    
    # Cooking
    'recipe': 2200, 'ingredients': 2300, 'temperature': 1100,
    'minutes': 500, 'degrees': 1000, 'oven': 1950,
    'chicken': 1900, 'beef': 2000, 'pasta': 2100, 'bread': 1700,
    'onion': 2600, 'garlic': 2700, 'tomato': 2800,
    
    # Tech
    'file': 1500, 'files': 1550, 'directory': 2500,
    'process': 1300, 'disk': 2200, 'space': 500,
    'command': 2000, 'terminal': 2700,
}

DEFAULT_RANK = 8000


@dataclass
class Fact:
    text: str
    position: np.ndarray
    id: str
    domain: str = ""


class ConceptSpaceEncoder:
    """
    Encoder based on universal concepts, not language-specific conjugations.
    
    Architecture:
    - Dims 0-19: Root concepts (DIE, BIRTH, MOVE, CREATE, etc.)
    - Dims 20-22: Temporal (PAST, PRESENT, FUTURE)
    - Dims 23-24: Aspect (PERFECTIVE, PROGRESSIVE)
    - Dims 25-49: Entity clusters (WASHINGTON, LINCOLN, FRANCE, etc.)
    - Dims 50-63: Learned/overflow
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.word_positions: Dict[str, np.ndarray] = {}
        self.word_weights: Dict[str, float] = {}
        self.facts: List[Fact] = []
        self.learning_rate = 0.08
        self.adjustments = 0
        
        self._bootstrap()
    
    def _bootstrap(self):
        """Initialize concept-based positions."""
        # Root concepts get dedicated dimensions
        for concept, info in ROOT_CONCEPTS.items():
            dim_idx = info['dim']
            for word in info['keywords']:
                if word not in self.word_positions:
                    pos = np.zeros(self.dim)
                    pos[dim_idx] = PHI  # Activate concept dimension
                    self.word_positions[word] = pos
                    self.word_weights[word] = self._rank_to_weight(
                        ZIPF_RANKS.get(word, DEFAULT_RANK)
                    )
        
        # Temporal markers
        for word in PAST_MARKERS:
            pos = np.zeros(self.dim)
            pos[20] = PHI  # PAST dimension
            self.word_positions[word] = pos
            self.word_weights[word] = self._rank_to_weight(ZIPF_RANKS.get(word, 50))
        
        for word in PRESENT_MARKERS:
            pos = np.zeros(self.dim)
            pos[21] = PHI  # PRESENT dimension
            self.word_positions[word] = pos
            self.word_weights[word] = self._rank_to_weight(ZIPF_RANKS.get(word, 50))
        
        for word in FUTURE_MARKERS:
            pos = np.zeros(self.dim)
            pos[22] = PHI  # FUTURE dimension
            self.word_positions[word] = pos
            self.word_weights[word] = self._rank_to_weight(ZIPF_RANKS.get(word, 50))
        
        # Entity clusters
        for i, (entity, words) in enumerate(ENTITY_CLUSTERS.items()):
            dim_idx = 25 + i
            if dim_idx >= 50:
                break
            
            # Create cluster centroid
            np.random.seed(hash(entity) % (2**32))
            centroid = np.random.randn(self.dim) * 0.1
            centroid[dim_idx] = PHI  # Primary dimension for this entity
            np.random.seed(None)
            
            for word in words:
                if word not in self.word_positions:
                    np.random.seed(hash(word) % (2**32))
                    offset = np.random.randn(self.dim) * 0.05
                    np.random.seed(None)
                    self.word_positions[word] = centroid + offset
                    self.word_weights[word] = self._rank_to_weight(
                        ZIPF_RANKS.get(word, DEFAULT_RANK)
                    )
        
        # Other Zipf words
        for word, rank in ZIPF_RANKS.items():
            if word not in self.word_positions:
                np.random.seed(hash(word) % (2**32))
                pos = np.random.randn(self.dim) * 0.1
                np.random.seed(None)
                self.word_positions[word] = pos
                self.word_weights[word] = self._rank_to_weight(rank)
    
    def _rank_to_weight(self, rank: int) -> float:
        return math.log2(rank + 1)
    
    def _get_position(self, word: str) -> np.ndarray:
        if word not in self.word_positions:
            # Check if it's a conjugation we can lemmatize
            lemma = LEMMA_MAP.get(word)
            if lemma and lemma in ROOT_CONCEPTS:
                pos = np.zeros(self.dim)
                pos[ROOT_CONCEPTS[lemma]['dim']] = PHI
                
                # Add temporal info from suffix
                if word.endswith('ed'):
                    pos[20] += 0.5  # Past hint
                elif word.endswith('ing'):
                    pos[23] += 0.5  # Progressive hint
                
                self.word_positions[word] = pos
                self.word_weights[word] = self._rank_to_weight(DEFAULT_RANK)
            else:
                # Unknown word - random position
                np.random.seed(hash(word) % (2**32))
                pos = np.random.randn(self.dim) * 0.3
                np.random.seed(None)
                self.word_positions[word] = pos
                self.word_weights[word] = self._rank_to_weight(DEFAULT_RANK)
        
        return self.word_positions[word]
    
    def _get_weight(self, word: str) -> float:
        if word not in self.word_weights:
            self.word_weights[word] = self._rank_to_weight(DEFAULT_RANK)
        return self.word_weights[word]
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
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
            position = position / total_weight
        
        return position
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = np.linalg.norm(a - b)
        return 1.0 / (1.0 + dist)
    
    def store(self, text: str, fact_id: str, domain: str = "") -> Fact:
        position = self._encode(text)
        fact = Fact(text=text, position=position, id=fact_id, domain=domain)
        self.facts.append(fact)
        return fact
    
    def query(self, text: str) -> Tuple[Optional[Fact], float]:
        if not self.facts:
            return None, 0.0
        
        query_pos = self._encode(text)
        best_fact, best_sim = None, -float('inf')
        
        for fact in self.facts:
            sim = self._similarity(query_pos, fact.position)
            if sim > best_sim:
                best_sim = sim
                best_fact = fact
        
        return best_fact, best_sim
    
    def learn(self, query_text: str, correct_fact_id: str) -> bool:
        matched, _ = self.query(query_text)
        if matched is None:
            return False
        
        correct = next((f for f in self.facts if f.id == correct_fact_id), None)
        if correct is None:
            return False
        
        if matched.id == correct_fact_id:
            return True
        
        # Adjust on error
        query_words = set(self._tokenize(query_text))
        correct_words = set(self._tokenize(correct.text))
        incorrect_words = set(self._tokenize(matched.text))
        
        attract = correct_words - incorrect_words
        repel = incorrect_words - correct_words
        
        for qw in query_words:
            q_pos = self._get_position(qw)
            weight = self._get_weight(qw)
            lr = self.learning_rate * min(weight / 10, 1.0)
            
            for aw in attract:
                a_pos = self._get_position(aw)
                self.word_positions[qw] = q_pos + lr * (a_pos - q_pos)
                q_pos = self.word_positions[qw]
            
            for rw in repel:
                r_pos = self._get_position(rw)
                diff = q_pos - r_pos
                dist = np.linalg.norm(diff) + 0.1
                self.word_positions[qw] = q_pos + lr * diff / dist
        
        for fact in self.facts:
            fact.position = self._encode(fact.text)
        
        self.adjustments += 1
        return False
    
    def train(self, qa_pairs: List[Tuple[str, str]], epochs: int = 10) -> Dict:
        stats = {'accuracy_history': []}
        
        for _ in range(epochs):
            correct = sum(1 for q, fid in qa_pairs if self.learn(q, fid))
            acc = correct / len(qa_pairs) if qa_pairs else 0
            stats['accuracy_history'].append(acc)
            if acc == 1.0:
                break
        
        stats['final_accuracy'] = stats['accuracy_history'][-1] if stats['accuracy_history'] else 0
        stats['adjustments'] = self.adjustments
        return stats


def main():
    print("=" * 70)
    print("CONCEPT SPACE ENCODER")
    print("Root Concepts + Temporal Dimensions (Language-Agnostic)")
    print("=" * 70)
    
    enc = ConceptSpaceEncoder(dim=64)
    
    # Test the concept mapping
    print("\n" + "=" * 70)
    print("CONCEPT MAPPING TEST")
    print("=" * 70)
    
    test_words = ['die', 'died', 'dying', 'death', 'dead', 'kill', 'killed']
    print("\nDeath concept variants:")
    for word in test_words:
        pos = enc._get_position(word)
        die_dim = pos[0]  # DIE is dim 0
        past_dim = pos[20]  # PAST
        prog_dim = pos[23]  # PROGRESSIVE
        print(f"  {word:10} → DIE={die_dim:.2f}, PAST={past_dim:.2f}, PROG={prog_dim:.2f}")
    
    # Store facts
    print("\n" + "=" * 70)
    print("STORING FACTS")
    print("=" * 70)
    
    facts = [
        # History
        ("Washington born 1732 Virginia", "gw_birth", "history"),
        ("Washington president United States 1789", "gw_president", "history"),
        ("Washington died 1799 Mount Vernon", "gw_death", "history"),
        ("Washington commanded army Revolutionary War", "gw_war", "history"),
        ("Lincoln born 1809 Kentucky", "lincoln_birth", "history"),
        ("Lincoln president 1861", "lincoln_president", "history"),
        ("Lincoln assassinated 1865 Ford Theatre", "lincoln_death", "history"),
        ("Lincoln Civil War emancipation", "lincoln_war", "history"),
        ("Jefferson wrote Declaration Independence", "jefferson_declaration", "history"),
        ("Jefferson president Louisiana Purchase", "jefferson_president", "history"),
        
        # Cooking
        ("boil pasta water salt 8 minutes", "pasta_boil", "cooking"),
        ("bake bread oven 350 degrees 45 minutes", "bread_bake", "cooking"),
        ("fry chicken oil pan golden crispy", "chicken_fry", "cooking"),
        ("roast beef oven 325 degrees medium rare", "beef_roast", "cooking"),
        ("sauté onions garlic butter soft", "onion_saute", "cooking"),
        ("grill steak high heat 4 minutes", "steak_grill", "cooking"),
        ("simmer tomato sauce low heat 30 minutes", "sauce_simmer", "cooking"),
        ("chop vegetables knife cutting board", "veg_chop", "cooking"),
        
        # Tech
        ("ls list files directory", "ls", "tech"),
        ("df disk space usage", "df", "tech"),
        ("ps process list running", "ps", "tech"),
        ("grep search text pattern file", "grep", "tech"),
        ("cat display file contents", "cat", "tech"),
        ("mkdir create directory", "mkdir", "tech"),
        ("rm remove delete file", "rm", "tech"),
        ("chmod change permissions", "chmod", "tech"),
        
        # Geography
        ("Paris capital France Europe", "paris", "geography"),
        ("London capital England Britain", "london", "geography"),
        ("Tokyo capital Japan Asia", "tokyo", "geography"),
        ("Berlin capital Germany Europe", "berlin", "geography"),
        ("Moscow capital Russia", "moscow", "geography"),
        ("Beijing capital China", "beijing", "geography"),
        ("Nile longest river Africa Egypt", "nile", "geography"),
        ("Amazon largest river South America Brazil", "amazon", "geography"),
        
        # Science
        ("Einstein relativity E equals mc squared", "einstein_relativity", "science"),
        ("Newton discovered gravity laws motion", "newton_gravity", "science"),
        ("Darwin theory evolution natural selection", "darwin_evolution", "science"),
        ("Curie discovered radioactivity radium", "curie_radioactivity", "science"),
        ("water molecule H2O hydrogen oxygen", "water_molecule", "science"),
        ("photosynthesis plants sunlight energy", "photosynthesis", "science"),
        ("DNA double helix genetic code", "dna", "science"),
        ("atom nucleus electrons protons", "atom", "science"),
    ]
    
    for text, fid, domain in facts:
        enc.store(text, fid, domain)
    print(f"Stored {len(facts)} facts")
    
    # Test queries
    print("\n" + "=" * 70)
    print("TESTING QUERIES (BEFORE TRAINING)")
    print("=" * 70)
    
    queries = [
        # History - using different conjugations
        ("when was Washington born", "gw_birth"),
        ("Washington president", "gw_president"),
        ("when did Washington die", "gw_death"),  # "die" should map to DIE concept
        ("Washington military commander", "gw_war"),
        ("Lincoln birth", "lincoln_birth"),
        ("Lincoln president", "lincoln_president"),
        ("how did Lincoln die", "lincoln_death"),  # "die" again
        ("Lincoln Civil War", "lincoln_war"),
        ("who wrote Declaration Independence", "jefferson_declaration"),
        ("Jefferson president", "jefferson_president"),
        
        # Cooking
        ("how to cook pasta", "pasta_boil"),
        ("bake bread temperature", "bread_bake"),
        ("fry chicken crispy", "chicken_fry"),
        ("roast beef", "beef_roast"),
        ("sauté onions garlic", "onion_saute"),
        ("grill steak", "steak_grill"),
        ("tomato sauce recipe", "sauce_simmer"),
        ("chop vegetables", "veg_chop"),
        
        # Tech
        ("list files", "ls"),
        ("show disk space", "df"),
        ("running processes", "ps"),
        ("search text file", "grep"),
        ("display file contents", "cat"),
        ("create directory", "mkdir"),
        ("delete file", "rm"),
        ("change permissions", "chmod"),
        
        # Geography
        ("capital of France", "paris"),
        ("capital of England", "london"),
        ("capital of Japan", "tokyo"),
        ("capital of Germany", "berlin"),
        ("capital of Russia", "moscow"),
        ("capital of China", "beijing"),
        ("longest river Africa", "nile"),
        ("largest river South America", "amazon"),
        
        # Science
        ("Einstein relativity", "einstein_relativity"),
        ("who discovered gravity", "newton_gravity"),
        ("theory of evolution", "darwin_evolution"),
        ("who discovered radioactivity", "curie_radioactivity"),
        ("what is water made of", "water_molecule"),
        ("how do plants make energy", "photosynthesis"),
        ("what is DNA", "dna"),
        ("structure of atom", "atom"),
    ]
    
    correct = 0
    by_domain = {}
    
    for query, expected in queries:
        matched, sim = enc.query(query)
        is_correct = matched.id == expected
        correct += is_correct
        
        domain = matched.domain if matched else "unknown"
        if domain not in by_domain:
            by_domain[domain] = {'correct': 0, 'total': 0}
        
        # Find expected domain
        expected_domain = next((f[2] for f in facts if f[1] == expected), "unknown")
        by_domain[expected_domain]['total'] += 1
        if is_correct:
            by_domain[expected_domain]['correct'] += 1
        
        marker = '✓' if is_correct else '✗'
        print(f"  {marker} \"{query}\" -> {matched.id}")
    
    print(f"\nOverall: {correct}/{len(queries)} = {correct/len(queries):.1%}")
    
    print("\nBy domain:")
    for domain, stats in sorted(by_domain.items()):
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {domain}: {stats['correct']}/{stats['total']} = {acc:.0%}")
    
    # Train on failures
    failures = [(q, e) for q, e in queries if enc.query(q)[0].id != e]
    
    if failures:
        print(f"\n" + "=" * 70)
        print(f"AUTOBALANCING ({len(failures)} failures)")
        print("=" * 70)
        
        stats = enc.train(failures, epochs=10)
        print(f"Accuracy history: {[f'{a:.0%}' for a in stats['accuracy_history']]}")
        print(f"Adjustments: {stats['adjustments']}")
        
        # Retest
        correct = sum(1 for q, e in queries if enc.query(q)[0].id == e)
        print(f"\nAfter training: {correct}/{len(queries)} = {correct/len(queries):.1%}")


if __name__ == "__main__":
    main()
