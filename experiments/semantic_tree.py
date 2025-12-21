#!/usr/bin/env python3
"""
Semantic Tree Encoder

Combines two complementary layers:
1. SEMANTIC CLUSTERS: Vocabulary bridging (die ↔ death ↔ assassinated)
2. RECURSIVE TREE: Self-organizing structure that grows with data

The semantic layer answers: "We know where new info goes"
The tree layer answers: "We organize as we receive"

As data grows:
- Small data: Clusters provide bridges, tree is shallow
- Medium data: Tree deepens, some bridges emerge from co-occurrence
- Large data: Tree is deep, bridges are rich, clusters become redundant
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re
import math

PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# SEMANTIC CLUSTERS: The vocabulary bridge layer
# =============================================================================

SEMANTIC_CLUSTERS = {
    # Life events
    'DEATH': ['die', 'died', 'dies', 'dying', 'death', 'dead', 'killed', 
              'assassinated', 'assassination', 'passed', 'perished', 'deceased'],
    'BIRTH': ['born', 'birth', 'birthday', 'birthplace', 'natal'],
    
    # Time
    'WHEN': ['when', 'year', 'date', 'time', 'era', 'period', 'century'],
    'WHERE': ['where', 'place', 'location', 'city', 'country', 'region'],
    
    # Actions
    'DISCOVER': ['discover', 'discovered', 'discovery', 'found', 'invented', 
                 'developed', 'created', 'theory', 'developed'],
    'LEAD': ['president', 'leader', 'led', 'ruled', 'governed', 'commanded',
             'first', 'second', 'third', '16th', 'king', 'queen'],
    
    # Geography
    'CAPITAL': ['capital', 'city', 'largest', 'biggest', 'main'],
    'COUNTRY': ['country', 'nation', 'state', 'kingdom', 'republic'],
    
    # Cooking
    'COOK': ['cook', 'cooking', 'recipe', 'prepare', 'make'],
    'BOIL': ['boil', 'boiling', 'water', 'pot', 'pasta', 'noodles'],
    'BAKE': ['bake', 'baking', 'oven', 'bread', 'dough', 'degrees'],
    'FRY': ['fry', 'frying', 'oil', 'pan', 'chicken', 'crispy', 'golden'],
    'GRILL': ['grill', 'grilling', 'steak', 'bbq', 'charcoal', 'medium', 'rare'],
    'ROAST': ['roast', 'roasting', 'vegetables', 'oven'],
    
    # Tech
    'LIST_CMD': ['ls', 'list', 'files', 'directory', 'directories', 'folder'],
    'SEARCH_CMD': ['grep', 'search', 'find', 'text', 'pattern'],
    'SHOW_CMD': ['cat', 'display', 'show', 'view', 'contents', 'print'],
    'DISK_CMD': ['df', 'disk', 'space', 'storage', 'usage', 'filesystem'],
    'PROCESS_CMD': ['ps', 'process', 'processes', 'running', 'task', 'pid'],
    
    # Historical figures
    'WASHINGTON': ['washington', 'george', 'mount', 'vernon', 'continental', 'first'],
    'LINCOLN': ['lincoln', 'abraham', 'abe', 'civil', 'emancipation', 'gettysburg', '16th'],
    'JEFFERSON': ['jefferson', 'thomas', 'declaration', 'independence'],
    'EINSTEIN': ['einstein', 'albert', 'relativity', 'e=mc2', 'photoelectric'],
    'NEWTON': ['newton', 'isaac', 'gravity', 'motion', 'calculus', 'apple'],
    'DARWIN': ['darwin', 'charles', 'evolution', 'species', 'selection', 'beagle', 'developed'],
    'CURIE': ['curie', 'marie', 'radioactivity', 'radium', 'polonium'],
    
    # Countries/Capitals
    'FRANCE': ['france', 'paris', 'french', 'eiffel', 'seine'],
    'ENGLAND': ['england', 'london', 'english', 'british', 'uk', 'thames', 'britain'],
    'JAPAN': ['japan', 'tokyo', 'japanese', 'fuji', 'nippon'],
    'GERMANY': ['germany', 'berlin', 'german', 'deutschland'],
    'RUSSIA': ['russia', 'moscow', 'russian', 'kremlin', 'soviet'],
    'CHINA': ['china', 'beijing', 'chinese', 'mandarin', 'great wall'],
}

# Build reverse lookup: word -> cluster name
WORD_TO_CLUSTER: Dict[str, str] = {}
for cluster_name, words in SEMANTIC_CLUSTERS.items():
    for word in words:
        WORD_TO_CLUSTER[word.lower()] = cluster_name


@dataclass
class TreeNode:
    """A node in the recursive tree structure."""
    name: str
    level: int
    position: np.ndarray
    cluster_signature: Set[str] = field(default_factory=set)  # Which clusters are active
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)
    facts: List[Tuple[str, str]] = field(default_factory=list)
    weight: float = 1.0
    access_count: int = 0


class SemanticTreeEncoder:
    """
    Encoder combining semantic clusters with recursive tree organization.
    """
    
    def __init__(self, dim: int = 32, branch_threshold: float = 0.5):
        self.dim = dim
        self.branch_threshold = branch_threshold
        
        # Root node
        self.root = TreeNode(
            name="ROOT",
            level=0,
            position=np.zeros(dim),
            cluster_signature=set()
        )
        
        # Cluster positions (fixed, from bootstrap)
        self.cluster_positions: Dict[str, np.ndarray] = {}
        self._init_cluster_positions()
        
        # Word positions (learned)
        self.word_positions: Dict[str, np.ndarray] = {}
        
        # Co-occurrence (for emergent bridges)
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Stats
        self.total_facts = 0
        self.total_nodes = 1
        self.max_depth = 0
    
    def _init_cluster_positions(self):
        """Initialize cluster positions with distinct directions."""
        for i, cluster_name in enumerate(SEMANTIC_CLUSTERS.keys()):
            np.random.seed(hash(cluster_name) % (2**32))
            pos = np.random.randn(self.dim)
            pos = pos / np.linalg.norm(pos) * PHI
            self.cluster_positions[cluster_name] = pos
            np.random.seed(None)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_clusters(self, text: str) -> Set[str]:
        """Get all clusters activated by the text."""
        words = self._tokenize(text)
        clusters = set()
        for word in words:
            if word in WORD_TO_CLUSTER:
                clusters.add(WORD_TO_CLUSTER[word])
        return clusters
    
    def _get_word_position(self, word: str) -> np.ndarray:
        """Get position for a word, using cluster if available."""
        if word in self.word_positions:
            return self.word_positions[word]
        
        # Check if word is in a cluster
        if word in WORD_TO_CLUSTER:
            cluster = WORD_TO_CLUSTER[word]
            # Position near cluster centroid with small offset
            np.random.seed(hash(word) % (2**32))
            offset = np.random.randn(self.dim) * 0.1
            np.random.seed(None)
            pos = self.cluster_positions[cluster] + offset
            self.word_positions[word] = pos
            return pos
        
        # Unknown word - random position
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.3
        np.random.seed(None)
        self.word_positions[word] = pos
        return pos
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        for word in words:
            position += self._get_word_position(word)
        
        return position / len(words)
    
    def _cluster_similarity(self, clusters1: Set[str], clusters2: Set[str]) -> float:
        """Jaccard similarity between cluster signatures."""
        if not clusters1 and not clusters2:
            return 0.0
        if not clusters1 or not clusters2:
            return 0.0
        intersection = len(clusters1 & clusters2)
        union = len(clusters1 | clusters2)
        return intersection / union if union > 0 else 0.0
    
    def _position_similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Cosine similarity between positions."""
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return np.dot(pos1, pos2) / (norm1 * norm2)
    
    def _combined_similarity(self, text: str, node: TreeNode) -> float:
        """Combined similarity using both clusters and positions."""
        clusters = self._get_clusters(text)
        position = self._encode_text(text)
        
        cluster_sim = self._cluster_similarity(clusters, node.cluster_signature)
        position_sim = self._position_similarity(position, node.position)
        
        # Weight cluster similarity higher (it's more reliable)
        return 0.6 * cluster_sim + 0.4 * position_sim
    
    def _find_best_child(self, node: TreeNode, text: str) -> Tuple[Optional[TreeNode], float]:
        """Find the child most similar to the text."""
        if not node.children:
            return None, 0.0
        
        best_child = None
        best_sim = -float('inf')
        
        for child in node.children.values():
            sim = self._combined_similarity(text, child)
            if sim > best_sim:
                best_sim = sim
                best_child = child
        
        return best_child, best_sim
    
    def _find_path(self, text: str) -> List[TreeNode]:
        """Find the path through the tree for given text."""
        path = [self.root]
        current = self.root
        
        while True:
            best_child, best_sim = self._find_best_child(current, text)
            
            if best_child is None or best_sim < self.branch_threshold:
                break
            
            path.append(best_child)
            current = best_child
        
        return path
    
    def _extract_node_name(self, text: str) -> str:
        """Extract a meaningful name for a new node."""
        clusters = self._get_clusters(text)
        if clusters:
            return '_'.join(sorted(clusters)[:3])
        
        words = self._tokenize(text)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'by'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        return '_'.join(words[:3]) if words else f"node_{self.total_nodes}"
    
    def _update_cooccurrence(self, words: List[str], window: int = 5):
        """Track co-occurrence for emergent bridges."""
        for i, word in enumerate(words):
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if i != j:
                    self.cooccurrence[word][words[j]] += 1
    
    def store(self, text: str, fact_id: str) -> List[str]:
        """Store a fact in the tree."""
        words = self._tokenize(text)
        self._update_cooccurrence(words)
        
        clusters = self._get_clusters(text)
        position = self._encode_text(text)
        
        path = self._find_path(text)
        current = path[-1]
        
        # Check if we need a new branch
        best_child, best_sim = self._find_best_child(current, text)
        
        if best_child is None or best_sim < self.branch_threshold:
            # Create new branch
            node_name = self._extract_node_name(text)
            new_node = TreeNode(
                name=node_name,
                level=current.level + 1,
                position=position.copy(),
                cluster_signature=clusters.copy()
            )
            current.children[node_name] = new_node
            current = new_node
            path.append(current)
            self.total_nodes += 1
            self.max_depth = max(self.max_depth, current.level)
        else:
            # Use existing branch, update its signature
            current = best_child
            path.append(current)
            # Merge cluster signatures
            current.cluster_signature |= clusters
            # Update position as weighted average
            current.position = (current.position * current.weight + position) / (current.weight + 1)
        
        # Store fact
        current.facts.append((text, fact_id))
        current.weight += 1
        current.access_count += 1
        
        # Update access counts along path
        for node in path:
            node.access_count += 1
        
        self.total_facts += 1
        
        return [n.name for n in path]
    
    def _collect_all_facts(self, node: TreeNode = None) -> List[Tuple[str, str, TreeNode]]:
        """Collect all facts from the tree."""
        if node is None:
            node = self.root
        
        facts = [(text, fid, node) for text, fid in node.facts]
        for child in node.children.values():
            facts.extend(self._collect_all_facts(child))
        return facts
    
    def query(self, text: str) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """Query the tree using cluster-first matching."""
        path = self._find_path(text)
        query_clusters = self._get_clusters(text)
        query_position = self._encode_text(text)
        
        best_fact = None
        best_id = None
        best_score = -float('inf')
        best_node = None
        
        # Search ALL facts, prioritizing by cluster overlap
        all_facts = self._collect_all_facts()
        
        for fact_text, fact_id, node in all_facts:
            fact_clusters = self._get_clusters(fact_text)
            fact_position = self._encode_text(fact_text)
            
            cluster_sim = self._cluster_similarity(query_clusters, fact_clusters)
            position_sim = self._position_similarity(query_position, fact_position)
            
            # Weight cluster similarity much higher - it's the semantic bridge
            score = 0.7 * cluster_sim + 0.3 * position_sim
            
            if score > best_score:
                best_score = score
                best_fact = fact_text
                best_id = fact_id
                best_node = node
        
        # Update access counts along path
        for node in path:
            node.access_count += 1
        
        return best_fact, best_id, best_score, [n.name for n in path]
    
    def get_structure(self, node: TreeNode = None, indent: int = 0) -> str:
        """Get string representation of tree structure."""
        if node is None:
            node = self.root
        
        lines = []
        prefix = "  " * indent
        facts_str = f" [{len(node.facts)} facts]" if node.facts else ""
        clusters_str = f" {{{','.join(sorted(node.cluster_signature))}}}" if node.cluster_signature else ""
        lines.append(f"{prefix}{node.name}{facts_str}{clusters_str}")
        
        # Sort children by access count (Pareto ordering)
        sorted_children = sorted(node.children.values(), key=lambda n: -n.access_count)
        for child in sorted_children:
            lines.append(self.get_structure(child, indent + 1))
        
        return '\n'.join(lines)
    
    def stats(self) -> Dict:
        return {
            'total_facts': self.total_facts,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'vocabulary': len(self.word_positions),
            'clusters': len(SEMANTIC_CLUSTERS),
        }


def main():
    print("=" * 70)
    print("SEMANTIC TREE ENCODER")
    print("Semantic clusters for bridging + Recursive tree for organization")
    print("=" * 70)
    
    enc = SemanticTreeEncoder(dim=32, branch_threshold=0.3)
    
    # Data batches - watch structure grow
    batches = [
        ("BATCH 1: US Presidents", [
            ("George Washington was born in 1732 in Virginia", "gw_birth"),
            ("George Washington was the first president of the United States", "gw_president"),
            ("George Washington died in 1799 at Mount Vernon", "gw_death"),
            ("Abraham Lincoln was born in 1809 in Kentucky", "lincoln_birth"),
            ("Abraham Lincoln was the 16th president", "lincoln_president"),
            ("Abraham Lincoln was assassinated in 1865", "lincoln_death"),
        ]),
        ("BATCH 2: Scientists", [
            ("Albert Einstein developed the theory of relativity", "einstein_relativity"),
            ("Isaac Newton discovered the laws of gravity and motion", "newton_gravity"),
            ("Charles Darwin developed the theory of evolution", "darwin_evolution"),
            ("Marie Curie discovered radioactivity and radium", "curie_radioactivity"),
        ]),
        ("BATCH 3: Geography", [
            ("Paris is the capital of France", "paris"),
            ("London is the capital of England", "london"),
            ("Tokyo is the capital of Japan", "tokyo"),
            ("Berlin is the capital of Germany", "berlin"),
            ("Moscow is the capital of Russia", "moscow"),
            ("Beijing is the capital of China", "beijing"),
        ]),
        ("BATCH 4: Cooking", [
            ("To boil pasta bring water to a boil and cook for 8 minutes", "pasta"),
            ("To bake bread mix dough and bake at 450 degrees", "bread"),
            ("To fry chicken coat in flour and fry in oil until golden", "chicken"),
            ("To grill steak season and grill 4 minutes per side for medium rare", "steak"),
            ("To roast vegetables toss with oil and roast at 425 degrees", "vegetables"),
        ]),
        ("BATCH 5: Linux Commands", [
            ("The ls command lists files and directories", "ls"),
            ("The grep command searches for text patterns in files", "grep"),
            ("The cat command displays file contents", "cat"),
            ("The df command shows disk space usage", "df"),
            ("The ps command shows running processes", "ps"),
        ]),
    ]
    
    for batch_name, facts in batches:
        print(f"\n{'='*70}")
        print(batch_name)
        print("=" * 70)
        
        for text, fid in facts:
            path = enc.store(text, fid)
            print(f"  → {' → '.join(path[-3:])}")  # Show last 3 nodes of path
        
        print(f"\n  Stats: {enc.stats()}")
    
    # Show final structure
    print("\n" + "=" * 70)
    print("FINAL TREE STRUCTURE")
    print("=" * 70)
    print(enc.get_structure())
    
    # Test queries
    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)
    
    test_cases = [
        # History - using different words than stored
        ("When was Washington born", "gw_birth"),
        ("Who was the first president", "gw_president"),
        ("When did Washington die", "gw_death"),
        ("When was Lincoln born", "lincoln_birth"),
        ("Lincoln president", "lincoln_president"),
        ("How did Lincoln die", "lincoln_death"),  # die ↔ assassinated bridge
        
        # Science
        ("What did Einstein discover", "einstein_relativity"),
        ("What did Newton discover", "newton_gravity"),
        ("Darwin evolution", "darwin_evolution"),
        ("Curie radioactivity", "curie_radioactivity"),
        
        # Geography - using different phrasings
        ("Capital of France", "paris"),
        ("Capital of England", "london"),
        ("Capital of Japan", "tokyo"),
        ("Capital of Germany", "berlin"),
        ("Capital of Russia", "moscow"),
        ("Capital of China", "beijing"),
        
        # Cooking - using "how to" queries
        ("How to cook pasta", "pasta"),
        ("How to bake bread", "bread"),
        ("How to fry chicken", "chicken"),
        ("How to grill steak", "steak"),
        ("How to roast vegetables", "vegetables"),
        
        # Linux - using natural language
        ("How to list files", "ls"),
        ("How to search text in files", "grep"),
        ("How to display file contents", "cat"),
        ("How to check disk space", "df"),
        ("How to see running processes", "ps"),
    ]
    
    correct = 0
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for query, expected in test_cases:
        fact, fid, score, path = enc.query(query)
        is_correct = fid == expected
        correct += is_correct
        
        # Categorize
        if 'born' in query.lower() or 'die' in query.lower() or 'president' in query.lower():
            cat = 'History'
        elif 'discover' in query.lower() or 'einstein' in query.lower() or 'newton' in query.lower() or 'darwin' in query.lower() or 'curie' in query.lower():
            cat = 'Science'
        elif 'capital' in query.lower():
            cat = 'Geography'
        elif 'cook' in query.lower() or 'bake' in query.lower() or 'fry' in query.lower() or 'grill' in query.lower() or 'roast' in query.lower():
            cat = 'Cooking'
        else:
            cat = 'Linux'
        
        by_category[cat]['total'] += 1
        if is_correct:
            by_category[cat]['correct'] += 1
        
        marker = "✓" if is_correct else "✗"
        print(f"  {marker} \"{query}\"")
        if fact:
            answer = fact[:50] + "..." if len(fact) > 50 else fact
            print(f"      → {answer} (score={score:.2f})")
        else:
            print(f"      → No match found")
    
    print(f"\n  Overall: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")
    print("\n  By category:")
    for cat, data in sorted(by_category.items()):
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"    {cat}: {data['correct']}/{data['total']} = {acc:.0%}")
    
    # Conversation test
    print("\n" + "=" * 70)
    print("CONVERSATION TEST")
    print("=" * 70)
    
    conversation = [
        "Tell me about George Washington",
        "When was he born",
        "When did he die",
        "What about Abraham Lincoln",
        "How did Lincoln die",
        "What is the capital of France",
        "What about Japan",
        "How do I cook pasta",
        "How do I grill a steak",
        "How do I list files in Linux",
        "What did Einstein discover",
    ]
    
    for q in conversation:
        fact, fid, score, path = enc.query(q)
        print(f"\n  User: {q}")
        if fact:
            print(f"  Bot:  {fact}")
        else:
            print(f"  Bot:  I don't know about that yet.")


if __name__ == "__main__":
    main()
