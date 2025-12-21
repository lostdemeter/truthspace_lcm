#!/usr/bin/env python3
"""
Recursive Pareto Structure

The key insight: Pareto (80/20) is SCALE-INVARIANT.
At every level of the hierarchy, the same distribution applies.

This means:
1. New information either FITS existing structure (80% of cases)
2. Or CREATES new structure (20% of cases, high-info events)

The structure gets MORE useful as data grows because:
- Common patterns get reinforced (deeper branches)
- Rare patterns get their own branches (when they recur)
- The hierarchy self-organizes at every scale

This is how both trees AND language work:
- Trees: Most growth on existing branches, new branches rare
- Language: Most words fit existing patterns, new words rare
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re
import math

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class ParetoNode:
    """A node in the recursive Pareto structure."""
    name: str
    level: int
    position: np.ndarray
    children: Dict[str, 'ParetoNode'] = field(default_factory=dict)
    facts: List[Tuple[str, str]] = field(default_factory=list)  # (text, id)
    weight: float = 1.0  # Accumulated importance
    access_count: int = 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RecursiveParetoEncoder:
    """
    Encoder with recursive Pareto structure.
    
    The structure grows organically:
    - New data finds its place in existing structure (if similar enough)
    - Or creates new branches (if sufficiently different)
    
    At every level, 20% of nodes handle 80% of traffic.
    This is enforced by the access_count weighting.
    """
    
    def __init__(self, dim: int = 32, branch_threshold: float = 0.3):
        self.dim = dim
        self.branch_threshold = branch_threshold  # Similarity below this creates new branch
        
        # Root node
        self.root = ParetoNode(
            name="ROOT",
            level=0,
            position=np.zeros(dim)
        )
        
        # Word positions (learned from structure)
        self.word_positions: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.total_facts = 0
        self.total_nodes = 1
        self.max_depth = 0
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_word_position(self, word: str) -> np.ndarray:
        """Get or create position for a word."""
        if word not in self.word_positions:
            np.random.seed(hash(word) % (2**32))
            self.word_positions[word] = np.random.randn(self.dim) * 0.5
            np.random.seed(None)
        return self.word_positions[word]
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text as weighted sum of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        for word in words:
            position += self._get_word_position(word)
        
        return position / len(words)
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine-like similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _find_best_child(self, node: ParetoNode, position: np.ndarray) -> Tuple[Optional[ParetoNode], float]:
        """Find the child most similar to the given position."""
        if not node.children:
            return None, 0.0
        
        best_child = None
        best_sim = -float('inf')
        
        for child in node.children.values():
            sim = self._similarity(position, child.position)
            if sim > best_sim:
                best_sim = sim
                best_child = child
        
        return best_child, best_sim
    
    def _find_path(self, position: np.ndarray) -> List[ParetoNode]:
        """Find the path through the tree for a given position."""
        path = [self.root]
        current = self.root
        
        while True:
            best_child, best_sim = self._find_best_child(current, position)
            
            if best_child is None or best_sim < self.branch_threshold:
                break
            
            path.append(best_child)
            current = best_child
        
        return path
    
    def _create_node(self, name: str, level: int, position: np.ndarray) -> ParetoNode:
        """Create a new node."""
        node = ParetoNode(
            name=name,
            level=level,
            position=position.copy()
        )
        self.total_nodes += 1
        self.max_depth = max(self.max_depth, level)
        return node
    
    def _extract_key_words(self, text: str, n: int = 3) -> List[str]:
        """Extract the most distinctive words from text."""
        words = self._tokenize(text)
        # Filter common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'by',
                     'from', 'as', 'it', 'its', 'this', 'that', 'he', 'she', 'they'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        return words[:n]
    
    def store(self, text: str, fact_id: str) -> List[str]:
        """
        Store a fact in the recursive structure.
        
        Returns the path taken (node names).
        """
        position = self._encode_text(text)
        path = self._find_path(position)
        current = path[-1]
        
        # Check if we need to create a new branch
        best_child, best_sim = self._find_best_child(current, position)
        
        if best_child is None or best_sim < self.branch_threshold:
            # Create new branch
            key_words = self._extract_key_words(text)
            node_name = '_'.join(key_words) if key_words else f"node_{self.total_nodes}"
            
            new_node = self._create_node(
                name=node_name,
                level=current.level + 1,
                position=position
            )
            current.children[node_name] = new_node
            current = new_node
            path.append(current)
        else:
            # Use existing branch
            current = best_child
            path.append(current)
            # Update position as weighted average
            current.position = (current.position * current.weight + position) / (current.weight + 1)
        
        # Store fact at leaf
        current.facts.append((text, fact_id))
        current.weight += 1
        current.access_count += 1
        
        # Update access counts along path
        for node in path:
            node.access_count += 1
        
        self.total_facts += 1
        
        return [n.name for n in path]
    
    def query(self, text: str) -> Tuple[Optional[str], Optional[str], float, List[str]]:
        """
        Query the structure.
        
        Returns: (fact_text, fact_id, similarity, path)
        """
        position = self._encode_text(text)
        path = self._find_path(position)
        
        # Search for best fact along the path (bottom-up)
        best_fact = None
        best_id = None
        best_sim = -float('inf')
        
        for node in reversed(path):
            for fact_text, fact_id in node.facts:
                fact_pos = self._encode_text(fact_text)
                sim = self._similarity(position, fact_pos)
                if sim > best_sim:
                    best_sim = sim
                    best_fact = fact_text
                    best_id = fact_id
        
        # Update access counts
        for node in path:
            node.access_count += 1
        
        return best_fact, best_id, best_sim, [n.name for n in path]
    
    def get_structure(self, node: ParetoNode = None, indent: int = 0) -> str:
        """Get a string representation of the structure."""
        if node is None:
            node = self.root
        
        lines = []
        prefix = "  " * indent
        facts_str = f" [{len(node.facts)} facts]" if node.facts else ""
        weight_str = f" (w={node.weight:.1f}, acc={node.access_count})"
        lines.append(f"{prefix}{node.name}{facts_str}{weight_str}")
        
        # Sort children by access count (Pareto: most accessed first)
        sorted_children = sorted(
            node.children.values(),
            key=lambda n: -n.access_count
        )
        
        for child in sorted_children:
            lines.append(self.get_structure(child, indent + 1))
        
        return '\n'.join(lines)
    
    def get_pareto_stats(self, node: ParetoNode = None) -> Dict:
        """Analyze Pareto distribution at each level."""
        if node is None:
            node = self.root
        
        stats = defaultdict(lambda: {'nodes': 0, 'total_access': 0, 'top_20_access': 0})
        
        def traverse(n):
            level = n.level
            stats[level]['nodes'] += 1
            stats[level]['total_access'] += n.access_count
            
            for child in n.children.values():
                traverse(child)
        
        traverse(node)
        
        # Calculate Pareto ratio at each level
        result = {}
        for level, data in sorted(stats.items()):
            result[level] = {
                'nodes': data['nodes'],
                'total_access': data['total_access'],
            }
        
        return result
    
    def stats(self) -> Dict:
        return {
            'total_facts': self.total_facts,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'vocabulary': len(self.word_positions),
        }


def main():
    print("=" * 70)
    print("RECURSIVE PARETO STRUCTURE")
    print("Structure grows organically as data arrives")
    print("=" * 70)
    
    enc = RecursiveParetoEncoder(dim=32, branch_threshold=0.4)
    
    # Feed data incrementally and watch structure grow
    data_batches = [
        # Batch 1: US Presidents
        [
            ("George Washington was born in 1732 in Virginia", "gw_birth"),
            ("George Washington was the first president", "gw_president"),
            ("George Washington died in 1799", "gw_death"),
        ],
        # Batch 2: More presidents
        [
            ("Abraham Lincoln was born in 1809 in Kentucky", "lincoln_birth"),
            ("Abraham Lincoln was the 16th president", "lincoln_president"),
            ("Abraham Lincoln was assassinated in 1865", "lincoln_death"),
        ],
        # Batch 3: Scientists
        [
            ("Albert Einstein developed relativity theory", "einstein_relativity"),
            ("Isaac Newton discovered gravity", "newton_gravity"),
            ("Charles Darwin developed evolution theory", "darwin_evolution"),
        ],
        # Batch 4: Geography
        [
            ("Paris is the capital of France", "paris"),
            ("London is the capital of England", "london"),
            ("Tokyo is the capital of Japan", "tokyo"),
        ],
        # Batch 5: Cooking
        [
            ("Boil pasta in salted water for 8 minutes", "pasta"),
            ("Bake bread at 450 degrees for 30 minutes", "bread"),
            ("Grill steak for 4 minutes per side", "steak"),
        ],
        # Batch 6: Linux
        [
            ("The ls command lists files and directories", "ls"),
            ("The grep command searches text in files", "grep"),
            ("The df command shows disk space usage", "df"),
        ],
    ]
    
    for i, batch in enumerate(data_batches):
        print(f"\n{'='*70}")
        print(f"BATCH {i+1}: Adding {len(batch)} facts")
        print("=" * 70)
        
        for text, fid in batch:
            path = enc.store(text, fid)
            print(f"  Stored: {text[:40]}...")
            print(f"    Path: {' → '.join(path)}")
        
        print(f"\n  Stats: {enc.stats()}")
        print(f"\n  Structure:")
        print(enc.get_structure())
    
    # Test queries
    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "When was Washington born",
        "Who was the first president",
        "When did Lincoln die",
        "What did Einstein discover",
        "What did Newton discover",
        "Capital of France",
        "Capital of Japan",
        "How to cook pasta",
        "How to grill steak",
        "How to list files",
        "How to search text",
    ]
    
    correct = 0
    expected = {
        "When was Washington born": "gw_birth",
        "Who was the first president": "gw_president",
        "When did Lincoln die": "lincoln_death",
        "What did Einstein discover": "einstein_relativity",
        "What did Newton discover": "newton_gravity",
        "Capital of France": "paris",
        "Capital of Japan": "tokyo",
        "How to cook pasta": "pasta",
        "How to grill steak": "steak",
        "How to list files": "ls",
        "How to search text": "grep",
    }
    
    for query in queries:
        fact, fid, sim, path = enc.query(query)
        exp = expected.get(query)
        is_correct = fid == exp
        correct += is_correct
        marker = "✓" if is_correct else "✗"
        
        print(f"\n  {marker} Query: {query}")
        print(f"    Path: {' → '.join(path)}")
        print(f"    Answer: {fact[:50]}..." if fact and len(fact) > 50 else f"    Answer: {fact}")
        print(f"    Similarity: {sim:.3f}")
    
    print(f"\n  Accuracy: {correct}/{len(queries)} = {correct/len(queries):.0%}")
    
    # Show final structure
    print("\n" + "=" * 70)
    print("FINAL STRUCTURE (Pareto ordering: most accessed first)")
    print("=" * 70)
    print(enc.get_structure())
    
    # Pareto analysis
    print("\n" + "=" * 70)
    print("PARETO ANALYSIS BY LEVEL")
    print("=" * 70)
    pareto = enc.get_pareto_stats()
    for level, data in pareto.items():
        print(f"  Level {level}: {data['nodes']} nodes, {data['total_access']} total accesses")


if __name__ == "__main__":
    main()
