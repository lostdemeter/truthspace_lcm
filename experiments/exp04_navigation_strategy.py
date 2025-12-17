#!/usr/bin/env python3
"""
Experiment 4: Hard vs Soft Navigation

Question: Should navigation be deterministic (hard) or probabilistic (soft)?

Hard navigation: Pick the single best matching child at each level
Soft navigation: Compute weighted scores across all children, combine results

Trade-offs:
- Hard: Fast, simple, but can make irrecoverable mistakes early
- Soft: More robust to ambiguity, but slower and more complex
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from truthspace_lcm import TruthSpace


@dataclass
class KnowledgeNode:
    """A node in the knowledge hierarchy."""
    name: str
    keywords: List[str]
    children: List['KnowledgeNode'] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)
    centroid: np.ndarray = None
    
    def is_terminal(self) -> bool:
        return len(self.children) == 0


# =============================================================================
# HIERARCHY (same as exp03 shallow)
# =============================================================================

def build_hierarchy() -> KnowledgeNode:
    """Build test hierarchy."""
    return KnowledgeNode(
        name="ROOT",
        keywords=[],
        children=[
            KnowledgeNode(
                name="BASH",
                keywords=["command", "file", "directory", "process", "system", "terminal"],
                children=[
                    KnowledgeNode("ls", ["list", "directory", "files", "show"], entries=["ls"]),
                    KnowledgeNode("mkdir", ["create", "make", "directory", "folder"], entries=["mkdir -p"]),
                    KnowledgeNode("touch", ["create", "file", "new", "touch"], entries=["touch"]),
                    KnowledgeNode("rm", ["delete", "remove", "file"], entries=["rm"]),
                    KnowledgeNode("ps", ["process", "list", "running", "show"], entries=["ps"]),
                    KnowledgeNode("df", ["disk", "space", "storage", "show"], entries=["df"]),
                ]
            ),
            KnowledgeNode(
                name="SOCIAL",
                keywords=["feel", "feeling", "hello", "thanks", "emotion", "think", "I", "me"],
                children=[
                    KnowledgeNode("greeting", ["hello", "hey", "hi", "greet"], entries=["Hello! How can I help?"]),
                    KnowledgeNode("thanks", ["thanks", "thank", "appreciate"], entries=["You're welcome!"]),
                    KnowledgeNode("feeling", ["feel", "feeling", "emotion", "touch", "down"], entries=["I understand how you feel."]),
                ]
            ),
            KnowledgeNode(
                name="CREATIVE",
                keywords=["write", "poem", "story", "create", "imagine", "compose"],
                children=[
                    KnowledgeNode("poem", ["poem", "poetry", "verse"], entries=["Here's a poem for you..."]),
                    KnowledgeNode("story", ["story", "tale", "narrative"], entries=["Once upon a time..."]),
                ]
            ),
        ]
    )


def compute_centroids(ts: TruthSpace, node: KnowledgeNode):
    """Recursively compute centroids."""
    if node.keywords:
        node.centroid = ts._encode(" ".join(node.keywords))
    else:
        node.centroid = np.zeros(12)
    
    for child in node.children:
        compute_centroids(ts, child)


# =============================================================================
# NAVIGATION STRATEGIES
# =============================================================================

def similarity(query_pos: np.ndarray, centroid: np.ndarray) -> float:
    """Compute similarity between query and centroid."""
    dist = np.sqrt(np.sum((query_pos - centroid) ** 2))
    return 1.0 / (1.0 + dist)


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax probabilities."""
    scores = np.array(scores) / temperature
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    return exp_scores / np.sum(exp_scores)


def navigate_hard(ts: TruthSpace, query: str, node: KnowledgeNode) -> Tuple[str, List[str]]:
    """Hard navigation: always pick the best child."""
    query_pos = ts._encode(query)
    path = []
    
    current = node
    while not current.is_terminal():
        path.append(current.name)
        
        best_child = None
        best_sim = -np.inf
        
        for child in current.children:
            sim = similarity(query_pos, child.centroid)
            if sim > best_sim:
                best_sim = sim
                best_child = child
        
        current = best_child
    
    path.append(current.name)
    entry = current.entries[0] if current.entries else None
    return entry, path


def navigate_soft(ts: TruthSpace, query: str, node: KnowledgeNode, 
                  temperature: float = 1.0, threshold: float = 0.1) -> Tuple[str, List[str], Dict]:
    """
    Soft navigation: compute weighted scores, explore multiple paths.
    
    Returns the entry from the highest-weighted path, plus debug info.
    """
    query_pos = ts._encode(query)
    
    def explore(current: KnowledgeNode, weight: float, path: List[str]) -> List[Tuple[float, str, List[str]]]:
        """Recursively explore all paths with their weights."""
        if current.is_terminal():
            entry = current.entries[0] if current.entries else None
            return [(weight, entry, path + [current.name])]
        
        # Compute similarities to all children
        sims = [similarity(query_pos, child.centroid) for child in current.children]
        probs = softmax(sims, temperature)
        
        results = []
        for child, prob in zip(current.children, probs):
            if prob * weight >= threshold:  # Prune low-probability paths
                child_results = explore(child, weight * prob, path + [current.name])
                results.extend(child_results)
        
        return results
    
    all_paths = explore(node, 1.0, [])
    
    if not all_paths:
        return None, [], {}
    
    # Sort by weight and return best
    all_paths.sort(key=lambda x: x[0], reverse=True)
    best_weight, best_entry, best_path = all_paths[0]
    
    debug_info = {
        'all_paths': all_paths[:5],  # Top 5 paths
        'best_weight': best_weight,
    }
    
    return best_entry, best_path, debug_info


def navigate_beam(ts: TruthSpace, query: str, node: KnowledgeNode, 
                  beam_width: int = 2) -> Tuple[str, List[str]]:
    """
    Beam search navigation: keep top-k candidates at each level.
    
    A middle ground between hard and soft.
    """
    query_pos = ts._encode(query)
    
    # Each beam entry: (cumulative_score, current_node, path)
    beam = [(1.0, node, [])]
    
    while beam:
        # Check if all beam entries are terminal
        if all(entry[1].is_terminal() for entry in beam):
            break
        
        next_beam = []
        
        for score, current, path in beam:
            if current.is_terminal():
                next_beam.append((score, current, path + [current.name]))
            else:
                # Expand to children
                for child in current.children:
                    sim = similarity(query_pos, child.centroid)
                    next_beam.append((score * sim, child, path + [current.name]))
        
        # Keep top beam_width entries
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]
    
    if not beam:
        return None, []
    
    best_score, best_node, best_path = beam[0]
    entry = best_node.entries[0] if best_node.entries else None
    return entry, best_path + [best_node.name]


# =============================================================================
# TEST CASES
# =============================================================================

TEST_QUERIES = [
    # Clear cases
    ("list the files", "ls", "BASH"),
    ("create a directory", "mkdir -p", "BASH"),
    ("hello there", "Hello! How can I help?", "SOCIAL"),
    ("write me a poem", "Here's a poem for you...", "CREATIVE"),
    
    # Ambiguous cases (the interesting ones)
    ("I'm feeling out of touch", "I understand how you feel.", "SOCIAL"),
    ("touch the file", "touch", "BASH"),
    ("create a file", "touch", "BASH"),
    ("create a story", "Once upon a time...", "CREATIVE"),
    ("show me the files", "ls", "BASH"),
    ("show me how you feel", "I understand how you feel.", "SOCIAL"),
]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the navigation strategy experiment."""
    print("=" * 70)
    print("EXPERIMENT 4: Hard vs Soft Navigation")
    print("=" * 70)
    print()
    
    ts = TruthSpace()
    root = build_hierarchy()
    compute_centroids(ts, root)
    
    strategies = [
        ("Hard", lambda q: navigate_hard(ts, q, root)),
        ("Soft (temp=1.0)", lambda q: navigate_soft(ts, q, root, temperature=1.0)[:2]),
        ("Soft (temp=0.5)", lambda q: navigate_soft(ts, q, root, temperature=0.5)[:2]),
        ("Beam (width=2)", lambda q: navigate_beam(ts, q, root, beam_width=2)),
        ("Beam (width=3)", lambda q: navigate_beam(ts, q, root, beam_width=3)),
    ]
    
    results = {name: {'correct': 0, 'total': 0, 'failures': []} for name, _ in strategies}
    
    for query, expected, domain in TEST_QUERIES:
        print(f"\nQuery: \"{query}\" (expected: {expected})")
        
        for name, nav_fn in strategies:
            entry, path = nav_fn(query)
            is_correct = entry == expected
            
            results[name]['total'] += 1
            if is_correct:
                results[name]['correct'] += 1
                status = "✓"
            else:
                results[name]['failures'].append((query, expected, entry))
                status = "✗"
            
            print(f"  {name}: {status} → {entry} (path: {' → '.join(path)})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, data in results.items():
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"{name}: {acc:.1%} ({data['correct']}/{data['total']})")
    
    # Best strategy
    best_name = max(results, key=lambda n: results[n]['correct'])
    print(f"\nBest strategy: {best_name}")
    
    # Analyze where strategies differ
    print("\n--- Divergence Analysis ---")
    for query, expected, domain in TEST_QUERIES:
        entries = {}
        for name, nav_fn in strategies:
            entry, _ = nav_fn(query)
            entries[name] = entry
        
        unique_entries = set(entries.values())
        if len(unique_entries) > 1:
            print(f"\nDivergence on: \"{query}\"")
            for name, entry in entries.items():
                correct = "✓" if entry == expected else "✗"
                print(f"  {name}: {entry} {correct}")
    
    return results


if __name__ == "__main__":
    run_experiment()
