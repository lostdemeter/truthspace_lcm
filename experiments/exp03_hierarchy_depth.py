#!/usr/bin/env python3
"""
Experiment 3: Hierarchy Depth

Question: What is the optimal hierarchy depth for knowledge navigation?

We test different hierarchy structures:
1. Flat (depth=1): All domains at same level
2. Shallow (depth=2): Domains → Concepts
3. Deep (depth=3): Domains → Categories → Concepts

Trade-offs:
- Too shallow: Poor discrimination between similar concepts
- Too deep: Slow navigation, sparse regions, potential overfitting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from truthspace_lcm import TruthSpace


@dataclass
class KnowledgeNode:
    """A node in the knowledge hierarchy."""
    name: str
    keywords: List[str]
    children: List['KnowledgeNode'] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)  # Leaf commands/responses
    centroid: Optional[np.ndarray] = None
    
    def is_terminal(self) -> bool:
        return len(self.children) == 0


# =============================================================================
# HIERARCHY DEFINITIONS
# =============================================================================

def build_flat_hierarchy() -> KnowledgeNode:
    """Depth 1: Everything at root level."""
    return KnowledgeNode(
        name="ROOT",
        keywords=[],
        children=[
            KnowledgeNode("ls", ["list", "directory", "files"], entries=["ls"]),
            KnowledgeNode("mkdir", ["create", "make", "directory", "folder"], entries=["mkdir -p"]),
            KnowledgeNode("touch", ["create", "file", "new"], entries=["touch"]),
            KnowledgeNode("rm", ["delete", "remove", "file"], entries=["rm"]),
            KnowledgeNode("cp", ["copy", "file", "duplicate"], entries=["cp"]),
            KnowledgeNode("mv", ["move", "rename", "file"], entries=["mv"]),
            KnowledgeNode("ps", ["process", "list", "running"], entries=["ps"]),
            KnowledgeNode("df", ["disk", "space", "storage"], entries=["df"]),
            KnowledgeNode("greeting", ["hello", "hey", "hi"], entries=["Hello! How can I help?"]),
            KnowledgeNode("thanks", ["thanks", "thank", "appreciate"], entries=["You're welcome!"]),
            KnowledgeNode("feeling", ["feel", "feeling", "emotion"], entries=["I understand."]),
        ]
    )


def build_shallow_hierarchy() -> KnowledgeNode:
    """Depth 2: Domains → Commands/Responses."""
    return KnowledgeNode(
        name="ROOT",
        keywords=[],
        children=[
            KnowledgeNode(
                name="BASH",
                keywords=["command", "file", "directory", "process", "system"],
                children=[
                    KnowledgeNode("ls", ["list", "directory", "files"], entries=["ls"]),
                    KnowledgeNode("mkdir", ["create", "make", "directory", "folder"], entries=["mkdir -p"]),
                    KnowledgeNode("touch", ["create", "file", "new"], entries=["touch"]),
                    KnowledgeNode("rm", ["delete", "remove", "file"], entries=["rm"]),
                    KnowledgeNode("cp", ["copy", "file", "duplicate"], entries=["cp"]),
                    KnowledgeNode("mv", ["move", "rename", "file"], entries=["mv"]),
                    KnowledgeNode("ps", ["process", "list", "running"], entries=["ps"]),
                    KnowledgeNode("df", ["disk", "space", "storage"], entries=["df"]),
                ]
            ),
            KnowledgeNode(
                name="SOCIAL",
                keywords=["feel", "feeling", "hello", "thanks", "emotion", "think"],
                children=[
                    KnowledgeNode("greeting", ["hello", "hey", "hi"], entries=["Hello! How can I help?"]),
                    KnowledgeNode("thanks", ["thanks", "thank", "appreciate"], entries=["You're welcome!"]),
                    KnowledgeNode("feeling", ["feel", "feeling", "emotion"], entries=["I understand."]),
                ]
            ),
        ]
    )


def build_deep_hierarchy() -> KnowledgeNode:
    """Depth 3: Domains → Categories → Commands/Responses."""
    return KnowledgeNode(
        name="ROOT",
        keywords=[],
        children=[
            KnowledgeNode(
                name="BASH",
                keywords=["command", "file", "directory", "process", "system"],
                children=[
                    KnowledgeNode(
                        name="FILE_OPS",
                        keywords=["file", "directory", "folder", "create", "delete", "copy"],
                        children=[
                            KnowledgeNode("ls", ["list", "directory", "files"], entries=["ls"]),
                            KnowledgeNode("mkdir", ["create", "make", "directory", "folder"], entries=["mkdir -p"]),
                            KnowledgeNode("touch", ["create", "file", "new"], entries=["touch"]),
                            KnowledgeNode("rm", ["delete", "remove", "file"], entries=["rm"]),
                            KnowledgeNode("cp", ["copy", "file", "duplicate"], entries=["cp"]),
                            KnowledgeNode("mv", ["move", "rename", "file"], entries=["mv"]),
                        ]
                    ),
                    KnowledgeNode(
                        name="SYSTEM_OPS",
                        keywords=["process", "disk", "memory", "system", "monitor"],
                        children=[
                            KnowledgeNode("ps", ["process", "list", "running"], entries=["ps"]),
                            KnowledgeNode("df", ["disk", "space", "storage"], entries=["df"]),
                        ]
                    ),
                ]
            ),
            KnowledgeNode(
                name="SOCIAL",
                keywords=["feel", "feeling", "hello", "thanks", "emotion", "think"],
                children=[
                    KnowledgeNode(
                        name="GREETINGS",
                        keywords=["hello", "hey", "hi", "greet"],
                        children=[
                            KnowledgeNode("greeting", ["hello", "hey", "hi"], entries=["Hello! How can I help?"]),
                        ]
                    ),
                    KnowledgeNode(
                        name="EMOTIONAL",
                        keywords=["feel", "feeling", "emotion", "think", "thanks"],
                        children=[
                            KnowledgeNode("thanks", ["thanks", "thank", "appreciate"], entries=["You're welcome!"]),
                            KnowledgeNode("feeling", ["feel", "feeling", "emotion"], entries=["I understand."]),
                        ]
                    ),
                ]
            ),
        ]
    )


# =============================================================================
# NAVIGATION
# =============================================================================

def compute_centroids(ts: TruthSpace, node: KnowledgeNode):
    """Recursively compute centroids for all nodes."""
    if node.keywords:
        node.centroid = ts._encode(" ".join(node.keywords))
    else:
        node.centroid = np.zeros(12)  # Root has no centroid
    
    for child in node.children:
        compute_centroids(ts, child)


def navigate(ts: TruthSpace, query: str, node: KnowledgeNode, path: List[str] = None) -> tuple:
    """Navigate hierarchy to find best match. Returns (path, entry, steps)."""
    if path is None:
        path = []
    
    query_pos = ts._encode(query)
    
    if node.is_terminal():
        # At leaf - return the entry
        return path + [node.name], node.entries[0] if node.entries else None, len(path)
    
    # Find best child
    best_child = None
    best_sim = -np.inf
    
    for child in node.children:
        if child.centroid is not None:
            dist = np.sqrt(np.sum((query_pos - child.centroid) ** 2))
            sim = 1.0 / (1.0 + dist)
            if sim > best_sim:
                best_sim = sim
                best_child = child
    
    if best_child is None:
        return path, None, len(path)
    
    return navigate(ts, query, best_child, path + [node.name])


# =============================================================================
# TEST CASES
# =============================================================================

TEST_QUERIES = [
    # BASH - File operations
    ("list the files", "ls", "BASH"),
    ("create a directory", "mkdir -p", "BASH"),
    ("create a new file", "touch", "BASH"),
    ("delete the file", "rm", "BASH"),
    ("copy the file", "cp", "BASH"),
    
    # BASH - System operations
    ("show running processes", "ps", "BASH"),
    ("show disk space", "df", "BASH"),
    
    # SOCIAL
    ("hello there", "Hello! How can I help?", "SOCIAL"),
    ("thanks for your help", "You're welcome!", "SOCIAL"),
    ("I'm feeling a bit down", "I understand.", "SOCIAL"),
    
    # Ambiguous (should go to correct domain)
    ("I'm feeling out of touch", "I understand.", "SOCIAL"),  # NOT touch command
    ("touch the config file", "touch", "BASH"),  # IS touch command
]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def evaluate_hierarchy(ts: TruthSpace, name: str, root: KnowledgeNode) -> dict:
    """Evaluate a hierarchy structure."""
    compute_centroids(ts, root)
    
    correct = 0
    total = 0
    total_steps = 0
    results = []
    
    for query, expected_entry, expected_domain in TEST_QUERIES:
        path, entry, steps = navigate(ts, query, root)
        
        is_correct = entry == expected_entry
        if is_correct:
            correct += 1
        total += 1
        total_steps += steps
        
        results.append({
            'query': query,
            'expected': expected_entry,
            'got': entry,
            'path': path,
            'steps': steps,
            'correct': is_correct,
        })
    
    return {
        'name': name,
        'accuracy': correct / total,
        'correct': correct,
        'total': total,
        'avg_steps': total_steps / total,
        'results': results,
    }


def run_experiment():
    """Run the hierarchy depth experiment."""
    print("=" * 70)
    print("EXPERIMENT 3: Hierarchy Depth")
    print("=" * 70)
    print()
    
    ts = TruthSpace()
    
    hierarchies = [
        ("Flat (depth=1)", build_flat_hierarchy()),
        ("Shallow (depth=2)", build_shallow_hierarchy()),
        ("Deep (depth=3)", build_deep_hierarchy()),
    ]
    
    all_results = []
    
    for name, root in hierarchies:
        print(f"\n--- {name} ---")
        result = evaluate_hierarchy(ts, name, root)
        all_results.append(result)
        
        print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        print(f"Avg navigation steps: {result['avg_steps']:.1f}")
        
        # Show failures
        failures = [r for r in result['results'] if not r['correct']]
        if failures:
            print("Failures:")
            for f in failures:
                print(f"  '{f['query']}' → {f['got']} (expected {f['expected']})")
                print(f"    Path: {' → '.join(f['path'])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        print(f"{result['name']}: {result['accuracy']:.1%} accuracy, {result['avg_steps']:.1f} avg steps")
    
    # Best by accuracy
    best = max(all_results, key=lambda r: r['accuracy'])
    print(f"\nBest accuracy: {best['name']} ({best['accuracy']:.1%})")
    
    # Analysis: accuracy vs steps trade-off
    print("\n--- Trade-off Analysis ---")
    for result in all_results:
        efficiency = result['accuracy'] / (result['avg_steps'] + 1)  # +1 to avoid div by 0
        print(f"{result['name']}: efficiency score = {efficiency:.2f}")
    
    return all_results


if __name__ == "__main__":
    run_experiment()
