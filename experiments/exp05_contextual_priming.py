#!/usr/bin/env python3
"""
Experiment 5: Contextual Priming

Question: How does conversation history affect domain selection?

Hypothesis: Recent queries create a "prior" that biases navigation toward
the same domain. This is similar to how humans stay "in context" during
a conversation.

We test:
1. No priming (baseline)
2. Single-query priming (last query affects current)
3. Multi-query priming (weighted history)
4. Decay priming (older queries have less influence)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Tuple, Optional
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
# HIERARCHY
# =============================================================================

def build_hierarchy() -> KnowledgeNode:
    """Build test hierarchy."""
    return KnowledgeNode(
        name="ROOT",
        keywords=[],
        children=[
            KnowledgeNode(
                name="BASH",
                keywords=["command", "file", "directory", "process", "system"],
                children=[
                    KnowledgeNode("ls", ["list", "directory", "files"], entries=["ls"]),
                    KnowledgeNode("mkdir", ["create", "make", "directory"], entries=["mkdir -p"]),
                    KnowledgeNode("touch", ["create", "file", "touch"], entries=["touch"]),
                    KnowledgeNode("rm", ["delete", "remove"], entries=["rm"]),
                ]
            ),
            KnowledgeNode(
                name="SOCIAL",
                keywords=["feel", "feeling", "hello", "thanks", "emotion", "think"],
                children=[
                    KnowledgeNode("greeting", ["hello", "hey", "hi"], entries=["Hello!"]),
                    KnowledgeNode("feeling", ["feel", "feeling", "touch", "emotion"], entries=["I understand."]),
                    KnowledgeNode("thanks", ["thanks", "appreciate"], entries=["You're welcome!"]),
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
# PRIMING STRATEGIES
# =============================================================================

class ConversationContext:
    """Maintains conversation history for priming."""
    
    def __init__(self, ts: TruthSpace, max_history: int = 5):
        self.ts = ts
        self.history: List[Tuple[str, np.ndarray]] = []  # (query, encoding)
        self.max_history = max_history
    
    def add(self, query: str):
        """Add a query to history."""
        encoding = self.ts._encode(query)
        self.history.append((query, encoding))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def clear(self):
        """Clear history."""
        self.history = []
    
    def get_prior_none(self) -> Optional[np.ndarray]:
        """No priming - return None."""
        return None
    
    def get_prior_last(self) -> Optional[np.ndarray]:
        """Single-query priming - use last query encoding."""
        if not self.history:
            return None
        return self.history[-1][1]
    
    def get_prior_mean(self) -> Optional[np.ndarray]:
        """Multi-query priming - mean of all history."""
        if not self.history:
            return None
        encodings = [enc for _, enc in self.history]
        return np.mean(encodings, axis=0)
    
    def get_prior_decay(self, decay: float = 0.5) -> Optional[np.ndarray]:
        """Decay priming - exponentially weighted history."""
        if not self.history:
            return None
        
        weights = []
        for i in range(len(self.history)):
            # Most recent has weight 1, older ones decay
            age = len(self.history) - 1 - i
            weights.append(decay ** age)
        
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize
        
        encodings = np.array([enc for _, enc in self.history])
        return np.average(encodings, axis=0, weights=weights)


def navigate_with_prior(ts: TruthSpace, query: str, node: KnowledgeNode,
                        prior: Optional[np.ndarray] = None,
                        prior_weight: float = 0.3) -> Tuple[str, List[str]]:
    """Navigate with optional prior bias."""
    query_pos = ts._encode(query)
    
    # Blend query with prior if available
    if prior is not None:
        effective_pos = (1 - prior_weight) * query_pos + prior_weight * prior
    else:
        effective_pos = query_pos
    
    path = []
    current = node
    
    while not current.is_terminal():
        path.append(current.name)
        
        best_child = None
        best_sim = -np.inf
        
        for child in current.children:
            dist = np.sqrt(np.sum((effective_pos - child.centroid) ** 2))
            sim = 1.0 / (1.0 + dist)
            if sim > best_sim:
                best_sim = sim
                best_child = child
        
        current = best_child
    
    path.append(current.name)
    entry = current.entries[0] if current.entries else None
    return entry, path


# =============================================================================
# TEST SCENARIOS
# =============================================================================

# Each scenario is a sequence of queries with expected results
# The key test: does context help disambiguate?

SCENARIOS = [
    {
        'name': 'BASH context helps disambiguate "touch"',
        'queries': [
            ("list the files", "ls", "BASH"),
            ("create a directory", "mkdir -p", "BASH"),
            ("touch the file", "touch", "BASH"),  # Should be BASH due to context
        ],
    },
    {
        'name': 'SOCIAL context helps disambiguate "touch"',
        'queries': [
            ("hello there", "Hello!", "SOCIAL"),
            ("how are you feeling", "I understand.", "SOCIAL"),
            ("I feel out of touch", "I understand.", "SOCIAL"),  # Should be SOCIAL due to context
        ],
    },
    {
        'name': 'Context switch: BASH to SOCIAL',
        'queries': [
            ("list files", "ls", "BASH"),
            ("create directory", "mkdir -p", "BASH"),
            ("hello", "Hello!", "SOCIAL"),  # Context switch
            ("I feel sad", "I understand.", "SOCIAL"),
        ],
    },
    {
        'name': 'Context switch: SOCIAL to BASH',
        'queries': [
            ("hey there", "Hello!", "SOCIAL"),
            ("thanks", "You're welcome!", "SOCIAL"),
            ("list files", "ls", "BASH"),  # Context switch
            ("delete the file", "rm", "BASH"),
        ],
    },
    {
        'name': 'Ambiguous without context',
        'queries': [
            ("touch", "touch", "BASH"),  # Ambiguous - could be either
        ],
    },
    {
        'name': 'Ambiguous with BASH context',
        'queries': [
            ("list files", "ls", "BASH"),
            ("touch", "touch", "BASH"),  # Should lean BASH
        ],
    },
    {
        'name': 'Ambiguous with SOCIAL context',
        'queries': [
            ("hello", "Hello!", "SOCIAL"),
            ("feeling down", "I understand.", "SOCIAL"),
            ("touch", "I understand.", "SOCIAL"),  # Should lean SOCIAL? (tricky)
        ],
    },
]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the contextual priming experiment."""
    print("=" * 70)
    print("EXPERIMENT 5: Contextual Priming")
    print("=" * 70)
    print()
    
    ts = TruthSpace()
    root = build_hierarchy()
    compute_centroids(ts, root)
    
    priming_strategies = [
        ("No priming", lambda ctx: ctx.get_prior_none()),
        ("Last query", lambda ctx: ctx.get_prior_last()),
        ("Mean history", lambda ctx: ctx.get_prior_mean()),
        ("Decay (0.5)", lambda ctx: ctx.get_prior_decay(0.5)),
        ("Decay (0.7)", lambda ctx: ctx.get_prior_decay(0.7)),
    ]
    
    # Results per strategy
    strategy_results = {name: {'correct': 0, 'total': 0} for name, _ in priming_strategies}
    
    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print('='*60)
        
        for strategy_name, get_prior_fn in priming_strategies:
            ctx = ConversationContext(ts)
            scenario_correct = 0
            
            print(f"\n  --- {strategy_name} ---")
            
            for query, expected, domain in scenario['queries']:
                prior = get_prior_fn(ctx)
                entry, path = navigate_with_prior(ts, query, root, prior)
                
                is_correct = entry == expected
                if is_correct:
                    scenario_correct += 1
                    strategy_results[strategy_name]['correct'] += 1
                strategy_results[strategy_name]['total'] += 1
                
                status = "✓" if is_correct else "✗"
                print(f"    {status} \"{query}\" → {entry} (expected: {expected})")
                
                # Add to context AFTER navigation
                ctx.add(query)
            
            print(f"    Scenario accuracy: {scenario_correct}/{len(scenario['queries'])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    for name, data in strategy_results.items():
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"{name}: {acc:.1%} ({data['correct']}/{data['total']})")
    
    best_name = max(strategy_results, key=lambda n: strategy_results[n]['correct'])
    print(f"\nBest strategy: {best_name}")
    
    # Analysis: where does priming help vs hurt?
    print("\n--- Priming Impact Analysis ---")
    
    # Compare no-priming vs best-priming on each scenario
    for scenario in SCENARIOS:
        ctx_none = ConversationContext(ts)
        ctx_best = ConversationContext(ts)
        
        none_correct = 0
        best_correct = 0
        
        for query, expected, domain in scenario['queries']:
            # No priming
            entry_none, _ = navigate_with_prior(ts, query, root, None)
            if entry_none == expected:
                none_correct += 1
            
            # Best priming (decay 0.7 based on typical results)
            prior = ctx_best.get_prior_decay(0.7)
            entry_best, _ = navigate_with_prior(ts, query, root, prior)
            if entry_best == expected:
                best_correct += 1
            
            ctx_none.add(query)
            ctx_best.add(query)
        
        diff = best_correct - none_correct
        if diff != 0:
            direction = "+" if diff > 0 else ""
            print(f"  {scenario['name']}: {direction}{diff} with priming")
    
    return strategy_results


if __name__ == "__main__":
    run_experiment()
