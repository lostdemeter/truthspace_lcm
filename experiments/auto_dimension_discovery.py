#!/usr/bin/env python3
"""
Automatic Dimension Discovery

This system discovers dimensions automatically without predetermining
how many or what they should be. It uses variance thresholds to decide
when to stop adding dimensions.

Key insight: Let the DATA tell us what dimensions exist.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass, field


@dataclass
class EmergentDimension:
    """A dimension that emerged from the data."""
    index: int
    variance_explained: float
    cumulative_variance: float
    negative_pole: str
    positive_pole: str
    negative_features: List[str]
    positive_features: List[str]
    positions: Dict[str, float]
    
    # Interpretation (discovered post-hoc)
    interpretation: str = ""
    correlation_with_hidden: Dict[str, float] = field(default_factory=dict)


class AutoDimensionDiscoverer:
    """
    Discovers dimensions automatically from behavioral data.
    
    The key innovation: we don't predetermine how many dimensions exist.
    We keep adding dimensions until:
    1. Cumulative variance exceeds threshold, OR
    2. Next dimension explains less than minimum variance, OR
    3. We hit a maximum number of dimensions
    """
    
    def __init__(self, 
                 variance_threshold: float = 0.85,
                 min_dimension_variance: float = 0.02,
                 max_dimensions: int = 15):
        self.variance_threshold = variance_threshold
        self.min_dimension_variance = min_dimension_variance
        self.max_dimensions = max_dimensions
        
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.agent_properties: Dict[str, Dict[str, float]] = {}  # Hidden properties for validation
        self.dimensions: List[EmergentDimension] = []
        self.agents: List[str] = []
        self.features: List[str] = []
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
    
    def ingest_corpus(self, corpus_path: str):
        """Ingest corpus and extract behavioral patterns."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        print(f"Ingesting {len(corpus['frames'])} frames...")
        
        for frame in corpus['frames']:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            hidden = frame.get('_hidden_properties', {})
            
            if not agent or not text:
                continue
            
            # Store hidden properties for validation
            if hidden and agent not in self.agent_properties:
                self.agent_properties[agent] = hidden
            
            # Extract verbs (words after agent name)
            words = text.lower().split()
            for i, w in enumerate(words):
                if agent in w.lower() and i + 1 < len(words):
                    verb = re.sub(r'[^a-z]', '', words[i + 1])
                    if len(verb) > 2:
                        self.agent_actions[agent][verb] += 1
                    break
        
        print(f"Found {len(self.agent_actions)} unique agents")
        print(f"Top agents by action diversity: {sorted([(a, len(v)) for a, v in self.agent_actions.items()], key=lambda x: -x[1])[:10]}")
    
    def build_feature_matrix(self) -> np.ndarray:
        """Build normalized feature matrix."""
        self.agents = list(self.agent_actions.keys())
        
        # Get all actions
        all_actions = set()
        for actions in self.agent_actions.values():
            all_actions.update(actions.keys())
        self.features = sorted(all_actions)
        
        n_agents = len(self.agents)
        n_features = len(self.features)
        
        print(f"Building feature matrix: {n_agents} agents × {n_features} features")
        
        # Build matrix with TF-IDF-like normalization
        X = np.zeros((n_agents, n_features))
        for i, agent in enumerate(self.agents):
            actions = self.agent_actions[agent]
            total = sum(actions.values())
            if total > 0:
                for j, action in enumerate(self.features):
                    X[i, j] = actions.get(action, 0) / total
        
        return X
    
    def discover_dimensions(self) -> List[EmergentDimension]:
        """
        Automatically discover dimensions using SVD.
        
        Returns dimensions until stopping criteria are met.
        """
        X = self.build_feature_matrix()
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # SVD
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance explained
        total_var = np.sum(self.S ** 2)
        var_ratios = (self.S ** 2) / total_var
        cumulative = np.cumsum(var_ratios)
        
        print(f"\n{'='*70}")
        print("AUTOMATIC DIMENSION DISCOVERY")
        print(f"{'='*70}")
        print(f"\nStopping criteria:")
        print(f"  - Cumulative variance > {self.variance_threshold*100:.0f}%")
        print(f"  - Single dimension variance < {self.min_dimension_variance*100:.1f}%")
        print(f"  - Maximum {self.max_dimensions} dimensions")
        
        # Discover dimensions until stopping criteria
        self.dimensions = []
        
        for i in range(min(len(self.S), self.max_dimensions)):
            var = var_ratios[i]
            cum_var = cumulative[i]
            
            # Check stopping criteria
            if var < self.min_dimension_variance:
                print(f"\n  Stopping: Dimension {i+1} variance ({var*100:.1f}%) < minimum ({self.min_dimension_variance*100:.1f}%)")
                break
            
            # Create dimension
            positions = self.U[:, i]
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            
            # Get top features
            feature_weights = self.Vt[i]
            neg_features = [self.features[j] for j in np.argsort(feature_weights)[:5]]
            pos_features = [self.features[j] for j in np.argsort(feature_weights)[-5:]]
            
            dim = EmergentDimension(
                index=i,
                variance_explained=float(var),
                cumulative_variance=float(cum_var),
                negative_pole=self.agents[min_idx],
                positive_pole=self.agents[max_idx],
                negative_features=neg_features,
                positive_features=pos_features,
                positions={self.agents[j]: float(positions[j]) for j in range(len(self.agents))},
            )
            
            self.dimensions.append(dim)
            
            print(f"\n  Dimension {i+1}: {var*100:.1f}% variance (cumulative: {cum_var*100:.1f}%)")
            print(f"    Poles: {dim.negative_pole} <---> {dim.positive_pole}")
            print(f"    - features: {neg_features}")
            print(f"    + features: {pos_features}")
            
            # Check cumulative variance threshold
            if cum_var >= self.variance_threshold:
                print(f"\n  Stopping: Cumulative variance ({cum_var*100:.1f}%) >= threshold ({self.variance_threshold*100:.0f}%)")
                break
        
        print(f"\n  TOTAL: {len(self.dimensions)} dimensions discovered")
        return self.dimensions
    
    def correlate_with_hidden_properties(self):
        """
        Correlate discovered dimensions with hidden properties.
        
        This tells us what the system ACTUALLY discovered.
        """
        if not self.agent_properties:
            print("\nNo hidden properties available for validation")
            return
        
        print(f"\n{'='*70}")
        print("CORRELATION WITH HIDDEN PROPERTIES")
        print(f"{'='*70}")
        
        # Get list of hidden property names
        sample_props = next(iter(self.agent_properties.values()))
        prop_names = list(sample_props.keys())
        
        # Build ground truth vectors
        ground_truth = {prop: [] for prop in prop_names}
        valid_agents = []
        
        for agent in self.agents:
            if agent in self.agent_properties:
                valid_agents.append(agent)
                for prop in prop_names:
                    ground_truth[prop].append(self.agent_properties[agent].get(prop, 0))
        
        print(f"\nValidating against {len(valid_agents)} agents with known properties")
        print(f"Hidden properties: {prop_names}")
        
        # Correlate each dimension with each property
        for dim in self.dimensions:
            positions = [dim.positions.get(a, 0) for a in valid_agents]
            
            print(f"\nDimension {dim.index + 1} ({dim.negative_pole} <-> {dim.positive_pole}):")
            
            best_corr = 0
            best_prop = ""
            
            for prop in prop_names:
                gt_values = ground_truth[prop]
                if np.std(positions) > 0 and np.std(gt_values) > 0:
                    corr = np.corrcoef(positions, gt_values)[0, 1]
                    dim.correlation_with_hidden[prop] = corr
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_prop = prop
                    
                    # Only print significant correlations
                    if abs(corr) > 0.3:
                        print(f"    {prop}: {corr:+.3f} {'***' if abs(corr) > 0.7 else '**' if abs(corr) > 0.5 else '*'}")
            
            # Set interpretation
            if abs(best_corr) > 0.5:
                dim.interpretation = f"{best_prop} ({best_corr:+.3f})"
                print(f"  → Best match: {best_prop.upper()} (r={best_corr:+.3f})")
            else:
                dim.interpretation = "mixed/unknown"
                print(f"  → No strong match (best: {best_prop} r={best_corr:+.3f})")
    
    def show_agent_positions(self, agents: List[str]):
        """Show positions of specific agents across all dimensions."""
        print(f"\n{'='*70}")
        print("AGENT POSITIONS IN EMERGENT SPACE")
        print(f"{'='*70}")
        
        for agent in agents:
            agent_lower = agent.lower()
            if agent_lower not in self.agents:
                print(f"\n{agent}: Not found in corpus")
                continue
            
            print(f"\n{agent.upper()}:")
            for dim in self.dimensions:
                pos = dim.positions.get(agent_lower, 0)
                # Determine which pole they're closer to
                pole = dim.positive_pole if pos > 0 else dim.negative_pole
                interp = dim.interpretation or f"Dim{dim.index+1}"
                print(f"  {interp}: {pos:+.3f} (toward {pole})")
    
    def find_similar(self, agent: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar agents."""
        agent_lower = agent.lower()
        if agent_lower not in self.agents:
            return []
        
        # Get position vector
        pos = np.array([dim.positions.get(agent_lower, 0) for dim in self.dimensions])
        
        # Compare to all others
        similarities = []
        for other in self.agents:
            if other != agent_lower:
                other_pos = np.array([dim.positions.get(other, 0) for dim in self.dimensions])
                dist = np.linalg.norm(pos - other_pos)
                similarities.append((other, dist))
        
        return sorted(similarities, key=lambda x: x[1])[:k]
    
    def find_opposite(self, agent: str) -> Optional[Tuple[str, float]]:
        """Find the most opposite agent."""
        agent_lower = agent.lower()
        if agent_lower not in self.agents:
            return None
        
        pos = np.array([dim.positions.get(agent_lower, 0) for dim in self.dimensions])
        
        max_dist = 0
        opposite = None
        
        for other in self.agents:
            if other != agent_lower:
                other_pos = np.array([dim.positions.get(other, 0) for dim in self.dimensions])
                dist = np.linalg.norm(pos - other_pos)
                if dist > max_dist:
                    max_dist = dist
                    opposite = other
        
        return (opposite, max_dist) if opposite else None


def main():
    print("=" * 70)
    print("AUTOMATIC DIMENSION DISCOVERY EXPERIMENT")
    print("=" * 70)
    
    # Load LLM-generated corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_generated.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run llm_corpus_generator.py first!")
        return None
    
    # Create discoverer with automatic stopping
    discoverer = AutoDimensionDiscoverer(
        variance_threshold=0.80,      # Stop when 80% variance explained
        min_dimension_variance=0.03,  # Stop if dimension < 3% variance
        max_dimensions=12,            # Hard cap
    )
    
    # Ingest and discover
    discoverer.ingest_corpus(str(corpus_path))
    dimensions = discoverer.discover_dimensions()
    
    # Correlate with hidden properties
    discoverer.correlate_with_hidden_properties()
    
    # Show positions for key characters
    test_agents = ['holmes', 'watson', 'moriarty', 'alice', 'queen', 'king', 
                   'child', 'elder', 'storm', 'robot', 'villain', 'sage']
    discoverer.show_agent_positions(test_agents)
    
    # Find similar and opposite
    print(f"\n{'='*70}")
    print("SIMILARITY ANALYSIS")
    print(f"{'='*70}")
    
    for agent in ['holmes', 'watson', 'villain', 'child']:
        similar = discoverer.find_similar(agent, k=3)
        opposite = discoverer.find_opposite(agent)
        print(f"\n{agent.upper()}:")
        print(f"  Similar: {[s[0] for s in similar]}")
        if opposite:
            print(f"  Opposite: {opposite[0]} (distance: {opposite[1]:.3f})")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nDiscovered {len(dimensions)} dimensions automatically:")
    for dim in dimensions:
        interp = dim.interpretation or "unknown"
        print(f"  Dim {dim.index+1}: {dim.variance_explained*100:.1f}% - {interp}")
        print(f"         {dim.negative_pole} <---> {dim.positive_pole}")
    
    total_var = sum(d.variance_explained for d in dimensions)
    print(f"\nTotal variance explained: {total_var*100:.1f}%")
    
    return discoverer


if __name__ == "__main__":
    discoverer = main()
