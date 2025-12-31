#!/usr/bin/env python3
"""
Test emergent dimension discovery on the clean dimensional corpus.

This tests whether the system can REDISCOVER the known dimensions
(gender, age, agency, animacy) from behavior patterns alone.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def load_corpus(path: str) -> Dict:
    """Load corpus from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_agent_behaviors(corpus: Dict) -> Dict[str, Dict]:
    """Extract behavioral patterns for each agent."""
    
    agent_data = defaultdict(lambda: {
        'actions': defaultdict(int),
        'targets': set(),
        'roles': set(),
        'frame_count': 0,
        'properties': None,  # Ground truth from corpus
    })
    
    for frame in corpus['frames']:
        agent = frame.get('agent', '').lower()
        text = frame.get('text', '')
        props = frame.get('properties', {})
        
        if not agent:
            continue
        
        agent_data[agent]['frame_count'] += 1
        agent_data[agent]['properties'] = props
        
        # Extract action (word after agent name)
        words = text.lower().split()
        for i, w in enumerate(words):
            if agent in w and i + 1 < len(words):
                verb = re.sub(r'[^a-z]', '', words[i + 1])
                if len(verb) > 2:
                    agent_data[agent]['actions'][verb] += 1
                break
        
        # Extract role (word before agent in "is a X" pattern)
        if ' is a ' in text.lower():
            match = re.search(r'is a (\w+)', text.lower())
            if match:
                agent_data[agent]['roles'].add(match.group(1))
        
        # Extract targets
        for i, w in enumerate(words):
            if agent in w:
                for target in words[i+2:i+5]:
                    target = re.sub(r'[^a-z]', '', target)
                    if len(target) > 2 and target not in {'the', 'and', 'with'}:
                        agent_data[agent]['targets'].add(target)
                break
    
    return dict(agent_data)


class EmergentDimensionDiscoverer:
    """Discovers dimensions from behavioral patterns."""
    
    def __init__(self, agent_data: Dict[str, Dict]):
        self.agent_data = agent_data
        self.discovered_dimensions = []
        
        # Verb categories (will be refined through discovery)
        self.high_agency_verbs = set()
        self.low_agency_verbs = set()
        self.male_indicators = set()
        self.female_indicators = set()
        self.young_indicators = set()
        self.old_indicators = set()
    
    def discover_verb_clusters(self):
        """Discover which verbs correlate with which properties."""
        
        # Use ground truth to bootstrap verb categories
        for agent, data in self.agent_data.items():
            props = data.get('properties', {})
            if not props:
                continue
            
            agency = props.get('agency', 0)
            gender = props.get('gender', 0)
            age = props.get('age', 0)
            
            for verb, count in data['actions'].items():
                if agency > 0.5:
                    self.high_agency_verbs.add(verb)
                elif agency < 0.3:
                    self.low_agency_verbs.add(verb)
                
                if gender < -0.3:
                    self.male_indicators.add(verb)
                elif gender > 0.3:
                    self.female_indicators.add(verb)
                
                if age < -0.3:
                    self.young_indicators.add(verb)
                elif age > 0.3:
                    self.old_indicators.add(verb)
        
        print("Discovered verb clusters:")
        print(f"  High agency: {list(self.high_agency_verbs)[:10]}")
        print(f"  Low agency: {list(self.low_agency_verbs)[:10]}")
        print(f"  Young indicators: {list(self.young_indicators)[:10]}")
        print(f"  Old indicators: {list(self.old_indicators)[:10]}")
    
    def compute_dimension_score(self, agent: str, dimension: str) -> float:
        """Compute a score for an agent on a dimension based on behavior."""
        
        data = self.agent_data.get(agent, {})
        actions = data.get('actions', {})
        total = sum(actions.values())
        
        if total == 0:
            return 0.0
        
        if dimension == 'agency':
            high = sum(actions.get(v, 0) for v in self.high_agency_verbs)
            low = sum(actions.get(v, 0) for v in self.low_agency_verbs)
            return (high - low) / total
        
        elif dimension == 'age':
            young = sum(actions.get(v, 0) for v in self.young_indicators)
            old = sum(actions.get(v, 0) for v in self.old_indicators)
            return (old - young) / total
        
        return 0.0
    
    def discover_dimensions_from_variance(self) -> List[Dict]:
        """
        Discover dimensions by finding axes of maximum variance.
        
        This is the key: we don't predefine dimensions, we find them
        by looking at what separates concepts.
        """
        
        agents = list(self.agent_data.keys())
        n_agents = len(agents)
        
        # Build feature matrix from behaviors
        # Features: action frequencies, role indicators, target patterns
        all_actions = set()
        for data in self.agent_data.values():
            all_actions.update(data['actions'].keys())
        
        action_list = sorted(all_actions)
        n_features = len(action_list)
        
        print(f"\nBuilding feature matrix: {n_agents} agents × {n_features} action features")
        
        # Build matrix
        X = np.zeros((n_agents, n_features))
        for i, agent in enumerate(agents):
            actions = self.agent_data[agent]['actions']
            total = sum(actions.values())
            if total > 0:
                for j, action in enumerate(action_list):
                    X[i, j] = actions.get(action, 0) / total
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # SVD to find principal components
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance explained
        total_var = np.sum(S ** 2)
        var_ratios = (S ** 2) / total_var
        
        print(f"\nVariance explained by top dimensions:")
        for i in range(min(6, len(var_ratios))):
            print(f"  Dimension {i+1}: {var_ratios[i]*100:.1f}%")
        
        # Interpret dimensions
        dimensions = []
        for dim_idx in range(min(4, len(S))):
            # Get agent positions on this dimension
            positions = U[:, dim_idx]
            
            # Find poles
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            
            # Get top features for this dimension
            feature_weights = Vt[dim_idx]
            top_pos_features = [action_list[j] for j in np.argsort(feature_weights)[-5:]]
            top_neg_features = [action_list[j] for j in np.argsort(feature_weights)[:5]]
            
            dim_info = {
                'index': dim_idx,
                'variance': float(var_ratios[dim_idx]),
                'negative_pole': agents[min_idx],
                'positive_pole': agents[max_idx],
                'negative_features': top_neg_features,
                'positive_features': top_pos_features,
                'positions': {agents[i]: float(positions[i]) for i in range(n_agents)},
            }
            dimensions.append(dim_info)
            
            print(f"\n  Dimension {dim_idx + 1}:")
            print(f"    Poles: {agents[min_idx]} <---> {agents[max_idx]}")
            print(f"    Negative features: {top_neg_features}")
            print(f"    Positive features: {top_pos_features}")
            
            # Show agents at each extreme
            sorted_agents = sorted(dim_info['positions'].items(), key=lambda x: x[1])
            print(f"    Low end: {[a for a, p in sorted_agents[:5]]}")
            print(f"    High end: {[a for a, p in sorted_agents[-5:]]}")
        
        self.discovered_dimensions = dimensions
        return dimensions
    
    def correlate_with_ground_truth(self):
        """Check if discovered dimensions correlate with known properties."""
        
        print("\n" + "=" * 70)
        print("CORRELATION WITH GROUND TRUTH")
        print("=" * 70)
        
        # Get ground truth values
        agents = list(self.agent_data.keys())
        ground_truth = {
            'gender': [],
            'age': [],
            'agency': [],
            'animacy': [],
        }
        
        for agent in agents:
            props = self.agent_data[agent].get('properties', {})
            for dim in ground_truth:
                ground_truth[dim].append(props.get(dim, 0))
        
        # Correlate each discovered dimension with each ground truth
        for disc_dim in self.discovered_dimensions:
            positions = [disc_dim['positions'].get(a, 0) for a in agents]
            
            print(f"\nDiscovered Dimension {disc_dim['index'] + 1}:")
            print(f"  (Poles: {disc_dim['negative_pole']} <-> {disc_dim['positive_pole']})")
            
            for gt_name, gt_values in ground_truth.items():
                # Pearson correlation
                if np.std(positions) > 0 and np.std(gt_values) > 0:
                    corr = np.corrcoef(positions, gt_values)[0, 1]
                    print(f"    Correlation with {gt_name}: {corr:.3f}")


def main():
    print("=" * 70)
    print("EMERGENT DIMENSION DISCOVERY ON CLEAN CORPUS")
    print("=" * 70)
    
    # Load clean corpus - prefer rich behavioral corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_rich_behavioral.json"
    
    if not corpus_path.exists():
        # Fall back to dimensional corpus
        corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_dimensional.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found")
        print("Run generate_rich_corpus.py or corpus_builder.py first!")
        return
    
    corpus = load_corpus(str(corpus_path))
    print(f"\nLoaded {len(corpus['frames'])} frames")
    
    # Extract behaviors
    print("\n--- Extracting Agent Behaviors ---")
    agent_data = extract_agent_behaviors(corpus)
    print(f"Found {len(agent_data)} unique agents")
    
    # Show sample
    print("\nSample agent behaviors:")
    for agent in list(agent_data.keys())[:5]:
        data = agent_data[agent]
        top_actions = sorted(data['actions'].items(), key=lambda x: -x[1])[:3]
        props = data.get('properties', {})
        print(f"  {agent}: actions={top_actions}, props={props}")
    
    # Discover dimensions
    print("\n--- Discovering Dimensions from Behavior ---")
    discoverer = EmergentDimensionDiscoverer(agent_data)
    discoverer.discover_verb_clusters()
    dimensions = discoverer.discover_dimensions_from_variance()
    
    # Correlate with ground truth
    discoverer.correlate_with_ground_truth()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nThe system discovered dimensions from BEHAVIOR alone.")
    print("Correlation with ground truth shows which known dimensions were rediscovered.")
    print("\nKey insight: If correlation is high, the dimension was successfully")
    print("discovered from behavioral patterns without being told what to look for.")
    
    return discoverer


if __name__ == "__main__":
    discoverer = main()
