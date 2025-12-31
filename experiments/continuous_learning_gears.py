#!/usr/bin/env python3
"""
Continuous Learning Gear System

A gear system that learns continuously as new data arrives.
Key features:
1. Incremental dimension discovery - add new dimensions when variance unexplained
2. Dimension refinement - update existing dimensions with new data
3. Automatic gear spawning - create new gears when new dimensions emerge
4. Stability tracking - monitor how stable dimensions are over time

This implements the "error = where to build" principle:
When the system can't explain variance, it adds new dimensions.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass, field
from datetime import datetime
import copy


@dataclass
class DimensionHistory:
    """Track the history of a dimension over time."""
    name: str
    creation_time: str
    updates: List[Dict] = field(default_factory=list)
    stability_scores: List[float] = field(default_factory=list)
    
    def add_update(self, variance: float, poles: Tuple[str, str], n_samples: int):
        self.updates.append({
            'time': datetime.now().isoformat(),
            'variance': variance,
            'poles': poles,
            'n_samples': n_samples,
        })
    
    def compute_stability(self) -> float:
        """Compute how stable this dimension has been."""
        if len(self.updates) < 2:
            return 0.0
        
        # Check if poles have been consistent
        pole_sets = [set(u['poles']) for u in self.updates]
        pole_consistency = sum(1 for i in range(1, len(pole_sets)) 
                              if pole_sets[i] == pole_sets[i-1]) / (len(pole_sets) - 1)
        
        # Check if variance has been stable
        variances = [u['variance'] for u in self.updates]
        if len(variances) > 1:
            var_std = np.std(variances)
            var_mean = np.mean(variances)
            variance_stability = 1.0 - min(1.0, var_std / (var_mean + 0.001))
        else:
            variance_stability = 0.0
        
        stability = (pole_consistency + variance_stability) / 2
        self.stability_scores.append(stability)
        return stability


@dataclass
class EmergentDimension:
    """A dimension that emerged from data."""
    index: int
    name: str
    variance_explained: float
    negative_pole: str
    positive_pole: str
    negative_features: List[str]
    positive_features: List[str]
    positions: Dict[str, float]
    history: DimensionHistory = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = DimensionHistory(
                name=self.name,
                creation_time=datetime.now().isoformat()
            )


class ContinuousLearningSystem:
    """
    A gear system that learns continuously from streaming data.
    
    Key principles:
    1. Start with no dimensions
    2. Add dimensions when unexplained variance is high
    3. Refine dimensions as more data arrives
    4. Track stability to know which dimensions are "real"
    """
    
    def __init__(self, 
                 min_dimension_variance: float = 0.025,
                 max_dimensions: int = 20,
                 unexplained_variance_threshold: float = 0.30):
        
        self.min_dimension_variance = min_dimension_variance
        self.max_dimensions = max_dimensions
        self.unexplained_variance_threshold = unexplained_variance_threshold
        
        # Data storage
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_frames: int = 0
        
        # Discovered dimensions and gears
        self.dimensions: List[EmergentDimension] = []
        self.dimension_history: Dict[str, DimensionHistory] = {}
        
        # SVD components (cached)
        self.agents: List[str] = []
        self.features: List[str] = []
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        
        # Learning state
        self.learning_cycles: int = 0
        self.variance_explained_history: List[float] = []
    
    def ingest_frame(self, frame: Dict):
        """Ingest a single frame incrementally."""
        text = frame.get('text', '')
        agent = frame.get('agent', '').lower()
        
        if not agent or not text or len(agent) < 2:
            return
        
        # Extract verbs
        words = text.lower().split()
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^a-z]', '', word)
            if len(word_clean) < 3:
                continue
            
            # Heuristic verb detection
            verb_endings = ['ed', 'ing', 'es', 's']
            if any(word_clean.endswith(e) for e in verb_endings):
                self.agent_actions[agent][word_clean] += 1
        
        self.total_frames += 1
    
    def ingest_batch(self, frames: List[Dict]):
        """Ingest a batch of frames."""
        for frame in frames:
            self.ingest_frame(frame)
    
    def ingest_corpus(self, corpus_path: str):
        """Ingest a full corpus file."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        self.ingest_batch(corpus['frames'])
        print(f"Ingested {len(corpus['frames'])} frames, total now: {self.total_frames}")
    
    def _build_feature_matrix(self) -> np.ndarray:
        """Build feature matrix from current data."""
        # Filter agents with sufficient data
        min_actions = 3
        valid_agents = {
            a: v for a, v in self.agent_actions.items()
            if sum(v.values()) >= min_actions and len(a) > 2
        }
        
        self.agents = list(valid_agents.keys())
        
        # Get all features
        all_actions = set()
        for actions in valid_agents.values():
            all_actions.update(actions.keys())
        self.features = sorted(all_actions)
        
        n_agents = len(self.agents)
        n_features = len(self.features)
        
        if n_agents < 2 or n_features < 2:
            return np.array([])
        
        # Build normalized matrix
        X = np.zeros((n_agents, n_features))
        for i, agent in enumerate(self.agents):
            actions = valid_agents[agent]
            total = sum(actions.values())
            if total > 0:
                for j, action in enumerate(self.features):
                    X[i, j] = actions.get(action, 0) / total
        
        return X
    
    def learn(self) -> Dict[str, Any]:
        """
        Run a learning cycle.
        
        Returns information about what was learned.
        """
        self.learning_cycles += 1
        
        result = {
            'cycle': self.learning_cycles,
            'total_frames': self.total_frames,
            'n_agents': 0,
            'n_features': 0,
            'dimensions_before': len(self.dimensions),
            'dimensions_after': 0,
            'new_dimensions': [],
            'refined_dimensions': [],
            'variance_explained': 0.0,
            'unexplained_variance': 0.0,
        }
        
        # Build feature matrix
        X = self._build_feature_matrix()
        if X.size == 0:
            print("Not enough data for learning")
            return result
        
        result['n_agents'] = len(self.agents)
        result['n_features'] = len(self.features)
        
        # Center and SVD
        X_centered = X - X.mean(axis=0)
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance analysis
        total_var = np.sum(self.S ** 2)
        var_ratios = (self.S ** 2) / total_var
        
        # Discover/refine dimensions
        old_dimensions = {d.name: d for d in self.dimensions}
        new_dimensions = []
        
        cumulative_var = 0.0
        for i in range(min(len(self.S), self.max_dimensions)):
            var = var_ratios[i]
            cumulative_var += var
            
            if var < self.min_dimension_variance:
                break
            
            # Get dimension info
            positions = self.U[:, i]
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            
            feature_weights = self.Vt[i]
            neg_features = [self.features[j] for j in np.argsort(feature_weights)[:5]]
            pos_features = [self.features[j] for j in np.argsort(feature_weights)[-5:]]
            
            neg_pole = self.agents[min_idx]
            pos_pole = self.agents[max_idx]
            
            # Check if this matches an existing dimension
            dim_name = f"Dim{i+1}"
            existing = old_dimensions.get(dim_name)
            
            if existing:
                # Refine existing dimension
                existing.variance_explained = float(var)
                existing.negative_pole = neg_pole
                existing.positive_pole = pos_pole
                existing.negative_features = neg_features
                existing.positive_features = pos_features
                existing.positions = {self.agents[j]: float(positions[j]) for j in range(len(self.agents))}
                existing.history.add_update(var, (neg_pole, pos_pole), self.total_frames)
                
                new_dimensions.append(existing)
                result['refined_dimensions'].append(dim_name)
            else:
                # Create new dimension
                dim = EmergentDimension(
                    index=i,
                    name=dim_name,
                    variance_explained=float(var),
                    negative_pole=neg_pole,
                    positive_pole=pos_pole,
                    negative_features=neg_features,
                    positive_features=pos_features,
                    positions={self.agents[j]: float(positions[j]) for j in range(len(self.agents))},
                )
                dim.history.add_update(var, (neg_pole, pos_pole), self.total_frames)
                
                new_dimensions.append(dim)
                result['new_dimensions'].append(dim_name)
        
        self.dimensions = new_dimensions
        result['dimensions_after'] = len(self.dimensions)
        result['variance_explained'] = cumulative_var
        result['unexplained_variance'] = 1.0 - cumulative_var
        
        self.variance_explained_history.append(cumulative_var)
        
        return result
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get a report on dimension stability."""
        report = {
            'learning_cycles': self.learning_cycles,
            'total_frames': self.total_frames,
            'dimensions': [],
        }
        
        for dim in self.dimensions:
            stability = dim.history.compute_stability()
            report['dimensions'].append({
                'name': dim.name,
                'variance': dim.variance_explained,
                'poles': (dim.negative_pole, dim.positive_pole),
                'stability': stability,
                'n_updates': len(dim.history.updates),
            })
        
        return report
    
    def analyze(self, concept: str) -> Dict[str, Any]:
        """Analyze a concept across all dimensions."""
        concept_lower = concept.lower()
        
        result = {
            'concept': concept,
            'found': concept_lower in self.agents,
            'dimensions': {},
            'similar': [],
            'opposite': None,
        }
        
        if not result['found']:
            return result
        
        for dim in self.dimensions:
            pos = dim.positions.get(concept_lower, 0)
            if pos > 0.15:
                classification = 'positive'
                pole = dim.positive_pole
            elif pos < -0.15:
                classification = 'negative'
                pole = dim.negative_pole
            else:
                classification = 'neutral'
                pole = 'center'
            
            result['dimensions'][dim.name] = {
                'position': pos,
                'class': classification,
                'pole': pole,
            }
        
        # Find similar
        result['similar'] = self._find_similar(concept_lower, k=5)
        result['opposite'] = self._find_opposite(concept_lower)
        
        return result
    
    def _find_similar(self, concept: str, k: int = 5) -> List[Tuple[str, float]]:
        if concept not in self.agents:
            return []
        
        pos = np.array([dim.positions.get(concept, 0) for dim in self.dimensions])
        
        similarities = []
        for other in self.agents:
            if other != concept:
                other_pos = np.array([dim.positions.get(other, 0) for dim in self.dimensions])
                dist = np.linalg.norm(pos - other_pos)
                similarities.append((other, dist))
        
        return sorted(similarities, key=lambda x: x[1])[:k]
    
    def _find_opposite(self, concept: str) -> Optional[Tuple[str, float]]:
        if concept not in self.agents:
            return None
        
        pos = np.array([dim.positions.get(concept, 0) for dim in self.dimensions])
        
        max_dist = 0
        opposite = None
        
        for other in self.agents:
            if other != concept:
                other_pos = np.array([dim.positions.get(other, 0) for dim in self.dimensions])
                dist = np.linalg.norm(pos - other_pos)
                if dist > max_dist:
                    max_dist = dist
                    opposite = other
        
        return (opposite, max_dist) if opposite else None
    
    def save_state(self, path: str):
        """Save the current learning state."""
        state = {
            'learning_cycles': self.learning_cycles,
            'total_frames': self.total_frames,
            'variance_explained_history': self.variance_explained_history,
            'dimensions': [
                {
                    'name': d.name,
                    'variance': d.variance_explained,
                    'negative_pole': d.negative_pole,
                    'positive_pole': d.positive_pole,
                    'negative_features': d.negative_features,
                    'positive_features': d.positive_features,
                    'history_updates': d.history.updates,
                }
                for d in self.dimensions
            ],
            'agents': self.agents,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str):
        """Load a saved learning state."""
        with open(path) as f:
            state = json.load(f)
        
        self.learning_cycles = state['learning_cycles']
        self.variance_explained_history = state['variance_explained_history']
        # Note: Full state restoration would require more work


def run_incremental_learning_experiment():
    """
    Demonstrate incremental learning by feeding data in batches.
    """
    print("=" * 70)
    print("CONTINUOUS LEARNING EXPERIMENT")
    print("=" * 70)
    
    # Load corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        return None
    
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    frames = corpus['frames']
    print(f"Total frames available: {len(frames)}")
    
    # Create continuous learning system
    system = ContinuousLearningSystem(
        min_dimension_variance=0.025,
        max_dimensions=15,
    )
    
    # Feed data in batches and learn incrementally
    batch_size = 100
    n_batches = (len(frames) + batch_size - 1) // batch_size
    
    print(f"\nFeeding data in {n_batches} batches of ~{batch_size} frames each")
    print()
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(frames))
        batch = frames[start:end]
        
        # Ingest batch
        system.ingest_batch(batch)
        
        # Learn
        result = system.learn()
        
        print(f"Batch {i+1}/{n_batches}: {result['total_frames']} frames, "
              f"{result['n_agents']} agents, {result['dimensions_after']} dimensions, "
              f"{result['variance_explained']*100:.1f}% variance explained")
        
        if result['new_dimensions']:
            print(f"  NEW dimensions: {result['new_dimensions']}")
    
    # Final stability report
    print()
    print("=" * 70)
    print("STABILITY REPORT")
    print("=" * 70)
    
    report = system.get_stability_report()
    print(f"\nAfter {report['learning_cycles']} learning cycles with {report['total_frames']} frames:")
    print()
    
    for dim_info in report['dimensions']:
        stability_bar = "█" * int(dim_info['stability'] * 10)
        print(f"  {dim_info['name']}: {dim_info['variance']*100:.1f}% variance")
        print(f"    Poles: {dim_info['poles'][0]} <---> {dim_info['poles'][1]}")
        print(f"    Stability: {stability_bar} ({dim_info['stability']:.2f})")
        print(f"    Updates: {dim_info['n_updates']}")
        print()
    
    # Test some concepts
    print("=" * 70)
    print("CONCEPT ANALYSIS")
    print("=" * 70)
    
    test_concepts = ['holmes', 'watson', 'villain', 'hero', 'child', 'sage', 'storm']
    
    for concept in test_concepts:
        analysis = system.analyze(concept)
        if analysis['found']:
            print(f"\n{concept.upper()}:")
            for dim_name, info in analysis['dimensions'].items():
                if info['class'] != 'neutral':
                    print(f"  {dim_name}: {info['class']} (toward {info['pole']})")
            if analysis['similar']:
                print(f"  Similar: {[s[0] for s in analysis['similar'][:3]]}")
    
    # Save state
    state_path = Path(__file__).parent / "continuous_learning_state.json"
    system.save_state(str(state_path))
    print(f"\nState saved to: {state_path}")
    
    return system


def simulate_streaming_data():
    """
    Simulate streaming data to show continuous learning in action.
    """
    print("=" * 70)
    print("STREAMING DATA SIMULATION")
    print("=" * 70)
    
    system = ContinuousLearningSystem()
    
    # Simulate streaming frames
    simulated_frames = [
        {"text": "Holmes investigates the mysterious crime", "agent": "holmes"},
        {"text": "Watson assists Holmes with the case", "agent": "watson"},
        {"text": "The villain schemes in the shadows", "agent": "villain"},
        {"text": "Holmes deduces the criminal's identity", "agent": "holmes"},
        {"text": "Watson documents the investigation", "agent": "watson"},
        {"text": "The hero rescues the innocent victim", "agent": "hero"},
        {"text": "The villain threatens the witnesses", "agent": "villain"},
        {"text": "Holmes analyzes the evidence carefully", "agent": "holmes"},
        {"text": "The child plays in the garden happily", "agent": "child"},
        {"text": "The sage advises the young prince", "agent": "sage"},
        {"text": "Watson supports Holmes through danger", "agent": "watson"},
        {"text": "The storm rages across the countryside", "agent": "storm"},
        {"text": "Holmes solves the mystery brilliantly", "agent": "holmes"},
        {"text": "The villain escapes into the night", "agent": "villain"},
        {"text": "The hero confronts the evil villain", "agent": "hero"},
    ]
    
    print("\nSimulating streaming data...")
    print()
    
    for i, frame in enumerate(simulated_frames):
        system.ingest_frame(frame)
        
        # Learn every 5 frames
        if (i + 1) % 5 == 0:
            result = system.learn()
            print(f"After {i+1} frames: {result['dimensions_after']} dimensions, "
                  f"{result['variance_explained']*100:.1f}% variance")
    
    print("\nFinal dimensions:")
    for dim in system.dimensions:
        print(f"  {dim.name}: {dim.negative_pole} <---> {dim.positive_pole} ({dim.variance_explained*100:.1f}%)")
    
    return system


if __name__ == "__main__":
    # Run the main incremental learning experiment
    system = run_incremental_learning_experiment()
    
    print()
    print("=" * 70)
    print("STREAMING SIMULATION")
    print("=" * 70)
    
    # Also show streaming simulation
    simulate_streaming_data()
