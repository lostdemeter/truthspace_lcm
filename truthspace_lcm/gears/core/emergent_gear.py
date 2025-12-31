"""
Emergent Gear System

Gears that discover and build themselves from data patterns.
Each gear corresponds to a discovered dimension.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
from collections import defaultdict
import re
import json


@dataclass
class DiscoveredDimension:
    """A dimension discovered from data patterns."""
    index: int
    name: str
    variance_explained: float
    negative_pole: str
    positive_pole: str
    negative_features: List[str]
    positive_features: List[str]
    positions: Dict[str, float]
    balance_point: float = 0.0
    
    def get_position(self, concept: str) -> float:
        """Get a concept's position on this dimension."""
        return self.positions.get(concept.lower(), self.balance_point)
    
    def classify(self, concept: str) -> str:
        """Classify a concept as positive, negative, or neutral on this dimension."""
        pos = self.get_position(concept)
        if pos > self.balance_point + 0.1:
            return 'positive'
        elif pos < self.balance_point - 0.1:
            return 'negative'
        return 'neutral'


class EmergentGear:
    """
    A gear that emerged from a discovered dimension.
    
    Each gear transforms state along its dimension axis.
    """
    
    def __init__(self, dimension: DiscoveredDimension, ratio: float = 1.0):
        self.dimension = dimension
        self.ratio = ratio
        self.enabled = True
        self.name = f"EmergentGear_{dimension.name}"
    
    def forward(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform state along this dimension."""
        if not self.enabled:
            return state
        
        concept = state.get('concept', state.get('entity', ''))
        if not concept:
            return state
        
        # Get position on this dimension
        position = self.dimension.get_position(concept)
        classification = self.dimension.classify(concept)
        
        # Add dimension info to state
        state[f'dim_{self.dimension.name}_position'] = position
        state[f'dim_{self.dimension.name}_class'] = classification
        state[f'dim_{self.dimension.name}_pole'] = (
            self.dimension.positive_pole if position > self.dimension.balance_point 
            else self.dimension.negative_pole
        )
        
        # Apply ratio-based transformation
        if self.ratio != 1.0:
            # Shift position toward balance point or away
            adjusted = self.dimension.balance_point + (position - self.dimension.balance_point) * self.ratio
            state[f'dim_{self.dimension.name}_adjusted'] = adjusted
        
        return state
    
    def __repr__(self):
        return f"EmergentGear({self.dimension.name}, {self.dimension.negative_pole}↔{self.dimension.positive_pole})"


class DimensionDiscoverer:
    """
    Discovers dimensions from behavioral patterns in data.
    
    This is the core of the emergent system - it analyzes data
    to find the natural axes of variation.
    """
    
    def __init__(self):
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.dimensions: List[DiscoveredDimension] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self.agents: List[str] = []
        self.features: List[str] = []
    
    def ingest_frames(self, frames: List[Dict]):
        """Extract behavioral patterns from frames."""
        for frame in frames:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not agent or not text:
                continue
            
            # Extract action (word after agent name)
            words = text.lower().split()
            for i, w in enumerate(words):
                if agent in w and i + 1 < len(words):
                    verb = re.sub(r'[^a-z]', '', words[i + 1])
                    if len(verb) > 2:
                        self.agent_actions[agent][verb] += 1
                    break
    
    def build_feature_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Build feature matrix from behavioral patterns."""
        self.agents = list(self.agent_actions.keys())
        
        # Get all actions as features
        all_actions = set()
        for actions in self.agent_actions.values():
            all_actions.update(actions.keys())
        self.features = sorted(all_actions)
        
        # Build matrix
        n_agents = len(self.agents)
        n_features = len(self.features)
        
        X = np.zeros((n_agents, n_features))
        for i, agent in enumerate(self.agents):
            actions = self.agent_actions[agent]
            total = sum(actions.values())
            if total > 0:
                for j, action in enumerate(self.features):
                    X[i, j] = actions.get(action, 0) / total
        
        self.feature_matrix = X
        return X, self.agents, self.features
    
    def discover(self, n_dims: Optional[int] = None, 
                 variance_threshold: float = 0.80) -> List[DiscoveredDimension]:
        """
        Discover dimensions using SVD.
        
        Args:
            n_dims: Number of dimensions to discover (None = auto)
            variance_threshold: Cumulative variance threshold for auto
            
        Returns:
            List of discovered dimensions
        """
        if self.feature_matrix is None:
            self.build_feature_matrix()
        
        X = self.feature_matrix
        if X.shape[0] < 2:
            return []
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance explained
        total_var = np.sum(S ** 2)
        var_ratios = (S ** 2) / total_var
        cumulative = np.cumsum(var_ratios)
        
        # Determine number of dimensions
        if n_dims is None:
            n_dims = np.searchsorted(cumulative, variance_threshold) + 1
            n_dims = max(1, min(n_dims, 10, len(S)))
        
        # Create dimensions
        self.dimensions = []
        for i in range(n_dims):
            positions = U[:, i]
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            
            # Get top features
            feature_weights = Vt[i]
            neg_features = [self.features[j] for j in np.argsort(feature_weights)[:5]]
            pos_features = [self.features[j] for j in np.argsort(feature_weights)[-5:]]
            
            dim = DiscoveredDimension(
                index=i,
                name=f"Dim{i+1}",
                variance_explained=float(var_ratios[i]),
                negative_pole=self.agents[min_idx],
                positive_pole=self.agents[max_idx],
                negative_features=neg_features,
                positive_features=pos_features,
                positions={self.agents[j]: float(positions[j]) for j in range(len(self.agents))},
                balance_point=float(np.median(positions)),
            )
            self.dimensions.append(dim)
        
        return self.dimensions
    
    def get_dimension(self, index: int) -> Optional[DiscoveredDimension]:
        """Get a discovered dimension by index."""
        if 0 <= index < len(self.dimensions):
            return self.dimensions[index]
        return None


class EmergentGearChain:
    """
    A gear chain that builds itself from discovered dimensions.
    
    Each discovered dimension becomes a gear in the chain.
    """
    
    def __init__(self, discoverer: Optional[DimensionDiscoverer] = None):
        self.discoverer = discoverer
        self.gears: List[EmergentGear] = []
        
        if discoverer and discoverer.dimensions:
            self._build_gears()
    
    def _build_gears(self):
        """Build gears from discovered dimensions."""
        self.gears = []
        for dim in self.discoverer.dimensions:
            gear = EmergentGear(dim)
            self.gears.append(gear)
    
    def add_gear(self, dimension: DiscoveredDimension, ratio: float = 1.0):
        """Add a gear for a dimension."""
        gear = EmergentGear(dimension, ratio)
        self.gears.append(gear)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state through all gears."""
        for gear in self.gears:
            state = gear.forward(state)
        return state
    
    def analyze(self, concept: str) -> Dict[str, Any]:
        """Analyze a concept through all dimensions."""
        state = {'concept': concept}
        state = self.process(state)
        
        analysis = {
            'concept': concept,
            'dimensions': {},
        }
        
        for gear in self.gears:
            dim_name = gear.dimension.name
            analysis['dimensions'][dim_name] = {
                'position': state.get(f'dim_{dim_name}_position', 0),
                'class': state.get(f'dim_{dim_name}_class', 'unknown'),
                'pole': state.get(f'dim_{dim_name}_pole', 'unknown'),
            }
        
        return analysis
    
    def find_similar(self, concept: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar concepts based on all dimensions."""
        if not self.discoverer:
            return []
        
        # Get concept's position vector
        positions = []
        for dim in self.discoverer.dimensions:
            positions.append(dim.get_position(concept))
        
        if not positions:
            return []
        
        concept_vec = np.array(positions)
        
        # Compare to all other concepts
        similarities = []
        for other in self.discoverer.agents:
            if other.lower() != concept.lower():
                other_positions = [dim.get_position(other) for dim in self.discoverer.dimensions]
                other_vec = np.array(other_positions)
                dist = np.linalg.norm(concept_vec - other_vec)
                similarities.append((other, dist))
        
        return sorted(similarities, key=lambda x: x[1])[:k]
    
    def __repr__(self):
        return f"EmergentGearChain({len(self.gears)} gears: {[g.dimension.name for g in self.gears]})"


def create_emergent_chain_from_corpus(corpus_path: str, n_dims: int = 4) -> EmergentGearChain:
    """
    Create an emergent gear chain from a corpus file.
    
    This is the main entry point for using the emergent system.
    """
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    discoverer = DimensionDiscoverer()
    discoverer.ingest_frames(corpus['frames'])
    discoverer.discover(n_dims=n_dims)
    
    chain = EmergentGearChain(discoverer)
    return chain
