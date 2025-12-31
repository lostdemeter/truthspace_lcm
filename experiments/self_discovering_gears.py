#!/usr/bin/env python3
"""
Self-Discovering Gear System

This system discovers its own dimensions and builds gears automatically.
It does NOT use predefined gears like RoleGear, TenseGear, etc.
Instead, it discovers what dimensions exist in the data and creates
gears for each discovered dimension.

The goal: Compare what the system discovers vs what we intentionally designed.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import re
from dataclasses import dataclass, field


@dataclass
class DiscoveredDimension:
    """A dimension discovered from data."""
    index: int
    name: str
    variance_explained: float
    negative_pole: str
    positive_pole: str
    negative_features: List[str]  # Verbs/behaviors at negative end
    positive_features: List[str]  # Verbs/behaviors at positive end
    positions: Dict[str, float]   # Agent -> position on this dimension
    
    def describe(self) -> str:
        """Generate a human-readable description of this dimension."""
        neg_verbs = ', '.join(self.negative_features[:3])
        pos_verbs = ', '.join(self.positive_features[:3])
        return f"{self.name}: {self.negative_pole} ({neg_verbs}) <---> {self.positive_pole} ({pos_verbs})"


@dataclass 
class EmergentGear:
    """A gear that emerged from a discovered dimension."""
    dimension: DiscoveredDimension
    name: str
    enabled: bool = True
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state through this gear."""
        if not self.enabled:
            return state
        
        # Get concept from state
        concept = state.get('concept', state.get('agent', state.get('entity', '')))
        if not concept:
            return state
        
        concept_lower = concept.lower()
        
        # Get position on this dimension
        position = self.dimension.positions.get(concept_lower, 0.0)
        
        # Classify
        if position > 0.15:
            classification = 'positive'
            pole = self.dimension.positive_pole
        elif position < -0.15:
            classification = 'negative'
            pole = self.dimension.negative_pole
        else:
            classification = 'neutral'
            pole = 'center'
        
        # Add to state
        state[f'{self.name}_position'] = position
        state[f'{self.name}_class'] = classification
        state[f'{self.name}_pole'] = pole
        
        return state
    
    def __repr__(self):
        return f"EmergentGear({self.name}: {self.dimension.negative_pole}↔{self.dimension.positive_pole})"


class SelfDiscoveringGearSystem:
    """
    A gear system that discovers and builds itself from data.
    
    This is the core innovation: instead of predefined gears,
    the system discovers what dimensions exist and creates gears for them.
    """
    
    def __init__(self):
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.dimensions: List[DiscoveredDimension] = []
        self.gears: List[EmergentGear] = []
        self.agents: List[str] = []
        self.features: List[str] = []
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        
        # Discovery parameters
        self.min_dimension_variance = 0.025  # Minimum variance to keep a dimension
        self.max_dimensions = 15
        self.variance_threshold = 0.85  # Stop when this much variance explained
    
    def ingest_corpus(self, corpus_path: str):
        """Ingest corpus and extract behavioral patterns."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        frames = corpus['frames']
        print(f"Ingesting {len(frames)} frames...")
        
        for frame in frames:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not agent or not text or len(agent) < 2:
                continue
            
            # Extract verbs (more sophisticated extraction)
            words = text.lower().split()
            
            # Find verbs after the agent or at sentence start
            for i, word in enumerate(words):
                # Clean word
                word_clean = re.sub(r'[^a-z]', '', word)
                if len(word_clean) < 3:
                    continue
                
                # Check if this looks like a verb (heuristic)
                verb_endings = ['ed', 'ing', 'es', 's', 'ly']
                verb_patterns = ['investigat', 'analyz', 'observ', 'deduc', 'solv',
                                'command', 'lead', 'follow', 'assist', 'support',
                                'scheme', 'plot', 'manipulat', 'threaten', 'attack',
                                'explor', 'question', 'wonder', 'learn', 'play',
                                'advis', 'teach', 'guid', 'remember', 'reflect',
                                'serv', 'obey', 'wait', 'help', 'protect',
                                'destroy', 'rag', 'flood', 'overwhelm', 'spread']
                
                is_verb = any(word_clean.endswith(e) for e in verb_endings)
                is_verb = is_verb or any(word_clean.startswith(p) for p in verb_patterns)
                
                if is_verb:
                    self.agent_actions[agent][word_clean] += 1
        
        # Filter out noise agents
        min_actions = 3
        self.agent_actions = {
            a: v for a, v in self.agent_actions.items() 
            if sum(v.values()) >= min_actions and len(a) > 2
        }
        
        print(f"Found {len(self.agent_actions)} agents with sufficient actions")
        
        # Show top agents
        agent_counts = [(a, sum(v.values())) for a, v in self.agent_actions.items()]
        agent_counts.sort(key=lambda x: -x[1])
        print(f"Top agents: {agent_counts[:10]}")
    
    def discover_dimensions(self) -> List[DiscoveredDimension]:
        """Discover dimensions using SVD on behavioral patterns."""
        
        # Build feature matrix
        self.agents = list(self.agent_actions.keys())
        all_actions = set()
        for actions in self.agent_actions.values():
            all_actions.update(actions.keys())
        self.features = sorted(all_actions)
        
        n_agents = len(self.agents)
        n_features = len(self.features)
        
        print(f"\nBuilding feature matrix: {n_agents} agents × {n_features} features")
        
        # Build normalized matrix
        X = np.zeros((n_agents, n_features))
        for i, agent in enumerate(self.agents):
            actions = self.agent_actions[agent]
            total = sum(actions.values())
            if total > 0:
                for j, action in enumerate(self.features):
                    X[i, j] = actions.get(action, 0) / total
        
        # Center
        X_centered = X - X.mean(axis=0)
        
        # SVD
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance explained
        total_var = np.sum(self.S ** 2)
        var_ratios = (self.S ** 2) / total_var
        cumulative = np.cumsum(var_ratios)
        
        print(f"\n{'='*70}")
        print("DIMENSION DISCOVERY")
        print(f"{'='*70}")
        
        # Discover dimensions
        self.dimensions = []
        
        for i in range(min(len(self.S), self.max_dimensions)):
            var = var_ratios[i]
            cum_var = cumulative[i]
            
            # Check stopping criteria
            if var < self.min_dimension_variance:
                print(f"\nStopping: Dimension {i+1} variance ({var*100:.1f}%) < minimum")
                break
            
            # Get positions and poles
            positions = self.U[:, i]
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            
            # Get top features
            feature_weights = self.Vt[i]
            neg_indices = np.argsort(feature_weights)[:5]
            pos_indices = np.argsort(feature_weights)[-5:]
            neg_features = [self.features[j] for j in neg_indices]
            pos_features = [self.features[j] for j in pos_indices]
            
            # Create dimension with auto-generated name
            dim = DiscoveredDimension(
                index=i,
                name=f"Dim{i+1}",
                variance_explained=float(var),
                negative_pole=self.agents[min_idx],
                positive_pole=self.agents[max_idx],
                negative_features=neg_features,
                positive_features=pos_features,
                positions={self.agents[j]: float(positions[j]) for j in range(n_agents)},
            )
            
            self.dimensions.append(dim)
            
            print(f"\n  {dim.name}: {var*100:.1f}% variance (cumulative: {cum_var*100:.1f}%)")
            print(f"    {dim.negative_pole} <---> {dim.positive_pole}")
            print(f"    - : {neg_features}")
            print(f"    + : {pos_features}")
            
            if cum_var >= self.variance_threshold:
                print(f"\nStopping: Cumulative variance ({cum_var*100:.1f}%) >= threshold")
                break
        
        print(f"\n  TOTAL: {len(self.dimensions)} dimensions discovered")
        return self.dimensions
    
    def build_gears(self) -> List[EmergentGear]:
        """Build gears from discovered dimensions."""
        
        print(f"\n{'='*70}")
        print("BUILDING EMERGENT GEARS")
        print(f"{'='*70}")
        
        self.gears = []
        
        for dim in self.dimensions:
            # Try to infer a meaningful name from the features
            name = self._infer_dimension_name(dim)
            dim.name = name
            
            gear = EmergentGear(dimension=dim, name=name)
            self.gears.append(gear)
            
            print(f"\n  Created: {gear}")
            print(f"    Variance: {dim.variance_explained*100:.1f}%")
        
        print(f"\n  TOTAL: {len(self.gears)} gears built")
        return self.gears
    
    def _infer_dimension_name(self, dim: DiscoveredDimension) -> str:
        """Try to infer a meaningful name for a dimension from its features."""
        
        # Check for known patterns
        neg = set(dim.negative_features)
        pos = set(dim.positive_features)
        
        # Agency patterns
        high_agency = {'commands', 'leads', 'decides', 'investigates', 'solves', 
                       'confronts', 'attacks', 'controls', 'dominates', 'rules'}
        low_agency = {'follows', 'obeys', 'waits', 'assists', 'serves', 
                      'watches', 'listens', 'supports', 'helps'}
        
        if pos & high_agency or neg & low_agency:
            return "Agency"
        if neg & high_agency or pos & low_agency:
            return "Agency"  # Inverted
        
        # Morality patterns
        good = {'helps', 'protects', 'saves', 'heals', 'supports', 'loves', 'nurtures'}
        evil = {'schemes', 'plots', 'betrays', 'destroys', 'manipulates', 'threatens'}
        
        if (pos & good) or (neg & evil):
            return "Morality"
        if (neg & good) or (pos & evil):
            return "Morality"
        
        # Age patterns
        young = {'plays', 'learns', 'grows', 'explores', 'wonders', 'asks', 'imagines'}
        old = {'remembers', 'advises', 'teaches', 'reflects', 'guides', 'rests'}
        
        if (pos & old) or (neg & young):
            return "Maturity"
        if (neg & old) or (pos & young):
            return "Maturity"
        
        # Animacy patterns
        abstract = {'spreads', 'persists', 'transforms', 'exists', 'flows', 'passes'}
        animate = {'thinks', 'feels', 'decides', 'wants', 'hopes', 'fears'}
        
        if (pos & animate) or (neg & abstract):
            return "Animacy"
        if (neg & animate) or (pos & abstract):
            return "Animacy"
        
        # Power patterns
        powerful = {'commands', 'rules', 'conquers', 'dominates', 'controls'}
        powerless = {'begs', 'pleads', 'cowers', 'suffers', 'endures'}
        
        if (pos & powerful) or (neg & powerless):
            return "Power"
        if (neg & powerful) or (pos & powerless):
            return "Power"
        
        # Default: use pole names
        return f"{dim.negative_pole[:4]}_{dim.positive_pole[:4]}"
    
    def process(self, concept: str) -> Dict[str, Any]:
        """Process a concept through all gears."""
        state = {'concept': concept}
        
        for gear in self.gears:
            state = gear.process(state)
        
        return state
    
    def analyze(self, concept: str) -> Dict[str, Any]:
        """Analyze a concept across all dimensions."""
        state = self.process(concept)
        
        analysis = {
            'concept': concept,
            'dimensions': {},
            'similar': [],
            'opposite': None,
        }
        
        for gear in self.gears:
            dim_name = gear.name
            analysis['dimensions'][dim_name] = {
                'position': state.get(f'{dim_name}_position', 0),
                'class': state.get(f'{dim_name}_class', 'unknown'),
                'pole': state.get(f'{dim_name}_pole', 'unknown'),
            }
        
        # Find similar and opposite
        analysis['similar'] = self.find_similar(concept, k=5)
        analysis['opposite'] = self.find_opposite(concept)
        
        return analysis
    
    def find_similar(self, concept: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar concepts."""
        concept_lower = concept.lower()
        if concept_lower not in self.agents:
            return []
        
        pos = np.array([dim.positions.get(concept_lower, 0) for dim in self.dimensions])
        
        similarities = []
        for other in self.agents:
            if other != concept_lower:
                other_pos = np.array([dim.positions.get(other, 0) for dim in self.dimensions])
                dist = np.linalg.norm(pos - other_pos)
                similarities.append((other, dist))
        
        return sorted(similarities, key=lambda x: x[1])[:k]
    
    def find_opposite(self, concept: str) -> Optional[Tuple[str, float]]:
        """Find the most opposite concept."""
        concept_lower = concept.lower()
        if concept_lower not in self.agents:
            return None
        
        pos = np.array([dim.positions.get(concept_lower, 0) for dim in self.dimensions])
        
        max_dist = 0
        opposite = None
        
        for other in self.agents:
            if other != concept_lower:
                other_pos = np.array([dim.positions.get(other, 0) for dim in self.dimensions])
                dist = np.linalg.norm(pos - other_pos)
                if dist > max_dist:
                    max_dist = dist
                    opposite = other
        
        return (opposite, max_dist) if opposite else None
    
    def compare_with_designed_gears(self):
        """Compare discovered gears with intentionally designed ones."""
        
        print(f"\n{'='*70}")
        print("COMPARISON: EMERGENT vs DESIGNED GEARS")
        print(f"{'='*70}")
        
        # Our intentionally designed gears
        designed_gears = [
            ("RoleGear", "Maps entities to semantic roles (agent, patient, etc.)"),
            ("TenseGear", "Handles temporal aspects (past, present, future)"),
            ("MoodGear", "Manages modality (certainty, possibility, necessity)"),
            ("VoiceGear", "Handles active/passive voice transformations"),
            ("AspectGear", "Manages aspect (perfective, imperfective, etc.)"),
            ("PolarityGear", "Handles negation and affirmation"),
        ]
        
        print("\n--- DESIGNED GEARS (Intentional) ---")
        for name, desc in designed_gears:
            print(f"  • {name}: {desc}")
        
        print("\n--- EMERGENT GEARS (Discovered) ---")
        for gear in self.gears:
            dim = gear.dimension
            print(f"  • {gear.name}: {dim.negative_pole} <---> {dim.positive_pole}")
            print(f"      Features: {dim.negative_features[:3]} vs {dim.positive_features[:3]}")
            print(f"      Variance: {dim.variance_explained*100:.1f}%")
        
        print("\n--- ANALYSIS ---")
        print("\nWhat the system discovered vs what we designed:")
        print()
        
        # Check for overlaps
        emergent_names = {g.name.lower() for g in self.gears}
        
        overlaps = []
        unique_designed = []
        unique_emergent = []
        
        # Check designed gears
        for name, _ in designed_gears:
            name_lower = name.lower().replace('gear', '')
            found = False
            for ename in emergent_names:
                if name_lower in ename or ename in name_lower:
                    overlaps.append((name, ename))
                    found = True
                    break
            if not found:
                unique_designed.append(name)
        
        # Check emergent gears
        designed_names = {n.lower().replace('gear', '') for n, _ in designed_gears}
        for gear in self.gears:
            ename = gear.name.lower()
            found = False
            for dname in designed_names:
                if dname in ename or ename in dname:
                    found = True
                    break
            if not found:
                unique_emergent.append(gear.name)
        
        print("Potential overlaps (similar concepts):")
        if overlaps:
            for designed, emergent in overlaps:
                print(f"  • {designed} ≈ {emergent}")
        else:
            print("  (none found)")
        
        print("\nUnique to DESIGNED (not discovered):")
        for name in unique_designed:
            print(f"  • {name}")
        
        print("\nUnique to EMERGENT (not designed):")
        for name in unique_emergent:
            print(f"  • {name}")
        
        print("\n--- KEY INSIGHT ---")
        print("""
The emergent system discovers dimensions based on BEHAVIORAL VARIANCE in the data.
The designed system was based on LINGUISTIC THEORY (roles, tense, mood, etc.).

Key differences:
1. Emergent dimensions are data-driven (what varies most in behavior)
2. Designed gears are theory-driven (what linguists say matters)
3. Emergent dimensions may capture things we didn't think to design for
4. Designed gears may capture things that don't vary much in our data

Neither is "better" - they serve different purposes:
- Emergent: Discover what's actually in the data
- Designed: Encode what we know should matter
""")


def main():
    print("=" * 70)
    print("SELF-DISCOVERING GEAR SYSTEM")
    print("=" * 70)
    
    # Load LLM-generated corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run llm_live_corpus_generator.py first!")
        return None
    
    # Create self-discovering system
    system = SelfDiscoveringGearSystem()
    system.ingest_corpus(str(corpus_path))
    system.discover_dimensions()
    system.build_gears()
    
    # Analyze some concepts
    print(f"\n{'='*70}")
    print("CONCEPT ANALYSIS")
    print(f"{'='*70}")
    
    test_concepts = ['holmes', 'watson', 'moriarty', 'alice', 'queen', 'king', 
                     'child', 'sage', 'servant', 'villain', 'hero', 'robot', 'storm']
    
    for concept in test_concepts:
        analysis = system.analyze(concept)
        if analysis['dimensions']:
            print(f"\n{concept.upper()}:")
            for dim_name, info in analysis['dimensions'].items():
                if info['class'] != 'neutral':
                    print(f"  {dim_name}: {info['class']} (toward {info['pole']})")
            if analysis['similar']:
                print(f"  Similar: {[s[0] for s in analysis['similar'][:3]]}")
            if analysis['opposite']:
                print(f"  Opposite: {analysis['opposite'][0]}")
    
    # Compare with designed gears
    system.compare_with_designed_gears()
    
    return system


if __name__ == "__main__":
    system = main()
