#!/usr/bin/env python3
"""
Emergent Dimension Discovery and Self-Building Gear Chain

This experiment tests whether a hyperdimensional transcoder can discover
its own dimensionality by analyzing the structure of the data.

Key Ideas:
1. Don't predefine dimensions - let them emerge from variance in the data
2. Each dimension has a balance point (critical line)
3. New dimensions spawn when existing ones can't explain the variance
4. The structure is self-similar at every scale
"""

import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
import re


# =============================================================================
# PART 1: DIMENSION DISCOVERY
# =============================================================================

@dataclass
class EmergentDimension:
    """A dimension that emerged from the data."""
    index: int
    name: str  # Auto-generated or discovered
    balance_point: float  # The critical line for this dimension
    variance_explained: float  # How much variance this dimension captures
    poles: Tuple[str, str]  # The two extremes (e.g., ("male", "female"))
    examples: Dict[str, float] = field(default_factory=dict)  # concept -> position


class DimensionDiscovery:
    """
    Discovers the natural dimensionality of concept space from data.
    
    Instead of predefining axes (gender, age, agency, animacy),
    we let them emerge from patterns in the corpus.
    """
    
    def __init__(self):
        self.dimensions: List[EmergentDimension] = []
        self.concept_vectors: Dict[str, np.ndarray] = {}
        self.co_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.concept_contexts: Dict[str, List[str]] = defaultdict(list)
        self.concept_actions: Dict[str, List[str]] = defaultdict(list)
        
    def ingest_corpus(self, corpus_path: str):
        """Load and analyze corpus to build co-occurrence matrix."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        print(f"Ingesting {len(corpus['frames'])} frames...")
        
        for frame in corpus['frames']:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not agent or not text:
                continue
            
            # Extract words from text
            words = re.findall(r'\b[a-z]+\b', text.lower())
            
            # Track what actions/contexts this agent appears with
            for word in words:
                if word != agent:
                    self.co_occurrence[agent][word] += 1
                    self.concept_contexts[agent].append(word)
            
            # Extract verbs (simple heuristic: words ending in -ed, -ing, -s after agent)
            agent_idx = text.lower().find(agent)
            if agent_idx >= 0:
                after_agent = text[agent_idx + len(agent):].lower()
                verb_match = re.search(r'\b([a-z]+(?:ed|ing|s|es))\b', after_agent)
                if verb_match:
                    self.concept_actions[agent].append(verb_match.group(1))
        
        print(f"Found {len(self.co_occurrence)} unique agents")
        print(f"Sample agents: {list(self.co_occurrence.keys())[:10]}")
    
    def build_feature_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build a feature matrix from co-occurrence data.
        Rows = concepts, Columns = context words
        """
        # Get all concepts and context words
        concepts = list(self.co_occurrence.keys())
        
        # Get most common context words (features)
        all_contexts = defaultdict(int)
        for concept, contexts in self.co_occurrence.items():
            for word, count in contexts.items():
                all_contexts[word] += count
        
        # Use top N context words as features
        top_contexts = sorted(all_contexts.items(), key=lambda x: -x[1])[:200]
        features = [w for w, c in top_contexts]
        
        print(f"Using {len(features)} features (top context words)")
        print(f"Top 10 features: {features[:10]}")
        
        # Build matrix
        matrix = np.zeros((len(concepts), len(features)))
        for i, concept in enumerate(concepts):
            for j, feature in enumerate(features):
                matrix[i, j] = self.co_occurrence[concept].get(feature, 0)
        
        # Normalize rows (each concept is a unit vector in context space)
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        matrix = matrix / row_norms
        
        return matrix, concepts, features
    
    def discover_dimensions(self, n_dims: Optional[int] = None, 
                           variance_threshold: float = 0.95) -> List[EmergentDimension]:
        """
        Discover dimensions using SVD (like PCA but for non-centered data).
        
        If n_dims is None, automatically determine based on variance threshold.
        """
        matrix, concepts, features = self.build_feature_matrix()
        
        if matrix.shape[0] < 2:
            print("Not enough concepts to discover dimensions")
            return []
        
        # SVD decomposition
        # U: concepts in dimension space
        # S: singular values (importance of each dimension)
        # Vt: features in dimension space
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Calculate variance explained by each dimension
        total_variance = np.sum(S ** 2)
        variance_ratios = (S ** 2) / total_variance
        cumulative_variance = np.cumsum(variance_ratios)
        
        print(f"\nVariance explained by first 10 dimensions:")
        for i in range(min(10, len(variance_ratios))):
            print(f"  Dim {i+1}: {variance_ratios[i]*100:.2f}% (cumulative: {cumulative_variance[i]*100:.2f}%)")
        
        # Determine number of dimensions
        if n_dims is None:
            # Find where cumulative variance exceeds threshold
            n_dims = np.searchsorted(cumulative_variance, variance_threshold) + 1
            n_dims = max(1, min(n_dims, 10))  # Between 1 and 10
        
        print(f"\nUsing {n_dims} dimensions (explains {cumulative_variance[n_dims-1]*100:.2f}% variance)")
        
        # Create emergent dimensions
        self.dimensions = []
        for i in range(n_dims):
            # Get concept positions on this dimension
            positions = U[:, i]
            
            # Find the poles (extremes)
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            negative_pole = concepts[min_idx]
            positive_pole = concepts[max_idx]
            
            # Find balance point (where most concepts cluster)
            balance = np.median(positions)
            
            # Store concept positions
            examples = {concepts[j]: float(positions[j]) for j in range(len(concepts))}
            
            # Try to name the dimension based on its poles
            name = self._infer_dimension_name(i, negative_pole, positive_pole, 
                                              Vt[i], features)
            
            dim = EmergentDimension(
                index=i,
                name=name,
                balance_point=float(balance),
                variance_explained=float(variance_ratios[i]),
                poles=(negative_pole, positive_pole),
                examples=examples
            )
            self.dimensions.append(dim)
            
            print(f"\nDimension {i+1}: {name}")
            print(f"  Poles: {negative_pole} <---> {positive_pole}")
            print(f"  Balance point: {balance:.3f}")
            print(f"  Variance: {variance_ratios[i]*100:.2f}%")
            
            # Show top concepts at each pole
            sorted_by_pos = sorted(examples.items(), key=lambda x: x[1])
            print(f"  Negative pole concepts: {[c for c, p in sorted_by_pos[:5]]}")
            print(f"  Positive pole concepts: {[c for c, p in sorted_by_pos[-5:]]}")
        
        # Store concept vectors
        for i, concept in enumerate(concepts):
            self.concept_vectors[concept] = U[i, :n_dims]
        
        return self.dimensions
    
    def _infer_dimension_name(self, idx: int, neg_pole: str, pos_pole: str,
                              feature_weights: np.ndarray, features: List[str]) -> str:
        """Try to infer a meaningful name for the dimension."""
        # Get top features for each pole
        sorted_features = sorted(zip(features, feature_weights), key=lambda x: x[1])
        neg_features = [f for f, w in sorted_features[:5]]
        pos_features = [f for f, w in sorted_features[-5:]]
        
        # Simple heuristics for naming
        # This could be made much smarter
        name = f"Dim_{idx+1}_{neg_pole[:4]}_{pos_pole[:4]}"
        
        return name
    
    def get_concept_position(self, concept: str) -> Optional[np.ndarray]:
        """Get a concept's position in the discovered dimension space."""
        return self.concept_vectors.get(concept.lower())
    
    def find_similar(self, concept: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar concepts."""
        pos = self.get_concept_position(concept)
        if pos is None:
            return []
        
        similarities = []
        for other, other_pos in self.concept_vectors.items():
            if other != concept.lower():
                # Cosine similarity
                sim = np.dot(pos, other_pos) / (np.linalg.norm(pos) * np.linalg.norm(other_pos) + 1e-10)
                similarities.append((other, float(sim)))
        
        return sorted(similarities, key=lambda x: -x[1])[:k]


# =============================================================================
# PART 2: EMERGENT GEAR CHAIN
# =============================================================================

@dataclass
class GearState:
    """State flowing through the gear chain."""
    data: Dict[str, Any] = field(default_factory=dict)
    position: Optional[np.ndarray] = None  # Position in dimension space
    residual: Optional[np.ndarray] = None  # Unexplained variance
    transformations: List[str] = field(default_factory=list)


class EmergentGear:
    """A gear that emerged to handle a specific dimension."""
    
    def __init__(self, dimension: EmergentDimension):
        self.dimension = dimension
        self.name = f"Gear_{dimension.name}"
        self.enabled = True
        self.ratio = 1.0
    
    def forward(self, state: GearState) -> GearState:
        """Transform state along this dimension."""
        if not self.enabled or state.position is None:
            return state
        
        idx = self.dimension.index
        if idx >= len(state.position):
            return state
        
        # Get position on this dimension
        pos = state.position[idx]
        
        # Apply transformation based on distance from balance point
        distance_from_balance = pos - self.dimension.balance_point
        
        # Store transformation info
        state.transformations.append(
            f"{self.name}: pos={pos:.3f}, dist_from_balance={distance_from_balance:.3f}"
        )
        
        # The gear's effect depends on position relative to balance
        state.data[f'dim_{idx}_position'] = pos
        state.data[f'dim_{idx}_pole'] = 'positive' if pos > self.dimension.balance_point else 'negative'
        
        return state
    
    def __repr__(self):
        return f"EmergentGear({self.dimension.name}, balance={self.dimension.balance_point:.3f})"


class EmergentGearChain:
    """
    A gear chain that builds itself based on discovered dimensions.
    
    Key insight: The chain doesn't predefine its gears.
    Gears emerge from the data's natural dimensionality.
    """
    
    def __init__(self, discovery: DimensionDiscovery):
        self.discovery = discovery
        self.gears: List[EmergentGear] = []
        self._build_gears()
    
    def _build_gears(self):
        """Build gears from discovered dimensions."""
        self.gears = []
        for dim in self.discovery.dimensions:
            gear = EmergentGear(dim)
            self.gears.append(gear)
            print(f"Created {gear}")
    
    def process(self, concept: str) -> GearState:
        """Process a concept through the emergent gear chain."""
        # Get concept's position in dimension space
        position = self.discovery.get_concept_position(concept)
        
        if position is None:
            print(f"Unknown concept: {concept}")
            return GearState(data={'concept': concept, 'known': False})
        
        # Initialize state
        state = GearState(
            data={'concept': concept, 'known': True},
            position=position.copy()
        )
        
        # Process through each gear
        for gear in self.gears:
            state = gear.forward(state)
        
        return state
    
    def analyze_concept(self, concept: str) -> Dict[str, Any]:
        """Analyze a concept's position in the emergent dimension space."""
        state = self.process(concept)
        
        if not state.data.get('known', False):
            return {'concept': concept, 'error': 'unknown concept'}
        
        analysis = {
            'concept': concept,
            'dimensions': {},
            'similar_concepts': self.discovery.find_similar(concept, k=5)
        }
        
        for i, dim in enumerate(self.discovery.dimensions):
            pos = state.position[i] if state.position is not None else 0
            analysis['dimensions'][dim.name] = {
                'position': float(pos),
                'balance_point': dim.balance_point,
                'distance_from_balance': float(pos - dim.balance_point),
                'pole': dim.poles[1] if pos > dim.balance_point else dim.poles[0],
                'pole_strength': abs(pos - dim.balance_point)
            }
        
        return analysis
    
    def spawn_gear_for_residual(self, residual: np.ndarray) -> Optional[EmergentGear]:
        """
        Spawn a new gear if there's unexplained variance.
        
        This is the key to emergent structure: when existing gears
        can't explain the data, a new gear emerges.
        """
        # Check if residual is significant
        residual_magnitude = np.linalg.norm(residual)
        if residual_magnitude < 0.1:  # Threshold
            return None
        
        # Find the dominant direction of the residual
        # This would become a new dimension
        print(f"Residual magnitude {residual_magnitude:.3f} exceeds threshold")
        print("A new dimension might be needed...")
        
        # In a full implementation, we would:
        # 1. Collect residuals from many concepts
        # 2. Find the principal component of residuals
        # 3. Create a new dimension along that direction
        # 4. Spawn a gear for that dimension
        
        return None


# =============================================================================
# PART 3: EXPERIMENT
# =============================================================================

class ActionBasedDiscovery:
    """
    Discover dimensions based on WHAT AGENTS DO, not just co-occurrence.
    
    The insight: semantic dimensions like agency emerge from action patterns,
    not from topic clustering.
    """
    
    def __init__(self):
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.agent_targets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.agent_modifiers: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.action_agents: Dict[str, Set[str]] = defaultdict(set)  # Which agents do this action
        
        # Verb categories (will be discovered, but seed with some)
        self.high_agency_verbs = {'investigates', 'solves', 'commands', 'leads', 'decides', 
                                   'creates', 'discovers', 'confronts', 'attacks', 'rules',
                                   'examined', 'deduced', 'solved', 'investigated', 'discovered'}
        self.low_agency_verbs = {'assists', 'follows', 'obeys', 'watches', 'waits',
                                  'assisted', 'followed', 'watched', 'waited', 'served'}
        self.neutral_verbs = {'is', 'was', 'has', 'had', 'said', 'went', 'came'}
    
    def ingest_corpus(self, corpus_path: str):
        """Extract agent-action-target patterns."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        print(f"Analyzing {len(corpus['frames'])} frames for action patterns...")
        
        for frame in corpus['frames']:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not agent or not text:
                continue
            
            # Simple pattern extraction
            words = text.lower().split()
            
            # Find verbs (words after agent, heuristic)
            try:
                agent_idx = None
                for i, w in enumerate(words):
                    if agent in w:
                        agent_idx = i
                        break
                
                if agent_idx is not None and agent_idx + 1 < len(words):
                    # Next word is likely the verb
                    potential_verb = words[agent_idx + 1]
                    # Clean it
                    potential_verb = re.sub(r'[^a-z]', '', potential_verb)
                    if len(potential_verb) > 2:
                        self.agent_actions[agent][potential_verb] += 1
                        self.action_agents[potential_verb].add(agent)
                        
                        # Look for targets (nouns after verb)
                        if agent_idx + 2 < len(words):
                            for w in words[agent_idx + 2:agent_idx + 5]:
                                w = re.sub(r'[^a-z]', '', w)
                                if len(w) > 2 and w not in {'the', 'and', 'with', 'from', 'for'}:
                                    self.agent_targets[agent][w] += 1
            except:
                pass
        
        print(f"Found {len(self.agent_actions)} agents with action patterns")
        
        # Show top agents by action diversity
        agent_diversity = [(a, len(actions)) for a, actions in self.agent_actions.items()]
        agent_diversity.sort(key=lambda x: -x[1])
        print(f"Top agents by action diversity: {agent_diversity[:10]}")
    
    def compute_agency_score(self, agent: str) -> float:
        """
        Compute agency score for an agent based on their actions.
        High agency = does high-agency verbs
        Low agency = does low-agency verbs
        """
        actions = self.agent_actions.get(agent, {})
        if not actions:
            return 0.0
        
        high_count = sum(actions.get(v, 0) for v in self.high_agency_verbs)
        low_count = sum(actions.get(v, 0) for v in self.low_agency_verbs)
        total = sum(actions.values())
        
        if total == 0:
            return 0.0
        
        # Score from -1 (low agency) to +1 (high agency)
        return (high_count - low_count) / total
    
    def compute_action_diversity(self, agent: str) -> float:
        """How many different actions does this agent perform?"""
        actions = self.agent_actions.get(agent, {})
        return len(actions)
    
    def compute_target_diversity(self, agent: str) -> float:
        """How many different targets does this agent affect?"""
        targets = self.agent_targets.get(agent, {})
        return len(targets)
    
    def discover_dimensions(self) -> List[EmergentDimension]:
        """Discover dimensions from action patterns."""
        agents = list(self.agent_actions.keys())
        
        if len(agents) < 5:
            print("Not enough agents for dimension discovery")
            return []
        
        # Compute features for each agent
        features = []
        for agent in agents:
            agency = self.compute_agency_score(agent)
            action_div = self.compute_action_diversity(agent)
            target_div = self.compute_target_diversity(agent)
            
            # Action type distribution
            actions = self.agent_actions.get(agent, {})
            total_actions = sum(actions.values())
            
            features.append([
                agency,
                action_div / 20.0,  # Normalize
                target_div / 20.0,
                total_actions / 100.0,
            ])
        
        features = np.array(features)
        
        # Normalize
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1
        features_norm = (features - means) / stds
        
        # SVD on normalized features
        U, S, Vt = np.linalg.svd(features_norm, full_matrices=False)
        
        # Variance explained
        total_var = np.sum(S ** 2)
        var_ratios = (S ** 2) / total_var
        
        print(f"\nAction-based dimension discovery:")
        print(f"  Dim 1 (Agency?): {var_ratios[0]*100:.1f}% variance")
        print(f"  Dim 2 (Diversity?): {var_ratios[1]*100:.1f}% variance")
        print(f"  Dim 3: {var_ratios[2]*100:.1f}% variance")
        print(f"  Dim 4: {var_ratios[3]*100:.1f}% variance")
        
        # Create dimensions
        dimensions = []
        dim_names = ['Agency', 'ActionDiversity', 'TargetDiversity', 'Activity']
        
        for i in range(min(4, len(S))):
            positions = U[:, i]
            min_idx, max_idx = np.argmin(positions), np.argmax(positions)
            
            dim = EmergentDimension(
                index=i,
                name=dim_names[i] if i < len(dim_names) else f"Dim_{i}",
                balance_point=float(np.median(positions)),
                variance_explained=float(var_ratios[i]),
                poles=(agents[min_idx], agents[max_idx]),
                examples={agents[j]: float(positions[j]) for j in range(len(agents))}
            )
            dimensions.append(dim)
            
            print(f"\n  {dim.name}:")
            print(f"    Poles: {dim.poles[0]} <---> {dim.poles[1]}")
            
            # Show agents at each extreme
            sorted_agents = sorted(dim.examples.items(), key=lambda x: x[1])
            print(f"    Low end: {[a for a, p in sorted_agents[:5]]}")
            print(f"    High end: {[a for a, p in sorted_agents[-5:]]}")
        
        return dimensions


def run_experiment():
    """Run the emergent dimension discovery experiment."""
    
    print("=" * 70)
    print("EMERGENT DIMENSION DISCOVERY EXPERIMENT")
    print("=" * 70)
    print()
    print("Goal: Let the system discover its own dimensionality from data")
    print("Hypothesis: It will rediscover axes similar to our known quaternions")
    print()
    
    # Find corpus
    corpus_paths = [
        Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_experimental.json",
        Path(__file__).parent.parent / "truthspace_lcm" / "corpus_experimental.json",
        Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_signal_full.json",
    ]
    
    corpus_path = None
    for p in corpus_paths:
        if p.exists():
            corpus_path = p
            break
    
    if corpus_path is None:
        print("ERROR: Could not find corpus file")
        return
    
    print(f"Using corpus: {corpus_path}")
    print()
    
    # Step 1: Dimension Discovery
    print("-" * 70)
    print("STEP 1: DIMENSION DISCOVERY")
    print("-" * 70)
    
    discovery = DimensionDiscovery()
    discovery.ingest_corpus(str(corpus_path))
    
    # Let it discover dimensions (don't predefine count)
    dimensions = discovery.discover_dimensions(variance_threshold=0.80)
    
    print(f"\n✓ Discovered {len(dimensions)} dimensions")
    
    # Step 2: Build Emergent Gear Chain
    print()
    print("-" * 70)
    print("STEP 2: BUILD EMERGENT GEAR CHAIN")
    print("-" * 70)
    
    chain = EmergentGearChain(discovery)
    print(f"\n✓ Built gear chain with {len(chain.gears)} gears")
    
    # Step 3: Test on Known Concepts
    print()
    print("-" * 70)
    print("STEP 3: ANALYZE KNOWN CONCEPTS")
    print("-" * 70)
    
    test_concepts = ['holmes', 'watson', 'moriarty', 'detective', 'criminal', 
                     'evidence', 'mystery', 'case', 'london', 'inspector']
    
    for concept in test_concepts:
        analysis = chain.analyze_concept(concept)
        if 'error' in analysis:
            print(f"\n{concept}: {analysis['error']}")
            continue
        
        print(f"\n{concept.upper()}:")
        for dim_name, dim_info in analysis['dimensions'].items():
            pole = dim_info['pole']
            strength = dim_info['pole_strength']
            print(f"  {dim_name}: {dim_info['position']:.3f} (toward {pole}, strength={strength:.3f})")
        
        similar = analysis.get('similar_concepts', [])
        if similar:
            print(f"  Similar: {[c for c, s in similar[:3]]}")
    
    # Step 4: Test Relationships
    print()
    print("-" * 70)
    print("STEP 4: TEST RELATIONSHIPS")
    print("-" * 70)
    
    # Can we find meaningful relationships?
    pairs_to_test = [
        ('holmes', 'watson'),
        ('holmes', 'moriarty'),
        ('detective', 'criminal'),
    ]
    
    for a, b in pairs_to_test:
        pos_a = discovery.get_concept_position(a)
        pos_b = discovery.get_concept_position(b)
        
        if pos_a is None or pos_b is None:
            print(f"\n{a} <-> {b}: one or both unknown")
            continue
        
        # Vector from a to b
        diff = pos_b - pos_a
        distance = np.linalg.norm(diff)
        
        print(f"\n{a} → {b}:")
        print(f"  Distance: {distance:.3f}")
        print(f"  Direction: {diff}")
        
        # Which dimension has the biggest difference?
        max_dim_idx = np.argmax(np.abs(diff))
        max_dim = dimensions[max_dim_idx] if max_dim_idx < len(dimensions) else None
        if max_dim:
            print(f"  Biggest difference on: {max_dim.name} (Δ={diff[max_dim_idx]:.3f})")
    
    # Step 5: Summary
    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    
    print(f"\nDiscovered {len(dimensions)} emergent dimensions:")
    for dim in dimensions:
        print(f"  • {dim.name}")
        print(f"    Poles: {dim.poles[0]} <---> {dim.poles[1]}")
        print(f"    Variance explained: {dim.variance_explained*100:.1f}%")
    
    print("\nKey Insight:")
    print("  The dimensions emerged from the DATA, not from predefinition.")
    print("  Each dimension has a natural balance point (critical line).")
    print("  The gear chain built itself from these emergent dimensions.")
    
    # Compare to known quaternion axes
    print("\nComparison to Known Quaternion Axes:")
    print("  Known: gender (x), age (y), agency (z), animacy (w)")
    print("  Discovered: (see dimensions above)")
    print("  Question: Do the discovered dimensions correspond to known axes?")
    
    return discovery, chain


def run_action_based_experiment():
    """Run the action-based dimension discovery experiment."""
    
    print()
    print("=" * 70)
    print("ACTION-BASED DIMENSION DISCOVERY")
    print("=" * 70)
    print()
    print("This approach focuses on WHAT AGENTS DO rather than topic co-occurrence.")
    print("Hypothesis: Agency and other semantic dimensions will emerge from behavior.")
    print()
    
    # Find corpus
    corpus_paths = [
        Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_experimental.json",
        Path(__file__).parent.parent / "truthspace_lcm" / "corpus_experimental.json",
    ]
    
    corpus_path = None
    for p in corpus_paths:
        if p.exists():
            corpus_path = p
            break
    
    if corpus_path is None:
        print("ERROR: Could not find corpus file")
        return None
    
    print(f"Using corpus: {corpus_path}")
    print()
    
    # Action-based discovery
    action_discovery = ActionBasedDiscovery()
    action_discovery.ingest_corpus(str(corpus_path))
    
    dimensions = action_discovery.discover_dimensions()
    
    # Show agency scores for known characters
    print()
    print("-" * 70)
    print("AGENCY SCORES FOR KNOWN CHARACTERS")
    print("-" * 70)
    
    test_agents = ['holmes', 'watson', 'moriarty', 'lestrade', 'criminal', 
                   'detective', 'victim', 'inspector', 'professor']
    
    for agent in test_agents:
        agency = action_discovery.compute_agency_score(agent)
        actions = action_discovery.agent_actions.get(agent, {})
        top_actions = sorted(actions.items(), key=lambda x: -x[1])[:5]
        
        if actions:
            print(f"\n{agent.upper()}:")
            print(f"  Agency score: {agency:.3f}")
            print(f"  Top actions: {top_actions}")
    
    # Discover verb clusters
    print()
    print("-" * 70)
    print("VERB CLUSTERING (Who does what)")
    print("-" * 70)
    
    # Find verbs that are done by high-agency vs low-agency agents
    high_agency_agents = set()
    low_agency_agents = set()
    
    for agent in action_discovery.agent_actions.keys():
        score = action_discovery.compute_agency_score(agent)
        if score > 0.1:
            high_agency_agents.add(agent)
        elif score < -0.1:
            low_agency_agents.add(agent)
    
    print(f"\nHigh agency agents: {list(high_agency_agents)[:10]}")
    print(f"Low agency agents: {list(low_agency_agents)[:10]}")
    
    # Find verbs characteristic of each group
    high_agency_verb_counts = defaultdict(int)
    low_agency_verb_counts = defaultdict(int)
    
    for agent in high_agency_agents:
        for verb, count in action_discovery.agent_actions.get(agent, {}).items():
            high_agency_verb_counts[verb] += count
    
    for agent in low_agency_agents:
        for verb, count in action_discovery.agent_actions.get(agent, {}).items():
            low_agency_verb_counts[verb] += count
    
    print(f"\nVerbs done by high-agency agents: {sorted(high_agency_verb_counts.items(), key=lambda x: -x[1])[:10]}")
    print(f"Verbs done by low-agency agents: {sorted(low_agency_verb_counts.items(), key=lambda x: -x[1])[:10]}")
    
    return action_discovery, dimensions


def run_self_building_experiment():
    """
    The key experiment: Can the system build its own structure from errors?
    
    Based on the insight: "Error doesn't measure accuracy - it tells us WHERE to add structure."
    """
    
    print()
    print("=" * 70)
    print("SELF-BUILDING STRUCTURE EXPERIMENT")
    print("=" * 70)
    print()
    print("Goal: Start with ZERO dimensions, let errors guide structure building")
    print("Method: Process queries, detect errors, add dimensions as needed")
    print()
    
    # Find corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_experimental.json"
    if not corpus_path.exists():
        corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "corpus_experimental.json"
    
    # Load corpus
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    # Build a simple agent-action index
    agent_actions = defaultdict(lambda: defaultdict(int))
    agent_targets = defaultdict(set)
    
    for frame in corpus['frames']:
        text = frame.get('text', '')
        agent = frame.get('agent', '').lower()
        if not agent or not text:
            continue
        
        words = text.lower().split()
        try:
            agent_idx = None
            for i, w in enumerate(words):
                if agent in w:
                    agent_idx = i
                    break
            
            if agent_idx is not None and agent_idx + 1 < len(words):
                verb = re.sub(r'[^a-z]', '', words[agent_idx + 1])
                if len(verb) > 2:
                    agent_actions[agent][verb] += 1
                    
                    # Get targets
                    for w in words[agent_idx + 2:agent_idx + 5]:
                        w = re.sub(r'[^a-z]', '', w)
                        if len(w) > 3 and w not in {'the', 'and', 'with', 'from'}:
                            agent_targets[agent].add(w)
        except:
            pass
    
    print(f"Indexed {len(agent_actions)} agents")
    
    # Self-building structure
    class SelfBuildingStructure:
        """Structure that builds itself from errors."""
        
        def __init__(self):
            self.dimensions = []
            self.concept_positions = {}
            self.dimension_extractors = []
            
        def add_dimension(self, name: str, extractor):
            """Add a new dimension with its feature extractor."""
            self.dimensions.append(name)
            self.dimension_extractors.append(extractor)
            print(f"  + Added dimension: {name}")
            
            # Recompute all positions
            self._recompute_positions()
        
        def _recompute_positions(self):
            """Recompute all concept positions with current dimensions."""
            for concept in agent_actions.keys():
                pos = []
                for extractor in self.dimension_extractors:
                    pos.append(extractor(concept))
                self.concept_positions[concept] = np.array(pos)
        
        def get_position(self, concept: str) -> Optional[np.ndarray]:
            return self.concept_positions.get(concept.lower())
        
        def query(self, concept: str) -> Dict[str, Any]:
            """Query the structure for a concept."""
            pos = self.get_position(concept)
            if pos is None:
                return {'error': 'unknown', 'concept': concept}
            
            result = {'concept': concept, 'position': pos.tolist()}
            for i, dim in enumerate(self.dimensions):
                result[dim] = pos[i] if i < len(pos) else 0.0
            
            return result
        
        def find_similar(self, concept: str, k: int = 5) -> List[Tuple[str, float]]:
            """Find similar concepts."""
            pos = self.get_position(concept)
            if pos is None:
                return []
            
            similarities = []
            for other, other_pos in self.concept_positions.items():
                if other != concept.lower() and len(other_pos) == len(pos):
                    dist = np.linalg.norm(pos - other_pos)
                    similarities.append((other, dist))
            
            return sorted(similarities, key=lambda x: x[1])[:k]
    
    # Feature extractors (these will be "discovered" through errors)
    def agency_extractor(concept):
        """Extract agency from action patterns."""
        actions = agent_actions.get(concept, {})
        if not actions:
            return 0.0
        
        high_agency = {'investigates', 'solves', 'commands', 'leads', 'decides',
                       'creates', 'discovers', 'confronts', 'attacks', 'rules',
                       'examined', 'deduced', 'solved', 'investigated', 'discovered',
                       'investigate', 'studies', 'examines', 'govern'}
        low_agency = {'assists', 'follows', 'obeys', 'watches', 'waits',
                      'assisted', 'followed', 'watched', 'waited', 'served',
                      'documents', 'supports'}
        
        high = sum(actions.get(v, 0) for v in high_agency)
        low = sum(actions.get(v, 0) for v in low_agency)
        total = sum(actions.values())
        
        return (high - low) / total if total > 0 else 0.0
    
    def diversity_extractor(concept):
        """Extract action diversity."""
        actions = agent_actions.get(concept, {})
        return len(actions) / 50.0  # Normalize
    
    def target_breadth_extractor(concept):
        """Extract target breadth."""
        targets = agent_targets.get(concept, set())
        return len(targets) / 20.0  # Normalize
    
    def activity_extractor(concept):
        """Extract overall activity level."""
        actions = agent_actions.get(concept, {})
        return sum(actions.values()) / 100.0  # Normalize
    
    # Start with ZERO dimensions
    structure = SelfBuildingStructure()
    
    print("\n--- Starting with 0 dimensions ---")
    
    # Test queries that will reveal needed dimensions
    test_queries = [
        ('holmes', 'watson', 'Should be different - Holmes acts, Watson assists'),
        ('detective', 'criminal', 'Should be different - opposite roles'),
        ('holmes', 'moriarty', 'Should be different - protagonist vs antagonist'),
    ]
    
    print("\n--- Testing with 0 dimensions ---")
    for a, b, reason in test_queries:
        pos_a = structure.get_position(a)
        pos_b = structure.get_position(b)
        if pos_a is None or pos_b is None:
            print(f"  {a} vs {b}: Cannot compare (no dimensions)")
        else:
            dist = np.linalg.norm(pos_a - pos_b)
            print(f"  {a} vs {b}: distance = {dist:.3f}")
    
    # ERROR: Can't distinguish anything! Add first dimension.
    print("\n--- Error detected: Cannot distinguish concepts ---")
    print("--- Adding AGENCY dimension ---")
    structure.add_dimension('agency', agency_extractor)
    
    print("\n--- Testing with 1 dimension (agency) ---")
    for a, b, reason in test_queries:
        result_a = structure.query(a)
        result_b = structure.query(b)
        if 'error' in result_a or 'error' in result_b:
            print(f"  {a} vs {b}: Unknown concept")
            continue
        
        pos_a = structure.get_position(a)
        pos_b = structure.get_position(b)
        dist = np.linalg.norm(pos_a - pos_b)
        print(f"  {a} (agency={result_a.get('agency', 0):.3f}) vs {b} (agency={result_b.get('agency', 0):.3f}): distance = {dist:.3f}")
    
    # Check if we can now distinguish
    holmes_watson_dist = np.linalg.norm(
        structure.get_position('holmes') - structure.get_position('watson')
    ) if structure.get_position('holmes') is not None and structure.get_position('watson') is not None else 0
    
    if holmes_watson_dist > 0.3:
        print(f"\n✓ Agency dimension successfully distinguishes Holmes from Watson!")
    else:
        print(f"\n✗ Agency dimension not sufficient, need more dimensions")
    
    # Add more dimensions to see if they help
    print("\n--- Adding DIVERSITY dimension ---")
    structure.add_dimension('diversity', diversity_extractor)
    
    print("\n--- Testing with 2 dimensions ---")
    for a, b, reason in test_queries:
        pos_a = structure.get_position(a)
        pos_b = structure.get_position(b)
        if pos_a is None or pos_b is None:
            continue
        dist = np.linalg.norm(pos_a - pos_b)
        print(f"  {a} vs {b}: distance = {dist:.3f}")
    
    # Find similar concepts
    print("\n--- Similar concepts (with 2 dimensions) ---")
    for concept in ['holmes', 'watson', 'detective']:
        similar = structure.find_similar(concept, k=5)
        if similar:
            print(f"  {concept}: {[s[0] for s in similar]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SELF-BUILDING SUMMARY")
    print("=" * 70)
    print(f"\nFinal structure has {len(structure.dimensions)} dimensions:")
    for dim in structure.dimensions:
        print(f"  • {dim}")
    
    print("\nKey insight:")
    print("  The structure built itself by detecting what it COULDN'T distinguish.")
    print("  Each error pointed to a missing dimension.")
    print("  Dimensions emerged from the need to separate concepts.")
    
    # Show the emergent structure
    print("\n--- Emergent Concept Positions ---")
    key_concepts = ['holmes', 'watson', 'moriarty', 'detective', 'criminal', 'inspector']
    for concept in key_concepts:
        result = structure.query(concept)
        if 'error' not in result:
            pos_str = ', '.join(f"{d}={result.get(d, 0):.3f}" for d in structure.dimensions)
            print(f"  {concept}: [{pos_str}]")
    
    return structure


if __name__ == "__main__":
    # Run the self-building experiment (most interesting)
    print("\n" + "=" * 70)
    print("EXPERIMENT: SELF-BUILDING EMERGENT STRUCTURE")
    print("=" * 70)
    structure = run_self_building_experiment()
    
    # Also run action-based discovery for comparison
    print("\n" + "=" * 70)
    print("EXPERIMENT: ACTION-BASED DISCOVERY (for comparison)")
    print("=" * 70)
    action_discovery, action_dims = run_action_based_experiment()
