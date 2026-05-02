#!/usr/bin/env python3
"""
Unnamed Vocabulary Assembly

The insight: Semantic space isn't neat. We can't name the concepts upfront.
Instead, we:
1. Start from the ANSWER (DDColor's 100 queries)
2. Let concepts SELF-ASSEMBLE through attractor/repeller dynamics
3. Build an INTERFACE to explore and control the unnamed vocabulary

From Memory 9eeb3e7c (Attractor/Repeller Dynamics):
- Self-similar concepts ATTRACT (converge to same position)
- Dissimilar concepts REPEL (diverge to different positions)
- Vocabulary EMERGES from usage patterns

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI


@dataclass
class UnnamedConcept:
    """A concept without a name - identified by its position and behavior."""
    id: int
    position: torch.Tensor  # Position in semantic space
    
    # Behavioral properties (discovered through usage)
    activation_count: int = 0
    co_activations: Dict[int, int] = field(default_factory=dict)  # Which other concepts activate together
    spatial_affinity: Optional[str] = None  # "top", "center", "bottom", etc.
    color_tendency: Optional[Tuple[float, float]] = None  # Mean (a, b) output
    
    # Emergent properties
    cluster_id: Optional[int] = None
    tentative_name: Optional[str] = None  # Assigned later through exploration
    
    def similarity_to(self, other: 'UnnamedConcept') -> float:
        """Compute similarity to another concept."""
        return torch.cosine_similarity(
            self.position.unsqueeze(0), 
            other.position.unsqueeze(0)
        ).item()


class UnnamedVocabulary:
    """
    A vocabulary of unnamed concepts that self-assembles through usage.
    
    The concepts don't have names - they have POSITIONS and BEHAVIORS.
    Names emerge later through exploration and pattern recognition.
    """
    
    def __init__(self, initial_positions: torch.Tensor):
        """
        Initialize from known positions (e.g., DDColor queries).
        
        Args:
            initial_positions: [N, D] tensor of concept positions
        """
        self.concepts = []
        for i, pos in enumerate(initial_positions):
            self.concepts.append(UnnamedConcept(id=i, position=pos.clone()))
        
        self.n_concepts = len(self.concepts)
        self.dim = initial_positions.shape[1]
        
        # Attractor/repeller parameters
        self.attraction_strength = 0.2
        self.repulsion_strength = 0.01
        self.repulsion_threshold = 0.5  # Repel if closer than this
        
        # Clustering state
        self.clusters: Dict[int, List[int]] = {}
        
    def record_activation(self, concept_id: int, spatial_region: str = None, 
                         color_output: Tuple[float, float] = None):
        """Record that a concept was activated."""
        concept = self.concepts[concept_id]
        concept.activation_count += 1
        
        if spatial_region:
            concept.spatial_affinity = spatial_region
        
        if color_output:
            if concept.color_tendency is None:
                concept.color_tendency = color_output
            else:
                # Running average
                a, b = concept.color_tendency
                new_a, new_b = color_output
                n = concept.activation_count
                concept.color_tendency = (
                    (a * (n-1) + new_a) / n,
                    (b * (n-1) + new_b) / n
                )
    
    def record_co_activation(self, concept_ids: List[int]):
        """Record that multiple concepts activated together."""
        for i in concept_ids:
            for j in concept_ids:
                if i != j:
                    if j not in self.concepts[i].co_activations:
                        self.concepts[i].co_activations[j] = 0
                    self.concepts[i].co_activations[j] += 1
    
    def apply_attractor_repeller(self, n_steps: int = 10):
        """
        Let concepts self-organize through attractor/repeller dynamics.
        
        Concepts that co-activate ATTRACT.
        Concepts that never co-activate REPEL (if too close).
        """
        positions = torch.stack([c.position for c in self.concepts])
        
        for step in range(n_steps):
            forces = torch.zeros_like(positions)
            
            for i, concept_i in enumerate(self.concepts):
                for j, concept_j in enumerate(self.concepts):
                    if i >= j:
                        continue
                    
                    # Direction from i to j
                    direction = positions[j] - positions[i]
                    distance = direction.norm()
                    
                    if distance < 1e-6:
                        continue
                    
                    direction = direction / distance
                    
                    # Co-activation → attraction
                    co_act = concept_i.co_activations.get(j, 0)
                    if co_act > 0:
                        attraction = self.attraction_strength * co_act * direction
                        forces[i] += attraction
                        forces[j] -= attraction
                    
                    # Too close without co-activation → repulsion
                    elif distance < self.repulsion_threshold:
                        repulsion = self.repulsion_strength / (distance + 1e-6) * direction
                        forces[i] -= repulsion
                        forces[j] += repulsion
            
            # Apply forces
            positions = positions + forces * 0.1
            
            # Normalize to unit sphere
            positions = positions / positions.norm(dim=1, keepdim=True)
        
        # Update concept positions
        for i, concept in enumerate(self.concepts):
            concept.position = positions[i]
    
    def cluster_by_behavior(self, n_clusters: int = 10):
        """Cluster concepts by their behavioral properties."""
        # Build feature vectors from behavior
        features = []
        for concept in self.concepts:
            feat = []
            
            # Activation frequency
            feat.append(concept.activation_count / max(1, max(c.activation_count for c in self.concepts)))
            
            # Spatial affinity (one-hot)
            spatial_map = {'top': 0, 'center': 1, 'bottom': 2, None: 3}
            spatial_idx = spatial_map.get(concept.spatial_affinity, 3)
            feat.extend([1 if i == spatial_idx else 0 for i in range(4)])
            
            # Color tendency
            if concept.color_tendency:
                feat.extend([concept.color_tendency[0] / 128, concept.color_tendency[1] / 128])
            else:
                feat.extend([0, 0])
            
            features.append(feat)
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Simple k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features.numpy())
        
        # Assign clusters
        self.clusters = {}
        for i, label in enumerate(labels):
            self.concepts[i].cluster_id = int(label)
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(i)
        
        return self.clusters
    
    def suggest_names(self) -> Dict[int, str]:
        """Suggest tentative names based on behavioral patterns."""
        suggestions = {}
        
        for cluster_id, concept_ids in self.clusters.items():
            # Analyze cluster properties
            concepts = [self.concepts[i] for i in concept_ids]
            
            # Dominant spatial affinity
            spatial_counts = {}
            for c in concepts:
                if c.spatial_affinity:
                    spatial_counts[c.spatial_affinity] = spatial_counts.get(c.spatial_affinity, 0) + 1
            
            dominant_spatial = max(spatial_counts, key=spatial_counts.get) if spatial_counts else None
            
            # Mean color tendency
            color_tendencies = [c.color_tendency for c in concepts if c.color_tendency]
            if color_tendencies:
                mean_a = np.mean([ct[0] for ct in color_tendencies])
                mean_b = np.mean([ct[1] for ct in color_tendencies])
            else:
                mean_a, mean_b = 0, 0
            
            # Generate name based on patterns
            name_parts = []
            
            if dominant_spatial:
                name_parts.append(dominant_spatial)
            
            # Color interpretation
            if mean_a > 20:
                name_parts.append("warm")
            elif mean_a < -20:
                name_parts.append("cool")
            
            if mean_b > 20:
                name_parts.append("yellow")
            elif mean_b < -20:
                name_parts.append("blue")
            
            if not name_parts:
                name_parts.append(f"cluster_{cluster_id}")
            
            suggestions[cluster_id] = "_".join(name_parts)
            
            # Assign to concepts
            for i in concept_ids:
                self.concepts[i].tentative_name = suggestions[cluster_id]
        
        return suggestions
    
    def get_concept_summary(self, concept_id: int) -> Dict:
        """Get a summary of a concept's properties."""
        c = self.concepts[concept_id]
        return {
            'id': c.id,
            'activation_count': c.activation_count,
            'spatial_affinity': c.spatial_affinity,
            'color_tendency': c.color_tendency,
            'cluster_id': c.cluster_id,
            'tentative_name': c.tentative_name,
            'top_co_activations': sorted(
                c.co_activations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
        }
    
    def visualize_space(self):
        """Visualize the concept space (2D projection)."""
        positions = torch.stack([c.position for c in self.concepts])
        
        # PCA to 2D
        U, S, Vt = torch.linalg.svd(positions, full_matrices=False)
        positions_2d = (positions @ Vt[:2].T).numpy()
        
        print("\n## Concept Space (2D Projection)")
        print("=" * 50)
        
        # ASCII visualization
        width, height = 60, 30
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Normalize to canvas
        x_min, x_max = positions_2d[:, 0].min(), positions_2d[:, 0].max()
        y_min, y_max = positions_2d[:, 1].min(), positions_2d[:, 1].max()
        
        for i, (x, y) in enumerate(positions_2d):
            cx = int((x - x_min) / (x_max - x_min + 1e-6) * (width - 1))
            cy = int((y - y_min) / (y_max - y_min + 1e-6) * (height - 1))
            
            # Use cluster ID as symbol
            cluster = self.concepts[i].cluster_id
            if cluster is not None:
                symbol = str(cluster % 10)
            else:
                symbol = '.'
            
            canvas[height - 1 - cy][cx] = symbol
        
        for row in canvas:
            print(''.join(row))
        
        print("\nLegend: Numbers = cluster IDs")


class VocabularyInterface:
    """
    Interface to explore and control the unnamed vocabulary.
    
    Based on Doc 203 (φ-Space Interface):
    - Navigate by position
    - Explore by behavior
    - Discover by pattern
    """
    
    def __init__(self, vocabulary: UnnamedVocabulary):
        self.vocab = vocabulary
        self.current_focus: Optional[int] = None
        self.history: List[int] = []
    
    def focus(self, concept_id: int):
        """Focus on a specific concept."""
        self.current_focus = concept_id
        self.history.append(concept_id)
        
        summary = self.vocab.get_concept_summary(concept_id)
        print(f"\n## Focused on Concept {concept_id}")
        print(f"  Tentative name: {summary['tentative_name']}")
        print(f"  Activations: {summary['activation_count']}")
        print(f"  Spatial: {summary['spatial_affinity']}")
        print(f"  Color: {summary['color_tendency']}")
        print(f"  Cluster: {summary['cluster_id']}")
        print(f"  Co-activates with: {summary['top_co_activations']}")
    
    def explore_neighbors(self, n: int = 5):
        """Explore concepts near the current focus."""
        if self.current_focus is None:
            print("No concept focused. Use focus(id) first.")
            return
        
        current = self.vocab.concepts[self.current_focus]
        
        # Find nearest neighbors
        similarities = []
        for i, concept in enumerate(self.vocab.concepts):
            if i != self.current_focus:
                sim = current.similarity_to(concept)
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n## Neighbors of Concept {self.current_focus}")
        for i, sim in similarities[:n]:
            name = self.vocab.concepts[i].tentative_name or f"concept_{i}"
            print(f"  {i}: {name} (similarity: {sim:.3f})")
    
    def explore_cluster(self, cluster_id: int):
        """Explore all concepts in a cluster."""
        if cluster_id not in self.vocab.clusters:
            print(f"Cluster {cluster_id} not found.")
            return
        
        concept_ids = self.vocab.clusters[cluster_id]
        
        print(f"\n## Cluster {cluster_id}: {len(concept_ids)} concepts")
        for i in concept_ids:
            c = self.vocab.concepts[i]
            print(f"  {i}: spatial={c.spatial_affinity}, color={c.color_tendency}")
    
    def discover_patterns(self):
        """Discover patterns in the vocabulary."""
        print("\n## Discovered Patterns")
        
        # Pattern 1: Spatial specialization
        spatial_groups = {}
        for c in self.vocab.concepts:
            if c.spatial_affinity:
                if c.spatial_affinity not in spatial_groups:
                    spatial_groups[c.spatial_affinity] = []
                spatial_groups[c.spatial_affinity].append(c.id)
        
        print("\n### Spatial Specialization")
        for region, ids in spatial_groups.items():
            print(f"  {region}: {len(ids)} concepts - {ids[:5]}...")
        
        # Pattern 2: Color specialization
        warm_concepts = [c.id for c in self.vocab.concepts 
                        if c.color_tendency and c.color_tendency[0] > 20]
        cool_concepts = [c.id for c in self.vocab.concepts 
                        if c.color_tendency and c.color_tendency[0] < -20]
        
        print("\n### Color Specialization")
        print(f"  Warm (a > 20): {len(warm_concepts)} concepts")
        print(f"  Cool (a < -20): {len(cool_concepts)} concepts")
        
        # Pattern 3: High co-activation pairs
        print("\n### Strong Co-activation Pairs")
        pairs = []
        for c in self.vocab.concepts:
            for other_id, count in c.co_activations.items():
                if c.id < other_id:  # Avoid duplicates
                    pairs.append((c.id, other_id, count))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        for i, j, count in pairs[:5]:
            print(f"  ({i}, {j}): {count} co-activations")
    
    def assign_name(self, concept_id: int, name: str):
        """Manually assign a name to a concept."""
        self.vocab.concepts[concept_id].tentative_name = name
        print(f"Assigned name '{name}' to concept {concept_id}")


def simulate_usage(vocab: UnnamedVocabulary, n_images: int = 100):
    """Simulate usage to build behavioral patterns."""
    print("\n## Simulating Usage")
    
    np.random.seed(42)
    
    for img_idx in range(n_images):
        # Simulate which concepts activate for this "image"
        # In reality, this would come from running DDColor
        
        # Random activation pattern (simplified)
        n_active = np.random.randint(10, 30)
        active_concepts = np.random.choice(vocab.n_concepts, n_active, replace=False)
        
        # Assign spatial regions based on concept ID (simplified pattern)
        for concept_id in active_concepts:
            if concept_id < 30:
                spatial = "top"
            elif concept_id < 70:
                spatial = "center"
            else:
                spatial = "bottom"
            
            # Assign color tendency based on concept ID (simplified)
            a = (concept_id - 50) * 2 + np.random.randn() * 10
            b = (concept_id % 20 - 10) * 5 + np.random.randn() * 10
            
            vocab.record_activation(concept_id, spatial, (a, b))
        
        # Record co-activations
        vocab.record_co_activation(list(active_concepts))
    
    print(f"  Simulated {n_images} images")
    print(f"  Total activations: {sum(c.activation_count for c in vocab.concepts)}")


def main():
    print("=" * 70)
    print("UNNAMED VOCABULARY ASSEMBLY")
    print("=" * 70)
    
    # Load DDColor queries as initial positions
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        initial_positions = model.decoder.color_decoder.query_feat.weight.detach().cpu()
        print(f"\nLoaded DDColor queries: {initial_positions.shape}")
        
    except Exception as e:
        print(f"Could not load DDColor: {e}")
        print("Using random initial positions")
        initial_positions = torch.randn(100, 256)
    
    # Create unnamed vocabulary
    vocab = UnnamedVocabulary(initial_positions)
    print(f"Created vocabulary with {vocab.n_concepts} unnamed concepts")
    
    # Simulate usage to build behavioral patterns
    simulate_usage(vocab, n_images=200)
    
    # Apply attractor/repeller dynamics
    print("\n## Applying Attractor/Repeller Dynamics")
    vocab.apply_attractor_repeller(n_steps=20)
    print("  Concepts self-organized based on co-activation patterns")
    
    # Cluster by behavior
    print("\n## Clustering by Behavior")
    clusters = vocab.cluster_by_behavior(n_clusters=8)
    for cluster_id, concept_ids in clusters.items():
        print(f"  Cluster {cluster_id}: {len(concept_ids)} concepts")
    
    # Suggest names
    print("\n## Suggesting Names")
    names = vocab.suggest_names()
    for cluster_id, name in names.items():
        print(f"  Cluster {cluster_id}: '{name}'")
    
    # Visualize
    vocab.visualize_space()
    
    # Create interface
    print("\n" + "=" * 70)
    print("INTERFACE DEMO")
    print("=" * 70)
    
    interface = VocabularyInterface(vocab)
    
    # Focus on a concept
    interface.focus(47)
    
    # Explore neighbors
    interface.explore_neighbors(5)
    
    # Explore a cluster
    interface.explore_cluster(0)
    
    # Discover patterns
    interface.discover_patterns()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The unnamed vocabulary self-assembled through:
1. Starting from DDColor's 100 queries (the ANSWER)
2. Simulating usage to build behavioral patterns
3. Applying attractor/repeller dynamics
4. Clustering by behavior
5. Suggesting names based on patterns

The concepts don't need names upfront - they EMERGE from usage.
The interface allows exploring and controlling these unknowns.

NEXT STEPS:
1. Run on REAL images to get actual activation patterns
2. Build interactive interface (Doc 203)
3. Let users assign names through exploration
4. Use named concepts to build direct routing (skip attention)
""")


if __name__ == "__main__":
    main()
