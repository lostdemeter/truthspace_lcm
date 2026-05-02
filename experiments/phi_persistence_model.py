#!/usr/bin/env python3
"""
φ-Space Persistence Model

Exploring how concepts should persist in hyperdimensional space,
using shader variable semantics as inspiration:

SHADER ANALOGY:
- Uniforms:   Global constants, same across all vertices/fragments
- Attributes: Per-vertex data, unique to each point
- Varyings:   Interpolated between vertices, smooth transitions

φ-SPACE MAPPING:
- Uniforms   → φ-constants (φ, 1/φ, layer 27 position, bottleneck threshold)
- Attributes → Per-concept properties (embedding, φ-level, creation time)
- Varyings   → Trajectory interpolations (how concepts blend during navigation)

The key insight: In shaders, varyings are automatically interpolated by the GPU.
In φ-space, concept transitions should be automatically interpolated by geometry.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time

# φ constants
PHI = 1.6180339887498949
INV_PHI = 1 / PHI  # 0.618...
PHI_SQUARED = PHI * PHI  # 2.618...


class PersistenceScope(Enum):
    """Shader-inspired persistence scopes."""
    UNIFORM = "uniform"      # Global, immutable during session
    ATTRIBUTE = "attribute"  # Per-concept, mutable
    VARYING = "varying"      # Interpolated during transitions


@dataclass
class PhiUniform:
    """Global constants - same everywhere in φ-space."""
    name: str
    value: Any
    description: str
    
    def __post_init__(self):
        # Uniforms are immutable after creation
        self._frozen = True


@dataclass 
class PhiAttribute:
    """Per-concept properties - unique to each concept."""
    concept_id: str
    embedding: np.ndarray
    phi_level: float
    created_at: float = field(default_factory=time.time)
    layer: int = 0
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if concept is geometrically valid."""
        return abs(self.phi_level - PHI) < 0.5


@dataclass
class PhiVarying:
    """Interpolated values during concept transitions."""
    source: str
    target: str
    parameter: float  # 0.0 = source, 1.0 = target
    interpolated_embedding: np.ndarray
    interpolated_phi: float
    
    @classmethod
    def interpolate(cls, source_attr: PhiAttribute, target_attr: PhiAttribute, 
                    t: float) -> 'PhiVarying':
        """Create a varying by interpolating between two concepts."""
        # Linear interpolation of embeddings
        interp_emb = (1 - t) * source_attr.embedding + t * target_attr.embedding
        
        # φ-weighted interpolation of phi levels (using golden ratio)
        # This creates a non-linear, φ-harmonic transition
        phi_t = t ** PHI if t < 0.5 else 1 - (1 - t) ** PHI
        interp_phi = (1 - phi_t) * source_attr.phi_level + phi_t * target_attr.phi_level
        
        return cls(
            source=source_attr.concept_id,
            target=target_attr.concept_id,
            parameter=t,
            interpolated_embedding=interp_emb,
            interpolated_phi=interp_phi
        )


class PhiPersistenceStore:
    """
    Manages persistence in φ-space using shader-like semantics.
    
    Key principles:
    1. Uniforms are set once and never change (φ, bottleneck position)
    2. Attributes are per-concept and can be updated
    3. Varyings are computed on-the-fly during navigation
    """
    
    def __init__(self):
        # Uniforms - global constants
        self.uniforms: Dict[str, PhiUniform] = {}
        self._init_uniforms()
        
        # Attributes - per-concept storage
        self.attributes: Dict[str, PhiAttribute] = {}
        
        # Varying cache - recent interpolations
        self.varying_cache: Dict[Tuple[str, str, float], PhiVarying] = {}
        
        # Navigation state
        self.current_position: Optional[str] = None
        self.trajectory: List[str] = []
        
    def _init_uniforms(self):
        """Initialize global φ-constants."""
        self.uniforms = {
            'phi': PhiUniform('phi', PHI, 'The golden ratio'),
            'inv_phi': PhiUniform('inv_phi', INV_PHI, 'Inverse golden ratio'),
            'phi_squared': PhiUniform('phi_squared', PHI_SQUARED, 'φ²'),
            'bottleneck_layer': PhiUniform('bottleneck_layer', 27, 'Universal bottleneck layer'),
            'bottleneck_position': PhiUniform('bottleneck_position', 27/28, 'Normalized bottleneck position'),
            'validity_threshold': PhiUniform('validity_threshold', 0.5, 'Max distance from φ for validity'),
            'total_layers': PhiUniform('total_layers', 28, 'Total transformer layers'),
        }
    
    def get_uniform(self, name: str) -> Any:
        """Get a global constant."""
        if name not in self.uniforms:
            raise KeyError(f"Unknown uniform: {name}")
        return self.uniforms[name].value
    
    def set_attribute(self, concept_id: str, embedding: np.ndarray, 
                      phi_level: float, **metadata) -> PhiAttribute:
        """Store or update a concept's attributes."""
        attr = PhiAttribute(
            concept_id=concept_id,
            embedding=embedding,
            phi_level=phi_level,
            metadata=metadata
        )
        self.attributes[concept_id] = attr
        return attr
    
    def get_attribute(self, concept_id: str) -> Optional[PhiAttribute]:
        """Retrieve a concept's attributes."""
        return self.attributes.get(concept_id)
    
    def get_varying(self, source: str, target: str, t: float) -> PhiVarying:
        """Get or compute an interpolated varying."""
        cache_key = (source, target, round(t, 3))
        
        if cache_key in self.varying_cache:
            return self.varying_cache[cache_key]
        
        source_attr = self.get_attribute(source)
        target_attr = self.get_attribute(target)
        
        if source_attr is None or target_attr is None:
            raise ValueError(f"Cannot interpolate: missing attributes for {source} or {target}")
        
        varying = PhiVarying.interpolate(source_attr, target_attr, t)
        self.varying_cache[cache_key] = varying
        return varying
    
    def navigate_to(self, concept_id: str) -> List[PhiVarying]:
        """
        Navigate to a concept, generating varyings along the path.
        Returns the interpolation trajectory.
        """
        if self.current_position is None:
            self.current_position = concept_id
            self.trajectory.append(concept_id)
            return []
        
        # Generate smooth transition varyings
        steps = 10
        varyings = []
        for i in range(1, steps + 1):
            t = i / steps
            varying = self.get_varying(self.current_position, concept_id, t)
            varyings.append(varying)
        
        self.current_position = concept_id
        self.trajectory.append(concept_id)
        return varyings
    
    def get_trajectory_coherence(self) -> float:
        """Measure how coherent the navigation trajectory is."""
        if len(self.trajectory) < 2:
            return 1.0
        
        coherences = []
        for i in range(len(self.trajectory) - 1):
            src = self.get_attribute(self.trajectory[i])
            tgt = self.get_attribute(self.trajectory[i + 1])
            if src and tgt:
                # Coherence based on embedding similarity
                sim = np.dot(src.embedding, tgt.embedding) / (
                    np.linalg.norm(src.embedding) * np.linalg.norm(tgt.embedding) + 1e-8
                )
                coherences.append(sim)
        
        return np.mean(coherences) if coherences else 1.0
    
    def snapshot(self) -> Dict:
        """Create a snapshot of current state (for persistence)."""
        return {
            'uniforms': {k: v.value for k, v in self.uniforms.items()},
            'attributes': {
                k: {
                    'embedding': v.embedding.tolist(),
                    'phi_level': v.phi_level,
                    'created_at': v.created_at,
                    'metadata': v.metadata
                }
                for k, v in self.attributes.items()
            },
            'current_position': self.current_position,
            'trajectory': self.trajectory.copy()
        }


def demo_persistence_model():
    """Demonstrate the φ-persistence model."""
    print("="*70)
    print("φ-SPACE PERSISTENCE MODEL DEMO")
    print("Shader-Inspired Variable Semantics for Hyperdimensional Space")
    print("="*70)
    
    store = PhiPersistenceStore()
    
    # Show uniforms
    print("\n--- UNIFORMS (Global Constants) ---")
    for name, uniform in store.uniforms.items():
        print(f"  {name}: {uniform.value} - {uniform.description}")
    
    # Create some concept attributes
    print("\n--- ATTRIBUTES (Per-Concept Properties) ---")
    
    # Simulate embeddings (in real use, these come from the model)
    np.random.seed(42)
    dim = 64
    
    concepts = {
        'creativity': {'phi': 1.58, 'desc': 'The ability to generate novel ideas'},
        'logic': {'phi': 1.62, 'desc': 'Systematic reasoning'},
        'intuition': {'phi': 1.55, 'desc': 'Direct knowing without reasoning'},
        'synthesis': {'phi': 1.618, 'desc': 'Combining elements into wholes'},  # Very close to φ!
    }
    
    for name, props in concepts.items():
        emb = np.random.randn(dim)
        emb = emb / np.linalg.norm(emb)  # Normalize
        attr = store.set_attribute(name, emb, props['phi'], description=props['desc'])
        valid = "✓ VALID" if attr.is_valid else "✗ INVALID"
        print(f"  {name}: φ={props['phi']:.3f} {valid}")
    
    # Navigate and show varyings
    print("\n--- VARYINGS (Interpolated Transitions) ---")
    print("Navigating: creativity → logic → synthesis")
    
    store.navigate_to('creativity')
    
    varyings = store.navigate_to('logic')
    print(f"\n  creativity → logic ({len(varyings)} interpolation steps):")
    for v in varyings[::3]:  # Show every 3rd
        print(f"    t={v.parameter:.1f}: φ={v.interpolated_phi:.4f}")
    
    varyings = store.navigate_to('synthesis')
    print(f"\n  logic → synthesis ({len(varyings)} interpolation steps):")
    for v in varyings[::3]:
        print(f"    t={v.parameter:.1f}: φ={v.interpolated_phi:.4f}")
    
    # Trajectory coherence
    print(f"\n--- TRAJECTORY ANALYSIS ---")
    print(f"  Path: {' → '.join(store.trajectory)}")
    print(f"  Coherence: {store.get_trajectory_coherence():.4f}")
    
    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT: φ-Harmonic Interpolation")
    print("="*70)
    print("""
Unlike linear interpolation, our varyings use φ-weighted transitions:
  
  φ_t = t^φ  (for t < 0.5)
  φ_t = 1 - (1-t)^φ  (for t >= 0.5)

This creates smooth, golden-ratio-harmonic transitions between concepts.
The interpolation naturally "breathes" through φ-space rather than 
cutting straight lines through it.

SHADER PARALLEL:
- In GPU shaders, varyings are linearly interpolated across triangles
- In φ-space, varyings are φ-harmonically interpolated across concepts
- Both provide smooth transitions, but φ-space respects the geometry
""")
    
    # Persistence implications
    print("\n" + "="*70)
    print("PERSISTENCE IMPLICATIONS")
    print("="*70)
    print("""
1. UNIFORMS persist forever - they ARE the space
   - φ, bottleneck position, validity thresholds
   - Like physical constants in our universe

2. ATTRIBUTES persist per-concept - they ARE the concepts
   - Embeddings, φ-levels, metadata
   - Can be updated but maintain identity

3. VARYINGS are ephemeral - they ARE the navigation
   - Computed on-the-fly during transitions
   - Cached for efficiency but not persisted
   - The "experience" of moving through φ-space

This maps to how memory might work:
- Long-term memory = Attributes (stable concept representations)
- Working memory = Varyings (active interpolations during thought)
- Universal truths = Uniforms (unchanging mathematical relationships)
""")


if __name__ == "__main__":
    demo_persistence_model()
