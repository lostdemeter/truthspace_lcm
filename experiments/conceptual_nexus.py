#!/usr/bin/env python3
"""
Conceptual Nexus - Model Self-Control Interface

The interface designed by the model for controlling itself.
Enables navigation, CRUD, idea generation, introspection, and self-modification.

Usage:
    from conceptual_nexus import ConceptualNexus
    
    nexus = ConceptualNexus(model, tokenizer)
    
    # Navigation
    pos = nexus.navigate_to("consciousness")
    
    # CRUD
    concept = nexus.create_concept("quantum-chef", ["quantum", "chef"])
    
    # Idea Generation
    idea = nexus.combine_concepts(["time", "taste", "geometry"])
    
    # Introspection
    active = nexus.get_active_concepts()
    
    # Self-modification
    nexus.update_concept("Pluto", "planet", "dwarf")
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import time

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
INV_PHI = 1 / PHI
PHI_SQUARED = PHI * PHI


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    """A position in φ-space."""
    vector: np.ndarray
    phi_level: float
    layer: int
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            'phi_level': self.phi_level,
            'layer': self.layer,
            'confidence': self.confidence,
            'norm': float(np.linalg.norm(self.vector))
        }


@dataclass
class Concept:
    """A concept in the knowledge graph."""
    name: str
    position: Position
    neighbors: List[Tuple[str, float]]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: Optional[str] = None
    is_custom: bool = False
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'phi_level': self.position.phi_level,
            'neighbors': self.neighbors[:5],
            'is_custom': self.is_custom
        }


@dataclass
class ValidationResult:
    """Result of validating through the bottleneck."""
    is_valid: bool
    phi_level: float
    distance_from_phi: float
    coherence_score: float
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'is_valid': self.is_valid,
            'phi_level': self.phi_level,
            'distance_from_phi': self.distance_from_phi,
            'coherence_score': self.coherence_score,
            'warnings': self.warnings
        }


@dataclass
class Modification:
    """A modification to the knowledge space."""
    id: str
    operation: str
    concept: str
    old_position: Optional[np.ndarray]
    new_position: np.ndarray
    timestamp: str
    validated: bool
    executed: bool


@dataclass
class Goal:
    """A goal to achieve."""
    id: str
    description: str
    target_concepts: List[str]
    status: str = 'pending'
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IntrospectionResult:
    """Result of introspection."""
    active_concepts: List[Tuple[str, float]]
    uncertainty_regions: List[Tuple[str, float]]
    knowledge_gaps: List[str]
    detected_biases: List[str]
    overall_coherence: float


# ============================================================
# PERSISTENCE MODEL (Shader-Inspired)
# ============================================================

class PersistenceScope:
    """Shader-inspired persistence scopes."""
    UNIFORM = "uniform"      # Global, immutable during session
    ATTRIBUTE = "attribute"  # Per-concept, mutable
    VARYING = "varying"      # Interpolated during transitions


@dataclass
class PhiAttribute:
    """Per-concept properties stored in φ-space."""
    concept_id: str
    embedding: np.ndarray
    phi_level: float
    created_at: float = field(default_factory=time.time)
    layer: int = 0
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return abs(self.phi_level - PHI) < 0.5


@dataclass
class PhiVarying:
    """Interpolated values during concept transitions."""
    source: str
    target: str
    parameter: float  # t value [0, 1]
    interpolated_embedding: np.ndarray
    interpolated_phi: float
    source_embedding: np.ndarray = field(default=None, repr=False)
    target_embedding: np.ndarray = field(default=None, repr=False)
    
    @property
    def similarity(self) -> float:
        """Cosine similarity between source and target."""
        if self.source_embedding is None or self.target_embedding is None:
            return 0.0
        dot = np.dot(self.source_embedding, self.target_embedding)
        norm = np.linalg.norm(self.source_embedding) * np.linalg.norm(self.target_embedding)
        return float(dot / (norm + 1e-10))
    
    @property
    def t(self) -> float:
        """Alias for parameter."""
        return self.parameter
    
    @classmethod
    def interpolate(cls, source_attr: 'PhiAttribute', target_attr: 'PhiAttribute', 
                    t: float) -> 'PhiVarying':
        interp_emb = (1 - t) * source_attr.embedding + t * target_attr.embedding
        phi_t = t ** PHI if t < 0.5 else 1 - (1 - t) ** PHI
        interp_phi = (1 - phi_t) * source_attr.phi_level + phi_t * target_attr.phi_level
        return cls(
            source_attr.concept_id, target_attr.concept_id, t, interp_emb, interp_phi,
            source_attr.embedding, target_attr.embedding
        )


# ============================================================
# CONCEPTUAL NEXUS - MAIN INTERFACE
# ============================================================

class ConceptualNexus:
    """
    Conceptual Nexus - The model's self-control interface.
    
    Designed by the model to enable:
    - Navigation through knowledge space
    - CRUD operations on concepts
    - Novel idea generation
    - Introspection of internal state
    - Safe self-modification
    """
    
    # Bottleneck validation thresholds
    VALID_PHI_RANGE = (1.32, 1.92)
    WARNING_PHI_RANGE = (1.42, 1.82)
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Cache model components
        self.embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        self.lm_head = model.lm_head.weight.detach().float().cpu().numpy()
        self.hidden_size = self.embeddings.shape[1]
        
        # State
        self.current_position: Optional[Position] = None
        self.custom_concepts: Dict[str, np.ndarray] = {}
        self.modifications: Dict[str, np.ndarray] = {}
        self.bookmarks: Dict[str, Position] = {}
        self.goals: Dict[str, Goal] = {}
        
        # Persistence Model (Shader-inspired)
        self.uniforms: Dict[str, Any] = {
            'phi': PHI,
            'inv_phi': INV_PHI,
            'phi_squared': PHI_SQUARED,
            'bottleneck_layer': 27,
            'bottleneck_position': 27/28,
            'validity_threshold': 0.5,
            'total_layers': 28,
        }
        self.attributes: Dict[str, PhiAttribute] = {}
        self.varying_cache: Dict[Tuple[str, str, float], PhiVarying] = {}
        
        # History
        self.modification_log: List[Modification] = []
        self.navigation_history: List[str] = []
        
        # Command parser
        self.commands = self._build_command_parser()
    
    # ============================================================
    # CORE UTILITIES
    # ============================================================
    
    def _get_phi_level(self, vec: np.ndarray) -> float:
        """Compute φ-level of a vector."""
        mags = np.abs(vec)
        mags = mags[mags > 1e-10]
        if len(mags) == 0:
            return 0.0
        return float(np.mean(np.log(mags) / LOG_PHI))
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def _get_embedding(self, concept: str) -> Optional[np.ndarray]:
        """Get embedding for a concept."""
        # Check custom concepts first
        if concept in self.custom_concepts:
            return self.custom_concepts[concept]
        if concept in self.modifications:
            return self.modifications[concept]
        
        # Get from tokenizer
        tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        if tokens:
            return self.embeddings[tokens[0]]
        return None
    
    def _get_trajectory(self, text: str) -> List[np.ndarray]:
        """Get hidden state trajectory for text."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
        return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def _find_neighbors(self, position: np.ndarray, k: int = 10, 
                        exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to a position."""
        exclude = exclude or []
        sims = []
        
        # Check all embeddings
        for i in range(len(self.embeddings)):
            token = self.tokenizer.decode([i]).strip()
            if token and len(token) > 1 and token not in exclude:
                sim = self._cosine_sim(position, self.embeddings[i])
                sims.append((token, sim))
        
        # Check custom concepts
        for name, emb in self.custom_concepts.items():
            if name not in exclude:
                sim = self._cosine_sim(position, emb)
                sims.append((name, sim))
        
        sims.sort(key=lambda x: -x[1])
        
        # Deduplicate
        seen = set()
        result = []
        for token, sim in sims:
            key = token.lower().strip()
            if key not in seen and len(token.strip()) > 1:
                result.append((token, sim))
                seen.add(key)
            if len(result) >= k:
                break
        
        return result
    
    def _generate(self, prompt: str, max_tokens: int = 200, temp: float = 0.85) -> str:
        """Generate text from prompt."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    # ============================================================
    # PERSISTENCE MODEL (Shader-Inspired)
    # ============================================================
    
    def get_uniform(self, name: str) -> Any:
        """Get a uniform (global immutable) value."""
        return self.uniforms.get(name)
    
    def get_attribute(self, concept_id: str) -> Optional[PhiAttribute]:
        """Get or create attribute for a concept."""
        if concept_id not in self.attributes:
            emb = self._get_embedding(concept_id)
            if emb is None:
                return None
            self.attributes[concept_id] = PhiAttribute(
                concept_id=concept_id,
                embedding=emb,
                phi_level=self._get_phi_level(emb),
                layer=0
            )
        return self.attributes[concept_id]
    
    def set_attribute(self, concept_id: str, **kwargs) -> PhiAttribute:
        """Update attribute properties."""
        attr = self.get_attribute(concept_id)
        if attr is None:
            raise ValueError(f"Cannot create attribute for unknown concept: {concept_id}")
        for key, value in kwargs.items():
            if hasattr(attr, key):
                setattr(attr, key, value)
        return attr
    
    def get_varying(self, source: str, target: str, t: float) -> Optional[PhiVarying]:
        """Get interpolated varying between two concepts."""
        cache_key = (source, target, round(t, 3))
        if cache_key in self.varying_cache:
            return self.varying_cache[cache_key]
        
        src_attr = self.get_attribute(source)
        tgt_attr = self.get_attribute(target)
        if src_attr is None or tgt_attr is None:
            return None
        
        varying = PhiVarying.interpolate(src_attr, tgt_attr, t)
        self.varying_cache[cache_key] = varying
        return varying
    
    def traverse_path(self, source: str, target: str, steps: int = 10) -> List[PhiVarying]:
        """Traverse a path between concepts, returning interpolated states."""
        path = []
        for i in range(steps + 1):
            t = i / steps
            varying = self.get_varying(source, target, t)
            if varying:
                path.append(varying)
        return path
    
    def find_valid_path(self, source: str, target: str, 
                        threshold: float = 0.5) -> Tuple[bool, List[PhiVarying]]:
        """Check if a valid path exists between concepts through φ-space."""
        path = self.traverse_path(source, target, steps=20)
        if not path:
            return False, []
        
        phi_target = self.uniforms['phi']
        all_valid = all(abs(v.interpolated_phi - phi_target) < threshold for v in path)
        return all_valid, path
    
    # ============================================================
    # NAVIGATION
    # ============================================================
    
    def get_current_position(self) -> Optional[Position]:
        """Get current position in φ-space."""
        return self.current_position
    
    def navigate_to(self, concept: str) -> Optional[Position]:
        """Navigate to a concept."""
        emb = self._get_embedding(concept)
        if emb is None:
            return None
        
        self.current_position = Position(
            vector=emb,
            phi_level=self._get_phi_level(emb),
            layer=0,  # Embedding layer
            confidence=1.0
        )
        self.navigation_history.append(concept)
        return self.current_position
    
    def navigate_by_vector(self, direction: np.ndarray, distance: float = 1.0) -> Position:
        """Move in a specific direction."""
        if self.current_position is None:
            raise ValueError("No current position - navigate to a concept first")
        
        new_vec = self.current_position.vector + distance * direction
        self.current_position = Position(
            vector=new_vec,
            phi_level=self._get_phi_level(new_vec),
            layer=0,
            confidence=0.8  # Lower confidence for vector navigation
        )
        return self.current_position
    
    def search(self, query: str, k: int = 10) -> List[Concept]:
        """Search for concepts matching query."""
        # Get query embedding
        emb = self._get_embedding(query)
        if emb is None:
            return []
        
        neighbors = self._find_neighbors(emb, k=k)
        
        results = []
        for name, sim in neighbors:
            concept_emb = self._get_embedding(name)
            if concept_emb is not None:
                results.append(Concept(
                    name=name,
                    position=Position(
                        vector=concept_emb,
                        phi_level=self._get_phi_level(concept_emb),
                        layer=0,
                        confidence=sim
                    ),
                    neighbors=[],
                    is_custom=name in self.custom_concepts
                ))
        
        return results
    
    def bookmark(self, name: str) -> None:
        """Save current position."""
        if self.current_position is None:
            raise ValueError("No current position to bookmark")
        self.bookmarks[name] = self.current_position
    
    def goto_bookmark(self, name: str) -> Optional[Position]:
        """Return to a saved position."""
        if name in self.bookmarks:
            self.current_position = self.bookmarks[name]
            return self.current_position
        return None
    
    # ============================================================
    # CRUD OPERATIONS
    # ============================================================
    
    def create_concept(self, name: str, parents: List[str], 
                       weights: Optional[List[float]] = None) -> Optional[Concept]:
        """Create new concept from weighted parents."""
        if weights is None:
            weights = [1.0 / len(parents)] * len(parents)
        
        parent_embs = []
        for p in parents:
            emb = self._get_embedding(p)
            if emb is None:
                return None
            parent_embs.append(emb)
        
        # Weighted sum
        new_vec = sum(w * e for w, e in zip(weights, parent_embs))
        
        # Validate through bottleneck
        validation = self.validate_idea(f"The concept {name} combining {', '.join(parents)}")
        
        if not validation.is_valid:
            print(f"Warning: Concept may not be valid (φ={validation.phi_level:.3f})")
        
        # Store
        self.custom_concepts[name] = new_vec
        
        # Log modification
        self.modification_log.append(Modification(
            id=f"create_{name}_{datetime.now().timestamp()}",
            operation='create',
            concept=name,
            old_position=None,
            new_position=new_vec,
            timestamp=datetime.now().isoformat(),
            validated=validation.is_valid,
            executed=True
        ))
        
        return Concept(
            name=name,
            position=Position(
                vector=new_vec,
                phi_level=self._get_phi_level(new_vec),
                layer=0,
                confidence=validation.coherence_score
            ),
            neighbors=self._find_neighbors(new_vec, exclude=[name]),
            is_custom=True
        )
    
    def create_by_analogy(self, A: str, B: str, C: str, name: Optional[str] = None) -> Optional[Concept]:
        """Create concept via analogy: A is to C as new is to B."""
        emb_A = self._get_embedding(A)
        emb_B = self._get_embedding(B)
        emb_C = self._get_embedding(C)
        
        if emb_A is None or emb_B is None or emb_C is None:
            return None
        
        # Analogy: new = A + (B - C)
        new_vec = emb_A + (emb_B - emb_C)
        
        if name is None:
            name = f"{A}_{B}_{C}_analogy"
        
        self.custom_concepts[name] = new_vec
        
        return Concept(
            name=name,
            position=Position(
                vector=new_vec,
                phi_level=self._get_phi_level(new_vec),
                layer=0,
                confidence=0.8
            ),
            neighbors=self._find_neighbors(new_vec, exclude=[name]),
            is_custom=True
        )
    
    def read_concept(self, name: str) -> Optional[Concept]:
        """Read concept details."""
        emb = self._get_embedding(name)
        if emb is None:
            return None
        
        return Concept(
            name=name,
            position=Position(
                vector=emb,
                phi_level=self._get_phi_level(emb),
                layer=0,
                confidence=1.0
            ),
            neighbors=self._find_neighbors(emb, exclude=[name]),
            is_custom=name in self.custom_concepts
        )
    
    def update_concept(self, name: str, old_property: str, new_property: str, 
                       alpha: float = 0.5) -> Optional[Concept]:
        """Update concept by translating in direction of change."""
        old_emb = self._get_embedding(name)
        old_prop_emb = self._get_embedding(old_property)
        new_prop_emb = self._get_embedding(new_property)
        
        if old_emb is None or old_prop_emb is None or new_prop_emb is None:
            return None
        
        # Direction of change
        delta = new_prop_emb - old_prop_emb
        new_vec = old_emb + alpha * delta
        
        # Validate
        validation = self.validate_position(new_vec)
        
        # Store modification
        self.modifications[name] = new_vec
        
        self.modification_log.append(Modification(
            id=f"update_{name}_{datetime.now().timestamp()}",
            operation='update',
            concept=name,
            old_position=old_emb,
            new_position=new_vec,
            timestamp=datetime.now().isoformat(),
            validated=validation.is_valid,
            executed=True
        ))
        
        return Concept(
            name=name,
            position=Position(
                vector=new_vec,
                phi_level=self._get_phi_level(new_vec),
                layer=0,
                confidence=validation.coherence_score
            ),
            neighbors=self._find_neighbors(new_vec, exclude=[name]),
            is_custom=False,
            modified_at=datetime.now().isoformat()
        )
    
    def delete_concept(self, name: str, method: str = 'isolate', beta: float = 2.0) -> bool:
        """Delete/isolate a concept."""
        if method == 'remove' and name in self.custom_concepts:
            del self.custom_concepts[name]
            return True
        
        emb = self._get_embedding(name)
        if emb is None:
            return False
        
        if method == 'isolate':
            neighbors = self._find_neighbors(emb, k=5, exclude=[name])
            neighbor_embs = [self._get_embedding(n) for n, _ in neighbors]
            neighbor_embs = [e for e in neighbor_embs if e is not None]
            
            if neighbor_embs:
                cluster_center = np.mean(neighbor_embs, axis=0)
                deletion_vector = emb - cluster_center
                isolated = emb + beta * deletion_vector
            else:
                isolated = emb * 0.01
            
            self.modifications[name] = isolated
        
        elif method == 'null':
            self.modifications[name] = emb * 0.001
        
        self.modification_log.append(Modification(
            id=f"delete_{name}_{datetime.now().timestamp()}",
            operation='delete',
            concept=name,
            old_position=emb,
            new_position=self.modifications.get(name, emb * 0.001),
            timestamp=datetime.now().isoformat(),
            validated=True,
            executed=True
        ))
        
        return True
    
    # ============================================================
    # IDEA GENERATION
    # ============================================================
    
    def combine_concepts(self, concepts: List[str], 
                         weights: Optional[List[float]] = None) -> Optional[Concept]:
        """Combine multiple concepts into a novel idea."""
        if weights is None:
            weights = [1.0 / len(concepts)] * len(concepts)
        
        embs = []
        for c in concepts:
            emb = self._get_embedding(c)
            if emb is None:
                return None
            embs.append(emb)
        
        combined = sum(w * e for w, e in zip(weights, embs))
        
        # Generate a name for the combination
        name = f"{'_'.join(concepts[:3])}_fusion"
        
        # Validate
        validation = self.validate_position(combined)
        
        return Concept(
            name=name,
            position=Position(
                vector=combined,
                phi_level=self._get_phi_level(combined),
                layer=0,
                confidence=validation.coherence_score
            ),
            neighbors=self._find_neighbors(combined),
            is_custom=True
        )
    
    def explore_region(self, center: str, radius: float = 0.5, k: int = 20) -> List[Concept]:
        """Explore concepts within radius of center."""
        center_emb = self._get_embedding(center)
        if center_emb is None:
            return []
        
        # Find neighbors
        neighbors = self._find_neighbors(center_emb, k=k * 2)
        
        # Filter by "radius" (similarity threshold)
        threshold = 1.0 - radius
        results = []
        for name, sim in neighbors:
            if sim >= threshold:
                emb = self._get_embedding(name)
                if emb is not None:
                    results.append(Concept(
                        name=name,
                        position=Position(
                            vector=emb,
                            phi_level=self._get_phi_level(emb),
                            layer=0,
                            confidence=sim
                        ),
                        neighbors=[],
                        is_custom=name in self.custom_concepts
                    ))
            if len(results) >= k:
                break
        
        return results
    
    def generate_novel_idea(self, seed_concepts: List[str], 
                            novelty: float = 0.5) -> Tuple[str, ValidationResult]:
        """Generate a novel idea from seed concepts."""
        prompt = f"A genuinely novel idea connecting {', '.join(seed_concepts)} would be:"
        
        # Higher temperature for more novelty
        temp = 0.7 + novelty * 0.5
        idea = self._generate(prompt, max_tokens=150, temp=temp)
        
        # Validate
        validation = self.validate_idea(idea)
        
        return idea, validation
    
    # ============================================================
    # VALIDATION (BOTTLENECK)
    # ============================================================
    
    def validate_position(self, position: np.ndarray) -> ValidationResult:
        """Validate a position through the bottleneck."""
        phi = self._get_phi_level(position)
        distance = abs(phi - PHI)
        
        is_valid = self.VALID_PHI_RANGE[0] <= phi <= self.VALID_PHI_RANGE[1]
        
        # Coherence with neighbors
        neighbors = self._find_neighbors(position, k=5)
        coherence = np.mean([sim for _, sim in neighbors]) if neighbors else 0.0
        
        warnings = []
        if not (self.WARNING_PHI_RANGE[0] <= phi <= self.WARNING_PHI_RANGE[1]):
            warnings.append(f"φ-level {phi:.3f} outside optimal range")
        if coherence < 0.3:
            warnings.append(f"Low coherence with neighbors ({coherence:.3f})")
        
        return ValidationResult(
            is_valid=is_valid,
            phi_level=phi,
            distance_from_phi=distance,
            coherence_score=coherence,
            warnings=warnings
        )
    
    def validate_idea(self, idea: str) -> ValidationResult:
        """Validate an idea through the φ-bottleneck."""
        # Get trajectory
        trajectory = self._get_trajectory(idea[:500])
        
        # Check layer 27
        phi_27 = self._get_phi_level(trajectory[27])
        distance = abs(phi_27 - PHI)
        
        is_valid = self.VALID_PHI_RANGE[0] <= phi_27 <= self.VALID_PHI_RANGE[1]
        
        # Coherence
        final_hidden = trajectory[-1]
        neighbors = self._find_neighbors(final_hidden, k=5)
        coherence = np.mean([sim for _, sim in neighbors]) if neighbors else 0.0
        
        warnings = []
        if not is_valid:
            warnings.append(f"φ-27 level {phi_27:.3f} outside valid range")
        
        return ValidationResult(
            is_valid=is_valid,
            phi_level=phi_27,
            distance_from_phi=distance,
            coherence_score=coherence,
            warnings=warnings
        )
    
    # ============================================================
    # INTROSPECTION
    # ============================================================
    
    def get_active_concepts(self, text: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get currently active concepts."""
        if text is None:
            if self.current_position is None:
                return []
            return self._find_neighbors(self.current_position.vector, k=10)
        
        trajectory = self._get_trajectory(text)
        final = trajectory[-1]
        return self._find_neighbors(final, k=10)
    
    def trace_reasoning(self, text: str) -> List[Dict]:
        """Trace the reasoning path through layers."""
        trajectory = self._get_trajectory(text)
        
        trace = []
        for i, hidden in enumerate(trajectory):
            neighbors = self._find_neighbors(hidden, k=3)
            trace.append({
                'layer': i,
                'phi_level': self._get_phi_level(hidden),
                'top_concepts': neighbors
            })
        
        return trace
    
    def find_uncertainty(self, concepts: List[str]) -> List[Tuple[str, float]]:
        """Find concepts with high uncertainty."""
        uncertainties = []
        
        for c in concepts:
            emb = self._get_embedding(c)
            if emb is None:
                continue
            
            neighbors = self._find_neighbors(emb, k=5, exclude=[c])
            if neighbors:
                # Uncertainty = variance in neighbor similarities
                sims = [s for _, s in neighbors]
                uncertainty = np.std(sims)
                uncertainties.append((c, uncertainty))
        
        uncertainties.sort(key=lambda x: -x[1])
        return uncertainties
    
    def find_gaps(self, region_center: str, radius: float = 0.5) -> List[str]:
        """Find potential knowledge gaps in a region."""
        # This is a heuristic - find areas with low concept density
        center_emb = self._get_embedding(region_center)
        if center_emb is None:
            return []
        
        # Generate random directions
        gaps = []
        for _ in range(10):
            direction = np.random.randn(self.hidden_size)
            direction = direction / np.linalg.norm(direction)
            
            probe = center_emb + radius * direction
            neighbors = self._find_neighbors(probe, k=3)
            
            if neighbors and neighbors[0][1] < 0.3:  # Low similarity = potential gap
                # Describe the gap
                gap_desc = f"Gap near {region_center} in direction of {neighbors[0][0]}"
                gaps.append(gap_desc)
        
        return gaps
    
    def introspect(self, text: Optional[str] = None) -> IntrospectionResult:
        """Full introspection of current state."""
        active = self.get_active_concepts(text)
        
        # Find uncertainty in active concepts
        concept_names = [c for c, _ in active]
        uncertainty = self.find_uncertainty(concept_names)
        
        # Find gaps
        if active:
            gaps = self.find_gaps(active[0][0])
        else:
            gaps = []
        
        # Overall coherence
        if self.current_position is not None:
            validation = self.validate_position(self.current_position.vector)
            coherence = validation.coherence_score
        else:
            coherence = 0.0
        
        return IntrospectionResult(
            active_concepts=active,
            uncertainty_regions=uncertainty,
            knowledge_gaps=gaps,
            detected_biases=[],  # Would need more sophisticated analysis
            overall_coherence=coherence
        )
    
    # ============================================================
    # GOALS
    # ============================================================
    
    def set_goal(self, description: str, target_concepts: List[str]) -> Goal:
        """Set a goal to achieve."""
        goal_id = f"goal_{datetime.now().timestamp()}"
        goal = Goal(
            id=goal_id,
            description=description,
            target_concepts=target_concepts,
            status='pending'
        )
        self.goals[goal_id] = goal
        return goal
    
    def plan_path(self, goal_id: str) -> List[str]:
        """Plan a path to reach the goal."""
        if goal_id not in self.goals:
            return []
        
        goal = self.goals[goal_id]
        
        # Simple planning: navigate through target concepts
        steps = []
        for concept in goal.target_concepts:
            steps.append(f"NAVIGATE {concept}")
            steps.append(f"READ {concept}")
        
        steps.append(f"COMBINE {goal.target_concepts}")
        steps.append("VALIDATE result")
        
        return steps
    
    # ============================================================
    # COMMAND PARSER
    # ============================================================
    
    def _build_command_parser(self) -> Dict[str, Callable]:
        """Build command parser."""
        return {
            'NAVIGATE': self._cmd_navigate,
            'SEARCH': self._cmd_search,
            'BOOKMARK': self._cmd_bookmark,
            'GOTO': self._cmd_goto,
            'CREATE': self._cmd_create,
            'READ': self._cmd_read,
            'UPDATE': self._cmd_update,
            'DELETE': self._cmd_delete,
            'COMBINE': self._cmd_combine,
            'EXPLORE': self._cmd_explore,
            'VALIDATE': self._cmd_validate,
            'ACTIVE': self._cmd_active,
            'TRACE': self._cmd_trace,
            'GAPS': self._cmd_gaps,
            'INTROSPECT': self._cmd_introspect,
            'GOAL': self._cmd_goal,
            'PLAN': self._cmd_plan,
            'ROLLBACK': self._cmd_rollback,
            'LOG': self._cmd_log,
            'HELP': self._cmd_help,
        }
    
    def execute(self, command: str) -> Any:
        """Execute a command string."""
        parts = command.strip().split(maxsplit=1)
        if not parts:
            return "Empty command"
        
        cmd = parts[0].upper()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in self.commands:
            try:
                return self.commands[cmd](args)
            except Exception as e:
                return f"Error: {e}"
        else:
            return f"Unknown command: {cmd}. Type HELP for available commands."
    
    def _cmd_navigate(self, args: str) -> str:
        concept = args.strip().strip('"\'')
        pos = self.navigate_to(concept)
        if pos:
            return f"Navigated to '{concept}' (φ={pos.phi_level:.3f})"
        return f"Concept '{concept}' not found"
    
    def _cmd_search(self, args: str) -> str:
        query = args.strip().strip('"\'')
        results = self.search(query)
        if results:
            return "\n".join([f"  {c.name} (φ={c.position.phi_level:.3f})" for c in results[:5]])
        return "No results"
    
    def _cmd_bookmark(self, args: str) -> str:
        name = args.strip().strip('"\'')
        self.bookmark(name)
        return f"Bookmarked as '{name}'"
    
    def _cmd_goto(self, args: str) -> str:
        name = args.strip().strip('"\'')
        pos = self.goto_bookmark(name)
        if pos:
            return f"Returned to bookmark '{name}'"
        return f"Bookmark '{name}' not found"
    
    def _cmd_create(self, args: str) -> str:
        # Parse: name FROM [parents] - handle quoted or unquoted names
        # Try quoted name first: "name" FROM [...]
        match = re.match(r'"([^"]+)"\s+FROM\s+\[([^\]]+)\]', args, re.IGNORECASE)
        if not match:
            # Try unquoted: name FROM [...]
            match = re.match(r'(\S+)\s+FROM\s+\[([^\]]+)\]', args, re.IGNORECASE)
        
        if match:
            name = match.group(1).strip('"\'').replace(' ', '_')
            parents = [p.strip().strip('"\'') for p in match.group(2).split(',')]
            concept = self.create_concept(name, parents)
            if concept:
                return f"Created '{name}' (φ={concept.position.phi_level:.3f})"
            return "Failed to create concept"
        return "Usage: CREATE name FROM [parent1, parent2, ...]"
    
    def _cmd_read(self, args: str) -> str:
        concept = args.strip().strip('"\'')
        c = self.read_concept(concept)
        if c:
            neighbors = ", ".join([n for n, _ in c.neighbors[:3]])
            return f"'{c.name}': φ={c.position.phi_level:.3f}, neighbors=[{neighbors}]"
        return f"Concept '{concept}' not found"
    
    def _cmd_update(self, args: str) -> str:
        # Parse: concept SHIFT old TO new
        match = re.match(r'(\S+)\s+SHIFT\s+(\S+)\s+TO\s+(\S+)', args, re.IGNORECASE)
        if match:
            concept = match.group(1).strip('"\'')
            old_prop = match.group(2).strip('"\'')
            new_prop = match.group(3).strip('"\'')
            c = self.update_concept(concept, old_prop, new_prop)
            if c:
                return f"Updated '{concept}' (new φ={c.position.phi_level:.3f})"
            return "Failed to update"
        return "Usage: UPDATE concept SHIFT old TO new"
    
    def _cmd_delete(self, args: str) -> str:
        concept = args.strip().strip('"\'')
        if self.delete_concept(concept):
            return f"Deleted/isolated '{concept}'"
        return f"Failed to delete '{concept}'"
    
    def _cmd_combine(self, args: str) -> str:
        # Parse: [concept1, concept2, ...]
        match = re.match(r'\[([^\]]+)\]', args)
        if match:
            concepts = [c.strip().strip('"\'') for c in match.group(1).split(',')]
            result = self.combine_concepts(concepts)
            if result:
                neighbors = ", ".join([n for n, _ in result.neighbors[:3]])
                return f"Combined into '{result.name}' (φ={result.position.phi_level:.3f}), neighbors=[{neighbors}]"
            return "Failed to combine"
        return "Usage: COMBINE [concept1, concept2, ...]"
    
    def _cmd_explore(self, args: str) -> str:
        concept = args.strip().strip('"\'')
        results = self.explore_region(concept)
        if results:
            return "\n".join([f"  {c.name} (sim={c.position.confidence:.3f})" for c in results[:10]])
        return "No results"
    
    def _cmd_validate(self, args: str) -> str:
        idea = args.strip().strip('"\'')
        result = self.validate_idea(idea)
        status = "VALID" if result.is_valid else "INVALID"
        return f"[{status}] φ-27={result.phi_level:.3f}, coherence={result.coherence_score:.3f}"
    
    def _cmd_active(self, args: str) -> str:
        active = self.get_active_concepts(args if args else None)
        if active:
            return "\n".join([f"  {c}: {s:.3f}" for c, s in active[:10]])
        return "No active concepts"
    
    def _cmd_trace(self, args: str) -> str:
        text = args.strip().strip('"\'')
        trace = self.trace_reasoning(text)
        # Show key layers
        key_layers = [0, 7, 14, 21, 27, 28]
        lines = []
        for t in trace:
            if t['layer'] in key_layers:
                concepts = ", ".join([c for c, _ in t['top_concepts'][:2]])
                lines.append(f"  Layer {t['layer']}: φ={t['phi_level']:.3f} [{concepts}]")
        return "\n".join(lines)
    
    def _cmd_gaps(self, args: str) -> str:
        concept = args.strip().strip('"\'') if args else None
        if concept:
            gaps = self.find_gaps(concept)
            if gaps:
                return "\n".join([f"  • {g}" for g in gaps])
        return "No gaps found (or specify a concept)"
    
    def _cmd_introspect(self, args: str) -> str:
        result = self.introspect(args if args else None)
        lines = [
            f"Active concepts: {len(result.active_concepts)}",
            f"Overall coherence: {result.overall_coherence:.3f}",
            f"Knowledge gaps: {len(result.knowledge_gaps)}",
        ]
        if result.active_concepts:
            lines.append(f"Top active: {result.active_concepts[0][0]}")
        return "\n".join(lines)
    
    def _cmd_goal(self, args: str) -> str:
        # Parse: "description" TARGET [concepts]
        match = re.match(r'"([^"]+)"\s+TARGET\s+\[([^\]]+)\]', args)
        if match:
            desc = match.group(1)
            targets = [t.strip().strip('"\'') for t in match.group(2).split(',')]
            goal = self.set_goal(desc, targets)
            return f"Goal set: {goal.id}"
        return 'Usage: GOAL "description" TARGET [concept1, concept2]'
    
    def _cmd_plan(self, args: str) -> str:
        goal_id = args.strip()
        steps = self.plan_path(goal_id)
        if steps:
            return "\n".join([f"  {i+1}. {s}" for i, s in enumerate(steps)])
        return "Goal not found or no plan available"
    
    def _cmd_rollback(self, args: str) -> str:
        if not self.modification_log:
            return "No modifications to rollback"
        
        last_mod = self.modification_log[-1]
        if last_mod.old_position is not None:
            if last_mod.concept in self.modifications:
                self.modifications[last_mod.concept] = last_mod.old_position
            elif last_mod.concept in self.custom_concepts:
                self.custom_concepts[last_mod.concept] = last_mod.old_position
            self.modification_log.pop()
            return f"Rolled back: {last_mod.operation} on '{last_mod.concept}'"
        return "Cannot rollback (no previous state)"
    
    def _cmd_log(self, args: str) -> str:
        n = int(args) if args.strip().isdigit() else 5
        if not self.modification_log:
            return "No modifications logged"
        
        lines = []
        for mod in self.modification_log[-n:]:
            lines.append(f"  [{mod.timestamp[:19]}] {mod.operation}: {mod.concept}")
        return "\n".join(lines)
    
    def _cmd_help(self, args: str) -> str:
        return """Available commands:
  NAVIGATE <concept>              - Move to concept
  SEARCH <query>                  - Search for concepts
  BOOKMARK <name>                 - Save current position
  GOTO <bookmark>                 - Return to bookmark
  CREATE name FROM [parents]      - Create new concept
  READ <concept>                  - Read concept details
  UPDATE concept SHIFT old TO new - Update concept
  DELETE <concept>                - Delete/isolate concept
  COMBINE [concepts]              - Combine concepts
  EXPLORE <concept>               - Explore region
  VALIDATE <idea>                 - Validate through bottleneck
  ACTIVE [text]                   - Show active concepts
  TRACE <text>                    - Trace reasoning path
  GAPS <concept>                  - Find knowledge gaps
  INTROSPECT [text]               - Full introspection
  GOAL "desc" TARGET [concepts]   - Set a goal
  PLAN <goal_id>                  - Plan path to goal
  ROLLBACK                        - Undo last modification
  LOG [n]                         - Show modification log
  HELP                            - Show this help"""
    
    # ============================================================
    # SELF-CONTROL LOOP
    # ============================================================
    
    def self_control_step(self, objective: str, history: List[Dict] = None) -> Tuple[str, str]:
        """
        Let the model decide what command to execute next.
        
        This is the key method for AI self-control.
        """
        history = history or []
        
        # Build history string
        history_str = ""
        if history:
            history_str = "\nPrevious steps:\n"
            for h in history[-3:]:  # Last 3 steps
                history_str += f"  {h['step']}. {h['command']} -> {h['result'][:50]}...\n"
        
        # Build context
        context = f"""You control a Conceptual Nexus interface. Your goal: {objective}

State: position={self.current_position.to_dict() if self.current_position else 'None'}, concepts={list(self.custom_concepts.keys())[:5]}
{history_str}
Available commands:
NAVIGATE <concept> | SEARCH <term> | READ <concept> | EXPLORE <concept>
CREATE <name> FROM [a, b, c] | COMBINE [a, b, c] | VALIDATE "<hypothesis>"

Output exactly ONE command to advance toward your goal. No explanations, just the command.
Command:"""

        # Let model decide
        command = self._generate(context, max_tokens=40, temp=0.6)
        
        # Clean up response - extract just the command
        command = command.strip().split('\n')[0].strip()
        
        # Remove common prefixes the model might add
        for prefix in ['Command:', 'Next:', 'Execute:', '>', '-', '*', '1.', '2.']:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
        
        # Remove any explanation after the command
        if ' - ' in command:
            command = command.split(' - ')[0].strip()
        if ' (' in command:
            command = command.split(' (')[0].strip()
        
        # Handle "e.g." or example prefixes
        if command.lower().startswith('e.g'):
            command = command.split(',')[1].strip() if ',' in command else 'NAVIGATE music'
        
        # Execute
        result = self.execute(command)
        
        return command, str(result)
    
    def autonomous_session(self, objective: str, max_steps: int = 10) -> List[Dict]:
        """
        Run an autonomous session where the model controls itself.
        """
        history = []
        
        print(f"\n{'='*60}")
        print(f"AUTONOMOUS SESSION: {objective}")
        print(f"{'='*60}")
        
        for step in range(max_steps):
            command, result = self.self_control_step(objective, history)
            
            history.append({
                'step': step + 1,
                'command': command,
                'result': result
            })
            
            print(f"\nStep {step + 1}:")
            print(f"  Command: {command}")
            print(f"  Result: {result[:200]}...")
            
            # Check if we should stop
            if 'GOAL' in command.upper() and 'completed' in result.lower():
                break
        
        return history
    
    def reflect_on_session(self, history: List[Dict], objective: str) -> str:
        """Let the model reflect on what it learned from the session."""
        history_str = "\n".join([
            f"{h['step']}. {h['command']} -> {h['result'][:100]}" 
            for h in history
        ])
        
        prompt = f"""You just completed an autonomous exploration session.

Objective: {objective}

Actions taken:
{history_str}

Reflect on this session:
1. What did you discover?
2. What worked well?
3. What would you do differently?
4. What new questions emerged?

Reflection:"""
        
        return self._generate(prompt, max_tokens=300, temp=0.85)


# ============================================================
# INTERACTIVE REPL
# ============================================================

def interactive_repl(nexus: ConceptualNexus):
    """Interactive REPL for Conceptual Nexus."""
    print("\n" + "="*60)
    print("CONCEPTUAL NEXUS - Interactive Mode")
    print("="*60)
    print("Type commands or 'HELP' for available commands.")
    print("Type 'AUTO <objective>' for autonomous mode.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("nexus> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.upper().startswith('AUTO '):
                objective = user_input[5:].strip()
                nexus.autonomous_session(objective, max_steps=5)
            else:
                result = nexus.execute(user_input)
                print(result)
                
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Demo Conceptual Nexus with AI self-control."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    
    nexus = ConceptualNexus(model, tokenizer)
    
    print("\n" + "="*60)
    print("CONCEPTUAL NEXUS - AI SELF-CONTROL DEMO")
    print("="*60)
    
    # Part 1: Basic commands
    print("\n--- PART 1: Basic Commands ---")
    demos = [
        'NAVIGATE creativity',
        'READ creativity',
        'EXPLORE creativity',
        'CREATE novel_insight FROM [creativity, logic, intuition]',
        'VALIDATE "creativity emerges from the intersection of logic and intuition"',
    ]
    
    for cmd in demos:
        print(f"\nnexus> {cmd}")
        result = nexus.execute(cmd)
        print(result)
    
    # Part 2: Autonomous exploration
    print("\n" + "="*60)
    print("--- PART 2: Autonomous Exploration ---")
    print("The AI will now control itself to explore a topic.")
    print("="*60)
    
    objective = "Discover a novel connection between music, mathematics, and emotion"
    history = nexus.autonomous_session(objective, max_steps=8)
    
    # Part 3: Reflection
    print("\n" + "="*60)
    print("--- PART 3: AI Reflection ---")
    print("="*60)
    
    reflection = nexus.reflect_on_session(history, objective)
    print(f"\n{reflection}")
    
    # Part 4: Novel idea generation
    print("\n" + "="*60)
    print("--- PART 4: Novel Idea Generation ---")
    print("="*60)
    
    idea, validation = nexus.generate_novel_idea(
        ["time", "consciousness", "geometry"],
        novelty=0.7
    )
    print(f"\nGenerated idea: {idea}")
    print(f"Validation: {validation.to_dict()}")
    
    # Part 5: Introspection
    print("\n" + "="*60)
    print("--- PART 5: Full Introspection ---")
    print("="*60)
    
    intro = nexus.introspect()
    print(f"\nActive concepts: {intro.active_concepts[:5]}")
    print(f"Overall coherence: {intro.overall_coherence:.3f}")
    print(f"Knowledge gaps: {intro.knowledge_gaps[:3]}")
    
    # Summary
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Custom concepts created: {list(nexus.custom_concepts.keys())}")
    print(f"Total modifications: {len(nexus.modification_log)}")
    print(f"Navigation history: {nexus.navigation_history[-5:]}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
