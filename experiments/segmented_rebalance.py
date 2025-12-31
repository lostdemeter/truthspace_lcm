#!/usr/bin/env python3
"""
Segmented Rebalancing with Anchor Points

Inspired by DNA zinc fingers and anchor points:
- Instead of rebalancing the entire structure, we have compartments
- Anchor points are fixed nodes that maintain structural integrity
- Each segment can be rebalanced independently without disturbing others
- Redundancy provides transdimensional error correction (like zeta symmetry)

Key concepts:
1. ANCHORS: High-confidence relationships that don't change (like zinc finger binding sites)
2. SEGMENTS: Groups of related concepts that can be rebalanced together
3. SYMMETRY: Enforce ζ(s) = ζ(1-s) style constraints across dimensions
"""

import json
import numpy as np
import requests
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fully_emergent_chains import FullyEmergentSemanticChain


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"


@dataclass
class Anchor:
    """A fixed relationship that acts as a structural anchor."""
    concept_a: str
    concept_b: str
    relationship: str  # 'opposite', 'similar'
    confidence: float  # 0-1, higher = more fixed
    dimension: int  # Which dimension this anchor constrains


@dataclass 
class Segment:
    """A compartment of related concepts that can be rebalanced together."""
    name: str
    concepts: List[str]
    anchors: List[Anchor]
    center: Optional[np.ndarray] = None  # Centroid of segment in embedding space


class SegmentedStructure:
    """
    Manages structure through segmented compartments with anchor points.
    
    Like DNA:
    - Zinc fingers bind to specific sequences (anchors)
    - Regions between anchors can be modified (segments)
    - Redundancy provides error correction (symmetry constraints)
    """
    
    def __init__(self, semantic_chain: FullyEmergentSemanticChain):
        self.semantic = semantic_chain
        self.anchors: List[Anchor] = []
        self.segments: Dict[str, Segment] = {}
        
        # Track which concepts are "locked" by anchors
        self.locked_concepts: Set[str] = set()
        
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call Ollama API."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.3}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM error: {e}")
        return None
    
    # =========================================================================
    # ANCHOR MANAGEMENT (Zinc Finger Binding Sites)
    # =========================================================================
    
    def discover_anchors(self, min_confidence: float = 0.8) -> List[Anchor]:
        """
        Discover natural anchor points in the structure.
        
        Anchors are relationships that:
        1. Have high semantic confidence (LLM agrees strongly)
        2. Are bidirectional (A opposite B AND B opposite A)
        3. Are stable across dimensions
        """
        print(f"\n{'─'*60}")
        print("DISCOVERING ANCHOR POINTS")
        print(f"{'─'*60}")
        
        discovered = []
        vocabulary = set(self.semantic.groups)
        
        # Test key relationships
        test_pairs = [
            ('hero', 'villain'),
            ('king', 'queen'),
            ('sage', 'child'),
            ('holmes', 'moriarty'),
            ('watson', 'holmes'),
        ]
        
        for a, b in test_pairs:
            if a not in vocabulary or b not in vocabulary:
                continue
            
            confidence = self._evaluate_anchor_confidence(a, b, 'opposite')
            if confidence >= min_confidence:
                # Find which dimension this relationship is strongest on
                dim = self._find_dominant_dimension(a, b)
                
                anchor = Anchor(
                    concept_a=a,
                    concept_b=b,
                    relationship='opposite',
                    confidence=confidence,
                    dimension=dim
                )
                discovered.append(anchor)
                self.locked_concepts.add(a)
                self.locked_concepts.add(b)
                print(f"  Anchor: {a} ↔ {b} (conf={confidence:.2f}, dim={dim})")
        
        self.anchors = discovered
        return discovered
    
    def _evaluate_anchor_confidence(self, a: str, b: str, rel_type: str) -> float:
        """Evaluate confidence that a relationship should be an anchor."""
        prompt = f"""Rate the semantic relationship: "{a.title()} is the {rel_type} of {b.title()}"

Scale: 0.0 (completely wrong) to 1.0 (definitely correct)

Consider:
- Is this a fundamental, well-known opposition?
- Would most people agree?
- Is it stable across contexts?

Answer with just a number between 0.0 and 1.0:"""

        response = self._call_llm(prompt, max_tokens=20)
        if response:
            try:
                # Extract number from response
                for word in response.split():
                    word = word.strip('.,')
                    if '.' in word:
                        val = float(word)
                        if 0 <= val <= 1:
                            return val
            except:
                pass
        return 0.5  # Default moderate confidence
    
    def _find_dominant_dimension(self, a: str, b: str) -> int:
        """Find which dimension best separates two concepts."""
        pos_a = self.semantic.get_position(a)
        pos_b = self.semantic.get_position(b)
        
        if pos_a is None or pos_b is None:
            return 0
        
        # Handle empty or mismatched arrays
        if len(pos_a) == 0 or len(pos_b) == 0:
            return 0
        
        # Ensure same length
        min_len = min(len(pos_a), len(pos_b))
        pos_a = pos_a[:min_len]
        pos_b = pos_b[:min_len]
        
        # Find dimension with maximum separation
        diff = np.abs(pos_a - pos_b)
        if len(diff) == 0:
            return 0
        return int(np.argmax(diff))
    
    def add_anchor(self, a: str, b: str, rel_type: str = 'opposite', 
                   confidence: float = 0.9):
        """Manually add an anchor point."""
        dim = self._find_dominant_dimension(a, b)
        anchor = Anchor(
            concept_a=a,
            concept_b=b,
            relationship=rel_type,
            confidence=confidence,
            dimension=dim
        )
        self.anchors.append(anchor)
        self.locked_concepts.add(a)
        self.locked_concepts.add(b)
        print(f"  Added anchor: {a} ↔ {b}")
    
    # =========================================================================
    # SEGMENT MANAGEMENT (Compartmentalized Regions)
    # =========================================================================
    
    def discover_segments(self) -> Dict[str, Segment]:
        """
        Discover natural segments in the structure.
        
        Segments are groups of concepts that:
        1. Cluster together in embedding space
        2. Share semantic domain
        3. Are bounded by anchors
        """
        print(f"\n{'─'*60}")
        print("DISCOVERING SEGMENTS")
        print(f"{'─'*60}")
        
        if self.semantic.U is None:
            return {}
        
        # Simple clustering by dominant dimension
        n_dims = min(5, self.semantic.U.shape[1])
        
        # Group concepts by their dominant dimension
        dim_groups: Dict[int, List[str]] = defaultdict(list)
        
        for i, concept in enumerate(self.semantic.groups):
            if concept in self.locked_concepts:
                continue  # Skip anchored concepts
            
            pos = self.semantic.U[i, :n_dims]
            dominant = int(np.argmax(np.abs(pos)))
            sign = 'pos' if pos[dominant] > 0 else 'neg'
            key = dominant * 2 + (1 if sign == 'pos' else 0)
            dim_groups[key].append(concept)
        
        # Create segments from groups
        segment_names = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 
                        'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa']
        
        for i, (key, concepts) in enumerate(dim_groups.items()):
            if len(concepts) < 2:
                continue
            
            name = segment_names[i % len(segment_names)]
            
            # Find anchors that bound this segment
            segment_anchors = [
                a for a in self.anchors 
                if a.concept_a in concepts or a.concept_b in concepts
            ]
            
            # Compute centroid
            indices = [self.semantic.groups.index(c) for c in concepts 
                      if c in self.semantic.groups]
            if indices:
                center = np.mean(self.semantic.U[indices, :n_dims], axis=0)
            else:
                center = None
            
            segment = Segment(
                name=name,
                concepts=concepts,
                anchors=segment_anchors,
                center=center
            )
            self.segments[name] = segment
            print(f"  Segment {name}: {len(concepts)} concepts")
        
        return self.segments
    
    # =========================================================================
    # SEGMENTED REBALANCING
    # =========================================================================
    
    def rebalance_segment(self, segment_name: str) -> Dict:
        """
        Rebalance a single segment without disturbing others.
        
        Like modifying DNA between zinc finger binding sites:
        - Anchors remain fixed
        - Only concepts within segment are adjusted
        - Maintains segment's relationship to anchors
        """
        if segment_name not in self.segments:
            return {'error': f'Segment {segment_name} not found'}
        
        segment = self.segments[segment_name]
        print(f"\n  Rebalancing segment: {segment_name}")
        
        if self.semantic.U is None:
            return {'error': 'No embedding matrix'}
        
        adjustments = 0
        
        # Get indices for segment concepts
        concept_indices = {}
        for concept in segment.concepts:
            if concept in self.semantic.groups:
                concept_indices[concept] = self.semantic.groups.index(concept)
        
        # Adjust positions relative to segment center
        if segment.center is not None:
            n_dims = len(segment.center)
            
            for concept, idx in concept_indices.items():
                if concept in self.locked_concepts:
                    continue  # Don't touch anchored concepts
                
                # Gentle pull toward segment center (cohesion)
                current = self.semantic.U[idx, :n_dims]
                direction = segment.center - current
                
                # Small adjustment (10% toward center)
                self.semantic.U[idx, :n_dims] += direction * 0.1
                adjustments += 1
        
        # Enforce anchor constraints within segment
        for anchor in segment.anchors:
            self._enforce_anchor(anchor)
        
        return {
            'segment': segment_name,
            'concepts_adjusted': adjustments,
            'anchors_enforced': len(segment.anchors)
        }
    
    def rebalance_all_segments(self) -> Dict:
        """Rebalance all segments independently."""
        print(f"\n{'─'*60}")
        print("SEGMENTED REBALANCING")
        print(f"{'─'*60}")
        
        results = {}
        for name in self.segments:
            results[name] = self.rebalance_segment(name)
        
        # Apply symmetry constraints (transdimensional error correction)
        self._apply_symmetry_constraints()
        
        return results
    
    def _enforce_anchor(self, anchor: Anchor):
        """Enforce a single anchor constraint."""
        if self.semantic.U is None:
            return
        
        a_idx = None
        b_idx = None
        
        for i, g in enumerate(self.semantic.groups):
            if g == anchor.concept_a:
                a_idx = i
            if g == anchor.concept_b:
                b_idx = i
        
        if a_idx is None or b_idx is None:
            return
        
        dim = anchor.dimension
        
        if anchor.relationship == 'opposite':
            # Ensure opposite signs on anchor dimension
            if self.semantic.U[a_idx, dim] * self.semantic.U[b_idx, dim] > 0:
                # Same sign - flip one (the one with smaller magnitude)
                if abs(self.semantic.U[a_idx, dim]) < abs(self.semantic.U[b_idx, dim]):
                    self.semantic.U[a_idx, dim] *= -1
                else:
                    self.semantic.U[b_idx, dim] *= -1
    
    # =========================================================================
    # TRANSDIMENSIONAL ERROR CORRECTION (Zeta Symmetry)
    # =========================================================================
    
    def _apply_symmetry_constraints(self):
        """
        Apply zeta-like symmetry constraints for error correction.
        
        Like ζ(s) = ζ(1-s), we enforce that the structure has
        symmetric properties across dimensions.
        
        This is the "redundancy" that provides error correction.
        """
        if self.semantic.U is None:
            return
        
        print(f"\n  Applying symmetry constraints...")
        
        n_dims = self.semantic.U.shape[1]
        
        # For each pair of dimensions, enforce approximate symmetry
        # This is like the functional equation of zeta
        for d1 in range(min(3, n_dims)):
            d2 = n_dims - 1 - d1  # Mirror dimension
            if d1 >= d2:
                continue
            
            # The variance on d1 should roughly equal variance on d2
            var1 = np.var(self.semantic.U[:, d1])
            var2 = np.var(self.semantic.U[:, d2])
            
            if var1 > 0 and var2 > 0:
                # Scale to equalize variances (soft constraint)
                ratio = np.sqrt(var1 / var2)
                if ratio > 1.5:
                    self.semantic.U[:, d1] /= np.sqrt(ratio)
                elif ratio < 0.67:
                    self.semantic.U[:, d2] /= np.sqrt(1/ratio)
        
        # Enforce that anchored pairs maintain their relationship
        for anchor in self.anchors:
            self._enforce_anchor(anchor)
    
    def compute_redundancy_score(self) -> float:
        """
        Compute how much redundancy/error-correction the structure has.
        
        Higher redundancy = more robust to perturbations.
        Like DNA codon redundancy.
        """
        if self.semantic.U is None:
            return 0.0
        
        # Measure: how many dimensions encode similar information?
        # High correlation between dimensions = redundancy
        n_dims = min(5, self.semantic.U.shape[1])
        
        correlations = []
        for d1 in range(n_dims):
            for d2 in range(d1 + 1, n_dims):
                corr = np.abs(np.corrcoef(
                    self.semantic.U[:, d1], 
                    self.semantic.U[:, d2]
                )[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            # Average correlation = redundancy
            return float(np.mean(correlations))
        return 0.0


def test_segmented_rebalance():
    """Test the segmented rebalancing system."""
    print("=" * 70)
    print("SEGMENTED REBALANCING WITH ANCHOR POINTS")
    print("(Inspired by DNA Zinc Fingers)")
    print("=" * 70)
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama not running!")
            return
        print("Ollama is running")
    except:
        print("Ollama not available!")
        return
    
    # Create and train model
    chain = FullyEmergentSemanticChain()
    
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    print(f"\nLoading corpus: {corpus_path}")
    count = chain.ingest_corpus(str(corpus_path))
    print(f"  Loaded {count} items")
    
    print("\nInitial training...")
    chain.learn_dimensions()
    print(f"  Dimensions: {len(chain.dimensions)}")
    print(f"  Concepts: {len(chain.groups)}")
    
    # Create segmented structure
    structure = SegmentedStructure(chain)
    
    # Phase 1: Discover anchors (zinc finger binding sites)
    anchors = structure.discover_anchors(min_confidence=0.6)
    
    # Manually add some known anchors if not discovered
    if not any(a.concept_a == 'hero' for a in anchors):
        structure.add_anchor('hero', 'villain', 'opposite', 0.95)
    
    # Phase 2: Discover segments (compartmentalized regions)
    segments = structure.discover_segments()
    
    # Phase 3: Show initial state
    print(f"\n{'─'*60}")
    print("INITIAL STATE")
    print(f"{'─'*60}")
    
    test_concepts = ['hero', 'villain', 'holmes', 'watson', 'sage', 'king']
    for concept in test_concepts:
        opposite = chain.find_opposite(concept)
        opp_str = opposite[0] if opposite else "none"
        locked = "🔒" if concept in structure.locked_concepts else ""
        print(f"  {concept}{locked}: opposite={opp_str}")
    
    # Phase 4: Segmented rebalancing
    print(f"\n{'='*60}")
    print("SEGMENTED REBALANCING")
    print(f"{'='*60}")
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        results = structure.rebalance_all_segments()
        
        # Show results
        for seg_name, result in results.items():
            if 'error' not in result:
                print(f"  {seg_name}: {result['concepts_adjusted']} adjusted, "
                      f"{result['anchors_enforced']} anchors enforced")
    
    # Phase 5: Final state
    print(f"\n{'─'*60}")
    print("FINAL STATE")
    print(f"{'─'*60}")
    
    for concept in test_concepts:
        opposite = chain.find_opposite(concept)
        opp_str = opposite[0] if opposite else "none"
        locked = "🔒" if concept in structure.locked_concepts else ""
        print(f"  {concept}{locked}: opposite={opp_str}")
    
    # Compute redundancy
    redundancy = structure.compute_redundancy_score()
    print(f"\nRedundancy score: {redundancy:.3f}")
    print(f"  (Higher = more error correction, like DNA codon redundancy)")
    
    # Show anchors
    print(f"\nAnchors (zinc finger binding sites):")
    for anchor in structure.anchors:
        print(f"  {anchor.concept_a} ↔ {anchor.concept_b} "
              f"(dim={anchor.dimension}, conf={anchor.confidence:.2f})")
    
    # Show segments
    print(f"\nSegments (compartmentalized regions):")
    for name, segment in structure.segments.items():
        print(f"  {name}: {segment.concepts[:5]}{'...' if len(segment.concepts) > 5 else ''}")
    
    return structure


if __name__ == "__main__":
    structure = test_segmented_rebalance()
