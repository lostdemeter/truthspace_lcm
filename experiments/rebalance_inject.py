#!/usr/bin/env python3
"""
Rebalance/Inject Training System

Two distinct operations for maintaining emergent structure:

1. REBALANCE: Adjust positions within vocabulary constraints
   - Evaluates current relationships
   - Corrects positions using only known concepts
   - Normalizes to maintain stable structure
   
2. INJECT: Add new concepts through careful prompting
   - When LLM suggests unknown concept, we go find data for it
   - Generate logical constructions to integrate new concept
   - Then rebalance to stabilize

The cycle: Evaluate → Inject (if needed) → Rebalance → Repeat
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
class InjectionRequest:
    """A request to inject a new concept."""
    concept: str
    reason: str  # Why we need this concept
    related_to: List[str]  # Existing concepts it relates to
    relationship_type: str  # 'opposite', 'similar', 'trait'


@dataclass
class RebalanceResult:
    """Result of a rebalance operation."""
    adjustments_made: int
    concepts_affected: List[str]
    stability_score: float  # How stable the structure is after rebalance


class StructureManager:
    """
    Manages the emergent structure through rebalance and inject operations.
    
    Key principle: The structure should be self-consistent. When we encounter
    something we don't know, we inject it carefully, then rebalance to maintain
    consistency.
    """
    
    def __init__(self, semantic_chain: FullyEmergentSemanticChain):
        self.semantic = semantic_chain
        self.injection_queue: List[InjectionRequest] = []
        self.known_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)
        # known_relationships[concept]['opposite'] = other_concept
        
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
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
    # REBALANCE OPERATION
    # =========================================================================
    
    def rebalance(self, concepts: List[str] = None) -> RebalanceResult:
        """
        Rebalance the structure using only known vocabulary.
        
        Steps:
        1. Evaluate current relationships for given concepts
        2. Identify corrections needed (constrained to vocabulary)
        3. Apply position adjustments
        4. Normalize to maintain stability
        """
        if concepts is None:
            concepts = self.semantic.groups[:20]
        
        print(f"\n{'─'*60}")
        print("REBALANCE OPERATION")
        print(f"{'─'*60}")
        
        adjustments = []
        concepts_affected = set()
        
        # Get vocabulary for constraints
        vocabulary = set(self.semantic.groups)
        vocab_list = [g.replace('_', ' ').title() for g in self.semantic.groups[:40]]
        
        for concept in concepts:
            # Evaluate opposite relationship
            correction = self._evaluate_and_correct_opposite(concept, vocabulary, vocab_list)
            if correction:
                adjustments.append(correction)
                concepts_affected.add(concept)
                concepts_affected.add(correction['target'])
        
        # Apply adjustments
        if adjustments:
            self._apply_adjustments(adjustments)
        
        # Normalize structure
        stability = self._normalize_structure()
        
        result = RebalanceResult(
            adjustments_made=len(adjustments),
            concepts_affected=list(concepts_affected),
            stability_score=stability
        )
        
        print(f"\nRebalance complete:")
        print(f"  Adjustments: {result.adjustments_made}")
        print(f"  Concepts affected: {len(result.concepts_affected)}")
        print(f"  Stability score: {result.stability_score:.3f}")
        
        return result
    
    def _evaluate_and_correct_opposite(self, concept: str, 
                                        vocabulary: Set[str],
                                        vocab_list: List[str]) -> Optional[Dict]:
        """Evaluate and correct opposite relationship within vocabulary."""
        # Skip if we already have a confirmed relationship for this concept
        if concept in self.known_relationships and 'opposite' in self.known_relationships[concept]:
            existing = self.known_relationships[concept]['opposite']
            if existing in vocabulary:
                # Already have a good relationship, skip
                return None
        
        result = self.semantic.find_opposite(concept)
        if not result:
            return None
        
        current_opposite = result[0]
        
        # Check if current opposite makes sense
        prompt = f"""Evaluate: "The opposite of {concept.title()} is {current_opposite.title()}"

KNOWN CONCEPTS: {', '.join(vocab_list)}

Is this a good semantic opposite? If not, choose a better one FROM THE LIST ABOVE.

Answer format:
CORRECT: yes/no
BETTER: [concept from list, or "none"]"""

        response = self._call_llm(prompt, max_tokens=100)
        if not response:
            return None
        
        # Parse response - extract single words only
        lines = response.strip().split('\n')
        correct = True
        better = None
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('correct:'):
                correct = 'yes' in line_lower
            elif line_lower.startswith('better:'):
                val = line.split(':', 1)[1].strip().lower()
                # Extract first word only, clean it
                val = val.split()[0] if val.split() else ''
                val = val.strip('",\'()[]')
                if val and val not in ['none', 'n/a', ''] and len(val) < 20:
                    better = val.replace(' ', '_')
        
        # Only return correction if target is in vocabulary
        if not correct and better and better in vocabulary:
            print(f"  {concept}: {current_opposite} → {better}")
            return {
                'concept': concept,
                'current': current_opposite,
                'target': better,
                'type': 'opposite'
            }
        elif not correct and better and better not in vocabulary and len(better) < 20:
            # Queue for injection
            self._queue_injection(better, f"opposite of {concept}", [concept], 'opposite')
        
        return None
    
    def _apply_adjustments(self, adjustments: List[Dict]):
        """Apply position adjustments to enforce relationships."""
        if self.semantic.U is None:
            return
        
        applied_pairs = set()
        
        for adj in adjustments:
            concept = adj['concept']
            target = adj['target']
            
            # Avoid duplicate adjustments
            pair = tuple(sorted([concept, target]))
            if pair in applied_pairs:
                continue
            applied_pairs.add(pair)
            
            # Find indices
            concept_idx = None
            target_idx = None
            for i, g in enumerate(self.semantic.groups):
                if g == concept:
                    concept_idx = i
                if g == target:
                    target_idx = i
            
            if concept_idx is None or target_idx is None:
                continue
            
            # For opposites: push to opposite positions on first few dimensions
            if adj['type'] == 'opposite':
                n_dims = min(3, self.semantic.U.shape[1])
                for d in range(n_dims):
                    # Push to opposite signs
                    self.semantic.U[concept_idx, d] = 0.4
                    self.semantic.U[target_idx, d] = -0.4
            
            # Record the relationship
            self.known_relationships[concept]['opposite'] = target
            self.known_relationships[target]['opposite'] = concept
    
    def _normalize_structure(self) -> float:
        """
        Normalize the U matrix to maintain stability while preserving relationships.
        
        Returns stability score (0-1, higher is better).
        """
        if self.semantic.U is None:
            return 0.0
        
        # First, re-enforce known relationships after any normalization
        self._enforce_known_relationships()
        
        # Soft normalization - scale down if values get too large
        max_val = np.max(np.abs(self.semantic.U))
        if max_val > 2.0:
            self.semantic.U = self.semantic.U / max_val * 1.5
        
        # Re-enforce relationships after scaling
        self._enforce_known_relationships()
        
        # Compute stability score based on variance distribution
        variances = np.var(self.semantic.U, axis=0)
        # Good stability = similar variance across dimensions
        stability = 1.0 - np.std(variances) / (np.mean(variances) + 1e-6)
        
        return max(0.0, min(1.0, stability))
    
    def _enforce_known_relationships(self):
        """Re-enforce all known relationships in the U matrix."""
        if self.semantic.U is None:
            return
        
        for concept, rels in self.known_relationships.items():
            if 'opposite' in rels:
                target = rels['opposite']
                
                # Find indices
                concept_idx = None
                target_idx = None
                for i, g in enumerate(self.semantic.groups):
                    if g == concept:
                        concept_idx = i
                    if g == target:
                        target_idx = i
                
                if concept_idx is not None and target_idx is not None:
                    # Ensure they're on opposite sides of first 3 dimensions
                    n_dims = min(3, self.semantic.U.shape[1])
                    for d in range(n_dims):
                        # If same sign, flip one
                        if self.semantic.U[concept_idx, d] * self.semantic.U[target_idx, d] > 0:
                            self.semantic.U[target_idx, d] *= -1
    
    # =========================================================================
    # INJECT OPERATION
    # =========================================================================
    
    # Words that should never be injected as concepts
    INVALID_CONCEPTS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'and',
        'or', 'but', 'if', 'then', 'so', 'than', 'too', 'very', 'just', 'not',
        'no', 'yes', 'none', 'n/a', 'na', 'null', 'undefined', 'unknown',
    }
    
    def _queue_injection(self, concept: str, reason: str, 
                         related_to: List[str], rel_type: str):
        """Queue a concept for injection."""
        # Validate concept
        concept_clean = concept.lower().strip()
        if concept_clean in self.INVALID_CONCEPTS:
            return
        if len(concept_clean) < 3 or len(concept_clean) > 30:
            return
        
        # Check if already queued
        for req in self.injection_queue:
            if req.concept == concept:
                return
        
        self.injection_queue.append(InjectionRequest(
            concept=concept,
            reason=reason,
            related_to=related_to,
            relationship_type=rel_type
        ))
        print(f"  Queued for injection: {concept} ({reason})")
    
    def inject(self, max_injections: int = 5) -> int:
        """
        Process injection queue - add new concepts to the structure.
        
        Steps:
        1. For each queued concept, generate training data
        2. Establish relationships with existing concepts
        3. Ingest into the chain
        
        Returns number of concepts injected.
        """
        if not self.injection_queue:
            print("\nNo concepts queued for injection.")
            return 0
        
        print(f"\n{'─'*60}")
        print("INJECT OPERATION")
        print(f"{'─'*60}")
        print(f"Processing {len(self.injection_queue)} queued concepts...")
        
        injected = 0
        
        for req in self.injection_queue[:max_injections]:
            frames = self._generate_injection_data(req)
            if frames:
                for frame in frames:
                    self.semantic.ingest_item(frame)
                injected += 1
                print(f"  Injected: {req.concept} ({len(frames)} frames)")
        
        # Clear processed items
        self.injection_queue = self.injection_queue[max_injections:]
        
        # Retrain to incorporate new data
        if injected > 0:
            print(f"\nRetraining with new data...")
            self.semantic.learn_dimensions()
        
        return injected
    
    def _generate_injection_data(self, req: InjectionRequest) -> List[Dict]:
        """Generate training data for a new concept."""
        frames = []
        concept = req.concept.replace('_', ' ').title()
        related = [r.replace('_', ' ').title() for r in req.related_to]
        
        # Generate behavioral sentences for the new concept
        prompt = f"""Generate 5 behavioral sentences for "{concept}".

Context: {concept} is the {req.relationship_type} of {', '.join(related)}.

Rules:
1. Start each sentence with "{concept}"
2. Second word should be a verb (action word)
3. Show characteristic behavior that makes sense given the context
4. Keep sentences 8-15 words

Generate 5 sentences:"""

        response = self._call_llm(prompt, max_tokens=400)
        if response:
            for line in response.strip().split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 15 and line.lower().startswith(concept.lower()):
                    frames.append({
                        'text': line,
                        'agent': req.concept.lower().replace(' ', '_'),
                        'source': 'injection',
                        'relationship': req.relationship_type,
                        'related_to': req.related_to,
                    })
        
        # Generate relationship sentences
        if req.relationship_type == 'opposite':
            for related_concept in req.related_to[:2]:
                related_name = related_concept.replace('_', ' ').title()
                prompt = f"""Generate 2 sentences showing {concept} and {related_name} as opposites.

Rules:
1. Show them in opposition or contrast
2. Each sentence should mention both
3. Keep sentences 8-15 words

Generate 2 sentences:"""

                response = self._call_llm(prompt, max_tokens=200)
                if response:
                    for line in response.strip().split('\n'):
                        line = line.strip().lstrip('0123456789.-) ')
                        if len(line) > 15:
                            # Add for both agents
                            if line.lower().startswith(concept.lower()):
                                frames.append({
                                    'text': line,
                                    'agent': req.concept.lower().replace(' ', '_'),
                                    'source': 'injection_relationship',
                                })
                            elif line.lower().startswith(related_name.lower()):
                                frames.append({
                                    'text': line,
                                    'agent': related_concept.lower(),
                                    'source': 'injection_relationship',
                                })
        
        return frames
    
    # =========================================================================
    # COMBINED CYCLE
    # =========================================================================
    
    def evaluate_inject_rebalance(self, concepts: List[str] = None) -> Dict:
        """
        Full cycle: Evaluate → Inject (if needed) → Rebalance
        
        This is the main entry point for maintaining structure.
        """
        if concepts is None:
            concepts = self.semantic.groups[:15]
        
        print(f"\n{'='*60}")
        print("EVALUATE → INJECT → REBALANCE CYCLE")
        print(f"{'='*60}")
        
        # Phase 1: Evaluate and queue injections
        print(f"\nPhase 1: Evaluating {len(concepts)} concepts...")
        self._evaluate_concepts(concepts)
        
        # Phase 2: Inject queued concepts
        injected = 0
        if self.injection_queue:
            print(f"\nPhase 2: Injecting {len(self.injection_queue)} new concepts...")
            injected = self.inject()
        else:
            print(f"\nPhase 2: No injections needed")
        
        # Phase 3: Rebalance
        print(f"\nPhase 3: Rebalancing structure...")
        rebalance_result = self.rebalance(concepts)
        
        return {
            'concepts_evaluated': len(concepts),
            'concepts_injected': injected,
            'adjustments_made': rebalance_result.adjustments_made,
            'stability_score': rebalance_result.stability_score,
            'injection_queue_remaining': len(self.injection_queue),
        }
    
    def _evaluate_concepts(self, concepts: List[str]):
        """Evaluate concepts and queue injections for unknown relationships."""
        vocabulary = set(self.semantic.groups)
        vocab_list = [g.replace('_', ' ').title() for g in self.semantic.groups[:40]]
        
        for concept in concepts:
            # Check opposite
            result = self.semantic.find_opposite(concept)
            if not result:
                continue
            
            current = result[0]
            
            # Ask LLM for ideal opposite - very constrained prompt
            prompt = f"""What is the semantic opposite of "{concept.title()}"?

KNOWN: {', '.join(vocab_list[:20])}

Answer with ONE WORD only:
FROM_LIST: [one word from list, or "none"]
IDEAL: [one word]"""

            response = self._call_llm(prompt, max_tokens=50)
            if not response:
                continue
            
            # Parse - extract single words only
            from_list = None
            ideal = None
            
            for line in response.strip().split('\n'):
                line_lower = line.lower().strip()
                if line_lower.startswith('from_list:'):
                    val = line.split(':', 1)[1].strip().lower()
                    # Extract first word only
                    val = val.split()[0] if val.split() else ''
                    val = val.strip('",\'()[]')
                    if val and val not in ['none', 'n/a'] and len(val) < 20:
                        from_list = val.replace(' ', '_')
                elif line_lower.startswith('ideal:'):
                    val = line.split(':', 1)[1].strip().lower()
                    # Extract first word only
                    val = val.split()[0] if val.split() else ''
                    val = val.strip('",\'()[]')
                    if val and val not in ['none', 'n/a'] and len(val) < 20:
                        ideal = val.replace(' ', '_')
            
            # If ideal is different from from_list and not in vocabulary, queue injection
            if ideal and ideal not in vocabulary and len(ideal) < 20:
                self._queue_injection(ideal, f"ideal opposite of {concept}", [concept], 'opposite')


def test_rebalance_inject():
    """Test the rebalance/inject system."""
    print("=" * 70)
    print("REBALANCE/INJECT TRAINING SYSTEM")
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
    
    # Create and train initial model
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
    
    # Show initial state
    print("\n" + "─" * 60)
    print("INITIAL STATE")
    print("─" * 60)
    
    test_concepts = ['hero', 'villain', 'holmes', 'watson', 'sage', 'king']
    for concept in test_concepts:
        opposite = chain.find_opposite(concept)
        opp_str = opposite[0] if opposite else "none"
        print(f"  {concept}: opposite={opp_str}")
    
    # Create manager and run cycle
    manager = StructureManager(chain)
    
    # Run multiple cycles
    for cycle in range(3):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle + 1}")
        print(f"{'='*60}")
        
        result = manager.evaluate_inject_rebalance(test_concepts)
        
        print(f"\nCycle {cycle + 1} results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        
        # Show current state
        print(f"\nCurrent opposites:")
        for concept in test_concepts:
            opposite = chain.find_opposite(concept)
            opp_str = opposite[0] if opposite else "none"
            print(f"  {concept}: opposite={opp_str}")
        
        time.sleep(0.5)
    
    # Final state
    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    
    for concept in test_concepts:
        opposite = chain.find_opposite(concept)
        traits = chain.describe_traits(concept)
        opp_str = opposite[0] if opposite else "none"
        print(f"  {concept}: opposite={opp_str}, traits={traits[:2] if traits else []}")
    
    print(f"\nKnown relationships:")
    for concept, rels in manager.known_relationships.items():
        if rels:
            print(f"  {concept}: {rels}")
    
    return manager


if __name__ == "__main__":
    manager = test_rebalance_inject()
