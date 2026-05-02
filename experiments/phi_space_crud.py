#!/usr/bin/env python3
"""
φ-Space CRUD Operations

Provides Create, Read, Update, Delete operations on the geometric knowledge space.

All operations reduce to:
  1. Vector addition (translation)
  2. Scalar multiplication (scaling)  
  3. Cosine similarity (reading relationships)

Usage:
    from phi_space_crud import PhiSpaceCRUD
    
    crud = PhiSpaceCRUD(model, tokenizer)
    
    # Create new concept
    pos = crud.create("quantum-chef", parents=["quantum", "chef", "scientist"])
    
    # Read concept
    info = crud.read("consciousness")
    
    # Update concept
    crud.update("Pluto", old_property="planet", new_property="dwarf planet")
    
    # Delete concept
    crud.delete("unicorn")
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import json
from datetime import datetime

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


@dataclass
class ConceptInfo:
    """Information about a concept in φ-space."""
    name: str
    position: np.ndarray
    phi_level: float
    neighbors: List[Tuple[str, float]]
    norm: float
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'phi_level': self.phi_level,
            'norm': self.norm,
            'neighbors': [(n, float(s)) for n, s in self.neighbors]
        }


@dataclass 
class CRUDOperation:
    """Record of a CRUD operation."""
    operation: str  # create, read, update, delete
    concept: str
    timestamp: str
    details: dict
    success: bool


class PhiSpaceCRUD:
    """CRUD operations on φ-space."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Cache embeddings (read-only reference to model)
        self.embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        self.lm_head = model.lm_head.weight.detach().float().cpu().numpy()
        
        # Custom concepts (our additions to the space)
        self.custom_concepts: Dict[str, np.ndarray] = {}
        
        # Modifications (updates/deletes)
        self.modifications: Dict[str, np.ndarray] = {}
        
        # Operation log
        self.operations: List[CRUDOperation] = []
    
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
        """Get embedding for a concept (checks custom first, then model)."""
        # Check custom concepts
        if concept in self.custom_concepts:
            return self.custom_concepts[concept]
        
        # Check modifications
        if concept in self.modifications:
            return self.modifications[concept]
        
        # Get from model
        tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        if tokens:
            return self.embeddings[tokens[0]]
        return None
    
    def _find_neighbors(self, position: np.ndarray, k: int = 10, 
                        exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to a position."""
        exclude = exclude or []
        
        # Check all embeddings
        sims = []
        for i in range(len(self.embeddings)):
            token = self.tokenizer.decode([i]).strip()
            if token and token not in exclude:
                sim = self._cosine_sim(position, self.embeddings[i])
                sims.append((token, sim))
        
        # Check custom concepts
        for name, emb in self.custom_concepts.items():
            if name not in exclude:
                sim = self._cosine_sim(position, emb)
                sims.append((name, sim))
        
        # Sort and return top k unique
        sims.sort(key=lambda x: -x[1])
        
        seen = set()
        result = []
        for token, sim in sims:
            if token.lower() not in seen and len(token) > 1:
                result.append((token, sim))
                seen.add(token.lower())
            if len(result) >= k:
                break
        
        return result
    
    # =========================================================
    # CREATE
    # =========================================================
    
    def create(
        self, 
        name: str, 
        parents: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        analogy: Optional[Tuple[str, str, str]] = None,
        position: Optional[np.ndarray] = None
    ) -> ConceptInfo:
        """
        Create a new concept in φ-space.
        
        Methods:
        1. Weighted combination of parents: position = Σ(weight_i × embed(parent_i))
        2. Analogy: position = A + (B - C) for "A is to C as new is to B"
        3. Direct position specification
        
        Args:
            name: Name for the new concept
            parents: List of parent concept names
            weights: Weights for each parent (default: equal)
            analogy: Tuple (A, B, C) for analogy-based creation
            position: Direct position vector
            
        Returns:
            ConceptInfo for the created concept
        """
        if position is not None:
            # Direct position
            new_pos = position
        elif analogy is not None:
            # Analogy: A + (B - C)
            A, B, C = analogy
            emb_A = self._get_embedding(A)
            emb_B = self._get_embedding(B)
            emb_C = self._get_embedding(C)
            
            if emb_A is None or emb_B is None or emb_C is None:
                raise ValueError(f"Could not find embeddings for analogy: {analogy}")
            
            new_pos = emb_A + (emb_B - emb_C)
        elif parents is not None:
            # Weighted combination
            if weights is None:
                weights = [1.0 / len(parents)] * len(parents)
            
            parent_embs = []
            for p in parents:
                emb = self._get_embedding(p)
                if emb is None:
                    raise ValueError(f"Could not find embedding for parent: {p}")
                parent_embs.append(emb)
            
            new_pos = sum(w * e for w, e in zip(weights, parent_embs))
        else:
            raise ValueError("Must provide parents, analogy, or position")
        
        # Store the new concept
        self.custom_concepts[name] = new_pos
        
        # Get info
        info = ConceptInfo(
            name=name,
            position=new_pos,
            phi_level=self._get_phi_level(new_pos),
            neighbors=self._find_neighbors(new_pos, exclude=[name]),
            norm=float(np.linalg.norm(new_pos))
        )
        
        # Log operation
        self.operations.append(CRUDOperation(
            operation='create',
            concept=name,
            timestamp=datetime.now().isoformat(),
            details={
                'method': 'position' if position is not None else 
                         'analogy' if analogy is not None else 'weighted',
                'parents': parents,
                'weights': weights,
                'analogy': analogy
            },
            success=True
        ))
        
        return info
    
    # =========================================================
    # READ
    # =========================================================
    
    def read(self, concept: str, k_neighbors: int = 10) -> Optional[ConceptInfo]:
        """
        Read a concept from φ-space.
        
        Args:
            concept: Name of the concept
            k_neighbors: Number of neighbors to return
            
        Returns:
            ConceptInfo or None if not found
        """
        position = self._get_embedding(concept)
        
        if position is None:
            self.operations.append(CRUDOperation(
                operation='read',
                concept=concept,
                timestamp=datetime.now().isoformat(),
                details={'error': 'not found'},
                success=False
            ))
            return None
        
        info = ConceptInfo(
            name=concept,
            position=position,
            phi_level=self._get_phi_level(position),
            neighbors=self._find_neighbors(position, k=k_neighbors, exclude=[concept]),
            norm=float(np.linalg.norm(position))
        )
        
        self.operations.append(CRUDOperation(
            operation='read',
            concept=concept,
            timestamp=datetime.now().isoformat(),
            details={'phi_level': info.phi_level, 'norm': info.norm},
            success=True
        ))
        
        return info
    
    # =========================================================
    # UPDATE
    # =========================================================
    
    def update(
        self,
        concept: str,
        old_property: Optional[str] = None,
        new_property: Optional[str] = None,
        direction: Optional[np.ndarray] = None,
        alpha: float = 0.5
    ) -> Optional[ConceptInfo]:
        """
        Update a concept in φ-space.
        
        Formula: new_position = old_position + α × (new_property - old_property)
        
        This is a TRANSLATION in the direction of the property change.
        
        Args:
            concept: Name of the concept to update
            old_property: Property to move away from
            new_property: Property to move toward
            direction: Direct direction vector (alternative to properties)
            alpha: Strength of the update (0-1)
            
        Returns:
            Updated ConceptInfo
        """
        old_pos = self._get_embedding(concept)
        if old_pos is None:
            return None
        
        if direction is not None:
            delta = direction
        elif old_property is not None and new_property is not None:
            old_emb = self._get_embedding(old_property)
            new_emb = self._get_embedding(new_property)
            
            if old_emb is None or new_emb is None:
                raise ValueError(f"Could not find embeddings for properties")
            
            delta = new_emb - old_emb
        else:
            raise ValueError("Must provide direction or (old_property, new_property)")
        
        # Apply update
        new_pos = old_pos + alpha * delta
        
        # Store modification
        self.modifications[concept] = new_pos
        
        info = ConceptInfo(
            name=concept,
            position=new_pos,
            phi_level=self._get_phi_level(new_pos),
            neighbors=self._find_neighbors(new_pos, exclude=[concept]),
            norm=float(np.linalg.norm(new_pos))
        )
        
        self.operations.append(CRUDOperation(
            operation='update',
            concept=concept,
            timestamp=datetime.now().isoformat(),
            details={
                'old_property': old_property,
                'new_property': new_property,
                'alpha': alpha,
                'old_phi': self._get_phi_level(old_pos),
                'new_phi': info.phi_level
            },
            success=True
        ))
        
        return info
    
    # =========================================================
    # DELETE
    # =========================================================
    
    def delete(
        self,
        concept: str,
        method: str = 'isolate',
        beta: float = 2.0
    ) -> bool:
        """
        Delete a concept from φ-space.
        
        Methods:
        - 'isolate': Move concept far from its cluster
        - 'remove': Remove from custom concepts (only works for custom)
        - 'null': Project to near-zero (effectively invisible)
        
        Args:
            concept: Name of the concept to delete
            method: Deletion method
            beta: Strength of isolation (for 'isolate' method)
            
        Returns:
            True if successful
        """
        if method == 'remove':
            # Only works for custom concepts
            if concept in self.custom_concepts:
                del self.custom_concepts[concept]
                self.operations.append(CRUDOperation(
                    operation='delete',
                    concept=concept,
                    timestamp=datetime.now().isoformat(),
                    details={'method': 'remove'},
                    success=True
                ))
                return True
            return False
        
        position = self._get_embedding(concept)
        if position is None:
            return False
        
        if method == 'isolate':
            # Find cluster center
            neighbors = self._find_neighbors(position, k=5, exclude=[concept])
            neighbor_embs = [self._get_embedding(n) for n, _ in neighbors]
            neighbor_embs = [e for e in neighbor_embs if e is not None]
            
            if neighbor_embs:
                cluster_center = np.mean(neighbor_embs, axis=0)
                deletion_vector = position - cluster_center
                isolated_pos = position + beta * deletion_vector
            else:
                # No neighbors, just scale down
                isolated_pos = position * 0.01
            
            self.modifications[concept] = isolated_pos
            
        elif method == 'null':
            # Project to near-zero
            self.modifications[concept] = position * 0.001
        
        self.operations.append(CRUDOperation(
            operation='delete',
            concept=concept,
            timestamp=datetime.now().isoformat(),
            details={'method': method, 'beta': beta},
            success=True
        ))
        
        return True
    
    # =========================================================
    # UTILITIES
    # =========================================================
    
    def compare(self, concept1: str, concept2: str) -> Dict:
        """Compare two concepts."""
        pos1 = self._get_embedding(concept1)
        pos2 = self._get_embedding(concept2)
        
        if pos1 is None or pos2 is None:
            return {'error': 'concept not found'}
        
        return {
            'concept1': concept1,
            'concept2': concept2,
            'similarity': self._cosine_sim(pos1, pos2),
            'distance': float(np.linalg.norm(pos1 - pos2)),
            'phi_level_1': self._get_phi_level(pos1),
            'phi_level_2': self._get_phi_level(pos2)
        }
    
    def list_custom(self) -> List[str]:
        """List all custom concepts."""
        return list(self.custom_concepts.keys())
    
    def list_modifications(self) -> List[str]:
        """List all modified concepts."""
        return list(self.modifications.keys())
    
    def save_state(self, filepath: str):
        """Save custom concepts and modifications."""
        state = {
            'custom_concepts': {k: v.tolist() for k, v in self.custom_concepts.items()},
            'modifications': {k: v.tolist() for k, v in self.modifications.items()},
            'operations': [
                {
                    'operation': op.operation,
                    'concept': op.concept,
                    'timestamp': op.timestamp,
                    'details': op.details,
                    'success': op.success
                }
                for op in self.operations
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load custom concepts and modifications."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.custom_concepts = {k: np.array(v) for k, v in state.get('custom_concepts', {}).items()}
        self.modifications = {k: np.array(v) for k, v in state.get('modifications', {}).items()}
    
    def display(self, info: ConceptInfo):
        """Pretty print concept info."""
        print(f"\n{'='*50}")
        print(f"CONCEPT: {info.name}")
        print(f"{'='*50}")
        print(f"φ-level: {info.phi_level:.4f}")
        print(f"Norm: {info.norm:.4f}")
        print(f"\nNeighbors:")
        for neighbor, sim in info.neighbors[:5]:
            print(f"  • {neighbor:20s} (sim={sim:.4f})")


def main():
    """Demo CRUD operations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    
    crud = PhiSpaceCRUD(model, tokenizer)
    
    print("\n" + "="*60)
    print("φ-SPACE CRUD DEMO")
    print("="*60)
    
    # CREATE
    print("\n[CREATE] New concept: 'quantum-chef'")
    info = crud.create("quantum-chef", parents=["quantum", "chef", "scientist"])
    crud.display(info)
    
    # CREATE with analogy
    print("\n[CREATE] New concept via analogy: 'digital-artist'")
    print("  (digital is to technology as artist is to creativity)")
    info2 = crud.create("digital-artist", analogy=("digital", "artist", "creativity"))
    crud.display(info2)
    
    # READ
    print("\n[READ] Existing concept: 'consciousness'")
    info3 = crud.read("consciousness")
    if info3:
        crud.display(info3)
    
    # UPDATE
    print("\n[UPDATE] Pluto: planet → dwarf planet")
    before = crud.read("Pluto")
    print(f"Before - similarity to 'planet': {crud.compare('Pluto', 'planet')['similarity']:.4f}")
    print(f"Before - similarity to 'dwarf': {crud.compare('Pluto', 'dwarf')['similarity']:.4f}")
    
    crud.update("Pluto", old_property="planet", new_property="dwarf", alpha=0.5)
    
    print(f"After  - similarity to 'planet': {crud.compare('Pluto', 'planet')['similarity']:.4f}")
    print(f"After  - similarity to 'dwarf': {crud.compare('Pluto', 'dwarf')['similarity']:.4f}")
    
    # DELETE
    print("\n[DELETE] Isolating 'unicorn'")
    before_neighbors = crud.read("unicorn")
    print(f"Before - top neighbor similarity: {before_neighbors.neighbors[0][1]:.4f}")
    
    crud.delete("unicorn", method='isolate', beta=2.0)
    
    after_info = crud.read("unicorn")
    print(f"After  - top neighbor similarity: {after_info.neighbors[0][1]:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Custom concepts created: {crud.list_custom()}")
    print(f"Concepts modified: {crud.list_modifications()}")
    print(f"Total operations: {len(crud.operations)}")
    
    # Save state
    crud.save_state("phi_space_state.json")
    print("\nState saved to phi_space_state.json")
    
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
