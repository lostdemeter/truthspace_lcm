#!/usr/bin/env python3
"""
Music Box Navigation: Drum + Comb = Emergent Music
===================================================

The Music Box Principle (Design 112):
  DRUM = Words positioned in φ-space
  COMB = find_nearest(position) decoder
  MUSIC = Output that emerges from interaction

The violation we were making:
  Storing ("hot", "opposite") → "cold" is embedding music in the comb.
  That's a lookup table, not geometry.

The correct approach:
  1. DRUM: Words have positions (embeddings in φ-space)
  2. COMB: Decoder reads positions (find_nearest)
  3. DELTA: Transformation is a vector, not a mapping

The key insight:
  The delta is applied to the POSITION.
  The answer EMERGES from find_nearest(position + delta).

For "opposite", we need to find the semantic dimension that encodes
opposition, then flip along that dimension.

The model's embedding space already has this structure.
We just need to find the right dimension(s).
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.coordinates import PhiCoordinates, PhiPoint

PHI = (1 + math.sqrt(5)) / 2


class MusicBoxNavigator:
    """
    Navigate using the Music Box principle.
    
    The drum is the embedding space.
    The comb is find_nearest.
    The delta is the transformation.
    The music emerges.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.coordinates = PhiCoordinates()
        
        # The DRUM: all word positions
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.vocab_size = self.all_embeds.shape[0]
        self.hidden_dim = self.all_embeds.shape[1]
        
        # Learned deltas for relationships
        self.deltas: Dict[str, torch.Tensor] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    # =========================================================================
    # THE COMB: find_nearest
    # =========================================================================
    
    def find_nearest(self, position: torch.Tensor, top_k: int = 5,
                     exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        The COMB: Read a position, find the nearest word.
        
        This is the decoder. It doesn't contain the music.
        The music emerges from the interaction of position and vocabulary.
        """
        sims = F.cosine_similarity(position.unsqueeze(0).float().to(self.device),
                                   self.all_embeds.float())
        
        if exclude:
            for word in exclude:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    sims[ids[0]] = -1
        
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item())
                for idx in top_indices]
    
    # =========================================================================
    # LEARNING THE DELTA (not the mapping!)
    # =========================================================================
    
    def learn_delta_pca(self, pairs: List[Tuple[str, str]], 
                        relationship: str) -> torch.Tensor:
        """
        Learn the delta using PCA on the difference vectors.
        
        Instead of averaging (which loses structure), we find the
        PRINCIPAL DIRECTION of the transformation.
        
        This is the "opposite axis" - the direction in embedding space
        that encodes opposition.
        """
        diffs = []
        
        for source, target in pairs:
            e_source = self.get_embedding(source)
            e_target = self.get_embedding(target)
            
            if e_source is None or e_target is None:
                continue
            
            diff = (e_target - e_source).float().cpu().numpy()
            diffs.append(diff)
        
        if len(diffs) < 2:
            raise ValueError("Need at least 2 pairs")
        
        # Stack differences
        diff_matrix = np.stack(diffs)
        
        # Find principal direction
        pca = PCA(n_components=1)
        pca.fit(diff_matrix)
        
        # The principal component is the "opposite direction"
        principal_direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        
        # Scale by average magnitude
        avg_magnitude = np.mean([np.linalg.norm(d) for d in diffs])
        delta = principal_direction * avg_magnitude
        
        self.deltas[relationship] = delta
        return delta
    
    def learn_delta_centroid(self, pairs: List[Tuple[str, str]], 
                              relationship: str) -> torch.Tensor:
        """
        Learn delta as the centroid of target positions minus source positions.
        
        This finds the "center of mass" shift.
        """
        source_embeds = []
        target_embeds = []
        
        for source, target in pairs:
            e_source = self.get_embedding(source)
            e_target = self.get_embedding(target)
            
            if e_source is None or e_target is None:
                continue
            
            source_embeds.append(e_source.cpu())
            target_embeds.append(e_target.cpu())
        
        if not source_embeds:
            raise ValueError("No valid pairs")
        
        source_centroid = torch.stack(source_embeds).mean(dim=0)
        target_centroid = torch.stack(target_embeds).mean(dim=0)
        
        delta = target_centroid - source_centroid
        self.deltas[relationship] = delta
        return delta
    
    # =========================================================================
    # THE MUSIC: Apply delta, find nearest
    # =========================================================================
    
    def transform(self, word: str, relationship: str,
                  scale: float = 1.0) -> Optional[Tuple[str, float]]:
        """
        Transform a word using the Music Box principle.
        
        1. Get word position (drum)
        2. Apply delta (rotation)
        3. Find nearest (comb)
        4. Return result (music)
        
        No lookup table consulted. The music emerges.
        """
        if relationship not in self.deltas:
            return None
        
        position = self.get_embedding(word)
        if position is None:
            return None
        
        delta = self.deltas[relationship].to(self.device)
        new_position = position + scale * delta
        
        nearest = self.find_nearest(new_position, top_k=1, exclude=[word])
        
        if nearest:
            return nearest[0]
        return None
    
    def transform_adaptive(self, word: str, relationship: str) -> Optional[Tuple[str, float]]:
        """
        Adaptive transformation that finds optimal scale.
        """
        if relationship not in self.deltas:
            return None
        
        best_result = None
        best_confidence = -1
        
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            result = self.transform(word, relationship, scale)
            if result and result[1] > best_confidence:
                best_result = result
                best_confidence = result[1]
        
        return best_result


def demo_music_box(model, tokenizer):
    """Demo the Music Box approach."""
    print("="*70)
    print("MUSIC BOX NAVIGATION")
    print("="*70)
    print("""
The Music Box Principle:
  DRUM = Word positions in embedding space
  COMB = find_nearest(position) decoder
  DELTA = Transformation vector (not mapping!)
  MUSIC = Answer that emerges
""")
    
    nav = MusicBoxNavigator(model, tokenizer)
    
    # Training pairs
    train_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("up", "down"),
        ("good", "bad"),
        ("happy", "sad"),
    ]
    
    # Test pairs
    test_pairs = [
        ("tall", "short"),
        ("bright", "dark"),
        ("hard", "soft"),
        ("wet", "dry"),
        ("young", "old"),
        ("rich", "poor"),
        ("loud", "quiet"),
        ("thick", "thin"),
    ]
    
    # Learn delta using PCA
    print("\n--- LEARNING DELTA (PCA) ---")
    delta_pca = nav.learn_delta_pca(train_pairs, "opposite_pca")
    print(f"  Delta magnitude: {delta_pca.norm():.2f}")
    
    # Learn delta using centroid
    print("\n--- LEARNING DELTA (CENTROID) ---")
    delta_centroid = nav.learn_delta_centroid(train_pairs, "opposite_centroid")
    print(f"  Delta magnitude: {delta_centroid.norm():.2f}")
    
    # Test PCA delta
    print("\n--- TESTING PCA DELTA ---")
    print("Training pairs:")
    train_correct = 0
    for source, target in train_pairs:
        result = nav.transform(source, "opposite_pca")
        got = result[0] if result else "?"
        found = target.lower() in got.lower()
        if found:
            train_correct += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:12s} (expected: {target}) {marker}")
    print(f"  Training: {train_correct}/{len(train_pairs)}")
    
    print("\nTest pairs (generalization):")
    test_correct = 0
    for source, target in test_pairs:
        result = nav.transform(source, "opposite_pca")
        got = result[0] if result else "?"
        found = target.lower() in got.lower()
        if found:
            test_correct += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:12s} (expected: {target}) {marker}")
    print(f"  Test: {test_correct}/{len(test_pairs)}")
    
    # Test centroid delta
    print("\n--- TESTING CENTROID DELTA ---")
    print("Training pairs:")
    train_correct_c = 0
    for source, target in train_pairs:
        result = nav.transform(source, "opposite_centroid")
        got = result[0] if result else "?"
        found = target.lower() in got.lower()
        if found:
            train_correct_c += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:12s} (expected: {target}) {marker}")
    print(f"  Training: {train_correct_c}/{len(train_pairs)}")
    
    print("\nTest pairs (generalization):")
    test_correct_c = 0
    for source, target in test_pairs:
        result = nav.transform(source, "opposite_centroid")
        got = result[0] if result else "?"
        found = target.lower() in got.lower()
        if found:
            test_correct_c += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:12s} (expected: {target}) {marker}")
    print(f"  Test: {test_correct_c}/{len(test_pairs)}")
    
    # Try adaptive
    print("\n--- ADAPTIVE TRANSFORMATION ---")
    adaptive_correct = 0
    for source, target in test_pairs:
        result = nav.transform_adaptive(source, "opposite_pca")
        got = result[0] if result else "?"
        found = target.lower() in got.lower()
        if found:
            adaptive_correct += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:12s} (expected: {target}) {marker}")
    print(f"  Adaptive: {adaptive_correct}/{len(test_pairs)}")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    demo_music_box(model, tokenizer)


if __name__ == "__main__":
    main()
