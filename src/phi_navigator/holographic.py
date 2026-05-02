#!/usr/bin/env python3
"""
Holographic Navigation: Projection-Based Answer Extraction
===========================================================

Instead of storing paths, use holographic projections.

The insight from Additive Error Stereoscopy:
  - 92.3% of information comes from structure, not explicit storage
  - "Errors are signals to exploit"
  - "Holes are noise to ignore"

Applied to semantic navigation:
  - The φ-lattice structure ENCODES relationships
  - We don't need to store paths - we need to PROJECT correctly
  - The answer emerges from the projection

Holographic Principle:
  - Information on the boundary encodes the interior
  - A 2D projection can reconstruct 3D structure
  - Similarly: a query projection can extract the answer

The Approach:
  1. Encode query concept in φ-space
  2. Project through a "relationship template"
  3. The projection lands on the answer

No storage needed - the geometry IS the knowledge.
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.coordinates import PhiCoordinates, PhiPoint

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)


@dataclass
class HolographicTemplate:
    """
    A holographic template for a relationship type.
    
    Instead of storing individual paths, we learn a TEMPLATE
    that can project any concept to its related concept.
    
    The template encodes the "shape" of the relationship.
    """
    name: str
    
    # The projection matrix in φ-space
    # This transforms (levels, signs) → (new_levels, new_signs)
    level_transform: torch.Tensor  # [dim, dim] or [dim] for diagonal
    sign_transform: torch.Tensor   # [dim] - probability of flip per dimension
    
    # Statistics
    n_examples: int = 0
    accuracy: float = 0.0


class HolographicNavigator:
    """
    Navigate using holographic projections instead of stored paths.
    
    Key insight: The relationship IS a projection.
    
    opposite(x) = project(x, opposite_template)
    gender(x) = project(x, gender_template)
    
    The template is learned from examples, but once learned,
    it generalizes to ANY concept.
    """
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if model else None
        
        self.coordinates = PhiCoordinates()
        self.templates: Dict[str, HolographicTemplate] = {}
        
        # Embedding cache
        self._embed_cache: Dict[str, torch.Tensor] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        if word in self._embed_cache:
            return self._embed_cache[word]
        if self.model is None:
            return None
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        embed = self.model.model.embed_tokens.weight[ids[0]].detach()
        self._embed_cache[word] = embed
        return embed
    
    def get_phi_point(self, word: str) -> Optional[PhiPoint]:
        embed = self.get_embedding(word)
        if embed is None:
            return None
        return self.coordinates.encode(embed.cpu())
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        if self.model is None:
            return []
        all_embeds = self.model.model.embed_tokens.weight.detach()
        sims = F.cosine_similarity(embed.unsqueeze(0).float().to(self.device),
                                   all_embeds.float())
        if exclude:
            for word in exclude:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    sims[ids[0]] = -1
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item())
                for idx in top_indices]
    
    # =========================================================================
    # TEMPLATE LEARNING
    # =========================================================================
    
    def learn_template(self, pairs: List[Tuple[str, str]], 
                       relationship: str) -> HolographicTemplate:
        """
        Learn a holographic template from example pairs.
        
        The template captures the "shape" of the relationship:
        - How do levels change? (level_transform)
        - Which signs flip? (sign_transform)
        
        Key insight: We learn the AVERAGE transformation,
        which captures the structural pattern.
        """
        dim = self.model.config.hidden_size if self.model else 3584
        
        level_deltas = []
        sign_flips = []
        
        for source, target in pairs:
            p_source = self.get_phi_point(source)
            p_target = self.get_phi_point(target)
            
            if p_source is None or p_target is None:
                continue
            
            # Level transformation
            level_delta = p_target.levels.float() - p_source.levels.float()
            level_deltas.append(level_delta)
            
            # Sign flip pattern
            flip = (p_source.signs != p_target.signs).float()
            sign_flips.append(flip)
        
        if not level_deltas:
            raise ValueError("No valid pairs found")
        
        # Average transformations
        avg_level_delta = torch.stack(level_deltas).mean(dim=0)
        avg_sign_flip = torch.stack(sign_flips).mean(dim=0)
        
        template = HolographicTemplate(
            name=relationship,
            level_transform=avg_level_delta,
            sign_transform=avg_sign_flip,
            n_examples=len(level_deltas),
        )
        
        self.templates[relationship] = template
        return template
    
    # =========================================================================
    # HOLOGRAPHIC PROJECTION
    # =========================================================================
    
    def project(self, word: str, relationship: str,
                threshold: float = 0.5) -> Optional[Tuple[str, float]]:
        """
        Project a word through a relationship template.
        
        This is the holographic operation:
        1. Encode word in φ-space
        2. Apply the template transformation
        3. Decode to find the answer
        
        No storage lookup - the answer emerges from the projection.
        """
        if relationship not in self.templates:
            return None
        
        template = self.templates[relationship]
        point = self.get_phi_point(word)
        
        if point is None:
            return None
        
        # Apply level transformation
        new_levels = point.levels.float() + template.level_transform
        
        # Apply sign transformation (flip where probability > threshold)
        flip_mask = (template.sign_transform > threshold)
        new_signs = point.signs.clone()
        new_signs[flip_mask] *= -1
        
        # Create new point
        new_point = PhiPoint(
            levels=new_levels.to(torch.int16),
            signs=new_signs
        )
        
        # Decode and find nearest
        new_embed = new_point.to_embedding().to(self.device)
        nearest = self.find_nearest(new_embed, top_k=1, exclude=[word])
        
        if nearest:
            return nearest[0]
        return None
    
    # =========================================================================
    # ADAPTIVE PROJECTION
    # =========================================================================
    
    def adaptive_project(self, word: str, relationship: str) -> Optional[Tuple[str, float]]:
        """
        Adaptive projection that finds the optimal threshold.
        
        Instead of fixed threshold=0.5, we try multiple thresholds
        and return the result with highest confidence.
        
        This is like "focusing" the holographic projection.
        """
        if relationship not in self.templates:
            return None
        
        best_result = None
        best_confidence = -1
        
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            result = self.project(word, relationship, threshold)
            if result and result[1] > best_confidence:
                best_result = result
                best_confidence = result[1]
        
        return best_result
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_template(self, pairs: List[Tuple[str, str]], 
                          relationship: str) -> float:
        """
        Validate a template on test pairs.
        
        Returns accuracy: what fraction of pairs does the template
        correctly project?
        """
        if relationship not in self.templates:
            return 0.0
        
        correct = 0
        total = 0
        
        for source, target in pairs:
            result = self.project(source, relationship)
            if result:
                # Check if target is in the result
                if target.lower() in result[0].lower():
                    correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        self.templates[relationship].accuracy = accuracy
        return accuracy


def demo_holographic_navigation(model, tokenizer):
    """Demo holographic navigation."""
    print("="*70)
    print("HOLOGRAPHIC NAVIGATION DEMO")
    print("="*70)
    
    nav = HolographicNavigator(model, tokenizer)
    
    # Training pairs
    train_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("up", "down"),
        ("good", "bad"),
        ("happy", "sad"),
    ]
    
    # Test pairs (NOT in training)
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
    
    # Learn template from training pairs
    print("\nLearning holographic template from 6 training pairs...")
    template = nav.learn_template(train_pairs, "opposite")
    print(f"  Template learned: {template.n_examples} examples")
    print(f"  Level transform mean: {template.level_transform.mean():.1f}")
    print(f"  Sign flip dims (>0.5): {(template.sign_transform > 0.5).sum().item()}")
    
    # Validate on training pairs
    print("\nValidating on TRAINING pairs:")
    train_acc = nav.validate_template(train_pairs, "opposite")
    for source, target in train_pairs:
        result = nav.project(source, "opposite")
        got = result[0] if result else "?"
        marker = "✓" if target.lower() in got.lower() else "✗"
        print(f"  {source:8s} → {got:8s} (expected: {target}) {marker}")
    print(f"  Training accuracy: {train_acc*100:.0f}%")
    
    # Test on NEW pairs (generalization)
    print("\nTesting on NEW pairs (generalization):")
    test_acc = nav.validate_template(test_pairs, "opposite")
    for source, target in test_pairs:
        result = nav.project(source, "opposite")
        got = result[0] if result else "?"
        marker = "✓" if target.lower() in got.lower() else "✗"
        print(f"  {source:8s} → {got:8s} (expected: {target}) {marker}")
    print(f"  Generalization accuracy: {test_acc*100:.0f}%")
    
    # Try adaptive projection
    print("\nTrying ADAPTIVE projection on test pairs:")
    adaptive_correct = 0
    for source, target in test_pairs:
        result = nav.adaptive_project(source, "opposite")
        got = result[0] if result else "?"
        found = target.lower() in got.lower()
        if found:
            adaptive_correct += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:8s} (expected: {target}) {marker}")
    print(f"  Adaptive accuracy: {adaptive_correct/len(test_pairs)*100:.0f}%")
    
    print("\n" + "="*70)
    print("COMPARISON: STORAGE vs HOLOGRAPHIC")
    print("="*70)
    print(f"""
STORAGE APPROACH:
  - Store every (source, relationship) → target
  - 100% accurate for stored paths
  - 0% for unstored paths (requires fallback)
  - O(n) storage where n = number of paths

HOLOGRAPHIC APPROACH:
  - Learn template from examples
  - {train_acc*100:.0f}% on training, {test_acc*100:.0f}% on test
  - Generalizes to ANY concept
  - O(1) storage (just the template)

THE TRADEOFF:
  Storage: Perfect accuracy, limited coverage
  Holographic: Approximate accuracy, unlimited coverage
""")


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
    
    demo_holographic_navigation(model, tokenizer)


if __name__ == "__main__":
    main()
