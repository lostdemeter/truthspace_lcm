#!/usr/bin/env python3
"""
Geometric Colorizer V3: Knowledge Chemistry

This version uses the full Knowledge Chemistry framework:
    - Atoms: Color properties with response curves
    - Molecules: Relationships (sky above ground, etc.)
    - Reactions: Transformations (sunset, shadows, etc.)

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.knowledge_base import (
    KnowledgeBase, KnowledgeAtom, KnowledgeMolecule, KnowledgeReaction,
    RelationType, ReactionTrigger, create_color_knowledge_base
)


class ChemistryColorizer:
    """
    A colorizer using the Knowledge Chemistry framework.
    
    This version uses:
        1. Atoms: Get color values based on luminance (response curves)
        2. Molecules: Enforce relationships (sky above ground)
        3. Reactions: Apply transformations (sunset, shadows)
    
    Example:
        colorizer = ChemistryColorizer()
        
        # Colorize with semantic regions
        ab = colorizer.colorize(grayscale, {
            "sky": sky_mask,
            "vegetation": grass_mask,
        })
        
        # Apply sunset reaction
        ab_sunset = colorizer.apply_reaction("Sunset", ab, strength=0.5)
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize the colorizer.
        
        Args:
            knowledge_base: Pre-built knowledge base, or None to use default
        """
        self.kb = knowledge_base or create_color_knowledge_base()
        
        print(f"ChemistryColorizer initialized:")
        print(f"  Atoms: {len(self.kb.atoms)}")
        print(f"  Molecules: {len(self.kb.molecules)}")
        print(f"  Reactions: {len(self.kb.reactions)}")
    
    def colorize(
        self, 
        grayscale: np.ndarray,
        semantic_map: Optional[Dict[str, np.ndarray]] = None,
        default_category: str = "neutral"
    ) -> np.ndarray:
        """
        Colorize a grayscale image using knowledge chemistry.
        
        Args:
            grayscale: Grayscale image [H, W] with values 0-1
            semantic_map: Dict of {category: mask} for semantic regions
            default_category: Category for unlabeled pixels
            
        Returns:
            ab channels [H, W, 2]
        """
        H, W = grayscale.shape
        ab = np.zeros((H, W, 2))
        
        # Track which pixels have been assigned
        assigned = np.zeros((H, W), dtype=bool)
        
        # Step 1: Apply semantic regions using atoms
        if semantic_map:
            for category, mask in semantic_map.items():
                if not mask.any():
                    continue
                
                # Get atoms for this category
                atoms = self.kb.get_atoms_by_category(category)
                if not atoms:
                    # Try to find a similar category
                    atoms = self._find_similar_atoms(category)
                
                if atoms:
                    # Use first atom (could be extended to blend)
                    atom = atoms[0]
                    
                    # Apply atom with response curve
                    for i in range(H):
                        for j in range(W):
                            if mask[i, j]:
                                lum = grayscale[i, j]
                                color = atom.get_value(lum)
                                ab[i, j, 0] = color[0]
                                ab[i, j, 1] = color[1]
                                assigned[i, j] = True
        
        # Step 2: Fill unassigned pixels with default
        if not assigned.all():
            default_atom = self.kb.get_atoms_by_category(default_category)
            if default_atom:
                atom = default_atom[0]
                for i in range(H):
                    for j in range(W):
                        if not assigned[i, j]:
                            lum = grayscale[i, j]
                            color = atom.get_value(lum)
                            ab[i, j, 0] = color[0]
                            ab[i, j, 1] = color[1]
        
        # Step 3: Apply molecular constraints
        ab = self._apply_molecules(ab, grayscale, semantic_map)
        
        # Step 4: Smooth edges
        ab = self._smooth_edges(ab, grayscale)
        
        return ab
    
    def _find_similar_atoms(self, category: str) -> List[KnowledgeAtom]:
        """Find atoms with similar category names."""
        # Simple substring matching
        for cat in self.kb._by_category:
            if category.lower() in cat.lower() or cat.lower() in category.lower():
                return self.kb.get_atoms_by_category(cat)
        return []
    
    def _apply_molecules(
        self, 
        ab: np.ndarray, 
        grayscale: np.ndarray,
        semantic_map: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Apply molecular constraints to enforce relationships."""
        if not semantic_map:
            return ab
        
        H, W = grayscale.shape
        
        # Check SkyAboveGround
        if "sky" in semantic_map and "vegetation" in semantic_map:
            sky_mask = semantic_map["sky"]
            ground_mask = semantic_map["vegetation"]
            
            # Find boundary between sky and ground
            for j in range(W):
                sky_rows = np.where(sky_mask[:, j])[0]
                ground_rows = np.where(ground_mask[:, j])[0]
                
                if len(sky_rows) > 0 and len(ground_rows) > 0:
                    sky_bottom = sky_rows.max()
                    ground_top = ground_rows.min()
                    
                    # Blend at boundary
                    if abs(sky_bottom - ground_top) < 5:
                        for i in range(max(0, sky_bottom - 2), min(H, ground_top + 3)):
                            blend = (i - sky_bottom + 2) / 5
                            blend = np.clip(blend, 0, 1)
                            
                            sky_atom = self.kb.get_atom("Sb")
                            ground_atom = self.kb.get_atom("Gg")
                            
                            if sky_atom and ground_atom:
                                lum = grayscale[i, j]
                                sky_color = sky_atom.get_value(lum)
                                ground_color = ground_atom.get_value(lum)
                                
                                ab[i, j, 0] = sky_color[0] * (1 - blend) + ground_color[0] * blend
                                ab[i, j, 1] = sky_color[1] * (1 - blend) + ground_color[1] * blend
        
        # Check WaterReflectsSky
        if "water" in semantic_map and "sky" in semantic_map:
            water_mask = semantic_map["water"]
            sky_mask = semantic_map["sky"]
            
            if water_mask.any() and sky_mask.any():
                # Get average sky color
                sky_ab = ab[sky_mask].mean(axis=0)
                
                # Blend water toward sky color
                water_atom = self.kb.get_atom("Ob")
                if water_atom:
                    for i in range(H):
                        for j in range(W):
                            if water_mask[i, j]:
                                lum = grayscale[i, j]
                                water_color = water_atom.get_value(lum)
                                
                                # Blend 30% toward sky
                                ab[i, j, 0] = water_color[0] * 0.7 + sky_ab[0] * 0.3
                                ab[i, j, 1] = water_color[1] * 0.7 + sky_ab[1] * 0.3
        
        return ab
    
    def _smooth_edges(self, ab: np.ndarray, grayscale: np.ndarray) -> np.ndarray:
        """Apply edge-aware smoothing."""
        H, W = ab.shape[:2]
        smoothed = ab.copy()
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                center_lum = grayscale[i, j]
                
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                weights = []
                values_a = []
                values_b = []
                
                for ni, nj in neighbors:
                    lum_diff = abs(grayscale[ni, nj] - center_lum)
                    weight = np.exp(-lum_diff * 10)
                    weights.append(weight)
                    values_a.append(ab[ni, nj, 0])
                    values_b.append(ab[ni, nj, 1])
                
                total_weight = sum(weights) + 1
                smoothed[i, j, 0] = (ab[i, j, 0] + sum(w * v for w, v in zip(weights, values_a))) / total_weight
                smoothed[i, j, 1] = (ab[i, j, 1] + sum(w * v for w, v in zip(weights, values_b))) / total_weight
        
        return smoothed
    
    def apply_reaction(
        self, 
        reaction_name: str, 
        ab: np.ndarray, 
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply a reaction to the colorized image.
        
        Args:
            reaction_name: Name of the reaction
            ab: Current ab channels
            strength: Reaction strength (0-1)
            
        Returns:
            Transformed ab channels
        """
        reaction = self.kb.get_reaction(reaction_name)
        if reaction is None:
            print(f"Warning: Reaction '{reaction_name}' not found")
            return ab
        
        H, W = ab.shape[:2]
        result = ab.copy()
        
        for i in range(H):
            for j in range(W):
                original = (ab[i, j, 0], ab[i, j, 1])
                transformed = reaction.apply(original, strength)
                result[i, j, 0] = transformed[0]
                result[i, j, 1] = transformed[1]
        
        return result
    
    def colorize_with_reactions(
        self,
        grayscale: np.ndarray,
        semantic_map: Dict[str, np.ndarray],
        reactions: List[Tuple[str, float]]
    ) -> np.ndarray:
        """
        Colorize and apply multiple reactions.
        
        Args:
            grayscale: Grayscale image
            semantic_map: Semantic regions
            reactions: List of (reaction_name, strength)
            
        Returns:
            Colorized and transformed ab channels
        """
        ab = self.colorize(grayscale, semantic_map)
        
        for reaction_name, strength in reactions:
            ab = self.apply_reaction(reaction_name, ab, strength)
        
        return ab


def lab_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB."""
    try:
        from skimage import color
        
        lab = np.zeros((*L.shape, 3))
        lab[..., 0] = L * 100
        lab[..., 1] = ab[..., 0]
        lab[..., 2] = ab[..., 1]
        
        rgb = color.lab2rgb(lab)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    except ImportError:
        rgb = np.zeros((*L.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip((L * 255 + ab[..., 0]).astype(int), 0, 255)
        rgb[..., 1] = np.clip((L * 255 - ab[..., 0] * 0.5 - ab[..., 1] * 0.5).astype(int), 0, 255)
        rgb[..., 2] = np.clip((L * 255 + ab[..., 1]).astype(int), 0, 255)
        return rgb


def create_test_images():
    """Create test images with semantic regions."""
    images = []
    
    # 1. Landscape (sky + ground)
    landscape = np.zeros((64, 64))
    sky_mask = np.zeros((64, 64), dtype=bool)
    ground_mask = np.zeros((64, 64), dtype=bool)
    
    for i in range(64):
        if i < 28:
            landscape[i, :] = 0.75 - 0.1 * i / 28
            sky_mask[i, :] = True
        elif i < 32:
            landscape[i, :] = 0.65
        else:
            landscape[i, :] = 0.35 + 0.15 * (i - 32) / 32
            ground_mask[i, :] = True
    
    images.append(("landscape", landscape, {"sky": sky_mask, "vegetation": ground_mask}))
    
    # 2. Beach (sky + water + sand)
    beach = np.zeros((64, 64))
    sky_mask = np.zeros((64, 64), dtype=bool)
    water_mask = np.zeros((64, 64), dtype=bool)
    sand_mask = np.zeros((64, 64), dtype=bool)
    
    for i in range(64):
        if i < 20:
            beach[i, :] = 0.8 - 0.1 * i / 20
            sky_mask[i, :] = True
        elif i < 40:
            beach[i, :] = 0.5 + 0.1 * np.sin((i - 20) * 0.3)
            water_mask[i, :] = True
        else:
            beach[i, :] = 0.6 + 0.1 * (i - 40) / 24
            sand_mask[i, :] = True
    
    images.append(("beach", beach, {"sky": sky_mask, "water": water_mask, "earth": sand_mask}))
    
    # 3. Forest
    forest = np.zeros((64, 64))
    sky_mask = np.zeros((64, 64), dtype=bool)
    foliage_mask = np.zeros((64, 64), dtype=bool)
    
    for i in range(64):
        for j in range(64):
            if i < 15:
                forest[i, j] = 0.7
                sky_mask[i, j] = True
            else:
                noise = np.sin(i * 0.3) * np.cos(j * 0.4) * 0.15
                forest[i, j] = 0.35 + noise + np.random.rand() * 0.1
                foliage_mask[i, j] = True
    
    images.append(("forest", forest, {"sky": sky_mask, "vegetation": foliage_mask}))
    
    # 4. Portrait (skin + background)
    portrait = np.zeros((64, 64))
    skin_mask = np.zeros((64, 64), dtype=bool)
    
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - 32)**2 + (j - 32)**2)
            if dist < 20:
                portrait[i, j] = 0.55 + 0.15 * (1 - dist / 20)
                skin_mask[i, j] = True
            else:
                portrait[i, j] = 0.3
    
    images.append(("portrait", portrait, {"skin": skin_mask}))
    
    return images


def evaluate_v3():
    """Evaluate the chemistry-based colorizer."""
    print("=" * 70)
    print("GEOMETRIC COLORIZER V3: Knowledge Chemistry")
    print("=" * 70)
    
    colorizer = ChemistryColorizer()
    
    results_dir = Path(__file__).parent / "results_v3"
    results_dir.mkdir(exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    
    test_images = create_test_images()
    
    for name, gray, semantic_map in test_images:
        print(f"\n--- {name} ---")
        
        # Basic colorization
        ab = colorizer.colorize(gray, semantic_map)
        
        # Statistics
        print(f"  a: [{ab[..., 0].min():.1f}, {ab[..., 0].max():.1f}]")
        print(f"  b: [{ab[..., 1].min():.1f}, {ab[..., 1].max():.1f}]")
        
        # Save
        rgb = lab_to_rgb(gray, ab)
        Image.fromarray(rgb).save(results_dir / f"{name}_chemistry.png")
        Image.fromarray((gray * 255).astype(np.uint8)).save(results_dir / f"{name}_gray.png")
        print(f"  Saved: {name}_chemistry.png")
        
        # Apply sunset reaction for landscape/beach
        if name in ["landscape", "beach"]:
            ab_sunset = colorizer.apply_reaction("Sunset", ab, strength=0.5)
            rgb_sunset = lab_to_rgb(gray, ab_sunset)
            Image.fromarray(rgb_sunset).save(results_dir / f"{name}_sunset.png")
            print(f"  Saved: {name}_sunset.png (with Sunset reaction)")
    
    # Summary
    print("\n" + "=" * 70)
    print("V3 EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n## What V3 Adds Over V2:")
    print("  ✓ Response curves: Colors adapt to luminance")
    print("  ✓ Molecular constraints: Sky-ground blending")
    print("  ✓ Water reflects sky: Relationship enforced")
    print("  ✓ Reactions: Sunset transformation")
    
    print("\n## Comparison:")
    print("  V1 (random φ-weights): Random colors")
    print("  V2 (statistics): Correct colors, no relationships")
    print("  V3 (chemistry): Correct colors + relationships + dynamics")
    
    print("\n## The Key Insight:")
    print("  Knowledge Chemistry provides:")
    print("  1. ATOMS: What things are (intrinsic properties)")
    print("  2. MOLECULES: How things relate (constraints)")
    print("  3. REACTIONS: How things change (dynamics)")
    print("")
    print("  This is more complete than just a periodic table.")


if __name__ == "__main__":
    evaluate_v3()
