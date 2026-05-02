#!/usr/bin/env python3
"""
Colorization Comb Design - Building a Color Comb from First Principles

The question: How do we build a comb that colorizes grayscale images
WITHOUT training a neural network?

The insight: Color is semantic, not pixel-level. The color of a pixel
depends on WHAT it represents, not just its grayscale value.

The approach:
1. Build a DRUM: Semantic concepts with their typical colors
2. Build a COMB: Map image regions to semantic concepts
3. The MUSIC: Colorized output emerges from drum + comb

This is how we'd "knowingly" build up to colorization.

Author: TruthSpace LCM Project
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')
from phi_space import PhiSpace, PhiPoint

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# STEP 1: THE COLOR DRUM
# =============================================================================
# 
# The drum contains semantic concepts with their typical colors.
# This is the "world knowledge" about what colors things are.
#
# Key insight: We're not storing pixel→color mappings.
# We're storing concept→color relationships.
# =============================================================================

@dataclass
class ColorConcept:
    """A semantic concept with its typical color distribution."""
    name: str
    typical_rgb: Tuple[int, int, int]  # Most common color
    rgb_variance: Tuple[float, float, float]  # How much it varies
    grayscale_range: Tuple[int, int]  # Typical grayscale values
    texture_signature: Optional[np.ndarray] = None  # Local pattern


def build_color_drum() -> PhiSpace:
    """
    Build the color drum - semantic concepts with their colors.
    
    This is the "world knowledge" that a trained AI learns implicitly.
    We're making it explicit and geometric.
    
    Dimensions:
    - dim 0: luminance (dark to bright)
    - dim 1: warmth (cool to warm)
    - dim 2: saturation (gray to vivid)
    - dim 3: naturalness (artificial to natural)
    """
    space = PhiSpace(dims=4, dim_names=['luminance', 'warmth', 'saturation', 'naturalness'])
    
    # Natural outdoor concepts
    concepts = [
        # (name, position, typical_rgb)
        # Sky
        ("clear_sky", [0.7, -0.5, 0.6, 1.0], (135, 206, 235)),
        ("cloudy_sky", [0.8, -0.2, 0.2, 1.0], (200, 200, 210)),
        ("sunset_sky", [0.6, 1.0, 0.9, 1.0], (255, 140, 80)),
        ("night_sky", [0.1, -0.3, 0.3, 1.0], (25, 25, 50)),
        
        # Vegetation
        ("grass", [0.4, 0.2, 0.7, 1.0], (86, 125, 70)),
        ("tree_leaves", [0.3, 0.1, 0.6, 1.0], (60, 100, 50)),
        ("autumn_leaves", [0.4, 0.8, 0.8, 1.0], (180, 100, 40)),
        ("flowers_red", [0.5, 0.9, 0.9, 1.0], (200, 50, 50)),
        ("flowers_yellow", [0.7, 0.7, 0.9, 1.0], (255, 220, 50)),
        
        # Earth/Ground
        ("soil", [0.3, 0.3, 0.4, 1.0], (100, 70, 50)),
        ("sand", [0.7, 0.4, 0.3, 1.0], (210, 190, 160)),
        ("rock", [0.4, 0.0, 0.2, 1.0], (120, 120, 120)),
        ("water", [0.4, -0.4, 0.5, 1.0], (70, 130, 180)),
        
        # Human/Skin
        ("skin_light", [0.7, 0.5, 0.4, 1.0], (230, 190, 170)),
        ("skin_medium", [0.5, 0.5, 0.5, 1.0], (180, 130, 100)),
        ("skin_dark", [0.3, 0.4, 0.5, 1.0], (100, 70, 50)),
        ("hair_blonde", [0.6, 0.5, 0.5, 1.0], (210, 180, 140)),
        ("hair_brown", [0.3, 0.3, 0.4, 1.0], (90, 60, 40)),
        ("hair_black", [0.1, 0.0, 0.2, 1.0], (30, 25, 25)),
        
        # Man-made
        ("concrete", [0.5, 0.0, 0.1, 0.0], (150, 150, 150)),
        ("brick", [0.4, 0.6, 0.5, 0.3], (160, 80, 60)),
        ("wood", [0.4, 0.4, 0.4, 0.5], (140, 100, 60)),
        ("metal", [0.6, -0.1, 0.1, 0.0], (180, 180, 190)),
        ("glass", [0.8, -0.2, 0.2, 0.0], (220, 230, 240)),
        
        # Fabric/Clothing (more variable, lower saturation certainty)
        ("fabric_white", [0.9, 0.0, 0.1, 0.2], (250, 250, 250)),
        ("fabric_black", [0.1, 0.0, 0.1, 0.2], (30, 30, 30)),
        ("fabric_red", [0.4, 0.9, 0.8, 0.2], (180, 50, 50)),
        ("fabric_blue", [0.4, -0.8, 0.7, 0.2], (50, 80, 160)),
    ]
    
    for name, position, rgb in concepts:
        space.add(name, position, metadata={'rgb': rgb})
    
    return space


# =============================================================================
# STEP 2: THE SEMANTIC DETECTOR
# =============================================================================
#
# This is the hard part: given a grayscale image region, what semantic
# concept does it represent?
#
# Traditional AI: Train a classifier on labeled images
# Our approach: Use geometric features that correlate with semantics
#
# Key insight: We can use TEXTURE and CONTEXT, not just grayscale value
# =============================================================================

class SemanticDetector:
    """
    Detect semantic concepts from grayscale image features.
    
    This maps image regions to positions in the color drum's φ-space.
    
    The detector uses:
    1. Local grayscale statistics (mean, variance)
    2. Texture patterns (edges, smoothness)
    3. Spatial context (position in image, neighbors)
    4. Global context (what else is in the image)
    """
    
    def __init__(self, color_drum: PhiSpace):
        self.drum = color_drum
        
        # Build grayscale→position mapping from drum
        self._build_grayscale_hints()
    
    def _build_grayscale_hints(self):
        """
        Build hints from grayscale values to likely concepts.
        
        This is NOT a lookup table - it's a starting point for
        geometric search.
        """
        self.grayscale_hints = {}
        
        for point in self.drum.points:
            rgb = point.metadata.get('rgb', (128, 128, 128))
            # Convert RGB to grayscale
            gray = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
            
            # Store as hint (multiple concepts can have same grayscale)
            if gray not in self.grayscale_hints:
                self.grayscale_hints[gray] = []
            self.grayscale_hints[gray].append(point)
    
    def extract_features(self, patch: np.ndarray, 
                         y_position: float, 
                         x_position: float) -> np.ndarray:
        """
        Extract geometric features from a grayscale patch.
        
        Returns a 4D position in the drum's φ-space.
        
        This is where the "comb" logic lives - mapping pixels to semantics.
        """
        # Feature 1: Luminance (normalized grayscale mean)
        luminance = patch.mean() / 255.0
        
        # Feature 2: Warmth estimate from position and texture
        # Sky is usually at top, ground at bottom
        # This is a PRIOR, not a rule
        warmth = 0.0
        if y_position < 0.3:  # Upper part of image
            warmth = -0.3  # Likely cooler (sky)
        elif y_position > 0.7:  # Lower part
            warmth = 0.2  # Likely warmer (ground)
        
        # Feature 3: Saturation estimate from variance
        # High variance often means more colorful
        variance = patch.std() / 128.0
        saturation = min(1.0, variance * 2)
        
        # Feature 4: Naturalness from texture
        # Smooth = more likely artificial, textured = more natural
        # Use local gradient magnitude as proxy
        if patch.shape[0] > 1 and patch.shape[1] > 1:
            gy = np.abs(np.diff(patch, axis=0)).mean()
            gx = np.abs(np.diff(patch, axis=1)).mean()
            texture = (gy + gx) / 255.0
            naturalness = min(1.0, texture * 3)
        else:
            naturalness = 0.5
        
        return np.array([luminance, warmth, saturation, naturalness])
    
    def detect(self, patch: np.ndarray, 
               y_position: float = 0.5, 
               x_position: float = 0.5,
               context: Dict = None) -> Tuple[str, np.ndarray]:
        """
        Detect the semantic concept for a grayscale patch.
        
        Returns (concept_name, rgb_color)
        """
        # Extract features → position in φ-space
        position = self.extract_features(patch, y_position, x_position)
        
        # Query the drum for nearest concept
        nearest = self.drum.query(position, k=3)
        
        if not nearest:
            return "unknown", np.array([128, 128, 128])
        
        # Weighted average of top-k colors based on distance
        total_weight = 0
        weighted_rgb = np.zeros(3)
        
        for concept_name, distance in nearest:
            point = self.drum[concept_name]
            rgb = np.array(point.metadata.get('rgb', (128, 128, 128)))
            
            # Weight by inverse distance (closer = more weight)
            weight = 1.0 / (distance + 0.1)
            weighted_rgb += weight * rgb
            total_weight += weight
        
        final_rgb = (weighted_rgb / total_weight).astype(int)
        
        return nearest[0][0], final_rgb


# =============================================================================
# STEP 3: THE COLORIZATION COMB
# =============================================================================
#
# The comb reads the grayscale image and produces color output.
# It uses the drum (semantic concepts) and detector (feature extraction).
#
# This is the full pipeline: grayscale → features → drum query → color
# =============================================================================

class ColorizationComb:
    """
    The colorization comb - transforms grayscale to color.
    
    This is NOT a trained neural network. It's a geometric pipeline:
    1. Divide image into patches
    2. Extract features from each patch
    3. Query the color drum for nearest semantic concept
    4. Apply the concept's color
    
    The "intelligence" is in the drum (world knowledge) and
    the feature extraction (what makes something look like sky vs grass).
    """
    
    def __init__(self, patch_size: int = 16):
        self.patch_size = patch_size
        self.drum = build_color_drum()
        self.detector = SemanticDetector(self.drum)
    
    def colorize(self, grayscale: np.ndarray) -> np.ndarray:
        """
        Colorize a grayscale image.
        
        Args:
            grayscale: HxW grayscale image (0-255)
        
        Returns:
            HxWx3 RGB image
        """
        H, W = grayscale.shape
        output = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Process in patches
        for y in range(0, H, self.patch_size):
            for x in range(0, W, self.patch_size):
                # Extract patch
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)
                patch = grayscale[y:y_end, x:x_end]
                
                # Normalized position (0-1)
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                # Detect semantic concept and get color
                concept, rgb = self.detector.detect(patch, y_pos, x_pos)
                
                # Apply color to patch
                output[y:y_end, x:x_end] = rgb
        
        return output
    
    def colorize_smooth(self, grayscale: np.ndarray) -> np.ndarray:
        """
        Colorize with smooth blending between patches.
        
        Uses overlapping patches and weighted averaging.
        """
        H, W = grayscale.shape
        output = np.zeros((H, W, 3), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)
        
        step = self.patch_size // 2  # 50% overlap
        
        for y in range(0, H, step):
            for x in range(0, W, step):
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)
                patch = grayscale[y:y_end, x:x_end]
                
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                concept, rgb = self.detector.detect(patch, y_pos, x_pos)
                
                # Gaussian weight (center of patch has more weight)
                patch_h, patch_w = y_end - y, x_end - x
                yy, xx = np.meshgrid(
                    np.linspace(-1, 1, patch_h),
                    np.linspace(-1, 1, patch_w),
                    indexing='ij'
                )
                weight = np.exp(-(xx**2 + yy**2) / 0.5)
                
                output[y:y_end, x:x_end] += weight[:, :, np.newaxis] * rgb
                weights[y:y_end, x:x_end] += weight
        
        # Normalize
        weights = np.maximum(weights, 1e-6)
        output = output / weights[:, :, np.newaxis]
        
        return output.astype(np.uint8)


# =============================================================================
# STEP 4: HOW TO IMPROVE THE COMB
# =============================================================================
#
# The basic comb above is naive. Here's how we'd improve it:
#
# 1. RICHER DRUM: More concepts, more color variations
# 2. BETTER FEATURES: Use texture analysis, edge detection, etc.
# 3. CONTEXT: Use global image features to disambiguate
# 4. LEARNING: Let the drum self-organize via attractor/repeller dynamics
#
# The key insight: We're not training a black box. We're building
# interpretable geometric structure.
# =============================================================================

def demo_colorization_approach():
    """Demonstrate the colorization comb design."""
    print("=" * 70)
    print("COLORIZATION COMB DESIGN")
    print("=" * 70)
    
    # Build the drum
    drum = build_color_drum()
    print(f"\n1. COLOR DRUM")
    print(f"   {len(drum)} semantic concepts with colors")
    print(f"   Dimensions: {drum.dim_names}")
    
    # Show some concepts
    print("\n   Sample concepts:")
    for name in ["clear_sky", "grass", "skin_light", "concrete"]:
        point = drum[name]
        rgb = point.metadata['rgb']
        print(f"     {name:15s}: pos={point.position}, rgb={rgb}")
    
    # Create detector
    detector = SemanticDetector(drum)
    print(f"\n2. SEMANTIC DETECTOR")
    print(f"   Maps grayscale patches → φ-space positions")
    
    # Test detection
    print("\n   Test detections:")
    
    # Simulate a bright, smooth patch (likely sky)
    sky_patch = np.ones((16, 16)) * 200  # Bright
    concept, rgb = detector.detect(sky_patch, y_position=0.1, x_position=0.5)
    print(f"     Bright patch at top: {concept} → RGB{tuple(rgb)}")
    
    # Simulate a medium, textured patch (likely vegetation)
    veg_patch = np.random.randint(80, 120, (16, 16)).astype(np.uint8)
    concept, rgb = detector.detect(veg_patch, y_position=0.6, x_position=0.5)
    print(f"     Medium textured patch: {concept} → RGB{tuple(rgb)}")
    
    # Simulate a dark, smooth patch (likely shadow/night)
    dark_patch = np.ones((16, 16)) * 30
    concept, rgb = detector.detect(dark_patch, y_position=0.5, x_position=0.5)
    print(f"     Dark smooth patch: {concept} → RGB{tuple(rgb)}")
    
    # Create comb
    comb = ColorizationComb(patch_size=16)
    print(f"\n3. COLORIZATION COMB")
    print(f"   Pipeline: grayscale → features → drum query → color")
    
    # Test on synthetic image
    print("\n4. SYNTHETIC TEST")
    
    # Create a simple gradient image (sky at top, ground at bottom)
    H, W = 64, 64
    test_image = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        # Bright at top (sky), darker at bottom (ground)
        brightness = int(255 * (1 - y/H) * 0.7 + 50)
        test_image[y, :] = brightness
    
    # Add some texture to bottom half
    test_image[H//2:, :] += np.random.randint(-20, 20, (H//2, W)).astype(np.uint8)
    test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    
    # Colorize
    colorized = comb.colorize(test_image)
    
    print(f"   Input: {H}x{W} grayscale gradient")
    print(f"   Output: {colorized.shape} RGB")
    print(f"   Top color (sky): RGB{tuple(colorized[5, W//2])}")
    print(f"   Bottom color (ground): RGB{tuple(colorized[H-5, W//2])}")
    
    print("\n" + "=" * 70)
    print("HOW TO BUILD A REAL COLORIZATION COMB")
    print("=" * 70)
    print("""
    The demo above shows the STRUCTURE of a colorization comb.
    To make it work well, we need:
    
    1. RICHER DRUM
       - More semantic concepts (thousands, not dozens)
       - Color distributions, not single colors
       - Learned from real images (but stored geometrically)
    
    2. BETTER FEATURES
       - Texture descriptors (Gabor filters, LBP)
       - Edge patterns
       - Multi-scale analysis
       - These map to φ-space dimensions
    
    3. CONTEXT
       - Global image features (is this indoor/outdoor?)
       - Neighboring patch consistency
       - Object detection (where are faces, sky, etc.)
    
    4. SELF-ORGANIZATION
       - Use attractor/repeller dynamics (Doc 022)
       - Let similar concepts cluster
       - Let dissimilar concepts separate
       - The drum organizes itself from examples
    
    The key insight:
    
    Traditional AI: Train end-to-end, structure is implicit
    Our approach: Build structure explicitly, operations are geometric
    
    Both can achieve similar results, but ours is:
    - Interpretable (we know what each dimension means)
    - Modular (swap drums, keep comb)
    - Incremental (add concepts without retraining)
    
    The "comb" is just: position → nearest → color
    The "intelligence" is in the drum's structure.
""")


if __name__ == "__main__":
    demo_colorization_approach()
