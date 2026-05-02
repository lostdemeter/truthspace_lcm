#!/usr/bin/env python3
"""
The Periodic Table of Color Knowledge

Like organizing elements by their properties, we organize color knowledge
by its geometric properties. What features define a color category?

The analogy:
    - Elements have: atomic number, mass, electron configuration
    - Colors have: luminance response, saturation curve, spatial behavior

We're trying to discover the "atomic properties" of color knowledge.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum


class LuminanceResponse(Enum):
    """How does this color respond to luminance?"""
    DARK_ONLY = "dark_only"           # Only appears in dark regions
    BRIGHT_ONLY = "bright_only"       # Only appears in bright regions
    PROPORTIONAL = "proportional"     # Saturation scales with luminance
    INVERSE = "inverse"               # Saturation inverse to luminance
    UNIFORM = "uniform"               # Same across all luminance
    THRESHOLD = "threshold"           # Appears above/below threshold


class SpatialBehavior(Enum):
    """How does this color behave spatially?"""
    UNIFORM = "uniform"               # Same everywhere (sky)
    GRADIENT = "gradient"             # Gradual change (sunset)
    TEXTURED = "textured"             # Varies locally (foliage)
    EDGE_BOUND = "edge_bound"         # Follows edges (shadows)
    BLOB = "blob"                     # Coherent regions (objects)


class SemanticCategory(Enum):
    """What semantic category does this color belong to?"""
    NATURAL_SKY = "natural_sky"
    NATURAL_VEGETATION = "natural_vegetation"
    NATURAL_EARTH = "natural_earth"
    NATURAL_WATER = "natural_water"
    ORGANIC_SKIN = "organic_skin"
    ORGANIC_WOOD = "organic_wood"
    ARTIFICIAL_METAL = "artificial_metal"
    ARTIFICIAL_FABRIC = "artificial_fabric"
    LIGHT_SHADOW = "light_shadow"
    LIGHT_HIGHLIGHT = "light_highlight"


@dataclass
class ColorAtom:
    """
    A fundamental unit of color knowledge - like an element in the periodic table.
    
    Properties:
        name: Human-readable name
        symbol: Short symbol (like element symbols)
        a_center: Center of a channel (-128 to 128)
        b_center: Center of b channel (-128 to 128)
        saturation_range: (min, max) saturation
        luminance_response: How it responds to luminance
        spatial_behavior: How it behaves spatially
        semantic_category: What it represents
        common_contexts: Where this color typically appears
    """
    name: str
    symbol: str
    a_center: float
    b_center: float
    saturation_range: Tuple[float, float]
    luminance_response: LuminanceResponse
    spatial_behavior: SpatialBehavior
    semantic_category: SemanticCategory
    common_contexts: List[str]
    
    @property
    def hue_angle(self) -> float:
        """Hue angle in degrees (0-360)."""
        return np.degrees(np.arctan2(self.b_center, self.a_center)) % 360
    
    @property
    def max_saturation(self) -> float:
        """Maximum saturation."""
        return np.sqrt(self.a_center**2 + self.b_center**2)
    
    def describe(self) -> str:
        return (
            f"{self.symbol} - {self.name}\n"
            f"  ab: ({self.a_center}, {self.b_center})\n"
            f"  Hue: {self.hue_angle:.0f}°, Sat: {self.max_saturation:.0f}\n"
            f"  Luminance: {self.luminance_response.value}\n"
            f"  Spatial: {self.spatial_behavior.value}\n"
            f"  Category: {self.semantic_category.value}"
        )


# =============================================================================
# THE PERIODIC TABLE OF COLORS
# =============================================================================

PERIODIC_TABLE = [
    # === NATURAL: SKY ===
    ColorAtom(
        name="Clear Sky Blue",
        symbol="Sb",
        a_center=-5,
        b_center=-40,
        saturation_range=(20, 50),
        luminance_response=LuminanceResponse.BRIGHT_ONLY,
        spatial_behavior=SpatialBehavior.GRADIENT,
        semantic_category=SemanticCategory.NATURAL_SKY,
        common_contexts=["sky", "daytime", "outdoor"]
    ),
    ColorAtom(
        name="Sunset Orange",
        symbol="So",
        a_center=30,
        b_center=40,
        saturation_range=(30, 60),
        luminance_response=LuminanceResponse.BRIGHT_ONLY,
        spatial_behavior=SpatialBehavior.GRADIENT,
        semantic_category=SemanticCategory.NATURAL_SKY,
        common_contexts=["sunset", "sunrise", "golden hour"]
    ),
    ColorAtom(
        name="Overcast Gray",
        symbol="Og",
        a_center=0,
        b_center=-5,
        saturation_range=(0, 10),
        luminance_response=LuminanceResponse.UNIFORM,
        spatial_behavior=SpatialBehavior.UNIFORM,
        semantic_category=SemanticCategory.NATURAL_SKY,
        common_contexts=["cloudy", "overcast", "foggy"]
    ),
    
    # === NATURAL: VEGETATION ===
    ColorAtom(
        name="Grass Green",
        symbol="Gg",
        a_center=-30,
        b_center=30,
        saturation_range=(20, 50),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.NATURAL_VEGETATION,
        common_contexts=["lawn", "field", "meadow"]
    ),
    ColorAtom(
        name="Forest Green",
        symbol="Fg",
        a_center=-25,
        b_center=15,
        saturation_range=(15, 40),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.NATURAL_VEGETATION,
        common_contexts=["forest", "trees", "foliage"]
    ),
    ColorAtom(
        name="Autumn Orange",
        symbol="Ao",
        a_center=20,
        b_center=40,
        saturation_range=(25, 55),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.NATURAL_VEGETATION,
        common_contexts=["autumn", "fall leaves", "deciduous"]
    ),
    
    # === NATURAL: EARTH ===
    ColorAtom(
        name="Soil Brown",
        symbol="Eb",
        a_center=15,
        b_center=20,
        saturation_range=(10, 30),
        luminance_response=LuminanceResponse.INVERSE,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.NATURAL_EARTH,
        common_contexts=["dirt", "soil", "ground"]
    ),
    ColorAtom(
        name="Sand Beige",
        symbol="Sd",
        a_center=5,
        b_center=15,
        saturation_range=(5, 20),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.NATURAL_EARTH,
        common_contexts=["beach", "desert", "sand"]
    ),
    ColorAtom(
        name="Rock Gray",
        symbol="Rg",
        a_center=0,
        b_center=5,
        saturation_range=(0, 15),
        luminance_response=LuminanceResponse.UNIFORM,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.NATURAL_EARTH,
        common_contexts=["stone", "rock", "mountain"]
    ),
    
    # === NATURAL: WATER ===
    ColorAtom(
        name="Ocean Blue",
        symbol="Ob",
        a_center=-10,
        b_center=-30,
        saturation_range=(15, 40),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.GRADIENT,
        semantic_category=SemanticCategory.NATURAL_WATER,
        common_contexts=["ocean", "sea", "deep water"]
    ),
    ColorAtom(
        name="River Teal",
        symbol="Rt",
        a_center=-15,
        b_center=-15,
        saturation_range=(10, 30),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.GRADIENT,
        semantic_category=SemanticCategory.NATURAL_WATER,
        common_contexts=["river", "stream", "lake"]
    ),
    
    # === ORGANIC: SKIN ===
    ColorAtom(
        name="Light Skin",
        symbol="Sl",
        a_center=12,
        b_center=12,
        saturation_range=(10, 25),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.BLOB,
        semantic_category=SemanticCategory.ORGANIC_SKIN,
        common_contexts=["face", "hands", "portrait"]
    ),
    ColorAtom(
        name="Medium Skin",
        symbol="Sm",
        a_center=18,
        b_center=20,
        saturation_range=(15, 35),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.BLOB,
        semantic_category=SemanticCategory.ORGANIC_SKIN,
        common_contexts=["face", "hands", "portrait"]
    ),
    ColorAtom(
        name="Dark Skin",
        symbol="Sd",
        a_center=20,
        b_center=25,
        saturation_range=(20, 40),
        luminance_response=LuminanceResponse.INVERSE,
        spatial_behavior=SpatialBehavior.BLOB,
        semantic_category=SemanticCategory.ORGANIC_SKIN,
        common_contexts=["face", "hands", "portrait"]
    ),
    
    # === ORGANIC: WOOD ===
    ColorAtom(
        name="Light Wood",
        symbol="Wl",
        a_center=8,
        b_center=20,
        saturation_range=(10, 25),
        luminance_response=LuminanceResponse.PROPORTIONAL,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.ORGANIC_WOOD,
        common_contexts=["furniture", "floor", "pine"]
    ),
    ColorAtom(
        name="Dark Wood",
        symbol="Wd",
        a_center=12,
        b_center=15,
        saturation_range=(10, 30),
        luminance_response=LuminanceResponse.INVERSE,
        spatial_behavior=SpatialBehavior.TEXTURED,
        semantic_category=SemanticCategory.ORGANIC_WOOD,
        common_contexts=["furniture", "oak", "walnut"]
    ),
    
    # === LIGHT: SHADOWS & HIGHLIGHTS ===
    ColorAtom(
        name="Cool Shadow",
        symbol="Sc",
        a_center=-5,
        b_center=-10,
        saturation_range=(5, 15),
        luminance_response=LuminanceResponse.DARK_ONLY,
        spatial_behavior=SpatialBehavior.EDGE_BOUND,
        semantic_category=SemanticCategory.LIGHT_SHADOW,
        common_contexts=["shadow", "shade", "dark area"]
    ),
    ColorAtom(
        name="Warm Highlight",
        symbol="Hw",
        a_center=5,
        b_center=10,
        saturation_range=(5, 15),
        luminance_response=LuminanceResponse.BRIGHT_ONLY,
        spatial_behavior=SpatialBehavior.EDGE_BOUND,
        semantic_category=SemanticCategory.LIGHT_HIGHLIGHT,
        common_contexts=["highlight", "specular", "bright spot"]
    ),
    ColorAtom(
        name="Neutral Gray",
        symbol="Ng",
        a_center=0,
        b_center=0,
        saturation_range=(0, 5),
        luminance_response=LuminanceResponse.UNIFORM,
        spatial_behavior=SpatialBehavior.UNIFORM,
        semantic_category=SemanticCategory.LIGHT_SHADOW,
        common_contexts=["neutral", "achromatic", "gray"]
    ),
]


def organize_by_property(atoms: List[ColorAtom], property_name: str) -> Dict:
    """Organize atoms by a specific property."""
    organized = {}
    
    for atom in atoms:
        if property_name == "luminance_response":
            key = atom.luminance_response.value
        elif property_name == "spatial_behavior":
            key = atom.spatial_behavior.value
        elif property_name == "semantic_category":
            key = atom.semantic_category.value
        elif property_name == "hue":
            # Group by hue quadrant
            hue = atom.hue_angle
            if hue < 90:
                key = "warm (0-90°)"
            elif hue < 180:
                key = "yellow-green (90-180°)"
            elif hue < 270:
                key = "cool (180-270°)"
            else:
                key = "magenta (270-360°)"
        else:
            key = "unknown"
        
        if key not in organized:
            organized[key] = []
        organized[key].append(atom)
    
    return organized


def print_periodic_table():
    """Print the periodic table of colors."""
    print("=" * 80)
    print("THE PERIODIC TABLE OF COLOR KNOWLEDGE")
    print("=" * 80)
    print()
    print("Like elements have atomic properties, colors have geometric properties:")
    print("  - Luminance Response: How color changes with brightness")
    print("  - Spatial Behavior: How color varies across space")
    print("  - Semantic Category: What the color represents")
    print()
    
    # Group by semantic category
    by_category = organize_by_property(PERIODIC_TABLE, "semantic_category")
    
    for category, atoms in sorted(by_category.items()):
        print(f"\n{'='*40}")
        print(f"  {category.upper()}")
        print(f"{'='*40}")
        
        for atom in atoms:
            print(f"\n  [{atom.symbol}] {atom.name}")
            print(f"      ab: ({atom.a_center:+.0f}, {atom.b_center:+.0f})")
            print(f"      Hue: {atom.hue_angle:.0f}°, Max Sat: {atom.max_saturation:.0f}")
            print(f"      Luminance: {atom.luminance_response.value}")
            print(f"      Spatial: {atom.spatial_behavior.value}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nBy Luminance Response:")
    by_lum = organize_by_property(PERIODIC_TABLE, "luminance_response")
    for key, atoms in sorted(by_lum.items()):
        symbols = ", ".join(a.symbol for a in atoms)
        print(f"  {key}: {symbols}")
    
    print("\nBy Spatial Behavior:")
    by_spatial = organize_by_property(PERIODIC_TABLE, "spatial_behavior")
    for key, atoms in sorted(by_spatial.items()):
        symbols = ", ".join(a.symbol for a in atoms)
        print(f"  {key}: {symbols}")
    
    print("\nBy Hue Quadrant:")
    by_hue = organize_by_property(PERIODIC_TABLE, "hue")
    for key, atoms in sorted(by_hue.items()):
        symbols = ", ".join(a.symbol for a in atoms)
        print(f"  {key}: {symbols}")


def generate_knowledge_shapes():
    """
    Generate the geometric shapes for each color atom.
    
    This is the key insight: each color atom has a SHAPE that defines
    how it behaves. The shape is defined by:
        1. Position in ab space (hue/saturation)
        2. Luminance response curve
        3. Spatial correlation function
    """
    print("\n" + "=" * 80)
    print("GEOMETRIC SHAPES OF COLOR KNOWLEDGE")
    print("=" * 80)
    
    print("\nEach color atom has a geometric shape defined by:")
    print("  1. ab position (where in color space)")
    print("  2. Luminance curve (how saturation varies with brightness)")
    print("  3. Spatial kernel (how color spreads)")
    print()
    
    for atom in PERIODIC_TABLE[:5]:  # Show first 5 as examples
        print(f"\n[{atom.symbol}] {atom.name}")
        print("-" * 40)
        
        # 1. ab position
        print(f"  Position: ({atom.a_center:+.0f}, {atom.b_center:+.0f})")
        
        # 2. Luminance curve
        print(f"  Luminance curve ({atom.luminance_response.value}):")
        for lum in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if atom.luminance_response == LuminanceResponse.DARK_ONLY:
                sat = atom.max_saturation * (1 - lum)
            elif atom.luminance_response == LuminanceResponse.BRIGHT_ONLY:
                sat = atom.max_saturation * lum
            elif atom.luminance_response == LuminanceResponse.PROPORTIONAL:
                sat = atom.max_saturation * (0.3 + 0.7 * lum)
            elif atom.luminance_response == LuminanceResponse.INVERSE:
                sat = atom.max_saturation * (1 - 0.5 * lum)
            else:
                sat = atom.max_saturation
            print(f"    L={lum:.2f} → Sat={sat:.1f}")
        
        # 3. Spatial behavior
        print(f"  Spatial kernel ({atom.spatial_behavior.value}):")
        if atom.spatial_behavior == SpatialBehavior.UNIFORM:
            print("    [1 1 1]")
            print("    [1 1 1]  (constant)")
            print("    [1 1 1]")
        elif atom.spatial_behavior == SpatialBehavior.GRADIENT:
            print("    [0.5 0.7 1.0]")
            print("    [0.5 0.7 1.0]  (directional)")
            print("    [0.5 0.7 1.0]")
        elif atom.spatial_behavior == SpatialBehavior.TEXTURED:
            print("    [0.8 1.0 0.9]")
            print("    [1.0 0.7 1.0]  (variable)")
            print("    [0.9 1.0 0.8]")
        elif atom.spatial_behavior == SpatialBehavior.EDGE_BOUND:
            print("    [0.1 0.5 0.1]")
            print("    [0.5 1.0 0.5]  (edge-following)")
            print("    [0.1 0.5 0.1]")
        elif atom.spatial_behavior == SpatialBehavior.BLOB:
            print("    [0.5 0.8 0.5]")
            print("    [0.8 1.0 0.8]  (gaussian)")
            print("    [0.5 0.8 0.5]")


def main():
    """Main entry point."""
    print_periodic_table()
    generate_knowledge_shapes()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
The periodic table of colors shows that color knowledge has STRUCTURE:

1. POSITION (ab values)
   - Each color has a specific location in color space
   - This is the "atomic number" of colors

2. LUMINANCE RESPONSE (how it changes with brightness)
   - Some colors only appear in shadows (cool shadow)
   - Some colors only appear in highlights (warm highlight)
   - Some scale proportionally (most natural colors)
   - This is like "electron configuration"

3. SPATIAL BEHAVIOR (how it spreads)
   - Uniform: sky, overcast
   - Gradient: sunset, water
   - Textured: foliage, earth
   - Edge-bound: shadows
   - Blob: skin, objects
   - This is like "bonding behavior"

4. SEMANTIC CATEGORY (what it represents)
   - Natural: sky, vegetation, earth, water
   - Organic: skin, wood
   - Light: shadows, highlights
   - This is like "element groups" (noble gases, metals, etc.)

By organizing color knowledge this way, we can:
- Predict what colors should appear in different contexts
- Generate appropriate colors from semantic labels
- Transfer knowledge between similar categories
- Build a "chemistry" of color combinations
""")
    
    print("\n" + "=" * 80)
    print(f"Total color atoms defined: {len(PERIODIC_TABLE)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
