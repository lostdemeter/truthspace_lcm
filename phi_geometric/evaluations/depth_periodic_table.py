#!/usr/bin/env python3
"""
Testing the Periodic Table Metaphor: Depth Estimation

Can we apply the same framework to a completely different problem?

For colorization, we had:
    - Position: ab values
    - Luminance Response: how color changes with brightness
    - Spatial Behavior: how color spreads
    - Category: semantic meaning

For depth, we need to find analogous properties.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum


# =============================================================================
# ATTEMPT 1: Direct Translation
# =============================================================================

class DepthRange(Enum):
    """Where in depth space does this atom live?"""
    NEAR = "near"           # Close to camera (0-2m)
    MID = "mid"             # Middle distance (2-10m)
    FAR = "far"             # Far away (10-50m)
    INFINITY = "infinity"   # Sky, horizon (50m+)


class EdgeResponse(Enum):
    """How does depth change at edges?"""
    SHARP = "sharp"         # Discontinuous (object boundaries)
    GRADUAL = "gradual"     # Smooth transition (curved surfaces)
    NONE = "none"           # No edge response (flat surfaces)


class SurfaceType(Enum):
    """What kind of surface is this?"""
    PLANAR = "planar"       # Flat (walls, floors)
    CURVED = "curved"       # Smooth curves (spheres, cylinders)
    TEXTURED = "textured"   # Rough/irregular (foliage, rocks)
    TRANSPARENT = "transparent"  # See-through (glass, water)


class SemanticDepthCategory(Enum):
    """What does this depth represent?"""
    SKY = "sky"
    GROUND = "ground"
    BUILDING = "building"
    VEGETATION = "vegetation"
    PERSON = "person"
    VEHICLE = "vehicle"
    OBJECT = "object"


@dataclass
class DepthAtom:
    """A fundamental unit of depth knowledge."""
    name: str
    symbol: str
    depth_range: DepthRange
    typical_depth: float  # meters
    edge_response: EdgeResponse
    surface_type: SurfaceType
    category: SemanticDepthCategory
    common_contexts: List[str]
    
    def describe(self) -> str:
        return (
            f"{self.symbol} - {self.name}\n"
            f"  Depth: {self.typical_depth}m ({self.depth_range.value})\n"
            f"  Edge: {self.edge_response.value}\n"
            f"  Surface: {self.surface_type.value}\n"
            f"  Category: {self.category.value}"
        )


# The Periodic Table of Depth
DEPTH_PERIODIC_TABLE = [
    # === SKY ===
    DepthAtom("Sky", "Sk", DepthRange.INFINITY, 1000.0,
              EdgeResponse.NONE, SurfaceType.PLANAR,
              SemanticDepthCategory.SKY, ["outdoor", "daytime"]),
    
    # === GROUND ===
    DepthAtom("Near Ground", "Gn", DepthRange.NEAR, 1.5,
              EdgeResponse.GRADUAL, SurfaceType.PLANAR,
              SemanticDepthCategory.GROUND, ["floor", "pavement"]),
    DepthAtom("Far Ground", "Gf", DepthRange.FAR, 20.0,
              EdgeResponse.GRADUAL, SurfaceType.PLANAR,
              SemanticDepthCategory.GROUND, ["field", "road"]),
    
    # === BUILDINGS ===
    DepthAtom("Wall", "Wa", DepthRange.MID, 5.0,
              EdgeResponse.SHARP, SurfaceType.PLANAR,
              SemanticDepthCategory.BUILDING, ["indoor", "urban"]),
    DepthAtom("Facade", "Fa", DepthRange.FAR, 15.0,
              EdgeResponse.SHARP, SurfaceType.PLANAR,
              SemanticDepthCategory.BUILDING, ["street", "urban"]),
    
    # === VEGETATION ===
    DepthAtom("Bush", "Bu", DepthRange.NEAR, 2.0,
              EdgeResponse.GRADUAL, SurfaceType.TEXTURED,
              SemanticDepthCategory.VEGETATION, ["garden", "park"]),
    DepthAtom("Tree", "Tr", DepthRange.MID, 8.0,
              EdgeResponse.GRADUAL, SurfaceType.TEXTURED,
              SemanticDepthCategory.VEGETATION, ["forest", "park"]),
    
    # === PEOPLE ===
    DepthAtom("Person Near", "Pn", DepthRange.NEAR, 1.5,
              EdgeResponse.SHARP, SurfaceType.CURVED,
              SemanticDepthCategory.PERSON, ["portrait", "indoor"]),
    DepthAtom("Person Far", "Pf", DepthRange.MID, 5.0,
              EdgeResponse.SHARP, SurfaceType.CURVED,
              SemanticDepthCategory.PERSON, ["street", "crowd"]),
    
    # === VEHICLES ===
    DepthAtom("Car Near", "Cn", DepthRange.NEAR, 3.0,
              EdgeResponse.SHARP, SurfaceType.CURVED,
              SemanticDepthCategory.VEHICLE, ["parking", "street"]),
    DepthAtom("Car Far", "Cf", DepthRange.FAR, 20.0,
              EdgeResponse.SHARP, SurfaceType.CURVED,
              SemanticDepthCategory.VEHICLE, ["highway", "street"]),
    
    # === OBJECTS ===
    DepthAtom("Table", "Tb", DepthRange.NEAR, 1.0,
              EdgeResponse.SHARP, SurfaceType.PLANAR,
              SemanticDepthCategory.OBJECT, ["indoor", "office"]),
    DepthAtom("Chair", "Ch", DepthRange.NEAR, 1.2,
              EdgeResponse.SHARP, SurfaceType.CURVED,
              SemanticDepthCategory.OBJECT, ["indoor", "office"]),
]


# =============================================================================
# EVALUATION: Does the metaphor work?
# =============================================================================

def evaluate_metaphor():
    """Evaluate whether the periodic table metaphor works for depth."""
    print("=" * 80)
    print("TESTING THE PERIODIC TABLE METAPHOR: DEPTH ESTIMATION")
    print("=" * 80)
    
    print("\n## The Translation")
    print("-" * 40)
    print("Color Property      → Depth Property")
    print("-" * 40)
    print("Position (ab)       → Depth Range (near/mid/far/infinity)")
    print("Luminance Response  → Edge Response (sharp/gradual/none)")
    print("Spatial Behavior    → Surface Type (planar/curved/textured)")
    print("Semantic Category   → Semantic Category (sky/ground/person/...)")
    
    print("\n## The Depth Atoms")
    print("-" * 40)
    for atom in DEPTH_PERIODIC_TABLE:
        print(f"\n{atom.describe()}")
    
    print("\n" + "=" * 80)
    print("EVALUATION: Does the metaphor work?")
    print("=" * 80)
    
    print("\n### What Works ✓")
    print("""
1. POSITION → DEPTH RANGE
   - Just like colors have positions in ab space, depths have positions
   - The "atomic number" analogy holds: sky=∞, ground=variable, objects=near
   
2. SEMANTIC CATEGORY → SEMANTIC CATEGORY
   - Direct translation: sky, ground, person, vehicle, etc.
   - This is the "element groups" analogy
   
3. SPATIAL BEHAVIOR → SURFACE TYPE
   - Planar surfaces = uniform depth
   - Curved surfaces = gradual depth change
   - Textured surfaces = noisy depth
   - This maps well!
""")
    
    print("### What Doesn't Work ✗")
    print("""
1. LUMINANCE RESPONSE → EDGE RESPONSE
   - The analogy is FORCED
   - Luminance response is about how color CHANGES with brightness
   - Edge response is about DISCONTINUITIES
   - These are different concepts
   
2. MISSING: Occlusion Relationships
   - Depth has a concept colors don't: occlusion
   - Object A is IN FRONT OF object B
   - This is a RELATIONAL property, not an atomic property
   
3. MISSING: Scale Ambiguity
   - A person at 2m looks the same as a person at 20m (just smaller)
   - Depth requires CONTEXT to disambiguate
   - Colors don't have this problem
""")
    
    print("\n### The Refined Insight")
    print("""
The periodic table metaphor works for SOME properties but not all.

GOOD FOR:
- Properties that are INTRINSIC to the atom (position, category)
- Properties that are INDEPENDENT (don't depend on other atoms)

BAD FOR:
- Properties that are RELATIONAL (occlusion, relative depth)
- Properties that require CONTEXT (scale, distance)

This suggests we need TWO types of knowledge:
1. ATOMIC knowledge (periodic table) - intrinsic properties
2. MOLECULAR knowledge (chemistry) - relationships between atoms
""")
    
    return True


# =============================================================================
# ATTEMPT 2: Refined Framework
# =============================================================================

def refined_framework():
    """Propose a refined framework based on evaluation."""
    print("\n" + "=" * 80)
    print("REFINED FRAMEWORK: Atoms + Molecules")
    print("=" * 80)
    
    print("""
## Level 1: Atoms (Intrinsic Properties)

Properties that belong to a single knowledge unit:
- Position in feature space
- Category/type
- Texture/surface properties
- Typical range/scale

## Level 2: Molecules (Relational Properties)

Properties that describe relationships between atoms:
- Occlusion (A in front of B)
- Adjacency (A next to B)
- Containment (A inside B)
- Causation (A causes B)

## Level 3: Reactions (Transformations)

How atoms and molecules change:
- Lighting changes color atoms
- Viewpoint changes depth atoms
- Time changes motion atoms

## The Revised Metaphor

| Chemistry | Geometric AI |
|-----------|--------------|
| Atom | Knowledge unit (intrinsic) |
| Molecule | Knowledge relationship (relational) |
| Reaction | Knowledge transformation |
| Periodic table | Catalog of atoms |
| Molecular formulas | Relationship patterns |
| Reaction equations | Transformation rules |
""")
    
    print("\n## Testing the Refined Framework on Depth")
    print("-" * 40)
    
    print("""
### Atoms (what we have)
- Sky: depth=∞, surface=planar
- Ground: depth=variable, surface=planar
- Person: depth=near, surface=curved

### Molecules (what we need to add)
- Person ON Ground: person.depth < ground.depth at same (x,y)
- Sky BEHIND Everything: sky.depth > all other depths
- Car ON Road: car.depth ≈ road.depth, car occludes road

### Reactions (transformations)
- Zoom In: all depths scale by factor
- Move Forward: near depths decrease, far depths stay similar
- Tilt Down: ground depth gradient changes
""")
    
    print("\n## Does This Work Better?")
    print("-" * 40)
    print("""
YES! The refined framework captures:
1. Intrinsic properties (atoms) - what we had
2. Relational properties (molecules) - what was missing
3. Transformations (reactions) - dynamic behavior

This is more complete than just the periodic table.
""")


# =============================================================================
# ATTEMPT 3: Apply to Language
# =============================================================================

def test_on_language():
    """Test the framework on language/text understanding."""
    print("\n" + "=" * 80)
    print("TESTING ON LANGUAGE: Does the framework generalize?")
    print("=" * 80)
    
    print("""
## Language Atoms

| Property | Language Analog |
|----------|-----------------|
| Position | Embedding location |
| Category | Part of speech, semantic field |
| Surface | Morphology (prefixes, suffixes) |
| Range | Frequency (common vs rare) |

### Example Language Atoms
- Noun.Person: position=person_region, category=noun, common
- Verb.Motion: position=motion_region, category=verb, common
- Adj.Color: position=color_region, category=adjective, common

## Language Molecules

| Relationship | Example |
|--------------|---------|
| Subject-Verb | "dog runs" |
| Verb-Object | "eat apple" |
| Modifier | "red apple" |
| Preposition | "on the table" |

## Language Reactions

| Transformation | Example |
|----------------|---------|
| Tense change | run → ran |
| Negation | is → is not |
| Question | statement → question |
| Passive | "dog bites man" → "man is bitten by dog" |
""")
    
    print("\n## Evaluation")
    print("-" * 40)
    print("""
The framework DOES generalize to language:
- Atoms = words/morphemes with intrinsic properties
- Molecules = syntactic relationships
- Reactions = grammatical transformations

But there's a key difference:
- Colors/depths are CONTINUOUS (ab values, meters)
- Language is DISCRETE (words, categories)

The framework works, but the IMPLEMENTATION differs.
""")


def main():
    """Run all tests."""
    # Test 1: Direct translation to depth
    evaluate_metaphor()
    
    # Test 2: Refined framework
    refined_framework()
    
    # Test 3: Apply to language
    test_on_language()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print("""
## Is the Periodic Table the Best Metaphor?

PARTIALLY. It captures ATOMIC (intrinsic) knowledge well, but misses:
- Relational knowledge (molecules)
- Transformational knowledge (reactions)

## The Refined Metaphor: Chemistry, not just Elements

| Level | What it captures | Example |
|-------|------------------|---------|
| Atoms | Intrinsic properties | "sky is blue" |
| Molecules | Relationships | "sky is above ground" |
| Reactions | Transformations | "sunset changes sky color" |

## Should We Iterate?

YES. The next iteration should:
1. Keep the atomic properties (position, category, surface, range)
2. Add molecular relationships (occlusion, adjacency, containment)
3. Add reaction rules (how knowledge transforms)

## The Guide

A refined guide for characterizing knowledge shapes:

### Step 1: Identify Atoms
- What are the fundamental units?
- What intrinsic properties do they have?
- Organize into a periodic table

### Step 2: Identify Molecules
- How do atoms relate to each other?
- What patterns of relationship exist?
- Define molecular formulas

### Step 3: Identify Reactions
- How does knowledge transform?
- What triggers transformations?
- Define reaction equations

### Step 4: Build the Knowledge Base
- Catalog all atoms
- Define all valid molecules
- Specify all reactions

### Step 5: Use for Geometric AI
- Project shapes using atomic properties
- Enforce molecular constraints
- Apply reaction rules for dynamics
""")


if __name__ == "__main__":
    main()
