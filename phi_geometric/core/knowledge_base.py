#!/usr/bin/env python3
"""
Knowledge Base: Atoms, Molecules, and Reactions

The Knowledge Chemistry framework for geometric AI.

Levels:
    1. Atoms - Intrinsic properties of knowledge units
    2. Molecules - Relationships between atoms
    3. Reactions - Transformations of atoms/molecules

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from enum import Enum


# =============================================================================
# LEVEL 1: ATOMS
# =============================================================================

class AtomProperty(Enum):
    """Standard atomic properties."""
    POSITION = "position"           # Location in feature space
    CATEGORY = "category"           # Semantic category
    SURFACE = "surface"             # Texture/behavior type
    RANGE = "range"                 # Typical value range
    RESPONSE = "response"           # How it responds to context


@dataclass
class KnowledgeAtom:
    """
    A fundamental unit of knowledge.
    
    Like an element in the periodic table, an atom has intrinsic
    properties that define its behavior.
    
    Attributes:
        name: Human-readable name
        symbol: Short symbol (like element symbols)
        position: Location in feature space (tuple of floats)
        category: Semantic category (string)
        surface: Texture/behavior type (string)
        value_range: Typical value range (min, max)
        response_curve: Function mapping context → value modifier
        metadata: Additional properties
    """
    name: str
    symbol: str
    position: Tuple[float, ...]
    category: str
    surface: str = "uniform"
    value_range: Tuple[float, float] = (0.0, 1.0)
    response_curve: Optional[Callable[[float], float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_value(self, context: float = 0.5) -> Tuple[float, ...]:
        """
        Get the atom's value given a context.
        
        Args:
            context: Context value (e.g., luminance for colors)
            
        Returns:
            Position modified by response curve
        """
        if self.response_curve is None:
            return self.position
        
        modifier = self.response_curve(context)
        return tuple(p * modifier for p in self.position)
    
    def __repr__(self):
        return f"Atom({self.symbol}: {self.name})"


# =============================================================================
# LEVEL 2: MOLECULES
# =============================================================================

class RelationType(Enum):
    """Types of relationships between atoms."""
    OCCLUSION = "occlusion"         # A in front of B
    ADJACENCY = "adjacency"         # A next to B
    CONTAINMENT = "containment"     # A inside B
    CAUSATION = "causation"         # A causes B
    SIMILARITY = "similarity"       # A like B
    ORDERING = "ordering"           # A before B
    HIERARCHY = "hierarchy"         # A parent of B


@dataclass
class KnowledgeMolecule:
    """
    A relationship between atoms.
    
    Molecules define how atoms relate to each other, like
    chemical bonds define how elements combine.
    
    Attributes:
        name: Human-readable name
        atoms: List of atoms in this molecule
        relation: Type of relationship
        constraint: Function that checks if relationship holds
        strength: How strong the relationship is (0-1)
        metadata: Additional properties
    """
    name: str
    atoms: List[KnowledgeAtom]
    relation: RelationType
    constraint: Optional[Callable[..., bool]] = None
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, *values) -> bool:
        """Check if the molecular constraint is satisfied."""
        if self.constraint is None:
            return True
        return self.constraint(*values)
    
    def __repr__(self):
        atom_symbols = "-".join(a.symbol for a in self.atoms)
        return f"Molecule({atom_symbols}: {self.relation.value})"


# =============================================================================
# LEVEL 3: REACTIONS
# =============================================================================

class ReactionTrigger(Enum):
    """What triggers a reaction."""
    LIGHTING = "lighting"           # Light changes
    VIEWPOINT = "viewpoint"         # Camera/view changes
    TIME = "time"                   # Temporal changes
    CONTEXT = "context"             # Context changes
    USER = "user"                   # User input


@dataclass
class KnowledgeReaction:
    """
    A transformation of atoms or molecules.
    
    Reactions define how knowledge changes, like chemical
    reactions define how compounds transform.
    
    Attributes:
        name: Human-readable name
        trigger: What triggers this reaction
        inputs: Atoms/molecules before transformation
        transform: Function that performs the transformation
        metadata: Additional properties
    """
    name: str
    trigger: ReactionTrigger
    inputs: List[str]  # Atom/molecule symbols
    transform: Callable[[Any, float], Any]  # (value, trigger_strength) → new_value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, value: Any, trigger_strength: float = 1.0) -> Any:
        """Apply the reaction transformation."""
        return self.transform(value, trigger_strength)
    
    def __repr__(self):
        return f"Reaction({self.name}: {self.trigger.value})"


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class KnowledgeBase:
    """
    A complete knowledge base with atoms, molecules, and reactions.
    
    This is the "chemistry set" for a specific domain.
    
    Example:
        kb = KnowledgeBase("colorization")
        
        # Add atoms
        kb.add_atom(KnowledgeAtom("Sky Blue", "Sb", (-5, -40), "sky"))
        kb.add_atom(KnowledgeAtom("Grass Green", "Gg", (-30, 30), "vegetation"))
        
        # Add molecules
        kb.add_molecule(KnowledgeMolecule(
            "SkyAboveGround",
            [kb.get_atom("Sb"), kb.get_atom("Gg")],
            RelationType.ADJACENCY
        ))
        
        # Add reactions
        kb.add_reaction(KnowledgeReaction(
            "Sunset",
            ReactionTrigger.TIME,
            ["Sb"],
            lambda ab, t: (ab[0] + 35*t, ab[1] + 80*t)  # Blue → Orange
        ))
    """
    
    def __init__(self, domain: str):
        self.domain = domain
        self.atoms: Dict[str, KnowledgeAtom] = {}
        self.molecules: Dict[str, KnowledgeMolecule] = {}
        self.reactions: Dict[str, KnowledgeReaction] = {}
        
        # Index by category for fast lookup
        self._by_category: Dict[str, List[str]] = {}
    
    def add_atom(self, atom: KnowledgeAtom):
        """Add an atom to the knowledge base."""
        self.atoms[atom.symbol] = atom
        
        # Index by category
        if atom.category not in self._by_category:
            self._by_category[atom.category] = []
        self._by_category[atom.category].append(atom.symbol)
    
    def add_molecule(self, molecule: KnowledgeMolecule):
        """Add a molecule to the knowledge base."""
        self.molecules[molecule.name] = molecule
    
    def add_reaction(self, reaction: KnowledgeReaction):
        """Add a reaction to the knowledge base."""
        self.reactions[reaction.name] = reaction
    
    def get_atom(self, symbol: str) -> Optional[KnowledgeAtom]:
        """Get an atom by symbol."""
        return self.atoms.get(symbol)
    
    def get_atoms_by_category(self, category: str) -> List[KnowledgeAtom]:
        """Get all atoms in a category."""
        symbols = self._by_category.get(category, [])
        return [self.atoms[s] for s in symbols]
    
    def get_molecule(self, name: str) -> Optional[KnowledgeMolecule]:
        """Get a molecule by name."""
        return self.molecules.get(name)
    
    def get_reaction(self, name: str) -> Optional[KnowledgeReaction]:
        """Get a reaction by name."""
        return self.reactions.get(name)
    
    def get_reactions_by_trigger(self, trigger: ReactionTrigger) -> List[KnowledgeReaction]:
        """Get all reactions with a specific trigger."""
        return [r for r in self.reactions.values() if r.trigger == trigger]
    
    def find_atom_for_context(
        self, 
        category: str, 
        context: float
    ) -> Optional[KnowledgeAtom]:
        """
        Find the best atom for a given category and context.
        
        Args:
            category: Semantic category
            context: Context value (e.g., luminance)
            
        Returns:
            Best matching atom or None
        """
        atoms = self.get_atoms_by_category(category)
        if not atoms:
            return None
        
        # For now, return first atom
        # Could be extended to select based on context
        return atoms[0]
    
    def apply_reactions(
        self, 
        value: Any, 
        trigger: ReactionTrigger, 
        strength: float = 1.0
    ) -> Any:
        """
        Apply all reactions with the given trigger.
        
        Args:
            value: Current value
            trigger: Reaction trigger
            strength: Trigger strength
            
        Returns:
            Transformed value
        """
        reactions = self.get_reactions_by_trigger(trigger)
        for reaction in reactions:
            value = reaction.apply(value, strength)
        return value
    
    def check_molecule(self, name: str, *values) -> bool:
        """Check if a molecular constraint is satisfied."""
        molecule = self.get_molecule(name)
        if molecule is None:
            return True
        return molecule.check(*values)
    
    def stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        return {
            "domain": self.domain,
            "atoms": len(self.atoms),
            "molecules": len(self.molecules),
            "reactions": len(self.reactions),
            "categories": len(self._by_category),
        }
    
    def describe(self) -> str:
        """Human-readable description."""
        lines = [
            f"KnowledgeBase: {self.domain}",
            f"  Atoms: {len(self.atoms)}",
            f"  Molecules: {len(self.molecules)}",
            f"  Reactions: {len(self.reactions)}",
            "",
            "  Categories:",
        ]
        for cat, symbols in self._by_category.items():
            lines.append(f"    {cat}: {', '.join(symbols)}")
        
        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_color_knowledge_base() -> KnowledgeBase:
    """
    Create a knowledge base for colorization.
    
    This encodes the color atoms, molecules, and reactions
    from the periodic table of colors.
    """
    kb = KnowledgeBase("colorization")
    
    # Response curves
    def bright_only(lum): return lum
    def dark_only(lum): return 1 - lum
    def proportional(lum): return 0.3 + 0.7 * lum
    def inverse(lum): return 1 - 0.5 * lum
    def uniform(lum): return 1.0
    
    # === ATOMS ===
    
    # Sky
    kb.add_atom(KnowledgeAtom("Clear Sky Blue", "Sb", (-5, -40), "sky", 
                              "gradient", (-50, 0), bright_only))
    kb.add_atom(KnowledgeAtom("Sunset Orange", "So", (30, 40), "sky",
                              "gradient", (0, 60), bright_only))
    kb.add_atom(KnowledgeAtom("Overcast Gray", "Og", (0, -5), "sky",
                              "uniform", (-10, 0), uniform))
    
    # Vegetation
    kb.add_atom(KnowledgeAtom("Grass Green", "Gg", (-30, 30), "vegetation",
                              "textured", (-50, 50), proportional))
    kb.add_atom(KnowledgeAtom("Forest Green", "Fg", (-25, 15), "vegetation",
                              "textured", (-40, 30), proportional))
    kb.add_atom(KnowledgeAtom("Autumn Orange", "Ao", (20, 40), "vegetation",
                              "textured", (0, 55), proportional))
    
    # Earth
    kb.add_atom(KnowledgeAtom("Soil Brown", "Eb", (15, 20), "earth",
                              "textured", (0, 30), inverse))
    kb.add_atom(KnowledgeAtom("Sand Beige", "Sd", (5, 15), "earth",
                              "textured", (0, 20), proportional))
    kb.add_atom(KnowledgeAtom("Rock Gray", "Rg", (0, 5), "earth",
                              "textured", (-5, 15), uniform))
    
    # Water
    kb.add_atom(KnowledgeAtom("Ocean Blue", "Ob", (-10, -30), "water",
                              "gradient", (-40, 0), proportional))
    kb.add_atom(KnowledgeAtom("River Teal", "Rt", (-15, -15), "water",
                              "gradient", (-30, 0), proportional))
    
    # Skin
    kb.add_atom(KnowledgeAtom("Light Skin", "Sl", (12, 12), "skin",
                              "blob", (5, 25), proportional))
    kb.add_atom(KnowledgeAtom("Medium Skin", "Sm", (18, 20), "skin",
                              "blob", (10, 35), proportional))
    kb.add_atom(KnowledgeAtom("Dark Skin", "Sk", (20, 25), "skin",
                              "blob", (15, 40), inverse))
    
    # Wood
    kb.add_atom(KnowledgeAtom("Light Wood", "Wl", (8, 20), "wood",
                              "textured", (5, 25), proportional))
    kb.add_atom(KnowledgeAtom("Dark Wood", "Wd", (12, 15), "wood",
                              "textured", (5, 30), inverse))
    
    # Light
    kb.add_atom(KnowledgeAtom("Cool Shadow", "Sc", (-5, -10), "shadow",
                              "edge_bound", (-15, 0), dark_only))
    kb.add_atom(KnowledgeAtom("Warm Highlight", "Hw", (5, 10), "highlight",
                              "edge_bound", (0, 15), bright_only))
    kb.add_atom(KnowledgeAtom("Neutral Gray", "Ng", (0, 0), "neutral",
                              "uniform", (-5, 5), uniform))
    
    # === MOLECULES ===
    
    # Adjacency relationships
    kb.add_molecule(KnowledgeMolecule(
        "SkyAboveGround",
        [kb.get_atom("Sb"), kb.get_atom("Gg")],
        RelationType.ADJACENCY,
        lambda sky_y, ground_y: sky_y < ground_y,
        strength=1.0
    ))
    
    kb.add_molecule(KnowledgeMolecule(
        "ShadowOnSurface",
        [kb.get_atom("Sc"), kb.get_atom("Ng")],
        RelationType.OCCLUSION,
        lambda shadow_lum, surface_lum: shadow_lum < surface_lum,
        strength=0.8
    ))
    
    kb.add_molecule(KnowledgeMolecule(
        "WaterReflectsSky",
        [kb.get_atom("Ob"), kb.get_atom("Sb")],
        RelationType.SIMILARITY,
        lambda water_b, sky_b: abs(water_b - sky_b) < 20,
        strength=0.7
    ))
    
    # === REACTIONS ===
    
    # Lighting reactions
    kb.add_reaction(KnowledgeReaction(
        "Sunset",
        ReactionTrigger.LIGHTING,
        ["Sb"],
        lambda ab, t: (ab[0] + 35 * t, ab[1] + 80 * t)  # Blue → Orange
    ))
    
    kb.add_reaction(KnowledgeReaction(
        "Shadow",
        ReactionTrigger.LIGHTING,
        ["*"],
        lambda ab, t: (ab[0] - 5 * t, ab[1] - 10 * t)  # Shift cool
    ))
    
    kb.add_reaction(KnowledgeReaction(
        "Highlight",
        ReactionTrigger.LIGHTING,
        ["*"],
        lambda ab, t: (ab[0] + 5 * t, ab[1] + 10 * t)  # Shift warm
    ))
    
    return kb


def test_knowledge_base():
    """Test the knowledge base."""
    print("=" * 60)
    print("KNOWLEDGE BASE TEST")
    print("=" * 60)
    
    kb = create_color_knowledge_base()
    print(kb.describe())
    
    # Test atom lookup
    print("\n--- Atom Lookup ---")
    sky = kb.get_atom("Sb")
    print(f"  Sky Blue: {sky}")
    print(f"    Position: {sky.position}")
    print(f"    At lum=0.2: {sky.get_value(0.2)}")
    print(f"    At lum=0.8: {sky.get_value(0.8)}")
    
    # Test category lookup
    print("\n--- Category Lookup ---")
    vegetation = kb.get_atoms_by_category("vegetation")
    print(f"  Vegetation atoms: {vegetation}")
    
    # Test molecule
    print("\n--- Molecule Check ---")
    sky_ground = kb.get_molecule("SkyAboveGround")
    print(f"  {sky_ground}")
    print(f"    Check(10, 50): {sky_ground.check(10, 50)}")  # sky_y < ground_y
    print(f"    Check(50, 10): {sky_ground.check(50, 10)}")  # Invalid
    
    # Test reaction
    print("\n--- Reaction Application ---")
    sunset = kb.get_reaction("Sunset")
    print(f"  {sunset}")
    original = (-5, -40)
    transformed = sunset.apply(original, 1.0)
    print(f"    Original: {original}")
    print(f"    After sunset (t=1.0): {transformed}")
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE TEST COMPLETE")
    print("=" * 60)
    
    return kb


if __name__ == "__main__":
    test_knowledge_base()
