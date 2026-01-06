"""
Quaternion Layers Experiment (Design 104, Option 2)

Explores using quaternion structure to organize φ-lattice dimensions:

Q = w + xi + yj + zk

Where each component is a 4D φ-lattice:
  w = Core Semantic  [domain, specificity, intent, formality]
  x = Grammatical    [tense, aspect, mood, voice]
  y = Contextual     [register, evidentiality, politeness, emphasis]
  z = Reserved       [for future/emergent dimensions]

Key insight: Predictable structure enables navigation AND generation.
If we know the path through the structure, we can traverse it both ways.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

class SemanticDim(IntEnum):
    """Core Semantic dimensions (w component)"""
    DOMAIN = 0       # What area of knowledge
    SPECIFICITY = 1  # How specific
    INTENT = 2       # What response expected
    FORMALITY = 3    # How formal


class GrammaticalDim(IntEnum):
    """Grammatical dimensions (x component)"""
    TENSE = 0        # past/present/future
    ASPECT = 1       # perfective/imperfective/progressive
    MOOD = 2         # indicative/subjunctive/imperative
    VOICE = 3        # active/passive


class ContextualDim(IntEnum):
    """Contextual dimensions (y component)"""
    REGISTER = 0     # formal/informal/technical/casual
    EVIDENTIALITY = 1  # direct/reported/inferred
    POLITENESS = 2   # neutral/polite/humble/honorific
    EMPHASIS = 3     # neutral/emphatic/contrastive


class ReservedDim(IntEnum):
    """Reserved dimensions (z component) - for future use"""
    RESERVED_0 = 0
    RESERVED_1 = 1
    RESERVED_2 = 2
    RESERVED_3 = 3


# Level meanings for each dimension - keyed by (layer, dim_index)
SEMANTIC_MEANINGS = {
    0: {3: 'hard_science', 2: 'technology', 1: 'general', 0: 'meta', -1: 'social'},  # DOMAIN
    1: {3: 'very_specific', 2: 'specific', 1: 'general', 0: 'vague'},  # SPECIFICITY
    2: {3: 'deep_explain', 2: 'explain', 1: 'inform', 0: 'acknowledge', -1: 'social'},  # INTENT
    3: {2: 'academic', 1: 'professional', 0: 'casual', -1: 'informal'},  # FORMALITY
}

GRAMMATICAL_MEANINGS = {
    0: {2: 'future', 1: 'present', 0: 'past', -1: 'timeless'},  # TENSE
    1: {2: 'prospective', 1: 'progressive', 0: 'perfective', -1: 'habitual'},  # ASPECT
    2: {2: 'imperative', 1: 'subjunctive', 0: 'indicative', -1: 'conditional'},  # MOOD
    3: {1: 'active', 0: 'passive', -1: 'middle'},  # VOICE
}

CONTEXTUAL_MEANINGS = {
    0: {2: 'technical', 1: 'formal', 0: 'neutral', -1: 'casual', -2: 'intimate'},  # REGISTER
    1: {2: 'direct', 1: 'reported', 0: 'inferred', -1: 'assumed'},  # EVIDENTIALITY
    2: {2: 'honorific', 1: 'polite', 0: 'neutral', -1: 'familiar'},  # POLITENESS
    3: {2: 'emphatic', 1: 'contrastive', 0: 'neutral', -1: 'diminished'},  # EMPHASIS
}


# =============================================================================
# QUATERNION POSITION
# =============================================================================

@dataclass
class QuaternionPosition:
    """
    A position in quaternion-layered φ-space.
    
    Each component (w, x, y, z) is a 4D vector of φ-levels.
    Total: 16 dimensions organized hierarchically.
    """
    w: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Semantic
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Grammatical
    y: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Contextual
    z: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Reserved
    
    @classmethod
    def from_levels(cls, w_levels: List[int] = None, 
                    x_levels: List[int] = None,
                    y_levels: List[int] = None,
                    z_levels: List[int] = None) -> 'QuaternionPosition':
        """Create position from φ-level indices."""
        def levels_to_pos(levels):
            if levels is None:
                return np.array([PHI ** 0] * 4)  # Default: all at level 0
            return np.array([PHI ** k for k in levels])
        
        return cls(
            w=levels_to_pos(w_levels),
            x=levels_to_pos(x_levels),
            y=levels_to_pos(y_levels),
            z=levels_to_pos(z_levels)
        )
    
    def to_levels(self) -> Dict[str, List[int]]:
        """Convert position back to φ-level indices."""
        def pos_to_levels(pos):
            levels = []
            for v in pos:
                if abs(v) < 1e-10:
                    levels.append(-15)
                else:
                    k = round(np.log(abs(v)) / np.log(PHI))
                    levels.append(max(-15, min(15, k)))
            return levels
        
        return {
            'w': pos_to_levels(self.w),
            'x': pos_to_levels(self.x),
            'y': pos_to_levels(self.y),
            'z': pos_to_levels(self.z)
        }
    
    def to_flat(self) -> np.ndarray:
        """Flatten to 16D vector."""
        return np.concatenate([self.w, self.x, self.y, self.z])
    
    @classmethod
    def from_flat(cls, flat: np.ndarray) -> 'QuaternionPosition':
        """Create from 16D flat vector."""
        return cls(
            w=flat[0:4],
            x=flat[4:8],
            y=flat[8:12],
            z=flat[12:16]
        )
    
    def distance(self, other: 'QuaternionPosition', 
                 layer_weights: Tuple[float, float, float, float] = None) -> float:
        """
        Weighted distance between quaternion positions.
        
        layer_weights: (w_weight, x_weight, y_weight, z_weight)
        Default: w (semantic) most important, then x (grammatical), etc.
        """
        if layer_weights is None:
            layer_weights = (PHI**2, PHI, 1.0, PHI**-1)
        
        w_dist = np.linalg.norm(self.w - other.w)
        x_dist = np.linalg.norm(self.x - other.x)
        y_dist = np.linalg.norm(self.y - other.y)
        z_dist = np.linalg.norm(self.z - other.z)
        
        return float(np.sqrt(
            layer_weights[0] * w_dist**2 +
            layer_weights[1] * x_dist**2 +
            layer_weights[2] * y_dist**2 +
            layer_weights[3] * z_dist**2
        ))
    
    def similarity(self, other: 'QuaternionPosition',
                   layer_weights: Tuple[float, float, float, float] = None) -> float:
        """Similarity score (0, 1]."""
        dist = self.distance(other, layer_weights)
        return 1.0 / (1.0 + dist)
    
    def semantic_distance(self, other: 'QuaternionPosition') -> float:
        """Distance considering only semantic layer (w)."""
        return float(np.linalg.norm(self.w - other.w))
    
    def grammatical_distance(self, other: 'QuaternionPosition') -> float:
        """Distance considering only grammatical layer (x)."""
        return float(np.linalg.norm(self.x - other.x))
    
    def describe(self) -> Dict[str, Dict[str, str]]:
        """Get human-readable description of position."""
        levels = self.to_levels()
        
        description = {
            'semantic': {},
            'grammatical': {},
            'contextual': {},
            'reserved': {}
        }
        
        # Semantic
        for dim in SemanticDim:
            level = levels['w'][dim.value]
            meanings = SEMANTIC_MEANINGS.get(dim.value, {})
            description['semantic'][dim.name.lower()] = meanings.get(level, f'level_{level}')
        
        # Grammatical
        for dim in GrammaticalDim:
            level = levels['x'][dim.value]
            meanings = GRAMMATICAL_MEANINGS.get(dim.value, {})
            description['grammatical'][dim.name.lower()] = meanings.get(level, f'level_{level}')
        
        # Contextual
        for dim in ContextualDim:
            level = levels['y'][dim.value]
            meanings = CONTEXTUAL_MEANINGS.get(dim.value, {})
            description['contextual'][dim.name.lower()] = meanings.get(level, f'level_{level}')
        
        return description
    
    def __repr__(self):
        levels = self.to_levels()
        return f"Q(w={levels['w']}, x={levels['x']}, y={levels['y']}, z={levels['z']})"


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def tense_shift(pos: QuaternionPosition, delta: int) -> QuaternionPosition:
    """
    Shift tense by delta levels.
    
    delta > 0: toward future
    delta < 0: toward past
    """
    new_x = pos.x.copy()
    current_level = round(np.log(new_x[GrammaticalDim.TENSE]) / np.log(PHI))
    new_level = current_level + delta
    new_x[GrammaticalDim.TENSE] = PHI ** new_level
    
    return QuaternionPosition(w=pos.w.copy(), x=new_x, y=pos.y.copy(), z=pos.z.copy())


def voice_flip(pos: QuaternionPosition) -> QuaternionPosition:
    """Flip between active and passive voice."""
    new_x = pos.x.copy()
    current_level = round(np.log(new_x[GrammaticalDim.VOICE]) / np.log(PHI))
    # Flip: 1 (active) <-> 0 (passive)
    new_level = 1 - current_level if current_level in [0, 1] else current_level
    new_x[GrammaticalDim.VOICE] = PHI ** new_level
    
    return QuaternionPosition(w=pos.w.copy(), x=new_x, y=pos.y.copy(), z=pos.z.copy())


def formality_shift(pos: QuaternionPosition, delta: int) -> QuaternionPosition:
    """Shift formality level."""
    new_w = pos.w.copy()
    current_level = round(np.log(new_w[SemanticDim.FORMALITY]) / np.log(PHI))
    new_level = current_level + delta
    new_w[SemanticDim.FORMALITY] = PHI ** new_level
    
    return QuaternionPosition(w=new_w, x=pos.x.copy(), y=pos.y.copy(), z=pos.z.copy())


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_tense_navigation():
    """
    Demonstrate tense navigation with the classic example:
    "I went to the store" vs "I will go to the store"
    """
    print("=" * 60)
    print("TENSE NAVIGATION DEMO")
    print("=" * 60)
    print()
    
    # Base concept: "go to store" - same semantic content
    # Semantic: general domain, specific action, informative intent, casual
    base_semantic = [1, 2, 1, 0]  # [domain, specificity, intent, formality]
    
    # "I went to the store" - past tense, perfective aspect, indicative, active
    past = QuaternionPosition.from_levels(
        w_levels=base_semantic,
        x_levels=[0, 0, 0, 1],  # [tense=past, aspect=perfective, mood=indicative, voice=active]
        y_levels=[0, 2, 0, 0],  # [register=neutral, evidentiality=direct, politeness=neutral, emphasis=neutral]
    )
    
    # "I will go to the store" - future tense, prospective aspect, indicative, active
    future = QuaternionPosition.from_levels(
        w_levels=base_semantic,
        x_levels=[2, 2, 0, 1],  # [tense=future, aspect=prospective, mood=indicative, voice=active]
        y_levels=[0, 2, 0, 0],
    )
    
    # "I am going to the store" - present tense, progressive aspect
    present = QuaternionPosition.from_levels(
        w_levels=base_semantic,
        x_levels=[1, 1, 0, 1],  # [tense=present, aspect=progressive, mood=indicative, voice=active]
        y_levels=[0, 2, 0, 0],
    )
    
    print("Sentence: 'I went to the store'")
    print(f"  Position: {past}")
    print(f"  Semantic: {past.describe()['semantic']}")
    print(f"  Grammatical: {past.describe()['grammatical']}")
    print()
    
    print("Sentence: 'I will go to the store'")
    print(f"  Position: {future}")
    print(f"  Semantic: {future.describe()['semantic']}")
    print(f"  Grammatical: {future.describe()['grammatical']}")
    print()
    
    print("Sentence: 'I am going to the store'")
    print(f"  Position: {present}")
    print(f"  Semantic: {present.describe()['semantic']}")
    print(f"  Grammatical: {present.describe()['grammatical']}")
    print()
    
    # Distances
    print("DISTANCES:")
    print(f"  past ↔ future (full):      {past.distance(future):.3f}")
    print(f"  past ↔ future (semantic):  {past.semantic_distance(future):.3f}")
    print(f"  past ↔ future (grammatical): {past.grammatical_distance(future):.3f}")
    print()
    print(f"  past ↔ present (full):     {past.distance(present):.3f}")
    print(f"  present ↔ future (full):   {present.distance(future):.3f}")
    print()
    
    # Key insight: semantic distance is 0!
    print("KEY INSIGHT:")
    print(f"  Semantic distance between all three = {past.semantic_distance(future):.3f}")
    print("  → The MEANING is identical. Only the GRAMMAR differs.")
    print()


def demo_transformation():
    """
    Demonstrate predictable transformations.
    
    If we know the transformation, we can navigate AND generate.
    """
    print("=" * 60)
    print("TRANSFORMATION DEMO")
    print("=" * 60)
    print()
    
    # Start with "I went to the store"
    start = QuaternionPosition.from_levels(
        w_levels=[1, 2, 1, 0],
        x_levels=[0, 0, 0, 1],  # past, perfective, indicative, active
        y_levels=[0, 2, 0, 0],
    )
    
    print("Starting position: 'I went to the store'")
    print(f"  {start}")
    print()
    
    # Transform: shift tense to future
    future = tense_shift(start, delta=2)  # past (0) + 2 = future (2)
    print("After tense_shift(+2): 'I will go to the store'")
    print(f"  {future}")
    print(f"  Grammatical: {future.describe()['grammatical']}")
    print()
    
    # Transform: flip voice
    passive = voice_flip(start)
    print("After voice_flip(): 'The store was gone to by me' (passive)")
    print(f"  {passive}")
    print(f"  Grammatical: {passive.describe()['grammatical']}")
    print()
    
    # Transform: increase formality
    formal = formality_shift(start, delta=2)
    print("After formality_shift(+2): 'I proceeded to the store' (formal)")
    print(f"  {formal}")
    print(f"  Semantic: {formal.describe()['semantic']}")
    print()
    
    # Compose transformations
    formal_future = formality_shift(tense_shift(start, 2), 2)
    print("Composed: tense_shift(+2) then formality_shift(+2)")
    print("  'I shall proceed to the store' (formal + future)")
    print(f"  {formal_future}")
    print()


def demo_generation_potential():
    """
    Demonstrate how predictable structure enables generation.
    
    Given a position, we can describe what text SHOULD be there.
    """
    print("=" * 60)
    print("GENERATION POTENTIAL DEMO")
    print("=" * 60)
    print()
    
    # Define a target position
    target = QuaternionPosition.from_levels(
        w_levels=[2, 2, 2, 1],  # technology, specific, explain, professional
        x_levels=[1, 1, 0, 1],  # present, progressive, indicative, active
        y_levels=[2, 2, 1, 0],  # technical, direct, polite, neutral
    )
    
    print("Target position:")
    print(f"  {target}")
    print()
    print("Semantic description:")
    desc = target.describe()
    for layer, dims in desc.items():
        if layer != 'reserved':
            print(f"  {layer}:")
            for dim, meaning in dims.items():
                print(f"    {dim}: {meaning}")
    print()
    
    print("PREDICTED TEXT CHARACTERISTICS:")
    print("  - Domain: technology (programming, software)")
    print("  - Specificity: specific (not vague, not hyper-specific)")
    print("  - Intent: explanation (teaching, describing how)")
    print("  - Formality: professional (clear, polite)")
    print("  - Tense: present (happening now)")
    print("  - Aspect: progressive (ongoing action)")
    print("  - Voice: active (subject does action)")
    print("  - Register: technical (domain terminology)")
    print("  - Evidentiality: direct (firsthand knowledge)")
    print()
    
    print("EXAMPLE TEXT THAT FITS THIS POSITION:")
    print('  "The function is processing the input data and returning')
    print('   the transformed result to the caller."')
    print()
    
    # Show nearby positions
    print("NEARBY POSITIONS (small transformations):")
    
    past_version = tense_shift(target, -1)
    print(f"  tense_shift(-1): {past_version.describe()['grammatical']['tense']}")
    print('    → "The function was processing the input data..."')
    
    passive_version = voice_flip(target)
    print(f"  voice_flip(): {passive_version.describe()['grammatical']['voice']}")
    print('    → "The input data is being processed by the function..."')
    
    casual_version = formality_shift(target, -1)
    print(f"  formality_shift(-1): {casual_version.describe()['semantic']['formality']}")
    print('    → "The function is crunching the data and spitting out results..."')
    print()


def demo_similarity_search():
    """
    Demonstrate similarity search with layer-aware weighting.
    """
    print("=" * 60)
    print("SIMILARITY SEARCH DEMO")
    print("=" * 60)
    print()
    
    # Query: "What is Python?" - present tense question about technology
    query = QuaternionPosition.from_levels(
        w_levels=[2, 2, 1, 0],  # technology, specific, inform, casual
        x_levels=[1, 0, 0, 1],  # present, perfective, indicative, active
        y_levels=[0, 0, 0, 0],  # neutral across contextual
    )
    
    # Candidates
    candidates = [
        ("Python is a programming language", QuaternionPosition.from_levels(
            w_levels=[2, 2, 1, 0],  # Exact semantic match!
            x_levels=[1, 0, 0, 1],  # present, perfective, indicative, active
            y_levels=[0, 2, 0, 0],
        )),
        ("Python was created in 1991", QuaternionPosition.from_levels(
            w_levels=[2, 2, 1, 0],  # Same semantic
            x_levels=[0, 0, 0, 1],  # PAST tense
            y_levels=[0, 2, 0, 0],
        )),
        ("Physics is the study of matter", QuaternionPosition.from_levels(
            w_levels=[3, 2, 1, 0],  # DIFFERENT domain (hard_science)
            x_levels=[1, 0, 0, 1],  # Same grammatical
            y_levels=[0, 2, 0, 0],
        )),
        ("Hello, how are you?", QuaternionPosition.from_levels(
            w_levels=[-1, 0, -1, -1],  # social, vague, social intent, informal
            x_levels=[1, 0, 0, 1],
            y_levels=[-1, 0, 1, 0],
        )),
    ]
    
    print(f"Query: 'What is Python?'")
    print(f"  Position: {query}")
    print()
    
    print("Candidates (sorted by similarity):")
    results = []
    for text, pos in candidates:
        sim = query.similarity(pos)
        sem_dist = query.semantic_distance(pos)
        gram_dist = query.grammatical_distance(pos)
        results.append((text, sim, sem_dist, gram_dist))
    
    results.sort(key=lambda x: -x[1])  # Sort by similarity descending
    
    for text, sim, sem_dist, gram_dist in results:
        print(f"  [{sim:.3f}] '{text}'")
        print(f"         semantic_dist={sem_dist:.3f}, grammatical_dist={gram_dist:.3f}")
    print()
    
    print("KEY INSIGHT:")
    print("  'Python is a programming language' wins because it matches")
    print("  BOTH semantic (technology, specific, inform) AND grammatical (present).")
    print()
    print("  'Python was created in 1991' is close but loses on tense (past vs present).")
    print("  'Physics is the study of matter' loses on domain (hard_science vs technology).")
    print()


if __name__ == "__main__":
    demo_tense_navigation()
    print("\n")
    demo_transformation()
    print("\n")
    demo_generation_potential()
    print("\n")
    demo_similarity_search()
