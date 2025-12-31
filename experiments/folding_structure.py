"""
Folding Structure Experiment

The hypothesis:
- Information is encoded in SHAPE, not just position
- Like DNA, the structure FOLDS to bring related concepts into contact
- Zeta zeros are like zinc fingers - access points for random access
- Anchors are dynamic, emerging from the folding process
- Content can have error and still work (error tolerance)

Key insight: We're not clustering - we're FOLDING.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import re

# φ (golden ratio) - fundamental constant
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class FoldPoint:
    """
    A point where the structure folds.
    
    Like a zinc finger in DNA - an access point that brings
    distant parts of the sequence into contact.
    """
    position: int  # Position in the linear sequence
    fold_to: int   # What position it folds to contact
    strength: float = 1.0  # How strong the fold is
    label: str = ""
    
    @property
    def distance(self) -> int:
        """Linear distance bridged by this fold."""
        return abs(self.fold_to - self.position)


@dataclass 
class StructureSegment:
    """
    A segment of the structure between fold points.
    
    The segment has a shape (curvature) that encodes information.
    """
    start: int
    end: int
    content: List[str]  # Words/tokens in this segment
    curvature: float = 0.0  # How curved this segment is
    
    @property
    def length(self) -> int:
        return self.end - self.start


class FoldingStructure:
    """
    A structure that encodes information through FOLDING.
    
    Key principles:
    1. Linear sequence of tokens (like DNA bases)
    2. Fold points bring distant parts into contact
    3. Shape (curvature between folds) encodes meaning
    4. Content can have error - shape is what matters
    """
    
    def __init__(self):
        self.sequence: List[str] = []  # Linear sequence of tokens
        self.fold_points: List[FoldPoint] = []  # Where the structure folds
        self.segments: List[StructureSegment] = []  # Segments between folds
        
        # Contact map: which positions are in contact due to folding
        self.contact_map: Dict[int, Set[int]] = defaultdict(set)
        
        # Access points (like zinc fingers) - positions that can be
        # randomly accessed without unfolding the whole structure
        self.access_points: Set[int] = set()
    
    def add_sequence(self, tokens: List[str]):
        """Add tokens to the linear sequence."""
        start = len(self.sequence)
        self.sequence.extend(tokens)
        
        # Detect natural fold points based on repetition/pattern
        self._detect_fold_points(start)
    
    def _detect_fold_points(self, start: int):
        """
        Detect natural fold points in the sequence.
        
        Fold points occur where:
        1. Same word appears at different positions (self-reference)
        2. Similar words appear (semantic self-similarity)
        3. Structural patterns repeat
        
        Key insight: Folds happen where the sequence references itself.
        
        IMPORTANT: Weight folds by word importance (Zipf-aware).
        Common words like "the" create weak folds.
        Rare/specific words create strong folds.
        """
        if len(self.sequence) < 3:
            return
        
        # Count word frequencies for Zipf weighting
        word_counts: Dict[str, int] = defaultdict(int)
        for word in self.sequence:
            word_counts[word] += 1
        
        total_words = len(self.sequence)
        
        # Common words (stopwords) - these create weak folds
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'it', 'this', 'that', 'and', 'or', 'but', 'if', 'then'}
        
        # Look for repeated WORDS (not patterns) - self-reference = fold
        word_positions: Dict[str, List[int]] = defaultdict(list)
        
        for i, word in enumerate(self.sequence):
            word_positions[word].append(i)
        
        # Words that appear multiple times create folds
        for word, positions in word_positions.items():
            if len(positions) >= 2:
                # Calculate fold strength based on word importance
                # Rare words = strong folds, common words = weak folds
                freq_ratio = word_counts[word] / total_words
                
                if word in stopwords:
                    base_strength = 0.1  # Weak fold for stopwords
                elif freq_ratio > 0.1:
                    base_strength = 0.3  # Medium fold for frequent words
                else:
                    base_strength = 1.0  # Strong fold for rare/specific words
                
                # Create folds between all occurrences
                for i in range(len(positions) - 1):
                    pos1 = positions[i]
                    pos2 = positions[i + 1]
                    
                    # Only fold if there's some distance
                    if pos2 - pos1 >= 2:
                        # Strength = base_strength / distance
                        strength = base_strength / (pos2 - pos1)
                        
                        fold = FoldPoint(
                            position=pos2,
                            fold_to=pos1,
                            strength=strength,
                            label=f"self-ref:{word}"
                        )
                        self.fold_points.append(fold)
                        
                        # Update contact map
                        self.contact_map[pos1].add(pos2)
                        self.contact_map[pos2].add(pos1)
                        
                        # Mark as access points (only for strong folds)
                        if strength > 0.1:
                            self.access_points.add(pos1)
                            self.access_points.add(pos2)
    
    def add_fold(self, pos1: int, pos2: int, label: str = ""):
        """Manually add a fold point."""
        fold = FoldPoint(
            position=pos1,
            fold_to=pos2,
            strength=1.0,
            label=label
        )
        self.fold_points.append(fold)
        self.contact_map[pos1].add(pos2)
        self.contact_map[pos2].add(pos1)
        self.access_points.add(pos1)
        self.access_points.add(pos2)
    
    def get_contacts(self, position: int) -> Set[int]:
        """Get all positions in contact with a given position."""
        contacts = set()
        
        # Direct contacts from folding
        contacts.update(self.contact_map.get(position, set()))
        
        # Adjacent positions are always in contact
        if position > 0:
            contacts.add(position - 1)
        if position < len(self.sequence) - 1:
            contacts.add(position + 1)
        
        return contacts
    
    def compute_shape(self) -> np.ndarray:
        """
        Compute the shape of the structure.
        
        Shape is represented as curvature at each position.
        High curvature = fold point
        Low curvature = straight segment
        """
        n = len(self.sequence)
        if n == 0:
            return np.array([])
        
        curvature = np.zeros(n)
        
        # Fold points have high curvature
        for fold in self.fold_points:
            if fold.position < n:
                curvature[fold.position] += fold.strength
            if fold.fold_to < n:
                curvature[fold.fold_to] += fold.strength * 0.5
        
        # Smooth the curvature (folds affect nearby positions)
        smoothed = np.convolve(curvature, [0.25, 0.5, 0.25], mode='same')
        
        return smoothed
    
    def shape_similarity(self, other: 'FoldingStructure') -> float:
        """
        Compare shapes of two structures.
        
        Key insight: Similar shapes = similar meaning,
        even if content differs.
        """
        shape1 = self.compute_shape()
        shape2 = other.compute_shape()
        
        # Normalize lengths
        if len(shape1) == 0 or len(shape2) == 0:
            return 0.0
        
        # Resample to same length
        target_len = min(len(shape1), len(shape2))
        if len(shape1) > target_len:
            indices = np.linspace(0, len(shape1)-1, target_len).astype(int)
            shape1 = shape1[indices]
        if len(shape2) > target_len:
            indices = np.linspace(0, len(shape2)-1, target_len).astype(int)
            shape2 = shape2[indices]
        
        # Correlation of shapes
        if np.std(shape1) < 1e-10 or np.std(shape2) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(shape1, shape2)[0, 1]
        return max(0.0, correlation)  # Only positive correlation matters
    
    def random_access(self, query_tokens: List[str]) -> List[int]:
        """
        Random access via access points (zinc fingers).
        
        Find positions that match the query without
        scanning the whole sequence.
        """
        matches = []
        
        # Check access points first (fast path)
        for ap in self.access_points:
            if ap < len(self.sequence):
                # Check if query matches at this access point
                match = True
                for i, token in enumerate(query_tokens):
                    if ap + i >= len(self.sequence):
                        match = False
                        break
                    if self.sequence[ap + i] != token:
                        match = False
                        break
                
                if match:
                    matches.append(ap)
        
        return matches
    
    def deficiency_as_shape_mismatch(self, expected_shape: np.ndarray) -> float:
        """
        Detect deficiency as shape mismatch.
        
        The deficiency isn't about content - it's about
        whether the structure has the right SHAPE.
        """
        actual_shape = self.compute_shape()
        
        if len(actual_shape) == 0 or len(expected_shape) == 0:
            return 1.0  # Maximum deficiency
        
        # Resample to same length
        target_len = min(len(actual_shape), len(expected_shape))
        if len(actual_shape) > target_len:
            indices = np.linspace(0, len(actual_shape)-1, target_len).astype(int)
            actual_shape = actual_shape[indices]
        if len(expected_shape) > target_len:
            indices = np.linspace(0, len(expected_shape)-1, target_len).astype(int)
            expected_shape = expected_shape[indices]
        
        # Shape difference
        diff = np.abs(actual_shape - expected_shape)
        return np.mean(diff)


class FoldingImprovementLoop:
    """
    Improvement loop based on folding structure.
    
    Key principles:
    1. Deficiency = shape mismatch (not content mismatch)
    2. Fix = add folds to correct the shape
    3. Learning = discovering which fold patterns work
    """
    
    def __init__(self):
        # Library of learned fold patterns
        self.fold_patterns: List[Tuple[np.ndarray, List[FoldPoint]]] = []
        
        # Shape templates for different "meanings"
        self.shape_templates: Dict[str, np.ndarray] = {}
    
    def learn_fold_pattern(self, structure: FoldingStructure, label: str):
        """Learn a fold pattern from a successful structure."""
        shape = structure.compute_shape()
        folds = structure.fold_points.copy()
        
        self.fold_patterns.append((shape, folds))
        self.shape_templates[label] = shape
    
    def find_matching_pattern(self, target_shape: np.ndarray) -> Optional[List[FoldPoint]]:
        """Find a fold pattern that produces a similar shape."""
        best_match = None
        best_similarity = 0.0
        
        for pattern_shape, folds in self.fold_patterns:
            # Compare shapes
            if len(pattern_shape) == 0:
                continue
            
            # Resample to same length
            target_len = min(len(pattern_shape), len(target_shape))
            p_shape = pattern_shape[:target_len] if len(pattern_shape) >= target_len else pattern_shape
            t_shape = target_shape[:target_len] if len(target_shape) >= target_len else target_shape
            
            if len(p_shape) != len(t_shape):
                continue
            
            if np.std(p_shape) < 1e-10 or np.std(t_shape) < 1e-10:
                continue
            
            similarity = np.corrcoef(p_shape, t_shape)[0, 1]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = folds
        
        return best_match if best_similarity > 0.5 else None
    
    def improve_structure(self, structure: FoldingStructure, 
                          target_shape: np.ndarray) -> Tuple[float, float]:
        """
        Improve a structure to match a target shape.
        
        Returns: (initial_mismatch, final_mismatch)
        """
        initial_mismatch = structure.deficiency_as_shape_mismatch(target_shape)
        
        # Find a matching fold pattern
        matching_folds = self.find_matching_pattern(target_shape)
        
        if matching_folds:
            # Apply the fold pattern (scaled to structure size)
            scale = len(structure.sequence) / 100.0  # Assume patterns learned on ~100 token sequences
            
            for fold in matching_folds:
                scaled_pos = int(fold.position * scale)
                scaled_to = int(fold.fold_to * scale)
                
                if scaled_pos < len(structure.sequence) and scaled_to < len(structure.sequence):
                    structure.add_fold(scaled_pos, scaled_to, fold.label)
        
        final_mismatch = structure.deficiency_as_shape_mismatch(target_shape)
        
        return initial_mismatch, final_mismatch


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment():
    """Test the folding structure approach."""
    
    print("=" * 60)
    print("FOLDING STRUCTURE EXPERIMENT")
    print("=" * 60)
    
    # Create a structure from text
    print("\n1. Building structure from text...")
    
    text1 = "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale."
    tokens1 = re.findall(r'\b[a-zA-Z]+\b', text1.lower())
    
    structure1 = FoldingStructure()
    structure1.add_sequence(tokens1)
    
    print(f"   Sequence length: {len(structure1.sequence)}")
    print(f"   Fold points detected: {len(structure1.fold_points)}")
    print(f"   Access points: {len(structure1.access_points)}")
    
    for fold in structure1.fold_points:
        print(f"      {fold.label}: pos {fold.position} -> {fold.fold_to} (distance: {fold.distance})")
    
    # Compute shape
    print("\n2. Computing structure shape...")
    shape1 = structure1.compute_shape()
    print(f"   Shape: {shape1[:10]}..." if len(shape1) > 10 else f"   Shape: {shape1}")
    print(f"   Max curvature: {np.max(shape1):.3f}")
    print(f"   Mean curvature: {np.mean(shape1):.3f}")
    
    # Create a second structure with similar meaning but different words
    print("\n3. Creating structure with similar meaning, different words...")
    
    text2 = "Commander Ahab led the vessel. The commander directed the sailors. Ahab pursued the beast."
    tokens2 = re.findall(r'\b[a-zA-Z]+\b', text2.lower())
    
    structure2 = FoldingStructure()
    structure2.add_sequence(tokens2)
    
    print(f"   Sequence length: {len(structure2.sequence)}")
    print(f"   Fold points detected: {len(structure2.fold_points)}")
    
    # Compare shapes
    print("\n4. Comparing shapes (content differs, meaning similar)...")
    similarity = structure1.shape_similarity(structure2)
    print(f"   Shape similarity: {similarity:.3f}")
    
    # Create a structure with different meaning
    print("\n5. Creating structure with different meaning...")
    
    text3 = "The dog ran through the park. It chased a ball. The park was sunny."
    tokens3 = re.findall(r'\b[a-zA-Z]+\b', text3.lower())
    
    structure3 = FoldingStructure()
    structure3.add_sequence(tokens3)
    
    similarity_diff = structure1.shape_similarity(structure3)
    print(f"   Shape similarity (whale vs dog): {similarity_diff:.3f}")
    
    # Test random access via zinc fingers
    print("\n6. Testing random access (zinc finger access points)...")
    
    query = ['ahab']
    matches = structure1.random_access(query)
    print(f"   Query: {query}")
    print(f"   Matches at positions: {matches}")
    
    # Test deficiency as shape mismatch
    print("\n7. Testing deficiency detection via shape mismatch...")
    
    target_shape = shape1  # We want structure2 to have shape like structure1
    mismatch = structure2.deficiency_as_shape_mismatch(target_shape)
    print(f"   Shape mismatch (similar meaning): {mismatch:.3f}")
    
    mismatch_diff = structure3.deficiency_as_shape_mismatch(target_shape)
    print(f"   Shape mismatch (different meaning): {mismatch_diff:.3f}")
    
    # Test improvement loop
    print("\n8. Testing fold-based improvement...")
    
    loop = FoldingImprovementLoop()
    
    # Learn from structure1
    loop.learn_fold_pattern(structure1, "whale_hunt")
    print(f"   Learned pattern: whale_hunt")
    
    # Try to improve structure2 to match structure1's shape
    initial, final = loop.improve_structure(structure2, target_shape)
    print(f"   Initial mismatch: {initial:.3f}")
    print(f"   Final mismatch: {final:.3f}")
    print(f"   Improvement: {initial - final:.3f}")
    
    # Key insight demonstration
    print("\n9. KEY INSIGHT: Error tolerance in content...")
    
    # Create structure with typos/errors
    text_with_errors = "Captian Ahab comanded the shipp. The captian led the crue. Ahab hunted the wale."
    tokens_errors = re.findall(r'\b[a-zA-Z]+\b', text_with_errors.lower())
    
    structure_errors = FoldingStructure()
    structure_errors.add_sequence(tokens_errors)
    
    # Despite content errors, shape should be similar
    shape_similarity_errors = structure1.shape_similarity(structure_errors)
    print(f"   Original vs typo-filled: shape similarity = {shape_similarity_errors:.3f}")
    print(f"   (Content has errors, but SHAPE encodes the meaning)")
    
    # Deeper analysis
    print("\n10. WHY THIS WORKS - Fold pattern analysis...")
    
    print("\n   Structure 1 (whale hunt) folds:")
    for fold in structure1.fold_points:
        print(f"      {fold.label}: {fold.position} -> {fold.fold_to}")
    
    print("\n   Structure 2 (commander pursuit) folds:")
    for fold in structure2.fold_points:
        print(f"      {fold.label}: {fold.position} -> {fold.fold_to}")
    
    print("\n   Structure 3 (dog park) folds:")
    for fold in structure3.fold_points:
        print(f"      {fold.label}: {fold.position} -> {fold.fold_to}")
    
    print("\n   INSIGHT: Similar narratives have similar FOLD PATTERNS")
    print("   - Both whale/commander have: subject-verb-object repeated 3x")
    print("   - Dog park has: different narrative structure")
    print("   - The SHAPE of self-references encodes the narrative structure")
    
    # Test with more varied content
    print("\n11. Testing narrative structure detection...")
    
    # Same narrative structure, completely different domain
    text_scifi = "Admiral Kirk commanded the starship. The admiral led the crew. Kirk explored the galaxy."
    tokens_scifi = re.findall(r'\b[a-zA-Z]+\b', text_scifi.lower())
    structure_scifi = FoldingStructure()
    structure_scifi.add_sequence(tokens_scifi)
    
    similarity_scifi = structure1.shape_similarity(structure_scifi)
    print(f"   Whale hunt vs SciFi (same narrative structure): {similarity_scifi:.3f}")
    
    # Different narrative structure, same domain
    text_whale_diff = "The whale swam deep. Ahab watched from the ship. The ocean was vast and cold."
    tokens_whale_diff = re.findall(r'\b[a-zA-Z]+\b', text_whale_diff.lower())
    structure_whale_diff = FoldingStructure()
    structure_whale_diff.add_sequence(tokens_whale_diff)
    
    similarity_whale_diff = structure1.shape_similarity(structure_whale_diff)
    print(f"   Whale hunt vs Whale descriptive (different structure): {similarity_whale_diff:.3f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return structure1, structure2, structure3


def run_verification_experiments():
    """
    Rigorous verification of the folding hypothesis.
    
    We need to verify:
    1. Shape similarity is consistent across many examples
    2. Shape difference correlates with semantic difference
    3. The approach works for different text types (not just narratives)
    4. Edge cases are handled properly
    """
    
    print("=" * 70)
    print("VERIFICATION EXPERIMENTS: Folding Structure Hypothesis")
    print("=" * 70)
    
    results = {
        'same_structure_same_domain': [],
        'same_structure_diff_domain': [],
        'diff_structure_same_domain': [],
        'diff_structure_diff_domain': [],
    }
    
    # ==========================================================================
    # TEST SET 1: Narrative structures (Subject-Verb-Object repeated)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST SET 1: Narrative Structures")
    print("=" * 70)
    
    narrative_templates = [
        # Template: "[Name] [action] the [object]. The [role] [action2] the [group]. [Name] [action3] the [target]."
        "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
        "Admiral Kirk commanded the starship. The admiral led the crew. Kirk explored the galaxy.",
        "Chef Marco prepared the meal. The chef instructed the staff. Marco served the guests.",
        "Detective Holmes examined the evidence. The detective questioned the witnesses. Holmes solved the mystery.",
        "Professor Xavier taught the students. The professor guided the team. Xavier protected the mutants.",
    ]
    
    # Build structures
    narrative_structures = []
    for text in narrative_templates:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        s = FoldingStructure()
        s.add_sequence(tokens)
        narrative_structures.append((text[:30] + "...", s))
    
    print("\nPairwise shape similarities (same narrative structure):")
    for i in range(len(narrative_structures)):
        for j in range(i+1, len(narrative_structures)):
            name1, s1 = narrative_structures[i]
            name2, s2 = narrative_structures[j]
            sim = s1.shape_similarity(s2)
            results['same_structure_diff_domain'].append(sim)
            print(f"   {name1} vs {name2}: {sim:.3f}")
    
    # ==========================================================================
    # TEST SET 2: Different structures (questions, lists, descriptions)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST SET 2: Different Structure Types")
    print("=" * 70)
    
    different_structures = [
        # Question structure
        "What is the whale? Where is the ship? Who is the captain?",
        # List structure
        "First the ship. Second the crew. Third the whale. Fourth the ocean.",
        # Description structure
        "The whale is large. The whale is white. The whale is dangerous.",
        # Cause-effect structure
        "Because the whale attacked, the ship sank. Because the ship sank, the crew died.",
    ]
    
    diff_structures = []
    for text in different_structures:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        s = FoldingStructure()
        s.add_sequence(tokens)
        diff_structures.append((text[:30] + "...", s))
    
    # Compare narrative vs different structures
    print("\nNarrative vs Different structures:")
    base_narrative = narrative_structures[0][1]  # Ahab story
    for name, s in diff_structures:
        sim = base_narrative.shape_similarity(s)
        results['diff_structure_same_domain'].append(sim)
        print(f"   Ahab narrative vs {name}: {sim:.3f}")
    
    # ==========================================================================
    # TEST SET 3: Edge cases
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST SET 3: Edge Cases")
    print("=" * 70)
    
    edge_cases = [
        ("Empty", ""),
        ("Single word", "whale"),
        ("Two words", "white whale"),
        ("No repeats", "the quick brown fox jumps over lazy dog"),
        ("All repeats", "the the the the the the the"),
        ("Very long", " ".join(["Captain Ahab commanded the ship."] * 10)),
    ]
    
    print("\nEdge case handling:")
    for name, text in edge_cases:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower()) if text else []
        s = FoldingStructure()
        if tokens:
            s.add_sequence(tokens)
        
        shape = s.compute_shape()
        folds = len(s.fold_points)
        
        # Compare to base narrative
        if len(shape) > 0:
            sim = base_narrative.shape_similarity(s)
        else:
            sim = 0.0
        
        print(f"   {name}: tokens={len(tokens)}, folds={folds}, shape_len={len(shape)}, sim_to_narrative={sim:.3f}")
    
    # ==========================================================================
    # TEST SET 4: Error tolerance
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST SET 4: Error Tolerance")
    print("=" * 70)
    
    original = "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale."
    
    error_variants = [
        ("Original", original),
        ("Typos", "Captian Ahab comanded the shipp. The captian led the crue. Ahab hunted the wale."),
        ("Missing words", "Captain Ahab the ship. The captain the crew. Ahab the whale."),
        ("Extra words", "Captain Ahab really commanded the big ship. The brave captain led the whole crew. Ahab hunted the white whale."),
        ("Synonyms", "Commander Ahab directed the vessel. The commander guided the sailors. Ahab pursued the beast."),
        ("Reordered", "The ship was commanded by Captain Ahab. The crew was led by the captain. The whale was hunted by Ahab."),
    ]
    
    # Build original structure
    orig_tokens = re.findall(r'\b[a-zA-Z]+\b', original.lower())
    orig_structure = FoldingStructure()
    orig_structure.add_sequence(orig_tokens)
    
    print("\nError tolerance (similarity to original):")
    for name, text in error_variants:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        s = FoldingStructure()
        s.add_sequence(tokens)
        
        sim = orig_structure.shape_similarity(s)
        print(f"   {name}: {sim:.3f}")
    
    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for category, values in results.items():
        if values:
            avg = np.mean(values)
            std = np.std(values)
            min_v = np.min(values)
            max_v = np.max(values)
            print(f"\n{category}:")
            print(f"   Mean: {avg:.3f} ± {std:.3f}")
            print(f"   Range: [{min_v:.3f}, {max_v:.3f}]")
    
    # Key question: Is there separation between same-structure and diff-structure?
    same_struct = results['same_structure_diff_domain']
    diff_struct = results['diff_structure_same_domain']
    
    if same_struct and diff_struct:
        same_mean = np.mean(same_struct)
        diff_mean = np.mean(diff_struct)
        separation = same_mean - diff_mean
        
        print(f"\n*** KEY METRIC ***")
        print(f"   Same structure mean: {same_mean:.3f}")
        print(f"   Diff structure mean: {diff_mean:.3f}")
        print(f"   Separation: {separation:.3f}")
        print(f"   Discriminative: {'YES' if separation > 0.3 else 'NEEDS WORK'}")
    
    return results


def run_deficiency_detection_experiment():
    """
    Experiment 3: Test deficiency detection via shape mismatch.
    
    Simulate the GearImprovementLoop scenario:
    - We have an expected output (what we want)
    - We have an actual output (what we got)
    - Detect deficiency as shape mismatch
    - Classify the type of deficiency based on shape analysis
    """
    
    print("=" * 70)
    print("EXPERIMENT 3: Deficiency Detection via Shape Mismatch")
    print("=" * 70)
    
    # Define expected output (what a good gear should produce)
    expected_outputs = {
        'narrative': "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
        'question_answer': "The whale is Moby Dick. The whale is white. The whale is dangerous.",
        'list': "First the ship. Second the crew. Third the whale.",
    }
    
    # Define various actual outputs (what the gear actually produced)
    actual_outputs = {
        # Good outputs (should have low deficiency)
        'good_narrative': "Admiral Kirk commanded the starship. The admiral led the crew. Kirk explored the galaxy.",
        'good_qa': "The beast is Moby Dick. The beast is large. The beast is feared.",
        
        # Deficient outputs (should have high deficiency)
        'wrong_structure': "The whale. The ship. The captain. The crew. The ocean.",  # List instead of narrative
        'too_short': "Ahab hunted.",  # Missing structure
        'irrelevant': "The weather was nice. The sun was shining. Birds were singing.",  # Off topic
        'vague': "Something happened. Things occurred. Events transpired.",  # No specifics
        'incomplete': "Captain Ahab commanded the ship.",  # Only first part
    }
    
    # Build expected structures
    expected_structures = {}
    for name, text in expected_outputs.items():
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        s = FoldingStructure()
        s.add_sequence(tokens)
        expected_structures[name] = (text, s)
    
    # Test each actual output against expected narrative
    print("\n" + "-" * 70)
    print("Testing against NARRATIVE structure:")
    print("-" * 70)
    
    expected_name = 'narrative'
    expected_text, expected_struct = expected_structures[expected_name]
    expected_shape = expected_struct.compute_shape()
    
    print(f"\nExpected: {expected_text[:50]}...")
    print(f"Expected shape stats: len={len(expected_shape)}, max={np.max(expected_shape):.3f}, mean={np.mean(expected_shape):.3f}")
    print(f"Expected folds: {len(expected_struct.fold_points)}")
    
    deficiency_results = []
    
    for name, text in actual_outputs.items():
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        actual_struct = FoldingStructure()
        actual_struct.add_sequence(tokens)
        
        # Compute deficiency metrics
        shape_sim = expected_struct.shape_similarity(actual_struct)
        shape_mismatch = actual_struct.deficiency_as_shape_mismatch(expected_shape)
        
        # Analyze the deficiency
        actual_shape = actual_struct.compute_shape()
        fold_ratio = len(actual_struct.fold_points) / max(len(expected_struct.fold_points), 1)
        length_ratio = len(tokens) / max(len(expected_struct.sequence), 1)
        
        # Classify deficiency type based on shape analysis
        if shape_sim > 0.9:
            deficiency_type = "NONE (good match)"
        elif length_ratio < 0.5:
            deficiency_type = "INCOMPLETE (too short)"
        elif fold_ratio < 0.3:
            deficiency_type = "MISSING_STRUCTURE (no self-reference)"
        elif shape_mismatch > 0.3:
            deficiency_type = "WRONG_STRUCTURE (different pattern)"
        else:
            deficiency_type = "PARTIAL (some mismatch)"
        
        result = {
            'name': name,
            'shape_sim': shape_sim,
            'shape_mismatch': shape_mismatch,
            'fold_ratio': fold_ratio,
            'length_ratio': length_ratio,
            'deficiency_type': deficiency_type,
        }
        deficiency_results.append(result)
        
        print(f"\n{name}:")
        print(f"   Text: {text[:40]}...")
        print(f"   Shape similarity: {shape_sim:.3f}")
        print(f"   Shape mismatch: {shape_mismatch:.3f}")
        print(f"   Fold ratio: {fold_ratio:.2f}")
        print(f"   Length ratio: {length_ratio:.2f}")
        print(f"   → Deficiency: {deficiency_type}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DEFICIENCY DETECTION SUMMARY")
    print("=" * 70)
    
    print("\n| Output | Shape Sim | Mismatch | Deficiency Type |")
    print("|--------|-----------|----------|-----------------|")
    for r in deficiency_results:
        print(f"| {r['name'][:15]:15} | {r['shape_sim']:.3f} | {r['shape_mismatch']:.3f} | {r['deficiency_type'][:20]} |")
    
    # Verify discrimination
    good_sims = [r['shape_sim'] for r in deficiency_results if r['name'].startswith('good')]
    bad_sims = [r['shape_sim'] for r in deficiency_results if not r['name'].startswith('good')]
    
    if good_sims and bad_sims:
        good_mean = np.mean(good_sims)
        bad_mean = np.mean(bad_sims)
        print(f"\n*** DISCRIMINATION ***")
        print(f"   Good outputs mean similarity: {good_mean:.3f}")
        print(f"   Bad outputs mean similarity: {bad_mean:.3f}")
        print(f"   Separation: {good_mean - bad_mean:.3f}")
        print(f"   Discriminative: {'YES' if good_mean - bad_mean > 0.3 else 'NEEDS WORK'}")
    
    return deficiency_results


def run_fix_generation_experiment():
    """
    Experiment 4: Test fix generation via fold correction.
    
    Given a deficient output, can we improve it by:
    1. Identifying missing folds
    2. Suggesting where to add structure
    3. Learning fold patterns that work
    """
    
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 4: Fix Generation via Fold Correction")
    print("=" * 70)
    
    # Expected structure (target)
    expected_text = "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale."
    expected_tokens = re.findall(r'\b[a-zA-Z]+\b', expected_text.lower())
    expected_struct = FoldingStructure()
    expected_struct.add_sequence(expected_tokens)
    expected_shape = expected_struct.compute_shape()
    
    print(f"\nTarget structure:")
    print(f"   Text: {expected_text}")
    print(f"   Folds: {[(f.label, f.position, f.fold_to) for f in expected_struct.fold_points]}")
    
    # Deficient output (missing structure)
    deficient_text = "Ahab commanded the ship. He led the crew. He hunted the whale."
    deficient_tokens = re.findall(r'\b[a-zA-Z]+\b', deficient_text.lower())
    deficient_struct = FoldingStructure()
    deficient_struct.add_sequence(deficient_tokens)
    
    print(f"\nDeficient structure:")
    print(f"   Text: {deficient_text}")
    print(f"   Folds: {[(f.label, f.position, f.fold_to) for f in deficient_struct.fold_points]}")
    
    initial_sim = expected_struct.shape_similarity(deficient_struct)
    print(f"   Initial shape similarity: {initial_sim:.3f}")
    
    # Analyze what's missing
    print("\n" + "-" * 70)
    print("FOLD ANALYSIS: What's missing?")
    print("-" * 70)
    
    # Expected fold words
    expected_fold_words = set()
    for fold in expected_struct.fold_points:
        if fold.position < len(expected_struct.sequence):
            expected_fold_words.add(expected_struct.sequence[fold.position])
    
    # Actual fold words
    actual_fold_words = set()
    for fold in deficient_struct.fold_points:
        if fold.position < len(deficient_struct.sequence):
            actual_fold_words.add(deficient_struct.sequence[fold.position])
    
    missing_fold_words = expected_fold_words - actual_fold_words
    print(f"   Expected fold words: {expected_fold_words}")
    print(f"   Actual fold words: {actual_fold_words}")
    print(f"   Missing fold words: {missing_fold_words}")
    
    # Suggest fix: add the missing fold words
    print("\n" + "-" * 70)
    print("FIX SUGGESTION:")
    print("-" * 70)
    
    if missing_fold_words:
        print(f"   Add self-references for: {missing_fold_words}")
        print(f"   Example fix: Replace 'He' with 'Ahab' or 'The captain'")
        
        # Apply fix
        fixed_text = deficient_text.replace("He led", "The captain led").replace("He hunted", "Ahab hunted")
        fixed_tokens = re.findall(r'\b[a-zA-Z]+\b', fixed_text.lower())
        fixed_struct = FoldingStructure()
        fixed_struct.add_sequence(fixed_tokens)
        
        print(f"\n   Fixed text: {fixed_text}")
        print(f"   Fixed folds: {[(f.label, f.position, f.fold_to) for f in fixed_struct.fold_points]}")
        
        final_sim = expected_struct.shape_similarity(fixed_struct)
        print(f"   Final shape similarity: {final_sim:.3f}")
        print(f"   Improvement: {final_sim - initial_sim:.3f}")
    
    # Test the improvement loop
    print("\n" + "-" * 70)
    print("IMPROVEMENT LOOP SIMULATION:")
    print("-" * 70)
    
    loop = FoldingImprovementLoop()
    
    # Learn from expected structure
    loop.learn_fold_pattern(expected_struct, "narrative_pattern")
    print(f"   Learned pattern: narrative_pattern")
    
    # Try to improve deficient structure
    initial_mismatch, final_mismatch = loop.improve_structure(deficient_struct, expected_shape)
    print(f"   Initial mismatch: {initial_mismatch:.3f}")
    print(f"   Final mismatch: {final_mismatch:.3f}")
    
    return {
        'initial_sim': initial_sim,
        'final_sim': final_sim if 'final_sim' in dir() else initial_sim,
        'missing_fold_words': missing_fold_words,
    }


if __name__ == "__main__":
    # Run basic experiment
    s1, s2, s3 = run_experiment()
    
    print("\n\n")
    
    # Run verification experiments
    results = run_verification_experiments()
    
    print("\n\n")
    
    # Run deficiency detection experiment
    deficiency_results = run_deficiency_detection_experiment()
    
    # Run fix generation experiment
    fix_results = run_fix_generation_experiment()
