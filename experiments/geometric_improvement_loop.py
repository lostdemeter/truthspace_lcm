"""
Geometric Improvement Loop Experiment

The hypothesis:
1. Deficiency = Distance in φ-space between expected and actual
2. Fix = Transform that reduces this distance
3. Learning = Discovering which transforms work for which regions

Instead of pattern matching and hardcoded categories, we let the
structure emerge from the geometry of the problem.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import re

# φ (golden ratio) - the fundamental constant
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class GeometricPosition:
    """A position in φ-space."""
    coords: np.ndarray  # The position vector
    confidence: float = 1.0
    
    @property
    def magnitude(self) -> float:
        return np.linalg.norm(self.coords)
    
    def distance_to(self, other: 'GeometricPosition') -> float:
        """Euclidean distance to another position."""
        return np.linalg.norm(self.coords - other.coords)
    
    def angle_to(self, other: 'GeometricPosition') -> float:
        """Cosine similarity (angle) to another position."""
        dot = np.dot(self.coords, other.coords)
        norms = self.magnitude * other.magnitude
        if norms < 1e-10:
            return 0.0
        return dot / norms


class GeometricEncoder:
    """
    Encodes text into φ-space positions.
    
    The key insight: words that appear in similar contexts
    should map to similar positions. This is emergent - we
    don't predefine categories, they emerge from usage.
    """
    
    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        
        # Word positions - learned from usage
        self.word_positions: Dict[str, np.ndarray] = {}
        
        # Co-occurrence tracking for emergent positioning
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[str, int] = defaultdict(int)
        
        # Position update history (for convergence tracking)
        self.position_history: Dict[str, List[np.ndarray]] = defaultdict(list)
    
    def _get_or_create_position(self, word: str) -> np.ndarray:
        """Get existing position or create initial random position."""
        word = word.lower()
        if word not in self.word_positions:
            # Initial position based on word hash for reproducibility
            seed = hash(word) % (2**31)
            rng = np.random.RandomState(seed)
            self.word_positions[word] = rng.randn(self.dimensions) * 0.1
        return self.word_positions[word]
    
    def learn_from_text(self, text: str, window_size: int = 5):
        """
        Learn word positions from text via co-occurrence.
        
        Words that appear together attract each other in φ-space.
        This is the emergent learning - no predefined categories.
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Track co-occurrences
        for i, word in enumerate(words):
            self.word_counts[word] += 1
            
            # Context window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    self.cooccurrence[word][context_word] += 1
        
        # Update positions based on co-occurrence (attractor/repeller dynamics)
        # Multiple iterations to let positions converge
        for _ in range(3):
            self._update_positions()
    
    def _update_positions(self, learning_rate: float = 0.1):
        """
        Update word positions based on co-occurrence.
        
        Words that co-occur ATTRACT each other.
        Words that DON'T co-occur REPEL each other.
        This is the geometric learning step with attractor/repeller dynamics.
        """
        all_words = list(self.word_counts.keys())
        
        for word in all_words:
            if self.word_counts[word] < 2:
                continue
            
            word_pos = self._get_or_create_position(word)
            contexts = self.cooccurrence.get(word, {})
            
            # Calculate ATTRACTION from context words (co-occur)
            attraction = np.zeros(self.dimensions)
            total_attraction_weight = 0
            
            for context_word, count in contexts.items():
                context_pos = self._get_or_create_position(context_word)
                
                # Weight by co-occurrence frequency (normalized)
                weight = count / self.word_counts[word]
                attraction += weight * (context_pos - word_pos)
                total_attraction_weight += weight
            
            # Calculate REPULSION from non-context words (don't co-occur)
            repulsion = np.zeros(self.dimensions)
            repulsion_count = 0
            
            for other_word in all_words:
                if other_word == word:
                    continue
                if other_word in contexts:
                    continue  # Skip words we co-occur with
                
                other_pos = self._get_or_create_position(other_word)
                diff = word_pos - other_pos
                dist = np.linalg.norm(diff)
                
                if dist < 0.5:  # Only repel if too close
                    # Repulsion force inversely proportional to distance
                    repulsion += 0.1 * diff / (dist + 1e-10)
                    repulsion_count += 1
            
            # Combine forces
            delta = np.zeros(self.dimensions)
            
            if total_attraction_weight > 0:
                delta += learning_rate * attraction / total_attraction_weight
            
            if repulsion_count > 0:
                delta += 0.05 * repulsion / repulsion_count  # Weaker repulsion
            
            if np.linalg.norm(delta) > 0:
                new_pos = word_pos + delta
                self.word_positions[word] = new_pos
                self.position_history[word].append(new_pos.copy())
    
    def encode(self, text: str) -> GeometricPosition:
        """
        Encode text as a position in φ-space.
        
        The position is the centroid of word positions,
        weighted by inverse frequency (Zipf-aware).
        
        Key insight: Rare words carry more meaning, common words are scaffolding.
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        if not words:
            return GeometricPosition(np.zeros(self.dimensions), confidence=0.0)
        
        # Filter out very common words (stopwords emerge from frequency)
        total_words = sum(self.word_counts.values()) or 1
        
        # Aggregate word positions with Zipf-aware weighting
        positions = []
        weights = []
        
        for word in words:
            freq = self.word_counts.get(word, 0)
            relative_freq = freq / total_words
            
            # Skip very high frequency words (emergent stopwords)
            if relative_freq > 0.05:
                continue
            
            if word in self.word_positions:
                positions.append(self.word_positions[word])
                # Weight by inverse frequency (rare words matter more)
                # Use φ-scaling for natural hierarchy
                weight = PHI ** (-np.log1p(freq))
                weights.append(weight)
            else:
                # Unknown word - use hash-based position with high weight
                # Unknown = rare = important
                positions.append(self._get_or_create_position(word))
                weights.append(PHI)  # High weight for unknown
        
        if not positions:
            # All words were stopwords, use all words with equal weight
            for word in words:
                positions.append(self._get_or_create_position(word))
                weights.append(1.0)
        
        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        centroid = np.average(positions, axis=0, weights=weights)
        confidence = min(1.0, len(positions) / 5.0)
        
        return GeometricPosition(centroid, confidence)


@dataclass
class GeometricDeficiency:
    """
    A deficiency detected geometrically.
    
    Instead of categories, we have:
    - distance: how far actual is from expected
    - direction: which way we need to move
    - region: emergent cluster this belongs to
    """
    distance: float
    direction: np.ndarray  # Unit vector from actual to expected
    expected_pos: GeometricPosition
    actual_pos: GeometricPosition
    region_id: Optional[int] = None  # Emergent cluster
    
    @property
    def severity(self) -> float:
        """Severity is just normalized distance."""
        return min(1.0, self.distance / 2.0)


class GeometricTransform:
    """
    A transform in φ-space.
    
    Transforms are learned from successful fixes.
    A transform that moved output closer to expected
    becomes a candidate for similar deficiencies.
    """
    
    def __init__(self, name: str, delta: np.ndarray):
        self.name = name
        self.delta = delta  # The translation vector
        self.successes: int = 0
        self.failures: int = 0
        self.total_improvement: float = 0.0
    
    @property
    def effectiveness(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5  # Unknown
        return self.successes / total
    
    def apply(self, pos: GeometricPosition) -> GeometricPosition:
        """Apply this transform to a position."""
        new_coords = pos.coords + self.delta
        return GeometricPosition(new_coords, pos.confidence)
    
    def record_result(self, improved: bool, delta: float = 0.0):
        """Record whether this transform helped."""
        if improved:
            self.successes += 1
            self.total_improvement += delta
        else:
            self.failures += 1


class GeometricImprovementLoop:
    """
    The geometric improvement loop.
    
    Core principles:
    1. Deficiency = distance in φ-space (tells us WHERE structure is missing)
    2. Fix = adding structure (not just transforming)
    3. Learning = the structure itself grows to fill gaps
    
    Key insight: Error doesn't measure accuracy - it tells us WHERE to build.
    Like zeta zeros marking where structure naturally exists.
    """
    
    def __init__(self, dimensions: int = 8):
        self.encoder = GeometricEncoder(dimensions)
        self.dimensions = dimensions
        
        # Learned transforms (emergent from successful fixes)
        self.transforms: List[GeometricTransform] = []
        
        # Region → transform mapping (emergent clustering)
        self.region_transforms: Dict[int, List[GeometricTransform]] = defaultdict(list)
        
        # Deficiency history for clustering
        self.deficiency_history: List[GeometricDeficiency] = []
        
        # Quality threshold
        self.distance_threshold = 0.5  # Below this = acceptable
        
        # === NEW: Structure nodes ===
        # These are "anchor points" in φ-space that we add when we find gaps
        # Like zeta zeros - fixed points where structure exists
        self.structure_nodes: List[Tuple[np.ndarray, str]] = []  # (position, label)
    
    def learn_corpus(self, texts: List[str]):
        """Learn word positions from a corpus."""
        for text in texts:
            self.encoder.learn_from_text(text)
    
    def add_structure_node(self, position: np.ndarray, label: str):
        """
        Add a structure node at a position.
        
        Structure nodes mark where meaningful structure exists.
        They act as REPELLERS for distant concepts - pushing apart
        things that shouldn't be near this node.
        
        Key insight: The node defines a boundary, not an attractor.
        """
        self.structure_nodes.append((position.copy(), label))
        
        # The node REPELS distant words (creates separation)
        # Words close to the node stay, words far get pushed further
        for word, word_pos in self.encoder.word_positions.items():
            dist = np.linalg.norm(word_pos - position)
            
            if dist > 0.2:  # Distant words get pushed away
                direction = word_pos - position  # Away from node
                push = 0.05 / (dist + 0.1)  # Weaker push for farther words
                self.encoder.word_positions[word] = word_pos + push * direction / (np.linalg.norm(direction) + 1e-10)
    
    def find_nearest_node(self, position: np.ndarray) -> Optional[Tuple[np.ndarray, str, float]]:
        """Find the nearest structure node to a position."""
        if not self.structure_nodes:
            return None
        
        best_node = None
        best_dist = float('inf')
        best_label = None
        
        for node_pos, label in self.structure_nodes:
            dist = np.linalg.norm(position - node_pos)
            if dist < best_dist:
                best_dist = dist
                best_node = node_pos
                best_label = label
        
        return (best_node, best_label, best_dist) if best_node is not None else None
    
    def detect_deficiency(self, expected: str, actual: str) -> GeometricDeficiency:
        """
        Detect deficiency as geometric distance.
        
        No categories - just distance and direction.
        """
        expected_pos = self.encoder.encode(expected)
        actual_pos = self.encoder.encode(actual)
        
        distance = expected_pos.distance_to(actual_pos)
        
        # Direction from actual to expected
        diff = expected_pos.coords - actual_pos.coords
        norm = np.linalg.norm(diff)
        direction = diff / norm if norm > 1e-10 else np.zeros(self.dimensions)
        
        deficiency = GeometricDeficiency(
            distance=distance,
            direction=direction,
            expected_pos=expected_pos,
            actual_pos=actual_pos
        )
        
        # Assign to emergent region (simple clustering for now)
        deficiency.region_id = self._assign_region(deficiency)
        
        self.deficiency_history.append(deficiency)
        
        return deficiency
    
    def _assign_region(self, deficiency: GeometricDeficiency) -> int:
        """
        Assign deficiency to an emergent region.
        
        Regions emerge from clustering similar deficiencies.
        """
        if not self.deficiency_history:
            return 0
        
        # Simple: cluster by direction similarity
        best_region = 0
        best_similarity = -1
        
        region_directions: Dict[int, List[np.ndarray]] = defaultdict(list)
        for d in self.deficiency_history:
            if d.region_id is not None:
                region_directions[d.region_id].append(d.direction)
        
        for region_id, directions in region_directions.items():
            avg_direction = np.mean(directions, axis=0)
            similarity = np.dot(deficiency.direction, avg_direction)
            if similarity > best_similarity:
                best_similarity = similarity
                best_region = region_id
        
        # If not similar enough, create new region
        if best_similarity < 0.7:
            best_region = max(region_directions.keys(), default=-1) + 1
        
        return best_region
    
    def create_transform_from_fix(self, before: GeometricPosition, 
                                   after: GeometricPosition,
                                   name: str = "learned") -> GeometricTransform:
        """
        Create a transform from a successful fix.
        
        The transform IS the difference between before and after.
        """
        delta = after.coords - before.coords
        transform = GeometricTransform(name, delta)
        self.transforms.append(transform)
        return transform
    
    def find_best_transform(self, deficiency: GeometricDeficiency) -> Optional[GeometricTransform]:
        """
        Find the best transform for this deficiency.
        
        Looks for transforms that:
        1. Point in a similar direction
        2. Have been effective in this region
        """
        if not self.transforms:
            return None
        
        best_transform = None
        best_score = -1
        
        for transform in self.transforms:
            # Score by direction alignment
            alignment = np.dot(transform.delta / (np.linalg.norm(transform.delta) + 1e-10),
                              deficiency.direction)
            
            # Weight by effectiveness
            score = alignment * transform.effectiveness
            
            # Bonus if this transform worked in this region
            if deficiency.region_id in self.region_transforms:
                if transform in self.region_transforms[deficiency.region_id]:
                    score *= 1.5
            
            if score > best_score:
                best_score = score
                best_transform = transform
        
        return best_transform if best_score > 0.3 else None
    
    def improve(self, expected: str, actual: str, 
                fix_function=None) -> Tuple[float, float, Optional[GeometricTransform]]:
        """
        Run one improvement iteration.
        
        Returns: (initial_distance, final_distance, transform_used)
        """
        # Detect deficiency
        deficiency = self.detect_deficiency(expected, actual)
        initial_distance = deficiency.distance
        
        print(f"Deficiency detected:")
        print(f"  Distance: {initial_distance:.3f}")
        print(f"  Region: {deficiency.region_id}")
        print(f"  Direction: {deficiency.direction[:4]}...")
        
        if initial_distance < self.distance_threshold:
            print("  Already acceptable!")
            return initial_distance, initial_distance, None
        
        # Try to find existing transform
        transform = self.find_best_transform(deficiency)
        
        if transform:
            print(f"  Found transform: {transform.name} (effectiveness: {transform.effectiveness:.2f})")
            
            # Apply transform (in real usage, this would modify the gear chain)
            new_pos = transform.apply(deficiency.actual_pos)
            new_distance = deficiency.expected_pos.distance_to(new_pos)
            
            improved = new_distance < initial_distance
            transform.record_result(improved, initial_distance - new_distance)
            
            if improved:
                self.region_transforms[deficiency.region_id].append(transform)
            
            print(f"  New distance: {new_distance:.3f} ({'improved' if improved else 'no improvement'})")
            return initial_distance, new_distance, transform
        
        # No existing transform - need to create one from a fix
        if fix_function:
            print("  No existing transform, applying fix function...")
            fixed_output = fix_function(actual, expected)
            fixed_pos = self.encoder.encode(fixed_output)
            
            new_distance = deficiency.expected_pos.distance_to(fixed_pos)
            
            if new_distance < initial_distance:
                # Learn this transform!
                new_transform = self.create_transform_from_fix(
                    deficiency.actual_pos, fixed_pos, 
                    f"learned_{len(self.transforms)}"
                )
                new_transform.record_result(True, initial_distance - new_distance)
                self.region_transforms[deficiency.region_id].append(new_transform)
                
                # === NEW: Add structure node at the midpoint ===
                # The fix created a bridge - mark this as a structural point
                midpoint = (deficiency.expected_pos.coords + fixed_pos.coords) / 2
                self.add_structure_node(midpoint, f"bridge_{len(self.structure_nodes)}")
                
                print(f"  Learned new transform: {new_transform.name}")
                print(f"  Added structure node: bridge_{len(self.structure_nodes)-1}")
                print(f"  New distance: {new_distance:.3f}")
                return initial_distance, new_distance, new_transform
        
        print("  No fix available")
        return initial_distance, initial_distance, None


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment():
    """Test the geometric improvement loop."""
    
    print("=" * 60)
    print("GEOMETRIC IMPROVEMENT LOOP EXPERIMENT")
    print("=" * 60)
    
    # Create the loop with lower threshold to see dynamics
    loop = GeometricImprovementLoop(dimensions=8)
    loop.distance_threshold = 0.1  # Lower threshold to trigger more fixes
    
    # Learn from some sample text (Moby Dick style)
    corpus = [
        "Captain Ahab commanded the ship Pequod hunting the white whale",
        "The captain led his crew across the ocean in pursuit of Moby Dick",
        "Ahab was the commander of the vessel, obsessed with the whale",
        "The ship sailed through the sea with the captain at the helm",
        "Ishmael joined the crew of the whaling ship under Captain Ahab",
        "The white whale Moby Dick was the target of Ahab's hunt",
        "The crew feared the captain's obsession with the great whale",
        "Across the ocean the Pequod sailed hunting whales",
        # Add more diverse content to create separation
        "The weather was stormy and cold that night",
        "Mathematics involves numbers and equations",
        "The dog ran through the park chasing a ball",
    ]
    
    print("\n1. Learning from corpus...")
    loop.learn_corpus(corpus)
    print(f"   Learned {len(loop.encoder.word_positions)} word positions")
    
    # Test encoding similarity
    print("\n2. Testing semantic similarity via distance...")
    test_pairs = [
        ("captain", "commander"),
        ("captain", "whale"),
        ("ship", "vessel"),
        ("ship", "ocean"),
        ("whale", "Moby Dick"),
    ]
    
    for w1, w2 in test_pairs:
        p1 = loop.encoder.encode(w1)
        p2 = loop.encoder.encode(w2)
        dist = p1.distance_to(p2)
        angle = p1.angle_to(p2)
        print(f"   {w1} <-> {w2}: distance={dist:.3f}, angle={angle:.3f}")
    
    # Test deficiency detection with more divergent examples
    print("\n3. Testing deficiency detection...")
    
    # More divergent: whale topic vs math topic
    expected = "Captain Ahab hunts the white whale on the ocean"
    actual = "The dog ran through the park"  # Completely different topic
    
    deficiency = loop.detect_deficiency(expected, actual)
    print(f"   Expected: '{expected}'")
    print(f"   Actual: '{actual}'")
    print(f"   Distance: {deficiency.distance:.3f}")
    print(f"   Severity: {deficiency.severity:.3f}")
    
    # Test improvement with a simple fix function
    print("\n4. Testing improvement loop with divergent content...")
    
    def simple_fix(actual: str, expected: str) -> str:
        """Simple fix: add key words from expected."""
        expected_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', expected.lower()))
        actual_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', actual.lower()))
        missing = expected_words - actual_words
        if missing:
            return actual + " " + " ".join(list(missing)[:3])
        return actual
    
    initial, final, transform = loop.improve(expected, actual, simple_fix)
    print(f"\n   Initial distance: {initial:.3f}")
    print(f"   Final distance: {final:.3f}")
    print(f"   Improvement: {initial - final:.3f}")
    
    # Test if learned transform applies to similar deficiency
    print("\n5. Testing learned transform on similar deficiency...")
    
    # Similar type of deficiency - wrong topic
    expected2 = "The ship sailed across the sea"
    actual2 = "Mathematics involves equations"  # Also wrong topic
    
    initial2, final2, transform2 = loop.improve(expected2, actual2)
    print(f"\n   Initial distance: {initial2:.3f}")
    print(f"   Final distance: {final2:.3f}")
    if transform2:
        print(f"   Used transform: {transform2.name}")
    
    # Show learned transforms
    print("\n6. Learned transforms:")
    for t in loop.transforms:
        print(f"   {t.name}: effectiveness={t.effectiveness:.2f}, "
              f"successes={t.successes}, failures={t.failures}")
        print(f"      delta magnitude: {np.linalg.norm(t.delta):.3f}")
    
    # Show emergent regions
    print("\n7. Emergent regions:")
    for region_id, transforms in loop.region_transforms.items():
        print(f"   Region {region_id}: {len(transforms)} transforms")
    
    # Show structure nodes
    print("\n8. Structure nodes added:")
    for pos, label in loop.structure_nodes:
        print(f"   {label}: magnitude={np.linalg.norm(pos):.3f}")
    
    # Analyze the geometric structure
    print("\n9. Geometric Analysis:")
    
    # Show word clusters by position
    if loop.encoder.word_positions:
        positions = np.array(list(loop.encoder.word_positions.values()))
        words = list(loop.encoder.word_positions.keys())
        
        # Find clusters using simple distance
        whale_words = ['whale', 'ahab', 'captain', 'ship', 'ocean', 'sea', 'hunt']
        other_words = ['dog', 'park', 'ran', 'ball', 'math', 'numbers', 'equations']
        
        whale_positions = [loop.encoder.word_positions.get(w) for w in whale_words if w in loop.encoder.word_positions]
        other_positions = [loop.encoder.word_positions.get(w) for w in other_words if w in loop.encoder.word_positions]
        
        if whale_positions and other_positions:
            whale_centroid = np.mean(whale_positions, axis=0)
            other_centroid = np.mean(other_positions, axis=0)
            
            cluster_distance = np.linalg.norm(whale_centroid - other_centroid)
            print(f"   Distance between 'whale' cluster and 'other' cluster: {cluster_distance:.3f}")
            
            # Intra-cluster spread
            whale_spread = np.mean([np.linalg.norm(p - whale_centroid) for p in whale_positions])
            other_spread = np.mean([np.linalg.norm(p - other_centroid) for p in other_positions])
            print(f"   Whale cluster spread: {whale_spread:.3f}")
            print(f"   Other cluster spread: {other_spread:.3f}")
            print(f"   Separation ratio: {cluster_distance / (whale_spread + other_spread + 1e-10):.2f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return loop


if __name__ == "__main__":
    loop = run_experiment()
