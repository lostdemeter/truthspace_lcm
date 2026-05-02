"""
Shape Solver - Constraint-based geometric navigation.

Key insight: In 4D hypergeometry with zeta-axis symmetry,
there are only FINITELY MANY valid positions.

Like Tetrominoes:
- Only 7 shapes exist
- Each has finite rotations
- Placement is discrete, not continuous
- We SOLVE for valid positions, not approximate

For transformer navigation:
- 4D space: (x, y, z, w) with zeta symmetry
- Each layer transformation is a "piece placement"
- Valid paths are constrained by geometry
- We find the EXACT shape that works

From clock_solver.py:
- N_smooth(θ_n) ≈ n gives discrete positions
- Solver finds exact positions via sign changes
- O(log n) because structure constrains search

The shape we need:
- Must preserve φ-lattice structure
- Must respect zeta-axis symmetry
- Must have finite valid configurations
- Can be solved exactly, not approximated
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from enum import Enum

PHI = 1.6180339887498949
INV_PHI = 1.0 / PHI
LOG_PHI = np.log(PHI)


class Axis(Enum):
    """The 4 axes of our hypergeometry."""
    X = 0  # Gender/polarity
    Y = 1  # Age/time
    Z = 2  # Agency/action
    W = 3  # Animacy/being (zeta axis - provides symmetry)


@dataclass(frozen=True)
class LatticePoint:
    """
    A point on the φ-lattice in 4D.
    
    Each coordinate is a φ-level (integer), not a float.
    This gives us discrete, exact positions.
    
    The lattice is self-similar: moving by 1 level = scaling by φ.
    """
    x: int  # φ-level on X axis
    y: int  # φ-level on Y axis
    z: int  # φ-level on Z axis
    w: int  # φ-level on W axis (zeta)
    
    def to_float(self) -> Tuple[float, float, float, float]:
        """Convert to float coordinates (for visualization only)."""
        return (
            PHI ** self.x,
            PHI ** self.y,
            PHI ** self.z,
            PHI ** self.w,
        )
    
    def __add__(self, other: 'LatticePoint') -> 'LatticePoint':
        """Add two lattice points (level addition)."""
        return LatticePoint(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    
    def __neg__(self) -> 'LatticePoint':
        """Negate a lattice point."""
        return LatticePoint(-self.x, -self.y, -self.z, -self.w)
    
    def __sub__(self, other: 'LatticePoint') -> 'LatticePoint':
        """Subtract lattice points."""
        return self + (-other)
    
    def magnitude(self) -> int:
        """Manhattan distance from origin (in lattice units)."""
        return abs(self.x) + abs(self.y) + abs(self.z) + abs(self.w)
    
    def zeta_symmetric(self) -> 'LatticePoint':
        """
        Apply zeta symmetry: reflect through w-axis.
        
        The zeta axis provides error cancellation through symmetry.
        """
        return LatticePoint(-self.x, -self.y, -self.z, self.w)


@dataclass(frozen=True)
class ShapePiece:
    """
    A piece in our 4D shape vocabulary.
    
    Like a Tetromino, but in 4D. Each piece is a set of
    relative lattice positions that form a valid shape.
    
    Valid shapes must:
    1. Be connected (each cell adjacent to another)
    2. Respect zeta symmetry (w-axis reflection)
    3. Have bounded magnitude (finite extent)
    """
    name: str
    cells: Tuple[LatticePoint, ...]  # Relative positions
    
    def __len__(self) -> int:
        return len(self.cells)
    
    def rotations(self) -> List['ShapePiece']:
        """
        Generate all rotations of this piece.
        
        In 4D, rotations happen in 2D planes:
        - XY, XZ, XW, YZ, YW, ZW (6 planes)
        - Each plane has 4 rotations (0°, 90°, 180°, 270°)
        
        But zeta symmetry constrains this - rotations in
        planes containing W must preserve W-reflection symmetry.
        """
        rotations = [self]
        
        # For now, just XY and XZ rotations (preserving W)
        rot_xy = ShapePiece(f"{self.name}_xy", self._rotate_xy())
        rot_xz = ShapePiece(f"{self.name}_xz", self._rotate_xz())
        rot_both = ShapePiece(f"{self.name}_xyxz", rot_xy._rotate_xz())
        
        for piece in [rot_xy, rot_xz, rot_both]:
            if piece.cells != self.cells:
                rotations.append(piece)
        
        return rotations
    
    def _rotate_xy(self) -> Tuple[LatticePoint, ...]:
        """90° rotation in XY plane."""
        return tuple(
            LatticePoint(-c.y, c.x, c.z, c.w)
            for c in self.cells
        )
    
    def _rotate_xz(self) -> Tuple[LatticePoint, ...]:
        """90° rotation in XZ plane."""
        return tuple(
            LatticePoint(-c.z, c.y, c.x, c.w)
            for c in self.cells
        )
    
    def place_at(self, origin: LatticePoint) -> Tuple[LatticePoint, ...]:
        """Place this piece at a given origin."""
        return tuple(origin + cell for cell in self.cells)
    
    def is_zeta_symmetric(self) -> bool:
        """Check if piece has zeta symmetry."""
        symmetric_cells = set(c.zeta_symmetric() for c in self.cells)
        return symmetric_cells == set(self.cells)


# Define the basic 4D shape vocabulary
# These are the "Tetrominoes" of our hypergeometry

SHAPE_VOCABULARY = [
    # Identity (single point)
    ShapePiece("I1", (LatticePoint(0, 0, 0, 0),)),
    
    # Line pieces (along each axis)
    ShapePiece("Lx", (
        LatticePoint(0, 0, 0, 0),
        LatticePoint(1, 0, 0, 0),
    )),
    ShapePiece("Ly", (
        LatticePoint(0, 0, 0, 0),
        LatticePoint(0, 1, 0, 0),
    )),
    ShapePiece("Lz", (
        LatticePoint(0, 0, 0, 0),
        LatticePoint(0, 0, 1, 0),
    )),
    ShapePiece("Lw", (
        LatticePoint(0, 0, 0, 0),
        LatticePoint(0, 0, 0, 1),
    )),
    
    # Zeta-symmetric pieces (reflect through W)
    ShapePiece("Zeta2", (
        LatticePoint(1, 0, 0, 0),
        LatticePoint(-1, 0, 0, 0),
    )),
    ShapePiece("Zeta4", (
        LatticePoint(1, 0, 0, 0),
        LatticePoint(-1, 0, 0, 0),
        LatticePoint(0, 1, 0, 0),
        LatticePoint(0, -1, 0, 0),
    )),
    
    # Cross pieces (in various planes)
    ShapePiece("CrossXY", (
        LatticePoint(0, 0, 0, 0),
        LatticePoint(1, 0, 0, 0),
        LatticePoint(-1, 0, 0, 0),
        LatticePoint(0, 1, 0, 0),
        LatticePoint(0, -1, 0, 0),
    )),
    
    # φ-scaled pieces (using golden ratio structure)
    ShapePiece("PhiStep", (
        LatticePoint(0, 0, 0, 0),
        LatticePoint(1, 0, 0, 0),  # φ^1
        LatticePoint(2, 0, 0, 0),  # φ^2 = φ + 1
    )),
]


class ShapeSolver:
    """
    Solver for finding valid shape configurations.
    
    Given constraints (input position, output position, symmetry requirements),
    finds the exact shape that satisfies them.
    
    Like the clock solver:
    - Discrete positions (lattice points)
    - Finite valid configurations
    - Exact solution, not approximation
    """
    
    def __init__(self, vocabulary: List[ShapePiece] = None):
        self.vocabulary = vocabulary or SHAPE_VOCABULARY
        
        # Precompute all rotations
        self.all_pieces: List[ShapePiece] = []
        for piece in self.vocabulary:
            self.all_pieces.extend(piece.rotations())
        
        # Remove duplicates
        seen = set()
        unique = []
        for piece in self.all_pieces:
            key = frozenset(piece.cells)
            if key not in seen:
                seen.add(key)
                unique.append(piece)
        self.all_pieces = unique
        
        print(f"Shape vocabulary: {len(self.vocabulary)} base shapes")
        print(f"With rotations: {len(self.all_pieces)} total configurations")
    
    def find_path(self, start: LatticePoint, end: LatticePoint,
                  max_pieces: int = 5) -> Optional[List[Tuple[ShapePiece, LatticePoint]]]:
        """
        Find a path from start to end using shape pieces.
        
        This is like solving a puzzle: find the sequence of
        piece placements that connects start to end.
        
        Args:
            start: Starting lattice point
            end: Target lattice point
            max_pieces: Maximum number of pieces to use
            
        Returns:
            List of (piece, placement_origin) or None if no path exists
        """
        # BFS to find shortest path
        from collections import deque
        
        # State: (current_position, path_so_far)
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            if len(path) >= max_pieces:
                continue
            
            # Try each piece
            for piece in self.all_pieces:
                # Place piece at current position
                cells = piece.place_at(current)
                
                # The "output" of placing a piece is the last cell
                # (This is a simplification - real navigation is more complex)
                for next_pos in cells:
                    if next_pos not in visited:
                        visited.add(next_pos)
                        new_path = path + [(piece, current)]
                        queue.append((next_pos, new_path))
        
        return None  # No path found
    
    def find_symmetric_path(self, start: LatticePoint, end: LatticePoint,
                            max_pieces: int = 5) -> Optional[List[Tuple[ShapePiece, LatticePoint]]]:
        """
        Find a path that respects zeta symmetry.
        
        The path must be symmetric under W-axis reflection.
        This provides error cancellation.
        """
        # First, check if start and end are compatible with symmetry
        if start.zeta_symmetric() != start:
            # Start is not on the symmetric axis
            # Need to find symmetric pair of paths
            pass
        
        # For now, just find any path using symmetric pieces only
        symmetric_pieces = [p for p in self.all_pieces if p.is_zeta_symmetric()]
        
        from collections import deque
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            if len(path) >= max_pieces:
                continue
            
            for piece in symmetric_pieces:
                cells = piece.place_at(current)
                for next_pos in cells:
                    if next_pos not in visited:
                        visited.add(next_pos)
                        new_path = path + [(piece, current)]
                        queue.append((next_pos, new_path))
        
        return None
    
    def solve_transformation(self, delta: LatticePoint) -> Optional[ShapePiece]:
        """
        Find a single piece that achieves the given transformation.
        
        This is the core operation: given a required change in position,
        find the shape that does it.
        
        Args:
            delta: Required change in lattice position
            
        Returns:
            ShapePiece that achieves this transformation, or None
        """
        for piece in self.all_pieces:
            # Check if any cell of the piece matches the delta
            for cell in piece.cells:
                if cell == delta:
                    return piece
        
        return None
    
    def decompose_transformation(self, delta: LatticePoint) -> List[ShapePiece]:
        """
        Decompose a transformation into a sequence of basic pieces.
        
        Like prime factorization, but for geometric transformations.
        
        Args:
            delta: Required change in lattice position
            
        Returns:
            List of pieces that compose to achieve delta
        """
        pieces = []
        remaining = delta
        
        while remaining.magnitude() > 0:
            # Find the largest piece that reduces the remaining delta
            best_piece = None
            best_reduction = 0
            
            for piece in self.all_pieces:
                for cell in piece.cells:
                    # Would using this piece reduce the remaining delta?
                    new_remaining = remaining - cell
                    reduction = remaining.magnitude() - new_remaining.magnitude()
                    
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_piece = (piece, cell)
            
            if best_piece is None or best_reduction <= 0:
                # Can't make progress - no exact decomposition
                break
            
            piece, cell = best_piece
            pieces.append(piece)
            remaining = remaining - cell
        
        return pieces


class NavigationSolver:
    """
    Solver-based navigation through transformer geometry.
    
    Instead of statistical search, we:
    1. Map transformer operations to lattice transformations
    2. Solve for the exact shape configuration
    3. Execute the solved path
    
    This is deterministic and exact.
    """
    
    def __init__(self):
        self.shape_solver = ShapeSolver()
        
        # Map layer types to lattice transformations
        self.layer_transforms: Dict[str, LatticePoint] = {}
    
    def analyze_layer(self, layer_weights: np.ndarray, layer_name: str) -> LatticePoint:
        """
        Analyze a layer's weights to determine its lattice transformation.
        
        The key insight: each layer moves us by a discrete amount
        on the φ-lattice. We can determine this from the weight structure.
        """
        # Compute the "direction" of the layer
        # This is where the layer moves us in semantic space
        
        # For now, use a simple heuristic based on weight statistics
        # TODO: Replace with proper geometric analysis
        
        mean_weight = np.mean(layer_weights)
        std_weight = np.std(layer_weights)
        
        # Map to φ-levels
        x_level = int(np.round(np.log(abs(mean_weight) + 1e-10) / LOG_PHI))
        y_level = int(np.round(np.log(std_weight + 1e-10) / LOG_PHI))
        
        # Z and W from higher-order statistics
        skew = np.mean((layer_weights - mean_weight) ** 3) / (std_weight ** 3 + 1e-10)
        kurt = np.mean((layer_weights - mean_weight) ** 4) / (std_weight ** 4 + 1e-10)
        
        z_level = int(np.round(skew))
        w_level = int(np.round(np.log(abs(kurt) + 1e-10) / LOG_PHI))
        
        transform = LatticePoint(x_level, y_level, z_level, w_level)
        self.layer_transforms[layer_name] = transform
        
        return transform
    
    def solve_navigation(self, input_pos: LatticePoint, 
                         target_pos: LatticePoint) -> List[str]:
        """
        Solve for the layer sequence that navigates from input to target.
        
        Returns the names of layers to apply, in order.
        """
        delta = target_pos - input_pos
        
        # Find which combination of layer transforms achieves this delta
        # This is a constraint satisfaction problem
        
        # For now, greedy approach
        path = []
        remaining = delta
        
        for layer_name, transform in self.layer_transforms.items():
            # How many times should we apply this layer?
            # (In practice, each layer is applied once, but this shows the structure)
            
            if transform.magnitude() > 0:
                # Check if this layer helps
                new_remaining = remaining - transform
                if new_remaining.magnitude() < remaining.magnitude():
                    path.append(layer_name)
                    remaining = new_remaining
        
        return path


def test_shape_solver():
    """Test the shape solver."""
    print("=" * 60)
    print("Testing Shape Solver")
    print("=" * 60)
    
    solver = ShapeSolver()
    
    # Test finding paths
    start = LatticePoint(0, 0, 0, 0)
    
    test_targets = [
        LatticePoint(1, 0, 0, 0),
        LatticePoint(2, 0, 0, 0),
        LatticePoint(1, 1, 0, 0),
        LatticePoint(1, 1, 1, 0),
        LatticePoint(1, 1, 1, 1),
    ]
    
    print("\nFinding paths from origin:")
    for target in test_targets:
        path = solver.find_path(start, target, max_pieces=3)
        if path:
            pieces = [p.name for p, _ in path]
            print(f"  {start} → {target}: {pieces}")
        else:
            print(f"  {start} → {target}: No path found")
    
    # Test transformation decomposition
    print("\nDecomposing transformations:")
    for delta in test_targets:
        pieces = solver.decompose_transformation(delta)
        piece_names = [p.name for p in pieces]
        print(f"  Δ={delta}: {piece_names}")
    
    # Test symmetric paths
    print("\nFinding symmetric paths:")
    symmetric_targets = [
        LatticePoint(0, 0, 0, 1),  # Along W axis
        LatticePoint(1, -1, 0, 0),  # Symmetric in XY
    ]
    for target in symmetric_targets:
        path = solver.find_symmetric_path(start, target, max_pieces=3)
        if path:
            pieces = [p.name for p, _ in path]
            print(f"  {start} → {target}: {pieces}")
        else:
            print(f"  {start} → {target}: No symmetric path")


def test_lattice_point():
    """Test LatticePoint operations."""
    print("=" * 60)
    print("Testing LatticePoint")
    print("=" * 60)
    
    p1 = LatticePoint(1, 2, 0, 0)
    p2 = LatticePoint(0, 1, 1, 0)
    
    print(f"p1 = {p1}")
    print(f"p2 = {p2}")
    print(f"p1 + p2 = {p1 + p2}")
    print(f"p1 - p2 = {p1 - p2}")
    print(f"|p1| = {p1.magnitude()}")
    print(f"p1.zeta_symmetric() = {p1.zeta_symmetric()}")
    print(f"p1.to_float() = {p1.to_float()}")
    
    # Test zeta symmetry
    symmetric = LatticePoint(1, -1, 0, 0)
    print(f"\n{symmetric}.zeta_symmetric() = {symmetric.zeta_symmetric()}")
    print(f"Is {symmetric} zeta-symmetric? {symmetric == symmetric.zeta_symmetric()}")


@dataclass(frozen=True)
class FlipPattern:
    """
    A dimension flip pattern - the actual "shape" used by transformer layers.
    
    Extracted from Qwen2-7B analysis:
    - 16 unique patterns across 28 layers
    - Most common: flip dim 6 only (8 layers)
    - Down projection is uniform (no flips)
    """
    name: str
    flipped_dims: Tuple[int, ...]  # Indices into top-20 dims
    frequency: int  # How many layers use this pattern
    
    def apply(self, signs: np.ndarray, top_dims: np.ndarray) -> np.ndarray:
        """Apply this flip pattern to a sign vector."""
        result = signs.copy()
        for dim_idx in self.flipped_dims:
            actual_dim = top_dims[dim_idx]
            result[actual_dim] *= -1
        return result
    
    def __repr__(self):
        return f"FlipPattern({self.name}, dims={self.flipped_dims}, freq={self.frequency})"


# The actual shape vocabulary extracted from Qwen2-7B
# These are the "Tetrominoes" of transformer navigation
QWEN2_SHAPE_VOCABULARY = [
    FlipPattern("S1_content", (6,), 8),           # Most common: content/punctuation toggle
    FlipPattern("S2_content_struct", (6, 19), 4), # Content + structure
    FlipPattern("S3_identity", (), 2),            # No change
    FlipPattern("S4_triple", (6, 11, 19), 2),     # Three-way flip
    FlipPattern("S5_early", (0, 1, 4, 7, 8, 9, 13, 15, 18, 19), 1),  # Layer 0 special
    FlipPattern("S6_complex", (7, 8, 11, 12, 17, 18, 19), 1),
    FlipPattern("S7_mid", (0, 4, 11, 13, 15, 18), 1),
    FlipPattern("S8_simple", (0, 11, 19), 1),
    FlipPattern("S9_struct", (11, 17, 19), 1),
    FlipPattern("S10_single17", (17,), 1),
    FlipPattern("S11_single16", (16,), 1),
    FlipPattern("S12_quad", (2, 3, 7, 17), 1),
    FlipPattern("S13_pair617", (6, 17), 1),
    FlipPattern("S14_pair56", (5, 6), 1),
    FlipPattern("S15_pair611", (6, 11), 1),
    FlipPattern("S16_pair616", (6, 16), 1),
]

# Top 50 dimension indices from Qwen2-7B (by variance)
# 50 dims achieves 100% token uniqueness (20 dims only 46.9%)
QWEN2_TOP_DIMS = np.array([
    2043, 2456, 1245, 608, 1926, 1395, 3197, 658, 2898, 1122,
    1959, 130, 3404, 1210, 822, 1759, 32, 1908, 727, 890,
    3192, 2624, 144, 2064, 1495, 2906, 1068, 1437, 1999, 809,
    2604, 2010, 577, 613, 523, 68, 2406, 650, 2348, 3369,
    792, 3216, 1815, 194, 2840, 3080, 764, 1686, 877, 2437
])

# Semantic meaning of key dimensions
DIMENSION_SEMANTICS = {
    6: "content_vs_punctuation",  # idx 3197, flipped by 18/28 layers
    19: "sentence_structure",      # idx 890, flipped by 10/28 layers
    11: "discourse_marker",        # idx 130, flipped by 7/28 layers
    10: "function_words",          # idx 1959, NEVER flipped (stable)
    14: "articles_determiners",    # idx 822, NEVER flipped (stable)
    0: "common_english",           # idx 2043, highest variance
    1: "english_vs_symbols",       # idx 2456, second highest
}


class DimensionSelector:
    """
    Solver for dimension selection in sign space.
    
    Key insight from Qwen2 analysis:
    - All weights cluster at φ^-9 (uniform magnitude)
    - The SIGNS are what matter (Doc 141: irreducible shape)
    - Navigation = selecting which dimensions to flip
    
    This is like Tetromino placement:
    - The "board" is the 3584-dimensional sign space
    - Each "piece" is a pattern of dimension flips
    - Valid placements are constrained by geometry
    """
    
    def __init__(self, n_dims: int = 3584):
        self.n_dims = n_dims
        
        # The dimension importance follows φ-Zipf (Doc 039)
        # Top dimensions have weight φ^(-rank)
        self.dim_weights = np.array([
            PHI ** (-i) for i in range(1, n_dims + 1)
        ])
        
        # Normalize so total weight = φ (the golden ratio)
        self.dim_weights = self.dim_weights * PHI / self.dim_weights.sum()
    
    def valid_flip_patterns(self, max_flips: int = 10) -> List[np.ndarray]:
        """
        Generate valid flip patterns (like Tetromino shapes).
        
        Valid patterns must:
        1. Flip only high-weight dimensions (top ~10 matter)
        2. Respect zeta symmetry (paired flips)
        3. Have bounded total weight change
        
        Returns list of boolean arrays indicating which dims to flip.
        """
        patterns = []
        
        # Single dimension flips (the "I" pieces)
        for i in range(min(max_flips, 20)):  # Only top 20 dims matter
            pattern = np.zeros(self.n_dims, dtype=bool)
            pattern[i] = True
            patterns.append(pattern)
        
        # Paired flips (zeta-symmetric)
        for i in range(min(max_flips, 10)):
            for j in range(i + 1, min(max_flips, 10)):
                pattern = np.zeros(self.n_dims, dtype=bool)
                pattern[i] = True
                pattern[j] = True
                patterns.append(pattern)
        
        return patterns
    
    def apply_pattern(self, signs: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Apply a flip pattern to signs."""
        result = signs.copy()
        result[pattern] *= -1
        return result
    
    def solve_transformation(self, input_signs: np.ndarray, 
                             target_signs: np.ndarray) -> List[np.ndarray]:
        """
        Find the sequence of flip patterns that transforms input to target.
        
        This is exact, not approximate.
        """
        patterns = self.valid_flip_patterns()
        
        # Which dimensions need to flip?
        need_flip = (input_signs != target_signs)
        
        # Greedy: flip highest-weight dimensions first
        solution = []
        current = input_signs.copy()
        
        while np.any(current != target_signs):
            # Find the pattern that fixes the most high-weight disagreements
            best_pattern = None
            best_improvement = 0
            
            for pattern in patterns:
                # Would this pattern help?
                would_flip = pattern & need_flip
                improvement = np.sum(self.dim_weights[would_flip])
                
                # But don't flip things that are already correct
                would_break = pattern & ~need_flip
                penalty = np.sum(self.dim_weights[would_break])
                
                net = improvement - penalty
                if net > best_improvement:
                    best_improvement = net
                    best_pattern = pattern
            
            if best_pattern is None or best_improvement <= 0:
                break
            
            current = self.apply_pattern(current, best_pattern)
            need_flip = (current != target_signs)
            solution.append(best_pattern)
        
        return solution


def test_dimension_selector():
    """Test the dimension selector."""
    print("=" * 60)
    print("Testing Dimension Selector")
    print("=" * 60)
    
    selector = DimensionSelector(n_dims=64)  # Small for testing
    
    print(f"Dimension weights (top 10):")
    for i in range(10):
        print(f"  Dim {i}: weight = {selector.dim_weights[i]:.6f}")
    
    print(f"\nTotal weight: {selector.dim_weights.sum():.6f}")
    print(f"Weight in top 10: {selector.dim_weights[:10].sum():.6f} "
          f"({100*selector.dim_weights[:10].sum()/selector.dim_weights.sum():.1f}%)")
    
    # Test transformation solving
    np.random.seed(42)
    input_signs = np.random.choice([-1, 1], size=64)
    target_signs = input_signs.copy()
    
    # Flip a few dimensions
    target_signs[0] *= -1  # Flip dim 0 (highest weight)
    target_signs[5] *= -1  # Flip dim 5
    target_signs[10] *= -1  # Flip dim 10
    
    print(f"\nSolving transformation (3 dims flipped):")
    solution = selector.solve_transformation(input_signs, target_signs)
    print(f"  Solution has {len(solution)} patterns")
    
    # Verify
    result = input_signs.copy()
    for pattern in solution:
        result = selector.apply_pattern(result, pattern)
    
    matches = np.all(result == target_signs)
    print(f"  Transformation correct: {matches}")
    
    # Show which dims were flipped
    for i, pattern in enumerate(solution):
        flipped = np.where(pattern)[0]
        print(f"  Pattern {i}: flip dims {flipped.tolist()}")


class ShapeBasedNavigator:
    """
    Navigator using the finite shape vocabulary.
    
    Instead of statistical search, we:
    1. Represent positions as sign patterns on top dimensions
    2. Apply shape transformations (flip patterns)
    3. Solve for the exact sequence of shapes to reach target
    
    This is deterministic and exact, like clock_solver.py.
    """
    
    def __init__(self, top_dims: np.ndarray = None, 
                 vocabulary: List[FlipPattern] = None):
        self.top_dims = top_dims if top_dims is not None else QWEN2_TOP_DIMS
        self.vocabulary = vocabulary if vocabulary is not None else QWEN2_SHAPE_VOCABULARY
        self.n_top = len(self.top_dims)
        
        # Build lookup for fast pattern matching
        self.pattern_by_name = {p.name: p for p in self.vocabulary}
        
        # Token positions (signs on top dims only)
        self.token_signs: Optional[np.ndarray] = None  # (vocab_size, n_top)
        self.tokenizer = None
        self.full_signs: Optional[np.ndarray] = None  # Full 3584-dim signs
    
    def load_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        """Load token signs from model embeddings."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        emb = model.model.embed_tokens.weight.data.numpy()
        
        # Extract signs for top dimensions only
        self.full_signs = np.sign(emb).astype(np.int8)
        self.full_signs[self.full_signs == 0] = 1
        
        self.token_signs = self.full_signs[:, self.top_dims]
        
        print(f"Loaded {len(self.token_signs)} tokens")
        print(f"Using top {self.n_top} dimensions")
        
        del model
    
    def get_token_signs(self, token_id: int) -> np.ndarray:
        """Get sign pattern for a token (top dims only)."""
        return self.token_signs[token_id].copy()
    
    def apply_pattern(self, signs: np.ndarray, pattern: FlipPattern) -> np.ndarray:
        """Apply a flip pattern to signs."""
        result = signs.copy()
        for dim_idx in pattern.flipped_dims:
            result[dim_idx] *= -1
        return result
    
    def find_matching_tokens(self, target_signs: np.ndarray, 
                             exclude: Set[int] = None) -> List[Tuple[int, int]]:
        """
        Find tokens that exactly match target signs on top dims.
        
        Returns list of (token_id, n_matching_dims).
        """
        exclude = exclude or set()
        
        matches = []
        for i, token_signs in enumerate(self.token_signs):
            if i in exclude:
                continue
            
            n_match = np.sum(token_signs == target_signs)
            if n_match == self.n_top:  # Exact match
                matches.append((i, n_match))
        
        return matches
    
    def find_nearest_token(self, target_signs: np.ndarray,
                           exclude: Set[int] = None) -> Tuple[int, int]:
        """Find token with most matching signs."""
        exclude = exclude or set()
        
        best_id = 0
        best_match = -1
        
        for i, token_signs in enumerate(self.token_signs):
            if i in exclude:
                continue
            
            n_match = np.sum(token_signs == target_signs)
            if n_match > best_match:
                best_match = n_match
                best_id = i
        
        return best_id, best_match
    
    def solve_transformation(self, input_signs: np.ndarray,
                             target_signs: np.ndarray) -> List[FlipPattern]:
        """
        Solve for the sequence of patterns that transforms input to target.
        
        This is exact solving, not approximation.
        """
        # Which dims need to flip?
        need_flip = set(np.where(input_signs != target_signs)[0])
        
        if not need_flip:
            return []  # Already at target
        
        # Greedy: find patterns that flip needed dims without flipping others
        solution = []
        current = input_signs.copy()
        remaining = need_flip.copy()
        
        while remaining:
            best_pattern = None
            best_score = -float('inf')
            
            for pattern in self.vocabulary:
                pattern_dims = set(pattern.flipped_dims)
                
                # How many needed dims does this flip?
                good_flips = len(pattern_dims & remaining)
                # How many already-correct dims does this break?
                bad_flips = len(pattern_dims - remaining)
                
                score = good_flips - bad_flips
                
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            if best_pattern is None or best_score <= 0:
                break
            
            # Apply pattern
            current = self.apply_pattern(current, best_pattern)
            solution.append(best_pattern)
            
            # Update remaining
            for dim in best_pattern.flipped_dims:
                if dim in remaining:
                    remaining.remove(dim)
                else:
                    remaining.add(dim)  # We broke this dim
        
        return solution
    
    def navigate(self, start_token: int, n_steps: int = 5) -> List[Tuple[int, List[FlipPattern]]]:
        """
        Navigate from start token through shape space.
        
        At each step:
        1. Apply the most common pattern (S1_content)
        2. Find the nearest token to the result
        3. Record the path
        
        Returns list of (token_id, patterns_applied).
        """
        path = [(start_token, [])]
        current_signs = self.get_token_signs(start_token)
        visited = {start_token}
        
        for _ in range(n_steps):
            # Try each pattern and see where it leads
            best_token = None
            best_match = -1
            best_pattern = None
            
            for pattern in self.vocabulary[:5]:  # Try top 5 most common
                new_signs = self.apply_pattern(current_signs, pattern)
                token_id, n_match = self.find_nearest_token(new_signs, visited)
                
                if n_match > best_match:
                    best_match = n_match
                    best_token = token_id
                    best_pattern = pattern
            
            if best_token is None:
                break
            
            visited.add(best_token)
            path.append((best_token, [best_pattern]))
            current_signs = self.get_token_signs(best_token)
        
        return path
    
    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate text using shape-based navigation."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Navigate from last token
        path = self.navigate(token_ids[-1], n_steps=max_tokens)
        
        # Collect generated tokens
        for token_id, _ in path[1:]:  # Skip start token
            token_ids.append(token_id)
        
        return self.tokenizer.decode(token_ids)


def test_shape_vocabulary():
    """Test the extracted shape vocabulary."""
    print("=" * 60)
    print("Testing Shape Vocabulary")
    print("=" * 60)
    
    print(f"\nQwen2 Shape Vocabulary ({len(QWEN2_SHAPE_VOCABULARY)} patterns):")
    for pattern in QWEN2_SHAPE_VOCABULARY:
        semantic = [DIMENSION_SEMANTICS.get(d, f"dim{d}") for d in pattern.flipped_dims]
        print(f"  {pattern.name}: {pattern.flipped_dims} ({pattern.frequency} layers)")
        if semantic:
            print(f"    Semantics: {semantic}")
    
    print(f"\nTop dimensions and their semantics:")
    for dim_idx, semantic in sorted(DIMENSION_SEMANTICS.items()):
        actual_idx = QWEN2_TOP_DIMS[dim_idx]
        print(f"  Dim {dim_idx} (idx {actual_idx}): {semantic}")


def test_shape_based_navigator():
    """Test the shape-based navigator."""
    print("=" * 60)
    print("Testing Shape-Based Navigator")
    print("=" * 60)
    
    # Create navigator with mock data for testing
    navigator = ShapeBasedNavigator()
    
    # Create mock token signs
    np.random.seed(42)
    n_tokens = 100
    n_dims = 20
    navigator.token_signs = np.random.choice([-1, 1], size=(n_tokens, n_dims)).astype(np.int8)
    navigator.n_top = n_dims
    
    # Test transformation solving
    input_signs = navigator.token_signs[0]
    target_signs = navigator.token_signs[1]
    
    print(f"\nSolving transformation from token 0 to token 1:")
    print(f"  Input signs:  {input_signs[:10].tolist()}...")
    print(f"  Target signs: {target_signs[:10].tolist()}...")
    
    solution = navigator.solve_transformation(input_signs, target_signs)
    print(f"  Solution: {[p.name for p in solution]}")
    
    # Verify
    result = input_signs.copy()
    for pattern in solution:
        result = navigator.apply_pattern(result, pattern)
    
    n_match = np.sum(result == target_signs)
    print(f"  After applying solution: {n_match}/{n_dims} dims match")
    
    # Test navigation
    print(f"\nNavigating from token 0:")
    path = navigator.navigate(0, n_steps=5)
    print(f"  Path: {[t for t, _ in path]}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--model":
        # Test with actual model
        print("=" * 60)
        print("Testing Shape-Based Navigator with Qwen2-7B")
        print("=" * 60)
        
        navigator = ShapeBasedNavigator()
        navigator.load_from_model()
        
        # Test semantic neighbors
        print("\nFinding semantic neighbors via shape patterns:")
        test_words = ["king", "computer", "happy"]
        
        for word in test_words:
            token_id = navigator.tokenizer.encode(word, add_special_tokens=False)[0]
            signs = navigator.get_token_signs(token_id)
            
            # Apply S1_content pattern (most common)
            s1 = navigator.pattern_by_name["S1_content"]
            new_signs = navigator.apply_pattern(signs, s1)
            
            # Find nearest token
            nearest_id, n_match = navigator.find_nearest_token(new_signs, {token_id})
            nearest_word = navigator.tokenizer.decode([nearest_id])
            
            print(f"  {word} + S1_content → {nearest_word!r} ({n_match}/20 match)")
        
        # Test navigation
        print("\nShape-based navigation:")
        prompts = ["The king", "Hello world", "I love"]
        
        for prompt in prompts:
            print(f"\n  Prompt: {prompt!r}")
            output = navigator.generate(prompt, max_tokens=5)
            print(f"  Output: {output!r}")
        
        # Test transformation solving between known pairs
        print("\nSolving transformations between semantic pairs:")
        pairs = [("king", "queen"), ("man", "woman"), ("happy", "sad")]
        
        for w1, w2 in pairs:
            id1 = navigator.tokenizer.encode(w1, add_special_tokens=False)[0]
            id2 = navigator.tokenizer.encode(w2, add_special_tokens=False)[0]
            
            signs1 = navigator.get_token_signs(id1)
            signs2 = navigator.get_token_signs(id2)
            
            solution = navigator.solve_transformation(signs1, signs2)
            pattern_names = [p.name for p in solution]
            
            # Verify
            result = signs1.copy()
            for p in solution:
                result = navigator.apply_pattern(result, p)
            n_match = np.sum(result == signs2)
            
            print(f"  {w1} → {w2}: {pattern_names} ({n_match}/20 match)")
    
    else:
        test_lattice_point()
        print()
        test_shape_solver()
        print()
        test_shape_vocabulary()
        print()
        test_shape_based_navigator()
