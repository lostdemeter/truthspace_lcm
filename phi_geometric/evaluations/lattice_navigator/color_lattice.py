"""
Color Knowledge Lattice

The lattice is a graph where:
- Nodes = (feature_signature, color, confidence, context) 
- Edges = valid transformations between nodes

Navigation through the lattice generates synthetic knowledge:
color associations we've never been shown, derived from seed axioms
and valid transformations, verified against natural image statistics.

Architecture (adapted from Ribbon Math v5):
1. Concept Layer: Feature signatures as coordinates
2. N_smooth Layer: Statistical plausibility measure
3. Structure Layer: Valid transformations (lattice edges)
4. Error Analysis: Deviation patterns reveal missing knowledge
5. Verification: Natural image statistics as ground truth
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import cv2
import glob

PHI = 1.618033988749895


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class ColorNode:
    """A node in the color knowledge lattice.
    
    Represents a single piece of color knowledge:
    'Features matching this signature should have this color.'
    """
    # Feature signature (what does this look like in grayscale?)
    brightness: float          # 0-1, normalized intensity
    texture_freq: float        # dominant Gabor frequency response
    texture_orient: float      # dominant orientation (radians)
    local_contrast: float      # 0-1, local variance
    edge_density: float        # 0-1, how many edges nearby
    
    # Spatial context
    y_position: float          # 0-1, vertical position (0=top, 1=bottom)
    x_position: float          # 0-1, horizontal position (0=left, 1=right)
    
    # Color assignment (what color should this be?)
    color_a: float             # Lab a channel (-50 to 50 typical)
    color_b: float             # Lab b channel (-50 to 50 typical)
    
    # Metadata
    confidence: float = 0.5    # 0-1, how sure are we?
    source: str = 'seed'       # 'seed', 'navigated', 'verified'
    generation: int = 0        # how many steps from a seed
    parent_id: int = -1        # which node spawned this one
    context: str = ''          # semantic label
    
    @property
    def hue_angle(self) -> float:
        """Angle in ab space (radians)."""
        return np.arctan2(self.color_b, self.color_a)
    
    @property 
    def saturation(self) -> float:
        """Distance from origin in ab space."""
        return np.sqrt(self.color_a**2 + self.color_b**2)
    
    @property
    def feature_vector(self) -> np.ndarray:
        """7-dim feature signature for matching."""
        return np.array([
            self.brightness, self.texture_freq, self.texture_orient,
            self.local_contrast, self.edge_density,
            self.y_position, self.x_position
        ])
    
    def distance_to(self, other: 'ColorNode') -> float:
        """Feature-space distance to another node."""
        return np.sqrt(np.sum((self.feature_vector - other.feature_vector)**2))


@dataclass
class LatticeEdge:
    """A valid transformation between lattice nodes.
    
    Represents a rule: 'If node A has these properties,
    then a related node B can be created by applying this transformation.'
    """
    name: str
    transform_fn: callable     # function(ColorNode) -> ColorNode
    applicability_fn: callable # function(ColorNode) -> bool
    confidence_decay: float = 0.9  # how much confidence drops per step


# =============================================================================
# SEED AXIOMS
# =============================================================================

def create_seed_axioms() -> List[ColorNode]:
    """
    The 5 seed axioms encoded as initial lattice nodes.
    
    These are NOT rules — they're STARTING POSITIONS on the lattice.
    Everything else will be navigated from these seeds.
    """
    seeds = []
    
    # === AXIOM 1: Sky ===
    # Bright, smooth, low contrast, top of image → blue
    seeds.append(ColorNode(
        brightness=0.7, texture_freq=0.1, texture_orient=0.0,
        local_contrast=0.1, edge_density=0.1,
        y_position=0.15, x_position=0.5,
        color_a=-5, color_b=-25,  # blue in Lab
        confidence=0.95, source='seed', context='sky'
    ))
    
    # === AXIOM 2: Vegetation / Foliage ===
    # Medium brightness, textured, middle-bottom → green
    seeds.append(ColorNode(
        brightness=0.4, texture_freq=0.6, texture_orient=0.5,
        local_contrast=0.4, edge_density=0.5,
        y_position=0.7, x_position=0.5,
        color_a=-15, color_b=20,  # green in Lab
        confidence=0.9, source='seed', context='vegetation'
    ))
    
    # === AXIOM 3: Earth / Ground ===
    # Medium-dark, moderate texture, bottom → brown/tan
    seeds.append(ColorNode(
        brightness=0.35, texture_freq=0.3, texture_orient=0.3,
        local_contrast=0.3, edge_density=0.3,
        y_position=0.85, x_position=0.5,
        color_a=10, color_b=20,  # brown/tan in Lab
        confidence=0.85, source='seed', context='earth'
    ))
    
    # === AXIOM 4: Skin tone ===
    # Medium brightness, smooth, low texture → warm tone
    seeds.append(ColorNode(
        brightness=0.55, texture_freq=0.15, texture_orient=0.0,
        local_contrast=0.15, edge_density=0.2,
        y_position=0.4, x_position=0.5,
        color_a=15, color_b=15,  # warm skin tone in Lab
        confidence=0.8, source='seed', context='skin'
    ))
    
    # === AXIOM 5: Water ===
    # Smooth, medium brightness, lower half → blue-gray
    seeds.append(ColorNode(
        brightness=0.45, texture_freq=0.1, texture_orient=1.57,
        local_contrast=0.15, edge_density=0.1,
        y_position=0.75, x_position=0.5,
        color_a=-5, color_b=-20,  # blue
        confidence=0.85, source='seed', context='water'
    ))
    
    # === Additional seeds for richer coverage ===
    
    # Deep blue sky (very bright, very smooth, very top)
    seeds.append(ColorNode(
        brightness=0.85, texture_freq=0.05, texture_orient=0.0,
        local_contrast=0.05, edge_density=0.05,
        y_position=0.05, x_position=0.5,
        color_a=-10, color_b=-35,  # deeper blue
        confidence=0.9, source='seed', context='deep_sky'
    ))
    
    # Warm road / pavement (medium, smooth, bottom)
    seeds.append(ColorNode(
        brightness=0.4, texture_freq=0.2, texture_orient=0.0,
        local_contrast=0.2, edge_density=0.15,
        y_position=0.8, x_position=0.5,
        color_a=5, color_b=10,  # warm gray
        confidence=0.8, source='seed', context='pavement'
    ))
    
    # Bright green grass (textured, bright, bottom)
    seeds.append(ColorNode(
        brightness=0.5, texture_freq=0.7, texture_orient=0.3,
        local_contrast=0.5, edge_density=0.6,
        y_position=0.8, x_position=0.5,
        color_a=-20, color_b=30,  # vivid green
        confidence=0.85, source='seed', context='grass'
    ))
    
    # Red/warm object (medium, moderate texture)
    seeds.append(ColorNode(
        brightness=0.45, texture_freq=0.3, texture_orient=0.0,
        local_contrast=0.3, edge_density=0.3,
        y_position=0.5, x_position=0.5,
        color_a=30, color_b=20,  # warm red
        confidence=0.7, source='seed', context='warm_object'
    ))
    
    # Blue object (medium, moderate texture)
    seeds.append(ColorNode(
        brightness=0.4, texture_freq=0.3, texture_orient=0.0,
        local_contrast=0.3, edge_density=0.3,
        y_position=0.5, x_position=0.5,
        color_a=-5, color_b=-30,  # blue object
        confidence=0.7, source='seed', context='blue_object'
    ))
    
    # Yellow / bright warm (bright, low texture)
    seeds.append(ColorNode(
        brightness=0.7, texture_freq=0.15, texture_orient=0.0,
        local_contrast=0.15, edge_density=0.1,
        y_position=0.5, x_position=0.5,
        color_a=5, color_b=40,  # yellow
        confidence=0.7, source='seed', context='yellow'
    ))
    
    # Dark shadow (not neutral — slightly warm)
    seeds.append(ColorNode(
        brightness=0.12, texture_freq=0.2, texture_orient=0.0,
        local_contrast=0.1, edge_density=0.2,
        y_position=0.5, x_position=0.5,
        color_a=3, color_b=2,  # slightly warm shadow
        confidence=0.6, source='seed', context='shadow'
    ))
    
    return seeds


# =============================================================================
# LATTICE TRANSFORMATIONS (EDGES)
# =============================================================================

def create_transformations() -> List[LatticeEdge]:
    """
    Define valid transformations between lattice nodes.
    
    Each transformation is a RULE of the physical/perceptual world:
    'If X, then Y is also plausible.'
    
    These are the EDGES of the lattice.
    """
    edges = []
    
    # --- Physical Transformations ---
    
    def shadow_transform(node: ColorNode) -> ColorNode:
        """Shadows: darker, slightly less saturated, slightly cool-shifted."""
        return ColorNode(
            brightness=node.brightness * 0.5,
            texture_freq=node.texture_freq,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast * 0.7,
            edge_density=node.edge_density,
            y_position=node.y_position,
            x_position=node.x_position,
            color_a=node.color_a * 0.8,
            color_b=node.color_b * 0.8 - 1,  # slight cool shift
            confidence=node.confidence * 0.9,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_shadow'
        )
    
    edges.append(LatticeEdge(
        name='shadow',
        transform_fn=shadow_transform,
        applicability_fn=lambda n: n.brightness > 0.2 and n.saturation > 3,
        confidence_decay=0.85
    ))
    
    def highlight_transform(node: ColorNode) -> ColorNode:
        """Highlights: brighter, warm-shifted, preserve saturation."""
        return ColorNode(
            brightness=min(1.0, node.brightness * 1.4),
            texture_freq=node.texture_freq * 0.8,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast * 0.6,
            edge_density=node.edge_density * 0.7,
            y_position=node.y_position,
            x_position=node.x_position,
            color_a=node.color_a * 0.85 + 3,
            color_b=node.color_b * 0.85 + 2,
            confidence=node.confidence * 0.85,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_highlight'
        )
    
    edges.append(LatticeEdge(
        name='highlight',
        transform_fn=highlight_transform,
        applicability_fn=lambda n: n.brightness < 0.85,
        confidence_decay=0.8
    ))
    
    def distance_haze_transform(node: ColorNode) -> ColorNode:
        """Atmospheric perspective: far objects → less saturated, blue-shifted."""
        return ColorNode(
            brightness=node.brightness * 0.9 + 0.1,
            texture_freq=node.texture_freq * 0.5,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast * 0.4,
            edge_density=node.edge_density * 0.3,
            y_position=max(0.0, node.y_position - 0.15),
            x_position=node.x_position,
            color_a=node.color_a * 0.6 - 1,
            color_b=node.color_b * 0.6 - 3,
            confidence=node.confidence * 0.75,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_distant'
        )
    
    edges.append(LatticeEdge(
        name='distance_haze',
        transform_fn=distance_haze_transform,
        applicability_fn=lambda n: n.local_contrast > 0.2,
        confidence_decay=0.7
    ))
    
    # --- Spatial Transformations ---
    
    def vertical_shift_up(node: ColorNode) -> ColorNode:
        """Same features but higher in image → slightly more sky-like."""
        return ColorNode(
            brightness=node.brightness,
            texture_freq=node.texture_freq,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast,
            edge_density=node.edge_density,
            y_position=max(0.0, node.y_position - 0.2),
            x_position=node.x_position,
            color_a=node.color_a * 0.9 - 1,
            color_b=node.color_b * 0.9 - 2,
            confidence=node.confidence * 0.8,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_higher'
        )
    
    edges.append(LatticeEdge(
        name='spatial_up',
        transform_fn=vertical_shift_up,
        applicability_fn=lambda n: n.y_position > 0.3,
        confidence_decay=0.7
    ))
    
    def vertical_shift_down(node: ColorNode) -> ColorNode:
        """Same features but lower in image → slightly more earth-like."""
        return ColorNode(
            brightness=node.brightness,
            texture_freq=node.texture_freq,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast,
            edge_density=node.edge_density,
            y_position=min(1.0, node.y_position + 0.2),
            x_position=node.x_position,
            color_a=node.color_a * 0.9 + 2,
            color_b=node.color_b * 0.9 + 2,
            confidence=node.confidence * 0.8,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_lower'
        )
    
    edges.append(LatticeEdge(
        name='spatial_down',
        transform_fn=vertical_shift_down,
        applicability_fn=lambda n: n.y_position < 0.7,
        confidence_decay=0.7
    ))
    
    # --- Texture Transformations ---
    
    def texture_smoothing(node: ColorNode) -> ColorNode:
        """Smoother version of same material → same color, different texture."""
        return ColorNode(
            brightness=node.brightness,
            texture_freq=node.texture_freq * 0.5,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast * 0.5,
            edge_density=node.edge_density * 0.5,
            y_position=node.y_position,
            x_position=node.x_position,
            color_a=node.color_a * 0.95,
            color_b=node.color_b * 0.95,
            confidence=node.confidence * 0.85,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_smooth'
        )
    
    edges.append(LatticeEdge(
        name='texture_smooth',
        transform_fn=texture_smoothing,
        applicability_fn=lambda n: n.texture_freq > 0.15,
        confidence_decay=0.8
    ))
    
    def texture_roughening(node: ColorNode) -> ColorNode:
        """Rougher version → more saturated, slightly shifted."""
        return ColorNode(
            brightness=node.brightness,
            texture_freq=min(1.0, node.texture_freq * 1.5),
            texture_orient=node.texture_orient,
            local_contrast=min(1.0, node.local_contrast * 1.3),
            edge_density=min(1.0, node.edge_density * 1.3),
            y_position=node.y_position,
            x_position=node.x_position,
            color_a=node.color_a * 1.15,
            color_b=node.color_b * 1.15,
            confidence=node.confidence * 0.75,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_rough'
        )
    
    edges.append(LatticeEdge(
        name='texture_rough',
        transform_fn=texture_roughening,
        applicability_fn=lambda n: n.texture_freq < 0.8,
        confidence_decay=0.75
    ))
    
    # --- Brightness Variations ---
    
    def brightness_vary(node: ColorNode, delta: float) -> ColorNode:
        """Same material at different brightness."""
        new_b = np.clip(node.brightness + delta, 0, 1)
        # Saturation typically peaks at medium brightness
        sat_factor = 1.0 - abs(new_b - 0.5) * 0.4
        return ColorNode(
            brightness=new_b,
            texture_freq=node.texture_freq,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast,
            edge_density=node.edge_density,
            y_position=node.y_position,
            x_position=node.x_position,
            color_a=node.color_a * sat_factor,
            color_b=node.color_b * sat_factor,
            confidence=node.confidence * 0.85,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_bright{"+" if delta>0 else "-"}'
        )
    
    for delta in [-0.15, -0.1, -0.05, 0.05, 0.1, 0.15]:
        edges.append(LatticeEdge(
            name=f'brightness_{delta:+.2f}',
            transform_fn=lambda n, d=delta: brightness_vary(n, d),
            applicability_fn=lambda n, d=delta: 0.05 < n.brightness + d < 0.95,
            confidence_decay=0.85
        ))
    
    # --- Orientation Variation ---
    
    def orient_vary(node: ColorNode) -> ColorNode:
        """Same texture at different orientation → same color."""
        return ColorNode(
            brightness=node.brightness,
            texture_freq=node.texture_freq,
            texture_orient=(node.texture_orient + np.pi/4) % np.pi,
            local_contrast=node.local_contrast,
            edge_density=node.edge_density,
            y_position=node.y_position,
            x_position=node.x_position,
            color_a=node.color_a,
            color_b=node.color_b,
            confidence=node.confidence * 0.9,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_rotated'
        )
    
    edges.append(LatticeEdge(
        name='orient_vary',
        transform_fn=orient_vary,
        applicability_fn=lambda n: n.texture_freq > 0.1,
        confidence_decay=0.9
    ))
    
    # --- Color Harmony (Golden Angle) ---
    
    def complementary_color(node: ColorNode) -> ColorNode:
        """Complementary color at opposite position → scene harmony."""
        return ColorNode(
            brightness=node.brightness,
            texture_freq=node.texture_freq * 0.8 + 0.1,
            texture_orient=node.texture_orient,
            local_contrast=node.local_contrast,
            edge_density=node.edge_density,
            y_position=1.0 - node.y_position,
            x_position=node.x_position,
            color_a=-node.color_a * 0.8,
            color_b=-node.color_b * 0.8,
            confidence=node.confidence * 0.6,
            source='navigated',
            generation=node.generation + 1,
            context=f'{node.context}_complement'
        )
    
    edges.append(LatticeEdge(
        name='complementary',
        transform_fn=complementary_color,
        applicability_fn=lambda n: n.saturation > 5 and n.confidence > 0.5,
        confidence_decay=0.5
    ))
    
    return edges


# =============================================================================
# N_SMOOTH: PLAUSIBILITY MEASURE
# =============================================================================

class NSmooth:
    """
    Measures how plausible a color node is, based on natural image statistics.
    
    Analogous to -log10(|error|) in ribbon math, but for color plausibility.
    Higher = more plausible.
    """
    
    def __init__(self):
        self.natural_stats = None
    
    def build_from_images(self, image_paths: List[str], max_images: int = 100):
        """Build natural image statistics for verification."""
        print('  Building natural image statistics...')
        
        all_brightness = []
        all_ab = []
        all_sat = []
        brightness_ab_pairs = []  # (brightness_bin, a, b) for conditional stats
        position_ab_pairs = []    # (y_bin, a, b)
        
        ct = 0
        for p in image_paths[:max_images]:
            im = cv2.imread(p)
            if im is None: continue
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            if hsv[:,:,1].std() < 30: continue
            ct += 1
            
            r = cv2.resize(im, (64, 64))
            lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
            L = lab[:,:,0].astype(float) / 255.0
            ab = lab[:,:,1:].astype(float) - 128
            sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
            
            all_brightness.append(L.flatten())
            all_ab.append(ab.reshape(-1, 2))
            all_sat.append(sat.flatten())
            
            # Brightness-conditional color stats
            h, w = L.shape
            for yi in range(h):
                for xi in range(w):
                    b_bin = int(L[yi, xi] * 10)  # 0-10
                    y_bin = int(yi / h * 5)       # 0-5
                    brightness_ab_pairs.append((b_bin, ab[yi, xi, 0], ab[yi, xi, 1]))
                    position_ab_pairs.append((y_bin, ab[yi, xi, 0], ab[yi, xi, 1]))
        
        all_ab = np.vstack(all_ab)
        all_sat = np.concatenate(all_sat)
        
        # Build conditional distributions
        # P(color | brightness)
        bright_cond = {}
        for b_bin, a, b in brightness_ab_pairs:
            if b_bin not in bright_cond:
                bright_cond[b_bin] = []
            bright_cond[b_bin].append([a, b])
        
        bright_stats = {}
        for b_bin, colors in bright_cond.items():
            colors = np.array(colors)
            bright_stats[b_bin] = {
                'mean_a': colors[:,0].mean(),
                'mean_b': colors[:,1].mean(),
                'std_a': colors[:,0].std(),
                'std_b': colors[:,1].std(),
                'mean_sat': np.sqrt(colors[:,0]**2 + colors[:,1]**2).mean(),
            }
        
        # P(color | y_position)
        pos_cond = {}
        for y_bin, a, b in position_ab_pairs:
            if y_bin not in pos_cond:
                pos_cond[y_bin] = []
            pos_cond[y_bin].append([a, b])
        
        pos_stats = {}
        for y_bin, colors in pos_cond.items():
            colors = np.array(colors)
            pos_stats[y_bin] = {
                'mean_a': colors[:,0].mean(),
                'mean_b': colors[:,1].mean(),
                'std_a': colors[:,0].std(),
                'std_b': colors[:,1].std(),
            }
        
        self.natural_stats = {
            'overall_mean_a': all_ab[:,0].mean(),
            'overall_mean_b': all_ab[:,1].mean(),
            'overall_std_a': all_ab[:,0].std(),
            'overall_std_b': all_ab[:,1].std(),
            'mean_sat': all_sat.mean(),
            'std_sat': all_sat.std(),
            'max_sat': np.percentile(all_sat, 99),
            'brightness_conditional': bright_stats,
            'position_conditional': pos_stats,
            'n_images': ct,
        }
        print(f'    Stats from {ct} images, mean_sat={all_sat.mean():.1f}')
    
    def compute(self, node: ColorNode) -> float:
        """
        Compute N_smooth for a color node.
        
        Returns a score 0-1 where:
        - 1.0 = perfectly plausible (matches natural image statistics exactly)
        - 0.0 = completely implausible
        """
        if self.natural_stats is None:
            return 0.5  # no stats available
        
        stats = self.natural_stats
        score = 0.0
        n_checks = 0
        
        # Check 1: Is saturation in natural range?
        sat = node.saturation
        if sat < stats['max_sat']:
            score += 1.0
        else:
            score += max(0, 1.0 - (sat - stats['max_sat']) / 20.0)
        n_checks += 1
        
        # Check 2: Is color consistent with brightness?
        b_bin = int(np.clip(node.brightness * 10, 0, 10))
        if b_bin in stats['brightness_conditional']:
            bc = stats['brightness_conditional'][b_bin]
            # Mahalanobis-like distance
            da = (node.color_a - bc['mean_a']) / (bc['std_a'] + 1e-8)
            db = (node.color_b - bc['mean_b']) / (bc['std_b'] + 1e-8)
            dist = np.sqrt(da**2 + db**2)
            score += max(0, 1.0 - dist / 4.0)  # within 4 sigma = ok
        else:
            score += 0.5
        n_checks += 1
        
        # Check 3: Is color consistent with position?
        y_bin = int(np.clip(node.y_position * 5, 0, 4))
        if y_bin in stats['position_conditional']:
            pc = stats['position_conditional'][y_bin]
            da = (node.color_a - pc['mean_a']) / (pc['std_a'] + 1e-8)
            db = (node.color_b - pc['mean_b']) / (pc['std_b'] + 1e-8)
            dist = np.sqrt(da**2 + db**2)
            score += max(0, 1.0 - dist / 4.0)
        else:
            score += 0.5
        n_checks += 1
        
        # Check 4: Saturation-brightness relationship
        # Mid-brightness tends to be most saturated
        expected_sat = stats['mean_sat'] * (1.0 - abs(node.brightness - 0.5) * 0.6)
        sat_diff = abs(sat - expected_sat) / (stats['std_sat'] + 1e-8)
        score += max(0, 1.0 - sat_diff / 3.0)
        n_checks += 1
        
        return score / n_checks


# =============================================================================
# LATTICE NAVIGATOR
# =============================================================================

class LatticeNavigator:
    """
    Navigates the color knowledge lattice from seed axioms.
    
    Generates synthetic knowledge by:
    1. Starting from seed axioms
    2. Applying valid transformations (lattice edges)
    3. Verifying each new node against N_smooth
    4. Accepting plausible nodes, rejecting implausible ones
    5. Repeating until the lattice is dense enough
    """
    
    def __init__(self):
        self.nodes: List[ColorNode] = []
        self.edges = create_transformations()
        self.n_smooth = NSmooth()
        self.generation_stats: Dict[int, Dict] = {}
    
    def initialize(self, image_paths: List[str]):
        """Initialize with seed axioms and build verification stats."""
        print('Initializing Lattice Navigator...')
        
        # Build verification statistics from natural images
        self.n_smooth.build_from_images(image_paths)
        
        # Plant seed axioms
        seeds = create_seed_axioms()
        for i, seed in enumerate(seeds):
            seed_score = self.n_smooth.compute(seed)
            seed.confidence *= seed_score  # adjust by plausibility
            self.nodes.append(seed)
        
        print(f'  {len(seeds)} seed axioms planted')
        
        # Score seeds
        for s in self.nodes:
            ns = self.n_smooth.compute(s)
            print(f'    {s.context}: a={s.color_a:+.0f} b={s.color_b:+.0f} '
                  f'N_smooth={ns:.3f} conf={s.confidence:.2f}')
    
    def navigate(self, max_generations: int = 5, min_confidence: float = 0.1,
                 min_n_smooth: float = 0.3, max_nodes_per_gen: int = 500,
                 verbose: bool = True):
        """
        Navigate the lattice, generating synthetic knowledge.
        
        At each generation:
        1. Take all current nodes
        2. Apply each valid transformation
        3. Check N_smooth of resulting node
        4. Accept if plausible, reject if not
        5. Keep only top-k by confidence per generation
        """
        print(f'\nNavigating lattice for {max_generations} generations '
              f'(max {max_nodes_per_gen}/gen)...')
        
        # Spatial bin for fast redundancy: quantize feature space
        occupied_bins = set()
        for n in self.nodes:
            occupied_bins.add(self._bin_key(n))
        
        for gen in range(max_generations):
            candidates = []
            attempted = 0
            rejected_conf = 0
            rejected_smooth = 0
            rejected_redundant = 0
            
            expandable = [n for n in self.nodes 
                         if n.generation == gen and n.confidence > min_confidence]
            
            for node in expandable:
                for edge in self.edges:
                    if not edge.applicability_fn(node):
                        continue
                    
                    attempted += 1
                    child = edge.transform_fn(node)
                    child.parent_id = id(node)
                    child.generation = gen + 1
                    
                    if child.confidence < min_confidence:
                        rejected_conf += 1
                        continue
                    
                    # Fast redundancy check via spatial binning
                    bk = self._bin_key(child)
                    if bk in occupied_bins:
                        rejected_redundant += 1
                        continue
                    
                    ns = self.n_smooth.compute(child)
                    if ns < min_n_smooth:
                        rejected_smooth += 1
                        continue
                    
                    child.confidence *= ns
                    candidates.append(child)
                    occupied_bins.add(bk)
            
            # Keep top-k by confidence
            candidates.sort(key=lambda n: -n.confidence)
            accepted = candidates[:max_nodes_per_gen]
            self.nodes.extend(accepted)
            for n in accepted:
                occupied_bins.add(self._bin_key(n))
            
            self.generation_stats[gen + 1] = {
                'expandable': len(expandable),
                'attempted': attempted,
                'accepted': len(accepted),
                'rejected_conf': rejected_conf,
                'rejected_smooth': rejected_smooth,
                'rejected_redundant': rejected_redundant,
                'total_nodes': len(self.nodes),
            }
            
            if verbose:
                print(f'  Gen {gen+1}: {len(expandable)} parents → '
                      f'{attempted} tried → {len(candidates)} passed → '
                      f'{len(accepted)} kept (total: {len(self.nodes)})')
        
        print(f'\nNavigation complete: {len(self.nodes)} total nodes '
              f'({len(self.nodes) - len(create_seed_axioms())} navigated)')
    
    @staticmethod
    def _bin_key(node: ColorNode, resolution: int = 20) -> tuple:
        """Quantize a node's features + color into a bin key for fast dedup."""
        return (
            int(node.brightness * resolution),
            int(node.texture_freq * resolution),
            int(node.local_contrast * resolution),
            int(node.y_position * resolution),
            int((node.color_a + 50) / 5),
            int((node.color_b + 50) / 5),
        )
    
    def colorize(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Colorize a grayscale image using lattice knowledge.
        
        Vectorized: compute all pixel features, then batch-match against lattice.
        Uses SELECTION (softmax top-k) not averaging.
        """
        h, w = gray_image.shape
        
        # Extract per-pixel features (vectorized)
        blur3 = cv2.GaussianBlur(gray_image.astype(float), (3,3), 0)
        blur15 = cv2.GaussianBlur(gray_image.astype(float), (15,15), 0)
        local_var = cv2.GaussianBlur(gray_image.astype(float)**2, (15,15), 0) - blur15**2
        local_contrast = np.sqrt(np.maximum(local_var, 0)) / 128.0
        
        gabor_responses = []
        for freq_scale in range(3):
            freq = 0.05 * (PHI ** freq_scale)
            for theta_idx in range(4):
                theta = theta_idx * np.pi / 4
                kernel = cv2.getGaborKernel((15,15), 3.0, theta, 1/freq, 0.5, 0)
                gabor_responses.append(cv2.filter2D(gray_image, cv2.CV_64F, kernel))
        gabor_stack = np.stack(gabor_responses, axis=-1)
        gabor_energy = np.sqrt(np.sum(gabor_stack**2, axis=-1))
        gabor_energy_norm = gabor_energy / (gabor_energy.max() + 1e-8)
        
        orient_responses = []
        for theta_idx in range(4):
            theta = theta_idx * np.pi / 4
            kernel = cv2.getGaborKernel((15,15), 3.0, theta, 0.1, 0.5, 0)
            orient_responses.append(cv2.filter2D(gray_image, cv2.CV_64F, kernel))
        orient_stack = np.stack(orient_responses, axis=-1)
        dominant_orient = np.argmax(np.abs(orient_stack), axis=-1).astype(float) * np.pi / 4
        
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_density = cv2.GaussianBlur(edge_mag, (15,15), 0)
        edge_density = edge_density / (edge_density.max() + 1e-8)
        
        yy, xx = np.meshgrid(np.arange(h) / h, np.arange(w) / w, indexing='ij')
        brightness = blur3 / 255.0
        
        # Stack all pixel features: [h*w, 7]
        pixel_features = np.stack([
            brightness, gabor_energy_norm, dominant_orient / np.pi,
            local_contrast, edge_density, yy, xx
        ], axis=-1).reshape(-1, 7)
        
        # Node arrays
        node_features = np.array([n.feature_vector for n in self.nodes])  # [N, 7]
        node_colors = np.array([[n.color_a, n.color_b] for n in self.nodes])  # [N, 2]
        node_confidences = np.array([n.confidence for n in self.nodes])  # [N]
        
        # Batch compute: distances [h*w, N]
        # Process in chunks to avoid memory blowup
        n_pixels = h * w
        n_nodes = len(self.nodes)
        ab_flat = np.zeros((n_pixels, 2))
        chunk_size = 1024
        
        for start in range(0, n_pixels, chunk_size):
            end = min(start + chunk_size, n_pixels)
            chunk_feat = pixel_features[start:end]  # [chunk, 7]
            
            # Euclidean distance [chunk, N]
            dists = np.sqrt(np.sum(
                (chunk_feat[:, np.newaxis, :] - node_features[np.newaxis, :, :])**2, 
                axis=2
            ))
            
            # Weights = confidence / distance
            weights = node_confidences[np.newaxis, :] / (dists + 0.01)  # [chunk, N]
            
            # Top-k selection per pixel (wider pool for voting)
            k = min(15, n_nodes)
            top_k_idx = np.argpartition(weights, -k, axis=1)[:, -k:]  # [chunk, k]
            
            # Gather top-k weights and colors
            rows_idx = np.arange(end - start)[:, np.newaxis]
            top_weights = weights[rows_idx, top_k_idx]  # [chunk, k]
            top_colors = node_colors[top_k_idx]          # [chunk, k, 2]
            
            # HUE-BIN VOTING: select dominant color family, don't blend
            n_bins = 6
            for pi in range(end - start):
                colors_k = top_colors[pi]    # [k, 2]
                weights_k = top_weights[pi]  # [k]
                
                # Compute hue angles
                hues = np.arctan2(colors_k[:, 1], colors_k[:, 0])  # [-pi, pi]
                sats = np.sqrt(colors_k[:, 0]**2 + colors_k[:, 1]**2)
                
                # Only vote with chromatic nodes (sat > 3)
                chromatic = sats > 3
                if chromatic.sum() < 2:
                    # Not enough chromatic nodes — use best single node
                    best = np.argmax(weights_k)
                    ab_flat[start + pi] = colors_k[best]
                    continue
                
                # Bin hues and vote (weighted by confidence/distance)
                bin_idx = ((hues + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
                bin_votes = np.zeros(n_bins)
                for j in range(k):
                    if chromatic[j]:
                        bin_votes[bin_idx[j]] += weights_k[j]
                
                # Winner takes all
                winning_bin = np.argmax(bin_votes)
                
                # Use only nodes from winning bin
                in_winning = bin_idx == winning_bin
                if in_winning.sum() == 0:
                    best = np.argmax(weights_k)
                    ab_flat[start + pi] = colors_k[best]
                else:
                    # Pick the single highest-weighted node in winning bin
                    mask_weights = np.where(in_winning, weights_k, -1)
                    best_in_bin = np.argmax(mask_weights)
                    ab_flat[start + pi] = colors_k[best_in_bin]
        
        ab = ab_flat.reshape(h, w, 2)
        
        # Light smoothing
        for c in range(2):
            ab[:,:,c] = cv2.bilateralFilter(ab[:,:,c].astype(np.float32), 7, 40, 40)
        
        return ab
    
    def report(self):
        """Print lattice statistics."""
        print('\n=== LATTICE REPORT ===')
        print(f'Total nodes: {len(self.nodes)}')
        
        by_source = {}
        by_context = {}
        for n in self.nodes:
            by_source[n.source] = by_source.get(n.source, 0) + 1
            ctx = n.context.split('_')[0] if n.context else 'unknown'
            by_context[ctx] = by_context.get(ctx, 0) + 1
        
        print(f'By source: {by_source}')
        print(f'By root context: {dict(sorted(by_context.items(), key=lambda x: -x[1]))}')
        
        confs = [n.confidence for n in self.nodes]
        print(f'Confidence: mean={np.mean(confs):.3f}, '
              f'min={np.min(confs):.3f}, max={np.max(confs):.3f}')
        
        if self.generation_stats:
            print('\nGeneration stats:')
            for gen, stats in sorted(self.generation_stats.items()):
                print(f'  Gen {gen}: {stats["accepted"]}/{stats["attempted"]} '
                      f'accepted ({stats["total_nodes"]} total)')


def node_similarity(a: ColorNode, b: ColorNode) -> float:
    """Compute similarity between two nodes (0=different, 1=identical)."""
    feat_dist = np.sqrt(np.sum((a.feature_vector - b.feature_vector)**2))
    color_dist = np.sqrt((a.color_a - b.color_a)**2 + (a.color_b - b.color_b)**2)
    
    feat_sim = np.exp(-feat_dist * 3)
    color_sim = np.exp(-color_dist / 10)
    
    return feat_sim * color_sim
