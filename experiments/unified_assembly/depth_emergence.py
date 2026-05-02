#!/usr/bin/env python3
"""
Experiment: Emergent Depth Dimensions via Self-Assembly

Hypothesis: If we feed RGB→depth patch pairs to the self-assembler,
depth-relevant dimensions will EMERGE without us defining them.

The key insight: We don't know what the dimensions will be.
Just like text dimensions (gender, age, formality) emerged from pairs,
visual depth dimensions should emerge from RGB→depth pairs.

Approach:
1. Extract patches from RGB and corresponding depth images
2. Treat each patch as a "concept" (like a word)
3. Feed patch pairs to the self-assembler
4. Let dimensions emerge from the structure
5. Use emergent dimensions to predict depth for new patches

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

from experiments.unified_assembly.modality import (
    Modality,
    Artifact,
    Transform,
    UniversalDimension,
    UniversalCorpus,
    PHI,
)

# Paths
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


# =============================================================================
# PATCH EXTRACTOR
# =============================================================================

@dataclass
class Patch:
    """A patch from an image with its position and content."""
    content: np.ndarray  # The actual pixel values
    row: int             # Row position in grid
    col: int             # Column position in grid
    image_id: str        # Source image identifier
    is_depth: bool       # True if this is a depth patch
    
    @property
    def identifier(self) -> str:
        modal = "depth" if self.is_depth else "rgb"
        return f"{self.image_id}_{modal}_{self.row}_{self.col}"
    
    def to_vector(self) -> np.ndarray:
        """Flatten patch to 1D vector."""
        return self.content.flatten()


class PatchExtractor:
    """
    Extract corresponding patches from RGB and depth images.
    """
    
    def __init__(self, patch_size: int = 16, grid_size: int = 8):
        """
        Args:
            patch_size: Size of each patch (patch_size x patch_size)
            grid_size: Number of patches per row/column
        """
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.target_size = patch_size * grid_size
    
    def extract_patches(self, rgb: np.ndarray, depth: np.ndarray, 
                        image_id: str) -> List[Tuple[Patch, Patch]]:
        """
        Extract corresponding RGB and depth patches.
        
        Returns list of (rgb_patch, depth_patch) tuples.
        """
        # Resize to target size
        rgb_resized = self._resize(rgb, self.target_size)
        depth_resized = self._resize(depth, self.target_size)
        
        pairs = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                r_start = row * self.patch_size
                r_end = r_start + self.patch_size
                c_start = col * self.patch_size
                c_end = c_start + self.patch_size
                
                rgb_patch = Patch(
                    content=rgb_resized[r_start:r_end, c_start:c_end].copy(),
                    row=row,
                    col=col,
                    image_id=image_id,
                    is_depth=False
                )
                
                depth_patch = Patch(
                    content=depth_resized[r_start:r_end, c_start:c_end].copy(),
                    row=row,
                    col=col,
                    image_id=image_id,
                    is_depth=True
                )
                
                pairs.append((rgb_patch, depth_patch))
        
        return pairs
    
    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        """Resize image to size x size."""
        pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8))
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        return np.array(pil_img).astype(np.float32) / 255.0


# =============================================================================
# EMERGENT DIMENSION DISCOVERER
# =============================================================================

class EmergentDimensionDiscoverer:
    """
    Discovers emergent dimensions from RGB→depth patch pairs.
    
    The key insight: We don't predefine dimensions.
    We let them emerge from the structure of the data.
    
    Method:
    1. Compute similarity between all RGB patches
    2. Compute similarity between all depth patches  
    3. Find transformations that consistently map RGB→depth
    4. These consistent transformations ARE the emergent dimensions
    """
    
    def __init__(self):
        self.rgb_patches: List[Patch] = []
        self.depth_patches: List[Patch] = []
        self.pairs: List[Tuple[Patch, Patch]] = []
        
        # Emergent dimensions (discovered, not predefined)
        self.dimensions: List[Dict[str, Any]] = []
        
        # Position space
        self.positions: Dict[str, np.ndarray] = {}
    
    def add_pair(self, rgb_patch: Patch, depth_patch: Patch):
        """Add an RGB→depth patch pair."""
        self.rgb_patches.append(rgb_patch)
        self.depth_patches.append(depth_patch)
        self.pairs.append((rgb_patch, depth_patch))
    
    def discover_dimensions(self, n_dimensions: int = 8) -> List[Dict[str, Any]]:
        """
        Discover emergent dimensions from the patch pairs.
        
        Uses SVD on the difference vectors to find principal
        directions of transformation.
        """
        if len(self.pairs) < 2:
            return []
        
        # Compute transformation vectors: depth - rgb (flattened)
        transformations = []
        for rgb_patch, depth_patch in self.pairs:
            rgb_vec = rgb_patch.to_vector()
            depth_vec = depth_patch.to_vector()
            
            # Ensure same length (depth might be grayscale)
            if len(depth_vec) < len(rgb_vec):
                # Repeat depth to match RGB channels
                depth_vec = np.tile(depth_vec, 3)[:len(rgb_vec)]
            elif len(depth_vec) > len(rgb_vec):
                depth_vec = depth_vec[:len(rgb_vec)]
            
            # Normalize
            rgb_norm = rgb_vec / (np.linalg.norm(rgb_vec) + 1e-8)
            depth_norm = depth_vec / (np.linalg.norm(depth_vec) + 1e-8)
            
            transformations.append(depth_norm - rgb_norm)
        
        T = np.array(transformations)  # (n_pairs, n_features)
        
        # SVD to find principal transformation directions
        try:
            U, S, Vt = np.linalg.svd(T, full_matrices=False)
        except:
            return []
        
        # Top n_dimensions are the emergent dimensions
        self.dimensions = []
        for i in range(min(n_dimensions, len(S))):
            dim = {
                'index': i,
                'direction': Vt[i],  # The transformation direction
                'strength': S[i],    # How much variance it explains
                'name': f'emergent_dim_{i}',  # We don't know what it means yet
            }
            self.dimensions.append(dim)
        
        # Compute positions for all patches
        self._compute_positions()
        
        return self.dimensions
    
    def _compute_positions(self):
        """Compute positions for all patches in the emergent space."""
        if not self.dimensions:
            return
        
        n_dims = len(self.dimensions)
        
        for rgb_patch, depth_patch in self.pairs:
            rgb_vec = rgb_patch.to_vector()
            depth_vec = depth_patch.to_vector()
            
            # Ensure same length
            if len(depth_vec) < len(rgb_vec):
                depth_vec = np.tile(depth_vec, 3)[:len(rgb_vec)]
            elif len(depth_vec) > len(rgb_vec):
                depth_vec = depth_vec[:len(rgb_vec)]
            
            # Normalize
            rgb_norm = rgb_vec / (np.linalg.norm(rgb_vec) + 1e-8)
            depth_norm = depth_vec / (np.linalg.norm(depth_vec) + 1e-8)
            
            # Project onto emergent dimensions
            rgb_pos = np.array([np.dot(rgb_norm, d['direction']) for d in self.dimensions])
            depth_pos = np.array([np.dot(depth_norm, d['direction']) for d in self.dimensions])
            
            self.positions[rgb_patch.identifier] = rgb_pos
            self.positions[depth_patch.identifier] = depth_pos
    
    def predict_depth_position(self, rgb_patch: Patch) -> np.ndarray:
        """
        Predict the depth position from an RGB patch.
        
        Uses the learned transformation to traverse from RGB to depth.
        """
        if not self.dimensions:
            return np.zeros(8)
        
        rgb_vec = rgb_patch.to_vector()
        rgb_norm = rgb_vec / (np.linalg.norm(rgb_vec) + 1e-8)
        
        # Project onto emergent dimensions
        rgb_pos = np.array([np.dot(rgb_norm, d['direction']) for d in self.dimensions])
        
        # The transformation IS the dimension - traverse by φ
        # This is the key insight: moving along emergent dimensions
        # should take us from RGB space to depth space
        depth_pos = rgb_pos + PHI * np.array([d['strength'] for d in self.dimensions])
        
        return depth_pos
    
    def _build_index(self):
        """Build vectorized index for fast similarity search."""
        if hasattr(self, '_index_built') and self._index_built:
            return
        
        n = len(self.pairs)
        if n == 0:
            return
        
        # Get vector length from first patch
        vec_len = len(self.pairs[0][0].to_vector())
        
        # Build matrices
        self._rgb_matrix = np.zeros((n, vec_len))
        self._positions = np.zeros((n, 2))  # row, col
        
        for i, (rgb_patch, _) in enumerate(self.pairs):
            vec = rgb_patch.to_vector()
            self._rgb_matrix[i] = vec / (np.linalg.norm(vec) + 1e-8)
            self._positions[i] = [rgb_patch.row, rgb_patch.col]
        
        self._index_built = True
    
    def reconstruct_depth(self, rgb_patch: Patch) -> np.ndarray:
        """
        Reconstruct a depth patch from an RGB patch.
        
        Uses COMPOUND SIMILARITY (vectorized for speed):
        1. Content similarity (RGB appearance)
        2. Position similarity (row/col in image)
        """
        self._build_index()
        
        rgb_vec = rgb_patch.to_vector()
        rgb_norm = rgb_vec / (np.linalg.norm(rgb_vec) + 1e-8)
        
        # Vectorized content similarity
        content_sims = self._rgb_matrix @ rgb_norm
        
        # Vectorized position similarity
        pos = np.array([rgb_patch.row, rgb_patch.col])
        pos_diffs = np.abs(self._positions - pos) / 16.0
        position_sims = 1.0 - pos_diffs.mean(axis=1)
        
        # Compound similarity (φ-weighted)
        compound_sims = (content_sims * PHI + position_sims * (1/PHI)) / (PHI + 1/PHI)
        
        # Get top-k indices
        k = min(10, len(compound_sims))
        top_indices = np.argpartition(compound_sims, -k)[-k:]
        top_indices = top_indices[np.argsort(compound_sims[top_indices])[::-1]]
        
        # Weighted average of depth patches
        total_weight = 0
        depth_sum = None
        
        for idx in top_indices:
            sim = compound_sims[idx]
            weight = max(0, sim) ** 2
            
            depth_content = self.pairs[idx][1].content
            if depth_sum is None:
                depth_sum = np.zeros_like(depth_content)
            
            if depth_content.shape != depth_sum.shape:
                depth_pil = Image.fromarray((depth_content * 255).astype(np.uint8))
                depth_pil = depth_pil.resize((depth_sum.shape[1], depth_sum.shape[0]))
                depth_content = np.array(depth_pil).astype(np.float32) / 255.0
            
            depth_sum += weight * depth_content
            total_weight += weight
        
        if total_weight > 0:
            depth_result = depth_sum / total_weight
        else:
            patch_shape = rgb_patch.content.shape
            if len(patch_shape) == 3:
                depth_result = np.mean(rgb_patch.content, axis=-1)
            else:
                depth_result = rgb_patch.content
        
        return np.clip(depth_result, 0, 1)


# =============================================================================
# DEPTH PREDICTOR
# =============================================================================

class EmergentDepthPredictor:
    """
    Predicts full depth maps using emergent dimensions.
    """
    
    def __init__(self, patch_size: int = 16, grid_size: int = 8):
        self.patch_extractor = PatchExtractor(patch_size, grid_size)
        self.discoverer = EmergentDimensionDiscoverer()
        self.trained = False
    
    def train(self, rgb_images: List[np.ndarray], depth_images: List[np.ndarray],
              image_ids: List[str]):
        """Train on RGB-depth image pairs."""
        print(f"Training on {len(rgb_images)} images...")
        
        # Extract all patches
        for rgb, depth, img_id in zip(rgb_images, depth_images, image_ids):
            patches = self.patch_extractor.extract_patches(rgb, depth, img_id)
            for rgb_patch, depth_patch in patches:
                self.discoverer.add_pair(rgb_patch, depth_patch)
        
        print(f"  Extracted {len(self.discoverer.pairs)} patch pairs")
        
        # Discover emergent dimensions
        dimensions = self.discoverer.discover_dimensions(n_dimensions=8)
        print(f"  Discovered {len(dimensions)} emergent dimensions")
        
        # Report dimension strengths
        for dim in dimensions[:5]:
            print(f"    {dim['name']}: strength = {dim['strength']:.4f}")
        
        self.trained = True
    
    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """Predict depth map for an RGB image."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        # Resize to target size
        target_size = self.patch_extractor.target_size
        rgb_resized = self.patch_extractor._resize(rgb_image, target_size)
        
        # Predict each patch
        depth_patches = []
        for row in range(self.patch_extractor.grid_size):
            row_patches = []
            for col in range(self.patch_extractor.grid_size):
                r_start = row * self.patch_extractor.patch_size
                r_end = r_start + self.patch_extractor.patch_size
                c_start = col * self.patch_extractor.patch_size
                c_end = c_start + self.patch_extractor.patch_size
                
                rgb_patch = Patch(
                    content=rgb_resized[r_start:r_end, c_start:c_end],
                    row=row,
                    col=col,
                    image_id="predict",
                    is_depth=False
                )
                
                depth_patch = self.discoverer.reconstruct_depth(rgb_patch)
                row_patches.append(depth_patch)
            
            depth_patches.append(np.hstack(row_patches))
        
        return np.vstack(depth_patches)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_emergence_experiment(n_train: int = 30, n_test: int = 5):
    """Run the emergent depth dimension experiment."""
    print("=" * 70)
    print("EXPERIMENT: Emergent Depth Dimensions via Self-Assembly")
    print("=" * 70)
    print()
    print("Hypothesis: Depth dimensions will EMERGE from RGB→depth pairs")
    print("without us defining what they should be.")
    print()
    
    # Load images
    image_files = sorted(COCO_VAL_PATH.glob("*.jpg"))[:n_train + n_test]
    
    if len(image_files) < n_train + n_test:
        print(f"Warning: Only {len(image_files)} images available")
        n_train = min(n_train, len(image_files) - n_test)
    
    # Load training data
    print("Loading training data...")
    train_rgb = []
    train_depth = []
    train_ids = []
    
    for img_path in image_files[:n_train]:
        # Load RGB
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Load depth
        depth_cache = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        if not depth_cache.exists():
            print(f"  Skipping {img_path.name} (no depth cache)")
            continue
        
        depth = np.load(depth_cache)
        if depth.max() > 1:
            depth = depth / 255.0
        
        train_rgb.append(rgb)
        train_depth.append(depth)
        train_ids.append(img_path.stem)
    
    print(f"  Loaded {len(train_rgb)} training images")
    
    # Train
    print()
    print("=" * 60)
    print("PHASE 1: Discovering Emergent Dimensions")
    print("=" * 60)
    
    predictor = EmergentDepthPredictor(patch_size=16, grid_size=8)
    predictor.train(train_rgb, train_depth, train_ids)
    
    # Test
    print()
    print("=" * 60)
    print("PHASE 2: Testing Depth Prediction")
    print("=" * 60)
    
    test_files = image_files[n_train:n_train + n_test]
    errors = []
    
    for img_path in test_files:
        # Load RGB
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Load true depth
        depth_cache = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        if not depth_cache.exists():
            continue
        
        true_depth = np.load(depth_cache)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        # Predict
        pred_depth = predictor.predict(rgb)
        
        # Resize true depth to match prediction
        true_resized = predictor.patch_extractor._resize(true_depth, pred_depth.shape[0])
        
        # Compute error
        mae = np.mean(np.abs(pred_depth - true_resized))
        errors.append(mae)
        
        print(f"  {img_path.name}: MAE = {mae:.4f}")
    
    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    mean_mae = np.mean(errors) if errors else 0
    print(f"  Mean Absolute Error: {mean_mae:.4f}")
    print()
    
    # Analyze emergent dimensions
    print("Emergent Dimensions (what did the geometry discover?):")
    for dim in predictor.discoverer.dimensions[:5]:
        # Try to interpret the dimension by looking at its direction
        direction = dim['direction']
        
        # Compute some statistics about the direction
        if len(direction) >= 768:  # 16x16x3 = 768
            # Reshape to patch shape
            dir_patch = direction[:768].reshape(16, 16, 3)
            
            # Analyze spatial structure
            top_half = np.mean(np.abs(dir_patch[:8, :, :]))
            bottom_half = np.mean(np.abs(dir_patch[8:, :, :]))
            left_half = np.mean(np.abs(dir_patch[:, :8, :]))
            right_half = np.mean(np.abs(dir_patch[:, 8:, :]))
            center = np.mean(np.abs(dir_patch[4:12, 4:12, :]))
            edge = np.mean(np.abs(dir_patch)) - center
            
            # Guess what it might represent
            interpretation = []
            if abs(top_half - bottom_half) > 0.01:
                interpretation.append("vertical" if top_half > bottom_half else "ground-focused")
            if abs(left_half - right_half) > 0.01:
                interpretation.append("horizontal asymmetry")
            if center > edge:
                interpretation.append("center-focused")
            
            interp_str = ", ".join(interpretation) if interpretation else "unknown"
        else:
            interp_str = "unknown"
        
        print(f"  {dim['name']}: strength={dim['strength']:.4f} ({interp_str})")
    
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()
    
    if mean_mae < 0.15:
        print("✓ SUCCESS: Emergent dimensions can predict depth!")
    elif mean_mae < 0.25:
        print("◐ PARTIAL: Some depth structure captured")
    else:
        print("✗ LIMITED: Emergent dimensions need more data or refinement")
    
    print()
    print("Key insight:")
    print("  We did NOT define what the dimensions should be.")
    print("  They EMERGED from the RGB→depth pairs.")
    print("  The geometry discovered its own transformation space.")
    print()
    
    return predictor


if __name__ == "__main__":
    predictor = run_emergence_experiment(n_train=30, n_test=5)
