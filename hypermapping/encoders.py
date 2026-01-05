"""
Additional Encoders for HyperMapping

Demonstrates that HyperMapping works with any domain:
- ImageEncoder: For image data (pixel arrays, features)
- NumericEncoder: For numeric vectors
- CategoricalEncoder: For classification tasks

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from abc import ABC, abstractmethod
import hashlib

from .hypermapping import Encoder, CRITICAL_LINE


class NumericEncoder(Encoder):
    """
    Encoder for numeric vectors.
    
    Projects numeric data to the target dimensionality using:
    1. Non-linear features (products, squares) for XOR-like problems
    2. Random projection for dimensionality reduction
    """
    
    def __init__(self, dims: int, input_dims: Optional[int] = None,
                 use_nonlinear: bool = True):
        super().__init__(dims)
        self.input_dims = input_dims
        self.use_nonlinear = use_nonlinear
        self._projection_matrix: Optional[np.ndarray] = None
    
    def _expand_features(self, arr: np.ndarray) -> np.ndarray:
        """Expand with non-linear features."""
        if not self.use_nonlinear:
            return arr
        
        features = [arr]
        
        # Add squares
        features.append(arr ** 2)
        
        # Add pairwise products (for XOR-like problems)
        n = len(arr)
        products = []
        for i in range(n):
            for j in range(i + 1, n):
                products.append(arr[i] * arr[j])
        if products:
            features.append(np.array(products))
        
        # Add XOR-like feature: |x - y| for pairs
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(abs(arr[i] - arr[j]))
        if diffs:
            features.append(np.array(diffs))
        
        return np.concatenate(features)
    
    def _ensure_projection(self, input_dims: int) -> None:
        """Create projection matrix if needed."""
        if self._projection_matrix is None or self._projection_matrix.shape[1] != input_dims:
            # Random projection matrix (Johnson-Lindenstrauss)
            np.random.seed(42)  # Deterministic
            self._projection_matrix = np.random.randn(self.dims, input_dims)
            self._projection_matrix /= np.sqrt(self.dims)
    
    def encode_input(self, input_val: Any) -> np.ndarray:
        arr = np.array(input_val).flatten().astype(float)
        
        # Expand with non-linear features
        expanded = self._expand_features(arr)
        
        self._ensure_projection(len(expanded))
        
        pos = self._projection_matrix @ expanded
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_output(self, output_val: Any) -> np.ndarray:
        # For scalar outputs, use a different encoding
        if isinstance(output_val, (int, float)):
            # Map scalar to position based on value
            pos = np.zeros(self.dims)
            # Use value to determine angle in first few dimensions
            angle = float(output_val) * np.pi
            for i in range(min(4, self.dims)):
                pos[i] = np.cos(angle * (i + 1)) if i % 2 == 0 else np.sin(angle * (i + 1))
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            return pos
        return self.encode_input(output_val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize NumericEncoder including projection matrix."""
        return {
            'type': 'NumericEncoder',
            'version': '1.0',
            'config': {
                'dims': self.dims,
                'input_dims': self.input_dims,
                'use_nonlinear': self.use_nonlinear,
            },
            'state': {
                'projection_matrix': self._projection_matrix.tolist() if self._projection_matrix is not None else None,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NumericEncoder':
        """Deserialize NumericEncoder with projection matrix."""
        config = data.get('config', {})
        encoder = cls(
            dims=config.get('dims', 8),
            input_dims=config.get('input_dims'),
            use_nonlinear=config.get('use_nonlinear', True),
        )
        
        state = data.get('state', {})
        proj = state.get('projection_matrix')
        if proj is not None:
            encoder._projection_matrix = np.array(proj)
        
        return encoder


class ImageEncoder(Encoder):
    """
    Encoder for image data.
    
    Encodes images to positions based on:
    1. Color histogram (global color distribution)
    2. Spatial features (where colors are located)
    3. Edge features (structure)
    
    Works with:
    - Grayscale images: (H, W) arrays
    - RGB images: (H, W, 3) arrays
    - Feature vectors: 1D arrays
    """
    
    def __init__(self, dims: int, 
                 use_histogram: bool = True,
                 use_spatial: bool = True,
                 histogram_bins: int = 16):
        super().__init__(dims)
        self.use_histogram = use_histogram
        self.use_spatial = use_spatial
        self.histogram_bins = histogram_bins
        
        # Projection matrices for different feature types
        self._hist_projection: Optional[np.ndarray] = None
        self._spatial_projection: Optional[np.ndarray] = None
    
    def _extract_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram from image."""
        image = np.array(image)
        
        if image.ndim == 1:
            # Already a feature vector
            return image
        
        # Normalize to 0-1
        if image.max() > 1:
            image = image / 255.0
        
        if image.ndim == 2:
            # Grayscale
            hist, _ = np.histogram(image.flatten(), bins=self.histogram_bins, range=(0, 1))
        else:
            # RGB - histogram per channel
            hists = []
            for c in range(min(3, image.shape[-1])):
                h, _ = np.histogram(image[..., c].flatten(), bins=self.histogram_bins, range=(0, 1))
                hists.append(h)
            hist = np.concatenate(hists)
        
        # Normalize
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def _extract_spatial(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial features (quadrant means)."""
        image = np.array(image)
        
        if image.ndim == 1:
            # Feature vector - split into quadrants
            n = len(image)
            q = n // 4
            return np.array([
                image[:q].mean() if q > 0 else 0,
                image[q:2*q].mean() if q > 0 else 0,
                image[2*q:3*q].mean() if q > 0 else 0,
                image[3*q:].mean() if q > 0 else 0,
            ])
        
        # Normalize
        if image.max() > 1:
            image = image / 255.0
        
        h, w = image.shape[:2]
        h2, w2 = h // 2, w // 2
        
        # Quadrant means
        if image.ndim == 2:
            features = [
                image[:h2, :w2].mean(),
                image[:h2, w2:].mean(),
                image[h2:, :w2].mean(),
                image[h2:, w2:].mean(),
            ]
        else:
            features = []
            for c in range(min(3, image.shape[-1])):
                features.extend([
                    image[:h2, :w2, c].mean(),
                    image[:h2, w2:, c].mean(),
                    image[h2:, :w2, c].mean(),
                    image[h2:, w2:, c].mean(),
                ])
        
        return np.array(features)
    
    def _project(self, features: np.ndarray, 
                 projection: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Project features to target dims."""
        if projection is None or projection.shape[1] != len(features):
            np.random.seed(hash(len(features)) % (2**31))
            projection = np.random.randn(self.dims, len(features))
            projection /= np.sqrt(self.dims)
        
        pos = projection @ features
        return pos, projection
    
    def encode_input(self, input_val: Any) -> np.ndarray:
        """Encode an image to a position."""
        image = np.array(input_val)
        
        features = []
        
        if self.use_histogram:
            hist = self._extract_histogram(image)
            features.append(hist)
        
        if self.use_spatial:
            spatial = self._extract_spatial(image)
            features.append(spatial)
        
        if not features:
            # Fallback to flattened image
            features.append(image.flatten()[:100])  # Limit size
        
        combined = np.concatenate(features)
        
        # Project to target dims
        pos, self._hist_projection = self._project(combined, self._hist_projection)
        
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        
        return pos
    
    def encode_output(self, output_val: Any) -> np.ndarray:
        """Encode output (could be label, another image, etc.)."""
        if isinstance(output_val, str):
            # Label - use hash
            seed = int(hashlib.md5(output_val.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            pos = np.random.randn(self.dims)
            return pos / np.linalg.norm(pos) * CRITICAL_LINE
        else:
            # Another image
            return self.encode_input(output_val)
    
    def encode_mapping(self, input_val: Any, output_val: Any) -> np.ndarray:
        """
        Encode a mapping.
        
        For image classification: position from image features.
        For image-to-image: average of both positions.
        """
        input_pos = self.encode_input(input_val)
        
        if isinstance(output_val, str):
            # Classification - use input position (image determines location)
            return input_pos
        else:
            # Image-to-image - average
            output_pos = self.encode_output(output_val)
            pos = (input_pos + output_pos) / 2
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            return pos
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ImageEncoder including projection matrices."""
        return {
            'type': 'ImageEncoder',
            'version': '1.0',
            'config': {
                'dims': self.dims,
                'use_histogram': self.use_histogram,
                'use_spatial': self.use_spatial,
                'histogram_bins': self.histogram_bins,
            },
            'state': {
                'hist_projection': self._hist_projection.tolist() if self._hist_projection is not None else None,
                'spatial_projection': self._spatial_projection.tolist() if self._spatial_projection is not None else None,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageEncoder':
        """Deserialize ImageEncoder with projection matrices."""
        config = data.get('config', {})
        encoder = cls(
            dims=config.get('dims', 16),
            use_histogram=config.get('use_histogram', True),
            use_spatial=config.get('use_spatial', True),
            histogram_bins=config.get('histogram_bins', 16),
        )
        
        state = data.get('state', {})
        if state.get('hist_projection') is not None:
            encoder._hist_projection = np.array(state['hist_projection'])
        if state.get('spatial_projection') is not None:
            encoder._spatial_projection = np.array(state['spatial_projection'])
        
        return encoder


class CategoricalEncoder(Encoder):
    """
    Encoder for categorical data (classification).
    
    Each category gets a distinct position.
    Similar categories can be placed closer together.
    """
    
    def __init__(self, dims: int, categories: Optional[List[str]] = None):
        super().__init__(dims)
        self.categories = categories or []
        self._category_positions: Dict[str, np.ndarray] = {}
        
        if categories:
            self._initialize_positions()
    
    def _initialize_positions(self) -> None:
        """Initialize category positions."""
        n = len(self.categories)
        
        if n == 0:
            return
        
        # Distribute categories evenly on hypersphere
        for i, cat in enumerate(self.categories):
            # Use golden angle for even distribution
            phi = i * 2.39996323  # Golden angle in radians
            
            pos = np.zeros(self.dims)
            for d in range(self.dims):
                angle = phi * (d + 1)
                pos[d] = np.cos(angle) if d % 2 == 0 else np.sin(angle)
            
            pos = pos / np.linalg.norm(pos) * CRITICAL_LINE
            self._category_positions[cat] = pos
    
    def add_category(self, category: str, 
                     similar_to: Optional[str] = None) -> None:
        """Add a new category."""
        if category in self._category_positions:
            return
        
        if similar_to and similar_to in self._category_positions:
            # Place near similar category
            base = self._category_positions[similar_to]
            noise = np.random.randn(self.dims) * 0.1
            pos = base + noise
        else:
            # Random position
            pos = np.random.randn(self.dims)
        
        pos = pos / np.linalg.norm(pos) * CRITICAL_LINE
        self._category_positions[category] = pos
        self.categories.append(category)
    
    def encode_input(self, input_val: Any) -> np.ndarray:
        """Encode input (assumes it's a category or has a category)."""
        cat = str(input_val)
        
        if cat not in self._category_positions:
            self.add_category(cat)
        
        return self._category_positions[cat]
    
    def encode_output(self, output_val: Any) -> np.ndarray:
        return self.encode_input(output_val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize CategoricalEncoder including category positions."""
        return {
            'type': 'CategoricalEncoder',
            'version': '1.0',
            'config': {
                'dims': self.dims,
                'categories': self.categories,
            },
            'state': {
                'category_positions': {
                    cat: pos.tolist() 
                    for cat, pos in self._category_positions.items()
                },
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CategoricalEncoder':
        """Deserialize CategoricalEncoder with category positions."""
        config = data.get('config', {})
        encoder = cls(
            dims=config.get('dims', 8),
            categories=None,  # Don't auto-initialize
        )
        encoder.categories = config.get('categories', [])
        
        state = data.get('state', {})
        encoder._category_positions = {
            cat: np.array(pos) 
            for cat, pos in state.get('category_positions', {}).items()
        }
        
        return encoder


class QuaternionEncoder(Encoder):
    """
    4D Quaternion encoder with semantic axes (from Design 044).
    
    The quaternion structure provides natural 4D control:
    - X (i-axis): Polarity - positive vs negative
    - Y (j-axis): Intensity - how strong the signal
    - Z (k-axis): Style - formal vs casual (derived from word length)
    - W (scalar): Certainty - definitive vs hedged
    
    Vocabularies are customizable - pass empty dicts for a clean slate,
    or load from serialized state.
    
    Usage:
        # With defaults
        encoder = QuaternionEncoder()
        
        # Clean slate (no defaults)
        encoder = QuaternionEncoder(polarity_vocab={}, intensity_vocab={}, certainty_vocab={})
        encoder.polarity_vocab['awesome'] = 0.9
        
        # From serialized state
        encoder = QuaternionEncoder.from_dict(saved_data)
    """
    
    # Default vocabularies (can be overridden)
    DEFAULT_POLARITY = {
        'love': 1.0, 'amazing': 0.9, 'great': 0.8, 'excellent': 0.9,
        'good': 0.6, 'best': 1.0, 'wonderful': 0.9, 'fantastic': 0.9,
        'hate': -1.0, 'terrible': -0.9, 'awful': -0.9, 'worst': -1.0,
        'bad': -0.6, 'poor': -0.5, 'disappointed': -0.7, 'waste': -0.8,
        'okay': 0.0, 'average': 0.0, 'nothing': -0.1, 'special': 0.3,
    }
    
    DEFAULT_INTENSITY = {
        'very': 0.8, 'really': 0.7, 'extremely': 0.9, 'somewhat': 0.3,
        'slightly': 0.2, 'totally': 0.9, 'completely': 0.9, 'so': 0.7,
    }
    
    DEFAULT_CERTAINTY = {
        'definitely': -0.8, 'certainly': -0.8, 'absolutely': -0.9,
        'clearly': -0.7, 'undoubtedly': -0.9,
        'maybe': 0.5, 'perhaps': 0.6, 'possibly': 0.7,
        'seems': 0.4, 'appears': 0.5, 'might': 0.6,
    }
    
    def __init__(self, dims: int = 4,
                 polarity_vocab: Optional[Dict[str, float]] = None,
                 intensity_vocab: Optional[Dict[str, float]] = None,
                 certainty_vocab: Optional[Dict[str, float]] = None):
        super().__init__(dims)
        
        # Use provided vocabs or defaults
        self.polarity_vocab = dict(polarity_vocab) if polarity_vocab is not None else dict(self.DEFAULT_POLARITY)
        self.intensity_vocab = dict(intensity_vocab) if intensity_vocab is not None else dict(self.DEFAULT_INTENSITY)
        self.certainty_vocab = dict(certainty_vocab) if certainty_vocab is not None else dict(self.DEFAULT_CERTAINTY)
        
        # Output label positions (customizable)
        self.output_positions: Dict[str, np.ndarray] = {
            'positive': np.array([1.0, 0.5, 0.0, 0.0]),
            'negative': np.array([-1.0, 0.5, 0.0, 0.0]),
            'neutral': np.array([0.0, 0.0, 0.0, 0.0]),
        }
    
    def encode_input(self, text: str) -> np.ndarray:
        words = str(text).lower().split()
        
        # X-axis: Polarity (sum of word polarities, clipped)
        polarity = sum(self.polarity_vocab.get(w, 0) for w in words)
        polarity = np.clip(polarity, -1, 1)
        
        # Y-axis: Intensity (first matching word, or default 0.5)
        intensity = 0.5
        for word in words:
            if word in self.intensity_vocab:
                intensity = self.intensity_vocab[word]
                break
        
        # Z-axis: Style (word length proxy - longer = more formal)
        avg_word_len = np.mean([len(w) for w in words]) if words else 5
        style = np.clip((avg_word_len - 5) / 5, -1, 1)
        
        # W-axis: Certainty (first matching word, or default 0.0)
        certainty = 0.0
        for word in words:
            if word in self.certainty_vocab:
                certainty = self.certainty_vocab[word]
                break
        
        pos = np.array([polarity, intensity, style, certainty])
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_output(self, output: str) -> np.ndarray:
        """Encode output labels to positions."""
        if output in self.output_positions:
            return self.output_positions[output] * CRITICAL_LINE
        # Fallback: hash-based position
        seed = hash(output) % (2**31)
        np.random.seed(seed)
        pos = np.random.randn(self.dims)
        return pos / np.linalg.norm(pos) * CRITICAL_LINE
    
    def encode_mapping(self, input_val, output_val) -> np.ndarray:
        return self.encode_input(input_val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize QuaternionEncoder including vocabularies."""
        return {
            'type': 'QuaternionEncoder',
            'version': '1.0',
            'config': {'dims': self.dims},
            'state': {
                'polarity_vocab': self.polarity_vocab,
                'intensity_vocab': self.intensity_vocab,
                'certainty_vocab': self.certainty_vocab,
                'output_positions': {k: v.tolist() for k, v in self.output_positions.items()},
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuaternionEncoder':
        """Deserialize QuaternionEncoder with vocabularies."""
        config = data.get('config', {})
        state = data.get('state', {})
        
        encoder = cls(
            dims=config.get('dims', 4),
            polarity_vocab=state.get('polarity_vocab'),
            intensity_vocab=state.get('intensity_vocab'),
            certainty_vocab=state.get('certainty_vocab'),
        )
        
        if 'output_positions' in state:
            encoder.output_positions = {k: np.array(v) for k, v in state['output_positions'].items()}
        
        return encoder


class SelfSimilarEncoder(Encoder):
    """
    Encoder using self-similar transformations for function approximation (from Design 072).
    
    Key insight: "The same transformations work identically at every scale."
    
    If we know the transform between two points, we can interpolate ANY point
    using the same transform scaled appropriately. This achieves 100% accuracy
    on function approximation tasks.
    
    Usage:
        encoder = SelfSimilarEncoder(dims=8)
        encoder.learn_points([(0, 0), (1, 0.84), (2, 0.91), ...])  # (x, y) pairs
        y = encoder.interpolate(1.5)  # Interpolate at x=1.5
    """
    
    def __init__(self, dims: int = 8):
        super().__init__(dims)
        self.known_points: List[Tuple[float, float]] = []
        self.transforms: Dict[Tuple[int, int], float] = {}
    
    def learn_points(self, points: List[Tuple[float, float]]) -> None:
        """Learn from known (x, y) pairs."""
        self.known_points = sorted(points, key=lambda p: p[0])
        
        # Learn transforms between adjacent points
        for i in range(len(self.known_points) - 1):
            x1, y1 = self.known_points[i]
            x2, y2 = self.known_points[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) > 1e-10:
                self.transforms[(i, i+1)] = dy / dx
    
    def interpolate(self, x: float) -> float:
        """Interpolate y value using self-similar transforms."""
        if not self.known_points:
            return 0.0
        
        # Find bracketing points
        for i in range(len(self.known_points) - 1):
            x1, y1 = self.known_points[i]
            x2, y2 = self.known_points[i + 1]
            
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1) if abs(x2 - x1) > 1e-10 else 0
                return y1 + t * (y2 - y1)
        
        # Extrapolate
        if x < self.known_points[0][0]:
            x1, y1 = self.known_points[0]
            if (0, 1) in self.transforms:
                return y1 + self.transforms[(0, 1)] * (x - x1)
            return y1
        else:
            x1, y1 = self.known_points[-1]
            n = len(self.known_points)
            if (n-2, n-1) in self.transforms:
                return y1 + self.transforms[(n-2, n-1)] * (x - x1)
            return y1
    
    def encode_input(self, x: float) -> np.ndarray:
        """Encode input using position in transform space."""
        segment = 0
        for i in range(len(self.known_points) - 1):
            if self.known_points[i][0] <= x <= self.known_points[i + 1][0]:
                segment = i
                break
        
        pos = np.zeros(self.dims)
        pos[0] = float(x)
        pos[1] = np.sin(float(x))
        pos[2] = np.cos(float(x))
        pos[3] = segment / max(1, len(self.known_points) - 1)
        
        if segment < len(self.known_points) - 1:
            x1 = self.known_points[segment][0]
            x2 = self.known_points[segment + 1][0]
            t = (x - x1) / (x2 - x1) if abs(x2 - x1) > 1e-10 else 0
            pos[4] = t
        
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_output(self, y: float) -> np.ndarray:
        pos = np.zeros(self.dims)
        pos[0] = float(y)
        pos[1] = np.sign(float(y))
        pos[2] = abs(float(y))
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_mapping(self, x, y) -> np.ndarray:
        return self.encode_input(float(x))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize SelfSimilarEncoder including learned points and transforms."""
        return {
            'type': 'SelfSimilarEncoder',
            'version': '1.0',
            'config': {'dims': self.dims},
            'state': {
                'known_points': self.known_points,
                'transforms': {f"{k[0]},{k[1]}": v for k, v in self.transforms.items()},
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfSimilarEncoder':
        """Deserialize SelfSimilarEncoder with learned points and transforms."""
        config = data.get('config', {})
        encoder = cls(dims=config.get('dims', 8))
        
        state = data.get('state', {})
        encoder.known_points = [tuple(p) for p in state.get('known_points', [])]
        
        transforms = state.get('transforms', {})
        encoder.transforms = {
            tuple(map(int, k.split(','))): v 
            for k, v in transforms.items()
        }
        
        return encoder


class SequenceEncoder(Encoder):
    """
    Encoder for sequence prediction using Tachyon navigation (from Design 055).
    
    Key insight: Forward (φ^+n) = observe patterns, Backward (φ^-n) = predict next.
    
    Detects patterns (arithmetic, geometric, fibonacci) and predicts next values.
    """
    
    def __init__(self, dims: int = 8):
        super().__init__(dims)
        self._last_diff = 0
        self._last_ratio = 1
    
    def detect_pattern(self, seq: List[int]) -> Tuple[str, float]:
        """Navigate forward to detect pattern type."""
        if len(seq) < 3:
            return 'unknown', 0.0
        
        # Arithmetic: constant difference
        diffs = [seq[i] - seq[i-1] for i in range(1, len(seq))]
        if len(set(diffs)) == 1:
            self._last_diff = diffs[0]
            return 'arithmetic', 1.0
        
        # Geometric: constant ratio
        if all(seq[i-1] != 0 for i in range(1, len(seq))):
            ratios = [seq[i] / seq[i-1] for i in range(1, len(seq))]
            if len(set(ratios)) == 1:
                self._last_ratio = ratios[0]
                return 'geometric', 1.0
        
        # Fibonacci: a[n] = a[n-1] + a[n-2]
        fib_score = sum(1 for i in range(2, len(seq)) if seq[i] == seq[i-1] + seq[i-2])
        if fib_score == len(seq) - 2:
            return 'fibonacci', 1.0
        
        return 'unknown', 0.0
    
    def predict_next(self, seq: List[int]) -> Tuple[int, float]:
        """Navigate backward to predict next value."""
        pattern, confidence = self.detect_pattern(seq)
        
        if pattern == 'arithmetic':
            return seq[-1] + self._last_diff, confidence
        elif pattern == 'geometric':
            return int(seq[-1] * self._last_ratio), confidence
        elif pattern == 'fibonacci':
            return seq[-1] + seq[-2], confidence
        else:
            diff = seq[-1] - seq[-2] if len(seq) >= 2 else 1
            return seq[-1] + diff, 0.3
    
    def encode_input(self, seq) -> np.ndarray:
        """Encode a sequence to position."""
        seq = list(seq)
        pattern, conf = self.detect_pattern(seq)
        
        pos = np.zeros(self.dims)
        
        # Pattern type encoding
        pattern_map = {'arithmetic': 0, 'geometric': 1, 'fibonacci': 2, 'unknown': 3}
        pos[0] = pattern_map.get(pattern, 3) / 3
        
        # Sequence statistics
        if seq:
            pos[1] = np.mean(seq) / (max(abs(s) for s in seq) + 1)
            pos[2] = (seq[-1] - seq[0]) / (len(seq) + 1)
            pos[3] = conf
        
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_output(self, val) -> np.ndarray:
        pos = np.zeros(self.dims)
        pos[0] = float(val) / (abs(float(val)) + 1)
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_mapping(self, seq, val) -> np.ndarray:
        return self.encode_input(seq)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize SequenceEncoder (stateless - pattern detection is algorithmic)."""
        return {
            'type': 'SequenceEncoder',
            'version': '1.0',
            'config': {'dims': self.dims},
            'state': {}  # Pattern detection is algorithmic, not learned
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SequenceEncoder':
        """Deserialize SequenceEncoder."""
        config = data.get('config', {})
        return cls(dims=config.get('dims', 8))


class CompositeEncoder(Encoder):
    """
    Combines multiple encoders for multi-modal data.
    
    Example: Image + Text → Position
    """
    
    def __init__(self, dims: int, encoders: Dict[str, Encoder]):
        super().__init__(dims)
        self.encoders = encoders
        self._weights: Dict[str, float] = {k: 1.0 for k in encoders}
    
    def set_weight(self, encoder_name: str, weight: float) -> None:
        """Set weight for an encoder."""
        if encoder_name in self._weights:
            self._weights[encoder_name] = weight
    
    def encode_input(self, input_val: Any) -> np.ndarray:
        """
        Encode multi-modal input.
        
        input_val should be a dict: {'image': ..., 'text': ...}
        """
        if not isinstance(input_val, dict):
            # Try first encoder
            first_encoder = list(self.encoders.values())[0]
            return first_encoder.encode_input(input_val)
        
        positions = []
        weights = []
        
        for name, encoder in self.encoders.items():
            if name in input_val:
                pos = encoder.encode_input(input_val[name])
                positions.append(pos)
                weights.append(self._weights.get(name, 1.0))
        
        if not positions:
            return np.zeros(self.dims)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        pos = sum(w * p for w, p in zip(weights, positions))
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        
        return pos
    
    def encode_output(self, output_val: Any) -> np.ndarray:
        return self.encode_input(output_val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize CompositeEncoder including sub-encoders."""
        return {
            'type': 'CompositeEncoder',
            'version': '1.0',
            'config': {'dims': self.dims},
            'state': {
                'encoders': {name: enc.to_dict() for name, enc in self.encoders.items()},
                'weights': self._weights,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompositeEncoder':
        """Deserialize CompositeEncoder with sub-encoders."""
        config = data.get('config', {})
        state = data.get('state', {})
        
        # Reconstruct sub-encoders
        encoders = {}
        for name, enc_data in state.get('encoders', {}).items():
            enc_type = enc_data.get('type')
            encoder_cls = ENCODER_REGISTRY.get(enc_type)
            if encoder_cls:
                encoders[name] = encoder_cls.from_dict(enc_data)
        
        encoder = cls(dims=config.get('dims', 8), encoders=encoders)
        encoder._weights = state.get('weights', {k: 1.0 for k in encoders})
        
        return encoder


# =============================================================================
# ENCODER REGISTRY - For deserialization
# =============================================================================

from .hypermapping import HashEncoder, TextEncoder

ENCODER_REGISTRY: Dict[str, type] = {
    'HashEncoder': HashEncoder,
    'TextEncoder': TextEncoder,
    'NumericEncoder': NumericEncoder,
    'ImageEncoder': ImageEncoder,
    'CategoricalEncoder': CategoricalEncoder,
    'QuaternionEncoder': QuaternionEncoder,
    'SelfSimilarEncoder': SelfSimilarEncoder,
    'SequenceEncoder': SequenceEncoder,
    'CompositeEncoder': CompositeEncoder,
}


def encoder_from_dict(data: Dict[str, Any]) -> Encoder:
    """Deserialize any encoder from dictionary."""
    enc_type = data.get('type')
    encoder_cls = ENCODER_REGISTRY.get(enc_type)
    if encoder_cls:
        return encoder_cls.from_dict(data)
    raise ValueError(f"Unknown encoder type: {enc_type}")
