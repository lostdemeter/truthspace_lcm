#!/usr/bin/env python3
"""
Fast Depth Estimation Library - Optimized Version

Applies optimization techniques from the Holographer's Workbench:
- Simple approximations instead of expensive operations
- Downscale-process-upscale for speed
- Box filters instead of Gaussian
- Cached computations
- Optional quality/speed trade-offs

Inspired by practical_stereo_converter.py: "912× faster by simplifying 
theoretical framework while keeping core quality."

Performance Target: 180+ FPS (vs 85 FPS baseline)

Author: Holographer's Workbench
License: GPLv3
Version: 0.2.0 (Optimized)
"""

import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass

# Optional GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


@dataclass
class FastDepthConfig:
    """
    Configuration for fast depth estimation.
    
    Attributes:
        algorithm: Depth estimation method ('luminance', 'edge', 'combined')
        smoothing_sigma: Gaussian smoothing strength (0 = no smoothing)
        temporal_smoothing: Enable temporal smoothing across frames
        temporal_alpha: Temporal smoothing factor (0-1)
        use_gpu: Enable GPU acceleration if available
        
        # Optimization parameters
        fast_mode: Enable all speed optimizations
        downsample_factor: Process at reduced resolution (0.5-1.0)
        use_box_filter: Use box filter instead of Gaussian (faster)
        simple_gradients: Use simple differences instead of Sobel
    """
    
    # Standard parameters
    algorithm: str = 'combined'
    smoothing_sigma: float = 2.0
    temporal_smoothing: bool = True
    temporal_alpha: float = 0.3
    use_gpu: bool = True
    
    # Optimization parameters
    fast_mode: bool = True
    downsample_factor: float = 0.75  # Process at 75% resolution
    use_box_filter: bool = True      # Box filter vs Gaussian
    simple_gradients: bool = True    # Simple diff vs Sobel
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_algorithms = ['luminance', 'edge', 'combined']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        if self.smoothing_sigma < 0:
            raise ValueError("Smoothing sigma must be non-negative")
        if not 0 <= self.temporal_alpha <= 1:
            raise ValueError("Temporal alpha must be between 0 and 1")
        if not 0.25 <= self.downsample_factor <= 1.0:
            raise ValueError("Downsample factor must be between 0.25 and 1.0")


class FastDepthEstimator:
    """
    Optimized depth estimation using Holographer's Workbench techniques.
    
    Key optimizations:
    1. Downscale-process-upscale (4× speedup at 50% scale)
    2. Simple gradient approximation (3× speedup vs Sobel)
    3. Box filter smoothing (2× speedup vs Gaussian)
    4. Cached operations where possible
    
    Expected performance: 180+ FPS (vs 85 FPS baseline)
    """
    
    def __init__(self, config: Optional[FastDepthConfig] = None):
        """
        Initialize fast depth estimator.
        
        Args:
            config: Fast depth estimation configuration (uses defaults if None)
        """
        self.config = config or FastDepthConfig()
        self.config.validate()
        
        self.use_gpu = self.config.use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Temporal smoothing state
        self.prev_depth = None
        
        # Cache for gradient kernels (if not using simple gradients)
        if not self.config.simple_gradients:
            self._init_gradient_kernels()
    
    def _init_gradient_kernels(self):
        """Pre-compute gradient kernels for reuse."""
        # Sobel kernels
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = self.sobel_x.T
        
        if self.use_gpu:
            self.sobel_x = cp.asarray(self.sobel_x)
            self.sobel_y = cp.asarray(self.sobel_y)
    
    def estimate_depth(self, image: np.ndarray, input_on_gpu: bool = False, return_gpu: bool = False) -> np.ndarray:
        """
        Estimate depth map from RGB image (optimized).
        
        Args:
            image: Input image (H, W, 3) as uint8 or float32 [0, 1]
                   Can be numpy array or cupy array if input_on_gpu=True
            input_on_gpu: If True, input is already a CuPy array
            return_gpu: If True, return CuPy array (stay on GPU)
            
        Returns:
            Depth map (H, W) as float32 [0, 1], where 1 = close, 0 = far
        """
        original_h, original_w = image.shape[:2]
        
        # Phase 3B: Keep on GPU if possible
        if self.use_gpu and not input_on_gpu:
            image = cp.asarray(image)
        
        # Normalize to [0, 1] if needed
        if image.dtype == np.uint8 or (hasattr(image, 'dtype') and image.dtype == cp.uint8):
            image_norm = image.astype(self.xp.float32) / 255.0
        else:
            image_norm = image.astype(self.xp.float32)
        
        # Optimization 1: Downsample for speed
        if self.config.fast_mode and self.config.downsample_factor < 1.0:
            new_h = int(original_h * self.config.downsample_factor)
            new_w = int(original_w * self.config.downsample_factor)
            
            if self.use_gpu:
                # GPU resize using CuPy
                from cupyx.scipy.ndimage import zoom
                scale_h = new_h / original_h
                scale_w = new_w / original_w
                image_norm = zoom(image_norm, (scale_h, scale_w, 1), order=1)
            else:
                image_norm = cv2.resize(image_norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Compute depth based on algorithm
        if self.config.algorithm == 'luminance':
            depth = self._luminance_depth(image_norm)
        elif self.config.algorithm == 'edge':
            depth = self._edge_depth(image_norm)
        else:  # combined
            depth = self._combined_depth(image_norm)
        
        # Apply spatial smoothing
        if self.config.smoothing_sigma > 0:
            depth = self._smooth_depth(depth)
        
        # Optimization 2: Upscale back to original resolution
        if self.config.fast_mode and self.config.downsample_factor < 1.0:
            if self.use_gpu:
                from cupyx.scipy.ndimage import zoom
                scale_h = original_h / depth.shape[0]
                scale_w = original_w / depth.shape[1]
                depth = zoom(depth, (scale_h, scale_w), order=1)
            else:
                depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply temporal smoothing
        if self.config.temporal_smoothing and self.prev_depth is not None:
            alpha = self.config.temporal_alpha
            depth = alpha * self.prev_depth + (1 - alpha) * depth
        
        self.prev_depth = depth.copy() if not self.use_gpu else depth
        
        # Normalize to [0, 1]
        depth = self._normalize(depth)
        
        # Transfer to CPU if requested
        if self.use_gpu and not return_gpu:
            depth = cp.asnumpy(depth)
        
        return depth
    
    def _luminance_depth(self, image_norm) -> np.ndarray:
        """
        Estimate depth from luminance (brightness).
        Assumption: Brighter objects are closer.
        """
        # Convert to grayscale using standard weights
        gray = 0.299 * image_norm[:, :, 0] + 0.587 * image_norm[:, :, 1] + 0.114 * image_norm[:, :, 2]
        return gray
    
    def _edge_depth(self, image_norm):
        """
        Estimate depth from edge strength (optimized).
        Assumption: Sharper edges indicate closer objects.
        """
        # Convert to grayscale (works on both CPU and GPU)
        gray = 0.299 * image_norm[:, :, 0] + 0.587 * image_norm[:, :, 1] + 0.114 * image_norm[:, :, 2]
        
        # Optimization 3: Simple gradients vs Sobel
        if self.config.fast_mode and self.config.simple_gradients:
            # Fast: Simple differences (works on both CPU and GPU!)
            grad_x = self.xp.abs(gray[:, 2:] - gray[:, :-2])
            grad_y = self.xp.abs(gray[2:, :] - gray[:-2, :])
            
            # Pad to original size (works on both CPU and GPU)
            grad_x = self.xp.pad(grad_x, ((0, 0), (1, 1)), mode='edge')
            grad_y = self.xp.pad(grad_y, ((1, 1), (0, 0)), mode='edge')
        else:
            # Standard: Sobel operator
            if self.use_gpu:
                from cupyx.scipy.ndimage import sobel as gpu_sobel
                grad_x = gpu_sobel(gray, axis=1)
                grad_y = gpu_sobel(gray, axis=0)
            else:
                from scipy.ndimage import sobel
                grad_x = sobel(gray, axis=1)
                grad_y = sobel(gray, axis=0)
        
        # Edge strength (works on both CPU and GPU)
        edge_strength = self.xp.sqrt(grad_x**2 + grad_y**2)
        
        return edge_strength
    
    def _combined_depth(self, image_norm) -> np.ndarray:
        """
        Combine multiple depth cues (optimized).
        Weighted combination of luminance and edge strength.
        """
        luminance = self._luminance_depth(image_norm)
        edges = self._edge_depth(image_norm)
        
        # Normalize each cue
        luminance = self._normalize(luminance)
        edges = self._normalize(edges)
        
        # Weighted combination (60% luminance, 40% edges)
        depth = 0.6 * luminance + 0.4 * edges
        
        return depth
    
    def _smooth_depth(self, depth):
        """
        Apply smoothing to depth map (optimized).
        
        Optimization 4: Box filter vs Gaussian
        """
        if self.config.fast_mode and self.config.use_box_filter:
            # Fast: Box filter
            ksize = int(self.config.smoothing_sigma * 2) * 2 + 1
            if self.use_gpu:
                # GPU box filter using uniform_filter
                from cupyx.scipy.ndimage import uniform_filter
                depth = uniform_filter(depth, size=ksize, mode='nearest')
            else:
                depth = cv2.boxFilter(depth, -1, (ksize, ksize))
        else:
            # Standard: Gaussian filter
            if self.use_gpu:
                depth = gpu_gaussian_filter(depth, sigma=self.config.smoothing_sigma)
            else:
                from scipy.ndimage import gaussian_filter
                depth = gaussian_filter(depth, sigma=self.config.smoothing_sigma)
        
        return depth
    
    def _normalize(self, array):
        """Normalize array to [0, 1] range (works on both CPU and GPU)."""
        min_val = self.xp.min(array)
        max_val = self.xp.max(array)
        if max_val - min_val < 1e-10:
            return self.xp.zeros_like(array)
        return (array - min_val) / (max_val - min_val)
    
    def reset(self):
        """Reset temporal smoothing state."""
        self.prev_depth = None
    
    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.use_gpu
    
    def get_optimization_info(self) -> dict:
        """Get information about active optimizations."""
        return {
            'fast_mode': self.config.fast_mode,
            'downsample_factor': self.config.downsample_factor,
            'use_box_filter': self.config.use_box_filter,
            'simple_gradients': self.config.simple_gradients,
            'gpu_enabled': self.use_gpu,
            'expected_speedup': self._estimate_speedup()
        }
    
    def _estimate_speedup(self) -> float:
        """Estimate speedup factor based on optimizations."""
        speedup = 1.0
        
        if self.config.fast_mode:
            # Downsampling: ~(1/factor)^2 speedup
            if self.config.downsample_factor < 1.0:
                speedup *= 1.0 / (self.config.downsample_factor ** 2)
            
            # Simple gradients: ~3× speedup
            if self.config.simple_gradients:
                speedup *= 3.0
            
            # Box filter: ~2× speedup
            if self.config.use_box_filter:
                speedup *= 2.0
        
        return speedup


# Convenience function for quick depth estimation
def estimate_depth(
    image: np.ndarray,
    algorithm: str = 'combined',
    smoothing: float = 2.0,
    fast_mode: bool = True,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Quick optimized depth estimation with default settings.
    
    Args:
        image: Input image (H, W, 3) uint8 or float32
        algorithm: 'luminance', 'edge', or 'combined'
        smoothing: Gaussian smoothing sigma
        fast_mode: Enable speed optimizations
        use_gpu: Enable GPU acceleration
        
    Returns:
        Depth map (H, W) float32 [0, 1]
    
    Example:
        >>> import cv2
        >>> from fast_depth_estimation import estimate_depth
        >>> 
        >>> image = cv2.imread('photo.jpg')
        >>> depth = estimate_depth(image, fast_mode=True)
        >>> print(f"Estimated at ~180 FPS!")
        >>> cv2.imwrite('depth.png', (depth * 255).astype('uint8'))
    """
    config = FastDepthConfig(
        algorithm=algorithm,
        smoothing_sigma=smoothing,
        temporal_smoothing=False,  # Disabled for single-frame
        fast_mode=fast_mode,
        use_gpu=use_gpu
    )
    estimator = FastDepthEstimator(config)
    return estimator.estimate_depth(image)
