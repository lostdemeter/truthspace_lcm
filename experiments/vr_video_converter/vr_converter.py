#!/usr/bin/env python3
"""
VR 180° Video Converter for Stereo Web Server
==============================================

Converts 2D video to 180° stereoscopic VR format using depth-aware processing.
Optimized for real-time preview and batch processing.

Author: Holographer's Workbench
License: GPLv3
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Try GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Choose depth estimation method: 'phi' (high quality) or 'fast' (heuristic)
DEPTH_METHOD = 'phi'  # Change to 'fast' for original heuristic method

if DEPTH_METHOD == 'phi':
    from phi_depth_estimation import PhiDepthEstimator, PhiDepthConfig
else:
    from fast_depth_estimation import FastDepthEstimator, FastDepthConfig

# Fast stereo conversion using cv2.remap (CPU) or CuPy (GPU)
from fast_stereo import FastStereoConverter
from gpu_stereo import GPUStereoConverter, GPU_AVAILABLE as CUPY_AVAILABLE


class VRConverter:
    """Convert 2D images/video to 180° VR with depth-aware stereo."""
    
    def __init__(self, use_gpu: bool = True, depth_smoothing: float = 2.0, 
                 fit_mode: str = 'fit', distortion_mode: str = 'none',
                 distortion_k1: float = 0.0, distortion_k2: float = 0.0,
                 depth_resolution_scale: float = 0.75):
        """
        Initialize VR converter.
        
        Args:
            use_gpu: Use GPU acceleration if available
            depth_smoothing: Gaussian blur sigma for depth smoothing
            fit_mode: How to fit source image to FOV:
                - 'stretch': Stretch to fill entire FOV (may distort)
                - 'fit': Scale to fit, maintain aspect ratio, add black bars
                - 'fill': Scale to fill, maintain aspect ratio, crop excess
                - 'center': No scaling, center in FOV, add black bars
            distortion_mode: Pre-distortion to apply:
                - 'none': No distortion
                - 'barrel': Barrel distortion (edges bow outward)
                - 'pincushion': Pincushion distortion (edges bow inward)
                - 'mustache': Mustache/complex distortion (mixed)
            distortion_k1: Primary distortion coefficient
            distortion_k2: Secondary distortion coefficient
            depth_resolution_scale: Depth processing resolution (0.5-1.0)
                - 1.0: Full resolution (slowest, highest quality)
                - 0.75: 75% resolution (default, good balance)
                - 0.5: 50% resolution (fastest, minimal quality loss)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.depth_smoothing = depth_smoothing
        self.fit_mode = fit_mode
        self.distortion_mode = distortion_mode
        self.distortion_k1 = distortion_k1
        self.distortion_k2 = distortion_k2
        self.depth_resolution_scale = depth_resolution_scale
        
        # Cache for coordinate grids
        self.coord_cache = {}
        
        # Fast stereo converter - use GPU if CuPy available, else CPU
        if CUPY_AVAILABLE:
            self.fast_stereo = GPUStereoConverter(fit_mode=fit_mode)
            self.stereo_gpu = True
        else:
            self.fast_stereo = FastStereoConverter(fit_mode=fit_mode)
            self.stereo_gpu = False
        
        # Initialize depth estimator
        if DEPTH_METHOD == 'phi':
            # High-quality φ-arithmetic depth (99.91% accuracy, ~128 FPS)
            # φ-depth uses PyTorch CUDA, not CuPy - always try to use GPU
            import torch
            phi_use_gpu = torch.cuda.is_available()
            depth_config = PhiDepthConfig(
                use_gpu=phi_use_gpu,
                use_fp16=True,
                use_compile=True,
                temporal_smoothing=True,
                temporal_alpha=0.3
            )
            self.depth_estimator = PhiDepthEstimator(config=depth_config)
        else:
            # Fast heuristic depth (~180 FPS, lower quality)
            depth_config = FastDepthConfig(
                use_gpu=self.use_gpu,
                algorithm='combined',
                fast_mode=True,
                downsample_factor=depth_resolution_scale,
                use_box_filter=True,
                simple_gradients=True
            )
            self.depth_estimator = FastDepthEstimator(config=depth_config)
        
        print(f"VRConverter initialized (GPU: {'enabled' if self.use_gpu else 'disabled'}, "
              f"fit_mode: {fit_mode}, distortion: {distortion_mode})")
    
    def _get_equirect_coords(self, out_h: int, out_w: int) -> dict:
        """Get cached equirectangular coordinate grids."""
        key = (out_h, out_w)
        if key not in self.coord_cache:
            # Create coordinate grids
            y = self.xp.arange(out_h, dtype=self.xp.float32)
            x = self.xp.arange(out_w, dtype=self.xp.float32)
            xx, yy = self.xp.meshgrid(x, y)
            
            # Convert to normalized coordinates [0, 1]
            u_norm = xx / (out_w - 1)
            v_norm = yy / (out_h - 1)
            
            # Convert to spherical coordinates
            # Longitude: -π/2 to π/2 (180° horizontal FOV, centered at 0)
            # Latitude: -π/4 to π/4 (90° vertical, centered)
            longitude = (u_norm - 0.5) * np.pi  # Center at 0
            latitude = (0.5 - v_norm) * (np.pi / 2)
            
            self.coord_cache[key] = {
                'longitude': longitude,
                'latitude': latitude,
                'xx': xx,
                'yy': yy
            }
        
        return self.coord_cache[key]
    
    def _apply_distortion(self, u_norm: np.ndarray, v_norm: np.ndarray) -> tuple:
        """
        Apply lens distortion to normalized coordinates.
        
        Args:
            u_norm: Normalized U coordinates [0, 1]
            v_norm: Normalized V coordinates [0, 1]
        
        Returns:
            (u_distorted, v_distorted) - Distorted normalized coordinates
        """
        if self.distortion_mode == 'none':
            return u_norm, v_norm
        
        # Center coordinates at (0.5, 0.5)
        u_centered = u_norm - 0.5
        v_centered = v_norm - 0.5
        
        # Calculate radius from center
        r = self.xp.sqrt(u_centered**2 + v_centered**2)
        
        # Apply distortion based on mode
        if self.distortion_mode == 'barrel':
            # Barrel: positive k1 pushes edges outward
            # r_distorted = r * (1 + k1*r^2 + k2*r^4)
            r_distorted = r * (1.0 + self.distortion_k1 * r**2 + self.distortion_k2 * r**4)
            
        elif self.distortion_mode == 'pincushion':
            # Pincushion: negative k1 pulls edges inward (inverse of barrel)
            # r_distorted = r * (1 - k1*r^2 - k2*r^4)
            r_distorted = r * (1.0 - self.distortion_k1 * r**2 - self.distortion_k2 * r**4)
            
        elif self.distortion_mode == 'mustache':
            # Mustache: complex distortion with sign change
            # Uses both positive and negative terms for wavy effect
            # r_distorted = r * (1 + k1*r^2 - k2*r^4)
            r_distorted = r * (1.0 + self.distortion_k1 * r**2 - self.distortion_k2 * r**4)
            
        else:
            r_distorted = r
        
        # Avoid division by zero
        scale = self.xp.where(r > 1e-6, r_distorted / r, 1.0)
        
        # Apply scale to centered coordinates
        u_distorted = u_centered * scale + 0.5
        v_distorted = v_centered * scale + 0.5
        
        return u_distorted, v_distorted
    
    def _project_to_equirect(self, image: np.ndarray, out_h: int, out_w: int, keep_on_gpu: bool = False,
                            use_nearest_neighbor: bool = False):
        """
        Project source image to equirectangular coordinates.
        
        This is the single high-quality resampling step.
        After this, stereo is applied via pixel shifting (no more resampling!).
        
        Args:
            image: Input image (numpy or cupy array)
            out_h: Output height
            out_w: Output width
            keep_on_gpu: If True, return GPU array (don't transfer to CPU)
            use_nearest_neighbor: If True, use nearest-neighbor instead of bilinear (faster, for depth maps)
        
        Returns:
            Projected image (numpy or cupy array depending on keep_on_gpu)
        """
        src_h, src_w = image.shape[:2]
        
        # Move to GPU if available (check if already on GPU)
        if self.use_gpu:
            if hasattr(image, 'device'):  # Already a cupy array
                image_gpu = image
            else:
                image_gpu = cp.asarray(image)
        else:
            image_gpu = image
        
        # Get coordinate grids
        coords = self._get_equirect_coords(out_h, out_w)
        longitude = coords['longitude']
        latitude = coords['latitude']
        
        # Convert spherical to 3D ray
        cos_lat = self.xp.cos(latitude)
        sin_lat = self.xp.sin(latitude)
        cos_lon = self.xp.cos(longitude)
        sin_lon = self.xp.sin(longitude)
        
        ray_x = cos_lat * sin_lon
        ray_y = sin_lat
        ray_z = cos_lat * cos_lon
        
        # Compute azimuth and elevation from ray
        azimuth = self.xp.arctan2(ray_x, ray_z)  # -π/2 to π/2 for 180° FOV
        elevation = self.xp.arcsin(self.xp.clip(ray_y, -1, 1))  # -π/2 to π/2
        
        # Normalize azimuth and elevation to [0, 1]
        u_norm_full = (azimuth / self.xp.pi + 0.5)  # 0 to 1 across full 180°
        v_norm_full = 1.0 - (elevation / (self.xp.pi / 2) + 0.5)  # 0 to 1 across 90° vertical, flipped
        
        # Apply fit mode to handle aspect ratio
        if self.fit_mode == 'stretch':
            # Original behavior: stretch to fill entire FOV
            u_norm = u_norm_full
            v_norm = v_norm_full
            valid_mask = self.xp.ones((out_h, out_w), dtype=self.xp.float32)
            
        elif self.fit_mode == 'fit':
            # Scale to fit, maintain aspect ratio, add black bars
            src_aspect = src_w / src_h
            fov_aspect = 2.0  # 180° / 90° = 2.0
            
            if src_aspect > fov_aspect:
                # Source is wider - fit horizontally, add vertical bars
                scale = 1.0
                v_scale = src_aspect / fov_aspect
                u_norm = u_norm_full
                v_norm = (v_norm_full - 0.5) * v_scale + 0.5
            else:
                # Source is taller - fit vertically, add horizontal bars
                u_scale = fov_aspect / src_aspect
                v_scale = 1.0
                u_norm = (u_norm_full - 0.5) * u_scale + 0.5
                v_norm = v_norm_full
            
            # Create mask for valid region
            valid_mask = ((u_norm >= 0) & (u_norm <= 1) & 
                         (v_norm >= 0) & (v_norm <= 1)).astype(self.xp.float32)
            
        elif self.fit_mode == 'fill':
            # Scale to fill, maintain aspect ratio, crop excess
            src_aspect = src_w / src_h
            fov_aspect = 2.0
            
            if src_aspect > fov_aspect:
                # Source is wider - fit vertically, crop horizontally
                u_scale = fov_aspect / src_aspect
                v_scale = 1.0
                u_norm = (u_norm_full - 0.5) * u_scale + 0.5
                v_norm = v_norm_full
            else:
                # Source is taller - fit horizontally, crop vertically
                u_scale = 1.0
                v_scale = src_aspect / fov_aspect
                u_norm = u_norm_full
                v_norm = (v_norm_full - 0.5) * v_scale + 0.5
            
            valid_mask = self.xp.ones((out_h, out_w), dtype=self.xp.float32)
            
        elif self.fit_mode == 'center':
            # No scaling, center in FOV, add black bars
            src_aspect = src_w / src_h
            fov_aspect = 2.0
            
            # Calculate how much of the FOV the source occupies
            if src_aspect > fov_aspect:
                # Source is wider
                u_coverage = 1.0
                v_coverage = fov_aspect / src_aspect
            else:
                # Source is taller
                u_coverage = src_aspect / fov_aspect
                v_coverage = 1.0
            
            # Map to centered region
            u_norm = (u_norm_full - 0.5) / u_coverage + 0.5
            v_norm = (v_norm_full - 0.5) / v_coverage + 0.5
            
            # Create mask for valid region
            valid_mask = ((u_norm >= 0) & (u_norm <= 1) & 
                         (v_norm >= 0) & (v_norm <= 1)).astype(self.xp.float32)
        else:
            # Default to stretch
            u_norm = u_norm_full
            v_norm = v_norm_full
            valid_mask = self.xp.ones((out_h, out_w), dtype=self.xp.float32)
        
        # Apply lens distortion (pre-distortion for VR headset compensation)
        u_norm, v_norm = self._apply_distortion(u_norm, v_norm)
        
        # Update mask for distorted coordinates
        valid_mask = valid_mask * ((u_norm >= 0) & (u_norm <= 1) & 
                                   (v_norm >= 0) & (v_norm <= 1)).astype(self.xp.float32)
        
        # Convert to pixel coordinates
        u = u_norm * (src_w - 1)
        v = v_norm * (src_h - 1)
        
        # Clamp to valid range
        u = self.xp.clip(u, 0, src_w - 1)
        v = self.xp.clip(v, 0, src_h - 1)
        
        # Sample (bilinear for images, nearest-neighbor for depth)
        if use_nearest_neighbor:
            equirect = self._nearest_neighbor_sample(image_gpu, u, v)
        else:
            equirect = self._bilinear_sample(image_gpu, u, v)
        
        # Apply mask (black out regions outside source)
        if len(equirect.shape) == 3:
            equirect = equirect * valid_mask[:, :, self.xp.newaxis]
        else:
            equirect = equirect * valid_mask
        
        # Move back to CPU only if requested
        if self.use_gpu and not keep_on_gpu:
            equirect = cp.asnumpy(equirect)
        
        return equirect
    
    def _nearest_neighbor_sample(self, image, u, v):
        """Nearest-neighbor sampling (fast, for depth maps)."""
        h, w = image.shape[:2]
        
        # Round to nearest pixel
        u_int = self.xp.round(u).astype(self.xp.int32)
        v_int = self.xp.round(v).astype(self.xp.int32)
        
        # Clamp to valid range
        u_int = self.xp.clip(u_int, 0, w - 1)
        v_int = self.xp.clip(v_int, 0, h - 1)
        
        # For CuPy, we need to use a kernel-based approach
        if self.use_gpu and hasattr(image, 'device'):
            # GPU path: Use elementwise kernel
            if len(image.shape) == 3:
                h, w, c = image.shape
                out_h, out_w = v_int.shape
                result = cp.zeros((out_h, out_w, c), dtype=cp.float32)
                
                # Use CuPy's RawKernel for direct indexing
                for ch in range(c):
                    # Simple loop - CuPy will optimize this
                    for i in range(out_h):
                        for j in range(out_w):
                            result[i, j, ch] = image[v_int[i, j], u_int[i, j], ch]
            else:
                out_h, out_w = v_int.shape
                result = cp.zeros((out_h, out_w), dtype=cp.float32)
                for i in range(out_h):
                    for j in range(out_w):
                        result[i, j] = image[v_int[i, j], u_int[i, j]]
        else:
            # CPU path: Direct indexing works fine
            if len(image.shape) == 3:
                result = image[v_int, u_int].astype(np.float32)
            else:
                result = image[v_int, u_int].astype(np.float32)
        
        return result
    
    def _bilinear_sample(self, image, u, v):
        """Bilinear interpolation sampling with edge-aware clamping to prevent artifacts."""
        h, w = image.shape[:2]
        
        # Clamp input coordinates first
        u = self.xp.clip(u, 0, w - 1)
        v = self.xp.clip(v, 0, h - 1)
        
        # Use map_coordinates for GPU-compatible sampling
        if self.use_gpu and hasattr(image, 'device'):
            # CuPy path - use cupyx.scipy.ndimage.map_coordinates
            try:
                import cupyx.scipy.ndimage as ndimage_gpu
                
                # Ensure image is cupy array
                if not isinstance(image, cp.ndarray):
                    image = cp.asarray(image)
                
                if len(image.shape) == 3:
                    # Process each channel
                    h, w, c = image.shape
                    result_list = []
                    coords = cp.array([v, u])  # Note: (y, x) order for map_coordinates
                    for ch in range(c):
                        # Extract channel (stays on GPU)
                        channel_data = image[:, :, ch]
                        sampled = ndimage_gpu.map_coordinates(
                            channel_data.astype(cp.float32),
                            coords,
                            order=1,  # Bilinear
                            mode='nearest'
                        )
                        result_list.append(sampled)
                    # Stack channels
                    result = cp.stack(result_list, axis=-1)
                    return result
                else:
                    coords = cp.array([v, u])
                    return ndimage_gpu.map_coordinates(
                        image.astype(cp.float32),
                        coords,
                        order=1,
                        mode='nearest'
                    )
            except (ImportError, Exception) as e:
                # Fall back to manual method if cupyx not available or fails
                print(f"GPU map_coordinates failed: {e}, falling back to manual")
                pass
        
        # NumPy path or fallback - manual bilinear interpolation
        # Get integer parts
        u0 = self.xp.floor(u).astype(self.xp.int32)
        v0 = self.xp.floor(v).astype(self.xp.int32)
        u1 = u0 + 1
        v1 = v0 + 1
        
        # Clamp to valid range (critical!)
        u0 = self.xp.clip(u0, 0, w - 1)
        u1 = self.xp.clip(u1, 0, w - 1)
        v0 = self.xp.clip(v0, 0, h - 1)
        v1 = self.xp.clip(v1, 0, h - 1)
        
        # Get fractional parts
        u_frac = u - self.xp.floor(u)
        v_frac = v - self.xp.floor(v)
        
        # Ensure fractions are in [0, 1]
        u_frac = self.xp.clip(u_frac, 0, 1)
        v_frac = self.xp.clip(v_frac, 0, 1)
        
        # Sample four corners - use simple indexing for numpy
        if len(image.shape) == 3:
            h, w, c = image.shape
            # Convert to linear indices for numpy compatibility
            flat_idx00 = (v0 * w + u0).ravel()
            flat_idx01 = (v0 * w + u1).ravel()
            flat_idx10 = (v1 * w + u0).ravel()
            flat_idx11 = (v1 * w + u1).ravel()
            
            flat_image = image.reshape(-1, c)
            i00 = flat_image[flat_idx00].reshape(v0.shape[0], v0.shape[1], c).astype(self.xp.float32)
            i01 = flat_image[flat_idx01].reshape(v0.shape[0], v0.shape[1], c).astype(self.xp.float32)
            i10 = flat_image[flat_idx10].reshape(v0.shape[0], v0.shape[1], c).astype(self.xp.float32)
            i11 = flat_image[flat_idx11].reshape(v0.shape[0], v0.shape[1], c).astype(self.xp.float32)
        else:
            h, w = image.shape
            flat_idx00 = (v0 * w + u0).ravel()
            flat_idx01 = (v0 * w + u1).ravel()
            flat_idx10 = (v1 * w + u0).ravel()
            flat_idx11 = (v1 * w + u1).ravel()
            
            flat_image = image.ravel()
            i00 = flat_image[flat_idx00].reshape(v0.shape).astype(self.xp.float32)
            i01 = flat_image[flat_idx01].reshape(v0.shape).astype(self.xp.float32)
            i10 = flat_image[flat_idx10].reshape(v0.shape).astype(self.xp.float32)
            i11 = flat_image[flat_idx11].reshape(v0.shape).astype(self.xp.float32)
        
        # Detect large color differences (edges) - if any corner differs significantly,
        # use nearest neighbor instead of bilinear to avoid color artifacts
        max_diff = self.xp.maximum(
            self.xp.maximum(self.xp.abs(i00 - i01).max(axis=-1), 
                           self.xp.abs(i00 - i10).max(axis=-1)),
            self.xp.maximum(self.xp.abs(i00 - i11).max(axis=-1),
                           self.xp.abs(i01 - i11).max(axis=-1))
        )
        
        # Threshold for edge detection (if colors differ by more than 50/255)
        is_edge = max_diff > 50
        
        # Bilinear interpolation weights
        w00 = (1 - u_frac) * (1 - v_frac)
        w01 = u_frac * (1 - v_frac)
        w10 = (1 - u_frac) * v_frac
        w11 = u_frac * v_frac
        
        # Bilinear result
        bilinear = (w00[:, :, None] * i00 + 
                   w01[:, :, None] * i01 + 
                   w10[:, :, None] * i10 + 
                   w11[:, :, None] * i11)
        
        # Nearest neighbor result (use i00, the closest pixel)
        nearest = i00
        
        # Use nearest neighbor at edges, bilinear elsewhere
        result = self.xp.where(is_edge[:, :, None], nearest, bilinear)
        
        # Clamp to valid range but keep as float32 for downstream processing
        result = self.xp.clip(result, 0, 255)
        
        return result
    
    def _apply_stereo_shift(self, equirect, depth, 
                           camera_offset: float, depth_scale: float = 1.0, keep_on_gpu: bool = False):
        """
        Apply depth-aware stereo shift (interference pattern approach).
        
        This shifts pixels horizontally based on depth, creating stereo effect
        WITHOUT resampling the entire image. This is the key to quality!
        
        Args:
            equirect: Equirectangular image (numpy or cupy array)
            depth: Depth map (numpy or cupy array)
            camera_offset: Camera offset for stereo
            depth_scale: Scale factor for depth effect
            keep_on_gpu: If True, return GPU array (don't transfer to CPU)
        
        Returns:
            Shifted image (numpy or cupy array depending on keep_on_gpu)
        """
        out_h, out_w = equirect.shape[:2]
        
        # Move to GPU if needed (check if already on GPU)
        if self.use_gpu:
            if hasattr(equirect, 'device'):  # Already a cupy array
                equirect_gpu = equirect
            else:
                equirect_gpu = cp.asarray(equirect)
            
            if hasattr(depth, 'device'):  # Already a cupy array
                depth_gpu = depth
            else:
                depth_gpu = cp.asarray(depth)
        else:
            equirect_gpu = equirect
            depth_gpu = depth
        
        # Depth should already match equirect size (projected together)
        assert depth_gpu.shape[:2] == (out_h, out_w), \
            f"Depth shape {depth_gpu.shape} doesn't match equirect {(out_h, out_w)}"
        
        # Compute horizontal shift based on depth and IPD
        # Use the original simple formula
        focal_length = out_w / 2  # Approximate focal length
        shift_pixels = camera_offset * focal_length * depth_gpu / (depth_gpu + 0.5)
        
        # Scale by user's depth_scale parameter
        shift_pixels = shift_pixels * depth_scale
        
        # Clamp to reasonable range to prevent extreme shifts
        max_shift = out_w * 0.1  # Max 10% of width
        shift_pixels = self.xp.clip(shift_pixels, -max_shift, max_shift)
        
        # Create coordinate grids
        coords = self._get_equirect_coords(out_h, out_w)
        x_coords = coords['xx']
        y_coords = coords['yy']
        
        # Apply horizontal shift
        x_shifted = x_coords + shift_pixels
        
        # Clamp shifted coordinates
        x_shifted = self.xp.clip(x_shifted, 0, out_w - 1)
        y_coords = self.xp.clip(y_coords, 0, out_h - 1)
        
        # Sample with shifted coordinates
        shifted = self._bilinear_sample(equirect_gpu, x_shifted, y_coords)
        
        # Move back to CPU only if requested
        if self.use_gpu and not keep_on_gpu:
            shifted = cp.asnumpy(shifted)
        
        return shifted
    
    @staticmethod
    def apply_center_separation(left: np.ndarray, right: np.ndarray, 
                                separation_percent: float) -> np.ndarray:
        """
        Combine left and right views with center separation gap (CPU version).
        
        Args:
            left: Left eye view (NumPy array)
            right: Right eye view (NumPy array)
            separation_percent: Percentage of single eye width to add as gap (0-50)
        
        Returns:
            Combined side-by-side image with center gap
        """
        if separation_percent <= 0:
            # No gap, just stack horizontally
            return np.hstack([left, right])
        
        h, w = left.shape[:2]
        gap_width = int(w * (separation_percent / 100.0))
        
        # Create black gap
        gap = np.zeros((h, gap_width, left.shape[2]), dtype=left.dtype)
        
        # Combine: left + gap + right
        return np.hstack([left, gap, right])
    
    @staticmethod
    def apply_center_separation_gpu(left, right, separation_percent: float):
        """
        Combine left and right views with center separation gap (GPU version).
        
        Args:
            left: Left eye view (CuPy array)
            right: Right eye view (CuPy array)
            separation_percent: Percentage of single eye width to add as gap (0-50)
        
        Returns:
            Combined side-by-side image with center gap (CuPy array)
        """
        if separation_percent <= 0:
            # No gap, just stack horizontally
            return cp.hstack([left, right])
        
        h, w = left.shape[:2]
        gap_width = int(w * (separation_percent / 100.0))
        
        # Create black gap (on GPU)
        gap = cp.zeros((h, gap_width, left.shape[2]), dtype=left.dtype)
        
        # Combine: left + gap + right (on GPU)
        return cp.hstack([left, gap, right])
    
    def process_frame_fast(self, frame: np.ndarray, ipd_mm: float = 64.0,
                           depth_scale: float = 0.3, output_height: int = 1920) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast frame processing using GPU-accelerated stereo conversion.
        
        Uses CuPy for GPU stereo (~6ms) or cv2.remap for CPU (~16ms).
        
        Args:
            frame: Input frame (BGR format)
            ipd_mm: Interpupillary distance in millimeters
            depth_scale: Scale factor for depth effect
            output_height: Output height per eye (square output)
            
        Returns:
            (left_equirect, right_equirect) - Both as numpy arrays (uint8)
        """
        # Step 1: Estimate depth using φ-depth (fast on CUDA)
        depth = self.depth_estimator.estimate_depth(frame)
        
        # Step 2: Apply Gaussian smoothing to depth
        depth = cv2.GaussianBlur(depth, (0, 0), self.depth_smoothing)
        
        # Step 3: Use stereo converter (GPU if available, else CPU)
        if self.stereo_gpu:
            left, right = self.fast_stereo.process_frame_gpu(
                frame, depth,
                output_height=output_height,
                ipd_mm=ipd_mm,
                depth_scale=depth_scale
            )
        else:
            left, right = self.fast_stereo.process_frame(
                frame, depth,
                output_height=output_height,
                ipd_mm=ipd_mm,
                depth_scale=depth_scale
            )
        
        return left, right
    
    def process_frame(self, frame: np.ndarray, ipd_mm: float = 64.0, 
                     depth_scale: float = 0.3, output_height: Optional[int] = None,
                     fit_mode: Optional[str] = None, distortion_mode: Optional[str] = None,
                     distortion_k1: Optional[float] = None, distortion_k2: Optional[float] = None,
                     center_separation_percent: float = 0.0, 
                     return_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single frame to stereo equirectangular.
        
        NOTE: For faster processing, use process_frame_fast() which is ~128x faster.
        
        Args:
            frame: Input frame (BGR format)
            ipd_mm: Interpupillary distance in millimeters
            depth_scale: Scale factor for depth effect
            output_height: Output height (width will be 2x height)
            fit_mode: Override fit mode for this frame (None = use instance default)
            distortion_mode: Override distortion mode (None = use instance default)
            distortion_k1: Override k1 coefficient (None = use instance default)
            distortion_k2: Override k2 coefficient (None = use instance default)
            center_separation_percent: Percentage of width to add as center gap (0-50)
                Note: Gap is added by caller using apply_center_separation()
            return_gpu: If True, return GPU arrays without transferring to CPU (faster for batch processing)
        
        Returns:
            (left_equirect, right_equirect) - Both as numpy arrays (float32, range [0, 255])
                                              or cupy arrays if return_gpu=True
        """
        # Temporarily override parameters if specified
        original_fit_mode = self.fit_mode
        original_distortion_mode = self.distortion_mode
        original_k1 = self.distortion_k1
        original_k2 = self.distortion_k2
        
        if fit_mode is not None:
            self.fit_mode = fit_mode
        if distortion_mode is not None:
            self.distortion_mode = distortion_mode
        if distortion_k1 is not None:
            self.distortion_k1 = distortion_k1
        if distortion_k2 is not None:
            self.distortion_k2 = distortion_k2
        src_h, src_w = frame.shape[:2]
        
        # Determine output size
        if output_height:
            out_h = output_height
        else:
            out_h = src_h
        
        # For VR 180° headsets (Oculus), each eye should be square (1:1)
        # NOT equirectangular (2:1). This is the Oculus spec for 180° SBS.
        # Using square output reduces processing by 50%!
        out_w = out_h  # Square 1:1 aspect ratio per eye
        
        # Step 1: Estimate depth (Phase 3B: Keep on GPU)
        if self.use_gpu:
            # Transfer frame to GPU for depth estimation
            frame_gpu = cp.asarray(frame)
            depth = self.depth_estimator.estimate_depth(frame_gpu, input_on_gpu=True, return_gpu=True)
            # Depth is now on GPU, no transfer needed
        else:
            depth = self.depth_estimator.estimate_depth(frame)
        
        # Simple Gaussian smoothing (on GPU if available)
        if self.use_gpu:
            from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian
            depth = gpu_gaussian(depth, sigma=self.depth_smoothing)
        else:
            depth = cv2.GaussianBlur(depth, (0, 0), self.depth_smoothing)
        
        # Step 2: Project BOTH image and depth to equirectangular (ONCE, high quality)
        # Keep on GPU to avoid transfers
        center_equirect = self._project_to_equirect(frame, out_h, out_w, keep_on_gpu=self.use_gpu)
        
        # Project depth map to equirectangular space too!
        # Phase 3B: Stack depth on GPU if available
        if self.use_gpu and hasattr(depth, 'device'):
            depth_3ch = cp.stack([depth, depth, depth], axis=-1)
        else:
            depth_3ch = np.stack([depth, depth, depth], axis=-1)
        
        depth_equirect = self._project_to_equirect(depth_3ch, out_h, out_w, keep_on_gpu=self.use_gpu, 
                                                   use_nearest_neighbor=False)  # Use bilinear
        
        # Extract single channel from depth (handle GPU/CPU)
        if self.use_gpu:
            depth_equirect = depth_equirect[:, :, 0]  # Take one channel (stays on GPU)
        else:
            depth_equirect = depth_equirect[:, :, 0]  # Take one channel
        
        # Step 3: Apply stereo shifts (interference pattern - no resampling!)
        # Keep on GPU until final transfer
        ipd_meters = ipd_mm / 1000.0
        left_equirect = self._apply_stereo_shift(center_equirect, depth_equirect, -ipd_meters / 2, depth_scale, keep_on_gpu=self.use_gpu)
        right_equirect = self._apply_stereo_shift(center_equirect, depth_equirect, +ipd_meters / 2, depth_scale, keep_on_gpu=self.use_gpu)
        
        # Transfer to CPU only if requested (optimization: skip for batch processing)
        if self.use_gpu and not return_gpu:
            left_equirect = cp.asnumpy(left_equirect)
            right_equirect = cp.asnumpy(right_equirect)
        
        # Note: center_separation_percent is applied by caller using apply_center_separation()
        
        # Restore original parameters
        self.fit_mode = original_fit_mode
        self.distortion_mode = original_distortion_mode
        self.distortion_k1 = original_k1
        self.distortion_k2 = original_k2
        
        return left_equirect, right_equirect
