"""
GPU Stereo Conversion
======================

GPU-accelerated stereo conversion using CuPy for real-time performance.
Replaces CPU-based cv2.remap with GPU kernels.

Performance target: ~2-3ms per frame (vs 15ms on CPU)
"""

import numpy as np
from typing import Tuple, Dict, Optional

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as gpu_map_coordinates
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available, falling back to CPU")

import cv2


class GPUStereoConverter:
    """
    GPU-accelerated stereo conversion using CuPy.
    
    Performance: ~2-3ms per frame vs 15ms on CPU
    """
    
    def __init__(self, fit_mode: str = 'fit'):
        self.fit_mode = fit_mode
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.coord_cache: Dict[tuple, dict] = {}
        
        if self.use_gpu:
            print("GPUStereoConverter: Using CUDA acceleration")
        else:
            print("GPUStereoConverter: Using CPU fallback")
    
    def _get_equirect_maps_gpu(self, src_h: int, src_w: int, out_h: int, out_w: int):
        """
        Get cached coordinate maps for equirectangular projection (on GPU).
        """
        key = (src_h, src_w, out_h, out_w, self.fit_mode)
        
        if key not in self.coord_cache:
            xp = self.xp
            
            # Create coordinate grids on GPU
            y = xp.arange(out_h, dtype=xp.float32)
            x = xp.arange(out_w, dtype=xp.float32)
            xx, yy = xp.meshgrid(x, y)
            
            # Normalize to [0, 1]
            u_norm = xx / (out_w - 1)
            v_norm = yy / (out_h - 1)
            
            # Convert to spherical coordinates (180° horizontal, 90° vertical FOV)
            longitude = (u_norm - 0.5) * xp.pi  # -π/2 to π/2
            latitude = (0.5 - v_norm) * (xp.pi / 2)  # -π/4 to π/4
            
            # Convert spherical to 3D ray
            cos_lat = xp.cos(latitude)
            sin_lat = xp.sin(latitude)
            cos_lon = xp.cos(longitude)
            sin_lon = xp.sin(longitude)
            
            ray_x = cos_lat * sin_lon
            ray_y = sin_lat
            ray_z = cos_lat * cos_lon
            
            # Compute azimuth and elevation
            azimuth = xp.arctan2(ray_x, ray_z)
            elevation = xp.arcsin(xp.clip(ray_y, -1, 1))
            
            # Normalize to [0, 1]
            u_norm_full = azimuth / xp.pi + 0.5
            v_norm_full = 1.0 - (elevation / (xp.pi / 2) + 0.5)
            
            # Apply fit mode
            if self.fit_mode == 'stretch':
                u_final = u_norm_full
                v_final = v_norm_full
            elif self.fit_mode == 'fit':
                src_aspect = src_w / src_h
                fov_aspect = 2.0
                
                if src_aspect > fov_aspect:
                    v_scale = src_aspect / fov_aspect
                    u_final = u_norm_full
                    v_final = (v_norm_full - 0.5) * v_scale + 0.5
                else:
                    u_scale = fov_aspect / src_aspect
                    u_final = (u_norm_full - 0.5) * u_scale + 0.5
                    v_final = v_norm_full
            elif self.fit_mode == 'fill':
                src_aspect = src_w / src_h
                fov_aspect = 2.0
                
                if src_aspect > fov_aspect:
                    u_scale = fov_aspect / src_aspect
                    u_final = (u_norm_full - 0.5) * u_scale + 0.5
                    v_final = v_norm_full
                else:
                    v_scale = src_aspect / fov_aspect
                    u_final = u_norm_full
                    v_final = (v_norm_full - 0.5) * v_scale + 0.5
            else:
                u_final = u_norm_full
                v_final = v_norm_full
            
            # Convert to pixel coordinates
            # DON'T clamp - let map_coordinates handle out-of-bounds with mode='constant'
            map_x = u_final * (src_w - 1)
            map_y = v_final * (src_h - 1)
            
            self.coord_cache[key] = {'map_x': map_x, 'map_y': map_y}
        
        return self.coord_cache[key]['map_x'], self.coord_cache[key]['map_y']
    
    def _remap_gpu(self, image, map_x, map_y):
        """
        GPU-accelerated bilinear interpolation remap.
        
        Uses CuPy's map_coordinates for fast GPU sampling.
        """
        xp = self.xp
        
        if len(image.shape) == 3:
            # Color image - process each channel
            h, w, c = image.shape
            result = xp.zeros((map_x.shape[0], map_x.shape[1], c), dtype=xp.float32)
            
            # Stack coordinates for map_coordinates (expects [y, x] order)
            coords = xp.stack([map_y, map_x], axis=0)
            
            for ch in range(c):
                channel = image[:, :, ch].astype(xp.float32)
                # Use mode='constant' with cval=0 for black borders (not 'nearest' which causes banding)
                result[:, :, ch] = gpu_map_coordinates(channel, coords, order=1, mode='constant', cval=0.0)
            
            return result.astype(xp.uint8)
        else:
            # Grayscale
            coords = xp.stack([map_y, map_x], axis=0)
            return gpu_map_coordinates(image.astype(xp.float32), coords, order=1, mode='constant', cval=0.0)
    
    def _remap_cpu(self, image, map_x, map_y):
        """CPU fallback using cv2.remap."""
        if self.use_gpu:
            map_x_cpu = cp.asnumpy(map_x)
            map_y_cpu = cp.asnumpy(map_y)
        else:
            map_x_cpu = map_x
            map_y_cpu = map_y
        
        return cv2.remap(image, map_x_cpu, map_y_cpu, cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT)
    
    def project_to_equirect_gpu(self, image, out_h: int, out_w: int):
        """
        Project image to equirectangular using GPU.
        
        Args:
            image: Input image (numpy or cupy array)
            out_h: Output height
            out_w: Output width
            
        Returns:
            Projected image (on GPU if available)
        """
        xp = self.xp
        src_h, src_w = image.shape[:2]
        
        # Transfer to GPU if needed
        if self.use_gpu and not isinstance(image, cp.ndarray):
            image_gpu = cp.asarray(image)
        else:
            image_gpu = image
        
        # Get coordinate maps
        map_x, map_y = self._get_equirect_maps_gpu(src_h, src_w, out_h, out_w)
        
        # Remap
        if self.use_gpu:
            return self._remap_gpu(image_gpu, map_x, map_y)
        else:
            return self._remap_cpu(image_gpu, map_x, map_y)
    
    def apply_stereo_shift_gpu(self, image, depth, shift_pixels: float):
        """
        Apply stereo shift based on depth (GPU accelerated).
        
        Args:
            image: Equirectangular image (H, W, 3) on GPU
            depth: Depth map (H, W) on GPU, normalized [0, 1]
            shift_pixels: Maximum horizontal shift in pixels
            
        Returns:
            Shifted image (on GPU)
        """
        xp = self.xp
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y = xp.arange(h, dtype=xp.float32)
        x = xp.arange(w, dtype=xp.float32)
        xx, yy = xp.meshgrid(x, y)
        
        # Compute shift based on depth
        pixel_shift = shift_pixels * depth
        
        # Apply shift to x coordinates
        map_x = xp.clip(xx - pixel_shift, 0, w - 1)
        map_y = yy
        
        # Remap
        if self.use_gpu:
            return self._remap_gpu(image, map_x, map_y)
        else:
            return self._remap_cpu(image, map_x, map_y)
    
    def process_frame_gpu(self, frame, depth,
                         output_height: int = 1920,
                         ipd_mm: float = 64.0,
                         depth_scale: float = 0.3) -> Tuple:
        """
        Process a single frame to stereo pair (GPU accelerated).
        
        Args:
            frame: Input BGR frame (numpy array)
            depth: Depth map (numpy array), normalized [0, 1]
            output_height: Output height per eye
            ipd_mm: Inter-pupillary distance in mm
            depth_scale: Depth effect strength (0.1-0.5 typical)
            
        Returns:
            (left_eye, right_eye) tuple as numpy arrays
        """
        xp = self.xp
        out_h = output_height
        out_w = output_height
        
        # Transfer to GPU
        if self.use_gpu:
            frame_gpu = cp.asarray(frame)
            depth_gpu = cp.asarray(depth)
        else:
            frame_gpu = frame
            depth_gpu = depth
        
        # Project to equirectangular (on GPU)
        equirect_image = self.project_to_equirect_gpu(frame_gpu, out_h, out_w)
        equirect_depth = self.project_to_equirect_gpu(depth_gpu, out_h, out_w)
        
        # Normalize depth
        depth_max = xp.max(equirect_depth)
        if depth_max > 0:
            equirect_depth = equirect_depth / depth_max
        
        # Calculate stereo shift in pixels
        max_shift_pixels = out_w * depth_scale * 0.02
        
        # Apply stereo shifts (on GPU)
        left = self.apply_stereo_shift_gpu(equirect_image, equirect_depth, +max_shift_pixels)
        right = self.apply_stereo_shift_gpu(equirect_image, equirect_depth, -max_shift_pixels)
        
        # Transfer back to CPU
        if self.use_gpu:
            left_cpu = cp.asnumpy(left)
            right_cpu = cp.asnumpy(right)
        else:
            left_cpu = left
            right_cpu = right
        
        return left_cpu, right_cpu


def benchmark_gpu_stereo():
    """Benchmark GPU vs CPU stereo conversion."""
    import time
    
    print("=" * 70)
    print("GPU STEREO BENCHMARK")
    print("=" * 70)
    print()
    
    # Create test data
    frame = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
    depth = np.random.rand(480, 854).astype(np.float32)
    
    # GPU converter
    gpu_converter = GPUStereoConverter(fit_mode='fit')
    
    # Warmup
    print("Warming up GPU...")
    for _ in range(10):
        left, right = gpu_converter.process_frame_gpu(frame, depth, output_height=1920)
    
    if GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
    
    # Benchmark GPU
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        left, right = gpu_converter.process_frame_gpu(frame, depth, output_height=1920)
    if GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
    gpu_time = (time.perf_counter() - t0) / n
    
    print(f"Output shape: {left.shape}")
    print()
    print("RESULTS:")
    print("-" * 50)
    print(f"  GPU stereo:  {gpu_time*1000:6.1f}ms  ({1/gpu_time:5.1f} FPS)")
    print()
    
    # Compare with CPU (cv2.remap)
    from fast_stereo import FastStereoConverter
    cpu_converter = FastStereoConverter(fit_mode='fit')
    
    # Warmup CPU
    for _ in range(5):
        left, right = cpu_converter.process_frame(frame, depth, output_height=1920)
    
    t0 = time.perf_counter()
    for _ in range(n):
        left, right = cpu_converter.process_frame(frame, depth, output_height=1920)
    cpu_time = (time.perf_counter() - t0) / n
    
    print(f"  CPU stereo:  {cpu_time*1000:6.1f}ms  ({1/cpu_time:5.1f} FPS)")
    print()
    print(f"  Speedup:     {cpu_time/gpu_time:.1f}x")
    print()
    
    # Test at different resolutions
    print("RESOLUTION SCALING (GPU):")
    print("-" * 50)
    for out_h in [960, 1440, 1920, 2160]:
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            left, right = gpu_converter.process_frame_gpu(frame, depth, output_height=out_h)
            if GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        fps = 1 / np.mean(times)
        print(f"  {out_h}x{out_h} per eye:  {np.mean(times)*1000:6.1f}ms  ({fps:5.1f} FPS)")


if __name__ == "__main__":
    benchmark_gpu_stereo()
