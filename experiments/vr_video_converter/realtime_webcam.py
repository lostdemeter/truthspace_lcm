#!/usr/bin/env python3
"""
Real-time Webcam Depth Estimation
==================================

Uses the optimized φ-depth estimator (torch.compile + FP16) for
real-time depth visualization from webcam input.

Usage:
    python realtime_webcam.py

Controls:
    q - Quit
    s - Save current frame
    c - Toggle colormap (magma/viridis/plasma/inferno)
    f - Toggle FPS display
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path

from phi_depth_estimation import PhiDepthEstimator, PhiDepthConfig


class RealtimeDepthCamera:
    """Real-time depth estimation using optimized φ-depth."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        
        # Initialize φ-depth estimator with optimizations
        print("Initializing optimized φ-depth estimator...")
        config = PhiDepthConfig(
            use_gpu=True,
            use_fp16=True,
            use_compile=True,
            temporal_smoothing=True,
            temporal_alpha=0.3
        )
        self.depth_estimator = PhiDepthEstimator(config=config)
        
        # Colormaps
        self.colormaps = [
            ('magma', cv2.COLORMAP_MAGMA),
            ('viridis', cv2.COLORMAP_VIRIDIS),
            ('plasma', cv2.COLORMAP_PLASMA),
            ('inferno', cv2.COLORMAP_INFERNO),
        ]
        self.colormap_idx = 0
        
        # FPS tracking
        self.fps_history = []
        self.show_fps = True
        self.frame_count = 0
        
    def run(self):
        """Run the real-time depth visualization."""
        print(f"Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return
        
        # Set camera resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        print()
        print("Controls:")
        print("  q - Quit")
        print("  s - Save current frame")
        print("  c - Toggle colormap")
        print("  f - Toggle FPS display")
        print()
        print("Starting real-time depth estimation...")
        
        # Give camera time to initialize
        print("Waiting for camera to initialize...")
        time.sleep(1.0)
        
        # Read a few frames to let camera auto-adjust
        for _ in range(10):
            ret, frame = cap.read()
            cv2.waitKey(1)
        
        if not ret or frame is None:
            print("Error: Camera not providing frames")
            cap.release()
            return
        
        print(f"Got frame: {frame.shape}")
        
        # Warmup depth estimator
        print("Warming up depth estimator...")
        for _ in range(3):
            _ = self.depth_estimator.estimate_depth(frame)
        print("Ready!")
        
        # Force window creation
        cv2.namedWindow('Real-time phi-Depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time phi-Depth', 1280, 480)
        
        last_time = time.perf_counter()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Cannot read frame")
                break
            
            # Estimate depth
            t0 = time.perf_counter()
            try:
                depth = self.depth_estimator.estimate_depth(frame)
            except Exception as e:
                print(f"Depth estimation error: {e}")
                continue
            inference_time = time.perf_counter() - t0
            
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}: {inference_time*1000:.1f}ms")
            
            # Convert depth to display format
            depth_np = depth.cpu().numpy() if torch.is_tensor(depth) else depth
            if depth_np.ndim == 3:
                depth_np = depth_np[0]  # Remove batch dim
            
            # Normalize to 0-255
            depth_norm = depth_np - depth_np.min()
            depth_norm = depth_norm / (depth_norm.max() + 1e-8)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            
            # Resize depth to match frame size
            depth_resized = cv2.resize(depth_uint8, (frame.shape[1], frame.shape[0]))
            
            # Apply colormap
            colormap_name, colormap = self.colormaps[self.colormap_idx]
            depth_colored = cv2.applyColorMap(depth_resized, colormap)
            
            # Calculate FPS
            current_time = time.perf_counter()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = np.mean(self.fps_history)
            
            # Create side-by-side display
            display = np.hstack([frame, depth_colored])
            
            # Add FPS overlay
            if self.show_fps:
                fps_text = f"FPS: {avg_fps:.1f} ({inference_time*1000:.1f}ms)"
                cv2.putText(display, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Colormap: {colormap_name}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Show - use waitKey with longer delay to ensure window updates
            cv2.imshow('Real-time phi-Depth', display)
            
            # Force GUI update
            cv2.pollKey()
            
            # Handle keyboard input - waitKey is required for window to update
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"depth_capture_{self.frame_count:04d}.png"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")
            elif key == ord('c'):
                self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
                print(f"Colormap: {self.colormaps[self.colormap_idx][0]}")
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print()
        print(f"Processed {self.frame_count} frames")
        print(f"Average FPS: {np.mean(self.fps_history):.1f}")


def main():
    print("="*60)
    print("REAL-TIME φ-DEPTH WEBCAM")
    print("="*60)
    print()
    print("Using optimized φ-depth estimator:")
    print("  • torch.compile for kernel fusion")
    print("  • FP16 inference")
    print("  • AIG integer shift-add decoder")
    print("  • Temporal smoothing")
    print()
    
    camera = RealtimeDepthCamera(camera_id=0)
    camera.run()


if __name__ == "__main__":
    main()
