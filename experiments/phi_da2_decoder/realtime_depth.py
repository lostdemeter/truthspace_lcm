"""
Real-time φ-Depth Webcam Application
=====================================

Captures webcam frames and renders depth using the φ-arithmetic decoder
in real-time.

Requirements:
- OpenCV (cv2)
- PyTorch
- transformers (for DA2 backbone)

Usage:
    python realtime_depth.py

Controls:
    q - Quit
    s - Save current frame
    m - Toggle colormap (magma/viridis/plasma)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))
from phi_decoder import PhiDecoder, PhiConfig

PHI = (1 + np.sqrt(5)) / 2


class PhiDepthCamera:
    """Real-time depth estimation using φ-arithmetic decoder."""
    
    def __init__(self, weights_path: Path = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load DA2 model (we need the backbone for feature extraction)
        print("Loading DA2 backbone...")
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        
        self.processor = AutoImageProcessor.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf'
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf'
        ).to(self.device)
        self.model.eval()
        
        # Load φ-decoder weights
        if weights_path is None:
            weights_path = Path(__file__).parent / 'phi_weights.bin'
        
        print(f"Loading φ-decoder weights ({weights_path.stat().st_size} bytes)...")
        self.config = PhiConfig(k_weights=512, bits_weights=16)
        self.decoder = PhiDecoder(self.config)
        self.decoder.load_weights(weights_path)
        
        # Pre-compute LUT for speed
        self._precompute_lut()
        
        # Register hook for feature extraction
        self.captured_features = None
        self._register_hook()
        
        # Colormap options
        self.colormaps = [cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA]
        self.colormap_names = ['magma', 'viridis', 'plasma']
        self.colormap_idx = 0
        
        print("Ready!")
    
    def _precompute_lut(self):
        """Pre-compute lookup table for φ values."""
        n_levels = self.config.n_levels_weights
        bias = self.config.bias_weights
        k = self.config.k_weights
        self.lut = np.array([PHI ** ((e - bias) / k) for e in range(n_levels)], dtype=np.float32)
    
    def _register_hook(self):
        """Register forward hook to capture head features."""
        def hook(module, input, output):
            self.captured_features = output.detach()
        
        self.hook_handle = self.model.head.activation1.register_forward_hook(hook)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return depth map.
        
        Args:
            frame: BGR image from webcam (H, W, 3)
            
        Returns:
            Depth map as uint8 image (H, W)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess for DA2
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (captures features via hook)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Get features
        features = self.captured_features.squeeze().cpu().numpy()
        H, W = features.shape[1], features.shape[2]
        features = features.transpose(1, 2, 0).reshape(-1, 32)
        
        # φ-decoder prediction
        depth = self._phi_predict(features)
        depth = depth.reshape(H, W)
        
        # Normalize to 0-255
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        
        return depth_norm
    
    def _phi_predict(self, features: np.ndarray) -> np.ndarray:
        """Fast φ-arithmetic prediction using pre-computed LUT."""
        # Get decoder weights
        w = self.decoder.weights
        bias = self.config.bias_weights
        k = self.config.k_weights
        
        # Convert features to φ-representation
        signs = np.sign(features).astype(np.int8)
        signs[signs == 0] = 1
        magnitudes = np.abs(features) + 1e-15
        exponents = (k * np.log(magnitudes) / np.log(PHI)).astype(np.int32) + bias
        exponents = np.clip(exponents, 0, len(self.lut) - 1)
        
        # LUT lookup for features
        feat_vals = self.lut[exponents] * signs
        
        # Get weight values
        weight_vals = self.lut[w.weights.exponents] * w.weights.signs
        mean_vals = self.lut[w.feature_mean.exponents] * w.feature_mean.signs
        
        # Center and dot product
        feat_centered = feat_vals - mean_vals
        pred = feat_centered @ weight_vals + w.target_mean.to_float()
        
        return pred
    
    def run(self, camera_id: int = 0):
        """
        Run the real-time depth visualization.
        
        Args:
            camera_id: Camera device ID (default 0)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nControls:")
        print("  q - Quit")
        print("  s - Save current frame")
        print("  m - Toggle colormap")
        print()
        
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            t0 = time.time()
            depth = self.process_frame(frame)
            process_time = (time.time() - t0) * 1000
            
            # Apply colormap
            depth_colored = cv2.applyColorMap(depth, self.colormaps[self.colormap_idx])
            
            # Resize depth to match frame size
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Add info overlay
            info_text = f"FPS: {fps:.1f} | Process: {process_time:.0f}ms | Colormap: {self.colormap_names[self.colormap_idx]}"
            cv2.putText(depth_colored, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colored, "phi-Depth (125 bytes)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create side-by-side display
            combined = np.hstack([frame, depth_colored])
            
            # Show
            cv2.imshow('phi-Depth Real-time', combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f'phi_depth_{timestamp}_rgb.png', frame)
                cv2.imwrite(f'phi_depth_{timestamp}_depth.png', depth_colored)
                print(f"Saved frames with timestamp {timestamp}")
            elif key == ord('m'):
                # Toggle colormap
                self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
                print(f"Colormap: {self.colormap_names[self.colormap_idx]}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hook_handle.remove()
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


def main():
    print("=" * 60)
    print("φ-DEPTH REAL-TIME WEBCAM")
    print("=" * 60)
    print()
    print("This application uses the 125-byte φ-decoder to render")
    print("depth estimation from your webcam in real-time.")
    print()
    
    # Check for camera
    import argparse
    parser = argparse.ArgumentParser(description='Real-time φ-depth from webcam')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file')
    args = parser.parse_args()
    
    weights_path = Path(args.weights) if args.weights else None
    
    try:
        app = PhiDepthCamera(weights_path)
        app.run(camera_id=args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
