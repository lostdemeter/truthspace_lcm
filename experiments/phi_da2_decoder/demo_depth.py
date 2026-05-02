"""
φ-Depth Demo Application
========================

Demo that works with either:
- Webcam (if available)
- Test images from COCO val set
- Any video file

This allows testing the φ-decoder without a physical webcam.

Usage:
    python demo_depth.py              # Try webcam, fall back to images
    python demo_depth.py --images     # Use test images
    python demo_depth.py --video FILE # Use video file
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


class PhiDepthDemo:
    """φ-Depth demonstration with multiple input sources."""
    
    def __init__(self, weights_path: Path = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load DA2 model
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
        
        # Pre-compute LUT
        self._precompute_lut()
        
        # Register hook
        self.captured_features = None
        self._register_hook()
        
        # Colormaps
        self.colormaps = [cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO]
        self.colormap_names = ['magma', 'viridis', 'plasma', 'inferno']
        self.colormap_idx = 0
        
        print("Ready!")
    
    def _precompute_lut(self):
        n_levels = self.config.n_levels_weights
        bias = self.config.bias_weights
        k = self.config.k_weights
        self.lut = np.array([PHI ** ((e - bias) / k) for e in range(n_levels)], dtype=np.float32)
    
    def _register_hook(self):
        def hook(module, input, output):
            self.captured_features = output.detach()
        self.hook_handle = self.model.head.activation1.register_forward_hook(hook)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and return depth map."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        features = self.captured_features.squeeze().cpu().numpy()
        H, W = features.shape[1], features.shape[2]
        features = features.transpose(1, 2, 0).reshape(-1, 32)
        
        depth = self._phi_predict(features).reshape(H, W)
        
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        
        return depth_norm
    
    def _phi_predict(self, features: np.ndarray) -> np.ndarray:
        w = self.decoder.weights
        bias = self.config.bias_weights
        k = self.config.k_weights
        
        signs = np.sign(features).astype(np.int8)
        signs[signs == 0] = 1
        magnitudes = np.abs(features) + 1e-15
        exponents = (k * np.log(magnitudes) / np.log(PHI)).astype(np.int32) + bias
        exponents = np.clip(exponents, 0, len(self.lut) - 1)
        
        feat_vals = self.lut[exponents] * signs
        weight_vals = self.lut[w.weights.exponents] * w.weights.signs
        mean_vals = self.lut[w.feature_mean.exponents] * w.feature_mean.signs
        
        feat_centered = feat_vals - mean_vals
        return feat_centered @ weight_vals + w.target_mean.to_float()
    
    def run_webcam(self, camera_id: int = 0):
        """Run with webcam input."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Could not open camera {camera_id}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self._run_loop(cap, "Webcam")
        cap.release()
        return True
    
    def run_video(self, video_path: str):
        """Run with video file input."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return False
        
        self._run_loop(cap, f"Video: {Path(video_path).name}")
        cap.release()
        return True
    
    def run_images(self, image_dir: Path = None):
        """Run with test images from COCO."""
        if image_dir is None:
            image_dir = Path('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017')
        
        # Get some test images
        image_files = sorted(image_dir.glob('*.jpg'))[:20]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return False
        
        print(f"\nFound {len(image_files)} images")
        print("Controls: q=quit, n=next, p=prev, m=colormap, s=save")
        
        idx = 0
        while True:
            img_path = image_files[idx]
            frame = cv2.imread(str(img_path))
            
            if frame is None:
                idx = (idx + 1) % len(image_files)
                continue
            
            # Resize if too large
            max_dim = 800
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
            t0 = time.time()
            depth = self.process_frame(frame)
            process_time = (time.time() - t0) * 1000
            
            depth_colored = cv2.applyColorMap(depth, self.colormaps[self.colormap_idx])
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            
            # Add info
            info = f"[{idx+1}/{len(image_files)}] {img_path.name} | {process_time:.0f}ms | {self.colormap_names[self.colormap_idx]}"
            cv2.putText(depth_colored, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colored, "phi-Depth (125 bytes)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            combined = np.hstack([frame, depth_colored])
            cv2.imshow('phi-Depth Demo', combined)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') or key == 83:  # Right arrow
                idx = (idx + 1) % len(image_files)
            elif key == ord('p') or key == 81:  # Left arrow
                idx = (idx - 1) % len(image_files)
            elif key == ord('m'):
                self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f'phi_depth_{timestamp}.png', combined)
                print(f"Saved phi_depth_{timestamp}.png")
        
        cv2.destroyAllWindows()
        return True
    
    def _run_loop(self, cap, source_name: str):
        """Main processing loop for video sources."""
        print(f"\nRunning with {source_name}")
        print("Controls: q=quit, m=colormap, s=save")
        
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            t0 = time.time()
            depth = self.process_frame(frame)
            process_time = (time.time() - t0) * 1000
            
            depth_colored = cv2.applyColorMap(depth, self.colormaps[self.colormap_idx])
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()
            
            info = f"FPS: {fps:.1f} | {process_time:.0f}ms | {self.colormap_names[self.colormap_idx]}"
            cv2.putText(depth_colored, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colored, "phi-Depth (125 bytes)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            combined = np.hstack([frame, depth_colored])
            cv2.imshow('phi-Depth Demo', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f'phi_depth_{timestamp}.png', combined)
                print(f"Saved phi_depth_{timestamp}.png")
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


def main():
    print("=" * 60)
    print("φ-DEPTH DEMO")
    print("=" * 60)
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description='φ-Depth Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--images', action='store_true', help='Use test images')
    parser.add_argument('--weights', type=str, help='Weights file path')
    args = parser.parse_args()
    
    weights_path = Path(args.weights) if args.weights else None
    
    try:
        demo = PhiDepthDemo(weights_path)
        
        if args.video:
            demo.run_video(args.video)
        elif args.images:
            demo.run_images()
        else:
            # Try webcam first, fall back to images
            print("\nTrying webcam...")
            if not demo.run_webcam(args.camera):
                print("Webcam not available, using test images...")
                demo.run_images()
        
        demo.cleanup()
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
