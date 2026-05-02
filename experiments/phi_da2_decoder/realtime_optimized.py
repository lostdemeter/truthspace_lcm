"""
Optimized Real-time φ-Depth
============================

Maximum performance using:
- FP16 (half precision)
- torch.compile (PyTorch 2.0+)
- CUDA φ-decoder

Target: 100+ FPS

Usage:
    python realtime_optimized.py
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))
from phi_cuda import PhiDepthModule


class OptimizedPhiDepth:
    """Maximum performance real-time depth."""
    
    def __init__(self, weights_path: Path = None, use_compile: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load DA2 backbone
        print("Loading DA2 backbone...")
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        
        self.processor = AutoImageProcessor.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf'
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf'
        ).to(self.device)
        self.model.eval()
        
        # Convert to FP16
        print("Converting to FP16...")
        self.model = self.model.half()
        
        # Compile backbone for speed
        if use_compile and hasattr(torch, 'compile'):
            print("Compiling backbone with torch.compile...")
            self.model.backbone = torch.compile(
                self.model.backbone, 
                mode='reduce-overhead',
                fullgraph=False
            )
            self.model.neck = torch.compile(
                self.model.neck,
                mode='reduce-overhead', 
                fullgraph=False
            )
        
        # Load CUDA φ-decoder (stays FP32 for accuracy)
        if weights_path is None:
            weights_path = Path(__file__).parent / 'phi_weights.bin'
        
        print(f"Loading CUDA φ-decoder ({weights_path.stat().st_size} bytes)...")
        self.phi_decoder = PhiDepthModule(weights_path).to(self.device)
        
        # Register hook
        self.captured_features = None
        self._register_hook()
        
        # Colormaps
        self.colormaps = [cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA]
        self.colormap_names = ['magma', 'viridis', 'plasma']
        self.colormap_idx = 0
        
        # Warmup
        print("Warming up...")
        self._warmup()
        
        print("Ready!")
    
    def _register_hook(self):
        def hook(module, input, output):
            self.captured_features = output
        self.hook_handle = self.model.head.activation1.register_forward_hook(hook)
    
    def _warmup(self):
        """Warmup to trigger JIT compilation."""
        dummy = torch.randn(1, 3, 518, 518, device=self.device, dtype=torch.float16)
        for _ in range(5):
            with torch.no_grad():
                # Run full model forward pass for proper warmup
                _ = self.model(dummy)
    
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with maximum speed."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device).half() for k, v in inputs.items()}
        
        # Forward through model
        _ = self.model(**inputs)
        
        # φ-decoder (convert features to FP32)
        features = self.captured_features.squeeze().float()
        depth = self.phi_decoder(features)
        
        # Normalize
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255)
        else:
            depth_norm = torch.zeros_like(depth)
        
        return depth_norm.cpu().numpy().astype(np.uint8)
    
    def run(self, camera_id: int = 0):
        """Run real-time visualization."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nControls: q=quit, s=save, m=colormap")
        print()
        
        fps_window = []
        
        while True:
            t_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            depth = self.process_frame(frame)
            
            depth_colored = cv2.applyColorMap(depth, self.colormaps[self.colormap_idx])
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            
            # FPS
            frame_time = time.perf_counter() - t_start
            fps_window.append(frame_time)
            if len(fps_window) > 30:
                fps_window.pop(0)
            fps = len(fps_window) / sum(fps_window)
            
            # Overlay
            cv2.putText(depth_colored, f"FPS: {fps:.1f} | CUDA+FP16+Compile", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colored, f"phi-Depth (125 bytes)", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            combined = np.hstack([frame, depth_colored])
            cv2.imshow('phi-Depth Optimized', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'phi_opt_{int(time.time())}.png', combined)
            elif key == ord('m'):
                self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
        
        cap.release()
        cv2.destroyAllWindows()
        self.hook_handle.remove()
        
        if fps_window:
            print(f"Average FPS: {len(fps_window) / sum(fps_window):.1f}")


def benchmark():
    """Benchmark optimized pipeline."""
    import time
    from PIL import Image
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    
    print("=" * 70)
    print("OPTIMIZED PIPELINE BENCHMARK")
    print("=" * 70)
    print()
    
    device = torch.device('cuda')
    
    # Load model
    processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').to(device)
    model.eval()
    
    # Load test image
    img_path = Path('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/000000000785.jpg')
    pil_image = Image.open(img_path).convert('RGB')
    inputs = processor(images=pil_image, return_tensors='pt')
    inputs_fp32 = {k: v.to(device) for k, v in inputs.items()}
    
    # φ-decoder
    weights_path = Path(__file__).parent / 'phi_weights.bin'
    phi_decoder = PhiDepthModule(weights_path).to(device)
    
    # Hook
    captured = {}
    def hook(module, input, output):
        captured['feat'] = output
    handle = model.head.activation1.register_forward_hook(hook)
    
    n_runs = 100
    
    # Baseline FP32
    print("1. Baseline (FP32)...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(**inputs_fp32)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(**inputs_fp32)
            depth = phi_decoder(captured['feat'].squeeze())
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - t0) / n_runs * 1000
    
    # FP16
    print("2. FP16...")
    model_fp16 = model.half()
    inputs_fp16 = {k: v.to(device).half() for k, v in inputs.items()}
    
    for _ in range(10):
        with torch.no_grad():
            _ = model_fp16(**inputs_fp16)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model_fp16(**inputs_fp16)
            depth = phi_decoder(captured['feat'].squeeze().float())
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - t0) / n_runs * 1000
    
    # FP16 + Compile
    print("3. FP16 + torch.compile...")
    model_compiled = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').to(device).half()
    model_compiled.eval()
    model_compiled.backbone = torch.compile(model_compiled.backbone, mode='reduce-overhead')
    model_compiled.neck = torch.compile(model_compiled.neck, mode='reduce-overhead')
    
    handle2 = model_compiled.head.activation1.register_forward_hook(hook)
    
    # Warmup compile
    for _ in range(10):
        with torch.no_grad():
            _ = model_compiled(**inputs_fp16)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model_compiled(**inputs_fp16)
            depth = phi_decoder(captured['feat'].squeeze().float())
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - t0) / n_runs * 1000
    
    handle.remove()
    handle2.remove()
    
    print()
    print(f"{'Configuration':<30} {'Time (ms)':>12} {'FPS':>10} {'Speedup':>10}")
    print("-" * 65)
    print(f"{'Baseline (FP32)':<30} {baseline_time:>12.2f} {1000/baseline_time:>10.1f} {'1.0x':>10}")
    print(f"{'FP16':<30} {fp16_time:>12.2f} {1000/fp16_time:>10.1f} {baseline_time/fp16_time:>9.1f}x")
    print(f"{'FP16 + torch.compile':<30} {compiled_time:>12.2f} {1000/compiled_time:>10.1f} {baseline_time/compiled_time:>9.1f}x")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark()
    else:
        print("=" * 60)
        print("φ-DEPTH OPTIMIZED (FP16 + torch.compile)")
        print("=" * 60)
        print()
        
        app = OptimizedPhiDepth(use_compile=not args.no_compile)
        app.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
