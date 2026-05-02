"""
Pipeline Profiler
==================

Profiles each step of the VR video conversion to identify bottlenecks.
"""

import cv2
import numpy as np
import time
from pathlib import Path


def profile_video_pipeline(video_path: str, num_frames: int = 50):
    """Profile each step of the pipeline."""
    
    print("=" * 70)
    print("VR VIDEO PIPELINE PROFILER")
    print("=" * 70)
    print()
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Profiling {num_frames} frames")
    print()
    
    # Read some frames first
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("ERROR: Could not read frames")
        return
    
    print(f"Read {len(frames)} frames for profiling")
    print()
    
    timings = {
        'video_read': [],
        'phi_depth_preprocess': [],
        'phi_depth_backbone': [],
        'phi_depth_decoder': [],
        'phi_depth_postprocess': [],
        'phi_depth_total': [],
        'stereo_conversion': [],
        'encoding': [],
    }
    
    # ========================================
    # Step 1: Video Reading
    # ========================================
    print("1. Profiling VIDEO READING...")
    cap = cv2.VideoCapture(video_path)
    for _ in range(num_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        timings['video_read'].append(time.perf_counter() - t0)
    cap.release()
    
    # ========================================
    # Step 2: φ-Depth Estimation (breakdown)
    # ========================================
    print("2. Profiling φ-DEPTH ESTIMATION...")
    
    import torch
    from PIL import Image
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').to(device)
    model.eval()
    model = model.half()
    
    # Load φ-decoder
    from phi_depth_estimation import PhiDepthModule
    weights_path = Path(__file__).parent.parent / 'phi_da2_decoder' / 'phi_weights.bin'
    phi_decoder = PhiDepthModule(weights_path).to(device)
    
    # Hook for features
    captured_features = [None]
    def hook(module, input, output):
        captured_features[0] = output
    handle = model.head.activation1.register_forward_hook(hook)
    
    # Warmup
    print("   Warming up...")
    for _ in range(5):
        pil = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(device).half() for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    
    torch.cuda.synchronize()
    
    # Profile each sub-step
    print("   Profiling sub-steps...")
    for frame in frames[:num_frames]:
        # Preprocess
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(device).half() for k, v in inputs.items()}
        torch.cuda.synchronize()
        timings['phi_depth_preprocess'].append(time.perf_counter() - t0)
        
        # Backbone forward
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        timings['phi_depth_backbone'].append(time.perf_counter() - t0)
        
        # φ-decoder
        t0 = time.perf_counter()
        features = captured_features[0].squeeze().float()
        depth = phi_decoder(features)
        torch.cuda.synchronize()
        timings['phi_depth_decoder'].append(time.perf_counter() - t0)
        
        # Postprocess (normalize + resize)
        t0 = time.perf_counter()
        depth_min, depth_max = depth.min(), depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth_np = depth.cpu().numpy()
        depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_LINEAR)
        timings['phi_depth_postprocess'].append(time.perf_counter() - t0)
    
    handle.remove()
    
    # Total depth time
    for i in range(len(timings['phi_depth_preprocess'])):
        total = (timings['phi_depth_preprocess'][i] + 
                timings['phi_depth_backbone'][i] + 
                timings['phi_depth_decoder'][i] + 
                timings['phi_depth_postprocess'][i])
        timings['phi_depth_total'].append(total)
    
    # ========================================
    # Step 3: Stereo Conversion
    # ========================================
    print("3. Profiling STEREO CONVERSION...")
    
    from vr_converter import VRConverter, DEPTH_METHOD
    
    # Create converter but skip depth estimator init
    class DummyDepthEstimator:
        def estimate_depth(self, frame, **kwargs):
            return np.random.rand(frame.shape[0], frame.shape[1]).astype(np.float32)
    
    # Use the actual converter with φ-depth for accurate profiling
    converter = VRConverter(use_gpu=False)
    
    # Profile stereo conversion using the FAST method
    for frame in frames[:num_frames]:
        t0 = time.perf_counter()
        # Use process_frame_fast which uses cv2.remap
        result = converter.process_frame_fast(frame, output_height=1920)
        timings['stereo_conversion'].append(time.perf_counter() - t0)
    
    
    # ========================================
    # Step 4: Video Encoding
    # ========================================
    print("4. Profiling VIDEO ENCODING...")
    
    from gpu_video_encoder import GPUVideoEncoder
    
    # Create test output
    test_output = "/tmp/profile_test.mp4"
    encoder = GPUVideoEncoder(test_output, 3840, 1920, 24)
    
    # Create dummy stereo frame
    stereo_frame = np.random.randint(0, 255, (1920, 3840, 3), dtype=np.uint8)
    
    for _ in range(num_frames):
        t0 = time.perf_counter()
        encoder.encode_frame_gpu(stereo_frame)
        timings['encoding'].append(time.perf_counter() - t0)
    
    encoder.finalize()
    
    # ========================================
    # Results
    # ========================================
    print()
    print("=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    print()
    
    def stats(times):
        if not times:
            return 0, 0, 0
        return np.mean(times) * 1000, np.std(times) * 1000, np.median(times) * 1000
    
    print(f"{'Step':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'Median (ms)':>12} {'FPS':>8}")
    print("-" * 74)
    
    total_time = 0
    for step, times in timings.items():
        if times:
            mean, std, median = stats(times)
            fps = 1000 / mean if mean > 0 else 0
            print(f"{step:<30} {mean:>12.2f} {std:>12.2f} {median:>12.2f} {fps:>8.1f}")
            if step not in ['phi_depth_total']:  # Don't double count
                if step.startswith('phi_depth_') and step != 'phi_depth_total':
                    pass  # Sub-steps, counted in total
                else:
                    total_time += mean
    
    print("-" * 74)
    
    # Calculate actual total per frame
    actual_total = (stats(timings['phi_depth_total'])[0] + 
                   stats(timings['stereo_conversion'])[0] + 
                   stats(timings['encoding'])[0])
    
    print(f"{'TOTAL PER FRAME':<30} {actual_total:>12.2f} {'-':>12} {'-':>12} {1000/actual_total:>8.1f}")
    print()
    
    # Breakdown
    print("BOTTLENECK ANALYSIS:")
    print("-" * 40)
    
    phi_total = stats(timings['phi_depth_total'])[0]
    stereo_total = stats(timings['stereo_conversion'])[0]
    encode_total = stats(timings['encoding'])[0]
    
    print(f"  φ-Depth estimation: {phi_total:.1f}ms ({phi_total/actual_total*100:.1f}%)")
    print(f"    - Preprocess:     {stats(timings['phi_depth_preprocess'])[0]:.1f}ms")
    print(f"    - Backbone:       {stats(timings['phi_depth_backbone'])[0]:.1f}ms")
    print(f"    - φ-Decoder:      {stats(timings['phi_depth_decoder'])[0]:.1f}ms")
    print(f"    - Postprocess:    {stats(timings['phi_depth_postprocess'])[0]:.1f}ms")
    print(f"  Stereo conversion:  {stereo_total:.1f}ms ({stereo_total/actual_total*100:.1f}%)")
    print(f"  Video encoding:     {encode_total:.1f}ms ({encode_total/actual_total*100:.1f}%)")
    print()
    
    # Time estimate for full video
    time_per_frame_s = actual_total / 1000
    total_video_time = total_frames * time_per_frame_s
    print(f"ESTIMATED TIME FOR FULL VIDEO ({total_frames} frames):")
    print(f"  {total_video_time:.1f} seconds = {total_video_time/60:.1f} minutes")
    print()
    
    return timings


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test1.mp4"
    profile_video_pipeline(video_path, num_frames=30)
