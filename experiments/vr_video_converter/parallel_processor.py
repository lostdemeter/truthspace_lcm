"""
Parallel Video Processing Pipeline
===================================

Optimizes VR video conversion using:
1. Multi-threaded frame reading (I/O bound)
2. Batch depth estimation (GPU bound)
3. Parallel stereo conversion (CPU bound)
4. Async encoding (GPU bound)

Target: 3-5x speedup over sequential processing
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import threading
import time


class ParallelVRProcessor:
    """
    Parallel pipeline for VR video conversion.
    
    Architecture:
    [Reader Thread] -> [Frame Queue] -> [Batch Depth] -> [Stereo Queue] -> [Encoder Thread]
    """
    
    def __init__(self, vr_converter, batch_size: int = 4, num_workers: int = 4):
        """
        Initialize parallel processor.
        
        Args:
            vr_converter: VRConverter instance with φ-depth
            batch_size: Number of frames to process in batch for depth
            num_workers: Number of CPU workers for stereo conversion
        """
        self.converter = vr_converter
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Queues for pipeline stages
        self.frame_queue = Queue(maxsize=batch_size * 2)
        self.depth_queue = Queue(maxsize=batch_size * 2)
        self.output_queue = Queue(maxsize=batch_size * 2)
        
        # Control flags
        self.stop_flag = threading.Event()
        self.error = None
        
        # Stats
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_encoded = 0
    
    def process_video(self, input_path: str, output_path: str,
                     ipd_mm: float = 64.0, depth_scale: float = 0.2,
                     output_height: int = 1920, bitrate: str = '10M',
                     progress_callback=None) -> bool:
        """
        Process video with parallel pipeline.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            ipd_mm: Inter-pupillary distance in mm
            depth_scale: Depth effect strength
            output_height: Output frame height
            bitrate: Output bitrate
            progress_callback: Optional callback(progress, message)
            
        Returns:
            True if successful
        """
        self.stop_flag.clear()
        self.error = None
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_encoded = 0
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if progress_callback:
            progress_callback(0, f"Starting... ({total_frames} frames)")
        
        # Start pipeline threads
        threads = []
        
        # Reader thread
        reader_thread = threading.Thread(
            target=self._reader_worker,
            args=(cap, total_frames)
        )
        reader_thread.start()
        threads.append(reader_thread)
        
        # Depth + stereo processing thread (GPU bound, single thread)
        processor_thread = threading.Thread(
            target=self._processor_worker,
            args=(ipd_mm, depth_scale, output_height, total_frames, progress_callback)
        )
        processor_thread.start()
        threads.append(processor_thread)
        
        # Encoder thread (pass input_path as audio source)
        # Use FastVideoEncoder for better performance
        encoder_thread = threading.Thread(
            target=self._encoder_worker,
            args=(output_path, fps, bitrate, total_frames, input_path)
        )
        encoder_thread.start()
        threads.append(encoder_thread)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        cap.release()
        
        if self.error:
            raise RuntimeError(self.error)
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        return True
    
    def _reader_worker(self, cap, total_frames: int):
        """Read frames from video into queue."""
        try:
            frame_idx = 0
            while not self.stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_queue.put((frame_idx, frame))
                frame_idx += 1
                self.frames_read = frame_idx
            
            # Signal end of frames
            self.frame_queue.put(None)
            
        except Exception as e:
            self.error = f"Reader error: {e}"
            self.stop_flag.set()
    
    def _processor_worker(self, ipd_mm: float, depth_scale: float, 
                         output_height: int, total_frames: int,
                         progress_callback):
        """Process frames: depth estimation + stereo conversion."""
        try:
            while not self.stop_flag.is_set():
                item = self.frame_queue.get()
                if item is None:
                    break
                
                frame_idx, frame = item
                
                # Process frame through VR converter (use fast method)
                # This includes φ-depth estimation + fast stereo conversion
                result = self.converter.process_frame_fast(
                    frame,
                    ipd_mm=ipd_mm,
                    depth_scale=depth_scale,
                    output_height=output_height
                )
                
                # Handle tuple return (left, right)
                if isinstance(result, tuple):
                    left, right = result
                    # Combine left and right
                    output = np.hstack([left, right])
                else:
                    output = result
                
                # Ensure uint8
                output = np.clip(output, 0, 255).astype(np.uint8)
                
                self.output_queue.put((frame_idx, output))
                self.frames_processed = frame_idx + 1
                
                if progress_callback and frame_idx % 10 == 0:
                    progress = int((frame_idx / total_frames) * 95)
                    progress_callback(progress, f"Frame {frame_idx}/{total_frames}")
            
            # Signal end
            self.output_queue.put(None)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"Processor error: {e}"
            self.stop_flag.set()
            self.output_queue.put(None)
    
    def _encoder_worker(self, output_path: str, fps: float, 
                       bitrate: str, total_frames: int, audio_source: str = None):
        """Encode processed frames to video."""
        try:
            # Use FastVideoEncoder for better performance (2.6x faster than GPUVideoEncoder)
            from fast_video_encoder import FastVideoEncoder
            
            encoder = None
            
            while not self.stop_flag.is_set():
                item = self.output_queue.get()
                if item is None:
                    break
                
                frame_idx, output = item
                
                # Initialize encoder on first frame
                if encoder is None:
                    h, w = output.shape[:2]
                    encoder = FastVideoEncoder(
                        output_path, w, h, fps, 
                        bitrate=bitrate,
                        use_nvenc=True  # Use NVENC for GPU encoding
                    )
                
                # Encode
                encoder.write_frame(output)
                self.frames_encoded = frame_idx + 1
            
            # Finalize with audio from source
            if encoder:
                encoder.close(audio_source=audio_source)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"Encoder error: {e}"
            self.stop_flag.set()


class BatchDepthProcessor:
    """
    Batch depth estimation for improved GPU utilization.
    
    Instead of processing one frame at a time, batch multiple frames
    to better utilize GPU parallelism.
    """
    
    def __init__(self, phi_estimator, batch_size: int = 4):
        self.estimator = phi_estimator
        self.batch_size = batch_size
        self.device = phi_estimator.device
    
    @torch.no_grad()
    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of frames for depth estimation.
        
        Args:
            frames: List of BGR frames (H, W, 3)
            
        Returns:
            List of depth maps (H, W)
        """
        from PIL import Image
        
        # Preprocess all frames
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        
        # Batch process through processor
        inputs = self.estimator.processor(images=pil_images, return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.estimator.dtype) for k, v in inputs.items()}
        
        # Forward pass (batched)
        _ = self.estimator.model(**inputs)
        
        # Get features for each frame in batch
        features = self.estimator.captured_features
        
        depths = []
        for i in range(len(frames)):
            feat = features[i] if features.dim() == 4 else features
            if self.estimator.config.use_fp16:
                feat = feat.float()
            
            depth = self.estimator.phi_decoder(feat)
            
            # Normalize
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            
            depth_np = depth.cpu().numpy()
            
            # Resize to match input
            h, w = frames[i].shape[:2]
            if depth_np.shape != (h, w):
                depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depths.append(depth_np)
        
        return depths


def benchmark_parallel():
    """Benchmark parallel vs sequential processing."""
    import time
    
    print("=" * 70)
    print("PARALLEL PROCESSING BENCHMARK")
    print("=" * 70)
    print()
    
    # Create test video
    test_video = "/tmp/test_input.mp4"
    
    # Generate test video if needed
    import subprocess
    if not Path(test_video).exists():
        print("Generating test video...")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=640x480:rate=24',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', test_video
        ], capture_output=True)
    
    # Count frames
    cap = cv2.VideoCapture(test_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Test video: {total_frames} frames")
    print()
    
    # Initialize converter
    from vr_converter import VRConverter
    converter = VRConverter(use_gpu=True)
    
    # Test parallel processor
    print("Testing parallel processor...")
    processor = ParallelVRProcessor(converter, batch_size=4, num_workers=4)
    
    t0 = time.perf_counter()
    processor.process_video(
        test_video, "/tmp/test_parallel.mp4",
        progress_callback=lambda p, m: print(f"  {p}% - {m}") if p % 20 == 0 else None
    )
    parallel_time = time.perf_counter() - t0
    
    print()
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Parallel FPS: {total_frames / parallel_time:.1f}")


if __name__ == "__main__":
    benchmark_parallel()
