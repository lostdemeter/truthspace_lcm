#!/usr/bin/env python3
"""
Simple test to verify the VR converter works.
Creates a short test video.
"""

import numpy as np
import cv2
import cupy as cp
from vr_converter import VRConverter
from gpu_video_encoder import GPUVideoEncoder

# Create test video
print("Creating test video...")
width, height = 640, 480
fps = 24
num_frames = 48  # 2 seconds

# Initialize converter
converter = VRConverter(use_gpu=True)
encoder = None

for i in range(num_frames):
    # Create test frame
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i+1}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Convert to stereo
    left, right = converter.process_frame(
        frame, 
        ipd_mm=64.0,
        depth_scale=0.2,
        output_height=960,
        return_gpu=True
    )
    
    # Combine
    output = VRConverter.apply_center_separation_gpu(left, right, 0.0)
    output = cp.clip(output, 0, 255).astype(cp.uint8)
    
    # Initialize encoder
    if encoder is None:
        h, w = output.shape[:2]
        encoder = GPUVideoEncoder('test_output.mp4', w, h, fps)
        print(f"Encoder: {w}x{h} @ {fps} FPS")
    
    # Encode
    encoder.encode_frame_gpu(output)
    
    if (i + 1) % 12 == 0:
        print(f"  Processed {i+1}/{num_frames} frames")

# Finalize
encoder.finalize()
print("✅ Done! Check test_output.mp4")
