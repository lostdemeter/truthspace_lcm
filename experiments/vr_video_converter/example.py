#!/usr/bin/env python3
"""
Simple example of using the VR video converter programmatically.
"""

from vr_converter import VRConverter
from gpu_video_encoder import GPUVideoEncoder
import cv2
import cupy as cp


def convert_video_simple(input_path, output_path):
    """
    Simple video conversion example.
    
    Args:
        input_path: Input video file
        output_path: Output VR video file
    """
    # Initialize converter
    converter = VRConverter(use_gpu=True)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Converting {input_path} to VR 180°...")
    print(f"Total frames: {total_frames}")
    
    encoder = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to stereo
        left, right = converter.process_frame(
            frame,
            ipd_mm=64.0,
            depth_scale=0.2,
            output_height=1920,
            return_gpu=True
        )
        
        # Combine views
        output = VRConverter.apply_center_separation_gpu(left, right, 0.0)
        output = cp.clip(output, 0, 255).astype(cp.uint8)
        
        # Initialize encoder on first frame
        if encoder is None:
            h, w = output.shape[:2]
            encoder = GPUVideoEncoder(output_path, w, h, fps)
        
        # Encode
        encoder.encode_frame_gpu(output)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    # Finalize
    if encoder:
        encoder.finalize()
    
    cap.release()
    print(f"✅ Done! Output: {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python example.py input.mp4 output.mp4")
        sys.exit(1)
    
    convert_video_simple(sys.argv[1], sys.argv[2])
