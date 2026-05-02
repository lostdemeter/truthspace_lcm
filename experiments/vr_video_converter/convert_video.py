#!/usr/bin/env python3
"""
Command-line interface for VR video conversion.

Converts 2D videos to VR 180° stereoscopic format with GPU acceleration.
"""

import argparse
import cv2
import time
import cupy as cp
from pathlib import Path

from vr_converter import VRConverter
from gpu_video_encoder import GPUVideoEncoder


def convert_video(input_path, output_path, ipd_mm=64.0, depth_scale=0.2,
                 resolution='3840x1920', bitrate='10M', gpu_id=0):
    """
    Convert 2D video to VR 180° format.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path
        ipd_mm: Interpupillary distance in millimeters
        depth_scale: Depth effect strength (0.0-1.0)
        resolution: Output resolution (WxH)
        bitrate: Target bitrate (e.g., '10M')
        gpu_id: GPU device ID
    """
    print("="*70)
    print("VR Video Converter - Real-Time GPU Processing")
    print("="*70)
    
    # Parse resolution
    if resolution == 'auto':
        output_height = None  # Will be set from input
    else:
        width, height = map(int, resolution.split('x'))
        output_height = height
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_height is None:
        output_height = height * 2  # Default: 2x input height
    
    print(f"\nInput Video:")
    print(f"  File: {input_path}")
    print(f"  Resolution: {width}×{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    
    print(f"\nOutput Settings:")
    print(f"  Resolution: {output_height*2}×{output_height} (VR 180°)")
    print(f"  IPD: {ipd_mm}mm")
    print(f"  Depth Scale: {depth_scale}")
    print(f"  Bitrate: {bitrate}")
    
    # Initialize converter
    converter = VRConverter(use_gpu=True)
    
    # Initialize encoder (will be created on first frame)
    encoder = None
    
    # Process frames
    print(f"\nProcessing...")
    start_time = time.time()
    frame_times = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Process frame to stereo
            left_view, right_view = converter.process_frame(
                frame,
                ipd_mm=ipd_mm,
                depth_scale=depth_scale,
                output_height=output_height,
                return_gpu=True
            )
            
            # Combine views
            output_frame = VRConverter.apply_center_separation_gpu(left_view, right_view, 0.0)
            
            # Convert to uint8
            output_frame = cp.clip(output_frame, 0, 255).astype(cp.uint8)
            
            # Initialize encoder on first frame
            if encoder is None:
                out_h, out_w = output_frame.shape[:2]
                encoder = GPUVideoEncoder(
                    output_path,
                    out_w,
                    out_h,
                    fps,
                    gpu_id=gpu_id,
                    bitrate=bitrate
                )
                print(f"  Encoder initialized: {out_w}×{out_h} @ {fps} FPS")
            
            # Encode frame
            encoder.encode_frame_gpu(output_frame)
            
            frame_count += 1
            frame_time = (time.time() - frame_start) * 1000
            frame_times.append(frame_time)
            
            # Progress update
            if frame_count % 30 == 0 or frame_count == total_frames:
                avg_time = sum(frame_times[-30:]) / min(30, len(frame_times))
                avg_fps = 1000 / avg_time
                progress = (frame_count / total_frames) * 100
                eta = (total_frames - frame_count) / avg_fps
                print(f"  [{progress:5.1f}%] Frame {frame_count:4d}/{total_frames} | "
                      f"{avg_fps:5.1f} FPS | ETA: {eta:5.1f}s")
        
        # Finalize encoder
        if encoder:
            print(f"\nFinalizing...")
            encoder.finalize()
        
    finally:
        cap.release()
    
    # Statistics
    total_time = time.time() - start_time
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
    
    print(f"\n{'='*70}")
    print("Conversion Complete!")
    print(f"{'='*70}")
    print(f"Frames processed: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average frame time: {avg_frame_time:.1f}ms")
    print(f"Real-time speed: {(total_frames/fps)/total_time:.2f}×")
    print(f"\nOutput: {output_path}")
    
    # Verify output
    import subprocess
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames',
        '-of', 'default=noprint_wrappers=1',
        output_path
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ Video verified successfully!")
        print(result.stdout)
    else:
        print(f"\n⚠️  Could not verify video (but it may still be valid)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert 2D video to VR 180° stereoscopic format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_video.py input.mp4 output.mp4
  
  # High quality
  python convert_video.py input.mp4 output.mp4 --resolution 3840x1920 --bitrate 20M
  
  # Stronger depth effect
  python convert_video.py input.mp4 output.mp4 --depth-scale 0.3 --ipd 70
  
  # Fast preview
  python convert_video.py input.mp4 output.mp4 --resolution 1920x960 --bitrate 5M
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output video file')
    parser.add_argument('--ipd', type=float, default=64.0,
                       help='Interpupillary distance in mm (default: 64)')
    parser.add_argument('--depth-scale', type=float, default=0.2,
                       help='Depth effect strength 0.0-1.0 (default: 0.2)')
    parser.add_argument('--resolution', default='3840x1920',
                       help='Output resolution WxH (default: 3840x1920)')
    parser.add_argument('--bitrate', default='10M',
                       help='Target bitrate (default: 10M)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Convert video
    try:
        convert_video(
            args.input,
            args.output,
            ipd_mm=args.ipd,
            depth_scale=args.depth_scale,
            resolution=args.resolution,
            bitrate=args.bitrate,
            gpu_id=args.gpu
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
