"""
Fast Video Encoder using FFmpeg pipe

Optimized for high-throughput VR video encoding:
- Uses FFmpeg subprocess with NVENC or libx264
- Avoids Python-based color conversion
- Minimal memory copies

Performance:
- NVENC: ~70 FPS at 3840x1920
- libx264 ultrafast: ~90 FPS at 3840x1920
"""

import subprocess
import numpy as np
from typing import Optional
import os


class FastVideoEncoder:
    """Fast video encoder using FFmpeg pipe."""
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 24.0,
        bitrate: str = "10M",
        use_nvenc: bool = True,
        preset: str = "p1",  # NVENC: p1-p7, libx264: ultrafast-veryslow
    ):
        """
        Initialize fast video encoder.
        
        Args:
            output_path: Output video file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            bitrate: Target bitrate (e.g., '10M')
            use_nvenc: Use NVIDIA NVENC (True) or libx264 (False)
            preset: Encoding preset (p1=fastest for NVENC, ultrafast for libx264)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.use_nvenc = use_nvenc
        self.preset = preset
        self.frame_count = 0
        
        # Build FFmpeg command
        if use_nvenc:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'h264_nvenc',
                '-preset', preset,
                '-tune', 'll',  # Low latency
                '-b:v', bitrate,
                '-bf', '0',  # No B-frames
                '-g', str(int(fps * 2)),  # GOP size
                output_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast' if preset == 'p1' else preset,
                '-tune', 'zerolatency',
                '-b:v', bitrate,
                output_path
            ]
        
        # Start FFmpeg process
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        encoder_name = 'NVENC' if use_nvenc else 'libx264'
        print(f"FastVideoEncoder initialized ({encoder_name}):")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Bitrate: {bitrate}")
        print(f"  Output: {output_path}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame: BGR frame as numpy array (H, W, 3), uint8
        """
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape mismatch: expected ({self.height}, {self.width}, 3), got {frame.shape}")
        
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Write raw bytes to FFmpeg stdin
        self.proc.stdin.write(frame.tobytes())
        self.frame_count += 1
    
    def close(self, audio_source: Optional[str] = None):
        """
        Close the encoder and finalize the video.
        
        Args:
            audio_source: Optional path to source video to copy audio from
        """
        self.proc.stdin.close()
        self.proc.wait()
        
        print(f"FastVideoEncoder: Wrote {self.frame_count} frames")
        
        # Add audio if requested
        if audio_source and os.path.exists(audio_source):
            self._add_audio(audio_source)
    
    def _add_audio(self, audio_source: str):
        """Add audio from source video to output."""
        # Check if source has audio
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            audio_source
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        if 'audio' not in result.stdout:
            print("  No audio in source video")
            return
        
        # Remux with audio
        temp_path = self.output_path + '.temp.mp4'
        os.rename(self.output_path, temp_path)
        
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_path,
            '-i', audio_source,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            self.output_path
        ]
        
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
        os.unlink(temp_path)
        print("  Added audio from source")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def benchmark():
    """Benchmark the fast video encoder."""
    import time
    import tempfile
    
    print("="*70)
    print("FAST VIDEO ENCODER BENCHMARK")
    print("="*70)
    print()
    
    # Create test frame
    frame = np.random.randint(0, 255, (1920, 3840, 3), dtype=np.uint8)
    n_frames = 100
    
    for use_nvenc in [True, False]:
        encoder_name = 'NVENC' if use_nvenc else 'libx264'
        print(f"\nTesting {encoder_name}...")
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            encoder = FastVideoEncoder(
                output_path=output_path,
                width=3840,
                height=1920,
                fps=24.0,
                bitrate='10M',
                use_nvenc=use_nvenc
            )
            
            # Warmup
            for _ in range(10):
                encoder.write_frame(frame)
            
            # Benchmark
            t0 = time.perf_counter()
            for _ in range(n_frames):
                encoder.write_frame(frame)
            elapsed = time.perf_counter() - t0
            
            encoder.close()
            
            print(f"  Encoded {n_frames} frames in {elapsed:.2f}s")
            print(f"  Per-frame: {elapsed/n_frames*1000:.2f} ms")
            print(f"  FPS: {n_frames/elapsed:.1f}")
            print(f"  File size: {os.path.getsize(output_path)/1024/1024:.1f} MB")
            
        finally:
            os.unlink(output_path)


if __name__ == "__main__":
    benchmark()
