# VR Video Converter

Real-time 2D to VR 180° video conversion with GPU acceleration.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Convert video (CLI)
python convert_video.py input.mp4 output_vr.mp4

# Or use web interface
python web_server.py
# Open http://localhost:5000
```

## Performance

- **36+ FPS** real-time processing
- **1.5× faster** than playback
- **GPU-accelerated** depth estimation and encoding

## Usage

### Command Line

```bash
# Basic
python convert_video.py video.mp4 vr_video.mp4

# High quality
python convert_video.py video.mp4 vr_video.mp4 --resolution 3840x1920 --bitrate 20M

# Stronger 3D effect
python convert_video.py video.mp4 vr_video.mp4 --depth-scale 0.3 --ipd 70
```

### Python API

```python
from vr_converter import VRConverter
from gpu_video_encoder import GPUVideoEncoder
import cv2
import cupy as cp

converter = VRConverter(use_gpu=True)
cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

encoder = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    left, right = converter.process_frame(
        frame, ipd_mm=64.0, depth_scale=0.2,
        output_height=1920, return_gpu=True
    )
    
    output = VRConverter.apply_center_separation_gpu(left, right, 0.0)
    output = cp.clip(output, 0, 255).astype(cp.uint8)
    
    if encoder is None:
        h, w = output.shape[:2]
        encoder = GPUVideoEncoder('output.mp4', w, h, fps)
    
    encoder.encode_frame_gpu(output)

encoder.finalize()
cap.release()
```

## Parameters

- `--ipd`: Eye distance in mm (default: 64)
- `--depth-scale`: 3D effect strength 0-1 (default: 0.2)
- `--resolution`: Output size (default: 3840x1920)
- `--bitrate`: Video bitrate (default: 10M)

## Test

```bash
python test_simple.py
# Creates test_output.mp4
```

## Requirements

- NVIDIA GPU with NVENC (GTX 1050+)
- CUDA 12.0+
- Python 3.8+
- 8GB+ GPU memory

## Files

- `convert_video.py` - CLI tool
- `web_server.py` - Web interface
- `vr_converter.py` - Core conversion
- `gpu_video_encoder.py` - NVENC encoding
- `fast_depth_estimation.py` - Depth estimation
- `example.py` - Simple API example
- `test_simple.py` - Quick test

## Output Format

- **Format:** MP4 (H.264)
- **Layout:** Side-by-side stereoscopic
- **Projection:** Equirectangular 180°
- **Compatible:** Oculus Quest, Meta Quest, PSVR2, etc.
