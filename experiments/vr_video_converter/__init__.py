"""
VR Video Converter - Real-time 2D to VR 180° conversion.

GPU-accelerated video processing for stereoscopic VR content creation.
"""

from .vr_converter import VRConverter
from .gpu_video_encoder import GPUVideoEncoder

__version__ = '1.0.0'
__all__ = ['VRConverter', 'GPUVideoEncoder']
