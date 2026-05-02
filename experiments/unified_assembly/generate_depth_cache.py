#!/usr/bin/env python3
"""
Generate depth cache for COCO val2017 images using Depth Anything V2.

This pre-computes depth maps so training can be faster.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    import torch
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    print("Transformers/torch not available")

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


def generate_depth_cache(n_images: int = 500):
    """Generate depth maps for n_images."""
    if not AVAILABLE:
        print("Cannot generate depth - transformers not available")
        return
    
    DEPTH_CACHE_PATH.mkdir(exist_ok=True)
    
    # Load model
    print("Loading Depth Anything V2...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    print("Model loaded!")
    
    # Get images
    image_files = sorted(COCO_VAL_PATH.glob("*.jpg"))[:n_images]
    print(f"Processing {len(image_files)} images...")
    
    processed = 0
    skipped = 0
    
    for i, img_path in enumerate(image_files):
        cache_file = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        
        if cache_file.exists():
            skipped += 1
            continue
        
        try:
            image = Image.open(img_path).convert("RGB")
            result = pipe(image)
            depth = np.array(result["depth"]).astype(np.float32)
            
            # Normalize to 0-1
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            np.save(cache_file, depth)
            processed += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} (new: {processed}, cached: {skipped})")
        
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
    
    print(f"\nDone! Processed {processed} new images, {skipped} already cached.")
    print(f"Total cached: {len(list(DEPTH_CACHE_PATH.glob('*.npy')))}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    generate_depth_cache(n)
