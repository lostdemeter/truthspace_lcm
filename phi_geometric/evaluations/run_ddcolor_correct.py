#!/usr/bin/env python3
"""
Run DDColor correctly using the official pipeline.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_ddcolor_correct():
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    from ddcolor.pipeline import ColorizationPipeline
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    model = model.to(DEVICE)
    
    pipeline = ColorizationPipeline(model, input_size=512, device=DEVICE)
    print("DDColor pipeline loaded")
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    # Test images
    test_images = ["000000000285.jpg", "000000000139.jpg", "000000000632.jpg"]
    
    for img_name in test_images:
        img_path = coco_path / img_name
        if not img_path.exists():
            continue
        
        print(f"\nProcessing: {img_name}")
        
        # Load with OpenCV (BGR)
        img_bgr = cv2.imread(str(img_path))
        
        # Run pipeline
        output_bgr = pipeline.process(img_bgr)
        
        # Save
        stem = img_path.stem
        cv2.imwrite(str(output_path / f"{stem}_ddcolor_pipeline.png"), output_bgr)
        
        # Also save grayscale for comparison
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(output_path / f"{stem}_gray_cv.png"), gray)
        
        print(f"  Saved: {stem}_ddcolor_pipeline.png")
    
    print("\nDone!")

if __name__ == "__main__":
    run_ddcolor_correct()
