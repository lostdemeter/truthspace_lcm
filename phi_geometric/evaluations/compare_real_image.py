#!/usr/bin/env python3
"""
Compare Minimal Colorizer vs DDColor on Real Images
"""

import numpy as np
from PIL import Image
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.evaluations.minimal_colorizer import colorize, lab_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_comparison():
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    # Load DDColor
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        ddcolor.eval()
        ddcolor = ddcolor.to(DEVICE)
        print("DDColor loaded")
    except Exception as e:
        print(f"DDColor not available: {e}")
        return
    
    # Test image
    img_path = coco_path / "000000000139.jpg"
    img = np.array(Image.open(img_path).convert("RGB"))
    
    # DDColor
    img_resized = np.array(Image.fromarray(img).resize((512, 512)))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = ddcolor(img_tensor)
    
    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
    
    # Ensure 3 channels
    if output_np.shape[-1] == 2:
        output_np = np.concatenate([output_np, np.zeros((*output_np.shape[:-1], 1), dtype=np.uint8)], axis=-1)
    
    ddcolor_result = np.array(Image.fromarray(output_np).resize((256, 256)))
    Image.fromarray(ddcolor_result).save(output_path / "000000000139_ddcolor.png")
    print(f"Saved DDColor result")
    
    # Create side-by-side comparison
    gray = np.array(Image.open(output_path / "000000000139_gray.png").resize((256, 256)))
    minimal = np.array(Image.open(output_path / "000000000139_minimal.png"))
    original = np.array(Image.open(output_path / "000000000139_original.png"))
    
    # Stack: gray | minimal | ddcolor | original
    if gray.ndim == 2:
        gray = np.stack([gray] * 3, axis=-1)
    
    comparison = np.concatenate([gray, minimal, ddcolor_result, original], axis=1)
    Image.fromarray(comparison).save(output_path / "real_image_comparison.png")
    print(f"Saved comparison")

if __name__ == "__main__":
    run_comparison()
