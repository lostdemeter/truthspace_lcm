#!/usr/bin/env python3
"""
Run DDColor directly on real images - no modifications.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_ddcolor():
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    model = model.to(DEVICE)
    print("DDColor loaded")
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    # The bear image
    img_path = coco_path / "000000000285.jpg"
    
    # Load original
    img = np.array(Image.open(img_path).convert("RGB"))
    print(f"Original shape: {img.shape}")
    
    # Resize for DDColor
    img_resized = np.array(Image.fromarray(img).resize((512, 512)))
    
    # To tensor
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    print(f"Input tensor: {img_tensor.shape}, range [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # Run DDColor
    with torch.no_grad():
        output = model(img_tensor)
    
    print(f"Output tensor: {output.shape}, range [{output.min():.3f}, {output.max():.3f}]")
    
    # Convert to numpy
    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    print(f"Output numpy: {output_np.shape}, range [{output_np.min():.3f}, {output_np.max():.3f}]")
    
    # Scale to 0-255
    output_uint8 = (output_np * 255).clip(0, 255).astype(np.uint8)
    print(f"Output uint8: {output_uint8.shape}, range [{output_uint8.min()}, {output_uint8.max()}]")
    
    # Check channels
    print(f"Channels: {output_uint8.shape[-1]}")
    
    # If only 2 channels, it's ab - need to combine with L
    if output_uint8.shape[-1] == 2:
        print("Output is ab channels - need to combine with L")
        # Get grayscale
        gray = 0.299 * img_resized[..., 0] + 0.587 * img_resized[..., 1] + 0.114 * img_resized[..., 2]
        
        # Combine L + ab
        from skimage import color
        lab = np.zeros((512, 512, 3))
        lab[..., 0] = gray / 255.0 * 100  # L channel
        lab[..., 1] = output_np[..., 0] * 128  # a channel (scale from model output)
        lab[..., 2] = output_np[..., 1] * 128  # b channel
        
        rgb = color.lab2rgb(lab)
        output_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        print(f"After LAB conversion: {output_uint8.shape}")
    
    # Resize back
    output_final = np.array(Image.fromarray(output_uint8).resize((img.shape[1], img.shape[0])))
    
    # Save
    Image.fromarray(output_final).save(output_path / "bear_ddcolor_correct.png")
    Image.fromarray(img).save(output_path / "bear_original.png")
    
    # Also save grayscale
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
    Image.fromarray(gray).save(output_path / "bear_gray.png")
    
    print(f"\nSaved: bear_ddcolor_correct.png, bear_original.png, bear_gray.png")

if __name__ == "__main__":
    run_ddcolor()
