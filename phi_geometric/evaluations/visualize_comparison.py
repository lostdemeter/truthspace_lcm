#!/usr/bin/env python3
"""
Visualize Colorization Comparison

Creates a side-by-side comparison of:
- Grayscale input
- V3 Chemistry output
- DDColor output (if available)

Saves to a PNG file that can be viewed.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import DDColor
DDCOLOR_AVAILABLE = False
try:
    sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    DDCOLOR_AVAILABLE = True
except ImportError:
    pass

from phi_geometric.evaluations.colorizer_v3_chemistry import ChemistryColorizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lab_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB."""
    try:
        from skimage import color
        lab = np.zeros((*L.shape, 3))
        lab[..., 0] = L * 100
        lab[..., 1] = np.clip(ab[..., 0], -128, 128)
        lab[..., 2] = np.clip(ab[..., 1], -128, 128)
        rgb = color.lab2rgb(lab)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    except ImportError:
        # Fallback
        rgb = np.zeros((*L.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip((L * 255).astype(int), 0, 255)
        rgb[..., 1] = np.clip((L * 255).astype(int), 0, 255)
        rgb[..., 2] = np.clip((L * 255).astype(int), 0, 255)
        return rgb


def create_test_images():
    """Create test images with semantic regions."""
    images = []
    
    # 1. Landscape (sky + ground)
    landscape = np.zeros((128, 128))
    sky_mask = np.zeros((128, 128), dtype=bool)
    ground_mask = np.zeros((128, 128), dtype=bool)
    
    for i in range(128):
        if i < 56:
            landscape[i, :] = 0.75 - 0.15 * i / 56
            sky_mask[i, :] = True
        elif i < 64:
            landscape[i, :] = 0.60
        else:
            landscape[i, :] = 0.30 + 0.20 * (i - 64) / 64
            ground_mask[i, :] = True
    
    images.append(("landscape", landscape, {"sky": sky_mask, "vegetation": ground_mask}))
    
    # 2. Beach (sky + water + sand)
    beach = np.zeros((128, 128))
    sky_mask = np.zeros((128, 128), dtype=bool)
    water_mask = np.zeros((128, 128), dtype=bool)
    sand_mask = np.zeros((128, 128), dtype=bool)
    
    for i in range(128):
        if i < 40:
            beach[i, :] = 0.80 - 0.10 * i / 40
            sky_mask[i, :] = True
        elif i < 80:
            beach[i, :] = 0.50 + 0.10 * np.sin((i - 40) * 0.15)
            water_mask[i, :] = True
        else:
            beach[i, :] = 0.65 + 0.10 * (i - 80) / 48
            sand_mask[i, :] = True
    
    images.append(("beach", beach, {"sky": sky_mask, "water": water_mask, "earth": sand_mask}))
    
    # 3. Portrait
    portrait = np.zeros((128, 128))
    skin_mask = np.zeros((128, 128), dtype=bool)
    
    for i in range(128):
        for j in range(128):
            dist = np.sqrt((i - 64)**2 + (j - 64)**2)
            if dist < 40:
                portrait[i, j] = 0.55 + 0.20 * (1 - dist / 40)
                skin_mask[i, j] = True
            else:
                portrait[i, j] = 0.25
    
    images.append(("portrait", portrait, {"skin": skin_mask}))
    
    # 4. Forest
    forest = np.zeros((128, 128))
    sky_mask = np.zeros((128, 128), dtype=bool)
    foliage_mask = np.zeros((128, 128), dtype=bool)
    
    for i in range(128):
        for j in range(128):
            if i < 30:
                forest[i, j] = 0.70
                sky_mask[i, j] = True
            else:
                noise = np.sin(i * 0.2) * np.cos(j * 0.25) * 0.15
                forest[i, j] = 0.35 + noise + np.random.rand() * 0.08
                foliage_mask[i, j] = True
    
    images.append(("forest", forest, {"sky": sky_mask, "vegetation": foliage_mask}))
    
    return images


def create_visualization():
    """Create visualization of all colorizers."""
    print("=" * 70)
    print("COLORIZATION VISUALIZATION")
    print("=" * 70)
    
    # Initialize V3 Chemistry
    v3 = ChemistryColorizer()
    
    # Initialize DDColor if available
    ddcolor_model = None
    if DDCOLOR_AVAILABLE:
        try:
            class DDColorHF(DDColor, PyTorchModelHubMixin):
                def __init__(self, config=None, **kwargs):
                    if isinstance(config, dict):
                        kwargs = {**config, **kwargs}
                    super().__init__(**kwargs)
            
            ddcolor_model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
            ddcolor_model.eval()
            ddcolor_model = ddcolor_model.to(DEVICE)
            print("DDColor loaded")
        except Exception as e:
            print(f"DDColor not available: {e}")
    
    # Test images
    test_images = create_test_images()
    
    # Output directory
    output_dir = Path(__file__).parent / "visualization"
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison for each image
    for name, gray, semantic_map in test_images:
        print(f"\nProcessing {name}...")
        
        # V3 Chemistry
        ab_v3 = v3.colorize(gray, semantic_map)
        rgb_v3 = lab_to_rgb(gray, ab_v3)
        
        # V3 with sunset reaction (for landscape/beach)
        if name in ["landscape", "beach"]:
            ab_v3_sunset = v3.apply_reaction("Sunset", ab_v3, strength=0.5)
            rgb_v3_sunset = lab_to_rgb(gray, ab_v3_sunset)
        else:
            rgb_v3_sunset = rgb_v3
        
        # DDColor
        if ddcolor_model is not None:
            try:
                gray_rgb = np.stack([gray * 255] * 3, axis=-1).astype(np.uint8)
                gray_rgb_resized = np.array(Image.fromarray(gray_rgb).resize((512, 512)))
                
                img_tensor = torch.from_numpy(gray_rgb_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = ddcolor_model(img_tensor)
                
                output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
                output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
                # Ensure 3 channels
                if output_np.ndim == 2:
                    output_np = np.stack([output_np] * 3, axis=-1)
                elif output_np.shape[-1] == 2:
                    output_np = np.concatenate([output_np, np.zeros((*output_np.shape[:-1], 1), dtype=np.uint8)], axis=-1)
                elif output_np.shape[-1] != 3:
                    output_np = output_np[..., :3]
                rgb_dd = np.array(Image.fromarray(output_np).resize((128, 128)))
            except Exception as e:
                print(f"  DDColor error: {e}")
                rgb_dd = np.zeros_like(rgb_v3)
        else:
            rgb_dd = np.zeros_like(rgb_v3)
        
        # Create comparison image
        # Layout: [Gray] [V3] [V3+Sunset] [DDColor]
        gray_rgb = np.stack([gray * 255] * 3, axis=-1).astype(np.uint8)
        
        # Add labels
        comparison = np.zeros((128 + 20, 128 * 4 + 30, 3), dtype=np.uint8)
        comparison[:, :, :] = 255  # White background
        
        # Place images
        comparison[20:148, 0:128, :] = gray_rgb
        comparison[20:148, 138:266, :] = rgb_v3
        comparison[20:148, 276:404, :] = rgb_v3_sunset
        comparison[20:148, 414:542, :] = rgb_dd
        
        # Save
        Image.fromarray(comparison).save(output_dir / f"{name}_comparison.png")
        print(f"  Saved: {name}_comparison.png")
        
        # Also save individual images
        Image.fromarray(gray_rgb).save(output_dir / f"{name}_gray.png")
        Image.fromarray(rgb_v3).save(output_dir / f"{name}_v3.png")
        Image.fromarray(rgb_v3_sunset).save(output_dir / f"{name}_v3_sunset.png")
        if ddcolor_model is not None:
            Image.fromarray(rgb_dd).save(output_dir / f"{name}_ddcolor.png")
    
    # Create a combined grid
    print("\nCreating combined grid...")
    
    grid_rows = len(test_images)
    grid_cols = 4
    cell_size = 128
    padding = 10
    label_height = 25
    
    grid_width = grid_cols * cell_size + (grid_cols + 1) * padding
    grid_height = grid_rows * cell_size + (grid_rows + 1) * padding + label_height
    
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Column labels
    labels = ["Grayscale", "V3 Chemistry", "V3 + Sunset", "DDColor"]
    
    for row_idx, (name, gray, semantic_map) in enumerate(test_images):
        # Load saved images
        gray_rgb = np.array(Image.open(output_dir / f"{name}_gray.png"))
        rgb_v3 = np.array(Image.open(output_dir / f"{name}_v3.png"))
        rgb_v3_sunset = np.array(Image.open(output_dir / f"{name}_v3_sunset.png"))
        
        if (output_dir / f"{name}_ddcolor.png").exists():
            rgb_dd = np.array(Image.open(output_dir / f"{name}_ddcolor.png"))
        else:
            rgb_dd = np.zeros_like(gray_rgb)
        
        images_row = [gray_rgb, rgb_v3, rgb_v3_sunset, rgb_dd]
        
        for col_idx, img in enumerate(images_row):
            y = label_height + row_idx * (cell_size + padding) + padding
            x = col_idx * (cell_size + padding) + padding
            
            # Resize if needed
            if img.shape[0] != cell_size or img.shape[1] != cell_size:
                img = np.array(Image.fromarray(img).resize((cell_size, cell_size)))
            
            grid[y:y+cell_size, x:x+cell_size, :] = img
    
    # Save grid
    Image.fromarray(grid).save(output_dir / "comparison_grid.png")
    print(f"\nSaved: {output_dir / 'comparison_grid.png'}")
    
    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 70)
    
    return output_dir


if __name__ == "__main__":
    output_dir = create_visualization()
