#!/usr/bin/env python3
"""
Compare V3 Chemistry vs DDColor (Pretrained Weights)

This is the key comparison:
- V3 Chemistry: Hand-coded knowledge (atoms, molecules, reactions)
- DDColor: Learned knowledge (pretrained weights on φ-lattice)

Both use the φ-geometric framework, but the knowledge source differs.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import DDColor
DDCOLOR_AVAILABLE = False
try:
    sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    DDCOLOR_AVAILABLE = True
    print("DDColor available")
except ImportError as e:
    print(f"DDColor not available: {e}")

from phi_geometric.evaluations.colorizer_v3_chemistry import ChemistryColorizer
from phi_geometric.core.knowledge_base import create_color_knowledge_base

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lab_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB."""
    try:
        from skimage import color
        lab = np.zeros((*L.shape, 3))
        lab[..., 0] = L * 100
        lab[..., 1] = ab[..., 0]
        lab[..., 2] = ab[..., 1]
        rgb = color.lab2rgb(lab)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    except ImportError:
        rgb = np.zeros((*L.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip((L * 255 + ab[..., 0]).astype(int), 0, 255)
        rgb[..., 1] = np.clip((L * 255).astype(int), 0, 255)
        rgb[..., 2] = np.clip((L * 255 + ab[..., 1]).astype(int), 0, 255)
        return rgb


def rgb_to_lab(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RGB to LAB, return (L, ab)."""
    try:
        from skimage import color
        # Ensure RGB has 3 channels
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[-1] != 3:
            # Pad or truncate to 3 channels
            if rgb.shape[-1] < 3:
                rgb = np.concatenate([rgb, np.zeros((*rgb.shape[:-1], 3 - rgb.shape[-1]))], axis=-1)
            else:
                rgb = rgb[..., :3]
        rgb_float = rgb.astype(float) / 255.0 if rgb.max() > 1 else rgb.astype(float)
        lab = color.rgb2lab(rgb_float)
        L = lab[..., 0] / 100.0  # Normalize to 0-1
        ab = lab[..., 1:3]
        return L, ab
    except ImportError:
        # Simple approximation
        L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        L = L / 255.0
        ab = np.zeros((*L.shape, 2))
        return L, ab


def create_test_images():
    """Create test images with semantic regions."""
    images = []
    
    # 1. Landscape (sky + ground)
    landscape = np.zeros((64, 64))
    sky_mask = np.zeros((64, 64), dtype=bool)
    ground_mask = np.zeros((64, 64), dtype=bool)
    
    for i in range(64):
        if i < 28:
            landscape[i, :] = 0.75 - 0.1 * i / 28
            sky_mask[i, :] = True
        elif i < 32:
            landscape[i, :] = 0.65
        else:
            landscape[i, :] = 0.35 + 0.15 * (i - 32) / 32
            ground_mask[i, :] = True
    
    images.append(("landscape", landscape, {"sky": sky_mask, "vegetation": ground_mask}))
    
    # 2. Portrait-like
    portrait = np.zeros((64, 64))
    skin_mask = np.zeros((64, 64), dtype=bool)
    
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - 32)**2 + (j - 32)**2)
            if dist < 20:
                portrait[i, j] = 0.55 + 0.15 * (1 - dist / 20)
                skin_mask[i, j] = True
            else:
                portrait[i, j] = 0.3
    
    images.append(("portrait", portrait, {"skin": skin_mask}))
    
    return images


def analyze_colorization(name: str, gray: np.ndarray, ab: np.ndarray) -> Dict:
    """Analyze colorization quality."""
    a = ab[..., 0]
    b = ab[..., 1]
    
    saturation = np.sqrt(a**2 + b**2).mean()
    
    # Check for semantic correctness
    has_blue = b.min() < -15  # Blue = negative b
    has_green = a.min() < -15 and b.max() > 15  # Green = negative a, positive b
    has_warm = a.max() > 10  # Warm = positive a
    
    return {
        "name": name,
        "a_range": (a.min(), a.max()),
        "b_range": (b.min(), b.max()),
        "saturation": saturation,
        "has_blue": has_blue,
        "has_green": has_green,
        "has_warm": has_warm,
    }


def run_comparison():
    """Compare V3 Chemistry vs DDColor."""
    print("=" * 70)
    print("V3 CHEMISTRY vs DDCOLOR COMPARISON")
    print("=" * 70)
    
    # Initialize V3 Chemistry
    print("\n--- V3 Chemistry Colorizer ---")
    v3 = ChemistryColorizer()
    
    # Initialize DDColor if available
    ddcolor_model = None
    if DDCOLOR_AVAILABLE:
        print("\n--- DDColor (Pretrained) ---")
        try:
            class DDColorHF(DDColor, PyTorchModelHubMixin):
                def __init__(self, config=None, **kwargs):
                    if isinstance(config, dict):
                        kwargs = {**config, **kwargs}
                    super().__init__(**kwargs)
            
            ddcolor_model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
            ddcolor_model.eval()
            ddcolor_model = ddcolor_model.to(DEVICE)
            print("  DDColor loaded successfully")
        except Exception as e:
            print(f"  Could not load DDColor: {e}")
            ddcolor_model = None
    
    # Test images
    test_images = create_test_images()
    
    results = []
    
    for name, gray, semantic_map in test_images:
        print(f"\n--- {name} ---")
        
        # V3 Chemistry
        ab_v3 = v3.colorize(gray, semantic_map)
        stats_v3 = analyze_colorization(f"{name}_v3", gray, ab_v3)
        print(f"  V3 Chemistry:")
        print(f"    a: [{stats_v3['a_range'][0]:.1f}, {stats_v3['a_range'][1]:.1f}]")
        print(f"    b: [{stats_v3['b_range'][0]:.1f}, {stats_v3['b_range'][1]:.1f}]")
        print(f"    Saturation: {stats_v3['saturation']:.1f}")
        
        # DDColor (if available)
        if ddcolor_model is not None:
            try:
                # DDColor expects RGB input
                gray_rgb = np.stack([gray * 255] * 3, axis=-1).astype(np.uint8)
                gray_rgb_resized = np.array(Image.fromarray(gray_rgb).resize((512, 512)))
                
                img_tensor = torch.from_numpy(gray_rgb_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = ddcolor_model(img_tensor)
                
                output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
                output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
                output_np = np.array(Image.fromarray(output_np).resize((64, 64)))
                
                # Convert to LAB
                L_dd, ab_dd = rgb_to_lab(output_np)
                stats_dd = analyze_colorization(f"{name}_ddcolor", L_dd, ab_dd)
                
                print(f"  DDColor (Pretrained):")
                print(f"    a: [{stats_dd['a_range'][0]:.1f}, {stats_dd['a_range'][1]:.1f}]")
                print(f"    b: [{stats_dd['b_range'][0]:.1f}, {stats_dd['b_range'][1]:.1f}]")
                print(f"    Saturation: {stats_dd['saturation']:.1f}")
                
                results.append({
                    "name": name,
                    "v3": stats_v3,
                    "ddcolor": stats_dd,
                    "ab_v3": ab_v3,
                    "ab_dd": ab_dd,
                })
            except Exception as e:
                print(f"  DDColor error: {e}")
                results.append({
                    "name": name,
                    "v3": stats_v3,
                    "ddcolor": None,
                    "ab_v3": ab_v3,
                    "ab_dd": None,
                })
        else:
            results.append({
                "name": name,
                "v3": stats_v3,
                "ddcolor": None,
                "ab_v3": ab_v3,
                "ab_dd": None,
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n## Key Differences")
    print("-" * 40)
    print("""
  V3 Chemistry:
    - Knowledge source: Hand-coded atoms, molecules, reactions
    - 19 color atoms with response curves
    - 3 molecular constraints (sky-ground, shadow, water-sky)
    - 3 reactions (sunset, shadow, highlight)
    - NO training, NO pretrained weights
    
  DDColor (Pretrained):
    - Knowledge source: Learned from millions of images
    - 100 color queries (learned embeddings)
    - 9 transformer layers with cross-attention
    - Weights ARE shapes on the φ-lattice
    - Trained on large dataset
""")
    
    print("\n## What V3 Chemistry Gets Right")
    print("-" * 40)
    for r in results:
        v3 = r["v3"]
        print(f"  {r['name']}:")
        print(f"    Blue sky: {v3['has_blue']}")
        print(f"    Green ground: {v3['has_green']}")
        print(f"    Warm tones: {v3['has_warm']}")
    
    if any(r["ddcolor"] for r in results):
        print("\n## What DDColor Gets Right")
        print("-" * 40)
        for r in results:
            if r["ddcolor"]:
                dd = r["ddcolor"]
                print(f"  {r['name']}:")
                print(f"    Blue sky: {dd['has_blue']}")
                print(f"    Green ground: {dd['has_green']}")
                print(f"    Warm tones: {dd['has_warm']}")
                print(f"    Saturation: {dd['saturation']:.1f} (vs V3: {r['v3']['saturation']:.1f})")
    
    print("\n## The Key Insight")
    print("-" * 40)
    print("""
  Both V3 Chemistry and DDColor use the φ-geometric framework.
  The difference is the SOURCE of knowledge:
  
  V3 Chemistry:
    - Knowledge is EXPLICIT (we wrote the rules)
    - Limited to what we know to encode
    - Transparent and interpretable
    - Works without training
    
  DDColor:
    - Knowledge is IMPLICIT (learned from data)
    - Captures patterns we didn't explicitly encode
    - Opaque but powerful
    - Requires training (or pretrained weights)
    
  The φ-lattice is the COMMON STRUCTURE.
  The knowledge is what differs.
""")
    
    print("\n## What DDColor Learned That V3 Doesn't Have")
    print("-" * 40)
    print("""
  1. TEXTURE-COLOR CORRELATIONS
     - DDColor learned that certain textures → certain colors
     - V3 only knows semantic categories, not textures
     
  2. CONTEXT-DEPENDENT COLORS
     - DDColor learned that the same object has different colors
       in different contexts (indoor vs outdoor, etc.)
     - V3 uses fixed colors per category
     
  3. SUBTLE GRADIENTS
     - DDColor learned smooth color transitions
     - V3 uses simple response curves
     
  4. OBJECT BOUNDARIES
     - DDColor learned to respect object edges
     - V3 uses simple edge-aware smoothing
""")
    
    print("\n## How to Bridge the Gap")
    print("-" * 40)
    print("""
  Option 1: REVERSE-ENGINEER DDColor's knowledge
    - Extract the 100 color queries as atoms
    - Extract attention patterns as molecules
    - Encode in our Knowledge Chemistry format
    
  Option 2: LEARN the knowledge
    - Train a "sculptor" to create atoms from examples
    - Use attractor/repeller dynamics to self-organize
    - Let the knowledge emerge from data
    
  Option 3: HYBRID approach
    - Use V3 Chemistry for explicit rules
    - Use DDColor for learned refinement
    - Combine both knowledge sources
""")
    
    return results


if __name__ == "__main__":
    run_comparison()
