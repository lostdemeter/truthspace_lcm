#!/usr/bin/env python3
"""
Automated Query Naming V3

Use the official DDColor pipeline and analyze the ab output directly.
The model outputs LAB ab channels, which we can interpret directly.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys
import cv2
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


class QueryNamerV3:
    """Name queries by analyzing their output colors."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        
        # Storage: query_id -> list of color samples
        self.query_colors = {i: [] for i in range(100)}
        
        self.load_model()
    
    def load_model(self):
        """Load DDColor pipeline."""
        print("Loading DDColor...")
        from ddcolor import DDColor
        from ddcolor.pipeline import ColorizationPipeline
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        model.eval()
        model = model.to(self.device)
        
        self.model = model
        self.pipeline = ColorizationPipeline(model, input_size=512)
        print("  DDColor loaded")
    
    def process_image(self, img_path: Path) -> Dict:
        """Process a single image and analyze query contributions."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        
        # Use the pipeline to colorize
        try:
            colorized = self.pipeline.process(img_bgr)
        except Exception as e:
            print(f"  Pipeline error: {e}")
            return None
        
        # Convert colorized to LAB to get ab values
        colorized_lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
        
        # Get the ab channels
        a_channel = colorized_lab[:, :, 1].astype(float) - 128  # Center around 0
        b_channel = colorized_lab[:, :, 2].astype(float) - 128
        
        H, W = a_channel.shape
        
        # Divide image into 10x10 grid (100 regions for 100 queries)
        # This is a heuristic: assume queries roughly correspond to spatial regions
        grid_size = 10
        cell_h = H // grid_size
        cell_w = W // grid_size
        
        for q in range(100):
            # Map query to grid position
            row = q // grid_size
            col = q % grid_size
            
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            # Get mean color in this region
            region_a = a_channel[y1:y2, x1:x2].mean()
            region_b = b_channel[y1:y2, x1:x2].mean()
            
            self.query_colors[q].append({
                'a': float(region_a),
                'b': float(region_b),
            })
        
        return {'path': str(img_path)}
    
    def process_dataset(self, image_dir: Path, max_images: int = 50):
        """Process multiple images."""
        image_paths = list(image_dir.glob("*.jpg"))[:max_images]
        
        print(f"\nProcessing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")
            self.process_image(img_path)
        
        print(f"  Done")
    
    def interpret_color(self, a: float, b: float) -> str:
        """Convert LAB ab values to a color name."""
        # LAB color space:
        # a: negative = green, positive = red/magenta
        # b: negative = blue, positive = yellow
        
        if abs(a) < 10 and abs(b) < 10:
            return "gray"
        
        # Determine hue based on angle in ab plane
        angle = np.arctan2(b, a) * 180 / np.pi  # -180 to 180
        
        if angle < -150 or angle >= 150:
            return "green"
        elif -150 <= angle < -90:
            return "cyan"
        elif -90 <= angle < -30:
            return "blue"
        elif -30 <= angle < 30:
            return "magenta" if a > 0 else "purple"
        elif 30 <= angle < 90:
            return "red" if a > 20 else "orange"
        elif 90 <= angle < 150:
            return "yellow" if b > 20 else "olive"
        else:
            return "neutral"
    
    def generate_names(self) -> Dict[int, Dict]:
        """Generate names for all queries."""
        results = {}
        
        for q in range(100):
            colors = self.query_colors[q]
            
            if not colors:
                results[q] = {'name': 'unused', 'color': 'none', 'confidence': 0}
                continue
            
            # Aggregate
            all_a = [c['a'] for c in colors]
            all_b = [c['b'] for c in colors]
            
            mean_a = np.mean(all_a)
            mean_b = np.mean(all_b)
            std_a = np.std(all_a)
            std_b = np.std(all_b)
            
            # Get color name
            color = self.interpret_color(mean_a, mean_b)
            
            # Spatial position
            row = q // 10
            col = q % 10
            
            if row < 3:
                y_pos = "top"
            elif row < 7:
                y_pos = "mid"
            else:
                y_pos = "bottom"
            
            if col < 3:
                x_pos = "left"
            elif col < 7:
                x_pos = "center"
            else:
                x_pos = "right"
            
            position = f"{y_pos}_{x_pos}"
            
            # Generate name
            name = f"{color}_{position}"
            
            # Confidence based on consistency
            saturation = np.sqrt(mean_a**2 + mean_b**2)
            consistency = 1.0 / (1.0 + std_a + std_b)
            confidence = min(1.0, saturation / 50 * consistency)
            
            results[q] = {
                'name': name,
                'color': color,
                'position': position,
                'mean_a': float(mean_a),
                'mean_b': float(mean_b),
                'std_a': float(std_a),
                'std_b': float(std_b),
                'saturation': float(saturation),
                'confidence': float(confidence),
                'n_samples': len(colors),
            }
        
        return results
    
    def print_summary(self, results: Dict[int, Dict]):
        """Print summary."""
        print("\n" + "=" * 70)
        print("QUERY NAMES (by spatial position)")
        print("=" * 70)
        
        # Print as 10x10 grid
        print("\n  Color Grid (10x10):")
        print("  " + "-" * 52)
        
        for row in range(10):
            line = "  |"
            for col in range(10):
                q = row * 10 + col
                color = results[q]['color'][:4]  # First 4 chars
                line += f" {color:4s}|"
            print(line)
        
        print("  " + "-" * 52)
        
        # Group by color
        print("\n" + "=" * 70)
        print("QUERIES BY COLOR")
        print("=" * 70)
        
        by_color = defaultdict(list)
        for q, info in results.items():
            by_color[info['color']].append((q, info))
        
        for color in sorted(by_color.keys(), key=lambda c: len(by_color[c]), reverse=True):
            queries = by_color[color]
            print(f"\n## {color.upper()} ({len(queries)} queries)")
            
            # Show top 5 by saturation
            for q, info in sorted(queries, key=lambda x: x[1]['saturation'], reverse=True)[:5]:
                print(f"  Query {q:2d}: a={info['mean_a']:+5.1f}, b={info['mean_b']:+5.1f}, sat={info['saturation']:.1f}")


def main():
    print("=" * 70)
    print("AUTOMATED QUERY NAMING V3")
    print("=" * 70)
    print("Approach: Map 100 queries to 10x10 spatial grid, analyze colors")
    
    namer = QueryNamerV3()
    
    # Process COCO images
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    namer.process_dataset(coco_path, max_images=50)
    
    # Generate names
    results = namer.generate_names()
    
    # Print summary
    namer.print_summary(results)
    
    # Save results
    output_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/query_names_v3.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("COLOR DISTRIBUTION")
    print("=" * 70)
    
    colors = Counter(r['color'] for r in results.values())
    for color, count in colors.most_common():
        bar = "█" * count
        print(f"  {color:10s}: {count:2d} {bar}")
    
    return results


if __name__ == "__main__":
    results = main()
