#!/usr/bin/env python3
"""
Automated Query Naming using Vision-Language Models V2

Simpler approach: 
1. Run DDColor on images, get colorized output
2. For each query, find which OUTPUT REGIONS it contributes to most
3. Crop those regions from the COLORIZED image
4. Ask BLIP-2 what color/object is there

The key insight: The output has 100 channels (one per query) before refine_net.
We can trace which queries contribute to which output regions.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


class QueryNamerV2:
    """Name queries by analyzing their contribution to colorized output."""
    
    def __init__(self, use_blip: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ddcolor = None
        self.blip_processor = None
        self.blip_model = None
        self.use_blip = use_blip
        
        # Storage: query_id -> list of (color_a, color_b, region_description)
        self.query_colors = {i: [] for i in range(100)}
        
        self.load_ddcolor()
        if use_blip:
            self.load_blip()
    
    def load_ddcolor(self):
        """Load DDColor model."""
        print("Loading DDColor...")
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        print("  DDColor loaded")
    
    def load_blip(self):
        """Load BLIP-2 model."""
        print("Loading BLIP-2 (this may take a minute)...")
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_name = "Salesforce/blip2-opt-2.7b"
        self.blip_processor = Blip2Processor.from_pretrained(model_name)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto"
        )
        print("  BLIP-2 loaded")
    
    def process_image_with_hooks(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Run DDColor and capture the per-query output before refine_net.
        
        Returns:
            colorized: [H, W, 3] BGR colorized image
            query_output: [100, H, W] per-query contribution
        """
        # Prepare input
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_size = 512
        img_resized = cv2.resize(img_rgb, (input_size, input_size))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        
        # To tensor
        tensor = torch.from_numpy(gray_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Hook to capture decoder output before refine_net
        captured_output = {}
        
        def hook_fn(module, input, output):
            captured_output['decoder_out'] = output.detach()
        
        # Register hook on color_decoder
        hook = self.ddcolor.decoder.color_decoder.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            # Run full model
            final_output = self.ddcolor(tensor)  # [1, 3, H, W]
            
            # Get decoder output (before refine_net)
            # This is [1, 100, H, W] - one channel per query
            decoder_out = captured_output.get('decoder_out')
        
        hook.remove()
        
        # Convert final output to colorized image
        output_rgb = final_output[0].cpu().permute(1, 2, 0).numpy()
        output_rgb = np.clip(output_rgb * 255, 0, 255).astype(np.uint8)
        colorized = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        
        if decoder_out is not None:
            query_output = decoder_out[0].cpu()  # [100, H, W]
        else:
            query_output = None
        
        return colorized, query_output, final_output[0].cpu()
    
    def analyze_query_contributions(self, query_output: torch.Tensor, colorized_rgb: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze which regions each query contributes to.
        
        Args:
            query_output: [100, H, W] per-query output
            colorized_rgb: [H, W, 3] colorized image in RGB
        
        Returns:
            Dict mapping query_id to {region, color, activation}
        """
        H, W = query_output.shape[1], query_output.shape[2]
        
        results = {}
        
        for q in range(100):
            q_out = query_output[q].numpy()  # [H, W]
            
            # Find activation statistics
            activation = np.abs(q_out)
            total_activation = activation.sum()
            
            if total_activation < 1e-6:
                results[q] = {'region': 'inactive', 'color': (0, 0), 'activation': 0}
                continue
            
            # Find center of mass
            y_coords = np.arange(H) / H
            x_coords = np.arange(W) / W
            
            y_weights = activation.sum(axis=1)
            x_weights = activation.sum(axis=0)
            
            y_center = (y_weights * y_coords).sum() / y_weights.sum()
            x_center = (x_weights * x_coords).sum() / x_weights.sum()
            
            # Determine region
            if y_center < 0.33:
                y_region = "top"
            elif y_center < 0.67:
                y_region = "center"
            else:
                y_region = "bottom"
            
            if x_center < 0.33:
                x_region = "left"
            elif x_center < 0.67:
                x_region = "center"
            else:
                x_region = "right"
            
            region = f"{y_region}_{x_region}"
            
            # Get the color in the peak activation region
            peak_y, peak_x = np.unravel_index(activation.argmax(), activation.shape)
            
            # Sample color from a small region around peak
            y1 = max(0, peak_y - 10)
            y2 = min(H, peak_y + 10)
            x1 = max(0, peak_x - 10)
            x2 = min(W, peak_x + 10)
            
            region_color = colorized_rgb[y1:y2, x1:x2].mean(axis=(0, 1))
            
            # Convert RGB to LAB for color analysis
            region_bgr = cv2.cvtColor(
                np.array([[region_color]], dtype=np.uint8), 
                cv2.COLOR_RGB2LAB
            )[0, 0]
            
            l, a, b = region_bgr
            a = a - 128  # Center around 0
            b = b - 128
            
            results[q] = {
                'region': region,
                'color_lab': (float(l), float(a), float(b)),
                'color_rgb': region_color.tolist(),
                'activation': float(total_activation),
                'peak': (int(peak_y), int(peak_x)),
            }
        
        return results
    
    def ask_blip_about_region(self, colorized: np.ndarray, peak: Tuple[int, int], size: int = 64) -> str:
        """Ask BLIP-2 about a region of the colorized image."""
        if not self.use_blip or self.blip_model is None:
            return "unknown"
        
        H, W = colorized.shape[:2]
        y, x = peak
        
        # Crop region
        y1 = max(0, y - size)
        y2 = min(H, y + size)
        x1 = max(0, x - size)
        x2 = min(W, x + size)
        
        crop = colorized[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        
        # Ask about color
        inputs = self.blip_processor(images=image, text="What is the main color?", return_tensors="pt")
        inputs = {k: v.to(self.blip_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.blip_model.generate(**inputs, max_new_tokens=10)
        
        answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return answer.strip().lower()
    
    def process_image(self, img_path: Path) -> Dict:
        """Process a single image."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        
        try:
            colorized, query_output, rgb_output = self.process_image_with_hooks(img_bgr)
        except Exception as e:
            print(f"  Error: {e}")
            return None
        
        if query_output is None:
            print(f"  No query output captured")
            return None
        
        # Convert output to RGB for color analysis
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        
        # Analyze query contributions
        contributions = self.analyze_query_contributions(query_output, colorized_rgb)
        
        # Record colors for each query
        for q, info in contributions.items():
            if info['activation'] > 0.1:  # Only record active queries
                self.query_colors[q].append({
                    'color_lab': info['color_lab'],
                    'region': info['region'],
                    'activation': info['activation'],
                })
        
        return {'path': str(img_path), 'contributions': contributions}
    
    def process_dataset(self, image_dir: Path, max_images: int = 30):
        """Process multiple images."""
        image_paths = list(image_dir.glob("*.jpg"))[:max_images]
        
        print(f"\nProcessing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")
            self.process_image(img_path)
        
        print(f"  Done")
    
    def generate_names(self) -> Dict[int, Dict]:
        """Generate names for all queries based on collected data."""
        results = {}
        
        for q in range(100):
            colors = self.query_colors[q]
            
            if not colors:
                results[q] = {
                    'name': 'inactive',
                    'color': 'none',
                    'region': 'none',
                    'confidence': 0.0,
                }
                continue
            
            # Aggregate color statistics
            all_a = [c['color_lab'][1] for c in colors]
            all_b = [c['color_lab'][2] for c in colors]
            all_regions = [c['region'] for c in colors]
            
            mean_a = np.mean(all_a)
            mean_b = np.mean(all_b)
            
            # Determine color name
            if mean_a > 20:
                if mean_b > 20:
                    color = "orange"
                elif mean_b < -20:
                    color = "magenta"
                else:
                    color = "red"
            elif mean_a < -20:
                if mean_b > 20:
                    color = "lime"
                elif mean_b < -20:
                    color = "cyan"
                else:
                    color = "green"
            else:
                if mean_b > 20:
                    color = "yellow"
                elif mean_b < -20:
                    color = "blue"
                else:
                    color = "neutral"
            
            # Most common region
            from collections import Counter
            region_counts = Counter(all_regions)
            top_region = region_counts.most_common(1)[0][0]
            
            # Generate name
            name = f"{color}_{top_region}"
            
            results[q] = {
                'name': name,
                'color': color,
                'region': top_region,
                'mean_a': float(mean_a),
                'mean_b': float(mean_b),
                'n_samples': len(colors),
                'confidence': min(1.0, len(colors) / 10),
            }
        
        return results
    
    def print_summary(self, results: Dict[int, Dict]):
        """Print summary of query names."""
        print("\n" + "=" * 70)
        print("QUERY NAMES")
        print("=" * 70)
        
        # Group by color
        by_color = defaultdict(list)
        for q, info in results.items():
            by_color[info['color']].append((q, info))
        
        for color in sorted(by_color.keys()):
            queries = by_color[color]
            print(f"\n## {color.upper()} ({len(queries)} queries)")
            for q, info in sorted(queries, key=lambda x: x[1].get('n_samples', 0), reverse=True)[:10]:
                print(f"  Query {q:2d}: {info['name']:<25} (a={info.get('mean_a', 0):+.0f}, b={info.get('mean_b', 0):+.0f}, n={info.get('n_samples', 0)})")


def main():
    print("=" * 70)
    print("AUTOMATED QUERY NAMING V2")
    print("=" * 70)
    
    # Don't use BLIP for now - just analyze colors directly
    namer = QueryNamerV2(use_blip=False)
    
    # Process COCO images
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    namer.process_dataset(coco_path, max_images=30)
    
    # Generate names
    results = namer.generate_names()
    
    # Print summary
    namer.print_summary(results)
    
    # Save results
    output_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/query_names_v2.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    active = [q for q, r in results.items() if r['confidence'] > 0]
    print(f"  Active queries: {len(active)}/100")
    
    from collections import Counter
    colors = Counter(r['color'] for r in results.values())
    print(f"\n  Color distribution:")
    for color, count in colors.most_common():
        print(f"    {color}: {count}")
    
    return results


if __name__ == "__main__":
    results = main()
