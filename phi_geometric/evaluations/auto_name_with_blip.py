#!/usr/bin/env python3
"""
Automated Query Naming with BLIP-2

Strategy:
1. Run DDColor on images to get colorized output
2. Divide each colorized image into 10x10 grid (100 regions)
3. For each region, ask BLIP-2: "What color is this? What object is this?"
4. Aggregate answers across images to name each "query" (region)

This gives us semantic labels like "blue_sky", "green_grass", "brown_wood"

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys
import cv2
from collections import defaultdict, Counter
from typing import Dict, List
import json
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


class SemanticQueryNamer:
    """Name queries using BLIP-2 for semantic understanding."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.blip_processor = None
        self.blip_model = None
        
        # Storage: region_id -> list of descriptions
        self.region_descriptions = {i: [] for i in range(100)}
        self.region_colors = {i: [] for i in range(100)}
        
        self.load_models()
    
    def load_models(self):
        """Load DDColor and BLIP-2."""
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
        
        self.ddcolor_model = model
        self.pipeline = ColorizationPipeline(model, input_size=512)
        print("  DDColor loaded")
        
        print("Loading BLIP-2 (Salesforce/blip2-opt-2.7b)...")
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_name = "Salesforce/blip2-opt-2.7b"
        self.blip_processor = Blip2Processor.from_pretrained(model_name)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto"
        )
        print("  BLIP-2 loaded")
    
    def ask_blip(self, image: Image.Image, question: str) -> str:
        """Ask BLIP-2 a question about an image."""
        # Use the prompt format that works with BLIP-2
        prompt = f"Question: {question} Answer:"
        inputs = self.blip_processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.blip_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.blip_model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=3,
                do_sample=False,
            )
        
        answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the answer if present
        answer = answer.replace(prompt, "").strip().lower()
        return answer
    
    def process_image(self, img_path: Path) -> Dict:
        """Process a single image."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        
        # Colorize
        try:
            colorized = self.pipeline.process(img_bgr)
        except Exception as e:
            print(f"  Error: {e}")
            return None
        
        # Convert to RGB for BLIP
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        H, W = colorized_rgb.shape[:2]
        
        # Divide into 10x10 grid
        grid_size = 10
        cell_h = H // grid_size
        cell_w = W // grid_size
        
        # Sample a subset of regions (not all 100 every time)
        # Sample 10 random regions per image
        regions_to_sample = np.random.choice(100, size=10, replace=False)
        
        for q in regions_to_sample:
            row = q // grid_size
            col = q % grid_size
            
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            # Extract region
            region = colorized_rgb[y1:y2, x1:x2]
            region_pil = Image.fromarray(region)
            
            # Ask BLIP about color
            color_answer = self.ask_blip(region_pil, "What is the main color?")
            self.region_colors[q].append(color_answer)
            
            # Ask BLIP about content
            content_answer = self.ask_blip(region_pil, "What is this? Answer in 1-2 words.")
            self.region_descriptions[q].append(content_answer)
        
        return {'path': str(img_path), 'regions_sampled': regions_to_sample.tolist()}
    
    def process_dataset(self, image_dir: Path, max_images: int = 20):
        """Process multiple images."""
        image_paths = list(image_dir.glob("*.jpg"))[:max_images]
        
        print(f"\nProcessing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths):
            print(f"  Processing {i + 1}/{len(image_paths)}: {img_path.name}")
            self.process_image(img_path)
        
        print(f"  Done")
    
    def generate_names(self) -> Dict[int, Dict]:
        """Generate names for all regions."""
        results = {}
        
        for q in range(100):
            colors = self.region_colors[q]
            descriptions = self.region_descriptions[q]
            
            if not colors:
                results[q] = {'name': 'unsampled', 'color': 'unknown', 'object': 'unknown', 'confidence': 0}
                continue
            
            # Find most common color
            color_counts = Counter(colors)
            top_color = color_counts.most_common(1)[0][0] if color_counts else 'unknown'
            
            # Find most common description words
            all_words = ' '.join(descriptions).split()
            # Filter common words
            stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'of', 'and', 'with', 'this', 'that', 'it'}
            word_counts = Counter(w for w in all_words if w not in stop_words and len(w) > 2)
            top_object = word_counts.most_common(1)[0][0] if word_counts else 'unknown'
            
            # Generate name
            name = f"{top_color}_{top_object}"
            
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
            
            results[q] = {
                'name': name,
                'color': top_color,
                'object': top_object,
                'position': f"{y_pos}_{x_pos}",
                'all_colors': colors,
                'all_descriptions': descriptions,
                'n_samples': len(colors),
                'confidence': len(colors) / 5,  # Expect ~5 samples per region
            }
        
        return results
    
    def print_summary(self, results: Dict[int, Dict]):
        """Print summary."""
        print("\n" + "=" * 70)
        print("SEMANTIC QUERY NAMES")
        print("=" * 70)
        
        # Print as 10x10 grid
        print("\n  Semantic Grid (10x10):")
        print("  " + "-" * 72)
        
        for row in range(10):
            line = "  |"
            for col in range(10):
                q = row * 10 + col
                name = results[q]['name'][:6]  # First 6 chars
                line += f" {name:6s}|"
            print(line)
        
        print("  " + "-" * 72)
        
        # Group by color
        print("\n" + "=" * 70)
        print("QUERIES BY COLOR")
        print("=" * 70)
        
        by_color = defaultdict(list)
        for q, info in results.items():
            by_color[info['color']].append((q, info))
        
        for color in sorted(by_color.keys(), key=lambda c: len(by_color[c]), reverse=True)[:10]:
            queries = by_color[color]
            print(f"\n## {color.upper()} ({len(queries)} queries)")
            for q, info in queries[:5]:
                pos = info.get('position', 'unknown')
                print(f"  Query {q:2d}: {info['name']:<25} ({pos})")
        
        # Group by object
        print("\n" + "=" * 70)
        print("QUERIES BY OBJECT")
        print("=" * 70)
        
        by_object = defaultdict(list)
        for q, info in results.items():
            by_object[info['object']].append((q, info))
        
        for obj in sorted(by_object.keys(), key=lambda o: len(by_object[o]), reverse=True)[:10]:
            queries = by_object[obj]
            print(f"\n## {obj.upper()} ({len(queries)} queries)")
            for q, info in queries[:5]:
                pos = info.get('position', 'unknown')
                print(f"  Query {q:2d}: {info['name']:<25} ({pos})")


def main():
    print("=" * 70)
    print("SEMANTIC QUERY NAMING WITH BLIP-2")
    print("=" * 70)
    
    namer = SemanticQueryNamer()
    
    # Process COCO images
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    namer.process_dataset(coco_path, max_images=20)
    
    # Generate names
    results = namer.generate_names()
    
    # Print summary
    namer.print_summary(results)
    
    # Save results
    output_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/semantic_query_names.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = main()
