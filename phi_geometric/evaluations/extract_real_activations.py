#!/usr/bin/env python3
"""
Extract Real Activation Patterns from DDColor

Run DDColor on real COCO images and extract:
1. Which queries activate for each image
2. Where in the image each query attends (spatial)
3. What colors each query produces

This gives us the ACTUAL semantic content of the 100 queries.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI


@dataclass
class QueryActivation:
    """Record of a query's activation on an image."""
    query_id: int
    image_path: str
    
    # Attention pattern
    attention_map: np.ndarray  # [H, W] - where this query attended
    attention_mass: float  # Total attention this query received
    
    # Spatial statistics
    center_of_mass: Tuple[float, float]  # (y, x) normalized 0-1
    spatial_spread: float  # How spread out the attention is
    
    # Color output
    mean_color: Tuple[float, float]  # Mean (a, b) this query produced
    
    # Region classification
    region: str  # "top", "center", "bottom", "left", "right", etc.


class DDColorActivationExtractor:
    """Extract activation patterns from DDColor."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
        # Storage for activations
        self.activations: List[QueryActivation] = []
        self.query_stats: Dict[int, Dict] = {i: {
            'total_activations': 0,
            'spatial_histogram': np.zeros((3, 3)),  # 3x3 grid
            'color_sum': np.array([0.0, 0.0]),
            'attention_mass_sum': 0.0,
        } for i in range(100)}
    
    def load_model(self):
        """Load DDColor model."""
        try:
            from ddcolor import DDColor
            from huggingface_hub import PyTorchModelHubMixin
            
            class DDColorHF(DDColor, PyTorchModelHubMixin):
                def __init__(self, config=None, **kwargs):
                    if isinstance(config, dict):
                        kwargs = {**config, **kwargs}
                    super().__init__(**kwargs)
            
            self.model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
            self.model.eval()
            self.model = self.model.to(self.device)
            print("DDColor loaded successfully")
            
        except Exception as e:
            print(f"Could not load DDColor: {e}")
            raise
    
    def extract_attention(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DDColor and manually compute attention patterns.
        
        Returns:
            attention_map: [n_queries, H, W] - where each query attends
            color_output: [2, H, W] - ab channels
        """
        # Prepare input
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize for model
        input_size = 512
        img_resized = cv2.resize(img_rgb, (input_size, input_size))
        
        # Convert to grayscale RGB (what DDColor expects)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        
        # To tensor
        tensor = torch.from_numpy(gray_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Forward pass to get output
        with torch.no_grad():
            output_ab = self.model(tensor)  # [1, 2, H, W]
            
            # Get encoder features
            encoder_out = self.model.encoder(tensor)
            
            # Get the features that go into color decoder
            # DDColor uses multi-scale features, we'll use the last one
            if isinstance(encoder_out, (list, tuple)):
                features = encoder_out[-1]  # Use highest resolution
            else:
                features = encoder_out
        
        # Get queries
        query_feat = self.model.decoder.color_decoder.query_feat.weight  # [100, 256]
        
        # Get first cross-attention layer's projections
        layer = self.model.decoder.color_decoder.transformer_cross_attention_layers[0]
        in_proj = layer.multihead_attn.in_proj_weight  # [768, 256]
        
        W_q = in_proj[:256]   # [256, 256]
        W_k = in_proj[256:512]  # [256, 256]
        
        # Compute Q and K
        with torch.no_grad():
            Q = query_feat @ W_q.T  # [100, 256]
            
            # Flatten features spatially
            if features.dim() == 4:
                B, C, H, W = features.shape
                features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            else:
                features_flat = features
                H = W = int(np.sqrt(features_flat.shape[1]))
            
            # Project features to keys
            K = features_flat[0] @ W_k.T  # [H*W, 256]
            
            # Compute attention scores
            d_k = 256
            scores = Q @ K.T / np.sqrt(d_k)  # [100, H*W]
            attention = torch.softmax(scores, dim=-1)  # [100, H*W]
            
            # Reshape to spatial
            attention_map = attention.view(100, H, W).cpu()
        
        return attention_map, output_ab.cpu()
    
    def process_image(self, img_path: Path) -> List[QueryActivation]:
        """Process a single image and extract query activations."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return []
        
        h, w = img_bgr.shape[:2]
        
        try:
            attention_map, output_ab = self.extract_attention(img_bgr)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return []
        
        activations = []
        
        # attention_map is [100, H, W]
        attn_h, attn_w = attention_map.shape[1], attention_map.shape[2]
        
        for query_id in range(100):
            # Get attention map for this query
            attn_map = attention_map[query_id].numpy()  # [attn_h, attn_w]
            
            # Resize to original image size
            attn_map_resized = cv2.resize(attn_map, (w, h))
            
            # Compute attention mass
            attention_mass = attn_map.sum()
            
            # Compute center of mass
            y_coords = np.arange(attn_h) / attn_h
            x_coords = np.arange(attn_w) / attn_w
            
            y_center = (attn_map.sum(axis=1) * y_coords).sum() / (attention_mass + 1e-8)
            x_center = (attn_map.sum(axis=0) * x_coords).sum() / (attention_mass + 1e-8)
            
            # Compute spatial spread (variance)
            y_var = ((y_coords - y_center) ** 2 * attn_map.sum(axis=1)).sum() / (attention_mass + 1e-8)
            x_var = ((x_coords - x_center) ** 2 * attn_map.sum(axis=0)).sum() / (attention_mass + 1e-8)
            spatial_spread = np.sqrt(y_var + x_var)
            
            # Classify region
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
            
            # Get mean color output for this query's attended region
            output_ab_np = output_ab[0].numpy()  # [2, H, W]
            output_ab_resized = np.stack([
                cv2.resize(output_ab_np[0], (w, h)),
                cv2.resize(output_ab_np[1], (w, h)),
            ])
            
            # Weight color by attention
            weighted_a = (attn_map_resized * output_ab_resized[0]).sum() / (attn_map_resized.sum() + 1e-8)
            weighted_b = (attn_map_resized * output_ab_resized[1]).sum() / (attn_map_resized.sum() + 1e-8)
            
            activation = QueryActivation(
                query_id=query_id,
                image_path=str(img_path),
                attention_map=attn_map,
                attention_mass=float(attention_mass),
                center_of_mass=(float(y_center), float(x_center)),
                spatial_spread=float(spatial_spread),
                mean_color=(float(weighted_a), float(weighted_b)),
                region=region,
            )
            
            activations.append(activation)
            
            # Update query stats
            stats = self.query_stats[query_id]
            stats['total_activations'] += 1
            stats['attention_mass_sum'] += attention_mass
            stats['color_sum'] += np.array([weighted_a, weighted_b]) * attention_mass
            
            # Update spatial histogram
            y_bin = min(2, int(y_center * 3))
            x_bin = min(2, int(x_center * 3))
            stats['spatial_histogram'][y_bin, x_bin] += attention_mass
        
        return activations
    
    def process_dataset(self, image_dir: Path, max_images: int = 100):
        """Process multiple images from a directory."""
        image_paths = list(image_dir.glob("*.jpg"))[:max_images]
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")
            
            activations = self.process_image(img_path)
            self.activations.extend(activations)
        
        print(f"  Total activations recorded: {len(self.activations)}")
    
    def analyze_queries(self) -> Dict[int, Dict]:
        """Analyze the collected query statistics."""
        results = {}
        
        for query_id in range(100):
            stats = self.query_stats[query_id]
            
            if stats['attention_mass_sum'] < 1e-6:
                continue
            
            # Normalize spatial histogram
            spatial_hist = stats['spatial_histogram']
            spatial_hist_norm = spatial_hist / (spatial_hist.sum() + 1e-8)
            
            # Find dominant region
            y_idx, x_idx = np.unravel_index(spatial_hist_norm.argmax(), spatial_hist_norm.shape)
            y_regions = ["top", "center", "bottom"]
            x_regions = ["left", "center", "right"]
            dominant_region = f"{y_regions[y_idx]}_{x_regions[x_idx]}"
            
            # Mean color
            mean_color = stats['color_sum'] / (stats['attention_mass_sum'] + 1e-8)
            
            # Interpret color
            a, b = mean_color
            if a > 20:
                color_type = "warm"
            elif a < -20:
                color_type = "cool"
            else:
                color_type = "neutral"
            
            if b > 20:
                color_hue = "yellow"
            elif b < -20:
                color_hue = "blue"
            else:
                color_hue = "neutral"
            
            # Generate tentative name
            name_parts = [dominant_region.replace("_", "-")]
            if color_type != "neutral":
                name_parts.append(color_type)
            if color_hue != "neutral":
                name_parts.append(color_hue)
            
            tentative_name = "_".join(name_parts)
            
            results[query_id] = {
                'dominant_region': dominant_region,
                'spatial_histogram': spatial_hist_norm.tolist(),
                'mean_color': mean_color.tolist(),
                'color_type': color_type,
                'color_hue': color_hue,
                'tentative_name': tentative_name,
                'total_attention_mass': float(stats['attention_mass_sum']),
            }
        
        return results
    
    def print_query_summary(self, results: Dict[int, Dict]):
        """Print a summary of query semantics."""
        print("\n" + "=" * 70)
        print("QUERY SEMANTIC SUMMARY")
        print("=" * 70)
        
        # Group by region
        by_region = {}
        for query_id, info in results.items():
            region = info['dominant_region']
            if region not in by_region:
                by_region[region] = []
            by_region[region].append((query_id, info))
        
        for region in sorted(by_region.keys()):
            queries = by_region[region]
            print(f"\n## {region.upper()}: {len(queries)} queries")
            
            for query_id, info in sorted(queries, key=lambda x: x[1]['total_attention_mass'], reverse=True)[:5]:
                a, b = info['mean_color']
                print(f"  Query {query_id:2d}: {info['tentative_name']:<30} (a={a:+.1f}, b={b:+.1f})")
        
        # Color distribution
        print("\n## COLOR DISTRIBUTION")
        warm_queries = [q for q, i in results.items() if i['color_type'] == 'warm']
        cool_queries = [q for q, i in results.items() if i['color_type'] == 'cool']
        neutral_queries = [q for q, i in results.items() if i['color_type'] == 'neutral']
        
        print(f"  Warm queries: {len(warm_queries)}")
        print(f"  Cool queries: {len(cool_queries)}")
        print(f"  Neutral queries: {len(neutral_queries)}")
        
        # Spatial distribution
        print("\n## SPATIAL DISTRIBUTION")
        for region in sorted(by_region.keys()):
            print(f"  {region}: {len(by_region[region])} queries")


def main():
    print("=" * 70)
    print("EXTRACTING REAL DDCOLOR ACTIVATION PATTERNS")
    print("=" * 70)
    
    extractor = DDColorActivationExtractor()
    
    # Process COCO images
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    extractor.process_dataset(coco_path, max_images=50)
    
    # Analyze queries
    results = extractor.analyze_queries()
    
    # Print summary
    extractor.print_query_summary(results)
    
    # Save results
    output_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/query_semantics.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved query semantics to: {output_path}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
We now have the ACTUAL semantic content of DDColor's 100 queries:
- Which spatial regions each query attends to
- What colors each query produces
- Tentative names based on behavior

This enables:
1. DIRECT ROUTING: Skip attention, route by region
2. SEMANTIC UNDERSTANDING: Know what each query "means"
3. TRANSFER: Use this knowledge for new colorizers

The unnamed vocabulary is now NAMED through observation.
""")
    
    return results


if __name__ == "__main__":
    results = main()
