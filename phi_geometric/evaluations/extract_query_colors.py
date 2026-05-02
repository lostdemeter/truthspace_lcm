#!/usr/bin/env python3
"""
Extract Query Colors from DDColor

Simple approach: Pass query features through color_embed MLP
to see what color each query produces.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


def main():
    print("=" * 70)
    print("EXTRACTING DDCOLOR QUERY COLORS")
    print("=" * 70)
    
    # Load model
    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    
    # Get query features
    query_feat = model.decoder.color_decoder.query_feat.weight.detach()  # [100, 256]
    print(f"Query features: {query_feat.shape}")
    
    # Get color_embed MLP
    color_embed = model.decoder.color_decoder.color_embed
    print(f"Color embed: {color_embed}")
    
    # Pass queries through color_embed
    with torch.no_grad():
        query_colors = color_embed(query_feat)  # [100, 2]
    
    query_colors_np = query_colors.numpy()
    
    print(f"\nQuery colors shape: {query_colors_np.shape}")
    print(f"Color range: a=[{query_colors_np[:, 0].min():.1f}, {query_colors_np[:, 0].max():.1f}], "
          f"b=[{query_colors_np[:, 1].min():.1f}, {query_colors_np[:, 1].max():.1f}]")
    
    # Analyze query colors
    print("\n" + "=" * 70)
    print("QUERY COLOR ANALYSIS")
    print("=" * 70)
    
    # Sort by a (warm/cool)
    sorted_by_a = np.argsort(query_colors_np[:, 0])
    
    print(f"\n## Top 15 WARMEST Queries (highest a = red)")
    for idx in sorted_by_a[-15:][::-1]:
        a, b = query_colors_np[idx]
        # Interpret the color
        if b > 10:
            hue = "orange/yellow"
        elif b < -10:
            hue = "magenta/pink"
        else:
            hue = "red"
        print(f"  Query {idx:2d}: a={a:+6.1f}, b={b:+6.1f} → {hue}")
    
    print(f"\n## Top 15 COOLEST Queries (lowest a = green)")
    for idx in sorted_by_a[:15]:
        a, b = query_colors_np[idx]
        if b > 10:
            hue = "yellow-green"
        elif b < -10:
            hue = "cyan/teal"
        else:
            hue = "green"
        print(f"  Query {idx:2d}: a={a:+6.1f}, b={b:+6.1f} → {hue}")
    
    # Sort by b (yellow/blue)
    sorted_by_b = np.argsort(query_colors_np[:, 1])
    
    print(f"\n## Top 15 YELLOWEST Queries (highest b)")
    for idx in sorted_by_b[-15:][::-1]:
        a, b = query_colors_np[idx]
        if a > 10:
            hue = "orange/gold"
        elif a < -10:
            hue = "lime/chartreuse"
        else:
            hue = "yellow"
        print(f"  Query {idx:2d}: a={a:+6.1f}, b={b:+6.1f} → {hue}")
    
    print(f"\n## Top 15 BLUEST Queries (lowest b)")
    for idx in sorted_by_b[:15]:
        a, b = query_colors_np[idx]
        if a > 10:
            hue = "purple/violet"
        elif a < -10:
            hue = "cyan/teal"
        else:
            hue = "blue"
        print(f"  Query {idx:2d}: a={a:+6.1f}, b={b:+6.1f} → {hue}")
    
    # Cluster by color
    print("\n" + "=" * 70)
    print("COLOR CLUSTERS")
    print("=" * 70)
    
    # Define color regions in LAB space
    clusters = {
        'warm_yellow': [],    # a > 0, b > 0
        'warm_blue': [],      # a > 0, b < 0
        'cool_yellow': [],    # a < 0, b > 0
        'cool_blue': [],      # a < 0, b < 0
        'neutral': [],        # near origin
    }
    
    for idx in range(100):
        a, b = query_colors_np[idx]
        
        if abs(a) < 10 and abs(b) < 10:
            clusters['neutral'].append(idx)
        elif a > 0 and b > 0:
            clusters['warm_yellow'].append(idx)
        elif a > 0 and b < 0:
            clusters['warm_blue'].append(idx)
        elif a < 0 and b > 0:
            clusters['cool_yellow'].append(idx)
        else:
            clusters['cool_blue'].append(idx)
    
    for cluster_name, query_ids in clusters.items():
        print(f"\n## {cluster_name.upper()}: {len(query_ids)} queries")
        if query_ids:
            colors = query_colors_np[query_ids]
            print(f"   Mean: a={colors[:, 0].mean():+.1f}, b={colors[:, 1].mean():+.1f}")
            print(f"   Queries: {query_ids[:10]}{'...' if len(query_ids) > 10 else ''}")
    
    # Save results
    results = {
        'query_colors': query_colors_np.tolist(),
        'clusters': {k: v for k, v in clusters.items()},
        'sorted_by_warmth': sorted_by_a.tolist(),
        'sorted_by_yellow': sorted_by_b.tolist(),
    }
    
    output_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/query_colors.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Create a semantic map
    print("\n" + "=" * 70)
    print("SEMANTIC MAP OF 100 QUERIES")
    print("=" * 70)
    
    print("""
The 100 queries form a COLOR VOCABULARY:

┌─────────────────────────────────────────────────────────────┐
│                        +b (yellow)                          │
│                            ▲                                │
│     COOL_YELLOW           │           WARM_YELLOW           │
│     (lime, chartreuse)    │           (orange, gold)        │
│                           │                                 │
│ -a ◄──────────────────────┼──────────────────────────► +a   │
│ (green)                   │                        (red)    │
│                           │                                 │
│     COOL_BLUE             │           WARM_BLUE             │
│     (cyan, teal)          │           (purple, magenta)     │
│                           │                                 │
│                           ▼                                 │
│                        -b (blue)                            │
└─────────────────────────────────────────────────────────────┘

Each query is a "color atom" that produces a specific color.
The attention mechanism selects which atoms to use for each pixel.
""")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
DDColor's 100 queries form a COLOR VOCABULARY:
- {len(clusters['warm_yellow'])} warm-yellow queries (orange, gold, skin tones)
- {len(clusters['warm_blue'])} warm-blue queries (purple, magenta, pink)
- {len(clusters['cool_yellow'])} cool-yellow queries (lime, chartreuse, grass)
- {len(clusters['cool_blue'])} cool-blue queries (cyan, teal, sky)
- {len(clusters['neutral'])} neutral queries (gray, brown, muted)

This is the SEMANTIC CONTENT we were looking for!
Each query has a specific color meaning.
The attention mechanism routes pixels to the right color queries.

NEXT STEPS:
1. Analyze which image regions activate which color clusters
2. Build direct routing: region → color cluster → output
3. Skip attention by using learned routing rules
""")
    
    return results


if __name__ == "__main__":
    results = main()
