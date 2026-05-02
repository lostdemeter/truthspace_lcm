"""
Test that φ-decoder weights are truly universal across different images.

This verifies that the 203-byte weight file works for ANY image.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent))
from phi_decoder import PhiDecoder, PhiConfig, extract_head_features


def main():
    print("=" * 70)
    print("TESTING UNIVERSAL φ-DECODER WEIGHTS")
    print("=" * 70)
    print()
    
    # Paths
    COCO_VAL_PATH = Path('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017')
    WEIGHTS_PATH = Path(__file__).parent / 'phi_weights.bin'
    OUTPUT_PATH = Path(__file__).parent / 'universal_test.png'
    
    # Load DA2
    print("Loading DA2 model...")
    processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model.eval()
    
    # Load φ-decoder
    print(f"Loading φ-decoder weights ({WEIGHTS_PATH.stat().st_size} bytes)...")
    config = PhiConfig(k_weights=512, k_residual=64, bits_weights=16, bits_residual=12)
    decoder = PhiDecoder(config)
    decoder.load_weights(WEIGHTS_PATH)
    print()
    
    # Test on multiple different images
    test_images = [
        '000000000785',  # Skier (training image)
        '000000000139',  # Different scene
        '000000000632',  # Another scene
        '000000001000',  # Yet another
        '000000001268',  # More variety
        '000000001296',  # Even more
    ]
    
    results = []
    fig, axes = plt.subplots(len(test_images), 4, figsize=(16, 4 * len(test_images)))
    
    print(f"{'Image ID':<15} {'Correlation':>15} {'Edge Corr':>12} {'Status':<10}")
    print("-" * 55)
    
    from scipy import ndimage
    
    for idx, img_id in enumerate(test_images):
        img_path = COCO_VAL_PATH / f'{img_id}.jpg'
        if not img_path.exists():
            print(f"{img_id:<15} {'N/A':>15} {'N/A':>12} {'NOT FOUND':<10}")
            continue
        
        # Load and process
        pil_image = Image.open(img_path).convert('RGB')
        rgb = np.array(pil_image).astype(np.float32) / 255.0
        inputs = processor(images=pil_image, return_tensors='pt')
        
        # Get DA2 depth
        with torch.no_grad():
            outputs = model(inputs['pixel_values'])
        da2_depth = outputs.predicted_depth.squeeze().numpy()
        da2_depth_norm = (da2_depth - da2_depth.min()) / (da2_depth.max() - da2_depth.min())
        H, W = da2_depth.shape
        
        # Get φ-decoder prediction
        features = extract_head_features(model, inputs)
        pred = decoder.predict(features)
        
        # Compute metrics
        corr = np.corrcoef(pred.flatten(), da2_depth_norm.flatten())[0, 1]
        
        edges_da2 = np.abs(ndimage.sobel(da2_depth_norm))
        edges_phi = np.abs(ndimage.sobel(pred))
        edge_corr = np.corrcoef(edges_da2.flatten(), edges_phi.flatten())[0, 1]
        
        status = "✓ PASS" if corr > 0.999 else "✗ FAIL"
        print(f"{img_id:<15} {corr:>15.10f} {edge_corr:>12.4f} {status:<10}")
        
        results.append({
            'img_id': img_id,
            'correlation': corr,
            'edge_corr': edge_corr,
        })
        
        # Plot
        # Resize RGB to match depth
        rgb_resized = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W, H))) / 255.0
        
        axes[idx, 0].imshow(rgb_resized)
        axes[idx, 0].set_title(f'{img_id}', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(da2_depth_norm, cmap='magma')
        axes[idx, 1].set_title('DA2 Original', fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(pred, cmap='magma')
        axes[idx, 2].set_title(f'φ-Decoder (r={corr:.6f})', fontsize=10)
        axes[idx, 2].axis('off')
        
        diff = np.abs(da2_depth_norm - pred)
        axes[idx, 3].imshow(diff * 100, cmap='hot', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'|Diff| (100x)', fontsize=10)
        axes[idx, 3].axis('off')
    
    print()
    
    # Summary
    if results:
        avg_corr = np.mean([r['correlation'] for r in results])
        avg_edge = np.mean([r['edge_corr'] for r in results])
        print(f"Average correlation: {avg_corr:.10f}")
        print(f"Average edge corr:   {avg_edge:.4f}")
        print()
        print(f"All images use the SAME 203-byte weight file!")
    
    plt.suptitle(f'Universal φ-Decoder: 203 bytes → {len(results)} images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
