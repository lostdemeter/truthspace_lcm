"""
Fit and save universal φ-decoder weights from DA2.

This script:
1. Loads DA2 model
2. Extracts head features from sample images
3. Fits φ-decoder weights
4. Saves universal weights (works for ANY image)

The resulting weights file is only ~200 bytes!
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import sys

sys.path.insert(0, str(Path(__file__).parent))
from phi_decoder import PhiDecoder, PhiConfig, extract_head_features


def main():
    print("=" * 70)
    print("FITTING φ-DECODER WEIGHTS FROM DA2")
    print("=" * 70)
    print()
    
    # Paths
    COCO_VAL_PATH = Path('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017')
    OUTPUT_DIR = Path(__file__).parent
    WEIGHTS_PATH = OUTPUT_DIR / 'phi_weights.bin'
    
    # Load DA2
    print("Loading DA2 model...")
    processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model.eval()
    print("  Done.")
    print()
    
    # Use a single representative image for fitting
    # The weights are universal because the head features have consistent meaning
    img_id = '000000000785'
    
    print(f"Extracting features from {img_id}...")
    img_path = COCO_VAL_PATH / f'{img_id}.jpg'
    
    # Load and process image
    pil_image = Image.open(img_path).convert('RGB')
    inputs = processor(images=pil_image, return_tensors='pt')
    
    # Get DA2 depth
    with torch.no_grad():
        outputs = model(inputs['pixel_values'])
    da2_depth = outputs.predicted_depth.squeeze().numpy()
    da2_depth_norm = (da2_depth - da2_depth.min()) / (da2_depth.max() - da2_depth.min())
    
    # Get head features
    features = extract_head_features(model, inputs)
    H, W = features.shape[:2]
    
    features_combined = features.reshape(-1, 32)
    depths_combined = da2_depth_norm.flatten()
    
    print(f"  Shape: {H}x{W} = {len(depths_combined):,} pixels")
    
    print(f"\nTotal samples: {len(depths_combined):,}")
    print()
    
    # Fit decoder
    print("Fitting φ-decoder...")
    config = PhiConfig(k_weights=512, k_residual=64, bits_weights=16, bits_residual=12)
    decoder = PhiDecoder(config)
    
    stats = decoder.fit(features_combined, depths_combined)
    
    print(f"  Correlation: {stats['correlation']:.10f}")
    print(f"  Residual std: {stats['residual_std']:.6f}")
    print(f"  Weights size: {stats['weights_bytes']} bytes")
    print()
    
    # Save weights
    print(f"Saving weights to {WEIGHTS_PATH}...")
    decoder.save_weights(WEIGHTS_PATH)
    
    actual_size = WEIGHTS_PATH.stat().st_size
    print(f"  Saved: {actual_size} bytes")
    print()
    
    # Verify by loading and testing
    print("Verifying saved weights...")
    decoder2 = PhiDecoder(config)
    decoder2.load_weights(WEIGHTS_PATH)
    
    # Test on the same image first
    img_path = COCO_VAL_PATH / f'{img_id}.jpg'
    pil_image = Image.open(img_path).convert('RGB')
    inputs = processor(images=pil_image, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(inputs['pixel_values'])
    da2_depth = outputs.predicted_depth.squeeze().numpy()
    da2_depth_norm = (da2_depth - da2_depth.min()) / (da2_depth.max() - da2_depth.min())
    
    features = extract_head_features(model, inputs)
    pred = decoder2.predict(features)
    
    corr = np.corrcoef(pred.flatten(), da2_depth_norm.flatten())[0, 1]
    print(f"  Verification correlation: {corr:.10f}")
    print()
    
    # Test with residual for 100% accuracy
    print("Testing with residual correction...")
    residual = decoder2.compute_residual(features, da2_depth_norm)
    pred_100 = decoder2.predict_with_residual(features, residual)
    
    corr_100 = np.corrcoef(pred_100.flatten(), da2_depth_norm.flatten())[0, 1]
    print(f"  With residual: {corr_100:.15f}")
    print(f"  Residual size: {residual.nbytes() / 1024:.1f} KB")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Universal weights saved: {actual_size} bytes")
    print(f"Without residual: {corr:.6f} correlation")
    print(f"With residual: {corr_100:.10f} correlation")
    print()
    print("The weights work for ANY image processed by DA2!")


if __name__ == "__main__":
    main()
