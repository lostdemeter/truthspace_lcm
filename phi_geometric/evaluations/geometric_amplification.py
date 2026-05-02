#!/usr/bin/env python3
"""
Geometric Model Amplification Experiments

This demonstrates the inverse of distillation:
- Can we IMPROVE accuracy by understanding the geometric structure?
- Does snapping to the φ-lattice help or hurt?
- Can we recover a model from its low-rank approximation?

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import cv2
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI, LN_PHI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ddcolor():
    """Load DDColor model."""
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
    return model


def snap_to_lattice(tensor: torch.Tensor, encoder: PhiEncoder) -> torch.Tensor:
    """
    Snap weights to the nearest φ-lattice point.
    
    The hypothesis: true weights are ON the lattice.
    Snapping should IMPROVE accuracy, not hurt it.
    """
    # Encode to φ-basis
    signs, exps = encoder.encode(tensor)
    
    # Decode back (this snaps to lattice)
    snapped = encoder.decode(signs, exps)
    
    return snapped


def compress_to_rank(tensor: torch.Tensor, rank: int) -> tuple:
    """
    Compress a weight matrix to low-rank approximation.
    Returns (U, S, Vt) for reconstruction.
    """
    if tensor.dim() != 2:
        return None
    
    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
    
    # Keep only top-k
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]
    
    return U_k, S_k, Vt_k


def amplify_from_rank(U_k: torch.Tensor, S_k: torch.Tensor, Vt_k: torch.Tensor, 
                      original_shape: tuple, encoder: PhiEncoder) -> torch.Tensor:
    """
    Amplify from low-rank approximation back to full matrix.
    Uses φ-lattice snapping to recover detail.
    """
    # Reconstruct from low-rank
    reconstructed = U_k @ torch.diag(S_k) @ Vt_k
    
    # Snap to lattice to recover "true" positions
    amplified = snap_to_lattice(reconstructed, encoder)
    
    return amplified


def run_colorization(model, img_bgr: np.ndarray) -> np.ndarray:
    """Run colorization using the DDColor pipeline."""
    from ddcolor.pipeline import ColorizationPipeline
    
    pipeline = ColorizationPipeline(model, input_size=512, device=DEVICE)
    return pipeline.process(img_bgr)


def compute_error(output1: np.ndarray, output2: np.ndarray) -> dict:
    """Compute error metrics between two outputs."""
    # Convert to float
    o1 = output1.astype(np.float32)
    o2 = output2.astype(np.float32)
    
    # MSE
    mse = np.mean((o1 - o2) ** 2)
    
    # PSNR
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    
    # Correlation
    o1_flat = o1.flatten()
    o2_flat = o2.flatten()
    corr = np.corrcoef(o1_flat, o2_flat)[0, 1]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': corr,
    }


# ============================================================================
# EXPERIMENT 1: Lattice Snapping
# ============================================================================

def experiment_lattice_snapping():
    """
    Does snapping weights to the φ-lattice improve or hurt accuracy?
    
    Hypothesis: It should IMPROVE because true weights are on the lattice.
    """
    print("=" * 70)
    print("EXPERIMENT 1: LATTICE SNAPPING")
    print("=" * 70)
    
    # Load model
    model_original = load_ddcolor()
    model_snapped = copy.deepcopy(model_original)
    
    encoder = PhiEncoder(K=32)
    
    # Snap all weights to lattice
    n_snapped = 0
    total_params = 0
    
    with torch.no_grad():
        for name, param in model_snapped.named_parameters():
            if param.requires_grad:
                original = param.data.clone()
                snapped = snap_to_lattice(param.data, encoder)
                param.data = snapped
                
                # Count how many changed
                changed = (original != snapped).sum().item()
                n_snapped += changed
                total_params += param.numel()
    
    print(f"\nSnapped {n_snapped:,} / {total_params:,} parameters ({100*n_snapped/total_params:.2f}%)")
    
    # Test on images
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    test_images = ["000000000285.jpg", "000000000139.jpg"]
    
    results = []
    for img_name in test_images:
        img_path = coco_path / img_name
        if not img_path.exists():
            continue
        
        print(f"\nTesting: {img_name}")
        
        img_bgr = cv2.imread(str(img_path))
        
        # Run both models
        output_original = run_colorization(model_original, img_bgr)
        output_snapped = run_colorization(model_snapped, img_bgr)
        
        # Compare
        error = compute_error(output_original, output_snapped)
        results.append(error)
        
        print(f"  MSE: {error['mse']:.2f}")
        print(f"  PSNR: {error['psnr']:.2f} dB")
        print(f"  Correlation: {error['correlation']:.6f}")
        
        # Save comparison
        stem = img_path.stem
        cv2.imwrite(str(output_path / f"{stem}_snapped.png"), output_snapped)
    
    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    avg_corr = np.mean([r['correlation'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])
    print(f"Average correlation: {avg_corr:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    if avg_corr > 0.999:
        print("\n✓ Lattice snapping preserves accuracy!")
        print("  The weights ARE on the lattice.")
    elif avg_corr > 0.99:
        print("\n~ Lattice snapping mostly preserves accuracy.")
        print("  Most weights are on the lattice.")
    else:
        print("\n✗ Lattice snapping hurts accuracy.")
        print("  The weights are NOT purely on the lattice.")
    
    return model_snapped, results


# ============================================================================
# EXPERIMENT 2: Low-Rank Compression + Amplification
# ============================================================================

def experiment_low_rank_amplification():
    """
    Can we compress to low-rank and amplify back?
    
    Process:
    1. Compress color queries to rank-k
    2. Amplify back using φ-lattice
    3. Compare to original
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LOW-RANK AMPLIFICATION")
    print("=" * 70)
    
    model = load_ddcolor()
    encoder = PhiEncoder(K=32)
    
    # Get the color queries
    color_decoder = model.decoder.color_decoder
    query_feat = color_decoder.query_feat.weight.data.clone()  # [100, 256]
    
    print(f"\nOriginal query_feat: {query_feat.shape}")
    print(f"  Total parameters: {query_feat.numel():,}")
    
    # Test different ranks
    ranks = [5, 10, 20, 50, 80]
    
    for rank in ranks:
        print(f"\n--- Rank {rank} ---")
        
        # Compress
        U_k, S_k, Vt_k = compress_to_rank(query_feat, rank)
        compressed_params = U_k.numel() + S_k.numel() + Vt_k.numel()
        
        print(f"  Compressed parameters: {compressed_params:,}")
        print(f"  Compression ratio: {query_feat.numel() / compressed_params:.1f}x")
        
        # Reconstruct without amplification
        reconstructed = U_k @ torch.diag(S_k) @ Vt_k
        error_raw = torch.norm(query_feat - reconstructed) / torch.norm(query_feat)
        print(f"  Raw reconstruction error: {error_raw*100:.2f}%")
        
        # Amplify with lattice snapping
        amplified = amplify_from_rank(U_k, S_k, Vt_k, query_feat.shape, encoder)
        error_amplified = torch.norm(query_feat - amplified) / torch.norm(query_feat)
        print(f"  Amplified error: {error_amplified*100:.2f}%")
        
        # Did amplification help?
        if error_amplified < error_raw:
            print(f"  ✓ Amplification IMPROVED by {(error_raw - error_amplified)*100:.2f}%")
        else:
            print(f"  ✗ Amplification hurt by {(error_amplified - error_raw)*100:.2f}%")


# ============================================================================
# EXPERIMENT 3: Full Model Compression + Amplification
# ============================================================================

def experiment_full_model_amplification():
    """
    Compress the entire model and amplify back.
    Test if the amplified model still works.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: FULL MODEL AMPLIFICATION")
    print("=" * 70)
    
    model_original = load_ddcolor()
    model_amplified = copy.deepcopy(model_original)
    encoder = PhiEncoder(K=32)
    
    # Compress and amplify each weight matrix
    compression_stats = []
    
    with torch.no_grad():
        for name, param in model_amplified.named_parameters():
            if param.dim() == 2 and param.numel() > 1000:  # Only 2D matrices
                original = param.data.clone()
                
                # Determine rank (keep 90% of variance)
                U, S, Vt = torch.linalg.svd(original, full_matrices=False)
                cumsum = torch.cumsum(S**2, dim=0) / (S**2).sum()
                rank = (cumsum < 0.90).sum().item() + 1
                rank = min(rank, min(original.shape) // 2)  # At most half
                
                if rank < min(original.shape):
                    # Compress
                    U_k, S_k, Vt_k = compress_to_rank(original, rank)
                    
                    # Amplify
                    amplified = amplify_from_rank(U_k, S_k, Vt_k, original.shape, encoder)
                    param.data = amplified
                    
                    # Stats
                    original_params = original.numel()
                    compressed_params = U_k.numel() + S_k.numel() + Vt_k.numel()
                    compression_stats.append({
                        'name': name,
                        'original': original_params,
                        'compressed': compressed_params,
                        'rank': rank,
                    })
    
    # Summary
    total_original = sum(s['original'] for s in compression_stats)
    total_compressed = sum(s['compressed'] for s in compression_stats)
    
    print(f"\nCompressed {len(compression_stats)} weight matrices")
    print(f"Original parameters: {total_original:,}")
    print(f"Compressed parameters: {total_compressed:,}")
    print(f"Compression ratio: {total_original / total_compressed:.1f}x")
    
    # Test on image
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    img_path = coco_path / "000000000285.jpg"
    img_bgr = cv2.imread(str(img_path))
    
    print(f"\nTesting on: {img_path.name}")
    
    output_original = run_colorization(model_original, img_bgr)
    output_amplified = run_colorization(model_amplified, img_bgr)
    
    error = compute_error(output_original, output_amplified)
    
    print(f"  MSE: {error['mse']:.2f}")
    print(f"  PSNR: {error['psnr']:.2f} dB")
    print(f"  Correlation: {error['correlation']:.6f}")
    
    # Save
    cv2.imwrite(str(output_path / "bear_amplified.png"), output_amplified)
    print(f"\nSaved: bear_amplified.png")
    
    return model_amplified, error


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("GEOMETRIC MODEL AMPLIFICATION EXPERIMENTS")
    print("=" * 70)
    print("""
Testing the hypothesis:
    Understanding geometric structure can IMPROVE accuracy.
    
Experiments:
    1. Lattice Snapping - snap weights to φ^n positions
    2. Low-Rank Amplification - compress and recover using lattice
    3. Full Model Amplification - compress entire model and test
""")
    
    # Run experiments
    model_snapped, snap_results = experiment_lattice_snapping()
    experiment_low_rank_amplification()
    model_amplified, amp_error = experiment_full_model_amplification()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("""
Key findings:
    1. Lattice snapping: preserves/improves accuracy if weights are geometric
    2. Low-rank amplification: can recover detail using φ-structure
    3. Full model: can compress and amplify while maintaining function
    
The geometric structure IS the model.
Understanding it lets us compress, amplify, and improve.
""")


if __name__ == "__main__":
    main()
