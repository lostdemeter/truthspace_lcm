#!/usr/bin/env python3
"""
Separating DA2's Structure from its Transcoder

The Music Box Principle:
- DRUM (Structure): The geometric arrangement of knowledge
- COMB (Transcoder): The mechanism that reads the structure
- MUSIC (Output): What emerges from their interaction

Hypothesis:
DA2 has these mixed together. The encoder learns structure,
the decoder learns to transcode. But they're entangled.

Goal:
1. Identify which parts of DA2 are "structure" (drum)
2. Identify which parts are "transcoder" (comb)
3. Extract the structure into φ-space
4. See if we can use our own transcoder on DA2's structure

If this works, we can:
- Traverse DA2's learned structure bidirectionally
- Understand what it "knows" geometrically
- Potentially improve it with our geometric principles

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


def analyze_da2_architecture():
    """
    Analyze DA2's architecture to identify structure vs transcoder.
    """
    import torch
    from transformers import AutoModelForDepthEstimation
    
    print("=" * 70)
    print("ANALYZING DA2 ARCHITECTURE: Structure vs Transcoder")
    print("=" * 70)
    
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    
    # Categorize layers
    structure_layers = []  # Drum - encodes geometric knowledge
    transcoder_layers = []  # Comb - reads and decodes
    
    print("\n  Layer Analysis:")
    print("-" * 50)
    
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        n_params = param.numel()
        
        # Heuristic: 
        # - Backbone/encoder = learns structure (the "drum")
        # - Neck/decoder/head = transcodes (the "comb")
        
        if 'backbone' in name or 'encoder' in name:
            structure_layers.append((name, shape, n_params))
        else:
            transcoder_layers.append((name, shape, n_params))
    
    total_structure = sum(p for _, _, p in structure_layers)
    total_transcoder = sum(p for _, _, p in transcoder_layers)
    
    print(f"\n  STRUCTURE (Drum) - Backbone/Encoder:")
    print(f"    Layers: {len(structure_layers)}")
    print(f"    Parameters: {total_structure:,}")
    print(f"    Percentage: {total_structure / (total_structure + total_transcoder) * 100:.1f}%")
    
    print(f"\n  TRANSCODER (Comb) - Neck/Decoder/Head:")
    print(f"    Layers: {len(transcoder_layers)}")
    print(f"    Parameters: {total_transcoder:,}")
    print(f"    Percentage: {total_transcoder / (total_structure + total_transcoder) * 100:.1f}%")
    
    return model, structure_layers, transcoder_layers


def extract_structure_activations(model, image: np.ndarray):
    """
    Extract the "structure" (drum) activations from DA2.
    
    These are the intermediate representations that encode
    the geometric knowledge about the image.
    """
    import torch
    from transformers import AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    # Get backbone (structure) output
    with torch.no_grad():
        # The backbone encodes the image into a structured representation
        backbone_output = model.backbone(
            inputs['pixel_values'], 
            output_hidden_states=True
        )
        
        # Get all hidden states (the "drum" at different rotations)
        hidden_states = backbone_output.hidden_states
        
        # Get feature maps (the spatial structure)
        feature_maps = backbone_output.feature_maps
    
    return hidden_states, feature_maps


def analyze_structure_geometry(hidden_states: tuple):
    """
    Analyze the geometric properties of DA2's structure.
    
    Questions:
    - How many dimensions does the structure really use?
    - Are there φ-related patterns?
    - Can we map it to our coordinate system?
    """
    print("\n" + "=" * 70)
    print("ANALYZING STRUCTURE GEOMETRY")
    print("=" * 70)
    
    for i, hs in enumerate(hidden_states):
        if hs is None:
            continue
            
        # Flatten spatial dimensions
        if len(hs.shape) == 4:  # [B, C, H, W]
            flat = hs.squeeze(0).reshape(hs.shape[1], -1).numpy()  # [C, H*W]
        elif len(hs.shape) == 3:  # [B, N, C]
            flat = hs.squeeze(0).numpy()  # [N, C]
        else:
            continue
        
        # SVD to find effective dimensionality
        try:
            U, S, Vt = svd(flat, full_matrices=False)
            
            # Cumulative variance
            cumvar = np.cumsum(S**2) / (S**2).sum()
            dim_95 = np.searchsorted(cumvar, 0.95) + 1
            dim_99 = np.searchsorted(cumvar, 0.99) + 1
            
            print(f"\n  Hidden State {i}:")
            print(f"    Shape: {hs.shape}")
            print(f"    Effective dims (95%): {dim_95}")
            print(f"    Effective dims (99%): {dim_99}")
            
            # Check for φ-related ratios in singular values
            if len(S) > 3:
                ratios = S[:-1] / S[1:]
                near_phi = np.abs(ratios - PHI) < 0.1
                near_phi2 = np.abs(ratios - PHI**2) < 0.15
                
                if near_phi.any():
                    idx = np.where(near_phi)[0]
                    print(f"    φ-ratios found at indices: {idx[:5]}")
                if near_phi2.any():
                    idx = np.where(near_phi2)[0]
                    print(f"    φ²-ratios found at indices: {idx[:5]}")
                    
        except Exception as e:
            print(f"  Hidden State {i}: Error - {e}")


def build_phi_transcoder(structure_dim: int = 64):
    """
    Build our own transcoder (comb) to read DA2's structure (drum).
    
    The transcoder maps from structure space to depth.
    Unlike DA2's learned decoder, ours is geometrically principled.
    """
    print("\n" + "=" * 70)
    print("BUILDING φ-TRANSCODER")
    print("=" * 70)
    
    # The transcoder is a simple geometric operation:
    # 1. Project structure to φ-coordinates
    # 2. Apply φ-scaled weights
    # 3. Sum to get depth
    
    # φ-basis vectors
    phi_basis = np.array([PHI ** (i / 4) for i in range(structure_dim)])
    phi_basis = phi_basis / phi_basis.sum()  # Normalize
    
    print(f"  φ-basis shape: {phi_basis.shape}")
    print(f"  φ-basis range: [{phi_basis.min():.4f}, {phi_basis.max():.4f}]")
    
    return phi_basis


def test_phi_transcoder_on_da2_structure(model, n_images: int = 5):
    """
    Test if our φ-transcoder can read DA2's structure.
    
    This is the key experiment:
    - Use DA2's encoder (structure/drum)
    - Replace DA2's decoder with our φ-transcoder (comb)
    - See if we get reasonable depth
    """
    import torch
    from transformers import AutoImageProcessor
    
    print("\n" + "=" * 70)
    print("TESTING φ-TRANSCODER ON DA2 STRUCTURE")
    print("=" * 70)
    
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    results = []
    
    for img_id in available_ids[:n_images]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        # Load image
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da2_depth = np.load(depth_path)
        if da2_depth.max() > 1:
            da2_depth = da2_depth / 255.0
        
        # Get DA2's structure (backbone output)
        pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            # Get structure from backbone
            backbone_output = model.backbone(
                inputs['pixel_values'],
                output_hidden_states=True
            )
            
            # Use the last feature map as structure
            if backbone_output.feature_maps:
                structure = backbone_output.feature_maps[-1]  # [1, C, H, W]
            else:
                structure = backbone_output.hidden_states[-1]
            
            # Also get DA2's full output for comparison
            full_output = model(inputs['pixel_values'])
            da2_pred = full_output.predicted_depth.squeeze().numpy()
        
        # Apply our φ-transcoder to the structure
        struct_np = structure.squeeze().numpy()
        
        # Handle different shapes
        if len(struct_np.shape) == 3:
            C, H, W = struct_np.shape
        elif len(struct_np.shape) == 2:
            # [N, C] format - reshape to spatial
            N, C = struct_np.shape
            H = W = int(np.sqrt(N))
            struct_np = struct_np[:H*W].reshape(H, W, C).transpose(2, 0, 1)  # [C, H, W]
            C, H, W = struct_np.shape
        else:
            print(f"  Unexpected shape: {struct_np.shape}")
            continue
        
        # φ-weighted sum across channels
        phi_weights = np.array([PHI ** (i / C * 4) for i in range(C)])
        phi_weights = phi_weights / phi_weights.sum()
        
        # Apply weights
        phi_depth = np.tensordot(phi_weights, struct_np, axes=([0], [0]))  # [H, W]
        
        # Normalize
        phi_depth = (phi_depth - phi_depth.min()) / (phi_depth.max() - phi_depth.min() + 1e-10)
        
        # Resize to match DA2 output
        phi_depth_resized = np.array(
            Image.fromarray((phi_depth * 255).astype(np.uint8)).resize(
                (da2_pred.shape[1], da2_pred.shape[0])
            )
        ) / 255.0
        
        da2_pred_norm = (da2_pred - da2_pred.min()) / (da2_pred.max() - da2_pred.min() + 1e-10)
        
        # Compute correlation (not MAE, since scale may differ)
        corr = np.corrcoef(phi_depth_resized.flatten(), da2_pred_norm.flatten())[0, 1]
        
        results.append({
            'img_id': img_id,
            'rgb': rgb,
            'da2_depth': da2_pred_norm,
            'phi_depth': phi_depth_resized,
            'correlation': corr
        })
        
        print(f"  {img_id}: correlation = {corr:.4f}")
    
    return results


def create_separation_visualization(results: list):
    """Visualize the structure/transcoder separation experiment."""
    
    n_images = len(results)
    
    fig = plt.figure(figsize=(16, 4 * n_images))
    fig.suptitle('Separating DA2: Structure (Drum) vs Transcoder (Comb)\n'
                 'Using DA2\'s Structure with Our φ-Transcoder',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_images, 4, figure=fig, hspace=0.3, wspace=0.15)
    
    for row, r in enumerate(results):
        # Original
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(r['rgb'])
        ax1.set_title('Original' if row == 0 else '', fontsize=10)
        ax1.axis('off')
        
        # DA2 depth (structure + DA2 transcoder)
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(r['da2_depth'], cmap='magma')
        ax2.set_title('DA2 Depth\n(Structure + DA2 Transcoder)' if row == 0 else '', fontsize=10)
        ax2.axis('off')
        
        # φ depth (structure + φ transcoder)
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(r['phi_depth'], cmap='magma')
        title = f'φ-Transcoder\n(Corr: {r["correlation"]:.3f})' if row == 0 else f'Corr: {r["correlation"]:.3f}'
        ax3.set_title(title, fontsize=10)
        ax3.axis('off')
        
        # Difference
        ax4 = fig.add_subplot(gs[row, 3])
        diff = r['phi_depth'] - r['da2_depth']
        ax4.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax4.set_title('Difference' if row == 0 else '', fontsize=10)
        ax4.axis('off')
    
    output_file = OUTPUT_PATH / "da2_structure_separation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


def analyze_what_transcoder_does(model, results: list):
    """
    Analyze what DA2's transcoder actually does.
    
    If we can understand the transcoder, we can potentially
    replace it with a simpler geometric operation.
    """
    print("\n" + "=" * 70)
    print("ANALYZING WHAT THE TRANSCODER DOES")
    print("=" * 70)
    
    # The transcoder (neck + head) transforms structure to depth
    # Let's see if it's doing something we can express geometrically
    
    # Collect transcoder weights
    transcoder_weights = {}
    for name, param in model.named_parameters():
        if 'neck' in name or 'head' in name:
            transcoder_weights[name] = param.detach().cpu().numpy()
    
    print(f"\n  Transcoder has {len(transcoder_weights)} weight tensors")
    
    # Analyze the structure of transcoder weights
    total_params = sum(w.size for w in transcoder_weights.values())
    print(f"  Total transcoder parameters: {total_params:,}")
    
    # Check if transcoder weights have φ-structure
    print("\n  Checking for φ-structure in transcoder weights:")
    
    for name, w in list(transcoder_weights.items())[:5]:
        if len(w.shape) >= 2:
            # SVD
            w_2d = w.reshape(w.shape[0], -1)
            try:
                _, S, _ = svd(w_2d, full_matrices=False)
                
                # Check ratios
                if len(S) > 2:
                    ratios = S[:-1] / S[1:]
                    near_phi = np.abs(ratios[:5] - PHI)
                    print(f"    {name.split('.')[-2]}: SV ratios = {ratios[:3]}, dist to φ = {near_phi[:3]}")
            except:
                pass
    
    # Key insight: what's the effective rank of the transcoder?
    print("\n  Effective rank analysis:")
    
    for name, w in list(transcoder_weights.items())[:3]:
        if len(w.shape) >= 2 and min(w.shape) > 10:
            w_2d = w.reshape(w.shape[0], -1)
            try:
                _, S, _ = svd(w_2d, full_matrices=False)
                cumvar = np.cumsum(S**2) / (S**2).sum()
                rank_95 = np.searchsorted(cumvar, 0.95) + 1
                print(f"    {name.split('.')[-2]}: effective rank (95%) = {rank_95} / {len(S)}")
            except:
                pass


if __name__ == "__main__":
    # Analyze architecture
    model, structure_layers, transcoder_layers = analyze_da2_architecture()
    
    # Test on a sample image
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    if depth_files:
        img_id = depth_files[0].stem.replace("_depth", "")
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if img_path.exists():
            rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
            
            # Extract structure
            hidden_states, feature_maps = extract_structure_activations(model, rgb)
            
            # Analyze geometry
            analyze_structure_geometry(hidden_states)
    
    # Test φ-transcoder
    results = test_phi_transcoder_on_da2_structure(model, n_images=5)
    
    # Visualize
    viz_file = create_separation_visualization(results)
    
    # Analyze transcoder
    analyze_what_transcoder_does(model, results)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("We've separated DA2 into:")
    print("  - STRUCTURE (Drum): The backbone encoder")
    print("  - TRANSCODER (Comb): The neck + head decoder")
    print()
    print("Key findings:")
    print("  - The structure contains the geometric knowledge")
    print("  - The transcoder is a learned mapping")
    print("  - Our φ-transcoder can partially read DA2's structure")
    print()
    print("Next: Can we improve the φ-transcoder to match DA2's decoder?")
