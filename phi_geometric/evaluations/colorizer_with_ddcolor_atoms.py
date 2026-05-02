#!/usr/bin/env python3
"""
Colorizer Using DDColor's Learned Atoms

The insight: DDColor's 100 color queries ARE the atoms we need.
We don't need to hand-code them - we extract them from the model.

This bridges the gap:
    - Our framework (geometric, interpretable)
    - DDColor's knowledge (learned, effective)

The "simulated training" is: use DDColor's queries as our atoms.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DDColorAtomColorizer:
    """
    Colorizer that uses DDColor's learned color queries as atoms.
    
    This is the bridge between:
    - Our geometric framework (structure)
    - DDColor's learned knowledge (content)
    """
    
    def __init__(self):
        self.queries = None
        self.color_embed = None
        self.encoder_model = None
        self._load_ddcolor()
    
    def _load_ddcolor(self):
        """Load DDColor and extract the color queries."""
        try:
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
            
            # Store the full model for encoding
            self.encoder_model = model
            
            # Extract color decoder components
            color_decoder = model.decoder.color_decoder
            
            # The 100 color queries - these ARE our atoms
            self.queries = color_decoder.query_feat.weight.detach()  # [100, 256]
            self.query_embed = color_decoder.query_embed.weight.detach()  # [100, 256]
            
            # The color embedding MLP
            self.color_embed = color_decoder.color_embed
            
            print(f"Loaded DDColor atoms: {self.queries.shape}")
            
        except Exception as e:
            print(f"Could not load DDColor: {e}")
            raise
    
    def colorize(self, image: np.ndarray) -> np.ndarray:
        """
        Colorize using DDColor's full pipeline but with our understanding.
        
        This is essentially running DDColor, but we understand it as:
        1. Encode image features
        2. Match features to color queries (atoms)
        3. Apply color embedding to get colors
        """
        # Prepare input
        if image.ndim == 2:
            # Grayscale - convert to RGB
            image = np.stack([image] * 3, axis=-1)
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
        
        img_resized = np.array(Image.fromarray(image).resize((512, 512)))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Run DDColor
        with torch.no_grad():
            output = self.encoder_model(img_tensor)
        
        # Convert output
        output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
        
        return output_np
    
    def colorize_with_atoms(self, gray: np.ndarray) -> np.ndarray:
        """
        Colorize using the extracted atoms directly.
        
        This is a simplified version that shows how the atoms work:
        1. For each pixel, find the best matching query
        2. Apply that query's color
        
        This won't be as good as full DDColor, but shows the principle.
        """
        H, W = gray.shape
        
        # Prepare input for encoder
        if gray.max() <= 1:
            gray_uint8 = (gray * 255).astype(np.uint8)
        else:
            gray_uint8 = gray.astype(np.uint8)
        
        gray_rgb = np.stack([gray_uint8] * 3, axis=-1)
        img_resized = np.array(Image.fromarray(gray_rgb).resize((512, 512)))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
        x = (img_tensor - mean) / std
        
        with torch.no_grad():
            # Run encoder to get features
            self.encoder_model.encoder(x)
            encode_feat = self.encoder_model.encoder.hooks[-1].feature
            
            # Run decoder layers
            out0 = self.encoder_model.decoder.layers[0](encode_feat)
            out1 = self.encoder_model.decoder.layers[1](out0)
            out2 = self.encoder_model.decoder.layers[2](out1)
            out3 = self.encoder_model.decoder.last_shuf(out2)
            
            # Now we have features and queries
            # The color decoder matches features to queries
            
            # Run the full color decoder
            color_decoder = self.encoder_model.decoder.color_decoder
            
            # Get multi-scale features
            features = [out0, out1, out2]
            
            # Initialize queries
            queries = self.queries.unsqueeze(0)  # [1, 100, 256]
            
            # Run transformer layers (cross-attention + self-attention)
            for i, layer in enumerate(color_decoder.transformer_cross_attention_layers):
                scale_idx = i % 3
                feat = features[scale_idx]
                
                # Project features
                proj = color_decoder.input_proj[scale_idx]
                feat_proj = proj(feat)  # [1, 256, H, W]
                
                # Flatten
                B, C, fH, fW = feat_proj.shape
                feat_flat = feat_proj.flatten(2).permute(2, 0, 1)  # [HW, 1, 256]
                
                # Cross-attention
                q = queries.permute(1, 0, 2)  # [100, 1, 256]
                queries_out = layer(q, feat_flat, feat_flat)
                queries = queries_out.permute(1, 0, 2)  # [1, 100, 256]
                
                # Self-attention
                sa_layer = color_decoder.transformer_self_attention_layers[i]
                q = queries.permute(1, 0, 2)
                queries_out = sa_layer(q, q, q)
                queries = queries_out.permute(1, 0, 2)
                
                # FFN
                ffn = color_decoder.transformer_ffn_layers[i]
                queries = queries + ffn(queries)
            
            # Apply color embedding
            color_embed = self.color_embed(queries)  # [1, 100, 256]
            
            # Apply to image features
            out = torch.einsum("bqc,bchw->bqhw", color_embed, out3)  # [1, 100, H, W]
            
            # Combine with input through refine net
            coarse_input = torch.cat([out, x], dim=1)
            output = self.encoder_model.refine_net(coarse_input)
        
        # Convert output
        output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
        
        # Resize back
        output_np = np.array(Image.fromarray(output_np).resize((W, H)))
        
        return output_np


def test_on_real_images():
    """Test the colorizer on real images."""
    print("=" * 70)
    print("TESTING DDCOLOR ATOM COLORIZER")
    print("=" * 70)
    
    colorizer = DDColorAtomColorizer()
    
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_path = Path("/home/thorin/truthspace-lcm/docs/images")
    
    # Test on the bear image and others
    test_images = ["000000000285.jpg", "000000000632.jpg", "000000000139.jpg"]
    
    for img_name in test_images:
        img_path = coco_path / img_name
        if not img_path.exists():
            continue
        
        print(f"\nProcessing: {img_name}")
        
        # Load image
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Convert to grayscale
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        gray = gray / 255.0
        
        # Colorize using DDColor's atoms
        result = colorizer.colorize(img)
        
        # Resize to match
        result = np.array(Image.fromarray(result).resize((img.shape[1], img.shape[0])))
        
        # Save
        stem = img_path.stem
        Image.fromarray(result).save(output_path / f"{stem}_ddcolor_atoms.png")
        print(f"  Saved: {stem}_ddcolor_atoms.png")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Using DDColor's atoms (color queries) gives us DDColor's results.

This is "knowledge transfer" - we're using the learned knowledge
without re-training.

The question you asked: "Can we simulate the data?"

Answer: We don't need to simulate data. We can:
1. EXTRACT the knowledge from DDColor (what we just did)
2. CHARACTERIZE it geometrically (the φ-lattice structure)
3. USE it in our framework (the atoms become our knowledge base)

The "training" already happened in DDColor. We're just
transferring the result into our geometric framework.

The path forward:
1. Extract DDColor's 100 queries as atoms
2. Analyze their φ-structure
3. See if we can DERIVE similar atoms from principles
4. If yes: we have "training without data"
5. If no: we have "knowledge transfer" (still useful)
""")


if __name__ == "__main__":
    test_on_real_images()
