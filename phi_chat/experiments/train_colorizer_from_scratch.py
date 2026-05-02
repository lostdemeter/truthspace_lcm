#!/usr/bin/env python3
"""
Train a Colorizer from Scratch

The goal: Build a simple colorizer, train it, then analyze its φ-structure.

This is the fail-fast approach:
1. If we can train a working colorizer, we can analyze what it learned
2. If the learned weights show φ-structure, that validates our hypothesis
3. If not, we learn where the hypothesis breaks down

Architecture (simple but effective):
- Input: Grayscale image (1 channel)
- Encoder: Simple CNN to extract features
- Decoder: Predict U, V channels
- Loss: L1 on U, V

Author: TruthSpace LCM Project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB color space."""
    from skimage import color
    return color.rgb2lab(rgb / 255.0)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB color space."""
    from skimage import color
    rgb = color.lab2rgb(lab)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


class ColorDataset(Dataset):
    """Dataset of grayscale -> color pairs."""
    
    def __init__(self, image_paths: List[Path], size: int = 256):
        self.image_paths = image_paths
        self.size = size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = img.resize((self.size, self.size))
        rgb = np.array(img)
        
        # Convert to LAB
        lab = rgb_to_lab(rgb)
        
        # L channel (grayscale input)
        L = lab[:, :, 0:1] / 50.0 - 1.0  # Normalize to [-1, 1]
        
        # ab channels (color target)
        ab = lab[:, :, 1:3] / 128.0  # Normalize to [-1, 1]
        
        # To tensors [C, H, W]
        L = torch.from_numpy(L.transpose(2, 0, 1)).float()
        ab = torch.from_numpy(ab.transpose(2, 0, 1)).float()
        
        return L, ab


class SimpleColorizer(nn.Module):
    """
    Simple colorizer network.
    
    Architecture designed to be analyzable:
    - Small number of layers
    - Clear structure
    - No batch norm (to keep weights interpretable)
    """
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(1, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        # Decoder
        self.dec4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_channels, 2, 3, padding=1)  # Output: 2 channels (a, b)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        
        # Decoder
        d4 = self.relu(self.dec4(e4))
        d3 = self.relu(self.dec3(d4))
        d2 = self.relu(self.dec2(d3))
        out = self.tanh(self.dec1(d2))
        
        return out


def train_colorizer(model, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3):
    """Train the colorizer."""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for L, ab in train_loader:
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            
            optimizer.zero_grad()
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(DEVICE), ab.to(DEVICE)
                pred_ab = model(L)
                val_loss += criterion(pred_ab, ab).item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"   Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    return train_losses, val_losses


def analyze_phi_structure(model):
    """Analyze φ-structure in trained weights."""
    print("\n" + "=" * 70)
    print("φ-STRUCTURE ANALYSIS OF TRAINED WEIGHTS")
    print("=" * 70)
    
    all_weights = []
    layer_stats = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.detach().cpu().numpy().flatten()
            all_weights.extend(weights.tolist())
            
            # Compute φ-levels
            def to_phi_level(v, k=32):
                if abs(v) < 1e-10:
                    return 0
                return int(round(k * np.log(abs(v)) / LN_PHI))
            
            levels = np.array([to_phi_level(w) for w in weights if abs(w) > 1e-10])
            
            if len(levels) > 1:
                sorted_levels = np.sort(levels)
                diffs = np.abs(np.diff(sorted_levels))
                fib_near = sum(1 for d in diffs if any(abs(d - f) <= 1 for f in FIBONACCI))
                fib_pct = 100 * fib_near / len(diffs) if len(diffs) > 0 else 0
                
                layer_stats.append({
                    'name': name,
                    'n_weights': len(weights),
                    'n_levels': len(np.unique(levels)),
                    'fib_pct': fib_pct
                })
    
    print("\nPer-layer analysis:")
    for stat in layer_stats:
        print(f"  {stat['name']}: {stat['n_weights']} weights, {stat['n_levels']} φ-levels, {stat['fib_pct']:.1f}% Fibonacci")
    
    # Overall analysis
    all_weights = np.array(all_weights)
    print(f"\nOverall: {len(all_weights)} weights")
    
    def to_phi_level(v, k=32):
        if abs(v) < 1e-10:
            return 0
        return int(round(k * np.log(abs(v)) / LN_PHI))
    
    levels = np.array([to_phi_level(w) for w in all_weights if abs(w) > 1e-10])
    sorted_levels = np.sort(levels)
    diffs = np.abs(np.diff(sorted_levels))
    fib_near = sum(1 for d in diffs if any(abs(d - f) <= 1 for f in FIBONACCI))
    
    print(f"  Unique φ-levels: {len(np.unique(levels))}")
    print(f"  Level differences near Fibonacci: {fib_near}/{len(diffs)} ({100*fib_near/len(diffs):.1f}%)")
    
    return all_weights, layer_stats


def colorize_image(model, rgb: np.ndarray) -> np.ndarray:
    """Colorize a single image."""
    model.eval()
    
    # Resize to model input size
    img = Image.fromarray(rgb)
    img = img.resize((256, 256))
    rgb_resized = np.array(img)
    
    # Convert to LAB
    lab = rgb_to_lab(rgb_resized)
    L = lab[:, :, 0:1] / 50.0 - 1.0
    
    # To tensor
    L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        pred_ab = model(L_tensor)
    
    # Convert back
    pred_ab = pred_ab.squeeze().cpu().numpy().transpose(1, 2, 0) * 128.0
    
    # Combine L and predicted ab
    L_original = lab[:, :, 0:1]
    lab_pred = np.concatenate([L_original, pred_ab], axis=2)
    
    # Convert to RGB
    rgb_pred = lab_to_rgb(lab_pred)
    
    # Resize back to original size
    rgb_pred = np.array(Image.fromarray(rgb_pred).resize((rgb.shape[1], rgb.shape[0])))
    
    return rgb_pred


def run_training():
    """Run full training and analysis."""
    print("=" * 70)
    print("TRAINING COLORIZER FROM SCRATCH")
    print("=" * 70)
    
    # Load images
    print("\n1. LOADING DATA")
    print("-" * 50)
    
    image_paths = sorted(COCO_PATH.glob("*.jpg"))
    train_paths = image_paths[:200]
    val_paths = image_paths[200:250]
    test_paths = image_paths[250:255]
    
    print(f"   Train: {len(train_paths)} images")
    print(f"   Val: {len(val_paths)} images")
    print(f"   Test: {len(test_paths)} images")
    
    train_dataset = ColorDataset(train_paths, size=256)
    val_dataset = ColorDataset(val_paths, size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model
    print("\n2. CREATING MODEL")
    print("-" * 50)
    
    model = SimpleColorizer(base_channels=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   Device: {DEVICE}")
    
    # Train
    print("\n3. TRAINING")
    print("-" * 50)
    
    train_losses, val_losses = train_colorizer(model, train_loader, val_loader, epochs=15, lr=1e-3)
    
    # Analyze φ-structure
    all_weights, layer_stats = analyze_phi_structure(model)
    
    # Test
    print("\n4. TESTING")
    print("-" * 50)
    
    results = []
    for img_path in test_paths:
        rgb = np.array(Image.open(img_path).convert('RGB'))
        colorized = colorize_image(model, rgb)
        mae = np.abs(colorized.astype(float) - rgb.astype(float)).mean()
        results.append((img_path.stem, rgb, colorized, mae))
        print(f"   {img_path.stem}: MAE = {mae:.2f}")
    
    avg_mae = np.mean([r[3] for r in results])
    print(f"\n   Average MAE: {avg_mae:.2f}")
    
    # Visualize
    print("\n5. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    
    for i, (name, original, colorized, mae) in enumerate(results):
        gray = np.mean(original, axis=2).astype(np.uint8)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Colorized ({mae:.1f})' if i == 0 else f'{mae:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=50)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Trained Colorizer: Average MAE = {avg_mae:.1f}', fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "trained_colorizer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'trained_colorizer.png'}")
    
    # Save model
    torch.save(model.state_dict(), OUTPUT_PATH / "colorizer_weights.pth")
    print(f"   Model saved to: {OUTPUT_PATH / 'colorizer_weights.pth'}")
    
    return model, results, avg_mae, all_weights


if __name__ == "__main__":
    model, results, avg_mae, all_weights = run_training()
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    print(f"""
   Results:
   - Average MAE: {avg_mae:.2f}
   - Total parameters: {sum(p.numel() for p in model.parameters()):,}
   
   The key question:
   Does the trained model exhibit φ-structure?
   
   If YES: The hypothesis is validated - learning finds φ-structure
   If NO: We learn where the hypothesis breaks down
""")
