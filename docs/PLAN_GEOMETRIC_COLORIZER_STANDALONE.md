# Plan: Geometric Colorizer Standalone GitHub Repository

## Overview

Extract the geometric colorizer from `truthspace-lcm` into a standalone, clone-and-run GitHub repository called **`geometric-colorizer`**.

The app captures frames from a USB webcam, converts them to grayscale, runs the grayscale through a geometric colorizer (DDColor with φ-soft gate replacement), and displays the original color feed side-by-side with the geometric recolorization in real-time.

## User Experience

```
$ git clone https://github.com/<user>/geometric-colorizer.git
$ cd geometric-colorizer
$ pip install -r requirements.txt
$ python colorize.py
```

First run downloads the DDColor model (~200MB) from HuggingFace and extracts geometric weights (cached for subsequent runs). A window opens showing:

- **Left panel:** Ground truth (what the camera actually sees, in color)
- **Right panel:** Geometric recolorization (grayscale → color via φ-soft gate network)

On-screen text shows:
- FPS counter
- Model info (V20 geometric, ~35M params)
- Controls: `[M] colormap mode  [S] save  [X] quit`

### Controls
| Key | Action |
|-----|--------|
| `M` | Toggle display mode (recolorized / side-by-side with grayscale / diff view) |
| `S` | Save current frame triplet (original, grayscale, recolorized) to `./captures/` |
| `X` | Quit |

## Architecture Summary

### What is the Geometric Colorizer?

DDColor is a 55M-parameter image colorization model (ConvNeXt encoder + transformer decoder). We reverse-engineered it and proved every component is a geometric (matrix) operation. The geometric version:

1. **Extracts all 55M weights** into a static `.npz` file
2. **Replaces GELU** with `φ-soft gate: (1/φ) × x × σ(φ·x)` — proven to match or beat GELU
3. **Replaces the 9-layer transformer decoder** (14.8M params) with a single color matrix (25.6K params)
4. **Compresses the UNet decoder** to rank 50% via SVD (12.4M → ~7.1M effective)
5. **Result:** ~35M params, statistically identical quality (Wilcoxon p > 0.05)

### The Pipeline
```
Webcam frame (color BGR)
    │
    ├──► Left panel: original color
    │
    ▼
Grayscale (cv2.cvtColor → GRAY → 3-channel)
    │
    ▼
ConvNeXt Encoder (18 blocks, φ-soft gate replaces GELU)
    │
    ▼
UNet Decoder (3 blocks, rank-50% SVD compressed)
    │
    ▼
Color Matrix (single matmul, replaces 9-layer transformer)
    │
    ▼
Refine Net (1×1 conv, trivial)
    │
    ▼
LAB → BGR output ──► Right panel: recolorized
```

## Source Files from truthspace-lcm

### Core Model Files

#### 1. `geometric_colorizer.py` — V20 Assembly (main model)
- **Source:** `/home/thorin/truthspace-lcm/phi_geometric/models/geometric_colorizer_v20_assembly.py` (585 lines)
- **What it provides:** `V20AssemblyColorizer` with `forward()` and `colorize()` methods
- **Changes needed:**
  - Remove `sys.path.insert` hacks
  - Remove dependency on `geometric_colorizer_v16_convnext.py` (inline what's needed or simplify)
  - Make `weights_path` and `color_matrix_path` configurable
  - Add GPU support (original runs on CPU only)
  - The `__main__` benchmark section should be removed (standalone test script instead)

#### 2. `geometric_colorizer_v16.py` — Full DDColor extraction (reference/fallback)
- **Source:** `/home/thorin/truthspace-lcm/phi_geometric/models/geometric_colorizer_v16_convnext.py` (396 lines)
- **What it provides:** `V16GeometricColorizer` — full DDColor replica with GELU (baseline for comparison)
- **Changes needed:**
  - Remove `sys.path.insert`
  - Fix `_init_position_embedding()` to not depend on `ddcolor` package path
  - Make weights path configurable

#### 3. `extract_weights.py` — Weight extraction script (run once)
- **NEW FILE** (based on patterns in V16 and ddcolor_geometric.py)
- Downloads DDColor from HuggingFace
- Extracts all weights to `weights/ddcolor_weights_static.npz`
- Extracts color matrix to `weights/v17_color_matrix.npz`
- Pre-computes and saves sinusoidal position embeddings
- Caches everything so subsequent runs are instant

#### 4. `colorize.py` — Main webcam app (NEW)
- **NEW FILE** — the entry point
- Captures webcam frames
- Converts to grayscale
- Runs through geometric colorizer
- Side-by-side display with original
- FPS counter, controls overlay

### Weight Files (Generated, NOT checked into git)

| File | Size | Source |
|------|------|--------|
| `weights/ddcolor_weights_static.npz` | ~204 MB | Extracted from DDColor via `extract_weights.py` |
| `weights/v17_color_matrix.npz` | ~100 KB | Extracted from DDColor via `extract_weights.py` |
| `weights/position_embedding.npz` | ~tiny | Pre-computed sinusoidal PE |

These are generated on first run. The `.gitignore` excludes them. A download/extract script handles setup.

### DDColor Reference Code (needed for weight extraction only)

The weight extraction step needs DDColor's model definition to load pretrained weights. Two options:

**Option A (Recommended): Minimal DDColor dependency**
- Include just enough DDColor code to load the model:
  - `ddcolor/model.py` — source: `/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference/ddcolor/model.py` (279 lines)
  - `ddcolor/__init__.py` — source: `/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference/ddcolor/__init__.py`
  - `basicsr/` subtree — source: `/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference/basicsr/` (the archs, specifically `ddcolor_arch_utils/`)
- These are only needed for `extract_weights.py`, not for inference

**Option B: pip install**
- `pip install ddcolor` if available, or clone DDColor repo
- Cleaner but adds external dependency

### Position Embedding

V16 currently loads the FULL DDColor model just to grab `pe_layer` (sinusoidal position embedding). This is wasteful. The standalone repo should:

1. **Pre-compute PE during extraction** and save to `weights/position_embedding.npz`
2. **Reimplement `PositionEmbeddingSine`** inline (~30 lines) — it's a standard sinusoidal encoding, no learned params

The implementation from `basicsr/archs/ddcolor_arch_utils/position_encoding.py`:
- Source: `/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference/basicsr/archs/ddcolor_arch_utils/position_encoding.py`

## Repository Structure

```
geometric-colorizer/
├── README.md
├── LICENSE                          # MIT
├── requirements.txt
├── .gitignore                       # Excludes weights/
├── colorize.py                      # Main webcam app (NEW)
├── geometric_colorizer.py           # V20 assembly model (cleaned from v20_assembly.py)
├── position_encoding.py             # Standalone sinusoidal PE (extracted from basicsr)
├── extract_weights.py               # One-time weight extraction from DDColor (NEW)
├── weights/                         # Generated directory (gitignored)
│   ├── ddcolor_weights_static.npz   # ~204 MB (extracted)
│   ├── v17_color_matrix.npz         # ~100 KB (extracted)
│   └── position_embedding.npz       # Pre-computed PE
├── ddcolor/                         # Minimal DDColor for weight extraction only
│   ├── __init__.py
│   └── model.py
└── basicsr/                         # Minimal basicsr for DDColor model definition
    └── archs/
        └── ddcolor_arch_utils/
            ├── __init__.py
            ├── convnext.py
            ├── transformer.py
            ├── transformer_utils.py
            ├── position_encoding.py
            ├── unet.py
            └── util.py
```

## requirements.txt

```
numpy
opencv-python
torch
torchvision
huggingface_hub
Pillow
scipy
```

Note: `transformers` is NOT needed (unlike phi-depth). We use `huggingface_hub` directly for `PyTorchModelHubMixin`.

## Key Implementation Notes

### Weight Extraction (first run)

```python
# extract_weights.py does this once:
from ddcolor import DDColor
from huggingface_hub import PyTorchModelHubMixin

class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)

model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')

# Extract all weights
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy()
for name, buf in model.named_buffers():
    weights[name] = buf.detach().cpu().numpy()

np.savez_compressed('weights/ddcolor_weights_static.npz', **weights)

# Extract color matrix (V17 discovery)
# ... (compute from decoder weights)

# Extract position embedding
pe_layer = model.decoder.color_decoder.pe_layer
# ... pre-compute for standard input sizes
```

### The φ-soft gate (core geometric replacement)

```python
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI  # 0.6180339887...

def phi_soft_gate(x):
    """Replaces GELU: (1/φ) × x × σ(φ·x)"""
    return INV_PHI * x * torch.sigmoid(PHI * x)
```

This is proven to match or beat GELU on colorization quality (Wilcoxon p > 0.05, Doc 247 Part 14).

### The color matrix (V17 discovery)

DDColor's 9-layer transformer decoder (14.8M params) can be replaced by a single matrix multiply:
```python
color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, img_features)
```
The `v17_color_matrix.npz` captures this. It's pre-computed during weight extraction.

### Grayscale → Color Pipeline

```python
# 1. Convert color frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# 2. Resize to model input size (256×256 for V20, 512×512 for V16)
resized = cv2.resize(gray_3ch, (256, 256))

# 3. To tensor, normalize
tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

# 4. Forward pass → ab channels in LAB space
ab_out = model.forward(tensor)

# 5. Combine with L channel from grayscale
img_lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2Lab)
L = img_lab[:, :, 0]
ab_np = ab_out[0, :2].permute(1, 2, 0).numpy()
ab_scaled = np.clip(ab_np + 128, 0, 255).astype(np.uint8)
output_lab = np.stack([L, ab_scaled[:, :, 0], ab_scaled[:, :, 1]], axis=-1)
output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
```

## README.md Outline

```markdown
# geometric-colorizer: Real-Time Image Colorization from Webcam

Captures your webcam feed, converts it to grayscale, and recolorizes it
using a geometric neural network — proving that colorization is pure geometry.

## What is this?

[Screenshot placeholder]

Left: original camera feed. Right: geometric recolorization from grayscale.

The geometric colorizer replaces DDColor's GELU activation with a φ-soft gate
and its 9-layer transformer decoder with a single matrix multiply, reducing
parameters from 55M to ~35M with no quality loss.

| Component | DDColor | Geometric |
|-----------|---------|-----------|
| Encoder activation | GELU | φ-soft: (1/φ)·x·σ(φ·x) |
| Color decoder | 9-layer transformer (14.8M) | Single matrix (25.6K) |
| UNet decoder | Full rank (12.4M) | Rank-50% SVD (~7.1M) |
| **Total params** | **55M** | **~35M** |
| Quality (RMSE) | baseline | statistically identical |

## Quick Start

    git clone https://github.com/<user>/geometric-colorizer.git
    cd geometric-colorizer
    pip install -r requirements.txt
    python colorize.py

First run downloads DDColor weights (~200 MB) and extracts geometric
parameters. Subsequent runs start instantly from cached weights.

## Controls

| Key | Action |
|-----|--------|
| M   | Toggle display mode |
| S   | Save frames to ./captures/ |
| X   | Quit |

## How It Works

DDColor (by Kang et al.) colorizes images using a ConvNeXt encoder and
transformer decoder. We proved every component is a geometric (matrix)
operation and replaced key nonlinearities with φ-arithmetic equivalents:

- **φ-soft gate**: `(1/φ) × x × σ(φ·x)` replaces GELU with 0 extra parameters
- **Color matrix**: A single 256×256 matrix replaces 9 transformer layers
- **SVD compression**: UNet weights compressed to rank 50% losslessly

The golden ratio φ = (1+√5)/2 appears because it's the optimal self-similar
scaling factor — the same reason it appears in attention heads (see our
Qwen2-7B reverse engineering work).

## Requirements

- Python 3.8+
- USB webcam
- ~2GB RAM (CPU) or GPU recommended for real-time
- ~200MB disk for cached weights

## License

MIT

## Credits

- [DDColor](https://github.com/piddnad/DDColor) by Kang et al. for the base model
- Part of the [TruthSpace Geometric LCM](https://github.com/<user>/truthspace-lcm) project
```

## Versions to Include

**Primary: V20 Assembly** (recommended default)
- φ-soft gate, color matrix, UNet rank-50%
- ~35M params, works at 256×256, faster
- Best demonstrates the geometric thesis

**Optional: V16 Full DDColor** (for comparison/validation)
- Full replica with GELU (original behavior)
- 55M params, works at 512×512, higher resolution
- Useful for A/B comparison: "Does the geometric version match?"

The webcam app should default to V20 but allow `--v16` flag for comparison mode showing both side by side.

## Implementation Steps (for the new chat)

1. **Create repo structure** — `mkdir geometric-colorizer && cd geometric-colorizer`
2. **Copy and clean `geometric_colorizer.py`** — Merge V20 assembly into a single self-contained file:
   - Inline the weight loading, encoder, UNet, color matrix, refine
   - Remove V16 import dependency
   - Add configurable `weights_dir` parameter
   - Add GPU support option
3. **Extract `position_encoding.py`** — Copy `PositionEmbeddingSine` from basicsr (standalone, ~60 lines)
4. **Copy DDColor reference code** — Minimal `ddcolor/` and `basicsr/` for weight extraction
5. **Write `extract_weights.py`** — Downloads model, extracts all weights + color matrix + PE
6. **Write `colorize.py`** — Webcam capture + display loop:
   - Load model (run extraction if weights missing)
   - Capture frame → grayscale → colorize → display side-by-side
   - FPS counter, controls overlay, save functionality
   - X to quit, M for mode, S to save
7. **Write `requirements.txt`**
8. **Write `README.md`**
9. **Write `.gitignore`** — Exclude `weights/`, `captures/`, `__pycache__/`
10. **Test** — `python colorize.py` should work end-to-end

## Key Source Files Reference

All source files that the new chat may need to reference:

| File | Location in truthspace-lcm | Role |
|------|---------------------------|------|
| V20 Assembly | `phi_geometric/models/geometric_colorizer_v20_assembly.py` | **Main model** (585 lines) |
| V16 ConvNeXt | `phi_geometric/models/geometric_colorizer_v16_convnext.py` | Reference full DDColor (396 lines) |
| V20 φ-scaffold | `phi_geometric/models/geometric_colorizer_v20_phi_scaffold.py` | Gate variants (292 lines) |
| V17 Minimal | `phi_geometric/models/geometric_colorizer_v17_minimal.py` | Color matrix discovery |
| DDColor model | `phi_chat/experiments/ddcolor_reference/ddcolor/model.py` | Original DDColor definition (279 lines) |
| DDColor init | `phi_chat/experiments/ddcolor_reference/ddcolor/__init__.py` | Package init |
| basicsr archs | `phi_chat/experiments/ddcolor_reference/basicsr/archs/` | ConvNeXt, transformer, UNet definitions |
| Position encoding | `phi_chat/experiments/ddcolor_reference/basicsr/archs/ddcolor_arch_utils/position_encoding.py` | Sinusoidal PE |
| DDColor geometric | `phi_chat/experiments/ddcolor_reference/ddcolor_geometric.py` | Weight extraction patterns |
| Extracted weights | `phi_geometric/evaluations/ddcolor_weights_static.npz` | 204MB weights (can copy instead of re-extracting) |
| Color matrix | `phi_geometric/evaluations/v17_color_matrix.npz` | 100KB color matrix |
| Doc 228 | `docs/design_considerations/228_geometric_colorizer_experiments.md` | Experiment history |
| Doc 229 | `docs/design_considerations/229_geometric_colorizer_shape_patterns.md` | Shape patterns |

## Differences from phi-depth Standalone

| Aspect | phi-depth | geometric-colorizer |
|--------|-----------|-------------------|
| Model source | HuggingFace (DA2) | HuggingFace (DDColor) → extracted weights |
| Weight size | 125 bytes | ~204 MB (extracted) |
| First run | Downloads DA2 backbone | Downloads DDColor + extracts weights |
| Inference | DA2 backbone + φ-decoder | Full geometric forward pass (no DDColor needed) |
| Display | Camera + depth map | Camera (color) + recolorization |
| Input processing | Direct camera frame | Camera → grayscale → 3-channel |
