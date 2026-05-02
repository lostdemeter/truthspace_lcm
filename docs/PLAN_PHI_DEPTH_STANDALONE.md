# Plan: phi-depth Standalone GitHub Repository

## Overview

Extract the real-time webcam depth estimation demo from `truthspace-lcm` into a standalone, clone-and-run GitHub repository called **`phi-depth`**.

The app captures frames from a USB webcam, runs them through Depth Anything V2's backbone, replaces DA2's decoder head with a 125-byte φ-arithmetic decoder, and displays the original camera feed side-by-side with a colorized depth map in real-time.

## User Experience

```
$ git clone https://github.com/<user>/phi-depth.git
$ cd phi-depth
$ pip install -r requirements.txt
$ python phi_depth.py
```

A window opens showing the webcam feed on the left and depth estimation on the right. On-screen text shows:
- FPS counter
- Current colormap name
- Controls reminder: `[M] colormap  [S] save  [X] quit`

### Controls
| Key | Action |
|-----|--------|
| `M` | Cycle colormap (magma → viridis → plasma → inferno → turbo) |
| `S` | Save current frame pair to `./captures/` |
| `X` | Quit |

## Repository Structure

```
phi-depth/
├── README.md                  # Project description, setup, usage
├── LICENSE                    # MIT
├── requirements.txt           # Minimal deps
├── phi_depth.py               # Main entry point (webcam → side-by-side display)
├── phi_decoder.py             # φ-arithmetic decoder (PhiDecoder, PhiConfig, PhiArray, PhiValue)
├── phi_compact.py             # Compact 125-byte storage format (CompactPhiWeights)
├── weights/
│   ├── phi_weights.bin        # Standard weights (203 bytes)
│   └── phi_weights_compact.bin # Compact weights (125 bytes)
└── fit_weights.py             # Optional: re-fit weights from DA2 (needs COCO images)
```

## Source Files from truthspace-lcm

Each file below needs to be copied and cleaned up for the standalone repo.

### 1. `phi_decoder.py` — Core decoder
- **Source:** `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/phi_decoder.py` (427 lines)
- **What it provides:** `PhiConfig`, `PhiValue`, `PhiArray`, `PhiDecoderWeights`, `PhiDecoder`, `extract_head_features()`
- **Changes needed:**
  - Remove `sys.path.insert` hacks
  - Keep as-is otherwise — this is clean, self-contained code

### 2. `phi_compact.py` — Compact storage format
- **Source:** `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/phi_compact.py` (279 lines)
- **What it provides:** `CompactPhiWeights` with 125-byte packing/unpacking
- **Changes needed:**
  - Remove `sys.path.insert`
  - Keep `from phi_decoder import ...` (same directory)

### 3. `phi_depth.py` — Main app (NEW, combine + simplify)
- **Based on:** `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/realtime_depth.py` (277 lines) and `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/demo_depth.py` (311 lines)
- **This is the main entry point.** Simplify from the existing two files into one clean script.
- **Key changes vs source:**
  - Quit key: `X` instead of `Q`
  - On-screen HUD: FPS, colormap name, controls text
  - Add `turbo` colormap to the cycle
  - Save to `./captures/` directory (auto-created)
  - Use `argparse` for `--camera` and `--weights`
  - No video/image fallback modes (keep it simple: webcam only)
  - Clean imports (no `sys.path` hacks)

### 4. `weights/phi_weights.bin` (203 bytes)
- **Source:** `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/phi_weights.bin`
- Binary copy, no changes needed

### 5. `weights/phi_weights_compact.bin` (125 bytes)
- **Source:** `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/phi_weights_compact.bin`
- Binary copy, no changes needed

### 6. `fit_weights.py` — Optional weight fitting
- **Source:** `/home/thorin/truthspace-lcm/experiments/phi_da2_decoder/fit_weights.py` (137 lines)
- **Changes needed:**
  - Remove hardcoded COCO path, use `--images` arg
  - Update output path to `weights/`
  - Add note that this is optional (pre-fitted weights are included)

## Files NOT to include

These are experimental/research files that would clutter the standalone repo:
- `phi_cuda.py` — CUDA-accelerated decoder (nice but adds complexity)
- `realtime_cuda.py` — Depends on phi_cuda
- `realtime_optimized.py` — Depends on phi_cuda + torch.compile
- `test_universal.py` — Depends on COCO val2017 images
- `universal_test.png` — 4MB test output image
- Any files from `experiments/unified_assembly/`

## requirements.txt

```
numpy
opencv-python
torch
transformers
Pillow
```

No pinned versions — let the user's environment resolve. The code uses standard APIs.

## README.md Outline

```markdown
# phi-depth: Real-Time Depth Estimation from Webcam

Real-time monocular depth estimation using a USB webcam.
Uses Depth Anything V2 as backbone with a 125-byte φ-arithmetic decoder head.

## What is this?

[Screenshot placeholder]

Camera feed (left) and depth map (right), running in real-time.

The φ-decoder replaces DA2's 108KB decoder head with 125 bytes of
φ-arithmetic weights, achieving 99.99% correlation with the original.

| Component | Size |
|-----------|------|
| DA2 backbone (ViT-S) | 94 MB (downloaded automatically) |
| φ-decoder weights | **125 bytes** |
| Correlation with full DA2 | 99.99% |

## Quick Start

    git clone https://github.com/<user>/phi-depth.git
    cd phi-depth
    pip install -r requirements.txt
    python phi_depth.py

First run downloads the DA2 backbone (~94 MB) from HuggingFace.

## Controls

| Key | Action |
|-----|--------|
| M   | Cycle colormap |
| S   | Save frame to ./captures/ |
| X   | Quit |

## How it works

DA2's decoder head is a linear projection from 32 features to depth.
We represent this projection using φ-exponent arithmetic:

    value = sign × φ^(exponent / k)

where φ = (1+√5)/2 is the golden ratio. This gives:
- Multiplication via exponent addition (no floating-point multiply)
- Equal relative precision at all scales
- 756,400× compression vs the full model

## Requirements

- Python 3.8+
- USB webcam
- GPU recommended (CPU works but slower)

## Re-fitting weights (optional)

If you want to re-derive the weights from scratch:

    python fit_weights.py --images /path/to/some/images/

The included weights already work universally for any image.

## License

MIT

## Credits

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for the backbone
- Part of the [TruthSpace Geometric LCM](https://github.com/<user>/truthspace-lcm) research project
```

## Implementation Steps (for the new chat)

1. **Create repo structure** — `mkdir phi-depth && cd phi-depth`
2. **Copy and clean `phi_decoder.py`** — Remove sys.path, keep everything else
3. **Copy and clean `phi_compact.py`** — Same treatment
4. **Copy weight files** — Binary copy to `weights/`
5. **Write `phi_depth.py`** — New simplified main app based on `realtime_depth.py`:
   - Side-by-side display
   - On-screen HUD with FPS, colormap, controls
   - X to quit, M for colormap, S to save
   - Clean arg parsing
6. **Clean `fit_weights.py`** — Remove hardcoded paths
7. **Write `requirements.txt`**
8. **Write `README.md`**
9. **Test** — `python phi_depth.py` should work with a webcam

## Key Implementation Notes

- The DA2 backbone (`depth-anything/Depth-Anything-V2-Small-hf`) is downloaded automatically from HuggingFace on first run via `transformers`. No manual download needed.
- The hook captures features from `model.head.activation1` — this is a specific layer in DA2's head that outputs 32-channel features at reduced resolution.
- The φ-decoder does: center features → dot product with 32 weights → add target mean. That's it. The entire "decoder" is a linear projection represented in φ-arithmetic.
- The weight file format is `PHI1` (203 bytes) or `PHI2` (125 bytes compact). Both contain 32 weight values, 32 feature means, and 1 target mean, all encoded as `sign × φ^(exponent/k)`.
