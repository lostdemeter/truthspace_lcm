"""
Phase 17D: The Gate Pattern as a Binary Code

Phase 17C proved: GELU creates a push-pull system where BOTH alive and dead
channels carry information. The gate pattern (which channels on/off) is
image-dependent.

NEW HYPOTHESIS: The MLP is a matched-filter sparse coder.
The gate pattern IS the encoding. If it has low dimensionality,
we can compress the PAIR (PW1, PW2) together.

Tests:
  1. Effective dimensionality of binary gate patterns across spatial positions
  2. Gate pattern clustering — do similar image regions produce similar codes?
  3. Gate-to-output correlation — how much of PW2 output is explained by just the PATTERN?
  4. Code stability — same image, different pixels: how many distinct codes?
  5. The bias structure — does the bias alone predict the gate pattern?
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


# ================================================================
# Collect gate patterns from real images
# ================================================================
print("\nCollecting gate patterns...")

N_IMGS = 5
# We'll focus on a few key blocks across stages
target_blocks = [(0, 1), (1, 1), (2, 0), (2, 4), (2, 8), (3, 0)]

gate_data = {k: [] for k in target_blocks}  # block -> list of [H*W, 4C] gate patterns

for img_idx in range(300, 300 + N_IMGS * 2):
    if len(gate_data[target_blocks[0]]) >= N_IMGS:
        break
    im = cv2.imread(images[img_idx])
    if im is None:
        continue

    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (t - mean_t) / std_t

    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0, 3, 1, 2)

        for stage_idx in range(4):
            dim = dims[stage_idx]
            if stage_idx > 0:
                prefix = f'encoder.arch.downsample_layers.{stage_idx}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (dims[stage_idx-1],),
                                 v16._get_weight(f'{prefix}.0.weight'),
                                 v16._get_weight(f'{prefix}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{prefix}.1.weight'),
                             v16._get_weight(f'{prefix}.1.bias'), stride=2)

            for block_idx in range(depths[stage_idx]):
                residual = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))

                key = (stage_idx, block_idx)
                if key in target_blocks:
                    # Collect binary gate pattern: [1, H, W, 4C] -> [H*W, 4C]
                    gate_binary = (pre_gelu > 0).float()[0].reshape(-1, pre_gelu.shape[-1])
                    # Also collect continuous pattern (GELU output magnitude)
                    gate_data[key].append({
                        'binary': gate_binary.numpy(),
                        'pre_gelu': pre_gelu[0].reshape(-1, pre_gelu.shape[-1]).numpy(),
                        'spatial_shape': (pre_gelu.shape[1], pre_gelu.shape[2]),
                    })

                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb


# ================================================================
# TEST 1: Effective dimensionality of gate patterns
# ================================================================
print()
print('=' * 70)
print('TEST 1: Effective Dimensionality of Gate Patterns')
print('=' * 70)
print()

print(f"{'Block':<8} {'Spatial':<10} {'4C':<6} {'Bits possible':<15} "
      f"{'PCA 90%':<10} {'PCA 95%':<10} {'PCA 99%':<10} {'Compress':<10}")
print("-" * 79)

for key in target_blocks:
    if not gate_data[key]:
        continue

    # Stack all spatial positions from all images
    all_binary = np.vstack([d['binary'] for d in gate_data[key]])  # [N_total, 4C]
    N, dim_4c = all_binary.shape
    spatial = gate_data[key][0]['spatial_shape']

    # PCA of binary patterns
    pca = PCA()
    pca.fit(all_binary)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    rank90 = np.searchsorted(cumvar, 0.90) + 1
    rank95 = np.searchsorted(cumvar, 0.95) + 1
    rank99 = np.searchsorted(cumvar, 0.99) + 1

    compress = dim_4c / rank90

    print(f"  {key[0]}.{key[1]:<5} {spatial[0]}×{spatial[1]:<6} {dim_4c:<6} "
          f"{dim_4c:<15} {rank90:<10} {rank95:<10} {rank99:<10} {compress:<10.1f}x")


# ================================================================
# TEST 2: How many distinct gate patterns exist?
# ================================================================
print()
print('=' * 70)
print('TEST 2: Distinct Gate Patterns per Block')
print('=' * 70)
print()

print(f"{'Block':<8} {'Total positions':<16} {'Unique patterns':<16} "
      f"{'Uniqueness %':<14} {'Hamming radius':<15}")
print("-" * 69)

for key in target_blocks:
    if not gate_data[key]:
        continue

    # First image only (to keep manageable)
    binary = gate_data[key][0]['binary']  # [H*W, 4C]
    N, dim_4c = binary.shape

    # Count unique rows
    unique = np.unique(binary, axis=0)
    n_unique = len(unique)

    # Average Hamming distance between random pairs of patterns
    idx = np.random.choice(N, min(500, N), replace=False)
    sample = binary[idx]
    hamming_dists = []
    for i in range(min(100, len(sample))):
        for j in range(i+1, min(100, len(sample))):
            hamming_dists.append(np.sum(sample[i] != sample[j]))
    mean_hamming = np.mean(hamming_dists)

    print(f"  {key[0]}.{key[1]:<5} {N:<16} {n_unique:<16} "
          f"{n_unique/N*100:<13.1f}% {mean_hamming:<15.1f}")


# ================================================================
# TEST 3: Bias predicts the gate pattern
# ================================================================
print()
print('=' * 70)
print('TEST 3: How Much Does the Bias Predict?')
print('=' * 70)
print()

print("The PW1 bias determines the DEFAULT gate state (without input).")
print("If bias < 0, the channel defaults to 'dead'.")
print("The input must overcome the bias to activate the channel.")
print()

print(f"{'Block':<8} {'Mean bias':<10} {'Bias<0 %':<10} {'Bias predicts gate':<20} "
      f"{'Input flips %':<14}")
print("-" * 62)

for key in target_blocks:
    if not gate_data[key]:
        continue

    prefix = f'encoder.arch.stages.{key[0]}.{key[1]}'
    bias = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()

    mean_bias = bias.mean()
    pct_neg = (bias < 0).mean() * 100

    # Gate prediction from bias alone: channel on if bias > 0
    bias_prediction = (bias > 0).astype(float)  # [4C]

    # Actual gate patterns
    binary = gate_data[key][0]['binary']  # [H*W, 4C]
    actual_mean = binary.mean(axis=0)  # [4C] average activation per channel

    # How well does bias sign predict average activation?
    bias_acc = np.mean((bias_prediction > 0.5) == (actual_mean > 0.5))

    # How many channels get FLIPPED by the input?
    # Channels where bias < 0 but sometimes activate (input overcomes bias)
    flipped = np.mean((bias < 0) & (actual_mean > 0.05))
    # And channels where bias > 0 but sometimes deactivate
    flipped += np.mean((bias > 0) & (actual_mean < 0.95))

    print(f"  {key[0]}.{key[1]:<5} {mean_bias:<10.3f} {pct_neg:<10.0f} "
          f"{bias_acc:<20.1%} {flipped:<14.1%}")


# ================================================================
# TEST 4: Information in the gate pattern vs the magnitude
# ================================================================
print()
print('=' * 70)
print('TEST 4: Gate Pattern vs Magnitude — Where\'s the Information?')
print('=' * 70)
print()

# Decompose pre-GELU into: sign (binary gate) × magnitude
# Then test: which carries more information for the PW2 output?

print("Reconstructing PW2 output from:")
print("  A) Full GELU output (baseline)")
print("  B) Just the binary gate pattern (sign only, magnitude = 1)")
print("  C) Just the magnitude (all channels alive, ignore gate)")
print()

for key in target_blocks:
    if not gate_data[key]:
        continue

    prefix = f'encoder.arch.stages.{key[0]}.{key[1]}'
    W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()

    pre_gelu_np = gate_data[key][0]['pre_gelu']  # [H*W, 4C]
    binary_np = gate_data[key][0]['binary']        # [H*W, 4C]

    # Full GELU output
    pre_gelu_t = torch.from_numpy(pre_gelu_np).float()
    full_gelu = geometric_gelu(pre_gelu_t).numpy()

    # Binary-only: sign pattern with unit magnitude
    sign_pattern = 2 * binary_np - 1  # {-1, +1}

    # Magnitude-only: all channels "alive" with original magnitude
    magnitude_only = np.abs(pre_gelu_np)

    # PW2 outputs
    W2_np = W2
    out_full = full_gelu @ W2_np.T
    out_sign = sign_pattern @ W2_np.T
    out_magnitude = magnitude_only @ W2_np.T

    # Correlation with full output
    corr_sign = np.corrcoef(out_full.flatten(), out_sign.flatten())[0, 1]
    corr_magnitude = np.corrcoef(out_full.flatten(), out_magnitude.flatten())[0, 1]

    # Explained variance (R²)
    ss_full = np.sum(out_full**2)
    r2_sign = 1 - np.sum((out_full - out_sign * np.std(out_full) / (np.std(out_sign) + 1e-10))**2) / ss_full
    r2_mag = 1 - np.sum((out_full - out_magnitude * np.std(out_full) / (np.std(out_magnitude) + 1e-10))**2) / ss_full

    print(f"  Block {key[0]}.{key[1]}:")
    print(f"    Sign pattern  → corr with full: {corr_sign:.4f}")
    print(f"    Magnitude     → corr with full: {corr_magnitude:.4f}")
    print(f"    {'SIGN WINS' if abs(corr_sign) > abs(corr_magnitude) else 'MAGNITUDE WINS'}")
    print()


# ================================================================
# TEST 5: The bias as the DEFAULT code
# ================================================================
print()
print('=' * 70)
print('TEST 5: Bias Structure — The Lock')
print('=' * 70)
print()

print("If PW1 bias determines the default gate state,")
print("how much of the PW2 output is already determined by the bias alone?")
print()

for key in target_blocks:
    if not gate_data[key]:
        continue

    prefix = f'encoder.arch.stages.{key[0]}.{key[1]}'
    bias = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
    W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
    b2 = v16._get_weight(f'{prefix}.pwconv2.bias').numpy()

    # Default GELU output (input = 0, just bias through GELU)
    bias_t = torch.from_numpy(bias).float()
    default_gelu = geometric_gelu(bias_t).numpy()

    # Default PW2 output from bias alone
    default_output = default_gelu @ W2.T + b2  # [C]

    # Actual PW2 output from real image
    pre_gelu_np = gate_data[key][0]['pre_gelu']
    full_gelu = geometric_gelu(torch.from_numpy(pre_gelu_np).float()).numpy()
    actual_output = full_gelu @ W2.T + b2  # [H*W, C]

    # How much of the actual output is explained by the default?
    actual_mean = actual_output.mean(axis=0)
    corr = np.corrcoef(default_output, actual_mean)[0, 1]

    # Fraction of variance explained by default
    residual = actual_output - default_output  # broadcast [H*W, C]
    var_total = np.var(actual_output)
    var_residual = np.var(residual)

    print(f"  Block {key[0]}.{key[1]}:")
    print(f"    Default output (bias only) corr with mean output: {corr:.4f}")
    print(f"    Variance explained by default: {(1 - var_residual/var_total)*100:.1f}%")
    print(f"    → {'Bias dominates!' if (1 - var_residual/var_total) > 0.5 else 'Input signal dominates.'}")
    print()


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 17D SUMMARY')
print('=' * 70)
