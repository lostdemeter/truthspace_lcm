"""
Compress φ-encoded weights using per-row uint8 quantization.

Each weight element goes from 3 bytes (int8 sign + int16 exp) to 2 bytes
(int8 sign + uint8 quantized exp), with per-row min/max metadata.

Compression: 1.5x (19.6 GB → 13.1 GB for 28 layers)
Correlation: 0.99992+ per weight matrix

Usage:
    python compress_phi_weights.py [--verify]
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, '.')
from phi_geometric.inference.phi_integer import phi_to_float

SRC_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
DST_DIR = 'experiments/model_reverse_engineering_v2/phi_model_compressed'

WEIGHT_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def compress_matrix(signs, exps):
    """
    Compress a weight matrix using per-row uint8 quantization.

    Returns:
        signs: int8 (unchanged)
        quant_exps: uint8 (per-row quantized exponents)
        row_min: int16 (per-row minimum exponent)
        row_max: int16 (per-row maximum exponent)
    """
    row_min = exps.min(axis=1).astype(np.int16)
    row_max = exps.max(axis=1).astype(np.int16)

    row_range = (row_max.astype(np.int32) - row_min.astype(np.int32))
    row_range_safe = np.maximum(row_range, 1)  # avoid div by zero

    # Quantize: map [row_min, row_max] → [0, 255]
    shifted = exps.astype(np.int32) - row_min.astype(np.int32)[:, np.newaxis]
    quant = np.round(shifted * 255.0 / row_range_safe[:, np.newaxis]).astype(np.uint8)

    return signs, quant, row_min, row_max


def decompress_matrix(signs, quant_exps, row_min, row_max):
    """Decompress back to (signs, int16 exponents)."""
    row_range = (row_max.astype(np.int32) - row_min.astype(np.int32))
    row_range_safe = np.maximum(row_range, 1)
    exps = (row_min.astype(np.int32)[:, np.newaxis] +
            quant_exps.astype(np.int32) * row_range_safe[:, np.newaxis] // 255)
    return signs, exps.astype(np.int16)


def compress_layer(src_dir, dst_dir, layer_idx, verify=False):
    """Compress one layer's weights."""
    src = os.path.join(src_dir, f'layer_{layer_idx:02d}')
    dst = os.path.join(dst_dir, f'layer_{layer_idx:02d}')
    os.makedirs(dst, exist_ok=True)

    total_orig = 0
    total_comp = 0

    for name in WEIGHT_NAMES:
        path = os.path.join(src, f'{name}.npz')
        if not os.path.exists(path):
            continue

        data = np.load(path)
        signs = data['signs']
        exps = data['exponents']

        orig_bytes = signs.nbytes + exps.nbytes

        # Compress
        c_signs, c_quant, c_min, c_max = compress_matrix(signs, exps)

        # Save compressed
        comp_path = os.path.join(dst, f'{name}.npz')
        np.savez_compressed(comp_path,
                            signs=c_signs,
                            quant_exps=c_quant,
                            row_min=c_min,
                            row_max=c_max)

        comp_bytes = c_signs.nbytes + c_quant.nbytes + c_min.nbytes + c_max.nbytes
        total_orig += orig_bytes
        total_comp += comp_bytes

        if verify:
            d_signs, d_exps = decompress_matrix(c_signs, c_quant, c_min, c_max)
            orig_f = phi_to_float(signs, exps).flatten()
            dec_f = phi_to_float(d_signs, d_exps).flatten()
            corr = float(np.corrcoef(orig_f, dec_f)[0, 1])
            max_err = int(np.max(np.abs(exps.astype(np.int32) - d_exps.astype(np.int32))))
            print(f'    {name:10s}  corr={corr:.8f}  max_exp_err={max_err}')

    # Copy norms and biases unchanged (they're tiny)
    for extra in ['norms.npz', 'biases.npz']:
        src_path = os.path.join(src, extra)
        if os.path.exists(src_path):
            dst_path = os.path.join(dst, extra)
            data = np.load(src_path)
            np.savez_compressed(dst_path, **dict(data))

    return total_orig, total_comp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Verify compression quality')
    parser.add_argument('--layers', type=int, default=28, help='Number of layers to compress')
    args = parser.parse_args()

    print("=" * 70)
    print("  φ-Weight Compression: Per-Row uint8 Quantization")
    print("=" * 70)

    os.makedirs(DST_DIR, exist_ok=True)

    # Copy config
    for name in ['config.json', 'verification.json']:
        src = os.path.join(SRC_DIR, name)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(DST_DIR, name))

    grand_orig = 0
    grand_comp = 0

    for layer_idx in range(args.layers):
        t0 = time.time()
        orig, comp = compress_layer(SRC_DIR, DST_DIR, layer_idx, verify=args.verify)
        dt = time.time() - t0
        grand_orig += orig
        grand_comp += comp
        print(f'  Layer {layer_idx:2d}: {orig/1e6:.0f}MB → {comp/1e6:.0f}MB  '
              f'({orig/comp:.2f}x)  {dt:.1f}s')

    print(f'\n  Total: {grand_orig/1e9:.2f} GB → {grand_comp/1e9:.2f} GB  '
          f'({grand_orig/grand_comp:.2f}x compression)')
    print(f'  RAM estimate for gimli: {grand_comp/1e9 + 1.5:.1f} GB (16 GB available)')

    # Save a marker file with compression metadata
    meta_path = os.path.join(DST_DIR, 'compression_meta.npz')
    np.savez(meta_path,
             method='per_row_uint8',
             original_phi_grid=128,
             layers=args.layers)
    print(f'\n  Compressed model saved to: {DST_DIR}')


if __name__ == '__main__':
    main()
