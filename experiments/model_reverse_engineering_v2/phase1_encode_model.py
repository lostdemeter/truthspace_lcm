#!/usr/bin/env python3
"""
Phase 1: φ-Encode Qwen2-7B
===========================

Converts all weight matrices to φ-integer format:
    value = sign × φ^(exponent / 128)

After this script completes, the GPU is never needed again.
All subsequent inference uses integer arithmetic only.

Output structure:
    phi_model/
    ├── config.json           — model config + encoding stats
    ├── embed_tokens.npz      — (152064, 3584) token embeddings
    ├── lm_head.npz           — (152064, 3584) output projection
    ├── final_norm.npz        — (3584,) final RMSNorm weight [float32]
    ├── layer_00/
    │   ├── q_proj.npz        — (3584, 3584)  attention Q
    │   ├── k_proj.npz        — (512, 3584)   attention K (GQA)
    │   ├── v_proj.npz        — (512, 3584)   attention V (GQA)
    │   ├── o_proj.npz        — (3584, 3584)  attention output
    │   ├── gate_proj.npz     — (18944, 3584) MLP gate
    │   ├── up_proj.npz       — (18944, 3584) MLP up
    │   ├── down_proj.npz     — (3584, 18944) MLP down
    │   ├── biases.npz        — Q/K/V biases [float32]
    │   └── norms.npz         — layernorm weights [float32]
    ├── layer_01/ ...
    └── verification.json     — per-component accuracy
"""

import sys
import os
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_NAME = "Qwen/Qwen2-7B"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "phi_model")


def encode_and_save(name: str, tensor: np.ndarray, path: str) -> dict:
    """φ-encode a weight tensor, save it, return stats."""
    t0 = time.perf_counter()
    phi = PhiEncoded.encode(tensor)
    encode_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    corr = phi.correlation(tensor)
    verify_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    phi.save(path)
    save_time = time.perf_counter() - t0

    disk_bytes = os.path.getsize(path)

    stats = {
        'name': name,
        'shape': list(tensor.shape),
        'params': int(tensor.size),
        'correlation': round(corr, 8),
        'original_bytes': int(tensor.astype(np.float32).nbytes),
        'encoded_bytes': int(phi.storage_bytes()),
        'disk_bytes': int(disk_bytes),
        'encode_ms': round(encode_time * 1000, 1),
        'save_ms': round(save_time * 1000, 1),
    }

    compression = tensor.astype(np.float32).nbytes / disk_bytes
    print(f"    {name:40s}  {str(tensor.shape):20s}  "
          f"corr={corr:.6f}  disk={disk_bytes/1e6:.1f}MB  "
          f"compress={compression:.1f}×  "
          f"({encode_time*1000:.0f}+{save_time*1000:.0f}ms)")

    return stats


def encode_layer(layer_idx: int, hf_layer, output_dir: str) -> list:
    """Encode all weights for one transformer layer."""
    layer_dir = os.path.join(output_dir, f'layer_{layer_idx:02d}')
    os.makedirs(layer_dir, exist_ok=True)

    stats = []

    # Weight matrices to φ-encode
    weight_sources = {
        'q_proj': hf_layer.self_attn.q_proj.weight,
        'k_proj': hf_layer.self_attn.k_proj.weight,
        'v_proj': hf_layer.self_attn.v_proj.weight,
        'o_proj': hf_layer.self_attn.o_proj.weight,
        'gate_proj': hf_layer.mlp.gate_proj.weight,
        'up_proj': hf_layer.mlp.up_proj.weight,
        'down_proj': hf_layer.mlp.down_proj.weight,
    }

    for wname, param in weight_sources.items():
        tensor = param.detach().float().numpy()
        path = os.path.join(layer_dir, f'{wname}.npz')
        stat = encode_and_save(
            f'layer_{layer_idx:02d}/{wname}', tensor, path
        )
        stats.append(stat)

    # Biases — store as float32 (tiny: <32KB total per layer)
    biases = {}
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        bias = getattr(hf_layer.self_attn, proj_name).bias
        if bias is not None:
            biases[f'{proj_name}_bias'] = bias.detach().float().numpy()

    o_bias = hf_layer.self_attn.o_proj.bias
    if o_bias is not None:
        biases['o_proj_bias'] = o_bias.detach().float().numpy()

    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        bias = getattr(hf_layer.mlp, proj_name).bias
        if bias is not None:
            biases[f'{proj_name}_bias'] = bias.detach().float().numpy()

    bias_path = os.path.join(layer_dir, 'biases.npz')
    if biases:
        np.savez_compressed(bias_path, **biases)
        bias_names = list(biases.keys())
        bias_bytes = sum(b.nbytes for b in biases.values())
        print(f"    {'biases':40s}  {bias_names}  {bias_bytes} bytes")
    else:
        # Save empty file to keep structure consistent
        np.savez_compressed(bias_path)
        print(f"    {'biases':40s}  (none)")

    # Layer norm weights — store as float32 (tiny: 7168 bytes each)
    norms = {
        'input_layernorm': hf_layer.input_layernorm.weight.detach().float().numpy(),
        'post_attention_layernorm': hf_layer.post_attention_layernorm.weight.detach().float().numpy(),
    }
    np.savez_compressed(os.path.join(layer_dir, 'norms.npz'), **norms)

    return stats


def main():
    print("=" * 90)
    print("  Phase 1: φ-Encode Qwen2-7B")
    print("  Format: sign(int8) × φ^(exponent(int16) / 128)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 90)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_stats = []
    total_params = 0
    total_original_bytes = 0
    total_disk_bytes = 0
    t_start = time.perf_counter()

    # ── Load model ──────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading {MODEL_NAME}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print()

    config = AutoConfig.from_pretrained(MODEL_NAME)

    # ── Encode embeddings ───────────────────────────────────────
    print("Encoding embed_tokens...")
    embed_w = model.model.embed_tokens.weight.detach().float().numpy()
    stat = encode_and_save(
        'embed_tokens', embed_w,
        os.path.join(OUTPUT_DIR, 'embed_tokens.npz')
    )
    all_stats.append(stat)
    del embed_w
    print()

    # ── Encode layers ───────────────────────────────────────────
    n_layers = len(model.model.layers)
    for i in range(n_layers):
        print(f"Layer {i}/{n_layers-1}:")
        layer_stats = encode_layer(i, model.model.layers[i], OUTPUT_DIR)
        all_stats.extend(layer_stats)

        # Free layer weights from GPU/CPU to reduce memory pressure
        model.model.layers[i] = None

        # Progress
        done = i + 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / done * (n_layers - done)
        print(f"    [{done}/{n_layers}] elapsed={elapsed:.0f}s  eta={eta:.0f}s")
        print()

    # ── Encode lm_head ──────────────────────────────────────────
    print("Encoding lm_head...")
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        lm_head_w = model.lm_head.weight.detach().float().numpy()
        stat = encode_and_save(
            'lm_head', lm_head_w,
            os.path.join(OUTPUT_DIR, 'lm_head.npz')
        )
        all_stats.append(stat)
        del lm_head_w
    else:
        print("    (tied to embed_tokens — will use embed_tokens.npz)")
    print()

    # ── Final norm ──────────────────────────────────────────────
    final_norm = model.model.norm.weight.detach().float().numpy()
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'final_norm.npz'),
        weight=final_norm
    )
    print(f"Final norm: {final_norm.shape}")
    print()

    # ── Free model ──────────────────────────────────────────────
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ── Save config ─────────────────────────────────────────────
    model_config = {
        'model_name': MODEL_NAME,
        'hidden_size': config.hidden_size,
        'num_attention_heads': config.num_attention_heads,
        'num_key_value_heads': config.num_key_value_heads,
        'intermediate_size': config.intermediate_size,
        'num_hidden_layers': config.num_hidden_layers,
        'head_dim': config.hidden_size // config.num_attention_heads,
        'vocab_size': config.vocab_size,
        'rope_theta': config.rope_theta,
        'tie_word_embeddings': config.tie_word_embeddings,
        'encoding': {
            'format': 'phi_encoded',
            'phi_grid': 128,
            'sign_dtype': 'int8',
            'exponent_dtype': 'int16',
            'bits_per_value': 24,
            'formula': 'sign × φ^(exponent / 128)',
        },
    }

    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)

    # ── Save verification ───────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'verification.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start

    total_params = sum(s['params'] for s in all_stats)
    total_original = sum(s['original_bytes'] for s in all_stats)
    total_disk = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, s['name'] + '.npz'))
        for s in all_stats
        if os.path.exists(os.path.join(OUTPUT_DIR, s['name'] + '.npz'))
    )

    # Also count layer subdirectory files
    total_disk = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fname in files:
            total_disk += os.path.getsize(os.path.join(root, fname))

    correlations = [s['correlation'] for s in all_stats]
    min_corr = min(correlations)
    avg_corr = np.mean(correlations)
    max_corr = max(correlations)

    print("=" * 90)
    print("  PHASE 1 COMPLETE — φ-Encoded Qwen2-7B")
    print("=" * 90)
    print()
    print(f"  Parameters:     {total_params:,}")
    print(f"  Float32 size:   {total_original / 1e9:.2f} GB")
    print(f"  Disk size:      {total_disk / 1e9:.2f} GB")
    print(f"  Compression:    {total_original / total_disk:.2f}× vs float32")
    print()
    print(f"  Correlation:    min={min_corr:.6f}  avg={avg_corr:.6f}  max={max_corr:.6f}")
    print()
    print(f"  Time:           {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Output:         {OUTPUT_DIR}")
    print()

    # Per-component summary
    print("  Component breakdown:")
    print(f"  {'Component':40s}  {'Params':>12s}  {'Disk MB':>8s}  {'Corr':>10s}")
    print("  " + "-" * 76)

    for s in all_stats:
        disk_mb = s.get('disk_bytes', 0) / 1e6
        print(f"  {s['name']:40s}  {s['params']:12,}  "
              f"{disk_mb:8.1f}  {s['correlation']:10.6f}")

    print()
    print("  The GPU is no longer needed.")
    print("  All subsequent computation uses integer arithmetic only.")
    print()


if __name__ == '__main__':
    main()
