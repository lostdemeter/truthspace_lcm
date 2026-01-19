#!/usr/bin/env python3
"""
Convert Qwen2-7B to φ-encoded format.

This script converts the HuggingFace Qwen2-7B-Instruct model to our
φ-encoded .phi format for 1.33x compression and 1.37x faster loading.

Usage:
    python convert_qwen2_to_phi.py --output /path/to/output

Author: TruthSpace LCM Team
License: GPLv3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Import our φ-encoding
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phi_model_storage import phi_encode, save_phi_tensor, SCALE, PHI, LOG_PHI


def convert_qwen2_to_phi(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    output_dir: str = "models/qwen2-7b-phi",
    layers: list = None,
):
    """
    Convert Qwen2 model to φ-encoded format.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for .phi files
        layers: List of layer indices to convert (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {model_name} to φ-encoded format")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Load config first
    print("\nLoading model config...")
    config = AutoConfig.from_pretrained(model_name)
    
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {config.num_key_value_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path / "tokenizer")
    
    # Load model
    print("\nLoading model (this may take a while)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    
    # Save config
    phi_config = {
        "model_name": model_name,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": getattr(config, 'rope_theta', 10000.0),
        "phi_scale": SCALE,
        "phi_version": 1,
    }
    
    with open(output_path / "config.json", "w") as f:
        json.dump(phi_config, f, indent=2)
    
    # Determine layers to convert
    n_layers = config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))
    
    print(f"\nConverting {len(layers)} layers...")
    
    total_original = 0
    total_phi = 0
    
    # Convert embeddings
    print("\nConverting embeddings...")
    embed_weight = model.model.embed_tokens.weight.data.float().numpy()
    embed_path = output_path / "embed_tokens.phi"
    save_phi_tensor(embed_path, embed_weight)
    
    original_bytes = embed_weight.nbytes
    phi_bytes = embed_path.stat().st_size
    total_original += original_bytes
    total_phi += phi_bytes
    print(f"  embed_tokens: {original_bytes/1e6:.1f} MB -> {phi_bytes/1e6:.1f} MB")
    
    # Convert layers
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}/{n_layers}...")
        layer_dir = output_path / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        
        layer = model.model.layers[layer_idx]
        
        # Attention weights
        weights = {
            "q_proj": layer.self_attn.q_proj.weight.data,
            "k_proj": layer.self_attn.k_proj.weight.data,
            "v_proj": layer.self_attn.v_proj.weight.data,
            "o_proj": layer.self_attn.o_proj.weight.data,
        }
        
        # MLP weights
        if hasattr(layer, 'mlp'):
            weights["gate_proj"] = layer.mlp.gate_proj.weight.data
            weights["up_proj"] = layer.mlp.up_proj.weight.data
            weights["down_proj"] = layer.mlp.down_proj.weight.data
        
        # LayerNorm (keep as float32 - small and needs precision)
        weights["input_layernorm"] = layer.input_layernorm.weight.data
        weights["post_attention_layernorm"] = layer.post_attention_layernorm.weight.data
        
        for name, tensor in weights.items():
            tensor_np = tensor.float().numpy()
            path = layer_dir / f"{name}.phi"
            save_phi_tensor(path, tensor_np)
            
            original_bytes = tensor_np.nbytes
            phi_bytes = path.stat().st_size
            total_original += original_bytes
            total_phi += phi_bytes
            
            print(f"  {name}: {tensor_np.shape} -> {phi_bytes/1e6:.1f} MB")
    
    # Convert final layer norm
    print("\nConverting final layer norm...")
    final_ln = model.model.norm.weight.data.float().numpy()
    save_phi_tensor(output_path / "norm.phi", final_ln)
    total_original += final_ln.nbytes
    total_phi += (output_path / "norm.phi").stat().st_size
    
    # Convert lm_head
    print("Converting lm_head...")
    lm_head = model.lm_head.weight.data.float().numpy()
    save_phi_tensor(output_path / "lm_head.phi", lm_head)
    total_original += lm_head.nbytes
    total_phi += (output_path / "lm_head.phi").stat().st_size
    
    # Summary
    compression = total_original / total_phi
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Original size: {total_original / 1e9:.2f} GB")
    print(f"  φ-encoded size: {total_phi / 1e9:.2f} GB")
    print(f"  Compression: {compression:.2f}x")
    print(f"  Output: {output_path}")
    print("=" * 60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2 to φ-encoded format")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct", help="Model name")
    parser.add_argument("--output", default="models/qwen2-7b-phi", help="Output directory")
    parser.add_argument("--layers", type=int, nargs="+", help="Specific layers to convert")
    args = parser.parse_args()
    
    convert_qwen2_to_phi(args.model, args.output, args.layers)


if __name__ == "__main__":
    main()
