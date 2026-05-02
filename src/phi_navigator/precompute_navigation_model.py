#!/usr/bin/env python3
"""
Precompute Navigation-Ready Model

This script converts a Qwen2 model into a navigation-ready format by:
1. Pre-folding biases into weight matrices
2. Computing augmented SVD for attention V→O path
3. Optionally quantizing to integers
4. Saving to disk for fast loading

The result is a model that can be loaded without recomputing SVD each time.

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import numpy as np
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time

PHI = 1.6180339887498949


@dataclass
class NavigationLayerConfig:
    """Configuration for one layer's navigation data."""
    layer_idx: int
    k: int  # SVD truncation rank
    hidden_dim: int
    use_integer: bool
    precision: int
    U_scale: float
    S_scale: float
    Vt_scale: float


def precompute_navigation_model(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    output_dir: str = "navigation_model",
    k: int = 512,
    use_integer: bool = True,
    precision: int = 10000,
    n_layers: Optional[int] = None
):
    """
    Precompute and save navigation-ready model.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save navigation data
        k: SVD truncation rank (512 for 100% correlation)
        use_integer: Whether to quantize to integers
        precision: Integer quantization precision
        n_layers: Number of layers to process (None = all)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Precomputing navigation model from {model_name}")
    print(f"  Output: {output_dir}")
    print(f"  k={k}, use_integer={use_integer}, precision={precision}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    print(f"  Loaded in {time.time() - start:.1f}s")
    
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    n_layers = n_layers or model.config.num_hidden_layers
    
    # Save embeddings and LM head
    print("\nSaving embeddings and LM head...")
    embeddings = model.model.embed_tokens.weight.data.numpy()
    lm_head = model.lm_head.weight.data.numpy()
    final_norm = model.model.norm.weight.data.numpy()
    
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "lm_head.npy"), lm_head)
    np.save(os.path.join(output_dir, "final_norm.npy"), final_norm)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Process each layer
    print(f"\nProcessing {n_layers} layers...")
    layer_configs = []
    
    total_original_size = 0
    total_compressed_size = 0
    
    for layer_idx in range(n_layers):
        layer_start = time.time()
        layer = model.model.layers[layer_idx]
        
        # Extract weights
        W_v = layer.self_attn.v_proj.weight.data.numpy().reshape(4, 128, 3584)
        W_o = layer.self_attn.o_proj.weight.data.numpy().reshape(3584, 28, 128)
        b_v = layer.self_attn.v_proj.bias.data.numpy().reshape(4, 128)
        ln_weight = layer.input_layernorm.weight.data.numpy()
        ln2_weight = layer.post_attention_layernorm.weight.data.numpy()
        
        # Build combined matrix (sum over all heads) - THIS IS THE BIAS FOLDING
        A_combined = np.zeros((3584, 3584))
        b_combined = np.zeros(3584)
        
        for kv_head in range(4):
            for q_head in range(kv_head * 7, (kv_head + 1) * 7):
                W_o_q = W_o[:, q_head, :]
                A_combined += W_o_q @ W_v[kv_head]
                b_combined += W_o_q @ b_v[kv_head]
        
        # Merge bias into matrix: [A | b]
        A_merged = np.column_stack([A_combined, b_combined])
        
        # SVD
        U, S, Vt = np.linalg.svd(A_merged, full_matrices=False)
        
        # Truncate to k
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Track sizes
        original_size = A_combined.size + b_combined.size
        compressed_size = U_k.size + S_k.size + Vt_k.size
        total_original_size += original_size
        total_compressed_size += compressed_size
        
        # Integer quantization
        U_scale, S_scale, Vt_scale = 1.0, 1.0, 1.0
        
        if use_integer:
            U_scale = float(np.max(np.abs(U_k)))
            S_scale = float(np.max(np.abs(S_k)))
            Vt_scale = float(np.max(np.abs(Vt_k)))
            
            U_int = np.round((U_k / U_scale) * precision).astype(np.int16)
            S_int = np.round((S_k / S_scale) * precision).astype(np.int16)
            Vt_int = np.round((Vt_k / Vt_scale) * precision).astype(np.int16)
            
            # Save integer versions
            np.save(os.path.join(output_dir, f"layer_{layer_idx}_U_int.npy"), U_int)
            np.save(os.path.join(output_dir, f"layer_{layer_idx}_S_int.npy"), S_int)
            np.save(os.path.join(output_dir, f"layer_{layer_idx}_Vt_int.npy"), Vt_int)
        else:
            # Save float versions
            np.save(os.path.join(output_dir, f"layer_{layer_idx}_U.npy"), U_k)
            np.save(os.path.join(output_dir, f"layer_{layer_idx}_S.npy"), S_k)
            np.save(os.path.join(output_dir, f"layer_{layer_idx}_Vt.npy"), Vt_k)
        
        # Save layer norms
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_ln1.npy"), ln_weight)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_ln2.npy"), ln2_weight)
        
        # Save MLP weights (for now, keep exact - could compress later)
        mlp_gate = layer.mlp.gate_proj.weight.data.numpy()
        mlp_up = layer.mlp.up_proj.weight.data.numpy()
        mlp_down = layer.mlp.down_proj.weight.data.numpy()
        
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_mlp_gate.npy"), mlp_gate)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_mlp_up.npy"), mlp_up)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_mlp_down.npy"), mlp_down)
        
        # Save full attention projections (for multi-token attention)
        W_q = layer.self_attn.q_proj.weight.data.numpy()
        W_k = layer.self_attn.k_proj.weight.data.numpy()
        W_v_full = layer.self_attn.v_proj.weight.data.numpy()
        W_o_full = layer.self_attn.o_proj.weight.data.numpy()
        b_q = layer.self_attn.q_proj.bias.data.numpy()
        b_k = layer.self_attn.k_proj.bias.data.numpy()
        b_v_full = layer.self_attn.v_proj.bias.data.numpy()
        
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_W_q.npy"), W_q)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_W_k.npy"), W_k)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_W_v.npy"), W_v_full)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_W_o.npy"), W_o_full)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_b_q.npy"), b_q)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_b_k.npy"), b_k)
        np.save(os.path.join(output_dir, f"layer_{layer_idx}_b_v.npy"), b_v_full)
        
        # Store config
        config = NavigationLayerConfig(
            layer_idx=layer_idx,
            k=k,
            hidden_dim=hidden_dim,
            use_integer=use_integer,
            precision=precision,
            U_scale=U_scale,
            S_scale=S_scale,
            Vt_scale=Vt_scale,
        )
        layer_configs.append(asdict(config))
        
        layer_time = time.time() - layer_start
        if layer_idx % 7 == 0:
            print(f"  Layer {layer_idx}: {layer_time:.1f}s, compression={original_size/compressed_size:.2f}x")
    
    # Save global config
    global_config = {
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "k": k,
        "use_integer": use_integer,
        "precision": precision,
        "layers": layer_configs,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(global_config, f, indent=2)
    
    # Report compression
    print(f"\nCompression summary:")
    print(f"  Original attention size: {total_original_size * 4 / 1e9:.2f} GB")
    print(f"  Compressed SVD size: {total_compressed_size * (2 if use_integer else 4) / 1e9:.2f} GB")
    print(f"  Compression ratio: {total_original_size * 4 / (total_compressed_size * (2 if use_integer else 4)):.2f}x")
    
    print(f"\nNavigation model saved to: {output_dir}")
    
    del model
    return output_dir


class PrecomputedNavigator:
    """
    Navigator that loads precomputed navigation data.
    
    Much faster startup than computing SVD on the fly.
    """
    
    def __init__(self, model_dir: str):
        """Load precomputed navigation model."""
        self.model_dir = model_dir
        
        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        
        self.hidden_dim = self.config["hidden_dim"]
        self.vocab_size = self.config["vocab_size"]
        self.n_layers = self.config["n_layers"]
        self.k = self.config["k"]
        self.use_integer = self.config["use_integer"]
        self.precision = self.config["precision"]
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load embeddings and LM head
        self.embeddings = np.load(os.path.join(model_dir, "embeddings.npy"))
        self.lm_head = np.load(os.path.join(model_dir, "lm_head.npy"))
        self.final_norm = np.load(os.path.join(model_dir, "final_norm.npy"))
        
        # Load layer data
        self.layers = []
        for layer_cfg in self.config["layers"]:
            layer_idx = layer_cfg["layer_idx"]
            
            layer_data = {
                "config": layer_cfg,
                "ln1": np.load(os.path.join(model_dir, f"layer_{layer_idx}_ln1.npy")),
                "ln2": np.load(os.path.join(model_dir, f"layer_{layer_idx}_ln2.npy")),
                "mlp_gate": np.load(os.path.join(model_dir, f"layer_{layer_idx}_mlp_gate.npy")),
                "mlp_up": np.load(os.path.join(model_dir, f"layer_{layer_idx}_mlp_up.npy")),
                "mlp_down": np.load(os.path.join(model_dir, f"layer_{layer_idx}_mlp_down.npy")),
                "W_q": np.load(os.path.join(model_dir, f"layer_{layer_idx}_W_q.npy")),
                "W_k": np.load(os.path.join(model_dir, f"layer_{layer_idx}_W_k.npy")),
                "W_v": np.load(os.path.join(model_dir, f"layer_{layer_idx}_W_v.npy")),
                "W_o": np.load(os.path.join(model_dir, f"layer_{layer_idx}_W_o.npy")),
                "b_q": np.load(os.path.join(model_dir, f"layer_{layer_idx}_b_q.npy")),
                "b_k": np.load(os.path.join(model_dir, f"layer_{layer_idx}_b_k.npy")),
                "b_v": np.load(os.path.join(model_dir, f"layer_{layer_idx}_b_v.npy")),
            }
            
            if self.use_integer:
                layer_data["U_int"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_U_int.npy"))
                layer_data["S_int"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_S_int.npy"))
                layer_data["Vt_int"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_Vt_int.npy"))
            else:
                layer_data["U"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_U.npy"))
                layer_data["S"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_S.npy"))
                layer_data["Vt"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_Vt.npy"))
            
            self.layers.append(layer_data)
    
    def layer_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """RMSNorm."""
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        return (x / rms) * weight
    
    def attention_augmented(self, x_norm: np.ndarray, layer: dict) -> np.ndarray:
        """Compute attention using precomputed augmented SVD."""
        x_aug = np.append(x_norm, 1.0)
        cfg = layer["config"]
        
        if self.use_integer:
            y = layer["Vt_int"].astype(np.float32) @ x_aug
            y = y / self.precision * cfg["Vt_scale"]
            
            z = (layer["S_int"].astype(np.float32) / self.precision * cfg["S_scale"]) * y
            
            out = (layer["U_int"].astype(np.float32) / self.precision * cfg["U_scale"]) @ z
        else:
            y = layer["Vt"] @ x_aug
            z = layer["S"] * y
            out = layer["U"] @ z
        
        return out
    
    def mlp_forward(self, x_norm: np.ndarray, layer: dict) -> np.ndarray:
        """MLP forward pass."""
        gate = layer["mlp_gate"] @ x_norm
        up = layer["mlp_up"] @ x_norm
        
        silu_gate = gate / (1 + np.exp(-gate))
        hidden = silu_gate * up
        
        return layer["mlp_down"] @ hidden
    
    def forward(self, token_id: int) -> np.ndarray:
        """Forward pass for single token."""
        x = self.embeddings[token_id].copy()
        
        for layer in self.layers:
            # Attention
            x_norm = self.layer_norm(x, layer["ln1"])
            attn_out = self.attention_augmented(x_norm, layer)
            x = x + attn_out
            
            # MLP
            x_norm = self.layer_norm(x, layer["ln2"])
            mlp_out = self.mlp_forward(x_norm, layer)
            x = x + mlp_out
        
        return x
    
    def predict_next(self, token_id: int, top_k: int = 5):
        """Predict next token."""
        hidden = self.forward(token_id)
        hidden = self.layer_norm(hidden, self.final_norm)
        logits = self.lm_head @ hidden
        
        top_indices = np.argsort(-logits)[:top_k]
        return [(self.tokenizer.decode([idx]), logits[idx]) for idx in top_indices]
    
    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate text (single-token attention mode)."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(input_ids)
        
        for _ in range(max_tokens):
            hidden = self.forward(generated[-1])
            hidden = self.layer_norm(hidden, self.final_norm)
            logits = self.lm_head @ hidden
            
            next_token = int(np.argmax(logits))
            generated.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)


def benchmark_navigator(nav, n_tokens: int = 10):
    """Benchmark navigator speed."""
    import time
    
    prompt = "The capital of France is"
    input_ids = nav.tokenizer.encode(prompt, add_special_tokens=False)
    
    print(f"\nBenchmarking: {n_tokens} tokens")
    print(f"Prompt: {prompt!r}")
    
    start = time.time()
    output = nav.generate(prompt, max_tokens=n_tokens)
    elapsed = time.time() - start
    
    tokens_per_second = n_tokens / elapsed
    ms_per_token = elapsed * 1000 / n_tokens
    
    print(f"Output: {output!r}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")
    print(f"       {ms_per_token:.0f} ms/token")
    
    return tokens_per_second


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--precompute":
        # Precompute navigation model
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "navigation_model"
        precompute_navigation_model(output_dir=output_dir)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Benchmark precomputed navigator
        model_dir = sys.argv[2] if len(sys.argv) > 2 else "navigation_model"
        
        print(f"Loading precomputed navigator from {model_dir}...")
        start = time.time()
        nav = PrecomputedNavigator(model_dir)
        print(f"Loaded in {time.time() - start:.2f}s")
        
        benchmark_navigator(nav, n_tokens=10)
    
    else:
        print("Usage:")
        print("  python precompute_navigation_model.py --precompute [output_dir]")
        print("  python precompute_navigation_model.py --benchmark [model_dir]")
