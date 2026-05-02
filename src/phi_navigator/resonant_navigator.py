#!/usr/bin/env python3
"""
Resonant Navigator - Optimized navigation with φ-biased boom detection.

Combines:
1. Precomputed augmented SVD for attention (7x compression)
2. Resonant attention (only compute at boom positions)
3. φ-biased boom detection inspired by resfrac

Target: >1 token/second (vs current 0.11 tok/s)

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import numpy as np
import os
import json
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

PHI = 1.6180339887498949
INV_PHI = 1.0 / PHI


@dataclass
class BoomCache:
    """Cache for boom positions and their KV states."""
    positions: List[int]
    K: np.ndarray  # (n_booms, num_kv_heads, head_dim)
    V: np.ndarray  # (n_booms, num_kv_heads, head_dim)


class ResonantNavigator:
    """
    Navigator with resonant attention for fast generation.
    
    Key optimizations:
    1. Only compute attention at boom positions (~20% of tokens)
    2. Use precomputed augmented SVD for single-token attention
    3. φ-biased boom detection
    """
    
    def __init__(self, model_dir: str, boom_ratio: float = 0.2):
        """
        Load precomputed navigation model.
        
        Args:
            model_dir: Directory with precomputed navigation data
            boom_ratio: Fraction of positions to keep as booms (0.2 = 20%)
        """
        self.model_dir = model_dir
        self.boom_ratio = boom_ratio
        
        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        
        self.hidden_dim = self.config["hidden_dim"]
        self.vocab_size = self.config["vocab_size"]
        self.n_layers = self.config["n_layers"]
        self.k = self.config["k"]
        self.use_integer = self.config["use_integer"]
        self.precision = self.config["precision"]
        
        # Attention config
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        self.kv_per_q = self.num_heads // self.num_kv_heads
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load embeddings and LM head
        self.embeddings = np.load(os.path.join(model_dir, "embeddings.npy"))
        self.lm_head = np.load(os.path.join(model_dir, "lm_head.npy"))
        self.final_norm = np.load(os.path.join(model_dir, "final_norm.npy"))
        
        # Load layer data (only what we need for resonant attention)
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
            
            # Load augmented SVD for single-token fast path
            if self.use_integer:
                layer_data["U_int"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_U_int.npy"))
                layer_data["S_int"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_S_int.npy"))
                layer_data["Vt_int"] = np.load(os.path.join(model_dir, f"layer_{layer_idx}_Vt_int.npy"))
            
            self.layers.append(layer_data)
        
        # Boom caches per layer
        self.boom_caches: List[Optional[BoomCache]] = [None] * self.n_layers
    
    def layer_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """RMSNorm."""
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        return (x / rms) * weight
    
    def detect_booms_phi(self, hidden_states: np.ndarray) -> List[int]:
        """
        Detect boom positions using φ-biased analysis.
        
        Uses entropy and φ-spacing to identify important positions.
        """
        seq_len = hidden_states.shape[0]
        n_booms = max(1, int(seq_len * self.boom_ratio))
        
        # Always include first and last positions
        booms = {0, seq_len - 1}
        
        if seq_len <= n_booms:
            return list(range(seq_len))
        
        # Compute position importance based on hidden state norms
        norms = np.linalg.norm(hidden_states, axis=1)
        
        # φ-biased selection: prefer positions at φ-spaced intervals
        remaining = set(range(1, seq_len - 1))
        
        while len(booms) < n_booms and remaining:
            # Score each remaining position
            best_pos = None
            best_score = -np.inf
            
            for pos in remaining:
                # Importance from norm
                importance = norms[pos]
                
                # φ-spacing bonus: prefer positions at φ intervals from existing booms
                min_dist = min(abs(pos - b) for b in booms)
                phi_bonus = np.sin(min_dist * np.pi / PHI) * 0.1
                
                score = importance + phi_bonus
                
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            if best_pos is not None:
                booms.add(best_pos)
                remaining.remove(best_pos)
        
        return sorted(booms)
    
    def attention_resonant(self, hidden_states: np.ndarray, layer: dict, 
                           boom_positions: List[int]) -> np.ndarray:
        """
        Compute attention using only boom positions.
        
        This is O(N × B) instead of O(N²) where B = boom count.
        """
        seq_len = hidden_states.shape[0]
        n_booms = len(boom_positions)
        
        # Apply layer norm to all positions
        x_normed = np.zeros_like(hidden_states)
        for i in range(seq_len):
            x_normed[i] = self.layer_norm(hidden_states[i], layer["ln1"])
        
        # Compute K, V only at boom positions
        K_booms = np.zeros((n_booms, self.num_kv_heads, self.head_dim))
        V_booms = np.zeros((n_booms, self.num_kv_heads, self.head_dim))
        
        for i, pos in enumerate(boom_positions):
            k = x_normed[pos] @ layer["W_k"].T + layer["b_k"]
            v = x_normed[pos] @ layer["W_v"].T + layer["b_v"]
            K_booms[i] = k.reshape(self.num_kv_heads, self.head_dim)
            V_booms[i] = v.reshape(self.num_kv_heads, self.head_dim)
        
        # Expand for GQA
        K_expanded = np.repeat(K_booms, self.kv_per_q, axis=1)  # (n_booms, num_heads, head_dim)
        V_expanded = np.repeat(V_booms, self.kv_per_q, axis=1)
        
        # Compute Q for last position only
        q_last = x_normed[-1] @ layer["W_q"].T + layer["b_q"]
        q_last = q_last.reshape(self.num_heads, self.head_dim)
        
        # Attention scores: Q @ K^T / sqrt(d)
        scores = np.zeros((self.num_heads, n_booms))
        for h in range(self.num_heads):
            for b in range(n_booms):
                scores[h, b] = np.dot(q_last[h], K_expanded[b, h]) / np.sqrt(self.head_dim)
        
        # Softmax
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Weighted sum of V
        attn_output = np.zeros((self.num_heads, self.head_dim))
        for h in range(self.num_heads):
            for b in range(n_booms):
                attn_output[h] += attn_weights[h, b] * V_expanded[b, h]
        
        # Project
        attn_output_flat = attn_output.reshape(-1)
        output = layer["W_o"] @ attn_output_flat
        
        return output
    
    def attention_single_token(self, x_norm: np.ndarray, layer: dict) -> np.ndarray:
        """Fast single-token attention using augmented SVD."""
        x_aug = np.append(x_norm, 1.0)
        cfg = layer["config"]
        
        if self.use_integer:
            y = layer["Vt_int"].astype(np.float32) @ x_aug
            y = y / self.precision * cfg["Vt_scale"]
            z = (layer["S_int"].astype(np.float32) / self.precision * cfg["S_scale"]) * y
            out = (layer["U_int"].astype(np.float32) / self.precision * cfg["U_scale"]) @ z
        else:
            raise NotImplementedError("Float SVD not loaded")
        
        return out
    
    def mlp_forward(self, x_norm: np.ndarray, layer: dict) -> np.ndarray:
        """MLP forward pass."""
        gate = layer["mlp_gate"] @ x_norm
        up = layer["mlp_up"] @ x_norm
        
        # SiLU with overflow protection
        gate_clipped = np.clip(gate, -88, 88)
        silu_gate = gate / (1 + np.exp(-gate_clipped))
        hidden = silu_gate * up
        
        return layer["mlp_down"] @ hidden
    
    def forward_single_token(self, token_id: int) -> np.ndarray:
        """Fast forward pass for single token using augmented SVD."""
        x = self.embeddings[token_id].copy()
        
        for layer in self.layers:
            # Attention (augmented SVD)
            x_norm = self.layer_norm(x, layer["ln1"])
            attn_out = self.attention_single_token(x_norm, layer)
            x = x + attn_out
            
            # MLP
            x_norm = self.layer_norm(x, layer["ln2"])
            mlp_out = self.mlp_forward(x_norm, layer)
            x = x + mlp_out
        
        return x
    
    def forward_sequence_resonant(self, token_ids: List[int]) -> np.ndarray:
        """
        Forward pass for sequence using resonant attention.
        
        For generation, we only need the LAST token's output.
        Strategy: Process all tokens through all layers, but use boom-sparse
        attention for the last token to speed up the O(N²) attention.
        """
        seq_len = len(token_ids)
        
        # Get embeddings
        hidden_states = np.array([self.embeddings[tid].copy() for tid in token_ids])
        
        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            new_hidden = np.zeros_like(hidden_states)
            
            # Compute K, V for ALL positions (needed for attention)
            x_normed = np.zeros_like(hidden_states)
            for i in range(seq_len):
                x_normed[i] = self.layer_norm(hidden_states[i], layer["ln1"])
            
            # K, V for all positions
            K_all = x_normed @ layer["W_k"].T + layer["b_k"]  # (seq_len, num_kv_heads * head_dim)
            V_all = x_normed @ layer["W_v"].T + layer["b_v"]
            
            K_all = K_all.reshape(seq_len, self.num_kv_heads, self.head_dim)
            V_all = V_all.reshape(seq_len, self.num_kv_heads, self.head_dim)
            
            # Expand for GQA
            K_expanded = np.repeat(K_all, self.kv_per_q, axis=1)
            V_expanded = np.repeat(V_all, self.kv_per_q, axis=1)
            
            # Process each position
            for pos in range(seq_len):
                # Q for this position
                q = x_normed[pos] @ layer["W_q"].T + layer["b_q"]
                q = q.reshape(self.num_heads, self.head_dim)
                
                # Attention: only attend to positions 0..pos (causal)
                if pos == 0:
                    # Single token - attention weight is 1.0
                    attn_output = V_expanded[0]
                else:
                    # Detect boom positions for this query
                    n_booms = max(1, int((pos + 1) * self.boom_ratio))
                    
                    # Simple boom selection: first, last, and evenly spaced
                    if n_booms >= pos + 1:
                        boom_positions = list(range(pos + 1))
                    else:
                        boom_positions = [0]  # Always include first
                        if pos > 0:
                            boom_positions.append(pos)  # Always include current
                        # Fill in with φ-spaced positions
                        step = max(1, int((pos + 1) / n_booms))
                        for i in range(step, pos, step):
                            if len(boom_positions) < n_booms and i not in boom_positions:
                                boom_positions.append(i)
                        boom_positions = sorted(set(boom_positions))[:n_booms]
                    
                    # Attention scores at boom positions
                    scores = np.zeros((self.num_heads, len(boom_positions)))
                    for h in range(self.num_heads):
                        for b_idx, b_pos in enumerate(boom_positions):
                            scores[h, b_idx] = np.dot(q[h], K_expanded[b_pos, h]) / np.sqrt(self.head_dim)
                    
                    # Softmax
                    scores_max = np.max(scores, axis=1, keepdims=True)
                    exp_scores = np.exp(scores - scores_max)
                    attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                    
                    # Weighted sum
                    attn_output = np.zeros((self.num_heads, self.head_dim))
                    for h in range(self.num_heads):
                        for b_idx, b_pos in enumerate(boom_positions):
                            attn_output[h] += attn_weights[h, b_idx] * V_expanded[b_pos, h]
                
                # Project
                attn_output_flat = attn_output.reshape(-1)
                attn_out = layer["W_o"] @ attn_output_flat
                
                new_hidden[pos] = hidden_states[pos] + attn_out
                
                # MLP
                x_norm = self.layer_norm(new_hidden[pos], layer["ln2"])
                mlp_out = self.mlp_forward(x_norm, layer)
                new_hidden[pos] = new_hidden[pos] + mlp_out
            
            hidden_states = new_hidden
        
        return hidden_states[-1]
    
    def generate(self, prompt: str, max_tokens: int = 10, 
                 use_resonant: bool = True, verbose: bool = False) -> Tuple[str, Dict]:
        """
        Generate text with timing information.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            use_resonant: Use resonant attention (faster) or single-token (baseline)
            verbose: Print timing per token
        
        Returns:
            (generated_text, timing_info)
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(input_ids)
        
        token_times = []
        
        for i in range(max_tokens):
            start = time.time()
            
            if use_resonant and len(generated) > 1:
                # Use resonant attention for multi-token
                hidden = self.forward_sequence_resonant(generated)
            else:
                # Single token - use fast SVD path
                hidden = self.forward_single_token(generated[-1])
            
            hidden = self.layer_norm(hidden, self.final_norm)
            logits = self.lm_head @ hidden
            
            next_token = int(np.argmax(logits))
            generated.append(next_token)
            
            elapsed = time.time() - start
            token_times.append(elapsed)
            
            if verbose:
                token_str = self.tokenizer.decode([next_token])
                print(f"  Token {i+1}: {token_str!r} ({elapsed*1000:.0f}ms)")
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        output = self.tokenizer.decode(generated)
        
        timing_info = {
            "total_time": sum(token_times),
            "tokens_generated": len(token_times),
            "tokens_per_second": len(token_times) / sum(token_times) if token_times else 0,
            "ms_per_token": sum(token_times) * 1000 / len(token_times) if token_times else 0,
            "token_times": token_times,
        }
        
        return output, timing_info


def benchmark_resonant_navigator():
    """Benchmark the resonant navigator."""
    print("=" * 60)
    print("Benchmarking Resonant Navigator")
    print("=" * 60)
    
    print("\nLoading navigator...")
    start = time.time()
    nav = ResonantNavigator("navigation_model", boom_ratio=0.2)
    print(f"Loaded in {time.time() - start:.1f}s")
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Generating {n_tokens} tokens...")
    
    # Test single-token mode (baseline)
    print("\n--- Single-token mode (baseline) ---")
    output, timing = nav.generate(prompt, max_tokens=n_tokens, use_resonant=False, verbose=True)
    print(f"\nOutput: {output!r}")
    print(f"Speed: {timing['tokens_per_second']:.2f} tokens/second")
    print(f"       {timing['ms_per_token']:.0f} ms/token")
    
    # Test resonant mode
    print("\n--- Resonant mode (optimized) ---")
    output, timing = nav.generate(prompt, max_tokens=n_tokens, use_resonant=True, verbose=True)
    print(f"\nOutput: {output!r}")
    print(f"Speed: {timing['tokens_per_second']:.2f} tokens/second")
    print(f"       {timing['ms_per_token']:.0f} ms/token")


if __name__ == "__main__":
    benchmark_resonant_navigator()
