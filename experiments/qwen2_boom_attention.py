#!/usr/bin/env python3
"""
Qwen2 Boom Attention: Real Model Integration
=============================================

Integrates cached boom attention with real Qwen2-7B model.

Key insight: Use actual attention patterns to detect boom positions,
then cache and reuse for generation speedup.

From Doc 159:
- 84-89% of attention mass at boom positions
- Booms occur at semantic boundaries
- Cross-layer consistency

This script:
1. Extracts real attention patterns from Qwen2
2. Detects boom positions using semantic structure
3. Validates accuracy vs full attention
4. Measures real speedup
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class BoomCache:
    """Cached boom structure for a layer."""
    boom_indices: torch.Tensor
    K_booms: torch.Tensor
    V_booms: torch.Tensor


class BoomAttentionLayer:
    """
    Wrapper that replaces standard attention with boom attention.
    """
    def __init__(self, max_booms: int = 32):
        self.max_booms = max_booms
        self.cache: Optional[BoomCache] = None
    
    def detect_booms_from_attention(
        self, 
        attn_weights: torch.Tensor,
        threshold: float = 0.02
    ) -> torch.Tensor:
        """
        Detect boom positions from actual attention weights.
        
        Args:
            attn_weights: (batch, heads, seq_len, seq_len)
            threshold: Minimum attention weight to be considered a boom
        
        Returns:
            boom_indices: Positions with high attention
        """
        # Average across heads and batch
        avg_attn = attn_weights.mean(dim=(0, 1))  # (seq_len, seq_len)
        
        # For each query position, find which keys get high attention
        # Sum attention received by each key position
        key_importance = avg_attn.sum(dim=0)  # (seq_len,)
        
        # Normalize
        key_importance = key_importance / key_importance.sum()
        
        # Select positions above threshold or top-k
        above_threshold = (key_importance > threshold).nonzero().squeeze(-1)
        
        if len(above_threshold) < self.max_booms:
            # Use top-k instead
            _, top_indices = torch.topk(key_importance, min(self.max_booms, len(key_importance)))
            boom_indices = torch.sort(top_indices)[0]
        else:
            boom_indices = above_threshold[:self.max_booms]
        
        return boom_indices
    
    def detect_booms_from_hidden(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect boom positions from hidden state structure.
        
        Uses gradient-based detection on hidden state norms.
        """
        # Compute hidden state norms
        norms = hidden_states.norm(dim=-1).squeeze(0)  # (seq_len,)
        seq_len = len(norms)
        
        if seq_len <= self.max_booms:
            return torch.arange(seq_len, device=hidden_states.device)
        
        # Gradient-based detection
        grad = torch.abs(norms[1:] - norms[:-1])
        grad = F.pad(grad, (1, 0), value=0)
        
        # Local maxima
        is_peak = torch.zeros(seq_len, device=hidden_states.device, dtype=torch.bool)
        if seq_len > 2:
            is_peak[1:-1] = (norms[1:-1] > norms[:-2]) & (norms[1:-1] > norms[2:])
        
        # Score = gradient + peak bonus
        scores = grad + is_peak.float() * grad.mean()
        scores[0] = scores.max() + 1  # Always include first
        scores[-1] = scores.max() + 0.5  # Always include last
        
        _, top_indices = torch.topk(scores, min(self.max_booms, seq_len))
        return torch.sort(top_indices)[0]
    
    def cache_from_kv(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        boom_indices: torch.Tensor
    ):
        """Cache K and V at boom positions."""
        self.cache = BoomCache(
            boom_indices=boom_indices,
            K_booms=K[:, :, boom_indices, :],
            V_booms=V[:, :, boom_indices, :]
        )
    
    def forward(
        self,
        Q: torch.Tensor,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Compute attention using cached booms.
        
        Args:
            Q: Query tensor (batch, heads, seq_len, head_dim)
            use_cache: Whether to use cached boom attention
        
        Returns:
            Attention output
        """
        if not use_cache or self.cache is None:
            raise ValueError("Cache not initialized")
        
        d_k = np.sqrt(Q.shape[-1])
        
        # Compute scores only for boom positions
        scores = torch.matmul(Q, self.cache.K_booms.transpose(-2, -1)) / d_k
        
        # Causal masking
        seq_len = Q.shape[2]
        positions = torch.arange(seq_len, device=Q.device).unsqueeze(1)
        boom_pos = self.cache.boom_indices.unsqueeze(0)
        causal_mask = positions < boom_pos
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and output
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, self.cache.V_booms)


def extract_qkv_from_layer(layer, hidden_states, position_ids, config):
    """Extract Q, K, V from a Qwen2 layer."""
    # Apply input layernorm
    hidden_norm = layer.input_layernorm(hidden_states)
    
    # Get Q, K, V projections
    bsz, seq_len, _ = hidden_norm.shape
    
    q = layer.self_attn.q_proj(hidden_norm)
    k = layer.self_attn.k_proj(hidden_norm)
    v = layer.self_attn.v_proj(hidden_norm)
    
    # Reshape for multi-head attention - get from config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Expand K, V for GQA
    num_key_value_groups = num_heads // num_kv_heads
    k = k.repeat_interleave(num_key_value_groups, dim=1)
    v = v.repeat_interleave(num_key_value_groups, dim=1)
    
    return q, k, v


def test_boom_attention_accuracy_with_model(model, tokenizer):
    """Test boom attention accuracy on real Qwen2 model."""
    print("=" * 70)
    print("QWEN2 BOOM ATTENTION - ACCURACY TEST")
    print("=" * 70)
    
    prompts = [
        "The capital of France is",
        "In mathematics, the number pi equals approximately",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        seq_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            # Get full model output
            outputs_full = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            full_text = tokenizer.decode(outputs_full[0], skip_special_tokens=True)
        
        print(f"  Full model: '{full_text}'")
        print(f"  Seq len: {seq_len}")
        
        # Now test with boom attention on a single forward pass
        with torch.no_grad():
            # Get hidden states and attention
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            # Analyze attention patterns
            for layer_idx in [0, 13, 27]:  # First, middle, last
                attn = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)
                
                # Create boom layer
                boom_layer = BoomAttentionLayer(max_booms=min(16, seq_len))
                boom_indices = boom_layer.detect_booms_from_attention(attn)
                
                # Compute attention mass at booms
                avg_attn = attn.mean(dim=(0, 1))  # (seq, seq)
                key_importance = avg_attn.sum(dim=0)
                boom_mass = key_importance[boom_indices].sum() / key_importance.sum()
                
                print(f"  Layer {layer_idx}: {len(boom_indices)} booms capture {boom_mass*100:.1f}% attention")


def benchmark_boom_generation_with_model(model, tokenizer):
    """Benchmark boom attention for generation."""
    print("\n" + "=" * 70)
    print("QWEN2 BOOM ATTENTION - GENERATION BENCHMARK")
    print("=" * 70)
    
    # Use a longer prompt to see real speedup
    prompt = """The theory of relativity, developed by Albert Einstein in the early 20th century, 
revolutionized our understanding of space, time, and gravity. The special theory of relativity, 
published in 1905, introduced the famous equation E=mc² and showed that the speed of light is 
constant for all observers. The general theory of relativity, completed in 1915, described gravity 
as the curvature of spacetime caused by mass and energy. This theory predicted phenomena such as 
gravitational waves, black holes, and the bending of light around massive objects. Einstein's work 
fundamentally changed physics and our understanding of the universe. The implications of relativity 
extend to modern technology including GPS satellites, which must account for relativistic effects 
to maintain accuracy. In summary, the theory states that"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Warm up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # Benchmark standard generation
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs_std = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    time_std = (time.perf_counter() - start) * 1000
    
    text_std = tokenizer.decode(outputs_std[0], skip_special_tokens=True)
    tokens_generated = outputs_std.shape[1] - inputs.input_ids.shape[1]
    
    print(f"\nStandard generation:")
    print(f"  Text: '{text_std}'")
    print(f"  Tokens: {tokens_generated}")
    print(f"  Time: {time_std:.1f}ms ({time_std/tokens_generated:.1f}ms/token)")
    
    # Now measure attention computation time separately
    print("\n--- Attention Timing Analysis ---")
    
    with torch.no_grad():
        # Get hidden states for one layer
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden = outputs.hidden_states[14]  # Middle layer input
        layer = model.model.layers[14]
        
        # Time standard attention
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            q, k, v = extract_qkv_from_layer(layer, hidden, None, model.config)
            d_k = np.sqrt(q.shape[-1])
            scores = torch.matmul(q, k.transpose(-2, -1)) / d_k
            attn_weights = F.softmax(scores, dim=-1)
            out_std = torch.matmul(attn_weights, v)
        torch.cuda.synchronize()
        time_attn_std = (time.perf_counter() - start) / 10 * 1000
        
        seq_len = hidden.shape[1]
        
        print(f"\nSingle layer attention (seq_len={seq_len}):")
        print(f"  Standard: {time_attn_std:.3f}ms")
        
        # Test different boom counts using ATTENTION-based detection
        # First compute full attention to get the pattern
        d_k = np.sqrt(q.shape[-1])
        full_scores = torch.matmul(q, k.transpose(-2, -1)) / d_k
        mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1).bool()
        full_scores = full_scores.masked_fill(mask, float('-inf'))
        full_attn = F.softmax(full_scores, dim=-1)
        
        print(f"\n  {'Booms':>8} {'Time (ms)':>12} {'Speedup':>10} {'Correlation':>12} {'Attn Mass':>10}")
        print("  " + "-" * 56)
        
        for max_booms in [16, 32, 64, 128]:
            if max_booms > seq_len:
                continue
                
            boom_layer = BoomAttentionLayer(max_booms=max_booms)
            # Use attention-based detection
            boom_indices = boom_layer.detect_booms_from_attention(full_attn)
            boom_layer.cache_from_kv(k, v, boom_indices)
            
            # Compute attention mass at booms
            avg_attn = full_attn.mean(dim=(0, 1))
            key_importance = avg_attn.sum(dim=0)
            attn_mass = key_importance[boom_indices].sum() / key_importance.sum()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                out_boom = boom_layer.forward(q)
            torch.cuda.synchronize()
            time_attn_boom = (time.perf_counter() - start) / 10 * 1000
            
            # Compute accuracy
            out_std_flat = out_std.float().flatten().cpu().numpy()
            out_boom_flat = out_boom.float().flatten().cpu().numpy()
            correlation = np.corrcoef(out_std_flat, out_boom_flat)[0, 1]
            
            speedup = time_attn_std / time_attn_boom
            
            print(f"  {len(boom_indices):>8} {time_attn_boom:>12.3f} {speedup:>10.2f}× {correlation:>12.4f} {attn_mass*100:>9.1f}%")


def test_full_layer_replacement():
    """Test replacing full attention with boom attention in generation."""
    print("\n" + "=" * 70)
    print("FULL LAYER REPLACEMENT TEST")
    print("=" * 70)
    
    print("\nLoading Qwen2-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    seq_len = inputs.input_ids.shape[1]
    
    print(f"\nPrompt: '{prompt}' (seq_len={seq_len})")
    
    # Get standard output
    with torch.no_grad():
        outputs_std = model(**inputs, output_hidden_states=True)
        logits_std = outputs_std.logits
        next_token_std = logits_std[0, -1, :].argmax()
        
    print(f"Standard next token: '{tokenizer.decode([next_token_std])}'")
    
    # Now try with boom attention on each layer
    print("\n--- Per-layer boom attention test ---")
    
    with torch.no_grad():
        hidden = model.model.embed_tokens(inputs.input_ids)
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        
        for layer_idx, layer in enumerate(model.model.layers):
            # Extract Q, K, V
            q, k, v = extract_qkv_from_layer(layer, hidden, position_ids)
            
            # Standard attention
            d_k = np.sqrt(q.shape[-1])
            scores = torch.matmul(q, k.transpose(-2, -1)) / d_k
            
            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_out_std = torch.matmul(attn_weights, v)
            
            # Boom attention
            boom_layer = BoomAttentionLayer(max_booms=min(16, seq_len))
            boom_indices = boom_layer.detect_booms_from_attention(attn_weights.unsqueeze(0))
            boom_layer.cache_from_kv(k, v, boom_indices)
            attn_out_boom = boom_layer.forward(q)
            
            # Compare
            correlation = np.corrcoef(
                attn_out_std.float().flatten().cpu().numpy(),
                attn_out_boom.float().flatten().cpu().numpy()
            )[0, 1]
            
            if layer_idx % 7 == 0:  # Print every 7th layer
                print(f"  Layer {layer_idx:2d}: {len(boom_indices)} booms, correlation={correlation:.4f}")
            
            # Continue with standard output for next layer
            attn_out_std = attn_out_std.transpose(1, 2).reshape(1, seq_len, -1)
            hidden = layer.self_attn.o_proj(attn_out_std)
            hidden = hidden + model.model.embed_tokens(inputs.input_ids)  # Residual (simplified)
            
            # MLP
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden


def test_token_accuracy_with_model(model, tokenizer):
    """Test if boom attention produces correct next token."""
    print("\n" + "=" * 70)
    print("TOKEN ACCURACY TEST")
    print("=" * 70)
    
    prompts = [
        "The capital of France is",
        "In mathematics, pi equals approximately",
        "The quick brown fox jumps over the",
        "Albert Einstein developed the theory of",
        "Water freezes at zero degrees",
    ]
    
    print(f"\n{'Prompt':<45} {'Standard':>12} {'Boom-64':>12} {'Match':>8}")
    print("-" * 80)
    
    correct = 0
    total = 0
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        seq_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            # Get hidden states
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            # Standard next token
            logits_std = outputs.logits
            next_token_std = logits_std[0, -1, :].argmax()
            token_std = tokenizer.decode([next_token_std])
            
            # Now compute with boom attention on last layer
            # Get hidden state before last layer
            hidden = outputs.hidden_states[-2]  # Before last layer
            layer = model.model.layers[-1]
            
            # Extract Q, K, V
            q, k, v = extract_qkv_from_layer(layer, hidden, None, model.config)
            
            # Compute full attention for boom detection
            d_k = np.sqrt(q.shape[-1])
            full_scores = torch.matmul(q, k.transpose(-2, -1)) / d_k
            mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1).bool()
            full_scores = full_scores.masked_fill(mask, float('-inf'))
            full_attn = F.softmax(full_scores, dim=-1)
            
            # Boom attention with 64 positions
            boom_layer = BoomAttentionLayer(max_booms=64)
            boom_indices = boom_layer.detect_booms_from_attention(full_attn)
            boom_layer.cache_from_kv(k, v, boom_indices)
            
            # Get boom attention output
            attn_out_boom = boom_layer.forward(q)
            
            # Complete the layer forward pass with boom attention
            attn_out_boom = attn_out_boom.transpose(1, 2).reshape(1, seq_len, -1)
            attn_out_boom = layer.self_attn.o_proj(attn_out_boom)
            
            # Residual + MLP
            hidden_boom = hidden + attn_out_boom
            residual = hidden_boom
            hidden_boom = layer.post_attention_layernorm(hidden_boom)
            hidden_boom = layer.mlp(hidden_boom)
            hidden_boom = residual + hidden_boom
            
            # Final norm and LM head
            hidden_boom = model.model.norm(hidden_boom)
            logits_boom = model.lm_head(hidden_boom)
            
            next_token_boom = logits_boom[0, -1, :].argmax()
            token_boom = tokenizer.decode([next_token_boom])
            
            match = "✓" if next_token_std == next_token_boom else "✗"
            if next_token_std == next_token_boom:
                correct += 1
            total += 1
            
            print(f"{prompt:<45} {token_std:>12} {token_boom:>12} {match:>8}")
    
    print("-" * 80)
    print(f"Token accuracy: {correct}/{total} ({correct/total*100:.1f}%)")


def main():
    print("=" * 70)
    print("QWEN2 BOOM ATTENTION INTEGRATION")
    print("=" * 70)
    print("""
Testing boom attention on real Qwen2-7B model:
1. Accuracy test - how much attention mass do booms capture?
2. Generation benchmark - real speedup measurement
3. Token accuracy - does boom attention produce correct tokens?
""")
    
    # Load model once
    print("\nLoading Qwen2-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager"
    )
    model.eval()
    
    # Run tests with shared model
    test_boom_attention_accuracy_with_model(model, tokenizer)
    benchmark_boom_generation_with_model(model, tokenizer)
    test_token_accuracy_with_model(model, tokenizer)


if __name__ == "__main__":
    main()
