#!/usr/bin/env python3
"""
Boom-Newton Attention: O(N) Attention via Zero-Hunting
=======================================================

Combines two insights:
1. rhzeros: Newton's method finds zeros in O(1) iterations with cached derivatives
2. Doc 159: "Boom" positions (phase transitions) capture 84-89% of attention mass

The key realization: We don't need to find attention PEAKS.
We need to find attention ZEROS (phase transitions).

In zeta: zeros mark where the function changes sign
In attention: "booms" mark where attention focus shifts

Detection method (from Doc 159):
- Sign pattern analysis (alternation rate)
- φ-level variance tracking
- Ratio complexity (continued fractions)

This gives O(N) detection of the ~20% of positions that matter.

Author: Based on rhzeros + boom hypothesis
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = (1 + np.sqrt(5)) / 2


def detect_booms_sign_pattern(x: torch.Tensor, window: int = 3) -> List[int]:
    """
    Detect boom positions using sign pattern analysis.
    
    A "boom" occurs where the sign pattern changes - this indicates
    a phase transition in the attention landscape.
    
    Complexity: O(N)
    
    Args:
        x: 1D tensor of values (e.g., attention scores or hidden states)
        window: Window size for pattern analysis
    
    Returns:
        List of boom positions
    """
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    n = len(x_np)
    
    if n < window + 1:
        return list(range(n))
    
    booms = []
    
    # Compute sign changes
    signs = np.sign(x_np)
    
    for i in range(window, n):
        # Count sign changes in window
        window_signs = signs[i-window:i+1]
        changes = np.sum(np.abs(np.diff(window_signs)) > 0)
        
        # Boom = sudden change in alternation pattern
        if i > window:
            prev_changes = np.sum(np.abs(np.diff(signs[i-window-1:i])) > 0)
            if abs(changes - prev_changes) >= 2:
                booms.append(i)
    
    return booms


def detect_booms_phi_variance(x: torch.Tensor, window: int = 5) -> List[int]:
    """
    Detect boom positions using φ-level variance tracking.
    
    Convert values to φ-levels and track variance. Booms occur
    where variance spikes (phase transition).
    
    Complexity: O(N)
    """
    x_np = np.abs(x.cpu().numpy()) if isinstance(x, torch.Tensor) else np.abs(x)
    n = len(x_np)
    
    if n < window + 1:
        return list(range(n))
    
    # Convert to φ-levels
    x_np = np.clip(x_np, 1e-10, None)  # Avoid log(0)
    phi_levels = np.log(x_np) / np.log(PHI)
    
    booms = []
    prev_var = 0
    
    for i in range(window, n):
        window_levels = phi_levels[i-window:i+1]
        curr_var = np.var(window_levels)
        
        # Boom = variance spike
        if curr_var > 2 * prev_var and prev_var > 0:
            booms.append(i)
        
        prev_var = curr_var
    
    return booms


def detect_booms_gradient(scores: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Detect boom positions using gradient magnitude.
    
    Booms occur where the gradient of attention scores is large,
    indicating a transition between attention regions.
    
    Complexity: O(N)
    """
    scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    n = len(scores_np)
    
    if n < 2:
        return list(range(n))
    
    # Compute gradient
    grad = np.abs(np.diff(scores_np))
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)
    
    # Booms where gradient exceeds threshold
    boom_threshold = grad_mean + threshold * grad_std
    booms = [i+1 for i in range(len(grad)) if grad[i] > boom_threshold]
    
    return booms


def detect_booms_combined(
    hidden_states: torch.Tensor,
    min_booms: int = 5,
    max_booms: int = 50
) -> List[int]:
    """
    Combined boom detection using multiple methods.
    
    Positions that appear in multiple detection methods are
    more likely to be true booms.
    
    Complexity: O(N)
    """
    # Use last hidden state dimension as signal
    if hidden_states.dim() > 1:
        signal = hidden_states.mean(dim=-1)
        if signal.dim() > 1:
            signal = signal.squeeze()
    else:
        signal = hidden_states
    
    n = len(signal)
    
    # Run all detectors
    booms_sign = set(detect_booms_sign_pattern(signal))
    booms_phi = set(detect_booms_phi_variance(signal))
    booms_grad = set(detect_booms_gradient(signal))
    
    # Count votes for each position
    votes = {}
    for pos in range(n):
        votes[pos] = 0
        if pos in booms_sign:
            votes[pos] += 1
        if pos in booms_phi:
            votes[pos] += 1
        if pos in booms_grad:
            votes[pos] += 1
    
    # Sort by votes, then by position
    sorted_positions = sorted(votes.keys(), key=lambda p: (-votes[p], p))
    
    # Take positions with at least 1 vote, up to max_booms
    booms = [p for p in sorted_positions if votes[p] >= 1][:max_booms]
    
    # Ensure minimum number of booms
    if len(booms) < min_booms:
        # Add evenly spaced positions
        spacing = n // min_booms
        for i in range(0, n, spacing):
            if i not in booms:
                booms.append(i)
            if len(booms) >= min_booms:
                break
    
    return sorted(booms)


def boom_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    hidden_states: torch.Tensor = None,
    max_booms: int = 20
) -> Tuple[torch.Tensor, List[int]]:
    """
    Attention using boom-detected anchor positions.
    
    Instead of O(N²) full attention, we:
    1. Detect boom positions in O(N) using integer operations
    2. Compute attention only to boom positions: O(N × k)
    3. Total: O(N × k) where k << N
    
    Args:
        Q, K, V: Query, Key, Value tensors (batch, heads, seq_len, head_dim)
        hidden_states: Hidden states for boom detection (optional)
        max_booms: Maximum number of boom positions
    
    Returns:
        output: Attention output
        booms: List of boom positions
    """
    batch, heads, seq_len, head_dim = Q.shape
    d_k = np.sqrt(head_dim)
    
    # Detect booms
    if hidden_states is not None:
        # Use hidden states for detection
        if hidden_states.dim() == 3:
            signal = hidden_states[0, :, 0]  # First batch, all positions, first dim
        else:
            signal = hidden_states.flatten()[:seq_len]
        booms = detect_booms_combined(signal, max_booms=max_booms)
    else:
        # Use K norms as proxy
        k_norms = K.norm(dim=-1).mean(dim=(0, 1))  # (seq_len,)
        booms = detect_booms_combined(k_norms, max_booms=max_booms)
    
    if len(booms) == 0:
        booms = [seq_len - 1]
    
    # Only compute attention to boom positions
    boom_indices = torch.tensor(booms, device=Q.device)
    K_booms = K[:, :, boom_indices, :]
    V_booms = V[:, :, boom_indices, :]
    
    # Compute scores only for booms: O(N × k × d)
    scores = torch.matmul(Q, K_booms.transpose(-2, -1)) / d_k
    
    # Causal mask for booms
    positions = torch.arange(seq_len, device=Q.device).unsqueeze(1)
    boom_positions = boom_indices.unsqueeze(0)
    causal_mask = positions < boom_positions
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax over booms
    attn_weights = F.softmax(scores, dim=-1)
    
    # Output: O(N × k × d)
    output = torch.matmul(attn_weights, V_booms)
    
    return output, booms


def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard O(N²) attention for comparison."""
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    seq_len = Q.shape[2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def benchmark():
    """Benchmark boom attention vs standard attention."""
    print("=" * 70)
    print("BOOM-NEWTON ATTENTION BENCHMARK")
    print("=" * 70)
    
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    batch_size = 1
    heads = 28
    head_dim = 128
    max_booms = 20
    
    print(f"\nConfig: batch={batch_size}, heads={heads}, head_dim={head_dim}, max_booms={max_booms}")
    print(f"{'Seq Len':>10} {'Standard (ms)':>15} {'Boom (ms)':>15} {'Speedup':>10} {'Booms':>10} {'Boom %':>10}")
    print("-" * 75)
    
    for seq_len in seq_lengths:
        Q = torch.randn(batch_size, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        K = torch.randn(batch_size, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        V = torch.randn(batch_size, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        
        # Warm up
        _ = standard_attention(Q, K, V)
        _ = boom_attention(Q, K, V, max_booms=max_booms)
        
        # Benchmark standard
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            out_std = standard_attention(Q, K, V)
        torch.cuda.synchronize()
        time_std = (time.perf_counter() - start) / 10 * 1000
        
        # Benchmark boom
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            out_boom, booms = boom_attention(Q, K, V, max_booms=max_booms)
        torch.cuda.synchronize()
        time_boom = (time.perf_counter() - start) / 10 * 1000
        
        speedup = time_std / time_boom
        boom_pct = len(booms) / seq_len * 100
        
        print(f"{seq_len:>10} {time_std:>15.3f} {time_boom:>15.3f} {speedup:>10.2f}× {len(booms):>10} {boom_pct:>9.1f}%")
    
    print("\n" + "=" * 70)
    print("ACCURACY TEST")
    print("=" * 70)
    
    seq_len = 256
    Q = torch.randn(1, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    K = torch.randn(1, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    V = torch.randn(1, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    
    out_std = standard_attention(Q, K, V)
    
    print(f"\nSeq len: {seq_len}")
    print(f"{'Max Booms':>12} {'Correlation':>15} {'MSE':>15} {'Boom %':>12}")
    print("-" * 60)
    
    for max_b in [5, 10, 20, 40, 80, 128]:
        out_boom, booms = boom_attention(Q, K, V, max_booms=max_b)
        
        out_std_flat = out_std.flatten().cpu().numpy()
        out_boom_flat = out_boom.flatten().cpu().numpy()
        
        correlation = np.corrcoef(out_std_flat, out_boom_flat)[0, 1]
        mse = np.mean((out_std_flat - out_boom_flat) ** 2)
        boom_pct = len(booms) / seq_len * 100
        
        print(f"{max_b:>12} {correlation:>15.4f} {mse:>15.6f} {boom_pct:>11.1f}%")


def analyze_real_attention():
    """Analyze boom detection on real Qwen2 attention patterns."""
    print("\n" + "=" * 70)
    print("REAL ATTENTION ANALYSIS")
    print("=" * 70)
    
    print("\nLoading Qwen/Qwen2-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,  # Use float32 to avoid bfloat16 issues
        device_map="cuda",
        attn_implementation="eager"
    )
    model.eval()
    
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, the Riemann hypothesis states that",
        "The capital of France is Paris, known for the Eiffel Tower.",
    ]
    
    for prompt in prompts:
        print(f"\n--- '{prompt[:50]}...' ---")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        seq_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden states for boom detection
        hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
        
        # Detect booms
        booms = detect_booms_combined(hidden[:, 0], max_booms=20)
        
        # Get actual attention pattern
        attn = outputs.attentions[-1][0, :, -1, :].mean(dim=0).cpu().numpy()
        
        # Find true attention peaks
        peaks = []
        for i in range(1, len(attn) - 1):
            if attn[i] > attn[i-1] and attn[i] > attn[i+1]:
                peaks.append(i)
        
        # Compute overlap
        boom_set = set(booms)
        peak_set = set(peaks)
        overlap = len(boom_set & peak_set)
        
        # Compute attention mass at booms
        boom_mass = sum(attn[b] for b in booms if b < len(attn))
        total_mass = attn.sum()
        
        print(f"  Seq len: {seq_len}")
        print(f"  Booms detected: {len(booms)} ({len(booms)/seq_len*100:.1f}%)")
        print(f"  True peaks: {len(peaks)}")
        print(f"  Overlap: {overlap}/{len(peaks)}")
        print(f"  Attention mass at booms: {boom_mass/total_mass*100:.1f}%")
        
        # Show tokens at boom positions
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        print(f"  Boom tokens: {[tokens[b] for b in booms[:10] if b < len(tokens)]}")


def main():
    print("=" * 70)
    print("BOOM-NEWTON ATTENTION")
    print("=" * 70)
    print("""
Combining rhzeros zero-hunting with boom hypothesis:

1. BOOM DETECTION (O(N))
   - Sign pattern analysis
   - φ-level variance tracking
   - Gradient magnitude
   
2. SPARSE ATTENTION (O(N × k))
   - Only attend to boom positions
   - k = ~20% of N (from Doc 159)
   
3. THEORETICAL SPEEDUP
   - Standard: O(N²)
   - Boom: O(N × k) where k << N
   - For N=1000, k=20: 50× speedup
""")
    
    benchmark()
    
    try:
        analyze_real_attention()
    except Exception as e:
        print(f"\nSkipping real attention analysis: {e}")


if __name__ == "__main__":
    main()
