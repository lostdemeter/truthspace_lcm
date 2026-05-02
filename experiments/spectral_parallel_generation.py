#!/usr/bin/env python3
"""
Spectral Parallel Generation Experiment

Hypothesis: Instead of generating tokens sequentially, we can generate them
in parallel using spectral resonance methods inspired by resfrac.

The approach:
1. Identify "boom positions" (spectral peaks) from the prompt
2. Solve boom positions in parallel using spectral windows
3. Interpolate non-boom positions

This is exploratory - we're testing whether the hypothesis has merit.

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949


@dataclass
class SpectralConfig:
    """Configuration for spectral generation."""
    sigma: float = 1.0  # Spectral window width
    n_candidates: int = 100  # Number of candidate sequences to evaluate
    boom_threshold: float = 0.5  # Entropy threshold for boom detection
    window_type: str = "gaussian"  # gaussian, sinc, triangle


def gaussian_window(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian spectral window."""
    return np.exp(-x**2 / (2 * sigma**2))


def sinc_window(x: np.ndarray, sigma: float) -> np.ndarray:
    """Sinc spectral window."""
    # Avoid division by zero
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    return np.sin(np.pi * x_safe / sigma) / (np.pi * x_safe / sigma)


def triangle_window(x: np.ndarray, sigma: float) -> np.ndarray:
    """Triangle (sinc²) spectral window."""
    s = sinc_window(x, sigma)
    return s ** 2


class SpectralGenerator:
    """
    Experimental spectral parallel token generator.
    
    Instead of sequential generation, attempts to solve multiple
    positions simultaneously using spectral resonance.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Get embedding matrix for spectral operations
        self.embeddings = self.model.model.embed_tokens.weight.data
        self.vocab_size = self.embeddings.shape[0]
        self.hidden_dim = self.embeddings.shape[1]
        
        print(f"Vocab size: {self.vocab_size}, Hidden dim: {self.hidden_dim}")
    
    def detect_boom_positions(self, prompt: str, n_future: int = 10) -> List[int]:
        """
        Predict boom positions for future tokens based on prompt.
        
        Uses the φ-biased heuristic: booms occur at semantic boundaries,
        approximately every φ² ≈ 2.618 tokens on average.
        """
        # For now, use a simple heuristic based on φ spacing
        # In a full implementation, this would use the boom detection from Doc 159
        
        boom_positions = []
        pos = 0
        while pos < n_future:
            boom_positions.append(pos)
            # Next boom at approximately φ² spacing
            pos += max(1, int(PHI ** 2))
        
        return boom_positions[:n_future]
    
    def get_semantic_trajectory(self, prompt: str, n_tokens: int = 10
                                ) -> torch.Tensor:
        """
        Compute the semantic trajectory for the output sequence.
        
        This is the "wide window" view - the general direction in embedding
        space that the output should follow.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # Last hidden state at last position
            last_hidden = outputs.hidden_states[-1][0, -1, :]
        
        # The trajectory is defined by the last hidden state
        # This encodes "where we're going" semantically
        return last_hidden
    
    def spectral_score(self, candidate_embedding: torch.Tensor,
                       target_trajectory: torch.Tensor,
                       config: SpectralConfig) -> float:
        """
        Score a candidate token using spectral window.
        
        The score measures how well the candidate "resonates" with
        the target semantic trajectory.
        """
        # Distance in embedding space
        distance = torch.norm(candidate_embedding - target_trajectory).item()
        
        # Apply spectral window
        if config.window_type == "gaussian":
            score = gaussian_window(np.array([distance]), config.sigma)[0]
        elif config.window_type == "sinc":
            score = sinc_window(np.array([distance]), config.sigma)[0]
        else:
            score = triangle_window(np.array([distance]), config.sigma)[0]
        
        return score
    
    def generate_parallel_candidates(self, prompt: str, n_tokens: int = 10,
                                     config: SpectralConfig = None
                                     ) -> Tuple[List[int], Dict]:
        """
        Generate tokens using spectral parallel approach.
        
        Instead of sequential generation, we:
        1. Compute the semantic trajectory
        2. Find tokens that resonate with the trajectory
        3. Select the highest-scoring sequence
        """
        if config is None:
            config = SpectralConfig()
        
        start_time = time.time()
        
        # Get semantic trajectory from prompt
        trajectory = self.get_semantic_trajectory(prompt, n_tokens)
        
        # Get top candidates based on trajectory similarity
        # This is the "spectral resonance" step
        with torch.no_grad():
            # Score all tokens by similarity to trajectory
            similarities = F.cosine_similarity(
                self.embeddings, 
                trajectory.unsqueeze(0),
                dim=1
            )
            
            # Get top candidates
            top_k = min(config.n_candidates, self.vocab_size)
            top_indices = torch.argsort(-similarities)[:top_k]
            top_scores = similarities[top_indices]
        
        # For each position, select token based on spectral score
        generated = []
        position_scores = []
        
        for pos in range(n_tokens):
            # Apply position-dependent weighting
            # Boom positions get higher weight
            boom_positions = self.detect_boom_positions(prompt, n_tokens)
            is_boom = pos in boom_positions
            
            if is_boom:
                # At boom positions, select highest-scoring token
                best_idx = top_indices[0].item()
            else:
                # At non-boom positions, interpolate
                # Use a softer selection based on position
                weight_idx = min(pos, len(top_indices) - 1)
                best_idx = top_indices[weight_idx].item()
            
            generated.append(best_idx)
            position_scores.append(similarities[best_idx].item())
        
        elapsed = time.time() - start_time
        
        info = {
            "time": elapsed,
            "tokens_per_second": n_tokens / elapsed,
            "boom_positions": boom_positions,
            "position_scores": position_scores,
        }
        
        return generated, info
    
    def generate_sequential(self, prompt: str, n_tokens: int = 10
                           ) -> Tuple[List[int], Dict]:
        """Standard sequential generation for comparison."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start_time
        
        generated = outputs[0, input_ids.shape[1]:].tolist()
        
        info = {
            "time": elapsed,
            "tokens_per_second": len(generated) / elapsed,
        }
        
        return generated, info


def compare_generation_methods():
    """Compare sequential vs spectral parallel generation."""
    print("=" * 60)
    print("Spectral Parallel Generation Experiment")
    print("=" * 60)
    
    gen = SpectralGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Generating {n_tokens} tokens...")
    
    # Sequential generation
    print("\n--- Sequential Generation ---")
    seq_tokens, seq_info = gen.generate_sequential(prompt, n_tokens)
    seq_text = gen.tokenizer.decode(seq_tokens)
    
    print(f"Output: {seq_text!r}")
    print(f"Time: {seq_info['time']:.2f}s")
    print(f"Speed: {seq_info['tokens_per_second']:.2f} tok/s")
    
    # Spectral parallel generation
    print("\n--- Spectral Parallel Generation ---")
    spec_tokens, spec_info = gen.generate_parallel_candidates(prompt, n_tokens)
    spec_text = gen.tokenizer.decode(spec_tokens)
    
    print(f"Output: {spec_text!r}")
    print(f"Time: {spec_info['time']:.2f}s")
    print(f"Speed: {spec_info['tokens_per_second']:.2f} tok/s")
    print(f"Boom positions: {spec_info['boom_positions']}")
    
    # Compare
    print("\n--- Comparison ---")
    print(f"Sequential: {seq_text!r}")
    print(f"Spectral:   {spec_text!r}")
    print(f"Speedup: {seq_info['time'] / spec_info['time']:.1f}x")
    
    # Check if any tokens match
    matches = sum(1 for a, b in zip(seq_tokens, spec_tokens) if a == b)
    print(f"Token matches: {matches}/{min(len(seq_tokens), len(spec_tokens))}")
    
    del gen.model


def explore_spectral_windows():
    """Explore different spectral window functions."""
    print("\n" + "=" * 60)
    print("Exploring Spectral Windows")
    print("=" * 60)
    
    gen = SpectralGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 5
    
    for window_type in ["gaussian", "sinc", "triangle"]:
        config = SpectralConfig(window_type=window_type, sigma=1.0)
        tokens, info = gen.generate_parallel_candidates(prompt, n_tokens, config)
        text = gen.tokenizer.decode(tokens)
        
        print(f"\n{window_type.upper()} window:")
        print(f"  Output: {text!r}")
        print(f"  Scores: {[f'{s:.3f}' for s in info['position_scores']]}")
    
    del gen.model


def analyze_boom_structure():
    """Analyze boom structure in actual model outputs."""
    print("\n" + "=" * 60)
    print("Analyzing Boom Structure")
    print("=" * 60)
    
    gen = SpectralGenerator()
    
    prompt = "The capital of France is"
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate with attention outputs
    with torch.no_grad():
        outputs = gen.model(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
    
    # Analyze attention entropy at each layer
    print("\nAttention entropy by layer (last token attending to all):")
    
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn shape: (batch, heads, seq, seq)
        # Get attention from last position
        last_attn = attn[0, :, -1, :]  # (heads, seq)
        
        # Compute entropy per head
        entropies = []
        for head in range(last_attn.shape[0]):
            probs = last_attn[head]
            probs = probs + 1e-10  # Avoid log(0)
            entropy = -torch.sum(probs * torch.log(probs)).item()
            entropies.append(entropy)
        
        mean_entropy = np.mean(entropies)
        
        if layer_idx % 7 == 0:
            print(f"  Layer {layer_idx:2d}: entropy = {mean_entropy:.3f}")
    
    # Find boom positions based on attention concentration
    print("\nBoom positions (high attention concentration):")
    
    # Use middle layer for analysis
    mid_layer = len(outputs.attentions) // 2
    attn = outputs.attentions[mid_layer][0]  # (heads, seq, seq)
    
    # Average over heads
    avg_attn = attn.mean(dim=0)  # (seq, seq)
    
    # For each position, how much attention does it receive?
    received_attn = avg_attn.sum(dim=0)  # (seq,)
    
    # Normalize
    received_attn = received_attn / received_attn.sum()
    
    # Find peaks
    threshold = 1.0 / len(received_attn) * PHI  # Above uniform by φ factor
    boom_mask = received_attn > threshold
    boom_positions = torch.where(boom_mask)[0].tolist()
    
    tokens = gen.tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Boom positions: {boom_positions}")
    print(f"  Boom tokens: {[tokens[i] for i in boom_positions if i < len(tokens)]}")
    print(f"  Boom ratio: {len(boom_positions) / len(tokens) * 100:.1f}%")
    
    del gen.model


def spectral_refinement_experiment():
    """
    Test spectral refinement approach.
    
    Instead of generating from scratch, we:
    1. Generate a draft using fast approximation
    2. Identify low-confidence positions
    3. Refine those positions using spectral methods
    """
    print("\n" + "=" * 60)
    print("Spectral Refinement Experiment")
    print("=" * 60)
    
    gen = SpectralGenerator()
    
    prompt = "The capital of France is"
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Step 1: Generate with logit analysis
    print("\n--- Step 1: Generate with confidence analysis ---")
    
    generated_tokens = []
    confidences = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = gen.model(current_ids)
            logits = outputs.logits[0, -1, :]
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top token and its probability
            top_prob, top_idx = probs.max(dim=-1)
            
            generated_tokens.append(top_idx.item())
            confidences.append(top_prob.item())
            
            # Append to sequence
            current_ids = torch.cat([
                current_ids, 
                top_idx.unsqueeze(0).unsqueeze(0)
            ], dim=1)
    
    text = gen.tokenizer.decode(generated_tokens)
    print(f"Generated: {text!r}")
    print(f"Confidences: {[f'{c:.3f}' for c in confidences]}")
    
    # Step 2: Identify low-confidence positions (potential boom positions)
    print("\n--- Step 2: Identify low-confidence positions ---")
    
    mean_conf = np.mean(confidences)
    low_conf_positions = [i for i, c in enumerate(confidences) if c < mean_conf]
    
    print(f"Mean confidence: {mean_conf:.3f}")
    print(f"Low-confidence positions: {low_conf_positions}")
    print(f"Low-confidence tokens: {[gen.tokenizer.decode([generated_tokens[i]]) for i in low_conf_positions]}")
    
    # Step 3: Analyze what the model "wanted" at each position
    print("\n--- Step 3: Top alternatives at each position ---")
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(min(5, n_tokens)):
            outputs = gen.model(current_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            top5_probs, top5_idx = probs.topk(5)
            
            print(f"\nPosition {i}:")
            for j, (prob, idx) in enumerate(zip(top5_probs, top5_idx)):
                token = gen.tokenizer.decode([idx.item()])
                marker = " ← SELECTED" if j == 0 else ""
                print(f"  {j+1}. {token!r}: {prob.item():.3f}{marker}")
            
            # Append selected token
            current_ids = torch.cat([
                current_ids,
                top5_idx[0].unsqueeze(0).unsqueeze(0)
            ], dim=1)
    
    # Step 4: Spectral analysis of token embeddings
    print("\n--- Step 4: Spectral structure of output ---")
    
    # Get embeddings of generated tokens
    token_embeddings = gen.embeddings[generated_tokens]
    
    # Compute pairwise similarities
    similarities = F.cosine_similarity(
        token_embeddings.unsqueeze(0),
        token_embeddings.unsqueeze(1),
        dim=2
    )
    
    print("Token similarity matrix (diagonal = 1.0):")
    print("     " + "  ".join([f"{i:5d}" for i in range(min(5, n_tokens))]))
    for i in range(min(5, n_tokens)):
        row = [f"{similarities[i, j].item():5.2f}" for j in range(min(5, n_tokens))]
        print(f"  {i}: " + "  ".join(row))
    
    # Compute spectral decomposition
    U, S, V = torch.svd(token_embeddings)
    
    print(f"\nSingular values (top 5): {S[:5].tolist()}")
    print(f"Effective rank (90% variance): {(S.cumsum(0) / S.sum() < 0.9).sum().item() + 1}")
    
    del gen.model


def parallel_boom_generation():
    """
    Test parallel generation at boom positions only.
    
    Hypothesis: If we can identify boom positions, we can generate
    those in parallel and interpolate the rest.
    """
    print("\n" + "=" * 60)
    print("Parallel Boom Generation")
    print("=" * 60)
    
    gen = SpectralGenerator()
    
    prompt = "The capital of France is"
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # First, generate normally to get reference
    print("\n--- Reference generation ---")
    with torch.no_grad():
        ref_output = gen.model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=gen.tokenizer.eos_token_id,
        )
    
    ref_tokens = ref_output[0, input_ids.shape[1]:].tolist()
    ref_text = gen.tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    # Analyze which positions are "boom" (high information content)
    print("\n--- Boom analysis ---")
    
    # Generate with entropy tracking
    entropies = []
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = gen.model(current_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Entropy of distribution
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)
            
            # Get next token
            next_token = logits.argmax().item()
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]])
            ], dim=1)
    
    # Boom positions have LOW entropy (model is confident)
    mean_entropy = np.mean(entropies)
    boom_positions = [i for i, e in enumerate(entropies) if e < mean_entropy]
    non_boom_positions = [i for i, e in enumerate(entropies) if e >= mean_entropy]
    
    print(f"Entropies: {[f'{e:.2f}' for e in entropies]}")
    print(f"Mean entropy: {mean_entropy:.2f}")
    print(f"Boom positions (low entropy): {boom_positions}")
    print(f"Non-boom positions (high entropy): {non_boom_positions}")
    
    # Show tokens at each position type
    print(f"\nBoom tokens: {[gen.tokenizer.decode([ref_tokens[i]]) for i in boom_positions if i < len(ref_tokens)]}")
    print(f"Non-boom tokens: {[gen.tokenizer.decode([ref_tokens[i]]) for i in non_boom_positions if i < len(ref_tokens)]}")
    
    # Key insight: boom positions are where the model is CERTAIN
    # These are the "spectral peaks" - the rest can be interpolated
    
    print("\n--- Boom coverage analysis ---")
    boom_count = len(boom_positions)
    total_count = n_tokens
    print(f"Boom ratio: {boom_count}/{total_count} = {boom_count/total_count*100:.1f}%")
    
    # If we could generate boom positions in parallel, we'd only need
    # to do sequential generation for the non-boom positions
    potential_speedup = total_count / (boom_count + 1)  # +1 for parallel boom pass
    print(f"Potential speedup (if booms parallel): {potential_speedup:.1f}x")
    
    del gen.model


if __name__ == "__main__":
    compare_generation_methods()
    # explore_spectral_windows()  # Skip - same output for all windows
    # analyze_boom_structure()  # Skip - needs eager attention
    spectral_refinement_experiment()
    parallel_boom_generation()
