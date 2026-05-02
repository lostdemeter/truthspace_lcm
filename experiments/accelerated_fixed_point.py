#!/usr/bin/env python3
"""
Accelerated Fixed-Point Methods for Token Generation
=====================================================

Building on the discovery that autoregression is a fixed-point problem,
this implements acceleration methods to reduce iterations from 11 to fewer.

Methods:
1. Anderson Acceleration (like DIIS in quantum chemistry)
2. Multi-token predictor from hidden state
3. Principal component projection (exploit rank-2 structure)

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949


class AndersonAccelerator:
    """
    Anderson acceleration for fixed-point iteration.
    
    Standard fixed-point: x_{k+1} = g(x_k)
    Anderson: x_{k+1} = (1-β) * Σ α_i x_{k-i} + β * Σ α_i g(x_{k-i})
    
    where α_i are chosen to minimize ||Σ α_i (g(x_{k-i}) - x_{k-i})||
    
    This is like DIIS (Direct Inversion in the Iterative Subspace) used
    in quantum chemistry for SCF convergence.
    """
    
    def __init__(self, m: int = 5, beta: float = 1.0):
        """
        Args:
            m: Number of previous iterates to use
            beta: Mixing parameter (1.0 = pure Anderson)
        """
        self.m = m
        self.beta = beta
        self.x_history = []
        self.g_history = []
        self.residual_history = []
    
    def reset(self):
        self.x_history = []
        self.g_history = []
        self.residual_history = []
    
    def step(self, x: np.ndarray, g_x: np.ndarray) -> np.ndarray:
        """
        Perform one Anderson acceleration step.
        
        Args:
            x: Current iterate
            g_x: g(x), the fixed-point function applied to x
        
        Returns:
            Next iterate
        """
        residual = g_x - x
        
        self.x_history.append(x.copy())
        self.g_history.append(g_x.copy())
        self.residual_history.append(residual.copy())
        
        # Keep only last m iterates
        if len(self.x_history) > self.m:
            self.x_history.pop(0)
            self.g_history.pop(0)
            self.residual_history.pop(0)
        
        m_k = len(self.residual_history)
        
        if m_k == 1:
            # First iteration: just use g(x)
            return g_x
        
        # Build residual matrix
        R = np.column_stack(self.residual_history)  # [n, m_k]
        
        # Solve least squares: min ||R @ α||^2 s.t. sum(α) = 1
        # This is equivalent to solving (R.T @ R) @ α = 0 with constraint
        
        # Use QR decomposition for stability
        try:
            # Solve: R @ α = 0, sum(α) = 1
            # Augmented system: [R; 1 1 ... 1] @ α = [0; 1]
            
            ones = np.ones((1, m_k))
            A = np.vstack([R, ones])
            b = np.zeros(R.shape[0] + 1)
            b[-1] = 1.0
            
            alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Compute accelerated iterate
            X = np.column_stack(self.x_history)
            G = np.column_stack(self.g_history)
            
            x_new = (1 - self.beta) * (X @ alpha) + self.beta * (G @ alpha)
            
            return x_new
            
        except np.linalg.LinAlgError:
            # Fall back to standard iteration
            return g_x


class MultiTokenPredictor(nn.Module):
    """
    Predict multiple future tokens from a single hidden state.
    
    The key insight: The hidden state contains information about ALL
    future tokens, not just the next one. We can train a small network
    to extract this information.
    
    Architecture:
        hidden_state [3584] -> Linear -> k token logits [k, vocab_size]
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int, k: int = 10):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # One head per future position
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(k)
        ])
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_dim] or [hidden_dim]
        
        Returns:
            logits: [batch, k, vocab_size] or [k, vocab_size]
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        logits = torch.stack([head(hidden_state) for head in self.heads], dim=1)
        
        if squeeze:
            logits = logits.squeeze(0)
        
        return logits


class AcceleratedGenerator:
    """
    Token generation using accelerated fixed-point methods.
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
        
        self.vocab_size = self.model.config.vocab_size
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"Vocab size: {self.vocab_size}, Hidden dim: {self.hidden_dim}")
    
    def generate_sequential(self, prompt: str, n_tokens: int) -> Tuple[List[int], Dict]:
        """Baseline sequential generation."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        total_time = time.time() - start_time
        
        tokens = outputs[0, prompt_len:].tolist()
        
        return tokens, {"time": total_time}
    
    def generate_fixed_point(self, prompt: str, n_tokens: int, 
                            max_iters: int = 20) -> Tuple[List[int], Dict]:
        """Standard fixed-point iteration."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Random initialization
        import random
        random.seed(42)
        current = [random.randint(0, 1000) for _ in range(n_tokens)]
        
        start_time = time.time()
        
        for iteration in range(max_iters):
            full_ids = torch.cat([
                input_ids,
                torch.tensor([current])
            ], dim=1)
            
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits[0]
            
            new_current = []
            for i in range(n_tokens):
                if i == 0:
                    pos_logits = logits[prompt_len - 1]
                else:
                    pos_logits = logits[prompt_len + i - 1]
                new_current.append(pos_logits.argmax().item())
            
            if new_current == current:
                break
            current = new_current
        
        total_time = time.time() - start_time
        
        return current, {"time": total_time, "iterations": iteration + 1}
    
    def generate_anderson(self, prompt: str, n_tokens: int,
                         max_iters: int = 20, m: int = 5) -> Tuple[List[int], Dict]:
        """
        Fixed-point iteration with Anderson acceleration.
        
        We work in embedding space for smooth optimization,
        then project back to discrete tokens.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Get embedding matrix
        embeddings = self.model.model.embed_tokens.weight.data  # [vocab, hidden]
        
        # Initialize with random embeddings
        import random
        random.seed(42)
        
        # Start with random token embeddings
        current_tokens = [random.randint(0, 1000) for _ in range(n_tokens)]
        current_emb = embeddings[current_tokens].numpy()  # [n_tokens, hidden]
        current_flat = current_emb.flatten()
        
        accelerator = AndersonAccelerator(m=m)
        
        start_time = time.time()
        
        for iteration in range(max_iters):
            # Reshape to tokens
            current_emb = current_flat.reshape(n_tokens, -1)
            
            # Find nearest tokens
            current_tokens = []
            for i in range(n_tokens):
                emb = torch.tensor(current_emb[i])
                sims = F.cosine_similarity(emb.unsqueeze(0), embeddings)
                current_tokens.append(sims.argmax().item())
            
            # Run model
            full_ids = torch.cat([
                input_ids,
                torch.tensor([current_tokens])
            ], dim=1)
            
            with torch.no_grad():
                outputs = self.model(full_ids, output_hidden_states=True)
                logits = outputs.logits[0]
                hidden_states = outputs.hidden_states[-1][0]  # Last layer
            
            # Get new tokens and their embeddings
            new_tokens = []
            new_emb = []
            
            for i in range(n_tokens):
                if i == 0:
                    pos_logits = logits[prompt_len - 1]
                else:
                    pos_logits = logits[prompt_len + i - 1]
                
                new_token = pos_logits.argmax().item()
                new_tokens.append(new_token)
                new_emb.append(embeddings[new_token].numpy())
            
            new_emb = np.stack(new_emb)
            new_flat = new_emb.flatten()
            
            # Anderson acceleration step
            accelerated_flat = accelerator.step(current_flat, new_flat)
            
            # Check convergence (in token space)
            if new_tokens == current_tokens:
                break
            
            current_flat = accelerated_flat
        
        total_time = time.time() - start_time
        
        # Final projection to tokens
        final_emb = current_flat.reshape(n_tokens, -1)
        final_tokens = []
        for i in range(n_tokens):
            emb = torch.tensor(final_emb[i])
            sims = F.cosine_similarity(emb.unsqueeze(0), embeddings)
            final_tokens.append(sims.argmax().item())
        
        return final_tokens, {"time": total_time, "iterations": iteration + 1}
    
    def generate_multi_token(self, prompt: str, n_tokens: int) -> Tuple[List[int], Dict]:
        """
        Generate using multi-token prediction from hidden state.
        
        Strategy:
        1. Get hidden state from prompt
        2. Use lm_head to predict first token
        3. Use hidden state trajectory to predict remaining tokens
        4. Verify with single forward pass
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        start_time = time.time()
        
        # Step 1: Get prompt hidden state
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]  # [hidden_dim]
            first_logits = outputs.logits[0, -1, :]
        
        # Step 2: Predict first token
        first_token = first_logits.argmax().item()
        
        # Step 3: Use the model's own structure to predict trajectory
        # Key insight: The hidden state evolves smoothly
        # We can approximate future hidden states
        
        # Get the "direction" by generating one more token
        with torch.no_grad():
            extended_ids = torch.cat([
                input_ids,
                torch.tensor([[first_token]])
            ], dim=1)
            outputs2 = self.model(extended_ids, output_hidden_states=True)
            second_hidden = outputs2.hidden_states[-1][0, -1, :]
        
        # Compute delta
        delta = second_hidden - prompt_hidden
        
        # Extrapolate hidden states
        predicted_tokens = [first_token]
        current_hidden = second_hidden
        
        lm_head = self.model.lm_head
        
        for i in range(1, n_tokens):
            # Extrapolate
            next_hidden = current_hidden + delta * 0.5  # Damped extrapolation
            
            # Predict token
            with torch.no_grad():
                logits = lm_head(next_hidden)
                token = logits.argmax().item()
            
            predicted_tokens.append(token)
            current_hidden = next_hidden
        
        # Step 4: Verify and fix with single forward pass
        full_ids = torch.cat([
            input_ids,
            torch.tensor([predicted_tokens])
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0]
        
        verified_tokens = []
        for i in range(n_tokens):
            if i == 0:
                pos_logits = logits[prompt_len - 1]
            else:
                pos_logits = logits[prompt_len + i - 1]
            verified_tokens.append(pos_logits.argmax().item())
        
        total_time = time.time() - start_time
        
        return verified_tokens, {
            "time": total_time,
            "predicted_matches": sum(1 for a, b in zip(predicted_tokens, verified_tokens) if a == b)
        }
    
    def generate_principal_first(self, prompt: str, n_tokens: int,
                                 max_iters: int = 10) -> Tuple[List[int], Dict]:
        """
        Generate by fixing principal (high-entropy) positions first.
        
        Strategy:
        1. Identify principal positions (highest entropy)
        2. Generate principal tokens sequentially
        3. Fixed-point iteration for remaining tokens
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        start_time = time.time()
        
        # Step 1: Generate sequentially to measure entropy
        entropies = []
        seq_tokens = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                entropies.append(entropy)
                
                token = logits.argmax().item()
                seq_tokens.append(token)
                
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[token]])
                ], dim=1)
        
        # Step 2: Identify principal positions (top 2 by entropy)
        sorted_by_entropy = sorted(range(n_tokens), key=lambda i: entropies[i], reverse=True)
        n_principal = 2
        principal_positions = set(sorted_by_entropy[:n_principal])
        
        # Step 3: Initialize with principal tokens correct, rest random
        import random
        random.seed(42)
        
        current = []
        for i in range(n_tokens):
            if i in principal_positions:
                current.append(seq_tokens[i])
            else:
                current.append(random.randint(0, 1000))
        
        # Step 4: Fixed-point iteration for non-principal positions
        for iteration in range(max_iters):
            full_ids = torch.cat([
                input_ids,
                torch.tensor([current])
            ], dim=1)
            
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits[0]
            
            new_current = current.copy()
            changes = 0
            
            for i in range(n_tokens):
                if i in principal_positions:
                    continue  # Don't change principal positions
                
                if i == 0:
                    pos_logits = logits[prompt_len - 1]
                else:
                    pos_logits = logits[prompt_len + i - 1]
                
                best_token = pos_logits.argmax().item()
                
                if best_token != current[i]:
                    new_current[i] = best_token
                    changes += 1
            
            current = new_current
            
            if changes == 0:
                break
        
        total_time = time.time() - start_time
        
        return current, {
            "time": total_time,
            "iterations": iteration + 1,
            "principal_positions": list(principal_positions),
            "entropies": entropies
        }


def test_all_methods():
    """Compare all acceleration methods."""
    print("\n" + "=" * 70)
    print("Comparing Acceleration Methods")
    print("=" * 70)
    
    gen = AcceleratedGenerator()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox jumps over",
    ]
    
    n_tokens = 10
    
    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt!r}")
        print("=" * 70)
        
        # Reference
        ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
        ref_text = gen.tokenizer.decode(ref_tokens)
        print(f"\nReference: {ref_text!r}")
        print(f"Reference time: {ref_info['time']:.2f}s")
        
        # Method 1: Standard fixed-point
        print("\n--- Standard Fixed-Point ---")
        fp_tokens, fp_info = gen.generate_fixed_point(prompt, n_tokens)
        fp_text = gen.tokenizer.decode(fp_tokens)
        fp_matches = sum(1 for a, b in zip(ref_tokens, fp_tokens) if a == b)
        print(f"Result: {fp_text!r}")
        print(f"Matches: {fp_matches}/{n_tokens}, Iters: {fp_info['iterations']}, Time: {fp_info['time']:.2f}s")
        
        # Method 2: Anderson acceleration
        print("\n--- Anderson Acceleration ---")
        aa_tokens, aa_info = gen.generate_anderson(prompt, n_tokens)
        aa_text = gen.tokenizer.decode(aa_tokens)
        aa_matches = sum(1 for a, b in zip(ref_tokens, aa_tokens) if a == b)
        print(f"Result: {aa_text!r}")
        print(f"Matches: {aa_matches}/{n_tokens}, Iters: {aa_info['iterations']}, Time: {aa_info['time']:.2f}s")
        
        # Method 3: Multi-token prediction
        print("\n--- Multi-Token Prediction ---")
        mt_tokens, mt_info = gen.generate_multi_token(prompt, n_tokens)
        mt_text = gen.tokenizer.decode(mt_tokens)
        mt_matches = sum(1 for a, b in zip(ref_tokens, mt_tokens) if a == b)
        print(f"Result: {mt_text!r}")
        print(f"Matches: {mt_matches}/{n_tokens}, Predicted matches: {mt_info['predicted_matches']}, Time: {mt_info['time']:.2f}s")
        
        # Method 4: Principal-first
        print("\n--- Principal-First ---")
        pf_tokens, pf_info = gen.generate_principal_first(prompt, n_tokens)
        pf_text = gen.tokenizer.decode(pf_tokens)
        pf_matches = sum(1 for a, b in zip(ref_tokens, pf_tokens) if a == b)
        print(f"Result: {pf_text!r}")
        print(f"Matches: {pf_matches}/{n_tokens}, Iters: {pf_info['iterations']}, Time: {pf_info['time']:.2f}s")
        print(f"Principal positions: {pf_info['principal_positions']}")
    
    del gen.model


def test_hidden_state_multi_token():
    """
    Test whether we can extract multiple tokens from a single hidden state.
    
    This is the key question: Does the hidden state contain enough information
    to predict tokens 2, 3, 4, ... without sequential generation?
    """
    print("\n" + "=" * 70)
    print("Hidden State Multi-Token Extraction")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    print(f"\nPrompt: {prompt!r}")
    
    # Generate reference with hidden states
    print("\n--- Collecting Hidden State Trajectory ---")
    
    hidden_trajectory = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]  # Last layer, last position
            hidden_trajectory.append(hidden.clone())
            
            logits = outputs.logits[0, -1, :]
            token = logits.argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]])
            ], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    # Now test: Can we predict token i from hidden state 0?
    print("\n--- Predicting Future Tokens from Initial Hidden State ---")
    
    lm_head = model.lm_head
    initial_hidden = hidden_trajectory[0]
    
    # Method 1: Direct prediction (just apply lm_head to initial hidden)
    print("\nMethod 1: Direct lm_head on initial hidden")
    with torch.no_grad():
        direct_logits = lm_head(initial_hidden)
        direct_token = direct_logits.argmax().item()
    
    print(f"  Token 0: pred={tokenizer.decode([direct_token])!r}, ref={tokenizer.decode([ref_tokens[0]])!r}, match={direct_token == ref_tokens[0]}")
    
    # Method 2: Linear extrapolation of hidden states
    print("\nMethod 2: Linear extrapolation")
    
    # Compute average delta from trajectory
    deltas = [hidden_trajectory[i+1] - hidden_trajectory[i] for i in range(len(hidden_trajectory)-1)]
    avg_delta = torch.stack(deltas).mean(dim=0)
    
    print(f"  Average delta norm: {torch.norm(avg_delta).item():.2f}")
    
    extrapolated_tokens = []
    current_h = initial_hidden.clone()
    
    with torch.no_grad():
        for i in range(n_tokens):
            logits = lm_head(current_h)
            token = logits.argmax().item()
            extrapolated_tokens.append(token)
            current_h = current_h + avg_delta
    
    print("  Extrapolated predictions:")
    for i, (pred, ref) in enumerate(zip(extrapolated_tokens, ref_tokens)):
        match = "✓" if pred == ref else "✗"
        print(f"    {i}: pred={tokenizer.decode([pred])!r}, ref={tokenizer.decode([ref])!r} {match}")
    
    ext_matches = sum(1 for a, b in zip(extrapolated_tokens, ref_tokens) if a == b)
    print(f"  Matches: {ext_matches}/{n_tokens}")
    
    # Method 3: Use actual hidden states (oracle)
    print("\nMethod 3: Oracle (actual hidden states)")
    
    oracle_tokens = []
    with torch.no_grad():
        for h in hidden_trajectory:
            logits = lm_head(h)
            token = logits.argmax().item()
            oracle_tokens.append(token)
    
    oracle_matches = sum(1 for a, b in zip(oracle_tokens, ref_tokens) if a == b)
    print(f"  Matches: {oracle_matches}/{n_tokens}")
    
    # Key insight: If oracle matches 100%, then the hidden state trajectory
    # contains all the information we need. The question is how to predict it.
    
    # Method 4: Polynomial fit to hidden state trajectory
    print("\nMethod 4: Polynomial fit to trajectory")
    
    # Fit a polynomial to each dimension of the hidden state
    H = torch.stack(hidden_trajectory).numpy()  # [n_tokens, hidden_dim]
    t = np.arange(n_tokens)
    
    # Fit degree-2 polynomial to first few points, extrapolate rest
    n_fit = 3  # Use first 3 points to fit
    
    poly_hidden = []
    for i in range(n_tokens):
        if i < n_fit:
            poly_hidden.append(hidden_trajectory[i])
        else:
            # Extrapolate using polynomial
            h_pred = np.zeros(H.shape[1])
            for d in range(H.shape[1]):
                coeffs = np.polyfit(t[:n_fit], H[:n_fit, d], deg=2)
                h_pred[d] = np.polyval(coeffs, i)
            poly_hidden.append(torch.tensor(h_pred, dtype=torch.float32))
    
    poly_tokens = []
    with torch.no_grad():
        for h in poly_hidden:
            logits = lm_head(h)
            token = logits.argmax().item()
            poly_tokens.append(token)
    
    print("  Polynomial predictions (fit on first 3):")
    for i, (pred, ref) in enumerate(zip(poly_tokens, ref_tokens)):
        match = "✓" if pred == ref else "✗"
        fit_marker = "(fit)" if i < n_fit else "(pred)"
        print(f"    {i} {fit_marker}: pred={tokenizer.decode([pred])!r}, ref={tokenizer.decode([ref])!r} {match}")
    
    poly_matches = sum(1 for a, b in zip(poly_tokens, ref_tokens) if a == b)
    print(f"  Matches: {poly_matches}/{n_tokens}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Oracle (actual hidden states): {oracle_matches}/{n_tokens}")
    print(f"Linear extrapolation: {ext_matches}/{n_tokens}")
    print(f"Polynomial fit (3 points): {poly_matches}/{n_tokens}")
    
    del model


if __name__ == "__main__":
    # Test hidden state multi-token extraction first
    test_hidden_state_multi_token()
    
    # Then compare all methods
    test_all_methods()
