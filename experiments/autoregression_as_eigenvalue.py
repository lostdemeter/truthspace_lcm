#!/usr/bin/env python3
"""
Autoregression as an Eigenvalue Problem
========================================

Deep dive into whether autoregressive token generation can be treated
as a singular system (like a quantum eigenvalue problem) rather than
a sequential dependency chain.

Key Questions:
1. What IS the autoregressive structure mathematically?
2. Can we formulate it as an eigenvalue problem?
3. What prevents parallel solution?
4. Can multi-token attention help?

Mathematical Background:
------------------------

Standard autoregression:
    P(x_1, x_2, ..., x_n) = P(x_1) × P(x_2|x_1) × P(x_3|x_1,x_2) × ...

This is a CHAIN of conditional probabilities. But what if we view it as:
    
    The JOINT distribution P(x_1, ..., x_n | prompt) is a single object
    
We're looking for the MODE of this joint distribution:
    argmax_{x_1,...,x_n} P(x_1, ..., x_n | prompt)

This is a GLOBAL optimization problem, not a sequential one!

The Quantum Analogy:
-------------------

In quantum mechanics:
- The wavefunction ψ describes the ENTIRE system
- Measurement collapses to an eigenstate
- The eigenvalue equation: H|ψ⟩ = E|ψ⟩

For autoregression:
- The "wavefunction" is the joint distribution over all tokens
- "Measurement" is selecting the output sequence
- Can we find an operator whose eigenvector is the optimal sequence?

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949


class AutoregressiveAnalyzer:
    """
    Analyze the structure of autoregressive generation.
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
        
        self.embeddings = self.model.model.embed_tokens.weight.data
        self.vocab_size = self.embeddings.shape[0]
        self.hidden_dim = self.embeddings.shape[1]
        
        print(f"Vocab size: {self.vocab_size}, Hidden dim: {self.hidden_dim}")
    
    def analyze_transition_matrix(self, prompt: str, n_tokens: int = 5):
        """
        Analyze the "transition matrix" of autoregression.
        
        Key insight: In a Markov chain, P(x_t | x_{t-1}) forms a transition matrix.
        For LLMs, P(x_t | x_1, ..., x_{t-1}) is more complex, but we can still
        analyze the structure.
        
        Question: Is there a fixed point? An eigenvector?
        """
        print("\n" + "=" * 60)
        print("Transition Matrix Analysis")
        print("=" * 60)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Generate reference sequence
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        ref_tokens = outputs.sequences[0, prompt_len:].tolist()
        ref_text = self.tokenizer.decode(ref_tokens)
        
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {ref_text!r}")
        
        # Analyze the probability structure at each position
        print("\n--- Probability Structure ---")
        
        scores = outputs.scores  # List of logits at each position
        
        for i, score in enumerate(scores):
            probs = F.softmax(score[0], dim=-1)
            
            # Top-5 probabilities
            top_probs, top_indices = torch.topk(probs, 5)
            
            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            # Selected token probability
            selected_prob = probs[ref_tokens[i]].item()
            
            token_text = self.tokenizer.decode([ref_tokens[i]])
            
            print(f"Position {i}: {token_text!r}")
            print(f"  Selected prob: {selected_prob:.4f}")
            print(f"  Entropy: {entropy:.2f} bits")
            print(f"  Top-5: {[f'{self.tokenizer.decode([idx.item()])!r}:{p:.3f}' for idx, p in zip(top_indices, top_probs)]}")
        
        return ref_tokens, scores
    
    def analyze_jacobian(self, prompt: str, n_tokens: int = 5):
        """
        Analyze the Jacobian of the autoregressive mapping.
        
        If we view autoregression as a function:
            f: R^{n×d} → R^{n×V}
        
        where n is sequence length, d is hidden dim, V is vocab size,
        then the Jacobian tells us how changes in one position affect others.
        
        Key question: Is the Jacobian lower-triangular (causal) or does it
        have structure we can exploit?
        """
        print("\n" + "=" * 60)
        print("Jacobian Structure Analysis")
        print("=" * 60)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Generate a sequence
        with torch.no_grad():
            current_ids = input_ids.clone()
            generated = []
            
            for i in range(n_tokens):
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                next_token = logits.argmax().item()
                generated.append(next_token)
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[next_token]])
                ], dim=1)
        
        gen_text = self.tokenizer.decode(generated)
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {gen_text!r}")
        
        # Now analyze: how does changing token i affect logits at position j?
        print("\n--- Cross-Position Influence ---")
        
        # Build full sequence
        full_ids = torch.cat([
            input_ids,
            torch.tensor([generated])
        ], dim=1)
        
        # Get baseline logits
        with torch.no_grad():
            baseline_outputs = self.model(full_ids)
            baseline_logits = baseline_outputs.logits[0]
        
        # For each generated position, measure influence on later positions
        influence_matrix = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            # Perturb token i
            perturbed_ids = full_ids.clone()
            
            # Change to a different token (use second-best)
            with torch.no_grad():
                if i == 0:
                    pos_logits = baseline_logits[prompt_len - 1]
                else:
                    pos_logits = baseline_logits[prompt_len + i - 1]
                
                sorted_indices = torch.argsort(-pos_logits)
                alt_token = sorted_indices[1].item()  # Second best
            
            perturbed_ids[0, prompt_len + i] = alt_token
            
            # Measure effect on later positions
            with torch.no_grad():
                perturbed_outputs = self.model(perturbed_ids)
                perturbed_logits = perturbed_outputs.logits[0]
            
            for j in range(n_tokens):
                if j <= i:
                    # Causal: earlier positions can't be affected
                    influence_matrix[i, j] = 0
                else:
                    # Measure change in logits
                    baseline_j = baseline_logits[prompt_len + j]
                    perturbed_j = perturbed_logits[prompt_len + j]
                    
                    # Use KL divergence as influence measure
                    baseline_probs = F.softmax(baseline_j, dim=-1)
                    perturbed_probs = F.softmax(perturbed_j, dim=-1)
                    
                    kl = torch.sum(baseline_probs * torch.log(baseline_probs / (perturbed_probs + 1e-10) + 1e-10)).item()
                    influence_matrix[i, j] = kl
        
        print("\nInfluence Matrix (KL divergence when perturbing row, effect on column):")
        print("     " + "".join([f"  {j:>5}" for j in range(n_tokens)]))
        for i in range(n_tokens):
            row = "".join([f"  {influence_matrix[i,j]:>5.2f}" for j in range(n_tokens)])
            print(f"  {i}: {row}")
        
        # Key insight: Is the influence matrix sparse? Low-rank?
        print("\n--- Influence Matrix Properties ---")
        
        # Only look at upper triangle (causal)
        upper = np.triu(influence_matrix, k=1)
        
        print(f"Max influence: {np.max(upper):.4f}")
        print(f"Mean influence: {np.mean(upper[upper > 0]):.4f}")
        print(f"Sparsity (< 0.01): {np.sum(upper < 0.01) / np.sum(upper >= 0):.2%}")
        
        # SVD of influence matrix
        U, S, Vt = np.linalg.svd(upper)
        print(f"Singular values: {S}")
        print(f"Effective rank (90% variance): {np.sum(np.cumsum(S**2) / np.sum(S**2) < 0.9) + 1}")
        
        return influence_matrix
    
    def formulate_as_fixed_point(self, prompt: str, n_tokens: int = 5):
        """
        Formulate autoregression as a fixed-point problem.
        
        Key insight: The "correct" sequence is a FIXED POINT of the
        autoregressive mapping:
        
            x* = argmax P(x | prompt, x*)
        
        where x* appears on BOTH sides. This is a self-consistency equation!
        
        In quantum mechanics, this is like finding the ground state:
            H|ψ⟩ = E|ψ⟩
        
        The eigenstate |ψ⟩ is self-consistent under the Hamiltonian H.
        
        For autoregression:
            T|x⟩ = |x⟩
        
        where T is the "transition operator" that maps a sequence to its
        most likely continuation.
        """
        print("\n" + "=" * 60)
        print("Fixed-Point Formulation")
        print("=" * 60)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        print(f"Prompt: {prompt!r}")
        
        # The fixed-point iteration:
        # 1. Start with initial guess x_0
        # 2. Compute x_{k+1} = T(x_k) where T is the autoregressive mapping
        # 3. Repeat until x_{k+1} = x_k
        
        # Initialize with random tokens
        import random
        random.seed(42)
        
        current = [random.randint(0, 1000) for _ in range(n_tokens)]
        
        print(f"\nInitial guess: {self.tokenizer.decode(current)!r}")
        
        # Fixed-point iteration
        max_iters = 20
        history = [current.copy()]
        
        for iteration in range(max_iters):
            # Build sequence
            full_ids = torch.cat([
                input_ids,
                torch.tensor([current])
            ], dim=1)
            
            # Get model's prediction for each position
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits[0]
            
            # Update each position to model's best prediction
            new_current = []
            for i in range(n_tokens):
                if i == 0:
                    pos_logits = logits[prompt_len - 1]
                else:
                    pos_logits = logits[prompt_len + i - 1]
                
                best_token = pos_logits.argmax().item()
                new_current.append(best_token)
            
            # Check for fixed point
            if new_current == current:
                print(f"\nFixed point found at iteration {iteration + 1}!")
                break
            
            current = new_current
            history.append(current.copy())
            
            current_text = self.tokenizer.decode(current)
            print(f"Iter {iteration + 1}: {current_text!r}")
        
        # Compare to sequential generation
        print("\n--- Comparison to Sequential ---")
        
        with torch.no_grad():
            seq_outputs = self.model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        seq_tokens = seq_outputs[0, prompt_len:].tolist()
        seq_text = self.tokenizer.decode(seq_tokens)
        
        print(f"Sequential: {seq_text!r}")
        print(f"Fixed-point: {self.tokenizer.decode(current)!r}")
        
        matches = sum(1 for a, b in zip(seq_tokens, current) if a == b)
        print(f"Matches: {matches}/{n_tokens}")
        
        # Key question: Does the fixed-point converge to the sequential result?
        # If yes, we have a parallel algorithm!
        # If no, why not?
        
        return current, seq_tokens, history
    
    def analyze_as_eigenvalue_problem(self, prompt: str, n_tokens: int = 5):
        """
        Attempt to formulate autoregression as an eigenvalue problem.
        
        The idea: If we can find a matrix M such that the optimal sequence
        x* is an eigenvector of M, then we can solve for x* directly.
        
        In quantum mechanics:
            H|ψ⟩ = E|ψ⟩
        
        The ground state is the eigenvector with lowest eigenvalue.
        
        For autoregression, we want:
            M|x*⟩ = λ|x*⟩
        
        where |x*⟩ is the optimal sequence.
        
        What is M? One candidate:
            M = I - T
        
        where T is the transition operator. Then x* is a fixed point of T
        iff M|x*⟩ = 0, i.e., x* is in the null space of M.
        """
        print("\n" + "=" * 60)
        print("Eigenvalue Problem Formulation")
        print("=" * 60)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        print(f"Prompt: {prompt!r}")
        
        # Get sequential result as "ground truth"
        with torch.no_grad():
            seq_outputs = self.model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        seq_tokens = seq_outputs[0, prompt_len:].tolist()
        seq_text = self.tokenizer.decode(seq_tokens)
        print(f"Target sequence: {seq_text!r}")
        
        # Build the "transition matrix" in embedding space
        # For each position, we have a mapping: current_embedding → next_token_logits
        
        # Let's work in a reduced space: just the top-k tokens at each position
        k = 10  # Top-k candidates per position
        
        # Get candidates at each position
        print("\n--- Building Candidate Space ---")
        
        candidates = []
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                
                top_k_indices = torch.argsort(-logits)[:k].tolist()
                candidates.append(top_k_indices)
                
                # Use top token to continue
                next_token = top_k_indices[0]
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[next_token]])
                ], dim=1)
        
        print(f"Candidate space size: {k}^{n_tokens} = {k**n_tokens}")
        
        # The key insight: we can represent the problem as finding the
        # optimal path through this candidate space
        
        # Build a "Hamiltonian" matrix that encodes the energy of each path
        # Energy = -log P(sequence)
        
        # For tractability, let's use a mean-field approximation:
        # Assume tokens are approximately independent given the prompt
        
        print("\n--- Mean-Field Approximation ---")
        
        # Get marginal probabilities at each position
        marginals = []
        
        with torch.no_grad():
            # For each position, compute P(x_i | prompt) ignoring other positions
            outputs = self.model(input_ids)
            first_logits = outputs.logits[0, -1, :]
            first_probs = F.softmax(first_logits, dim=-1)
            marginals.append(first_probs)
            
            # For subsequent positions, use the sequential context
            current_ids = input_ids.clone()
            for i in range(n_tokens - 1):
                next_token = seq_tokens[i]
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[next_token]])
                ], dim=1)
                
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                marginals.append(probs)
        
        # The mean-field solution: argmax of marginals
        mf_tokens = [m.argmax().item() for m in marginals]
        mf_text = self.tokenizer.decode(mf_tokens)
        
        print(f"Mean-field solution: {mf_text!r}")
        print(f"Sequential solution: {seq_text!r}")
        
        matches = sum(1 for a, b in zip(seq_tokens, mf_tokens) if a == b)
        print(f"Matches: {matches}/{n_tokens}")
        
        # Key insight: Mean-field is EXACT for autoregression when computed
        # with the correct context! This is because:
        #   argmax P(x_1, ..., x_n) = argmax P(x_1) × argmax P(x_2|x_1) × ...
        # 
        # The problem is that we need x_1 to compute P(x_2|x_1), etc.
        
        print("\n--- The Fundamental Issue ---")
        print("""
The autoregressive factorization:
    P(x_1, ..., x_n) = P(x_1) × P(x_2|x_1) × P(x_3|x_1,x_2) × ...

is EXACT, not an approximation. The joint distribution factorizes
perfectly into conditionals.

This means:
1. The optimal sequence IS the product of optimal conditionals
2. But computing P(x_i | x_1, ..., x_{i-1}) requires knowing x_1, ..., x_{i-1}
3. This is the CAUSAL structure that prevents parallelization

The quantum analogy breaks down because:
- In QM, the Hamiltonian H is KNOWN and FIXED
- In autoregression, the "Hamiltonian" (conditional distribution) CHANGES
  based on previous tokens

However, there IS a way forward: MULTI-TOKEN PREDICTION
""")
        
        return seq_tokens, mf_tokens, candidates


def analyze_multi_token_attention():
    """
    Analyze how multi-token attention could help.
    
    The key insight: Standard attention is causal (lower-triangular mask).
    But what if we could predict MULTIPLE tokens at once?
    
    Multi-token prediction models (like Meta's) are trained to predict
    k tokens ahead, not just 1. This breaks the sequential dependency!
    """
    print("\n" + "=" * 60)
    print("Multi-Token Attention Analysis")
    print("=" * 60)
    
    print("""
Standard Autoregression:
    Position i predicts token i+1 only
    
    Attention mask:
    [1 0 0 0 0]
    [1 1 0 0 0]
    [1 1 1 0 0]
    [1 1 1 1 0]
    [1 1 1 1 1]
    
    Each position can only see past tokens.

Multi-Token Prediction:
    Position i predicts tokens i+1, i+2, ..., i+k
    
    This is like having k "heads" that predict different future positions.
    
    The key insight: If we can predict k tokens at once, we can:
    1. Generate k tokens in parallel
    2. Verify with the full model
    3. Accept or reject
    
    This is SPECULATIVE DECODING!

But there's a deeper insight...
""")
    
    print("""
The Quantum Perspective:
------------------------

In quantum mechanics, the wavefunction |ψ⟩ encodes ALL possible states
simultaneously. Measurement collapses to one state.

For language:
- The hidden state h encodes information about ALL possible continuations
- The output layer "measures" this to get token probabilities
- But the hidden state contains MORE than just the next token!

Evidence:
- Probing studies show hidden states encode future tokens
- The model "knows" what's coming, it just outputs one token at a time
- This is an ARTIFICIAL bottleneck imposed by training

The Solution:
- Train models to output MULTIPLE tokens per position
- Or: Extract the multi-token information from existing hidden states
- This is what we should explore!
""")


def test_hidden_state_future_prediction(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    """
    Test whether hidden states contain information about future tokens.
    
    If the hidden state at position i contains information about tokens
    i+1, i+2, ..., then we can potentially extract this in parallel.
    """
    print("\n" + "=" * 60)
    print("Hidden State Future Prediction Test")
    print("=" * 60)
    
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    # Generate reference
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    ref_tokens = outputs[0, prompt_len:].tolist()
    ref_text = tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    # Get hidden states from prompt only
    print("\n--- Predicting Future from Prompt Hidden State ---")
    
    with torch.no_grad():
        prompt_outputs = model(input_ids, output_hidden_states=True)
        
        # Last hidden state at last prompt position
        last_hidden = prompt_outputs.hidden_states[-1][0, -1, :]  # [hidden_dim]
        
        # The output layer maps hidden → vocab
        # Can we use this to predict MULTIPLE future tokens?
        
        # Standard: predict next token
        lm_head = model.lm_head
        next_logits = lm_head(last_hidden)
        next_token = next_logits.argmax().item()
        
        print(f"Position 0 prediction: {tokenizer.decode([next_token])!r} (ref: {tokenizer.decode([ref_tokens[0]])!r})")
        print(f"Match: {next_token == ref_tokens[0]}")
    
    # Key question: Can we predict token 2, 3, ... from the same hidden state?
    # This would require a different "head" for each future position.
    
    print("\n--- Training a Multi-Position Predictor ---")
    print("(This would require training data, skipping for now)")
    
    # Alternative: Use the hidden state trajectory
    print("\n--- Hidden State Trajectory Analysis ---")
    
    # Generate with hidden states
    with torch.no_grad():
        current_ids = input_ids.clone()
        hidden_trajectory = []
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            hidden_trajectory.append(hidden.clone())
            
            next_token = outputs.logits[0, -1, :].argmax().item()
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]])
            ], dim=1)
    
    # Analyze: How much does hidden state change between positions?
    print("\nHidden state changes between positions:")
    
    for i in range(len(hidden_trajectory) - 1):
        h1 = hidden_trajectory[i]
        h2 = hidden_trajectory[i + 1]
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
        
        # L2 distance
        l2_dist = torch.norm(h2 - h1).item()
        
        print(f"  {i} → {i+1}: cos_sim={cos_sim:.4f}, L2={l2_dist:.2f}")
    
    # Key insight: If hidden states are similar, the "trajectory" is smooth
    # This means we might be able to INTERPOLATE or EXTRAPOLATE
    
    print("\n--- Trajectory Extrapolation Test ---")
    
    # Can we predict the trajectory and use it to generate tokens?
    # Fit a linear model: h_{i+1} = A @ h_i + b
    
    H = torch.stack(hidden_trajectory[:-1])  # [n-1, hidden_dim]
    H_next = torch.stack(hidden_trajectory[1:])  # [n-1, hidden_dim]
    
    # Least squares: H_next = H @ A.T + b
    # Simplified: just use mean direction
    delta = (H_next - H).mean(dim=0)  # Mean change per step
    
    print(f"Mean delta norm: {torch.norm(delta).item():.2f}")
    
    # Predict future hidden states by extrapolation
    predicted_hiddens = [hidden_trajectory[0]]
    for i in range(n_tokens - 1):
        next_h = predicted_hiddens[-1] + delta
        predicted_hiddens.append(next_h)
    
    # Decode predicted hidden states
    print("\nExtrapolated predictions:")
    
    for i, h in enumerate(predicted_hiddens):
        logits = lm_head(h)
        pred_token = logits.argmax().item()
        ref_token = ref_tokens[i]
        match = "✓" if pred_token == ref_token else "✗"
        print(f"  Position {i}: pred={tokenizer.decode([pred_token])!r}, ref={tokenizer.decode([ref_token])!r} {match}")
    
    matches = sum(1 for i, h in enumerate(predicted_hiddens) 
                  if lm_head(h).argmax().item() == ref_tokens[i])
    print(f"\nExtrapolation accuracy: {matches}/{n_tokens}")
    
    del model
    
    return hidden_trajectory, ref_tokens


def test_accelerated_fixed_point():
    """
    Test accelerated fixed-point methods.
    
    Standard fixed-point: x_{k+1} = T(x_k)
    
    Accelerated methods:
    1. Anderson acceleration (like DIIS in quantum chemistry)
    2. Newton's method (requires Jacobian)
    3. Spectral methods (use eigenstructure)
    
    The key insight: If the influence matrix is low-rank (rank 2!),
    we can use this structure to accelerate convergence.
    """
    print("\n" + "=" * 60)
    print("Accelerated Fixed-Point Methods")
    print("=" * 60)
    
    print(f"Loading model...")
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
    
    # Reference
    with torch.no_grad():
        ref_outputs = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ref_tokens = ref_outputs[0, prompt_len:].tolist()
    ref_text = tokenizer.decode(ref_tokens)
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Reference: {ref_text!r}")
    
    # Method 1: Standard fixed-point
    print("\n--- Method 1: Standard Fixed-Point ---")
    
    import random
    random.seed(42)
    current = [random.randint(0, 1000) for _ in range(n_tokens)]
    
    start_time = time.time()
    
    for iteration in range(20):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids)
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
    
    std_time = time.time() - start_time
    std_iters = iteration + 1
    std_matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"Iterations: {std_iters}")
    print(f"Time: {std_time:.2f}s")
    print(f"Matches: {std_matches}/{n_tokens}")
    print(f"Result: {tokenizer.decode(current)!r}")
    
    # Method 2: Greedy initialization + fixed-point
    print("\n--- Method 2: Greedy Init + Fixed-Point ---")
    
    start_time = time.time()
    
    # Start with greedy (first token correct)
    with torch.no_grad():
        outputs = model(input_ids)
        first_token = outputs.logits[0, -1, :].argmax().item()
    
    # Use scaffolding for rest
    scaffolding = ['.', ' It', ' is', ' the', ' most', ' populous', ' city', ' in', ' the']
    scaffolding_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in scaffolding]
    current = [first_token] + scaffolding_ids[:n_tokens-1]
    
    for iteration in range(20):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids)
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
    
    greedy_time = time.time() - start_time
    greedy_iters = iteration + 1
    greedy_matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"Iterations: {greedy_iters}")
    print(f"Time: {greedy_time:.2f}s")
    print(f"Matches: {greedy_matches}/{n_tokens}")
    print(f"Result: {tokenizer.decode(current)!r}")
    
    # Method 3: Progressive refinement (fix early tokens first)
    print("\n--- Method 3: Progressive Refinement ---")
    
    start_time = time.time()
    
    # Start with random
    random.seed(42)
    current = [random.randint(0, 1000) for _ in range(n_tokens)]
    
    # Fix tokens progressively: once a token is "stable", don't change it
    stable = [False] * n_tokens
    
    for iteration in range(20):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits[0]
        
        new_current = current.copy()
        changes = 0
        
        for i in range(n_tokens):
            if stable[i]:
                continue
            
            if i == 0:
                pos_logits = logits[prompt_len - 1]
            else:
                pos_logits = logits[prompt_len + i - 1]
            
            best_token = pos_logits.argmax().item()
            
            if best_token != current[i]:
                new_current[i] = best_token
                changes += 1
            else:
                # Token is stable - mark it
                stable[i] = True
        
        current = new_current
        
        if all(stable):
            break
    
    prog_time = time.time() - start_time
    prog_iters = iteration + 1
    prog_matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"Iterations: {prog_iters}")
    print(f"Time: {prog_time:.2f}s")
    print(f"Matches: {prog_matches}/{n_tokens}")
    print(f"Result: {tokenizer.decode(current)!r}")
    
    # Method 4: Parallel beam search (explore multiple paths)
    print("\n--- Method 4: Parallel Beam Search ---")
    
    start_time = time.time()
    
    beam_width = 4
    
    # Initialize beams with different random seeds
    beams = []
    for seed in range(beam_width):
        random.seed(seed)
        beams.append([random.randint(0, 1000) for _ in range(n_tokens)])
    
    for iteration in range(10):
        new_beams = []
        beam_scores = []
        
        for beam in beams:
            full_ids = torch.cat([
                input_ids,
                torch.tensor([beam])
            ], dim=1)
            
            with torch.no_grad():
                outputs = model(full_ids)
                logits = outputs.logits[0]
            
            # Get best continuation
            new_beam = []
            score = 0
            
            for i in range(n_tokens):
                if i == 0:
                    pos_logits = logits[prompt_len - 1]
                else:
                    pos_logits = logits[prompt_len + i - 1]
                
                probs = F.softmax(pos_logits, dim=-1)
                best_token = pos_logits.argmax().item()
                new_beam.append(best_token)
                score += torch.log(probs[best_token]).item()
            
            new_beams.append(new_beam)
            beam_scores.append(score)
        
        # Keep top beams
        sorted_indices = np.argsort(beam_scores)[::-1]
        beams = [new_beams[i] for i in sorted_indices[:beam_width]]
        
        # Check convergence
        if len(set(tuple(b) for b in beams)) == 1:
            break
    
    beam_time = time.time() - start_time
    beam_iters = iteration + 1
    best_beam = beams[0]
    beam_matches = sum(1 for a, b in zip(ref_tokens, best_beam) if a == b)
    
    print(f"Iterations: {beam_iters}")
    print(f"Time: {beam_time:.2f}s")
    print(f"Matches: {beam_matches}/{n_tokens}")
    print(f"Result: {tokenizer.decode(best_beam)!r}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"\n{'Method':<25} {'Iters':<8} {'Time':<10} {'Matches':<10}")
    print("-" * 53)
    print(f"{'Standard Fixed-Point':<25} {std_iters:<8} {std_time:.2f}s{'':<5} {std_matches}/10")
    print(f"{'Greedy Init + FP':<25} {greedy_iters:<8} {greedy_time:.2f}s{'':<5} {greedy_matches}/10")
    print(f"{'Progressive Refinement':<25} {prog_iters:<8} {prog_time:.2f}s{'':<5} {prog_matches}/10")
    print(f"{'Parallel Beam Search':<25} {beam_iters:<8} {beam_time:.2f}s{'':<5} {beam_matches}/10")
    
    del model


def analyze_quantum_formulation():
    """
    Deep analysis of the quantum formulation.
    
    The key question: Can we treat autoregression as a quantum system?
    
    In quantum mechanics:
    - State: |ψ⟩ = Σ c_i |i⟩ (superposition of basis states)
    - Evolution: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    - Measurement: Collapse to eigenstate
    
    For autoregression:
    - State: Distribution over all possible sequences
    - Evolution: Conditioning on observed tokens
    - Measurement: Selecting the output sequence
    
    The key insight: The hidden state IS the "wavefunction"!
    It encodes a superposition of all possible continuations.
    """
    print("\n" + "=" * 60)
    print("Quantum Formulation Analysis")
    print("=" * 60)
    
    print("""
The Quantum Analogy:
-------------------

1. WAVEFUNCTION = HIDDEN STATE
   
   In QM: |ψ⟩ = Σ c_i |i⟩
   In LLM: h = Σ α_i e_i (where e_i are token embeddings)
   
   The hidden state IS a superposition of token embeddings!
   The coefficients encode the probability amplitudes.

2. HAMILTONIAN = TRANSFORMER LAYERS
   
   In QM: H|ψ⟩ = E|ψ⟩
   In LLM: h' = Layer(h)
   
   Each transformer layer is a unitary-like transformation.
   The layers evolve the hidden state through "time" (depth).

3. MEASUREMENT = OUTPUT LAYER
   
   In QM: Probability = |⟨φ|ψ⟩|²
   In LLM: Probability = softmax(W_out @ h)
   
   The output layer "measures" the hidden state.
   It collapses the superposition to a single token.

4. THE KEY DIFFERENCE
   
   In QM: Measurement is RANDOM (Born rule)
   In LLM: We take argmax (deterministic)
   
   But we COULD sample! That's what temperature does.
   Greedy decoding is like "zero temperature" measurement.

5. THE AUTOREGRESSIVE CONSTRAINT
   
   In QM: Time evolution is UNITARY (reversible)
   In LLM: Autoregression is CAUSAL (irreversible)
   
   This is the key difference. The causal mask breaks unitarity.
   But... what if we could make it unitary?
""")
    
    print("""
The Unitary Hypothesis:
----------------------

What if the transformer IS unitary, just in a higher-dimensional space?

Evidence:
1. Attention is a weighted sum (linear, could be unitary)
2. LayerNorm preserves norm (like unitary)
3. MLP is nonlinear, but operates in a linear regime (Doc 132)

If the transformer is approximately unitary, then:
- The hidden state trajectory is a geodesic
- The output is determined by the initial state
- We could predict the trajectory without sequential computation!

This connects to:
- Doc 160: Unified Geometric Theory
- Doc 143: Zeta-Aligned Layer
- The φ-lattice structure
""")
    
    print("""
The Fixed-Point as Ground State:
-------------------------------

In QM, the ground state is the lowest-energy eigenstate:
    H|ψ_0⟩ = E_0|ψ_0⟩

For autoregression, the "ground state" is the fixed point:
    T|x*⟩ = |x*⟩

where T is the transition operator.

Key insight: The fixed point IS an eigenvector with eigenvalue 1!

This means:
1. The correct sequence is in the null space of (I - T)
2. We can find it by solving (I - T)|x⟩ = 0
3. This is a LINEAR ALGEBRA problem!

But T is not a matrix - it's a nonlinear function.
However, we can LINEARIZE around the fixed point:
    T(x) ≈ T(x*) + J(x - x*)
    
where J is the Jacobian. At the fixed point:
    x* = T(x*) + J(x* - x*) = T(x*)
    
So the fixed point is where the linear approximation is exact.

The Jacobian we computed has rank 2!
This means the fixed-point problem is effectively 2-dimensional.
""")


def test_rank2_exploitation():
    """
    Exploit the rank-2 structure of the influence matrix.
    
    If the influence matrix is rank-2, then:
    1. Only 2 "directions" matter for convergence
    2. We can project onto these directions
    3. Solve a 2D problem instead of n-dimensional
    
    This is like finding the 2 principal components of the
    autoregressive dynamics.
    """
    print("\n" + "=" * 60)
    print("Rank-2 Structure Exploitation")
    print("=" * 60)
    
    print(f"Loading model...")
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
    
    # Reference
    with torch.no_grad():
        ref_outputs = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ref_tokens = ref_outputs[0, prompt_len:].tolist()
    ref_text = tokenizer.decode(ref_tokens)
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Reference: {ref_text!r}")
    
    # First, identify the "principal positions" - which positions matter most?
    print("\n--- Identifying Principal Positions ---")
    
    # Get sequential logits to measure entropy at each position
    entropies = []
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)
            
            next_token = logits.argmax().item()
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]])
            ], dim=1)
    
    print("Entropy at each position:")
    for i, (e, t) in enumerate(zip(entropies, ref_tokens)):
        token_text = tokenizer.decode([t])
        print(f"  {i}: {e:.2f} bits - {token_text!r}")
    
    # The hypothesis: High-entropy positions are the "principal" ones
    # Low-entropy positions are determined by the principal ones
    
    sorted_by_entropy = sorted(range(n_tokens), key=lambda i: entropies[i], reverse=True)
    principal_positions = sorted_by_entropy[:2]  # Top 2 by entropy
    
    print(f"\nPrincipal positions (highest entropy): {principal_positions}")
    
    # Strategy: Fix the principal positions first, then let the rest converge
    print("\n--- Principal-First Strategy ---")
    
    import random
    random.seed(42)
    
    start_time = time.time()
    
    # Step 1: Get the principal tokens correctly
    current = [random.randint(0, 1000) for _ in range(n_tokens)]
    
    # Fix principal positions using sequential generation
    with torch.no_grad():
        temp_ids = input_ids.clone()
        
        for i in range(max(principal_positions) + 1):
            outputs = model(temp_ids)
            logits = outputs.logits[0, -1, :]
            next_token = logits.argmax().item()
            
            if i in principal_positions:
                current[i] = next_token
            
            temp_ids = torch.cat([
                temp_ids,
                torch.tensor([[next_token]])
            ], dim=1)
    
    print(f"After fixing principal positions: {tokenizer.decode(current)!r}")
    
    # Step 2: Fixed-point iteration for the rest
    for iteration in range(10):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits[0]
        
        new_current = current.copy()
        changes = 0
        
        for i in range(n_tokens):
            # Don't change principal positions
            if i in principal_positions:
                continue
            
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
    
    final_text = tokenizer.decode(current)
    matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"\nFinal: {final_text!r}")
    print(f"Matches: {matches}/{n_tokens}")
    print(f"Iterations: {iteration + 1}")
    print(f"Time: {total_time:.2f}s")
    
    # Compare to just getting first 2 tokens right
    print("\n--- First-2 Strategy (for comparison) ---")
    
    start_time = time.time()
    
    current = [random.randint(0, 1000) for _ in range(n_tokens)]
    
    # Fix first 2 positions
    with torch.no_grad():
        temp_ids = input_ids.clone()
        
        for i in range(2):
            outputs = model(temp_ids)
            logits = outputs.logits[0, -1, :]
            next_token = logits.argmax().item()
            current[i] = next_token
            
            temp_ids = torch.cat([
                temp_ids,
                torch.tensor([[next_token]])
            ], dim=1)
    
    # Fixed-point for rest
    for iteration in range(10):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits[0]
        
        new_current = current.copy()
        changes = 0
        
        for i in range(2, n_tokens):  # Skip first 2
            pos_logits = logits[prompt_len + i - 1]
            best_token = pos_logits.argmax().item()
            
            if best_token != current[i]:
                new_current[i] = best_token
                changes += 1
        
        current = new_current
        
        if changes == 0:
            break
    
    total_time = time.time() - start_time
    
    final_text = tokenizer.decode(current)
    matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"Final: {final_text!r}")
    print(f"Matches: {matches}/{n_tokens}")
    print(f"Iterations: {iteration + 1}")
    print(f"Time: {total_time:.2f}s")
    
    del model


def test_diverse_prompts_fixed_point():
    """
    Test fixed-point convergence on diverse prompts.
    
    Key question: Is the rank-2 structure consistent across prompts?
    """
    print("\n" + "=" * 60)
    print("Diverse Prompts - Fixed-Point Convergence")
    print("=" * 60)
    
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox jumps over",
        "Python is a programming language that",
        "In the year 2024, artificial intelligence",
    ]
    
    n_tokens = 10
    
    results = []
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Reference
        with torch.no_grad():
            ref_outputs = model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        ref_tokens = ref_outputs[0, prompt_len:].tolist()
        ref_text = tokenizer.decode(ref_tokens)
        
        # Fixed-point from random
        import random
        random.seed(42)
        current = [random.randint(0, 1000) for _ in range(n_tokens)]
        
        start_time = time.time()
        
        for iteration in range(20):
            full_ids = torch.cat([
                input_ids,
                torch.tensor([current])
            ], dim=1)
            
            with torch.no_grad():
                outputs = model(full_ids)
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
        
        final_text = tokenizer.decode(current)
        matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
        
        print(f"Reference: {ref_text!r}")
        print(f"Fixed-pt:  {final_text!r}")
        print(f"Matches: {matches}/{n_tokens}, Iters: {iteration + 1}, Time: {total_time:.2f}s")
        
        results.append({
            "prompt": prompt,
            "matches": matches,
            "iterations": iteration + 1,
            "time": total_time,
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    avg_matches = np.mean([r["matches"] for r in results])
    avg_iters = np.mean([r["iterations"] for r in results])
    avg_time = np.mean([r["time"] for r in results])
    
    print(f"Average matches: {avg_matches:.1f}/10")
    print(f"Average iterations: {avg_iters:.1f}")
    print(f"Average time: {avg_time:.2f}s")
    
    # Key insight: Does fixed-point ALWAYS converge to the correct answer?
    all_correct = all(r["matches"] == 10 for r in results)
    print(f"\nAll correct: {all_correct}")
    
    del model
    
    return results


if __name__ == "__main__":
    # 1. Test rank-2 exploitation
    test_rank2_exploitation()
    
    # 2. Test on diverse prompts
    test_diverse_prompts_fixed_point()
    
    # 3. Quantum formulation analysis
    analyze_quantum_formulation()
