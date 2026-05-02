#!/usr/bin/env python3
"""
Scaffolding-First Generation Experiment

Hypothesis: We can generate text faster by:
1. Predicting scaffolding (low-entropy tokens) first
2. Using spectral constraints to solve content (high-entropy tokens)

The scaffolding defines the structure, content fills in the meaning.

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949


@dataclass
class ScaffoldingConfig:
    """Configuration for scaffolding-first generation."""
    entropy_threshold: float = 2.0  # Below this = scaffolding
    n_content_candidates: int = 50  # Candidates for content positions
    use_spectral_constraints: bool = True


class ScaffoldingGenerator:
    """
    Scaffolding-first text generator.
    
    Strategy:
    1. Identify common scaffolding patterns (function words, punctuation)
    2. Generate scaffolding positions first (they're predictable)
    3. Use spectral methods to fill content positions
    """
    
    # Common scaffolding tokens (function words, punctuation)
    SCAFFOLDING_PATTERNS = {
        '.', ',', '!', '?', ':', ';', '-', '(', ')', '"', "'",
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'of', 'in', 'to', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'or', 'but', 'if', 'then', 'that', 'which', 'who',
        'it', 'its', 'this', 'that', 'these', 'those',
        'I', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her',
    }
    
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
        
        # Build scaffolding token set
        self._build_scaffolding_vocab()
        
        print(f"Vocab size: {self.vocab_size}")
        print(f"Scaffolding tokens: {len(self.scaffolding_ids)}")
    
    def _build_scaffolding_vocab(self):
        """Build set of scaffolding token IDs."""
        self.scaffolding_ids: Set[int] = set()
        
        for pattern in self.SCAFFOLDING_PATTERNS:
            # Try different tokenizations
            for prefix in ['', ' ', '\n']:
                text = prefix + pattern
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                self.scaffolding_ids.update(ids)
        
        # Also add common punctuation tokens directly
        for char in '.,!?;:\'"()-':
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            self.scaffolding_ids.update(ids)
    
    def is_scaffolding_token(self, token_id: int) -> bool:
        """Check if a token is scaffolding."""
        return token_id in self.scaffolding_ids
    
    def analyze_entropy_structure(self, prompt: str, n_tokens: int = 10
                                   ) -> Tuple[List[int], List[float], List[bool]]:
        """
        Generate tokens and analyze entropy structure.
        
        Returns:
            (tokens, entropies, is_scaffolding)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        tokens = []
        entropies = []
        is_scaffolding = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                # Entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                # Top token
                top_idx = logits.argmax().item()
                
                tokens.append(top_idx)
                entropies.append(entropy)
                is_scaffolding.append(self.is_scaffolding_token(top_idx))
                
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[top_idx]])
                ], dim=1)
        
        return tokens, entropies, is_scaffolding
    
    def predict_scaffolding_positions(self, prompt: str, n_tokens: int = 10,
                                       config: ScaffoldingConfig = None
                                       ) -> List[int]:
        """
        Predict which positions will be scaffolding.
        
        Uses heuristics based on:
        1. Position in sentence (punctuation at ends)
        2. φ-spacing patterns
        3. Common scaffolding patterns
        """
        if config is None:
            config = ScaffoldingConfig()
        
        # Heuristic: scaffolding tends to appear at regular intervals
        # Based on our observation: positions 1, 3, 6, 7, 8, 9 were scaffolding
        # This suggests scaffolding density increases toward end of sentence
        
        scaffolding_positions = []
        
        for pos in range(n_tokens):
            # Heuristic 1: Position 1 is often punctuation after first content word
            if pos == 1:
                scaffolding_positions.append(pos)
            # Heuristic 2: Positions after φ² spacing tend to be scaffolding
            elif pos > 2 and (pos % 3 == 0 or pos > 5):
                scaffolding_positions.append(pos)
        
        return scaffolding_positions
    
    def generate_with_scaffolding_first(self, prompt: str, n_tokens: int = 10,
                                         config: ScaffoldingConfig = None
                                         ) -> Tuple[List[int], Dict]:
        """
        Generate using scaffolding-first approach.
        
        Strategy:
        1. Predict scaffolding positions
        2. Generate scaffolding tokens (fast, predictable)
        3. Fill content positions using spectral constraints
        """
        if config is None:
            config = ScaffoldingConfig()
        
        start_time = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Step 1: Predict scaffolding positions
        scaffolding_positions = self.predict_scaffolding_positions(prompt, n_tokens, config)
        content_positions = [i for i in range(n_tokens) if i not in scaffolding_positions]
        
        # Step 2: Initialize output with placeholders
        output_tokens = [None] * n_tokens
        
        # Step 3: Generate scaffolding tokens first
        # For scaffolding, we use a simplified model: just predict based on position
        scaffolding_candidates = self._get_scaffolding_candidates(prompt, n_tokens)
        
        for pos in scaffolding_positions:
            if pos < len(scaffolding_candidates):
                output_tokens[pos] = scaffolding_candidates[pos]
        
        # Step 4: Generate content tokens using spectral constraints
        # Content tokens must "fit" between scaffolding
        content_tokens = self._generate_content_spectral(
            prompt, output_tokens, content_positions, config
        )
        
        for pos, token in zip(content_positions, content_tokens):
            output_tokens[pos] = token
        
        # Fill any remaining None positions with fallback
        for i, token in enumerate(output_tokens):
            if token is None:
                output_tokens[i] = self._fallback_token(prompt, i)
        
        elapsed = time.time() - start_time
        
        info = {
            "time": elapsed,
            "tokens_per_second": n_tokens / elapsed,
            "scaffolding_positions": scaffolding_positions,
            "content_positions": content_positions,
            "scaffolding_ratio": len(scaffolding_positions) / n_tokens,
        }
        
        return output_tokens, info
    
    def _get_scaffolding_candidates(self, prompt: str, n_tokens: int) -> List[int]:
        """Get likely scaffolding tokens for each position."""
        # Use a single forward pass to get top scaffolding candidates
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        candidates = []
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            
            # For each position, find best scaffolding token
            for pos in range(n_tokens):
                # Mask non-scaffolding tokens
                masked_logits = logits.clone()
                for i in range(self.vocab_size):
                    if i not in self.scaffolding_ids:
                        masked_logits[i] = float('-inf')
                
                best_scaffolding = masked_logits.argmax().item()
                candidates.append(best_scaffolding)
        
        return candidates
    
    def _generate_content_spectral(self, prompt: str, partial_output: List[Optional[int]],
                                    content_positions: List[int],
                                    config: ScaffoldingConfig) -> List[int]:
        """
        Generate content tokens using spectral constraints.
        
        The content must "resonate" with:
        1. The prompt semantics
        2. The scaffolding structure
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Get semantic trajectory from prompt
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            trajectory = outputs.hidden_states[-1][0, -1, :]
        
        content_tokens = []
        
        for pos in content_positions:
            # Get top candidates based on trajectory similarity
            with torch.no_grad():
                similarities = F.cosine_similarity(
                    self.embeddings,
                    trajectory.unsqueeze(0),
                    dim=1
                )
                
                # Exclude scaffolding tokens from content
                for scaff_id in self.scaffolding_ids:
                    similarities[scaff_id] = -1.0
                
                # Get top candidate
                top_idx = similarities.argmax().item()
                content_tokens.append(top_idx)
                
                # Update trajectory based on selected token
                # (Simple: just average with selected embedding)
                trajectory = 0.7 * trajectory + 0.3 * self.embeddings[top_idx]
        
        return content_tokens
    
    def _fallback_token(self, prompt: str, position: int) -> int:
        """Fallback token generation for unfilled positions."""
        # Use the most common token (usually space or 'the')
        return self.tokenizer.encode(' the', add_special_tokens=False)[0]
    
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
        
        tokens = outputs[0, input_ids.shape[1]:].tolist()
        
        info = {
            "time": elapsed,
            "tokens_per_second": len(tokens) / elapsed,
        }
        
        return tokens, info


def test_scaffolding_generation():
    """Test scaffolding-first generation."""
    print("=" * 60)
    print("Scaffolding-First Generation Test")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Analyze entropy structure first
    print("\n--- Entropy Structure Analysis ---")
    tokens, entropies, is_scaff = gen.analyze_entropy_structure(prompt, n_tokens)
    
    for i, (tok, ent, scaff) in enumerate(zip(tokens, entropies, is_scaff)):
        tok_str = gen.tokenizer.decode([tok])
        scaff_str = "SCAFF" if scaff else "CONTENT"
        print(f"  {i}: {tok_str!r:15} entropy={ent:.2f}  {scaff_str}")
    
    # Sequential generation (reference)
    print("\n--- Sequential Generation (Reference) ---")
    seq_tokens, seq_info = gen.generate_sequential(prompt, n_tokens)
    seq_text = gen.tokenizer.decode(seq_tokens)
    
    print(f"Output: {seq_text!r}")
    print(f"Time: {seq_info['time']:.2f}s")
    print(f"Speed: {seq_info['tokens_per_second']:.2f} tok/s")
    
    # Scaffolding-first generation
    print("\n--- Scaffolding-First Generation ---")
    scaff_tokens, scaff_info = gen.generate_with_scaffolding_first(prompt, n_tokens)
    scaff_text = gen.tokenizer.decode(scaff_tokens)
    
    print(f"Output: {scaff_text!r}")
    print(f"Time: {scaff_info['time']:.2f}s")
    print(f"Speed: {scaff_info['tokens_per_second']:.2f} tok/s")
    print(f"Scaffolding positions: {scaff_info['scaffolding_positions']}")
    print(f"Content positions: {scaff_info['content_positions']}")
    
    # Compare
    print("\n--- Comparison ---")
    print(f"Sequential: {seq_text!r}")
    print(f"Scaffolding: {scaff_text!r}")
    
    # Token-by-token comparison
    matches = sum(1 for a, b in zip(seq_tokens, scaff_tokens) if a == b)
    print(f"Token matches: {matches}/{min(len(seq_tokens), len(scaff_tokens))}")
    
    # Speedup
    speedup = seq_info['time'] / scaff_info['time']
    print(f"Speedup: {speedup:.1f}x")
    
    del gen.model
    
    return seq_text, scaff_text, speedup


def test_multiple_prompts():
    """Test scaffolding-first on multiple prompts."""
    print("\n" + "=" * 60)
    print("Testing Multiple Prompts")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox jumps over",
        "In the beginning, there was",
        "Python is a programming language that",
    ]
    
    results = []
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        
        # Sequential
        seq_tokens, seq_info = gen.generate_sequential(prompt, 10)
        seq_text = gen.tokenizer.decode(seq_tokens)
        
        # Scaffolding-first
        scaff_tokens, scaff_info = gen.generate_with_scaffolding_first(prompt, 10)
        scaff_text = gen.tokenizer.decode(scaff_tokens)
        
        matches = sum(1 for a, b in zip(seq_tokens, scaff_tokens) if a == b)
        speedup = seq_info['time'] / scaff_info['time']
        
        print(f"  Sequential:  {seq_text!r}")
        print(f"  Scaffolding: {scaff_text!r}")
        print(f"  Matches: {matches}/10, Speedup: {speedup:.1f}x")
        
        results.append({
            "prompt": prompt,
            "sequential": seq_text,
            "scaffolding": scaff_text,
            "matches": matches,
            "speedup": speedup,
        })
    
    # Summary
    print("\n--- Summary ---")
    avg_matches = np.mean([r["matches"] for r in results])
    avg_speedup = np.mean([r["speedup"] for r in results])
    print(f"Average matches: {avg_matches:.1f}/10")
    print(f"Average speedup: {avg_speedup:.1f}x")
    
    del gen.model
    
    return results


def iterative_refinement_experiment():
    """
    Test iterative refinement approach.
    
    Instead of one-shot scaffolding-first, we:
    1. Generate initial guess (fast)
    2. Identify low-confidence positions
    3. Refine those positions using full model
    4. Repeat until convergence
    """
    print("\n" + "=" * 60)
    print("Iterative Refinement Experiment")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Step 1: Get initial scaffolding-first output
    print("\n--- Step 1: Initial Scaffolding-First ---")
    initial_tokens, initial_info = gen.generate_with_scaffolding_first(prompt, n_tokens)
    initial_text = gen.tokenizer.decode(initial_tokens)
    print(f"Initial: {initial_text!r}")
    
    # Step 2: Score each position using full model
    print("\n--- Step 2: Score Positions ---")
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    
    position_scores = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i, token in enumerate(initial_tokens):
            outputs = gen.model(current_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Score = probability of the token we chose
            score = probs[token].item()
            position_scores.append(score)
            
            # Append token
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]])
            ], dim=1)
    
    print("Position scores:")
    for i, (tok, score) in enumerate(zip(initial_tokens, position_scores)):
        tok_str = gen.tokenizer.decode([tok])
        quality = "GOOD" if score > 0.1 else "BAD"
        print(f"  {i}: {tok_str!r:15} score={score:.4f}  {quality}")
    
    # Step 3: Refine low-scoring positions
    print("\n--- Step 3: Refine Low-Scoring Positions ---")
    
    refined_tokens = initial_tokens.copy()
    threshold = 0.05  # Refine positions with score < 5%
    
    positions_to_refine = [i for i, s in enumerate(position_scores) if s < threshold]
    print(f"Positions to refine: {positions_to_refine}")
    
    # Refine each position using full model
    with torch.no_grad():
        for pos in positions_to_refine:
            # Build context up to this position
            context_ids = list(input_ids[0].tolist()) + refined_tokens[:pos]
            context_tensor = torch.tensor([context_ids])
            
            outputs = gen.model(context_tensor)
            logits = outputs.logits[0, -1, :]
            
            # Get best token
            best_token = logits.argmax().item()
            refined_tokens[pos] = best_token
            
            old_str = gen.tokenizer.decode([initial_tokens[pos]])
            new_str = gen.tokenizer.decode([best_token])
            print(f"  Position {pos}: {old_str!r} -> {new_str!r}")
    
    refined_text = gen.tokenizer.decode(refined_tokens)
    print(f"\nRefined: {refined_text!r}")
    
    # Compare with reference
    print("\n--- Comparison with Reference ---")
    ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
    ref_text = gen.tokenizer.decode(ref_tokens)
    
    print(f"Reference:   {ref_text!r}")
    print(f"Initial:     {initial_text!r}")
    print(f"Refined:     {refined_text!r}")
    
    initial_matches = sum(1 for a, b in zip(ref_tokens, initial_tokens) if a == b)
    refined_matches = sum(1 for a, b in zip(ref_tokens, refined_tokens) if a == b)
    
    print(f"\nInitial matches: {initial_matches}/10")
    print(f"Refined matches: {refined_matches}/10")
    print(f"Improvement: +{refined_matches - initial_matches} tokens")
    
    del gen.model


def speculative_decoding_experiment():
    """
    Test speculative decoding approach.
    
    This is a well-known technique:
    1. Use a small/fast model to generate draft tokens
    2. Verify with the full model in parallel
    3. Accept verified tokens, reject and regenerate bad ones
    
    We adapt this using entropy-based speculation:
    - High-confidence positions: accept draft
    - Low-confidence positions: verify with full model
    """
    print("\n" + "=" * 60)
    print("Speculative Decoding with Entropy-Based Verification")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Reference generation
    print("\n--- Reference (Sequential) ---")
    ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
    ref_text = gen.tokenizer.decode(ref_tokens)
    print(f"Output: {ref_text!r}")
    print(f"Time: {ref_info['time']:.2f}s")
    
    # Speculative approach: generate all tokens in one pass, then verify
    print("\n--- Speculative Decoding ---")
    
    start_time = time.time()
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    
    # Step 1: Generate draft using greedy decoding with entropy tracking
    draft_tokens = []
    draft_entropies = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = gen.model(current_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            # Top token
            top_idx = logits.argmax().item()
            
            draft_tokens.append(top_idx)
            draft_entropies.append(entropy)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[top_idx]])
            ], dim=1)
    
    draft_text = gen.tokenizer.decode(draft_tokens)
    draft_time = time.time() - start_time
    
    print(f"Draft: {draft_text!r}")
    print(f"Draft time: {draft_time:.2f}s")
    
    # Step 2: Identify high-entropy (uncertain) positions
    mean_entropy = np.mean(draft_entropies)
    uncertain_positions = [i for i, e in enumerate(draft_entropies) if e > mean_entropy]
    certain_positions = [i for i, e in enumerate(draft_entropies) if e <= mean_entropy]
    
    print(f"\nEntropies: {[f'{e:.2f}' for e in draft_entropies]}")
    print(f"Mean entropy: {mean_entropy:.2f}")
    print(f"Certain positions (keep): {certain_positions}")
    print(f"Uncertain positions (verify): {uncertain_positions}")
    
    # Step 3: For uncertain positions, we would verify with a larger model
    # Here we just show which tokens would be kept vs verified
    
    print("\nToken analysis:")
    for i, (tok, ent) in enumerate(zip(draft_tokens, draft_entropies)):
        tok_str = gen.tokenizer.decode([tok])
        status = "KEEP" if i in certain_positions else "VERIFY"
        print(f"  {i}: {tok_str!r:15} entropy={ent:.2f}  {status}")
    
    # The key insight: certain positions don't need verification
    # This could enable parallel verification of uncertain positions
    
    verification_ratio = len(uncertain_positions) / n_tokens
    potential_speedup = 1 / (1 - (1 - verification_ratio) * 0.5)  # Rough estimate
    
    print(f"\nVerification ratio: {verification_ratio*100:.0f}%")
    print(f"Potential speedup (if uncertain verified in parallel): {potential_speedup:.1f}x")
    
    del gen.model


def parallel_verification_experiment():
    """
    Test parallel verification of multiple token positions.
    
    Key insight: We can verify multiple positions in a single forward pass
    by computing the probability of each token given its context.
    """
    print("\n" + "=" * 60)
    print("Parallel Verification Experiment")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Generate reference
    ref_tokens, _ = gen.generate_sequential(prompt, n_tokens)
    ref_text = gen.tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    # Generate a "draft" (in practice, this would come from a smaller model)
    # For now, we'll use the reference but corrupt some positions
    draft_tokens = ref_tokens.copy()
    
    # Corrupt positions 2, 4, 6 with random tokens
    import random
    random.seed(42)
    corrupted_positions = [2, 4, 6]
    for pos in corrupted_positions:
        draft_tokens[pos] = random.randint(0, 1000)
    
    draft_text = gen.tokenizer.decode(draft_tokens)
    print(f"Draft (corrupted): {draft_text!r}")
    print(f"Corrupted positions: {corrupted_positions}")
    
    # Parallel verification: compute probability of each draft token
    print("\n--- Parallel Verification ---")
    
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    full_sequence = torch.cat([
        input_ids,
        torch.tensor([draft_tokens])
    ], dim=1)
    
    start_time = time.time()
    
    with torch.no_grad():
        # Single forward pass for entire sequence
        outputs = gen.model(full_sequence)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
    
    verification_time = time.time() - start_time
    
    # For each draft position, check if the draft token matches the model's prediction
    prompt_len = input_ids.shape[1]
    
    print("\nVerification results:")
    verified_tokens = []
    needs_regeneration = []
    
    for i in range(n_tokens):
        # Logits at position (prompt_len + i - 1) predict token at position (prompt_len + i)
        position_logits = logits[prompt_len + i - 1]
        probs = F.softmax(position_logits, dim=-1)
        
        draft_token = draft_tokens[i]
        draft_prob = probs[draft_token].item()
        best_token = position_logits.argmax().item()
        best_prob = probs[best_token].item()
        
        # Accept if draft token has reasonable probability (> 1%)
        accept = draft_prob > 0.01
        
        if accept:
            verified_tokens.append(draft_token)
        else:
            verified_tokens.append(best_token)
            needs_regeneration.append(i)
        
        draft_str = gen.tokenizer.decode([draft_token])
        best_str = gen.tokenizer.decode([best_token])
        status = "ACCEPT" if accept else "REJECT"
        
        print(f"  {i}: draft={draft_str!r:10} p={draft_prob:.4f}  best={best_str!r:10} p={best_prob:.4f}  {status}")
    
    verified_text = gen.tokenizer.decode(verified_tokens)
    
    print(f"\nVerified: {verified_text!r}")
    print(f"Verification time: {verification_time*1000:.0f}ms (single forward pass)")
    print(f"Positions needing regeneration: {needs_regeneration}")
    
    # Compare
    matches_draft = sum(1 for a, b in zip(ref_tokens, draft_tokens) if a == b)
    matches_verified = sum(1 for a, b in zip(ref_tokens, verified_tokens) if a == b)
    
    print(f"\nDraft matches: {matches_draft}/10")
    print(f"Verified matches: {matches_verified}/10")
    print(f"Improvement: +{matches_verified - matches_draft} tokens")
    
    del gen.model


def iterative_parallel_generation():
    """
    The key experiment: Generate tokens using iterative parallel refinement.
    
    Strategy:
    1. Start with a draft (could be random, could be from fast model)
    2. Verify all positions in parallel (single forward pass)
    3. Replace rejected tokens with model's predictions
    4. Repeat until convergence
    
    This is like solving a crossword puzzle iteratively.
    """
    print("\n" + "=" * 60)
    print("Iterative Parallel Generation")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Reference generation
    ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
    ref_text = gen.tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    print(f"Reference time: {ref_info['time']:.2f}s")
    
    # Start with random draft
    print("\n--- Iterative Parallel Generation ---")
    
    import random
    random.seed(42)
    
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    # Initialize with random tokens from top-1000 most common
    draft_tokens = [random.randint(0, 1000) for _ in range(n_tokens)]
    
    start_time = time.time()
    
    max_iterations = 10
    iteration_times = []
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Build full sequence
        full_sequence = torch.cat([
            input_ids,
            torch.tensor([draft_tokens])
        ], dim=1)
        
        # Single forward pass
        with torch.no_grad():
            outputs = gen.model(full_sequence)
            logits = outputs.logits[0]
        
        # Check each position and update if needed
        changes = 0
        new_tokens = []
        
        for i in range(n_tokens):
            # Logits at position (prompt_len + i - 1) predict token at position (prompt_len + i)
            if i == 0:
                # First generated token: use last prompt position
                position_logits = logits[prompt_len - 1]
            else:
                # Subsequent tokens: use previous generated position
                position_logits = logits[prompt_len + i - 1]
            
            probs = F.softmax(position_logits, dim=-1)
            
            draft_token = draft_tokens[i]
            draft_prob = probs[draft_token].item()
            best_token = position_logits.argmax().item()
            
            # Accept if probability > threshold, otherwise use best
            if draft_prob > 0.01:
                new_tokens.append(draft_token)
            else:
                new_tokens.append(best_token)
                if best_token != draft_token:
                    changes += 1
        
        draft_tokens = new_tokens
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        draft_text = gen.tokenizer.decode(draft_tokens)
        matches = sum(1 for a, b in zip(ref_tokens, draft_tokens) if a == b)
        
        print(f"  Iter {iteration+1}: {draft_text!r}")
        print(f"          matches={matches}/10, changes={changes}, time={iter_time*1000:.0f}ms")
        
        # Converged if no changes
        if changes == 0:
            print(f"  Converged after {iteration+1} iterations!")
            break
    
    total_time = time.time() - start_time
    
    final_text = gen.tokenizer.decode(draft_tokens)
    final_matches = sum(1 for a, b in zip(ref_tokens, draft_tokens) if a == b)
    
    print(f"\n--- Results ---")
    print(f"Reference:  {ref_text!r}")
    print(f"Generated:  {final_text!r}")
    print(f"Matches: {final_matches}/10")
    print(f"Total time: {total_time:.2f}s")
    print(f"Iterations: {iteration+1}")
    print(f"Avg time per iteration: {np.mean(iteration_times)*1000:.0f}ms")
    
    # Compare with sequential
    print(f"\nSequential time: {ref_info['time']:.2f}s")
    print(f"Parallel time: {total_time:.2f}s")
    print(f"Speedup: {ref_info['time']/total_time:.2f}x")
    
    del gen.model
    
    return final_matches, total_time, ref_info['time']


def smart_draft_parallel_generation():
    """
    Use a smarter initial draft based on prompt trajectory.
    
    Instead of random tokens, we:
    1. Get the semantic trajectory from the prompt
    2. Find tokens that are similar to the trajectory
    3. Use those as the initial draft
    """
    print("\n" + "=" * 60)
    print("Smart Draft Parallel Generation")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Reference
    ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
    ref_text = gen.tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    # Get semantic trajectory
    print("\n--- Creating Smart Draft ---")
    
    with torch.no_grad():
        outputs = gen.model(input_ids, output_hidden_states=True)
        trajectory = outputs.hidden_states[-1][0, -1, :]
        
        # Also get the logits for first position
        first_logits = outputs.logits[0, -1, :]
    
    # Create draft using top tokens from first position + trajectory similarity
    draft_tokens = []
    
    # First token: use actual model prediction
    draft_tokens.append(first_logits.argmax().item())
    
    # Remaining tokens: use trajectory similarity
    with torch.no_grad():
        similarities = F.cosine_similarity(
            gen.embeddings,
            trajectory.unsqueeze(0),
            dim=1
        )
        
        # Get top-k similar tokens
        top_k = 100
        top_indices = torch.argsort(-similarities)[:top_k]
        
        # Use top tokens for remaining positions
        for i in range(1, n_tokens):
            # Cycle through top tokens
            draft_tokens.append(top_indices[i % top_k].item())
    
    draft_text = gen.tokenizer.decode(draft_tokens)
    print(f"Initial draft: {draft_text!r}")
    
    # Now iterate
    print("\n--- Iterative Refinement ---")
    
    start_time = time.time()
    max_iterations = 10
    
    for iteration in range(max_iterations):
        full_sequence = torch.cat([
            input_ids,
            torch.tensor([draft_tokens])
        ], dim=1)
        
        with torch.no_grad():
            outputs = gen.model(full_sequence)
            logits = outputs.logits[0]
        
        changes = 0
        new_tokens = []
        
        for i in range(n_tokens):
            if i == 0:
                position_logits = logits[prompt_len - 1]
            else:
                position_logits = logits[prompt_len + i - 1]
            
            probs = F.softmax(position_logits, dim=-1)
            
            draft_token = draft_tokens[i]
            draft_prob = probs[draft_token].item()
            best_token = position_logits.argmax().item()
            
            if draft_prob > 0.01:
                new_tokens.append(draft_token)
            else:
                new_tokens.append(best_token)
                if best_token != draft_token:
                    changes += 1
        
        draft_tokens = new_tokens
        
        draft_text = gen.tokenizer.decode(draft_tokens)
        matches = sum(1 for a, b in zip(ref_tokens, draft_tokens) if a == b)
        
        print(f"  Iter {iteration+1}: {draft_text!r} (matches={matches}, changes={changes})")
        
        if changes == 0:
            print(f"  Converged!")
            break
    
    total_time = time.time() - start_time
    
    final_text = gen.tokenizer.decode(draft_tokens)
    final_matches = sum(1 for a, b in zip(ref_tokens, draft_tokens) if a == b)
    
    print(f"\n--- Results ---")
    print(f"Reference:  {ref_text!r}")
    print(f"Generated:  {final_text!r}")
    print(f"Matches: {final_matches}/10")
    print(f"Time: {total_time:.2f}s (sequential: {ref_info['time']:.2f}s)")
    print(f"Speedup: {ref_info['time']/total_time:.2f}x")
    
    del gen.model


def one_shot_parallel_generation():
    """
    The ultimate test: Can we generate correct output in ONE forward pass?
    
    Strategy:
    1. Use the model's own predictions from a single forward pass
    2. Each position predicts based on the PROMPT only (not previous generated tokens)
    3. See how close we get to the reference
    
    This tests whether the prompt contains enough information to predict all tokens.
    """
    print("\n" + "=" * 60)
    print("One-Shot Parallel Generation")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox jumps over",
        "Python is a programming language that",
        "In the year 2024, artificial intelligence",
    ]
    
    results = []
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        
        n_tokens = 10
        
        # Reference
        ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
        ref_text = gen.tokenizer.decode(ref_tokens)
        
        input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # One-shot: Get all predictions from prompt alone
        start_time = time.time()
        
        with torch.no_grad():
            outputs = gen.model(input_ids)
            first_logits = outputs.logits[0, -1, :]
        
        # First token is easy - just use the prediction
        one_shot_tokens = [first_logits.argmax().item()]
        
        # For remaining tokens, we need to think differently
        # The model can only predict the NEXT token, not tokens 2, 3, 4...
        # So one-shot parallel is fundamentally limited
        
        # BUT: What if we use the hidden state to predict ALL tokens?
        # This is the spectral hypothesis - the hidden state contains
        # information about the entire output trajectory
        
        one_shot_time = time.time() - start_time
        
        # For now, just show the first token prediction
        first_token_text = gen.tokenizer.decode([one_shot_tokens[0]])
        ref_first_token = gen.tokenizer.decode([ref_tokens[0]])
        
        match = one_shot_tokens[0] == ref_tokens[0]
        
        print(f"  Reference first token: {ref_first_token!r}")
        print(f"  One-shot first token:  {first_token_text!r}")
        print(f"  Match: {match}")
        print(f"  One-shot time: {one_shot_time*1000:.0f}ms")
        
        results.append({
            "prompt": prompt,
            "match": match,
            "ref_first": ref_first_token,
            "pred_first": first_token_text,
        })
    
    # Summary
    print("\n--- Summary ---")
    matches = sum(1 for r in results if r["match"])
    print(f"First token accuracy: {matches}/{len(results)} = {matches/len(results)*100:.0f}%")
    
    del gen.model
    
    return results


def two_pass_generation():
    """
    Two-pass generation: The practical sweet spot.
    
    Strategy:
    1. First pass: Generate draft using greedy decoding (fast)
    2. Second pass: Verify and fix in parallel (single forward pass)
    
    This should give us correct output in 2 forward passes per token
    instead of 1 forward pass per token.
    """
    print("\n" + "=" * 60)
    print("Two-Pass Generation")
    print("=" * 60)
    
    gen = ScaffoldingGenerator()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    print(f"\nPrompt: {prompt!r}")
    
    # Reference
    ref_tokens, ref_info = gen.generate_sequential(prompt, n_tokens)
    ref_text = gen.tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    print(f"Reference time: {ref_info['time']:.2f}s")
    
    input_ids = gen.tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    # Pass 1: Generate draft using standard sequential decoding
    print("\n--- Pass 1: Generate Draft ---")
    
    start_time = time.time()
    
    draft_tokens = []
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = gen.model(current_ids)
            logits = outputs.logits[0, -1, :]
            next_token = logits.argmax().item()
            draft_tokens.append(next_token)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]])
            ], dim=1)
    
    pass1_time = time.time() - start_time
    draft_text = gen.tokenizer.decode(draft_tokens)
    
    print(f"Draft: {draft_text!r}")
    print(f"Pass 1 time: {pass1_time:.2f}s")
    
    # Pass 2: Verify in parallel
    print("\n--- Pass 2: Parallel Verification ---")
    
    pass2_start = time.time()
    
    full_sequence = torch.cat([
        input_ids,
        torch.tensor([draft_tokens])
    ], dim=1)
    
    with torch.no_grad():
        outputs = gen.model(full_sequence)
        logits = outputs.logits[0]
    
    # Check each position
    verified_tokens = []
    corrections = 0
    
    for i in range(n_tokens):
        if i == 0:
            position_logits = logits[prompt_len - 1]
        else:
            position_logits = logits[prompt_len + i - 1]
        
        probs = F.softmax(position_logits, dim=-1)
        draft_token = draft_tokens[i]
        draft_prob = probs[draft_token].item()
        best_token = position_logits.argmax().item()
        
        # Accept if probability > 1%
        if draft_prob > 0.01:
            verified_tokens.append(draft_token)
        else:
            verified_tokens.append(best_token)
            corrections += 1
    
    pass2_time = time.time() - pass2_start
    
    verified_text = gen.tokenizer.decode(verified_tokens)
    
    print(f"Verified: {verified_text!r}")
    print(f"Corrections: {corrections}")
    print(f"Pass 2 time: {pass2_time:.2f}s")
    
    # Results
    total_time = pass1_time + pass2_time
    
    draft_matches = sum(1 for a, b in zip(ref_tokens, draft_tokens) if a == b)
    verified_matches = sum(1 for a, b in zip(ref_tokens, verified_tokens) if a == b)
    
    print(f"\n--- Results ---")
    print(f"Reference: {ref_text!r}")
    print(f"Draft:     {draft_text!r} ({draft_matches}/10 matches)")
    print(f"Verified:  {verified_text!r} ({verified_matches}/10 matches)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Sequential time: {ref_info['time']:.2f}s")
    print(f"Speedup: {ref_info['time']/total_time:.2f}x")
    
    # The key insight: if draft is already correct, verification is just confirmation
    # If draft has errors, we catch them in one pass
    
    del gen.model


if __name__ == "__main__":
    # iterative_parallel_generation()  # Works but slow (10 iterations)
    # smart_draft_parallel_generation()  # Similar
    one_shot_parallel_generation()
    two_pass_generation()
