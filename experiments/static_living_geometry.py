#!/usr/bin/env python3
"""
Static + Living Geometry: Eliminating Autoregression
=====================================================

The hypothesis:
- STATIC GEOMETRY: Platonic Ideals, angles, DRUM (startup state)
- LIVING GEOMETRY: Current position, accumulated rotations (memory)

If memory and computation are the same geometric operation,
then the output is implicit in the rotation - we don't need
to generate token-by-token.

Key insight:
- Autoregression: Token → Forward pass → Token → Forward pass → ...
- Geometric: Input shape → Rotate → Output shape (all at once)

The entire response is the ENDPOINT of a geometric trajectory.

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


@dataclass
class StaticGeometry:
    """
    The startup state - fixed, never changes.
    
    Contains:
    - Platonic Ideals (dimension intersections)
    - Relationship angles
    - DRUM (token embeddings)
    - Pattern templates
    """
    ideals: Dict[str, torch.Tensor] = field(default_factory=dict)
    angles: Dict[str, float] = field(default_factory=dict)
    drum: Optional[torch.Tensor] = None  # Token embeddings
    templates: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class LivingGeometry:
    """
    The memory state - evolves while maintaining structure.
    
    Contains:
    - Current position in semantic space
    - Accumulated rotations from context
    - Active ideals (which relationships are primed)
    - Conversation shape
    """
    position: Optional[torch.Tensor] = None
    accumulated_rotation: float = 0.0
    active_ideals: List[str] = field(default_factory=list)
    context_shape: Optional[torch.Tensor] = None


class GeometricStateMachine:
    """
    A state machine that uses geometry instead of autoregression.
    
    The key insight: if memory = computation = rotation,
    then we can compute the output shape directly without
    generating token-by-token.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize static geometry
        self.static = StaticGeometry()
        self.static.drum = model.model.embed_tokens.weight.data
        
        # Initialize living geometry
        self.living = LivingGeometry()
        
        # Learn static geometry from model
        self._learn_static_geometry()
    
    def _learn_static_geometry(self):
        """Learn Platonic Ideals and angles from the model."""
        
        # Capital-of relationship
        capital_pairs = [
            ("France", " Paris"),
            ("Germany", " Berlin"),
            ("Italy", " Rome"),
            ("Spain", " Madrid"),
        ]
        
        axes = []
        angles = []
        
        for entity, answer in capital_pairs:
            e_ids = self.tokenizer.encode(entity, add_special_tokens=False)
            a_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            
            e_embed = self.static.drum[e_ids[0]]
            a_embed = self.static.drum[a_ids[0]]
            
            # Compute rotation
            e_norm = e_embed / e_embed.norm()
            a_norm = a_embed / a_embed.norm()
            
            cos_angle = (e_norm @ a_norm).clamp(-1, 1)
            angle = torch.acos(cos_angle) * 180 / np.pi
            angles.append(angle.item())
            
            # Compute axis (orthogonal component)
            a_orth = a_norm - (a_norm @ e_norm) * e_norm
            if a_orth.norm() > 1e-6:
                a_orth = a_orth / a_orth.norm()
                axes.append(a_orth)
        
        # Store ideal and angle
        self.static.ideals["capital-of"] = torch.stack(axes).mean(dim=0)
        self.static.angles["capital-of"] = np.mean(angles)
        
        print(f"Learned capital-of: angle={self.static.angles['capital-of']:.1f}°")
    
    def encode_input(self, text: str) -> torch.Tensor:
        """
        Encode input text into a geometric shape.
        
        This is NOT token-by-token encoding.
        It's encoding the ENTIRE input as a shape.
        """
        input_ids = self.tokenizer.encode(text, return_tensors='pt')[0]
        
        # Get embeddings for all tokens
        embeds = self.static.drum[input_ids]
        
        # The "shape" is the trajectory through embedding space
        # For now, use the mean as a simple representation
        shape = embeds.mean(dim=0)
        
        # Update living geometry
        self.living.position = shape
        self.living.context_shape = embeds
        
        return shape
    
    def detect_relationship(self, text: str) -> Optional[str]:
        """Detect which relationship is being asked about."""
        text_lower = text.lower()
        
        if "capital" in text_lower:
            return "capital-of"
        elif "opposite" in text_lower:
            return "opposite-of"
        
        return None
    
    def extract_entity(self, text: str, relationship: str) -> Optional[torch.Tensor]:
        """Extract the entity embedding from the input."""
        # Simple extraction - find the key entity
        if relationship == "capital-of":
            # Look for country names
            countries = ["France", "Germany", "Italy", "Spain", "Japan", "China"]
            for country in countries:
                if country in text:
                    ids = self.tokenizer.encode(country, add_special_tokens=False)
                    return self.static.drum[ids[0]]
        
        return None
    
    def rotate_toward_ideal(self, entity: torch.Tensor, relationship: str) -> torch.Tensor:
        """
        Rotate entity toward the Platonic Ideal for this relationship.
        
        This is the CORE operation - no autoregression needed.
        """
        ideal = self.static.ideals[relationship]
        angle_deg = self.static.angles[relationship]
        angle_rad = angle_deg * np.pi / 180
        
        # Normalize entity
        e_norm = entity / entity.norm()
        
        # Compute axis (direction toward ideal, orthogonal to entity)
        axis = ideal - (ideal @ e_norm) * e_norm
        if axis.norm() > 1e-6:
            axis = axis / axis.norm()
        else:
            axis = ideal
        
        # Apply rotation: answer = cos(θ)*entity + sin(θ)*axis*|entity|
        answer = np.cos(angle_rad) * entity + np.sin(angle_rad) * axis * entity.norm()
        
        return answer
    
    def decode_output(self, shape: torch.Tensor) -> str:
        """
        Decode a geometric shape back to text.
        
        This finds the nearest token(s) to the shape.
        """
        # Find nearest token
        sims = F.cosine_similarity(shape.unsqueeze(0), self.static.drum)
        top_idx = sims.argmax()
        
        return self.tokenizer.decode([top_idx])
    
    def generate_geometric(self, text: str) -> Tuple[str, str]:
        """
        Generate response using geometry, not autoregression.
        
        Returns: (answer, method)
        """
        # Step 1: Detect relationship
        relationship = self.detect_relationship(text)
        if relationship is None:
            return "", "unknown_relationship"
        
        # Step 2: Extract entity
        entity = self.extract_entity(text, relationship)
        if entity is None:
            return "", "unknown_entity"
        
        # Step 3: Rotate toward ideal (THE KEY STEP)
        answer_shape = self.rotate_toward_ideal(entity, relationship)
        
        # Step 4: Decode to token
        answer = self.decode_output(answer_shape)
        
        return answer, "geometric"
    
    def generate_autoregressive(self, text: str) -> str:
        """Traditional autoregressive generation for comparison."""
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token = outputs.logits[0, -1, :].argmax()
        
        return self.tokenizer.decode([next_token])


def compare_methods(model, tokenizer):
    """Compare geometric vs autoregressive generation."""
    print("\n" + "=" * 70)
    print("Geometric vs Autoregressive Generation")
    print("=" * 70)
    
    gsm = GeometricStateMachine(model, tokenizer)
    
    test_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    print("\n--- Comparison ---")
    print(f"{'Prompt':<30} | {'Geometric':<15} | {'Autoregressive':<15}")
    print("-" * 65)
    
    for prompt in test_prompts:
        geo_answer, method = gsm.generate_geometric(prompt)
        auto_answer = gsm.generate_autoregressive(prompt)
        
        print(f"{prompt:<30} | {geo_answer:<15} | {auto_answer:<15}")


def explore_multi_token_output(model, tokenizer):
    """
    Explore if we can generate multiple tokens at once.
    
    The hypothesis: if the output is a SHAPE, it might
    contain multiple tokens implicitly.
    """
    print("\n" + "=" * 70)
    print("Multi-Token Output Exploration")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Get the hidden state for a prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
    
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
        h_final = outputs.hidden_states[-1][0, -1, :]
    
    # The hidden state is a SHAPE
    # Can we decode multiple tokens from it?
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Hidden state shape: {h_final.shape}")
    
    # Method 1: Top-k tokens from hidden state
    print("\n--- Top-k Tokens from Hidden State ---")
    
    lm_head = model.lm_head.weight.data
    logits = h_final @ lm_head.T
    
    top_k = 10
    top_indices = logits.argsort(descending=True)[:top_k]
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    
    print(f"Top {top_k} tokens: {top_tokens}")
    
    # Method 2: Decompose hidden state into components
    print("\n--- Hidden State Decomposition ---")
    
    # SVD of hidden state (treating it as a 1D signal)
    # Reshape to 2D for SVD
    h_2d = h_final.view(56, 64)  # 3584 = 56 * 64
    U, S, Vt = torch.linalg.svd(h_2d, full_matrices=False)
    
    print(f"Top 5 singular values: {S[:5].tolist()}")
    
    # Each singular value might correspond to a "token slot"
    # Reconstruct with different numbers of components
    for k in [1, 5, 10]:
        h_k = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).flatten()
        
        logits_k = h_k @ lm_head.T
        top_idx = logits_k.argmax()
        top_token = tokenizer.decode([top_idx])
        
        print(f"  k={k}: top token = {top_token!r}")


def explore_shape_contains_sequence(model, tokenizer):
    """
    Explore if a single shape can contain a sequence.
    
    Hypothesis: The hidden state at position -1 might encode
    not just the next token, but the entire continuation.
    """
    print("\n" + "=" * 70)
    print("Does Shape Contain Sequence?")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Generate a sequence autoregressively
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    generated_tokens = []
    hidden_states_at_each_step = []
    
    for i in range(5):  # Generate 5 tokens
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            hidden_states_at_each_step.append(h)
            
            next_token = outputs.logits[0, -1, :].argmax()
            generated_tokens.append(next_token.item())
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(generated_tokens)
    print(f"\nGenerated: {prompt}{generated_text}")
    
    # Now check: does the FIRST hidden state contain info about ALL tokens?
    print("\n--- First Hidden State vs All Generated Tokens ---")
    
    h_first = hidden_states_at_each_step[0]
    
    for i, token_id in enumerate(generated_tokens):
        token_embed = embed[token_id]
        
        # Similarity between first hidden state and each generated token
        sim = F.cosine_similarity(h_first.unsqueeze(0), token_embed.unsqueeze(0)).item()
        
        token_text = tokenizer.decode([token_id])
        print(f"  Token {i+1} ({token_text!r}): similarity to h_first = {sim:.4f}")
    
    # Check if hidden states are similar across generation steps
    print("\n--- Hidden State Similarity Across Steps ---")
    
    for i in range(len(hidden_states_at_each_step)):
        for j in range(i+1, len(hidden_states_at_each_step)):
            sim = F.cosine_similarity(
                hidden_states_at_each_step[i].unsqueeze(0),
                hidden_states_at_each_step[j].unsqueeze(0)
            ).item()
            print(f"  h[{i}] vs h[{j}]: {sim:.4f}")


def synthesize_findings():
    """Synthesize findings about static + living geometry."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Static + Living Geometry")
    print("=" * 70)
    print("""
Key Insight:

If memory = computation = rotation toward Platonic Ideals,
then we have TWO geometric states:

1. STATIC GEOMETRY (Startup State)
   - Platonic Ideals (fixed points in semantic space)
   - Relationship angles (77° for capital-of, etc.)
   - DRUM (token embeddings)
   - Pattern templates

   This is LOADED ONCE and NEVER CHANGES.

2. LIVING GEOMETRY (Memory State)
   - Current position in semantic space
   - Accumulated rotations from context
   - Active ideals (which relationships are primed)
   - Conversation shape

   This EVOLVES but maintains geometric structure.

WHY NO AUTOREGRESSION?
======================

Autoregression: Token → Forward pass → Token → Forward pass → ...
Geometric:      Input shape → Rotate → Output shape (all at once)

The key insight: The OUTPUT is the ENDPOINT of a rotation.
If we know:
  - Starting position (entity)
  - Target ideal (relationship)
  - Rotation angle (relationship strength)

Then the answer is DETERMINED GEOMETRICALLY.
No need to generate token-by-token.

CHALLENGES:
===========

1. Multi-token outputs: How does one rotation produce multiple tokens?
   - Hypothesis: The shape CONTAINS the sequence
   - The hidden state encodes not just next token, but continuation

2. Variable-length outputs: How do we know when to stop?
   - Hypothesis: The rotation has a natural endpoint
   - When we reach the ideal, generation stops

3. Complex relationships: What about multi-step reasoning?
   - Hypothesis: Multiple rotations in sequence
   - Each rotation moves toward a different ideal

IMPLICATIONS:
=============

If this works, we can:
1. Generate entire responses in ONE geometric operation
2. No KV cache needed (no autoregression)
3. Constant time generation (not O(n) in output length)
4. Memory IS the geometry (no separate storage)

This would be a fundamental shift from:
  "Generate token by token"
to:
  "Rotate to the answer"
""")


def main():
    print("=" * 70)
    print("Static + Living Geometry: Eliminating Autoregression")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Compare geometric vs autoregressive
    compare_methods(model, tokenizer)
    
    # Explore multi-token output
    explore_multi_token_output(model, tokenizer)
    
    # Explore if shape contains sequence
    explore_shape_contains_sequence(model, tokenizer)
    
    # Synthesis
    synthesize_findings()


if __name__ == "__main__":
    main()
