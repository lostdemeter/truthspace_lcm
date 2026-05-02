#!/usr/bin/env python3
"""
Unnamed Concept Space
======================

Work in pure geometric space first, map to language later.

The insight: The model doesn't think in words - it thinks in POSITIONS.
We should do the same:

1. Token space → Unnamed concept space (positions, not words)
2. Navigate in concept space to produce geometric results
3. Map back to language only when needed

This is the "unnamed compound" problem from Doc 115:
- Some positions have words (cottage, mansion)
- Some positions DON'T have words (large + dog = ?)
- We work with positions first, words second

Run with:
    python src/phi_navigator/unnamed_concept_space.py
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class ConceptPosition:
    """A position in concept space - may or may not have a name."""
    signs: torch.Tensor  # The geometric position (sign pattern)
    names: List[str] = field(default_factory=list)  # Words that map to this position (may be empty)
    
    def __repr__(self):
        if self.names:
            return f"Concept({self.names[0]})"
        else:
            return f"Concept(unnamed, hash={hash(tuple(self.signs.tolist())) % 10000})"


class UnnamedConceptSpace:
    """
    A geometric space where we work with positions first, words second.
    
    Key insight: Not every position needs a name. We can:
    1. Navigate to a position
    2. Check if that position has a name
    3. If not, describe it geometrically or find nearest named position
    """
    
    def __init__(self, embeddings: torch.Tensor, tokenizer):
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.hidden_dim = embeddings.shape[1]
        self.vocab_size = embeddings.shape[0]
        
        # All token signs (the raw geometric space)
        self.token_signs = torch.sign(embeddings).to(torch.int8)
        self.token_signs[self.token_signs == 0] = 1
        
        # Named positions (word -> position)
        self.named_positions: Dict[str, ConceptPosition] = {}
        
        # All positions we've discovered (including unnamed)
        self.all_positions: List[ConceptPosition] = []
        
        # Transformation patterns (dimension_name -> flip pattern)
        self.transformations: Dict[str, torch.Tensor] = {}
        
        # Input-output pairs for learning transformations
        self.pairs: List[Tuple[ConceptPosition, ConceptPosition, str]] = []
        
        logger.info(f"UnnamedConceptSpace: {self.vocab_size} tokens, {self.hidden_dim} dims")
    
    def _get_token_signs(self, token_id: int) -> torch.Tensor:
        """Get sign pattern for a single token."""
        return self.token_signs[token_id]
    
    def _phrase_to_position(self, phrase: str) -> ConceptPosition:
        """Convert a phrase to a concept position (averaging tokens)."""
        tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
        if len(tokens) == 0:
            return ConceptPosition(signs=torch.zeros(self.hidden_dim, dtype=torch.int8))
        
        # Average embeddings, then take signs
        avg_embed = self.embeddings[tokens].mean(dim=0)
        signs = torch.sign(avg_embed).to(torch.int8)
        signs[signs == 0] = 1
        
        return ConceptPosition(signs=signs, names=[phrase])
    
    def add_named_position(self, phrase: str) -> ConceptPosition:
        """Add a named position to the space."""
        if phrase in self.named_positions:
            return self.named_positions[phrase]
        
        pos = self._phrase_to_position(phrase)
        self.named_positions[phrase] = pos
        self.all_positions.append(pos)
        return pos
    
    def add_pair(self, source: str, target: str, relationship: str):
        """Add an input-output pair. Transformations will EMERGE from pairs."""
        source_pos = self.add_named_position(source)
        target_pos = self.add_named_position(target)
        
        self.pairs.append((source_pos, target_pos, relationship))
        
        # Update transformation pattern for this relationship
        self._update_transformation(relationship)
    
    def _update_transformation(self, relationship: str):
        """Update the transformation pattern based on all pairs of this type."""
        relevant_pairs = [(s, t) for s, t, r in self.pairs if r == relationship]
        
        if not relevant_pairs:
            return
        
        # Compute flip pattern from all pairs
        flip_sum = torch.zeros(self.hidden_dim, dtype=torch.float32)
        
        for source_pos, target_pos in relevant_pairs:
            flip = (source_pos.signs != target_pos.signs).float()
            flip_sum += flip
        
        # Flip pattern: dimensions that flip in >50% of pairs
        flip_prob = flip_sum / len(relevant_pairs)
        self.transformations[relationship] = (flip_prob > 0.5).float()
    
    def navigate(self, source: ConceptPosition, transformation: str) -> ConceptPosition:
        """
        Navigate from source position using a transformation.
        
        Returns a NEW POSITION - which may or may not have a name!
        """
        if transformation not in self.transformations:
            logger.warning(f"Unknown transformation: {transformation}")
            return source
        
        flip_pattern = self.transformations[transformation]
        
        # Apply transformation
        new_signs = source.signs.float().clone()
        new_signs[flip_pattern > 0.5] *= -1
        new_signs = new_signs.to(torch.int8)
        
        # Create new position (unnamed by default)
        new_pos = ConceptPosition(signs=new_signs)
        self.all_positions.append(new_pos)
        
        return new_pos
    
    def find_nearest_named(self, position: ConceptPosition, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the nearest named positions to a given position.
        
        This is how we MAP from concept space back to language.
        """
        results = []
        
        for name, named_pos in self.named_positions.items():
            agreement = (position.signs == named_pos.signs).float().sum().item()
            similarity = agreement / self.hidden_dim * 100
            results.append((name, similarity))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def find_nearest_token(self, position: ConceptPosition, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the nearest tokens in the FULL vocabulary.
        
        This searches ALL 152k tokens, not just named positions.
        """
        # Compute similarity to all tokens
        pos_signs = position.signs.float()
        all_signs = self.token_signs.float()
        
        # Agreement count for each token
        agreements = (all_signs == pos_signs).float().sum(dim=1)
        similarities = agreements / self.hidden_dim * 100
        
        # Get top-k
        top_indices = torch.topk(similarities, top_k).indices
        
        results = []
        for idx in top_indices:
            token_id = idx.item()
            token = self.tokenizer.decode([token_id])
            sim = similarities[token_id].item()
            results.append((token, sim))
        
        return results
    
    def describe_position(self, position: ConceptPosition) -> Dict:
        """
        Describe a position in multiple ways:
        1. Nearest named positions (if any)
        2. Nearest tokens from vocabulary
        3. Geometric description (which dimensions are +/-)
        """
        # Nearest named
        nearest_named = self.find_nearest_named(position, top_k=3)
        
        # Nearest tokens
        nearest_tokens = self.find_nearest_token(position, top_k=5)
        
        # Geometric: count +/- signs
        n_positive = (position.signs > 0).sum().item()
        n_negative = (position.signs < 0).sum().item()
        
        return {
            "has_name": len(position.names) > 0,
            "names": position.names,
            "nearest_named": nearest_named,
            "nearest_tokens": nearest_tokens,
            "geometry": {
                "positive_dims": n_positive,
                "negative_dims": n_negative,
                "ratio": n_positive / (n_positive + n_negative) if (n_positive + n_negative) > 0 else 0.5
            }
        }
    
    def auto_discover_transformations(self) -> Dict[str, int]:
        """
        Automatically discover transformations from the token space.
        
        This is the key to automation: we don't manually define pairs,
        we DISCOVER them from the structure of the embedding space.
        """
        # Strategy: Find tokens that are "close" in sign space but differ
        # in specific dimensions - these define natural transformations
        
        discovered = {}
        
        # Sample random token pairs and analyze their differences
        n_samples = 1000
        indices = torch.randperm(self.vocab_size)[:n_samples]
        
        flip_counts = torch.zeros(self.hidden_dim)
        
        for i in range(0, n_samples - 1, 2):
            idx1, idx2 = indices[i].item(), indices[i+1].item()
            signs1 = self.token_signs[idx1].float()
            signs2 = self.token_signs[idx2].float()
            
            flips = (signs1 != signs2).float()
            flip_counts += flips
        
        # Dimensions that flip frequently are "active" dimensions
        flip_freq = flip_counts / (n_samples // 2)
        
        # Find clusters of frequently-flipping dimensions
        active_dims = (flip_freq > 0.4).sum().item()
        stable_dims = (flip_freq < 0.1).sum().item()
        
        discovered["active_dimensions"] = active_dims
        discovered["stable_dimensions"] = stable_dims
        discovered["flip_frequency_mean"] = flip_freq.mean().item()
        discovered["flip_frequency_std"] = flip_freq.std().item()
        
        return discovered
    
    def get_stats(self) -> Dict:
        """Get statistics about the concept space."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "named_positions": len(self.named_positions),
            "all_positions": len(self.all_positions),
            "transformations": list(self.transformations.keys()),
            "pairs": len(self.pairs),
        }


def discover_natural_clusters(space: UnnamedConceptSpace, n_samples: int = 5000) -> Dict:
    """
    Discover natural clusters in the embedding space using SVD.
    
    This is fully automated - no manual pairs needed!
    """
    logger.info(f"Discovering natural structure from {n_samples} token samples...")
    
    # Sample tokens
    indices = torch.randperm(space.vocab_size)[:n_samples]
    sample_signs = space.token_signs[indices].float()
    
    # SVD to find principal directions
    U, S, Vt = torch.linalg.svd(sample_signs, full_matrices=False)
    
    # The singular values tell us about the structure
    total_var = (S**2).sum()
    cumulative_var = torch.cumsum(S**2, dim=0) / total_var * 100
    
    logger.info(f"  Top singular values:")
    for i in range(min(10, len(S))):
        logger.info(f"    SV[{i}]: {S[i].item():.2f} (cumulative: {cumulative_var[i].item():.1f}%)")
    
    # Project tokens onto top principal components
    n_components = 10
    projections = U[:, :n_components] * S[:n_components]  # [n_samples, n_components]
    
    # Find tokens at extremes of each principal direction
    logger.info(f"\n  Tokens at extremes of principal directions:")
    
    results = []
    for pc in range(min(5, n_components)):
        proj_pc = projections[:, pc]
        
        # Top 5 positive
        top_pos_idx = torch.topk(proj_pc, 5).indices
        top_pos_tokens = [space.tokenizer.decode([indices[i].item()]) for i in top_pos_idx]
        
        # Top 5 negative
        top_neg_idx = torch.topk(-proj_pc, 5).indices
        top_neg_tokens = [space.tokenizer.decode([indices[i].item()]) for i in top_neg_idx]
        
        logger.info(f"    PC{pc}: {top_neg_tokens[:3]} ←→ {top_pos_tokens[:3]}")
        
        results.append({
            "pc": pc,
            "positive_tokens": top_pos_tokens,
            "negative_tokens": top_neg_tokens,
            "singular_value": S[pc].item(),
        })
    
    # The principal directions ARE the emergent dimensions!
    # Store them as transformations
    for i in range(min(5, n_components)):
        # The principal direction defines which dimensions to flip
        direction = Vt[i]  # [hidden_dim]
        flip_pattern = (direction.abs() > direction.abs().mean()).float()
        space.transformations[f"auto_pc{i}"] = flip_pattern
    
    logger.info(f"\n  Stored {min(5, n_components)} auto-discovered transformations (PC0-PC4)")
    
    return {"principal_components": results, "singular_values": S[:10].tolist()}


def auto_discover_response_pattern(space: UnnamedConceptSpace) -> torch.Tensor:
    """
    Automatically discover what makes a "response" different from an "input".
    
    Strategy: Compare tokens that look like inputs vs tokens that look like outputs.
    """
    logger.info("Auto-discovering response pattern...")
    
    # Heuristic: Tokens starting with capital letters are more likely responses
    # Tokens starting with lowercase are more likely inputs
    
    input_tokens = []
    output_tokens = []
    
    for token_id in range(min(50000, space.vocab_size)):  # Sample first 50k tokens
        token = space.tokenizer.decode([token_id])
        
        if len(token) < 2:
            continue
        
        # Skip special tokens
        if token.startswith('<') or token.startswith('['):
            continue
        
        # Classify by first character
        first_char = token.lstrip()[0] if token.lstrip() else ''
        
        if first_char.isupper():
            output_tokens.append(token_id)
        elif first_char.islower():
            input_tokens.append(token_id)
    
    logger.info(f"  Found {len(input_tokens)} input-like tokens, {len(output_tokens)} output-like tokens")
    
    if len(input_tokens) < 100 or len(output_tokens) < 100:
        logger.warning("Not enough tokens for pattern discovery")
        return None
    
    # Sample and compute average signs
    input_sample = input_tokens[:1000]
    output_sample = output_tokens[:1000]
    
    input_avg = space.token_signs[input_sample].float().mean(dim=0)
    output_avg = space.token_signs[output_sample].float().mean(dim=0)
    
    # The difference reveals the "response transformation"
    diff = output_avg - input_avg
    
    # Dimensions where output is consistently different from input
    response_pattern = (diff.abs() > 0.3).float()  # Threshold for significance
    
    n_significant = (response_pattern > 0).sum().item()
    logger.info(f"  Response pattern: {n_significant} significant dimensions ({n_significant/space.hidden_dim*100:.1f}%)")
    
    return response_pattern


def discover_semantic_transformations(space: UnnamedConceptSpace) -> Dict:
    """
    Discover semantic transformations by analyzing the embedding structure.
    
    Key insight: Sign patterns encode semantic relationships!
    - was ↔ were (verb tense)
    - he ↔ she (gender)
    - is ↔ are (singular/plural)
    """
    logger.info("Discovering semantic transformations from embedding structure...")
    
    # Filter to English-like tokens (ASCII, reasonable length)
    english_tokens = []
    for token_id in range(min(50000, space.vocab_size)):
        token = space.tokenizer.decode([token_id]).strip()
        if len(token) >= 2 and len(token) <= 10 and token.isalpha() and token.isascii():
            english_tokens.append(token_id)
    
    logger.info(f"  Found {len(english_tokens)} English-like tokens")
    
    if len(english_tokens) < 1000:
        return {}
    
    # Sample tokens
    sample_size = min(2000, len(english_tokens))
    sample_ids = english_tokens[:sample_size]
    sample_signs = space.token_signs[sample_ids].float()
    
    # Compute pairwise AGREEMENT (not dot product)
    # Agreement = (signs_i == signs_j).sum() / hidden_dim
    # For +1/-1 signs: agreement = (1 + dot_product) / 2
    dot_products = sample_signs @ sample_signs.T / space.hidden_dim
    agreements = (1 + dot_products) / 2  # Convert to 0-1 range
    
    # Find pairs with high agreement (> 60% means they share structure)
    interesting_pairs = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            agreement = agreements[i, j].item()
            if 0.60 < agreement < 0.80:  # Similar but different
                token_i = space.tokenizer.decode([sample_ids[i]]).strip()
                token_j = space.tokenizer.decode([sample_ids[j]]).strip()
                
                # Skip if same token (with different casing)
                if token_i.lower() == token_j.lower():
                    continue
                
                # Compute flip pattern
                signs_i = sample_signs[i]
                signs_j = sample_signs[j]
                flip = (signs_i != signs_j).float()
                n_flips = flip.sum().item()
                
                interesting_pairs.append({
                    "token1": token_i,
                    "token2": token_j,
                    "agreement": agreement,
                    "n_flips": int(n_flips),
                    "flip_pattern": flip,
                })
    
    # Sort by agreement (higher = more interesting)
    interesting_pairs.sort(key=lambda x: -x["agreement"])
    
    logger.info(f"  Found {len(interesting_pairs)} interesting pairs (60-80% agreement)")
    
    # Show top pairs
    logger.info("  Top semantic pair candidates:")
    for pair in interesting_pairs[:30]:
        logger.info(f"    {pair['token1']:15s} ↔ {pair['token2']:15s} (agreement={pair['agreement']:.1%}, flips={pair['n_flips']})")
    
    # Extract transformations from discovered pairs
    # Group pairs by their flip patterns to find semantic dimensions
    if interesting_pairs:
        # Use SVD on flip patterns to find common transformations
        flip_matrix = torch.stack([p["flip_pattern"] for p in interesting_pairs[:50]])
        U, S, Vt = torch.linalg.svd(flip_matrix)
        
        logger.info(f"\n  Transformation structure (SVD on flip patterns):")
        total_var = (S**2).sum()
        for i in range(min(5, len(S))):
            var_pct = (S[i]**2 / total_var * 100).item()
            logger.info(f"    Component {i}: {var_pct:.1f}% variance")
        
        # Store top transformations
        for i in range(min(3, len(S))):
            direction = Vt[i]
            flip_pattern = (direction.abs() > direction.abs().mean()).float()
            space.transformations[f"semantic_{i}"] = flip_pattern
            n_flips = (flip_pattern > 0.5).sum().item()
            logger.info(f"    Stored semantic_{i}: {n_flips} flip dimensions")
    
    return {"pairs": interesting_pairs[:100]}


def demo_unnamed_concept_space():
    """
    Demo: Working in unnamed concept space.
    
    1. Navigate to positions (not words)
    2. Check if positions have names
    3. Map back to language when needed
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info("Loading model for embeddings...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    embeddings = model.model.embed_tokens.weight.detach().float().cpu()
    
    del model
    torch.cuda.empty_cache()
    
    # Create unnamed concept space
    space = UnnamedConceptSpace(embeddings, tokenizer)
    
    # =========================================================================
    # PHASE 1: Auto-discover structure from token space
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Auto-discovering structure from token space")
    logger.info("="*60)
    
    discovered = space.auto_discover_transformations()
    logger.info(f"  Active dimensions: {discovered['active_dimensions']}")
    logger.info(f"  Stable dimensions: {discovered['stable_dimensions']}")
    logger.info(f"  Flip frequency: {discovered['flip_frequency_mean']:.3f} ± {discovered['flip_frequency_std']:.3f}")
    
    # =========================================================================
    # PHASE 2: Add some pairs to learn transformations
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Learning transformations from pairs")
    logger.info("="*60)
    
    # Greeting transformation
    greeting_pairs = [
        ("hello", "Hello!"),
        ("hi", "Hi there!"),
        ("hey", "Hey!"),
    ]
    for inp, out in greeting_pairs:
        space.add_pair(inp, out, "greeting_response")
    
    # Question transformation
    qa_pairs = [
        ("what is", "It is"),
        ("who is", "That is"),
        ("how do", "You can"),
    ]
    for inp, out in qa_pairs:
        space.add_pair(inp, out, "question_answer")
    
    logger.info(f"  Learned {len(space.transformations)} transformations")
    for name, pattern in space.transformations.items():
        n_flips = (pattern > 0.5).sum().item()
        logger.info(f"    {name}: {n_flips} flip dimensions ({n_flips/space.hidden_dim*100:.1f}%)")
    
    # =========================================================================
    # PHASE 3: Navigate to UNNAMED positions
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Navigating to unnamed positions")
    logger.info("="*60)
    
    # Start with a known position
    start_phrase = "hello"
    start_pos = space.add_named_position(start_phrase)
    logger.info(f"\nStarting position: {start_phrase}")
    
    # Navigate using greeting_response transformation
    result_pos = space.navigate(start_pos, "greeting_response")
    
    logger.info(f"\nAfter greeting_response transformation:")
    description = space.describe_position(result_pos)
    
    logger.info(f"  Has name: {description['has_name']}")
    logger.info(f"  Nearest named positions:")
    for name, sim in description['nearest_named']:
        logger.info(f"    {name}: {sim:.1f}%")
    
    logger.info(f"  Nearest tokens from vocabulary:")
    for token, sim in description['nearest_tokens']:
        logger.info(f"    '{token}': {sim:.1f}%")
    
    # =========================================================================
    # PHASE 4: Test with unknown inputs
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Testing with unknown inputs")
    logger.info("="*60)
    
    unknown_inputs = ["howdy", "greetings", "yo", "sup"]
    
    for inp in unknown_inputs:
        # Create position for unknown input
        inp_pos = space._phrase_to_position(inp)
        
        # Navigate
        result_pos = space.navigate(inp_pos, "greeting_response")
        
        # Describe result
        desc = space.describe_position(result_pos)
        
        # Get best match
        if desc['nearest_named']:
            best_name, best_sim = desc['nearest_named'][0]
        else:
            best_name, best_sim = desc['nearest_tokens'][0]
        
        logger.info(f"  {inp:15s} → {best_name:20s} ({best_sim:.1f}%)")
    
    # =========================================================================
    # PHASE 5: Explore the FULL vocabulary
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 5: Exploring full vocabulary for response patterns")
    logger.info("="*60)
    
    # Find tokens that are natural "responses" (high agreement with response positions)
    response_positions = [space.named_positions[name] for name in ["Hello!", "Hi there!", "Hey!"]]
    
    # Average response position
    avg_response_signs = torch.zeros(space.hidden_dim)
    for pos in response_positions:
        avg_response_signs += pos.signs.float()
    avg_response_signs = torch.sign(avg_response_signs).to(torch.int8)
    avg_response_signs[avg_response_signs == 0] = 1
    
    avg_response_pos = ConceptPosition(signs=avg_response_signs)
    
    logger.info("Tokens most similar to 'average greeting response':")
    nearest = space.find_nearest_token(avg_response_pos, top_k=20)
    for token, sim in nearest:
        # Filter out weird tokens
        if token.strip() and len(token) > 1:
            logger.info(f"  '{token}': {sim:.1f}%")
    
    # =========================================================================
    # PHASE 6: Auto-discover response pattern (NO MANUAL PAIRS)
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 6: Auto-discovering response pattern")
    logger.info("="*60)
    
    response_pattern = auto_discover_response_pattern(space)
    
    if response_pattern is not None:
        # Store as an auto-discovered transformation
        space.transformations["auto_response"] = response_pattern
        
        # Test it on some inputs
        logger.info("\nTesting auto-discovered response transformation:")
        test_inputs = ["hello", "what", "help", "thanks", "bye"]
        
        for inp in test_inputs:
            inp_pos = space._phrase_to_position(inp)
            
            # Apply auto-discovered transformation
            new_signs = inp_pos.signs.float().clone()
            new_signs[response_pattern > 0.5] *= -1
            new_signs = new_signs.to(torch.int8)
            
            result_pos = ConceptPosition(signs=new_signs)
            
            # Find nearest tokens
            nearest = space.find_nearest_token(result_pos, top_k=3)
            tokens_str = ", ".join([f"'{t}'" for t, s in nearest])
            logger.info(f"  {inp:10s} → {tokens_str}")
    
    # =========================================================================
    # PHASE 7: Discover natural clusters
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 7: Discovering natural clusters in token space")
    logger.info("="*60)
    
    cluster_results = discover_natural_clusters(space, n_samples=10000)
    
    # =========================================================================
    # PHASE 8: Discover semantic transformations automatically
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 8: Discovering semantic transformations")
    logger.info("="*60)
    
    semantic_results = discover_semantic_transformations(space)
    
    return space


if __name__ == "__main__":
    space = demo_unnamed_concept_space()
