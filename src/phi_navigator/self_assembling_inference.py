#!/usr/bin/env python3
"""
Self-Assembling Inference Engine
=================================

Based on Docs 114-115: Dimensions EMERGE from transformation pairs.

The hypothesis: If we feed enough input-output pairs, response dimensions
will emerge naturally - just like semantic dimensions emerged from word pairs.

Key principles:
1. Pairs are the source of truth
2. Dimensions emerge from pairs (not predefined)
3. Platonic Ideals anchor multiple dimensions
4. Navigation follows emergent structure

Run with:
    python src/phi_navigator/self_assembling_inference.py
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class TransformationPair:
    """A single input-output pair that defines a transformation."""
    source: str  # Input (e.g., "hello")
    target: str  # Output (e.g., "Hello!")
    relationship: str  # Type of transformation (e.g., "greeting_response")
    
    def __hash__(self):
        return hash((self.source, self.target, self.relationship))


@dataclass
class EmergentDimension:
    """A dimension that emerged from transformation pairs."""
    name: str
    index: int
    source_pairs: List[TransformationPair] = field(default_factory=list)
    pole_negative: List[str] = field(default_factory=list)  # Source words
    pole_positive: List[str] = field(default_factory=list)  # Target words
    
    def __repr__(self):
        return f"Dimension({self.name}, {len(self.source_pairs)} pairs)"


@dataclass
class PlatonicIdeal:
    """A concept that anchors multiple dimensions."""
    word: str
    dimensions_anchored: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    
    def __repr__(self):
        return f"Ideal({self.word}, anchors={list(self.dimensions_anchored)})"


class SelfAssemblingSpace:
    """
    A geometric space where dimensions emerge from transformation pairs.
    
    Based on Doc 115: Self-Assembling Corpus Roadmap
    """
    
    def __init__(self, embeddings: torch.Tensor, tokenizer):
        self.embeddings = embeddings  # [vocab, hidden_dim]
        self.tokenizer = tokenizer
        self.hidden_dim = embeddings.shape[1]
        
        # Source of truth: transformation pairs
        self.pairs: List[TransformationPair] = []
        
        # Emergent structure
        self.dimensions: Dict[str, EmergentDimension] = {}
        self.positions: Dict[str, torch.Tensor] = {}  # word -> n-dim position
        self.ideals: Dict[str, PlatonicIdeal] = {}
        
        # Sign patterns for navigation
        self.concept_signs: Dict[str, torch.Tensor] = {}
        
        # Flip patterns per dimension (emergent)
        self.flip_patterns: Dict[str, torch.Tensor] = {}
        
        logger.info(f"SelfAssemblingSpace initialized with {embeddings.shape[0]} embeddings")
    
    def _get_phrase_embedding(self, phrase: str) -> torch.Tensor:
        """Get embedding for a phrase (average of tokens)."""
        tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
        if len(tokens) == 0:
            return torch.zeros(self.hidden_dim)
        return self.embeddings[tokens].mean(dim=0)
    
    def _get_phrase_signs(self, phrase: str) -> torch.Tensor:
        """Get sign pattern for a phrase."""
        emb = self._get_phrase_embedding(phrase)
        signs = torch.sign(emb).to(torch.int8)
        signs[signs == 0] = 1
        return signs
    
    def add_pair(self, source: str, target: str, relationship: str):
        """
        Add a transformation pair. This is the ONLY way to add knowledge.
        
        Dimensions emerge from pairs - we don't predefine them.
        """
        pair = TransformationPair(source, target, relationship)
        self.pairs.append(pair)
        
        # Store sign patterns
        if source not in self.concept_signs:
            self.concept_signs[source] = self._get_phrase_signs(source)
        if target not in self.concept_signs:
            self.concept_signs[target] = self._get_phrase_signs(target)
        
        # Check if this relationship type is new
        if relationship not in self.dimensions:
            self._create_dimension(relationship)
        
        # Add to existing dimension
        dim = self.dimensions[relationship]
        dim.source_pairs.append(pair)
        if source not in dim.pole_negative:
            dim.pole_negative.append(source)
        if target not in dim.pole_positive:
            dim.pole_positive.append(target)
        
        # Update flip pattern for this dimension
        self._update_flip_pattern(relationship)
    
    def _create_dimension(self, name: str):
        """Create a new emergent dimension."""
        index = len(self.dimensions)
        dim = EmergentDimension(name=name, index=index)
        self.dimensions[name] = dim
        logger.info(f"NEW DIMENSION EMERGED: {name} (index={index})")
    
    def _update_flip_pattern(self, dimension_name: str):
        """Update the flip pattern for a dimension based on its pairs."""
        dim = self.dimensions[dimension_name]
        
        if len(dim.source_pairs) == 0:
            return
        
        # Compute flip pattern from all pairs in this dimension
        flip_sum = torch.zeros(self.hidden_dim, dtype=torch.float32)
        
        for pair in dim.source_pairs:
            source_signs = self.concept_signs[pair.source].float()
            target_signs = self.concept_signs[pair.target].float()
            flip = (source_signs != target_signs).float()
            flip_sum += flip
        
        # Flip pattern: dimensions that flip in >50% of pairs
        flip_prob = flip_sum / len(dim.source_pairs)
        self.flip_patterns[dimension_name] = (flip_prob > 0.5).float()
    
    def discover_dimensions(self) -> int:
        """
        Discover emergent dimensions from pairs.
        
        Returns the number of dimensions discovered.
        """
        # Dimensions are already created when pairs are added
        # This method can be used for additional analysis
        
        logger.info(f"Discovered {len(self.dimensions)} dimensions:")
        for name, dim in self.dimensions.items():
            logger.info(f"  {name}: {len(dim.source_pairs)} pairs")
            logger.info(f"    Sources: {dim.pole_negative[:5]}...")
            logger.info(f"    Targets: {dim.pole_positive[:5]}...")
        
        return len(self.dimensions)
    
    def discover_ideals(self) -> List[PlatonicIdeal]:
        """
        Discover Platonic Ideals - concepts that anchor multiple dimensions.
        """
        # Count how many dimensions each word anchors (as source)
        anchor_counts: Dict[str, Set[str]] = defaultdict(set)
        
        for pair in self.pairs:
            anchor_counts[pair.source].add(pair.relationship)
        
        # Words anchoring 2+ dimensions are candidate ideals
        ideals = []
        for word, dims in anchor_counts.items():
            if len(dims) >= 2:
                ideal = PlatonicIdeal(
                    word=word,
                    dimensions_anchored=dims,
                    confidence=len(dims) / len(self.dimensions) if self.dimensions else 0
                )
                self.ideals[word] = ideal
                ideals.append(ideal)
        
        logger.info(f"Discovered {len(ideals)} Platonic Ideals:")
        for ideal in ideals:
            logger.info(f"  {ideal}")
        
        return ideals
    
    def navigate(self, source: str, dimension: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Navigate from source to target using emergent structure.
        
        If dimension is specified, use that dimension's flip pattern.
        If not, detect the best dimension based on input similarity.
        
        Returns: (result, score, dimension_used)
        """
        if source not in self.concept_signs:
            self.concept_signs[source] = self._get_phrase_signs(source)
        
        source_signs = self.concept_signs[source].float()
        
        # If no dimension specified, detect the best one
        if dimension is None:
            dimension = self._detect_dimension(source)
        
        if dimension and dimension in self.flip_patterns:
            # Use detected/specified dimension
            flip_pattern = self.flip_patterns[dimension]
            target_signs = source_signs.clone()
            target_signs[flip_pattern > 0.5] *= -1
            
            # Find nearest in RESPONSE space (targets only)
            dim = self.dimensions[dimension]
            best_word, best_score = self._find_nearest_in_set(
                target_signs, 
                candidates=set(dim.pole_positive),  # Only search responses
                exclude={source}
            )
            return best_word, best_score, dimension
        
        return None, 0.0, None
    
    def _detect_dimension(self, source: str) -> Optional[str]:
        """
        Detect which dimension the source belongs to.
        
        Uses similarity to known sources (pole_negative) of each dimension.
        """
        source_signs = self.concept_signs.get(source)
        if source_signs is None:
            source_signs = self._get_phrase_signs(source)
        
        best_dim = None
        best_similarity = -float('inf')
        
        for dim_name, dim in self.dimensions.items():
            # Compute average similarity to known sources
            similarities = []
            for known_source in dim.pole_negative:
                if known_source in self.concept_signs:
                    known_signs = self.concept_signs[known_source]
                    sim = (source_signs == known_signs).float().sum().item()
                    similarities.append(sim)
            
            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_dim = dim_name
        
        return best_dim
    
    def _find_nearest_in_set(self, target_signs: torch.Tensor, candidates: Set[str], exclude: Set[str] = None) -> Tuple[str, float]:
        """Find the nearest concept to target_signs within a specific set."""
        if exclude is None:
            exclude = set()
        
        best_word = None
        best_score = -float('inf')
        
        for word in candidates:
            if word in exclude:
                continue
            
            if word not in self.concept_signs:
                self.concept_signs[word] = self._get_phrase_signs(word)
            
            signs = self.concept_signs[word]
            agreement = (signs.float() == target_signs).float().sum().item()
            if agreement > best_score:
                best_score = agreement
                best_word = word
        
        return best_word, best_score / self.hidden_dim * 100
    
    def _find_nearest(self, target_signs: torch.Tensor, exclude: Set[str] = None) -> Tuple[str, float]:
        """Find the nearest concept to target_signs."""
        if exclude is None:
            exclude = set()
        
        best_word = None
        best_score = -float('inf')
        
        for word, signs in self.concept_signs.items():
            if word in exclude:
                continue
            
            agreement = (signs.float() == target_signs).float().sum().item()
            if agreement > best_score:
                best_score = agreement
                best_word = word
        
        return best_word, best_score / self.hidden_dim * 100
    
    def get_stats(self) -> Dict:
        """Get statistics about the self-assembled space."""
        return {
            "pairs": len(self.pairs),
            "dimensions": len(self.dimensions),
            "concepts": len(self.concept_signs),
            "ideals": len(self.ideals),
            "dimension_names": list(self.dimensions.keys()),
        }


def demo_self_assembling_inference():
    """
    Demo: Self-assembling inference for chat responses.
    
    We feed input-output pairs and let response dimensions EMERGE.
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
    
    # Create self-assembling space
    space = SelfAssemblingSpace(embeddings, tokenizer)
    
    # =========================================================================
    # PHASE 1: Feed input-output pairs
    # Dimensions will EMERGE from these pairs
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Feeding input-output pairs")
    logger.info("="*60)
    
    # Greeting pairs
    greeting_pairs = [
        ("hello", "Hello!"),
        ("hi", "Hi there!"),
        ("hey", "Hey!"),
        ("good morning", "Good morning!"),
        ("greetings", "Greetings!"),
        ("howdy", "Howdy!"),
        ("yo", "Hey!"),
    ]
    
    for inp, out in greeting_pairs:
        space.add_pair(inp, out, "greeting_response")
    
    # Question-answer pairs
    qa_pairs = [
        ("what is your name", "My name is Assistant"),
        ("who are you", "I am an AI assistant"),
        ("what can you do", "I can help with many tasks"),
        ("how are you", "I am doing well"),
        ("what time is it", "I don't have access to the current time"),
    ]
    
    for q, a in qa_pairs:
        space.add_pair(q, a, "question_answer")
    
    # Command-acknowledgment pairs
    command_pairs = [
        ("help me", "I'd be happy to help"),
        ("do this", "I'll do that for you"),
        ("please explain", "Let me explain"),
        ("tell me about", "Here's what I know about"),
        ("show me", "Here's what you asked for"),
    ]
    
    for cmd, ack in command_pairs:
        space.add_pair(cmd, ack, "command_response")
    
    # Farewell pairs
    farewell_pairs = [
        ("goodbye", "Goodbye!"),
        ("bye", "Bye!"),
        ("see you", "See you later!"),
        ("take care", "Take care!"),
        ("later", "See you later!"),
        ("gotta go", "Goodbye!"),
    ]
    
    for inp, out in farewell_pairs:
        space.add_pair(inp, out, "farewell_response")
    
    # Gratitude pairs (NEW DIMENSION WILL EMERGE)
    gratitude_pairs = [
        ("thanks", "You're welcome!"),
        ("thank you", "You're welcome!"),
        ("appreciate it", "Happy to help!"),
        ("that helps", "Glad I could help!"),
    ]
    
    for inp, out in gratitude_pairs:
        space.add_pair(inp, out, "gratitude_response")
    
    # Affirmation pairs (NEW DIMENSION)
    affirmation_pairs = [
        ("yes", "Great!"),
        ("okay", "Sounds good!"),
        ("sure", "Perfect!"),
        ("alright", "Excellent!"),
        ("got it", "Understood!"),
    ]
    
    for inp, out in affirmation_pairs:
        space.add_pair(inp, out, "affirmation_response")
    
    # Negation pairs (NEW DIMENSION)
    negation_pairs = [
        ("no", "Okay, no problem."),
        ("nope", "Alright, understood."),
        ("not really", "I understand."),
        ("never mind", "No worries!"),
    ]
    
    for inp, out in negation_pairs:
        space.add_pair(inp, out, "negation_response")
    
    # =========================================================================
    # PHASE 2: Discover emergent structure
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Discovering emergent structure")
    logger.info("="*60)
    
    n_dims = space.discover_dimensions()
    ideals = space.discover_ideals()
    
    stats = space.get_stats()
    logger.info(f"\nSpace statistics:")
    logger.info(f"  Pairs: {stats['pairs']}")
    logger.info(f"  Dimensions: {stats['dimensions']}")
    logger.info(f"  Concepts: {stats['concepts']}")
    logger.info(f"  Ideals: {stats['ideals']}")
    
    # =========================================================================
    # PHASE 3: Test navigation (inference via navigation)
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Testing navigation (inference)")
    logger.info("="*60)
    
    # Test known inputs
    test_inputs = [
        ("hello", "greeting_response"),
        ("hi", "greeting_response"),
        ("what is your name", "question_answer"),
        ("help me", "command_response"),
        ("goodbye", "farewell_response"),
    ]
    
    logger.info("\n--- Known inputs (with dimension hint) ---")
    for inp, dim in test_inputs:
        result, score, dim_used = space.navigate(inp, dimension=dim)
        logger.info(f"  {inp:25s} → {result:30s} (score={score:.1f}%)")
    
    # Test unknown inputs (auto-detect dimension)
    unknown_inputs = [
        "hi there",      # Should detect greeting
        "what's up",     # Should detect greeting
        "can you help",  # Should detect command
        "later",         # Should detect farewell
        "how do I",      # Should detect question
        "thanks",        # New - should detect something
        "please do",     # Should detect command
    ]
    
    logger.info("\n--- Unknown inputs (auto-detect dimension) ---")
    for inp in unknown_inputs:
        result, score, dim_used = space.navigate(inp)
        logger.info(f"  {inp:25s} → {result:30s} (dim={dim_used}, score={score:.1f}%)")
    
    # =========================================================================
    # PHASE 4: Analyze flip patterns
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Analyzing emergent flip patterns")
    logger.info("="*60)
    
    for dim_name, flip_pattern in space.flip_patterns.items():
        n_flips = (flip_pattern > 0.5).sum().item()
        logger.info(f"  {dim_name}: {n_flips} flip dimensions ({n_flips/len(flip_pattern)*100:.1f}%)")
    
    # Check if flip patterns have common structure (like crystalline 50/50)
    if len(space.flip_patterns) >= 2:
        patterns = list(space.flip_patterns.values())
        pattern_matrix = torch.stack(patterns)
        U, S, Vt = torch.linalg.svd(pattern_matrix)
        
        variance_first = (S[0]**2 / (S**2).sum() * 100).item()
        logger.info(f"\n  Common core (first SV): {variance_first:.1f}% variance")
        logger.info(f"  This is the UNIVERSAL response transformation!")
    
    return space


if __name__ == "__main__":
    space = demo_self_assembling_inference()
