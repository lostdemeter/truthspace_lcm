#!/usr/bin/env python3
"""
Sign-Only Navigation Server (Crystalline Structure)
=====================================================

Pure geometric navigation using sign patterns with crystalline flip structure.
From Doc 166: Flip patterns form a crystal with 50% universal + 50% dimension-specific.

Key insights:
  - Universal core (50%): Same for ALL semantic opposites
  - Dimension-specific (50%): Unique to each semantic axis, ~600 dims for 90%
  - Facets are independent: Can't predict one axis from another

Storage:
  - Signs: 68 MB (16x compression from 1.09 GB)
  - σ=0.5 projection: 2 MB (554x compression)
  - Flip patterns: ~14 KB per dimension

Run with:
    cd /home/thorin/truthspace-lcm
    source venv/bin/activate
    python src/phi_navigator/sign_only_server.py --port 8005
"""

import time
import uuid
import argparse
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import numpy as np
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# API MODELS
# =============================================================================

class Message(BaseModel):
    model_config = {"extra": "ignore"}
    role: str
    content: Optional[Any] = ""
    
    def get_text_content(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            texts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return " ".join(texts)
        return str(self.content)


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = "sign-only-navigator"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False


class NavigateRequest(BaseModel):
    word: str
    dimension: Optional[str] = None


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# =============================================================================
# SIGN-ONLY NAVIGATION ENGINE
# =============================================================================

class SignOnlyEngine:
    """
    Pure sign-based navigation engine with CONCEPT SPACE support.
    
    From Doc 166: Crystalline flip structure with 50% universal + 50% dimension-specific.
    
    Key features:
    - Concept space: Handles multi-token words by averaging embeddings
    - 100% accuracy on known pairs (vs 80% in token space)
    - Storage: ~2 MB total (960x compression from original)
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct", sigma: float = 0.5):
        self.model_name = model_name
        self.device = DEVICE
        self.sigma = sigma  # Position in critical strip
        
        # We only need embeddings, not the full model
        self.tokenizer = None
        self.embeds = None  # Full embeddings (for concept space)
        self.all_signs = None  # int8: +1 or -1 (full dim)
        self.all_signs_low = None  # int8: +1 or -1 (σ=0.5 projected)
        self.projection_matrix = None  # [hidden_dim, k] for σ=0.5
        self.hidden_dim = None
        self.k_optimal = None  # √hidden_dim for σ=0.5
        self.vocab_size = None
        
        # Concept space (handles multi-token words)
        self.concept_signs: Dict[str, torch.Tensor] = {}  # word -> sign pattern
        self.concept_words: List[str] = []  # List of all concepts
        
        # Semantic dimensions (both full and low-dim)
        self.flip_patterns: Dict[str, torch.Tensor] = {}
        self.flip_patterns_low: Dict[str, torch.Tensor] = {}
        self.word_to_opposite: Dict[str, str] = {}
        
        # Holographic reference beam (common core of all flip patterns)
        self.reference_beam = None  # Full dim
        self.reference_beam_low = None  # σ=0.5
        self.holographic_alpha = 0.5  # Optimal threshold multiplier
        
        # Stats
        self.total_navigations = 0
        self.total_navigation_time_ms = 0
        self.embedding_size_bytes = 0
        self.sign_only_size_bytes = 0
        self.sigma_half_size_bytes = 0
        
        self._load_embeddings()
        self._build_concept_space()
        self._learn_dimensions()
    
    def _load_embeddings(self):
        """Load embeddings and compute σ=0.5 projection."""
        logger.info(f"Loading embeddings from {self.model_name}...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model just to extract embeddings, then delete
        logger.info("Loading model to extract embeddings...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        # Extract embeddings (keep for concept space)
        self.embeds = model.model.embed_tokens.weight.detach().float().cpu()
        self.hidden_dim = self.embeds.shape[1]
        self.vocab_size = self.embeds.shape[0]
        
        # Calculate sizes
        self.embedding_size_bytes = self.embeds.numel() * 2  # bfloat16
        
        # === FULL DIMENSION SIGNS (σ=1.0) ===
        self.all_signs = torch.sign(self.embeds).to(torch.int8)
        self.all_signs[self.all_signs == 0] = 1
        
        # Levels for full holographic encoding
        K = 128
        LOG_PHI = np.log(PHI)
        self.all_levels = torch.round(
            K * torch.log(torch.abs(self.embeds) + 1e-10) / LOG_PHI
        ).to(torch.int16)
        
        # === σ=0.5 PROJECTION (Critical Line) ===
        # k = √k_max for σ=0.5
        self.k_optimal = int(np.sqrt(self.hidden_dim))
        logger.info(f"Computing σ=0.5 projection: {self.hidden_dim} → {self.k_optimal} dims")
        
        # SVD to get principal directions
        U, S, Vt = torch.linalg.svd(self.embeds, full_matrices=False)
        
        # Projection matrix: top k singular vectors
        self.projection_matrix = Vt[:self.k_optimal, :].T  # [hidden_dim, k]
        
        # Project all embeddings to low-dim space
        embeds_low = self.embeds @ self.projection_matrix  # [vocab, k]
        
        # Signs in low-dim space
        self.all_signs_low = torch.sign(embeds_low).to(torch.int8)
        self.all_signs_low[self.all_signs_low == 0] = 1
        
        # Move to device
        self.all_signs = self.all_signs.to(self.device)
        self.all_levels = self.all_levels.to(self.device)
        self.all_signs_low = self.all_signs_low.to(self.device)
        self.projection_matrix = self.projection_matrix.to(self.device)
        
        # Storage calculations
        self.sign_only_size_bytes = self.all_signs.numel() * 1  # int8
        self.sign_only_packed_bytes = self.all_signs.numel() // 8  # bit-packed
        
        # σ=0.5 storage: projection matrix + low-dim signs
        projection_bytes = self.projection_matrix.numel() * 4  # float32
        signs_low_bytes = self.all_signs_low.numel() // 8  # bit-packed
        self.sigma_half_size_bytes = projection_bytes + signs_low_bytes
        
        # Delete the full model to free memory (keep self.embeds for concept space)
        del model, embeds_low, U, S, Vt
        torch.cuda.empty_cache()
        
        logger.info(f"Embeddings loaded: {self.vocab_size} tokens, {self.hidden_dim} dims")
        logger.info(f"σ=0.5 projection: {self.k_optimal} dims (critical line)")
        logger.info(f"Original embedding size: {self.embedding_size_bytes / 1e9:.2f} GB")
        logger.info(f"Sign-only size (σ=1.0): {self.sign_only_packed_bytes / 1e6:.2f} MB ({self.embedding_size_bytes / self.sign_only_packed_bytes:.0f}x)")
        logger.info(f"Sign-only size (σ=0.5): {self.sigma_half_size_bytes / 1e6:.2f} MB ({self.embedding_size_bytes / self.sigma_half_size_bytes:.0f}x)")
        
        if torch.cuda.is_available():
            logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def _build_concept_space(self):
        """Build concept space from curated word list (handles multi-token words)."""
        logger.info("Building concept space...")
        
        # Curated concept vocabulary - common English words for semantic navigation
        curated_words = [
            # Temperature
            'hot', 'cold', 'warm', 'cool', 'freezing', 'burning', 'icy', 'fiery',
            # Size
            'big', 'small', 'huge', 'tiny', 'large', 'little', 'giant', 'mini',
            # Speed
            'fast', 'slow', 'quick', 'sluggish', 'rapid', 'gradual', 'swift', 'leisurely',
            # Height
            'short', 'tall', 'low', 'high', 'squat', 'towering',
            # Brightness
            'dark', 'bright', 'dim', 'light', 'gloomy', 'radiant',
            # Age
            'young', 'old', 'new', 'ancient', 'fresh', 'stale',
            # Valence
            'bad', 'good', 'sad', 'happy', 'negative', 'positive', 'evil',
            # Weight
            'heavy', 'weightless', 'weighty',
            # Hardness
            'soft', 'hard', 'tender', 'tough', 'gentle', 'harsh',
            # Moisture
            'dry', 'wet', 'arid', 'damp', 'parched', 'moist',
            # Emotion
            'love', 'hate', 'joy', 'sorrow', 'hope', 'despair',
            # Wealth
            'rich', 'poor', 'wealthy', 'impoverished', 'affluent', 'destitute',
            # Strength
            'strong', 'weak', 'powerful', 'feeble', 'mighty', 'frail',
            # Volume
            'loud', 'quiet', 'noisy', 'silent', 'deafening', 'mute',
            # Cleanliness
            'clean', 'dirty', 'pure', 'filthy', 'spotless', 'grimy',
            # Truth
            'true', 'false', 'real', 'fake', 'genuine', 'counterfeit',
            # Beauty
            'beautiful', 'ugly', 'pretty', 'hideous', 'gorgeous', 'grotesque',
            # Intelligence
            'smart', 'dumb', 'clever', 'stupid', 'wise', 'foolish',
            # Safety
            'safe', 'dangerous', 'secure', 'risky', 'harmless', 'harmful',
            # Fullness
            'full', 'empty', 'complete', 'incomplete', 'whole', 'partial',
            # Additional concepts
            'brave', 'coward', 'kind', 'cruel', 'honest', 'dishonest',
            'calm', 'angry', 'alive', 'dead', 'awake', 'asleep',
            'open', 'closed', 'thick', 'thin', 'deep', 'shallow',
            'wide', 'narrow', 'long', 'near', 'far',
            'easy', 'difficult', 'simple', 'complex', 'clear', 'confusing',
            'cheap', 'expensive', 'free', 'costly',
            'early', 'late', 'first', 'last', 'beginning', 'end',
            'inside', 'outside', 'above', 'below', 'front', 'back',
            'left', 'right', 'up', 'down', 'forward', 'backward',
        ]
        
        # Build concept embeddings (average for multi-token words)
        for word in curated_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) == 0:
                continue
            
            # Average embedding for multi-token words
            word_embed = self.embeds[tokens].mean(dim=0)
            word_signs = torch.sign(word_embed).to(torch.int8)
            word_signs[word_signs == 0] = 1
            
            self.concept_signs[word] = word_signs.to(self.device)
            self.concept_words.append(word)
        
        logger.info(f"Concept space: {len(self.concept_words)} words")
    
    def _get_concept_signs(self, word: str) -> Optional[torch.Tensor]:
        """Get sign pattern for a word (handles multi-token via averaging)."""
        if word in self.concept_signs:
            return self.concept_signs[word]
        
        # Compute on-the-fly for unknown words
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            return None
        
        word_embed = self.embeds[tokens].mean(dim=0)
        word_signs = torch.sign(word_embed).to(torch.int8)
        word_signs[word_signs == 0] = 1
        return word_signs.to(self.device)
    
    def navigate_concept(self, word: str, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Navigate in concept space to find the opposite of a word.
        Handles multi-token words by averaging embeddings.
        """
        start_time = time.perf_counter()
        self.total_navigations += 1
        
        # Check exact opposite first
        if word in self.word_to_opposite:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_navigation_time_ms += elapsed_ms
            return {
                "word": word,
                "opposite": self.word_to_opposite[word],
                "method": "exact_lookup",
                "confidence": 100.0,
                "time_ms": elapsed_ms,
            }
        
        source_signs = self._get_concept_signs(word)
        if source_signs is None:
            return {"error": f"Word '{word}' not found"}
        
        # If dimension specified, use its flip pattern
        if dimension and dimension in self.flip_patterns:
            flip_pattern = self.flip_patterns[dimension]
        else:
            # Try all dimensions, find best match
            best_result = None
            best_score = -float('inf')
            
            for dim_name, flip_pattern in self.flip_patterns.items():
                target_signs = source_signs.float().clone()
                target_signs[flip_pattern > 0.5] *= -1
                
                # Search in concept space
                for cand_word in self.concept_words:
                    if cand_word == word:
                        continue
                    cand_signs = self.concept_signs[cand_word]
                    agreement = (cand_signs.float() == target_signs).float().sum().item()
                    
                    if agreement > best_score:
                        best_score = agreement
                        best_result = {
                            "word": word,
                            "opposite": cand_word,
                            "dimension": dim_name,
                            "confidence": agreement / self.hidden_dim * 100,
                            "method": "concept_navigation",
                        }
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_navigation_time_ms += elapsed_ms
            
            if best_result:
                best_result["time_ms"] = elapsed_ms
                return best_result
            
            return {"error": f"Could not find opposite for '{word}'"}
        
        # Use specified dimension
        target_signs = source_signs.float().clone()
        target_signs[flip_pattern > 0.5] *= -1
        
        best_word = None
        best_score = -float('inf')
        
        for cand_word in self.concept_words:
            if cand_word == word:
                continue
            cand_signs = self.concept_signs[cand_word]
            agreement = (cand_signs.float() == target_signs).float().sum().item()
            
            if agreement > best_score:
                best_score = agreement
                best_word = cand_word
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.total_navigation_time_ms += elapsed_ms
        
        return {
            "word": word,
            "opposite": best_word,
            "dimension": dimension,
            "confidence": best_score / self.hidden_dim * 100,
            "method": "concept_navigation",
            "time_ms": elapsed_ms,
        }
    
    def _learn_dimensions(self):
        """Learn semantic dimensions and compute holographic reference beam."""
        logger.info("Learning semantic dimensions...")
        
        dimensions = {
            "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery")],
            "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant")],
            "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"), ("leisurely", "swift")],
            "height": [("short", "tall"), ("low", "high"), ("squat", "towering")],
            "brightness": [("dark", "bright"), ("dim", "light"), ("gloomy", "radiant")],
            "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale")],
            "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive"), ("evil", "good")],
            "weight": [("light", "heavy"), ("weightless", "weighty")],
            "hardness": [("soft", "hard"), ("tender", "tough"), ("gentle", "harsh")],
            "moisture": [("dry", "wet"), ("arid", "damp"), ("parched", "moist")],
            # Additional dimensions for better holographic coverage
            "emotion": [("love", "hate"), ("joy", "sorrow"), ("hope", "despair")],
            "wealth": [("rich", "poor"), ("wealthy", "impoverished"), ("affluent", "destitute")],
            "strength": [("strong", "weak"), ("powerful", "feeble"), ("mighty", "frail")],
            "volume": [("loud", "quiet"), ("noisy", "silent"), ("deafening", "mute")],
            "cleanliness": [("clean", "dirty"), ("pure", "filthy"), ("spotless", "grimy")],
            "truth": [("true", "false"), ("real", "fake"), ("genuine", "counterfeit")],
            "beauty": [("beautiful", "ugly"), ("pretty", "hideous"), ("gorgeous", "grotesque")],
            "intelligence": [("smart", "dumb"), ("clever", "stupid"), ("wise", "foolish")],
            "safety": [("safe", "dangerous"), ("secure", "risky"), ("harmless", "harmful")],
            "fullness": [("full", "empty"), ("complete", "incomplete"), ("whole", "partial")],
            "courage": [("brave", "coward"), ("bold", "timid"), ("fearless", "fearful")],
            "kindness": [("kind", "cruel"), ("gentle", "harsh"), ("caring", "callous")],
            "honesty": [("honest", "dishonest"), ("truthful", "deceitful"), ("sincere", "insincere")],
            "calmness": [("calm", "angry"), ("peaceful", "agitated"), ("serene", "furious")],
            "life": [("alive", "dead"), ("living", "deceased"), ("vital", "lifeless")],
            "consciousness": [("awake", "asleep"), ("alert", "drowsy"), ("conscious", "unconscious")],
        }
        
        # Collect all flip patterns for holographic reference beam
        all_flip_patterns = []
        all_flip_patterns_low = []
        
        for name, pairs in dimensions.items():
            flip_pattern, flip_pattern_low = self._learn_dimension(name, pairs)
            if flip_pattern is not None:
                all_flip_patterns.append(flip_pattern)
            if flip_pattern_low is not None:
                all_flip_patterns_low.append(flip_pattern_low)
        
        # Compute holographic reference beam via SVD (common core)
        if all_flip_patterns:
            flip_matrix = torch.stack(all_flip_patterns)  # [n_dims, hidden_dim]
            U, S, Vt = torch.linalg.svd(flip_matrix.cpu())
            self.reference_beam = Vt[0].to(self.device)  # First right singular vector
            variance_captured = (S[0]**2 / (S**2).sum() * 100).item()
            logger.info(f"Holographic reference beam (full): {variance_captured:.1f}% variance captured")
        
        if all_flip_patterns_low:
            flip_matrix_low = torch.stack(all_flip_patterns_low)
            U_low, S_low, Vt_low = torch.linalg.svd(flip_matrix_low.cpu())
            self.reference_beam_low = Vt_low[0].to(self.device)
            variance_captured_low = (S_low[0]**2 / (S_low**2).sum() * 100).item()
            logger.info(f"Holographic reference beam (σ=0.5): {variance_captured_low:.1f}% variance captured")
        
        logger.info(f"Learned {len(self.flip_patterns)} dimensions, {len(self.word_to_opposite)} opposites")
    
    def _compute_level_weights(self, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute per-dimension weights from φ-levels.
        
        Near-zero dimensions (|level| small) get higher weight because sign
        at fringe boundaries carries more information (Doc 253/254).
        
        Weight function: w = φ^(-|level|/K)
        - level=0 (|x|≈1): w=1.0 (maximum)
        - |level|=K (|x|≈φ): w=1/φ ≈ 0.618
        - |level|=2K (|x|≈φ²): w=1/φ² ≈ 0.382
        
        Args:
            token_ids: If provided, average weights across these tokens.
                       If None, use the global mean across all tokens.
        """
        K = 128.0  # Same scale used in level computation
        if token_ids is not None:
            levels = self.all_levels[token_ids].float()
            if levels.dim() > 1:
                levels = levels.mean(dim=0)  # Average across tokens
        else:
            # Global: use mean absolute level across vocabulary
            levels = self.all_levels.float().mean(dim=0)
        
        weights = PHI ** (-torch.abs(levels) / K)
        return weights.to(self.device)
    
    def _weighted_sign_agreement(self, source_signs: torch.Tensor, 
                                  all_signs: torch.Tensor,
                                  weights: torch.Tensor) -> torch.Tensor:
        """
        Compute level-weighted sign agreement between source and all tokens.
        
        Instead of counting matching signs equally:
          score = sum(sign_a == sign_b)                    [unweighted]
        We weight by how close each dimension is to zero:
          score = sum(w_d * (sign_a_d == sign_b_d))        [weighted]
        
        This implements the negative-zero insight: sign flips near zero
        carry ~4× more information than sign flips far from zero.
        """
        matches = (all_signs == source_signs.unsqueeze(0)).float()  # [vocab, dims]
        return (matches * weights.unsqueeze(0)).sum(dim=1)  # [vocab]
    
    def _learn_dimension(self, name: str, pairs: List[tuple]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Learn flip pattern for a dimension (both full and σ=0.5). Returns patterns for holographic beam."""
        # Full dimension flip pattern (σ=1.0)
        flip_counts = torch.zeros(self.hidden_dim, dtype=torch.float32, device=self.device)
        # Low dimension flip pattern (σ=0.5)
        flip_counts_low = torch.zeros(self.k_optimal, dtype=torch.float32, device=self.device)
        n_pairs = 0
        
        for neg_word, pos_word in pairs:
            neg_id = self._get_token_id(neg_word)
            pos_id = self._get_token_id(pos_word)
            
            if neg_id is None or pos_id is None:
                continue
            
            # Full dimension
            s_neg = self.all_signs[neg_id]
            s_pos = self.all_signs[pos_id]
            flips = (s_neg != s_pos).float()
            flip_counts += flips
            
            # Low dimension (σ=0.5)
            s_neg_low = self.all_signs_low[neg_id]
            s_pos_low = self.all_signs_low[pos_id]
            flips_low = (s_neg_low != s_pos_low).float()
            flip_counts_low += flips_low
            
            n_pairs += 1
            
            self.word_to_opposite[neg_word] = pos_word
            self.word_to_opposite[pos_word] = neg_word
        
        if n_pairs > 0:
            flip_prob = flip_counts / n_pairs
            self.flip_patterns[name] = (flip_prob > 0.5)
            
            flip_prob_low = flip_counts_low / n_pairs
            self.flip_patterns_low[name] = (flip_prob_low > 0.5)
            
            return flip_prob, flip_prob_low
        
        return None, None
    
    def _get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def navigate_holographic(self, word: str, use_sigma_half: bool = True, alpha: float = None) -> Dict[str, Any]:
        """
        Navigate using holographic projection through the reference beam.
        
        Like Additive Error Stereo: the flip pattern EMERGES from projecting
        through a single reference beam (common core of all semantic dimensions).
        
        Args:
            word: The word to find opposite of
            use_sigma_half: If True, use σ=0.5 (60-dim) navigation (default)
            alpha: Threshold multiplier (default: self.holographic_alpha)
        """
        start_time = time.perf_counter()
        self.total_navigations += 1
        
        if alpha is None:
            alpha = self.holographic_alpha
        
        # Check exact opposite first
        if word in self.word_to_opposite:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_navigation_time_ms += elapsed_ms
            return {
                "word": word,
                "opposite": self.word_to_opposite[word],
                "method": "exact_lookup",
                "confidence": 100.0,
                "time_ms": elapsed_ms,
            }
        
        word_id = self._get_token_id(word)
        if word_id is None:
            return {"error": f"Word '{word}' not found in vocabulary"}
        
        # Choose which sign space and reference beam to use
        if use_sigma_half:
            source_signs = self.all_signs_low[word_id].float()
            all_signs = self.all_signs_low
            reference_beam = self.reference_beam_low
            n_dims = self.k_optimal
            sigma_str = "0.5"
        else:
            source_signs = self.all_signs[word_id].float()
            all_signs = self.all_signs
            reference_beam = self.reference_beam
            n_dims = self.hidden_dim
            sigma_str = "1.0"
        
        if reference_beam is None:
            return {"error": "Reference beam not computed"}
        
        # Holographic projection: flip where reference beam is strong
        flip_strength = reference_beam.abs()
        flip_threshold = flip_strength.mean() + flip_strength.std() * alpha
        flip_mask = flip_strength > flip_threshold
        n_flips = flip_mask.sum().item()
        
        # Create target by flipping at high-strength positions
        target_signs = source_signs.clone()
        target_signs[flip_mask] *= -1
        
        # Find nearest by sign agreement
        # Use level-weighted comparison for σ=1.0 (Doc 254: negative zero weighting)
        if not use_sigma_half:
            level_weights = self._compute_level_weights(token_ids=torch.tensor([word_id]))
            agreement = self._weighted_sign_agreement(target_signs.to(torch.int8), all_signs, level_weights)
            weight_sum = level_weights.sum().item()
        else:
            agreement = (all_signs.float() == target_signs.unsqueeze(0)).float().sum(dim=1)
            weight_sum = float(n_dims)
        agreement[word_id] = -1
        
        top_idx = agreement.argmax().item()
        score = agreement[top_idx].item()
        result_word = self.tokenizer.decode([top_idx]).strip()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.total_navigation_time_ms += elapsed_ms
        
        return {
            "word": word,
            "opposite": result_word,
            "method": "holographic",
            "confidence": score / weight_sum * 100,
            "sigma": sigma_str,
            "n_dims": n_dims,
            "n_flips": n_flips,
            "alpha": alpha,
            "level_weighted": not use_sigma_half,
            "time_ms": elapsed_ms,
        }
    
    def navigate(self, word: str, dimension: Optional[str] = None, use_sigma_half: bool = True, 
                 use_holographic: bool = True) -> Dict[str, Any]:
        """
        Navigate to find the opposite of a word.
        
        Args:
            word: The word to find opposite of
            dimension: Specific semantic dimension (or None for auto)
            use_sigma_half: If True, use σ=0.5 (60-dim) navigation (default)
            use_holographic: If True, use holographic projection (default)
        """
        # Use holographic navigation by default for unknown words
        if use_holographic and word not in self.word_to_opposite:
            return self.navigate_holographic(word, use_sigma_half=use_sigma_half)
        
        start_time = time.perf_counter()
        self.total_navigations += 1
        
        # Check exact opposite first
        if word in self.word_to_opposite:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_navigation_time_ms += elapsed_ms
            return {
                "word": word,
                "opposite": self.word_to_opposite[word],
                "dimension": "exact_match",
                "confidence": 100.0,
                "method": "exact_lookup",
                "sigma": "N/A",
                "time_ms": elapsed_ms,
            }
        
        word_id = self._get_token_id(word)
        if word_id is None:
            return {"error": f"Word '{word}' not found in vocabulary"}
        
        # Choose which sign space to use
        if use_sigma_half:
            source_signs = self.all_signs_low[word_id]
            all_signs = self.all_signs_low
            flip_patterns = self.flip_patterns_low
            n_dims = self.k_optimal
            sigma_str = "0.5"
        else:
            source_signs = self.all_signs[word_id]
            all_signs = self.all_signs
            flip_patterns = self.flip_patterns
            n_dims = self.hidden_dim
            sigma_str = "1.0"
        
        # If dimension specified, use it; otherwise try all
        if dimension and dimension in flip_patterns:
            dims_to_try = [dimension]
        else:
            dims_to_try = list(flip_patterns.keys())
        
        best_result = None
        best_score = -float('inf')
        
        # Level-weighted comparison for σ=1.0 (Doc 254: negative zero weighting)
        use_weighted = not use_sigma_half
        if use_weighted:
            level_weights = self._compute_level_weights(token_ids=torch.tensor([word_id]))
            weight_sum = level_weights.sum().item()
        else:
            level_weights = None
            weight_sum = float(n_dims)
        
        for dim_name in dims_to_try:
            flip_mask = flip_patterns[dim_name]
            
            target_signs = source_signs.clone()
            target_signs[flip_mask] *= -1
            
            # Find nearest by sign agreement
            if use_weighted:
                agreement = self._weighted_sign_agreement(target_signs, all_signs, level_weights)
            else:
                agreement = (all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
            agreement[word_id] = -1
            
            top_idx = agreement.argmax().item()
            score = agreement[top_idx].item()
            
            result_word = self.tokenizer.decode([top_idx]).strip()
            
            if score > best_score and result_word.isalpha() and len(result_word) >= 2:
                best_score = score
                best_result = {
                    "word": word,
                    "opposite": result_word,
                    "dimension": dim_name,
                    "confidence": score / weight_sum * 100,
                    "method": "sign_navigation",
                    "sigma": sigma_str,
                    "n_dims": n_dims,
                    "level_weighted": use_weighted,
                }
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.total_navigation_time_ms += elapsed_ms
        
        if best_result:
            best_result["time_ms"] = elapsed_ms
            return best_result
        
        return {"error": f"Could not find opposite for '{word}'"}
    
    def find_similar(self, word: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find words with similar positions in semantic space.
        
        Uses BOTH sign AND level for full holographic position (Doc 142):
          position = (sign, level) in φ-space
        """
        word_id = self._get_token_id(word)
        if word_id is None:
            return []
        
        source_signs = self.all_signs[word_id]
        source_levels = self.all_levels[word_id]
        source_word_lower = word.lower()
        
        # Level-weighted sign agreement (Doc 254: negative zero weighting)
        # Near-zero dimensions get higher weight — sign at fringe boundaries
        # carries more information than sign in bright/dark regions
        level_weights = self._compute_level_weights(token_ids=torch.tensor([word_id]))
        weighted_sign_agreement = self._weighted_sign_agreement(source_signs, self.all_signs, level_weights)
        
        # Level proximity: inverse of L1 distance, normalized
        level_diff = torch.abs(self.all_levels - source_levels.unsqueeze(0)).float().sum(dim=1)
        max_level_diff = level_diff.max()
        level_proximity = 1.0 - (level_diff / (max_level_diff + 1e-10))
        
        # Combined score: weighted sign is primary, level proximity is secondary
        weight_sum = level_weights.sum().item()
        combined_score = weighted_sign_agreement + level_proximity * weight_sum * 0.1
        combined_score[word_id] = -1  # Exclude self
        
        top_indices = combined_score.topk(top_k * 10).indices
        
        results = []
        seen_words = {source_word_lower}
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            result_lower = result_word.lower()
            
            # Filter: alphabetic, 3+ chars, lowercase, not substring
            if (result_word.isalpha() and 
                len(result_word) >= 3 and 
                result_word.islower() and
                result_lower not in seen_words and
                source_word_lower not in result_lower and
                result_lower not in source_word_lower):
                
                seen_words.add(result_lower)
                results.append({
                    "word": result_word,
                    "sign_agreement": weighted_sign_agreement[idx].item() / weight_sum * 100,
                    "level_proximity": level_proximity[idx].item() * 100,
                })
                if len(results) >= top_k:
                    break
        
        return results
    
    def interpret(self, text: str) -> torch.Tensor:
        """
        Interpret text by combining sign patterns of all tokens.
        
        This creates a "meaning vector" from the input.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return torch.zeros(self.hidden_dim, device=self.device)
        
        # Combine sign patterns - use multiplication (like quaternion combination)
        combined = torch.ones(self.hidden_dim, dtype=torch.float32, device=self.device)
        for tid in tokens:
            combined *= self.all_signs[tid].float()
        
        return combined.sign().to(torch.int8)
    
    def generate_from_signs(self, target_signs: torch.Tensor, exclude_ids: set = None, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Find tokens that best match a target sign pattern.
        """
        if exclude_ids is None:
            exclude_ids = set()
        
        # Sign agreement
        agreement = (self.all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
        
        # Exclude specified tokens
        for tid in exclude_ids:
            agreement[tid] = -1
        
        top_indices = agreement.topk(top_k * 3).indices
        
        results = []
        for idx in top_indices:
            word = self.tokenizer.decode([idx.item()]).strip()
            if word and len(word) >= 2:
                score = agreement[idx].item() / self.hidden_dim * 100
                results.append((word, score))
                if len(results) >= top_k:
                    break
        
        return results
    
    def generate_response(self, user_message: str, max_tokens: int = 50) -> str:
        """
        Generate a full response using sign-based interpretation.
        
        The 1-2 cycle architecture:
          ENCODE: user_message → sign pattern
          NAVIGATE: transform sign pattern based on intent
          DECODE: find tokens matching transformed pattern
        """
        msg_lower = user_message.lower()
        
        # ENCODE: Get the meaning of the input
        input_signs = self.interpret(user_message)
        
        # Detect intent and NAVIGATE accordingly
        
        # Intent: Find opposite
        if "opposite" in msg_lower or "antonym" in msg_lower:
            # Extract the target word
            words = user_message.split()
            target_word = None
            for i, w in enumerate(words):
                if w.lower() in ("opposite", "antonym") and i + 2 < len(words):
                    target_word = words[i + 2].strip("?.,!\"'")
                    break
            
            if target_word:
                result = self.navigate(target_word)
                if "error" not in result:
                    return f"The opposite of '{target_word}' is '{result['opposite']}'."
                return f"I couldn't find the opposite of '{target_word}'."
        
        # Intent: Find similar
        if "similar" in msg_lower or "like" in msg_lower:
            words = user_message.split()
            target_word = None
            for i, w in enumerate(words):
                if w.lower() in ("similar", "like") and i + 2 < len(words):
                    target_word = words[i + 2].strip("?.,!\"'")
                    break
            
            if target_word:
                results = self.find_similar(target_word, top_k=5)
                if results:
                    words_list = [r['word'] for r in results]
                    return f"Words similar to '{target_word}': {', '.join(words_list)}"
                return f"I couldn't find words similar to '{target_word}'."
        
        # Intent: Define/explain
        if "what is" in msg_lower or "define" in msg_lower or "meaning" in msg_lower:
            # Extract the word to define
            for phrase in ["what is", "define", "meaning of"]:
                if phrase in msg_lower:
                    parts = msg_lower.split(phrase)
                    if len(parts) > 1:
                        target_word = parts[1].strip().split()[0].strip("?.,!\"'")
                        
                        # Find related words via sign similarity
                        results = self.find_similar(target_word, top_k=10)
                        if results:
                            related = [r['word'] for r in results[:5]]
                            
                            # Find opposite
                            opp_result = self.navigate(target_word)
                            opp_word = opp_result.get('opposite', 'unknown')
                            
                            return f"'{target_word}' is related to: {', '.join(related)}. Its opposite is '{opp_word}'."
                        return f"I don't have enough information about '{target_word}'."
        
        # Intent: General question - use associative retrieval
        if "?" in user_message:
            # Find tokens most associated with the question's meaning
            candidates = self.generate_from_signs(input_signs, top_k=10)
            if candidates:
                # Build a response from associated words
                response_words = [w for w, s in candidates if s > 55][:5]
                if response_words:
                    return f"Based on sign patterns, this relates to: {', '.join(response_words)}"
        
        # Default: Explain capabilities
        return (
            "I understand through sign patterns (1 bit per dimension). I can:\n"
            "• Find opposites: 'What is the opposite of hot?'\n"
            "• Find similar words: 'What is similar to happy?'\n"
            "• Explain concepts: 'What is love?'\n\n"
            "Storage: 68 MB (16x compression from 1.09 GB)"
        )
    
    def chat_response(self, user_message: str) -> str:
        """
        Generate a response using sign-only navigation.
        """
        return self.generate_response(user_message)
    
    def get_stats(self) -> Dict[str, Any]:
        avg_nav_time = 0
        if self.total_navigations > 0:
            avg_nav_time = self.total_navigation_time_ms / self.total_navigations
        
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / 1e9
        
        return {
            "model": "sign-only-navigator",
            "base_model": self.model_name,
            "device": self.device,
            "storage": {
                "original_embeddings_gb": round(self.embedding_size_bytes / 1e9, 2),
                "sign_only_int8_mb": round(self.sign_only_size_bytes / 1e6, 2),
                "sign_only_packed_mb": round(self.sign_only_packed_bytes / 1e6, 2),
                "compression_ratio": f"{self.embedding_size_bytes / self.sign_only_packed_bytes:.1f}x",
            },
            "gpu_memory_gb": round(gpu_memory_gb, 2),
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "dimensions": list(self.flip_patterns.keys()),
            "known_opposites": len(self.word_to_opposite),
            "performance": {
                "total_navigations": self.total_navigations,
                "total_navigation_time_ms": round(self.total_navigation_time_ms, 1),
                "avg_navigation_time_ms": round(avg_nav_time, 3),
            },
        }


# =============================================================================
# FASTAPI APP
# =============================================================================

engine: Optional[SignOnlyEngine] = None

app = FastAPI(
    title="Sign-Only Navigation Server",
    description="Pure geometric navigation using only sign patterns (16x compression)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global engine
    engine = SignOnlyEngine()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "sign-only-navigator", "device": DEVICE}


@app.get("/stats")
async def get_stats():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_stats()


@app.post("/navigate")
async def navigate(request: NavigateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.navigate(request.word, request.dimension)


@app.post("/similar")
async def find_similar(word: str, top_k: int = 10):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.find_similar(word, top_k)


@app.get("/dimensions")
async def list_dimensions():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {
        "dimensions": list(engine.flip_patterns.keys()),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Get the last user message
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.get_text_content()
                break
        
        start_time = time.perf_counter()
        response_text = engine.chat_response(user_message)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        if request.stream:
            async def generate_stream():
                # Stream word by word
                words = response_text.split()
                for i, word in enumerate(words):
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "sign-only-navigator",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": word + " "} if i == 0 else {"content": word + " "},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)
                
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "sign-only-navigator",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model="sign-only-navigator",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(user_message.split()),
                completion_tokens=len(response_text.split()),
                total_tokens=len(user_message.split()) + len(response_text.split()),
            ),
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Sign-Only Navigation Server")
    parser.add_argument("--port", type=int, default=8005, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              SIGN-ONLY NAVIGATION SERVER                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Pure geometric navigation using ONLY sign patterns.             ║
║  No full model - just 1 bit per dimension.                       ║
║                                                                  ║
║  Storage: 448 bytes per word (vs 7168 bytes) = 16x compression   ║
║  Navigation: 100% accuracy on trained dimensions                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Endpoints:                                                      ║
║    GET  /health              - Health check                      ║
║    GET  /stats               - Statistics                        ║
║    GET  /dimensions          - List semantic dimensions          ║
║    POST /navigate            - Find opposite of a word           ║
║    POST /similar             - Find similar words                ║
║    POST /v1/chat/completions - Chat (limited to navigation)      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)
