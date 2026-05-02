#!/usr/bin/env python3
"""
TruthSpace v1: Concept Composition via Geometric Gates
========================================================

Proves the TruthSpace hypothesis: concepts have fixed geometric positions,
and relationships between concepts are geometric operations (gates) that
can be learned from examples and applied to new concepts.

The key insight: vector addition treats all dimensions equally. But some
dimensions encode entity IDENTITY (should be preserved), some encode the
RELATIONSHIP (should be shifted), and some are NOISE (should be suppressed).
A geometric gate classifies each dimension and applies the right operation.

Tests:
1. Extract entity/answer embeddings from Qwen2-7B (φ-encoded)
2. Learn "capital-of" relationship as a per-dimension gate
3. Apply gate to unseen entities → predict capitals
4. Compare against vector addition and analogy baselines
5. Test concept composition (dragon + shrimp → lobster)
6. Probe for mathematical structure in concept positions

DC 297 §8.6-8.9: Complexity Ladder + Error Not Compression + TruthSpace

Author: TruthSpace LCM Team
License: GPLv3
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations

# Path setup
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "phi_model"
sys.path.insert(0, str(PROJECT_ROOT))

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
PHI_GRID = 128


# =============================================================================
# φ-ENCODING UTILITIES
# =============================================================================

def phi_encode(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode float values to (signs, exponents) in φ-basis."""
    signs = np.sign(values).astype(np.int8)
    signs[signs == 0] = 1
    magnitudes = np.abs(values).astype(np.float64) + 1e-20
    exponents = np.round(PHI_GRID * np.log(magnitudes) / LOG_PHI).astype(np.int16)
    return signs, exponents


def phi_decode(signs: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Decode (signs, exponents) from φ-basis to float."""
    return (
        signs.astype(np.float64)
        * PHI ** (exponents.astype(np.float64) / PHI_GRID)
    ).astype(np.float32)


# =============================================================================
# GATE PRIMITIVES (from geometric_ipa)
# =============================================================================

def ideal_gate(x: np.ndarray) -> np.ndarray:
    """
    IdealGate(x) = x * sigmoid(sqrt(8/pi) * x * (1 + (4-pi)/(6*pi) * x^2))
    The exact geometric form of GELU.
    """
    coeff = np.sqrt(8 / np.pi)
    cubic_coeff = (4 - np.pi) / (6 * np.pi)
    inner = coeff * x * (1 + cubic_coeff * x ** 2)
    return x * (1 / (1 + np.exp(-np.clip(inner, -50, 50))))


def gate_step(x: np.ndarray, target: float, sharpness: float) -> np.ndarray:
    """
    Width-1 rectangular pulse at target.
    gate_step(x, t, s) = [IdealGate(s*(x-(t-0.5))) - IdealGate(s*(x-(t+0.5)))] / s
    """
    left = ideal_gate(sharpness * (x - (target - 0.5)))
    right = ideal_gate(sharpness * (x - (target + 0.5)))
    return (left - right) / sharpness


# =============================================================================
# CONCEPT DRUM
# =============================================================================

@dataclass
class Concept:
    """A concept with position in TruthSpace."""
    name: str
    token_id: int
    token_str: str
    embedding: np.ndarray      # float32, shape (3584,)
    phi_signs: np.ndarray      # int8, shape (3584,)
    phi_exponents: np.ndarray  # int16, shape (3584,)


# =============================================================================
# RELATIONSHIP GATE
# =============================================================================

@dataclass
class DimensionRule:
    """Classification of what a single dimension does in a relationship."""
    dim: int
    rule_type: str   # PRESERVE, SHIFT, FLIP, COMB
    shift_value: float = 0.0
    confidence: float = 0.0


@dataclass
class RelationshipGate:
    """
    A relationship as a per-dimension gate.

    Each dimension is classified as:
    - PRESERVE: value unchanged (entity identity)
    - SHIFT: value shifts by a fixed delta (relationship transform)
    - FLIP: sign changes (relationship inversion)
    - COMB: value is unreliable, zero/scale down (noise)

    This is the geometric analog of the Spectrometer's per-dimension rules
    (F62), but applied to RELATIONSHIPS rather than layer transformations.
    """
    name: str
    dim_types: np.ndarray    # int8 array: 0=PRESERVE, 1=SHIFT, 2=FLIP, 3=COMB
    shift_values: np.ndarray # float32 array: shift amount per dim
    confidences: np.ndarray  # float32 array: confidence per dim

    # Counts
    n_preserve: int = 0
    n_shift: int = 0
    n_flip: int = 0
    n_comb: int = 0

    TYPE_PRESERVE = 0
    TYPE_SHIFT = 1
    TYPE_FLIP = 2
    TYPE_COMB = 3

    def apply(self, entity_embedding: np.ndarray,
              comb_scale: float = 0.0) -> np.ndarray:
        """Apply this relationship gate to an entity embedding."""
        result = entity_embedding.copy()

        # SHIFT dims: add the relationship delta
        shift_mask = self.dim_types == self.TYPE_SHIFT
        result[shift_mask] += self.shift_values[shift_mask]

        # FLIP dims: negate + shift
        flip_mask = self.dim_types == self.TYPE_FLIP
        result[flip_mask] *= -1
        result[flip_mask] += self.shift_values[flip_mask]

        # COMB dims: scale down (suppress noise)
        comb_mask = self.dim_types == self.TYPE_COMB
        result[comb_mask] *= comb_scale

        # PRESERVE dims: unchanged (no operation needed)
        return result

    def apply_weighted(self, entity_embedding: np.ndarray,
                       comb_scale: float = 0.0) -> np.ndarray:
        """Apply gate with confidence weighting on shifts."""
        result = entity_embedding.copy()

        shift_mask = self.dim_types == self.TYPE_SHIFT
        result[shift_mask] += self.shift_values[shift_mask] * self.confidences[shift_mask]

        flip_mask = self.dim_types == self.TYPE_FLIP
        blend = self.confidences[flip_mask]
        result[flip_mask] = (
            result[flip_mask] * (1 - blend)
            + (-result[flip_mask] + self.shift_values[flip_mask]) * blend
        )

        comb_mask = self.dim_types == self.TYPE_COMB
        result[comb_mask] *= comb_scale

        return result

    def apply_hybrid(self, entity_embedding: np.ndarray,
                     avg_delta: np.ndarray) -> np.ndarray:
        """
        Hybrid gate: per-dim operations where confident, VA delta where not.

        - SHIFT dims: use per-dim shift (learned, specific)
        - FLIP dims: flip sign + per-dim shift
        - PRESERVE dims: keep entity value (no change)
        - COMB dims: use avg_delta (regularized, like vector addition)

        This combines the gate's specificity with VA's regularization.
        """
        result = entity_embedding.copy()

        shift_mask = self.dim_types == self.TYPE_SHIFT
        result[shift_mask] += self.shift_values[shift_mask]

        flip_mask = self.dim_types == self.TYPE_FLIP
        result[flip_mask] *= -1
        result[flip_mask] += self.shift_values[flip_mask]

        # COMB dims: use VA delta instead of zeroing
        comb_mask = self.dim_types == self.TYPE_COMB
        result[comb_mask] += avg_delta[comb_mask]

        return result

    def apply_soft(self, entity_embedding: np.ndarray,
                   avg_delta: np.ndarray,
                   blend_floor: float = 0.3) -> np.ndarray:
        """
        Soft gate: confidence-weighted blend between per-dim and VA delta.

        For each dimension:
          result = entity + confidence * per_dim_delta + (1 - confidence) * avg_delta

        High confidence → trust the per-dim shift.
        Low confidence → fall back to VA delta (regularized).
        blend_floor ensures even low-confidence dims get some VA signal.
        """
        result = entity_embedding.copy()

        # Compute per-dim delta (what the gate wants to do)
        gate_delta = np.zeros_like(entity_embedding)

        shift_mask = self.dim_types == self.TYPE_SHIFT
        gate_delta[shift_mask] = self.shift_values[shift_mask]

        flip_mask = self.dim_types == self.TYPE_FLIP
        gate_delta[flip_mask] = (-2 * entity_embedding[flip_mask]
                                 + self.shift_values[flip_mask])

        # For COMB dims, gate_delta stays 0 → fully uses avg_delta
        # For PRESERVE dims, both gate_delta and avg_delta contribute

        # Blend: confidence determines how much to trust gate vs VA
        conf = np.clip(self.confidences, blend_floor, 1.0)
        result += conf * gate_delta + (1 - conf) * avg_delta

        return result

    def apply_sign_gate(self, entity_signs: np.ndarray,
                        entity_exps: np.ndarray,
                        avg_delta: np.ndarray) -> np.ndarray:
        """
        Sign-space gate: exact sign operations + VA magnitude shifts.

        Signs are binary (exact, structural). Magnitudes are continuous (noisy).
        Separate the two: apply learned sign rules exactly, but use VA delta
        for magnitude shifts (regularized).
        """
        # Start with VA result (entity + avg_delta)
        result_float = phi_decode(entity_signs, entity_exps) + avg_delta

        # Override sign for FLIP dims (exact binary operation)
        flip_mask = self.dim_types == self.TYPE_FLIP
        result_float[flip_mask] = np.abs(result_float[flip_mask]) * (-entity_signs[flip_mask].astype(np.float32))

        # For PRESERVE dims, force sign to match entity
        preserve_mask = self.dim_types == self.TYPE_PRESERVE
        result_float[preserve_mask] = (
            np.abs(result_float[preserve_mask])
            * entity_signs[preserve_mask].astype(np.float32)
        )

        return result_float

    @classmethod
    def learn(cls, name: str, pairs: List[Tuple[np.ndarray, np.ndarray]],
              preserve_threshold: float = 0.1,
              shift_cv_threshold: float = 0.5,
              flip_threshold: float = 0.8) -> 'RelationshipGate':
        """
        Learn a relationship gate from (entity, answer) embedding pairs.

        For each dimension d:
        1. Compute delta_d = answer[d] - entity[d] for each pair
        2. If mean(|delta_d|) < threshold relative to entity magnitude → PRESERVE
        3. If sign consistently flips across pairs → FLIP
        4. If CV(delta_d) < threshold (consistent shift) → SHIFT
        5. Otherwise → COMB (unreliable, noisy)
        """
        n_pairs = len(pairs)
        n_dims = len(pairs[0][0])

        # Compute deltas for each pair: shape (n_pairs, n_dims)
        deltas = np.array([answer - entity for entity, answer in pairs])

        # Check sign changes
        entity_signs = np.array([np.sign(entity) for entity, _ in pairs])
        answer_signs = np.array([np.sign(answer) for _, answer in pairs])
        sign_products = entity_signs * answer_signs  # -1 where sign flipped

        # Per-dimension statistics
        mean_delta = np.mean(deltas, axis=0)
        std_delta = np.std(deltas, axis=0)
        mean_abs_delta = np.mean(np.abs(deltas), axis=0)

        # Entity magnitudes for relative thresholding
        entity_mags = np.mean(np.abs(np.array([e for e, _ in pairs])), axis=0)
        relative_delta = mean_abs_delta / (entity_mags + 1e-20)

        # Sign flip consistency (fraction of pairs where sign flipped)
        flip_rate = np.mean(sign_products < 0, axis=0)

        # Classify each dimension
        dim_types = np.full(n_dims, cls.TYPE_COMB, dtype=np.int8)
        shift_values = np.zeros(n_dims, dtype=np.float32)
        confidences = np.zeros(n_dims, dtype=np.float32)

        n_preserve = n_shift = n_flip = n_comb = 0

        for d in range(n_dims):
            if relative_delta[d] < preserve_threshold:
                # Delta tiny relative to value → PRESERVE
                dim_types[d] = cls.TYPE_PRESERVE
                confidences[d] = 1.0 - relative_delta[d] / preserve_threshold
                n_preserve += 1

            elif flip_rate[d] >= flip_threshold:
                # Sign consistently flips → FLIP
                dim_types[d] = cls.TYPE_FLIP
                shift_values[d] = mean_delta[d]
                confidences[d] = flip_rate[d]
                n_flip += 1

            elif mean_abs_delta[d] > 1e-10 and std_delta[d] < abs(mean_delta[d]) * shift_cv_threshold:
                # Low variance relative to mean → consistent SHIFT
                dim_types[d] = cls.TYPE_SHIFT
                shift_values[d] = mean_delta[d]
                confidences[d] = max(0, 1.0 - std_delta[d] / (abs(mean_delta[d]) + 1e-20))
                n_shift += 1

            else:
                # High variance → COMB (noise)
                dim_types[d] = cls.TYPE_COMB
                confidences[d] = 0.0
                n_comb += 1

        return cls(
            name=name,
            dim_types=dim_types,
            shift_values=shift_values,
            confidences=confidences,
            n_preserve=n_preserve,
            n_shift=n_shift,
            n_flip=n_flip,
            n_comb=n_comb,
        )


# =============================================================================
# BOOM GATE — Integer-Space Relationship Detection
# =============================================================================

@dataclass
class BoomGate:
    """
    Relationship gate that works in INTEGER space (φ-exponents + signs).

    The boom hypothesis (DC 159): information concentrates at phase transitions
    where values lock onto integer structure. In embedding space, "boom dimensions"
    are those where the exponent delta between entity→answer is a consistent
    integer across all training pairs.

    Pre-boom dims: exponent delta is chaotic (high variance) → noise
    Boom dims: exponent delta is locked (low variance) → signal

    This is fundamentally different from float-space gates:
    - Exponents are int16, so deltas are integers
    - A consistent integer delta = boom (locked on)
    - Sign relationships are binary (exact, no noise)
    """
    name: str
    # Per-dimension boom classification
    boom_mask: np.ndarray       # bool: True = boom dimension
    sign_flip_mask: np.ndarray  # bool: True = sign flips at this dim
    sign_keep_mask: np.ndarray  # bool: True = sign preserved at this dim
    exp_deltas: np.ndarray      # int16: median exponent delta per dim
    exp_spreads: np.ndarray     # float: IQR of exponent deltas per dim
    n_boom: int
    n_sign_flip: int
    n_sign_keep: int
    n_noise: int

    @classmethod
    def learn(cls, name: str,
              entity_signs_list: List[np.ndarray],
              entity_exps_list: List[np.ndarray],
              answer_signs_list: List[np.ndarray],
              answer_exps_list: List[np.ndarray],
              boom_spread_threshold: float = 2.0,
              sign_consistency_threshold: float = 0.9,
              ) -> 'BoomGate':
        """
        Learn boom dimensions from (entity, answer) pairs in integer space.

        For each dimension:
        1. Compute exp_delta = answer_exp - entity_exp (integer subtraction)
        2. Compute sign_product = entity_sign * answer_sign
        3. Boom = dim where IQR(exp_delta) <= threshold (locked integer shift)
        4. Sign flip = dim where sign consistently flips
        5. Sign keep = dim where sign consistently preserved
        """
        n_pairs = len(entity_signs_list)
        n_dims = len(entity_signs_list[0])

        # Compute exponent deltas: shape (n_pairs, n_dims)
        exp_deltas_all = np.array([
            (a_exp.astype(np.int32) - e_exp.astype(np.int32))
            for e_exp, a_exp in zip(entity_exps_list, answer_exps_list)
        ])

        # Sign products: -1 = flip, +1 = keep
        sign_products = np.array([
            (e_sign.astype(np.int32) * a_sign.astype(np.int32))
            for e_sign, a_sign in zip(entity_signs_list, answer_signs_list)
        ])

        # Per-dimension statistics
        median_exp_delta = np.median(exp_deltas_all, axis=0).astype(np.int16)
        q25 = np.percentile(exp_deltas_all, 25, axis=0)
        q75 = np.percentile(exp_deltas_all, 75, axis=0)
        iqr = q75 - q25  # interquartile range — robust spread measure

        # Sign consistency
        flip_rate = np.mean(sign_products < 0, axis=0)
        keep_rate = np.mean(sign_products > 0, axis=0)

        # Classify
        boom_mask = iqr <= boom_spread_threshold
        sign_flip_mask = flip_rate >= sign_consistency_threshold
        sign_keep_mask = keep_rate >= sign_consistency_threshold

        n_boom = int(np.sum(boom_mask))
        n_sign_flip = int(np.sum(sign_flip_mask))
        n_sign_keep = int(np.sum(sign_keep_mask))
        n_noise = n_dims - n_boom  # dims that are NOT booms

        return cls(
            name=name,
            boom_mask=boom_mask,
            sign_flip_mask=sign_flip_mask,
            sign_keep_mask=sign_keep_mask,
            exp_deltas=median_exp_delta,
            exp_spreads=iqr.astype(np.float32),
            n_boom=n_boom,
            n_sign_flip=n_sign_flip,
            n_sign_keep=n_sign_keep,
            n_noise=n_noise,
        )

    def apply_boom_only(self, entity_signs: np.ndarray,
                        entity_exps: np.ndarray) -> np.ndarray:
        """
        Pure boom gate: apply integer operations at boom dims only.
        Non-boom dims keep entity value (identity).

        This is the cleanest test: does integer-locked structure alone
        produce the right answer?
        """
        # Start with entity
        new_signs = entity_signs.copy().astype(np.int8)
        new_exps = entity_exps.copy().astype(np.int32)

        # At boom dims: apply integer exponent shift
        new_exps[self.boom_mask] += self.exp_deltas[self.boom_mask].astype(np.int32)

        # At sign-flip dims: flip sign
        flip_and_boom = self.boom_mask & self.sign_flip_mask
        new_signs[flip_and_boom] *= -1

        return phi_decode(new_signs, new_exps.astype(np.int16))

    def apply_boom_plus_va(self, entity_signs: np.ndarray,
                           entity_exps: np.ndarray,
                           entity_embedding: np.ndarray,
                           avg_delta: np.ndarray) -> np.ndarray:
        """
        Boom + VA hybrid: integer operations at boom dims, VA at non-boom.
        """
        # VA baseline everywhere
        result = entity_embedding + avg_delta

        # Override boom dims with integer result
        new_signs = entity_signs.copy().astype(np.int8)
        new_exps = entity_exps.copy().astype(np.int32)
        new_exps[self.boom_mask] += self.exp_deltas[self.boom_mask].astype(np.int32)

        flip_and_boom = self.boom_mask & self.sign_flip_mask
        new_signs[flip_and_boom] *= -1

        boom_values = phi_decode(new_signs, new_exps.astype(np.int16))
        result[self.boom_mask] = boom_values[self.boom_mask]

        return result

    def apply_sign_boom(self, entity_signs: np.ndarray,
                        entity_exps: np.ndarray,
                        entity_embedding: np.ndarray,
                        avg_delta: np.ndarray) -> np.ndarray:
        """
        Sign boom: exact sign operations + VA magnitude.

        Signs are binary and exact. The boom insight: sign flips are
        the most reliable integer signal. Use VA for everything else.
        """
        # VA baseline
        result = entity_embedding + avg_delta

        # Override sign at flip dims: force sign to be flipped entity sign
        result[self.sign_flip_mask] = (
            np.abs(result[self.sign_flip_mask])
            * (-entity_signs[self.sign_flip_mask].astype(np.float32))
        )

        # Override sign at keep dims: force sign to match entity
        result[self.sign_keep_mask] = (
            np.abs(result[self.sign_keep_mask])
            * entity_signs[self.sign_keep_mask].astype(np.float32)
        )

        return result

    def apply_boom_weighted(self, entity_signs: np.ndarray,
                            entity_exps: np.ndarray,
                            entity_embedding: np.ndarray,
                            avg_delta: np.ndarray,
                            max_spread: float = 4.0) -> np.ndarray:
        """
        Confidence-weighted boom: blend boom result with VA based on spread.

        Low spread (tight lock) → trust boom (integer shift).
        High spread (noisy) → trust VA (regularized float).
        This is the continuous version of the hard boom_mask cutoff.
        """
        # VA baseline
        va_result = entity_embedding + avg_delta

        # Boom result (integer ops everywhere, even noisy dims)
        new_signs = entity_signs.copy().astype(np.int8)
        new_exps = entity_exps.copy().astype(np.int32) + self.exp_deltas.astype(np.int32)

        # Sign flips
        new_signs[self.sign_flip_mask] *= -1

        boom_result = phi_decode(new_signs, new_exps.astype(np.int16))

        # Confidence: inverse of spread, clamped to [0, 1]
        confidence = np.clip(1.0 - self.exp_spreads / max_spread, 0.0, 1.0)

        # Blend
        return confidence * boom_result + (1.0 - confidence) * va_result

    def compose_int(self, other: 'BoomGate') -> 'BoomGate':
        """
        Compose two boom gates by integer addition of exponent deltas.

        This is the key advantage over float: integer shifts compose EXACTLY.
        country --capital--> city --language--> language
        = country --(capital+language)--> language

        In float space, this accumulates noise. In integer space, it's exact.
        """
        # Exponent deltas compose by addition (integer, exact)
        composed_deltas = (self.exp_deltas.astype(np.int32)
                           + other.exp_deltas.astype(np.int32)).astype(np.int16)

        # Sign flips compose by XOR (flip+flip = keep, flip+keep = flip)
        # In multiplicative sign space: -1 * -1 = +1, -1 * +1 = -1
        composed_sign_flip = self.sign_flip_mask ^ other.sign_flip_mask
        composed_sign_keep = ~composed_sign_flip & (self.sign_keep_mask | other.sign_keep_mask)

        # Boom mask: intersection (both must be locked for composition to be reliable)
        composed_boom = self.boom_mask & other.boom_mask

        # Spread: sum of spreads (worst case)
        composed_spread = self.exp_spreads + other.exp_spreads

        return BoomGate(
            name=f"{self.name}+{other.name}",
            boom_mask=composed_boom,
            sign_flip_mask=composed_sign_flip,
            sign_keep_mask=composed_sign_keep,
            exp_deltas=composed_deltas,
            exp_spreads=composed_spread,
            n_boom=int(np.sum(composed_boom)),
            n_sign_flip=int(np.sum(composed_sign_flip)),
            n_sign_keep=int(np.sum(composed_sign_keep)),
            n_noise=int(np.sum(~composed_boom)),
        )

    def boom_spectrum(self) -> dict:
        """Analyze the boom dimension distribution for diagnostics."""
        spreads = self.exp_spreads
        return {
            "total_dims": len(spreads),
            "boom_dims": self.n_boom,
            "boom_pct": self.n_boom / len(spreads),
            "sign_flip_dims": self.n_sign_flip,
            "sign_keep_dims": self.n_sign_keep,
            "noise_dims": self.n_noise,
            "median_boom_spread": float(np.median(spreads[self.boom_mask])) if self.n_boom > 0 else 0,
            "median_noise_spread": float(np.median(spreads[~self.boom_mask])) if self.n_noise > 0 else 0,
            "boom_exp_delta_median": float(np.median(np.abs(self.exp_deltas[self.boom_mask]))) if self.n_boom > 0 else 0,
        }


# =============================================================================
# EMBEDDING LOADING
# =============================================================================

def load_embeddings():
    """Load token embeddings from φ-encoded model."""
    from phi_geometric.inference.phi_types import PhiEncoded

    path = str(MODEL_DIR / "embed_tokens.npz")
    print(f"  Loading embeddings from {path}...")
    t0 = time.time()
    phi = PhiEncoded.load(path)
    print(f"  Shape: {phi.shape}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    embeddings = phi.decode()
    print(f"  Decoded to float32: {embeddings.shape}  ({time.time() - t0:.1f}s)")
    return embeddings, phi.signs, phi.exponents


def load_tokenizer():
    """Load Qwen2-7B tokenizer vocabulary."""
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    if not os.path.exists(cache_dir):
        print(f"  ERROR: HuggingFace cache not found at {cache_dir}")
        return None, None

    snapshots = os.listdir(cache_dir)
    if not snapshots:
        print("  ERROR: No snapshots found")
        return None, None

    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    if not os.path.exists(vocab_file):
        print(f"  ERROR: tokenizer.json not found at {vocab_file}")
        return None, None

    print(f"  Loading tokenizer from {vocab_file}...")
    with open(vocab_file, "r") as f:
        tokenizer_data = json.load(f)

    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    token_to_id = {}
    for tok, idx in vocab.items():
        token_to_id[tok] = idx

    print(f"  Vocabulary: {len(id_to_token)} tokens")
    return id_to_token, token_to_id


def find_token_id(word: str, token_to_id: Dict[str, int]) -> Tuple[Optional[int], Optional[str]]:
    """Find the token ID for a word, trying various forms."""
    candidates = [
        word, word.lower(), word.capitalize(), word.upper(),
        f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
        f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}",
        # Qwen2 uses byte-level BPE, try common patterns
        f" {word}", f" {word.lower()}", f" {word.capitalize()}",
    ]
    for c in candidates:
        if c in token_to_id:
            return token_to_id[c], c
    return None, None


# =============================================================================
# VOCABULARY SEARCH
# =============================================================================

class VocabSearcher:
    """Fast cosine similarity search over the full vocabulary."""

    def __init__(self, embeddings: np.ndarray, id_to_token: Dict[int, str]):
        print("  Building vocabulary search index...")
        t0 = time.time()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb_normed = embeddings / (norms + 1e-20)
        self.id_to_token = id_to_token
        self.n_vocab = embeddings.shape[0]
        print(f"  Index built: {self.n_vocab} tokens  ({time.time() - t0:.1f}s)")

    def rank_of(self, composed: np.ndarray, target_tid: int,
                exclude_tids: set = None) -> int:
        """Rank of target token by cosine similarity to composed vector."""
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        target_sim = sims[target_tid]
        return int(np.sum(sims > target_sim))

    def top_k(self, composed: np.ndarray, k: int = 5,
              exclude_tids: set = None) -> List[Tuple[str, float]]:
        """Top k tokens by cosine similarity."""
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        top_idx = np.argsort(sims)[-k:][::-1]
        return [(self.id_to_token.get(int(i), f"?{i}"), float(sims[i]))
                for i in top_idx]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print()
    print("=" * 80)
    print("  TruthSpace v1: Concept Composition via Geometric Gates")
    print("  DC 297 §8.9 — Where Ideas Are Mathematically Verifiable")
    print("=" * 80)
    print()

    # ─── Phase 1: Load Data ───────────────────────────────────────────

    print("─── Phase 1: Load Concept Embeddings ───")
    print()

    embeddings, all_signs, all_exponents = load_embeddings()
    id_to_token, token_to_id = load_tokenizer()
    if id_to_token is None:
        print("FATAL: Could not load tokenizer")
        return

    searcher = VocabSearcher(embeddings, id_to_token)
    print()

    # ─── Phase 2: Build Concept Drum ─────────────────────────────────

    print("─── Phase 2: Build Concept Drum ───")
    print()

    # Known country → capital pairs (training)
    capital_train = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Egypt", "Cairo"),
    ]

    # Held-out country → capital pairs (test)
    capital_test = [
        ("Australia", "Canberra"),
        ("Thailand", "Bangkok"),
        ("Poland", "Warsaw"),
        ("Norway", "Oslo"),
        ("Sweden", "Stockholm"),
        ("India", "Delhi"),
        ("Brazil", "Brasilia"),
        ("Korea", "Seoul"),
    ]

    # Language relationship
    language_train = [
        ("France", "French"),
        ("Germany", "German"),
        ("Japan", "Japanese"),
        ("China", "Chinese"),
        ("Spain", "Spanish"),
    ]

    language_test = [
        ("Italy", "Italian"),
        ("Portugal", "Portuguese"),
        ("Russia", "Russian"),
    ]

    # Gender flip relationship
    gender_all = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("brother", "sister"),
        ("son", "daughter"),
        ("husband", "wife"),
    ]
    gender_train = gender_all[:4]  # king→queen, man→woman, boy→girl, father→mother
    gender_test = gender_all[4:]   # brother→sister, son→daughter, husband→wife

    # Antonym relationship
    antonym_all = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("good", "bad"),
        ("old", "young"),
        ("dark", "light"),
        ("long", "short"),
        ("up", "down"),
    ]
    antonym_train = antonym_all[:5]
    antonym_test = antonym_all[5:]

    # City → Language (for multi-hop: country→capital→language)
    city_lang_pairs = [
        ("Paris", "French"),
        ("Berlin", "German"),
        ("Tokyo", "Japanese"),
        ("Beijing", "Chinese"),
        ("Cairo", "Arabic"),
    ]

    # Combined capital dataset for LOO cross-validation
    capital_all = capital_train + capital_test

    # Collect all words we need
    all_words = set()
    for pairs in [capital_train, capital_test, language_train, language_test,
                  gender_all, antonym_all, city_lang_pairs]:
        for a, b in pairs:
            all_words.add(a)
            all_words.add(b)

    # Extra cities/languages for search space validation
    extras = [
        "London", "Rome", "Madrid", "Lisbon", "Moscow", "Vienna",
        "Brussels", "Amsterdam", "Copenhagen", "Helsinki", "Dublin",
        "Athens", "Ankara", "Seoul", "Taipei", "Manila", "Jakarta",
        "English", "Italian", "Portuguese", "Russian", "Arabic",
        "Korean", "Thai", "Polish", "Norwegian", "Swedish",
        "Dutch", "Greek", "Turkish", "Hindi",
    ]
    all_words.update(extras)

    # Dragon Shrimp and compound concept words
    compound_words = [
        "dragon", "shrimp", "lobster",
        "sun", "flower", "sunflower",
        "star", "fish", "starfish",
        "rain", "bow", "rainbow",
        "foot", "ball", "football",
        "sea", "horse", "seahorse",
        "water", "fall", "waterfall",
        "fire", "fly", "firefly",
    ]
    all_words.update(compound_words)

    # Load concepts
    concepts = {}
    missing = []
    for word in sorted(all_words):
        tid, tok = find_token_id(word, token_to_id)
        if tid is not None:
            concepts[word] = Concept(
                name=word,
                token_id=tid,
                token_str=tok,
                embedding=embeddings[tid].copy(),
                phi_signs=all_signs[tid].copy(),
                phi_exponents=all_exponents[tid].copy(),
            )
        else:
            missing.append(word)

    print(f"  Loaded {len(concepts)}/{len(all_words)} concepts")
    if missing:
        print(f"  Missing: {missing}")
    print()

    # ─── Phase 3: Learn Relationship Gates ───────────────────────────

    print("─── Phase 3: Learn Relationship Gates ───")
    print()

    def make_pairs(pair_list):
        """Extract embedding pairs for concepts we found."""
        pairs = []
        for entity_name, answer_name in pair_list:
            if entity_name in concepts and answer_name in concepts:
                pairs.append((
                    concepts[entity_name].embedding,
                    concepts[answer_name].embedding,
                ))
        return pairs

    # Learn capital-of gate
    cap_pairs = make_pairs(capital_train)
    print(f"  Learning 'capital-of' from {len(cap_pairs)} pairs...")
    capital_gate = RelationshipGate.learn("capital-of", cap_pairs)
    print(f"    PRESERVE: {capital_gate.n_preserve:4d} dims  ({capital_gate.n_preserve / 3584 * 100:5.1f}%)")
    print(f"    SHIFT:    {capital_gate.n_shift:4d} dims  ({capital_gate.n_shift / 3584 * 100:5.1f}%)")
    print(f"    FLIP:     {capital_gate.n_flip:4d} dims  ({capital_gate.n_flip / 3584 * 100:5.1f}%)")
    print(f"    COMB:     {capital_gate.n_comb:4d} dims  ({capital_gate.n_comb / 3584 * 100:5.1f}%)")
    print()

    # Learn language-of gate
    lang_pairs = make_pairs(language_train)
    print(f"  Learning 'language-of' from {len(lang_pairs)} pairs...")
    language_gate = RelationshipGate.learn("language-of", lang_pairs)
    print(f"    PRESERVE: {language_gate.n_preserve:4d} dims  ({language_gate.n_preserve / 3584 * 100:5.1f}%)")
    print(f"    SHIFT:    {language_gate.n_shift:4d} dims  ({language_gate.n_shift / 3584 * 100:5.1f}%)")
    print(f"    FLIP:     {language_gate.n_flip:4d} dims  ({language_gate.n_flip / 3584 * 100:5.1f}%)")
    print(f"    COMB:     {language_gate.n_comb:4d} dims  ({language_gate.n_comb / 3584 * 100:5.1f}%)")
    print()

    # ─── Phase 3b: Learn Boom Gates (Integer Space) ────────────────

    print("─── Phase 3b: Learn Boom Gates (Integer Space) ───")
    print()

    def make_int_pairs(pair_list):
        """Extract sign/exponent arrays for concepts we found."""
        e_signs, e_exps, a_signs, a_exps = [], [], [], []
        for entity_name, answer_name in pair_list:
            if entity_name in concepts and answer_name in concepts:
                e_signs.append(concepts[entity_name].phi_signs)
                e_exps.append(concepts[entity_name].phi_exponents)
                a_signs.append(concepts[answer_name].phi_signs)
                a_exps.append(concepts[answer_name].phi_exponents)
        return e_signs, e_exps, a_signs, a_exps

    # Learn capital boom gate — sweep thresholds
    cap_es, cap_ee, cap_as, cap_ae = make_int_pairs(capital_train)

    print("  Boom threshold sweep (capital-of):")
    print(f"  {'spread_t':>8s}  {'BOOM':>6s}  {'NOISE':>6s}  {'SignFlip':>8s}  {'SignKeep':>8s}")
    print("  " + "-" * 50)
    for bt in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        bg = BoomGate.learn("sweep", cap_es, cap_ee, cap_as, cap_ae,
                            boom_spread_threshold=bt)
        print(f"  {bt:>8.1f}  {bg.n_boom:>6d}  {bg.n_noise:>6d}  "
              f"{bg.n_sign_flip:>8d}  {bg.n_sign_keep:>8d}")
    print()

    # Default boom gate
    cap_boom = BoomGate.learn("capital-of-boom", cap_es, cap_ee, cap_as, cap_ae,
                               boom_spread_threshold=2.0)
    spec = cap_boom.boom_spectrum()
    print(f"  Capital boom gate (spread_t=2.0):")
    print(f"    Boom dims:     {spec['boom_dims']:5d}  ({spec['boom_pct']:.1%})")
    print(f"    Sign flip:     {spec['sign_flip_dims']:5d}")
    print(f"    Sign keep:     {spec['sign_keep_dims']:5d}")
    print(f"    Noise dims:    {spec['noise_dims']:5d}")
    print(f"    Boom spread median:  {spec['median_boom_spread']:.2f}")
    print(f"    Noise spread median: {spec['median_noise_spread']:.2f}")
    print(f"    Boom |Δexp| median:  {spec['boom_exp_delta_median']:.1f}")
    print()

    # Language boom gate
    lang_es, lang_ee, lang_as, lang_ae = make_int_pairs(language_train)
    lang_boom = BoomGate.learn("language-of-boom", lang_es, lang_ee, lang_as, lang_ae,
                                boom_spread_threshold=2.0)
    spec_l = lang_boom.boom_spectrum()
    print(f"  Language boom gate (spread_t=2.0):")
    print(f"    Boom dims:     {spec_l['boom_dims']:5d}  ({spec_l['boom_pct']:.1%})")
    print(f"    Sign flip:     {spec_l['sign_flip_dims']:5d}")
    print(f"    Sign keep:     {spec_l['sign_keep_dims']:5d}")
    print(f"    Boom |Δexp| median:  {spec_l['boom_exp_delta_median']:.1f}")
    print()

    # ─── Phase 4: Capital Prediction ─────────────────────────────────

    print("─── Phase 4: Capital Prediction ───")
    print()

    # Vector addition baseline: average delta
    avg_cap_delta = np.mean(
        [concepts[a].embedding - concepts[e].embedding
         for e, a in capital_train if e in concepts and a in concepts],
        axis=0,
    )

    # Analogy baseline: use France→Paris as reference
    ref_e = concepts.get("France")
    ref_a = concepts.get("Paris")
    ref_delta = ref_a.embedding - ref_e.embedding if ref_e and ref_a else None

    # Also learn gate with relaxed params (from sweep: preserve_t=0.5, cv_t=1.5)
    capital_gate_relax = RelationshipGate.learn("capital-of-relax", cap_pairs,
                                                 preserve_threshold=0.5,
                                                 shift_cv_threshold=1.5)
    print(f"  Relaxed gate (pt=0.5, cv=1.5):")
    print(f"    PRESERVE: {capital_gate_relax.n_preserve}  SHIFT: {capital_gate_relax.n_shift}  "
          f"FLIP: {capital_gate_relax.n_flip}  COMB: {capital_gate_relax.n_comb}")
    print()

    # Method labels — focus on boom gate methods vs VecAdd
    methods = [
        "VecAdd", "BoomOnly", "Boom+VA", "BoomWt",
        "SignBoom", "Soft", "GateC1",
    ]
    header = f"  {'Entity':>12s} → {'Answer':>12s}  " + "  ".join(f"{m:>8s}" for m in methods)

    # --- Test set (the real test) ---
    print("  Test pairs (unseen entities):")
    print(header)
    print("  " + "-" * (30 + 10 * len(methods)))

    results_cap = {m: [] for m in methods}

    for entity_name, answer_name in capital_test:
        e = concepts.get(entity_name)
        a = concepts.get(answer_name)
        if not e or not a:
            print(f"  {entity_name:>12s} → {answer_name:>12s}  SKIP")
            continue

        ex = {e.token_id}

        ranks = {}
        ranks["VecAdd"]   = searcher.rank_of(e.embedding + avg_cap_delta, a.token_id, ex)
        ranks["BoomOnly"] = searcher.rank_of(cap_boom.apply_boom_only(e.phi_signs, e.phi_exponents), a.token_id, ex)
        ranks["Boom+VA"]  = searcher.rank_of(cap_boom.apply_boom_plus_va(e.phi_signs, e.phi_exponents, e.embedding, avg_cap_delta), a.token_id, ex)
        ranks["BoomWt"]   = searcher.rank_of(cap_boom.apply_boom_weighted(e.phi_signs, e.phi_exponents, e.embedding, avg_cap_delta), a.token_id, ex)
        ranks["SignBoom"]  = searcher.rank_of(cap_boom.apply_sign_boom(e.phi_signs, e.phi_exponents, e.embedding, avg_cap_delta), a.token_id, ex)
        ranks["Soft"]     = searcher.rank_of(capital_gate.apply_soft(e.embedding, avg_cap_delta, blend_floor=0.3), a.token_id, ex)
        ranks["GateC1"]   = searcher.rank_of(capital_gate.apply(e.embedding, comb_scale=1.0), a.token_id, ex)

        for m in methods:
            results_cap[m].append(ranks[m])

        rank_str = "  ".join(f"r={ranks[m]:<5d}" for m in methods)
        print(f"  {entity_name:>12s} → {answer_name:>12s}  {rank_str}")

    print()

    # Print summary table
    print("  Method comparison (capital test, lower = better):")
    print(f"  {'Method':<12s}  {'Mean':>7s}  {'Median':>7s}  {'Best':>5s}  {'Worst':>6s}")
    print("  " + "-" * 45)
    for m in methods:
        rs = results_cap[m]
        if rs:
            print(f"  {m:<12s}  {np.mean(rs):>7.0f}  {np.median(rs):>7.0f}  "
                  f"{min(rs):>5d}  {max(rs):>6d}")
    print()

    # ─── Phase 5: Language Prediction ────────────────────────────────

    print("─── Phase 5: Language Prediction ───")
    print()

    avg_lang_delta = np.mean(
        [concepts[a].embedding - concepts[e].embedding
         for e, a in language_train if e in concepts and a in concepts],
        axis=0,
    )

    language_gate_relax = RelationshipGate.learn("language-of-relax", lang_pairs,
                                                  preserve_threshold=0.5,
                                                  shift_cv_threshold=1.5)

    lang_methods = ["VecAdd", "BoomOnly", "Boom+VA", "BoomWt", "SignBoom", "Soft", "GateC1"]
    results_lang = {m: [] for m in lang_methods}

    lang_header = f"  {'Entity':>12s} → {'Answer':>12s}  " + "  ".join(f"{m:>8s}" for m in lang_methods)
    print(lang_header)
    print("  " + "-" * (30 + 10 * len(lang_methods)))

    for entity_name, answer_name in language_test:
        e = concepts.get(entity_name)
        a = concepts.get(answer_name)
        if not e or not a:
            print(f"  {entity_name:>12s} → {answer_name:>12s}  SKIP")
            continue

        ex = {e.token_id}

        ranks = {}
        ranks["VecAdd"]   = searcher.rank_of(e.embedding + avg_lang_delta, a.token_id, ex)
        ranks["BoomOnly"] = searcher.rank_of(lang_boom.apply_boom_only(e.phi_signs, e.phi_exponents), a.token_id, ex)
        ranks["Boom+VA"]  = searcher.rank_of(lang_boom.apply_boom_plus_va(e.phi_signs, e.phi_exponents, e.embedding, avg_lang_delta), a.token_id, ex)
        ranks["BoomWt"]   = searcher.rank_of(lang_boom.apply_boom_weighted(e.phi_signs, e.phi_exponents, e.embedding, avg_lang_delta), a.token_id, ex)
        ranks["SignBoom"]  = searcher.rank_of(lang_boom.apply_sign_boom(e.phi_signs, e.phi_exponents, e.embedding, avg_lang_delta), a.token_id, ex)
        ranks["Soft"]     = searcher.rank_of(language_gate.apply_soft(e.embedding, avg_lang_delta, blend_floor=0.3), a.token_id, ex)
        ranks["GateC1"]   = searcher.rank_of(language_gate.apply(e.embedding, comb_scale=1.0), a.token_id, ex)

        for m in lang_methods:
            results_lang[m].append(ranks[m])

        rank_str = "  ".join(f"r={ranks[m]:<5d}" for m in lang_methods)
        print(f"  {entity_name:>12s} → {answer_name:>12s}  {rank_str}")

    print()
    print("  Method comparison (language test, lower = better):")
    print(f"  {'Method':<12s}  {'Mean':>7s}  {'Median':>7s}  {'Best':>5s}  {'Worst':>6s}")
    print("  " + "-" * 45)
    for m in lang_methods:
        rs = results_lang[m]
        if rs:
            print(f"  {m:<12s}  {np.mean(rs):>7.0f}  {np.median(rs):>7.0f}  "
                  f"{min(rs):>5d}  {max(rs):>6d}")
    print()

    # ─── Phase 6: Dragon Shrimp Concept Composition ──────────────────

    print("─── Phase 6: Dragon Shrimp & Compound Concepts ───")
    print()

    compound_tests = [
        ("dragon", "shrimp", "lobster",    "龙虾 = dragon+shrimp"),
        ("sun",    "flower", "sunflower",  "compound word"),
        ("star",   "fish",   "starfish",   "compound word"),
        ("rain",   "bow",    "rainbow",    "compound word"),
        ("foot",   "ball",   "football",   "compound word"),
        ("sea",    "horse",  "seahorse",   "compound word"),
        ("water",  "fall",   "waterfall",  "compound word"),
        ("fire",   "fly",    "firefly",    "compound word"),
    ]

    # Learn compound gate from known pairs
    compound_train_names = [
        ("sun",   "flower", "sunflower"),
        ("star",  "fish",   "starfish"),
        ("rain",  "bow",    "rainbow"),
        ("water", "fall",   "waterfall"),
    ]

    compound_pairs = []
    for a_name, b_name, c_name in compound_train_names:
        if a_name in concepts and b_name in concepts and c_name in concepts:
            # For compound composition, the "entity" is A+B sum, "answer" is C
            sum_ab = concepts[a_name].embedding + concepts[b_name].embedding
            compound_pairs.append((sum_ab, concepts[c_name].embedding))

    compound_gate = None
    if len(compound_pairs) >= 2:
        print(f"  Learning 'compound' gate from {len(compound_pairs)} pairs...")
        compound_gate = RelationshipGate.learn("compound", compound_pairs,
                                                preserve_threshold=0.15,
                                                shift_cv_threshold=0.6)
        print(f"    PRESERVE: {compound_gate.n_preserve}  SHIFT: {compound_gate.n_shift}  "
              f"FLIP: {compound_gate.n_flip}  COMB: {compound_gate.n_comb}")
        print()

    print("  Vector addition (A + B → ?):")
    print(f"  {'A':>10s} + {'B':>10s} → {'Expected':>12s}  {'VecAdd':>7s}  "
          f"{'Gate':>7s}  Top-5")
    print("  " + "-" * 80)

    results_compound = {"vec_add": [], "gate": []}

    for word_a, word_b, expected, desc in compound_tests:
        if word_a not in concepts or word_b not in concepts or expected not in concepts:
            print(f"  {word_a:>10s} + {word_b:>10s} → {expected:>12s}  SKIP ({desc})")
            continue

        ca = concepts[word_a]
        cb = concepts[word_b]
        ce = concepts[expected]
        ex = {ca.token_id, cb.token_id}

        # Vector addition
        vec_sum = ca.embedding + cb.embedding
        r_va = searcher.rank_of(vec_sum, ce.token_id, ex)
        results_compound["vec_add"].append(r_va)

        # Gate composition (if learned)
        r_gate = -1
        if compound_gate is not None:
            gate_composed = compound_gate.apply(vec_sum, comb_scale=0.0)
            r_gate = searcher.rank_of(gate_composed, ce.token_id, ex)
            results_compound["gate"].append(r_gate)

        # Top 5 for vector addition
        top5 = searcher.top_k(vec_sum, k=5, exclude_tids=ex)
        top5_str = ", ".join([f"{t[0]}" for t in top5])

        gate_str = f"r={r_gate:<5d}" if r_gate >= 0 else "  N/A  "
        print(f"  {word_a:>10s} + {word_b:>10s} → {expected:>12s}  "
              f"r={r_va:<5d}  {gate_str}  [{top5_str}]")

    print()

    # ─── Phase 7: Gate Threshold Sweep ───────────────────────────────

    print("─── Phase 7: Gate Parameter Sweep ───")
    print()
    print("  Sweeping preserve_threshold and shift_cv_threshold...")
    print()

    best_mean_rank = float("inf")
    best_params = None

    print(f"  {'pres_t':>6s}  {'cv_t':>6s}  {'PRES':>5s}  {'SHFT':>5s}  "
          f"{'FLIP':>5s}  {'COMB':>5s}  {'mean_r':>7s}  {'med_r':>7s}")
    print("  " + "-" * 65)

    for pres_t in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for cv_t in [0.3, 0.5, 0.7, 1.0, 1.5]:
            gate = RelationshipGate.learn("sweep", cap_pairs,
                                          preserve_threshold=pres_t,
                                          shift_cv_threshold=cv_t)

            # Quick test on capital_test using hybrid (best method so far)
            ranks = []
            for entity_name, answer_name in capital_test:
                e = concepts.get(entity_name)
                a = concepts.get(answer_name)
                if not e or not a:
                    continue
                composed = gate.apply_hybrid(e.embedding, avg_cap_delta)
                r = searcher.rank_of(composed, a.token_id, {e.token_id})
                ranks.append(r)

            if ranks:
                mean_r = np.mean(ranks)
                med_r = np.median(ranks)

                print(f"  {pres_t:>6.2f}  {cv_t:>6.1f}  "
                      f"{gate.n_preserve:>5d}  {gate.n_shift:>5d}  "
                      f"{gate.n_flip:>5d}  {gate.n_comb:>5d}  "
                      f"{mean_r:>7.0f}  {med_r:>7.0f}"
                      + ("  ← BEST" if mean_r < best_mean_rank else ""))

                if mean_r < best_mean_rank:
                    best_mean_rank = mean_r
                    best_params = (pres_t, cv_t)

    print()
    if best_params:
        print(f"  Best parameters: preserve_t={best_params[0]}, cv_t={best_params[1]}")
        print(f"  Best mean rank: {best_mean_rank:.0f}")
    print()

    # ─── Phase 8: Mathematical Structure Probe ───────────────────────

    print("─── Phase 8: Mathematical Structure in Concept Positions ───")
    print()

    # Check φ-structure in entity embeddings
    print("  Entity φ-encoding statistics:")
    print(f"  {'Concept':>12s}  {'exp_μ':>7s}  {'exp_σ':>6s}  {'near_φ^k':>8s}  {'frac+':>6s}")
    print("  " + "-" * 50)

    for name in ["France", "Germany", "Japan", "China", "Paris", "Berlin", "Tokyo"]:
        c = concepts.get(name)
        if not c:
            continue

        exps_f = c.phi_exponents.astype(float)
        exp_mean = np.mean(exps_f)
        exp_std = np.std(exps_f)

        # What fraction of exponents are near integer multiples of PHI_GRID?
        # (values that are near integer powers of φ)
        exp_mod = np.abs(exps_f) % PHI_GRID
        near_integer = np.mean((exp_mod < 5) | (exp_mod > PHI_GRID - 5))

        frac_pos = np.mean(c.phi_signs > 0)

        print(f"  {name:>12s}  {exp_mean:>7.1f}  {exp_std:>6.1f}  "
              f"{near_integer:>8.1%}  {frac_pos:>6.1%}")

    print()

    # Check if the capital-of SHIFT values follow φ^(-k) decay
    shift_mask = capital_gate.dim_types == RelationshipGate.TYPE_SHIFT
    shift_vals = capital_gate.shift_values[shift_mask]

    if len(shift_vals) > 10:
        abs_shifts = np.sort(np.abs(shift_vals))[::-1]
        # Consecutive ratios
        ratios = abs_shifts[:-1] / (abs_shifts[1:] + 1e-20)
        valid = (ratios > 0.5) & (ratios < 5.0)  # Filter outliers
        if np.sum(valid) > 0:
            near_phi = np.mean((ratios[valid] > PHI * 0.85) & (ratios[valid] < PHI * 1.15))
            near_one = np.mean((ratios[valid] > 0.85) & (ratios[valid] < 1.15))

            print(f"  Capital-of SHIFT magnitude decay pattern ({len(shift_vals)} dims):")
            print(f"    Adjacent ratio ≈ φ (±15%): {near_phi:.1%}")
            print(f"    Adjacent ratio ≈ 1 (±15%): {near_one:.1%}")
            print(f"    Median ratio: {np.median(ratios[valid]):.4f}  (φ = {PHI:.4f})")

    print()

    # Cross-concept cosine similarity structure
    print("  Cross-concept cosine similarities:")
    country_names = ["France", "Germany", "Japan", "China", "Egypt"]
    capital_names = ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo"]

    for pair_type, names_a, names_b in [
        ("Country-Country", country_names, country_names),
        ("Capital-Capital", capital_names, capital_names),
        ("Country-Capital (matched)", country_names, capital_names),
    ]:
        sims = []
        for i, na in enumerate(names_a):
            for j, nb in enumerate(names_b):
                if na == nb or (pair_type.startswith("Country-Capital") and i != j):
                    continue
                if pair_type.startswith("Country-Capital") and i != j:
                    continue
                ca = concepts.get(na)
                cb = concepts.get(nb)
                if ca and cb:
                    cos = float(np.dot(ca.embedding, cb.embedding)
                                / (np.linalg.norm(ca.embedding) * np.linalg.norm(cb.embedding) + 1e-20))
                    sims.append(cos)

        if sims:
            print(f"    {pair_type:<35s}  mean={np.mean(sims):.4f}  "
                  f"min={min(sims):.4f}  max={max(sims):.4f}")

    print()

    # ─── Phase 9: Generalization Tests ──────────────────────────────

    print("─── Phase 9: Generalization Tests ───")
    print()

    gen_results = {}

    # --- 9a: Leave-One-Out Cross-Validation on Capitals ---
    print("  9a. Leave-One-Out Cross-Validation (Capitals)")
    print()

    loo_ranks = {"VecAdd": [], "BoomWt": [], "Boom+VA": [], "BoomOnly": []}
    n_cap_all = len(capital_all)

    for hold_idx in range(n_cap_all):
        held_entity, held_answer = capital_all[hold_idx]
        if held_entity not in concepts or held_answer not in concepts:
            continue

        # Train on all except held-out
        loo_train = [capital_all[i] for i in range(n_cap_all) if i != hold_idx]

        # VA delta from LOO train set
        loo_deltas = []
        for e_name, a_name in loo_train:
            if e_name in concepts and a_name in concepts:
                loo_deltas.append(concepts[a_name].embedding - concepts[e_name].embedding)
        if not loo_deltas:
            continue
        loo_avg_delta = np.mean(loo_deltas, axis=0)

        # Boom gate from LOO train set
        loo_es, loo_ee, loo_as, loo_ae = make_int_pairs(loo_train)
        if len(loo_es) < 2:
            continue
        loo_boom = BoomGate.learn("loo", loo_es, loo_ee, loo_as, loo_ae,
                                   boom_spread_threshold=2.0)

        e = concepts[held_entity]
        a = concepts[held_answer]
        ex = {e.token_id}

        loo_ranks["VecAdd"].append(searcher.rank_of(e.embedding + loo_avg_delta, a.token_id, ex))
        loo_ranks["BoomOnly"].append(searcher.rank_of(loo_boom.apply_boom_only(e.phi_signs, e.phi_exponents), a.token_id, ex))
        loo_ranks["Boom+VA"].append(searcher.rank_of(loo_boom.apply_boom_plus_va(e.phi_signs, e.phi_exponents, e.embedding, loo_avg_delta), a.token_id, ex))
        loo_ranks["BoomWt"].append(searcher.rank_of(loo_boom.apply_boom_weighted(e.phi_signs, e.phi_exponents, e.embedding, loo_avg_delta), a.token_id, ex))

    print(f"  {'Method':<12s}  {'mean':>7s}  {'median':>7s}  {'best':>6s}  {'worst':>6s}  {'wins':>5s}")
    print("  " + "-" * 55)
    va_loo = loo_ranks["VecAdd"]
    for method in ["VecAdd", "BoomOnly", "Boom+VA", "BoomWt"]:
        r = loo_ranks[method]
        if r:
            wins = sum(1 for a, b in zip(r, va_loo) if a < b) if method != "VecAdd" else "-"
            print(f"  {method:<12s}  {np.mean(r):>7.0f}  {np.median(r):>7.0f}  "
                  f"{min(r):>6d}  {max(r):>6d}  {wins!s:>5s}")
    gen_results["loo_capitals"] = {k: list(v) for k, v in loo_ranks.items()}
    print()

    # --- 9b: New Relationship Types ---
    print("  9b. New Relationship Types")
    print()

    for rel_name, train_pairs, test_pairs in [
        ("gender-flip", gender_train, gender_test),
        ("antonym", antonym_train, antonym_test),
    ]:
        # Check we have enough concepts
        fl_train = [(e, a) for e, a in train_pairs if e in concepts and a in concepts]
        fl_test = [(e, a) for e, a in test_pairs if e in concepts and a in concepts]

        if len(fl_train) < 2:
            print(f"  {rel_name}: SKIP (only {len(fl_train)} train pairs found)")
            continue

        print(f"  {rel_name}: {len(fl_train)} train, {len(fl_test)} test")

        # VA delta
        rel_avg_delta = np.mean(
            [concepts[a].embedding - concepts[e].embedding for e, a in fl_train],
            axis=0,
        )

        # Boom gate
        rel_es, rel_ee, rel_as, rel_ae = make_int_pairs(fl_train)
        rel_boom = BoomGate.learn(f"{rel_name}-boom", rel_es, rel_ee, rel_as, rel_ae,
                                   boom_spread_threshold=2.0)
        spec_r = rel_boom.boom_spectrum()
        print(f"    Boom: {spec_r['boom_dims']} dims ({spec_r['boom_pct']:.1%})  "
              f"SignFlip: {spec_r['sign_flip_dims']}  SignKeep: {spec_r['sign_keep_dims']}")

        # Test
        rel_ranks = {"VecAdd": [], "BoomWt": [], "Boom+VA": [], "SignBoom": []}
        print(f"    {'Entity':>12s} → {'Expected':>12s}  {'VecAdd':>7s}  {'BoomWt':>7s}  {'Boom+VA':>7s}  {'SignBm':>7s}")
        print("    " + "-" * 65)

        for e_name, a_name in fl_test + fl_train:
            e = concepts[e_name]
            a = concepts[a_name]
            ex = {e.token_id}
            is_test = (e_name, a_name) in fl_test

            r_va = searcher.rank_of(e.embedding + rel_avg_delta, a.token_id, ex)
            r_bw = searcher.rank_of(rel_boom.apply_boom_weighted(e.phi_signs, e.phi_exponents, e.embedding, rel_avg_delta), a.token_id, ex)
            r_bva = searcher.rank_of(rel_boom.apply_boom_plus_va(e.phi_signs, e.phi_exponents, e.embedding, rel_avg_delta), a.token_id, ex)
            r_sb = searcher.rank_of(rel_boom.apply_sign_boom(e.phi_signs, e.phi_exponents, e.embedding, rel_avg_delta), a.token_id, ex)

            tag = "TEST" if is_test else "train"
            print(f"    {e_name:>12s} → {a_name:>12s}  {r_va:>7d}  {r_bw:>7d}  {r_bva:>7d}  {r_sb:>7d}  {tag}")

            if is_test:
                rel_ranks["VecAdd"].append(r_va)
                rel_ranks["BoomWt"].append(r_bw)
                rel_ranks["Boom+VA"].append(r_bva)
                rel_ranks["SignBoom"].append(r_sb)

        if rel_ranks["VecAdd"]:
            print(f"    TEST summary:")
            for method in ["VecAdd", "BoomWt", "Boom+VA", "SignBoom"]:
                r = rel_ranks[method]
                if r:
                    print(f"      {method:<12s}  mean={np.mean(r):7.0f}  median={np.median(r):7.0f}")
        gen_results[rel_name] = {k: list(v) for k, v in rel_ranks.items()}
        print()

    # --- 9c: Multi-Hop Composition (country → capital → language) ---
    print("  9c. Multi-Hop Composition (country → capital → language)")
    print()

    # Learn city→language boom gate
    cl_es, cl_ee, cl_as, cl_ae = make_int_pairs(city_lang_pairs)
    if len(cl_es) >= 2:
        city_lang_boom = BoomGate.learn("city-lang-boom", cl_es, cl_ee, cl_as, cl_ae,
                                         boom_spread_threshold=2.0)
        spec_cl = city_lang_boom.boom_spectrum()
        print(f"  City→Lang boom gate: {spec_cl['boom_dims']} boom dims ({spec_cl['boom_pct']:.1%})")

        # Compose: capital_boom + city_lang_boom via integer addition
        composed_boom = cap_boom.compose_int(city_lang_boom)
        spec_comp = composed_boom.boom_spectrum()
        print(f"  Composed (cap+city_lang) boom gate: {spec_comp['boom_dims']} boom dims ({spec_comp['boom_pct']:.1%})")
        print(f"    (Intersection of individual boom masks)")
        print()

        # Test: country → language via composed gate vs direct language gate
        # Direct: avg delta from country → language pairs
        multihop_tests = [
            ("France", "French"),
            ("Germany", "German"),
            ("Japan", "Japanese"),
            ("China", "Chinese"),
            ("Egypt", "Arabic"),    # training data
            ("Italy", "Italian"),   # unseen
            ("Russia", "Russian"),  # unseen
        ]

        mh_ranks = {"VecAdd(direct)": [], "Boom(direct)": [],
                     "VecAdd(2hop)": [], "Boom(composed)": [], "BoomWt(composed)": []}

        print(f"  {'Country':>12s} → {'Language':>12s}  {'VA_dir':>7s}  {'Bm_dir':>7s}  "
              f"{'VA_2hop':>7s}  {'Bm_comp':>7s}  {'BmWt_comp':>7s}")
        print("  " + "-" * 75)

        for c_name, l_name in multihop_tests:
            if c_name not in concepts or l_name not in concepts:
                continue
            c = concepts[c_name]
            l = concepts[l_name]
            ex = {c.token_id}

            # Direct: country → language (single hop)
            r_va_direct = searcher.rank_of(c.embedding + avg_lang_delta, l.token_id, ex)

            r_boom_direct = searcher.rank_of(
                lang_boom.apply_boom_weighted(c.phi_signs, c.phi_exponents, c.embedding, avg_lang_delta),
                l.token_id, ex)

            # Two-hop VA: country + cap_delta + city_lang_delta
            cl_avg_delta = np.mean(
                [concepts[a].embedding - concepts[e].embedding
                 for e, a in city_lang_pairs if e in concepts and a in concepts],
                axis=0,
            )
            two_hop_va = c.embedding + avg_cap_delta + cl_avg_delta
            r_va_2hop = searcher.rank_of(two_hop_va, l.token_id, ex)

            # Composed boom gate (integer composition)
            r_boom_comp = searcher.rank_of(
                composed_boom.apply_boom_only(c.phi_signs, c.phi_exponents),
                l.token_id, ex)

            r_boomwt_comp = searcher.rank_of(
                composed_boom.apply_boom_weighted(
                    c.phi_signs, c.phi_exponents, c.embedding,
                    avg_cap_delta + cl_avg_delta),
                l.token_id, ex)

            print(f"  {c_name:>12s} → {l_name:>12s}  {r_va_direct:>7d}  {r_boom_direct:>7d}  "
                  f"{r_va_2hop:>7d}  {r_boom_comp:>7d}  {r_boomwt_comp:>7d}")

            mh_ranks["VecAdd(direct)"].append(r_va_direct)
            mh_ranks["Boom(direct)"].append(r_boom_direct)
            mh_ranks["VecAdd(2hop)"].append(r_va_2hop)
            mh_ranks["Boom(composed)"].append(r_boom_comp)
            mh_ranks["BoomWt(composed)"].append(r_boomwt_comp)

        print()
        print(f"  Multi-hop summary:")
        for method in mh_ranks:
            r = mh_ranks[method]
            if r:
                print(f"    {method:<20s}  mean={np.mean(r):7.0f}  median={np.median(r):7.0f}")
        gen_results["multihop"] = {k: list(v) for k, v in mh_ranks.items()}
        print()
    else:
        print("  SKIP: not enough city→language pairs found")
        print()

    # --- 9d: Training Size Ablation ---
    print("  9d. Training Size Ablation (Capitals)")
    print()

    ablation_sizes = [2, 3, 4, 5, 7, 10]
    abl_results = {}

    # Use all capital pairs, split dynamically
    cap_all_valid = [(e, a) for e, a in capital_all if e in concepts and a in concepts]

    if len(cap_all_valid) >= 5:
        print(f"  Total valid pairs: {len(cap_all_valid)}")
        print(f"  {'N_train':>8s}  {'VA_mean':>8s}  {'BmWt_mean':>9s}  {'VA_med':>7s}  {'BmWt_med':>8s}  {'BmWt_wins':>9s}")
        print("  " + "-" * 60)

        for n_train in ablation_sizes:
            if n_train >= len(cap_all_valid):
                continue

            abl_va_ranks = []
            abl_bw_ranks = []

            # Multiple random LOO-style splits for robustness
            # Use all possible splits of size n_train
            all_combos = list(combinations(range(len(cap_all_valid)), n_train))
            # Cap at 20 random combos to keep runtime sane
            if len(all_combos) > 20:
                rng = np.random.RandomState(42)
                combo_indices = rng.choice(len(all_combos), 20, replace=False)
                all_combos = [all_combos[i] for i in combo_indices]

            for combo in all_combos:
                combo_set = set(combo)
                abl_train = [cap_all_valid[i] for i in combo]
                abl_test = [cap_all_valid[i] for i in range(len(cap_all_valid)) if i not in combo_set]

                if not abl_test:
                    continue

                # VA delta
                abl_avg_delta = np.mean(
                    [concepts[a].embedding - concepts[e].embedding for e, a in abl_train],
                    axis=0,
                )

                # Boom gate
                abl_es, abl_ee, abl_as, abl_ae = make_int_pairs(abl_train)
                if len(abl_es) < 2:
                    continue
                abl_boom = BoomGate.learn("ablation", abl_es, abl_ee, abl_as, abl_ae,
                                           boom_spread_threshold=2.0)

                for e_name, a_name in abl_test:
                    e = concepts[e_name]
                    a = concepts[a_name]
                    ex = {e.token_id}
                    abl_va_ranks.append(searcher.rank_of(e.embedding + abl_avg_delta, a.token_id, ex))
                    abl_bw_ranks.append(searcher.rank_of(
                        abl_boom.apply_boom_weighted(e.phi_signs, e.phi_exponents, e.embedding, abl_avg_delta),
                        a.token_id, ex))

            if abl_va_ranks:
                wins = sum(1 for a, b in zip(abl_bw_ranks, abl_va_ranks) if a < b)
                print(f"  {n_train:>8d}  {np.mean(abl_va_ranks):>8.0f}  {np.mean(abl_bw_ranks):>9.0f}  "
                      f"{np.median(abl_va_ranks):>7.0f}  {np.median(abl_bw_ranks):>8.0f}  "
                      f"{wins:>4d}/{len(abl_bw_ranks)}")
                abl_results[n_train] = {
                    "va_mean": float(np.mean(abl_va_ranks)),
                    "boomwt_mean": float(np.mean(abl_bw_ranks)),
                    "n_tests": len(abl_va_ranks),
                }

        gen_results["ablation"] = abl_results
        print()
    else:
        print("  SKIP: not enough capital pairs for ablation")
        print()

    # --- 9e: Generalization Verdict ---
    print("  9e. GENERALIZATION VERDICT")
    print()

    n_relationship_types_tested = 0
    n_where_boom_wins = 0
    for key in gen_results:
        if key in ("loo_capitals", "gender-flip", "antonym"):
            va_ranks = gen_results[key].get("VecAdd", [])
            bw_ranks = gen_results[key].get("BoomWt", [])
            if va_ranks and bw_ranks:
                n_relationship_types_tested += 1
                if np.mean(bw_ranks) <= np.mean(va_ranks):
                    n_where_boom_wins += 1
                    verdict = "BOOM ≤ VA"
                else:
                    verdict = "VA wins"
                print(f"    {key:<20s}  VA mean={np.mean(va_ranks):7.0f}  "
                      f"BoomWt mean={np.mean(bw_ranks):7.0f}  → {verdict}")

    print()
    if n_relationship_types_tested > 0:
        print(f"  Boom gate wins or ties on {n_where_boom_wins}/{n_relationship_types_tested} relationship types")
        if n_where_boom_wins == n_relationship_types_tested:
            print(f"  → GENERALIZES across all tested relationship types")
        elif n_where_boom_wins > 0:
            print(f"  → PARTIALLY generalizes ({n_where_boom_wins}/{n_relationship_types_tested})")
        else:
            print(f"  → Does NOT generalize — boom gate loses on all types")
    print()

    # ─── Summary ─────────────────────────────────────────────────────

    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()

    print("  Gate Classification:")
    for gate_name, gate in [("capital-of", capital_gate), ("language-of", language_gate)]:
        print(f"    {gate_name}:")
        print(f"      PRESERVE {gate.n_preserve:4d}  SHIFT {gate.n_shift:4d}  "
              f"FLIP {gate.n_flip:4d}  COMB {gate.n_comb:4d}")
    print()

    for test_name, res in [("Capital (test)", results_cap),
                            ("Language (test)", results_lang)]:
        if not res.get("VecAdd", res.get("vec_add")):
            continue
        print(f"  {test_name} — rank in {searcher.n_vocab:,} vocab:")
        for key in sorted(res.keys()):
            ranks = res[key]
            if ranks:
                print(f"    {key:<12s}  mean={np.mean(ranks):8.0f}  "
                      f"median={np.median(ranks):8.0f}  "
                      f"best={min(ranks):6d}  worst={max(ranks):6d}")
        print()

    if results_compound.get("vec_add"):
        print(f"  Compound Concepts — rank in {searcher.n_vocab:,} vocab:")
        for method, key in [("Vector Addition", "vec_add"), ("Gate", "gate")]:
            ranks = results_compound.get(key, [])
            if ranks:
                print(f"    {method:<20s}  mean={np.mean(ranks):8.0f}  "
                      f"median={np.median(ranks):8.0f}")
        print()

    print("  KEY QUESTION: Does dimension-aware gating outperform uniform vector addition?")
    print()

    # Determine verdict — compare best gate method vs VecAdd
    va_mean = np.mean(results_cap["VecAdd"]) if results_cap.get("VecAdd") else float("inf")
    best_gate_name = None
    best_gate_mean = float("inf")
    for key in results_cap:
        if key == "VecAdd" or key == "Analogy":
            continue
        ranks = results_cap[key]
        if ranks:
            m = np.mean(ranks)
            if m < best_gate_mean:
                best_gate_mean = m
                best_gate_name = key

    if best_gate_name and best_gate_mean < va_mean:
        ratio = va_mean / (best_gate_mean + 1)
        print(f"  ✓ YES — {best_gate_name} (mean={best_gate_mean:.0f}) beats VecAdd (mean={va_mean:.0f})")
        print(f"    → Gate composition is VIABLE")
    elif best_gate_name and best_gate_mean == va_mean:
        print(f"  ~ TIE — {best_gate_name} and VecAdd perform equally (mean={va_mean:.0f})")
    else:
        print(f"  ✗ NOT YET — VecAdd (mean={va_mean:.0f}) beats best gate {best_gate_name} (mean={best_gate_mean:.0f})")
        print(f"    → Gate classification needs refinement")

    print()

    # ─── Save Results ────────────────────────────────────────────────

    output = {
        "capital_gate": {
            "n_preserve": capital_gate.n_preserve,
            "n_shift": capital_gate.n_shift,
            "n_flip": capital_gate.n_flip,
            "n_comb": capital_gate.n_comb,
        },
        "language_gate": {
            "n_preserve": language_gate.n_preserve,
            "n_shift": language_gate.n_shift,
            "n_flip": language_gate.n_flip,
            "n_comb": language_gate.n_comb,
        },
        "capital_test": results_cap,
        "language_test": results_lang,
        "compound_test": results_compound,
        "generalization": gen_results,
        "best_sweep_params": {
            "preserve_threshold": best_params[0] if best_params else None,
            "shift_cv_threshold": best_params[1] if best_params else None,
            "mean_rank": best_mean_rank if best_mean_rank < float("inf") else None,
        },
    }

    out_file = SCRIPT_DIR / "results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_file}")
    print()


if __name__ == "__main__":
    main()
