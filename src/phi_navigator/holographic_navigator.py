"""
Holographic Navigator - Navigation using holographic projection and error LUTs.

Key insights from DA2 and Additive Error Stereo:

1. DA2 (Doc 125):
   - 32 weights at full resolution = 99.98% accuracy
   - φ-grid conversion turns multiplication into integer addition
   - 16KB LUT for φ^(e/k) values
   - The "intelligence" is in the geometric structure

2. Additive Error Stereo:
   - Errors as signals, not artifacts
   - Holes are negligible (6.2% of error) - can be zeroed
   - 92.3% of error from "perfect regions" - structure carries info
   - Holographic projection - infer missing info from structure

Application to navigation:
- Store transformation flip patterns as LUT (like DA2's 16KB LUT)
- Use holographic projection - infer full transform from key dims
- Ignore low-variance dims (like zeroing holes in stereo)
- The "error" between tokens IS the semantic transformation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import json

PHI = 1.6180339887498949


@dataclass
class TransformationLUT:
    """
    Lookup table for a semantic transformation.
    
    Like DA2's 16KB LUT, but for semantic transformations.
    Stores the flip pattern and example pairs.
    """
    name: str  # Can be unnamed (e.g., "transform_0")
    flip_pattern: np.ndarray  # Which dims flip (bool, n_dims)
    key_dims: np.ndarray  # Top dims by importance (indices)
    variance_explained: float
    example_pairs: List[Tuple[str, str]] = field(default_factory=list)
    
    @property
    def n_flips(self) -> int:
        return int(np.sum(self.flip_pattern))
    
    @property
    def n_key_dims(self) -> int:
        return len(self.key_dims)
    
    def apply(self, signs: np.ndarray) -> np.ndarray:
        """Apply this transformation to a sign vector."""
        result = signs.copy()
        result[self.flip_pattern] *= -1
        return result
    
    def apply_key_only(self, signs: np.ndarray) -> np.ndarray:
        """Apply transformation to key dims only (holographic)."""
        result = signs.copy()
        key_flips = self.flip_pattern[self.key_dims]
        result[self.key_dims[key_flips]] *= -1
        return result
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'flip_pattern': self.flip_pattern.tolist(),
            'key_dims': self.key_dims.tolist(),
            'variance_explained': self.variance_explained,
            'example_pairs': self.example_pairs,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TransformationLUT':
        return cls(
            name=d['name'],
            flip_pattern=np.array(d['flip_pattern']),
            key_dims=np.array(d['key_dims']),
            variance_explained=d['variance_explained'],
            example_pairs=d.get('example_pairs', []),
        )


class HolographicNavigator:
    """
    Navigator using holographic projection and transformation LUTs.
    
    The approach:
    1. Build LUTs from known semantic pairs (like DA2's 32 weights)
    2. For navigation, apply LUT to key dims only (holographic projection)
    3. Infer full transformation from key dims (like stereo's error encoding)
    4. Ignore low-variance dims (like zeroing holes)
    """
    
    def __init__(self, n_key_dims: int = 32):
        """
        Args:
            n_key_dims: Number of key dimensions to use (like DA2's 32 features)
        """
        self.n_key_dims = n_key_dims
        self.n_dims = 3584
        
        # Token data
        self.token_signs: Optional[np.ndarray] = None
        self.tokenizer = None
        self.vocab_size: int = 0
        
        # Transformation LUTs
        self.luts: Dict[str, TransformationLUT] = {}
        
        # Dimension importance (by variance)
        self.dim_importance: Optional[np.ndarray] = None
        self.dim_order: Optional[np.ndarray] = None
    
    def load_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        """Load token signs and compute dimension importance."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        emb = model.model.embed_tokens.weight.data.numpy()
        self.n_dims = emb.shape[1]
        self.vocab_size = len(emb)
        
        # Extract signs
        self.token_signs = np.sign(emb).astype(np.int8)
        self.token_signs[self.token_signs == 0] = 1
        
        # Compute dimension importance by variance
        variances = np.var(emb, axis=0)
        self.dim_order = np.argsort(-variances)
        self.dim_importance = variances[self.dim_order]
        
        # Normalize importance (like φ-Zipf weighting)
        self.dim_importance = self.dim_importance / self.dim_importance.sum()
        
        print(f"Loaded {self.vocab_size} tokens, {self.n_dims} dimensions")
        print(f"Top {self.n_key_dims} dims capture {self.dim_importance[:self.n_key_dims].sum():.1%} of variance")
        
        del model
    
    def build_lut_from_pairs(self, name: str, 
                             pairs: List[Tuple[str, str]]) -> TransformationLUT:
        """
        Build a transformation LUT from example pairs.
        
        Like DA2's linear fit on 32 features, but for sign flips.
        """
        flip_patterns = []
        
        for w1, w2 in pairs:
            try:
                id1 = self.tokenizer.encode(w1, add_special_tokens=False)[0]
                id2 = self.tokenizer.encode(w2, add_special_tokens=False)[0]
                
                flip = (self.token_signs[id1] != self.token_signs[id2])
                flip_patterns.append(flip)
            except Exception as e:
                print(f"  Skipping {w1}↔{w2}: {e}")
        
        if not flip_patterns:
            raise ValueError(f"No valid pairs for {name}")
        
        # Stack and find common pattern
        flip_matrix = np.array(flip_patterns, dtype=np.float32)
        
        # The "common core" is dims that flip in most pairs
        flip_frequency = np.mean(flip_matrix, axis=0)
        
        # Threshold: flip if >50% of pairs flip this dim
        common_flip = flip_frequency > 0.5
        
        # Find key dims: highest flip frequency among top-variance dims
        key_dims = self.dim_order[:self.n_key_dims]
        
        # Compute variance explained (like SVD)
        # How much of the flip pattern is captured by the common core?
        reconstructed = common_flip.astype(float)
        errors = flip_matrix - reconstructed
        total_var = np.var(flip_matrix)
        residual_var = np.var(errors)
        variance_explained = 1 - (residual_var / (total_var + 1e-10))
        
        lut = TransformationLUT(
            name=name,
            flip_pattern=common_flip,
            key_dims=key_dims,
            variance_explained=variance_explained,
            example_pairs=pairs,
        )
        
        self.luts[name] = lut
        
        print(f"Built LUT '{name}': {lut.n_flips} flips, {variance_explained:.1%} variance explained")
        
        return lut
    
    def build_standard_luts(self):
        """Build LUTs for standard semantic transformations."""
        
        standard_transforms = {
            'gender': [
                ('he', 'she'), ('him', 'her'), ('his', 'her'),
                ('man', 'woman'), ('boy', 'girl'),
                ('king', 'queen'), ('prince', 'princess'),
                ('father', 'mother'), ('brother', 'sister'),
            ],
            'number': [
                ('was', 'were'), ('is', 'are'), ('has', 'have'),
                ('do', 'does'), ('go', 'goes'),
                ('I', 'we'), ('my', 'our'), ('me', 'us'),
            ],
            'tense': [
                ('make', 'made'), ('run', 'ran'), ('go', 'went'),
                ('see', 'saw'), ('take', 'took'), ('give', 'gave'),
                ('come', 'came'), ('know', 'knew'),
            ],
            'antonym': [
                ('happy', 'sad'), ('big', 'small'), ('hot', 'cold'),
                ('good', 'bad'), ('fast', 'slow'), ('high', 'low'),
                ('light', 'dark'), ('old', 'new'),
            ],
        }
        
        print("Building standard transformation LUTs...")
        for name, pairs in standard_transforms.items():
            try:
                self.build_lut_from_pairs(name, pairs)
            except Exception as e:
                print(f"  Failed to build {name}: {e}")
    
    def navigate(self, source_word: str, transform_name: str,
                 use_holographic: bool = True) -> Tuple[str, float]:
        """
        Navigate from source word using a transformation LUT.
        
        Args:
            source_word: Starting word
            transform_name: Which LUT to apply
            use_holographic: If True, only transform key dims (faster, approximate)
            
        Returns:
            (target_word, agreement)
        """
        if transform_name not in self.luts:
            raise ValueError(f"Unknown transform: {transform_name}")
        
        lut = self.luts[transform_name]
        
        source_id = self.tokenizer.encode(source_word, add_special_tokens=False)[0]
        source_signs = self.token_signs[source_id].copy()
        
        # Apply transformation
        if use_holographic:
            target_signs = lut.apply_key_only(source_signs)
        else:
            target_signs = lut.apply(source_signs)
        
        # Find nearest token
        best_id = 0
        best_agreement = 0.0
        
        for i in range(self.vocab_size):
            if i == source_id:
                continue
            
            agreement = np.mean(self.token_signs[i] == target_signs)
            if agreement > best_agreement:
                best_agreement = agreement
                best_id = i
        
        target_word = self.tokenizer.decode([best_id])
        
        return target_word, best_agreement
    
    def navigate_holographic(self, source_word: str, transform_name: str) -> Tuple[str, float]:
        """
        Navigate using holographic projection.
        
        Only transforms key dims, then finds token that matches
        on those dims while inferring the rest.
        
        This is like the stereo approach: use the "error" (flip pattern)
        on key dims, let the structure infer the rest.
        """
        if transform_name not in self.luts:
            raise ValueError(f"Unknown transform: {transform_name}")
        
        lut = self.luts[transform_name]
        
        source_id = self.tokenizer.encode(source_word, add_special_tokens=False)[0]
        source_signs = self.token_signs[source_id].copy()
        
        # Get key dims
        key_dims = lut.key_dims
        key_flips = lut.flip_pattern[key_dims]
        
        # Target signs on key dims
        target_key_signs = source_signs[key_dims].copy()
        target_key_signs[key_flips] *= -1
        
        # Find token that best matches on key dims
        # (holographic: the rest is inferred from structure)
        best_id = 0
        best_key_match = 0
        best_full_agreement = 0.0
        
        for i in range(self.vocab_size):
            if i == source_id:
                continue
            
            # Match on key dims only
            token_key_signs = self.token_signs[i, key_dims]
            key_match = np.sum(token_key_signs == target_key_signs)
            
            if key_match > best_key_match:
                best_key_match = key_match
                best_id = i
                best_full_agreement = np.mean(self.token_signs[i] == source_signs)
            elif key_match == best_key_match:
                # Tie-break by full agreement
                full_agreement = np.mean(self.token_signs[i] == source_signs)
                if full_agreement > best_full_agreement:
                    best_id = i
                    best_full_agreement = full_agreement
        
        target_word = self.tokenizer.decode([best_id])
        key_agreement = best_key_match / len(key_dims)
        
        return target_word, key_agreement
    
    def test_lut(self, transform_name: str):
        """Test a LUT on its example pairs and new words."""
        if transform_name not in self.luts:
            print(f"Unknown transform: {transform_name}")
            return
        
        lut = self.luts[transform_name]
        
        print(f"\n{'='*60}")
        print(f"Testing LUT: {transform_name}")
        print(f"{'='*60}")
        print(f"Flips: {lut.n_flips}, Key dims: {lut.n_key_dims}")
        print(f"Variance explained: {lut.variance_explained:.1%}")
        
        # Test on example pairs
        print(f"\nExample pairs (training data):")
        correct = 0
        for w1, w2 in lut.example_pairs:
            try:
                result, agreement = self.navigate_holographic(w1, transform_name)
                match = "✓" if result.strip().lower() == w2.lower() else "✗"
                if match == "✓":
                    correct += 1
                print(f"  {w1:10} → {result:10} (expected: {w2:10}) {match} ({agreement:.1%})")
            except:
                print(f"  {w1:10} → error")
        
        print(f"\nAccuracy on training pairs: {correct}/{len(lut.example_pairs)}")
        
        # Test on new words (generalization)
        print(f"\nGeneralization test:")
        if transform_name == 'gender':
            test_words = ['actor', 'uncle', 'nephew', 'husband', 'son']
            expected = ['actress', 'aunt', 'niece', 'wife', 'daughter']
        elif transform_name == 'tense':
            test_words = ['eat', 'drink', 'write', 'read', 'speak']
            expected = ['ate', 'drank', 'wrote', 'read', 'spoke']
        elif transform_name == 'antonym':
            test_words = ['up', 'in', 'open', 'start', 'love']
            expected = ['down', 'out', 'close', 'end', 'hate']
        else:
            test_words = []
            expected = []
        
        for w1, w2 in zip(test_words, expected):
            try:
                result, agreement = self.navigate_holographic(w1, transform_name)
                match = "✓" if result.strip().lower() == w2.lower() else "✗"
                print(f"  {w1:10} → {result:10} (expected: {w2:10}) {match} ({agreement:.1%})")
            except:
                print(f"  {w1:10} → error")
    
    def save_luts(self, path: str):
        """Save all LUTs to file."""
        data = {
            'n_key_dims': self.n_key_dims,
            'n_dims': self.n_dims,
            'luts': {name: lut.to_dict() for name, lut in self.luts.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.luts)} LUTs to {path}")
    
    def load_luts(self, path: str):
        """Load LUTs from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.n_key_dims = data['n_key_dims']
        self.n_dims = data['n_dims']
        self.luts = {
            name: TransformationLUT.from_dict(lut_data)
            for name, lut_data in data['luts'].items()
        }
        
        print(f"Loaded {len(self.luts)} LUTs from {path}")


def test_holographic_navigator():
    """Test the holographic navigator."""
    print("=" * 60)
    print("Testing Holographic Navigator")
    print("=" * 60)
    
    navigator = HolographicNavigator(n_key_dims=32)
    navigator.load_from_model()
    
    # Build standard LUTs
    navigator.build_standard_luts()
    
    # Test each LUT
    for transform_name in navigator.luts:
        navigator.test_lut(transform_name)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_holographic_navigator()
    else:
        print("Usage: python holographic_navigator.py --test")
        print("\nThis module implements navigation using holographic projection")
        print("and transformation LUTs, inspired by DA2 and Additive Error Stereo.")
