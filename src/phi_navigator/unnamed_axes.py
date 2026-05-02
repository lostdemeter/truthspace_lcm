"""
Unnamed Axes - Fully automated semantic axis discovery.

Key insight: The model was trained on vastly more semantic relationships
than we could ever manually label. There are likely hundreds of dimensions
encoding things we can't even guess.

Approach:
1. Discover axes automatically via SVD on transformation patterns
2. Keep axes unnamed (just indices: axis_0, axis_1, ...)
3. Navigate using axis indices, not semantic labels
4. Optionally label axes later based on observed behavior

From Doc 167 (Self-Assembling Navigation):
- "Work in geometric space first, map to language later"
- "Not every position needs a name"
- "Navigate to positions that may not have words"

The self-assembly process:
1. Find token pairs with high sign agreement (60-80%)
2. Extract flip patterns for each pair
3. Apply SVD to find common transformation axes
4. Use these axes for navigation (unnamed)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

PHI = 1.6180339887498949


@dataclass
class UnnamedAxis:
    """
    An unnamed semantic axis discovered from the embedding structure.
    
    The axis is defined by which dimensions flip when traversing it.
    We don't know what it "means" - just that it's a consistent
    transformation pattern in the data.
    """
    index: int                          # Axis number (0, 1, 2, ...)
    flip_pattern: np.ndarray            # Which dims flip (bool array)
    variance_explained: float           # How much variance this axis captures
    discovered_from: List[Tuple[str, str]] = field(default_factory=list)  # Pairs that defined it
    
    # Optional: human-assigned label (can be None forever)
    label: Optional[str] = None
    
    @property
    def n_flips(self) -> int:
        return int(np.sum(self.flip_pattern))
    
    @property
    def flip_ratio(self) -> float:
        return self.n_flips / len(self.flip_pattern)
    
    def apply(self, signs: np.ndarray) -> np.ndarray:
        """Apply this axis transformation to a sign vector."""
        result = signs.copy()
        result[self.flip_pattern] *= -1
        return result
    
    def __repr__(self):
        label_str = f" ({self.label})" if self.label else ""
        return f"Axis_{self.index}{label_str}: {self.n_flips} flips, {self.variance_explained:.1%} var"


class UnnamedAxisDiscovery:
    """
    Discovers unnamed semantic axes from the embedding structure.
    
    The process:
    1. Find token pairs with high sign agreement
    2. Extract flip patterns
    3. Apply SVD to find common axes
    4. Return unnamed axes for navigation
    """
    
    def __init__(self, n_dims: int = 3584):
        self.n_dims = n_dims
        self.axes: List[UnnamedAxis] = []
        
        # Token data
        self.token_signs: Optional[np.ndarray] = None
        self.tokenizer = None
        self.vocab_size: int = 0
        
        # Discovered pairs
        self.discovered_pairs: List[Tuple[int, int, float]] = []  # (id1, id2, agreement)
    
    def load_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        """Load token signs from model embeddings."""
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
        
        print(f"Loaded {self.vocab_size} tokens, {self.n_dims} dimensions")
        
        del model
    
    def sign_agreement(self, id1: int, id2: int) -> float:
        """Compute sign agreement between two tokens."""
        return np.mean(self.token_signs[id1] == self.token_signs[id2])
    
    def discover_pairs(self, min_agreement: float = 0.60, 
                       max_agreement: float = 0.80,
                       sample_size: int = 5000,
                       words_only: bool = True) -> List[Tuple[int, int, float]]:
        """
        Discover token pairs with high sign agreement.
        
        Random pairs have ~50% agreement.
        Semantically related pairs have 60-80% agreement.
        
        Args:
            words_only: If True, only consider word tokens (not punctuation, etc.)
        """
        print(f"Discovering pairs with {min_agreement:.0%}-{max_agreement:.0%} agreement...")
        
        # Filter to word tokens if requested
        if words_only:
            word_ids = []
            for i in range(self.vocab_size):
                token = self.tokenizer.decode([i])
                # Keep tokens that are mostly alphabetic and reasonable length
                clean = token.strip()
                if len(clean) >= 2 and clean.isalpha():
                    word_ids.append(i)
            print(f"Filtered to {len(word_ids)} word tokens")
        else:
            word_ids = list(range(self.vocab_size))
        
        np.random.seed(42)
        pairs_checked = 0
        
        self.discovered_pairs = []
        
        # Check random pairs among word tokens
        for _ in range(sample_size * 20):
            idx1 = np.random.randint(0, len(word_ids))
            idx2 = np.random.randint(0, len(word_ids))
            i, j = word_ids[idx1], word_ids[idx2]
            
            if i != j:
                agreement = self.sign_agreement(i, j)
                if min_agreement <= agreement <= max_agreement:
                    self.discovered_pairs.append((i, j, agreement))
                pairs_checked += 1
        
        # Remove duplicates
        seen = set()
        unique_pairs = []
        for i, j, a in self.discovered_pairs:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((i, j, a))
        
        self.discovered_pairs = unique_pairs
        
        # Sort by agreement (highest first)
        self.discovered_pairs.sort(key=lambda x: -x[2])
        
        print(f"Checked {pairs_checked} pairs, found {len(self.discovered_pairs)} with target agreement")
        
        return self.discovered_pairs
    
    def extract_flip_patterns(self, pairs: List[Tuple[int, int, float]] = None) -> np.ndarray:
        """
        Extract flip patterns from discovered pairs.
        
        Returns matrix of shape (n_pairs, n_dims) where each row is a flip pattern.
        """
        pairs = pairs or self.discovered_pairs
        
        flip_matrix = np.zeros((len(pairs), self.n_dims), dtype=np.float32)
        
        for idx, (id1, id2, _) in enumerate(pairs):
            flip_matrix[idx] = (self.token_signs[id1] != self.token_signs[id2]).astype(np.float32)
        
        return flip_matrix
    
    def discover_axes(self, n_axes: int = 50, 
                      pairs: List[Tuple[int, int, float]] = None) -> List[UnnamedAxis]:
        """
        Discover unnamed axes via SVD on flip patterns.
        
        Args:
            n_axes: Number of axes to discover
            pairs: Pairs to use (default: self.discovered_pairs)
            
        Returns:
            List of UnnamedAxis objects
        """
        pairs = pairs or self.discovered_pairs
        
        if not pairs:
            print("No pairs discovered yet. Run discover_pairs() first.")
            return []
        
        print(f"Discovering {n_axes} axes from {len(pairs)} pairs...")
        
        # Extract flip patterns
        flip_matrix = self.extract_flip_patterns(pairs)
        
        # Center the data
        flip_mean = np.mean(flip_matrix, axis=0)
        flip_centered = flip_matrix - flip_mean
        
        # SVD
        U, S, Vt = np.linalg.svd(flip_centered, full_matrices=False)
        
        # Total variance
        total_var = np.sum(S ** 2)
        
        # Create unnamed axes
        self.axes = []
        
        for i in range(min(n_axes, len(S))):
            # The axis direction is Vt[i]
            axis_direction = Vt[i]
            
            # Convert to flip pattern (threshold at 0)
            # Positive values = flip, negative = don't flip
            flip_pattern = axis_direction > 0
            
            # Variance explained
            var_explained = (S[i] ** 2) / total_var
            
            # Which pairs contributed most to this axis?
            pair_contributions = np.abs(U[:, i])
            top_pair_indices = np.argsort(-pair_contributions)[:5]
            top_pairs = [(self.tokenizer.decode([pairs[idx][0]]),
                         self.tokenizer.decode([pairs[idx][1]]))
                        for idx in top_pair_indices]
            
            axis = UnnamedAxis(
                index=i,
                flip_pattern=flip_pattern,
                variance_explained=var_explained,
                discovered_from=top_pairs,
            )
            
            self.axes.append(axis)
        
        print(f"Discovered {len(self.axes)} axes")
        print(f"Top 5 axes explain {sum(a.variance_explained for a in self.axes[:5]):.1%} of variance")
        
        return self.axes
    
    def navigate(self, source_id: int, axis_indices: List[int], 
                 exclude: Set[int] = None) -> Tuple[int, float]:
        """
        Navigate from source token along specified axes.
        
        Args:
            source_id: Starting token ID
            axis_indices: Which axes to apply (by index)
            exclude: Token IDs to exclude from results
            
        Returns:
            (target_id, agreement) - nearest token after transformation
        """
        exclude = exclude or set()
        
        # Get source signs
        source_signs = self.token_signs[source_id].copy()
        
        # Apply axes
        for axis_idx in axis_indices:
            if axis_idx < len(self.axes):
                source_signs = self.axes[axis_idx].apply(source_signs)
        
        # Find nearest token
        best_id = 0
        best_agreement = 0.0
        
        for i in range(self.vocab_size):
            if i in exclude or i == source_id:
                continue
            
            agreement = np.mean(self.token_signs[i] == source_signs)
            if agreement > best_agreement:
                best_agreement = agreement
                best_id = i
        
        return best_id, best_agreement
    
    def analyze_axis(self, axis_idx: int, n_examples: int = 10):
        """
        Analyze what an axis does by showing example transformations.
        
        This is for human interpretation - the axis works regardless of labeling.
        """
        if axis_idx >= len(self.axes):
            print(f"Axis {axis_idx} not found")
            return
        
        axis = self.axes[axis_idx]
        
        print(f"\n{'='*60}")
        print(f"Axis {axis_idx}: {axis.n_flips} flips, {axis.variance_explained:.1%} variance")
        print(f"{'='*60}")
        
        print(f"\nDiscovered from pairs:")
        for w1, w2 in axis.discovered_from[:5]:
            print(f"  {w1!r} ↔ {w2!r}")
        
        # Test on random tokens
        print(f"\nExample transformations:")
        np.random.seed(axis_idx)
        
        for _ in range(n_examples):
            source_id = np.random.randint(0, min(10000, self.vocab_size))
            target_id, agreement = self.navigate(source_id, [axis_idx])
            
            source_word = self.tokenizer.decode([source_id])
            target_word = self.tokenizer.decode([target_id])
            
            print(f"  {source_word!r} → {target_word!r} ({agreement:.1%})")
    
    def save_axes(self, path: str):
        """Save discovered axes to file."""
        data = {
            'n_dims': self.n_dims,
            'axes': [
                {
                    'index': a.index,
                    'flip_pattern': a.flip_pattern.tolist(),
                    'variance_explained': a.variance_explained,
                    'discovered_from': a.discovered_from,
                    'label': a.label,
                }
                for a in self.axes
            ]
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {len(self.axes)} axes to {path}")
    
    def load_axes(self, path: str):
        """Load axes from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.n_dims = data['n_dims']
        self.axes = [
            UnnamedAxis(
                index=a['index'],
                flip_pattern=np.array(a['flip_pattern']),
                variance_explained=a['variance_explained'],
                discovered_from=a['discovered_from'],
                label=a.get('label'),
            )
            for a in data['axes']
        ]
        
        print(f"Loaded {len(self.axes)} axes from {path}")


def test_unnamed_axes():
    """Test the unnamed axis discovery."""
    print("=" * 60)
    print("Testing Unnamed Axis Discovery")
    print("=" * 60)
    
    discovery = UnnamedAxisDiscovery()
    discovery.load_from_model()
    
    # Use lower threshold since random word pairs max at ~56%
    # Anything above 54% is in top 0.6% - likely semantic
    pairs = discovery.discover_pairs(
        min_agreement=0.54, 
        max_agreement=0.80, 
        sample_size=10000,  # More samples to find rare high-agreement pairs
        words_only=True
    )
    
    # Show some discovered pairs
    print("\nSample discovered pairs (sorted by agreement):")
    for id1, id2, agreement in pairs[:30]:
        w1 = discovery.tokenizer.decode([id1])
        w2 = discovery.tokenizer.decode([id2])
        print(f"  {w1!r} ↔ {w2!r}: {agreement:.1%}")
    
    if len(pairs) < 10:
        print("\nNot enough pairs found. Try lowering threshold or increasing sample size.")
        return
    
    # Discover axes
    axes = discovery.discover_axes(n_axes=20)
    
    # Analyze top axes
    for i in range(min(5, len(axes))):
        discovery.analyze_axis(i, n_examples=5)
    
    # Test navigation
    print("\n" + "=" * 60)
    print("Testing Navigation with Unnamed Axes")
    print("=" * 60)
    
    test_words = ["king", "happy", "run", "big"]
    
    for word in test_words:
        try:
            source_id = discovery.tokenizer.encode(word, add_special_tokens=False)[0]
            
            print(f"\n{word}:")
            for axis_idx in range(min(3, len(axes))):
                target_id, agreement = discovery.navigate(source_id, [axis_idx])
                target_word = discovery.tokenizer.decode([target_id])
                print(f"  + axis_{axis_idx} → {target_word!r} ({agreement:.1%})")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_unnamed_axes()
    else:
        print("Usage: python unnamed_axes.py --test")
        print("\nThis module discovers semantic axes automatically from the")
        print("embedding structure. Axes are unnamed - just indices.")
        print("Labels can be assigned later based on observed behavior.")
