"""
Geodesic Navigator - Pure geometric navigation using φ-Zipf duality.

From Doc 039 (φ-Zipf Duality):
- φ^n for encoding (outward expansion)
- φ^(-n) for weighting (inward contraction)
- Same fractal, opposite directions
- The structure IS the navigation

From Doc 047 (Geodesic Generation):
- Generation = walking through concept space
- Not token prediction, but geodesic paths
- φ-dial controls direction and depth
- Grammar applied only at projection (end)

From Doc 141 (Irreducible Shape):
- Signs define 3584 critical lines (hyperplanes)
- Each sign = which side of a critical line
- This IS the geometric structure

The key insight: Navigation is φ-weighted geodesic walking.
- Each dimension has a φ^(-rank) importance weight
- Walking = moving along weighted directions
- The path IS the computation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
INV_PHI = 1.0 / PHI


@dataclass
class PhiWeightedPosition:
    """
    Position in φ-weighted concept space.
    
    Each dimension has:
    - sign: which side of the critical line (+1/-1)
    - weight: φ^(-rank) importance (geometric, not statistical)
    
    The position IS the concept - no separate "embedding".
    """
    signs: np.ndarray      # (dim,) int8, values in {-1, +1}
    weights: np.ndarray    # (dim,) float32, φ^(-rank) per dimension
    
    @classmethod
    def from_signs_with_phi_weights(cls, signs: np.ndarray, 
                                     dim_ranks: np.ndarray) -> 'PhiWeightedPosition':
        """
        Create position from signs with φ-based dimension weights.
        
        dim_ranks: rank of each dimension (1 = most important)
        weights: φ^(-rank) for each dimension
        """
        weights = np.power(PHI, -dim_ranks.astype(np.float32))
        return cls(signs=signs.astype(np.int8), weights=weights)
    
    @classmethod
    def from_embedding(cls, embedding: np.ndarray) -> 'PhiWeightedPosition':
        """
        Convert embedding to φ-weighted position.
        
        Signs come from the embedding values.
        Weights come from φ^(-rank) where rank is based on |embedding| magnitude.
        """
        signs = np.sign(embedding).astype(np.int8)
        signs[signs == 0] = 1
        
        # Rank dimensions by magnitude (largest = rank 1)
        magnitudes = np.abs(embedding)
        ranks = np.argsort(np.argsort(-magnitudes)) + 1  # 1-indexed ranks
        
        weights = np.power(PHI, -ranks.astype(np.float32))
        
        return cls(signs=signs, weights=weights)
    
    def weighted_position(self) -> np.ndarray:
        """Get the weighted position vector: sign × weight."""
        return self.signs.astype(np.float32) * self.weights
    
    def phi_similarity(self, other: 'PhiWeightedPosition') -> float:
        """
        φ-weighted similarity (geometric, not statistical).
        
        This is NOT Hamming distance. It's:
        sum(sign_agreement × φ^(-rank))
        
        Dimensions where signs agree contribute positively.
        More important dimensions (lower rank) contribute more.
        """
        agreement = (self.signs == other.signs).astype(np.float32)
        disagreement = 1.0 - agreement
        
        # Weighted sum: agreement contributes +weight, disagreement -weight
        score = np.sum(agreement * self.weights) - np.sum(disagreement * self.weights)
        
        # Normalize by total weight
        total_weight = np.sum(self.weights)
        return score / total_weight
    
    def geodesic_step(self, direction: 'PhiWeightedPosition', 
                      step_size: float = 0.1) -> 'PhiWeightedPosition':
        """
        Take a geodesic step in the given direction.
        
        In sign space, a "step" means flipping signs to move toward
        the direction. The key insight from φ-Zipf duality:
        
        - Higher-weight dimensions (low rank) should flip FIRST
        - They contribute more to similarity, so flipping them
          moves us faster toward the target
        
        step_size controls how many dimensions to flip per step.
        We flip the highest-weight disagreeing dimensions.
        """
        # Which dimensions disagree with direction?
        disagree = (self.signs != direction.signs)
        
        if not np.any(disagree):
            # Already at target
            return PhiWeightedPosition(signs=self.signs.copy(), 
                                       weights=self.weights.copy())
        
        # Get weights of disagreeing dimensions
        disagree_weights = np.where(disagree, self.weights, 0.0)
        
        # How many to flip? Proportional to step_size and number disagreeing
        n_disagree = np.sum(disagree)
        n_to_flip = max(1, int(n_disagree * step_size))
        
        # Flip the highest-weight disagreeing dimensions
        # (These contribute most to similarity)
        flip_indices = np.argsort(-disagree_weights)[:n_to_flip]
        
        new_signs = self.signs.copy()
        # Only flip if actually disagreeing
        for idx in flip_indices:
            if disagree[idx]:
                new_signs[idx] *= -1
        
        return PhiWeightedPosition(signs=new_signs, weights=self.weights.copy())


class GeodesicPath:
    """
    A geodesic path through concept space.
    
    From Doc 047: "Generation = Walking through concept space"
    
    The path is a sequence of positions, each connected by
    φ-weighted geodesic steps.
    """
    
    def __init__(self, start: PhiWeightedPosition):
        self.positions: List[PhiWeightedPosition] = [start]
        self.directions: List[PhiWeightedPosition] = []
    
    def walk_toward(self, target: PhiWeightedPosition, 
                    n_steps: int = 10) -> 'GeodesicPath':
        """
        Walk from current position toward target.
        
        This creates a geodesic path - the "shortest" path
        in φ-weighted space.
        """
        current = self.positions[-1]
        step_size = 1.0 / n_steps
        
        for _ in range(n_steps):
            # Direction is the target
            next_pos = current.geodesic_step(target, step_size)
            self.positions.append(next_pos)
            self.directions.append(target)
            current = next_pos
        
        return self
    
    def endpoint(self) -> PhiWeightedPosition:
        """Get the final position on the path."""
        return self.positions[-1]
    
    def length(self) -> int:
        """Number of steps in the path."""
        return len(self.positions) - 1


class GeodesicNavigator:
    """
    Navigator using φ-weighted geodesic paths.
    
    Architecture:
    1. Tokens → φ-weighted positions (signs + φ^(-rank) weights)
    2. Navigation → geodesic walking through concept space
    3. Output → nearest position in token space
    
    This is purely geometric:
    - No statistical distance (Hamming)
    - No learned parameters
    - φ-weighting is the navigation rule
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Token positions in φ-weighted space
        self.token_positions: Optional[List[PhiWeightedPosition]] = None
        
        # Dimension ranks (computed from embedding statistics)
        self.dim_ranks: Optional[np.ndarray] = None
        
        # Layer transformations as geodesic directions
        self.layer_directions: List[Dict[str, PhiWeightedPosition]] = []
        
        # Tokenizer
        self.tokenizer = None
        
        # Config
        self.hidden_dim = None
        self.vocab_size = None
    
    def compute_dimension_ranks(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute dimension ranks based on variance across vocabulary.
        
        Dimensions with higher variance are more "important" for
        distinguishing tokens → lower rank → higher φ^(-rank) weight.
        
        This is computed once from embeddings, not learned.
        """
        variances = np.var(embeddings, axis=0)
        ranks = np.argsort(np.argsort(-variances)) + 1  # 1-indexed, highest var = rank 1
        return ranks
    
    def convert_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                           max_layers: int = 28):
        """Convert model to geodesic representation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Get embeddings and compute dimension ranks
        print("Computing dimension ranks from embeddings...")
        emb_weight = model.model.embed_tokens.weight.data.numpy()
        self.dim_ranks = self.compute_dimension_ranks(emb_weight)
        
        # Convert embeddings to φ-weighted positions
        print("Converting embeddings to φ-weighted positions...")
        self.token_positions = []
        for i in range(self.vocab_size):
            signs = np.sign(emb_weight[i]).astype(np.int8)
            signs[signs == 0] = 1
            pos = PhiWeightedPosition.from_signs_with_phi_weights(signs, self.dim_ranks)
            self.token_positions.append(pos)
        
        # Convert layer weights to geodesic directions
        # Each layer defines a "direction" in concept space
        n_layers = min(max_layers, len(model.model.layers))
        print(f"Converting {n_layers} layers to geodesic directions...")
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            layer_dict = {}
            
            # MLP gate defines the primary transformation direction
            gate_weight = layer.mlp.gate_proj.weight.data.numpy()
            # Average across output dimensions to get a single direction
            gate_direction = np.mean(gate_weight, axis=0)
            gate_signs = np.sign(gate_direction).astype(np.int8)
            gate_signs[gate_signs == 0] = 1
            layer_dict['gate'] = PhiWeightedPosition.from_signs_with_phi_weights(
                gate_signs, self.dim_ranks
            )
            
            # Down projection defines the output direction
            down_weight = layer.mlp.down_proj.weight.data.numpy()
            down_direction = np.mean(down_weight, axis=1)  # Average across input
            down_signs = np.sign(down_direction).astype(np.int8)
            down_signs[down_signs == 0] = 1
            layer_dict['down'] = PhiWeightedPosition.from_signs_with_phi_weights(
                down_signs, self.dim_ranks
            )
            
            self.layer_directions.append(layer_dict)
            
            if layer_idx % 5 == 0:
                print(f"  Layer {layer_idx} converted")
        
        print(f"Converted {n_layers} layers")
        
        # Show φ-weight distribution
        weights = np.power(PHI, -self.dim_ranks.astype(np.float32))
        print(f"\nφ-weight statistics:")
        print(f"  Max weight (rank 1): {weights.max():.6f}")
        print(f"  Min weight (rank {self.hidden_dim}): {weights.min():.10f}")
        print(f"  Total weight: {weights.sum():.4f}")
        
        del model
    
    def get_token_position(self, token_id: int) -> PhiWeightedPosition:
        """Get φ-weighted position for a token."""
        return self.token_positions[token_id]
    
    def find_nearest_token(self, position: PhiWeightedPosition) -> Tuple[int, float]:
        """
        Find token with highest φ-similarity to position.
        
        This is geometric nearest-neighbor, not statistical.
        """
        best_idx = 0
        best_score = float('-inf')
        
        for i, token_pos in enumerate(self.token_positions):
            score = position.phi_similarity(token_pos)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx, best_score
    
    def navigate_layer(self, position: PhiWeightedPosition, 
                       layer_idx: int) -> PhiWeightedPosition:
        """
        Navigate through a layer via geodesic walking.
        
        The layer's gate direction defines where to walk.
        The layer's down direction defines the output mapping.
        """
        layer = self.layer_directions[layer_idx]
        
        # Walk toward gate direction (the transformation)
        path = GeodesicPath(position)
        path.walk_toward(layer['gate'], n_steps=5)
        
        # Then walk toward down direction (the output)
        path.walk_toward(layer['down'], n_steps=5)
        
        return path.endpoint()
    
    def find_semantic_neighbors(self, token_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        Find k nearest tokens in φ-weighted space.
        
        This is the core geometric operation: given a position,
        find the closest positions in the vocabulary.
        """
        position = self.get_token_position(token_id)
        
        scores = []
        for i, token_pos in enumerate(self.token_positions):
            if i != token_id:
                score = position.phi_similarity(token_pos)
                scores.append((i, score))
        
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
    
    def geodesic_generate(self, prompt: str, max_tokens: int = 20) -> str:
        """
        Generate using geodesic walking through concept space.
        
        From Doc 047: Generation = walking geodesics, not token prediction.
        
        Instead of transforming through layers, we:
        1. Start at the prompt's position
        2. Walk toward semantically related positions
        3. Project the path to tokens
        """
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Start at last token's position
        current_pos = self.get_token_position(token_ids[-1])
        
        generated = []
        visited = set(token_ids)
        
        for _ in range(max_tokens):
            # Find nearest unvisited token
            best_token = None
            best_score = float('-inf')
            
            for i, token_pos in enumerate(self.token_positions):
                if i not in visited:
                    score = current_pos.phi_similarity(token_pos)
                    if score > best_score:
                        best_score = score
                        best_token = i
            
            if best_token is None:
                break
            
            # Walk toward this token
            target_pos = self.get_token_position(best_token)
            path = GeodesicPath(current_pos)
            path.walk_toward(target_pos, n_steps=5)
            current_pos = path.endpoint()
            
            token_ids.append(best_token)
            visited.add(best_token)
            generated.append((best_token, best_score))
            
            if best_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(token_ids)
    
    def navigate(self, token_ids: List[int]) -> PhiWeightedPosition:
        """Navigate through model using geodesic paths."""
        # Start at last token's position
        position = self.get_token_position(token_ids[-1])
        
        # Walk through each layer
        for layer_idx in range(len(self.layer_directions)):
            position = self.navigate_layer(position, layer_idx)
        
        return position
    
    def predict_next_token(self, token_ids: List[int]) -> Tuple[int, float]:
        """Predict next token via geodesic navigation."""
        output_position = self.navigate(token_ids)
        return self.find_nearest_token(output_position)
    
    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text using geodesic navigation."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        for _ in range(max_tokens):
            next_token, score = self.predict_next_token(token_ids)
            token_ids.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(token_ids)


def test_phi_weighted_position():
    """Test PhiWeightedPosition."""
    print("=" * 60)
    print("Testing PhiWeightedPosition")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 64
    
    # Create two positions
    signs1 = np.random.choice([-1, 1], size=dim).astype(np.int8)
    signs2 = np.random.choice([-1, 1], size=dim).astype(np.int8)
    ranks = np.arange(1, dim + 1)  # Rank 1 to 64
    
    pos1 = PhiWeightedPosition.from_signs_with_phi_weights(signs1, ranks)
    pos2 = PhiWeightedPosition.from_signs_with_phi_weights(signs2, ranks)
    
    print(f"Dimension: {dim}")
    print(f"Weight range: [{pos1.weights.min():.10f}, {pos1.weights.max():.6f}]")
    print(f"Total weight: {pos1.weights.sum():.6f}")
    
    # φ-similarity
    sim = pos1.phi_similarity(pos2)
    print(f"\nφ-similarity between random positions: {sim:.4f}")
    
    # Self-similarity should be 1.0
    self_sim = pos1.phi_similarity(pos1)
    print(f"Self-similarity: {self_sim:.4f}")
    
    # Opposite should be -1.0
    opposite = PhiWeightedPosition(signs=-pos1.signs, weights=pos1.weights)
    opp_sim = pos1.phi_similarity(opposite)
    print(f"Opposite similarity: {opp_sim:.4f}")
    
    # Geodesic step
    print(f"\nGeodesic step test:")
    stepped = pos1.geodesic_step(pos2, step_size=0.5)
    sim_after_step = stepped.phi_similarity(pos2)
    print(f"  Similarity to target after step: {sim_after_step:.4f}")
    print(f"  (Started at {sim:.4f})")
    print()


def test_geodesic_path():
    """Test GeodesicPath."""
    print("=" * 60)
    print("Testing GeodesicPath")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 64
    ranks = np.arange(1, dim + 1)
    
    # Create start and target
    start_signs = np.random.choice([-1, 1], size=dim).astype(np.int8)
    target_signs = np.random.choice([-1, 1], size=dim).astype(np.int8)
    
    start = PhiWeightedPosition.from_signs_with_phi_weights(start_signs, ranks)
    target = PhiWeightedPosition.from_signs_with_phi_weights(target_signs, ranks)
    
    initial_sim = start.phi_similarity(target)
    print(f"Initial similarity to target: {initial_sim:.4f}")
    
    # Walk toward target
    path = GeodesicPath(start)
    path.walk_toward(target, n_steps=20)
    
    final_sim = path.endpoint().phi_similarity(target)
    print(f"Final similarity to target: {final_sim:.4f}")
    print(f"Path length: {path.length()} steps")
    
    # Show similarity progression
    print(f"\nSimilarity progression:")
    for i in [0, 5, 10, 15, 20]:
        if i < len(path.positions):
            sim = path.positions[i].phi_similarity(target)
            print(f"  Step {i}: {sim:.4f}")
    print()


def test_phi_vs_hamming():
    """Compare φ-weighted similarity to Hamming distance."""
    print("=" * 60)
    print("Comparing φ-Weighted vs Hamming (Statistical)")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 64
    ranks = np.arange(1, dim + 1)
    
    # Create reference position
    ref_signs = np.random.choice([-1, 1], size=dim).astype(np.int8)
    ref = PhiWeightedPosition.from_signs_with_phi_weights(ref_signs, ranks)
    
    # Create positions with different numbers of flipped signs
    print(f"{'Flips':>6} | {'Hamming':>10} | {'φ-Weighted':>12} | {'Difference':>10}")
    print("-" * 50)
    
    for n_flips in [0, 1, 5, 10, 20, 32, 64]:
        # Flip n_flips signs (prioritize high-rank = low-weight dimensions)
        test_signs = ref_signs.copy()
        flip_indices = np.argsort(-ranks)[:n_flips]  # Flip lowest-weight first
        test_signs[flip_indices] *= -1
        
        test = PhiWeightedPosition(signs=test_signs, weights=ref.weights)
        
        # Hamming distance (statistical)
        hamming = np.sum(ref_signs != test_signs) / dim
        
        # φ-weighted similarity (geometric)
        phi_sim = ref.phi_similarity(test)
        
        print(f"{n_flips:>6} | {hamming:>10.4f} | {phi_sim:>12.4f} | {phi_sim - (1 - 2*hamming):>10.4f}")
    
    print(f"\nKey insight: φ-weighting penalizes flipping important dimensions more.")
    print(f"Hamming treats all dimensions equally (statistical).")
    print(f"φ-weighted respects the geometric structure.")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        navigator = GeodesicNavigator()
        navigator.convert_from_model(max_layers=2)
        
        # Test semantic neighbors
        print("\n" + "=" * 60)
        print("Testing Semantic Neighbors (φ-weighted)")
        print("=" * 60)
        
        test_words = ["king", "hello", "computer", "love"]
        for word in test_words:
            token_id = navigator.tokenizer.encode(word, add_special_tokens=False)[0]
            neighbors = navigator.find_semantic_neighbors(token_id, k=5)
            neighbor_words = [navigator.tokenizer.decode([tid]) for tid, _ in neighbors]
            scores = [f"{s:.4f}" for _, s in neighbors]
            print(f"{word}: {list(zip(neighbor_words, scores))}")
        
        # Test geodesic generation
        print("\n" + "=" * 60)
        print("Testing Geodesic Generation")
        print("=" * 60)
        
        prompt = "The king"
        print(f"Prompt: {prompt}")
        output = navigator.geodesic_generate(prompt, max_tokens=10)
        print(f"Output: {output}")
        
        # Also test layer-based navigation
        print("\n" + "=" * 60)
        print("Testing Layer Navigation")
        print("=" * 60)
        
        prompt = "Hello"
        print(f"Prompt: {prompt}")
        output = navigator.generate(prompt, max_tokens=10)
        print(f"Output: {output}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--neighbors":
        # Quick test of semantic neighbors without full model
        navigator = GeodesicNavigator()
        navigator.convert_from_model(max_layers=0)  # Just embeddings
        
        test_words = ["king", "queen", "man", "woman", "computer", "software"]
        for word in test_words:
            token_id = navigator.tokenizer.encode(word, add_special_tokens=False)[0]
            neighbors = navigator.find_semantic_neighbors(token_id, k=10)
            neighbor_words = [navigator.tokenizer.decode([tid]) for tid, _ in neighbors]
            print(f"\n{word}:")
            for i, (nw, (_, score)) in enumerate(zip(neighbor_words, neighbors)):
                print(f"  {i+1}. {nw!r}: {score:.4f}")
    
    else:
        test_phi_weighted_position()
        test_geodesic_path()
        test_phi_vs_hamming()
