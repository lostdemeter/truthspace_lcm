"""
Discriminant Space Navigator - Navigation in the true semantic space.

Key insight from prior work (Docs 133, 134, 135):
- MESH = W_q.T @ W_k has effective rank 106, not 3584
- Singular values follow φ-Zipf: S[i] ∝ 1/i^(1/φ)
- The U, V bases encode semantic directions
- S provides the "W-axis" - universal constant that anchors computations

This is the correct space for navigation:
- Raw embedding signs don't encode semantics directly
- Discriminant space IS the semantic structure
- Navigate in 106-dim discriminant space, not 3584-dim embedding space

From Doc 135:
- Top dims (few, important): Platonic Ideals, specific relationships
- Bottom dims (many, common): Structural patterns
- The φ-Zipf hierarchy tells us which dims matter
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch

PHI = 1.6180339887498949
INV_PHI = 1.0 / PHI


@dataclass
class DiscriminantBasis:
    """
    The discriminant basis for one attention head.
    
    MESH = U @ diag(S) @ Vt
    
    Where:
    - U: (hidden_dim, k) - left singular vectors (query directions)
    - S: (k,) - singular values (the W-axis, follows φ-Zipf)
    - Vt: (k, hidden_dim) - right singular vectors (key directions)
    """
    U: np.ndarray      # (hidden_dim, k)
    S: np.ndarray      # (k,)
    Vt: np.ndarray     # (k, hidden_dim)
    layer: int
    head: int
    
    @property
    def k(self) -> int:
        return len(self.S)
    
    @property
    def hidden_dim(self) -> int:
        return self.U.shape[0]
    
    def project_to_discriminant(self, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project hidden states to discriminant space.
        
        Returns (U_proj, V_proj) where:
        - U_proj = hidden @ U  (query-like projection)
        - V_proj = hidden @ Vt.T  (key-like projection)
        """
        U_proj = hidden @ self.U      # (..., k)
        V_proj = hidden @ self.Vt.T   # (..., k)
        return U_proj, V_proj
    
    def discriminant_position(self, hidden: np.ndarray) -> np.ndarray:
        """
        Get position in discriminant space (scaled by S).
        
        This is the "true" semantic position.
        """
        U_proj, V_proj = self.project_to_discriminant(hidden)
        # Combine U and V projections, scaled by S
        return (U_proj * self.S + V_proj * self.S) / 2
    
    def phi_zipf_weights(self) -> np.ndarray:
        """
        Compute φ-Zipf weights for each dimension.
        
        From Doc 135: S[i] ∝ 1/i^(1/φ)
        """
        ranks = np.arange(1, self.k + 1)
        weights = 1.0 / (ranks ** INV_PHI)
        return weights / weights.sum()


class DiscriminantNavigator:
    """
    Navigator that operates in discriminant space.
    
    Instead of navigating in raw embedding space (3584 dims),
    we navigate in discriminant space (106 dims) where the
    semantic structure actually lives.
    """
    
    def __init__(self, k: int = 106):
        """
        Args:
            k: Number of discriminant dimensions (default 106 for 99.5% accuracy)
        """
        self.k = k
        self.bases: Dict[Tuple[int, int], DiscriminantBasis] = {}
        self.tokenizer = None
        self.embeddings: Optional[np.ndarray] = None
        
        # Token positions in discriminant space
        self.token_discriminant_positions: Optional[np.ndarray] = None
    
    def load_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                        layers: List[int] = None, heads: List[int] = None):
        """
        Load model and compute discriminant bases.
        
        Args:
            model_name: HuggingFace model name
            layers: Which layers to analyze (default: [0])
            heads: Which heads to analyze (default: [0])
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        layers = layers or [0]
        heads = heads or [0]
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        # Get embeddings
        self.embeddings = model.model.embed_tokens.weight.data.numpy()
        print(f"Loaded {len(self.embeddings)} token embeddings")
        
        # Compute MESH and SVD for each layer/head
        hidden_dim = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        head_dim = hidden_dim // num_heads
        
        print(f"Computing discriminant bases (k={self.k})...")
        
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            
            # Get Q and K weight matrices
            W_q = layer.self_attn.q_proj.weight.data.numpy()  # (hidden_dim, hidden_dim)
            W_k = layer.self_attn.k_proj.weight.data.numpy()  # (kv_dim, hidden_dim)
            
            for head_idx in heads:
                # Extract head-specific weights
                q_start = head_idx * head_dim
                q_end = (head_idx + 1) * head_dim
                
                # For GQA, K heads may be fewer
                num_kv_heads = model.config.num_key_value_heads
                kv_head_idx = head_idx * num_kv_heads // num_heads
                k_start = kv_head_idx * head_dim
                k_end = (kv_head_idx + 1) * head_dim
                
                W_q_head = W_q[q_start:q_end, :]  # (head_dim, hidden_dim)
                W_k_head = W_k[k_start:k_end, :]  # (head_dim, hidden_dim)
                
                # Compute MESH = W_q.T @ W_k
                MESH = W_q_head.T @ W_k_head  # (hidden_dim, hidden_dim)
                
                # SVD
                U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
                
                # Keep top-k
                basis = DiscriminantBasis(
                    U=U[:, :self.k],
                    S=S[:self.k],
                    Vt=Vt[:self.k, :],
                    layer=layer_idx,
                    head=head_idx,
                )
                
                self.bases[(layer_idx, head_idx)] = basis
                
                # Verify φ-Zipf
                ranks = np.arange(1, min(50, self.k) + 1)
                log_s = np.log(S[:50])
                log_ranks = np.log(ranks)
                slope, _ = np.polyfit(log_ranks, log_s, 1)
                
                print(f"  Layer {layer_idx}, Head {head_idx}: "
                      f"S[0]={S[0]:.3f}, φ-Zipf slope={-slope:.3f} (target: {INV_PHI:.3f})")
        
        # Compute token positions in discriminant space (using first basis)
        first_basis = list(self.bases.values())[0]
        self._compute_token_positions(first_basis)
        
        del model
    
    def _compute_token_positions(self, basis: DiscriminantBasis):
        """Compute all token positions in discriminant space."""
        print("Computing token positions in discriminant space...")
        
        self.token_discriminant_positions = np.zeros(
            (len(self.embeddings), basis.k), dtype=np.float32
        )
        
        for i, emb in enumerate(self.embeddings):
            self.token_discriminant_positions[i] = basis.discriminant_position(emb)
        
        print(f"Computed {len(self.token_discriminant_positions)} positions")
    
    def discriminant_distance(self, pos1: np.ndarray, pos2: np.ndarray,
                              basis: DiscriminantBasis = None) -> float:
        """
        Compute distance in discriminant space.
        
        Uses φ-Zipf weighting: top dims matter more.
        """
        basis = basis or list(self.bases.values())[0]
        weights = basis.phi_zipf_weights()
        
        diff = pos1 - pos2
        return np.sqrt(np.sum(weights * diff**2))
    
    def discriminant_similarity(self, pos1: np.ndarray, pos2: np.ndarray,
                                basis: DiscriminantBasis = None) -> float:
        """
        Compute cosine similarity in discriminant space.
        
        Uses φ-Zipf weighting.
        """
        basis = basis or list(self.bases.values())[0]
        weights = np.sqrt(basis.phi_zipf_weights())
        
        w1 = weights * pos1
        w2 = weights * pos2
        
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return np.dot(w1, w2) / (norm1 * norm2)
    
    def find_nearest_tokens(self, position: np.ndarray, k: int = 5,
                            exclude: set = None) -> List[Tuple[int, float]]:
        """
        Find k nearest tokens to a position in discriminant space.
        """
        exclude = exclude or set()
        
        scores = []
        for i, pos in enumerate(self.token_discriminant_positions):
            if i in exclude:
                continue
            sim = self.discriminant_similarity(position, pos)
            scores.append((i, sim))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
    
    def get_token_position(self, token_id: int) -> np.ndarray:
        """Get token's position in discriminant space."""
        return self.token_discriminant_positions[token_id].copy()
    
    def analyze_semantic_pairs(self, pairs: List[Tuple[str, str]]):
        """
        Analyze semantic pairs in discriminant space.
        
        This should show that related pairs are close in discriminant space.
        """
        print("\nSemantic pair analysis in discriminant space:")
        print("=" * 60)
        
        for w1, w2 in pairs:
            try:
                id1 = self.tokenizer.encode(w1, add_special_tokens=False)[0]
                id2 = self.tokenizer.encode(w2, add_special_tokens=False)[0]
                
                pos1 = self.get_token_position(id1)
                pos2 = self.get_token_position(id2)
                
                dist = self.discriminant_distance(pos1, pos2)
                sim = self.discriminant_similarity(pos1, pos2)
                
                print(f"{w1:12} ↔ {w2:12}: dist={dist:.4f}, sim={sim:.4f}")
            except Exception as e:
                print(f"{w1:12} ↔ {w2:12}: error - {e}")
    
    def find_semantic_neighbors(self, word: str, k: int = 10):
        """Find nearest neighbors in discriminant space."""
        token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
        position = self.get_token_position(token_id)
        
        neighbors = self.find_nearest_tokens(position, k=k+1, exclude={token_id})
        
        print(f"\nNearest neighbors to '{word}' in discriminant space:")
        for tid, sim in neighbors[:k]:
            neighbor = self.tokenizer.decode([tid])
            print(f"  {neighbor!r}: {sim:.4f}")
    
    def navigate(self, start_word: str, direction_word: str, 
                 steps: int = 5, step_size: float = 0.1) -> List[str]:
        """
        Navigate from start_word in the direction of direction_word.
        
        This is geodesic-like navigation in discriminant space.
        """
        start_id = self.tokenizer.encode(start_word, add_special_tokens=False)[0]
        dir_id = self.tokenizer.encode(direction_word, add_special_tokens=False)[0]
        
        start_pos = self.get_token_position(start_id)
        dir_pos = self.get_token_position(dir_id)
        
        # Direction vector
        direction = dir_pos - start_pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        path = [start_word]
        current_pos = start_pos.copy()
        visited = {start_id}
        
        for _ in range(steps):
            # Move in direction
            current_pos = current_pos + step_size * direction
            
            # Find nearest unvisited token
            neighbors = self.find_nearest_tokens(current_pos, k=10, exclude=visited)
            
            if not neighbors:
                break
            
            best_id, best_sim = neighbors[0]
            visited.add(best_id)
            
            word = self.tokenizer.decode([best_id])
            path.append(word)
            
            # Update position to actual token position
            current_pos = self.get_token_position(best_id)
        
        return path


def test_discriminant_navigator():
    """Test the discriminant navigator."""
    print("=" * 60)
    print("Testing Discriminant Navigator")
    print("=" * 60)
    
    # Use gender-specialized heads (14-20) based on Doc 135
    navigator = DiscriminantNavigator(k=106)
    navigator.load_from_model(layers=[0], heads=[15])
    
    # Test semantic pairs
    pairs = [
        ("king", "queen"),
        ("king", "kingdom"),
        ("man", "woman"),
        ("happy", "sad"),
        ("happy", "joy"),
        ("computer", "software"),
        ("the", "a"),
    ]
    navigator.analyze_semantic_pairs(pairs)
    
    # Test semantic neighbors
    test_words = ["king", "computer", "happy"]
    for word in test_words:
        navigator.find_semantic_neighbors(word, k=5)
    
    # Test navigation
    print("\n" + "=" * 60)
    print("Testing Navigation")
    print("=" * 60)
    
    print("\nNavigating from 'king' toward 'queen':")
    path = navigator.navigate("king", "queen", steps=5, step_size=0.2)
    print(f"  Path: {path}")
    
    print("\nNavigating from 'happy' toward 'sad':")
    path = navigator.navigate("happy", "sad", steps=5, step_size=0.2)
    print(f"  Path: {path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_discriminant_navigator()
    else:
        print("Usage: python discriminant_navigator.py --test")
        print("\nThis module implements navigation in discriminant space,")
        print("where the true semantic structure lives (106 dims, not 3584).")
