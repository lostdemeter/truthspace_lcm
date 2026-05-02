"""
MESH Computer: Pre-compute combined matrices to eliminate error compounding.

The MESH principle: Self-referential operations (A @ B where A and B share a source)
compound errors. Pre-computing the combined form eliminates this.

Examples:
- Attention: MESH = W_q.T @ W_k
- Bilinear: MESH = W_a.T @ W_b
"""

import torch
from typing import Tuple, Dict, Optional
from .encoder import PhiEncoder


class MESHComputer:
    """
    Pre-compute combined matrices for error-free φ-arithmetic.
    
    The key insight: Instead of encoding W_q and W_k separately and then
    computing Q @ K.T, we pre-compute MESH = W_q.T @ W_k and encode that.
    
    This eliminates the multiplicative error compounding that occurs when
    two separately-encoded matrices are multiplied.
    """
    
    def __init__(self, encoder: Optional[PhiEncoder] = None):
        self.encoder = encoder or PhiEncoder()
    
    def compute_attention_mesh(
        self, 
        W_q: torch.Tensor, 
        W_k: torch.Tensor,
        b_q: Optional[torch.Tensor] = None,
        b_k: Optional[torch.Tensor] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute MESH for attention: score = input @ MESH @ input.T
        
        With biases:
            score = input @ MESH @ input.T
                  + input @ cross_qk
                  + cross_kq @ input.T
                  + bias_term
        
        Args:
            W_q: Query weight matrix [head_dim, hidden_dim]
            W_k: Key weight matrix [head_dim, hidden_dim]
            b_q: Optional query bias [head_dim]
            b_k: Optional key bias [head_dim]
            
        Returns:
            Dict with φ-encoded matrices:
                - mesh: (signs, exps) for W_q.T @ W_k
                - cross_qk: (signs, exps) for W_q.T @ b_k (if biases)
                - cross_kq: (signs, exps) for b_q @ W_k (if biases)
                - bias_term: scalar b_q @ b_k (if biases)
        """
        result = {}
        
        # Core MESH: W_q.T @ W_k
        mesh = W_q.T @ W_k
        result['mesh'] = self.encoder.encode(mesh)
        
        # Handle biases
        if b_q is not None and b_k is not None:
            cross_qk = W_q.T @ b_k
            cross_kq = b_q @ W_k
            bias_term = b_q @ b_k
            
            result['cross_qk'] = self.encoder.encode(cross_qk)
            result['cross_kq'] = self.encoder.encode(cross_kq)
            result['bias_term'] = bias_term.item()
        
        return result
    
    def compute_bilinear_mesh(
        self,
        W_a: torch.Tensor,
        W_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MESH for generic bilinear operation: x @ MESH @ y.T
        
        Args:
            W_a: First weight matrix
            W_b: Second weight matrix
            
        Returns:
            (signs, exps) for W_a.T @ W_b
        """
        mesh = W_a.T @ W_b
        return self.encoder.encode(mesh)
    
    def compute_projection_mesh(
        self,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute all MESH matrices for a full attention layer.
        
        For attention: softmax(Q @ K.T / sqrt(d)) @ V
        
        We pre-compute:
            - MESH_qk = W_q.T @ W_k (for scores)
            - W_v encoded (for value projection)
        
        Args:
            W_q, W_k, W_v: Query, Key, Value weight matrices
            
        Returns:
            Dict with φ-encoded matrices
        """
        return {
            'mesh_qk': self.encoder.encode(W_q.T @ W_k),
            'w_v': self.encoder.encode(W_v)
        }
    
    def verify_mesh_accuracy(
        self,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        test_input: torch.Tensor
    ) -> Dict[str, float]:
        """
        Verify MESH accuracy by comparing direct vs MESH computation.
        
        Direct: (input @ W_q) @ (input @ W_k).T
        MESH:   input @ (W_q.T @ W_k) @ input.T
        
        These should be mathematically identical, but φ-encoding
        of MESH has less error than separate encoding of W_q and W_k.
        """
        # Direct computation (ground truth)
        Q = test_input @ W_q.T
        K = test_input @ W_k.T
        scores_direct = Q @ K.T
        
        # MESH computation
        mesh = W_q.T @ W_k
        scores_mesh = test_input @ mesh @ test_input.T
        
        # φ-encoded MESH computation
        mesh_signs, mesh_exps = self.encoder.encode(mesh)
        mesh_decoded = self.encoder.decode(mesh_signs, mesh_exps)
        scores_phi = test_input @ mesh_decoded @ test_input.T
        
        # Separate φ-encoding (for comparison)
        wq_signs, wq_exps = self.encoder.encode(W_q)
        wk_signs, wk_exps = self.encoder.encode(W_k)
        W_q_decoded = self.encoder.decode(wq_signs, wq_exps)
        W_k_decoded = self.encoder.decode(wk_signs, wk_exps)
        Q_phi = test_input @ W_q_decoded.T
        K_phi = test_input @ W_k_decoded.T
        scores_separate = Q_phi @ K_phi.T
        
        # Compute errors
        def correlation(a, b):
            return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        
        return {
            'mesh_vs_direct': correlation(scores_mesh, scores_direct),
            'phi_mesh_vs_direct': correlation(scores_phi, scores_direct),
            'phi_separate_vs_direct': correlation(scores_separate, scores_direct),
            'mesh_improvement': (
                (1 - abs(1 - correlation(scores_phi, scores_direct))) /
                (1 - abs(1 - correlation(scores_separate, scores_direct)) + 1e-10)
            )
        }


def test_mesh():
    """Test MESH computation."""
    print("Testing MESHComputer...")
    
    computer = MESHComputer()
    
    # Create test matrices (typical attention dimensions)
    hidden_dim = 256
    head_dim = 64
    seq_len = 32
    
    W_q = torch.randn(head_dim, hidden_dim) * 0.1
    W_k = torch.randn(head_dim, hidden_dim) * 0.1
    test_input = torch.randn(seq_len, hidden_dim)
    
    # Verify accuracy
    stats = computer.verify_mesh_accuracy(W_q, W_k, test_input)
    
    print(f"  MESH vs Direct: {stats['mesh_vs_direct']:.6f}")
    print(f"  φ-MESH vs Direct: {stats['phi_mesh_vs_direct']:.6f}")
    print(f"  φ-Separate vs Direct: {stats['phi_separate_vs_direct']:.6f}")
    print(f"  MESH improvement: {stats['mesh_improvement']:.2f}x")
    
    return computer


if __name__ == "__main__":
    test_mesh()
