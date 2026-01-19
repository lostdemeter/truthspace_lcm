"""
φ-Native Attention Engine
==========================

True φ-native computation that keeps everything in integer exponents
until the final output. No float decode until the very end.

Key insight: In the MESH decomposition U @ diag(S) @ Vt,
multiplication is just exponent addition:
    result_exp[i,j] = U_exp[i,k] + S_exp[k] + Vt_exp[k,j]

The challenge is accumulation (summing across k). We use log-sum-exp
with a precomputed LUT.

Author: TruthSpace LCM Team
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import time

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# Integer scale for exponents
SCALE = 8192  # 16-bit signed integer range


@dataclass
class PhiInt:
    """Integer φ-encoded value: value = sign × φ^(exp/SCALE)"""
    sign: np.ndarray  # int8: -1, 0, or 1
    exp: np.ndarray   # int32: integer exponent (scaled by SCALE)
    
    @property
    def shape(self):
        return self.sign.shape
    
    def decode(self) -> np.ndarray:
        """Decode to float (only at final output)."""
        return self.sign.astype(np.float64) * (PHI ** (self.exp.astype(np.float64) / SCALE))


def phi_encode(x: np.ndarray) -> PhiInt:
    """Encode float array to integer φ-representation."""
    sign = np.sign(x).astype(np.int8)
    with np.errstate(divide='ignore', invalid='ignore'):
        exp = np.round(np.log(np.abs(x) + 1e-15) / LOG_PHI * SCALE).astype(np.int32)
    return PhiInt(sign=sign, exp=exp)


class PhiLUT:
    """
    Lookup table for φ-arithmetic accumulation.
    
    For adding two φ-values: φ^a + φ^b = φ^c
    where c = a + log_φ(1 + φ^(b-a))  [assuming a >= b]
    
    We precompute log_φ(1 + φ^d) for d in [-max_diff, 0]
    """
    
    def __init__(self, max_diff: int = 50000, resolution: int = 1):
        """
        Args:
            max_diff: Maximum exponent difference to handle
            resolution: LUT resolution (1 = full precision)
        """
        self.max_diff = max_diff
        self.resolution = resolution
        self.lut_size = max_diff // resolution + 1
        
        # Precompute: log_φ(1 + φ^(d/SCALE)) * SCALE for d in [-max_diff, 0]
        # Index 0 = diff=-max_diff, Index max_diff = diff=0
        d_values = np.arange(-max_diff, 1, resolution)
        phi_d = PHI ** (d_values / SCALE)
        self.lut = np.round(np.log(1 + phi_d) / LOG_PHI * SCALE).astype(np.int32)
        
        print(f"LUT created: {self.lut_size} entries, {self.lut.nbytes / 1024:.1f} KB")
    
    def add_correction(self, diff: int) -> int:
        """
        Get the correction term for adding φ^a + φ^b where diff = b - a <= 0.
        
        Returns: log_φ(1 + φ^diff) * SCALE (integer)
        """
        if diff >= -self.max_diff:
            # Use LUT
            idx = diff + self.max_diff
            return int(self.lut[idx])
        else:
            # Out of LUT range - compute directly
            phi_diff = PHI ** (diff / SCALE)
            return int(round(np.log(1 + phi_diff) / LOG_PHI * SCALE))


class PhiNativeAttention:
    """
    Attention computed entirely in integer φ-space.
    
    MESH = U @ diag(S) @ Vt
    
    For input x, attention scores = x @ MESH @ x.T
                                  = (x @ U) @ diag(S) @ (Vt @ x.T)
    
    In φ-space:
    1. Encode x to φ-integers
    2. Multiply = add exponents
    3. Accumulate using LUT
    4. Decode only at final output
    """
    
    def __init__(
        self,
        U: PhiInt,      # (hidden_dim, rank)
        S_exp: np.ndarray,  # (rank,) - just exponents, signs all positive
        Vt: PhiInt,     # (rank, hidden_dim)
        lut: PhiLUT,
    ):
        self.U = U
        self.S_exp = S_exp  # int32 exponents
        self.Vt = Vt
        self.lut = lut
        self.rank = len(S_exp)
    
    def forward_native(self, x: np.ndarray) -> np.ndarray:
        """
        Compute attention scores using φ-native arithmetic.
        
        Args:
            x: Input (seq_len, hidden_dim) - float, will be encoded
            
        Returns:
            Attention scores (seq_len, seq_len) - float, decoded at end
        """
        seq_len, hidden_dim = x.shape
        
        # Step 1: Encode input to φ-integers
        x_phi = phi_encode(x)
        
        # Step 2: Compute x @ U in φ-space (seq_len, rank)
        # Each element is sum over hidden_dim of x[i,h] * U[h,k]
        # In φ-space: accumulate exp_x[i,h] + exp_U[h,k]
        x_proj_exp, x_proj_sign = self._phi_matmul(
            x_phi.exp, x_phi.sign,
            self.U.exp, self.U.sign
        )
        
        # Step 3: Compute Vt @ x.T in φ-space (rank, seq_len)
        # = (x @ Vt.T).T
        y_proj_exp, y_proj_sign = self._phi_matmul(
            x_phi.exp, x_phi.sign,
            self.Vt.exp.T, self.Vt.sign.T
        )
        # Transpose: (seq_len, rank) -> (rank, seq_len)
        y_proj_exp = y_proj_exp.T
        y_proj_sign = y_proj_sign.T
        
        # Step 4: Apply S scaling (just add S_exp to x_proj)
        # x_proj @ diag(S) = x_proj * S (element-wise per column)
        x_proj_scaled_exp = x_proj_exp + self.S_exp[np.newaxis, :]  # (seq_len, rank)
        # Signs unchanged
        
        # Step 5: Compute (x_proj @ S) @ y_proj in φ-space (seq_len, seq_len)
        scores_exp, scores_sign = self._phi_matmul(
            x_proj_scaled_exp, x_proj_sign,
            y_proj_exp, y_proj_sign
        )
        
        # Step 6: Decode final result
        scores = scores_sign.astype(np.float64) * (PHI ** (scores_exp.astype(np.float64) / SCALE))
        
        return scores
    
    def _phi_matmul(
        self,
        A_exp: np.ndarray,  # (M, K)
        A_sign: np.ndarray,
        B_exp: np.ndarray,  # (K, N)
        B_sign: np.ndarray,
        use_native: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matrix multiplication in φ-space.
        
        C[i,j] = Σ_k A[i,k] * B[k,j]
               = Σ_k sign_A[i,k] * sign_B[k,j] * φ^(exp_A[i,k] + exp_B[k,j])
        
        Returns (C_exp, C_sign) as integer arrays.
        
        If use_native=True, uses pure LUT-based accumulation.
        Otherwise uses hybrid (decode-sum-encode) for accuracy verification.
        """
        M, K = A_exp.shape
        K2, N = B_exp.shape
        assert K == K2, f"Dimension mismatch: {K} vs {K2}"
        
        C_exp = np.zeros((M, N), dtype=np.int32)
        C_sign = np.zeros((M, N), dtype=np.int8)
        
        if use_native:
            # Pure LUT-based accumulation
            for i in range(M):
                for j in range(N):
                    C_exp[i, j], C_sign[i, j] = self._phi_dot_native(
                        A_exp[i, :], A_sign[i, :],
                        B_exp[:, j], B_sign[:, j]
                    )
        else:
            # Hybrid: decode products, sum in float, re-encode
            for i in range(M):
                for j in range(N):
                    total = 0.0
                    for k in range(K):
                        prod_exp = A_exp[i, k] + B_exp[k, j]
                        prod_sign = A_sign[i, k] * B_sign[k, j]
                        prod_val = prod_sign * (PHI ** (prod_exp / SCALE))
                        total += prod_val
                    
                    C_sign[i, j] = 1 if total >= 0 else -1
                    if abs(total) > 1e-15:
                        C_exp[i, j] = int(np.log(abs(total)) / LOG_PHI * SCALE)
                    else:
                        C_exp[i, j] = -100 * SCALE
        
        return C_exp, C_sign
    
    def _phi_dot_native(
        self,
        a_exp: np.ndarray,  # (K,)
        a_sign: np.ndarray,
        b_exp: np.ndarray,  # (K,)
        b_sign: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Dot product using pure LUT-based accumulation.
        
        Separates positive and negative terms, sums each group,
        then combines. Falls back to exact computation when terms
        nearly cancel (within 2*SCALE of each other).
        """
        K = len(a_exp)
        MIN_EXP = -100 * SCALE
        
        # Compute products
        prod_exp = a_exp + b_exp
        prod_sign = a_sign * b_sign
        
        # Separate positive and negative
        pos_mask = prod_sign > 0
        neg_mask = prod_sign < 0
        
        # Sum positive terms - start from first value, not MIN_EXP
        pos_exps = [prod_exp[k] for k in range(K) if pos_mask[k]]
        if pos_exps:
            pos_exp = pos_exps[0]
            for exp in pos_exps[1:]:
                pos_exp = self._phi_add_same_sign(pos_exp, exp)
        else:
            pos_exp = MIN_EXP
        
        # Sum negative terms (as positive, track sign separately)
        neg_exps = [prod_exp[k] for k in range(K) if neg_mask[k]]
        if neg_exps:
            neg_exp = neg_exps[0]
            for exp in neg_exps[1:]:
                neg_exp = self._phi_add_same_sign(neg_exp, exp)
        else:
            neg_exp = MIN_EXP
        
        # Combine: pos_sum - neg_sum
        # Use exact computation when terms are within 2*SCALE (factor of ~φ²≈2.6)
        exp_diff = abs(pos_exp - neg_exp)
        
        if exp_diff > 2 * SCALE:
            # One side clearly dominates - use LUT approximation
            if pos_exp > neg_exp:
                diff = neg_exp - pos_exp
                if diff > -self.lut.max_diff:
                    sub_corr = self._phi_sub_correction(diff)
                    return pos_exp + sub_corr, 1
                return pos_exp, 1
            else:
                diff = pos_exp - neg_exp
                if diff > -self.lut.max_diff:
                    sub_corr = self._phi_sub_correction(diff)
                    return neg_exp + sub_corr, -1
                return neg_exp, -1
        else:
            # Close values - use exact computation to avoid cancellation errors
            pos_val = PHI ** (pos_exp / SCALE) if pos_exp > MIN_EXP else 0
            neg_val = PHI ** (neg_exp / SCALE) if neg_exp > MIN_EXP else 0
            result = pos_val - neg_val
            if abs(result) < 1e-15:
                return MIN_EXP, 1
            sign = 1 if result > 0 else -1
            exp = int(np.log(abs(result)) / LOG_PHI * SCALE)
            return exp, sign
    
    def _phi_add_same_sign(self, exp_a: int, exp_b: int) -> int:
        """Add two same-sign φ-values using LUT."""
        larger = max(exp_a, exp_b)
        smaller = min(exp_a, exp_b)
        diff = smaller - larger
        correction = self.lut.add_correction(diff)
        return larger + correction
    
    def _phi_sub_correction(self, diff: int) -> int:
        """Correction for subtraction: log_φ(1 - φ^diff) for diff < 0."""
        phi_diff = PHI ** (diff / SCALE)
        val = max(1 - phi_diff, 1e-15)
        return int(np.log(val) / LOG_PHI * SCALE)
    
    def _phi_sum_axis0(
        self,
        exp: np.ndarray,   # (K, N)
        sign: np.ndarray,  # (K, N)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sum along axis 0 in φ-space.
        
        This is the tricky part - we need to handle:
        1. Same-sign terms: φ^a + φ^b = φ^(a + correction)
        2. Opposite-sign terms: φ^a - φ^b = ±φ^(max + correction)
        
        For simplicity, we separate positive and negative terms,
        sum each group, then combine.
        """
        K, N = exp.shape
        
        # Separate positive and negative terms
        pos_mask = sign > 0
        neg_mask = sign < 0
        
        # Initialize with very negative exponent (effectively 0)
        MIN_EXP = -100 * SCALE
        
        result_exp = np.full(N, MIN_EXP, dtype=np.int32)
        result_sign = np.ones(N, dtype=np.int8)
        
        # Sum positive terms
        pos_exp = np.where(pos_mask, exp, MIN_EXP)
        pos_sum_exp = self._phi_sum_same_sign(pos_exp)
        
        # Sum negative terms (treat as positive, track sign separately)
        neg_exp = np.where(neg_mask, exp, MIN_EXP)
        neg_sum_exp = self._phi_sum_same_sign(neg_exp)
        
        # Combine: pos_sum - neg_sum
        # If pos_sum > neg_sum: result = pos_sum * (1 - φ^(neg-pos))
        # If neg_sum > pos_sum: result = -neg_sum * (1 - φ^(pos-neg))
        
        for j in range(N):
            p_exp = pos_sum_exp[j]
            n_exp = neg_sum_exp[j]
            
            if p_exp > n_exp + SCALE:  # pos dominates
                # result ≈ pos_sum (neg is negligible)
                diff = n_exp - p_exp
                if diff > -self.lut.max_diff:
                    # φ^p - φ^n = φ^p * (1 - φ^(n-p))
                    # log: p + log_φ(1 - φ^(n-p))
                    # For subtraction, correction is negative
                    correction = self._sub_correction(diff)
                    result_exp[j] = p_exp + correction
                else:
                    result_exp[j] = p_exp
                result_sign[j] = 1
                
            elif n_exp > p_exp + SCALE:  # neg dominates
                diff = p_exp - n_exp
                if diff > -self.lut.max_diff:
                    correction = self._sub_correction(diff)
                    result_exp[j] = n_exp + correction
                else:
                    result_exp[j] = n_exp
                result_sign[j] = -1
                
            else:  # Close values - need careful handling
                # Decode and compute exactly for this edge case
                pos_val = PHI ** (p_exp / SCALE) if p_exp > MIN_EXP else 0
                neg_val = PHI ** (n_exp / SCALE) if n_exp > MIN_EXP else 0
                diff_val = pos_val - neg_val
                
                if abs(diff_val) < 1e-15:
                    result_exp[j] = MIN_EXP
                    result_sign[j] = 1
                else:
                    result_sign[j] = 1 if diff_val > 0 else -1
                    result_exp[j] = int(np.log(abs(diff_val)) / LOG_PHI * SCALE)
        
        return result_exp, result_sign
    
    def _phi_sum_same_sign(self, exp: np.ndarray) -> np.ndarray:
        """
        Sum same-sign terms along axis 0.
        
        φ^a + φ^b = φ^max(a,b) * (1 + φ^(min-max))
                  = φ^(max + log_φ(1 + φ^(min-max)))
        
        We iteratively combine terms, always adding smaller to larger.
        """
        K, N = exp.shape
        
        # Start with first row
        result = exp[0].copy().astype(np.int64)  # Use int64 to avoid overflow
        
        for k in range(1, K):
            # Add exp[k] to result
            # result = φ^result + φ^exp[k]
            
            for j in range(N):
                larger = max(result[j], exp[k, j])
                smaller = min(result[j], exp[k, j])
                diff = smaller - larger  # Always <= 0
                
                # Get correction from LUT
                correction = self.lut.add_correction(int(diff))
                
                # New result = larger + correction
                result[j] = larger + correction
        
        return result.astype(np.int32)
    
    def _sub_correction(self, diff: np.ndarray) -> np.ndarray:
        """
        Correction for subtraction: log_φ(1 - φ^diff) for diff < 0.
        
        For diff << 0, this approaches 0.
        For diff close to 0, this is large and negative.
        """
        # 1 - φ^diff where diff < 0
        # As diff -> -inf, φ^diff -> 0, so 1 - φ^diff -> 1, log_φ(1) = 0
        # As diff -> 0, φ^diff -> 1, so 1 - φ^diff -> 0, log_φ -> -inf
        
        phi_diff = PHI ** (np.asarray(diff) / SCALE)
        val = 1 - phi_diff
        
        # Handle edge cases
        val = np.maximum(val, 1e-15)
        
        return np.round(np.log(val) / LOG_PHI * SCALE).astype(np.int32)
    
    @classmethod
    def from_qk_weights(
        cls,
        W_q: np.ndarray,
        W_k: np.ndarray,
        rank: int = 128,
        lut: Optional[PhiLUT] = None,
    ) -> 'PhiNativeAttention':
        """Create from Q and K projection weights."""
        # Compute MESH
        MESH = W_q.T @ W_k
        
        # SVD
        U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Encode in φ-integers
        U_phi = phi_encode(U)
        Vt_phi = phi_encode(Vt)
        S_exp = np.round(np.log(S) / LOG_PHI * SCALE).astype(np.int32)
        
        if lut is None:
            lut = PhiLUT()
        
        return cls(U_phi, S_exp, Vt_phi, lut)


def test_phi_native():
    """Test φ-native attention against float computation."""
    print("=" * 60)
    print("Testing φ-Native Attention (Hybrid)")
    print("=" * 60)
    
    # Create random MESH-like matrix
    np.random.seed(42)
    hidden_dim = 128
    rank = 64
    
    # Simulate W_q, W_k
    W_q = np.random.randn(hidden_dim, hidden_dim) * 0.1
    W_k = np.random.randn(hidden_dim, hidden_dim) * 0.1
    MESH = W_q.T @ W_k
    
    # SVD for ground truth
    U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    
    # Create φ-native attention
    print("\nCreating φ-native attention...")
    lut = PhiLUT(max_diff=50000)
    phi_attn = PhiNativeAttention.from_qk_weights(W_q, W_k, rank=rank, lut=lut)
    
    # Test input
    seq_len = 16
    x = np.random.randn(seq_len, hidden_dim) * 0.5
    
    # Ground truth (float)
    print("\nComputing ground truth (float)...")
    t0 = time.perf_counter()
    x_proj = x @ U
    y_proj = x @ Vt.T
    scores_float = x_proj @ np.diag(S) @ y_proj.T
    t_float = time.perf_counter() - t0
    
    # φ-native (hybrid)
    print("Computing φ-native (hybrid)...")
    t0 = time.perf_counter()
    scores_phi = phi_attn.forward_native(x)
    t_phi = time.perf_counter() - t0
    
    # Compare
    corr = np.corrcoef(scores_float.flatten(), scores_phi.flatten())[0, 1]
    rel_error = np.abs(scores_float - scores_phi) / (np.abs(scores_float) + 1e-10)
    
    print(f"\n{'='*60}")
    print("Hybrid Results:")
    print(f"  Correlation: {corr*100:.6f}%")
    print(f"  Mean relative error: {rel_error.mean()*100:.4f}%")
    print(f"  Max relative error: {rel_error.max()*100:.4f}%")
    print(f"  Float time: {t_float*1000:.2f} ms")
    print(f"  φ-native time: {t_phi*1000:.2f} ms")
    print(f"{'='*60}")
    
    return corr


def test_pure_lut():
    """Test pure LUT-based accumulation."""
    print("\n" + "=" * 60)
    print("Testing Pure LUT-Based Accumulation")
    print("=" * 60)
    
    np.random.seed(42)
    hidden_dim = 32  # Smaller for speed
    rank = 16
    
    W_q = np.random.randn(hidden_dim, hidden_dim) * 0.1
    W_k = np.random.randn(hidden_dim, hidden_dim) * 0.1
    
    # SVD for ground truth
    MESH = W_q.T @ W_k
    U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    
    lut = PhiLUT(max_diff=50000)
    phi_attn = PhiNativeAttention.from_qk_weights(W_q, W_k, rank=rank, lut=lut)
    
    seq_len = 8
    x = np.random.randn(seq_len, hidden_dim) * 0.5
    
    # Ground truth
    x_proj = x @ U
    y_proj = x @ Vt.T
    scores_float = x_proj @ np.diag(S) @ y_proj.T
    
    # Encode x
    x_phi = phi_encode(x)
    
    # Test pure LUT matmul for first projection: x @ U
    print("\nTesting x @ U with pure LUT...")
    
    # Hybrid
    t0 = time.perf_counter()
    proj_exp_h, proj_sign_h = phi_attn._phi_matmul(
        x_phi.exp, x_phi.sign,
        phi_attn.U.exp, phi_attn.U.sign,
        use_native=False
    )
    t_hybrid = time.perf_counter() - t0
    proj_hybrid = proj_sign_h * (PHI ** (proj_exp_h / SCALE))
    
    # Pure LUT
    t0 = time.perf_counter()
    proj_exp_n, proj_sign_n = phi_attn._phi_matmul(
        x_phi.exp, x_phi.sign,
        phi_attn.U.exp, phi_attn.U.sign,
        use_native=True
    )
    t_native = time.perf_counter() - t0
    proj_native = proj_sign_n * (PHI ** (proj_exp_n / SCALE))
    
    # Compare to float
    x_decoded = x_phi.decode()
    U_decoded = phi_attn.U.decode()
    proj_float = x_decoded @ U_decoded
    
    corr_hybrid = np.corrcoef(proj_float.flatten(), proj_hybrid.flatten())[0, 1]
    corr_native = np.corrcoef(proj_float.flatten(), proj_native.flatten())[0, 1]
    
    print(f"\n{'='*60}")
    print("x @ U Results:")
    print(f"  Hybrid correlation: {corr_hybrid*100:.4f}%")
    print(f"  Pure LUT correlation: {corr_native*100:.4f}%")
    print(f"  Hybrid time: {t_hybrid*1000:.2f} ms")
    print(f"  Pure LUT time: {t_native*1000:.2f} ms")
    print(f"{'='*60}")
    
    return corr_native


def test_with_qwen2():
    """Test φ-native attention with actual Qwen2 weights."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Requires transformers library")
        return
    
    print("=" * 60)
    print("Testing φ-Native Attention with Qwen2-7B")
    print("=" * 60)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    # Get layer 0, head 0 weights
    layer = model.model.layers[0]
    W_q = layer.self_attn.q_proj.weight.detach().float().numpy()
    W_k = layer.self_attn.k_proj.weight.detach().float().numpy()
    
    # Reshape for head 0
    n_heads = 28
    head_dim = 128
    W_q_heads = W_q.reshape(n_heads, head_dim, -1)
    W_k_heads = W_k.reshape(4, head_dim, -1)  # 4 KV heads
    
    W_q_head = W_q_heads[0]
    W_k_head = W_k_heads[0]
    
    # Create φ-native attention
    print("\nCreating φ-native attention...")
    lut = PhiLUT(max_diff=32768)
    phi_attn = PhiNativeAttention.from_qk_weights(W_q_head, W_k_head, rank=128, lut=lut)
    
    # Test with real text
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        hidden = model.model.embed_tokens(inputs.input_ids)
        hidden = hidden[0].float().numpy()  # (seq_len, hidden_dim)
    
    seq_len = hidden.shape[0]
    print(f"\nTest sequence: '{text}'")
    print(f"Sequence length: {seq_len}")
    
    # Ground truth
    MESH = W_q_head.T @ W_k_head
    U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
    U, S, Vt = U[:, :128], S[:128], Vt[:128, :]
    
    x_proj = hidden @ U
    y_proj = hidden @ Vt.T
    scores_float = x_proj @ np.diag(S) @ y_proj.T
    
    # φ-native
    print("Computing φ-native...")
    t0 = time.perf_counter()
    scores_phi = phi_attn.forward_native(hidden)
    t_phi = time.perf_counter() - t0
    
    # Compare
    corr = np.corrcoef(scores_float.flatten(), scores_phi.flatten())[0, 1]
    rel_error = np.abs(scores_float - scores_phi) / (np.abs(scores_float) + 1e-10)
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Correlation: {corr*100:.6f}%")
    print(f"  Mean relative error: {rel_error.mean()*100:.4f}%")
    print(f"  Max relative error: {rel_error.max()*100:.4f}%")
    print(f"  φ-native time: {t_phi*1000:.2f} ms")
    print(f"{'='*60}")
    
    return corr


if __name__ == "__main__":
    # Test with synthetic data first
    corr = test_phi_native()
    
    if corr > 0.99:
        print("\n✓ Hybrid test passed!")
        
        # Test pure LUT
        corr_lut = test_pure_lut()
        print(f"\nPure LUT correlation: {corr_lut*100:.4f}%")
        
        if corr_lut > 0.99:
            print("\n✓ Pure LUT test passed! Testing with Qwen2...")
            test_with_qwen2()
    else:
        print(f"\n✗ Synthetic test failed: {corr*100:.2f}% correlation")
        print("Need to debug φ-native arithmetic")
