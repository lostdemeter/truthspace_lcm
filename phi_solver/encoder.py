"""
φ-Encoder: Encode/decode values in φ-basis.

The fundamental representation: value = sign × φ^(exponent/K)

This is the geometric representation of neural network weights.
"""

import torch
import numpy as np
from typing import Tuple

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


class PhiEncoder:
    """
    Encode and decode values in φ-basis.
    
    Every value is represented as: value = sign × φ^(exponent/K)
    
    Args:
        K: Resolution parameter (higher = more precision)
           K=32 gives ~3% precision per step, 99.92% accuracy
           K=128 gives ~0.8% precision per step, 99.99% accuracy
        bias: Exponent bias (centers the range around typical weight magnitudes)
    """
    
    def __init__(self, K: int = 32, bias: int = 4096):
        self.K = K
        self.bias = bias
        
        # Pre-compute LUT for φ^(e/K) values
        self.max_exp = 8192
        phi_powers_np = np.zeros(self.max_exp, dtype=np.float64)
        for e in range(self.max_exp):
            exp_val = (e - bias) / K
            exp_val = max(-300, min(300, exp_val))
            phi_powers_np[e] = PHI ** exp_val
        self.phi_powers = torch.from_numpy(phi_powers_np).float()
        
        # Pre-compute addition LUT: φ^a + φ^b = φ^(b + LUT[a-b])
        # This enables φ-addition via integer operations
        self._build_addition_lut()
    
    def _build_addition_lut(self):
        """Build LUT for φ-addition."""
        max_diff = 1024
        self.add_lut = torch.zeros(2 * max_diff + 1, dtype=torch.long)
        
        for d in range(-max_diff, max_diff + 1):
            # φ^d + 1 = φ^(LUT[d])
            # LUT[d] = K * log_φ(φ^(d/K) + 1)
            val = PHI ** (d / self.K) + 1
            self.add_lut[d + max_diff] = int(round(self.K * np.log(val) / LN_PHI))
    
    def encode(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode float values to φ-basis (sign, exponent).
        
        Args:
            values: Float tensor of any shape
            
        Returns:
            signs: Tensor of {-1, 0, +1}
            exponents: Tensor of integers
        """
        signs = torch.sign(values)
        
        # Handle zeros
        magnitudes = torch.abs(values)
        magnitudes = torch.clamp(magnitudes, min=1e-30)
        
        # Compute exponents: e = K * log_φ(|v|) + bias
        exponents = self.K * torch.log(magnitudes) / LN_PHI + self.bias
        exponents = torch.round(exponents).long()
        exponents = torch.clamp(exponents, 0, self.max_exp - 1)
        
        # Zero values get exponent 0
        exponents = torch.where(values == 0, torch.zeros_like(exponents), exponents)
        
        return signs, exponents
    
    def decode(self, signs: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
        """
        Decode φ-basis (sign, exponent) back to float values.
        
        Args:
            signs: Tensor of {-1, 0, +1}
            exponents: Tensor of integers
            
        Returns:
            Float tensor
        """
        phi_powers = self.phi_powers.to(exponents.device)
        exponents = torch.clamp(exponents, 0, self.max_exp - 1)
        values = signs.float() * phi_powers[exponents]
        return values
    
    def multiply_exponents(self, a_exp: torch.Tensor, b_exp: torch.Tensor) -> torch.Tensor:
        """
        φ-multiplication via exponent addition.
        
        φ^a × φ^b = φ^(a+b)
        
        In our representation with bias:
        result_exp = a_exp + b_exp - bias
        """
        return a_exp + b_exp - self.bias
    
    def verify_accuracy(self, original: torch.Tensor) -> dict:
        """
        Verify encoding accuracy for a tensor.
        
        Returns:
            dict with correlation, MAE, max_error
        """
        signs, exps = self.encode(original)
        reconstructed = self.decode(signs, exps)
        
        # Flatten for correlation
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        
        # Correlation
        corr = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0, 1].item()
        
        # MAE
        mae = torch.abs(original - reconstructed).mean().item()
        
        # Max error
        max_err = torch.abs(original - reconstructed).max().item()
        
        # Relative error
        rel_err = (torch.abs(original - reconstructed) / (torch.abs(original) + 1e-10)).mean().item()
        
        return {
            "correlation": corr,
            "mae": mae,
            "max_error": max_err,
            "relative_error": rel_err
        }


def test_encoder():
    """Test the φ-encoder."""
    print("Testing PhiEncoder...")
    
    encoder = PhiEncoder(K=32)
    
    # Test with random weights (typical neural network range)
    weights = torch.randn(1000, 1000) * 0.1
    
    # Encode and decode
    signs, exps = encoder.encode(weights)
    reconstructed = encoder.decode(signs, exps)
    
    # Verify
    stats = encoder.verify_accuracy(weights)
    print(f"  Correlation: {stats['correlation']:.6f}")
    print(f"  MAE: {stats['mae']:.6f}")
    print(f"  Relative error: {stats['relative_error']:.4%}")
    
    # Test multiplication
    a = torch.tensor([1.0, 2.0, 0.5])
    b = torch.tensor([2.0, 0.5, 3.0])
    
    a_signs, a_exps = encoder.encode(a)
    b_signs, b_exps = encoder.encode(b)
    
    # Multiply in φ-space
    prod_signs = a_signs * b_signs
    prod_exps = encoder.multiply_exponents(a_exps, b_exps)
    prod_phi = encoder.decode(prod_signs, prod_exps)
    
    # Compare to direct multiplication
    prod_direct = a * b
    
    print(f"\n  Multiplication test:")
    print(f"    Direct: {prod_direct.tolist()}")
    print(f"    φ-space: {prod_phi.tolist()}")
    
    return encoder


if __name__ == "__main__":
    test_encoder()
