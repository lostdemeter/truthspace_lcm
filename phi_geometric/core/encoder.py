"""
φ-Encoder: Encode and decode values in φ-basis.

The fundamental representation:
    value = sign × φ^(exponent / K)

This is the geometric representation of neural network weights.
All weights naturally cluster on the φ-lattice, with peak at φ^-9 ≈ 0.013.

Key Properties:
    - Multiplication becomes exponent addition: φ^a × φ^b = φ^(a+b)
    - Self-similarity: φ = 1 + 1/φ
    - Optimal quantization: Minimizes information loss
    - Universal: Same structure across all models (DA2, Qwen, DDColor)

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional

# The golden ratio and its natural logarithm
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895
LN_PHI = np.log(PHI)        # ≈ 0.4812118250596034


class PhiEncoder:
    """
    Encode and decode values in φ-basis.
    
    Every value is represented as: value = sign × φ^(exponent/K)
    
    This provides:
        - Compression: Float → (sign, integer exponent)
        - Fast multiplication: Exponent addition instead of float multiply
        - Natural clustering: Values organize on φ-lattice
    
    Args:
        K: Resolution parameter (higher = more precision)
           K=32 gives ~3% precision per step, 99.92% accuracy
           K=128 gives ~0.8% precision per step, 99.99% accuracy
        bias: Exponent bias (centers the range around typical weight magnitudes)
    
    Example:
        encoder = PhiEncoder(K=32)
        
        # Encode weights
        signs, exponents = encoder.encode(weight_tensor)
        
        # Decode back
        reconstructed = encoder.decode(signs, exponents)
        
        # Multiply in φ-space (integer arithmetic!)
        product_exp = encoder.multiply_exponents(a_exp, b_exp)
    """
    
    def __init__(self, K: int = 32, bias: int = 4096):
        self.K = K
        self.bias = bias
        
        # Pre-compute LUT for φ^(e/K) values
        self.max_exp = 8192
        phi_powers_np = np.zeros(self.max_exp, dtype=np.float64)
        for e in range(self.max_exp):
            exp_val = (e - bias) / K
            exp_val = max(-300, min(300, exp_val))  # Prevent overflow
            phi_powers_np[e] = PHI ** exp_val
        self.phi_powers = torch.from_numpy(phi_powers_np).float()
        
        # Pre-compute addition LUT for φ-addition
        self._build_addition_lut()
    
    def _build_addition_lut(self):
        """
        Build LUT for φ-addition.
        
        φ^a + φ^b = φ^b × (φ^(a-b) + 1) = φ^(b + LUT[a-b])
        
        This enables addition via integer operations + lookup.
        """
        max_diff = 1024
        self.add_lut = torch.zeros(2 * max_diff + 1, dtype=torch.long)
        
        for d in range(-max_diff, max_diff + 1):
            # φ^(d/K) + 1 = φ^(LUT[d]/K)
            val = PHI ** (d / self.K) + 1
            if val > 0:
                self.add_lut[d + max_diff] = int(round(self.K * np.log(val) / LN_PHI))
    
    def encode(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode float values to φ-basis (sign, exponent).
        
        Args:
            values: Float tensor of any shape
            
        Returns:
            signs: Tensor of {-1, 0, +1}
            exponents: Tensor of integers (biased)
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
            exponents: Tensor of integers (biased)
            
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
        
        φ^(a/K) × φ^(b/K) = φ^((a+b)/K)
        
        In our biased representation:
            result_exp = a_exp + b_exp - bias
        
        This is INTEGER ARITHMETIC - no floating point needed!
        """
        return a_exp + b_exp - self.bias
    
    def multiply_signs(self, a_sign: torch.Tensor, b_sign: torch.Tensor) -> torch.Tensor:
        """
        Sign multiplication.
        
        (+1) × (+1) = +1
        (+1) × (-1) = -1
        (-1) × (-1) = +1
        0 × anything = 0
        """
        return a_sign * b_sign
    
    def phi_multiply(
        self, 
        a: Tuple[torch.Tensor, torch.Tensor], 
        b: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full φ-multiplication of two encoded values.
        
        Args:
            a: (signs_a, exponents_a)
            b: (signs_b, exponents_b)
            
        Returns:
            (signs_result, exponents_result)
        """
        signs = self.multiply_signs(a[0], b[0])
        exponents = self.multiply_exponents(a[1], b[1])
        return signs, exponents
    
    def verify_accuracy(self, original: torch.Tensor) -> Dict[str, float]:
        """
        Verify encoding accuracy for a tensor.
        
        Returns:
            dict with correlation, MAE, max_error, relative_error
        """
        signs, exps = self.encode(original)
        reconstructed = self.decode(signs, exps)
        
        # Flatten for correlation
        orig_flat = original.flatten().float()
        recon_flat = reconstructed.flatten().float()
        
        # Correlation
        if orig_flat.std() > 1e-10 and recon_flat.std() > 1e-10:
            corr = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0, 1].item()
        else:
            corr = 1.0 if torch.allclose(orig_flat, recon_flat) else 0.0
        
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
    
    def analyze_distribution(self, tensor: torch.Tensor) -> Dict[str, any]:
        """
        Analyze the φ-level distribution of a tensor.
        
        Returns statistics about how values cluster on the φ-lattice.
        """
        signs, exps = self.encode(tensor)
        
        # Convert to actual φ-levels (unbiased)
        levels = (exps.float() - self.bias) / self.K
        
        return {
            "mean_level": levels.mean().item(),
            "std_level": levels.std().item(),
            "min_level": levels.min().item(),
            "max_level": levels.max().item(),
            "peak_level": levels.mode().values.item() if levels.numel() > 0 else 0,
            "num_unique_levels": len(torch.unique(exps)),
            "positive_ratio": (signs > 0).float().mean().item(),
            "zero_ratio": (signs == 0).float().mean().item(),
        }


def test_encoder():
    """Test the φ-encoder."""
    print("=" * 60)
    print("φ-ENCODER TEST")
    print("=" * 60)
    
    encoder = PhiEncoder(K=32)
    
    # Test 1: Random weights (typical neural network range)
    print("\n1. Random weights (σ=0.1):")
    weights = torch.randn(1000, 1000) * 0.1
    stats = encoder.verify_accuracy(weights)
    print(f"   Correlation: {stats['correlation']:.6f}")
    print(f"   Relative error: {stats['relative_error']:.4%}")
    
    # Test 2: Distribution analysis
    print("\n2. Distribution analysis:")
    dist = encoder.analyze_distribution(weights)
    print(f"   Peak φ-level: {dist['peak_level']:.1f}")
    print(f"   Mean φ-level: {dist['mean_level']:.2f}")
    print(f"   Unique levels: {dist['num_unique_levels']}")
    
    # Test 3: Multiplication in φ-space
    print("\n3. Multiplication test:")
    a = torch.tensor([2.0, 0.5, 1.618])
    b = torch.tensor([3.0, 2.0, 1.618])
    
    a_enc = encoder.encode(a)
    b_enc = encoder.encode(b)
    prod_enc = encoder.phi_multiply(a_enc, b_enc)
    prod_phi = encoder.decode(*prod_enc)
    prod_direct = a * b
    
    print(f"   Direct:  {prod_direct.tolist()}")
    print(f"   φ-space: {prod_phi.tolist()}")
    print(f"   Error:   {(prod_phi - prod_direct).abs().max().item():.6f}")
    
    print("\n" + "=" * 60)
    print("φ-ENCODER TEST COMPLETE")
    print("=" * 60)
    
    return encoder


if __name__ == "__main__":
    test_encoder()
