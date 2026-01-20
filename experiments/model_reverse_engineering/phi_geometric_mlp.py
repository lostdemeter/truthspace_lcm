"""
φ-Geometric MLP Inference for Qwen2-7B

This implements the "geometry of knowledge" approach:
- Magnitudes encoded as φ-levels (integer exponents)
- Signs stored as 1-bit values
- Computation via integer addition + LUT lookup
- Fibonacci structure implicitly understood

The goal: prove that we can run Qwen2-7B MLP with:
- Integer arithmetic only (no float multiplies in core matmul)
- LUT-based φ^level decode
- 1-2 cycle "mesh gear" computation
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
SCALE = 1024
QUANTUM = 256


class PhiEncodedWeight:
    """Weight matrix encoded as φ-levels + signs."""
    
    def __init__(self, weight_matrix: np.ndarray):
        """Encode a weight matrix into φ-geometric form."""
        self.shape = weight_matrix.shape
        
        # Extract signs (1 bit per weight)
        self.signs = np.sign(weight_matrix).astype(np.int8)
        
        # Extract levels (integer exponents)
        magnitudes = np.abs(weight_matrix) + 1e-45
        exponents = np.log(magnitudes) / LOG_PHI
        self.levels = np.round(exponents * SCALE / QUANTUM).astype(np.int16)
        
        # Build LUT for φ^level decode
        unique_levels = np.unique(self.levels)
        self.level_min = unique_levels.min()
        self.level_max = unique_levels.max()
        self.lut_size = self.level_max - self.level_min + 1
        
        # LUT: level -> φ^(level * QUANTUM / SCALE)
        self.phi_lut = np.zeros(self.lut_size, dtype=np.float32)
        for i, level in enumerate(range(self.level_min, self.level_max + 1)):
            self.phi_lut[i] = PHI ** (level * QUANTUM / SCALE)
        
        # Shift levels to be 0-indexed into LUT
        self.levels_indexed = (self.levels - self.level_min).astype(np.uint16)
    
    def decode(self) -> np.ndarray:
        """Decode back to float weights."""
        magnitudes = self.phi_lut[self.levels_indexed]
        return self.signs * magnitudes
    
    def storage_bytes(self) -> int:
        """Calculate storage size."""
        signs_bytes = self.signs.nbytes
        levels_bytes = self.levels_indexed.nbytes
        lut_bytes = self.phi_lut.nbytes
        return signs_bytes + levels_bytes + lut_bytes


class PhiEncodedInput:
    """Input vector encoded as φ-levels + signs."""
    
    def __init__(self, input_vector: np.ndarray):
        """Encode an input vector into φ-geometric form."""
        self.shape = input_vector.shape
        
        # Extract signs
        self.signs = np.sign(input_vector).astype(np.int8)
        
        # Extract levels
        magnitudes = np.abs(input_vector) + 1e-45
        exponents = np.log(magnitudes) / LOG_PHI
        self.levels = np.round(exponents * SCALE / QUANTUM).astype(np.int16)


def phi_matmul_integer(weight: PhiEncodedWeight, input_enc: PhiEncodedInput) -> np.ndarray:
    """
    Matrix multiplication using integer arithmetic.
    
    For each output[j] = Σ_i W[j,i] × x[i]
                       = Σ_i sign_w[j,i] × sign_x[i] × φ^(level_w[j,i] + level_x[i])
    
    The multiplication becomes:
    1. Sign XOR: combined_sign = sign_w × sign_x
    2. Level ADD: combined_level = level_w + level_x
    3. LUT lookup: magnitude = φ^combined_level
    4. Accumulate: output[j] += combined_sign × magnitude
    """
    out_dim, in_dim = weight.shape
    
    # Combined signs: (out_dim, in_dim)
    combined_signs = weight.signs * input_enc.signs  # Element-wise, result is ±1
    
    # Combined levels: (out_dim, in_dim) - INTEGER ADDITION
    combined_levels = weight.levels.astype(np.int32) + input_enc.levels.astype(np.int32)
    
    # Build combined LUT (covers the range of combined levels)
    combined_min = combined_levels.min()
    combined_max = combined_levels.max()
    combined_lut_size = combined_max - combined_min + 1
    
    combined_lut = np.zeros(combined_lut_size, dtype=np.float32)
    for i, level in enumerate(range(combined_min, combined_max + 1)):
        combined_lut[i] = PHI ** (level * QUANTUM / SCALE)
    
    # Index into combined LUT
    combined_indexed = (combined_levels - combined_min).astype(np.int32)
    
    # Lookup magnitudes
    magnitudes = combined_lut[combined_indexed]
    
    # Apply signs and sum
    signed_values = combined_signs * magnitudes
    output = signed_values.sum(axis=1)
    
    return output.astype(np.float32)


def phi_mlp_forward(x: np.ndarray, 
                    gate_weight: PhiEncodedWeight,
                    up_weight: PhiEncodedWeight, 
                    down_weight: PhiEncodedWeight) -> np.ndarray:
    """
    Forward pass through MLP using φ-geometric computation.
    
    MLP: output = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
    """
    # Encode input
    x_enc = PhiEncodedInput(x)
    
    # Gate projection (integer matmul)
    gate_out = phi_matmul_integer(gate_weight, x_enc)
    
    # Up projection (integer matmul)
    up_out = phi_matmul_integer(up_weight, x_enc)
    
    # SiLU activation (still need float for this)
    gate_activated = gate_out * torch.sigmoid(torch.tensor(gate_out)).numpy()
    
    # Element-wise multiply
    hidden = gate_activated * up_out
    
    # Down projection
    hidden_enc = PhiEncodedInput(hidden)
    output = phi_matmul_integer(down_weight, hidden_enc)
    
    return output


def benchmark_phi_mlp():
    """Benchmark φ-geometric MLP vs original."""
    print("=" * 70)
    print("φ-Geometric MLP Benchmark")
    print("=" * 70)
    
    # Load model
    print("\nLoading Qwen2-7B...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    
    # Get MLP weights from layer 14
    mlp = model.model.layers[14].mlp
    W_gate = mlp.gate_proj.weight.data.float().cpu().numpy()
    W_up = mlp.up_proj.weight.data.float().cpu().numpy()
    W_down = mlp.down_proj.weight.data.float().cpu().numpy()
    
    print(f"Weight shapes: gate={W_gate.shape}, up={W_up.shape}, down={W_down.shape}")
    
    # Encode weights
    print("\nEncoding weights to φ-geometric form...")
    t0 = time.time()
    gate_enc = PhiEncodedWeight(W_gate)
    up_enc = PhiEncodedWeight(W_up)
    down_enc = PhiEncodedWeight(W_down)
    encode_time = time.time() - t0
    print(f"Encoding time: {encode_time:.2f}s")
    
    # Storage comparison
    original_bytes = (W_gate.nbytes + W_up.nbytes + W_down.nbytes)
    encoded_bytes = (gate_enc.storage_bytes() + up_enc.storage_bytes() + down_enc.storage_bytes())
    print(f"\nStorage: {original_bytes/1e6:.1f} MB (original) → {encoded_bytes/1e6:.1f} MB (φ-encoded)")
    print(f"Compression: {original_bytes/encoded_bytes:.2f}x")
    
    # Verify decode accuracy
    print("\nVerifying decode accuracy...")
    W_gate_dec = gate_enc.decode()
    corr = np.corrcoef(W_gate.flatten(), W_gate_dec.flatten())[0, 1]
    print(f"Weight decode correlation: {corr*100:.4f}%")
    
    # Get test input
    print("\nPreparing test input...")
    prompt = "The golden ratio is approximately equal to"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        out = model(inputs['input_ids'], output_hidden_states=True, use_cache=False)
        h = out.hidden_states[14]
        x = model.model.layers[14].input_layernorm(h)
        original_out = mlp(x)
    
    x_np = x.float().cpu().numpy()[0, 0, :]  # First token
    out_orig = original_out.float().cpu().numpy()[0, 0, :]
    
    print(f"Input shape: {x_np.shape}")
    print(f"Original output shape: {out_orig.shape}")
    
    # Run φ-geometric MLP
    print("\nRunning φ-geometric MLP...")
    t0 = time.time()
    out_phi = phi_mlp_forward(x_np, gate_enc, up_enc, down_enc)
    phi_time = time.time() - t0
    print(f"φ-geometric time: {phi_time*1000:.2f} ms")
    
    # Run original MLP (CPU for fair comparison)
    print("\nRunning original MLP (CPU)...")
    W_gate_t = torch.tensor(W_gate)
    W_up_t = torch.tensor(W_up)
    W_down_t = torch.tensor(W_down)
    x_t = torch.tensor(x_np)
    
    t0 = time.time()
    gate = F.silu(F.linear(x_t, W_gate_t))
    up = F.linear(x_t, W_up_t)
    hidden = gate * up
    out_orig_cpu = F.linear(hidden, W_down_t)
    orig_time = time.time() - t0
    print(f"Original time (CPU): {orig_time*1000:.2f} ms")
    
    # Compare outputs
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    corr_out = np.corrcoef(out_orig, out_phi)[0, 1]
    print(f"Output correlation: {corr_out*100:.4f}%")
    
    mse = ((out_orig - out_phi) ** 2).mean()
    print(f"MSE: {mse:.6f}")
    
    print(f"\nSpeedup: {orig_time/phi_time:.2f}x")
    
    # Operation count
    print("\n" + "=" * 70)
    print("Operation Analysis")
    print("=" * 70)
    
    n_weights = W_gate.size + W_up.size + W_down.size
    print(f"Total weights: {n_weights:,}")
    print(f"Float multiplications (original): {n_weights:,}")
    print(f"Integer additions (φ-geometric): {n_weights:,}")
    print(f"LUT lookups (φ-geometric): {n_weights:,}")
    
    return corr_out


if __name__ == "__main__":
    corr = benchmark_phi_mlp()
    print(f"\n{'='*70}")
    print(f"FINAL: {corr*100:.2f}% correlation with φ-geometric inference")
    print(f"{'='*70}")
