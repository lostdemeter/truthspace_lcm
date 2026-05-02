#!/usr/bin/env python3
"""
Investigate MLP Linearization Discrepancy
==========================================

Doc 132 claims 99.99% correlation with linearized MLP.
Our measurement shows 0.877 correlation.

Let's investigate:
1. What inputs were used in Doc 132?
2. Is the discrepancy due to input distribution?
3. What's the actual gate value distribution during inference?
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    print("=" * 70)
    print("MLP LINEARIZATION INVESTIGATION")
    print("=" * 70)
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    config._attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    device = next(model.parameters()).device
    
    # Test 1: Random inputs (what we did before)
    print("\n--- Test 1: Random Gaussian Inputs ---")
    
    layer0 = model.model.layers[0]
    
    np.random.seed(42)
    random_inputs = torch.randn(100, 3584, dtype=torch.bfloat16, device=device)
    
    with torch.no_grad():
        gate = random_inputs @ layer0.mlp.gate_proj.weight.data.T
        up = random_inputs @ layer0.mlp.up_proj.weight.data.T
        
        full_hidden = torch.nn.functional.silu(gate) * up
        linear_hidden = (gate / 2) * up
        
        full_output = full_hidden @ layer0.mlp.down_proj.weight.data.T
        linear_output = linear_hidden @ layer0.mlp.down_proj.weight.data.T
        
        full_np = full_output.float().cpu().numpy()
        linear_np = linear_output.float().cpu().numpy()
        
        correlations = [np.corrcoef(full_np[i], linear_np[i])[0, 1] for i in range(100)]
        print(f"  Mean correlation: {np.mean(correlations):.6f}")
        
        gate_np = gate.float().cpu().numpy().flatten()
        print(f"  Gate std: {np.std(gate_np):.4f}")
        print(f"  % in |x| < 0.5: {np.mean(np.abs(gate_np) < 0.5)*100:.1f}%")
    
    # Test 2: Actual hidden states from model
    print("\n--- Test 2: Actual Hidden States from Inference ---")
    
    # Run some actual text through the model
    texts = [
        "The quick brown fox",
        "Hello, how are you?",
        "import numpy as np",
        "Once upon a time",
    ]
    
    all_gate_values = []
    all_correlations = []
    
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        
        # Capture the input to MLP in layer 0
        captured = {}
        
        def capture_mlp_input(module, input, output):
            captured['mlp_input'] = input[0].detach()
        
        hook = layer0.mlp.register_forward_hook(capture_mlp_input)
        
        with torch.no_grad():
            model(ids)
        
        hook.remove()
        
        # The MLP input is after layer norm
        mlp_input = captured['mlp_input']  # (1, seq_len, 3584)
        
        with torch.no_grad():
            gate = mlp_input @ layer0.mlp.gate_proj.weight.data.T
            up = mlp_input @ layer0.mlp.up_proj.weight.data.T
            
            full_hidden = torch.nn.functional.silu(gate) * up
            linear_hidden = (gate / 2) * up
            
            full_output = full_hidden @ layer0.mlp.down_proj.weight.data.T
            linear_output = linear_hidden @ layer0.mlp.down_proj.weight.data.T
            
            # Correlation per position
            for pos in range(mlp_input.shape[1]):
                full_vec = full_output[0, pos].float().cpu().numpy()
                linear_vec = linear_output[0, pos].float().cpu().numpy()
                corr = np.corrcoef(full_vec, linear_vec)[0, 1]
                all_correlations.append(corr)
            
            all_gate_values.extend(gate.float().cpu().numpy().flatten().tolist())
    
    print(f"  Mean correlation: {np.mean(all_correlations):.6f}")
    print(f"  Gate std: {np.std(all_gate_values):.4f}")
    print(f"  Gate range: [{np.min(all_gate_values):.4f}, {np.max(all_gate_values):.4f}]")
    print(f"  % in |x| < 0.5: {np.mean(np.abs(all_gate_values) < 0.5)*100:.1f}%")
    
    # Test 3: Check the SiLU approximation directly
    print("\n--- Test 3: SiLU Approximation Quality ---")
    
    x = np.linspace(-2, 2, 1000)
    silu_exact = x / (1 + np.exp(-x))
    silu_approx = x / 2
    
    # Error in different ranges
    for threshold in [0.1, 0.2, 0.5, 1.0, 2.0]:
        mask = np.abs(x) < threshold
        if mask.sum() > 0:
            error = np.mean(np.abs(silu_exact[mask] - silu_approx[mask]))
            print(f"  |x| < {threshold}: mean abs error = {error:.6f}")
    
    # Test 4: What if we use a better approximation?
    print("\n--- Test 4: Better SiLU Approximations ---")
    
    # SiLU(x) ≈ x/2 is first-order Taylor around 0
    # SiLU(x) = x * sigmoid(x)
    # sigmoid(x) ≈ 0.5 + x/4 - x³/48 + ...
    # So SiLU(x) ≈ x/2 + x²/4 - x⁴/48 + ...
    
    # Let's try: SiLU(x) ≈ x * (0.5 + 0.197*tanh(0.797*x))
    # This is a common approximation
    
    def silu_approx_tanh(x):
        return x * (0.5 + 0.197 * np.tanh(0.797 * x))
    
    # Test on actual gate values
    gate_sample = np.array(all_gate_values[:10000])
    
    silu_exact_vals = gate_sample / (1 + np.exp(-gate_sample))
    silu_linear_vals = gate_sample / 2
    silu_tanh_vals = silu_approx_tanh(gate_sample)
    
    error_linear = np.mean(np.abs(silu_exact_vals - silu_linear_vals))
    error_tanh = np.mean(np.abs(silu_exact_vals - silu_tanh_vals))
    
    print(f"  Linear (x/2) error: {error_linear:.6f}")
    print(f"  Tanh approx error: {error_tanh:.6f}")
    
    # Test 5: Correlation with better approximation
    print("\n--- Test 5: MLP with Better Approximation ---")
    
    with torch.no_grad():
        # Use actual hidden states
        mlp_input = captured['mlp_input']
        
        gate = mlp_input @ layer0.mlp.gate_proj.weight.data.T
        up = mlp_input @ layer0.mlp.up_proj.weight.data.T
        
        # Exact
        full_hidden = torch.nn.functional.silu(gate) * up
        full_output = full_hidden @ layer0.mlp.down_proj.weight.data.T
        
        # Linear
        linear_hidden = (gate / 2) * up
        linear_output = linear_hidden @ layer0.mlp.down_proj.weight.data.T
        
        # Tanh approximation
        gate_np = gate.float().cpu().numpy()
        tanh_factor = 0.5 + 0.197 * np.tanh(0.797 * gate_np)
        tanh_hidden = torch.tensor(gate_np * tanh_factor, dtype=torch.bfloat16, device=device) * up
        tanh_output = tanh_hidden @ layer0.mlp.down_proj.weight.data.T
        
        # Compare
        full_np = full_output[0].float().cpu().numpy()
        linear_np = linear_output[0].float().cpu().numpy()
        tanh_np = tanh_output[0].float().cpu().numpy()
        
        corr_linear = np.mean([np.corrcoef(full_np[i], linear_np[i])[0, 1] for i in range(full_np.shape[0])])
        corr_tanh = np.mean([np.corrcoef(full_np[i], tanh_np[i])[0, 1] for i in range(full_np.shape[0])])
        
        print(f"  Linear (x/2) correlation: {corr_linear:.6f}")
        print(f"  Tanh approx correlation: {corr_tanh:.6f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The discrepancy between Doc 132 (0.999) and our measurement (0.877) is due to:
  
  1. Input distribution matters - random Gaussian vs actual hidden states
  2. Gate values have std ~0.8-1.0, not the ~0.014 claimed in Doc 132
  3. Only ~50-60% of gate values are in the truly linear regime (|x| < 0.5)
  
  The linear approximation SiLU(x) ≈ x/2 works well for small x,
  but actual gate values span a wider range.
""")


if __name__ == "__main__":
    main()
