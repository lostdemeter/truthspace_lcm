"""
Phase 6: Find the exact layer where integer correlation drops.
"""

import sys, time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, get_silu_lut, get_softmax_lut, PhiRoPEInt,
    float_to_phi, phi_to_float,
)
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

# Import the layer function from the forward pass script
sys.path.insert(0, 'experiments/model_reverse_engineering_v2')
from phase6_integer_forward_pass import integer_forward_layer


def main():
    print("Finding the precision cliff...")
    
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    rope_int = PhiRoPEInt(head_dim=128, rope_theta=1_000_000.0)
    
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    
    tokens = tokenizer.encode("The capital of France is")
    
    # Start from embedding
    hidden_float = engine.embedding(tokens)[np.newaxis, :, :]
    h_s, h_e = float_to_phi(hidden_float[0])
    h_s = h_s[np.newaxis]; h_e = h_e[np.newaxis]
    
    float_out = hidden_float.copy()
    
    prev_corr = 1.0
    for i in range(28):
        t0 = time.time()
        
        # Float path
        float_out = engine.layers[i](float_out, pure=False)
        
        # Integer path
        h_s, h_e = integer_forward_layer(
            engine.layers[i], h_s, h_e, rope_int, i)
        
        # Compare
        int_out = phi_to_float(h_s, h_e)
        corr = float(np.corrcoef(float_out.flatten(), int_out.flatten())[0, 1])
        dt = time.time() - t0
        
        delta = corr - prev_corr
        marker = " ← CLIFF!" if delta < -0.01 else ""
        print(f"  Layer {i:2d}: corr={corr:.8f}  Δ={delta:+.8f}  ({dt:.1f}s){marker}")
        prev_corr = corr
        
        if corr < 0.5:
            print(f"\n  Stopping — correlation too low at layer {i}")
            break
    
    print("\nDone.")


if __name__ == '__main__':
    main()
