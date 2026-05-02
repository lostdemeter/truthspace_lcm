#!/usr/bin/env python3
"""
Deep Diagnose: Why is Q/K projection only 0.35 cosine?
=======================================================

Q projection (before RoPE): 0.35 cosine
K projection (before RoPE): 0.14 cosine

This should be 1.0 if we're just doing h @ W.T

Let's check:
1. Is there a bias?
2. Is the layer norm different?
3. Is there something else in the attention module?

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DeepDiagnoseAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def inspect_layer3_structure(self):
        """Inspect the structure of layer 3 attention."""
        print(f"\n--- Inspecting Layer 3 Structure ---")
        
        layer3 = self.model.model.layers[3]
        attn = layer3.self_attn
        
        print(f"\n  Attention module type: {type(attn)}")
        print(f"\n  Attention module attributes:")
        for name, param in attn.named_parameters():
            print(f"    {name}: {param.shape}")
        
        print(f"\n  Attention module children:")
        for name, child in attn.named_children():
            print(f"    {name}: {type(child)}")
        
        # Check for biases
        print(f"\n  Checking for biases:")
        print(f"    q_proj.bias: {attn.q_proj.bias is not None}")
        print(f"    k_proj.bias: {attn.k_proj.bias is not None}")
        print(f"    v_proj.bias: {attn.v_proj.bias is not None}")
        print(f"    o_proj.bias: {attn.o_proj.bias is not None}")
        
        # Check layer norm
        ln = layer3.input_layernorm
        print(f"\n  Layer norm type: {type(ln)}")
        print(f"    weight shape: {ln.weight.shape}")
        if hasattr(ln, 'bias') and ln.bias is not None:
            print(f"    bias shape: {ln.bias.shape}")
        else:
            print(f"    bias: None")
        if hasattr(ln, 'eps'):
            print(f"    eps: {ln.eps}")
        
        # Check RoPE
        if hasattr(attn, 'rotary_emb'):
            rotary = attn.rotary_emb
            print(f"\n  RoPE type: {type(rotary)}")
            for name, param in rotary.named_parameters():
                print(f"    {name}: {param.shape}")
            for name, buf in rotary.named_buffers():
                print(f"    {name} (buffer): {buf.shape}")
    
    def trace_single_forward(self, A: int, B: int):
        """Trace a single forward pass through layer 3."""
        print(f"\n--- Tracing Forward Pass (A={A}, B={B}) ---")
        
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        layer3 = self.model.model.layers[3]
        attn = layer3.self_attn
        
        captured = {}
        
        # Capture layer norm input and output
        def capture_ln(module, input, output):
            captured['ln_input'] = input[0].detach().float().cpu().numpy()
            captured['ln_output'] = output.detach().float().cpu().numpy()
        
        # Capture Q, K, V projections
        def capture_q(module, input, output):
            captured['q_input'] = input[0].detach().float().cpu().numpy()
            captured['q_output'] = output.detach().float().cpu().numpy()
        
        def capture_k(module, input, output):
            captured['k_input'] = input[0].detach().float().cpu().numpy()
            captured['k_output'] = output.detach().float().cpu().numpy()
        
        def capture_v(module, input, output):
            captured['v_input'] = input[0].detach().float().cpu().numpy()
            captured['v_output'] = output.detach().float().cpu().numpy()
        
        hooks = [
            layer3.input_layernorm.register_forward_hook(capture_ln),
            attn.q_proj.register_forward_hook(capture_q),
            attn.k_proj.register_forward_hook(capture_k),
            attn.v_proj.register_forward_hook(capture_v),
        ]
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
            
            print(f"\n  Captured shapes:")
            for key, val in captured.items():
                print(f"    {key}: {val.shape}")
            
            # Check if Q input matches LN output
            print(f"\n  Checking Q input vs LN output:")
            q_input = captured['q_input'][0]  # (2, hidden)
            ln_output = captured['ln_output'][0]  # (2, hidden)
            
            cos_q_ln = np.dot(q_input[1].flatten(), ln_output[1].flatten()) / (
                np.linalg.norm(q_input[1]) * np.linalg.norm(ln_output[1]) + 1e-10)
            print(f"    Cosine (Q input vs LN output): {cos_q_ln:.6f}")
            print(f"    Max diff: {np.abs(q_input - ln_output).max():.6f}")
            
            # Manual Q projection
            print(f"\n  Manual Q projection:")
            W_q = attn.q_proj.weight.data.float().cpu().numpy()
            b_q = attn.q_proj.bias
            if b_q is not None:
                b_q = b_q.data.float().cpu().numpy()
                print(f"    Q bias shape: {b_q.shape}")
            
            q_manual = ln_output[1] @ W_q.T
            if b_q is not None:
                q_manual = q_manual + b_q
            
            q_actual = captured['q_output'][0, 1]
            
            cos_q = np.dot(q_manual, q_actual) / (
                np.linalg.norm(q_manual) * np.linalg.norm(q_actual) + 1e-10)
            print(f"    Cosine (manual Q vs actual Q): {cos_q:.6f}")
            print(f"    Max diff: {np.abs(q_manual - q_actual).max():.6f}")
            
            # Check norms
            print(f"\n  Norms:")
            print(f"    LN output norm: {np.linalg.norm(ln_output[1]):.4f}")
            print(f"    Q actual norm: {np.linalg.norm(q_actual):.4f}")
            print(f"    Q manual norm: {np.linalg.norm(q_manual):.4f}")
            
            # Check first few values
            print(f"\n  First 10 values:")
            print(f"    Q actual: {q_actual[:10]}")
            print(f"    Q manual: {q_manual[:10]}")
            print(f"    Diff:     {(q_actual - q_manual)[:10]}")
            
            return captured
            
        finally:
            for hook in hooks:
                hook.remove()
    
    def test_precision(self, A: int, B: int):
        """Test if precision is the issue."""
        print(f"\n--- Testing Precision ---")
        
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        layer3 = self.model.model.layers[3]
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            h2 = outputs.hidden_states[3][0]  # (2, hidden)
        
        # Try computing in float16 vs float32
        ln = layer3.input_layernorm
        W_q = layer3.self_attn.q_proj.weight.data
        b_q = layer3.self_attn.q_proj.bias
        
        # Float16 computation
        h2_16 = h2.half()
        ln_out_16 = ln(h2_16)
        q_16 = ln_out_16 @ W_q.T
        if b_q is not None:
            q_16 = q_16 + b_q
        
        # Float32 computation
        h2_32 = h2.float()
        ln_weight_32 = ln.weight.float()
        
        # Manual RMS norm in float32
        rms = torch.sqrt(torch.mean(h2_32[1]**2) + 1e-6)
        ln_out_32 = (h2_32[1] / rms) * ln_weight_32
        
        W_q_32 = W_q.float()
        q_32 = ln_out_32 @ W_q_32.T
        if b_q is not None:
            q_32 = q_32 + b_q.float()
        
        # Compare
        q_16_np = q_16[0, 1].float().cpu().numpy()
        q_32_np = q_32.cpu().numpy()
        
        cos = np.dot(q_16_np, q_32_np) / (
            np.linalg.norm(q_16_np) * np.linalg.norm(q_32_np) + 1e-10)
        
        print(f"  Cosine (float16 vs float32): {cos:.6f}")
        print(f"  Max diff: {np.abs(q_16_np - q_32_np).max():.6f}")
        
        # Now compare float32 manual to actual
        captured = {}
        
        def capture_q(module, input, output):
            captured['q'] = output.detach().float().cpu().numpy()
        
        hook = layer3.self_attn.q_proj.register_forward_hook(capture_q)
        
        try:
            with torch.no_grad():
                self.model(input_ids)
            
            q_actual = captured['q'][0, 1]
            
            cos_actual = np.dot(q_32_np, q_actual) / (
                np.linalg.norm(q_32_np) * np.linalg.norm(q_actual) + 1e-10)
            
            print(f"  Cosine (float32 manual vs actual): {cos_actual:.6f}")
            
        finally:
            hook.remove()


def main():
    print("=" * 70)
    print("DEEP DIAGNOSE: WHY IS Q/K PROJECTION ONLY 0.35 COSINE?")
    print("=" * 70)
    
    analyzer = DeepDiagnoseAnalyzer()
    
    # 1. Inspect layer 3 structure
    analyzer.inspect_layer3_structure()
    
    # 2. Trace a single forward pass
    captured = analyzer.trace_single_forward(100, 200)
    
    # 3. Test precision
    analyzer.test_precision(100, 200)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
