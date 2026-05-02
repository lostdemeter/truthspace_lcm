#!/usr/bin/env python3
"""
Simple SVD Quality Analysis - Memory Efficient
==============================================

Test SVD approximation quality one layer at a time.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_single_token_prediction(model, tokenizer, k_values=[1000, 1500, 2000, 2500, 3000, 3500]):
    """Test how SVD affects single token prediction."""
    print("\n" + "="*70)
    print("SINGLE TOKEN PREDICTION TEST")
    print("="*70)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline
    with torch.no_grad():
        baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_top5 = torch.topk(baseline_probs, 5)
    
    print(f"\nPrompt: '{prompt}'")
    print("\nBaseline top-5:")
    for i in range(5):
        token_id = baseline_top5.indices[0, i].item()
        prob = baseline_top5.values[0, i].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token}' ({prob*100:.1f}%)")
    
    baseline_top1 = baseline_top5.indices[0, 0].item()
    
    print("\n" + "-"*50)
    print("Testing SVD truncation (ALL 28 layers, MLP only):")
    print("-"*50)
    
    for k in k_values:
        print(f"\nk={k}...")
        
        # Store original weights and apply SVD
        original_weights = {}
        
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            original_weights[layer_idx] = {}
            
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                original_weights[layer_idx][name] = proj.weight.data.clone()
                
                # Do SVD on CPU
                with torch.no_grad():
                    W = proj.weight.data.float().cpu()
                    if min(W.shape) > k:
                        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                        W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
                        proj.weight.data = W_approx.to(proj.weight.dtype).to(DEVICE)
                        del U, S, Vt, W_approx
                    del W
                gc.collect()
                torch.cuda.empty_cache()
        
        # Test prediction
        with torch.no_grad():
            approx_outputs = model(**inputs)
            approx_logits = approx_outputs.logits[:, -1, :]
            approx_probs = F.softmax(approx_logits, dim=-1)
            approx_top5 = torch.topk(approx_probs, 5)
        
        approx_top1 = approx_top5.indices[0, 0].item()
        
        # KL divergence
        kl_div = F.kl_div(
            approx_probs.log(),
            baseline_probs,
            reduction='batchmean'
        ).item()
        
        top1_match = baseline_top1 == approx_top1
        correct_prob = approx_probs[0, baseline_top1].item()
        baseline_correct_prob = baseline_probs[0, baseline_top1].item()
        
        print(f"  Top-1 match: {top1_match}")
        print(f"  KL divergence: {kl_div:.6f}")
        print(f"  Correct token prob: {correct_prob*100:.1f}% (baseline: {baseline_correct_prob*100:.1f}%)")
        
        if not top1_match:
            approx_token = tokenizer.decode([approx_top1])
            baseline_token = tokenizer.decode([baseline_top1])
            print(f"  MISMATCH: '{baseline_token}' → '{approx_token}'")
        
        # Restore weights
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                proj.weight.data = original_weights[layer_idx][name]
        
        del original_weights
        gc.collect()
        torch.cuda.empty_cache()


def test_generation(model, tokenizer, k_values=[2000, 2500, 3000, 3500]):
    """Test how SVD affects multi-token generation."""
    print("\n" + "="*70)
    print("MULTI-TOKEN GENERATION TEST")
    print("="*70)
    
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Baseline: {baseline_text}")
    
    print("\n" + "-"*50)
    
    for k in k_values:
        print(f"\nk={k}...")
        
        # Store original weights and apply SVD
        original_weights = {}
        
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            original_weights[layer_idx] = {}
            
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                original_weights[layer_idx][name] = proj.weight.data.clone()
                
                with torch.no_grad():
                    W = proj.weight.data.float().cpu()
                    if min(W.shape) > k:
                        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                        W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
                        proj.weight.data = W_approx.to(proj.weight.dtype).to(DEVICE)
                        del U, S, Vt, W_approx
                    del W
                gc.collect()
                torch.cuda.empty_cache()
        
        # Generate
        with torch.no_grad():
            approx_output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        approx_text = tokenizer.decode(approx_output[0], skip_special_tokens=True)
        
        # Count matching tokens
        baseline_tokens = baseline_output[0].tolist()
        approx_tokens = approx_output[0].tolist()
        matches = sum(1 for b, a in zip(baseline_tokens, approx_tokens) if b == a)
        total = len(baseline_tokens)
        
        print(f"  Token match: {matches}/{total} ({matches/total*100:.0f}%)")
        print(f"  Output: {approx_text}")
        
        # Restore weights
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                proj.weight.data = original_weights[layer_idx][name]
        
        del original_weights
        gc.collect()
        torch.cuda.empty_cache()


def analyze_variance_explained():
    """Summarize variance explained from previous run."""
    print("\n" + "="*70)
    print("VARIANCE EXPLAINED SUMMARY (from previous analysis)")
    print("="*70)
    
    # Data from previous run
    data = """
    Layer 0 - gate_proj: 90%@1353, 95%@1725, 99%@2585, Zipf α=0.239
    Layer 0 - up_proj:   90%@1565, 95%@1966, 99%@2853, Zipf α=0.083
    Layer 0 - down_proj: 90%@2621, 95%@3000, 99%@3424, Zipf α=0.152
    Layer 0 - q_proj:    90%@1002, 95%@1267, 99%@1855, Zipf α=0.376
    
    Layer 14 - gate_proj: 90%@2632, 95%@3018, 99%@3432, Zipf α=0.160
    Layer 14 - up_proj:   90%@2718, 95%@3075, 99%@3448, Zipf α=0.086
    Layer 14 - down_proj: 90%@2677, 95%@3044, 99%@3439, Zipf α=0.127
    
    Layer 21 - gate_proj: 90%@2676, 95%@3044, 99%@3435, Zipf α=0.141
    Layer 21 - up_proj:   90%@2725, 95%@3078, 99%@3447, Zipf α=0.090
    Layer 21 - down_proj: 90%@2708, 95%@3057, 99%@3433, Zipf α=0.090
    """
    
    print(data)
    
    print("\nKEY FINDINGS:")
    print("-" * 50)
    print("1. down_proj needs k=3400+ for 99% variance (nearly FULL RANK)")
    print("2. Later layers need MORE rank than early layers")
    print("3. Zipf α ≈ 0.09-0.24 (NOT 1/φ = 0.618) - SLOW decay")
    print("4. At k=2000: only 77-79% variance for down_proj")
    print("5. At k=1000: only 47-52% variance for down_proj")
    print("")
    print("IMPLICATION: SVD truncation loses 20-50% of the signal!")


def main():
    print("Loading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    
    # Summary of previous findings
    analyze_variance_explained()
    
    # Test single token prediction
    test_single_token_prediction(model, tokenizer, k_values=[2000, 2500, 3000, 3400])
    
    # Test generation
    test_generation(model, tokenizer, k_values=[2500, 3000, 3400])
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The MLP weight matrices in Qwen2-7B are NEARLY FULL RANK:
- down_proj needs k=3400/3584 (95%) for 99% variance
- Singular values decay SLOWLY (Zipf α ≈ 0.1, not 0.618)

This means:
1. SVD truncation at k<3000 loses significant information
2. The "long tail" of singular values carries critical signal
3. LOD-based speedup is fundamentally limited for this architecture

The φ-Zipf hypothesis (α ≈ 1/φ = 0.618) does NOT hold for MLP weights.
Attention weights may be different (previous work showed α ≈ 0.65).
""")


if __name__ == "__main__":
    main()
