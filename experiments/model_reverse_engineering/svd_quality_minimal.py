#!/usr/bin/env python3
"""
Minimal SVD Quality Analysis - Test one layer at a time
========================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

DEVICE = "cuda"


def test_layer_by_layer(model, tokenizer):
    """Test SVD impact on individual layers."""
    print("\n" + "="*70)
    print("LAYER-BY-LAYER SVD SENSITIVITY TEST")
    print("="*70)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline
    with torch.no_grad():
        baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_top1 = baseline_probs.argmax().item()
        baseline_token = tokenizer.decode([baseline_top1])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Baseline prediction: '{baseline_token}' ({baseline_probs[0, baseline_top1].item()*100:.1f}%)")
    
    k = 2000  # Test with k=2000
    
    print(f"\nTesting each layer with k={k} SVD truncation:")
    print("-" * 60)
    
    results = []
    
    for layer_idx in range(model.config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        
        # Store original weights
        original_weights = {}
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, name)
            original_weights[name] = proj.weight.data.clone()
        
        # Apply SVD truncation to this layer only
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, name)
            W = proj.weight.data.float().cpu()
            
            if min(W.shape) > k:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
                proj.weight.data = W_approx.to(proj.weight.dtype).cuda()
                del U, S, Vt, W_approx
            del W
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Test prediction
        with torch.no_grad():
            approx_outputs = model(**inputs)
            approx_logits = approx_outputs.logits[:, -1, :]
            approx_probs = F.softmax(approx_logits, dim=-1)
            approx_top1 = approx_probs.argmax().item()
        
        # KL divergence
        kl_div = F.kl_div(
            approx_probs.log(),
            baseline_probs,
            reduction='batchmean'
        ).item()
        
        match = approx_top1 == baseline_top1
        correct_prob = approx_probs[0, baseline_top1].item()
        
        results.append({
            'layer': layer_idx,
            'kl_div': kl_div,
            'match': match,
            'correct_prob': correct_prob,
        })
        
        status = "✓" if match else "✗"
        print(f"Layer {layer_idx:2d}: KL={kl_div:.6f}, correct_prob={correct_prob*100:.1f}% {status}")
        
        # Restore original weights
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, name)
            proj.weight.data = original_weights[name]
        
        del original_weights
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "-"*60)
    mismatches = sum(1 for r in results if not r['match'])
    avg_kl = np.mean([r['kl_div'] for r in results])
    max_kl = max(r['kl_div'] for r in results)
    max_kl_layer = [r['layer'] for r in results if r['kl_div'] == max_kl][0]
    
    print(f"Mismatches: {mismatches}/{len(results)}")
    print(f"Avg KL divergence: {avg_kl:.6f}")
    print(f"Max KL divergence: {max_kl:.6f} (layer {max_kl_layer})")
    
    return results


def test_cumulative_layers(model, tokenizer):
    """Test SVD on cumulative layers (1, 2, 4, 8, 16, 28)."""
    print("\n" + "="*70)
    print("CUMULATIVE LAYER SVD TEST")
    print("="*70)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline
    with torch.no_grad():
        baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_top1 = baseline_probs.argmax().item()
        baseline_token = tokenizer.decode([baseline_top1])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Baseline: '{baseline_token}' ({baseline_probs[0, baseline_top1].item()*100:.1f}%)")
    
    k = 2500
    layer_counts = [1, 2, 4, 8, 14, 28]
    
    print(f"\nTesting cumulative layers with k={k}:")
    print("-" * 60)
    
    for n_layers in layer_counts:
        # Store original weights
        original_weights = {}
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            original_weights[layer_idx] = {}
            
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                original_weights[layer_idx][name] = proj.weight.data.clone()
                
                W = proj.weight.data.float().cpu()
                if min(W.shape) > k:
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
                    proj.weight.data = W_approx.to(proj.weight.dtype).cuda()
                    del U, S, Vt, W_approx
                del W
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Test prediction
        with torch.no_grad():
            approx_outputs = model(**inputs)
            approx_logits = approx_outputs.logits[:, -1, :]
            approx_probs = F.softmax(approx_logits, dim=-1)
            approx_top1 = approx_probs.argmax().item()
            approx_token = tokenizer.decode([approx_top1])
        
        kl_div = F.kl_div(
            approx_probs.log(),
            baseline_probs,
            reduction='batchmean'
        ).item()
        
        match = approx_top1 == baseline_top1
        correct_prob = approx_probs[0, baseline_top1].item()
        
        status = "✓" if match else f"✗ (got '{approx_token}')"
        print(f"{n_layers:2d} layers: KL={kl_div:.6f}, correct_prob={correct_prob*100:.1f}% {status}")
        
        # Restore weights
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                proj.weight.data = original_weights[layer_idx][name]
        
        del original_weights
        gc.collect()
        torch.cuda.empty_cache()


def test_k_threshold(model, tokenizer):
    """Find minimum k that preserves prediction."""
    print("\n" + "="*70)
    print("MINIMUM K THRESHOLD TEST (4 middle layers)")
    print("="*70)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline
    with torch.no_grad():
        baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_top1 = baseline_probs.argmax().item()
        baseline_token = tokenizer.decode([baseline_top1])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Baseline: '{baseline_token}' ({baseline_probs[0, baseline_top1].item()*100:.1f}%)")
    
    # Test layers 12-15 (middle layers)
    test_layers = [12, 13, 14, 15]
    k_values = [1000, 1500, 2000, 2500, 3000, 3200, 3400, 3500]
    
    print(f"\nTesting layers {test_layers} with different k values:")
    print("-" * 60)
    
    for k in k_values:
        # Store original weights
        original_weights = {}
        
        for layer_idx in test_layers:
            layer = model.model.layers[layer_idx]
            original_weights[layer_idx] = {}
            
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                original_weights[layer_idx][name] = proj.weight.data.clone()
                
                W = proj.weight.data.float().cpu()
                if min(W.shape) > k:
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
                    proj.weight.data = W_approx.to(proj.weight.dtype).cuda()
                    del U, S, Vt, W_approx
                del W
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Test prediction
        with torch.no_grad():
            approx_outputs = model(**inputs)
            approx_logits = approx_outputs.logits[:, -1, :]
            approx_probs = F.softmax(approx_logits, dim=-1)
            approx_top1 = approx_probs.argmax().item()
            approx_token = tokenizer.decode([approx_top1])
        
        kl_div = F.kl_div(
            approx_probs.log(),
            baseline_probs,
            reduction='batchmean'
        ).item()
        
        match = approx_top1 == baseline_top1
        correct_prob = approx_probs[0, baseline_top1].item()
        
        status = "✓" if match else f"✗ (got '{approx_token}')"
        print(f"k={k:4d}: KL={kl_div:.6f}, correct_prob={correct_prob*100:.1f}% {status}")
        
        # Restore weights
        for layer_idx in test_layers:
            layer = model.model.layers[layer_idx]
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                proj.weight.data = original_weights[layer_idx][name]
        
        del original_weights
        gc.collect()
        torch.cuda.empty_cache()


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
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    # Test 1: Layer-by-layer sensitivity
    test_layer_by_layer(model, tokenizer)
    
    # Test 2: Cumulative layers
    test_cumulative_layers(model, tokenizer)
    
    # Test 3: K threshold
    test_k_threshold(model, tokenizer)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
KEY FINDINGS FROM PREVIOUS ANALYSIS:
- down_proj needs k=3400+ for 99% variance (nearly FULL RANK)
- Zipf α ≈ 0.09-0.16 (NOT 1/φ = 0.618) - singular values decay SLOWLY
- At k=2000: only 77-79% variance captured for down_proj

IMPLICATION:
SVD-based LOD cannot achieve significant speedup without quality loss
because the MLP weights are nearly full-rank. The "long tail" of 
singular values carries critical semantic information.
""")


if __name__ == "__main__":
    main()
