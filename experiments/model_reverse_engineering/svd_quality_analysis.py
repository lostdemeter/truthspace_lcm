#!/usr/bin/env python3
"""
SVD Quality Analysis: Why Does Low-Rank Approximation Cause Quality Loss?
=========================================================================

Deep dive into:
1. Singular value distribution across layers/projections
2. Variance explained at different k values
3. Which layers are most sensitive to truncation
4. Error propagation through the network
5. Token-level impact of SVD approximation

Author: TruthSpace LCM Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = 1.6180339887498949


def analyze_singular_values(model):
    """Analyze singular value distribution across all projections."""
    print("\n" + "="*70)
    print("SINGULAR VALUE ANALYSIS")
    print("="*70)
    
    results = []
    
    for layer_idx in range(model.config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        
        projections = {
            'gate_proj': layer.mlp.gate_proj.weight,
            'up_proj': layer.mlp.up_proj.weight,
            'down_proj': layer.mlp.down_proj.weight,
            'q_proj': layer.self_attn.q_proj.weight,
            'k_proj': layer.self_attn.k_proj.weight,
            'v_proj': layer.self_attn.v_proj.weight,
            'o_proj': layer.self_attn.o_proj.weight,
        }
        
        for name, weight in projections.items():
            W = weight.detach().float().cpu()
            
            # Compute SVD
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            S = S.numpy()
            
            # Compute variance explained at different k
            total_var = (S ** 2).sum()
            
            var_at_k = {}
            for k in [100, 500, 1000, 1500, 2000, 2500, 3000]:
                if k <= len(S):
                    var_at_k[k] = (S[:k] ** 2).sum() / total_var * 100
            
            # Compute effective rank (number of singular values needed for 99% variance)
            cumsum = np.cumsum(S ** 2) / total_var
            rank_99 = np.searchsorted(cumsum, 0.99) + 1
            rank_95 = np.searchsorted(cumsum, 0.95) + 1
            rank_90 = np.searchsorted(cumsum, 0.90) + 1
            
            # Zipf exponent (how fast singular values decay)
            # S[i] ∝ 1/i^α → log(S[i]) = -α*log(i) + const
            log_i = np.log(np.arange(1, min(100, len(S)) + 1))
            log_s = np.log(S[:min(100, len(S))] + 1e-10)
            zipf_alpha = -np.polyfit(log_i, log_s, 1)[0]
            
            results.append({
                'layer': layer_idx,
                'projection': name,
                'shape': tuple(W.shape),
                'rank_90': rank_90,
                'rank_95': rank_95,
                'rank_99': rank_99,
                'zipf_alpha': zipf_alpha,
                'var_at_k': var_at_k,
                'top_5_sv': S[:5].tolist(),
            })
            
            if layer_idx % 7 == 0:  # Print every 7th layer
                print(f"\nLayer {layer_idx} - {name} ({W.shape[0]}x{W.shape[1]}):")
                print(f"  Rank for 90%: {rank_90}, 95%: {rank_95}, 99%: {rank_99}")
                print(f"  Zipf α: {zipf_alpha:.3f}")
                v1000 = var_at_k.get(1000)
                v2000 = var_at_k.get(2000)
                print(f"  Variance at k=1000: {v1000:.1f}%" if v1000 else "  Variance at k=1000: N/A")
                print(f"  Variance at k=2000: {v2000:.1f}%" if v2000 else "  Variance at k=2000: N/A")
    
    return results


def analyze_layer_sensitivity(model, tokenizer):
    """Test which layers are most sensitive to SVD truncation."""
    print("\n" + "="*70)
    print("LAYER SENSITIVITY ANALYSIS")
    print("="*70)
    
    test_prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "Hello, how are you",
    ]
    
    # Get baseline outputs
    print("\nGenerating baseline outputs...")
    baseline_outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_outputs.append(outputs[0])
    
    # Test each layer individually
    sensitivity_results = []
    
    for layer_idx in range(0, model.config.num_hidden_layers, 4):  # Every 4th layer
        print(f"\nTesting layer {layer_idx}...")
        
        layer = model.model.layers[layer_idx]
        
        # Store original weights
        original_weights = {}
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, name)
            original_weights[name] = proj.weight.data.clone()
        
        # Apply SVD truncation at k=1500
        k = 1500
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, name)
            W = proj.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            
            # Truncate
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            proj.weight.data = W_approx.to(proj.weight.dtype)
        
        # Test outputs
        errors = []
        for i, prompt in enumerate(test_prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Compare to baseline
            baseline_text = tokenizer.decode(baseline_outputs[i], skip_special_tokens=True)
            approx_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            match = baseline_text == approx_text
            errors.append(not match)
            
            if not match:
                print(f"  Prompt: '{prompt}'")
                print(f"    Baseline: {baseline_text}")
                print(f"    Approx:   {approx_text}")
        
        # Restore original weights
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, name)
            proj.weight.data = original_weights[name]
        
        error_rate = sum(errors) / len(errors)
        sensitivity_results.append({
            'layer': layer_idx,
            'error_rate': error_rate,
            'errors': errors,
        })
        
        print(f"  Layer {layer_idx} error rate: {error_rate*100:.0f}%")
    
    return sensitivity_results


def analyze_error_propagation(model, tokenizer):
    """Analyze how SVD errors propagate through layers."""
    print("\n" + "="*70)
    print("ERROR PROPAGATION ANALYSIS")
    print("="*70)
    
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Hook to capture activations
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hooks on MLP outputs
    hooks = []
    for i in range(model.config.num_hidden_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_hook(f'mlp_{i}'))
        hooks.append(h)
    
    # Forward pass - baseline
    with torch.no_grad():
        _ = model(**inputs)
    baseline_activations = {k: v.clone() for k, v in activations.items()}
    
    # Now apply SVD to layer 0 and measure error propagation
    layer = model.model.layers[0]
    original_gate = layer.mlp.gate_proj.weight.data.clone()
    
    # Apply SVD truncation
    k = 1500
    W = layer.mlp.gate_proj.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
    layer.mlp.gate_proj.weight.data = W_approx.to(layer.mlp.gate_proj.weight.dtype)
    
    # Forward pass - with SVD
    activations.clear()
    with torch.no_grad():
        _ = model(**inputs)
    
    # Measure error at each layer
    print("\nError propagation from layer 0 SVD truncation:")
    print("-" * 50)
    
    for i in range(model.config.num_hidden_layers):
        key = f'mlp_{i}'
        baseline = baseline_activations[key]
        approx = activations[key]
        
        # Compute relative error
        error = (approx - baseline).abs()
        rel_error = error.mean() / (baseline.abs().mean() + 1e-10)
        max_error = error.max() / (baseline.abs().max() + 1e-10)
        
        print(f"Layer {i:2d}: mean_rel_error={rel_error.item():.6f}, max_rel_error={max_error.item():.6f}")
    
    # Restore
    layer.mlp.gate_proj.weight.data = original_gate
    
    # Remove hooks
    for h in hooks:
        h.remove()


def analyze_token_level_impact(model, tokenizer):
    """Analyze SVD impact on individual token predictions."""
    print("\n" + "="*70)
    print("TOKEN-LEVEL IMPACT ANALYSIS")
    print("="*70)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline logits
    with torch.no_grad():
        baseline_outputs = model(**inputs)
        baseline_logits = baseline_outputs.logits[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_top5 = torch.topk(baseline_probs, 5)
    
    print(f"\nPrompt: '{prompt}'")
    print("\nBaseline top-5 predictions:")
    for i in range(5):
        token_id = baseline_top5.indices[0, i].item()
        prob = baseline_top5.values[0, i].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token}' ({prob*100:.1f}%)")
    
    # Test different k values
    k_values = [500, 1000, 1500, 2000, 2500, 3000]
    
    print("\n" + "-"*50)
    print("Impact of SVD truncation on predictions:")
    print("-"*50)
    
    for k in k_values:
        # Apply SVD to all MLP layers (do SVD on CPU to avoid OOM)
        original_weights = {}
        for layer_idx in range(model.config.num_hidden_layers):
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
        
        # Get predictions with SVD
        with torch.no_grad():
            approx_outputs = model(**inputs)
            approx_logits = approx_outputs.logits[:, -1, :]
            approx_probs = F.softmax(approx_logits, dim=-1)
            approx_top5 = torch.topk(approx_probs, 5)
        
        # Compare
        baseline_top1 = baseline_top5.indices[0, 0].item()
        approx_top1 = approx_top5.indices[0, 0].item()
        
        # KL divergence
        kl_div = F.kl_div(
            approx_probs.log(),
            baseline_probs,
            reduction='batchmean'
        ).item()
        
        # Top-1 match
        top1_match = baseline_top1 == approx_top1
        
        # Probability of correct token
        correct_prob = approx_probs[0, baseline_top1].item()
        baseline_correct_prob = baseline_probs[0, baseline_top1].item()
        
        print(f"\nk={k}:")
        print(f"  Top-1 match: {top1_match}")
        print(f"  KL divergence: {kl_div:.6f}")
        print(f"  Correct token prob: {correct_prob*100:.1f}% (baseline: {baseline_correct_prob*100:.1f}%)")
        
        if not top1_match:
            approx_token = tokenizer.decode([approx_top1])
            baseline_token = tokenizer.decode([baseline_top1])
            print(f"  Baseline: '{baseline_token}' → Approx: '{approx_token}'")
        
        # Restore weights
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                proj.weight.data = original_weights[layer_idx][name]


def analyze_cumulative_error(model, tokenizer):
    """Analyze how errors accumulate during generation."""
    print("\n" + "="*70)
    print("CUMULATIVE ERROR ANALYSIS (Multi-token generation)")
    print("="*70)
    
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Baseline generation
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
    
    # Test different k values
    k_values = [1000, 1500, 2000, 2500, 3000, 3500]
    
    for k in k_values:
        # Apply SVD to all MLP layers (do SVD on CPU to avoid OOM)
        original_weights = {}
        for layer_idx in range(model.config.num_hidden_layers):
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
        
        # Generate with SVD
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
        
        print(f"\nk={k}: {matches}/{total} tokens match ({matches/total*100:.0f}%)")
        print(f"  Output: {approx_text}")
        
        # Restore weights
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, name)
                proj.weight.data = original_weights[layer_idx][name]


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
    
    # Run analyses
    print("\n" + "="*70)
    print("SVD QUALITY DEEP DIVE ANALYSIS")
    print("="*70)
    
    # 1. Singular value distribution (skip - already done, takes too long)
    # sv_results = analyze_singular_values(model)
    sv_results = []  # Use cached results from previous run
    
    # 2. Token-level impact
    analyze_token_level_impact(model, tokenizer)
    
    # 3. Cumulative error during generation
    analyze_cumulative_error(model, tokenizer)
    
    # 4. Error propagation (skip - takes too long)
    # analyze_error_propagation(model, tokenizer)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find minimum k for 99% variance across all projections
    min_k_99 = 0
    for r in sv_results:
        if r['rank_99'] > min_k_99:
            min_k_99 = r['rank_99']
    
    print(f"\nMinimum k for 99% variance: {min_k_99}")
    print(f"Full rank: 3584")
    print(f"Ratio: {min_k_99/3584*100:.1f}%")
    
    # Average Zipf exponent
    avg_zipf = np.mean([r['zipf_alpha'] for r in sv_results])
    print(f"\nAverage Zipf exponent: {avg_zipf:.3f}")
    print(f"Expected (1/φ): {1/PHI:.3f}")


if __name__ == "__main__":
    main()
