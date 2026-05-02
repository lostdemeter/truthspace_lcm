#!/usr/bin/env python3
"""
Profile φ-Lattice Model vs Original Qwen2-7B
=============================================

Compares:
1. Memory usage
2. Generation speed (tokens/second)
3. Output quality
4. First token latency
"""

import torch
import torch.nn.functional as F
import time
import gc
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi_lattice(weight):
    """Encode weight to φ-lattice (K=128 scaling)."""
    signs = torch.sign(weight)
    signs[signs == 0] = 1
    magnitudes = weight.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    levels = levels.clamp(min=-16384, max=16383)
    return levels.to(torch.int16), signs.to(torch.int8)


def decode_phi_lattice(levels, signs):
    """Decode φ-lattice to weights."""
    exponents = levels.float() / K_SCALE
    magnitudes = torch.exp(exponents * LOG_PHI)
    return signs.float() * magnitudes


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def profile_model(model, tokenizer, name, prompts, n_tokens=100):
    """Profile a model's performance."""
    print(f"\n{'='*60}")
    print(f"PROFILING: {name}")
    print(f"{'='*60}")
    
    model.eval()
    
    # Memory
    memory_gb = get_gpu_memory()
    print(f"\nGPU Memory: {memory_gb:.2f} GB")
    
    results = {
        'name': name,
        'memory_gb': memory_gb,
        'first_token_ms': [],
        'tokens_per_sec': [],
        'responses': []
    }
    
    for prompt in prompts:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_len = inputs['input_ids'].shape[1]
        
        # Warmup
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        
        # First token latency
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        torch.cuda.synchronize()
        first_token_ms = (time.perf_counter() - start) * 1000
        results['first_token_ms'].append(first_token_ms)
        
        # Full generation speed
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - start
        
        n_generated = generated.shape[1] - input_len
        tokens_per_sec = n_generated / gen_time
        results['tokens_per_sec'].append(tokens_per_sec)
        
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        results['responses'].append(response[:200])
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  First token: {first_token_ms:.1f}ms")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s ({n_generated} tokens in {gen_time:.2f}s)")
        print(f"  Response: {response[:80]}...")
    
    # Averages
    avg_first = sum(results['first_token_ms']) / len(results['first_token_ms'])
    avg_speed = sum(results['tokens_per_sec']) / len(results['tokens_per_sec'])
    
    print(f"\n--- SUMMARY ---")
    print(f"Avg first token: {avg_first:.1f}ms")
    print(f"Avg speed: {avg_speed:.1f} tok/s")
    
    results['avg_first_token_ms'] = avg_first
    results['avg_tokens_per_sec'] = avg_speed
    
    return results


def convert_to_phi_lattice(model):
    """Convert all attention Q projections to φ-lattice."""
    print("\nConverting Q projections to φ-lattice...")
    
    correlations = []
    for i, layer in enumerate(model.model.layers):
        q_proj = layer.self_attn.q_proj
        original_weight = q_proj.weight.data.float()
        
        # Encode and decode
        levels, signs = encode_phi_lattice(original_weight)
        reconstructed = decode_phi_lattice(levels, signs)
        
        # Compute correlation
        corr = torch.corrcoef(torch.stack([
            original_weight.flatten(),
            reconstructed.flatten()
        ]))[0, 1].item()
        correlations.append(corr)
        
        # Replace weights
        q_proj.weight.data.copy_(reconstructed.to(q_proj.weight.dtype))
        
        if i % 7 == 0:
            print(f"  Layer {i}: {corr*100:.4f}% correlation")
    
    print(f"  Mean correlation: {sum(correlations)/len(correlations)*100:.4f}%")
    return correlations


def main():
    print("="*70)
    print("φ-LATTICE MODEL vs ORIGINAL QWEN2-7B PROFILING")
    print("="*70)
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What is 15 * 17?",
        "Who wrote Romeo and Juliet?",
    ]
    
    # Load original model
    print("\nLoading Qwen2-7B (original)...")
    torch.cuda.empty_cache()
    gc.collect()
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    # Profile original
    original_results = profile_model(model, tokenizer, "Original Qwen2-7B", prompts)
    
    # Convert to φ-lattice
    convert_to_phi_lattice(model)
    
    # Profile φ-lattice
    phi_results = profile_model(model, tokenizer, "φ-Lattice Qwen2-7B", prompts)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Original':>15} {'φ-Lattice':>15} {'Diff':>15}")
    print("-"*70)
    
    print(f"{'GPU Memory (GB)':<25} {original_results['memory_gb']:>15.2f} {phi_results['memory_gb']:>15.2f} {phi_results['memory_gb']-original_results['memory_gb']:>+15.2f}")
    print(f"{'First Token (ms)':<25} {original_results['avg_first_token_ms']:>15.1f} {phi_results['avg_first_token_ms']:>15.1f} {phi_results['avg_first_token_ms']-original_results['avg_first_token_ms']:>+15.1f}")
    print(f"{'Speed (tok/s)':<25} {original_results['avg_tokens_per_sec']:>15.1f} {phi_results['avg_tokens_per_sec']:>15.1f} {phi_results['avg_tokens_per_sec']-original_results['avg_tokens_per_sec']:>+15.1f}")
    
    # Response comparison
    print("\n" + "="*70)
    print("RESPONSE COMPARISON")
    print("="*70)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt}")
        print(f"  Original:  {original_results['responses'][i][:100]}...")
        print(f"  φ-Lattice: {phi_results['responses'][i][:100]}...")
        
        if original_results['responses'][i] == phi_results['responses'][i]:
            print("  ✓ IDENTICAL")
        else:
            print("  ⚠ DIFFERENT")


if __name__ == "__main__":
    main()
