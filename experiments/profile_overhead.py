#!/usr/bin/env python3
"""
Profile the fixed overhead in our generation pipeline.
"""

import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"

def profile_overhead():
    print("="*70)
    print("PROFILING FIXED OVERHEAD")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    
    # Test prompts
    short_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    long_prompt = "<|im_start|>system\nYou are a helpful AI assistant. Be concise and direct.<|im_end|>\n<|im_start|>user\nExplain quantum computing in simple terms.<|im_end|>\n<|im_start|>assistant\n"
    
    # Warmup
    inputs = tokenizer(short_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()
    
    print("\n" + "="*70)
    print("OVERHEAD BREAKDOWN")
    print("="*70)
    
    # 1. Tokenization
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = tokenizer(long_prompt, return_tensors="pt")
        times.append(time.perf_counter() - start)
    tokenize_time = np.mean(times) * 1000
    print(f"\n1. Tokenization: {tokenize_time:.3f}ms")
    
    # 2. Move to GPU
    inputs_cpu = tokenizer(long_prompt, return_tensors="pt")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = {k: v.to(DEVICE) for k, v in inputs_cpu.items()}
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    move_time = np.mean(times) * 1000
    print(f"2. Move to GPU: {move_time:.3f}ms")
    
    # 3. First forward pass (prompt processing)
    inputs = tokenizer(long_prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs['input_ids'].shape[1]
    
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inputs['input_ids'], use_cache=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    first_forward_time = np.mean(times) * 1000
    print(f"3. First forward (prompt={prompt_len} tokens): {first_forward_time:.3f}ms")
    
    # 4. Single token generation (with KV cache)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(next_token, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    single_token_time = np.mean(times) * 1000
    print(f"4. Single token generation: {single_token_time:.3f}ms ({1000/single_token_time:.1f} tok/s theoretical)")
    
    # 5. Decoding
    output_ids = torch.randint(0, 1000, (1, 100)).to(DEVICE)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        times.append(time.perf_counter() - start)
    decode_time = np.mean(times) * 1000
    print(f"5. Decoding (100 tokens): {decode_time:.3f}ms")
    
    # 6. Full generate() call overhead
    print(f"\n" + "="*70)
    print("GENERATE() CALL ANALYSIS")
    print("="*70)
    
    for max_tokens in [1, 5, 10, 20]:
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        total_time = np.mean(times) * 1000
        generated = outputs.shape[1] - prompt_len
        per_token = total_time / generated if generated > 0 else 0
        overhead = total_time - (generated * single_token_time)
        
        print(f"  {max_tokens:2d} tokens: {total_time:.1f}ms total, {per_token:.1f}ms/tok, ~{overhead:.1f}ms overhead")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_overhead = tokenize_time + move_time + first_forward_time + decode_time
    print(f"""
Fixed overhead breakdown:
  Tokenization:     {tokenize_time:.1f}ms
  Move to GPU:      {move_time:.1f}ms  
  First forward:    {first_forward_time:.1f}ms  ← BIGGEST
  Decoding:         {decode_time:.1f}ms
  ─────────────────────────
  Total:            {total_overhead:.1f}ms

Per-token generation: {single_token_time:.1f}ms ({1000/single_token_time:.1f} tok/s)

OPTIMIZATION OPPORTUNITIES:

1. PROMPT CACHING
   - Cache the KV states for common system prompts
   - Saves {first_forward_time:.1f}ms per request with same prefix

2. CONTINUOUS BATCHING
   - Process multiple requests together
   - Amortize first forward pass across requests

3. SPECULATIVE DECODING
   - Use small model to draft, large model to verify
   - Can 2-3x throughput

4. CUDA GRAPHS
   - Capture and replay CUDA operations
   - Reduces Python/CUDA overhead

5. TENSOR PARALLELISM
   - Split model across GPUs
   - Reduces per-token latency
""")


if __name__ == "__main__":
    profile_overhead()
