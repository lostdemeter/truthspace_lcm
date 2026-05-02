#!/usr/bin/env python3
"""
Profile Qwen2 generation pipeline to identify bottlenecks.
"""

import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"

def profile_generation():
    print("="*70)
    print("PROFILING QWEN2 GENERATION PIPELINE")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",  # Flash attention
    )
    model.eval()
    
    print(f"Model loaded. GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    # Test prompt
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in simple terms.<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs['input_ids'].shape[1]
    print(f"Prompt length: {prompt_len} tokens")
    
    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    # Profile different generation lengths
    print("\n" + "="*70)
    print("GENERATION SPEED BY LENGTH")
    print("="*70)
    
    for max_tokens in [10, 50, 100, 200]:
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
        elapsed = time.perf_counter() - start
        
        generated = outputs.shape[1] - prompt_len
        tok_per_sec = generated / elapsed
        
        print(f"  {max_tokens:3d} tokens: {elapsed*1000:.1f}ms = {tok_per_sec:.1f} tok/s")
    
    # Profile with CUDA events for more precision
    print("\n" + "="*70)
    print("DETAILED PROFILING (100 tokens)")
    print("="*70)
    
    # Use torch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    
    # Print top operations
    print("\nTop CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # Profile individual components
    print("\n" + "="*70)
    print("COMPONENT BREAKDOWN")
    print("="*70)
    
    # Test tokenization
    start = time.perf_counter()
    for _ in range(100):
        _ = tokenizer(prompt, return_tensors="pt")
    tokenize_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  Tokenization: {tokenize_time:.2f}ms per call")
    
    # Test single forward pass
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            _ = model(inputs['input_ids'])
    torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / 10 * 1000
    print(f"  Single forward pass: {forward_time:.2f}ms")
    
    # Test decode
    output_ids = outputs[0]
    start = time.perf_counter()
    for _ in range(100):
        _ = tokenizer.decode(output_ids, skip_special_tokens=True)
    decode_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  Decoding: {decode_time:.2f}ms per call")
    
    # Check if we're memory bound
    print("\n" + "="*70)
    print("MEMORY ANALYSIS")
    print("="*70)
    print(f"  GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"  GPU Memory max: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Test with different batch sizes (if memory allows)
    print("\n" + "="*70)
    print("BATCH SIZE IMPACT")
    print("="*70)
    
    for batch_size in [1, 2]:
        try:
            batch_inputs = {
                'input_ids': inputs['input_ids'].repeat(batch_size, 1),
                'attention_mask': inputs['attention_mask'].repeat(batch_size, 1),
            }
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            total_tokens = (outputs.shape[1] - prompt_len) * batch_size
            tok_per_sec = total_tokens / elapsed
            
            print(f"  Batch {batch_size}: {elapsed*1000:.1f}ms = {tok_per_sec:.1f} tok/s total")
        except RuntimeError as e:
            print(f"  Batch {batch_size}: OOM")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Typical bottlenecks for LLM generation:

1. MEMORY BANDWIDTH - Moving weights from VRAM to compute units
   - 7B model = 14GB weights (bfloat16)
   - RTX 3090 Ti: 1008 GB/s bandwidth
   - Theoretical max: 1008 / 14 = 72 forward passes/sec
   - But we need to load weights for EACH token!

2. KV CACHE - Grows with sequence length
   - Each layer stores K, V for all positions
   - 28 layers × 4 KV heads × 128 dim × seq_len × 2 bytes

3. ATTENTION COMPUTATION - O(n²) with sequence length
   - SDPA/Flash attention helps but still scales

4. PYTHON OVERHEAD - GIL, function calls, etc.

For 20 tok/s on 7B model, this is actually reasonable!
- vLLM achieves ~30-40 tok/s on similar hardware
- Our overhead is likely in the Python/FastAPI layer
""")


if __name__ == "__main__":
    profile_generation()
