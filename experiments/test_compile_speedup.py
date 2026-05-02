#!/usr/bin/env python3
"""
Test torch.compile() speedup for Qwen2 generation.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"

def test_compile():
    print("="*70)
    print("TESTING torch.compile() SPEEDUP")
    print("="*70)
    
    model_name = "Qwen/Qwen2-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test 1: Without compile
    print("\n1. Loading model WITHOUT torch.compile()...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    
    prompt = "<|im_start|>user\nExplain quantum computing briefly.<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs['input_ids'].shape[1]
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    # Benchmark without compile
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    no_compile_time = sum(times) / len(times)
    generated = outputs.shape[1] - prompt_len
    no_compile_tps = generated / no_compile_time
    print(f"   Without compile: {no_compile_time*1000:.1f}ms = {no_compile_tps:.1f} tok/s")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Test 2: With torch.compile()
    print("\n2. Loading model WITH torch.compile()...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    
    # Compile the model
    print("   Compiling model (this may take a minute)...")
    model = torch.compile(model, mode="reduce-overhead")
    
    # Warmup (compilation happens here)
    print("   Warmup (triggering compilation)...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    # Benchmark with compile
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    compile_time = sum(times) / len(times)
    generated = outputs.shape[1] - prompt_len
    compile_tps = generated / compile_time
    print(f"   With compile: {compile_time*1000:.1f}ms = {compile_tps:.1f} tok/s")
    
    # Summary
    speedup = no_compile_time / compile_time
    print(f"\n" + "="*70)
    print(f"RESULT: torch.compile() gives {speedup:.2f}x speedup")
    print(f"  Without: {no_compile_tps:.1f} tok/s")
    print(f"  With:    {compile_tps:.1f} tok/s")
    print("="*70)


if __name__ == "__main__":
    test_compile()
