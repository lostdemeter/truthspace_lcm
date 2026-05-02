#!/usr/bin/env python3
"""
φ-Encoded Qwen2-7B: Speed Optimization Suite
=============================================

Tests every optimization to maximize tokens/second:
1. Baseline (float16, HF generate)
2. torch.compile() 
3. Flash Attention / SDPA
4. INT8 quantization (bitsandbytes)
5. Combined best

Usage:
    python phi_speed_optimization.py
"""

import numpy as np
import torch
import torch.nn as nn
import os
import sys
import gc
import time
import argparse

PHI = (1 + np.sqrt(5)) / 2
GRID = 128
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')

BENCHMARK_PROMPTS = [
    "The meaning of life is",
    "In a galaxy far far away,",
    "The quick brown fox",
    "Once upon a time there was a little",
    "Artificial intelligence will",
]
GEN_TOKENS = 100


def decode_phi_to_tensor(path):
    d = np.load(path)
    signs = d['signs'].astype(np.float32)
    exponents = d['exponents'].astype(np.float32)
    return torch.from_numpy(
        signs * (np.float32(PHI) ** (exponents / np.float32(GRID)))
    ).half()


def build_state_dict():
    state_dict = {}
    t_start = time.time()

    print("    embed_tokens...", end='', flush=True)
    state_dict['model.embed_tokens.weight'] = decode_phi_to_tensor(
        os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(" lm_head...", end='', flush=True)
    state_dict['lm_head.weight'] = decode_phi_to_tensor(
        os.path.join(MODEL_DIR, 'lm_head.npz'))
    fn = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))
    state_dict['model.norm.weight'] = torch.from_numpy(
        fn['weight'].astype(np.float32)).half()

    for layer_idx in range(28):
        if layer_idx % 7 == 0:
            print(f" L{layer_idx}", end='', flush=True)
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        prefix = f'model.layers.{layer_idx}'

        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        state_dict[f'{prefix}.input_layernorm.weight'] = torch.from_numpy(
            norms['input_layernorm'].astype(np.float32)).half()
        state_dict[f'{prefix}.post_attention_layernorm.weight'] = torch.from_numpy(
            norms['post_attention_layernorm'].astype(np.float32)).half()

        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        state_dict[f'{prefix}.self_attn.q_proj.bias'] = torch.from_numpy(
            biases['q_proj_bias'].astype(np.float32)).half()
        state_dict[f'{prefix}.self_attn.k_proj.bias'] = torch.from_numpy(
            biases['k_proj_bias'].astype(np.float32)).half()
        state_dict[f'{prefix}.self_attn.v_proj.bias'] = torch.from_numpy(
            biases['v_proj_bias'].astype(np.float32)).half()

        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            state_dict[f'{prefix}.self_attn.{proj}.weight'] = decode_phi_to_tensor(
                os.path.join(layer_dir, f'{proj}.npz'))

        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            state_dict[f'{prefix}.mlp.{proj}.weight'] = decode_phi_to_tensor(
                os.path.join(layer_dir, f'{proj}.npz'))

        gc.collect()

    print(f" ({time.time()-t_start:.0f}s)")
    return state_dict


def load_model_to_gpu(state_dict):
    from transformers import AutoConfig, Qwen2ForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
    config.torch_dtype = torch.float16

    for key in state_dict:
        state_dict[key] = state_dict[key].to(device='cuda', dtype=torch.float16)
    gc.collect()

    with torch.device('meta'):
        model = Qwen2ForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=False)

    # Fix RoPE buffers
    for name, module in model.named_modules():
        for bname, buf in list(module.named_buffers(recurse=False)):
            if buf.device == torch.device('meta'):
                if 'inv_freq' in bname:
                    head_dim = config.hidden_size // config.num_attention_heads
                    inv_freq = 1.0 / (config.rope_theta ** (
                        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    module.register_buffer(bname, inv_freq.to('cuda'))
                else:
                    module.register_buffer(bname,
                        torch.zeros_like(buf, device='cuda', dtype=torch.float16))

    model.eval()
    return model


def benchmark_generate(model, tokenizer, label, gen_tokens=GEN_TOKENS, n_warmup=2):
    """Benchmark generation speed, return avg tok/s and sample output."""
    device = next(model.parameters()).device

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            inp = tokenizer("Hello", return_tensors="pt").to(device)
            _ = model.generate(**inp, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    total_tokens = 0
    total_time = 0.0
    sample_output = ""

    for prompt in BENCHMARK_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=gen_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        output_len = outputs.shape[1] - input_len
        total_tokens += output_len
        total_time += elapsed

        if not sample_output:
            sample_output = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True)[:80]

    avg_tps = total_tokens / total_time
    vram = torch.cuda.memory_allocated() / 1024**3

    print(f"  {label:30s}  {avg_tps:6.1f} tok/s  {vram:5.1f} GB  │ {sample_output[:60]}...")
    return avg_tps, vram


def main():
    print("=" * 80)
    print("  φ-Encoded Qwen2-7B: Speed Optimization Suite")
    print("=" * 80)
    print()

    # GPU info
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
    print()

    # ─── Load model ───────────────────────────────────────────────
    print("  Loading φ-encoded weights...", end='', flush=True)
    state_dict = build_state_dict()

    print("  Moving to GPU + creating model...")
    model = load_model_to_gpu(state_dict)
    del state_dict; gc.collect(); torch.cuda.empty_cache()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ─── Test 1: Baseline ─────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  TEST 1: Baseline (float16, HF generate)")
    print("─" * 80)
    tps, vram = benchmark_generate(model, tokenizer, "Baseline FP16")
    results['Baseline FP16'] = (tps, vram)

    # ─── Test 2: Check attention backend ──────────────────────────
    print("\n" + "─" * 80)
    print("  TEST 2: Attention Backend Check")
    print("─" * 80)
    attn_impl = getattr(model.config, '_attn_implementation', 'unknown')
    print(f"  Current attention: {attn_impl}")

    # Force SDPA (Flash Attention via PyTorch)
    if attn_impl != 'sdpa':
        print("  Switching to SDPA...")
        model.config._attn_implementation = 'sdpa'
        tps, vram = benchmark_generate(model, tokenizer, "SDPA Attention")
        results['SDPA Attention'] = (tps, vram)
    else:
        print("  Already using SDPA (optimal)")
        results['SDPA Attention'] = results['Baseline FP16']

    # Try flash_attention_2 if available
    try:
        model.config._attn_implementation = 'flash_attention_2'
        tps, vram = benchmark_generate(model, tokenizer, "Flash Attention 2")
        results['Flash Attention 2'] = (tps, vram)
    except Exception as e:
        print(f"  Flash Attention 2 failed: {e}")
        model.config._attn_implementation = 'sdpa'

    # ─── Test 3: torch.compile() ──────────────────────────────────
    print("\n" + "─" * 80)
    print("  TEST 3: torch.compile()")
    print("─" * 80)
    try:
        print("  Compiling model (this takes a minute)...")
        model_compiled = torch.compile(model, mode="reduce-overhead")
        tps, vram = benchmark_generate(model_compiled, tokenizer, "torch.compile(reduce-overhead)")
        results['torch.compile'] = (tps, vram)
        del model_compiled
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    gc.collect(); torch.cuda.empty_cache()

    # ─── Test 4: INT8 quantization ────────────────────────────────
    print("\n" + "─" * 80)
    print("  TEST 4: INT8 Quantization (bitsandbytes)")
    print("─" * 80)
    try:
        import bitsandbytes as bnb

        print("  Quantizing linear layers to INT8...")
        t0 = time.time()
        n_quantized = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 512:
                # Replace with INT8 linear
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model

                int8_linear = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                int8_linear.weight = bnb.nn.Int8Params(
                    module.weight.data,
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                if module.bias is not None:
                    int8_linear.bias = module.bias

                setattr(parent, child_name, int8_linear)
                n_quantized += 1

        gc.collect(); torch.cuda.empty_cache()
        print(f"  Quantized {n_quantized} layers in {time.time()-t0:.1f}s")

        vram_after = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM after INT8: {vram_after:.1f} GB")

        tps, vram = benchmark_generate(model, tokenizer, "INT8 (bitsandbytes)")
        results['INT8 bitsandbytes'] = (tps, vram)

    except Exception as e:
        print(f"  INT8 quantization failed: {e}")
        import traceback; traceback.print_exc()

    # ─── Test 5: INT8 + torch.compile ─────────────────────────────
    if 'INT8 bitsandbytes' in results:
        print("\n" + "─" * 80)
        print("  TEST 5: INT8 + torch.compile()")
        print("─" * 80)
        try:
            model_compiled = torch.compile(model, mode="reduce-overhead")
            tps, vram = benchmark_generate(model_compiled, tokenizer,
                                           "INT8 + torch.compile")
            results['INT8 + compile'] = (tps, vram)
            del model_compiled
        except Exception as e:
            print(f"  Combined failed: {e}")

    # ─── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n  {'Configuration':30s}  {'tok/s':>8s}  {'VRAM':>7s}  {'vs Base':>8s}")
    print("  " + "─" * 60)

    base_tps = results.get('Baseline FP16', (1, 0))[0]
    for name, (tps, vram) in sorted(results.items(), key=lambda x: -x[1][0]):
        speedup = tps / base_tps
        marker = " ★" if tps == max(v[0] for v in results.values()) else ""
        print(f"  {name:30s}  {tps:7.1f}  {vram:5.1f} GB  {speedup:7.2f}×{marker}")

    best_name = max(results, key=lambda k: results[k][0])
    best_tps = results[best_name][0]
    print(f"\n  ★ Best: {best_name} at {best_tps:.1f} tok/s "
          f"({best_tps/base_tps:.2f}× baseline)")
    print()


if __name__ == '__main__':
    main()
