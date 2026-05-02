#!/usr/bin/env python3
"""
φ-Encoded Qwen2-7B: Custom GPU Inference Engine
=================================================

Bypasses HF's generate() entirely. Custom forward pass with KV cache
on GPU, based on our TorchNavigator and PhiQwen2Engine patterns.

Optimizations tested:
  1. Custom generate loop (no HF overhead)
  2. Manual KV cache (no HF cache management)
  3. torch.compile() on custom forward (no HF graph breaks)
  4. Linearized MLP: SiLU(gate)×up ≈ (gate×up)/2 (DC 152)
  5. Combined best

Usage:
    python phi_custom_gpu.py              # run all benchmarks
    python phi_custom_gpu.py --chat       # interactive chat mode
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import gc
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

PHI = (1 + np.sqrt(5)) / 2
GRID = 128
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')

# Qwen2-7B architecture
HIDDEN_DIM = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 18944
NUM_LAYERS = 28
ROPE_THETA = 1_000_000.0
VOCAB_SIZE = 152064


@dataclass
class LayerWeights:
    """All weights for one transformer layer, on GPU."""
    # Attention
    W_q: torch.Tensor      # (3584, 3584)
    W_k: torch.Tensor      # (512, 3584)
    W_v: torch.Tensor      # (512, 3584)
    W_o: torch.Tensor      # (3584, 3584)
    b_q: torch.Tensor      # (3584,)
    b_k: torch.Tensor      # (512,)
    b_v: torch.Tensor      # (512,)
    # MLP
    W_gate: torch.Tensor   # (18944, 3584)
    W_up: torch.Tensor     # (18944, 3584)
    W_down: torch.Tensor   # (3584, 18944)
    # Norms
    ln1: torch.Tensor      # (3584,)
    ln2: torch.Tensor      # (3584,)


class KVCache:
    """Manual KV cache for efficient generation."""

    def __init__(self, n_layers: int, max_seq: int = 4096, device='cuda'):
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.device = device
        self.seq_len = 0
        self._write_end = 0  # tracks where the latest append wrote to
        # Pre-allocate buffers
        self.K = torch.zeros(n_layers, max_seq, NUM_KV_HEADS, HEAD_DIM,
                             dtype=torch.float16, device=device)
        self.V = torch.zeros(n_layers, max_seq, NUM_KV_HEADS, HEAD_DIM,
                             dtype=torch.float16, device=device)

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append new K,V to cache. k,v: (new_seq, kv_heads, head_dim)"""
        new_len = k.shape[0]
        end = self.seq_len + new_len
        self.K[layer_idx, self.seq_len:end] = k
        self.V[layer_idx, self.seq_len:end] = v
        self._write_end = end

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K,V including just-appended data."""
        return (self.K[layer_idx, :self._write_end],
                self.V[layer_idx, :self._write_end])

    def advance(self, n: int):
        """Advance sequence position after all layers processed."""
        self.seq_len += n
        self._write_end = self.seq_len

    def reset(self):
        self.seq_len = 0
        self._write_end = 0


class PhiGPUEngine:
    """
    Custom GPU inference engine for φ-encoded Qwen2-7B.
    No HuggingFace model. No generate(). Pure PyTorch.
    """

    def __init__(self, device='cuda', use_linear_mlp=False):
        self.device = device
        self.use_linear_mlp = use_linear_mlp

        self.embeddings: Optional[torch.Tensor] = None
        self.lm_head: Optional[torch.Tensor] = None
        self.final_norm: Optional[torch.Tensor] = None
        self.layers: List[LayerWeights] = []
        self.cos_cache: Optional[torch.Tensor] = None
        self.sin_cache: Optional[torch.Tensor] = None

    def load_weights(self, model_dir: str = MODEL_DIR, verbose=True):
        """Load φ-encoded weights, decode to float16, move to GPU."""
        t_start = time.time()

        def decode_phi(path):
            d = np.load(path)
            s = d['signs'].astype(np.float32)
            e = d['exponents'].astype(np.float32)
            return torch.from_numpy(s * (np.float32(PHI) ** (e / np.float32(GRID)))
                                    ).half().to(self.device)

        if verbose:
            print("  Loading φ-encoded weights to GPU...")

        self.embeddings = decode_phi(os.path.join(model_dir, 'embed_tokens.npz'))
        if verbose: print(f"    embed_tokens: {self.embeddings.shape}")

        self.lm_head = decode_phi(os.path.join(model_dir, 'lm_head.npz'))
        if verbose: print(f"    lm_head: {self.lm_head.shape}")

        fn = np.load(os.path.join(model_dir, 'final_norm.npz'))
        self.final_norm = torch.from_numpy(fn['weight'].astype(np.float32)
                                           ).half().to(self.device)

        for i in range(NUM_LAYERS):
            layer_dir = os.path.join(model_dir, f'layer_{i:02d}')
            if verbose and i % 7 == 0:
                print(f"    layer {i}...", end='', flush=True)

            norms = np.load(os.path.join(layer_dir, 'norms.npz'))
            biases = np.load(os.path.join(layer_dir, 'biases.npz'))

            layer = LayerWeights(
                W_q=decode_phi(os.path.join(layer_dir, 'q_proj.npz')),
                W_k=decode_phi(os.path.join(layer_dir, 'k_proj.npz')),
                W_v=decode_phi(os.path.join(layer_dir, 'v_proj.npz')),
                W_o=decode_phi(os.path.join(layer_dir, 'o_proj.npz')),
                b_q=torch.from_numpy(biases['q_proj_bias'].astype(np.float32)).half().to(self.device),
                b_k=torch.from_numpy(biases['k_proj_bias'].astype(np.float32)).half().to(self.device),
                b_v=torch.from_numpy(biases['v_proj_bias'].astype(np.float32)).half().to(self.device),
                W_gate=decode_phi(os.path.join(layer_dir, 'gate_proj.npz')),
                W_up=decode_phi(os.path.join(layer_dir, 'up_proj.npz')),
                W_down=decode_phi(os.path.join(layer_dir, 'down_proj.npz')),
                ln1=torch.from_numpy(norms['input_layernorm'].astype(np.float32)).half().to(self.device),
                ln2=torch.from_numpy(norms['post_attention_layernorm'].astype(np.float32)).half().to(self.device),
            )
            self.layers.append(layer)
            gc.collect()

        if verbose:
            elapsed = time.time() - t_start
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"\n    Done: {elapsed:.0f}s, {vram:.1f} GB VRAM")

        # Pre-compute RoPE
        self._init_rope(4096)

    def _init_rope(self, max_seq: int):
        """Pre-compute RoPE cos/sin tables."""
        inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=self.device) / HEAD_DIM))
        positions = torch.arange(max_seq, dtype=torch.float32, device=self.device)
        freqs = torch.outer(positions, inv_freq)  # (max_seq, head_dim/2)
        self.cos_cache = torch.cos(freqs).half()
        self.sin_cache = torch.sin(freqs).half()

    @staticmethod
    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps=1e-6) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        return ((x.float() / rms) * weight.float()).half()

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to (seq, heads, head_dim) tensor."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # cos, sin: (seq, head_dim/2) → broadcast over heads
        c = cos.unsqueeze(1)  # (seq, 1, head_dim/2)
        s = sin.unsqueeze(1)
        out = torch.empty_like(x)
        out[..., ::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out

    def forward_layer(self, hidden: torch.Tensor, layer: LayerWeights,
                      layer_idx: int, pos_start: int,
                      kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        """
        Forward one layer. hidden: (seq_len, hidden_dim)
        """
        seq_len = hidden.shape[0]

        # ─── Attention ───
        normed = self.rms_norm(hidden, layer.ln1)

        # Q, K, V projections
        Q = F.linear(normed, layer.W_q, layer.b_q)  # (seq, 3584)
        K = F.linear(normed, layer.W_k, layer.b_k)  # (seq, 512)
        V = F.linear(normed, layer.W_v, layer.b_v)  # (seq, 512)

        # Reshape for multi-head
        Q = Q.view(seq_len, NUM_HEADS, HEAD_DIM)
        K = K.view(seq_len, NUM_KV_HEADS, HEAD_DIM)
        V = V.view(seq_len, NUM_KV_HEADS, HEAD_DIM)

        # Apply RoPE
        cos = self.cos_cache[pos_start:pos_start + seq_len]
        sin = self.sin_cache[pos_start:pos_start + seq_len]
        Q = self.apply_rope(Q, cos, sin)
        K = self.apply_rope(K, cos, sin)

        # KV cache
        if kv_cache is not None:
            kv_cache.append(layer_idx, K, V)
            K_full, V_full = kv_cache.get(layer_idx)
        else:
            K_full, V_full = K, V

        total_len = K_full.shape[0]

        # GQA expansion: 4 KV heads → 28 Q heads
        K_exp = K_full.unsqueeze(2).expand(-1, -1, NUM_HEADS // NUM_KV_HEADS, -1)
        K_exp = K_exp.reshape(total_len, NUM_HEADS, HEAD_DIM)
        V_exp = V_full.unsqueeze(2).expand(-1, -1, NUM_HEADS // NUM_KV_HEADS, -1)
        V_exp = V_exp.reshape(total_len, NUM_HEADS, HEAD_DIM)

        # Attention scores: Q (seq, heads, dim) × K (total, heads, dim)
        # → (heads, seq, total)
        scale = HEAD_DIM ** -0.5
        scores = torch.einsum('shd,thd->hst', Q.float(), K_exp.float()) * scale

        # Causal mask
        if seq_len > 1 or kv_cache is None:
            mask = torch.full((seq_len, total_len), float('-inf'),
                              device=self.device, dtype=torch.float32)
            for i in range(seq_len):
                mask[i, :pos_start + i + 1] = 0.0
            scores = scores + mask.unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1).half()

        # Weighted sum
        attn_out = torch.einsum('hst,thd->shd', attn_weights, V_exp)
        attn_out = attn_out.reshape(seq_len, NUM_HEADS * HEAD_DIM)

        # Output projection + residual
        hidden = hidden + F.linear(attn_out, layer.W_o)

        # ─── MLP ───
        normed = self.rms_norm(hidden, layer.ln2)

        gate = F.linear(normed, layer.W_gate)
        up = F.linear(normed, layer.W_up)

        if self.use_linear_mlp:
            # Linearized: SiLU(gate) × up ≈ (gate × up) / 2
            mlp_hidden = (gate * up) * 0.5
        else:
            mlp_hidden = F.silu(gate) * up

        mlp_out = F.linear(mlp_hidden, layer.W_down)
        hidden = hidden + mlp_out

        return hidden

    @torch.inference_mode()
    def forward(self, token_ids: List[int],
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        """
        Forward pass: token_ids → logits for last position.
        Returns: (vocab_size,) logits tensor
        """
        seq_len = len(token_ids)
        pos_start = kv_cache.seq_len if kv_cache is not None else 0

        # Embedding lookup
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        hidden = self.embeddings[ids_tensor]  # (seq_len, hidden_dim)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden = self.forward_layer(hidden, layer, i, pos_start, kv_cache)

        # Advance KV cache position
        if kv_cache is not None:
            kv_cache.advance(seq_len)

        # Final norm + LM head (only last position)
        h_last = self.rms_norm(hidden[-1:], self.final_norm)
        logits = F.linear(h_last, self.lm_head)  # (1, vocab_size)

        return logits[0]  # (vocab_size,)

    @torch.inference_mode()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 100,
                 temperature: float = 0.0, eos_token_id: int = 151643,
                 verbose: bool = False) -> List[int]:
        """
        Autoregressive generation with KV cache.
        No HuggingFace. No generate(). Just forward + argmax.
        """
        kv_cache = KVCache(NUM_LAYERS, max_seq=len(prompt_ids) + max_new_tokens + 16,
                           device=self.device)
        generated = list(prompt_ids)

        # Prefill: process entire prompt
        t0 = time.time()
        logits = self.forward(prompt_ids, kv_cache=kv_cache)
        prefill_time = time.time() - t0

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = logits.argmax().item()

        if verbose:
            print(f"    Prefill: {len(prompt_ids)} tok in {prefill_time:.2f}s "
                  f"({len(prompt_ids)/prefill_time:.0f} tok/s)")

        if next_token == eos_token_id:
            return generated

        generated.append(next_token)

        # Decode: one token at a time
        decode_times = []
        for step in range(1, max_new_tokens):
            t0 = time.time()
            logits = self.forward([next_token], kv_cache=kv_cache)
            dt = time.time() - t0
            decode_times.append(dt)

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = logits.argmax().item()

            if next_token == eos_token_id:
                break
            generated.append(next_token)

        if verbose and decode_times:
            avg_ms = sum(decode_times) / len(decode_times) * 1000
            avg_tps = 1.0 / (sum(decode_times) / len(decode_times))
            print(f"    Decode: {len(decode_times)} tok, {avg_ms:.1f} ms/tok, {avg_tps:.1f} tok/s")

        return generated


def benchmark_engine(engine, tokenizer, label, gen_tokens=100, n_warmup=2):
    """Benchmark generation speed."""
    prompts = [
        "The meaning of life is",
        "In a galaxy far far away,",
        "The quick brown fox",
        "Once upon a time there was a little",
        "Artificial intelligence will",
    ]

    # Warmup
    for _ in range(n_warmup):
        ids = tokenizer.encode("Hello", add_special_tokens=False)
        engine.generate(ids, max_new_tokens=10)
    torch.cuda.synchronize()

    total_tokens = 0
    total_time = 0.0
    sample_out = ""

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        torch.cuda.synchronize()
        t0 = time.time()
        output_ids = engine.generate(ids, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        new_tokens = len(output_ids) - len(ids)
        total_tokens += new_tokens
        total_time += elapsed

        if not sample_out:
            sample_out = tokenizer.decode(output_ids[len(ids):])[:60]

    avg_tps = total_tokens / total_time
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  {label:35s}  {avg_tps:6.1f} tok/s  {vram:5.1f} GB  │ {sample_out}...")
    return avg_tps, vram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat', action='store_true')
    args = parser.parse_args()

    print("=" * 80)
    print("  φ-Encoded Qwen2-7B: Custom GPU Engine")
    print("  No HF generate(). Custom forward + KV cache.")
    print("=" * 80)
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total\n")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

    results = {}

    # ─── Test 1: Custom engine (standard MLP) ─────────────────────
    print("─" * 80)
    print("  TEST 1: Custom GPU engine (standard SiLU MLP)")
    print("─" * 80)
    engine = PhiGPUEngine(use_linear_mlp=False)
    engine.load_weights()
    tps, vram = benchmark_engine(engine, tokenizer, "Custom forward + KV cache")
    results['Custom (SiLU)'] = (tps, vram)

    # ─── Test 2: Custom engine (linearized MLP) ───────────────────
    print("\n" + "─" * 80)
    print("  TEST 2: Custom GPU engine (linearized MLP: gate×up/2)")
    print("─" * 80)
    engine.use_linear_mlp = True
    tps, vram = benchmark_engine(engine, tokenizer, "Custom + linear MLP (DC 152)")
    results['Custom (linear MLP)'] = (tps, vram)
    engine.use_linear_mlp = False  # reset

    # ─── Test 3: torch.compile on custom forward ──────────────────
    print("\n" + "─" * 80)
    print("  TEST 3: torch.compile() on custom forward_layer")
    print("─" * 80)
    try:
        # Compile the forward_layer method
        original_forward_layer = engine.forward_layer
        engine.forward_layer = torch.compile(engine.forward_layer, mode="reduce-overhead")
        print("  Compiled. Running benchmark (first run includes compilation)...")
        # Extra warmup for compilation
        ids = tokenizer.encode("Hello world", add_special_tokens=False)
        for _ in range(3):
            engine.generate(ids, max_new_tokens=20)
        torch.cuda.synchronize()
        tps, vram = benchmark_engine(engine, tokenizer, "Custom + torch.compile")
        results['Custom + compile'] = (tps, vram)
        engine.forward_layer = original_forward_layer
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    # ─── Test 4: Compile + linear MLP ─────────────────────────────
    print("\n" + "─" * 80)
    print("  TEST 4: torch.compile + linearized MLP")
    print("─" * 80)
    try:
        engine.use_linear_mlp = True
        engine.forward_layer = torch.compile(engine.forward_layer, mode="reduce-overhead")
        ids = tokenizer.encode("Hello world", add_special_tokens=False)
        for _ in range(3):
            engine.generate(ids, max_new_tokens=20)
        torch.cuda.synchronize()
        tps, vram = benchmark_engine(engine, tokenizer, "Compile + linear MLP")
        results['Compile + linear MLP'] = (tps, vram)
    except Exception as e:
        print(f"  Combined failed: {e}")

    # ─── HF baseline for comparison ───────────────────────────────
    print("\n" + "─" * 80)
    print("  TEST 5: HF generate() baseline (for comparison)")
    print("─" * 80)
    try:
        # Reuse the already-loaded weights by building HF model
        from transformers import AutoConfig, Qwen2ForCausalLM

        # Build minimal state dict from engine weights
        print("  Building HF model from engine weights...")
        config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
        config.torch_dtype = torch.float16

        state_dict = {}
        state_dict['model.embed_tokens.weight'] = engine.embeddings
        state_dict['lm_head.weight'] = engine.lm_head
        state_dict['model.norm.weight'] = engine.final_norm

        for i, L in enumerate(engine.layers):
            p = f'model.layers.{i}'
            state_dict[f'{p}.input_layernorm.weight'] = L.ln1
            state_dict[f'{p}.post_attention_layernorm.weight'] = L.ln2
            state_dict[f'{p}.self_attn.q_proj.weight'] = L.W_q
            state_dict[f'{p}.self_attn.k_proj.weight'] = L.W_k
            state_dict[f'{p}.self_attn.v_proj.weight'] = L.W_v
            state_dict[f'{p}.self_attn.o_proj.weight'] = L.W_o
            state_dict[f'{p}.self_attn.q_proj.bias'] = L.b_q
            state_dict[f'{p}.self_attn.k_proj.bias'] = L.b_k
            state_dict[f'{p}.self_attn.v_proj.bias'] = L.b_v
            state_dict[f'{p}.mlp.gate_proj.weight'] = L.W_gate
            state_dict[f'{p}.mlp.up_proj.weight'] = L.W_up
            state_dict[f'{p}.mlp.down_proj.weight'] = L.W_down

        with torch.device('meta'):
            hf_model = Qwen2ForCausalLM(config)

        # Fix RoPE
        for name, module in hf_model.named_modules():
            for bname, buf in list(module.named_buffers(recurse=False)):
                if buf.device == torch.device('meta'):
                    if 'inv_freq' in bname:
                        inv_freq = 1.0 / (ROPE_THETA ** (
                            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
                        module.register_buffer(bname, inv_freq.to('cuda'))
                    else:
                        module.register_buffer(bname,
                            torch.zeros_like(buf, device='cuda', dtype=torch.float16))

        hf_model.load_state_dict(state_dict, assign=True, strict=False)
        hf_model.eval()

        # Benchmark HF generate
        prompts = [
            "The meaning of life is",
            "In a galaxy far far away,",
            "The quick brown fox",
            "Once upon a time there was a little",
            "Artificial intelligence will",
        ]
        # Warmup
        inp = tokenizer("Hello", return_tensors="pt").to('cuda')
        with torch.no_grad():
            hf_model.generate(**inp, max_new_tokens=10, do_sample=False)
            hf_model.generate(**inp, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

        total_tokens = 0
        total_time = 0.0
        sample_out = ""
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            input_len = inputs['input_ids'].shape[1]
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                outputs = hf_model.generate(**inputs, max_new_tokens=100, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            new_tok = outputs.shape[1] - input_len
            total_tokens += new_tok
            total_time += elapsed
            if not sample_out:
                sample_out = tokenizer.decode(outputs[0][input_len:])[:60]

        hf_tps = total_tokens / total_time
        hf_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  {'HF generate() baseline':35s}  {hf_tps:6.1f} tok/s  {hf_vram:5.1f} GB  │ {sample_out}...")
        results['HF generate()'] = (hf_tps, hf_vram)

        del hf_model
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        print(f"  HF baseline failed: {e}")
        import traceback; traceback.print_exc()

    # ─── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)

    hf_base = results.get('HF generate()', (44.6, 14.2))[0]
    print(f"\n  {'Configuration':35s}  {'tok/s':>8s}  {'VRAM':>7s}  {'vs HF':>7s}")
    print("  " + "─" * 65)

    best_tps = 0
    best_name = ""
    for name, (tps, vram) in sorted(results.items(), key=lambda x: -x[1][0]):
        speedup = tps / hf_base if hf_base > 0 else 0
        marker = ""
        if tps > best_tps:
            best_tps = tps
            best_name = name
        print(f"  {name:35s}  {tps:7.1f}  {vram:5.1f} GB  {speedup:6.2f}×")

    print(f"\n  ★ Best: {best_name} at {best_tps:.1f} tok/s "
          f"({best_tps/hf_base:.2f}× HF baseline)")

    # ─── Chat mode ────────────────────────────────────────────────
    if args.chat:
        print("\n" + "=" * 80)
        print("  φ-ENCODED CHAT (Custom Engine)")
        print("  Type 'quit' to exit")
        print("=" * 80 + "\n")

        # Reset to best config
        engine.use_linear_mlp = False

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_input or user_input.lower() == 'quit':
                print("Goodbye!")
                break

            ids = tokenizer.encode(user_input, add_special_tokens=False)
            torch.cuda.synchronize()
            t0 = time.time()
            output_ids = engine.generate(ids, max_new_tokens=256, temperature=0.7)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            new_tokens = len(output_ids) - len(ids)
            text = tokenizer.decode(output_ids[len(ids):])
            tps = new_tokens / elapsed if elapsed > 0 else 0

            print(f"\nA: {text}")
            print(f"  [{new_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]\n")


if __name__ == '__main__':
    main()
