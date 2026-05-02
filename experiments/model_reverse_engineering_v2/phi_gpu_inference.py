#!/usr/bin/env python3
"""
φ-Encoded Qwen2-7B GPU Inference
=================================

Converts φ-encoded weights to float16, loads into HuggingFace
Qwen2ForCausalLM, and runs GPU inference.

This proves: the φ-encoded model (sign × φ^(exp/128)) works as
a real chatbot, validating the TruthSpace hypothesis that
structure IS information.

Usage:
    python phi_gpu_inference.py              # benchmark + chat
    python phi_gpu_inference.py --benchmark  # benchmark only
    python phi_gpu_inference.py --chat       # chat only
"""

import numpy as np
import torch
import torch.nn as nn
import os
import sys
import gc
import time
import argparse
import unicodedata

PHI = (1 + np.sqrt(5)) / 2
GRID = 128
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')


# ─── Vocabulary Partitioning (F17) ────────────────────────────────────

def build_bpe_byte_decoder():
    bs = list(range(ord('!'), ord('~')+1)) + \
         list(range(ord('\u00a1'), ord('\u00ac')+1)) + \
         list(range(ord('\u00ae'), ord('\u00ff')+1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

BPE_BYTE_DECODER = build_bpe_byte_decoder()

def decode_bpe_token(tok_str):
    try:
        return bytes([BPE_BYTE_DECODER[c] for c in tok_str]).decode('utf-8', errors='replace')
    except (KeyError, UnicodeDecodeError):
        return tok_str

def is_english_token(tok_str):
    decoded = decode_bpe_token(tok_str).strip()
    if not decoded or decoded == '\ufffd':
        return True
    for ch in decoded:
        if ch == '\ufffd': continue
        cp = ord(ch)
        cat = unicodedata.category(ch)
        # Allow: ASCII letters, Latin Extended, Common (punct/digits/symbols)
        is_latin = cp < 0x0080 or (0x0080 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF)
        is_common = not cat.startswith('L')
        if not (is_latin or is_common):
            return False
    return True

def get_english_ids(tokenizer_vocab_path=None):
    """Get English token IDs from tokenizer vocab."""
    import json
    for candidate in [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
    ]:
        if os.path.exists(candidate):
            snapshots = os.listdir(candidate)
            if snapshots:
                vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                if os.path.exists(vocab_file):
                    with open(vocab_file, 'r') as f:
                        tokenizer_data = json.load(f)
                    vocab = tokenizer_data.get('model', {}).get('vocab', {})
                    id_to_token = {idx: tok for tok, idx in vocab.items()}
                    english_ids = []
                    for tid in range(max(id_to_token.keys()) + 1):
                        tok = id_to_token.get(tid, '')
                        if not tok or tok.startswith('<|') or tok.startswith('<unused'):
                            english_ids.append(tid)
                        elif is_english_token(tok):
                            english_ids.append(tid)
                    return english_ids
    return None


class ReducedLMHead(nn.Module):
    """LM head that only computes logits for English tokens.
    Returns full-vocab-sized logits with -inf for non-English tokens.
    The matmul is ~31% smaller (104K vs 152K output dim)."""

    def __init__(self, full_weight, english_ids):
        super().__init__()
        self.english_ids = torch.tensor(english_ids, dtype=torch.long, device=full_weight.device)
        self.reduced_weight = nn.Parameter(full_weight[self.english_ids])
        self.vocab_size = full_weight.shape[0]
        # Pre-allocate scatter index for fast expansion
        self.register_buffer('_english_ids', self.english_ids)
        del self.english_ids

    def forward(self, hidden_states):
        # Reduced matmul: (batch, seq, dim) @ (n_english, dim).T
        reduced_logits = torch.matmul(hidden_states, self.reduced_weight.T)
        # Expand to full vocab with -inf
        shape = hidden_states.shape[:-1] + (self.vocab_size,)
        full_logits = torch.full(shape, torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device, dtype=hidden_states.dtype)
        full_logits[..., self._english_ids] = reduced_logits
        return full_logits


# ─── Weight loading ───────────────────────────────────────────────────

def decode_phi_to_tensor(path):
    """Decode φ-encoded .npz to a float16 PyTorch tensor."""
    d = np.load(path)
    signs = d['signs'].astype(np.float32)
    exponents = d['exponents'].astype(np.float32)
    # sign × φ^(exponent / 128)
    values = signs * (np.float32(PHI) ** (exponents / np.float32(GRID)))
    return torch.from_numpy(values).half()


def build_state_dict():
    """
    Convert all φ-encoded weights to a HuggingFace Qwen2 state dict.
    Converts layer by layer to manage memory.
    """
    state_dict = {}
    t_start = time.time()

    # Embedding
    print("  Converting embed_tokens...")
    sys.stdout.flush()
    t0 = time.time()
    state_dict['model.embed_tokens.weight'] = decode_phi_to_tensor(
        os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(f"    {state_dict['model.embed_tokens.weight'].shape} ({time.time()-t0:.1f}s)")

    # LM head
    print("  Converting lm_head...")
    sys.stdout.flush()
    t0 = time.time()
    state_dict['lm_head.weight'] = decode_phi_to_tensor(
        os.path.join(MODEL_DIR, 'lm_head.npz'))
    print(f"    {state_dict['lm_head.weight'].shape} ({time.time()-t0:.1f}s)")

    # Final norm (not φ-encoded, just float32)
    print("  Converting final norm...")
    fn = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))
    state_dict['model.norm.weight'] = torch.from_numpy(
        fn['weight'].astype(np.float32)).half()

    # Layers
    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        prefix = f'model.layers.{layer_idx}'
        t0 = time.time()
        print(f"  Converting layer {layer_idx:2d}...", end='', flush=True)

        # Norms (not φ-encoded)
        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        state_dict[f'{prefix}.input_layernorm.weight'] = torch.from_numpy(
            norms['input_layernorm'].astype(np.float32)).half()
        state_dict[f'{prefix}.post_attention_layernorm.weight'] = torch.from_numpy(
            norms['post_attention_layernorm'].astype(np.float32)).half()

        # Biases (not φ-encoded)
        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        state_dict[f'{prefix}.self_attn.q_proj.bias'] = torch.from_numpy(
            biases['q_proj_bias'].astype(np.float32)).half()
        state_dict[f'{prefix}.self_attn.k_proj.bias'] = torch.from_numpy(
            biases['k_proj_bias'].astype(np.float32)).half()
        state_dict[f'{prefix}.self_attn.v_proj.bias'] = torch.from_numpy(
            biases['v_proj_bias'].astype(np.float32)).half()

        # Attention weights (φ-encoded)
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            state_dict[f'{prefix}.self_attn.{proj}.weight'] = decode_phi_to_tensor(
                os.path.join(layer_dir, f'{proj}.npz'))

        # MLP weights (φ-encoded)
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            state_dict[f'{prefix}.mlp.{proj}.weight'] = decode_phi_to_tensor(
                os.path.join(layer_dir, f'{proj}.npz'))

        elapsed = time.time() - t0
        print(f" ({elapsed:.1f}s)")
        sys.stdout.flush()

        # Force garbage collection after each layer
        gc.collect()

    total = time.time() - t_start
    n_params = sum(v.numel() for v in state_dict.values())
    size_gb = sum(v.numel() * v.element_size() for v in state_dict.values()) / 1024**3
    print(f"\n  Conversion complete: {n_params/1e9:.2f}B params, {size_gb:.1f} GB, {total:.0f}s")
    return state_dict


def load_model(state_dict):
    """Load state dict into Qwen2ForCausalLM and move to GPU."""
    from transformers import AutoConfig, Qwen2ForCausalLM

    print("\n  Creating model from config...")
    sys.stdout.flush()
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
    config.torch_dtype = torch.float16

    # Move state dict tensors to CUDA directly to avoid double memory
    print("  Moving state dict to GPU...")
    sys.stdout.flush()
    t0 = time.time()
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device='cuda', dtype=torch.float16)
    gc.collect()
    print(f"  State dict on GPU in {time.time()-t0:.1f}s")

    # Create model on meta device, then load with assign=True
    print("  Creating model shell and loading weights...")
    sys.stdout.flush()
    t0 = time.time()
    with torch.device('meta'):
        model = Qwen2ForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=False)

    # Materialize any remaining meta tensors (e.g., RoPE inv_freq)
    print("  Initializing RoPE buffers...")
    for name, module in model.named_modules():
        for bname, buf in list(module.named_buffers(recurse=False)):
            if buf.device == torch.device('meta'):
                # Re-create the buffer on CUDA with proper values
                if 'inv_freq' in bname:
                    head_dim = config.hidden_size // config.num_attention_heads
                    inv_freq = 1.0 / (config.rope_theta ** (
                        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    module.register_buffer(bname, inv_freq.to('cuda'))
                else:
                    # Generic: just zero-initialize on CUDA
                    module.register_buffer(bname,
                        torch.zeros_like(buf, device='cuda', dtype=torch.float16))

    model.eval()
    print(f"  Model ready in {time.time()-t0:.1f}s")

    # Report VRAM usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    return model


def benchmark(model, tokenizer, device='cuda'):
    """Benchmark inference speed."""
    print("\n" + "=" * 70)
    print("  BENCHMARK")
    print("=" * 70)

    prompts = [
        "The meaning of life is",
        "In a galaxy far far away,",
        "The quick brown fox",
        "Once upon a time there was a little",
        "Artificial intelligence will",
    ]

    # Warmup
    print("\n  Warmup...")
    with torch.no_grad():
        inputs = tokenizer("Hello", return_tensors="pt").to(device)
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark: prefill + generation
    gen_tokens = 100
    print(f"\n  Generating {gen_tokens} tokens per prompt...\n")

    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_tokens,
                do_sample=False,
                temperature=1.0,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        output_len = outputs.shape[1] - input_len
        tps = output_len / elapsed

        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        total_tokens += output_len
        total_time += elapsed

        # Show first 80 chars of generation
        display = generated_text[:100].replace('\n', ' ')
        print(f"  Prompt: {repr(prompt)}")
        print(f"    → {display}...")
        print(f"    {output_len} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
        print()

    avg_tps = total_tokens / total_time
    print(f"  Average: {avg_tps:.1f} tokens/second")
    print(f"  Total: {total_tokens} tokens in {total_time:.1f}s")

    # VRAM after generation
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM after generation: {allocated:.1f} GB")

    return avg_tps


def chat(model, tokenizer, device='cuda'):
    """Interactive chat mode."""
    print("\n" + "=" * 70)
    print("  φ-ENCODED QWEN2-7B CHAT")
    print("  (Type 'quit' to exit, 'clear' to reset)")
    print("=" * 70)
    print()

    messages = []
    system_msg = "You are a helpful assistant."

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        if user_input.lower() == 'clear':
            messages = []
            print("(conversation cleared)\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # Build prompt with chat template
        prompt_messages = [{"role": "system", "content": system_msg}] + messages
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].shape[1]

        # Check if context is getting too long
        if input_len > 4096:
            print("(context too long, clearing older messages)")
            messages = messages[-4:]
            prompt_messages = [{"role": "system", "content": system_msg}] + messages
            text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs['input_ids'].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        output_tokens = outputs.shape[1] - input_len
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        tps = output_tokens / elapsed if elapsed > 0 else 0
        print(f"\nAssistant: {response}")
        print(f"  [{output_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]\n")

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="φ-Encoded Qwen2-7B GPU Inference")
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark only')
    parser.add_argument('--chat', action='store_true', help='Run chat only')
    parser.add_argument('--english', action='store_true', help='Use English-only vocab partition (F17)')
    args = parser.parse_args()

    # Default: both
    if not args.benchmark and not args.chat:
        args.benchmark = True
        args.chat = True

    print("=" * 70)
    print("  φ-Encoded Qwen2-7B → GPU Inference")
    print("  TruthSpace: Structure IS Information")
    print("=" * 70)
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("  ERROR: No CUDA GPU available")
        return
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
    print()

    # Convert weights
    print("  Phase 1: Converting φ-encoded weights to float16")
    print("  " + "-" * 50)
    state_dict = build_state_dict()

    # Load model
    print("\n  Phase 2: Loading into Qwen2ForCausalLM on GPU")
    print("  " + "-" * 50)
    model = load_model(state_dict)

    # Load tokenizer
    print("\n  Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer ready ({tokenizer.vocab_size} tokens)")

    # Apply vocabulary partitioning if requested
    if args.english:
        print("\n  Phase 3: Applying English-only vocabulary partition (F17)")
        print("  " + "-" * 50)
        english_ids = get_english_ids()
        if english_ids:
            n_eng = len(english_ids)
            full_weight = model.lm_head.weight.data
            print(f"  English tokens: {n_eng:,d} / {full_weight.shape[0]:,d} "
                  f"({n_eng/full_weight.shape[0]*100:.1f}%)")
            reduced_head = ReducedLMHead(full_weight, english_ids)
            # Replace lm_head
            old_size = full_weight.numel() * full_weight.element_size() / 1024**3
            new_size = reduced_head.reduced_weight.numel() * reduced_head.reduced_weight.element_size() / 1024**3
            model.lm_head = reduced_head
            gc.collect()
            torch.cuda.empty_cache()
            saved = old_size - new_size
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  lm_head: {old_size:.2f} GB → {new_size:.2f} GB (saved {saved:.2f} GB)")
            print(f"  VRAM after partition: {allocated:.1f} GB")
        else:
            print("  WARNING: Could not load vocab, running without partition")

    # Run
    if args.benchmark:
        benchmark(model, tokenizer)

    if args.chat:
        chat(model, tokenizer)


if __name__ == '__main__':
    main()
