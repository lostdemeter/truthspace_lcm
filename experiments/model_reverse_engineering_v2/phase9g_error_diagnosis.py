#!/usr/bin/env python3
"""
Phase 9g: Diagnose the 3-5% Error in phi_softmax Attention
============================================================

phi_softmax is mathematically EXACT (φ^(x/ln(φ)) = e^x).
So where does the 3-5% error come from on 60 prompts?

Hypotheses:
  H1: bfloat16 accumulation — our manual loop accumulates differently
  H2: RoPE implementation differs from model's built-in
  H3: Causal mask handling edge case
  H4: The failing prompts have near-tied top tokens (margin issue)

Tests:
  1. Identify exact failing prompts and their logit margins
  2. Test float32 QK computation vs bfloat16
  3. Compare our RoPE vs model's RoPE output directly
  4. Check if using model's own RoPE cache fixes things
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import math

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

print("=" * 80)
print("  PHASE 9g: ERROR DIAGNOSIS")
print("=" * 80)
print()

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager",
)
model.eval()

N_LAYERS = 28; NUM_HEADS = 28; NUM_KV_HEADS = 4; HEAD_DIM = 128
HEADS_PER_KV = 7; HIDDEN_DIM = 3584


# ================================================================
# Helpers
# ================================================================
def phi_softmax_torch(scores, dim=-1):
    s = scores - scores.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)

def apply_rotary_pos_emb(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

def get_rope_cache(seq_len, head_dim, device, dtype):
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


# ================================================================
# Get model's own RoPE embeddings
# ================================================================
def get_model_rope(seq_len, device, dtype):
    """Get the RoPE cos/sin from the model's own rotary embedding."""
    layer0 = model.model.layers[0].self_attn
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = layer0.rotary_emb(
        torch.zeros(1, 1, seq_len, HEAD_DIM, device=device, dtype=dtype),
        position_ids=pos_ids
    )
    return cos, sin


PROMPTS = [
    "The capital of France is", "The capital of Japan is",
    "The capital of Brazil is", "The capital of Egypt is",
    "The capital of Canada is", "The capital of India is",
    "The capital of Germany is", "The capital of Australia is",
    "The capital of Italy is", "The capital of Mexico is",
    "The chemical symbol for gold is", "The speed of light is approximately",
    "Water freezes at zero degrees", "The largest planet in our solar system is",
    "Albert Einstein developed the theory of", "The atomic number of carbon is",
    "DNA stands for deoxyribonucleic", "The boiling point of water is",
    "Photosynthesis converts sunlight into", "Newton's first law states that an object",
    "To be or not to", "Roses are red, violets are",
    "The quick brown fox jumps over the", "Once upon a time in a land",
    "All that glitters is not", "A penny saved is a penny",
    "The early bird catches the", "Actions speak louder than",
    "When in Rome, do as the", "An apple a day keeps the",
    "The square root of 144 is", "In mathematics, pi is approximately equal to",
    "2 + 2 =", "The derivative of x squared is",
    "A triangle has three sides and three", "The Fibonacci sequence starts with 0, 1,",
    "The area of a circle is pi times the", "10 divided by 2 equals",
    "The sum of angles in a triangle is", "A prime number is only divisible by",
    "The color of grass is", "Shakespeare wrote many",
    "Barack Obama was the", "The largest ocean is the",
    "The color of the sky is usually", "Artificial intelligence is a field of",
    "The Great Wall of China is located in", "The Mona Lisa was painted by",
    "The tallest mountain in the world is", "In machine learning, a neural network",
    "The theory of evolution by natural selection was proposed by Charles",
    "In the year 1969, the first humans landed on the",
    "The periodic table of elements was organized by Dmitri",
    "According to the general theory of relativity, gravity is the curvature of",
    "The human body has approximately 206 bones and the largest bone is the",
    "In computer science, an algorithm is a step by step procedure for",
    "The French Revolution began in the year 1789 and led to the",
    "Quantum mechanics describes the behavior of particles at the",
    "The mitochondria is often called the powerhouse of the",
    "The Declaration of Independence was signed in the year",
]


# ================================================================
# TEST 1: Identify failures and measure margins
# ================================================================
print("-" * 80)
print("  TEST 1: Identify failures and logit margins")
print("-" * 80)
print()

# Baseline
print("  Getting baselines...")
baselines = []
for p in PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    logits = out.logits[0, -1, :].float()
    top2 = logits.topk(2)
    baselines.append({
        'prompt': p,
        'ids': ids,
        'logits': logits,
        'top1_id': top2.indices[0].item(),
        'top1_val': top2.values[0].item(),
        'top2_id': top2.indices[1].item(),
        'top2_val': top2.values[1].item(),
        'margin': (top2.values[0] - top2.values[1]).item(),
        'top1_word': tokenizer.decode([top2.indices[0].item()]),
        'top2_word': tokenizer.decode([top2.indices[1].item()]),
    })

# phi_softmax attention
def run_phi_softmax(ids, compute_dtype=torch.float32, use_model_rope=False):
    """Run phi_softmax attention with configurable precision."""
    b, s = ids.shape[0], ids.shape[1]

    # Capture layernormed inputs to attention
    attn_inputs = {}
    hooks = []
    for li in range(N_LAYERS):
        def mk(storage, layer_idx):
            def hf(module, input, output):
                storage[layer_idx] = output.detach()
            return hf
        h = model.model.layers[li].input_layernorm.register_forward_hook(mk(attn_inputs, li))
        hooks.append(h)

    # Compute our attention per head
    def make_attn_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            h_normed = args[0] if args else kwargs.get('hidden_states')
            if h_normed is None:
                return output
            b, s, _ = h_normed.shape

            with torch.no_grad():
                Q = module.q_proj(h_normed).to(compute_dtype)
                K = module.k_proj(h_normed).to(compute_dtype)
                V = module.v_proj(h_normed)

            Q = Q.reshape(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            K = K.reshape(b, s, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            Vk = V.reshape(b, s, NUM_KV_HEADS, HEAD_DIM)
            Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)

            if use_model_rope:
                cos, sin = get_model_rope(s, h_normed.device, compute_dtype)
                # Model rope returns (batch, seq, head_dim) — reshape
                Q_rope = apply_rotary_pos_emb(Q, cos.unsqueeze(1), sin.unsqueeze(1))
                K_rope = apply_rotary_pos_emb(K, cos.unsqueeze(1), sin.unsqueeze(1))
            else:
                c, sn = get_rope_cache(s, HEAD_DIM, h_normed.device, compute_dtype)
                Q_rope = apply_rotary_pos_emb(Q, c, sn)
                K_rope = apply_rotary_pos_emb(K, c, sn)

            Ke = K_rope.repeat_interleave(HEADS_PER_KV, dim=1)
            out = torch.zeros(b, s, NUM_HEADS, HEAD_DIM, device=h_normed.device, dtype=h_normed.dtype)
            mask = torch.triu(torch.ones(s, s, device=h_normed.device), diagonal=1).bool()

            for hd in range(NUM_HEADS):
                sc = Q_rope[0, hd] @ Ke[0, hd].T / math.sqrt(HEAD_DIM)
                sc.masked_fill_(mask, float('-inf'))
                w = phi_softmax_torch(sc.to(compute_dtype), dim=-1)
                out[0, :, hd, :] = (w @ Ve[0, :, hd, :].to(compute_dtype)).to(h_normed.dtype)

            geo = module.o_proj(out.reshape(b, s, NUM_HEADS * HEAD_DIM))
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    attn_hooks = []
    for li in range(N_LAYERS):
        hk = model.model.layers[li].self_attn.register_forward_hook(
            make_attn_hook(li), with_kwargs=True)
        attn_hooks.append(hk)

    with torch.no_grad():
        out = model(ids, return_dict=True)

    for h in hooks:
        h.remove()
    for h in attn_hooks:
        h.remove()

    return out.logits[0, -1, :].float()


# Test with default settings (float32 QK, our RoPE)
print("  Testing phi_softmax (float32, our RoPE)...")
failures_f32 = []
for i, bl in enumerate(baselines):
    geo_logits = run_phi_softmax(bl['ids'], compute_dtype=torch.float32, use_model_rope=False)
    geo_top1 = geo_logits.argmax().item()
    cos = F.cosine_similarity(bl['logits'].unsqueeze(0), geo_logits.unsqueeze(0)).item()
    if geo_top1 != bl['top1_id']:
        geo_word = tokenizer.decode([geo_top1])
        # Where does the correct token rank?
        correct_rank = (geo_logits >= geo_logits[bl['top1_id']]).sum().item()
        failures_f32.append({
            'idx': i, 'prompt': bl['prompt'],
            'base_word': bl['top1_word'], 'geo_word': geo_word,
            'margin': bl['margin'], 'cos': cos,
            'correct_rank': correct_rank,
            'base_logit_diff': (bl['logits'][bl['top1_id']] - bl['logits'][geo_top1]).item(),
            'geo_logit_diff': (geo_logits[bl['top1_id']] - geo_logits[geo_top1]).item(),
        })

print(f"\n  phi_softmax (float32, our RoPE): {len(PROMPTS) - len(failures_f32)}/{len(PROMPTS)}")
if failures_f32:
    print()
    for f in failures_f32:
        print(f"  FAIL #{f['idx']:2d}: \"{f['prompt'][:50]}\"")
        print(f"    base='{f['base_word']}' got='{f['geo_word']}'")
        print(f"    baseline margin (top1-top2): {f['margin']:.3f}")
        print(f"    base logit(correct)-logit(got): {f['base_logit_diff']:.3f}")
        print(f"    geo  logit(correct)-logit(got): {f['geo_logit_diff']:.3f}")
        print(f"    correct token rank in geo: {f['correct_rank']}")
        print(f"    cos(base, geo): {f['cos']:.6f}")
        print()

# ================================================================
# TEST 2: Try bfloat16 QK to see if precision matters
# ================================================================
print("-" * 80)
print("  TEST 2: bfloat16 vs float32 QK computation")
print("-" * 80)
print()

print("  Testing phi_softmax (bfloat16 QK)...")
failures_bf16 = []
for i, bl in enumerate(baselines):
    geo_logits = run_phi_softmax(bl['ids'], compute_dtype=torch.bfloat16, use_model_rope=False)
    geo_top1 = geo_logits.argmax().item()
    if geo_top1 != bl['top1_id']:
        failures_bf16.append(i)

print(f"  bfloat16 QK: {len(PROMPTS) - len(failures_bf16)}/{len(PROMPTS)}")
print(f"  float32 QK:  {len(PROMPTS) - len(failures_f32)}/{len(PROMPTS)}")
print()

# ================================================================
# TEST 3: Use model's own RoPE
# ================================================================
print("-" * 80)
print("  TEST 3: Our RoPE vs model's RoPE")
print("-" * 80)
print()

# First check if our RoPE matches
test_ids = baselines[0]['ids']
s = test_ids.shape[1]
our_cos, our_sin = get_rope_cache(s, HEAD_DIM, torch.device("cuda"), torch.float32)
try:
    model_cos, model_sin = get_model_rope(s, torch.device("cuda"), torch.float32)
    # Compare shapes and values
    print(f"  Our RoPE shape:   cos={our_cos.shape}, sin={our_sin.shape}")
    print(f"  Model RoPE shape: cos={model_cos.shape}, sin={model_sin.shape}")

    # Reshape to compare
    our_c = our_cos.squeeze()  # (seq, head_dim)
    mdl_c = model_cos.squeeze()  # might be different shape
    print(f"  Our cos squeezed: {our_c.shape}")
    print(f"  Model cos squeezed: {mdl_c.shape}")

    if our_c.shape == mdl_c.shape:
        diff = (our_c - mdl_c).abs().max().item()
        print(f"  Max cos difference: {diff:.2e}")
    else:
        # Try to align
        min_len = min(our_c.shape[0], mdl_c.shape[0])
        if our_c.dim() == mdl_c.dim():
            diff = (our_c[:min_len] - mdl_c[:min_len]).abs().max().item()
            print(f"  Max cos difference (first {min_len} positions): {diff:.2e}")
except Exception as e:
    print(f"  Model RoPE extraction failed: {e}")
    print("  Testing with model's internal rope via hooks instead...")

print()

# ================================================================
# TEST 4: Compare attention outputs head-by-head
# ================================================================
print("-" * 80)
print("  TEST 4: Per-head attention output comparison")
print("-" * 80)
print()

# For each failing prompt, compare our attention output vs model's
# at each layer to find where the divergence starts
if failures_f32:
    fail = failures_f32[0]
    ids = baselines[fail['idx']]['ids']
    prompt = fail['prompt']
    print(f"  Diagnosing: \"{prompt}\"")
    print()

    # Capture model's real attention outputs per layer
    real_attn_outputs = {}
    hooks = []
    for li in range(N_LAYERS):
        def mk(storage, layer_idx):
            def hf(module, args, kwargs, output):
                if isinstance(output, tuple):
                    storage[layer_idx] = output[0].detach().clone()
                else:
                    storage[layer_idx] = output.detach().clone()
            return hf
        h = model.model.layers[li].self_attn.register_forward_hook(mk(real_attn_outputs, li), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        model(ids, return_dict=True)

    for h in hooks:
        h.remove()

    # Now run with phi_softmax and capture our outputs
    geo_attn_outputs = {}
    hooks2 = []

    def make_capture_and_replace(layer_idx, storage):
        def hook_fn(module, args, kwargs, output):
            h_normed = args[0] if args else kwargs.get('hidden_states')
            if h_normed is None:
                return output
            b, s, _ = h_normed.shape
            with torch.no_grad():
                Q = module.q_proj(h_normed).float()
                K = module.k_proj(h_normed).float()
                V = module.v_proj(h_normed)
            Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
            K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
            Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
            Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
            c, sn = get_rope_cache(s, HEAD_DIM, h_normed.device, torch.float32)
            Q = apply_rotary_pos_emb(Q, c, sn)
            K = apply_rotary_pos_emb(K, c, sn)
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h_normed.device, dtype=h_normed.dtype)
            mask = torch.triu(torch.ones(s,s,device=h_normed.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
                sc.masked_fill_(mask, float('-inf'))
                w = phi_softmax_torch(sc, dim=-1)
                out[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h_normed.dtype)
            geo = module.o_proj(out.reshape(b,s,NUM_HEADS*HEAD_DIM))
            storage[layer_idx] = geo.detach().clone()
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    for li in range(N_LAYERS):
        h = model.model.layers[li].self_attn.register_forward_hook(
            make_capture_and_replace(li, geo_attn_outputs), with_kwargs=True)
        hooks2.append(h)

    with torch.no_grad():
        model(ids, return_dict=True)

    for h in hooks2:
        h.remove()

    # Compare per-layer
    print(f"  {'Layer':>5s}  {'cos(real,geo)':>12s}  {'max_diff':>10s}  {'mean_diff':>10s}")
    print("  " + "-" * 45)
    for li in range(N_LAYERS):
        real = real_attn_outputs[li][0, -1, :].float()
        geo = geo_attn_outputs[li][0, -1, :].float()
        cos = F.cosine_similarity(real.unsqueeze(0), geo.unsqueeze(0)).item()
        maxd = (real - geo).abs().max().item()
        meand = (real - geo).abs().mean().item()
        flag = " <<<" if cos < 0.999 else ""
        print(f"  L{li:2d}    {cos:12.8f}  {maxd:10.4f}  {meand:10.6f}{flag}")

print()

# ================================================================
# TEST 5: Is it phi_softmax specifically, or would standard softmax
# in our manual implementation also fail?
# ================================================================
print("-" * 80)
print("  TEST 5: phi_softmax vs standard softmax in manual implementation")
print("-" * 80)
print()

def run_manual_softmax(ids):
    """Same as phi_softmax path but using torch standard softmax."""
    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            h_normed = args[0] if args else kwargs.get('hidden_states')
            if h_normed is None: return output
            b, s, _ = h_normed.shape
            with torch.no_grad():
                Q = module.q_proj(h_normed).float()
                K = module.k_proj(h_normed).float()
                V = module.v_proj(h_normed)
            Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
            K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
            Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
            Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
            c, sn = get_rope_cache(s, HEAD_DIM, h_normed.device, torch.float32)
            Q = apply_rotary_pos_emb(Q, c, sn)
            K = apply_rotary_pos_emb(K, c, sn)
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h_normed.device, dtype=h_normed.dtype)
            mask = torch.triu(torch.ones(s,s,device=h_normed.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
                sc.masked_fill_(mask, float('-inf'))
                w = torch.softmax(sc, dim=-1)  # STANDARD softmax
                out[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h_normed.dtype)
            geo = module.o_proj(out.reshape(b,s,NUM_HEADS*HEAD_DIM))
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    hooks = []
    for li in range(N_LAYERS):
        h = model.model.layers[li].self_attn.register_forward_hook(
            make_hook(li), with_kwargs=True)
        hooks.append(h)
    with torch.no_grad():
        out = model(ids, return_dict=True)
    for h in hooks: h.remove()
    return out.logits[0, -1, :].float()

# Compare phi_softmax vs standard softmax on failing prompts
n_phi = 0; n_std = 0
for i, bl in enumerate(baselines):
    phi_logits = run_phi_softmax(bl['ids'])
    std_logits = run_manual_softmax(bl['ids'])
    if phi_logits.argmax().item() == bl['top1_id']: n_phi += 1
    if std_logits.argmax().item() == bl['top1_id']: n_std += 1

print(f"  Manual phi_softmax: {n_phi}/{len(PROMPTS)}")
print(f"  Manual std softmax: {n_std}/{len(PROMPTS)}")
print()
if n_phi == n_std:
    print("  → Same accuracy. Error is NOT from phi_softmax itself.")
    print("  → Error comes from the manual attention implementation.")
elif n_std > n_phi:
    print(f"  → Standard softmax is better by {n_std - n_phi}.")
    print("  → phi_softmax introduces additional precision error.")
else:
    print(f"  → phi_softmax is actually better by {n_phi - n_std}!")

print()
print("=" * 80)
print("  DONE")
print("=" * 80)
