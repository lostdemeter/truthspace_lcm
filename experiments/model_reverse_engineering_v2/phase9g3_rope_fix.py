#!/usr/bin/env python3
"""
Phase 9g3: Does matching model's exact RoPE fix the 3/60 failures?

Root cause from 9g/9g2:
- phi_softmax vs standard softmax: identical results (not the issue)
- bfloat16 vs float32 QK: identical results (not the issue)
- Our RoPE vs model's RoPE: max diff = 1.91e-03 (bfloat16 rounding)
- This small diff compounds across 28 layers → cos=-0.35 at L27
- Failing prompts have tiny margins (0.125, 0.250, 0.875)

Fix: compute RoPE in bfloat16 to match model exactly.
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import math

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager",
)
model.eval()

N_LAYERS = 28; NUM_HEADS = 28; NUM_KV_HEADS = 4; HEAD_DIM = 128
HEADS_PER_KV = 7

def phi_softmax_torch(scores, dim=-1):
    s = scores - scores.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)


# ================================================================
# Three RoPE implementations to compare
# ================================================================
def rope_ours_f32(seq_len, device):
    """Our original: float32 throughout."""
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]  # (1, 1, seq, dim)

def rope_model_match(seq_len, device):
    """Match model's Qwen2RotaryEmbedding exactly."""
    # The model's rotary_emb computes in float32 but casts to model dtype (bfloat16)
    # Let's replicate exactly what the model does
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq  # stored on device
    pos = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Model casts to bfloat16 then back — this is where precision is lost
    cos = emb.cos().to(torch.bfloat16).float()
    sin = emb.sin().to(torch.bfloat16).float()
    return cos[None, None], sin[None, None]

def rope_capture_from_model(ids):
    """Capture actual RoPE values the model uses during forward pass."""
    captured = {}
    def pre_hook(module, args, kwargs):
        if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
            c, s = kwargs['position_embeddings']
            captured['cos'] = c.detach().float()
            captured['sin'] = s.detach().float()
    h = model.model.layers[0].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        model(ids)
    h.remove()
    c = captured['cos']  # (batch, seq, dim)
    s = captured['sin']
    return c.unsqueeze(1), s.unsqueeze(1)  # (batch, 1, seq, dim) for head broadcast


# Verify all three match
print("=" * 80)
print("  ROPE COMPARISON")
print("=" * 80)
print()

test_ids = tokenizer.encode("The capital of Mexico is", return_tensors="pt").to("cuda")
s = test_ids.shape[1]

c_f32, s_f32 = rope_ours_f32(s, torch.device("cuda"))
c_match, s_match = rope_model_match(s, torch.device("cuda"))
c_cap, s_cap = rope_capture_from_model(test_ids)

# Compare
c_f32_sq = c_f32.squeeze()[:s]
c_match_sq = c_match.squeeze()[:s]
c_cap_sq = c_cap.squeeze()[:s]

print(f"  f32 vs model-match: max diff = {(c_f32_sq - c_match_sq).abs().max().item():.2e}")
print(f"  f32 vs captured:    max diff = {(c_f32_sq - c_cap_sq).abs().max().item():.2e}")
print(f"  model-match vs captured: max diff = {(c_match_sq - c_cap_sq).abs().max().item():.2e}")
print()


# ================================================================
# Test all 60 prompts with each RoPE variant
# ================================================================
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

# Baselines
print("  Getting baselines...")
base_ids = []
base_margins = []
for p in PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    logits = out.logits[0, -1, :].float()
    top2 = logits.topk(2)
    base_ids.append(top2.indices[0].item())
    base_margins.append((top2.values[0] - top2.values[1]).item())
print("  Done.")
print()


def apply_rotary(x, cos, sin):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)


def test_config(name, rope_fn):
    """Test phi_softmax attention with a given RoPE function."""

    def make_hook(li, rope_func):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None: return output
            b, s, _ = h.shape
            with torch.no_grad():
                Q = module.q_proj(h).float()
                K = module.k_proj(h).float()
                V = module.v_proj(h)
            Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
            K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
            Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
            Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
            cos, sin = rope_func(s, h.device)
            Q = apply_rotary(Q, cos, sin)
            K = apply_rotary(K, cos, sin)
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out_t = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
            mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
                sc.masked_fill_(mask, float('-inf'))
                w = phi_softmax_torch(sc, dim=-1)
                out_t[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h.dtype)
            geo = module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM))
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    nm = 0
    fails = []
    for pi, p in enumerate(PROMPTS):
        ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
        hooks = []
        for li in range(N_LAYERS):
            hk = model.model.layers[li].self_attn.register_forward_hook(
                make_hook(li, rope_fn), with_kwargs=True)
            hooks.append(hk)
        with torch.no_grad():
            out = model(ids, return_dict=True)
        for hk in hooks: hk.remove()
        if out.logits[0,-1,:].float().argmax().item() == base_ids[pi]:
            nm += 1
        else:
            got = tokenizer.decode([out.logits[0,-1,:].float().argmax().item()])
            base = tokenizer.decode([base_ids[pi]])
            fails.append((pi, p[:45], base, got, base_margins[pi]))

    print(f"  {name:>40s}: {nm}/{len(PROMPTS)} ({nm/len(PROMPTS):.0%})")
    for fi, fp, fb, fg, margin in fails:
        print(f"    FAIL #{fi}: \"{fp}\" base='{fb}' got='{fg}' margin={margin:.3f}")
    return nm


print("=" * 80)
print("  TESTING ROPE VARIANTS (60 prompts)")
print("=" * 80)
print()

# Test 1: Our original float32 RoPE
test_config("phi_softmax + our f32 RoPE", rope_ours_f32)
print()

# Test 2: Model-matched RoPE (same inv_freq, same bfloat16 cast)
test_config("phi_softmax + model-matched RoPE", rope_model_match)
print()

# Test 3: Captured RoPE from model (ground truth)
# Need a version that captures per-prompt
def rope_captured_factory():
    """Returns a rope_fn that captures from the model for each seq_len."""
    cache = {}
    def rope_fn(seq_len, device):
        if seq_len not in cache:
            # Generate dummy input of right length
            dummy = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            captured = {}
            def pre_hook(module, args, kwargs):
                if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
                    c, s = kwargs['position_embeddings']
                    captured['cos'] = c.detach().float()
                    captured['sin'] = s.detach().float()
            h = model.model.layers[0].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
            with torch.no_grad():
                model(dummy)
            h.remove()
            cache[seq_len] = (captured['cos'].unsqueeze(1), captured['sin'].unsqueeze(1))
        return cache[seq_len]
    return rope_fn

test_config("phi_softmax + captured model RoPE", rope_captured_factory())
print()

# Test 4: Compute in float32 but cast to bfloat16 then back (simulating model)
def rope_f32_via_bf16(seq_len, device):
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Cast through bfloat16 to match model precision
    cos = emb.cos().to(torch.bfloat16).float()
    sin = emb.sin().to(torch.bfloat16).float()
    return cos[None, None], sin[None, None]

test_config("phi_softmax + f32-via-bf16 RoPE", rope_f32_via_bf16)
print()

# Test 5: Do QK in bfloat16 too (fully match model numerics)
def test_full_bf16():
    """Everything in bfloat16 — match model numerics exactly."""
    def make_hook(li):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None: return output
            b, s, _ = h.shape
            with torch.no_grad():
                Q = module.q_proj(h)  # stays bfloat16
                K = module.k_proj(h)
                V = module.v_proj(h)
            Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
            K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
            Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
            Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
            # RoPE in bfloat16
            inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device=h.device, dtype=torch.float32) / HEAD_DIM))
            pos = torch.arange(s, device=h.device, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(torch.bfloat16)[None, None]
            sin = emb.sin().to(torch.bfloat16)[None, None]
            Q = apply_rotary(Q, cos, sin)
            K = apply_rotary(K, cos, sin)
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out_t = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
            mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = (Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)).float()
                sc.masked_fill_(mask, float('-inf'))
                w = phi_softmax_torch(sc, dim=-1)
                out_t[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h.dtype)
            geo = module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM))
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    nm = 0
    fails = []
    for pi, p in enumerate(PROMPTS):
        ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
        hooks = []
        for li in range(N_LAYERS):
            hk = model.model.layers[li].self_attn.register_forward_hook(
                make_hook(li), with_kwargs=True)
            hooks.append(hk)
        with torch.no_grad():
            out = model(ids, return_dict=True)
        for hk in hooks: hk.remove()
        if out.logits[0,-1,:].float().argmax().item() == base_ids[pi]:
            nm += 1
        else:
            got = tokenizer.decode([out.logits[0,-1,:].float().argmax().item()])
            base = tokenizer.decode([base_ids[pi]])
            fails.append((pi, p[:45], base, got, base_margins[pi]))

    print(f"  {'phi_softmax fully bf16 matched':>40s}: {nm}/{len(PROMPTS)} ({nm/len(PROMPTS):.0%})")
    for fi, fp, fb, fg, margin in fails:
        print(f"    FAIL #{fi}: \"{fp}\" base='{fb}' got='{fg}' margin={margin:.3f}")
    return nm

test_full_bf16()
print()

print("=" * 80)
print("  MARGIN ANALYSIS")
print("=" * 80)
print()

margins_sorted = sorted(enumerate(base_margins), key=lambda x: x[1])
print(f"  Smallest 10 baseline margins:")
for idx, m in margins_sorted[:10]:
    print(f"    #{idx:2d}: margin={m:.3f}  \"{PROMPTS[idx][:50]}\"")
print()
print(f"  Margin stats: min={min(base_margins):.3f}, median={np.median(base_margins):.3f}, mean={np.mean(base_margins):.3f}")

print()
print("=" * 80)
print("  DONE")
print("=" * 80)
