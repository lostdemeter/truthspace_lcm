#!/usr/bin/env python3
"""
Phase 9g4: Focused dtype test — is the 3/60 error from float32/bfloat16 mixing?

Prior findings:
- phi_softmax vs standard softmax: same results (not phi_softmax)
- float32 vs bfloat16 QK scores: same results (not QK precision)
- Our RoPE vs model's exact RoPE: same results (not RoPE diff)
- Failing prompts have tiny margins (0.125, 0.250, 0.875)
- Per-layer cos ~0.99, compounds to cos=-0.35 at L27

Hypothesis: our implementation computes Q,K in float32 then casts
back to bfloat16. The model stays in bfloat16 throughout. The tiny
per-layer difference from different accumulation patterns compounds.

Test: run attention ENTIRELY in bfloat16 (no float32 upcast) to
match the model's native precision path.
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

def apply_rotary(x, cos, sin):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

def get_rope(seq_len, device, dtype):
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

# Baselines
print("Getting baselines...")
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
print(f"  {len(PROMPTS)} baselines ready")
print()


def test_variant(name, compute_dtype, rope_dtype, softmax_fn):
    """Test with specific dtype choices."""
    def make_hook(li):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None: return output
            b, s, _ = h.shape
            with torch.no_grad():
                Q = module.q_proj(h).to(compute_dtype)
                K = module.k_proj(h).to(compute_dtype)
                V = module.v_proj(h)
            Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
            K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
            Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
            Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
            cos, sin = get_rope(s, h.device, rope_dtype)
            Q = apply_rotary(Q, cos.to(compute_dtype), sin.to(compute_dtype))
            K = apply_rotary(K, cos.to(compute_dtype), sin.to(compute_dtype))
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out_t = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
            mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
                sc = sc.float()  # softmax always in float32
                sc.masked_fill_(mask, float('-inf'))
                w = softmax_fn(sc, dim=-1)
                out_t[0,:,hd,:] = (w.to(compute_dtype) @ Ve[0,:,hd,:].to(compute_dtype)).to(h.dtype)
            geo = module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM))
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    nm = 0; fails = []
    for pi, p in enumerate(PROMPTS):
        ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
        hooks = []
        for li in range(N_LAYERS):
            hooks.append(model.model.layers[li].self_attn.register_forward_hook(
                make_hook(li), with_kwargs=True))
        with torch.no_grad():
            out = model(ids, return_dict=True)
        for hk in hooks: hk.remove()
        if out.logits[0,-1,:].float().argmax().item() == base_ids[pi]:
            nm += 1
        else:
            got = tokenizer.decode([out.logits[0,-1,:].float().argmax().item()])
            base = tokenizer.decode([base_ids[pi]])
            fails.append((pi, p[:45], base, got, base_margins[pi]))
    print(f"  {name:>50s}: {nm}/{len(PROMPTS)} ({nm/len(PROMPTS):.0%})")
    for fi, fp, fb, fg, m in fails:
        print(f"    FAIL #{fi}: \"{fp}\" base='{fb}' got='{fg}' margin={m:.3f}")
    return nm, fails


def phi_softmax(scores, dim=-1):
    s = scores - scores.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)


print("=" * 80)
print("  DTYPE VARIANTS (60 prompts)")
print("=" * 80)
print()

# A: Our original (QK in f32, RoPE in f32, phi_softmax)
test_variant("A: QK=f32, RoPE=f32, phi_softmax",
             torch.float32, torch.float32, phi_softmax)
print()

# B: All bf16 except softmax (match model's native path)
test_variant("B: QK=bf16, RoPE=bf16, phi_softmax",
             torch.bfloat16, torch.bfloat16, phi_softmax)
print()

# C: All bf16 with standard softmax
test_variant("C: QK=bf16, RoPE=bf16, std softmax",
             torch.bfloat16, torch.bfloat16, lambda s, dim: torch.softmax(s, dim=dim))
print()

# D: Mixed like model (QK in bf16 via projection, RoPE bf16, softmax f32)
test_variant("D: QK=bf16, RoPE=bf16, std softmax (model-like)",
             torch.bfloat16, torch.bfloat16, lambda s, dim: torch.softmax(s, dim=dim))
print()

# E: QK in f32, RoPE in bf16 (model's RoPE precision, our QK precision)
test_variant("E: QK=f32, RoPE=bf16, phi_softmax",
             torch.float32, torch.bfloat16, phi_softmax)
print()


print("=" * 80)
print("  MARGIN ANALYSIS")
print("=" * 80)
print()
margins_sorted = sorted(enumerate(base_margins), key=lambda x: x[1])
print(f"  10 smallest margins:")
for idx, m in margins_sorted[:10]:
    print(f"    #{idx:2d}: margin={m:.3f}  \"{PROMPTS[idx][:50]}\"")
print()
print(f"  Stats: min={min(base_margins):.3f} median={np.median(base_margins):.3f} mean={np.mean(base_margins):.3f}")
print()

# Key question: does the model get the EXACT same output if we
# re-run it? (check for non-determinism)
print("=" * 80)
print("  NON-DETERMINISM CHECK")
print("=" * 80)
print()

n_nondeterministic = 0
for pi, p in enumerate(PROMPTS):
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out1 = model(ids, return_dict=True)
        out2 = model(ids, return_dict=True)
    l1 = out1.logits[0,-1,:].float()
    l2 = out2.logits[0,-1,:].float()
    if (l1 - l2).abs().max().item() > 0:
        n_nondeterministic += 1

print(f"  Non-deterministic prompts: {n_nondeterministic}/{len(PROMPTS)}")
print()

print("=" * 80)
print("  DONE")
print("=" * 80)
