#!/usr/bin/env python3
"""
Phase 9g2: RoPE comparison — our implementation vs model's internal RoPE.
Also: single-layer ablation to find which layer causes divergence.
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

# ================================================================
# Compare RoPE implementations
# ================================================================
print("=" * 80)
print("  ROPE COMPARISON")
print("=" * 80)
print()

# Our RoPE
def our_rope(seq_len, device, dtype):
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

# Model's RoPE — hook into the attention to capture cos/sin
# Let's look at how Qwen2 actually implements RoPE
print("  Inspecting model's attention implementation...")
attn0 = model.model.layers[0].self_attn
print(f"  Attention class: {type(attn0).__name__}")
print(f"  Has rotary_emb: {hasattr(attn0, 'rotary_emb')}")

# Check the model's rotary_emb on the model level
print(f"  Model has rotary_emb: {hasattr(model.model, 'rotary_emb')}")

# Check how position embeddings are computed
# In newer transformers, rotary_emb is on the model, not per-layer
if hasattr(model.model, 'rotary_emb'):
    rotary = model.model.rotary_emb
    print(f"  Rotary emb class: {type(rotary).__name__}")
    print(f"  Rotary emb attributes: {[a for a in dir(rotary) if not a.startswith('_')]}")

    # Get model's cos/sin
    pos_ids = torch.arange(10, device="cuda").unsqueeze(0)
    model_cos, model_sin = rotary(
        torch.zeros(1, 1, 10, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        position_ids=pos_ids
    )
    print(f"  Model cos shape: {model_cos.shape}")
    print(f"  Model sin shape: {model_sin.shape}")
else:
    print("  No rotary_emb on model.model — checking forward hooks")

print()

# Hook into the actual attention forward to capture cos/sin that get used
print("  Capturing actual RoPE values via forward hook on attention...")

# Patch the attention to capture the position_embeddings argument
rope_captures = {}

def capture_rope_hook(layer_idx):
    def hook_fn(module, args, kwargs):
        # The Qwen2Attention.forward signature includes position_embeddings
        if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
            cos, sin = kwargs['position_embeddings']
            rope_captures[layer_idx] = (cos.detach().clone(), sin.detach().clone())
    return hook_fn

hooks = []
for li in range(N_LAYERS):
    h = model.model.layers[li].self_attn.register_forward_pre_hook(
        capture_rope_hook(li), with_kwargs=True)
    hooks.append(h)

test_prompt = "The capital of Mexico is"
test_ids = tokenizer.encode(test_prompt, return_tensors="pt").to("cuda")
seq_len = test_ids.shape[1]

with torch.no_grad():
    model(test_ids)

for h in hooks:
    h.remove()

print(f"  Captured RoPE for {len(rope_captures)} layers, seq_len={seq_len}")

if rope_captures:
    # Compare with our implementation
    li = 0
    model_cos, model_sin = rope_captures[li]
    our_cos, our_sin = our_rope(seq_len, torch.device("cuda"), torch.float32)

    print(f"\n  Model cos shape: {model_cos.shape}")
    print(f"  Our cos shape:   {our_cos.shape}")
    print(f"  Model cos dtype: {model_cos.dtype}")

    # Reshape to compare (model may have batch dim, etc.)
    mc = model_cos.float().squeeze()
    oc = our_cos.float().squeeze()
    ms = model_sin.float().squeeze()
    os_ = our_sin.float().squeeze()

    print(f"  Model cos squeezed: {mc.shape}")
    print(f"  Our cos squeezed:   {oc.shape}")

    if mc.shape == oc.shape:
        cos_diff = (mc - oc).abs()
        sin_diff = (ms - os_).abs()
        print(f"\n  Cos max diff: {cos_diff.max().item():.2e}")
        print(f"  Cos mean diff: {cos_diff.mean().item():.2e}")
        print(f"  Sin max diff: {sin_diff.max().item():.2e}")
        print(f"  Sin mean diff: {sin_diff.mean().item():.2e}")

        if cos_diff.max().item() > 0.001:
            print("\n  *** SIGNIFICANT ROPE DIFFERENCE FOUND ***")
            # Find where they differ
            print(f"  Position with max cos diff: {cos_diff.max(dim=-1).values.argmax().item()}")
            print(f"  Dim with max cos diff: {cos_diff.max(dim=0).values.argmax().item()}")

            # Print first few values
            print(f"\n  First position, first 10 dims:")
            print(f"  Model: {mc[0, :10].tolist()}")
            print(f"  Ours:  {oc[0, :10].tolist()}")
        else:
            print("\n  → RoPE implementations MATCH (diff < 0.001)")
    else:
        print(f"\n  Shape mismatch! Trying to align...")
        # Try various reshapes
        if mc.dim() == 2 and oc.dim() == 2:
            min_s = min(mc.shape[0], oc.shape[0])
            min_d = min(mc.shape[1], oc.shape[1])
            diff = (mc[:min_s, :min_d] - oc[:min_s, :min_d]).abs().max().item()
            print(f"  Max diff (aligned): {diff:.2e}")

    # Check: are ALL layers' RoPE the same?
    if len(rope_captures) > 1:
        all_same = True
        for li in range(1, min(5, len(rope_captures))):
            c0, s0 = rope_captures[0]
            ci, si = rope_captures[li]
            if (c0 - ci).abs().max().item() > 1e-6:
                all_same = False
                print(f"  Layer {li} RoPE differs from Layer 0!")
        if all_same:
            print(f"  → All layers share the same RoPE (as expected)")

print()

# ================================================================
# Check how model applies RoPE internally
# ================================================================
print("=" * 80)
print("  MODEL'S ROPE APPLICATION")
print("=" * 80)
print()

# Look at the source code of the attention module
import inspect
try:
    src = inspect.getsource(type(attn0).forward)
    # Find the lines related to RoPE
    lines = src.split('\n')
    rope_lines = [l for l in lines if 'rotary' in l.lower() or 'rope' in l.lower() or 'cos' in l.lower() or 'sin' in l.lower() or 'position' in l.lower()]
    print("  RoPE-related lines in attention forward:")
    for l in rope_lines[:20]:
        print(f"    {l.strip()}")
except Exception as e:
    print(f"  Could not inspect: {e}")

print()

# Also check how the model applies rotary to Q and K
# Our version: apply_rotary(Q, cos, sin) where Q is (batch, heads, seq, head_dim)
# Model version might differ in how it interleaves real/imaginary parts

# The key question: does Qwen2 use the "rotate_half" convention or "complex" convention?
# rotate_half: [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]  (first half, second half)
# complex:     [x_0r, x_0i, x_1r, x_1i, ...] → complex rotation

# Check by looking at the actual Q/K before and after RoPE
print("  Checking RoPE application convention...")

# Capture Q before and after RoPE
pre_rope_q = {}
post_rope_q = {}

def capture_qk_hook(layer_idx):
    def hook_fn(module, args, kwargs, output):
        h = args[0] if args else kwargs.get('hidden_states')
        if h is None: return output
        b, s, _ = h.shape
        with torch.no_grad():
            q = module.q_proj(h).float()
        q = q.reshape(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        pre_rope_q[layer_idx] = q[0, 0, 0, :8].clone().cpu()  # head 0, pos 0, first 8 dims
    return hook_fn

h0 = model.model.layers[0].self_attn.register_forward_hook(
    capture_qk_hook(0), with_kwargs=True)

with torch.no_grad():
    out = model(test_ids, output_attentions=True, return_dict=True)

h0.remove()

# Now apply our RoPE to the same Q and see if it matches
if 0 in pre_rope_q:
    q_pre = pre_rope_q[0]
    print(f"  Q[0,0,0,:8] (pre-RoPE): {q_pre.tolist()}")

    # Our RoPE at position 0: cos=1, sin=0, so Q should be unchanged
    # At position 1: cos and sin are nontrivial
    # The real test is position > 0

print()

# ================================================================
# DEFINITIVE TEST: Replace attention at only L27 with phi_softmax
# If L27 alone causes the failures, the error source is there.
# ================================================================
print("=" * 80)
print("  SINGLE-LAYER ABLATION: Which layer causes the failures?")
print("=" * 80)
print()

def phi_softmax_torch(scores, dim=-1):
    s = scores - scores.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)

FAIL_PROMPTS = [
    ("The capital of Mexico is", None),
    ("The Great Wall of China is located in", None),
    ("In computer science, an algorithm is a step by step procedure for", None),
]

# Get baselines
for i, (p, _) in enumerate(FAIL_PROMPTS):
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    FAIL_PROMPTS[i] = (p, out.logits[0,-1,:].float().argmax().item())

def make_phi_hook(layer_idx):
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
        c, sn = our_rope(s, HEAD_DIM, h.device, torch.float32) if False else (None, None)
        # Use captured model RoPE if available
        if layer_idx in rope_captures:
            mc, ms = rope_captures[layer_idx]
            mc = mc.float(); ms = ms.float()
            # Need to reshape to match (batch, 1, seq, dim) for broadcasting
            if mc.dim() == 3:  # (batch, seq, dim)
                mc = mc.unsqueeze(1)  # (batch, 1, seq, dim)
                ms = ms.unsqueeze(1)
        else:
            inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device=h.device, dtype=torch.float32) / HEAD_DIM))
            pos = torch.arange(s, device=h.device, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            mc = emb.cos()[None, None]
            ms = emb.sin()[None, None]

        def apply_rope(x, cos, sin):
            x1 = x[..., :x.shape[-1]//2]
            x2 = x[..., x.shape[-1]//2:]
            return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

        Q = apply_rope(Q, mc, ms)
        K = apply_rope(K, mc, ms)
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

# Test: replace ONLY layer X
print(f"  Testing single-layer replacement on {len(FAIL_PROMPTS)} failing prompts...")
print(f"  {'Layer':>5s}  Correct/3  Note")
print("  " + "-" * 30)

for test_layer in range(N_LAYERS):
    nm = 0
    for p, base_id in FAIL_PROMPTS:
        ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
        hk = model.model.layers[test_layer].self_attn.register_forward_hook(
            make_phi_hook(test_layer), with_kwargs=True)
        with torch.no_grad():
            out = model(ids, return_dict=True)
        hk.remove()
        if out.logits[0,-1,:].float().argmax().item() == base_id:
            nm += 1
    flag = " ← causes failure" if nm < 3 else ""
    print(f"  L{test_layer:2d}    {nm}/3{flag}")

print()

# ================================================================
# Now test: use MODEL'S RoPE in our phi_softmax implementation
# ================================================================
print("=" * 80)
print("  FIX TEST: Use model's own RoPE in phi_softmax")
print("=" * 80)
print()

# Re-run full phi_softmax with model's captured RoPE on ALL 60 prompts
ALL_PROMPTS = [
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

# Get baselines for all
print("  Getting all baselines...")
all_base_ids = []
for p in ALL_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    all_base_ids.append(out.logits[0,-1,:].float().argmax().item())

# Need to capture model RoPE for each seq_len we'll encounter
# First collect unique seq lengths
seq_lens = set()
for p in ALL_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt")
    seq_lens.add(ids.shape[1])

print(f"  Unique sequence lengths: {sorted(seq_lens)}")

# Capture RoPE for each seq length
rope_by_seqlen = {}
for sl in seq_lens:
    # Run a dummy forward pass at this length to capture RoPE
    dummy_ids = tokenizer.encode(ALL_PROMPTS[0], return_tensors="pt").to("cuda")
    # Pad or truncate
    if dummy_ids.shape[1] < sl:
        dummy_ids = F.pad(dummy_ids, (0, sl - dummy_ids.shape[1]), value=0)
    else:
        dummy_ids = dummy_ids[:, :sl]

    captures = {}
    hooks = []
    for li in range(1):  # Only need layer 0 since all layers share RoPE
        h = model.model.layers[li].self_attn.register_forward_pre_hook(
            capture_rope_hook(li), with_kwargs=True)
        hooks.append(h)
    with torch.no_grad():
        model(dummy_ids)
    for h in hooks: h.remove()

    if 0 in captures:
        rope_by_seqlen[sl] = captures[0]
    elif 0 in rope_captures and rope_captures[0][0].shape[-2] >= sl:
        rope_by_seqlen[sl] = (rope_captures[0][0][:, :sl, :], rope_captures[0][1][:, :sl, :])

# Use global rope_captures from earlier if available
print(f"  Cached RoPE for seq lengths: {sorted(rope_by_seqlen.keys())}")

# If we didn't get all, use our own RoPE (which we need to verify first)
# Actually let's just capture the RoPE properly for each prompt
print()
print("  Full test with model RoPE (capturing per-prompt)...")

def run_with_model_rope(prompt):
    """Run phi_softmax attention using the model's own RoPE."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    s = ids.shape[1]

    # First pass: capture RoPE
    captures = {}
    hooks = []
    h = model.model.layers[0].self_attn.register_forward_pre_hook(
        capture_rope_hook(0), with_kwargs=True)
    hooks.append(h)
    with torch.no_grad():
        model(ids)
    for hk in hooks: hk.remove()

    if 0 not in captures:
        # Fallback: use our RoPE
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
        pos = torch.arange(s, device="cuda", dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        model_c = emb.cos()[None, None]
        model_s = emb.sin()[None, None]
    else:
        model_c, model_s = captures[0]
        model_c = model_c.float()
        model_s = model_s.float()
        if model_c.dim() == 3:
            model_c = model_c.unsqueeze(1)
            model_s = model_s.unsqueeze(1)

    # Second pass: replace attention with phi_softmax using model's RoPE
    def make_hook(li, mc, ms):
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

            def apply_rope(x, cos, sin):
                x1 = x[..., :x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
                return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

            Q = apply_rope(Q, mc, ms); K = apply_rope(K, mc, ms)
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out_t = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
            mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
                sc.masked_fill_(mask, float('-inf'))
                w = phi_softmax_torch(sc, dim=-1)
                out_t[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h.dtype)
            return (module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM)),) + output[1:] if isinstance(output, tuple) else module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM))
        return hook_fn

    attn_hooks = []
    for li in range(N_LAYERS):
        hk = model.model.layers[li].self_attn.register_forward_hook(
            make_hook(li, model_c, model_s), with_kwargs=True)
        attn_hooks.append(hk)
    with torch.no_grad():
        out = model(ids, return_dict=True)
    for hk in attn_hooks: hk.remove()
    return out.logits[0,-1,:].float()

# Quick test on just the 3 failing prompts first
print("  Quick test on failing prompts (model RoPE):")
for p, base_id in FAIL_PROMPTS:
    logits = run_with_model_rope(p)
    got = logits.argmax().item()
    match = "✓" if got == base_id else "✗"
    print(f"    {match} \"{p[:50]}\" → '{tokenizer.decode([got])}'")

print()
print("  Full 60-prompt test (model RoPE)...")
nm_model_rope = 0
for pi, p in enumerate(ALL_PROMPTS):
    logits = run_with_model_rope(p)
    if logits.argmax().item() == all_base_ids[pi]:
        nm_model_rope += 1

print(f"  With model's RoPE: {nm_model_rope}/{len(ALL_PROMPTS)} ({nm_model_rope/len(ALL_PROMPTS):.0%})")
print()

# Compare: our RoPE
print("  Full 60-prompt test (our RoPE)...")
def run_with_our_rope(prompt):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    s = ids.shape[1]
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(s, device="cuda", dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    mc = emb.cos()[None, None]
    ms = emb.sin()[None, None]

    def make_hook(li, mc, ms):
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
            def apply_rope(x, cos, sin):
                x1 = x[..., :x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
                return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
            Q = apply_rope(Q, mc, ms); K = apply_rope(K, mc, ms)
            Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
            out_t = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
            mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
            for hd in range(NUM_HEADS):
                sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
                sc.masked_fill_(mask, float('-inf'))
                w = phi_softmax_torch(sc, dim=-1)
                out_t[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h.dtype)
            return (module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM)),) + output[1:] if isinstance(output, tuple) else module.o_proj(out_t.reshape(b,s,NUM_HEADS*HEAD_DIM))
        return hook_fn

    attn_hooks = []
    for li in range(N_LAYERS):
        hk = model.model.layers[li].self_attn.register_forward_hook(
            make_hook(li, mc, ms), with_kwargs=True)
        attn_hooks.append(hk)
    with torch.no_grad():
        out = model(ids, return_dict=True)
    for hk in attn_hooks: hk.remove()
    return out.logits[0,-1,:].float()

nm_our_rope = 0
for pi, p in enumerate(ALL_PROMPTS):
    logits = run_with_our_rope(p)
    if logits.argmax().item() == all_base_ids[pi]:
        nm_our_rope += 1

print(f"  With our RoPE:    {nm_our_rope}/{len(ALL_PROMPTS)} ({nm_our_rope/len(ALL_PROMPTS):.0%})")
print()
print("=" * 80)
print("  DONE")
print("=" * 80)
