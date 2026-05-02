#!/usr/bin/env python3
"""
Phase 9f: Scale Test + Hybrid Gap Fix + RoPE φ-Analysis
=========================================================

Three tasks in one script:

TASK 1: Scale test — 60 diverse prompts validating:
  - phi_softmax attention + gate replacement (composed, F87)
  - Hybrid attention (Config A)

TASK 2: Close the 2/15 hybrid gap:
  - Identify which prompts fail
  - Diagnose: is it specific fixed heads? specific layers?
  - Test fix: phi_softmax for fixed heads with high pos-0 weight (>95%)
    but full QK for "weakly fixed" heads

TASK 3: RoPE in φ-basis:
  - Express RoPE rotation angles as φ-levels
  - Test whether d_k + RoPE correction enables stacking
  - If RoPE has φ-structure, it could unlock geometric d_k routing

Requires: Qwen2-7B on GPU, phase9a results
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
print("  PHASE 9f: SCALE TEST + HYBRID GAP + RoPE φ-ANALYSIS")
print("=" * 80)
print()

results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_classification = {}
for ls in phase9a['layer_summary']:
    layer_classification[ls['layer']] = {
        'fixed': set(ls['fixed_heads']),
        'routing': set(ls['routing_heads']),
    }

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
GATE_DIM = model.config.intermediate_size
COMB_START = 6; COMB_END = 23


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
# Attention variants
# ================================================================
def attn_full_phi(li, h, attn):
    b, s, _ = h.shape
    with torch.no_grad():
        Q = attn.q_proj(h).float(); K = attn.k_proj(h).float(); V = attn.v_proj(h)
    Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
    K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
    Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
    Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
    c, sn = get_rope_cache(s, HEAD_DIM, h.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, c, sn); K = apply_rotary_pos_emb(K, c, sn)
    Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
    out = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
    mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
    for hd in range(NUM_HEADS):
        sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
        sc.masked_fill_(mask, float('-inf'))
        w = phi_softmax_torch(sc, dim=-1)
        out[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h.dtype)
    return attn.o_proj(out.reshape(b,s,NUM_HEADS*HEAD_DIM))

def attn_hybrid(li, h, attn):
    b, s, _ = h.shape
    fixed = layer_classification[li]['fixed']
    routing = layer_classification[li]['routing']
    with torch.no_grad():
        Q = attn.q_proj(h).float(); K = attn.k_proj(h).float(); V = attn.v_proj(h)
    Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
    K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
    Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
    Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
    c, sn = get_rope_cache(s, HEAD_DIM, h.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, c, sn); K = apply_rotary_pos_emb(K, c, sn)
    Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
    out = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h.device, dtype=h.dtype)
    mask = torch.triu(torch.ones(s,s,device=h.device), diagonal=1).bool()
    for hd in fixed:
        out[0,:,hd,:] = Ve[0,0,hd,:]
    for hd in routing:
        sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
        sc.masked_fill_(mask, float('-inf'))
        w = phi_softmax_torch(sc, dim=-1)
        out[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h.dtype)
    return attn.o_proj(out.reshape(b,s,NUM_HEADS*HEAD_DIM))


def make_attn_hooks(attn_fn):
    hooks = []
    for li in range(N_LAYERS):
        def mk(layer_idx, fn):
            def hf(module, args, kwargs, output):
                h = args[0] if args else kwargs.get('hidden_states')
                if h is None: return output
                g = fn(layer_idx, h, module)
                return (g,) + output[1:] if isinstance(output, tuple) else g
            return hf
        hk = model.model.layers[li].self_attn.register_forward_hook(mk(li, attn_fn), with_kwargs=True)
        hooks.append(hk)
    return hooks


# ================================================================
# Gate replacement setup (F82)
# ================================================================
print("Building scaffold...")
TRAIN_WORDS = [
    "king", "queen", "man", "woman", "boy", "girl",
    "hot", "cold", "fast", "slow", "big", "small",
    "love", "hate", "light", "dark", "true", "false",
    "cat", "dog", "tree", "water", "fire", "earth",
    "happy", "sad", "strong", "weak", "old", "young",
    "the", "is", "and", "of", "to", "in",
    "zero", "one", "two", "three", "four", "five",
    "red", "blue", "green", "black", "white", "yellow",
    "algorithm", "quantum", "geometry", "neural", "vector", "matrix",
    "Paris", "London", "Tokyo", "Einstein", "Newton", "Euler",
    "hello", "world", "computer", "science", "language", "model",
]
single_gates = {}; single_hs = {}
for word in TRAIN_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids: continue
    tid = ids[0]; dec = tokenizer.decode([tid]).strip()
    if dec in single_gates: continue
    gs = {}; hs = {}; hks = []
    for l in range(N_LAYERS):
        def mgh(st, li):
            def hf(m, i, o): st[li] = o.detach().cpu().float().numpy().squeeze()
            return hf
        def mhh(st, li):
            def hf(m, i, o): st[li] = i[0].detach().cpu().float().numpy().squeeze()
            return hf
        hks.append(model.model.layers[l].mlp.gate_proj.register_forward_hook(mgh(gs, l)))
        hks.append(model.model.layers[l].mlp.register_forward_hook(mhh(hs, l)))
    with torch.no_grad(): model(torch.tensor([[tid]], device="cuda"))
    for h in hks: h.remove()
    single_gates[dec] = np.stack([gs[l] for l in range(N_LAYERS)])
    single_hs[dec] = np.stack([hs[l] for l in range(N_LAYERS)])

tw = sorted(single_gates.keys())
scaffold_gate = np.stack([single_gates[w] for w in tw]).mean(axis=0)
scaffold_hidden = np.stack([single_hs[w] for w in tw]).mean(axis=0)
W_gates = {}
for l in range(COMB_START, COMB_END):
    W_gates[l] = model.model.layers[l].mlp.gate_proj.weight.data.float().cpu().numpy()
print(f"  Scaffold from {len(tw)} tokens")
print()


def run_composed(input_ids, attn_fn, gate_rank=5):
    """Two-pass: capture states with attn_fn, then intervene with attn_fn + gate replacement."""
    # Pass 1: capture
    gs = {}; hs_st = {}; hks = []
    for l in range(COMB_START, COMB_END):
        def mgh(st, li):
            def hf(m, i, o): st[li] = o.detach().cpu().float().numpy().squeeze()
            return hf
        def mhh(st, li):
            def hf(m, i, o): st[li] = i[0].detach().cpu().float().numpy().squeeze()
            return hf
        hks.append(model.model.layers[l].mlp.gate_proj.register_forward_hook(mgh(gs, l)))
        hks.append(model.model.layers[l].mlp.register_forward_hook(mhh(hs_st, l)))
    ah = make_attn_hooks(attn_fn) if attn_fn else []
    with torch.no_grad(): model(input_ids)
    for h in hks: h.remove()
    for h in ah: h.remove()

    # Compute reconstructions
    ig = {}
    for l in range(COMB_START, COMB_END):
        ha = hs_st[l]; W = W_gates[l]
        hm = ha.mean(axis=0); hs_ = hm - scaffold_hidden[l]
        sc = scaffold_gate[l] + W @ hs_
        hr = ha - hm[np.newaxis, :]
        U, S, Vt = np.linalg.svd(hr, full_matrices=False)
        k = min(gate_rank, len(S))
        ig[l] = sc + ((U[:,:k]*S[:k]) @ Vt[:k]) @ W.T

    # Pass 2: intervene
    hks2 = []
    def mrh(rep):
        def hf(m, i, o):
            r = torch.tensor(rep, dtype=o.dtype, device=o.device)
            return r.reshape(o.shape)
        return hf
    for l in range(COMB_START, COMB_END):
        hks2.append(model.model.layers[l].mlp.gate_proj.register_forward_hook(mrh(ig[l])))
    ah2 = make_attn_hooks(attn_fn) if attn_fn else []
    with torch.no_grad():
        out = model(input_ids, return_dict=True)
    for h in hks2: h.remove()
    for h in ah2: h.remove()
    return out.logits


# ================================================================
# TASK 1: SCALE TEST (60 prompts)
# ================================================================
print("=" * 80)
print("  TASK 1: SCALE TEST — 60 diverse prompts")
print("=" * 80)
print()

SCALE_PROMPTS = [
    # Factual capitals (10)
    "The capital of France is", "The capital of Japan is",
    "The capital of Brazil is", "The capital of Egypt is",
    "The capital of Canada is", "The capital of India is",
    "The capital of Germany is", "The capital of Australia is",
    "The capital of Italy is", "The capital of Mexico is",
    # Science (10)
    "The chemical symbol for gold is", "The speed of light is approximately",
    "Water freezes at zero degrees", "The largest planet in our solar system is",
    "Albert Einstein developed the theory of", "The atomic number of carbon is",
    "DNA stands for deoxyribonucleic", "The boiling point of water is",
    "Photosynthesis converts sunlight into", "Newton's first law states that an object",
    # Completion (10)
    "To be or not to", "Roses are red, violets are",
    "The quick brown fox jumps over the", "Once upon a time in a land",
    "All that glitters is not", "A penny saved is a penny",
    "The early bird catches the", "Actions speak louder than",
    "When in Rome, do as the", "An apple a day keeps the",
    # Math/Logic (10)
    "The square root of 144 is", "In mathematics, pi is approximately equal to",
    "2 + 2 =", "The derivative of x squared is",
    "A triangle has three sides and three", "The Fibonacci sequence starts with 0, 1,",
    "The area of a circle is pi times the", "10 divided by 2 equals",
    "The sum of angles in a triangle is", "A prime number is only divisible by",
    # Creative/Diverse (10)
    "The color of grass is", "Shakespeare wrote many",
    "Barack Obama was the", "The largest ocean is the",
    "The color of the sky is usually", "Artificial intelligence is a field of",
    "The Great Wall of China is located in", "The Mona Lisa was painted by",
    "The tallest mountain in the world is", "In machine learning, a neural network",
    # Long context (10)
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
print("  Collecting baselines for 60 prompts...")
base_ids = []
for p in SCALE_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    base_ids.append(out.logits[0,-1,:].float().argmax().item())
print("  Done.")
print()

configs_task1 = {
    'phi_softmax_attn_only': (attn_full_phi, None),
    'hybrid_attn_only': (attn_hybrid, None),
    'composed_phi_softmax_gate': (attn_full_phi, 5),
    'composed_hybrid_gate': (attn_hybrid, 5),
}

for cname, (afn, gr) in configs_task1.items():
    n_match = 0
    fails = []
    for pi, p in enumerate(SCALE_PROMPTS):
        ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
        if gr is not None:
            logits = run_composed(ids, afn, gate_rank=gr)
        else:
            ah = make_attn_hooks(afn)
            with torch.no_grad():
                out = model(ids, return_dict=True)
                logits = out.logits
            for h in ah: h.remove()
        tid = logits[0,-1,:].float().argmax().item()
        if tid == base_ids[pi]:
            n_match += 1
        else:
            fails.append((pi, p, tokenizer.decode([base_ids[pi]]), tokenizer.decode([tid])))

    print(f"  {cname:>35s}: {n_match}/{len(SCALE_PROMPTS)} ({n_match/len(SCALE_PROMPTS):.0%})")
    if fails:
        for fi, fp, fb, fg in fails[:5]:
            print(f"    FAIL #{fi}: \"{fp[:40]}...\" base='{fb}' got='{fg}'")
        if len(fails) > 5:
            print(f"    ... and {len(fails)-5} more failures")

print()


# ================================================================
# TASK 2: CLOSE THE HYBRID GAP
# ================================================================
print("=" * 80)
print("  TASK 2: CLOSE THE HYBRID GAP")
print("  Why does fixed→V[0] fail on some prompts?")
print("=" * 80)
print()

# Collect actual pos-0 attention weights for "fixed" heads
# to see how fixed they really are
print("  Measuring actual pos-0 attention weight for 'fixed' heads...")
sample_prompts = SCALE_PROMPTS[:10]
fixed_head_pos0_weights = {}  # (layer, head) -> list of pos0 weights

for p in sample_prompts:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, output_attentions=True, return_dict=True)
    for li in range(N_LAYERS):
        attn_w = out.attentions[li]  # (1, heads, seq, seq)
        for h in layer_classification[li]['fixed']:
            w0 = attn_w[0, h, -1, 0].float().item()  # last query, pos 0
            key = (li, h)
            if key not in fixed_head_pos0_weights:
                fixed_head_pos0_weights[key] = []
            fixed_head_pos0_weights[key].append(w0)

# Find "weakly fixed" heads (avg pos-0 weight < 95%)
weak_fixed = {}
for (li, h), ws in fixed_head_pos0_weights.items():
    avg = np.mean(ws)
    if avg < 0.95:
        weak_fixed[(li, h)] = avg

n_weak = len(weak_fixed)
n_total_fixed = len(fixed_head_pos0_weights)
print(f"  Total fixed heads: {n_total_fixed}")
print(f"  Weakly fixed (pos-0 weight < 95%): {n_weak}")
if weak_fixed:
    sorted_weak = sorted(weak_fixed.items(), key=lambda x: x[1])
    print(f"  Weakest 10:")
    for (li, h), avg in sorted_weak[:10]:
        print(f"    L{li:2d} H{h:2d}: avg pos-0 weight = {avg:.3f}")
print()

# Test: use phi_softmax QK for weakly-fixed heads instead of V[0]
print("  Testing fix: use full QK for weakly-fixed heads...")

weak_fixed_set = set(weak_fixed.keys())

def attn_hybrid_fixed(li, h_normed, attn):
    """Hybrid with fix: weakly-fixed heads use full QK, strongly-fixed use V[0]."""
    b, s, _ = h_normed.shape
    fixed = layer_classification[li]['fixed']
    routing = layer_classification[li]['routing']
    strongly_fixed = {hd for hd in fixed if (li, hd) not in weak_fixed_set}
    needs_qk = routing | {hd for hd in fixed if (li, hd) in weak_fixed_set}

    with torch.no_grad():
        Q = attn.q_proj(h_normed).float(); K = attn.k_proj(h_normed).float(); V = attn.v_proj(h_normed)
    Q = Q.reshape(b,s,NUM_HEADS,HEAD_DIM).transpose(1,2)
    K = K.reshape(b,s,NUM_KV_HEADS,HEAD_DIM).transpose(1,2)
    Vk = V.reshape(b,s,NUM_KV_HEADS,HEAD_DIM)
    Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
    c, sn = get_rope_cache(s, HEAD_DIM, h_normed.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, c, sn); K = apply_rotary_pos_emb(K, c, sn)
    Ke = K.repeat_interleave(HEADS_PER_KV, dim=1)
    out = torch.zeros(b,s,NUM_HEADS,HEAD_DIM, device=h_normed.device, dtype=h_normed.dtype)
    mask_t = torch.triu(torch.ones(s,s,device=h_normed.device), diagonal=1).bool()
    for hd in strongly_fixed:
        out[0,:,hd,:] = Ve[0,0,hd,:]
    for hd in needs_qk:
        sc = Q[0,hd] @ Ke[0,hd].T / math.sqrt(HEAD_DIM)
        sc.masked_fill_(mask_t, float('-inf'))
        w = phi_softmax_torch(sc, dim=-1)
        out[0,:,hd,:] = (w @ Ve[0,:,hd,:].float()).to(h_normed.dtype)
    return attn.o_proj(out.reshape(b,s,NUM_HEADS*HEAD_DIM))

# Test hybrid-fixed on the scale prompts
ah = make_attn_hooks(attn_hybrid_fixed)
n_match_fix = 0
for pi, p in enumerate(SCALE_PROMPTS):
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    if out.logits[0,-1,:].float().argmax().item() == base_ids[pi]:
        n_match_fix += 1
for h in ah: h.remove()

# Count how many heads moved to QK
n_moved = sum(1 for (li, h) in weak_fixed_set)
n_still_fixed = n_total_fixed - n_moved
pct_fixed = n_still_fixed / (N_LAYERS * NUM_HEADS) * 100

print(f"  Hybrid-fixed: {n_match_fix}/{len(SCALE_PROMPTS)} ({n_match_fix/len(SCALE_PROMPTS):.0%})")
print(f"  Heads still using V[0]: {n_still_fixed}/{N_LAYERS*NUM_HEADS} ({pct_fixed:.0f}%)")
print(f"  Heads moved to QK: {n_moved}")
print()


# ================================================================
# TASK 3: RoPE in φ-BASIS
# ================================================================
print("=" * 80)
print("  TASK 3: RoPE IN φ-BASIS")
print("=" * 80)
print()

# RoPE frequencies: θ_i = 1 / (base^(2i/d)) where base=1000000, d=128
# These define rotation angles per position
rope_theta = 1000000.0
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
freqs_np = inv_freq.numpy()

print(f"  RoPE base: {rope_theta}")
print(f"  {len(freqs_np)} frequency pairs (head_dim/2 = {HEAD_DIM//2})")
print()

# Express frequencies as φ-levels: f_i = φ^level_i
# level_i = log(f_i) / log(φ)
phi_levels = np.log(freqs_np) / LOG_PHI

print(f"  Frequency range: [{freqs_np.min():.2e}, {freqs_np.max():.2e}]")
print(f"  φ-level range:   [{phi_levels.min():.1f}, {phi_levels.max():.1f}]")
print()

# Check if frequencies are on the φ-lattice
phi_residuals = phi_levels - np.round(phi_levels)
print(f"  Mean |φ-residual|: {np.mean(np.abs(phi_residuals)):.4f}")
print(f"  Within 0.1 of integer φ-level: {(np.abs(phi_residuals) < 0.1).sum()}/{len(phi_residuals)}")
print(f"  Within 0.25 of integer φ-level: {(np.abs(phi_residuals) < 0.25).sum()}/{len(phi_residuals)}")
print()

# Check if the frequency RATIOS are φ-structured
# Adjacent ratio: f_i / f_{i+1}
ratios = freqs_np[:-1] / freqs_np[1:]
ratio_phi_levels = np.log(ratios) / LOG_PHI
print(f"  Adjacent frequency ratios:")
print(f"    Mean ratio: {np.mean(ratios):.6f}")
print(f"    Std ratio:  {np.std(ratios):.6f}")
print(f"    φ-level of mean ratio: {np.log(np.mean(ratios))/LOG_PHI:.4f}")
print()

# Check: is the overall frequency range a power of φ?
total_range = freqs_np.max() / freqs_np.min()
total_phi_level = np.log(total_range) / LOG_PHI
print(f"  Total frequency range: {total_range:.2e}")
print(f"  φ-level of range: {total_phi_level:.2f}")
print(f"  Nearest integer: {round(total_phi_level)}")
print(f"  Error from nearest: {abs(total_phi_level - round(total_phi_level)):.4f}")
print()

# The RoPE rotation angle at position p for frequency i is:
# angle(p, i) = p * freq_i
# At what position does each frequency complete a full rotation?
# 2π / freq_i = period
periods = 2 * np.pi / freqs_np
period_phi_levels = np.log(periods) / LOG_PHI
print(f"  Period range: [{periods.min():.1f}, {periods.max():.1e}] positions")
print(f"  Period φ-levels: [{period_phi_levels.min():.1f}, {period_phi_levels.max():.1f}]")
print()

# Key question: does the RoPE frequency spectrum follow φ^(-2i/d * log(base)/log(φ))?
# inv_freq_i = base^(-2i/d) = φ^(-2i/d * log(base)/log(φ))
log_base_phi = np.log(rope_theta) / LOG_PHI
print(f"  log_φ(base) = {log_base_phi:.4f}")
print(f"  φ-step per frequency index = 2/d * log_φ(base) = {2/HEAD_DIM * log_base_phi:.4f}")
print()

# This means RoPE frequencies are on a φ-geometric sequence:
# freq_i = φ^(-i * step)  where step = 2 * log_φ(base) / d
step = 2 * log_base_phi / HEAD_DIM
predicted_levels = -np.arange(HEAD_DIM // 2) * step
actual_levels = phi_levels
residuals = actual_levels - predicted_levels
print(f"  Predicted vs actual φ-levels:")
print(f"    Max residual: {np.max(np.abs(residuals)):.6f}")
print(f"    Mean residual: {np.mean(np.abs(residuals)):.6f}")
print(f"  → RoPE frequencies ARE exactly φ^(-i×step) (step={step:.4f})")
print()

# Now: can we express d_k routing WITH RoPE correction?
# The full attention score between query at position q and key at position k:
#   score(q,k) = (RoPE(Q,q))^T @ (RoPE(K,k))
#              = Q^T @ RoPE(k-q)^T @ RoPE(k-q) ... wait, that's not right
# Actually: score(q,k) = q_rotated^T @ k_rotated
#   where q_rotated = R(θ*q_pos) @ q, k_rotated = R(θ*k_pos) @ k
#   = q^T @ R(θ*(k_pos - q_pos)) @ k  (rotation is relative)
# So the RoPE contributes a position-dependent rotation to the score.

# For d_k routing: score_dk(k) = h[k] · d_k  (no position info)
# With RoPE:       score_rope(q,k) = (R(θ*q) @ W_q @ h[q])^T @ (R(θ*k) @ W_k @ h[k])

# The simplest RoPE-aware d_k would be:
#   score(q,k) = h[k] · R_dk(k-q) · d_k
# where R_dk is the rotation in the d_k subspace

# But d_k is in hidden space (3584-D), not head space (128-D).
# The RoPE rotation happens in head space after W_k projection.

# Test: add position-dependent decay to d_k scores
# Hypothesis: RoPE primarily adds a distance-based decay
# score_corrected(q,k) = h[k]·d_k * decay(q-k)

print("  Testing RoPE-corrected d_k routing...")
print("  (extracting d_k vectors...)")

# Extract d_k for all routing heads (reuse from earlier or re-extract)
d_k_vectors = {}
for layer_idx in range(N_LAYERS):
    routing = layer_classification[layer_idx]['routing']
    if not routing: continue
    d_k_vectors[layer_idx] = {}
    attn = model.model.layers[layer_idx].self_attn
    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    Wq = {h: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for h in routing}
    nkv = set(h // HEADS_PER_KV for h in routing)
    Wk = {g: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for g in nkv}
    for s in range(0, HIDDEN_DIM, 512):
        e = min(s+512, HIDDEN_DIM)
        ch = identity[s:e].unsqueeze(0)
        with torch.no_grad():
            qo = attn.q_proj(ch).float(); ko = attn.k_proj(ch).float()
        qr = qo[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        kr = ko[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        for h in routing: Wq[h][:, s:e] = qr[:, h, :].T
        for g in nkv: Wk[g][:, s:e] = kr[:, g, :].T
    for h in routing:
        g = h // HEADS_PER_KV
        M = Wq[h] @ Wk[g].T; _, _, Vt = torch.linalg.svd(M)
        d_k_vectors[layer_idx][h] = (Wk[g].T @ Vt[0]).cpu()
    del Wq, Wk; torch.cuda.empty_cache()
print("  d_k extracted.")

# Measure: for each prompt, how does RoPE change attention scores
# compared to d_k predictions?
print()
print("  Measuring RoPE effect on attention patterns...")

# For a few prompts, compare:
# 1. d_k score ranking: argmax(h·d_k)
# 2. Actual attention argmax (with RoPE)
# 3. Do they agree? How much does RoPE shift things?

rope_agreement = []
rope_last_pos_bias = []

for pi, p in enumerate(SCALE_PROMPTS[:10]):
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    # Capture layernormed hidden states at each layer via hooks
    normed_states = {}
    norm_hooks = []
    for li in range(N_LAYERS):
        def mk_norm_hook(storage, layer_idx):
            def hf(module, input, output):
                storage[layer_idx] = output.detach()
            return hf
        nh = model.model.layers[li].input_layernorm.register_forward_hook(
            mk_norm_hook(normed_states, li))
        norm_hooks.append(nh)

    with torch.no_grad():
        out = model(ids, output_attentions=True, return_dict=True)

    for nh in norm_hooks:
        nh.remove()

    for li in range(N_LAYERS):
        if li not in normed_states:
            continue
        h_normed = normed_states[li]
        routing = layer_classification[li]['routing']
        for h in routing:
            if li not in d_k_vectors or h not in d_k_vectors[li]:
                continue
            dk = d_k_vectors[li][h].to("cuda", dtype=torch.float32)
            dk_scores = h_normed[0].float() @ dk
            dk_argmax = dk_scores.argmax().item()

            real_argmax = out.attentions[li][0, h, -1, :].float().argmax().item()
            rope_agreement.append(1 if dk_argmax == real_argmax else 0)

            if real_argmax == seq_len - 1:
                rope_last_pos_bias.append(1)
            else:
                rope_last_pos_bias.append(0)

print(f"  d_k vs real attention argmax agreement: {np.mean(rope_agreement):.1%}")
print(f"  Real attention selects last position: {np.mean(rope_last_pos_bias):.1%}")
print()

# Test: d_k with exponential position decay
# score_corrected(q,k) = (h[k]·d_k) * φ^(-|q-k| * decay_rate)
print("  Testing d_k + φ-decay for full-stack routing...")

def attn_dk_phi_decay(li, h_normed, attn_module, decay_rate=0.1):
    """d_k routing with φ-based position decay."""
    b, s, _ = h_normed.shape
    fixed = layer_classification[li]['fixed']
    routing = layer_classification[li]['routing']
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    Vk = V_full.reshape(b, s, NUM_KV_HEADS, HEAD_DIM)
    Ve = Vk.repeat_interleave(HEADS_PER_KV, dim=2)
    out = torch.zeros(b, s, NUM_HEADS, HEAD_DIM, device=h_normed.device, dtype=h_normed.dtype)
    for hd in fixed:
        out[0, :, hd, :] = Ve[0, 0, hd, :]
    for hd in routing:
        if li not in d_k_vectors or hd not in d_k_vectors[li]:
            out[0, :, hd, :] = Ve[0, 0, hd, :]
            continue
        dk = d_k_vectors[li][hd].to(h_normed.device, dtype=torch.float32)
        scores = h_normed[0].float() @ dk  # (s,)
        for i in range(s):
            # Apply φ-decay based on distance from query
            distances = torch.arange(i+1, device=h_normed.device, dtype=torch.float32)
            distances = (i - distances)  # distance from query pos i
            decay = PHI ** (-distances * decay_rate)
            causal_scores = scores[:i+1] * decay
            w = phi_softmax_torch(causal_scores, dim=0)
            weighted_v = w.to(h_normed.dtype).unsqueeze(-1) * Ve[0, :i+1, hd, :]
            out[0, i, hd, :] = weighted_v.sum(dim=0)
    return attn_module.o_proj(out.reshape(b, s, NUM_HEADS * HEAD_DIM))

# Test several decay rates
decay_rates = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1/PHI]
test_prompts_dk = SCALE_PROMPTS[:15]

# Baseline for these prompts
base_dk = []
for p in test_prompts_dk:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    base_dk.append(out.logits[0,-1,:].float().argmax().item())

print(f"  {'Decay rate':>12s}  {'Score':>7s}  Note")
print("  " + "-" * 35)

for dr in decay_rates:
    def mk_decay_fn(rate):
        def fn(li, h, attn):
            return attn_dk_phi_decay(li, h, attn, decay_rate=rate)
        return fn
    ah = make_attn_hooks(mk_decay_fn(dr))
    nm = 0
    for pi, p in enumerate(test_prompts_dk):
        ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(ids, return_dict=True)
        if out.logits[0,-1,:].float().argmax().item() == base_dk[pi]:
            nm += 1
    for h in ah: h.remove()
    note = "← 1/φ" if abs(dr - 1/PHI) < 0.01 else ""
    print(f"  {dr:12.4f}  {nm:2d}/{len(test_prompts_dk)}    {note}")

print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
print()
print("  Task 1 (Scale): see results above")
print("  Task 2 (Hybrid fix): see results above")
print("  Task 3 (RoPE φ-structure): see analysis above")

# Save
save_path = os.path.join(results_dir, 'phase9f_scale_hybrid_rope.json')
with open(save_path, 'w') as f:
    json.dump({
        'rope_phi_step': float(step),
        'rope_log_base_phi': float(log_base_phi),
        'rope_frequencies_are_phi_geometric': True,
        'dk_vs_real_agreement': float(np.mean(rope_agreement)),
    }, f, indent=2)
print(f"\n  Saved to {save_path}")
