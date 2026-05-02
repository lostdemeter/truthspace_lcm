#!/usr/bin/env python3
"""
Phase 10q: Compound Machine Hypothesis — Doc 262
Test whether the LLM is 3 separate machines (Compressor/Processor/Targeter)
operating in different gate-state media.

Tests:
  1. Independent linearization: approx each machine separately vs global
  2. Interface dimensionality: effective rank at machine boundaries
  3. Transfer function per machine: perturbation response
  4. Gate medium verification: machine params by gate state
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
from collections import defaultdict
PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10q: COMPOUND MACHINE HYPOTHESIS (Doc 262)")
print("="*80)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

# Machine boundaries
COMPRESSOR = list(range(0, 4))    # L0-3: DRUM
PROCESSOR  = list(range(4, 26))   # L4-25: COMB
TARGETER   = list(range(26, 28))  # L26-27: MUSIC

def machine_of(li):
    if li < 4: return "COMPRESSOR"
    if li < 26: return "PROCESSOR"
    return "TARGETER"

# Load layer classifications
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_cls = {}
for ls in phase9a['layer_summary']:
    layer_cls[ls['layer']] = {'fixed': set(ls['fixed_heads']), 'routing': set(ls['routing_heads'])}

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
NL=28; NH=28; NKV=4; HD=128; HPK=7; HDIM=3584; MAXS=64; ROPE_THETA=1e6

def phi_softmax(s, dim=-1):
    s = s - s.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI); return p / p.sum(dim=dim, keepdim=True)

def rope_cache(slen, dev, dt):
    inv = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,device=dev,dtype=torch.float32)/HD))
    pos = torch.arange(slen, device=dev, dtype=torch.float32)
    f = torch.outer(pos, inv); e = torch.cat((f,f), dim=-1)
    return e.cos().to(dt)[None,None], e.sin().to(dt)[None,None]

def apply_rope(x, cos, sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return x*cos + torch.cat((-x2, x1), dim=-1)*sin

inv_freq = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,dtype=torch.float32)/HD))

def rope_rotate_vector(v, delta, inv_freq):
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    v1, v2 = v[:len(v)//2], v[len(v)//2:]
    return v * cos_d + torch.cat((-v2, v1)) * sin_d

def rope_rotate_matrix_cols(M, delta, inv_freq):
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    M1, M2 = M[:HD//2, :], M[HD//2:, :]
    return M * cos_d.unsqueeze(1) + torch.cat((-M2, M1), dim=0) * sin_d.unsqueeze(1)

# Build bias-aware tables
exec(open(os.path.join(os.path.dirname(__file__), 'phase10p_build_tables.py')).read())

# Attention functions
def attn_real_qk(layer_idx, h_normed, attn_module):
    batch, seq_len, _ = h_normed.shape
    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).to(torch.bfloat16)
        K = attn_module.k_proj(h_normed).to(torch.bfloat16)
        V_full = attn_module.v_proj(h_normed)
    Q = Q.reshape(batch, seq_len, NH, HD).transpose(1, 2)
    K = K.reshape(batch, seq_len, NKV, HD).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NKV, HD)
    V_exp = V_kv.repeat_interleave(HPK, dim=2)
    cos, sin = rope_cache(seq_len, h_normed.device, torch.bfloat16)
    Q = apply_rope(Q, cos, sin); K = apply_rope(K, cos, sin)
    K_exp = K.repeat_interleave(HPK, dim=1)
    attn_out = torch.zeros(batch, seq_len, NH, HD, device=h_normed.device, dtype=h_normed.dtype)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
    for hd in range(NH):
        sc = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HD)).float()
        sc.masked_fill_(mask, float('-inf'))
        w = phi_softmax(sc, dim=-1)
        attn_out[0, :, hd, :] = (w.to(torch.bfloat16) @ V_exp[0, :, hd, :]).to(h_normed.dtype)
    combined = attn_out.reshape(batch, seq_len, NH * HD)
    with torch.no_grad(): return attn_module.o_proj(combined)

def attn_bias_aware(layer_idx, h_normed, attn_module):
    batch, seq_len, _ = h_normed.shape
    fixed = layer_cls[layer_idx]['fixed']; routing = layer_cls[layer_idx]['routing']
    with torch.no_grad(): V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NKV, HD)
    V_exp = V_kv.repeat_interleave(HPK, dim=2)
    attn_out = torch.zeros(batch, seq_len, NH, HD, device=h_normed.device, dtype=h_normed.dtype)
    for h in fixed: attn_out[0, :, h, :] = V_exp[0, 0, h, :]
    h_float = h_normed[0].float().cpu()
    for h in routing:
        tbl = head_tables[(layer_idx, h)]
        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                d = i - j
                scores[i, j] = (tbl['baseline'][d].item() + (h_float[i] @ tbl['c_q'][d]).item()
                    + (tbl['c_k'][d] @ h_float[j]).item())
        scores = scores.to(h_normed.device)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)
    combined = attn_out.reshape(batch, seq_len, NH * HD)
    with torch.no_grad(): return attn_module.o_proj(combined)

# Hook-based forward with fine-grained capture
def run_with_capture(input_ids, attn_fn_map):
    hooks = []; layer_inputs = {}; attn_outputs = {}; layer_outputs = {}
    for li in range(NL):
        def make_pre(idx):
            def hk(mod, args):
                h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                layer_inputs[idx] = h[0, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].register_forward_pre_hook(make_pre(li)))
    for li in range(NL):
        fn = attn_fn_map.get(li, attn_real_qk)
        def make_attn(idx, f):
            def hk(mod, args, kwargs, output):
                h = args[0] if args else kwargs.get('hidden_states')
                if h is None: return output
                geo = f(idx, h, mod)
                attn_outputs[idx] = geo[0, -1, :].detach().float().cpu()
                return (geo,) + output[1:] if isinstance(output, tuple) else geo
            return hk
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(
            make_attn(li, fn), with_kwargs=True))
    for li in range(NL):
        def make_post(idx):
            def hk(mod, args, output):
                h_out = output[0] if isinstance(output, tuple) else output
                layer_outputs[idx] = h_out[0, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].register_forward_hook(make_post(li)))
    try:
        with torch.no_grad(): out = model(input_ids, return_dict=True)
        logits = out.logits
    finally:
        for hk in hooks: hk.remove()
    return logits, layer_inputs, attn_outputs, layer_outputs

# Load analysis
exec(open(os.path.join(os.path.dirname(__file__), 'phase10q_analysis.py')).read())
