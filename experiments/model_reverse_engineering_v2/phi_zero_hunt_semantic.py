#!/usr/bin/env python3
"""
Zero-Hunting on Real Model Hidden States — Semantic Impact Test

Does precision gate-flipping via ε-group phase shifts produce
semantically meaningful output changes?

Pipeline:
  1. Load φ-encoded model on GPU
  2. For target layers, compute ε-group masks from φ-encoded weights
  3. Run baseline inference on knowledge prompts
  4. Compute zero spectrum for REAL hidden states at each MLP layer
  5. Apply ε-group phase shifts that flip exactly N gate dimensions
  6. Re-run inference, measure semantic impact on output
"""

import os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
LN_PHI = np.log(PHI)

# ═══════════════════════════════════════════════════════════════════
# Model loading (from phi_gpu_inference.py)
# ═══════════════════════════════════════════════════════════════════

def decode_phi_to_tensor(path):
    d = np.load(path)
    signs = d['signs'].astype(np.float32)
    exponents = d['exponents'].astype(np.float32)
    values = signs * (np.float32(PHI) ** (exponents / np.float32(PHI_GRID)))
    return torch.from_numpy(values).half()

def build_state_dict():
    state_dict = {}
    print("  Converting embed_tokens + lm_head...", flush=True)
    state_dict['model.embed_tokens.weight'] = decode_phi_to_tensor(
        os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    state_dict['lm_head.weight'] = decode_phi_to_tensor(
        os.path.join(MODEL_DIR, 'lm_head.npz'))
    fn = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))
    state_dict['model.norm.weight'] = torch.from_numpy(fn['weight'].astype(np.float32)).half()
    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        prefix = f'model.layers.{layer_idx}'
        t0 = time.time()
        print(f"  Layer {layer_idx:2d}...", end='', flush=True)
        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        state_dict[f'{prefix}.input_layernorm.weight'] = torch.from_numpy(
            norms['input_layernorm'].astype(np.float32)).half()
        state_dict[f'{prefix}.post_attention_layernorm.weight'] = torch.from_numpy(
            norms['post_attention_layernorm'].astype(np.float32)).half()
        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        for bp in ['q_proj', 'k_proj', 'v_proj']:
            state_dict[f'{prefix}.self_attn.{bp}.bias'] = torch.from_numpy(
                biases[f'{bp}_bias'].astype(np.float32)).half()
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            state_dict[f'{prefix}.self_attn.{proj}.weight'] = decode_phi_to_tensor(
                os.path.join(layer_dir, f'{proj}.npz'))
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            state_dict[f'{prefix}.mlp.{proj}.weight'] = decode_phi_to_tensor(
                os.path.join(layer_dir, f'{proj}.npz'))
        print(f" ({time.time()-t0:.1f}s)", flush=True)
        gc.collect()
    return state_dict

def load_model(state_dict):
    from transformers import AutoConfig, Qwen2ForCausalLM
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
    config.torch_dtype = torch.float16
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device='cuda', dtype=torch.float16)
    gc.collect()
    with torch.device('meta'):
        model = Qwen2ForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=False)
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

# ═══════════════════════════════════════════════════════════════════
# ε-group mask + zero spectrum
# ═══════════════════════════════════════════════════════════════════

def levels(W): return W.exponents.astype(np.int32) // PHI_GRID

def compute_eps_mask(layer_idx):
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    W = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
    lvl = levels(W).astype(np.float32)
    U, s, Vt = np.linalg.svd(lvl, full_matrices=False)
    lvl_r1 = np.round(np.outer(U[:, 0] * s[0], Vt[0, :])).astype(np.int32)
    eps_int = lvl.astype(np.int32) - lvl_r1
    W.clear_cache(); del W
    unique, counts = np.unique(eps_int, return_counts=True)
    top_eps = int(unique[np.argmax(counts)])
    return (eps_int == top_eps), top_eps

def compute_zero_spectrum(h_post_norm, gate_W_cpu, eps_mask_cpu):
    h = (gate_W_cpu @ h_post_norm).flatten()
    W_masked = gate_W_cpu * eps_mask_cpu
    c = (W_masked @ h_post_norm).flatten()
    valid = np.abs(c) > 1e-15
    ratio = np.full(len(h), np.nan)
    ratio[valid] = 1.0 - h[valid] / c[valid]
    has_zero = valid & (ratio > 0)
    delta = np.full(len(h), np.nan)
    delta[has_zero] = np.log(ratio[has_zero]) / LN_PHI
    zero_dims = np.where(has_zero)[0]
    zero_deltas = delta[has_zero]
    order = np.argsort(np.abs(zero_deltas))
    return zero_dims[order], zero_deltas[order], h

# ═══════════════════════════════════════════════════════════════════
# Intervention helpers
# ═══════════════════════════════════════════════════════════════════

def get_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[0, -1].float().cpu()

def get_shifted_logits(model, input_ids, layer_idx, mask_gpu, delta):
    W = model.model.layers[layer_idx].mlp.gate_proj.weight.data
    W[mask_gpu] *= PHI ** delta
    with torch.no_grad():
        out = model(input_ids)
    W[mask_gpu] /= PHI ** delta
    return out.logits[0, -1].float().cpu()

def capture_mlp_input(model, input_ids, layer_idx):
    captured = {}
    def hook_fn(module, args):
        h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
        captured['h'] = h[0, -1, :].detach().float().cpu().numpy()
    hook = model.model.layers[layer_idx].mlp.register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    hook.remove()
    return captured['h']

def compare(logits_base, logits_mod, tokenizer):
    cos = F.cosine_similarity(logits_base.unsqueeze(0), logits_mod.unsqueeze(0)).item()
    t1b = logits_base.argmax().item()
    t1m = logits_mod.argmax().item()
    top5b = set(logits_base.topk(5).indices.tolist())
    top5m = set(logits_mod.topk(5).indices.tolist())
    p_b = F.softmax(logits_base, dim=0)
    p_m = F.softmax(logits_mod, dim=0)
    kl = F.kl_div(p_m.log(), p_b, reduction='sum').item()
    return {
        'cos': cos, 'match': t1b == t1m,
        'top5': len(top5b & top5m) / 5, 'kl': kl,
        'tok_base': tokenizer.decode([t1b]).strip(),
        'tok_mod': tokenizer.decode([t1m]).strip(),
    }

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  ZERO-HUNTING SEMANTIC IMPACT TEST")
    print("=" * 70)
    sys.stdout.flush()

    # Load model
    print("\n  Phase 1: Loading model on GPU...")
    sys.stdout.flush()
    sd = build_state_dict()
    model = load_model(sd)
    del sd; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

    PROMPTS = [
        "The capital of France is",
        "The capital of Japan is",
        "Water freezes at a temperature of",
        "The largest planet in our solar system is",
        "Albert Einstein developed the theory of",
    ]
    TARGET_LAYERS = [0, 5, 15, 22, 23, 27]
    N_FLIPS = [1, 5, 10, 50, 100, 500]

    # Compute ε-group masks
    print(f"\n  Phase 2: ε-group masks for layers {TARGET_LAYERS}...")
    sys.stdout.flush()
    layer_data = {}
    for li in TARGET_LAYERS:
        t0 = time.time()
        mask_np, top_eps = compute_eps_mask(li)
        gate_W_cpu = model.model.layers[li].mlp.gate_proj.weight.data.float().cpu().numpy().astype(np.float64)
        mask_gpu = torch.from_numpy(mask_np).to('cuda')
        layer_data[li] = {'mask_np': mask_np, 'mask_gpu': mask_gpu,
                          'top_eps': top_eps, 'gate_W': gate_W_cpu}
        print(f"    L{li}: ε={top_eps}, {np.sum(mask_np):,d} elems [{time.time()-t0:.1f}s]", flush=True)
        gc.collect()

    # Run experiments
    print(f"\n  Phase 3: Semantic impact")
    print("=" * 70)
    sys.stdout.flush()

    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        logits_base = get_logits(model, input_ids)
        tok_base = tokenizer.decode([logits_base.argmax().item()]).strip()
        print(f"\n  PROMPT: \"{prompt}\" → \"{tok_base}\"")
        print(f"  {'Layer':>5s} {'N':>5s} {'δ':>10s} {'cos':>6s} {'top1':>5s} "
              f"{'top5':>5s} {'KL':>8s} {'Prediction':>12s} {'#zeros':>7s}")
        sys.stdout.flush()

        for li in TARGET_LAYERS:
            ld = layer_data[li]
            h_real = capture_mlp_input(model, input_ids, li).astype(np.float64)
            z_dims, z_deltas, h_gate = compute_zero_spectrum(h_real, ld['gate_W'], ld['mask_np'])

            for nf in N_FLIPS:
                if nf > len(z_dims):
                    continue
                d = float(np.abs(z_deltas[nf - 1]))
                signs = np.sign(z_deltas[:nf])
                d_dir = d if np.sum(signs > 0) >= np.sum(signs < 0) else -d

                logits_mod = get_shifted_logits(model, input_ids, li, ld['mask_gpu'], d_dir)
                r = compare(logits_base, logits_mod, tokenizer)
                m = "✓" if r['match'] else "✗"
                pred = f"\"{r['tok_mod']}\"" if not r['match'] else ""
                print(f"  {li:>5d} {nf:>5d} {d_dir:>10.6f} {r['cos']:>6.4f} "
                      f"{m:>5s} {r['top5']:>5.0%} {r['kl']:>8.4f} "
                      f"{pred:>12s} {len(z_dims):>7d}")
                sys.stdout.flush()

    # Detailed sweep on France/L22-23
    print(f"\n{'='*70}")
    print(f"  DETAILED: France @ L22, L23")
    print(f"{'='*70}")
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    logits_base = get_logits(model, input_ids)

    for li in [22, 23]:
        ld = layer_data[li]
        h_real = capture_mlp_input(model, input_ids, li).astype(np.float64)
        z_dims, z_deltas, h_gate = compute_zero_spectrum(h_real, ld['gate_W'], ld['mask_np'])
        mag = np.abs(h_gate); mx = np.max(mag)
        p1 = np.sum((h_gate > 0) & (mag > 0.1*mx))
        p0 = np.sum((h_gate > 0) & (mag <= 0.1*mx))
        m0 = np.sum((h_gate <= 0) & (mag <= 0.1*mx))
        m1 = np.sum((h_gate <= 0) & (mag > 0.1*mx))
        print(f"\n  L{li}: {len(z_dims)} zeros, states +1={p1} +0={p0} -0={m0} -1={m1}")
        print(f"  h range [{h_gate.min():.3f}, {h_gate.max():.3f}]")
        print(f"  {'N':>5s} {'δ':>10s} {'cos':>6s} {'top1':>8s} {'KL':>8s}")

        for nf in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
            if nf > len(z_dims): break
            d = float(np.abs(z_deltas[nf-1]))
            signs = np.sign(z_deltas[:nf])
            d_dir = d if np.sum(signs > 0) >= np.sum(signs < 0) else -d
            logits_mod = get_shifted_logits(model, input_ids, li, ld['mask_gpu'], d_dir)
            r = compare(logits_base, logits_mod, tokenizer)
            m = "✓" if r['match'] else f"✗→{r['tok_mod']}"
            print(f"  {nf:>5d} {d_dir:>10.6f} {r['cos']:>6.4f} {m:>8s} {r['kl']:>8.4f}")
            sys.stdout.flush()

    print(f"\n  Done.")

if __name__ == '__main__':
    run()
