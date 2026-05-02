#!/usr/bin/env python3
"""
Non-Trivial Zero Hunting on the Phase Shift Control Surface
============================================================

DC 295 showed that per-dimension gate zeros lie in the null space —
they have zero semantic leverage. Those are the "trivial zeros."

This script hunts the NON-TRIVIAL zeros: the δ values where the
COLLECTIVE MLP output changes enough to flip the model's prediction.

These are the transformer analog of Riemann zeros — where all terms
in the sum conspire to produce a different answer.

Method (the three-stage pipeline, adapted):
  Stage 1 (Compressor): Coarse sweep of δ, map the logit gap landscape
  Stage 2 (Processor):  Bisect sign changes to high precision
  Stage 3 (Targeter):   Analyze semantics at each non-trivial zero

The logit gap function:
  f(δ) = logit[baseline_top1](δ) - max(logit[others])(δ)

When f(δ) > 0: baseline prediction still wins
When f(δ) = 0: NON-TRIVIAL ZERO — prediction is exactly balanced
When f(δ) < 0: prediction has flipped to something new
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
# Model loading (from phi_zero_hunt_semantic.py)
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
# ε-group mask
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

# ═══════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════

def get_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[0, -1].float()

def shift_and_eval(model, input_ids, layer_idx, mask_gpu, delta, top1_id):
    """Apply ε-group shift, run inference, return logit gap and top prediction.
    
    logit_gap = logit[baseline_top1] - max(logit[others])
    Positive = baseline still wins. Zero = non-trivial zero. Negative = flipped.
    """
    W = model.model.layers[layer_idx].mlp.gate_proj.weight.data
    scale = PHI ** delta
    W[mask_gpu] *= scale
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1].float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return float('nan'), -1, logits
        top1_logit = logits[top1_id].item()
        # Mask out the baseline top-1 to find best alternative
        logits_alt = logits.clone()
        logits_alt[top1_id] = float('-inf')
        alt_id = logits_alt.argmax().item()
        alt_logit = logits_alt[alt_id].item()
        gap = top1_logit - alt_logit
        return gap, alt_id, logits
    finally:
        W[mask_gpu] /= scale

# ═══════════════════════════════════════════════════════════════════
# The Three-Stage Pipeline
# ═══════════════════════════════════════════════════════════════════

def stage1_coarse_sweep(model, input_ids, layer_idx, mask_gpu, top1_id,
                        delta_range=(-5.0, 12.0), n_points=69):
    """Stage 1 (Compressor): Map the logit gap landscape."""
    deltas = np.linspace(delta_range[0], delta_range[1], n_points)
    gaps = []
    alt_ids = []
    valid = []
    
    for d in deltas:
        gap, alt_id, _ = shift_and_eval(model, input_ids, layer_idx, mask_gpu, d, top1_id)
        gaps.append(gap)
        alt_ids.append(alt_id)
        valid.append(not np.isnan(gap))
    
    return deltas, np.array(gaps), alt_ids, np.array(valid)


def stage2_bisect(model, input_ids, layer_idx, mask_gpu, top1_id,
                  delta_lo, delta_hi, n_iter=40):
    """Stage 2 (Processor): Bisect a sign change to high precision."""
    # Verify bracket
    gap_lo, _, _ = shift_and_eval(model, input_ids, layer_idx, mask_gpu, delta_lo, top1_id)
    gap_hi, _, _ = shift_and_eval(model, input_ids, layer_idx, mask_gpu, delta_hi, top1_id)
    
    if np.isnan(gap_lo) or np.isnan(gap_hi):
        return None
    if np.sign(gap_lo) == np.sign(gap_hi):
        return None
    
    history = []
    for i in range(n_iter):
        mid = (delta_lo + delta_hi) / 2.0
        gap_mid, alt_id, logits = shift_and_eval(model, input_ids, layer_idx, mask_gpu, mid, top1_id)
        
        if np.isnan(gap_mid):
            delta_hi = mid
            continue
        
        history.append((mid, gap_mid, alt_id))
        
        if gap_mid > 0:
            delta_lo = mid
        else:
            delta_hi = mid
    
    delta_star = (delta_lo + delta_hi) / 2.0
    return {
        'delta': delta_star,
        'precision': delta_hi - delta_lo,
        'history': history,
    }


def stage3_analyze(model, input_ids, layer_idx, mask_gpu, top1_id,
                   delta_star, tokenizer):
    """Stage 3 (Targeter): Full analysis at the non-trivial zero."""
    # Evaluate at δ* and δ* ± small offsets
    offsets = [-0.01, -0.001, 0.0, 0.001, 0.01]
    results = []
    
    for off in offsets:
        d = delta_star + off
        gap, alt_id, logits = shift_and_eval(model, input_ids, layer_idx, mask_gpu, d, top1_id)
        
        if np.isnan(gap):
            results.append({'delta': d, 'gap': float('nan'), 'top5': [], 'nan': True})
            continue
        
        top5_vals, top5_ids = logits.topk(5)
        top5 = [(tokenizer.decode([tid]).strip(), tv) 
                for tid, tv in zip(top5_ids.tolist(), top5_vals.tolist())]
        
        pred_id = logits.argmax().item()
        pred_tok = tokenizer.decode([pred_id]).strip()
        results.append({
            'delta': d, 'gap': gap, 'pred': pred_tok, 'pred_id': pred_id,
            'top5': top5, 'nan': False,
        })
    
    # Also get baseline logits for comparison
    gap_base, _, logits_base = shift_and_eval(model, input_ids, layer_idx, mask_gpu, 0.0, top1_id)
    top5_base_vals, top5_base_ids = logits_base.topk(5)
    top5_base = [(tokenizer.decode([tid]).strip(), tv)
                 for tid, tv in zip(top5_base_ids.tolist(), top5_base_vals.tolist())]
    
    return results, top5_base


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  NON-TRIVIAL ZERO HUNTING — THE COLLECTIVE ZEROS")
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
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", None),
        ("Albert Einstein developed the theory of", "rel"),
    ]
    TARGET_LAYERS = [5, 15, 22, 23, 27]

    # Compute ε-group masks
    print(f"\n  Phase 2: ε-group masks for layers {TARGET_LAYERS}...")
    sys.stdout.flush()
    layer_masks = {}
    for li in TARGET_LAYERS:
        t0 = time.time()
        mask_np, top_eps = compute_eps_mask(li)
        mask_gpu = torch.from_numpy(mask_np).to('cuda')
        layer_masks[li] = {'mask_gpu': mask_gpu, 'top_eps': top_eps,
                           'n_elems': int(np.sum(mask_np))}
        print(f"    L{li}: ε={top_eps}, {layer_masks[li]['n_elems']:,d} elems [{time.time()-t0:.1f}s]",
              flush=True)
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # Hunt non-trivial zeros
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Phase 3: Hunting non-trivial zeros")
    print("=" * 70)
    sys.stdout.flush()

    all_zeros = []

    for prompt_text, expected in PROMPTS:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to('cuda')
        logits_base = get_logits(model, input_ids)
        top1_id = logits_base.argmax().item()
        top1_tok = tokenizer.decode([top1_id]).strip()

        sorted_vals, sorted_ids = logits_base.sort(descending=True)
        top2_id = sorted_ids[1].item()
        top2_tok = tokenizer.decode([top2_id]).strip()
        baseline_gap = (sorted_vals[0] - sorted_vals[1]).item()

        print(f"\n  PROMPT: \"{prompt_text}\"")
        print(f"  Baseline: \"{top1_tok}\" (gap to \"{top2_tok}\" = {baseline_gap:.3f})")
        sys.stdout.flush()

        for li in TARGET_LAYERS:
            lm = layer_masks[li]
            print(f"\n    Layer {li} (ε={lm['top_eps']}, {lm['n_elems']:,d} elems):")
            sys.stdout.flush()

            # Stage 1: Coarse sweep
            t0 = time.time()
            deltas, gaps, alt_ids, valid = stage1_coarse_sweep(
                model, input_ids, li, lm['mask_gpu'], top1_id,
                delta_range=(-5.0, 12.0), n_points=69)
            t1 = time.time()

            # Print coarse landscape
            print(f"      Stage 1 (Compressor): {len(deltas)} points in {t1-t0:.1f}s")
            print(f"      δ range: [{deltas[0]:.1f}, {deltas[-1]:.1f}]")

            # Find valid range (before NaN)
            nan_start = None
            for i, v in enumerate(valid):
                if not v:
                    nan_start = i
                    break
            if nan_start is not None:
                print(f"      NaN boundary at δ ≈ {deltas[nan_start]:.2f} (φ^δ = {PHI**deltas[nan_start]:.1f}×)")

            # Find sign changes in gap
            sign_changes = []
            for i in range(len(gaps) - 1):
                if not valid[i] or not valid[i+1]:
                    continue
                if gaps[i] > 0 and gaps[i+1] <= 0:
                    sign_changes.append((i, 'pos→neg'))
                elif gaps[i] <= 0 and gaps[i+1] > 0:
                    sign_changes.append((i, 'neg→pos'))

            if not sign_changes:
                # Print the gap profile for diagnosis
                print(f"      No sign changes found. Gap profile:")
                # Show key points
                for idx in [0, len(gaps)//4, len(gaps)//2, 3*len(gaps)//4, len(gaps)-1]:
                    if idx < len(gaps) and valid[idx]:
                        tok = tokenizer.decode([alt_ids[idx]]).strip() if alt_ids[idx] >= 0 else "?"
                        print(f"        δ={deltas[idx]:>6.2f}: gap={gaps[idx]:>8.3f}, "
                              f"alt=\"{tok}\"")
                if nan_start and nan_start > 0:
                    print(f"        (last valid δ={deltas[nan_start-1]:.2f}: "
                          f"gap={gaps[nan_start-1]:.3f})")
                continue

            print(f"      Found {len(sign_changes)} sign change(s)!")
            sys.stdout.flush()

            # Stage 2: Bisect each sign change
            for sc_idx, (i, direction) in enumerate(sign_changes):
                d_lo, d_hi = float(deltas[i]), float(deltas[i+1])
                print(f"\n      Sign change #{sc_idx+1} ({direction}) "
                      f"in [{d_lo:.3f}, {d_hi:.3f}]")
                sys.stdout.flush()

                t0 = time.time()
                result = stage2_bisect(model, input_ids, li, lm['mask_gpu'],
                                       top1_id, d_lo, d_hi, n_iter=40)
                t1 = time.time()

                if result is None:
                    print(f"      Bisection failed (NaN or lost bracket)")
                    continue

                delta_star = result['delta']
                precision = result['precision']
                phi_delta = PHI ** delta_star

                print(f"      Stage 2 (Processor): δ* = {delta_star:.10f} "
                      f"(±{precision:.2e}) in {t1-t0:.1f}s")
                print(f"      φ^δ* = {phi_delta:.6f}× "
                      f"(scaling ε-group by {phi_delta:.4f})")
                sys.stdout.flush()

                # Stage 3: Analyze
                analysis, top5_base = stage3_analyze(
                    model, input_ids, li, lm['mask_gpu'],
                    top1_id, delta_star, tokenizer)

                print(f"\n      Stage 3 (Targeter): Semantics at δ*")
                print(f"      Baseline top-5: {top5_base}")

                for a in analysis:
                    if a['nan']:
                        print(f"        δ={a['delta']:.6f}: NaN")
                        continue
                    marker = "◀ ZERO" if abs(a['delta'] - delta_star) < 0.0001 else ""
                    print(f"        δ={a['delta']:.6f}: gap={a['gap']:>+8.4f}  "
                          f"pred=\"{a['pred']}\"  top5={[t[0] for t in a['top5']]}  {marker}")

                # What does the model switch TO?
                # Look at the first point past the zero
                post_zero = [a for a in analysis if not a['nan'] and a['gap'] <= 0]
                if post_zero:
                    new_pred = post_zero[0]['pred']
                    print(f"\n      ★ NON-TRIVIAL ZERO: \"{top1_tok}\" → \"{new_pred}\" "
                          f"at δ = {delta_star:.8f}")
                    print(f"        φ^δ = {phi_delta:.6f}")
                    print(f"        This is the collective cancellation point —")
                    print(f"        where {lm['n_elems']:,d} weight elements conspire")
                    print(f"        to flip the prediction.")

                all_zeros.append({
                    'prompt': prompt_text, 'layer': li,
                    'delta': delta_star, 'phi_delta': phi_delta,
                    'from': top1_tok,
                    'to': post_zero[0]['pred'] if post_zero else '?',
                    'precision': precision,
                    'baseline_gap': baseline_gap,
                })

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY: ALL NON-TRIVIAL ZEROS")
    print(f"{'='*70}")

    if not all_zeros:
        print("  No non-trivial zeros found in the scanned range.")
    else:
        print(f"\n  {'Prompt':<45s} {'Layer':>5s} {'δ*':>12s} {'φ^δ*':>8s} "
              f"{'From':>8s} {'To':>8s} {'Precision':>10s}")
        print(f"  {'-'*45} {'-'*5} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for z in all_zeros:
            print(f"  {z['prompt']:<45s} {z['layer']:>5d} {z['delta']:>12.8f} "
                  f"{z['phi_delta']:>8.4f} {z['from']:>8s} {z['to']:>8s} "
                  f"{z['precision']:>10.2e}")

        # Analysis: is δ* consistent across prompts for the same layer?
        print(f"\n  Cross-prompt analysis:")
        for li in TARGET_LAYERS:
            layer_zeros = [z for z in all_zeros if z['layer'] == li]
            if len(layer_zeros) >= 2:
                deltas = [z['delta'] for z in layer_zeros]
                print(f"    L{li}: {len(layer_zeros)} zeros, "
                      f"δ range [{min(deltas):.4f}, {max(deltas):.4f}], "
                      f"spread = {max(deltas)-min(deltas):.4f}")
                # Are they content-dependent (like trivial zeros)?
                if max(deltas) - min(deltas) > 0.5:
                    print(f"          → Content-dependent (spread > 0.5)")
                else:
                    print(f"          → Structurally consistent (spread < 0.5)")
            elif len(layer_zeros) == 1:
                print(f"    L{li}: 1 zero at δ={layer_zeros[0]['delta']:.4f}")
            else:
                print(f"    L{li}: no zeros found")

        # The Riemann analogy
        print(f"\n  Riemann analogy:")
        print(f"    Trivial zeros (DC 295): per-dimension, closed-form, null leverage")
        print(f"    Non-trivial zeros (this): collective, found by sweep+bisect, semantic flip")
        if all_zeros:
            mean_delta = np.mean([z['delta'] for z in all_zeros])
            print(f"    Mean δ* = {mean_delta:.4f} (φ^δ* = {PHI**mean_delta:.4f}×)")
            mean_gap = np.mean([z['baseline_gap'] for z in all_zeros])
            print(f"    Mean baseline gap = {mean_gap:.3f}")
            print(f"    The non-trivial zeros live in the EXPLOSIVE regime (δ >> 0.1)")
            print(f"    — exactly where individual gate dim flips accumulate enough")
            print(f"    collective perturbation to overcome the baseline gap.")

    print(f"\n  Done.")


if __name__ == '__main__':
    run()
