#!/usr/bin/env python3
"""
Critical Line Hunt — Does a Universal Constraint Govern Non-Trivial Zeros?
==========================================================================

DC 296 found 21 non-trivial zeros of the transformer. The Riemann
Hypothesis says all non-trivial ζ zeros lie on Re(s) = 1/2.

Is there an analogous constraint for the transformer?

The experiment:
  1. 15 prompts with varying baseline gaps (0.1 to 5.0+)
  2. Dense sweep (150 points) at 3 key layers (L15, L22, L27)
  3. Find ALL non-trivial zeros per prompt/layer
  4. Test normalizations for universality:
     a. Raw δ* — do zeros cluster?
     b. φ^δ* / baseline_gap — does leverage ratio collapse?
     c. δ* vs log(gap) — linear critical line?
     d. Zero spacing — regular like Riemann zeros?
  5. Try 2nd ε-group to see if different groups share zeros

If there's a critical line, the zeros from different prompts should
collapse onto a single curve when properly normalized.
"""

import os, sys, time, gc
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
LN_PHI = np.log(PHI)

# ═══════════════════════════════════════════════════════════════════
# Model loading (shared with phi_collective_zero_hunt.py)
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
# ε-group masks (supports multiple groups)
# ═══════════════════════════════════════════════════════════════════

def levels(W): return W.exponents.astype(np.int32) // PHI_GRID

def compute_eps_masks(layer_idx, n_groups=2):
    """Compute masks for top-N ε-groups."""
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    W = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
    lvl = levels(W).astype(np.float32)
    U, s, Vt = np.linalg.svd(lvl, full_matrices=False)
    lvl_r1 = np.round(np.outer(U[:, 0] * s[0], Vt[0, :])).astype(np.int32)
    eps_int = lvl.astype(np.int32) - lvl_r1
    W.clear_cache(); del W
    unique, counts = np.unique(eps_int, return_counts=True)
    order = np.argsort(-counts)
    masks = []
    for i in range(min(n_groups, len(order))):
        eidx = order[i]
        eps_val = int(unique[eidx])
        mask = (eps_int == eps_val)
        masks.append((mask, eps_val, int(counts[eidx])))
    return masks

# ═══════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════

def get_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[0, -1].float()

def shift_and_eval(model, input_ids, layer_idx, mask_gpu, delta, top1_id):
    W = model.model.layers[layer_idx].mlp.gate_proj.weight.data
    scale = PHI ** delta
    W[mask_gpu] *= scale
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1].float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return float('nan'), -1
        top1_logit = logits[top1_id].item()
        logits_alt = logits.clone()
        logits_alt[top1_id] = float('-inf')
        alt_id = logits_alt.argmax().item()
        alt_logit = logits_alt[alt_id].item()
        gap = top1_logit - alt_logit
        return gap, alt_id
    finally:
        W[mask_gpu] /= scale

def find_zeros(model, input_ids, layer_idx, mask_gpu, top1_id,
               delta_range=(-3.0, 12.0), n_points=151):
    """Dense sweep + bisection for all zeros in range."""
    deltas = np.linspace(delta_range[0], delta_range[1], n_points)
    gaps = np.full(n_points, np.nan)
    
    for i, d in enumerate(deltas):
        gap, _ = shift_and_eval(model, input_ids, layer_idx, mask_gpu, d, top1_id)
        gaps[i] = gap
    
    # Find sign changes
    zeros = []
    for i in range(len(gaps) - 1):
        if np.isnan(gaps[i]) or np.isnan(gaps[i+1]):
            continue
        if gaps[i] * gaps[i+1] < 0:
            # Bisect
            lo, hi = float(deltas[i]), float(deltas[i+1])
            for _ in range(40):
                mid = (lo + hi) / 2.0
                g, alt = shift_and_eval(model, input_ids, layer_idx, mask_gpu, mid, top1_id)
                if np.isnan(g):
                    hi = mid
                    continue
                if g > 0:
                    lo = mid
                else:
                    hi = mid
            delta_star = (lo + hi) / 2.0
            zeros.append(delta_star)
    
    # Find NaN boundary
    nan_delta = None
    for i, g in enumerate(gaps):
        if np.isnan(g):
            nan_delta = float(deltas[i])
            break
    
    return zeros, deltas, gaps, nan_delta

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  CRITICAL LINE HUNT — UNIVERSAL CONSTRAINT ON NON-TRIVIAL ZEROS?")
    print("=" * 70)
    sys.stdout.flush()

    # Load model
    print("\n  Phase 1: Loading model...")
    sys.stdout.flush()
    sd = build_state_dict()
    model = load_model(sd)
    del sd; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

    # 15 prompts spanning different knowledge types and gap sizes
    PROMPTS = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "Albert Einstein developed the theory of",
        "The largest planet in our solar system is",
        "Water freezes at a temperature of",
        "The speed of light is approximately",
        "The chemical symbol for gold is",
        "Oxygen has an atomic number of",
        "The Great Wall of China is located in",
        "Shakespeare wrote the play",
        "The first president of the United States was",
        "The square root of 144 is",
    ]
    TARGET_LAYERS = [15, 22, 27]

    # Compute ε-group masks (top 2 groups per layer)
    print(f"\n  Phase 2: ε-group masks for layers {TARGET_LAYERS}...")
    sys.stdout.flush()
    layer_masks = {}
    for li in TARGET_LAYERS:
        t0 = time.time()
        masks = compute_eps_masks(li, n_groups=2)
        gpu_masks = []
        for mask_np, eps_val, count in masks:
            gpu_masks.append({
                'mask_gpu': torch.from_numpy(mask_np).to('cuda'),
                'eps': eps_val, 'count': count,
            })
        layer_masks[li] = gpu_masks
        info = ", ".join([f"ε={m['eps']}({m['count']:,d})" for m in gpu_masks])
        print(f"    L{li}: {info} [{time.time()-t0:.1f}s]", flush=True)
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Hunt zeros with top ε-group
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Phase 3: Dense sweep + zero finding (top ε-group)")
    print("=" * 70)
    sys.stdout.flush()

    # Collect all data
    all_data = []  # list of dicts

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        logits_base = get_logits(model, input_ids)
        top1_id = logits_base.argmax().item()
        top1_tok = tokenizer.decode([top1_id]).strip()

        sorted_vals, sorted_ids = logits_base.sort(descending=True)
        top2_id = sorted_ids[1].item()
        top2_tok = tokenizer.decode([top2_id]).strip()
        baseline_gap = (sorted_vals[0] - sorted_vals[1]).item()

        print(f"\n  [{pi+1:2d}/15] \"{prompt}\"")
        print(f"    → \"{top1_tok}\" (gap={baseline_gap:.3f} to \"{top2_tok}\")")
        sys.stdout.flush()

        for li in TARGET_LAYERS:
            mask_info = layer_masks[li][0]  # top ε-group
            t0 = time.time()
            zeros, deltas, gaps, nan_delta = find_zeros(
                model, input_ids, li, mask_info['mask_gpu'], top1_id,
                delta_range=(-3.0, 12.0), n_points=151)
            elapsed = time.time() - t0

            # Record
            for z in zeros:
                all_data.append({
                    'prompt_idx': pi, 'prompt': prompt,
                    'layer': li, 'eps_group': 0,
                    'top1': top1_tok, 'gap': baseline_gap,
                    'delta_star': z, 'phi_delta': PHI ** z,
                    'nan_boundary': nan_delta,
                })

            n_z = len(zeros)
            z_str = ", ".join([f"{z:.3f}" for z in zeros]) if zeros else "none"
            print(f"    L{li}: {n_z} zero(s) at δ*=[{z_str}] ({elapsed:.1f}s)"
                  + (f" NaN@{nan_delta:.1f}" if nan_delta else ""))
            sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Test 2nd ε-group on a few prompts
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Phase 4: 2nd ε-group comparison (first 5 prompts)")
    print("=" * 70)
    sys.stdout.flush()

    for pi in range(5):
        prompt = PROMPTS[pi]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        logits_base = get_logits(model, input_ids)
        top1_id = logits_base.argmax().item()
        top1_tok = tokenizer.decode([top1_id]).strip()
        sorted_vals, _ = logits_base.sort(descending=True)
        baseline_gap = (sorted_vals[0] - sorted_vals[1]).item()

        for li in TARGET_LAYERS:
            if len(layer_masks[li]) < 2:
                continue
            mask_info = layer_masks[li][1]  # 2nd ε-group
            zeros, _, _, nan_delta = find_zeros(
                model, input_ids, li, mask_info['mask_gpu'], top1_id,
                delta_range=(-3.0, 12.0), n_points=101)

            for z in zeros:
                all_data.append({
                    'prompt_idx': pi, 'prompt': prompt,
                    'layer': li, 'eps_group': 1,
                    'top1': top1_tok, 'gap': baseline_gap,
                    'delta_star': z, 'phi_delta': PHI ** z,
                    'nan_boundary': nan_delta,
                })

            z_str = ", ".join([f"{z:.3f}" for z in zeros]) if zeros else "none"
            print(f"    [{pi+1}] L{li} ε-grp1: {len(zeros)} zero(s) at δ*=[{z_str}]")
        sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Analysis — search for the critical line
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  ANALYSIS: SEARCHING FOR THE CRITICAL LINE")
    print(f"{'='*70}")

    if not all_data:
        print("  No zeros found!")
        return

    # Filter to top ε-group only for main analysis
    grp0 = [d for d in all_data if d['eps_group'] == 0]
    grp1 = [d for d in all_data if d['eps_group'] == 1]

    print(f"\n  Total zeros: {len(all_data)} ({len(grp0)} grp0, {len(grp1)} grp1)")

    # A. Raw δ* statistics per layer
    print(f"\n  A. Raw δ* distribution (top ε-group):")
    print(f"    {'Layer':>5s} {'N':>3s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'median':>8s}")
    for li in TARGET_LAYERS:
        layer_zeros = [d['delta_star'] for d in grp0 if d['layer'] == li]
        if layer_zeros:
            arr = np.array(layer_zeros)
            print(f"    L{li:>3d} {len(arr):>3d} {arr.mean():>8.3f} {arr.std():>8.3f} "
                  f"{arr.min():>8.3f} {arr.max():>8.3f} {np.median(arr):>8.3f}")

    # B. Leverage ratio φ^δ* / gap
    print(f"\n  B. Leverage ratio φ^δ* / baseline_gap:")
    print(f"    {'Layer':>5s} {'N':>3s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'CV':>8s}")
    for li in TARGET_LAYERS:
        ratios = [d['phi_delta'] / d['gap'] for d in grp0
                  if d['layer'] == li and d['gap'] > 0.01]
        if ratios:
            arr = np.array(ratios)
            cv = arr.std() / arr.mean() if arr.mean() > 0 else float('inf')
            print(f"    L{li:>3d} {len(arr):>3d} {arr.mean():>8.2f} {arr.std():>8.2f} "
                  f"{arr.min():>8.2f} {arr.max():>8.2f} {cv:>8.3f}")

    # C. Log relationship: δ* vs ln(gap)
    print(f"\n  C. Linear regression: δ* = a × ln(gap) + b")
    for li in TARGET_LAYERS:
        pts = [(d['delta_star'], np.log(d['gap']))
               for d in grp0 if d['layer'] == li and d['gap'] > 0.01]
        if len(pts) >= 3:
            ds = np.array([p[0] for p in pts])
            lg = np.array([p[1] for p in pts])
            # Linear fit
            A = np.vstack([lg, np.ones(len(lg))]).T
            result = np.linalg.lstsq(A, ds, rcond=None)
            slope, intercept = result[0]
            residuals = ds - (slope * lg + intercept)
            rmse = np.sqrt(np.mean(residuals**2))
            r_squared = 1 - np.sum(residuals**2) / np.sum((ds - ds.mean())**2)
            print(f"    L{li}: δ* = {slope:.3f} × ln(gap) + {intercept:.3f}  "
                  f"(R²={r_squared:.3f}, RMSE={rmse:.3f}, N={len(pts)})")

    # D. Normalized coordinate: ξ = δ* - α × ln(gap)
    # Try to find the α that minimizes spread
    print(f"\n  D. Optimal normalization: ξ = δ* - α × ln(gap)")
    for li in TARGET_LAYERS:
        pts = [(d['delta_star'], np.log(d['gap']))
               for d in grp0 if d['layer'] == li and d['gap'] > 0.01]
        if len(pts) >= 3:
            ds = np.array([p[0] for p in pts])
            lg = np.array([p[1] for p in pts])
            best_alpha = None
            best_std = float('inf')
            for alpha in np.linspace(-3, 3, 601):
                xi = ds - alpha * lg
                s = xi.std()
                if s < best_std:
                    best_std = s
                    best_alpha = alpha
            xi_opt = ds - best_alpha * lg
            print(f"    L{li}: α={best_alpha:.3f}, ξ = {xi_opt.mean():.3f} ± {xi_opt.std():.3f} "
                  f"(raw std={ds.std():.3f}, reduction={1-xi_opt.std()/ds.std():.1%})")

    # E. Zero spacing within each prompt/layer
    print(f"\n  E. Zero spacing analysis:")
    spacings = []
    for li in TARGET_LAYERS:
        for pi in range(len(PROMPTS)):
            zeros_here = sorted([d['delta_star'] for d in grp0
                                if d['layer'] == li and d['prompt_idx'] == pi])
            if len(zeros_here) >= 2:
                for i in range(len(zeros_here) - 1):
                    sp = zeros_here[i+1] - zeros_here[i]
                    spacings.append({'layer': li, 'prompt_idx': pi, 'spacing': sp})
    if spacings:
        sp_arr = np.array([s['spacing'] for s in spacings])
        print(f"    Total spacings: {len(spacings)}")
        print(f"    Mean spacing: {sp_arr.mean():.3f}")
        print(f"    Std spacing:  {sp_arr.std():.3f}")
        print(f"    Min/Max:      {sp_arr.min():.3f} / {sp_arr.max():.3f}")
        # Is spacing related to gap?
        for li in TARGET_LAYERS:
            sp_layer = [s for s in spacings if s['layer'] == li]
            if sp_layer:
                arr = np.array([s['spacing'] for s in sp_layer])
                print(f"    L{li}: {len(arr)} spacings, mean={arr.mean():.3f} ± {arr.std():.3f}")
    else:
        print(f"    Too few multi-zero cases for spacing analysis")

    # F. ε-group comparison
    print(f"\n  F. ε-group comparison (grp0 vs grp1):")
    for li in TARGET_LAYERS:
        z0 = sorted([d['delta_star'] for d in grp0 if d['layer'] == li])
        z1 = sorted([d['delta_star'] for d in grp1 if d['layer'] == li])
        if z0 and z1:
            print(f"    L{li}: grp0 mean={np.mean(z0):.3f} ({len(z0)} zeros), "
                  f"grp1 mean={np.mean(z1):.3f} ({len(z1)} zeros)")
            # Do they share zeros? Check if any grp1 zero is within 0.5 of a grp0 zero
            shared = 0
            for z in z1:
                if any(abs(z - z0i) < 0.5 for z0i in z0):
                    shared += 1
            print(f"          shared (within 0.5): {shared}/{len(z1)}")
        elif z0:
            print(f"    L{li}: grp0 has {len(z0)} zeros, grp1 has none")
        elif z1:
            print(f"    L{li}: grp0 has none, grp1 has {len(z1)} zeros")

    # G. Per-prompt: first zero δ* vs gap (the critical line candidate)
    print(f"\n  G. First zero per prompt/layer (critical line test):")
    print(f"    {'Layer':>5s} {'Prompt':>45s} {'gap':>6s} {'δ*₁':>8s} {'φ^δ*₁':>8s} "
          f"{'φ^δ*/gap':>8s} {'ln(gap)':>7s}")
    for li in TARGET_LAYERS:
        print(f"    --- Layer {li} ---")
        for pi in range(len(PROMPTS)):
            zeros_here = sorted([d['delta_star'] for d in grp0
                                if d['layer'] == li and d['prompt_idx'] == pi])
            gap = None
            for d in grp0:
                if d['layer'] == li and d['prompt_idx'] == pi:
                    gap = d['gap']
                    break
            if not gap:
                # Get gap from baseline
                input_ids = tokenizer(PROMPTS[pi], return_tensors="pt").input_ids.to('cuda')
                logits_base = get_logits(model, input_ids)
                sv, _ = logits_base.sort(descending=True)
                gap = (sv[0] - sv[1]).item()

            if zeros_here:
                d1 = zeros_here[0]
                pd1 = PHI ** d1
                ratio = pd1 / gap if gap > 0.01 else float('inf')
                print(f"    L{li:>3d} {PROMPTS[pi]:>45s} {gap:>6.3f} {d1:>8.3f} "
                      f"{pd1:>8.3f} {ratio:>8.2f} {np.log(gap):>7.3f}")
            else:
                print(f"    L{li:>3d} {PROMPTS[pi]:>45s} {gap:>6.3f} {'none':>8s}")

    # H. Summary verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT: CRITICAL LINE?")
    print(f"{'='*70}")

    # Check if any normalization achieves CV < 0.2 (indicating collapse)
    for li in TARGET_LAYERS:
        # Raw δ*
        raw = np.array([d['delta_star'] for d in grp0 if d['layer'] == li])
        # Leverage ratio
        ratios = np.array([d['phi_delta'] / d['gap'] for d in grp0
                          if d['layer'] == li and d['gap'] > 0.01])
        if len(raw) >= 3:
            cv_raw = raw.std() / raw.mean() if raw.mean() != 0 else float('inf')
            cv_ratio = ratios.std() / ratios.mean() if len(ratios) >= 3 and ratios.mean() != 0 else float('inf')
            print(f"\n  L{li}:")
            print(f"    Raw δ*:     CV = {cv_raw:.3f} {'← tight' if cv_raw < 0.2 else ''}")
            print(f"    φ^δ*/gap:   CV = {cv_ratio:.3f} {'← tight' if cv_ratio < 0.2 else ''}")
            if cv_raw < 0.2:
                print(f"    ★ δ* itself may be the critical line (constant per layer)")
            elif cv_ratio < 0.2:
                print(f"    ★ φ^δ*/gap collapses — critical line is δ* = log_φ(C × gap)")

    print(f"\n  Done.")


if __name__ == '__main__':
    run()
