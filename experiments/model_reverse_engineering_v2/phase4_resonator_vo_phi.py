"""
Phase 4: Does the Resonator's V/O follow φ^levels × signs?

From phase4_resonator_simplify.py we know:
  - d_k reduces to sign-only (all -1s) = 6/6
  - V/O rank-k SVD fails at 4/6 even at rank-128
  - V/O S[0]=43.7 captures 91.7% of energy

Questions:
  A. Do V and O weight matrices individually sit on the φ-lattice?
  B. Does sign(VO) × φ^round(log_φ(|VO|)) preserve 6/6?
  C. Does sign-only VO work? (ultimate test: is knowledge just signs?)
  D. Does the factored form (V then O separately) preserve φ-structure?
  E. Can we φ-quantize V and O separately and still get 6/6?
"""

import sys, numpy as np, time, gc
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def finish_forward(engine, hidden_start, start_layer):
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    s = np.sort(logits[0, -1, :])[::-1]
    return idx, tok, s[0] - s[1]


def phi_lattice_analysis(W, name):
    """Analyze how well a weight matrix sits on the φ-lattice."""
    flat = W.flatten()
    nz = flat[np.abs(flat) > 1e-10]
    phi_lvl = np.log(np.abs(nz)) / LOG_PHI
    phi_rnd = np.round(phi_lvl)
    resid = phi_lvl - phi_rnd

    print(f"\n  {name} φ-lattice analysis ({W.shape}):")
    print(f"    Range: [{flat.min():.6f}, {flat.max():.6f}]")
    print(f"    Nonzero: {len(nz)}/{len(flat)}")
    print(f"    φ-level range: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
    print(f"    Mean |residual|: {np.abs(resid).mean():.4f}")
    print(f"    Within 0.1 of int φ-level: {(np.abs(resid)<0.1).mean()*100:.1f}%")
    print(f"    Within 0.2 of int φ-level: {(np.abs(resid)<0.2).mean()*100:.1f}%")
    print(f"    Within 0.3 of int φ-level: {(np.abs(resid)<0.3).mean()*100:.1f}%")

    # Distribution
    lc = {}
    for l in phi_rnd:
        k = int(l); lc[k] = lc.get(k, 0) + 1
    for k in sorted(lc.keys()):
        pct = lc[k]/len(nz)*100
        bar = "#" * min(int(pct), 50)
        print(f"    φ^{k:+3d}: {lc[k]:6d} ({pct:5.1f}%)  {bar}")

    # Signs
    n_pos = (flat > 1e-10).sum()
    n_neg = (flat < -1e-10).sum()
    print(f"    Signs: + {n_pos}  - {n_neg}  ({n_pos/(n_pos+n_neg)*100:.1f}% positive)")

    return phi_lvl, phi_rnd, resid


def run_resonator_custom(engine, tokenizer, target_layer, d_k_vec,
                         v_fn, o_fn, prompts, label=""):
    """Test resonator with custom V/O functions.
    v_fn: hidden_dim -> head_dim
    o_fn: head_dim -> hidden_dim
    """
    layer = engine.layers[target_layer]
    attn = layer.attention
    n_pass = 0; fm = None; fp = False

    for prompt in prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == target_layer:
                full_out = lo(h.copy())
                break
            h = lo(h)

        normed = rms_norm(h, attn.norm_weight)
        kf = normed[0] @ d_k_vec
        sp = int(np.argmax(kf))
        h_sel = normed[0, sp, :]  # (hidden_dim,)

        # Custom V then O
        v_out = v_fn(h_sel)       # (head_dim,)
        attn_c = o_fn(v_out)      # (hidden_dim,)

        pa = h.copy()
        pa[0, -1, :] += attn_c

        mlp = layer.mlp
        nm = rms_norm(pa, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mo = phi_linear(mlp.W_down, phi_silu(g) * u)
        so = pa + mo

        lf = finish_forward(engine, full_out, target_layer)
        ls = finish_forward(engine, so, target_layer)
        fi, ft, _ = get_top1(lf, tokenizer)
        si, st, sm = get_top1(ls, tokenizer)
        if si == fi: n_pass += 1
        if 'France' in prompt: fm = sm; fp = si == fi

    fs = "✓" if fp else "✗"
    ms = f"margin={fm:.3f}" if fm is not None else ""
    print(f"  {label:>50s}: {n_pass}/6  France={fs} {ms}", flush=True)
    return n_pass


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    target_layer = 23
    head_idx = 6
    attn = engine.layers[target_layer].attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    kv_group = head_idx // heads_per_kv
    hidden_dim = engine.hidden_dim

    prompts = [
        'The capital of France is',
        'The largest ocean is the',
        'The color of grass is',
        'Barack Obama was the',
        'To be or not to',
        'Roses are red, violets are',
    ]

    # === Extract W_v, W_o for head 6 ===
    # IMPORTANT: extract weights WITHOUT bias to avoid contamination.
    # Bias is stored separately and applied in the test function.
    print("\nExtracting weights (bias-free)...", flush=True)
    I = np.eye(hidden_dim, dtype=np.float32)
    Wk = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wq = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wv = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]
        # Extract WITHOUT bias for clean weight matrices
        qo = phi_linear(attn.W_q, c)[0].reshape(-1, num_heads, head_dim)
        ko = phi_linear(attn.W_k, c)[0].reshape(-1, num_kv_heads, head_dim)
        vo = phi_linear(attn.W_v, c)[0].reshape(-1, num_kv_heads, head_dim)
        Wq[:, s:e] = qo[:, head_idx, :].T
        Wk[:, s:e] = ko[:, kv_group, :].T
        Wv[:, s:e] = vo[:, kv_group, :].T
        if s % 1024 == 0: print(f"  {e}/{hidden_dim}...", flush=True)

    # Extract biases separately
    # Q bias for head_idx, K bias for kv_group, V bias for kv_group
    zero_in = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    if attn.b_q is not None:
        bq_full = phi_linear(attn.W_q, zero_in, attn.b_q)[0, 0] - phi_linear(attn.W_q, zero_in)[0, 0]
        bq_full = bq_full.reshape(num_heads, head_dim)
        bq_head = bq_full[head_idx]
    else:
        bq_head = np.zeros(head_dim, dtype=np.float32)
    if attn.b_k is not None:
        bk_full = phi_linear(attn.W_k, zero_in, attn.b_k)[0, 0] - phi_linear(attn.W_k, zero_in)[0, 0]
        bk_full = bk_full.reshape(num_kv_heads, head_dim)
        bk_group = bk_full[kv_group]
    else:
        bk_group = np.zeros(head_dim, dtype=np.float32)
    if attn.b_v is not None:
        bv_full = phi_linear(attn.W_v, zero_in, attn.b_v)[0, 0] - phi_linear(attn.W_v, zero_in)[0, 0]
        bv_full = bv_full.reshape(num_kv_heads, head_dim)
        bv_group = bv_full[kv_group]
    else:
        bv_group = np.zeros(head_dim, dtype=np.float32)
    print(f"  V bias norm: {np.linalg.norm(bv_group):.4f}", flush=True)

    h6in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h6in[0, 0, :] = 0.0
        h6in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h6in)[0, 0, :]

    # d_k from MESH SVD
    MESH = Wq @ Wk.T
    U, S, Vt = np.linalg.svd(MESH)
    d_k = Wk.T @ Vt[0, :]
    # Use sign-only d_k (established: all -1s, 6/6)
    dk_sign = np.sign(d_k)

    print(f"\nWv: {Wv.shape}, Wo: {Wo.shape}")
    print(f"d_k sign check: all negative = {(d_k < 0).all()}", flush=True)

    # =========================================================================
    #   Part A: φ-lattice structure of Wv and Wo individually
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: φ-lattice structure of V and O weight matrices")
    print("=" * 80)

    phi_lattice_analysis(Wv, "W_v (head_dim × hidden_dim)")
    phi_lattice_analysis(Wo, "W_o (hidden_dim × head_dim)")

    # Also analyze VO combined
    VO = Wo @ Wv  # (hidden_dim × hidden_dim)
    phi_lattice_analysis(VO, "VO combined (hidden × hidden)")

    # =========================================================================
    #   Part B: φ-quantized VO — does sign × φ^round(log_φ|w|) work?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: φ-quantized V/O end-to-end")
    print("=" * 80)

    # Baseline: full V/O with sign-only d_k
    print("\n  Using sign-only d_k for all tests:", flush=True)

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv @ h + bv_group, lambda v: Wo @ v,
        prompts, "Full Wv, Full Wo (baseline)")

    # φ-quantize Wv
    Wv_phi = np.sign(Wv) * PHI ** np.round(np.log(np.abs(Wv) + 1e-20) / LOG_PHI)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_phi @ h + bv_group, lambda v: Wo @ v,
        prompts, "φ-quantized Wv, Full Wo")

    # φ-quantize Wo
    Wo_phi = np.sign(Wo) * PHI ** np.round(np.log(np.abs(Wo) + 1e-20) / LOG_PHI)
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv @ h + bv_group, lambda v: Wo_phi @ v,
        prompts, "Full Wv, φ-quantized Wo")

    # Both φ-quantized
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_phi @ h + bv_group, lambda v: Wo_phi @ v,
        prompts, "φ-quantized Wv, φ-quantized Wo")

    # =========================================================================
    #   Part C: Sign-only V/O — is knowledge just signs?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Sign-only V/O")
    print("=" * 80)

    Wv_sign = np.sign(Wv)
    Wo_sign = np.sign(Wo)

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_sign @ h + bv_group, lambda v: Wo @ v,
        prompts, "Sign-only Wv, Full Wo")

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv @ h + bv_group, lambda v: Wo_sign @ v,
        prompts, "Full Wv, Sign-only Wo")

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_sign @ h + bv_group, lambda v: Wo_sign @ v,
        prompts, "Sign-only Wv, Sign-only Wo")

    # Scale-corrected signs: sign × mean(|W|)
    Wv_smean = Wv_sign * np.mean(np.abs(Wv))
    Wo_smean = Wo_sign * np.mean(np.abs(Wo))
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_smean @ h + bv_group, lambda v: Wo_smean @ v,
        prompts, "Scaled-sign Wv, Scaled-sign Wo")

    # =========================================================================
    #   Part D: Per-row φ-level structure (does each output dim have one level?)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Per-row φ-level structure")
    print("=" * 80)

    # For Wv (head_dim × hidden_dim), check if each row has a dominant φ-level
    print("\n  W_v per-row analysis (each row = one head_dim output):")
    row_levels = []
    for r in range(min(head_dim, 128)):
        row = Wv[r, :]
        nz = row[np.abs(row) > 1e-10]
        if len(nz) == 0:
            row_levels.append(None)
            continue
        lvls = np.log(np.abs(nz)) / LOG_PHI
        median_lvl = np.median(lvls)
        std_lvl = np.std(lvls)
        row_levels.append((median_lvl, std_lvl))
    
    medians = [r[0] for r in row_levels if r is not None]
    stds = [r[1] for r in row_levels if r is not None]
    print(f"    Median φ-level per row: mean={np.mean(medians):.2f}, std={np.std(medians):.2f}")
    print(f"    Within-row φ-level std: mean={np.mean(stds):.2f}, min={np.min(stds):.2f}, max={np.max(stds):.2f}")
    if np.mean(stds) < 0.5:
        print(f"    → Rows are φ-level homogeneous! Each output dim has one level.")
    else:
        print(f"    → Rows are NOT homogeneous — mixed φ-levels within each row.")

    # Same for Wo
    print(f"\n  W_o per-row analysis (each row = one hidden_dim output):")
    wo_stds = []
    wo_medians = []
    for r in range(hidden_dim):
        row = Wo[r, :]
        nz = row[np.abs(row) > 1e-10]
        if len(nz) < 2: continue
        lvls = np.log(np.abs(nz)) / LOG_PHI
        wo_medians.append(np.median(lvls))
        wo_stds.append(np.std(lvls))
    print(f"    Median φ-level per row: mean={np.mean(wo_medians):.2f}, std={np.std(wo_medians):.2f}")
    print(f"    Within-row φ-level std: mean={np.mean(wo_stds):.2f}, min={np.min(wo_stds):.2f}, max={np.max(wo_stds):.2f}")

    # =========================================================================
    #   Part E: Factored φ-quantization with per-row scale
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Per-row scaled φ-quantization")
    print("=" * 80)

    # For each row, compute row-scale = mean(|row|), then quantize sign × φ^level
    # and multiply by row-scale. This preserves per-row magnitude while using φ-levels.
    def row_scaled_phi(W):
        W_out = np.zeros_like(W)
        for r in range(W.shape[0]):
            row = W[r, :]
            scale = np.mean(np.abs(row)) + 1e-20
            normalized = row / scale
            W_out[r, :] = scale * np.sign(normalized) * PHI ** np.round(
                np.log(np.abs(normalized) + 1e-20) / LOG_PHI)
        return W_out

    Wv_rsphi = row_scaled_phi(Wv)
    Wo_rsphi = row_scaled_phi(Wo)

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_rsphi @ h + bv_group, lambda v: Wo @ v,
        prompts, "Row-scaled-φ Wv, Full Wo")

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv @ h + bv_group, lambda v: Wo_rsphi @ v,
        prompts, "Full Wv, Row-scaled-φ Wo")

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_rsphi @ h + bv_group, lambda v: Wo_rsphi @ v,
        prompts, "Row-scaled-φ Wv, Row-scaled-φ Wo")

    # Per-row sign-only (with row scale)
    def row_scaled_sign(W):
        W_out = np.zeros_like(W)
        for r in range(W.shape[0]):
            row = W[r, :]
            scale = np.mean(np.abs(row)) + 1e-20
            W_out[r, :] = scale * np.sign(row)
        return W_out

    Wv_rss = row_scaled_sign(Wv)
    Wo_rss = row_scaled_sign(Wo)

    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_rss @ h + bv_group, lambda v: Wo_rss @ v,
        prompts, "Row-scaled-sign Wv, Row-scaled-sign Wo")

    # =========================================================================
    #   Part F: The absolute minimum — what's the simplest that works?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part F: Simplest working configuration")
    print("=" * 80)

    # Full d_k (for comparison)
    run_resonator_custom(
        engine, tokenizer, target_layer, d_k,
        lambda h: Wv @ h + bv_group, lambda v: Wo @ v,
        prompts, "Full d_k + Full V/O (original)")

    # Sign d_k + Full V/O
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv @ h + bv_group, lambda v: Wo @ v,
        prompts, "Sign d_k + Full V/O")

    # Sign d_k + φ-quantized V/O
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_phi @ h + bv_group, lambda v: Wo_phi @ v,
        prompts, "Sign d_k + φ-quant V/O")

    # Sign d_k + row-scaled-φ V/O
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_rsphi @ h + bv_group, lambda v: Wo_rsphi @ v,
        prompts, "Sign d_k + row-scaled-φ V/O")

    # Sign d_k + row-scaled-sign V/O
    run_resonator_custom(
        engine, tokenizer, target_layer, dk_sign,
        lambda h: Wv_rss @ h + bv_group, lambda v: Wo_rss @ v,
        prompts, "Sign d_k + row-scaled-sign V/O")

    # =========================================================================
    #   Part G: Parameter counts
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part G: Parameter counts for each configuration")
    print("=" * 80)

    configs = [
        ("Full d_k + Full V/O", hidden_dim * 32 + (head_dim * hidden_dim + hidden_dim * head_dim) * 32),
        ("Sign d_k + Full V/O", hidden_dim * 1 + (head_dim * hidden_dim + hidden_dim * head_dim) * 32),
        ("Sign d_k + φ-quant V/O", hidden_dim * 1 + (head_dim * hidden_dim + hidden_dim * head_dim) * 5),
        ("Sign d_k + row-scaled-φ V/O",
         hidden_dim * 1 + (head_dim * hidden_dim + hidden_dim * head_dim) * 5 + (head_dim + hidden_dim) * 32),
        ("Sign d_k + row-scaled-sign V/O",
         hidden_dim * 1 + (head_dim * hidden_dim + hidden_dim * head_dim) * 1 + (head_dim + hidden_dim) * 32),
        ("Sign d_k + sign-only V/O", hidden_dim * 1 + (head_dim * hidden_dim + hidden_dim * head_dim) * 1),
    ]

    full_attn_bits = 51_380_224 * 32
    for name, bits in configs:
        kb = bits / 8 / 1024
        ratio = full_attn_bits / bits if bits > 0 else float('inf')
        print(f"  {name:>45s}: {bits:>12,} bits  ({kb:>8.1f} KB)  {ratio:>6.0f}× vs full attn")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
