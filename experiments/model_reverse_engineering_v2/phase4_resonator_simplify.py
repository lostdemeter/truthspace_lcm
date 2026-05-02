"""
Phase 4: Simplifying the Geometric Resonator

Questions:
  A. Does d_k have φ-lattice structure?
  B. Is d_k sparse? How many dimensions matter for routing?
  C. Can d_k be quantized (signs only, ternary, top-k)?
  D. Can V/O be simplified (SVD rank)?
  E. Absolute minimum representation for 6/6?
"""

import sys, numpy as np, time, gc
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI_CONST = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI_CONST)


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
    sorted_l = np.sort(logits[0, -1, :])[::-1]
    return idx, tok, sorted_l[0] - sorted_l[1]


def test_dk(engine, tokenizer, target_layer, d_k_vec, W_v_grp, W_o_h6,
            kv_group, head_dim, num_kv_heads, prompts, label=""):
    layer = engine.layers[target_layer]
    attn = layer.attention
    n_pass = 0
    fm = None
    fp = False
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
        v_in = normed[:, sp:sp+1, :]
        V = phi_linear(attn.W_v, v_in, attn.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        vs = V[0, 0, kv_group, :]
        ac = W_o_h6 @ vs
        pa = h.copy()
        pa[0, -1, :] += ac
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
    ms = f"margin={fm:.3f}" if fm else ""
    print(f"  {label:>45s}: {n_pass}/6  France={fs} {ms}", flush=True)
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

    # Extract weights
    print("\nExtracting weights...", flush=True)
    I = np.eye(hidden_dim, dtype=np.float32)
    Wq = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wk = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    Wv = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    for s in range(0, hidden_dim, 512):
        e = min(s + 512, hidden_dim)
        c = I[s:e][np.newaxis, :, :]
        qo = phi_linear(attn.W_q, c, attn.b_q)[0].reshape(-1, num_heads, head_dim)
        ko = phi_linear(attn.W_k, c, attn.b_k)[0].reshape(-1, num_kv_heads, head_dim)
        vo = phi_linear(attn.W_v, c, attn.b_v)[0].reshape(-1, num_kv_heads, head_dim)
        Wq[:, s:e] = qo[:, head_idx, :].T
        Wk[:, s:e] = ko[:, kv_group, :].T
        Wv[:, s:e] = vo[:, kv_group, :].T
        if s % 1024 == 0: print(f"  {e}/{hidden_dim}...", flush=True)

    h6in = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    Wo = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        h6in[0, 0, :] = 0.0
        h6in[0, 0, head_idx * head_dim + d] = 1.0
        Wo[:, d] = phi_linear(attn.W_o, h6in)[0, 0, :]

    MESH = Wq @ Wk.T
    U, S, Vt = np.linalg.svd(MESH)
    d_k = Wk.T @ Vt[0, :]
    print(f"MESH S[0]={S[0]:.1f}, ||d_k||={np.linalg.norm(d_k):.4f}", flush=True)

    sorted_idx = np.argsort(np.abs(d_k))[::-1]

    # === Part A: φ-lattice structure ===
    print("\n" + "=" * 80)
    print("  Part A: φ-lattice structure of d_k")
    print("=" * 80)
    dk_abs = np.abs(d_k)
    nz = dk_abs[dk_abs > 1e-10]
    phi_lvl = np.log(nz) / LOG_PHI
    phi_rnd = np.round(phi_lvl)
    resid = phi_lvl - phi_rnd
    print(f"  Range: [{d_k.min():.6f}, {d_k.max():.6f}], std={d_k.std():.6f}")
    print(f"  φ-level range: [{phi_lvl.min():.2f}, {phi_lvl.max():.2f}]")
    print(f"  Mean |residual| from integer φ-level: {np.abs(resid).mean():.4f}")
    print(f"  Within 0.1 of integer: {(np.abs(resid)<0.1).mean()*100:.1f}%")
    print(f"  Within 0.2 of integer: {(np.abs(resid)<0.2).mean()*100:.1f}%")
    print(f"  Signs: + {(d_k>0).sum()}  - {(d_k<0).sum()}")
    lc = {}
    for l in phi_rnd:
        k = int(l); lc[k] = lc.get(k, 0) + 1
    for k in sorted(lc.keys()):
        print(f"    φ^{k:+3d}: {lc[k]:5d}  {'#'*min(lc[k]//10,50)}")

    # === Part B: Sparsity ===
    print("\n" + "=" * 80)
    print("  Part B: How many dimensions matter?")
    print("=" * 80)
    cum = np.cumsum(d_k[sorted_idx]**2) / np.sum(d_k**2)
    for t in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = int(np.searchsorted(cum, t)) + 1
        print(f"  {t*100:.0f}% energy: top {k} dims ({k/len(d_k)*100:.1f}%)")

    print(f"\n  Routing with sparse d_k:", flush=True)
    test_dk(engine, tokenizer, target_layer, d_k, Wv, Wo, kv_group, head_dim,
            num_kv_heads, prompts, "Full d_k (3584)")
    for tk in [10, 25, 50, 100, 200, 500, 1000]:
        sdk = np.zeros_like(d_k)
        sdk[sorted_idx[:tk]] = d_k[sorted_idx[:tk]]
        test_dk(engine, tokenizer, target_layer, sdk, Wv, Wo, kv_group,
                head_dim, num_kv_heads, prompts, f"Top-{tk} d_k")

    # === Part C: Quantized d_k ===
    print("\n" + "=" * 80)
    print("  Part C: Quantized d_k")
    print("=" * 80)

    test_dk(engine, tokenizer, target_layer, np.sign(d_k), Wv, Wo, kv_group,
            head_dim, num_kv_heads, prompts, "Sign-only (1 bit/dim)")

    for tp in [25, 50, 75]:
        th = np.percentile(np.abs(d_k), tp)
        dt = np.zeros_like(d_k)
        dt[d_k > th] = 1.0; dt[d_k < -th] = -1.0
        nn = (dt != 0).sum()
        test_dk(engine, tokenizer, target_layer, dt, Wv, Wo, kv_group,
                head_dim, num_kv_heads, prompts, f"Ternary p{tp} ({nn} nz)")

    for tk in [10, 25, 50, 100]:
        dss = np.zeros_like(d_k)
        dss[sorted_idx[:tk]] = np.sign(d_k[sorted_idx[:tk]])
        test_dk(engine, tokenizer, target_layer, dss, Wv, Wo, kv_group,
                head_dim, num_kv_heads, prompts, f"Top-{tk} signs")

    dkphi = np.sign(d_k) * PHI_CONST ** np.round(
        np.log(np.abs(d_k) + 1e-20) / LOG_PHI)
    test_dk(engine, tokenizer, target_layer, dkphi, Wv, Wo, kv_group,
            head_dim, num_kv_heads, prompts, "φ-quantized d_k")

    # === Part D: V/O rank structure ===
    print("\n" + "=" * 80)
    print("  Part D: V/O projection rank structure")
    print("=" * 80)
    # Combined V→O is Wo @ Wv (hidden → hidden, rank ≤ 128)
    VO = Wo @ Wv  # (hidden, hidden) — the full "fetch and project" operation
    U_vo, S_vo, Vt_vo = np.linalg.svd(VO, full_matrices=False)
    print(f"  VO matrix: {VO.shape}")
    print(f"  Top 10 SVs: {S_vo[:10]}")
    total_e = (S_vo**2).sum()
    for r in [1, 2, 3, 5, 10, 20]:
        pct = (S_vo[:r]**2).sum() / total_e * 100
        print(f"  Rank-{r:2d}: {pct:.1f}% of VO energy")

    # Test rank-k VO approximation
    print(f"\n  End-to-end with rank-k VO:", flush=True)
    for r in [1, 2, 3, 5, 10, 20, 128]:
        Ur = U_vo[:, :r]
        Sr = S_vo[:r]
        Vr = Vt_vo[:r, :]
        def make_v_fn(Vr_=Vr):
            return lambda h: Vr_ @ h
        def make_o_fn(Ur_=Ur, Sr_=Sr):
            return lambda v: Ur_ @ (Sr_ * v)
        test_dk(engine, tokenizer, target_layer, d_k, Wv, Wo, kv_group,
                head_dim, num_kv_heads, prompts, f"Full V/O (rank-{r} unused)")
        # Actually test with the rank-k VO
        layer_ = engine.layers[target_layer]
        attn_ = layer_.attention
        n_pass = 0; fm = None; fp = False
        for prompt in prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    full_out = lo(h.copy())
                    break
                h = lo(h)
            normed = rms_norm(h, attn_.norm_weight)
            kf = normed[0] @ d_k
            sp = int(np.argmax(kf))
            h_sel = normed[0, sp, :]
            # rank-k VO: output = Ur @ diag(Sr) @ Vr @ h_sel
            proj = Vr @ h_sel      # (r,)
            out = Ur @ (Sr * proj)  # (hidden,)
            pa = h.copy()
            pa[0, -1, :] += out
            mlp = layer_.mlp
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
        ms = f"margin={fm:.3f}" if fm else ""
        print(f"  {'Rank-'+str(r)+' VO':>45s}: {n_pass}/6  France={fs} {ms}",
              flush=True)

    # === Part E: Summary ===
    print("\n" + "=" * 80)
    print("  Part E: Parameter count summary")
    print("=" * 80)
    print(f"  Full attention layer 23: {3*hidden_dim*(num_heads*head_dim) + (num_heads*head_dim)*hidden_dim:,} params")
    print(f"  d_k alone (3584 floats): {hidden_dim:,} params")
    print(f"  d_k + VO rank-k:")
    for r in [1, 2, 5, 10]:
        # d_k: hidden_dim, VO rank-r: hidden*(r) + r + hidden*(r) = 2*hidden*r + r
        p = hidden_dim + 2 * hidden_dim * r + r
        print(f"    rank-{r:2d}: {p:,} params  ({p*4/1024:.1f} KB)")
    print(f"  Original: {(3*hidden_dim*(num_heads*head_dim) + (num_heads*head_dim)*hidden_dim)*4/1024/1024:.1f} MB")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
