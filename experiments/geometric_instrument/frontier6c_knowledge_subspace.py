"""
Frontier 6c: φ-Basis in Knowledge Subspace
=============================================
F6b showed: per-dimension φ-encoding gives sparse flips (7-10%) but
flipping ALL of them still predicts Paris. The answer isn't in
individual dimension signs — it's in the ANGLE to the knowledge
direction within M_h's aperture.

Hypothesis: project COMB outputs into M_h's SVD basis (the 66-d Lens
aperture from F125), THEN φ-encode. In this basis:
- The answer should be a small rotation (a few sign flips)
- Sign flips here correspond to actual answer changes
- The top ~10 dims carry the answer, dims 10-66 carry identity

Tests:
  1. Project COMB outputs into M_h's SVD basis (128-d, eff rank 66)
  2. φ-encode in this basis — how many sign flips between countries?
  3. Flip signs in knowledge subspace — does this change the answer?
  4. How many dims needed? (top 10 for answers, top 66 for identity)
  5. Can we navigate France→Germany by flipping signs in knowledge space?
"""

import sys, os, time
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_integer import phi_to_float
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI_CONST = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI_CONST)
K_SCALE = 1000


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def encode_phi(x):
    signs = np.sign(x).astype(np.int8)
    signs[signs == 0] = 1
    mags = np.abs(x).astype(np.float64) + 1e-45
    levels = np.round(K_SCALE * np.log(mags) / LOG_PHI).astype(np.int16)
    return signs, levels


def decode_phi(signs, levels):
    mags = PHI_CONST ** (levels.astype(np.float64) / K_SCALE)
    return signs.astype(np.float64) * mags


def run_layers(engine, h, start, end):
    for li in range(start, end):
        layer = engine.layers[li]
        attn, mlp = layer.attention, layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        sl = h.shape[1]
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        h_pa = h + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)
        gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
        h = h_pa + phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))
    return h


def predict_token(engine, tokenizer, h):
    for attr in ['final_norm_weight', 'norm_weight', 'ln_f_weight']:
        if hasattr(engine, attr):
            final_norm_w = getattr(engine, attr)
            break
    else:
        final_norm_w = engine.final_norm.weight
    h_last = rms_norm(h[:, -1:, :], final_norm_w)
    lm_w = engine.lm_head_weight if hasattr(engine, 'lm_head_weight') else engine.lm_head.weight
    logits = phi_linear(lm_w, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def get_head_matrices(W_v, b_v, W_o, head_idx, hd=128, nh=28, nkv=4):
    kv = head_idx // (nh // nkv)
    W_v_h = W_v[kv*hd:(kv+1)*hd, :]
    b_v_h = b_v[kv*hd:(kv+1)*hd]
    W_o_h = W_o[:, head_idx*hd:(head_idx+1)*hd]
    return W_v_h, b_v_h, W_o_h


def main():
    print("=" * 80)
    print("  Frontier 6c: φ-Basis in Knowledge Subspace")
    print("=" * 80)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Extract M_h's SVD basis from L23 H6 (the knowledge head)
    # ═══════════════════════════════════════════════════════════
    print("\n  Extracting M_h SVD basis (L23 H6)...", flush=True)
    attn23 = engine.layers[23].attention
    W_v_23 = phi_to_float(attn23.W_v.signs, attn23.W_v.exponents)
    W_o_23 = phi_to_float(attn23.W_o.signs, attn23.W_o.exponents)
    b_v_23 = attn23.b_v.copy()
    W_v_h6, b_v_h6, W_o_h6 = get_head_matrices(W_v_23, b_v_23, W_o_23, 6)

    # SVD of the output projection W_o_h6.T (128 x 3584)
    # This defines the 128-d knowledge subspace
    W_o_h6_T = W_o_h6.T  # 128 x 3584
    U_o, S_o, Vt_o = np.linalg.svd(W_o_h6_T, full_matrices=False)
    # U_o: 128x128 (rotation in value space)
    # S_o: 128 (singular values)
    # Vt_o: 128x3584 (basis vectors in hidden space)

    energy_o = np.cumsum(S_o ** 2) / np.sum(S_o ** 2)
    rank66 = int(np.searchsorted(energy_o, 0.90) + 1)
    rank10 = 10  # critical answer rank from F125
    print(f"  W_o_h6.T SVD: rank@90% = {rank66}")
    print(f"  S_o[0:5] = {S_o[:5].round(3)}")
    print(f"  S_o[9:11] = {S_o[9:11].round(3)} (answer boundary)")
    print(f"  S_o[65:67] = {S_o[65:67].round(3)} (identity boundary)")

    # Also get the inner matrix SVD for the full pipeline
    inner = W_v_h6 @ W_o_h6  # 128x128
    U_inner, S_inner, Vt_inner = np.linalg.svd(inner, full_matrices=False)
    inner_energy = np.cumsum(S_inner ** 2) / np.sum(S_inner ** 2)
    inner_rank66 = int(np.searchsorted(inner_energy, 0.90) + 1)
    print(f"  Inner matrix (W_v @ W_o) rank@90% = {inner_rank66}")

    # Projection: hidden state → knowledge subspace coordinates
    # h_proj = h @ Vt_o.T  → 128 coefficients in knowledge basis
    # Reconstruct: h_recon = h_proj @ Vt_o → back to 3584-d

    prompts = [
        'The capital of France is',
        'The capital of Germany is',
        'The capital of Japan is',
        'I really love eating pizza',
        'Please help me find this',
        'Once upon a time there',
        'How does the engine work',
    ]
    working = [(p, tokenizer.encode(p)) for p in prompts
               if len(tokenizer.encode(p)) == 5]
    print(f"  Using {len(working)} prompts")

    COMB_S, COMB_E = 10, 21

    # ═══════════════════════════════════════════════════════════
    # Collect hidden states
    # ═══════════════════════════════════════════════════════════
    print("\n  Running forward passes...")
    h_before, h_after, h_fin, bl_pred = {}, {}, {}, {}
    for prompt, tids in working:
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, COMB_S)
        h_before[prompt] = h.copy()
        h = run_layers(engine, h, COMB_S, COMB_E)
        h_after[prompt] = h.copy()
        h = run_layers(engine, h, COMB_E, n_layers)
        h_fin[prompt] = h.copy()
        _, tok, _ = predict_token(engine, tokenizer, h)
        bl_pred[prompt] = tok[0]
        print(f"    '{prompt}' → {tok[0]!r}")

    caps = [p for p, _ in working if 'capital' in p]
    divs = [p for p, _ in working if 'capital' not in p]

    # ═══════════════════════════════════════════════════════════
    # Inv 1: Project COMB outputs into knowledge subspace
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 1: COMB Outputs in Knowledge Subspace")
    print("=" * 80)

    # Project last-position hidden state after COMB into Vt_o basis
    proj_h = {}  # 128-d knowledge coordinates
    for prompt, _ in working:
        h_last = h_after[prompt][0, -1].astype(np.float64)
        proj = h_last @ Vt_o.T  # 128 coefficients
        proj_h[prompt] = proj

        # How much energy is in the knowledge subspace?
        recon = proj @ Vt_o
        recon_cos = cosine(recon, h_last)
        in_energy = np.linalg.norm(recon) / np.linalg.norm(h_last)
        print(f"  '{prompt[:30]}': ||proj||={np.linalg.norm(proj):.1f} "
              f"recon_cos={recon_cos:.4f} energy_frac={in_energy:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 2: φ-Encode in knowledge subspace — sign flips
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: φ-Encode in 128-d Knowledge Subspace")
    print("=" * 80)

    phi_proj = {}
    for prompt, _ in working:
        s, l = encode_phi(proj_h[prompt])
        phi_proj[prompt] = (s, l)

    # Sign flips between capital pairs in knowledge subspace
    print("\n  --- Capital pairs in knowledge subspace ---")
    for i, p1 in enumerate(caps):
        for p2 in caps[i+1:]:
            s1, l1 = phi_proj[p1]
            s2, l2 = phi_proj[p2]
            xor = (s1 * s2).astype(np.int8)
            nf = np.sum(xor == -1)
            # Compare: full 3584-d had 261-368 flips
            # Break down by answer dims (top 10) vs identity dims (10-66) vs noise (66+)
            nf_ans = np.sum(xor[:10] == -1)
            nf_id = np.sum(xor[10:66] == -1)
            nf_noise = np.sum(xor[66:] == -1)
            print(f"  {p1[15:21]} vs {p2[15:21]}: {nf}/128 flips "
                  f"(ans:{nf_ans}/10, id:{nf_id}/56, noise:{nf_noise}/62)")

    print("\n  --- Cross-structure (France vs diverse) ---")
    for p2 in divs:
        s1, _ = phi_proj[caps[0]]
        s2, _ = phi_proj[p2]
        xor = (s1 * s2).astype(np.int8)
        nf = np.sum(xor == -1)
        nf_ans = np.sum(xor[:10] == -1)
        print(f"  France vs '{p2[:25]}': {nf}/128 flips (ans:{nf_ans}/10)")

    # ═══════════════════════════════════════════════════════════
    # Inv 3: Navigate in knowledge subspace — flip signs to change answer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: Navigate by Flipping Signs in Knowledge Subspace")
    print("=" * 80)

    ref = caps[0]  # France
    for tgt in caps[1:]:
        s_r, l_r = phi_proj[ref]
        s_t, l_t = phi_proj[tgt]
        xor = (s_r * s_t).astype(np.int8)
        flips = np.where(xor == -1)[0]
        print(f"\n  France → {tgt[15:21]}: {len(flips)} flips in knowledge subspace")

        # Strategy: modify France's COMB output by changing its projection
        # in knowledge subspace, then reconstruct to 3584-d

        h_ref = h_after[ref][0, -1].astype(np.float64)
        proj_ref = proj_h[ref].copy()

        # 3a: Flip all sign-difference dims in knowledge subspace
        proj_mod = proj_ref.copy()
        for d in flips:
            proj_mod[d] = -np.abs(proj_mod[d]) if s_t[d] == -1 else np.abs(proj_mod[d])

        # Reconstruct: replace the knowledge-subspace component of h
        h_recon = h_ref - (proj_ref @ Vt_o) + (proj_mod @ Vt_o)
        h_test = h_after[ref].copy()
        h_test[0, -1] = h_recon.astype(np.float32)

        h_post = run_layers(engine, h_test, COMB_E, n_layers)
        _, tok, _ = predict_token(engine, tokenizer, h_post)
        cos_t = cosine(h_post[0, -1], h_fin[tgt][0, -1])
        print(f"    3a (flip all {len(flips)} in KS):  → {tok[0]!r} cos={cos_t:.4f}")

        # 3b: Flip signs AND copy target levels in knowledge subspace
        proj_mod_b = proj_ref.copy()
        tgt_decoded = decode_phi(s_t, l_t)
        proj_mod_b[flips] = tgt_decoded[flips]

        h_recon_b = h_ref - (proj_ref @ Vt_o) + (proj_mod_b @ Vt_o)
        h_test_b = h_after[ref].copy()
        h_test_b[0, -1] = h_recon_b.astype(np.float32)

        h_post_b = run_layers(engine, h_test_b, COMB_E, n_layers)
        _, tok_b, _ = predict_token(engine, tokenizer, h_post_b)
        cos_t_b = cosine(h_post_b[0, -1], h_fin[tgt][0, -1])
        print(f"    3b (flip+levels in KS):  → {tok_b[0]!r} cos={cos_t_b:.4f}")

        # 3c: Replace ENTIRE knowledge subspace projection with target's
        proj_tgt = proj_h[tgt].copy()
        h_recon_c = h_ref - (proj_ref @ Vt_o) + (proj_tgt @ Vt_o)
        h_test_c = h_after[ref].copy()
        h_test_c[0, -1] = h_recon_c.astype(np.float32)

        h_post_c = run_layers(engine, h_test_c, COMB_E, n_layers)
        _, tok_c, _ = predict_token(engine, tokenizer, h_post_c)
        cos_t_c = cosine(h_post_c[0, -1], h_fin[tgt][0, -1])
        print(f"    3c (full KS replacement): → {tok_c[0]!r} cos={cos_t_c:.4f}")

        # 3d: Replace only TOP-10 dims (answer subspace)
        proj_mod_d = proj_ref.copy()
        proj_mod_d[:10] = proj_tgt[:10]
        h_recon_d = h_ref - (proj_ref @ Vt_o) + (proj_mod_d @ Vt_o)
        h_test_d = h_after[ref].copy()
        h_test_d[0, -1] = h_recon_d.astype(np.float32)

        h_post_d = run_layers(engine, h_test_d, COMB_E, n_layers)
        _, tok_d, _ = predict_token(engine, tokenizer, h_post_d)
        cos_t_d = cosine(h_post_d[0, -1], h_fin[tgt][0, -1])
        print(f"    3d (top-10 answer dims): → {tok_d[0]!r} cos={cos_t_d:.4f}")

        # 3e: Replace only TOP-10 dims WITH φ-sign flips (no oracle levels)
        proj_mod_e = proj_ref.copy()
        for d in range(10):
            if xor[d] == -1:
                proj_mod_e[d] = -proj_mod_e[d]  # Just flip the sign
        h_recon_e = h_ref - (proj_ref @ Vt_o) + (proj_mod_e @ Vt_o)
        h_test_e = h_after[ref].copy()
        h_test_e[0, -1] = h_recon_e.astype(np.float32)

        h_post_e = run_layers(engine, h_test_e, COMB_E, n_layers)
        _, tok_e, _ = predict_token(engine, tokenizer, h_post_e)
        cos_t_e = cosine(h_post_e[0, -1], h_fin[tgt][0, -1])
        print(f"    3e (flip ans signs only): → {tok_e[0]!r} cos={cos_t_e:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 4: Minimum dims needed in knowledge subspace
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: Minimum Knowledge Dims for Answer Change")
    print("=" * 80)

    for tgt in caps[1:]:
        ans = 'Berlin' if 'Germany' in tgt else 'Tokyo'
        proj_ref_v = proj_h[ref].copy()
        proj_tgt_v = proj_h[tgt].copy()

        print(f"\n  France → {tgt[15:21]} (target: {ans}):")
        for k in [1, 2, 3, 5, 10, 20, 30, 50, 66, 100, 128]:
            if k > 128:
                continue
            proj_mod = proj_ref_v.copy()
            proj_mod[:k] = proj_tgt_v[:k]
            h_recon = h_after[ref][0, -1].astype(np.float64)
            h_recon = h_recon - (proj_ref_v @ Vt_o) + (proj_mod @ Vt_o)
            h_test = h_after[ref].copy()
            h_test[0, -1] = h_recon.astype(np.float32)

            h_post = run_layers(engine, h_test, COMB_E, n_layers)
            _, tok, _ = predict_token(engine, tokenizer, h_post)
            m = ans.lower() in tok[0].strip().lower()
            print(f"    top-{k:3d}: {'✓' if m else '✗'} → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    # Inv 5: The irreducible representation — what's the minimum?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Irreducible Representation — Sign Flips in Answer Dims")
    print("=" * 80)

    # For each capital pair: which of the top-10 dims have sign flips?
    # Can we navigate by flipping ONLY those specific dims?
    for tgt in caps[1:]:
        ans = 'Berlin' if 'Germany' in tgt else 'Tokyo'
        s_r, l_r = phi_proj[ref]
        s_t, l_t = phi_proj[tgt]
        xor_10 = (s_r[:10] * s_t[:10]).astype(np.int8)
        flip_10 = np.where(xor_10 == -1)[0]

        print(f"\n  France → {tgt[15:21]}:")
        print(f"    Top-10 sign flips: {flip_10.tolist()} ({len(flip_10)} flips)")
        print(f"    Top-10 φ-levels (France): {l_r[:10].tolist()}")
        print(f"    Top-10 φ-levels (target): {l_t[:10].tolist()}")
        print(f"    Level diffs at flips: "
              f"{[int(l_t[d] - l_r[d]) for d in flip_10]}")

        # Float values in knowledge subspace
        print(f"    Float vals (France): {proj_h[ref][:10].round(2).tolist()}")
        print(f"    Float vals (target): {proj_h[tgt][:10].round(2).tolist()}")
        print(f"    Diffs: {(proj_h[tgt][:10] - proj_h[ref][:10]).round(2).tolist()}")

        # Navigate by flipping only the answer-dim signs
        proj_mod = proj_h[ref].copy()
        for d in flip_10:
            proj_mod[d] = -proj_mod[d]

        h_recon = h_after[ref][0, -1].astype(np.float64)
        h_recon = h_recon - (proj_h[ref] @ Vt_o) + (proj_mod @ Vt_o)
        h_test = h_after[ref].copy()
        h_test[0, -1] = h_recon.astype(np.float32)
        h_post = run_layers(engine, h_test, COMB_E, n_layers)
        _, tok, _ = predict_token(engine, tokenizer, h_post)
        print(f"    Flip only [{','.join(str(d) for d in flip_10)}]: → {tok[0]!r}")

        # Try individual dim flips
        for d in range(10):
            proj_single = proj_h[ref].copy()
            proj_single[d] = -proj_single[d]
            h_recon_s = h_after[ref][0, -1].astype(np.float64)
            h_recon_s = h_recon_s - (proj_h[ref] @ Vt_o) + (proj_single @ Vt_o)
            h_test_s = h_after[ref].copy()
            h_test_s[0, -1] = h_recon_s.astype(np.float32)
            h_post_s = run_layers(engine, h_test_s, COMB_E, n_layers)
            _, tok_s, _ = predict_token(engine, tokenizer, h_post_s)
            flip_marker = " ← FLIP" if d in flip_10 else ""
            print(f"    Flip dim {d}: → {tok_s[0]!r}{flip_marker}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary: Knowledge Subspace φ-Basis")
    print("=" * 80)
    print("""
  Key questions answered:
  1. How many sign flips in knowledge subspace? (vs 261-368 in raw 3584-d)
  2. Can sign flips in knowledge subspace change the answer?
  3. How many knowledge dims needed for answer navigation?
  4. What is the irreducible representation of "France → Germany"?
""")


if __name__ == '__main__':
    main()
