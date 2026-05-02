"""
Frontier 6d: Rank-1 Direction × φ-Level — The Irreducible Answer
==================================================================
F146: rank-1 of COMB delta = 3/3 correct (one direction captures answer)
F147: signs = structure, levels = content (answer in magnitudes, not signs)

Combined hypothesis: the irreducible answer representation is a
SINGLE φ-LEVEL along the rank-1 COMB direction. Each entity maps
to a different level (scalar) along this direction, and that scalar
IS the answer.

Tests:
  1. Extract rank-1 direction from COMB deltas via SVD
  2. Project each entity's COMB output onto this direction → scalar
  3. φ-encode the scalar → (sign, level) — is the sign shared? levels different?
  4. Navigate: change the scalar to target's value along rank-1 direction
  5. Predictability: can we get the rank-1 direction from a single reference?
  6. Cross-structure: does the rank-1 direction generalize?
"""

import sys, os, time
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI_CONST = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI_CONST)
K_SCALE = 1000


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def encode_phi(x):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    signs = np.sign(x).astype(np.int8)
    signs[signs == 0] = 1
    mags = np.abs(x) + 1e-45
    levels = np.round(K_SCALE * np.log(mags) / LOG_PHI).astype(np.int64)
    return signs, levels


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


def main():
    print("=" * 80)
    print("  Frontier 6d: Rank-1 Direction × φ-Level")
    print("=" * 80)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s")

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
    # Collect hidden states before/after COMB and final
    # ═══════════════════════════════════════════════════════════
    print("\n  Running forward passes...")
    h_before, h_after, h_fin, bl_pred = {}, {}, {}, {}
    deltas = {}
    for prompt, tids in working:
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, COMB_S)
        h_before[prompt] = h.copy()
        h = run_layers(engine, h, COMB_S, COMB_E)
        h_after[prompt] = h.copy()
        deltas[prompt] = (h_after[prompt] - h_before[prompt]).copy()
        h = run_layers(engine, h, COMB_E, n_layers)
        h_fin[prompt] = h.copy()
        _, tok, _ = predict_token(engine, tokenizer, h)
        bl_pred[prompt] = tok[0]
        print(f"    '{prompt}' → {tok[0]!r}")

    caps = [p for p, _ in working if 'capital' in p]
    divs = [p for p, _ in working if 'capital' not in p]

    # ═══════════════════════════════════════════════════════════
    # Inv 1: Extract rank-1 direction from COMB deltas
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 1: Rank-1 Direction of COMB Deltas")
    print("=" * 80)

    # SVD of capital deltas at last position
    delta_matrix = np.array([deltas[p][0, -1] for p in caps])  # (N_caps, 3584)
    U, S, Vt = np.linalg.svd(delta_matrix, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    print(f"  SVD of {len(caps)} capital deltas:")
    print(f"    S[0:5] = {S[:5].round(2)}")
    print(f"    Energy: rank-1={energy[0]*100:.1f}%, rank-2={energy[1]*100:.1f}%, "
          f"rank-3={energy[2]*100:.1f}%")

    d_rank1 = Vt[0]  # The dominant direction (3584-d)
    d_rank1_norm = d_rank1 / np.linalg.norm(d_rank1)

    # Also get per-prompt rank-1 directions
    for prompt in caps:
        delta_last = deltas[prompt][0, -1].astype(np.float64)
        _, s_p, vt_p = np.linalg.svd(delta_last[np.newaxis, :], full_matrices=False)
        d_prompt = vt_p[0]
        cos_vs_global = cosine(d_prompt, d_rank1)
        print(f"    '{prompt[15:21]}' rank-1 cos vs global: {cos_vs_global:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 2: Project onto rank-1 → scalar per entity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: Scalar Projection onto Rank-1 Direction")
    print("=" * 80)

    # Project COMB outputs (h_after) onto rank-1 direction
    print("\n  Last-position projection onto rank-1:")
    proj_scalars = {}
    for prompt, _ in working:
        h_last = h_after[prompt][0, -1].astype(np.float64)
        scalar = np.dot(h_last, d_rank1_norm)
        proj_scalars[prompt] = scalar
        s, l = encode_phi(scalar)
        print(f"    '{prompt[:30]}': scalar={scalar:.4f}  "
              f"φ-sign={int(s[0]):+d}  φ-level={int(l[0])}")

    # Delta projections
    print("\n  Delta projection onto rank-1:")
    delta_scalars = {}
    for prompt, _ in working:
        d_last = deltas[prompt][0, -1].astype(np.float64)
        scalar = np.dot(d_last, d_rank1_norm)
        delta_scalars[prompt] = scalar
        s, l = encode_phi(scalar)
        print(f"    '{prompt[:30]}': Δscalar={scalar:.4f}  "
              f"φ-sign={int(s[0]):+d}  φ-level={int(l[0])}")

    # ═══════════════════════════════════════════════════════════
    # Inv 3: Navigate by changing scalar along rank-1
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: Navigate by Changing Rank-1 Scalar")
    print("=" * 80)

    ref = caps[0]
    for tgt in caps[1:]:
        ref_scalar = proj_scalars[ref]
        tgt_scalar = proj_scalars[tgt]
        diff = tgt_scalar - ref_scalar

        s_r, l_r = encode_phi(ref_scalar)
        s_t, l_t = encode_phi(tgt_scalar)
        print(f"\n  France → {tgt[15:21]}:")
        print(f"    ref_scalar={ref_scalar:.4f} (φ-level={int(l_r[0])})")
        print(f"    tgt_scalar={tgt_scalar:.4f} (φ-level={int(l_t[0])})")
        print(f"    diff={diff:.4f}  level_diff={int(l_t[0] - l_r[0])}")

        # 3a: Shift France's h_after by the scalar difference along rank-1
        h_mod = h_after[ref].copy()
        h_mod[0, -1] = h_mod[0, -1].astype(np.float64) + diff * d_rank1_norm
        h_mod = h_mod.astype(np.float32)

        h_post = run_layers(engine, h_mod, COMB_E, n_layers)
        _, tok, _ = predict_token(engine, tokenizer, h_post)
        cos_t = cosine(h_post[0, -1], h_fin[tgt][0, -1])
        print(f"    3a (shift scalar): → {tok[0]!r} cos={cos_t:.4f}")

        # 3b: Replace France's projection with target's (full rank-1 swap)
        h_mod_b = h_after[ref].copy().astype(np.float64)
        # Remove France's rank-1 component, add target's
        h_mod_b[0, -1] -= ref_scalar * d_rank1_norm
        h_mod_b[0, -1] += tgt_scalar * d_rank1_norm
        h_mod_b = h_mod_b.astype(np.float32)

        h_post_b = run_layers(engine, h_mod_b, COMB_E, n_layers)
        _, tok_b, _ = predict_token(engine, tokenizer, h_post_b)
        cos_t_b = cosine(h_post_b[0, -1], h_fin[tgt][0, -1])
        print(f"    3b (replace rank-1): → {tok_b[0]!r} cos={cos_t_b:.4f}")

        # 3c: Use the DELTA rank-1 direction (per-prompt, not global)
        ref_delta = deltas[ref][0, -1].astype(np.float64)
        tgt_delta = deltas[tgt][0, -1].astype(np.float64)
        _, _, vt_ref = np.linalg.svd(ref_delta[np.newaxis, :], full_matrices=False)
        d_ref = vt_ref[0]

        # Project both deltas onto ref's rank-1 direction
        proj_r = np.dot(ref_delta, d_ref)
        proj_t = np.dot(tgt_delta, d_ref)

        # Shift: replace ref's rank-1 component of DELTA with target's
        new_delta = ref_delta - proj_r * d_ref + proj_t * d_ref
        h_mod_c = h_before[ref].copy().astype(np.float64)
        h_mod_c[0, -1] += new_delta
        h_mod_c = h_mod_c.astype(np.float32)

        h_post_c = run_layers(engine, h_mod_c, COMB_E, n_layers)
        _, tok_c, _ = predict_token(engine, tokenizer, h_post_c)
        cos_t_c = cosine(h_post_c[0, -1], h_fin[tgt][0, -1])
        print(f"    3c (delta rank-1 swap): → {tok_c[0]!r} cos={cos_t_c:.4f}")

        # 3d: Full delta replacement (oracle baseline)
        h_mod_d = h_before[ref].copy().astype(np.float64)
        h_mod_d[0, -1] += tgt_delta
        h_mod_d = h_mod_d.astype(np.float32)

        h_post_d = run_layers(engine, h_mod_d, COMB_E, n_layers)
        _, tok_d, _ = predict_token(engine, tokenizer, h_post_d)
        cos_t_d = cosine(h_post_d[0, -1], h_fin[tgt][0, -1])
        print(f"    3d (full delta oracle): → {tok_d[0]!r} cos={cos_t_d:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 4: All positions — do they all need navigation?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: Per-Position Rank-1 Analysis")
    print("=" * 80)

    for pos in range(5):
        delta_pos = np.array([deltas[p][0, pos] for p in caps])
        _, S_pos, Vt_pos = np.linalg.svd(delta_pos, full_matrices=False)
        e_pos = np.cumsum(S_pos ** 2) / np.sum(S_pos ** 2)
        d_pos = Vt_pos[0]

        # Project each capital onto this direction
        scalars = []
        for p in caps:
            s = np.dot(deltas[p][0, pos].astype(np.float64), d_pos)
            scalars.append(s)

        cos_vs_last = cosine(d_pos, d_rank1)
        print(f"  Position {pos}: rank-1 energy={e_pos[0]*100:.1f}%  "
              f"cos_vs_lastpos={cos_vs_last:.4f}  "
              f"scalars=[{', '.join(f'{s:.2f}' for s in scalars)}]")

    # ═══════════════════════════════════════════════════════════
    # Inv 5: Navigate ALL positions along rank-1
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Full Rank-1 Navigation (All Positions)")
    print("=" * 80)

    for tgt in caps[1:]:
        h_mod = h_before[ref].copy().astype(np.float64)

        for pos in range(h_mod.shape[1]):
            # Get rank-1 direction for this position from all capitals
            delta_pos = np.array([deltas[p][0, pos] for p in caps])
            _, _, Vt_pos = np.linalg.svd(delta_pos, full_matrices=False)
            d_pos = Vt_pos[0]

            # Project ref and target deltas
            proj_r = np.dot(deltas[ref][0, pos].astype(np.float64), d_pos)
            proj_t = np.dot(deltas[tgt][0, pos].astype(np.float64), d_pos)

            # Apply: h_before + ref_delta - ref_rank1 + tgt_rank1
            h_mod[0, pos] += deltas[ref][0, pos].astype(np.float64)
            h_mod[0, pos] -= proj_r * d_pos
            h_mod[0, pos] += proj_t * d_pos

        h_mod = h_mod.astype(np.float32)
        h_post = run_layers(engine, h_mod, COMB_E, n_layers)
        _, tok, _ = predict_token(engine, tokenizer, h_post)
        cos_t = cosine(h_post[0, -1], h_fin[tgt][0, -1])
        print(f"  France → {tgt[15:21]} (all-pos rank-1): → {tok[0]!r} cos={cos_t:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 6: The φ-level representation of the answer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 6: φ-Level as Entity Identifier")
    print("=" * 80)

    # If the rank-1 scalar is the answer, its φ-level is the
    # irreducible representation
    print("\n  Rank-1 scalar φ-encoding:")
    print(f"  {'Prompt':30s}  {'scalar':>10s}  {'sign':>5s}  {'level':>7s}  {'pred':>8s}")
    print("  " + "─" * 65)
    for prompt, _ in working:
        scalar = proj_scalars[prompt]
        s, l = encode_phi(scalar)
        print(f"  {prompt[:30]:30s}  {scalar:10.4f}  {int(s[0]):+5d}  {int(l[0]):7d}  "
              f"{bl_pred[prompt]:>8s}")

    # Level differences between capitals
    print("\n  Level differences (capitals):")
    for i, p1 in enumerate(caps):
        for p2 in caps[i+1:]:
            s1, l1 = encode_phi(proj_scalars[p1])
            s2, l2 = encode_phi(proj_scalars[p2])
            ldiff = int(l2[0] - l1[0])
            sdiff = "SAME" if s1[0] == s2[0] else "FLIP"
            ratio = proj_scalars[p2] / proj_scalars[p1] if proj_scalars[p1] != 0 else float('inf')
            print(f"    {p1[15:21]} → {p2[15:21]}: "
                  f"level_diff={ldiff:+5d}  sign={sdiff}  ratio={ratio:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 7: Can we predict rank-1 direction from one reference?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 7: Rank-1 Direction Predictability")
    print("=" * 80)

    # Use France as the reference, compute rank-1 from France alone
    ref_delta_last = deltas[ref][0, -1].astype(np.float64)
    d_france = ref_delta_last / (np.linalg.norm(ref_delta_last) + 1e-20)

    # How well does France's delta direction match the global rank-1?
    cos_vs_global = cosine(d_france, d_rank1_norm)
    print(f"  France delta direction vs global rank-1: cos={cos_vs_global:.4f}")

    # Navigate using France's direction only
    print("\n  Navigate using France's delta direction:")
    for tgt in caps[1:]:
        h_mod = h_after[ref].copy().astype(np.float64)
        # Project difference onto France's direction
        ref_proj = np.dot(h_after[ref][0, -1].astype(np.float64), d_france)
        tgt_proj = np.dot(h_after[tgt][0, -1].astype(np.float64), d_france)
        diff = tgt_proj - ref_proj
        h_mod[0, -1] += diff * d_france
        h_mod = h_mod.astype(np.float32)

        h_post = run_layers(engine, h_mod, COMB_E, n_layers)
        _, tok, _ = predict_token(engine, tokenizer, h_post)
        cos_t = cosine(h_post[0, -1], h_fin[tgt][0, -1])
        print(f"    France → {tgt[15:21]}: → {tok[0]!r} cos={cos_t:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 8: Cross-structure test — does rank-1 direction work for diverse?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 8: Cross-Structure Rank-1 Direction")
    print("=" * 80)

    # Compute rank-1 of diverse deltas
    if len(divs) >= 2:
        delta_div = np.array([deltas[p][0, -1] for p in divs])
        _, S_div, Vt_div = np.linalg.svd(delta_div, full_matrices=False)
        e_div = np.cumsum(S_div ** 2) / np.sum(S_div ** 2)
        d_div = Vt_div[0]
        cos_cap_div = cosine(d_rank1, d_div)
        print(f"  Diverse rank-1 energy: {e_div[0]*100:.1f}%")
        print(f"  cos(capital rank-1, diverse rank-1): {cos_cap_div:.4f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  The rank-1 COMB direction × φ-level:
  - Each entity → a scalar projection → a single φ-level
  - If navigation works: answer = one φ-level along one direction
  - That's the irreducible representation: sign + level + direction
""")


if __name__ == '__main__':
    main()
