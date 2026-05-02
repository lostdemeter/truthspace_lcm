"""
Frontier 6b: φ-Basis Analysis of the COMB Zone
=================================================
F146: cache fails because answer lives in 1.7% of delta signal.
In float space = tiny perturbation. In φ-basis = sparse sign flips.

Tests:
  1. Encode COMB deltas in φ-basis (signs + levels)
  2. XOR signs between same-structure prompts → answer-carrying dims
  3. Minimum flips for correct answer
  4. Reconstruct answer from sign-flip dims only
  5. Overlap of answer dims across targets
  6. φ-basis irreducibility vs float
"""

import sys, os, time
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
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


def main():
    print("=" * 80)
    print("  Frontier 6b: φ-Basis Analysis of the COMB Zone")
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
    # Inv 1: φ-Encode COMB outputs
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 1: φ-Encode COMB Outputs (last pos)")
    print("=" * 80)

    phi_h = {}
    for prompt, _ in working:
        s, l = encode_phi(h_after[prompt][0, -1])
        phi_h[prompt] = (s, l)
        print(f"  '{prompt[:30]}': +:{np.sum(s==1)} -:{np.sum(s==-1)} "
              f"lvl=[{l.min()},{l.max()}] std={np.std(l.astype(float)):.0f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 2: Sign XOR — Where do prompts differ?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: Sign XOR Between Prompts")
    print("=" * 80)

    print("\n  --- Capital pairs (h_after signs) ---")
    for i, p1 in enumerate(caps):
        for p2 in caps[i+1:]:
            s1, l1 = phi_h[p1]
            s2, l2 = phi_h[p2]
            xor = (s1 * s2).astype(np.int8)
            nf = np.sum(xor == -1)
            fm = (xor == -1)
            ld_flip = np.mean(np.abs(l1[fm].astype(float) - l2[fm].astype(float))) if np.any(fm) else 0
            ld_same = np.mean(np.abs(l1[~fm].astype(float) - l2[~fm].astype(float)))
            print(f"  {p1[15:21]} vs {p2[15:21]}: {nf} flips ({100*nf/len(xor):.1f}%) "
                  f"lvl_diff@flip={ld_flip:.0f} @same={ld_same:.0f}")

    print("\n  --- Cross-structure (France vs diverse) ---")
    for p2 in divs:
        s1, _ = phi_h[caps[0]]
        s2, _ = phi_h[p2]
        nf = np.sum((s1 * s2) == -1)
        print(f"  France vs '{p2[:25]}': {nf} flips ({100*nf/len(s1):.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # Inv 3: Reconstruct answer from sign flips
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: Reconstruct Answer From Sign Flips")
    print("=" * 80)

    ref = caps[0]
    for tgt in caps[1:]:
        s_r, l_r = phi_h[ref]
        s_t, l_t = phi_h[tgt]
        xor = (s_r * s_t).astype(np.int8)
        flips = np.where(xor == -1)[0]
        print(f"\n  France → {tgt[15:21]}: {len(flips)} sign flips")

        # 3a: flip signs only (keep France levels)
        rs = s_r.copy(); rs[flips] *= -1
        h_r = h_after[ref].copy()
        h_r[0, -1] = decode_phi(rs, l_r).astype(np.float32)
        h_p = run_layers(engine, h_r, COMB_E, n_layers)
        _, tok, _ = predict_token(engine, tokenizer, h_p)
        cos_t = cosine(h_p[0, -1], h_fin[tgt][0, -1])
        print(f"    3a (signs only):  → {tok[0]!r} cos={cos_t:.4f}")

        # 3b: flip signs + copy target levels at flip dims
        rs2 = s_r.copy(); rl2 = l_r.copy()
        rs2[flips] *= -1; rl2[flips] = l_t[flips]
        h_r2 = h_after[ref].copy()
        h_r2[0, -1] = decode_phi(rs2, rl2).astype(np.float32)
        h_p2 = run_layers(engine, h_r2, COMB_E, n_layers)
        _, tok2, _ = predict_token(engine, tokenizer, h_p2)
        cos_t2 = cosine(h_p2[0, -1], h_fin[tgt][0, -1])
        print(f"    3b (signs+levels): → {tok2[0]!r} cos={cos_t2:.4f}")

        # 3c: all positions
        h_r3 = h_after[ref].copy()
        for pos in range(h_r3.shape[1]):
            sr, lr = encode_phi(h_after[ref][0, pos])
            st, lt = encode_phi(h_after[tgt][0, pos])
            fp = np.where((sr * st) == -1)[0]
            sr[fp] *= -1; lr[fp] = lt[fp]
            h_r3[0, pos] = decode_phi(sr, lr).astype(np.float32)
        h_p3 = run_layers(engine, h_r3, COMB_E, n_layers)
        _, tok3, _ = predict_token(engine, tokenizer, h_p3)
        cos_t3 = cosine(h_p3[0, -1], h_fin[tgt][0, -1])
        print(f"    3c (all positions): → {tok3[0]!r} cos={cos_t3:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 4: Minimum flips for correct answer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: Minimum Flips for Correct Answer")
    print("=" * 80)

    for tgt in caps[1:]:
        s_r, l_r = phi_h[ref]
        s_t, l_t = phi_h[tgt]
        flips = np.where((s_r * s_t) == -1)[0]
        # Rank by level magnitude
        rank = np.argsort(np.abs(l_r[flips].astype(float)))[::-1]

        ans = 'Berlin' if 'Germany' in tgt else 'Tokyo'
        print(f"\n  France → {tgt[15:21]} ({len(flips)} flips, target: {ans}):")

        for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, len(flips)]:
            if k > len(flips): k = len(flips)
            top_k = flips[rank[:k]]
            rs = s_r.copy(); rl = l_r.copy()
            rs[top_k] *= -1; rl[top_k] = l_t[top_k]
            h_r = h_after[ref].copy()
            h_r[0, -1] = decode_phi(rs, rl).astype(np.float32)
            h_p = run_layers(engine, h_r, COMB_E, n_layers)
            _, tok, _ = predict_token(engine, tokenizer, h_p)
            m = ans.lower() in tok[0].strip().lower()
            print(f"    k={k:4d}: {'✓' if m else '✗'} → {tok[0]!r}")
            if k == len(flips): break

    # ═══════════════════════════════════════════════════════════
    # Inv 5: Overlap of answer-carrying dims
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Overlap of Answer-Carrying Dimensions")
    print("=" * 80)

    s_r, _ = phi_h[ref]
    flip_sets = {}
    for tgt in caps[1:]:
        s_t, _ = phi_h[tgt]
        flip_sets[tgt] = set(np.where((s_r * s_t) == -1)[0].tolist())

    if len(flip_sets) >= 2:
        tgts = list(flip_sets.keys())
        a, b = flip_sets[tgts[0]], flip_sets[tgts[1]]
        inter = a & b
        union = a | b
        print(f"  France→{tgts[0][15:21]}: {len(a)} dims")
        print(f"  France→{tgts[1][15:21]}: {len(b)} dims")
        print(f"  Intersection: {len(inter)}, Union: {len(union)}")
        print(f"  Jaccard: {len(inter)/len(union):.3f}" if union else "")

        # How many shared flips also flip for diverse?
        s_d, _ = phi_h[divs[0]]
        div_flips = set(np.where((s_r * s_d) == -1)[0].tolist())
        shared_in_div = inter & div_flips
        print(f"  Of {len(inter)} shared, {len(shared_in_div)} also flip for diverse")
        print(f"  → {len(inter) - len(shared_in_div)} capital-SPECIFIC")

    # ═══════════════════════════════════════════════════════════
    # Inv 6: φ-basis irreducibility vs float
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 6: φ-Basis vs Float Irreducibility")
    print("=" * 80)

    for tgt in caps[1:]:
        h_r = h_after[ref][0, -1].astype(np.float64)
        h_t = h_after[tgt][0, -1].astype(np.float64)
        fd = np.abs(h_r - h_t)
        fds = np.sort(fd)[::-1]
        fe = np.cumsum(fds ** 2)
        ft = fe[-1]
        k90 = int(np.searchsorted(fe, 0.9 * ft)) + 1
        k99 = int(np.searchsorted(fe, 0.99 * ft)) + 1

        s1, _ = phi_h[ref]; s2, _ = phi_h[tgt]
        nf = np.sum((s1 * s2) == -1)

        print(f"\n  France → {tgt[15:21]}:")
        print(f"    Float: {k90} dims for 90%, {k99} dims for 99%")
        print(f"    φ-basis: {nf} sign flips (binary)")
        print(f"    Ratio: float-90%/φ-flips = {k90/nf:.2f}×")

        # Sign-only reconstruction quality
        rs = s1.copy()
        fm = (s1 * s2) == -1
        rs[fm] *= -1
        h_sign_only = decode_phi(rs, encode_phi(h_r)[1])
        cos_sign = cosine(h_sign_only, h_t)
        cos_float = cosine(h_r, h_t)
        print(f"    cos(ref, tgt) float = {cos_float:.4f}")
        print(f"    cos(sign-flipped ref, tgt) = {cos_sign:.4f}")

    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  If φ-basis sign flips are sparse and sufficient for answer reconstruction,
  then the COMB zone's content separation IS an XOR operation in φ-space:
  flip a small set of signs to navigate from one answer to another.

  This would mean: the 1.7% float difference = a handful of sign flips
  in the irreducible φ-representation.
""")


if __name__ == '__main__':
    main()
