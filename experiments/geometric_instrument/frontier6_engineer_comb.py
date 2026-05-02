"""
Frontier 6: Engineering the COMB Zone
=======================================
F145 found the COMB zone (L10-L20) is a Content Separator using push-pull
interference. Can we ENGINEER a replacement?

Three tests, most aggressive first:

  Test A: SKIP — Remove L10-L20 entirely. Feed L9 output directly to L21.
          If extraction layers can still produce correct answers, the
          content separation is redundant.

  Test B: CACHE — Per-structure-class cached COMB zone transformation.
          Record the NET change (h_out_L20 - h_in_L10) for a reference
          prompt of each structure class, then replay it for new prompts
          of the same class.

  Test C: LOW-RANK — Approximate the COMB zone's net transformation as
          a rank-k projection. If rank@90% = 4-5 (F145), a rank-5
          approximation might suffice.

Fail-fast: no fallbacks, no graceful degradation.
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


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def run_layers(engine, h, start, end):
    """Run layers [start, end) on hidden state h (shape [1, seq, dim])."""
    for li in range(start, end):
        layer = engine.layers[li]
        attn = layer.attention
        mlp = layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        sl = h.shape[1]

        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke = np.repeat(K, hpk, axis=1)
        Ve = np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        attn_out = phi_linear(attn.W_o, ao)
        h_pa = h + attn_out

        nm = rms_norm(h_pa, mlp.norm_weight)
        gate_pre = phi_linear(mlp.W_gate, nm)
        up = phi_linear(mlp.W_up, nm)
        gate_act = phi_silu(gate_pre)
        intermediate = gate_act * up
        mlp_out = phi_linear(mlp.W_down, intermediate)

        h = h_pa + mlp_out

    return h


def predict_token(engine, tokenizer, h):
    """Given final hidden state h [1, seq, dim], predict top-5 tokens at last pos."""
    final_norm_w = None
    for attr in ['final_norm_weight', 'norm_weight', 'ln_f_weight']:
        if hasattr(engine, attr):
            final_norm_w = getattr(engine, attr)
            break
    if final_norm_w is None:
        final_norm_w = engine.final_norm.weight if hasattr(engine, 'final_norm') else None
    if final_norm_w is None:
        raise RuntimeError("Cannot find final norm weight")

    h_last = rms_norm(h[:, -1:, :], final_norm_w)
    lm_w = engine.lm_head_weight if hasattr(engine, 'lm_head_weight') else engine.lm_head.weight
    logits = phi_linear(lm_w, h_last)
    logits = logits[0, 0]

    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def main():
    print("=" * 80, flush=True)
    print("  Frontier 6: Engineering the COMB Zone", flush=True)
    print("=" * 80, flush=True)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s ({n_layers} layers)", flush=True)

    # Test prompts: 3 capital (same structure) + 4 diverse
    prompts = [
        'The capital of France is',
        'The capital of Germany is',
        'The capital of Japan is',
        'I really love eating pizza',
        'Please help me find this',
        'Once upon a time there',
        'How does the engine work',
    ]
    expected_answers = ['Paris', 'Berlin', 'Tokyo', None, None, None, None]

    working = []
    for prompt, exp in zip(prompts, expected_answers):
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            working.append((prompt, tids, exp))
    print(f"  Using {len(working)} prompts", flush=True)

    # COMB zone boundaries — test multiple skip ranges
    COMB_CONFIGS = [
        ('L10-L20 (full COMB)', 10, 21),
        ('L10-L15 (first half)', 10, 16),
        ('L15-L20 (second half)', 15, 21),
        ('L8-L22 (wide skip)', 8, 22),
        ('L12-L18 (narrow core)', 12, 19),
    ]

    # ═══════════════════════════════════════════════════════════
    # Baseline: Full model forward pass
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  BASELINE: Full model forward pass", flush=True)
    print("=" * 80, flush=True)

    baseline_results = {}
    baseline_h_states = {}  # h at various layer boundaries

    for prompt, tids, exp in working:
        h = engine.embedding(tids)[np.newaxis, :, :]

        # Save hidden states at key boundaries
        states = {'emb': h.copy()}
        for li in range(n_layers):
            h = run_layers(engine, h, li, li + 1)
            states[f'L{li}'] = h.copy()

        top5_idx, top5_tok, logits = predict_token(engine, tokenizer, h)
        baseline_results[prompt] = {
            'top1': top5_tok[0], 'top5': top5_tok, 'top5_idx': top5_idx,
            'logits': logits, 'expected': exp,
        }
        baseline_h_states[prompt] = states

        match = "✓" if exp and exp.strip().lower() in top5_tok[0].strip().lower() else ("?" if exp is None else "✗")
        print(f"  {match} '{prompt}' → {top5_tok[0]!r} (top5: {top5_tok})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # TEST A: SKIP — Remove COMB zone layers entirely
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  TEST A: SKIP — Remove COMB zone layers entirely", flush=True)
    print("=" * 80, flush=True)

    for config_name, skip_start, skip_end in COMB_CONFIGS:
        print(f"\n  --- Skip {config_name} ---", flush=True)
        correct, total = 0, 0

        for prompt, tids, exp in working:
            # Run L0 through skip_start-1
            h_pre = baseline_h_states[prompt][f'L{skip_start - 1}']

            # Skip directly to skip_end
            h_post = run_layers(engine, h_pre, skip_end, n_layers)

            top5_idx, top5_tok, logits = predict_token(engine, tokenizer, h_post)

            # Compare to baseline
            bl = baseline_results[prompt]
            h_bl = baseline_h_states[prompt][f'L{n_layers - 1}']
            cos_final = cosine(h_post[0, -1], h_bl[0, -1])

            match_bl = top5_tok[0].strip() == bl['top1'].strip()
            if exp:
                match_exp = exp.strip().lower() in top5_tok[0].strip().lower()
                total += 1
                if match_exp:
                    correct += 1
                tag = "✓" if match_exp else "✗"
            else:
                tag = "=" if match_bl else "≠"

            print(f"    {tag} '{prompt}' → {top5_tok[0]!r} "
                  f"(baseline: {bl['top1']!r}) cos_final={cos_final:.4f}", flush=True)

        if total > 0:
            print(f"  Score: {correct}/{total} capital-fact correct", flush=True)

    # ═══════════════════════════════════════════════════════════
    # TEST B: CACHE — Per-structure cached COMB transformation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  TEST B: CACHE — Per-structure cached COMB transformation", flush=True)
    print("=" * 80, flush=True)

    # Use "The capital of France is" as reference for capital prompts
    # Use "I really love eating pizza" as reference for diverse prompts
    # Cache the NET CHANGE through the COMB zone
    comb_start, comb_end = 10, 21

    # Compute reference deltas
    ref_capital = 'The capital of France is'
    ref_diverse = 'I really love eating pizza'

    def compute_comb_delta(prompt_name):
        """Net hidden state change from L(comb_start-1) output to L(comb_end-1) output."""
        h_before = baseline_h_states[prompt_name][f'L{comb_start - 1}']
        h_after = baseline_h_states[prompt_name][f'L{comb_end - 1}']
        return h_after - h_before

    delta_capital = compute_comb_delta(ref_capital)
    delta_diverse = compute_comb_delta(ref_diverse)

    print(f"\n  Reference deltas computed:", flush=True)
    print(f"    Capital ref (France): ||delta|| = {np.linalg.norm(delta_capital):.1f}", flush=True)
    print(f"    Diverse ref (pizza):  ||delta|| = {np.linalg.norm(delta_diverse):.1f}", flush=True)

    # Test B1: Apply reference delta to same-structure prompts
    print(f"\n  --- B1: Exact cached delta ---", flush=True)
    correct, total = 0, 0

    for prompt, tids, exp in working:
        # Choose reference delta based on structure class
        is_capital = 'capital' in prompt
        ref_delta = delta_capital if is_capital else delta_diverse

        # Apply cached delta instead of running COMB layers
        h_pre = baseline_h_states[prompt][f'L{comb_start - 1}']
        h_cached = h_pre + ref_delta

        # Run remaining layers
        h_post = run_layers(engine, h_cached, comb_end, n_layers)

        top5_idx, top5_tok, logits = predict_token(engine, tokenizer, h_post)

        bl = baseline_results[prompt]
        h_bl = baseline_h_states[prompt][f'L{n_layers - 1}']
        cos_final = cosine(h_post[0, -1], h_bl[0, -1])

        if exp:
            match_exp = exp.strip().lower() in top5_tok[0].strip().lower()
            total += 1
            if match_exp:
                correct += 1
            tag = "✓" if match_exp else "✗"
        else:
            tag = "=" if top5_tok[0].strip() == bl['top1'].strip() else "≠"

        ref_name = "France" if is_capital else "pizza"
        print(f"    {tag} '{prompt}' → {top5_tok[0]!r} "
              f"(ref={ref_name}, baseline={bl['top1']!r}) cos={cos_final:.4f}", flush=True)

    if total > 0:
        print(f"  Score: {correct}/{total} capital-fact correct", flush=True)

    # Test B2: Per-prompt cached delta (leave-one-out for same-structure)
    print(f"\n  --- B2: Leave-one-out cached delta (capital prompts) ---", flush=True)
    capital_prompts = [(p, t, e) for p, t, e in working if 'capital' in p]
    correct, total = 0, 0

    for i, (prompt, tids, exp) in enumerate(capital_prompts):
        # Use AVERAGE delta from OTHER capital prompts
        other_deltas = []
        for j, (op, ot, oe) in enumerate(capital_prompts):
            if j != i:
                other_deltas.append(compute_comb_delta(op))
        avg_delta = np.mean(other_deltas, axis=0)

        h_pre = baseline_h_states[prompt][f'L{comb_start - 1}']
        h_cached = h_pre + avg_delta

        h_post = run_layers(engine, h_cached, comb_end, n_layers)
        top5_idx, top5_tok, logits = predict_token(engine, tokenizer, h_post)

        bl = baseline_results[prompt]
        cos_final = cosine(h_post[0, -1], baseline_h_states[prompt][f'L{n_layers - 1}'][0, -1])

        match_exp = exp.strip().lower() in top5_tok[0].strip().lower() if exp else False
        total += 1
        if match_exp:
            correct += 1
        tag = "✓" if match_exp else "✗"

        print(f"    {tag} '{prompt}' → {top5_tok[0]!r} "
              f"(avg-other, baseline={bl['top1']!r}) cos={cos_final:.4f}", flush=True)

    print(f"  Score: {correct}/{total} capital-fact correct (leave-one-out)", flush=True)

    # Test B3: Position-specific cached delta
    print(f"\n  --- B3: Per-position cached delta ---", flush=True)
    correct, total = 0, 0

    for prompt, tids, exp in working:
        is_capital = 'capital' in prompt

        # Position-specific delta from reference
        ref = ref_capital if is_capital else ref_diverse
        h_ref_before = baseline_h_states[ref][f'L{comb_start - 1}']
        h_ref_after = baseline_h_states[ref][f'L{comb_end - 1}']

        h_pre = baseline_h_states[prompt][f'L{comb_start - 1}']

        # Per-position: scale delta by the ratio of norms
        delta = h_ref_after - h_ref_before
        h_cached = h_pre.copy()
        for pos in range(h_pre.shape[1]):
            pre_norm = np.linalg.norm(h_pre[0, pos])
            ref_pre_norm = np.linalg.norm(h_ref_before[0, pos])
            scale = pre_norm / (ref_pre_norm + 1e-20)
            h_cached[0, pos] = h_pre[0, pos] + delta[0, pos] * scale

        h_post = run_layers(engine, h_cached, comb_end, n_layers)
        top5_idx, top5_tok, logits = predict_token(engine, tokenizer, h_post)

        bl = baseline_results[prompt]
        cos_final = cosine(h_post[0, -1], baseline_h_states[prompt][f'L{n_layers - 1}'][0, -1])

        if exp:
            match_exp = exp.strip().lower() in top5_tok[0].strip().lower()
            total += 1
            if match_exp:
                correct += 1
            tag = "✓" if match_exp else "✗"
        else:
            tag = "=" if top5_tok[0].strip() == bl['top1'].strip() else "≠"

        print(f"    {tag} '{prompt}' → {top5_tok[0]!r} "
              f"(baseline={bl['top1']!r}) cos={cos_final:.4f}", flush=True)

    if total > 0:
        print(f"  Score: {correct}/{total} capital-fact correct (norm-scaled)", flush=True)

    # ═══════════════════════════════════════════════════════════
    # TEST C: LOW-RANK — Rank-k approximation of COMB zone
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  TEST C: LOW-RANK — Approximate COMB zone net change", flush=True)
    print("=" * 80, flush=True)

    # Collect net deltas from all prompts at last position
    all_deltas_last = []
    all_h_pre_last = []
    for prompt, tids, exp in working:
        delta = compute_comb_delta(prompt)
        all_deltas_last.append(delta[0, -1])  # last position
        all_h_pre_last.append(baseline_h_states[prompt][f'L{comb_start - 1}'][0, -1])

    delta_matrix = np.array(all_deltas_last)  # [n_prompts, 3584]
    h_pre_matrix = np.array(all_h_pre_last)   # [n_prompts, 3584]

    # SVD of the delta matrix
    U, S, Vt = np.linalg.svd(delta_matrix, full_matrices=False)
    print(f"\n  SVD of COMB net-delta (last pos, {len(working)} prompts):", flush=True)
    energy = np.cumsum(S ** 2)
    total_energy = energy[-1]
    for k in range(min(len(S), 7)):
        pct = 100 * energy[k] / total_energy
        print(f"    rank {k+1}: S={S[k]:.2f}, cumulative energy={pct:.1f}%", flush=True)

    # Try rank-k approximations
    for rank in [1, 2, 3, 5, 7]:
        if rank > len(S):
            continue
        print(f"\n  --- C: Rank-{rank} approximation ---", flush=True)

        # Build rank-k basis from the delta SVD
        basis = Vt[:rank]  # [rank, 3584]

        correct, total_c = 0, 0
        for prompt, tids, exp in working:
            h_pre = baseline_h_states[prompt][f'L{comb_start - 1}'].copy()
            true_delta = compute_comb_delta(prompt)

            # Project true delta onto rank-k basis (per position)
            h_approx = h_pre.copy()
            for pos in range(h_pre.shape[1]):
                d = true_delta[0, pos]
                # Project: d_approx = sum_i (d · v_i) v_i
                coeffs = basis @ d  # [rank]
                d_approx = coeffs @ basis  # [3584]
                h_approx[0, pos] = h_pre[0, pos] + d_approx

            h_post = run_layers(engine, h_approx, comb_end, n_layers)
            top5_idx, top5_tok, logits = predict_token(engine, tokenizer, h_post)

            bl = baseline_results[prompt]
            cos_final = cosine(h_post[0, -1], baseline_h_states[prompt][f'L{n_layers - 1}'][0, -1])

            # Reconstruction quality
            recon_cos = cosine(
                h_approx[0, -1],
                baseline_h_states[prompt][f'L{comb_end - 1}'][0, -1]
            )

            if exp:
                match_exp = exp.strip().lower() in top5_tok[0].strip().lower()
                total_c += 1
                if match_exp:
                    correct += 1
                tag = "✓" if match_exp else "✗"
            else:
                tag = "=" if top5_tok[0].strip() == bl['top1'].strip() else "≠"

            print(f"    {tag} '{prompt}' → {top5_tok[0]!r} "
                  f"(baseline={bl['top1']!r}) cos_final={cos_final:.4f} "
                  f"recon_cos={recon_cos:.4f}", flush=True)

        if total_c > 0:
            print(f"  Score: {correct}/{total_c} capital-fact correct", flush=True)

    # ═══════════════════════════════════════════════════════════
    # TEST C2: LOW-RANK without oracle delta (predict from input)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  TEST C2: LOW-RANK — Predict delta from input (no oracle)", flush=True)
    print("=" * 80, flush=True)

    # Can we predict the COMB delta from the pre-COMB hidden state?
    # Learn a linear map: h_pre → delta using leave-one-out
    print(f"\n  Learning linear map h_pre → delta (leave-one-out):", flush=True)

    for pos_name, pos_idx in [('last', -1), ('BOS', 0)]:
        print(f"\n  --- Position: {pos_name} ---", flush=True)

        # Collect data
        H_pre = np.array([
            baseline_h_states[p][f'L{comb_start - 1}'][0, pos_idx]
            for p, _, _ in working
        ])  # [n, 3584]
        D_true = np.array([
            compute_comb_delta(p)[0, pos_idx]
            for p, _, _ in working
        ])  # [n, 3584]

        for i, (prompt, tids, exp) in enumerate(working):
            # Leave-one-out: train on all EXCEPT i
            mask = np.ones(len(working), dtype=bool)
            mask[i] = False
            H_train = H_pre[mask]
            D_train = D_true[mask]

            # Simplest predictor: average delta from training set
            d_pred_mean = np.mean(D_train, axis=0)

            # Better: nearest neighbor in h_pre space
            dists = np.array([cosine(H_pre[i], H_train[j]) for j in range(len(H_train))])
            nn_idx = np.argmax(dists)
            d_pred_nn = D_train[nn_idx]

            # How good is the prediction?
            true_d = D_true[i]
            cos_mean = cosine(d_pred_mean, true_d)
            cos_nn = cosine(d_pred_nn, true_d)

            print(f"    '{prompt}': cos(mean_pred, true)={cos_mean:.4f}, "
                  f"cos(nn_pred, true)={cos_nn:.4f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Summary: Frontier 6 — Engineering the COMB Zone", flush=True)
    print("=" * 80, flush=True)
    print("""
  Test A (SKIP):  Can extraction layers work without content separation?
  Test B (CACHE): Can we replay a cached transformation for same-structure?
  Test C (RANK):  Can we approximate the delta in a low-rank subspace?
  Test C2 (PRED): Can we predict the delta from the input?

  The fail-fast verdict: which approaches preserve correct predictions?
""", flush=True)


if __name__ == '__main__':
    main()
