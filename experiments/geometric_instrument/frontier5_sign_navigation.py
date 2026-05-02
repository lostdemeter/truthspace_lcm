"""
Frontier 5: Sign-Space Navigation
====================================
Previous frontiers measured in float space (cosine similarity).
DC 253/254/255 established that:
  - Signs carry 4x more info than magnitudes at near-zero
  - Gate codes are token-UNIVERSAL (base-collapse RMS=0.0085)
  - Gate transitions propagate at 1/φ per layer
  - The 4-state gate IS the computation mechanism

This experiment measures in φ-SPACE:
  1. Sign agreement (weighted Hamming) vs float cosine across prompts
  2. Sign stability through layers — does it follow the 1/φ speed limit?
  3. Gate code predictability — are they truly token-universal on our prompts?
  4. Cross-structure similarity in SIGN space (not float space)
  5. Can sign patterns alone predict the output token?
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

LOG_PHI = np.log((1 + np.sqrt(5)) / 2)  # ≈ 0.481


def decode_weight(w):
    return w.decode() if isinstance(w, PhiEncoded) else w


def to_signs(x):
    """Extract sign pattern from float array. Preserve -0 vs +0."""
    s = np.sign(x).astype(np.int8)
    s[s == 0] = 1
    return s


def to_4state(gate_pre_silu):
    """Classify gate activations into 4 states per DC 253."""
    # Boundaries at ±log(φ) ≈ ±0.481
    codes = np.zeros_like(gate_pre_silu, dtype=np.int8)
    codes[gate_pre_silu < -LOG_PHI] = 0   # CONTRACT (-1)
    codes[(gate_pre_silu >= -LOG_PHI) & (gate_pre_silu < 0)] = 1  # PRESERVE- (-0)
    codes[(gate_pre_silu >= 0) & (gate_pre_silu < LOG_PHI)] = 2   # PRESERVE+ (+0)
    codes[gate_pre_silu >= LOG_PHI] = 3   # EXPAND (+1)
    return codes


def sign_agreement(a, b):
    """Fraction of dimensions where signs agree."""
    return float(np.mean(a == b))


def weighted_sign_agreement(a, b, levels_a):
    """Sign agreement weighted by inverse magnitude (DC 254 §4.1).
    Dimensions near zero get higher weight."""
    weights = 1.0 / (1.0 + np.abs(levels_a).astype(np.float32))
    agree = (a == b).astype(np.float32)
    return float(np.sum(agree * weights) / np.sum(weights))


def float_cosine(a, b):
    """Standard float cosine similarity."""
    return float(np.dot(a.ravel(), b.ravel()) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 80, flush=True)
    print("  Frontier 5: Sign-Space Navigation", flush=True)
    print("=" * 80, flush=True)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s", flush=True)

    prompts = [
        'The capital of France is',
        'The capital of Germany is',
        'The capital of Japan is',
        'I really love eating pizza',
        'Please help me find this',
        'Once upon a time there',
        'How does the engine work',
    ]

    working = []
    for prompt in prompts:
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            working.append((prompt, tids))
    print(f"  Using {len(working)} N=5 prompts", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Capture: Hidden states, signs, gate codes at every layer
    # ═══════════════════════════════════════════════════════════
    print("\n  Capturing hidden states and gate codes...", end="", flush=True)

    all_data = {}  # prompt -> {layer -> {h, signs, levels, gate_codes, gate_pre}}
    for prompt, tids in working:
        h = engine.embedding(tids)[np.newaxis, :, :]
        layer_data = {}

        # Store embedding layer
        layer_data[-1] = {
            'h': h[0].copy(),
            'signs': to_signs(h[0]),
        }

        for li in range(n_layers):
            layer = engine.layers[li]
            attn = layer.attention
            mlp = layer.mlp
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            sl = h.shape[1]

            # Attention
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

            # MLP — capture gate activations BEFORE SiLU
            nm = rms_norm(h_pa, mlp.norm_weight)
            gate_pre = phi_linear(mlp.W_gate, nm)  # pre-SiLU gate
            up = phi_linear(mlp.W_up, nm)
            mlp_out = phi_linear(mlp.W_down, phi_silu(gate_pre) * up)

            h = h_pa + mlp_out

            # Encode hidden state into φ-space
            phi_enc = PhiEncoded.encode(h[0])

            layer_data[li] = {
                'h': h[0].copy(),
                'signs': phi_enc.signs.copy(),
                'levels': phi_enc.exponents.copy(),
                'gate_pre': gate_pre[0].copy(),  # [seq, mlp_dim]
                'gate_codes': to_4state(gate_pre[0]),
            }

        all_data[prompt] = layer_data
        gc.collect()

    print(" done", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 1: Sign Agreement vs Float Cosine (cross-prompt)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Investigation 1: Sign Agreement vs Float Cosine (cross-prompt)", flush=True)
    print("=" * 80, flush=True)

    prompt_names = [p for p, _ in working]

    # Compare across all prompt pairs at each layer, at the LAST position
    print("\n  Last position (prediction position):", flush=True)
    print(f"  {'Layer':>6} | {'Float cos':>10} {'Sign agree':>10} {'Wtd sign':>10} | {'Float cos':>10} {'Sign agree':>10}", flush=True)
    print(f"  {'':>6} | {'(same-struct)':>10} {'(same-struct)':>10} {'(same-struct)':>10} | {'(cross-struct)':>10} {'(cross-struct)':>10}", flush=True)

    capital_prompts = [p for p in prompt_names if 'capital' in p]
    diverse_prompts = [p for p in prompt_names if 'capital' not in p]

    for li in [-1, 0, 1, 2, 3, 5, 10, 15, 20, 23, 26, 27]:
        # Same-structure (capital vs capital)
        same_fcos, same_sagr, same_wsgr = [], [], []
        for i in range(len(capital_prompts)):
            for j in range(i + 1, len(capital_prompts)):
                p1, p2 = capital_prompts[i], capital_prompts[j]
                d1, d2 = all_data[p1][li], all_data[p2][li]
                same_fcos.append(float_cosine(d1['h'][-1], d2['h'][-1]))
                same_sagr.append(sign_agreement(d1['signs'][-1], d2['signs'][-1]))
                if 'levels' in d1:
                    same_wsgr.append(weighted_sign_agreement(
                        d1['signs'][-1], d2['signs'][-1], d1['levels'][-1]))
                else:
                    same_wsgr.append(same_sagr[-1])

        # Cross-structure (capital vs diverse)
        cross_fcos, cross_sagr = [], []
        for p1 in capital_prompts:
            for p2 in diverse_prompts:
                d1, d2 = all_data[p1][li], all_data[p2][li]
                cross_fcos.append(float_cosine(d1['h'][-1], d2['h'][-1]))
                cross_sagr.append(sign_agreement(d1['signs'][-1], d2['signs'][-1]))

        lbl = f"L{li:3d}" if li >= 0 else " emb"
        print(f"  {lbl:>6} | {np.mean(same_fcos):10.4f} {np.mean(same_sagr):10.4f} "
              f"{np.mean(same_wsgr):10.4f} | {np.mean(cross_fcos):10.4f} "
              f"{np.mean(cross_sagr):10.4f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: Sign Transition Rate — Does it follow 1/φ?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Investigation 2: Sign Transition Rate Through Layers", flush=True)
    print("=" * 80, flush=True)

    phi_inv = 1 / ((1 + np.sqrt(5)) / 2)  # 1/φ ≈ 0.618

    print(f"\n  Sign flip rate per layer (fraction of dims that change sign):", flush=True)
    print(f"  1/φ = {phi_inv:.4f}", flush=True)
    print(f"  {'Layer':>6} | {'BOS(p0)':>8} {'pos1':>8} {'pos2':>8} {'pos3':>8} {'Last':>8} | {'Mean':>8}", flush=True)

    # Average across prompts
    for li in range(n_layers):
        if li == 0:
            prev_key = -1  # embedding
        else:
            prev_key = li - 1

        rates_by_pos = [[] for _ in range(5)]
        for prompt in prompt_names:
            prev_signs = all_data[prompt][prev_key]['signs']
            curr_signs = all_data[prompt][li]['signs']
            for pos in range(5):
                flip_rate = 1.0 - sign_agreement(prev_signs[pos], curr_signs[pos])
                rates_by_pos[pos].append(flip_rate)

        mean_rates = [np.mean(r) for r in rates_by_pos]
        overall_mean = np.mean(mean_rates)
        if li in [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 23, 26, 27]:
            print(f"  L{li:4d} | {mean_rates[0]:8.4f} {mean_rates[1]:8.4f} "
                  f"{mean_rates[2]:8.4f} {mean_rates[3]:8.4f} {mean_rates[4]:8.4f} | "
                  f"{overall_mean:8.4f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Gate Code Universality — Are They Token-Independent?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Investigation 3: Gate Code Universality (DC 255 §3.2 Validation)", flush=True)
    print("=" * 80, flush=True)

    # At each layer, compare gate codes across prompts at the LAST position
    # If gate codes are token-universal, they should be nearly identical
    print(f"\n  Gate code agreement between prompt pairs at last position:", flush=True)
    print(f"  {'Layer':>6} | {'Same-struct':>11} {'Cross-struct':>12} | {'C%':>6} {'P-%':>6} {'P+%':>6} {'X%':>6}", flush=True)

    for li in range(n_layers):
        # Gate code agreement
        same_agree, cross_agree = [], []
        for i in range(len(capital_prompts)):
            for j in range(i + 1, len(capital_prompts)):
                g1 = all_data[capital_prompts[i]][li]['gate_codes'][-1]
                g2 = all_data[capital_prompts[j]][li]['gate_codes'][-1]
                same_agree.append(float(np.mean(g1 == g2)))
        for p1 in capital_prompts:
            for p2 in diverse_prompts:
                g1 = all_data[p1][li]['gate_codes'][-1]
                g2 = all_data[p2][li]['gate_codes'][-1]
                cross_agree.append(float(np.mean(g1 == g2)))

        # Gate state distribution (average across all prompts)
        all_codes = np.concatenate([all_data[p][li]['gate_codes'][-1:] for p in prompt_names])
        c_pct = 100 * np.mean(all_codes == 0)
        pn_pct = 100 * np.mean(all_codes == 1)
        pp_pct = 100 * np.mean(all_codes == 2)
        x_pct = 100 * np.mean(all_codes == 3)

        if li in [0, 1, 2, 3, 5, 10, 15, 18, 20, 23, 25, 27]:
            print(f"  L{li:4d} | {np.mean(same_agree):11.4f} {np.mean(cross_agree):12.4f} | "
                  f"{c_pct:5.1f}% {pn_pct:5.1f}% {pp_pct:5.1f}% {x_pct:5.1f}%", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: BOS Sign Pattern — Universal?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Investigation 4: BOS Sign Pattern Universality", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Sign agreement at BOS (pos 0) across ALL prompt pairs:", flush=True)
    print(f"  {'Layer':>6} | {'Sign agree':>10} {'Wtd sign':>10} {'Float cos':>10}", flush=True)

    for li in [-1, 0, 1, 2, 3, 5, 10, 15, 20, 23, 26, 27]:
        agrees, wt_agrees, fcoses = [], [], []
        for i in range(len(prompt_names)):
            for j in range(i + 1, len(prompt_names)):
                d1, d2 = all_data[prompt_names[i]][li], all_data[prompt_names[j]][li]
                agrees.append(sign_agreement(d1['signs'][0], d2['signs'][0]))
                fcoses.append(float_cosine(d1['h'][0], d2['h'][0]))
                if 'levels' in d1:
                    wt_agrees.append(weighted_sign_agreement(
                        d1['signs'][0], d2['signs'][0], d1['levels'][0]))
                else:
                    wt_agrees.append(agrees[-1])

        lbl = f"L{li:3d}" if li >= 0 else " emb"
        print(f"  {lbl:>6} | {np.mean(agrees):10.4f} {np.mean(wt_agrees):10.4f} "
              f"{np.mean(fcoses):10.4f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Can Signs Predict the Output Token?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Investigation 5: Sign-Only Output Prediction", flush=True)
    print("=" * 80, flush=True)

    # The question: if we ONLY have the sign pattern of the final hidden state
    # (no magnitudes), can we still predict the correct token?
    # Method: compute logits using signs × 1.0 (unit magnitude) instead of real h
    fnw = decode_weight(engine.final_norm_weight)

    facts = {
        'France':  ('The capital of France is', ' Paris'),
        'Japan':   ('The capital of Japan is', ' Tokyo'),
        'Germany': ('The capital of Germany is', ' Berlin'),
    }

    diverse_tests = [
        ('I really love eating pizza', None),
        ('Once upon a time there', None),
        ('How does the engine work', None),
    ]

    # Get baselines
    for i, (prompt, _) in enumerate(diverse_tests):
        h = all_data[prompt][27]['h'][-1:]  # last pos, last layer
        normed = rms_norm(h[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        diverse_tests[i] = (prompt, int(np.argmax(logits)))

    print("\n  Method A: Full float hidden state (baseline):", flush=True)
    for country, (prompt, answer) in facts.items():
        h = all_data[prompt][27]['h'][-1:]
        normed = rms_norm(h[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        ans_tid = tokenizer.encode(answer)[0]
        rank = int(np.sum(logits > logits[ans_tid]))
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {country:>8}: {ok}", flush=True)

    for prompt, real_tok in diverse_tests:
        h = all_data[prompt][27]['h'][-1:]
        normed = rms_norm(h[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        rank = int(np.sum(logits > logits[real_tok]))
        ok = "✓" if rank == 0 else f"rank={rank}"
        real_word = tokenizer.decode([real_tok])
        print(f"    '{prompt[:30]}' → '{real_word}' {ok}", flush=True)

    print("\n  Method B: Sign-only (signs × 1.0, no magnitudes):", flush=True)
    for country, (prompt, answer) in facts.items():
        signs = all_data[prompt][27]['signs'][-1:].astype(np.float32)
        normed = rms_norm(signs[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        ans_tid = tokenizer.encode(answer)[0]
        rank = int(np.sum(logits > logits[ans_tid]))
        pred = tokenizer.decode([int(np.argmax(logits))])
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {country:>8}: {ok} (pred='{pred}')", flush=True)

    for prompt, real_tok in diverse_tests:
        signs = all_data[prompt][27]['signs'][-1:].astype(np.float32)
        normed = rms_norm(signs[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        rank = int(np.sum(logits > logits[real_tok]))
        pred = tokenizer.decode([int(np.argmax(logits))])
        ok = "✓" if rank == 0 else f"rank={rank}"
        real_word = tokenizer.decode([real_tok])
        print(f"    '{prompt[:30]}' → pred='{pred}' real='{real_word}' {ok}", flush=True)

    print("\n  Method C: Sign × φ^(level/128) (full φ-space reconstruction):", flush=True)
    for country, (prompt, answer) in facts.items():
        signs = all_data[prompt][27]['signs'][-1:]
        levels = all_data[prompt][27]['levels'][-1:]
        phi = (1 + np.sqrt(5)) / 2
        reconstructed = signs.astype(np.float64) * (phi ** (levels.astype(np.float64) / 128.0))
        reconstructed = reconstructed.astype(np.float32)
        normed = rms_norm(reconstructed[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        ans_tid = tokenizer.encode(answer)[0]
        rank = int(np.sum(logits > logits[ans_tid]))
        ok = "✓" if rank == 0 else f"rank={rank}"
        # Also measure reconstruction error
        real_h = all_data[prompt][27]['h'][-1:]
        recon_cos = float_cosine(reconstructed, real_h)
        print(f"    {country:>8}: {ok} (recon cos={recon_cos:.6f})", flush=True)

    for prompt, real_tok in diverse_tests:
        signs = all_data[prompt][27]['signs'][-1:]
        levels = all_data[prompt][27]['levels'][-1:]
        phi = (1 + np.sqrt(5)) / 2
        reconstructed = signs.astype(np.float64) * (phi ** (levels.astype(np.float64) / 128.0))
        reconstructed = reconstructed.astype(np.float32)
        normed = rms_norm(reconstructed[np.newaxis, :, :], fnw)
        logits = engine.lm_head(normed)[0, 0, :]
        rank = int(np.sum(logits > logits[real_tok]))
        ok = "✓" if rank == 0 else f"rank={rank}"
        real_word = tokenizer.decode([real_tok])
        print(f"    '{prompt[:30]}' → real='{real_word}' {ok} ", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 6: Sign-Space Cross-Structure Convergence
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Investigation 6: Cross-Structure Convergence — Sign vs Float", flush=True)
    print("=" * 80, flush=True)

    # Key question: In sign space, do different prompt structures
    # converge FASTER or SLOWER than in float space?
    # And specifically: at the positions where float cosine was low,
    # is sign agreement actually high?

    print(f"\n  Per-position cross-structure metrics (France vs pizza prompt):", flush=True)
    p1 = 'The capital of France is'
    p2 = 'I really love eating pizza'
    print(f"  {'Layer':>6} | {'pos':>3} | {'Float cos':>10} {'Sign agree':>10} {'Wtd sign':>10} {'Gate agree':>10}", flush=True)

    for li in [0, 1, 3, 5, 10, 15, 20, 27]:
        d1, d2 = all_data[p1][li], all_data[p2][li]
        for pos in [0, 2, 4]:  # BOS, middle, last
            fc = float_cosine(d1['h'][pos], d2['h'][pos])
            sa = sign_agreement(d1['signs'][pos], d2['signs'][pos])
            if 'levels' in d1:
                wa = weighted_sign_agreement(d1['signs'][pos], d2['signs'][pos], d1['levels'][pos])
                ga = float(np.mean(d1['gate_codes'][pos] == d2['gate_codes'][pos]))
            else:
                wa = sa
                ga = 0.0
            print(f"  L{li:4d} | {pos:3d} | {fc:10.4f} {sa:10.4f} {wa:10.4f} {ga:10.4f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Summary: Sign-Space Navigation", flush=True)
    print("=" * 80, flush=True)
    print("""
  KEY QUESTIONS ANSWERED:
  1. Sign agreement vs float cosine — which is more informative for
     cross-structure similarity?
  2. Sign transition rates — do they follow the 1/φ speed limit?
  3. Gate codes — are they truly token-universal on our test prompts?
  4. BOS sign convergence — does sign space show faster convergence?
  5. Sign-only prediction — can we predict output from signs alone?
  6. Per-position sign vs float — where do they diverge?

  IMPLICATION: If sign agreement is high where float cosine is low,
  then we've been measuring the wrong thing. The φ-space representation
  may already contain the navigable structure we need.
""", flush=True)


if __name__ == '__main__':
    main()
