#!/usr/bin/env python3
"""
DC 305 Frontier Experiments — Three Open Questions from DC 303 §10

Q1. Does iterative IRD gravity converge to the transformer's hidden-state trajectory?
    Method: run gravity in a loop with small alpha; track food-alignment per iteration.
    Compare curve shape to transformer's per-layer food-alignment (from Qwen2-1.5B).

Q2. Geometric softmax competition — does relative weighting beat additive sum
    when conflicting context words are present?
    Method: test cookie with ['recipe', 'login'] simultaneously under both schemes.

Q3. Bidirectional mutual gravity on a full sentence — does it produce a coherent
    representation?
    Method: N-body simulation where every word attracts every other.
    Measure cluster formation and compare same polysemous word across two contexts.

Usage:
    python dc305_frontier_experiments.py           # all three
    python dc305_frontier_experiments.py --q1      # Q1 only (needs torch)
    python dc305_frontier_experiments.py --q2      # Q2 only
    python dc305_frontier_experiments.py --q3      # Q3 only
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

FOOD_REF_WORDS = ['bread', 'soup', 'cake', 'rice', 'pasta', 'egg', 'milk', 'cheese']
SMALL_MODEL    = "Qwen/Qwen2-1.5B-Instruct"


# ── Shared helpers ────────────────────────────────────────────────────────────

def ird_food_align(lcm, proj):
    """Cosine similarity of *proj* with the IRD food-reference centroid."""
    vecs = []
    for fw in FOOD_REF_WORDS:
        try:
            p, _ = lcm._get_proj(fw)
            vecs.append(p.astype(np.float64))
        except RuntimeError:
            pass
    centroid = np.mean(vecs, axis=0)
    cn = centroid / (np.linalg.norm(centroid) + 1e-20)
    pn = proj.astype(np.float64)
    pn = pn / (np.linalg.norm(pn) + 1e-20)
    return float(np.dot(pn, cn))


def apply_gravity_step(p, context_projs, alpha, falloff='exp'):
    """Single gravity correction step on raw projection vector *p*."""
    correction = np.zeros_like(p)
    for p_ctx in context_projs:
        diff = p_ctx - p
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            continue
        if falloff == 'inv_sq':
            w = 1.0 / (dist * dist)
        elif falloff == 'inv':
            w = 1.0 / dist
        else:  # 'exp'
            w = np.exp(-dist)
        correction += alpha * w * diff
    p_new = p + correction
    norm = np.linalg.norm(p_new)
    return p_new / (norm + 1e-20)


def apply_softmax_gravity_step(p, context_projs, alpha):
    """
    Gravity with softmax-normalised attention weights (Q2 variant).
    Competitive: the context word geometrically closest to p dominates.
    """
    if not context_projs:
        return p
    # Cosine affinities
    pn = p / (np.linalg.norm(p) + 1e-20)
    affinities = []
    for p_ctx in context_projs:
        cn = p_ctx / (np.linalg.norm(p_ctx) + 1e-20)
        affinities.append(float(np.dot(pn, cn)))
    affs = np.array(affinities)
    affs_sm = np.exp(affs - affs.max())  # numerically stable
    affs_sm /= affs_sm.sum()
    correction = np.zeros_like(p)
    for i, p_ctx in enumerate(context_projs):
        diff = p_ctx - p
        correction += alpha * affs_sm[i] * diff
    p_new = p + correction
    return p_new / (np.linalg.norm(p_new) + 1e-20)


def get_context_projs(lcm, context_words):
    """Return IRD projections for the subset of context_words in vocabulary."""
    out = []
    valid = []
    for w in context_words:
        try:
            p, _ = lcm._get_proj(w)
            out.append(p.astype(np.float64))
            valid.append(w)
        except RuntimeError:
            pass
    return out, valid


# ── Q1 ────────────────────────────────────────────────────────────────────────

def run_q1(lcm):
    """
    Does iterative IRD gravity converge in the same direction as the
    transformer's layer-by-layer contextualisation?

    We compare:
      - food-alignment of 'cookie' per IRD iteration (with culinary / HTTP context)
      - food-alignment of 'cookie' per transformer layer (Qwen2-1.5B)
    """
    print("\n" + "="*65)
    print("Q1 — Iterative IRD Gravity vs Transformer Layer Trajectory")
    print("="*65)

    ALPHA    = 0.15   # small alpha → gradual, many iterations
    MAX_ITER = 60
    CONV_EPS = 1e-6

    test_cases = [
        ('cookie', ['bake', 'flour', 'recipe'],   'culinary'),
        ('cookie', ['browser', 'session', 'login'], 'HTTP'),
        ('bass',   ['guitar', 'solo', 'play'],    'music'),
        ('bass',   ['fish',   'catch', 'river'],  'aquatic'),
    ]

    print(f"\n  alpha={ALPHA}, max_iter={MAX_ITER}")
    print(f"\n  {'Case':<24s}  iter_conv  final_align  Δ_from_native")

    ird_curves = {}
    for word, ctx, label in test_cases:
        ctx_projs, valid_ctx = get_context_projs(lcm, ctx)
        if not ctx_projs:
            print(f"  {word}+{label}: no context words in vocab")
            continue

        p0, _ = lcm._get_proj(word)
        p = p0.astype(np.float64)
        align_native = ird_food_align(lcm, p)
        curve = [align_native]

        conv_iter = MAX_ITER
        for i in range(MAX_ITER):
            p_new = apply_gravity_step(p, ctx_projs, ALPHA)
            delta = float(np.linalg.norm(p_new - p))
            p = p_new
            curve.append(ird_food_align(lcm, p))
            if delta < CONV_EPS:
                conv_iter = i + 1
                break

        key = f"{word}_{label}"
        ird_curves[key] = curve
        delta_from_native = curve[-1] - align_native
        direction = "↑ food" if delta_from_native > 0 else "↓ food"
        print(f"  {word:<8s} [{label:<8s}]  "
              f"i={conv_iter:<4d}  {curve[-1]:+.4f}      "
              f"{delta_from_native:+.4f} {direction}  ctx={valid_ctx}")

    # Now extract transformer trajectories
    print(f"\n  Extracting transformer per-layer food-alignment...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SMALL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            SMALL_MODEL, dtype=torch.float32, device_map='cpu')
        model.eval()
        n_layers = model.config.num_hidden_layers

        emb_matrix = model.model.embed_tokens.weight.detach().numpy()

        food_ref_embs = []
        for fw in FOOD_REF_WORDS:
            ids = tok.encode(' ' + fw, add_special_tokens=False)
            if ids:
                food_ref_embs.append(emb_matrix[ids[0]])
        food_centroid_15b = np.mean(food_ref_embs, axis=0)
        food_cn_15b = food_centroid_15b / (np.linalg.norm(food_centroid_15b) + 1e-20)

        def tf_food_align(h):
            hn = h / (np.linalg.norm(h) + 1e-20)
            return float(np.dot(hn, food_cn_15b))

        transformer_curves = {}
        tf_cases = [
            ('cookie', 'bake a cookie with flour',       'culinary'),
            ('cookie', 'clear the browser cookie',       'HTTP'),
            ('bass',   'play bass guitar in the band',   'music'),
            ('bass',   'catch the large bass near shore','aquatic'),
        ]

        for target, sentence, label in tf_cases:
            inputs = tok(sentence, return_tensors='pt')
            tokens = [tok.decode([t]) for t in inputs['input_ids'][0]]
            # Find target token position
            target_ids_a = tok.encode(' ' + target, add_special_tokens=False)
            target_ids_b = tok.encode(target, add_special_tokens=False)
            target_pos = None
            for i, t in enumerate(inputs['input_ids'][0]):
                if t.item() in (target_ids_a[:1] + target_ids_b[:1]):
                    target_pos = i
                    break
            if target_pos is None:
                print(f"  Warning: '{target}' not found in '{sentence}'")
                continue

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            curve = [tf_food_align(hs[0, target_pos, :].numpy())
                     for hs in out.hidden_states]
            key = f"{target}_{label}"
            transformer_curves[key] = curve

        # Compare IRD iteration curves to transformer layer curves
        print(f"\n  Comparison: IRD iter curve vs Transformer layer curve")
        print(f"  (shape correlation tells us if they follow the same trajectory)\n")
        print(f"  {'Case':<22s}  IRD_shape_r  TF_direction  IRD_direction  Match?")
        print("  " + "-"*65)

        from scipy.stats import pearsonr

        for word, label in [('cookie', 'culinary'), ('cookie', 'HTTP'),
                             ('bass', 'music'), ('bass', 'aquatic')]:
            ird_key = f"{word}_{label}"
            tf_key  = f"{word}_{label}"
            if ird_key not in ird_curves or tf_key not in transformer_curves:
                continue
            ic = ird_curves[ird_key]
            tc = transformer_curves[tf_key]
            # Resample IRD curve to same length as TF curve
            n = len(tc)
            indices = np.linspace(0, len(ic)-1, n).astype(int)
            ic_resampled = [ic[i] for i in indices]
            r, pval = pearsonr(ic_resampled, tc)
            tf_dir  = "↑" if tc[-1] > tc[0]  else "↓"
            ird_dir = "↑" if ic[-1] > ic[0]  else "↓"
            match   = "✓ AGREE" if tf_dir == ird_dir else "✗ DIFFER"
            print(f"  {word:<8s} [{label:<8s}]  r={r:+.3f}       "
                  f"TF:{tf_dir}    IRD:{ird_dir}    {match}")

        del model

    except ImportError:
        print("  SKIP transformer comparison: torch/transformers not available")

    # Print sample IRD convergence curves (first 10 + last value)
    print(f"\n  IRD convergence profiles (food-align per iteration):")
    for word, ctx, label in test_cases:
        key = f"{word}_{label}"
        if key not in ird_curves:
            continue
        curve = ird_curves[key]
        sample = curve[:10] + (["..."] if len(curve) > 12 else []) + [curve[-1]]
        vals = [f"{v:+.4f}" if isinstance(v, float) else v for v in sample]
        print(f"  {word:<8s} [{label:<8s}]:  {' '.join(vals)}")


# ── Q2 ────────────────────────────────────────────────────────────────────────

def run_q2(lcm):
    """
    Geometric softmax competition vs additive sum gravity.

    Tests with CONFLICTING context: both culinary and HTTP words present.
    The softmax version should let the closer cluster dominate.
    The additive sum version should partially cancel.
    """
    print("\n" + "="*65)
    print("Q2 — Softmax Competition vs Additive Sum (Conflicting Context)")
    print("="*65)

    ALPHA = 0.5

    test_configs = [
        {
            'word':    'cookie',
            'ctx_a':   ['recipe', 'bake', 'flour'],
            'ctx_b':   ['browser', 'session', 'login'],
            'ctx_mix': ['recipe', 'login'],
            'label':   'cookie — culinary vs HTTP',
        },
        {
            'word':    'bass',
            'ctx_a':   ['guitar', 'solo', 'play'],
            'ctx_b':   ['fish', 'catch', 'river'],
            'ctx_mix': ['guitar', 'fish'],
            'label':   'bass — music vs aquatic',
        },
        {
            'word':    'bank',
            'ctx_a':   ['river', 'water', 'stream'],
            'ctx_b':   ['money', 'deposit', 'loan'],
            'ctx_mix': ['river', 'money'],
            'label':   'bank — geography vs finance',
        },
    ]

    for cfg in test_configs:
        word    = cfg['word']
        ctx_a   = cfg['ctx_a']
        ctx_b   = cfg['ctx_b']
        ctx_mix = cfg['ctx_mix']

        p0, _ = lcm._get_proj(word)
        p0 = p0.astype(np.float64)
        align0 = ird_food_align(lcm, p0)

        projs_a,   valid_a   = get_context_projs(lcm, ctx_a)
        projs_b,   valid_b   = get_context_projs(lcm, ctx_b)
        projs_mix, valid_mix = get_context_projs(lcm, ctx_mix)

        def score(method, projs):
            if not projs:
                return p0.copy()
            if method == 'additive':
                return apply_gravity_step(p0, projs, ALPHA)
            else:
                return apply_softmax_gravity_step(p0, projs, ALPHA)

        print(f"\n  ── {cfg['label']} ──────────────────────────────────────")
        print(f"  Native food-align: {align0:+.4f}")
        print(f"\n  {'Context':<20s}  {'Method':<10s}  food_align  Δ_from_native")
        print("  " + "-"*60)

        for ctx_label, projs, valid in [
            (f"clean_A ({valid_a})",   projs_a,   valid_a),
            (f"clean_B ({valid_b})",   projs_b,   valid_b),
            (f"mixed   ({valid_mix})", projs_mix, valid_mix),
        ]:
            for method in ['additive', 'softmax']:
                p_new = score(method, projs)
                align = ird_food_align(lcm, p_new)
                delta = align - align0
                bar   = "█" * int(abs(delta) * 80) + ("↑" if delta > 0 else "↓")
                print(f"  {ctx_label:<20s}  {method:<10s}  {align:+.4f}    "
                      f"{delta:+.4f}  {bar}")

        # Key test: mixed context — which method better separates the two senses?
        projs_a_m, _ = get_context_projs(lcm, ctx_mix[:1])   # first = sense A
        projs_b_m, _ = get_context_projs(lcm, ctx_mix[1:])   # second = sense B

        add_a  = ird_food_align(lcm, score('additive', projs_a_m))
        add_b  = ird_food_align(lcm, score('additive', projs_b_m))
        sft_a  = ird_food_align(lcm, score('softmax',  projs_a_m))
        sft_b  = ird_food_align(lcm, score('softmax',  projs_b_m))
        add_mix = ird_food_align(lcm, score('additive', projs_mix))
        sft_mix = ird_food_align(lcm, score('softmax',  projs_mix))

        sep_add = abs(add_a - add_b)
        sep_sft = abs(sft_a - sft_b)

        print(f"\n  Separation (single-word context): additive={sep_add:.4f}  "
              f"softmax={sep_sft:.4f}")
        winner = "softmax" if sep_sft > sep_add else "additive"
        print(f"  Mixed-context convergence:        additive={add_mix:+.4f}  "
              f"softmax={sft_mix:+.4f}")
        print(f"  → Better single-sense separation: {winner}")


# ── Q3 ────────────────────────────────────────────────────────────────────────

def run_q3(lcm):
    """
    Bidirectional mutual gravity on a full sentence.
    Every word attracts every other word. Iterate until convergence.

    Tests:
      - Do words in the same sentence cluster together?
      - Does the polysemous word end up in different positions
        in two different contextual sentences?
    """
    print("\n" + "="*65)
    print("Q3 — Bidirectional Mutual Gravity (N-Body Sentence Simulation)")
    print("="*65)

    ALPHA    = 0.08
    MAX_ITER = 150
    CONV_EPS = 1e-7

    test_sentences = [
        ('cookie_culinary', ['cookie', 'recipe', 'bake', 'flour', 'butter', 'sugar']),
        ('cookie_http',     ['cookie', 'browser', 'session', 'login', 'token']),
        ('bass_music',      ['bass', 'guitar', 'solo', 'band', 'play', 'song']),
        ('bass_aquatic',    ['bass', 'fish', 'catch', 'river', 'water', 'swim']),
    ]

    final_positions = {}

    for label, words in test_sentences:
        # Get native projections for words in vocabulary
        valid = {}
        for w in words:
            try:
                p, _ = lcm._get_proj(w)
                valid[w] = p.astype(np.float64)
            except RuntimeError:
                pass
        if not valid:
            continue

        word_list = list(valid.keys())
        positions = {w: valid[w].copy() for w in word_list}

        conv_iter = MAX_ITER
        for iter_n in range(MAX_ITER):
            new_pos  = {}
            max_delta = 0.0
            for w in word_list:
                p    = positions[w]
                ctx  = [positions[other] for other in word_list if other != w]
                p_new = apply_gravity_step(p, ctx, ALPHA)
                new_pos[w] = p_new
                max_delta = max(max_delta, float(np.linalg.norm(p_new - p)))
            positions = new_pos
            if max_delta < CONV_EPS:
                conv_iter = iter_n + 1
                break

        final_positions[label] = {w: positions[w] for w in word_list}

        # Measure intra-sentence coherence
        word_list2 = word_list
        n = len(word_list2)
        cosines = []
        for i in range(n):
            for j in range(i+1, n):
                p1 = positions[word_list2[i]]
                p2 = positions[word_list2[j]]
                cos = float(np.dot(p1, p2) /
                            (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-20))
                cosines.append(cos)
        mean_cos = np.mean(cosines) if cosines else 0.0

        print(f"\n  Sentence [{label}]: {word_list}")
        print(f"  Converged at iter={conv_iter}, "
              f"intra-sentence mean cos={mean_cos:.4f}")

        # Show pairwise cosines between content pairs
        print(f"  {'Pair':<28s}  native_cos  final_cos  Δ")
        for i in range(min(n, 4)):
            for j in range(i+1, min(n, 5)):
                w1, w2 = word_list2[i], word_list2[j]
                p1_n, _ = lcm._get_proj(w1)
                p2_n, _ = lcm._get_proj(w2)
                cos_native = float(np.dot(p1_n, p2_n) /
                                   (np.linalg.norm(p1_n) * np.linalg.norm(p2_n) + 1e-20))
                p1_f = final_positions[label][w1]
                p2_f = final_positions[label][w2]
                cos_final = float(np.dot(p1_f, p2_f) /
                                  (np.linalg.norm(p1_f) * np.linalg.norm(p2_f) + 1e-20))
                delta = cos_final - cos_native
                bar   = "→ tighter" if delta > 0.005 else ("→ looser" if delta < -0.005 else "→ same")
                print(f"  {w1:<14s}↔{w2:<14s}  {cos_native:+.4f}     "
                      f"{cos_final:+.4f}   {delta:+.4f}  {bar}")

    # Cross-sentence comparison: same polysemous word in two contexts
    print(f"\n  ── Cross-context position of 'cookie' ─────────────────────")
    if 'cookie_culinary' in final_positions and 'cookie_http' in final_positions:
        p_cul = final_positions['cookie_culinary'].get('cookie')
        p_htp = final_positions['cookie_http'].get('cookie')
        if p_cul is not None and p_htp is not None:
            # Compare to native
            p_nat, _ = lcm._get_proj('cookie')
            p_nat = p_nat.astype(np.float64)
            cos_nat_cul = float(np.dot(p_nat, p_cul) /
                                (np.linalg.norm(p_nat) * np.linalg.norm(p_cul) + 1e-20))
            cos_nat_htp = float(np.dot(p_nat, p_htp) /
                                (np.linalg.norm(p_nat) * np.linalg.norm(p_htp) + 1e-20))
            cos_cul_htp = float(np.dot(p_cul, p_htp) /
                                (np.linalg.norm(p_cul) * np.linalg.norm(p_htp) + 1e-20))
            food_nat = ird_food_align(lcm, p_nat)
            food_cul = ird_food_align(lcm, p_cul)
            food_htp = ird_food_align(lcm, p_htp)
            print(f"  cookie_native:  food_align={food_nat:+.4f}")
            print(f"  cookie_culinary food_align={food_cul:+.4f}  "
                  f"cos(nat, cul)={cos_nat_cul:+.4f}")
            print(f"  cookie_http:    food_align={food_htp:+.4f}  "
                  f"cos(nat, htp)={cos_nat_htp:+.4f}")
            print(f"  cos(culinary, http) = {cos_cul_htp:+.4f}")
            separation = abs(food_cul - food_htp)
            print(f"\n  Food-alignment separation = {separation:.4f}")
            if separation > 0.05:
                print(f"  ✓ Bidirectional gravity SEPARATES the two senses of 'cookie'")
            else:
                print(f"  ✗ Bidirectional gravity does NOT separate the senses")

    print(f"\n  ── Cross-context position of 'bass' ──────────────────────")
    if 'bass_music' in final_positions and 'bass_aquatic' in final_positions:
        p_mus = final_positions['bass_music'].get('bass')
        p_aq  = final_positions['bass_aquatic'].get('bass')
        if p_mus is not None and p_aq is not None:
            p_nat, _ = lcm._get_proj('bass')
            p_nat = p_nat.astype(np.float64)
            cos_mus_aq = float(np.dot(p_mus, p_aq) /
                               (np.linalg.norm(p_mus) * np.linalg.norm(p_aq) + 1e-20))
            food_nat = ird_food_align(lcm, p_nat.astype(np.float64))
            food_mus = ird_food_align(lcm, p_mus)
            food_aq  = ird_food_align(lcm, p_aq)
            print(f"  bass_native:   food_align={food_nat:+.4f}")
            print(f"  bass_music:    food_align={food_mus:+.4f}")
            print(f"  bass_aquatic:  food_align={food_aq:+.4f}")
            print(f"  cos(music, aquatic) = {cos_mus_aq:+.4f}")
            separation = abs(food_mus - food_aq)
            print(f"\n  Food-alignment separation = {separation:.4f}")
            if separation > 0.02:
                print(f"  ✓ Bidirectional gravity SEPARATES the two senses of 'bass'")
            else:
                print(f"  ✗ Bidirectional gravity does NOT separate the senses")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q1', action='store_true')
    parser.add_argument('--q2', action='store_true')
    parser.add_argument('--q3', action='store_true')
    args = parser.parse_args()
    run_all = not args.q1 and not args.q2 and not args.q3

    print("Loading LCM...")
    lcm = build_lcm()

    if args.q1 or run_all:
        run_q1(lcm)

    if args.q2 or run_all:
        run_q2(lcm)

    if args.q3 or run_all:
        run_q3(lcm)
