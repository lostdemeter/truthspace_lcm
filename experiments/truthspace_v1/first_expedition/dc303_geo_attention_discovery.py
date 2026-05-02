#!/usr/bin/env python3
"""
DC 303 Probe — Geometric Attention Discovery

Tests the hypothesis: transformer attention IS context gravity.

Two-part experiment:

  Part 1 — Static pairwise similarity
    For word pairs, compare:
      a. IRD geometric proximity: cos(p_i, p_j) from our LCMIndex
      b. L0 attention logit: Q(embed_i)·K(embed_j) / sqrt(d) from Qwen2-7B φ-decoded Q,K
    Measure Pearson r — do they agree?

  Part 2 — In-context hidden state shift
    Run Qwen2-1.5B on polysemous test sentences, extract hidden states.
    Compare:
      - h("cookie" | "cookie recipe")  vs  h("cookie" | "cookie login")
    at L22 (the knowledge extraction layer).
    Then compare to what our context_correct_proj() does to p("cookie").
    If transformer hidden state shift ≈ IRD gravity correction, the hypothesis holds.

Usage:
    python dc303_geo_attention_discovery.py            # full experiment
    python dc303_geo_attention_discovery.py --part1    # static only (no model)
    python dc303_geo_attention_discovery.py --part2    # context shift only
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from delta_library import build_lcm

# ── Constants ────────────────────────────────────────────────────────────────
PHI      = 1.6180339887
PHI_GRID = 128

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'model_reverse_engineering_v2', 'phi_model')

# Qwen2-7B dims (from phi_model/config.json)
HIDDEN_DIM  = 3584
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM    = 128

SMALL_MODEL = "Qwen/Qwen2-1.5B-Instruct"   # for live forward passes (3 GB)

# Food reference words for domain alignment scoring
FOOD_REF_WORDS = ['bread', 'soup', 'cake', 'rice', 'pasta', 'egg', 'milk', 'cheese']

# ── φ decoding ────────────────────────────────────────────────────────────────

def phi_decode(signs: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """sign × φ^(exp / PHI_GRID) → float32"""
    return (signs.astype(np.float64) *
            (PHI ** (exponents.astype(np.float64) / PHI_GRID))).astype(np.float32)


def load_phi_npz(path: str) -> np.ndarray:
    d = np.load(path)
    return phi_decode(d['signs'], d['exponents'])


# ── Word embedding lookup ──────────────────────────────────────────────────────

def get_word_embeddings_7b(words, tokenizer):
    """
    Load only the necessary rows from embed_tokens.npz.
    Decodes just the requested tokens — no need to decode all 152K embeddings.
    """
    print(f"  Resolving token IDs for {len(words)} words...")
    word_to_tid = {}
    multi_token = []
    for w in words:
        ids = tokenizer.encode(' ' + w, add_special_tokens=False)
        if len(ids) == 1:
            word_to_tid[w] = ids[0]
        else:
            ids2 = tokenizer.encode(w, add_special_tokens=False)
            if len(ids2) == 1:
                word_to_tid[w] = ids2[0]
            else:
                multi_token.append((w, ids))

    if multi_token:
        print(f"  Multi-token words (skipped): {[w for w,_ in multi_token]}")

    emb_path = os.path.join(MODEL_DIR, 'embed_tokens.npz')
    print(f"  Loading embed_tokens.npz ({os.path.getsize(emb_path)/1e6:.0f} MB)...")
    data     = np.load(emb_path)
    signs    = data['signs']
    exponents = data['exponents']

    word_embeds = {}
    for w, tid in word_to_tid.items():
        word_embeds[w] = phi_decode(signs[tid], exponents[tid])  # (3584,)

    del data, signs, exponents
    print(f"  Loaded {len(word_embeds)} embeddings.")
    return word_embeds, word_to_tid


# ── Part 1 ─────────────────────────────────────────────────────────────────────

def run_part1(lcm, tokenizer, layers=(0, 5, 22, 23)):
    """
    Compare IRD cosine similarity to L0 attention logit for word pairs.
    Tests: does geometry predict what the transformer will attend to?
    """
    print("\n" + "="*65)
    print("PART 1 — Static Pairwise Similarity: IRD cos vs Attn Logit")
    print("="*65)

    # Test pairs covering polysemy and clean semantics
    test_pairs = [
        # Polysemous contrasts
        ('cookie', 'recipe'),     # culinary sense
        ('cookie', 'login'),      # HTTP sense
        ('cookie', 'flour'),      # culinary sense
        ('bass', 'guitar'),       # music sense
        ('bass', 'fish'),         # aquatic sense
        ('bank', 'river'),        # geographic sense
        ('bank', 'money'),        # finance sense
        # Clean semantic pairs (should be strong)
        ('france', 'paris'),
        ('king',   'queen'),
        ('bread',  'flour'),
        ('sugar',  'butter'),
        # Distant pairs (should be weak)
        ('cookie',    'mountain'),
        ('algorithm', 'bread'),
        ('france',    'sugar'),
    ]

    all_words = list({w for pair in test_pairs for w in pair})
    word_embeds, word_to_tid = get_word_embeddings_7b(all_words, tokenizer)

    # Load Q, K for each requested layer
    print(f"\n  Loading Q, K matrices for layers {layers}...")
    layer_qk = {}
    for layer_idx in layers:
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        W_q = load_phi_npz(os.path.join(layer_dir, 'q_proj.npz'))  # (3584, 3584)
        W_k = load_phi_npz(os.path.join(layer_dir, 'k_proj.npz'))  # (512, 3584)
        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        b_q = biases['q_proj_bias'].astype(np.float32)              # (3584,)
        b_k = biases['k_proj_bias'].astype(np.float32)              # (512,)
        layer_qk[layer_idx] = (W_q, W_k, b_q, b_k)
        print(f"    Layer {layer_idx}: W_q{W_q.shape} W_k{W_k.shape}")

    def attn_logit_pair(w1, w2, layer_idx):
        """Mean attention logit across all 28 heads for word pair (w1, w2)."""
        if w1 not in word_embeds or w2 not in word_embeds:
            return None
        W_q, W_k, b_q, b_k = layer_qk[layer_idx]
        e1 = word_embeds[w1].astype(np.float32)
        e2 = word_embeds[w2].astype(np.float32)
        # Q for each query head: W_q[h*128:(h+1)*128, :] @ e1 + b_q[h*128:(h+1)*128]
        # K for corresponding KV head: W_k[(h//7)*128:((h//7)+1)*128, :] @ e2 + ...
        heads_per_kv = NUM_Q_HEADS // NUM_KV_HEADS  # 7
        logits = []
        for h in range(NUM_Q_HEADS):
            qs = h * HEAD_DIM
            qe = qs + HEAD_DIM
            kv_h = h // heads_per_kv
            ks = kv_h * HEAD_DIM
            ke = ks + HEAD_DIM
            q = W_q[qs:qe, :] @ e1 + b_q[qs:qe]
            k = W_k[ks:ke, :] @ e2 + b_k[ks:ke]
            logits.append(float(np.dot(q, k) / np.sqrt(HEAD_DIM)))
        return np.mean(logits)

    def ird_cos(w1, w2):
        try:
            p1, _ = lcm._get_proj(w1)
            p2, _ = lcm._get_proj(w2)
            p1 = p1.astype(np.float64)
            p2 = p2.astype(np.float64)
            return float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-20))
        except RuntimeError:
            return None

    # Gather results
    print(f"\n  {'Pair':<28s}  IRD_cos  " +
          "  ".join(f"L{l:02d}_logit" for l in layers))
    print("  " + "-"*72)

    ird_vals = []
    l0_vals  = []
    results  = []

    for w1, w2 in test_pairs:
        ic = ird_cos(w1, w2)
        l_logits = {l: attn_logit_pair(w1, w2, l) for l in layers}
        pair_label = f"{w1}↔{w2}"
        ird_str  = f"{ic:+.3f}" if ic is not None else "  n/a"
        log_strs = [f"{l_logits[l]:+7.3f}" if l_logits[l] is not None else "    n/a"
                    for l in layers]
        print(f"  {pair_label:<28s}  {ird_str}   " + "  ".join(log_strs))
        if ic is not None and l_logits[layers[0]] is not None:
            ird_vals.append(ic)
            l0_vals.append(l_logits[layers[0]])
            results.append((w1, w2, ic, {l: l_logits[l] for l in layers}))

    # Pearson correlation
    if len(ird_vals) >= 3:
        r = float(np.corrcoef(ird_vals, l0_vals)[0, 1])
        print(f"\n  Pearson r(IRD_cos, L{layers[0]}_logit) = {r:+.4f}  "
              f"(n={len(ird_vals)} pairs)")
        print(f"  {'Interpretation:':15s} ", end='')
        if r > 0.7:
            print("STRONG — geometry closely predicts attention")
        elif r > 0.4:
            print("MODERATE — geometry partially predicts attention")
        elif r > 0:
            print("WEAK positive correlation")
        else:
            print("No positive correlation (attention driven by other factors)")

    # Key polysemy test
    print("\n  ── Polysemy disambiguation signal ──────────────────────────")
    for base, sense_a, sense_b, label in [
        ('cookie', 'recipe', 'login',  'culinary vs HTTP'),
        ('bass',   'guitar', 'fish',   'music vs aquatic'),
        ('bank',   'river',  'money',  'geography vs finance'),
    ]:
        ird_a = ird_cos(base, sense_a)
        ird_b = ird_cos(base, sense_b)
        l0_a  = attn_logit_pair(base, sense_a, layers[0])
        l0_b  = attn_logit_pair(base, sense_b, layers[0])
        if None not in (ird_a, ird_b, l0_a, l0_b):
            ird_dir = "✓ culinary>HTTP" if ird_a > ird_b else "✗ wrong direction"
            attn_dir = "✓ culinary>HTTP" if l0_a > l0_b else "✗ wrong direction"
            print(f"  {base}+{sense_a} vs {base}+{sense_b}  ({label})")
            print(f"    IRD:  {ird_a:+.3f} vs {ird_b:+.3f}  → {ird_dir}")
            print(f"    Attn: {l0_a:+.3f} vs {l0_b:+.3f}  → {attn_dir}")
            agree = "AGREE" if (ird_a > ird_b) == (l0_a > l0_b) else "DISAGREE"
            print(f"    → Geometry and attention: {agree}")
        else:
            print(f"  {base}: some words not in vocab, skipping")


# ── Part 2 ─────────────────────────────────────────────────────────────────────

def run_part2(lcm):
    """
    Extract hidden states from Qwen2-1.5B on polysemous sentences.
    Compare the shift in h("cookie" | culinary context) vs h("cookie" | HTTP context)
    to the shift our context_correct_proj() applies.
    """
    print("\n" + "="*65)
    print("PART 2 — In-Context Hidden State Shift vs IRD Gravity")
    print("="*65)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  SKIP: torch/transformers not available")
        return

    print(f"\n  Loading {SMALL_MODEL}...")
    tok15 = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL,
        torch_dtype=torch.float32,
        device_map='cpu',
        output_hidden_states=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, hidden={model.config.hidden_size}")

    def get_hidden_states(text, target_word):
        """
        Run model on *text*, return hidden states at all layers for the
        first token of *target_word* in the text.
        """
        inputs = tok15(text, return_tensors='pt')
        tokens = [tok15.decode([t]) for t in inputs['input_ids'][0]]

        # Find position of target word token
        target_tok_ids = tok15.encode(' ' + target_word, add_special_tokens=False)
        target_pos = None
        for i, t in enumerate(inputs['input_ids'][0]):
            if t.item() == target_tok_ids[0]:
                target_pos = i
                break
        if target_pos is None:
            target_tok_ids2 = tok15.encode(target_word, add_special_tokens=False)
            for i, t in enumerate(inputs['input_ids'][0]):
                if t.item() == target_tok_ids2[0]:
                    target_pos = i
                    break

        if target_pos is None:
            print(f"  WARNING: '{target_word}' not found in tokenized '{text}'")
            print(f"  Tokens: {tokens}")
            return None, None

        with torch.no_grad():
            out = model(**inputs)
        # out.hidden_states: tuple of (seq_len, hidden) tensors per layer
        # shape: [n_layers+1][seq_len, hidden_size]
        hs = [layer_hs[0, target_pos, :].numpy()
              for layer_hs in out.hidden_states]
        return hs, tokens

    def hidden_food_alignment(hs, layer_idx):
        """
        Cosine similarity of hidden state at *layer_idx* with the centroid
        of food-reference word embeddings (in this model's embedding space).
        """
        # Get food reference word embeddings from this model
        food_embeds = []
        emb_matrix = model.model.embed_tokens.weight.detach().numpy()
        for fw in FOOD_REF_WORDS:
            fids = tok15.encode(' ' + fw, add_special_tokens=False)
            if fids:
                food_embeds.append(emb_matrix[fids[0]])
        food_centroid = np.mean(food_embeds, axis=0)
        food_cn       = food_centroid / (np.linalg.norm(food_centroid) + 1e-20)

        h = hs[layer_idx]
        h_n = h / (np.linalg.norm(h) + 1e-20)
        return float(np.dot(h_n, food_cn))

    # Test sentences — same polysemous word, two different contexts
    test_cases = [
        {
            'word':       'cookie',
            'sentence_a': 'bake a cookie with flour',
            'sentence_b': 'clear the browser cookie',
            'label_a':    'culinary',
            'label_b':    'HTTP',
        },
        {
            'word':       'bass',
            'sentence_a': 'play bass guitar solo',
            'sentence_b': 'catch the large bass fish',
            'label_a':    'music',
            'label_b':    'aquatic',
        },
    ]

    key_layers = [0, n_layers // 4, n_layers // 2, n_layers - 1]

    for tc in test_cases:
        word = tc['word']
        print(f"\n  ── '{word}' polysemy disambiguation ────────────────────")
        print(f"  Sentence A [{tc['label_a']}]: '{tc['sentence_a']}'")
        print(f"  Sentence B [{tc['label_b']}]: '{tc['sentence_b']}'")

        hs_a, toks_a = get_hidden_states(tc['sentence_a'], word)
        hs_b, toks_b = get_hidden_states(tc['sentence_b'], word)

        if hs_a is None or hs_b is None:
            continue

        print(f"\n  {'Layer':<8s}  {'food_align_A':>14s}  {'food_align_B':>14s}  "
              f"{'Δ (A-B)':>10s}  Direction")
        print("  " + "-"*60)

        for l in key_layers:
            fa = hidden_food_alignment(hs_a, l)
            fb = hidden_food_alignment(hs_b, l)
            delta = fa - fb
            direction = "✓ A>B (culinary shifted up)" if delta > 0 else "✗ B>A"
            print(f"  L{l:<6d}  {fa:+.4f}        {fb:+.4f}        {delta:+.4f}    {direction}")

        # Compare to IRD context gravity prediction
        print(f"\n  IRD context gravity prediction for '{word}':")
        ctx_a_words = [w for w in tc['sentence_a'].split() if w != word and len(w) > 2]
        ctx_b_words = [w for w in tc['sentence_b'].split() if w != word and len(w) > 2]

        try:
            # Compute food alignment in IRD space
            food_vecs = []
            for fw in FOOD_REF_WORDS:
                try:
                    p, _ = lcm._get_proj(fw)
                    food_vecs.append(p.astype(np.float64))
                except RuntimeError:
                    pass
            food_centroid_ird = np.mean(food_vecs, axis=0)
            food_cn_ird = food_centroid_ird / (np.linalg.norm(food_centroid_ird) + 1e-20)

            def ird_food_align(proj):
                p = proj.astype(np.float64)
                p /= (np.linalg.norm(p) + 1e-20)
                return float(np.dot(p, food_cn_ird))

            p_word, _ = lcm._get_proj(word)
            align_native = ird_food_align(p_word)

            ctx_a_ok = [w for w in ctx_a_words if _word_in_lcm(lcm, w)]
            ctx_b_ok = [w for w in ctx_b_words if _word_in_lcm(lcm, w)]

            if ctx_a_ok:
                p_corr_a = lcm.context_correct_proj(word, ctx_a_ok)
                align_a = ird_food_align(p_corr_a)
            else:
                align_a = align_native

            if ctx_b_ok:
                p_corr_b = lcm.context_correct_proj(word, ctx_b_ok)
                align_b = ird_food_align(p_corr_b)
            else:
                align_b = align_native

            delta_ird = align_a - align_b
            print(f"  Native (no context):   food_align={align_native:+.4f}")
            print(f"  Context A [{tc['label_a']}]: food_align={align_a:+.4f}  "
                  f"(ctx={ctx_a_ok})")
            print(f"  Context B [{tc['label_b']}]: food_align={align_b:+.4f}  "
                  f"(ctx={ctx_b_ok})")
            print(f"  Δ_IRD (A-B) = {delta_ird:+.4f}")

            # Final comparison
            transformer_agrees = (hs_a is not None and
                                   hidden_food_alignment(hs_a, -1) >
                                   hidden_food_alignment(hs_b, -1))
            geometry_agrees = delta_ird > 0
            print(f"\n  Transformer says '{word}' is MORE food-like in context A: "
                  f"{'YES ✓' if transformer_agrees else 'NO ✗'}")
            print(f"  IRD gravity says same:                               "
                  f"{'YES ✓' if geometry_agrees else 'NO ✗'}")
            if transformer_agrees == geometry_agrees:
                print(f"  *** AGREEMENT: geometry and transformer AGREE ***")
            else:
                print(f"  *** DISAGREEMENT: geometry and transformer differ ***")

        except RuntimeError as e:
            print(f"  IRD lookup failed: {e}")


def _word_in_lcm(lcm, w):
    try:
        lcm._get_proj(w)
        return True
    except RuntimeError:
        return False


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part1', action='store_true')
    parser.add_argument('--part2', action='store_true')
    args = parser.parse_args()
    run_both = not args.part1 and not args.part2

    print("Loading LCM...")
    lcm = build_lcm()

    if args.part1 or run_both:
        from transformers import AutoTokenizer
        print("Loading Qwen2-7B tokenizer (for token ID lookup)...")
        tokenizer7b = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        run_part1(lcm, tokenizer7b)

    if args.part2 or run_both:
        run_part2(lcm)
