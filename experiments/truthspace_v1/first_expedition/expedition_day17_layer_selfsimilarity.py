#!/usr/bin/env python3
"""
Expedition Day 17 — Cross-Layer Self-Similarity

Hypothesis:
  If the transformer is a self-similar geometric system, the SAME
  mathematical invariants (Killing vectors, Lagrange L4/L5 clusters)
  should appear at EVERY layer's hidden states, not just at the
  vocabulary embedding.

  The geometry is the same operation applied to progressively refined
  representations. Error compounds because each layer amplifies whatever
  the previous layer produced — good or bad.

Measurements:
  1. Killing vector direction consistency across layers
     cos(Δ_L, Δ_0) for each relationship type: does the functional
     delta direction stay stable from embedding to final layer?

  2. Within-relationship universality at each layer
     At layer L, do all gender pairs point in the same direction?
     Mean pairwise cosine across pairs — does universality strengthen,
     weaken, or stay constant?

  3. Layer-to-layer drift: cos(Δ_L, Δ_{L+1}) — where is the geometry
     being significantly reorganised?

  4. Lagrange L4/L5 at selected layers
     Run a small vocabulary through the model. Compute L4/L5 for key
     pairs at L0, L7, L14, L21, L28. Do the same Trojan clusters appear?

  5. Delta magnitude evolution: does the Killing vector grow, shrink,
     or stay constant as information refines?

Model: Qwen/Qwen2-1.5B-Instruct (28 layers, hidden=1536)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL = "Qwen/Qwen2-1.5B-Instruct"

# ── Killing pairs ─────────────────────────────────────────────────────────────
KILLING_PAIRS = {
    'gender':      [('king','queen'), ('man','woman'), ('boy','girl'),
                    ('actor','actress'), ('prince','princess')],
    'comparative': [('big','bigger'), ('fast','faster'), ('old','older'),
                    ('strong','stronger'), ('small','smaller')],
    'plural':      [('cat','cats'), ('dog','dogs'), ('tree','trees'),
                    ('bird','birds'), ('car','cars')],
    'past':        [('walk','walked'), ('run','ran'), ('eat','ate'),
                    ('see','saw'), ('speak','spoke')],
    'capital':     [('france','paris'), ('germany','berlin'), ('japan','tokyo'),
                    ('spain','madrid'), ('italy','rome')],
}

# Small vocab for Lagrange tests (run all through model)
LAGRANGE_VOCAB = [
    'king', 'queen', 'prince', 'princess', 'emperor', 'empress',
    'lord', 'lady', 'duke', 'duchess', 'knight', 'noble',
    'kingdom', 'royal', 'crown', 'throne',
    'big', 'bigger', 'small', 'smaller', 'fast', 'faster',
    'strong', 'stronger', 'tall', 'taller', 'heavy', 'heavier',
    'deep', 'deeper', 'wide', 'wider', 'thick', 'thicker',
    'france', 'paris', 'germany', 'berlin', 'japan', 'tokyo',
    'italy', 'rome', 'spain', 'madrid', 'england', 'london',
    'australia', 'canberra', 'canada', 'ottawa',
    'cat', 'cats', 'dog', 'dogs', 'bird', 'birds',
    'man', 'woman', 'boy', 'girl', 'actor', 'actress',
]

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {SMALL_MODEL}...")
    tok = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL,
        torch_dtype=torch.float32,
        device_map='cpu',
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden   = model.config.hidden_size
    print(f"  Loaded: {n_layers} layers, hidden={hidden}")
    return model, tok, n_layers, hidden


def get_word_hidden_states(model, tok, word):
    """
    Return array of shape (n_layers+1, hidden_size):
    hidden state at the word's first token position, at every layer.
    Layer 0 = embedding output (pre-transformer).
    Layer k = after k-th transformer block.
    """
    import torch
    # Encode with leading space (standard subword convention)
    for variant in (' ' + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if ids:
            target_id = ids[0]
            break
    else:
        return None

    # Full sentence: BOS + word token(s)
    inputs = tok(word, return_tensors='pt')
    input_ids = inputs['input_ids'][0]

    # Find word token position
    target_pos = None
    for i, t in enumerate(input_ids):
        if t.item() == target_id:
            target_pos = i
            break
    if target_pos is None:
        target_pos = len(input_ids) - 1  # fallback: last token

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # out.hidden_states: tuple len n_layers+1, each (1, seq_len, hidden)
    states = np.stack([
        hs[0, target_pos, :].numpy()
        for hs in out.hidden_states
    ])  # (n_layers+1, hidden)
    return states


# ── Utility ───────────────────────────────────────────────────────────────────

def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    model, tok, n_layers, hidden_dim = load_model()

    # ── Pre-cache all words we'll need ────────────────────────────────────────
    all_words = set(LAGRANGE_VOCAB)
    for pairs in KILLING_PAIRS.values():
        for a, b in pairs:
            all_words.update([a, b])

    print(f"\n  Caching hidden states for {len(all_words)} words...")
    cache = {}
    for w in sorted(all_words):
        hs = get_word_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs  # (n_layers+1, hidden)
        else:
            print(f"    WARNING: '{w}' not tokenisable")
    print(f"  Cached {len(cache)} words.")

    # ── Section 1: Killing vector direction across layers ─────────────────────
    print(f"\n{'='*65}")
    print(f"DAY 17 — Cross-Layer Self-Similarity")
    print(f"{'='*65}")

    print(f"\n── Section 1: Killing vector direction consistency across layers ──")
    print(f"  cos(Δ_L, Δ_0) for each relationship — does direction persist?\n")

    # Table header
    layer_sample = list(range(0, n_layers+1, max(1, (n_layers+1)//10)))
    if n_layers not in layer_sample:
        layer_sample.append(n_layers)
    header = f"  {'Rel':<12}" + "".join(f"  L{l:<3}" for l in layer_sample)
    print(header)
    print("  " + "─" * len(header))

    rel_layer_deltas = {}  # {rel: {layer: mean_delta}}
    for rel, pairs in KILLING_PAIRS.items():
        layer_deltas = {}
        for L in range(n_layers + 1):
            deltas_at_L = []
            for a, b in pairs:
                if a in cache and b in cache:
                    d = cache[b][L] - cache[a][L]
                    deltas_at_L.append(d)
            if deltas_at_L:
                layer_deltas[L] = np.mean(deltas_at_L, axis=0)
        rel_layer_deltas[rel] = layer_deltas

        if 0 not in layer_deltas:
            continue
        d0 = layer_deltas[0]
        row = f"  {rel:<12}"
        for L in layer_sample:
            if L in layer_deltas:
                c = cos(layer_deltas[L], d0)
                row += f"  {c:+.2f}"
            else:
                row += f"  {'?':>5}"
        print(row)

    # ── Section 2: Within-relationship universality at each layer ─────────────
    print(f"\n── Section 2: Within-relationship universality at key layers ──────")
    print(f"  Mean pairwise cosine of individual pair deltas — high = universal\n")

    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]
    key_layers = sorted(set(key_layers))

    print(f"  {'Rel':<12}" + "".join(f"  L{l:<4}" for l in key_layers))
    print("  " + "─" * (14 + 7 * len(key_layers)))

    for rel, pairs in KILLING_PAIRS.items():
        row = f"  {rel:<12}"
        for L in key_layers:
            deltas_at_L = []
            for a, b in pairs:
                if a in cache and b in cache:
                    d = cache[b][L] - cache[a][L]
                    deltas_at_L.append(d / (np.linalg.norm(d) + 1e-20))
            if len(deltas_at_L) < 2:
                row += f"  {'?':>6}"
                continue
            # Mean pairwise cosine
            n = len(deltas_at_L)
            sims = []
            for i in range(n):
                for j in range(i+1, n):
                    sims.append(cos(deltas_at_L[i], deltas_at_L[j]))
            row += f"  {np.mean(sims):+.3f}"
        print(row)

    # ── Section 3: Layer-to-layer drift ──────────────────────────────────────
    print(f"\n── Section 3: Layer-to-layer drift cos(Δ_L, Δ_L+1) ────────────────")
    print(f"  Where does the geometry reorganise most sharply?\n")

    print(f"  {'Rel':<12}  " + "  ".join(f"L{l}→{l+1}" for l in range(0, n_layers, n_layers//8)))
    print("  " + "─" * 70)

    jump_layers = list(range(0, n_layers, max(1, n_layers//8)))

    for rel in KILLING_PAIRS:
        ld = rel_layer_deltas.get(rel, {})
        row = f"  {rel:<12}  "
        cols = []
        for L in jump_layers:
            if L in ld and L+1 in ld:
                c = cos(ld[L], ld[L+1])
                cols.append(f"{c:+.3f}")
            else:
                cols.append("  ?  ")
        row += "  ".join(cols)
        print(row)

    # ── Section 4: Lagrange L4/L5 at multiple layers ─────────────────────────
    print(f"\n── Section 4: Lagrange L4/L5 Trojan clusters at multiple layers ───")
    print(f"  Do the same clusters appear at every scale?\n")

    lagrange_test_pairs = [
        ('king', 'queen', 'gender'),
        ('big',  'bigger', 'comparative'),
        ('france', 'paris', 'capital'),
    ]

    lag_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]
    lag_layers = sorted(set(lag_layers))
    vocab_in_cache = [w for w in LAGRANGE_VOCAB if w in cache]

    for w_a, w_b, label in lagrange_test_pairs:
        if w_a not in cache or w_b not in cache:
            print(f"  {w_a}/{w_b}: not in cache")
            continue

        print(f"  {w_a} ↔ {w_b}  ({label})")
        for L in lag_layers:
            pa = cache[w_a][L].astype(np.float64)
            pb = cache[w_b][L].astype(np.float64)
            pa_n = pa / (np.linalg.norm(pa) + 1e-20)
            pb_n = pb / (np.linalg.norm(pb) + 1e-20)

            mid  = pa_n + pb_n
            mid_n = mid / (np.linalg.norm(mid) + 1e-20)
            perp = pb_n - np.dot(pb_n, pa_n) * pa_n
            perp_n = perp / (np.linalg.norm(perp) + 1e-20)

            L4_dir = 0.866 * mid_n + 0.5 * perp_n
            L5_dir = 0.866 * mid_n - 0.5 * perp_n
            L4_dir /= (np.linalg.norm(L4_dir) + 1e-20)
            L5_dir /= (np.linalg.norm(L5_dir) + 1e-20)

            # Score all vocab words against L4 and L5
            vocab_vecs_L = np.stack([
                cache[w][L].astype(np.float64) for w in vocab_in_cache
            ])
            # Normalise rows
            norms = np.linalg.norm(vocab_vecs_L, axis=1, keepdims=True)
            vocab_norm = vocab_vecs_L / (norms + 1e-20)

            sims_L4 = vocab_norm @ L4_dir
            sims_L5 = vocab_norm @ L5_dir

            # Exclude anchor words
            for excl in (w_a, w_b):
                if excl in vocab_in_cache:
                    idx = vocab_in_cache.index(excl)
                    sims_L4[idx] = -9999
                    sims_L5[idx] = -9999

            top3_L4 = [vocab_in_cache[i] for i in np.argsort(sims_L4)[-3:][::-1]]
            top3_L5 = [vocab_in_cache[i] for i in np.argsort(sims_L5)[-3:][::-1]]
            ab_cos  = cos(pa_n, pb_n)

            print(f"    L{L:2d}  (cos_ab={ab_cos:+.3f})  "
                  f"L4=[{', '.join(top3_L4)}]  L5=[{', '.join(top3_L5)}]")
        print()

    # ── Section 5: Delta magnitude evolution ─────────────────────────────────
    print(f"\n── Section 5: Killing vector magnitude across layers ───────────────")
    print(f"  ||Δ_L|| normalized to L0 — does the signal grow, shrink, stabilise?\n")

    print(f"  {'Rel':<12}" + "".join(f"  L{l:<3}" for l in layer_sample))
    print("  " + "─" * len(header))

    for rel in KILLING_PAIRS:
        ld = rel_layer_deltas.get(rel, {})
        if 0 not in ld:
            continue
        mag0 = np.linalg.norm(ld[0]) + 1e-20
        row = f"  {rel:<12}"
        for L in layer_sample:
            if L in ld:
                m = np.linalg.norm(ld[L]) / mag0
                row += f"  {m:.2f} "
            else:
                row += f"  {'?':>5}"
        print(row)

    # ── Section 6: Cross-relationship orthogonality at each layer ─────────────
    print(f"\n── Section 6: Cross-relationship orthogonality across layers ───────")
    print(f"  cos(Δ_gender, Δ_comparative) etc. — do distinct Killing vectors")
    print(f"  stay orthogonal at every layer?\n")

    rel_names = list(rel_layer_deltas.keys())
    pairs_to_check = [
        ('gender', 'comparative'),
        ('gender', 'plural'),
        ('gender', 'past'),
        ('comparative', 'plural'),
        ('capital', 'gender'),
    ]

    print(f"  {'Pair':<28}" + "".join(f"  L{l:<3}" for l in layer_sample))
    print("  " + "─" * (30 + 6 * len(layer_sample)))

    for ra, rb in pairs_to_check:
        if ra not in rel_layer_deltas or rb not in rel_layer_deltas:
            continue
        lda = rel_layer_deltas[ra]
        ldb = rel_layer_deltas[rb]
        row = f"  {ra+'↔'+rb:<28}"
        for L in layer_sample:
            if L in lda and L in ldb:
                c = cos(lda[L], ldb[L])
                row += f"  {c:+.3f}"
            else:
                row += f"  {'?':>5}"
        print(row)

    print(f"\n{'='*65}")
    print(f"Day 17 complete.")
    print(f"{'='*65}")
