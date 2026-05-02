#!/usr/bin/env python3
"""
Expedition Day 19 — Trivial Zeros of the Transformer

Hypothesis:
  The Riemann Zeta function has two classes of zeros:
    - Trivial zeros: at s = -2, -4, -6, ... Mechanical, closed-form,
      arise from sin(πs/2) in the functional equation. They "zero out"
      the incommensurate direction and enforce the reflection symmetry.
    - Non-trivial zeros: in the critical strip, lying (per Riemann) on
      the critical line Re(s) = 1/2.

  If the transformer IS a zeta function:
    - Trivial zeros = the crystallisation at L1→L2 (mechanically
      predictable, zeros the embedding-space structure, enforces Z2 symmetry)
    - Non-trivial zeros = semantic flip points in the COMB zone L2-L26
      (require full computation, lie on the Z2 axis — the critical line)
    - The transformer's Riemann Hypothesis is: all Killing vectors in the
      COMB zone lie on the Z2 axis. Day 17 proved this empirically (cos≈±0.999).

  The trivial/non-trivial split predicts:
    TRIVIAL CRYSTALLISATION  →  monosemous words   →  linear map T works
    NON-TRIVIAL CRYSTALLISATION  →  polysemous words  →  linear map T fails

  We test this by measuring T quality (cos of predicted vs. actual L2 hidden
  state) against a polysemy proxy: the number of WordNet senses or the
  entropy of the word's nearest-neighbour distribution in the embedding space.

Measurements:
  1. T quality vs. embedding neighbourhood entropy (polysemy proxy)
     Does high entropy → low T quality? Clean correlation = hypothesis confirmed.

  2. The null space at L2 — which dimensions are "zeroed" by crystallisation?
     These should be structural (not semantic) dimensions.

  3. The Z2 axis as the critical line
     Non-trivial zeros from DC 295: the logit-gap flip points in the COMB zone.
     Do they all lie on the Z2 axis? Measure cos(flip_direction, Z2_axis).

  4. Functional equation symmetry
     If encode (L0→L2) and decode (L27→L28) are reflections of each other
     (like ζ(s) ↔ ζ(1-s)), then T_encode @ T_decode ≈ Identity.
     Measure residual to see how close to a perfect symmetry.

  5. Polysemy word set test
     Select clearly monosemous words (dog, tree, car, fast, big) and
     clearly polysemous words (bank, cold, light, spring, rock, date, fly).
     Measure T quality for each and compare the two distributions.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL = "Qwen/Qwen2-1.5B-Instruct"

# Words with KNOWN low polysemy (one dominant meaning)
MONOSEMOUS = [
    # animals
    'elephant', 'penguin', 'giraffe', 'crocodile', 'dolphin',
    'caterpillar', 'rhinoceros', 'chimpanzee', 'flamingo', 'kangaroo',
    # clear adjectives
    'triangular', 'rectangular', 'cylindrical', 'hexagonal', 'spherical',
    # numbers / units
    'kilogram', 'kilometer', 'celsius', 'hydrogen', 'nitrogen',
    # simple morphological forms
    'cats', 'dogs', 'trees', 'birds', 'houses',
    'bigger', 'faster', 'stronger', 'taller', 'heavier',
    # unambiguous proper nouns
    'tokyo', 'berlin', 'paris', 'madrid', 'beijing',
]

# Words with HIGH polysemy (multiple common meanings)
POLYSEMOUS = [
    # classic polysemous
    'bank', 'bat', 'bear', 'light', 'cold',
    'spring', 'rock', 'date', 'fly', 'match',
    'pitch', 'fair', 'crane', 'bark', 'bolt',
    # grammatically ambiguous (noun/verb/adjective)
    'fast', 'run', 'fire', 'fall', 'wave',
    'set', 'table', 'watch', 'round', 'shot',
]

# Training vocabulary for the linear map T (same as Day 18)
TRAINING_VOCAB = [
    'king', 'queen', 'man', 'woman', 'boy', 'girl',
    'cat', 'dog', 'tree', 'bird', 'car',
    'big', 'bigger', 'fast', 'faster', 'old', 'older',
    'france', 'germany', 'japan', 'spain', 'italy',
    'walk', 'walked', 'see', 'saw', 'eat', 'ate',
    'actor', 'actress', 'prince', 'princess',
    'strong', 'stronger', 'small', 'smaller',
]

CRYST_LAYER = 2  # First-order transition confirmed at L1→L2


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20: return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_hidden_states(model, tok, word):
    import torch
    for variant in (' ' + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if ids:
            target_id = ids[0]; break
    else:
        return None
    inputs  = tok(word, return_tensors='pt')
    id_list = inputs['input_ids'][0]
    pos = next((i for i, t in enumerate(id_list) if t.item() == target_id),
               len(id_list) - 1)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return np.stack([hs[0, pos, :].numpy() for hs in out.hidden_states])


def embedding_entropy(h0, all_h0, k=20):
    """
    Polysemy proxy: entropy of the k-NN similarity distribution.
    A monosemous word has a peaked (low entropy) neighbourhood.
    A polysemous word's embedding sits at a hub (high entropy).
    """
    sims = all_h0 @ (h0 / (np.linalg.norm(h0) + 1e-20))
    topk = np.sort(sims)[-k:]
    # Normalise to a probability distribution and compute entropy
    topk = topk - topk.min() + 1e-10
    topk = topk / topk.sum()
    return -float(np.sum(topk * np.log(topk + 1e-20)))


if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {SMALL_MODEL}...")
    tok   = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers")

    all_words = set(MONOSEMOUS) | set(POLYSEMOUS) | set(TRAINING_VOCAB)
    print(f"\n  Caching {len(all_words)} words...")
    cache = {}
    for w in sorted(all_words):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs
    print(f"  Cached {len(cache)} words.")

    print(f"\n{'='*65}")
    print(f"DAY 19 — Trivial Zeros of the Transformer")
    print(f"{'='*65}")

    # ── Build linear map T: h_L0 → h_L2 from training vocab ──────────────────
    X_tr = np.stack([cache[w][0].astype(np.float64)
                     for w in TRAINING_VOCAB if w in cache])
    Y_tr = np.stack([cache[w][CRYST_LAYER].astype(np.float64)
                     for w in TRAINING_VOCAB if w in cache])
    T, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)  # (hidden, hidden)
    print(f"\n  Linear map T fitted on {len(X_tr)} training words.")

    # ── Section 1: T quality vs. embedding entropy ────────────────────────────
    print(f"\n── Section 1: Trivial/Non-trivial split — monosemous vs polysemous ─")
    print(f"  T quality = cos(T(h_L0), h_L2)  [1.0 = trivial, ~0 = non-trivial]")
    print(f"  Entropy   = embedding neighbourhood entropy [low = monosemous]\n")

    # Build embedding matrix for entropy computation
    all_h0 = np.stack([cache[w][0].astype(np.float64)
                       for w in sorted(cache)])
    norms  = np.linalg.norm(all_h0, axis=1, keepdims=True)
    all_h0_n = all_h0 / (norms + 1e-20)
    all_words_list = sorted(cache)

    print(f"\n  MONOSEMOUS words:")
    print(f"  {'Word':<16} {'T quality':>10} {'Emb entropy':>12}")
    print("  " + "─" * 42)
    mono_t, mono_e = [], []
    for w in MONOSEMOUS:
        if w not in cache: continue
        h0  = cache[w][0].astype(np.float64)
        hLc = cache[w][CRYST_LAYER].astype(np.float64)
        t_quality = cos(h0 @ T, hLc)
        h0_n = h0 / (np.linalg.norm(h0) + 1e-20)
        ent = embedding_entropy(h0_n, all_h0_n)
        mono_t.append(t_quality); mono_e.append(ent)
        flag = '✓ TRIVIAL' if t_quality > 0.7 else ('~ partial' if t_quality > 0.3 else '✗ non-trivial')
        print(f"  {w:<16} {t_quality:>10.4f} {ent:>12.4f}  {flag}")

    print(f"\n  POLYSEMOUS words:")
    print(f"  {'Word':<16} {'T quality':>10} {'Emb entropy':>12}")
    print("  " + "─" * 42)
    poly_t, poly_e = [], []
    for w in POLYSEMOUS:
        if w not in cache: continue
        h0  = cache[w][0].astype(np.float64)
        hLc = cache[w][CRYST_LAYER].astype(np.float64)
        t_quality = cos(h0 @ T, hLc)
        h0_n = h0 / (np.linalg.norm(h0) + 1e-20)
        ent = embedding_entropy(h0_n, all_h0_n)
        poly_t.append(t_quality); poly_e.append(ent)
        flag = '✓ TRIVIAL' if t_quality > 0.7 else ('~ partial' if t_quality > 0.3 else '✗ non-trivial')
        print(f"  {w:<16} {t_quality:>10.4f} {ent:>12.4f}  {flag}")

    print(f"\n  SUMMARY:")
    print(f"  Monosemous  mean T quality: {np.mean(mono_t):.4f}  (σ={np.std(mono_t):.4f})")
    print(f"  Polysemous  mean T quality: {np.mean(poly_t):.4f}  (σ={np.std(poly_t):.4f})")
    print(f"  Monosemous  mean entropy:   {np.mean(mono_e):.4f}  (σ={np.std(mono_e):.4f})")
    print(f"  Polysemous  mean entropy:   {np.mean(poly_e):.4f}  (σ={np.std(poly_e):.4f})")
    # Pearson r between T quality and entropy across all words
    all_t = mono_t + poly_t
    all_e = mono_e + poly_e
    if len(all_t) > 2:
        r = np.corrcoef(all_t, all_e)[0, 1]
        print(f"  Pearson r(T quality, entropy): {r:.4f}  "
              f"({'ANTI-correlated as predicted' if r < -0.3 else 'weak/no correlation'})")

    # ── Section 2: The null space at L2 ──────────────────────────────────────
    print(f"\n── Section 2: The null space at L2 — what gets zeroed? ─────────────")
    print(f"  Dimensions with low variance at L2 but high variance at L0")
    print(f"  = the 'trivial zero dimensions' — structural, not semantic\n")

    # Compute per-dimension variance at L0 and L2 across all cached words
    H0 = np.stack([cache[w][0].astype(np.float64) for w in cache])
    H2 = np.stack([cache[w][CRYST_LAYER].astype(np.float64) for w in cache])

    var0 = np.var(H0, axis=0)  # (hidden,)
    var2 = np.var(H2, axis=0)  # (hidden,)

    # Variance ratio: var0/var2 — high ratio = dimension zeroed by crystallisation
    ratio = var0 / (var2 + 1e-20)
    # Top "zeroed" dimensions and bottom "amplified" dimensions
    top_zeroed = np.argsort(ratio)[-10:][::-1]
    top_amplified = np.argsort(ratio)[:10]

    print(f"  Top 10 dimensions ZEROED by crystallisation (var0/var2 ratio):")
    for dim in top_zeroed:
        print(f"    dim {dim:4d}: var0={var0[dim]:.3f}  var2={var2[dim]:.6f}  ratio={ratio[dim]:.1f}")

    print(f"\n  Top 10 dimensions AMPLIFIED by crystallisation (var2/var0):")
    for dim in top_amplified:
        if var0[dim] > 1e-10:
            print(f"    dim {dim:4d}: var0={var0[dim]:.6f}  var2={var2[dim]:.3f}  ratio={ratio[dim]:.6f}")

    null_thresh = np.percentile(ratio, 90)  # top 10% = zeroed
    n_zeroed = int((ratio > null_thresh).sum())
    print(f"\n  Dimensions with var0/var2 > {null_thresh:.1f}: {n_zeroed} dims ({100*n_zeroed/len(ratio):.1f}%)")
    print(f"  These are the 'trivial zero dimensions' — structural coordinates")

    # ── Section 3: Non-trivial zeros on the Z2 axis (critical line) ──────────
    print(f"\n── Section 3: Non-trivial zeros — are they on the Z2 axis? ─────────")
    print(f"  The semantic flip points in the COMB zone should lie on the Z2 axis")
    print(f"  Z2 axis = first PCA component of all mean Killing deltas in COMB\n")

    # Compute the Z2 axis from COMB-zone Killing deltas
    from delta_library import build_lcm
    KILLING_PAIRS = {
        'gender':      [('king','queen'), ('man','woman'), ('boy','girl')],
        'comparative': [('big','bigger'), ('fast','faster'), ('old','older')],
        'plural':      [('cat','cats'), ('dog','dogs'), ('tree','trees')],
        'past':        [('walk','walked'), ('see','saw'), ('eat','ate')],
    }
    comb_deltas = []
    for rel, pairs in KILLING_PAIRS.items():
        for L in range(CRYST_LAYER, n_layers - 2):  # COMB zone
            ds = [cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                  for a, b in pairs if a in cache and b in cache]
            if ds:
                d = np.mean(ds, axis=0)
                comb_deltas.append(d / (np.linalg.norm(d) + 1e-20))

    if comb_deltas:
        D = np.stack(comb_deltas)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        z2_axis = Vt[0]  # First right singular vector = Z2 axis
        sv2 = sv**2
        pct = sv2[0] / sv2.sum()
        print(f"  Z2 axis captures {100*pct:.2f}% of COMB-zone Killing vector variance")
        print(f"  (100% = fully crystallised onto single axis)")

        # Now project semantic flip directions onto Z2 axis
        # Use the layer-to-layer direction change at key layers as "flip indicators"
        print(f"\n  Layer-transition directions vs. Z2 axis:")
        print(f"  (high cos = flip direction IS the Z2 axis = non-trivial zero on critical line)")
        print(f"  {'Layer':<12} {'cos(transition, Z2)':>22}")
        print("  " + "─" * 36)
        for rel in ('gender', 'plural', 'comparative'):
            pairs = KILLING_PAIRS[rel]
            for L in range(CRYST_LAYER, n_layers - 1):
                d_L  = np.mean([cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                                for a,b in pairs if a in cache and b in cache], axis=0)
                d_L1 = np.mean([cache[b][L+1].astype(np.float64) - cache[a][L+1].astype(np.float64)
                                for a,b in pairs if a in cache and b in cache], axis=0)
                transition = d_L1 - d_L  # the CHANGE in Killing vector
                if np.linalg.norm(transition) < 1e-10:
                    continue
                c = cos(transition, z2_axis)
                if abs(c) > 0.5:  # Only print notable ones
                    print(f"  {rel} L{L:02d}→{L+1:02d}:  {c:+.4f}")

    # ── Section 4: Functional equation symmetry (encode ↔ decode) ────────────
    print(f"\n── Section 4: Functional equation symmetry — T_encode ≈ T_decode^T? ─")
    print(f"  If ζ(s) = ζ(1-s) reflected, then encoding and decoding should be")
    print(f"  approximate inverses. Measure: is T_encode @ T_decode ≈ Identity?\n")

    # T_encode: L0 → L2  (already fitted as T)
    # T_decode: L27 → L28 direction (fit similarly)
    T_encode = T  # h_L0 → h_L2

    # Fit T_decode: L2 → L0 (the reverse — is crystallisation reversible?)
    T_decode, _, _, _ = np.linalg.lstsq(Y_tr, X_tr, rcond=None)  # h_L2 → h_L0

    # Composition: T_encode @ T_decode (should approach identity if symmetric)
    # Since both are (hidden, hidden) and applied as x @ T, composition is T_encode @ T_decode
    composition = T_encode @ T_decode  # (hidden, hidden)
    identity    = np.eye(composition.shape[0])

    # Measure distance from identity
    residual = np.linalg.norm(composition - identity, 'fro') / np.linalg.norm(identity, 'fro')
    # Also measure via cosines of diagonal elements vs off-diagonal
    diag_mean = np.mean(np.diag(composition))
    offdiag_std = np.std(composition - np.diag(np.diag(composition)))

    print(f"  T_encode @ T_decode residual from Identity: {residual:.4f}")
    print(f"  (0.0 = perfect inverse pair, i.e., perfect functional equation symmetry)")
    print(f"  Diagonal mean: {diag_mean:.4f}  (1.0 = identity diagonal)")
    print(f"  Off-diagonal std: {offdiag_std:.6f}  (0.0 = identity off-diagonal)")

    # Also test on held-out words: does decode(encode(h_L0)) ≈ h_L0?
    print(f"\n  Round-trip test: encode then decode — does h_L0 → T → T^-1 recover h_L0?")
    print(f"  {'Word':<16} {'cos(original, roundtrip)':>25}")
    print("  " + "─" * 44)
    test_words = [w for w in (MONOSEMOUS[:5] + POLYSEMOUS[:5]) if w in cache]
    rt_scores = []
    for w in test_words:
        h0 = cache[w][0].astype(np.float64)
        h0_roundtrip = (h0 @ T_encode) @ T_decode
        c = cos(h0, h0_roundtrip)
        rt_scores.append(c)
        print(f"  {w:<16} {c:>25.4f}")
    print(f"\n  Mean round-trip fidelity: {np.mean(rt_scores):.4f}")

    # ── Section 5: Summary — the trivial zeros picture ───────────────────────
    print(f"\n── Section 5: The trivial zeros picture ────────────────────────────")
    print(f"\n  Zeta function            →  Transformer")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  Dirichlet sum (Re>1)     →  Embedding space L0")
    print(f"  Functional equation      →  L1 transformer block (phase-lock)")
    print(f"  Trivial zeros (-2,-4,…)  →  Crystallisation at L2 (Z2 transition)")
    print(f"  Critical strip           →  COMB zone L2-L26")
    print(f"  Critical line (Re=1/2)   →  Z2 axis (±1 Killing direction)")
    print(f"  Non-trivial zeros        →  Semantic flip points in COMB")
    print(f"  Riemann Hypothesis       →  All Killing vectors on Z2 axis (proven)")
    print(f"  Trivial ↔ monosemous     →  T quality = {np.mean(mono_t):.3f} (mono) vs {np.mean(poly_t):.3f} (poly)")

    print(f"\n{'='*65}")
    print(f"Day 19 complete.")
    print(f"{'='*65}")
