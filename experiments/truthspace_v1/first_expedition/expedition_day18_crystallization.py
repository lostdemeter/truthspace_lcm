#!/usr/bin/env python3
"""
Expedition Day 18 — The Crystallisation Transition

Context (arXiv 2505.09117 — Discrete Time Quasi-Crystals in Rydberg chains):
  DTQC arises when two incommensurate DTC phases are coupled at a boundary.
  The crystallised phase only exists within a specific driving-frequency window.
  Outside it: decouple (too fast) or thermalize (too slow).
  Order parameter: Z2 binary (antiferromagnetic ±1).
  Entanglement: low in bulk, confined to boundary.

  Our transformer correspondences (from Day 17):
  - L0 (embedding) ↔ L-subsystem (word-frequency DTC)
  - L3-L26 (COMB) ↔ crystallised phase (meaning-frequency)
  - L0→L2 ↔ boundary coupling / phase-locking zone
  - ±1 Killing axis ↔ Z2/Z2' order parameter
  - Lagrange L4/L5 ↔ sum/difference beat-frequency peaks

Measurements:
  1. Fine-grained scan L0→L6: find exact crystallisation layer
     - france↔paris cos_ab at every layer (not sampled)
     - Within-relationship universality at every layer
     - Cross-relationship alignment (Z2 order) at every layer

  2. Phase transition characterisation
     - Is the transition sharp (1st order) or gradual (crossover)?
     - Fit logistic curve to order parameter vs layer depth
     - Find the "Curie layer" — half-maximum of order parameter

  3. Shortcut test: algebraic characterisation of L0→L3
     - Compute mean transformation matrix (PCA of hidden state changes)
     - Does applying this matrix to L0 directly reproduce L3 geometry?
     - Test on held-out word pairs not used in fitting

  4. Rydberg analogy: entanglement entropy proxy
     - At each layer, compute variance of individual pair deltas
     - Variance = geometric analogue of entanglement entropy
     - Should peak at boundary layers, be low in COMB bulk

  5. Z2 symmetry measurement
     - At each layer: project all Killing deltas onto first PCA axis
     - Measure how much variance is captured by first component
     - If ~100%: full Z2 crystallisation; if <50%: amorphous/disordered
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL = "Qwen/Qwen2-1.5B-Instruct"

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

# Pairs used ONLY for shortcut test (not in fitting)
HOLDOUT_PAIRS = {
    'gender':      [('brother','sister'), ('father','mother'), ('son','daughter')],
    'comparative': [('tall','taller'), ('heavy','heavier'), ('dark','darker')],
    'capital':     [('china','beijing'), ('russia','moscow')],
}

def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20: return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {SMALL_MODEL}...")
    tok = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    n = model.config.num_hidden_layers
    h = model.config.hidden_size
    print(f"  Loaded: {n} layers, hidden={h}")
    return model, tok, n, h

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
    import torch
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return np.stack([hs[0, pos, :].numpy() for hs in out.hidden_states])

if __name__ == '__main__':
    import torch
    model, tok, n_layers, hidden_dim = load_model()

    # Cache everything
    all_words = set()
    for d in (KILLING_PAIRS, HOLDOUT_PAIRS):
        for pairs in d.values():
            for a, b in pairs:
                all_words.update([a, b])

    print(f"\n  Caching {len(all_words)} words...")
    cache = {}
    for w in sorted(all_words):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs
    print(f"  Cached {len(cache)} words across {n_layers+1} layers.")

    print(f"\n{'='*65}")
    print(f"DAY 18 — Crystallisation Transition")
    print(f"{'='*65}")

    # ── Section 1: Fine-grained scan L0→L8 — finding the transition ──────────
    print(f"\n── Section 1: Fine-grained crystallisation scan (every layer) ──────")

    # Compute per-layer mean Killing delta for every relationship
    rel_ld = {}
    for rel, pairs in KILLING_PAIRS.items():
        ld = {}
        for L in range(n_layers + 1):
            ds = [cache[b][L] - cache[a][L]
                  for a, b in pairs if a in cache and b in cache]
            if ds:
                ld[L] = np.mean(ds, axis=0)
        rel_ld[rel] = ld

    # (a) france↔paris cos_ab at every layer
    print(f"\n  (a) france↔paris cos_ab across all layers")
    print(f"  L:  ", end='')
    vals = []
    for L in range(n_layers + 1):
        if 'france' in cache and 'paris' in cache:
            pa = cache['france'][L].astype(np.float64)
            pb = cache['paris'][L].astype(np.float64)
            c  = cos(pa / (np.linalg.norm(pa)+1e-20),
                     pb / (np.linalg.norm(pb)+1e-20))
            vals.append((L, c))

    # Print as bar chart (text)
    max_val = max(v for _, v in vals)
    for L, c in vals:
        bar = '█' * int(40 * max(0, c) / max(max_val, 0.01))
        flag = ' ← transition' if L > 0 and abs(c - vals[L-1][1]) > 0.1 else ''
        print(f"  L{L:02d}: {c:+.4f}  {bar}{flag}")

    # (b) Z2 order parameter: how collapsed are all Killing vectors to ±1?
    print(f"\n  (b) Z2 order parameter at each layer")
    print(f"      = fraction of cross-rel variance captured by 1st PCA component")
    print(f"      (1.0 = fully crystallised; 0.2 = amorphous)\n")

    z2_vals = []
    for L in range(n_layers + 1):
        # Collect all normalised mean deltas at this layer
        all_d = []
        for rel, ld in rel_ld.items():
            if L in ld:
                d = ld[L]
                dn = d / (np.linalg.norm(d) + 1e-20)
                all_d.append(dn)
        if len(all_d) < 2:
            z2_vals.append((L, 0.0))
            continue
        D = np.stack(all_d)  # (n_rels, hidden)
        # PCA: singular values of D
        _, sv, _ = np.linalg.svd(D, full_matrices=False)
        sv2 = sv**2
        z2 = float(sv2[0] / (sv2.sum() + 1e-20))
        z2_vals.append((L, z2))

    for L, z2 in z2_vals:
        bar = '█' * int(40 * z2)
        flag = ' ← CRYSTALLISED' if z2 > 0.9 else (' ← forming' if z2 > 0.5 else '')
        print(f"  L{L:02d}: {z2:.4f}  {bar}{flag}")

    # (c) Within-relationship universality at every layer (plural — clearest signal)
    print(f"\n  (c) Plural universality at every layer (mean pairwise cos of pair deltas)")
    for L in range(n_layers + 1):
        pair_ds = []
        for a, b in KILLING_PAIRS['plural']:
            if a in cache and b in cache:
                d = cache[b][L] - cache[a][L]
                pair_ds.append(d / (np.linalg.norm(d) + 1e-20))
        if len(pair_ds) < 2:
            continue
        sims = [cos(pair_ds[i], pair_ds[j])
                for i in range(len(pair_ds))
                for j in range(i+1, len(pair_ds))]
        u = np.mean(sims)
        bar = '█' * int(40 * max(0, u))
        flag = ' ← CRYSTALLISED' if u > 0.9 else (' ← forming' if u > 0.4 else '')
        print(f"  L{L:02d}: {u:+.4f}  {bar}{flag}")

    # ── Section 2: Phase transition shape ────────────────────────────────────
    print(f"\n── Section 2: Phase transition — sharp or gradual? ─────────────────")
    print(f"  Layer-to-layer jump in Z2 order parameter (dZ2/dL)\n")

    z2_arr = np.array([v for _, v in z2_vals])
    dz2 = np.diff(z2_arr)
    for i, dv in enumerate(dz2):
        bar = '█' * int(40 * abs(dv) / (abs(dz2).max() + 1e-10))
        sign = '+' if dv > 0 else '-'
        print(f"  L{i:02d}→L{i+1:02d}: {sign}{abs(dv):.4f}  {bar}")

    # ── Section 3: Shortcut test — characterise L0→L3 transform ─────────────
    print(f"\n── Section 3: Shortcut — can we learn the L0→L3 mapping? ──────────")
    print(f"  Fit a linear map T: h_L0 → h_L3 from training pairs")
    print(f"  Test: does T(h_L0_holdout) ≈ h_L3_holdout?\n")

    # Determine crystallisation layer from Z2 data
    cryst_layer = next((L for L, z2 in z2_vals if z2 > 0.85), 3)
    print(f"  Crystallisation layer detected: L{cryst_layer}")

    # Build training set: (h_L0, h_Lc) for all training pairs
    X_train, Y_train = [], []
    for pairs in KILLING_PAIRS.values():
        for a, b in pairs:
            for w in (a, b):
                if w in cache:
                    X_train.append(cache[w][0].astype(np.float64))
                    Y_train.append(cache[w][cryst_layer].astype(np.float64))

    X = np.stack(X_train)  # (N, hidden)
    Y = np.stack(Y_train)  # (N, hidden)

    # Least-squares: T = argmin ||XT - Y||_F, solved as T = (X^T X)^{-1} X^T Y
    # Use pseudoinverse for stability
    print(f"  Training on {len(X)} word vectors ({hidden_dim}-dim → {hidden_dim}-dim)...")
    T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (hidden, hidden)
    print(f"  Map T fitted. Shape: {T.shape}")

    # Evaluate on training set
    Y_pred_train = X @ T
    cos_train = np.mean([cos(Y_pred_train[i], Y[i]) for i in range(len(Y))])
    print(f"  Train cos(predicted, actual): {cos_train:.4f}")

    # Evaluate on holdout set
    ho_results = []
    for rel, pairs in HOLDOUT_PAIRS.items():
        for a, b in pairs:
            for w in (a, b):
                if w not in cache:
                    continue
                h0  = cache[w][0].astype(np.float64)
                hLc = cache[w][cryst_layer].astype(np.float64)
                h_pred = h0 @ T
                c_pred_actual = cos(h_pred, hLc)
                # Also compare to just using h0 directly (baseline)
                c_raw = cos(h0, hLc)
                ho_results.append((w, rel, c_raw, c_pred_actual))

    print(f"\n  Holdout words: cos(h_L0, h_Lc)  vs  cos(T(h_L0), h_Lc)\n")
    print(f"  {'Word':<14} {'Rel':<12} {'Baseline(raw)':<16} {'Mapped(T)'}")
    print("  " + "─" * 55)
    baseline_mean = np.mean([r[2] for r in ho_results])
    mapped_mean   = np.mean([r[3] for r in ho_results])
    for w, rel, c_raw, c_pred in ho_results:
        improvement = '↑' if c_pred > c_raw else '↓'
        print(f"  {w:<14} {rel:<12} {c_raw:+.4f}           {c_pred:+.4f}  {improvement}")
    print(f"\n  Mean baseline: {baseline_mean:.4f}   Mean mapped: {mapped_mean:.4f}   "
          f"Δ={mapped_mean - baseline_mean:+.4f}")

    # ── Section 4: Entanglement entropy proxy ────────────────────────────────
    print(f"\n── Section 4: Variance (EE proxy) at each layer ───────────────────")
    print(f"  Geometric EE = variance of individual pair deltas around mean")
    print(f"  Should peak at boundary layers L0→L1 and L27→L28, low in bulk\n")

    for rel in ('gender', 'plural', 'capital'):
        print(f"  {rel}:")
        pairs = KILLING_PAIRS[rel]
        for L in range(n_layers + 1):
            ds = [cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                  for a, b in pairs if a in cache and b in cache]
            if len(ds) < 2:
                continue
            ds_arr = np.stack(ds)
            # Variance = mean squared distance from mean
            mean_d = ds_arr.mean(axis=0)
            var = float(np.mean([np.linalg.norm(d - mean_d)**2 for d in ds]))
            # Normalise by magnitude of mean
            mag = np.linalg.norm(mean_d)
            rel_var = var / (mag**2 + 1e-20)
            bar = '█' * min(40, int(20 * rel_var / max(1, rel_var)))
            flag = ' ← boundary' if rel_var > 0.5 else ''
            print(f"    L{L:02d}: relvar={rel_var:.3f}  {bar}{flag}")
        print()

    # ── Section 5: Full Killing vector crystallisation summary ───────────────
    print(f"\n── Section 5: Killing vector coherence vs layer (all relationships) ─")
    print(f"  Mean cos(Δ_i, mean_Δ) per relationship — coherence of individual pairs\n")

    print(f"  {'L':<4}" + "".join(f"  {r:<12}" for r in KILLING_PAIRS))
    print("  " + "─" * (4 + 14 * len(KILLING_PAIRS)))

    for L in range(n_layers + 1):
        row = f"  {L:<4}"
        for rel, pairs in KILLING_PAIRS.items():
            ds = [cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                  for a, b in pairs if a in cache and b in cache]
            if not ds:
                row += f"  {'?':<12}"; continue
            ds_arr = np.stack(ds)
            mean_d = ds_arr.mean(axis=0)
            mean_dn = mean_d / (np.linalg.norm(mean_d) + 1e-20)
            coherence = np.mean([cos(d / (np.linalg.norm(d)+1e-20), mean_dn)
                                 for d in ds])
            row += f"  {coherence:+.4f}    "
        print(row)

    print(f"\n{'='*65}")
    print(f"Day 18 complete.")
    print(f"{'='*65}")
