#!/usr/bin/env python3
"""
Expedition Day 20 — Fourth Dimension Rotation

Hypothesis:
  The 23 COMB layers are NOT just preserving the Z2 crystal for trivial words.
  For non-trivial words (OOT, specialized vocabulary), they execute a slow
  rotation through the "fourth dimension" — the plane perpendicular to the
  Z2 axis — that accumulates across layers until the word resolves into its
  semantic basin.

  This is structurally identical to how non-trivial zeros of ζ(s) = 0 are
  resolved: the imaginary part t accumulates rotations in the complex plane
  until the oscillating sum cancels. The layer depth = the imaginary part t.

  Trivial words (bank, cats, run): already in their semantic basin at L2.
    → cos(h_word, cluster_centroid) HIGH at L2, FLAT across COMB.
    → Perpendicular (fourth-dim) component SMALL at L2.

  Non-trivial words (elephant, tokyo, cylindrical): off-basin at L2.
    → cos(h_word, cluster_centroid) LOW at L2, RISING across COMB.
    → Perpendicular component LARGE at L2, DECREASING across COMB (spiraling in).

  The layer at which cos peaks = the non-trivial zero (the resolution depth).

Measurements:
  1. Semantic cluster alignment across COMB layers
     For each word: cos(h_word[L], centroid_excluding_word[L]) for L in COMB.

  2. Within-cluster variance at each COMB layer
     Trivial clusters: low and flat. Non-trivial: starts high, decreases.

  3. The fourth-dimension rotation component
     At each layer, decompose the hidden state change Δh = h[L+1] - h[L]
     into Z2-axis component and perpendicular (fourth-dim) component.
     Trivial: changes mostly along Z2. Non-trivial: large perpendicular.

  4. Angular velocity measurement
     For non-trivial words: does the rotation toward the semantic basin
     have a constant angular velocity? Constant rate → the "t" in ζ(1/2 + it).

  5. DC 295 connection: resolution layers for specific words
     Tokyo resolves at L15/L22/L23. Does cos(h_tokyo, cities_centroid)
     peak at these same layers?
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL = "Qwen/Qwen2-1.5B-Instruct"
CRYST_LAYER = 2

# Semantic clusters
CLUSTERS = {
    'animals_large': {
        'trivial':     [],
        'nontrivial':  ['elephant', 'rhinoceros', 'crocodile', 'dolphin',
                        'chimpanzee', 'kangaroo', 'flamingo', 'penguin',
                        'giraffe', 'caterpillar'],
    },
    'cities': {
        'trivial':     [],
        'nontrivial':  ['tokyo', 'berlin', 'paris', 'madrid', 'beijing',
                        'rome', 'london', 'moscow', 'vienna', 'oslo'],
    },
    'plurals': {
        'trivial':     ['cats', 'dogs', 'trees', 'birds', 'houses',
                        'cars', 'books', 'chairs', 'windows', 'tables'],
        'nontrivial':  [],
    },
    'comparatives': {
        'trivial':     ['bigger', 'faster', 'stronger', 'older', 'smaller',
                        'taller', 'heavier', 'darker', 'softer', 'louder'],
        'nontrivial':  [],
    },
    'common_nouns': {
        'trivial':     ['bank', 'rock', 'spring', 'light', 'fire',
                        'watch', 'table', 'match', 'run', 'fall'],
        'nontrivial':  [],
    },
    'elements': {
        'trivial':     [],
        'nontrivial':  ['hydrogen', 'nitrogen', 'oxygen', 'carbon',
                        'helium', 'calcium', 'sodium', 'chlorine'],
    },
}

KILLING_PAIRS_Z2 = {
    'gender':      [('king','queen'), ('man','woman'), ('boy','girl')],
    'comparative': [('big','bigger'), ('fast','faster'), ('old','older')],
    'plural':      [('cat','cats'), ('dog','dogs'), ('tree','trees')],
}


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

    # All words to cache
    all_words = set()
    for cluster in CLUSTERS.values():
        all_words.update(cluster['trivial'])
        all_words.update(cluster['nontrivial'])
    for pairs in KILLING_PAIRS_Z2.values():
        for a, b in pairs:
            all_words.update([a, b])

    print(f"\n  Caching {len(all_words)} words...")
    cache = {}
    for w in sorted(all_words):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs
    print(f"  Cached {len(cache)} words.")

    print(f"\n{'='*65}")
    print(f"DAY 20 — Fourth Dimension Rotation")
    print(f"{'='*65}")

    # ── Compute Z2 axis from Killing vectors in COMB zone ─────────────────────
    comb_deltas = []
    for rel, pairs in KILLING_PAIRS_Z2.items():
        for L in range(CRYST_LAYER, n_layers - 2):
            ds = [cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                  for a, b in pairs if a in cache and b in cache]
            if ds:
                d = np.mean(ds, axis=0)
                comb_deltas.append(d / (np.linalg.norm(d) + 1e-20))
    D = np.stack(comb_deltas)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    z2_axis = Vt[0]
    print(f"\n  Z2 axis computed. First SV captures {100*sv[0]**2/np.sum(sv**2):.2f}% variance.")

    # ── Section 1: Semantic cluster alignment across COMB layers ──────────────
    print(f"\n── Section 1: Cluster alignment across COMB layers ─────────────────")
    print(f"  cos(h_word[L], centroid_excluding_word[L])")
    print(f"  Trivial: flat at high value. Non-trivial: grows (spiral in).\n")

    comb_layers = list(range(CRYST_LAYER, n_layers + 1))

    cluster_results = {}
    for cname, cluster in CLUSTERS.items():
        for kind in ('trivial', 'nontrivial'):
            words = [w for w in cluster[kind] if w in cache]
            if len(words) < 3:
                continue
            key = f"{cname}_{kind}"
            print(f"  {key} ({len(words)} words):")
            word_curves = {}
            for w in words:
                curve = []
                for L in comb_layers:
                    others = [cache[ow][L].astype(np.float64)
                              for ow in words if ow != w and ow in cache]
                    if not others: curve.append(0.0); continue
                    centroid = np.mean(others, axis=0)
                    hw = cache[w][L].astype(np.float64)
                    curve.append(cos(hw / (np.linalg.norm(hw)+1e-20),
                                     centroid / (np.linalg.norm(centroid)+1e-20)))
                word_curves[w] = curve

            # Mean curve across words
            mean_curve = np.mean(list(word_curves.values()), axis=0)
            cluster_results[key] = mean_curve

            # Print compact representation
            # Show L2 (crystallisation), L14 (mid-COMB), L26 (pre-melt), L28 (output)
            check_layers = [CRYST_LAYER, 5, 10, 15, 20, 26, 28]
            header = "  L:  " + "".join(f"  L{l:02d}" for l in check_layers)
            print(header)
            row = "  val: "
            for l in check_layers:
                idx = comb_layers.index(l) if l in comb_layers else -1
                val = mean_curve[idx] if idx >= 0 else 0
                row += f"  {val:+.3f}"
            print(row)

            # Show trend: rising or flat?
            early = np.mean(mean_curve[:4])   # L2-L5
            late  = np.mean(mean_curve[-4:])  # L25-L28
            trend = '↑ RISING (non-trivial spiral)' if late - early > 0.05 else \
                    ('↓ FALLING' if early - late > 0.05 else '→ FLAT (trivial, already in basin)')
            print(f"  Early avg: {early:.4f}  Late avg: {late:.4f}  Trend: {trend}\n")

    # ── Section 2: Fourth-dimension rotation component ────────────────────────
    print(f"\n── Section 2: Fourth-dimension rotation component per word ─────────")
    print(f"  |Δh| projected onto Z2 axis vs. perpendicular ('fourth dim')")
    print(f"  Trivial: small perp at L2. Non-trivial: large perp at L2, shrinking.\n")

    test_trivial    = ['cats', 'dogs', 'bigger', 'faster', 'bank', 'rock']
    test_nontrivial = ['elephant', 'tokyo', 'berlin', 'hydrogen', 'nitrogen']
    test_words = [(w, 'trivial') for w in test_trivial if w in cache] + \
                 [(w, 'nontrivial') for w in test_nontrivial if w in cache]

    print(f"  {'Word':<14} {'Kind':<12} "
          + "".join(f" L{l:02d}_perp" for l in [2,5,10,15,20,26]) +
          "  trend")
    print("  " + "─" * 90)

    perp_curves = {}
    for w, kind in test_words:
        perps = []
        for L in range(CRYST_LAYER, n_layers + 1):
            # Decompose the layer-to-layer change in hidden state
            hw = cache[w][L].astype(np.float64)
            hw_n = hw / (np.linalg.norm(hw) + 1e-20)
            # Project onto Z2 axis
            z2_component = np.dot(hw_n, z2_axis) * z2_axis
            perp_component = hw_n - z2_component
            perp_mag = float(np.linalg.norm(perp_component))
            perps.append(perp_mag)
        perp_curves[(w, kind)] = perps
        check = [2, 5, 10, 15, 20, 26]
        vals = [perps[l - CRYST_LAYER] if (l - CRYST_LAYER) < len(perps) else 0
                for l in check]
        trend = '↓ SHRINKING' if vals[-1] < vals[0] - 0.02 else \
                ('→ FLAT' if abs(vals[-1] - vals[0]) < 0.02 else '↑ GROWING')
        row = f"  {w:<14} {kind:<12} " + " ".join(f"{v:.4f}" for v in vals) + f"  {trend}"
        print(row)

    # ── Section 3: Angular velocity — constant t? ─────────────────────────────
    print(f"\n── Section 3: Angular velocity across COMB layers ──────────────────")
    print(f"  Rate of change of cos(h_word[L], h_word[L+1])")
    print(f"  Constant rate → 'rotation speed' = the imaginary part t\n")

    print(f"  {'Word':<14} {'Kind':<12} "
          + "".join(f" Δ{l:02d}" for l in range(CRYST_LAYER, min(CRYST_LAYER+10, n_layers))) +
          "  mean_Δ  variance")
    print("  " + "─" * 100)

    for w, kind in test_words:
        deltas = []
        for L in range(CRYST_LAYER, n_layers):
            hL  = cache[w][L].astype(np.float64)
            hL1 = cache[w][L+1].astype(np.float64)
            # Angle between consecutive layers
            c = cos(hL / (np.linalg.norm(hL)+1e-20),
                    hL1 / (np.linalg.norm(hL1)+1e-20))
            angle = float(np.arccos(np.clip(c, -1, 1)))
            deltas.append(angle)
        first10 = deltas[:10]
        mean_d = np.mean(deltas[1:-2])  # exclude boundary layers
        var_d  = np.std(deltas[1:-2])
        row = f"  {w:<14} {kind:<12} " + " ".join(f"{d:.3f}" for d in first10) + \
              f"  {mean_d:.4f}  {var_d:.4f}"
        print(row)

    # ── Section 4: Resolution layers — where does alignment peak? ─────────────
    print(f"\n── Section 4: Resolution layer — where does cluster alignment peak? ─")
    print(f"  For non-trivial words: the peak layer = the non-trivial zero depth\n")

    # Use cities cluster for tokyo/berlin
    city_words = [w for w in ['tokyo', 'berlin', 'paris', 'madrid', 'beijing']
                  if w in cache]
    animal_words = [w for w in ['elephant', 'rhinoceros', 'dolphin', 'penguin', 'kangaroo']
                    if w in cache]
    plural_words = [w for w in ['cats', 'dogs', 'trees', 'birds', 'houses']
                    if w in cache]

    for group_name, group_words in [('cities', city_words),
                                     ('animals', animal_words),
                                     ('plurals', plural_words)]:
        if len(group_words) < 3: continue
        print(f"  {group_name}:")
        for w in group_words:
            others = [ow for ow in group_words if ow != w]
            peak_layer, peak_val = CRYST_LAYER, -1.0
            curve = []
            for L in comb_layers:
                centroids = [cache[ow][L].astype(np.float64) for ow in others if ow in cache]
                if not centroids: curve.append(0.0); continue
                c_mean = np.mean(centroids, axis=0)
                hw = cache[w][L].astype(np.float64)
                v = cos(hw / (np.linalg.norm(hw)+1e-20),
                        c_mean / (np.linalg.norm(c_mean)+1e-20))
                curve.append(v)
                if v > peak_val:
                    peak_val, peak_layer = v, L
            l2_val = curve[0]
            # Short curve display
            compact = " ".join(f"{v:.2f}" for v in curve[::3])
            print(f"    {w:<12} L2={l2_val:+.4f}  peak=L{peak_layer:02d}({peak_val:+.4f})  [{compact}]")
        print()

    # ── Section 5: Summary table ──────────────────────────────────────────────
    print(f"\n── Section 5: Summary ───────────────────────────────────────────────")

    trivial_early = []
    trivial_late  = []
    nontrivial_early = []
    nontrivial_late  = []

    for key, curve in cluster_results.items():
        early = np.mean(curve[:4])
        late  = np.mean(curve[-4:])
        if 'trivial' in key and 'nontrivial' not in key:
            trivial_early.append(early)
            trivial_late.append(late)
        elif 'nontrivial' in key:
            nontrivial_early.append(early)
            nontrivial_late.append(late)

    if trivial_early:
        print(f"  Trivial clusters:    L2 alignment={np.mean(trivial_early):.4f}  "
              f"L26 alignment={np.mean(trivial_late):.4f}  "
              f"Δ={np.mean(trivial_late)-np.mean(trivial_early):+.4f}")
    if nontrivial_early:
        print(f"  Non-trivial clusters: L2 alignment={np.mean(nontrivial_early):.4f}  "
              f"L26 alignment={np.mean(nontrivial_late):.4f}  "
              f"Δ={np.mean(nontrivial_late)-np.mean(nontrivial_early):+.4f}")

    print(f"\n  Prediction:")
    print(f"  Trivial Δ ≈ 0 (already in basin at L2)")
    print(f"  Non-trivial Δ > 0 (spiral in across COMB layers)")
    if trivial_early and nontrivial_early:
        t_delta = np.mean(trivial_late) - np.mean(trivial_early)
        n_delta = np.mean(nontrivial_late) - np.mean(nontrivial_early)
        if n_delta > t_delta + 0.02:
            print(f"  RESULT: CONFIRMED — non-trivial rises {n_delta:+.4f} vs trivial {t_delta:+.4f}")
        else:
            print(f"  RESULT: NOT CONFIRMED — non-trivial {n_delta:+.4f}, trivial {t_delta:+.4f}")

    print(f"\n{'='*65}")
    print(f"Day 20 complete.")
    print(f"{'='*65}")
