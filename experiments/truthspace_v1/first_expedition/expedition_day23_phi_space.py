#!/usr/bin/env python3
"""
Expedition Day 23 — φ-Space Structure

Hypothesis:
  The perp longitude direction φ = perp_vec / |perp_vec| carries individual
  word identity. DC 309 showed that latitude θ encodes semantic CLASS. The
  question for Day 23: is φ-space (the 1535D equatorial plane) itself
  structured by real-world semantic attributes, or is it an opaque lookup code?

  Predictions:
  1. GEOGRAPHIC CLUSTERING: European cities (berlin, paris, madrid, vienna,
     london, rome) cluster together in φ-space; Asian cities (tokyo, beijing,
     seoul, mumbai) cluster separately. φ-PCA PC1 ≈ "Europe vs Asia".

  2. TAXONOMIC CLUSTERING: Large African mammals (elephant, rhinoceros,
     hippopotamus) cluster; marine animals (dolphin, whale) cluster separately;
     birds (penguin, eagle) cluster separately.

  3. φ ARITHMETIC: Analogical reasoning works in φ-space.
     tokyo − beijing + paris ≈ berlin  (swap Asian for European city)
     elephant − rhinoceros + dolphin ≈ whale  (swap land for marine mammal)

  4. φ STABILITY: The perp direction is stable across COMB layers — φ at L5
     correlates strongly with φ at L20 (identity is fixed early, not built up).

  5. CROSS-CLASS SHARING: Some φ directions are shared across classes,
     corresponding to real-world attributes ("large", "ancient", "Asian").

Measurements:
  1. φ vectors for all words at L14 (mid-COMB)
  2. Pairwise φ cosine similarity matrices — intra vs inter sub-class
  3. PCA on φ within each class — top 2 PCs, variance explained, sub-class separation
  4. φ arithmetic: nearest-neighbour retrieval in φ-space
  5. Cross-layer φ stability: cosine(φ_L5, φ_L20) for all words
  6. Cross-class φ sharing: PCA across all classes — common axes?
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
CRYST_LAYER  = 2
MID_COMB     = 14

CITIES = {
    'Europe': ['berlin', 'paris', 'madrid', 'vienna', 'london', 'rome'],
    'Asia':   ['tokyo', 'beijing', 'seoul', 'mumbai', 'bangkok'],
    'Other':  ['cairo', 'sydney', 'nairobi'],
}

ANIMALS = {
    'large_land': ['elephant', 'rhinoceros', 'hippopotamus', 'giraffe'],
    'primate':    ['chimpanzee', 'gorilla', 'orangutan'],
    'marine':     ['dolphin', 'whale', 'octopus'],
    'bird':       ['penguin', 'eagle', 'parrot'],
    'reptile':    ['crocodile', 'python', 'iguana'],
}

ELEMENTS = {
    'noble_gas':  ['helium', 'neon', 'argon'],
    'atmospheric':['nitrogen', 'oxygen'],
    'light_solid':['carbon', 'silicon', 'sulfur'],
    'metal':      ['iron', 'copper', 'gold', 'silver'],
    'reactive':   ['hydrogen', 'sodium', 'potassium'],
}

KILLING_PAIRS = {
    'plural':      [('cat','cats'), ('dog','dogs'), ('tree','trees'),
                    ('bird','birds'), ('house','houses')],
    'gender':      [('man','woman'), ('king','queen'), ('boy','girl')],
    'comparative': [('big','bigger'), ('fast','faster'), ('old','older')],
}


def cos_sim(a, b):
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
    with __import__('torch').no_grad():
        out = model(**inputs, output_hidden_states=True)
    return np.stack([hs[0, pos, :].numpy() for hs in out.hidden_states])


def phi_vec(h, z2_axis):
    """Return the normalised perp direction (φ) for hidden state h."""
    hn    = h / (np.linalg.norm(h) + 1e-20)
    z2v   = float(np.dot(hn, z2_axis))
    perp  = hn - z2v * z2_axis
    pm    = np.linalg.norm(perp)
    return perp / (pm + 1e-20), pm, z2v


def pca_2d(vecs):
    """Return 2D PCA projections and explained variance ratios."""
    M = np.stack(vecs)
    M = M - M.mean(0)
    if M.shape[0] < 2: return M[:, :2], np.array([1.0, 0.0])
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    ev = s**2 / (s**2).sum()
    proj = M @ Vt[:2].T
    return proj, ev[:2], Vt[:2]


if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {SMALL_MODEL}...")
    tok   = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    n_layers = model.config.num_hidden_layers

    all_labeled_words = {}
    for region, words in CITIES.items():
        for w in words: all_labeled_words[w] = ('city', region)
    for taxon, words in ANIMALS.items():
        for w in words: all_labeled_words[w] = ('animal', taxon)
    for etype, words in ELEMENTS.items():
        for w in words: all_labeled_words[w] = ('element', etype)
    for rel, pairs in KILLING_PAIRS.items():
        for a, b in pairs:
            all_labeled_words[a] = ('killing_src', rel)
            all_labeled_words[b] = ('killing_tgt', rel)

    print(f"  Caching {len(all_labeled_words)} words...")
    cache = {}
    for w in sorted(all_labeled_words):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs

    cached_labels = {w: all_labeled_words[w] for w in cache}
    print(f"  Cached {len(cache)} words.")

    # Build Z2 axis
    comb_deltas = []
    for rel, pairs in KILLING_PAIRS.items():
        for L in range(CRYST_LAYER, n_layers - 2):
            ds = [cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                  for a, b in pairs if a in cache and b in cache]
            if ds:
                d = np.mean(ds, axis=0)
                comb_deltas.append(d / (np.linalg.norm(d) + 1e-20))
    _, sv, Vt = np.linalg.svd(np.stack(comb_deltas), full_matrices=False)
    z2_axis = Vt[0]
    print(f"  Z2 axis: {100*sv[0]**2/np.sum(sv**2):.2f}% variance\n")

    print(f"{'='*70}")
    print(f"DAY 23 — φ-Space Structure")
    print(f"{'='*70}")

    # ── Precompute φ vectors at multiple layers ───────────────────────────────
    phi_at = {}
    for w in cache:
        phi_at[w] = {}
        for L in [5, MID_COMB, 20, 26]:
            p, pm, z2v = phi_vec(cache[w][L].astype(np.float64), z2_axis)
            phi_at[w][L] = (p, pm, z2v)

    # ── Section 1: Intra vs inter sub-class φ similarity ─────────────────────
    print(f"\n── Section 1: Sub-class φ clustering ───────────────────────────────")
    print(f"  Mean cosine similarity between φ vectors within and between sub-classes.")
    print(f"  Prediction: intra > inter for each class.\n")

    def subclass_phi_analysis(group_dict, group_name, layer=MID_COMB):
        subclasses = {k: [w for w in v if w in cache] for k, v in group_dict.items()}
        subclasses = {k: v for k, v in subclasses.items() if len(v) >= 2}
        if len(subclasses) < 2:
            return
        print(f"  ── {group_name} (L{layer}) ──")
        intra_sims, inter_sims = [], []
        all_words = [(w, k) for k, ws in subclasses.items() for w in ws]
        for i, (wi, ki) in enumerate(all_words):
            for j, (wj, kj) in enumerate(all_words):
                if j <= i: continue
                s = cos_sim(phi_at[wi][layer][0], phi_at[wj][layer][0])
                if ki == kj: intra_sims.append((s, wi, wj, ki))
                else:         inter_sims.append((s, wi, wj, ki, kj))

        intra_mean = np.mean([x[0] for x in intra_sims]) if intra_sims else 0
        inter_mean = np.mean([x[0] for x in inter_sims]) if inter_sims else 0
        print(f"    Intra-subclass mean φ sim: {intra_mean:+.4f}  "
              f"Inter-subclass mean φ sim: {inter_mean:+.4f}  "
              f"Δ = {intra_mean - inter_mean:+.4f}")

        if intra_mean - inter_mean > 0.05:
            print(f"    ✓ CLUSTERING CONFIRMED — sub-classes separate in φ-space")
        elif intra_mean - inter_mean > 0.01:
            print(f"    ~ WEAK CLUSTERING")
        else:
            print(f"    ✗ NO CLUSTERING — φ-space is isotropic for this class")

        print(f"    Top 3 within-subclass pairs:")
        for s, wi, wj, k in sorted(intra_sims, reverse=True)[:3]:
            print(f"      [{k}] {wi} / {wj}: φ_cos = {s:+.4f}")
        print(f"    Bottom 3 cross-subclass pairs (most similar):")
        for s, wi, wj, ki, kj in sorted(inter_sims, reverse=True)[:3]:
            print(f"      [{ki} / {kj}] {wi} / {wj}: φ_cos = {s:+.4f}")
        print()

    subclass_phi_analysis(CITIES,   "Cities",   MID_COMB)
    subclass_phi_analysis(ANIMALS,  "Animals",  MID_COMB)
    subclass_phi_analysis(ELEMENTS, "Elements", MID_COMB)

    # ── Section 2: PCA on φ-space within each class ───────────────────────────
    print(f"\n── Section 2: PCA on φ-space ────────────────────────────────────────")
    print(f"  Top 2 PCA components of φ vectors within each class.")
    print(f"  Do PC1 / PC2 separate known sub-class labels?\n")

    def pca_separation(group_dict, group_name, layer=MID_COMB):
        all_words_labeled = [(w, k) for k, ws in group_dict.items()
                             for w in ws if w in cache]
        if len(all_words_labeled) < 4:
            return
        words   = [x[0] for x in all_words_labeled]
        labels  = [x[1] for x in all_words_labeled]
        phi_vecs = [phi_at[w][layer][0] for w in words]

        proj, ev, axes = pca_2d(phi_vecs)

        print(f"  ── {group_name} φ-PCA (L{layer}) ──")
        print(f"    PC1 var: {100*ev[0]:.1f}%   PC2 var: {100*ev[1]:.1f}%")
        print(f"    {'Word':<14} {'Subclass':<14} PC1      PC2")
        print("    " + "─" * 46)
        for w, lbl, (p1, p2) in zip(words, labels, proj):
            print(f"    {w:<14} {lbl:<14} {p1:+6.3f}  {p2:+6.3f}")

        unique_labels = list(dict.fromkeys(labels))
        if len(unique_labels) >= 2:
            centroids = {l: np.mean([proj[i] for i, lb in enumerate(labels) if lb == l],
                                    axis=0)
                         for l in unique_labels}
            print(f"\n    Sub-class centroids in PC1–PC2 space:")
            for l, c in centroids.items():
                print(f"      {l:<14}: PC1={c[0]:+.3f}  PC2={c[1]:+.3f}")
            centroid_seps = []
            ks = list(centroids.keys())
            for i in range(len(ks)):
                for j in range(i+1, len(ks)):
                    sep = np.linalg.norm(centroids[ks[i]] - centroids[ks[j]])
                    centroid_seps.append(sep)
            print(f"    Mean centroid separation: {np.mean(centroid_seps):.3f}")
        print()

    pca_separation(CITIES,   "Cities",   MID_COMB)
    pca_separation(ANIMALS,  "Animals",  MID_COMB)
    pca_separation(ELEMENTS, "Elements", MID_COMB)

    # ── Section 3: φ arithmetic ───────────────────────────────────────────────
    print(f"\n── Section 3: φ arithmetic ──────────────────────────────────────────")
    print(f"  Test: a − b + c ≈ d  in φ-space?")
    print(f"  Retrieve nearest word by φ cosine similarity.\n")

    all_phi_vecs = {w: phi_at[w][MID_COMB][0] for w in cache}

    def phi_nearest(query_phi, exclude=(), top=3):
        sims = [(cos_sim(query_phi, phi_at[w][MID_COMB][0]), w)
                for w in cache if w not in exclude]
        return sorted(sims, reverse=True)[:top]

    def phi_analogy(a, b, c, expected, label):
        if not all(w in cache for w in [a, b, c]):
            print(f"  SKIP (missing words): {label}")
            return
        qa = phi_at[a][MID_COMB][0]
        qb = phi_at[b][MID_COMB][0]
        qc = phi_at[c][MID_COMB][0]
        query = qa - qb + qc
        query = query / (np.linalg.norm(query) + 1e-20)
        nn = phi_nearest(query, exclude=(a, b, c))
        found = nn[0][1] if nn else '???'
        hit = '✓' if found == expected else '✗'
        print(f"  {hit} φ({a}) − φ({b}) + φ({c}) → expected={expected}")
        for sim, w in nn:
            marker = ' ←' if w == expected else ''
            print(f"      {w:<16} φ_cos={sim:.4f}{marker}")
        print()

    print("  CITY analogies (geographic swap):")
    phi_analogy('tokyo',   'beijing', 'paris',   'berlin',    'Asian→European capital')
    phi_analogy('berlin',  'paris',   'tokyo',   'beijing',   'European→Asian capital')
    phi_analogy('london',  'berlin',  'tokyo',   'beijing',   'British→Japanese capital?')
    print("  ANIMAL analogies (taxonomic swap):")
    phi_analogy('elephant','rhinoceros','dolphin','whale',     'land-mammal→marine')
    phi_analogy('penguin', 'eagle',    'dolphin', 'whale',    'bird→marine')
    phi_analogy('crocodile','python',  'eagle',   'penguin',  'reptile→bird')

    # ── Section 4: φ stability across COMB layers ─────────────────────────────
    print(f"\n── Section 4: φ stability across COMB layers ────────────────────────")
    print(f"  Is φ fixed early (identity encoded at L2) or built up across COMB?")
    print(f"  cos(φ_L5, φ_L20) per word.\n")

    print(f"  {'Word':<16} {'Class':<14} cos(φ_5,φ_14) cos(φ_5,φ_20) cos(φ_5,φ_26) verdict")
    print("  " + "─" * 70)

    all_stabilities = []
    test_words_stable = (
        [w for w in cache if cached_labels[w][0] == 'city'][:8] +
        [w for w in cache if cached_labels[w][0] == 'animal'][:8] +
        [w for w in cache if cached_labels[w][0] == 'element'][:4]
    )

    for w in sorted(test_words_stable)[:24]:
        cls, sub = cached_labels[w]
        s1 = cos_sim(phi_at[w][5][0],  phi_at[w][MID_COMB][0])
        s2 = cos_sim(phi_at[w][5][0],  phi_at[w][20][0])
        s3 = cos_sim(phi_at[w][5][0],  phi_at[w][26][0])
        all_stabilities.append((s1, s2, s3))
        verdict = 'STABLE' if min(s1, s2, s3) > 0.8 else \
                  ('DRIFTING' if s3 < 0.5 else 'PARTIAL')
        print(f"  {w:<16} {sub:<14} {s1:.4f}       {s2:.4f}       {s3:.4f}       {verdict}")

    m1 = np.mean([x[0] for x in all_stabilities])
    m2 = np.mean([x[1] for x in all_stabilities])
    m3 = np.mean([x[2] for x in all_stabilities])
    print(f"\n  Mean cos(φ_L5, φ_Lx):  L14={m1:.4f}  L20={m2:.4f}  L26={m3:.4f}")
    if min(m1, m2, m3) > 0.8:
        print(f"  φ IS STABLE — identity is fixed at L5, maintained through COMB")
    elif m3 < 0.5:
        print(f"  φ DRIFTS — identity direction changes significantly across COMB")
    else:
        print(f"  φ PARTIALLY STABLE — moderate drift across COMB")

    # ── Section 5: Cross-class φ sharing ─────────────────────────────────────
    print(f"\n── Section 5: Cross-class φ sharing ─────────────────────────────────")
    print(f"  PCA across all words — do common φ axes correspond to shared attributes?")
    print(f"  Words with high PC1 loading vs low PC1 loading:\n")

    all_words_list = sorted(cache.keys())
    all_phi_matrix = np.stack([phi_at[w][MID_COMB][0] for w in all_words_list])
    all_phi_centered = all_phi_matrix - all_phi_matrix.mean(0)
    _, sv_all, Vt_all = np.linalg.svd(all_phi_centered, full_matrices=False)
    ev_all = sv_all**2 / (sv_all**2).sum()

    print(f"  Global φ PCA — top 5 components: "
          f"{100*ev_all[0]:.1f}% {100*ev_all[1]:.1f}% "
          f"{100*ev_all[2]:.1f}% {100*ev_all[3]:.1f}% {100*ev_all[4]:.1f}%")

    global_proj = all_phi_centered @ Vt_all[:3].T
    pc1_scores  = [(global_proj[i, 0], all_words_list[i]) for i in range(len(all_words_list))]
    pc1_scores.sort()

    print(f"\n  PC1 gradient (lowest → highest projection):")
    print(f"  {'Word':<16} {'Class':<12} PC1_score")
    print("  " + "─" * 40)
    endpoints = pc1_scores[:6] + [('...', '...')] + pc1_scores[-6:]
    for item in endpoints:
        if item[0] == '...':
            print(f"  ...")
            continue
        score, w = item
        cls, sub = cached_labels.get(w, ('?', '?'))
        print(f"  {w:<16} {sub:<12} {score:+.4f}")

    pc2_scores = [(global_proj[i, 1], all_words_list[i]) for i in range(len(all_words_list))]
    pc2_scores.sort()
    print(f"\n  PC2 gradient (lowest → highest projection):")
    print(f"  {'Word':<16} {'Class':<12} PC2_score")
    print("  " + "─" * 40)
    endpoints2 = pc2_scores[:6] + [('...', '...')] + pc2_scores[-6:]
    for item in endpoints2:
        if item[0] == '...':
            print(f"  ...")
            continue
        score, w = item
        cls, sub = cached_labels.get(w, ('?', '?'))
        print(f"  {w:<16} {sub:<12} {score:+.4f}")

    # ── Section 6: φ distance vs. known semantic distance ────────────────────
    print(f"\n── Section 6: φ distance vs. known semantic attributes ───────────────")
    print(f"  Cities: are geographically closer cities also closer in φ-space?\n")

    CITY_LATLON = {
        'tokyo':   (35.7, 139.7),  'beijing': (39.9, 116.4),
        'seoul':   (37.6, 127.0),  'mumbai':  (19.1,  72.9),
        'bangkok': (13.8, 100.5),  'berlin':  (52.5,  13.4),
        'paris':   (48.9,   2.4),  'madrid':  (40.4,  -3.7),
        'vienna':  (48.2,  16.4),  'london':  (51.5,  -0.1),
        'rome':    (41.9,  12.5),  'cairo':   (30.1,  31.2),
        'sydney':  (-33.9, 151.2), 'nairobi': (-1.3,  36.8),
    }

    city_words = [w for w in CITY_LATLON if w in cache]
    if len(city_words) >= 4:
        geo_dists, phi_dists = [], []
        for i, ci in enumerate(city_words):
            for j, cj in enumerate(city_words):
                if j <= i: continue
                lat1, lon1 = CITY_LATLON[ci]; lat2, lon2 = CITY_LATLON[cj]
                geo_d = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5
                phi_d = 1.0 - cos_sim(phi_at[ci][MID_COMB][0], phi_at[cj][MID_COMB][0])
                geo_dists.append(geo_d); phi_dists.append(phi_d)

        from scipy.stats import pearsonr
        r, pval = pearsonr(geo_dists, phi_dists)
        print(f"  Correlation(geographic_distance, φ_distance) = {r:.4f}  "
              f"(p={pval:.3f})")
        if abs(r) > 0.3:
            print(f"  {'POSITIVE' if r > 0 else 'NEGATIVE'} correlation — "
                  f"φ-space {'preserves' if r > 0 else 'inverts'} geographic proximity")
        else:
            print(f"  NO GEOGRAPHIC CORRELATION — φ-space does not encode location")

        print(f"\n  Closest pairs in φ-space vs. geographic proximity:")
        phi_pairs = sorted([(1-cos_sim(phi_at[ci][MID_COMB][0], phi_at[cj][MID_COMB][0]),
                             ci, cj) for i,ci in enumerate(city_words)
                            for j,cj in enumerate(city_words) if j>i])
        print(f"  {'Pair':<22} φ_dist  geo_dist  same_continent?")
        print("  " + "─" * 55)
        continents = {'tokyo':'Asia','beijing':'Asia','seoul':'Asia','mumbai':'Asia',
                      'bangkok':'Asia','berlin':'Europe','paris':'Europe','madrid':'Europe',
                      'vienna':'Europe','london':'Europe','rome':'Europe','cairo':'Africa',
                      'sydney':'Oceania','nairobi':'Africa'}
        for phi_d, ci, cj in phi_pairs[:8]:
            lat1,lon1=CITY_LATLON[ci]; lat2,lon2=CITY_LATLON[cj]
            geo_d=((lat1-lat2)**2+(lon1-lon2)**2)**0.5
            same = continents.get(ci,'?') == continents.get(cj,'?')
            print(f"  {ci+' / '+cj:<22} {phi_d:.4f}  {geo_d:7.1f}   "
                  f"{'YES' if same else 'NO'}")

    # ── Section 7: Summary ────────────────────────────────────────────────────
    print(f"\n── Section 7: Summary ───────────────────────────────────────────────")
    print(f"""
  The φ-space (perp longitude) is the 1535D identity space on the Bloch sphere.
  Day 23 tests whether this space is:
    (a) An opaque lookup code — each word at an arbitrary position, no structure
    (b) A semantic attribute space — positions encode real-world properties

  If (b): the Bloch sphere latitude-longitude description is complete:
    θ (latitude)  = semantic zone / class membership
    φ (longitude) = semantic content / individual attributes
  And the full model can be read as a spherical coordinate atlas.
    """)

    print(f"{'='*70}")
    print(f"Day 23 complete.")
    print(f"{'='*70}")
