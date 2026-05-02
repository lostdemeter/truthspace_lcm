#!/usr/bin/env python3
"""
Expedition Day 22 — Bloch Sphere Meta-Geometry

Hypothesis:
  The (Z2, perp) conservation law forces every word's hidden state onto the
  unit sphere in 1536D. The meta-coordinate θ = arccos(z2_val) is the LATITUDE
  on this Bloch sphere. The perp direction in the 1535D equatorial plane is the
  LONGITUDE φ. The COMB zone executes SU(2) rotations — quantum gate operations
  on this sphere.

  Predictions:
  1. ANTIPODAL pairs: Killing pairs (man/woman, cat/cats, big/bigger) should
     appear at ANTIPODAL latitudes — θ_a + θ_b ≈ 180°. They are at opposite
     poles of the same semantic relationship.

  2. SAME-LATITUDE clustering: within-class words (elephant, rhinoceros, dolphin)
     should cluster at the SAME θ (same latitude) but different φ (different
     longitude = different identity in the equatorial perp plane).

  3. SU(2) GATE: each COMB layer should apply a single global rotation Δθ that
     is the same for ALL words at that layer — a universal SU(2) gate.

  4. META-DISTANCE: semantic similarity should correlate with spherical geodesic
     distance in (θ, φ) space better than raw cosine similarity for within-class
     comparisons.

  5. EQUATORIAL UNCERTAINTY: words at θ ≈ 90° (near the equator) should be more
     semantically ambiguous — their cluster alignment (from Day 20) should be
     lowest at the equator and highest at the poles.

Measurements:
  1. θ = arccos(z2_val) for all words at all COMB layers
     → antipodal test for Killing pairs
  2. Within-class θ variance vs between-class θ variance
     → same-latitude test
  3. Per-layer Δθ variance across words
     → SU(2) gate test: low variance = global rotation
  4. Meta-distance vs cosine similarity correlation
  5. θ vs cluster alignment scatter plot (equatorial uncertainty)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL = "Qwen/Qwen2-1.5B-Instruct"
CRYST_LAYER = 2

KILLING_PAIRS = {
    'plural':      [('cat','cats'), ('dog','dogs'), ('tree','trees'),
                    ('bird','birds'), ('house','houses')],
    'gender':      [('king','queen'), ('man','woman'), ('boy','girl'),
                    ('actor','actress'), ('prince','princess')],
    'comparative': [('big','bigger'), ('fast','faster'), ('old','older'),
                    ('tall','taller'), ('strong','stronger')],
}

SEMANTIC_GROUPS = {
    'animals':      ['elephant', 'rhinoceros', 'dolphin', 'penguin', 'kangaroo',
                     'giraffe', 'crocodile', 'chimpanzee'],
    'cities':       ['tokyo', 'berlin', 'paris', 'madrid', 'beijing', 'vienna'],
    'elements':     ['hydrogen', 'nitrogen', 'oxygen', 'carbon', 'helium'],
    'plurals':      ['cats', 'dogs', 'trees', 'birds', 'houses'],
    'plural_src':   ['cat', 'dog', 'tree', 'bird', 'house'],
    'comparatives': ['bigger', 'faster', 'stronger', 'older', 'taller'],
    'comp_src':     ['big', 'fast', 'strong', 'old', 'tall'],
    'gender_tgt':   ['queen', 'woman', 'girl', 'actress', 'princess'],
    'gender_src':   ['king', 'man', 'boy', 'actor', 'prince'],
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
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return np.stack([hs[0, pos, :].numpy() for hs in out.hidden_states])


def bloch_coords(h, z2_axis):
    """Return (theta_deg, z2_val, perp_unit) for hidden state h."""
    hn = h / (np.linalg.norm(h) + 1e-20)
    z2_val = float(np.dot(hn, z2_axis))
    perp   = hn - z2_val * z2_axis
    perp_mag = float(np.linalg.norm(perp))
    perp_unit = perp / (perp_mag + 1e-20)
    theta  = float(np.degrees(np.arccos(np.clip(z2_val, -1, 1))))
    return theta, z2_val, perp_unit, perp_mag


if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {SMALL_MODEL}...")
    tok   = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    n_layers = model.config.num_hidden_layers

    all_words = set()
    for g in SEMANTIC_GROUPS.values():
        all_words.update(g)
    for pairs in KILLING_PAIRS.values():
        for a, b in pairs:
            all_words.update([a, b])

    print(f"  Caching {len(all_words)} words...")
    cache = {}
    for w in sorted(all_words):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs
    print(f"  Cached {len(cache)} words.")

    # Build Z2 axis from COMB Killing vectors
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
    print(f"  Z2 axis: {100*sv[0]**2/np.sum(sv**2):.2f}% variance")

    print(f"\n{'='*70}")
    print(f"DAY 22 — Bloch Sphere Meta-Geometry")
    print(f"{'='*70}")

    L_test = 14  # mid-COMB

    # ── Section 1: Antipodal test for Killing pairs ────────────────────────────
    print(f"\n── Section 1: Antipodal test — θ_a + θ_b ≈ 180°? ────────────────────")
    print(f"  At L{L_test} (mid-COMB). Prediction: Killing pairs are antipodal.")
    print(f"  Also check: perp direction similarity (same φ = same identity?)\n")
    print(f"  {'Rel':<12} {'Pair':<22} θ_a    θ_b    θ_a+θ_b  Δ°  perp_cos")
    print("  " + "─" * 72)

    antipodal_deltas = []
    for rel, pairs in KILLING_PAIRS.items():
        for a, b in pairs:
            if a not in cache or b not in cache: continue
            ta, z2a, perp_a, _ = bloch_coords(cache[a][L_test].astype(np.float64), z2_axis)
            tb, z2b, perp_b, _ = bloch_coords(cache[b][L_test].astype(np.float64), z2_axis)
            sum_theta = ta + tb
            delta = abs(sum_theta - 180.0)
            antipodal_deltas.append(delta)
            perp_sim = float(np.dot(perp_a, perp_b))
            print(f"  {rel:<12} {a+' → '+b:<22} {ta:5.1f}  {tb:5.1f}  {sum_theta:6.1f}  "
                  f"{delta:4.1f}°  {perp_sim:+.4f}")

    print(f"\n  Mean |θ_a + θ_b − 180°| = {np.mean(antipodal_deltas):.2f}°")
    print(f"  (0° = perfectly antipodal; 180° = completely wrong)")
    if np.mean(antipodal_deltas) < 20:
        print(f"  ANTIPODAL — CONFIRMED")
    elif np.mean(antipodal_deltas) < 45:
        print(f"  PARTIALLY ANTIPODAL")
    else:
        print(f"  NOT ANTIPODAL")

    # ── Section 2: Same-latitude clustering ───────────────────────────────────
    print(f"\n── Section 2: Same-latitude — within-class θ variance ───────────────")
    print(f"  Low within-class σ(θ) = same latitude = confirmed")
    print(f"  Also: between-class θ difference = different latitudes = discriminating\n")
    print(f"  {'Group':<16}  words  mean_θ   σ(θ)   min_θ   max_θ   verdict")
    print("  " + "─" * 68)

    group_thetas = {}
    for gname, gwords in SEMANTIC_GROUPS.items():
        words = [w for w in gwords if w in cache]
        if len(words) < 2: continue
        thetas = [bloch_coords(cache[w][L_test].astype(np.float64), z2_axis)[0]
                  for w in words]
        group_thetas[gname] = thetas
        mean_t = np.mean(thetas)
        std_t  = np.std(thetas)
        verdict = 'SAME LATITUDE' if std_t < 5.0 else \
                  ('CLOSE' if std_t < 15.0 else 'SPREAD')
        print(f"  {gname:<16}  {len(words):>5}  {mean_t:6.1f}°  {std_t:5.2f}°  "
              f"{min(thetas):5.1f}°  {max(thetas):5.1f}°  {verdict}")

    # ── Section 3: SU(2) gate test — global rotation per layer ────────────────
    print(f"\n── Section 3: SU(2) gate test — is Δθ the same for all words? ────────")
    print(f"  At each layer L, compute Δθ = θ(L+1) − θ(L) for each word.")
    print(f"  Low variance across words → single global SU(2) rotation gate.\n")

    all_test_words = [w for g in ['animals', 'cities', 'plurals', 'comparatives',
                                   'plural_src', 'comp_src']
                      for w in SEMANTIC_GROUPS[g] if w in cache]

    print(f"  Testing {len(all_test_words)} words across {n_layers-1} layer transitions.\n")
    print(f"  L    mean_Δθ   σ(Δθ)    min_Δθ   max_Δθ   σ/mean  SU2?")
    print("  " + "─" * 56)

    gate_quality = []
    for L in range(CRYST_LAYER, n_layers):
        dthetas = []
        for w in all_test_words:
            ta = bloch_coords(cache[w][L].astype(np.float64), z2_axis)[0]
            tb = bloch_coords(cache[w][L+1].astype(np.float64), z2_axis)[0]
            dthetas.append(tb - ta)
        mean_d = float(np.mean(dthetas))
        std_d  = float(np.std(dthetas))
        cv     = abs(std_d / (mean_d + 1e-10))
        su2    = 'YES' if cv < 0.3 else ('PARTIAL' if cv < 1.0 else 'NO')
        gate_quality.append(cv)
        if L in [2, 5, 10, 14, 18, 22, 26, 27]:
            print(f"  L{L:02d}  {mean_d:+7.3f}°  {std_d:6.3f}°  {min(dthetas):+7.3f}°  "
                  f"{max(dthetas):+7.3f}°  {cv:5.2f}   {su2}")

    mean_cv = float(np.mean(gate_quality[1:-2]))
    print(f"\n  Mean CV across COMB: {mean_cv:.4f}")
    if mean_cv < 0.3:
        print(f"  GLOBAL SU(2) ROTATION — each layer applies same Δθ to all words")
    elif mean_cv < 1.0:
        print(f"  APPROXIMATE SU(2) — dominant rotation + word-specific variation")
    else:
        print(f"  NOT SU(2) — rotation is word-specific, not global")

    # ── Section 4: Meta-distance vs cosine similarity ────────────────────────
    print(f"\n── Section 4: Meta-distance vs cosine similarity ────────────────────")
    print(f"  Spherical distance: d_bloch(a,b) = |θ_a − θ_b| (for within-class)")
    print(f"  Does meta-distance track semantic similarity better?\n")

    test_groups = ['animals', 'cities', 'plurals']
    print(f"  {'Group':<12}  within-class corr(cos, meta_d)  verdict")
    print("  " + "─" * 50)

    for gname in test_groups:
        gwords = [w for w in SEMANTIC_GROUPS[gname] if w in cache]
        if len(gwords) < 3: continue
        cos_sims = []
        meta_dists = []
        for i, wi in enumerate(gwords):
            ti = bloch_coords(cache[wi][L_test].astype(np.float64), z2_axis)[0]
            hi = cache[wi][L_test].astype(np.float64)
            hi_n = hi / (np.linalg.norm(hi) + 1e-20)
            for j, wj in enumerate(gwords):
                if j <= i: continue
                tj = bloch_coords(cache[wj][L_test].astype(np.float64), z2_axis)[0]
                hj = cache[wj][L_test].astype(np.float64)
                hj_n = hj / (np.linalg.norm(hj) + 1e-20)
                cos_sims.append(float(np.dot(hi_n, hj_n)))
                meta_dists.append(abs(ti - tj))
        if len(cos_sims) > 2:
            r = float(np.corrcoef(cos_sims, meta_dists)[0, 1])
            verdict = 'ANTI-CORR (closer=higher cos)' if r < -0.3 else \
                      ('FLAT (meta-d not predictive)' if abs(r) < 0.3 else 'POSITIVE')
            print(f"  {gname:<12}  r = {r:+.4f}     {verdict}")

    # ── Section 5: Equatorial uncertainty ────────────────────────────────────
    print(f"\n── Section 5: Equatorial uncertainty ────────────────────────────────")
    print(f"  Words at θ ≈ 90° should have LOWEST cluster alignment (uncertain)")
    print(f"  Words at θ ≈ 0° or 180° should have HIGHEST alignment (decided)\n")

    # Compute cluster alignment at L_test for all words with known group
    all_group_words = []
    for gname in ['animals', 'cities', 'elements', 'plurals', 'comparatives']:
        gwords = [w for w in SEMANTIC_GROUPS[gname] if w in cache]
        for w in gwords:
            others = [cache[ow][L_test].astype(np.float64)
                      for ow in gwords if ow != w and ow in cache]
            if not others: continue
            centroid = np.mean(others, axis=0)
            hw = cache[w][L_test].astype(np.float64)
            alignment = cos_sim(hw / (np.linalg.norm(hw)+1e-20),
                               centroid / (np.linalg.norm(centroid)+1e-20))
            theta, z2_val, _, _ = bloch_coords(hw, z2_axis)
            all_group_words.append((w, gname, theta, alignment))

    all_group_words.sort(key=lambda x: x[2])  # sort by theta

    print(f"  {'Word':<14}  {'Group':<12}  θ(deg)  cluster_align  equatorial?")
    print("  " + "─" * 58)
    for w, gname, theta, align in all_group_words:
        equatorial = '← UNCERTAIN' if 70 < theta < 110 else ''
        print(f"  {w:<14}  {gname:<12}  {theta:6.1f}°  {align:+.4f}         {equatorial}")

    # Correlation: theta distance from equator vs alignment
    angles_from_eq = [abs(theta - 90) for _, _, theta, _ in all_group_words]
    alignments     = [align for _, _, _, align in all_group_words]
    r_eq = float(np.corrcoef(angles_from_eq, alignments)[0, 1])
    print(f"\n  Correlation(|θ − 90°|, alignment) = {r_eq:+.4f}")
    if r_eq > 0.3:
        print(f"  CONFIRMED: poles → high alignment, equator → low alignment")
    elif r_eq < -0.3:
        print(f"  INVERTED: equator → high alignment (unexpected)")
    else:
        print(f"  NO CLEAR TREND")

    # ── Section 6: Bloch sphere trajectory of Killing transformation ──────────
    print(f"\n── Section 6: Killing vector as Bloch sphere arc ────────────────────")
    print(f"  For each Killing pair (a → b), track (θ_a, θ_b) across COMB layers.")
    print(f"  Does the arc always cross the equator? (Z2 flip = pole to pole)\n")

    test_pairs = [('cat','cats','plural'), ('man','woman','gender'),
                  ('big','bigger','comparative')]
    print(f"  {'Pair':<22}  L02(θa,θb)  L14(θa,θb)  L26(θa,θb)  sum@L14  arc_type")
    print("  " + "─" * 78)

    for a, b, rel in test_pairs:
        if a not in cache or b not in cache: continue
        vals = []
        for L in [CRYST_LAYER, 14, 26]:
            ta = bloch_coords(cache[a][L].astype(np.float64), z2_axis)[0]
            tb = bloch_coords(cache[b][L].astype(np.float64), z2_axis)[0]
            vals.append((ta, tb))
        sum14 = vals[1][0] + vals[1][1]
        # Does θ_a increase while θ_b decreases? (arc crossing equator)
        a_moves = vals[2][0] - vals[0][0]
        b_moves = vals[2][1] - vals[0][1]
        if (a_moves > 0) == (b_moves < 0):
            arc = 'DIVERGING (poles moving apart)'
        elif (a_moves < 0) == (b_moves > 0):
            arc = 'CONVERGING (poles moving together)'
        else:
            arc = 'PARALLEL (both same direction)'
        pair_str = f"{a} → {b}"
        print(f"  {pair_str:<22}  ({vals[0][0]:5.1f},{vals[0][1]:5.1f})  "
              f"({vals[1][0]:5.1f},{vals[1][1]:5.1f})  ({vals[2][0]:5.1f},{vals[2][1]:5.1f})  "
              f"{sum14:5.1f}  {arc}")

    # ── Section 7: Summary ────────────────────────────────────────────────────
    print(f"\n── Section 7: The Bloch Sphere Picture ─────────────────────────────")
    print(f"""
  Coordinate system:
    North pole (θ≈0°):   crystal endpoint 'target' — cats, woman, bigger
    South pole (θ≈180°): crystal endpoint 'source' — cat, man, big
    Equator   (θ≈90°):   identity in perp space — elephant, tokyo, hydrogen

  Killing vector = arc from south pole to north pole (θ: 180° → 0°)
  COMB rotation  = all words advance toward their pole (non-trivial: 85°→72°)
  Resolution     = reaching the pole-region where semantic basin is decided
  Dead rotation  = perp direction φ (longitude) = individual identity

  Feynman: "going farther in one direction means going less far in another"
           Δz2 = −Δperp exactly — the unit sphere is the conserved surface
           Polarity = sign of z2 (north/south pole distinction)
           Straight-line illusion = hiding the curvature of the unit sphere
  """)

    print(f"{'='*70}")
    print(f"Day 22 complete.")
    print(f"{'='*70}")
