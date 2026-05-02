#!/usr/bin/env python3
"""
Expedition Day 16 — k-NN Conditioned Delta Retrieval

Hypothesis (from DC 279/283):
  The model's COMB zone (L10-L20) uses content-specific channel routing.
  For 'australia → canberra', attention selects australia's specific
  interference configuration — NOT a global mean capital direction.

  Our Day 15 global delta is equivalent to averaging all photon paths from
  different sources to different detectors, then trying to navigate a new
  source with that average. It works for universal morphological relations
  (comparative, plural — same direction for all inputs) but fails for
  factual relations (capital — each country has its own path).

Fix: locally-conditioned delta
  For query (source, relationship):
    1. Find k nearest training-source concepts to source in projection space
    2. Weight each training pair's individual delta by cos(source, training_source)
    3. Apply weighted combination — local content-specific delta

Test:
  - Full Day 15 query bank (43 queries)
  - Compare: global mean vs k=1, k=3, k=5, k=10, softmax-temperature scaling
  - Focus on OOT queries where global mean failed
  - Measure: does conditioned retrieval recover OOT accuracy?

Secondary test:
  - Lagrange point mapping: between two 'massive' concepts (france/paris),
    find concepts forming near-equilateral triangles — the Trojan asteroids
    of semantic space.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

# ── Same query bank as Day 15 ─────────────────────────────────────────────────
FUNCTIONAL_RELS = {
    'capital':     [('france','paris'),('germany','berlin'),('italy','rome'),
                    ('spain','madrid'),('japan','tokyo'),('china','beijing'),
                    ('russia','moscow'),('brazil','brasilia')],
    'plural':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                    ('book','books'),('tree','trees'),('bird','birds'),
                    ('child','children'),('mouse','mice'),('foot','feet')],
    'gender':      [('king','queen'),('man','woman'),('boy','girl'),('actor','actress'),
                    ('prince','princess'),('son','daughter'),('father','mother'),
                    ('brother','sister'),('husband','wife'),('uncle','aunt')],
    'past':        [('run','ran'),('walk','walked'),('eat','ate'),('write','wrote'),
                    ('speak','spoke'),('take','took'),('go','went'),('see','saw')],
    'comparative': [('big','bigger'),('small','smaller'),('fast','faster'),
                    ('old','older'),('young','younger'),('strong','stronger')],
    'country_lang':[('france','french'),('germany','german'),('spain','spanish'),
                    ('italy','italian'),('russia','russian'),('china','chinese'),
                    ('japan','japanese'),('portugal','portuguese')],
}

QUERIES = [
    ('france','capital','paris'), ('germany','capital','berlin'),
    ('japan','capital','tokyo'), ('italy','capital','rome'),
    ('spain','capital','madrid'), ('china','capital','beijing'),
    ('russia','capital','moscow'), ('brazil','capital','brasilia'),
    ('australia','capital','canberra'), ('canada','capital','ottawa'),
    ('cat','plural','cats'), ('mouse','plural','mice'),
    ('child','plural','children'), ('foot','plural','feet'),
    ('tree','plural','trees'), ('book','plural','books'),
    ('city','plural','cities'), ('knife','plural','knives'),
    ('king','gender','queen'), ('man','gender','woman'),
    ('actor','gender','actress'), ('brother','gender','sister'),
    ('father','gender','mother'), ('prince','gender','princess'),
    ('emperor','gender','empress'), ('waiter','gender','waitress'),
    ('run','past','ran'), ('walk','past','walked'),
    ('eat','past','ate'), ('see','past','saw'),
    ('write','past','wrote'), ('speak','past','spoke'),
    ('swim','past','swam'), ('fly','past','flew'),
    ('big','comparative','bigger'), ('fast','comparative','faster'),
    ('old','comparative','older'), ('strong','comparative','stronger'),
    ('tall','comparative','taller'), ('heavy','comparative','heavier'),
    ('france','country_lang','french'), ('germany','country_lang','german'),
    ('japan','country_lang','japanese'),
]

TRAINING_PAIRS = {
    (rel, a.lower(), b.lower())
    for rel, pairs in FUNCTIONAL_RELS.items()
    for a, b in pairs
}


# ── Core functions ────────────────────────────────────────────────────────────

def build_pair_store(lcm):
    """
    For each relationship type, store individual (source_proj, target_proj, delta)
    for every training pair that can be looked up.
    """
    store = {}
    for rel, pairs in FUNCTIONAL_RELS.items():
        pair_deltas = []
        for a, b in pairs:
            try:
                pa, _  = lcm._get_proj(a)
                pb, _  = lcm._get_proj(b)
                pa = pa.astype(np.float64)
                pb = pb.astype(np.float64)
                d  = pb - pa
                pair_deltas.append({
                    'src_word': a,
                    'tgt_word': b,
                    'src_proj': pa,
                    'delta':    d,
                })
            except RuntimeError:
                pass
        store[rel] = pair_deltas
    return store


def global_delta(store, rel):
    """Plain mean of all pair deltas — Day 15 baseline."""
    deltas = [p['delta'] for p in store[rel]]
    return np.mean(deltas, axis=0) if deltas else None


def knn_delta(lcm, store, rel, source_word, k, temperature=1.0):
    """
    Locally-conditioned delta: weighted by cos(source, training_source).
    temperature > 1  → sharper weighting (nearest neighbour dominates)
    temperature = 1  → plain cosine weighting
    temperature → 0  → approaches global mean
    """
    pairs = store[rel]
    if not pairs:
        return None
    try:
        src_proj, _ = lcm._get_proj(source_word)
        src_proj = src_proj.astype(np.float64)
        src_norm = src_proj / (np.linalg.norm(src_proj) + 1e-20)
    except RuntimeError:
        return None

    # Cosine similarities to all training sources
    sims = []
    for p in pairs:
        pn = p['src_proj'] / (np.linalg.norm(p['src_proj']) + 1e-20)
        sims.append(float(np.dot(src_norm, pn)))

    sims = np.array(sims)

    # k-NN: keep only top-k, zero the rest
    if k < len(pairs):
        threshold = np.sort(sims)[-k]
        mask = sims >= threshold
        sims = np.where(mask, sims, -np.inf)

    # Softmax with temperature (operates on the kept sims)
    # Shift to avoid exp overflow: subtract max before exp
    valid = sims > -np.inf
    if not valid.any():
        return global_delta(store, rel)

    s_valid = sims[valid]
    s_scaled = s_valid * temperature
    s_shifted = s_scaled - s_scaled.max()
    weights_valid = np.exp(s_shifted)
    weights_valid /= weights_valid.sum()

    # Weighted sum of deltas
    weighted = np.zeros_like(pairs[0]['delta'])
    idx = 0
    for i, p in enumerate(pairs):
        if valid[i]:
            weighted += weights_valid[idx] * p['delta']
            idx += 1

    return weighted


def answer_query(lcm, source, delta, P, expected):
    try:
        src_proj, src_idx = lcm._get_proj(source)
    except RuntimeError:
        return None, None, None, "source not in vocab"
    src_proj = src_proj.astype(np.float64)
    derived  = src_proj + delta
    derived /= (np.linalg.norm(derived) + 1e-20)
    sims = P @ derived
    if src_idx is not None:
        sims[src_idx] = -9999
    top10_idx = np.argsort(sims)[-10:][::-1]
    results   = [(lcm.words[i], float(sims[i])) for i in top10_idx]
    words_lower = [w.lower() for w, _ in results]
    rank = words_lower.index(expected.lower()) + 1 if expected.lower() in words_lower else None
    conf = float(sims[top10_idx[0]])
    return results, rank, conf, None


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading LCM...")
    lcm   = build_lcm()
    P     = lcm.projections.astype(np.float64)

    print(f"\n{'='*65}")
    print(f"DAY 16 — k-NN Conditioned Delta Retrieval")
    print(f"{'='*65}")

    store = build_pair_store(lcm)
    print(f"\n  Training pairs loaded:")
    for rel, pairs in store.items():
        print(f"    {rel:<14}: {len(pairs)} pairs")

    # ── Section 1: Side-by-side accuracy comparison ───────────────────────────
    print(f"\n── Section 1: Global mean vs k-NN (k=1,3,5) accuracy ──────────")

    variants = {
        'global':  lambda src, rel: global_delta(store, rel),
        'k=1':     lambda src, rel: knn_delta(lcm, store, rel, src, k=1, temperature=5.0),
        'k=3':     lambda src, rel: knn_delta(lcm, store, rel, src, k=3, temperature=2.0),
        'k=5':     lambda src, rel: knn_delta(lcm, store, rel, src, k=5, temperature=1.5),
        'k=10':    lambda src, rel: knn_delta(lcm, store, rel, src, k=10, temperature=1.0),
    }

    # Track stats per variant
    stats = {v: {'correct': 0, 'total': 0, 'in_c': 0, 'in_t': 0, 'out_c': 0, 'out_t': 0}
             for v in variants}

    # Collect results for display
    rows = []
    for source, rel, expected in QUERIES:
        if rel not in store:
            continue
        in_train = (rel, source.lower(), expected.lower()) in TRAINING_PAIRS
        row = {'source': source, 'rel': rel, 'expected': expected,
               'in_train': in_train, 'results': {}}
        for vname, delta_fn in variants.items():
            d = delta_fn(source, rel)
            if d is None:
                row['results'][vname] = (None, None, '?')
                continue
            _, rank, conf, err = answer_query(lcm, source, d, P, expected)
            correct = (rank is not None and rank <= 5)
            row['results'][vname] = (rank, conf, '✓' if correct else '✗')
            stats[vname]['total'] += 1
            if correct:
                stats[vname]['correct'] += 1
            if in_train:
                stats[vname]['in_t'] += 1
                if correct:
                    stats[vname]['in_c'] += 1
            else:
                stats[vname]['out_t'] += 1
                if correct:
                    stats[vname]['out_c'] += 1
        rows.append(row)

    # Print header
    v_names = list(variants.keys())
    print(f"\n  {'IT':<2}  {'Type':<14}  {'Source':<12}  {'Expected':<12}  " +
          "  ".join(f"{v:<10}" for v in v_names))
    print("  " + "─" * (50 + 12 * len(v_names)))

    for row in rows:
        tag = 'IT' if row['in_train'] else 'OT'
        cols = []
        for vname in v_names:
            rank, conf, hit = row['results'].get(vname, (None, None, '?'))
            rank_str = f"r{rank}" if rank else "—"
            cols.append(f"{hit}{rank_str:<7}")
        print(f"  {tag}  {row['rel']:<14}  {row['source']:<12}  {row['expected']:<12}  " +
              "  ".join(cols))

    # ── Section 2: Accuracy summary ───────────────────────────────────────────
    print(f"\n── Section 2: Accuracy summary ─────────────────────────────────")
    print(f"\n  {'Variant':<10}  {'Overall':<14}  {'In-training':<16}  {'Out-of-training'}")
    print("  " + "─" * 60)
    for vname, s in stats.items():
        acc     = 100 * s['correct'] / max(s['total'], 1)
        in_acc  = 100 * s['in_c']    / max(s['in_t'],  1)
        out_acc = 100 * s['out_c']   / max(s['out_t'], 1)
        print(f"  {vname:<10}  {s['correct']}/{s['total']} ({acc:.1f}%)    "
              f"{s['in_c']}/{s['in_t']} ({in_acc:.1f}%)      "
              f"{s['out_c']}/{s['out_t']} ({out_acc:.1f}%)")

    # ── Section 3: Deep dive on failures ────────────────────────────────────
    print(f"\n── Section 3: Which k-NN neighbours are used for OOT queries? ──")
    oot_queries = [(s, r, e) for s, r, e in QUERIES
                   if (r, s.lower(), e.lower()) not in TRAINING_PAIRS]

    for source, rel, expected in oot_queries:
        pairs = store[rel]
        if not pairs:
            continue
        try:
            src_proj, _ = lcm._get_proj(source)
            src_proj = src_proj.astype(np.float64)
            src_norm = src_proj / (np.linalg.norm(src_proj) + 1e-20)
        except RuntimeError:
            continue

        sims = []
        for p in pairs:
            pn = p['src_proj'] / (np.linalg.norm(p['src_proj']) + 1e-20)
            sims.append((float(np.dot(src_norm, pn)), p['src_word'], p['tgt_word']))
        sims.sort(reverse=True)

        print(f"\n  {source} → {expected} ({rel})")
        print(f"    Nearest training sources:")
        for sim, sw, tw in sims[:4]:
            print(f"      cos={sim:+.4f}  {sw} → {tw}")

        # Show what each k gives
        for k, T in [(1, 5.0), (3, 2.0), (5, 1.5)]:
            d = knn_delta(lcm, store, rel, source, k=k, temperature=T)
            if d is None:
                continue
            results, rank, conf, _ = answer_query(lcm, source, d, P, expected)
            top3 = ", ".join(f"{w}({s:.3f})" for w, s in results[:3])
            hit  = "✓" if rank and rank <= 5 else "✗"
            print(f"    k={k}: {top3}  {hit} (expected rank={rank})")

    # ── Section 4: Lagrange points ────────────────────────────────────────────
    print(f"\n── Section 4: Lagrange points — Trojan asteroids in semantic space ──")
    print(f"  Between two 'massive' concepts, do concepts cluster at ±60°?\n")

    lagrange_pairs = [
        ('france',  'paris',   'capital'),
        ('king',    'queen',   'gender'),
        ('dog',     'animal',  'hypernym'),
        ('big',     'bigger',  'comparative'),
    ]

    for w_a, w_b, label in lagrange_pairs:
        try:
            pa, _ = lcm._get_proj(w_a)
            pb, _ = lcm._get_proj(w_b)
        except RuntimeError:
            print(f"  {w_a}/{w_b}: not in vocab")
            continue

        pa = pa.astype(np.float64)
        pb = pb.astype(np.float64)
        pa_n = pa / (np.linalg.norm(pa) + 1e-20)
        pb_n = pb / (np.linalg.norm(pb) + 1e-20)

        # L4/L5 direction: unit vector in plane of pa, pb at ±60° from midpoint
        # Midpoint direction
        mid    = pa_n + pb_n
        mid_n  = mid  / (np.linalg.norm(mid) + 1e-20)
        # Perpendicular in plane of pa, pb
        perp   = pb_n - np.dot(pb_n, pa_n) * pa_n
        perp_n = perp / (np.linalg.norm(perp) + 1e-20)

        # L4 = mid + sin(30°)*perp (equilateral: 60° from each vertex)
        # cos(60°) = 0.5 → sin(30°) = 0.5 relative scaling
        L4 = 0.866 * mid_n + 0.5 * perp_n   # 30° tilt from midpoint = 60° triangle
        L5 = 0.866 * mid_n - 0.5 * perp_n
        L4 /= (np.linalg.norm(L4) + 1e-20)
        L5 /= (np.linalg.norm(L5) + 1e-20)

        # Retrieve nearest concepts to L4 and L5
        sims_L4 = P @ L4
        sims_L5 = P @ L5
        # Exclude a and b themselves
        for word in (w_a, w_b):
            try:
                _, idx = lcm._get_proj(word)
                if idx is not None:
                    sims_L4[idx] = -9999
                    sims_L5[idx] = -9999
            except RuntimeError:
                pass

        top5_L4 = [(lcm.words[i], float(sims_L4[i]))
                   for i in np.argsort(sims_L4)[-5:][::-1]]
        top5_L5 = [(lcm.words[i], float(sims_L5[i]))
                   for i in np.argsort(sims_L5)[-5:][::-1]]

        cos_ab = float(np.dot(pa_n, pb_n))
        print(f"  {w_a} ↔ {w_b}  ({label}, cos={cos_ab:.3f})")
        L4_str = ", ".join(f"{w}({s:.3f})" for w, s in top5_L4)
        L5_str = ", ".join(f"{w}({s:.3f})" for w, s in top5_L5)
        print(f"    L4: {L4_str}")
        print(f"    L5: {L5_str}")
        print()

    # ── Section 5: Temperature sensitivity ───────────────────────────────────
    print(f"\n── Section 5: Temperature sensitivity for 'australia → capital' ──")
    for T in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        d = knn_delta(lcm, store, 'capital', 'australia', k=3, temperature=T)
        if d is None:
            continue
        results, rank, conf, _ = answer_query(lcm, 'australia', d, P, 'canberra')
        top3 = ", ".join(f"{w}({s:.3f})" for w, s in results[:3])
        hit  = "✓" if rank and rank <= 5 else "✗"
        print(f"    T={T:<6}  {top3}  {hit}")

    print(f"\n── Section 6: Same for 'swim → past tense' ─────────────────────")
    for T in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        d = knn_delta(lcm, store, 'past', 'swim', k=3, temperature=T)
        if d is None:
            continue
        results, rank, conf, _ = answer_query(lcm, 'swim', d, P, 'swam')
        top3 = ", ".join(f"{w}({s:.3f})" for w, s in results[:3])
        hit  = "✓" if rank and rank <= 5 else "✗"
        print(f"    T={T:<6}  {top3}  {hit}")
