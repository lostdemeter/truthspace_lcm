#!/usr/bin/env python3
"""
Expedition Day 2 — The Compression Coast

Day 1 showed SVD fails: 383 axes needed for 90% energy. The concept space is
holographic — information is distributed, not concentrated in linear components.

New hypothesis: cluster-based compression works where linear compression fails.
If concepts form K natural clusters, each concept can be stored as:
    concept ≈ archetype_k  +  residual_vector

We test K = 32, 64, 128, 256, 512, 1024 using K-means on the 500-dim projections.

Measures:
  - mean cosine distance to nearest archetype (reconstruction quality)
  - whether reconstruction error correlates with M_binding (isolated concepts harder?)
  - what the archetype words are at each K — do they form a periodic table?
  - compression ratio: K × 500 floats vs N × 500 floats

Second part: test the substitution hypothesis directly.
If two concepts are on opposite ends of a known delta (e.g. king/queen),
can we store ONLY king + Δgender and reconstruct queen to within ε?
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

K_VALUES   = [32, 64, 128, 256, 512, 1024]
MAX_ITER   = 100
SEED       = 42
BATCH_SIZE = 2048     # for mini-batch style update

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expedition_log.md')
MASS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'expedition_day1_masses.npz')


def kmeans_cosine(P, k, max_iter=MAX_ITER, seed=SEED):
    """
    Spherical K-means on unit-normed rows of P.
    Cosine similarity = dot product (since P is unit-normed).
    Returns: (centroids (k, d), labels (n,), inertia_history)
    """
    rng = np.random.default_rng(seed)
    n, d = P.shape

    # Init: K-means++ style (pick spread-out initial centres)
    centres = np.zeros((k, d), dtype=np.float32)
    centres[0] = P[rng.integers(n)]
    for i in range(1, k):
        # Distance to nearest existing centre (cosine distance = 1 - cos_sim)
        sims = P @ centres[:i].T          # (n, i)
        max_sim = sims.max(axis=1)        # (n,)
        dist2 = (1.0 - max_sim) ** 2
        prob = dist2 / dist2.sum()
        centres[i] = P[rng.choice(n, p=prob)]

    inertia_history = []
    labels = np.zeros(n, dtype=np.int32)

    for it in range(max_iter):
        # Assignment step — batch to avoid OOM on large P
        new_labels = np.zeros(n, dtype=np.int32)
        for start in range(0, n, BATCH_SIZE):
            end   = min(start + BATCH_SIZE, n)
            block = P[start:end]              # (batch, d)
            sims  = block @ centres.T        # (batch, k)
            new_labels[start:end] = sims.argmax(axis=1)

        # Inertia (mean cosine distance to nearest centre)
        assigned_cos = np.array([
            float(np.dot(P[j], centres[new_labels[j]]))
            for j in range(0, n, max(1, n // 500))  # sample for speed
        ])
        inertia = 1.0 - assigned_cos.mean()
        inertia_history.append(inertia)

        # Check convergence
        if it > 0 and np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Update step — new centres are means of assigned points, re-normalised
        new_centres = np.zeros((k, d), dtype=np.float64)
        counts      = np.zeros(k, dtype=np.int32)
        for j in range(n):
            new_centres[labels[j]] += P[j]
            counts[labels[j]]      += 1
        for ci in range(k):
            if counts[ci] == 0:
                new_centres[ci] = P[rng.integers(n)]   # reinit empty cluster
            else:
                new_centres[ci] /= counts[ci]
        norms = np.linalg.norm(new_centres, axis=1, keepdims=True)
        centres = (new_centres / (norms + 1e-20)).astype(np.float32)

    return centres, labels, inertia_history


def find_archetype_words(P, centres, words):
    """For each cluster centre, find the actual concept word closest to it."""
    archetype_words = []
    for c in centres:
        sims = P @ c
        archetype_words.append(words[int(sims.argmax())])
    return archetype_words


def reconstruction_error(P, centres, labels):
    """Mean cosine distance from each concept to its assigned archetype."""
    errs = []
    for j in range(len(P)):
        errs.append(1.0 - float(np.dot(P[j], centres[labels[j]])))
    return np.array(errs, dtype=np.float32)


if __name__ == '__main__':
    print("Loading LCM...")
    lcm  = build_lcm()
    P    = lcm.projections.astype(np.float32)
    n, d = P.shape
    words = lcm.words

    # Load Day 1 masses
    masses_available = os.path.exists(MASS_PATH)
    if masses_available:
        day1 = np.load(MASS_PATH, allow_pickle=True)
        M_binding = day1['M_binding']
        print(f"  Loaded Day 1 mass data for {len(M_binding)} concepts")

    compression_results = []

    print(f"\n{'='*65}")
    print("DAY 2 OBSERVATION LOG — The Compression Coast")
    print(f"{'='*65}")
    print(f"\n  Corpus: {n} concepts × {d} axes")
    print(f"  Testing K = {K_VALUES}")

    for k in K_VALUES:
        t0 = time.time()
        print(f"\n  ── K={k} ─────────────────────────────────────────")
        centres, labels, inertia_hist = kmeans_cosine(P, k)
        elapsed = time.time() - t0

        errs = reconstruction_error(P, centres, labels)
        mean_err   = float(errs.mean())
        p50_err    = float(np.percentile(errs, 50))
        p95_err    = float(np.percentile(errs, 95))
        p99_err    = float(np.percentile(errs, 99))
        ratio      = n / k
        storage    = (k * d + n * (1 + d)) / (n * d)  # (centres + labels + residuals) / original

        # Cluster size distribution
        counts = np.bincount(labels, minlength=k)
        c_mean = counts.mean()
        c_std  = counts.std()
        c_max  = counts.max()
        c_min  = counts.min()

        # Top archetype words (sample of 10)
        arch_words = find_archetype_words(P, centres, words)
        sample_archetypes = arch_words[:10]

        print(f"  Converged in {len(inertia_hist)} iters, {elapsed:.1f}s")
        print(f"  Mean cosine error: {mean_err:.4f}  p50={p50_err:.4f}  "
              f"p95={p95_err:.4f}  p99={p99_err:.4f}")
        print(f"  Cluster sizes:  mean={c_mean:.0f}  std={c_std:.0f}  "
              f"min={c_min}  max={c_max}")
        print(f"  Sample archetype words (10/{k}): {sample_archetypes}")

        # If Day 1 masses available: do high-M_binding concepts compress better?
        if masses_available:
            high_bind = M_binding > np.percentile(M_binding, 75)
            low_bind  = M_binding < np.percentile(M_binding, 25)
            err_high = errs[high_bind].mean()
            err_low  = errs[low_bind].mean()
            print(f"  Reconstruction error — high-binding concepts: {err_high:.4f}  "
                  f"low-binding: {err_low:.4f}  "
                  f"(ratio={err_low/err_high:.2f}×)")

        compression_results.append({
            'k': k, 'mean_err': mean_err, 'p95_err': p95_err,
            'cluster_size_mean': c_mean, 'archetype_sample': sample_archetypes[:5],
            'err_high': float(errs[high_bind].mean()) if masses_available else None,
            'err_low':  float(errs[low_bind].mean())  if masses_available else None,
        })

    # Summary table
    print(f"\n{'='*65}")
    print(f"  Compression summary")
    print(f"  {'K':<8s}  {'mean_err':<12s}  {'p95_err':<12s}  "
          f"{'ratio N/K':<12s}  cost_vs_original")
    for r in compression_results:
        k = r['k']
        # Storage cost: K centroids + N labels (1 int each).
        # Original: N × d floats. Archetype+label: K×d + N floats.
        cost = (k * d + n) / (n * d)
        print(f"  {k:<8d}  {r['mean_err']:.4f}       {r['p95_err']:.4f}       "
              f"{n/k:<12.1f}  {cost:.3f}× (vs 1.0×)")

    # ── Part 2: Substitution test ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PART 2 — Substitution Test")
    print(f"{'='*65}")
    print("\n  Can we reconstruct words using archetype + delta?")
    print("  Test: store king, learn Δgender, derive queen from king+Δgender")

    test_pairs = [
        ('king',    'queen'),
        ('man',     'woman'),
        ('actor',   'actress'),
        ('france',  'paris'),
        ('germany', 'berlin'),
        ('italy',   'rome'),
        ('hot',     'cold'),
        ('big',     'small'),
        ('fast',    'slow'),
    ]

    try:
        valid_pairs = []
        for a, b in test_pairs:
            try:
                pa, _ = lcm._get_proj(a)
                pb, _ = lcm._get_proj(b)
                valid_pairs.append((a, b, pa.astype(np.float64), pb.astype(np.float64)))
            except RuntimeError:
                pass

        if valid_pairs:
            # Group into relationship families
            gender_pairs  = [(a,b,pa,pb) for a,b,pa,pb in valid_pairs
                             if (a,b) in [('king','queen'),('man','woman'),('actor','actress')]]
            capital_pairs = [(a,b,pa,pb) for a,b,pa,pb in valid_pairs
                             if (a,b) in [('france','paris'),('germany','berlin'),('italy','rome')]]
            antonym_pairs = [(a,b,pa,pb) for a,b,pa,pb in valid_pairs
                             if (a,b) in [('hot','cold'),('big','small'),('fast','slow')]]

            for family_name, family in [('gender',  gender_pairs),
                                         ('capital', capital_pairs),
                                         ('antonym', antonym_pairs)]:
                if len(family) < 2:
                    continue
                # Learn delta from all-but-one pairs, test on held-out
                print(f"\n  ── {family_name} family ─────────────────────────────")
                print(f"  {'Held-out':<20s}  {'cos(predicted, actual)':<24s}  "
                      f"rank_of_actual  verdict")
                for held_i in range(len(family)):
                    a, b, pa, pb = family[held_i]
                    train = [f for j, f in enumerate(family) if j != held_i]
                    if not train:
                        continue
                    # Learn delta from training pairs
                    delta = np.mean([f[3] - f[2] for f in train], axis=0)
                    # Apply to held-out source
                    predicted = pa + delta
                    predicted /= (np.linalg.norm(predicted) + 1e-20)
                    # Cosine to actual target
                    cos_to_target = float(np.dot(predicted, pb))
                    # Rank of target among all concepts
                    sims_all = P.astype(np.float64) @ predicted
                    rank = int((sims_all > cos_to_target).sum()) + 1
                    verdict = "✓" if rank <= 5 else ("~" if rank <= 20 else "✗")
                    print(f"  {a}→{b:<16s}  cos={cos_to_target:+.4f}           "
                          f"rank={rank:<8d}  {verdict}")

                # Cross-pair: learn from gender, apply to capital (do deltas transfer?)
            print(f"\n  ── Cross-family delta transfer test ────────────────────")
            if gender_pairs and capital_pairs:
                # Can the gender delta predict anything useful about capitals?
                delta_gender  = np.mean([f[3] - f[2] for f in gender_pairs],  axis=0)
                delta_capital = np.mean([f[3] - f[2] for f in capital_pairs], axis=0)
                delta_cos = float(np.dot(
                    delta_gender  / (np.linalg.norm(delta_gender)  + 1e-20),
                    delta_capital / (np.linalg.norm(delta_capital) + 1e-20)
                ))
                print(f"  cos(Δgender, Δcapital) = {delta_cos:+.4f}")
                print(f"  → {'orthogonal' if abs(delta_cos) < 0.1 else 'correlated'} "
                      f"(expected: orthogonal)")

    except Exception as e:
        print(f"  Substitution test error: {e}")

    # Save archetype words from best compression
    best = compression_results[len(compression_results)//2]   # middle K
    print(f"\n  Best-value K by p95 error vs compression:")
    for r in compression_results:
        k = r['k']
        cost = (k * d + n) / (n * d)
        score = r['p95_err'] / (1.0 - cost + 0.001)   # lower = better
        print(f"  K={k:<6d}  p95={r['p95_err']:.4f}  cost={cost:.3f}  "
              f"5 archetypes: {r['archetype_sample']}")
