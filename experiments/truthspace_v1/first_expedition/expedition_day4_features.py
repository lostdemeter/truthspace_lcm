#!/usr/bin/env python3
"""
Expedition Day 4 — Gravitational Features vs Transformer Architecture

Three tests:

TEST A: M_binding as retrieval confidence
  Hypothesis: high-binding words are retrieved correctly at higher rank than
  low-binding words. If so, M_binding is a RETRIEVAL CONFIDENCE SCORE
  that LLMs could expose but don't.
  Method: for a set of analogical test pairs (a:b::c:d), measure whether
  the rank of d when applying the delta to c correlates with M_binding(c).

TEST B: Delta-axis alignment
  Hypothesis: the 9 functional delta vectors align with specific IRD axes.
  If the IRD construction already discovered the Killing vectors as axes,
  the deltas should have large projections onto a small number of axes
  (sparse in the IRD basis).
  Method: project each functional delta onto all 500 IRD axes, measure
  sparsity (Gini coefficient) and find top axes.

TEST C: Gravitational feature inventory
  Compute and tabulate which gravitational features are:
    - Present in current LLMs (implicit)
    - Present in our IRD system (explicit)
    - Missing from both (novel)
  Using the data from Days 1-3 to ground each claim.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

MASS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'expedition_day1_masses.npz')

# Functional relationship types from Day 3 (precision-based, LOO rank ≤ 5)
FUNCTIONAL_RELS = {
    'gender_noun': [
        ('king','queen'),('man','woman'),('boy','girl'),('actor','actress'),
        ('prince','princess'),('son','daughter'),('father','mother'),
        ('brother','sister'),('husband','wife'),('uncle','aunt'),
    ],
    'country_capital': [
        ('france','paris'),('germany','berlin'),('italy','rome'),
        ('spain','madrid'),('japan','tokyo'),('china','beijing'),
        ('russia','moscow'),('brazil','brasilia'),
    ],
    'singular_plural': [
        ('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
        ('book','books'),('tree','trees'),('bird','birds'),
        ('child','children'),('mouse','mice'),('foot','feet'),
    ],
    'present_past': [
        ('run','ran'),('walk','walked'),('eat','ate'),('write','wrote'),
        ('speak','spoke'),('take','took'),('go','went'),('see','saw'),
        ('give','gave'),('make','made'),
    ],
    'adjective_comparative': [
        ('big','bigger'),('small','smaller'),('fast','faster'),
        ('old','older'),('young','younger'),('strong','stronger'),
        ('long','longer'),('short','shorter'),('hot','hotter'),
        ('cold','colder'),
    ],
    'hypernym_entity': [
        ('dog','animal'),('car','vehicle'),('apple','fruit'),
        ('oak','tree'),('rose','flower'),('salmon','fish'),
        ('eagle','bird'),('violin','instrument'),
    ],
    'language_to_country': [
        ('english','england'),('french','france'),('german','germany'),
        ('spanish','spain'),('italian','italy'),('japanese','japan'),
        ('russian','russia'),('chinese','china'),
    ],
    'verb_noun_agent': [
        ('teach','teacher'),('build','builder'),('write','writer'),
        ('paint','painter'),('play','player'),('drive','driver'),
        ('bake','baker'),('manage','manager'),
    ],
    'antonym_temperature': [
        ('hot','cold'),('warm','cool'),('boiling','freezing'),
        ('heat','cold'),
    ],
}


def gini(v):
    """Gini coefficient of abs values — 0=uniform, 1=completely sparse."""
    a = np.sort(np.abs(v))
    n = len(a)
    idx = np.arange(1, n + 1)
    return float(1 - 2 * (a * (n - idx + 1)).sum() / (n * a.sum() + 1e-20))


if __name__ == '__main__':
    print("Loading LCM...")
    lcm  = build_lcm()
    P    = lcm.projections.astype(np.float64)
    n, d = P.shape
    words = lcm.words
    A    = lcm.axis_vectors.astype(np.float64)   # (n_axes, embed_dim) = (500, 3584)

    day1 = np.load(MASS_PATH, allow_pickle=True)
    M_binding = day1['M_binding'].astype(np.float64)

    print(f"\n{'='*65}")
    print("DAY 4 — Gravitational Features vs Transformer Architecture")
    print(f"{'='*65}")

    # ── TEST A: M_binding as retrieval confidence ─────────────────────────────
    print(f"\n── TEST A: M_binding as retrieval confidence ───────────────")
    print(f"  Hypothesis: higher M_binding(source) → better retrieval rank")
    print(f"  Testing over all functional relationships, LOO style\n")

    # Collect (source_word, target_word, rank, M_binding_source) across all rels
    all_sources  = []
    all_m_bind   = []
    all_loo_ranks = []

    for rel_name, pairs in FUNCTIONAL_RELS.items():
        valid = []
        for a, b in pairs:
            try:
                pa, _ = lcm._get_proj(a)
                pb, _ = lcm._get_proj(b)
                valid.append((a, b, pa.astype(np.float64), pb.astype(np.float64)))
            except RuntimeError:
                pass
        if len(valid) < 2:
            continue

        deltas = [pb - pa for _, _, pa, pb in valid]

        for held_i, (a, b, pa, pb) in enumerate(valid):
            train_d = [deltas[j] for j in range(len(valid)) if j != held_i]
            delta_loo = np.mean(train_d, axis=0)
            predicted = pa + delta_loo
            predicted /= (np.linalg.norm(predicted) + 1e-20)
            cos_to_b = float(np.dot(predicted, pb / (np.linalg.norm(pb) + 1e-20)))
            rank = int((P @ predicted > cos_to_b).sum()) + 1

            a_lower = a.lower()
            if a_lower in lcm.word_set:
                i = lcm.word_set[a_lower]
                all_m_bind.append(float(M_binding[i]))
                all_loo_ranks.append(rank)
                all_sources.append(a)

    all_m = np.array(all_m_bind)
    all_r = np.array(all_loo_ranks, dtype=float)

    # Correlation
    corr = float(np.corrcoef(all_m, all_r)[0, 1])
    # Log-rank correlation (ranks are right-skewed)
    log_corr = float(np.corrcoef(all_m, np.log1p(all_r))[0, 1])

    print(f"  n pairs analysed: {len(all_m)}")
    print(f"  corr(M_binding, rank):          {corr:+.4f}")
    print(f"  corr(M_binding, log(rank)):     {log_corr:+.4f}")

    # Quartile analysis
    q = np.percentile(all_m, [25, 50, 75])
    labels = ['Q1 (lowest M)', 'Q2', 'Q3', 'Q4 (highest M)']
    masks  = [
        all_m <= q[0],
        (all_m > q[0]) & (all_m <= q[1]),
        (all_m > q[1]) & (all_m <= q[2]),
        all_m > q[2],
    ]
    print(f"\n  Retrieval rank by M_binding quartile:")
    print(f"  {'Quartile':<20s}  {'n':<6s}  {'median rank':<14s}  mean rank  rank≤5")
    for lab, mask in zip(labels, masks):
        if mask.sum() == 0:
            continue
        r_q = all_r[mask]
        print(f"  {lab:<20s}  {mask.sum():<6d}  {np.median(r_q):<14.1f}  "
              f"{r_q.mean():<10.1f}  {(r_q<=5).mean()*100:.0f}%")

    if log_corr < -0.1:
        print(f"\n  VERDICT: M_binding IS a retrieval confidence signal "
              f"(higher binding → lower rank)")
    elif abs(log_corr) < 0.05:
        print(f"\n  VERDICT: M_binding does NOT predict retrieval rank")
    else:
        print(f"\n  VERDICT: weak signal (log_corr={log_corr:.3f})")

    # ── TEST B: Delta-axis alignment ──────────────────────────────────────────
    print(f"\n── TEST B: Functional deltas in the IRD axis basis ─────────")
    print(f"  Hypothesis: functional deltas are sparse in the 500-axis IRD basis")
    print(f"  (if IRD axes ARE the Killing vectors, deltas should align with axes)\n")

    # Compute mean delta vector for each functional relationship
    # Then project into the IRD axis space: delta_in_ird = delta_raw @ A.T
    # But wait: the IRD axes A are in embedding space (3584-dim).
    # The projections P = concept_embs @ A.T.
    # The deltas we computed are in PROJECTION space (500-dim), not embedding space.
    # So the question becomes: is the delta vector in 500-dim space sparse?

    print(f"  {'Relationship':<28s}  {'Gini':<8s}  {'top-3 axes':<30s}  "
          f"top-axis energy")
    delta_vectors = {}
    for rel_name, pairs in FUNCTIONAL_RELS.items():
        valid = []
        for a, b in pairs:
            try:
                pa, _ = lcm._get_proj(a)
                pb, _ = lcm._get_proj(b)
                valid.append((pb.astype(np.float64) - pa.astype(np.float64)))
            except RuntimeError:
                pass
        if not valid:
            continue
        delta = np.mean(valid, axis=0)
        delta_norm = delta / (np.linalg.norm(delta) + 1e-20)
        delta_vectors[rel_name] = delta_norm

        # Sparsity in the 500-dim projection space
        g = gini(delta_norm)
        # Top axes by abs value
        top3 = np.argsort(np.abs(delta_norm))[-3:][::-1]
        top3_vals = delta_norm[top3]
        top3_energy = (delta_norm[top3] ** 2).sum() / (delta_norm ** 2).sum()
        top3_str = "  ".join([f"ax{a}({v:+.3f})" for a, v in zip(top3, top3_vals)])

        print(f"  {rel_name:<28s}  {g:.4f}    {top3_str:<30s}  {top3_energy:.3f}")

    # Compare sparsity of functional deltas vs random vectors
    rng = np.random.default_rng(42)
    random_ginis = [gini(v) for v in rng.standard_normal((100, d))]
    print(f"\n  Random vector Gini (mean ± std): "
          f"{np.mean(random_ginis):.4f} ± {np.std(random_ginis):.4f}")
    delta_ginis = [gini(v) for v in delta_vectors.values()]
    print(f"  Functional delta Gini (mean ± std): "
          f"{np.mean(delta_ginis):.4f} ± {np.std(delta_ginis):.4f}")
    if np.mean(delta_ginis) > np.mean(random_ginis) + 2 * np.std(random_ginis):
        print(f"  VERDICT: Deltas ARE sparser than random → aligned with IRD axes ✓")
    else:
        print(f"  VERDICT: Deltas NOT sparser than random → IRD axes ≠ Killing vectors")

    # Check: are any two functional deltas aligned with the same axis?
    print(f"\n  ── Top-axis collision check ──────────────────────────────")
    print(f"  (if two deltas share a top axis, they may overlap in concept space)")
    top1_axes = {}
    for rel_name, d_v in delta_vectors.items():
        top1 = int(np.argmax(np.abs(d_v)))
        top1_axes.setdefault(top1, []).append(rel_name)
    for ax, rels in sorted(top1_axes.items()):
        if len(rels) > 1:
            print(f"  Axis {ax}: shared by {rels}")
        else:
            print(f"  Axis {ax}: {rels[0]}")

    # ── TEST C: Gravitational feature inventory ───────────────────────────────
    print(f"\n── TEST C: Gravitational feature inventory ─────────────────")
    print(f"  Based on Days 1-3 experimental data\n")

    features = [
        # (Feature name, In LLMs, In IRD, Novel, Evidence)
        ("Semantic mass (M_binding)",
         "Implicit (never measured)",
         "Explicit (Day 1)",
         "Can be precomputed for all concepts",
         f"Numbers/countries bind at 0.31, polysemous words at 0.14"),

        ("Gravitational black hole (cross-domain words)",
         "Implicit (high attention weight)",
         "Measurable (M_global Day 1)",
         "Named and filterable",
         "metabol/coherence/crane: M_global=0.051"),

        ("Semantic vacuum (function words near origin)",
         "Implicit (low semantic weight)",
         "Measurable (M_global Day 1, 'the'=0.030)",
         "Can be excluded from context gravity",
         "the/and: M_global=0.029-0.030, rank 12-13k/25k"),

        ("Polysemy as low binding mass",
         "Implicit (model 'knows' some words are hard)",
         "Quantified (Day 1, cookie=rank 25607/25674)",
         "Retrieval confidence score: M_binding → P(correct)",
         "cookie=0.137, berlin=0.228, bake=0.185"),

        ("Escape velocity (min context to disambiguate)",
         "Not available",
         "Derivable from basin boundaries",
         "Context sufficiency score for any query",
         "Needs Day 5 experiment"),

        ("Functional deltas as relationship store",
         "Implicit in attention heads",
         "Explicit (9 deltas, Day 3)",
         "Direct 4x compression of morphological vocabulary",
         "9 types, all LOO rank≤6, mostly orthogonal"),

        ("Relationship orthogonality",
         "Implicit (heads don't interfere)",
         "Measured (Day 3, most |cos|<0.1)",
         "Can assign one head per relationship type",
         "gender↔tense: cos=+0.001 (near-perfect orthogonal)"),

        ("Killing vectors vs local transformations",
         "Both learned identically",
         "Classified (Day 3)",
         "Model could apply Killing vectors directly, skip local ones",
         "9 functional / 5 local / 6 partial out of 20 types"),

        ("Holographic spectrum (distributed info)",
         "Implicit (model uses all layers)",
         "Measured (Day 1, SVD flat)",
         "Explains why layer-by-layer attention is needed",
         "383 axes for 90% energy, vs 50 axes for images"),

        ("Delta-axis sparsity",
         "Not known",
         "Measured (Day 4, TEST B)",
         "If sparse, specific axes encode specific relationships",
         "Results above"),

        ("M_binding as retrieval confidence",
         "Not available (no confidence output)",
         "Computable O(k) per word",
         "Native confidence signal for any retrieval",
         "Results above (TEST A)"),

        ("N-body sentence centroid",
         "Not available (no explicit sentence gravity)",
         "Implemented (DC 305 Q3)",
         "Single-pass sentence embedding without encoder",
         "Centroid separates contexts well (0.39 separation)"),
    ]

    print(f"  {'Feature':<35s}  {'LLM has?':<22s}  {'IRD has?':<22s}  Novel value")
    print("  " + "─" * 110)
    for feat, llm, ird, novel, evidence in features:
        print(f"  {feat:<35s}  {llm:<22s}  {ird:<22s}  {novel}")
        print(f"    Evidence: {evidence}")
        print()
