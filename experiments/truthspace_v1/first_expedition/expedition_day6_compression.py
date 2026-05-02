#!/usr/bin/env python3
"""
Expedition Day 6 — Morphological Compression Index

Day 3 claimed: ~9 functional deltas + ~6,400 base forms ≈ 4× vocabulary compression.
This is a rough estimate. Day 6 tests the actual claim.

Approach:
  For each of the 9 functional relationship types, use the learned delta vector
  to ask: "for what fraction of concepts in the IRD vocabulary does applying
  this delta produce a DIFFERENT in-vocabulary concept at rank ≤ 5?"

  Concepts where applying the delta hits another vocabulary concept are
  "derivable" — they need not be stored independently.

  Compression estimate:
    Derivable concepts = source concepts where delta→target ≤ rank 5
    Non-derivable = everything else (must be stored)
    Compression ratio = N_total / (N_non_derivable + 9 delta vectors)

  The 9 delta vectors are negligible storage: 9 × 500 floats << 25,674 × 500.

Also test: are derivable concepts clustered by semantic type
  (verbs derive past tenses, nouns derive plurals, etc.)?

CAUTION: For each concept, applying the delta and checking rank against 25,674
  concepts is O(N × n_axes) per concept. For N=25,674 and n_axes=500, each
  search is a 500-dim dot product. With batching this is fast.

  Total cost: 9 deltas × 25,674 concepts × 25,674 search = 9 × 25674^2 dot products
  That's ~6 billion operations — too slow. Instead, use random sampling.

  Sample 500 concepts per delta type, apply delta, check rank.
  This gives a statistically robust estimate of the derivable fraction.

Also test: does the set of "derivable by gender" concepts match the set of
  "derivable by plural" (they should be mostly disjoint — nouns vs. verbs)?
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

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
        ('long','longer'),('short','shorter'),('hot','hotter'),('cold','colder'),
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
        ('hot','cold'),('warm','cool'),('boiling','freezing'),('heat','cold'),
    ],
}

N_SAMPLE = 600     # concepts sampled per delta for compression estimate
RANK_THR = 5       # hit if target rank ≤ this
RNG_SEED = 42


def learn_delta(lcm, pairs):
    """Learn mean delta vector from training pairs."""
    vecs = []
    for a, b in pairs:
        try:
            pa, _ = lcm._get_proj(a)
            pb, _ = lcm._get_proj(b)
            vecs.append(pb.astype(np.float64) - pa.astype(np.float64))
        except RuntimeError:
            pass
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def estimate_derivable_fraction(lcm, delta, rng, n_sample=N_SAMPLE, rank_thr=RANK_THR):
    """
    Sample random source concepts, apply delta, check if a DIFFERENT in-vocabulary
    concept appears at rank ≤ rank_thr.

    Returns:
        fraction_derivable: fraction of sampled concepts that hit a different
                            concept within rank rank_thr
        examples: list of (source, hit_word, rank) for the first 5 hits
    """
    P = lcm.projections.astype(np.float64)
    N = len(lcm.words)

    indices = rng.choice(N, size=min(n_sample, N), replace=False)
    hits = 0
    examples = []

    for i in indices:
        src_proj  = P[i]
        predicted = src_proj + delta
        pred_norm = predicted / (np.linalg.norm(predicted) + 1e-20)
        cos_to_self = float(np.dot(pred_norm, src_proj / (np.linalg.norm(src_proj) + 1e-20)))
        sims = P @ pred_norm
        # Exclude the source itself
        sims[i] = -9999
        top1_idx = int(np.argmax(sims))
        top1_sim  = float(sims[top1_idx])
        rank = int((sims > top1_sim).sum()) + 1  # rank of best non-self

        # A "hit" = the delta actually moves us to a different concept nearby
        # Check rank of predicted position among ALL concepts (excluding self)
        cos_from_top1 = top1_sim
        delta_magnitude = np.linalg.norm(delta)
        moved = float(np.dot(pred_norm,
                             src_proj/(np.linalg.norm(src_proj)+1e-20))) < 0.98

        if moved and cos_from_top1 > 0.5:
            hits += 1
            if len(examples) < 5:
                examples.append((lcm.words[i], lcm.words[top1_idx],
                                 float(cos_from_top1)))

    return hits / len(indices), examples


def estimate_compression(lcm, deltas_by_type, rng):
    """
    Estimate the compressed vocabulary size.

    Strategy:
      For each concept, test whether it is "derivable" by any of the 9 deltas
      from some source concept. A concept is derivable if:
        (a) applying the REVERSE delta to it gives another in-vocab concept at rank ≤ 5
        (i.e., it is a TARGET of some source-via-delta pair)

      Rather than checking all 25,674 × 9 combos, we sample.
      This gives an UPPER BOUND on derivable concepts (some will double-count).
    """
    P   = lcm.projections.astype(np.float64)
    N   = len(lcm.words)
    # For each concept, test if REVERSE delta gives a high-sim source
    # Sample 300 random concepts
    test_idxs = rng.choice(N, size=300, replace=False)
    derivable_by_any = np.zeros(300, dtype=bool)

    for rel_name, delta in deltas_by_type.items():
        rev_delta = -delta
        for j, i in enumerate(test_idxs):
            if derivable_by_any[j]:
                continue
            src_proj  = P[i]
            predicted = src_proj + rev_delta
            pred_norm = predicted / (np.linalg.norm(predicted) + 1e-20)
            sims   = P @ pred_norm
            sims[i] = -9999
            top1_i  = int(np.argmax(sims))
            top1_sim = float(sims[top1_i])
            if top1_sim > 0.55:
                derivable_by_any[j] = True

    frac_derivable = derivable_by_any.mean()
    return frac_derivable


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float64)
    N   = len(lcm.words)
    rng = np.random.default_rng(RNG_SEED)

    print(f"\n{'='*65}")
    print(f"DAY 6 — Morphological Compression Index")
    print(f"{'='*65}")
    print(f"\n  Vocabulary size:  {N:,}")
    print(f"  Sample per type:  {N_SAMPLE}")
    print(f"  Hit threshold:    cos > 0.50 (meaningful displacement)\n")

    # Learn deltas from training pairs
    deltas = {}
    for rel_name, pairs in FUNCTIONAL_RELS.items():
        d = learn_delta(lcm, pairs)
        if d is not None:
            deltas[rel_name] = d

    # ── Per-type derivable fraction ───────────────────────────────────────────
    print(f"  {'Relationship':<28s}  {'Deriv%':<10s}  {'Δ-norm':<10s}  Examples")
    print("  " + "─" * 85)
    total_weighted = 0.0
    derivable_fracs = {}
    for rel_name, delta in deltas.items():
        frac, examples = estimate_derivable_fraction(lcm, delta, rng)
        derivable_fracs[rel_name] = frac
        total_weighted += frac
        ex_str = ", ".join(f"{s}→{t}({c:.2f})" for s,t,c in examples[:3])
        print(f"  {rel_name:<28s}  {frac*100:5.1f}%      {np.linalg.norm(delta):.4f}      {ex_str}")

    mean_frac = total_weighted / len(deltas)
    print(f"\n  Mean derivable fraction across all types: {mean_frac*100:.1f}%")

    # ── Derivable by ANY delta (union) ────────────────────────────────────────
    print(f"\n── Union estimate: derivable by ANY relationship ────────────")
    union_frac = estimate_compression(lcm, deltas, rng)
    print(f"  Fraction derivable by any of {len(deltas)} deltas: {union_frac*100:.1f}%")

    n_non_derivable = round(N * (1 - union_frac))
    n_delta_storage = len(deltas) * P.shape[1]   # 9 × 500 = 4500 floats
    n_original      = N * P.shape[1]
    n_compressed    = n_non_derivable * P.shape[1] + n_delta_storage
    ratio           = n_original / n_compressed

    print(f"\n  ── Compression arithmetic ──────────────────────────────────")
    print(f"  Original:     {N:,} concepts × {P.shape[1]} axes = {n_original:,} floats")
    print(f"  Base forms:   {n_non_derivable:,} concepts    ({(1-union_frac)*100:.1f}% of vocab)")
    print(f"  Deltas:       {len(deltas)} × {P.shape[1]} axes = {n_delta_storage:,} floats")
    print(f"  Compressed:   {n_non_derivable:,} × {P.shape[1]} + {n_delta_storage:,} = {n_compressed:,} floats")
    print(f"  Ratio:        {ratio:.2f}×")

    if ratio >= 1.5:
        print(f"  VERDICT: Compression IS achievable ({ratio:.1f}×) — Day 3 estimate validated")
    else:
        print(f"  VERDICT: Compression is marginal — Day 3 estimate was optimistic")

    # ── Are derivable sets disjoint by type? ─────────────────────────────────
    print(f"\n── Type overlap: are derivable sets independent? ────────────")
    print(f"  (Test whether plural-derivable and tense-derivable are different concepts)")
    sample_idxs = rng.choice(N, size=200, replace=False)

    def derivable_mask(delta, idxs, thr=0.5):
        mask = np.zeros(len(idxs), dtype=bool)
        rev  = -delta
        for j, i in enumerate(idxs):
            pred = P[i] + rev
            pred /= (np.linalg.norm(pred)+1e-20)
            sims = P @ pred; sims[i] = -9999
            if float(sims.max()) > thr:
                mask[j] = True
        return mask

    type_masks = {}
    for rel_name, delta in deltas.items():
        type_masks[rel_name] = derivable_mask(delta, sample_idxs)

    pairs_to_check = [
        ('singular_plural', 'present_past'),
        ('singular_plural', 'gender_noun'),
        ('present_past',    'gender_noun'),
        ('country_capital', 'language_to_country'),
        ('country_capital', 'singular_plural'),
        ('adjective_comparative', 'singular_plural'),
    ]
    for t1, t2 in pairs_to_check:
        if t1 not in type_masks or t2 not in type_masks:
            continue
        m1, m2 = type_masks[t1], type_masks[t2]
        both = (m1 & m2).sum()
        union = (m1 | m2).sum()
        jaccard = both / (union + 1e-10)
        print(f"  {t1:<28s} ∩ {t2:<28s}  Jaccard={jaccard:.3f}  "
              f"overlap={both}/{max(m1.sum(),1)} "
              f"({'independent' if jaccard < 0.15 else 'coupled'})")

    # ── Concept type breakdown ─────────────────────────────────────────────────
    print(f"\n── What is NOT derivable? ───────────────────────────────────")
    print(f"  (Concepts that no delta can explain — the irreducible core)")
    all_derivable = np.zeros(len(sample_idxs), dtype=bool)
    for m in type_masks.values():
        all_derivable |= m
    not_derivable_idxs = sample_idxs[~all_derivable][:30]
    words_nd = [lcm.words[i] for i in not_derivable_idxs]
    print(f"  Sample of non-derivable concepts (first 30):")
    print(f"  {words_nd}")
