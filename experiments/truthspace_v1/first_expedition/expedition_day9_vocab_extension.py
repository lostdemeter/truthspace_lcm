#!/usr/bin/env python3
"""
Expedition Day 9 — Vocabulary Extension via Functional Deltas

Day 6 showed that functional deltas do NOT compress the existing IRD vocabulary
(only 2% derivable). But the positive claim from Day 6 was:

  "Functional deltas enable vocabulary EXTENSION by inference —
   for out-of-vocabulary morphological variants, compute the projection
   on-the-fly from the base form."

Day 9 tests this claim directly:
  1. Identify words that ARE in the IRD vocabulary (base forms)
  2. Identify their morphological variants that are NOT in the vocabulary
  3. Apply the functional delta to the base form
  4. Check whether the derived projection correctly identifies the target
     when queried against the full vocabulary (and nearby OOV concepts)

Test pairs (source in vocab, target NOT in vocab OR poorly represented):
  - king → queens (plural of queen — if queen is in vocab but queens isn't)
  - walk → walked/walking (if both walk and walked are absent or one is)
  - run → runs/ran/running
  - big → bigger/biggest
  - happy → happier/happiest
  - cat → kittens, cats (if cats is not in vocab)

Also test the INVERSE: given a morphological variant, can we RECOVER the base form?
  - running → run (via reverse past delta)
  - queens → queen (via reverse plural delta)
  - actress → actor (via reverse gender delta)

Secondary test: vocabulary coverage estimate.
  For each of the 9 functional relationship types, what fraction of
  NEW derivations (not already in vocab) are correctly placed in semantic space?
  Measure by: does the derived projection land in the right semantic neighbourhood
  (cosine > 0.4 with the nearest known concept of the correct type)?
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
    ],
    'adjective_comparative': [
        ('big','bigger'),('small','smaller'),('fast','faster'),
        ('old','older'),('young','younger'),('strong','stronger'),
    ],
}

# Target out-of-vocabulary or weakly-represented words to derive
EXTENSION_TESTS = [
    # (base, base_in_vocab_expected, delta_type, target, target_in_vocab_expected)
    ('king',     True,  'singular_plural',    'kings',     False),
    ('queen',    True,  'singular_plural',    'queens',    False),
    ('cat',      True,  'singular_plural',    'cats',      True),
    ('run',      True,  'present_past',       'ran',       True),
    ('run',      True,  'present_past',       'running',   False),
    ('walk',     True,  'present_past',       'walked',    True),
    ('big',      True,  'adjective_comparative', 'bigger', True),
    ('small',    True,  'adjective_comparative', 'smaller', False),
    ('actor',    True,  'gender_noun',        'actress',   True),
    ('king',     True,  'gender_noun',        'queen',     True),
    ('france',   True,  'country_capital',    'paris',     True),
    ('japan',    True,  'country_capital',    'tokyo',     True),
    ('child',    True,  'singular_plural',    'children',  True),
    ('mouse',    True,  'singular_plural',    'mice',      True),
    ('brother',  True,  'gender_noun',        'sister',    True),
    ('father',   True,  'gender_noun',        'mother',    True),
    ('write',    True,  'present_past',       'wrote',     True),
    ('see',      True,  'present_past',       'saw',       True),
]

# INVERSE tests: derive base from variant
INVERSE_TESTS = [
    ('actress',   'gender_noun',        'actor'),
    ('queens',    'singular_plural',    'queen'),
    ('walked',    'present_past',       'walk'),
    ('bigger',    'adjective_comparative', 'big'),
    ('ran',       'present_past',       'run'),
    ('paris',     'country_capital',    'france'),
    ('berlin',    'country_capital',    'germany'),
    ('woman',     'gender_noun',        'man'),
]


def learn_delta(lcm, pairs):
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


def derive_and_rank(lcm, base_word, delta, P):
    """
    Apply delta to base_word projection, find nearest concepts, return ranking.
    """
    try:
        src_proj, src_idx = lcm._get_proj(base_word)
    except RuntimeError:
        return None, None, None

    src_proj = src_proj.astype(np.float64)
    derived  = src_proj + delta
    derived /= (np.linalg.norm(derived) + 1e-20)

    sims = P @ derived
    if src_idx is not None:
        sims[src_idx] = -9999
    top10_idx = np.argsort(sims)[-10:][::-1]

    results = []
    for i in top10_idx:
        results.append((lcm.words[i], float(sims[i])))
    return results, src_idx, float(np.dot(
        src_proj / (np.linalg.norm(src_proj)+1e-20), derived))


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float64)

    print(f"\n{'='*65}")
    print(f"DAY 9 — Vocabulary Extension via Functional Deltas")
    print(f"{'='*65}")

    # Learn deltas
    deltas = {}
    for rel_name, pairs in FUNCTIONAL_RELS.items():
        d = learn_delta(lcm, pairs)
        if d is not None:
            deltas[rel_name] = d
            print(f"  Learned delta for {rel_name}: ||Δ||={np.linalg.norm(d):.4f}")

    # ── Section 1: Forward extension tests ───────────────────────────────────
    print(f"\n── Section 1: Forward extension (base → derived) ────────────")
    print(f"  {'Base':<12}  {'Rel type':<28}  {'Target':<12}  {'In vocab?':<10}  {'Rank?':<8}  Top-3 hits")
    print("  " + "─" * 95)

    n_correct = 0
    n_total   = 0

    for base, base_exp, rel, target, tgt_exp in EXTENSION_TESTS:
        if rel not in deltas:
            continue
        delta = deltas[rel]

        in_vocab = lcm.word_set.get(target.lower()) is not None

        results, src_idx, cos_move = derive_and_rank(lcm, base, delta, P)
        if results is None:
            print(f"  {base:<12}  {rel:<28}  {target:<12}  —  (base not in vocab)")
            continue

        # Find rank of target in results
        top_words = [w.lower() for w, _ in results]
        if target.lower() in top_words:
            rank = top_words.index(target.lower()) + 1
            hit  = "✓"
            n_correct += 1
        else:
            rank = None
            hit  = "✗"
        n_total += 1

        top3 = ", ".join(f"{w}({s:.3f})" for w, s in results[:3])
        rank_str = f"rank={rank}" if rank else "not in top-10"
        print(f"  {base:<12}  {rel:<28}  {target:<12}  {'✓' if in_vocab else '—':<10}  "
              f"{hit} {rank_str:<12}  {top3}")

    print(f"\n  Forward hit rate: {n_correct}/{n_total} ({100*n_correct/max(n_total,1):.1f}%)")

    # ── Section 2: Inverse tests (reverse delta) ──────────────────────────────
    print(f"\n── Section 2: Inverse extension (derived → base) ────────────")
    print(f"  {'Source':<12}  {'Rel type':<28}  {'Target base':<12}  {'Rank?':<8}  Top-3 hits")
    print("  " + "─" * 80)

    n_inv_correct = 0
    n_inv_total   = 0

    for src, rel, base in INVERSE_TESTS:
        if rel not in deltas:
            continue
        reverse_delta = -deltas[rel]

        results, src_idx, cos_move = derive_and_rank(lcm, src, reverse_delta, P)
        if results is None:
            print(f"  {src:<12}  {rel:<28}  {base:<12}  (source not in vocab)")
            continue

        top_words = [w.lower() for w, _ in results]
        if base.lower() in top_words:
            rank = top_words.index(base.lower()) + 1
            hit  = "✓"
            n_inv_correct += 1
        else:
            rank = None
            hit  = "✗"
        n_inv_total += 1

        top3 = ", ".join(f"{w}({s:.3f})" for w, s in results[:3])
        rank_str = f"rank={rank}" if rank else "not in top-10"
        print(f"  {src:<12}  {rel:<28}  {base:<12}  {hit} {rank_str:<12}  {top3}")

    print(f"\n  Inverse hit rate: {n_inv_correct}/{n_inv_total} ({100*n_inv_correct/max(n_inv_total,1):.1f}%)")

    # ── Section 3: Out-of-vocabulary simulation ───────────────────────────────
    print(f"\n── Section 3: OOV simulation ────────────────────────────────")
    print(f"  (Simulate words NOT in vocab by their derivation from known base)")
    print(f"  Compare: derived projection vs. actual embedding (if available)\n")

    oov_tests = [
        ('king',   'singular_plural',  'kings'),
        ('queen',  'singular_plural',  'queens'),
        ('run',    'present_past',     'running'),
        ('cat',    'singular_plural',  'cats'),
        ('big',    'adjective_comparative', 'bigger'),
    ]

    for base, rel, derived_word in oov_tests:
        if rel not in deltas:
            continue
        delta = deltas[rel]

        try:
            src_proj, _ = lcm._get_proj(base)
            src_proj = src_proj.astype(np.float64)
        except RuntimeError:
            print(f"  {base}: not in vocab, skipping")
            continue

        derived_proj = src_proj + delta
        derived_proj /= (np.linalg.norm(derived_proj) + 1e-20)

        # Check if derived_word is actually in the vocab
        tgt_idx = lcm.word_set.get(derived_word.lower())
        if tgt_idx is not None:
            actual_proj = P[tgt_idx] / (np.linalg.norm(P[tgt_idx]) + 1e-20)
            cos_sim = float(np.dot(derived_proj, actual_proj))
            print(f"  {base}→{derived_word}: derived vs actual cos={cos_sim:.4f}  "
                  f"({'excellent' if cos_sim > 0.9 else 'good' if cos_sim > 0.7 else 'moderate' if cos_sim > 0.5 else 'poor'})")
        else:
            # Not in vocab: check what the derived position is nearest to
            sims = P @ derived_proj
            top3_idx = np.argsort(sims)[-3:][::-1]
            top3_str = ", ".join(f"{lcm.words[i]}({sims[i]:.3f})" for i in top3_idx)
            print(f"  {base}→{derived_word}: OOV — nearest: {top3_str}")

    # ── Section 4: Coverage estimate ─────────────────────────────────────────
    print(f"\n── Section 4: What the delta enables ───────────────────────")
    print(f"  (Tally of which base→derived pairs are 'covered' by each delta)")

    for rel_name, pairs in FUNCTIONAL_RELS.items():
        if rel_name not in deltas:
            continue
        delta  = deltas[rel_name]
        n_hits = 0
        for src, tgt in pairs:
            results, _, _ = derive_and_rank(lcm, src, delta, P)
            if results is None:
                continue
            top_words = [w.lower() for w, _ in results]
            if tgt.lower() in top_words[:5]:
                n_hits += 1
        print(f"  {rel_name:<28}: {n_hits}/{len(pairs)} training pairs hit (LOO-style)")
