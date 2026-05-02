#!/usr/bin/env python3
"""
Expedition Day 3 — The Relationship Type Survey

Day 2 established: delta substitution works for gender and capital, fails for
antonyms. The question is WHY — and how many relationship types ARE universal.

Hypothesis (to test or disprove):
  Universal relationship deltas are KILLING VECTORS of the semantic manifold —
  directions of translational symmetry along which the manifold is invariant.
  Moving every concept by Δgender preserves all pairwise distances.
  Antonyms fail because "antonymness" is not a global symmetry — it is a
  LOCAL axis inversion, different per dimension.

We test ~20 relationship types across multiple pairs each.
For each type, we compute:
  - delta_consistency: how well does the delta generalise across pairs?
    (leave-one-out cosine of predicted to actual target)
  - delta_variance: std of pairwise deltas — low variance = universal Killing vector
  - n_pairs: number of valid pairs tested

Killing vector candidates: high consistency, low variance, across many pairs.
Local transformations: high variance, inconsistent across pairs.

Also test: does the Killing vector count match the number of IRD axes?
(The isometry group dimension = number of independent symmetries of the manifold)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

# ── Relationship test batteries ────────────────────────────────────────────────

RELATIONSHIPS = {

    'gender_noun': [
        ('king', 'queen'), ('man', 'woman'), ('boy', 'girl'), ('actor', 'actress'),
        ('prince', 'princess'), ('son', 'daughter'), ('father', 'mother'),
        ('brother', 'sister'), ('husband', 'wife'), ('uncle', 'aunt'),
        ('grandfather', 'grandmother'), ('nephew', 'niece'),
    ],

    'country_capital': [
        ('france', 'paris'), ('germany', 'berlin'), ('italy', 'rome'),
        ('spain', 'madrid'), ('japan', 'tokyo'), ('china', 'beijing'),
        ('russia', 'moscow'), ('brazil', 'brasilia'), ('india', 'delhi'),
        ('egypt', 'cairo'), ('mexico', 'mexico city'), ('poland', 'warsaw'),
    ],

    'singular_plural': [
        ('cat', 'cats'), ('dog', 'dogs'), ('house', 'houses'), ('car', 'cars'),
        ('book', 'books'), ('tree', 'trees'), ('bird', 'birds'), ('fish', 'fish'),
        ('child', 'children'), ('mouse', 'mice'), ('foot', 'feet'), ('tooth', 'teeth'),
    ],

    'present_past': [
        ('run', 'ran'), ('walk', 'walked'), ('eat', 'ate'), ('write', 'wrote'),
        ('speak', 'spoke'), ('take', 'took'), ('go', 'went'), ('see', 'saw'),
        ('give', 'gave'), ('make', 'made'), ('come', 'came'), ('know', 'knew'),
    ],

    'adjective_comparative': [
        ('big', 'bigger'), ('small', 'smaller'), ('fast', 'faster'),
        ('old', 'older'), ('young', 'younger'), ('strong', 'stronger'),
        ('long', 'longer'), ('short', 'shorter'), ('hot', 'hotter'),
        ('cold', 'colder'), ('hard', 'harder'), ('soft', 'softer'),
    ],

    'antonym_temperature': [
        ('hot', 'cold'), ('warm', 'cool'), ('boiling', 'freezing'),
        ('heat', 'cold'), ('fire', 'ice'),
    ],

    'antonym_size': [
        ('big', 'small'), ('large', 'tiny'), ('huge', 'miniature'),
        ('giant', 'dwarf'), ('tall', 'short'),
    ],

    'antonym_speed': [
        ('fast', 'slow'), ('quick', 'sluggish'), ('rapid', 'gradual'),
        ('swift', 'slow'),
    ],

    'antonym_moral': [
        ('good', 'evil'), ('honest', 'dishonest'), ('kind', 'cruel'),
        ('brave', 'cowardly'), ('truth', 'lie'),
    ],

    'animal_sound': [
        ('dog', 'bark'), ('cat', 'meow'), ('cow', 'moo'),
        ('duck', 'quack'), ('snake', 'hiss'), ('lion', 'roar'),
    ],

    'animal_offspring': [
        ('cat', 'kitten'), ('dog', 'puppy'), ('cow', 'calf'),
        ('horse', 'foal'), ('sheep', 'lamb'), ('pig', 'piglet'),
        ('duck', 'duckling'), ('bear', 'cub'),
    ],

    'food_ingredient': [
        ('bread', 'flour'), ('cake', 'sugar'), ('pasta', 'wheat'),
        ('cheese', 'milk'), ('butter', 'cream'), ('wine', 'grape'),
        ('beer', 'barley'), ('chocolate', 'cocoa'),
    ],

    'profession_tool': [
        ('painter', 'brush'), ('carpenter', 'hammer'), ('surgeon', 'scalpel'),
        ('chef', 'knife'), ('writer', 'pen'), ('photographer', 'camera'),
    ],

    'hypernym_entity': [
        ('dog', 'animal'), ('car', 'vehicle'), ('apple', 'fruit'),
        ('oak', 'tree'), ('rose', 'flower'), ('salmon', 'fish'),
        ('eagle', 'bird'), ('diamond', 'mineral'), ('violin', 'instrument'),
    ],

    'language_to_country': [
        ('english', 'england'), ('french', 'france'), ('german', 'germany'),
        ('spanish', 'spain'), ('italian', 'italy'), ('japanese', 'japan'),
        ('russian', 'russia'), ('chinese', 'china'),
    ],

    'verb_noun_agent': [
        ('teach', 'teacher'), ('build', 'builder'), ('write', 'writer'),
        ('paint', 'painter'), ('play', 'player'), ('drive', 'driver'),
        ('bake', 'baker'), ('manage', 'manager'),
    ],

    'material_object': [
        ('wood', 'table'), ('metal', 'sword'), ('glass', 'bottle'),
        ('stone', 'wall'), ('cotton', 'shirt'), ('leather', 'shoe'),
    ],

    'color_fruit': [
        ('red', 'apple'), ('yellow', 'banana'), ('orange', 'orange'),
        ('green', 'lime'), ('purple', 'grape'), ('blue', 'blueberry'),
    ],

    'scale_relation': [
        ('atom', 'molecule'), ('cell', 'organ'), ('organ', 'body'),
        ('word', 'sentence'), ('sentence', 'paragraph'), ('letter', 'word'),
        ('second', 'minute'), ('minute', 'hour'), ('hour', 'day'),
    ],

    'sentiment_flip': [
        ('happy', 'sad'), ('love', 'hate'), ('hope', 'despair'),
        ('success', 'failure'), ('joy', 'grief'), ('peace', 'war'),
    ],

}


def measure_relationship(lcm, pairs, name):
    """
    For a list of (source, target) word pairs representing a relationship type,
    compute:
      - individual deltas
      - delta consistency (leave-one-out cross-validation)
      - delta variance (std of delta magnitudes)
    Returns a dict of results.
    """
    P = lcm.projections.astype(np.float64)

    # Collect valid pairs
    valid = []
    for a, b in pairs:
        try:
            pa, _ = lcm._get_proj(a)
            pb, _ = lcm._get_proj(b)
            valid.append((a, b, pa.astype(np.float64), pb.astype(np.float64)))
        except RuntimeError:
            pass

    if len(valid) < 2:
        return None

    # Individual deltas
    deltas = np.array([pb - pa for _, _, pa, pb in valid])   # (n, d)
    # Normalise for direction comparison
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    deltas_n = deltas / (norms + 1e-20)

    # Mean delta direction
    mean_delta = deltas.mean(axis=0)
    mean_delta_n = mean_delta / (np.linalg.norm(mean_delta) + 1e-20)

    # Delta variance: std of cosines between individual deltas and mean
    delta_cosines = deltas_n @ mean_delta_n    # (n,)
    delta_consistency = float(delta_cosines.mean())
    delta_variance    = float(delta_cosines.std())

    # Leave-one-out consistency: predict held-out target from average of others
    loo_ranks = []
    loo_cosines = []
    n = len(valid)
    for held_i in range(n):
        train_deltas = [deltas[j] for j in range(n) if j != held_i]
        if not train_deltas:
            continue
        delta_loo = np.mean(train_deltas, axis=0)
        a, b, pa, pb = valid[held_i]
        predicted = pa + delta_loo
        predicted /= (np.linalg.norm(predicted) + 1e-20)
        cos_to_target = float(np.dot(predicted, pb / (np.linalg.norm(pb) + 1e-20)))
        sims_all = P @ predicted
        rank = int((sims_all > cos_to_target).sum()) + 1
        loo_ranks.append(rank)
        loo_cosines.append(cos_to_target)

    return {
        'name':               name,
        'n_valid':            len(valid),
        'delta_consistency':  delta_consistency,
        'delta_variance':     delta_variance,
        'loo_mean_rank':      float(np.mean(loo_ranks)) if loo_ranks else None,
        'loo_median_rank':    float(np.median(loo_ranks)) if loo_ranks else None,
        'loo_mean_cos':       float(np.mean(loo_cosines)) if loo_cosines else None,
        'rank_le5':           sum(1 for r in loo_ranks if r <= 5),
        'rank_le20':          sum(1 for r in loo_ranks if r <= 20),
        'mean_delta_norm':    float(np.mean(norms)),
        'valid_pairs':        [(a, b) for a, b, _, _ in valid],
    }


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()

    print(f"\n{'='*70}")
    print("DAY 3 OBSERVATION LOG — The Relationship Type Survey")
    print(f"{'='*70}")
    print(f"\n  Testing {len(RELATIONSHIPS)} relationship types...")
    print(f"  {'Relationship':<28s}  {'n':<4s}  {'Δ-consist':<12s}  "
          f"{'Δ-var':<8s}  {'LOO med rank':<14s}  {'≤5':<4s}  {'≤20':<4s}  "
          f"Killing?")
    print("  " + "─" * 90)

    results = {}
    for name, pairs in RELATIONSHIPS.items():
        r = measure_relationship(lcm, pairs, name)
        if r is None:
            print(f"  {name:<28s}  insufficient valid pairs")
            continue
        results[name] = r

        loo_med   = f"{r['loo_median_rank']:.0f}" if r['loo_median_rank'] else "—"
        is_killing = (r['delta_consistency'] > 0.6
                      and r['delta_variance'] < 0.3
                      and r['loo_mean_rank'] is not None
                      and r['loo_mean_rank'] < 50)
        flag = "✓ YES" if is_killing else ("~ partial" if r['delta_consistency'] > 0.4 else "✗ NO")

        print(f"  {name:<28s}  {r['n_valid']:<4d}  "
              f"{r['delta_consistency']:+.3f}       "
              f"{r['delta_variance']:.3f}    "
              f"{loo_med:<14s}  "
              f"{r['rank_le5']:<4d}  {r['rank_le20']:<4d}  {flag}")

    # ── Killing vector analysis ────────────────────────────────────────────────
    killing = {k: v for k, v in results.items()
               if v['delta_consistency'] > 0.6 and v['delta_variance'] < 0.3}
    partial = {k: v for k, v in results.items()
               if 0.4 <= v['delta_consistency'] <= 0.6}
    local   = {k: v for k, v in results.items()
               if v['delta_consistency'] < 0.4}

    print(f"\n  ── Classification ──────────────────────────────────────────")
    print(f"  Killing vectors (universal):  {sorted(killing.keys())}")
    print(f"  Partial symmetries:           {sorted(partial.keys())}")
    print(f"  Local transformations:        {sorted(local.keys())}")

    # ── Orthogonality of Killing vectors ──────────────────────────────────────
    if len(killing) >= 2:
        print(f"\n  ── Killing vector mutual angles ────────────────────────────")
        print(f"  (cos = 0 → orthogonal symmetries; cos ≠ 0 → coupled symmetries)")
        kv_names = sorted(killing.keys())
        kv_deltas = {}
        for name in kv_names:
            pairs_vecs = [(lcm._get_proj(a)[0].astype(np.float64),
                           lcm._get_proj(b)[0].astype(np.float64))
                          for a, b in results[name]['valid_pairs']
                          if True]
            delta = np.mean([b - a for a, b in pairs_vecs], axis=0)
            delta /= (np.linalg.norm(delta) + 1e-20)
            kv_deltas[name] = delta

        print(f"  {'':28s}  " + "  ".join(f"{n[:12]:<12s}" for n in kv_names))
        for n1 in kv_names:
            row = f"  {n1:<28s}  "
            for n2 in kv_names:
                c = float(np.dot(kv_deltas[n1], kv_deltas[n2]))
                row += f"{c:+.3f}       "
            print(row)

    # ── Counting independent symmetries ───────────────────────────────────────
    print(f"\n  ── Symmetry count ──────────────────────────────────────────")
    print(f"  Killing vectors identified: {len(killing)}")
    print(f"  Partial symmetries:         {len(partial)}")
    print(f"  Local transformations:      {len(local)}")
    total_pairs = sum(v['n_valid'] for v in results.values())
    derivable   = sum(v['n_valid'] for v in killing.values())
    print(f"\n  Valid pairs tested:    {total_pairs}")
    print(f"  Derivable via Killing: {derivable} ({derivable/total_pairs*100:.1f}%)")

    # ── What antonyms reveal about the manifold ────────────────────────────────
    print(f"\n  ── Antonym anatomy ─────────────────────────────────────────")
    print(f"  (Why are antonyms not Killing vectors?)")
    antonym_types = [k for k in results if k.startswith('antonym')]
    for name in antonym_types:
        r = results[name]
        print(f"\n  {name}:")
        print(f"    delta_consistency = {r['delta_consistency']:.3f}  "
              f"(across {r['n_valid']} pairs)")
        # Show individual delta cosines to mean
        valid = []
        for a, b in r['valid_pairs']:
            try:
                pa, _ = lcm._get_proj(a)
                pb, _ = lcm._get_proj(b)
                valid.append((a, b, pa.astype(np.float64), pb.astype(np.float64)))
            except RuntimeError:
                pass
        if valid:
            deltas = np.array([pb - pa for _, _, pa, pb in valid])
            mean_d = deltas.mean(axis=0)
            mean_d /= (np.linalg.norm(mean_d) + 1e-20)
            for i, (a, b, _, _) in enumerate(valid):
                d = deltas[i] / (np.linalg.norm(deltas[i]) + 1e-20)
                cos = float(np.dot(d, mean_d))
                print(f"    {a}→{b}: delta_cos_to_mean = {cos:+.3f}")

    # ── Cross-type delta angles ────────────────────────────────────────────────
    print(f"\n  ── Selected pairwise delta angles ──────────────────────────")
    probe_pairs = [
        ('gender_noun', 'country_capital'),
        ('gender_noun', 'singular_plural'),
        ('gender_noun', 'present_past'),
        ('gender_noun', 'sentiment_flip'),
        ('antonym_temperature', 'antonym_size'),
        ('antonym_temperature', 'antonym_speed'),
        ('antonym_temperature', 'antonym_moral'),
        ('antonym_size', 'antonym_speed'),
        ('animal_offspring', 'food_ingredient'),
        ('hypernym_entity', 'verb_noun_agent'),
    ]
    kv_deltas_all = {}
    for name, r in results.items():
        if name not in kv_deltas_all:
            try:
                vecs = []
                for a, b in r['valid_pairs']:
                    try:
                        pa, _ = lcm._get_proj(a)
                        pb, _ = lcm._get_proj(b)
                        vecs.append(pb.astype(np.float64) - pa.astype(np.float64))
                    except RuntimeError:
                        pass
                if vecs:
                    mean_v = np.mean(vecs, axis=0)
                    mean_v /= (np.linalg.norm(mean_v) + 1e-20)
                    kv_deltas_all[name] = mean_v
            except Exception:
                pass

    for t1, t2 in probe_pairs:
        if t1 in kv_deltas_all and t2 in kv_deltas_all:
            cos = float(np.dot(kv_deltas_all[t1], kv_deltas_all[t2]))
            note = "orthogonal" if abs(cos) < 0.1 else ("coupled" if abs(cos) < 0.3 else "aligned")
            print(f"  {t1:<28s} ↔ {t2:<28s}  cos={cos:+.4f}  [{note}]")
