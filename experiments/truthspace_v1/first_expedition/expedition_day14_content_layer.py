#!/usr/bin/env python3
"""
Expedition Day 14 — The Content Layer: What Are the Other 480 Axes?

We have named:
  - 8 Grammar axes: Ax2, 5, 7, 15, 17, 18, 40, 54 (Killing vectors)
  - 8 Register/routing axes: Ax9, 110, 168, 171, 236, 307, 374, 375

That leaves ~484 uncharacterised axes. If Language = Grammar ⊗ Register ⊗ Content,
the remaining axes should be the CONTENT layer — the actual semantic meaning of concepts.

Tests:
  1. Sample 20 random axes, inspect top/bottom vocabulary → categorise as:
     grammar, register, topical-domain, entity-type, unknown
  2. Measure axis mass distribution: do content axes have broader or narrower
     vocabulary distributions than grammar axes?
  3. For selected topical-domain axes, verify: do top-50 concepts belong to
     one coherent semantic field?
  4. Are there axes that correspond to recognisable categories like:
     biology, geography, technology, emotion, temporality, colour, etc.?
  5. Test: how many axes are needed to distinguish between pairs of concepts
     in the same broad domain (e.g., apple vs orange, piano vs violin)?
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

GRAMMAR_AXES  = {2, 5, 7, 15, 17, 18, 40, 54}
REGISTER_AXES = {9, 110, 168, 171, 236, 307, 374, 375}
NAMED_AXES    = GRAMMAR_AXES | REGISTER_AXES

# Sample axes to investigate
SAMPLE_AXES = [
    1, 3, 4, 6, 8, 10, 11, 12, 13, 14,
    20, 25, 30, 35, 50, 75, 100, 150, 200, 250,
    300, 350, 400, 450, 499,
]

# Domain probe words — to find axes that separate semantic domains
DOMAIN_PROBES = {
    'biology':    ['cell','protein','DNA','enzyme','tissue','bacteria',
                   'evolution','genome','mitosis','photosynthesis'],
    'music':      ['melody','chord','rhythm','tempo','harmony','scale',
                   'note','instrument','symphony','bass'],
    'geography':  ['mountain','river','ocean','desert','continent','latitude',
                   'longitude','climate','terrain','peninsula'],
    'technology': ['algorithm','software','hardware','network','processor',
                   'database','circuit','protocol','compiler','bandwidth'],
    'emotion':    ['happiness','sadness','anger','fear','love','grief',
                   'anxiety','joy','disgust','envy'],
    'food':       ['bread','meat','vegetable','fruit','spice','recipe',
                   'ingredient','cuisine','flavour','nutrition'],
    'legal':      ['law','contract','court','verdict','judge','plaintiff',
                   'statute','jurisdiction','evidence','testimony'],
    'sports':     ['goal','score','team','tournament','athlete','champion',
                   'referee','stadium','coach','competition'],
}

N_TOP = 20


def get_axis_vocab(lcm, axis_idx, P, n=N_TOP):
    """Top and bottom N concepts on axis axis_idx."""
    col = P[:, axis_idx].astype(np.float64)
    top_idx = np.argsort(col)[-n:][::-1]
    bot_idx = np.argsort(col)[:n]
    top = [(lcm.words[i], float(col[i])) for i in top_idx]
    bot = [(lcm.words[i], float(col[i])) for i in bot_idx]
    return top, bot


def domain_axis_scores(lcm, domain_words, P):
    """
    For each axis, compute mean |projection| of domain words minus global mean.
    Axes where domain words have unusually high/low projections are domain-relevant.
    """
    global_mean = np.abs(P).mean(axis=0)  # (n_axes,)
    domain_projs = []
    for w in domain_words:
        try:
            proj, _ = lcm._get_proj(w)
            domain_projs.append(np.abs(proj.astype(np.float64)))
        except RuntimeError:
            pass
    if not domain_projs:
        return None
    domain_mean = np.mean(domain_projs, axis=0)
    return domain_mean - global_mean  # positive = domain uses this axis more than average


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float64)   # (N, 500)
    n_concepts, n_axes = P.shape

    print(f"\n{'='*65}")
    print(f"DAY 14 — The Content Layer: What Are the Other 480 Axes?")
    print(f"{'='*65}")
    print(f"  Total axes: {n_axes}, Grammar: {len(GRAMMAR_AXES)}, "
          f"Register: {len(REGISTER_AXES)}, Unnamed: {n_axes - len(NAMED_AXES)}")

    # ── Section 1: Sample axes, inspect vocabulary ────────────────────────────
    print(f"\n── Section 1: Vocabulary of sampled unnamed axes ────────────")
    for ax in SAMPLE_AXES:
        if ax in NAMED_AXES:
            continue
        top, bot = get_axis_vocab(lcm, ax, P, n=10)
        top_str  = "  ".join(f"{w}" for w, _ in top[:6])
        bot_str  = "  ".join(f"{w}" for w, _ in bot[:6])
        ax_name  = lcm.axis_names[ax][:35]
        print(f"\n  Ax{ax:<4} [{ax_name}]")
        print(f"    + {top_str}")
        print(f"    - {bot_str}")

    # ── Section 2: Axis mass distribution (sparsity per axis) ─────────────────
    print(f"\n── Section 2: Axis sparsity — fraction of concepts near zero ──")
    print(f"  (Comparing grammar, register, and unnamed axes)\n")

    thresholds = [0.02, 0.05, 0.10]
    groups = {
        'grammar (n=8)':   sorted(GRAMMAR_AXES),
        'register (n=8)':  sorted(REGISTER_AXES),
        'unnamed (n=484)': [i for i in range(n_axes) if i not in NAMED_AXES],
    }

    for group_name, indices in groups.items():
        cols = P[:, indices]  # (N, k)
        for thr in thresholds:
            frac_near_zero = float((np.abs(cols) < thr).mean())
            print(f"  {group_name:<22} |proj| < {thr}: {100*frac_near_zero:.1f}% of (concept, axis) pairs near zero")
        print()

    # ── Section 3: Domain axis search ─────────────────────────────────────────
    print(f"\n── Section 3: Which axes separate semantic domains? ─────────")
    print(f"  (Finding axes where domain words project unusually high)\n")

    domain_results = {}
    for domain, words in DOMAIN_PROBES.items():
        scores = domain_axis_scores(lcm, words, P)
        if scores is None:
            continue
        top_axes = np.argsort(scores)[-5:][::-1]
        domain_results[domain] = (scores, top_axes)
        print(f"  {domain:<12}: top 5 axes = "
              f"{', '.join(f'Ax{i}({scores[i]:+.4f})' for i in top_axes)}")
        for i in top_axes[:2]:
            tag = "[GRAMMAR]" if i in GRAMMAR_AXES else \
                  "[REGISTER]" if i in REGISTER_AXES else "[content?]"
            top_v, _ = get_axis_vocab(lcm, i, P, n=5)
            top_str  = ", ".join(w for w, _ in top_v)
            print(f"    Ax{i:<4} {tag}: {top_str}")
        print()

    # ── Section 4: Within-domain concept discrimination ───────────────────────
    print(f"\n── Section 4: How many axes separate concepts in same domain? ──")
    pairs = [
        ('apple',    'orange',   'fruit'),
        ('piano',    'violin',   'music'),
        ('france',   'germany',  'country'),
        ('happiness','sadness',  'emotion'),
        ('run',      'walk',     'motion verb'),
        ('algorithm','protocol', 'technology'),
        ('lion',     'tiger',    'large cat'),
        ('red',      'blue',     'colour'),
    ]

    print(f"  {'Pair':<30}  {'Cos sim':<10}  {'n axes >0.05 diff':<20}  Top discriminating axis")
    print("  " + "─" * 75)
    for w1, w2, domain in pairs:
        try:
            p1, _ = lcm._get_proj(w1)
            p2, _ = lcm._get_proj(w2)
        except RuntimeError:
            print(f"  {w1}/{w2}: not in vocab")
            continue
        p1 = p1.astype(np.float64)
        p2 = p2.astype(np.float64)
        cos_sim = float(np.dot(p1/np.linalg.norm(p1), p2/np.linalg.norm(p2)))
        diff    = np.abs(p1 - p2)
        n_disc  = int((diff > 0.05).sum())
        top_ax  = int(np.argmax(diff))
        top_tag = "[G]" if top_ax in GRAMMAR_AXES else \
                  "[R]" if top_ax in REGISTER_AXES else "[C]"
        print(f"  {w1}/{w2:<22} ({domain:<12})  cos={cos_sim:.4f}    "
              f"{n_disc:<20}  Ax{top_ax}{top_tag} (diff={diff[top_ax]:.4f})")

    # ── Section 5: Axis coverage of concept fingerprint ───────────────────────
    print(f"\n── Section 5: How many axes define a concept? ───────────────")
    print(f"  (Cumulative variance captured by top-k axes per concept)\n")
    test_words = ['king', 'cookie', 'paris', 'run', 'democracy', 'enzyme', 'piano']
    for word in test_words:
        try:
            proj, _ = lcm._get_proj(word)
        except RuntimeError:
            continue
        proj = proj.astype(np.float64)
        sq   = proj ** 2
        sq_sorted = np.sort(sq)[::-1]
        total     = sq.sum()
        cumvar    = np.cumsum(sq_sorted) / total
        n50  = int(np.searchsorted(cumvar, 0.50)) + 1
        n90  = int(np.searchsorted(cumvar, 0.90)) + 1
        n99  = int(np.searchsorted(cumvar, 0.99)) + 1
        top_ax = int(np.argmax(np.abs(proj)))
        tag    = "[G]" if top_ax in GRAMMAR_AXES else \
                 "[R]" if top_ax in REGISTER_AXES else "[C]"
        print(f"  {word:<14}: 50%={n50:>4} axes, 90%={n90:>4} axes, 99%={n99:>4} axes  "
              f"top={top_ax}{tag}({proj[top_ax]:+.3f})")
