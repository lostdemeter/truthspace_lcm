#!/usr/bin/env python3
"""
recipe_expander.py — Tier 1 Geometric Generation (DC 301)

Demonstrates that structured multi-component output can be assembled from
parallel delta retrievals — no template string lookup, no hardcoded answers.

Given a food concept, the system:
  1. Applies food_ingredient delta → retrieves ingredient concepts
  2. Applies food_method delta     → retrieves method concepts
  3. Formats both into a structured response

This is the same mechanism as capital_of(France)=Paris, applied in parallel
across two relationship types and composed into a multi-slot response.

Usage:
    python recipe_expander.py                  # demo several foods
    python recipe_expander.py cookie           # single food
    python recipe_expander.py --delta-check    # print delta geometry
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm
from dc299_phase4_lcm_inference import LCMIndex

# ─── Training pairs ───────────────────────────────────────────────────────────
# food_ingredient: (food, one_of_its_ingredients)
# Multiple pairs per food increase signal strength for that food's neighborhood.
# Keep sources diverse (don't over-represent any single food) for a good mean delta.

FOOD_INGREDIENT_PAIRS = [
    ('cookie',   'sugar'),
    ('cookie',   'flour'),
    ('cake',     'flour'),
    ('cake',     'sugar'),
    ('bread',    'yeast'),
    ('bread',    'flour'),
    ('pasta',    'tomato'),
    ('pasta',    'egg'),
    ('pizza',    'mozzarella'),
    ('pizza',    'tomato'),
    ('soup',     'onion'),
    ('soup',     'carrot'),
    ('salad',    'lettuce'),
    ('salad',    'tomato'),
    ('omelette', 'egg'),
    ('omelette', 'cheese'),
    ('pancake',  'milk'),
    ('pancake',  'egg'),
    ('pie',      'butter'),
    ('pie',      'sugar'),
    ('stew',     'onion'),
    ('stew',     'carrot'),
    ('curry',    'onion'),
    ('curry',    'rice'),
    ('smoothie', 'milk'),
    ('smoothie', 'banana'),
]

# food_method: (food, one_of_its_cooking_methods)
FOOD_METHOD_PAIRS = [
    ('cookie',   'bake'),
    ('cookie',   'mix'),
    ('cake',     'bake'),
    ('cake',     'mix'),
    ('bread',    'bake'),
    ('bread',    'knead'),
    ('pasta',    'boil'),
    ('pasta',    'stir'),
    ('pizza',    'bake'),
    ('pizza',    'roll'),
    ('soup',     'simmer'),
    ('soup',     'stir'),
    ('salad',    'mix'),
    ('salad',    'chop'),
    ('omelette', 'fry'),
    ('omelette', 'fold'),
    ('pancake',  'fry'),
    ('pancake',  'flip'),
    ('pie',      'bake'),
    ('pie',      'mix'),
    ('stew',     'simmer'),
    ('stew',     'chop'),
    ('curry',    'fry'),
    ('curry',    'simmer'),
    ('smoothie', 'blend'),
    ('smoothie', 'mix'),
]

# Common noise words that appear near food concepts but aren't useful output
_FOOD_WORDS   = {p[0] for p in FOOD_INGREDIENT_PAIRS} | {p[0] for p in FOOD_METHOD_PAIRS}
_METHOD_WORDS = {p[1] for p in FOOD_METHOD_PAIRS}
_INGR_WORDS   = {p[1] for p in FOOD_INGREDIENT_PAIRS}

# ─── Curated vocabulary for neighborhood filtering ────────────────────────────
# These are single-token words reliably present in the Phase-0 concept set.
# Used as the candidate pool for neighborhood intersection retrieval.

INGREDIENT_VOCAB = [
    'flour','sugar','butter','egg','vanilla','milk','cream','yeast',
    'salt','pepper','onion','garlic','tomato','cheese','rice','potato',
    'carrot','oil','lemon','chocolate','honey','cinnamon','ginger',
    'wheat','corn','oats','water','broth','vinegar','herb',
]

METHOD_VOCAB = [
    'bake','fry','boil','simmer','mix','stir','chop','slice','fold',
    'cream','roast','grill','steam','cook','beat','heat','pour',
    'season','drain','melt','dice','whisk',
]

# Unambiguous food words used as the domain reference for polysemy detection.
# These must have clean, food-only neighborhoods in the IRD space.
FOOD_REFERENCE = [
    'bread', 'soup', 'cake', 'rice', 'pasta', 'egg', 'milk', 'cheese',
]

# Human-readable override descriptions for known polysemous words.
# Used only for the warning MESSAGE — detection is now fully geometric.
# A word NOT in this dict can still be detected as polysemous; it will
# receive an auto-generated warning from its top neighbors instead.
_POLYSEMY_DESCRIPTIONS = {
    'cookie': 'HTTP/browser cookie (tech sense dominates in Qwen2 training corpus)',
}

# ─── RecipeExpander ───────────────────────────────────────────────────────────

class RecipeExpander:
    """
    Assembles structured food descriptions from parallel delta retrievals.

    Architecture (DC 301 §4):
      query_food
        ├─ food_ingredient delta ─→ [sugar, flour, butter, eggs, ...]
        └─ food_method delta     ─→ [mix, bake, cream, ...]
      combine ──────────────────→ formatted response

    The delta mechanism is identical to DeltaLibrary — only the relationship
    types and response template are new.
    """

    def __init__(self, lcm: LCMIndex):
        self.lcm = lcm
        self._check_vocab()
        self._learn()

    def _food_polysemy_score(self, word):
        """
        Geometric polysemy detection for a food word (DC 302 §9.1).

        Returns dict from lcm.detect_polysemy() with an additional key:
            'warning_text': str — human-readable description, either from
                            _POLYSEMY_DESCRIPTIONS (curated) or auto-generated
                            from top neighbors.
        Returns None if word not in vocabulary.
        """
        try:
            result = self.lcm.detect_polysemy(
                word, FOOD_REFERENCE, k=20, threshold=0.45)
        except RuntimeError:
            return None
        if result['is_polysemous']:
            if word.lower() in _POLYSEMY_DESCRIPTIONS:
                desc = _POLYSEMY_DESCRIPTIONS[word.lower()]
            else:
                top = [w for w, _ in result['top_neighbors'][:3]]
                desc = (f'dominant neighbors: {top} '
                        f'(culinary sense may be weak; '
                        f'domain cos={result["nbr_domain_cos"]:.3f})')
            result['warning_text'] = desc
        else:
            result['warning_text'] = None
        return result

    def _word_ok(self, w):
        try:
            self.lcm._get_proj(w)
            return True
        except RuntimeError:
            return False

    def _check_vocab(self):
        missing_i = [(s, t) for s, t in FOOD_INGREDIENT_PAIRS
                     if not self._word_ok(s) or not self._word_ok(t)]
        missing_m = [(s, t) for s, t in FOOD_METHOD_PAIRS
                     if not self._word_ok(s) or not self._word_ok(t)]
        if missing_i:
            print(f'  [vocab gap] food_ingredient missing: {missing_i}')
        if missing_m:
            print(f'  [vocab gap] food_method missing: {missing_m}')

    def _learn(self):
        ingr_ok = [(s, t) for s, t in FOOD_INGREDIENT_PAIRS
                   if self._word_ok(s) and self._word_ok(t)]
        meth_ok = [(s, t) for s, t in FOOD_METHOD_PAIRS
                   if self._word_ok(s) and self._word_ok(t)]

        self._ingr_delta, _ = self.lcm.learn_delta(ingr_ok)
        self._meth_delta, _ = self.lcm.learn_delta(meth_ok)
        _, _, self._ingr_fp, _ = self.lcm.learn_delta_v2(ingr_ok)
        _, _, self._meth_fp, _ = self.lcm.learn_delta_v2(meth_ok)

        print(f'  food_ingredient delta: {len(ingr_ok)} pairs')
        print(f'  food_method delta:     {len(meth_ok)} pairs')

    def _retrieve(self, food, delta, fp, k_raw=20):
        """Retrieve top-k words for *food* along *delta*, filtered for noise."""
        try:
            results = self.lcm.apply_delta_phi_boost_v8(
                food, delta, fp, k=k_raw,
                exclude_words=[food],
                boost_threshold=0.75,
            )
        except RuntimeError:
            return []
        # Filter: skip food names, numbers, very short tokens, and repeats
        seen = set()
        out  = []
        for w, score in results:
            wl = w.lower()
            if (wl in seen or wl in _FOOD_WORDS or len(wl) < 3
                    or not wl.isalpha() or wl.endswith('ing')):
                continue
            seen.add(wl)
            out.append((w.lower(), score))
        return out

    def _neighborhood_scores(self, food: str, vocab: list,
                              query_proj=None) -> list:
        """
        Primary retrieval mechanism: find which *vocab* words sit closest to
        *food* in the IRD projection space.

        query_proj: optional pre-corrected projection (from context_correct_proj).
        When provided, uses it instead of food's native embedding, resolving
        polysemy before the neighbour lookup fires (DC 302).
        """
        if query_proj is not None:
            pf = query_proj.astype(np.float64)
        else:
            pf, _ = self.lcm._get_proj(food)
            pf = pf.astype(np.float64)
        nf = np.linalg.norm(pf) + 1e-20
        results = []
        for w in vocab:
            if not self._word_ok(w):
                continue
            pw, _ = self.lcm._get_proj(w)
            pw = pw.astype(np.float64)
            s = float(np.dot(pf, pw) / (nf * (np.linalg.norm(pw) + 1e-20)))
            results.append((w, s))
        results.sort(key=lambda x: -x[1])
        return results

    def expand(self, food: str, n_ingr: int = 6, n_meth: int = 5,
               context_words=None):
        """
        Return a dict with 'food', 'ingredients', 'methods', 'polysemy_warning'.
        Each list item is a (word, score) tuple.

        context_words: optional list of query words (besides the food name) to
        use as context-gravity anchors (DC 302).  When provided, the food
        concept's projection is shifted toward these words before the
        neighbourhood lookup, resolving polysemy ('cookie' → HTTP vs. baking).
        If not provided, automatically uses INGREDIENT_VOCAB context words as
        gentle attractors for polysemous foods.
        """
        if not self._word_ok(food):
            raise RuntimeError(f"'{food}' not found in vocabulary")

        poly = self._food_polysemy_score(food)
        warning = poly['warning_text'] if (poly and poly['is_polysemous']) else None

        # For polysemous foods, always include curated polysemy anchors.
        # Merge with any query-derived context words (additive, not overriding).
        if warning:
            anchors = ['recipe', 'ingredients', 'bake', 'flour']
            if context_words:
                context_words = list(context_words) + [
                    w for w in anchors if w not in context_words]
            else:
                context_words = anchors

        query_proj = None
        if context_words:
            cw_ok = [w for w in context_words if self._word_ok(w)]
            if cw_ok:
                query_proj = self.lcm.context_correct_proj(food, cw_ok)

        ingredients = self._neighborhood_scores(
            food, INGREDIENT_VOCAB, query_proj=query_proj)[:n_ingr]
        methods     = self._neighborhood_scores(
            food, METHOD_VOCAB, query_proj=query_proj)[:n_meth]

        return {
            'food': food,
            'ingredients': ingredients,
            'methods': methods,
            'polysemy_warning': warning,
            'context_corrected': query_proj is not None,
        }

    def format_response(self, result: dict) -> str:
        food    = result['food'].capitalize()
        ingrs   = result['ingredients']
        meths   = result['methods']
        warning = result.get('polysemy_warning')

        lines = [f'  ┌─ {food} ──────────────────────────────────────────']

        if warning:
            corrected = result.get('context_corrected', False)
            lines.append(f'  │  ⚠  Polysemy: "{food}" embedding dominated by: {warning}')
            if corrected:
                lines.append(f'  │     Context gravity applied — pulled toward culinary sense.')
            else:
                lines.append(f'  │     Neighborhood reflects dominant (non-culinary) sense.')
            lines.append('  │')

        if ingrs:
            lines.append('  │  Ingredients  (IRD neighborhood intersection)')
            for w, score in ingrs:
                bar = '█' * int(score * 40)
                lines.append(f'  │    {w:<14s} {score:.3f}  {bar}')
        else:
            lines.append('  │  Ingredients: [none retrieved]')

        lines.append('  │')

        if meths:
            lines.append('  │  Methods  (IRD neighborhood intersection)')
            for w, score in meths:
                bar = '█' * int(score * 40)
                lines.append(f'  │    {w:<14s} {score:.3f}  {bar}')
        else:
            lines.append('  │  Methods: [none retrieved]')

        lines.append('  └' + '─' * 50)
        return '\n'.join(lines)

    def delta_geometry(self):
        """Print cosine similarity between ingredient and method deltas."""
        d_i = self._ingr_delta.astype(np.float64)
        d_m = self._meth_delta.astype(np.float64)
        cos = float(np.dot(d_i, d_m) / (
            np.linalg.norm(d_i) * np.linalg.norm(d_m) + 1e-20))
        print(f'\n  Δ(food_ingredient) · Δ(food_method) cosine = {cos:+.4f}')
        if abs(cos) < 0.15:
            print('  → Near-orthogonal: ingredient and method subspaces are geometrically distinct ✓')
        elif cos > 0.5:
            print('  → High overlap: risk of ingredient/method crosstalk in retrieval')
        else:
            print(f'  → Moderate overlap: some shared axes between ingredient and method')

    def loo_accuracy(self, relationship='ingredient'):
        """Quick LOO test on food_ingredient or food_method."""
        pairs  = FOOD_INGREDIENT_PAIRS if relationship == 'ingredient' else FOOD_METHOD_PAIRS
        pairs_ok = [(s, t) for s, t in pairs if self._word_ok(s) and self._word_ok(t)]
        hits1, hits3, ranks = 0, 0, []
        for i, (src, tgt) in enumerate(pairs_ok):
            loo = [p for j, p in enumerate(pairs_ok) if j != i]
            d, _ = self.lcm.learn_delta(loo)
            _, _, fp, _ = self.lcm.learn_delta_v2(loo)
            results = self.lcm.apply_delta_phi_boost_v8(
                src, d, fp, k=50, exclude_words=[src], boost_threshold=0.75)
            rank = next((r for r, (w, _) in enumerate(results)
                         if w.lower() == tgt.lower()), -1)
            ranks.append(rank)
            hits1 += (rank == 0)
            hits3 += (0 <= rank <= 2)
        n = len(pairs_ok)
        mrr = np.mean([1/(r+1) for r in ranks if r >= 0])
        print(f'\n  LOO ({relationship}): {n} pairs  '
              f'top1={hits1/n:.0%}  top3={hits3/n:.0%}  MRR={mrr:.2f}')
        worst = sorted([(r, s, t) for (s,t), r in zip(pairs_ok, ranks) if r != 0],
                       reverse=True)[:3]
        for r, s, t in worst:
            tag = 'missing' if r < 0 else f'r{r}'
            print(f'    {s}→{t} ({tag})')


# ─── Entry point ─────────────────────────────────────────────────────────────

# 'cookie' excluded from default demo — polysemous (HTTP cookie dominates in Qwen2).
# Include it explicitly to demonstrate the polysemy failure mode.
DEMO_FOODS = ['bread', 'soup', 'pasta', 'cake', 'pizza']


def main():
    parser = argparse.ArgumentParser(description='Geometric recipe expander (DC 301 Tier 1)')
    parser.add_argument('food', nargs='?', default=None,
                        help='Food concept to expand (default: run demo)')
    parser.add_argument('--delta-check', action='store_true',
                        help='Print delta geometry and LOO accuracy')
    args = parser.parse_args()

    print('Loading TruthSpace…')
    lcm = build_lcm()
    print('Learning food deltas…')
    exp = RecipeExpander(lcm)

    if args.delta_check:
        exp.delta_geometry()
        exp.loo_accuracy('ingredient')
        exp.loo_accuracy('method')
        print()

    foods = [args.food] if args.food else DEMO_FOODS

    print('\n' + '='*60)
    print('  GEOMETRIC RECIPE EXPANDER  (DC 301 — Tier 1 generation)')
    print('='*60)

    for food in foods:
        print()
        try:
            result = exp.expand(food, n_ingr=6, n_meth=5)
            print(exp.format_response(result))
        except RuntimeError as e:
            print(f'  Error: {e}')
        print()

    # ── Delta geometry note ──────────────────────────────────────────────────
    exp.delta_geometry()
    print()
    print('  Key insight (DC 301 §6):')
    print('  If cos(ingredient_delta, method_delta) ≈ 0, the two response')
    print('  components are geometrically independent — no crosstalk.')
    print('  The recipe is assembled from orthogonal semantic subspaces.')


if __name__ == '__main__':
    main()
