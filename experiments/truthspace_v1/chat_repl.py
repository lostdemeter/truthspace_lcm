#!/usr/bin/env python3
"""
chat_repl.py — TruthSpace Geometric Chat REPL

No LLM weights, no softmax, no attention matrices.
Pure IRD projections + φ-boost retrieval via DeltaLibrary.

Usage:
    python chat_repl.py           # interactive REPL
    python chat_repl.py --demo    # run canned demo queries and exit
"""

import sys, os, argparse, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import DeltaLibrary, build_lcm
from recipe_expander import RecipeExpander

# ─── Response templates ───────────────────────────────────────────────────────

TEMPLATES = {
    'capital_of':       'The capital of {SRC} is {ANS}.',
    'city_country':     '{SRC} is in {ANS}.',
    'country_language': 'The language spoken in {SRC} is {ANS}.',
    'country_currency': 'The currency of {SRC} is {ANS}.',
    'male_female':      'The female form of "{src}" is "{ans}".',
    'antonym':          'The opposite of "{src}" is "{ans}".',
    'antonym_speed':    'The speed-opposite of "{src}" is "{ans}".',
    'antonym_size':     'The size-opposite of "{src}" is "{ans}".',
    'antonym_valence':  'The opposite of "{src}" is "{ans}".',
    'comparative':      'The comparative of "{src}" is "{ans}".',
    'past_tense':       'The past tense of "{src}" is "{ans}".',
}

HELP_TEXT = """\
TruthSpace Geometric Chat — pure geometry, no LLM weights

RELATIONSHIPS & EXAMPLE QUERIES
  capital_of       What is the capital of France?
  capital_of       Capital of Norway?
  city_country     What country is Paris in?
  city_country     Which country is Tokyo located in?
  country_language What language is spoken in Germany?
  country_language Language of Japan?
  country_currency Currency of Japan?  (low confidence — see note)
  male_female      Female version of king?
  male_female      What is the female equivalent of hero?
  comparative      Comparative form of big?
  comparative      Compare: old
  past_tense       Past tense of run?
  past_tense       Past: eat
  antonym_speed    Opposite of fast?
  antonym_size     Opposite of big?
  antonym_valence  Opposite of good?

MULTI-HOP (chained relationships)
  Language of the capital of France?
  What language is spoken in the capital of Japan?
  Currency of the capital of... (experimental)

RECIPE EXPANDER  (Tier 1 generation — DC 301)
  Recipe for bread?
  Ingredients for soup?
  How do I cook pasta?
  Give me a cake recipe
  Note: 'cookie' is polysemous (HTTP cookie) — system warns automatically.

DEBUG MODE
  Prefix any query with '!' to reveal routing internals
  Example:  !capital of Norway

COMMANDS
  help / ?          show this text
  relations         list all loaded relationships with LOO accuracy
  exit / quit / q   exit

NOTE: currency answers require an 'is_currency' seed axis not yet in
Phase-1b.  Geographic relationships (capital, language, city_country)
and morphological ones (comparative) are the most reliable.\
"""

# ─── Stopwords that are never mistaken for source entities ───────────────────

STOPS = {
    'what', 'is', 'the', 'of', 'do', 'does', 'which', 'in', 'a', 'an', 'are',
    'speak', 'spoken', 'there', 'version', 'equivalent', 'word', 'for', 'form',
    'tense', 'comparative', 'opposite', 'female', 'feminine', 'past', 'country',
    'nation', 'language', 'capital', 'city', 'currency', 'compare', 'compare',
    'located', 'its', 'where', 'how', 'who', 'when', 'was', 'were', 'been',
    'use', 'uses', 'used', 'more', 'most', 'than', 'and', 'or', 'but',
    'recipe', 'ingredients', 'ingredient', 'tell', 'give', 'me', 'us', 'please',
    'make', 'cook', 'prepare', 'bake', 'need', 'needs', 'go', 'into', 'used',
}

# Keywords that trigger the recipe expander path
_RECIPE_KWS = {'recipe', 'ingredients', 'ingredient', 'cook', 'prepare', 'make', 'bake'}

# ─── Chained query patterns ───────────────────────────────────────────────────
# (inner_kws, outer_kws, inner_rel, outer_rel)

CHAIN_PATTERNS = [
    ({'capital', 'city'},   {'language', 'spoken', 'tongue', 'speak'},
     'capital_of', 'country_language', True),
    ({'capital', 'city'},   {'currency', 'money', 'coin', 'cash'},
     'capital_of', 'country_currency', True),
    ({'capital', 'city'},   {'female', 'feminine', 'woman'},
     'capital_of', 'male_female', False),
]

# ─── Antonym source→sub-delta lookup ─────────────────────────────────────────
# Map known source words to the best sub-delta for that word.

ANTONYM_DOMAIN = {}   # populated at runtime from library pairs


def _build_antonym_domain(lib):
    for rel in ('antonym_speed', 'antonym_size', 'antonym_valence'):
        if rel not in lib:
            continue
        for s, t in lib._pairs[rel]:
            ANTONYM_DOMAIN[s.lower()] = rel
            ANTONYM_DOMAIN[t.lower()] = rel


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fmt(template_key, src, ans):
    tmpl = TEMPLATES.get(template_key, '{SRC} → {ANS}')
    # Normalize: prevent all-caps vocabulary tokens (BAD, SMALL) leaking through.
    # capitalize() = first letter upper, rest lower (handles PARIS→Paris, bad→Bad).
    return tmpl.format(
        src=src.lower(), ans=ans.lower(),
        SRC=src.capitalize(), ANS=ans.capitalize(),
    )


def _find_source(words, all_sources, lib):
    """Return the first word in *words* that is a known source entity."""
    for w in words:
        if w in all_sources:
            return w
    # Fallback: any non-stop word found in the vocabulary
    for w in words:
        if w not in STOPS and lib._word_ok(w):
            return w
    return None


def _route_antonym(source, lib):
    """Pick the most appropriate antonym sub-delta for *source*."""
    sub = ANTONYM_DOMAIN.get(source.lower())
    if sub and sub in lib:
        return sub
    # No known domain: score all sub-deltas, return best
    best_rel, best_score = 'antonym_valence', -9e9
    for rel in ('antonym_speed', 'antonym_size', 'antonym_valence'):
        if rel not in lib:
            continue
        try:
            sp, _ = lib.lcm._get_proj(source)
            d = lib._delta[rel].astype(np.float64)
            sp = sp.astype(np.float64)
            n_sp = np.linalg.norm(sp); n_d = np.linalg.norm(d)
            score = float(np.dot(sp, d)) / (n_sp * n_d + 1e-20)
            if abs(score) > best_score:
                best_score = abs(score)
                best_rel = rel
        except RuntimeError:
            pass
    return best_rel


# ─── Single-hop answer ────────────────────────────────────────────────────────

def answer_single(lib, words, all_sources, debug):
    """Route a query to one relationship and return a formatted answer line."""
    rel = lib.route(words)

    # Special case: antonym queries — pick domain-specific sub-delta
    antonym_keywords = {'opposite', 'antonym', 'contrary', 'reverse', 'not',
                        'speed', 'size', 'valence'}
    if rel in ('antonym', None) and antonym_keywords & set(words):
        source = _find_source(words, all_sources, lib)
        if source:
            rel = _route_antonym(source, lib)
        else:
            rel = 'antonym_valence'  # last resort

    if rel is None:
        return None, None, None   # signal: no route found

    source = _find_source(words, all_sources, lib)
    if source is None:
        return rel, None, None    # signal: no source found

    results = lib.answer(source, rel, k=5)
    return rel, source, results


# ─── Chained (multi-hop) answer ───────────────────────────────────────────────

def answer_chain(lib, words, all_sources, inner_rel, outer_rel, debug,
                 outer_on_source=False):
    source = _find_source(words, all_sources, lib)
    if source is None:
        return None, None, None, None

    r1 = lib.answer(source, inner_rel, k=3)
    if not r1:
        return inner_rel, source, None, None

    intermediate = r1[0][0]
    # outer_on_source: apply outer_rel to the original country, not the city.
    # This avoids out-of-distribution queries like Paris → country_language.
    outer_src = source if outer_on_source else intermediate
    r2 = lib.answer(outer_src, outer_rel, k=5)
    final_results = r2 if r2 else []

    if debug:
        arrow = f'{source}→{outer_src}' if outer_on_source else outer_src
        print(f'  [chain: {source} →[{inner_rel}]→ {intermediate} '
              f'→[{outer_rel} on {outer_src}]→ '
              f'{final_results[0][0] if final_results else "?"}]')

    return outer_rel, intermediate, source, final_results


# ─── REPL ─────────────────────────────────────────────────────────────────────

def run_repl(lib, expander=None, demo_queries=None):
    all_sources = set()
    for name in lib._rel:
        for s, _ in lib._pairs[name]:
            all_sources.add(s.lower())

    _build_antonym_domain(lib)

    banner = (
        '\n'
        '  ╔══════════════════════════════════════════════════╗\n'
        '  ║  TruthSpace Geometric Chat                       ║\n'
        '  ║  No LLM · No softmax · Pure φ-geometry           ║\n'
        '  ║  type  help  for examples   exit  to quit        ║\n'
        '  ╚══════════════════════════════════════════════════╝\n'
    )
    print(banner)

    def process(line, local_debug=False):
        if not line.strip():
            return
        low = line.lower().strip()

        if low in ('exit', 'quit', 'q'):
            return 'EXIT'
        if low in ('help', '?', 'h'):
            print(HELP_TEXT); return
        if low == 'relations':
            print(f'\n  {"relationship":<20s} {"pairs":>5}  {"top1":>5}  {"top3":>5}  MRR')
            print('  ' + '─' * 50)
            for name in lib._rel:
                vp = lib._pairs[name]
                desc = lib._rel[name]['description']
                print(f'  {name:<20s} {len(vp):>5}   (run lib.report() for full LOO)')
            print()
            return

        local_debug_flag = local_debug
        if line.startswith('!'):
            line = line[1:].strip()
            local_debug_flag = True

        words = re.sub(r'[^\w\s]', ' ', line.lower()).split()

        # ── Recipe expansion (Tier 1 generation — DC 301) ────────────────────
        if expander and _RECIPE_KWS & set(words):
            food = next((w for w in words if w not in STOPS
                         and len(w) > 2 and expander._word_ok(w)), None)
            if food:
                ctx = [w for w in words if w != food and w not in STOPS
                       and len(w) > 2]
                if local_debug_flag:
                    print(f'  [recipe expander: food="{food}" context={ctx}]')
                try:
                    result = expander.expand(food, n_ingr=6, n_meth=4,
                                             context_words=ctx or None)
                    print(expander.format_response(result))
                except RuntimeError as e:
                    print(f'  {e}')
                return
            else:
                print("  Could not identify a food concept in that query.")
                return

        # ── Multi-hop detection ──────────────────────────────────────────────
        for inner_kws, outer_kws, inner_rel, outer_rel, outer_on_src in CHAIN_PATTERNS:
            if inner_kws & set(words) and outer_kws & set(words):
                if local_debug_flag:
                    print(f'  [multi-hop detected: {inner_rel} → {outer_rel}'
                          f'{" (outer on source)" if outer_on_src else ""}]')
                out_rel, via, src, results = answer_chain(
                    lib, words, all_sources, inner_rel, outer_rel,
                    local_debug_flag, outer_on_source=outer_on_src)
                if results:
                    ans   = results[0][0]
                    alts  = [w.capitalize() for w, _ in results[1:3]
                             if w.lower() != ans.lower()]
                    resp  = _fmt(out_rel, via, ans)
                    alt_s = f'  (also: {", ".join(alts)})' if alts else ''
                    print(f'  [via {src} → {via}]  {resp}{alt_s}')
                elif via:
                    print(f'  No result for second step on "{via}".')
                else:
                    print(f'  Could not identify the subject.')
                return

        # ── Single-hop ───────────────────────────────────────────────────────
        rel, source, results = answer_single(lib, words, all_sources,
                                             local_debug_flag)
        # Context words = non-stop query words other than source (DC 302)
        context_words = [w for w in words if source and w != source
                         and w not in STOPS and len(w) > 2]
        if results and context_words:
            # Re-run with context gravity if we got initial results
            try:
                results = lib.answer(source, rel, k=5,
                                     context_words=context_words)
            except Exception:
                pass  # fall back to uncorrected results
        if rel is None:
            print("  Not sure how to answer that.  (type 'help' for examples)")
            return
        if source is None:
            print(f'  Routed to [{rel}] but could not identify the subject.')
            return
        if not results:
            print(f'  [{rel}] No answer found for "{source}".')
            return

        if local_debug_flag:
            print(f'  [routing: "{source}" → {rel}]')
            for i, (w, s) in enumerate(results[:5]):
                print(f'  [{i}] {w:<20s} score={s:.4f}')

        ans  = results[0][0]
        alts = [w.capitalize() for w, _ in results[1:3]
                if w.lower() != ans.lower()]
        resp = _fmt(rel, source, ans)
        alt_s = f'  (also: {", ".join(alts)})' if alts else ''
        print(f'  {resp}{alt_s}')

    # ── Demo mode: run canned queries and exit ───────────────────────────────
    if demo_queries:
        for q in demo_queries:
            print(f'> {q}')
            sig = process(q)
            if sig == 'EXIT':
                break
            print()
        return

    # ── Interactive loop ─────────────────────────────────────────────────────
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n  Bye.')
            break
        sig = process(line)
        if sig == 'EXIT':
            print('  Bye.')
            break


# ─── Entry point ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # ── Geographic ───────────────────────────────────────────────────────────
    'What is the capital of France?',
    'Capital of Norway?',
    'What country is Paris in?',
    'Which country is Tokyo located in?',
    'What language is spoken in Germany?',
    'Language of Japan?',
    'Currency of Japan?',
    'Currency of UK?',
    # ── Gender ───────────────────────────────────────────────────────────────
    'Female version of king?',
    'What is the female equivalent of hero?',
    'Female of husband?',
    # ── Morphological ────────────────────────────────────────────────────────
    'Comparative form of big?',
    'Compare: fast',
    'Past tense of eat?',
    'Past tense of think?',
    # ── Antonyms ─────────────────────────────────────────────────────────────
    'Opposite of fast?',
    'Opposite of big?',
    'Opposite of good?',
    # ── Multi-hop ────────────────────────────────────────────────────────────
    'What language is spoken in the capital of France?',
    'Language of the capital of Japan?',
    'Language of the capital of Norway?',
    # ── Recipe expander (Tier 1 generation) ──────────────────────────────────
    'Recipe for bread?',
    'Ingredients for soup?',
    'How do I cook pasta?',
    'Give me a cake recipe',
    # ── Debug mode ───────────────────────────────────────────────────────────
    '!Capital of Germany?',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TruthSpace Geometric Chat REPL')
    parser.add_argument('--demo', action='store_true',
                        help='Run canned demo queries and exit')
    args = parser.parse_args()

    print('Loading TruthSpace…')
    lcm = build_lcm()
    print('Learning delta library…')
    lib = DeltaLibrary(lcm)
    print(f'  {len(lib._rel)} relationships loaded: {", ".join(lib._rel)}\n')
    print('Building recipe expander…')
    expander = RecipeExpander(lcm)
    print()

    run_repl(lib, expander=expander,
             demo_queries=DEMO_QUERIES if args.demo else None)
