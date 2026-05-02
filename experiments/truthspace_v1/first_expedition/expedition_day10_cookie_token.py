#!/usr/bin/env python3
"""
Expedition Day 10 — Cookie at the Token Level

Day 7 showed that N-body gravity disambiguates ALL polysemous words with just
1 context word. But that was at the CONCEPT LEVEL (mass-weighted centroid of
discrete vocabulary words).

Day 10 tests disambiguation at the TOKEN LEVEL — using actual projection
coordinates rather than vocabulary centroids. The question:

  Given a sentence like "The user deleted the cookie from their browser",
  does the IRD's concept-level disambiguation produce the RIGHT reading of
  "cookie" when we use ALL the sentence words as context gravity?

  And: does the WRONG context ("She baked a cookie for dessert") also
  unambiguously pull "cookie" to the food basin?

Methodology:
  For each sentence, collect ALL content words (skip stopwords).
  Compute the mass-weighted N-body centroid of all context words.
  Compute similarity of this centroid to:
    (a) cookie's food projection neighbourhood
    (b) cookie's tech (browser) projection neighbourhood
  The basin with higher similarity = disambiguation result.

Also test:
  - Does the result CHANGE as we read the sentence left-to-right
    (incremental disambiguation)?
  - At which word does "cookie" commit to one basin?
  - Are there sentences where the disambiguation is WRONG?
  - What is the minimum sentence length for reliable disambiguation?

Test sentences:
  FOOD context:
  1. "She baked a cookie for dessert"
  2. "The chocolate chip cookie crumbled"
  3. "Add the cookies to the baking tray"

  TECH context:
  4. "The browser saved a cookie to track the session"
  5. "Delete your cookies to fix the login problem"
  6. "The server sent a cookie with the HTTP response"

  AMBIGUOUS / HARD:
  7. "She checked the cookie settings"   (cookie=food? or cookie=browser?)
  8. "The cookie monster ate the cookie"  (both food — same basin)
  9. "Fortune cookie says your password expires soon"  (food+tech mix)
  10. "Clear the cookie dough from the counter"  (food, but 'clear' is tech)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

MASS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'expedition_day1_masses.npz')

STOPWORDS = {
    'the','a','an','to','for','from','with','their','this','that','these',
    'those','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall',
    'of','in','on','at','by','as','if','it','its','and','or','but','not',
    'she','he','they','we','i','you','her','him','them','us','me','my','your',
    'his','our','their','they','also','just','very','so','then','when','all',
    'add','get','set','use','can','you','your','says','said',
}

# Basin reference words for cookie disambiguation
FOOD_BASIN    = ['bake','sugar','flour','chocolate','oven','sweet','dessert','recipe',
                 'dough','butter','eat','snack','biscuit','crumble','pastry']
TECH_BASIN    = ['browser','web','internet','session','data','server','privacy',
                 'http','login','track','cache','password','delete','site','request']

TEST_SENTENCES = [
    # (label, sentence, expected_basin)
    ("food-1",   "She baked a cookie for dessert",                        "food"),
    ("food-2",   "The chocolate chip cookie crumbled on the plate",       "food"),
    ("food-3",   "Add the cookies to the baking tray",                    "food"),
    ("tech-1",   "The browser saved a cookie to track the session",       "tech"),
    ("tech-2",   "Delete your cookies to fix the login problem",          "tech"),
    ("tech-3",   "The server sent a cookie with the HTTP response",       "tech"),
    ("ambig-1",  "She checked the cookie settings",                       "ambig"),
    ("ambig-2",  "The cookie monster ate the cookie",                     "food"),
    ("ambig-3",  "Fortune cookie says your password expires soon",        "ambig"),
    ("ambig-4",  "Clear the cookie dough from the counter",              "food"),
]


def get_word_proj(lcm, word):
    try:
        proj, idx = lcm._get_proj(word)
        return proj.astype(np.float64), idx
    except RuntimeError:
        return None, None


def mass_weighted_centroid(lcm, words, M):
    P = lcm.projections.astype(np.float64)
    vecs   = []
    masses = []
    for w in words:
        proj, idx = get_word_proj(lcm, w)
        if proj is None:
            continue
        vecs.append(proj)
        if idx is not None:
            masses.append(float(M[idx]))
        else:
            masses.append(0.15)
    if not vecs:
        return None
    vecs   = np.array(vecs)
    masses = np.array(masses)
    masses /= masses.sum()
    c = (vecs * masses[:, np.newaxis]).sum(axis=0)
    c /= (np.linalg.norm(c) + 1e-20)
    return c


def basin_similarity(centroid, basin_centroid):
    return float(np.dot(centroid, basin_centroid))


def tokenise(sentence):
    import re
    return [w.lower().strip(".,!?;:\"'") for w in sentence.split()
            if w.lower().strip(".,!?;:\"'") not in STOPWORDS
            and len(w.strip(".,!?;:\"'")) > 1]


if __name__ == '__main__':
    print("Loading LCM...")
    lcm  = build_lcm()
    day1 = np.load(MASS_PATH, allow_pickle=True)
    M    = day1['M_binding'].astype(np.float64)
    P    = lcm.projections.astype(np.float64)

    print(f"\n{'='*65}")
    print(f"DAY 10 — Cookie at the Token Level")
    print(f"{'='*65}")

    # Build reference basin centroids
    food_centroid = mass_weighted_centroid(lcm, FOOD_BASIN, M)
    tech_centroid = mass_weighted_centroid(lcm, TECH_BASIN, M)

    print(f"\n  Food basin centroid built from: {FOOD_BASIN[:5]}...")
    print(f"  Tech basin centroid built from: {TECH_BASIN[:5]}...")
    cos_basins = float(np.dot(food_centroid, tech_centroid))
    print(f"  Cosine similarity between basins: {cos_basins:.4f}  "
          f"({'distinct' if cos_basins < 0.5 else 'overlapping'})")

    # Cookie's own projection
    cookie_proj, cookie_idx = get_word_proj(lcm, 'cookie')
    if cookie_idx is not None:
        m_cookie = float(M[cookie_idx])
        c_food0  = float(np.dot(cookie_proj / np.linalg.norm(cookie_proj), food_centroid))
        c_tech0  = float(np.dot(cookie_proj / np.linalg.norm(cookie_proj), tech_centroid))
        print(f"\n  'cookie' M_binding={m_cookie:.4f}")
        print(f"  'cookie' baseline: food_cos={c_food0:.4f}  tech_cos={c_tech0:.4f}  "
              f"default_basin={'food' if c_food0 > c_tech0 else 'tech'}")

    # ── Full-sentence disambiguation ──────────────────────────────────────────
    print(f"\n── Section 1: Full-sentence disambiguation ──────────────────")
    print(f"  {'Label':<12}  {'Expected':<8}  {'Result':<8}  {'Food cos':<10}  "
          f"{'Tech cos':<10}  {'Correct?':<8}  Sentence")
    print("  " + "─" * 85)

    n_correct = 0
    n_total   = 0
    results_table = []

    for label, sentence, expected in TEST_SENTENCES:
        words = tokenise(sentence)
        # Remove 'cookie' itself from context
        ctx_words = [w for w in words if w not in ('cookie', 'cookies')]
        if not ctx_words:
            ctx_words = words

        centroid = mass_weighted_centroid(lcm, ctx_words, M)
        if centroid is None:
            print(f"  {label:<12}  {expected:<8}  (no context words found)")
            continue

        food_cos = basin_similarity(centroid, food_centroid)
        tech_cos = basin_similarity(centroid, tech_centroid)
        result   = 'food' if food_cos > tech_cos else 'tech'

        if expected == 'ambig':
            correct_str = "—"
        else:
            correct_str = "✓" if result == expected else "✗"
            if result == expected:
                n_correct += 1
            n_total += 1

        results_table.append((label, sentence, expected, result, food_cos, tech_cos))
        print(f"  {label:<12}  {expected:<8}  {result:<8}  {food_cos:.4f}      "
              f"{tech_cos:.4f}      {correct_str:<8}  \"{sentence[:50]}\"")

    if n_total > 0:
        print(f"\n  Accuracy on non-ambiguous sentences: {n_correct}/{n_total} "
              f"({100*n_correct/n_total:.1f}%)")

    # ── Incremental disambiguation (left-to-right) ────────────────────────────
    print(f"\n── Section 2: Incremental disambiguation ────────────────────")
    print(f"  (At which word does 'cookie' commit to the correct basin?)\n")

    for label, sentence, expected in TEST_SENTENCES[:6]:
        words = tokenise(sentence)
        ctx_words = [w for w in words if w not in ('cookie', 'cookies')]
        print(f"  [{label}] \"{sentence}\"")
        print(f"  Context words: {ctx_words}")
        if not ctx_words:
            print(f"  (No context words)")
            continue
        committed = None
        for n in range(1, len(ctx_words)+1):
            ctx = ctx_words[:n]
            c   = mass_weighted_centroid(lcm, ctx, M)
            if c is None:
                continue
            food_cos = float(np.dot(c, food_centroid))
            tech_cos = float(np.dot(c, tech_centroid))
            result   = 'food' if food_cos > tech_cos else 'tech'
            gap      = abs(food_cos - tech_cos)
            marker   = ""
            if committed is None and result == expected and gap > 0.02:
                committed = n
                marker = f" ← COMMIT ({ctx[-1]})"
            print(f"    n={n}: ctx={ctx[-1]:<15} food={food_cos:.3f}  tech={tech_cos:.3f}  "
                  f"→{result}{marker}")
        if committed:
            print(f"  Committed at word {committed}/{len(ctx_words)}: '{ctx_words[committed-1]}'")
        print()

    # ── Hard cases: ambiguous sentences ──────────────────────────────────────
    print(f"\n── Section 3: Hard cases analysis ───────────────────────────")
    for label, sentence, expected in TEST_SENTENCES[6:]:
        words = tokenise(sentence)
        ctx_words = [w for w in words if w not in ('cookie', 'cookies')]
        centroid = mass_weighted_centroid(lcm, ctx_words, M)
        if centroid is None:
            continue

        food_cos = basin_similarity(centroid, food_centroid)
        tech_cos = basin_similarity(centroid, tech_centroid)
        result   = 'food' if food_cos > tech_cos else 'tech'
        gap      = abs(food_cos - tech_cos)

        # Identify which words pulled which direction
        word_pulls = []
        for w in ctx_words:
            proj, _ = get_word_proj(lcm, w)
            if proj is None:
                continue
            wn = proj / (np.linalg.norm(proj)+1e-20)
            f  = float(np.dot(wn, food_centroid))
            t  = float(np.dot(wn, tech_centroid))
            word_pulls.append((w, f, t, 'food' if f>t else 'tech'))

        print(f"  [{label}, expected={expected}] \"{sentence}\"")
        print(f"  Result: {result} (food={food_cos:.3f}, tech={tech_cos:.3f}, gap={gap:.3f})")
        print(f"  Per-word basin affinity:")
        for w, f, t, basin in word_pulls:
            print(f"    {w:<15} food={f:.3f}  tech={t:.3f}  → {basin}")
        print()

    # ── Sensitivity: word count vs disambiguation confidence ──────────────────
    print(f"\n── Section 4: How confidence scales with context size ────────")
    sentence = "The browser saved a cookie to track the session data"
    all_ctx  = [w for w in tokenise(sentence) if w not in ('cookie','cookies')]
    print(f"  Sentence: \"{sentence}\"")
    print(f"  Context words: {all_ctx}\n")
    print(f"  {'n_ctx':<6}  {'food_cos':<10}  {'tech_cos':<10}  {'gap':<8}  Basin")
    for n in range(1, len(all_ctx)+1):
        c = mass_weighted_centroid(lcm, all_ctx[:n], M)
        if c is None:
            continue
        food_cos = float(np.dot(c, food_centroid))
        tech_cos = float(np.dot(c, tech_centroid))
        gap      = tech_cos - food_cos
        result   = 'tech' if tech_cos > food_cos else 'food'
        print(f"  {n:<6}  {food_cos:.6f}  {tech_cos:.6f}  {gap:+.4f}  {result}")
