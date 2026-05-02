#!/usr/bin/env python3
"""
Expedition Day 7 — Escape Velocity

The escape velocity of a polysemous word is the minimum number of context
words needed to pull it out of its default semantic basin into a specific
target basin.

Gravitational metaphor:
  A polysemous word sits near the boundary between two or more semantic basins.
  Without context, it falls toward the lowest-energy basin (most common usage).
  Context words act as additional masses that shift the centroid of the
  combined system. The escape velocity is how many context masses you need
  to add before the centroid crosses into the target basin.

We test with words that have well-known polysemies:
  - "bank":   finance basin vs. river basin
  - "bat":    sports basin vs. animal basin
  - "crane":  bird basin vs. machine basin
  - "python": snake basin vs. programming language basin
  - "cookie": food basin vs. browser cookie basin
  - "spring": season basin vs. mechanical spring basin
  - "light":  weight basin vs. photon basin
  - "bark":   dog sound basin vs. tree bark basin
  - "palm":   tree basin vs. hand basin
  - "iron":   metal basin vs. clothes iron basin

For each word:
  1. Compute its uncorrected projection (default basin estimate)
  2. Define a target context set (e.g., for "bank" (finance): ["money", "loan", "interest", "account"])
  3. Compute N-body centroid with 1, 2, 3, ... N context words
  4. Find the minimum N where the centroid crosses from default to target basin
     (measured as: cosine similarity to target centroid > cosine to default centroid)
  5. Record: word, M_binding, default basin, target basin, escape_velocity

Prediction from Day 4:
  Escape velocity is inversely correlated with M_binding —
  low-binding (polysemous) words require more context to escape.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

MASS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'expedition_day1_masses.npz')

# Polysemous test words and their two basins
# Format: (word, default_context_words, target_context_words)
POLYSEMY_TESTS = {
    'bank': (
        ['river', 'water', 'stream', 'shore', 'mud', 'fish', 'bank', 'sandy', 'slope'],
        ['money', 'loan', 'interest', 'account', 'deposit', 'credit', 'savings', 'finance'],
    ),
    'bat': (
        ['baseball', 'hit', 'swing', 'player', 'sport', 'game', 'pitch', 'wood'],
        ['fly', 'nocturnal', 'cave', 'wing', 'mammal', 'echolocation', 'night'],
    ),
    'crane': (
        ['construction', 'lift', 'metal', 'tower', 'cable', 'machine', 'heavy'],
        ['bird', 'fly', 'feather', 'nest', 'migrate', 'wing', 'flock'],
    ),
    'python': (
        ['snake', 'reptile', 'constrict', 'jungle', 'scales', 'bite', 'hiss'],
        ['code', 'programming', 'script', 'function', 'library', 'import', 'data'],
    ),
    'cookie': (
        ['bake', 'sugar', 'flour', 'chocolate', 'oven', 'sweet', 'dessert'],
        ['browser', 'web', 'internet', 'session', 'data', 'server', 'privacy'],
    ),
    'spring': (
        ['season', 'flower', 'warm', 'bloom', 'april', 'rain', 'green'],
        ['coil', 'metal', 'compress', 'elastic', 'bounce', 'mechanical'],
    ),
    'light': (
        ['bright', 'sun', 'lamp', 'shine', 'glow', 'photon', 'illumination'],
        ['weight', 'thin', 'feather', 'float', 'soft', 'delicate'],
    ),
    'bark': (
        ['dog', 'loud', 'sound', 'howl', 'growl', 'noise'],
        ['tree', 'wood', 'trunk', 'rough', 'peel', 'outer', 'layer'],
    ),
    'palm': (
        ['tree', 'tropical', 'coconut', 'beach', 'shade', 'leaves'],
        ['hand', 'finger', 'touch', 'grip', 'press', 'flat'],
    ),
    'iron': (
        ['metal', 'steel', 'hard', 'element', 'ore', 'rust', 'magnetic'],
        ['clothes', 'press', 'wrinkle', 'steam', 'shirt', 'flat', 'fabric'],
    ),
}

MAX_CONTEXT = 8   # max context words to try


def n_body_centroid(lcm, words):
    """
    Compute mass-weighted centroid of a set of words.
    Uses M_binding as mass weight.
    Returns normalised centroid projection in 500-dim space.
    """
    P = lcm.projections.astype(np.float64)
    day1 = np.load(MASS_PATH, allow_pickle=True)
    M = day1['M_binding'].astype(np.float64)

    vecs  = []
    masses = []
    for w in words:
        try:
            proj, idx = lcm._get_proj(w)
            vecs.append(proj.astype(np.float64))
            if idx is not None:
                masses.append(float(M[idx]))
            else:
                masses.append(0.2)   # default mass for out-of-vocab
        except RuntimeError:
            pass
    if not vecs:
        return None
    vecs   = np.array(vecs)
    masses = np.array(masses)
    masses /= masses.sum()
    centroid = (vecs * masses[:, np.newaxis]).sum(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-20)
    return centroid


def find_escape_velocity(lcm, word, default_ctx, target_ctx, max_n=MAX_CONTEXT):
    """
    Return (escape_velocity, default_cos_series, target_cos_series).
    Escape velocity = minimum N context words from target_ctx such that
    centroid(word + N_target) is more similar to target_centroid than to default_centroid.
    """
    P = lcm.projections.astype(np.float64)
    day1 = np.load(MASS_PATH, allow_pickle=True)
    M = day1['M_binding'].astype(np.float64)

    # Reference centroids for each basin
    default_centroid = n_body_centroid(lcm, default_ctx[:5])
    target_centroid  = n_body_centroid(lcm, target_ctx[:5])
    if default_centroid is None or target_centroid is None:
        return None, [], []

    # Word's own projection
    try:
        word_proj, widx = lcm._get_proj(word)
        word_proj = word_proj.astype(np.float64)
    except RuntimeError:
        return None, [], []

    word_mass = float(M[widx]) if widx is not None else 0.2

    default_cos_series = []
    target_cos_series  = []
    escape_v = None

    # Gradually add target context words
    accumulated = [word]
    for n_ctx in range(0, min(max_n, len(target_ctx)) + 1):
        curr_ctx = accumulated.copy()

        centroid = n_body_centroid(lcm, curr_ctx)
        if centroid is None:
            continue

        cos_to_default = float(np.dot(centroid, default_centroid))
        cos_to_target  = float(np.dot(centroid, target_centroid))

        default_cos_series.append(cos_to_default)
        target_cos_series.append(cos_to_target)

        if escape_v is None and cos_to_target > cos_to_default and n_ctx > 0:
            escape_v = n_ctx

        if n_ctx < len(target_ctx):
            accumulated.append(target_ctx[n_ctx])

    return escape_v, default_cos_series, target_cos_series


if __name__ == '__main__':
    print("Loading LCM...")
    lcm  = build_lcm()
    day1 = np.load(MASS_PATH, allow_pickle=True)
    M    = day1['M_binding'].astype(np.float64)

    print(f"\n{'='*65}")
    print(f"DAY 7 — Escape Velocity of Polysemous Words")
    print(f"{'='*65}")
    print(f"\n  Prediction: low M_binding → high escape velocity")
    print(f"  (polysemous words need more context to pick a basin)\n")

    results = []

    for word, (default_ctx, target_ctx) in POLYSEMY_TESTS.items():
        ev, def_series, tgt_series = find_escape_velocity(
            lcm, word, default_ctx, target_ctx)

        # Get M_binding for word
        idx = lcm.word_set.get(word.lower())
        m_bind = float(M[idx]) if idx is not None else float('nan')

        results.append((word, m_bind, ev, def_series, tgt_series))

        ev_str = str(ev) if ev is not None else "≥8 (never)"
        print(f"  {word:<12s}  M={m_bind:.4f}  escape_v={ev_str}")
        if def_series and tgt_series:
            n = min(len(def_series), len(tgt_series), 5)
            pairs = [f"n{i}:({def_series[i]:.3f},{tgt_series[i]:.3f})" for i in range(n)]
            print(f"    cos(def,tgt) @ n=0..{n-1}: {' | '.join(pairs)}")

    # ── Correlation M_binding vs escape_velocity ──────────────────────────────
    print(f"\n── Correlation: M_binding vs escape_velocity ────────────────")
    valid = [(m, ev) for (w, m, ev, _, _) in results if ev is not None and not np.isnan(m)]
    never = [(w, m) for (w, m, ev, _, _) in results if ev is None]

    if valid:
        ms  = np.array([m for m, _ in valid])
        evs = np.array([ev for _, ev in valid], dtype=float)
        corr = float(np.corrcoef(ms, evs)[0, 1])
        print(f"  n valid (escaped):  {len(valid)}")
        print(f"  n never escaped:    {len(never)}")
        print(f"  corr(M, escape_v):  {corr:+.4f}")
        if corr < -0.3:
            print(f"  VERDICT: M_binding DOES predict escape velocity ✓ (negative corr)")
        elif abs(corr) < 0.15:
            print(f"  VERDICT: No clear correlation — context volume matters more than mass")
        else:
            print(f"  VERDICT: Positive correlation (unexpected — more mass = harder to escape)")

    # ── Words that never escaped ───────────────────────────────────────────────
    if never:
        print(f"\n  Words that never escaped (stuck in default basin after {MAX_CONTEXT} context words):")
        for w, m in never:
            print(f"    {w:<12s}  M={m:.4f}")

    # ── Sorted table ──────────────────────────────────────────────────────────
    print(f"\n── Full results sorted by escape velocity ───────────────────")
    print(f"  {'Word':<12s}  {'M_binding':<12s}  {'Escape V':<10s}  cos_gap at escape")
    results_sorted = sorted(results, key=lambda x: (x[2] if x[2] is not None else 999))
    for word, m, ev, def_s, tgt_s in results_sorted:
        ev_str = str(ev) if ev is not None else "≥8"
        if ev is not None and ev < len(def_s) and ev < len(tgt_s):
            gap = tgt_s[ev] - def_s[ev]
            gap_str = f"{gap:+.4f}"
        else:
            gap_str = "—"
        print(f"  {word:<12s}  {m:.4f}        {ev_str:<10s}  {gap_str}")

    # ── Escape velocity across context sizes ─────────────────────────────────
    print(f"\n── Context accumulation curves ─────────────────────────────")
    print(f"  (default_cos, target_cos) as context grows n=0..{min(5,MAX_CONTEXT)}")
    for word, m, ev, def_s, tgt_s in results_sorted[:6]:
        n = min(len(def_s), len(tgt_s), 6)
        print(f"\n  {word} (M={m:.3f}, ev={ev}):")
        for i in range(n):
            marker = " ← ESCAPE" if ev is not None and i == ev else ""
            print(f"    n={i}: default={def_s[i]:.3f}  target={tgt_s[i]:.3f}  "
                  f"gap={tgt_s[i]-def_s[i]:+.3f}{marker}")

    # ── Escape velocity vs M_binding scatter ──────────────────────────────────
    print(f"\n── M_binding groups and escape velocity ─────────────────────")
    high_m = [(w, m, ev) for w, m, ev, _, _ in results if m > 0.20 and ev is not None]
    low_m  = [(w, m, ev) for w, m, ev, _, _ in results if m <= 0.20 and ev is not None]
    if high_m:
        print(f"  High M (>0.20, n={len(high_m)}):  mean ev = {np.mean([ev for _,_,ev in high_m]):.1f}")
        for w,m,ev in high_m: print(f"    {w}: M={m:.3f} ev={ev}")
    if low_m:
        print(f"  Low M (≤0.20, n={len(low_m)}):   mean ev = {np.mean([ev for _,_,ev in low_m]):.1f}")
        for w,m,ev in low_m: print(f"    {w}: M={m:.3f} ev={ev}")
