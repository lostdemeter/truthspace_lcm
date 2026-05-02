#!/usr/bin/env python3
"""
Expedition Day 13 — Three-Sense Disambiguation

Day 10 showed 100% accuracy on 2-sense cookie disambiguation. Day 13 tests
words with 3 or 4 well-separated senses using N-body basin gravity.

Test words:
  bank    — financial / river / blood / memory
  cold    — temperature / illness / emotion (unfriendly)
  date    — calendar / fruit / romantic encounter
  light   — illumination / weight / colour (pale)
  fly     — insect / aviation / verb (move fast)
  spring  — season / water source / coiled metal / verb (jump)

For each word:
  1. Build 3-4 reference basin centroids from seed words
  2. Check: are the basins actually distinct (pairwise cosine < 0.5)?
  3. Test disambiguation on sentences designed for each sense
  4. Test "hard" sentences that mix signals from multiple senses
  5. Report which basin wins and the confidence gap between top-2

Also test: incremental disambiguation — at what word does the sentence
commit to one basin in a multi-sense word?
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
    'his','our','said','says','just','so','then','when','all','also','very',
    'after','before','over','under','there','where','what','which','who',
    'been','into','than','some','out','up','its','now','even','only','back',
}

WORD_SENSES = {
    'bank': {
        'financial': ['money','loan','deposit','account','interest','vault',
                      'withdrawal','teller','savings','investment'],
        'river':     ['water','shore','stream','flood','erosion','current',
                      'fishing','mud','reeds','flood'],
        'blood':     ['donation','hospital','plasma','transfusion','donor',
                      'medical','blood','storage','clinic','hemoglobin'],
    },
    'cold': {
        'temperature': ['winter','frost','ice','freeze','shiver','snow',
                        'jacket','warm','heating','celsius'],
        'illness':     ['cough','flu','sneezing','fever','runny','throat',
                        'medication','sick','nose','rest'],
        'emotion':     ['unfriendly','distant','aloof','rejection','hostile',
                        'indifferent','silence','ignored','detached','frosty'],
    },
    'date': {
        'calendar':  ['schedule','appointment','deadline','month','year',
                      'calendar','timestamp','event','agenda','planned'],
        'fruit':     ['palm','sweet','dried','middle','desert','tropical',
                      'sugar','sticky','harvest','medjool'],
        'romantic':  ['dinner','restaurant','romantic','couple','attraction',
                      'flowers','valentines','kiss','love','meet'],
    },
    'light': {
        'illumination': ['lamp','photon','bright','shine','sun','candle',
                         'bulb','beam','glow','shadow'],
        'weight':       ['feather','thin','weightless','float','carry',
                         'portable','hollow','airy','burden','lift'],
        'colour':       ['pale','pastel','blonde','tint','beige','cream',
                         'faded','washed','ivory','wan'],
    },
    'spring': {
        'season':   ['flower','bloom','april','warmer','birds','renewal',
                     'growth','thaw','pollen','rain'],
        'coil':     ['metal','compression','rebound','mechanism','elastic',
                     'bounce','coil','mattress','suspension','tension'],
        'water':    ['freshwater','natural','mineral','underground','fountain',
                     'aquifer','stream','well','pure','source'],
    },
}

TEST_SENTENCES = {
    'bank': [
        ("financial-1", "She deposited her savings at the bank",        "financial"),
        ("financial-2", "The bank raised interest rates again",          "financial"),
        ("river-1",     "We sat on the bank watching the river flow",    "river"),
        ("river-2",     "The bank eroded after the flood",               "river"),
        ("blood-1",     "Please donate at the blood bank this weekend",  "blood"),
        ("hard-1",      "The bank account was frozen",                   "financial"),
        ("hard-2",      "They walked along the bank",                    "river"),
    ],
    'cold': [
        ("temp-1",   "It was cold outside so she wore her coat",         "temperature"),
        ("temp-2",   "The cold snap froze the pipes",                    "temperature"),
        ("illness-1","He caught a cold and stayed home from work",       "illness"),
        ("illness-2","She sneezed all day with her cold",                "illness"),
        ("emot-1",   "His cold response left her feeling rejected",      "emotion"),
        ("emot-2",   "She gave him a cold stare across the table",       "emotion"),
        ("hard-1",   "Cold hands warm heart",                            "temperature"),
    ],
    'date': [
        ("cal-1",  "The date for the meeting is Thursday",               "calendar"),
        ("cal-2",  "Please note the submission date on the form",        "calendar"),
        ("fruit-1","She ate a date from the palm tree",                  "fruit"),
        ("fruit-2","Medjool dates are sweet and chewy",                  "fruit"),
        ("rom-1",  "Their first date was at an Italian restaurant",      "romantic"),
        ("rom-2",  "He nervously asked her out on a date",               "romantic"),
        ("hard-1", "What is the date today",                             "calendar"),
    ],
    'light': [
        ("illu-1", "Turn on the light so we can see",                   "illumination"),
        ("illu-2", "The beam of light crossed the dark room",           "illumination"),
        ("wt-1",   "The bag is surprisingly light for its size",        "weight"),
        ("wt-2",   "The feather was light as air",                      "weight"),
        ("col-1",  "She chose a light blue for the bedroom walls",      "colour"),
        ("col-2",  "The light fabric shimmered in the breeze",          "colour"),
        ("hard-1", "Can you shed some light on this problem",           "illumination"),
    ],
}


def tokenise(sentence):
    return [w.lower().strip(".,!?;:\"'") for w in sentence.split()
            if w.lower().strip(".,!?;:\"'") not in STOPWORDS
            and len(w.strip(".,!?;:\"'")) > 1]


def build_basin(lcm, words, M):
    P = lcm.projections.astype(np.float64)
    vecs, masses = [], []
    for w in words:
        try:
            proj, idx = lcm._get_proj(w)
            vecs.append(proj.astype(np.float64))
            masses.append(float(M[idx]) if idx is not None else 0.15)
        except RuntimeError:
            pass
    if not vecs:
        return None
    vecs = np.array(vecs)
    masses = np.array(masses)
    masses /= masses.sum()
    c = (vecs * masses[:, None]).sum(0)
    c /= (np.linalg.norm(c) + 1e-20)
    return c


def disambiguate(lcm, sentence, target_word, basins, M):
    words = tokenise(sentence)
    ctx   = [w for w in words if w.lower() not in (target_word.lower(),
                                                     target_word.lower()+'s')]
    if not ctx:
        ctx = words
    P = lcm.projections.astype(np.float64)
    vecs, masses = [], []
    for w in ctx:
        try:
            proj, idx = lcm._get_proj(w)
            vecs.append(proj.astype(np.float64))
            masses.append(float(M[idx]) if idx is not None else 0.15)
        except RuntimeError:
            pass
    if not vecs:
        return None, {}

    vecs = np.array(vecs)
    masses = np.array(masses)
    masses /= masses.sum()
    centroid = (vecs * masses[:, None]).sum(0)
    centroid /= (np.linalg.norm(centroid) + 1e-20)

    scores = {name: float(np.dot(centroid, b)) for name, b in basins.items()}
    winner = max(scores, key=scores.get)
    return winner, scores


if __name__ == '__main__':
    print("Loading LCM...")
    lcm  = build_lcm()
    day1 = np.load(MASS_PATH, allow_pickle=True)
    M    = day1['M_binding'].astype(np.float64)

    print(f"\n{'='*65}")
    print(f"DAY 13 — Three-Sense Disambiguation")
    print(f"{'='*65}")

    # ── Section 1: Basin construction and separation ───────────────────────────
    print(f"\n── Section 1: Basin separation matrix ───────────────────────")
    for word, senses in WORD_SENSES.items():
        basins = {name: build_basin(lcm, seeds, M)
                  for name, seeds in senses.items()}
        basins = {k: v for k, v in basins.items() if v is not None}
        sense_names = list(basins.keys())
        print(f"\n  '{word}' — {len(basins)} senses")
        for i, n1 in enumerate(sense_names):
            for j, n2 in enumerate(sense_names):
                if i < j:
                    cos = float(np.dot(basins[n1], basins[n2]))
                    sep = "good" if cos < 0.4 else "moderate" if cos < 0.6 else "WEAK"
                    print(f"    {n1} vs {n2}: cos={cos:.4f}  ({sep})")

        # Word's default basin
        try:
            proj, idx = lcm._get_proj(word)
            proj = proj.astype(np.float64) / np.linalg.norm(proj)
            scores = {n: float(np.dot(proj, b)) for n, b in basins.items()}
            winner = max(scores, key=scores.get)
            scores_str = "  ".join(f"{n}={v:.3f}" for n, v in scores.items())
            print(f"    Default (no context): {scores_str}  → {winner}")
        except RuntimeError:
            pass

    # ── Section 2: Full-sentence disambiguation ────────────────────────────────
    print(f"\n── Section 2: Full-sentence disambiguation ──────────────────")
    total_correct = 0
    total_tests   = 0
    for word, sentence_list in TEST_SENTENCES.items():
        senses = WORD_SENSES[word]
        basins = {name: build_basin(lcm, seeds, M)
                  for name, seeds in senses.items()}
        basins = {k: v for k, v in basins.items() if v is not None}

        print(f"\n  '{word}':")
        print(f"  {'Label':<14}  {'Expected':<14}  {'Result':<14}  {'Gap':<8}  {'Correct?':<8}  Scores")
        print("  " + "─" * 90)
        for label, sentence, expected in sentence_list:
            winner, scores = disambiguate(lcm, sentence, word, basins, M)
            if winner is None:
                print(f"  {label:<14}  {expected:<14}  (no context)")
                continue
            sorted_scores = sorted(scores.values(), reverse=True)
            gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
            correct = winner == expected
            correct_str = "✓" if correct else "✗"
            if expected != 'ambig':
                total_correct += int(correct)
                total_tests   += 1
            scores_str = "  ".join(f"{n[:4]}={v:.3f}" for n, v in sorted(scores.items(), key=lambda x: -x[1]))
            print(f"  {label:<14}  {expected:<14}  {winner:<14}  {gap:.4f}    {correct_str:<8}  {scores_str}")

    print(f"\n  Overall accuracy: {total_correct}/{total_tests} "
          f"({100*total_correct/max(total_tests,1):.1f}%)")

    # ── Section 3: Incremental disambiguation for 3-sense word ────────────────
    print(f"\n── Section 3: Incremental disambiguation — 'cold' ───────────")
    cold_basins = {name: build_basin(lcm, seeds, M)
                   for name, seeds in WORD_SENSES['cold'].items()}
    cold_basins = {k: v for k, v in cold_basins.items() if v is not None}

    for label, sentence, expected in TEST_SENTENCES['cold'][:4]:
        words = tokenise(sentence)
        ctx   = [w for w in words if w != 'cold']
        if not ctx:
            continue
        print(f"\n  [{label}] \"{sentence}\"")
        print(f"  Context: {ctx}")
        print(f"  {'n':<4}  {'word':<15}  " +
              "  ".join(f"{n[:4]:<8}" for n in cold_basins) + "  winner")
        for n in range(1, len(ctx)+1):
            sub = ctx[:n]
            _, scores = disambiguate(lcm, ' '.join(['cold'] + sub), 'cold', cold_basins, M)
            if not scores:
                continue
            winner = max(scores, key=scores.get)
            row = f"  {n:<4}  {sub[-1]:<15}  " + \
                  "  ".join(f"{scores.get(nm, 0):.4f}  " for nm in cold_basins) + f"  {winner}"
            print(row)

    # ── Section 4: Minimum context needed for 3-sense confidence ─────────────
    print(f"\n── Section 4: Minimum context for 3-sense confidence ────────")
    THRESHOLD = 0.05   # gap between top-2 basins must exceed this
    for word, sentence_list in TEST_SENTENCES.items():
        senses = WORD_SENSES[word]
        basins = {name: build_basin(lcm, seeds, M)
                  for name, seeds in senses.items()}
        basins = {k: v for k, v in basins.items() if v is not None}
        print(f"\n  '{word}':")
        for label, sentence, expected in sentence_list[:4]:
            words = tokenise(sentence)
            ctx   = [w for w in words if w.lower() != word.lower()]
            commit_n = None
            for n in range(1, len(ctx)+1):
                _, scores = disambiguate(lcm, ' '.join([word] + ctx[:n]), word, basins, M)
                if not scores:
                    continue
                sv = sorted(scores.values(), reverse=True)
                gap = sv[0] - sv[1] if len(sv) > 1 else 0
                winner = max(scores, key=scores.get)
                if gap > THRESHOLD and winner == expected and commit_n is None:
                    commit_n = n
            print(f"    [{label}] commit at n={commit_n}/{len(ctx)}  "
                  f"expected={expected}  sentence=\"{sentence[:55]}\"")
