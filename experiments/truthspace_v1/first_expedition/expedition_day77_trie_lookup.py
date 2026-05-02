#!/usr/bin/env python3
"""
Day 77 — φ-Trie as Generative Lookup Table

Day 76 confirmed: Hamming distance in 8-dim ternary address space is
a perfect monotone predictor of logit cosine similarity
(d=0→0.909, d=8→0.785, 9/9 steps strictly decreasing).

The natural next question: can we USE the ternary trie as a generative
lookup table?

LEAVE-ONE-OUT TEST:
  For each probe token w:
    1. Hide w from the lookup table (LOO setting)
    2. Find all remaining tokens within Hamming radius r = 0, 1, 2, 3, 4
    3. Average their logit distributions (uniform + distance-weighted)
    4. Compare predicted logits to actual logits:
       - logit cosine similarity
       - top-10 / top-50 / top-100 token overlap
       - argmax accuracy (does prediction pick the same top token?)

PREDICTION:
  - Trie lookup at r≤2 outperforms the global mean baseline
  - Prediction quality decreases with Hamming radius
  - Same-leaf neighbors (r=0) give the best predictions
  - Numbers/quantifiers (three/four/five, same leaf, sim=0.993) will
    have near-perfect predictions from each other
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day77_trie_lookup.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

# ── Top-8 decision points (from Day 75/76) ───────────────────────────────────
DECISION_AXES = [
    ("gender_medium",            "Gender (medium, L27)",         27),
    ("comparative_short",        "Comparative (short, L15)",     15),
    ("hypernym_medium",          "Hypernym (medium, L28)",       28),
    ("plural_long",              "Plural (long, L1)",             1),
    ("synonym_short",            "Synonym (short, L28)",         28),
    ("concrete_abstract_medium", "Concrete→Abstract (med, L28)", 28),
    ("past_tense_long",          "Past tense (long, L28)",       28),
    ("antonym_short",            "Antonym (short, L28)",         28),
]

AXIS_PAIRS = {
    "gender_medium": [
        ("The king ruled with great wisdom",    "The queen ruled with great wisdom"),
        ("A man walked through the forest",     "A woman walked through the forest"),
        ("The boy kicked the ball hard",        "The girl kicked the ball hard"),
        ("His brother arrived at the party",    "His sister arrived at the party"),
        ("The father worked to feed family",    "The mother worked to feed family"),
        ("A son was born in the winter",        "A daughter was born in the winter"),
        ("The prince rode across the land",     "The princess rode across the land"),
        ("The actor played a leading role",     "The actress played a leading role"),
    ],
    "comparative_short": [
        ("The fast car",   "The faster car"),
        ("A big dog",      "A bigger dog"),
        ("The cold wind",  "The colder wind"),
        ("A tall tree",    "A taller tree"),
        ("The old house",  "The older house"),
        ("A bright star",  "A brighter star"),
        ("The dark room",  "The darker room"),
        ("A hard rock",    "A harder rock"),
    ],
    "hypernym_medium": [
        ("The dog ran away from danger",    "The animal ran away from danger"),
        ("A rose bloomed in the garden",    "A flower bloomed in the garden"),
        ("The oak crashed in the storm",    "The tree crashed in the storm"),
        ("The car sped past the sign",      "The vehicle sped past the sign"),
        ("The eagle soared above the hill", "The bird soared above the hill"),
        ("The ruby gleamed in the light",   "The gem gleamed in the light"),
        ("The soldier marched into fight",  "The person marched into fight"),
        ("The hammer struck the nail",      "The tool struck the nail"),
    ],
    "plural_long": [
        ("A dog played happily in the open green field",    "Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window", "The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist",    "Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm",   "The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk",          "Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road",   "The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky",     "Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text",   "The words appeared clearly in the printed text"),
    ],
    "synonym_short": [
        ("He is big",         "He is large"),
        ("She is small",      "She is tiny"),
        ("He runs fast",      "He runs quick"),
        ("It is cold",        "It is frigid"),
        ("She is happy",      "She is joyful"),
        ("He spoke loudly",   "He spoke noisily"),
        ("It is hard",        "It is difficult"),
        ("He is old",         "He is aged"),
    ],
    "concrete_abstract_medium": [
        ("The stone is too heavy to lift",  "The burden is too heavy to lift"),
        ("The iron chain has broken now",   "The bond between them has broken"),
        ("The long road leads to the sea",  "The long journey leads to the sea"),
        ("The high wall blocks the view",   "The high barrier blocks the view"),
        ("The flame slowly fades away",     "The hope slowly fades away"),
        ("The strong root grips the soil",  "The strong base grips the earth"),
        ("The bridge connects two banks",   "The bond connects two communities"),
        ("The small key opens the door",    "The small answer opens the path"),
    ],
    "past_tense_long": [
        ("I walk to the market every single morning",       "I walked to the market every single morning"),
        ("She runs through the park after her long work",   "She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house",  "He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden",       "They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days",         "We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend",      "She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire",  "They sang together around the evening campfire"),
    ],
    "antonym_short": [
        ("It is hot",         "It is cold"),
        ("He runs fast",      "He runs slow"),
        ("The light is on",   "The dark is on"),
        ("The news is good",  "The news is bad"),
        ("It is hard",        "It is soft"),
        ("She is happy",      "She is sad"),
        ("He is strong",      "He is weak"),
        ("It is the first",   "It is the last"),
    ],
}

PROBE_TOKENS = [
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
    "tree", "flower", "rock", "stone", "wood", "leaf", "grass", "root",
    "house", "door", "window", "table", "chair", "book", "cup", "key",
    "car", "road", "bridge", "boat", "ship", "plane", "train", "bike",
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start", "stop",
    "think", "know", "see", "hear", "feel", "love", "hate", "want",
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
    "the", "a", "and", "or", "not", "is", "was", "in", "on", "of",
    "to", "from", "with", "for", "he", "she", "it", "they",
    "dogs", "cats", "trees", "faster", "biggest", "running", "walked",
    "quickly", "slowly", "better", "worse", "more", "less", "most",
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "water", "fire", "earth", "air", "sun", "moon", "star", "sky",
    "love", "hate", "truth", "beauty", "freedom", "power",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    "one", "two", "three", "four", "five", "ten", "many", "few",
    "all", "some", "none", "most", "each", "every",
    "共", "的", "在", "了",
]

ENGLISH_SKIP = {"共", "的", "在", "了"}

MAX_RADIUS = 4    # sweep r = 0, 1, 2, 3, 4
TOP_K_LIST = [10, 50, 100]

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def top_k_overlap(pred, actual, k):
    tp = set(np.argsort(pred)[-k:])
    ta = set(np.argsort(actual)[-k:])
    return len(tp & ta) / k

def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def get_h_at_layer(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

def get_logits(word):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].numpy().astype(np.float32)

# ── Build T2 axes ─────────────────────────────────────────────────────────────
print("Building T2 axes ...")
t2_axes = {}
for (ak, label, layer) in DECISION_AXES:
    diffs = []
    for s1, s2 in AXIS_PAIRS[ak]:
        h1 = get_h_at_layer(s1, layer)
        h2 = get_h_at_layer(s2, layer)
        d  = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if diffs:
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        t2_axes[ak] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    else:
        t2_axes[ak] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {label}")
print()

# ── Collect probe token hidden states + logits ────────────────────────────────
print(f"Collecting hidden states for {len(PROBE_TOKENS)} probe tokens ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        hs  = {ak: get_h_at_layer(" " + word.strip(), layer)
               for (ak, _, layer) in DECISION_AXES}
        lv  = get_logits(word)
        tokens_data[word] = {"hs": hs, "logits": lv}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words         = list(tokens_data.keys())
english_words = [w for w in words if w not in ENGLISH_SKIP]
logit_vecs    = {w: tokens_data[w]["logits"] for w in words}
global_mean_logits = np.mean([logit_vecs[w] for w in words], axis=0)
print(f"  Collected {len(words)} tokens ({len(english_words)} English)\n")

# ── Calibrate thresholds + assign ternary addresses ──────────────────────────
classes = {}
for (ak, _, layer) in DECISION_AXES:
    axis  = t2_axes[ak]
    if np.linalg.norm(axis) < 1e-6:
        for w in words: classes.setdefault(w, {})[ak] = "U"
        continue
    projs = np.array([float(np.dot(tokens_data[w]["hs"][ak], axis)) for w in words])
    max_p = np.percentile(projs, 95)
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    for i, w in enumerate(words):
        p = projs[i]
        classes.setdefault(w, {})[ak] = ("H" if p > hi else "L" if p < lo else "U")

axis_keys = [ak for ak, _, _ in DECISION_AXES]
addresses = {w: "".join(classes[w][ak] for ak in axis_keys) for w in words}

n_leaves = len(set(addresses.values()))
print(f"Ternary trie: {n_leaves} leaves among {len(words)} tokens\n")

# ── Leave-one-out prediction ──────────────────────────────────────────────────
print("Running leave-one-out prediction ...")
print()

# Results by radius: r → list of (cosim, top-k overlaps, argmax_match)
r_results  = {r: {"cosim": [], "topk": {k: [] for k in TOP_K_LIST},
                   "argmax_match": [], "n_neighbors": []}
              for r in range(MAX_RADIUS + 1)}
global_results = {"cosim": [], "topk": {k: [] for k in TOP_K_LIST}, "argmax_match": []}

# Precompute all pairwise Hamming distances
n = len(words)
hamm_matrix = np.zeros((n, n), dtype=np.int32)
for i in range(n):
    for j in range(n):
        hamm_matrix[i, j] = hamming(addresses[words[i]], addresses[words[j]])

for i, w in enumerate(words):
    actual = logit_vecs[w]
    actual_top50 = set(np.argsort(actual)[-50:])

    # Global mean baseline (excluding w itself)
    other_logits = np.array([logit_vecs[words[j]] for j in range(n) if j != i])
    global_pred  = np.mean(other_logits, axis=0)
    global_results["cosim"].append(cos_sim(global_pred, actual))
    for k in TOP_K_LIST:
        global_results["topk"][k].append(top_k_overlap(global_pred, actual, k))
    global_results["argmax_match"].append(
        int(np.argmax(global_pred) == np.argmax(actual)))

    # For each radius r: find neighbors within Hamming distance r (excluding w)
    # Use cumulative neighborhood (all j with dist ≤ r)
    for r in range(MAX_RADIUS + 1):
        neighbors = [j for j in range(n) if j != i and hamm_matrix[i, j] <= r]
        if not neighbors:
            r_results[r]["cosim"].append(float("nan"))
            for k in TOP_K_LIST: r_results[r]["topk"][k].append(float("nan"))
            r_results[r]["argmax_match"].append(float("nan"))
            r_results[r]["n_neighbors"].append(0)
            continue

        # Distance-weighted average (exp(-d))
        neighbor_logits = np.array([logit_vecs[words[j]] for j in neighbors])
        weights = np.array([math.exp(-hamm_matrix[i, j]) for j in neighbors],
                           dtype=np.float32)
        weights /= weights.sum()
        pred = neighbor_logits.T @ weights   # (vocab,)

        r_results[r]["cosim"].append(cos_sim(pred, actual))
        for k in TOP_K_LIST:
            r_results[r]["topk"][k].append(top_k_overlap(pred, actual, k))
        r_results[r]["argmax_match"].append(
            int(np.argmax(pred) == np.argmax(actual)))
        r_results[r]["n_neighbors"].append(len(neighbors))

# ── Summary table ─────────────────────────────────────────────────────────────
def safe_mean(lst):
    valid = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(valid)) if valid else float("nan")

print("=" * 72)
print("Leave-one-out prediction: cosine similarity vs Hamming radius")
print("=" * 72)
print(f"  {'radius':>8}  {'cos_sim':>9}  {'top-10':>8}  {'top-50':>8}  "
      f"{'top-100':>8}  {'argmax':>8}  {'n_nbrs':>8}")

global_cos = safe_mean(global_results["cosim"])
global_t10 = safe_mean(global_results["topk"][10])
global_t50 = safe_mean(global_results["topk"][50])
global_t100= safe_mean(global_results["topk"][100])
global_am  = safe_mean(global_results["argmax_match"])
print(f"  {'baseline':>8}  {global_cos:>9.4f}  {global_t10:>8.3f}  "
      f"{global_t50:>8.3f}  {global_t100:>8.3f}  {global_am:>8.3f}  {'(all)':>8}")

for r in range(MAX_RADIUS + 1):
    cos_m = safe_mean(r_results[r]["cosim"])
    t10_m = safe_mean(r_results[r]["topk"][10])
    t50_m = safe_mean(r_results[r]["topk"][50])
    t100_m= safe_mean(r_results[r]["topk"][100])
    am_m  = safe_mean(r_results[r]["argmax_match"])
    nn_m  = safe_mean(r_results[r]["n_neighbors"])
    better = "✓" if cos_m > global_cos else " "
    print(f"  {f'r≤{r}':>8}  {cos_m:>9.4f}{better} {t10_m:>8.3f}  "
          f"{t50_m:>8.3f}  {t100_m:>8.3f}  {am_m:>8.3f}  {nn_m:>8.1f}")
print()

# ── Best-case analysis: tokens WITH same-leaf neighbors ──────────────────────
print("=" * 72)
print("Same-leaf (r=0) tokens: best predictions")
print("=" * 72)
r0_with_nbrs = [(words[i], r_results[0]["cosim"][i], r_results[0]["n_neighbors"][i])
                for i in range(len(words))
                if r_results[0]["n_neighbors"][i] > 0
                and not (isinstance(r_results[0]["cosim"][i], float)
                         and math.isnan(r_results[0]["cosim"][i]))]
r0_with_nbrs.sort(key=lambda x: -x[1])

print(f"  Tokens with same-leaf neighbors: {len(r0_with_nbrs)}/{len(words)}")
print(f"  Mean cosim for same-leaf tokens: "
      f"{float(np.mean([x[1] for x in r0_with_nbrs])):.4f}" if r0_with_nbrs else "  (none)")
print()
print(f"  {'token':>12}  {'cos_sim':>9}  {'n_nbrs':>8}  {'address':>10}")
for w, cs, nn in r0_with_nbrs[:15]:
    print(f"  {w:>12}  {cs:>9.4f}  {nn:>8}  {addresses[w]:>10}")
print()

# ── Worst-case: singleton leaves ──────────────────────────────────────────────
print("=" * 72)
print("Singleton analysis: tokens with NO same-leaf neighbors (r=0)")
print("=" * 72)
singletons = [words[i] for i in range(len(words)) if r_results[0]["n_neighbors"][i] == 0]
print(f"  Singletons: {len(singletons)}/{len(words)}")
# For singletons, compare r=1 vs baseline
singleton_idxs = [words.index(w) for w in singletons]
if singleton_idxs:
    r1_cos_singletons = [r_results[1]["cosim"][i] for i in singleton_idxs
                         if not (isinstance(r_results[1]["cosim"][i], float)
                                 and math.isnan(r_results[1]["cosim"][i]))]
    gl_cos_singletons = [global_results["cosim"][i] for i in singleton_idxs]
    print(f"  Singletons at r≤1: cos_sim={float(np.mean(r1_cos_singletons)):.4f}  "
          f"(vs global baseline {float(np.mean(gl_cos_singletons)):.4f})")
print()

# ── Focused test: number tokens ───────────────────────────────────────────────
number_words = ["one", "two", "three", "four", "five", "ten"]
number_words = [w for w in number_words if w in words]
print("=" * 72)
print("Focal test: number tokens (same leaf expected)")
print("=" * 72)
for w in number_words:
    i = words.index(w)
    addr = addresses[w]
    nbrs = [words[j] for j in range(len(words)) if j != i and hamm_matrix[i, j] == 0]
    cs   = r_results[0]["cosim"][i] if r_results[0]["n_neighbors"][i] > 0 else float("nan")
    print(f"  {w:>8}  {addr}  same-leaf: [{', '.join(nbrs)}]  cos_sim={cs:.4f}")
print()

# ── Cross-leaf semantic families ─────────────────────────────────────────────
print("=" * 72)
print("Semantic families: within-family Hamming distances")
print("=" * 72)
FAMILIES = {
    "numbers":    ["one", "two", "three", "four", "five", "ten", "many", "few"],
    "animals":    ["dog", "cat", "horse", "wolf", "lion", "bear", "deer", "rabbit"],
    "verbs":      ["run", "walk", "swim", "fly", "eat", "sleep", "talk", "write"],
    "adjectives": ["fast", "slow", "big", "small", "hot", "cold", "hard", "soft"],
    "function":   ["the", "a", "and", "or", "not", "is", "was", "in"],
}

for family, members in FAMILIES.items():
    fam = [w for w in members if w in words]
    if len(fam) < 2: continue
    idxs = [words.index(w) for w in fam]
    dists = [hamm_matrix[idxs[i], idxs[j]]
             for i in range(len(idxs)) for j in range(i+1, len(idxs))]
    sims  = [cos_sim(logit_vecs[fam[i]], logit_vecs[fam[j]])
             for i in range(len(fam)) for j in range(i+1, len(fam))]
    print(f"  {family:>12}: mean_hamming={float(np.mean(dists)):.2f}  "
          f"mean_sim={float(np.mean(sims)):.4f}  "
          f"members: {' '.join(fam)}")
print()

# ── Final summary ─────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 77 Summary")
print("=" * 72)

best_r = max(range(MAX_RADIUS + 1), key=lambda r: safe_mean(r_results[r]["cosim"]))
best_cos = safe_mean(r_results[best_r]["cosim"])
print(f"  Best lookup radius: r≤{best_r}  cos_sim={best_cos:.4f}")
print(f"  Baseline (global mean): {global_cos:.4f}")
print(f"  Improvement: {best_cos - global_cos:+.4f}")
print()
print(f"  Day 76 same-leaf direct: 0.9092 (n=34 pairs, not LOO)")
print(f"  Day 77 LOO same-leaf:    {safe_mean(r_results[0]['cosim']):.4f}")
print(f"  Day 77 LOO best (r≤{best_r}):  {best_cos:.4f}")
print()
print("  Monotone prediction quality vs Hamming radius:")
for r in range(MAX_RADIUS + 1):
    print(f"    r≤{r}: {safe_mean(r_results[r]['cosim']):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "by_radius": {
        str(r): {
            "mean_cosim":         safe_mean(r_results[r]["cosim"]),
            "mean_argmax_match":  safe_mean(r_results[r]["argmax_match"]),
            "mean_n_neighbors":   safe_mean(r_results[r]["n_neighbors"]),
            "topk":               {str(k): safe_mean(r_results[r]["topk"][k])
                                   for k in TOP_K_LIST},
        } for r in range(MAX_RADIUS + 1)
    },
    "global_baseline": {
        "mean_cosim":        global_cos,
        "mean_argmax_match": global_am,
        "topk":              {str(k): safe_mean(global_results["topk"][k])
                               for k in TOP_K_LIST},
    },
    "best_radius": best_r,
    "improvement_over_baseline": best_cos - global_cos,
    "n_tokens_with_same_leaf_nbr": len(r0_with_nbrs),
    "n_singletons": len(singletons),
    "addresses": {w: addresses[w] for w in words},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 77 complete.")
