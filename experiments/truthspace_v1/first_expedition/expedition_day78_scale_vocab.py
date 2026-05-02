#!/usr/bin/env python3
"""
Day 78 — Scale to 500 Words: Trie Density and Metric Robustness

Day 77: 78% of 164 probe tokens are singletons (no same-leaf neighbor).
The trie with 164 tokens is too sparse to fill the 3^8 = 6561 possible
ternary leaves. We need ~5000 tokens to fill half the address space.

This experiment scales to ~500 common English words to test:
  1. Does the ternary metric property hold at 3× vocabulary size?
  2. Does trie density (leaves populated / pairs in same leaf) increase?
  3. Does LOO prediction quality improve with more tokens in the lookup?
  4. Which semantic word families cluster tightest in ternary space?

EFFICIENCY: Single forward pass per token, extracting ALL 4 required
layers (L1, L15, L27, L28) at once. 4× faster than Day 76/77.

PREDICTION:
  - Metric property still holds (Hamming ↓ monotonically with cos_sim)
  - Trie density increases: same-leaf neighbor rate > 22% (was Day 77)
  - LOO improvement > +0.0161 (was Day 77) due to denser leaves
  - Word families (numbers, animals, color, body-parts) cluster at Hamming ≤ 2
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day78_scale_vocab.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

REQUIRED_LAYERS = [1, 15, 27, 28]    # only extract these 4

DECISION_AXES = [
    ("gender_medium",            27),
    ("comparative_short",        15),
    ("hypernym_medium",          28),
    ("plural_long",               1),
    ("synonym_short",            28),
    ("concrete_abstract_medium", 28),
    ("past_tense_long",          28),
    ("antonym_short",            28),
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

# ── ~500 common English words across semantic categories ─────────────────────
PROBE_TOKENS = [
    # ---- ANIMALS (50) ----
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
    "pig", "sheep", "goat", "duck", "hen", "crow", "owl",
    "turtle", "lizard", "crab", "lobster", "octopus", "beetle",
    "butterfly", "worm", "fly", "mosquito", "cricket", "spider",
    "salmon", "tuna", "herring", "sparrow", "robin", "finch", "parrot",
    # ---- PLANTS & NATURE (30) ----
    "tree", "flower", "rock", "stone", "wood", "leaf", "grass", "root",
    "river", "mountain", "ocean", "forest", "desert", "cloud", "rain",
    "snow", "wind", "sun", "moon", "star", "sky", "earth", "soil",
    "seed", "branch", "bark", "thorn", "moss", "mushroom", "coral",
    # ---- OBJECTS (40) ----
    "house", "door", "window", "table", "chair", "book", "cup", "key",
    "car", "road", "bridge", "boat", "ship", "plane", "train", "bike",
    "knife", "fork", "spoon", "plate", "bowl", "glass", "bottle", "box",
    "bag", "rope", "wire", "nail", "hammer", "wheel", "clock", "lamp",
    "pen", "paper", "cloth", "thread", "button", "ring", "coin", "mirror",
    # ---- BODY PARTS (20) ----
    "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg",
    "head", "heart", "blood", "bone", "skin", "hair", "finger", "toe",
    "back", "chest", "neck", "shoulder",
    # ---- VERBS — actions (40) ----
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start", "stop",
    "think", "know", "see", "hear", "feel", "love", "hate", "want",
    "give", "take", "make", "find", "lose", "push", "pull", "turn",
    "move", "go", "come", "fall", "rise", "grow", "kill", "help",
    # ---- VERBS — inflected (10) ----
    "ran", "walked", "jumped", "flew", "ate", "saw", "heard", "broke",
    "built", "wrote",
    # ---- ADJECTIVES (40) ----
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
    "good", "bad", "right", "wrong", "high", "low", "long", "short",
    "wide", "narrow", "deep", "shallow", "thick", "thin", "heavy", "light",
    "clean", "dirty", "sweet", "bitter", "sharp", "dull", "loud", "quiet",
    # ---- COMPARATIVE / SUPERLATIVE (10) ----
    "faster", "slower", "bigger", "smaller", "better", "worse",
    "biggest", "smallest", "best", "worst",
    # ---- ADVERBS (10) ----
    "quickly", "slowly", "often", "never", "always", "very", "quite",
    "really", "just", "still",
    # ---- FUNCTION WORDS (30) ----
    "the", "a", "and", "or", "not", "is", "was", "in", "on", "of",
    "to", "from", "with", "for", "he", "she", "it", "they", "we",
    "I", "you", "his", "her", "their", "my", "your", "its", "our",
    "but", "if",
    # ---- NUMBERS & QUANTIFIERS (20) ----
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand",
    "many", "few", "more", "less", "most", "least", "all", "some",
    # ---- GENDER / SOCIAL (20) ----
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "brother", "sister", "father", "mother", "son", "daughter",
    "husband", "wife", "prince", "princess", "actor", "actress",
    # ---- COLORS (12) ----
    "red", "blue", "green", "yellow", "white", "black", "brown",
    "orange", "purple", "pink", "gray", "gold",
    # ---- ABSTRACT (30) ----
    "love", "hate", "truth", "beauty", "freedom", "power",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    "fear", "joy", "pain", "trust", "faith", "peace",
    "war", "law", "right", "duty", "honor", "shame", "pride", "guilt",
    "anger", "grief",
    # ---- PLACES (20) ----
    "city", "town", "village", "country", "island", "valley", "cave",
    "bridge", "castle", "market", "church", "school", "hospital",
    "garden", "field", "park", "lake", "coast", "cliff", "path",
    # ---- FOOD & DRINK (15) ----
    "bread", "meat", "fruit", "milk", "water", "fire", "oil", "salt",
    "sugar", "coffee", "wine", "beer", "tea", "egg", "cheese",
    # ---- PLURAL FORMS (10) ----
    "dogs", "cats", "trees", "birds", "horses", "men", "women",
    "children", "hands", "eyes",
]

FAMILIES = {
    "animals_common":  ["dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger", "elephant", "bear"],
    "animals_insect":  ["ant", "bee", "butterfly", "beetle", "mosquito", "cricket", "spider", "worm", "fly", "frog"],
    "body_parts":      ["hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg", "head", "heart"],
    "colors":          ["red", "blue", "green", "yellow", "white", "black", "brown", "orange", "purple", "pink"],
    "numbers":         ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "quantifiers":     ["many", "few", "more", "less", "most", "least", "all", "some"],
    "adjectives_size": ["big", "small", "long", "short", "wide", "narrow", "deep", "shallow", "thick", "thin"],
    "adjectives_val":  ["good", "bad", "right", "wrong", "hard", "soft", "strong", "weak", "fast", "slow"],
    "verbs_motion":    ["run", "walk", "jump", "swim", "fly", "go", "come", "fall", "rise", "move"],
    "verbs_mental":    ["think", "know", "see", "hear", "feel", "love", "hate", "want", "find", "give"],
    "function_words":  ["the", "a", "and", "or", "not", "is", "was", "in", "on", "of"],
    "gender_terms":    ["king", "queen", "man", "woman", "boy", "girl", "brother", "sister", "father", "mother"],
    "abstract":        ["love", "hate", "truth", "hope", "fear", "joy", "pain", "peace", "war", "freedom"],
    "places":          ["city", "town", "village", "country", "island", "valley", "field", "lake", "park", "forest"],
}

MAX_RADIUS = 4
TOP_K_LIST = [10, 50, 100]

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def top_k_overlap(pred, actual, k):
    return len(set(np.argsort(pred)[-k:]) & set(np.argsort(actual)[-k:])) / k

def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

# ── Single-pass extraction: get ALL required layer hidden states at once ───────
def get_required_layers(text):
    """Single forward pass, return dict layer_idx → hidden_state (last pos)."""
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {l: out.hidden_states[l][0, pos, :].numpy().astype(np.float32)
            for l in REQUIRED_LAYERS}

def get_logits(word):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].numpy().astype(np.float32)

# ── Build T2 axes ─────────────────────────────────────────────────────────────
print("Building T2 axes (single-pass extraction) ...")
t2_axes = {}
for (ak, layer) in DECISION_AXES:
    diffs = []
    for s1, s2 in AXIS_PAIRS[ak]:
        h1 = get_required_layers(s1)[layer]
        h2 = get_required_layers(s2)[layer]
        d  = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if diffs:
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        t2_axes[ak] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    else:
        t2_axes[ak] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {ak}")
print()

# ── Deduplicate probe tokens ──────────────────────────────────────────────────
PROBE_TOKENS = list(dict.fromkeys(PROBE_TOKENS))   # preserve order, remove dups
print(f"Probe vocabulary: {len(PROBE_TOKENS)} tokens")

# ── Collect all hidden states and logits in single passes ─────────────────────
print(f"Computing hidden states (single pass per token) ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        hs_all = get_required_layers(" " + word.strip())   # {layer: vector}
        lv     = get_logits(word)
        # Map axis_key → hidden vector at its layer
        hs_by_axis = {ak: hs_all[layer] for (ak, layer) in DECISION_AXES}
        tokens_data[word] = {"hs": hs_by_axis, "logits": lv}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words         = list(tokens_data.keys())
logit_vecs    = {w: tokens_data[w]["logits"] for w in words}
global_mean   = np.mean([logit_vecs[w] for w in words], axis=0)
print(f"  Collected {len(words)} tokens\n")

# ── Calibrate thresholds + ternary addresses ──────────────────────────────────
classes = {}
for (ak, layer) in DECISION_AXES:
    axis = t2_axes[ak]
    if np.linalg.norm(axis) < 1e-6:
        for w in words: classes.setdefault(w, {})[ak] = "U"
        continue
    projs   = np.array([float(np.dot(tokens_data[w]["hs"][ak], axis)) for w in words])
    max_p   = np.percentile(projs, 95)
    hi, lo  = max_p * INV_PHI, max_p * INV_PHI2
    for i, w in enumerate(words):
        p = projs[i]
        classes.setdefault(w, {})[ak] = ("H" if p > hi else "L" if p < lo else "U")

axis_keys = [ak for ak, _ in DECISION_AXES]
addresses  = {w: "".join(classes[w][ak] for ak in axis_keys) for w in words}
leaf_cnts  = Counter(addresses.values())
n_leaves   = len(leaf_cnts)
n_occupied_ge2 = sum(1 for c in leaf_cnts.values() if c >= 2)

print(f"Trie: {n_leaves} leaves  ({n_occupied_ge2} with ≥2 tokens)"
      f"  out of 3^8=6561 possible")
print()

# ── Ternary metric property ───────────────────────────────────────────────────
all_pairs = [(words[i], words[j])
             for i in range(len(words))
             for j in range(i+1, len(words))]

n = len(words)
hamm_matrix = np.zeros((n, n), dtype=np.int32)
for i in range(n):
    for j in range(n):
        hamm_matrix[i, j] = hamming(addresses[words[i]], addresses[words[j]])

by_hamming = defaultdict(list)
for w1, w2 in all_pairs:
    d   = hamming(addresses[w1], addresses[w2])
    sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
    by_hamming[d].append(sim)

print("=" * 72)
print("Ternary metric property (Hamming distance → logit cosine)")
print("=" * 72)
print(f"  {'dist':>5}  {'mean_sim':>9}  {'n_pairs':>9}  monotone")
prev_sim = 1.1
monotone = True
for d in sorted(by_hamming.keys()):
    m = float(np.mean(by_hamming[d]))
    n_p = len(by_hamming[d])
    mono = "↓" if m < prev_sim else "↑ BREAK"
    if m > prev_sim: monotone = False
    print(f"  {d:>5}  {m:>9.4f}  {n_p:>9}  {mono}")
    prev_sim = m
print(f"\n  Overall monotone: {'YES ✓' if monotone else 'NO'}")
print(f"  d=0 mean: {float(np.mean(by_hamming[0])):.4f}  "
      f"d=8 mean: {float(np.mean(by_hamming.get(8, [0.0]))):.4f}  "
      f"range: {float(np.mean(by_hamming[0])) - float(np.mean(by_hamming.get(8, [0.0]))):.4f}")
print()

# ── Leave-one-out prediction ──────────────────────────────────────────────────
print("Running LOO prediction sweep ...")
r_results = {r: {"cosim": [], "topk": {k: [] for k in TOP_K_LIST},
                  "n_neighbors": []}
             for r in range(MAX_RADIUS + 1)}
global_results = {"cosim": [], "topk": {k: [] for k in TOP_K_LIST}}

for i, w in enumerate(words):
    actual = logit_vecs[w]
    other  = [j for j in range(len(words)) if j != i]
    global_pred = np.mean([logit_vecs[words[j]] for j in other], axis=0)
    global_results["cosim"].append(cos_sim(global_pred, actual))
    for k in TOP_K_LIST:
        global_results["topk"][k].append(top_k_overlap(global_pred, actual, k))

    for r in range(MAX_RADIUS + 1):
        nbrs = [j for j in other if hamm_matrix[i, j] <= r]
        if not nbrs:
            r_results[r]["cosim"].append(float("nan"))
            for k in TOP_K_LIST: r_results[r]["topk"][k].append(float("nan"))
            r_results[r]["n_neighbors"].append(0)
            continue
        wts = np.array([math.exp(-hamm_matrix[i, j]) for j in nbrs], dtype=np.float32)
        wts /= wts.sum()
        pred = np.array([logit_vecs[words[j]] for j in nbrs]).T @ wts
        r_results[r]["cosim"].append(cos_sim(pred, actual))
        for k in TOP_K_LIST:
            r_results[r]["topk"][k].append(top_k_overlap(pred, actual, k))
        r_results[r]["n_neighbors"].append(len(nbrs))

def safe_mean(lst):
    v = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")

glb_cos  = safe_mean(global_results["cosim"])
glb_t10  = safe_mean(global_results["topk"][10])
glb_t50  = safe_mean(global_results["topk"][50])
glb_t100 = safe_mean(global_results["topk"][100])

print("=" * 72)
print("LOO prediction quality vs Hamming radius")
print("=" * 72)
print(f"  {'radius':>8}  {'cos_sim':>9}  {'Δbaseline':>10}  "
      f"{'top-10':>8}  {'top-50':>8}  {'avg_nbrs':>9}")
print(f"  {'baseline':>8}  {glb_cos:>9.4f}  {'—':>10}  "
      f"{glb_t10:>8.3f}  {glb_t50:>8.3f}")

for r in range(MAX_RADIUS + 1):
    cos_m = safe_mean(r_results[r]["cosim"])
    t10_m = safe_mean(r_results[r]["topk"][10])
    t50_m = safe_mean(r_results[r]["topk"][50])
    nn_m  = safe_mean(r_results[r]["n_neighbors"])
    delta = cos_m - glb_cos
    mark  = " ✓" if delta > 0 else "  "
    print(f"  {f'r≤{r}':>8}  {cos_m:>9.4f}  {delta:>+10.4f}{mark}  "
          f"{t10_m:>8.3f}  {t50_m:>8.3f}  {nn_m:>9.1f}")
print()

# ── Trie density ──────────────────────────────────────────────────────────────
print("=" * 72)
print("Trie density")
print("=" * 72)
n_singleton = sum(1 for v in leaf_cnts.values() if v == 1)
n_pair      = sum(1 for v in leaf_cnts.values() if v == 2)
n_larger    = sum(1 for v in leaf_cnts.values() if v >= 3)
pct_with_nbr = 100 * sum(c for c in leaf_cnts.values() if c >= 2) / len(words)
print(f"  {len(words)} tokens  →  {n_leaves} unique leaves")
print(f"  Singletons: {n_singleton}  Pairs: {n_pair}  ≥3: {n_larger}")
print(f"  Tokens WITH same-leaf neighbor: {pct_with_nbr:.1f}%")
print(f"  3^8 = 6561 addresses, {100*n_leaves/6561:.1f}% occupied")
print()
print("  Top 20 most populated leaves:")
for addr, cnt in leaf_cnts.most_common(20):
    ww = [w for w in words if addresses[w] == addr]
    print(f"    [{addr}] ({cnt}): {' '.join(ww[:12])}")
print()

# ── Semantic family Hamming distances ─────────────────────────────────────────
print("=" * 72)
print("Semantic family clustering in ternary space")
print("=" * 72)
print(f"  {'family':>20}  {'mean_H':>7}  {'mean_sim':>9}  {'sd_H':>6}")

family_results = {}
for family, members in FAMILIES.items():
    fam = [w for w in members if w in words]
    if len(fam) < 2: continue
    idxs  = [words.index(w) for w in fam]
    dists = [hamm_matrix[idxs[a], idxs[b]]
             for a in range(len(idxs)) for b in range(a+1, len(idxs))]
    sims  = [cos_sim(logit_vecs[fam[a]], logit_vecs[fam[b]])
             for a in range(len(fam)) for b in range(a+1, len(fam))]
    mh = float(np.mean(dists))
    ms = float(np.mean(sims))
    sh = float(np.std(dists))
    family_results[family] = {"mean_hamming": mh, "mean_sim": ms, "std_hamming": sh,
                               "members": fam[:8]}
    print(f"  {family:>20}  {mh:>7.2f}  {ms:>9.4f}  {sh:>6.2f}")

print()

# ── Per-family tightest pairs ─────────────────────────────────────────────────
print("Tightest pairs within each family (min Hamming):")
for family, members in FAMILIES.items():
    fam  = [w for w in members if w in words]
    if len(fam) < 2: continue
    idxs = [words.index(w) for w in fam]
    best_d = min(hamm_matrix[idxs[a], idxs[b]]
                 for a in range(len(idxs)) for b in range(a+1, len(idxs)))
    tight = [(fam[a], fam[b], hamm_matrix[idxs[a], idxs[b]])
             for a in range(len(idxs)) for b in range(a+1, len(idxs))
             if hamm_matrix[idxs[a], idxs[b]] == best_d]
    print(f"  {family}: d={best_d}  pairs: "
          f"{', '.join(f'{p[0]}+{p[1]}' for p in tight[:3])}")
print()

# ── Day 76/77/78 comparison ───────────────────────────────────────────────────
best_r  = max(range(MAX_RADIUS+1), key=lambda r: safe_mean(r_results[r]["cosim"]))
best_cos = safe_mean(r_results[best_r]["cosim"])

print("=" * 72)
print("Comparison: Day 76 (164 tokens) vs Day 78 (" + str(len(words)) + " tokens)")
print("=" * 72)
d0_mean = float(np.mean(by_hamming[0])) if 0 in by_hamming else float("nan")
d8_mean = float(np.mean(by_hamming.get(8, [float("nan")])))
print(f"""
  Quantity                Day 76 (164)  Day 78 ({len(words)})
  Metric monotone         YES           {'YES ✓' if monotone else 'NO'}
  d=0 mean cosim          0.9092        {d0_mean:.4f}
  d=8 mean cosim          0.7849        {d8_mean:.4f}
  d=0 range               0.124         {d0_mean-d8_mean:.4f}
  LOO best radius         r≤3           r≤{best_r}
  LOO best cosim          0.9303        {best_cos:.4f}
  LOO improvement         +0.0161       {best_cos-glb_cos:+.4f}
  Same-leaf nbr coverage  22%           {pct_with_nbr:.1f}%
  Leaves occupied         142           {n_leaves}
  Singleton rate          78%           {100*n_singleton/len(words):.1f}%
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "n_tokens": len(words),
    "n_leaves": n_leaves,
    "n_singleton": n_singleton,
    "pct_with_same_leaf_nbr": pct_with_nbr,
    "monotone": monotone,
    "hamming_vs_sim": {str(d): float(np.mean(v)) for d, v in by_hamming.items()},
    "d0_mean": d0_mean, "d8_mean": d8_mean,
    "loo_by_radius": {
        str(r): {"cosim": safe_mean(r_results[r]["cosim"]),
                 "topk10": safe_mean(r_results[r]["topk"][10]),
                 "topk50": safe_mean(r_results[r]["topk"][50])}
        for r in range(MAX_RADIUS+1)
    },
    "loo_global_baseline": {"cosim": glb_cos, "topk10": glb_t10, "topk50": glb_t50},
    "loo_best_radius": best_r,
    "loo_improvement": best_cos - glb_cos,
    "family_results": family_results,
    "top20_leaves": [{"addr": a, "count": c,
                      "words": [w for w in words if addresses[w] == a][:10]}
                     for a, c in leaf_cnts.most_common(20)],
    "addresses": {w: addresses[w] for w in words},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 78 complete.")
