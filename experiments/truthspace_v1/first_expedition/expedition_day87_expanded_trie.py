#!/usr/bin/env python3
"""
Day 87 — Expanded Trie Dimensionality Sweep

DC 323 (Days 76–81): 8-bit φ-trie gives LOO cosim 0.9303 at r≤3.
DC 324 (Days 82–86): 20 orthogonal transformation dimensions confirmed.

KEY QUESTION: Does adding new axes IMPROVE the trie's LOO accuracy?
Is there a sweet spot between discrimination (more bits) and sparsity
(fewer tokens per leaf)?

TEST: Build tries with 8, 10, 12, 14, 16, 20 axes.
  For each dimensionality:
    1. Compute T2 axes at L28 for the top-k types
    2. Build ternary addresses for 401 probe tokens
    3. Run LOO generative lookup: predict logits from Hamming-near neighbors
    4. Measure best LOO cosim (over radii r=0..6)
    5. Measure trie density: % of tokens sharing a leaf, singleton rate

AXIS SELECTION (for 8→20 expansion):
  Rank by coherence from Day 86 (pooled 16-pair core) + new axes by
  their residual distance from the existing subspace (clearest signal):
    Tier 1 (original 8): gender, comparative, hypernym, plural,
                         synonym, concrete, past_tense, antonym
    Tier 2 (clearest new): passive (97.1% novel), question (95.4%),
                           negation (94.4%), causation (96.2%)
    Tier 3: part_whole, modality, degree, definiteness
    Tier 4: spatial, temporal, possession, aspect
"""
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day87_expanded_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI       = (1 + math.sqrt(5)) / 2
INV_PHI   = 1 / PHI
INV_PHI2  = 1 / PHI**2
TARGET_LAYER = 28

# ── Axis inventory ordered by novelty (residual from existing subspace) ───────
# Tier 1: core 8
# Tier 2: most novel new (passive 97.1%, causation 96.2%, question 95.4%, negation 94.4%)
# Tier 3: part_whole 87.7%, possession 87.5%, modality 87.9%, spatial 84.8%
# Tier 4: degree 84.3%, definiteness 82.1%, aspect 81.5%, temporal 68.3%
ORDERED_AXIS_NAMES = [
    "gender", "comparative", "hypernym", "plural",      # core 8
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",     # tier 2
    "part_whole", "possession", "modality", "spatial",  # tier 3
    "degree", "definiteness", "aspect", "temporal",     # tier 4
]

ALL_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom",   "The queen ruled with great wisdom"),
        ("A man walked through the forest",    "A woman walked through the forest"),
        ("The boy kicked the ball hard",       "The girl kicked the ball hard"),
        ("His brother arrived at the party",   "His sister arrived at the party"),
        ("The father worked to feed family",   "The mother worked to feed family"),
        ("A son was born in the winter",       "A daughter was born in the winter"),
        ("The prince rode across the land",    "The princess rode across the land"),
        ("The actor played a leading role",    "The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car", "The faster car"), ("A big dog", "A bigger dog"),
        ("The cold wind", "The colder wind"), ("A tall tree", "A taller tree"),
        ("The old house", "The older house"), ("A bright star", "A brighter star"),
        ("The dark room", "The darker room"), ("A hard rock", "A harder rock"),
    ],
    "hypernym": [
        ("The dog ran away from danger",    "The animal ran away from danger"),
        ("A rose bloomed in the garden",    "A flower bloomed in the garden"),
        ("The oak crashed in the storm",    "The tree crashed in the storm"),
        ("The car sped past the sign",      "The vehicle sped past the sign"),
        ("The eagle soared above the hill", "The bird soared above the hill"),
        ("The ruby gleamed in the light",   "The gem gleamed in the light"),
        ("The soldier marched into fight",  "The person marched into fight"),
        ("The hammer struck the nail",      "The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field",    "Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window", "The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist",    "Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm",   "The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk",          "Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road",   "The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky",     "Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text",   "The words appeared clearly in the printed text"),
    ],
    "synonym": [
        ("He is big", "He is large"), ("She is small", "She is tiny"),
        ("He runs fast", "He runs quick"), ("It is cold", "It is frigid"),
        ("She is happy", "She is joyful"), ("He spoke loudly", "He spoke noisily"),
        ("It is hard", "It is difficult"), ("He is old", "He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift",  "The burden is too heavy to lift"),
        ("The iron chain has broken now",   "The bond between them has broken"),
        ("The long road leads to the sea",  "The long journey leads to the sea"),
        ("The high wall blocks the view",   "The high barrier blocks the view"),
        ("The flame slowly fades away",     "The hope slowly fades away"),
        ("The strong root grips the soil",  "The strong base grips the earth"),
        ("The bridge connects two banks",   "The bond connects two communities"),
        ("The small key opens the door",    "The small answer opens the path"),
    ],
    "past_tense": [
        ("I walk to the market every single morning",        "I walked to the market every single morning"),
        ("She runs through the park after her long work",    "She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house",   "He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden",        "They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days",          "We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend",       "She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting", "He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire",   "They sang together around the evening campfire"),
    ],
    "antonym": [
        ("It is hot", "It is cold"), ("He runs fast", "He runs slow"),
        ("The light is on", "The dark is on"), ("The news is good","The news is bad"),
        ("It is hard", "It is soft"), ("She is happy", "She is sad"),
        ("He is strong", "He is weak"), ("It is the first", "It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse",         "The mouse was chased by the cat"),
        ("John broke the window",            "The window was broken by John"),
        ("The chef cooked the meal",         "The meal was cooked by the chef"),
        ("The dog bit the man",              "The man was bitten by the dog"),
        ("The teacher helped the student",   "The student was helped by the teacher"),
        ("The storm destroyed the house",    "The house was destroyed by the storm"),
        ("The artist painted the picture",   "The picture was painted by the artist"),
        ("The king signed the document",     "The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day",    "The ground gets completely wet"),
        ("The fire burns for a long time",  "The wood turns to ash slowly"),
        ("The sun heats the cold earth",    "The ice melts quickly in spring"),
        ("The wind blows the tree branches","The leaves fall to the ground"),
        ("The child cries very loudly",     "The mother comes running in"),
        ("The ball rolls off the tall edge","The ball falls to the floor"),
        ("The teacher praises the student", "The student feels very proud"),
        ("The glass breaks on hard stone",  "The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today",         "Is she very tired today"),
        ("He can swim really well",         "Can he swim really well"),
        ("They went to the market",         "Did they go to the market"),
        ("The car broke down again",        "Did the car break down again"),
        ("The dog is hungry now",           "Is the dog hungry now"),
        ("She wrote the letter herself",    "Did she write the letter herself"),
        ("He knows the right answer",       "Does he know the right answer"),
        ("The house looks very old",        "Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast",    "The dog is not fast"),
        ("She can swim well",  "She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good",   "The food is not good"),
        ("They work hard",     "They do not work hard"),
        ("The water is cold",  "The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today", "It will not rain today"),
    ],
    "part_whole": [
        ("She touched the finger gently",   "She touched the hand gently"),
        ("A leaf fell from the branch",     "A leaf fell from the tree"),
        ("The wheel turned on the road",    "The car turned on the road"),
        ("He hurt his knee badly",          "He hurt his leg badly"),
        ("The petal dropped to the ground", "The flower dropped to the ground"),
        ("The brick cracked in the heat",   "The wall cracked in the heat"),
        ("The key stuck in the lock",       "The key stuck in the door"),
        ("A chapter is hard to read",       "A book is hard to read"),
    ],
    "possession": [
        ("John has a very nice red car",          "That is John's very nice red car"),
        ("The teacher owns the old book",         "That is the teacher's old book"),
        ("She has a small white cat",             "That is her small white cat"),
        ("The king owns the golden crown",        "That is the king's golden crown"),
        ("The child has a favorite blue toy",     "That is the child's favorite blue toy"),
        ("The dog has a long leather collar",     "That is the dog's long leather collar"),
        ("He has a big wooden house",             "That is his big wooden house"),
        ("The shop has a special red door",       "That is the shop's special red door"),
    ],
    "modality": [
        ("She walks to the office every day",   "She must walk to the office every day"),
        ("He reads the news in the morning",    "He should read the news in the morning"),
        ("They swim in the cold lake",          "They can swim in the cold lake"),
        ("The student works hard all week",     "The student has to work hard all week"),
        ("The doctor sees ten patients",        "The doctor may see ten patients"),
        ("She writes her report carefully",     "She might write her report carefully"),
        ("He speaks at the big conference",     "He could speak at the big conference"),
        ("They arrive before the long meeting", "They ought to arrive before the meeting"),
    ],
    "spatial": [
        ("The cat sits on the table",            "The cat sits under the table"),
        ("The book lies inside the box",         "The book lies outside the box"),
        ("The bird flies above the old tree",    "The bird flies below the old tree"),
        ("The key is in the kitchen drawer",     "The key is on the kitchen drawer"),
        ("The car parked in front of the house", "The car parked behind the house"),
        ("The child stands near the door",       "The child stands far from the door"),
        ("The cup is to the left",               "The cup is to the right"),
        ("The dog ran into the room",            "The dog ran out of the room"),
    ],
    "degree": [
        ("It is warm outside today",       "It is hot outside today"),
        ("The food is good today",         "The food is excellent today"),
        ("He is a little tired now",       "He is extremely tired now"),
        ("The light was dim in the room",  "The light was blinding in the room"),
        ("She was slightly upset",         "She was furious"),
        ("The wind is gentle today",       "The wind is violent today"),
        ("The sound was soft",             "The sound was deafening"),
        ("He moved slowly at first",       "He moved instantly at first"),
    ],
    "definiteness": [
        ("A dog walked down the road",      "The dog walked down the road"),
        ("A cat sat by the window",         "The cat sat by the window"),
        ("A man stood at the corner",       "The man stood at the corner"),
        ("A bird sang in the morning",      "The bird sang in the morning"),
        ("A book sat on the table",         "The book sat on the table"),
        ("A car stopped at the light",      "The car stopped at the light"),
        ("A child played in the park",      "The child played in the park"),
        ("A storm came without warning",    "The storm came without warning"),
    ],
    "aspect": [
        ("She reads the long book",             "She is reading the long book"),
        ("He runs through the open park",       "He is running through the open park"),
        ("They build a tall brick wall",        "They are building a tall brick wall"),
        ("The child plays with the small toy",  "The child is playing with the small toy"),
        ("The chef cooks the evening meal",     "The chef is cooking the evening meal"),
        ("She writes a long difficult letter",  "She is writing a long difficult letter"),
        ("He paints the old wooden fence",      "He is painting the old wooden fence"),
        ("The dog chases the small brown cat",  "The dog is chasing the small brown cat"),
    ],
    "temporal": [
        ("Yesterday she walked to the market",   "Tomorrow she will walk to the market"),
        ("He studied hard last year",            "He will study hard next year"),
        ("They built the bridge long ago",       "They will build the bridge soon"),
        ("She cooked dinner an hour ago",        "She will cook dinner in an hour"),
        ("The rain fell hard last night",        "The rain will fall hard tonight"),
        ("He spoke at the meeting yesterday",    "He will speak at the meeting tomorrow"),
        ("The leaves fell in the autumn",        "The leaves will fall in the autumn"),
        ("The old man worked here years before", "The old man will work here years after"),
    ],
}

# 401 probe tokens
PROBE_TOKENS = [
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
    "pig", "sheep", "goat", "duck", "hen", "crow", "owl", "turtle",
    "lizard", "crab", "lobster", "octopus", "beetle", "butterfly", "worm",
    "tree", "flower", "rock", "stone", "wood", "leaf", "grass", "root",
    "river", "mountain", "ocean", "forest", "desert", "cloud", "rain",
    "house", "door", "window", "table", "chair", "book", "cup", "key",
    "car", "road", "bridge", "boat", "ship", "plane", "train", "bike",
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start", "stop",
    "think", "know", "see", "hear", "feel", "love", "hate", "want",
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
    "the", "a", "and", "or", "not", "is", "was", "in", "on", "of",
    "to", "from", "with", "for", "he", "she", "it", "they",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "many", "few", "more", "less", "most", "all", "some", "none",
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "red", "blue", "green", "yellow", "white", "black", "brown", "gold",
    "love", "hate", "truth", "freedom", "power", "time", "space", "hope",
]

def angle_deg(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(math.degrees(math.acos(float(np.clip(c, -1, 1)))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def run_model(text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, output_attentions=False)
    pos = inp["input_ids"].shape[1] - 1
    h28 = out.hidden_states[TARGET_LAYER][0, pos, :].numpy().astype(np.float32)
    logits = out.logits[0, pos, :].numpy().astype(np.float32)
    return h28, logits

def compute_t2(pairs):
    diffs = []
    for s1, s2 in pairs:
        h1, _ = run_model(s1)
        h2, _ = run_model(s2)
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if not diffs: return np.zeros(hidden_size, dtype=np.float32)
    v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
    return (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

# ── Compute all 20 T2 axes ────────────────────────────────────────────────────
print("Computing 20 T2 axes at L28 ...")
axes = {}
for name in ORDERED_AXIS_NAMES:
    axes[name] = compute_t2(ALL_PAIRS[name])
    print(f"  {name}")
print()

# ── Extract hidden states and logits for 401 probe tokens ────────────────────
print("Extracting hidden states + logits for 401 probe tokens ...")
hiddens = []; logits_all = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        h, lg = run_model(" " + word.strip())
        hiddens.append(h); logits_all.append(lg); valid_words.append(word)
    except Exception as e:
        print(f"  SKIP {word}: {e}")
hiddens    = np.array(hiddens, dtype=np.float32)
logits_all = np.array(logits_all, dtype=np.float32)
N = len(valid_words)
print(f"  {N} tokens extracted\n")

# Logit cosine similarity matrix (upper triangle)
logits_n = logits_all / (np.linalg.norm(logits_all, axis=1, keepdims=True) + 1e-10)
cosim_mat = (logits_n @ logits_n.T).astype(np.float32)

# ── Build ternary addresses at various dimensionalities ──────────────────────
def compute_ternary_address(h, axis_list, axes_dict):
    """Return string address H/U/L for each axis in axis_list."""
    projs = []
    for name in axis_list:
        projs.append(float(np.dot(h, axes_dict[name])))
    projs = np.array(projs)
    # Thresholds from this axis set
    max95 = float(np.percentile(np.abs(projs), 95))
    if max95 < 1e-6:
        return "U" * len(axis_list)
    addr = []
    for p in projs:
        if p > INV_PHI * max95:   addr.append("H")
        elif p < -INV_PHI * max95: addr.append("L")   # antonym direction
        elif abs(p) < INV_PHI2 * max95: addr.append("U")
        else:
            if p > 0: addr.append("H")
            else: addr.append("L")
    return "".join(addr)

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def loo_generative_lookup(addresses, cosim_mat, max_radius):
    """LOO lookup: predict token i's logit cosine from Hamming neighbors."""
    N = len(addresses)
    results = {r: [] for r in range(max_radius + 1)}

    for i in range(N):
        # Find all neighbors within each radius
        by_radius = {r: [] for r in range(max_radius + 1)}
        for j in range(N):
            if j == i: continue
            d = hamming(addresses[i], addresses[j])
            for r in range(d, max_radius + 1):
                by_radius[r].append(j)

        for r in range(max_radius + 1):
            neighbors = by_radius[r]
            if not neighbors: continue
            # mean cosim between i and its neighbors (ground truth)
            mean_cosim = float(np.mean([cosim_mat[i, j] for j in neighbors]))
            results[r].append(mean_cosim)

    return {r: float(np.mean(v)) if v else None for r, v in results.items()}

print("=" * 72)
print("Dimensionality sweep: LOO cosim vs number of trie axes")
print("=" * 72)
print(f"  {'dims':>6}  {'axes[:dims]':>40}  {'best_r':>6}  {'best_LOO':>10}  {'coverage%':>10}")

DIM_LEVELS = [8, 10, 12, 14, 16, 20]
sweep_results = {}

for n_dims in DIM_LEVELS:
    axis_subset = ORDERED_AXIS_NAMES[:n_dims]

    # Compute per-token projections and global max95 for thresholding
    all_projs = []
    for h in hiddens:
        for name in axis_subset:
            all_projs.append(float(np.dot(h, axes[name])))
    max95_global = float(np.percentile(np.abs(all_projs), 95))

    # Compute addresses
    addresses = []
    for h in hiddens:
        projs = [float(np.dot(h, axes[name])) for name in axis_subset]
        addr = []
        for p in projs:
            if p > INV_PHI * max95_global:     addr.append("H")
            elif p < -INV_PHI * max95_global:  addr.append("L")
            elif abs(p) < INV_PHI2 * max95_global: addr.append("U")
            else:
                addr.append("H" if p > 0 else "L")
        addresses.append("".join(addr))

    # Leaf statistics
    from collections import Counter
    leaf_counts = Counter(addresses)
    n_leaves    = len(leaf_counts)
    n_singleton = sum(1 for c in leaf_counts.values() if c == 1)
    coverage    = 100 * (N - n_singleton) / N   # % in non-singleton leaf

    # LOO at radii 0..min(4, n_dims//2)
    max_r = min(4, n_dims // 2)
    loo   = loo_generative_lookup(addresses, cosim_mat, max_r)
    valid = {r: v for r, v in loo.items() if v is not None}
    best_r    = max(valid, key=valid.get) if valid else -1
    best_cosim = valid[best_r] if valid else float("nan")

    sweep_results[n_dims] = {
        "axes": axis_subset,
        "n_leaves": n_leaves,
        "singleton_rate": n_singleton / N,
        "coverage_pct": coverage,
        "loo_by_radius": {r: float(v) if v else None for r, v in loo.items()},
        "best_r": best_r,
        "best_loo_cosim": best_cosim,
    }

    axes_str = f"{axis_subset[0]}...{axis_subset[-1]}" if n_dims > 3 else ",".join(axis_subset)
    print(f"  {n_dims:>6}  {axes_str:>40}  {best_r:>6}  {best_cosim:>10.4f}  {coverage:>10.1f}%")

print()

# ── Compare with Day 77 baseline (8D LOO) ────────────────────────────────────
print("=" * 72)
print("LOO by radius for each dimensionality")
print("=" * 72)
print(f"  {'r':>4}  " + "  ".join(f"{d:>8}D" for d in DIM_LEVELS))
for r in range(5):
    row = f"  {r:>4}  "
    for d in DIM_LEVELS:
        v = sweep_results[d]["loo_by_radius"].get(r)
        row += f"  {v:>8.4f}" if v else f"  {'—':>8}"
    print(row)
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 87 Summary")
print("=" * 72)
best_overall_dim = max(DIM_LEVELS, key=lambda d: sweep_results[d]["best_loo_cosim"])
print(f"""
  LOO cosim (best radius) by dimensionality:
  {'dims':>6}  {'best_r':>6}  {'best_LOO':>10}  {'coverage%':>10}
""")
for d in DIM_LEVELS:
    r = sweep_results[d]
    print(f"  {d:>6}  {r['best_r']:>6}  {r['best_loo_cosim']:>10.4f}  {r['coverage_pct']:>10.1f}%")
print(f"""
  Day 77 baseline (8D, r≤3): 0.9303
  Best dimensionality: {best_overall_dim}D (LOO={sweep_results[best_overall_dim]['best_loo_cosim']:.4f})
  
  Optimal dim (discrimination vs sparsity trade-off):
    More dims → higher discrimination → but more singletons
    Coverage drops as dims increase; best LOO may plateau or decrease
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump(sweep_results, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 87 complete.")
