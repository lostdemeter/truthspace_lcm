#!/usr/bin/env python3
"""
Day 91 — Exact Generative LOO: True Apples-to-Apples vs Day 77

Day 90 used weighted-mean(pairwise_cosim) for LOO.
Day 77/78 used cosim(weighted_mean(logit_vectors), actual_logit).

These are DIFFERENT. The generative metric measures: "if I construct
a predicted logit distribution by averaging my neighbors, how similar
is that to my true distribution?" This is higher when the neighborhood
is COHERENT (vectors point in a common direction), not just when the
pairwise distances are small.

This day: implement exact Day 78 LOO (generative prediction), apply
it to 8D/12D/20D tries with correct per-axis thresholding.

EXPECTED:
  - 8D should reproduce Day 77's ~0.9303 at r≤3 (true apples-to-apples)
  - 20D may be higher at r=0 but degrade at r>0 (too sparse for aggregation)
  - 12D may achieve the best balance: r≤3 result close to or above 0.9303
"""
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day91_generative_loo.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

# Day 78 proven optimal layers for core 8
DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
}

AXIS_NAMES_20 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
    "part_whole", "possession", "modality", "spatial",
    "degree", "definiteness", "aspect", "temporal",
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
        ("John has a very nice red car",       "That is John's very nice red car"),
        ("The teacher owns the old book",      "That is the teacher's old book"),
        ("She has a small white cat",          "That is her small white cat"),
        ("The king owns the golden crown",     "That is the king's golden crown"),
        ("The child has a favorite blue toy",  "That is the child's favorite blue toy"),
        ("The dog has a long leather collar",  "That is the dog's long leather collar"),
        ("He has a big wooden house",          "That is his big wooden house"),
        ("The shop has a special red door",    "That is the shop's special red door"),
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
        ("A dog walked down the road",   "The dog walked down the road"),
        ("A cat sat by the window",      "The cat sat by the window"),
        ("A man stood at the corner",    "The man stood at the corner"),
        ("A bird sang in the morning",   "The bird sang in the morning"),
        ("A book sat on the table",      "The book sat on the table"),
        ("A car stopped at the light",   "The car stopped at the light"),
        ("A child played in the park",   "The child played in the park"),
        ("A storm came without warning", "The storm came without warning"),
    ],
    "aspect": [
        ("She reads the long book",            "She is reading the long book"),
        ("He runs through the open park",      "He is running through the open park"),
        ("They build a tall brick wall",       "They are building a tall brick wall"),
        ("The child plays with the small toy", "The child is playing with the small toy"),
        ("The chef cooks the evening meal",    "The chef is cooking the evening meal"),
        ("She writes a long difficult letter", "She is writing a long difficult letter"),
        ("He paints the old wooden fence",     "He is painting the old wooden fence"),
        ("The dog chases the small brown cat", "The dog is chasing the small brown cat"),
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

PROBE_TOKENS = list(dict.fromkeys([
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
    "pig", "sheep", "goat", "duck", "hen", "crow", "owl",
    "turtle", "lizard", "crab", "lobster", "octopus", "beetle",
    "butterfly", "worm", "fly", "mosquito", "cricket", "spider",
    "salmon", "tuna", "herring", "sparrow", "robin", "finch", "parrot",
    "tree", "flower", "rock", "stone", "wood", "leaf", "grass", "root",
    "river", "mountain", "ocean", "forest", "desert", "cloud", "rain",
    "snow", "wind", "sun", "moon", "star", "sky", "earth", "soil",
    "seed", "branch", "bark", "thorn", "moss", "mushroom", "coral",
    "house", "door", "window", "table", "chair", "book", "cup", "key",
    "car", "road", "bridge", "boat", "ship", "plane", "train", "bike",
    "knife", "fork", "spoon", "plate", "bowl", "glass", "bottle", "box",
    "bag", "rope", "wire", "nail", "hammer", "wheel", "clock", "lamp",
    "pen", "paper", "cloth", "thread", "button", "ring", "coin", "mirror",
    "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg",
    "head", "heart", "blood", "bone", "skin", "hair", "finger", "toe",
    "back", "chest", "neck", "shoulder",
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start", "stop",
    "think", "know", "see", "hear", "feel", "love", "hate", "want",
    "give", "take", "make", "find", "lose", "push", "pull", "turn",
    "move", "go", "come", "fall", "rise", "grow", "kill", "help",
    "ran", "walked", "jumped", "flew", "ate", "saw", "heard", "broke",
    "built", "wrote",
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
    "good", "bad", "right", "wrong", "high", "low", "long", "short",
    "wide", "narrow", "deep", "shallow", "thick", "thin", "heavy", "light",
    "clean", "dirty", "sweet", "bitter", "sharp", "dull", "loud", "quiet",
    "faster", "slower", "bigger", "smaller", "better", "worse",
    "biggest", "smallest", "best", "worst",
    "quickly", "slowly", "often", "never", "always", "very", "quite",
    "really", "just", "still",
    "the", "a", "and", "or", "not", "is", "was", "in", "on", "of",
    "to", "from", "with", "for", "he", "she", "it", "they", "we",
    "I", "you", "his", "her", "their", "my", "your", "its", "our",
    "but", "if",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand",
    "many", "few", "more", "less", "most", "least", "all", "some",
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "brother", "sister", "father", "mother", "son", "daughter",
    "husband", "wife", "prince", "princess", "actor", "actress",
    "red", "blue", "green", "yellow", "white", "black", "brown",
    "orange", "purple", "pink", "gray", "gold",
    "love", "hate", "truth", "beauty", "freedom", "power",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    "fear", "joy", "pain", "trust", "faith", "peace",
    "war", "law", "right", "duty", "honor", "shame", "pride", "guilt",
    "anger", "grief",
    "city", "town", "village", "country", "island", "valley", "cave",
    "bridge", "castle", "market", "church", "school", "hospital",
    "garden", "field", "park", "lake", "coast", "cliff", "path",
    "bread", "meat", "fruit", "milk", "water", "fire", "oil", "salt",
    "sugar", "coffee", "wine", "beer", "tea", "egg", "cheese",
    "dogs", "cats", "trees", "birds", "horses", "men", "women",
    "children", "hands", "eyes",
]))

AXIS_LAYER = {n: DAY78_LAYERS.get(n, 28) for n in AXIS_NAMES_20}
REQUIRED_LAYERS = sorted(set(AXIS_LAYER.values()))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}  tokens={len(PROBE_TOKENS)}  layers={REQUIRED_LAYERS}\n")

def get_layers(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    hs  = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}
    lg  = out.logits[0, pos, :].numpy().astype(np.float32)
    return hs, lg

# ── T2 axes ───────────────────────────────────────────────────────────────────
print("Computing 20 T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_20:
    L = AXIS_LAYER[name]
    diffs = []
    for s1, s2 in ALL_PAIRS[name]:
        try:
            h1, _ = get_layers(s1, [L]); h2, _ = get_layers(s2, [L])
            d = h2[L] - h1[L]; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    if diffs:
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    else:
        t2_axes[name] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {name:<15} L{L}")
print()

# ── Token extraction ──────────────────────────────────────────────────────────
print("Extracting hidden states + logits ...")
hs_by_layer = {L: [] for L in REQUIRED_LAYERS}
logits_list = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        hs, lg = get_layers(" " + word.strip(), REQUIRED_LAYERS)
        for L in REQUIRED_LAYERS: hs_by_layer[L].append(hs[L])
        logits_list.append(lg); valid_words.append(word)
    except: pass
for L in REQUIRED_LAYERS:
    hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
logits_arr = np.array(logits_list, dtype=np.float32)   # shape (N, V)
N = len(valid_words)
print(f"  {N} tokens\n")

# ── Day 78-exact per-axis thresholding ────────────────────────────────────────
def classify_day78(projs):
    max_p = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * len(projs)
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

def build_addresses(axis_subset):
    per_axis = {}
    for name in axis_subset:
        L = AXIS_LAYER[name]; ax = t2_axes[name]
        if np.linalg.norm(ax) < 1e-6:
            per_axis[name] = ["U"] * N; continue
        projs = [float(np.dot(hs_by_layer[L][i], ax)) for i in range(N)]
        per_axis[name] = classify_day78(projs)
    return ["".join(per_axis[n][i] for n in axis_subset) for i in range(N)]

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# ── EXACT Day 78 generative LOO ───────────────────────────────────────────────
def generative_loo(addresses, max_r):
    """
    Day 78-exact: for each token at radius r, predict logit vector as
    exp(-hamm)-weighted mean of neighbor logit vectors, then compute
    cosine similarity to actual logit vector.
    Also compute a global baseline (mean of all others).
    """
    results = {r: [] for r in range(max_r + 1)}
    global_cosims = []

    for i in range(N):
        actual = logits_arr[i]
        others = list(range(N)); others.remove(i)

        # Global baseline
        global_pred = np.mean(logits_arr[others], axis=0)
        na = np.linalg.norm(actual); np_ = np.linalg.norm(global_pred)
        if na > 1e-10 and np_ > 1e-10:
            global_cosims.append(float(np.dot(actual, global_pred) / (na * np_)))

        for r in range(max_r + 1):
            nbrs = [(j, hamming(addresses[i], addresses[j]))
                    for j in others
                    if hamming(addresses[i], addresses[j]) <= r]
            if not nbrs: continue
            wts  = np.array([math.exp(-d) for _, d in nbrs], dtype=np.float32)
            wts /= wts.sum()
            pred = np.sum(logits_arr[[j for j, _ in nbrs]] * wts[:, None], axis=0)
            np_ = np.linalg.norm(pred)
            if na > 1e-10 and np_ > 1e-10:
                results[r].append(float(np.dot(actual, pred) / (na * np_)))

    baseline = float(np.mean(global_cosims)) if global_cosims else float("nan")
    loo = {r: float(np.mean(v)) if v else None for r, v in results.items()}
    return loo, baseline

# ── Run for 8D, 12D, 20D ────────────────────────────────────────────────────
CONFIGS = {
    "8D (Day78 layers)":  AXIS_NAMES_20[:8],
    "12D (tier1+tier2)":  AXIS_NAMES_20[:12],
    "20D (all axes)":     AXIS_NAMES_20[:20],
}
MAX_R = 5
all_results = {}

print("=" * 72)
print("Generative LOO: cosim(weighted_mean(logit_neighbors), actual_logit)")
print("Day 77 baseline: 0.9303 at r≤3")
print("=" * 72)
print(f"  {'config':>22}  {'n_leaves':>9}  {'coverage%':>10}  {'baseline':>10}  {'best_r':>7}  {'best_LOO':>10}")

for cfg, axis_subset in CONFIGS.items():
    addrs = build_addresses(axis_subset)
    lc = Counter(addrs)
    n_leaves = len(lc); n_sing = sum(1 for c in lc.values() if c == 1)
    cov = 100 * (N - n_sing) / N

    loo, baseline = generative_loo(addrs, MAX_R)
    valid = {r: v for r, v in loo.items() if v is not None}
    best_r = max(valid, key=valid.get); best_v = valid[best_r]

    all_results[cfg] = {
        "n_leaves": n_leaves, "coverage_pct": cov,
        "baseline": baseline, "loo": loo,
        "best_r": best_r, "best_loo": best_v,
    }
    print(f"  {cfg:>22}  {n_leaves:>9}  {cov:>10.1f}%  {baseline:>10.4f}  {best_r:>7}  {best_v:>10.4f}")

print()
print(f"  {'r':>4}  " + "  ".join(f"{k:>22}" for k in CONFIGS))
for r in range(MAX_R + 1):
    row = f"  {r:>4}  "
    for k in CONFIGS:
        v = all_results[k]["loo"].get(r)
        row += f"  {v:>22.4f}" if v else f"  {'—':>22}"
    print(row)
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 91 Summary")
print("=" * 72)
print(f"\n  Day 77 (8D, r≤3, generative):  0.9303\n")
for k, r in all_results.items():
    delta = r["best_loo"] - 0.9303
    print(f"  {k}: LOO={r['best_loo']:.4f} r={r['best_r']}  ({delta:+.4f} vs Day77)")
    print(f"    coverage={r['coverage_pct']:.1f}%  baseline={r['baseline']:.4f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 91 complete.")
