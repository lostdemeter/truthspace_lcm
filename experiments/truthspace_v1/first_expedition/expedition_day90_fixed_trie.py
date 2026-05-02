#!/usr/bin/env python3
"""
Day 90 — Fixed Trie: Per-Axis Thresholding + Hamming-Weighted LOO

Days 87-89 had two bugs compared to the Day 77/78 methodology:

Bug 1 (CRITICAL): Global max95 across all axes instead of per-axis.
  Day 78: max_p = np.percentile(projs_for_THIS_axis, 95)
  Days 87-89: max95 = np.percentile(ALL projections from ALL axes, 95)
  Result: when axes have different magnitudes (esp. mixed layers),
  most projections fall into U → very few unique addresses.

Bug 2 (IMPORTANT): Unweighted LOO mean instead of Hamming-weighted.
  Day 78: wts = exp(-hamm); pred = weighted mean of neighbors
  Days 87-89: simple mean over neighbors at r≤max_r
  Result: Day 78's exp(-hamm) weighting upweights closer neighbors
  → r=3 is optimal (includes same+nearby with distance weighting)
  → My r=0 was always optimal because simple mean degrades at r>0.

Also: Day 78's threshold formula:
  hi = max_p * INV_PHI  (H: p > hi)
  lo = max_p * INV_PHI2 (L: p < lo, includes near-zero and negatives)
  U:  lo ≤ p ≤ hi

This day: apply all three fixes. Test dimensionality levels 8,12,20.
PREDICTION:
  - 8D (core, per-axis) should reproduce Day 77 baseline ~0.9303
  - 12D (per-axis, hamming-weighted) may improve on 0.9303
  - 20D best dimensionality to be confirmed with correct methodology
"""
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day90_fixed_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

# Original 8 axes from Day 78 (with their proven optimal layers)
DAY78_LAYERS = {
    "gender":    27,
    "comparative": 15,
    "hypernym":  28,
    "plural":     1,
    "synonym":   28,
    "concrete":  28,
    "past_tense":28,
    "antonym":   28,
}

# Ordered axis names for expansion tests
AXIS_NAMES_20 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
    "part_whole", "possession", "modality", "spatial",
    "degree", "definiteness", "aspect", "temporal",
]

# Default layer for new axes (will use L28 as fallback)
NEW_AX_LAYER = 28

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

# Build layer set: Day78 mixed layers for core 8, L28 for new 12
AXIS_LAYER = {}
for name in AXIS_NAMES_20:
    AXIS_LAYER[name] = DAY78_LAYERS.get(name, NEW_AX_LAYER)
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

# ── Compute T2 axes ────────────────────────────────────────────────────────────
print("Computing 20 T2 axes (each at its assigned layer) ...")
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

# ── Extract token hidden states at required layers ────────────────────────────
print("Extracting token hidden states at all required layers ...")
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
logits_arr = np.array(logits_list, dtype=np.float32)
N = len(valid_words)
logits_n  = logits_arr / (np.linalg.norm(logits_arr, axis=1, keepdims=True) + 1e-10)
cosim_mat = (logits_n @ logits_n.T).astype(np.float32)
print(f"  {N} tokens extracted\n")

# ── Day 78-exact thresholding (PER AXIS, unsigned) ───────────────────────────
def classify_day78(projs):
    """Day 78 exact formula: H/U/L per axis independently."""
    max_p = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * len(projs)
    hi = max_p * INV_PHI   # 0.618 × max
    lo = max_p * INV_PHI2  # 0.382 × max
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

def build_addresses(axis_subset):
    """Build ternary addresses using per-axis Day78 thresholding."""
    per_axis_classes = {}
    for name in axis_subset:
        L   = AXIS_LAYER[name]
        ax  = t2_axes[name]
        if np.linalg.norm(ax) < 1e-6:
            per_axis_classes[name] = ["U"] * N
            continue
        projs = [float(np.dot(hs_by_layer[L][i], ax)) for i in range(N)]
        per_axis_classes[name] = classify_day78(projs)
    addresses = []
    for i in range(N):
        addresses.append("".join(per_axis_classes[name][i] for name in axis_subset))
    return addresses

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def loo_hamming_weighted(addresses, cosim_mat, max_r):
    """Day 78 exact LOO: exp(-hamm) weighted mean of neighbors at r≤max_r."""
    results_r = {r: [] for r in range(max_r + 1)}
    for i in range(N):
        for max_r_test in range(max_r + 1):
            nbrs = []
            for j in range(N):
                if j == i: continue
                d = hamming(addresses[i], addresses[j])
                if d <= max_r_test: nbrs.append((j, d))
            if not nbrs: continue
            wts  = np.array([math.exp(-d) for _, d in nbrs], dtype=np.float32)
            wts /= wts.sum()
            pred = np.array([cosim_mat[i, j] for j, _ in nbrs])
            results_r[max_r_test].append(float(np.dot(pred, wts)))
    return {r: float(np.mean(v)) if v else None for r, v in results_r.items()}

# ── Test configurations ───────────────────────────────────────────────────────
CONFIGS = {
    "8D_day78_layers":   AXIS_NAMES_20[:8],    # exact Day 78 setup
    "12D_tier1+tier2":   AXIS_NAMES_20[:12],
    "20D_all":           AXIS_NAMES_20[:20],
}

MAX_R = 5
results = {}

print("=" * 72)
print("LOO comparison (Hamming-weighted, per-axis threshold = Day 78 method)")
print("=" * 72)
print(f"  {'config':>20}  {'n_leaves':>9}  {'coverage%':>10}  {'best_r':>7}  {'best_LOO':>10}")

for cfg_name, axis_subset in CONFIGS.items():
    addrs = build_addresses(axis_subset)
    leaf_cnt   = Counter(addrs)
    n_leaves   = len(leaf_cnt)
    n_sing     = sum(1 for c in leaf_cnt.values() if c == 1)
    coverage   = 100 * (N - n_sing) / N

    loo = loo_hamming_weighted(addrs, cosim_mat, MAX_R)
    valid = {r: v for r, v in loo.items() if v is not None}
    best_r  = max(valid, key=valid.get)
    best_v  = valid[best_r]

    results[cfg_name] = {
        "axes": axis_subset, "n_leaves": n_leaves,
        "coverage_pct": coverage, "loo_by_r": loo,
        "best_r": best_r, "best_loo": best_v,
    }
    print(f"  {cfg_name:>20}  {n_leaves:>9}  {coverage:>10.1f}%  {best_r:>7}  {best_v:>10.4f}")

print()

# ── LOO by radius table ───────────────────────────────────────────────────────
cfg_keys = list(CONFIGS.keys())
print(f"  {'r':>4}  " + "  ".join(f"{k:>20}" for k in cfg_keys))
for r in range(MAX_R + 1):
    row = f"  {r:>4}  "
    for k in cfg_keys:
        v = results[k]["loo_by_r"].get(r)
        row += f"  {v:>20.4f}" if v else f"  {'—':>20}"
    print(row)
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 90 Summary")
print("=" * 72)
print(f"""
  Day 77 (8D, mixed layers, r≤3):  0.9303  ← original baseline
  
  Day 90 results (Hamming-weighted, per-axis threshold):""")
for k, r in results.items():
    vs_77 = r["best_loo"] - 0.9303
    print(f"  {k:>20}: {r['best_loo']:.4f}  r={r['best_r']}  ({vs_77:+.4f} vs Day 77)")

best_cfg = max(results, key=lambda k: results[k]["best_loo"])
print(f"""
  Best config: {best_cfg}  LOO={results[best_cfg]['best_loo']:.4f}
  
  Axis layer assignments for 20D:""")
for name in AXIS_NAMES_20:
    print(f"    {name:>15}: L{AXIS_LAYER[name]}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({k: {**v, "axes": v["axes"]} for k, v in results.items()},
              f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 90 complete.")
