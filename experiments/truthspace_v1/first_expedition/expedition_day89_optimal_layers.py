#!/usr/bin/env python3
"""
Day 89 — Optimal Layer Per Axis: Mixed-Layer 20D Trie

Day 77/78 found that different transformation types are most
discriminative at different layers:
  plural    → L1   (morphological)
  compar    → L15  (syntactic)
  gender    → L27  (lexical-semantic)
  others    → L28  (semantic output)

Day 88: L28-only 20D trie gave LOO=0.9058, below Day 77 mixed 0.9303.
The mixed-layer approach matters more than dimensional expansion alone.

THIS DAY: For each of the 20 axes, find the layer that maximizes
single-axis LOO cosim (r=0) over 401 probe tokens. Then build the
optimal mixed-layer 20D trie and measure full LOO.

LAYERS TESTED: L1, L8, L15, L22, L27, L28
EFFICIENCY: Extract ALL layers in one forward pass per token/pair.
"""
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day89_optimal_layers.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

TEST_LAYERS = [1, 8, 15, 22, 27, 28]

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

PROBE_TOKENS = [
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
]
PROBE_TOKENS = list(dict.fromkeys(PROBE_TOKENS))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}  probe tokens={len(PROBE_TOKENS)}\n")

def get_all_layers(text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    hs  = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
           for L in TEST_LAYERS}
    lg  = out.logits[0, pos, :].numpy().astype(np.float32)
    return hs, lg

# ── T2 axes at ALL layers for all 20 axes ─────────────────────────────────────
print("Computing T2 axes at layers", TEST_LAYERS, "for all 20 axes ...")
axes_by_layer = {L: {} for L in TEST_LAYERS}

for name in AXIS_NAMES_20:
    diffs_by_layer = {L: [] for L in TEST_LAYERS}
    for s1, s2 in ALL_PAIRS[name]:
        try:
            h1, _ = get_all_layers(s1)
            h2, _ = get_all_layers(s2)
            for L in TEST_LAYERS:
                d = h2[L] - h1[L]; n = np.linalg.norm(d)
                if n > 1e-6: diffs_by_layer[L].append(d / n)
        except:
            pass
    for L in TEST_LAYERS:
        if diffs_by_layer[L]:
            v = np.mean(diffs_by_layer[L], axis=0); nv = np.linalg.norm(v)
            axes_by_layer[L][name] = (v / nv if nv > 1e-6 else
                                       np.zeros(hidden_size)).astype(np.float32)
        else:
            axes_by_layer[L][name] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {name}")
print()

# ── Extract hidden states + logits for all probe tokens ──────────────────────
print("Extracting hidden states at all layers for 401 tokens ...")
hs_by_layer = {L: [] for L in TEST_LAYERS}
logits_list = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        hs, lg = get_all_layers(" " + word.strip())
        for L in TEST_LAYERS:
            hs_by_layer[L].append(hs[L])
        logits_list.append(lg); valid_words.append(word)
    except:
        pass
for L in TEST_LAYERS:
    hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
logits_arr = np.array(logits_list, dtype=np.float32)
N = len(valid_words)
logits_n  = logits_arr / (np.linalg.norm(logits_arr, axis=1, keepdims=True) + 1e-10)
cosim_mat = (logits_n @ logits_n.T).astype(np.float32)
print(f"  {N} tokens extracted\n")

def get_max95(projs):
    return float(np.percentile(np.abs(projs), 95))

def addr_bit(proj, max95):
    if proj > INV_PHI * max95:      return "H"
    elif proj < -INV_PHI * max95:   return "L"
    elif abs(proj) < INV_PHI2 * max95: return "U"
    else:                           return ("H" if proj > 0 else "L")

def loo_r0(addresses, cosim_mat):
    results = []
    for i in range(N):
        neighbors = [j for j in range(N) if j != i and addresses[j] == addresses[i]]
        if neighbors:
            results.append(float(np.mean([cosim_mat[i, j] for j in neighbors])))
    return float(np.mean(results)) if results else float("nan")

# ── Find optimal layer per axis (single-axis LOO r=0) ─────────────────────────
print("=" * 72)
print("Optimal layer per axis (single-axis LOO r=0 over 401 tokens)")
print("=" * 72)
print(f"  {'axis':>15}  " + "  ".join(f"L{L:>2}" for L in TEST_LAYERS) + "  best_layer")

optimal_layer = {}
axis_layer_loo = {}

for name in AXIS_NAMES_20:
    loo_per_layer = {}
    for L in TEST_LAYERS:
        ax = axes_by_layer[L][name]
        if np.linalg.norm(ax) < 1e-6: continue
        projs = [float(np.dot(hs_by_layer[L][i], ax)) for i in range(N)]
        max95 = get_max95(projs)
        if max95 < 1e-6: continue
        addrs = [addr_bit(projs[i], max95) for i in range(N)]
        loo_per_layer[L] = loo_r0(addrs, cosim_mat)
    best_L = max(loo_per_layer, key=loo_per_layer.get) if loo_per_layer else 28
    optimal_layer[name] = best_L
    axis_layer_loo[name] = loo_per_layer
    loo_str = "  ".join(f"{loo_per_layer.get(L, 0):.4f}" for L in TEST_LAYERS)
    print(f"  {name:>15}  {loo_str}  L{best_L}")

print()

# ── Build mixed-layer 20D trie ────────────────────────────────────────────────
print("=" * 72)
print("Building mixed-layer 20D trie with optimal layers")
print("=" * 72)

# Compute all-axis projections using optimal layers
all_projs = []
for i in range(N):
    for name in AXIS_NAMES_20:
        L  = optimal_layer[name]
        ax = axes_by_layer[L][name]
        all_projs.append(float(np.dot(hs_by_layer[L][i], ax)))
max95_global = get_max95(all_projs)

# Build addresses
addresses_20 = []
for i in range(N):
    addr = []
    for name in AXIS_NAMES_20:
        L  = optimal_layer[name]
        ax = axes_by_layer[L][name]
        p  = float(np.dot(hs_by_layer[L][i], ax))
        addr.append(addr_bit(p, max95_global))
    addresses_20.append("".join(addr))

leaf_counts = Counter(addresses_20)
n_leaves    = len(leaf_counts)
n_sing      = sum(1 for c in leaf_counts.values() if c == 1)
coverage    = 100 * (N - n_sing) / N
print(f"  20D mixed-layer: {n_leaves} leaves, {N-n_sing}/{N} in non-singleton ({coverage:.1f}%)")

def loo_lookup(addresses, max_r):
    results = {r: [] for r in range(max_r + 1)}
    for i in range(N):
        by_r = {r: [] for r in range(max_r + 1)}
        for j in range(N):
            if j == i: continue
            d = sum(a != b for a, b in zip(addresses[i], addresses[j]))
            for r in range(d, max_r + 1):
                by_r[r].append(j)
        for r in range(max_r + 1):
            if by_r[r]:
                results[r].append(float(np.mean([cosim_mat[i, j] for j in by_r[r]])))
    return {r: float(np.mean(v)) if v else None for r, v in results.items()}

loo_20 = loo_lookup(addresses_20, max_r=5)
print(f"\n  Mixed-layer 20D LOO by radius:")
valid = {r: v for r, v in loo_20.items() if v is not None}
best_r = max(valid, key=valid.get); best_loo = valid[best_r]
for r, v in sorted(valid.items()):
    marker = " ← BEST" if r == best_r else ""
    print(f"  r={r}: {v:.4f}{marker}")
print()

# ── Also test mixed-layer 8D trie (core axes only) ───────────────────────────
print("=" * 72)
print("Mixed-layer 8D trie (core axes, optimal layers)")
print("=" * 72)
core_8 = AXIS_NAMES_20[:8]
all_projs_8 = []
for i in range(N):
    for name in core_8:
        L = optimal_layer[name]; ax = axes_by_layer[L][name]
        all_projs_8.append(float(np.dot(hs_by_layer[L][i], ax)))
max95_8 = get_max95(all_projs_8)

addresses_8 = []
for i in range(N):
    addr = []
    for name in core_8:
        L = optimal_layer[name]; ax = axes_by_layer[L][name]
        p = float(np.dot(hs_by_layer[L][i], ax))
        addr.append(addr_bit(p, max95_8))
    addresses_8.append("".join(addr))

leaf_counts_8 = Counter(addresses_8)
n_leaves_8 = len(leaf_counts_8); n_sing_8 = sum(1 for c in leaf_counts_8.values() if c == 1)
coverage_8 = 100 * (N - n_sing_8) / N
print(f"  8D mixed-layer: {n_leaves_8} leaves, coverage {coverage_8:.1f}%")
loo_8 = loo_lookup(addresses_8, max_r=5)
valid8 = {r: v for r, v in loo_8.items() if v is not None}
best_r8 = max(valid8, key=valid8.get); best_loo8 = valid8[best_r8]
for r, v in sorted(valid8.items()):
    m = " ← BEST" if r == best_r8 else ""
    print(f"  r={r}: {v:.4f}{m}")
print()

# ── Final comparison ──────────────────────────────────────────────────────────
print("=" * 72)
print("Day 89 Summary: LOO Comparison")
print("=" * 72)
print(f"""
  Day 77 (8D, mixed layers, r≤3):       0.9303
  Day 88 (20D, L28-only, r=0):          0.9058
  Day 89 (8D, mixed layers, optimal):   {best_loo8:.4f}  r={best_r8}
  Day 89 (20D, mixed layers, optimal):  {best_loo:.4f}  r={best_r}

  Optimal layers found:
""")
for name in AXIS_NAMES_20:
    L = optimal_layer[name]
    loo_at_best = axis_layer_loo[name].get(L, float("nan"))
    print(f"  {name:>15}: L{L}  (single-axis LOO={loo_at_best:.4f})")

save = {
    "optimal_layer": {k: int(v) for k, v in optimal_layer.items()},
    "axis_layer_loo": {k: {str(L): float(v) for L, v in vs.items()}
                        for k, vs in axis_layer_loo.items()},
    "mixed_20d_loo": {r: float(v) if v else None for r, v in loo_20.items()},
    "mixed_8d_loo":  {r: float(v) if v else None for r, v in loo_8.items()},
    "mixed_20d_best_r": int(best_r), "mixed_20d_best_loo": float(best_loo),
    "mixed_8d_best_r":  int(best_r8), "mixed_8d_best_loo": float(best_loo8),
    "mixed_20d_coverage": float(coverage),
    "mixed_8d_coverage":  float(coverage_8),
    "day77_baseline": 0.9303,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 89 complete.")
