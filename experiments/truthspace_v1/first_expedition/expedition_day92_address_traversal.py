#!/usr/bin/env python3
"""
Day 92 — Address Traversal: Is the Trie a Semantic Coordinate System?

DC 325 confirmed 12D φ-trie gives LOO=0.9443. But is the address more
than a hash? Does it encode STRUCTURE?

HYPOTHESIS: The 12-bit ternary address is a semantic coordinate.
  - Bit 0 (gender@L27): L=masculine-coded, H=feminine-coded
  - Bit 2 (hypernym@L28): H=more-general, L=more-specific
  - Bit 3 (plural@L1): H=plural-form, L=singular-form
  - etc.

If true: flipping bit k in a token's address navigates to a
semantically related token that differs by exactly transformation k.
Examples:
  king (addr: ...L...) → flip gender → queen (addr: ...H...)
  dog  (addr: ...L...) → flip plural  → dogs  (addr: ...H...)
  fast (addr: ...L...) → flip compar  → faster(addr: ...H...)

TEST:
  1. Compute 12D addresses for all 401 tokens
  2. For each token, construct 12 "flipped" addresses (one bit at a time)
  3. Find the closest token to each flipped address (Hamming-nearest)
  4. Evaluate: do known semantic pairs land at each other's flipped address?
  5. Rank: for gender flip of "king", does "queen" appear in top-5?

GROUND TRUTH PAIRS (known semantic transformation pairs):
  gender:     king/queen, man/woman, boy/girl, brother/sister,
              father/mother, son/daughter, prince/princess, actor/actress
  plural:     dog/dogs, cat/cats, tree/trees, bird/birds, hand/hands, eye/eyes
  past_tense: run/ran, walk/walked, jump/jumped, fly/flew, eat/ate
  comparative: fast/faster, big/bigger, good/better, slow/slower, bad/worse
  antonym:    hot/cold, big/small, fast/slow, hard/soft, happy/sad
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day92_address_traversal.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]
AXIS_LAYER = {n: DAY78_LAYERS.get(n, 28) for n in AXIS_NAMES_12}

# Ground truth pairs for controlled evaluation
GROUND_TRUTH = {
    "gender":     [("king","queen"),("man","woman"),("boy","girl"),
                   ("brother","sister"),("father","mother"),("son","daughter"),
                   ("prince","princess"),("actor","actress")],
    "plural":     [("dog","dogs"),("cat","cats"),("tree","trees"),
                   ("bird","birds"),("hand","hands"),("eye","eyes")],
    "past_tense": [("run","ran"),("walk","walked"),("jump","jumped"),
                   ("fly","flew"),("eat","ate"),("build","built"),
                   ("write","wrote"),("break","broke")],
    "comparative":[("fast","faster"),("big","bigger"),("slow","slower"),
                   ("small","smaller"),("good","better"),("bad","worse")],
    "antonym":    [("hot","cold"),("big","small"),("fast","slow"),
                   ("hard","soft"),("happy","sad"),("strong","weak"),
                   ("good","bad"),("old","new")],
}

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

REQUIRED_LAYERS = sorted(set(AXIS_LAYER.values()))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}  tokens={len(PROBE_TOKENS)}\n")

def get_layers(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    hs  = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}
    lg  = out.logits[0, pos, :].numpy().astype(np.float32)
    return hs, lg

# ── T2 axes ───────────────────────────────────────────────────────────────────
print("Computing 12 T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
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
print("Extracting hidden states for all tokens ...")
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
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── Per-axis Day78 thresholding ───────────────────────────────────────────────
per_axis_classes = {}
per_axis_max95   = {}
for name in AXIS_NAMES_12:
    L = AXIS_LAYER[name]; ax = t2_axes[name]
    if np.linalg.norm(ax) < 1e-6:
        per_axis_classes[name] = ["U"] * N; per_axis_max95[name] = 0.0; continue
    projs = np.array([float(np.dot(hs_by_layer[L][i], ax)) for i in range(N)])
    max_p = float(np.percentile(projs, 95))
    per_axis_max95[name] = max_p
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    per_axis_classes[name] = ["H" if p > hi else "L" if p < lo else "U"
                               for p in projs]

addresses = ["".join(per_axis_classes[n][i] for n in AXIS_NAMES_12)
             for i in range(N)]
addr_to_words = defaultdict(list)
for i, w in enumerate(valid_words):
    addr_to_words[addresses[i]].append(w)

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# ── Address traversal: flip single bits ──────────────────────────────────────
FLIP_MAP = {"H": "L", "L": "H", "U": "H"}   # U→H: move toward transformation

def flip_bit(addr, bit_idx):
    lst = list(addr)
    lst[bit_idx] = FLIP_MAP[lst[bit_idx]]
    return "".join(lst)

def find_nearest(target_addr, exclude_words=None, top_k=5):
    """Find top_k tokens nearest to target_addr by Hamming distance."""
    exclude = set(exclude_words or [])
    candidates = [(w, hamming(target_addr, addresses[word_idx[w]]))
                  for w in valid_words if w not in exclude and w in word_idx]
    candidates.sort(key=lambda x: x[1])
    return candidates[:top_k]

# ── Controlled traversal evaluation ──────────────────────────────────────────
print("=" * 72)
print("Address traversal evaluation: known semantic pairs")
print("=" * 72)

results_by_axis = {}
for axis_name, pairs in GROUND_TRUTH.items():
    if axis_name not in AXIS_NAMES_12:
        continue
    bit_idx = AXIS_NAMES_12.index(axis_name)
    print(f"\nAxis {bit_idx}: {axis_name} (L{AXIS_LAYER[axis_name]})")
    print(f"  {'source':>12}  src_bit  {'target':>10}  tgt_bit  match?  rank  top3")

    hits = 0; total = 0
    pair_results = []
    for src, tgt in pairs:
        if src not in word_idx or tgt not in word_idx:
            continue
        si = word_idx[src]; ti = word_idx[tgt]
        src_bit = per_axis_classes[axis_name][si]
        tgt_bit = per_axis_classes[axis_name][ti]
        src_addr = addresses[si]
        flipped  = flip_bit(src_addr, bit_idx)

        # Find tokens at/near flipped address
        nearest = find_nearest(flipped, exclude_words=[src], top_k=10)
        nearest_words = [w for w, _ in nearest]
        tgt_rank = nearest_words.index(tgt) if tgt in nearest_words else -1
        top3     = nearest_words[:3]
        match    = tgt_rank >= 0 and tgt_rank < 5

        print(f"  {src:>12}  {src_bit:>7}  {tgt:>10}  {tgt_bit:>7}  "
              f"{'✓' if match else '✗':>6}  "
              f"{tgt_rank if tgt_rank >= 0 else 'miss':>4}  "
              f"{'/'.join(top3)}")
        if match: hits += 1
        total += 1
        pair_results.append({
            "src": src, "tgt": tgt,
            "src_bit": src_bit, "tgt_bit": tgt_bit,
            "src_addr": src_addr, "flipped_addr": flipped,
            "top5": nearest_words[:5], "tgt_rank": tgt_rank,
            "hit": match,
        })

    rate = hits / total if total > 0 else 0
    print(f"\n  {axis_name}: {hits}/{total} correct ({100*rate:.1f}%)")
    results_by_axis[axis_name] = {"pairs": pair_results, "hit_rate": rate,
                                  "hits": hits, "total": total}

# ── Cross-check: address bit distributions for controlled pairs ───────────────
print()
print("=" * 72)
print("Address bit analysis for known pairs by category")
print("=" * 72)
for axis_name, pairs in GROUND_TRUTH.items():
    if axis_name not in AXIS_NAMES_12: continue
    bit_idx = AXIS_NAMES_12.index(axis_name)
    src_bits = []; tgt_bits = []
    for src, tgt in pairs:
        if src in word_idx and tgt in word_idx:
            src_bits.append(per_axis_classes[axis_name][word_idx[src]])
            tgt_bits.append(per_axis_classes[axis_name][word_idx[tgt]])
    src_dist = Counter(src_bits); tgt_dist = Counter(tgt_bits)
    same = sum(1 for s, t in zip(src_bits, tgt_bits) if s == t)
    diff = sum(1 for s, t in zip(src_bits, tgt_bits) if s != t)
    print(f"  {axis_name:>15}: src {dict(src_dist)}  tgt {dict(tgt_dist)}  "
          f"same_bit={same}  diff_bit={diff}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 92 Summary")
print("=" * 72)
total_hits = sum(r["hits"] for r in results_by_axis.values())
total_pairs = sum(r["total"] for r in results_by_axis.values())
print(f"\n  Overall traversal accuracy: {total_hits}/{total_pairs} "
      f"({100*total_hits/total_pairs:.1f}% if pairs present in vocab)")
print(f"\n  Per-axis:")
for name, r in results_by_axis.items():
    print(f"    {name:>15}: {r['hits']:>2}/{r['total']:>2}  ({100*r['hit_rate']:.0f}%)")

print(f"""
  INTERPRETATION:
  - High hit rate → trie IS a semantic coordinate system
  - Low hit rate  → trie is a hash, not a navigable structure
  - Mixed results → some axes are semantic coordinates, others not

  Key: for the trie to be a true semantic coordinate:
    src_bit ≠ tgt_bit for most pairs (bit ENCODES the feature)
    flipped address nearest-neighbor IS the semantic target
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"results_by_axis": results_by_axis,
               "axis_names": AXIS_NAMES_12,
               "vocab_size": N,
               "total_hits": total_hits,
               "total_pairs": total_pairs},
              f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 92 complete.")
