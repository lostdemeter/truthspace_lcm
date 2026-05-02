#!/usr/bin/env python3
"""
Day 94 — Hybrid Trie: Token-Level Morphological Axes at Optimal Layers

Day 93 showed:
  - Token-level and sentence-level axes are 77.6° apart (different signals)
  - Plural token coherence=0.488 at L28 (best morphological signal)
  - past_tense token coherence=0.347 at L28
  - But Day 93 used L28 for all token-level axes
  - Day 78 found plural works best at L1 (early layers preserve morphology)

Hypothesis: token-level morphological axes at EARLY LAYERS (L1) may be
more coherent and navigable than at L28. The token embedding (L0/L1) is
where inflectional morphology (plural -s, past -ed) lives as a signal,
before contextual blending drowns it out at deep layers.

EXPERIMENT:
  1. Compute token-level T2 axes for plural, past_tense, comparative
     at layers [1, 8, 15, 22, 27, 28]
  2. Measure pairwise coherence at each layer
  3. Find optimal layer per axis (maximum coherence)
  4. Build 12D hybrid trie:
     - Core axes: Day78 sentence-level (gender@L27, comparative@L15,
       hypernym@L28, plural@L1, synonym@L28, concrete@L28,
       past_tense@L28, antonym@L28)
     - Replace plural and past_tense with token-level versions at
       their optimal layers
  5. Test LOO (generative, Day91 method) + address traversal (Day92 method)
  6. Compare: hybrid trie vs original Day 91/92 sentence-level trie
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day94_hybrid_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2
TEST_LAYERS = [1, 8, 15, 22, 27, 28]

# Original Day 78 sentence-level pairs
SENT_PAIRS = {
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

# Token-level pairs for morphological axes
TOK_PAIRS = {
    "plural": [
        (" dog"," dogs"), (" cat"," cats"), (" tree"," trees"),
        (" bird"," birds"), (" hand"," hands"), (" eye"," eyes"),
        (" horse"," horses"), (" man"," men"), (" woman"," women"),
        (" child"," children"), (" foot"," feet"), (" book"," books"),
        (" car"," cars"), (" house"," houses"), (" star"," stars"),
        (" word"," words"),
    ],
    "past_tense": [
        (" run"," ran"), (" walk"," walked"), (" jump"," jumped"),
        (" fly"," flew"), (" eat"," ate"), (" see"," saw"),
        (" build"," built"), (" write"," wrote"), (" break"," broke"),
        (" hear"," heard"), (" go"," went"), (" come"," came"),
        (" give"," gave"), (" find"," found"), (" make"," made"),
        (" fall"," fell"),
    ],
    "comparative": [
        (" fast"," faster"), (" slow"," slower"), (" big"," bigger"),
        (" small"," smaller"), (" hot"," hotter"), (" cold"," colder"),
        (" old"," older"), (" good"," better"), (" bad"," worse"),
        (" hard"," harder"), (" soft"," softer"), (" long"," longer"),
        (" short"," shorter"), (" dark"," darker"), (" bright"," brighter"),
        (" loud"," louder"),
    ],
}

# Ground truth traversal pairs
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

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def get_layers(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    hs  = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}
    lg  = out.logits[0, pos, :].numpy().astype(np.float32)
    return hs, lg

# ── Phase 1: Token-level coherence sweep over layers ─────────────────────────
print("=" * 72)
print("Phase 1: Token-level axis coherence by layer")
print("=" * 72)

tok_axes_by_layer = {}   # {axis_name: {layer: unit_vector}}
tok_coherence     = {}   # {axis_name: {layer: float}}

for axis_name, pairs in TOK_PAIRS.items():
    tok_axes_by_layer[axis_name] = {}
    tok_coherence[axis_name]     = {}
    print(f"\n  {axis_name}:")
    for L in TEST_LAYERS:
        diffs = []
        for s1, s2 in pairs:
            try:
                h1, _ = get_layers(s1, [L])
                h2, _ = get_layers(s2, [L])
                d = h2[L] - h1[L]; n = np.linalg.norm(d)
                if n > 1e-6: diffs.append(d / n)
            except: pass
        if len(diffs) < 2:
            tok_coherence[axis_name][L] = 0.0; continue
        # Pairwise cosines
        cosines = []
        for i in range(len(diffs)):
            for j in range(i+1, len(diffs)):
                cosines.append(float(np.dot(diffs[i], diffs[j])))
        coh = float(np.mean(cosines))
        tok_coherence[axis_name][L] = coh
        # Compute mean axis
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        tok_axes_by_layer[axis_name][L] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
        print(f"    L{L:>2}: coherence={coh:.3f}")

# Best layer per token-level axis
best_tok_layer = {name: max(tok_coherence[name], key=tok_coherence[name].get)
                  for name in TOK_PAIRS}
print(f"\n  Best layers: {best_tok_layer}")

# ── Phase 2: Build sentence-level axes (Day78 method) ─────────────────────────
print()
print("=" * 72)
print("Phase 2: Sentence-level axes (Day78 method)")
print("=" * 72)

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]

sent_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in SENT_PAIRS.get(name, []):
        try:
            h1, _ = get_layers(s1, [L])
            h2, _ = get_layers(s2, [L])
            d = h2[L] - h1[L]; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    if diffs:
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        sent_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    else:
        sent_axes[name] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {name:<15} L{L}")

# ── Phase 3: Extract probe token hidden states ────────────────────────────────
print()
print("Extracting probe token hidden states ...")
all_layers_needed = sorted(set(list(DAY78_LAYERS.values()) +
                               [best_tok_layer[n] for n in TOK_PAIRS]))
hs_by_layer = {L: [] for L in all_layers_needed}
logits_list = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        hs, lg = get_layers(" " + word.strip(), all_layers_needed)
        for L in all_layers_needed: hs_by_layer[L].append(hs[L])
        logits_list.append(lg); valid_words.append(word)
    except: pass
for L in all_layers_needed:
    hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
logits_arr = np.array(logits_list, dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens  layers={all_layers_needed}\n")

def classify_axis(axis_vec, layer_hs):
    if np.linalg.norm(axis_vec) < 1e-6: return ["U"] * N
    projs = [float(np.dot(layer_hs[i], axis_vec)) for i in range(N)]
    max_p = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * N
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

def hamming(a, b): return sum(x != y for x, y in zip(a, b))

def build_trie_addresses(axis_config):
    """axis_config: list of (name, axis_vec, layer)"""
    classes = {}
    for name, ax, L in axis_config:
        classes[name] = classify_axis(ax, hs_by_layer[L])
    return ["".join(classes[n][i] for n, _, _ in axis_config) for i in range(N)]

# ── Phase 4: Compare sentence-level vs hybrid tries ───────────────────────────
# Config A: original Day91 sentence-level 12D
config_sent = [(n, sent_axes[n], DAY78_LAYERS[n]) for n in AXIS_NAMES_12]

# Config B: hybrid — replace plural, past_tense, comparative with token-level
def hybrid_config(tok_axes_by_layer, best_tok_layer):
    replace = {"plural", "past_tense", "comparative"}
    cfg = []
    for n in AXIS_NAMES_12:
        if n in replace:
            L = best_tok_layer.get(n, DAY78_LAYERS[n])
            ax = tok_axes_by_layer[n].get(L, sent_axes[n])
        else:
            ax = sent_axes[n]; L = DAY78_LAYERS[n]
        cfg.append((n, ax, L))
    return cfg

config_hyb = hybrid_config(tok_axes_by_layer, best_tok_layer)

# LOO generative lookup
def generative_loo(addresses, max_r=5):
    results = {r: [] for r in range(max_r + 1)}
    global_cosims = []
    for i in range(N):
        actual = logits_arr[i]; na = np.linalg.norm(actual)
        others = [j for j in range(N) if j != i]
        gp = np.mean(logits_arr[others], axis=0)
        ng = np.linalg.norm(gp)
        if na > 1e-10 and ng > 1e-10:
            global_cosims.append(float(np.dot(actual, gp) / (na * ng)))
        for r in range(max_r + 1):
            nbrs = [(j, hamming(addresses[i], addresses[j]))
                    for j in others if hamming(addresses[i], addresses[j]) <= r]
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

# Address traversal
FLIP_MAP = {"H": "L", "L": "H", "U": "H"}
def traversal_test(addresses, gt_axis_pairs, axis_bit_idx):
    hits = 0; total = 0; details = []
    for src, tgt in gt_axis_pairs:
        if src not in word_idx or tgt not in word_idx: continue
        si = word_idx[src]
        flipped = list(addresses[si])
        flipped[axis_bit_idx] = FLIP_MAP[flipped[axis_bit_idx]]
        flipped = "".join(flipped)
        nearest = sorted([(j, hamming(flipped, addresses[j]))
                          for j in range(N) if j != si], key=lambda x: x[1])
        top5 = [valid_words[j] for j, _ in nearest[:5]]
        ti = word_idx[tgt]
        rank = next((k for k, (j, _) in enumerate(nearest) if j == ti), -1)
        hit = 0 <= rank < 5
        if hit: hits += 1
        total += 1
        details.append({"src": src, "tgt": tgt, "rank": rank, "top5": top5, "hit": hit})
    return hits, total, details

print("=" * 72)
print("Phase 4: LOO + traversal — sentence-level vs hybrid")
print("=" * 72)

all_results = {}
for cfg_name, cfg in [("sentence-level", config_sent), ("hybrid", config_hyb)]:
    addrs = build_trie_addresses(cfg)
    lc = Counter(addrs)
    n_leaves = len(lc); n_sing = sum(1 for c in lc.values() if c == 1)
    cov = 100 * (N - n_sing) / N

    loo, baseline = generative_loo(addrs, max_r=5)
    valid_loo = {r: v for r, v in loo.items() if v is not None}
    best_r   = max(valid_loo, key=valid_loo.get)
    best_loo = valid_loo[best_r]

    print(f"\n  {cfg_name:>20}: leaves={n_leaves} cov={cov:.1f}% "
          f"LOO={best_loo:.4f} r={best_r}")
    print(f"  {'r':>4}  LOO")
    for r in range(6):
        v = loo.get(r)
        print(f"  {r:>4}  {v:.4f}" if v else f"  {r:>4}  —")

    # Traversal for this config
    print(f"\n  Traversal ({cfg_name}):")
    trav_hits_total = 0; trav_total_total = 0
    trav_results = {}
    for axis_name, gt_pairs in GROUND_TRUTH.items():
        if axis_name not in AXIS_NAMES_12: continue
        bit_idx = AXIS_NAMES_12.index(axis_name)
        h, t, det = traversal_test(addrs, gt_pairs, bit_idx)
        trav_hits_total += h; trav_total_total += t
        trav_results[axis_name] = {"hits": h, "total": t}
        print(f"    {axis_name:>15}: {h}/{t} ({100*h/max(1,t):.0f}%)")
    print(f"    {'OVERALL':>15}: {trav_hits_total}/{trav_total_total} "
          f"({100*trav_hits_total/max(1,trav_total_total):.0f}%)")

    all_results[cfg_name] = {
        "n_leaves": n_leaves, "coverage": cov, "loo": loo,
        "best_r": best_r, "best_loo": best_loo, "baseline": baseline,
        "traversal": trav_results,
        "trav_total_hits": trav_hits_total,
        "trav_total": trav_total_total,
    }

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 94 Summary")
print("=" * 72)
print(f"""
  TOKEN-LEVEL AXIS COHERENCE BY LAYER:
  {'axis':>15}  {'L1':>6}  {'L8':>6}  {'L15':>6}  {'L22':>6}  {'L27':>6}  {'L28':>6}  best""")
for name in TOK_PAIRS:
    row = f"  {name:>15}  "
    for L in TEST_LAYERS:
        v = tok_coherence[name].get(L, 0)
        row += f"{v:>6.3f}  "
    row += f"L{best_tok_layer[name]}"
    print(row)

print(f"""
  COMPARISON:
                        LOO(best_r)   traversal
  sentence-level:  {all_results['sentence-level']['best_loo']:.4f} r={all_results['sentence-level']['best_r']}     {all_results['sentence-level']['trav_total_hits']}/{all_results['sentence-level']['trav_total']} ({100*all_results['sentence-level']['trav_total_hits']/max(1,all_results['sentence-level']['trav_total']):.0f}%)
  hybrid:          {all_results['hybrid']['best_loo']:.4f} r={all_results['hybrid']['best_r']}     {all_results['hybrid']['trav_total_hits']}/{all_results['hybrid']['trav_total']} ({100*all_results['hybrid']['trav_total_hits']/max(1,all_results['hybrid']['trav_total']):.0f}%)
  Day 77 baseline: 0.9303 r=3

  {'hybrid > sentence-level' if all_results['hybrid']['best_loo'] > all_results['sentence-level']['best_loo'] else 'sentence-level >= hybrid'} in LOO
  {'hybrid > sentence-level' if all_results['hybrid']['trav_total_hits'] > all_results['sentence-level']['trav_total_hits'] else 'sentence-level >= hybrid'} in traversal
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "tok_coherence": tok_coherence,
        "best_tok_layers": best_tok_layer,
        "results": all_results,
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 94 complete.")
