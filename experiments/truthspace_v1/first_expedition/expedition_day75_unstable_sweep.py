#!/usr/bin/env python3
"""
Day 75 — UNSTABLE-Zone Archaeology

Day 74 revealed: sentence-level T2 axes collapse all English tokens
to 2 leaves (no within-English discrimination). The comparative axis
in Day 70 was special: INTERMEDIATE context (~7 tokens) at L27 left
concrete nouns UNSTABLE, splitting English into two meaningful groups.

This experiment systematically finds ALL such "semantic decision points":
triplets (axis, layer, context_length) that maximize the within-English
UNSTABLE fraction.

Setup:
  - 8 transformation axes
  - 3 context length variants: SHORT (~4 tok), MEDIUM (~7 tok), LONG (~10 tok)
  - ALL 28 layers (captured in one pass per sentence, then sweep analytically)
  - 164 probe tokens

Output:
  1. Heatmap: within-English UNSTABLE fraction for every (axis, layer, length)
  2. Top-20 semantic decision points ranked by within-English UNSTABLE fraction
  3. Semantic inspection of top-10: what word classes land in H / L / U?
  4. Rich multi-axis trie built from optimal decision points: within-leaf separation

PREDICTION:
  - Plural axis: UNSTABLE for uncountable nouns (water, air, sand) at some layer
  - Tense axis: UNSTABLE for deverbal nouns (run, walk, talk) at some layer
  - Gender axis: UNSTABLE for gender-neutral nouns (child, parent) at some layer
  - Antonym axis: UNSTABLE for gradient adjectives (warm, damp) at some layer
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day75_unstable_sweep.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI         # 0.618
INV_PHI2 = 1 / PHI**2      # 0.382

N_LAYERS = 29               # 0..28
PROBE_LAYERS = list(range(N_LAYERS))

# ── Three context lengths per transformation type ─────────────────────────────
# Each triplet: (short_pair, medium_pair, long_pair)
# SHORT  ≈ 4 tokens   MEDIUM ≈ 7 tokens   LONG ≈ 10 tokens

SHORT_PAIRS = {
    "comparative": [
        ("The fast car",      "The faster car"),
        ("A big dog",         "A bigger dog"),
        ("The cold wind",     "The colder wind"),
        ("A tall tree",       "A taller tree"),
        ("The old house",     "The older house"),
        ("A bright star",     "A brighter star"),
        ("The dark room",     "The darker room"),
        ("A hard rock",       "A harder rock"),
    ],
    "plural": [
        ("I have a dog",      "I have dogs"),
        ("I see a cat",       "I see cats"),
        ("A bird sang",       "Birds sang"),
        ("The tree fell",     "The trees fell"),
        ("A book sat there",  "Books sat there"),
        ("The car drove",     "The cars drove"),
        ("A star shone",      "Stars shone"),
        ("The word means",    "The words mean"),
    ],
    "past_tense": [
        ("I walk home",       "I walked home"),
        ("She runs fast",     "She ran fast"),
        ("He eats now",       "He ate then"),
        ("They build it",     "They built it"),
        ("We swim here",      "We swam here"),
        ("She writes well",   "She wrote well"),
        ("He speaks softly",  "He spoke softly"),
        ("They sing loudly",  "They sang loudly"),
    ],
    "gender": [
        ("The king rules",    "The queen rules"),
        ("A man walked",      "A woman walked"),
        ("The boy played",    "The girl played"),
        ("His brother came",  "His sister came"),
        ("The father works",  "The mother works"),
        ("A son was born",    "A daughter was born"),
        ("The prince rode",   "The princess rode"),
        ("The actor left",    "The actress left"),
    ],
    "antonym": [
        ("It is hot",         "It is cold"),
        ("He runs fast",      "He runs slow"),
        ("The light is on",   "The dark is on"),
        ("The news is good",  "The news is bad"),
        ("It is hard",        "It is soft"),
        ("She is happy",      "She is sad"),
        ("He is strong",      "He is weak"),
        ("It is the first",   "It is the last"),
    ],
    "hypernym": [
        ("The dog ran",       "The animal ran"),
        ("A rose bloomed",    "A flower bloomed"),
        ("The oak fell",      "The tree fell"),
        ("The car sped",      "The vehicle sped"),
        ("The eagle soared",  "The bird soared"),
        ("The ruby shone",    "The gem shone"),
        ("The soldier came",  "The person came"),
        ("The hammer struck", "The tool struck"),
    ],
    "synonym": [
        ("He is big",         "He is large"),
        ("She is small",      "She is tiny"),
        ("He runs fast",      "He runs quick"),
        ("It is cold",        "It is frigid"),
        ("She is happy",      "She is joyful"),
        ("He spoke loudly",   "He spoke noisily"),
        ("It is hard",        "It is difficult"),
        ("He is old",         "He is aged"),
    ],
    "concrete_abstract": [
        ("The stone falls",   "The burden falls"),
        ("The fire burns",    "The anger burns"),
        ("The chain broke",   "The bond broke"),
        ("The road is long",  "The path is long"),
        ("The wall is high",  "The barrier is high"),
        ("The light fades",   "The hope fades"),
        ("The root holds",    "The base holds"),
        ("The key opens",     "The answer opens"),
    ],
}

MEDIUM_PAIRS = {
    "comparative": [
        ("The fast car won the race",       "The faster car won the race"),
        ("The big dog barked at strangers", "The bigger dog barked at strangers"),
        ("A cold wind swept the valley",    "A colder wind swept the valley"),
        ("The tall tree swayed in wind",    "The taller tree swayed in wind"),
        ("The old bridge still stands",     "The older bridge still stands"),
        ("A bright star lit the night",     "A brighter star lit the night"),
        ("The dark room hid the shape",     "The darker room hid the shape"),
        ("A hard problem took long time",   "A harder problem took long time"),
    ],
    "plural": [
        ("A dog played in the yard",       "Dogs played in the yard"),
        ("The cat sat by the window",      "The cats sat by the window"),
        ("A bird sang in the morning",     "Birds sang in the morning"),
        ("The tree fell in the storm",     "The trees fell in the storm"),
        ("A book sat on the table",        "Books sat on the table"),
        ("The car drove down the road",    "The cars drove down the road"),
        ("A star shone in the dark sky",   "Stars shone in the dark sky"),
        ("The word means many things",     "The words mean many things"),
    ],
    "past_tense": [
        ("I walk to the store each day",   "I walked to the store each day"),
        ("She runs every morning before work","She ran every morning before work"),
        ("He eats breakfast at the table", "He ate breakfast at the table"),
        ("They build houses for people",   "They built houses for people"),
        ("We swim in the lake on weekends","We swam in the lake on weekends"),
        ("She writes letters to her friend","She wrote letters to her friend"),
        ("He speaks quietly in the room",  "He spoke quietly in the room"),
        ("They sing songs at the campfire","They sang songs at the campfire"),
    ],
    "gender": [
        ("The king ruled with great wisdom",  "The queen ruled with great wisdom"),
        ("A man walked through the forest",   "A woman walked through the forest"),
        ("The boy kicked the ball hard",      "The girl kicked the ball hard"),
        ("His brother arrived at the party",  "His sister arrived at the party"),
        ("The father worked to feed family",  "The mother worked to feed family"),
        ("A son was born in the winter",      "A daughter was born in the winter"),
        ("The prince rode across the land",   "The princess rode across the land"),
        ("The actor played a leading role",   "The actress played a leading role"),
    ],
    "antonym": [
        ("It is very hot outside today",   "It is very cold outside today"),
        ("He drives fast on the highway",  "He drives slow on the highway"),
        ("The room was bright all day",    "The room was dark all day"),
        ("She gave a good answer back",    "She gave a bad answer back"),
        ("The surface felt very hard",     "The surface felt very soft"),
        ("She was extremely happy then",   "She was extremely sad then"),
        ("The old man was very strong",    "The old man was very weak"),
        ("It was the very first time",     "It was the very last time"),
    ],
    "hypernym": [
        ("The dog ran away from danger",   "The animal ran away from danger"),
        ("A rose bloomed in the garden",   "A flower bloomed in the garden"),
        ("The oak crashed in the storm",   "The tree crashed in the storm"),
        ("The car sped past the sign",     "The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light",  "The gem gleamed in the light"),
        ("The soldier marched into fight", "The person marched into fight"),
        ("The hammer struck the nail",     "The tool struck the nail"),
    ],
    "synonym": [
        ("He is an extremely big man",     "He is an extremely large man"),
        ("She is a very small creature",   "She is a very tiny creature"),
        ("He runs incredibly fast today",  "He runs incredibly quick today"),
        ("The air feels terribly cold now","The air feels terribly frigid now"),
        ("She was incredibly happy there", "She was incredibly joyful there"),
        ("He always speaks very loudly",   "He always speaks very noisily"),
        ("This task is extremely hard",    "This task is extremely difficult"),
        ("The professor was quite old",    "The professor was quite aged"),
    ],
    "concrete_abstract": [
        ("The stone is too heavy to lift", "The burden is too heavy to lift"),
        ("The iron chain has broken now",  "The bond between them has broken"),
        ("The long road leads to the sea", "The long journey leads to the sea"),
        ("The high wall blocks the view",  "The high barrier blocks the view"),
        ("The flame slowly fades away",    "The hope slowly fades away"),
        ("The strong root grips the soil", "The strong base grips the earth"),
        ("The bridge connects two banks",  "The bond connects two communities"),
        ("The small key opens the door",   "The small answer opens the path"),
    ],
}

LONG_PAIRS = {
    "comparative": [
        ("The fast runner reached the finish line today",  "The faster runner reached the finish line today"),
        ("A big wave crashed against the rocky shore",    "A bigger wave crashed against the rocky shore"),
        ("The cold morning air made her shiver badly",    "The colder morning air made her shiver badly"),
        ("A tall mountain cast a very long shadow",       "A taller mountain cast a very long shadow"),
        ("The old bridge swayed in the strong winter wind","The older bridge swayed in the strong winter wind"),
        ("A bright flame burned through many long hours", "A brighter flame burned through many long hours"),
        ("The dark valley hid the path from all sight",   "The darker valley hid the path from all sight"),
        ("A hard exam question appeared at the very end", "A harder exam question appeared at the very end"),
    ],
    "plural": [
        ("A dog played happily in the open green field",     "Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window",  "The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist",     "Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm",    "The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk",           "Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road",    "The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky",      "Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text",    "The words appeared clearly in the printed text"),
    ],
    "past_tense": [
        ("I walk to the market every single morning",     "I walked to the market every single morning"),
        ("She runs through the park after her long work", "She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden",     "They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days",       "We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend",    "She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire","They sang together around the evening campfire"),
    ],
    "gender": [
        ("The king ruled his kingdom with great wisdom",    "The queen ruled her kingdom with great wisdom"),
        ("A man walked alone through the quiet dark forest","A woman walked alone through the quiet dark forest"),
        ("The boy kicked the ball across the green field",  "The girl kicked the ball across the green field"),
        ("His brother arrived late to the big family dinner","His sister arrived late to the big family dinner"),
        ("The father worked hard to support the whole family","The mother worked hard to support the whole family"),
        ("A son was born on that cold and bitter morning",  "A daughter was born on that cold and bitter morning"),
        ("The prince rode his horse through the small village","The princess rode her horse through the small village"),
        ("The actor received a well-deserved standing ovation","The actress received a well-deserved standing ovation"),
    ],
    "antonym": [
        ("The weather is extremely hot and very humid today","The weather is extremely cold and very humid today"),
        ("He drives very fast on the long empty highway",   "He drives very slow on the long empty highway"),
        ("The room was completely bright from the warm sun","The room was completely dark from the warm sun"),
        ("She gave a very good answer to the hard question","She gave a very bad answer to the hard question"),
        ("The old stone surface felt very hard to the touch","The old stone surface felt very soft to the touch"),
        ("She was extremely happy about the wonderful news","She was extremely sad about the wonderful news"),
        ("The old man by the river was incredibly strong",  "The old man by the river was incredibly weak"),
        ("He was always among the very first to arrive",    "He was always among the very last to arrive"),
    ],
    "hypernym": [
        ("The dog ran far away from the busy city park",     "The animal ran far away from the busy city park"),
        ("A rose bloomed beautifully in the spring garden",  "A flower bloomed beautifully in the spring garden"),
        ("The oak crashed loudly down in the heavy storm",   "The tree crashed loudly down in the heavy storm"),
        ("The car sped away quickly down the long highway",  "The vehicle sped away quickly down the long highway"),
        ("The eagle soared very high above the tall mountain","The bird soared very high above the tall mountain"),
        ("The ruby gleamed deep red in the bright candlelight","The gem gleamed deep red in the bright candlelight"),
        ("The soldier marched bravely forward into battle",  "The person marched bravely forward into battle"),
        ("The hammer struck the nail hard and precisely",    "The tool struck the nail hard and precisely"),
    ],
    "synonym": [
        ("He is an extremely big and very powerful animal",  "He is an extremely large and very powerful animal"),
        ("She is a very small and quite delicate creature",  "She is a very tiny and quite delicate creature"),
        ("He runs incredibly fast across the open field",    "He runs incredibly quick across the open field"),
        ("The air outside feels terribly cold on this day",  "The air outside feels terribly frigid on this day"),
        ("She was incredibly happy about the final result",  "She was incredibly joyful about the final result"),
        ("He always speaks very loudly when in the room",    "He always speaks very noisily when in the room"),
        ("This difficult problem is extremely hard to solve","This difficult problem is extremely hard to solve"),
        ("The old professor was quite old and very wise",    "The old professor was quite aged and very wise"),
    ],
    "concrete_abstract": [
        ("The stone block is far too heavy for me to move",  "The burden is far too heavy for me to move"),
        ("The iron chain between them is completely broken", "The bond between them is completely broken"),
        ("The long road stretched far into the dark horizon","The long journey stretched far into the dark horizon"),
        ("The high wall blocked all the bright warm light",  "The high barrier blocked all the bright warm light"),
        ("The bright candle slowly faded out in the dark",   "The bright hope slowly faded out in the dark"),
        ("The strong root held firm in the very hard soil",  "The strong foundation held firm in the earth"),
        ("The open bridge connects the two distant banks",   "The bond connects the two distant communities"),
        ("The small key opens the most important old door",  "The small answer opens the most important path"),
    ],
}

ALL_PAIRS = {"short": SHORT_PAIRS, "medium": MEDIUM_PAIRS, "long": LONG_PAIRS}
TYPE_NAMES = list(SHORT_PAIRS.keys())
LENGTH_NAMES = ["short", "medium", "long"]

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

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
print(f"  hidden={model.config.hidden_size}\n")

def get_all_layers_last(text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    # (29, hidden)
    return np.stack([out.hidden_states[l][0, pos, :].numpy().astype(np.float32)
                     for l in range(N_LAYERS)])

def get_logits(word):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].numpy().astype(np.float32)

# ── Build T2 axes for all (type, length, layer) ──────────────────────────────
print("Building T2 axes for 8 types × 3 lengths × 28 layers ...")
t2_axes = defaultdict(lambda: defaultdict(dict))
# t2_axes[type_name][length_name][layer] = unit vector (hidden,)

for length_name, pairs_dict in ALL_PAIRS.items():
    for type_name, pairs in pairs_dict.items():
        # Accumulate diffs per layer
        diffs_by_layer = defaultdict(list)
        for s1, s2 in pairs:
            h1 = get_all_layers_last(s1)   # (29, d)
            h2 = get_all_layers_last(s2)
            diff = h2 - h1                  # (29, d)
            for l in range(N_LAYERS):
                d = diff[l]; n = np.linalg.norm(d)
                if n > 1e-6: diffs_by_layer[l].append(d / n)
        hidden_size = model.config.hidden_size
        for l in range(N_LAYERS):
            if not diffs_by_layer[l]:
                t2_axes[type_name][length_name][l] = np.zeros(hidden_size, dtype=np.float32)
                continue
            v = np.mean(diffs_by_layer[l], axis=0)
            nv = np.linalg.norm(v)
            t2_axes[type_name][length_name][l] = (v / nv if nv > 1e-6 else np.zeros(hidden_size, dtype=np.float32)).astype(np.float32)
        print(f"  {type_name}/{length_name}: done")

print()

# ── Collect probe token hidden states and logits ─────────────────────────────
print(f"Computing hidden states for {len(PROBE_TOKENS)} probe tokens ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        all_h  = get_all_layers_last(" " + word.strip())   # (29, d)
        logits = get_logits(word)
        tokens_data[word] = {"hs": all_h, "logits": logits}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words         = list(tokens_data.keys())
english_words = [w for w in words if w not in ENGLISH_SKIP]
logit_vecs    = {w: tokens_data[w]["logits"] for w in words}
print(f"  Collected {len(words)} tokens ({len(english_words)} English)\n")

# ── Per-(type, length, layer): classify and measure within-English UNSTABLE ──
print("Sweeping UNSTABLE fractions across all (type, length, layer) ...")
print()

results = {}      # results[(type, length, layer)] = {n_h, n_l, n_u, u_frac_eng}

for type_name in TYPE_NAMES:
    for length_name in LENGTH_NAMES:
        for layer in range(N_LAYERS):
            axis = t2_axes[type_name][length_name][layer]
            if np.linalg.norm(axis) < 1e-6:
                continue   # degenerate layer (e.g. layer 0 with same last token)
            projs = np.array([float(np.dot(tokens_data[w]["hs"][layer], axis))
                              for w in words])
            max_p = np.percentile(projs, 95)
            hi    = max_p * INV_PHI
            lo    = max_p * INV_PHI2

            # classify all tokens
            classes = {}
            for i, w in enumerate(words):
                p = projs[i]
                if p > hi:   classes[w] = "H"
                elif p < lo: classes[w] = "L"
                else:        classes[w] = "U"

            # within-English UNSTABLE fraction
            eng_classes = [classes[w] for w in english_words]
            n_h = eng_classes.count("H")
            n_l = eng_classes.count("L")
            n_u = eng_classes.count("U")
            u_frac = n_u / len(english_words)

            results[(type_name, length_name, layer)] = {
                "n_h": n_h, "n_l": n_l, "n_u": n_u,
                "u_frac_eng": u_frac,
                "classes": {w: classes[w] for w in words},
            }

print("Sweep complete.\n")

# ── Top-20 decision points by within-English UNSTABLE fraction ───────────────
sorted_results = sorted(results.items(), key=lambda x: -x[1]["u_frac_eng"])

print("=" * 72)
print("Top-20 semantic decision points (by within-English UNSTABLE fraction)")
print("=" * 72)
print(f"  {'axis':>20}  {'len':>6}  L  n_H  n_L  n_U  u_frac")
for (type_n, length_n, layer), v in sorted_results[:20]:
    print(f"  {type_n:>20}  {length_n:>6}  {layer:>2}  "
          f"{v['n_h']:>4}  {v['n_l']:>4}  {v['n_u']:>4}  {v['u_frac_eng']:.3f}")
print()

# ── Semantic inspection: top-10 ───────────────────────────────────────────────
print("=" * 72)
print("Semantic inspection: what's in H / L / U for top-10?")
print("=" * 72)
for (type_n, length_n, layer), v in sorted_results[:10]:
    classes = v["classes"]
    h_words = [w for w in english_words if classes[w] == "H"]
    l_words = [w for w in english_words if classes[w] == "L"]
    u_words = [w for w in english_words if classes[w] == "U"]
    print(f"\n  [{type_n}/{length_n}/L{layer}]  u_frac={v['u_frac_eng']:.3f}")
    print(f"  H ({len(h_words)}): {' '.join(h_words[:20])}")
    print(f"  L ({len(l_words)}): {' '.join(l_words[:20])}")
    print(f"  U ({len(u_words)}): {' '.join(u_words[:20])}")
print()

# ── Find best per-axis decision point (one per type, maximizing u_frac) ──────
print("=" * 72)
print("Best decision point per axis (max within-English U fraction)")
print("=" * 72)
best_per_axis = {}
for type_n in TYPE_NAMES:
    candidates = [(k, v) for k, v in sorted_results if k[0] == type_n]
    best_key, best_val = candidates[0]   # already sorted by u_frac
    best_per_axis[type_n] = (best_key, best_val)
    print(f"  {type_n:>20}: {best_key[1]:>6}/L{best_key[2]:>2}  "
          f"u_frac={best_val['u_frac_eng']:.3f}  "
          f"H={best_val['n_h']:>3} L={best_val['n_l']:>3} U={best_val['n_u']:>3}")
print()

# ── Build rich multi-axis trie from top-k decision points ────────────────────
all_pairs = [(words[i], words[j])
             for i in range(len(words))
             for j in range(i + 1, len(words))]

print("=" * 72)
print("Rich φ-trie: using top-k semantic decision points (varied axes/layers)")
print("=" * 72)
print()

trie_results = {}

for top_k in [4, 8, 16, 32]:
    # Take top-k decision points with DIVERSITY:
    # at most 2 per axis, at most 3 per layer, prefer unique (axis, layer) combinations
    selected = []
    axis_count    = Counter()
    layer_count   = Counter()
    used_al_pairs = set()

    for (type_n, length_n, layer), v in sorted_results:
        if len(selected) >= top_k: break
        if axis_count[type_n] >= 2: continue
        if layer_count[layer] >= 3: continue
        if (type_n, layer) in used_al_pairs: continue
        if v["u_frac_eng"] < 0.05: break   # stop once u_frac too small
        selected.append(((type_n, length_n, layer), v))
        axis_count[type_n]    += 1
        layer_count[layer]    += 1
        used_al_pairs.add((type_n, layer))

    n_bits = len(selected)
    if n_bits == 0:
        print(f"  top-{top_k}: no decision points with u_frac>=0.05, skip")
        continue

    # Build leaf paths
    paths = {}
    for w in words:
        bits = []
        for (type_n, length_n, layer), v in selected:
            bits.append(v["classes"][w])
        paths[w] = "".join(bits)

    path_counts = Counter(paths.values())

    same_s, diff_s = [], []
    for (w1, w2) in all_pairs:
        if "U" in paths[w1] or "U" in paths[w2]: continue
        sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
        if paths[w1] == paths[w2]: same_s.append(sim)
        else:                       diff_s.append(sim)

    same_m = float(np.mean(same_s)) if same_s else float("nan")
    diff_m = float(np.mean(diff_s)) if diff_s else float("nan")
    sep    = same_m - diff_m
    n_lv   = len(path_counts)

    trie_results[top_k] = {
        "n_bits": n_bits, "same_mean": same_m, "diff_mean": diff_m,
        "separation": sep, "n_leaves": n_lv,
        "n_same_pairs": len(same_s), "n_diff_pairs": len(diff_s),
        "selected_axes": [(type_n, length_n, layer) for (type_n, length_n, layer), _ in selected],
    }

    # Which axes/layers were selected?
    sel_str = " | ".join(f"{t}/{ln}/L{l}" for (t, ln, l), _ in selected[:4])
    if len(selected) > 4: sel_str += " ..."
    verdict = "CONFIRMED" if sep > 0.05 else "WEAK" if sep > 0.0 else "FAILED"
    print(f"  top-{top_k:>2} ({n_bits:>2} bits)  sep={sep:+.4f}  leaves={n_lv}  {verdict}")
    print(f"    axes: {sel_str}")
    top_leaves = [f"{p[:12]}({c})" for p, c in path_counts.most_common(4)]
    print(f"    top leaves: {', '.join(top_leaves)}")
    print()

# ── Leaf content for the best rich trie (top-8) ──────────────────────────────
if 8 in trie_results:
    print("=" * 72)
    print("Leaf contents: top-8 rich trie")
    print("=" * 72)

    selected_8 = []
    axis_count  = Counter(); layer_count = Counter(); used = set()
    for (type_n, length_n, layer), v in sorted_results:
        if len(selected_8) >= 8: break
        if axis_count[type_n] >= 2: continue
        if layer_count[layer] >= 3: continue
        if (type_n, layer) in used: continue
        if v["u_frac_eng"] < 0.05: break
        selected_8.append(((type_n, length_n, layer), v))
        axis_count[type_n] += 1; layer_count[layer] += 1; used.add((type_n, layer))

    paths_8 = {}
    for w in words:
        bits = [v["classes"][w] for (_, _, _), v in selected_8]
        paths_8[w] = "".join(bits)
    for path, count in sorted(Counter(paths_8.values()).items(), key=lambda x: -x[1]):
        if count < 2: continue
        tokens_here = [w for w in words if paths_8[w] == path]
        print(f"  [{path}] ({count}): {' '.join(tokens_here[:24])}")
    print()

# ── UNSTABLE fraction heatmap summary (best layer per axis×length) ────────────
print("=" * 72)
print("Best layer per (axis, length) — max within-English UNSTABLE fraction")
print("=" * 72)
print(f"  {'axis':>20}  {'short':>10}  {'medium':>10}  {'long':>10}")
for type_n in TYPE_NAMES:
    row = []
    for ln in LENGTH_NAMES:
        candidates = [(k[2], v["u_frac_eng"]) for k, v in results.items()
                      if k[0] == type_n and k[1] == ln]
        best_layer, best_frac = max(candidates, key=lambda x: x[1])
        row.append(f"L{best_layer:>2} {best_frac:.3f}")
    print(f"  {type_n:>20}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}")
print()

# ── Final summary ─────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 75 Summary")
print("=" * 72)
best_trie = max(trie_results.values(), key=lambda x: x["separation"]) if trie_results else None
if best_trie:
    print(f"  Best rich trie:  sep={best_trie['separation']:+.4f}  "
          f"leaves={best_trie['n_leaves']}  bits={best_trie['n_bits']}")
print(f"  Day 70 baseline: sep=+0.4548  leaves=4  bits=4")
print(f"  Day 73 diff-PCA: sep=+0.5127  leaves=158  bits=32")
print()
print("  Top UNSTABLE-zone decision points:")
for (type_n, length_n, layer), v in sorted_results[:5]:
    print(f"    {type_n}/{length_n}/L{layer}  u_frac={v['u_frac_eng']:.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "trie_results": trie_results,
    "top20_decision_points": [
        {"key": list(k), "u_frac_eng": v["u_frac_eng"],
         "n_h": v["n_h"], "n_l": v["n_l"], "n_u": v["n_u"]}
        for k, v in sorted_results[:20]
    ],
    "best_per_axis": {
        type_n: {"key": list(best_key), "u_frac_eng": best_val["u_frac_eng"]}
        for type_n, (best_key, best_val) in best_per_axis.items()
    },
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 75 complete.")
