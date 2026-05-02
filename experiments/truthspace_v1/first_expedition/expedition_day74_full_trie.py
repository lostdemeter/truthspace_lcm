#!/usr/bin/env python3
"""
Day 74 — The Complete φ-Trie: 8 Full-Sentence T2 Axes

Day 73 found that:
  1. All 8 semantic transformation types are mutually orthogonal (80-90°)
  2. Diff-PCA with 8 axes (32 bits) reaches +0.5127 using short phrase pairs
  3. Manual T2 is 7.5× stronger per axis because it uses full sentence pairs

This experiment builds the COMPLETE φ-trie using full-sentence T2 pairs
for all 8 transformation types. If sentence-level quality gives each axis
~0.45 separation (as the comparative axis did), combining 8 orthogonal axes
should produce dramatically tighter semantic clusters than any single axis.

PREDICTIONS:
  1. Each of the 8 sentence-level T2 axes gives separation > +0.10 per axis
  2. With all 8 axes (32 bits), separation >> +0.5127 (exceeds Day 73)
  3. The within-leaf semantic coherence approaches 1.0 for large clusters
  4. Pairwise T2 angles (sentence-level) are still ~85-90° → truly orthogonal

The complete φ-trie should give each token a unique 32-bit semantic address
with extremely tight within-leaf clustering.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day74_full_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2
TRIE_LAYERS = [5, 14, 22, 27]

# ── Full-sentence T2 pairs for 8 transformation types ───────────────────────
# Each pair ends with the SAME word so the last-token hidden state captures
# the full-sentence transformation effect.
FULL_SENTENCE_PAIRS = {
    "comparative": [
        ("The fast runner reached the finish line",   "The faster runner reached the finish line"),
        ("A big wave crashed against the shore",      "A bigger wave crashed against the shore"),
        ("The cold morning made her shiver badly",    "The colder morning made her shiver badly"),
        ("A small light flickered in the darkness",   "A smaller light flickered in the darkness"),
        ("The old bridge swayed in the strong wind",  "The older bridge swayed in the strong wind"),
        ("A bright flame burned for many hours",      "A brighter flame burned for many hours"),
        ("The tall mountain cast a long shadow",      "The taller mountain cast a long shadow"),
        ("A hard question appeared on the exam",      "A harder question appeared on the exam"),
        ("The young bird learned to fly alone",       "The younger bird learned to fly alone"),
        ("A soft sound echoed in the empty room",     "A softer sound echoed in the empty room"),
    ],
    "plural": [
        ("A dog played happily in the open field",    "Dogs played happily in the open field"),
        ("The bird sang softly in the morning mist",  "The birds sang softly in the morning mist"),
        ("A tree fell down in the heavy storm",       "Trees fell down in the heavy storm"),
        ("The child laughed loudly at the joke",      "The children laughed loudly at the joke"),
        ("A book sat open on the wooden desk",        "Books sat open on the wooden desk"),
        ("The car drove slowly down the long road",   "The cars drove slowly down the long road"),
        ("A star shone brightly in the clear sky",    "Stars shone brightly in the clear sky"),
        ("The flower bloomed early in the spring",    "The flowers bloomed early in the spring"),
        ("A key opened the old rusty lock",           "Keys opened the old rusty lock"),
        ("The word appeared clearly in the text",     "The words appeared clearly in the text"),
    ],
    "past_tense": [
        ("I walk to the market every single morning", "I walked to the market every single morning"),
        ("She runs through the park after her work",  "She ran through the park after her work"),
        ("He eats breakfast before leaving the house","He ate breakfast before leaving the house"),
        ("They build a wall around the entire garden","They built a wall around the entire garden"),
        ("We swim in the lake on warm summer days",   "We swam in the lake on warm summer days"),
        ("She writes a letter to her old friend",     "She wrote a letter to her old friend"),
        ("He speaks quietly during the long meeting", "He spoke quietly during the long meeting"),
        ("They sing together around the campfire",    "They sang together around the campfire"),
        ("I think about this problem every single day","I thought about this problem every single day"),
        ("She feels nervous before each performance", "She felt nervous before each performance"),
    ],
    "gender": [
        ("The king ruled his kingdom with great wisdom",  "The queen ruled her kingdom with great wisdom"),
        ("A man walked alone through the quiet forest",   "A woman walked alone through the quiet forest"),
        ("The boy kicked the ball across the green field","The girl kicked the ball across the green field"),
        ("His brother arrived late to the family dinner", "His sister arrived late to the family dinner"),
        ("The father worked hard to support his family",  "The mother worked hard to support her family"),
        ("A son was born on that cold winter morning",    "A daughter was born on that cold winter morning"),
        ("The uncle told stories to the young children",  "The aunt told stories to the young children"),
        ("The prince rode his horse through the village", "The princess rode her horse through the village"),
        ("A husband cooked dinner for his tired wife",    "A wife cooked dinner for her tired husband"),
        ("The actor received a great standing ovation",   "The actress received a great standing ovation"),
    ],
    "antonym": [
        ("The weather is extremely hot and very humid",  "The weather is extremely cold and very humid"),
        ("He drives very fast on the empty highway",     "He drives very slow on the empty highway"),
        ("The room was completely bright from the sun",  "The room was completely dark from the sun"),
        ("She gave a very good answer to the question",  "She gave a very bad answer to the question"),
        ("The surface felt very hard under his fingers", "The surface felt very soft under his fingers"),
        ("She was extremely happy about the good news",  "She was extremely sad about the good news"),
        ("The old man was incredibly strong and fit",    "The old man was incredibly weak and frail"),
        ("The mountain trail was very long and steep",   "The mountain trail was very short and steep"),
        ("The sun was burning very hot in the sky",      "The wind was blowing very cold in the sky"),
        ("He was among the very first to arrive",        "He was among the very last to arrive"),
    ],
    "hypernym": [
        ("The dog ran far away from the busy park",    "The animal ran far away from the busy park"),
        ("A rose bloomed beautifully in the garden",   "A flower bloomed beautifully in the garden"),
        ("The oak crashed down in the heavy storm",    "The tree crashed down in the heavy storm"),
        ("The car sped away down the long highway",    "The vehicle sped away down the long highway"),
        ("A salmon swam upstream in the cold river",   "A fish swam upstream in the cold river"),
        ("The eagle soared high above the mountain",   "The bird soared high above the mountain"),
        ("The ruby gleamed red in the bright light",   "The gem gleamed red in the bright light"),
        ("The piano filled the hall with sweet music", "The instrument filled the hall with sweet music"),
        ("The soldier marched forward into the fight", "The person marched forward into the fight"),
        ("The hammer struck the nail very precisely",  "The tool struck the nail very precisely"),
    ],
    "synonym": [
        ("He is an extremely big and powerful animal", "He is an extremely large and powerful animal"),
        ("She is a very small and delicate creature",  "She is a very tiny and delicate creature"),
        ("He runs incredibly fast across the field",   "He runs incredibly quick across the field"),
        ("The air outside feels terribly cold today",  "The air outside feels terribly frigid today"),
        ("She was incredibly happy about the result",  "She was incredibly joyful about the result"),
        ("He always speaks very loudly in the room",   "He always speaks very noisily in the room"),
        ("This problem is extremely hard to resolve",  "This problem is extremely difficult to resolve"),
        ("She always moves so slowly in the morning",  "She always moves so gradually in the morning"),
        ("The old professor was quite old and wise",   "The old professor was quite aged and wise"),
        ("The painting looks absolutely beautiful here","The painting looks absolutely gorgeous here"),
    ],
    "concrete_abstract": [
        ("The stone block is far too heavy to move",  "The burden is far too heavy to move"),
        ("The iron chain is completely broken apart",  "The bond is completely broken apart"),
        ("The long road stretched far into the horizon","The long journey stretched far into the horizon"),
        ("The high wall blocked all the bright light", "The high barrier blocked all the bright light"),
        ("The deep river flowed quietly to the sea",   "The deep sorrow flowed quietly to the sea"),
        ("The bright candle slowly faded in the dark", "The bright hope slowly faded in the dark"),
        ("The strong root held firm in the hard soil", "The strong foundation held firm in the hard soil"),
        ("The open bridge connects the two far banks", "The bond connects the two communities together"),
        ("The small key opens a very important door",  "The small answer opens a very important path"),
        ("The cold rain soaked everything to the bone","The cold fear soaked everything to the bone"),
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

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
print(f"  hidden={model.config.hidden_size}\n")

def get_last_h(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {l: out.hidden_states[l][0, pos, :].numpy().astype(np.float32)
            for l in layers}

def get_logits(word):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].numpy().astype(np.float32)

# ── Build full-sentence T2 axes for all 8 types ──────────────────────────────
print("Building full-sentence T2 directions for 8 types × 4 layers ...")
t2_axes = {}
type_names = list(FULL_SENTENCE_PAIRS.keys())

for t_name, pairs in FULL_SENTENCE_PAIRS.items():
    t2_axes[t_name] = {}
    for layer in TRIE_LAYERS:
        diffs = []
        for s1, s2 in pairs:
            h1 = get_last_h(s1, [layer])[layer]
            h2 = get_last_h(s2, [layer])[layer]
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        v = np.mean(diffs, axis=0)
        nv = np.linalg.norm(v)
        t2_axes[t_name][layer] = (v / nv if nv > 1e-6 else v).astype(np.float32)
    print(f"  {t_name}: done")
print()

# ── Cross-type angles at each layer ─────────────────────────────────────────
print("Pairwise T2 angles (sentence-level) at L14:")
for i, ti in enumerate(type_names):
    for j, tj in enumerate(type_names):
        if j <= i: continue
        ang = math.degrees(math.acos(
            min(1.0, abs(float(np.dot(t2_axes[ti][14], t2_axes[tj][14]))))
        ))
        print(f"  {ti:>20} ∠ {tj:<20}  {ang:.1f}°")
print()

# ── Collect probe token hidden states + logits ──────────────────────────────
print(f"Computing hidden states for {len(PROBE_TOKENS)} probe tokens ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        hs     = get_last_h(" " + word.strip(), TRIE_LAYERS)
        logits = get_logits(word)
        tokens_data[word] = {"hs": hs, "logits": logits}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words      = list(tokens_data.keys())
logit_vecs = {w: tokens_data[w]["logits"] for w in words}
print(f"  Collected {len(words)} tokens\n")

# ── Calibrate thresholds per (type, layer) ───────────────────────────────────
def get_projs_for_axis(t_name, layer):
    H = np.stack([tokens_data[w]["hs"][layer] for w in words])
    return H @ t2_axes[t_name][layer]

thresholds = {}
print("Calibrating thresholds + single-axis separations ...")
print(f"  {'type':>20}  {'L':>3}  {'H':>4}  {'L':>4}  {'U':>4}  bimodal")
for t_name in type_names:
    thresholds[t_name] = {}
    for layer in TRIE_LAYERS:
        projs   = get_projs_for_axis(t_name, layer)
        max_p   = np.percentile(projs, 95)
        hi, lo  = max_p * INV_PHI, max_p * INV_PHI2
        n_h = int((projs > hi).sum())
        n_l = int((projs < lo).sum())
        n_u = len(words) - n_h - n_l
        bm  = (n_h + n_l) / len(words)
        thresholds[t_name][layer] = (lo, hi, max_p)
        print(f"  {t_name:>20}  L{layer:<2}  {n_h:>4}  {n_l:>4}  {n_u:>4}  {bm:.3f}")
print()

def classify(proj, t_name, layer):
    lo, hi, _ = thresholds[t_name][layer]
    if proj > hi:  return "H"
    if proj < lo:  return "L"
    return "U"

all_pairs = [(words[i], words[j])
             for i in range(len(words))
             for j in range(i + 1, len(words))]

# ── Per-axis single-axis separation (baseline) ───────────────────────────────
print("Per-axis single-axis separation (1 type × 4 layers = 4-bit):")
per_axis_sep = {}
for t_name in type_names:
    paths = {}
    for w in words:
        bits = []
        for layer in TRIE_LAYERS:
            proj = float(np.dot(tokens_data[w]["hs"][layer], t2_axes[t_name][layer]))
            bits.append(classify(proj, t_name, layer))
        paths[w] = "".join(bits)

    same_s, diff_s = [], []
    for (w1, w2) in all_pairs:
        if "U" in paths[w1] or "U" in paths[w2]: continue
        sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
        if paths[w1] == paths[w2]: same_s.append(sim)
        else: diff_s.append(sim)

    sep = float(np.mean(same_s)) - float(np.mean(diff_s)) if same_s and diff_s else float("nan")
    per_axis_sep[t_name] = sep
    n_lv = len(Counter(paths.values()))
    print(f"  {t_name:>20}:  sep={sep:+.4f}  leaves={n_lv}  "
          f"({'CONFIRMED' if sep > 0.05 else 'WEAK'})")
print()

# ── Multi-axis trie: 1, 2, 4, 8 axes ────────────────────────────────────────
print("=" * 72)
print("Full φ-trie: separation vs number of sentence-level T2 axes")
print("=" * 72)
print()

full_trie_results = {}

for n_axes in [1, 2, 4, 8]:
    axes_used = type_names[:n_axes]
    n_bits    = n_axes * len(TRIE_LAYERS)

    paths = {}
    for w in words:
        bits = []
        for t_name in axes_used:
            for layer in TRIE_LAYERS:
                proj = float(np.dot(tokens_data[w]["hs"][layer], t2_axes[t_name][layer]))
                bits.append(classify(proj, t_name, layer))
        paths[w] = "".join(bits)

    path_counts = Counter(paths.values())

    same_s, diff_s = [], []
    for (w1, w2) in all_pairs:
        if "U" in paths[w1] or "U" in paths[w2]: continue
        sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
        if paths[w1] == paths[w2]: same_s.append(sim)
        else: diff_s.append(sim)

    same_m = float(np.mean(same_s)) if same_s else float("nan")
    diff_m = float(np.mean(diff_s)) if diff_s else float("nan")
    sep    = same_m - diff_m
    n_lv   = len(path_counts)

    full_trie_results[n_axes] = {
        "n_bits": n_bits, "same_mean": same_m, "diff_mean": diff_m,
        "separation": sep, "n_leaves": n_lv,
        "n_same_pairs": len(same_s), "n_diff_pairs": len(diff_s),
    }
    verdict = "CONFIRMED" if sep > 0.10 else "WEAK" if sep > 0.02 else "FAILED"
    print(f"  {n_axes} axes / {n_bits:>2} bits  [{', '.join(axes_used[:3])}{'...' if n_axes > 3 else ''}]")
    print(f"    same-leaf: {same_m:.4f}  diff-leaf: {diff_m:.4f}  "
          f"sep: {sep:+.4f}  leaves: {n_lv}  {verdict}")
    top = [f"{p[:8]}({c})" for p, c in path_counts.most_common(5)]
    print(f"    top leaves: {', '.join(top)}")
    print()

# ── Spot check: semantic content at 2-axis (8-bit) leaf ─────────────────────
print("=" * 72)
print("Spot check: 2-axis (8-bit) leaf semantic contents")
print("=" * 72)

paths_2ax = {}
for w in words:
    bits = []
    for t_name in type_names[:2]:
        for layer in TRIE_LAYERS:
            proj = float(np.dot(tokens_data[w]["hs"][layer], t2_axes[t_name][layer]))
            bits.append(classify(proj, t_name, layer))
    paths_2ax[w] = "".join(bits)

pc_2ax = Counter(paths_2ax.values())
for path, count in sorted(pc_2ax.items(), key=lambda x: -x[1]):
    if count < 3: continue
    tokens_here = [w for w in words if paths_2ax[w] == path]
    print(f"  [{path}] ({count}): {' '.join(tokens_here[:20])}")
print()

# ── Final comparison table ────────────────────────────────────────────────────
print("=" * 72)
print("Complete comparison across all methods (Days 70-74)")
print("=" * 72)
print(f"""
  Day  Method                         sep       bits
  70   Manual T2 (comparative, 1ax)  +0.4548     4
  71   4 manual axes (all same)      +0.4548     4  (no improvement)
  72   Full-state PCA (1ax)          +0.0294     4  (identity manifold)
  73   Diff-PCA (1ax, phrases)       +0.0604     4
  73   Diff-PCA (8ax, phrases)       +0.5127    32
  74   Full T2 (1ax, sentences)       {per_axis_sep[type_names[0]]:+.4f}     4
  74   Full T2 (8ax, sentences)       {full_trie_results[8]['separation']:+.4f}    32
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "per_axis_sep": per_axis_sep,
    "full_trie_results": full_trie_results,
    "phi_pair": {"inv_phi": INV_PHI, "inv_phi2": INV_PHI2},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 74 complete.")
