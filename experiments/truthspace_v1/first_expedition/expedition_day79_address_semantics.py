#!/usr/bin/env python3
"""
Day 79 — Address Semantics: What Does Each Bit Mean?

No model runs. Pure analysis of pre-computed Day 78 ternary addresses.

The 8-bit ternary address from Day 78 encodes:
  bit 0: gender_medium/L27
  bit 1: comparative_short/L15
  bit 2: hypernym_medium/L28
  bit 3: plural_long/L1
  bit 4: synonym_short/L28
  bit 5: concrete_abstract_medium/L28
  bit 6: past_tense_long/L28
  bit 7: antonym_short/L28

QUESTIONS:
  1. Does Hamming distance predict part-of-speech match?
     (same POS → lower Hamming distance?)
  2. What does H/U/L mean for each bit?
     (which POS classes fall into H, U, L for each axis?)
  3. What is the mutual information between each bit and POS/category?
  4. Can we recover word categories from ternary addresses alone?
     (k-means on ternary addresses → do clusters match semantic families?)

APPROACH:
  - Assign 8 coarse POS tags to all 401 words (NOUN/VERB/ADJ/ADV/FUNC/NUM/COLOR/NAME)
  - Assign 14 semantic category labels (animals, body, objects, etc.)
  - Compute within-POS vs across-POS mean Hamming
  - For each bit: compute H/U/L distribution by POS
  - Compute mutual information between each bit and POS
  - Cluster addresses with k-means (k=8), compare to POS distribution
"""
import json, math
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = str(SCRIPT_DIR / "day78_scale_vocab.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day79_address_semantics.json")

# ── Load pre-computed data ────────────────────────────────────────────────────
with open(INPUT_FILE) as f:
    data = json.load(f)

addresses = data["addresses"]     # word → 8-char string
words     = list(addresses.keys())
hamm_vs_sim = {int(k): v for k, v in data["hamming_vs_sim"].items()}

AXIS_NAMES = [
    "gender",    "comparative", "hypernym",   "plural",
    "synonym",   "concrete",    "past_tense",  "antonym",
]

def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# ── POS and semantic category labels ─────────────────────────────────────────
# Coarse POS: N=noun, V=verb, J=adjective, R=adverb, F=function, Q=number/quantifier,
#             C=color, G=gender/name
POS = {
    "N": [
        "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
        "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
        "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
        "pig", "sheep", "goat", "duck", "hen", "crow", "owl",
        "turtle", "lizard", "crab", "lobster", "octopus", "beetle",
        "butterfly", "worm", "salmon", "tuna", "herring", "sparrow",
        "robin", "finch", "parrot",
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
        "love", "hate", "truth", "beauty", "freedom", "power",
        "time", "space", "mind", "body", "soul", "life", "death", "hope",
        "fear", "joy", "pain", "trust", "faith", "peace",
        "war", "law", "right", "duty", "honor", "shame", "pride", "guilt",
        "anger", "grief",
        "city", "town", "village", "country", "island", "valley", "cave",
        "castle", "market", "church", "school", "hospital",
        "garden", "field", "park", "lake", "coast", "cliff", "path",
        "bread", "meat", "fruit", "milk", "water", "fire", "oil", "salt",
        "sugar", "coffee", "wine", "beer", "tea", "egg", "cheese",
        "dogs", "cats", "trees", "birds", "horses", "men", "women",
        "children", "hands", "eyes",
        "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
        "brother", "sister", "father", "mother", "son", "daughter",
        "husband", "wife", "prince", "princess", "actor", "actress",
    ],
    "V": [
        "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
        "write", "read", "build", "break", "open", "close", "start", "stop",
        "think", "know", "see", "hear", "feel", "love", "hate", "want",
        "give", "take", "make", "find", "lose", "push", "pull", "turn",
        "move", "go", "come", "fall", "rise", "grow", "kill", "help",
        "ran", "walked", "jumped", "flew", "ate", "saw", "heard", "broke",
        "built", "wrote",
    ],
    "J": [
        "fast", "slow", "big", "small", "hot", "cold", "old", "new",
        "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
        "good", "bad", "right", "wrong", "high", "low", "long", "short",
        "wide", "narrow", "deep", "shallow", "thick", "thin", "heavy", "light",
        "clean", "dirty", "sweet", "bitter", "sharp", "dull", "loud", "quiet",
        "faster", "slower", "bigger", "smaller", "better", "worse",
        "biggest", "smallest", "best", "worst",
        "red", "blue", "green", "yellow", "white", "black", "brown",
        "orange", "purple", "pink", "gray", "gold",
    ],
    "R": [
        "quickly", "slowly", "often", "never", "always", "very", "quite",
        "really", "just", "still",
    ],
    "F": [
        "the", "a", "and", "or", "not", "is", "was", "in", "on", "of",
        "to", "from", "with", "for", "he", "she", "it", "they", "we",
        "I", "you", "his", "her", "their", "my", "your", "its", "our",
        "but", "if",
    ],
    "Q": [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "hundred", "thousand",
        "many", "few", "more", "less", "most", "least", "all", "some",
    ],
}

# Assign POS to all words (handle overlaps: love appears in both N and V → take first)
word_pos = {}
for pos_tag, wlist in POS.items():
    for w in wlist:
        if w in addresses and w not in word_pos:
            word_pos[w] = pos_tag
for w in addresses:
    if w not in word_pos:
        word_pos[w] = "?"

# ── 1. Within-POS vs across-POS Hamming distances ─────────────────────────────
print("=" * 72)
print("1. Within-POS vs across-POS Hamming distance")
print("=" * 72)

all_w = [w for w in words if word_pos.get(w, "?") != "?"]
pos_tags = sorted(set(word_pos[w] for w in all_w))

within_pos_dists = defaultdict(list)
across_pos_dists = []

for i in range(len(all_w)):
    for j in range(i+1, len(all_w)):
        w1, w2 = all_w[i], all_w[j]
        d = hamming(addresses[w1], addresses[w2])
        if word_pos[w1] == word_pos[w2]:
            within_pos_dists[word_pos[w1]].append(d)
        else:
            across_pos_dists.append(d)

print(f"  {'POS':>4}  {'n_words':>7}  {'within_H':>9}  {'examples'}")
for pos in sorted(pos_tags):
    ww = [w for w in all_w if word_pos[w] == pos]
    wh = within_pos_dists[pos]
    wm = float(np.mean(wh)) if wh else float("nan")
    print(f"  {pos:>4}  {len(ww):>7}  {wm:>9.2f}  {' '.join(ww[:6])}")

across_m = float(np.mean(across_pos_dists))
all_dists = [d for dlist in within_pos_dists.values() for d in dlist] + across_pos_dists
overall_m = float(np.mean(all_dists))
print(f"\n  Across-POS mean Hamming: {across_m:.2f}")
print(f"  Overall mean Hamming:    {overall_m:.2f}")
print(f"\n  Within-POS Hamming is LOWER than across-POS for:")
for pos in sorted(pos_tags):
    wh = within_pos_dists[pos]
    wm = float(np.mean(wh)) if wh else float("nan")
    mark = "✓" if wm < across_m else " "
    print(f"    {pos}: {wm:.2f} vs {across_m:.2f} {mark}")
print()

# ── 2. Per-bit H/U/L distribution by POS ─────────────────────────────────────
print("=" * 72)
print("2. Per-bit H/U/L distribution by POS")
print("=" * 72)

for bit_i, axis_name in enumerate(AXIS_NAMES):
    print(f"\n  Bit {bit_i} ({axis_name}):")
    for pos in sorted(pos_tags):
        ww = [w for w in all_w if word_pos[w] == pos]
        if not ww: continue
        hul = Counter(addresses[w][bit_i] for w in ww)
        total = len(ww)
        h_pct = 100*hul.get("H",0)/total
        u_pct = 100*hul.get("U",0)/total
        l_pct = 100*hul.get("L",0)/total
        dominant = max("H","U","L", key=lambda x: hul.get(x,0))
        print(f"    {pos:>4}: H={h_pct:4.0f}% U={u_pct:4.0f}% L={l_pct:4.0f}%  "
              f"→ dominant={dominant}  [{' '.join(ww[:4])}...]")
print()

# ── 3. Mutual information: each bit vs POS ───────────────────────────────────
print("=" * 72)
print("3. Mutual information: each bit with POS label")
print("=" * 72)

def mutual_information(bit_vals, labels):
    """Compute MI between a discrete variable (bit) and labels."""
    n = len(bit_vals)
    joint  = Counter(zip(bit_vals, labels))
    px     = Counter(bit_vals)
    py     = Counter(labels)
    mi = 0.0
    for (x, y), cnt in joint.items():
        p_xy = cnt / n
        p_x  = px[x] / n
        p_y  = py[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi

bit_mi = []
for bit_i, axis_name in enumerate(AXIS_NAMES):
    bit_vals = [addresses[w][bit_i] for w in all_w]
    labels   = [word_pos[w] for w in all_w]
    mi       = mutual_information(bit_vals, labels)
    bit_mi.append((axis_name, mi))

bit_mi.sort(key=lambda x: -x[1])
print(f"  {'axis':>25}  {'MI (bits)':>10}")
for name, mi in bit_mi:
    print(f"  {name:>25}  {mi:>10.4f}")
print()
print(f"  Most informative bit: {bit_mi[0][0]} ({bit_mi[0][1]:.4f} bits)")
print(f"  Least informative:    {bit_mi[-1][0]} ({bit_mi[-1][1]:.4f} bits)")
print()

# ── 4. Bit entropy and conditional entropy ────────────────────────────────────
print("=" * 72)
print("4. Bit entropy and conditional entropy given POS")
print("=" * 72)

def entropy(vals):
    cnt = Counter(vals); n = len(vals)
    return -sum((c/n)*math.log2(c/n) for c in cnt.values() if c > 0)

for bit_i, axis_name in enumerate(AXIS_NAMES):
    bit_vals = [addresses[w][bit_i] for w in all_w]
    H_bit    = entropy(bit_vals)
    # Conditional entropy H(bit | POS)
    H_cond   = 0.0
    for pos in pos_tags:
        ww = [w for w in all_w if word_pos[w] == pos]
        if not ww: continue
        bv = [addresses[w][bit_i] for w in ww]
        H_cond += (len(ww)/len(all_w)) * entropy(bv)
    MI_check = H_bit - H_cond
    print(f"  {axis_name:>25}:  H={H_bit:.3f}  H|POS={H_cond:.3f}  MI={MI_check:.4f}")
print()

# ── 5. Ternary k-means clustering ──────────────────────────────────────────────
print("=" * 72)
print("5. K-means clustering of ternary addresses (k=8)")
print("   (encoding: H=2, U=1, L=0)")
print("=" * 72)

# Encode addresses as integer vectors
encode = {"H": 2, "U": 1, "L": 0}
X = np.array([[encode[c] for c in addresses[w]] for w in all_w], dtype=np.float32)

# Simple k-means
from numpy.random import RandomState
rng = RandomState(42)

def kmeans(X, k, n_iter=100):
    n = X.shape[0]
    idx  = rng.choice(n, k, replace=False)
    centers = X[idx].copy()
    labels  = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # Assign
        dists = np.array([[np.sum((x - c)**2) for c in centers] for x in X])
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels): break
        labels = new_labels
        # Update
        for ki in range(k):
            members = X[labels == ki]
            if len(members) > 0:
                centers[ki] = members.mean(axis=0)
    return labels

for k in [6, 8, 14]:
    labels = kmeans(X, k)
    # For each cluster, compute dominant POS
    print(f"\n  k={k} clusters:")
    for ki in range(k):
        members = [all_w[i] for i in range(len(all_w)) if labels[i] == ki]
        if not members: continue
        pos_dist = Counter(word_pos[w] for w in members)
        dominant_pos = pos_dist.most_common(1)[0]
        purity = dominant_pos[1] / len(members)
        print(f"    C{ki:02d} ({len(members):>3}): dom={dominant_pos[0]} "
              f"purity={purity:.2f}  {' '.join(members[:8])}")

print()

# ── 6. Per-bit semantic interpretation ────────────────────────────────────────
print("=" * 72)
print("6. Semantic interpretation of each bit (top H and L words)")
print("=" * 72)

for bit_i, axis_name in enumerate(AXIS_NAMES):
    h_words = [w for w in all_w if addresses[w][bit_i] == "H"]
    l_words = [w for w in all_w if addresses[w][bit_i] == "L"]
    u_words = [w for w in all_w if addresses[w][bit_i] == "U"]
    # POS distribution in H and L
    h_pos = Counter(word_pos[w] for w in h_words)
    l_pos = Counter(word_pos[w] for w in l_words)
    print(f"\n  Bit {bit_i} [{axis_name}]:")
    print(f"    H ({len(h_words)}): "
          f"POS={dict(h_pos.most_common(3))}  "
          f"words: {' '.join(h_words[:10])}")
    print(f"    L ({len(l_words)}): "
          f"POS={dict(l_pos.most_common(3))}  "
          f"words: {' '.join(l_words[:10])}")
    print(f"    U ({len(u_words)}): {' '.join(u_words[:8])}")

print()

# ── 7. Address distance vs POS match ─────────────────────────────────────────
print("=" * 72)
print("7. Hamming distance distribution: same-POS vs diff-POS")
print("=" * 72)

same_pos_dists = []
diff_pos_dists = []
for i in range(len(all_w)):
    for j in range(i+1, len(all_w)):
        d = hamming(addresses[all_w[i]], addresses[all_w[j]])
        if word_pos[all_w[i]] == word_pos[all_w[j]]:
            same_pos_dists.append(d)
        else:
            diff_pos_dists.append(d)

sm = float(np.mean(same_pos_dists))
dm = float(np.mean(diff_pos_dists))
effect = dm - sm
print(f"  Same-POS mean Hamming: {sm:.3f}  ({len(same_pos_dists)} pairs)")
print(f"  Diff-POS mean Hamming: {dm:.3f}  ({len(diff_pos_dists)} pairs)")
print(f"  Effect size: {effect:+.3f}  "
      f"({'same-POS closer ✓' if effect > 0 else 'NO POS clustering'})")
print()

# ── 8. Address uniqueness and completeness ────────────────────────────────────
print("=" * 72)
print("8. Address space analysis")
print("=" * 72)
addr_counts = Counter(addresses.values())
n_unique   = len(addr_counts)
n_occupied = len([c for c in addr_counts.values() if c > 0])
theoretical = 3**8
print(f"  Total words: {len(all_w)}")
print(f"  Unique addresses: {n_unique} / {theoretical} possible ({100*n_unique/theoretical:.1f}%)")
print()

# Which address positions (bits) are most predictive?
print("  Bit predictive power (MI with POS, sorted desc):")
for name, mi in bit_mi:
    bar = "█" * int(mi * 40)
    print(f"  {name:>25}: {mi:.4f}  {bar}")
print()

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "within_pos_mean_hamming": {pos: float(np.mean(d)) if d else None
                                 for pos, d in within_pos_dists.items()},
    "across_pos_mean_hamming": across_m,
    "same_pos_mean_hamming": sm,
    "diff_pos_mean_hamming": dm,
    "pos_hamming_effect": effect,
    "bit_mi_with_pos": {name: mi for name, mi in bit_mi},
    "pos_clustering_confirmed": effect > 0,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 79 complete.")
