#!/usr/bin/env python3
"""
Day 76 — Ternary H/U/L φ-Trie

Day 75 found that the standard metric (exclude UNSTABLE from all pairs)
breaks at >4 axes: with 30-44% UNSTABLE per axis, nearly every token
has at least one U bit, leaving zero non-UNSTABLE pairs.

The fix: treat UNSTABLE as a FIRST-CLASS TERNARY category.
  - H = HIGH  (projection > 1/φ  × max95)
  - U = UNSTABLE (projection in φ-pair forbidden zone)
  - L = LOW   (projection < 1/φ² × max95)

A token's semantic address is an 8-character string over {H, U, L}.
This defines a ternary φ-trie.

This experiment:
1. Builds 8-bit ternary addresses for 160 English probe tokens using
   the top-8 decision points from Day 75 (diverse axes + layers)
2. Measures same-leaf vs diff-leaf logit cosine (ALL tokens, no exclusions)
3. Tests: does Hamming distance in ternary address space predict
   logit cosine distance? (If yes: the φ-trie IS a semantic metric)
4. Inspects leaf contents for semantic coherence

PREDICTIONS:
  - Same-leaf similarity > diff-leaf (even including U tokens)
  - Mean logit cosine is monotonically DECREASING with Hamming distance
  - The 8-bit ternary trie has 50-100 occupied leaves
  - Leaves are semantically coherent across all 3 zones (H, U, L)
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day76_ternary_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI         # 0.618
INV_PHI2 = 1 / PHI**2      # 0.382

# ── The 8 top decision points from Day 75 (axis, context, layer, u_frac) ─────
# 1. gender/medium/L27       u_frac=0.444
# 2. comparative/short/L15   u_frac=0.419
# 3. hypernym/medium/L28     u_frac=0.419
# 4. plural/long/L1          u_frac=0.375
# 5. synonym/short/L28       u_frac=0.325
# 6. concrete_abstract/medium/L28  u_frac=0.319
# 7. past_tense/long/L28     u_frac=0.300
# 8. antonym/short/L28       u_frac=0.219

DECISION_AXES = [
    ("gender_medium",            "Gender (medium, L27)",         27),
    ("comparative_short",        "Comparative (short, L15)",     15),
    ("hypernym_medium",          "Hypernym (medium, L28)",       28),
    ("plural_long",              "Plural (long, L1)",             1),
    ("synonym_short",            "Synonym (short, L28)",         28),
    ("concrete_abstract_medium", "Concrete→Abstract (med, L28)", 28),
    ("past_tense_long",          "Past tense (long, L28)",       28),
    ("antonym_short",            "Antonym (short, L28)",         28),
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
        ("The fast car",      "The faster car"),
        ("A big dog",         "A bigger dog"),
        ("The cold wind",     "The colder wind"),
        ("A tall tree",       "A taller tree"),
        ("The old house",     "The older house"),
        ("A bright star",     "A brighter star"),
        ("The dark room",     "The darker room"),
        ("A hard rock",       "A harder rock"),
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
        ("I walk to the market every single morning",      "I walked to the market every single morning"),
        ("She runs through the park after her long work",  "She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house", "He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden",      "They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days",        "We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend",     "She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire", "They sang together around the evening campfire"),
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

def ternary_hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def get_h_at_layer(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

def get_logits(word):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].numpy().astype(np.float32)

# ── Build T2 axis for each decision point ────────────────────────────────────
print("Building T2 axes for 8 decision points ...")
t2_axes = {}

for (axis_key, label, layer) in DECISION_AXES:
    pairs  = AXIS_PAIRS[axis_key]
    diffs  = []
    for s1, s2 in pairs:
        h1 = get_h_at_layer(s1, layer)
        h2 = get_h_at_layer(s2, layer)
        d  = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if diffs:
        v  = np.mean(diffs, axis=0)
        nv = np.linalg.norm(v)
        t2_axes[axis_key] = (v / nv).astype(np.float32) if nv > 1e-6 else np.zeros(hidden_size, dtype=np.float32)
    else:
        t2_axes[axis_key] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {label}")

print()

# ── Collect probe token hidden states and logits ─────────────────────────────
print(f"Computing hidden states for {len(PROBE_TOKENS)} probe tokens ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        hs_per_axis = {}
        for (axis_key, _, layer) in DECISION_AXES:
            hs_per_axis[axis_key] = get_h_at_layer(" " + word.strip(), layer)
        logits = get_logits(word)
        tokens_data[word] = {"hs": hs_per_axis, "logits": logits}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words         = list(tokens_data.keys())
english_words = [w for w in words if w not in ENGLISH_SKIP]
logit_vecs    = {w: tokens_data[w]["logits"] for w in words}
print(f"  Collected {len(words)} tokens ({len(english_words)} English)\n")

# ── Calibrate thresholds and classify ────────────────────────────────────────
thresholds = {}
classes    = {}    # classes[word][axis_key] = "H" | "U" | "L"

print("Calibrating thresholds ...")
for (axis_key, label, layer) in DECISION_AXES:
    axis = t2_axes[axis_key]
    if np.linalg.norm(axis) < 1e-6:
        thresholds[axis_key] = (0, 0, 0)
        for w in words: classes.setdefault(w, {})[axis_key] = "U"
        continue
    projs   = np.array([float(np.dot(tokens_data[w]["hs"][axis_key], axis)) for w in words])
    max_p   = np.percentile(projs, 95)
    hi, lo  = max_p * INV_PHI, max_p * INV_PHI2
    thresholds[axis_key] = (lo, hi, max_p)

    n_h = int((projs > hi).sum())
    n_l = int((projs < lo).sum())
    n_u = len(words) - n_h - n_l
    # within-English
    eng_projs = np.array([float(np.dot(tokens_data[w]["hs"][axis_key], axis))
                          for w in english_words])
    eu = int(((eng_projs <= hi) & (eng_projs >= lo)).sum())
    print(f"  {label:>35}:  H={n_h:>4} L={n_l:>4} U={n_u:>4}  eng_U={eu:>4}")

    for i, w in enumerate(words):
        p = projs[i]
        if   p > hi: c = "H"
        elif p < lo: c = "L"
        else:        c = "U"
        classes.setdefault(w, {})[axis_key] = c

print()

# ── Build ternary addresses ───────────────────────────────────────────────────
axis_keys = [ak for ak, _, _ in DECISION_AXES]
addresses = {w: "".join(classes[w][ak] for ak in axis_keys) for w in words}

print(f"Ternary address space: {len(set(addresses.values()))} unique addresses "
      f"among {len(words)} tokens\n")

# ── Same-leaf vs diff-leaf (no UNSTABLE exclusion) ───────────────────────────
all_pairs  = [(words[i], words[j])
              for i in range(len(words))
              for j in range(i+1, len(words))]

same_sims  = []
diff_sims  = []
for w1, w2 in all_pairs:
    sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
    if addresses[w1] == addresses[w2]: same_sims.append(sim)
    else:                               diff_sims.append(sim)

same_m = float(np.mean(same_sims)) if same_sims else float("nan")
diff_m = float(np.mean(diff_sims)) if diff_sims else float("nan")
sep    = same_m - diff_m

print("=" * 72)
print("Ternary trie: same-leaf vs diff-leaf (all tokens, no exclusion)")
print("=" * 72)
print(f"  same-leaf: {same_m:.4f}  ({len(same_sims)} pairs)")
print(f"  diff-leaf: {diff_m:.4f}  ({len(diff_sims)} pairs)")
print(f"  separation: {sep:+.4f}  "
      f"({'CONFIRMED' if sep > 0.05 else 'WEAK' if sep > 0 else 'FAILED'})")
print()

# ── Hamming distance vs logit cosine ─────────────────────────────────────────
print("=" * 72)
print("Hamming distance in ternary address → logit cosine")
print("=" * 72)
by_hamming = defaultdict(list)
for w1, w2 in all_pairs:
    d   = ternary_hamming(addresses[w1], addresses[w2])
    sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
    by_hamming[d].append(sim)

print(f"  {'dist':>5}  {'mean_sim':>9}  {'n_pairs':>8}  monotone")
prev_sim = 1.1
monotone = True
for d in sorted(by_hamming.keys()):
    m = float(np.mean(by_hamming[d]))
    n = len(by_hamming[d])
    mono = "↓" if m < prev_sim else "↑ BREAK"
    if m > prev_sim: monotone = False
    print(f"  {d:>5}  {m:>9.4f}  {n:>8}  {mono}")
    prev_sim = m
print(f"\n  Overall monotone: {'YES ✓' if monotone else 'NO — non-monotone'}")
print()

# ── Leaf contents (by size) ───────────────────────────────────────────────────
print("=" * 72)
print("Ternary leaf contents (leaves with ≥ 2 tokens)")
print("=" * 72)
leaf_to_words = defaultdict(list)
for w in words:
    leaf_to_words[addresses[w]].append(w)

multi_leaves = [(leaf, ww) for leaf, ww in leaf_to_words.items() if len(ww) >= 2]
multi_leaves.sort(key=lambda x: -len(x[1]))

for leaf, ww in multi_leaves:
    n_h = leaf.count("H"); n_u = leaf.count("U"); n_l = leaf.count("L")
    avg_sim = float(np.mean([cos_sim(logit_vecs[ww[i]], logit_vecs[ww[j]])
                             for i in range(len(ww)) for j in range(i+1, len(ww))])) \
              if len(ww) > 1 else float("nan")
    print(f"  [{leaf}] ({len(ww)})  H={n_h} U={n_u} L={n_l}  "
          f"sim={avg_sim:.3f}:  {' '.join(ww[:20])}")
print()

# ── U-zone analysis: same-U pairs vs random ──────────────────────────────────
print("=" * 72)
print("UNSTABLE zone analysis: same-U-zone vs same-H-zone vs same-L-zone")
print("=" * 72)
for zone in ["H", "U", "L"]:
    zone_words = [w for w in words if all(classes[w][ak] == zone for ak in axis_keys)]
    if len(zone_words) < 2:
        print(f"  Zone {zone}: only {len(zone_words)} tokens, skip")
        continue
    sims = [cos_sim(logit_vecs[zone_words[i]], logit_vecs[zone_words[j]])
            for i in range(len(zone_words)) for j in range(i+1, len(zone_words))]
    print(f"  Zone {zone} (all-{zone}): {len(zone_words)} tokens, "
          f"mean_sim={float(np.mean(sims)):.4f}  "
          f"tokens: {' '.join(zone_words[:12])}")
print()

# ── Per-axis bit information: what does each bit split? ──────────────────────
print("=" * 72)
print("Per-axis semantic split (H vs L tokens, excluding U)")
print("=" * 72)
for (axis_key, label, layer) in DECISION_AXES:
    h_words = [w for w in english_words if classes[w][axis_key] == "H"]
    l_words = [w for w in english_words if classes[w][axis_key] == "L"]
    u_words = [w for w in english_words if classes[w][axis_key] == "U"]
    print(f"\n  {label}:")
    print(f"    H ({len(h_words)}): {' '.join(h_words[:18])}")
    print(f"    L ({len(l_words)}): {' '.join(l_words[:18])}")
    print(f"    U ({len(u_words)}): {' '.join(u_words[:18])}")
print()

# ── Final comparison table ────────────────────────────────────────────────────
print("=" * 72)
print("Complete comparison (Days 70–76)")
print("=" * 72)
leaf_sizes = sorted([len(v) for v in leaf_to_words.values()], reverse=True)
print(f"""
  Day  Method                           sep      leaves  note
  70   Manual T2 (comp, 1ax, binary)   +0.454     4      UNSTABLE excluded
  71   4 manual axes (all same)         +0.454     4      same as Day 70
  72   Full-state PCA (1ax)             +0.029     9      identity manifold
  73   Diff-PCA (8ax, 32-bit, binary)   +0.513   158      UNSTABLE excluded
  74   Full T2 (8ax sentences)          +0.553     5      language only
  75   Rich 4-bit trie (binary)         +0.083    52      UNSTABLE excluded
  76   8-bit ternary (no exclusion)     {sep:+.3f}   {len(leaf_to_words):>3}      ALL tokens included
""")
print(f"  Leaf sizes (top 10): {leaf_sizes[:10]}")
print(f"  Monotone Hamming: {'YES' if monotone else 'NO'}")
print(f"  Hamming d=0 sim: {float(np.mean(by_hamming[0])):.4f}" if 0 in by_hamming else "")
print(f"  Hamming d=8 sim: {float(np.mean(by_hamming.get(8, [float('nan')]))):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "separation": sep, "same_mean": same_m, "diff_mean": diff_m,
    "n_leaves": len(leaf_to_words),
    "hamming_vs_sim": {str(d): float(np.mean(v)) for d, v in by_hamming.items()},
    "monotone_hamming": monotone,
    "leaf_sizes": leaf_sizes[:20],
    "addresses": {w: addresses[w] for w in words},
    "phi_pair": {"inv_phi": INV_PHI, "inv_phi2": INV_PHI2},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 76 complete.")
