#!/usr/bin/env python3
"""
Day 81 — T2 Axis Rotation Test: Does the Transformer Rotate or Just Scale?

Day 80 confirmed: the φ-trie structure exists at L0 AND at L28.
The transformer amplifies (1.9× average) but doesn't create it.

KEY QUESTION: Are the T2 axis DIRECTIONS the same across layers?

  H_LENS:   angle(T2_L0, T2_L28) ≈ 0°  → transformer only scales/focuses
  H_ROTATE: angle(T2_L0, T2_L28) >> 0° → transformer rotates the subspace

ALSO TEST:
  1. Are L0 T2 axes orthogonal to each other?
     (Day 73 confirmed L28 pairwise angles 80-90° — does L0 show the same?)
  2. Is T2_L0 ⊥ PC0_L0 (identity manifold at L0)?
     (Day 72 confirmed T2_L28 ⊥ PC0_L28 at 89°)
  3. Track axis rotation across ALL layers (0→1→...→28)
     Using the 4 sentence-pair axes from the main Day 78 set

PREDICTIONS:
  H_LENS predicts:   all angles < 45° (same direction across layers)
  H_ROTATE predicts: angles > 45° (especially L0 vs L28)
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day81_axis_rotation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + math.sqrt(5)) / 2

AXIS_NAMES = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete",   "past_tense", "antonym",
]

AXIS_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom",    "The queen ruled with great wisdom"),
        ("A man walked through the forest",     "A woman walked through the forest"),
        ("The boy kicked the ball hard",        "The girl kicked the ball hard"),
        ("His brother arrived at the party",    "His sister arrived at the party"),
        ("The father worked to feed family",    "The mother worked to feed family"),
        ("A son was born in the winter",        "A daughter was born in the winter"),
        ("The prince rode across the land",     "The princess rode across the land"),
        ("The actor played a leading role",     "The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car",   "The faster car"),
        ("A big dog",      "A bigger dog"),
        ("The cold wind",  "The colder wind"),
        ("A tall tree",    "A taller tree"),
        ("The old house",  "The older house"),
        ("A bright star",  "A brighter star"),
        ("The dark room",  "The darker room"),
        ("A hard rock",    "A harder rock"),
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
        ("He is big",         "He is large"),
        ("She is small",      "She is tiny"),
        ("He runs fast",      "He runs quick"),
        ("It is cold",        "It is frigid"),
        ("She is happy",      "She is joyful"),
        ("He spoke loudly",   "He spoke noisily"),
        ("It is hard",        "It is difficult"),
        ("He is old",         "He is aged"),
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
        ("I walk to the market every single morning",       "I walked to the market every single morning"),
        ("She runs through the park after her long work",   "She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house",  "He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden",       "They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days",         "We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend",      "She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire",  "They sang together around the evening campfire"),
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
}

# ── 401 probe words (same as Day 78) for PC0 computation ─────────────────────
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

# Layers to test: representative sweep + fine resolution at key zones
SWEEP_LAYERS = list(range(0, 29))   # all 29 layers

def angle_deg(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(math.degrees(math.acos(float(np.clip(c, -1, 1)))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_layers    = model.config.num_hidden_layers
print(f"  hidden={hidden_size}  n_layers={n_layers}\n")

def get_all_layers(text):
    """Single pass, return all hidden states (last position)."""
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return [out.hidden_states[l][0, pos, :].numpy().astype(np.float32)
            for l in range(n_layers + 1)]   # 0..28 inclusive

# ── Compute T2 axes at ALL layers for all 8 types ────────────────────────────
print("Computing T2 axes at all 29 layers for 8 axis types ...")
print("(8 types × 8 sentence pairs × 2 sentences = 128 forward passes)\n")

# t2_all[axis_name][layer] = unit vector or zero
t2_all = {name: [None] * (n_layers + 1) for name in AXIS_NAMES}

for name in AXIS_NAMES:
    pairs = AXIS_PAIRS[name]
    # Collect difference vectors per layer
    diffs_by_layer = [[] for _ in range(n_layers + 1)]
    for s1, s2 in pairs:
        h1_all = get_all_layers(s1)
        h2_all = get_all_layers(s2)
        for l in range(n_layers + 1):
            d = h2_all[l] - h1_all[l]
            n = np.linalg.norm(d)
            if n > 1e-6:
                diffs_by_layer[l].append(d / n)

    for l in range(n_layers + 1):
        if diffs_by_layer[l]:
            v = np.mean(diffs_by_layer[l], axis=0)
            nv = np.linalg.norm(v)
            t2_all[name][l] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
        else:
            t2_all[name][l] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {name} done")
print()

# ── Compute PC0 (identity manifold) at each layer ────────────────────────────
print("Computing PC0 at each layer (probe vocabulary) ...")
probe_hidden = {l: [] for l in [0, 1, 15, 27, 28]}

for word in PROBE_TOKENS[:80]:   # use 80 words for speed
    try:
        h_all = get_all_layers(" " + word.strip())
        for l in [0, 1, 15, 27, 28]:
            probe_hidden[l].append(h_all[l])
    except Exception as e:
        print(f"  SKIP {word}: {e}")

pc0_by_layer = {}
for l in [0, 1, 15, 27, 28]:
    H = np.array(probe_hidden[l], dtype=np.float32)
    H -= H.mean(0)
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    pc0_by_layer[l] = Vt[0]   # first right singular vector
    print(f"  PC0 at L{l} computed")
print()

# ── Test 1: axis direction across layers ─────────────────────────────────────
print("=" * 72)
print("Test 1: T2 axis direction across layers (angle from L0)")
print("(angle = degrees between T2 axis at that layer vs L0)")
print("=" * 72)
print(f"  L    " + "  ".join(f"{n:>12}" for n in AXIS_NAMES))

layer_angles_from_l0 = {name: [] for name in AXIS_NAMES}
for l in SWEEP_LAYERS:
    angles = []
    for name in AXIS_NAMES:
        v0 = t2_all[name][0]
        vl = t2_all[name][l]
        if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(vl) < 1e-6:
            a = float("nan")
        else:
            a = angle_deg(v0, vl)
            a = min(a, 180 - a)   # take acute angle (same axis, flipped)
        layer_angles_from_l0[name].append(a)
        angles.append(a)
    if l in [0, 1, 5, 10, 15, 20, 27, 28]:
        a_str = "  ".join(f"{a:>12.1f}" if not math.isnan(a) else f"{'nan':>12}" for a in angles)
        print(f"  L{l:>2}  {a_str}")
print()

# ── Test 2: axis stability — L0 vs L28 angles ────────────────────────────────
print("=" * 72)
print("Test 2: L0 vs L28 axis direction (the key rotation test)")
print("=" * 72)
print(f"  {'axis':>20}  {'angle_L0_L28':>14}  {'verdict':>12}")
rotation_angles = {}
for name in AXIS_NAMES:
    v0  = t2_all[name][0]
    v28 = t2_all[name][28]
    if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v28) < 1e-6:
        a = float("nan"); verdict = "DEGENERATE"
    else:
        a = min(angle_deg(v0, v28), 180 - angle_deg(v0, v28))
        verdict = ("SAME" if a < 15 else
                   "SMALL_ROT" if a < 45 else
                   "LARGE_ROT" if a < 90 else "PERPENDICULAR")
    rotation_angles[name] = a
    print(f"  {name:>20}  {a:>14.1f}°  {verdict}")

mean_rot = float(np.nanmean(list(rotation_angles.values())))
print(f"\n  Mean L0→L28 rotation: {mean_rot:.1f}°")
print(f"  H_LENS  (< 15°): {'CONFIRMED' if mean_rot < 15 else 'REJECTED'}")
print(f"  H_SMALL_ROT (15-45°): {'CONFIRMED' if 15 <= mean_rot < 45 else 'REJECTED'}")
print(f"  H_ROTATE (≥ 45°): {'CONFIRMED' if mean_rot >= 45 else 'REJECTED'}")
print()

# ── Test 3: are L0 T2 axes orthogonal? ────────────────────────────────────────
print("=" * 72)
print("Test 3: Pairwise orthogonality of T2 axes at L0 vs L28")
print("=" * 72)

for layer_label, layer_idx in [("L0", 0), ("L28", 28)]:
    vecs = [t2_all[name][layer_idx] for name in AXIS_NAMES]
    print(f"\n  {layer_label} pairwise angles (lower triangle):")
    print("  " + "  ".join(f"{n[:6]:>8}" for n in AXIS_NAMES))
    for i in range(len(AXIS_NAMES)):
        row = f"  {AXIS_NAMES[i][:8]:>8}"
        for j in range(len(AXIS_NAMES)):
            if j > i:
                row += " " * 9
            elif j == i:
                row += f"{'—':>9}"
            else:
                a = min(angle_deg(vecs[i], vecs[j]), 180 - angle_deg(vecs[i], vecs[j]))
                row += f"{a:>9.1f}"
        print(row)

    # Summary
    angles = []
    for i in range(len(AXIS_NAMES)):
        for j in range(i):
            a = min(angle_deg(vecs[i], vecs[j]), 180 - angle_deg(vecs[i], vecs[j]))
            angles.append(a)
    print(f"  {layer_label} pairwise: min={min(angles):.1f}°  mean={np.mean(angles):.1f}°  max={max(angles):.1f}°")
    n_orth = sum(1 for a in angles if a > 70)
    print(f"  {layer_label} pairs with angle > 70°: {n_orth}/{len(angles)}")
print()

# ── Test 4: T2 ⊥ PC0 at L0? ──────────────────────────────────────────────────
print("=" * 72)
print("Test 4: T2 ⊥ PC0 (identity manifold) at each layer")
print("  (Day 72 found T2_L28 ⊥ PC0_L28 at ~89° — does L0 show same?)")
print("=" * 72)

for l in [0, 1, 15, 27, 28]:
    if l not in pc0_by_layer: continue
    pc0 = pc0_by_layer[l]
    print(f"\n  L{l}:")
    for name in AXIS_NAMES:
        v = t2_all[name][l]
        if np.linalg.norm(v) < 1e-6: continue
        a = angle_deg(v, pc0)
        a_from_90 = abs(a - 90)
        mark = "⊥" if a_from_90 < 10 else ("~⊥" if a_from_90 < 20 else "  ")
        print(f"    {name:>15}: angle={a:5.1f}°  dev_from_90={a_from_90:4.1f}°  {mark}")
print()

# ── Test 5: layer-by-layer rotation (how fast does axis rotate?) ───────────────
print("=" * 72)
print("Test 5: Layer-by-layer axis rotation (angle from previous layer)")
print("  (consecutive layer rotation speed)")
print("=" * 72)
print(f"  L→L+1  " + "  ".join(f"{n[:8]:>8}" for n in AXIS_NAMES))

for l in range(1, n_layers + 1):
    consec_angles = []
    for name in AXIS_NAMES:
        vp = t2_all[name][l-1]
        vn = t2_all[name][l]
        if np.linalg.norm(vp) < 1e-6 or np.linalg.norm(vn) < 1e-6:
            consec_angles.append(float("nan"))
        else:
            a = min(angle_deg(vp, vn), 180 - angle_deg(vp, vn))
            consec_angles.append(a)
    if l in [1, 2, 5, 10, 14, 15, 16, 20, 25, 27, 28]:
        a_str = "  ".join(f"{a:>8.1f}" if not math.isnan(a) else f"{'nan':>8}" for a in consec_angles)
        mean_a = float(np.nanmean(consec_angles))
        print(f"  L{l-1:>2}→{l:<2}  {a_str}  (mean={mean_a:.1f}°)")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 81 Summary")
print("=" * 72)
v0_vecs  = [t2_all[n][0]  for n in AXIS_NAMES]
v28_vecs = [t2_all[n][28] for n in AXIS_NAMES]
orth_l0  = [min(angle_deg(v0_vecs[i], v0_vecs[j]), 180-angle_deg(v0_vecs[i], v0_vecs[j]))
            for i in range(8) for j in range(i)]
orth_l28 = [min(angle_deg(v28_vecs[i], v28_vecs[j]), 180-angle_deg(v28_vecs[i], v28_vecs[j]))
            for i in range(8) for j in range(i)]

print(f"""
  T2 axes at L0: pairwise angles
    min={min(orth_l0):.1f}°  mean={np.mean(orth_l0):.1f}°  max={max(orth_l0):.1f}°
    pairs > 70°: {sum(1 for a in orth_l0 if a > 70)}/28
    
  T2 axes at L28: pairwise angles
    min={min(orth_l28):.1f}°  mean={np.mean(orth_l28):.1f}°  max={max(orth_l28):.1f}°
    pairs > 70°: {sum(1 for a in orth_l28 if a > 70)}/28
  
  L0 → L28 rotation: mean={mean_rot:.1f}°
  Verdict: {'LENS (same direction)' if mean_rot < 25 else 'PARTIAL_ROTATION' if mean_rot < 55 else 'FULL_ROTATION'}
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "mean_l0_l28_rotation_deg": mean_rot,
    "per_axis_rotation": rotation_angles,
    "l0_pairwise_angles": {f"{AXIS_NAMES[i]}_{AXIS_NAMES[j]}":
                            min(angle_deg(v0_vecs[i], v0_vecs[j]), 180-angle_deg(v0_vecs[i], v0_vecs[j]))
                            for i in range(8) for j in range(i)},
    "l28_pairwise_angles": {f"{AXIS_NAMES[i]}_{AXIS_NAMES[j]}":
                             min(angle_deg(v28_vecs[i], v28_vecs[j]), 180-angle_deg(v28_vecs[i], v28_vecs[j]))
                             for i in range(8) for j in range(i)},
    "layer_sweep_angles_from_l0": {name: [float(a) if not math.isnan(a) else None
                                          for a in layer_angles_from_l0[name]]
                                   for name in AXIS_NAMES},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 81 complete.")
