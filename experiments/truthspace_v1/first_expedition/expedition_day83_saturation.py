#!/usr/bin/env python3
"""
Day 83 — Saturation Test: Where Does the Transformation Subspace End?

Day 82: 4 new axes (negation, passive, spatial, temporal) all nearly
completely outside the existing 8D subspace (3.8–31% overlap).
The 12 combined axes are mutually orthogonal.

NOW: Test 4 more transformation types against the combined 12D:
  1. degree/intensity: "warm" → "hot" → "burning" (scalar intensification)
  2. part-whole:  "finger" → "hand", "leaf" → "tree" (meronymy)
  3. question:    "She is tired" → "Is she tired?" (assertion → question)
  4. causation:   "The rain falls" → "The ground gets wet" (cause → effect)

The SATURATION POINT is the dimension count where new T2 axes start
landing INSIDE the existing subspace (fraction_explained > 0.5).

PREDICTION:
  If 12 = natural limit → next 4 axes will have high fraction_explained
  If transformation space is truly open → next 4 will also be NEW_DIM

ALSO: Estimate total dimensionality via the cumulative SVD spectrum of
all axes found so far (12 + new 4 = 16).

The 0.8% residual variance in hidden states ≈ 1536 × 0.008 ≈ 12.3 dims.
This predicts the transformation subspace has EXACTLY ~12 dimensions.
If correct: days 83 axes will mostly project onto existing 12D.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day83_saturation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

TARGET_LAYER = 28

ALL_12_NAMES = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "negation", "passive", "spatial", "temporal",
]

ALL_12_PAIRS = {
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
        ("The fast car",  "The faster car"),
        ("A big dog",     "A bigger dog"),
        ("The cold wind", "The colder wind"),
        ("A tall tree",   "A taller tree"),
        ("The old house", "The older house"),
        ("A bright star", "A brighter star"),
        ("The dark room", "The darker room"),
        ("A hard rock",   "A harder rock"),
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
        ("He is big",       "He is large"),
        ("She is small",    "She is tiny"),
        ("He runs fast",    "He runs quick"),
        ("It is cold",      "It is frigid"),
        ("She is happy",    "She is joyful"),
        ("He spoke loudly", "He spoke noisily"),
        ("It is hard",      "It is difficult"),
        ("He is old",       "He is aged"),
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
        ("It is hot",       "It is cold"),
        ("He runs fast",    "He runs slow"),
        ("The light is on", "The dark is on"),
        ("The news is good","The news is bad"),
        ("It is hard",      "It is soft"),
        ("She is happy",    "She is sad"),
        ("He is strong",    "He is weak"),
        ("It is the first", "It is the last"),
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
    "temporal": [
        ("Yesterday she walked to the market",     "Tomorrow she will walk to the market"),
        ("He studied hard last year",              "He will study hard next year"),
        ("They built the bridge long ago",         "They will build the bridge soon"),
        ("She cooked dinner an hour ago",          "She will cook dinner in an hour"),
        ("The rain fell hard last night",          "The rain will fall hard tonight"),
        ("He spoke at the meeting yesterday",      "He will speak at the meeting tomorrow"),
        ("The leaves fell in the autumn",          "The leaves will fall in the autumn"),
        ("The old man worked here years before",   "The old man will work here years after"),
    ],
}

# ── 4 candidate saturation axes ──────────────────────────────────────────────
SAT_PAIRS = {
    "degree": [
        ("It is warm outside today",        "It is hot outside today"),
        ("The food is good today",          "The food is excellent today"),
        ("He is a little tired now",        "He is extremely tired now"),
        ("The light was dim in the room",   "The light was blinding in the room"),
        ("She was slightly upset",          "She was furious"),
        ("The wind is gentle today",        "The wind is violent today"),
        ("The sound was soft",              "The sound was deafening"),
        ("He moved slowly at first",        "He moved instantly at first"),
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
}

def angle_deg(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(math.degrees(math.acos(float(np.clip(c, -1, 1)))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def get_hidden_last(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

def compute_t2_axis(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        h1 = get_hidden_last(s1, layer)
        h2 = get_hidden_last(s2, layer)
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if not diffs: return np.zeros(hidden_size, dtype=np.float32)
    v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
    return (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

def project_onto_subspace(v, U):
    coords  = U.T @ v
    v_proj  = U @ coords
    frac    = float(np.dot(v_proj, v_proj) / (np.dot(v, v) + 1e-10))
    return v_proj, frac

# ── Compute all 12 existing axes ─────────────────────────────────────────────
print("Computing 12 existing T2 axes at L28 ...")
existing = {}
for name in ALL_12_NAMES:
    existing[name] = compute_t2_axis(ALL_12_PAIRS[name], TARGET_LAYER)
    print(f"  {name}")
print()

# ── Build 12D orthonormal basis ───────────────────────────────────────────────
M12 = np.array([existing[n] for n in ALL_12_NAMES], dtype=np.float32)
U12, S12, _ = np.linalg.svd(M12.T, full_matrices=False)   # U12: (1536, 12)
print(f"12D subspace SVD: {' '.join(f'{v:.3f}' for v in S12/S12.sum())}\n")

# ── Compute 4 saturation candidate axes ──────────────────────────────────────
print("Computing 4 saturation candidate axes at L28 ...")
sat_axes = {}
for name in SAT_PAIRS:
    sat_axes[name] = compute_t2_axis(SAT_PAIRS[name], TARGET_LAYER)
    print(f"  {name}")
print()

# ── Test: fraction explained by 12D vs 8D subspace ───────────────────────────
M8 = M12[:8]   # first 8 existing axes
U8, _, _ = np.linalg.svd(M8.T, full_matrices=False)

print("=" * 72)
print("Saturation test: fraction explained by 8D vs 12D subspace")
print("=" * 72)
print(f"  {'axis':>12}  {'frac_8D':>9}  {'frac_12D':>9}  {'residual%':>10}  verdict")

frac_8d  = {}
frac_12d = {}
for name, v in sat_axes.items():
    if np.linalg.norm(v) < 1e-6:
        print(f"  {name:>12}  DEGENERATE"); continue
    _, f8  = project_onto_subspace(v, U8)
    _, f12 = project_onto_subspace(v, U12)
    frac_8d[name]  = f8
    frac_12d[name] = f12
    residual_pct = 100 * (1 - f12)
    verdict = ("IN_12D" if f12 > 0.70 else
               "PARTIAL" if f12 > 0.40 else
               "NEW_DIM")
    print(f"  {name:>12}  {f8:>9.4f}  {f12:>9.4f}  {residual_pct:>10.1f}%  {verdict}")

print()
saturated = sum(1 for f in frac_12d.values() if f > 0.70)
print(f"  Axes landing inside 12D: {saturated}/{len(frac_12d)}")
print(f"  SATURATION AT 12D: {'YES' if saturated >= 3 else 'NO — space extends beyond 12D'}")
print()

# ── Angles between saturation candidates and all 12 existing ─────────────────
print("=" * 72)
print("Angles between saturation candidates and existing 12 axes")
print("=" * 72)
print(f"  {'':>12}  " + "  ".join(f"{n[:6]:>7}" for n in ALL_12_NAMES))

for name, v in sat_axes.items():
    if np.linalg.norm(v) < 1e-6: continue
    angles = []
    for en in ALL_12_NAMES:
        a = angle_deg(v, existing[en])
        angles.append(min(a, 180 - a))
    nearest_i = int(np.argmin(angles))
    a_str = "  ".join(f"{a:>7.1f}" for a in angles)
    print(f"  {name:>12}  {a_str}")
    print(f"  {'':>12}  nearest: {ALL_12_NAMES[nearest_i]} ({angles[nearest_i]:.1f}°)")
print()

# ── New axes mutually orthogonal? ─────────────────────────────────────────────
print("=" * 72)
print("Saturation candidates: pairwise angles")
print("=" * 72)
sat_names = list(sat_axes.keys())
for i in range(len(sat_names)):
    for j in range(i):
        a = angle_deg(sat_axes[sat_names[i]], sat_axes[sat_names[j]])
        a = min(a, 180 - a)
        print(f"  {sat_names[j]:>12} ⊥ {sat_names[i]:<12}: {a:.1f}°")
print()

# ── Full 16D SVD ──────────────────────────────────────────────────────────────
print("=" * 72)
print("SVD: 12 existing + 4 new = 16 combined axes")
print("=" * 72)

all_16 = [v for v in list(existing.values()) + list(sat_axes.values())
          if np.linalg.norm(v) > 1e-6]
M16 = np.array(all_16, dtype=np.float32)
_, S16, _ = np.linalg.svd(M16.T, full_matrices=False)
S16_norm  = S16 / S16.sum()
cumvar    = np.cumsum(S16_norm)

print(f"  Singular values (first 16): {' '.join(f'{v:.3f}' for v in S16_norm[:16])}")
n90  = int(np.searchsorted(cumvar, 0.90)) + 1
n95  = int(np.searchsorted(cumvar, 0.95)) + 1
print(f"  90% variance at dimension: {n90}")
print(f"  95% variance at dimension: {n95}")
print(f"  12-dim explains: {100*cumvar[11]:.1f}%")
print(f"  16-dim explains: {100*cumvar[min(15,len(cumvar)-1)]:.1f}%")
print()

# Isotropic check: ratio of max/min singular value
sv_ratio = float(S16_norm[0] / S16_norm[min(15, len(S16_norm)-1)])
print(f"  Isotropy: max_sv / min_sv = {sv_ratio:.2f}  (1.0 = perfect isotropy)")
print()

# ── Estimate intrinsic dimensionality ────────────────────────────────────────
print("=" * 72)
print("Estimate: intrinsic dimensionality of transformation subspace")
print("=" * 72)
print(f"""
  From Day 72: identity manifold explains 99.2% of hidden state variance
  Transformation subspace: 0.8% × 1536 dims = {int(0.008 * 1536):.0f} dimensions

  From 16-axis SVD:
    8D  explains {100*cumvar[7]:.1f}% of axis-span variance
    12D explains {100*cumvar[11]:.1f}% of axis-span variance
    16D explains {100*cumvar[min(15,len(cumvar)-1)]:.1f}% of axis-span variance

  Saturation candidates inside 12D: {saturated}/{len(frac_12d)}

  ESTIMATE: transformation subspace has ~{n95} dimensions total
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "frac_explained_8D":  frac_8d,
    "frac_explained_12D": frac_12d,
    "saturated_in_12D": saturated,
    "saturation_verdict": "YES" if saturated >= 3 else "NO",
    "cumvar_12D": float(cumvar[11]),
    "cumvar_16D": float(cumvar[min(15, len(cumvar)-1)]),
    "n95_dim": n95,
    "isotropy_ratio": sv_ratio,
    "singular_values_16D": [float(v) for v in S16_norm[:16]],
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 83 complete.")
