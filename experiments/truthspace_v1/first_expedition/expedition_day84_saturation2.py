#!/usr/bin/env python3
"""
Day 84 — Saturation Round 4: Possession, Definiteness, Modality, Aspect

Days 82-83: 8 new transformation axes added (negation, passive, spatial,
temporal, degree, part_whole, question, causation). None project onto
prior subspace (all < 17% explained). Combined 16D SVD: 95% at dim 15.

THIS ROUND: 4 more candidates against the combined 16D subspace:
  1. possession:   "John has a book" → "That is John's book"
  2. definiteness: "A cat walked by" → "The cat walked by"
  3. modality:     "She walks"       → "She must walk"
  4. aspect:       "She reads"       → "She is reading"

SATURATION PREDICTION:
  If transformation space is ~16D total → 3-4 axes should land inside 16D
  If it extends further     → all 4 will be NEW_DIM again

STOPPING CRITERION:
  If ≥3 of 4 candidates have frac_16D > 0.60 → SATURATED
  Report intrinsic dimensionality estimate.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day84_saturation2.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

TARGET_LAYER = 28

ALL_16_NAMES = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "negation", "passive", "spatial", "temporal",
    "degree", "part_whole", "question", "causation",
]

ALL_16_PAIRS = {
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
        ("The light is on", "The dark is on"), ("The news is good", "The news is bad"),
        ("It is hard", "It is soft"), ("She is happy", "She is sad"),
        ("He is strong", "He is weak"), ("It is the first", "It is the last"),
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

# ── 4 new saturation candidates ───────────────────────────────────────────────
NEW_PAIRS = {
    "possession": [
        ("John has a very nice red car",        "That is John's very nice red car"),
        ("The teacher owns the old book",       "That is the teacher's old book"),
        ("She has a small white cat",           "That is her small white cat"),
        ("The king owns the golden crown",      "That is the king's golden crown"),
        ("The child has a favorite blue toy",   "That is the child's favorite blue toy"),
        ("The dog has a long leather collar",   "That is the dog's long leather collar"),
        ("He has a big wooden house",           "That is his big wooden house"),
        ("The shop has a special red door",     "That is the shop's special red door"),
    ],
    "definiteness": [
        ("A dog walked down the road",          "The dog walked down the road"),
        ("A cat sat by the window",             "The cat sat by the window"),
        ("A man stood at the corner",           "The man stood at the corner"),
        ("A bird sang in the morning",          "The bird sang in the morning"),
        ("A book sat on the table",             "The book sat on the table"),
        ("A car stopped at the light",          "The car stopped at the light"),
        ("A child played in the park",          "The child played in the park"),
        ("A storm came without warning",        "The storm came without warning"),
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
    "aspect": [
        ("She reads the long book",             "She is reading the long book"),
        ("He runs through the open park",       "He is running through the open park"),
        ("They build a tall brick wall",        "They are building a tall brick wall"),
        ("The child plays with the small toy",  "The child is playing with the small toy"),
        ("The chef cooks the evening meal",     "The chef is cooking the evening meal"),
        ("She writes a long difficult letter",  "She is writing a long difficult letter"),
        ("He paints the old wooden fence",      "He is painting the old wooden fence"),
        ("The dog chases the small brown cat",  "The dog is chasing the small brown cat"),
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

def project_onto(v, U):
    coords = U.T @ v; v_proj = U @ coords
    return v_proj, float(np.dot(v_proj, v_proj) / (np.dot(v, v) + 1e-10))

# ── Build 16D subspace ────────────────────────────────────────────────────────
print("Computing 16 existing T2 axes at L28 ...")
existing = {}
for name in ALL_16_NAMES:
    existing[name] = compute_t2_axis(ALL_16_PAIRS[name], TARGET_LAYER)
    print(f"  {name}")

M16 = np.array([existing[n] for n in ALL_16_NAMES], dtype=np.float32)
U16, S16, _ = np.linalg.svd(M16.T, full_matrices=False)
print(f"\n16D SVD: {' '.join(f'{v:.3f}' for v in (S16/S16.sum())[:8])} ...\n")

# ── Compute new candidates ────────────────────────────────────────────────────
print("Computing 4 new candidate axes at L28 ...")
new_axes = {}
for name in NEW_PAIRS:
    new_axes[name] = compute_t2_axis(NEW_PAIRS[name], TARGET_LAYER)
    print(f"  {name}")
print()

# ── Saturation test: fraction explained by 16D ───────────────────────────────
print("=" * 72)
print("Saturation test: fraction explained by 16D subspace")
print("=" * 72)
print(f"  {'axis':>14}  {'frac_16D':>10}  {'residual%':>11}  verdict")

frac_results = {}
for name, v in new_axes.items():
    if np.linalg.norm(v) < 1e-6:
        print(f"  {name:>14}  DEGENERATE"); continue
    _, f16 = project_onto(v, U16)
    frac_results[name] = f16
    res_pct = 100 * (1 - f16)
    verdict = ("SATURATED" if f16 > 0.60 else
               "PARTIAL"   if f16 > 0.35 else
               "NEW_DIM")
    print(f"  {name:>14}  {f16:>10.4f}  {res_pct:>11.1f}%  {verdict}")

sat_count = sum(1 for f in frac_results.values() if f > 0.60)
print(f"\n  Axes saturated in 16D: {sat_count}/{len(frac_results)}")
print(f"  SATURATION AT 16D: {'YES ✓' if sat_count >= 3 else 'NO — space extends beyond 16D'}")
print()

# ── Pairwise angles between new axes ─────────────────────────────────────────
print("=" * 72)
print("New axes pairwise angles")
print("=" * 72)
new_names = list(new_axes.keys())
for i in range(len(new_names)):
    for j in range(i):
        v1, v2 = new_axes[new_names[i]], new_axes[new_names[j]]
        a = min(angle_deg(v1, v2), 180 - angle_deg(v1, v2))
        print(f"  {new_names[j]:>14} ⊥ {new_names[i]:<14}: {a:.1f}°")
print()

# ── Angles vs all existing 16 ─────────────────────────────────────────────────
print("=" * 72)
print("New axes vs existing 16: nearest neighbor")
print("=" * 72)
for name, v in new_axes.items():
    if np.linalg.norm(v) < 1e-6: continue
    angles = [min(angle_deg(v, existing[en]), 180 - angle_deg(v, existing[en]))
              for en in ALL_16_NAMES]
    nearest_i = int(np.argmin(angles))
    print(f"  {name:>14}: nearest={ALL_16_NAMES[nearest_i]} ({angles[nearest_i]:.1f}°)  "
          f"min={min(angles):.1f}°  mean={float(np.mean(angles)):.1f}°")
print()

# ── Full 20D SVD ──────────────────────────────────────────────────────────────
all_vecs = [v for v in list(existing.values()) + list(new_axes.values())
            if np.linalg.norm(v) > 1e-6]
M20 = np.array(all_vecs, dtype=np.float32)
_, S20, _ = np.linalg.svd(M20.T, full_matrices=False)
S20_norm  = S20 / S20.sum()
cumvar    = np.cumsum(S20_norm)

n90 = int(np.searchsorted(cumvar, 0.90)) + 1
n95 = int(np.searchsorted(cumvar, 0.95)) + 1
print("=" * 72)
print(f"SVD: 16 existing + 4 new = {M20.shape[0]} combined axes")
print("=" * 72)
print(f"  Singular values: {' '.join(f'{v:.3f}' for v in S20_norm[:20])}")
print(f"  90% variance at dim: {n90}")
print(f"  95% variance at dim: {n95}")
print(f"  16-dim explains: {100*cumvar[15]:.1f}%")
print(f"  20-dim explains: {100*cumvar[min(19,len(cumvar)-1)]:.1f}%")
iso = float(S20_norm[0] / S20_norm[min(19, len(S20_norm)-1)])
print(f"  Isotropy ratio: {iso:.2f}")
print()

# ── Running count of confirmed orthogonal dimensions ──────────────────────────
print("=" * 72)
print("Running total: confirmed orthogonal transformation dimensions")
print("=" * 72)
print(f"""
  Day 73 (original 8):  gender, comparative, hypernym, plural,
                        synonym, concrete, past_tense, antonym
  Day 82 (+4):          negation, passive, spatial, temporal
  Day 83 (+4):          degree, part_whole, question, causation
  Day 84 (+4):          possession, definiteness, modality, aspect

  Total axes added:  20
  Saturation at 16D: {'YES' if sat_count >= 3 else 'NO'}
  SVD-based estimate of true dimensionality: ~{n95}
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "frac_16D": frac_results,
    "saturated_in_16D": sat_count,
    "saturation_verdict": "YES" if sat_count >= 3 else "NO",
    "n90_dim": n90,
    "n95_dim": n95,
    "cumvar_16D": float(cumvar[15]) if len(cumvar) > 15 else None,
    "cumvar_20D": float(cumvar[min(19, len(cumvar)-1)]),
    "isotropy_20D": iso,
    "singular_values_20D": [float(v) for v in S20_norm[:20]],
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 84 complete.")
