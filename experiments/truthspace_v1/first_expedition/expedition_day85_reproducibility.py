#!/usr/bin/env python3
"""
Day 85 — Reproducibility Test: Are T2 Axes Real or High-Dimensional Noise?

CRITICAL CONTEXT (Day 84):
In R^1536, random unit vectors have E[angle] ≈ 88.5°, std ≈ 2.3°.
The 3-sigma range is 81.6–95.4°. Every "pairwise orthogonality" result
(80–90°) we have reported falls within the range expected for RANDOM vectors.

The saturation tests may be measuring high-dimensional geometry artifacts:
in R^1536 ANY two unit vectors are approximately orthogonal.

REPRODUCIBILITY TEST (the decisive experiment):
For each transformation type, compute two independent T2 axes:
  T2_A = axis from sentence pair SET A (original 8 pairs)
  T2_B = axis from sentence pair SET B (8 completely different pairs)

PREDICTIONS:
  H_REAL:  angle(T2_A, T2_B) << 90°   (< 45° expected for clean axes)
           axes reproduce across different sentence pairs
  H_NOISE: angle(T2_A, T2_B) ≈ 88.5° (high-dimensional random noise)
           axes are just arbitrary directions in R^1536

ALSO MEASURE:
  1. Coherence: ||mean(norm_diffs)|| before normalization
     Real: coherence >> sqrt(8/1536) ≈ 0.072 (random baseline)
  2. Cross-type discrimination: A-axis for type X should be closer
     to B-axis for type X than to any axis for type Y
  3. Random baselines: 4 sets of truly random sentence pairs
     (sentences that share NO semantic transformation relationship)
     Should show coherence ≈ 0.072 and angle ≈ 88.5° with everything

This is the GROUND TRUTH test that all previous experiments assumed
but never measured.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day85_reproducibility.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

TARGET_LAYER = 28

# ── SET A: original sentence pairs ───────────────────────────────────────────
SET_A = {
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
}

# ── SET B: completely new sentence pairs for the SAME transformation types ────
SET_B = {
    "gender": [
        ("The uncle arrived at the old farm",    "The aunt arrived at the old farm"),
        ("A husband cooked the evening meal",    "A wife cooked the evening meal"),
        ("The grandfather taught the children",  "The grandmother taught the children"),
        ("A waiter brought the food quickly",    "A waitress brought the food quickly"),
        ("The bull grazed in the green field",   "The cow grazed in the green field"),
        ("A duke traveled to the far palace",    "A duchess traveled to the far palace"),
        ("A wizard cast the powerful spell",     "A witch cast the powerful spell"),
        ("The monk prayed in the cold chapel",   "The nun prayed in the cold chapel"),
    ],
    "comparative": [
        ("A small boat",    "A smaller boat"),
        ("The young woman", "The younger woman"),
        ("A clean road",    "A cleaner road"),
        ("The weak signal", "The weaker signal"),
        ("A wide river",    "A wider river"),
        ("The loud noise",  "The louder noise"),
        ("A thick wall",    "A thicker wall"),
        ("The deep well",   "The deeper well"),
    ],
    "hypernym": [
        ("The cat ran through the alley",        "The animal ran through the alley"),
        ("A trout swam against the current",     "A fish swam against the current"),
        ("The piano filled the whole room",      "The instrument filled the whole room"),
        ("The truck blocked the narrow road",    "The vehicle blocked the narrow road"),
        ("A sparrow landed on the branch",       "A bird landed on the branch"),
        ("The emerald shone on the table",       "The gem shone on the table"),
        ("The nurse cared for the patient",      "The worker cared for the patient"),
        ("The saw cut through the plank",        "The tool cut through the plank"),
    ],
    "plural": [
        ("A horse grazed in the open meadow",   "Horses grazed in the open meadow"),
        ("The apple fell from the heavy branch","The apples fell from the heavy branch"),
        ("A stone sank in the cold river",      "Stones sank in the cold river"),
        ("The flower bloomed in early spring",  "The flowers bloomed in early spring"),
        ("A cloud drifted across the blue sky", "Clouds drifted across the blue sky"),
        ("The child ran through the city park", "The children ran through the city park"),
        ("A window opened on the upper floor",  "Windows opened on the upper floor"),
        ("The coin fell onto the hard ground",  "The coins fell onto the hard ground"),
    ],
    "synonym": [
        ("He is slim",       "He is thin"),
        ("She is smart",     "She is clever"),
        ("The car is fast",  "The car is rapid"),
        ("It is wet",        "It is moist"),
        ("He is tired",      "He is weary"),
        ("She spoke gently", "She spoke softly"),
        ("It is strange",    "It is odd"),
        ("He is rich",       "He is wealthy"),
    ],
    "concrete": [
        ("The chain held the heavy gate shut",   "The bond held the heavy gate shut"),
        ("A fence surrounded the open garden",   "A boundary surrounded the open garden"),
        ("The ladder reached the high roof",     "The path reached the high roof"),
        ("The anchor held the boat in place",    "The tie held the boat in place"),
        ("The gate swung open in the breeze",    "The entry swung open in the breeze"),
        ("The column supported the stone ceiling","The pillar supported the stone ceiling"),
        ("The thread connected the two pieces",  "The link connected the two pieces"),
        ("The shelf held the heavy old books",   "The support held the heavy old books"),
    ],
    "past_tense": [
        ("They fly over the mountains every year",  "They flew over the mountains every year"),
        ("She draws a picture on Sunday morning",   "She drew a picture on Sunday morning"),
        ("He sees the doctor every other week",     "He saw the doctor every other week"),
        ("We grow tomatoes in the back garden",     "We grew tomatoes in the back garden"),
        ("The river rises in the spring season",    "The river rose in the spring season"),
        ("She teaches at the local primary school", "She taught at the local primary school"),
        ("He throws the ball across the wide yard", "He threw the ball across the wide yard"),
        ("They drive through the city on weekends", "They drove through the city on weekends"),
    ],
    "antonym": [
        ("The room is bright and welcoming",    "The room is dim and welcoming"),
        ("The path was rough and difficult",    "The path was smooth and difficult"),
        ("The answer is right this time",       "The answer is wrong this time"),
        ("She arrived far too early",           "She arrived far too late"),
        ("The water is very deep here",         "The water is very shallow here"),
        ("The problem is very easy now",        "The problem is very hard now"),
        ("The box is open on the shelf",        "The box is closed on the shelf"),
        ("The weather was clear all day",       "The weather was cloudy all day"),
    ],
}

# ── RANDOM baselines: sentences with NO semantic transformation relationship ──
RANDOM_PAIRS = {
    "random_A": [
        ("The dog barked loudly at night",       "She opened the red window slowly"),
        ("A cloud passed over the mountain",     "He drove to the distant supermarket"),
        ("The music played in the old house",    "Flowers grew in the cold winter"),
        ("She read the blue book quickly",       "The river ran beside the green field"),
        ("He cooked rice for his family",        "The bird flew above the tall trees"),
        ("The clock ticked in the empty room",   "She smiled at the long story"),
        ("The train arrived at noon today",      "He painted the wooden floor quickly"),
        ("A stone fell into the still pond",     "She walked alone through the dark fog"),
    ],
    "random_B": [
        ("The chair is in the corner room",      "He threw the ball over the fence"),
        ("Snow fell on the quiet town below",    "The light was very bright today"),
        ("She wore a green coat outside",        "He finished the difficult project"),
        ("The door was open wide all day",       "A fish swam near the river bank"),
        ("He wrote the address on paper",        "The storm lasted for three days"),
        ("The coffee was too hot to drink",      "She planted seeds in the soft earth"),
        ("The cat climbed the old oak tree",     "He fixed the broken wooden door"),
        ("A boat sailed near the rocky shore",   "She sang a song in the evening"),
    ],
    "random_C": [
        ("The library closed at nine sharp",     "He ran ten miles every morning"),
        ("She painted her room bright yellow",   "A horse grazed in the open field"),
        ("The old phone rang in the hall",       "Snow covered the roof overnight"),
        ("He bought bread at the corner shop",   "The river flooded the low valley"),
        ("She found the key under the mat",      "He climbed to the very top"),
        ("The teacher wrote on the white board", "A bee landed on the red flower"),
        ("He fell asleep in the large chair",    "The wind shook the tall trees hard"),
        ("The moon was full and very bright",    "She finished her long homework"),
    ],
    "random_D": [
        ("A fox hid under the old fence",        "The ship sailed out of the harbor"),
        ("She counted all the coins slowly",     "He wore a hat in the cold rain"),
        ("The lamp gave off a warm glow",        "A leaf fell into the still water"),
        ("He locked the door twice at night",    "She baked cookies in the oven"),
        ("The map showed the old mountain road", "A rabbit ran across the green lawn"),
        ("She watered the plants every evening", "He missed the early morning bus"),
        ("The flag flew high above the tower",   "She wrote a long note to herself"),
        ("He tasted the soup and smiled wide",   "The bench stood empty in the rain"),
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
n_random_baseline_per_dim = 8
print(f"  hidden={hidden_size}")
print(f"  Expected angle for random unit vectors in R^{hidden_size}:")
exp_rand_angle = float(math.degrees(math.acos(0)))   # exactly 90°
std_rand = float(math.degrees(1 / math.sqrt(hidden_size)))   # approx std in degrees
print(f"  E[angle] ≈ 90°,  std ≈ {std_rand:.2f}°,  3σ range: [{90-3*std_rand:.1f}, {90+3*std_rand:.1f}]°\n")

def get_hidden_last(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

def compute_t2_axis_with_coherence(pairs, layer):
    """Returns (unit_axis, coherence) where coherence = ||mean(unit_diffs)||."""
    diffs = []
    for s1, s2 in pairs:
        h1 = get_hidden_last(s1, layer)
        h2 = get_hidden_last(s2, layer)
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if not diffs:
        return np.zeros(hidden_size, dtype=np.float32), 0.0
    mean_diff = np.mean(diffs, axis=0)
    coherence = float(np.linalg.norm(mean_diff))   # KEY METRIC
    nv = np.linalg.norm(mean_diff)
    axis = (mean_diff / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    return axis, coherence

# ── Random coherence baseline ─────────────────────────────────────────────────
print("Computing random unit vector coherence baseline ...")
# Theoretical: E[||mean of k unit vectors||] = sqrt(k/d)
k, d = 8, hidden_size
rand_coherence_theory = math.sqrt(k / d)
print(f"  Theoretical: E[coherence of {k} random unit vecs in R^{d}] = {rand_coherence_theory:.4f}\n")

# ── Compute T2 axes for SET A and SET B ──────────────────────────────────────
print("Computing T2 axes: SET A, SET B, and RANDOM pairs ...")
axes_A    = {}
axes_B    = {}
cohere_A  = {}
cohere_B  = {}
axes_rand = {}
cohere_rand = {}

AXIS_NAMES = list(SET_A.keys())

for name in AXIS_NAMES:
    axes_A[name],   cohere_A[name]   = compute_t2_axis_with_coherence(SET_A[name], TARGET_LAYER)
    axes_B[name],   cohere_B[name]   = compute_t2_axis_with_coherence(SET_B[name], TARGET_LAYER)
    print(f"  {name:>15}  cohA={cohere_A[name]:.4f}  cohB={cohere_B[name]:.4f}")

for name in RANDOM_PAIRS:
    axes_rand[name], cohere_rand[name] = compute_t2_axis_with_coherence(RANDOM_PAIRS[name], TARGET_LAYER)
    print(f"  {name:>15}  coh={cohere_rand[name]:.4f}  [RANDOM]")
print()

# ── Test 1: Coherence — real vs random ───────────────────────────────────────
print("=" * 72)
print(f"Test 1: Coherence (||mean_diff||): real axes vs random baseline")
print(f"  Theoretical random baseline: {rand_coherence_theory:.4f}")
print("=" * 72)
print(f"  {'type':>15}  {'coher_A':>9}  {'coher_B':>9}  {'mean':>9}  {'vs_rand_ratio':>14}")

coher_ratio_list = []
for name in AXIS_NAMES:
    ca, cb = cohere_A[name], cohere_B[name]
    mean_c = (ca + cb) / 2
    ratio  = mean_c / rand_coherence_theory
    coher_ratio_list.append(ratio)
    print(f"  {name:>15}  {ca:>9.4f}  {cb:>9.4f}  {mean_c:>9.4f}  {ratio:>14.1f}×")

print()
rand_coher_mean = float(np.mean(list(cohere_rand.values())))
rand_ratio = rand_coher_mean / rand_coherence_theory
print(f"  {'RANDOM (mean)':>15}  {'':>9}  {'':>9}  {rand_coher_mean:>9.4f}  {rand_ratio:>14.1f}×")
print(f"  Mean real/random coherence ratio: {float(np.mean(coher_ratio_list)):.1f}×")
print()

# ── Test 2: Reproducibility — angle between T2_A and T2_B ────────────────────
print("=" * 72)
print("Test 2: Reproducibility — angle between SET A and SET B axes")
print(f"  Random expected: 90.0°  ±{std_rand:.1f}°  (3σ range: {90-3*std_rand:.1f}–{90+3*std_rand:.1f}°)")
print("=" * 72)
print(f"  {'type':>15}  {'angle_A_B':>11}  {'vs_random_baseline':>20}  verdict")

repro_angles = {}
for name in AXIS_NAMES:
    a = min(angle_deg(axes_A[name], axes_B[name]),
            180 - angle_deg(axes_A[name], axes_B[name]))
    repro_angles[name] = a
    # How many std devs below 90°?
    sigmas_from_rand = (90 - a) / std_rand
    verdict = ("REPRODUCIBLE" if sigmas_from_rand > 5 else
               "LIKELY_REAL"  if sigmas_from_rand > 2 else
               "MARGINAL"     if sigmas_from_rand > 0 else
               "NOT_REPRO")
    print(f"  {name:>15}  {a:>11.2f}°  {sigmas_from_rand:>+20.1f}σ  {verdict}")

print()
mean_repro = float(np.mean(list(repro_angles.values())))
mean_sigma = (90 - mean_repro) / std_rand
print(f"  Mean A-B angle: {mean_repro:.2f}°  ({mean_sigma:.1f}σ below random baseline)")
print(f"  H_REAL  (mean < 45°, > 5σ below 90°): {'CONFIRMED' if mean_repro < 45 and mean_sigma > 5 else 'REJECTED'}")
print(f"  H_NOISE (mean ≈ 90°):                  {'CONFIRMED' if mean_repro > 85 else 'REJECTED'}")
print()

# ── Test 3: Random baselines — do they also reproduce? ───────────────────────
print("=" * 72)
print("Test 3: Random baseline cross-comparisons")
print("=" * 72)
rand_names = list(RANDOM_PAIRS.keys())
for i in range(len(rand_names)):
    for j in range(i):
        a = min(angle_deg(axes_rand[rand_names[i]], axes_rand[rand_names[j]]),
                180 - angle_deg(axes_rand[rand_names[i]], axes_rand[rand_names[j]]))
        sig = (90 - a) / std_rand
        print(f"  {rand_names[j]} ↔ {rand_names[i]}: {a:.2f}°  ({sig:+.1f}σ)")
print()

# ── Test 4: Cross-type discrimination ────────────────────────────────────────
print("=" * 72)
print("Test 4: Cross-type discrimination")
print("  Can SET_A axes correctly predict their SET_B partners?")
print("  (nearest neighbor in B-space from A-axis)")
print("=" * 72)

n_correct = 0
for name in AXIS_NAMES:
    a_ax = axes_A[name]
    angles_to_B = {n: min(angle_deg(a_ax, axes_B[n]), 180-angle_deg(a_ax, axes_B[n]))
                   for n in AXIS_NAMES}
    nearest = min(angles_to_B, key=angles_to_B.get)
    correct = (nearest == name)
    if correct: n_correct += 1
    self_angle  = angles_to_B[name]
    near_angle  = angles_to_B[nearest]
    print(f"  {name:>15}: self={self_angle:.1f}°  nearest_B={nearest}({near_angle:.1f}°)  {'✓' if correct else '✗'}")

print(f"\n  Correct (A matches own B): {n_correct}/{len(AXIS_NAMES)}")
print(f"  Random expected correct by chance: 1/{len(AXIS_NAMES)} = {1/len(AXIS_NAMES):.0%}")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 85 Summary")
print("=" * 72)
real_coherence = float(np.mean([(cohere_A[n]+cohere_B[n])/2 for n in AXIS_NAMES]))
print(f"""
  Random baseline coherence: {rand_coherence_theory:.4f}  (theoretical)
  Random observed coherence: {rand_coher_mean:.4f}  (from 4 random pair sets)
  Real semantic coherence:   {real_coherence:.4f}  (mean across 8 types × 2 sets)
  Coherence ratio (real/random): {real_coherence/rand_coherence_theory:.1f}×

  Reproducibility (A-B angle): {mean_repro:.2f}°  ({mean_sigma:.1f}σ below 90°)
  Cross-type discrimination: {n_correct}/{len(AXIS_NAMES)} correct

  VERDICT:
  {'T2 AXES ARE REAL SEMANTIC DIRECTIONS' if mean_repro < 45 else
   'T2 AXES ARE LIKELY REAL (weak reproducibility)' if mean_repro < 70 else
   'T2 AXES ARE HIGH-DIMENSIONAL NOISE'}
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "coherence_A":       {k: float(v) for k, v in cohere_A.items()},
    "coherence_B":       {k: float(v) for k, v in cohere_B.items()},
    "coherence_random":  {k: float(v) for k, v in cohere_rand.items()},
    "rand_coherence_theory": float(rand_coherence_theory),
    "rand_coherence_observed": float(rand_coher_mean),
    "mean_real_coherence": float(real_coherence),
    "coherence_ratio": float(real_coherence / rand_coherence_theory),
    "reproducibility_angles": {k: float(v) for k, v in repro_angles.items()},
    "mean_repro_angle": float(mean_repro),
    "sigmas_below_random": float(mean_sigma),
    "cross_type_correct": int(n_correct),
    "cross_type_total": len(AXIS_NAMES),
    "h_real_confirmed": bool(mean_repro < 45 and mean_sigma > 5),
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 85 complete.")
