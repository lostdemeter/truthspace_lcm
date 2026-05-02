#!/usr/bin/env python3
"""
Day 82 — Completeness Test: Is the 8D Transformation Subspace Complete?

Day 73 found 8 orthogonal transformation axes (comparative, plural,
past_tense, gender, antonym, hypernym, synonym, concrete_abstract).
Day 81 confirmed all 28 pairwise angles are 72–90° at L28.

KEY QUESTION: Are 8 dimensions SUFFICIENT, or do other transformation
types require additional orthogonal dimensions?

TEST: Introduce 4 new transformation types at L28:
  1. negation       (positive → negative: "is fast" → "is not fast")
  2. passive        (active → passive: "chased the mouse" → "was chased")
  3. spatial        (on → under, inside → outside, above → below)
  4. temporal       (past → future: "walked" → "will walk")

For each new type:
  1. Compute T2 axis at L28 using 8 sentence pairs
  2. Project onto the span of the existing 8 T2 axes
  3. Measure: fraction of variance EXPLAINED by existing subspace
  4. Measure: residual (the component NOT in the existing 8D subspace)

PREDICTIONS:
  H_COMPLETE:  fraction explained ≈ 1.0 for all 4 new types
               (they all lie within the existing 8D subspace)
  H_EXTEND:    fraction explained << 1.0 for at least one new type
               (the existing 8D is incomplete; we need more axes)

ALSO: Where in the existing 8 does each new type project?
      (reveals which existing axes are most similar to negation, etc.)
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day82_completeness.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

TARGET_LAYER = 28    # working frame (Day 81 confirmed this)

AXIS_NAMES = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete",   "past_tense", "antonym",
]

# ── Existing 8 T2 axis sentence pairs ────────────────────────────────────────
EXISTING_PAIRS = {
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

# ── 4 NEW transformation types ────────────────────────────────────────────────
NEW_PAIRS = {
    "negation": [
        ("The dog is fast",              "The dog is not fast"),
        ("She can swim well",            "She cannot swim well"),
        ("He knows the answer",          "He does not know the answer"),
        ("The food is good",             "The food is not good"),
        ("They work hard",               "They do not work hard"),
        ("The water is cold",            "The water is not cold"),
        ("The house looks old",          "The house does not look old"),
        ("It will rain today",           "It will not rain today"),
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
        d  = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    if not diffs: return np.zeros(hidden_size, dtype=np.float32)
    v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
    return (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

# ── Compute existing 8 T2 axes at L28 ────────────────────────────────────────
print(f"Computing existing 8 T2 axes at L{TARGET_LAYER} ...")
existing_axes = {}
for name in AXIS_NAMES:
    existing_axes[name] = compute_t2_axis(EXISTING_PAIRS[name], TARGET_LAYER)
    print(f"  {name}")
print()

# ── Build orthonormal basis of existing 8D subspace ──────────────────────────
M = np.array([existing_axes[n] for n in AXIS_NAMES], dtype=np.float32)   # (8, 1536)
# SVD: M.T = U @ S @ Vt, U has orthonormal columns spanning span(M.T) = span(M rows)
U_sub, S_sub, _ = np.linalg.svd(M.T, full_matrices=False)   # U_sub: (1536, 8)
# U_sub columns are orthonormal basis of span(existing axes)
# Projection onto subspace: v_proj = U_sub @ U_sub.T @ v

def project_onto_subspace(v, U):
    """Project v onto the column space of U."""
    coords   = U.T @ v           # (k,)
    v_proj   = U @ coords        # (1536,)
    frac_expl = float(np.dot(v_proj, v_proj) / (np.dot(v, v) + 1e-10))
    return v_proj, frac_expl

# Verify existing axes explain each other (sanity check)
print("Sanity check: existing axes project onto subspace (should ≈ 1.0):")
for name in AXIS_NAMES:
    _, f = project_onto_subspace(existing_axes[name], U_sub)
    print(f"  {name:>15}: {f:.4f}")
print()

# ── Compute 4 new T2 axes at L28 ─────────────────────────────────────────────
print(f"Computing 4 new T2 axes at L{TARGET_LAYER} ...")
new_axes = {}
for name in NEW_PAIRS:
    new_axes[name] = compute_t2_axis(NEW_PAIRS[name], TARGET_LAYER)
    print(f"  {name}")
print()

# ── Test 1: Fraction explained by existing 8D subspace ────────────────────────
print("=" * 72)
print("Test 1: How much of each new axis is explained by existing 8D?")
print("=" * 72)
print(f"  {'new_axis':>12}  {'frac_explained':>16}  {'residual%':>10}  verdict")

frac_explained = {}
for name, v_new in new_axes.items():
    if np.linalg.norm(v_new) < 1e-6:
        print(f"  {name:>12}  DEGENERATE")
        continue
    v_proj, frac = project_onto_subspace(v_new, U_sub)
    residual_pct = 100 * (1 - frac)
    verdict = ("IN_SUBSPACE" if frac > 0.80 else
               "PARTIAL"    if frac > 0.50 else
               "NEW_DIM")
    frac_explained[name] = frac
    print(f"  {name:>12}  {frac:>16.4f}  {residual_pct:>10.1f}%  {verdict}")

print()
completeness = all(f > 0.80 for f in frac_explained.values())
print(f"  H_COMPLETE (all > 80%): {'CONFIRMED' if completeness else 'REJECTED'}")
print()

# ── Test 2: Angles between new axes and existing axes ────────────────────────
print("=" * 72)
print("Test 2: Angles between new axes and each existing axis (L28)")
print("=" * 72)
print(f"  {'':>12}  " + "  ".join(f"{n[:6]:>7}" for n in AXIS_NAMES) + "  nearest")

for name, v_new in new_axes.items():
    if np.linalg.norm(v_new) < 1e-6:
        print(f"  {name:>12}  DEGENERATE"); continue
    angles = []
    for en in AXIS_NAMES:
        a = angle_deg(v_new, existing_axes[en])
        a = min(a, 180 - a)
        angles.append(a)
    nearest_i = int(np.argmin(angles))
    angles_str = "  ".join(f"{a:>7.1f}" for a in angles)
    print(f"  {name:>12}  {angles_str}  {AXIS_NAMES[nearest_i]}({angles[nearest_i]:.1f}°)")
print()

# ── Test 3: Pairwise angles between new and existing axes ─────────────────────
print("=" * 72)
print("Test 3: New axes pairwise angles (are new axes orthogonal to each other?)")
print("=" * 72)
new_names = list(new_axes.keys())
print(f"  {'':>12}  " + "  ".join(f"{n[:8]:>9}" for n in new_names))
for i, n1 in enumerate(new_names):
    v1 = new_axes[n1]
    if np.linalg.norm(v1) < 1e-6: continue
    row = f"  {n1:>12}"
    for j, n2 in enumerate(new_names):
        v2 = new_axes[n2]
        if j == i: row += f"{'—':>9}"
        elif j > i: row += " " * 10
        else:
            a = min(angle_deg(v1, v2), 180 - angle_deg(v1, v2))
            row += f"{a:>9.1f}"
    print(row)
print()

# ── Test 4: Full 12D subspace (existing 8 + new 4 residuals) ─────────────────
print("=" * 72)
print("Test 4: After adding new axes, how much additional subspace is needed?")
print("=" * 72)

all_axes    = list(existing_axes.values()) + list(new_axes.values())
M12         = np.array([v for v in all_axes if np.linalg.norm(v) > 1e-6], dtype=np.float32)
_, S12, _   = np.linalg.svd(M12.T, full_matrices=False)
S12_norm    = S12 / S12.sum()

print(f"  SVD of [{len(AXIS_NAMES)} existing + {len(new_axes)} new] = {M12.shape[0]} total axes:")
print(f"  Singular value spectrum (first 14): {' '.join(f'{v:.3f}' for v in S12_norm[:14])}")

# How many dimensions needed for 95% variance?
cumvar = np.cumsum(S12_norm)
n95 = int(np.searchsorted(cumvar, 0.95)) + 1
print(f"  95% variance reached at dimension: {n95}")
print(f"  8-dim existing:   {100*cumvar[7]:.1f}%")
print(f"  12-dim (8+4):     {100*cumvar[11]:.1f}%")
print()

# ── Test 5: Residual components of new axes (orthogonal to existing 8) ────────
print("=" * 72)
print("Test 5: Residual axes (components NOT in existing 8D subspace)")
print("=" * 72)
residual_axes = {}
for name, v_new in new_axes.items():
    if np.linalg.norm(v_new) < 1e-6: continue
    v_proj, frac = project_onto_subspace(v_new, U_sub)
    v_res = v_new - v_proj
    nv_res = np.linalg.norm(v_res)
    if nv_res > 1e-6:
        residual_axes[name] = v_res / nv_res
        print(f"  {name}: residual norm = {nv_res:.4f}  ({100*(1-frac):.1f}% of v_new)")
print()

# Pairwise angles between residuals
if len(residual_axes) >= 2:
    print("  Pairwise angles between residual vectors:")
    rnames = list(residual_axes.keys())
    for i in range(len(rnames)):
        for j in range(i):
            a = min(angle_deg(residual_axes[rnames[i]], residual_axes[rnames[j]]),
                    180 - angle_deg(residual_axes[rnames[i]], residual_axes[rnames[j]]))
            print(f"    {rnames[j]} ⊥ {rnames[i]}: {a:.1f}°")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 82 Summary")
print("=" * 72)
print(f"""
  Existing 8 axes cover these fractions of new axes:
  {'':>12}  {'frac_explained':>16}  verdict""")
for name, frac in frac_explained.items():
    v = "IN_SUBSPACE" if frac > 0.80 else "PARTIAL" if frac > 0.50 else "NEW_DIM"
    print(f"  {name:>12}  {frac:>16.4f}  {v}")

print(f"""
  H_COMPLETE (8D sufficient): {'CONFIRMED' if completeness else 'REJECTED'}

  SVD spectrum of 12 combined axes:
    8-dim explains: {100*cumvar[7]:.1f}%
    12-dim explains: {100*cumvar[11]:.1f}%
    95% variance needs: {n95} dimensions
""")

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "frac_explained": frac_explained,
    "h_complete_confirmed": completeness,
    "cumvar_8dim":  float(cumvar[7]),
    "cumvar_12dim": float(cumvar[11] if len(cumvar) > 11 else float("nan")),
    "n_dim_for_95pct": n95,
    "singular_values_12dim": [float(v) for v in S12_norm[:14]],
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 82 complete.")
