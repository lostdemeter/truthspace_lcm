#!/usr/bin/env python3
"""
Day 120 — T2 Layer-Sweep: When Do Semantic Axes Emerge?

Day 78 found optimal layers for each T2 axis. Day 114b confirmed
projections work best at per-axis correct layers.

QUESTION: Do T2 axes emerge GRADUALLY across the network, or do they
appear sharply at specific layers (phase transitions)?

EXPERIMENT: For each of the 12 T2 axes, compute the axis vector at
each test layer, then measure:
  1. Cramér's V (discrete class separation)
  2. Cohen's d (continuous projection delta)
  3. Cross-layer cosine (does the axis DIRECTION change across layers?)

Test layers: 1, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 27, 28

Predictions:
  A. Early-layer axes (plural L1): strong at L1, degraded at later layers
  B. Late-layer axes (past_tense L28): weak at early layers, emerge late
  C. The axis DIRECTION may rotate as the representation builds up
  D. Some axes may show a phase transition (sharp V increase between layers)

Also compute: T2 gram matrix (inter-axis cosines) at each test layer
  - Are axes MORE orthogonal at their optimal layer?
  - Is orthogonality a property of a specific layer range?
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import chi2_contingency

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day120_t2_layer_sweep.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

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
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The news is good","The news is bad"),("She is happy","She is sad"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The child cries very loudly","The mother comes running in"),
        ("The glass breaks on hard stone","The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The dog is hungry now","Is the dog hungry now"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
    ],
}

# Test tokens per axis (from Day 114b vocabulary)
AXIS_TEST_TOKENS = {
    "gender":      ["king", "queen", "man", "woman", "boy", "girl",
                    "father", "mother", "son", "daughter", "actor", "actress"],
    "comparative": ["fast", "faster", "big", "bigger", "old", "older",
                    "cold", "colder", "tall", "taller", "bright", "brighter"],
    "hypernym":    ["dog", "animal", "rose", "flower", "car", "vehicle",
                    "eagle", "bird", "ruby", "gem", "hammer", "tool"],
    "plural":      ["dog", "dogs", "cat", "cats", "tree", "trees",
                    "bird", "birds", "book", "books", "star", "stars"],
    "synonym":     ["big", "large", "small", "tiny", "fast", "quick",
                    "cold", "frigid", "happy", "joyful", "old", "aged"],
    "concrete":    ["stone", "burden", "road", "journey", "wall", "barrier",
                    "flame", "hope", "root", "base", "bridge", "bond"],
    "past_tense":  ["walk", "walked", "run", "ran", "eat", "ate",
                    "see", "saw", "build", "built", "swim", "swam"],
    "antonym":     ["hot", "cold", "fast", "slow", "good", "bad",
                    "happy", "sad", "strong", "weak", "old", "new"],
    "passive":     ["chased", "chased", "cooked", "cooked", "destroyed", "destroyed",
                    "painted", "painted", "broken", "broken", "helped", "helped"],
    "causation":   ["rain", "wet", "fire", "ash", "wind", "fall",
                    "cry", "comfort", "break", "spill", "heat", "melt"],
    "question":    ["is", "was", "can", "could", "does", "did",
                    "are", "were", "has", "had", "will", "would"],
    "negation":    ["not", "never", "no", "none", "neither", "nor",
                    "nothing", "nobody", "nowhere", "without", "lack", "absent"],
}

# Assign binary labels for each axis's test tokens
AXIS_LABELS = {
    "gender":     ["M","F","M","F","M","F","M","F","M","F","M","F"],
    "comparative":["base","comp","base","comp","base","comp","base","comp","base","comp","base","comp"],
    "hypernym":   ["specific","generic","specific","generic","specific","generic","specific","generic","specific","generic","specific","generic"],
    "plural":     ["sing","plur","sing","plur","sing","plur","sing","plur","sing","plur","sing","plur"],
    "synonym":    ["base","syn","base","syn","base","syn","base","syn","base","syn","base","syn"],
    "concrete":   ["conc","abst","conc","abst","conc","abst","conc","abst","conc","abst","conc","abst"],
    "past_tense": ["pres","past","pres","past","pres","past","pres","past","pres","past","pres","past"],
    "antonym":    ["pos","neg","pos","neg","pos","neg","pos","neg","pos","neg","pos","neg"],
    "passive":    ["act","pass","act","pass","act","pass","act","pass","act","pass","act","pass"],
    "causation":  ["cause","effect","cause","effect","cause","effect","cause","effect","cause","effect","cause","effect"],
    "question":   ["decl","quest","decl","quest","decl","quest","decl","quest","decl","quest","decl","quest"],
    "negation":   ["neg","neg","neg","neg","neg","neg","neg","neg","neg","neg","neg","neg"],
}

TEST_LAYERS = [1, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 27, 28]

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

def phi_bin(x):
    if   x >  INV_PHI:  return "H"
    elif x < -INV_PHI2: return "L"
    else:               return "U"

def cramers_v(proj_vals, labels):
    bins  = [phi_bin(p) for p in proj_vals]
    cats  = sorted(set(labels)); ternary = ["H","U","L"]
    table = [[sum(1 for b,l in zip(bins,labels) if b==t and l==c)
              for c in cats] for t in ternary]
    arr = np.array(table)
    if arr.sum() == 0 or arr.shape[1] < 2: return 0.0
    try:
        chi2, _, _, _ = chi2_contingency(arr)
        n = arr.sum(); k = min(arr.shape)
        return float(math.sqrt(chi2 / (n * (k-1)))) if n > 0 else 0.0
    except: return 0.0

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def get_last_hs_at_all_layers(text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
            for L in TEST_LAYERS}

print("Computing T2 axes at each test layer ...")
# axes_at_layer[axis_name][layer] = unit vector
axes_at_layer = {ax: {} for ax in AXIS_NAMES_12}
for ax_name in AXIS_NAMES_12:
    pairs = AXIS_SENTENCE_PAIRS.get(ax_name, [])
    for L in TEST_LAYERS:
        diffs = []
        for s1, s2 in pairs:
            try:
                inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
                with torch.no_grad():
                    o1 = model(**inp1, output_hidden_states=True)
                    o2 = model(**inp2, output_hidden_states=True)
                h1 = o1.hidden_states[L][0,-1,:].numpy().astype(np.float32)
                h2 = o2.hidden_states[L][0,-1,:].numpy().astype(np.float32)
                d = h2-h1; n = np.linalg.norm(d)
                if n > 1e-6: diffs.append(d/n)
            except: pass
        v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, np.float32)
        nv = np.linalg.norm(v)
        axes_at_layer[ax_name][L] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

print("Extracting token hidden states at all test layers ...")
token_hs = {}
for ax_name in AXIS_NAMES_12:
    tokens = AXIS_TEST_TOKENS[ax_name]
    token_hs[ax_name] = {}
    for w in set(tokens):
        inp = tok(" " + w, return_tensors="pt")
        try:
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            pos = inp["input_ids"].shape[1] - 1
            token_hs[ax_name][w] = {
                L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                for L in TEST_LAYERS
            }
        except: pass
print("  Done.\n")

# ── Exp 1: Cramér's V per axis per layer ──────────────────────────────────────
print("=" * 80)
print("Exp 1: Cramér's V per Axis per Layer")
print("       (measures discrete H/U/L class separation at each layer)")
print("=" * 80)

# Print header
header = f"  {'axis':>14}  {'opt_L':>5}  " + "  ".join(f"L{L:02d}" for L in TEST_LAYERS)
print(header)
print("  " + "-" * (len(header)-2))

cramers_v_results = {}
for ax_name in AXIS_NAMES_12:
    L_opt = DAY78_LAYERS[ax_name]
    tokens = AXIS_TEST_TOKENS[ax_name]
    labels = AXIS_LABELS[ax_name]
    row = {}
    for L in TEST_LAYERS:
        axis = axes_at_layer[ax_name][L]
        projs = []
        valid_labels = []
        for w, lbl in zip(tokens, labels):
            if w not in token_hs[ax_name]: continue
            h = token_hs[ax_name][w].get(L)
            if h is None: continue
            projs.append(float(np.dot(normed(h), axis)))
            valid_labels.append(lbl)
        V = cramers_v(projs, valid_labels) if len(set(valid_labels)) > 1 else 0.0
        row[L] = V
    cramers_v_results[ax_name] = row
    vals_str = "  ".join(f"{row[L]:.3f}" for L in TEST_LAYERS)
    peak_L = max(row, key=row.get)
    peak_v = row[peak_L]
    opt_v  = row.get(L_opt, 0)
    print(f"  {ax_name:>14}  L{L_opt:02d}  {vals_str}   peak=L{peak_L:02d}({peak_v:.3f})")

# ── Exp 2: Cross-layer axis direction cosine ──────────────────────────────────
print()
print("=" * 80)
print("Exp 2: Cross-Layer Axis Direction Cosine")
print("       (does the axis DIRECTION rotate as depth increases?)")
print("=" * 80)
print("       Anchored to optimal layer. High = axis stable, Low = axis rotates.")
print()

cross_layer_results = {}
for ax_name in AXIS_NAMES_12:
    L_opt = DAY78_LAYERS[ax_name]
    ax_opt = axes_at_layer[ax_name][L_opt]
    row = {}
    for L in TEST_LAYERS:
        ax_L = axes_at_layer[ax_name][L]
        cos  = float(abs(np.dot(ax_opt, ax_L)))
        row[L] = cos
    cross_layer_results[ax_name] = row
    vals_str = "  ".join(f"{row[L]:.3f}" for L in TEST_LAYERS)
    print(f"  {ax_name:>14}  L{L_opt:02d}  {vals_str}")

# ── Exp 3: T2 Gram matrix (inter-axis orthogonality) at each layer ────────────
print()
print("=" * 80)
print("Exp 3: T2 Gram Matrix Off-diagonal Mean per Layer")
print("       (measures how orthogonal the 12 axes are at each layer)")
print("=" * 80)
print()
print(f"  {'layer':>8}  {'offdiag_mean':>14}  {'offdiag_max':>13}  {'orthogonality':>15}")
print(f"  {'-'*55}")

gram_results = {}
for L in TEST_LAYERS:
    axvecs = np.stack([axes_at_layer[ax][L] for ax in AXIS_NAMES_12])
    G = axvecs @ axvecs.T
    np.fill_diagonal(G, 0)
    upper = G[np.triu_indices(12, k=1)]
    off_mean = float(np.mean(np.abs(upper)))
    off_max  = float(np.max(np.abs(upper)))
    orth_tag = "NEAR-ORTHO" if off_mean < 0.08 else "MODERATE" if off_mean < 0.15 else "CORRELATED"
    gram_results[L] = {"offdiag_mean": off_mean, "offdiag_max": off_max}
    print(f"  {f'L{L:02d}':>8}  {off_mean:>14.4f}  {off_max:>13.4f}  {orth_tag:>15}")

# ── Exp 4: Peak layer vs Day78 optimal layer ─────────────────────────────────
print()
print("=" * 80)
print("Exp 4: Peak Cramér's V Layer vs Day78 Optimal Layer")
print("=" * 80)
print()
print(f"  {'axis':>14}  {'day78_L':>8}  {'day78_V':>8}  "
      f"{'peak_L':>8}  {'peak_V':>8}  {'match?':>8}  {'emergence_type':>15}")
print(f"  {'-'*80}")

emergence_results = {}
for ax_name in AXIS_NAMES_12:
    L_opt = DAY78_LAYERS[ax_name]
    row   = cramers_v_results[ax_name]
    V_opt = row.get(L_opt, 0)
    peak_L = max(row, key=row.get)
    peak_V = row[peak_L]

    # Determine emergence type from V profile
    early_v  = max(row.get(L,0) for L in [1,3,5])
    mid_v    = max(row.get(L,0) for L in [10,13,15])
    late_v   = max(row.get(L,0) for L in [23,25,27,28])
    if late_v > mid_v and late_v > early_v:
        etype = "late-emerging"
    elif early_v > mid_v and early_v > late_v:
        etype = "early-emerging"
    elif mid_v > early_v and mid_v > late_v:
        etype = "mid-emerging"
    else:
        etype = "plateau"

    match = "YES" if abs(peak_L - L_opt) <= 3 else "NO"
    emergence_results[ax_name] = {
        "day78_L": L_opt, "day78_V": V_opt, "peak_L": peak_L, "peak_V": peak_V,
        "match": match, "emergence_type": etype
    }
    print(f"  {ax_name:>14}  {f'L{L_opt:02d}':>8}  {V_opt:>8.3f}  "
          f"  {f'L{peak_L:02d}':>6}  {peak_V:>8.3f}  {match:>8}  {etype:>15}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("Day 120 Summary — T2 Layer Emergence Profile")
print("=" * 80)

n_match = sum(1 for r in emergence_results.values() if r["match"] == "YES")
n_total = len(emergence_results)
late_axes = [ax for ax, r in emergence_results.items() if r["emergence_type"] == "late-emerging"]
early_axes = [ax for ax, r in emergence_results.items() if r["emergence_type"] == "early-emerging"]
mid_axes = [ax for ax, r in emergence_results.items() if r["emergence_type"] == "mid-emerging"]

best_orth_L = min(gram_results, key=lambda L: gram_results[L]["offdiag_mean"])
best_orth_v = gram_results[best_orth_L]["offdiag_mean"]

print(f"""
  Day78 optimal layer matches peak Cramér's V layer: {n_match}/{n_total}

  Emergence types:
    Early (L1-5):    {len(early_axes)} axes: {', '.join(early_axes) or 'none'}
    Mid (L10-15):    {len(mid_axes)} axes: {', '.join(mid_axes) or 'none'}
    Late (L23-28):   {len(late_axes)} axes: {', '.join(late_axes) or 'none'}

  T2 Gram matrix most orthogonal at: L{best_orth_L:02d} (offdiag_mean={best_orth_v:.4f})

  VERDICT:
  {'→ T2 axes emerge at FIXED layers (phase transitions) — not gradual' if n_match > 8 else
   '→ T2 axes emerge gradually — peak layer differs from Day78 optimal' if n_match < 5 else
   '→ T2 axes show MIXED emergence: some fixed, some gradual'}

  KEY FINDING:
  {'→ LATE-EMERGING DOMINATES: semantic axes form primarily in late layers (L23-28)' if len(late_axes) > 6 else
   '→ DISTRIBUTED EMERGENCE: different axes emerge at different network depths' if len(early_axes) > 2 and len(late_axes) > 2 else
   '→ EARLY EMERGENCE: most axes form in early layers, later layers refine them'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "cramers_v_results": cramers_v_results,
        "cross_layer_results": cross_layer_results,
        "gram_results": gram_results,
        "emergence_results": emergence_results,
        "test_layers": TEST_LAYERS,
        "day78_layers": DAY78_LAYERS,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 120 complete.")
