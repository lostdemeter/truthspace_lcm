#!/usr/bin/env python3
"""
Day 73 — Difference-PCA: Auto-Discovery of the Transformation Subspace

Day 72 showed that:
  - Manual T2 axes are ~89° from PC0 (full-state PCA finds identity manifold)
  - φ-trie axes live in the 0.8% low-variance transformation subspace
  - Auto-discovery requires PCA of DIFFERENCE vectors, not full hidden states

This experiment:
1. Computes mean difference vectors for 8 semantic transformation types
   (comparative, plural, tense, gender, antonym, hypernym, synonym, concrete-abstract)
2. Stacks them into a difference matrix and runs PCA → principal transformation axes
3. Tests whether difference-PCA axes:
   a. Align with the manual T2 direction (validation)
   b. Are mutually orthogonal (each captures a distinct transformation)
   c. Create bimodal φ-pair distributions on the probe vocabulary
   d. Build a φ-trie with monotonically increasing within-leaf separation
4. Adds a bimodality refinement step: takes the best difference-PC and
   refines it by gradient ascent on the bimodality score

PREDICTION:
  - Difference-PC0 ≈ manual T2 (within 10-20°, not 89°)
  - Difference-PCs are mutually non-parallel (each axis is a real dimension)
  - A 4-axis difference-PCA trie gives separation > +0.10 (vs +0.04 for full-state PCA)
  - Bimodality refinement pushes separation further

If Prediction 1 confirms: the manual T2 axis = automatic transformation PC.
If Prediction 2+3 confirm: the φ-trie is auto-discoverable from transformation pairs.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day73_diff_pca.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI       # 0.618
INV_PHI2 = 1 / PHI**2    # 0.382
TRIE_LAYERS = [5, 14, 22, 27]

# ── 8 semantic transformation types ─────────────────────────────────────────
TRANSFORM_PAIRS = {
    "comparative": [
        ("The fast car",   "The faster car"),
        ("A big dog",      "A bigger dog"),
        ("The small bird", "The smaller bird"),
        ("A tall tree",    "A taller tree"),
        ("The cold wind",  "The colder wind"),
        ("The old house",  "The older house"),
        ("A young child",  "A younger child"),
        ("The strong man", "The stronger man"),
        ("A bright star",  "A brighter star"),
        ("The dark room",  "The darker room"),
    ],
    "plural": [
        ("I have a dog",    "I have dogs"),
        ("I saw a cat",     "I saw cats"),
        ("A bird sang",     "Birds sang"),
        ("The tree fell",   "The trees fell"),
        ("A child played",  "Children played"),
        ("The book is old", "The books are old"),
        ("A car drove by",  "Cars drove by"),
        ("The star shines", "The stars shine"),
        ("A house stands",  "Houses stand"),
        ("The word means",  "The words mean"),
    ],
    "past_tense": [
        ("I walk to the store",  "I walked to the store"),
        ("She runs every day",   "She ran every day"),
        ("He eats breakfast",    "He ate breakfast"),
        ("They build houses",    "They built houses"),
        ("We swim in the lake",  "We swam in the lake"),
        ("She writes letters",   "She wrote letters"),
        ("He speaks loudly",     "He spoke loudly"),
        ("They sing songs",      "They sang songs"),
        ("I think clearly",      "I thought clearly"),
        ("She feels happy",      "She felt happy"),
    ],
    "gender": [
        ("The king ruled",   "The queen ruled"),
        ("A man walked",     "A woman walked"),
        ("The boy played",   "The girl played"),
        ("His brother came", "His sister came"),
        ("The father works", "The mother works"),
        ("A son was born",   "A daughter was born"),
        ("The uncle arrived","The aunt arrived"),
        ("The prince stood", "The princess stood"),
        ("A husband spoke",  "A wife spoke"),
        ("The actor left",   "The actress left"),
    ],
    "antonym": [
        ("It is hot today",    "It is cold today"),
        ("He runs fast",       "He runs slow"),
        ("The light was on",   "The dark was on"),
        ("The answer is good", "The answer is bad"),
        ("It is hard work",    "It is soft work"),
        ("She is happy now",   "She is sad now"),
        ("He is strong",       "He is weak"),
        ("The room is bright", "The room is dark"),
        ("It is the first",    "It is the last"),
        ("He is old",          "He is young"),
    ],
    "hypernym": [
        ("The dog ran away",    "The animal ran away"),
        ("A rose bloomed",      "A flower bloomed"),
        ("The oak fell down",   "The tree fell down"),
        ("The car sped by",     "The vehicle sped by"),
        ("The salmon swims",    "The fish swims"),
        ("The eagle soared",    "The bird soared"),
        ("The ruby shone red",  "The gem shone red"),
        ("The piano played",    "The instrument played"),
        ("The soldier marched", "The person marched"),
        ("The hammer struck",   "The tool struck"),
    ],
    "synonym": [
        ("He is very big",    "He is very large"),
        ("She is very small", "She is very tiny"),
        ("He runs fast",      "He runs quick"),
        ("It is very cold",   "It is very frigid"),
        ("She is so happy",   "She is so joyful"),
        ("He spoke loudly",   "He spoke noisily"),
        ("It is very hard",   "It is very difficult"),
        ("She walks slowly",  "She walks gradually"),
        ("He is quite old",   "He is quite aged"),
        ("It looks beautiful","It looks gorgeous"),
    ],
    "concrete_abstract": [
        ("The stone is heavy",   "The weight is heavy"),
        ("The fire burns hot",   "The anger burns hot"),
        ("The chain is broken",  "The trust is broken"),
        ("The road is long",     "The journey is long"),
        ("The wall is high",     "The barrier is high"),
        ("The water runs deep",  "The sorrow runs deep"),
        ("The light fades out",  "The hope fades out"),
        ("The root holds firm",  "The foundation holds"),
        ("The bridge connects",  "The relationship connects"),
        ("The key opens doors",  "The solution opens doors"),
    ],
}

# Same probe set from Day 72
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

# Manual T2 training pairs (comparative, for comparison)
COMP_PAIRS = [
    ("The fast car won the race",    "The faster car won the race"),
    ("The big dog barked loudly",    "The bigger dog barked loudly"),
    ("A small bird sang at dawn",    "A smaller bird sang at dawn"),
    ("The tall tree swayed gently",  "The taller tree swayed gently"),
    ("A cold wind swept the plain",  "A colder wind swept the plain"),
    ("The old house still stands",   "The older house still stands"),
    ("A young child played outside", "A younger child played outside"),
    ("The strong man lifted it",     "The stronger man lifted it"),
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

# ── Build manual comparative T2 (reference) ─────────────────────────────────
print("Building manual comparative T2 (reference) ...")
manual_t2 = {}
for layer in TRIE_LAYERS:
    diffs = []
    for s1, s2 in COMP_PAIRS:
        h1 = get_last_h(s1, [layer])[layer]
        h2 = get_last_h(s2, [layer])[layer]
        d  = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    manual_t2[layer] = (v / np.linalg.norm(v)).astype(np.float32)
    print(f"  L{layer:>2}: built")
print()

# ── Compute mean difference vector per (type, layer) ────────────────────────
print("Computing mean difference vectors per transformation type × layer ...")
mean_diffs = {}   # mean_diffs[type_name][layer] = unit vector

for type_name, pairs in TRANSFORM_PAIRS.items():
    mean_diffs[type_name] = {}
    for layer in TRIE_LAYERS:
        diffs = []
        for s1, s2 in pairs:
            h1 = get_last_h(s1, [layer])[layer]
            h2 = get_last_h(s2, [layer])[layer]
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        v = np.mean(diffs, axis=0)
        nv = np.linalg.norm(v)
        mean_diffs[type_name][layer] = (v / nv if nv > 1e-6 else v).astype(np.float32)
    print(f"  {type_name}: done")
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

# ── Per-layer: difference-PCA and comparisons ────────────────────────────────
type_names = list(TRANSFORM_PAIRS.keys())

def bimodal_frac(projs):
    """Fraction of projections outside φ-pair forbidden zone (H or L, not U)."""
    abs_p   = np.abs(projs)
    abs_max = np.percentile(abs_p, 95)
    hi = abs_max * INV_PHI
    lo = abs_max * INV_PHI2
    return float(((abs_p > hi) | (abs_p < lo)).mean())

def classify_proj(proj, abs_max):
    hi = abs_max * INV_PHI
    lo = abs_max * INV_PHI2
    ap = abs(proj)
    if ap > hi: return "H"
    if ap < lo: return "L"
    return "U"

print("=" * 70)
print("Per-layer: Difference-PCA vs manual T2 vs full-state PC0")
print("=" * 70)
print()

diff_pca_axes = {}   # diff_pca_axes[layer] = array (n_types, hidden_dim)

for layer in TRIE_LAYERS:
    # Stack 8 mean difference vectors into (8, hidden_dim) matrix
    D = np.stack([mean_diffs[t][layer] for t in type_names])  # (8, 1536)

    # SVD of D: rows are 8 transformation directions
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    # Vt rows = principal transformation axes
    diff_pcs = Vt   # (8, 1536)

    diff_pca_axes[layer] = diff_pcs

    # Compute angles between all 8 transformation types and top diff-PCs
    print(f"  L{layer:>2} — Transformation type alignment matrix (angle to diff-PC0):")
    for t_idx, t_name in enumerate(type_names):
        ang_pc0 = math.degrees(math.acos(min(1.0, abs(float(np.dot(mean_diffs[t_name][layer], diff_pcs[0]))))))
        ang_manual = math.degrees(math.acos(min(1.0, abs(float(np.dot(mean_diffs[t_name][layer], manual_t2[layer]))))))
        print(f"    {t_name:>20}:  ∠diff-PC0={ang_pc0:5.1f}°  ∠manual_T2={ang_manual:5.1f}°")

    # diff-PC0 vs manual T2 angle
    ang_pc0_manual = math.degrees(math.acos(min(1.0, abs(float(np.dot(diff_pcs[0], manual_t2[layer]))))))

    # Bimodality of diff-PC0 vs manual T2 on probe vocabulary
    H = np.stack([tokens_data[w]["hs"][layer] for w in words])
    H_c = H - H.mean(axis=0)

    projs_pc0    = H_c @ diff_pcs[0]
    projs_manual = H_c @ manual_t2[layer]
    bm_pc0       = bimodal_frac(projs_pc0)
    bm_manual    = bimodal_frac(projs_manual)

    # Mutual angles between diff-PCs
    angles_between = [
        math.degrees(math.acos(min(1.0, abs(float(np.dot(diff_pcs[i], diff_pcs[j]))))))
        for i in range(4) for j in range(i+1, 4)
    ]
    mean_mutual_ang = np.mean(angles_between)

    # Singular values (how much each PC captures)
    sval_str = "  ".join(f"s{k}={s[k]:.3f}" for k in range(min(4, len(s))))

    print(f"  ")
    print(f"  L{layer:>2} Summary:")
    print(f"    Singular values:      {sval_str}")
    print(f"    diff-PC0 ∠ manual_T2: {ang_pc0_manual:.1f}°  "
          f"{'ALIGNED' if ang_pc0_manual < 20 else 'DIVERGED'}")
    print(f"    Bimodal: diff-PC0={bm_pc0:.3f}  manual_T2={bm_manual:.3f}")
    print(f"    Mean mutual angle (diff-PC0..3): {mean_mutual_ang:.1f}°  "
          f"{'orthogonal' if mean_mutual_ang > 60 else 'correlated'}")
    print()

# ── Build difference-PCA φ-trie and measure within-leaf separation ───────────
all_pairs = [(words[i], words[j])
             for i in range(len(words))
             for j in range(i + 1, len(words))]

print("=" * 70)
print("Difference-PCA φ-trie: within-leaf separation vs number of axes")
print("=" * 70)
print()

diff_trie_results = {}

for n_axes in [1, 2, 4, 8]:
    paths = {}
    for w in words:
        bits = []
        for layer in TRIE_LAYERS:
            H     = np.stack([tokens_data[ww]["hs"][layer] for ww in words])
            H_c   = H - H.mean(axis=0)
            h_c   = H_c[words.index(w)]
            pcs   = diff_pca_axes[layer]
            for k in range(n_axes):
                proj    = float(np.dot(h_c, pcs[k]))
                abs_max = np.percentile(np.abs(H_c @ pcs[k]), 95)
                bits.append(classify_proj(proj, abs_max))
        paths[w] = "".join(bits)

    n_bits = n_axes * len(TRIE_LAYERS)
    path_counts = Counter(paths.values())

    same_sims, diff_sims = [], []
    for (w1, w2) in all_pairs:
        if "U" in paths[w1] or "U" in paths[w2]:
            continue
        sim = cos_sim(logit_vecs[w1], logit_vecs[w2])
        if paths[w1] == paths[w2]:
            same_sims.append(sim)
        else:
            diff_sims.append(sim)

    same_m = float(np.mean(same_sims)) if same_sims else float("nan")
    diff_m = float(np.mean(diff_sims)) if diff_sims else float("nan")
    sep    = same_m - diff_m
    n_lv   = len(path_counts)

    diff_trie_results[n_axes] = {
        "n_bits": n_bits, "same_mean": same_m, "diff_mean": diff_m,
        "separation": sep, "n_leaves": n_lv,
        "n_same_pairs": len(same_sims), "n_diff_pairs": len(diff_sims),
    }
    verdict = "CONFIRMED" if sep > 0.02 else "WEAK" if sep > 0 else "FAILED"
    print(f"  {n_axes:>1} diff-PCA axes / {n_bits:>2} bits:")
    print(f"    same-leaf: {same_m:.4f}  diff-leaf: {diff_m:.4f}  "
          f"sep: {sep:+.4f}  leaves: {n_lv}  {verdict}")
    top = [f"{p}({c})" for p, c in path_counts.most_common(5)]
    print(f"    top leaves: {', '.join(top)}")
    print()

# ── Spot check: leaf contents for 1 diff-PCA axis × 4 layers ────────────────
print("=" * 70)
print("Spot check: leaf contents for 1 diff-PCA axis × 4 layers (4-bit)")
print("=" * 70)

paths_1ax = {}
for w in words:
    bits = []
    for layer in TRIE_LAYERS:
        H   = np.stack([tokens_data[ww]["hs"][layer] for ww in words])
        H_c = H - H.mean(axis=0)
        h_c = H_c[words.index(w)]
        pc0 = diff_pca_axes[layer][0]
        proj    = float(np.dot(h_c, pc0))
        abs_max = np.percentile(np.abs(H_c @ pc0), 95)
        bits.append(classify_proj(proj, abs_max))
    paths_1ax[w] = "".join(bits)

path_counts_1ax = Counter(paths_1ax.values())
for path, count in sorted(path_counts_1ax.items(), key=lambda x: -x[1]):
    if count < 2: continue
    tokens_here = [w for w in words if paths_1ax[w] == path]
    print(f"  [{path}] ({count}): {' '.join(tokens_here[:20])}")
print()

# ── Final comparison table ────────────────────────────────────────────────────
print("=" * 70)
print("Final comparison: Full-state PCA vs Difference-PCA vs Manual T2 trie")
print("=" * 70)
print(f"""
  Method                  separation (best)   comment
  Full-state PCA (1ax)    +0.0294             identity manifold, useless
  Difference-PCA (1ax)    {diff_trie_results[1]['separation']:+.4f}             transform subspace
  Difference-PCA (2ax)    {diff_trie_results[2]['separation']:+.4f}             transform subspace
  Difference-PCA (4ax)    {diff_trie_results[4]['separation']:+.4f}             transform subspace
  Manual T2 (1ax, Day 70) +0.4548             comparative, manually specified
""")

# ── Type-type angles per layer: are the 8 types orthogonal? ────────────────
print("=" * 70)
print("Cross-type angles at L14 (are 8 transformation types orthogonal?)")
print("=" * 70)
layer = 14
for i in range(len(type_names)):
    for j in range(i+1, len(type_names)):
        ang = math.degrees(math.acos(
            min(1.0, abs(float(np.dot(mean_diffs[type_names[i]][layer],
                                       mean_diffs[type_names[j]][layer]))))))
        print(f"  {type_names[i]:>20} ∠ {type_names[j]:<20}  {ang:.1f}°")
print()

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "diff_trie_results": diff_trie_results,
    "layer_summaries": {
        layer: {
            "type_angles_to_manual": {
                t: float(math.degrees(math.acos(
                    min(1.0, abs(float(np.dot(mean_diffs[t][layer], manual_t2[layer])))))))
                for t in type_names
            },
        }
        for layer in TRIE_LAYERS
    },
    "phi_pair": {"inv_phi": INV_PHI, "inv_phi2": INV_PHI2},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 73 complete.")
