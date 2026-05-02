#!/usr/bin/env python3
"""
Day 72 — T2 Auto-Discovery: PCA of Hidden State Space

Day 71 showed that all manually-specified T2 axes (comparative, plural,
tense, gender) are PARALLEL within the English cluster at L5-L22. They
all point the same direction (English/non-English separator) and only the
comparative axis shows within-English variation at L27.

This experiment asks: what are the NATURAL axes of the hidden state space?
If we run PCA on the English-cluster hidden states at each layer, the
principal components ARE the φ-trie's natural axes.

PREDICTIONS:
1. The top PCA components at each layer show bimodal projections with
   φ-pair forbidden zones (proving PCA re-discovers the φ-trie structure)
2. PCA axes at L14 and L22 are NOT parallel to each other, unlike manual
   T2 axes (each layer has its own natural axes)
3. A trie built from top-4 PCA axes × 4 layers gives monotonically
   increasing within-leaf separation as we add more PCA axes
4. The manual T2 axes ALIGN with the top PCA component (they were
   accidentally finding the model's principal axis all along)

If predictions 1 & 3 confirm: the φ-trie is PCA-addressable —
no training pairs needed, just decompose the hidden state matrix.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day72_auto_axes.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI       # 0.618
INV_PHI2 = 1 / PHI**2    # 0.382
TRIE_LAYERS = [5, 14, 22, 27]
N_PCA_AXES  = 8           # test up to 8 PCA axes

# Manual T2 axes for comparison (comparative only, already validated)
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

# Larger probe: 200 English tokens + 8 non-English
PROBE_TOKENS = [
    # Animals (animate concrete)
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
    # Plants / objects (inanimate concrete)
    "tree", "flower", "rock", "stone", "wood", "leaf", "grass", "root",
    "house", "door", "window", "table", "chair", "book", "cup", "key",
    "car", "road", "bridge", "boat", "ship", "plane", "train", "bike",
    # Actions (verbs)
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start", "stop",
    "think", "know", "see", "hear", "feel", "love", "hate", "want",
    "give", "take", "make", "find", "use", "keep", "work", "move",
    # Qualities (adjectives)
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
    "good", "bad", "right", "wrong", "real", "false", "long", "short",
    "high", "low", "deep", "wide", "thin", "thick", "clean", "dirty",
    # Function words
    "the", "a", "and", "or", "but", "not", "is", "was", "has",
    "in", "on", "at", "by", "of", "to", "from", "with", "for",
    "he", "she", "it", "we", "they", "this", "that", "which", "who",
    # Morphological forms
    "dogs", "cats", "trees", "faster", "biggest", "running", "walked",
    "quickly", "slowly", "better", "worse", "more", "less", "most",
    # Abstract nouns
    "love", "hate", "truth", "beauty", "justice", "freedom", "power",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    "fear", "joy", "pain", "peace", "war", "law", "art", "science",
    # Numbers / quantifiers
    "one", "two", "three", "four", "five", "ten", "many", "few",
    "all", "some", "none", "most", "each", "every", "both", "either",
    # Non-English (should form separate cluster)
    "共", "的", "在", "了", "le", "la", "der", "und",
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

def get_hidden_states(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {l: out.hidden_states[l][0, pos, :].numpy().astype(np.float32) for l in layers}

def get_logits(word):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :].numpy().astype(np.float32)

def build_manual_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        i1, i2 = tok(s1, return_tensors="pt"), tok(s2, return_tensors="pt")
        with torch.no_grad():
            h1 = model(**i1, output_hidden_states=True).hidden_states[layer][0, -1].numpy()
            h2 = model(**i2, output_hidden_states=True).hidden_states[layer][0, -1].numpy()
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

# ── Collect hidden states + logits ──────────────────────────────────────────
print(f"Computing hidden states for {len(PROBE_TOKENS)} tokens ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        hs     = get_hidden_states(" " + word.strip(), TRIE_LAYERS)
        logits = get_logits(word)
        tokens_data[word] = {"hs": hs, "logits": logits}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words    = list(tokens_data.keys())
logit_vecs = {w: tokens_data[w]["logits"] for w in words}
print(f"  Collected {len(words)} tokens\n")

# ── Build manual T2 at each layer (for comparison) ──────────────────────────
print("Building manual comparative T2 directions ...")
manual_t2 = {}
for layer in TRIE_LAYERS:
    manual_t2[layer] = build_manual_t2(COMP_PAIRS, layer)
    print(f"  L{layer:>2}: built")
print()

# ── PCA decomposition per layer ─────────────────────────────────────────────
print("Running PCA on hidden states per layer ...")
pca_results = {}
all_results  = {}

for layer in TRIE_LAYERS:
    H = np.stack([tokens_data[w]["hs"][layer] for w in words])  # (N, d)
    H_c = H - H.mean(axis=0)  # center

    # SVD (economy): U @ diag(s) @ Vt, rows of Vt are principal components
    U, s, Vt = np.linalg.svd(H_c, full_matrices=False)
    pcs = Vt[:N_PCA_AXES]  # (N_PCA_AXES, d) — top PCA directions

    # Variance explained
    var_explained = s**2 / (s**2).sum()

    # For each PC, project all tokens and check bimodality
    pc_analysis = []
    for k in range(N_PCA_AXES):
        projs = H_c @ pcs[k]  # (N,) projections
        max_p = np.percentile(projs, 95)
        min_p = np.percentile(projs, 5)

        # Use absolute max for threshold calibration
        abs_max = max(abs(max_p), abs(min_p))
        hi = abs_max * INV_PHI
        lo = abs_max * INV_PHI2

        # Count in zones (using absolute projection)
        abs_projs = np.abs(projs)
        n_hi = int((abs_projs > hi).sum())
        n_lo = int((abs_projs < lo).sum())
        n_us = int(((abs_projs >= lo) & (abs_projs <= hi)).sum())

        # Bimodality score: fraction outside the forbidden zone
        bimodal_frac = (n_hi + n_lo) / len(words)

        # Angle with manual T2 at this layer
        angle_with_manual = math.degrees(math.acos(
            min(1.0, abs(float(np.dot(pcs[k], manual_t2[layer]))))
        ))

        pc_analysis.append({
            "k": k,
            "var_explained": float(var_explained[k]),
            "max_proj": float(max_p),
            "min_proj": float(min_p),
            "n_high": n_hi,
            "n_low": n_lo,
            "n_unstable": n_us,
            "bimodal_frac": float(bimodal_frac),
            "angle_with_manual_t2": float(angle_with_manual),
        })

    # Also measure manual T2 bimodality at this layer for comparison
    manual_projs = H_c @ manual_t2[layer]
    mt_max = np.percentile(manual_projs, 95)
    mt_hi  = mt_max * INV_PHI
    mt_lo  = mt_max * INV_PHI2
    mt_bimodal = float(((manual_projs > mt_hi) | (manual_projs < mt_lo)).mean())

    pca_results[layer] = {
        "variance_explained": [float(v) for v in var_explained[:N_PCA_AXES]],
        "cumulative_var": float(var_explained[:N_PCA_AXES].sum()),
        "pc_analysis": pc_analysis,
        "manual_t2_bimodal_frac": mt_bimodal,
        "pcs": pcs.tolist(),
    }

    print(f"  L{layer:>2}:")
    print(f"    Manual T2 bimodal fraction: {mt_bimodal:.3f}")
    print(f"    Top-8 PCs variance explained: "
          f"{', '.join(f'{v:.3f}' for v in var_explained[:8])}")
    print(f"    {'PC':>3}  {'var_exp':>8}  {'H':>5}  {'L':>5}  {'U':>5}  "
          f"{'bimodal':>8}  {'∠manual':>8}")
    for pa in pc_analysis:
        print(f"    PC{pa['k']:>1}  {pa['var_explained']:>8.4f}  "
              f"{pa['n_high']:>5}  {pa['n_low']:>5}  {pa['n_unstable']:>5}  "
              f"{pa['bimodal_frac']:>8.3f}  {pa['angle_with_manual_t2']:>7.1f}°")
    print()

# ── Cross-layer PC alignment ─────────────────────────────────────────────────
print("=" * 70)
print("Cross-layer PC alignment (are PCs at L14 parallel to PCs at L22?)")
print("=" * 70)
print()
for (la, lb) in [(5, 14), (14, 22), (22, 27), (5, 27)]:
    pcs_a = np.array(pca_results[la]["pcs"])
    pcs_b = np.array(pca_results[lb]["pcs"])
    angles = []
    for k in range(min(4, N_PCA_AXES)):
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(pcs_a[k], pcs_b[k]))))))
        angles.append(ang)
    angle_str = "  ".join(f"PC{k}:{a:.1f}°" for k, a in enumerate(angles))
    print(f"  L{la:>2}↔L{lb:>2}:  {angle_str}")
print()

# ── Build PCA-trie leaf paths and measure within-leaf similarity ─────────────
print("=" * 70)
print("PCA-trie: within-leaf similarity vs number of PCA axes")
print("=" * 70)
print()

def pca_classify(proj, abs_max):
    hi = abs_max * INV_PHI
    lo = abs_max * INV_PHI2
    ap = abs(proj)
    if ap > hi: return "H"
    if ap < lo: return "L"
    return "U"

all_pairs = [(words[i], words[j])
             for i in range(len(words))
             for j in range(i + 1, len(words))]

# Pre-compute per-layer centered matrices and projections
layer_H_c = {}
for layer in TRIE_LAYERS:
    H = np.stack([tokens_data[w]["hs"][layer] for w in words])
    layer_H_c[layer] = H - H.mean(axis=0)

pca_trie_results = {}
for n_axes in [1, 2, 4, 8]:
    # Use PC0..n_axes-1 at EACH of the 4 layers
    paths = {}
    for w in words:
        bits = []
        for layer in TRIE_LAYERS:
            pcs = np.array(pca_results[layer]["pcs"])
            h_c = layer_H_c[layer][words.index(w)]
            for k in range(n_axes):
                proj = float(np.dot(h_c, pcs[k]))
                # calibrate: use abs_max from pc_analysis
                pa   = pca_results[layer]["pc_analysis"][k]
                abs_max = max(abs(pa["max_proj"]), abs(pa["min_proj"]))
                bits.append(pca_classify(proj, abs_max))
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

    pca_trie_results[n_axes] = {
        "n_bits": n_bits,
        "same_mean": same_m,
        "diff_mean": diff_m,
        "separation": sep,
        "n_leaves": n_lv,
        "n_same_pairs": len(same_sims),
        "n_diff_pairs": len(diff_sims),
    }

    verdict = "CONFIRMED" if sep > 0.02 else "WEAK" if sep > 0 else "FAILED"
    print(f"  {n_axes:>1} PCA axes / {n_bits:>2} bits:")
    print(f"    same-leaf: {same_m:.4f}  diff-leaf: {diff_m:.4f}  "
          f"sep: {sep:+.4f}  leaves: {n_lv}  {verdict}")
    top_lv = [f"{p}({c})" for p, c in path_counts.most_common(5)]
    print(f"    top leaves: {', '.join(top_lv)}")
    print()

# ── Spot check: 2-layer PCA (L14+L22 top-2 PCs each = 4-bit) ──────────────
print("=" * 70)
print("Spot check: leaf contents for 1 PCA axis × 4 layers (4-bit)")
print("=" * 70)

n_ax = 1
paths_1pc = {}
for w in words:
    bits = []
    for layer in TRIE_LAYERS:
        pcs = np.array(pca_results[layer]["pcs"])
        h_c = layer_H_c[layer][words.index(w)]
        proj = float(np.dot(h_c, pcs[0]))
        pa   = pca_results[layer]["pc_analysis"][0]
        abs_max = max(abs(pa["max_proj"]), abs(pa["min_proj"]))
        bits.append(pca_classify(proj, abs_max))
    paths_1pc[w] = "".join(bits)

path_counts_1pc = Counter(paths_1pc.values())
for path, count in sorted(path_counts_1pc.items(), key=lambda x: -x[1]):
    if count < 2:
        continue
    tokens_here = [w for w in words if paths_1pc[w] == path]
    print(f"  [{path}] ({count}): {' '.join(tokens_here[:20])}")
print()

# ── Manual T2 vs PCA PC0: alignment at each layer ────────────────────────────
print("=" * 70)
print("Manual comparative T2 vs PCA PC0: angle at each layer")
print("=" * 70)
for layer in TRIE_LAYERS:
    pcs = np.array(pca_results[layer]["pcs"])
    ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(pcs[0], manual_t2[layer]))))))
    print(f"  L{layer:>2}: angle(PC0, manual_T2) = {ang:.2f}°"
          f"  {'ALIGNED (<10°)' if ang < 10 else 'DIVERGED'}")
print()

# ── Synthesis ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("SYNTHESIS")
print("=" * 70)

# Best bimodal PCs at L14 and L22
for layer in [14, 22]:
    best_k = max(pca_results[layer]["pc_analysis"],
                 key=lambda pa: pa["bimodal_frac"])
    print(f"  Best bimodal PC at L{layer}: "
          f"PC{best_k['k']} (bimodal={best_k['bimodal_frac']:.3f}, "
          f"var_exp={best_k['var_explained']:.4f})")

# Cross-layer PC0 divergence
pcs14 = np.array(pca_results[14]["pcs"])
pcs22 = np.array(pca_results[22]["pcs"])
ang_14_22 = math.degrees(math.acos(min(1.0, abs(float(np.dot(pcs14[0], pcs22[0]))))))

# Multi-axis monotone
seps = [pca_trie_results[n]["separation"] for n in [1, 2, 4, 8]]
monotone = all(seps[i] >= seps[i-1] - 0.005 for i in range(1, len(seps)))

print(f"""
  PC0 angle L14 ↔ L22:  {ang_14_22:.1f}°
  {'PCs diverge across layers → unique axes per layer' if ang_14_22 > 10 else 'PCs parallel across layers → same axis'}

  PCA trie separation trajectory:
    1 ax (4-bit):  {seps[0]:+.4f}
    2 ax (8-bit):  {seps[1]:+.4f}
    4 ax (16-bit): {seps[2]:+.4f}
    8 ax (32-bit): {seps[3]:+.4f}
  Monotonically increasing: {monotone}

  Manual T2 vs PC0: already printed above.
  If aligned (<10°): manual T2 = PC0 = model's principal semantic axis.
  If diverged:       manual T2 captures a different direction than PC0.
""")

# ── Save ──────────────────────────────────────────────────────────────────────
save_data = {
    "pca_trie_results": pca_trie_results,
    "layer_pca_summary": {
        layer: {
            "variance_explained": pca_results[layer]["variance_explained"],
            "cumulative_var": pca_results[layer]["cumulative_var"],
            "pc_analysis": pca_results[layer]["pc_analysis"],
            "manual_t2_bimodal_frac": pca_results[layer]["manual_t2_bimodal_frac"],
        }
        for layer in TRIE_LAYERS
    },
    "phi_pair": {"inv_phi": INV_PHI, "inv_phi2": INV_PHI2},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 72 complete.")
