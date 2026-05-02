#!/usr/bin/env python3
"""
Day 71 — Multi-Axis φ-Trie + Context Resolution of UNSTABLE Tokens

Part A: Multi-Axis φ-Trie
  Day 70 showed that the comparative T2 axis alone only creates 2 distinct
  paths at L5-L22 (English vs non-English). A useful φ-trie needs multiple
  axes — each axis adds one bit per layer, giving finer semantic resolution.

  We build 4 T2 axes (comparative, plural, tense, gender) × 4 layers =
  16-bit paths. Prediction: within-leaf cosine similarity increases
  monotonically as we go from 4-bit → 8-bit → 12-bit → 16-bit paths.

Part B: Context Resolution of UNSTABLE Tokens
  Day 70 found that 42/144 tokens are UNSTABLE at L27 on the comparative axis.
  These are ALL concrete nouns: lion, eagle, bear, sky, moon, elephant...
  Hypothesis: they are UNSTABLE because the model can't commit without context.

  We test: does comparative context (e.g. "The lion is much larger than...")
  resolve these tokens to HIGH, and does non-comparative context
  (e.g. "I saw a lion at the zoo") resolve them to LOW?

  If yes: the UNSTABLE zone = semantic ambiguity, resolved by attention.
  This means attention IS the φ-trie router.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day71_multi_axis_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI       # 0.618
INV_PHI2 = 1 / PHI**2    # 0.382
TRIE_LAYERS = [5, 14, 22, 27]

# ── Training pairs for 4 T2 axes ──────────────────────────────────────────────
AXIS_PAIRS = {
    "comparative": [
        ("The fast car won the race",    "The faster car won the race"),
        ("The big dog barked loudly",    "The bigger dog barked loudly"),
        ("A small bird sang at dawn",    "A smaller bird sang at dawn"),
        ("The tall tree swayed gently",  "The taller tree swayed gently"),
        ("A cold wind swept the plain",  "A colder wind swept the plain"),
        ("The old house still stands",   "The older house still stands"),
        ("A young child played outside", "A younger child played outside"),
        ("The strong man lifted it",     "The stronger man lifted it"),
    ],
    "plural": [
        ("I saw a dog in the park",       "I saw dogs in the park"),
        ("A cat sat on the mat",          "Cats sat on the mat"),
        ("The bird sang at dawn",         "The birds sang at dawn"),
        ("A child played outside",        "Children played outside"),
        ("The house stands on the hill",  "The houses stand on the hill"),
        ("A book sat on the table",       "Books sat on the table"),
        ("The star shines at night",      "The stars shine at night"),
        ("A word was spoken softly",      "Words were spoken softly"),
    ],
    "tense": [
        ("I walk to the store",       "I walked to the store"),
        ("She runs every morning",    "She ran every morning"),
        ("He eats breakfast alone",   "He ate breakfast alone"),
        ("They build a new house",    "They built a new house"),
        ("We swim in the lake",       "We swam in the lake"),
        ("She writes the letter",     "She wrote the letter"),
        ("The dog barks loudly",      "The dog barked loudly"),
        ("I read the news",           "I read the news yesterday"),
    ],
    "gender": [
        ("The king ruled the kingdom",  "The queen ruled the kingdom"),
        ("A man walked down the road",  "A woman walked down the road"),
        ("The boy played football",     "The girl played football"),
        ("His brother came home",       "His sister came home"),
        ("The father works hard",       "The mother works hard"),
        ("A son was born that day",     "A daughter was born that day"),
        ("The uncle arrived late",      "The aunt arrived late"),
        ("The prince stood tall",       "The princess stood tall"),
    ],
}

PROBE_TOKENS = [
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start",
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy",
    "the", "a", "and", "or", "but", "not", "is", "was", "has",
    "in", "on", "at", "by", "of", "to", "from", "with", "for",
    "dogs", "cats", "birds", "running", "walked", "faster", "biggest",
    "quickly", "slowly", "easily", "better", "worse", "more", "less",
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "water", "fire", "earth", "air", "sun", "moon", "star", "sky",
    "love", "hate", "truth", "false", "good", "evil", "light", "dark",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    "code", "data", "model", "graph", "tree", "node", "edge", "root",
    "math", "science", "art", "music", "word", "text", "book", "page",
    "one", "two", "three", "four", "five", "ten", "many", "few",
    "all", "some", "none", "most", "each", "every", "both", "either",
    "共", "的", "在", "了",
]

# Context templates for UNSTABLE resolution (Part B)
# {word} is replaced by the target token
CONTEXT_COMPARATIVE = "The {word} is much larger and stronger than the other one."
CONTEXT_NONCOMP     = "I saw a {word} at the park yesterday."

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
print(f"  n_layers={model.config.num_hidden_layers}  hidden={model.config.hidden_size}\n")

def get_last_hidden(text, layers):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    pos = inputs["input_ids"].shape[1] - 1
    return {l: out.hidden_states[l][0, pos, :].numpy().astype(np.float32) for l in layers}

def get_logits(word):
    inputs = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].numpy().astype(np.float32)

def build_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        i1 = tok(s1, return_tensors="pt")
        i2 = tok(s2, return_tensors="pt")
        with torch.no_grad():
            h1 = model(**i1, output_hidden_states=True).hidden_states[layer][0, -1, :].numpy()
            h2 = model(**i2, output_hidden_states=True).hidden_states[layer][0, -1, :].numpy()
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def get_word_hidden_in_context(sentence, word, layer):
    """Hidden state of `word` at `layer` when embedded in full `sentence`."""
    inputs   = tok(sentence, return_tensors="pt")
    ids      = inputs["input_ids"][0]
    word_ids = tok(" " + word.strip())["input_ids"]
    # find the last token of the word in the sentence
    target_id = word_ids[-1]
    positions = (ids == target_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return None
    pos = int(positions[0])
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

# ── Build all T2 axes ────────────────────────────────────────────────────────
print("Building T2 directions for all 4 axes × 4 layers ...")
t2_axes = {}
for axis_name, pairs in AXIS_PAIRS.items():
    t2_axes[axis_name] = {}
    for layer in TRIE_LAYERS:
        t2_axes[axis_name][layer] = build_t2(pairs, layer)
    print(f"  {axis_name}: built at layers {TRIE_LAYERS}")
print()

# ── Collect hidden states + logits for all probe tokens ─────────────────────
print(f"Computing hidden states + logits for {len(PROBE_TOKENS)} probe tokens ...")
tokens_data = {}
for word in PROBE_TOKENS:
    try:
        hs      = get_last_hidden(" " + word.strip(), TRIE_LAYERS)
        logits  = get_logits(word)
        _axes   = list(AXIS_PAIRS.keys())
        projs   = {ax: {l: float(np.dot(hs[l], t2_axes[ax][l]))
                        for l in TRIE_LAYERS}
                   for ax in _axes}
        tokens_data[word] = {"hs": hs, "logits": logits, "projs": projs}
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words = list(tokens_data.keys())
print(f"  Collected {len(words)} tokens\n")

# ── Calibrate thresholds per (axis, layer) ───────────────────────────────────
print("Calibrating thresholds per (axis, layer) ...")
thresholds = {}
for ax in AXIS_PAIRS:
    thresholds[ax] = {}
    for layer in TRIE_LAYERS:
        projs = [tokens_data[w]["projs"][ax][layer] for w in words]
        max_p = np.percentile(projs, 95)
        hi    = max_p * INV_PHI
        lo    = max_p * INV_PHI2
        thresholds[ax][layer] = (lo, hi, max_p)
        n_h = sum(1 for p in projs if p > hi)
        n_l = sum(1 for p in projs if p < lo)
        n_u = sum(1 for p in projs if lo <= p <= hi)
        print(f"  [{ax:>11}] L{layer:>2}: max={max_p:.2f}  "
              f"H={n_h:3d}  L={n_l:3d}  U={n_u:3d}")
print()

# ── Assign leaf paths: 4-bit per axis, 16-bit total ──────────────────────────
def classify(proj, ax, layer):
    lo, hi, _ = thresholds[ax][layer]
    if proj > hi:  return "H"
    if proj < lo:  return "L"
    return "U"

axis_names = list(AXIS_PAIRS.keys())

def get_path_n_axes(word, n_axes):
    """Build path using first n_axes axes, all 4 layers each."""
    bits = []
    for ax in axis_names[:n_axes]:
        for l in TRIE_LAYERS:
            bits.append(classify(tokens_data[word]["projs"][ax][l], ax, l))
    return "".join(bits)

all_pairs = [(words[i], words[j])
             for i in range(len(words))
             for j in range(i+1, len(words))]

logit_vecs = {w: tokens_data[w]["logits"] for w in words}

print("=" * 72)
print("Part A — Multi-axis φ-trie: within-leaf similarity vs number of axes")
print("=" * 72)
print()

results_by_axes = {}
for n_axes in [1, 2, 3, 4]:
    n_bits = n_axes * len(TRIE_LAYERS)   # 4, 8, 12, 16
    axes_str = "+".join(axis_names[:n_axes])

    paths = {w: get_path_n_axes(w, n_axes) for w in words}
    from collections import Counter
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

    same_m   = float(np.mean(same_sims))  if same_sims else float("nan")
    diff_m   = float(np.mean(diff_sims))  if diff_sims else float("nan")
    sep      = same_m - diff_m
    n_leaves = len(path_counts)
    n_same   = len(same_sims)
    n_diff   = len(diff_sims)
    results_by_axes[n_axes] = {
        "n_bits": n_bits, "axes": axes_str,
        "same_mean": same_m, "diff_mean": diff_m,
        "separation": sep,
        "n_leaves": n_leaves, "n_same_pairs": n_same, "n_diff_pairs": n_diff,
    }
    verdict = "CONFIRMED" if sep > 0.02 else "WEAK" if sep > 0 else "FAILED"
    print(f"  {n_axes} ax / {n_bits:>2} bits  [{axes_str}]")
    print(f"    same-leaf: {same_m:.4f}  diff-leaf: {diff_m:.4f}  "
          f"sep: {sep:+.4f}  leaves: {n_leaves}  {verdict}")
    top_leaves = [f"{p}({c})" for p, c in path_counts.most_common(6)]
    print(f"    top leaves: {', '.join(top_leaves)}")
    print()

# ── Spot check: semantic contents per leaf at 2 axes (8-bit) ─────────────────
print("=" * 72)
print("Spot check: 2-axis (8-bit) leaf contents")
print("=" * 72)
paths_2ax = {w: get_path_n_axes(w, 2) for w in words}
path_counts_2ax = Counter(paths_2ax.values())

for path, count in sorted(path_counts_2ax.items(), key=lambda x: -x[1]):
    if count < 3:
        continue
    tokens_here = [w for w in words if paths_2ax[w] == path]
    print(f"  [{path}] ({count}): {' '.join(tokens_here[:20])}")
print()

# ── Part B: Context Resolution of UNSTABLE Tokens ────────────────────────────
print("=" * 72)
print("Part B — Context resolution: do UNSTABLE tokens snap with context?")
print("=" * 72)
print()

# Find the UNSTABLE tokens on comparative axis at L27 (from Day 70)
ax_comp = "comparative"
layer_27 = 27
lo27, hi27, max27 = thresholds[ax_comp][layer_27]

unstable_words = [w for w in words
                  if classify(tokens_data[w]["projs"][ax_comp][layer_27],
                              ax_comp, layer_27) == "U"]
print(f"UNSTABLE on comparative L27: {len(unstable_words)} tokens")
print(f"  {' '.join(unstable_words[:20])}\n")
print(f"  L27 thresholds: lo={lo27:.2f}  hi={hi27:.2f}  max95={max27:.2f}\n")

print("Resolving UNSTABLE tokens with context ...")
resolution_results = {}
for word in unstable_words:
    # Baseline: standalone
    base_proj = tokens_data[word]["projs"][ax_comp][layer_27]
    base_cls  = classify(base_proj, ax_comp, layer_27)

    # Context A: comparative ("The X is much larger and stronger than the other one.")
    sent_a = CONTEXT_COMPARATIVE.format(word=word)
    h_a    = get_word_hidden_in_context(sent_a, word, layer_27)
    if h_a is not None:
        proj_a = float(np.dot(h_a, t2_axes[ax_comp][layer_27]))
        cls_a  = classify(proj_a, ax_comp, layer_27)
    else:
        proj_a, cls_a = float("nan"), "?"

    # Context B: non-comparative ("I saw a X at the park yesterday.")
    sent_b = CONTEXT_NONCOMP.format(word=word)
    h_b    = get_word_hidden_in_context(sent_b, word, layer_27)
    if h_b is not None:
        proj_b = float(np.dot(h_b, t2_axes[ax_comp][layer_27]))
        cls_b  = classify(proj_b, ax_comp, layer_27)
    else:
        proj_b, cls_b = float("nan"), "?"

    resolution_results[word] = {
        "base_proj": float(base_proj), "base_cls": base_cls,
        "comp_proj": proj_a,  "comp_cls": cls_a,
        "noncomp_proj": proj_b, "noncomp_cls": cls_b,
        "resolved_correctly":
            (cls_a == "H" or cls_a == "U") and (cls_b == "L" or cls_b == "U"),
    }
    print(f"  {word:>12}  base={base_proj:6.2f}({base_cls})"
          f"  comp={proj_a:7.2f}({cls_a})"
          f"  noncomp={proj_b:7.2f}({cls_b})")

# Summary statistics for Part B
n_comp_resolved_h = sum(1 for r in resolution_results.values() if r["comp_cls"] == "H")
n_comp_resolved_u = sum(1 for r in resolution_results.values() if r["comp_cls"] == "U")
n_comp_resolved_l = sum(1 for r in resolution_results.values() if r["comp_cls"] == "L")
n_noncomp_l = sum(1 for r in resolution_results.values() if r["noncomp_cls"] == "L")
n_noncomp_u = sum(1 for r in resolution_results.values() if r["noncomp_cls"] == "U")
n_noncomp_h = sum(1 for r in resolution_results.values() if r["noncomp_cls"] == "H")
n_total = len(resolution_results)

# Measure: did comparative context push projections UP vs baseline?
comp_deltas   = [r["comp_proj"]   - r["base_proj"] for r in resolution_results.values()
                 if not math.isnan(r["comp_proj"])]
noncomp_deltas= [r["noncomp_proj"] - r["base_proj"] for r in resolution_results.values()
                 if not math.isnan(r["noncomp_proj"])]

print()
print("=" * 72)
print("Context Resolution Summary")
print("=" * 72)
print(f"""
  UNSTABLE tokens tested: {n_total}

  After comparative context ("The X is much larger..."):
    → HIGH:     {n_comp_resolved_h}/{n_total} ({100*n_comp_resolved_h/n_total:.0f}%)
    → UNSTABLE: {n_comp_resolved_u}/{n_total} ({100*n_comp_resolved_u/n_total:.0f}%)
    → LOW:      {n_comp_resolved_l}/{n_total} ({100*n_comp_resolved_l/n_total:.0f}%)
    Mean T2 Δ vs baseline: {np.mean(comp_deltas):+.2f}

  After non-comparative context ("I saw a X yesterday"):
    → HIGH:     {n_noncomp_h}/{n_total} ({100*n_noncomp_h/n_total:.0f}%)
    → UNSTABLE: {n_noncomp_u}/{n_total} ({100*n_noncomp_u/n_total:.0f}%)
    → LOW:      {n_noncomp_l}/{n_total} ({100*n_noncomp_l/n_total:.0f}%)
    Mean T2 Δ vs baseline: {np.mean(noncomp_deltas):+.2f}

  Prediction: comp context → HIGH (push up), noncomp → LOW (push down)
  Directional: comp_Δ > 0 AND noncomp_Δ < 0 →
    {('CONFIRMED ✓' if np.mean(comp_deltas) > 0 and np.mean(noncomp_deltas) < 0 else 'NOT CONFIRMED ✗')}
""")

# ── Increasing-bits trajectory (Part A synthesis) ────────────────────────────
print("=" * 72)
print("Part A Synthesis — does within-leaf similarity increase with more bits?")
print("=" * 72)
seps = [results_by_axes[n]["separation"] for n in [1, 2, 3, 4]]
monotone = all(seps[i] >= seps[i-1] - 0.01 for i in range(1, len(seps)))
print(f"""
  4-bit  (1 axis): sep = {seps[0]:+.4f}
  8-bit  (2 axes): sep = {seps[1]:+.4f}
  12-bit (3 axes): sep = {seps[2]:+.4f}
  16-bit (4 axes): sep = {seps[3]:+.4f}

  Monotonically increasing: {monotone}

  Overall verdict:
    {'φ-TRIE MULTI-AXIS CONFIRMED ✓' if monotone and seps[-1] > seps[0] else 'PARTIAL — some axes do not add info'}
""")

# ── Save ──────────────────────────────────────────────────────────────────────
save_data = {
    "part_a": results_by_axes,
    "part_b": {
        "n_unstable": n_total,
        "n_comp_high": n_comp_resolved_h,
        "n_comp_low": n_comp_resolved_l,
        "n_noncomp_low": n_noncomp_l,
        "n_noncomp_high": n_noncomp_h,
        "mean_comp_delta": float(np.mean(comp_deltas)),
        "mean_noncomp_delta": float(np.mean(noncomp_deltas)),
        "token_results": {k: v for k, v in resolution_results.items()},
    },
    "phi_pair": {"inv_phi": INV_PHI, "inv_phi2": INV_PHI2},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 71 complete.")
