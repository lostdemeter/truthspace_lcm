#!/usr/bin/env python3
"""
Day 70 — φ-Trie Leaf Path Verification

Hypothesis: transformer inference = φ-trie traversal.

A φ-trie assigns each token a binary leaf path: at each layer, the token's
T2 projection is classified HIGH (> 1/φ × max) or LOW (< 1/φ² × max).
The path is the sequence of HIGH/LOW bits across a selected set of layers.

PREDICTION: tokens with the SAME leaf path should have nearly identical
output logit distributions (their "meaning" is at the same trie leaf).
Tokens with different leaf paths should have more dissimilar logit distributions.

Specifically:
  same_path_cosine_sim >> diff_path_cosine_sim

We also check:
  - Does each additional bit increase within-path similarity? (layerwise test)
  - What semantic categories land in the same leaf? (spot check)
  - Does leaf path predict similarity better than raw embedding cosine? (baseline)
  - Are UNSTABLE tokens (in the forbidden zone) actually rare? (consistency check)

Measurements:
  - T2 projections at L5, L14, L22, L27 (comparative axis)
  - φ-pair thresholds calibrated dynamically per layer from probe set
  - Output logit vector = lm_head(h_L27) for each token as single-token prompt
  - Cosine similarity between logit vectors
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day70_phi_trie.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI     = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI      # 0.618
INV_PHI2 = 1 / PHI**2   # 0.382

TRIE_LAYERS = [5, 14, 22, 27]      # layers used to build the leaf path

# Training pairs for comparative T2 direction
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

# Diverse vocabulary probe — 150 tokens covering multiple semantic categories
PROBE_TOKENS = [
    # Animals
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    # Actions
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start",
    # Qualities
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy",
    # Function words
    "the", "a", "and", "or", "but", "not", "is", "was", "has",
    "in", "on", "at", "by", "of", "to", "from", "with", "for",
    # Morphological forms
    "dogs", "cats", "birds", "running", "walked", "faster", "biggest",
    "quickly", "slowly", "easily", "better", "worse", "more", "less",
    # Semantic content
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "water", "fire", "earth", "air", "sun", "moon", "star", "sky",
    # Abstract
    "love", "hate", "truth", "false", "good", "evil", "light", "dark",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    # Technical
    "code", "data", "model", "graph", "tree", "node", "edge", "root",
    "math", "science", "art", "music", "word", "text", "book", "page",
    # Quantifiers / numbers
    "one", "two", "three", "four", "five", "ten", "many", "few",
    "all", "some", "none", "most", "each", "every", "both", "either",
    # Non-English (expect different trie paths)
    "共", "的", "在", "了",
]

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers   = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
vocab_size = model.config.vocab_size
print(f"  n_layers={n_layers}  hidden={hidden_dim}  vocab={vocab_size}\n")

def get_hidden_states(text, layers):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states  # tuple of (n_layers+1) tensors
    # last token position
    pos = inputs["input_ids"].shape[1] - 1
    return {l: hs[l][0, pos, :].numpy().astype(np.float32) for l in layers}

def get_logits_single_token(word):
    """Forward pass for a single token, return output logit vector."""
    inputs = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    # logits at last position
    return out.logits[0, -1, :].numpy().astype(np.float32)

def build_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        inputs1 = tok(s1, return_tensors="pt")
        inputs2 = tok(s2, return_tensors="pt")
        with torch.no_grad():
            h1 = model(**inputs1, output_hidden_states=True).hidden_states[layer][0, -1, :].numpy()
            h2 = model(**inputs2, output_hidden_states=True).hidden_states[layer][0, -1, :].numpy()
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6:
            diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

# ── Build T2 directions ───────────────────────────────────────────────────────
print("Building comparative T2 directions ...")
t2 = {}
for layer in TRIE_LAYERS:
    t2[layer] = build_t2(COMP_PAIRS, layer)
    print(f"  T2 at L{layer:>2}: built")
print()

# ── Collect hidden states and logits for all probe tokens ────────────────────
print(f"Computing hidden states + logits for {len(PROBE_TOKENS)} probe tokens ...")
tokens_data = {}

for word in PROBE_TOKENS:
    try:
        hs = get_hidden_states(" " + word.strip(), TRIE_LAYERS + [27])
        logits = get_logits_single_token(word)
        t2_projs = {l: float(np.dot(hs[l], t2[l])) for l in TRIE_LAYERS}
        tokens_data[word] = {
            "hs":    {l: hs[l] for l in TRIE_LAYERS},
            "logits": logits,
            "t2_projs": t2_projs,
        }
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

words = list(tokens_data.keys())
print(f"  Collected {len(words)} tokens\n")

# ── Calibrate φ-pair thresholds per layer ────────────────────────────────────
print("Calibrating φ-pair thresholds per layer ...")
thresholds = {}
for layer in TRIE_LAYERS:
    projs = [tokens_data[w]["t2_projs"][layer] for w in words]
    max_p = np.percentile(projs, 95)   # robust max
    hi    = max_p * INV_PHI            # 0.618 × max
    lo    = max_p * INV_PHI2           # 0.382 × max
    thresholds[layer] = (lo, hi, max_p)
    n_high = sum(1 for p in projs if p > hi)
    n_low  = sum(1 for p in projs if p < lo)
    n_uns  = sum(1 for p in projs if lo <= p <= hi)
    print(f"  L{layer:>2}: max95={max_p:.2f}  lo={lo:.2f}  hi={hi:.2f}"
          f"  HIGH={n_high}  LOW={n_low}  UNSTABLE={n_uns}")
print()

# ── Assign leaf paths ─────────────────────────────────────────────────────────
def classify_layer(proj, layer):
    lo, hi, _ = thresholds[layer]
    if proj > hi:  return "H"
    if proj < lo:  return "L"
    return "U"   # unstable (forbidden zone)

leaf_paths = {}
for w in words:
    path = "".join(classify_layer(tokens_data[w]["t2_projs"][l], l) for l in TRIE_LAYERS)
    leaf_paths[w] = path

# Print leaf path distribution
from collections import Counter
path_counts = Counter(leaf_paths.values())
print("Leaf path distribution (4-bit, H=HIGH, L=LOW, U=UNSTABLE):")
for path, count in sorted(path_counts.items(), key=lambda x: -x[1]):
    tokens_in_path = [w for w in words if leaf_paths[w] == path]
    sample = " ".join(tokens_in_path[:8])
    print(f"  {path}: {count:3d} tokens   e.g.: {sample}")
print()

n_unstable = sum(1 for p in leaf_paths.values() if "U" in p)
print(f"  Tokens with at least one UNSTABLE bit: {n_unstable}/{len(words)} "
      f"({100*n_unstable/len(words):.1f}%) — prediction: rare")
print()

# ── Compute pairwise similarities ─────────────────────────────────────────────
print("Computing pairwise cosine similarities ...")

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

# Get logit vectors and raw embeddings
logit_vecs = {w: tokens_data[w]["logits"] for w in words}
embed_matrix = model.get_input_embeddings().weight.detach().numpy()

embed_vecs = {}
for w in words:
    ids = tok(" " + w.strip(), return_tensors="pt")["input_ids"][0]
    last_id = int(ids[-1])
    embed_vecs[w] = embed_matrix[last_id]

# Pairwise analysis: same-path vs diff-path, across 1/2/3/4 bits
all_pairs = [(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]

results_by_nbits = {}
for n_bits in [1, 2, 3, 4]:
    layers_used = TRIE_LAYERS[:n_bits]
    paths_n = {}
    for w in words:
        path = "".join(classify_layer(tokens_data[w]["t2_projs"][l], l) for l in layers_used)
        paths_n[w] = path

    same_logit_sims = []
    diff_logit_sims = []
    same_embed_sims = []
    diff_embed_sims = []
    same_pairs_list = []
    diff_pairs_list = []

    for (w1, w2) in all_pairs:
        # Skip if either has a U in this n-bit path
        if "U" in paths_n[w1] or "U" in paths_n[w2]:
            continue
        sim_logit = cos_sim(logit_vecs[w1], logit_vecs[w2])
        sim_embed = cos_sim(embed_vecs[w1], embed_vecs[w2])
        if paths_n[w1] == paths_n[w2]:
            same_logit_sims.append(sim_logit)
            same_embed_sims.append(sim_embed)
            same_pairs_list.append((w1, w2, sim_logit))
        else:
            diff_logit_sims.append(sim_logit)
            diff_embed_sims.append(sim_embed)
            diff_pairs_list.append((w1, w2, sim_logit))

    same_mean  = np.mean(same_logit_sims)  if same_logit_sims  else float("nan")
    diff_mean  = np.mean(diff_logit_sims)  if diff_logit_sims  else float("nan")
    separation = same_mean - diff_mean

    same_emb_mean = np.mean(same_embed_sims)  if same_embed_sims  else float("nan")
    diff_emb_mean = np.mean(diff_embed_sims)  if diff_embed_sims  else float("nan")
    emb_separation = same_emb_mean - diff_emb_mean

    results_by_nbits[n_bits] = {
        "layers": layers_used,
        "same_logit_mean": same_mean,
        "diff_logit_mean": diff_mean,
        "separation": separation,
        "same_embed_mean": same_emb_mean,
        "diff_embed_mean": diff_emb_mean,
        "embed_separation": emb_separation,
        "n_same_pairs": len(same_logit_sims),
        "n_diff_pairs": len(diff_logit_sims),
    }

    layers_str = "+".join(f"L{l}" for l in layers_used)
    print(f"  {n_bits}-bit path ({layers_str}):")
    print(f"    Logit cosine:  same-path = {same_mean:.4f}  diff-path = {diff_mean:.4f}"
          f"  separation = {separation:+.4f}  "
          f"({'CONFIRMED' if separation > 0.02 else 'WEAK' if separation > 0 else 'FAILED'})")
    print(f"    Embed cosine:  same-path = {same_emb_mean:.4f}  diff-path = {diff_emb_mean:.4f}"
          f"  separation = {emb_separation:+.4f}  (baseline)")
    print(f"    Pairs: same={len(same_logit_sims)}  diff={len(diff_logit_sims)}")
    print()

# ── Spot check: what tokens share the same 4-bit leaf path? ──────────────────
print("=" * 70)
print("Spot check: semantic content of each 4-bit leaf")
print("=" * 70)

for path, count in sorted(path_counts.items(), key=lambda x: -x[1]):
    if count < 2:
        continue
    tokens_here = [w for w in words if leaf_paths[w] == path]
    # Sort by cosine similarity to centroid
    logit_matrix = np.stack([logit_vecs[w] for w in tokens_here])
    centroid = logit_matrix.mean(axis=0)
    sims_to_centroid = [cos_sim(logit_vecs[w], centroid) for w in tokens_here]
    ranked = sorted(zip(tokens_here, sims_to_centroid), key=lambda x: -x[1])
    token_str = "  ".join(f"{w}({s:.2f})" for w, s in ranked[:15])
    print(f"  [{path}] {count} tokens: {token_str}")
print()

# ── Key result: top same-path pairs with HIGH similarity ─────────────────────
print("=" * 70)
print("Top 20 same-path pairs by logit cosine similarity (4-bit path)")
print("=" * 70)

all_same_pairs = [(w1, w2, cos_sim(logit_vecs[w1], logit_vecs[w2]))
                  for (w1, w2) in all_pairs
                  if leaf_paths[w1] == leaf_paths[w2]
                  and "U" not in leaf_paths[w1]]
all_same_pairs.sort(key=lambda x: -x[2])
for w1, w2, s in all_same_pairs[:20]:
    path = leaf_paths[w1]
    print(f"  [{path}]  {w1:>12} — {w2:<12}  cos={s:.4f}")

print()
print("=" * 70)
print("Bottom 10 same-path pairs (lowest similarity despite same path)")
print("=" * 70)
for w1, w2, s in all_same_pairs[-10:]:
    path = leaf_paths[w1]
    print(f"  [{path}]  {w1:>12} — {w2:<12}  cos={s:.4f}")

# ── Layerwise bit contribution ─────────────────────────────────────────────────
print()
print("=" * 70)
print("Layerwise: does each additional bit increase within-path similarity?")
print("=" * 70)

print(f"  {'bits':>6}  {'layers':>20}  {'same_cos':>9}  {'diff_cos':>9}  {'separation':>11}  verdict")
prev_sep = -1.0
for n_bits in [1, 2, 3, 4]:
    r = results_by_nbits[n_bits]
    layers_str = "+".join(f"L{l}" for l in r["layers"])
    sep = r["separation"]
    increasing = "↑" if sep > prev_sep else "↓" if sep < prev_sep else "="
    verdict = "CONFIRMED" if sep > 0.02 else "WEAK" if sep > 0 else "FAILED"
    print(f"  {n_bits:>6}  {layers_str:>20}  "
          f"{r['same_logit_mean']:>9.4f}  {r['diff_logit_mean']:>9.4f}  "
          f"{sep:>+11.4f}  {increasing} {verdict}")
    prev_sep = sep

# ── φ-trie vs embedding baseline comparison ──────────────────────────────────
print()
print("=" * 70)
print("φ-trie leaf path vs raw embedding: which predicts similarity better?")
print("=" * 70)

# For 4-bit path:
r4 = results_by_nbits[4]
trie_uplift = r4["separation"]
emb_uplift  = r4["embed_separation"]
trie_wins   = trie_uplift > emb_uplift
print(f"  φ-trie (4-bit) logit separation:   {trie_uplift:+.4f}")
print(f"  Raw embedding logit separation:    {emb_uplift:+.4f}")
print(f"  φ-trie adds information beyond raw embedding: {trie_wins}")
print(f"  (If φ-trie > embedding: the trie captures structure the raw embedding misses)")

# ── Final synthesis ────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SYNTHESIS — φ-Trie Hypothesis Verdict")
print("=" * 70)

r4 = results_by_nbits[4]
confirmed = r4["separation"] > 0.02
increasing_bits = all(
    results_by_nbits[n]["separation"] >= results_by_nbits[n-1]["separation"] - 0.01
    for n in [2, 3, 4]
)
unstable_rare = n_unstable / len(words) < 0.15

print(f"""
  Prediction 1: same-path tokens have higher logit similarity
    same_path cos = {r4['same_logit_mean']:.4f}
    diff_path cos = {r4['diff_logit_mean']:.4f}
    separation    = {r4['separation']:+.4f}
    Result: {'CONFIRMED ✓' if confirmed else 'NOT CONFIRMED ✗'}

  Prediction 2: more bits → more separation (each layer adds information)
    1-bit: {results_by_nbits[1]['separation']:+.4f}
    2-bit: {results_by_nbits[2]['separation']:+.4f}
    3-bit: {results_by_nbits[3]['separation']:+.4f}
    4-bit: {results_by_nbits[4]['separation']:+.4f}
    Result: {'CONFIRMED ✓' if increasing_bits else 'PARTIAL'}

  Prediction 3: UNSTABLE tokens are rare (forbidden zone is real)
    Tokens with UNSTABLE bit: {n_unstable}/{len(words)} ({100*n_unstable/len(words):.1f}%)
    Result: {'CONFIRMED ✓' if unstable_rare else 'NOT CONFIRMED ✗'}

  Overall: {'φ-TRIE IS REAL' if confirmed and unstable_rare else 'HYPOTHESIS NOT CONFIRMED'}
""")

# Save
save_data = {
    "results_by_nbits": {
        k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
            for kk, vv in v.items() if kk != "layers"}
        for k, v in results_by_nbits.items()
    },
    "leaf_paths": leaf_paths,
    "path_counts": dict(path_counts),
    "n_unstable": n_unstable,
    "n_tokens": len(words),
    "phi_pair": {"inv_phi": INV_PHI, "inv_phi2": INV_PHI2},
    "thresholds": {l: {"lo": float(lo), "hi": float(hi), "max95": float(mx)}
                   for l, (lo, hi, mx) in thresholds.items()},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 70 complete.")
