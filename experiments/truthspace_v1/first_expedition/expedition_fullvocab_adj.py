#!/usr/bin/env python3
"""
Day 250 — Large-Scale Adjective Paradigm Mining

Goal: Mine ALL adjective comparative pairs from the full W_E vocabulary
(V=151,936 tokens), then definitively test whether cos(pos,comp) is
φ-quantized at scale (n >> 24).

Approach:
  1. Find all English base adjective tokens (single-token, alphabetic)
  2. For each, check if comparative form ("-er") is also single-token
  3. Compute cos(emb(pos), emb(comp)) for all valid pairs
  4. Compute the distribution statistics

  Also check superlative ("-est") for three-point arcs.

Questions:
  A. What is the distribution of cos(pos,comp) over ALL adj pairs?
     Is it concentrated near cos(π/(2φ)) = 0.5878?
  B. How many pairs fall within ±0.05 of the φ-cosine?
  C. Is the adj_degree paradigm genuinely φ-quantized, or just
     "nearby" due to the typical cosine range for morphological pairs?
  D. Compare to plural (-s) and past_tense (-ed) distributions.
  E. What is the chord coherence for the full-vocabulary adj set
     vs the hand-picked 24-word set?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "fullvocab_adj.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + np.sqrt(5)) / 2
PHI_COS  = np.cos(np.pi / (2 * PHI))   # cos(π/(2φ)) ≈ 0.5878
print(f"φ = {PHI:.6f}")
print(f"cos(π/(2φ)) = {PHI_COS:.6f}")
print(f"π/(2φ) = {np.degrees(np.pi/(2*PHI)):.3f}°\n")

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cos_sim(a, b): return float(np.dot(normed(a), normed(b)))

# ── Build token → word map ─────────────────────────────────────────
print("Building token→word map (finding single-space-prefix tokens)...")
token_to_word = {}  # token_id → word (strip leading space)
word_to_token = {}  # word → token_id
for tid in range(V):
    decoded = tok.decode([tid])
    # Single-token with leading space = standard word token
    if decoded.startswith(" ") and len(decoded) > 1:
        word = decoded[1:]  # strip leading space
        token_to_word[tid] = word
        word_to_token[word.lower()] = tid

print(f"  Single-space-prefix tokens: {len(token_to_word)}\n")

# ── Part A: Adjective comparative pairs (word + "er") ─────────────
print("=" * 70)
print("PART A: MINING ADJ_DEGREE PAIRS (word + 'er')")
print("        Finding all base → comparative single-token pairs")
print("=" * 70)
print()

def is_likely_adj_base(w):
    """Heuristic: alphabetic, 2-8 chars, common English adjective pattern."""
    return w.isalpha() and 2 <= len(w) <= 10

adj_pairs = []  # (base_word, comp_word, cos_val)
for tid, word in token_to_word.items():
    w = word.lower()
    if not is_likely_adj_base(w):
        continue
    # Try comparative form
    comp_variants = []
    # Standard: word + "er"
    comp_variants.append(w + "er")
    # e-ending: word + "r" (e.g., "nice" → "nicer")
    if w.endswith("e"):
        comp_variants.append(w + "r")
    # Doubling: word ends in consonant-vowel-consonant (e.g., "big" → "bigger")
    if (len(w) >= 3 and w[-1] not in "aeiou" and w[-2] in "aeiou"
            and w[-3] not in "aeiou"):
        comp_variants.append(w + w[-1] + "er")
    # y-ending: word ends in y (e.g., "happy" → "happier")
    if w.endswith("y"):
        comp_variants.append(w[:-1] + "ier")

    for comp in comp_variants:
        comp_tid = word_to_token.get(comp)
        if comp_tid is not None and comp_tid != tid:
            c = cos_sim(W_E[tid], W_E[comp_tid])
            adj_pairs.append((w, comp, c, tid, comp_tid))

print(f"  Found {len(adj_pairs)} base→comp single-token pairs\n")

# ── Part B: Distribution of cos(pos,comp) ─────────────────────────
cos_vals = np.array([c for _, _, c, _, _ in adj_pairs])

print("=" * 70)
print("PART B: DISTRIBUTION OF cos(pos, comp) FOR ALL ADJ PAIRS")
print("=" * 70)
print()

print(f"  N = {len(cos_vals)}")
print(f"  mean  = {cos_vals.mean():.4f}")
print(f"  std   = {cos_vals.std():.4f}")
print(f"  min   = {cos_vals.min():.4f}")
print(f"  max   = {cos_vals.max():.4f}")
print(f"  median= {np.median(cos_vals):.4f}")
print()
print(f"  φ-cosine = cos(π/(2φ)) = {PHI_COS:.4f}")
print(f"  |mean - φ-cosine| = {abs(cos_vals.mean() - PHI_COS):.4f}")
print()

# Histogram
bins = np.arange(-1.0, 1.05, 0.05)
hist, _ = np.histogram(cos_vals, bins=bins)
print("  Histogram of cos(pos,comp) [bin width = 0.05]:")
for i, cnt in enumerate(hist):
    lo, hi = bins[i], bins[i+1]
    if cnt > 0:
        bar = "#" * min(cnt, 50)
        marker = " <-- φ-cos" if abs((lo+hi)/2 - PHI_COS) < 0.03 else ""
        print(f"    [{lo:+.2f},{hi:+.2f}): {cnt:4d} {bar}{marker}")
print()

# ── Part C: φ-quantization test ────────────────────────────────────
print("=" * 70)
print("PART C: φ-QUANTIZATION TEST")
print("        What fraction of pairs fall within ±0.05 of φ-cosine?")
print("=" * 70)
print()

windows = [0.02, 0.05, 0.10, 0.15]
for w in windows:
    in_window = np.sum(np.abs(cos_vals - PHI_COS) <= w)
    frac = in_window / len(cos_vals)
    # Expected under uniform distribution in this region
    expected_uniform = 2 * w / 2.0  # approximate
    print(f"  |cos - φ| ≤ {w:.2f}: {in_window}/{len(cos_vals)} = {frac:.3f}  "
          f"(~{expected_uniform:.3f} expected uniform)")

print()
# Best fit: what value of cos is the mode?
hist_fine, bin_edges_fine = np.histogram(cos_vals, bins=100)
mode_idx = np.argmax(hist_fine)
mode_val = (bin_edges_fine[mode_idx] + bin_edges_fine[mode_idx+1]) / 2
print(f"  Mode of distribution: {mode_val:.4f}")
print(f"  Distance from φ-cosine ({PHI_COS:.4f}): {abs(mode_val - PHI_COS):.4f}")

# ── Part D: Compare to plural and past_tense distributions ─────────
print()
print("=" * 70)
print("PART D: COMPARE TO PLURAL AND PAST_TENSE DISTRIBUTIONS")
print("=" * 70)
print()

# Plural: word → words (add "s")
plural_pairs = []
for tid, word in token_to_word.items():
    w = word.lower()
    if not (is_likely_adj_base(w) and not w.endswith("s")):
        continue
    # word + "s"
    pl = w + "s"
    pl_tid = word_to_token.get(pl)
    if pl_tid is not None and pl_tid != tid:
        c = cos_sim(W_E[tid], W_E[pl_tid])
        plural_pairs.append((w, pl, c))
    # word + "es"
    pl2 = w + "es"
    pl2_tid = word_to_token.get(pl2)
    if pl2_tid is not None and pl2_tid != tid and pl2_tid != pl_tid:
        c = cos_sim(W_E[tid], W_E[pl2_tid])
        plural_pairs.append((w, pl2, c))

# Past tense: word → worded (add "ed")
past_pairs = []
for tid, word in token_to_word.items():
    w = word.lower()
    if not is_likely_adj_base(w):
        continue
    # word + "ed"
    pt = w + "ed"
    pt_tid = word_to_token.get(pt)
    if pt_tid is not None and pt_tid != tid:
        c = cos_sim(W_E[tid], W_E[pt_tid])
        past_pairs.append((w, pt, c))
    # e-ending: word + "d"
    if w.endswith("e"):
        pt2 = w + "d"
        pt2_tid = word_to_token.get(pt2)
        if pt2_tid is not None and pt2_tid != tid:
            c = cos_sim(W_E[tid], W_E[pt2_tid])
            past_pairs.append((w, pt2, c))

print(f"  adj_comp:   N={len(adj_pairs)}")
print(f"  plural:     N={len(plural_pairs)}")
print(f"  past_tense: N={len(past_pairs)}")
print()

for name, pairs in [("adj_comp", adj_pairs), ("plural", plural_pairs),
                    ("past_tense", past_pairs)]:
    cvs = np.array([c for _, _, c, *_ in pairs])
    if len(cvs) == 0: continue
    in_phi = np.sum(np.abs(cvs - PHI_COS) <= 0.05)
    print(f"  {name:<12}: mean={cvs.mean():.4f}  std={cvs.std():.4f}  "
          f"median={np.median(cvs):.4f}  |in_phi(±0.05)|={in_phi}/{len(cvs)}")

# ── Part E: Chord coherence for full-vocabulary adj set ─────────────
print()
print("=" * 70)
print("PART E: CHORD COHERENCE FOR FULL-VOCABULARY VS HAND-PICKED")
print("=" * 70)
print()

# Filter to adj pairs with cos in [0.45, 0.70] (main adj_degree range)
adj_core = [(w, c, tid, ctid) for w, comp, cos, tid, ctid in adj_pairs
            if 0.45 <= cos <= 0.70]
print(f"  Adj pairs with cos in [0.45, 0.70]: {len(adj_core)}")

if len(adj_core) > 10:
    # Compute chord coherence from a random sample (max 200 pairs)
    rng = np.random.default_rng(42)
    sample_size = min(200, len(adj_core))
    sample = rng.choice(len(adj_core), size=sample_size, replace=False)
    chords = []
    for idx in sample:
        w, c, tid, ctid = adj_core[idx]
        chord = W_E[ctid] - W_E[tid]
        chords.append(chord / (np.linalg.norm(chord) + 1e-8))
    chords = np.array(chords)

    # Mean pairwise cosine
    n_chord = len(chords)
    sims = chords @ chords.T
    upper_tri = sims[np.triu_indices(n_chord, k=1)]
    mean_pair_cos = float(upper_tri.mean())
    print(f"  Chord coherence (n={sample_size}): mean_pair_cos = {mean_pair_cos:.4f}")
    print(f"  Random baseline (R^{H}): 1/sqrt(H) ≈ {1/np.sqrt(H):.4f}")
    print(f"  Hand-picked 24-word set: mean_pair_cos = 0.360")

# ── Part F: Top / bottom pairs by cos_val ────────────────────────────
print()
print("=" * 70)
print("PART F: SAMPLE ADJ PAIRS — MOST AND LEAST φ-ALIGNED")
print("=" * 70)
print()

adj_sorted = sorted(adj_pairs, key=lambda x: abs(x[2] - PHI_COS))
print(f"  Top-20 pairs closest to φ-cosine ({PHI_COS:.4f}):")
for w, comp, c, _, _ in adj_sorted[:20]:
    print(f"    {w:<12} → {comp:<14}: cos={c:.4f}  Δ={c-PHI_COS:+.4f}")

print()
print(f"  Top-20 pairs FARTHEST from φ-cosine:")
for w, comp, c, _, _ in adj_sorted[-20:][::-1]:
    print(f"    {w:<12} → {comp:<14}: cos={c:.4f}  Δ={c-PHI_COS:+.4f}")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total adj pairs found: {len(adj_pairs)}")
print(f"  cos(pos,comp) mean = {cos_vals.mean():.4f}")
print(f"  cos(pos,comp) std  = {cos_vals.std():.4f}")
print(f"  cos(π/(2φ))        = {PHI_COS:.4f}")
print(f"  Distance from φ    = {abs(cos_vals.mean() - PHI_COS):.4f}")
print()
if abs(cos_vals.mean() - PHI_COS) < 0.02:
    print("  VERDICT: adj_degree IS φ-QUANTIZED (mean within 0.02 of φ-cosine)")
elif abs(cos_vals.mean() - PHI_COS) < 0.05:
    print("  VERDICT: adj_degree IS APPROXIMATELY φ-QUANTIZED (within 0.05)")
else:
    print(f"  VERDICT: adj_degree mean differs from φ-cosine by "
          f"{abs(cos_vals.mean() - PHI_COS):.3f} — NOT definitively φ-quantized")
print()
print(f"  For context: hand-picked 24 pairs gave mean cos = 0.567")
print(f"  Extended English (Day 235) gave mean cos = 0.598")

# Save
out = {
    "n_adj_pairs": len(adj_pairs),
    "cos_mean": float(cos_vals.mean()),
    "cos_std": float(cos_vals.std()),
    "cos_median": float(np.median(cos_vals)),
    "phi_cosine": float(PHI_COS),
    "distance_from_phi": float(abs(cos_vals.mean() - PHI_COS)),
    "mode": float(mode_val),
    "top20_closest": [(w, comp, float(c)) for w, comp, c, _, _ in adj_sorted[:20]],
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Full-vocabulary adjective analysis complete.")
