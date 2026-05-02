#!/usr/bin/env python3
"""
Day 69 — Bloch Sphere / φ-Zipf Unification Hypothesis

Hypothesis: The Bloch sphere organisation we observed (English = positive
T2 hemisphere, non-English = equator/negative) is not specific to the
comparative axis — it is what φ-Zipf distributed vocabulary data ALWAYS
does when organised into attractor basins on a compact curved manifold.
The structures we've catalogued (Resonator, Gyroscope, Content Separator,
Completeness Gate) are all Bloch sphere operations at different scales.

Tests:
  E1 — Cross-axis universality:
       Build T2 directions for plural, tense, and gender axes.
       Measure T2 projections across vocabulary for each axis.
       Prediction: ALL axes show English-cluster = positive hemisphere.
       The macro structure is axis-independent. Micro structure differs.

  E2 — Layer nesting:
       Measure comparative T2 projections at L5, L14, L22, L27.
       Prediction: variance within English cluster increases with depth.
       The deeper the layer, the more the Bloch sphere is "zoomed in"
       on the within-English morphological distinctions.

  E3 — Zipf structure of T2 projections:
       For 2000 vocabulary tokens, compute T2 projections.
       Sort and fit to Zipf / power-law.
       Prediction: the projection distribution follows φ-Zipf decay.

  E4 — φ-pair boundaries:
       From DC 247: φ^(+0) = 1/φ = 0.618, φ^(-0) = 1/φ² = 0.382.
       The gate-pair sums to 1: 1/φ + 1/φ² = 1 EXACTLY.
       Do T2 projections within the English cluster show special
       boundaries at max_proj × 1/φ and max_proj × 1/φ²?
       Prediction: tokens near these boundaries are the "equator
       residents" — structural words at the edge of the cluster.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day69_bloch_unification.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + math.sqrt(5)) / 2   # ≈ 1.618
INV_PHI  = 1 / PHI              # ≈ 0.618
INV_PHI2 = 1 / PHI**2           # ≈ 0.382

# ── Training pairs for four T2 axes ──────────────────────────────────────────
AXES = {
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
        ("I saw one cat in the garden",  "I saw two cats in the garden"),
        ("There was a dog on the path",  "There were two dogs on the path"),
        ("A bird landed on the branch",  "Two birds landed on the branch"),
        ("The child ran to the door",    "The children ran to the door"),
        ("A car stopped at the light",   "Two cars stopped at the light"),
        ("The man waved at the crowd",   "The men waved at the crowd"),
        ("A tree fell in the storm",     "Two trees fell in the storm"),
        ("The book was on the shelf",    "The books were on the shelf"),
    ],
    "tense": [
        ("Every day he walks to work",   "Yesterday he walked to work"),
        ("She talks to him each morning","She talked to him that morning"),
        ("The dog jumps over the fence", "The dog jumped over the fence"),
        ("He plays the piano at noon",   "He played the piano at noon"),
        ("The train arrives at six",     "The train arrived at six"),
        ("She runs along the river",     "She ran along the river"),
        ("They sing every Sunday",       "They sang that Sunday"),
        ("The cat climbs the tree",      "The cat climbed the tree"),
    ],
    "gender": [
        ("The king ruled wisely",        "The queen ruled wisely"),
        ("My uncle told a story",        "My aunt told a story"),
        ("The actor took the stage",     "The actress took the stage"),
        ("He smiled at the crowd",       "She smiled at the crowd"),
        ("The boy ran across the field", "The girl ran across the field"),
        ("His voice echoed in the hall", "Her voice echoed in the hall"),
        ("The man opened the door",      "The woman opened the door"),
        ("The prince rode away",         "The princess rode away"),
    ],
}

# English vocabulary probe — diverse sample
ENGLISH_PROBE = [
    "dogs", "cats", "walk", "run", "fast", "slow", "big", "small",
    "quickly", "slowly", "beautiful", "ugly", "happy", "sad",
    "the", "a", "and", "or", "but", "not", "is", "was", "has",
    "king", "queen", "man", "woman", "boy", "girl",
    "positive", "negative", "zero", "one", "two", "three",
    "than", "more", "less", "most", "least", "very", "quite",
    "red", "blue", "green", "hot", "cold", "old", "new",
    "tech", "science", "art", "music", "water", "fire", "earth",
]

# Non-English / structural probe
NONENG_PROBE = [
    "共", "的", "是", "在", "了",  # Chinese
    "le", "la", "les", "et", "ou",  # French
    "der", "die", "das", "und",     # German
]

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers   = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
vocab_size = model.config.vocab_size
print(f"  n_layers={n_layers}  hidden={hidden_dim}  vocab={vocab_size}\n")

def get_hs_last(text, layer):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].numpy().astype(np.float32)

def get_hs_word(word, layer):
    return get_hs_last(" " + word.strip(), layer)

def build_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        h1 = get_hs_last(s1, layer)
        h2 = get_hs_last(s2, layer)
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6:
            diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

def t2_proj(h, t2):
    return float(np.dot(h, t2))

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("E1 — Cross-axis universality: do all T2 axes share the same macro structure?")
print("=" * 70)
print("  Building T2 directions for all four axes at L14.\n")

t2_vectors = {}
for axis_name, pairs in AXES.items():
    t2_vectors[axis_name] = build_t2(pairs, 14)
    print(f"  Built {axis_name} T2  ||v||={np.linalg.norm(t2_vectors[axis_name]):.4f}")

print()

# Cross-axis angles
print("  Cross-axis angles (how different are the four T2 directions?):")
axis_names = list(AXES.keys())
for i in range(len(axis_names)):
    for j in range(i+1, len(axis_names)):
        v1 = t2_vectors[axis_names[i]]
        v2 = t2_vectors[axis_names[j]]
        cos_ab = float(np.dot(v1, v2))
        angle  = math.degrees(math.acos(min(1.0, abs(cos_ab))))
        print(f"    {axis_names[i]:>12} ⟂ {axis_names[j]:<12}: cos={cos_ab:+.4f}  angle={angle:.1f}°")

print()

# Measure projections for English and non-English tokens on each axis
e1_results = {}

print(f"  {'token':>15}  " + "  ".join(f"{a[:6]:>8}" for a in axis_names))
print(f"  {'-'*60}")

all_eng_projs  = {a: [] for a in axis_names}
all_neng_projs = {a: [] for a in axis_names}

for word in ENGLISH_PROBE[:20]:
    try:
        h = get_hs_word(word, 14)
        projs = {a: t2_proj(h, t2_vectors[a]) for a in axis_names}
        for a in axis_names:
            all_eng_projs[a].append(projs[a])
        row = "  ".join(f"{projs[a]:>8.1f}" for a in axis_names)
        print(f"  {word:>15}  {row}")
    except Exception as e:
        print(f"  {word:>15}  ERROR: {e}")

print(f"  {'-'*60}")
print(f"  {'[non-English]':>15}")
for word in NONENG_PROBE[:8]:
    try:
        h = get_hs_word(word, 14)
        projs = {a: t2_proj(h, t2_vectors[a]) for a in axis_names}
        for a in axis_names:
            all_neng_projs[a].append(projs[a])
        row = "  ".join(f"{projs[a]:>8.2f}" for a in axis_names)
        sign = "+" if all(projs[a] > 0 for a in axis_names) else "-/mix"
        print(f"  {word:>15}  {row}  ({sign})")
    except Exception as e:
        print(f"  {word:>15}  ERROR: {e}")

print()
print(f"  Summary — mean projection by axis:")
for a in axis_names:
    eng_mean  = np.mean(all_eng_projs[a])  if all_eng_projs[a]  else float('nan')
    neng_mean = np.mean(all_neng_projs[a]) if all_neng_projs[a] else float('nan')
    all_pos   = all(p > 0 for p in all_eng_projs[a])
    print(f"    {a:>12}: English mean={eng_mean:+.2f}  non-English mean={neng_mean:+.2f}"
          f"  all_English_positive={all_pos}")

e1_results = {
    "english_projs": {a: all_eng_projs[a] for a in axis_names},
    "noneng_projs":  {a: all_neng_projs[a] for a in axis_names},
}

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E2 — Layer nesting: does within-English variance increase with depth?")
print("=" * 70)
print("  Comparative T2 at L5, L14, L22, L27 — measuring std(proj) across English.\n")

PROBE_LAYERS = [5, 14, 22, 27]
e2_results = {}

for layer in PROBE_LAYERS:
    t2_L = build_t2(AXES["comparative"], layer)
    projs = []
    for word in ENGLISH_PROBE:
        try:
            h = get_hs_word(word, layer)
            projs.append((word, t2_proj(h, t2_L)))
        except Exception:
            pass

    projs_vals = [p for _, p in projs]
    mean_p = np.mean(projs_vals)
    std_p  = np.std(projs_vals)
    min_p  = min(projs_vals)
    max_p  = max(projs_vals)
    range_p = max_p - min_p

    # φ-pair boundaries
    phi_high = max_p * INV_PHI   # 1/φ × max
    phi_low  = max_p * INV_PHI2  # 1/φ² × max

    # How many tokens fall below the φ boundaries?
    below_phi_high = sum(1 for p in projs_vals if p < phi_high)
    below_phi_low  = sum(1 for p in projs_vals if p < phi_low)

    print(f"  L{layer:>2}: mean={mean_p:+.2f}  std={std_p:.3f}  "
          f"range=[{min_p:+.2f}, {max_p:+.2f}]  width={range_p:.3f}")
    print(f"        φ-pair boundaries: 1/φ×max={phi_high:.2f}  1/φ²×max={phi_low:.2f}")
    print(f"        tokens below 1/φ×max: {below_phi_high}/{len(projs_vals)}")
    print(f"        tokens below 1/φ²×max: {below_phi_low}/{len(projs_vals)}")

    # Print the 5 tokens with lowest and highest projection
    projs_sorted = sorted(projs, key=lambda x: x[1])
    low5  = "  ".join(f"[{w}:{p:.1f}]" for w, p in projs_sorted[:5])
    high5 = "  ".join(f"[{w}:{p:.1f}]" for w, p in projs_sorted[-5:])
    print(f"        lowest:  {low5}")
    print(f"        highest: {high5}")
    print()

    e2_results[f"L{layer}"] = {
        "mean": mean_p, "std": std_p, "min": min_p, "max": max_p,
        "range": range_p, "phi_high_boundary": phi_high, "phi_low_boundary": phi_low,
        "below_phi_high": below_phi_high, "below_phi_low": below_phi_low,
        "token_projs": dict(projs),
    }

print("  Prediction: std should INCREASE from L5 to L27 (deeper = more discriminating).")
stds = [e2_results[f"L{l}"]["std"] for l in PROBE_LAYERS]
monotone = all(stds[i] <= stds[i+1] for i in range(len(stds)-1))
print(f"  std sequence: {' < '.join(f'{s:.3f}' for s in stds)}")
print(f"  Monotone increasing: {monotone}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E3 — Zipf structure of T2 projections across vocabulary")
print("=" * 70)
print("  Sampling 2000 tokens, computing T2 projections, checking Zipf fit.\n")

t2_comp_L14 = t2_vectors["comparative"]

# Sample 2000 tokens by decoding token IDs
np.random.seed(42)
sample_ids = np.random.choice(vocab_size, size=3000, replace=False)
token_projs = []
for tid in sample_ids:
    tok_str = tok.decode([int(tid)]).strip()
    if not tok_str or len(tok_str) < 1:
        continue
    try:
        h = get_hs_word(tok_str, 14)
        p = t2_proj(h, t2_comp_L14)
        token_projs.append((tok_str, p, int(tid)))
        if len(token_projs) >= 2000:
            break
    except Exception:
        pass

print(f"  Collected {len(token_projs)} token T2 projections.")

projs_sorted_desc = sorted(token_projs, key=lambda x: -x[1])
projs_vals_all = [p for _, p, _ in projs_sorted_desc]

# Zipf fit: log(proj) vs log(rank) should be linear
pos_projs = [(i+1, p) for i, (_, p, _) in enumerate(projs_sorted_desc) if p > 0]
if len(pos_projs) > 10:
    log_ranks = np.log([r for r, _ in pos_projs])
    log_projs = np.log([p for _, p in pos_projs])
    # Linear fit
    coeffs = np.polyfit(log_ranks, log_projs, 1)
    zipf_exp = -coeffs[0]
    r2 = np.corrcoef(log_ranks, log_projs)[0,1]**2
    print(f"  Positive-projection Zipf fit:")
    print(f"    Zipf exponent α = {zipf_exp:.4f}  (Zipf's law: α≈1.0)")
    print(f"    R² of log-log fit: {r2:.4f}")
    print(f"    (α≈1 = exact Zipf, α<1 = sub-Zipf, α>1 = super-Zipf)")
    phi_zipf = abs(zipf_exp - 1.0) < 0.3
    print(f"    Near-Zipf (|α-1|<0.3): {phi_zipf}")
else:
    r2, zipf_exp = 0, 0
    print("  Insufficient positive projections for Zipf fit.")

# Distribution statistics
n_positive = sum(1 for p in projs_vals_all if p > 0)
n_negative = sum(1 for p in projs_vals_all if p < 0)
n_near_zero = sum(1 for p in projs_vals_all if abs(p) < 1.0)
print(f"\n  Distribution breakdown:")
print(f"    T2-positive tokens:  {n_positive}/{len(projs_vals_all)} ({100*n_positive/len(projs_vals_all):.1f}%)")
print(f"    T2-negative tokens:  {n_negative}/{len(projs_vals_all)} ({100*n_negative/len(projs_vals_all):.1f}%)")
print(f"    Near-zero (|p|<1):   {n_near_zero}/{len(projs_vals_all)} ({100*n_near_zero/len(projs_vals_all):.1f}%)")
print(f"    Max: {projs_vals_all[0]:.2f}  Min: {projs_vals_all[-1]:.2f}")

# Top and bottom tokens
print(f"\n  Most T2-positive (top 10): ")
for tok_str, p, _ in projs_sorted_desc[:10]:
    print(f"    {tok_str:>20}  {p:+.2f}")
print(f"\n  Most T2-negative (bottom 10): ")
for tok_str, p, _ in projs_sorted_desc[-10:]:
    print(f"    {tok_str:>20}  {p:+.2f}")

e3_results = {
    "n_tokens": len(token_projs),
    "n_positive": n_positive, "n_negative": n_negative, "n_near_zero": n_near_zero,
    "zipf_exponent": zipf_exp, "log_log_r2": r2,
    "max_proj": projs_vals_all[0] if projs_vals_all else None,
    "min_proj": projs_vals_all[-1] if projs_vals_all else None,
}

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E4 — φ-pair boundaries in the T2 projection distribution")
print("=" * 70)
print(f"  DC 247: φ^(+0) = 1/φ = {INV_PHI:.4f}  φ^(-0) = 1/φ² = {INV_PHI2:.4f}")
print(f"  1/φ + 1/φ² = {INV_PHI + INV_PHI2:.6f} (should be 1.0 exactly)")
print(f"  Question: do T2 projections show special structure at these fractions of max?\n")

max_proj = projs_vals_all[0] if projs_vals_all else 30.0
phi_threshold_high = max_proj * INV_PHI   # = max × 0.618
phi_threshold_low  = max_proj * INV_PHI2  # = max × 0.382
phi_midpoint       = max_proj * 0.5       # standard midpoint

print(f"  Max projection: {max_proj:.2f}")
print(f"  Standard midpoint (0.5 × max): {phi_midpoint:.2f}")
print(f"  φ-pair high (1/φ × max = 0.618 × max): {phi_threshold_high:.2f}")
print(f"  φ-pair low  (1/φ² × max = 0.382 × max): {phi_threshold_low:.2f}\n")

# Count tokens in each zone
zone_above_phi_high = [(t, p) for t, p, _ in projs_sorted_desc if p >= phi_threshold_high]
zone_between        = [(t, p) for t, p, _ in projs_sorted_desc if phi_threshold_low <= p < phi_threshold_high]
zone_below_phi_low  = [(t, p) for t, p, _ in projs_sorted_desc if 0 < p < phi_threshold_low]
zone_negative       = [(t, p) for t, p, _ in projs_sorted_desc if p < 0]

print(f"  Token distribution across φ-pair zones:")
print(f"    Above 1/φ×max  (>{phi_threshold_high:.1f}):  {len(zone_above_phi_high):5d} tokens"
      f"  ({100*len(zone_above_phi_high)/len(projs_sorted_desc):.1f}%)")
print(f"    1/φ²×max to 1/φ×max ({phi_threshold_low:.1f} to {phi_threshold_high:.1f}):  "
      f"{len(zone_between):5d} tokens  ({100*len(zone_between)/len(projs_sorted_desc):.1f}%)")
print(f"    0 to 1/φ²×max  (0 to {phi_threshold_low:.1f}):  {len(zone_below_phi_low):5d} tokens"
      f"  ({100*len(zone_below_phi_low)/len(projs_sorted_desc):.1f}%)")
print(f"    Negative         (<0):  {len(zone_negative):5d} tokens"
      f"  ({100*len(zone_negative)/len(projs_sorted_desc):.1f}%)")

# What kind of tokens live near the φ-pair boundaries?
print(f"\n  Tokens near 1/φ×max boundary (±2 units around {phi_threshold_high:.1f}):")
near_phi_high = [(t, p) for t, p, _ in projs_sorted_desc
                 if abs(p - phi_threshold_high) < 2.0]
for t, p in near_phi_high[:15]:
    print(f"    {t:>20}  {p:+.2f}")

print(f"\n  Tokens near 1/φ²×max boundary (±2 units around {phi_threshold_low:.1f}):")
near_phi_low = [(t, p) for t, p, _ in projs_sorted_desc
                if abs(p - phi_threshold_low) < 2.0]
for t, p in near_phi_low[:15]:
    print(f"    {t:>20}  {p:+.2f}")

# Is there a density gap at the φ boundaries? (histogram check)
print(f"\n  Histogram of T2 projections (density check for φ-pair gaps):")
bins = np.linspace(min(0, min(projs_vals_all)), max_proj, 30)
hist, edges = np.histogram(projs_vals_all, bins=bins)
max_hist = max(hist)
for i, (count, left) in enumerate(zip(hist, edges[:-1])):
    bar_len = int(25 * count / max_hist)
    bar = "█" * bar_len + "░" * (25 - bar_len)
    # Mark φ-pair boundaries
    mark = ""
    if abs(left - phi_threshold_high) < (edges[1] - edges[0]):
        mark = f"  ← 1/φ×max ({phi_threshold_high:.1f})"
    if abs(left - phi_threshold_low) < (edges[1] - edges[0]):
        mark = f"  ← 1/φ²×max ({phi_threshold_low:.1f})"
    if abs(left) < (edges[1] - edges[0]):
        mark = "  ← zero"
    print(f"  {left:>6.1f}  [{bar}] {count:4d}{mark}")

e4_results = {
    "max_proj": max_proj,
    "phi_threshold_high": phi_threshold_high,
    "phi_threshold_low":  phi_threshold_low,
    "n_above_phi_high":   len(zone_above_phi_high),
    "n_between":          len(zone_between),
    "n_below_phi_low":    len(zone_below_phi_low),
    "n_negative":         len(zone_negative),
    "total":              len(projs_sorted_desc),
}

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SYNTHESIS — Bloch Sphere / φ-Zipf Unification")
print("=" * 70)

print("""
  PREDICTION 1 (E1): All four T2 axes show English = T2-positive
  PREDICTION 2 (E2): Within-English variance increases with layer depth
  PREDICTION 3 (E3): T2 projections follow Zipf distribution (α ≈ 1)
  PREDICTION 4 (E4): φ-pair boundaries (1/φ, 1/φ²) appear as special zones
""")

# P1 verdict
all_pos_by_axis = {
    a: all(p > 0 for p in all_eng_projs[a])
    for a in axis_names
}
p1_confirmed = all(all_pos_by_axis.values())
print(f"  P1 (cross-axis universality): {'CONFIRMED' if p1_confirmed else 'PARTIAL/FAILED'}")
for a, v in all_pos_by_axis.items():
    print(f"    {a}: all_English_positive = {v}")

# P2 verdict
p2_confirmed = monotone
print(f"\n  P2 (layer nesting): {'CONFIRMED' if p2_confirmed else 'PARTIAL/FAILED'}")
for l, s in zip(PROBE_LAYERS, stds):
    print(f"    L{l}: std = {s:.4f}")

# P3 verdict
p3_confirmed = r2 > 0.8 and phi_zipf
print(f"\n  P3 (Zipf structure): {'CONFIRMED' if p3_confirmed else 'PARTIAL/FAILED'}")
print(f"    Zipf exponent α = {zipf_exp:.4f}  R² = {r2:.4f}")

# P4 verdict
frac_above = len(zone_above_phi_high) / len(projs_sorted_desc)
# φ-pair prediction: 1/φ² fraction above high boundary (≈ 0.382 if distribution is φ-Zipf)
p4_verdict = f"φ fraction above 1/φ×max: {frac_above:.3f} (φ-Zipf predicts {INV_PHI2:.3f} = 1/φ²)"
print(f"\n  P4 (φ-pair boundaries):")
print(f"    {p4_verdict}")

print()
print("=" * 70)

# Save
results = {
    "e1": e1_results,
    "e2": e2_results,
    "e3": e3_results,
    "e4": e4_results,
    "verdicts": {
        "P1_cross_axis": p1_confirmed,
        "P2_layer_nesting": p2_confirmed,
        "P3_zipf": p3_confirmed,
        "P4_phi_pair": e4_results,
    }
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 69 complete.")
