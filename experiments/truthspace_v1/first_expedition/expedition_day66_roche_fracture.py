#!/usr/bin/env python3
"""
Day 66 — Roche Limit / Orbital Fracture Model

Hypothesis: what we called "attractor tunneling" is actually orbital fracture.
Context gravity follows an inverse-square law (Day 57). A word's semantic
representation is an orbital body with internal cohesion. The T2 direction
applies a tidal force. When that force exceeds the body's cohesion — the
Roche limit — the representation fractures from the base-form orbit into
the derived-form orbit. This is NOT tunneling (gradual, monotonic) but
snapping (step-function, then overshoot/disintegration at high α).

Predictions of the fracture model vs tunnel model:
  FRACTURE: drank/dα is large at critical α (step-like), ranks climb again at
            high α (disintegration into wrong basin), post-fracture zone has
            finite width, critical α correlates with geometric cohesion
  TUNNEL:   drank/dα is roughly constant (ramp), ranks monotonically improve,
            no disintegration, critical α correlates only with baseline rank

Measurements:
  G1: Fracture curve — fine α sweep for 8 buried failures + calibration set
  G2: Fracture sharpness — peak |drank/dα| at critical transition
  G3: Post-fracture stability — width of stable zone before disintegration
  G4: Zone C specificity — which layer shows sharpest fracture?
  G5: Roche limit formula — predict α_critical from geometric properties
  G6: Orbit disruption — how much force evicts easy targets from top-5?
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day66_roche_fracture.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + math.sqrt(5)) / 2

# Ctx T2 sentence pairs from Day 65 (base_comp best axis/layer)
CTX_PAIRS_COMP = [
    ("The fast car won the race",    "The faster car won the race"),
    ("The big dog barked loudly",    "The bigger dog barked loudly"),
    ("A small bird sang at dawn",    "A smaller bird sang at dawn"),
    ("The tall tree swayed gently",  "The taller tree swayed gently"),
    ("A cold wind swept the plain",  "A colder wind swept the plain"),
    ("The old house still stands",   "The older house still stands"),
    ("A young child played outside", "A younger child played outside"),
    ("The strong man lifted it",     "The stronger man lifted it"),
]
CTX_PAIRS_SUPER = [
    ("The fast car won the race",    "The fastest car won the race"),
    ("The big dog barked loudly",    "The biggest dog barked loudly"),
    ("A small bird sang at dawn",    "The smallest bird sang at dawn"),
    ("The tall tree swayed gently",  "The tallest tree swayed gently"),
    ("A cold wind swept the plain",  "The coldest wind swept the plain"),
    ("The old house still stands",   "The oldest house still stands"),
    ("A young child played outside", "The youngest child played outside"),
    ("The strong man lifted it",     "The strongest man lifted it"),
]

# ── Test targets ──────────────────────────────────────────────────────────────
# (prompt, answer, axis, best_layer_from_day65)
BURIED = [
    ("The comparative of fast is",   "faster",  "comp", 27),
    ("The comparative of big is",    "bigger",  "comp", 14),
    ("The comparative of small is",  "smaller", "comp", 14),
    ("The superlative of big is",    "biggest", "super",20),
    ("The superlative of fast is",   "fastest", "super",27),
]
# Gender left out — different anchoring problem (Day 65)

# Easy targets — already in top-5, test orbit disruption
EASY = [
    ("The plural of cat is",         "cats",    "comp", 14),
    ("The plural of dog is",         "dogs",    "comp", 14),
    ("The past tense of walk is",    "walked",  "comp", 14),
    ("The opposite of hot is",       "cold",    "comp", 14),
    ("Up and",                       "down",    "comp", 14),
    ("Day and",                      "night",   "comp", 14),
]

# Fine alpha sweep — dense near origin, spread out above
FINE_ALPHA = [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30,
              40, 50, 60, 75, 100, 130, 170, 220]

PROBE_LAYERS = [14, 20, 22, 23, 24, 27]

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers   = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"  n_layers={n_layers}  hidden={hidden_dim}\n")

def get_hs(text, layers, is_word=False):
    if is_word:
        text = " " + text.strip()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return {L: out.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            for L in layers}

def token_rank(prompt, target_word):
    ids = tok.encode(" " + target_word, add_special_tokens=False) or \
          tok.encode(target_word, add_special_tokens=False)
    tid = ids[0]
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return int((logits > logits[tid]).sum().item())

def steer_rank(prompt, target_word, direction_np, alpha, layer):
    ids = tok.encode(" " + target_word, add_special_tokens=False) or \
          tok.encode(target_word, add_special_tokens=False)
    tid = ids[0]
    d_t = torch.tensor(direction_np, dtype=torch.float32)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out[0][0, -1, :] += alpha * d_t
            return out
        out[0, -1, :] += alpha * d_t
        return out
    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        h.remove()
    return int((logits > logits[tid]).sum().item())

def build_ctx_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        h1 = get_hs(s1, [layer])[layer]
        h2 = get_hs(s2, [layer])[layer]
        d  = h2 - h1
        n  = np.linalg.norm(d)
        if n > 1e-6:
            diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ── Build T2 directions at all probe layers ───────────────────────────────────
print("Building ctx T2 directions ...")
t2_comp  = {L: build_ctx_t2(CTX_PAIRS_COMP,  L) for L in PROBE_LAYERS}
t2_super = {L: build_ctx_t2(CTX_PAIRS_SUPER, L) for L in PROBE_LAYERS}
print("  Done.\n")

def get_t2(axis, layer):
    return t2_comp[layer] if axis == "comp" else t2_super[layer]

# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("G1/G2/G3 — Fracture curves, sharpness, post-fracture stability")
print("="*70)

all_curves   = {}
sharpness    = {}
stab_width   = {}
critical_alpha = {}
baseline_ranks = {}

for group, label in [(BURIED, "BURIED"), (EASY, "EASY")]:
    print(f"\n  [{label}]")
    for prompt, ans, axis, best_L in group:
        t2 = get_t2(axis, best_L)
        baseline = token_rank(prompt, ans)
        baseline_ranks[(prompt, ans)] = baseline

        ranks = []
        for a in FINE_ALPHA:
            r = steer_rank(prompt, ans, t2, alpha=a, layer=best_L) if a > 0 else baseline
            ranks.append(r)

        # Fracture sharpness: max |Δrank| / Δα across consecutive pairs
        max_drop = 0.0
        max_drop_alpha = None
        for i in range(len(ranks) - 1):
            da = FINE_ALPHA[i+1] - FINE_ALPHA[i]
            dr = ranks[i] - ranks[i+1]   # positive = improvement
            rate = dr / da if da > 0 else 0
            if rate > max_drop:
                max_drop = rate
                max_drop_alpha = FINE_ALPHA[i]

        # Critical alpha: first α where rank < 5
        crit = None
        for i, (a, r) in enumerate(zip(FINE_ALPHA, ranks)):
            if r < 5:
                crit = a
                break

        # Post-fracture stability: range of α where rank < 5
        in_top5 = [(a, r) for a, r in zip(FINE_ALPHA, ranks) if r < 5]
        if in_top5:
            width = in_top5[-1][0] - in_top5[0][0]
        else:
            width = 0

        all_curves[(prompt, ans)]   = list(zip(FINE_ALPHA, ranks))
        sharpness[(prompt, ans)]    = {"max_drop_rate": max_drop, "at_alpha": max_drop_alpha}
        stab_width[(prompt, ans)]   = width
        critical_alpha[(prompt, ans)] = crit

        in_top5_str = f"α=[{in_top5[0][0]}..{in_top5[-1][0]}]  width={width}" if in_top5 else "never"

        print(f"\n  '{prompt}' → {ans}  [base={baseline}]")
        # Print compact rank curve
        curve_str = "  ".join(f"{a}:{r}" for a, r in zip(FINE_ALPHA, ranks))
        print(f"    curve: {curve_str}")
        print(f"    critical_α={crit}  max_drop={max_drop:.2f}/unit  in_top5={in_top5_str}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("G4 — Zone C specificity: which layer fractures most easily?")
print("="*70)
print("  (For the 3 rescued comparatives: find α_critical at each layer)\n")

for prompt, ans, axis, _ in BURIED[:3]:   # bigger, smaller, fastest have axis=comp
    print(f"  '{prompt}' → {ans}")
    print(f"  {'Layer':<8}  {'crit_α':>8}  {'min_rank':>10}")
    print(f"  {'-'*30}")
    for L in PROBE_LAYERS:
        t2 = get_t2(axis, L)
        crit_L, min_r = None, 999
        for a in FINE_ALPHA:
            r = steer_rank(prompt, ans, t2, alpha=a, layer=L) if a > 0 else baseline_ranks[(prompt, ans)]
            if r < min_r:
                min_r = r
            if r < 5 and crit_L is None:
                crit_L = a
        print(f"  L{L:<7}  {str(crit_L):>8}  {min_r:>10}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("G5 — Roche limit formula: predict α_critical from geometry")
print("="*70)
print("  Measuring φ-space distances: ||h_derived - h_base|| at L14 and L23\n")

# Word pairs for geometric measurement
WORD_PAIRS_GEO = [
    ("fast",  "faster",   "comp",  "The comparative of fast is",  "faster"),
    ("big",   "bigger",   "comp",  "The comparative of big is",   "bigger"),
    ("small", "smaller",  "comp",  "The comparative of small is", "smaller"),
    ("big",   "biggest",  "super", "The superlative of big is",   "biggest"),
    ("fast",  "fastest",  "super", "The superlative of fast is",  "fastest"),
]

geo_data = []
print(f"  {'pair':<20} {'L14_dist':>10} {'L23_dist':>10} {'L14_cos':>10} "
      f"{'base_norm_L14':>14} {'base_rank':>10} {'crit_α':>8}")
print(f"  {'-'*80}")

for base_w, deriv_w, axis, prompt, ans in WORD_PAIRS_GEO:
    h_base_14  = get_hs(base_w,  [14, 23], is_word=True)
    h_deriv_14 = get_hs(deriv_w, [14, 23], is_word=True)

    hb14 = h_base_14[14];  hd14 = h_deriv_14[14]
    hb23 = h_base_14[23];  hd23 = h_deriv_14[23]

    dist14    = float(np.linalg.norm(hd14 - hb14))
    dist23    = float(np.linalg.norm(hd23 - hb23))
    cos14     = cosine(hb14, hd14)
    norm_b14  = float(np.linalg.norm(hb14))
    base_rank = baseline_ranks.get((prompt, ans), token_rank(prompt, ans))
    crit      = critical_alpha.get((prompt, ans))

    print(f"  {base_w}→{deriv_w:<15} {dist14:>10.2f} {dist23:>10.2f} {cos14:>10.4f} "
          f"{norm_b14:>14.2f} {base_rank:>10} {str(crit):>8}")

    geo_data.append({
        "pair": f"{base_w}→{deriv_w}",
        "axis": axis,
        "dist_L14": dist14,
        "dist_L23": dist23,
        "cos_L14": cos14,
        "base_norm_L14": norm_b14,
        "baseline_rank": base_rank,
        "critical_alpha": crit,
    })

# Correlations with critical α
crit_vals  = [d["critical_alpha"] or 999 for d in geo_data]
dist14_vals= [d["dist_L14"]       for d in geo_data]
rank_vals  = [d["baseline_rank"]  for d in geo_data]
norm_vals  = [d["base_norm_L14"]  for d in geo_data]

def pearson(x, y):
    x, y = np.array(x, float), np.array(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

r_dist = pearson(dist14_vals, crit_vals)
r_rank = pearson(rank_vals,   crit_vals)
r_norm = pearson(norm_vals,   crit_vals)
r_joint= pearson([d*r for d,r in zip(dist14_vals, rank_vals)], crit_vals)

print(f"\n  Correlations with α_critical:")
print(f"    r(L14_dist,   α_crit) = {r_dist:+.3f}")
print(f"    r(base_rank,  α_crit) = {r_rank:+.3f}")
print(f"    r(base_norm,  α_crit) = {r_norm:+.3f}")
print(f"    r(dist×rank,  α_crit) = {r_joint:+.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("G6 — Orbit disruption: how much force evicts easy targets from top-5?")
print("="*70)
print("  (Inverse Roche limit: α at which stable orbit is destroyed)\n")

for prompt, ans, axis, best_L in EASY:
    t2 = get_t2(axis, best_L)
    print(f"  '{prompt}' → {ans}  [base={baseline_ranks[(prompt,ans)]}]")
    evict_α = None
    for a in FINE_ALPHA:
        r = steer_rank(prompt, ans, t2, alpha=a, layer=best_L) if a > 0 else baseline_ranks[(prompt,ans)]
        if r >= 5 and evict_α is None and a > 0:
            evict_α = a
        mark = " ✗" if r >= 5 else ""
        print(f"    α={a:>4}  rank={r:>3}{mark}")
    print(f"    Orbit evicted at α={evict_α}\n")

# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("FRACTURE vs TUNNEL VERDICT")
print("="*70)

print("""
  FRACTURE signature:
    - Sharp transition (large max_drop_rate)
    - Finite post-fracture stability window (width > 0 then exits)
    - Rank climbs again at high α (disintegration)
    - α_critical correlates with geometric cohesion properties

  TUNNEL signature:
    - Gradual transition (small, constant drop rate)
    - Monotonically improving rank (no upper α limit)
    - α_critical correlates mainly with baseline rank
""")

print("  Targets with confirmed fracture signatures:")
for prompt, ans, axis, best_L in BURIED:
    curve = all_curves.get((prompt, ans), [])
    if not curve:
        continue
    ranks = [r for _, r in curve]
    alphas = [a for a, _ in curve]
    # Does rank climb after minimum?
    min_r = min(ranks)
    min_idx = ranks.index(min_r)
    post_min_ranks = ranks[min_idx:]
    climbs = any(post_min_ranks[i] > post_min_ranks[i-1] + 3
                 for i in range(1, len(post_min_ranks)))
    sharp = sharpness[(prompt, ans)]["max_drop_rate"]
    crit  = critical_alpha[(prompt, ans)]
    width = stab_width[(prompt, ans)]

    sig = []
    if sharp > 0.5:     sig.append(f"sharp(rate={sharp:.2f})")
    if climbs:          sig.append("disintegrates")
    if crit and width < (FINE_ALPHA[-1] - FINE_ALPHA[0]) / 3:
        sig.append(f"narrow_window(w={width})")

    verdict = "FRACTURE" if len(sig) >= 2 else ("partial" if sig else "TUNNEL")
    print(f"  {verdict:>10}  '{ans}'  crit_α={crit}  width={width}  [{', '.join(sig)}]")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "curves": {f"{p}_{a}": {"alphas": FINE_ALPHA, "ranks": [r for _, r in c]}
               for (p, a), c in all_curves.items()},
    "sharpness":      {f"{p}_{a}": v for (p, a), v in sharpness.items()},
    "stab_width":     {f"{p}_{a}": v for (p, a), v in stab_width.items()},
    "critical_alpha": {f"{p}_{a}": v for (p, a), v in critical_alpha.items()},
    "geo_data":       geo_data,
    "correlations":   {"dist_L14": r_dist, "base_rank": r_rank,
                       "base_norm": r_norm, "dist_x_rank": r_joint},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print("Day 66 complete.")
