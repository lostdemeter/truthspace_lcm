#!/usr/bin/env python3
"""
Day 65 — Morphological T2 Tunneling

Day 64 identified 8 buried morphological failures that magnitude perturbation
cannot rescue (ranks 8–45, getting worse with higher α). They fall into three
transformation categories:

  base_comparative : fast→faster, big→bigger, small→smaller
  base_superlative : fast→fastest, big→biggest
  gender_morph     : queen→king, man→woman, woman→man

Hypothesis: the correct T2 direction for each morphological axis will act as
an attractor tunnel — a directed geometric displacement that moves the hidden
state from the base-form basin into the morphological-form basin.

Approach:
  T1: Build isolated T2 (word pairs, direct diff at each layer)
  T2: Build contextual T2 (sentence pairs, same approach as Day 62)
  T3: Layer sweep — which layer responds best to each morphological T2?
  T4: Alpha sweep — what magnitude minimises steered rank for each failure?
  T5: Cross-axis test — does base_comparative T2 help superlative failures?
  T6: Best-config matrix — report the single best setting per failure
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day65_morpho_t2.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PROBE_LAYERS = [14, 20, 22, 23, 24, 27]
ALPHA_SWEEP  = [5, 10, 15, 20, 30, 50, 75, 100]

# ── The 8 buried failures from Day 64 ────────────────────────────────────────
FAILURES = [
    # (prompt, answer, axis)
    ("The comparative of fast is",   "faster",  "base_comp"),
    ("The comparative of big is",    "bigger",  "base_comp"),
    ("The comparative of small is",  "smaller", "base_comp"),
    ("The superlative of big is",    "biggest", "base_super"),
    ("The superlative of fast is",   "fastest", "base_super"),
    ("The male version of queen is", "king",    "gender"),
    ("The female version of man is", "woman",   "gender"),
    ("The male version of woman is", "man",     "gender"),
]

# ── Word pairs for ISOLATED T2 construction ──────────────────────────────────
ISO_PAIRS = {
    "base_comp": [
        ("fast", "faster"), ("big", "bigger"), ("small", "smaller"),
        ("tall", "taller"), ("cold", "colder"), ("old", "older"),
        ("young", "younger"), ("strong", "stronger"), ("bright", "brighter"),
        ("dark", "darker"), ("loud", "louder"), ("quick", "quicker"),
    ],
    "base_super": [
        ("fast", "fastest"), ("big", "biggest"), ("small", "smallest"),
        ("tall", "tallest"), ("cold", "coldest"), ("old", "oldest"),
        ("young", "youngest"), ("strong", "strongest"), ("bright", "brightest"),
        ("dark", "darkest"), ("loud", "loudest"), ("quick", "quickest"),
    ],
    "gender": [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
        ("husband", "wife"), ("uncle", "aunt"), ("grandfather", "grandmother"),
        ("prince", "princess"), ("duke", "duchess"), ("lord", "lady"),
    ],
}

# ── Sentence pairs for CONTEXTUAL T2 construction ────────────────────────────
CTX_PAIRS = {
    "base_comp": [
        ("The fast car won the race",    "The faster car won the race"),
        ("The big dog barked loudly",    "The bigger dog barked loudly"),
        ("A small bird sang at dawn",    "A smaller bird sang at dawn"),
        ("The tall tree swayed gently",  "The taller tree swayed gently"),
        ("A cold wind swept the plain",  "A colder wind swept the plain"),
        ("The old house still stands",   "The older house still stands"),
        ("A young child played outside", "A younger child played outside"),
        ("The strong man lifted it",     "The stronger man lifted it"),
    ],
    "base_super": [
        ("The fast car won the race",    "The fastest car won the race"),
        ("The big dog barked loudly",    "The biggest dog barked loudly"),
        ("A small bird sang at dawn",    "The smallest bird sang at dawn"),
        ("The tall tree swayed gently",  "The tallest tree swayed gently"),
        ("A cold wind swept the plain",  "The coldest wind swept the plain"),
        ("The old house still stands",   "The oldest house still stands"),
        ("A young child played outside", "The youngest child played outside"),
        ("The strong man lifted it",     "The strongest man lifted it"),
    ],
    "gender": [
        ("The king ruled the kingdom",   "The queen ruled the kingdom"),
        ("A man walked down the street", "A woman walked down the street"),
        ("The boy ran in the park",      "The girl ran in the park"),
        ("A father taught his child",    "A mother taught her child"),
        ("The son greeted the guests",   "The daughter greeted the guests"),
        ("The brother arrived first",    "The sister arrived first"),
        ("The husband cooked dinner",    "The wife cooked dinner"),
        ("The prince rode away",         "The princess rode away"),
    ],
}

# ── Model load ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers   = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"  n_layers={n_layers}  hidden={hidden_dim}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def get_hs(text, layers, is_word=False):
    """Extract last-token hidden states at requested layers."""
    if is_word:
        text = " " + text.strip()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return {L: out.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            for L in layers}

def steer_rank(prompt, target_word, direction_np, alpha, layer):
    target_ids = tok.encode(" " + target_word, add_special_tokens=False)
    if not target_ids:
        target_ids = tok.encode(target_word, add_special_tokens=False)
    tid = target_ids[0]
    d_t = torch.tensor(direction_np, dtype=torch.float32)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            out[0][0, -1, :] += alpha * d_t
            return out
        out[0, -1, :] += alpha * d_t
        return out

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        handle.remove()
    return int((logits > logits[tid]).sum().item())

def token_rank(prompt, target_word):
    target_ids = tok.encode(" " + target_word, add_special_tokens=False)
    if not target_ids:
        target_ids = tok.encode(target_word, add_special_tokens=False)
    tid = target_ids[0]
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return int((logits > logits[tid]).sum().item())

def build_iso_t2(axis, layer):
    """Isolated T2: mean of normalised (w2_hs - w1_hs) for word pairs."""
    diffs = []
    for w1, w2 in ISO_PAIRS[axis]:
        h1 = get_hs(w1, [layer], is_word=True)[layer]
        h2 = get_hs(w2, [layer], is_word=True)[layer]
        d  = h2 - h1
        n  = np.linalg.norm(d)
        if n > 1e-6:
            diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)

def build_ctx_t2(axis, layer):
    """Contextual T2: mean of normalised (s2_hs - s1_hs) for sentence pairs."""
    diffs = []
    for s1, s2 in CTX_PAIRS[axis]:
        h1 = get_hs(s1, [layer])[layer]
        h2 = get_hs(s2, [layer])[layer]
        d  = h2 - h1
        n  = np.linalg.norm(d)
        if n > 1e-6:
            diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)

# ══════════════════════════════════════════════════════════════════════════════
print("Building all T2 directions (isolated + contextual) at probe layers ...")
AXES = ["base_comp", "base_super", "gender"]
t2_iso = {}   # (axis, layer) -> direction
t2_ctx = {}   # (axis, layer) -> direction
for axis in AXES:
    for L in PROBE_LAYERS:
        t2_iso[(axis, L)] = build_iso_t2(axis, L)
        t2_ctx[(axis, L)] = build_ctx_t2(axis, L)
        print(f"  {axis} L{L}  cos(iso,ctx)={cosine(t2_iso[(axis,L)], t2_ctx[(axis,L)]):+.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Baseline ranks for the 8 failures")
print("="*70)
baselines = {}
for prompt, ans, axis in FAILURES:
    r = token_rank(prompt, ans)
    baselines[(prompt, ans)] = r
    print(f"  rank {r:>4}  '{prompt}' → {ans}  [{axis}]")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T1/T2 — Layer sweep at α=20 (correct axis, iso vs ctx)")
print("="*70)
layer_sweep_results = {}

for prompt, ans, axis in FAILURES:
    print(f"\n  '{prompt}' → {ans}  (axis={axis})")
    print(f"  {'Layer':<8} {'iso_rank':>10} {'ctx_rank':>10}  baseline={baselines[(prompt,ans)]}")
    print(f"  {'-'*35}")
    best_iso = (999, None, None)
    best_ctx = (999, None, None)
    for L in PROBE_LAYERS:
        r_iso = steer_rank(prompt, ans, t2_iso[(axis, L)], alpha=20, layer=L)
        r_ctx = steer_rank(prompt, ans, t2_ctx[(axis, L)], alpha=20, layer=L)
        mark_i = " ★" if r_iso < 5 else ""
        mark_c = " ★" if r_ctx < 5 else ""
        print(f"  L{L:<7} {r_iso:>10}{mark_i}  {r_ctx:>10}{mark_c}")
        if r_iso < best_iso[0]:
            best_iso = (r_iso, L, "iso")
        if r_ctx < best_ctx[0]:
            best_ctx = (r_ctx, L, "ctx")
    layer_sweep_results[(prompt, ans)] = {"best_iso": best_iso, "best_ctx": best_ctx}

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T3 — Alpha sweep for each failure at best layer (iso)")
print("="*70)
alpha_sweep_results = {}

for prompt, ans, axis in FAILURES:
    best_iso = layer_sweep_results[(prompt, ans)]["best_iso"]
    best_ctx = layer_sweep_results[(prompt, ans)]["best_ctx"]
    best_L   = best_iso[1]
    t2_dir   = t2_iso[(axis, best_L)]

    print(f"\n  '{prompt}' → {ans}  best_L={best_L}")
    print(f"  {'α':<8} {'iso_rank':>10} {'ctx_rank':>10}")
    print(f"  {'-'*30}")
    best_alpha_iso, best_rank_iso = 20, best_iso[0]
    for a in ALPHA_SWEEP:
        r_iso = steer_rank(prompt, ans, t2_iso[(axis, best_L)], alpha=a, layer=best_L)
        r_ctx = steer_rank(prompt, ans, t2_ctx[(axis, best_L)], alpha=a, layer=best_L)
        mark_i = " ★" if r_iso < 5 else ""
        mark_c = " ★" if r_ctx < 5 else ""
        print(f"  {a:<8} {r_iso:>10}{mark_i}  {r_ctx:>10}{mark_c}")
        if r_iso < best_rank_iso:
            best_rank_iso = r_iso
            best_alpha_iso = a
    alpha_sweep_results[(prompt, ans)] = {
        "best_alpha": best_alpha_iso, "best_rank": best_rank_iso, "best_L": best_L
    }

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T4 — Negative direction test (steer AWAY from the transformation)")
print("="*70)
print("  If the correct direction helps, the negative should hurt.")
for prompt, ans, axis in FAILURES[:3]:   # sample 3
    best_L = layer_sweep_results[(prompt, ans)]["best_iso"][1]
    d = t2_iso[(axis, best_L)]
    r_pos = steer_rank(prompt, ans, d,  alpha=20, layer=best_L)
    r_neg = steer_rank(prompt, ans, -d, alpha=20, layer=best_L)
    r_bas = baselines[(prompt, ans)]
    print(f"  '{prompt[:45]}' → {ans}")
    print(f"    baseline={r_bas}  +dir={r_pos}  -dir={r_neg}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T5 — Cross-axis test (wrong T2 on each failure)")
print("="*70)
print("  Does the wrong morphological T2 also help, or is it axis-specific?\n")
cross_axes = {"base_comp": "base_super", "base_super": "base_comp",
              "gender": "base_comp"}
for prompt, ans, axis in FAILURES:
    wrong_axis = cross_axes[axis]
    best_L = layer_sweep_results[(prompt, ans)]["best_iso"][1]
    r_correct = steer_rank(prompt, ans, t2_iso[(axis,    best_L)], alpha=20, layer=best_L)
    r_wrong   = steer_rank(prompt, ans, t2_iso[(wrong_axis, best_L)], alpha=20, layer=best_L)
    r_bas = baselines[(prompt, ans)]
    diff = r_wrong - r_correct
    spec = "SPECIFIC" if diff > 3 else "not specific"
    print(f"  [{axis:>10}] '{prompt[:38]}' → {ans}")
    print(f"    base={r_bas}  correct={r_correct}  wrong={r_wrong}  Δ={diff:+}  ({spec})")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T6 — Best configuration matrix (per failure)")
print("="*70)
print(f"\n  {'Prompt':<42} {'ans':<10} {'base':>5} {'best_r':>7} {'rescued':>8}  config")
print(f"  {'-'*85}")

all_results = []
rescued = 0
for prompt, ans, axis in FAILURES:
    best_iso_r, best_iso_L, _ = layer_sweep_results[(prompt, ans)]["best_iso"]
    best_ctx_r, best_ctx_L, _ = layer_sweep_results[(prompt, ans)]["best_ctx"]
    alpha_res = alpha_sweep_results[(prompt, ans)]

    # best overall across iso + ctx
    if best_iso_r <= best_ctx_r:
        best_r = best_iso_r
        best_L = best_iso_L
        kind   = "iso"
        d      = t2_iso[(axis, best_L)]
    else:
        best_r = best_ctx_r
        best_L = best_ctx_L
        kind   = "ctx"
        d      = t2_ctx[(axis, best_L)]

    # final check: use best_alpha
    best_alpha = alpha_res["best_alpha"]
    final_r = steer_rank(prompt, ans, d, alpha=best_alpha, layer=best_L)
    is_rescued = final_r < 5
    if is_rescued:
        rescued += 1
    mark = "★ YES" if is_rescued else "  no"

    print(f"  '{prompt[:40]}'  {ans:<10} {baselines[(prompt,ans)]:>5} {final_r:>7}  {mark:>8}  "
          f"{kind} L{best_L} α={best_alpha}")

    all_results.append({
        "prompt": prompt, "answer": ans, "axis": axis,
        "baseline_rank": baselines[(prompt, ans)],
        "best_rank": final_r,
        "rescued": is_rescued,
        "config": {"kind": kind, "layer": best_L, "alpha": best_alpha}
    })

print(f"\n  Rescued: {rescued}/{len(FAILURES)}  ({100*rescued/len(FAILURES):.0f}%)")
print(f"  Magnitude-only baseline: 0/{len(FAILURES)} (0%)")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INTERPRETATION SUMMARY — Day 65")
print("="*70)

cos_comp_super = cosine(t2_iso[("base_comp", 23)], t2_iso[("base_super", 23)])
cos_comp_gen   = cosine(t2_iso[("base_comp", 23)], t2_iso[("gender",    23)])
cos_super_gen  = cosine(t2_iso[("base_super",23)], t2_iso[("gender",    23)])
cos_l14_l23_c  = cosine(t2_iso[("base_comp", 14)], t2_iso[("base_comp", 23)])
cos_l14_l23_s  = cosine(t2_iso[("base_super",14)], t2_iso[("base_super",23)])
cos_l14_l23_g  = cosine(t2_iso[("gender",   14)], t2_iso[("gender",    23)])
cos_iso_ctx_c  = cosine(t2_iso[("base_comp", 23)], t2_ctx[("base_comp", 23)])
cos_iso_ctx_s  = cosine(t2_iso[("base_super",23)], t2_ctx[("base_super",23)])
cos_iso_ctx_g  = cosine(t2_iso[("gender",   23)], t2_ctx[("gender",    23)])

print(f"""
  T2 direction geometry at L23:
    cos(base_comp, base_super) = {cos_comp_super:+.3f}
    cos(base_comp, gender)     = {cos_comp_gen:+.3f}
    cos(base_super, gender)    = {cos_super_gen:+.3f}

  Axis stability L14 vs L23 (isolated):
    cos(base_comp  L14, L23) = {cos_l14_l23_c:+.3f}
    cos(base_super L14, L23) = {cos_l14_l23_s:+.3f}
    cos(gender     L14, L23) = {cos_l14_l23_g:+.3f}

  Isolated vs contextual T2 (L23):
    cos(base_comp  iso, ctx) = {cos_iso_ctx_c:+.3f}
    cos(base_super iso, ctx) = {cos_iso_ctx_s:+.3f}
    cos(gender     iso, ctx) = {cos_iso_ctx_g:+.3f}

  Failures rescued: {rescued}/{len(FAILURES)}
""")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "axis_cosines_L23": {
        "base_comp_vs_super": cos_comp_super,
        "base_comp_vs_gender": cos_comp_gen,
        "base_super_vs_gender": cos_super_gen,
    },
    "axis_stability_L14_vs_L23": {
        "base_comp":  cos_l14_l23_c,
        "base_super": cos_l14_l23_s,
        "gender":     cos_l14_l23_g,
    },
    "iso_vs_ctx_L23": {
        "base_comp":  cos_iso_ctx_c,
        "base_super": cos_iso_ctx_s,
        "gender":     cos_iso_ctx_g,
    },
    "failures": all_results,
    "rescued": rescued,
    "total_failures": len(FAILURES),
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 65 complete.")
