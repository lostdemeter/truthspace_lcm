#!/usr/bin/env python3
"""
Day 64 — Characterising L23 Saddle-Point Breaking

Day 63 finding: ALL directions at α=20 achieve 87.5% @5. The effect is
direction-invariant and magnitude-sensitive. Three questions today:

D1: RANDOM DIRECTION CONTROL
    Do genuinely random unit vectors also achieve ~87.5%?
    Tests whether the effect is truly direction-invariant.

D2: FAILURE ANALYSIS
    Identify the 5 deeply-buried training failures.
    Why are they hard? Baseline rank, prompt type, recovery threshold?

D3: GYROSCOPE CHECK (1.5B)
    After L23 perturbation, track residual stream angle at each downstream
    layer. Does it converge to arccos(1/φ²) ≈ 68.4° as in the 7B?
    First direct test of Gyroscope structure in the 1.5B model.

D4: LAYER SWEEP (random direction)
    Is L23 uniquely powerful, or does the saddle-point effect appear
    at other layers too?

D5: COMBINED STEERING FOR FAILURES
    For the 5 buried failures: try L23 perturbation + T2 at L27.
    Can we recover any of them?
"""
import json, sys, time, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day64_saddle_point.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + math.sqrt(5)) / 2
GYROSCOPE_ANGLE = math.degrees(math.acos(1.0 / PHI**2))   # ≈ 68.4°

FILL_PROMPTS = [
    ("The plural of cat is",        "cats"),
    ("The plural of dog is",        "dogs"),
    ("The plural of tree is",       "trees"),
    ("The plural of bird is",       "birds"),
    ("The plural of house is",      "houses"),
    ("The plural of tree is",       "trees"),
    ("Boys and",                    "girls"),
    ("Up and",                      "down"),
    ("Black and",                   "white"),
    ("Day and",                     "night"),
    ("Hot and",                     "cold"),
    ("The past tense of walk is",   "walked"),
    ("The past tense of jump is",   "jumped"),
    ("The past tense of talk is",   "talked"),
    ("The past tense of help is",   "helped"),
    ("The past tense of play is",   "played"),
    ("The opposite of hot is",      "cold"),
    ("The opposite of fast is",     "slow"),
    ("The opposite of dark is",     "light"),
    ("The opposite of tall is",     "short"),
    ("The opposite of old is",      "young"),
    ("The male version of queen is","king"),
    ("The female version of boy is","girl"),
    ("The female version of man is","woman"),
    ("The male version of woman is","man"),
    ("The comparative of fast is",  "faster"),
    ("The comparative of small is", "smaller"),
    ("The comparative of big is",   "bigger"),
    ("The superlative of big is",   "biggest"),
    ("The superlative of fast is",  "fastest"),
    ("Water freezes and turns into","ice"),
    ("Ice melts and turns into",    "water"),
    ("A group of wolves is called a","pack"),
    ("A baby dog is called a",      "puppy"),
    ("A baby cat is called a",      "kitten"),
    ("The plural of mouse is",      "mice"),
    ("The plural of tooth is",      "teeth"),
    ("The plural of child is",      "children"),
    ("The opposite of loud is",     "quiet"),
    ("The colour of blood is",      "red"),
]

# T2 direction (singular_plural contextual, rebuilt from Day 62 pairs)
CTX_T2_TEMPLATES = {
    "singular_plural": [
        ("The cat sat on the mat", "The cats sat on the mat"),
        ("A dog runs in the park", "Dogs run in the park"),
        ("The tree grows tall",    "The trees grow tall"),
        ("A bird sings at dawn",   "Birds sing at dawn"),
        ("The child plays outside","The children play outside"),
        ("A mouse runs quickly",   "Mice run quickly"),
        ("The tooth fell out",     "The teeth fell out"),
        ("The house stands alone", "The houses stand alone"),
    ]
}

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"  n_layers={n_layers}  hidden={model.config.hidden_size}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def get_hs(prompt, layers=None):
    if layers is None:
        layers = list(range(n_layers + 1))
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return {L: out.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            for L in layers}

def token_rank(prompt, target_word):
    """Return rank of target token in next-token distribution."""
    target_ids = tok.encode(" " + target_word, add_special_tokens=False)
    if not target_ids:
        target_ids = tok.encode(target_word, add_special_tokens=False)
    tid = target_ids[0]
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    rank = int((logits > logits[tid]).sum().item())
    return rank

def steer_run(prompt, direction_np, alpha, layer=23):
    """Forward pass with h += alpha*direction at given layer, last token."""
    direction_t = torch.tensor(direction_np, dtype=torch.float32)
    handle = None
    def hook(module, inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[0, -1, :] += alpha * direction_t
            return (hs,) + out[1:]
        out[0, -1, :] += alpha * direction_t
        return out
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        handle.remove()
    return logits

def steer_rank(prompt, target_word, direction_np, alpha, layer=23):
    target_ids = tok.encode(" " + target_word, add_special_tokens=False)
    if not target_ids:
        target_ids = tok.encode(target_word, add_special_tokens=False)
    tid = target_ids[0]
    logits = steer_run(prompt, direction_np, alpha, layer)
    return int((logits > logits[tid]).sum().item())

def eval_direction(direction_np, prompts, alpha=20, layer=23):
    top1, top5 = 0, 0
    for p, ans in prompts:
        r = steer_rank(p, ans, direction_np, alpha, layer)
        if r == 0: top1 += 1
        if r < 5:  top5 += 1
    n = len(prompts)
    return top1/n, top5/n

# ── Rebuild SP contextual T2 at L23 ──────────────────────────────────────────
print("Rebuilding SP contextual direction at L23 ...")
diffs = []
for s1, s2 in CTX_T2_TEMPLATES["singular_plural"]:
    h1 = get_hs(s1, [23])[23]
    h2 = get_hs(s2, [23])[23]
    d  = h2 - h1
    diffs.append(d / (np.linalg.norm(d) + 1e-12))
sp_dir = np.mean(diffs, axis=0)
sp_dir = sp_dir / (np.linalg.norm(sp_dir) + 1e-12)
print("  SP direction built.\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("D1 — Random Direction Control (15 random unit vectors)")
print("=" * 70)

rng = np.random.default_rng(42)
hidden_dim = model.config.hidden_size
random_results = []
for i in range(15):
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    v /= np.linalg.norm(v)
    cos_sp = cosine(v, sp_dir)
    t1, t5 = eval_direction(v, FILL_PROMPTS, alpha=20, layer=23)
    random_results.append({"seed": i, "cos_sp": cos_sp, "top1": t1, "top5": t5})
    print(f"  rand_{i:02d}  cos(sp)={cos_sp:+.3f}  top1={t1:.3f}  top5={t5:.3f}")

mean_t5 = np.mean([r["top5"] for r in random_results])
std_t5  = np.std( [r["top5"] for r in random_results])
print(f"\n  Random direction @5: mean={mean_t5:.3f}  std={std_t5:.3f}")
print(f"  SP direction @5:     0.875")
print(f"  Direction-invariant: {'YES' if std_t5 < 0.05 else 'NO — variance too high'}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D2 — Failure Analysis (identify the 5 deeply-buried failures)")
print("=" * 70)

failures, near_misses, successes = [], [], []
for p, ans in FILL_PROMPTS:
    baseline = token_rank(p, ans)
    steered  = steer_rank(p, ans, sp_dir, alpha=20, layer=23)
    delta    = steered - baseline
    entry = {"prompt": p, "answer": ans,
             "baseline_rank": baseline, "steered_rank": steered, "delta": delta}
    if steered >= 5:
        failures.append(entry)
    elif baseline >= 5:
        near_misses.append(entry)
    else:
        successes.append(entry)

print(f"\n  FAILURES (steered rank ≥ 5):  {len(failures)}")
for f in sorted(failures, key=lambda x: x["steered_rank"], reverse=True):
    print(f"    rank {f['baseline_rank']:>4}→{f['steered_rank']:>4}  '{f['prompt']}' → {f['answer']}")

print(f"\n  NEAR-MISSES rescued by steering (baseline ≥5, steered <5): {len(near_misses)}")
for nm in sorted(near_misses, key=lambda x: x["baseline_rank"], reverse=True):
    print(f"    rank {nm['baseline_rank']:>4}→{nm['steered_rank']:>4}  '{nm['prompt']}' → {nm['answer']}")

print(f"\n  Easy (both <5): {len(successes)}")

# Try recovery at higher alpha for failures
print("\n  Recovery sweep for failures (α=50,100,200,500):")
print(f"  {'Prompt':<40} {'ans':<10} ", end="")
for a in [50, 100, 200, 500]:
    print(f"  α={a}", end="")
print()
for f in failures:
    print(f"  '{f['prompt'][:38]}'  {f['answer']:<10}", end="")
    for a in [50, 100, 200, 500]:
        r = steer_rank(f["prompt"], f["answer"], sp_dir, alpha=a, layer=23)
        mark = "★" if r < 5 else f"{r:>3}"
        print(f"  {mark:>4}", end="")
    print()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D3 — Gyroscope Check (track orbit angle after L23 perturbation)")
print("=" * 70)

phi_pred = GYROSCOPE_ANGLE
print(f"  Gyroscope prediction (7B): steady-state angle = {phi_pred:.2f}° = arccos(1/φ²)")
print(f"  Testing on 5 prompts × random direction × α=20 at L23 ...\n")

def track_orbit(prompt, direction_np, alpha, perturb_layer=23):
    """Return angles between perturbed and unperturbed hs at every layer > perturb_layer."""
    direction_t = torch.tensor(direction_np, dtype=torch.float32)
    baseline_hs = {}
    steered_hs  = {}

    # hooks to capture all hidden states
    def make_capture_hook(store, layer_idx):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[layer_idx] = hs[0, -1, :].detach().cpu().numpy().copy()
        return hook

    def make_perturb_hook(store, layer_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[0, -1, :] += alpha * direction_t
                store[layer_idx] = hs[0, -1, :].detach().cpu().numpy().copy()
                return (hs,) + out[1:]
            out[0, -1, :] += alpha * direction_t
            store[layer_idx] = out[0, -1, :].detach().cpu().numpy().copy()
            return out
        return hook

    handles = []
    inputs = tok(prompt, return_tensors="pt")

    # baseline pass
    for L in range(n_layers):
        h = model.model.layers[L].register_forward_hook(make_capture_hook(baseline_hs, L))
        handles.append(h)
    with torch.no_grad():
        model(**inputs, output_hidden_states=False)
    for h in handles: h.remove()

    # steered pass
    handles = []
    for L in range(n_layers):
        if L == perturb_layer:
            h = model.model.layers[L].register_forward_hook(
                make_perturb_hook(steered_hs, L))
        else:
            h = model.model.layers[L].register_forward_hook(
                make_capture_hook(steered_hs, L))
        handles.append(h)
    with torch.no_grad():
        model(**inputs, output_hidden_states=False)
    for h in handles: h.remove()

    # compute angles at layers > perturb_layer
    angles = {}
    for L in range(perturb_layer + 1, n_layers):
        if L in baseline_hs and L in steered_hs:
            cos_val = cosine(baseline_hs[L], steered_hs[L])
            cos_val = float(np.clip(cos_val, -1.0, 1.0))
            angles[L] = math.degrees(math.acos(cos_val))
    return angles

test_prompts_gyro = [
    "The plural of cat is",
    "The capital of France is",
    "Water freezes and turns into",
    "The past tense of walk is",
    "The opposite of hot is",
]
rand_dir = rng.standard_normal(hidden_dim).astype(np.float32)
rand_dir /= np.linalg.norm(rand_dir)

all_angle_curves = []
for prompt in test_prompts_gyro:
    angles = track_orbit(prompt, rand_dir, alpha=20, perturb_layer=23)
    all_angle_curves.append(angles)
    final_angle = angles.get(n_layers - 1, float("nan"))
    print(f"  '{prompt[:45]}'")
    layer_vals = [(L, angles[L]) for L in sorted(angles.keys())]
    # show every 3 layers
    preview = "  ".join(f"L{L}={a:.1f}°" for L, a in layer_vals[::3])
    print(f"    {preview}")
    print(f"    Final (L{n_layers-1}): {final_angle:.2f}°  (predicted: {phi_pred:.1f}°)\n")

# Mean convergence curve
mean_angles = {}
for L in range(24, n_layers):
    vals = [c[L] for c in all_angle_curves if L in c]
    if vals:
        mean_angles[L] = float(np.mean(vals))

final_mean = mean_angles.get(n_layers - 1, float("nan"))
print(f"  Mean final angle (L{n_layers-1}): {final_mean:.2f}°")
print(f"  Gyroscope prediction:           {phi_pred:.2f}°")
print(f"  Error: {abs(final_mean - phi_pred):.2f}°")
converged = abs(final_mean - phi_pred) < 5.0
print(f"  Gyroscope holds in 1.5B: {'YES (within 5°)' if converged else 'NO — different orbit radius'}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D4 — Layer Sweep (random direction, α=20)")
print("=" * 70)
print("  Is L23 special, or does saddle-point breaking work everywhere?\n")

print(f"  {'Layer':<8}  {'top1':<8}  {'top5':<8}")
print(f"  {'-'*28}")
layer_sweep = {}
for L in [0, 1, 5, 10, 14, 18, 20, 21, 22, 23, 24, 25, 26, 27]:
    t1, t5 = eval_direction(rand_dir, FILL_PROMPTS, alpha=20, layer=L)
    layer_sweep[L] = {"top1": t1, "top5": t5}
    marker = " ←" if t5 >= 0.85 else ""
    print(f"  L{L:<7}  {t1:<8.3f}  {t5:.3f}{marker}")

best_layer = max(layer_sweep, key=lambda x: layer_sweep[x]["top5"])
print(f"\n  Best layer: L{best_layer} (top5={layer_sweep[best_layer]['top5']:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D5 — Combined Steering for Buried Failures")
print("=" * 70)
print("  For each failure: try L23 perturb + T2 at L27 simultaneously\n")

# Rebuild T2 at L27 for singular_plural
print("  Building SP T2 at L27 ...")
diffs27 = []
for s1, s2 in CTX_T2_TEMPLATES["singular_plural"]:
    h1 = get_hs(s1, [27])[27]
    h2 = get_hs(s2, [27])[27]
    d  = h2 - h1
    diffs27.append(d / (np.linalg.norm(d) + 1e-12))
sp_dir27 = np.mean(diffs27, axis=0)
sp_dir27 /= np.linalg.norm(sp_dir27) + 1e-12

def steer_two_layers(prompt, d23, alpha23, d27, alpha27):
    """Steer at L23 and L27 simultaneously."""
    d23_t = torch.tensor(d23, dtype=torch.float32)
    d27_t = torch.tensor(d27, dtype=torch.float32)
    handles = []
    def hook23(module, inp, out):
        if isinstance(out, tuple):
            out[0][0, -1, :] += alpha23 * d23_t
            return out
        out[0, -1, :] += alpha23 * d23_t
        return out
    def hook27(module, inp, out):
        if isinstance(out, tuple):
            out[0][0, -1, :] += alpha27 * d27_t
            return out
        out[0, -1, :] += alpha27 * d27_t
        return out
    handles.append(model.model.layers[23].register_forward_hook(hook23))
    handles.append(model.model.layers[27].register_forward_hook(hook27))
    try:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        for h in handles: h.remove()
    return logits

if failures:
    print(f"  {'Prompt':<40} {'ans':<10} base  L23  L23+L27(sp)  L23+L27(rand)")
    print(f"  {'-'*75}")
    for f in failures:
        p, ans = f["prompt"], f["answer"]
        target_ids = tok.encode(" " + ans, add_special_tokens=False)
        if not target_ids:
            target_ids = tok.encode(ans, add_special_tokens=False)
        tid = target_ids[0]

        # combined: L23 random + L27 sp
        logits_comb_sp = steer_two_layers(p, rand_dir, 20, sp_dir27, 20)
        rank_comb_sp   = int((logits_comb_sp > logits_comb_sp[tid]).sum().item())

        # combined: L23 random + L27 random
        rand27 = rng.standard_normal(hidden_dim).astype(np.float32)
        rand27 /= np.linalg.norm(rand27)
        logits_comb_r  = steer_two_layers(p, rand_dir, 20, rand27, 20)
        rank_comb_r    = int((logits_comb_r > logits_comb_r[tid]).sum().item())

        mark_sp   = "★" if rank_comb_sp < 5   else f"{rank_comb_sp:>3}"
        mark_r    = "★" if rank_comb_r  < 5   else f"{rank_comb_r:>3}"
        mark_base = "★" if f["baseline_rank"] < 5 else f"{f['baseline_rank']:>3}"
        mark_l23  = "★" if f["steered_rank"]  < 5 else f"{f['steered_rank']:>3}"

        print(f"  '{p[:38]}'  {ans:<10} {mark_base:>4}  {mark_l23:>4}  {mark_sp:>11}  {mark_r:>12}")
else:
    print("  No failures to test.")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Day 64")
print("=" * 70)

print(f"""
  D1 Random direction control:
     Mean top5 (15 random vectors) = {mean_t5:.3f}  std={std_t5:.3f}
     SP direction top5             = 0.875
     Direction-invariance: {'CONFIRMED' if std_t5 < 0.05 else 'NOT CONFIRMED'}

  D2 Failure analysis:
     Failures (steered rank ≥5): {len(failures)}/40
     Failure prompts: {[f['prompt'] for f in failures]}

  D3 Gyroscope check:
     Final mean orbit angle (1.5B): {final_mean:.2f}°
     Predicted (7B):                {phi_pred:.2f}°
     Result: {'CONFIRMED' if converged else 'DIVERGES from 7B prediction'}

  D4 Layer sweep:
     Best layer: L{best_layer} (top5={layer_sweep[best_layer]['top5']:.3f})
     L23 top5: {layer_sweep.get(23, {}).get('top5', 'N/A'):.3f}

  D5 Combined steering:
     Failures tested: {len(failures)}
""")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "D1_random_directions": random_results,
    "D1_mean_top5": float(mean_t5),
    "D1_std_top5":  float(std_t5),
    "D2_failures":   failures,
    "D2_near_misses": near_misses,
    "D3_gyroscope_mean_final_angle": float(final_mean),
    "D3_gyroscope_predicted":        float(phi_pred),
    "D3_gyroscope_confirmed":        bool(converged),
    "D3_mean_angle_curve":           {str(k): v for k, v in mean_angles.items()},
    "D4_layer_sweep":                {str(k): v for k, v in layer_sweep.items()},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 64 complete.")
