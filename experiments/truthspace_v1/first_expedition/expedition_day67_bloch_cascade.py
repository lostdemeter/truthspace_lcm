#!/usr/bin/env python3
"""
Day 67 — Bloch Sphere Packing and Displacement Cascade

Hypothesis (from Day 66 conformal return signal):
  Each token owns a "Bloch sphere" in semantic space — a region it dominates.
  When a T2 tidal force evicts a token from its sphere, it lands inside a
  neighbouring sphere that is already densely packed. The intruder must
  displace something, which displaces something else... a cascade.
  This cascade is what the exponential rank explosion looks like from outside.
  The "conformal return" (ranks decreasing at extreme α) is the cascade
  settling as the token is absorbed by one specific sphere's attractor.

If this is right, the cascade should be STRUCTURED:
  - Phase 2 (fracture onset): token's immediate semantic neighbours appear
    in top-10, displacing the original token
  - Phase 3 (disintegration): wave propagates outward — ever-more-distant
    tokens flood through top-10 in sequence
  - Phase 4 (conformal return): cascade settles, top-10 stabilises into a
    new coherent neighbourhood — the "antipodal Bloch sphere"

If the cascade is random (not structured), the displacing tokens should have
no semantic relationship to the evicted token. If structured, they should
form a coherent semantic neighbourhood ordered by proximity.

Measurements:
  B1: Full α sweep — track top-10 tokens at every α step
  B2: Cascade structure — are displacing tokens semantic neighbours?
  B3: Antipodal attractor — which token is rank 0 at extreme α?
  B4: Cascade entropy — variance of top-10 composition across the sweep
  B5: Settling signature — at what α does the top-10 stabilise?
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day67_bloch_cascade.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Ctx T2 pairs (comparative)
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

# Test prompts — one easy (eviction cascade) and one fracture cascade
TEST_CASES = [
    # (prompt, target, axis, best_layer, label)
    ("The plural of dog is",         "dogs",   "comp", 14, "easy_dogs"),
    ("The plural of cat is",         "cats",   "comp", 14, "easy_cats"),
    ("The comparative of big is",    "bigger", "comp", 14, "fracture_bigger"),
    ("The comparative of fast is",   "faster", "comp", 27, "incomplete_faster"),
]

# Fine sweep — dense at boundaries, wide range to catch conformal return
FINE_ALPHA = [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30,
              40, 50, 60, 75, 100, 130, 170, 220, 300, 400, 600, 1000]

TOP_K = 15   # tokens to track at each α

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers   = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
vocab_size = model.config.vocab_size
print(f"  n_layers={n_layers}  hidden={hidden_dim}  vocab={vocab_size}\n")

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def get_hs(text, layers, is_word=False):
    if is_word: text = " " + text.strip()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return {L: out.hidden_states[L][0, -1, :].numpy().astype(np.float32) for L in layers}

def get_logits(prompt, direction_np=None, alpha=0, layer=14):
    inputs = tok(prompt, return_tensors="pt")
    if direction_np is None or alpha == 0:
        with torch.no_grad():
            return model(**inputs).logits[0, -1, :].numpy()
    d_t = torch.tensor(direction_np, dtype=torch.float32)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out[0][0, -1, :] += alpha * d_t; return out
        out[0, -1, :] += alpha * d_t; return out
    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].numpy()
    finally:
        h.remove()
    return logits

def build_ctx_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        h1 = get_hs(s1, [layer])[layer]; h2 = get_hs(s2, [layer])[layer]
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)

def decode_top_k(logits, k=15):
    """Return list of (token_str, rank, logit_val) for top-k."""
    top_ids = np.argsort(logits)[::-1][:k]
    results = []
    for rank, tid in enumerate(top_ids):
        tok_str = tok.decode([tid]).strip()
        results.append({"token": tok_str, "id": int(tid),
                        "rank": rank, "logit": float(logits[tid])})
    return results

def token_rank(logits, target_word):
    ids = tok.encode(" " + target_word, add_special_tokens=False) or \
          tok.encode(target_word, add_special_tokens=False)
    tid = ids[0]
    return int((logits > logits[tid]).sum())

# ── Build T2 directions ───────────────────────────────────────────────────────
print("Building T2 directions ...")
t2_by_layer = {L: build_ctx_t2(CTX_PAIRS_COMP, L) for L in [14, 27]}
print("  Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("B1/B2/B3 — Full α sweep: top-15 tokens at each step")
print("="*70)

all_results = {}

for prompt, target, axis, best_L, label in TEST_CASES:
    t2 = t2_by_layer[best_L]
    print(f"\n{'─'*70}")
    print(f"  {label.upper()}: '{prompt}' → {target}  (L{best_L})")
    print(f"{'─'*70}")

    target_ids = tok.encode(" " + target, add_special_tokens=False) or \
                 tok.encode(target, add_special_tokens=False)
    target_id = target_ids[0]

    sweep_data = []
    rank0_tokens = {}   # α → token at rank 0
    target_ranks = {}   # α → rank of target

    # Track all tokens that appear in top-15 across the sweep
    top15_ever = set()

    for a in FINE_ALPHA:
        logits = get_logits(prompt, t2 if a > 0 else None, alpha=a, layer=best_L)
        top15  = decode_top_k(logits, k=TOP_K)
        t_rank = token_rank(logits, target)

        rank0_tok = top15[0]["token"]
        rank0_tokens[a]   = rank0_tok
        target_ranks[a]   = t_rank
        for entry in top15:
            top15_ever.add(entry["token"])

        sweep_data.append({
            "alpha": a,
            "target_rank": t_rank,
            "rank0_token": rank0_tok,
            "top15": top15,
        })

        # Print summary row
        top5_str = "  ".join(
            f"[{e['token'][:8]}]" + ("★" if e["token"].strip() == target.strip() else "")
            for e in top15[:5])
        print(f"  α={a:>5}  tgt_rank={t_rank:>6}  top5: {top5_str}")

    all_results[label] = {
        "prompt": prompt, "target": target, "layer": best_L,
        "sweep": sweep_data,
        "total_unique_top15": len(top15_ever),
    }

    print(f"\n  Total unique tokens appearing in top-15 across sweep: {len(top15_ever)}")
    print(f"  Tokens that appeared: {sorted(top15_ever)[:30]}...")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B2 — Cascade structure: semantic distance of displacing tokens")
print("="*70)
print("  Are displacing tokens semantic neighbours of the target?\n")

for prompt, target, axis, best_L, label in TEST_CASES[:2]:  # easy cases
    t2  = t2_by_layer[best_L]
    print(f"  {label}: '{prompt}' → {target}")

    # Get target token embedding at L14
    h_target = get_hs(" " + target, [14], is_word=True)[14]

    # For each α, get rank-0 token and measure its semantic distance to target
    print(f"  {'α':>6}  {'rank-0 token':>15}  {'cos_to_target':>14}  tgt_rank")
    print(f"  {'-'*55}")
    sweep = all_results[label]["sweep"]
    for step in sweep:
        a     = step["alpha"]
        r0    = step["rank0_token"]
        t_rank= step["target_rank"]
        if r0:
            h_r0  = get_hs(r0, [14], is_word=True)[14]
            cos_r0= cosine(h_target, h_r0)
        else:
            cos_r0 = float("nan")
        print(f"  {a:>6}  {r0:>15}  {cos_r0:>14.4f}  {t_rank}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B3 — Antipodal attractor: what is rank 0 at extreme α?")
print("="*70)

for prompt, target, axis, best_L, label in TEST_CASES:
    sweep = all_results[label]["sweep"]
    # Find α where target rank is maximum (deepest void)
    max_rank_step = max(sweep, key=lambda s: s["target_rank"])
    # Find rank-0 token at extreme alpha
    extreme_step  = next(s for s in reversed(sweep) if s["alpha"] == FINE_ALPHA[-1])

    print(f"\n  {label}:")
    print(f"    Target '{target}' deepest at α={max_rank_step['alpha']} "
          f"(rank {max_rank_step['target_rank']})")
    print(f"    Rank-0 at α={max_rank_step['alpha']}: '{max_rank_step['rank0_token']}'")
    print(f"    Rank-0 at α={FINE_ALPHA[-1]} (extreme): '{extreme_step['rank0_token']}'")
    print(f"    Top-5 at α={FINE_ALPHA[-1]}:",
          "  ".join(f"[{e['token'][:10]}]" for e in extreme_step["top15"][:5]))

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B4 — Cascade entropy: how much does the top-15 composition change?")
print("="*70)
print("  Low entropy = stable orbit, High entropy = cascade active\n")

for prompt, target, axis, best_L, label in TEST_CASES:
    sweep = all_results[label]["sweep"]
    print(f"  {label}:")

    prev_set = None
    for step in sweep:
        curr_set = {e["token"] for e in step["top15"]}
        if prev_set is not None:
            jaccard = len(curr_set & prev_set) / len(curr_set | prev_set)
            overlap = len(curr_set & prev_set)
            print(f"    α={step['alpha']:>5}  top15_overlap={overlap:>3}/15  "
                  f"jaccard={jaccard:.3f}  rank0=[{step['rank0_token'][:10]}]  "
                  f"tgt={step['target_rank']}")
        else:
            print(f"    α={step['alpha']:>5}  (baseline)  "
                  f"rank0=[{step['rank0_token'][:10]}]  tgt={step['target_rank']}")
        prev_set = curr_set
    print()

# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("B5 — Settling signature: when does the top-10 stabilise?")
print("="*70)
print("  Cascade settling = Jaccard similarity stays high for 3+ consecutive steps\n")

for prompt, target, axis, best_L, label in TEST_CASES:
    sweep = all_results[label]["sweep"]
    jaccards = []
    for i in range(1, len(sweep)):
        curr = {e["token"] for e in sweep[i]["top15"]}
        prev = {e["token"] for e in sweep[i-1]["top15"]}
        jaccards.append((sweep[i]["alpha"], len(curr & prev) / len(curr | prev)))

    # Find where Jaccard stays > 0.8 for 3 consecutive steps
    settle_alpha = None
    for i in range(len(jaccards) - 2):
        if all(jaccards[i+j][1] > 0.7 for j in range(3)):
            settle_alpha = jaccards[i][0]
            break

    print(f"  {label}: settles at α≈{settle_alpha}")
    # Print Jaccard curve summary
    for a, j in jaccards:
        bar = "█" * int(j * 20) + "░" * (20 - int(j * 20))
        print(f"    α={a:>5}  [{bar}] {j:.3f}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("SYNTHESIS — Bloch Sphere Packing Model")
print("="*70)

print("""
  PREDICTIONS OF THE MODEL:
    1. Cascade is structured: displacing tokens are semantic neighbours first
    2. Top-15 diversity is high during disintegration (cascade active)
    3. Top-15 stabilises into coherent cluster (new Bloch sphere)
    4. Antipodal attractor is semantically coherent (not random)
    5. Jaccard drops to near-zero at peak disintegration, rises at settling

  Checking against data:
""")

for prompt, target, axis, best_L, label in TEST_CASES[:2]:
    sweep    = all_results[label]["sweep"]
    unique   = all_results[label]["total_unique_top15"]

    # Phase 3 detection: step with most top-15 turnover
    jaccards_vals = []
    for i in range(1, len(sweep)):
        curr = {e["token"] for e in sweep[i]["top15"]}
        prev = {e["token"] for e in sweep[i-1]["top15"]}
        jaccards_vals.append(len(curr & prev) / len(curr | prev))
    min_j = min(jaccards_vals)
    min_j_alpha = sweep[jaccards_vals.index(min_j) + 1]["alpha"]

    extreme = next(s for s in reversed(sweep) if s["alpha"] == FINE_ALPHA[-1])
    coherent_end = len({e["token"][:3] for e in extreme["top15"][:5]}) < 4

    print(f"  {label}:")
    print(f"    Unique tokens in top-15 across sweep: {unique} / {vocab_size}")
    print(f"    Min Jaccard (peak cascade): {min_j:.3f} at α={min_j_alpha}")
    print(f"    Extreme top-5 coherent: {coherent_end}")
    extremes = [e["token"] for e in extreme["top15"][:5]]
    print(f"    Extreme top-5 tokens: {extremes}")
    print()

# Save
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 67 complete.")
