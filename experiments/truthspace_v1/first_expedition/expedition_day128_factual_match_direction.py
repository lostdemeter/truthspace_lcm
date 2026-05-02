#!/usr/bin/env python3
"""
Day 128 — Factual Match Direction at L25

Day 127 found that the full hidden state at L25 achieves MRR=0.783 (78% oracle),
but there's a 22% gap that requires DIRECTED factual knowledge.

QUESTION: Does h_L25(correct_answer) - h_L25(wrong_answer) point consistently
across different prompts in the same factual category?

  Δ_capitals = h(Paris) - h(London)  (correct - wrong, France context)
  Δ_capitals = h(Tokyo) - h(Seoul)   (correct - wrong, Japan context)
  Δ_capitals = h(Berlin) - h(Frankfurt) (correct - wrong, Germany context)

If these Δ vectors are consistent (high pairwise cosine), there exists a
UNIVERSAL "factual match direction" for the capitals category.
If inconsistent (low pairwise cosine), factual directions are content-specific.

This determines whether the 22% oracle gap can be closed with an axis
(like T2 axes) or requires full computation.

EXPERIMENT:
  1. For each (correct, wrong) pair within a category, compute Δ = h_L25(correct) - h_L25(wrong)
  2. Measure pairwise cosine of all Δ vectors (stability test, like Day 123)
  3. Compute a "category factual axis" = mean Δ
  4. Test if this axis helps rank candidates better (closes the 22% gap)
  5. Compare factual direction stability across:
     - Isolated words (no prompt context)
     - Words in retrieval context ("The capital of France is Paris/London")
     - Words in neutral context ("Paris is a word. London is a word.")

LAYERS TO TEST: L15, L20, L23, L25, L27 (focused on the L25 peak region)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from itertools import combinations

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day128_factual_match_direction.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

TEST_LAYERS   = [15, 20, 23, 25, 27]
PRIMARY_LAYER = 25

# Factual pairs: (category, correct, wrong_pool, prompt_template)
FACTUAL_GROUPS = {
    "capitals": {
        "pairs": [
            ("Paris",  ["London","Rome","Berlin","Madrid"],   "The capital city of France is"),
            ("Tokyo",  ["Osaka","Beijing","Seoul","Bangkok"],  "The capital city of Japan is"),
            ("Berlin", ["Frankfurt","Vienna","Warsaw","Amsterdam"], "The capital city of Germany is"),
            ("Madrid", ["Lisbon","Rome","Paris","Brussels"],   "The capital city of Spain is"),
            ("Rome",   ["Milan","Paris","Vienna","Athens"],    "The capital city of Italy is"),
            ("Moscow", ["Petersburg","Kiev","Minsk","Warsaw"], "The capital city of Russia is"),
        ],
    },
    "languages": {
        "pairs": [
            ("Portuguese", ["Spanish","Italian","French","English"], "The official language of Brazil is"),
            ("Arabic",     ["Hebrew","Turkish","Persian","Urdu"],    "The official language of Egypt is"),
            ("Mandarin",   ["Japanese","Korean","Cantonese","Thai"], "The official language of China is"),
            ("Hindi",      ["Urdu","Bengali","Tamil","Punjabi"],     "The official language of India is"),
        ],
    },
    "antonyms": {
        "pairs": [
            ("cold",    ["warm","lukewarm","mild","chilly"],   "The opposite of hot is"),
            ("sad",     ["angry","bored","worried","tired"],   "The opposite of happy is"),
            ("dark",    ["bright","light","dim","pale"],       "The opposite of light is"),
            ("slow",    ["fast","quick","rapid","swift"],      "The opposite of slow is"),
            ("small",   ["large","big","huge","great"],        "The opposite of large is"),
        ],
    },
    "hypernyms": {
        "pairs": [
            ("dog",    ["cat","rabbit","horse","pet"],          "A poodle is a type of"),
            ("flower", ["tree","bush","grass","weed"],          "A rose is a type of"),
            ("tool",   ["machine","device","appliance","weapon"],"A hammer is a type of"),
            ("animal", ["plant","insect","fungus","bacterium"], "A whale is a type of"),
        ],
    },
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_layers    = model.config.num_hidden_layers
print(f"  hidden={hidden_size}\n")

def get_hs(text, layers):
    inp = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                for L in layers}
    except:
        return {L: np.zeros(hidden_size, np.float32) for L in layers}

def get_logprob(prompt, word):
    inp = tok(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            lp = torch.log_softmax(logits, dim=-1).numpy()
        ids = tok(" " + word, add_special_tokens=False)["input_ids"]
        return float(lp[ids[0]]) if ids else float("-inf")
    except:
        return float("-inf")

print("=" * 72)
print("Part 1: Factual Match Direction Stability (isolated words)")
print("=" * 72)
print()
print(f"  {'category':>12}  {'L':>4}  {'mean_cos':>10}  {'min_cos':>10}  {'max_cos':>10}  {'verdict':>10}")
print(f"  {'-'*62}")

stability_results = {}
all_axes = {}

for cat_name, cat_data in FACTUAL_GROUPS.items():
    pairs  = cat_data["pairs"]
    stability_results[cat_name] = {}
    all_axes[cat_name] = {}

    for L in TEST_LAYERS:
        # Compute Δ = h(correct) - h(wrong_pool_mean) for each pair
        deltas = []
        for correct, wrong_pool, prompt in pairs:
            h_cor = get_hs(" " + correct, [L])[L]
            h_wrg = np.mean([get_hs(" " + w, [L])[L] for w in wrong_pool], axis=0).astype(np.float32)
            d = h_cor - h_wrg
            nv = np.linalg.norm(d)
            if nv > 1e-6: deltas.append((d/nv).astype(np.float32))

        if len(deltas) < 2:
            stability_results[cat_name][L] = {"mean": 0.0, "deltas": []}
            continue

        # Pairwise cosines
        pw = [cosine(deltas[i], deltas[j])
              for i, j in combinations(range(len(deltas)), 2)]
        mean_cos = float(np.mean(pw))
        verdict  = ("STABLE"  if mean_cos > 0.5 else
                    "PARTIAL" if mean_cos > 0.3 else "VARIABLE")

        # Mean axis
        axis = np.mean(deltas, axis=0).astype(np.float32)
        nv   = np.linalg.norm(axis)
        axis = (axis/nv if nv > 1e-6 else axis)
        all_axes[cat_name][L] = axis

        stability_results[cat_name][L] = {
            "mean": mean_cos, "min": float(np.min(pw)), "max": float(np.max(pw)),
            "verdict": verdict, "n_deltas": len(deltas),
        }
        print(f"  {cat_name:>12}  L{L:02d}  {mean_cos:>+10.4f}  {float(np.min(pw)):>+10.4f}  "
              f"{float(np.max(pw)):>+10.4f}  {verdict:>10}")
    print()

print()
print("=" * 72)
print("Part 2: Does the factual axis help ranking? (L25 axis applied to prompt context)")
print("=" * 72)
print()

# For each category, compute the factual axis at L25 from isolated words,
# then project candidate hidden states at L25 (isolated) onto the axis.
# Does this axis improve MRR?
print(f"  {'category':>12}  {'MRR_baseline':>14}  {'MRR_factual':>13}  {'MRR_combo':>11}")
print(f"  {'-'*56}")

mrr_improvements = []
for cat_name, cat_data in FACTUAL_GROUPS.items():
    pairs = cat_data["pairs"]
    if PRIMARY_LAYER not in all_axes.get(cat_name, {}): continue
    fax = all_axes[cat_name][PRIMARY_LAYER]

    base_mrrs = []; fax_mrrs = []; combo_mrrs = []
    for correct, wrong_pool, prompt in pairs:
        all_cands = [correct] + wrong_pool

        # Baseline: L25 full cosine (Day 127 best method)
        ctx_h = get_hs(prompt, [PRIMARY_LAYER])[PRIMARY_LAYER]
        cand_hs = {w: get_hs(" "+w, [PRIMARY_LAYER])[PRIMARY_LAYER] for w in all_cands}
        base_sims = {w: cosine(ctx_h, cand_hs[w]) for w in all_cands}

        # Factual axis projection
        fax_projs = {w: float(np.dot(normed(cand_hs[w]), fax)) for w in all_cands}

        # Combined
        b_v = np.array([base_sims[w] for w in all_cands])
        f_v = np.array([fax_projs[w] for w in all_cands])
        b_n = (b_v - b_v.min()) / (b_v.max()-b_v.min() + 1e-8)
        f_n = (f_v - f_v.min()) / (f_v.max()-f_v.min() + 1e-8)
        combo = 0.7 * b_n + 0.3 * f_n
        combo_sims = {w: float(combo[i]) for i, w in enumerate(all_cands)}

        def mrr_1(scores, higher=True):
            ranked = sorted(all_cands, key=lambda w: (-scores[w] if higher else scores[w]))
            r = next((i+1 for i,w in enumerate(ranked) if w == correct), len(all_cands)+1)
            return 1.0/r

        base_mrrs.append(mrr_1(base_sims))
        fax_mrrs.append(mrr_1(fax_projs))
        combo_mrrs.append(mrr_1(combo_sims))

    mb = float(np.mean(base_mrrs)); mf = float(np.mean(fax_mrrs)); mc = float(np.mean(combo_mrrs))
    mrr_improvements.append(mc - mb)
    print(f"  {cat_name:>12}  {mb:>14.4f}  {mf:>13.4f}  {mc:>11.4f}  "
          f"({'+'if mc>=mb else '-'}{abs(mc-mb):.4f})")

print()
print("=" * 72)
print("Part 3: Factual direction in RETRIEVAL CONTEXT (words in context vs isolated)")
print("=" * 72)
print()
print("  Does Δ(correct-wrong) have higher stability in retrieval context?")
print()

for cat_name in list(FACTUAL_GROUPS.keys())[:2]:  # first two categories
    pairs   = FACTUAL_GROUPS[cat_name]["pairs"]
    L       = PRIMARY_LAYER
    print(f"  [{cat_name}]")

    iso_deltas = []; ctx_deltas = []
    for correct, wrong_pool, prompt in pairs:
        # Isolated deltas
        h_cor_iso = get_hs(" " + correct, [L])[L]
        h_wrg_iso = np.mean([get_hs(" " + w, [L])[L] for w in wrong_pool[:2]], axis=0).astype(np.float32)
        d_iso = h_cor_iso - h_wrg_iso
        nv = np.linalg.norm(d_iso)
        if nv > 1e-6: iso_deltas.append(d_iso/nv)

        # Contextual deltas
        h_cor_ctx = get_hs(prompt + " " + correct, [L])[L]
        h_wrg_ctx = np.mean([get_hs(prompt + " " + w, [L])[L] for w in wrong_pool[:2]], axis=0).astype(np.float32)
        d_ctx = h_cor_ctx - h_wrg_ctx
        nv = np.linalg.norm(d_ctx)
        if nv > 1e-6: ctx_deltas.append(d_ctx/nv)

    if len(iso_deltas) >= 2:
        pw_iso = [cosine(iso_deltas[i], iso_deltas[j])
                  for i,j in combinations(range(len(iso_deltas)), 2)]
        pw_ctx = [cosine(ctx_deltas[i], ctx_deltas[j])
                  for i,j in combinations(range(len(ctx_deltas)), 2)]
        print(f"    Isolated  stability:  mean_cos={np.mean(pw_iso):.4f}")
        print(f"    Contextual stability: mean_cos={np.mean(pw_ctx):.4f}")
        print(f"    Context {'improves' if np.mean(pw_ctx)>np.mean(pw_iso) else 'DOES NOT improve'} factual direction consistency")
    print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 128 Summary — Factual Match Direction at L25")
print("=" * 72)
print()

# Best stability per category
for cat_name in FACTUAL_GROUPS:
    best_L    = max(TEST_LAYERS, key=lambda L: stability_results[cat_name].get(L, {}).get("mean", -999))
    best_mean = stability_results[cat_name].get(best_L, {}).get("mean", 0)
    best_verdict = stability_results[cat_name].get(best_L, {}).get("verdict", "N/A")
    print(f"  {cat_name:>12}: best stability L{best_L} mean_cos={best_mean:.4f}  ({best_verdict})")

print()
overall_stab = np.mean([stability_results[cat].get(PRIMARY_LAYER, {}).get("mean", 0)
                        for cat in FACTUAL_GROUPS])
avg_improvement = float(np.mean(mrr_improvements)) if mrr_improvements else 0

print(f"  Overall factual direction stability at L{PRIMARY_LAYER}: {overall_stab:.4f}")
print(f"  Mean MRR improvement from factual axis: {avg_improvement:+.4f}")
print()
print(f"  VERDICT:")
if overall_stab > 0.5:
    print(f"  → STABLE: Factual match directions are consistent across instances")
    print(f"  → A universal 'factual correctness axis' exists at L{PRIMARY_LAYER}")
    print(f"  → The 22% oracle gap CAN potentially be closed with this axis")
elif overall_stab > 0.3:
    print(f"  → PARTIAL: Some factual direction consistency ({overall_stab:.3f})")
    print(f"  → Category-specific factual axes exist but not universal across categories")
else:
    print(f"  → VARIABLE: Factual match directions are content-specific ({overall_stab:.3f})")
    print(f"  → No universal 'factual correctness axis' — similar to Day 123 semantic axes")
    print(f"  → The 22% gap requires directed weight-matrix computation, not a static axis")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "stability_results": {
            cat: {str(L): v for L, v in d.items() if isinstance(v, dict)}
            for cat, d in stability_results.items()
        },
        "overall_stability_L25": float(overall_stab),
        "avg_mrr_improvement": float(avg_improvement),
        "primary_layer": PRIMARY_LAYER,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 128 complete.")
