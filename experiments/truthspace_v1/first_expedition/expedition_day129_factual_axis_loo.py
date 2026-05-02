#!/usr/bin/env python3
"""
Day 129 — Factual Axis LOO Generalization Test

Day 128 found: mean factual axis (Δ = h(correct) - h(wrong_pool_mean))
achieves MRR=0.806-1.000 on the SAME pairs used to compute it.

QUESTION: Does the factual axis generalize to HELD-OUT pairs?
  Leave-One-Out (LOO) test:
    For each pair i:
      axis_LOO_i = mean(Δ_j for j ≠ i)  [exclude pair i]
      rank pair i using axis_LOO_i
  LOO MRR = average MRR across all held-out pairs

If LOO MRR ≈ in-sample MRR → axis generalizes → true geometric property
If LOO MRR ≈ random (0.314) → axis overfits → only works on training pairs

Also test:
  1. Per-category LOO MRR at L25
  2. LOO MRR vs number of training pairs (1, 2, ..., N-1)
  3. Compare: factual axis LOO vs T2 axis vs full cosine L25 vs log-prob oracle
  4. Whether the factual axis at L25 TRANSFERS across categories
     (axis learned from capitals applied to languages — does it generalize?)

LAYERS: Primary test at L25 (Day 127 optimal), also verify at L23.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day129_factual_axis_loo.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PRIMARY_LAYER = 25
ALSO_LAYERS   = [23, 27]

FACTUAL_GROUPS = {
    "capitals": [
        ("Paris",      ["London","Rome","Berlin","Madrid"],       "The capital city of France is"),
        ("Tokyo",      ["Osaka","Beijing","Seoul","Bangkok"],      "The capital city of Japan is"),
        ("Berlin",     ["Frankfurt","Vienna","Warsaw","Amsterdam"],"The capital city of Germany is"),
        ("Madrid",     ["Lisbon","Rome","Paris","Brussels"],       "The capital city of Spain is"),
        ("Rome",       ["Milan","Paris","Vienna","Athens"],        "The capital city of Italy is"),
        ("Moscow",     ["Petersburg","Kiev","Minsk","Warsaw"],     "The capital city of Russia is"),
    ],
    "languages": [
        ("Portuguese", ["Spanish","Italian","French","English"],  "The official language of Brazil is"),
        ("Arabic",     ["Hebrew","Turkish","Persian","Urdu"],     "The official language of Egypt is"),
        ("Mandarin",   ["Japanese","Korean","Cantonese","Thai"],  "The official language of China is"),
        ("Hindi",      ["Urdu","Bengali","Tamil","Punjabi"],      "The official language of India is"),
    ],
    "antonyms": [
        ("cold",    ["warm","lukewarm","mild","chilly"],    "The opposite of hot is"),
        ("sad",     ["angry","bored","worried","tired"],    "The opposite of happy is"),
        ("dark",    ["bright","sunny","light","clear"],     "The opposite of bright is"),
        ("slow",    ["fast","quick","rapid","swift"],       "The opposite of fast is"),
        ("small",   ["large","big","huge","enormous"],      "The opposite of large is"),
        ("wrong",   ["right","correct","accurate","true"],  "The opposite of right is"),
    ],
    "hypernyms": [
        ("dog",     ["cat","rabbit","horse","pet"],          "A poodle is a type of"),
        ("flower",  ["tree","bush","grass","weed"],          "A rose is a type of"),
        ("tool",    ["machine","device","appliance","weapon"],"A hammer is a type of"),
        ("bird",    ["insect","reptile","fish","mammal"],    "An eagle is a type of"),
        ("gem",     ["metal","mineral","crystal","stone"],   "A ruby is a type of"),
    ],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

all_layers = sorted(set([PRIMARY_LAYER] + ALSO_LAYERS))

def get_hs_layers(text, layers):
    inp = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}
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

# ── Pre-compute all hidden states ──────────────────────────────────────────────
print("Pre-computing hidden states for all words ...")
word_hs = {}  # word -> {L: h}
all_words = set()
for cat_pairs in FACTUAL_GROUPS.values():
    for correct, wrong_pool, prompt in cat_pairs:
        all_words.add(correct)
        all_words.update(wrong_pool)
for w in sorted(all_words):
    word_hs[w] = get_hs_layers(" " + w, all_layers)
print(f"  Cached {len(all_words)} words.\n")

# ── LOO experiment ─────────────────────────────────────────────────────────────
print("=" * 72)
print(f"LOO Factual Axis Test (primary layer L{PRIMARY_LAYER})")
print("=" * 72)
print()

loo_results = {}
all_loo_mrrs = {L: [] for L in all_layers}
all_insample_mrrs = {L: [] for L in all_layers}
all_fullcos_mrrs = {L: [] for L in all_layers}
all_logprob_mrrs = []

for cat_name, cat_pairs in FACTUAL_GROUPS.items():
    N = len(cat_pairs)
    loo_results[cat_name] = {"pairs": [], "loo_mrrs": {L: [] for L in all_layers},
                              "insample_mrrs": {L: [] for L in all_layers}}

    print(f"  [{cat_name}]  N={N} pairs")
    for L in all_layers:
        loo_mrrs = []
        insample_mrrs = []

        # Compute all Δ vectors at this layer
        deltas = []
        for correct, wrong_pool, _ in cat_pairs:
            h_cor = word_hs[correct][L]
            h_wrg = np.mean([word_hs[w][L] for w in wrong_pool], axis=0).astype(np.float32)
            d = h_cor - h_wrg
            nv = np.linalg.norm(d)
            deltas.append((d/nv if nv > 1e-6 else np.zeros_like(d)).astype(np.float32))

        # Full in-sample axis (all N pairs)
        axis_full = normed(np.mean(deltas, axis=0).astype(np.float32))

        for i, (correct, wrong_pool, prompt) in enumerate(cat_pairs):
            all_cands = [correct] + wrong_pool

            # LOO axis: exclude pair i
            loo_deltas = [deltas[j] for j in range(N) if j != i]
            axis_loo   = normed(np.mean(loo_deltas, axis=0).astype(np.float32))

            # Rank by LOO axis projection
            projs_loo  = {w: float(np.dot(normed(word_hs[w][L]), axis_loo)) for w in all_cands}
            ranked_loo = sorted(all_cands, key=lambda w: -projs_loo[w])
            rank_loo   = next((j+1 for j,w in enumerate(ranked_loo) if w == correct), N+1)

            # Rank by in-sample axis
            projs_is   = {w: float(np.dot(normed(word_hs[w][L]), axis_full)) for w in all_cands}
            ranked_is  = sorted(all_cands, key=lambda w: -projs_is[w])
            rank_is    = next((j+1 for j,w in enumerate(ranked_is) if w == correct), N+1)

            loo_mrrs.append(1.0/rank_loo)
            insample_mrrs.append(1.0/rank_is)

        loo_mean = float(np.mean(loo_mrrs))
        is_mean  = float(np.mean(insample_mrrs))
        loo_results[cat_name]["loo_mrrs"][L] = loo_mrrs
        loo_results[cat_name]["insample_mrrs"][L] = insample_mrrs
        all_loo_mrrs[L].append(loo_mean)
        all_insample_mrrs[L].append(is_mean)

        if L == PRIMARY_LAYER:
            print(f"    L{L}: LOO MRR={loo_mean:.4f}  in-sample={is_mean:.4f}  "
                  f"{'✓ GENERALIZES' if loo_mean > 0.5 else '~ partial' if loo_mean > 0.35 else '✗ overfits'}")

    # Full cosine at primary layer
    for i, (correct, wrong_pool, prompt) in enumerate(cat_pairs):
        all_cands = [correct] + wrong_pool
        ctx_h = get_hs_layers(prompt, [PRIMARY_LAYER])[PRIMARY_LAYER]
        sims  = {w: cosine(ctx_h, word_hs[w][PRIMARY_LAYER]) for w in all_cands}
        ranked = sorted(all_cands, key=lambda w: -sims[w])
        rank   = next((j+1 for j,w in enumerate(ranked) if w == correct), len(all_cands)+1)
        all_fullcos_mrrs[PRIMARY_LAYER].append(1.0/rank)

        # Log-prob
        lp_scores = {w: get_logprob(prompt, w) for w in all_cands}
        ranked_lp = sorted(all_cands, key=lambda w: -lp_scores[w])
        rank_lp   = next((j+1 for j,w in enumerate(ranked_lp) if w == correct), len(all_cands)+1)
        all_logprob_mrrs.append(1.0/rank_lp)
    print()

# ── LOO vs training set size ────────────────────────────────────────────────────
print("=" * 72)
print("LOO MRR vs number of training pairs (L25)")
print("=" * 72)
print()

# For each category with N>=4 pairs, test LOO with k=1,2,...,N-1 training pairs
for cat_name, cat_pairs in FACTUAL_GROUPS.items():
    N = len(cat_pairs)
    if N < 4: continue
    L = PRIMARY_LAYER
    print(f"  [{cat_name}] N={N}")

    deltas = []
    for correct, wrong_pool, _ in cat_pairs:
        h_cor = word_hs[correct][L]
        h_wrg = np.mean([word_hs[w][L] for w in wrong_pool], axis=0).astype(np.float32)
        d = h_cor - h_wrg
        nv = np.linalg.norm(d)
        deltas.append((d/nv if nv > 1e-6 else np.zeros_like(d)).astype(np.float32))

    print(f"  {'k_train':>8}  {'k_test':>8}  {'mean_LOO_MRR':>14}")
    print(f"  {'-'*36}")
    for k_train in range(1, N):
        # Use first k_train pairs to build axis, test on remaining N-k_train
        axis_k = normed(np.mean(deltas[:k_train], axis=0).astype(np.float32))
        test_mrrs = []
        for i in range(k_train, N):
            correct, wrong_pool, _ = cat_pairs[i]
            all_cands = [correct] + wrong_pool
            projs = {w: float(np.dot(normed(word_hs[w][L]), axis_k)) for w in all_cands}
            ranked = sorted(all_cands, key=lambda w: -projs[w])
            rank   = next((j+1 for j,w in enumerate(ranked) if w == correct), len(all_cands)+1)
            test_mrrs.append(1.0/rank)
        print(f"  {k_train:>8}  {N-k_train:>8}  {float(np.mean(test_mrrs)):>14.4f}")
    print()

# ── Cross-category transfer ────────────────────────────────────────────────────
print("=" * 72)
print("Cross-category transfer: does the axis from category A work on category B?")
print("=" * 72)
print()

L = PRIMARY_LAYER
cat_axes = {}
for cat_name, cat_pairs in FACTUAL_GROUPS.items():
    deltas = []
    for correct, wrong_pool, _ in cat_pairs:
        h_cor = word_hs[correct][L]
        h_wrg = np.mean([word_hs[w][L] for w in wrong_pool], axis=0).astype(np.float32)
        d = h_cor - h_wrg
        nv = np.linalg.norm(d)
        if nv > 1e-6: deltas.append(d/nv)
    if deltas: cat_axes[cat_name] = normed(np.mean(deltas, axis=0).astype(np.float32))

cats = list(cat_axes.keys())
print(f"  {'axis→':>12}  " + "  ".join(f"{'→'+c:>12}" for c in cats))
print(f"  {'-'*(14 + 14*len(cats))}")
transfer_matrix = {}
for src_cat in cats:
    row = f"  {src_cat:>12}  "
    transfer_matrix[src_cat] = {}
    for tgt_cat in cats:
        axis = cat_axes[src_cat]
        test_pairs = FACTUAL_GROUPS[tgt_cat]
        mrrs = []
        for correct, wrong_pool, _ in test_pairs:
            all_cands = [correct] + wrong_pool
            projs  = {w: float(np.dot(normed(word_hs[w][L]), axis)) for w in all_cands}
            ranked = sorted(all_cands, key=lambda w: -projs[w])
            rank   = next((j+1 for j,w in enumerate(ranked) if w == correct), len(all_cands)+1)
            mrrs.append(1.0/rank)
        m = float(np.mean(mrrs))
        transfer_matrix[src_cat][tgt_cat] = m
        marker = " *" if src_cat == tgt_cat else "  "
        row += f"  {m:>10.4f}{marker}"
    print(row)
print()
print("  * = in-sample (diagonal)")

# ── Aggregate summary ───────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 129 Summary — Factual Axis Generalization")
print("=" * 72)

n_total = sum(len(p) for p in FACTUAL_GROUPS.values())
mean_loo     = float(np.mean([v for vv in all_loo_mrrs[PRIMARY_LAYER] for v in [vv]]))
mean_is      = float(np.mean([v for vv in all_insample_mrrs[PRIMARY_LAYER] for v in [vv]]))
mean_fullcos = float(np.mean(all_fullcos_mrrs[PRIMARY_LAYER]))
mean_logprob = float(np.mean(all_logprob_mrrs))
random_mrr   = float(np.mean([1/(1+4) for _ in range(n_total)]))  # approx: 1 correct + 4 wrong

# Per-category summary at PRIMARY_LAYER
print(f"""
  Method                  MRR      vs_random
  ─────────────────────────────────────────────
  log-prob (oracle)      {mean_logprob:.4f}    (+{mean_logprob-random_mrr:.4f})
  full cos L{PRIMARY_LAYER} (Day127)  {mean_fullcos:.4f}    (+{mean_fullcos-random_mrr:.4f})
  factual axis in-sample {mean_is:.4f}    (+{mean_is-random_mrr:.4f})
  factual axis LOO       {mean_loo:.4f}    (+{mean_loo-random_mrr:.4f})
  Random baseline        {random_mrr:.4f}    ---
""")

print(f"  Per-category LOO MRR at L{PRIMARY_LAYER}:")
for cat_name in FACTUAL_GROUPS:
    loo_m = float(np.mean(loo_results[cat_name]["loo_mrrs"][PRIMARY_LAYER]))
    is_m  = float(np.mean(loo_results[cat_name]["insample_mrrs"][PRIMARY_LAYER]))
    gap   = loo_m - is_m
    print(f"    {cat_name:>12}: LOO={loo_m:.4f}  in-sample={is_m:.4f}  gap={gap:+.4f}  "
          f"{'✓' if loo_m > 0.5 else '~' if loo_m > 0.35 else '✗'}")

print()
print("  Cross-category transfer (diagonal=in-sample, off-diagonal=transfer):")
for src_cat in cats:
    on_diag   = transfer_matrix[src_cat][src_cat]
    off_diags = [transfer_matrix[src_cat][t] for t in cats if t != src_cat]
    print(f"    {src_cat:>12}: diag={on_diag:.4f}  "
          f"off-diag_mean={np.mean(off_diags):.4f}  "
          f"{'TRANSFERS' if np.mean(off_diags) > 0.45 else 'does not transfer'}")

print()
generalization = mean_loo > 0.5
transfer_test  = np.mean([transfer_matrix[s][t]
                          for s in cats for t in cats if s != t]) > 0.45
print(f"  VERDICT:")
print(f"  Factual axis generalizes to LOO: {'YES ✓' if generalization else 'NO ✗'}")
print(f"  Factual axis transfers across categories: {'YES ✓' if transfer_test else 'NO ✗'}")
if generalization:
    print(f"  → The factual axis IS a true geometric property of L{PRIMARY_LAYER} space")
    print(f"  → Category-specific axes can rank unseen pairs within the same category")
else:
    print(f"  → The factual axis overfits to training examples (low LOO generalization)")
    print(f"  → Ranking within a category requires seeing examples from that category")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "loo_results": {
            cat: {
                "loo_mrrs": {str(L): v for L, v in d["loo_mrrs"].items()},
                "insample_mrrs": {str(L): v for L, v in d["insample_mrrs"].items()},
            } for cat, d in loo_results.items()
        },
        "transfer_matrix": transfer_matrix,
        "summary": {
            "mean_loo_mrr": mean_loo,
            "mean_insample_mrr": mean_is,
            "mean_fullcos_mrr": mean_fullcos,
            "mean_logprob_mrr": mean_logprob,
            "random_mrr": random_mrr,
        }
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 129 complete.")
