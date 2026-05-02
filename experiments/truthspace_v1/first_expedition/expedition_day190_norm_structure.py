#!/usr/bin/env python3
"""
Day 190 — W_E Norm Structure

HYPOTHESIS: The L2 norm of W_E embeddings encodes token 'importance' or
frequency. High-frequency tokens may have larger or smaller norms.
Norm may correlate with semantic category (function words vs content words).

EXPERIMENTS:
  1. Norm distribution across all 151k vocab tokens
     - Mean, std, percentiles
     - Shape of distribution (normal? log-normal? bimodal?)

  2. Norm vs token type
     - Punctuation / special tokens
     - Common function words (the, is, of, ...)
     - Common content words (house, run, blue, ...)
     - Single-letter tokens
     - Digit tokens (0-9)
     - Proper nouns (France, Paris, ...)
     - Sub-word tokens (Ġsyll, ilia, ...)

  3. Norm of domain pairs (TYPE_BC relations)
     - Do source and target words have similar norms?
     - Does norm difference correlate with retrieval accuracy?

  4. Norm across relation types
     - capitals vs antonyms vs gender vs thematic
     - Is norm a discriminator between encoding types?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day190_norm_structure.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

FUNCTION_WORDS = ["the","is","of","a","in","to","and","it","that","this",
                  "be","was","for","on","are","but","not","they","with","he",
                  "at","his","by","from","we","or","an","will","my","one"]

CONTENT_WORDS = ["house","run","blue","large","mountain","water","stone",
                 "time","world","play","book","light","hand","face","tree",
                 "road","river","fire","gold","bird","fish","ship","door",
                 "sleep","dream","city","music","dance","color","cloud"]

DIGITS = [str(i) for i in range(10)]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")

DOMAIN_PAIRS = {
    "capitals": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                 ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                 ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm")],
    "antonyms": [("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
                 ("light","dark"),("old","young"),("loud","quiet")],
    "gender":   [("king","queen"),("man","woman"),("boy","girl"),
                 ("prince","princess"),("actor","actress")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

# Compute norms for all tokens
all_norms = np.linalg.norm(W_E, axis=1)

# ── Experiment 1: Norm distribution ─────────────────────────────────
print("Experiment 1: Full norm distribution")
print("-" * 60)
pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
pct_vals = np.percentile(all_norms, pcts)
print(f"  Mean:    {all_norms.mean():.4f}")
print(f"  Std:     {all_norms.std():.4f}")
print(f"  Min:     {all_norms.min():.4f}")
print(f"  Max:     {all_norms.max():.4f}")
print(f"  Percentiles:")
for p, v in zip(pcts, pct_vals):
    print(f"    p{p:>3}: {v:.4f}")
print()

# Histogram buckets
buckets = [0, 20, 40, 60, 80, 100, 120, 140, 160, 200, 300, 10000]
counts = np.histogram(all_norms, bins=buckets)[0]
print("  Norm histogram:")
for i in range(len(counts)):
    pct = 100*counts[i]/V
    bar = "#" * int(pct/2)
    print(f"    [{buckets[i]:>5}-{buckets[i+1]:>5}]: {counts[i]:>7} ({pct:5.1f}%) {bar}")
print()

# ── Experiment 2: Norm by token type ────────────────────────────────
def tid1(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def tid_raw(word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def norm_of(word, space_prefix=True):
    fn = tid1 if space_prefix else tid_raw
    t = fn(word)
    return float(all_norms[t]) if t is not None else None

print("Experiment 2: Norm by token type")
print("-" * 60)

categories = {
    "function_words": FUNCTION_WORDS,
    "content_words":  CONTENT_WORDS,
    "digits":         DIGITS,
    "letters":        LETTERS,
}

cat_norms = {}
for cat, words in categories.items():
    norms = []
    for w in words:
        n = norm_of(w, space_prefix=(cat in ("function_words","content_words")))
        if n: norms.append(n)
    if norms:
        cat_norms[cat] = {"mean": float(np.mean(norms)), "std": float(np.std(norms)),
                          "n": len(norms)}
        print(f"  {cat:<20}: mean={np.mean(norms):.2f}  std={np.std(norms):.2f}  (n={len(norms)})")

print()

# Special tokens
special_ids = list(range(10))  # first 10 tokens usually special
print("  First 10 vocabulary tokens and their norms:")
for i in special_ids:
    tok_str = tok.convert_ids_to_tokens([i])[0]
    print(f"    id={i:>5}  tok={tok_str:<20}  norm={all_norms[i]:.4f}")
print()

# ── Experiment 3: Norm of domain pairs ──────────────────────────────
print("Experiment 3: Norm of domain pair words")
print("-" * 60)
print(f"  {'Domain':>12}  {'word':>15}  {'role':>6}  norm")
print("  " + "-"*50)

domain_norm_results = {}
for domain, pairs in DOMAIN_PAIRS.items():
    src_norms, tgt_norms = [], []
    for a, b in pairs:
        na = norm_of(a)
        nb = norm_of(b)
        if na and nb:
            src_norms.append(na)
            tgt_norms.append(nb)
            print(f"  {domain:>12}  {a:>15} (src)  {na:.2f}")
            print(f"  {domain:>12}  {b:>15} (tgt)  {nb:.2f}")
    if src_norms:
        norm_diff = np.mean(np.abs(np.array(src_norms) - np.array(tgt_norms)))
        print(f"  {'':>12}  {'mean |src-tgt|':>15}        {norm_diff:.4f}")
        domain_norm_results[domain] = {
            "src_mean": float(np.mean(src_norms)),
            "tgt_mean": float(np.mean(tgt_norms)),
            "src_std":  float(np.std(src_norms)),
            "tgt_std":  float(np.std(tgt_norms)),
            "mean_abs_diff": float(norm_diff),
        }
    print()

# ── Experiment 4: Does norm predict retrieval accuracy? ─────────────
print("Experiment 4: Norm vs direction retrieval accuracy")
print("-" * 60)
# For capitals, compute per-pair LOO accuracy vs norm difference
pairs = DOMAIN_PAIRS["capitals"]
ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
tgt_vocab = {b: W_E[tid1(b)] for _, b in ok}

per_pair_results = []
for i, (a, b) in enumerate(ok):
    loo = [normed(W_E[tid1(bb)] - W_E[tid1(aa)]) for aa, bb in ok if aa != a]
    if not loo: continue
    d = normed(np.mean(loo, axis=0))
    q = W_E[tid1(a)] + d
    cands = {w: float(np.dot(normed(q), normed(tgt_vocab[w])))
             for w in tgt_vocab if w != a}
    correct = max(cands, key=lambda w: cands[w]) == b
    norm_a = float(all_norms[tid1(a)])
    norm_b = float(all_norms[tid1(b)])
    per_pair_results.append({"pair": (a, b), "correct": correct,
                             "norm_a": norm_a, "norm_b": norm_b,
                             "norm_diff": abs(norm_a - norm_b)})
    print(f"  {a:>10}→{b:<10}  norm_src={norm_a:.1f}  norm_tgt={norm_b:.1f}  "
          f"diff={abs(norm_a-norm_b):.1f}  {'✓' if correct else '✗'}")

correct_norms = [r["norm_diff"] for r in per_pair_results if r["correct"]]
wrong_norms   = [r["norm_diff"] for r in per_pair_results if not r["correct"]]
if correct_norms and wrong_norms:
    print(f"\n  Mean |norm_diff| for CORRECT: {np.mean(correct_norms):.2f}")
    print(f"  Mean |norm_diff| for WRONG:   {np.mean(wrong_norms):.2f}")

results = {
    "norm_distribution": {
        "mean": float(all_norms.mean()),
        "std":  float(all_norms.std()),
        "min":  float(all_norms.min()),
        "max":  float(all_norms.max()),
        "percentiles": {str(p): float(v) for p, v in zip(pcts, pct_vals)},
    },
    "category_norms": cat_norms,
    "domain_norms": domain_norm_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 190 complete.")
