#!/usr/bin/env python3
"""
Day 200 — k-Shot Accuracy Scaling

QUESTION: How does TYPE_BC LOO retrieval accuracy grow with k (number of
known training pairs used to estimate the mean direction)?

For each domain, we run LOO at k=1,2,3,4,5,6,8,10,ALL:
  - Subsample k pairs from the known set (leave one out as query)
  - Estimate mean direction from those k pairs
  - Retrieve target for the held-out source
  - Average over all leave-one-out positions and 20 random subsamples per k

EXPECTED FINDINGS:
  - k=1: accuracy depends entirely on which pair is chosen (high variance)
  - k=2: some averaging begins
  - k~5: approaching plateau?
  - k=ALL: previously measured ground truth

DOMAINS: capitals, gender, past_tense, superlative, plurals
ALSO: TYPE_ADJACENT (antonyms) — does it benefit from k at all?

OUTPUT: accuracy ± std per (domain, k)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day200_kshot_scaling.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
N_RESAMPLES = 30  # random subsamples per k

DOMAINS = {
    "capitals":    [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                    ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                    ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
                    ("Korea","Seoul"),("Poland","Warsaw"),("Turkey","Ankara")],
    "gender":      [("king","queen"),("man","woman"),("boy","girl"),
                    ("prince","princess"),("actor","actress"),("hero","heroine"),
                    ("waiter","waitress"),("duke","duchess")],
    "antonyms":    [("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
                    ("light","dark"),("old","young"),("loud","quiet"),
                    ("sharp","dull"),("rich","poor"),("thick","thin")],
    "past_tense":  [("run","ran"),("eat","ate"),("go","went"),("see","saw"),
                    ("come","came"),("give","gave"),("take","took"),
                    ("make","made"),("say","said"),("know","knew")],
    "superlative": [("big","biggest"),("fast","fastest"),("old","oldest"),
                    ("cold","coldest"),("smart","smartest"),("long","longest"),
                    ("hard","hardest"),("dark","darkest")],
    "plurals":     [("cat","cats"),("dog","dogs"),("house","houses"),
                    ("tree","trees"),("book","books"),("car","cars"),
                    ("bird","birds"),("ship","ships"),("hand","hands"),
                    ("eye","eyes"),("road","roads"),("door","doors")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a,b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                      normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def tid1(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_emb(word):
    t = tid1(word)
    return W_E[t].astype(np.float64) if t is not None else None

rng = np.random.default_rng(42)

def kshot_accuracy_bc(pairs, k, n_resamples=N_RESAMPLES):
    """
    For each held-out query (src, tgt), sample k pairs from the remaining
    n-1 known pairs, estimate mean direction, retrieve, check correctness.
    Return mean accuracy ± std over all (query, resample) combinations.
    """
    ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
    if len(ok) < k + 1: return None, None
    target_set = {b: get_emb(b) for _, b in ok}
    all_accs = []
    for i, (qa, qb) in enumerate(ok):
        pool = [(a, b) for a, b in ok if a != qa]  # n-1 pairs
        if len(pool) < k: continue
        for _ in range(n_resamples):
            chosen = pool if k >= len(pool) else [pool[j] for j in
                     rng.choice(len(pool), k, replace=False)]
            diffs = [normed(get_emb(b) - get_emb(a)) for a, b in chosen]
            mean_dir = normed(np.mean(diffs, axis=0))
            query = get_emb(qa) + mean_dir
            sims = {w: cosine(query, e) for w, e in target_set.items()
                    if w != qa}
            pred = max(sims, key=lambda w: sims[w])
            all_accs.append(1.0 if pred == qb else 0.0)
    if not all_accs: return None, None
    return float(np.mean(all_accs)), float(np.std(all_accs))

def kshot_accuracy_adjacent(pairs, n_resamples=1):
    """TYPE_ADJACENT: no direction needed, just nearest neighbour."""
    ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
    if len(ok) < 2: return None, None
    target_set = {b: get_emb(b) for _, b in ok}
    accs = []
    for qa, qb in ok:
        se = get_emb(qa)
        sims = {w: cosine(se, e) for w, e in target_set.items() if w != qa}
        pred = max(sims, key=lambda w: sims[w])
        accs.append(1.0 if pred == qb else 0.0)
    return float(np.mean(accs)), float(np.std(accs))

# ── Run k-shot scaling ────────────────────────────────────────────────
results = {}
k_values = [1, 2, 3, 4, 5, 6, 8, 10, 999]  # 999 = use all

print(f"{'Domain':<14}  " + "  ".join(f"k={k:<4}" for k in k_values))
print("-" * 90)

for domain, pairs in DOMAINS.items():
    ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
    n = len(ok)
    row_accs = []
    row_stds = []
    domain_results = {"n": n, "k_results": {}}

    for k in k_values:
        actual_k = min(k, n - 1)
        if domain == "antonyms":
            acc, std = kshot_accuracy_adjacent(ok)
        else:
            acc, std = kshot_accuracy_bc(ok, actual_k)
        row_accs.append(acc)
        row_stds.append(std)
        domain_results["k_results"][str(actual_k)] = {
            "accuracy": acc, "std": std
        }

    results[domain] = domain_results
    acc_str = "  ".join(
        f"{a:.3f}" if a is not None else " None" for a in row_accs
    )
    print(f"  {domain:<14}  {acc_str}")

# ── Print std row ─────────────────────────────────────────────────────
print()
print("Standard deviations:")
print(f"{'Domain':<14}  " + "  ".join(f"k={k:<4}" for k in k_values))
print("-" * 90)
for domain in DOMAINS:
    stds = [results[domain]["k_results"][str(min(k, results[domain]["n"]-1))]["std"]
            for k in k_values]
    std_str = "  ".join(
        f"{s:.3f}" if s is not None else " None" for s in stds
    )
    print(f"  {domain:<14}  {std_str}")

# ── Saturation analysis ───────────────────────────────────────────────
print()
print("Saturation analysis (k where 90% of max accuracy is reached):")
print("-" * 60)
for domain, dr in results.items():
    kr = dr["k_results"]
    valid = [(int(k), v["accuracy"]) for k, v in kr.items()
             if v["accuracy"] is not None]
    if not valid: continue
    valid.sort()
    max_acc = max(a for _, a in valid)
    threshold = 0.90 * max_acc
    sat_k = next((k for k, a in valid if a >= threshold), valid[-1][0])
    all_k_str = "  ".join(f"k{k}={a:.3f}" for k, a in valid)
    print(f"  {domain:<14}  sat_k={sat_k}  max={max_acc:.3f}  [{all_k_str}]")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=lambda x: None if x is None else float(x))
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 200 complete.")
