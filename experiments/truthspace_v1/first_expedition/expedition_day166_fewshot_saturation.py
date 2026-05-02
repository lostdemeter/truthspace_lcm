#!/usr/bin/env python3
"""
Day 166 — Few-Shot Saturation Curve

Day 164 showed directions are domain-specific and work with ~4 examples.
QUESTION: How does accuracy grow as k (training examples) increases from 1 to N?

For each domain, run leave-one-out (LOO) evaluation at each k:
  k=1: direction from 1 pair, test on remaining
  k=2: direction from 2 pairs, test on remaining
  ...
  k=N-1: direction from N-1 pairs, test on held-out 1

Domains tested:
  capitals    (12 pairs total)
  antonyms    (12 pairs total)
  gender      (8 pairs total)
  metals      (6 pairs total)
  planets     (8 pairs total)
  colors_temp (9 pairs total)
  languages   (9 pairs total)

HYPOTHESIS A (fast saturation): 2-3 examples already defines direction precisely.
HYPOTHESIS B (slow growth): accuracy grows linearly up to k=6-8.
HYPOTHESIS C (domain differences): some domains need more examples than others
  due to internal cluster geometry.
"""
import json
from pathlib import Path
import numpy as np
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day166_fewshot_saturation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Full domain pools ────────────────────────────────────────────
DOMAINS = {
    "capitals": [
        ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),
        ("China","Beijing"),("Italy","Rome"),("Spain","Madrid"),
        ("Russia","Moscow"),("Greece","Athens"),("Poland","Warsaw"),
        ("Sweden","Stockholm"),("Brazil","Brasilia"),("Egypt","Cairo"),
    ],
    "antonyms": [
        ("hot","cold"),("big","small"),("fast","slow"),("dark","light"),
        ("good","bad"),("young","old"),("rich","poor"),("clean","dirty"),
        ("loud","quiet"),("strong","weak"),("early","late"),("easy","hard"),
    ],
    "gender": [
        ("king","queen"),("prince","princess"),("actor","actress"),
        ("son","daughter"),("father","mother"),("brother","sister"),
        ("man","woman"),("boy","girl"),
    ],
    "metals": [
        ("iron","metal"),("copper","metal"),("aluminum","metal"),
        ("tin","metal"),("zinc","metal"),("lead","metal"),
    ],
    "planets": [
        ("Mercury","rocky"),("Venus","rocky"),("Earth","rocky"),("Mars","rocky"),
        ("Jupiter","gas"),("Saturn","gas"),("Uranus","gas"),("Neptune","gas"),
    ],
    "colors_temp": [
        ("red","warm"),("orange","warm"),("yellow","warm"),
        ("blue","cool"),("green","cool"),("purple","cool"),
        ("white","neutral"),("black","neutral"),("gray","neutral"),
    ],
    "languages": [
        ("Germany","German"),("France","French"),("Spain","Spanish"),
        ("Japan","Japanese"),("Italy","Italian"),("Greece","Greek"),
        ("Poland","Polish"),("Sweden","Swedish"),("Russia","Russian"),
    ],
}

VOCAB = [
    # capitals
    "Paris","Berlin","Tokyo","Beijing","Rome","Madrid","Moscow","Athens",
    "Warsaw","Stockholm","Brasilia","Cairo","London","Dublin","Vienna","Oslo",
    "Lisbon","Budapest","Prague","Bucharest",
    # countries
    "France","Germany","Japan","China","Italy","Spain","Russia","Greece",
    "Poland","Sweden","Brazil","Egypt","England","Ireland","Austria","Norway",
    "Portugal","Hungary","Czech","Romania",
    # languages
    "French","German","Japanese","Chinese","Italian","Spanish","Russian",
    "Greek","Polish","Swedish","English","Arabic","Hindi","Korean","Turkish",
    # antonyms
    "hot","cold","big","small","fast","slow","dark","light","good","bad",
    "young","old","rich","poor","clean","dirty","loud","quiet","strong",
    "weak","early","late","easy","hard","warm","cool","bright","rough","smooth",
    # gender
    "king","queen","prince","princess","actor","actress","son","daughter",
    "father","mother","brother","sister","man","woman","boy","girl",
    "uncle","aunt","husband","wife","duke","duchess","hero","heroine",
    # metals
    "metal","iron","copper","gold","silver","aluminum","zinc","tin","lead",
    "steel","bronze","brass","platinum",
    # planets
    "Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
    "rocky","gas","solid","inner","outer","planet","moon","star","comet",
    # colors
    "red","blue","yellow","green","orange","purple","white","black","gray",
    "neutral","primary","secondary",
    # animals
    "animal","insect","bird","fish","mammal","cat","dog","horse","whale","eagle",
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
print(f"  H={W_E.shape[1]}\n")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

vocab_ok   = [w for w in dict.fromkeys(VOCAB) if tid(w)]
vocab_embs = {w: W_E[tid(w)] for w in vocab_ok}
print(f"Vocabulary: {len(vocab_ok)} single-token words\n")

def make_dir(pairs):
    ds = [normed(W_E[tid(b)] - W_E[tid(a)])
          for a, b in pairs if tid(a) and tid(b)]
    return normed(np.mean(ds, axis=0)) if ds else None

def entity_excl(src, direction, exclude):
    eid = tid(src)
    if eid is None: return None, 0.0
    e = W_E[eid].copy()
    if direction is not None: e = e + direction
    cands = [w for w in vocab_ok if w not in exclude]
    scores = {w: cosine(e, vocab_embs[w]) for w in cands}
    top1 = max(cands, key=lambda w: scores[w])
    return top1, scores[top1]

# ─── Saturation curve ─────────────────────────────────────────────
print("="*68)
print("Few-Shot Saturation Curves (LOO evaluation at each k)")
print("="*68)
print()

all_results = {}

for domain_name, all_pairs in DOMAINS.items():
    # Filter to single-token pairs only
    ok_pairs = [(a,b) for a,b in all_pairs if tid(a) and tid(b)]
    N = len(ok_pairs)
    if N < 3:
        print(f"  {domain_name}: only {N} valid pairs, skip")
        continue

    print(f"  {domain_name} ({N} valid pairs):")
    k_accuracies = {}

    for k in range(1, N):
        # Average over all C(N, k) subsets of size k as training (up to 50 subsets)
        subsets = list(itertools.combinations(range(N), k))
        if len(subsets) > 50:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(subsets), 50, replace=False)
            subsets = [subsets[i] for i in idx]

        total_correct = 0
        total_test    = 0
        for train_idx in subsets:
            train_pairs = [ok_pairs[i] for i in train_idx]
            test_pairs  = [ok_pairs[i] for i in range(N) if i not in train_idx]
            d = make_dir(train_pairs)
            for src, tgt in test_pairs:
                pred, _ = entity_excl(src, d, {src})
                if pred == tgt: total_correct += 1
                total_test += 1

        acc = total_correct / total_test if total_test else 0
        k_accuracies[k] = round(acc, 3)
        print(f"    k={k}: {total_correct}/{total_test} = {acc:.3f}")

    # Also test k=0 (no direction)
    total_correct_0 = 0
    for src, tgt in ok_pairs:
        pred, _ = entity_excl(src, None, {src})
        if pred == tgt: total_correct_0 += 1
    k0 = round(total_correct_0 / N, 3)
    k_accuracies[0] = k0
    print(f"    k=0 (no dir): {total_correct_0}/{N} = {k0:.3f}")

    # Find saturation point (within 5% of max)
    max_acc = max(k_accuracies.values())
    sat_k = min(k for k, a in k_accuracies.items() if a >= max_acc - 0.05)
    print(f"    Saturation at k={sat_k} (max={max_acc:.3f})")
    print()

    all_results[domain_name] = {
        "n_valid": N, "k_accuracies": k_accuracies,
        "k0": k0, "max_acc": max_acc, "saturation_k": sat_k
    }

# ─── Summary table ────────────────────────────────────────────────
print("="*68)
print("Summary: Accuracy by k across domains")
print("="*68)
print()

max_k = max(r["n_valid"] - 1 for r in all_results.values())
header = f"{'domain':>14} | {'k=0':>5} " + "".join(f"| {'k='+str(k):>5} " for k in range(1, min(max_k+1, 9)))
print("  " + header)
print("  " + "-"*len(header))
for dname, r in all_results.items():
    row = f"  {dname:>14} | {r['k0']:>5.3f} "
    for k in range(1, min(max_k+1, 9)):
        if k in r["k_accuracies"]:
            row += f"| {r['k_accuracies'][k]:>5.3f} "
        else:
            row += f"|  {'—':>4} "
    row += f"  → sat.k={r['saturation_k']}"
    print(row)

print()
print("Key findings:")
for dname, r in all_results.items():
    k1 = r["k_accuracies"].get(1, 0)
    k2 = r["k_accuracies"].get(2, 0)
    delta = r["max_acc"] - r["k0"]
    print(f"  {dname:>14}: k=0→{r['k0']:.2f}, k=1→{k1:.2f}, k=2→{k2:.2f}, "
          f"max={r['max_acc']:.2f} (Δ={delta:+.2f}), sat.k={r['saturation_k']}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 166 complete.")
