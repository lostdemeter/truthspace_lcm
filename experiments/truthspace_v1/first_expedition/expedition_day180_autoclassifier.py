#!/usr/bin/env python3
"""
Day 180 — Auto-Classifier Validation on Unseen Domains

DC 356 derived a decision tree to classify relational encoding type
from three geometric signals measured with k=3 training pairs:
  H1: direction consistency (inter-pair cosine of diff vectors)
  H2: target compactness (mean pairwise cosine of target embeddings)
  n:  number of unique target words

Classification rules:
  H1 < 0.10                       → TYPE_A  (proximity / absent)
  H1 ≥ 0.10, H2 < 0.15           → THEMATIC (not in W_E)
  H1 ≥ 0.10, H2 ≥ 0.35, n ≤ 3   → TYPE_DE  (multi-pole routing)
  H1 ≥ 0.20, H2 ≥ 0.25, n > 3   → TYPE_BC  (direction retrieval)
  else                            → UNKNOWN

VALIDATION: Apply classifier to 6 NEW domains never seen before.
Then run empirical LOO accuracy and check if predicted type matches observed.

NEW DOMAINS:
  1. countries → continent          (expected: TYPE_D, multi-pole: 6 continents)
  2. animal → size_class            (expected: TYPE_DE, 2-3 poles: large/small)
  3. sport → indoor_outdoor         (expected: TYPE_DE, 2 poles)
  4. element_symbol → state         (expected: THEMATIC or TYPE_DE)
  5. word → syllable_count          (expected: THEMATIC, no direction)
  6. country → currency_name        (expected: TYPE_BC or C)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day180_autoclassifier.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

NEW_DOMAINS = {
    "country_continent": [
        ("France","Europe"),("Germany","Europe"),("Italy","Europe"),
        ("Spain","Europe"),("Japan","Asia"),("China","Asia"),
        ("India","Asia"),("Russia","Europe"),("Brazil","America"),
        ("Mexico","America"),("Canada","America"),("Egypt","Africa"),
        ("Kenya","Africa"),("Nigeria","Africa"),("Australia","Australia"),
        ("Korea","Asia"),("Turkey","Asia"),("Greece","Europe"),
    ],
    "animal_size": [
        ("elephant","large"),("whale","large"),("lion","large"),
        ("horse","large"),("bear","large"),("cat","small"),
        ("mouse","small"),("rabbit","small"),("hamster","small"),
        ("sparrow","small"),("dog","medium"),("sheep","medium"),
        ("pig","medium"),("deer","medium"),
    ],
    "sport_venue": [
        ("tennis","court"),("basketball","court"),("swimming","pool"),
        ("boxing","ring"),("hockey","rink"),("football","field"),
        ("baseball","field"),("golf","course"),("bowling","lane"),
        ("gymnastics","gym"),
    ],
    "country_currency": [
        ("France","euro"),("Germany","euro"),("Italy","euro"),
        ("Spain","euro"),("Japan","yen"),("China","yuan"),
        ("Russia","ruble"),("Brazil","real"),("India","rupee"),
        ("Korea","won"),("Sweden","krona"),("Poland","zloty"),
    ],
    "word_category": [
        ("Paris","noun"),("London","noun"),("run","verb"),
        ("jump","verb"),("fast","adjective"),("slow","adjective"),
        ("quickly","adverb"),("softly","adverb"),
        ("iron","noun"),("gold","noun"),("swim","verb"),("grow","verb"),
    ],
    "number_magnitude": [
        ("one","small"),("two","small"),("three","small"),
        ("hundred","large"),("million","large"),("billion","large"),
        ("ten","medium"),("fifty","medium"),("twenty","medium"),
    ],
}

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

def compute_signals(pairs):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    if len(ok) < 3: return None
    diffs = [normed(W_E[tid(b)] - W_E[tid(a)]) for a, b in ok]
    tgts  = [normed(W_E[tid(b)]) for a, b in ok]
    n = len(diffs)
    cos_pairs = [cosine(diffs[i], diffs[j]) for i in range(n) for j in range(i+1, n)]
    tgt_pairs = [cosine(tgts[i], tgts[j])   for i in range(n) for j in range(i+1, n)]
    h1 = float(np.mean(cos_pairs)) if cos_pairs else 0.0
    h2 = float(np.mean(tgt_pairs)) if tgt_pairs else 0.0
    n_unique = len(set(b for _, b in ok))
    return {"h1": h1, "h2": h2, "n_unique": n_unique, "n_pairs": len(ok)}

def classify(h1, h2, n_unique):
    if h1 < 0.10:
        return "TYPE_A"
    if h2 < 0.15:
        return "THEMATIC"
    if h2 >= 0.35 and n_unique <= 4:
        return "TYPE_DE"
    if h1 >= 0.20 and h2 >= 0.25 and n_unique > 4:
        return "TYPE_BC"
    return "UNKNOWN"

def loo_accuracy(pairs, tgt_vocab):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b) and b in tgt_vocab]
    nc = 0
    for a, b in ok:
        loo_diffs = [normed(W_E[tid(bb)] - W_E[tid(aa)])
                     for aa, bb in ok if aa != a and tid(aa) and tid(bb)]
        if not loo_diffs: continue
        loo_dir = normed(np.mean(loo_diffs, axis=0))
        query = W_E[tid(a)] + loo_dir
        cands = {w: cosine(query, tgt_vocab[w]) for w in tgt_vocab if w != a}
        if not cands: continue
        pred = max(cands, key=lambda w: cands[w])
        nc += (pred == b)
    return nc / len(ok) if ok else 0.0

def proximity_accuracy(pairs, tgt_vocab):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b) and b in tgt_vocab]
    nc = 0
    for a, b in ok:
        query = W_E[tid(a)]
        cands = {w: cosine(query, tgt_vocab[w]) for w in tgt_vocab if w != a}
        if not cands: continue
        pred = max(cands, key=lambda w: cands[w])
        nc += (pred == b)
    return nc / len(ok) if ok else 0.0

print(f"{'Domain':>22}  {'H1':>6}  {'H2':>6}  {'n_uniq':>6}  {'Pred':>10}  {'LOO_dir':>8}  {'LOO_prox':>9}")
print("-"*85)

results = {}
for name, pairs in NEW_DOMAINS.items():
    sig = compute_signals(pairs)
    if sig is None:
        print(f"  {name}: skip (insufficient single-token pairs)")
        continue

    pred_type = classify(sig["h1"], sig["h2"], sig["n_unique"])
    ok_pairs  = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    tgt_vocab = {b: W_E[tid(b)] for _, b in ok_pairs}

    acc_dir  = loo_accuracy(ok_pairs, tgt_vocab)
    acc_prox = proximity_accuracy(ok_pairs, tgt_vocab)

    # Determine observed encoding from accuracy
    if acc_dir >= 0.70:
        obs = "TYPE_BC"
    elif acc_prox >= 0.50:
        obs = "TYPE_A"
    else:
        obs = "TYPE_DE or THEMATIC"

    match = "✓" if (pred_type == obs or
                     (pred_type in {"TYPE_DE","THEMATIC"} and obs == "TYPE_DE or THEMATIC")) else "✗"

    print(f"  {name:>22}  {sig['h1']:>6.3f}  {sig['h2']:>6.3f}  {sig['n_unique']:>6}  "
          f"{pred_type:>10}  {acc_dir:>8.3f}  {acc_prox:>9.3f}  {match}")

    results[name] = {**sig, "predicted": pred_type, "LOO_dir": acc_dir,
                     "LOO_prox": acc_prox, "observed": obs, "match": match == "✓"}

# ─── Classifier accuracy ─────────────────────────────────────────
print()
print("="*60)
n_correct = sum(1 for r in results.values() if r["match"])
print(f"Classifier accuracy: {n_correct}/{len(results)} correct")
print()
for name, r in results.items():
    print(f"  {name}: predicted={r['predicted']}, LOO_dir={r['LOO_dir']:.3f}, LOO_prox={r['LOO_prox']:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 180 complete.")
