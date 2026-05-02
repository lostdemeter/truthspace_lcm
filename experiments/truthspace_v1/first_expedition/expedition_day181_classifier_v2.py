#!/usr/bin/env python3
"""
Day 181 — Revised 2-Stage Auto-Classifier

Day 180 failure: geometric-only classifier (H1/H2/n_unique) gets 1/6.
Root cause: proximity accuracy (TYPE_A) can dominate even when H1/H2 are moderate.

REVISED CLASSIFIER — 2-Stage with empirical accuracy as primary signal:

  Stage 1 (empirical, free):
    Compute proximity accuracy with k=3 test pairs (no training, just k=0 NN).
    If prox_acc ≥ 0.55 → TYPE_A (proximity-dominant)

  Stage 2 (geometric, for non-A):
    H1 < 0.10                         → TYPE_A_WEAK  (no consistent direction)
    H2 < 0.15                         → THEMATIC     (absent from W_E)
    H2 ≥ 0.40 AND H2_cv < 0.30       → TYPE_DE      (compact target cluster = multi-pole)
    H1 ≥ 0.20 AND H2 ≥ 0.25          → TYPE_BC      (direction works)
    else                              → TYPE_A_MODERATE (proximity, borderline)

  H2_cv = coefficient of variation of pairwise target cosines
  (low CV means targets all similar distances = single cluster; high CV = multiple clusters)

VALIDATION: Run on all 12 Day-178 domains + all 6 Day-180 domains = 18 total.
Compare predicted type to observed (direction acc vs proximity acc).
Target: ≥ 14/18 correct (78%).
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day181_classifier_v2.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ALL_DOMAINS = {
    # ── Day 178 domains (known encoding types) ───────────────────
    "capitals": ([
        ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
        ("Russia","Moscow"),("Greece","Athens"),("Poland","Warsaw"),
        ("Sweden","Stockholm"),("Korea","Seoul"),
    ], "TYPE_BC"),
    "languages": ([
        ("France","French"),("Germany","German"),("Italy","Italian"),
        ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
        ("Russia","Russian"),("Greece","Greek"),("Sweden","Swedish"),
    ], "TYPE_BC"),
    "gender": ([
        ("king","queen"),("man","woman"),("boy","girl"),
        ("prince","princess"),("lord","lady"),("actor","actress"),
        ("waiter","waitress"),("hero","heroine"),
    ], "TYPE_BC"),
    "metal_to_category": ([
        ("iron","metal"),("copper","metal"),("aluminum","metal"),
        ("tin","metal"),("zinc","metal"),("lead","metal"),
        ("gold","metal"),("silver","metal"),
    ], "TYPE_BC"),
    "animal_sound": ([
        ("dog","bark"),("cat","meow"),("cow","moo"),("duck","quack"),
        ("lion","roar"),("bird","tweet"),("frog","croak"),("bee","buzz"),
    ], "THEMATIC"),
    "metal_property": ([
        ("iron","magnetic"),("copper","conductive"),("gold","malleable"),
        ("silver","reflective"),("aluminum","lightweight"),("lead","heavy"),
    ], "THEMATIC"),
    "season_weather": ([
        ("winter","snow"),("summer","heat"),("spring","rain"),("autumn","wind"),
        ("winter","cold"),("summer","hot"),("spring","mild"),("autumn","cool"),
    ], "THEMATIC"),
    "number_parity": ([
        ("one","odd"),("two","even"),("three","odd"),("four","even"),
        ("five","odd"),("six","even"),("seven","odd"),("eight","even"),
    ], "TYPE_DE"),
    "planet_type": ([
        ("Mercury","rocky"),("Venus","rocky"),("Earth","rocky"),("Mars","rocky"),
        ("Jupiter","gas"),("Saturn","gas"),("Uranus","gas"),("Neptune","gas"),
    ], "TYPE_DE"),
    "color_temperature": ([
        ("red","warm"),("orange","warm"),("yellow","warm"),
        ("blue","cool"),("green","cool"),("purple","cool"),
    ], "TYPE_DE"),
    "antonym_hot": ([
        ("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
        ("light","dark"),("old","young"),("loud","quiet"),("rich","poor"),
    ], "TYPE_A"),
    "insect_category": ([
        ("ant","insect"),("bee","insect"),("fly","insect"),
        ("moth","insect"),("wasp","insect"),("beetle","insect"),
    ], "TYPE_BC"),
    # ── Day 180 domains (discovered types from experiment) ─────────
    "country_continent": ([
        ("France","Europe"),("Germany","Europe"),("Italy","Europe"),
        ("Spain","Europe"),("Japan","Asia"),("China","Asia"),
        ("India","Asia"),("Russia","Europe"),("Brazil","America"),
        ("Mexico","America"),("Canada","America"),("Egypt","Africa"),
        ("Korea","Asia"),("Greece","Europe"),
    ], "TYPE_A"),     # proximity won (0.556 vs 0.389)
    "animal_size": ([
        ("elephant","large"),("whale","large"),("lion","large"),
        ("horse","large"),("cat","small"),("mouse","small"),
        ("rabbit","small"),("dog","medium"),("sheep","medium"),
    ], "TYPE_A"),     # proximity 0.500 >= direction 0.417
    "sport_venue": ([
        ("tennis","court"),("basketball","court"),("swimming","pool"),
        ("boxing","ring"),("hockey","rink"),("football","field"),
        ("baseball","field"),("golf","course"),("bowling","lane"),
    ], "TYPE_A"),     # proximity 0.875, direction 0.000
    "country_currency": ([
        ("France","euro"),("Germany","euro"),("Italy","euro"),
        ("Spain","euro"),("Japan","yen"),("China","yuan"),
        ("Russia","ruble"),("India","rupee"),("Korea","won"),
        ("Sweden","krona"),("Poland","zloty"),
    ], "TYPE_A"),     # proximity 0.750, direction 0.500
    "word_category": ([
        ("Paris","noun"),("London","noun"),("run","verb"),
        ("jump","verb"),("fast","adjective"),("slow","adjective"),
        ("quickly","adverb"),("softly","adverb"),
    ], "TYPE_DE"),    # grammatical categories: multi-pole
    "number_magnitude": ([
        ("one","small"),("two","small"),("three","small"),
        ("hundred","large"),("million","large"),("billion","large"),
        ("ten","medium"),("fifty","medium"),
    ], "TYPE_A"),     # proximity 0.667, direction 0.000
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

def compute_all_signals(pairs):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    if len(ok) < 3: return None
    diffs = [normed(W_E[tid(b)] - W_E[tid(a)]) for a, b in ok]
    tgts  = [normed(W_E[tid(b)]) for a, b in ok]
    n = len(diffs)
    cos_pairs  = [cosine(diffs[i], diffs[j]) for i in range(n) for j in range(i+1,n)]
    tgt_pairs  = [cosine(tgts[i], tgts[j])   for i in range(n) for j in range(i+1,n)]
    h1 = float(np.mean(cos_pairs)) if cos_pairs else 0.0
    h2 = float(np.mean(tgt_pairs)) if tgt_pairs else 0.0
    h2_cv = float(np.std(tgt_pairs)/max(np.mean(tgt_pairs),1e-8)) if tgt_pairs else 0.0
    n_unique = len(set(b for _, b in ok))
    return {"h1": h1, "h2": h2, "h2_cv": h2_cv, "n_unique": n_unique, "n_pairs": len(ok)}

def acc_direction(pairs, tgt_vocab):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b) and b in tgt_vocab]
    nc = 0
    for a, b in ok:
        loo = [normed(W_E[tid(bb)]-W_E[tid(aa)]) for aa,bb in ok if aa!=a]
        if not loo: continue
        d = normed(np.mean(loo, axis=0))
        q = W_E[tid(a)] + d
        cands = {w: cosine(q, tgt_vocab[w]) for w in tgt_vocab if w != a}
        if cands: nc += (max(cands, key=lambda w: cands[w]) == b)
    return nc/len(ok) if ok else 0.0

def acc_proximity(pairs, tgt_vocab):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b) and b in tgt_vocab]
    nc = 0
    for a, b in ok:
        cands = {w: cosine(W_E[tid(a)], tgt_vocab[w]) for w in tgt_vocab if w != a}
        if cands: nc += (max(cands, key=lambda w: cands[w]) == b)
    return nc/len(ok) if ok else 0.0

def classify_v2(prox_acc, h1, h2, h2_cv, n_unique):
    # Stage 1: empirical proximity check
    if prox_acc >= 0.55:
        return "TYPE_A"
    # Stage 2: geometric signals
    if h1 < 0.10:
        return "TYPE_A_WEAK"
    if h2 < 0.15:
        return "THEMATIC"
    if h2 >= 0.40 and h2_cv < 0.30 and n_unique <= 5:
        return "TYPE_DE"
    if h1 >= 0.20 and h2 >= 0.25:
        return "TYPE_BC"
    return "TYPE_A_MODERATE"

def types_match(pred, truth):
    # Normalize for comparison
    EQUIV = {
        "TYPE_A": {"TYPE_A","TYPE_A_WEAK","TYPE_A_MODERATE"},
        "TYPE_BC": {"TYPE_BC"},
        "TYPE_DE": {"TYPE_DE"},
        "THEMATIC": {"THEMATIC"},
    }
    pred_grp  = next((k for k,v in EQUIV.items() if pred  in v), pred)
    truth_grp = next((k for k,v in EQUIV.items() if truth in v), truth)
    return pred_grp == truth_grp

print(f"{'Domain':>22}  {'H1':>5}  {'H2':>5} {'H2cv':>5}  {'prox':>5}  "
      f"{'dir':>5}  {'Pred':>18}  {'Truth':>10}  {'OK':>3}")
print("-"*105)

results = {}
n_correct = 0
for name, (pairs, true_type) in ALL_DOMAINS.items():
    sig = compute_all_signals(pairs)
    if sig is None:
        print(f"  {name}: skip")
        continue
    ok_pairs  = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    tgt_vocab = {b: W_E[tid(b)] for _, b in ok_pairs}
    prox = acc_proximity(ok_pairs, tgt_vocab)
    dirr = acc_direction(ok_pairs, tgt_vocab)
    pred = classify_v2(prox, sig["h1"], sig["h2"], sig["h2_cv"], sig["n_unique"])
    ok   = types_match(pred, true_type)
    n_correct += ok
    mark = "✓" if ok else "✗"
    print(f"  {name:>22}  {sig['h1']:>5.3f}  {sig['h2']:>5.3f}  {sig['h2_cv']:>5.2f}  "
          f"{prox:>5.3f}  {dirr:>5.3f}  {pred:>18}  {true_type:>10}  {mark:>3}")
    results[name] = {**sig, "prox": prox, "dir": dirr, "pred": pred,
                     "truth": true_type, "ok": ok}

print()
print(f"  Classifier v2 accuracy: {n_correct}/{len(results)} "
      f"({100*n_correct/len(results):.0f}%)")

# ─── Where does v2 still fail? ────────────────────────────────────
print()
print("  Failures:")
for name, r in results.items():
    if not r["ok"]:
        print(f"    {name}: pred={r['pred']} truth={r['truth']} "
              f"prox={r['prox']:.3f} dir={r['dir']:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"results": results,
               "accuracy": n_correct/len(results) if results else 0},
              f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 181 complete.")
