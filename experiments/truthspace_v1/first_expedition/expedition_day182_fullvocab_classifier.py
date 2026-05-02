#!/usr/bin/env python3
"""
Day 182 — Full-Vocabulary Classifier

Day 181 finding: in a restricted vocabulary, proximity accuracy is trivially
high for ALL encoding types — the taxonomy (A, B, C, D, E) was implicitly
defined in a FULL vocabulary context.

FIX: Build a MIXED vocabulary containing:
  1. All domain source+target words (cross-contamination: every domain can
     see every other domain's words as distractors)
  2. Extra distractor words (common nouns, adjectives, verbs)

Then re-measure proximity and direction accuracy in this full mixed vocabulary
and see which method wins per domain → that is the empirical encoding type.

HYPOTHESIS:
  TYPE_A: proximity wins even in full vocab (hot→cold, tennis→court)
  TYPE_BC: direction wins in full vocab (France→French, king→queen)
  TYPE_DE: neither wins without routing (parity, planet type)
  THEMATIC: neither wins (season_weather, maybe animal_sound)

This should restore the original Day 162 taxonomy and validate the geometric
signals (H1/H2) as predictors of direction advantage over proximity.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day182_fullvocab_classifier.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAINS = {
    "capitals": ([
        ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
        ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
        ("Korea","Seoul"),("Poland","Warsaw"),
    ], "TYPE_BC"),
    "languages": ([
        ("France","French"),("Germany","German"),("Italy","Italian"),
        ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
        ("Russia","Russian"),("Greece","Greek"),("Sweden","Swedish"),
    ], "TYPE_BC"),
    "gender": ([
        ("king","queen"),("man","woman"),("boy","girl"),
        ("prince","princess"),("lord","lady"),("actor","actress"),
        ("hero","heroine"),
    ], "TYPE_BC"),
    "animal_sound": ([
        ("dog","bark"),("cat","meow"),("cow","moo"),("duck","quack"),
        ("lion","roar"),("bird","tweet"),("bee","buzz"),("frog","croak"),
    ], "THEMATIC"),
    "metal_property": ([
        ("iron","magnetic"),("copper","conductive"),("gold","malleable"),
        ("silver","reflective"),("aluminum","lightweight"),("lead","heavy"),
    ], "THEMATIC"),
    "season_weather": ([
        ("winter","snow"),("summer","heat"),("spring","rain"),("autumn","wind"),
    ], "THEMATIC"),
    "antonym": ([
        ("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
        ("light","dark"),("old","young"),("loud","quiet"),
    ], "TYPE_A"),
    "number_parity": ([
        ("one","odd"),("two","even"),("three","odd"),("four","even"),
        ("five","odd"),("six","even"),("seven","odd"),("eight","even"),
    ], "TYPE_DE"),
    "planet_type": ([
        ("Mercury","rocky"),("Venus","rocky"),("Earth","rocky"),("Mars","rocky"),
        ("Jupiter","gas"),("Saturn","gas"),("Uranus","gas"),("Neptune","gas"),
    ], "TYPE_DE"),
    "sport_venue": ([
        ("tennis","court"),("basketball","court"),("swimming","pool"),
        ("boxing","ring"),("hockey","rink"),("football","field"),
        ("baseball","field"),("golf","course"),
    ], "TYPE_A"),
    "country_currency": ([
        ("France","euro"),("Germany","euro"),("Italy","euro"),
        ("Japan","yen"),("China","yuan"),("Russia","ruble"),
        ("India","rupee"),("Korea","won"),("Sweden","krona"),
    ], "TYPE_A"),
    "country_continent": ([
        ("France","Europe"),("Germany","Europe"),("Japan","Asia"),
        ("China","Asia"),("India","Asia"),("Brazil","America"),
        ("Mexico","America"),("Egypt","Africa"),("Korea","Asia"),
    ], "TYPE_A"),
}

# Extra distractors: common English words to dilute the vocabulary
DISTRACTORS = [
    "table","chair","window","door","book","water","fire","stone","tree","road",
    "city","house","river","mountain","ocean","cloud","night","morning","color","music",
    "green","blue","red","yellow","black","white","orange","purple","pink","brown",
    "run","walk","jump","eat","drink","sleep","think","speak","write","read",
    "strong","weak","long","short","wide","narrow","deep","high","low","flat",
    "happy","sad","angry","afraid","brave","clever","stupid","rich","poor","young",
    "tiger","wolf","eagle","shark","whale","ant","spider","snake","rabbit","horse",
    "physics","chemistry","biology","history","music","art","science","math","language",
    "north","south","east","west","left","right","above","below","inside","outside",
    "begin","end","open","close","push","pull","rise","fall","grow","shrink",
    "silver","copper","zinc","steel","tin","platinum","nickel","chrome","bronze",
    "ocean","desert","forest","jungle","tundra","prairie","savanna","swamp","glacier",
    "carbon","oxygen","hydrogen","nitrogen","sodium","calcium","potassium","chlorine",
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
print(f"  H={W_E.shape[1]}")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

# Build full mixed vocabulary
all_words = set(DISTRACTORS)
for pairs, _ in DOMAINS.values():
    for a, b in pairs:
        all_words.update([a, b])
full_vocab = {w: W_E[tid(w)] for w in all_words if tid(w)}
print(f"  Full vocabulary size: {len(full_vocab)}\n")

def direction_acc_full(pairs):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    nc = 0
    for a, b in ok:
        loo = [normed(W_E[tid(bb)]-W_E[tid(aa)]) for aa,bb in ok if aa!=a]
        if not loo: continue
        d = normed(np.mean(loo, axis=0))
        q = W_E[tid(a)] + d
        cands = {w: cosine(q, full_vocab[w]) for w in full_vocab if w != a}
        if cands and max(cands, key=lambda w: cands[w]) == b:
            nc += 1
    return nc/len(ok) if ok else 0.0

def proximity_acc_full(pairs):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    nc = 0
    for a, b in ok:
        cands = {w: cosine(W_E[tid(a)], full_vocab[w]) for w in full_vocab if w != a}
        if cands and max(cands, key=lambda w: cands[w]) == b:
            nc += 1
    return nc/len(ok) if ok else 0.0

def compute_signals(pairs):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    if len(ok) < 3: return None
    diffs = [normed(W_E[tid(b)] - W_E[tid(a)]) for a, b in ok]
    tgts  = [normed(W_E[tid(b)]) for a, b in ok]
    n = len(diffs)
    h1 = float(np.mean([cosine(diffs[i],diffs[j]) for i in range(n) for j in range(i+1,n)]))
    h2 = float(np.mean([cosine(tgts[i],tgts[j])   for i in range(n) for j in range(i+1,n)]))
    return {"h1": h1, "h2": h2}

print(f"{'Domain':>20}  {'H1':>5}  {'H2':>5}  {'prox_full':>9}  {'dir_full':>8}  "
      f"{'Winner':>8}  {'Truth':>10}  {'OK':>3}")
print("-"*85)

results = {}
n_correct = 0
for name, (pairs, true_type) in DOMAINS.items():
    sig = compute_signals(pairs)
    if not sig: continue
    prox = proximity_acc_full(pairs)
    dirr = direction_acc_full(pairs)

    if dirr >= 0.70:
        obs = "TYPE_BC"
    elif prox >= 0.55:
        obs = "TYPE_A"
    else:
        obs = "TYPE_DE_or_THEMATIC"

    # Simplified geometric classifier using only H1/H2
    def classify_geo(h1, h2):
        if h1 < 0.10: return "TYPE_A"
        if h2 < 0.15: return "THEMATIC"
        if h2 >= 0.40 and h2 <= 0.80: return "TYPE_DE"
        if h1 >= 0.20 and h2 >= 0.25: return "TYPE_BC"
        return "UNKNOWN"

    pred_geo = classify_geo(sig["h1"], sig["h2"])

    def types_match(p, t):
        a = {"TYPE_A","TYPE_A_WEAK","TYPE_A_MODERATE"}
        if p in a and t in a: return True
        return p == t

    ok_geo   = types_match(pred_geo, true_type)
    ok_obs   = types_match(obs, true_type) or \
               (true_type in {"TYPE_DE","THEMATIC"} and obs == "TYPE_DE_or_THEMATIC")

    winner = "DIR" if dirr > prox else "PROX" if prox > dirr else "TIE"
    n_correct += ok_obs
    mark = "✓" if ok_obs else "✗"

    print(f"  {name:>20}  {sig['h1']:>5.3f}  {sig['h2']:>5.3f}  {prox:>9.3f}  {dirr:>8.3f}  "
          f"{winner:>8}  {true_type:>10}  {mark:>3}")

    results[name] = {**sig, "prox_full": prox, "dir_full": dirr,
                     "winner": winner, "obs": obs,
                     "pred_geo": pred_geo, "truth": true_type, "ok": ok_obs}

print()
print(f"  Observed type matches ground truth: {n_correct}/{len(results)}")
print()

# ─── What H1/H2 can now predict ────────────────────────────────────
print("  Geometric H1/H2 predictor (full vocab context):")
n_geo = sum(1 for r in results.values()
            if types_match(r["pred_geo"], r["truth"]) or
               (r["truth"] in {"TYPE_DE","THEMATIC"} and r["pred_geo"] in {"TYPE_DE","THEMATIC"}))
print(f"  H1/H2 classifier accuracy: {n_geo}/{len(results)}")
print()

# Key analysis: does direction advantage correlate with H1?
print("  H1 vs (dir_full - prox_full) [direction advantage]:")
for name, r in sorted(results.items(), key=lambda x: -x[1]["h1"]):
    adv = r["dir_full"] - r["prox_full"]
    print(f"    {name:>20}  H1={r['h1']:.3f}  adv={adv:+.3f}  "
          f"{'DIR wins' if adv>0.05 else 'PROX wins' if adv<-0.05 else 'TIE'}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 182 complete.")
