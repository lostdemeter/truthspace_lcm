#!/usr/bin/env python3
"""
Day 216 — Hidden Direction Vectors in TYPE_ADJACENT Domains

Question: do "weak direction" domains (dc=0.05–0.15) have a useful
directional component at full 30k vocab, even if dc < 0.15 threshold?

Hypothesis:
  A: dc=0.135 (past_tense_D) has a direction vector that beats
     proximity at full vocab (rank < 5 with direction, rank=2 without)
  B: dc < 0.05 (antonyms dc=0.020) — no useful direction at full vocab
  C: There is a dc threshold between 0.05 and 0.15 where direction
     starts to be useful for retrieval at full vocab

Method:
  1. Build full 30k word pool (same as Day 214)
  2. For each domain, compute direction vector from training pairs
  3. Test BOTH proximity and direction on test pairs at full vocab
  4. Measure: rank_nn (proximity) vs rank_bc (direction)
  5. Sweep domains across dc range:
     dc=0.000: antonyms (unsup) dc≈0.020
     dc=0.135: past_tense_D
     dc=0.159: antonyms_sup_size
     dc=0.252: gender
     dc=0.283: plurals
     dc=0.317: past_tense_B
     dc=0.368: capitals
     dc=0.378: past_tense_F
     dc=0.413: superlative
     dc=0.827: numbers

  6. Plot: rank_bc vs dc (scatter) — where is the knee?
  7. Specifically: for dc=0.135 (past_tense_D), does direction help?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day216_weak_direction.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ALL_DOMAINS = {
    "antonyms_unsup": {
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "past_tense_D": {
        "train": [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
        "test":  [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
    },
    "antonyms_sup_size": {
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "gender": {
        "train": [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress"),("hero","heroine")],
        "test":  [("father","mother"),("brother","sister"),("son","daughter"),
                  ("husband","wife"),("uncle","aunt"),("waiter","waitress")],
    },
    "plurals": {
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "past_tense_B": {
        "train": [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
        "test":  [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
    },
    "capitals": {
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi")],
    },
    "past_tense_F": {
        "train": [("go","went"),("have","had"),("do","did"),
                  ("take","took"),("give","gave"),("make","made")],
        "test":  [("come","came"),("get","got"),("stand","stood"),
                  ("leave","left"),("bring","brought"),("buy","bought")],
    },
    "superlative": {
        "train": [("big","biggest"),("fast","fastest"),("long","longest"),
                  ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        "test":  [("hard","hardest"),("dark","darkest"),("soft","softest"),
                  ("warm","warmest"),("slow","slowest"),("small","smallest")],
    },
    "numbers": {
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6")],
        "test":  [("seven","7"),("eight","8"),("nine","9")],
    },
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

def tid1_bare(word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_emb(word):
    t = tid1(word) or tid1_bare(word)
    return W_E[t].astype(np.float64) if t is not None else None

def ok_pairs(pairs):
    return [(a,b) for a,b in pairs
            if get_emb(a) is not None and get_emb(b) is not None]

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b)-get_emb(a)) for a,b in p]
    pw = [cosine(diffs[i],diffs[j])
          for i in range(len(diffs)) for j in range(i+1,len(diffs))]
    return float(np.mean(pw))

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    diffs = [normed(get_emb(b)-get_emb(a)) for a,b in p]
    return normed(np.mean(diffs, axis=0))

# Build full word pool
print("Building full single-token English word pool ...")
word_pool = {}
for token_id in range(V):
    decoded = tok.decode([token_id])
    if not decoded.startswith(" "): continue
    word = decoded[1:]
    if not word.isalpha(): continue
    if len(word) < 2: continue
    if not word.islower(): continue
    word_pool[word] = token_id

# Also add capitalised words (for capitals domain)
cap_pool = {}
for token_id in range(V):
    decoded = tok.decode([token_id])
    if not decoded.startswith(" "): continue
    word = decoded[1:]
    if not word.isalpha(): continue
    if len(word) < 2: continue
    if word[0].isupper() and word[1:].islower():
        cap_pool[word] = token_id

# Combined pool: lowercase + capitalised
all_pool = {}
for w,t in word_pool.items():
    all_pool[w] = W_E[t].astype(np.float64)
for w,t in cap_pool.items():
    if w not in all_pool:
        all_pool[w] = W_E[t].astype(np.float64)

# Add digit tokens for numbers domain
for d in ["1","2","3","4","5","6","7","8","9"]:
    t = tid1_bare(d)
    if t is not None:
        all_pool[d] = W_E[t].astype(np.float64)

print(f"  Full pool (lower + cap + digits): {len(all_pool)} tokens\n")

# Add all test targets to pool
for cfg in ALL_DOMAINS.values():
    for a,b in cfg["train"]+cfg["test"]:
        for w in (a,b):
            if w not in all_pool:
                e = get_emb(w)
                if e is not None: all_pool[w] = e

print(f"  Pool with test targets: {len(all_pool)} tokens\n")

def eval_both(domain_name, cfg, vocab):
    """Evaluate proximity and direction retrieval on test pairs."""
    train = cfg["train"]; test = cfg["test"]
    p_test = ok_pairs(test)
    mdir = mean_dir(train)
    if not p_test: return None

    nn_correct = 0; bc_correct = 0
    nn_ranks = []; bc_ranks = []

    for src, tgt in p_test:
        se = get_emb(src)
        if se is None: continue

        # Proximity
        nn_sims = [(w, cosine(se, e)) for w,e in vocab.items() if w != src]
        nn_sims.sort(key=lambda x: x[1], reverse=True)
        nn_pred = nn_sims[0][0] if nn_sims else None
        nn_rank = next((i for i,(w,_) in enumerate(nn_sims) if w == tgt), len(nn_sims))
        if nn_pred == tgt: nn_correct += 1
        nn_ranks.append(nn_rank)

        # Direction
        if mdir is not None:
            query = se + mdir
            bc_sims = [(w, cosine(query, e)) for w,e in vocab.items() if w != src]
            bc_sims.sort(key=lambda x: x[1], reverse=True)
            bc_pred = bc_sims[0][0] if bc_sims else None
            bc_rank = next((i for i,(w,_) in enumerate(bc_sims) if w == tgt), len(bc_sims))
            if bc_pred == tgt: bc_correct += 1
            bc_ranks.append(bc_rank)
        else:
            bc_ranks.append(len(vocab))

    n = len(p_test)
    return {
        "n": n,
        "nn_acc":    nn_correct / n,
        "bc_acc":    bc_correct / n,
        "nn_rank":   float(np.mean(nn_ranks)),
        "bc_rank":   float(np.mean(bc_ranks)),
        "delta_rank": float(np.mean(nn_ranks)) - float(np.mean(bc_ranks)),
    }

# ── Main evaluation ───────────────────────────────────────────────────
print("=" * 78)
print(f"FULL 30k VOCAB: Proximity vs Direction across all dc levels")
print("=" * 78)
print(f"\n  {'Domain':<20}  {'dc':>6}  {'nn_acc':>7}  {'bc_acc':>7}  "
      f"{'nn_rank':>8}  {'bc_rank':>8}  {'Δrank':>7}  {'bc wins?':>9}")
print()

results = {}
for domain_name, cfg in ALL_DOMAINS.items():
    train = cfg["train"]
    dc = dir_consistency(train)
    r = eval_both(domain_name, cfg, all_pool)
    if r is None: continue

    bc_wins = r["bc_rank"] < r["nn_rank"]
    delta_str = f"{r['delta_rank']:>+7.1f}"
    wins_str  = "YES" if bc_wins else ("tie" if r["bc_rank"]==r["nn_rank"] else "no")
    print(f"  {domain_name:<20}  {dc:>6.3f}  {r['nn_acc']:>7.3f}  {r['bc_acc']:>7.3f}  "
          f"{r['nn_rank']:>8.1f}  {r['bc_rank']:>8.1f}  {delta_str}  {wins_str:>9}")
    results[domain_name] = {**r, "dir_consistency": dc}

print()
print("=" * 78)
print("WEAK DIRECTION ANALYSIS: does dc=0.05-0.15 help at full vocab?")
print("=" * 78)

weak_domains = [(d, r) for d, r in results.items()
                if 0.05 <= r["dir_consistency"] <= 0.20]
strong_domains = [(d, r) for d, r in results.items()
                  if r["dir_consistency"] > 0.20]
no_dir_domains = [(d, r) for d, r in results.items()
                  if r["dir_consistency"] < 0.05]

print(f"\n  No direction (dc<0.05): {len(no_dir_domains)} domains")
for d, r in no_dir_domains:
    print(f"    {d:<22} dc={r['dir_consistency']:.3f}  "
          f"nn_rank={r['nn_rank']:.1f}  bc_rank={r['bc_rank']:.1f}  "
          f"bc_wins={'YES' if r['bc_rank'] < r['nn_rank'] else 'no'}")

print(f"\n  Weak direction (0.05≤dc≤0.20): {len(weak_domains)} domains")
for d, r in weak_domains:
    print(f"    {d:<22} dc={r['dir_consistency']:.3f}  "
          f"nn_rank={r['nn_rank']:.1f}  bc_rank={r['bc_rank']:.1f}  "
          f"bc_wins={'YES' if r['bc_rank'] < r['nn_rank'] else 'no'}")

print(f"\n  Strong direction (dc>0.20): {len(strong_domains)} domains")
for d, r in strong_domains:
    print(f"    {d:<22} dc={r['dir_consistency']:.3f}  "
          f"nn_rank={r['nn_rank']:.1f}  bc_rank={r['bc_rank']:.1f}  "
          f"bc_wins={'YES' if r['bc_rank'] < r['nn_rank'] else 'no'}")

print()
print("SUMMARY:")
any_weak_wins = any(r["bc_rank"] < r["nn_rank"] for d,r in weak_domains)
all_strong_win = all(r["bc_rank"] < r["nn_rank"] for d,r in strong_domains)
print(f"  Weak direction improves rank at full vocab: {any_weak_wins}")
print(f"  Strong direction improves rank at full vocab: {all_strong_win}")

# Find approximate dc threshold where bc starts winning
sorted_by_dc = sorted(results.items(), key=lambda x: x[1]["dir_consistency"])
print("\n  dc → bc_rank (ordered by dc):")
for d, r in sorted_by_dc:
    indicator = " ←WIN" if r["bc_rank"] < r["nn_rank"] else ""
    print(f"    dc={r['dir_consistency']:.3f}  bc_rank={r['bc_rank']:5.1f}  "
          f"nn_rank={r['nn_rank']:5.1f}  {d}{indicator}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 216 complete.")
