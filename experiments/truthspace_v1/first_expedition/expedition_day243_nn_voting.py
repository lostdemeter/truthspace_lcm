#!/usr/bin/env python3
"""
Day 243 — NN-Voting for Rank-N Paradigms

Day 239 showed antonym/gender/plural are rank-N (no dominant direction).
Day 242 showed plural centroid is in non-English space; gender is multilingual.

Question: for rank-N paradigms, can nearest-neighbour (NN) voting
outperform the mean direction?

NN-Voting strategy:
  Given query word A and k training pairs (a_i, b_i):
  1. For each training pair, compute the analogical transfer:
     candidate_i = emb(A) + (emb(b_i) - emb(a_i))
     nearest(candidate_i) = top-1 in pool
  2. Vote: the most frequently predicted token wins
  This is equivalent to asking each training pair "if b_i is to a_i
  as ? is to A" and taking the plurality vote.

Alternative: k-NN pool query
  1. For each training pair (a_i, b_i), find the pair where a_i is
     most similar to the query A.
  2. Return b_i from the most similar training pair.
  (Locality: if big→small, and query is "large", find the training pair
   where source ≈ large, which is large→tiny, and return tiny.)

Methods to compare:
  A. mean_dir (baseline): emb(A) + mean(normed(b_i - a_i))
  B. nn_vote: vote among top-1 from each analogical transfer candidate
  C. pair_match: return b_i where a_i is most similar to A (kNN)
  D. weighted_vote: weight each vote by cos(emb(a_i), emb(A))

Paradigms to test:
  - antonym_size: big↔small etc.  (rank-N, acc was low)
  - antonym_speed: fast↔slow etc. (rank-N)
  - gender: king→queen etc.       (rank-N, acc=80%)
  - plural: cat→cats etc.         (rank-N, acc=89%)
  - adj_sup: big→biggest          (rank-2, acc=100%, as control)

For each paradigm:
  - LOO (leave-one-out) evaluation: for each pair (a,b),
    use all OTHER pairs as training, test on (a,b).
  - Compare mean_dir vs nn_vote vs pair_match vs weighted_vote.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day243_nn_voting.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PARADIGMS = {
    "antonym_size":  [("big","small"),("large","tiny"),("huge","little"),
                      ("tall","short"),("wide","narrow"),("thick","thin"),
                      ("broad","slim"),("vast","minute"),("giant","miniature"),
                      ("massive","petite"),("enormous","microscopic"),
                      ("heavy","light"),("long","brief"),("grand","modest")],
    "antonym_speed": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                      ("rapid","gradual"),("hasty","leisurely"),("brisk","languid"),
                      ("speedy","unhurried"),("nimble","sluggish"),
                      ("sharp","dull"),("prompt","delayed")],
    "antonym_temp":  [("hot","cold"),("warm","cool"),("burning","freezing"),
                      ("scorching","icy"),("boiling","chilly"),("heated","chilled"),
                      ("fiery","frosty"),("sweltering","frigid")],
    "gender":        [("king","queen"),("man","woman"),("boy","girl"),
                      ("prince","princess"),("actor","actress"),("hero","heroine"),
                      ("monk","nun"),("duke","duchess"),("lord","lady"),
                      ("wizard","witch"),("nephew","niece"),("lion","lioness"),
                      ("father","mother"),("son","daughter"),("brother","sister"),
                      ("husband","wife"),("grandfather","grandmother")],
    "plural":        [("cat","cats"),("dog","dogs"),("house","houses"),
                      ("tree","trees"),("book","books"),("car","cars"),
                      ("bird","birds"),("ship","ships"),("hand","hands"),
                      ("door","doors"),("lamp","lamps"),("wall","walls"),
                      ("king","kings"),("boy","boys"),("word","words"),
                      ("stone","stones"),("cloud","clouds"),("road","roads"),
                      ("horse","horses"),("town","towns")],
    "adj_sup":       [("big","biggest"),("fast","fastest"),("long","longest"),
                      ("small","smallest"),("hard","hardest"),("bright","brightest"),
                      ("dark","darkest"),("rich","richest"),("deep","deepest"),
                      ("wide","widest"),("high","highest"),("low","lowest"),
                      ("old","oldest"),("young","youngest"),("hot","hottest"),
                      ("tall","tallest"),("strong","strongest"),("weak","weakest"),
                      ("short","shortest"),("cool","coolest")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b):
    return float(np.dot(normed(np.array(a, dtype=np.float64)),
                        normed(np.array(b, dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def tid1(w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
def tid1_bare(w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
def get_emb(w):
    t = tid1(w) or tid1_bare(w)
    return W_E[t].astype(np.float64) if t is not None else None
def is_single(w): return get_emb(w) is not None
def ok_pairs(pairs): return [(a,b) for a,b in pairs if is_single(a) and is_single(b)]

print("Building pool ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))

for pairs in PARADIGMS.values():
    for a, b in pairs:
        for w in [a, b]:
            if w not in pool_words:
                e = get_emb(w)
                if e is not None:
                    pool_words.append(w); pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E  = np.array(pool_embs, dtype=np.float32)
nv = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
En = (E / nv).astype(np.float32)
print(f"  Pool: {N} tokens\n")

def top1(query_emb, exclude=None):
    qn = normed(query_emb).astype(np.float32)
    sims = En @ qn
    order = np.argsort(-sims)
    for idx in order:
        w = pool_words[idx]
        if exclude and w in exclude: continue
        return w
    return None

# ── Methods ───────────────────────────────────────────────────────────

def method_mean_dir(query_a, train_pairs):
    """mean direction: emb(A) + mean(normed(b - a))"""
    ea = get_emb(query_a)
    ds = [normed(get_emb(b) - get_emb(a)) for a,b in train_pairs
          if get_emb(a) is not None and get_emb(b) is not None]
    if not ds or ea is None: return None
    d = normed(np.mean(ds, axis=0))
    return top1(ea + d, exclude={query_a})

def method_nn_vote(query_a, train_pairs):
    """NN-vote: vote among analogical transfer candidates"""
    ea = get_emb(query_a)
    if ea is None: return None
    votes = []
    for a, b in train_pairs:
        ea_tr = get_emb(a); eb_tr = get_emb(b)
        if ea_tr is None or eb_tr is None: continue
        candidate = ea + (eb_tr - ea_tr)
        pred = top1(candidate, exclude={query_a, a, b})
        if pred: votes.append(pred)
    if not votes: return None
    cnt = Counter(votes)
    return cnt.most_common(1)[0][0]

def method_pair_match(query_a, train_pairs):
    """pair-match: return b where a is most similar to query"""
    ea = get_emb(query_a)
    if ea is None: return None
    best_sim = -2.0; best_b = None
    for a, b in train_pairs:
        ea_tr = get_emb(a)
        if ea_tr is None: continue
        sim = cosine(ea, ea_tr)
        if sim > best_sim:
            best_sim = sim; best_b = b
    # Return the b from the most similar pair
    if best_b is None: return None
    # But we want to retrieve it properly from the pool
    # Apply the direction from that best pair to the query
    _, best_b_full = None, None
    for a, b in train_pairs:
        ea_tr = get_emb(a)
        if ea_tr is None: continue
        if b == best_b:
            best_a, best_b_full = a, b; break
    if best_b_full is None: return best_b
    ea_tr = get_emb(best_a); eb_tr = get_emb(best_b_full)
    if ea_tr is None or eb_tr is None: return best_b
    d = eb_tr - ea_tr
    return top1(ea + d, exclude={query_a})

def method_weighted_vote(query_a, train_pairs):
    """weighted vote: each candidate's vote weighted by cos(a_i, query)"""
    ea = get_emb(query_a)
    if ea is None: return None
    weighted = {}
    for a, b in train_pairs:
        ea_tr = get_emb(a); eb_tr = get_emb(b)
        if ea_tr is None or eb_tr is None: continue
        weight = max(0.0, cosine(ea, ea_tr))
        candidate = ea + (eb_tr - ea_tr)
        pred = top1(candidate, exclude={query_a, a, b})
        if pred:
            weighted[pred] = weighted.get(pred, 0.0) + weight
    if not weighted: return None
    return max(weighted, key=weighted.get)

def method_exemplar_match(query_a, train_pairs, k=3):
    """Top-k pair match: weighted direction from k most similar source words"""
    ea = get_emb(query_a)
    if ea is None: return None
    sims = []
    for a, b in train_pairs:
        ea_tr = get_emb(a)
        if ea_tr is None: continue
        sims.append((cosine(ea, ea_tr), a, b))
    sims.sort(reverse=True)
    top_sims = sims[:k]
    # Weighted mean direction from top-k pairs
    total_w = sum(max(0, s) for s,_,_ in top_sims)
    if total_w < 1e-8: return method_mean_dir(query_a, train_pairs)
    d = np.zeros(H, dtype=np.float64)
    for s, a, b in top_sims:
        ea_tr = get_emb(a); eb_tr = get_emb(b)
        if ea_tr is None or eb_tr is None: continue
        d += max(0, s) * normed(eb_tr - ea_tr)
    d = normed(d)
    return top1(ea + d, exclude={query_a})

METHODS = {
    "mean_dir":      method_mean_dir,
    "nn_vote":       method_nn_vote,
    "pair_match":    method_pair_match,
    "weighted_vote": method_weighted_vote,
    "exemplar_k3":   lambda q, p: method_exemplar_match(q, p, k=3),
    "exemplar_k5":   lambda q, p: method_exemplar_match(q, p, k=5),
}

# ── LOO evaluation ────────────────────────────────────────────────────
print("=" * 70)
print("LEAVE-ONE-OUT evaluation: method comparison per paradigm")
print("=" * 70)
print()

results = {}
method_names = list(METHODS.keys())

hdr = "  " + f"{'paradigm':<16}" + "".join(f"  {m[:12]:>12}" for m in method_names)
print(hdr)
print()

for pname, pairs in PARADIGMS.items():
    p = ok_pairs(pairs)
    if len(p) < 4:
        print(f"  {pname:<16}  (skipped, n={len(p)} < 4)")
        continue

    counts = {m: 0 for m in method_names}
    total = 0

    for i, (a, b) in enumerate(p):
        train = [pp for j, pp in enumerate(p) if j != i]
        if not train: continue
        total += 1
        for mname, mfunc in METHODS.items():
            pred = mfunc(a, train)
            if pred == b:
                counts[mname] += 1

    row = f"  {pname:<16}" + "".join(
        f"  {counts[m]:>2}/{total}={counts[m]/total:.3f}" for m in method_names
    )
    print(row)
    results[pname] = {"n": total, "counts": counts}

# ── Detailed failure analysis ─────────────────────────────────────────
print()
print("=" * 70)
print("FAILURE ANALYSIS: where does each method fail?")
print("=" * 70)
print()

for pname in ["antonym_size", "gender"]:
    pairs = PARADIGMS[pname]
    p = ok_pairs(pairs)
    if len(p) < 4: continue
    print(f"\n  {pname} failures:")
    for i, (a, b) in enumerate(p):
        train = [pp for j, pp in enumerate(p) if j != i]
        preds = {}
        for mname, mfunc in METHODS.items():
            preds[mname] = mfunc(a, train)
        row = f"    {a:<12}→{b:<12} | "
        row += " | ".join(f"{m[:4]}={preds[m]}" for m in method_names)
        if not all(preds[m] == b for m in method_names):
            row += " ← DISAGREEMENT"
        print(row)

# ── Direction quality per pair ─────────────────────────────────────────
print()
print("=" * 70)
print("DIRECTION QUALITY: per-pair direction cosine to best-known direction")
print("(How consistent is each training pair's direction with the LOO mean?)")
print("=" * 70)
print()

for pname in ["antonym_size", "gender", "adj_sup"]:
    pairs = PARADIGMS[pname]
    p = ok_pairs(pairs)
    if not p: continue
    # Compute the LOO direction stability
    all_dirs = [normed(get_emb(b) - get_emb(a)) for a,b in p
                if get_emb(a) is not None and get_emb(b) is not None]
    global_mean = normed(np.mean(all_dirs, axis=0))
    cos_to_mean = [cosine(d, global_mean) for d in all_dirs]
    print(f"  {pname:<16}  cos_to_mean: min={min(cos_to_mean):>+.3f}  "
          f"max={max(cos_to_mean):>+.3f}  mean={np.mean(cos_to_mean):>+.3f}  "
          f"std={np.std(cos_to_mean):.3f}")
    if pname != "adj_sup":
        print(f"    Most deviant pairs:")
        order = np.argsort(cos_to_mean)
        for idx in order[:3]:
            a, b = p[idx]
            print(f"    {a:<10}→{b:<10}  cos={cos_to_mean[idx]:>+.3f}")

print()

output = {
    "loo_results": {pn: {"n": r["n"],
                          "accs": {m: r["counts"][m]/r["n"] for m in method_names}}
                    for pn, r in results.items()}
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"Saved: {OUTPUT_FILE}")
print("Day 243 complete.")
