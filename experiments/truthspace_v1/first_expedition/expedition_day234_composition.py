#!/usr/bin/env python3
"""
Day 234 — Geometric Composition: Can Direction Vectors Be Added?

Hypothesis: If d1 maps A→B and d2 maps B→C, then d1+d2 maps A→C.
  emb(A) + d1 + d2 ≈ emb(C)

If true, W_E supports direction composition — geometric traversal
of multi-hop relational chains without stopping at intermediate nodes.
This would be a strong structural property of the embedding space.

Experiments:
  A. Direction alignment test:
     For each two-hop chain (A→B→C), measure:
       cos(d_AC, normed(d_AB + d_BC))
     where d_AB = mean direction A→B, d_BC = mean direction B→C,
     d_AC = mean direction A→C (computed directly).
     Perfect composition: cos = 1.0.
     No composition: cos = 0.0.

  B. Retrieval composition test:
     For each two-hop source A, compute:
       qt_composed = normed(emb(A) + d1 + d2)
     Rank-of and top-1 retrieval. Is the correct terminal C retrieved?
     Compare to: single-hop retrieval of C directly.

  C. Two-hop chains to test:
     Chain 1: GENDER then PLURAL
       actor → actress → actresses (if single-token)
       king  → queen   → queens
       waiter → waitress → waitresses (if single-token)
       hero  → heroine → heroines (if single-token)

     Chain 2: ANTONYM then SUPERLATIVE
       fast → slow → slowest
       big  → small → smallest
       long → short → shortest
       hard → soft  → softest (softest might be multi-token)

     Chain 3: WORD → PLURAL → PAST TENSE? (doesn't compose semantically)
     
     Chain 4: SUPERLATIVE composition
       big → bigger → biggest (do we have a comparative direction?)
       Let's measure: d(big→biggest) vs d(big→bigger) + d(bigger→biggest)

  D. Direction additivity:
     Measure whether d1 + d2 = d_direct for all available chain types.
     This is the key mathematical question: is W_E an additive direction space?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day234_composition.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Two-hop chain definitions ─────────────────────────────────────────
# Each chain: step1 pairs (A→B), step2 pairs (B→C), test_sources [A]
# Expected terminal C = step2(step1(A))
CHAINS = {
    "gender_then_plural": {
        "step1": {
            "name": "gender",
            "pairs": [("king","queen"),("man","woman"),("boy","girl"),
                      ("prince","princess"),("actor","actress"),("hero","heroine")],
        },
        "step2": {
            "name": "plural",
            "pairs": [("cat","cats"),("dog","dogs"),("house","houses"),
                      ("tree","trees"),("book","books"),("car","cars")],
        },
        "probes": [
            ("king", "queen", "queens"),
            ("man",  "woman", "women"),
            ("boy",  "girl",  "girls"),
            ("actor","actress","actresses"),
            ("hero", "heroine","heroines"),
        ],
    },
    "antonym_speed_then_superlative": {
        "step1": {
            "name": "antonym_speed",
            "pairs": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                      ("rapid","gradual"),("hasty","leisurely")],
        },
        "step2": {
            "name": "superlative",
            "pairs": [("big","biggest"),("fast","fastest"),("long","longest"),
                      ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        },
        "probes": [
            ("fast",  "slow",  "slowest"),
            ("long",  "short", "shortest"),
            ("hard",  "soft",  "softest"),
            ("smart", "dumb",  "dumbest"),
        ],
    },
    "antonym_size_then_superlative": {
        "step1": {
            "name": "antonym_size",
            "pairs": [("big","small"),("large","tiny"),("huge","little"),
                      ("tall","short"),("wide","narrow"),("thick","thin")],
        },
        "step2": {
            "name": "superlative",
            "pairs": [("big","biggest"),("fast","fastest"),("long","longest"),
                      ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        },
        "probes": [
            ("big",  "small",  "smallest"),
            ("large","tiny",   "tiniest"),
            ("tall", "short",  "shortest"),
            ("wide", "narrow", "narrowest"),
        ],
    },
    "gender_then_superlative": {
        "step1": {
            "name": "gender",
            "pairs": [("king","queen"),("man","woman"),("boy","girl"),
                      ("prince","princess"),("actor","actress"),("hero","heroine")],
        },
        "step2": {
            "name": "superlative",
            "pairs": [("big","biggest"),("fast","fastest"),("long","longest"),
                      ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        },
        "probes": [
            ("king",  "queen",    "queens"),     # semantically undefined but tests geometry
            ("man",   "woman",    "women"),
        ],
    },
    "plural_then_antonym_speed": {
        "step1": {
            "name": "plural",
            "pairs": [("cat","cats"),("dog","dogs"),("house","houses"),
                      ("tree","trees"),("book","books"),("car","cars")],
        },
        "step2": {
            "name": "antonym_speed",
            "pairs": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                      ("rapid","gradual"),("hasty","leisurely")],
        },
        "probes": [
            ("fast",  "fasts",   "slows"),     # semantically odd but tests direction
            ("quick", "quicks",  "sluggishs"),
        ],
    },
    "comparative_to_superlative": {
        "step1": {
            "name": "comparative",
            "pairs": [("big","bigger"),("fast","faster"),("long","longer"),
                      ("smart","smarter"),("small","smaller"),("hard","harder")],
        },
        "step2": {
            "name": "comparative_to_superlative",
            "pairs": [("bigger","biggest"),("faster","fastest"),("longer","longest"),
                      ("smarter","smartest"),("smaller","smallest"),("harder","hardest")],
        },
        "probes": [
            ("big",   "bigger",   "biggest"),
            ("fast",  "faster",   "fastest"),
            ("long",  "longer",   "longest"),
            ("small", "smaller",  "smallest"),
            ("hard",  "harder",   "hardest"),
        ],
    },
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
def is_single(w):
    return get_emb(w) is not None
def ok_pairs(pairs):
    return [(a, b) for a, b in pairs if is_single(a) and is_single(b)]

print("Building normed matrix ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w)
        pool_embs.append(W_E[tid].astype(np.float32))

# Add all probe words
for cdata in CHAINS.values():
    for _, step in [("step1", cdata["step1"]), ("step2", cdata["step2"])]:
        for a, b in step["pairs"]:
            for w in (a, b):
                if w not in pool_words:
                    e = get_emb(w)
                    if e is not None:
                        pool_words.append(w); pool_embs.append(e.astype(np.float32))
    for a, b, c in cdata["probes"]:
        for w in (a, b, c):
            if w not in pool_words:
                e = get_emb(w)
                if e is not None:
                    pool_words.append(w); pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E = np.array(pool_embs, dtype=np.float32)
norms_v = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
E_normed = (E / norms_v).astype(np.float32)
word_to_idx = {w: i for i, w in enumerate(pool_words)}
print(f"  Pool: {N} tokens\n")

def top_k(qt, k=10, exclude=None):
    qn = normed(qt).astype(np.float32)
    sims = E_normed @ qn
    order = np.argsort(-sims)
    out = []
    for idx in order:
        w = pool_words[idx]
        if exclude and w == exclude: continue
        out.append((w, float(sims[idx])))
        if len(out) >= k: break
    return out

def rank_of(target, qt, exclude=None):
    top = top_k(qt, k=2000, exclude=exclude)
    for i, (w, _) in enumerate(top):
        if w == target: return i
    return -1

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None, 0
    dirs = [normed(get_emb(b) - get_emb(a)) for a, b in p]
    return normed(np.mean(dirs, axis=0)), len(p)

def build_axis(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return None
    diffs = [normed(get_emb(a) - get_emb(b)) for a, b in p]
    return normed(np.mean(diffs, axis=0))

# ── Part A: Direction alignment test ────────────────────────────────
print("=" * 70)
print("PART A: Direction alignment: cos(d_direct, normed(d1+d2))")
print("=" * 70)
print()
print(f"  {'chain':<35}  {'d_direct vs d_composed':>22}  "
      f"{'|d1|':>5}  {'|d2|':>5}")

alignment_results = {}
for cname, cdata in CHAINS.items():
    d1, n1 = mean_dir(cdata["step1"]["pairs"])
    d2, n2 = mean_dir(cdata["step2"]["pairs"])
    if d1 is None or d2 is None:
        print(f"  {cname:<35}  FAILED (no pairs)")
        continue

    d_composed = normed(d1 + d2)

    # Build direct A→C pairs from probes
    direct_pairs = [(a, c) for a, b, c in cdata["probes"]
                    if is_single(a) and is_single(c)]
    d_direct, nd = mean_dir(direct_pairs)
    if d_direct is None:
        align = float("nan")
    else:
        align = cosine(d_direct, d_composed)

    print(f"  {cname:<35}  cos={align:>+.4f}  "
          f"n_direct={nd:>2}  n1={n1:>2}  n2={n2:>2}")
    alignment_results[cname] = {
        "cos_align": align, "n1": n1, "n2": n2, "n_direct": nd
    }

# ── Part B: Retrieval composition test ──────────────────────────────
print()
print("=" * 70)
print("PART B: Retrieval — does emb(A) + d1 + d2 retrieve correct C?")
print("=" * 70)
print()

retrieval_results = {}
for cname, cdata in CHAINS.items():
    d1, _ = mean_dir(cdata["step1"]["pairs"])
    d2, _ = mean_dir(cdata["step2"]["pairs"])
    if d1 is None or d2 is None: continue

    print(f"  {cname}:")
    print(f"    {'A':<10}  {'B':<12}  {'C':<12}  "
          f"{'pred_composed':<16}  {'rank_C':>7}  "
          f"{'pred_step2(B)':<16}  {'rank_C_step2':>12}")

    chain_results = []
    for a, b, c in cdata["probes"]:
        if not (is_single(a) and is_single(b) and is_single(c)):
            print(f"    {a:<10}  {b:<12}  {c:<12}  "
                  f"EC_TOKENIZE")
            continue

        ea = get_emb(a)

        # Composed: emb(A) + d1 + d2
        qt_composed = normed(ea + d1 + d2)
        top_composed = top_k(qt_composed, k=5, exclude=a)
        pred_c = top_composed[0][0] if top_composed else None
        rank_c = rank_of(c, qt_composed, exclude=a)

        # Step2 from B: emb(B) + d2 (one-hop from intermediate)
        eb = get_emb(b)
        qt_step2 = normed(eb + d2)
        top_step2 = top_k(qt_step2, k=5, exclude=b)
        pred_s2   = top_step2[0][0] if top_step2 else None
        rank_c_s2 = rank_of(c, qt_step2, exclude=b)

        ok_c  = "OK" if pred_c  == c else "  "
        ok_s2 = "OK" if pred_s2 == c else "  "
        print(f"    {a:<10}  {b:<12}  {c:<12}  "
              f"{str(pred_c):<14}  {ok_c}  {rank_c:>6}  "
              f"{str(pred_s2):<14}  {ok_s2}  {rank_c_s2:>11}")

        chain_results.append({
            "a": a, "b": b, "c": c,
            "pred_composed": pred_c, "rank_c_composed": rank_c,
            "correct_composed": (pred_c == c),
            "pred_step2": pred_s2, "rank_c_step2": rank_c_s2,
            "correct_step2": (pred_s2 == c),
        })
    retrieval_results[cname] = chain_results
    print()

# ── Part C: Single-hop baseline comparison ───────────────────────────
print("=" * 70)
print("PART C: Single-hop step1 accuracy (A→B) for reference")
print("=" * 70)
print()

for cname, cdata in CHAINS.items():
    d1, _ = mean_dir(cdata["step1"]["pairs"])
    if d1 is None: continue
    p = ok_pairs(cdata["step1"]["pairs"])
    if not p: continue
    correct = 0
    for a, b in p:
        ea = get_emb(a)
        qt = normed(ea + d1)
        top = top_k(qt, k=3, exclude=a)
        pred = top[0][0] if top else None
        if pred == b: correct += 1
    print(f"  {cname:<35}  step1_acc = {correct}/{len(p)} = {correct/len(p):.3f}")
print()
print("=" * 70)
print("PART C: Single-hop step2 accuracy (B→C) for reference")
print("=" * 70)
print()
for cname, cdata in CHAINS.items():
    d2, _ = mean_dir(cdata["step2"]["pairs"])
    if d2 is None: continue
    p = ok_pairs(cdata["step2"]["pairs"])
    if not p: continue
    correct = 0
    for a, b in p:
        ea = get_emb(a)
        qt = normed(ea + d2)
        top = top_k(qt, k=3, exclude=a)
        pred = top[0][0] if top else None
        if pred == b: correct += 1
    print(f"  {cname:<35}  step2_acc = {correct}/{len(p)} = {correct/len(p):.3f}")

# ── Part D: Direction additivity sweep ───────────────────────────────
print()
print("=" * 70)
print("PART D: Direction additivity — cos(d1+d2, d_direct) for known chains")
print("=" * 70)
print()
print("  Expected if W_E is ADDITIVE: cos close to 1.0")
print("  Expected if W_E is NOT additive: cos close to 0.0 or negative")
print()
print(f"  {'chain':<35}  {'cos':>8}  interpretation")
for cname, res in alignment_results.items():
    cos = res["cos_align"]
    if np.isnan(cos):
        interp = "NO_DIRECT_PAIRS"
    elif cos > 0.90:
        interp = "STRONG composition"
    elif cos > 0.70:
        interp = "MODERATE composition"
    elif cos > 0.40:
        interp = "WEAK composition"
    elif cos > 0.0:
        interp = "NEAR-ORTHOGONAL (no composition)"
    else:
        interp = "ANTI-CORRELATED (composition inverted)"
    print(f"  {cname:<35}  {cos:>8.4f}  {interp}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  Composition hypothesis: emb(A) + d1 + d2 ≈ emb(C)")
print("  Tested on chains: gender→plural, antonym→superlative, comparative→superlative")
print("  Key measure: cos(d_direct_AC, normed(d_AB + d_BC))")

# Aggregate composition correctness
total_comp = sum(len(v) for v in retrieval_results.values())
correct_comp = sum(r["correct_composed"] for v in retrieval_results.values() for r in v)
correct_s2   = sum(r["correct_step2"]    for v in retrieval_results.values() for r in v)
if total_comp > 0:
    print(f"\n  Composed retrieval:  {correct_comp}/{total_comp} = {correct_comp/total_comp:.3f}")
    print(f"  Step2-from-B:        {correct_s2}/{total_comp}   = {correct_s2/total_comp:.3f}")

output = {
    "alignment": alignment_results,
    "retrieval": retrieval_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 234 complete.")
