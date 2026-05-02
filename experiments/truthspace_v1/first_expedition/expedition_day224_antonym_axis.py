#!/usr/bin/env python3
"""
Day 224 — Antonym Axis Investigation

antonyms_sup_size TYPE_ANTONYM retrieval gives acc=0.333 at 42k vocab.
Test pairs: deep->shallow, high->low, long->short.

Questions:
  1. What does the size axis actually retrieve for each test pair?
     (rank of correct answer, what does rank-0 return?)
  2. Are the test pairs pure SIZE antonyms, or do they have competing axes?
     (deep<->shallow is also depth/height; high<->low is also altitude/status)
  3. How accurate is the size axis on its own TRAINING pairs?
     (is the axis self-consistent at 42k vocab?)
  4. What is rank of correct target for each training pair?
  5. Can axis quality be improved with more curated size pairs?
  6. How do other attribute axes (temperature, speed, brightness) compare?
  7. Is the issue vocab density (42k too many distractors) or axis impurity?
     -> Test the size axis on a CURATED 500-word pool to separate the causes.

Method:
  - Build size axis from training pairs
  - For each test and training pair: retrieve top-10, record rank
  - Build additional axes: temperature, speed, brightness, age
  - Test each axis on its own pairs at 42k and 500-word pool
  - Measure axis self-consistency (dc of axis vs each pair's A-B direction)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day224_antonym_axis.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ATTRIBUTE_AXES = {
    "size": {
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "temperature": {
        "train": [("hot","cold"),("warm","cool"),("boiling","freezing"),
                  ("scorching","icy"),("burning","chilly")],
        "test":  [("heated","chilled"),("roasting","frigid")],
    },
    "speed": {
        "train": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                  ("rapid","gradual"),("hasty","leisurely")],
        "test":  [("brisk","sluggish"),("speedy","dawdling")],
    },
    "brightness": {
        "train": [("bright","dark"),("light","dim"),("shining","gloomy"),
                  ("brilliant","murky"),("radiant","shadowy")],
        "test":  [("luminous","dull"),("vivid","faint")],
    },
    "age": {
        "train": [("old","young"),("ancient","modern"),("elderly","youthful"),
                  ("aged","new"),("mature","fresh")],
        "test":  [("antique","contemporary"),("veteran","novice")],
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
    return [(a,b) for a,b in pairs if get_emb(a) is not None and get_emb(b) is not None]

# Build full pool + curated 500-word pool
print("Building vocab pools ...")
all_pool = {}
for token_id in range(V):
    decoded = tok.decode([token_id])
    if not decoded.startswith(" "): continue
    word = decoded[1:]
    if not word.isalpha() or len(word) < 2: continue
    if word.islower() or (word[0].isupper() and word[1:].islower()):
        all_pool[word] = W_E[token_id].astype(np.float64)

# Seed all antonym words into pool
for cfg in ATTRIBUTE_AXES.values():
    for a,b in cfg["train"]+cfg["test"]:
        for w in (a,b):
            if w not in all_pool:
                e = get_emb(w)
                if e is not None: all_pool[w] = e

print(f"  Full pool: {len(all_pool)} tokens")

# Build curated pool: sample 500 words that are actual English words
# plus all antonym pairs
import random
random.seed(42)
common_words = [w for w in all_pool if 3 <= len(w) <= 8 and w.islower()]
random.shuffle(common_words)
curated_pool = {w: all_pool[w] for w in common_words[:500]}
# add all antonym words
for cfg in ATTRIBUTE_AXES.values():
    for a,b in cfg["train"]+cfg["test"]:
        for w in (a,b):
            e = get_emb(w)
            if e is not None: curated_pool[w] = e
print(f"  Curated pool: {len(curated_pool)} tokens\n")

def build_axis(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    diffs = [normed(get_emb(a)-get_emb(b)) for a,b in p]
    return normed(np.mean(diffs, axis=0))

def retrieve_top10(src, axis, src_proj, vocab):
    se = get_emb(src)
    if se is None: return [], -1
    target_dir = axis if src_proj < 0 else -axis
    query = normed(se + target_dir)
    sims = [(w, cosine(query, e)) for w,e in vocab.items() if w != src]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:10], sims

def get_rank(target, sims_sorted):
    return next((i for i,(w,_) in enumerate(sims_sorted) if w==target), len(sims_sorted))

# ── Part 1: SIZE AXIS DEEP DIVE ───────────────────────────────────────
print("=" * 70)
print("PART 1: Size axis — training pair self-consistency")
print("=" * 70)

size_train = ATTRIBUTE_AXES["size"]["train"]
size_test  = ATTRIBUTE_AXES["size"]["test"]
size_axis  = build_axis(size_train)

print("\n  Training pairs (A->B = 'big' direction):")
for a, b in ok_pairs(size_train):
    ae = get_emb(a); be = get_emb(b)
    d = normed(ae - be)
    align = cosine(d, size_axis)
    print(f"    {a}->{b}:  axis_align={align:.3f}")

print("\n  Axis self-retrieval on TRAINING pairs (curated 500-word pool):")
train_correct_c = 0; train_correct_f = 0
for a, b in ok_pairs(size_train):
    ae = get_emb(a)
    proj = float(np.dot(normed(ae), size_axis))
    top10_c, sims_c = retrieve_top10(a, size_axis, proj, curated_pool)
    top10_f, sims_f = retrieve_top10(a, size_axis, proj, all_pool)
    rank_c = get_rank(b, sims_c)
    rank_f = get_rank(b, sims_f)
    top1_c = top10_c[0][0] if top10_c else "?"
    top1_f = top10_f[0][0] if top10_f else "?"
    ok_c = "OK" if rank_c == 0 else "  "
    ok_f = "OK" if rank_f == 0 else "  "
    if rank_c == 0: train_correct_c += 1
    if rank_f == 0: train_correct_f += 1
    print(f"    {a:>10} -> {b:<10}  "
          f"500w: {ok_c} rank={rank_c:3d} top1={top1_c:<12}  "
          f"42k: {ok_f} rank={rank_f:3d} top1={top1_f:<12}")

n_train = len(ok_pairs(size_train))
print(f"\n  Training acc: 500w={train_correct_c}/{n_train}  42k={train_correct_f}/{n_train}")

print("\n  TEST pairs:")
test_correct_c = 0; test_correct_f = 0
for a, b in ok_pairs(size_test):
    ae = get_emb(a)
    proj = float(np.dot(normed(ae), size_axis))
    top10_c, sims_c = retrieve_top10(a, size_axis, proj, curated_pool)
    top10_f, sims_f = retrieve_top10(a, size_axis, proj, all_pool)
    rank_c = get_rank(b, sims_c)
    rank_f = get_rank(b, sims_f)
    top1_c = top10_c[0][0] if top10_c else "?"
    top1_f = top10_f[0][0] if top10_f else "?"
    ok_c = "OK" if rank_c == 0 else "  "
    ok_f = "OK" if rank_f == 0 else "  "
    if rank_c == 0: test_correct_c += 1
    if rank_f == 0: test_correct_f += 1
    print(f"    {a:>10} -> {b:<10}  "
          f"500w: {ok_c} rank={rank_c:3d} top1={top1_c:<12}  "
          f"42k: {ok_f} rank={rank_f:3d} top1={top1_f:<12}")
    # Print top-5 at 42k to diagnose what wins
    print(f"      42k top-5: {[w for w,_ in top10_f[:5]]}")

n_test = len(ok_pairs(size_test))
print(f"\n  Test acc: 500w={test_correct_c}/{n_test}  42k={test_correct_f}/{n_test}")

# ── Part 2: ALL AXES COMPARISON ───────────────────────────────────────
print()
print("=" * 70)
print("PART 2: All attribute axes — self-consistency and retrieval")
print("=" * 70)

results = {}
for attr, cfg in ATTRIBUTE_AXES.items():
    axis = build_axis(cfg["train"])
    if axis is None:
        print(f"\n  {attr}: AXIS FAILED (missing tokens)")
        continue

    p_train = ok_pairs(cfg["train"])
    p_test  = ok_pairs(cfg["test"])

    # Axis alignment on training pairs
    train_aligns = [cosine(normed(get_emb(a)-get_emb(b)), axis)
                    for a,b in p_train]
    mean_align = float(np.mean(train_aligns)) if train_aligns else 0.0

    # Retrieval accuracy at 42k
    def acc_at_pool(pairs, pool):
        cor = 0
        for a,b in pairs:
            ae = get_emb(a)
            proj = float(np.dot(normed(ae), axis))
            _, sims = retrieve_top10(a, axis, proj, pool)
            if get_rank(b, sims) == 0: cor += 1
        return cor / len(pairs) if pairs else 0.0

    acc_train_500 = acc_at_pool(p_train, curated_pool)
    acc_train_42k = acc_at_pool(p_train, all_pool)
    acc_test_500  = acc_at_pool(p_test,  curated_pool)
    acc_test_42k  = acc_at_pool(p_test,  all_pool)

    print(f"\n  {attr}:  axis_align={mean_align:.3f}")
    print(f"    train: 500w={acc_train_500:.3f}  42k={acc_train_42k:.3f}  "
          f"(n={len(p_train)})")
    print(f"    test:  500w={acc_test_500:.3f}   42k={acc_test_42k:.3f}  "
          f"(n={len(p_test)})")

    results[attr] = {
        "mean_align": mean_align,
        "acc_train_500w": acc_train_500, "acc_train_42k": acc_train_42k,
        "acc_test_500w":  acc_test_500,  "acc_test_42k":  acc_test_42k,
    }

# ── Part 3: SIZE AXIS — VOCAB DENSITY TEST ────────────────────────────
print()
print("=" * 70)
print("PART 3: Size axis accuracy vs pool size (diagnose density effect)")
print("=" * 70)

pool_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, len(all_pool)]
size_test_pairs = ok_pairs(size_test)

print(f"\n  Test pairs: {size_test_pairs}")
print(f"  {'pool':>8}  {'acc':>6}  {'deep rank':>10}  {'high rank':>10}  {'long rank':>10}")

common_sorted = sorted([w for w in all_pool if w.islower()], key=lambda w: len(w))

density_results = []
for ps in pool_sizes:
    if ps >= len(all_pool):
        pool = all_pool
        ps = len(all_pool)
    else:
        words = common_sorted[:ps]
        pool = {w: all_pool[w] for w in words if w in all_pool}
        for a,b in size_test_pairs:
            for w in (a,b):
                if get_emb(w) is not None: pool[w] = get_emb(w)

    ranks = []
    for a, b in size_test_pairs:
        ae = get_emb(a)
        proj = float(np.dot(normed(ae), size_axis))
        _, sims = retrieve_top10(a, size_axis, proj, pool)
        ranks.append(get_rank(b, sims))

    acc = sum(1 for r in ranks if r == 0) / len(ranks)
    rank_str = "  ".join(f"{r:>5}" for r in ranks[:3])
    print(f"  {ps:>8}  {acc:>6.3f}  {rank_str}")
    density_results.append({"pool_size": ps, "acc": acc, "ranks": ranks})

print()
print("=" * 70)
print("SUMMARY: Antonym Axis Diagnosis")
print("=" * 70)
print()
print("  Key question: is 0.333 test acc caused by:")
print("  A) Impure test pairs (deep/high/long not purely SIZE)?")
print("  B) Vocab density (42k too many distractors)?")
print("  C) Both?")
print()
print("  Evidence from Part 3 density scan:")
print("  If acc stays low at small pool sizes -> axis impurity (A)")
print("  If acc drops from high at small to low at 42k -> density (B)")

output = {
    "size_train_acc_500w": train_correct_c / n_train,
    "size_train_acc_42k":  train_correct_f / n_train,
    "size_test_acc_500w":  test_correct_c  / n_test,
    "size_test_acc_42k":   test_correct_f  / n_test,
    "all_axes": results,
    "density_scan": density_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 224 complete.")
