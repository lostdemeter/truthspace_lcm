#!/usr/bin/env python3
"""
Day 228 — Homogeneous Size Axis Hypothesis

DC 379 proposed: the mixed size axis (big/large/huge/tall/wide/thick spans
volume/height/breadth/thickness sub-dimensions) produces low axis_align=0.547.
Hypothesis: length-extent training pairs (all linear extent) yield
axis_align > 0.70 and improve test acc beyond 0.333.

Test pairs to predict: deep->shallow, high->low, long->short.
These are all length/extent antonyms, so a length-extent axis should work.

Experiments:
  A. Build multiple semantically homogeneous axis variants for SIZE:
       - "length_extent": long/tall/deep/broad -> short/low/shallow/narrow
       - "volume":        big/large/huge/massive -> small/tiny/little/minute
       - "dimension":     wide/broad/thick/fat -> narrow/slim/thin/slender
     Measure axis_align and retrieve on test pairs.

  B. Scan over all possible 2-6 pair SUBSETS of the original 6 size pairs
     to find which subset gives the highest axis_align.
     This reveals: which size sub-dimensions are most internally consistent?

  C. Per-sub-dimension axis retrieval:
     For test pair (deep->shallow): use length_extent axis
     For test pair (high->low): use length_extent axis
     For test pair (long->short): use length_extent axis
     Compare to original heterogeneous axis.

  D. Measure axis_align for a range of potential TYPE_ANTONYM attributes
     to map the axis_align landscape of Qwen2's W_E.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day228_homogeneous_axis.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

AXIS_VARIANTS = {
    "size_mixed": {
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "size_volume": {
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("massive","minute"),("enormous","petite"),("vast","compact")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "size_length": {
        "train": [("long","short"),("tall","low"),("deep","shallow"),
                  ("broad","narrow"),("extended","brief"),("lengthy","brief")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "size_length_v2": {
        "train": [("long","short"),("tall","low"),("deep","shallow"),
                  ("broad","narrow"),("high","low"),("far","near")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "size_dimension": {
        "train": [("wide","narrow"),("broad","slim"),("thick","thin"),
                  ("fat","slender"),("bulky","lean"),("stout","slight")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
}

EXTRA_AXES = {
    "weight":     [("heavy","light"),("massive","weightless"),("dense","sparse"),
                   ("weighty","featherweight"),("hefty","flimsy")],
    "temperature":[("hot","cold"),("warm","cool"),("burning","freezing"),
                   ("boiling","icy"),("scorching","chilly")],
    "speed":      [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                   ("rapid","gradual"),("hasty","leisurely")],
    "brightness": [("bright","dark"),("light","dim"),("vivid","dull"),
                   ("radiant","murky"),("shining","gloomy")],
    "loudness":   [("loud","quiet"),("noisy","silent"),("deafening","hushed"),
                   ("boisterous","muted"),("thunderous","whispered")],
    "roughness":  [("rough","smooth"),("coarse","fine"),("jagged","polished"),
                   ("scratchy","silky"),("rugged","sleek")],
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

def tid1(w):
    ids = tok(" "+w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None
def tid1_bare(w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None
def get_emb(w):
    t = tid1(w) or tid1_bare(w)
    return W_E[t].astype(np.float64) if t is not None else None
def ok_pairs(pairs):
    return [(a,b) for a,b in pairs if get_emb(a) is not None and get_emb(b) is not None]

print("Building vocab pool ...")
all_pool = {}
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        all_pool[w] = W_E[tid].astype(np.float64)
for cfg in list(AXIS_VARIANTS.values()) + list({"x": {"train": p, "test": []}} for p in [EXTRA_AXES[k] for k in EXTRA_AXES]):
    for key in ["train", "test"]:
        pairs = cfg.get(key, cfg) if isinstance(cfg, dict) else cfg
        if isinstance(pairs, list):
            for a,b in pairs:
                for w in (a,b):
                    if w not in all_pool:
                        e = get_emb(w)
                        if e is not None: all_pool[w] = e
for pairs in EXTRA_AXES.values():
    for a,b in pairs:
        for w in (a,b):
            if w not in all_pool:
                e = get_emb(w)
                if e is not None: all_pool[w] = e
print(f"  Pool: {len(all_pool)} tokens\n")

def build_axis(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return None, []
    diffs = [normed(get_emb(a)-get_emb(b)) for a,b in p]
    axis  = normed(np.mean(diffs, axis=0))
    aligns = [cosine(d, axis) for d in diffs]
    return axis, aligns

def retrieve_axis_result(src, axis):
    se = get_emb(src)
    if se is None or axis is None: return None
    proj = float(np.dot(normed(se), axis))
    tdir = axis if proj < 0 else -axis
    sims = [(w, cosine(normed(se+tdir), e)) for w,e in all_pool.items() if w != src]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[0][0] if sims else None, sims

def acc_on_test(test_pairs, axis):
    p = ok_pairs(test_pairs)
    if not p: return 0.0, []
    results = []
    for a,b in p:
        pred, sims = retrieve_axis_result(a, axis)
        rank = next((i for i,(w,_) in enumerate(sims) if w==b), -1)
        results.append((a,b,pred,rank))
    return sum(1 for _,b,p,_ in results if p==b)/len(results), results

# ── Part A: Axis variant comparison ──────────────────────────────────
print("=" * 70)
print("PART A: Size axis variants (mixed vs homogeneous)")
print("=" * 70)
print(f"\n  {'variant':<20}  {'n_ok':>5}  {'axis_align':>10}  "
      f"{'test_acc':>8}  train_words")
print()

variant_results = {}
for vname, vcfg in AXIS_VARIANTS.items():
    p_train = ok_pairs(vcfg["train"])
    p_test  = ok_pairs(vcfg["test"])
    axis, aligns = build_axis(p_train)
    if axis is None:
        print(f"  {vname:<20}  FAILED (insufficient ok pairs)")
        continue
    mean_aa = float(np.mean(aligns)) if aligns else 0.0
    test_acc, test_results = acc_on_test(p_test, axis)

    print(f"  {vname:<20}  {len(p_train):>5}  {mean_aa:>10.3f}  {test_acc:>8.3f}")
    # Show test pair details
    for a,b,pred,rank in test_results:
        ok = "OK" if pred==b else "  "
        print(f"    {a:>10} -> {b:<10}  pred={str(pred):<12}  rank={rank:3d}  {ok}")
    variant_results[vname] = {"axis_align": mean_aa, "test_acc": test_acc,
                               "n_train": len(p_train)}

# ── Part B: Subset scan of original 6 size pairs ─────────────────────
print()
print("=" * 70)
print("PART B: Best subset of original 6 size pairs by axis_align")
print("=" * 70)

size_all = ok_pairs([("big","small"),("large","tiny"),("huge","little"),
                     ("tall","short"),("wide","narrow"),("thick","thin")])
size_test = ok_pairs([("deep","shallow"),("high","low"),("long","short")])

best_subsets = []
print(f"\n  Scanning all subsets of size 2-6 from {len(size_all)} pairs ...")
for k in range(2, len(size_all)+1):
    for subset in combinations(size_all, k):
        axis, aligns = build_axis(list(subset))
        if axis is None: continue
        aa = float(np.mean(aligns))
        test_acc, _ = acc_on_test(size_test, axis)
        best_subsets.append((aa, test_acc, list(subset), k))

best_subsets.sort(reverse=True)
print(f"\n  Top 10 subsets by axis_align:")
print(f"  {'k':>3}  {'aa':>7}  {'test_acc':>8}  pairs")
for aa, tacc, subset, k in best_subsets[:10]:
    pair_str = " | ".join(f"{a}->{b}" for a,b in subset)
    print(f"  {k:>3}  {aa:>7.3f}  {tacc:>8.3f}  {pair_str}")

print(f"\n  Top 10 subsets by test_acc (tiebreak: axis_align):")
best_by_acc = sorted(best_subsets, key=lambda x: (x[1], x[0]), reverse=True)
for aa, tacc, subset, k in best_by_acc[:10]:
    pair_str = " | ".join(f"{a}->{b}" for a,b in subset)
    print(f"  {k:>3}  {aa:>7.3f}  {tacc:>8.3f}  {pair_str}")

# ── Part C: Extra attribute axes survey ──────────────────────────────
print()
print("=" * 70)
print("PART C: Extra attribute axes axis_align landscape")
print("=" * 70)
print(f"\n  {'attribute':<14}  {'n_ok':>5}  {'axis_align':>10}  "
      f"{'train_acc_42k':>13}")

extra_results = {}
for attr, pairs in EXTRA_AXES.items():
    p = ok_pairs(pairs)
    axis, aligns = build_axis(p)
    if axis is None or not p:
        print(f"  {attr:<14}  FAILED")
        continue
    aa = float(np.mean(aligns)) if aligns else 0.0

    # self-retrieval on training pairs
    correct = 0
    for a,b in p:
        pred, _ = retrieve_axis_result(a, axis)
        if pred == b: correct += 1
    train_acc = correct / len(p) if p else 0.0

    print(f"  {attr:<14}  {len(p):>5}  {aa:>10.3f}  {train_acc:>13.3f}")
    extra_results[attr] = {"n_ok": len(p), "axis_align": aa, "train_acc_42k": train_acc}

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

# Find best size variant
best_v = max(variant_results.items(), key=lambda x: x[1]["axis_align"])
print(f"  Best size axis variant: {best_v[0]}")
print(f"    axis_align={best_v[1]['axis_align']:.3f}  test_acc={best_v[1]['test_acc']:.3f}")
print()

# Axis align threshold confirmation
print("  Axis_align landscape (all attributes, sorted):")
all_axes = {}
for vname, vr in variant_results.items():
    all_axes[f"size/{vname}"] = vr["axis_align"]
for attr, er in extra_results.items():
    all_axes[attr] = er["axis_align"]
for k,v in sorted(all_axes.items(), key=lambda x: x[1], reverse=True):
    verdict = "RELIABLE" if v > 0.70 else ("PARTIAL" if v > 0.55 else "FAILS")
    print(f"    {k:<25}  {v:.3f}  {verdict}")

output = {
    "variant_results": variant_results,
    "best_subset_top10": [(aa,tacc,[f"{a}->{b}" for a,b in s]) for aa,tacc,s,_ in best_subsets[:10]],
    "extra_results": extra_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 228 complete.")
