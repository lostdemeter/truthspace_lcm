#!/usr/bin/env python3
"""
Day 231 — Rank Gap Metric for Retrieval Confidence

Day 230 showed cosine threshold degeneracy=0 for all antonym queries
in 1536-d space (threshold too high). The correct metric is rank_gap:
  rank_gap = sim(rank-1) - sim(rank-2)

Hypothesis: rank_gap > 0.05 -> reliable retrieval
            rank_gap < 0.02 -> centroid collapse / failure likely

Experiments:
  A. Compute rank_gap for every pair in every domain at 42k vocab.
     Compare mean rank_gap of correct vs wrong retrievals.
     Find threshold that maximally separates them.

  B. Rank gap distributions:
     - TYPE_BC (capitals, gender, numbers, plurals, etc.)
     - TYPE_ANTONYM low-deg (speed, weight, roughness)
     - TYPE_ANTONYM high-deg (size, brightness, temperature, loudness)
     - TYPE_ADJACENT (antonyms_unsup)
     Compare distributions: are they separable?

  C. Extended rank gap (top-1 vs top-k gap):
     rank_gap_k = sim(rank-1) - sim(rank-k) for k in [2, 3, 5, 10]
     Does a wider window improve predictive power?

  D. Runtime confidence pipeline:
     conf = rank_gap (or tanh(rank_gap / scale))
     Threshold: if conf < conf_threshold, flag as "unreliable"
     Measure: what fraction of flagged pairs would be correct/wrong?
     Precision of "unreliable" flag.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day231_rank_gap.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ALL_DOMAINS = {
    "capitals": {
        "type": "TYPE_BC", "attribute": None,
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi")],
    },
    "gender": {
        "type": "TYPE_BC", "attribute": None,
        "train": [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress"),("hero","heroine")],
        "test":  [("father","mother"),("brother","sister"),("son","daughter"),
                  ("husband","wife"),("uncle","aunt"),("waiter","waitress")],
    },
    "plurals": {
        "type": "TYPE_BC", "attribute": None,
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "numbers": {
        "type": "TYPE_BC", "attribute": None,
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6")],
        "test":  [("seven","7"),("eight","8"),("nine","9")],
    },
    "superlative": {
        "type": "TYPE_BC", "attribute": None,
        "train": [("big","biggest"),("fast","fastest"),("long","longest"),
                  ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        "test":  [("hard","hardest"),("dark","darkest"),("soft","softest"),
                  ("warm","warmest"),("slow","slowest"),("small","smallest")],
    },
    "antonyms_sup_speed": {
        "type": "TYPE_ANTONYM", "attribute": "speed",
        "train": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                  ("rapid","gradual"),("hasty","leisurely")],
        "test":  [("brisk","sluggish"),("speedy","dawdling")],
    },
    "antonyms_sup_size": {
        "type": "TYPE_ANTONYM", "attribute": "size",
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "antonyms_unsup": {
        "type": "TYPE_ADJACENT", "attribute": None,
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "no_change_verbs": {
        "type": "IDENTITY", "attribute": None,
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost")],
    },
}

EXTRA_ANTONYM_AXES = {
    "speed":       [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                    ("rapid","gradual"),("hasty","leisurely")],
    "weight":      [("heavy","light"),("massive","weightless"),("dense","sparse"),
                    ("weighty","featherweight"),("hefty","flimsy")],
    "roughness":   [("rough","smooth"),("coarse","fine"),("jagged","polished"),
                    ("scratchy","silky"),("rugged","sleek")],
    "size":        [("big","small"),("large","tiny"),("huge","little"),
                    ("tall","short"),("wide","narrow"),("thick","thin")],
    "brightness":  [("bright","dark"),("light","dim"),("vivid","dull"),
                    ("radiant","murky"),("shining","gloomy")],
    "temperature": [("hot","cold"),("warm","cool"),("burning","freezing"),
                    ("boiling","icy"),("scorching","chilly")],
    "loudness":    [("loud","quiet"),("noisy","silent"),("deafening","hushed"),
                    ("boisterous","muted"),("thunderous","whispered")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a, dtype=np.float64)),
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
def ok_pairs(pairs):
    return [(a, b) for a, b in pairs
            if get_emb(a) is not None and get_emb(b) is not None]

print("Building normed matrix ...")
pool_words = []
pool_embs  = []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w)
        pool_embs.append(W_E[tid].astype(np.float32))
for d in "123456789":
    t = tid1_bare(d)
    if t and d not in pool_words:
        pool_words.append(d)
        pool_embs.append(W_E[t].astype(np.float32))
for cfg in ALL_DOMAINS.values():
    for pairs_key in ("train", "test"):
        for a, b in cfg[pairs_key]:
            for w in (a, b):
                if w not in pool_words:
                    e = get_emb(w)
                    if e is not None:
                        pool_words.append(w)
                        pool_embs.append(e.astype(np.float32))
for pairs in EXTRA_ANTONYM_AXES.values():
    for a, b in pairs:
        for w in (a, b):
            if w not in pool_words:
                e = get_emb(w)
                if e is not None:
                    pool_words.append(w)
                    pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E = np.array(pool_embs, dtype=np.float32)
norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
E_normed = (E / norms).astype(np.float32)
word_to_idx = {w: i for i, w in enumerate(pool_words)}
print(f"  Pool: {N} tokens\n")

def top_k_sims(query_emb, k=10, exclude=None):
    qn = normed(query_emb).astype(np.float32)
    sims = E_normed @ qn
    order = np.argsort(-sims)
    results = []
    for idx in order:
        w = pool_words[idx]
        if exclude and w == exclude: continue
        results.append((w, float(sims[idx])))
        if len(results) >= k: break
    return results

def rank_gaps(query_emb, ks=(2, 3, 5, 10), exclude=None):
    top = top_k_sims(query_emb, k=max(ks) + 1, exclude=exclude)
    if not top: return {}
    s0 = top[0][1]
    return {f"gap_1v{k}": s0 - top[k-1][1] if len(top) >= k else 0.0 for k in ks}

def build_axis(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return None
    diffs = [normed(get_emb(a) - get_emb(b)) for a, b in p]
    return normed(np.mean(diffs, axis=0))

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    return normed(np.mean([normed(get_emb(b) - get_emb(a)) for a, b in p], axis=0))

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b) - get_emb(a)) for a, b in p]
    pw = [cosine(diffs[i], diffs[j])
          for i in range(len(diffs)) for j in range(i + 1, len(diffs))]
    return float(np.mean(pw))

# Build all axes
antonym_axes = {attr: build_axis(pairs) for attr, pairs in EXTRA_ANTONYM_AXES.items()}

def query_terminal(src, mdir_or_axis, is_axis=False):
    se = get_emb(src)
    if se is None: return None
    if is_axis:
        proj = float(np.dot(normed(se), mdir_or_axis))
        tdir = mdir_or_axis if proj < 0 else -mdir_or_axis
        return normed(se + tdir)
    return normed(se + mdir_or_axis)

# ── Part A: Rank gap per pair, all domains ────────────────────────────
print("=" * 70)
print("PART A: Rank gap for every pair in every domain")
print("=" * 70)
print()

all_pair_records = []

for dname, cfg in ALL_DOMAINS.items():
    dtype   = cfg["type"]
    attr    = cfg["attribute"]
    train_p = ok_pairs(cfg["train"])
    test_p  = ok_pairs(cfg["test"])
    all_pairs_here = train_p + test_p

    # Determine retrieval mechanism
    if dtype == "IDENTITY":
        mdir = None; axis = None
    elif dtype == "TYPE_BC":
        mdir = mean_dir(cfg["train"]); axis = None
    elif dtype == "TYPE_ANTONYM":
        axis = antonym_axes.get(attr); mdir = None
    else:  # TYPE_ADJACENT
        mdir = None; axis = None

    print(f"  {dname} [{dtype}]:")
    print(f"    {'src':<12}  {'tgt':<12}  {'pred':<12}  {'ok':>3}  "
          f"{'gap_1v2':>8}  {'gap_1v3':>8}  {'gap_1v5':>8}  {'top1_sim':>9}")

    for a, b in all_pairs_here:
        set_label = "train" if (a, b) in train_p else "test"
        se = get_emb(a)
        if se is None: continue

        if dtype == "IDENTITY":
            qt = normed(se)
        elif dtype == "TYPE_BC" and mdir is not None:
            qt = query_terminal(a, mdir, is_axis=False)
        elif dtype == "TYPE_ANTONYM" and axis is not None:
            qt = query_terminal(a, axis, is_axis=True)
        else:
            qt = normed(se)

        top = top_k_sims(qt, k=10, exclude=a)
        pred = top[0][0] if top else None
        ok   = (pred == b)

        gaps = rank_gaps(qt, ks=(2, 3, 5, 10), exclude=a)
        top1_sim = top[0][1] if top else 0.0

        print(f"    {a:<12}  {b:<12}  {str(pred):<12}  {'OK' if ok else '  ':>3}  "
              f"{gaps.get('gap_1v2',0):>8.4f}  {gaps.get('gap_1v3',0):>8.4f}  "
              f"{gaps.get('gap_1v5',0):>8.4f}  {top1_sim:>9.4f}  [{set_label}]")

        all_pair_records.append({
            "domain": dname, "type": dtype, "attribute": attr,
            "src": a, "tgt": b, "pred": pred, "correct": ok,
            "set": set_label,
            "gap_1v2": gaps.get("gap_1v2", 0),
            "gap_1v3": gaps.get("gap_1v3", 0),
            "gap_1v5": gaps.get("gap_1v5", 0),
            "gap_1v10": gaps.get("gap_1v10", 0),
            "top1_sim": top1_sim,
        })
    print()

# ── Part B: Distributions by category ────────────────────────────────
print("=" * 70)
print("PART B: Rank gap distributions by category")
print("=" * 70)
print()

cats = {
    "TYPE_BC":          [r for r in all_pair_records if r["type"] == "TYPE_BC"],
    "ANTONYM_low_deg":  [r for r in all_pair_records
                         if r["type"] == "TYPE_ANTONYM"
                         and r["attribute"] in ("speed",)],
    "ANTONYM_high_deg": [r for r in all_pair_records
                         if r["type"] == "TYPE_ANTONYM"
                         and r["attribute"] == "size"],
    "TYPE_ADJACENT":    [r for r in all_pair_records if r["type"] == "TYPE_ADJACENT"],
    "IDENTITY":         [r for r in all_pair_records if r["type"] == "IDENTITY"],
}

print(f"  {'category':<22}  {'n':>5}  {'mean_gap_1v2':>13}  {'mean_gap_1v5':>13}  "
      f"{'mean_top1':>10}  acc")
for cat, recs in cats.items():
    if not recs: continue
    g2  = float(np.mean([r["gap_1v2"]  for r in recs]))
    g5  = float(np.mean([r["gap_1v5"]  for r in recs]))
    t1  = float(np.mean([r["top1_sim"] for r in recs]))
    acc = sum(1 for r in recs if r["correct"]) / len(recs)
    print(f"  {cat:<22}  {len(recs):>5}  {g2:>13.4f}  {g5:>13.4f}  {t1:>10.4f}  {acc:.3f}")

# ── Part C: Gap vs acc by attribute ──────────────────────────────────
print()
print("=" * 70)
print("PART C: Rank gap by attribute (extra antonym axes, training pairs only)")
print("=" * 70)
print()
print(f"  {'attr':<14}  {'train_acc':>9}  {'mean_gap_1v2':>13}  "
      f"{'gap_correct':>12}  {'gap_wrong':>10}")

attr_gap_results = {}
for attr, pairs in EXTRA_ANTONYM_AXES.items():
    axis = antonym_axes.get(attr)
    if axis is None: continue
    p = ok_pairs(pairs)
    correct_gaps = []; wrong_gaps = []
    correct = 0
    for a, b in p:
        qt = query_terminal(a, axis, is_axis=True)
        if qt is None: continue
        top = top_k_sims(qt, k=5, exclude=a)
        pred = top[0][0] if top else None
        gap = top[0][1] - top[1][1] if len(top) >= 2 else 0.0
        if pred == b:
            correct += 1; correct_gaps.append(gap)
        else:
            wrong_gaps.append(gap)
    train_acc = correct / len(p) if p else 0.0
    mc = float(np.mean(correct_gaps)) if correct_gaps else 0.0
    mw = float(np.mean(wrong_gaps))   if wrong_gaps   else 0.0
    print(f"  {attr:<14}  {train_acc:>9.3f}  {float(np.mean(correct_gaps+wrong_gaps)):>13.4f}  "
          f"{mc:>12.4f}  {mw:>10.4f}")
    attr_gap_results[attr] = {"train_acc": train_acc,
                               "mean_gap_correct": mc, "mean_gap_wrong": mw}

# ── Part D: Optimal gap threshold ────────────────────────────────────
print()
print("=" * 70)
print("PART D: Optimal gap_1v2 threshold for confidence signal")
print("=" * 70)
print()

correct_g = [r["gap_1v2"] for r in all_pair_records if r["correct"]]
wrong_g   = [r["gap_1v2"] for r in all_pair_records if not r["correct"]]
print(f"  Correct pairs n={len(correct_g)}: mean={np.mean(correct_g):.4f} "
      f"median={np.median(correct_g):.4f} min={np.min(correct_g):.4f}")
print(f"  Wrong   pairs n={len(wrong_g)}:  mean={np.mean(wrong_g):.4f} "
      f"median={np.median(wrong_g):.4f} min={np.min(wrong_g):.4f}")
print()

# Scan thresholds
thresholds = sorted(set(r["gap_1v2"] for r in all_pair_records))
best_t = None; best_f1 = 0; best_results = None
print(f"  threshold  tp  fp  tn  fn  precision  recall  f1")
printed = set()
for t in np.arange(0.00, 0.15, 0.005):
    tp = sum(1 for r in all_pair_records if r["gap_1v2"] >= t and r["correct"])
    fp = sum(1 for r in all_pair_records if r["gap_1v2"] >= t and not r["correct"])
    tn = sum(1 for r in all_pair_records if r["gap_1v2"] < t and not r["correct"])
    fn = sum(1 for r in all_pair_records if r["gap_1v2"] < t and r["correct"])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    key = (tp, fp, tn, fn)
    if key not in printed:
        printed.add(key)
        print(f"  t={t:.3f}:    {tp:3d}  {fp:3d}  {tn:3d}  {fn:3d}  "
              f"{prec:>9.3f}  {rec:>6.3f}  {f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1; best_t = t
        best_results = (tp, fp, tn, fn, prec, rec)

print()
if best_t is not None and best_results:
    tp, fp, tn, fn, prec, rec = best_results
    print(f"  Best threshold: t={best_t:.3f}  F1={best_f1:.3f}  "
          f"precision={prec:.3f}  recall={rec:.3f}")
    print(f"  tp={tp}  fp={fp}  tn={tn}  fn={fn}")
    print(f"  Interpretation: gap >= {best_t:.3f} -> predict CORRECT")
    print(f"  Coverage: {tp+fp}/{len(all_pair_records)} pairs flagged as reliable")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  Rank gap = sim(rank-1) - sim(rank-2) at query terminal")
print("  Higher gap -> more unique top-1 -> more reliable retrieval")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "pair_records": all_pair_records,
        "attr_gap_results": attr_gap_results,
        "best_threshold": best_t,
        "best_f1": best_f1,
    }, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 231 complete.")
