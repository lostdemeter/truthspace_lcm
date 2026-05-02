#!/usr/bin/env python3
"""
Day 222 — Cross-DC Generalisation Threshold Scan

Day 221 introduced cross-dc: mean cosine(train_mean_dir, per-test-pair
displacements). It measures whether the training direction generalises
to test pairs. Observed so far:
  past_tense_F: cross-dc=0.436 -> acc=1.000
  past_tense_E: cross-dc=0.216 -> acc=0.750

Questions:
  1. What is the cross-dc for all 12 domains?
  2. Is cross-dc a better predictor of full-vocab acc than dc_train alone?
  3. What cross-dc threshold separates reliable from unreliable domains?
  4. How does cross-dc change as k (number of training pairs) increases?
     With k=2 training pairs, is cross-dc already predictive?

Method:
  For each domain with test pairs:
    - Compute dc_train (training direction consistency)
    - Compute cross-dc at k=2,3,4,5,6 training pairs
    - Measure full-vocab acc at each k
  Build: cross-dc vs acc scatter plot data
  Find: minimum cross-dc for acc >= 0.75 (empirical threshold)

Also: apply the antonyms_sup fix (check attribute BEFORE dc) and
measure the improvement.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day222_crossdc_scan.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAINS = {
    "capitals": {
        "expected": "TYPE_BC",
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi")],
    },
    "gender": {
        "expected": "TYPE_BC",
        "train": [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress"),("hero","heroine")],
        "test":  [("father","mother"),("brother","sister"),("son","daughter"),
                  ("husband","wife"),("uncle","aunt"),("waiter","waitress")],
    },
    "plurals": {
        "expected": "TYPE_BC",
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "superlative": {
        "expected": "TYPE_BC",
        "train": [("big","biggest"),("fast","fastest"),("long","longest"),
                  ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        "test":  [("hard","hardest"),("dark","darkest"),("soft","softest"),
                  ("warm","warmest"),("slow","slowest"),("small","smallest")],
    },
    "past_tense_F": {
        "expected": "TYPE_BC",
        "train": [("go","went"),("come","came"),("get","got"),("run","ran"),
                  ("eat","ate"),("see","saw"),("drive","drove"),("take","took"),
                  ("give","gave"),("make","made")],
        "test":  [("ride","rode"),("write","wrote"),("draw","drew"),
                  ("know","knew"),("grow","grew"),("choose","chose"),
                  ("wake","woke"),("shake","shook"),("break","broke"),
                  ("steal","stole")],
    },
    "past_tense_E": {
        "expected": "TYPE_BC",
        "train": [("stand","stood"),("leave","left"),("bring","brought"),
                  ("buy","bought"),("keep","kept"),("feel","felt")],
        "test":  [("sleep","slept"),("sweep","swept"),("deal","dealt"),
                  ("mean","meant")],
    },
    "past_tense_D": {
        "expected": "TYPE_BC",
        "train": [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
        "test":  [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
    },
    "past_tense_B": {
        "expected": "TYPE_BC",
        "train": [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
        "test":  [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
    },
    "numbers": {
        "expected": "TYPE_BC",
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6")],
        "test":  [("seven","7"),("eight","8"),("nine","9")],
    },
    "antonyms_unsup": {
        "expected": "TYPE_ADJACENT",
        "attribute": None,
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "antonyms_sup_size": {
        "expected": "TYPE_ANTONYM",
        "attribute": "size",
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "no_change_verbs": {
        "expected": "IDENTITY",
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost")],
    },
}

ANTONYM_AXES_DEF = {
    "size": [("big","small"),("large","tiny"),("huge","little"),
             ("tall","short"),("wide","narrow"),("thick","thin")],
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

def cross_dc(train_pairs, test_pairs):
    mdir = mean_dir(train_pairs)
    if mdir is None: return 0.0
    p_test = ok_pairs(test_pairs)
    if not p_test: return 0.0
    test_diffs = [normed(get_emb(b)-get_emb(a)) for a,b in p_test]
    return float(np.mean([cosine(mdir, d) for d in test_diffs]))

# Build full pool
print("Building vocab pool ...")
all_pool = {}
for token_id in range(V):
    decoded = tok.decode([token_id])
    if not decoded.startswith(" "): continue
    word = decoded[1:]
    if not word.isalpha() or len(word) < 2: continue
    if word.islower() or (word[0].isupper() and word[1:].islower()):
        all_pool[word] = W_E[token_id].astype(np.float64)
for d in "123456789":
    t = tid1_bare(d)
    if t: all_pool[d] = W_E[t].astype(np.float64)
for cfg in DOMAINS.values():
    for a,b in cfg["train"]+cfg["test"]:
        for w in (a,b):
            if w not in all_pool:
                e = get_emb(w)
                if e is not None: all_pool[w] = e
print(f"  Pool: {len(all_pool)} tokens\n")

antonym_axes = {}
for attr, pairs in ANTONYM_AXES_DEF.items():
    p = ok_pairs(pairs)
    diffs = [normed(get_emb(a)-get_emb(b)) for a,b in p if get_emb(a) is not None and get_emb(b) is not None]
    if diffs: antonym_axes[attr] = normed(np.mean(diffs, axis=0))

def retrieve_bc(src, mdir):
    se = get_emb(src)
    if se is None: return None
    q = se + mdir
    sims = [(w, cosine(q, e)) for w,e in all_pool.items() if w != src]
    return max(sims, key=lambda x: x[1])[0] if sims else None

def retrieve_nn(src):
    se = get_emb(src)
    if se is None: return None
    sims = [(w, cosine(se, e)) for w,e in all_pool.items() if w != src]
    return max(sims, key=lambda x: x[1])[0] if sims else None

def retrieve_antonym(src, attribute):
    se = get_emb(src)
    if se is None or attribute not in antonym_axes: return retrieve_nn(src)
    axis = antonym_axes[attribute]
    proj = float(np.dot(normed(se), axis))
    tdir = axis if proj < 0 else -axis
    q = normed(se + tdir)
    sims = [(w, cosine(q, e)) for w,e in all_pool.items() if w != src]
    return max(sims, key=lambda x: x[1])[0] if sims else None

# v4b classifier: check attribute FIRST (fixes antonyms_sup_size)
def classify_v4b(train_pairs, attribute=None):
    p = ok_pairs(train_pairs)
    if any(a == b for a,b in p):
        return "IDENTITY"
    # ATTRIBUTE CHECK FIRST
    if attribute is not None and attribute in antonym_axes:
        return "TYPE_ANTONYM"
    if len(p) >= 2 and dir_consistency(p) > 0.10:
        return "TYPE_BC"
    return "TYPE_ADJACENT"

def eval_domain(train_sub, test_pairs, pred, attribute=None):
    mdir = mean_dir(train_sub) if pred == "TYPE_BC" else None
    p_test = ok_pairs(test_pairs)
    if not p_test: return 0.0, 0, 0
    correct = 0
    for src, tgt in p_test:
        if pred == "IDENTITY":
            p_tgt = src
        elif pred == "TYPE_BC" and mdir is not None:
            p_tgt = retrieve_bc(src, mdir)
        elif pred == "TYPE_ANTONYM" and attribute:
            p_tgt = retrieve_antonym(src, attribute)
        else:
            p_tgt = retrieve_nn(src)
        if p_tgt == tgt: correct += 1
    return correct / len(p_test), correct, len(p_test)

# ── Part 1: Cross-DC per domain (full training set) ───────────────────
print("=" * 70)
print("PART 1: Cross-DC per domain (full training set)")
print("=" * 70)
print(f"\n  {'Domain':<22}  {'dc_train':>8}  {'cross-dc':>8}  "
      f"{'acc_full':>8}  {'predictor?':>10}")
print()

domain_crossdc = {}
for name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]
    attribute = cfg.get("attribute", None)
    pred = classify_v4b(train, attribute)

    dc_t = dir_consistency(train)
    cdc  = cross_dc(train, test) if pred == "TYPE_BC" else 0.0
    acc_full, _, _ = eval_domain(train, test, pred, attribute)

    predictor = ""
    if pred == "TYPE_BC":
        if cdc > 0.15: predictor = "HIGH"
        elif cdc > 0.05: predictor = "MOD"
        else: predictor = "LOW"

    print(f"  {name:<22}  {dc_t:>8.3f}  {cdc:>8.3f}  "
          f"{acc_full:>8.3f}  {predictor:>10}")
    domain_crossdc[name] = {"dc_train": dc_t, "cross_dc": cdc,
                             "acc_full": acc_full, "pred": pred}

# ── Part 2: Cross-DC vs accuracy for k = 2..full ─────────────────────
print()
print("=" * 70)
print("PART 2: Cross-DC and accuracy as k training pairs increases")
print("=" * 70)

k_scan_domains = {
    "capitals":       DOMAINS["capitals"],
    "gender":         DOMAINS["gender"],
    "past_tense_F":   DOMAINS["past_tense_F"],
    "past_tense_E":   DOMAINS["past_tense_E"],
    "past_tense_D":   DOMAINS["past_tense_D"],
    "superlative":    DOMAINS["superlative"],
    "numbers":        DOMAINS["numbers"],
    "antonyms_unsup": DOMAINS["antonyms_unsup"],
}

k_results = {}
for name, cfg in k_scan_domains.items():
    train = ok_pairs(cfg["train"])
    test  = cfg["test"]
    attribute = cfg.get("attribute", None)
    max_k = len(train)

    print(f"\n  {name} (max_k={max_k}):")
    print(f"    {'k':>4}  {'dc_k':>7}  {'cross-dc':>8}  {'acc':>6}")
    domain_k = {}
    for k in range(2, min(max_k+1, 11)):
        # Use first k training pairs (ordered)
        train_k = train[:k]
        dc_k    = dir_consistency(train_k)
        cdc_k   = cross_dc(train_k, test)
        pred_k  = classify_v4b(train_k, attribute)
        acc_k, _, _ = eval_domain(train_k, test, pred_k, attribute)
        print(f"    {k:>4}  {dc_k:>7.3f}  {cdc_k:>8.3f}  {acc_k:>6.3f}")
        domain_k[k] = {"dc": dc_k, "cross_dc": cdc_k, "acc": acc_k, "pred": pred_k}
    k_results[name] = domain_k

# ── Part 3: Full v4b evaluation (attribute check first) ───────────────
print()
print("=" * 70)
print("PART 3: v4b full evaluation (attribute check BEFORE dc check)")
print("=" * 70)
print(f"\n  {'Domain':<22}  {'Expected':<14}  {'Predicted':<14}  "
      f"{'dc':>6}  {'cross-dc':>8}  {'acc':>6}")
print()

total_c = 0; total_n = 0
v4b_results = {}
for name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]
    attribute = cfg.get("attribute", None)
    expected = cfg["expected"]

    pred = classify_v4b(train, attribute)
    dc_t = dir_consistency(train)
    cdc  = cross_dc(train, test) if pred == "TYPE_BC" else 0.0
    acc, c, n = eval_domain(train, test, pred, attribute)
    total_c += c; total_n += n
    match = pred == expected
    print(f"  {name:<22}  {expected:<14}  {pred:<14}  "
          f"{dc_t:>6.3f}  {cdc:>8.3f}  {acc:>6.3f}"
          + ("" if match else "  X"))
    v4b_results[name] = {"expected": expected, "predicted": pred,
                          "dc_train": dc_t, "cross_dc": cdc,
                          "acc": acc, "correct_cls": match}

overall = total_c / total_n if total_n else 0
cls_c   = sum(1 for d in v4b_results.values() if d["correct_cls"])
print(f"\n  v4b OVERALL: {total_c}/{total_n} = {overall:.3f}")
print(f"  v4b Classification: {cls_c}/{len(v4b_results)} correct")

print()
print("=" * 70)
print("SUMMARY: Cross-DC Threshold Analysis")
print("=" * 70)
print("\n  Cross-DC vs Accuracy (TYPE_BC domains only, full training):")
bc_domains = [(n,d) for n,d in domain_crossdc.items() if d["pred"]=="TYPE_BC"]
bc_sorted  = sorted(bc_domains, key=lambda x: x[1]["cross_dc"])
for n,d in bc_sorted:
    print(f"    cross-dc={d['cross_dc']:.3f}  acc={d['acc_full']:.3f}  {n}")

# Find threshold
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
print("\n  Threshold sensitivity (cross-dc > T -> predict acc >= 0.75):")
for T in thresholds:
    hi = [(n,d) for n,d in bc_domains if d["cross_dc"] > T]
    lo = [(n,d) for n,d in bc_domains if d["cross_dc"] <= T]
    hi_good = sum(1 for n,d in hi if d["acc_full"] >= 0.75)
    lo_bad  = sum(1 for n,d in lo if d["acc_full"] < 0.75)
    tp = hi_good; fp = len(hi) - hi_good
    tn = lo_bad;  fn = len(lo) - lo_bad
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"    T={T:.2f}: TP={tp} FP={fp} TN={tn} FN={fn}  "
          f"prec={prec:.2f} rec={rec:.2f}")

output = {
    "domain_crossdc": domain_crossdc,
    "k_scan": k_results,
    "v4b_overall": overall,
    "v4b_results": v4b_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 222 complete.")
