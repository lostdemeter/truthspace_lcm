#!/usr/bin/env python3
"""
Day 226 — Pair-Lookup Augmentation + Axis Quality Threshold + Pipeline v5

DC 378 concluded:
  1. axis_align > 0.70 is the reliability threshold for TYPE_ANTONYM
  2. Tight target synonym clusters cause axis centroid collapse
  3. pair-lookup (memorise known B for known A) can recover training pairs
     but this is a non-geometric fallback

This experiment:
  A. Measure target cluster tightness for each attribute axis
     tightness = mean pairwise cosine among B-words in training pairs
     Prediction: size high tightness -> fails; speed low tightness -> works

  B. Test pair-lookup augmented TYPE_ANTONYM:
     If source is in training set: return stored B directly
     Else: use axis retrieval
     Measure: does this improve acc on training pairs without geometric cost?

  C. Build pipeline v5 incorporating all fixes:
     - v4b routing (attribute before dc)
     - pair-lookup augmentation for TYPE_ANTONYM with axis_align < 0.70
     - cross-dc runtime validation signal
     Evaluate on full 42k domain set

  D. Re-examine if speed TYPE_ANTONYM unseen test pairs succeed at full vocab.
     Known: brisk->sluggish  speedy->dawdling (test pairs)
     Does the speed axis retrieve correctly at 42k?

  E. Establish the final honest accuracy ceiling after all fixes.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day226_antonym_v5.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Full domain set for pipeline v5
DOMAINS = {
    "capitals": {
        "type": "TYPE_BC",
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi")],
    },
    "gender": {
        "type": "TYPE_BC",
        "train": [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress"),("hero","heroine")],
        "test":  [("father","mother"),("brother","sister"),("son","daughter"),
                  ("husband","wife"),("uncle","aunt"),("waiter","waitress")],
    },
    "plurals": {
        "type": "TYPE_BC",
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "superlative": {
        "type": "TYPE_BC",
        "train": [("big","biggest"),("fast","fastest"),("long","longest"),
                  ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        "test":  [("hard","hardest"),("dark","darkest"),("soft","softest"),
                  ("warm","warmest"),("slow","slowest"),("small","smallest")],
    },
    "past_tense_F": {
        "type": "TYPE_BC",
        "train": [("go","went"),("come","came"),("get","got"),("run","ran"),
                  ("eat","ate"),("see","saw"),("drive","drove"),("take","took"),
                  ("give","gave"),("make","made")],
        "test":  [("ride","rode"),("write","wrote"),("draw","drew"),
                  ("know","knew"),("grow","grew"),("choose","chose"),
                  ("wake","woke"),("shake","shook"),("break","broke"),
                  ("steal","stole")],
    },
    "past_tense_E": {
        "type": "TYPE_BC",
        "train": [("stand","stood"),("leave","left"),("bring","brought"),
                  ("buy","bought"),("keep","kept"),("feel","felt")],
        "test":  [("sleep","slept"),("sweep","swept"),("deal","dealt"),
                  ("mean","meant")],
    },
    "past_tense_D": {
        "type": "TYPE_BC",
        "train": [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
        "test":  [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
    },
    "past_tense_B": {
        "type": "TYPE_BC",
        "train": [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
        "test":  [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
    },
    "numbers": {
        "type": "TYPE_BC",
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6")],
        "test":  [("seven","7"),("eight","8"),("nine","9")],
    },
    "antonyms_unsup": {
        "type": "TYPE_ADJACENT",
        "attribute": None,
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "antonyms_sup_size": {
        "type": "TYPE_ANTONYM",
        "attribute": "size",
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "antonyms_sup_speed": {
        "type": "TYPE_ANTONYM",
        "attribute": "speed",
        "train": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                  ("rapid","gradual"),("hasty","leisurely")],
        "test":  [("brisk","sluggish"),("speedy","dawdling")],
    },
    "no_change_verbs": {
        "type": "IDENTITY",
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost")],
    },
}

ANTONYM_AXES_DEF = {
    "size":  [("big","small"),("large","tiny"),("huge","little"),
              ("tall","short"),("wide","narrow"),("thick","thin")],
    "speed": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
              ("rapid","gradual"),("hasty","leisurely")],
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

# Build axes + measure axis_align and target tightness
antonym_axes   = {}
axis_quality   = {}

for attr, pairs in ANTONYM_AXES_DEF.items():
    p = ok_pairs(pairs)
    if not p: continue
    diffs = [normed(get_emb(a)-get_emb(b)) for a,b in p]
    axis  = normed(np.mean(diffs, axis=0))
    antonym_axes[attr] = axis

    # axis_align = mean cosine(each diff, axis)
    align = float(np.mean([cosine(d, axis) for d in diffs]))

    # target cluster tightness = mean pairwise cosine among B-words
    b_embs = [normed(get_emb(b)) for _,b in p]
    tightness_vals = [cosine(b_embs[i], b_embs[j])
                      for i in range(len(b_embs))
                      for j in range(i+1, len(b_embs))]
    tightness = float(np.mean(tightness_vals)) if tightness_vals else 0.0

    axis_quality[attr] = {"axis_align": align, "target_tightness": tightness}

print("AXIS QUALITY METRICS:")
print(f"  {'attr':<12}  {'axis_align':>10}  {'tgt_tightness':>13}  {'prediction':>12}")
for attr, q in axis_quality.items():
    aa = q["axis_align"]; tt = q["target_tightness"]
    prediction = "RELIABLE" if aa > 0.70 else ("PARTIAL" if aa > 0.55 else "FAILS")
    print(f"  {attr:<12}  {aa:>10.3f}  {tt:>13.3f}  {prediction:>12}")
print()

# ── Retrieval functions ───────────────────────────────────────────────
def retrieve_bc(src, mdir):
    se = get_emb(src)
    if se is None: return None
    sims = [(w, cosine(se+mdir, e)) for w,e in all_pool.items() if w != src]
    return max(sims, key=lambda x: x[1])[0] if sims else None

def retrieve_nn(src):
    se = get_emb(src)
    if se is None: return None
    sims = [(w, cosine(se, e)) for w,e in all_pool.items() if w != src]
    return max(sims, key=lambda x: x[1])[0] if sims else None

def retrieve_axis(src, attribute):
    se = get_emb(src)
    if se is None or attribute not in antonym_axes: return retrieve_nn(src)
    axis = antonym_axes[attribute]
    proj = float(np.dot(normed(se), axis))
    tdir = axis if proj < 0 else -axis
    sims = [(w, cosine(normed(se+tdir), e)) for w,e in all_pool.items() if w != src]
    return max(sims, key=lambda x: x[1])[0] if sims else None

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    return normed(np.mean([normed(get_emb(b)-get_emb(a)) for a,b in p], axis=0))

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b)-get_emb(a)) for a,b in p]
    pw = [cosine(diffs[i],diffs[j])
          for i in range(len(diffs)) for j in range(i+1,len(diffs))]
    return float(np.mean(pw))

# ── Part A: Target cluster tightness confirmation ─────────────────────
print("=" * 70)
print("PART A: Target cluster tightness vs axis failure")
print("=" * 70)
print()
print("  Prediction: high target tightness -> axis centroid collapse -> fails")
for attr, q in axis_quality.items():
    aa = q["axis_align"]; tt = q["target_tightness"]
    p  = ok_pairs(ANTONYM_AXES_DEF[attr])
    targets = [b for _,b in p]
    print(f"\n  {attr}:  axis_align={aa:.3f}  target_tightness={tt:.3f}")
    print(f"    targets: {targets}")

# ── Part B: Pair-lookup augmented retrieval ───────────────────────────
print()
print("=" * 70)
print("PART B: Pair-lookup augmented TYPE_ANTONYM (training pairs)")
print("=" * 70)

for attr, pairs_def in ANTONYM_AXES_DEF.items():
    p = ok_pairs(pairs_def)
    lookup = {a: b for a,b in p}  # memorise training pairs

    axis_correct   = 0
    lookup_correct = 0
    aa = axis_quality[attr]["axis_align"]
    print(f"\n  {attr} (axis_align={aa:.3f}):")
    print(f"    {'src':<12} {'tgt':<12} {'axis_pred':<12} {'lookup_pred':<12}  axis  lookup")
    for src, tgt in p:
        axis_pred   = retrieve_axis(src, attr)
        lookup_pred = lookup.get(src, axis_pred)
        a_ok = axis_pred == tgt
        l_ok = lookup_pred == tgt
        if a_ok:  axis_correct   += 1
        if l_ok:  lookup_correct += 1
        print(f"    {src:<12} {tgt:<12} {str(axis_pred):<12} {str(lookup_pred):<12}  "
              f"{'OK' if a_ok else '  '}    {'OK' if l_ok else ''}")
    print(f"    axis={axis_correct}/{len(p)}  lookup={lookup_correct}/{len(p)}")

# ── Part C: Pipeline v5 full evaluation ──────────────────────────────
print()
print("=" * 70)
print("PART C: Pipeline v5 full evaluation (42k vocab)")
print("=" * 70)

AXIS_ALIGN_THRESHOLD = 0.70  # use pair-lookup if axis_align < threshold

def classify_v5(train_pairs, attribute=None):
    p = ok_pairs(train_pairs)
    if any(a == b for a,b in p): return "IDENTITY"
    if attribute is not None and attribute in antonym_axes: return "TYPE_ANTONYM"
    if len(p) >= 2 and dir_consistency(p) > 0.10: return "TYPE_BC"
    return "TYPE_ADJACENT"

def eval_domain_v5(train_pairs, test_pairs, pred, attribute=None):
    p_test = ok_pairs(test_pairs) or ok_pairs(train_pairs)
    if not p_test: return 0.0, 0, 0

    # Build lookup for known training pairs
    train_lookup = {a: b for a,b in ok_pairs(train_pairs)}
    mdir = mean_dir(train_pairs) if pred == "TYPE_BC" else None
    aa   = axis_quality.get(attribute, {}).get("axis_align", 0.0) if attribute else 0.0
    use_lookup = (pred == "TYPE_ANTONYM" and aa < AXIS_ALIGN_THRESHOLD)

    correct = 0
    for src, tgt in p_test:
        if pred == "IDENTITY":
            p_tgt = src
        elif pred == "TYPE_BC" and mdir is not None:
            p_tgt = retrieve_bc(src, mdir)
        elif pred == "TYPE_ANTONYM":
            if use_lookup and src in train_lookup:
                p_tgt = train_lookup[src]
            else:
                p_tgt = retrieve_axis(src, attribute)
        else:
            p_tgt = retrieve_nn(src)
        if p_tgt == tgt: correct += 1
    return correct / len(p_test), correct, len(p_test)

print(f"\n  {'Domain':<22}  {'Type':<14}  {'dc':>6}  "
      f"{'aa':>6}  {'acc_v4b':>7}  {'acc_v5':>7}  note")
print()

total_v5 = 0; total_n = 0
v5_results = {}
for name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]
    expected = cfg["type"]
    attribute = cfg.get("attribute", None)

    pred = classify_v5(train, attribute)
    dc   = dir_consistency(train)
    aa   = axis_quality.get(attribute, {}).get("axis_align", 0.0) if attribute else 0.0
    acc, c, n = eval_domain_v5(train, test, pred, attribute)
    total_v5 += c; total_n += n

    # v4b reference acc (from Day 222)
    V4B_REF = {
        "capitals": 1.000, "gender": 1.000, "plurals": 0.833,
        "superlative": 1.000, "past_tense_F": 1.000, "past_tense_E": 0.750,
        "past_tense_D": 1.000, "past_tense_B": 1.000, "numbers": 1.000,
        "antonyms_unsup": 0.000, "antonyms_sup_size": 0.333,
        "antonyms_sup_speed": "NEW", "no_change_verbs": 1.000,
    }
    v4b = V4B_REF.get(name, "?")
    delta = ""
    if isinstance(v4b, float):
        d = acc - v4b
        delta = f"+{d:.3f}" if d > 0 else (f"{d:.3f}" if d < 0 else " same")

    use_lookup = (pred == "TYPE_ANTONYM" and aa < AXIS_ALIGN_THRESHOLD and aa > 0)
    note = "lookup" if use_lookup else ""
    print(f"  {name:<22}  {pred:<14}  {dc:>6.3f}  "
          f"{aa:>6.3f}  {str(v4b):>7}  {acc:>7.3f}  {delta}  {note}")
    v5_results[name] = {"pred": pred, "acc": acc, "n": n,
                         "dc": dc, "axis_align": aa}

overall_v5 = total_v5 / total_n if total_n else 0
print(f"\n  v5 OVERALL: {total_v5}/{total_n} = {overall_v5:.3f}")

# ── Part D: Speed axis test pair detail ──────────────────────────────
print()
print("=" * 70)
print("PART D: Speed axis test pairs in detail")
print("=" * 70)
speed_test = [("brisk","sluggish"),("speedy","dawdling")]
speed_p = ok_pairs(speed_test)
speed_aa = axis_quality.get("speed",{}).get("axis_align",0.0)
print(f"\n  speed axis_align={speed_aa:.3f}  test pairs: {speed_p}")
for src, tgt in speed_p:
    pred_v5 = retrieve_axis(src, "speed")
    se = get_emb(src)
    axis = antonym_axes["speed"]
    proj = float(np.dot(normed(se), axis))
    sims = [(w, cosine(normed(se + (axis if proj<0 else -axis)), e))
            for w,e in all_pool.items() if w != src]
    sims.sort(key=lambda x: x[1], reverse=True)
    rank = next((i for i,(w,_) in enumerate(sims) if w==tgt), -1)
    top5 = [w for w,_ in sims[:5]]
    print(f"  {src} -> {tgt}: pred={pred_v5}  rank={rank}  top5={top5}")

print()
print("=" * 70)
print("SUMMARY: Pipeline v5 vs v4b")
print("=" * 70)
print(f"\n  v4b: 49/59 = 0.831  (12 domains, 42k vocab)")
print(f"  v5:  {total_v5}/{total_n} = {overall_v5:.3f}  (13 domains, 42k vocab)")
print()
print("  Changes:")
print(f"    + antonyms_sup_speed: NEW domain (speed TYPE_ANTONYM)")
print(f"    + antonyms_sup_size: pair-lookup augment (axis_align={axis_quality['size']['axis_align']:.3f} < 0.70)")
print(f"    Target tightness:  size={axis_quality['size']['target_tightness']:.3f}  speed={axis_quality['speed']['target_tightness']:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"axis_quality": axis_quality, "v5_results": v5_results,
               "v5_overall": overall_v5}, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 226 complete.")
