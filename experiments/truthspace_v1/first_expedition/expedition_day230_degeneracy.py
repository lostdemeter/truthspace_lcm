#!/usr/bin/env python3
"""
Day 230 — Antonym Cluster Degeneracy Measurement

DC 380 proposed: target_degeneracy = |tokens with cos_sim > threshold
to the axis terminal| predicts TYPE_ANTONYM retrieval failure.

This experiment:
  A. For each attribute axis, measure degeneracy of the axis terminal
     at multiple thresholds (0.80, 0.85, 0.90, 0.95).
     Compare to observed train_acc. Find the threshold that best
     separates high-acc from low-acc attributes.

  B. Measure degeneracy per training pair (per-source):
     For each (A, B) in training: what is the degeneracy of the
     query terminal normed(emb(A) + target_dir)?
     Does per-source degeneracy predict whether THAT pair's retrieval
     succeeds or fails?

  C. Measure degeneracy for TYPE_BC domains:
     Expect very low degeneracy (Paris has no near-synonyms).
     Compare TYPE_BC degeneracy to TYPE_ANTONYM degeneracy.

  D. Scan threshold t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95] and
     find the value where degeneracy > k_t predicts failure and
     degeneracy <= k_t predicts success.
     Use ROC-style analysis across all pairs in all domains.

  E. Build a runtime degeneracy confidence signal:
     conf(A, attribute) = 1 / (1 + log(1 + degeneracy(A, t=0.85)))
     Higher degeneracy -> lower confidence.
     Does this correlate with retrieval accuracy?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day230_degeneracy.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ATTRIBUTE_AXES = {
    "speed":      [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                   ("rapid","gradual"),("hasty","leisurely")],
    "weight":     [("heavy","light"),("massive","weightless"),("dense","sparse"),
                   ("weighty","featherweight"),("hefty","flimsy")],
    "roughness":  [("rough","smooth"),("coarse","fine"),("jagged","polished"),
                   ("scratchy","silky"),("rugged","sleek")],
    "size":       [("big","small"),("large","tiny"),("huge","little"),
                   ("tall","short"),("wide","narrow"),("thick","thin")],
    "brightness": [("bright","dark"),("light","dim"),("vivid","dull"),
                   ("radiant","murky"),("shining","gloomy")],
    "temperature":[("hot","cold"),("warm","cool"),("burning","freezing"),
                   ("boiling","icy"),("scorching","chilly")],
    "loudness":   [("loud","quiet"),("noisy","silent"),("deafening","hushed"),
                   ("boisterous","muted"),("thunderous","whispered")],
}

BC_DOMAINS = {
    "capitals": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                 ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
    "gender":   [("king","queen"),("man","woman"),("boy","girl"),
                 ("prince","princess"),("actor","actress"),("hero","heroine")],
    "numbers":  [("one","1"),("two","2"),("three","3"),
                 ("four","4"),("five","5"),("six","6")],
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

print("Building vocab pool + normed matrix ...")
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

# Add all word tokens
for pairs in list(ATTRIBUTE_AXES.values()) + list(BC_DOMAINS.values()):
    for a,b in pairs:
        for w in (a,b):
            if w not in pool_words:
                e = get_emb(w)
                if e is not None:
                    pool_words.append(w)
                    pool_embs.append(e.astype(np.float32))
for d in "123456789":
    t = tid1_bare(d)
    if t and d not in pool_words:
        pool_words.append(d)
        pool_embs.append(W_E[t].astype(np.float32))

N = len(pool_words)
E = np.array(pool_embs, dtype=np.float32)
# Norm each row
norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
E_normed = (E / norms).astype(np.float32)
word_to_idx = {w: i for i,w in enumerate(pool_words)}
print(f"  Pool: {N} tokens\n")

def degeneracy(query_emb, threshold):
    """Count tokens with cos_sim > threshold to query (excluding query itself)."""
    qn = normed(query_emb).astype(np.float32)
    sims = E_normed @ qn  # (N,)
    return int(np.sum(sims > threshold))

def build_axis_and_mean_dir(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return None, None
    diffs = [normed(get_emb(a)-get_emb(b)) for a,b in p]
    axis  = normed(np.mean(diffs, axis=0))
    # mean_dir in the B direction (source -> target direction)
    mean_d = normed(np.mean([normed(get_emb(b)-get_emb(a)) for a,b in p], axis=0))
    return axis, mean_d

def query_terminal(src, axis):
    se = get_emb(src)
    if se is None or axis is None: return None
    proj = float(np.dot(normed(se), axis))
    tdir = axis if proj < 0 else -axis
    return normed(se + tdir)

def retrieval_acc(pairs, axis):
    p = ok_pairs(pairs)
    if not p: return 0.0, []
    correct = 0; detail = []
    for a,b in p:
        qt = query_terminal(a, axis)
        if qt is None:
            detail.append((a,b,False,None))
            continue
        sims = [(pool_words[i], float(E_normed[i] @ qt.astype(np.float32)))
                for i in range(N) if pool_words[i] != a]
        sims.sort(key=lambda x: x[1], reverse=True)
        pred = sims[0][0] if sims else None
        ok = (pred == b)
        if ok: correct += 1
        detail.append((a,b,ok,pred))
    return correct/len(p), detail

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

# ── Part A: Per-attribute degeneracy ─────────────────────────────────
print("=" * 70)
print("PART A: Per-attribute degeneracy at axis terminal centroid")
print("=" * 70)
print()
print(f"  {'attr':<14}  {'train_acc':>9}  {'deg@0.80':>8}  {'deg@0.85':>8}  "
      f"{'deg@0.90':>8}  {'deg@0.95':>8}")

attr_results = {}
for attr, pairs in ATTRIBUTE_AXES.items():
    axis, _ = build_axis_and_mean_dir(pairs)
    if axis is None:
        print(f"  {attr:<14}  FAILED")
        continue
    train_acc, detail = retrieval_acc(pairs, axis)

    # Compute the "mean axis terminal" (centroid of all query terminals)
    p = ok_pairs(pairs)
    terminals = [query_terminal(a, axis) for a,b in p if query_terminal(a,axis) is not None]
    if not terminals:
        continue
    mean_terminal = normed(np.mean(terminals, axis=0))

    degs = {t: degeneracy(mean_terminal, t) for t in THRESHOLDS}
    print(f"  {attr:<14}  {train_acc:>9.3f}  "
          + "  ".join(f"{degs[t]:>8}" for t in THRESHOLDS))
    attr_results[attr] = {"train_acc": train_acc, "degeneracy": degs,
                           "mean_terminal_norms": len(terminals)}

# ── Part B: Per-pair degeneracy ──────────────────────────────────────
print()
print("=" * 70)
print("PART B: Per-pair degeneracy at each source's query terminal")
print("=" * 70)
print()

per_pair_data = []
for attr, pairs in ATTRIBUTE_AXES.items():
    axis, _ = build_axis_and_mean_dir(pairs)
    if axis is None: continue
    p = ok_pairs(pairs)
    print(f"  {attr}:")
    print(f"    {'src':<12}  {'tgt':<12}  {'correct':>7}  "
          + "  ".join(f"deg@{t}" for t in THRESHOLDS))
    for a,b in p:
        qt = query_terminal(a, axis)
        if qt is None: continue
        sims = [(pool_words[i], float(E_normed[i] @ qt.astype(np.float32)))
                for i in range(N) if pool_words[i] != a]
        sims.sort(key=lambda x: x[1], reverse=True)
        pred = sims[0][0] if sims else None
        ok = (pred == b)
        degs = {t: degeneracy(qt, t) for t in THRESHOLDS}
        print(f"    {a:<12}  {b:<12}  {'OK' if ok else '  ':>7}  "
              + "  ".join(f"{degs[t]:>6}" for t in THRESHOLDS))
        per_pair_data.append({
            "attr": attr, "src": a, "tgt": b, "pred": pred,
            "correct": ok,
            "degeneracy": {str(t): degs[t] for t in THRESHOLDS}
        })
    print()

# ── Part C: TYPE_BC degeneracy ───────────────────────────────────────
print()
print("=" * 70)
print("PART C: TYPE_BC degeneracy (expected very low)")
print("=" * 70)
print()

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    return normed(np.mean([normed(get_emb(b)-get_emb(a)) for a,b in p], axis=0))

bc_results = {}
for dname, pairs in BC_DOMAINS.items():
    mdir = mean_dir(pairs)
    if mdir is None: continue
    p = ok_pairs(pairs)
    print(f"  {dname}:")
    print(f"    {'src':<14}  {'tgt':<14}  {'correct':>7}  "
          + "  ".join(f"deg@{t}" for t in THRESHOLDS))
    bc_pair_data = []
    correct = 0
    for a,b in p:
        se = get_emb(a)
        if se is None: continue
        qt = normed(se + mdir)
        sims = [(pool_words[i], float(E_normed[i] @ qt.astype(np.float32)))
                for i in range(N) if pool_words[i] != a]
        sims.sort(key=lambda x: x[1], reverse=True)
        pred = sims[0][0] if sims else None
        ok = (pred == b)
        if ok: correct += 1
        degs = {t: degeneracy(qt, t) for t in THRESHOLDS}
        print(f"    {a:<14}  {b:<14}  {'OK' if ok else '  ':>7}  "
              + "  ".join(f"{degs[t]:>6}" for t in THRESHOLDS))
        bc_pair_data.append({"src":a,"tgt":b,"correct":ok,
                              "degeneracy":{str(t):degs[t] for t in THRESHOLDS}})
    acc = correct/len(p) if p else 0.0
    print(f"    acc={acc:.3f}\n")
    bc_results[dname] = {"acc": acc, "pairs": bc_pair_data}

# ── Part D: Threshold scan + separation analysis ──────────────────────
print()
print("=" * 70)
print("PART D: Optimal threshold — degeneracy predicts correct/wrong")
print("=" * 70)
print()

# Collect all pairs with degeneracy and correctness
all_pairs = per_pair_data.copy()
for dname, bcd in bc_results.items():
    for p in bcd["pairs"]:
        all_pairs.append({"attr": f"bc/{dname}", **p})

print(f"  Total pairs: {len(all_pairs)}")
print(f"  Correct: {sum(1 for p in all_pairs if p['correct'])}")
print(f"  Wrong:   {sum(1 for p in all_pairs if not p['correct'])}")
print()

for t in THRESHOLDS:
    correct_degs = [p["degeneracy"][str(t)] for p in all_pairs if p["correct"]]
    wrong_degs   = [p["degeneracy"][str(t)] for p in all_pairs if not p["correct"]]
    if not correct_degs or not wrong_degs: continue
    mean_c = float(np.mean(correct_degs))
    mean_w = float(np.mean(wrong_degs))
    sep = mean_w - mean_c  # positive = wrong has higher degeneracy

    # Best k_threshold: maximize (correct below k) + (wrong above k)
    all_degs = [(p["degeneracy"][str(t)], p["correct"]) for p in all_pairs]
    ks = sorted(set(d for d,_ in all_degs))
    best_k = None; best_score = -1
    for k in ks:
        tp = sum(1 for d,c in all_degs if d <= k and c)      # correct predicted correct
        tn = sum(1 for d,c in all_degs if d > k and not c)   # wrong predicted wrong
        score = tp + tn
        if score > best_score:
            best_score = score; best_k = k
    total = len(all_pairs)
    print(f"  t={t:.2f}:  mean_correct={mean_c:.1f}  mean_wrong={mean_w:.1f}  "
          f"sep={sep:+.1f}  best_k={best_k}  accuracy@best_k={best_score}/{total}")

# ── Part E: Runtime confidence signal ────────────────────────────────
print()
print("=" * 70)
print("PART E: Runtime confidence signal (1/(1+log(1+deg)))")
print("=" * 70)
print()

CONF_THRESHOLD = 0.85
print(f"  Using degeneracy threshold t={CONF_THRESHOLD}")
print()
print(f"  {'attr':<14}  {'train_acc':>9}  {'mean_conf_correct':>17}  {'mean_conf_wrong':>15}")

for attr, pairs in ATTRIBUTE_AXES.items():
    axis, _ = build_axis_and_mean_dir(pairs)
    if axis is None: continue
    p = ok_pairs(pairs)
    correct_confs = []; wrong_confs = []
    for a,b in p:
        qt = query_terminal(a, axis)
        if qt is None: continue
        deg = degeneracy(qt, CONF_THRESHOLD)
        conf = 1.0 / (1.0 + np.log1p(deg))
        # Check if this pair retrieves correctly
        sims = [(pool_words[i], float(E_normed[i] @ qt.astype(np.float32)))
                for i in range(N) if pool_words[i] != a]
        sims.sort(key=lambda x: x[1], reverse=True)
        pred = sims[0][0] if sims else None
        if pred == b: correct_confs.append(conf)
        else:         wrong_confs.append(conf)

    mc = float(np.mean(correct_confs)) if correct_confs else 0.0
    mw = float(np.mean(wrong_confs))   if wrong_confs   else 0.0
    ta = attr_results.get(attr,{}).get("train_acc",0.0)
    print(f"  {attr:<14}  {ta:>9.3f}  {mc:>17.3f}  {mw:>15.3f}")

print()
print("=" * 70)
print("SUMMARY: Degeneracy as predictive metric")
print("=" * 70)
print()
print("  Key question: does degeneracy(terminal, t) predict retrieval failure?")
print()
print("  Expected: correct pairs have LOW degeneracy")
print("            wrong pairs have HIGH degeneracy")
print()
print("  TYPE_BC degeneracy expected to be lowest of all (unique targets)")
print("  TYPE_ANTONYM(size) expected to be highest (many small-synonyms)")

output = {
    "attr_results": attr_results,
    "per_pair_data": per_pair_data,
    "bc_results": bc_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 230 complete.")
