#!/usr/bin/env python3
"""
Day 240 — Universal Transformation Direction Deep-Dive

Day 239 found: the mean of 10 paradigm mean-directions gives a
"universal" vector on which ALL morphological targets score HIGHER
than their sources (delta > 0 for all 5 paradigms tested).

Questions:
  A. What IS this universal direction?
     - Is it the same as the adj_pos2sup direction (cos=0.69 in Day 239)?
     - Or is it truly distinct from all individual paradigm directions?
     - What tokens have extreme projections? Is it a "derived-form" axis?

  B. Can we improve it?
     - Compute it as the mean of MORE paradigm directions (not just 10)
     - Or use SVD on the full stack of ALL paradigm difference vectors
       (the "global morphological transformation" subspace)
     - Does the improved version give larger deltas?

  C. Is the universal direction useful for retrieval CONFIDENCE?
     - Day 231 failed to find a runtime confidence signal.
     - Hypothesis: projection of (query_terminal - source) onto the
       universal direction gives a confidence score:
         high projection -> correct retrieval (moved in universal direction)
         low projection  -> wrong retrieval (did not move in right direction)
     - Test on adj_pos2sup, gender, plural, past_tense, capital.

  D. Relationship to adj_pos2sup direction:
     - cos(d_universal, d_adj_pos2sup) was 0.69 in Day 239
     - But we want to know: is this just because adj_pos2sup DOMINATES
       the universal direction (large step magnitude 0.555)?
     - Recompute without adj_pos2sup. Does the direction change?
     - Recompute with step-magnitude normalization.

  E. Why is adj_pos2sup delta (0.457) 3x larger than other paradigms (~0.16)?
     - Superlatives are specific tokens (-est forms)
     - They cluster in a distinct region of W_E
     - Other paradigms have more overlap between source and target distributions
     - Measure distribution overlap: histogram of source/target projections
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day240_universal_dir.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ALL_PAIRS = {
    "adj_pos2sup":  [("big","biggest"),("fast","fastest"),("long","longest"),
                     ("small","smallest"),("hard","hardest"),("bright","brightest"),
                     ("dark","darkest"),("rich","richest"),("deep","deepest"),
                     ("wide","widest"),("high","highest"),("low","lowest"),
                     ("old","oldest"),("young","youngest"),("hot","hottest"),
                     ("tall","tallest"),("strong","strongest"),("weak","weakest"),
                     ("short","shortest"),("cold","coldest"),("warm","warmest"),
                     ("soft","softest"),("clean","cleanest"),("cheap","cheapest"),
                     ("cool","coolest"),("great","greatest"),("safe","safest")],
    "adj_pos2comp": [("big","bigger"),("fast","faster"),("long","longer"),
                     ("small","smaller"),("hard","harder"),("bright","brighter"),
                     ("dark","darker"),("rich","richer"),("deep","deeper"),
                     ("wide","wider"),("high","higher"),("low","lower"),
                     ("old","older"),("young","younger"),("hot","hotter"),
                     ("tall","taller"),("strong","stronger"),("weak","weaker"),
                     ("short","shorter"),("cool","cooler"),("great","greater"),
                     ("safe","safer"),("cheap","cheaper"),("clean","cleaner")],
    "gender":       [("king","queen"),("man","woman"),("boy","girl"),
                     ("prince","princess"),("actor","actress"),("hero","heroine"),
                     ("monk","nun"),("duke","duchess"),("lord","lady"),
                     ("wizard","witch"),("nephew","niece"),("lion","lioness")],
    "plural":       [("cat","cats"),("dog","dogs"),("house","houses"),
                     ("tree","trees"),("book","books"),("car","cars"),
                     ("bird","birds"),("ship","ships"),("hand","hands"),
                     ("door","doors"),("lamp","lamps"),("wall","walls"),
                     ("king","kings"),("boy","boys"),("word","words"),
                     ("stone","stones"),("cloud","clouds"),("leaf","leaves")],
    "past_tense":   [("walk","walked"),("talk","talked"),("call","called"),
                     ("pull","pulled"),("fill","filled"),("turn","turned"),
                     ("look","looked"),("move","moved"),("push","pushed"),
                     ("help","helped"),("play","played"),("stay","stayed"),
                     ("lock","locked"),("jump","jumped"),("land","landed"),
                     ("ask","asked"),("work","worked"),("open","opened")],
    "capital":      [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                     ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                     ("India","Delhi"),("Russia","Moscow"),("Greece","Athens"),
                     ("Egypt","Cairo"),("Mexico","Mexico"),("Poland","Warsaw")],
    "antonym_size": [("big","small"),("large","tiny"),("huge","little"),
                     ("tall","short"),("wide","narrow"),("thick","thin"),
                     ("broad","slim"),("vast","minute"),("giant","miniature")],
    "antonym_speed":[("fast","slow"),("quick","sluggish"),("swift","plodding"),
                     ("rapid","gradual"),("hasty","leisurely"),("brisk","languid")],
    "antonym_temp": [("hot","cold"),("warm","cool"),("burning","freezing"),
                     ("scorching","icy"),("boiling","chilly")],
    "antonym_light":[("bright","dark"),("light","gloomy"),("shiny","dull"),
                     ("vivid","faded"),("radiant","dim"),("gleaming","murky")],
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

def mean_dir_from_pairs(pairs):
    p = ok_pairs(pairs)
    if not p: return None, 0
    ds = [normed(get_emb(b) - get_emb(a)) for a,b in p]
    return normed(np.mean(ds, axis=0)), len(p)

print("Building pool ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))

for pairs in ALL_PAIRS.values():
    for a, b in pairs:
        for w in [a, b]:
            if w not in pool_words:
                e = get_emb(w)
                if e is not None:
                    pool_words.append(w); pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E = np.array(pool_embs, dtype=np.float32)
norms_v = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
E_normed = (E / norms_v).astype(np.float32)
print(f"  Pool: {N} tokens\n")

def top_k(qt, k=5, exclude=None):
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

# ── Build all mean directions ─────────────────────────────────────────
print("Building mean directions ...")
mean_dirs = {}
pair_counts = {}
for pname, pairs in ALL_PAIRS.items():
    d, n = mean_dir_from_pairs(pairs)
    if d is not None:
        mean_dirs[pname] = d; pair_counts[pname] = n
        print(f"  {pname:<18}  n={n}")
print()

# ── Part A: What IS the universal direction? ─────────────────────────
print("=" * 70)
print("PART A: Universal transformation direction — characterization")
print("=" * 70)
print()

d_univ = normed(np.mean(list(mean_dirs.values()), axis=0))

print(f"  cos(d_univ, each paradigm mean_dir):")
for pname in sorted(mean_dirs.keys()):
    c = cosine(d_univ, mean_dirs[pname])
    print(f"    {pname:<18}  cos={c:>+.4f}")

# Project all pool tokens onto universal direction
projs = E_normed.astype(np.float64) @ normed(d_univ)
order_pos = np.argsort(-projs)
order_neg = np.argsort(projs)

print()
print(f"  Top 25 POSITIVE (target-form-like):")
print("  " + ", ".join(f"{pool_words[i]}({projs[i]:+.3f})" for i in order_pos[:25]))
print()
print(f"  Top 25 NEGATIVE (source-form-like):")
print("  " + ", ".join(f"{pool_words[i]}({projs[i]:+.3f})" for i in order_neg[:25]))

# ── Part B: SVD-based improved universal direction ────────────────────
print()
print("=" * 70)
print("PART B: SVD of all paradigm difference vectors (global morph subspace)")
print("=" * 70)
print()

all_diffs = []
diff_labels = []
for pname, pairs in ALL_PAIRS.items():
    for a, b in ok_pairs(pairs):
        ea = get_emb(a); eb = get_emb(b)
        if ea is None or eb is None: continue
        all_diffs.append(eb - ea)
        diff_labels.append(pname)

D = np.array(all_diffs, dtype=np.float64)
print(f"  Total difference vectors: {len(D)}")
print(f"  Computing SVD ...")
U, S, Vt = np.linalg.svd(D, full_matrices=False)
total_var = float((S**2).sum())

print(f"\n  Singular value spectrum (top 10):")
print(f"  {'k':>3}  {'S[k]':>10}  {'var_cumul':>10}")
cumvar = 0.0
for i in range(min(10, len(S))):
    cumvar += S[i]**2 / total_var
    print(f"  {i:>3}  {S[i]:>10.1f}  {cumvar:>10.3f}")

d_svd1 = Vt[0].astype(np.float64)
# Align with positive delta direction
sample_delta = float(normed(get_emb("fastest") - get_emb("fast")) @ d_svd1)
if sample_delta < 0:
    d_svd1 = -d_svd1
    print("\n  (Flipped SVD1 to align with target direction)")

print(f"\n  cos(d_svd1, d_univ): {cosine(d_svd1, d_univ):>+.4f}")
print(f"\n  cos(d_svd1, each paradigm mean_dir):")
for pname in sorted(mean_dirs.keys()):
    c = cosine(d_svd1, mean_dirs[pname])
    print(f"    {pname:<18}  cos={c:>+.4f}")

# Source-target deltas for SVD direction
print()
print(f"  Source->target deltas on SVD1 direction:")
for pname in ["adj_pos2sup","gender","plural","past_tense","capital","antonym_size"]:
    if pname not in ALL_PAIRS: continue
    src_p, tgt_p = [], []
    for a, b in ok_pairs(ALL_PAIRS[pname]):
        ea = get_emb(a); eb = get_emb(b)
        if ea is None or eb is None: continue
        src_p.append(float(normed(ea) @ d_svd1))
        tgt_p.append(float(normed(eb) @ d_svd1))
    if not src_p: continue
    delta = np.mean(tgt_p) - np.mean(src_p)
    sep = (np.mean(tgt_p) - np.mean(src_p)) / (np.std(src_p + tgt_p) + 1e-8)
    print(f"  {pname:<18}  delta={delta:>+.3f}  sep={sep:>+.3f}  "
          f"src={np.mean(src_p):>+.3f}  tgt={np.mean(tgt_p):>+.3f}")

# ── Part C: Universal direction as retrieval confidence signal ─────────
print()
print("=" * 70)
print("PART C: Universal direction as confidence signal for retrieval")
print("=" * 70)
print()
print("  For each retrieval attempt: measure delta = (emb(pred) - emb(src)) · d_svd1")
print("  H: high delta -> correct retrieval, low delta -> wrong retrieval")
print()
print(f"  {'paradigm':<18}  {'acc':>6}  {'delta_correct':>13}  {'delta_wrong':>11}  {'sep':>6}")

conf_results = {}
for pname in ["adj_pos2sup","adj_pos2comp","gender","plural","past_tense","capital"]:
    if pname not in ALL_PAIRS or pname not in mean_dirs: continue
    d_task = mean_dirs[pname]
    correct_deltas, wrong_deltas = [], []
    ok_count = 0; total = 0
    for a, b in ok_pairs(ALL_PAIRS[pname]):
        ea = get_emb(a); eb = get_emb(b)
        if ea is None or eb is None: continue
        total += 1
        qt = normed(ea + d_task)
        pred = top_k(qt, k=1, exclude=a)[0][0]
        ep = get_emb(pred)
        if ep is None: continue
        delta = float(normed(ep) @ d_svd1) - float(normed(ea) @ d_svd1)
        if pred == b:
            ok_count += 1; correct_deltas.append(delta)
        else:
            wrong_deltas.append(delta)
    acc = ok_count / total if total else 0
    m_c = float(np.mean(correct_deltas)) if correct_deltas else float("nan")
    m_w = float(np.mean(wrong_deltas)) if wrong_deltas else float("nan")
    if correct_deltas and wrong_deltas:
        all_d = correct_deltas + wrong_deltas
        sep = (m_c - m_w) / (np.std(all_d) + 1e-8)
    else:
        sep = float("nan")
    n_c = len(correct_deltas); n_w = len(wrong_deltas)
    print(f"  {pname:<18}  {acc:>6.3f}  "
          f"{m_c:>+6.3f}(n={n_c:>2})  "
          f"{m_w:>+6.3f}(n={n_w:>2})  {sep:>+6.3f}")
    conf_results[pname] = {"acc": acc, "delta_correct": m_c,
                           "delta_wrong": m_w, "sep": sep}

# ── Part D: Without adj_pos2sup, does direction change? ──────────────
print()
print("=" * 70)
print("PART D: Universal direction with and without adj_pos2sup")
print("=" * 70)
print()

excl_dirs = {k: v for k, v in mean_dirs.items() if k not in
             ["adj_pos2sup","adj_pos2comp","adj_comp2sup"]}
d_univ_noAdj = normed(np.mean(list(excl_dirs.values()), axis=0))

print(f"  Excluding adj_degree paradigms ({len(excl_dirs)} remaining):")
print(f"  cos(d_univ_noAdj, d_univ):     {cosine(d_univ_noAdj, d_univ):>+.4f}")
print(f"  cos(d_univ_noAdj, d_svd1):     {cosine(d_univ_noAdj, d_svd1):>+.4f}")
print(f"  cos(d_univ_noAdj, d_adj_pos2sup): "
      f"{cosine(d_univ_noAdj, mean_dirs.get('adj_pos2sup', d_univ)):>+.4f}")
print()
print(f"  Source->target deltas WITHOUT adj_degree:")
for pname in ["gender","plural","past_tense","capital","antonym_size"]:
    if pname not in ALL_PAIRS: continue
    src_p, tgt_p = [], []
    for a, b in ok_pairs(ALL_PAIRS[pname]):
        ea = get_emb(a); eb = get_emb(b)
        if ea is None or eb is None: continue
        src_p.append(float(normed(ea) @ d_univ_noAdj))
        tgt_p.append(float(normed(eb) @ d_univ_noAdj))
    if not src_p: continue
    delta = np.mean(tgt_p) - np.mean(src_p)
    print(f"  {pname:<18}  delta={delta:>+.3f}")

# ── Part E: Why is adj_pos2sup delta 3x larger? ───────────────────────
print()
print("=" * 70)
print("PART E: Why is adj_pos2sup delta (0.46) ~3x larger than other paradigms?")
print("=" * 70)
print()
print("  Distribution overlap analysis:")
print()
print(f"  {'paradigm':<18}  {'src_min':>8}  {'src_max':>8}  {'tgt_min':>8}  "
      f"{'tgt_max':>8}  {'overlap':>8}  {'effect_size':>11}")
for pname in ["adj_pos2sup","adj_pos2comp","gender","plural","past_tense","capital"]:
    if pname not in ALL_PAIRS or pname not in mean_dirs: continue
    src_p, tgt_p = [], []
    for a, b in ok_pairs(ALL_PAIRS[pname]):
        ea = get_emb(a); eb = get_emb(b)
        if ea is None or eb is None: continue
        src_p.append(float(normed(ea) @ d_svd1))
        tgt_p.append(float(normed(eb) @ d_svd1))
    if len(src_p) < 2: continue
    src_arr = np.array(src_p); tgt_arr = np.array(tgt_p)
    # Overlap: fraction of source range that overlaps with target range
    s_lo, s_hi = src_arr.min(), src_arr.max()
    t_lo, t_hi = tgt_arr.min(), tgt_arr.max()
    overlap_lo = max(s_lo, t_lo); overlap_hi = min(s_hi, t_hi)
    src_range = s_hi - s_lo + 1e-8
    overlap_frac = max(0, overlap_hi - overlap_lo) / src_range
    # Cohen's d
    pooled_std = np.sqrt((src_arr.std()**2 + tgt_arr.std()**2) / 2) + 1e-8
    effect_d = (tgt_arr.mean() - src_arr.mean()) / pooled_std
    print(f"  {pname:<18}  {s_lo:>8.3f}  {s_hi:>8.3f}  {t_lo:>8.3f}  "
          f"{t_hi:>8.3f}  {overlap_frac:>8.3f}  {effect_d:>11.3f}")

output = {
    "univ_dir_cos": {pn: cosine(d_univ, mean_dirs[pn]) for pn in mean_dirs},
    "svd1_cos": {pn: cosine(d_svd1, mean_dirs[pn]) for pn in mean_dirs},
    "conf_results": conf_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 240 complete.")
