#!/usr/bin/env python3
"""
Day 235 — Direction Cosine Matrix + Morphological Line Survey

DC 382 raised two open questions:
  1. Full direction cosine matrix: which pairs of known directions are
     co-linear (composable) vs orthogonal (non-composable)?
  2. Line spacing: is d(A->A_comp) = d(A_comp->A_sup) for each word?
     Are the steps equal? Is the morphological line perfectly uniform?

Experiments:
  A. Build all known direction vectors from training data.
     Compute N×N cosine matrix for all direction pairs.
     Print heatmap-style summary.

  B. Morphological line spacing test:
     For each adjective A (big, fast, long, small, hard, bright, clean):
       d1 = emb(A_comp) - emb(A)      e.g. bigger - big
       d2 = emb(A_sup)  - emb(A_comp) e.g. biggest - bigger
       cos(d1, d2) = step-to-step alignment
       |d1| vs |d2|: are step magnitudes equal?
       (emb(A_sup) - emb(A)) / 2 ?= emb(A_comp) - emb(A)
         i.e., is A_comp the MIDPOINT between A and A_sup?

  C. Composition predictor validation:
     For all pairs (di, dj) with cos(di,dj) measured, predict:
       can_compose = cos(di, dj) > threshold?
     Validate against Day 234 retrieval results.

  D. Three-hop composition test (if time allows):
     d1 + d2 + d3 for collinear triples:
       positive -> comparative -> superlative -> ? (no further step)
     Instead: antonym + comparative + superlative
       big -> small -> smaller -> smallest
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day235_direction_matrix.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── All known direction training data ────────────────────────────────
DIRECTIONS = {
    "gender":         [("king","queen"),("man","woman"),("boy","girl"),
                       ("prince","princess"),("actor","actress"),("hero","heroine")],
    "plural":         [("cat","cats"),("dog","dogs"),("house","houses"),
                       ("tree","trees"),("book","books"),("car","cars")],
    "past_tense":     [("walk","walked"),("talk","talked"),("call","called"),
                       ("pull","pulled"),("fill","filled"),("turn","turned")],
    "superlative":    [("big","biggest"),("fast","fastest"),("long","longest"),
                       ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
    "comparative":    [("big","bigger"),("fast","faster"),("long","longer"),
                       ("smart","smarter"),("small","smaller"),("hard","harder")],
    "comp_to_sup":    [("bigger","biggest"),("faster","fastest"),("longer","longest"),
                       ("smarter","smartest"),("smaller","smallest"),("harder","hardest")],
    "antonym_speed":  [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                       ("rapid","gradual"),("hasty","leisurely")],
    "antonym_size":   [("big","small"),("large","tiny"),("huge","little"),
                       ("tall","short"),("wide","narrow"),("thick","thin")],
    "antonym_weight": [("heavy","light"),("massive","weightless"),("dense","sparse"),
                       ("weighty","featherweight"),("hefty","flimsy")],
    "antonym_rough":  [("rough","smooth"),("coarse","fine"),("jagged","polished"),
                       ("scratchy","silky"),("rugged","sleek")],
    "capital":        [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                       ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
    "numbers":        [("one","1"),("two","2"),("three","3"),
                       ("four","4"),("five","5"),("six","6")],
}

# ── Morphological line words ──────────────────────────────────────────
DEGREE_WORDS = [
    ("big",    "bigger",    "biggest"),
    ("fast",   "faster",    "fastest"),
    ("long",   "longer",    "longest"),
    ("small",  "smaller",   "smallest"),
    ("hard",   "harder",    "hardest"),
    ("bright", "brighter",  "brightest"),
    ("clean",  "cleaner",   "cleanest"),
    ("cold",   "colder",    "coldest"),
    ("warm",   "warmer",    "warmest"),
    ("dark",   "darker",    "darkest"),
    ("soft",   "softer",    "softest"),
    ("rich",   "richer",    "richest"),
    ("deep",   "deeper",    "deepest"),
    ("wide",   "wider",     "widest"),
    ("high",   "higher",    "highest"),
    ("low",    "lower",     "lowest"),
    ("old",    "older",     "oldest"),
    ("young",  "younger",   "youngest"),
]

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
def ok_pairs(pairs):
    return [(a, b) for a, b in pairs if is_single(a) and is_single(b)]

print("Building pool ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w)
        pool_embs.append(W_E[tid].astype(np.float32))

for triplet in DEGREE_WORDS:
    for w in triplet:
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

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None, 0
    dirs = [normed(get_emb(b) - get_emb(a)) for a, b in p]
    return normed(np.mean(dirs, axis=0)), len(p)

# ── Build all direction vectors ───────────────────────────────────────
print("Building direction vectors ...")
dir_vecs = {}
for name, pairs in DIRECTIONS.items():
    d, n = mean_dir(pairs)
    if d is not None:
        dir_vecs[name] = d
        print(f"  {name:<18} n={n}")
print()

# ── Part A: N×N cosine matrix ─────────────────────────────────────────
print("=" * 70)
print("PART A: Direction cosine matrix")
print("=" * 70)
print()

names = sorted(dir_vecs.keys())
M = len(names)
cos_matrix = np.zeros((M, M))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        cos_matrix[i, j] = cosine(dir_vecs[ni], dir_vecs[nj])

# Print header
print("  " + " " * 18 + "  ".join(f"{n[:8]:>8}" for n in names))
for i, ni in enumerate(names):
    row = f"  {ni:<18}"
    for j in range(M):
        v = cos_matrix[i, j]
        row += f"  {v:>+.3f}"
    print(row)

# Highlight strong alignments
print()
print("  Pairs with |cos| > 0.50:")
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        if i >= j: continue
        v = cos_matrix[i, j]
        if abs(v) > 0.50:
            flag = "PARALLEL" if v > 0.70 else ("WEAK_PAR" if v > 0.50 else "ANTI")
            print(f"    {ni:<18} <-> {nj:<18}  cos={v:>+.4f}  {flag}")

# ── Part B: Morphological line spacing ───────────────────────────────
print()
print("=" * 70)
print("PART B: Morphological line spacing (positive → comparative → superlative)")
print("=" * 70)
print()
print(f"  {'word':<8}  {'comp':>10}  {'sup':>10}  "
      f"{'cos(d1,d2)':>10}  {'|d1|':>7}  {'|d2|':>7}  "
      f"{'|d_total|':>9}  {'midpoint_err':>12}")

spacing_results = []
d1_all, d2_all = [], []
for pos, comp, sup in DEGREE_WORDS:
    e_pos  = get_emb(pos)
    e_comp = get_emb(comp)
    e_sup  = get_emb(sup)
    if e_pos is None or e_comp is None or e_sup is None:
        print(f"  {pos:<8}  EC_TOKENIZE  ({comp}, {sup})")
        continue

    d1 = e_comp - e_pos   # raw (un-normed)
    d2 = e_sup  - e_comp
    d_total = e_sup - e_pos

    cos12 = cosine(d1, d2)
    mag1  = float(np.linalg.norm(d1))
    mag2  = float(np.linalg.norm(d2))
    mag_t = float(np.linalg.norm(d_total))

    # Midpoint error: how far is e_comp from (e_pos + e_sup) / 2?
    midpoint = (e_pos + e_sup) / 2
    mid_err  = float(np.linalg.norm(e_comp - midpoint)) / (mag_t + 1e-8)

    print(f"  {pos:<8}  {comp:>10}  {sup:>10}  "
          f"{cos12:>10.4f}  {mag1:>7.2f}  {mag2:>7.2f}  "
          f"{mag_t:>9.2f}  {mid_err:>12.4f}")
    d1_all.append(normed(d1)); d2_all.append(normed(d2))
    spacing_results.append({
        "pos": pos, "comp": comp, "sup": sup,
        "cos_d1_d2": cos12, "mag_d1": mag1, "mag_d2": mag2,
        "mag_total": mag_t, "midpoint_err": mid_err,
    })

if d1_all:
    mean_cos = float(np.mean([cosine(d1_all[i], d2_all[i]) for i in range(len(d1_all))]))
    print(f"\n  Mean cos(d1,d2) across all words: {mean_cos:.4f}")
    mean_mid = float(np.mean([r["midpoint_err"] for r in spacing_results]))
    print(f"  Mean midpoint error:              {mean_mid:.4f}")
    print(f"  (0.0 = perfect midpoint; 1.0 = halfway from midpoint to endpoint)")

# ── Part C: Composition predictor validation ─────────────────────────
print()
print("=" * 70)
print("PART C: Composition predictor — cos(d1,d2) > 0.70 predicts composable?")
print("=" * 70)
print()
print("  (From Day 234 results)")
print()
day234_chains = {
    "comparative->superlative":  ("comp_to_sup", "superlative",          0.9808, True),
    "antonym_size+superlative":  ("antonym_size", "superlative",         0.7379, True),
    "gender+plural":             ("gender",       "plural",               0.6438, False),
    "plural+antonym_speed":      ("plural",       "antonym_speed",        0.5060, False),
    "gender+superlative":        ("gender",       "superlative",          0.4367, False),
    "antonym_speed+superlative": ("antonym_speed","superlative",          0.3609, False),
}
print(f"  {'chain':<35}  {'cos_d1d2':>9}  {'composed?':>9}  {'predicted':>9}")
for chain, (d1n, d2n, known_cos, known_ok) in day234_chains.items():
    if d1n in dir_vecs and d2n in dir_vecs:
        measured_cos = cosine(dir_vecs[d1n], dir_vecs[d2n])
        predicted_ok = measured_cos > 0.70
    else:
        measured_cos = float("nan"); predicted_ok = None
    correct_pred = (predicted_ok == known_ok) if predicted_ok is not None else None
    print(f"  {chain:<35}  {measured_cos:>9.4f}  "
          f"{'YES' if known_ok else 'NO':>9}  "
          f"{'YES' if predicted_ok else 'NO':>9}  "
          f"{'CORRECT' if correct_pred else 'WRONG' if correct_pred is False else '?'}")

# ── Part D: Three-hop composition ────────────────────────────────────
print()
print("=" * 70)
print("PART D: Three-hop antonym+comparative+superlative")
print("=" * 70)
print()
print("  Chain: A -> antonym(A) -> comparative(antonym(A)) -> superlative(antonym(A))")
print("  Example: big -> small -> smaller -> smallest")
print()

d_ant  = dir_vecs.get("antonym_size")
d_comp = dir_vecs.get("comparative")
d_sup2 = dir_vecs.get("comp_to_sup")

three_hop_probes = [
    ("big",  "small",  "smaller",  "smallest"),
    ("fast", "slow",   "slower",   "slowest"),
    ("long", "short",  "shorter",  "shortest"),
    ("hard", "soft",   "softer",   "softest"),
]

if d_ant is not None and d_comp is not None and d_sup2 is not None:
    print(f"  {'A':<8}  {'B':<10}  {'C':<10}  {'D':<12}  "
          f"{'pred_2hop':<14}  {'pred_3hop':<14}  {'rank_D':>6}")
    for a, b, c, d_word in three_hop_probes:
        if not all(is_single(w) for w in [a, b, c, d_word]):
            print(f"  {a:<8}  EC_TOKENIZE")
            continue
        ea = get_emb(a)

        # 2-hop: emb(A) + d_ant + d_comp -> should get C
        qt_2 = normed(ea + d_ant + d_comp)
        pred_2 = top_k(qt_2, k=1, exclude=a)[0][0]

        # 3-hop: emb(A) + d_ant + d_comp + d_sup2 -> should get D
        qt_3 = normed(ea + d_ant + d_comp + d_sup2)
        top3 = top_k(qt_3, k=10, exclude=a)
        pred_3 = top3[0][0]
        rank_d = next((i for i, (w, _) in enumerate(top3) if w == d_word), -1)

        ok2 = "OK" if pred_2 == c else "  "
        ok3 = "OK" if pred_3 == d_word else "  "
        print(f"  {a:<8}  {b:<10}  {c:<10}  {d_word:<12}  "
              f"{pred_2:<12}  {ok2}  {pred_3:<12}  {ok3}  {rank_d:>6}")
else:
    print("  SKIP: missing direction vectors")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  Direction cosine matrix reveals geometric structure of W_E:")
print("  - Same-paradigm directions: nearly parallel (high cos)")
print("  - Cross-paradigm directions: orthogonal (low cos)")
print()
print("  Morphological line test: cos(d1,d2) for individual words")
print("  If all words have high cos: line is universal morphological structure")

# Build output
output = {
    "cos_matrix": {
        "names": names,
        "values": cos_matrix.tolist(),
    },
    "spacing_results": spacing_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 235 complete.")
