#!/usr/bin/env python3
"""
Day 246 — Paradigm Direction Orthogonality

Question: are the mean_dir vectors for different morphological paradigms
approximately orthogonal in W_E?

Day 245 found: adj + mean_gender → same adj; noun + mean_degree → same noun.
This cross-paradigm failure could be because:
  (a) The paradigm mean_dir vectors are orthogonal to each other
  (b) The shift lands outside the target word-class manifold

Test A: compute pairwise cos(mean_dir_i, mean_dir_j) for all paradigm pairs.
  If orthogonal: |cos| ≈ 0 → paradigms live in independent subspaces
  If aligned: |cos| > 0.5 → paradigms share a common direction

Test B: do the mean_dir vectors align with the known φ-trie coordinate axes?
  From DC 327: the shared degree plane has axes e1, e2 from the SVD of
  all (comp-pos) vectors. Is mean_dir_pc ≈ e1 (the first shared axis)?

Test C: how much variance does each mean_dir direction explain in the
  vocabulary? Project all W_E rows onto each mean_dir. What's the distribution?
  If the direction is paradigm-specific: the projected distribution should
  show a clear semantic cluster (not spread over all tokens uniformly).

Test D: can we identify adj_degree pairs in W_E purely geometrically?
  For each word w, find all w' with cos(w,w') ≈ cos_target and
  (w' - w) direction ≈ mean_dir_pc direction.
  What fraction of such pairs are actual adj_degree morphological pairs?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "paradigm_ortho.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2

ADJ_TRIPLES = [
    ("big","bigger","biggest"), ("fast","faster","fastest"),
    ("long","longer","longest"), ("small","smaller","smallest"),
    ("hard","harder","hardest"), ("bright","brighter","brightest"),
    ("dark","darker","darkest"), ("rich","richer","richest"),
    ("deep","deeper","deepest"), ("wide","wider","widest"),
    ("high","higher","highest"), ("low","lower","lowest"),
    ("old","older","oldest"), ("young","younger","youngest"),
    ("hot","hotter","hottest"), ("tall","taller","tallest"),
    ("strong","stronger","strongest"), ("weak","weaker","weakest"),
    ("short","shorter","shortest"), ("cool","cooler","coolest"),
    ("great","greater","greatest"), ("safe","safer","safest"),
    ("cheap","cheaper","cheapest"), ("clean","cleaner","cleanest"),
]

ALL_PARADIGMS = {
    "gender":     [("king","queen"),("man","woman"),("boy","girl"),
                   ("prince","princess"),("actor","actress"),("hero","heroine"),
                   ("monk","nun"),("duke","duchess"),("lord","lady"),
                   ("wizard","witch"),("nephew","niece"),("lion","lioness"),
                   ("father","mother"),("son","daughter"),("brother","sister")],
    "plural":     [("cat","cats"),("dog","dogs"),("house","houses"),
                   ("tree","trees"),("book","books"),("car","cars"),
                   ("bird","birds"),("ship","ships"),("hand","hands"),
                   ("door","doors"),("king","kings"),("boy","boys")],
    "past_tense": [("walk","walked"),("talk","talked"),("call","called"),
                   ("pull","pulled"),("look","looked"),("play","played"),
                   ("stay","stayed"),("jump","jumped"),("work","worked"),
                   ("move","moved"),("help","helped"),("turn","turned")],
    "capital":    [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                   ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                   ("India","Delhi"),("Russia","Moscow"),("Greece","Athens"),
                   ("Poland","Warsaw"),("Turkey","Ankara")],
    "antonym_size":[("big","small"),("large","tiny"),("huge","little"),
                    ("tall","short"),("wide","narrow"),("thick","thin"),
                    ("broad","slim"),("heavy","light"),("long","brief")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cos_sim(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def tid1(w):
    for pref in [" ", ""]:
        ids = tok(pref + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: return ids[0]
    return None

def get_emb(w):
    t = tid1(w)
    return W_E[t].copy() if t is not None else None

# Compute mean_dir for each paradigm
paradigm_dirs = {}
paradigm_diffs = {}

# adj_pos2comp
diffs = []
for p, c, s in ADJ_TRIPLES:
    P = get_emb(p); C = get_emb(c)
    if P is not None and C is not None:
        diffs.append(C - P)
if diffs:
    paradigm_diffs["adj_pos2comp"] = np.array(diffs)
    paradigm_dirs["adj_pos2comp"] = np.mean(diffs, axis=0)

# adj_comp2sup
diffs = []
for p, c, s in ADJ_TRIPLES:
    C = get_emb(c); S = get_emb(s)
    if C is not None and S is not None:
        diffs.append(S - C)
if diffs:
    paradigm_diffs["adj_comp2sup"] = np.array(diffs)
    paradigm_dirs["adj_comp2sup"] = np.mean(diffs, axis=0)

for pname, pairs in ALL_PARADIGMS.items():
    diffs = []
    for a_w, b_w in pairs:
        A = get_emb(a_w); B = get_emb(b_w)
        if A is not None and B is not None:
            diffs.append(B - A)
    if diffs:
        paradigm_diffs[pname] = np.array(diffs)
        paradigm_dirs[pname] = np.mean(diffs, axis=0)

print(f"  Loaded {len(paradigm_dirs)} paradigm directions\n")

# ── Part A: Pairwise cosine similarity between mean_dir vectors ───────
print("=" * 70)
print("PART A: PAIRWISE COSINE BETWEEN PARADIGM MEAN_DIR VECTORS")
print("        |cos| ≈ 0 → orthogonal (independent subspaces)")
print("        |cos| > 0.5 → aligned (shared directions)")
print("=" * 70)
print()

paradigm_names = list(paradigm_dirs.keys())
n_p = len(paradigm_names)
cos_matrix = np.zeros((n_p, n_p))
for i, pi in enumerate(paradigm_names):
    for j, pj in enumerate(paradigm_names):
        cos_matrix[i, j] = cos_sim(paradigm_dirs[pi], paradigm_dirs[pj])

# Print matrix
header = f"{'':>16}" + "".join(f"  {p[:10]:>10}" for p in paradigm_names)
print(f"  {header}")
for i, pi in enumerate(paradigm_names):
    row = f"  {pi:>16}" + "".join(f"  {cos_matrix[i,j]:>10.4f}"
                                   for j in range(n_p))
    print(row)
print()

# Off-diagonal analysis
off_diag = [(abs(cos_matrix[i,j]), paradigm_names[i], paradigm_names[j])
            for i in range(n_p) for j in range(i+1, n_p)]
off_diag.sort(reverse=True)
print(f"  Top pairwise |cos| (off-diagonal, sorted):")
for cos_val, pi, pj in off_diag[:8]:
    print(f"    {pi:>16} ↔ {pj:<16}  |cos| = {cos_val:.4f}")
print()
mean_off = np.mean([c for c,_,_ in off_diag])
print(f"  Mean |cos| (all off-diagonal) = {mean_off:.4f}")
print(f"  (Random vectors in R^{H}: expected |cos| ≈ {1/np.sqrt(H):.4f})")

# ── Part B: Mean_dir alignment with vocabulary PCA axes ───────────────
print()
print("=" * 70)
print("PART B: PARADIGM DIRECTIONS vs VOCABULARY PCA AXES")
print("        Do paradigm mean_dirs align with top PCA directions of W_E?")
print("=" * 70)
print()

# Compute top-10 PCA directions of W_E
print("  Computing W_E SVD (top-20 components) ...")
W_c = W_E - np.mean(W_E, axis=0)  # center
_, sv, Vt = np.linalg.svd(W_c[:10000], full_matrices=False)  # sample for speed
print(f"  Top singular values: {sv[:5].round(3)}")
print()

for pname in paradigm_names[:5]:
    d = normed(paradigm_dirs[pname])
    projs = [abs(float(np.dot(d, Vt[k]))) for k in range(min(20, len(sv)))]
    top_k = np.argsort(projs)[-3:][::-1]
    print(f"  {pname:>16}: top PCA alignments: "
          + ", ".join(f"PC{k}={projs[k]:.4f}" for k in top_k))

# ── Part C: Vocabulary projection distribution ────────────────────────
print()
print("=" * 70)
print("PART C: VOCABULARY PROJECTION ONTO EACH MEAN_DIR")
print("        How much does each direction discriminate tokens?")
print("=" * 70)
print()

for pname in paradigm_names:
    d = normed(paradigm_dirs[pname]).astype(np.float32)
    projs = W_E @ d.astype(np.float64)
    # Variance explained
    var_proj = float(np.var(projs))
    var_total = float(np.mean(np.var(W_E, axis=0)))
    # Distribution stats
    p95 = float(np.percentile(projs, 95))
    p5  = float(np.percentile(projs, 5))
    p50 = float(np.median(projs))
    print(f"  {pname:>16}:  var={var_proj:.6f}  "
          f"range=[{p5:.4f}, {p95:.4f}]  median={p50:.4f}")

# ── Part D: Geometric adj_degree pair identification ─────────────────
print()
print("=" * 70)
print("PART D: GEOMETRIC ADJ_DEGREE PAIR IDENTIFICATION")
print("        Find word pairs (w, w') with:")
print(f"          cos(w, w') ≈ 0.567 [±0.05]")
print(f"          cos((w'-w), mean_dir_pc) ≈ high")
print("        What fraction are actual adj_degree pairs?")
print("=" * 70)
print()

mean_dir_pc = paradigm_dirs["adj_pos2comp"]
mean_dir_pc_n = normed(mean_dir_pc).astype(np.float32)
COS_TARGET = 0.567
COS_TOL    = 0.05
DIR_THRESH = 0.60   # minimum cos with mean_dir_pc for the chord direction

# Build pool of single-token English words (lowercase, 3-10 chars)
word_pool = []; pool_tids = []; pool_embs = []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not (w.isalpha() and 3 <= len(w) <= 12 and w.islower()): continue
    word_pool.append(w); pool_tids.append(tid)
    pool_embs.append(W_E[tid])
pool_embs = np.array(pool_embs, dtype=np.float32)
pool_norms = np.array([np.linalg.norm(pool_embs[i]) for i in range(len(pool_embs))],
                      dtype=np.float32)
print(f"  Word pool: {len(word_pool)} single-token lowercase English words")
print()

# For each word in pool, project vocabulary to find candidates
# Sample 500 words from the pool to keep runtime manageable
np.random.seed(42)
sample_idx = np.random.choice(len(word_pool), min(500, len(word_pool)), replace=False)

# Known adj_degree training set for evaluation
known_adj_pairs = set()
for p, c, s in ADJ_TRIPLES:
    known_adj_pairs.add((p, c)); known_adj_pairs.add((c, s))

# Ground truth: known adj_degree comparatives
known_comp_words = set(c for p, c, s in ADJ_TRIPLES)
known_pos_words  = set(p for p, c, s in ADJ_TRIPLES)

candidate_pairs = []; true_pos = 0; false_pos = 0
pool_embs_n = pool_embs / (pool_norms[:, None] + 1e-8)

for si in sample_idx:
    w_emb = pool_embs[si]
    w_n = pool_embs_n[si]
    # Cosine similarities to all pool words
    cos_vals = (pool_embs_n @ w_n.astype(np.float32))
    # Find candidates with cos ≈ COS_TARGET
    mask = np.abs(cos_vals - COS_TARGET) < COS_TOL
    mask[si] = False  # exclude self
    cand_idx = np.where(mask)[0]
    for ci in cand_idx:
        chord = pool_embs[ci].astype(np.float64) - w_emb.astype(np.float64)
        chord_n = normed(chord).astype(np.float32)
        dir_cos = float(np.dot(chord_n, mean_dir_pc_n))
        if dir_cos >= DIR_THRESH:
            src = word_pool[si]; tgt = word_pool[ci]
            is_known = (src, tgt) in known_adj_pairs
            candidate_pairs.append((src, tgt, float(cos_vals[ci]),
                                    dir_cos, is_known))
            if is_known: true_pos += 1
            else: false_pos += 1

print(f"  Geometric search (500 source words, cos∈[{COS_TARGET-COS_TOL:.3f},{COS_TARGET+COS_TOL:.3f}],")
print(f"  chord_cos≥{DIR_THRESH}):")
print(f"    Total candidate pairs: {len(candidate_pairs)}")
print(f"    Known adj_degree pairs (true pos): {true_pos}")
print(f"    Other pairs (false pos): {false_pos}")
if candidate_pairs:
    precision = true_pos / len(candidate_pairs) if candidate_pairs else 0
    print(f"    Precision: {precision:.3f}")
    print()
    print(f"  Sample candidates (true_pos first):")
    known = [(s, t, c, d) for s, t, c, d, k in candidate_pairs if k][:5]
    unknown = [(s, t, c, d) for s, t, c, d, k in candidate_pairs if not k][:10]
    if known:
        print(f"    TRUE  (known adj_degree):")
        for s, t, c, d in known:
            print(f"      {s:>12} → {t:<12}  cos={c:.4f}  dir_cos={d:.4f}")
    if unknown:
        print(f"    FALSE (other word pairs, dir_cos≥{DIR_THRESH}):")
        for s, t, c, d in unknown[:8]:
            print(f"      {s:>12} → {t:<12}  cos={c:.4f}  dir_cos={d:.4f}")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  1. PARADIGM DIRECTION ORTHOGONALITY:")
print(f"     Mean |cos| (off-diagonal) = {mean_off:.4f}")
rand_expected = 1 / np.sqrt(H)
print(f"     Random baseline in R^{H} = {rand_expected:.4f}")
if mean_off < 3 * rand_expected:
    print(f"     NEAR-ORTHOGONAL: paradigm directions are approximately")
    print(f"     independent. Cross-paradigm failure is due to orthogonality.")
else:
    print(f"     NOT ORTHOGONAL: paradigm directions share significant structure.")

output = {
    "paradigm_cos_matrix": {
        paradigm_names[i]: {paradigm_names[j]: float(cos_matrix[i,j])
                             for j in range(n_p)}
        for i in range(n_p)
    },
    "mean_off_diagonal_cos": float(mean_off),
    "random_baseline_cos": float(rand_expected),
    "n_candidate_pairs": len(candidate_pairs),
    "n_true_pos": true_pos,
    "n_false_pos": false_pos,
    "precision": float(true_pos / len(candidate_pairs)) if candidate_pairs else 0,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Paradigm orthogonality analysis complete.")
