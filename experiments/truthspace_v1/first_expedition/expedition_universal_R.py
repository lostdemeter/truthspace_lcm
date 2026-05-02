#!/usr/bin/env python3
"""
Universal R Investigation — Is R≈0.33 Specific to Morphological Pairs?

The geometric audit established: for every morphological pair (A, B),
the circumscribed circle of (O, emb(A), emb(B)) has R ≈ 0.33.

Key question: is this R specific to morphological encoding, or is it
a general property of the embedding space W_E?

Tests:
  A. RANDOM PAIRS: sample 1000 random single-token word pairs.
     Compute R for (O, emb(A), emb(B)). What is the distribution?

  B. SEMANTIC NEIGHBORS: for each word, sample pairs from its
     k-nearest semantic neighbors. Morphological pairs might be a
     special subset of semantic neighbors.

  C. RANDOM PAIRS STRATIFIED BY COSINE SIMILARITY:
     Group pairs by cos(A, B): {near=0.7-0.9, mid=0.4-0.6, far=0.1-0.3}
     Is R consistent within a similarity band?

  D. R vs EMBEDDING NORMS: does R depend on ||A||, ||B||, or ||A||+||B||?
     If R ≈ (||A||+||B||)/k for some k, then R is just an artifact of scale.

  E. THEORETICAL R FOR RANDOM UNIT VECTORS: if A, B are unit vectors
     in R^H with cos(A,B) = c, what is E[R]?
     Compute analytically and compare to empirical.

  F. MORPHOLOGICAL vs RANDOM COMPARISON:
     Show distribution of R for each group side by side.
     Is morphological R a special value, or just typical for word pairs?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "universal_R.json")
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
                   ("Egypt","Cairo"),("Poland","Warsaw"),("Turkey","Ankara")],
    "antonym_size":[("big","small"),("large","tiny"),("huge","little"),
                    ("tall","short"),("wide","narrow"),("thick","thin"),
                    ("broad","slim"),("heavy","light"),("long","brief")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
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
    return W_E[t].copy() if t is not None else None

def circumscribed_R_OAB(A, B):
    """Radius of circumscribed circle of (O, A, B) in the plane spanned by A, B."""
    O = np.zeros(len(A))
    v1 = A - O; v2 = B - O
    # 2D coordinates (O as origin)
    # Use SVD for a numerically stable 2D basis
    D = np.stack([v1, v2], axis=1)
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    if sv[1] < 1e-10: return None, None
    e1, e2 = U[:,0], U[:,1]
    p2 = np.array([0., 0.])  # O
    a2 = np.array([float(np.dot(v1, e1)), float(np.dot(v1, e2))])  # A
    b2 = np.array([float(np.dot(v2, e1)), float(np.dot(v2, e2))])  # B
    ax, ay = p2; bx, by = a2; cx, cy = b2
    Dv = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    if abs(Dv) < 1e-12: return None, None
    ux = ((ax**2+ay**2)*(by-cy)+(bx**2+by**2)*(cy-ay)+(cx**2+cy**2)*(ay-by))/Dv
    uy = ((ax**2+ay**2)*(cx-bx)+(bx**2+by**2)*(ax-cx)+(cx**2+cy**2)*(bx-ax))/Dv
    R = float(np.sqrt((ax-ux)**2+(ay-uy)**2))
    cos_AB = float(np.dot(normed(A), normed(B)))
    return R, cos_AB

# ── Build a pool of clean English single-token words ─────────────────
print("  Building word pool ...")
word_pool = []
word_embs = []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not (w.isalpha() and 3 <= len(w) <= 10 and w.islower()): continue
    word_pool.append(w)
    word_embs.append(W_E[tid])
word_embs = np.array(word_embs)
print(f"  Pool size: {len(word_pool)}\n")

np.random.seed(42)

# ── Part A: Random pairs ──────────────────────────────────────────────
print("=" * 70)
print("PART A: RANDOM PAIRS (1000 random pairs from word pool)")
print("=" * 70)
print()

N_random = 1000
idx1 = np.random.choice(len(word_pool), N_random, replace=False)
idx2 = np.random.choice(len(word_pool), N_random, replace=False)
# Ensure no duplicate pairs
mask = idx1 != idx2
idx1, idx2 = idx1[mask], idx2[mask]

random_Rs = []; random_cos = []
for i, j in zip(idx1, idx2):
    R, cos_ab = circumscribed_R_OAB(word_embs[i], word_embs[j])
    if R is not None:
        random_Rs.append(R)
        random_cos.append(cos_ab)

random_Rs = np.array(random_Rs)
random_cos = np.array(random_cos)
print(f"  Random pairs: n={len(random_Rs)}")
print(f"  R: mean={np.mean(random_Rs):.4f}  std={np.std(random_Rs):.4f}  "
      f"median={np.median(random_Rs):.4f}")
print(f"  cos(A,B): mean={np.mean(random_cos):.4f}  std={np.std(random_cos):.4f}")
print(f"  R percentiles: p5={np.percentile(random_Rs,5):.4f}  "
      f"p25={np.percentile(random_Rs,25):.4f}  p75={np.percentile(random_Rs,75):.4f}  "
      f"p95={np.percentile(random_Rs,95):.4f}")

# ── Part B: Morphological pair R values ──────────────────────────────
print()
print("=" * 70)
print("PART B: MORPHOLOGICAL PAIRS — R per paradigm")
print("=" * 70)
print()

morph_Rs = {}
morph_cos = {}
# adj_pos2comp
adj_pairs = [(p, c) for p, c, s in ADJ_TRIPLES]
for pname, pairs in [("adj_pos2comp", adj_pairs)] + list(ALL_PARADIGMS.items()):
    Rs = []; cos_vals = []
    for a_w, b_w in pairs:
        A = get_emb(a_w); B = get_emb(b_w)
        if A is None or B is None: continue
        R, cos_ab = circumscribed_R_OAB(A, B)
        if R is not None:
            Rs.append(R); cos_vals.append(cos_ab)
    if Rs:
        morph_Rs[pname] = Rs; morph_cos[pname] = cos_vals
        print(f"  {pname:<16}  n={len(Rs):>2}  R={np.mean(Rs):.4f}±{np.std(Rs):.4f}  "
              f"cos={np.mean(cos_vals):.4f}±{np.std(cos_vals):.4f}")

# ── Part C: Stratified by cosine similarity ───────────────────────────
print()
print("=" * 70)
print("PART C: RANDOM PAIRS STRATIFIED BY COSINE SIMILARITY")
print("        Is R consistent within a similarity band?")
print("=" * 70)
print()

bands = [
    ("far   cos∈[-0.1, 0.1]", -0.1, 0.1),
    ("low   cos∈[ 0.1, 0.3]",  0.1, 0.3),
    ("mid   cos∈[ 0.3, 0.5]",  0.3, 0.5),
    ("high  cos∈[ 0.5, 0.7]",  0.5, 0.7),
    ("near  cos∈[ 0.7, 0.9]",  0.7, 0.9),
]
for band_name, lo, hi in bands:
    mask = (random_cos >= lo) & (random_cos < hi)
    band_Rs = random_Rs[mask]
    if len(band_Rs) < 5: continue
    print(f"  {band_name}:  n={len(band_Rs):>3}  R={np.mean(band_Rs):.4f}±{np.std(band_Rs):.4f}")

# Also show where morphological paradigms fall
print()
print("  Morphological paradigm cosine ranges:")
for pname, cos_vals in morph_cos.items():
    print(f"    {pname:<16}  cos mean={np.mean(cos_vals):.4f}  "
          f"range=[{min(cos_vals):.3f}, {max(cos_vals):.3f}]")

# ── Part D: R vs embedding norms ──────────────────────────────────────
print()
print("=" * 70)
print("PART D: R vs EMBEDDING NORMS")
print("        If R = f(||A||, ||B||), it's just a scale artifact.")
print("=" * 70)
print()

rand_nA = np.array([np.linalg.norm(word_embs[i]) for i in idx1[:len(random_Rs)]])
rand_nB = np.array([np.linalg.norm(word_embs[j]) for j in idx2[:len(random_Rs)]])
norm_sum = rand_nA + rand_nB
norm_prod = rand_nA * rand_nB
norm_max  = np.maximum(rand_nA, rand_nB)

for name, vals in [("||A||", rand_nA), ("||B||", rand_nB),
                    ("||A||+||B||", norm_sum), ("||A||·||B||", norm_prod),
                    ("max(||A||,||B||)", norm_max)]:
    # Trim to same length as random_Rs
    v = vals[:len(random_Rs)]
    corr = float(np.corrcoef(random_Rs, v)[0, 1])
    print(f"  corr(R, {name:<18}) = {corr:>+.4f}")

# Is R ≈ (||A||+||B||)/k?
ratio = random_Rs / (norm_sum[:len(random_Rs)] / 2)
print(f"\n  R / mean_norm = {np.mean(ratio):.4f} ± {np.std(ratio):.4f}")
print(f"  (If R = mean_norm × k, then ratio = k)")

# ── Part E: Theoretical R for random vectors ──────────────────────────
print()
print("=" * 70)
print("PART E: THEORETICAL R vs EMPIRICAL")
print("        For (O, A, B) with ||A||=||B||=r and cos(A,B)=c:")
print("        R_theory = r / (2·sin(acos(c)/2))")
print("=" * 70)
print()

# The circumscribed circle of (O, A, B) where O is origin:
# By the inscribed angle theorem, R = |OA| / (2*sin(angle_OAB))
# But easier: half the longest side divided by the sine of opposite angle.
# For the circumscribed circle of 3 points with sides a, b, c and area K:
# R = abc / (4K)
# For (O, A, B): sides = |A| = nA, |B| = nB, |A-B| = d_AB
# K = 0.5 * |A × B| = 0.5 * nA * nB * sin(theta) where theta = angle(A,B)
# R = nA * nB * d_AB / (2 * nA * nB * sin(theta)) = d_AB / (2*sin(theta))
# d_AB = sqrt(nA^2 + nB^2 - 2*nA*nB*cos(theta))

# For typical nA ≈ nB ≈ 0.6 and cos = 0.4:
nA_typ = 0.60; nB_typ = 0.60
cos_vals_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print(f"  {'cos(A,B)':>10}  {'theta':>10}  {'d_AB(nA=nB=0.6)':>18}  "
      f"{'R_theory':>10}  {'R_morph_range':>14}")
morph_R_by_cos = {k: (np.mean(morph_Rs[k]), np.mean(morph_cos[k]))
                  for k in morph_Rs}
for c in cos_vals_test:
    theta = np.arccos(c)
    d_AB = np.sqrt(nA_typ**2 + nB_typ**2 - 2*nA_typ*nB_typ*c)
    sin_theta = np.sin(theta)
    R_theory = d_AB / (2 * sin_theta) if sin_theta > 1e-8 else float("nan")
    print(f"  {c:>10.2f}  {np.degrees(theta):>9.1f}°  {d_AB:>18.4f}  "
          f"{R_theory:>10.4f}")

print()
print("  Morphological cos(A,B) and R values:")
for pname, Rs in morph_Rs.items():
    R_m = np.mean(Rs)
    c_m = np.mean(morph_cos[pname])
    nA_mean = np.mean([np.linalg.norm(get_emb(a))
                       for a, b in (ALL_PARADIGMS.get(pname) or adj_pairs)
                       if get_emb(a) is not None])
    nB_mean = np.mean([np.linalg.norm(get_emb(b))
                       for a, b in (ALL_PARADIGMS.get(pname) or adj_pairs)
                       if get_emb(b) is not None])
    theta = np.arccos(np.clip(c_m, -1, 1))
    d_AB = np.sqrt(nA_mean**2 + nB_mean**2 - 2*nA_mean*nB_mean*c_m)
    sin_t = np.sin(theta)
    R_pred = d_AB / (2 * sin_t) if sin_t > 1e-8 else float("nan")
    print(f"  {pname:<16}  cos={c_m:.4f}  ||A||={nA_mean:.4f}  ||B||={nB_mean:.4f}  "
          f"R_meas={R_m:.4f}  R_theory={R_pred:.4f}  "
          f"{'MATCH' if abs(R_m - R_pred) < 0.01 else 'DIFF':>6}")

# ── Part F: Is morphological R special? ──────────────────────────────
print()
print("=" * 70)
print("PART F: MORPHOLOGICAL R vs RANDOM R AT SAME COSINE SIMILARITY")
print("        Compare morphological R to random R at matched cosine bands.")
print("=" * 70)
print()

for pname, Rs in morph_Rs.items():
    c_m = np.mean(morph_cos[pname])
    R_m = np.mean(Rs)
    # Find random pairs with similar cosine
    tol = 0.05
    mask = np.abs(random_cos - c_m) < tol
    matched_Rs = random_Rs[mask]
    if len(matched_Rs) < 5:
        tol = 0.10
        mask = np.abs(random_cos - c_m) < tol
        matched_Rs = random_Rs[mask]
    if len(matched_Rs) == 0:
        print(f"  {pname:<16}  (no matched random pairs)")
        continue
    R_rand = np.mean(matched_Rs)
    diff = R_m - R_rand
    print(f"  {pname:<16}  R_morph={R_m:.4f}  R_random={R_rand:.4f}  "
          f"diff={diff:>+.4f}  (n_rand={len(matched_Rs)})")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  Random pairs:     R = {np.mean(random_Rs):.4f} ± {np.std(random_Rs):.4f}")
print(f"  Morphological:    R ≈ 0.32-0.38 (paradigm-specific)")
print()
print(f"  The theoretical formula R = d_AB / (2*sin(theta)) where:")
print(f"    d_AB = Euclidean distance between A and B")
print(f"    theta = angle between A and B")
print(f"  If this formula matches measured R for all paradigms,")
print(f"  then R is NOT a special property of morphological pairs —")
print(f"  it's DETERMINED by (||A||, ||B||, cos(A,B)) for ANY pair.")

output = {
    "random_R_mean": float(np.mean(random_Rs)),
    "random_R_std":  float(np.std(random_Rs)),
    "random_cos_mean": float(np.mean(random_cos)),
    "morph_R": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in morph_Rs.items()},
    "morph_cos": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                  for k, v in morph_cos.items()},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Universal R analysis complete.")
