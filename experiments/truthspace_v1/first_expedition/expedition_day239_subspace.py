#!/usr/bin/env python3
"""
Day 239 — W_E Paradigm Subspace Structure

DC 383 open question: How many independent paradigm subspaces exist?
Are they truly orthogonal, or only approximately?

Approach: For each paradigm, build the DIFFERENCE MATRIX from all
training pairs, compute its SVD to extract the subspace basis, then
measure PRINCIPAL ANGLES between pairs of paradigm subspaces.

Principal angles (canonical angles) give the formal measure of
subspace similarity:
  - θ_min = 0°: subspaces share at least one direction
  - θ_min = 90°: subspaces are completely orthogonal

Experiments:
  A. Paradigm subspace dimensionality:
     For each paradigm, stack all un-normed difference vectors.
     Compute SVD. How fast do singular values decay?
     Is each paradigm 1-dimensional (rank-1 subspace) or multi-dimensional?

  B. Principal angles between paradigm subspaces:
     For each pair of paradigms, compute the canonical angles between
     their leading k-dimensional subspaces (k=1,2,3).
     cos(θ_i) = i-th singular value of B1.T @ B2 where B1, B2 are
     orthonormal bases for the two subspaces.

  C. Does the antonym direction lie in the degree subspace?
     Project antonym_size direction onto the adj_degree subspace.
     Project antonym_size onto the full degree basis.
     Is the antonym-size direction decomposable as:
       d_antonym = α * d_degree + noise?

  D. Intra-paradigm variance structure:
     For each paradigm, how much variance is captured by rank-1 subspace?
     Is there a second dimension (rank-2 structure)?
     Compare: adj_degree, gender, plural, antonym_size, antonym_speed.

  E. The "paradigm coordinate" hypothesis:
     Can we encode any paradigm membership as a coordinate on a universal
     paradigm axis? i.e., is there a single direction in W_E such that
     projecting any word onto it reveals which paradigm it belongs to?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day239_subspace.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PARADIGMS = {
    "adj_pos2sup":   [("big","biggest"),("fast","fastest"),("long","longest"),
                      ("small","smallest"),("hard","hardest"),("bright","brightest"),
                      ("dark","darkest"),("rich","richest"),("deep","deepest"),
                      ("wide","widest"),("high","highest"),("low","lowest"),
                      ("old","oldest"),("young","youngest"),("hot","hottest"),
                      ("tall","tallest"),("strong","strongest"),("weak","weakest"),
                      ("short","shortest")],
    "adj_pos2comp":  [("big","bigger"),("fast","faster"),("long","longer"),
                      ("small","smaller"),("hard","harder"),("bright","brighter"),
                      ("dark","darker"),("rich","richer"),("deep","deeper"),
                      ("wide","wider"),("high","higher"),("low","lower"),
                      ("old","older"),("young","younger"),("hot","hotter"),
                      ("tall","taller"),("strong","stronger"),("weak","weaker"),
                      ("short","shorter")],
    "adj_comp2sup":  [("bigger","biggest"),("faster","fastest"),("longer","longest"),
                      ("smaller","smallest"),("harder","hardest"),("brighter","brightest"),
                      ("darker","darkest"),("richer","richest"),("deeper","deepest"),
                      ("wider","widest"),("higher","highest"),("lower","lowest"),
                      ("older","oldest"),("younger","youngest"),("hotter","hottest"),
                      ("taller","tallest"),("stronger","strongest"),("weaker","weakest"),
                      ("shorter","shortest")],
    "gender":        [("king","queen"),("man","woman"),("boy","girl"),
                      ("prince","princess"),("actor","actress"),("hero","heroine"),
                      ("monk","nun"),("duke","duchess"),("lord","lady"),
                      ("wizard","witch"),("sir","madam"),("nephew","niece")],
    "plural":        [("cat","cats"),("dog","dogs"),("house","houses"),
                      ("tree","trees"),("book","books"),("car","cars"),
                      ("bird","birds"),("ship","ships"),("hand","hands"),
                      ("door","doors"),("lamp","lamps"),("wall","walls"),
                      ("king","kings"),("boy","boys"),("word","words")],
    "past_tense":    [("walk","walked"),("talk","talked"),("call","called"),
                      ("pull","pulled"),("fill","filled"),("turn","turned"),
                      ("look","looked"),("move","moved"),("push","pushed"),
                      ("help","helped"),("play","played"),("stay","stayed"),
                      ("lock","locked"),("jump","jumped"),("land","landed")],
    "antonym_size":  [("big","small"),("large","tiny"),("huge","little"),
                      ("tall","short"),("wide","narrow"),("thick","thin"),
                      ("broad","slim"),("vast","minute"),("giant","miniature")],
    "antonym_speed": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                      ("rapid","gradual"),("hasty","leisurely"),("brisk","languid")],
    "antonym_bright":[("bright","dark"),("light","gloomy"),("shiny","dull"),
                      ("vivid","faded"),("radiant","dim"),("gleaming","murky")],
    "capital":       [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                      ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                      ("India","Delhi"),("Russia","Moscow"),("Brazil","Brasilia"),
                      ("Greece","Athens"),("Egypt","Cairo"),("Mexico","Mexico")],
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
def ok_pairs(pairs):
    return [(a,b) for a,b in pairs if is_single(a) and is_single(b)]

# ── Build difference matrices ─────────────────────────────────────────
print("Building difference matrices ...")
diff_mats = {}
mean_dirs = {}
for pname, pairs in PARADIGMS.items():
    p = ok_pairs(pairs)
    if not p: continue
    diffs = np.array([get_emb(b) - get_emb(a) for a,b in p], dtype=np.float64)
    diff_mats[pname] = diffs
    mean_dirs[pname] = normed(np.mean([normed(d) for d in diffs], axis=0))
    print(f"  {pname:<18}  n={len(p)}")
print()

def get_svd_basis(diffs, k=None):
    """Return orthonormal basis for the row space of diffs matrix."""
    U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    if k is None: k = len(S)
    return Vt[:k], S

def principal_angles(B1, B2):
    """
    Compute principal (canonical) angles between subspaces spanned by rows of B1, B2.
    B1: (k1, H), B2: (k2, H) — each row is a unit basis vector.
    Returns cos values of principal angles.
    """
    # Gram matrix
    G = B1 @ B2.T
    # SVD of Gram matrix gives cos of principal angles
    sv = np.linalg.svd(G, compute_uv=False)
    return np.clip(sv, 0, 1)  # clip numerical noise

# ── Part A: Paradigm subspace dimensionality ──────────────────────────
print("=" * 70)
print("PART A: Paradigm subspace dimensionality")
print("=" * 70)
print()
print(f"  {'paradigm':<18}  {'n':>3}  "
      f"{'S[0]':>8}  {'S[1]':>8}  {'S[2]':>8}  "
      f"{'var_k1':>7}  {'var_k2':>7}  {'var_k3':>7}  {'eff_rank':>8}")

svd_results = {}
for pname, diffs in diff_mats.items():
    basis, S = get_svd_basis(diffs, k=min(len(diffs), 10))
    total_var = float((S**2).sum())
    var_k1 = float(S[0]**2 / total_var)
    var_k2 = float((S[0]**2 + S[1]**2) / total_var) if len(S) > 1 else var_k1
    var_k3 = float((S[:3]**2).sum() / total_var) if len(S) > 2 else var_k2
    eff_rank = float((S**2).sum()**2 / (S**4).sum())  # participation ratio
    print(f"  {pname:<18}  {len(diffs):>3}  "
          f"{S[0]:>8.1f}  {S[1]:>8.1f}  {S[2] if len(S)>2 else 0:>8.1f}  "
          f"{var_k1:>7.3f}  {var_k2:>7.3f}  {var_k3:>7.3f}  {eff_rank:>8.2f}")
    svd_results[pname] = {
        "n": len(diffs), "S": S[:5].tolist(),
        "var_k1": var_k1, "var_k2": var_k2, "var_k3": var_k3,
        "eff_rank": eff_rank,
    }

# ── Part B: Principal angles between paradigm subspaces ───────────────
print()
print("=" * 70)
print("PART B: Principal angles between paradigm subspaces (k=3 basis)")
print("         cos(θ) values — 1.0 = same direction, 0.0 = orthogonal")
print("=" * 70)
print()

pnames = sorted(diff_mats.keys())
K = 3  # number of basis vectors to use

bases = {}
for pname in pnames:
    diffs = diff_mats[pname]
    B, S = get_svd_basis(diffs, k=min(K, len(diffs)))
    # Normalize each row to unit length (SVD rows are already unit)
    bases[pname] = B.astype(np.float64)

print(f"  Row headers = subspace A,  Column headers = subspace B")
print(f"  Each cell shows [max_cos, 2nd_cos, 3rd_cos] across the k=3 basis vectors")
print()

hdr = " ".join(f"{n[:10]:>12}" for n in pnames)
print(f"  {'':20} {hdr}")

pa_results = {}
for ni in pnames:
    row = f"  {ni:<20}"
    for nj in pnames:
        if ni == nj:
            row += f"  {'[1.00,---]':>12}"
        else:
            B1 = bases[ni]; B2 = bases[nj]
            cos_vals = principal_angles(B1, B2)
            c0 = f"{cos_vals[0]:.2f}" if len(cos_vals) > 0 else "---"
            c1 = f"{cos_vals[1]:.2f}" if len(cos_vals) > 1 else "---"
            cell = f"[{c0},{c1}]"
            row += f"  {cell:>12}"
            key = f"{ni}|{nj}"
            pa_results[key] = cos_vals[:3].tolist()
    print(row)

# Highlight strongest cross-paradigm angles
print()
print("  Pairs with max_cos > 0.15 (some shared structure):")
for ni in pnames:
    for nj in pnames:
        if ni >= nj: continue
        key = f"{ni}|{nj}"
        if key in pa_results and pa_results[key][0] > 0.15:
            print(f"    {ni:<20} <-> {nj:<20}  max_cos={pa_results[key][0]:.4f}")

# ── Part C: Antonym direction in degree subspace ──────────────────────
print()
print("=" * 70)
print("PART C: Antonym directions projected onto adj degree subspace")
print("=" * 70)
print()
print("  Q: Is d_antonym = α * d_degree + noise?")
print("  i.e., is the antonym direction in the degree subspace?")
print()

DEG_K = 5  # use 5-dimensional degree subspace
deg_diffs = diff_mats.get("adj_pos2sup")
if deg_diffs is not None:
    deg_basis, deg_S = get_svd_basis(deg_diffs, k=DEG_K)
    deg_var_k = [(deg_S[:i+1]**2).sum() / (deg_S**2).sum() for i in range(DEG_K)]
    print(f"  Degree subspace (k={DEG_K}) captures variance:")
    for i, v in enumerate(deg_var_k):
        print(f"    k={i+1}: {v:.3f}")
    print()

    for aname in ["antonym_size", "antonym_speed", "antonym_bright", "gender"]:
        if aname not in mean_dirs: continue
        d = mean_dirs[aname].astype(np.float64)
        # Project d onto degree subspace
        proj = deg_basis @ d  # (K,) coefficients
        d_in = deg_basis.T @ proj  # (H,) projection into subspace
        d_out = d - d_in  # component orthogonal to subspace
        frac_in  = float(np.linalg.norm(d_in)**2)
        frac_out = float(np.linalg.norm(d_out)**2)
        total    = frac_in + frac_out
        print(f"  {aname:<18}  "
              f"fraction in deg subspace: {frac_in/total:.3f}  "
              f"fraction out: {frac_out/total:.3f}")
        print(f"    coefficients on degree axes: "
              f"{' '.join(f'{c:>+.3f}' for c in proj[:5])}")
else:
    print("  adj_pos2sup not available")

# ── Part D: Intra-paradigm variance structure ─────────────────────────
print()
print("=" * 70)
print("PART D: Intra-paradigm variance structure — is each paradigm rank-1?")
print("=" * 70)
print()

print("  Is there meaningful 2nd/3rd dimension in each paradigm?")
print()
print(f"  {'paradigm':<18}  {'var1':>6}  {'var2':>6}  {'var3':>6}  {'S1/S2':>7}  {'S1/S3':>7}  rank?")
for pname in sorted(svd_results.keys()):
    r = svd_results[pname]
    S = r["S"]
    ratio12 = float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float("inf")
    ratio13 = float(S[0] / S[2]) if len(S) > 2 and S[2] > 0 else float("inf")
    rank = "rank-1" if ratio12 > 3.0 else ("rank-2" if ratio12 > 1.5 else "rank-N")
    print(f"  {pname:<18}  {r['var_k1']:>6.3f}  "
          f"{r['var_k2']-r['var_k1']:>6.3f}  "
          f"{r['var_k3']-r['var_k2']:>6.3f}  "
          f"{ratio12:>7.2f}  {ratio13:>7.2f}  {rank}")

# ── Part E: Universal paradigm coordinate ────────────────────────────
print()
print("=" * 70)
print("PART E: Is there a universal paradigm coordinate?")
print("        i.e., a single W_E direction that separates paradigm members?")
print("=" * 70)
print()

# Build a matrix of mean directions for all paradigms
# Project all training words from each paradigm onto this "universal axis"
# The universal axis = mean of all paradigm mean directions

all_mean_dirs = np.array([mean_dirs[p] for p in sorted(mean_dirs.keys())], dtype=np.float64)
universal = normed(np.mean(all_mean_dirs, axis=0))
print(f"  Universal direction computed from {len(all_mean_dirs)} paradigm mean directions")
print()
print(f"  cos(universal, each paradigm mean_dir):")
for pname in sorted(mean_dirs.keys()):
    c = cosine(universal, mean_dirs[pname])
    print(f"    {pname:<18}  cos={c:>+.4f}")

print()
print("  Projection of paradigm SOURCE words onto universal direction:")
for pname in ["adj_pos2sup", "gender", "plural", "past_tense", "capital"]:
    if pname not in diff_mats: continue
    pairs = PARADIGMS[pname]
    src_projs = []
    tgt_projs = []
    for a, b in ok_pairs(pairs):
        ea = get_emb(a); eb = get_emb(b)
        if ea is None or eb is None: continue
        src_projs.append(float(normed(ea) @ universal))
        tgt_projs.append(float(normed(eb) @ universal))
    if not src_projs: continue
    print(f"  {pname:<18}  src_mean={np.mean(src_projs):>+.3f} ± {np.std(src_projs):.3f}  "
          f"tgt_mean={np.mean(tgt_projs):>+.3f} ± {np.std(tgt_projs):.3f}  "
          f"delta={np.mean(tgt_projs)-np.mean(src_projs):>+.3f}")

output = {
    "svd_results": svd_results,
    "principal_angles": pa_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 239 complete.")
