#!/usr/bin/env python3
"""
Rotation Sign Prediction — The Final Missing Piece

Corrected oracle achieves 100% when the rotation sign is known.
But 7/23 go CCW, 16/23 go CW — the sign is word-specific.

Can we predict the rotation sign WITHOUT knowing the comparative form?

Key insight: all comparatives project positively onto e2_shared
(the comparative axis of the shared degree plane). When we try
both +π/φ and -π/φ rotations in the private plane, we can pick
the sign whose result has the LARGER positive e2_shared projection.

This is a purely geometric criterion that doesn't require knowing
the comparative form — only knowledge of the shared degree plane
axes (which are derived from the full training set).

Tests:
  A. SIGN_FROM_E2: predict sign by which rotation increases e2_shared
     component. Evaluate LOO accuracy.

  B. SIGN_FROM_TARGET_CLUSTER: predict sign by proximity to the
     centroid of known comparative embeddings (LOO).

  C. FULL GEOMETRIC PIPELINE:
     Given only emb(pos) and the shared degree plane + circle center:
     1. Estimate private plane from k nearest training words
     2. Rotate by π/φ with sign from e2_shared criterion
     3. Evaluate LOO accuracy
     Compare to mean_dir baseline (22/23 = 95.7%).

  D. SIGN PATTERN ANALYSIS: what property of emb(pos) correlates
     with rotation sign? E1_shared, e2_shared projections?
     Embedding norm? Semantic cluster?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "sign_predict.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2
OMEGA_STEP = 180 / PHI  # π/φ degrees

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

pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))
E_pool = np.array(pool_embs, dtype=np.float32)
En = (E_pool / (np.linalg.norm(E_pool, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
print(f"  Pool: {len(pool_words)}\n")

def top1(q, exclude=None):
    qn = normed(q).astype(np.float32)
    sims = En @ qn
    for idx in np.argsort(-sims):
        w = pool_words[idx]
        if exclude and w in exclude: continue
        return w
    return None

def svd_plane(P, C, S):
    v1 = C - P; v2 = S - P
    D = np.stack([v1, v2], axis=1)
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    if sv[1] < 1e-10: return None, None
    return U[:,0], U[:,1]

def circumscribed_2d(p2, q2, r2):
    ax, ay = p2; bx, by = q2; cx, cy = r2
    D = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    if abs(D) < 1e-12: return None
    ux = ((ax**2+ay**2)*(by-cy)+(bx**2+by**2)*(cy-ay)+(cx**2+cy**2)*(ay-by))/D
    uy = ((ax**2+ay**2)*(cx-bx)+(bx**2+by**2)*(ax-cx)+(cx**2+cy**2)*(bx-ax))/D
    return float(ux), float(uy), float(np.sqrt((ax-ux)**2+(ay-uy)**2))

def signed_arc_deg(ang1, ang2):
    d = ang2 - ang1
    while d >  180: d -= 360
    while d < -180: d += 360
    return d

def rotate_around_center(v, e1, e2, cx_2d, cy_2d, angle_rad):
    a = float(np.dot(v, e1)); b = float(np.dot(v, e2))
    a_c = a - cx_2d; b_c = b - cy_2d
    cos_a = np.cos(angle_rad); sin_a = np.sin(angle_rad)
    a_r = a_c * cos_a - b_c * sin_a + cx_2d
    b_r = a_c * sin_a + b_c * cos_a + cy_2d
    v_perp = v - a * e1 - b * e2
    return v_perp + a_r * e1 + b_r * e2

# ── Precompute private planes, circle centers, and actual signs ───────
triples_info = {}
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C_emb = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C_emb is None or S is None: continue
    e1, e2 = svd_plane(P, C_emb, S)
    if e1 is None: continue
    v1 = C_emb - P; v2 = S - P
    p2 = np.array([0.,0.])
    c2 = np.array([float(np.dot(v1,e1)), float(np.dot(v1,e2))])
    s2 = np.array([float(np.dot(v2,e1)), float(np.dot(v2,e2))])
    circ = circumscribed_2d(p2, c2, s2)
    if circ is None: continue
    cx, cy, R = circ
    ang_p = np.degrees(np.arctan2(p2[1]-cy, p2[0]-cx))
    ang_c = np.degrees(np.arctan2(c2[1]-cy, c2[0]-cx))
    arc_pc = signed_arc_deg(ang_p, ang_c)
    triples_info[pos_w] = {
        "e1": e1, "e2": e2, "cx": cx, "cy": cy, "R": R,
        "arc_pc": arc_pc, "sign": np.sign(arc_pc),
        "P": P, "C": C_emb, "S": S,
        "comp_w": comp_w, "sup_w": sup_w,
    }

n_total = len(triples_info)
print(f"  Private planes computed: {n_total}\n")

# ── Shared degree plane (LOO) ─────────────────────────────────────────
def get_shared_plane_and_center(exclude_words=None):
    diffs = []
    for pos_w, d in triples_info.items():
        if exclude_words and pos_w in exclude_words: continue
        C_emb = d["C"]; S = d["S"]; P = d["P"]
        diffs.append(C_emb - P); diffs.append(S - P)
    if not diffs: return None, None, None
    D = np.array(diffs).T
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    e1_sh = U[:,0]; e2_sh = U[:,1]
    # Mean circle center in shared plane coordinates
    # The circle center in H-dim: P + cx*e1 + cy*e2, project onto (e1_sh, e2_sh)
    centers_sh = []
    for pos_w, d in triples_info.items():
        if exclude_words and pos_w in exclude_words: continue
        center_H = d["P"] + d["cx"] * d["e1"] + d["cy"] * d["e2"]
        cx_sh = float(np.dot(center_H, e1_sh))
        cy_sh = float(np.dot(center_H, e2_sh))
        centers_sh.append((cx_sh, cy_sh))
    if centers_sh:
        cx_mean = np.mean([c[0] for c in centers_sh])
        cy_mean = np.mean([c[1] for c in centers_sh])
    else:
        cx_mean = cy_mean = 0.0
    return e1_sh, e2_sh, (cx_mean, cy_mean)

def get_local_plane(pos_w, k=5, exclude_words=None):
    P = triples_info[pos_w]["P"]
    sims = []
    for pw, d in triples_info.items():
        if pw == pos_w: continue
        if exclude_words and pw in exclude_words: continue
        s = float(np.dot(normed(P), normed(d["P"])))
        sims.append((s, pw))
    sims.sort(reverse=True)
    top_k = sims[:k]
    total_w = sum(max(0, s) for s, _ in top_k)
    if total_w < 1e-8: return None, None, None
    e1_est = np.zeros(H); e2_est = np.zeros(H)
    cx_est = 0.; cy_est = 0.
    for sim, pw in top_k:
        w = max(0, sim) / total_w
        e1_est += w * triples_info[pw]["e1"]
        e2_est += w * triples_info[pw]["e2"]
        cx_est += w * triples_info[pw]["cx"]
        cy_est += w * triples_info[pw]["cy"]
    basis = np.stack([e1_est, e2_est], axis=1)
    U, sv, _ = np.linalg.svd(basis, full_matrices=False)
    if sv[1] < 1e-10: return None, None, None
    return U[:,0], U[:,1], (cx_est, cy_est)

# ── Part A: Sign prediction from e2_shared ───────────────────────────
print("=" * 70)
print("PART A: SIGN PREDICTION from e2_shared (comparative axis)")
print("        Choose sign that puts rotation result in +e2 region.")
print("        Purely geometric: uses only shared plane, no oracle.")
print("=" * 70)
print()

print(f"  {'word':<8}  {'actual_sign':>12}  {'pred_sign':>10}  {'correct':>8}  "
      f"{'e2+rot':>8}  {'e2-rot':>8}  {'pred_word':>12}  ok")

sign_e2_acc = 0; full_geo_acc = 0; n = 0
results_A = []

words_list = list(triples_info.keys())
for pos_w in words_list:
    d = triples_info[pos_w]
    P = d["P"]; comp_w = d["comp_w"]

    e1_sh, e2_sh, (cx_sh, cy_sh) = get_shared_plane_and_center(exclude_words={pos_w})
    if e1_sh is None: continue

    e1_loc, e2_loc, (cx_loc, cy_loc) = get_local_plane(pos_w, k=5, exclude_words={pos_w})
    if e1_loc is None:
        e1_loc, e2_loc = e1_sh, e2_sh; cx_loc, cy_loc = cx_sh, cy_sh

    # Try both signs in the local plane around local center
    rot_pos = rotate_around_center(P, e1_loc, e2_loc, cx_loc, cy_loc,
                                   np.radians(+OMEGA_STEP))
    rot_neg = rotate_around_center(P, e1_loc, e2_loc, cx_loc, cy_loc,
                                   np.radians(-OMEGA_STEP))

    # Predict sign: pick the one with larger e2_shared projection
    e2_pos = float(np.dot(rot_pos, e2_sh))
    e2_neg = float(np.dot(rot_neg, e2_sh))
    pred_sign = +1 if e2_pos > e2_neg else -1
    actual_sign = int(np.sign(d["arc_pc"]))
    sign_ok = (pred_sign == actual_sign)
    if sign_ok: sign_e2_acc += 1

    # Full geometric prediction: use predicted sign
    rot_pred = rot_pos if pred_sign > 0 else rot_neg
    pred_word = top1(rot_pred, exclude={pos_w})
    full_ok = (pred_word == comp_w)
    if full_ok: full_geo_acc += 1
    n += 1

    print(f"  {pos_w:<8}  {actual_sign:>12}  {pred_sign:>10}  "
          f"{'Y' if sign_ok else 'N':>8}  {e2_pos:>8.4f}  {e2_neg:>8.4f}  "
          f"{pred_word:<12}  {'Y' if full_ok else 'N'}")
    results_A.append({"word": pos_w, "actual_sign": actual_sign,
                       "pred_sign": pred_sign, "sign_ok": sign_ok,
                       "full_ok": full_ok, "e2_pos": e2_pos, "e2_neg": e2_neg})

print()
print(f"  Sign prediction accuracy: {sign_e2_acc}/{n} = {sign_e2_acc/n:.3f}")
print(f"  Full geometric pipeline:  {full_geo_acc}/{n} = {full_geo_acc/n:.3f}")
print(f"  (mean_dir baseline: 22/23 = 0.957)")

# ── Part B: Sign from target cluster centroid ──────────────────────
print()
print("=" * 70)
print("PART B: SIGN PREDICTION from comparative cluster centroid (LOO)")
print("        Choose sign closer to LOO centroid of all comp embeddings")
print("=" * 70)
print()

cluster_acc = 0
for pos_w in words_list:
    d = triples_info[pos_w]
    P = d["P"]; comp_w = d["comp_w"]
    # LOO centroid of comparatives
    comp_embs = [triples_info[pw]["C"] for pw in words_list if pw != pos_w]
    centroid = np.mean(comp_embs, axis=0)
    e1_loc, e2_loc, (cx_loc, cy_loc) = get_local_plane(pos_w, k=5, exclude_words={pos_w})
    if e1_loc is None:
        e1_sh, e2_sh, (cx_sh, cy_sh) = get_shared_plane_and_center(exclude_words={pos_w})
        e1_loc, e2_loc = e1_sh, e2_sh; cx_loc, cy_loc = cx_sh, cy_sh
    rot_pos = rotate_around_center(P, e1_loc, e2_loc, cx_loc, cy_loc,  np.radians(+OMEGA_STEP))
    rot_neg = rotate_around_center(P, e1_loc, e2_loc, cx_loc, cy_loc, np.radians(-OMEGA_STEP))
    cos_pos = float(np.dot(normed(rot_pos), normed(centroid)))
    cos_neg = float(np.dot(normed(rot_neg), normed(centroid)))
    rot_pred = rot_pos if cos_pos > cos_neg else rot_neg
    pred_word = top1(rot_pred, exclude={pos_w})
    if pred_word == comp_w: cluster_acc += 1

print(f"  Sign from comp cluster centroid: {cluster_acc}/{n} = {cluster_acc/n:.3f}")

# ── Part C: Sign from majority vote (baseline) ───────────────────────
print()
print("=" * 70)
print("PART C: MAJORITY VOTE BASELINE")
print("        Most words rotate CW (negative). Predict -π/φ always.")
print("=" * 70)
print()
majority_acc = 0
for pos_w in words_list:
    d = triples_info[pos_w]
    P = d["P"]; comp_w = d["comp_w"]
    e1_loc, e2_loc, (cx_loc, cy_loc) = get_local_plane(pos_w, k=5, exclude_words={pos_w})
    if e1_loc is None:
        e1_sh, e2_sh, (cx_sh, cy_sh) = get_shared_plane_and_center(exclude_words={pos_w})
        e1_loc, e2_loc = e1_sh, e2_sh; cx_loc, cy_loc = cx_sh, cy_sh
    # Always use CW (-π/φ) since 16/23 go CW
    rot_pred = rotate_around_center(P, e1_loc, e2_loc, cx_loc, cy_loc, np.radians(-OMEGA_STEP))
    pred_word = top1(rot_pred, exclude={pos_w})
    if pred_word == comp_w: majority_acc += 1

print(f"  Majority vote (always CW): {majority_acc}/{n} = {majority_acc/n:.3f}")

# ── Part D: What property of emb(pos) predicts sign? ─────────────────
print()
print("=" * 70)
print("PART D: GEOMETRIC PROPERTIES vs ROTATION SIGN")
print("        What correlates with the sign?")
print("=" * 70)
print()

e1_sh, e2_sh, _ = get_shared_plane_and_center()

props = {}
for pos_w, d in triples_info.items():
    P = d["P"]; s = d["sign"]
    props[pos_w] = {
        "sign": s,
        "norm": float(np.linalg.norm(P)),
        "e1_proj": float(np.dot(P, e1_sh)),
        "e2_proj": float(np.dot(P, e2_sh)),
        "e1_e2_ratio": float(np.dot(P, e1_sh) / (abs(np.dot(P, e2_sh)) + 1e-8)),
        "R": d["R"],
    }

# Compute correlation of each property with sign
for prop_name in ["norm", "e1_proj", "e2_proj", "R"]:
    vals = np.array([props[w][prop_name] for w in words_list if w in props])
    signs = np.array([props[w]["sign"] for w in words_list if w in props])
    corr = float(np.corrcoef(vals, signs)[0, 1])
    sep = abs(np.mean(vals[signs > 0]) - np.mean(vals[signs < 0]))
    print(f"  {prop_name:<12}  corr={corr:>+.3f}  "
          f"mean(CCW)={np.mean(vals[signs>0]):.4f}  "
          f"mean(CW)={np.mean(vals[signs<0]):.4f}  "
          f"separation={sep:.4f}")

# Can e2_proj of emb(pos) predict sign?
e2_vals = np.array([props[w]["e2_proj"] for w in words_list if w in props])
signs    = np.array([props[w]["sign"]    for w in words_list if w in props])
threshold = 0.0  # sign=+1 if e2_proj > threshold else -1
pred_signs = np.where(e2_vals > threshold, +1, -1)
sign_acc_e2proj = float(np.mean(pred_signs == signs))
print()
print(f"  Sign from e2_proj of pos (threshold={threshold}): "
      f"{int(sign_acc_e2proj*n)}/{n} = {sign_acc_e2proj:.3f}")

# Optimal threshold
thresholds = np.linspace(e2_vals.min(), e2_vals.max(), 100)
best_t, best_acc = 0, 0
for t in thresholds:
    acc = float(np.mean(np.where(e2_vals > t, +1, -1) == signs))
    if acc > best_acc: best_acc = acc; best_t = t
print(f"  Optimal threshold: e2_proj > {best_t:.4f}: "
      f"{int(best_acc*n)}/{n} = {best_acc:.3f}")

print()
print("=" * 70)
print("SUMMARY: Full geometric pipeline accuracy")
print("=" * 70)
print()
print(f"  mean_dir (baseline):         22/{n} = {22/n:.3f}")
print(f"  Arc rotation (sign from e2): {full_geo_acc}/{n} = {full_geo_acc/n:.3f}")
print(f"  Arc rotation (cluster cen):  {cluster_acc}/{n} = {cluster_acc/n:.3f}")
print(f"  Arc rotation (majority CW):  {majority_acc}/{n} = {majority_acc/n:.3f}")
print(f"  Oracle (sign known):         23/{n} = {23/n:.3f}")

output = {
    "sign_e2_acc": sign_e2_acc / n,
    "full_geo_acc": full_geo_acc / n,
    "cluster_acc": cluster_acc / n,
    "majority_acc": majority_acc / n,
    "oracle_acc": 23 / n,
    "mean_dir_acc": 22 / n,
    "results_A": results_A,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Sign prediction analysis complete.")
