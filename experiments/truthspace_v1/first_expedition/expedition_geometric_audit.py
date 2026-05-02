#!/usr/bin/env python3
"""
Geometric Audit of W_E Morphological Transformations

The TruthSpace hypothesis says: geometry IS computation.
This means we should not be asking "does the mean direction retrieve
the right word?" but rather: "WHAT GEOMETRIC TRANSFORMATION maps
emb(A) to emb(B) in W_E?"

This audit answers:

  A. NORM AUDIT: Is ||emb(b)|| / ||emb(a)|| constant across a paradigm?
     A pure rotation preserves norm. A translation does not.

  B. LINEAR MAP: What is M such that M @ emb(a) ≈ emb(b)?
     Compute via least-squares: M = B @ pinv(A).
     SVD of M: if all singular values ≈ 1, M is approximately an isometry.

  C. TRANSLATION vs ROTATION residuals:
     Model 1 (translation): b = a + d
     Model 2 (rotation):    b = M @ a
     Which fits better? Residual = mean ||b_pred - b_true||₂

  D. ANGLE GEOMETRY for adj_degree:
     On the unit hypersphere, compute the spherical arc lengths:
       θ(pos→comp) = arccos(cos(pos, comp))
       θ(comp→sup) = arccos(cos(comp, sup))
       θ(pos→sup)  = arccos(cos(pos,  sup))
     Is θ(pos→comp) ≈ θ(comp→sup)? (equal steps on sphere)
     Is θ(pos→sup) ≈ θ(pos→comp) + θ(comp→sup)? (curved path, not straight)
     Is the ratio θ(pos→comp) / θ(comp→sup) related to φ?

  E. GEODESIC TEST:
     On the unit sphere, the great circle from normed(pos) to normed(sup)
     passes through some intermediate point. Is normed(comp) on this geodesic?
     Measure: how far is normed(comp) from the great circle?

  F. ROTATION MATRIX STRUCTURE:
     Decompose M = U S V^T for each paradigm.
     What are the top singular values?
     How many singular values deviate from 1? (rank of deviation from identity)
     What is (M - I)? Low-rank structure reveals the geometry.

  G. φ-AUDIT:
     For each paradigm, compute the characteristic angle θ = mean(arccos(cos(a,b))).
     Is θ related to arccos(1/φ) = 51.83° or arccos(1/φ²) = 67.97°?
     Is the norm ratio |b|/|a| related to φ or 1/φ?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "geometric_audit.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2  # 1.6180...

PARADIGMS = {
    "adj_pos2sup":  [("big","biggest"),("fast","fastest"),("long","longest"),
                     ("small","smallest"),("hard","hardest"),("bright","brightest"),
                     ("dark","darkest"),("rich","richest"),("deep","deepest"),
                     ("wide","widest"),("high","highest"),("low","lowest"),
                     ("old","oldest"),("young","youngest"),("hot","hottest"),
                     ("tall","tallest"),("strong","strongest"),("weak","weakest"),
                     ("short","shortest"),("cool","coolest"),("great","greatest"),
                     ("safe","safest"),("cheap","cheapest"),("clean","cleanest")],
    "adj_pos2comp": [("big","bigger"),("fast","faster"),("long","longer"),
                     ("small","smaller"),("hard","harder"),("bright","brighter"),
                     ("dark","darker"),("rich","richer"),("deep","deeper"),
                     ("wide","wider"),("high","higher"),("low","lower"),
                     ("old","older"),("young","younger"),("hot","hotter"),
                     ("tall","taller"),("strong","stronger"),("weak","weaker"),
                     ("short","shorter"),("cool","cooler"),("great","greater"),
                     ("safe","safer"),("cheap","cheaper"),("clean","cleaner")],
    "adj_comp2sup": [("bigger","biggest"),("faster","fastest"),("longer","longest"),
                     ("smaller","smallest"),("harder","hardest"),("brighter","brightest"),
                     ("darker","darkest"),("richer","richest"),("deeper","deepest"),
                     ("wider","widest"),("higher","highest"),("lower","lowest"),
                     ("older","oldest"),("younger","youngest"),("hotter","hottest"),
                     ("taller","tallest"),("stronger","strongest"),("weaker","weakest"),
                     ("shorter","shortest"),("cooler","coolest")],
    "gender":       [("king","queen"),("man","woman"),("boy","girl"),
                     ("prince","princess"),("actor","actress"),("hero","heroine"),
                     ("monk","nun"),("duke","duchess"),("lord","lady"),
                     ("wizard","witch"),("nephew","niece"),("lion","lioness"),
                     ("father","mother"),("son","daughter"),("brother","sister"),
                     ("husband","wife"),("grandfather","grandmother")],
    "plural":       [("cat","cats"),("dog","dogs"),("house","houses"),
                     ("tree","trees"),("book","books"),("car","cars"),
                     ("bird","birds"),("ship","ships"),("hand","hands"),
                     ("door","doors"),("king","kings"),("boy","boys"),
                     ("word","words"),("stone","stones"),("cloud","clouds"),
                     ("road","roads"),("horse","horses"),("town","towns")],
    "past_tense":   [("walk","walked"),("talk","talked"),("call","called"),
                     ("pull","pulled"),("fill","filled"),("turn","turned"),
                     ("look","looked"),("move","moved"),("push","pushed"),
                     ("help","helped"),("play","played"),("stay","stayed"),
                     ("lock","locked"),("jump","jumped"),("land","landed"),
                     ("ask","asked"),("work","worked"),("open","opened")],
    "antonym_size": [("big","small"),("large","tiny"),("huge","little"),
                     ("tall","short"),("wide","narrow"),("thick","thin"),
                     ("broad","slim"),("heavy","light"),("long","brief"),
                     ("grand","modest"),("vast","minute"),("giant","miniature")],
    "capital":      [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                     ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                     ("India","Delhi"),("Russia","Moscow"),("Greece","Athens"),
                     ("Egypt","Cairo"),("Poland","Warsaw"),("Turkey","Ankara")],
}

ADJ_DEGREE_TRIPLES = [
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
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def angle_deg(a, b):
    c = float(np.clip(np.dot(normed(a), normed(b)), -1, 1))
    return float(np.degrees(np.arccos(c)))
def angle_rad(a, b):
    c = float(np.clip(np.dot(normed(a), normed(b)), -1, 1))
    return float(np.arccos(c))

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
def is_single(w): return get_emb(w) is not None
def ok_pairs(pairs): return [(a,b) for a,b in pairs if is_single(a) and is_single(b)]

# ── Part A: Norm audit ────────────────────────────────────────────────
print("=" * 70)
print("PART A: NORM AUDIT — is the transformation norm-preserving?")
print("        A rotation has ||b||/||a|| = 1 exactly.")
print("=" * 70)
print()
print(f"  {'paradigm':<16}  {'norm_a':>8}  {'norm_b':>8}  {'ratio_b/a':>10}  "
      f"{'ratio_std':>10}  {'type'}")

norm_results = {}
for pname, pairs in PARADIGMS.items():
    p = ok_pairs(pairs)
    if not p: continue
    norms_a = np.array([np.linalg.norm(get_emb(a)) for a,b in p])
    norms_b = np.array([np.linalg.norm(get_emb(b)) for a,b in p])
    ratios  = norms_b / norms_a
    r_mean = float(ratios.mean()); r_std = float(ratios.std())
    kind = "ISOMETRY" if r_std < 0.02 and abs(r_mean - 1) < 0.05 else \
           "SCALING" if r_std < 0.05 else "VARIABLE"
    print(f"  {pname:<16}  {norms_a.mean():>8.2f}  {norms_b.mean():>8.2f}  "
          f"{r_mean:>10.4f}  {r_std:>10.4f}  {kind}")
    norm_results[pname] = {"norm_a": float(norms_a.mean()), "norm_b": float(norms_b.mean()),
                           "ratio_mean": r_mean, "ratio_std": r_std}

# ── Part B: Linear map M ──────────────────────────────────────────────
print()
print("=" * 70)
print("PART B: LINEAR MAP — what M maps emb(a) → emb(b)?")
print("        M = B @ pinv(A).  SVD of M: singular values near 1 → rotation.")
print("        Residual: mean ||M@a - b||₂")
print("=" * 70)
print()
print(f"  {'paradigm':<16}  {'S_min':>7}  {'S_mean':>7}  {'S_max':>7}  "
      f"{'S_std':>7}  {'rank(M-I)':>10}  {'resid':>8}")

map_results = {}
for pname, pairs in PARADIGMS.items():
    p = ok_pairs(pairs)
    if len(p) < 4: continue
    A = np.array([get_emb(a) for a,b in p]).T  # (H, n)
    B = np.array([get_emb(b) for a,b in p]).T  # (H, n)

    # Least-squares: M @ A ≈ B  =>  M = B @ pinv(A)
    # pinv via SVD (truncated for stability)
    Ua, Sa, Vta = np.linalg.svd(A, full_matrices=False)
    # Only use components with singular value > 0.1
    thresh = 0.1 * Sa[0]
    k_use = int((Sa > thresh).sum())
    A_pinv = (Vta[:k_use].T * (1.0 / Sa[:k_use])) @ Ua[:, :k_use].T
    M = B @ A_pinv  # (H, H) — but rank at most n

    # SVD of M to characterize the transformation
    # M has rank at most n (n pairs), so SVD is limited
    # Instead: compute M^T M (Gram matrix in output space)
    # Better: directly SVD M (but H=1536 × H=1536 is expensive)
    # Approximate: compute singular values of M restricted to the span of A
    # M @ Ua[:, :k_use] = B @ A_pinv @ Ua[:, :k_use]
    M_small = M @ Ua[:, :k_use]  # H × k_use
    _, Sm, _ = np.linalg.svd(M_small, full_matrices=False)

    # Residual
    B_pred = M @ A  # H × n
    resid  = float(np.linalg.norm(B_pred - B, 'fro') / np.sqrt(len(p)))

    # Rank of (M - I): how many dimensions does M deviate from identity?
    # Approximate: measure ||M @ Ua[:, :k_use] - Ua[:, :k_use]||_F
    dev = float(np.linalg.norm(M_small - Ua[:, :k_use], 'fro'))

    print(f"  {pname:<16}  {Sm.min():>7.3f}  {Sm.mean():>7.3f}  {Sm.max():>7.3f}  "
          f"{Sm.std():>7.3f}  {dev:>10.3f}  {resid:>8.3f}")
    map_results[pname] = {"S_min": float(Sm.min()), "S_mean": float(Sm.mean()),
                          "S_max": float(Sm.max()), "S_std": float(Sm.std()),
                          "dev_from_I": dev, "residual": resid}

# ── Part C: Translation vs rotation residuals ─────────────────────────
print()
print("=" * 70)
print("PART C: TRANSLATION vs ROTATION — which model fits better?")
print("        Translation: b = a + d (additive, word-independent)")
print("        Rotation:    b = M @ a (multiplicative)")
print("        Normed translation: b/||b|| ≈ normed(a + d)")
print("=" * 70)
print()
print(f"  {'paradigm':<16}  {'transl_resid':>13}  {'rot_resid':>10}  winner")

comp_results = {}
for pname, pairs in PARADIGMS.items():
    p = ok_pairs(pairs)
    if len(p) < 4: continue
    A_list = [get_emb(a) for a,b in p]
    B_list = [get_emb(b) for a,b in p]

    # Translation model: d = mean(b - a), residual = mean ||a+d - b||
    diffs = [B_list[i] - A_list[i] for i in range(len(p))]
    d_mean = np.mean(diffs, axis=0)
    transl_resid = float(np.mean([np.linalg.norm((A_list[i] + d_mean) - B_list[i])
                                   for i in range(len(p))]))

    # Rotation model M
    A = np.array(A_list).T; B = np.array(B_list).T
    Ua, Sa, Vta = np.linalg.svd(A, full_matrices=False)
    thresh = 0.1 * Sa[0]
    k_use  = int((Sa > thresh).sum())
    A_pinv = (Vta[:k_use].T * (1.0 / Sa[:k_use])) @ Ua[:, :k_use].T
    M      = B @ A_pinv
    B_pred = M @ A
    rot_resid = float(np.linalg.norm(B_pred - B, 'fro') / np.sqrt(len(p)))

    winner = "ROTATION" if rot_resid < transl_resid else "TRANSLATION"
    print(f"  {pname:<16}  {transl_resid:>13.3f}  {rot_resid:>10.3f}  {winner}")
    comp_results[pname] = {"translation_residual": transl_resid,
                           "rotation_residual": rot_resid}

# ── Part D: Angle geometry for adj_degree triples ─────────────────────
print()
print("=" * 70)
print("PART D: ANGLE GEOMETRY on the unit sphere (adj_degree triples)")
print("        θ = arccos(cos(a, b)) in degrees")
print("        Is θ(pos→comp) + θ(comp→sup) = θ(pos→sup)? (curved path?)")
print("        Ratio θ(pos→comp) / θ(comp→sup) = ?  (compare to φ)")
print("=" * 70)
print()
print(f"  {'word':<8}  {'θ(p→c)':>8}  {'θ(c→s)':>8}  {'θ(p→s)':>8}  "
      f"{'sum_steps':>10}  {'excess':>8}  {'ratio':>7}  {'close_to'}")

angle_rows = []
for pos, comp, sup in ADJ_DEGREE_TRIPLES:
    ep = get_emb(pos); ec = get_emb(comp); es = get_emb(sup)
    if ep is None or ec is None or es is None: continue
    th_pc = angle_deg(ep, ec)
    th_cs = angle_deg(ec, es)
    th_ps = angle_deg(ep, es)
    excess = (th_pc + th_cs) - th_ps  # > 0 means curved (not straight line)
    ratio  = th_pc / th_cs if th_cs > 0 else float('inf')
    # What is ratio close to?
    close = ""
    if abs(ratio - PHI) < 0.1:        close = "φ"
    elif abs(ratio - 1/PHI) < 0.1:    close = "1/φ"
    elif abs(ratio - 1.0) < 0.05:     close = "1"
    elif abs(ratio - PHI**2) < 0.2:   close = "φ²"
    print(f"  {pos:<8}  {th_pc:>8.3f}  {th_cs:>8.3f}  {th_ps:>8.3f}  "
          f"{th_pc+th_cs:>10.3f}  {excess:>8.3f}  {ratio:>7.4f}  {close}")
    angle_rows.append({"word": pos, "theta_pc": th_pc, "theta_cs": th_cs,
                       "theta_ps": th_ps, "excess": excess, "ratio": ratio})

# Summary statistics
if angle_rows:
    ratios   = [r["ratio"] for r in angle_rows]
    excesses = [r["excess"] for r in angle_rows]
    th_pcs   = [r["theta_pc"] for r in angle_rows]
    th_css   = [r["theta_cs"] for r in angle_rows]
    th_pss   = [r["theta_ps"] for r in angle_rows]
    print()
    print(f"  SUMMARY:")
    print(f"    θ(pos→comp): mean={np.mean(th_pcs):.3f}°  std={np.std(th_pcs):.3f}°")
    print(f"    θ(comp→sup): mean={np.mean(th_css):.3f}°  std={np.std(th_css):.3f}°")
    print(f"    θ(pos→sup):  mean={np.mean(th_pss):.3f}°  std={np.std(th_pss):.3f}°")
    print(f"    ratio θ(p→c)/θ(c→s): mean={np.mean(ratios):.4f}  std={np.std(ratios):.4f}")
    print(f"    φ = {PHI:.4f},  1/φ = {1/PHI:.4f},  φ² = {PHI**2:.4f}")
    print(f"    excess (path curvature): mean={np.mean(excesses):.3f}°  std={np.std(excesses):.3f}°")
    print(f"    excess/θ(p→s): mean={np.mean([r['excess']/r['theta_ps'] for r in angle_rows]):.3f}")

# ── Part E: Geodesic test ─────────────────────────────────────────────
print()
print("=" * 70)
print("PART E: GEODESIC TEST — is normed(comp) on the great circle pos→sup?")
print("        On the unit sphere, great circle = slerp(normed(pos), normed(sup), t)")
print("        Find t* that minimises distance to normed(comp).")
print("        If comp is ON the geodesic, the residual = 0.")
print("=" * 70)
print()
print(f"  {'word':<8}  {'t*':>6}  {'gc_dist°':>9}  {'arc_frac':>9}  on_geodesic?")

geo_rows = []
for pos, comp, sup in ADJ_DEGREE_TRIPLES:
    ep = get_emb(pos); ec = get_emb(comp); es = get_emb(sup)
    if ep is None or ec is None or es is None: continue
    np_ = normed(ep); nc = normed(ec); ns = normed(es)

    # SLERP: slerp(p, s, t) = sin((1-t)ω)*p/sin(ω) + sin(t*ω)*s/sin(ω)
    # where ω = arccos(p·s)
    omega = angle_rad(ep, es)
    if abs(np.sin(omega)) < 1e-8:
        print(f"  {pos:<8}  (degenerate)")
        continue

    # Find t* minimising ||slerp(pos,sup,t) - comp||
    # Scan t in [0,1]
    best_t, best_dist = 0.0, float("inf")
    for t in np.linspace(0, 1, 1000):
        slp = (np.sin((1-t)*omega) * np_ + np.sin(t*omega) * ns) / np.sin(omega)
        dist = float(np.degrees(np.arccos(np.clip(float(np.dot(slp, nc)), -1, 1))))
        if dist < best_dist:
            best_dist = dist; best_t = float(t)

    # t* = fraction of the arc at which comp is closest to the geodesic
    # For equidistant: t* should = θ(pos→comp) / θ(pos→sup)
    arc_frac = angle_rad(ep, ec) / (angle_rad(ep, es) + 1e-8)
    on_geo = "YES" if best_dist < 1.0 else f"NO ({best_dist:.2f}°)"
    print(f"  {pos:<8}  {best_t:>6.3f}  {best_dist:>9.4f}  {arc_frac:>9.4f}  {on_geo}")
    geo_rows.append({"word": pos, "t_star": best_t, "gc_dist": best_dist,
                     "arc_frac": arc_frac})

if geo_rows:
    gd = [r["gc_dist"] for r in geo_rows]
    ts = [r["t_star"]  for r in geo_rows]
    af = [r["arc_frac"] for r in geo_rows]
    print(f"\n  SUMMARY:")
    print(f"    Geodesic distance: mean={np.mean(gd):.4f}°  max={max(gd):.4f}°")
    print(f"    t* (where comp sits):    mean={np.mean(ts):.4f}  std={np.std(ts):.4f}")
    print(f"    arc_frac (expected t*):  mean={np.mean(af):.4f}  std={np.std(af):.4f}")
    print(f"    t* - arc_frac (offset):  mean={np.mean([r['t_star']-r['arc_frac'] for r in geo_rows]):.4f}")

# ── Part F: Rotation matrix structure ────────────────────────────────
print()
print("=" * 70)
print("PART F: ROTATION MATRIX STRUCTURE — SVD of (M - I)")
print("        (M - I) reveals HOW M differs from identity.")
print("        Low-rank (M-I) → M is close to a rank-k rotation.")
print("=" * 70)
print()
print(f"  {'paradigm':<16}  {'||M-I||_F':>10}  {'rank_0.1':>9}  "
      f"{'S1(M-I)':>10}  {'S2(M-I)':>10}")

for pname in ["adj_pos2sup","adj_pos2comp","gender","plural","past_tense","antonym_size"]:
    pairs = PARADIGMS.get(pname, [])
    p = ok_pairs(pairs)
    if len(p) < 4: continue
    A = np.array([get_emb(a) for a,b in p]).T
    B = np.array([get_emb(b) for a,b in p]).T
    Ua, Sa, Vta = np.linalg.svd(A, full_matrices=False)
    k = int((Sa > 0.1 * Sa[0]).sum())
    A_pinv = (Vta[:k].T * (1.0/Sa[:k])) @ Ua[:,:k].T
    M = B @ A_pinv

    # Project M onto the span of A (the only part we can measure)
    M_sub  = Ua[:,:k].T @ M @ Ua[:,:k]   # k×k matrix in A's subspace
    I_sub  = np.eye(k)
    dM_sub = M_sub - I_sub
    _, S_dM, _ = np.linalg.svd(dM_sub)
    frob   = float(np.linalg.norm(dM_sub, 'fro'))
    rank01 = int((S_dM > 0.1 * S_dM[0]).sum())
    print(f"  {pname:<16}  {frob:>10.3f}  {rank01:>9}  "
          f"{S_dM[0]:>10.3f}  {S_dM[1] if len(S_dM)>1 else 0:>10.3f}")

# ── Part G: φ-audit ───────────────────────────────────────────────────
print()
print("=" * 70)
print("PART G: φ-AUDIT — are transformation parameters related to φ?")
print(f"        φ={PHI:.4f}  1/φ={1/PHI:.4f}  arccos(1/φ)={np.degrees(np.arccos(1/PHI)):.2f}°")
print(f"        arccos(1/φ²)={np.degrees(np.arccos(1/PHI**2)):.2f}°")
print("=" * 70)
print()
print("  Per-paradigm characteristic angle θ_char = mean(arccos(cos(a_n, b_n)))")
print(f"  {'paradigm':<16}  {'θ_char':>8}  {'||b||/||a||':>12}  φ-match?")

phi_results = {}
for pname, pairs in PARADIGMS.items():
    p = ok_pairs(pairs)
    if not p: continue
    angles = [angle_deg(get_emb(a), get_emb(b)) for a,b in p]
    ratios = [np.linalg.norm(get_emb(b)) / np.linalg.norm(get_emb(a)) for a,b in p]
    th = float(np.mean(angles)); nr = float(np.mean(ratios))
    # Check φ-matches
    phi_cands = {
        "arccos(1/φ)=51.83°": np.degrees(np.arccos(1/PHI)),
        "arccos(1/φ²)=67.97°": np.degrees(np.arccos(1/PHI**2)),
        "arccos(φ-1)=51.83°": np.degrees(np.arccos(PHI-1)),
        "90/φ=55.6°": 90/PHI,
        "180/φ²=68.6°": 180/PHI**2,
        "180/φ=111.2°": 180/PHI,
    }
    match = ""
    for label, val in phi_cands.items():
        if abs(th - val) < 2.0:
            match = f"≈ {label}"
            break
    print(f"  {pname:<16}  {th:>8.2f}°  {nr:>12.4f}  {match}")
    phi_results[pname] = {"theta_char": th, "norm_ratio": nr}

output = {
    "norm_results": norm_results,
    "map_results": map_results,
    "comp_results": comp_results,
    "angle_rows": angle_rows,
    "geo_rows": geo_rows,
    "phi_results": phi_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Geometric Audit complete.")
