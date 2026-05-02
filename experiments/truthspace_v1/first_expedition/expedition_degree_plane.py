#!/usr/bin/env python3
"""
Degree Plane Comparison — Do adj_degree triples share a common 2D plane?

Fixes Part B of degree_arc.py: use SVD instead of Gram-Schmidt to build
orthonormal bases for each triple's 2D plane. SVD guarantees orthonormality
numerically; Gram-Schmidt accumulates error at H=1536.

Questions:
  A. PLANE SHARING: do all 24 adj_degree triples lie in the SAME 2D plane?
     Measure: principal angles between each pair of triple planes.
     If mean max_cos ≈ 1: single shared plane.
     If mean max_cos << 1: each word has its own plane.

  B. BEST-FIT SHARED PLANE: fit a single 2D subspace to ALL 24 triples
     simultaneously via SVD of the stacked difference-vector matrix.
     How well does this shared plane fit each individual triple?

  C. WITHIN-PLANE ARC GEOMETRY REVISITED:
     Using SVD basis (numerically clean), recompute R, Ω, t_comp.
     Confirm the φ-matches from degree_arc.py are real, not artefacts.

  D. WHAT DOES THE SHARED PLANE ENCODE?
     Project the full vocabulary onto the 2 shared degree plane axes.
     What tokens have extreme projections?
     Does axis-1 = degree direction? Does axis-2 = something else?

  E. ARC PARAMETERS vs WORD PROPERTIES:
     Is R correlated with anything about the adjective?
     (word length, frequency rank, sentiment polarity, etc.)
     Simple test: is R larger for "big" (common) vs "safe" (less common)?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "degree_plane.json")
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

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(normed(a), normed(b))), -1, 1))))

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

def svd_plane_basis(P, C, S):
    """
    Build orthonormal 2D basis for the plane containing P, C, S
    using SVD (numerically stable). Returns (e1, e2, coords_2d).
    e1, e2 are guaranteed orthonormal in R^H.
    coords_2d: dict with 2D coordinates of each point (relative to P).
    """
    v1 = C - P
    v2 = S - P
    D = np.stack([v1, v2], axis=1)  # H × 2
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    e1 = U[:, 0]; e2 = U[:, 1]
    # 2D coordinates of P, C, S
    p2 = np.array([0.0, 0.0])
    c2 = np.array([float(np.dot(v1, e1)), float(np.dot(v1, e2))])
    s2 = np.array([float(np.dot(v2, e1)), float(np.dot(v2, e2))])
    return e1, e2, p2, c2, s2

def circumscribed_circle_2d(p2, c2, s2):
    ax, ay = p2; bx, by = c2; cx_, cy_ = s2
    D = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(D) < 1e-12: return None
    ux = ((ax**2+ay**2)*(by-cy_) + (bx**2+by**2)*(cy_-ay) +
          (cx_**2+cy_**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx_-bx) + (bx**2+by**2)*(ax-cx_) +
          (cx_**2+cy_**2)*(bx-ax)) / D
    R = float(np.sqrt((ax-ux)**2+(ay-uy)**2))
    return float(ux), float(uy), R

def arc_angle_2d(center, pt):
    return float(np.arctan2(pt[1]-center[1], pt[0]-center[0]))

def signed_arc(a1, a2):
    d = a2 - a1
    while d > np.pi:  d -= 2*np.pi
    while d < -np.pi: d += 2*np.pi
    return d

# ── Collect SVD planes and arc parameters ────────────────────────────
triples_data = []
planes_e1 = []; planes_e2 = []

print("Computing SVD plane bases and arc parameters ...")
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C is None or S is None: continue
    e1, e2, p2, c2, s2 = svd_plane_basis(P, C, S)

    # Verify orthonormality
    assert abs(float(np.dot(e1, e1)) - 1) < 1e-10, "e1 not unit"
    assert abs(float(np.dot(e2, e2)) - 1) < 1e-10, "e2 not unit"
    assert abs(float(np.dot(e1, e2))) < 1e-10, "e1,e2 not orthogonal"

    circ = circumscribed_circle_2d(p2, c2, s2)
    if circ is None: continue
    cx, cy, R = circ; cen = np.array([cx, cy])

    ang_p = arc_angle_2d(cen, p2)
    ang_c = arc_angle_2d(cen, c2)
    ang_s = arc_angle_2d(cen, s2)
    arc_pc = signed_arc(ang_p, ang_c)
    arc_cs = signed_arc(ang_c, ang_s)
    arc_total = arc_pc + arc_cs
    Omega   = float(np.degrees(abs(arc_total)))
    Omega_pc = float(np.degrees(abs(arc_pc)))
    Omega_cs = float(np.degrees(abs(arc_cs)))
    t_comp  = abs(arc_pc) / (abs(arc_total) + 1e-8)

    # Origin in 2D coords (embedding-space zero projected into plane)
    # Zero vector = P + (0 - P) => in 2D: (-P·e1, -P·e2)
    orig_2d = np.array([-float(np.dot(P, e1)), -float(np.dot(P, e2))])
    d_origin = float(np.linalg.norm(orig_2d - cen))

    planes_e1.append(e1); planes_e2.append(e2)
    triples_data.append({
        "word": pos_w, "R": R, "Omega": Omega,
        "Omega_pc": Omega_pc, "Omega_cs": Omega_cs,
        "t_comp": t_comp, "d_origin": d_origin,
        "orientation": "CCW" if arc_total > 0 else "CW",
    })

print(f"  Processed {len(triples_data)} triples\n")

# Print summary
Rs = [d["R"] for d in triples_data]
Omegas = [d["Omega"] for d in triples_data]
Omega_pcs = [d["Omega_pc"] for d in triples_data]
Omega_css = [d["Omega_cs"] for d in triples_data]
ts = [d["t_comp"] for d in triples_data]
ds = [d["d_origin"] for d in triples_data]
ccw = sum(1 for d in triples_data if d["orientation"] == "CCW")

print("PART C: Arc parameters (SVD basis, numerically clean):")
print(f"  R:        mean={np.mean(Rs):.4f}  std={np.std(Rs):.4f}")
print(f"  Ω_total:  mean={np.mean(Omegas):.3f}°  std={np.std(Omegas):.3f}°")
print(f"  Ω_pc:     mean={np.mean(Omega_pcs):.3f}°  std={np.std(Omega_pcs):.3f}°")
print(f"  Ω_cs:     mean={np.mean(Omega_css):.3f}°  std={np.std(Omega_css):.3f}°")
print(f"  t_comp:   mean={np.mean(ts):.4f}  std={np.std(ts):.4f}")
print(f"  d_origin: mean={np.mean(ds):.4f}  std={np.std(ds):.4f}")
print(f"  CCW: {ccw}/{len(triples_data)}")
print()
print(f"  φ = {PHI:.4f}")
print(f"  π/φ = {180/PHI:.3f}°   (vs Ω_pc {np.mean(Omega_pcs):.3f}°,  diff={abs(np.mean(Omega_pcs)-180/PHI):.3f}°)")
print(f"  2π/φ = {360/PHI:.3f}°  (vs Ω_total {np.mean(Omegas):.3f}°, diff={abs(np.mean(Omegas)-360/PHI):.3f}°)")
print(f"  |t - 0.5| = {abs(np.mean(ts)-0.5):.4f}")
print(f"  |t - 1/φ| = {abs(np.mean(ts)-1/PHI):.4f}")

# ── Part A: Plane sharing — principal angles via SVD ──────────────────
print()
print("=" * 70)
print("PART A: PLANE SHARING — principal angles between each pair of 2D planes")
print("        Using SVD-guaranteed orthonormal bases (no Gram-Schmidt errors)")
print("        Principal angles: cos(θ_i) = singular values of A^T @ B")
print("        If planes are IDENTICAL: all cos = 1")
print("        If planes are RANDOM:    cos ≈ 0")
print("=" * 70)
print()

n = len(triples_data)
words = [d["word"] for d in triples_data]

# Build 2D basis matrices: each is (H, 2) with orthonormal columns
bases = [np.stack([planes_e1[i], planes_e2[i]], axis=1) for i in range(n)]

# Pairwise principal angles
cos1_mat = np.zeros((n, n))  # max cosine (first principal angle)
cos2_mat = np.zeros((n, n))  # min cosine (second principal angle)

for i in range(n):
    for j in range(n):
        # Principal angles: singular values of B_i^T @ B_j (both (H,2))
        G = bases[i].T @ bases[j]  # 2 × 2
        sv = np.linalg.svd(G, compute_uv=False)
        sv = np.clip(sv, 0, 1)  # clip to [0,1] (SVD of orthonormal matrices)
        cos1_mat[i, j] = sv[0]
        cos2_mat[i, j] = sv[1]

# Off-diagonal
off1 = cos1_mat[np.triu_indices(n, k=1)]
off2 = cos2_mat[np.triu_indices(n, k=1)]

print(f"  First principal angle cosines (cos θ₁):")
print(f"    mean={np.mean(off1):.4f}  std={np.std(off1):.4f}  "
      f"min={np.min(off1):.4f}  max={np.max(off1):.4f}")
print(f"  Second principal angle cosines (cos θ₂):")
print(f"    mean={np.mean(off2):.4f}  std={np.std(off2):.4f}  "
      f"min={np.min(off2):.4f}  max={np.max(off2):.4f}")
print()

# θ in degrees
theta1_mean = float(np.degrees(np.arccos(np.mean(off1))))
theta2_mean = float(np.degrees(np.arccos(np.mean(off2))))
print(f"  Mean principal angles: θ₁={theta1_mean:.2f}°, θ₂={theta2_mean:.2f}°")
print()
print(f"  Interpretation:")
print(f"    θ₁ near 0°: planes share one common direction (line alignment)")
print(f"    θ₁,θ₂ near 0°: planes are nearly identical")
print(f"    θ₁,θ₂ near 90°: planes are completely orthogonal")

# Most and least aligned pairs
pairs = [(float(cos1_mat[i,j]), words[i], words[j])
         for i in range(n) for j in range(i+1, n)]
pairs.sort(reverse=True)
print()
print("  Top 5 most aligned plane pairs (cos θ₁):")
for cv, wi, wj in pairs[:5]:
    print(f"    {wi:<10} ↔ {wj:<10}  cos(θ₁)={cv:.4f}")
print("  Bottom 5 least aligned:")
for cv, wi, wj in pairs[-5:][::-1]:
    print(f"    {wi:<10} ↔ {wj:<10}  cos(θ₁)={cv:.4f}")

# ── Part B: Best-fit shared plane ─────────────────────────────────────
print()
print("=" * 70)
print("PART B: BEST-FIT SHARED DEGREE PLANE")
print("        SVD of all difference vectors [C-P, S-P] stacked")
print("        The top 2 left singular vectors span the shared degree plane")
print("=" * 70)
print()

all_diffs = []
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C is None or S is None: continue
    all_diffs.append(C - P)
    all_diffs.append(S - P)

D_all = np.array(all_diffs).T  # H × (2*n_triples)
U_shared, S_shared, _ = np.linalg.svd(D_all, full_matrices=False)
e1_shared = U_shared[:, 0]
e2_shared = U_shared[:, 1]

total_var = float((S_shared**2).sum())
print(f"  Singular value spectrum (top 10):")
print(f"  {'k':>3}  {'S[k]':>8}  {'var%':>8}  {'cumvar%':>10}")
cumvar = 0.0
for i in range(min(10, len(S_shared))):
    v = (S_shared[i]**2) / total_var * 100
    cumvar += v
    print(f"  {i:>3}  {S_shared[i]:>8.3f}  {v:>8.2f}%  {cumvar:>10.2f}%")

print()
print(f"  How well does the shared plane fit each individual triple?")
print(f"  Fit quality = fraction of variance explained by shared e1,e2")
print(f"  {'word':<10}  {'fit%':>6}  {'resid%':>8}  quality")
fit_qualities = []
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C is None or S is None: continue
    v1 = C - P; v2 = S - P
    # Fraction of v1, v2 explained by shared plane
    proj1 = float(np.dot(v1, e1_shared))**2 + float(np.dot(v1, e2_shared))**2
    proj2 = float(np.dot(v2, e1_shared))**2 + float(np.dot(v2, e2_shared))**2
    fit1 = proj1 / (np.linalg.norm(v1)**2 + 1e-8)
    fit2 = proj2 / (np.linalg.norm(v2)**2 + 1e-8)
    fit_mean = (fit1 + fit2) / 2
    fit_pct = fit_mean * 100
    quality = "GOOD" if fit_pct > 80 else "MEDIUM" if fit_pct > 60 else "POOR"
    print(f"  {pos_w:<10}  {fit_pct:>6.1f}%  {100-fit_pct:>8.1f}%  {quality}")
    fit_qualities.append(fit_pct)

print()
print(f"  mean fit = {np.mean(fit_qualities):.1f}%  std = {np.std(fit_qualities):.1f}%")

# ── Part D: What does the shared degree plane encode? ─────────────────
print()
print("=" * 70)
print("PART D: SHARED DEGREE PLANE — vocabulary projections")
print("        Project all pool tokens onto (e1_shared, e2_shared)")
print("        What words have extreme projections?")
print("=" * 70)
print()

# Build pool
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid])
E_pool = np.array(pool_embs, dtype=np.float64)
print(f"  Pool size: {len(pool_words)}\n")

proj_e1 = E_pool @ e1_shared
proj_e2 = E_pool @ e2_shared

for axis_name, proj, axis_vec in [("e1 (shared degree axis 1)", proj_e1, e1_shared),
                                   ("e2 (shared degree axis 2)", proj_e2, e2_shared)]:
    order_pos = np.argsort(-proj)
    order_neg = np.argsort(proj)
    top_pos = [(pool_words[i], float(proj[i])) for i in order_pos[:20]]
    top_neg = [(pool_words[i], float(proj[i])) for i in order_neg[:20]]
    print(f"  {axis_name}:")
    print(f"    + end: " + "  ".join(f"{w}({v:+.3f})" for w,v in top_pos[:10]))
    print(f"          " + "  ".join(f"{w}({v:+.3f})" for w,v in top_pos[10:]))
    print(f"    - end: " + "  ".join(f"{w}({v:+.3f})" for w,v in top_neg[:10]))
    print(f"          " + "  ".join(f"{w}({v:+.3f})" for w,v in top_neg[10:]))
    # Where do specific degree forms land?
    test_words = ["big","bigger","biggest","fast","faster","fastest",
                  "high","higher","highest","old","older","oldest"]
    print(f"    Degree words: " + "  ".join(
        f"{w}({float(get_emb(w) @ axis_vec):+.3f})" for w in test_words if get_emb(w) is not None
    ))
    print()

output = {
    "arc_params": triples_data,
    "plane_principal_angles": {
        "cos1_mean": float(np.mean(off1)), "cos1_std": float(np.std(off1)),
        "cos2_mean": float(np.mean(off2)), "cos2_std": float(np.std(off2)),
        "theta1_mean_deg": theta1_mean, "theta2_mean_deg": theta2_mean,
    },
    "shared_plane_fit": {
        "mean_fit_pct": float(np.mean(fit_qualities)),
        "sv_top5": [float(s) for s in S_shared[:5]],
    }
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"Saved: {OUTPUT_FILE}")
print("Degree plane analysis complete.")
