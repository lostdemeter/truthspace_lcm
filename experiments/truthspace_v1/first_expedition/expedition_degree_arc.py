#!/usr/bin/env python3
"""
Degree Arc Geometry — What Curve Is pos→comp→sup?

The geometric audit showed:
  - pos→comp→sup subtends 112° total arc vs 61° direct arc on unit sphere
  - comp is 49° off the great circle connecting pos and sup
  - Each single step ≈ 90/φ ≈ 55.6° on the sphere

Three points in R^1536 always lie in a unique 2D affine plane.
In that plane, three points define a unique circumscribed circle.

Questions:
  A. CIRCUMSCRIBED CIRCLE: For each triple (pos, comp, sup), find the
     circumscribed circle in the 2D plane they span.
     - radius R
     - arc angle Ω (total arc pos→comp→sup on this circle)
     - position of comp on the arc (fraction t ∈ [0,1])
     Is Ω consistent across words? Related to φ?

  B. CROSS-WORD PLANE COMPARISON:
     Do different triples (big, fast, long...) share the same 2D plane?
     Measure: cosine similarity between the normal vectors of each triple's
     2D plane. If all planes are parallel, there's a UNIVERSAL DEGREE PLANE.

  C. ORIGIN RELATIONSHIP:
     Where is the origin relative to the circumscribed circle?
     - Is the origin inside or outside the circle?
     - What is the distance from origin to the circle center?
     This tells us whether the circle is a "great circle" (origin on the
     circle) or a "small circle" (origin off the circle) on the sphere.

  D. ARC ANGLE φ-TEST:
     The total arc angle Ω: is it related to φ?
     Candidate values: 2π/φ = 222.5°, 2π/φ² = 137.5° (golden angle),
     π/φ = 111.2°, π·(2-φ) = 59.0°, etc.

  E. CONSISTENCY OF COMP POSITION ON ARC:
     The parameter t (where comp sits between pos and sup on the arc):
     Is t consistent? Is t = 1/φ or 1/φ²?
     t = 0.5 means comp is the arc midpoint.
     t = 1/φ ≈ 0.618 means comp is the golden section of the arc.

  F. ARC REGULARITY:
     Are the two sub-arcs pos→comp and comp→sup equal (equidistant)?
     From the audit: θ_sphere(p→c) ≈ 54.7°, θ_sphere(c→s) ≈ 57.2°
     But in the 2D plane, the arc angles may be different.
     If the circle is centered differently, equal sphere angles don't
     imply equal arc angles in the circle.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "degree_arc.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2  # 1.6180...

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
    c = float(np.clip(np.dot(normed(a), normed(b)), -1, 1))
    return float(np.degrees(np.arccos(c)))

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

def circumscribed_circle_2d(P, C, S):
    """
    Given three 2D points P, C, S, find the circumscribed circle.
    Returns (cx, cy, R) — center and radius.
    Uses the perpendicular bisector method.
    """
    ax, ay = P
    bx, by = C
    cx_, cy_ = S
    D = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(D) < 1e-12:
        return None  # collinear
    ux = ((ax**2 + ay**2) * (by - cy_) + (bx**2 + by**2) * (cy_ - ay) +
          (cx_**2 + cy_**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx_ - bx) + (bx**2 + by**2) * (ax - cx_) +
          (cx_**2 + cy_**2) * (bx - ax)) / D
    R = float(np.sqrt((ax - ux)**2 + (ay - uy)**2))
    return float(ux), float(uy), R

def arc_angle(center_2d, point_2d):
    """Angle (radians) from center to point in 2D."""
    dx = point_2d[0] - center_2d[0]
    dy = point_2d[1] - center_2d[1]
    return float(np.arctan2(dy, dx))

def signed_arc(a1, a2):
    """Signed arc from angle a1 to a2 (shortest or canonical path)."""
    diff = a2 - a1
    while diff > np.pi:  diff -= 2 * np.pi
    while diff < -np.pi: diff += 2 * np.pi
    return diff

# ── Project triple into its 2D plane ─────────────────────────────────
def project_to_2d_plane(P, C, S):
    """
    Project three R^H points into their 2D affine plane.
    Returns (p2d, c2d, s2d, normal1, normal2) where normal1, normal2
    are the two orthonormal basis vectors of the plane.
    Also returns the plane's normal (in H-dim space) = any vector ⊥ to the plane.
    """
    # Translate to P as origin
    v1 = C - P
    v2 = S - P
    # Gram-Schmidt orthonormalization
    e1 = normed(v1)
    e2_raw = v2 - np.dot(v2, e1) * e1
    if np.linalg.norm(e2_raw) < 1e-10:
        return None  # degenerate
    e2 = normed(e2_raw)
    # 2D coordinates in this basis
    p2 = np.array([0.0, 0.0])
    c2 = np.array([np.dot(v1, e1), np.dot(v1, e2)])
    s2 = np.array([np.dot(v2, e1), np.dot(v2, e2)])
    return p2, c2, s2, e1, e2

results = []
plane_normals = []  # to compare across words

print("=" * 70)
print("CIRCUMSCRIBED CIRCLE ANALYSIS for adj_degree triples")
print("=" * 70)
print()
print(f"  {'word':<8}  {'R':>7}  {'Ω_total°':>9}  {'Ω_pc°':>8}  {'Ω_cs°':>8}  "
      f"{'t_comp':>7}  {'t_phi?':>8}  {'d_origin':>9}  φ_match")

for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w)
    C = get_emb(comp_w)
    S = get_emb(sup_w)
    if P is None or C is None or S is None:
        continue

    proj = project_to_2d_plane(P, C, S)
    if proj is None:
        print(f"  {pos_w:<8}  (degenerate)")
        continue

    p2, c2, s2, e1, e2 = proj

    circ = circumscribed_circle_2d(p2, c2, s2)
    if circ is None:
        print(f"  {pos_w:<8}  (collinear)")
        continue

    cx, cy, R = circ
    center_2d = np.array([cx, cy])

    # Angles of each point on the circle
    ang_p = arc_angle(center_2d, p2)
    ang_c = arc_angle(center_2d, c2)
    ang_s = arc_angle(center_2d, s2)

    # Arc from P to C to S (going from P through C to S)
    # We need the arc that passes through C
    # Determine orientation: P→C→S should be a consistent direction
    arc_pc = signed_arc(ang_p, ang_c)
    arc_cs = signed_arc(ang_c, ang_s)
    arc_ps_total = arc_pc + arc_cs

    Omega_total = float(np.degrees(abs(arc_ps_total)))
    Omega_pc    = float(np.degrees(abs(arc_pc)))
    Omega_cs    = float(np.degrees(abs(arc_cs)))

    # t_comp = fraction of total arc at comp
    t_comp = abs(arc_pc) / (abs(arc_ps_total) + 1e-8)

    # Distance from origin to circle center
    # Origin in 2D plane: project origin (O = 0 vector) into the plane
    # P is the 2D origin in our basis, and P is NOT the H-dim origin
    # We need to find where the H-dim origin lands in the 2D plane
    # H-dim origin in 2D coords: ((-P)·e1, (-P)·e2)
    orig_in_plane = np.array([-np.dot(P, e1), -np.dot(P, e2)])
    d_origin_to_circle = float(np.linalg.norm(orig_in_plane - center_2d))

    # φ-match for t_comp
    phi_match = ""
    if abs(t_comp - 1/PHI) < 0.05:     phi_match = "1/φ"
    elif abs(t_comp - 1/PHI**2) < 0.05: phi_match = "1/φ²"
    elif abs(t_comp - 0.5) < 0.05:      phi_match = "1/2"
    elif abs(t_comp - 1 - 1/PHI) < 0.05: phi_match = "φ-1"

    # Plane normal (in H-dim space): e1 × e2 is not well-defined in H-dim
    # Instead use the normal to the 2D subspace as any vector ⊥ to both e1, e2
    # For comparison across words, use the 2D plane basis as a 2-column matrix
    plane_normals.append((pos_w, e1, e2))

    # φ-match for Omega_total
    omega_match = ""
    omega_cands = {
        "2π/φ²=137.5°": 360/PHI**2,
        "π/φ=111.2°": 180/PHI,
        "π/φ²=68.5°": 180/PHI**2,
        "180°": 180.0,
        "2π/φ=222.5°": 360/PHI,
        "360/3=120°": 120.0,
    }
    for label, val in omega_cands.items():
        if abs(Omega_total - val) < 5.0:
            omega_match = label; break

    print(f"  {pos_w:<8}  {R:>7.2f}  {Omega_total:>9.2f}  {Omega_pc:>8.2f}  "
          f"{Omega_cs:>8.2f}  {t_comp:>7.4f}  {phi_match:>8}  "
          f"{d_origin_to_circle:>9.2f}  {omega_match}")

    results.append({
        "word": pos_w, "R": R, "Omega_total": Omega_total,
        "Omega_pc": Omega_pc, "Omega_cs": Omega_cs,
        "t_comp": t_comp, "d_origin": d_origin_to_circle,
        "arc_orientation": "CCW" if arc_ps_total > 0 else "CW",
    })

# ── Summary statistics ────────────────────────────────────────────────
if results:
    Rs        = [r["R"] for r in results]
    Omegas    = [r["Omega_total"] for r in results]
    Omegas_pc = [r["Omega_pc"] for r in results]
    Omegas_cs = [r["Omega_cs"] for r in results]
    ts        = [r["t_comp"] for r in results]
    ds        = [r["d_origin"] for r in results]

    print()
    print("  SUMMARY:")
    print(f"    R (circle radius):     mean={np.mean(Rs):.3f}  std={np.std(Rs):.3f}  "
          f"min={min(Rs):.3f}  max={max(Rs):.3f}")
    print(f"    Ω_total (arc angle):   mean={np.mean(Omegas):.3f}°  "
          f"std={np.std(Omegas):.3f}°")
    print(f"    Ω_pc:                  mean={np.mean(Omegas_pc):.3f}°  "
          f"std={np.std(Omegas_pc):.3f}°")
    print(f"    Ω_cs:                  mean={np.mean(Omegas_cs):.3f}°  "
          f"std={np.std(Omegas_cs):.3f}°")
    print(f"    t_comp (arc fraction): mean={np.mean(ts):.4f}  std={np.std(ts):.4f}")
    print(f"    d_origin (R^H):        mean={np.mean(ds):.3f}  std={np.std(ds):.3f}")
    print()
    print(f"    φ = {PHI:.4f},  1/φ = {1/PHI:.4f},  1/φ² = {1/PHI**2:.4f}")
    print(f"    2π/φ² = 137.51°,  π/φ = 111.25°,  2π/φ = 222.49°")
    print()
    print(f"    t_comp vs φ-fractions:")
    print(f"      |mean(t) - 1/φ|  = {abs(np.mean(ts) - 1/PHI):.4f}")
    print(f"      |mean(t) - 1/φ²| = {abs(np.mean(ts) - 1/PHI**2):.4f}")
    print(f"      |mean(t) - 0.5|  = {abs(np.mean(ts) - 0.5):.4f}")
    print()
    print(f"    Ω_total vs φ-angles:")
    print(f"      |mean(Ω) - 137.51°| = {abs(np.mean(Omegas) - 360/PHI**2):.3f}°")
    print(f"      |mean(Ω) - 111.25°| = {abs(np.mean(Omegas) - 180/PHI):.3f}°")
    print(f"      |mean(Ω) - 180°|    = {abs(np.mean(Omegas) - 180):.3f}°")
    print()
    ccw = sum(1 for r in results if r["arc_orientation"] == "CCW")
    cw  = len(results) - ccw
    print(f"    Arc orientation: {ccw} CCW, {cw} CW")

# ── Part B: Cross-word plane comparison ───────────────────────────────
print()
print("=" * 70)
print("PART B: CROSS-WORD PLANE COMPARISON")
print("  Is the 2D degree plane the same across all adjective triples?")
print("  cos(e1_i, e1_j) and cos(e2_i, e2_j) — do planes align?")
print("=" * 70)
print()

if len(plane_normals) >= 2:
    # For each pair of words, measure alignment of their 2D planes
    # Two planes align if they span the same 2D subspace
    # Measure via principal angles between 2D subspaces
    all_e1s = [e1 for _, e1, _ in plane_normals]
    all_e2s = [e2 for _, e2, _ in plane_normals]
    words   = [w  for w, _, _ in plane_normals]

    # Principal angles between plane i and plane j
    def plane_principal_angles(e1a, e2a, e1b, e2b):
        """Max cosine alignment between 2D planes."""
        Ba = np.array([e1a, e2a])  # 2 × H
        Bb = np.array([e1b, e2b])  # 2 × H
        G  = Ba @ Bb.T  # 2 × 2 gram matrix
        sv = np.linalg.svd(G, compute_uv=False)
        return sv  # cosines of principal angles

    # Compute all pairwise max cosines
    n = len(words)
    max_cos_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sv = plane_principal_angles(all_e1s[i], all_e2s[i],
                                        all_e1s[j], all_e2s[j])
            max_cos_mat[i, j] = sv[0]

    # Off-diagonal statistics
    off_diag = max_cos_mat[np.triu_indices(n, k=1)]
    print(f"  Max cosine between 2D plane pairs (principal angle 1):")
    print(f"    mean = {np.mean(off_diag):.4f}")
    print(f"    std  = {np.std(off_diag):.4f}")
    print(f"    min  = {np.min(off_diag):.4f}")
    print(f"    max  = {np.max(off_diag):.4f}")
    print()
    print(f"  If planes are IDENTICAL, max_cos = 1.0 for all pairs.")
    print(f"  If planes are RANDOM,     max_cos ≈ 0.0 for all pairs.")
    print()
    # Show strongest alignments
    top_pairs = sorted(
        [(max_cos_mat[i,j], words[i], words[j])
         for i in range(n) for j in range(i+1, n)],
        reverse=True
    )[:5]
    print(f"  Top 5 most aligned plane pairs:")
    for cos_val, wi, wj in top_pairs:
        print(f"    {wi:<10} ↔ {wj:<10}  max_cos={cos_val:.4f}")
    bot_pairs = top_pairs[-5:][::-1] if len(top_pairs) >= 5 else []
    if bot_pairs:
        print(f"  Bottom 5 least aligned plane pairs:")
        for cos_val, wi, wj in bot_pairs:
            print(f"    {wi:<10} ↔ {wj:<10}  max_cos={cos_val:.4f}")

# ── Part C: Origin relationship ────────────────────────────────────────
print()
print("=" * 70)
print("PART C: ORIGIN RELATIONSHIP")
print("  Where does the embedding-space origin sit relative to the circle?")
print("  d_origin ≈ R: origin IS on the circle (great circle)")
print("  d_origin >> R: origin is far outside (small circle, not through origin)")
print("  d_origin = 0: origin IS the circle center")
print("=" * 70)
print()
if results:
    print(f"  mean(d_origin) = {np.mean(ds):.3f}")
    print(f"  mean(R)        = {np.mean(Rs):.3f}")
    print(f"  d_origin / R   = {np.mean(ds)/np.mean(Rs):.4f}")
    print()
    print(f"  d_origin >> R: origin is far from the circumscribed circle.")
    print(f"  This means the arc is NOT a great circle segment.")
    print(f"  The arc is a chord of a much larger circle whose center")
    print(f"  is far from the embedding-space origin.")

output = {"results": results,
          "summary": {
              "R_mean": float(np.mean(Rs)) if results else None,
              "Omega_mean": float(np.mean(Omegas)) if results else None,
              "t_comp_mean": float(np.mean(ts)) if results else None,
          } if results else {}}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Degree arc analysis complete.")
