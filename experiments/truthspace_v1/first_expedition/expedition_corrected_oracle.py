#!/usr/bin/env python3
"""
Corrected Oracle Rotation — Can arc rotation achieve 100% with correct center?

The previous oracle failed because:
  1. Wrong center: rotated around embedding-space origin instead of
     the circumscribed circle center C in the private plane.
  2. Wrong direction: used +π/φ but some words need -π/φ.

This experiment uses the TRUE arc parameters:
  - Circle center C (from the circumscribed circle of pos/comp/sup)
  - Correct rotation direction (sign of arc_pc from the actual arc)
  - Correct arc angle (the actual arc_pc, not the assumed π/φ)

Expected result: if the arc structure is geometrically self-consistent,
rotating emb(pos) by arc_pc around C in the private plane should give
exactly emb(comp). This is guaranteed by construction.

But: does the nearest-neighbour retrieval from the rotated vector find
the correct comp token? This depends on whether the rotation lands close
enough to emb(comp) to beat all competitors.

Tests:
  A. EXACT ARC ROTATION: rotate by exact arc_pc around true C.
     Should give exactly emb(comp) → cos = 1.0, accuracy = 100%
     if the rotation implementation is correct.

  B. CANONICAL ARC ROTATION: rotate by π/φ around true C,
     using correct sign. How close is π/φ to the actual arc_pc?
     Compare accuracy vs exact arc rotation.

  C. SHARED PLANE + TRUE CENTER: rotate in shared plane around
     the shared plane's approximate center. Does knowing the center
     improve over shared plane rotation?

  D. CHORD LENGTH ANALYSIS: for each paradigm, compute the chord
     length 2R·sin(Ω/2) and compare to the mean(||b-a||₂) over
     the training pairs. These should agree if the arc model is right.

  E. PARADIGM CHORD TABLE: characteristic chord lengths across
     all paradigms — this is the semantic distance measure.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "corrected_oracle.json")
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
    "gender":      [("king","queen"),("man","woman"),("boy","girl"),
                    ("prince","princess"),("actor","actress"),("hero","heroine"),
                    ("monk","nun"),("duke","duchess"),("lord","lady"),
                    ("wizard","witch"),("nephew","niece"),("lion","lioness"),
                    ("father","mother"),("son","daughter"),("brother","sister"),
                    ("husband","wife"),("grandfather","grandmother")],
    "plural":      [("cat","cats"),("dog","dogs"),("house","houses"),
                    ("tree","trees"),("book","books"),("car","cars"),
                    ("bird","birds"),("ship","ships"),("hand","hands"),
                    ("door","doors"),("king","kings"),("boy","boys"),
                    ("word","words"),("stone","stones"),("cloud","clouds"),
                    ("road","roads"),("horse","horses"),("town","towns")],
    "past_tense":  [("walk","walked"),("talk","talked"),("call","called"),
                    ("pull","pulled"),("fill","filled"),("turn","turned"),
                    ("look","looked"),("move","moved"),("push","pushed"),
                    ("help","helped"),("play","played"),("stay","stayed"),
                    ("lock","locked"),("jump","jumped"),("land","landed"),
                    ("ask","asked"),("work","worked"),("open","opened")],
    "capital":     [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
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

def svd_plane(P, Q, R_pt):
    v1 = Q - P; v2 = R_pt - P
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

def arc_angle_2d(cx, cy, x, y): return float(np.arctan2(y-cy, x-cx))

def signed_arc(a1, a2):
    d = a2-a1
    while d >  np.pi: d -= 2*np.pi
    while d < -np.pi: d += 2*np.pi
    return d

def rotate_around_center(v, e1, e2, cx_2d, cy_2d, angle_rad):
    """
    Rotate the in-plane component of v by angle_rad around (cx_2d, cy_2d)
    in the 2D plane spanned by e1, e2.
    Out-of-plane component is unchanged.
    """
    # 2D coordinates of v (relative to plane origin = P)
    a = float(np.dot(v, e1))
    b = float(np.dot(v, e2))
    # Translate to rotation center
    a_c = a - cx_2d; b_c = b - cy_2d
    # Rotate
    cos_a = np.cos(angle_rad); sin_a = np.sin(angle_rad)
    a_r = a_c * cos_a - b_c * sin_a + cx_2d
    b_r = a_c * sin_a + b_c * cos_a + cy_2d
    # Reconstruct H-dim vector
    v_perp = v - a * e1 - b * e2
    return v_perp + a_r * e1 + b_r * e2

# ── Part A: Exact arc rotation ────────────────────────────────────────
print("=" * 70)
print("PART A: EXACT ARC ROTATION (rotate by actual arc_pc around true C)")
print("        Expected: reconstruct comp EXACTLY (cos ≈ 1.0, accuracy 100%)")
print("=" * 70)
print()
print(f"  {'word':<8}  {'arc_pc°':>9}  {'cos_exact':>10}  {'pred_exact':>12}  "
      f"{'correct':>8}  {'cos_pi/phi':>11}  {'pred_pi/phi':>12}  correct")

exact_acc = 0; canonical_acc = 0; n = 0
exact_results = []

for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C_emb = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C_emb is None or S is None: continue

    e1, e2 = svd_plane(P, C_emb, S)
    if e1 is None: continue

    # 2D coordinates (P as origin)
    v1 = C_emb - P; v2 = S - P
    p2 = np.array([0.0, 0.0])
    c2 = np.array([float(np.dot(v1, e1)), float(np.dot(v1, e2))])
    s2 = np.array([float(np.dot(v2, e1)), float(np.dot(v2, e2))])

    circ = circumscribed_2d(p2, c2, s2)
    if circ is None: continue
    cx, cy, R = circ

    # Arc angles of each point around circle center
    ang_p = arc_angle_2d(cx, cy, p2[0], p2[1])
    ang_c = arc_angle_2d(cx, cy, c2[0], c2[1])
    arc_pc = signed_arc(ang_p, ang_c)  # exact arc angle with sign
    arc_pc_deg = float(np.degrees(arc_pc))

    # Rotate P by exact arc_pc around true center (cx, cy)
    rotated_exact = rotate_around_center(P, e1, e2, cx, cy, arc_pc)
    cos_exact = float(np.dot(normed(rotated_exact), normed(C_emb)))
    pred_exact = top1(rotated_exact, exclude={pos_w})

    # Rotate P by canonical π/φ (correct sign from arc_pc) around true center
    canonical_angle = np.sign(arc_pc) * np.radians(180/PHI)
    rotated_canon = rotate_around_center(P, e1, e2, cx, cy, canonical_angle)
    cos_canon = float(np.dot(normed(rotated_canon), normed(C_emb)))
    pred_canon = top1(rotated_canon, exclude={pos_w})

    ok_exact  = pred_exact  == comp_w
    ok_canon  = pred_canon  == comp_w
    if ok_exact:   exact_acc     += 1
    if ok_canon:   canonical_acc += 1
    n += 1

    print(f"  {pos_w:<8}  {arc_pc_deg:>9.2f}  {cos_exact:>10.4f}  {pred_exact:<12}  "
          f"{'Y' if ok_exact else 'N':>8}  {cos_canon:>11.4f}  {pred_canon:<12}  "
          f"{'Y' if ok_canon else 'N'}")
    exact_results.append({"word": pos_w, "arc_pc": arc_pc_deg,
                          "cos_exact": cos_exact, "cos_canon": cos_canon,
                          "exact_correct": ok_exact, "canon_correct": ok_canon})

print()
print(f"  EXACT arc rotation:       {exact_acc:>2}/{n} = {exact_acc/n:.3f}")
print(f"  CANONICAL π/φ rotation:   {canonical_acc:>2}/{n} = {canonical_acc/n:.3f}")
print(f"  (mean_dir baseline was 22/23 = 0.957)")

# ── Part B: Arc angle distribution ───────────────────────────────────
print()
print("=" * 70)
print("PART B: ACTUAL ARC ANGLE DISTRIBUTION")
print("        How close is each word's actual arc_pc to π/φ = 111.25°?")
print("=" * 70)
print()
if exact_results:
    arcs = [r["arc_pc"] for r in exact_results]
    arcs_abs = [abs(a) for a in arcs]
    print(f"  Actual arc_pc values (signed): {[f'{a:.1f}' for a in arcs]}")
    print()
    print(f"  |arc_pc|: mean={np.mean(arcs_abs):.2f}°  std={np.std(arcs_abs):.2f}°")
    print(f"  π/φ = {180/PHI:.2f}°")
    print(f"  |mean(|arc_pc|) - π/φ| = {abs(np.mean(arcs_abs) - 180/PHI):.2f}°")
    print()
    pos_arcs = sum(1 for a in arcs if a > 0)
    neg_arcs = sum(1 for a in arcs if a < 0)
    print(f"  Sign distribution: {pos_arcs} positive (CCW), {neg_arcs} negative (CW)")
    print(f"  -> Rotation direction is NOT globally consistent")

# ── Part C: Shared plane with approximate center ─────────────────────
print()
print("=" * 70)
print("PART C: SHARED PLANE + APPROXIMATE CENTER")
print("        The shared plane center approximates all individual centers.")
print("        Does knowing the approximate center help?")
print("=" * 70)
print()

# Compute shared degree plane and its approximate center
all_diffs = []
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C_emb = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C_emb is None or S is None: continue
    all_diffs.append(C_emb - P); all_diffs.append(S - P)
D_all = np.array(all_diffs).T
U_sh, _, _ = np.linalg.svd(D_all, full_matrices=False)
e1_sh, e2_sh = U_sh[:,0], U_sh[:,1]

# Mean center in the shared plane
centers_2d = []
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C_emb = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C_emb is None or S is None: continue
    e1, e2 = svd_plane(P, C_emb, S)
    if e1 is None: continue
    v1 = C_emb - P; v2 = S - P
    p2 = np.array([0.0, 0.0])
    c2 = np.array([float(np.dot(v1, e1)), float(np.dot(v1, e2))])
    s2 = np.array([float(np.dot(v2, e1)), float(np.dot(v2, e2))])
    circ = circumscribed_2d(p2, c2, s2)
    if circ is None: continue
    cx, cy, R = circ
    # Project center back to H-dim, then to shared plane
    center_H = P + cx * e1 + cy * e2
    cx_sh = float(np.dot(center_H, e1_sh))
    cy_sh = float(np.dot(center_H, e2_sh))
    centers_2d.append((cx_sh, cy_sh))

if centers_2d:
    cx_mean = np.mean([c[0] for c in centers_2d])
    cy_mean = np.mean([c[1] for c in centers_2d])
    print(f"  Mean circle center in shared plane: ({cx_mean:.4f}, {cy_mean:.4f})")
    print(f"  Distance from shared-plane origin: "
          f"{np.sqrt(cx_mean**2 + cy_mean**2):.4f}")
    print()

# ── Part D: Chord length analysis ────────────────────────────────────
print()
print("=" * 70)
print("PART D+E: CHORD LENGTHS — semantic distance measure per paradigm")
print("          chord = 2R·sin(Ω/2)  vs  measured ||b-a||₂")
print("=" * 70)
print()
print(f"  {'paradigm':<16}  {'R':>6}  {'Ω':>7}  {'chord_arc':>10}  "
      f"{'chord_meas':>11}  {'diff':>7}  φ-class")

chord_results = {}
PHI_CLASSES = {
    "π/φ=111.2°": 180/PHI,
    "π/2=90.0°": 90,
    "2π/3=120°": 120,
    "5π/6=150°": 150,
}

for pname, pairs in ALL_PARADIGMS.items():
    valid = [(a, b) for a, b in pairs if get_emb(a) is not None and get_emb(b) is not None]
    if not valid: continue

    # Measured chord lengths
    chords_meas = [float(np.linalg.norm(get_emb(b) - get_emb(a))) for a, b in valid]

    # Arc-predicted chord: from multi-paradigm results
    # Use the (O, a, b) arc Ω values
    O = np.zeros(H)
    Rs = []; Omegas = []
    for a_w, b_w in valid:
        A = get_emb(a_w); B = get_emb(b_w)
        # Build (O, A, B) arc
        v1 = A - O; v2 = B - O
        D = np.stack([v1, v2], axis=1)
        U_loc, sv_loc, _ = np.linalg.svd(D, full_matrices=False)
        if sv_loc[1] < 1e-10: continue
        e1_loc, e2_loc = U_loc[:,0], U_loc[:,1]
        p2_ = np.array([0., 0.])
        a2_ = np.array([float(np.dot(v1, e1_loc)), float(np.dot(v1, e2_loc))])
        b2_ = np.array([float(np.dot(v2, e1_loc)), float(np.dot(v2, e2_loc))])
        circ_loc = circumscribed_2d(p2_, a2_, b2_)
        if circ_loc is None: continue
        cx_loc, cy_loc, R_loc = circ_loc
        ang_a = arc_angle_2d(cx_loc, cy_loc, a2_[0], a2_[1])
        ang_b = arc_angle_2d(cx_loc, cy_loc, b2_[0], b2_[1])
        arc_ab = abs(signed_arc(ang_a, ang_b))
        chord_arc = 2 * R_loc * np.sin(arc_ab / 2)
        Rs.append(R_loc); Omegas.append(float(np.degrees(arc_ab)))

    if not Rs: continue
    R_m = float(np.mean(Rs)); Om = float(np.mean(Omegas))
    chord_arc = 2 * R_m * np.sin(np.radians(Om) / 2)
    chord_meas = float(np.mean(chords_meas))
    diff = chord_arc - chord_meas

    # φ-class
    phi_class = min(PHI_CLASSES.items(), key=lambda x: abs(Om - x[1]))[0]

    print(f"  {pname:<16}  {R_m:>6.3f}  {Om:>7.2f}°  {chord_arc:>10.4f}  "
          f"{chord_meas:>11.4f}  {diff:>7.4f}  {phi_class}")
    chord_results[pname] = {"R": R_m, "Omega": Om, "chord_arc": chord_arc,
                            "chord_meas": chord_meas}

print()
print("  chord_arc: predicted from arc model (2R·sin(Ω/2))")
print("  chord_meas: actual mean ||b-a||₂ over training pairs")
print("  diff: chord_arc - chord_meas")
print("  Small |diff| = arc model accurately predicts chord length")

output = {
    "exact_acc": exact_acc / n if n > 0 else 0,
    "canonical_acc": canonical_acc / n if n > 0 else 0,
    "exact_results": exact_results,
    "chord_results": chord_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Corrected oracle analysis complete.")
