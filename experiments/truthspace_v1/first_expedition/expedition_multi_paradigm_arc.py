#!/usr/bin/env python3
"""
Multi-Paradigm Arc Geometry

DC 385 established: adj_degree triples trace a consistent circular arc
in a private 2D plane, with R≈0.342, Ω≈229.6°≈2π/φ, d_origin≈0.044.

Key finding: circle center ≈ embedding-space origin (d_origin/R = 0.128).

Extension to pair-based paradigms:
  - adj_degree had THREE points per triple → circumscribed circle
  - Other paradigms only have PAIRS (source, target)

Natural extension: include the embedding-space origin O=(0,...,0) as a
third point. Since d_origin≈0 for adj_degree arcs (circle center ≈ O),
the triangle (O, source, target) defines a circle that should be similar
to the adj_degree circumscribed circles.

For adj_degree, also verify: does the triangle (O, pos, comp) give the
same circle as the triangle (pos, comp, sup)? They should, if O ≈ circle
center.

Questions:
  A. For each paradigm, what is the circumscribed circle of (O, a, b)?
     Compare R, Ω_ab (arc a→b), and consistency across pairs.

  B. Do different paradigms have different characteristic arc angles Ω_ab?
     adj_degree steps ≈ π/φ. What about gender, past_tense, plural?

  C. Is R consistent within a paradigm? Between paradigms?

  D. For adj_degree triples: does circumscribed circle of (O, pos, comp)
     match the circle of (pos, comp, sup)?

  E. For past_tense: do we have base→past→-ing triples?
     If so, compute the three-point arc directly.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "multi_paradigm_arc.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2

PARADIGMS = {
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
    "antonym_size": [("big","small"),("large","tiny"),("huge","little"),
                     ("tall","short"),("wide","narrow"),("thick","thin"),
                     ("broad","slim"),("heavy","light"),("long","brief")],
}

# For three-point arc verification (Part D)
ADJ_TRIPLES = [
    ("big","bigger","biggest"), ("fast","faster","fastest"),
    ("long","longer","longest"), ("small","smaller","smallest"),
    ("hard","harder","hardest"), ("bright","brighter","brightest"),
    ("old","older","oldest"), ("tall","taller","tallest"),
    ("strong","stronger","strongest"), ("cool","cooler","coolest"),
]

# For past_tense with -ing triples (Part E)
VERB_TRIPLES = [
    ("walk","walked","walking"), ("talk","talked","talking"),
    ("call","called","calling"), ("pull","pulled","pulling"),
    ("look","looked","looking"), ("play","played","playing"),
    ("stay","stayed","staying"), ("jump","jumped","jumping"),
    ("work","worked","working"), ("move","moved","moving"),
    ("help","helped","helping"), ("turn","turned","turning"),
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

def svd_basis_2d(P, Q, R_pt):
    """SVD orthonormal 2D basis for plane of (P, Q, R_pt)."""
    v1 = Q - P; v2 = R_pt - P
    D = np.stack([v1, v2], axis=1)
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    if len(U[0]) < 2 or sv[1] < 1e-10: return None
    e1 = U[:, 0]; e2 = U[:, 1]
    p2 = np.array([0.0, 0.0])
    q2 = np.array([float(np.dot(v1, e1)), float(np.dot(v1, e2))])
    r2 = np.array([float(np.dot(v2, e1)), float(np.dot(v2, e2))])
    return e1, e2, p2, q2, r2

def circumscribed_circle(p2, q2, r2):
    ax, ay = p2; bx, by = q2; cx, cy = r2
    D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-12: return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) +
          (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) +
          (cx**2+cy**2)*(bx-ax)) / D
    R = float(np.sqrt((ax-ux)**2+(ay-uy)**2))
    return float(ux), float(uy), R

def arc_angle_2d(cx, cy, x, y):
    return float(np.arctan2(y-cy, x-cx))

def signed_arc(a1, a2):
    d = a2 - a1
    while d >  np.pi: d -= 2*np.pi
    while d < -np.pi: d += 2*np.pi
    return d

def arc_of_triple(P, Q, R_pt, label_P="P", label_Q="Q", label_R="R"):
    """Compute arc geometry for three points P, Q, R_pt."""
    proj = svd_basis_2d(P, Q, R_pt)
    if proj is None: return None
    e1, e2, p2, q2, r2 = proj
    circ = circumscribed_circle(p2, q2, r2)
    if circ is None: return None
    cx, cy, R = circ
    ap = arc_angle_2d(cx, cy, p2[0], p2[1])
    aq = arc_angle_2d(cx, cy, q2[0], q2[1])
    ar = arc_angle_2d(cx, cy, r2[0], r2[1])
    arc_pq = signed_arc(ap, aq)
    arc_qr = signed_arc(aq, ar)
    arc_pr = arc_pq + arc_qr
    # Origin in 2D (embedding zero projected into plane)
    orig = np.array([-float(np.dot(P, e1)), -float(np.dot(P, e2))])
    d_orig = float(np.linalg.norm(orig - np.array([cx, cy])))
    return {
        "R": R,
        "Omega_PQ": float(np.degrees(abs(arc_pq))),
        "Omega_QR": float(np.degrees(abs(arc_qr))),
        "Omega_total": float(np.degrees(abs(arc_pr))),
        "t_Q": abs(arc_pq)/(abs(arc_pr)+1e-8),
        "d_origin": d_orig,
        "orientation": "CCW" if arc_pr > 0 else "CW",
    }

# ── Part A: Pair-based arc (O, source, target) ──────────────────────
print("=" * 70)
print("PART A: Pair arc geometry via (origin, source, target)")
print("        The embedding-space zero vector O is the third point.")
print("=" * 70)
print()
O = np.zeros(H, dtype=np.float64)  # embedding-space origin

print(f"  {'paradigm':<15}  n  {'R':>7}  {'R_std':>6}  {'Ω_src-tgt':>10}  "
      f"{'Ω_std':>7}  {'t_src':>7}  {'consist'}")

all_results = {}
for pname, pairs in PARADIGMS.items():
    arc_data = []
    for a_w, b_w in pairs:
        A = get_emb(a_w); B = get_emb(b_w)
        if A is None or B is None: continue
        # Arc of (O, A, B): src = A is the "second" point after O
        res = arc_of_triple(O, A, B)
        if res is None: continue
        arc_data.append(res)

    if not arc_data: continue
    Rs = [d["R"] for d in arc_data]
    Omegas = [d["Omega_QR"] for d in arc_data]  # arc from A to B
    ts     = [d["t_Q"] for d in arc_data]        # position of A (= t_src)
    ccw    = sum(1 for d in arc_data if d["orientation"] == "CCW")
    n = len(arc_data)
    # Consistency: std/mean ratio
    consist = "HIGH" if np.std(Rs)/np.mean(Rs) < 0.08 else \
              "MED" if np.std(Rs)/np.mean(Rs) < 0.15 else "LOW"
    print(f"  {pname:<15}  {n}  {np.mean(Rs):>7.3f}  {np.std(Rs):>6.3f}  "
          f"{np.mean(Omegas):>10.2f}°  {np.std(Omegas):>7.2f}°  "
          f"{np.mean(ts):>7.4f}  {consist}")
    all_results[pname] = {
        "R_mean": float(np.mean(Rs)), "R_std": float(np.std(Rs)),
        "Omega_mean": float(np.mean(Omegas)), "Omega_std": float(np.std(Omegas)),
        "t_src_mean": float(np.mean(ts)), "n": n,
    }

# ── Part B: Arc angle comparison across paradigms ────────────────────
print()
print("=" * 70)
print("PART B: Arc angle Ω_ab comparison across paradigms")
print("        For (O, source, target): what arc does source→target subtend?")
print("        Reference: adj_degree step Ω_pc ≈ π/φ = 111.25°")
print("=" * 70)
print()
phi_ref = 180/PHI  # π/φ in degrees
print(f"  π/φ = {phi_ref:.2f}°    2π/φ = {360/PHI:.2f}°")
print(f"  π   = 180.00°  2π/3 = 120.00°  π/2 = 90.00°  π/3 = 60.00°")
print()
for pname, res in all_results.items():
    om = res["Omega_mean"]
    matches = []
    for label, val in [("π/φ=111.25°", 180/PHI), ("2π/φ=222.5°", 360/PHI),
                        ("120°=2π/3", 120), ("90°=π/2", 90),
                        ("60°=π/3", 60), ("180°=π", 180),
                        ("π/φ²=68.75°", 180/PHI**2)]:
        if abs(om - val) < 8: matches.append(label)
    match_str = ", ".join(matches) if matches else "no match"
    print(f"  {pname:<15}  Ω={om:>7.2f}°  {match_str}")

# ── Part D: Verify adj_degree: (O,pos,comp) vs (pos,comp,sup) ────────
print()
print("=" * 70)
print("PART D: Verification — does (O, pos, comp) give the same circle as")
print("        (pos, comp, sup)? If d_origin≈0 is correct, they should match.")
print("=" * 70)
print()
print(f"  {'word':<8}  {'R_Opc':>7}  {'R_pcs':>7}  {'Ω(p→c)_Opc':>12}  "
      f"{'Ω(p→c)_pcs':>12}  match?")

for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C is None or S is None: continue
    # (O, pos, comp) circle: arc of P→C in this circle
    r1 = arc_of_triple(O, P, C)
    # (pos, comp, sup) circle: arc of P→C in this circle
    r2 = arc_of_triple(P, C, S)
    if r1 is None or r2 is None: continue
    R_Opc = r1["R"]; Omega_pc_Opc = r1["Omega_QR"]  # arc O→P is Omega_PQ, P→C is Omega_QR
    R_pcs = r2["R"]; Omega_pc_pcs = r2["Omega_PQ"]  # P is first, C is second
    ok = abs(R_Opc - R_pcs) < 0.05 and abs(Omega_pc_Opc - Omega_pc_pcs) < 10
    print(f"  {pos_w:<8}  {R_Opc:>7.3f}  {R_pcs:>7.3f}  {Omega_pc_Opc:>12.2f}°  "
          f"{Omega_pc_pcs:>12.2f}°  {'YES' if ok else 'NO'}")

# ── Part E: Verb triples (base, past, -ing) ─────────────────────────
print()
print("=" * 70)
print("PART E: Verb triples — arc of (base, past_tense, -ing_form)")
print("        Compare to adj_degree triple arc.")
print("=" * 70)
print()
print(f"  {'verb':<8}  {'R':>7}  {'Ω_total':>9}  {'Ω_b→p':>8}  {'Ω_p→ing':>9}  "
      f"{'t_past':>8}  {'d_orig':>8}")

verb_arcs = []
for base_w, past_w, ing_w in VERB_TRIPLES:
    A = get_emb(base_w); B = get_emb(past_w); C = get_emb(ing_w)
    if A is None or B is None or C is None:
        print(f"  {base_w:<8}  (missing token)")
        continue
    res = arc_of_triple(A, B, C)
    if res is None:
        print(f"  {base_w:<8}  (degenerate)")
        continue
    print(f"  {base_w:<8}  {res['R']:>7.3f}  {res['Omega_total']:>9.2f}°  "
          f"{res['Omega_PQ']:>8.2f}°  {res['Omega_QR']:>9.2f}°  "
          f"{res['t_Q']:>8.4f}  {res['d_origin']:>8.3f}")
    verb_arcs.append(res)

if verb_arcs:
    Rs = [r["R"] for r in verb_arcs]
    Ots = [r["Omega_total"] for r in verb_arcs]
    print(f"\n  SUMMARY (verb triples):")
    print(f"    R:       mean={np.mean(Rs):.3f}  std={np.std(Rs):.3f}")
    print(f"    Ω_total: mean={np.mean(Ots):.2f}°  std={np.std(Ots):.2f}°")
    print()
    print(f"  vs adj_degree: R=0.342  Ω_total=229.6°")

# ── Summary table ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY: Arc geometry comparison across all paradigms")
print("=" * 70)
print()
print(f"  {'paradigm':<15}  {'R_mean':>8}  {'Ω_ab':>8}  {'t_src':>7}  notes")
for pname, res in all_results.items():
    notes = []
    if abs(res["Omega_mean"] - 180/PHI) < 8:    notes.append("≈π/φ")
    if abs(res["Omega_mean"] - 360/PHI) < 8:    notes.append("≈2π/φ")
    if abs(res["Omega_mean"] - 120) < 8:         notes.append("≈2π/3")
    if abs(res["t_src_mean"] - 0.5) < 0.05:     notes.append("t≈½")
    if abs(res["t_src_mean"] - 1/PHI) < 0.05:   notes.append("t≈1/φ")
    print(f"  {pname:<15}  {res['R_mean']:>8.3f}  {res['Omega_mean']:>8.2f}°  "
          f"{res['t_src_mean']:>7.4f}  {', '.join(notes)}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"pair_arcs": all_results, "verb_arcs": [
        {k: float(v) if not isinstance(v, str) else v for k, v in r.items()}
        for r in verb_arcs
    ]}, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Multi-paradigm arc analysis complete.")
