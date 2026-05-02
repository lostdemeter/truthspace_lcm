#!/usr/bin/env python3
"""
Arc Rotation Retrieval — Purely Geometric Morphology

DC 385 established: each adj_degree triple lies on a circle of radius
R≈0.342 and arc Ω≈π/φ per step in a PRIVATE 2D plane for that word.

Key question: can we USE this arc structure for retrieval?
If geometry IS computation, rotating by π/φ in the private plane
should give the comparative form.

Problem: the private plane is not known for a novel word.
We only know it after seeing the comparative/superlative form.

So: what CAN we do geometrically?

Approach 1 — SHARED PLANE ROTATION:
  The shared degree plane (SVD of all difference vectors) captures
  41.4% of individual arc variance. Rotate emb(pos) by π/φ in the
  shared plane. Compare to mean_dir.

Approach 2 — LOCAL PLANE ESTIMATION:
  For a novel word W, estimate its private plane from the k nearest
  training words (whose planes we know). Weighted by similarity.
  Then rotate in the estimated plane.

Approach 3 — DIRECT ARC CONSTRUCTION:
  Given emb(pos) and the known arc parameters (R=0.342, Ω=π/φ),
  find the 2D plane for W by searching for the rotation that maps
  emb(pos) to a point approximately in the comp cluster.
  (Self-consistent: find the plane that makes W's arc consistent
  with the universal arc parameters.)

These methods are compared against:
  - mean_dir (baseline)
  - oracle: rotation in the TRUE private plane (upper bound)

LOO evaluation on all 24 adj_degree triples.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "arc_rotation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2
R_CANONICAL = 0.342  # universal arc radius
OMEGA_STEP  = 180 / PHI  # π/φ in degrees ≈ 111.25°

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
def is_single(w): return get_emb(w) is not None

# ── Pool for retrieval ────────────────────────────────────────────────
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))
for t3 in ADJ_TRIPLES:
    for w in t3:
        if w not in pool_words:
            e = get_emb(w)
            if e is not None:
                pool_words.append(w); pool_embs.append(e.astype(np.float32))
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
    """SVD 2D basis for plane of (P,C,S)."""
    v1 = C - P; v2 = S - P
    D = np.stack([v1, v2], axis=1)
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    if sv[1] < 1e-10: return None
    return U[:,0], U[:,1]

def rotate_in_plane(v, e1, e2, angle_deg):
    """
    Rotate vector v by angle_deg within the 2D plane spanned by e1, e2.
    The rotation is around the component of v perpendicular to the plane.
    Only the in-plane component is rotated; out-of-plane stays fixed.
    """
    angle_rad = np.radians(angle_deg)
    # Project v onto plane
    a = float(np.dot(v, e1))
    b = float(np.dot(v, e2))
    # Rotate 2D component
    cos_a = np.cos(angle_rad); sin_a = np.sin(angle_rad)
    a_new = a * cos_a - b * sin_a
    b_new = a * sin_a + b * cos_a
    # Out-of-plane component stays
    v_perp = v - a * e1 - b * e2
    return v_perp + a_new * e1 + b_new * e2

# ── Build private planes for all triples ─────────────────────────────
private_planes = {}  # pos_word → (e1, e2)
for pos_w, comp_w, sup_w in ADJ_TRIPLES:
    P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C is None or S is None: continue
    res = svd_plane(P, C, S)
    if res is None: continue
    private_planes[pos_w] = (res[0], res[1])

print(f"  Private planes: {len(private_planes)}/{len(ADJ_TRIPLES)}\n")

# ── Build shared degree plane (LOO: exclude test word) ────────────────
def get_shared_plane(exclude_words=None):
    """
    Compute shared degree plane from all triples except excluded words.
    Returns (e1_shared, e2_shared).
    """
    diffs = []
    for pos_w, comp_w, sup_w in ADJ_TRIPLES:
        if exclude_words and pos_w in exclude_words: continue
        P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
        if P is None or C is None or S is None: continue
        diffs.append(C - P)
        diffs.append(S - P)
    if not diffs: return None, None
    D = np.array(diffs).T
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    return U[:,0], U[:,1]

# ── Methods ───────────────────────────────────────────────────────────

def method_mean_dir(pos_w, train_triples, target="comp"):
    """Baseline: add mean direction."""
    ea = get_emb(pos_w)
    if ea is None: return None
    if target == "comp":
        diffs = [get_emb(c) - get_emb(p) for p,c,s in train_triples
                 if get_emb(p) is not None and get_emb(c) is not None]
    else:
        diffs = [get_emb(s) - get_emb(p) for p,c,s in train_triples
                 if get_emb(p) is not None and get_emb(s) is not None]
    if not diffs: return None
    d = normed(np.mean(diffs, axis=0))
    return top1(ea + d, exclude={pos_w})

def method_shared_plane_rotation(pos_w, train_triples, target="comp"):
    """
    Rotate emb(pos) by π/φ (comp) or 2×π/φ (sup) in the shared degree plane.
    Shared plane is fit on training triples (LOO).
    """
    ea = get_emb(pos_w)
    if ea is None: return None
    e1, e2 = get_shared_plane(exclude_words={pos_w})
    if e1 is None: return None
    angle = OMEGA_STEP if target == "comp" else 2 * OMEGA_STEP
    rotated = rotate_in_plane(ea, e1, e2, angle)
    return top1(rotated, exclude={pos_w})

def method_local_plane_rotation(pos_w, train_triples, target="comp", k=5):
    """
    Estimate private plane for pos_w by weighting nearest training words'
    private planes by similarity of their source embeddings.
    Then rotate in this estimated plane.
    """
    ea = get_emb(pos_w)
    if ea is None: return None
    # Find k nearest source words in training
    sims = []
    for p, c, s in train_triples:
        ep = get_emb(p)
        if ep is None: continue
        sims.append((float(np.dot(normed(ea), normed(ep))), p))
    sims.sort(reverse=True)
    top_k = sims[:k]
    if not top_k: return None
    total_w = sum(max(0, s) for s, _ in top_k)
    if total_w < 1e-8: return None
    # Weighted combination of private plane basis vectors
    e1_est = np.zeros(H); e2_est = np.zeros(H)
    for sim, p_w in top_k:
        if p_w not in private_planes: continue
        e1_w, e2_w = private_planes[p_w]
        e1_est += max(0, sim) * e1_w
        e2_est += max(0, sim) * e2_w
    # Re-orthonormalize via SVD
    basis = np.stack([e1_est, e2_est], axis=1)
    U, sv, _ = np.linalg.svd(basis, full_matrices=False)
    if sv[1] < 1e-10: return None
    e1_est, e2_est = U[:,0], U[:,1]
    angle = OMEGA_STEP if target == "comp" else 2 * OMEGA_STEP
    rotated = rotate_in_plane(ea, e1_est, e2_est, angle)
    return top1(rotated, exclude={pos_w})

def method_oracle_rotation(pos_w, train_triples, target="comp"):
    """
    Oracle: rotate in the TRUE private plane for pos_w.
    This is the upper bound for plane-rotation methods.
    (Not a fair LOO method — uses the true plane which requires comp/sup.)
    """
    ea = get_emb(pos_w)
    if ea is None: return None
    if pos_w not in private_planes: return None
    e1, e2 = private_planes[pos_w]
    angle = OMEGA_STEP if target == "comp" else 2 * OMEGA_STEP
    rotated = rotate_in_plane(ea, e1, e2, angle)
    return top1(rotated, exclude={pos_w})

# ── LOO evaluation ────────────────────────────────────────────────────
METHODS = {
    "mean_dir":       (method_mean_dir, {}),
    "shared_plane":   (method_shared_plane_rotation, {}),
    "local_plane_k3": (method_local_plane_rotation, {"k": 3}),
    "local_plane_k5": (method_local_plane_rotation, {"k": 5}),
    "oracle_plane":   (method_oracle_rotation, {}),
}

print("=" * 70)
print("LOO EVALUATION — comp prediction (pos → comp)")
print("=" * 70)
print()

triples = [(p,c,s) for p,c,s in ADJ_TRIPLES if is_single(p) and is_single(c) and is_single(s)]
n = len(triples)
comp_counts = {m: 0 for m in METHODS}
sup_counts  = {m: 0 for m in METHODS}

# Per-word results
print(f"  {'word':<8}  {'comp':>6}  {'mean':>5}  {'shr':>5}  "
      f"{'loc3':>5}  {'loc5':>5}  {'orc':>5}")
for i, (pos_w, comp_w, sup_w) in enumerate(triples):
    train = [t for j, t in enumerate(triples) if j != i]
    comp_preds = {}; sup_preds = {}
    for mname, (mfunc, mkw) in METHODS.items():
        comp_preds[mname] = mfunc(pos_w, train, target="comp", **mkw)
        sup_preds[mname]  = mfunc(pos_w, train, target="sup",  **mkw)
        if comp_preds[mname] == comp_w: comp_counts[mname] += 1
        if sup_preds[mname]  == sup_w:  sup_counts[mname]  += 1
    row = f"  {pos_w:<8}  {comp_w:<6}  "
    row += "  ".join(f"{'Y' if comp_preds[m]==comp_w else 'N':>4}" for m in list(METHODS.keys())[:5])
    print(row)

print()
print("  COMP accuracy:")
for mname in METHODS:
    print(f"    {mname:<20}  {comp_counts[mname]:>2}/{n}  =  {comp_counts[mname]/n:.3f}")

print()
print("=" * 70)
print("LOO EVALUATION — sup prediction (pos → sup)")
print("=" * 70)
print()
print("  SUP accuracy:")
for mname in METHODS:
    print(f"    {mname:<20}  {sup_counts[mname]:>2}/{n}  =  {sup_counts[mname]/n:.3f}")

# ── Geometric analysis of rotation quality ──────────────────────────
print()
print("=" * 70)
print("ROTATION QUALITY: cosine of (predicted, actual) comp/sup embeddings")
print("How close is the rotated vector to the actual target embedding?")
print("=" * 70)
print()
e1_shared, e2_shared = get_shared_plane()

print(f"  {'word':<8}  {'cos(rot_shr,comp)':>18}  {'cos(rot_orc,comp)':>18}  "
      f"{'cos(mean+dir,comp)':>20}")
cos_shared = []; cos_oracle = []; cos_mean = []
for pos_w, comp_w, sup_w in triples:
    P = get_emb(pos_w); C = get_emb(comp_w); S = get_emb(sup_w)
    if P is None or C is None: continue
    # Mean direction
    diffs = [get_emb(c) - get_emb(p_) for p_,c,s in triples
             if p_ != pos_w and get_emb(p_) is not None and get_emb(c) is not None]
    d_mean = normed(np.mean(diffs, axis=0))
    pred_mean = P + d_mean
    cos_m = float(np.dot(normed(pred_mean), normed(C)))
    # Shared plane rotation
    rot_shr = rotate_in_plane(P, e1_shared, e2_shared, OMEGA_STEP)
    cos_s = float(np.dot(normed(rot_shr), normed(C)))
    # Oracle rotation
    if pos_w in private_planes:
        e1_p, e2_p = private_planes[pos_w]
        rot_orc = rotate_in_plane(P, e1_p, e2_p, OMEGA_STEP)
        cos_o = float(np.dot(normed(rot_orc), normed(C)))
    else:
        cos_o = float("nan")
    print(f"  {pos_w:<8}  {cos_s:>18.4f}  {cos_o:>18.4f}  {cos_m:>20.4f}")
    cos_shared.append(cos_s); cos_oracle.append(cos_o)
    cos_mean.append(cos_m)

cos_oracle = [c for c in cos_oracle if not np.isnan(c)]
print(f"\n  Mean cosine similarities:")
print(f"    Shared plane rotation: {np.mean(cos_shared):.4f} ± {np.std(cos_shared):.4f}")
print(f"    Oracle plane rotation: {np.mean(cos_oracle):.4f} ± {np.std(cos_oracle):.4f}")
print(f"    Mean direction:        {np.mean(cos_mean):.4f} ± {np.std(cos_mean):.4f}")

output = {
    "comp_accuracy": {m: comp_counts[m]/n for m in METHODS},
    "sup_accuracy":  {m: sup_counts[m]/n  for m in METHODS},
    "cos_shared_mean": float(np.mean(cos_shared)),
    "cos_oracle_mean": float(np.mean(cos_oracle)),
    "cos_mean_dir_mean": float(np.mean(cos_mean)),
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Arc rotation analysis complete.")
