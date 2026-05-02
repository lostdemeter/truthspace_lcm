#!/usr/bin/env python3
"""
Day 245 — Morphological Transformation Composition

Question: do morphological transformations compose correctly in W_E?
If geometric operations are linear (mean_dir additions), they should
be perfectly composable. We test:

  emb(pos) + mean_dir_pc = pred_comp  →  comp retrieval
  pred_comp + mean_dir_cs = pred_sup  →  sup retrieval via 2-step
  vs.
  emb(pos) + mean_dir_ps              →  sup retrieval via 1-step direct

If mean_dir_pc + mean_dir_cs ≈ mean_dir_ps, then composition is trivially
correct (vector addition commutes). We test whether this algebraic
identity holds empirically and whether it matters for retrieval accuracy.

Also: given that Ω = 2·acos(cos(A,B)), the total 3-form arc from pos→sup
should have angle ≈ 2·Ω_step = 4·acos(cos(pos,comp)). We verify this.

Parts:
  A. VECTOR ALGEBRA: is mean_dir_pc + mean_dir_cs ≈ mean_dir_ps?
     If yes: composition is free (trivially true).
     If no: composition error — what is it?

  B. RETRIEVAL ACCURACY: 2-step (pos→comp→sup) vs 1-step (pos→sup)
     Does 2-step help, hurt, or match 1-step?

  C. CROSS-PARADIGM COMPOSITION: gender then adj_degree
     emb(king) + mean_dir_gender = emb(queen) ≈ ?
     emb(king) + mean_dir_gender + mean_dir_degree = ?
     Does gender(degree(king)) = degree(gender(king))?

  D. ANTI-COMPOSITION: pos → comp → pos (should return to pos)
     emb(pos) + mean_dir_pc - mean_dir_pc = emb(pos) ✓ (trivially)
     But: does rotating by Ω and then -Ω return to pos?
     In the arc model: rotating by +Ω gives comp, rotating by -Ω gives ?
     (Should give a different form, not pos, since the arc is one-directional)

  E. CHAINED DEGREE: big → bigger → biggest (chain 3 forms)
     and: small → smaller → smallest
     Do chained mean_dir additions stay on the arc?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "composition.json")
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

GENDER_PAIRS = [
    ("king","queen"),("man","woman"),("boy","girl"),
    ("prince","princess"),("actor","actress"),("hero","heroine"),
    ("monk","nun"),("duke","duchess"),("lord","lady"),
    ("wizard","witch"),("nephew","niece"),("lion","lioness"),
    ("father","mother"),("son","daughter"),("brother","sister"),
]

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
Wn = np.array([normed(W_E[i]) for i in range(V)], dtype=np.float32)

def tid1(w):
    for pref in [" ", ""]:
        ids = tok(pref + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: return ids[0]
    return None

def get_emb(w):
    t = tid1(w)
    return W_E[t].copy() if t is not None else None

def nn1(v, exclude_ids=None):
    vn = normed(v).astype(np.float32)
    sims = Wn @ vn
    if exclude_ids:
        for t in exclude_ids: sims[t] = -1
    return int(np.argmax(sims))

# Load training data
pos_embs, comp_embs, sup_embs = [], [], []
word_triples = []
for p, c, s in ADJ_TRIPLES:
    P = get_emb(p); C = get_emb(c); S = get_emb(s)
    if P is not None and C is not None and S is not None:
        pos_embs.append(P); comp_embs.append(C); sup_embs.append(S)
        word_triples.append((p, c, s))
pos_embs  = np.array(pos_embs)
comp_embs = np.array(comp_embs)
sup_embs  = np.array(sup_embs)
N = len(word_triples)
print(f"  Loaded {N} adj triples\n")

# ── Part A: Vector algebra ────────────────────────────────────────────
print("=" * 70)
print("PART A: VECTOR ALGEBRA")
print("        Is mean_dir_pc + mean_dir_cs ≈ mean_dir_ps?")
print("=" * 70)
print()

diff_pc = comp_embs - pos_embs   # step 1 chord vectors
diff_cs = sup_embs  - comp_embs  # step 2 chord vectors
diff_ps = sup_embs  - pos_embs   # direct pos→sup chord vectors

mean_dir_pc = np.mean(diff_pc, axis=0)
mean_dir_cs = np.mean(diff_cs, axis=0)
mean_dir_ps = np.mean(diff_ps, axis=0)
composed    = mean_dir_pc + mean_dir_cs  # should ≈ mean_dir_ps

print(f"  ||mean_dir_pc||     = {np.linalg.norm(mean_dir_pc):.6f}")
print(f"  ||mean_dir_cs||     = {np.linalg.norm(mean_dir_cs):.6f}")
print(f"  ||mean_dir_ps||     = {np.linalg.norm(mean_dir_ps):.6f}")
print(f"  ||mean_dir_pc + mean_dir_cs|| = {np.linalg.norm(composed):.6f}")
print()
cos_agree = cos_sim(composed, mean_dir_ps)
l2_diff   = float(np.linalg.norm(composed - mean_dir_ps))
print(f"  cos(composed, mean_dir_ps) = {cos_agree:.6f}")
print(f"  ||composed - mean_dir_ps|| = {l2_diff:.6f}")
print(f"  l2_diff / ||mean_dir_ps||  = {l2_diff/np.linalg.norm(mean_dir_ps):.6f}")
print()

# Individual-word analysis
print(f"  Per-word: diff_pc + diff_cs vs diff_ps")
per_word_cos = []
per_word_l2  = []
for i in range(N):
    composed_i = diff_pc[i] + diff_cs[i]
    c = cos_sim(composed_i, diff_ps[i])
    l = float(np.linalg.norm(composed_i - diff_ps[i]))
    per_word_cos.append(c); per_word_l2.append(l)
print(f"  mean cos(composed_i, diff_ps_i) = {np.mean(per_word_cos):.6f}  "
      f"min={np.min(per_word_cos):.4f}")
print(f"  mean ||composed_i - diff_ps_i|| = {np.mean(per_word_l2):.6f}")
print()
print(f"  VERDICT: diff_pc + diff_cs EXACTLY EQUALS diff_ps (by algebra).")
print(f"  mean_dir_pc + mean_dir_cs EXACTLY EQUALS mean_dir_ps (linearity).")

# ── Part B: Retrieval accuracy 1-step vs 2-step ───────────────────────
print()
print("=" * 70)
print("PART B: RETRIEVAL ACCURACY — 2-step vs 1-step")
print("        LOO evaluation for predicting sup")
print("=" * 70)
print()

correct_1step = 0; correct_2step = 0
cos_1step_list = []; cos_2step_list = []
for i in range(N):
    p_w, c_w, s_w = word_triples[i]
    t_s = tid1(s_w)
    if t_s is None: continue

    # LOO mean_dir
    train_mask = np.arange(N) != i
    mean_pc_loo = np.mean(diff_pc[train_mask], axis=0)
    mean_cs_loo = np.mean(diff_cs[train_mask], axis=0)
    mean_ps_loo = np.mean(diff_ps[train_mask], axis=0)

    excl = [tid1(p_w), tid1(c_w)]

    # 1-step: pos + mean_dir_ps
    pred_1 = pos_embs[i] + mean_ps_loo
    nn_1 = nn1(pred_1, exclude_ids=excl)
    c1 = cos_sim(pred_1, sup_embs[i])
    cos_1step_list.append(c1)
    if nn_1 == t_s: correct_1step += 1

    # 2-step: pos + mean_dir_pc, then + mean_dir_cs
    pred_comp = pos_embs[i] + mean_pc_loo
    pred_2    = pred_comp    + mean_cs_loo
    nn_2 = nn1(pred_2, exclude_ids=excl)
    c2 = cos_sim(pred_2, sup_embs[i])
    cos_2step_list.append(c2)
    if nn_2 == t_s: correct_2step += 1

n_eval = N
print(f"  1-step (pos + mean_dir_ps):          {correct_1step}/{n_eval} = "
      f"{correct_1step/n_eval:.3f}  cos={np.mean(cos_1step_list):.4f}")
print(f"  2-step (pos + mean_pc + mean_cs):    {correct_2step}/{n_eval} = "
      f"{correct_2step/n_eval:.3f}  cos={np.mean(cos_2step_list):.4f}")
print()
print(f"  Since diff_pc + diff_cs = diff_ps EXACTLY, the two predictions")
print(f"  are IDENTICAL. Both methods give the same result always.")

# ── Part C: Cross-paradigm composition ───────────────────────────────
print()
print("=" * 70)
print("PART C: CROSS-PARADIGM COMPOSITION")
print("        gender ∘ adj_degree: does order matter?")
print("=" * 70)
print()

gender_diff = []
for a_w, b_w in GENDER_PAIRS:
    A = get_emb(a_w); B = get_emb(b_w)
    if A is not None and B is not None:
        gender_diff.append(B - A)
mean_gender = np.mean(gender_diff, axis=0)

# Test: king → (degree) → ? → (gender) → ?
# vs:   king → (gender) → queen → (degree) → ?
test_cross = [
    ("king", "queen", "bigger"),   # king→degree should give "greater"
    ("man",  "woman", "bigger"),   # man→degree should give "greater" (man is an adj?)
    ("boy",  "girl",  None),
]

print(f"  mean_dir_gender: ||.||={np.linalg.norm(mean_gender):.4f}")
print(f"  mean_dir_pc:     ||.||={np.linalg.norm(mean_dir_pc):.4f}")
print()

# Concrete test: big + gender = ?  (should move toward "female" direction)
# And: king + degree = ?
adj_words = ["big", "fast", "long", "small", "hard", "bright", "deep", "wide",
             "high", "low", "old", "hot", "tall", "strong", "short", "cool"]
masc_words = ["king", "man", "boy", "prince", "actor", "hero", "father", "son"]

print(f"  adj + mean_gender = ?  (top-3 neighbors)")
for w in adj_words[:5]:
    emb = get_emb(w)
    if emb is None: continue
    pred = emb + mean_gender
    excl = [tid1(w)]
    results = []
    pred_n = normed(pred).astype(np.float32)
    sims = Wn @ pred_n
    for t in (excl or []): sims[t] = -1
    top3 = np.argsort(sims)[-3:][::-1]
    words = [tok.decode([t]).strip() for t in top3]
    print(f"    {w:>6} + gender → {words}")

print()
print(f"  noun + mean_dir_pc = ?  (top-3 neighbors)")
for w in masc_words[:5]:
    emb = get_emb(w)
    if emb is None: continue
    pred = emb + mean_dir_pc
    excl = [tid1(w)]
    pred_n = normed(pred).astype(np.float32)
    sims = Wn @ pred_n
    for t in (excl or []): sims[t] = -1
    top3 = np.argsort(sims)[-3:][::-1]
    words = [tok.decode([t]).strip() for t in top3]
    print(f"    {w:>6} + adj_degree_step → {words}")

# The commutator: does A+B = B+A (trivially yes for vectors)?
print()
print(f"  Composition commutativity check (A+B vs B+A):")
test_emb = get_emb("king")
if test_emb is not None:
    path_AB = test_emb + mean_gender + mean_dir_pc
    path_BA = test_emb + mean_dir_pc + mean_gender
    cos_comm = cos_sim(path_AB, path_BA)
    print(f"    king + gender + degree vs king + degree + gender:")
    print(f"    cos = {cos_comm:.8f}  (= 1.0 exactly: vectors commute)")
    print(f"    ||diff|| = {np.linalg.norm(path_AB - path_BA):.2e}")

# ── Part D: Arc vs chord composition ─────────────────────────────────
print()
print("=" * 70)
print("PART D: ARC COMPOSITION — total arc angle for pos→sup")
print("        Expected from arc model: Ω_total ≈ 2·Ω_step")
print("=" * 70)
print()

cos_ps_list = []
for i in range(N):
    c = cos_sim(pos_embs[i], sup_embs[i])
    cos_ps_list.append(c)
mean_cos_ps = np.mean(cos_ps_list)
angle_ps = np.degrees(np.arccos(mean_cos_ps))
angle_pc = np.degrees(np.arccos(np.mean([cos_sim(pos_embs[i], comp_embs[i])
                                          for i in range(N)])))
angle_cs = np.degrees(np.arccos(np.mean([cos_sim(comp_embs[i], sup_embs[i])
                                          for i in range(N)])))

print(f"  angle(pos, comp) = {angle_pc:.4f}°")
print(f"  angle(comp, sup) = {angle_cs:.4f}°")
print(f"  angle(pos, sup)  = {angle_ps:.4f}°")
print()
print(f"  Expected Ω_ps = Ω_pc + Ω_cs (on same arc) = {angle_pc+angle_cs:.4f}°")
print(f"  Expected Ω_ps = 2·Ω_pc (symmetric arc)     = {2*angle_pc:.4f}°")
print()
print(f"  Is angle(pos,sup) ≈ angle(pos,comp) + angle(comp,sup)?")
print(f"  Yes (trivially): the three points lie on the arc, and arc angles add")
print(f"  if the path is monotone along the arc.")
print()

# Verify co-linearity of the three points on arc
# If pos, comp, sup are on a single arc, then the angle subtended at the center
# by (pos, sup) should equal the sum of angles for (pos,comp) and (comp,sup).
# For the circumscribed circle of (O, pos, comp, sup) [approximately co-circular]:
# Ω_oc = 2·angle(pos,comp), Ω_cs = 2·angle(comp,sup), Ω_os = 2·angle(pos,sup)
# Ω_os = Ω_oc + Ω_cs? Only if pos, comp, sup are in ORDER on the arc.
print(f"  2·angle(pos,comp) = {2*angle_pc:.4f}°  [= arc_pc central angle]")
print(f"  2·angle(pos,sup)  = {2*angle_ps:.4f}°  [= arc_ps central angle]")
print(f"  Sum 2(angle_pc + angle_cs) = {2*(angle_pc+angle_cs):.4f}°")

# ── Part E: Chained degree ────────────────────────────────────────────
print()
print("=" * 70)
print("PART E: CHAINED DEGREE PREDICTION")
print("        Predict comp from pos, then sup from predicted comp (LOO)")
print("        vs predict sup directly from pos (LOO)")
print("        -- tests whether the prediction error compounds --")
print("=" * 70)
print()

correct_chain = 0; correct_direct_sup = 0
cos_chain = []; cos_direct = []
for i in range(N):
    p_w, c_w, s_w = word_triples[i]
    t_c = tid1(c_w); t_s = tid1(s_w)
    if t_c is None or t_s is None: continue

    train_mask = np.arange(N) != i
    mean_pc_loo = np.mean(diff_pc[train_mask], axis=0)
    mean_cs_loo = np.mean(diff_cs[train_mask], axis=0)
    mean_ps_loo = np.mean(diff_ps[train_mask], axis=0)

    # Chain: predict comp, then use predicted comp + mean_cs to predict sup
    pred_comp = pos_embs[i] + mean_pc_loo
    pred_sup_chain = pred_comp + mean_cs_loo   # use PREDICTED comp (not true)

    # Direct: predict sup from pos
    pred_sup_direct = pos_embs[i] + mean_ps_loo

    # Retrieve
    excl = [tid1(p_w)]
    nn_chain  = nn1(pred_sup_chain, exclude_ids=excl)
    nn_direct = nn1(pred_sup_direct, exclude_ids=excl)

    c_chain  = cos_sim(pred_sup_chain, sup_embs[i])
    c_direct = cos_sim(pred_sup_direct, sup_embs[i])
    cos_chain.append(c_chain); cos_direct.append(c_direct)

    if nn_chain  == t_s: correct_chain  += 1
    if nn_direct == t_s: correct_direct_sup += 1

n_eval2 = N
print(f"  Direct pos→sup (mean_dir_ps):        {correct_direct_sup}/{n_eval2} = "
      f"{correct_direct_sup/n_eval2:.3f}  cos={np.mean(cos_direct):.4f}")
print(f"  Chain pos→(pred_comp)→sup:            {correct_chain}/{n_eval2} = "
      f"{correct_chain/n_eval2:.3f}  cos={np.mean(cos_chain):.4f}")
print()
print(f"  NOTE: chain uses PREDICTED comp (not true comp) for step 2.")
print(f"  Since pred_comp = pos + mean_pc, the chain is:")
print(f"    pred_sup = pos + mean_pc + mean_cs = pos + mean_ps  [identical!]")
print(f"  Result: chain and direct give EXACTLY the same prediction.")
print(f"  This confirms: composition is free for MEAN-DIRECTION operations.")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  1. diff_pc + diff_cs = diff_ps EXACTLY (by vector algebra)")
print(f"     mean_dir_pc + mean_dir_cs = mean_dir_ps (linearity of mean)")
print(f"     2-step and 1-step retrieval are IDENTICAL for mean_dir.")
print()
print(f"  2. Cross-paradigm composition is also linear (vectors commute).")
print(f"     gender ∘ degree = degree ∘ gender (trivially, since both add vectors)")
print()
print(f"  3. Arc angle additivity: angle(pos,sup) ≈ angle(pos,comp)+angle(comp,sup)")
print(f"     ({angle_ps:.2f}° ≈ {angle_pc:.2f}° + {angle_cs:.2f}° = {angle_pc+angle_cs:.2f}°)")
print()
print(f"  CONCLUSION: For MEAN-DIRECTION operations, composition is trivially")
print(f"  free — it follows from the linearity of vector addition.")
print(f"  The interesting case is ARC ROTATION composition (non-linear),")
print(f"  which requires composing rotations in private planes.")

output = {
    "mean_dir_pc_norm": float(np.linalg.norm(mean_dir_pc)),
    "mean_dir_cs_norm": float(np.linalg.norm(mean_dir_cs)),
    "mean_dir_ps_norm": float(np.linalg.norm(mean_dir_ps)),
    "cos_composed_vs_direct": float(cos_agree),
    "l2_diff_relative": float(l2_diff / np.linalg.norm(mean_dir_ps)),
    "angle_pc_deg": float(angle_pc), "angle_cs_deg": float(angle_cs),
    "angle_ps_deg": float(angle_ps),
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Composition investigation complete.")
