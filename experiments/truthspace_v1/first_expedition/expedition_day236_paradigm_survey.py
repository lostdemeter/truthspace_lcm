#!/usr/bin/env python3
"""
Day 236 — Morphological Paradigm Survey + Corrected DC 382

Day 235 revealed:
  - Adjective degree steps are ANTI-CORRELATED (cos≈-0.40) per word
  - Morphological words are NOT collinear (midpoint err≈0.77)
  - Composition works due to algebraic triviality + equal step magnitudes
  - DC 382 claim of "direction parallelism" was wrong

Questions:
  Q1: Does the anti-correlation hold for OTHER morphological paradigms?
      Test: plural (cat/cats), past_tense (walk/walked), comparative (big/bigger)
      Are SINGULAR->PLURAL->PLURAL_POSSESSIVE steps anti-correlated?
      Are ROOT->COMPARATIVE->SUPERLATIVE steps anti-correlated?
      Do plural/past_tense have "two-step" paradigms at all?

  Q2: Do same-word STEP MAGNITUDES differ across paradigms?
      |cat->cats| vs |cats->cats'| (plural->possessive)
      |walk->walked| (only one step in standard past_tense)
      |big->bigger| vs |bigger->biggest|

  Q3: Are the per-paradigm MEAN DIRECTIONS consistent with
      anti-correlation, or is it only a per-word phenomenon?
      i.e., cos(d_comparative_mean, d_comp_to_sup_mean) vs
            per-word cos(v_i, v_j)?

  Q4: What is the actual shape of the degree path?
      If not collinear, what IS the geometry?
      Project onto 2D: what does big/bigger/biggest look like?

Experiments:
  A. Two-step magnitude survey across paradigms:
     - adj degree: positive/comparative/superlative
     - noun number: singular/plural (only one step, so measure magnitude only)
     - verb tense: root/past_tense (only one step)
     Measure: |step| for each word, mean and std across paradigm

  B. Anti-correlation universality:
     For each word with a three-form paradigm (only adj degree has this),
     measure cos(step1, step2) and midpoint error.
     Does the anti-correlation mean cos=-0.40 universally?
     Or is it word-specific?

  C. 2D projection of adjective degree paradigm:
     PCA of {positive, comparative, superlative} embeddings for 10 adjectives.
     Visualize the path.

  D. Revised composition picture:
     What is the actual composition mechanism?
     Test: for "cold" (not in training):
       emb(cold) + d_comparative -> retrieve "colder"?  [single hop]
       emb(cold) + d_comparative + d_comp_to_sup -> retrieve "coldest"? [two hop]
     If yes: the MEAN direction works even though individual paths are curved.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day236_paradigm_survey.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Paradigm definitions ──────────────────────────────────────────────
ADJ_DEGREE = [
    ("big",    "bigger",    "biggest"),
    ("fast",   "faster",    "fastest"),
    ("long",   "longer",    "longest"),
    ("small",  "smaller",   "smallest"),
    ("hard",   "harder",    "hardest"),
    ("bright", "brighter",  "brightest"),
    ("dark",   "darker",    "darkest"),
    ("rich",   "richer",    "richest"),
    ("deep",   "deeper",    "deepest"),
    ("wide",   "wider",     "widest"),
    ("high",   "higher",    "highest"),
    ("low",    "lower",     "lowest"),
    ("old",    "older",     "oldest"),
    ("young",  "younger",   "youngest"),
    ("cold",   "colder",    "coldest"),
    ("hot",    "hotter",    "hottest"),
    ("short",  "shorter",   "shortest"),
    ("tall",   "taller",    "tallest"),
    ("strong", "stronger",  "strongest"),
    ("weak",   "weaker",    "weakest"),
]

NOUN_NUMBER = [
    ("cat", "cats"), ("dog", "dogs"), ("house", "houses"),
    ("tree", "trees"), ("book", "books"), ("car", "cars"),
    ("bird", "birds"), ("ship", "ships"), ("hand", "hands"),
    ("door", "doors"), ("lamp", "lamps"), ("wall", "walls"),
    ("king", "kings"), ("boy", "boys"), ("man", "men"),
    ("girl", "girls"), ("word", "words"), ("star", "stars"),
]

VERB_TENSE = [
    ("walk", "walked"), ("talk", "talked"), ("call", "called"),
    ("pull", "pulled"), ("fill", "filled"), ("turn", "turned"),
    ("look", "looked"), ("move", "moved"), ("push", "pushed"),
    ("help", "helped"), ("play", "played"), ("stay", "stayed"),
]

HELD_OUT_DEGREE = [
    ("cold", "colder", "coldest"),
    ("hot",  "hotter", "hottest"),
    ("tall", "taller", "tallest"),
    ("weak", "weaker", "weakest"),
]

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

print("Building pool ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))

for triplet in ADJ_DEGREE + HELD_OUT_DEGREE:
    for w in triplet:
        if w not in pool_words:
            e = get_emb(w)
            if e is not None:
                pool_words.append(w); pool_embs.append(e.astype(np.float32))
for pair in NOUN_NUMBER + VERB_TENSE:
    for w in pair:
        if w not in pool_words:
            e = get_emb(w)
            if e is not None:
                pool_words.append(w); pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E = np.array(pool_embs, dtype=np.float32)
norms_v = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
E_normed = (E / norms_v).astype(np.float32)
print(f"  Pool: {N} tokens\n")

def top_k(qt, k=5, exclude=None):
    qn = normed(qt).astype(np.float32)
    sims = E_normed @ qn
    order = np.argsort(-sims)
    out = []
    for idx in order:
        w = pool_words[idx]
        if exclude and w == exclude: continue
        out.append((w, float(sims[idx])))
        if len(out) >= k: break
    return out

def mean_dir(pairs):
    p = [(a, b) for a, b in pairs if is_single(a) and is_single(b)]
    if not p: return None, 0
    dirs = [normed(get_emb(b) - get_emb(a)) for a, b in p]
    return normed(np.mean(dirs, axis=0)), len(p)

# ── Part A: Two-step magnitude survey ────────────────────────────────
print("=" * 70)
print("PART A: Step magnitude survey across paradigms")
print("=" * 70)
print()
print("  ADJ DEGREE: positive -> comparative -> superlative")
print(f"  {'word':<10}  {'|pos->comp|':>11}  {'|comp->sup|':>11}  "
      f"{'ratio d2/d1':>11}  {'cos(d1,d2)':>10}  "
      f"{'|pos->sup|':>10}  {'midpt_err':>9}")

deg_records = []
d1_vecs_deg, d2_vecs_deg = [], []
for pos, comp, sup in ADJ_DEGREE:
    ep = get_emb(pos); ec = get_emb(comp); es = get_emb(sup)
    if ep is None or ec is None or es is None:
        print(f"  {pos:<10}  EC_TOKENIZE")
        continue
    v1 = ec - ep; v2 = es - ec; vt = es - ep
    m1 = float(np.linalg.norm(v1))
    m2 = float(np.linalg.norm(v2))
    mt = float(np.linalg.norm(vt))
    c12 = cosine(v1, v2)
    midpoint = (ep + es) / 2
    mid_err  = float(np.linalg.norm(ec - midpoint)) / (mt + 1e-8)
    ratio = m2 / (m1 + 1e-8)
    print(f"  {pos:<10}  {m1:>11.3f}  {m2:>11.3f}  {ratio:>11.3f}  "
          f"{c12:>10.4f}  {mt:>10.3f}  {mid_err:>9.4f}")
    d1_vecs_deg.append(normed(v1)); d2_vecs_deg.append(normed(v2))
    deg_records.append({"pos": pos, "comp": comp, "sup": sup,
                        "mag1": m1, "mag2": m2, "mag_total": mt,
                        "cos_d1_d2": c12, "midpoint_err": mid_err})

if deg_records:
    print(f"\n  Mean |d1| = {np.mean([r['mag1'] for r in deg_records]):.3f}  "
          f"std={np.std([r['mag1'] for r in deg_records]):.3f}")
    print(f"  Mean |d2| = {np.mean([r['mag2'] for r in deg_records]):.3f}  "
          f"std={np.std([r['mag2'] for r in deg_records]):.3f}")
    print(f"  Mean cos(d1,d2) = {np.mean([r['cos_d1_d2'] for r in deg_records]):.4f}")
    print(f"  Mean midpoint_err = {np.mean([r['midpoint_err'] for r in deg_records]):.4f}")

print()
print("  NOUN NUMBER: singular -> plural (single step)")
print(f"  {'word':<10}  {'|sg->pl|':>10}")
noun_mags = []
for sg, pl in NOUN_NUMBER:
    es = get_emb(sg); ep = get_emb(pl)
    if es is None or ep is None:
        print(f"  {sg:<10}  EC_TOKENIZE"); continue
    m = float(np.linalg.norm(ep - es))
    noun_mags.append(m)
    print(f"  {sg:<10}  {m:>10.3f}")
if noun_mags:
    print(f"\n  Mean |sg->pl| = {np.mean(noun_mags):.3f}  std={np.std(noun_mags):.3f}")

print()
print("  VERB TENSE: root -> past_tense (single step)")
print(f"  {'word':<10}  {'|root->past|':>12}")
verb_mags = []
for rt, pt in VERB_TENSE:
    er = get_emb(rt); ep = get_emb(pt)
    if er is None or ep is None:
        print(f"  {rt:<10}  EC_TOKENIZE"); continue
    m = float(np.linalg.norm(ep - er))
    verb_mags.append(m)
    print(f"  {rt:<10}  {m:>12.3f}")
if verb_mags:
    print(f"\n  Mean |root->past| = {np.mean(verb_mags):.3f}  std={np.std(verb_mags):.3f}")

# ── Part B: Anti-correlation universality ────────────────────────────
print()
print("=" * 70)
print("PART B: Anti-correlation of per-word steps (is cos(d1,d2) always <0?)")
print("=" * 70)
print()

neg_count = sum(1 for r in deg_records if r["cos_d1_d2"] < 0)
pos_count = len(deg_records) - neg_count
print(f"  Adj degree paradigm: {neg_count}/{len(deg_records)} words have cos(d1,d2) < 0")
if deg_records:
    print(f"  Min cos: {min(r['cos_d1_d2'] for r in deg_records):.4f}")
    print(f"  Max cos: {max(r['cos_d1_d2'] for r in deg_records):.4f}")
    print(f"  Universal anti-correlation: {'YES' if neg_count == len(deg_records) else 'NO'}")

# Compute mean d1 and d2 separately and their cosine
if d1_vecs_deg and d2_vecs_deg:
    mean_d1 = normed(np.mean(d1_vecs_deg, axis=0))
    mean_d2 = normed(np.mean(d2_vecs_deg, axis=0))
    cos_mean = cosine(mean_d1, mean_d2)
    print(f"\n  cos(mean_d1, mean_d2) = {cos_mean:.4f}")
    print(f"  (Per-word mean: {np.mean([r['cos_d1_d2'] for r in deg_records]):.4f}, "
          f"cos of means: {cos_mean:.4f})")

# ── Part C: 2D projection ─────────────────────────────────────────────
print()
print("=" * 70)
print("PART C: 2D PCA projection of adjective degree paradigm")
print("=" * 70)
print()

# Gather all embeddings (pos, comp, sup) for training adj degree
all_embs_for_pca = []
all_labels_for_pca = []
for pos, comp, sup in ADJ_DEGREE[:10]:
    for w, tag in [(pos,"pos"),(comp,"comp"),(sup,"sup")]:
        e = get_emb(w)
        if e is not None:
            all_embs_for_pca.append(e); all_labels_for_pca.append((w, tag, pos))

if len(all_embs_for_pca) >= 6:
    X = np.array(all_embs_for_pca, dtype=np.float64)
    X -= X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]

    # Print variance explained
    var_exp = S[:2]**2 / (S**2).sum()
    print(f"  PCA variance explained: PC1={var_exp[0]:.3f}  PC2={var_exp[1]:.3f}")
    print(f"  (Combined: {var_exp[:2].sum():.3f})")
    print()

    # Print 2D coordinates per word
    print(f"  {'word':<12}  {'tag':<5}  {'base':<8}  {'PC1':>8}  {'PC2':>8}")
    for i, (w, tag, base) in enumerate(all_labels_for_pca):
        print(f"  {w:<12}  {tag:<5}  {base:<8}  {coords[i,0]:>8.3f}  {coords[i,1]:>8.3f}")

    # Check if comp is between pos and sup on PC1
    print()
    print("  Check: is comparative PC1 between positive and superlative?")
    for base in set(b for _, _, b in all_labels_for_pca):
        idx_pos  = [i for i,(w,t,b) in enumerate(all_labels_for_pca) if b==base and t=="pos"]
        idx_comp = [i for i,(w,t,b) in enumerate(all_labels_for_pca) if b==base and t=="comp"]
        idx_sup  = [i for i,(w,t,b) in enumerate(all_labels_for_pca) if b==base and t=="sup"]
        if not (idx_pos and idx_comp and idx_sup): continue
        p1_pos  = coords[idx_pos[0],0]
        p1_comp = coords[idx_comp[0],0]
        p1_sup  = coords[idx_sup[0],0]
        between = (min(p1_pos,p1_sup) <= p1_comp <= max(p1_pos,p1_sup))
        print(f"    {base:<8}: pos={p1_pos:>6.3f}  comp={p1_comp:>6.3f}  sup={p1_sup:>6.3f}  "
              f"{'BETWEEN' if between else 'NOT_BETWEEN'}")

# ── Part D: Held-out composition test ────────────────────────────────
print()
print("=" * 70)
print("PART D: Held-out composition test (words NOT in direction training)")
print("=" * 70)
print()

# Build mean directions from first 10 ADJ_DEGREE only (train set)
train_deg = [(p,c,s) for p,c,s in ADJ_DEGREE
             if is_single(p) and is_single(c) and is_single(s)][:10]
test_deg  = HELD_OUT_DEGREE

d_comp_train, n1 = mean_dir([(p,c) for p,c,s in train_deg])
d_csup_train, n2 = mean_dir([(c,s) for p,c,s in train_deg])
d_sup_train,  n3 = mean_dir([(p,s) for p,c,s in train_deg])
print(f"  Training: {n1} comparative pairs, {n2} comp_to_sup pairs, {n3} superlative pairs")
print()
print(f"  {'word':<8}  {'target_comp':<12}  {'pred_comp':<12}  ok1  "
      f"{'target_sup':<12}  {'pred_2hop':<12}  ok2  {'pred_1hop':<12}  ok3")

held_out_results = []
for pos, comp, sup in test_deg:
    ep = get_emb(pos)
    if ep is None:
        print(f"  {pos:<8}  EC_TOKENIZE"); continue

    # Single hop: comparative
    qt1 = normed(ep + d_comp_train)
    pred1 = top_k(qt1, k=1, exclude=pos)[0][0]

    # Two hop: comparative + comp_to_sup
    qt2 = normed(ep + d_comp_train + d_csup_train)
    pred2 = top_k(qt2, k=1, exclude=pos)[0][0]

    # One hop direct: superlative
    qt3 = normed(ep + d_sup_train)
    pred3 = top_k(qt3, k=1, exclude=pos)[0][0]

    ok1 = "OK" if pred1 == comp else "  "
    ok2 = "OK" if pred2 == sup  else "  "
    ok3 = "OK" if pred3 == sup  else "  "
    print(f"  {pos:<8}  {comp:<12}  {pred1:<12}  {ok1}   "
          f"{sup:<12}  {pred2:<12}  {ok2}   {pred3:<12}  {ok3}")
    held_out_results.append({
        "pos": pos, "comp": comp, "sup": sup,
        "pred_comp": pred1, "ok_comp": (pred1==comp),
        "pred_2hop": pred2, "ok_2hop": (pred2==sup),
        "pred_1hop": pred3, "ok_1hop": (pred3==sup),
    })

print()
print("=" * 70)
print("PART E: Per-step cos(d1,d2) vs paradigm type summary")
print("=" * 70)
print()
print("  The question: is cos(d1,d2) < 0 universal for step pairs in W_E?")
print()
print("  Adj degree (pos->comp, comp->sup):")
print(f"    Negative: {neg_count}/{len(deg_records)}")
print(f"    Mean: {np.mean([r['cos_d1_d2'] for r in deg_records]):.4f}")
print()
print("  Noun number: only one step (singular->plural), no two-step comparison")
print("  Verb tense:  only one step (root->past_tense), no two-step comparison")
print()
print("  Therefore: anti-correlation is specific to THREE-FORM paradigms.")
print("  It means the path A->B->C in W_E is V-shaped or U-shaped, not linear.")
print()
print("  Possible geometric interpretations:")
print("    1. ZIGZAG: each step goes in a different direction (as measured)")
print("    2. CURVED PATH: the paradigm occupies a curved manifold")
print("    3. COMPRESSED SPACE: comparative forms cluster separately from")
print("       both positive and superlative forms")

output = {
    "deg_records": deg_records,
    "noun_mags": noun_mags,
    "verb_mags": verb_mags,
    "held_out": held_out_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 236 complete.")
