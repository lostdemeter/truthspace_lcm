#!/usr/bin/env python3
"""
Day 237 — PCA Degree Dimension: Extraction and Retrieval Test

Day 236 showed:
  - PC1 of {pos,comp,sup} paradigm set orders all words correctly
  - Anti-correlation of per-word steps is universal (19/19)
  - Mean_dir generalizes to held-out words (3/3)
  - PC1 explains only 10.9% of variance (path is multi-dimensional)

Today:
  A. Extract the PCA degree dimension (PC1) and compare it to:
     - d_superlative: mean(normed(sup - pos))
     - d_comparative: mean(normed(comp - pos))
     - d_comp_to_sup: mean(normed(sup - comp))
     Are these all aligned with PC1? If so, PC1 IS the degree direction.

  B. Retrieval test: PCA direction vs mean_dir
     Does using PC1 as the direction improve or hurt retrieval accuracy
     for comparative and superlative tasks?

  C. What IS the degree dimension?
     Project all 151,936 W_E tokens onto PC1 of the degree paradigm.
     What tokens have extreme positive/negative projections?
     This reveals what the degree dimension "means" to the model.

  D. Multi-paradigm PCA:
     Extract PC1 for each of our known paradigms:
       - gender (king/queen, man/woman, ...)
       - plural (cat/cats, dog/dogs, ...)
       - past_tense (walk/walked, ...)
       - adj_degree (big/bigger/biggest, ...)
     Measure cos between each paradigm's PC1 and its mean_dir.
     Are these always aligned? Which paradigm shows most variance in PC1?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day237_degree_dim.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ADJ_DEGREE_TRAIN = [
    ("big","bigger","biggest"), ("fast","faster","fastest"),
    ("long","longer","longest"), ("small","smaller","smallest"),
    ("hard","harder","hardest"), ("bright","brighter","brightest"),
    ("dark","darker","darkest"), ("rich","richer","richest"),
    ("deep","deeper","deepest"), ("wide","wider","widest"),
]
ADJ_DEGREE_TEST = [
    ("high","higher","highest"), ("low","lower","lowest"),
    ("old","older","oldest"), ("young","younger","youngest"),
    ("hot","hotter","hottest"), ("tall","taller","tallest"),
    ("weak","weaker","weakest"), ("strong","stronger","strongest"),
    ("short","shorter","shortest"),
]

GENDER_PAIRS = [("king","queen"),("man","woman"),("boy","girl"),
                ("prince","princess"),("actor","actress"),("hero","heroine")]
PLURAL_PAIRS = [("cat","cats"),("dog","dogs"),("house","houses"),
                ("tree","trees"),("book","books"),("car","cars"),
                ("bird","birds"),("ship","ships"),("hand","hands")]
PAST_PAIRS   = [("walk","walked"),("talk","talked"),("call","called"),
                ("pull","pulled"),("fill","filled"),("turn","turned"),
                ("look","looked"),("move","moved"),("push","pushed")]

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
def ok_triples(triples):
    return [(p,c,s) for p,c,s in triples if is_single(p) and is_single(c) and is_single(s)]

print("Building pool ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))

all_words = set()
for trp in ADJ_DEGREE_TRAIN + ADJ_DEGREE_TEST:
    all_words.update(trp)
for p in GENDER_PAIRS + PLURAL_PAIRS + PAST_PAIRS:
    all_words.update(p)
for w in all_words:
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
    p = ok_pairs(pairs)
    if not p: return None, 0
    dirs = [normed(get_emb(b) - get_emb(a)) for a,b in p]
    return normed(np.mean(dirs, axis=0)), len(p)

def pca_direction(pairs_with_labels):
    """
    pairs_with_labels: list of (word, emb) tuples
    Returns PC1 direction from the mean-centered set.
    """
    X = np.array([e for _, e in pairs_with_labels], dtype=np.float64)
    X -= X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[0].astype(np.float64)

def pca_dir_degree():
    """PC1 of the adj degree training set {pos, comp, sup}."""
    items = []
    for pos, comp, sup in ADJ_DEGREE_TRAIN:
        for w in [pos, comp, sup]:
            e = get_emb(w)
            if e is not None: items.append((w, e))
    return pca_direction(items)

def pca_dir_pair(pairs):
    """PC1 of a set of word pairs."""
    items = []
    for a, b in ok_pairs(pairs):
        for w in [a, b]:
            e = get_emb(w)
            if e is not None: items.append((w, e))
    return pca_direction(items)

# ── Part A: PCA degree vs mean directions ─────────────────────────────
print("=" * 70)
print("PART A: PCA degree dimension vs mean directions")
print("=" * 70)
print()

triples_train = ok_triples(ADJ_DEGREE_TRAIN)

d_sup, _  = mean_dir([(p,s) for p,c,s in triples_train])
d_comp, _ = mean_dir([(p,c) for p,c,s in triples_train])
d_c2s, _  = mean_dir([(c,s) for p,c,s in triples_train])
d_pca     = pca_dir_degree()

# d_pca may point in either direction; align with d_sup
if cosine(d_pca, d_sup) < 0:
    d_pca = -d_pca
    print("  (Flipped PC1 to align with d_superlative)")

print(f"  cos(d_sup,   d_pca): {cosine(d_sup, d_pca):>+.4f}")
print(f"  cos(d_comp,  d_pca): {cosine(d_comp, d_pca):>+.4f}")
print(f"  cos(d_c2s,   d_pca): {cosine(d_c2s, d_pca):>+.4f}")
print(f"  cos(d_sup,   d_comp):{cosine(d_sup, d_comp):>+.4f}")
print(f"  cos(d_sup,   d_c2s): {cosine(d_sup, d_c2s):>+.4f}")
print(f"  cos(d_comp,  d_c2s): {cosine(d_comp, d_c2s):>+.4f}")
print()
print("  Interpretation:")
print("    If cos(d_pca, d_sup) >> cos(d_pca, d_comp): PCA is more superlative-aligned")
print("    If cos(d_pca, d_sup) ≈ cos(d_pca, d_comp): PCA is a midpoint direction")

# ── Part B: Retrieval test PCA vs mean_dir ────────────────────────────
print()
print("=" * 70)
print("PART B: Retrieval: PCA direction vs mean_dir on TEST triples")
print("=" * 70)
print()

triples_test = ok_triples(ADJ_DEGREE_TEST)

print(f"  {'method':<15}  {'comparative':>11}  {'superlative':>11}  {'2hop_sup':>9}")
print()

def retrieval_acc(triples, direction, label, comp=False, sup2=False, d2=None):
    ok = 0; total = 0
    for pos, comp_w, sup in triples:
        ep = get_emb(pos)
        if ep is None: continue
        total += 1
        target = comp_w if comp else sup
        if sup2 and d2 is not None:
            qt = normed(ep + direction + d2)
        else:
            qt = normed(ep + direction)
        pred = top_k(qt, k=1, exclude=pos)[0][0]
        if pred == target: ok += 1
    return ok, total

comp_mean_ok, comp_total = retrieval_acc(triples_test, d_comp, "mean_dir", comp=True)
sup_mean_ok,  sup_total  = retrieval_acc(triples_test, d_sup,  "mean_dir", comp=False)
s2h_mean_ok,  _          = retrieval_acc(triples_test, d_comp, "mean_dir",
                                          comp=False, sup2=True, d2=d_c2s)

comp_pca_ok, _  = retrieval_acc(triples_test, d_pca,  "pca", comp=True)
sup_pca_ok,  _  = retrieval_acc(triples_test, d_pca,  "pca", comp=False)
# For PCA 2-hop, use d_pca twice (since it's the degree direction)
s2h_pca_ok,  _  = retrieval_acc(triples_test, d_pca,  "pca",
                                 comp=False, sup2=True, d2=d_pca)

n = sup_total
print(f"  {'mean_dir':<15}  {comp_mean_ok}/{n}={comp_mean_ok/n:.3f}  "
      f"{sup_mean_ok}/{n}={sup_mean_ok/n:.3f}  {s2h_mean_ok}/{n}={s2h_mean_ok/n:.3f}")
print(f"  {'pca_PC1':<15}  {comp_pca_ok}/{n}={comp_pca_ok/n:.3f}  "
      f"{sup_pca_ok}/{n}={sup_pca_ok/n:.3f}  {s2h_pca_ok}/{n}={s2h_pca_ok/n:.3f}")

# Also test on training words (sanity check)
triples_trn = ok_triples(ADJ_DEGREE_TRAIN)
n_trn = len(triples_trn)
comp_m_trn, _ = retrieval_acc(triples_trn, d_comp, "mean_trn", comp=True)
sup_m_trn,  _ = retrieval_acc(triples_trn, d_sup,  "mean_trn", comp=False)
comp_p_trn, _ = retrieval_acc(triples_trn, d_pca,  "pca_trn",  comp=True)
sup_p_trn,  _ = retrieval_acc(triples_trn, d_pca,  "pca_trn",  comp=False)
print()
print(f"  [TRAIN] mean_dir:  comp={comp_m_trn}/{n_trn}  sup={sup_m_trn}/{n_trn}")
print(f"  [TRAIN] pca_PC1:   comp={comp_p_trn}/{n_trn}  sup={sup_p_trn}/{n_trn}")

# ── Part C: What IS the degree dimension? ────────────────────────────
print()
print("=" * 70)
print("PART C: What tokens have extreme projections onto the degree dimension?")
print("=" * 70)
print()

# Project all pool tokens
projs = E_normed.astype(np.float64) @ normed(d_pca)
order_pos = np.argsort(-projs)
order_neg = np.argsort(projs)

print(f"  Top 30 POSITIVE projections (most 'superlative-like'):")
print("  " + ", ".join(f"{pool_words[i]}({projs[i]:+.3f})" for i in order_pos[:30]))
print()
print(f"  Top 30 NEGATIVE projections (most 'positive-like'):")
print("  " + ", ".join(f"{pool_words[i]}({projs[i]:+.3f})" for i in order_neg[:30]))
print()

# Where do our training triples fall?
print("  Training word projections onto PC1:")
print(f"  {'word':<10}  {'pos_proj':>9}  {'comp_proj':>10}  {'sup_proj':>9}  ordering")
for pos, comp, sup in triples_train:
    ep, ec, es = get_emb(pos), get_emb(comp), get_emb(sup)
    if ep is None or ec is None or es is None: continue
    pp = float(normed(ep) @ normed(d_pca))
    cp = float(normed(ec) @ normed(d_pca))
    sp = float(normed(es) @ normed(d_pca))
    order_str = "pos<comp<sup" if pp < cp < sp else \
                "sup<comp<pos" if pp > cp > sp else "UNORDERED"
    print(f"  {pos:<10}  {pp:>9.4f}  {cp:>10.4f}  {sp:>9.4f}  {order_str}")

# ── Part D: Multi-paradigm PCA ────────────────────────────────────────
print()
print("=" * 70)
print("PART D: Multi-paradigm PCA — PC1 vs mean_dir for each paradigm")
print("=" * 70)
print()

paradigms = {
    "gender":   (GENDER_PAIRS, False),
    "plural":   (PLURAL_PAIRS, False),
    "past_tense": (PAST_PAIRS, False),
    "adj_degree": ([(p,s) for p,c,s in triples_train], False),
}
print(f"  {'paradigm':<14}  {'cos(PC1,mean_dir)':>18}  {'var_PC1':>8}  {'n':>3}")
pca_results = {}
for pname, (pairs, _) in paradigms.items():
    p = ok_pairs(pairs)
    if not p:
        print(f"  {pname:<14}  NO_PAIRS"); continue
    d_mean, n = mean_dir(p)
    d_pca_p = pca_dir_pair(p)
    # Align
    if cosine(d_pca_p, d_mean) < 0:
        d_pca_p = -d_pca_p
    align = cosine(d_pca_p, d_mean)

    # Variance explained by PC1
    items = []
    for a, b in p:
        for w in [a, b]:
            e = get_emb(w)
            if e is not None: items.append(e)
    X = np.array(items, dtype=np.float64)
    X -= X.mean(axis=0)
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    var_pc1 = float(S[0]**2 / (S**2).sum())

    print(f"  {pname:<14}  cos={align:>+.4f}           {var_pc1:>8.3f}  {n:>3}")
    pca_results[pname] = {"cos_pca_mean": align, "var_pc1": var_pc1, "n": n}

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  PCA degree dimension (PC1):")
print(f"    cos(PC1, d_superlative): {cosine(d_pca, d_sup):>+.4f}")
print(f"    cos(PC1, d_comparative): {cosine(d_pca, d_comp):>+.4f}")
print()
print("  Retrieval: PCA vs mean_dir")
print(f"    mean_dir comp acc: {comp_mean_ok}/{n}  pca comp acc: {comp_pca_ok}/{n}")
print(f"    mean_dir sup  acc: {sup_mean_ok}/{n}  pca sup  acc: {sup_pca_ok}/{n}")
print()
print("  Key question answered: does PCA direction add information over mean_dir?")

output = {
    "cos_alignments": {
        "sup_vs_pca": cosine(d_sup, d_pca),
        "comp_vs_pca": cosine(d_comp, d_pca),
        "c2s_vs_pca": cosine(d_c2s, d_pca),
    },
    "retrieval_test": {
        "mean_dir_comp": comp_mean_ok, "mean_dir_sup": sup_mean_ok,
        "pca_comp": comp_pca_ok, "pca_sup": sup_pca_ok,
        "n_test": int(n),
    },
    "pca_paradigms": pca_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 237 complete.")
