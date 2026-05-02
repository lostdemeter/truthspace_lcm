#!/usr/bin/env python3
"""
Day 242 — Token Cluster Analysis: What Families Live Near Each Archetype?

DC 384 found: superlatives form a GEOMETRICALLY ISOLATED CLUSTER (overlap=0.000).
This raises: what exactly lives in each archetype region?

Questions:
  A. Centroid proximity: who are the k=50 nearest neighbours to
     the centroid of each paradigm's TARGET word set?
     - Superlative centroid: what surrounds it?
     - Comparative centroid: what surrounds it?
     - Past-tense centroid: what surrounds it?
     - Plural centroid? Gender target centroid?

  B. Morphological purity of each centroid neighbourhood:
     Among the top-50 neighbours of each target centroid, what fraction
     are members of the same paradigm family?
     e.g., nearest to superlative centroid: are they all -est words?

  C. Cross-contamination:
     Do comparative forms appear near the superlative centroid?
     Do superlative forms appear near the comparative centroid?
     Do past-tense forms contaminate the plural neighbourhood?

  D. Source-centroid neighbourhood:
     What lives near the centroid of SOURCE words (base adjectives)?
     Does it contain function words (as suggested by Part C of Day 237)?

  E. Paradigm cluster statistics:
     - Intra-cluster cosine similarity (how tight is each cluster?)
     - Distance between TARGET and SOURCE centroids
     - Radius of each cluster (max cosine spread)
"""
import json
import re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day242_clusters.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PARADIGM_PAIRS = {
    "adj_sup":  [("big","biggest"),("fast","fastest"),("long","longest"),
                 ("small","smallest"),("hard","hardest"),("bright","brightest"),
                 ("dark","darkest"),("rich","richest"),("deep","deepest"),
                 ("wide","widest"),("high","highest"),("low","lowest"),
                 ("old","oldest"),("young","youngest"),("hot","hottest"),
                 ("tall","tallest"),("strong","strongest"),("weak","weakest"),
                 ("short","shortest"),("cool","coolest"),("great","greatest"),
                 ("safe","safest"),("cheap","cheapest"),("clean","cleanest")],
    "adj_comp": [("big","bigger"),("fast","faster"),("long","longer"),
                 ("small","smaller"),("hard","harder"),("bright","brighter"),
                 ("dark","darker"),("rich","richer"),("deep","deeper"),
                 ("wide","wider"),("high","higher"),("low","lower"),
                 ("old","older"),("young","younger"),("hot","hotter"),
                 ("tall","taller"),("strong","stronger"),("weak","weaker"),
                 ("short","shorter"),("cool","cooler"),("great","greater"),
                 ("safe","safer"),("cheap","cheaper"),("clean","cleaner")],
    "past_tense": [("walk","walked"),("talk","talked"),("call","called"),
                   ("pull","pulled"),("fill","filled"),("turn","turned"),
                   ("look","looked"),("move","moved"),("push","pushed"),
                   ("help","helped"),("play","played"),("stay","stayed"),
                   ("lock","locked"),("jump","jumped"),("land","landed"),
                   ("ask","asked"),("work","worked"),("open","opened"),
                   ("rain","rained"),("join","joined"),("pass","passed")],
    "plural":  [("cat","cats"),("dog","dogs"),("house","houses"),
                ("tree","trees"),("book","books"),("car","cars"),
                ("bird","birds"),("ship","ships"),("hand","hands"),
                ("door","doors"),("lamp","lamps"),("wall","walls"),
                ("king","kings"),("boy","boys"),("word","words"),
                ("stone","stones"),("cloud","clouds"),("road","roads")],
    "gender":  [("king","queen"),("man","woman"),("boy","girl"),
                ("prince","princess"),("actor","actress"),("hero","heroine"),
                ("monk","nun"),("duke","duchess"),("lord","lady"),
                ("wizard","witch"),("nephew","niece")],
    "capital": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                ("India","Delhi"),("Russia","Moscow"),("Greece","Athens"),
                ("Egypt","Cairo"),("Mexico","Mexico"),("Poland","Warsaw")],
}

# Morphological suffix detectors
def is_superlative(w):
    return w.endswith("est") and len(w) > 4
def is_comparative(w):
    return w.endswith("er") and len(w) > 3 and not w.endswith("ster")
def is_past_tense(w):
    return w.endswith("ed") and len(w) > 3
def is_plural(w):
    return w.endswith("s") and not w.endswith("ss") and len(w) > 2
def is_base_adjective(w):
    return (not is_superlative(w) and not is_comparative(w) and
            not is_past_tense(w) and not is_plural(w) and w.islower())

def morph_tag(w):
    if is_superlative(w): return "sup"
    if is_comparative(w): return "comp"
    if is_past_tense(w):  return "past"
    if is_plural(w):      return "plural"
    return "base"

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b):
    return float(np.dot(normed(np.array(a,dtype=np.float64)),
                        normed(np.array(b,dtype=np.float64))))

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
def ok_pairs(pairs): return [(a,b) for a,b in pairs if is_single(a) and is_single(b)]

print("Building pool ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))

for pairs in PARADIGM_PAIRS.values():
    for a, b in pairs:
        for w in [a, b]:
            if w not in pool_words:
                e = get_emb(w)
                if e is not None:
                    pool_words.append(w); pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E  = np.array(pool_embs, dtype=np.float32)
nv = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
En = (E / nv).astype(np.float32)
print(f"  Pool: {N} tokens\n")

def top_k_cos(centroid, k=60, exclude=None):
    qn = normed(centroid).astype(np.float32)
    sims = En @ qn
    order = np.argsort(-sims)
    out = []
    for idx in order:
        w = pool_words[idx]
        if exclude and w in exclude: continue
        out.append((w, float(sims[idx])))
        if len(out) >= k: break
    return out

def centroid_of(words):
    embs = [get_emb(w) for w in words if is_single(w)]
    if not embs: return None
    return np.mean(embs, axis=0)

# ── Part A+B: Centroid neighbourhoods and morphological purity ────────
print("=" * 70)
print("PART A+B: Centroid neighbourhoods and morphological purity")
print("=" * 70)
print()

cluster_results = {}
K_NBR = 60

for pname, pairs in PARADIGM_PAIRS.items():
    p = ok_pairs(pairs)
    if not p: continue
    srcs = [a for a,b in p]
    tgts = [b for a,b in p]
    cen_src = centroid_of(srcs)
    cen_tgt = centroid_of(tgts)

    src_set = set(srcs)
    tgt_set = set(tgts)
    all_words_set = src_set | tgt_set

    # Source centroid neighbourhood
    nbr_src = top_k_cos(cen_src, k=K_NBR, exclude=all_words_set)
    # Target centroid neighbourhood
    nbr_tgt = top_k_cos(cen_tgt, k=K_NBR, exclude=all_words_set)

    # Compute intra-cluster similarity (mean pairwise cosine of target embs)
    tgt_embs = np.array([normed(get_emb(w)) for w in tgts if is_single(w)], dtype=np.float64)
    if len(tgt_embs) > 1:
        G = tgt_embs @ tgt_embs.T
        n_tgt = len(tgt_embs)
        intra_sim = float((G.sum() - n_tgt) / (n_tgt * (n_tgt - 1)))
    else:
        intra_sim = 1.0

    # Centroid-to-centroid cosine
    c2c = cosine(cen_src, cen_tgt) if cen_src is not None and cen_tgt is not None else 0.0

    print(f"  ── {pname} ──  n={len(p)}  intra_sim={intra_sim:.3f}  "
          f"cen_src↔cen_tgt cos={c2c:.3f}")

    # Morphological breakdown of TARGET neighbourhood
    print(f"    TARGET centroid top-20 (excluding training words):")
    tgt_morph = {"sup":0,"comp":0,"past":0,"plural":0,"base":0}
    for w, sim in nbr_tgt[:20]:
        tag = morph_tag(w)
        tgt_morph[tag] = tgt_morph.get(tag, 0) + 1
    top20_tgt = [(w,f"{s:.3f}") for w,s in nbr_tgt[:20]]
    print("    " + "  ".join(f"{w}({s})" for w, s in top20_tgt[:10]))
    print("    " + "  ".join(f"{w}({s})" for w, s in top20_tgt[10:]))
    print(f"    Morph breakdown: " + "  ".join(f"{k}={v}" for k,v in tgt_morph.items()))

    # SOURCE centroid top-10
    print(f"    SOURCE centroid top-10:")
    top10_src = [(w,f"{s:.3f}") for w,s in nbr_src[:10]]
    print("    " + "  ".join(f"{w}({s})" for w, s in top10_src))
    print()

    cluster_results[pname] = {
        "n": len(p), "intra_sim": intra_sim, "cen_cos": c2c,
        "nbr_tgt_top20": [w for w,_ in nbr_tgt[:20]],
        "nbr_src_top10": [w for w,_ in nbr_src[:10]],
        "tgt_morph": tgt_morph,
    }

# ── Part C: Cross-contamination matrix ───────────────────────────────
print("=" * 70)
print("PART C: Cross-contamination — how many TARGET nbrs belong to each paradigm?")
print("=" * 70)
print()
print("  Row = paradigm, columns = fraction of top-60 TARGET neighbours that")
print("  are morphologically consistent with each column paradigm's TARGET forms")
print()

for pname, res in cluster_results.items():
    pairs = PARADIGM_PAIRS[pname]
    p = ok_pairs(pairs)
    cen_tgt = centroid_of([b for a,b in p])
    if cen_tgt is None: continue
    nbr60 = top_k_cos(cen_tgt, k=60, exclude=set(a for a,b in p) | set(b for a,b in p))
    nbr_words = [w for w,_ in nbr60]

    # Classify each neighbour
    sup_n  = sum(1 for w in nbr_words if is_superlative(w))
    comp_n = sum(1 for w in nbr_words if is_comparative(w))
    past_n = sum(1 for w in nbr_words if is_past_tense(w))
    plu_n  = sum(1 for w in nbr_words if is_plural(w))
    base_n = sum(1 for w in nbr_words
                 if not is_superlative(w) and not is_comparative(w)
                 and not is_past_tense(w) and not is_plural(w))
    n = len(nbr_words)
    print(f"  {pname:<12}  sup={sup_n/n:.2f}  comp={comp_n/n:.2f}  "
          f"past={past_n/n:.2f}  plu={plu_n/n:.2f}  base={base_n/n:.2f}")

# ── Part D: Source centroid analysis ─────────────────────────────────
print()
print("=" * 70)
print("PART D: Source centroid analysis — what lives near base-form centroids?")
print("=" * 70)
print()

adj_base_words = list({a for a,b in PARADIGM_PAIRS["adj_sup"]} |
                      {a for a,b in PARADIGM_PAIRS["adj_comp"]})
cen_adj_base = centroid_of(adj_base_words)
nbr_base = top_k_cos(cen_adj_base, k=40, exclude=set(adj_base_words))
print(f"  BASE ADJECTIVE centroid top-40 neighbours:")
print("  " + "  ".join(f"{w}({s:.3f})" for w, s in nbr_base[:20]))
print("  " + "  ".join(f"{w}({s:.3f})" for w, s in nbr_base[20:40]))
print()
base_sup  = sum(1 for w,_ in nbr_base if is_superlative(w))
base_comp = sum(1 for w,_ in nbr_base if is_comparative(w))
base_fn   = sum(1 for w,_ in nbr_base
               if w in {"and","or","the","a","in","on","of","to","for",
                        "is","as","at","by","an","not","with","it","that"})
print(f"  Morph breakdown: sup={base_sup}/{len(nbr_base)}  "
      f"comp={base_comp}/{len(nbr_base)}  function_words={base_fn}/{len(nbr_base)}")

# ── Part E: Cluster statistics ────────────────────────────────────────
print()
print("=" * 70)
print("PART E: Cluster statistics — tightness of each paradigm's TARGET cluster")
print("=" * 70)
print()
print(f"  {'paradigm':<12}  {'n':>3}  {'intra_sim':>10}  {'cen↔cen':>9}  "
      f"{'radius_src':>10}  {'radius_tgt':>10}")

for pname, pairs in PARADIGM_PAIRS.items():
    p = ok_pairs(pairs)
    if not p: continue
    srcs = [a for a,b in p]
    tgts = [b for a,b in p]
    cen_src = centroid_of(srcs)
    cen_tgt = centroid_of(tgts)
    if cen_src is None or cen_tgt is None: continue

    # Radius = mean cosine of members to their centroid
    src_embs = np.array([normed(get_emb(w)) for w in srcs if is_single(w)], dtype=np.float64)
    tgt_embs = np.array([normed(get_emb(w)) for w in tgts if is_single(w)], dtype=np.float64)
    r_src = float(np.mean(src_embs @ normed(cen_src)))
    r_tgt = float(np.mean(tgt_embs @ normed(cen_tgt)))

    c2c = cosine(cen_src, cen_tgt)
    intra = cluster_results[pname]["intra_sim"] if pname in cluster_results else float("nan")
    print(f"  {pname:<12}  {len(p):>3}  {intra:>10.3f}  {c2c:>9.3f}  "
          f"{r_src:>10.3f}  {r_tgt:>10.3f}")

print()
print("  Interpretation:")
print("    intra_sim: mean pairwise cosine within target cluster (1=identical)")
print("    cen↔cen:   cosine between source and target centroids")
print("    radius:    mean cos of cluster members to their centroid (1=tight)")

output = {"cluster_results": {k: {kk: vv for kk, vv in v.items()
                                   if not isinstance(vv, np.ndarray)}
                               for k, v in cluster_results.items()}}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 242 complete.")
