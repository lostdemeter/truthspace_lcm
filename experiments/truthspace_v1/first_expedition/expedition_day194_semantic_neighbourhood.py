#!/usr/bin/env python3
"""
Day 194 — W_E Semantic Neighbourhood Structure

QUESTION: Within the semantic subspace of W_E (orthogonal to script axes),
do semantically related words form tight clusters? How does cluster diameter
compare to the relational step size (mean direction magnitude)?

EXPERIMENTS:
  1. Intra-cluster cosine similarity for semantic categories
     - Countries, capital cities, color words, number words, animals,
       body parts, common verbs, common adjectives
     - Measure mean pairwise cosine within each category

  2. Inter-cluster distance
     - Mean cosine between categories
     - Is there semantic separation between categories in W_E?

  3. Cluster diameter vs relational step size
     - For country→capital, how large is the "country cluster"?
     - How large is the "capital cluster"?
     - How large is the mean relational direction?
     - Is the step comparable to, larger, or smaller than cluster diameter?

  4. Nearest neighbours from vocabulary
     - For a sample word, who are its 10 nearest neighbours in W_E?
     - Are they semantically related?
     - Is the nearest neighbour structure meaningful?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day194_semantic_neighbourhood.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

CATEGORIES = {
    "countries":  ["France","Germany","Italy","Spain","Japan","China","Russia",
                   "Greece","Sweden","Poland","Korea","Brazil","Canada","India",
                   "Turkey","Egypt","Australia","Mexico","Argentina"],
    "capitals":   ["Paris","Berlin","Rome","Madrid","Tokyo","Beijing","Moscow",
                   "Athens","Stockholm","Warsaw","Seoul","Ottawa","Delhi",
                   "Ankara","Cairo","Canberra"],
    "colors":     ["red","blue","green","yellow","white","black","brown","pink",
                   "purple","orange","gray","silver","gold","violet","cyan"],
    "numbers":    ["one","two","three","four","five","six","seven","eight",
                   "nine","ten","eleven","twelve","twenty","hundred","thousand"],
    "animals":    ["dog","cat","horse","bird","fish","lion","tiger","bear",
                   "wolf","fox","rabbit","mouse","cow","pig","sheep","deer"],
    "body_parts": ["hand","head","face","eye","ear","nose","mouth","arm","leg",
                   "foot","back","heart","brain","blood","bone","skin"],
    "common_verbs":["run","walk","eat","sleep","speak","think","know","feel",
                   "see","hear","say","give","take","make","come","go"],
    "adjectives": ["big","small","fast","slow","hot","cold","old","new","good",
                   "bad","hard","soft","light","dark","long","short"],
    "relations_src": ["France","Germany","Italy","Spain","Japan","China",
                      "Russia","Greece","Sweden","Korea"],
    "relations_tgt": ["Paris","Berlin","Rome","Madrid","Tokyo","Beijing",
                      "Moscow","Athens","Stockholm","Seoul"],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a)), normed(np.array(b))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def tid1(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_emb(word):
    t = tid1(word)
    return W_E[t] if t is not None else None

# ── Experiment 1 & 2: Intra- and inter-cluster cosines ──────────────
print("Experiment 1: Intra-cluster cosine similarity")
print("-" * 60)
cat_embs = {}
cat_stats = {}
for cat, words in CATEGORIES.items():
    embs = [(w, get_emb(w)) for w in words]
    embs = [(w, e) for w, e in embs if e is not None]
    cat_embs[cat] = embs
    if len(embs) < 2: continue
    vecs = np.array([e for _, e in embs])
    # All pairwise cosines
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed_vecs = vecs / (norms + 1e-8)
    cos_matrix = normed_vecs @ normed_vecs.T
    n = len(embs)
    # Off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    pairwise = cos_matrix[mask]
    cat_stats[cat] = {
        "n": n,
        "mean_cos": float(np.mean(pairwise)),
        "std_cos":  float(np.std(pairwise)),
        "min_cos":  float(np.min(pairwise)),
        "max_cos":  float(np.max(pairwise)),
        "centroid_norm": float(np.linalg.norm(np.mean(normed_vecs, axis=0))),
    }
    print(f"  {cat:<16} n={n:<3} mean_cos={np.mean(pairwise):.4f}  "
          f"std={np.std(pairwise):.4f}  "
          f"centroid_norm={np.linalg.norm(np.mean(normed_vecs,axis=0)):.4f}")
print()

print("Experiment 2: Inter-cluster cosine (centroid distances)")
print("-" * 60)
centroids = {}
for cat, embs in cat_embs.items():
    if not embs: continue
    vecs = np.array([e for _, e in embs])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed_vecs = vecs / (norms + 1e-8)
    centroids[cat] = normed(np.mean(normed_vecs, axis=0))

cat_list = [c for c in CATEGORIES.keys() if c in centroids]
print(f"  {'':>16}  " + "  ".join(f"{c[:8]}" for c in cat_list[:8]))
inter_cos = {}
for i, c1 in enumerate(cat_list):
    row = []
    for c2 in cat_list:
        c = cosine(centroids[c1], centroids[c2])
        row.append(c)
    inter_cos[c1] = {c2: row[j] for j, c2 in enumerate(cat_list)}
    print(f"  {c1:<16}  " + "  ".join(f"{row[j]:.4f}" for j in range(min(8,len(row)))))
print()

# ── Experiment 3: Cluster diameter vs relational step ────────────────
print("Experiment 3: Cluster diameter vs relational step size")
print("-" * 60)

# Country cluster diameter = max pairwise distance
src_words = [w for w, e in cat_embs["relations_src"]]
tgt_words = [w for w, e in cat_embs["relations_tgt"]]
src_vecs = np.array([e for _, e in cat_embs["relations_src"]])
tgt_vecs = np.array([e for _, e in cat_embs["relations_tgt"]])

def cluster_diameter(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    nv = vecs / (norms + 1e-8)
    cos_m = nv @ nv.T
    n = len(vecs)
    mask = ~np.eye(n, dtype=bool)
    pairwise = cos_m[mask]
    return float(np.mean(pairwise)), float(np.min(pairwise))

src_mean, src_min = cluster_diameter(src_vecs)
tgt_mean, tgt_min = cluster_diameter(tgt_vecs)

# Relational step sizes
steps = []
for (_, se), (_, te) in zip(cat_embs["relations_src"], cat_embs["relations_tgt"]):
    d = normed(te - se)
    steps.append(d)
mean_step = normed(np.mean(steps, axis=0))
step_magnitudes = [np.linalg.norm(te - se)
                   for (_, se), (_, te) in
                   zip(cat_embs["relations_src"], cat_embs["relations_tgt"])]

print(f"  Country cluster (src):")
print(f"    Mean pairwise cos:  {src_mean:.4f}")
print(f"    Min pairwise cos:   {src_min:.4f}")
print(f"  Capital cluster (tgt):")
print(f"    Mean pairwise cos:  {tgt_mean:.4f}")
print(f"    Min pairwise cos:   {tgt_min:.4f}")
print(f"  Relational step (country→capital):")
print(f"    Mean step L2 magnitude:  {np.mean(step_magnitudes):.4f}")
print(f"    Std step magnitude:      {np.std(step_magnitudes):.4f}")
print(f"    Mean step / emb norm:    "
      f"{np.mean(step_magnitudes) / 0.65:.4f}  (emb norm ≈ 0.65)")
print()

# Cross-cluster: source centroid vs target centroid
src_centroid = normed(np.mean(src_vecs / (np.linalg.norm(src_vecs,axis=1,keepdims=True)+1e-8), axis=0))
tgt_centroid = normed(np.mean(tgt_vecs / (np.linalg.norm(tgt_vecs,axis=1,keepdims=True)+1e-8), axis=0))
centroid_cos = cosine(src_centroid, tgt_centroid)
print(f"  Centroid(countries) vs Centroid(capitals): cos = {centroid_cos:.4f}")
print()

# ── Experiment 4: Nearest neighbours ────────────────────────────────
print("Experiment 4: Nearest neighbours in W_E")
print("-" * 60)
probe_words = ["France","Paris","king","red","dog","run","big","water"]
for word in probe_words:
    e = get_emb(word)
    if e is None: continue
    e_n = normed(e)
    sims = W_E @ e_n / (np.linalg.norm(W_E, axis=1) + 1e-8)
    top10 = np.argsort(sims)[-11:][::-1]
    nn = [tok.convert_ids_to_tokens([i])[0] for i in top10 if i != tid1(word)][:8]
    print(f"  {word:<10} → {nn}")
print()

results = {
    "intra_cluster": cat_stats,
    "inter_cluster_centroids": inter_cos,
    "relational_step": {
        "src_cluster_mean_cos": src_mean,
        "tgt_cluster_mean_cos": tgt_mean,
        "mean_step_magnitude": float(np.mean(step_magnitudes)),
        "std_step_magnitude": float(np.std(step_magnitudes)),
        "centroid_to_centroid_cos": centroid_cos,
    },
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved: {OUTPUT_FILE}")
print("Day 194 complete.")
