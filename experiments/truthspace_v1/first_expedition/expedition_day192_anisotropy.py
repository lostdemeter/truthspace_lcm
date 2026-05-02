#!/usr/bin/env python3
"""
Day 192 — W_E Anisotropy

QUESTION: Is the W_E embedding space isotropic (uniform in all directions)
or does it have preferred axes of high token density?

An isotropic distribution of unit vectors on S^(H-1) has all singular values
of the data matrix equal. Anisotropy = some directions contain more
variance (more tokens aligned that way) than others.

EXPERIMENTS:
  1. Randomized SVD on W_E (top-50 singular values)
     Compare to expected spectrum for uniform random unit vectors.

  2. What do the top principal axes represent?
     Find the 20 nearest tokens to each of the top-5 PC directions.
     If the space is structured, top PCs should correspond to semantic clusters.

  3. Do relational directions (capital, gender, language) align with W_E PCs?
     Cosine of each relational mean-direction with each of the top-20 PCs.
     If relational directions are in the 'signal' subspace vs noise subspace.

  4. Center-subtracted (mean-removed) anisotropy
     Remove the global mean embedding, re-run SVD.
     Tests whether anisotropy is just a DC offset (all tokens shifted one way).
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.utils.extmath import randomized_svd

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day192_anisotropy.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAIN_PAIRS = {
    "capitals":  [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                  ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
                  ("Korea","Seoul"),("Poland","Warsaw")],
    "languages": [("France","French"),("Germany","German"),("Italy","Italian"),
                  ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
                  ("Russia","Russian"),("Greece","Greek")],
    "gender":    [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress")],
    "antonyms":  [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

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

# ── Experiment 1: Randomized SVD ────────────────────────────────────
print("Experiment 1: Randomized SVD on W_E (top-50)")
print("-" * 60)
n_components = 50
U, s, Vt = randomized_svd(W_E, n_components=n_components, random_state=42)

# Normalize by maximum to see relative magnitudes
s_rel = s / s[0]
total_var = np.sum(W_E**2)  # Frobenius^2 approx
var_s = s**2
var_pct = var_s / np.sum(var_s[:n_components]) * 100  # % of top-50 variance

print(f"  Top-20 relative singular values (s/s[0]):")
for i in range(20):
    bar = "#" * int(s_rel[i] * 30)
    print(f"    PC{i+1:>2}: {s_rel[i]:.4f}  {bar}")
print()

# Variance explained by top-k within the top-50
cumvar = np.cumsum(var_pct)
print(f"  Variance (within top-50 PCs) explained by top-k:")
for k in [1, 2, 3, 5, 10, 20, 50]:
    print(f"    k={k:>3}: {cumvar[k-1]:.2f}%")
print()

# Expected for random unit vectors on S^(H-1):
# Each PC explains 1/H of variance → all equal. s_rel ≈ const for all.
# Actual top-1 explains >> 1/H if anisotropic.
expected_flat = 100 / n_components  # if perfectly flat over top-50
print(f"  Expected if flat (uniform):  {expected_flat:.2f}% per PC")
print(f"  Actual PC1:                  {var_pct[0]:.2f}%  (ratio={var_pct[0]/expected_flat:.2f}x)")
print()

# ── Experiment 2: What are the top PCs? ─────────────────────────────
print("Experiment 2: Nearest tokens to top-5 PC directions")
print("-" * 60)
pc_tokens = {}
for pc_idx in range(5):
    direction = Vt[pc_idx]  # right singular vector = direction in H-space
    # Find tokens most aligned with this direction (positive and negative)
    cosines_pos = W_E @ direction / (np.linalg.norm(W_E, axis=1) + 1e-8)
    top_pos = np.argsort(cosines_pos)[-10:][::-1]
    top_neg = np.argsort(cosines_pos)[:10]

    pos_tokens = [(tok.convert_ids_to_tokens([i])[0], float(cosines_pos[i]))
                  for i in top_pos]
    neg_tokens = [(tok.convert_ids_to_tokens([i])[0], float(cosines_pos[i]))
                  for i in top_neg]

    print(f"  PC{pc_idx+1} (s_rel={s_rel[pc_idx]:.4f}):")
    print(f"    +pole: {', '.join(f'{t}({c:.3f})' for t,c in pos_tokens[:5])}")
    print(f"    -pole: {', '.join(f'{t}({c:.3f})' for t,c in neg_tokens[:5])}")
    pc_tokens[f"PC{pc_idx+1}"] = {"pos": pos_tokens, "neg": neg_tokens}
print()

# ── Experiment 3: Relational direction alignment with PCs ────────────
print("Experiment 3: Relational direction cosine with top-20 PCs")
print("-" * 60)
rel_directions = {}
for domain, pairs in DOMAIN_PAIRS.items():
    diffs = []
    for a, b in pairs:
        ta, tb = tid1(a), tid1(b)
        if ta and tb:
            d = W_E[tb] - W_E[ta]
            diffs.append(normed(d.astype(np.float64)))
    if diffs:
        rel_directions[domain] = normed(np.mean(diffs, axis=0))

print(f"  {'Domain':>12}  " + "  ".join(f"PC{i+1}" for i in range(10)))
print("  " + "-"*75)
rel_pc_align = {}
for domain, rel_dir in rel_directions.items():
    aligns = []
    for i in range(20):
        pc_dir = Vt[i].astype(np.float64)
        c = abs(cosine(rel_dir, pc_dir))
        aligns.append(c)
    top5_str = "  ".join(f"{aligns[i]:.3f}" for i in range(10))
    print(f"  {domain:>12}  {top5_str}")
    rel_pc_align[domain] = aligns
    best_pc = int(np.argmax(aligns)) + 1
    print(f"  {'':>12}  Best: PC{best_pc} ({max(aligns):.4f})")
print()

# ── Experiment 4: Mean-centered anisotropy ──────────────────────────
print("Experiment 4: Mean-centered W_E anisotropy")
print("-" * 60)
mean_emb = W_E.mean(axis=0)
print(f"  Mean embedding norm: {np.linalg.norm(mean_emb):.4f}")
W_center = W_E - mean_emb[None, :]
U2, s2, Vt2 = randomized_svd(W_center, n_components=n_components, random_state=42)
s2_rel = s2 / s2[0]
var2_pct = s2**2 / np.sum(s2**2) * 100
cum2 = np.cumsum(var2_pct)
print(f"  Top-10 relative singular values (CENTERED):")
for i in range(10):
    bar = "#" * int(s2_rel[i] * 30)
    print(f"    PC{i+1:>2}: {s2_rel[i]:.4f}  {bar}")
print(f"\n  Centered variance (top-50) explained by top-k:")
for k in [1, 2, 3, 5, 10, 20, 50]:
    print(f"    k={k:>3}: {cum2[k-1]:.2f}%")
print()

# Nearest tokens to centered PC1
direction2 = Vt2[0]
cosines2 = W_center @ direction2 / (np.linalg.norm(W_center, axis=1) + 1e-8)
top_pos2 = np.argsort(cosines2)[-10:][::-1]
top_neg2 = np.argsort(cosines2)[:10]
print(f"  Centered PC1 +pole: "
      f"{', '.join(tok.convert_ids_to_tokens([i])[0] for i in top_pos2[:8])}")
print(f"  Centered PC1 -pole: "
      f"{', '.join(tok.convert_ids_to_tokens([i])[0] for i in top_neg2[:8])}")

results = {
    "singular_values": s.tolist(),
    "s_rel": s_rel.tolist(),
    "var_pct": var_pct.tolist(),
    "pc_tokens": pc_tokens,
    "rel_pc_alignment": rel_pc_align,
    "centered": {
        "mean_emb_norm": float(np.linalg.norm(mean_emb)),
        "s_rel": s2_rel.tolist(),
        "var_pct": var2_pct.tolist(),
    },
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 192 complete.")
