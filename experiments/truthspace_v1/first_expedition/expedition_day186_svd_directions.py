#!/usr/bin/env python3
"""
Day 186 — SVD on Difference-Vector Matrices: Principal Relational Directions

QUESTION: The 1536-dimensional W_E space contains relational directions.
How many independent relational directions are there?
Can we identify a basis of principal relational directions via SVD?

METHOD:
  For each TYPE_BC domain, build matrix D where row_i = normed(W_E[tgt_i] - W_E[src_i]).
  Run SVD on D → singular values tell us effective dimensionality.
  Top singular vectors (right singular vectors V^T) are the principal directions.

  Then: are principal directions from DIFFERENT domains orthogonal?
  Do related domains (country→capital vs country→language) share principal directions?

ACROSS-DOMAIN ANALYSIS:
  Stack all TYPE_BC difference vectors into one matrix D_all.
  SVD of D_all → how many independent relational directions exist in W_E?
  Plot singular value decay to estimate effective dimensionality.

DOMAINS: capitals, languages, gender, country→currency (discovered TYPE_BC in Day 182)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day186_svd_directions.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAINS = {
    "capitals": [
        ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
        ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
        ("Korea","Seoul"),("Poland","Warsaw"),("Brazil","Brasilia"),
        ("Canada","Ottawa"),("India","Delhi"),("Turkey","Ankara"),
    ],
    "languages": [
        ("France","French"),("Germany","German"),("Italy","Italian"),
        ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
        ("Russia","Russian"),("Greece","Greek"),("Sweden","Swedish"),
        ("Korea","Korean"),("Poland","Polish"),("Turkey","Turkish"),
    ],
    "gender": [
        ("king","queen"),("man","woman"),("boy","girl"),
        ("prince","princess"),("lord","lady"),("actor","actress"),
        ("hero","heroine"),("duke","duchess"),("monk","nun"),
    ],
    "country_currency": [
        ("France","euro"),("Germany","euro"),("Italy","euro"),
        ("Spain","euro"),("Japan","yen"),("China","yuan"),
        ("Russia","ruble"),("India","rupee"),("Korea","won"),
        ("Sweden","krona"),("Poland","zloty"),("Turkey","lira"),
    ],
    "antonyms": [
        ("hot","cold"),("big","small"),("fast","slow"),
        ("hard","soft"),("light","dark"),("old","young"),
        ("loud","quiet"),("rich","poor"),("strong","weak"),
    ],
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
print(f"  H={W_E.shape[1]}\n")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def build_diff_matrix(pairs):
    diffs = []
    ok_pairs = []
    for a, b in pairs:
        ta, tb = tid(a), tid(b)
        if ta and tb:
            d = W_E[tb] - W_E[ta]
            diffs.append(normed(d.astype(np.float64)))
            ok_pairs.append((a, b))
    return np.array(diffs, dtype=np.float64), ok_pairs

results = {}
domain_svecs = {}   # top-5 right singular vectors per domain

print(f"{'Domain':>20}  {'n':>4}  {'sv1':>8}  {'sv2':>8}  {'sv3':>8}  "
      f"{'sv4':>8}  {'sv5':>8}  {'eff_dim':>8}")
print("-"*80)

for name, pairs in DOMAINS.items():
    D, ok = build_diff_matrix(pairs)
    if len(D) < 3: continue
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    s_norm = s / s[0]  # normalize by top singular value

    # Effective dimensionality: number of singular values > 0.1 * s[0]
    eff_dim = int(np.sum(s_norm > 0.10))
    top5 = s_norm[:5]
    print(f"  {name:>20}  {len(D):>4}  "
          + "  ".join(f"{v:>8.4f}" for v in top5)
          + f"  {eff_dim:>8}")

    domain_svecs[name] = {"Vt": Vt[:5].tolist(), "s": s[:10].tolist(),
                           "n": len(D), "eff_dim": eff_dim}

print()
print("─" * 80)
print("Cross-domain principal direction alignment (cosine of top-1 singular vectors):")
print()

domain_names = list(domain_svecs.keys())
print(f"{'':>20}", end="")
for n in domain_names:
    print(f"  {n[:10]:>10}", end="")
print()

for n1 in domain_names:
    v1 = np.array(domain_svecs[n1]["Vt"][0])
    print(f"  {n1:>20}", end="")
    for n2 in domain_names:
        v2 = np.array(domain_svecs[n2]["Vt"][0])
        c = abs(cosine(v1, v2))
        print(f"  {c:>10.4f}", end="")
    print()

print()
print("─" * 80)
print("Combined SVD across all TYPE_BC domains:")
all_bc_diffs = []
for name in ["capitals", "languages", "gender", "country_currency"]:
    D, _ = build_diff_matrix(DOMAINS[name])
    all_bc_diffs.extend(D.tolist())

D_all = np.array(all_bc_diffs, dtype=np.float64)
U_all, s_all, Vt_all = np.linalg.svd(D_all, full_matrices=False)
s_all_norm = s_all / s_all[0]
eff_dim_all = int(np.sum(s_all_norm > 0.10))
print(f"  n={len(D_all)}, eff_dim(>0.10)={eff_dim_all}")
print(f"  Top-20 normalised singular values:")
print("  " + "  ".join(f"{v:.4f}" for v in s_all_norm[:20]))

# ─── How much variance does the top-1 direction capture? ───────────────────
var_explained = (s_all**2) / np.sum(s_all**2)
print(f"\n  Variance explained by top-k directions (cumulative):")
cum = np.cumsum(var_explained)
for k in [1, 2, 3, 5, 10, 20]:
    print(f"    k={k:>3}: {cum[k-1]:.4f}")

# Test: does the top-1 combined direction work as a universal relational direction?
# Try using it for LOO on capitals:
print()
print("─" * 80)
print("Top-1 combined direction applied to each domain (LOO accuracy):")
top1_dir = normed(Vt_all[0].astype(np.float32))

for name in ["capitals", "languages", "gender", "country_currency", "antonyms"]:
    D, ok = build_diff_matrix(DOMAINS[name])
    if len(ok) < 3: continue
    tgt_vocab = {b: W_E[tid(b)] for _, b in ok}
    nc = 0
    for a, b in ok:
        q = W_E[tid(a)] + top1_dir
        cands = {w: cosine(q, tgt_vocab[w]) for w in tgt_vocab if w != a}
        if cands and max(cands, key=lambda w: cands[w]) == b:
            nc += 1
    acc = nc / len(ok)
    print(f"  {name:>20}: {acc:.3f}")

results = {
    "per_domain": domain_svecs,
    "combined_bc": {
        "n": len(D_all),
        "eff_dim": eff_dim_all,
        "s_norm": s_all_norm[:20].tolist(),
        "var_explained_top10": cum[9].tolist(),
        "top1_dir": top1_dir.tolist(),
    }
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 186 complete.")
