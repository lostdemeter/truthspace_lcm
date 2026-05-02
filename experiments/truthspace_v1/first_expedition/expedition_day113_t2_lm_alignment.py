#!/usr/bin/env python3
"""
Day 113 — T2-to-LM-Internal Alignment

Days 70-112 built the φ-trie using T2 axes derived from contrast sentence pairs.
Days 1-70 (model reverse engineering) characterized the LM's ACTUAL internal
geometric structure:
  - H6 L23 geometric selector: d_k = W_k^T @ v₁ (Finding 40)
  - M_h Lens: near-isometric projection through ~66D aperture (Finding 124)
  - Spectrometer per-dimension rules (Findings 1-38)

KEY QUESTION:
Do the 12 T2 trie axes align with the LM's actual internal direction vectors?

If YES → the trie is discovering the LM's own coordinate system
          (strong validation of TruthSpace hypothesis)
If NO  → the T2 axes are external semantic coordinates, not LM-internal
          (trie is a parallel geometry, not a reflection of the LM)

EXPERIMENT:
  1. Compute all 12 T2 axis directions in 1536D hidden space
  2. Compute the LM's H6 L23 selector direction d_k from W_k matrix
  3. Compute d_q from W_q matrix
  4. Measure cosine similarity: each T2 axis vs d_k, d_q
  5. Measure T2 axis mutual orthogonality (Gram matrix)
  6. Compare to random baseline
  7. Measure T2 axes' participation in LM's top SVD components at L23

PREDICTION UNDER HYPOTHESIS:
  - T2 axes span a meaningful subspace of the LM's representation space
  - Some T2 axes should align with d_k (> 0.3 cosine)
  - T2 axes should NOT be random (higher alignment than random vectors)

PREDICTION UNDER NULL HYPOTHESIS:
  - T2 axes align with d_k no better than random vectors
  - T2 axes are semantic in a DIFFERENT geometry than the LM uses
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day113_t2_lm_alignment.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("His brother arrived at the party","His sister arrived at the party"),
        ("The father worked to feed family","The mother worked to feed family"),
        ("A son was born in the winter","A daughter was born in the winter"),
        ("The prince rode across the land","The princess rode across the land"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
        ("The dark room","The darker room"),("A hard rock","A harder rock"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The oak crashed in the storm","The tree crashed in the storm"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The soldier marched into fight","The person marched into fight"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk","Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road","The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky","Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text","The words appeared clearly in the printed text"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He spoke loudly","He spoke noisily"),
        ("It is hard","It is difficult"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The iron chain has broken now","The bond between them has broken"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
        ("The strong root grips the soil","The strong base grips the earth"),
        ("The bridge connects two banks","The bond connects two communities"),
        ("The small key opens the door","The small answer opens the path"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days","We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend","She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire","They sang together around the evening campfire"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The light is on","The dark is on"),("The news is good","The news is bad"),
        ("It is hard","It is soft"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is the first","It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("John broke the window","The window was broken by John"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The dog bit the man","The man was bitten by the dog"),
        ("The teacher helped the student","The student was helped by the teacher"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
        ("The king signed the document","The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The sun heats the cold earth","The ice melts quickly in spring"),
        ("The wind blows the tree branches","The leaves fall to the ground"),
        ("The child cries very loudly","The mother comes running in"),
        ("The ball rolls off the tall edge","The ball falls to the floor"),
        ("The teacher praises the student","The student feels very proud"),
        ("The glass breaks on hard stone","The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The car broke down again","Did the car break down again"),
        ("The dog is hungry now","Is the dog hungry now"),
        ("She wrote the letter herself","Did she write the letter herself"),
        ("He knows the right answer","Does he know the right answer"),
        ("The house looks very old","Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today","It will not rain today"),
    ],
}

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden_size={hidden_size}")

cfg        = model.config
n_heads    = cfg.num_attention_heads
n_kv_heads = cfg.num_key_value_heads
head_dim   = hidden_size // n_heads
print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}\n")

# ── Extract T2 axis directions ─────────────────────────────────────────────────
def get_last_h(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_last_h(s1, L); h2 = get_last_h(s2, L)
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    v  = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    print(f"  {name:>14}: n_pairs={len(diffs)}, norm={nv:.4f}")

# ── Extract LM internal directions ────────────────────────────────────────────
print("\nExtracting LM internal directions ...")

# H6 L23 geometric selector (Finding 40):
# W_q, W_k for layer 23
# Qwen2 attention: q_proj, k_proj, v_proj, o_proj
L23 = model.model.layers[22]   # 0-indexed
W_q = L23.self_attn.q_proj.weight.data.float().numpy()  # (n_heads*head_dim, hidden)
W_k = L23.self_attn.k_proj.weight.data.float().numpy()  # (n_kv_heads*head_dim, hidden)

# Head 6 Q projection: rows head_dim*6 .. head_dim*7
h6_q_rows = W_q[6*head_dim : 7*head_dim, :]   # (head_dim, hidden)
h6_k_rows = W_k[6*head_dim : 7*head_dim, :]   # (head_dim, hidden) — if kv_head exists

# SVD of H6 Q/K weight matrices
print("  SVD of H6 L23 W_q ...")
U_q, S_q, Vh_q = np.linalg.svd(h6_q_rows, full_matrices=False)
# v1_q: first right singular vector of W_q → direction in hidden space
v1_q = Vh_q[0, :]  # (hidden,)

print("  SVD of H6 L23 W_k ...")
# For GQA, head 6 maps to kv head: head_6 // (n_heads // n_kv_heads)
kv_group_size = n_heads // n_kv_heads
kv_head_idx   = 6 // kv_group_size
h6_k_rows_gqa = W_k[kv_head_idx*head_dim : (kv_head_idx+1)*head_dim, :]
U_k, S_k, Vh_k = np.linalg.svd(h6_k_rows_gqa, full_matrices=False)
v1_k = Vh_k[0, :]  # (hidden,)

# d_q = W_q^T @ u1_q, d_k = W_k^T @ u1_k (Finding 40 formula)
d_q = h6_q_rows.T @ U_q[:, 0]  # (hidden,)
d_k = h6_k_rows_gqa.T @ U_k[:, 0]
d_q = d_q / np.linalg.norm(d_q)
d_k = d_k / np.linalg.norm(d_k)

print(f"  d_q norm={np.linalg.norm(d_q):.4f}")
print(f"  d_k norm={np.linalg.norm(d_k):.4f}")
print(f"  cos(d_q, d_k) = {float(np.dot(d_q, d_k)):.4f}  (Finding 40: expected ~1.0)")

# Also extract L22 and L27 versions for comparison
print("\n  Extracting selector directions at L15, L22, L24 (comparison layers) ...")
lm_directions = {"d_q_L23_H6": d_q, "d_k_L23_H6": d_k}

for l_idx, l_name in [(14, "L15"), (21, "L22"), (23, "L24")]:
    try:
        Lx    = model.model.layers[l_idx]
        Wq_x  = Lx.self_attn.q_proj.weight.data.float().numpy()
        Wk_x  = Lx.self_attn.k_proj.weight.data.float().numpy()
        h6q   = Wq_x[6*head_dim : 7*head_dim, :]
        kvi   = 6 // (n_heads // n_kv_heads)
        h6k   = Wk_x[kvi*head_dim : (kvi+1)*head_dim, :]
        Uq,_,Vhq = np.linalg.svd(h6q, full_matrices=False)
        Uk,_,Vhk = np.linalg.svd(h6k, full_matrices=False)
        dq = h6q.T @ Uq[:,0]; dq /= np.linalg.norm(dq)
        dk = h6k.T @ Uk[:,0]; dk /= np.linalg.norm(dk)
        lm_directions[f"d_q_{l_name}_H6"] = dq
        lm_directions[f"d_k_{l_name}_H6"] = dk
        print(f"    cos(d_q,d_k) {l_name} H6: {float(np.dot(dq,dk)):.4f}")
    except Exception as e:
        print(f"    {l_name}: {e}")

# ── Alignment: T2 axes vs LM internal directions ──────────────────────────────
print("\n" + "=" * 72)
print("Exp 1: T2 Axis Alignment with LM Internal Directions")
print("=" * 72)
print()

dir_names = list(lm_directions.keys())
header = f"  {'axis':>14}  " + "  ".join(f"{d[:12]:>12}" for d in dir_names)
print(header)
print("  " + "-" * (14 + 14*len(dir_names)))

alignment_results = {}
for ax_name in AXIS_NAMES_12:
    v = t2_axes[ax_name]
    cosims = {}
    for d_name, d_vec in lm_directions.items():
        c = float(abs(np.dot(v, d_vec)))   # abs: direction is arbitrary
        cosims[d_name] = c
    alignment_results[ax_name] = cosims
    vals = "  ".join(f"{cosims[d]:>12.4f}" for d in dir_names)
    print(f"  {ax_name:>14}  {vals}")

# Random baseline
print()
print("  (random baseline: expected cos ~ 1/sqrt(hidden) for random vectors)")
rand_baseline = 1.0 / math.sqrt(hidden_size)
print(f"  1/sqrt({hidden_size}) = {rand_baseline:.4f}")

rng = np.random.default_rng(42)
rand_vecs  = rng.standard_normal((50, hidden_size)).astype(np.float32)
rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
rand_cosims = {d: float(np.mean(np.abs(rand_vecs @ lm_directions[d])))
               for d in dir_names}
print(f"  {'random (50 vecs)':>14}  " + "  ".join(f"{rand_cosims[d]:>12.4f}" for d in dir_names))

# ── T2 Axis Mutual Orthogonality ──────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: T2 Axis Mutual Orthogonality (Gram Matrix)")
print("=" * 72)
print()

t2_matrix = np.stack([t2_axes[ax] for ax in AXIS_NAMES_12], axis=0)  # (12, hidden)
gram = t2_matrix @ t2_matrix.T  # (12, 12) — cosine since axes are unit-normed

print("  Gram matrix (|cosines|): rows/cols = T2 axes in order")
print(f"  {' '*14}", end="")
for ax in AXIS_NAMES_12: print(f"  {ax[:4]:>4}", end="")
print()
for i, ax_i in enumerate(AXIS_NAMES_12):
    print(f"  {ax_i:>14}", end="")
    for j, ax_j in enumerate(AXIS_NAMES_12):
        print(f"  {abs(gram[i,j]):>4.2f}", end="")
    print()

off_diag = [abs(gram[i,j]) for i in range(12) for j in range(12) if i != j]
print(f"\n  Off-diagonal mean: {np.mean(off_diag):.4f}  (0=orthogonal, 1=parallel)")
print(f"  Off-diagonal max:  {np.max(off_diag):.4f}")
print(f"  Off-diagonal std:  {np.std(off_diag):.4f}")

# ── SVD of T2 axis matrix: how many dimensions does it span? ──────────────────
print()
print("=" * 72)
print("Exp 3: T2 Axis Subspace — Effective Dimensionality")
print("=" * 72)
print()

U_t2, S_t2, Vh_t2 = np.linalg.svd(t2_matrix, full_matrices=False)
explained = S_t2**2 / np.sum(S_t2**2)
cumulative = np.cumsum(explained)
print("  Singular values of T2 axis matrix (12 axes × 1536D):")
for i, (s, e, c) in enumerate(zip(S_t2, explained, cumulative)):
    bar = "█" * int(e * 40)
    print(f"  SV{i+1:>2}: {s:8.4f}  var={e:.3f}  cumvar={c:.3f}  {bar}")

eff_dim = int(np.searchsorted(cumulative, 0.90)) + 1
print(f"\n  90% variance in {eff_dim} dimensions (out of 12 axes, 1536D space)")

# ── T2 alignment with LM's top principal directions at L23 ────────────────────
print()
print("=" * 72)
print("Exp 4: T2 Axes vs LM W_q Principal Directions at L23")
print("=" * 72)
print()

# Full W_q at L23 (all heads): (n_heads*head_dim, hidden) = (1536, 1536) for Qwen2 1.5B
W_q_full = L23.self_attn.q_proj.weight.data.float().numpy()  # (1536, 1536)
# SVD: top directions in HIDDEN space that W_q projects to
print("  SVD of full L23 W_q (top 20 right singular vectors) ...")
U_wq, S_wq, Vh_wq = np.linalg.svd(W_q_full, full_matrices=False)
# Vh_wq rows are right singular vectors in hidden space, shape (min(1536,1536), 1536)

print("  Alignment of each T2 axis with top 20 W_q SVD directions:")
print(f"  {'axis':>14}  {'max_cosim':>10}  {'best_SV':>8}  {'mean_top20':>12}")
print(f"  {'-'*50}")

lm_svd_results = {}
for ax_name in AXIS_NAMES_12:
    v = t2_axes[ax_name]
    # Cosine with each of top 20 right singular vectors
    cosims_sv = [abs(float(np.dot(v, Vh_wq[k, :]))) for k in range(20)]
    best_k    = int(np.argmax(cosims_sv))
    max_c     = cosims_sv[best_k]
    mean_c    = np.mean(cosims_sv)
    lm_svd_results[ax_name] = {"max_cosim": max_c, "best_sv": best_k, "mean_top20": mean_c}
    print(f"  {ax_name:>14}  {max_c:>10.4f}  {best_k:>8}  {mean_c:>12.4f}")

# Random baseline for SVD alignment
rand_svd_cosims = [abs(float(np.dot(rv, Vh_wq[k,:])))
                   for rv in rand_vecs[:20] for k in range(20)]
print(f"  {'random':>14}  {np.max(rand_svd_cosims):>10.4f}  {'—':>8}  {np.mean(rand_svd_cosims):>12.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 113 Summary — T2 Axis / LM Internal Alignment")
print("=" * 72)

# Best T2-to-LM alignments
best_t2_dk = max(alignment_results, key=lambda a: alignment_results[a]["d_k_L23_H6"])
max_dk     = alignment_results[best_t2_dk]["d_k_L23_H6"]
mean_dk    = np.mean([alignment_results[a]["d_k_L23_H6"] for a in AXIS_NAMES_12])
rand_dk    = rand_cosims["d_k_L23_H6"]

best_svd   = max(lm_svd_results, key=lambda a: lm_svd_results[a]["max_cosim"])
max_sv     = lm_svd_results[best_svd]["max_cosim"]
rand_sv    = np.mean(rand_svd_cosims)

print(f"""
  T2 axes vs d_k (L23 H6 selector):
    Best axis:   {best_t2_dk} = {max_dk:.4f}
    Mean all 12: {mean_dk:.4f}
    Random baseline: {rand_dk:.4f}
    Signal above random: {mean_dk - rand_dk:+.4f}

  T2 axes vs W_q SVD directions (top 20):
    Best axis:   {best_svd} = {max_sv:.4f}
    Random baseline: {rand_sv:.4f}
    Signal above random: {max_sv - rand_sv:+.4f}

  T2 axis orthogonality:
    Off-diagonal mean: {np.mean(off_diag):.4f}  (0=orthogonal)
    Effective dimensionality (90% var): {eff_dim} / 12 axes

  VERDICT:
  {'T2 axes ARE ALIGNED with LM internal directions (>2× random baseline)' if mean_dk > 2*rand_dk else
   'T2 axes show WEAK alignment with LM internal directions' if mean_dk > 1.5*rand_dk else
   'T2 axes are NOT aligned with LM internal directions (≈ random)'}

  INTERPRETATION:
  {'→ TruthSpace hypothesis SUPPORTED: trie discovers LM coordinates' if mean_dk > 2*rand_dk else
   '→ TruthSpace hypothesis PARTIALLY SUPPORTED: weak alignment' if mean_dk > 1.5*rand_dk else
   '→ T2 axes are external semantic coordinates, parallel to but NOT the LM internal geometry'}

  {'→ The φ-trie operates in a DIFFERENT geometric space than the LM' if mean_dk < 1.5*rand_dk else
   '→ The φ-trie and LM share common geometric structure'}
""")

# Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "alignment_results": alignment_results,
        "lm_svd_results": lm_svd_results,
        "random_baseline_dk": rand_dk,
        "random_baseline_svd": float(np.mean(rand_svd_cosims)),
        "gram_off_diag_mean": float(np.mean(off_diag)),
        "gram_off_diag_max": float(np.max(off_diag)),
        "effective_dim_90pct": eff_dim,
        "singular_values": S_t2.tolist(),
        "cos_dq_dk_L23": float(np.dot(d_q, d_k)),
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 113 complete.")
