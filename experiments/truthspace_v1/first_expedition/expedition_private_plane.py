#!/usr/bin/env python3
"""
Day 251 — Private Plane Orientation and Semantic Type

Question: does semantic type determine the arc plane orientation?
If yes: same-type adjectives share similar arc planes -> high intra-type
        chord coherence, low cross-type chord coherence.
If no:  the private plane is word-specific, unpredictable from type.

Semantic subclasses tested:
  SIZE:        big, large, huge, small, tiny, great, wide, broad, narrow, thick, thin
  TEMPERATURE: hot, warm, cool, cold, icy
  SPEED:       fast, quick, slow
  INTENSITY:   bright, dark, loud, soft, strong, weak, heavy, light, hard, tough
  TEMPORAL:    old, young, early, late, long, short, new
  QUALITY:     nice, fine, clean, clear, safe, simple, rich, poor

For each subclass: compute mean chord vector and pairwise chord cosines.
Then compare intra-class coherence vs cross-class coherence.

Also: test whether the mean_dir for a subclass retrieves better than
the global mean_dir (LOO accuracy by semantic subclass).
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "private_plane.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

SEMANTIC_GROUPS = {
    "SIZE":        [("big","bigger"),("large","larger"),("small","smaller"),
                   ("wide","wider"),("broad","broader"),("narrow","narrower"),
                   ("thick","thicker"),("thin","thinner"),("tall","taller"),
                   ("short","shorter"),("long","longer"),("deep","deeper"),
                   ("high","higher"),("low","lower")],
    "TEMPERATURE": [("hot","hotter"),("warm","warmer"),("cool","cooler"),
                   ("cold","colder")],
    "SPEED":       [("fast","faster"),("quick","quicker"),("slow","slower")],
    "INTENSITY":   [("bright","brighter"),("dark","darker"),("loud","louder"),
                   ("soft","softer"),("strong","stronger"),("weak","weaker"),
                   ("heavy","heavier"),("light","lighter"),("hard","harder"),
                   ("tough","tougher")],
    "TEMPORAL":    [("old","older"),("young","younger"),("late","later"),
                   ("long","longer"),("short","shorter"),("new","newer")],
    "QUALITY":     [("nice","nicer"),("fine","finer"),("clean","cleaner"),
                   ("clear","clearer"),("safe","safer"),("simple","simpler"),
                   ("rich","richer"),("poor","poorer"),("cheap","cheaper")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cos_sim(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
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

# ── Build chord vectors per group ─────────────────────────────────────
print("Building chord vectors per semantic group ...")
group_chords = {}  # group -> list of (base, comp, chord_normed)
for grp, pairs in SEMANTIC_GROUPS.items():
    chords = []
    for base, comp in pairs:
        e_base = get_emb(base); e_comp = get_emb(comp)
        if e_base is None or e_comp is None: continue
        ch = e_comp - e_base
        chords.append((base, comp, normed(ch)))
    group_chords[grp] = chords
    print(f"  {grp:<12}: {len(chords)}/{len(pairs)} pairs loaded")

# ── Part A: Intra-group chord coherence ───────────────────────────────
print()
print("=" * 70)
print("PART A: INTRA-GROUP CHORD COHERENCE")
print("        Does same semantic type → aligned arc planes?")
print("=" * 70)
print()

random_baseline = 1.0 / np.sqrt(H)
print(f"  Random baseline (R^{H}): 1/sqrt(H) = {random_baseline:.4f}\n")

intra_coherences = {}
group_mean_dirs = {}
print(f"  {'Group':<12}  {'n':>4}  {'mean_pair_cos':>14}  {'ratio/random':>13}  {'mean_dir_norm':>14}")
for grp, chords in group_chords.items():
    if len(chords) < 2: continue
    ch_vecs = np.array([c for _, _, c in chords])
    n = len(ch_vecs)
    sims = ch_vecs @ ch_vecs.T
    ut = sims[np.triu_indices(n, k=1)]
    mean_pc = float(ut.mean())
    intra_coherences[grp] = mean_pc
    mean_dir = ch_vecs.mean(axis=0)
    group_mean_dirs[grp] = mean_dir
    ratio = mean_pc / random_baseline
    print(f"  {grp:<12}  {n:>4}  {mean_pc:>14.4f}  {ratio:>13.2f}x  "
          f"{np.linalg.norm(mean_dir):>14.4f}")

# ── Part B: Cross-group chord coherence ───────────────────────────────
print()
print("=" * 70)
print("PART B: CROSS-GROUP CHORD COHERENCE MATRIX")
print("        Do different semantic types share arc plane orientations?")
print("=" * 70)
print()

groups = list(group_chords.keys())
n_g = len(groups)
cross_mat = np.zeros((n_g, n_g))
for i, g1 in enumerate(groups):
    for j, g2 in enumerate(groups):
        if i == j:
            cross_mat[i, j] = intra_coherences.get(g1, 0)
            continue
        ch1 = np.array([c for _, _, c in group_chords[g1]])
        ch2 = np.array([c for _, _, c in group_chords[g2]])
        if len(ch1) == 0 or len(ch2) == 0:
            cross_mat[i, j] = 0; continue
        # All pairwise cosines between groups
        sims = ch1 @ ch2.T
        cross_mat[i, j] = float(sims.mean())

print(f"  {'':12}  " + "  ".join(f"{g[:6]:>8}" for g in groups))
for i, g1 in enumerate(groups):
    row = f"  {g1:<12}  " + "  ".join(f"{cross_mat[i,j]:>8.4f}" for j in range(n_g))
    print(row)

# Off-diagonal mean
off_diag = [cross_mat[i, j] for i in range(n_g) for j in range(n_g) if i != j]
print(f"\n  Diagonal (intra-group) mean: {np.mean([cross_mat[i,i] for i in range(n_g)]):.4f}")
print(f"  Off-diagonal (cross-group) mean: {np.mean(off_diag):.4f}")
print(f"  Random baseline: {random_baseline:.4f}")

# ── Part C: LOO accuracy by semantic subclass ─────────────────────────
print()
print("=" * 70)
print("PART C: LOO RETRIEVAL ACCURACY — SUBCLASS mean_dir vs GLOBAL mean_dir")
print("=" * 70)
print()

# Global mean_dir from all pairs
all_chords_raw = []
for grp, chords in group_chords.items():
    for base, comp, ch_n in chords:
        e_base = get_emb(base); e_comp = get_emb(comp)
        if e_base is not None and e_comp is not None:
            all_chords_raw.append(e_comp - e_base)
global_mean_dir = np.mean(all_chords_raw, axis=0)

def loo_accuracy(pairs_list, direction_vec, excl_source=True):
    """LOO: for each pair (A,B), predict B from A + direction_vec."""
    correct = 0
    for base, comp in pairs_list:
        e_base = get_emb(base)
        t_comp = tid1(comp)
        if e_base is None or t_comp is None: continue
        pred = e_base + direction_vec
        pred_n = normed(pred).astype(np.float32)
        sims = Wn @ pred_n
        # Exclude source word
        t_base = tid1(base)
        if t_base is not None: sims[t_base] = -2.0
        if int(np.argmax(sims)) == t_comp:
            correct += 1
    total = sum(1 for b, c in pairs_list
                if get_emb(b) is not None and tid1(c) is not None)
    return correct, total

print(f"  {'Group':<12}  {'n':>4}  {'global_acc':>11}  {'subclass_acc':>13}  {'improvement':>12}")
results_loo = {}
for grp, pairs in SEMANTIC_GROUPS.items():
    chords_in_group = group_chords[grp]
    if len(chords_in_group) < 3: continue

    # LOO with GLOBAL mean_dir
    g_cor, g_tot = loo_accuracy(pairs, global_mean_dir)

    # LOO with SUBCLASS-specific mean_dir (LOO: exclude current pair)
    s_cor = 0; s_tot = 0
    for k, (base, comp) in enumerate(pairs):
        e_base = get_emb(base); t_comp = tid1(comp)
        if e_base is None or t_comp is None: continue
        # Subclass mean_dir excluding current pair
        sub_chords = [ch for i, (_, _, ch) in enumerate(chords_in_group)
                      if chords_in_group[i][0] != base]
        if len(sub_chords) == 0: continue
        sub_mean = np.mean([
            get_emb(c) - get_emb(b)
            for b, c, _ in chords_in_group if b != base
            if get_emb(b) is not None and get_emb(c) is not None
        ], axis=0)
        pred = e_base + sub_mean
        pred_n = normed(pred).astype(np.float32)
        sims = Wn @ pred_n
        t_base = tid1(base)
        if t_base is not None: sims[t_base] = -2.0
        if int(np.argmax(sims)) == t_comp: s_cor += 1
        s_tot += 1

    g_pct = g_cor / g_tot if g_tot > 0 else 0
    s_pct = s_cor / s_tot if s_tot > 0 else 0
    improvement = s_pct - g_pct
    print(f"  {grp:<12}  {g_tot:>4}  "
          f"{g_cor}/{g_tot}={g_pct:.2f}  "
          f"{s_cor}/{s_tot}={s_pct:.2f}  "
          f"{improvement:>+.2f}")
    results_loo[grp] = {"global": g_pct, "subclass": s_pct}

# ── Part D: Mean-dir of each group — are they orthogonal? ─────────────
print()
print("=" * 70)
print("PART D: CROSS-GROUP MEAN_DIR COSINES")
print("        Are the group mean transformation directions orthogonal?")
print("=" * 70)
print()

gmds = {g: normed(group_mean_dirs[g]) for g in groups if g in group_mean_dirs}
gmd_groups = list(gmds.keys())
print(f"  {'':12}  " + "  ".join(f"{g[:6]:>8}" for g in gmd_groups))
for g1 in gmd_groups:
    row = f"  {g1:<12}  " + "  ".join(
        f"{cos_sim(gmds[g1], gmds[g2]):>8.4f}" for g2 in gmd_groups)
    print(row)

# ── Part E: Are subclass planes aligned with W_E principal axes? ──────
print()
print("=" * 70)
print("PART E: ALIGNMENT OF GROUP MEAN_DIR WITH W_E GLOBAL DIRECTIONS")
print("        Project group mean_dirs onto top-10 PCA axes of W_E")
print("=" * 70)
print()

# Quick PCA of W_E (center and SVD)
print("  Computing top-10 PCA axes of W_E ...")
W_E_center = W_E - W_E.mean(axis=0)
# Use random projection for efficiency: project onto random 100D first
rng = np.random.default_rng(42)
proj_mat = rng.standard_normal((H, 100)) / np.sqrt(100)
W_proj = W_E_center @ proj_mat
U, S, Vt = np.linalg.svd(W_proj, full_matrices=False)
# The top PCA axes in original space: W_E_center.T @ U[:, :10] / sqrt(n)
pca_axes_raw = W_E_center.T @ U[:, :10]
pca_axes = np.array([normed(pca_axes_raw[:, k]) for k in range(10)])  # (10, H)
print(f"  PCA axis strengths (proxy): {S[:5].round(1)}\n")

print(f"  Group alignment with top-10 PCA axes:")
print(f"  {'Group':<12}  " + "  ".join(f"PC{k+1:>3}" for k in range(10)))
for g in gmd_groups:
    md = normed(gmds[g]).astype(np.float64)
    alignments = [float(np.dot(md, pca_axes[k])) for k in range(10)]
    row = f"  {g:<12}  " + "  ".join(f"{a:>5.3f}" for a in alignments)
    max_align = max(abs(a) for a in alignments)
    print(f"{row}  max={max_align:.3f}")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
mean_intra = np.mean(list(intra_coherences.values()))
print(f"  Mean intra-group chord coherence: {mean_intra:.4f}")
print(f"  Mean cross-group chord coherence: {np.mean(off_diag):.4f}")
print(f"  Random baseline:                  {random_baseline:.4f}")
print()
print(f"  Intra/random ratio: {mean_intra/random_baseline:.2f}x")
print(f"  Cross/random ratio: {np.mean(off_diag)/random_baseline:.2f}x")
print()
if mean_intra > 3 * np.mean(off_diag):
    print("  VERDICT: Private plane IS substantially determined by semantic type")
    print("           (intra >> cross coherence)")
elif mean_intra > 1.5 * np.mean(off_diag):
    print("  VERDICT: Private plane is PARTIALLY determined by semantic type")
else:
    print("  VERDICT: Private plane is NOT primarily determined by semantic type")
    print("           (intra ≈ cross coherence)")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "intra_coherences": intra_coherences,
        "cross_matrix": cross_mat.tolist(),
        "groups": groups,
        "loo_results": results_loo,
        "random_baseline": float(random_baseline),
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Private plane analysis complete.")
