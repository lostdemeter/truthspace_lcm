#!/usr/bin/env python3
"""
Day 247 — Antonym LOO with Nearest-Neighbour Voting

Background: antonym_size mean_dir retrieval = 0% accuracy (compared to
adj_degree 95.7%). The antonym arc angle has high variance (std=15°),
making a single mean direction useless.

Hypothesis: for each source word, the BEST analogical antonym predictor
is not the global mean direction but a weighted average from the k
most similar source words in the training set (whose antonyms are known).

Methods tested:
  0. Baseline: mean_dir (global average chord vector)
  1. 1-NN: use chord direction from single nearest training source
  2. kNN-linear: weighted average of k nearest training chords (cos-weighted)
  3. kNN-analogy: Mikolov-style analogy: pred = w + (nb_tgt - nb_src)
     where nb is the nearest known antonym training pair
  4. Oracle: use the EXACT chord (only possible for in-sample words)
  5. Angle-corrected: use mean_dir but rescaled to cos(A,B) target
     Since cos(antonym pairs) ≈ 0.23, a rescaled direction might work

Also: investigate WHY mean_dir fails for antonyms:
  - Plot chord direction variance (std of chord directions)
  - Compare to adj_degree (which has low variance)

And: revisit plural and past_tense retrieval with NN methods.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "antonym_nn.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PARADIGMS = {
    "antonym_size": [
        ("big","small"), ("large","tiny"), ("huge","little"),
        ("tall","short"), ("wide","narrow"), ("thick","thin"),
        ("broad","slim"), ("heavy","light"), ("long","brief"),
        ("hot","cold"), ("fast","slow"), ("hard","soft"),
        ("loud","quiet"), ("dark","bright"), ("old","young"),
        ("strong","weak"), ("rich","poor"), ("deep","shallow"),
        ("early","late"), ("clean","dirty"), ("warm","cool"),
        ("sharp","dull"), ("sweet","sour"), ("rough","smooth"),
    ],
    "adj_degree": [
        ("big","bigger"), ("fast","faster"), ("long","longer"),
        ("small","smaller"), ("hard","harder"), ("bright","brighter"),
        ("dark","darker"), ("rich","richer"), ("deep","deeper"),
        ("wide","wider"), ("high","higher"), ("low","lower"),
        ("old","older"), ("young","younger"), ("hot","hotter"),
        ("tall","taller"), ("strong","stronger"), ("weak","weaker"),
        ("short","shorter"), ("cool","cooler"), ("great","greater"),
        ("safe","safer"), ("cheap","cheaper"), ("clean","cleaner"),
    ],
    "plural": [
        ("cat","cats"), ("dog","dogs"), ("house","houses"),
        ("tree","trees"), ("book","books"), ("car","cars"),
        ("bird","birds"), ("ship","ships"), ("hand","hands"),
        ("door","doors"), ("king","kings"), ("boy","boys"),
        ("road","roads"), ("room","rooms"), ("eye","eyes"),
        ("foot","feet"), ("tooth","teeth"), ("man","men"),
    ],
}

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

# Load paradigm data
pdata = {}
for pname, pairs in PARADIGMS.items():
    srcs, tgts = [], []
    words = []
    for a_w, b_w in pairs:
        A = get_emb(a_w); B = get_emb(b_w)
        if A is not None and B is not None:
            srcs.append(A); tgts.append(B); words.append((a_w, b_w))
    pdata[pname] = {
        "srcs": np.array(srcs), "tgts": np.array(tgts), "words": words
    }
    print(f"  {pname}: {len(words)} pairs loaded")

# ── Part A: Chord direction variance analysis ─────────────────────────
print()
print("=" * 70)
print("PART A: CHORD DIRECTION VARIANCE ANALYSIS")
print("        Why does mean_dir work for adj_degree but not antonyms?")
print("=" * 70)
print()

for pname in PARADIGMS:
    d = pdata[pname]
    if len(d["srcs"]) == 0: continue
    chords = d["tgts"] - d["srcs"]
    chord_norms = np.array([normed(c) for c in chords])
    # Mean chord direction
    mean_chord = np.mean(chords, axis=0)
    mean_chord_n = normed(mean_chord)
    # Alignment of each chord with mean
    alignments = [float(np.dot(chord_norms[i], mean_chord_n))
                  for i in range(len(chord_norms))]
    # Pairwise chord cosines
    pair_cos = []
    for i in range(len(chord_norms)):
        for j in range(i+1, len(chord_norms)):
            pair_cos.append(float(np.dot(chord_norms[i], chord_norms[j])))
    print(f"  {pname:>16}  n={len(chords):>2}  "
          f"mean_align={np.mean(alignments):.4f}  "
          f"min_align={np.min(alignments):.4f}  "
          f"mean_pair_cos={np.mean(pair_cos):.4f}")
print()
print("  High mean_align → mean_dir is a good representative of all chords")
print("  Low mean_align → mean_dir is a poor representative (high variance)")

# ── Part B: LOO evaluation — all methods ─────────────────────────────
print()
print("=" * 70)
print("PART B: LOO RETRIEVAL — ALL METHODS")
print("=" * 70)

results = {}
K_NN = 5  # for kNN methods

for pname in PARADIGMS:
    d = pdata[pname]
    N = len(d["srcs"])
    if N < 4: continue
    srcs = d["srcs"]; tgts = d["tgts"]; words = d["words"]
    chords = tgts - srcs
    srcs_n = np.array([normed(s) for s in srcs], dtype=np.float32)

    counts = {k: 0 for k in ["mean_dir", "1nn", "knn_linear",
                              "knn_analogy", "oracle"]}
    cos_acc = {k: [] for k in counts}

    for i in range(N):
        a_w, b_w = words[i]
        t_b = tid1(b_w)
        if t_b is None: continue
        excl = [tid1(a_w)]

        train = [j for j in range(N) if j != i]
        src_i = srcs[i]; tgt_i = tgts[i]

        # 0. Mean_dir
        mean_d = np.mean(chords[train], axis=0)
        pred0 = src_i + mean_d
        nn0 = nn1(pred0, excl)
        cos_acc["mean_dir"].append(cos_sim(pred0, tgt_i))
        if nn0 == t_b: counts["mean_dir"] += 1

        # Compute src similarities to training sources
        src_i_n = normed(src_i).astype(np.float32)
        sims_to_train = srcs_n[train] @ src_i_n  # shape (len(train),)

        # 1. 1-NN: use chord from nearest training source
        nn_idx = train[int(np.argmax(sims_to_train))]
        pred1 = src_i + chords[nn_idx]
        nn1v = nn1(pred1, excl)
        cos_acc["1nn"].append(cos_sim(pred1, tgt_i))
        if nn1v == t_b: counts["1nn"] += 1

        # 2. kNN-linear: top-k weighted average
        k = min(K_NN, len(train))
        topk_rel = np.argsort(sims_to_train)[-k:]
        topk_abs = [train[r] for r in topk_rel]
        weights = np.array([sims_to_train[r] for r in topk_rel])
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            pred2 = src_i + np.sum([weights[ki] * chords[topk_abs[ki]]
                                     for ki in range(k)], axis=0)
        else:
            pred2 = src_i + mean_d
        nn2 = nn1(pred2, excl)
        cos_acc["knn_linear"].append(cos_sim(pred2, tgt_i))
        if nn2 == t_b: counts["knn_linear"] += 1

        # 3. kNN-analogy: Mikolov-style
        # pred = src_i + (tgt_nb - src_nb) for nearest neighbor nb
        nb_src = srcs[nn_idx]; nb_tgt = tgts[nn_idx]
        pred3 = src_i + (nb_tgt - nb_src)
        nn3 = nn1(pred3, excl)
        cos_acc["knn_analogy"].append(cos_sim(pred3, tgt_i))
        if nn3 == t_b: counts["knn_analogy"] += 1

        # 4. Oracle: exact chord (uses true pair — upper bound)
        pred4 = src_i + chords[i]  # = tgt_i exactly
        nn4 = nn1(pred4, [tid1(a_w)])  # don't exclude target
        cos_acc["oracle"].append(cos_sim(pred4, tgt_i))
        if nn4 == t_b: counts["oracle"] += 1

    print()
    print(f"  PARADIGM: {pname}  (n={N})")
    print(f"  {'Method':<16}  {'acc':>5}  {'cos':>7}")
    for mname in ["mean_dir", "1nn", "knn_linear", "knn_analogy", "oracle"]:
        n_eval = len(cos_acc[mname])
        acc = counts[mname] / n_eval if n_eval > 0 else 0
        c = np.mean(cos_acc[mname]) if cos_acc[mname] else 0
        print(f"  {mname:<16}  {counts[mname]:>3}/{n_eval}  {acc:.3f}  {c:.4f}")

    results[pname] = {k: counts[k] for k in counts}
    results[pname]["N"] = N

# ── Part C: Why 1-NN helps (or doesn't) ──────────────────────────────
print()
print("=" * 70)
print("PART C: NEAREST-NEIGHBOR ANALYSIS FOR ANTONYMS")
print("        Who is the nearest training source for each antonym?")
print("=" * 70)
print()

d = pdata["antonym_size"]
N = len(d["srcs"])
if N > 0:
    srcs = d["srcs"]; tgts = d["tgts"]; words = d["words"]
    srcs_n = np.array([normed(s) for s in srcs], dtype=np.float32)
    print(f"  {'src':>10}  {'tgt':>10}  {'1nn_src':>12}  {'1nn_sim':>8}  "
          f"{'1nn_tgt':>10}  {'correct'}  chord_cos")
    for i in range(min(N, 15)):
        a_w, b_w = words[i]
        train = [j for j in range(N) if j != i]
        src_n = normed(srcs[i]).astype(np.float32)
        sims  = srcs_n[train] @ src_n
        nb_idx = train[int(np.argmax(sims))]
        nb_sim = float(sims[train.index(nb_idx)] if nb_idx in train
                       else np.max(sims))
        nb_sim = float(np.max(sims))
        nb_src_w, nb_tgt_w = words[nb_idx]
        # 1-NN prediction
        pred = srcs[i] + (tgts[nb_idx] - srcs[nb_idx])
        t_b = tid1(b_w)
        nn_pred = nn1(pred, [tid1(a_w)])
        pred_word = tok.decode([nn_pred]).strip()
        correct = "✓" if nn_pred == t_b else "✗"
        # Chord cosine alignment
        chord_i = normed(tgts[i] - srcs[i])
        chord_nb = normed(tgts[nb_idx] - srcs[nb_idx])
        c = float(np.dot(chord_i, chord_nb))
        print(f"  {a_w:>10}  {b_w:>10}  {nb_src_w:>12}  {nb_sim:>8.4f}  "
              f"{nb_tgt_w:>10}  {correct}   {c:.4f}  → {pred_word}")

# ── Part D: Best achievable with per-word direction ───────────────────
print()
print("=" * 70)
print("PART D: WHAT WOULD PERFECT DIRECTION PREDICTION GIVE?")
print("        Upper bound: if we knew the exact chord direction")
print("        but had to SCALE it to reach the target norm.")
print("=" * 70)
print()

d = pdata["antonym_size"]
if len(d["srcs"]) > 0:
    srcs = d["srcs"]; tgts = d["tgts"]; words = d["words"]
    correct_dir = 0; correct_norm = 0
    mean_chord_norm = np.mean([np.linalg.norm(tgts[i] - srcs[i])
                                for i in range(len(srcs))])
    for i in range(len(srcs)):
        a_w, b_w = words[i]
        t_b = tid1(b_w)
        if t_b is None: continue
        # Perfect direction + mean norm scaling
        exact_chord = tgts[i] - srcs[i]
        exact_dir = normed(exact_chord)
        pred_dir_only = srcs[i] + exact_dir * mean_chord_norm
        nn_d = nn1(pred_dir_only, [tid1(a_w)])
        if nn_d == t_b: correct_dir += 1
        # Perfect chord (oracle)
        pred_exact = tgts[i]
        nn_e = nn1(pred_exact, [tid1(a_w)])
        if nn_e == t_b: correct_norm += 1
    N_eval = len(srcs)
    print(f"  Perfect direction + mean norm: {correct_dir}/{N_eval} = "
          f"{correct_dir/N_eval:.3f}")
    print(f"  Perfect chord (oracle target): {correct_norm}/{N_eval} = "
          f"{correct_norm/N_eval:.3f}")
    print()
    print(f"  Mean antonym chord norm: {mean_chord_norm:.4f}")
    print(f"  Mean adj_degree chord norm: "
          f"{np.mean([np.linalg.norm(pdata['adj_degree']['tgts'][i] - pdata['adj_degree']['srcs'][i]) for i in range(len(pdata['adj_degree']['srcs']))]):.4f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Antonym NN analysis complete.")
