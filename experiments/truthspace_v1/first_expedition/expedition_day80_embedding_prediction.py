#!/usr/bin/env python3
"""
Day 80 — Embedding Prediction Test: Is the Trie Already in L0?

The ternary φ-trie address of a word is computed from hidden states
at layers 1, 15, 27, 28. A key question: is this structure ALREADY
present in the raw token embedding (L0), before any transformer
processing?

TWO HYPOTHESES:
  H_A: Trie structure is ALREADY in the embedding.
       → L0 linear classifier should predict ternary bits well.
       → The transformer just makes existing structure more accessible.
       → Implication: the embedding IS the compressed semantic fingerprint.

  H_B: Trie structure EMERGES during forward processing.
       → L0 classifier fails (≈ majority baseline).
       → The transformer computes the structure from raw embeddings.
       → Implication: semantic geometry requires compositional processing.

TEST:
  1. Extract L0 embeddings for all 401 probe words (embedding matrix lookup)
  2. For each of the 8 trie bits: train LOO logistic regression on L0 to
     predict H/U/L label (from Day 78)
  3. Also test L1, L15, L27, L28 embeddings (the actual decision layers)
  4. Plot accuracy: L0 < L1 < ... < L28 would confirm progressive emergence

ALSO: Test whether cosine distance in L0 embedding space correlates
with Hamming distance in ternary address space.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
INPUT_FILE  = str(SCRIPT_DIR / "day78_scale_vocab.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day80_embedding_prediction.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

AXIS_NAMES = [
    "gender",   "comparative", "hypernym",  "plural",
    "synonym",  "concrete",    "past_tense", "antonym",
]

REQUIRED_LAYERS = [0, 1, 15, 27, 28]   # L0 = raw embedding

# ── Load pre-computed ternary addresses ───────────────────────────────────────
with open(INPUT_FILE) as f:
    saved = json.load(f)
addresses = saved["addresses"]   # word → 8-char string over {H,U,L}
words = list(addresses.keys())
print(f"Loaded {len(words)} word addresses from Day 78\n")

# ── Load model (to extract embedding matrix) ─────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

# ── Efficient extraction: single pass per word, all REQUIRED_LAYERS ──────────
print(f"Extracting hidden states at L{REQUIRED_LAYERS} for {len(words)} words ...")
word_hiddens = {}   # word → {layer: vector}
for word in words:
    try:
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        word_hiddens[word] = {
            l: out.hidden_states[l][0, pos, :].numpy().astype(np.float32)
            for l in REQUIRED_LAYERS
        }
    except Exception as e:
        print(f"  SKIP {word!r}: {e}")

valid_words = [w for w in words if w in word_hiddens]
print(f"  Extracted {len(valid_words)} words\n")

# ── Simple LOO logistic regression (pure numpy, no sklearn) ─────────────────
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def loo_logistic_accuracy(X, y_str):
    """Leave-one-out logistic regression accuracy.
    X: (n, d) float32 features
    y_str: list of n labels (strings)
    Returns: per-class accuracy and overall accuracy.
    """
    classes = sorted(set(y_str))
    y = np.array([classes.index(c) for c in y_str], dtype=np.int32)
    n, d = X.shape
    correct = 0
    preds   = []

    for i in range(n):
        X_train = np.vstack([X[:i], X[i+1:]])
        y_train = np.concatenate([y[:i], y[i+1:]])
        x_test  = X[i]

        # Normalize features
        mu, sg = X_train.mean(0), X_train.std(0) + 1e-8
        Xt = (X_train - mu) / sg
        xt = (x_test  - mu) / sg

        # Train 1-vs-rest LR with gradient descent (50 steps)
        k = len(classes)
        W = np.zeros((k, d), dtype=np.float32)
        lr_rate = 0.01
        for _ in range(50):
            # Forward
            scores = Xt @ W.T   # (n-1, k)
            # Softmax
            probs  = np.array([softmax(s) for s in scores])   # (n-1, k)
            # Gradient
            dL = probs.copy()
            dL[np.arange(n-1), y_train] -= 1
            dL /= (n - 1)
            grad = dL.T @ Xt   # (k, d)
            W -= lr_rate * grad

        # Predict
        pred = np.argmax(W @ xt)
        preds.append(pred)
        if pred == y[i]: correct += 1

    overall_acc = correct / n
    # Per-class accuracy
    per_class = {}
    for ci, c in enumerate(classes):
        idxs = [i for i, c2 in enumerate(y_str) if c2 == c]
        hits = sum(1 for i in idxs if preds[i] == y[i])
        per_class[c] = hits / len(idxs) if idxs else 0.0

    # Majority baseline
    majority = max(Counter(y_str).values()) / n
    return overall_acc, per_class, majority

from collections import Counter

def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# ── Per-layer, per-bit LOO accuracy ──────────────────────────────────────────
print("=" * 72)
print("LOO logistic regression accuracy: predict trie bit from hidden state")
print("=" * 72)
print(f"  {'layer':>7}  " + "  ".join(f"{a:>11}" for a in AXIS_NAMES))
print(f"  {'':>7}  " + "  ".join(f"{'acc/base':>11}" for _ in AXIS_NAMES))

all_results = {}
for layer in REQUIRED_LAYERS:
    X_mat  = np.array([word_hiddens[w][layer] for w in valid_words], dtype=np.float32)
    layer_accs = []
    layer_bases= []
    for bit_i in range(8):
        y_str = [addresses[w][bit_i] for w in valid_words]
        acc, pc, base = loo_logistic_accuracy(X_mat, y_str)
        layer_accs.append(acc)
        layer_bases.append(base)
    all_results[layer] = {"accs": layer_accs, "bases": layer_bases}

    accs_str  = "  ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(layer_accs, layer_bases))
    print(f"  L{layer:>5}:  {accs_str}")

print()

# ── Summary: L0 vs L28 per bit ────────────────────────────────────────────────
print("=" * 72)
print("L0 vs L28 accuracy per bit (does transformer improve prediction?)")
print("=" * 72)
print(f"  {'bit/axis':>20}  {'L0_acc':>8}  {'L28_acc':>8}  {'baseline':>8}  "
      f"{'L0_gain':>8}  {'L28_gain':>8}  progression")

for bit_i, name in enumerate(AXIS_NAMES):
    l0_acc  = all_results[0]["accs"][bit_i]
    l28_acc = all_results[28]["accs"][bit_i]
    base    = all_results[0]["bases"][bit_i]
    l0_gain = l0_acc - base
    l28_gain = l28_acc - base
    prog    = " ".join(f"{all_results[l]['accs'][bit_i]:.2f}" for l in REQUIRED_LAYERS)
    print(f"  {name:>20}  {l0_acc:>8.3f}  {l28_acc:>8.3f}  {base:>8.3f}  "
          f"{l0_gain:>+8.3f}  {l28_gain:>+8.3f}  {prog}")

print()

# Key question: does L0 beat baseline?
l0_gains   = [all_results[0]["accs"][i] - all_results[0]["bases"][i]
              for i in range(8)]
l28_gains  = [all_results[28]["accs"][i] - all_results[28]["bases"][i]
              for i in range(8)]
l0_pos     = sum(1 for g in l0_gains if g > 0.02)
l28_pos    = sum(1 for g in l28_gains if g > 0.02)

print(f"  L0  beats baseline by >2% on {l0_pos}/8 bits")
print(f"  L28 beats baseline by >2% on {l28_pos}/8 bits")
h_a = "CONFIRMED" if l0_pos >= 6 else "PARTIAL" if l0_pos >= 3 else "REJECTED"
h_b = "CONFIRMED" if l0_pos <= 2 else "PARTIAL" if l0_pos <= 5 else "REJECTED"
print(f"\n  H_A (trie already in L0):      {h_a}")
print(f"  H_B (trie emerges in L28):     {h_b}")
print()

# ── L0 cosine distance vs trie Hamming ────────────────────────────────────────
print("=" * 72)
print("L0 embedding cosine similarity vs Hamming distance in trie address")
print("=" * 72)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

by_hamming_l0  = {d: [] for d in range(9)}
by_hamming_l28 = {d: [] for d in range(9)}

for i in range(len(valid_words)):
    for j in range(i+1, len(valid_words)):
        w1, w2 = valid_words[i], valid_words[j]
        d   = hamming(addresses[w1], addresses[w2])
        s0  = cos_sim(word_hiddens[w1][0],  word_hiddens[w2][0])
        s28 = cos_sim(word_hiddens[w1][28], word_hiddens[w2][28])
        by_hamming_l0[d].append(s0)
        by_hamming_l28[d].append(s28)

print(f"  {'d':>4}  {'L0_cosim':>10}  {'L28_cosim':>10}  {'n_pairs':>8}")
l0_monotone  = True
l28_monotone = True
prev_l0 = prev_l28 = 1.1
for d in range(9):
    if not by_hamming_l0[d]: continue
    m0   = float(np.mean(by_hamming_l0[d]))
    m28  = float(np.mean(by_hamming_l28[d]))
    n    = len(by_hamming_l0[d])
    mark0  = "↓" if m0  < prev_l0  else "↑"
    mark28 = "↓" if m28 < prev_l28 else "↑"
    if m0  > prev_l0:  l0_monotone  = False
    if m28 > prev_l28: l28_monotone = False
    print(f"  {d:>4}  {m0:>10.4f}{mark0} {m28:>10.4f}{mark28} {n:>8}")
    prev_l0, prev_l28 = m0, m28

print(f"\n  L0  monotone: {'YES ✓' if l0_monotone  else 'NO'}")
print(f"  L28 monotone: {'YES ✓' if l28_monotone else 'NO'}")
l0_range  = float(np.mean(by_hamming_l0[0]))  - float(np.mean(by_hamming_l0[8]))
l28_range = float(np.mean(by_hamming_l28[0])) - float(np.mean(by_hamming_l28[8]))
print(f"\n  L0  range (d=0 to d=8): {l0_range:+.4f}")
print(f"  L28 range (d=0 to d=8): {l28_range:+.4f}")
print(f"  L28/L0 range ratio: {l28_range/l0_range:.2f}x" if l0_range != 0 else "")
print()

# ── PCA variance in L0 vs L28 for the trie subspace ─────────────────────────
print("=" * 72)
print("PCA in L0 vs L28: how much variance is in trie-relevant subspace?")
print("=" * 72)

# Build difference vectors between address-H and address-L tokens for each bit
for layer in [0, 1, 28]:
    H_vecs = defaultdict_list = []   # won't use defaultdict
    from collections import defaultdict
    bit_H_vecs = defaultdict(list)
    bit_L_vecs = defaultdict(list)
    for w in valid_words:
        for bit_i in range(8):
            c = addresses[w][bit_i]
            v = word_hiddens[w][layer]
            if c == "H": bit_H_vecs[bit_i].append(v)
            elif c == "L": bit_L_vecs[bit_i].append(v)

    # Mean difference vector for each bit
    diff_vecs = []
    for bit_i in range(8):
        if not bit_H_vecs[bit_i] or not bit_L_vecs[bit_i]: continue
        h_mean = np.mean(bit_H_vecs[bit_i], axis=0)
        l_mean = np.mean(bit_L_vecs[bit_i], axis=0)
        diff   = h_mean - l_mean
        n      = np.linalg.norm(diff)
        if n > 1e-6:
            diff_vecs.append(diff / n)

    if not diff_vecs: continue

    # SVD of difference matrix
    D = np.array(diff_vecs, dtype=np.float32)
    _, sv, _ = np.linalg.svd(D, full_matrices=False)
    sv_norm = sv / sv.sum()
    print(f"  L{layer:>2} difference-vector SVD (first 4 singular values):")
    print(f"    {' '.join(f'{v:.3f}' for v in sv_norm[:4])}")
    print(f"    PC0 explains {100*sv_norm[0]:.1f}% of trie subspace variance")
print()

# ── Save ─────────────────────────────────────────────────────────────────────
save_data = {
    "per_layer_per_bit_accuracy": {
        str(l): {"accs": all_results[l]["accs"],
                 "bases": all_results[l]["bases"]}
        for l in REQUIRED_LAYERS
    },
    "l0_gains":  l0_gains,
    "l28_gains": l28_gains,
    "l0_beats_baseline": l0_pos,
    "l28_beats_baseline": l28_pos,
    "hypothesis_A": h_a,
    "hypothesis_B": h_b,
    "hamming_vs_l0_cosim":  {str(d): float(np.mean(v)) for d, v in by_hamming_l0.items()  if v},
    "hamming_vs_l28_cosim": {str(d): float(np.mean(v)) for d, v in by_hamming_l28.items() if v},
    "l0_monotone":  l0_monotone,
    "l28_monotone": l28_monotone,
    "l0_range":  l0_range,
    "l28_range": l28_range,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 80 complete.")
