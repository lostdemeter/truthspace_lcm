#!/usr/bin/env python3
"""
Day 210 — Special-Case Encoding: Antonym Axes + Numbers Cross-Script

Two open problems from Day 208:
  1. ANTONYMS: acc=0.500 with proximity; need per-attribute flip axis
  2. NUMBERS: ordinal detection fails; word↔digit is cross-script

EXPERIMENT A: Antonym Semantic Axes
  Hypothesis: each antonym pair flips along a dedicated semantic axis.
  Temperature axis: hot↔cold
  Size axis:        big↔small
  Speed axis:       fast↔slow
  Volume axis:      loud↔quiet
  Age axis:         old↔young
  Light axis:       light↔dark
  Sharpness axis:   sharp↔dull
  Texture axis:     hard↔soft
  Wealth axis:      rich↔poor
  Thickness:        thick↔thin

  For each attribute, compute the axis vector = (pos - neg) / 2
  Measure: pairwise cosine between all attribute axes → are they parallel?
  Test: can the attribute axis retrieve the antonym?
  Test: can axis from ONE pair retrieve antonym of a DIFFERENT pair
        on the same attribute?

EXPERIMENT B: Numbers Cross-Script
  Hypothesis: word-number tokens (one,two,...) and digit tokens (1,2,...)
  are cross-script neighbors in W_E, like ice/冰 from Day 194.

  For each number 1-20:
    - Find the word token (one, two, ...)
    - Find the digit token (1, 2, ...)
    - Measure cosine(word_emb, digit_emb)
    - Rank of digit in nn(word) over full vocab subset
    - Are word and digit nearest neighbors?

  Also test: is there a consistent direction word→digit across numbers?
  This would indicate TYPE_BC encoding. If no consistent direction but
  high proximity → TYPE_ADJACENT (same concept, different script).

EXPERIMENT C: Combined pipeline improvement
  Add attribute-axis antonym retrieval to the pipeline.
  Measure accuracy improvement over proximity-only antonym retrieval.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day210_special_cases.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Antonym pairs grouped by semantic attribute axis
ANTONYM_AXES = {
    "temperature": [("hot","cold"),("warm","cool"),("burning","freezing")],
    "size":        [("big","small"),("large","tiny"),("huge","little"),
                    ("tall","short"),("wide","narrow"),("thick","thin")],
    "speed":       [("fast","slow"),("quick","lazy"),("rapid","sluggish")],
    "volume":      [("loud","quiet"),("noisy","silent"),("sharp","soft")],
    "age":         [("old","young"),("ancient","modern"),("mature","new")],
    "brightness":  [("light","dark"),("bright","dim"),("clear","murky")],
    "sharpness":   [("sharp","dull"),("keen","blunt"),("acute","obtuse")],
    "texture":     [("hard","soft"),("rough","smooth"),("rigid","flexible")],
    "wealth":      [("rich","poor"),("wealthy","broke"),("lavish","sparse")],
    "emotion":     [("happy","sad"),("joyful","miserable"),("glad","angry")],
}

# All antonym pairs (training) and test pairs
ANTONYM_TRAIN = [
    ("hot","cold"),("big","small"),("fast","slow"),
    ("hard","soft"),("light","dark"),("old","young"),
    ("loud","quiet"),("sharp","dull"),("rich","poor"),
    ("thick","thin"),("wide","narrow"),("deep","shallow"),
]

# Numbers: word form → digit form
NUMBERS = [
    ("one","1"),("two","2"),("three","3"),("four","4"),("five","5"),
    ("six","6"),("seven","7"),("eight","8"),("nine","9"),("ten","10"),
    ("eleven","11"),("twelve","12"),("thirteen","13"),("fourteen","14"),
    ("fifteen","15"),("sixteen","16"),("seventeen","17"),("eighteen","18"),
    ("nineteen","19"),("twenty","20"),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a,b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
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

def tid1_bare(word):
    """Try without leading space (for digits like '1', '2')."""
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_emb(word, bare=False):
    t = (tid1_bare if bare else tid1)(word)
    return W_E[t].astype(np.float64) if t is not None else None

# ── EXPERIMENT A: Antonym Semantic Axes ──────────────────────────────
print("=" * 70)
print("EXPERIMENT A: ANTONYM SEMANTIC AXES")
print("=" * 70)

# 1. Compute per-attribute axis vectors
print("\n1. Per-attribute axis vectors (positive − negative direction):")
axis_vectors = {}
for attr, pairs in ANTONYM_AXES.items():
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    if not ok: continue
    # axis = mean of (pos_emb - neg_emb) normalized
    diffs = [normed(get_emb(a) - get_emb(b)) for a,b in ok]
    axis  = normed(np.mean(diffs, axis=0))
    axis_vectors[attr] = axis
    # Self-consistency
    if len(ok) >= 2:
        sc = np.mean([cosine(diffs[i], diffs[j])
                      for i in range(len(diffs))
                      for j in range(i+1, len(diffs))])
    else:
        sc = 1.0
    print(f"  {attr:<14}: n={len(ok)}  self-consistency={sc:.3f}  "
          + ("ok_pairs: " + ", ".join(f"{a}/{b}" for a,b in ok[:2])))

# 2. Pairwise cosines between attribute axes
print("\n2. Cross-attribute axis cosines (are axes distinct?):")
attrs = list(axis_vectors.keys())
cos_matrix = np.zeros((len(attrs), len(attrs)))
for i, a in enumerate(attrs):
    for j, b in enumerate(attrs):
        cos_matrix[i,j] = cosine(axis_vectors[a], axis_vectors[b])
header = " " * 14 + "  ".join(f"{a[:6]:>6}" for a in attrs)
print(f"  {header}")
for i, a in enumerate(attrs):
    row = f"  {a:<14}"
    for j in range(len(attrs)):
        row += f"  {cos_matrix[i,j]:>6.3f}"
    print(row)

mean_off_diag = np.mean([cos_matrix[i,j] for i in range(len(attrs))
                          for j in range(len(attrs)) if i != j])
print(f"\n  Mean off-diagonal cosine: {mean_off_diag:.3f}")
print(f"  (0.000 = orthogonal axes, 1.000 = same axis)")

# 3. Antonym retrieval using attribute axis
# Build retrieval vocab of antonym words only
ant_words = set()
for pairs in ANTONYM_AXES.values():
    for a,b in pairs:
        if tid1(a): ant_words.add(a)
        if tid1(b): ant_words.add(b)
for a,b in ANTONYM_TRAIN:
    if tid1(a): ant_words.add(a)
    if tid1(b): ant_words.add(b)
ant_vocab = {w: get_emb(w) for w in ant_words if get_emb(w) is not None}
print(f"\n3. Antonym retrieval vocabulary: {len(ant_vocab)} words")

def retrieve_axis(src, axis, vocab=ant_vocab):
    se = get_emb(src)
    if se is None: return None
    # Score = cosine(emb, axis) — highest score = most "positive" end
    # For antonym: subtract src's projection onto axis, then find opposite
    src_proj = float(np.dot(normed(se), axis))
    # Target should be at -src_proj along the axis
    target_dir = axis if src_proj < 0 else -axis
    query = normed(se + target_dir)
    sims  = {w: cosine(query, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

def retrieve_nn(src, vocab=ant_vocab):
    se = get_emb(src)
    if se is None: return None
    sims = {w: cosine(se, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

# Test: per-attribute axis retrieval vs proximity
print("\n4. Per-attribute axis retrieval vs proximity:")
print(f"  {'Attribute':<14}  {'n':>2}  {'axis_acc':>9}  {'nn_acc':>7}")
axis_results = {}
for attr, pairs in ANTONYM_AXES.items():
    if attr not in axis_vectors: continue
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    if not ok: continue
    axis = axis_vectors[attr]
    ax_c = sum(1 for a,b in ok if retrieve_axis(a, axis) == b)
    nn_c = sum(1 for a,b in ok if retrieve_nn(a) == b)
    ax_acc = ax_c / len(ok)
    nn_acc = nn_c / len(ok)
    print(f"  {attr:<14}  {len(ok):>2}  {ax_acc:>9.3f}  {nn_acc:>7.3f}")
    axis_results[attr] = {"n": len(ok), "axis_acc": ax_acc, "nn_acc": nn_acc}

# 5. Cross-attribute retrieval: use axis from ONE attribute on ANOTHER
print("\n5. Cross-attribute axis transfer (LOO on antonym domains):")
for src_attr in list(ANTONYM_AXES.keys())[:5]:
    if src_attr not in axis_vectors: continue
    ax = axis_vectors[src_attr]
    for tgt_attr in list(ANTONYM_AXES.keys())[:5]:
        if src_attr == tgt_attr or tgt_attr not in ANTONYM_AXES: continue
        ok = [(a,b) for a,b in ANTONYM_AXES[tgt_attr]
              if tid1(a) and tid1(b)]
        if not ok: continue
        cx = sum(1 for a,b in ok if retrieve_axis(a, ax) == b)
        nn = sum(1 for a,b in ok if retrieve_nn(a) == b)
        print(f"  {src_attr[:8]:<8}→{tgt_attr[:8]:<8}: "
              f"axis={cx/len(ok):.2f}  nn={nn/len(ok):.2f}  n={len(ok)}")

# ── EXPERIMENT B: Numbers Cross-Script ───────────────────────────────
print()
print("=" * 70)
print("EXPERIMENT B: NUMBERS CROSS-SCRIPT ENCODING")
print("=" * 70)

print("\n1. Single-token check for word and digit forms:")
num_ok = []
for word, digit in NUMBERS:
    tw = tid1(word)
    td = tid1_bare(digit)
    td2 = tid1(" " + digit) if not td else td  # try with space too
    tid_used = td if td else tid1(" " + digit)
    both_ok = tw is not None and tid_used is not None
    if both_ok: num_ok.append((word, digit, tid_used))
    print(f"  {word:>10} → {digit:<4}  "
          f"word_tok={tw is not None}  digit_tok={tid_used is not None}"
          + (f"  both single-token" if both_ok else ""))
print(f"\n  {len(num_ok)}/{len(NUMBERS)} number pairs both single-token\n")

if num_ok:
    print("2. Cosine similarity and nn_rank (word → digit):")
    print(f"  {'word':>10} → {'digit':<4}  {'cos':>6}  "
          f"{'rank_in_digits':>14}  {'rank_in_all':>11}")
    # Digit vocab for ranking
    digit_vocab = {}
    for _, d, td in num_ok:
        digit_vocab[d] = W_E[td].astype(np.float64)
    sims_data = []
    for word, digit, td in num_ok:
        we = get_emb(word)
        de = W_E[td].astype(np.float64)
        if we is None: continue
        c = cosine(we, de)
        # Rank among digit tokens only
        digit_sims = {d: cosine(we, digit_vocab[d]) for d in digit_vocab}
        d_ranked = sorted(digit_sims, key=lambda d: digit_sims[d], reverse=True)
        d_rank = d_ranked.index(digit) if digit in d_ranked else len(d_ranked)
        sims_data.append((word, digit, c, d_rank))
        print(f"  {word:>10} → {digit:<4}  {c:>6.3f}  "
              f"{d_rank:>14}  (digit rank)")

    mean_cos = np.mean([c for _,_,c,_ in sims_data])
    mean_drank = np.mean([r for _,_,_,r in sims_data])
    print(f"\n  Mean cosine (word, digit): {mean_cos:.3f}")
    print(f"  Mean rank among digit tokens: {mean_drank:.2f}")

    print("\n3. Direction consistency of word→digit displacement:")
    if len(num_ok) >= 2:
        diffs = []
        for word, digit, td in num_ok:
            we = get_emb(word)
            de = W_E[td].astype(np.float64)
            if we is not None:
                diffs.append(normed(de - we))
        pw = [cosine(diffs[i], diffs[j])
              for i in range(len(diffs))
              for j in range(i+1, len(diffs))]
        dc = float(np.mean(pw))
        print(f"  dir_consistency: {dc:.4f}")
        if dc > 0.15:
            print("  → DIRECTIONAL: word→digit has a consistent direction")
        else:
            print("  → NOT DIRECTIONAL: word→digit has no consistent direction")

    print("\n4. Nearest digit neighbor of each word number:")
    for word, digit, td in num_ok[:10]:
        we = get_emb(word)
        if we is None: continue
        digit_sims = sorted([(d, cosine(we, digit_vocab[d]))
                              for d in digit_vocab],
                            key=lambda x: x[1], reverse=True)
        top3 = ", ".join(f"{d}({s:.3f})" for d,s in digit_sims[:3])
        print(f"  nn_digit({word:>10}) → [{top3}]  "
              + ("✓" if digit_sims[0][0] == digit else "✗"))

# ── EXPERIMENT C: Pipeline with attribute axis ────────────────────────
print()
print("=" * 70)
print("EXPERIMENT C: ANTONYM PIPELINE IMPROVEMENT")
print("=" * 70)
print("  Strategy: use ALL available antonym axis vectors as an ensemble")
print("  For each query word, detect which attribute axis to use via:")
print("    max_attr = argmax |src_projection_onto_axis|")
print("  Then flip along that axis.\n")

# Combined axis retrieval: find best-fitting axis for each query
def retrieve_best_axis(src, axis_vecs, vocab=ant_vocab):
    se = get_emb(src)
    if se is None: return None, None
    sn = normed(se)
    # Find axis with largest absolute projection
    best_attr = None; best_proj = 0.0
    for attr, ax in axis_vecs.items():
        p = abs(float(np.dot(sn, ax)))
        if p > best_proj:
            best_proj = p; best_attr = attr
    if best_attr is None: return retrieve_nn(src, vocab), "nn"
    axis = axis_vecs[best_attr]
    src_proj = float(np.dot(sn, axis))
    target_dir = axis if src_proj < 0 else -axis
    query = normed(se + target_dir)
    sims  = {w: cosine(query, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w]), best_attr

correct_axis = 0; correct_nn = 0; n_test = 0
print(f"  {'Src':>8} → {'Tgt':<8}  {'Best axis':<14}  "
      f"{'axis_pred':<10}  {'nn_pred':<10}  {'axis?':>6}  {'nn?':>4}")
for src, tgt in ANTONYM_TRAIN:
    if not tid1(src) or not tid1(tgt): continue
    pred_axis, attr = retrieve_best_axis(src, axis_vectors)
    pred_nn   = retrieve_nn(src)
    ax_ok = (pred_axis == tgt)
    nn_ok = (pred_nn  == tgt)
    if ax_ok: correct_axis += 1
    if nn_ok: correct_nn   += 1
    n_test += 1
    print(f"  {src:>8} → {tgt:<8}  {str(attr):<14}  "
          f"{str(pred_axis):<10}  {str(pred_nn):<10}  "
          f"{'✓' if ax_ok else '✗':>6}  {'✓' if nn_ok else '✗':>4}")

if n_test > 0:
    print(f"\n  Axis ensemble:   {correct_axis}/{n_test} = {correct_axis/n_test:.3f}")
    print(f"  Proximity (nn):  {correct_nn}/{n_test}  = {correct_nn/n_test:.3f}")
    print(f"  Improvement:     {(correct_axis-correct_nn)/n_test:+.3f}")

# Save
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "axis_results": axis_results,
        "num_ok_count": len(num_ok) if num_ok else 0,
        "antonym_axis_acc":    correct_axis/n_test if n_test else 0,
        "antonym_nn_acc":      correct_nn/n_test   if n_test else 0,
    }, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 210 complete.")
