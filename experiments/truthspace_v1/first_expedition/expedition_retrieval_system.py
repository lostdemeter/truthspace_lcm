#!/usr/bin/env python3
"""
Days 255-256 — TruthSpace Multi-Paradigm Geometric Retrieval System

A minimal system that answers morphological analogy queries using ONLY
W_E geometry (no transformer forward pass beyond embedding lookup).

Query format: given (A, B, C) — find D such that A:B :: C:D
where the A→B transformation is one of the known paradigms.

System components:
1. Paradigm axis library (mean_dir + calibrated scale, built from training pairs)
2. Paradigm identifier (project chord A→B onto each axis, pick highest)
3. Retrieval (C + scale * mean_dir → nearest neighbour)

Day 256: per-paradigm scale calibration via LOO on training set.
  adj_degree=1.0, plural=0.8, past_tense=1.5
  Overall accuracy: 49/53 = 92.5% (vs 45/53=84.9% with scale=1 for all)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "retrieval_system.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Paradigm training pairs ───────────────────────────────────────────
# These are the training pairs used to build the mean_dir library.
# Test pairs are held out (not in these lists).

TRAIN = {
    "adj_degree": [
        ("big","bigger"), ("fast","faster"), ("long","longer"), ("hot","hotter"),
        ("cold","colder"), ("tall","taller"), ("bright","brighter"), ("dark","darker"),
        ("strong","stronger"), ("weak","weaker"), ("hard","harder"), ("soft","softer"),
        ("wide","wider"), ("deep","deeper"), ("thick","thicker"), ("thin","thinner"),
        ("loud","louder"), ("clean","cleaner"), ("clear","clearer"), ("safe","safer"),
    ],
    "plural": [
        ("cat","cats"), ("dog","dogs"), ("book","books"), ("car","cars"),
        ("tree","trees"), ("bird","birds"), ("hand","hands"), ("eye","eyes"),
        ("word","words"), ("world","worlds"), ("year","years"), ("day","days"),
        ("time","times"), ("part","parts"), ("room","rooms"), ("door","doors"),
        ("name","names"), ("line","lines"), ("place","places"), ("face","faces"),
    ],
    "past_tense": [
        ("walk","walked"), ("talk","talked"), ("play","played"), ("work","worked"),
        ("help","helped"), ("call","called"), ("move","moved"), ("turn","turned"),
        ("love","loved"), ("use","used"), ("start","started"), ("ask","asked"),
        ("live","lived"), ("want","wanted"), ("need","needed"), ("wait","waited"),
        ("open","opened"), ("close","closed"), ("show","showed"), ("stop","stopped"),
    ],
    "gender": [
        ("king","queen"), ("man","woman"), ("boy","girl"), ("actor","actress"),
        ("hero","heroine"), ("prince","princess"), ("lion","lioness"), ("duke","duchess"),
    ],
    "superlative": [
        ("big","biggest"), ("fast","fastest"), ("long","longest"), ("tall","tallest"),
        ("old","oldest"), ("hot","hottest"), ("cold","coldest"), ("dark","darkest"),
        ("clean","cleanest"), ("clear","clearest"),
    ],
}

# ── Test pairs (held-out, not in TRAIN) ───────────────────────────────
TEST = {
    "adj_degree": [
        ("old","older"), ("young","younger"), ("small","smaller"), ("short","shorter"),
        ("nice","nicer"), ("rich","richer"), ("poor","poorer"), ("cheap","cheaper"),
        ("rough","rougher"), ("tough","tougher"), ("smart","smarter"), ("cool","cooler"),
        ("warm","warmer"), ("great","greater"), ("fine","finer"), ("sharp","sharper"),
        ("quiet","quieter"), ("quick","quicker"), ("broad","broader"), ("plain","plainer"),
    ],
    "plural": [
        ("city","cities"), ("man","men"), ("woman","women"), ("child","children"),
        ("flower","flowers"), ("house","houses"), ("chair","chairs"), ("table","tables"),
        ("street","streets"), ("town","towns"), ("phone","phones"), ("stone","stones"),
        ("boat","boats"), ("horse","horses"), ("river","rivers"), ("mountain","mountains"),
        ("garden","gardens"), ("letter","letters"), ("color","colors"), ("shape","shapes"),
    ],
    "past_tense": [
        ("like","liked"), ("watch","watched"), ("climb","climbed"), ("add","added"),
        ("fill","filled"), ("pull","pulled"), ("pick","picked"), ("reach","reached"),
        ("point","pointed"), ("vote","voted"), ("drop","dropped"), ("pass","passed"),
        ("raise","raised"), ("land","landed"), ("mark","marked"), ("burn","burned"),
        ("hunt","hunted"), ("form","formed"), ("act","acted"), ("record","recorded"),
    ],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
V, H = W_E.shape
print(f"  V={V}, H={H}")

W_n = np.array([normed(W_E[i]) for i in range(V)], dtype=np.float32)
print("  Normalized W_E built.\n")

def tid1(w):
    for pref in [" ", ""]:
        ids = tok(pref + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: return ids[0]
    return None

def get_emb(w):
    t = tid1(w)
    return W_E[t].copy() if t is not None else None

# ── Per-paradigm calibrated scales (Day 256, LOO-optimized) ─────────
PARADIGM_SCALES = {
    "adj_degree":  1.0,
    "plural":      0.8,
    "past_tense":  1.5,
    "gender":      1.0,
    "superlative": 1.0,
}

# ── Build paradigm mean_dir library ──────────────────────────────────
print("Building paradigm mean_dir library from training pairs ...")
library = {}
for paradigm, pairs in TRAIN.items():
    chords = []
    for src, tgt in pairs:
        es, et = get_emb(src), get_emb(tgt)
        if es is None or et is None: continue
        chords.append(et - es)
    if len(chords) == 0: continue
    mean_dir = normed(np.mean(chords, axis=0))
    library[paradigm] = mean_dir
    scale = PARADIGM_SCALES.get(paradigm, 1.0)
    print(f"  {paradigm:<14}: {len(chords)} training pairs, scale={scale}")

# ── Paradigm identification ───────────────────────────────────────────
def identify_paradigm(src, tgt):
    """Given a (src, tgt) pair, identify the paradigm by aligning the chord."""
    es, et = get_emb(src), get_emb(tgt)
    if es is None or et is None: return None, {}
    chord = normed(et - es)
    scores = {p: float(np.dot(chord, d)) for p, d in library.items()}
    best = max(scores, key=scores.get)
    return best, scores

# ── Retrieval ─────────────────────────────────────────────────────────
def retrieve(query_emb, excl_tids, paradigm=None):
    """Predict target: query_emb + scale * mean_dir[paradigm] → NN."""
    if paradigm is None: return None
    direction = library[paradigm]
    scale = PARADIGM_SCALES.get(paradigm, 1.0)
    pred = query_emb + scale * direction
    pred_n = normed(pred).astype(np.float32)
    sims = W_n @ pred_n
    for t in excl_tids:
        if t is not None: sims[t] = -2.0
    return int(np.argmax(sims))

# ── Evaluation ───────────────────────────────────────────────────────
print()
print("=" * 75)
print("EVALUATION: Held-out test pairs")
print("=" * 75)
print()

results = {}
total_correct = 0
total_pairs   = 0

for paradigm, pairs in TEST.items():
    correct_oracle   = 0  # using correct paradigm label
    correct_inferred = 0  # using inferred paradigm
    correct_wrong    = 0  # using wrong paradigm (control)
    skipped          = 0
    paradigm_correct = 0
    n_valid          = 0

    # For control: pick the "other" paradigm (first non-matching)
    others = [p for p in library if p != paradigm]
    other_paradigm = others[0] if others else None

    for src, tgt in pairs:
        es = get_emb(src)
        t_tgt = tid1(tgt)
        t_src = tid1(src)
        if es is None or t_tgt is None:
            skipped += 1; continue

        n_valid += 1

        # Oracle: use correct paradigm
        pred_oracle = retrieve(es, [t_src], paradigm)
        if pred_oracle == t_tgt: correct_oracle += 1

        # Inferred: identify paradigm from (src, tgt) and use that
        inferred, scores = identify_paradigm(src, tgt)
        pred_inferred = retrieve(es, [t_src], inferred)
        if pred_inferred == t_tgt: correct_inferred += 1
        if inferred == paradigm: paradigm_correct += 1

        # Wrong paradigm (control)
        if other_paradigm:
            pred_wrong = retrieve(es, [t_src], other_paradigm)
            if pred_wrong == t_tgt: correct_wrong += 1

    oracle_acc   = correct_oracle   / n_valid if n_valid > 0 else 0
    inferred_acc = correct_inferred / n_valid if n_valid > 0 else 0
    wrong_acc    = correct_wrong    / n_valid if n_valid > 0 else 0
    paradigm_id_acc = paradigm_correct / n_valid if n_valid > 0 else 0

    results[paradigm] = {
        "oracle": oracle_acc, "inferred": inferred_acc,
        "wrong": wrong_acc, "n": n_valid,
        "paradigm_id_acc": paradigm_id_acc,
    }

    print(f"  {paradigm:<14}: n={n_valid}")
    print(f"    Oracle   (correct paradigm):  {correct_oracle}/{n_valid} = {oracle_acc:.3f}")
    print(f"    Inferred (auto paradigm id):   {correct_inferred}/{n_valid} = {inferred_acc:.3f}")
    print(f"    Wrong    (other paradigm):     {correct_wrong}/{n_valid} = {wrong_acc:.3f}")
    print(f"    Paradigm ID accuracy:          {paradigm_correct}/{n_valid} = {paradigm_id_acc:.3f}")
    print()

    total_correct += correct_inferred
    total_pairs   += n_valid

print(f"  OVERALL (inferred): {total_correct}/{total_pairs} = {total_correct/total_pairs:.3f}")

# ── Cross-paradigm inference table ───────────────────────────────────
print()
print("=" * 75)
print("PARADIGM IDENTIFICATION ACCURACY (A:B pair → correct paradigm?)")
print("=" * 75)
print()

for paradigm, pairs in {**TRAIN, **TEST}.items():
    if paradigm not in library: continue
    counts = {p: 0 for p in library}
    n_valid = 0
    for src, tgt in pairs:
        es, et = get_emb(src), get_emb(tgt)
        if es is None or et is None: continue
        n_valid += 1
        inferred, _ = identify_paradigm(src, tgt)
        if inferred in counts: counts[inferred] += 1
    if n_valid == 0: continue
    pct_correct = counts[paradigm] / n_valid
    other_str = ", ".join(f"{p[:4]}:{c}" for p, c in counts.items() if p != paradigm and c > 0)
    print(f"  {paradigm:<14}: {counts[paradigm]}/{n_valid} = {pct_correct:.2f} correct"
          + (f"  (other: {other_str})" if other_str else ""))

# ── Analogy task format ────────────────────────────────────────────────
print()
print("=" * 75)
print("ANALOGY TASK: A:B :: C:?  (uses inferred paradigm from A:B)")
print("=" * 75)
print()

ANALOGIES = [
    ("big","bigger","tall","?","taller"),
    ("walk","walked","play","?","played"),
    ("cat","cats","dog","?","dogs"),
    ("fast","faster","slow","?","slower"),
    ("love","loved","use","?","used"),
    ("house","houses","tree","?","trees"),
    ("kind","kinder","smart","?","smarter"),
    ("open","opened","close","?","closed"),
    ("long","longer","short","?","shorter"),
    ("woman","women","man","?","men"),
]

correct_analogy = 0
for A, B, C, _, expected in ANALOGIES:
    eC = get_emb(C)
    t_expected = tid1(expected)
    t_C = tid1(C)
    if eC is None or t_expected is None: continue

    inferred, scores = identify_paradigm(A, B)
    pred_tid = retrieve(eC, [t_C], inferred)
    pred_word = tok.decode([pred_tid]).strip() if pred_tid is not None else "?"
    ok = pred_tid == t_expected
    if ok: correct_analogy += 1
    status = "✓" if ok else "✗"
    print(f"  {A}:{B} :: {C}:? → {pred_word:>12} (expected: {expected})  [{status}]  "
          f"paradigm={inferred}")

print(f"\n  Analogy accuracy: {correct_analogy}/{len(ANALOGIES)} = {correct_analogy/len(ANALOGIES):.2f}")

# ── Save ──────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "test_results": {k: {**v, "n": int(v["n"])} for k, v in results.items()},
        "total_inferred_acc": total_correct / total_pairs if total_pairs > 0 else 0,
        "analogy_acc": correct_analogy / len(ANALOGIES),
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Retrieval system evaluation complete.")
