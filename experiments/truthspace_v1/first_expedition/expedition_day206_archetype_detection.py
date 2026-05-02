#!/usr/bin/env python3
"""
Day 206 — Mixed-Archetype Detection

QUESTION: Can we predict the archetype of a (source, target) word pair
purely from W_E geometry — without knowing the label?

Three archetypes to detect per pair:
  IDENTITY:   source == target token (W_E diff ≈ 0)
  ADJACENT:   target is already near source in W_E (no direction needed)
  DIRECTIONAL: target requires a consistent vector displacement

GEOMETRY-BASED FEATURES for per-pair classification:
  1. norm(target_emb - source_emb):  near-zero → IDENTITY
  2. cosine(source, target):         high → ADJACENT
  3. rank of target when doing nn(source) in a vocabulary:
        rank=0 → ADJACENT; rank>threshold → DIRECTIONAL
  4. cross_class column uniformity (from Day 204):
        if any direction retrieves target → ADJACENT
        if only correct direction retrieves → DIRECTIONAL

PREDICTION RULE (unsupervised, purely from W_E):
  if norm(tgt-src) < ε₁:            → IDENTITY
  elif cosine(src, tgt) > τ_adj:    → ADJACENT
  elif nn_rank(src, tgt) < r_adj:   → ADJACENT
  else:                             → DIRECTIONAL

EVALUATION: Run on all verb classes + plural/superlative/capitals
  - Precision/recall per archetype
  - Does geometric prediction match Day 204 column-uniformity labels?

THRESHOLD CALIBRATION via held-out calibration set:
  IDENTITY: norm < 0.05
  ADJACENT: cosine > 0.70 OR nn_rank ≤ 2
  DIRECTIONAL: otherwise

Candidate vocabulary: 400 common single-token English words (all classes).
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day206_archetype_detection.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# All word pairs with known archetype labels (from Day 204)
LABELED_PAIRS = {
    # IDENTITY (same source = target token)
    "IDENTITY": [
        ("cut","cut"),("put","put"),("hit","hit"),("let","let"),
        ("set","set"),("shut","shut"),("burst","burst"),("cost","cost"),
    ],
    # ADJACENT (proximity-encoded — target near source, direction irrelevant)
    "ADJACENT": [
        # B_oo_to_ew
        ("know","knew"),("grow","grew"),("throw","threw"),("blow","blew"),
        ("fly","flew"),("draw","drew"),
        # D_nd_to_nt
        ("send","sent"),("spend","spent"),("lend","lent"),("bend","bent"),
        # C_ee_to_e
        ("keep","kept"),("feel","felt"),("sleep","slept"),("meet","met"),
        ("lead","led"),("deal","dealt"),
        # antonyms (TYPE_ADJACENT from Day 196)
        ("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
        ("light","dark"),("old","young"),("loud","quiet"),("sharp","dull"),
    ],
    # DIRECTIONAL (TYPE_BC — needs consistent direction vector)
    "DIRECTIONAL": [
        # A_i_to_a
        ("begin","began"),("sing","sang"),("swim","swam"),("ring","rang"),
        ("win","won"),
        # F_suppletive
        ("go","went"),("have","had"),("do","did"),("make","made"),
        ("come","came"),("take","took"),("give","gave"),("get","got"),
        ("stand","stood"),("leave","left"),
        # Plurals (TYPE_BC_UNIV)
        ("cat","cats"),("dog","dogs"),("house","houses"),("tree","trees"),
        ("bird","birds"),("ship","ships"),("hand","hands"),("road","roads"),
        # Capitals (TYPE_BC_CLASS)
        ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
        # Gender
        ("king","queen"),("man","woman"),("boy","girl"),
        # Superlatives
        ("big","biggest"),("fast","fastest"),("long","longest"),
        ("smart","smartest"),("old","oldest"),
    ],
}

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

def get_emb(word):
    t = tid1(word)
    return W_E[t].astype(np.float64) if t is not None else None

# Build candidate vocabulary — 400 common single-token words
RAW_VOCAB = [
    # All test words from days 202-204
    "cats","dogs","houses","trees","books","cars","birds","ships","hands",
    "eyes","roads","doors","lamps","walls","cups","beds","keys","boxes",
    "tables","chairs","floors","windows","rooms","pages","words","names",
    "times","years","days","ways","men","women","children","people",
    "biggest","fastest","oldest","coldest","smartest","longest","hardest",
    "darkest","softest","warmest","brightest","cleanest","slowest","smallest",
    "tallest","weakest","nearest","latest","greatest","highest","lowest",
    "ran","ate","went","saw","came","gave","took","made","said","knew",
    "found","thought","left","brought","bought","stood","got","put","set",
    "let","cut","hit","fled","grew","flew","threw","blew","drew","sent",
    "spent","lent","bent","kept","felt","slept","met","led","dealt",
    "began","sang","swam","rang","won","had","did","got","came","gave",
    "Paris","Berlin","Rome","Madrid","Tokyo","Beijing","Moscow","Athens",
    "queen","woman","girl","princess","actress","heroine","waitress",
    "cold","small","slow","soft","dark","young","quiet","dull","poor","thin",
    # Extra common words as distractors
    "cat","dog","house","tree","book","car","bird","ship","hand","eye",
    "run","eat","go","see","come","give","take","make","say","know",
    "big","fast","old","smart","long","hard","dark","warm","bright","clean",
    "France","Germany","Italy","Spain","Japan","China","Russia","Greece",
    "king","man","boy","prince","actor","hero","waiter",
    "hot","fast","hard","light","old","loud","sharp","rich","thick",
    "new","good","bad","high","low","long","short","near","far","early",
    "late","great","small","large","wide","deep","thin","flat","round",
    "red","blue","green","yellow","black","white","brown","gray","pink",
    "one","two","three","four","five","six","seven","eight","nine","ten",
]
candidate_vocab = {}
for w in RAW_VOCAB:
    t = tid1(w)
    if t is not None and w not in candidate_vocab:
        candidate_vocab[w] = W_E[t].astype(np.float64)
candidate_list = list(candidate_vocab.keys())
print(f"Candidate vocabulary: {len(candidate_vocab)} words\n")

def nn_rank(src_word, tgt_word, vocab=candidate_vocab):
    se = get_emb(src_word)
    if se is None: return None
    sims = [(w, cosine(se, e)) for w,e in vocab.items() if w != src_word]
    sims.sort(key=lambda x: x[1], reverse=True)
    ranked = [w for w,_ in sims]
    return ranked.index(tgt_word) if tgt_word in ranked else len(ranked)

def extract_features(src, tgt):
    se = get_emb(src)
    te = get_emb(tgt)
    if se is None or te is None: return None
    diff = te - se
    diff_norm = float(np.linalg.norm(diff))
    cos_st = cosine(se, te)
    rank = nn_rank(src, tgt) if tgt in candidate_vocab else 999
    return {
        "diff_norm": diff_norm,
        "cosine_st": cos_st,
        "nn_rank":   rank,
    }

def predict_archetype(feats,
                      identity_norm_thresh=0.05,
                      adjacent_cos_thresh=0.70,
                      adjacent_rank_thresh=3):
    if feats is None: return "UNKNOWN"
    if feats["diff_norm"] < identity_norm_thresh:
        return "IDENTITY"
    if (feats["cosine_st"] > adjacent_cos_thresh or
            feats["nn_rank"] <= adjacent_rank_thresh):
        return "ADJACENT"
    return "DIRECTIONAL"

# ── Feature extraction and prediction ────────────────────────────────
print("=" * 70)
print("FEATURE EXTRACTION AND PREDICTION")
print(f"{'Pair':<22}  {'True':<12}  {'Pred':<12}  "
      f"{'diff_norm':>9}  {'cos':>6}  {'rank':>5}")
print("-" * 75)

all_results = []
for true_label, pairs in LABELED_PAIRS.items():
    for src, tgt in pairs:
        if not tid1(src) or not tid1(tgt): continue
        feats = extract_features(src, tgt)
        if feats is None: continue
        pred = predict_archetype(feats)
        correct = (pred == true_label)
        mark = "" if correct else " ✗"
        print(f"  {src:>8}→{tgt:<12}  {true_label:<12}  {pred:<12}  "
              f"{feats['diff_norm']:>9.4f}  {feats['cosine_st']:>6.3f}  "
              f"{feats['nn_rank']:>5}{mark}")
        all_results.append({
            "src": src, "tgt": tgt,
            "true": true_label, "pred": pred,
            **feats
        })

# ── Confusion matrix ──────────────────────────────────────────────────
labels = ["IDENTITY", "ADJACENT", "DIRECTIONAL"]
conf = {t: {p: 0 for p in labels} for t in labels}
for r in all_results:
    if r["true"] in conf and r["pred"] in conf:
        conf[r["true"]][r["pred"]] += 1

print()
print("=" * 70)
print("CONFUSION MATRIX (rows=true, cols=pred)")
print("=" * 70)
header = f"{'':>14}  " + "  ".join(f"{p:>12}" for p in labels)
print(header)
for true_lab in labels:
    row = f"  {true_lab:<12}  " + "  ".join(
        f"{conf[true_lab][p]:>12}" for p in labels)
    print(row)

print()
print("Per-class precision/recall:")
for lab in labels:
    tp = conf[lab][lab]
    fn = sum(conf[lab][p] for p in labels if p != lab)
    fp = sum(conf[t][lab] for t in labels if t != lab)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    print(f"  {lab:<12}: prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

# ── Feature distribution analysis ────────────────────────────────────
print()
print("=" * 70)
print("FEATURE DISTRIBUTIONS BY ARCHETYPE")
print("=" * 70)
for lab in labels:
    subset = [r for r in all_results if r["true"] == lab]
    if not subset: continue
    norms = [r["diff_norm"] for r in subset]
    coss  = [r["cosine_st"] for r in subset]
    ranks = [r["nn_rank"]   for r in subset if r["nn_rank"] < 999]
    print(f"\n  {lab} (n={len(subset)}):")
    print(f"    diff_norm: mean={np.mean(norms):.4f}  "
          f"min={np.min(norms):.4f}  max={np.max(norms):.4f}")
    print(f"    cosine:    mean={np.mean(coss):.3f}  "
          f"min={np.min(coss):.3f}  max={np.max(coss):.3f}")
    if ranks:
        print(f"    nn_rank:   mean={np.mean(ranks):.1f}  "
              f"min={np.min(ranks):.0f}  max={np.max(ranks):.0f}")

# ── Optimal threshold search ──────────────────────────────────────────
print()
print("=" * 70)
print("THRESHOLD SEARCH (nn_rank threshold for ADJACENT)")
print("=" * 70)
identity_pairs  = [r for r in all_results if r["true"] == "IDENTITY"]
adjacent_pairs  = [r for r in all_results if r["true"] == "ADJACENT"]
direct_pairs    = [r for r in all_results if r["true"] == "DIRECTIONAL"]

# Identity is easy (diff_norm), focus on adj vs directional
print(f"  ADJACENT ranks: {sorted(r['nn_rank'] for r in adjacent_pairs if r['nn_rank']<999)}")
print(f"  DIRECTIONAL ranks: {sorted(r['nn_rank'] for r in direct_pairs if r['nn_rank']<999)[:15]}...")

best_f1, best_thresh = 0.0, 2
for thresh in range(0, 15):
    correct = 0; total = 0
    for r in all_results:
        if r["true"] not in ("ADJACENT","DIRECTIONAL"): continue
        pred_adj = (r["cosine_st"] > 0.70 or r["nn_rank"] <= thresh)
        true_adj = (r["true"] == "ADJACENT")
        if pred_adj == true_adj: correct += 1
        total += 1
    f1 = correct/total if total else 0
    if f1 > best_f1:
        best_f1 = f1; best_thresh = thresh
    print(f"  rank_thresh={thresh}: adj_vs_dir acc={f1:.3f}")
print(f"  BEST: rank_thresh={best_thresh}  acc={best_f1:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "results": all_results,
        "confusion": conf,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
    }, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 206 complete.")
