#!/usr/bin/env python3
"""
Day 204 — Irregular Verb Inflection Classes

QUESTION: Is TYPE_BC_CLASS (past_tense) actually multiple distinct
geometric classes, one per inflection pattern?

English irregular past tense falls into well-known phonological classes:
  Class A — vowel change i→a:   begin/began, sing/sang, swim/swam, ring/rang
  Class B — vowel change i→u:   build/built, drink/drank, shrink/shrank (complex)
  Class C — vowel change ee→e:  keep/kept, feel/felt, sleep/slept, mean/meant
  Class D — vowel change ee→aw: see/saw, flee/fled
  Class E — vowel change oo→ew: know/knew, grow/grew, throw/threw, blow/blew
  Class F — suppletive/unique:  go/went, be/was, have/had, do/did, say/said
  Class G — no change:          cut/cut, put/put, hit/hit, let/let, set/set
  Class H — -d/-t ending:       build/built, send/sent, spend/spent, lend/lent

For each class, we ask:
  1. Is there a consistent TYPE_BC direction within the class?
  2. Does within-class direction transfer work?
  3. Does cross-class direction transfer fail?

METHOD:
  For each class with ≥3 single-token pairs:
    - Compute direction consistency (pairwise cosine of diffs)
    - LOO accuracy within class
    - Apply class direction to members of OTHER classes → expect low accuracy

All pairs where both source AND target are single tokens only.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day204_verb_classes.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

VERB_CLASSES = {
    "A_i_to_a":   [("begin","began"),("sing","sang"),("swim","swam"),
                   ("ring","rang"),("spring","sprang"),("drink","drank"),
                   ("sink","sank"),("spin","span"),("win","won")],
    "B_oo_to_ew": [("know","knew"),("grow","grew"),("throw","threw"),
                   ("blow","blew"),("fly","flew"),("draw","drew")],
    "C_ee_to_e":  [("keep","kept"),("feel","felt"),("sleep","slept"),
                   ("meet","met"),("flee","fled"),("read","read"),
                   ("lead","led"),("deal","dealt"),("mean","meant")],
    "D_nd_to_nt": [("send","sent"),("spend","spent"),("lend","lent"),
                   ("bend","bent"),("build","built"),("find","found")],
    "E_no_change": [("cut","cut"),("put","put"),("hit","hit"),
                    ("let","let"),("set","set"),("shut","shut"),
                    ("burst","burst"),("cost","cost")],
    "F_suppletive":[("go","went"),("have","had"),("do","did"),
                    ("say","said"),("make","made"),("come","came"),
                    ("take","took"),("give","gave"),("see","saw"),
                    ("get","got"),("buy","bought"),("think","thought"),
                    ("bring","brought"),("leave","left"),("stand","stood")],
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

def mean_direction(pairs):
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    if not ok: return None, 0
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in ok]
    return normed(np.mean(diffs, axis=0)), len(ok)

def dir_consistency(pairs):
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    if len(ok) < 2: return 0.0, 0
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in ok]
    pw = [cosine(diffs[i], diffs[j])
          for i in range(len(diffs)) for j in range(i+1, len(diffs))]
    return float(np.mean(pw)), len(ok)

def loo_accuracy(pairs, target_vocab=None):
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    if len(ok) < 2: return None, None
    tv = {w: get_emb(w) for _,w in ok} if target_vocab is None else target_vocab
    correct = 0; ranks = []
    for i, (qa, qb) in enumerate(ok):
        pool = [(a,b) for a,b in ok if a != qa]
        if not pool: continue
        diffs = [normed(get_emb(b) - get_emb(a)) for a,b in pool]
        mdir  = normed(np.mean(diffs, axis=0))
        query = get_emb(qa) + mdir
        sims  = {w: cosine(query, e) for w,e in tv.items() if w != qa}
        ranked = sorted(sims, key=lambda w: sims[w], reverse=True)
        if ranked[0] == qb: correct += 1
        rank = ranked.index(qb) if qb in ranked else len(ranked)
        ranks.append(rank)
    return correct/len(ok), float(np.mean(ranks))

def cross_class_accuracy(src_pairs, tgt_pairs, tgt_vocab=None):
    """Apply direction from src_pairs to sources in tgt_pairs."""
    src_ok = [(a,b) for a,b in src_pairs if tid1(a) and tid1(b)]
    tgt_ok = [(a,b) for a,b in tgt_pairs if tid1(a) and tid1(b)]
    if not src_ok or not tgt_ok: return None
    mdir, _ = mean_direction(src_ok)
    if mdir is None: return None
    tv = {w: get_emb(w) for _,w in tgt_ok} if tgt_vocab is None else tgt_vocab
    correct = 0
    for qa, qb in tgt_ok:
        se = get_emb(qa)
        if se is None: continue
        query = se + mdir
        sims  = {w: cosine(query, e) for w,e in tv.items() if w != qa}
        pred  = max(sims, key=lambda w: sims[w])
        if pred == qb: correct += 1
    return correct / len(tgt_ok)

# ── Filter to single-token pairs ──────────────────────────────────────
print("Single-token pairs per class:")
filtered = {}
for cls, pairs in VERB_CLASSES.items():
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    filtered[cls] = ok
    skipped = len(pairs) - len(ok)
    print(f"  {cls:<16}: {len(ok):>2}/{len(pairs)} single-token"
          + (f"  (skipped {skipped})" if skipped else ""))
print()

# ── Within-class analysis ─────────────────────────────────────────────
print("=" * 70)
print("WITHIN-CLASS DIRECTION CONSISTENCY AND LOO ACCURACY")
print("=" * 70)
results = {}
for cls, ok in filtered.items():
    if len(ok) < 2: continue
    dc, n = dir_consistency(ok)
    acc, mr = loo_accuracy(ok) if len(ok) >= 3 else (None, None)
    acc_str = f"{acc:.3f}" if acc is not None else "N/A"
    mr_str  = f"{mr:.2f}"  if mr  is not None else "N/A"
    print(f"  {cls:<16}  n={n:<2}  dir={dc:.3f}  acc={acc_str}  rank={mr_str}")
    results[cls] = {"n": n, "dir_consistency": dc,
                    "loo_accuracy": acc, "mean_rank": mr,
                    "pairs": ok}
print()

# ── Cross-class transfer matrix ───────────────────────────────────────
print("=" * 70)
print("CROSS-CLASS DIRECTION TRANSFER MATRIX")
print("  Rows = direction trained on, Cols = tested on")
print("=" * 70)
cls_list = [c for c,ok in filtered.items() if len(ok) >= 2]
# Header
header = "              " + "  ".join(f"{c[:8]:>8}" for c in cls_list)
print(header)
print("  " + "-" * (len(header) - 2))

cross_matrix = {}
for src_cls in cls_list:
    src_ok = filtered[src_cls]
    row = f"  {src_cls[:14]:<14}"
    cross_matrix[src_cls] = {}
    for tgt_cls in cls_list:
        tgt_ok = filtered[tgt_cls]
        if src_cls == tgt_cls:
            acc_val = results[src_cls]["loo_accuracy"]
            cell = f"  {'LOO':>6}" if acc_val is None else f"  {acc_val:>6.3f}"
        else:
            acc_val = cross_class_accuracy(src_ok, tgt_ok)
            cell = f"  {'N/A':>6}" if acc_val is None else f"  {acc_val:>6.3f}"
        row += cell
        cross_matrix[src_cls][tgt_cls] = acc_val
    print(row)
print()

# ── Within-class k=1 transfer ─────────────────────────────────────────
print("=" * 70)
print("WITHIN-CLASS k=1 TRANSFER (single exemplar → rest of class)")
print("=" * 70)
for cls, ok in filtered.items():
    if len(ok) < 3: continue
    k1_accs = []
    for i, (ta, tb) in enumerate(ok):
        mdir, _ = mean_direction([(ta, tb)])
        if mdir is None: continue
        tv = {b: get_emb(b) for _,b in ok}
        correct = 0
        for qa, qb in ok:
            if qa == ta: continue
            se = get_emb(qa)
            if se is None: continue
            query = se + mdir
            sims = {w: cosine(query, e) for w,e in tv.items() if w != qa}
            pred = max(sims, key=lambda w: sims[w])
            if pred == qb: correct += 1
        acc1 = correct / (len(ok) - 1)
        k1_accs.append((f"{ta}→{tb}", acc1))
    if k1_accs:
        best  = max(k1_accs, key=lambda x: x[1])
        worst = min(k1_accs, key=lambda x: x[1])
        mean  = np.mean([a for _,a in k1_accs])
        std   = np.std([a for _,a in k1_accs])
        print(f"  {cls:<16}: mean={mean:.3f} std={std:.3f}  "
              f"best={best[1]:.3f}({best[0]})  "
              f"worst={worst[1]:.3f}({worst[0]})")
print()

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY: Are irregular verb classes geometrically distinct?")
print("=" * 70)
for cls in cls_list:
    dc = results[cls]["dir_consistency"]
    acc = results[cls]["loo_accuracy"]
    label = ("COHESIVE" if dc >= 0.20 else
             "PARTIAL"  if dc >= 0.10 else "DIFFUSE")
    acc_str = f"{acc:.3f}" if acc is not None else "N/A"
    print(f"  {cls:<16}: {label:<9}  dir={dc:.3f}  loo={acc_str}")

# Diagonal vs off-diagonal in cross-class matrix
diag = [cross_matrix[c][c] for c in cls_list
        if cross_matrix[c][c] is not None]
off  = [cross_matrix[r][c] for r in cls_list for c in cls_list
        if r != c and cross_matrix[r][c] is not None]
if diag and off:
    print(f"\n  Mean diagonal (within-class): {np.mean(diag):.3f}")
    print(f"  Mean off-diagonal (cross-class): {np.mean(off):.3f}")
    print(f"  Diagonal advantage: {np.mean(diag) - np.mean(off):+.3f}")

with open(OUTPUT_FILE, "w") as f:
    serial = {c: {k: v for k,v in d.items() if k != "pairs"}
              for c,d in results.items()}
    serial["cross_matrix"] = cross_matrix
    json.dump(serial, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 204 complete.")
