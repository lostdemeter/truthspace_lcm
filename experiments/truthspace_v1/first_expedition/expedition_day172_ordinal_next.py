#!/usr/bin/env python3
"""
Day 172 — Ordinal 'Next' Prediction on the Number Line

DC 352 predicted: because PC0 of numbers IS the number line (r=0.989),
the direction one→two ≈ two→three ≈ three→four etc.
If so, a single "increment" direction should predict the successor of any number.

TESTS:
  T1: Is the 'next' direction consistent? (cosine between adjacent differences)
  T2: Using the 'next' direction (trained on k=2 pairs), predict successors LOO
  T3: Does the number line generalize to ordinals (first,second,third)?
  T4: Does it generalize to alphabet sequence (a,b,c,...)?
  T5: Does it generalize to months (January,February,...)?
  T6: Reverse direction: predecessor ('prev') prediction

HYPOTHESIS: The number line axis in W_E enables ordinal traversal.
  one + next_direction → two  (Type B: k=2 saturates at 100%)
  This would be a purely geometric arithmetic operation.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day172_ordinal_next.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Sequences ────────────────────────────────────────────────────
NUMBERS = ["one","two","three","four","five","six","seven","eight","nine","ten",
           "eleven","twelve","thirteen","fourteen","fifteen"]
ORDINALS = ["first","second","third","fourth","fifth","sixth","seventh","eighth",
            "ninth","tenth","eleventh","twelfth"]
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")

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

def make_next_dir(seq_ok, exclude_idx=None):
    """Mean normed difference of consecutive pairs (optionally exclude one)."""
    diffs = []
    for i in range(len(seq_ok)-1):
        if exclude_idx is not None and (i == exclude_idx or i+1 == exclude_idx):
            continue
        a, b = seq_ok[i], seq_ok[i+1]
        diffs.append(normed(W_E[tid(b)] - W_E[tid(a)]))
    return normed(np.mean(diffs, axis=0)) if diffs else None

def entity_from_vocab(src_emb, direction, vocab_embs, exclude):
    e = src_emb.copy()
    if direction is not None: e = e + direction
    cands = {w: cosine(e, v) for w, v in vocab_embs.items() if w not in exclude}
    top1 = max(cands, key=lambda w: cands[w])
    return top1, cands[top1]

# ─── Helper: test a sequence ─────────────────────────────────────
def test_sequence(name, words):
    ok = [w for w in words if tid(w)]
    if len(ok) < 4:
        print(f"  {name}: only {len(ok)} single-token words, skip")
        return {}
    vocab_embs = {w: W_E[tid(w)] for w in ok}

    print(f"\n{'='*64}")
    print(f"Sequence: {name}  ({len(ok)} single-token words)")
    print(f"  Words: {ok}")
    print()

    # T1: Consistency of 'next' direction
    diffs = [normed(W_E[tid(ok[i+1])] - W_E[tid(ok[i])]) for i in range(len(ok)-1)]
    cos_matrix = np.array([[cosine(diffs[i], diffs[j]) for j in range(len(diffs))]
                            for i in range(len(diffs))])
    mean_consistency = (cos_matrix.sum() - len(diffs)) / (len(diffs)*(len(diffs)-1)) if len(diffs) > 1 else 0
    print(f"  T1: Direction consistency (mean inter-diff cosine): {mean_consistency:.3f}")
    print(f"       Pairs: {[f'{ok[i]}→{ok[i+1]}' for i in range(len(ok)-1)]}")

    # T2: LOO 'next' prediction
    nc, n = 0, 0
    for i in range(len(ok)-1):
        src, tgt = ok[i], ok[i+1]
        # LOO: build direction excluding the pair (i, i+1)
        d = make_next_dir(ok, exclude_idx=i)
        if d is None: continue
        pred, score = entity_from_vocab(W_E[tid(src)], d, vocab_embs, {src})
        ok_pred = (pred == tgt)
        if ok_pred: nc += 1
        n += 1
        print(f"    {src:>12} + next_dir → pred: {pred:<12} target: {tgt}  "
              f"{'✓' if ok_pred else '✗'}  score={score:.3f}")
    acc = nc/n if n else 0
    print(f"\n  T2 LOO 'next' accuracy: {nc}/{n} = {acc:.3f}")

    # T3: Fixed-k directions (k=1,2,3 training pairs, test on rest)
    print()
    for k in [1, 2, 3]:
        if len(ok) < k+2: continue
        train_pairs = list(zip(ok[:k], ok[1:k+1]))
        test_pairs  = list(zip(ok[k:], ok[k+1:]))
        train_diffs = [normed(W_E[tid(b)] - W_E[tid(a)]) for a,b in train_pairs]
        d = normed(np.mean(train_diffs, axis=0))
        nc_k = sum(1 for a,b in test_pairs
                   if entity_from_vocab(W_E[tid(a)], d, vocab_embs, {a})[0] == b)
        acc_k = nc_k/len(test_pairs) if test_pairs else 0
        print(f"  k={k} (train first {k} pairs, test rest): {nc_k}/{len(test_pairs)} = {acc_k:.3f}")

    # T4: Reverse 'prev' direction
    prev_d = make_next_dir(ok)
    if prev_d is not None:
        prev_d_rev = normed(-prev_d)
        nc_rev = 0
        for i in range(1, len(ok)):
            src, tgt = ok[i], ok[i-1]
            pred, _ = entity_from_vocab(W_E[tid(src)], prev_d_rev, vocab_embs, {src})
            if pred == tgt: nc_rev += 1
        acc_rev = nc_rev/(len(ok)-1)
        print(f"\n  T4 'prev' (negated direction): {nc_rev}/{len(ok)-1} = {acc_rev:.3f}")

    return {"consistency": mean_consistency, "loo_acc": acc, "n": n}

# ─── Run all sequences ────────────────────────────────────────────
results = {}
results["numbers"]  = test_sequence("Numbers (one..fifteen)", NUMBERS)
results["ordinals"] = test_sequence("Ordinals (first..twelfth)", ORDINALS)
results["months"]   = test_sequence("Months (January..December)", MONTHS)

# Alphabet — check which single-token
alpha_ok = [c for c in ALPHABET if tid(c)]
print(f"\nAlphabet single-token chars: {len(alpha_ok)} → {alpha_ok[:10]}...")
if len(alpha_ok) >= 6:
    results["alphabet"] = test_sequence("Alphabet (a..z)", ALPHABET)

# ─── Cross-sequence transfer ──────────────────────────────────────
print(f"\n{'='*64}")
print("CROSS-SEQUENCE TRANSFER: Does 'next' transfer across sequences?")
print(f"{'='*64}\n")

nums_ok  = [w for w in NUMBERS  if tid(w)]
ords_ok  = [w for w in ORDINALS if tid(w)]
mons_ok  = [w for w in MONTHS   if tid(w)]

# Direction from numbers, test on ordinals
if len(nums_ok) >= 3 and len(ords_ok) >= 3:
    num_d = make_next_dir(nums_ok)
    ord_vocab = {w: W_E[tid(w)] for w in ords_ok}
    nc_xfer = sum(1 for i in range(len(ords_ok)-1)
                  if entity_from_vocab(W_E[tid(ords_ok[i])], num_d, ord_vocab,
                                       {ords_ok[i]})[0] == ords_ok[i+1])
    acc_xfer = nc_xfer/(len(ords_ok)-1)
    print(f"  Numbers→Ordinals transfer: {nc_xfer}/{len(ords_ok)-1} = {acc_xfer:.3f}")

# Direction from ordinals, test on numbers
if len(ords_ok) >= 3 and len(nums_ok) >= 3:
    ord_d = make_next_dir(ords_ok)
    num_vocab = {w: W_E[tid(w)] for w in nums_ok}
    nc_xfer2 = sum(1 for i in range(len(nums_ok)-1)
                   if entity_from_vocab(W_E[tid(nums_ok[i])], ord_d, num_vocab,
                                        {nums_ok[i]})[0] == nums_ok[i+1])
    acc_xfer2 = nc_xfer2/(len(nums_ok)-1)
    print(f"  Ordinals→Numbers transfer: {nc_xfer2}/{len(nums_ok)-1} = {acc_xfer2:.3f}")

# Direction from months, test on numbers
if len(mons_ok) >= 3 and len(nums_ok) >= 3:
    mon_d = make_next_dir(mons_ok)
    nc_xfer3 = sum(1 for i in range(len(nums_ok)-1)
                   if entity_from_vocab(W_E[tid(nums_ok[i])], mon_d, num_vocab,
                                        {nums_ok[i]})[0] == nums_ok[i+1])
    acc_xfer3 = nc_xfer3/(len(nums_ok)-1)
    print(f"  Months→Numbers transfer: {nc_xfer3}/{len(nums_ok)-1} = {acc_xfer3:.3f}")

# ─── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("Summary")
print(f"{'='*64}")
for name, r in results.items():
    if r:
        print(f"  {name:>12}: consistency={r.get('consistency',0):.3f}, "
              f"LOO_acc={r.get('loo_acc',0):.3f}")

print()
print("Key question: Does the number line axis support ordinal arithmetic?")
print("  If LOO_acc is high → Type B (one direction encodes all successors)")
print("  If consistency is high but LOO low → consistent but noisy (Type C)")
print("  If both low → sequence not encoded as traversable axis")

with open(OUTPUT_FILE, "w") as f:
    json.dump({k: {kk: float(vv) for kk,vv in v.items()} for k,v in results.items() if v}, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 172 complete.")
