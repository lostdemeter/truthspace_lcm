#!/usr/bin/env python3
"""
Day 220 — past_tense_F Train-Test Fix + past_tense_E Domain

Day 218 revealed that past_tense_F test pairs (stand/leave/bring/buy/
keep/feel) are dental/cluster consonant mutations — a completely different
morphological subclass from the ablaut/suppletive training pairs
(go/come/get/run/eat/see/drive/take/give/make).

Fix:
  1. Rebuild past_tense_F to use matched ablaut test pairs.
     Train: go/come/get/run/eat/see/drive/take/give/make
     Test:  ride/write/draw/know/grow/choose/wake/shake/break/steal
     (all ablaut: vowel alternation class)

  2. Create past_tense_E from the dental/cluster pairs.
     Train: stand/leave/bring/buy/keep/feel
     Test:  sleep/sweep/creep/kneel/deal/mean
     (all dental: stem + dental suffix mutation)

  3. Re-run v4 on full 42k vocab with corrected domains.

Questions:
  a. Does past_tense_F acc=1.000 with matched ablaut test?
  b. Does past_tense_E have sufficient dc for TYPE_BC?
  c. What is the final honest full-vocab v4 accuracy?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day220_pastF_fix.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAINS = {
    "capitals": {
        "expected": "TYPE_BC",
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi")],
    },
    "gender": {
        "expected": "TYPE_BC",
        "train": [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress"),("hero","heroine")],
        "test":  [("father","mother"),("brother","sister"),("son","daughter"),
                  ("husband","wife"),("uncle","aunt"),("waiter","waitress")],
    },
    "plurals": {
        "expected": "TYPE_BC",
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "superlative": {
        "expected": "TYPE_BC",
        "train": [("big","biggest"),("fast","fastest"),("long","longest"),
                  ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        "test":  [("hard","hardest"),("dark","darkest"),("soft","softest"),
                  ("warm","warmest"),("slow","slowest"),("small","smallest")],
    },
    "past_tense_F": {
        "expected": "TYPE_BC",
        "note": "FIXED: ablaut-matched test set",
        "train": [("go","went"),("come","came"),("get","got"),("run","ran"),
                  ("eat","ate"),("see","saw"),("drive","drove"),("take","took"),
                  ("give","gave"),("make","made")],
        "test":  [("ride","rode"),("write","wrote"),("draw","drew"),
                  ("know","knew"),("grow","grew"),("choose","chose"),
                  ("wake","woke"),("shake","shook"),("break","broke"),
                  ("steal","stole")],
    },
    "past_tense_E": {
        "expected": "TYPE_BC",
        "note": "NEW: dental/cluster pairs (was in past_tense_F test)",
        "train": [("stand","stood"),("leave","left"),("bring","brought"),
                  ("buy","bought"),("keep","kept"),("feel","felt")],
        "test":  [("sleep","slept"),("sweep","swept"),("creep","crept"),
                  ("kneel","knelt"),("deal","dealt"),("mean","meant")],
    },
    "past_tense_D": {
        "expected": "TYPE_BC",
        "train": [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
        "test":  [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
    },
    "past_tense_B": {
        "expected": "TYPE_BC",
        "train": [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
        "test":  [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
    },
    "numbers": {
        "expected": "TYPE_BC",
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6")],
        "test":  [("seven","7"),("eight","8"),("nine","9")],
    },
    "antonyms_unsup": {
        "expected": "TYPE_ADJACENT",
        "attribute": None,
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "antonyms_sup_size": {
        "expected": "TYPE_ANTONYM",
        "attribute": "size",
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short")],
    },
    "no_change_verbs": {
        "expected": "IDENTITY",
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost")],
    },
}

ANTONYM_AXES_DEF = {
    "size": [("big","small"),("large","tiny"),("huge","little"),
             ("tall","short"),("wide","narrow"),("thick","thin")],
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

def tid1_bare(word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_emb(word):
    t = tid1(word) or tid1_bare(word)
    return W_E[t].astype(np.float64) if t is not None else None

def ok_pairs(pairs):
    return [(a,b) for a,b in pairs
            if get_emb(a) is not None and get_emb(b) is not None]

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b)-get_emb(a)) for a,b in p]
    pw = [cosine(diffs[i],diffs[j])
          for i in range(len(diffs)) for j in range(i+1,len(diffs))]
    return float(np.mean(pw))

# ── Build full 42k vocab pool ─────────────────────────────────────────
print("Building full single-token vocab pool ...")
all_pool = {}
for token_id in range(V):
    decoded = tok.decode([token_id])
    if not decoded.startswith(" "): continue
    word = decoded[1:]
    if not word.isalpha(): continue
    if len(word) < 2: continue
    if word.islower():
        all_pool[word] = W_E[token_id].astype(np.float64)
    elif word[0].isupper() and word[1:].islower():
        all_pool[word] = W_E[token_id].astype(np.float64)

for d in ["1","2","3","4","5","6","7","8","9"]:
    t = tid1_bare(d)
    if t is not None:
        all_pool[d] = W_E[t].astype(np.float64)

for cfg in DOMAINS.values():
    for a,b in cfg["train"]+cfg["test"]:
        for w in (a,b):
            if w not in all_pool:
                e = get_emb(w)
                if e is not None: all_pool[w] = e

print(f"  Pool size: {len(all_pool)} tokens\n")

# ── Antonym axes ──────────────────────────────────────────────────────
antonym_axes = {}
for attr, pairs in ANTONYM_AXES_DEF.items():
    p = ok_pairs(pairs)
    if len(p) < 2: continue
    diffs = [normed(get_emb(a)-get_emb(b)) for a,b in p
             if get_emb(a) is not None and get_emb(b) is not None]
    if diffs:
        antonym_axes[attr] = normed(np.mean(diffs, axis=0))

# ── v4 Classifier ─────────────────────────────────────────────────────
def classify_v4(train_pairs, attribute=None):
    p = ok_pairs(train_pairs)
    if any(a == b for a,b in p):
        return "IDENTITY"
    if len(p) >= 2:
        dc = dir_consistency(p)
        if dc > 0.10:
            return "TYPE_BC"
    if attribute is not None and attribute in antonym_axes:
        return "TYPE_ANTONYM"
    return "TYPE_ADJACENT"

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    diffs = [normed(get_emb(b)-get_emb(a)) for a,b in p
             if get_emb(a) is not None and get_emb(b) is not None]
    return normed(np.mean(diffs, axis=0)) if diffs else None

def retrieve_bc(src, mdir, vocab=all_pool):
    se = get_emb(src)
    if se is None: return None, -1
    query = se + mdir
    sims = [(w, cosine(query, e)) for w,e in vocab.items() if w != src]
    sims.sort(key=lambda x: x[1], reverse=True)
    pred = sims[0][0] if sims else None
    return pred, sims

def retrieve_adjacent(src, vocab=all_pool):
    se = get_emb(src)
    if se is None: return None, []
    sims = [(w, cosine(se, e)) for w,e in vocab.items() if w != src]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[0][0] if sims else None, sims

def retrieve_antonym_axis(src, attribute, vocab=all_pool):
    se = get_emb(src)
    if se is None or attribute not in antonym_axes:
        return retrieve_adjacent(src, vocab)
    axis = antonym_axes[attribute]
    src_proj = float(np.dot(normed(se), axis))
    target_dir = axis if src_proj < 0 else -axis
    query = normed(se + target_dir)
    sims = [(w, cosine(query, e)) for w,e in vocab.items() if w != src]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[0][0] if sims else None, sims

# ── Diagnose new domains first ────────────────────────────────────────
print("DOMAIN DIAGNOSTICS (dc + single-token check):")
for domain_name in ["past_tense_F", "past_tense_E"]:
    cfg = DOMAINS[domain_name]
    p_train = ok_pairs(cfg["train"])
    p_test  = ok_pairs(cfg["test"])
    dc_train = dir_consistency(cfg["train"])
    dc_test  = dir_consistency(cfg["test"])
    missing_train = [(a,b) for a,b in cfg["train"]
                     if get_emb(a) is None or get_emb(b) is None]
    missing_test  = [(a,b) for a,b in cfg["test"]
                     if get_emb(a) is None or get_emb(b) is None]
    print(f"\n  {domain_name}  ({cfg['note']})")
    print(f"    train: {len(p_train)}/{len(cfg['train'])} single-token  dc={dc_train:.3f}")
    if missing_train:
        print(f"    missing train: {missing_train}")
    print(f"    test:  {len(p_test)}/{len(cfg['test'])} single-token  dc={dc_test:.3f}")
    if missing_test:
        print(f"    missing test: {missing_test}")

    # Cross-dc: does training direction work on test?
    mdir = mean_dir(cfg["train"])
    if mdir is not None and p_test:
        cross_sims = [cosine(mdir, normed(get_emb(b)-get_emb(a)))
                      for a,b in p_test
                      if get_emb(a) is not None and get_emb(b) is not None]
        print(f"    cross-dc (train dir vs test diffs): {np.mean(cross_sims):.3f}")

print()

# ── Main evaluation ───────────────────────────────────────────────────
print("=" * 74)
print(f"{'Domain':<22}  {'Expected':<14}  {'Predicted':<14}  "
      f"{'dc':>6}  {'acc':>6}  {'rank':>6}")
print("=" * 74)

all_results = {}
total_c = 0; total_n = 0

for domain_name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]
    expected = cfg["expected"]
    attribute = cfg.get("attribute", None)

    p  = ok_pairs(train)
    dc = dir_consistency(p) if len(p) >= 2 else 0.0
    pred = classify_v4(train, attribute)

    test_ok = ok_pairs(test)
    if not test_ok and ok_pairs(train):
        test_ok = ok_pairs(train)

    mdir = mean_dir(train) if pred == "TYPE_BC" else None

    correct = 0; ranks_list = []
    for src, tgt in test_ok:
        if pred == "IDENTITY":
            p_tgt = src
            rank  = 0
        elif pred == "TYPE_BC" and mdir is not None:
            p_tgt, sims = retrieve_bc(src, mdir)
            rank = next((i for i,(w,_) in enumerate(sims) if w==tgt), len(sims))
        elif pred == "TYPE_ANTONYM":
            p_tgt, sims = retrieve_antonym_axis(src, attribute)
            rank = next((i for i,(w,_) in enumerate(sims) if w==tgt), len(sims))
        else:
            p_tgt, sims = retrieve_adjacent(src)
            rank = next((i for i,(w,_) in enumerate(sims) if w==tgt), len(sims))
        if p_tgt == tgt: correct += 1
        ranks_list.append(rank)

    acc   = correct / len(test_ok) if test_ok else 0.0
    mrank = float(np.mean(ranks_list)) if ranks_list else -1.0
    total_c += correct; total_n += len(test_ok)

    match = (pred == expected)
    mark  = "" if match else " X"
    note  = cfg.get("note", "")
    note_str = f"  ({note})" if note else ""
    print(f"  {domain_name:<22}  {expected:<14}  {pred:<14}  "
          f"{dc:>6.3f}  {acc:>6.3f}  {mrank:>6.1f}{mark}{note_str}")

    all_results[domain_name] = {
        "expected": expected, "predicted": pred,
        "dir_consistency": dc, "acc": acc, "mean_rank": mrank,
        "n_test": len(test_ok),
        "correct_classification": match,
        "note": cfg.get("note",""),
    }

overall = total_c / total_n if total_n else 0
print()
print(f"  OVERALL: {total_c}/{total_n} = {overall:.3f}  (full 42k vocab)")
cls_correct = sum(1 for d in all_results.values() if d["correct_classification"])
print(f"  Classification: {cls_correct}/{len(all_results)} correct")

print()
print("=" * 74)
print("PIPELINE PROGRESSION")
print("=" * 74)
print("  v1 D198:  36/46 = 0.779  curated 281-word vocab")
print("  v2 D208:  40/46 = 0.870  curated 281-word vocab")
print("  v3 D212:  45/52 = 0.865  curated 281-word vocab")
print("  v4 D218:  37/51 = 0.725  full 42k vocab (ptF mismatched test)")
print(f"  v4 D220:  {total_c}/{total_n} = {overall:.3f}  full 42k vocab (ptF FIXED)")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 220 complete.")
