#!/usr/bin/env python3
"""
Day 214 — Vocabulary Size Effect on Retrieval Accuracy

Question: how does retrieval accuracy change as the retrieval vocabulary
grows from curated (281 words) toward the full single-token vocab?

Hypothesis:
  TYPE_BC:       direction disambiguates → accuracy MAINTAINED at large vocab
  TYPE_ADJACENT: proximity fails → accuracy DROPS at large vocab
  IDENTITY:      trivial → constant

Method:
  1. Build a pool of single-token English words filtered from the full
     151,936-token Qwen vocab (using a heuristic: strip leading space,
     check it's alphanumeric and length >= 2).
  2. Subsample vocabulary at sizes: 300, 500, 1000, 2000, 5000, 10000,
     and full single-token-English (estimate ~20k-40k words).
  3. For each vocab size: run v3 pipeline on the 4 representative domains:
       capitals    (TYPE_BC,       dc=0.368)
       plurals     (TYPE_BC,       dc=0.283)
       past_tense_D (TYPE_ADJACENT, dc=0.135)
       antonyms    (TYPE_ADJACENT,  dc=0.020)
  4. Plot accuracy vs log(vocab_size) for each domain.

The test pairs include the CORRECT answer in the vocab at all sizes
(by construction — we always include ground truth tokens).

We also check:
  - How does rank(correct answer) change with vocab size?
  - Is the correct-answer rank for TYPE_BC stable?
  - Does TYPE_ADJACENT rank degrade monotonically?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day214_vocab_size.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAIN_CONFIGS = {
    "capitals": {
        "type":  "TYPE_BC",
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi")],
    },
    "plurals": {
        "type":  "TYPE_BC",
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "past_tense_D": {
        "type":  "TYPE_ADJACENT",
        "train": [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
        "test":  [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
    },
    "antonyms": {
        "type":  "TYPE_ADJACENT",
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
}

VOCAB_SIZES = [300, 500, 1000, 2000, 5000, 10000, 20000]

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

# ── Build single-token English word pool ─────────────────────────────
print("Building single-token English word pool from full vocabulary ...")
word_pool = {}
for token_id in range(V):
    decoded = tok.decode([token_id])
    # Must start with a space (word-initial), strip it, then check
    if not decoded.startswith(" "): continue
    word = decoded[1:]  # strip leading space
    if not word.isalpha(): continue
    if len(word) < 2: continue
    if not word.islower(): continue  # keep lowercase only for cleanliness
    word_pool[word] = token_id

print(f"  Single-token lowercase English words: {len(word_pool)}\n")

# Always include all test targets in vocab
always_include = set()
for cfg in DOMAIN_CONFIGS.values():
    for a,b in cfg["train"] + cfg["test"]:
        always_include.add(a)
        always_include.add(b)

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
    return [(a,b) for a,b in pairs if get_emb(a) is not None and get_emb(b) is not None]

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in p]
    pw = [cosine(diffs[i], diffs[j])
          for i in range(len(diffs)) for j in range(i+1, len(diffs))]
    return float(np.mean(pw))

def mean_dir(train_pairs):
    p = ok_pairs(train_pairs)
    if not p: return None
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in p]
    return normed(np.mean(diffs, axis=0))

def build_vocab(size):
    """Build retrieval vocab of given size, always including test targets."""
    # Start with required words
    vocab = {}
    for w in always_include:
        e = get_emb(w)
        if e is not None: vocab[w] = e
    # Fill remainder from word_pool, sorted by token_id for reproducibility
    needed = max(0, size - len(vocab))
    extras = [w for w in sorted(word_pool, key=lambda w: word_pool[w])
              if w not in vocab]
    for w in extras[:needed]:
        e = get_emb(w)
        if e is not None: vocab[w] = e
    return vocab

def evaluate_domain(domain_name, cfg, vocab):
    train = cfg["train"]; test = cfg["test"]; dtype = cfg["type"]
    p_test = ok_pairs(test)
    if not p_test: return None, None
    mdir = mean_dir(train)
    correct = 0; ranks = []
    for src, tgt in p_test:
        se = get_emb(src)
        if se is None: continue
        if dtype == "TYPE_BC" and mdir is not None:
            query = se + mdir
        else:
            query = se
        sims = [(w, cosine(query, e)) for w,e in vocab.items() if w != src]
        sims.sort(key=lambda x: x[1], reverse=True)
        pred = sims[0][0] if sims else None
        rank = next((i for i,(w,_) in enumerate(sims) if w == tgt), len(sims))
        if pred == tgt: correct += 1
        ranks.append(rank)
    acc  = correct / len(p_test)
    mrank = float(np.mean(ranks))
    return acc, mrank

# ── Main experiment ───────────────────────────────────────────────────
results = {}
print("=" * 74)
print(f"{'Domain':<14}  {'Type':<14}  " +
      "  ".join(f"{s:>6}" for s in VOCAB_SIZES))
print("=" * 74)

for domain_name, cfg in DOMAIN_CONFIGS.items():
    accs = []; ranks_list = []
    for size in VOCAB_SIZES:
        vocab = build_vocab(size)
        actual_size = len(vocab)
        acc, mrank = evaluate_domain(domain_name, cfg, vocab)
        accs.append(acc if acc is not None else 0.0)
        ranks_list.append(mrank if mrank is not None else 0.0)
    print(f"  {domain_name:<14}  {cfg['type']:<14}  " +
          "  ".join(f"{a:>6.3f}" for a in accs))
    results[domain_name] = {
        "type": cfg["type"],
        "accuracy": {str(s): a for s,a in zip(VOCAB_SIZES, accs)},
        "mean_rank": {str(s): r for s,r in zip(VOCAB_SIZES, ranks_list)},
    }

print()
print("MEAN RANK OF CORRECT ANSWER")
print(f"{'Domain':<14}  {'Type':<14}  " +
      "  ".join(f"{s:>6}" for s in VOCAB_SIZES))
for domain_name, cfg in DOMAIN_CONFIGS.items():
    ranks = [results[domain_name]["mean_rank"][str(s)] for s in VOCAB_SIZES]
    print(f"  {domain_name:<14}  {cfg['type']:<14}  " +
          "  ".join(f"{r:>6.1f}" for r in ranks))

print()
print("ACCURACY CHANGE (300 → 20000):")
for domain_name, cfg in DOMAIN_CONFIGS.items():
    a_start = results[domain_name]["accuracy"]["300"]
    a_end   = results[domain_name]["accuracy"]["20000"]
    delta   = a_end - a_start
    print(f"  {domain_name:<14}  {cfg['type']:<14}  "
          f"{a_start:.3f} → {a_end:.3f}  Δ={delta:+.3f}")

# ── Full pool accuracy ────────────────────────────────────────────────
print()
print("FULL SINGLE-TOKEN POOL ACCURACY:")
full_vocab = {w: W_E[tid].astype(np.float64)
              for w,tid in word_pool.items()}
# Add any test targets not in pool
for w in always_include:
    if w not in full_vocab:
        e = get_emb(w)
        if e is not None: full_vocab[w] = e
print(f"  Pool size: {len(full_vocab)} words")
for domain_name, cfg in DOMAIN_CONFIGS.items():
    acc, mrank = evaluate_domain(domain_name, cfg, full_vocab)
    results[domain_name]["accuracy"]["full"] = acc
    results[domain_name]["mean_rank"]["full"] = mrank
    print(f"  {domain_name:<14}  {cfg['type']:<14}  acc={acc:.3f}  rank={mrank:.1f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 214 complete.")
