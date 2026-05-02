#!/usr/bin/env python3
"""
Day 202 — Cross-Domain Direction Transfer

QUESTION: Does a direction vector learned from a training set of word pairs
generalize to completely UNSEEN words that were never in the training set?

For morphological domains (plurals, superlative, past_tense), split the
known pairs into TRAIN and TEST sets with zero overlap:

  TRAIN: compute mean direction from training pairs
  TEST:  apply that direction to test source words → retrieve target
         target must be selected from a larger candidate vocabulary
         (not just test targets — all single-token words from vocabulary)

This tests TRUE zero-shot transfer:
  - "cat→cats" was never in the test set
  - Can the plural direction predict that "lamp→lamps", "wall→walls", etc.?

Also test CROSS-DOMAIN transfer:
  - Train on PLURAL direction, apply to SUPERLATIVE targets → should fail
  - Train on PLURAL direction, apply to novel PLURAL targets → should work
  - Confirms direction is domain-specific

EXPERIMENTS:
  1. Within-domain transfer: train on 6 pairs, test on 6 unseen pairs
     (plurals, superlative, past_tense)
  2. Zero-shot from k=1: single training pair → predict 20 held-out test pairs
  3. Cross-domain confusion: apply plural direction to superlative targets

For experiment 1 and 2, the target vocabulary is:
  All single-token words from a list of common English nouns/verbs/adjectives
  (the ground truth target should rank #1 or near #1 in this vocab)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day202_direction_transfer.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Training pairs (learned direction from these)
TRAIN = {
    "plurals":    [("cat","cats"),("dog","dogs"),("house","houses"),
                   ("tree","trees"),("book","books"),("car","cars")],
    "superlative":[("big","biggest"),("fast","fastest"),("old","oldest"),
                   ("cold","coldest"),("smart","smartest"),("long","longest")],
    "past_tense": [("run","ran"),("eat","ate"),("go","went"),
                   ("see","saw"),("come","came"),("give","gave")],
}

# Test pairs (completely held-out — not in TRAIN)
TEST = {
    "plurals":    [("bird","birds"),("ship","ships"),("hand","hands"),
                   ("eye","eyes"),("road","roads"),("door","doors"),
                   ("lamp","lamps"),("wall","walls"),("cup","cups"),
                   ("bed","beds"),("key","keys"),("box","boxes")],
    "superlative":[("hard","hardest"),("dark","darkest"),("soft","softest"),
                   ("warm","warmest"),("bright","brightest"),("clean","cleanest"),
                   ("slow","slowest"),("small","smallest"),("tall","tallest"),
                   ("weak","weakest")],
    "past_tense": [("take","took"),("make","made"),("say","said"),
                   ("know","knew"),("find","found"),("think","thought"),
                   ("leave","left"),("bring","brought"),("buy","bought"),
                   ("stand","stood")],
}

# For cross-domain confusion test
CROSS_TEST = {
    # Apply plural direction to superlative targets (expect failure)
    "plural_dir→superlative_tgt": {
        "direction_from": "plurals",
        "test_pairs": TEST["superlative"],
    },
    # Apply superlative direction to plural targets (expect failure)
    "superlative_dir→plural_tgt": {
        "direction_from": "superlative",
        "test_pairs": TEST["plurals"],
    },
    # Apply past_tense direction to plural targets (expect failure)
    "past_dir→plural_tgt": {
        "direction_from": "past_tense",
        "test_pairs": TEST["plurals"],
    },
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

# Build candidate vocabulary for open-set retrieval:
# Common English words that are single tokens, used as the retrieval pool.
# We include all test targets + common decoys to make it a real test.
CANDIDATE_WORDS = [
    # plurals of common nouns
    "cats","dogs","houses","trees","books","cars","birds","ships","hands",
    "eyes","roads","doors","lamps","walls","cups","beds","keys","boxes",
    "tables","chairs","floors","windows","rooms","pages","words","names",
    "times","years","days","ways","men","women","children","people",
    "things","parts","cases","faces","places","points","groups","numbers",
    # superlatives
    "biggest","fastest","oldest","coldest","smartest","longest","hardest",
    "darkest","softest","warmest","brightest","cleanest","slowest","smallest",
    "tallest","weakest","nearest","latest","greatest","highest","lowest",
    "newest","oldest","youngest","richest","poorest","safest","quietest",
    # past tense
    "ran","ate","went","saw","came","gave","took","made","said","knew",
    "found","thought","left","brought","bought","stood","got","put","set",
    "let","cut","hit","put","lost","felt","kept","sent","held","read",
    # base forms (decoys)
    "cat","dog","house","tree","book","car","bird","ship","hand","eye",
    "run","eat","go","see","come","give","take","make","say","know",
    "big","fast","old","cold","smart","long","hard","dark","soft","warm",
]
# Deduplicate and filter to single-token
candidate_vocab = {}
for w in CANDIDATE_WORDS:
    t = tid1(w)
    if t is not None and w not in candidate_vocab:
        candidate_vocab[w] = W_E[t].astype(np.float64)
print(f"Candidate vocabulary: {len(candidate_vocab)} single-token words\n")

def mean_direction(pairs):
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]
    if not ok: return None, 0
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in ok]
    return normed(np.mean(diffs, axis=0)), len(ok)

def retrieve(src_word, direction, vocab=candidate_vocab, exclude=None):
    se = get_emb(src_word)
    if se is None: return None, []
    query = se + direction
    sims = {w: cosine(query, e) for w,e in vocab.items()
            if w != src_word and (exclude is None or w not in exclude)}
    ranked = sorted(sims, key=lambda w: sims[w], reverse=True)
    return ranked[0] if ranked else None, ranked

def evaluate_transfer(domain_name, train_pairs, test_pairs, direction,
                       label="", verbose=True):
    ok = [(a,b) for a,b in test_pairs if tid1(a) and tid1(b)]
    if not ok: return None
    correct = 0
    ranks = []
    hits_at_5 = 0
    for a, b in ok:
        _, ranked = retrieve(a, direction)
        if not ranked: continue
        if ranked[0] == b: correct += 1
        rank = ranked.index(b) if b in ranked else len(ranked)
        if rank < 5: hits_at_5 += 1
        ranks.append(rank)
    acc = correct / len(ok)
    h5  = hits_at_5 / len(ok)
    mr  = float(np.mean(ranks))
    if verbose:
        print(f"  {label:<35}  n={len(ok):<3}  acc={acc:.3f}  "
              f"H@5={h5:.3f}  mean_rank={mr:.2f}")
    return {"accuracy": acc, "hits_at_5": h5, "mean_rank": mr, "n": len(ok)}

# ── Experiment 1: Within-domain transfer (train 6, test on held-out) ─
print("=" * 70)
print("EXP 1: Within-domain transfer (train=6 pairs, test on held-out pairs)")
print("=" * 70)
results = {}
for domain in ["plurals", "superlative", "past_tense"]:
    train_dir, n_train = mean_direction(TRAIN[domain])
    test_pairs = TEST[domain]
    label = f"{domain} (k={n_train} train)"
    r = evaluate_transfer(domain, TRAIN[domain], test_pairs, train_dir,
                          label=label)
    results[f"within_{domain}"] = r
print()

# ── Experiment 2: Single-pair transfer (k=1) ─────────────────────────
print("=" * 70)
print("EXP 2: Zero-shot from k=1 (single training pair)")
print("=" * 70)
for domain in ["plurals", "superlative", "past_tense"]:
    test_pairs = TEST[domain]
    # Try each training pair as the solo exemplar
    for train_pair in TRAIN[domain][:4]:  # test first 4 exemplars
        dir1, n1 = mean_direction([train_pair])
        if dir1 is None:
            print(f"  {domain} k=1 [{train_pair[0]}→{train_pair[1]}]  SKIP (multi-token)")
            continue
        label = f"  {domain} k=1 [{train_pair[0]}→{train_pair[1]}]"
        r = evaluate_transfer(domain, [train_pair], test_pairs, dir1,
                              label=label)
        results[f"k1_{domain}_{train_pair[0]}"] = r
    print()

# ── Experiment 3: Cross-domain confusion ─────────────────────────────
print("=" * 70)
print("EXP 3: Cross-domain direction confusion")
print("=" * 70)
for test_name, cfg in CROSS_TEST.items():
    src_domain = cfg["direction_from"]
    dir_vec, n_train = mean_direction(TRAIN[src_domain])
    label = test_name
    r = evaluate_transfer(test_name, TRAIN[src_domain],
                          cfg["test_pairs"], dir_vec, label=label)
    results[f"cross_{test_name}"] = r
print()

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY: Within-domain transfer accuracy")
print("=" * 70)
for domain in ["plurals", "superlative", "past_tense"]:
    r = results.get(f"within_{domain}")
    if r:
        print(f"  {domain:<16}: acc={r['accuracy']:.3f}  H@5={r['hits_at_5']:.3f}")
print()
print("Cross-domain (expect near-zero):")
for test_name in CROSS_TEST:
    r = results.get(f"cross_{test_name}")
    if r:
        print(f"  {test_name:<40}: acc={r['accuracy']:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 202 complete.")
