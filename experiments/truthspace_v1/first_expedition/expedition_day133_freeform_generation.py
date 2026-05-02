#!/usr/bin/env python3
"""
Day 133 — Free-Form Generation Test

Days 124-132 built a geometric ranking pipeline for candidate sets.
This is the ultimate test: can the pipeline generate next tokens WITHOUT
access to the LM's forward pass (no logits, no softmax)?

APPROACH:
  1. Pre-compute T2 addresses for TOP-K vocabulary words (K=200 most common)
  2. Given a prompt:
     a. Compute T2 of last token hidden state
     b. Classify query category from T2 centroid
     c. Route to sub-ranker (T2/struct_axis/L25)
     d. Rank the K vocabulary words
     e. Top-ranked word = predicted next token
  3. Compare to LM's actual top-1 prediction (ground truth)

Also measure:
  - Rank of LM top-1 prediction in geometric ranking (1=perfect)
  - Top-1 agreement between geometric pipeline and LM
  - Per-query-type breakdown

This directly tests: Does T2 geometry approximate the LM's next-token
distribution for freely generated text?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day133_freeform_generation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
L_BEST      = 25
VOCAB_K     = 300  # test against top-K common vocabulary words

TEST_PROMPTS = [
    # Tense (T2 method should excel)
    "Yesterday he walked to the",
    "Yesterday she ran through the",
    "Last week they built a",
    "The children played",
    # Gender (T2 method)
    "The king and",
    "The father and his",
    "The actor and",
    # Antonyms/opposites (struct_axis)
    "The opposite of hot is",
    "The opposite of large is",
    "The opposite of dark is",
    # Capitals (L25)
    "The capital city of France is",
    "The capital city of Japan is",
    "The capital city of Germany is",
    # Hypernyms (L25)
    "A poodle is a type of",
    "A rose is a type of",
    "An eagle is a type of",
    # Free-form sentences (no specific category)
    "The sun rises in the",
    "Water boils at one hundred degrees",
    "She opened the door and",
    "He read the book and then",
    "The cat sat on the",
]

ROUTING_TABLE = {
    "tense": "t2", "gender": "t2",
    "antonyms": "struct_axis",
    "hypernyms": "l25", "capitals": "l25", "languages": "l25",
    "unknown": "l25",  # default
}

DAY78_LAYERS = {"gender":27,"comparative":15,"hypernym":28,"plural":1,"synonym":28,"concrete":28,
                "past_tense":28,"antonym":28,"passive":28,"causation":28,"question":28,"negation":28}
AXIS_NAMES_12 = ["gender","comparative","hypernym","plural","synonym","concrete",
                 "past_tense","antonym","passive","causation","question","negation"]
AXIS_PAIRS = {
    "gender":[("The king ruled","The queen ruled"),("A man walked","A woman walked"),("His brother arrived","His sister arrived")],
    "comparative":[("The fast car","The faster car"),("A big dog","A bigger dog"),("The tall tree","The taller tree")],
    "hypernym":[("The dog ran","The animal ran"),("A rose bloomed","A flower bloomed"),("The car sped","The vehicle sped")],
    "plural":[("A dog played","Dogs played"),("The cat sat","The cats sat"),("A bird sang","Birds sang")],
    "synonym":[("He is big","He is large"),("She is small","She is tiny"),("It is cold","It is frigid")],
    "concrete":[("The stone is heavy","The burden is heavy"),("The long road","The long journey")],
    "past_tense":[("I walk every morning","I walked every morning"),("She runs through park","She ran through park"),("He eats","He ate")],
    "antonym":[("It is hot","It is cold"),("He runs fast","He runs slow"),("She is happy","She is sad")],
    "passive":[("The cat chased mouse","The mouse was chased"),("John broke window","The window was broken")],
    "causation":[("The rain falls","The ground gets wet"),("The fire burns","The wood turns to ash")],
    "question":[("She is tired","Is she tired"),("He can swim","Can he swim")],
    "negation":[("The dog is fast","The dog is not fast"),("She can swim","She cannot swim")],
}
TRAIN_CENTROIDS = {
    "antonyms": ["The opposite of hot is","The opposite of happy is","The opposite of bright is",
                 "The opposite of fast is","The opposite of large is","The opposite of right is"],
    "capitals": ["The capital city of France is","The capital city of Japan is","The capital city of Germany is",
                 "The capital city of Spain is","The capital city of Italy is","The capital city of Russia is"],
    "languages":["The official language of Brazil is","The official language of Egypt is",
                 "The official language of China is","The official language of India is"],
    "hypernyms": ["A poodle is a type of","A rose is a type of","A hammer is a type of",
                  "An eagle is a type of","A ruby is a type of"],
    "tense":     ["Yesterday he","Yesterday she","Yesterday they"],
    "gender":    ["The king and","The queen and","The father and"],
}
ANTONYM_TRAIN = [
    ("cold", ["warm","lukewarm"], "The opposite of hot is"),
    ("sad",  ["angry","bored"],  "The opposite of happy is"),
    ("dark", ["bright","sunny"], "The opposite of bright is"),
    ("slow", ["quick","rapid"],  "The opposite of fast is"),
    ("small",["big","huge"],     "The opposite of large is"),
    ("wrong",["correct","accurate"],"The opposite of right is"),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
T2_LAYERS  = sorted(set(DAY78_LAYERS.values()))
ALL_LAYERS = sorted(set([L_BEST] + T2_LAYERS))
V = tok.vocab_size
print(f"  hidden={H}, vocab={V}\n")

def get_hs(text, layers=None):
    if layers is None: layers = ALL_LAYERS
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

def get_logits_and_top(prompt, k=10):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
        lp     = torch.log_softmax(logits, dim=-1)
    top_ids  = torch.argsort(lp, descending=True)[:k].tolist()
    top_words = [tok.decode([i]).strip() for i in top_ids]
    return top_ids, top_words, lp.numpy()

# ── Build T2 axes ──────────────────────────────────────────────────────────────
print("Building T2 axes ...")
t2_axes = {}
for ax in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax]; diffs = []
    for s1, s2 in AXIS_PAIRS.get(ax, []):
        inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
        with torch.no_grad():
            h1 = model(**inp1, output_hidden_states=True).hidden_states[L][0,-1,:].numpy().astype(np.float32)
            h2 = model(**inp2, output_hidden_states=True).hidden_states[L][0,-1,:].numpy().astype(np.float32)
        d = h2-h1; nv = np.linalg.norm(d)
        if nv > 1e-6: diffs.append(d/nv)
    v = np.mean(diffs, axis=0) if diffs else np.zeros(H, np.float32)
    nv = np.linalg.norm(v); t2_axes[ax] = (v/nv if nv>1e-6 else v).astype(np.float32)
print("  Done.\n")

def t2_vec(hs):
    v = np.zeros(12, np.float32)
    for k, ax in enumerate(AXIS_NAMES_12):
        h = normed(hs.get(DAY78_LAYERS[ax], np.zeros(H))); v[k] = float(np.dot(h, t2_axes[ax]))
    return v

# Build struct_axis for antonyms
print("Building struct_axis (antonyms) ...")
ant_deltas = []
for correct, wrong_pool, _ in ANTONYM_TRAIN:
    h_c = get_hs(" "+correct, [L_BEST])[L_BEST]
    h_w = np.mean([get_hs(" "+w, [L_BEST])[L_BEST] for w in wrong_pool], axis=0).astype(np.float32)
    d = h_c - h_w; nv = np.linalg.norm(d)
    if nv > 1e-6: ant_deltas.append(d/nv)
struct_axis = normed(np.mean(ant_deltas, axis=0).astype(np.float32))
print("  Done.\n")

# Build T2 centroids per category
print("Building category T2 centroids ...")
cat_centroids = {}
for cat, prompts in TRAIN_CENTROIDS.items():
    vecs = [t2_vec(get_hs(p, T2_LAYERS)) for p in prompts]
    m = np.mean(vecs, axis=0)
    cat_centroids[cat] = normed(m.astype(np.float32))
print("  Done.\n")

def classify_prompt(prompt):
    v = t2_vec(get_hs(prompt, T2_LAYERS))
    scores = {cat: cosine(v, cat_centroids[cat]) for cat in cat_centroids}
    return max(scores, key=lambda c: scores[c])

# ── Build TOP-K vocabulary ────────────────────────────────────────────────────
print(f"Building top-{VOCAB_K} vocabulary T2 cache ...")
# Get top-K tokens by language model probability across many prompts
# (use an aggregate of logits to find the most common predicted tokens)
# For efficiency: just use frequency-based selection from vocab
anchor_prompts = ["The", "A", "He", "She", "Yesterday", "The capital", "A type of", "The opposite"]
all_top_ids: Counter = Counter()
for ap in anchor_prompts:
    ids, _, _ = get_logits_and_top(ap, k=100)
    for i in ids: all_top_ids[i] += 1

top_vocab_ids = [i for i, _ in all_top_ids.most_common(VOCAB_K)]
top_vocab_words = [tok.decode([i]).strip() for i in top_vocab_ids]
# Filter to single-token words that decode cleanly
vocab_pairs = [(i, w) for i, w in zip(top_vocab_ids, top_vocab_words) if len(w) > 0 and w.isalpha()]
print(f"  Kept {len(vocab_pairs)} clean vocabulary words\n")

# Pre-compute hidden states for vocab words at all layers
print(f"Pre-computing vocab HS ({len(vocab_pairs)} words) ...")
vocab_hs = {}
for i, (tok_id, word) in enumerate(vocab_pairs):
    if i % 50 == 0: print(f"  {i}/{len(vocab_pairs)} ...", end="\r", flush=True)
    vocab_hs[word] = get_hs(" " + word, ALL_LAYERS)
print(f"  Done. {len(vocab_hs)} words cached.\n")

def rank_vocab(prompt_hs, method, ctx_hs):
    words = list(vocab_hs.keys())
    if method == "t2":
        ctx_t2 = t2_vec(ctx_hs)
        scores = {w: cosine(t2_vec(vocab_hs[w]), ctx_t2) for w in words}
    elif method == "struct_axis":
        scores = {w: float(np.dot(normed(vocab_hs[w][L_BEST]), struct_axis)) for w in words}
    else:  # l25
        h_ctx = ctx_hs[L_BEST]
        scores = {w: cosine(h_ctx, vocab_hs[w][L_BEST]) for w in words}
    ranked = sorted(words, key=lambda w: -scores[w])
    return ranked, scores

# ── Main generation test ───────────────────────────────────────────────────────
print("="*72)
print("Day 133: Free-Form Generation Test")
print("="*72)
print()

all_results = []
for prompt in TEST_PROMPTS:
    ctx_hs = get_hs(prompt, ALL_LAYERS)
    cat    = classify_prompt(prompt)
    method = ROUTING_TABLE.get(cat, "l25")

    # Geometric ranking
    geo_ranked, geo_scores = rank_vocab(t2_vec(ctx_hs), method, ctx_hs)
    geo_top1 = geo_ranked[0]

    # LM oracle top-1
    lm_ids, lm_words, lp = get_logits_and_top(prompt, k=20)
    lm_top1 = lm_words[0]

    # Rank of LM top-1 in geometric ranking (lower = better)
    lm_in_vocab = lm_top1 in geo_scores
    if lm_in_vocab:
        rank_of_lm_top1 = next((i+1 for i,w in enumerate(geo_ranked) if w == lm_top1), len(geo_ranked)+1)
    else:
        rank_of_lm_top1 = None

    # Top-5 geometric candidates that are also in LM top-20
    geo_top5_in_lm = [w for w in geo_ranked[:20] if w in lm_words[:20]]

    agreement = geo_top1 == lm_top1
    overlap_10 = len([w for w in geo_ranked[:10] if w in lm_words[:10]])

    all_results.append({
        "prompt": prompt, "cat": cat, "method": method,
        "geo_top1": geo_top1, "lm_top1": lm_top1,
        "top1_agree": agreement,
        "rank_of_lm_top1": rank_of_lm_top1,
        "overlap_10": overlap_10,
        "geo_top5": geo_ranked[:5],
        "lm_top5": lm_words[:5],
    })

    rank_str = f"LM_top1 @ geo_rank={rank_of_lm_top1}" if rank_of_lm_top1 else "LM_top1 not in vocab"
    print(f"  [{cat:>10}|{method:>11}] {prompt!r}")
    print(f"    geo={geo_ranked[:5]}  lm={lm_words[:5]}")
    print(f"    agree={agreement}  {rank_str}  overlap@10={overlap_10}")
    print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("="*72)
print("Summary")
print("="*72)
n = len(all_results)
n_agree    = sum(1 for r in all_results if r["top1_agree"])
valid_ranks = [r["rank_of_lm_top1"] for r in all_results if r["rank_of_lm_top1"] is not None]
mean_rank  = float(np.mean(valid_ranks)) if valid_ranks else float("nan")
mean_overlap = float(np.mean([r["overlap_10"] for r in all_results]))
random_agree = 1 / len(vocab_hs) if vocab_hs else 0.0

print(f"""
  Top-1 agreement (geo==lm):   {n_agree}/{n} = {n_agree/n:.3f}
  Mean rank of LM top-1 in geo: {mean_rank:.2f} (out of {len(vocab_hs)} words)
  Mean top-10 overlap:          {mean_overlap:.2f}/10
  Random top-1 agreement:       {random_agree:.4f}
  Random top-10 overlap:        {10*len(lm_words[:10])/max(len(vocab_hs),1):.2f}/10
""")

print("  Per-category breakdown:")
for cat_name in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"] == cat_name]
    n_ag  = sum(1 for r in cat_r if r["top1_agree"])
    vranks = [r["rank_of_lm_top1"] for r in cat_r if r["rank_of_lm_top1"] is not None]
    mr    = float(np.mean(vranks)) if vranks else float("nan")
    meth  = ROUTING_TABLE.get(cat_name, "l25")
    print(f"    {cat_name:>12} [{meth:>11}]: agree={n_ag}/{len(cat_r)}  mean_rank={mr:.1f}")

print()
if n_agree / n > 0.3:
    print("  VERDICT: Geometric pipeline agrees with LM top-1 at > 30% of prompts.")
    print("  → Geometric structure carries significant next-token predictive information.")
elif n_agree / n > 0.1:
    print("  VERDICT: Geometric pipeline agrees at 10-30% — above random, but weak.")
else:
    print("  VERDICT: Agreement ≤ 10% — geometric ranking does not predict free tokens.")

if mean_rank <= 10:
    print(f"  BONUS: LM top-1 appears at geometric rank {mean_rank:.1f} on average — top-10!")
elif mean_rank <= 30:
    print(f"  BONUS: LM top-1 appears at geometric rank {mean_rank:.1f} — top quartile of vocab.")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "n_agree": n_agree,
        "n_total": n,
        "top1_agreement": n_agree / n,
        "mean_rank_of_lm_top1": mean_rank,
        "mean_top10_overlap": mean_overlap,
        "vocab_size": len(vocab_hs),
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 133 complete.")
