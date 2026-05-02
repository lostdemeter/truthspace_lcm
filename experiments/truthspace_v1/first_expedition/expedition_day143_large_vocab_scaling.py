#!/usr/bin/env python3
"""
Day 143 — Large Vocabulary Scaling of entity_excl

Days 139-141: entity_excl achieves 79% on held-out prompts with 234-word vocab.
Question: Does this hold when vocabulary is 10x or 100x larger?

Two concerns:
  1. Signal dilution: more candidates = harder to rank correct answer at rank 1
  2. Noise tokens: proper nouns, subwords, etc. may cluster near entity

TEST: Use top-K single-token English words by model vocabulary frequency:
  K = 234 (baseline), 500, 1000, 2000

For each K, run the Day 141 held-out prompts and measure:
  - Top-1 agreement with LM oracle
  - Mean rank of oracle top-1 in entity_excl ranking
  - Overlap@10

Also test: does entity rank stay stable as K grows?
  If entity_excl rank = 1 at K=234, does it stay 1 at K=1000?
  Or does the answer get pushed down by new words?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day143_large_vocab_scaling.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Curated 234-word vocab (baseline)
VOCAB_CURATED = [
    "walked","ran","ate","built","wrote","read","said","went","came","took",
    "made","got","saw","gave","knew","thought","found","told","became","left",
    "brought","bought","taught","caught","fought","heard","held","kept","sent",
    "fell","felt","grew","slept","spent","stood","wore","won",
    "drove","flew","swam","sang","sat","laid","paid","played","stayed","opened",
    "turned","looked","stopped","asked","started","tried","closed","moved","lived",
    "walk","run","eat","build","write","say","come","take","make","get","see",
    "give","know","think","find","tell","become","leave",
    "cold","hot","big","small","fast","slow","dark","light","happy","sad",
    "good","bad","strong","weak","young","old","loud","quiet","easy","hard",
    "clean","dirty","rich","poor","safe","early","late",
    "Paris","London","Rome","Berlin","Madrid","Tokyo","Moscow","Beijing",
    "Sydney","Ottawa","Canberra","Brasilia","Cairo","Delhi","Seoul","Bangkok",
    "Vienna","Warsaw","Athens","Lisbon","Brussels","Amsterdam","Oslo","Stockholm",
    "English","French","Spanish","German","Italian","Portuguese","Arabic",
    "Mandarin","Japanese","Korean","Hindi","Russian","Turkish","Persian",
    "Bengali","Tamil","Urdu","Polish","Dutch","Swedish","Greek",
    "animal","plant","tool","vehicle","food","music","sport","color","number",
    "language","country","city","flower","tree","bird","fish","dog","cat","horse",
    "instrument","weapon","machine","device","metal","mineral","crystal","gem",
    "king","queen","prince","princess","duke","duchess","emperor","empress",
    "father","mother","brother","sister","son","daughter","uncle","aunt",
    "man","woman","boy","girl","actor","actress","hero","heroine",
    "east","west","north","south","morning","evening","night","water","fire",
    "door","house","book","table","chair","window","street","road","park",
    "school","office","market","store","church","castle","palace","bridge",
    "then","also","soon","just","very","still","again","always","never",
    "first","last","next","before","after","here","there",
]

# Same held-out cases as Day 141 (fixed exclusions)
HELD_OUT = [
    ("The capital city of Russia is",     "Russia",  {"Russia","Russian"},  "Moscow",    "capitals"),
    ("The capital city of China is",      "China",   {"China","Chinese"},   "Beijing",   "capitals"),
    ("The capital city of Australia is",  "Australia",{"Australia"},        "Canberra",  "capitals"),
    ("The capital city of Greece is",     "Greece",  {"Greece","Greek"},    "Athens",    "capitals"),
    ("The capital city of Poland is",     "Poland",  {"Poland","Polish"},   "Warsaw",    "capitals"),
    ("The capital city of Sweden is",     "Sweden",  {"Sweden","Swedish"},  "Stockholm", "capitals"),
    ("The official language of Germany is","Germany",{"Germany"},           "German",   "languages"),
    ("The official language of Japan is", "Japan",   {"Japan"},             "Japanese", "languages"),
    ("The official language of Korea is", "Korea",   {"Korea"},             "Korean",   "languages"),
    ("A hammer is a type of",  "hammer", {"hammer"}, "tool",       "hypernyms"),
    ("A ruby is a type of",    "ruby",   {"ruby"},   "gem",        "hypernyms"),
    ("A whale is a type of",   "whale",  {"whale"},  "animal",     "hypernyms"),
    ("A violin is a type of",  "violin", {"violin"}, "instrument", "hypernyms"),
    ("The opposite of good is",   "good",   {"good"},   "bad",    "antonyms"),
    ("The opposite of fast is",   "fast",   {"fast"},   "slow",   "antonyms"),
    ("The opposite of clean is",  "clean",  {"clean"},  "dirty",  "antonyms"),
    ("The opposite of rich is",   "rich",   {"rich"},   "poor",   "antonyms"),
    ("The opposite of loud is",   "loud",   {"loud"},   "quiet",  "antonyms"),
    ("The opposite of strong is", "strong", {"strong"}, "weak",   "antonyms"),
    ("The prince and", "prince", {"prince"},  "princess","gender"),
    ("The duke and",   "duke",   {"duke"},    "duchess", "gender"),
    ("The actor and",  "actor",  {"actor"},   "actress", "gender"),
    ("The son and",    "son",    {"son"},     "daughter","gender"),
    ("The opposite of happy is",  "happy",  {"happy"},  "sad",    "antonyms_extra"),
    ("The opposite of early is",  "early",  {"early"},  "late",   "antonyms_extra"),
    ("The opposite of old is",    "old",    {"old"},    "young",  "antonyms_extra"),
    ("The opposite of easy is",   "easy",   {"easy"},   "hard",   "antonyms_extra"),
    # Tense (expected to still fail, but track rank)
    ("Last month she",  "she",  {"she"},  "went", "tense"),
    ("Last week they",  "they", {"they"}, "went", "tense"),
]

K_VALUES = [234, 500, 1000, 2000]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
print(f"  hidden={H}  W_E shape={W_E.shape}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_logprob_vocab(prompt, vocab_tok_ids):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    return {w: float(lp[vocab_tok_ids[w]]) for w in vocab_tok_ids}

# Build extended vocabularies
# Start from curated 234-word, then expand using frequency-sorted LM vocab
print("Building extended vocabularies ...")

# Get all single-token words (alphabetic, len>=3, not in curated) sorted by freq
vocab_size = W_E.shape[0]
# Use LM's own unigram log-probs as a proxy for token frequency
# Frequency ≈ tokens with highest average log-prob across a few prompts
# Simpler: just use W_E norm as a proxy (common tokens have higher norm)
all_norms  = np.linalg.norm(W_E, axis=1)  # (V,) — proxy for usage

import re
candidate_words = set(VOCAB_CURATED)
# Expand with more single-token alphabetic words from the tokenizer vocabulary
for tid in range(vocab_size):
    dec = tok.decode([tid]).strip()
    if re.match(r'^[A-Za-z]{3,}$', dec) and len(tid := tok(" "+dec, add_special_tokens=False)["input_ids"]) == 1:
        candidate_words.add(dec)
    if len(candidate_words) > 5000: break  # cap scan

# Actually scan by token id and check for valid single-token words
extended = {}
for tid in range(min(vocab_size, 32000)):
    dec_raw = tok.decode([tid])
    # Single leading space → it's a word token in LLM tokenizers
    dec = dec_raw.lstrip("▁Ġ ").strip()
    if not re.match(r'^[A-Za-z]{3,12}$', dec): continue
    # Verify it tokenizes back to single token
    back_ids = tok(" "+dec, add_special_tokens=False)["input_ids"]
    if len(back_ids) != 1 or back_ids[0] != tid: continue
    extended[dec] = {"tid": tid, "emb": W_E[tid], "norm": float(all_norms[tid])}

print(f"  Found {len(extended)} single-token alphabetic words in vocab")

# Sort by norm (proxy for frequency) and take top K
sorted_words = sorted(extended.keys(), key=lambda w: -extended[w]["norm"])

# Build vocab sets for each K
vocab_sets = {}
for K in K_VALUES:
    # Always include the curated 234-word set + top-(K-234) by norm
    curated_ok = [w for w in VOCAB_CURATED if w in extended]
    top_K_words = sorted_words[:K]
    combined = list(set(curated_ok + top_K_words))[:K]
    vocab_sets[K] = {w: extended[w] for w in combined if w in extended}
    print(f"  K={K}: {len(vocab_sets[K])} words in vocab")

print()
print("="*72)
print("Day 143: Large Vocabulary Scaling")
print("="*72)

# For each K, run all held-out cases
scale_results = {K: [] for K in K_VALUES}

for K in K_VALUES:
    voc = vocab_sets[K]
    vocab_ids = {w: voc[w]["tid"] for w in voc}
    vocab_embs = {w: voc[w]["emb"] for w in voc}
    voc_words = list(voc.keys())

    for prompt, entity_word, exclude, expected, cat in HELD_OUT:
        excl_words = [w for w in voc_words if w not in exclude]

        # Oracle log-prob
        lp = get_logprob_vocab(prompt, vocab_ids)
        oracle_ranked = sorted(excl_words, key=lambda w: -lp[w])
        oracle_top1 = oracle_ranked[0]

        # entity_excl
        eid = get_token_id(entity_word)
        if eid is None:
            eid = next((extended[w]["tid"] for w in extended if w.lower()==entity_word.lower()), 0)
        entity_emb = W_E[eid]
        scores = {w: cosine(entity_emb, vocab_embs[w]) for w in excl_words}
        ranked = sorted(excl_words, key=lambda w: -scores[w])
        entity_top1 = ranked[0]

        rank = next((i+1 for i,w in enumerate(ranked) if w==oracle_top1), len(ranked)+1)
        ov10 = len(set(ranked[:10]) & set(oracle_ranked[:10]))
        agree = entity_top1 == oracle_top1

        scale_results[K].append({
            "prompt": prompt, "entity": entity_word, "cat": cat,
            "oracle_top1": oracle_top1, "entity_top1": entity_top1,
            "rank": rank, "ov10": ov10, "agree": agree,
        })

# Print results
print()
print(f"  {'K':>6}  {'top-1 agree':>12}  {'mean_rank':>10}  {'ov@10':>8}  {'ratio':>8}")
print(f"  {'-'*54}")
for K in K_VALUES:
    res = scale_results[K]
    n = len(res)
    n_agree = sum(r["agree"] for r in res)
    mean_r = float(np.mean([r["rank"] for r in res]))
    mean_ov = float(np.mean([r["ov10"] for r in res]))
    exp10 = 10*10/K
    print(f"  {K:>6}  {n_agree}/{n} ({n_agree/n:.3f})  {mean_r:>10.1f}  {mean_ov:>8.2f}  {mean_ov/exp10:>8.2f}x")

# Per-category at each K
print()
print("  Per-category top-1 agree:")
cats = sorted(set(r["cat"] for r in scale_results[234]))
print(f"  {'Category':>16}  " + "  ".join(f"K={K}" for K in K_VALUES))
print(f"  {'-'*65}")
for cat in cats:
    row_str = f"  {cat:>16}  "
    for K in K_VALUES:
        cat_r = [r for r in scale_results[K] if r["cat"]==cat]
        if cat_r:
            ac = sum(r["agree"] for r in cat_r)/len(cat_r)
            row_str += f"{ac:.2f}       "
        else:
            row_str += "N/A        "
    print(row_str)

# Rank stability
print()
print("  Oracle rank stability across K (prompts that agree at K=234):")
agreed_at_234 = [r["prompt"] for r in scale_results[234] if r["agree"]]
for r234 in scale_results[234]:
    if r234["agree"]:
        prompt = r234["prompt"]
        ranks = [next((r["rank"] for r in scale_results[K] if r["prompt"]==prompt), "?") for K in K_VALUES]
        print(f"    {r234['entity']:>12}: {' → '.join(str(rk) for rk in ranks)}")

print()
print("  VERDICT:")
r234 = sum(r["agree"] for r in scale_results[234])/len(scale_results[234])
r_max = max(K_VALUES)
r_max_rate = sum(r["agree"] for r in scale_results[r_max])/len(scale_results[r_max])
print(f"  K=234: {r234:.1%}  K={r_max}: {r_max_rate:.1%}  Delta: {r_max_rate-r234:+.3f}")
if abs(r_max_rate - r234) < 0.1:
    print("  → ROBUST: entity_excl performance stable across vocabulary sizes")
elif r_max_rate < r234 - 0.1:
    print("  → DEGRADES: larger vocab introduces noise; 234-word curation is critical")
else:
    print("  → IMPROVES: larger vocab helps; curation was unnecessary")

with open(OUTPUT_FILE, "w") as f:
    json.dump({K: scale_results[K] for K in K_VALUES}, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 143 complete.")
