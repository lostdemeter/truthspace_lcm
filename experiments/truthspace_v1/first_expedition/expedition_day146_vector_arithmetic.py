#!/usr/bin/env python3
"""
Day 146 — Systematic Vector Arithmetic in W_E

Day 145 showed: France + (Japan - Germany) direction → Tokyo rank=1.
This is the word2vec "king - man + woman = queen" result for factual knowledge.

QUESTION: How general is this?
  Does France + (China - Germany) → Beijing?
  Does France + (Brazil - Germany) → Brasilia?
  Does Italy  + (Japan  - Germany) → Tokyo?

TEST: All (source_country, target_country, base_country) combinations
  where source ≠ target ≠ base, and we ask:
  source + normalize(target - base) × scale → target_capital at rank 1?

ALSO TEST:
  Semantic vector arithmetic for other knowledge types:
    antonyms: hot + (cold - fast) → slow?
    gender:   king + (queen - prince) → princess?
    capitals: France + (Japan - Germany) at various scales

METRIC: top-1 rank of target_capital in the surgically modified model.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day146_vector_arithmetic.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().clone()  # frozen copy
print(f"  hidden={H}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def logprob_rank(prompt, target_word, model):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1)
    tid = get_token_id(target_word)
    if tid is None: return -1
    return int((lp > lp[tid]).sum().item())

def top3_tokens(prompt, model):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1)
    return [tok.decode([t]).strip() for t in lp.topk(3).indices.tolist()]

def reset_we(m): 
    with torch.no_grad(): m.model.embed_tokens.weight.copy_(W_E)

def patch_token(m, tid, new_emb):
    with torch.no_grad(): m.model.embed_tokens.weight[tid] = torch.tensor(new_emb, dtype=torch.float32)

# Countries and capitals
COUNTRY_CAPITAL = {
    "France":    "Paris",
    "Germany":   "Berlin",
    "Italy":     "Rome",
    "Spain":     "Madrid",
    "Japan":     "Tokyo",
    "China":     "Beijing",
    "Russia":    "Moscow",
    "Brazil":    "Brasilia",
    "Egypt":     "Cairo",
    "Greece":    "Athens",
    "Poland":    "Warsaw",
    "Sweden":    "Stockholm",
}

PROMPT_TEMPLATE = "The capital city of {} is"

print("Building embedding dictionary ...")
tok_ids = {}; embs = {}
for w in list(COUNTRY_CAPITAL.keys()) + list(COUNTRY_CAPITAL.values()):
    tid = get_token_id(w)
    if tid:
        tok_ids[w] = tid
        embs[w] = W_E[tid].numpy().astype(np.float32)
    else:
        print(f"  WARNING: {w!r} is not single-token")
print()

# PART 1: Country vector arithmetic — France + (target - base) → target_capital
print("="*72)
print("PART 1: Capital City Vector Arithmetic")
print("France + (target_country - base_country) → target_capital?")
print("="*72)
print()

source_country = "France"
base_country   = "Germany"
SCALE          = 1.0  # from Day 145 Surgery C

results_part1 = []
for target_country, target_capital in COUNTRY_CAPITAL.items():
    if target_country in (source_country, base_country): continue
    if target_country not in embs or base_country not in embs: continue
    if source_country not in embs: continue

    src_emb  = embs[source_country].copy()
    tgt_emb  = embs[target_country]
    base_emb = embs[base_country]
    direction = normed(tgt_emb - base_emb)
    new_emb  = src_emb + SCALE * direction

    reset_we(model)
    patch_token(model, tok_ids[source_country], new_emb)

    prompt = PROMPT_TEMPLATE.format(source_country)
    rank_src_cap  = logprob_rank(prompt, COUNTRY_CAPITAL[source_country], model)
    rank_tgt_cap  = logprob_rank(prompt, target_capital, model)
    top3           = top3_tokens(prompt, model)

    win = "✓" if rank_tgt_cap < 5 else "~" if rank_tgt_cap < 50 else "✗"
    print(f"  {source_country} + ({target_country}-{base_country}) → "
          f"{target_capital}? {win} rank={rank_tgt_cap:>4}  "
          f"(src_cap={COUNTRY_CAPITAL[source_country]} rank={rank_src_cap:>4})  "
          f"top3={top3}")
    results_part1.append({
        "source": source_country, "target": target_country, "base": base_country,
        "target_capital": target_capital, "rank": rank_tgt_cap,
        "src_cap_rank": rank_src_cap, "top3": top3,
        "win": win,
    })

# PART 2: Does the arithmetic work for any source country?
print()
print("="*72)
print("PART 2: Source Country Generalization")
print("source + (Japan - Germany) → Tokyo?  [Day 145 best direction]")
print("="*72)
print()

target_country = "Japan"; base_country = "Germany"; target_capital = "Tokyo"
if target_country in embs and base_country in embs:
    direction = normed(embs[target_country] - embs[base_country])
    results_part2 = []
    for source_country, source_capital in COUNTRY_CAPITAL.items():
        if source_country in (target_country, base_country): continue
        if source_country not in embs: continue

        src_emb = embs[source_country].copy()
        new_emb = src_emb + SCALE * direction

        reset_we(model)
        patch_token(model, tok_ids[source_country], new_emb)
        prompt = PROMPT_TEMPLATE.format(source_country)
        rank_tgt = logprob_rank(prompt, target_capital, model)
        rank_src = logprob_rank(prompt, source_capital, model)
        top3     = top3_tokens(prompt, model)
        win = "✓" if rank_tgt < 5 else "~" if rank_tgt < 50 else "✗"
        print(f"  {source_country:>10} + (Japan-Germany) → Tokyo? {win} "
              f"rank={rank_tgt:>4}  (own={source_capital} rank={rank_src:>4})  top3={top3}")
        results_part2.append({
            "source": source_country, "rank_tgt": rank_tgt, "rank_src": rank_src, "top3": top3, "win": win
        })

# PART 3: Antonym vector arithmetic
print()
print("="*72)
print("PART 3: Antonym Vector Arithmetic")
print("hot + (cold - fast) → slow?")
print("="*72)

ANTONYM_PAIRS = [
    ("hot","cold"), ("big","small"), ("fast","slow"),
    ("dark","light"), ("good","bad"), ("young","old"),
    ("rich","poor"), ("clean","dirty"), ("loud","quiet"),
]

antonym_words = set(w for p in ANTONYM_PAIRS for w in p)
ant_ids = {}; ant_embs = {}
for w in antonym_words:
    tid = get_token_id(w)
    if tid:
        ant_ids[w] = tid
        ant_embs[w] = W_E[tid].numpy().astype(np.float32)

print()
results_part3 = []
for src_word, src_antonym in ANTONYM_PAIRS:
    for tgt_word, tgt_antonym in ANTONYM_PAIRS:
        if (src_word, src_antonym) == (tgt_word, tgt_antonym): continue
        if src_word not in ant_ids or tgt_word not in ant_ids: continue
        if tgt_antonym not in ant_ids: continue
        # Arithmetic: src_word + (tgt_antonym - tgt_word) should give src_antonym?
        # hot + (slow - fast) → cold?
        if tgt_word == src_word or tgt_antonym == src_antonym: continue
        direction = normed(ant_embs[tgt_antonym] - ant_embs[tgt_word])
        new_emb   = ant_embs[src_word] + SCALE * direction
        cos_to_expected = cosine(new_emb, ant_embs.get(src_antonym, np.zeros(H)))
        results_part3.append({
            "src": src_word, "tgt": tgt_word, "tgt_ant": tgt_antonym,
            "expected": src_antonym, "cos_expected": cos_to_expected,
        })

# Sort by cos_expected
results_part3.sort(key=lambda r: -r["cos_expected"])
print("  Top-10 antonym vector arithmetic by cosine similarity:")
print(f"  {'src + (tgt_ant - tgt)':>32}  {'expected':>10}  cos")
for r in results_part3[:10]:
    label = f"{r['src']} + ({r['tgt_ant']} - {r['tgt']})"
    print(f"  {label:>32}  →  {r['expected']:>10}  {r['cos_expected']:.3f}")

# PART 4: Purely embedding arithmetic (no model forward pass needed)
print()
print("="*72)
print("PART 4: Pure W_E Vector Arithmetic (entity_excl style)")
print("France + (Japan - Germany) in W_E space → nearest capital?")
print("="*72)

all_capital_ids = {cap: tok_ids[cap] for cap in COUNTRY_CAPITAL.values() if cap in tok_ids}
all_country_ids = {cnt: tok_ids[cnt] for cnt in COUNTRY_CAPITAL.keys() if cnt in tok_ids}

print()
results_part4 = []
for source_c, source_cap in COUNTRY_CAPITAL.items():
    for target_c, target_cap in COUNTRY_CAPITAL.items():
        if source_c == target_c: continue
        if source_c not in embs or target_c not in embs: continue
        for base_c in ["Germany", "France", "Japan"]:
            if base_c in (source_c, target_c): continue
            if base_c not in embs: continue
            direction = normed(embs[target_c] - embs[base_c])
            result_emb = embs[source_c] + SCALE * direction
            # Find nearest capital (excluding source capital)
            best_cap = None; best_cos = -1
            for cap, cap_id in all_capital_ids.items():
                if cap == source_cap: continue
                c = cosine(result_emb, W_E[cap_id].numpy().astype(np.float32))
                if c > best_cos: best_cos = c; best_cap = cap
            results_part4.append({
                "source": source_c, "target": target_c, "base": base_c,
                "result_emb_nearest": best_cap, "target_cap": target_cap,
                "hit": best_cap == target_cap, "cos": best_cos,
            })

n4 = len(results_part4); hits4 = sum(r["hit"] for r in results_part4)
print(f"  Pure W_E arithmetic: {hits4}/{n4} = {hits4/n4:.3f}")
print()
print("  Hits:")
for r in results_part4:
    if r["hit"]:
        print(f"    {r['source']}+({r['target']}-{r['base']}) → {r['result_emb_nearest']} ✓")
print()
print("  Misses (sample):")
for r in results_part4[:8]:
    if not r["hit"]:
        print(f"    {r['source']}+({r['target']}-{r['base']}) → got={r['result_emb_nearest']} expected={r['target_cap']}")

reset_we(model)

print()
print("="*72)
print("Summary")
print("="*72)
n1 = len(results_part1); hits1 = sum(1 for r in results_part1 if r["win"]=="✓")
top5_1 = sum(1 for r in results_part1 if r["rank"] < 5)
print(f"\n  Part 1 (France + target-Germany → target_cap): {top5_1}/{n1} in top-5 rank")
n2 = len(results_part2); top5_2 = sum(1 for r in results_part2 if r["rank_tgt"] < 5)
print(f"  Part 2 (source + Japan-Germany → Tokyo):       {top5_2}/{n2} in top-5 rank")
print(f"  Part 4 (pure W_E arithmetic):                  {hits4}/{n4} = {hits4/n4:.3f}")

if top5_1/n1 >= 0.5:
    print(f"\n  VERDICT: Vector arithmetic GENERALIZES ({top5_1}/{n1} = {top5_1/n1:.0%})")
else:
    print(f"\n  VERDICT: Vector arithmetic PARTIAL ({top5_1}/{n1} = {top5_1/n1:.0%})")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "part1": results_part1, "part2": results_part2,
        "part3_top10": results_part3[:10], "part4_summary": {"hits":hits4,"total":n4}
    }, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 146 complete.")
