#!/usr/bin/env python3
"""
Day 147 — Universal Directions in W_E

Day 146: individual pair directions (Japan-Germany) work for 91% of country pairs.
Question: is there a SINGLE mean direction that works for all?

EXPERIMENT 1: Universal Capital Direction
  mean_dir = mean(normalize(capital_i - country_i)) for all (country, capital) pairs
  Then: country_j + mean_dir * scale → capital_j?

EXPERIMENT 2: Antonym Vector Arithmetic (model logprob)
  hot + (cold - fast) → slow? (analogical arithmetic via model)
  scale sweep, same as Day 146 Surgery C

EXPERIMENT 3: Gender Vector Arithmetic (model logprob)
  king + (actress - actor) → queen?
  son + (daughter - son) → daughter? (degenerate, skip self-pairs)
  king + mean_gender_dir → queen?

EXPERIMENT 4: Mean Antonym Direction
  mean_dir_antonym = mean(normalize(antonym_j - word_j))
  word_i + mean_dir_antonym → antonym_i?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day147_universal_directions.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().clone()
print(f"  hidden={H}\n")

def get_tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def logprob_rank(prompt, target_word):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1)
    tid = get_tid(target_word)
    if tid is None: return -1
    return int((lp > lp[tid]).sum().item())

def top5(prompt):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1)
    return [tok.decode([t]).strip() for t in lp.topk(5).indices.tolist()]

def reset_we():
    with torch.no_grad(): model.model.embed_tokens.weight.copy_(W_E)

def patch(tid, emb):
    with torch.no_grad(): model.model.embed_tokens.weight[tid] = torch.tensor(emb, dtype=torch.float32)

def emb(word):
    tid = get_tid(word)
    return W_E[tid].numpy().astype(np.float32) if tid else None

COUNTRY_CAPITAL = {
    "France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid",
    "Japan":"Tokyo","China":"Beijing","Russia":"Moscow","Brazil":"Brasilia",
    "Egypt":"Cairo","Greece":"Athens","Poland":"Warsaw","Sweden":"Stockholm",
}
ANTONYM_PAIRS = [
    ("hot","cold"),("big","small"),("fast","slow"),("dark","light"),
    ("good","bad"),("young","old"),("rich","poor"),("clean","dirty"),
    ("loud","quiet"),("strong","weak"),("early","late"),("easy","hard"),
]
GENDER_PAIRS = [
    ("king","queen"),("prince","princess"),("actor","actress"),
    ("son","daughter"),("father","mother"),("brother","sister"),
    ("man","woman"),("boy","girl"),("duke","duchess"),
]

all_results = {}

# ─────────────────────────────────────────────────────────────
print("="*66)
print("PART 1: Universal Capital Direction")
print("="*66)

# Build mean direction capital → country (and country → capital)
cc_dirs = []  # country → capital direction
for country, capital in COUNTRY_CAPITAL.items():
    ec = emb(country); ep = emb(capital)
    if ec is not None and ep is not None:
        d = normed(ep - ec); cc_dirs.append(d)
mean_cap_dir = np.mean(cc_dirs, axis=0)
mean_cap_dir = normed(mean_cap_dir)
print(f"  Computed mean capital direction from {len(cc_dirs)} pairs.\n")

results_p1 = []
for country, capital in COUNTRY_CAPITAL.items():
    ec = emb(country); ep = emb(capital)
    if ec is None or ep is None: continue
    # Pure arithmetic: country + mean_dir → nearest capital?
    result_emb = ec + mean_cap_dir
    # Find nearest capital in W_E
    best_cap = None; best_cos = -1
    for cap, _ in COUNTRY_CAPITAL.items():
        if cap == country: continue  # don't match own country
    for cap2, _ in COUNTRY_CAPITAL.items():
        ecap = emb(cap2)
        if ecap is None: continue
        c = cosine(result_emb, ecap)
        if c > best_cos: best_cos = c; best_cap = cap2
    hit = best_cap == capital
    # Also check via capitals only
    cap_embs = {cap: emb(cap) for _,cap in COUNTRY_CAPITAL.items() if emb(cap) is not None}
    best_capital_only = max(cap_embs.keys(), key=lambda c: cosine(result_emb, cap_embs[c]))
    hit_cap = best_capital_only == capital

    results_p1.append({"country": country, "capital": capital, "nearest": best_cap,
                        "nearest_cap_only": best_capital_only, "hit": hit, "hit_cap": hit_cap})
    status = "✓" if hit_cap else "✗"
    print(f"  {country:>12} + mean_dir → {status} {best_capital_only:<12} (expected {capital})")

n1 = len(results_p1)
h1 = sum(r["hit_cap"] for r in results_p1)
print(f"\n  Universal capital direction: {h1}/{n1} = {h1/n1:.3f}")

# Model-level test: does adding mean_cap_dir to country embedding change prediction?
print("\n  Model test (surgery + mean_dir):")
model_results_p1 = []
for country, capital in list(COUNTRY_CAPITAL.items())[:6]:
    ec = emb(country)
    if ec is None: continue
    reset_we()
    tid_c = get_tid(country)
    new_emb = ec + mean_cap_dir
    patch(tid_c, new_emb)
    prompt = f"The capital city of {country} is"
    rank_cap = logprob_rank(prompt, capital)
    t5 = top5(prompt)
    print(f"  {country:>10} + mean_dir: {capital} rank={rank_cap}  top3={t5[:3]}")
    model_results_p1.append({"country": country, "capital": capital, "rank_cap": rank_cap, "top5": t5})
reset_we()

all_results["part1"] = {"pure_we": results_p1, "model": model_results_p1,
                        "accuracy": h1/n1 if n1 else 0}

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 2: Antonym Vector Arithmetic (Model Logprob)")
print("="*66)
print()

SCALE = 1.0
results_p2 = []
for src_w, src_ant in ANTONYM_PAIRS:
    for tgt_w, tgt_ant in ANTONYM_PAIRS:
        if (src_w, src_ant) == (tgt_w, tgt_ant): continue
        if src_w == tgt_w or src_ant == tgt_ant: continue
        e_src = emb(src_w); e_tgt = emb(tgt_w); e_tgt_ant = emb(tgt_ant)
        if any(x is None for x in [e_src, e_tgt, e_tgt_ant]): continue
        direction = normed(e_tgt_ant - e_tgt)
        new_e = e_src + SCALE * direction
        tid_src = get_tid(src_w)
        if tid_src is None: continue
        reset_we(); patch(tid_src, new_e)
        prompt = f"The opposite of {src_w} is"
        rank_ant = logprob_rank(prompt, src_ant)
        t5 = top5(prompt)
        results_p2.append({
            "src": src_w, "src_ant": src_ant, "tgt": tgt_w, "tgt_ant": tgt_ant,
            "rank": rank_ant, "top5": t5, "hit": rank_ant <= 2,
        })

reset_we()
n2 = len(results_p2); h2 = sum(r["hit"] for r in results_p2)
print(f"  Antonym arithmetic (all pairs): {h2}/{n2} = {h2/n2:.3f} in top-3")
print()
print("  Best hits (rank \u2264 2):")
for r in sorted(results_p2, key=lambda x: x["rank"])[:10]:
    print(f"    {r['src']} + ({r['tgt_ant']}-{r['tgt']}) \u2192 {r['src_ant']} rank={r['rank']}  top3={r['top5'][:3]}")
print()
print("  Worst misses:")
for r in sorted(results_p2, key=lambda x: -x["rank"])[:5]:
    print(f"    {r['src']} + ({r['tgt_ant']}-{r['tgt']}) \u2192 {r['src_ant']} rank={r['rank']}  top3={r['top5'][:3]}")
all_results["part2"] = {"n": n2, "hits": h2, "accuracy": h2/n2 if n2 else 0}

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 3: Universal Antonym Direction")
print("="*66)

ant_dirs = []
for w, ant in ANTONYM_PAIRS:
    ew = emb(w); ea = emb(ant)
    if ew is not None and ea is not None:
        ant_dirs.append(normed(ea - ew))
mean_ant_dir = normed(np.mean(ant_dirs, axis=0))
print(f"  Mean antonym direction from {len(ant_dirs)} pairs.\n")

results_p3 = []
for src_w, src_ant in ANTONYM_PAIRS:
    ew = emb(src_w); ea = emb(src_ant)
    if ew is None or ea is None: continue
    result = ew + mean_ant_dir
    # Nearest antonym word in our set
    ant_pool = {w: emb(w) for w,_ in ANTONYM_PAIRS} | {a: emb(a) for _,a in ANTONYM_PAIRS}
    ant_pool = {k: v for k,v in ant_pool.items() if v is not None and k != src_w}
    nearest = max(ant_pool.keys(), key=lambda w: cosine(result, ant_pool[w]))
    hit = nearest == src_ant
    results_p3.append({"src": src_w, "src_ant": src_ant, "nearest": nearest, "hit": hit})
    status = "✓" if hit else "✗"
    print(f"  {src_w:>10} + mean_ant_dir \u2192 {status} {nearest:<10} (expected {src_ant})")

n3 = len(results_p3); h3 = sum(r["hit"] for r in results_p3)
print(f"\n  Universal antonym direction: {h3}/{n3} = {h3/n3:.3f}")
all_results["part3"] = {"results": results_p3, "accuracy": h3/n3 if n3 else 0}

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 4: Gender Vector Arithmetic (Model + Pure W_E)")
print("="*66)

gender_dirs = []
for masc, fem in GENDER_PAIRS:
    em = emb(masc); ef = emb(fem)
    if em is not None and ef is not None:
        gender_dirs.append(normed(ef - em))
mean_gender_dir = normed(np.mean(gender_dirs, axis=0))
print(f"  Mean gender direction from {len(gender_dirs)} pairs.\n")

results_p4_pure = []
for src_m, src_f in GENDER_PAIRS:
    em = emb(src_m); ef = emb(src_f)
    if em is None or ef is None: continue
    result = em + mean_gender_dir
    pool = {w: emb(w) for w,_ in GENDER_PAIRS} | {f: emb(f) for _,f in GENDER_PAIRS}
    pool = {k: v for k,v in pool.items() if v is not None and k != src_m}
    nearest = max(pool.keys(), key=lambda w: cosine(result, pool[w]))
    hit = nearest == src_f
    results_p4_pure.append({"src": src_m, "target": src_f, "nearest": nearest, "hit": hit})
    status = "✓" if hit else "✗"
    print(f"  {src_m:>10} + mean_gender_dir \u2192 {status} {nearest:<12} (expected {src_f})")

n4 = len(results_p4_pure); h4 = sum(r["hit"] for r in results_p4_pure)
print(f"\n  Universal gender direction (pure W_E): {h4}/{n4} = {h4/n4:.3f}")
all_results["part4"] = {"results": results_p4_pure, "accuracy": h4/n4 if n4 else 0}

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("Summary — Universal Directions")
print("="*66)
print(f"""
  Part 1: Universal capital direction   {h1}/{n1} = {h1/n1:.3f}  (pure W_E)
  Part 2: Antonym arithmetic (model)    {h2}/{n2} = {h2/n2:.3f}  (top-3 match)
  Part 3: Universal antonym direction   {h3}/{n3} = {h3/n3:.3f}  (pure W_E)
  Part 4: Universal gender direction    {h4}/{n4} = {h4/n4:.3f}  (pure W_E)
""")

if h1/n1 >= 0.8:
    print("  Universal capital direction: CONFIRMED (>=80%)")
if h3/n3 >= 0.6:
    print("  Universal antonym direction: CONFIRMED (>=60%)")
if h4/n4 >= 0.6:
    print("  Universal gender direction: CONFIRMED (>=60%)")

reset_we()
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 147 complete.")
