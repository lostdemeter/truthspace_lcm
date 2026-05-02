#!/usr/bin/env python3
"""
Day 174 — Multi-Relation Composition in W_E

Directions for different relations are orthogonal (DC 349).
This raises the question: does composing TWO directions (entity + d1 + d2)
produce useful results, or do they destructively interfere?

Four experiments:

  EXP 1: Additive composition
    entity + d1 + d2 → nearest neighbor
    Is the result meaningful? Can you retrieve BOTH attributes simultaneously?
    Test: France + capital_dir → Paris (baseline)
          France + language_dir → French (baseline)
          France + capital_dir + language_dir → ??? (interference test)

  EXP 2: Sequential composition
    Does (entity + d1 + d2) == ((entity + d1) + d2) == entity + (d1 + d2)?
    Additivity test — directions should commute and associate if linear.

  EXP 3: Direction interference
    Does adding direction d2 harm the retrieval quality for d1?
    Metric: does rank of target drop when unrelated direction is added?

  EXP 4: Disambiguation via direction
    "Paris" is ambiguous (city vs name). Can a direction disambiguate?
    "Bank" is ambiguous (river vs financial). Can a direction steer meaning?
    Test cross-domain steering of ambiguous tokens.

  EXP 5: Multi-hop reasoning by direction chain
    France + capital_dir = Paris
    Paris + ?? = French (can we chain?)
    king + gender_dir → queen (Step 1, known)
    queen + capital_dir → ??? (Step 2: capital of queen's country)
    Does chaining directions produce sensible multi-hop answers?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day174_multirelation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Vocabulary ──────────────────────────────────────────────────
VOCAB = [
    # countries and capitals
    "France","Germany","Italy","Spain","Japan","China","India","Russia",
    "Brazil","Mexico","Canada","Poland","Sweden","Korea","Greece","Turkey",
    "Paris","Berlin","Rome","Madrid","Tokyo","Beijing","Delhi","Moscow",
    "Brasilia","Mexico","Ottawa","Warsaw","Stockholm","Seoul","Athens","Ankara",
    # languages
    "French","German","Italian","Spanish","Japanese","Chinese","Hindi","Russian",
    "Portuguese","Swedish","Polish","Korean","Greek","Turkish",
    # gender pairs
    "king","queen","man","woman","boy","girl","prince","princess",
    "lord","lady","actor","actress","waiter","waitress","hero","heroine",
    # metals/materials
    "iron","copper","aluminum","tin","zinc","lead","gold","silver","steel",
    "metal","material","element","mineral","alloy",
    # cities (not capitals)
    "London","New","York","Los","Angeles","Shanghai","Mumbai","Dubai",
    # ambiguous tokens
    "Paris","bank","bat","crane","spring","bark","mine","match","punch","pool",
    # category labels
    "city","capital","country","language","person","name","place",
    "warm","cool","neutral","rocky","gas","Europe","Asia","continent",
    # general
    "hot","cold","big","small","fast","slow","good","bad",
    "the","and","is","are","of","in","a","an",
]

# Training pairs
CAPITAL_PAIRS   = [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                   ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                   ("India","Delhi"),("Russia","Moscow"),("Poland","Warsaw")]
LANGUAGE_PAIRS  = [("France","French"),("Germany","German"),("Italy","Italian"),
                   ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
                   ("Russia","Russian"),("Greece","Greek"),("Sweden","Swedish")]
GENDER_PAIRS    = [("king","queen"),("man","woman"),("boy","girl"),
                   ("prince","princess"),("lord","lady"),("actor","actress")]
METAL_PAIRS     = [("iron","metal"),("copper","metal"),("aluminum","metal"),
                   ("tin","metal"),("zinc","metal"),("lead","metal")]

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

vocab_ok   = [w for w in dict.fromkeys(VOCAB) if tid(w)]
vocab_embs = {w: W_E[tid(w)] for w in vocab_ok}

def make_dir(pairs, excl_src=None, excl_tgt=None):
    ds = [normed(W_E[tid(b)] - W_E[tid(a)])
          for a, b in pairs
          if tid(a) and tid(b) and a != excl_src and b != excl_tgt]
    return normed(np.mean(ds, axis=0)) if ds else None

def top_k(query_emb, k=5, exclude=None):
    excl = set(exclude or [])
    scores = {w: cosine(query_emb, v) for w, v in vocab_embs.items() if w not in excl}
    return sorted(scores.items(), key=lambda x: -x[1])[:k]

def rank_of(query_emb, target):
    scores = {w: cosine(query_emb, vocab_embs[w]) for w in vocab_ok if w in vocab_embs}
    ranked = sorted(vocab_ok, key=lambda w: -scores[w])
    return ranked.index(target) + 1 if target in ranked else -1

# ─── Build directions (LOO for capitals/languages) ───────────────
cap_dir  = make_dir(CAPITAL_PAIRS)
lang_dir = make_dir(LANGUAGE_PAIRS)
gen_dir  = make_dir(GENDER_PAIRS)
met_dir  = make_dir(METAL_PAIRS)

# Verify direction orthogonality
print(f"Direction cosines (should be ~0 across domains):")
dirs = {"capital": cap_dir, "language": lang_dir, "gender": gen_dir, "metal": met_dir}
for n1, d1 in dirs.items():
    for n2, d2 in dirs.items():
        if n1 < n2:
            print(f"  cos({n1}, {n2}) = {cosine(d1, d2):.3f}")
print()

# ─── EXP 1: Additive composition ─────────────────────────────────
print("="*64)
print("EXP 1: Additive Composition  (entity + d1 + d2)")
print("="*64)

test_countries = ["France","Germany","Italy","Spain","Japan","China","Russia"]
results_exp1 = []
for country in test_countries:
    eid = tid(country)
    if not eid: continue
    e = W_E[eid].copy()

    # Baselines
    cap_loos = make_dir(CAPITAL_PAIRS, excl_src=country)
    lang_loos = make_dir(LANGUAGE_PAIRS, excl_src=country)

    t1 = top_k(e + cap_loos, 3, {country})
    t2 = top_k(e + lang_loos, 3, {country})
    t3 = top_k(e + cap_loos + lang_loos, 5, {country})
    t4 = top_k(e + lang_loos + cap_loos, 5, {country})  # commutative check

    # Known targets
    cap_tgt  = dict(CAPITAL_PAIRS).get(country, "?")
    lang_tgt = dict(LANGUAGE_PAIRS).get(country, "?")

    r_cap_single  = rank_of(e + cap_loos, cap_tgt)
    r_cap_dual    = rank_of(e + cap_loos + lang_loos, cap_tgt)
    r_lang_single = rank_of(e + lang_loos, lang_tgt)
    r_lang_dual   = rank_of(e + lang_loos + cap_loos, lang_tgt)

    print(f"\n  {country}:")
    print(f"    + cap_dir  → {[w for w,_ in t1][:3]}  (target: {cap_tgt}, rank={r_cap_single})")
    print(f"    + lang_dir → {[w for w,_ in t2][:3]}  (target: {lang_tgt}, rank={r_lang_single})")
    print(f"    + cap+lang → {[w for w,_ in t3][:3]}")
    print(f"      cap rank in dual: {r_cap_dual}  lang rank in dual: {r_lang_dual}")
    print(f"    + lang+cap → {[w for w,_ in t4][:3]}  (commutative check)")

    results_exp1.append({
        "country": country, "cap_tgt": cap_tgt, "lang_tgt": lang_tgt,
        "r_cap_single": r_cap_single, "r_cap_dual": r_cap_dual,
        "r_lang_single": r_lang_single, "r_lang_dual": r_lang_dual,
    })

# Summary
avg_cap_single  = np.mean([r["r_cap_single"]  for r in results_exp1])
avg_cap_dual    = np.mean([r["r_cap_dual"]    for r in results_exp1])
avg_lang_single = np.mean([r["r_lang_single"] for r in results_exp1])
avg_lang_dual   = np.mean([r["r_lang_dual"]   for r in results_exp1])
print(f"\n  Mean rank comparison:")
print(f"    capital rank:  single={avg_cap_single:.1f}, dual={avg_cap_dual:.1f}")
print(f"    language rank: single={avg_lang_single:.1f}, dual={avg_lang_dual:.1f}")
print(f"    Adding d2 {'HURTS' if avg_cap_dual > avg_cap_single else 'HELPS or NEUTRAL'} capital retrieval")
print(f"    Adding d1 {'HURTS' if avg_lang_dual > avg_lang_single else 'HELPS or NEUTRAL'} language retrieval")

# ─── EXP 4: Disambiguation via direction ─────────────────────────
print()
print("="*64)
print("EXP 4: Disambiguation via Direction")
print("="*64)
print()

# "Paris" as a city vs a person name
paris_ambiguous = ["Paris","London","Athens","Rome","Berlin"]
disambig_vocab = vocab_ok + ["Hilton","Texas","singer","celebrity","model"]
disambig_embs  = {w: W_E[tid(w)] for w in disambig_vocab if tid(w)}

for word in paris_ambiguous:
    if not tid(word): continue
    e = W_E[tid(word)].copy()
    # Without direction
    baseline = [(w, cosine(e, disambig_embs[w]))
                for w in disambig_embs if w != word]
    baseline.sort(key=lambda x: -x[1])
    # With capital_dir (steer toward city/capital meaning)
    steered = [(w, cosine(e + cap_dir, disambig_embs[w]))
               for w in disambig_embs if w != word]
    steered.sort(key=lambda x: -x[1])
    print(f"  '{word}':")
    print(f"    baseline top5: {[w for w,_ in baseline[:5]]}")
    print(f"    +cap_dir top5: {[w for w,_ in steered[:5]]}")

# ─── EXP 5: Multi-hop via direction chain ─────────────────────────
print()
print("="*64)
print("EXP 5: Multi-Hop Direction Chain")
print("="*64)
print()

# Hop 1: country → capital
# Hop 2: ??? → language of (capital's country)
# Can we go France → French, or France → Paris → language_of_France?

for country in ["France","Germany","Japan","Italy"]:
    eid = tid(country)
    if not eid: continue
    e = W_E[eid].copy()

    # Direct: country + language_dir → language
    lang_loos = make_dir(LANGUAGE_PAIRS, excl_src=country)
    top_lang = top_k(e + lang_loos, 3, {country})

    # 2-hop: country → capital (step1), then capital + ??? → language
    cap_loos = make_dir(CAPITAL_PAIRS, excl_src=country)
    step1_top = top_k(e + cap_loos, 1, {country})
    hop1_word = step1_top[0][0]
    hop1_eid  = tid(hop1_word)
    if hop1_eid:
        e_hop1 = W_E[hop1_eid].copy()
        top_from_cap = top_k(e_hop1 + lang_loos, 3, {hop1_word, country})
    else:
        top_from_cap = []

    tgt_lang = dict(LANGUAGE_PAIRS).get(country, "?")
    tgt_cap  = dict(CAPITAL_PAIRS).get(country, "?")
    print(f"  {country}:")
    print(f"    Direct:  {country} + lang_dir → {[w for w,_ in top_lang]}  (target: {tgt_lang})")
    print(f"    Hop 1:   {country} + cap_dir → {hop1_word} (target: {tgt_cap})")
    print(f"    Hop 2:   {hop1_word} + lang_dir → {[w for w,_ in top_from_cap]}")
    print()

# ─── Summary ─────────────────────────────────────────────────────
print("="*64)
print("Summary")
print("="*64)
print(f"  Exp 1: Adding d2 to entity+d1 →")
print(f"    cap rank: single={avg_cap_single:.1f}, dual={avg_cap_dual:.1f}")
print(f"    lang rank: single={avg_lang_single:.1f}, dual={avg_lang_dual:.1f}")
print(f"  Key question: is W_E linear? Do directions compose?")
print(f"  If dual rank ≈ single rank: directions are orthogonal and non-interfering")
print(f"  If dual rank >> single rank: directions interfere destructively")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"exp1": results_exp1,
               "summary": {"avg_cap_single": avg_cap_single, "avg_cap_dual": avg_cap_dual,
                            "avg_lang_single": avg_lang_single, "avg_lang_dual": avg_lang_dual}},
              f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 174 complete.")
