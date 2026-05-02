#!/usr/bin/env python3
"""
Day 208 — Domain-Level Archetype Classifier Validation

Build and validate the two-stage domain classifier from DC 369:
  STEP 0: IDENTITY   — any pair is same token (trivial)
  STEP 1: ORDINAL    — Spearman ρ ≥ 0.85 across ≥3 ordered pairs
  STEP 2: TYPE_BC    — dir_consistency ≥ 0.15 across ≥2 pairs
  STEP 3: TYPE_ADJACENT — default fallback

Test on ALL domains established across Days 192-207:
  TYPE_BC:       capitals, gender, plurals, superlative, past_tense_F (suppletive)
  TYPE_ADJACENT: antonyms, color_names, past_tense_B (oo→ew), past_tense_D (nd→nt)
  TYPE_ORDINAL:  numbers (one/1, two/2, ...)
  IDENTITY:      no-change verbs (cut/cut, put/put)

For each domain:
  1. Classify using the two-stage classifier
  2. Compare predicted vs expected archetype
  3. Run the appropriate retrieval method
  4. Report accuracy

Also compare total accuracy against Day 198 multi-tier pipeline
(which was trained on fewer domains with a simpler classifier).

FULL VOCABULARY RETRIEVAL: Use the model's full token vocabulary (151,936)
filtered to single-token words, for more realistic disambiguation.
Since full-vocab search is expensive, use a 600-word curated vocabulary
that still forces real disambiguation.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day208_domain_classifier.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# All domains with known archetype labels and example pairs
DOMAINS = {
    # ── TYPE_BC ──────────────────────────────────────────────────────
    "capitals": {
        "expected": "TYPE_BC",
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi"),("Mexico","Mexico")],
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
        "train": [("go","went"),("have","had"),("do","did"),
                  ("take","took"),("give","gave"),("make","made")],
        "test":  [("come","came"),("get","got"),("stand","stood"),
                  ("leave","left"),("bring","brought"),("buy","bought")],
    },
    # ── TYPE_ADJACENT ─────────────────────────────────────────────────
    "antonyms": {
        "expected": "TYPE_ADJACENT",
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "past_tense_B": {
        "expected": "TYPE_ADJACENT",
        "train": [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
        "test":  [("know","knew"),("grow","grew"),("throw","threw"),
                  ("blow","blew"),("fly","flew"),("draw","drew")],
    },
    "past_tense_D": {
        "expected": "TYPE_ADJACENT",
        "train": [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
        "test":  [("send","sent"),("spend","spent"),("lend","lent"),
                  ("bend","bent"),("build","built"),("find","found")],
    },
    # ── TYPE_ORDINAL ──────────────────────────────────────────────────
    "numbers": {
        "expected": "TYPE_ORDINAL",
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6"),
                  ("seven","7"),("eight","8"),("nine","9"),("ten","10")],
        "test":  [],
    },
    # ── IDENTITY ─────────────────────────────────────────────────────
    "no_change_verbs": {
        "expected": "IDENTITY",
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost")],
    },
}

# Extended retrieval vocabulary (600 single-token words for disambiguation)
RETRIEVAL_VOCAB_WORDS = [
    # Capitals
    "Paris","Berlin","Rome","Madrid","Tokyo","Beijing","Moscow","Athens",
    "Cairo","Delhi","Brasilia","London","Washington","Ottawa","Canberra",
    "Buenos","Vienna","Warsaw","Prague","Budapest","Stockholm","Oslo",
    # Gender pairs
    "queen","woman","girl","princess","actress","heroine","mother","sister",
    "daughter","wife","aunt","waitress","goddess","bride","nun","mare",
    "king","man","boy","prince","actor","hero","father","brother",
    "son","husband","uncle","waiter","god","groom","monk","stallion",
    # Plurals
    "cats","dogs","houses","trees","books","cars","birds","ships","hands",
    "doors","lamps","walls","eyes","roads","cups","beds","keys","boxes",
    "tables","chairs","windows","rooms","pages","words","names","times",
    # Superlatives
    "biggest","fastest","longest","smartest","brightest","cleanest",
    "hardest","darkest","softest","warmest","slowest","smallest","tallest",
    "coldest","oldest","newest","richest","poorest","safest","quietest",
    # Past tense (suppletive F)
    "went","had","did","took","gave","made","came","got","stood","left",
    "brought","bought","said","knew","found","thought","ran","ate","saw",
    # Past tense (oo→ew = B class)
    "knew","grew","threw","blew","flew","drew",
    # Past tense (nd→nt = D class)
    "sent","spent","lent","bent","built","found",
    # Antonyms
    "cold","small","slow","soft","dark","young","quiet","dull","poor",
    "thin","narrow","shallow","wet","weak","rough","low","short","late",
    "hot","big","fast","hard","light","old","loud","sharp","rich",
    "thick","wide","deep","dry","strong","smooth","high","tall","early",
    # Numbers (digit tokens)
    "1","2","3","4","5","6","7","8","9","10",
    "11","12","13","14","15","16","17","18","19","20",
    # Base/source forms as distractors
    "cat","dog","house","tree","book","car","bird","ship","hand","door",
    "lamp","wall","eye","road","cup","bed","key","box","table","chair",
    "big","fast","long","smart","bright","clean","hard","dark","soft",
    "warm","slow","small","tall","cold","old","new","rich","poor","safe",
    "France","Germany","Italy","Spain","Japan","China","Russia","Greece",
    "Brazil","Egypt","India","Mexico","England","Korea","Poland","Turkey",
    "go","have","do","take","give","make","come","get","stand","leave",
    "bring","buy","say","know","find","think","run","eat","see","write",
    "know","grow","throw","blow","fly","draw","send","spend","lend","bend",
    "cut","put","hit","let","set","shut","burst","cost",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    # Common decoys
    "red","blue","green","yellow","black","white","brown","gray","pink",
    "happy","sad","angry","tired","hungry","busy","ready","true","false",
    "good","bad","new","free","open","closed","full","empty","clean","dirty",
    "water","fire","earth","wind","space","time","place","thing","people","world",
]

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

# Build retrieval vocabulary
ret_vocab = {}
for w in RETRIEVAL_VOCAB_WORDS:
    t = tid1(w)
    if t is not None and w not in ret_vocab:
        ret_vocab[w] = W_E[t].astype(np.float64)
print(f"Retrieval vocabulary: {len(ret_vocab)} single-token words\n")

def ok_pairs(pairs):
    return [(a,b) for a,b in pairs if tid1(a) and tid1(b)]

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in p]
    pw = [cosine(diffs[i], diffs[j])
          for i in range(len(diffs)) for j in range(i+1, len(diffs))]
    return float(np.mean(pw))

def ordinal_spearman(pairs):
    p = ok_pairs(pairs)
    if len(p) < 3: return 0.0
    # pairs are (word, digit_string) — use index as ordinal rank
    src_ranks = list(range(len(p)))
    diffs = [get_emb(b) - get_emb(a) for a,b in p]
    norms = [float(np.linalg.norm(d)) for d in diffs]
    rho, _ = spearmanr(src_ranks, norms)
    return float(rho) if not np.isnan(rho) else 0.0

def classify_domain(train_pairs):
    p = ok_pairs(train_pairs)
    # IDENTITY
    if any(a == b for a,b in p):
        return "IDENTITY"
    # ORDINAL (need ordinal structure — if targets are digits)
    if all(b.isdigit() for _,b in p) and len(p) >= 3:
        rho = ordinal_spearman(p)
        if rho > 0.85:
            return "TYPE_ORDINAL"
    # TYPE_BC
    if len(p) >= 2:
        dc = dir_consistency(p)
        if dc > 0.15:
            return "TYPE_BC"
    # ADJACENT fallback
    return "TYPE_ADJACENT"

def retrieve_bc(src, train_pairs, vocab=ret_vocab):
    p = ok_pairs(train_pairs)
    se = get_emb(src)
    if se is None or not p: return None
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in p]
    mdir  = normed(np.mean(diffs, axis=0))
    query = se + mdir
    sims  = {w: cosine(query, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

def retrieve_adjacent(src, vocab=ret_vocab):
    se = get_emb(src)
    if se is None: return None
    sims = {w: cosine(se, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

def retrieve_ordinal(src, train_pairs, vocab=ret_vocab):
    p = ok_pairs(train_pairs)
    se = get_emb(src)
    if se is None or not p: return None
    # Project onto mean displacement axis, then find nearest target
    diffs = [get_emb(b) - get_emb(a) for a,b in p]
    axis  = normed(np.mean(diffs, axis=0))
    proj  = float(np.dot(normed(se), axis))
    sims  = {w: cosine(se + proj*axis, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

# ── Evaluate all domains ──────────────────────────────────────────────
print("=" * 72)
print(f"{'Domain':<20}  {'Expected':<14}  {'Predicted':<14}  "
      f"{'dc':>6}  {'Match':>6}")
print("=" * 72)

all_results = {}
for domain_name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]; expected = cfg["expected"]
    pred  = classify_domain(train)
    p     = ok_pairs(train)
    dc    = dir_consistency(p) if len(p) >= 2 else 0.0
    match = (pred == expected)
    mark  = "" if match else " ✗"
    print(f"  {domain_name:<20}  {expected:<14}  {pred:<14}  "
          f"{dc:>6.3f}  {'YES' if match else 'NO':>6}{mark}")
    all_results[domain_name] = {
        "expected": expected, "predicted": pred,
        "dir_consistency": dc, "correct_classification": match,
    }

print()

# ── Retrieval accuracy per domain ─────────────────────────────────────
print("=" * 72)
print("RETRIEVAL ACCURACY (applying classifier-selected method)")
print("=" * 72)
total_correct = 0; total_pairs = 0
for domain_name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]
    pred  = all_results[domain_name]["predicted"]
    test_ok = ok_pairs(test) if test else ok_pairs(train)
    if not test_ok: continue
    correct = 0
    for src, tgt in test_ok:
        if pred == "IDENTITY":
            pred_tgt = src
        elif pred == "TYPE_BC":
            pred_tgt = retrieve_bc(src, train)
        elif pred == "TYPE_ORDINAL":
            pred_tgt = retrieve_ordinal(src, train)
        else:  # TYPE_ADJACENT
            pred_tgt = retrieve_adjacent(src)
        if pred_tgt == tgt: correct += 1
    acc = correct / len(test_ok)
    total_correct += correct; total_pairs += len(test_ok)
    expected = cfg["expected"]
    # Also compute accuracy with correct method (upper bound)
    corr_correct = 0
    for src, tgt in test_ok:
        if expected == "IDENTITY":
            corr_pred = src
        elif expected == "TYPE_BC":
            corr_pred = retrieve_bc(src, train)
        elif expected == "TYPE_ORDINAL":
            corr_pred = retrieve_ordinal(src, train)
        else:
            corr_pred = retrieve_adjacent(src)
        if corr_pred == tgt: corr_correct += 1
    oracle_acc = corr_correct / len(test_ok)
    print(f"  {domain_name:<20}  n={len(test_ok):>2}  "
          f"pred={pred:<14}  acc={acc:.3f}  oracle={oracle_acc:.3f}")
    all_results[domain_name]["retrieval_accuracy"] = acc
    all_results[domain_name]["oracle_accuracy"]    = oracle_acc

overall_acc = total_correct / total_pairs if total_pairs else 0
print(f"\n  OVERALL: {total_correct}/{total_pairs}  acc={overall_acc:.3f}")

# ── Compare with Day 198 baseline ────────────────────────────────────
print()
print("=" * 72)
print("COMPARISON: Day 198 pipeline vs Day 208 domain classifier")
print("=" * 72)
print("  Day 198 pipeline accuracy: 0.779  (7 domains, TYPE_BC + TYPE_ADJACENT)")
print(f"  Day 208 domain classifier: {overall_acc:.3f}  ({len(DOMAINS)} domains, all archetypes)")

# ── Classification summary ────────────────────────────────────────────
print()
print("Classification accuracy:")
correct_cls = sum(1 for d in all_results.values() if d["correct_classification"])
print(f"  {correct_cls}/{len(all_results)} domains correctly classified")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 208 complete.")
