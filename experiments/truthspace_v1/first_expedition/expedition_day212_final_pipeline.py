#!/usr/bin/env python3
"""
Day 212 — Full Pipeline Reassembly

Incorporates all fixes from Days 206-211:
  - Day 206: per-pair archetype detection fails; use domain-level
  - Day 208: dir_consistency threshold=0.15 for TYPE_BC
  - Day 210: numbers dc=0.850 → TYPE_BC (not ordinal); remove Spearman
  - Day 210: antonym axes orthogonal; TYPE_ANTONYM needs attribute label
  - Day 211: TYPE_BC_DIGIT, DUAL_ARCHETYPE, TYPE_ANTONYM added to taxonomy

REVISED PIPELINE (v3):
  STEP 0: IDENTITY       — norm(tgt-src) < 0.05
  STEP 1: TYPE_BC        — dir_consistency(train) > 0.15, k≥2
                           (catches numbers dc=0.850, capitals dc=0.368, etc.)
  STEP 2: TYPE_ANTONYM   — if domain_label indicates antonym + attribute known
                           apply per-attribute axis retrieval
  STEP 3: TYPE_ADJACENT  — fallback (proximity nn)

DOMAINS TESTED (11 total, adding numbers):
  TYPE_BC:       capitals, gender, plurals, superlative, past_tense_F,
                 numbers (NEW — previously misclassified as TYPE_ORDINAL)
  TYPE_ADJACENT: antonyms, past_tense_B, past_tense_D
  IDENTITY:      no_change_verbs

EVALUATION:
  1. Classification accuracy (predicted vs expected archetype)
  2. Retrieval accuracy per domain
  3. Overall accuracy vs Day 208 baseline (0.870)
  4. Breakdown showing which domains improved

ANTONYM ROUTING: tested in two modes:
  a. unsupervised: fallback to proximity (TYPE_ADJACENT)
  b. supervised:   use pre-computed attribute axis vectors
                   (requires knowing attribute label)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day212_final_pipeline.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Domain definitions ────────────────────────────────────────────────
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
        "train": [("go","went"),("have","had"),("do","did"),
                  ("take","took"),("give","gave"),("make","made")],
        "test":  [("come","came"),("get","got"),("stand","stood"),
                  ("leave","left"),("bring","brought"),("buy","bought")],
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
    "antonyms_sup_speed": {
        "expected": "TYPE_ANTONYM",
        "attribute": "speed",
        "train": [("fast","slow"),("quick","lazy"),("rapid","sluggish")],
        "test":  [("swift","sluggish")],
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
    "no_change_verbs": {
        "expected": "IDENTITY",
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost")],
    },
}

# Per-attribute antonym axes (from Day 210)
ANTONYM_AXES_DEF = {
    "temperature": [("hot","cold"),("warm","cool"),("burning","freezing")],
    "size":        [("big","small"),("large","tiny"),("huge","little"),
                    ("tall","short"),("wide","narrow"),("thick","thin")],
    "speed":       [("fast","slow"),("quick","lazy"),("rapid","sluggish")],
    "volume":      [("loud","quiet"),("noisy","silent"),("sharp","soft")],
    "age":         [("old","young"),("ancient","modern"),("mature","new")],
    "brightness":  [("light","dark"),("bright","dim"),("clear","murky")],
    "sharpness":   [("sharp","dull"),("keen","blunt"),("acute","obtuse")],
    "texture":     [("hard","soft"),("rough","smooth"),("rigid","flexible")],
    "wealth":      [("rich","poor"),("wealthy","broke"),("lavish","sparse")],
    "emotion":     [("happy","sad"),("joyful","miserable"),("glad","angry")],
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
            if (tid1(a) or tid1_bare(a)) and (tid1(b) or tid1_bare(b))]

def dir_consistency(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return 0.0
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in p
             if get_emb(a) is not None and get_emb(b) is not None]
    if len(diffs) < 2: return 0.0
    pw = [cosine(diffs[i], diffs[j])
          for i in range(len(diffs)) for j in range(i+1, len(diffs))]
    return float(np.mean(pw))

# Build extended retrieval vocabulary
RET_WORDS = [
    "Paris","Berlin","Rome","Madrid","Tokyo","Beijing","Moscow","Athens",
    "Cairo","Delhi","Brasilia","London","Buenos","Vienna","Warsaw","Stockholm",
    "queen","woman","girl","princess","actress","heroine","mother","sister",
    "daughter","wife","aunt","waitress","goddess","bride","mare",
    "king","man","boy","prince","actor","hero","father","brother",
    "son","husband","uncle","waiter","god","groom","stallion",
    "cats","dogs","houses","trees","books","cars","birds","ships","hands",
    "doors","lamps","walls","eyes","roads","cups","beds","keys","boxes",
    "tables","chairs","windows","rooms","pages","words","names","times",
    "biggest","fastest","longest","smartest","brightest","cleanest",
    "hardest","darkest","softest","warmest","slowest","smallest","tallest",
    "coldest","oldest","newest","richest","quietest","safest","shortest",
    "went","had","did","took","gave","made","came","got","stood","left",
    "brought","bought","said","knew","found","thought","ran","ate","saw",
    "grew","threw","blew","flew","drew","sent","spent","lent","bent",
    "built","found",
    "1","2","3","4","5","6","7","8","9",
    "cold","small","slow","soft","dark","young","quiet","dull","poor",
    "thin","narrow","shallow","wet","weak","rough","low","short","late",
    "hot","big","fast","hard","light","old","loud","sharp","rich",
    "thick","wide","deep","dry","strong","smooth","high","tall","early",
    "cat","dog","house","tree","book","car","bird","ship","hand","door",
    "lamp","wall","eye","road","cup","bed","key","box","table","chair",
    "big","fast","long","smart","bright","clean","hard","dark","soft",
    "warm","slow","small","tall","cold","old","new","rich","poor","safe",
    "France","Germany","Italy","Spain","Japan","China","Russia","Greece",
    "Brazil","Egypt","India","Mexico","England","Korea","Poland","Turkey",
    "go","have","do","take","give","make","come","get","stand","leave",
    "bring","buy","say","know","find","think","run","eat","see","write",
    "throw","blow","fly","draw","send","spend","lend","bend","build",
    "cut","put","hit","let","set","shut","burst","cost",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    # antonym vocab
    "cool","warm","burning","freezing","tiny","huge","little","tall",
    "quick","lazy","rapid","sluggish","noisy","silent","ancient","modern",
    "mature","bright","dim","clear","murky","keen","blunt","acute","obtuse",
    "smooth","rigid","flexible","wealthy","broke","lavish","sparse",
    "happy","sad","joyful","miserable","glad","angry","swift",
]

ret_vocab = {}
for w in RET_WORDS:
    t = tid1(w) or tid1_bare(w)
    if t is not None and w not in ret_vocab:
        ret_vocab[w] = W_E[t].astype(np.float64)
print(f"Retrieval vocabulary: {len(ret_vocab)} single-token words\n")

# Pre-compute antonym attribute axes
antonym_axes = {}
for attr, pairs in ANTONYM_AXES_DEF.items():
    p = ok_pairs(pairs)
    if len(p) < 2: continue
    diffs = [normed(get_emb(a) - get_emb(b)) for a,b in p
             if get_emb(a) is not None and get_emb(b) is not None]
    if diffs:
        antonym_axes[attr] = normed(np.mean(diffs, axis=0))

def retrieve_bc(src, train_pairs, vocab=ret_vocab):
    p = ok_pairs(train_pairs)
    se = get_emb(src)
    if se is None or not p: return None
    diffs = [normed(get_emb(b) - get_emb(a)) for a,b in p
             if get_emb(a) is not None and get_emb(b) is not None]
    if not diffs: return None
    mdir  = normed(np.mean(diffs, axis=0))
    query = se + mdir
    sims  = {w: cosine(query, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

def retrieve_adjacent(src, vocab=ret_vocab):
    se = get_emb(src)
    if se is None: return None
    sims = {w: cosine(se, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

def retrieve_antonym_axis(src, attribute, vocab=ret_vocab):
    se = get_emb(src)
    if se is None or attribute not in antonym_axes: return retrieve_adjacent(src, vocab)
    axis = antonym_axes[attribute]
    src_proj = float(np.dot(normed(se), axis))
    target_dir = axis if src_proj < 0 else -axis
    query = normed(se + target_dir)
    sims  = {w: cosine(query, e) for w,e in vocab.items() if w != src}
    return max(sims, key=lambda w: sims[w])

# ── V3 Pipeline Classifier ────────────────────────────────────────────
def classify_v3(domain_name, train_pairs, attribute=None):
    p = ok_pairs(train_pairs)
    # STEP 0: IDENTITY
    if any(a == b for a,b in p):
        return "IDENTITY"
    # STEP 1: TYPE_BC (covers numbers dc=0.850 + all directional domains)
    if len(p) >= 2:
        dc = dir_consistency(p)
        if dc > 0.15:
            return "TYPE_BC"
    # STEP 2: TYPE_ANTONYM (supervised — only if attribute label provided)
    if attribute is not None and attribute in antonym_axes:
        return "TYPE_ANTONYM"
    # STEP 3: TYPE_ADJACENT
    return "TYPE_ADJACENT"

# ── Evaluation ────────────────────────────────────────────────────────
print("=" * 74)
print(f"{'Domain':<24}  {'Expected':<14}  {'Predicted':<14}  "
      f"{'dc':>6}  {'Match':>6}")
print("=" * 74)

all_results = {}
for domain_name, cfg in DOMAINS.items():
    train = cfg["train"]; expected = cfg["expected"]
    attribute = cfg.get("attribute", None)
    p  = ok_pairs(train)
    dc = dir_consistency(p) if len(p) >= 2 else 0.0
    pred = classify_v3(domain_name, train, attribute)
    match = (pred == expected)
    mark  = "" if match else " ✗"
    print(f"  {domain_name:<24}  {expected:<14}  {pred:<14}  "
          f"{dc:>6.3f}  {'YES' if match else 'NO':>6}{mark}")
    all_results[domain_name] = {
        "expected": expected, "predicted": pred,
        "dir_consistency": dc, "correct_classification": match,
    }

print()
print("=" * 74)
print("RETRIEVAL ACCURACY")
print("=" * 74)
total_c = 0; total_n = 0
for domain_name, cfg in DOMAINS.items():
    train = cfg["train"]; test = cfg["test"]
    attribute = cfg.get("attribute", None)
    pred  = all_results[domain_name]["predicted"]
    test_ok = ok_pairs(test) if test else ok_pairs(train)
    if not test_ok: continue

    correct = 0
    for src, tgt in test_ok:
        if pred == "IDENTITY":
            p_tgt = src
        elif pred == "TYPE_BC":
            p_tgt = retrieve_bc(src, train)
        elif pred == "TYPE_ANTONYM":
            p_tgt = retrieve_antonym_axis(src, attribute)
        else:
            p_tgt = retrieve_adjacent(src)
        if p_tgt == tgt: correct += 1

    acc = correct / len(test_ok)
    total_c += correct; total_n += len(test_ok)

    expected = cfg["expected"]
    # Oracle accuracy
    oracle_c = 0
    for src, tgt in test_ok:
        if expected == "IDENTITY":
            op = src
        elif expected in ("TYPE_BC",):
            op = retrieve_bc(src, train)
        elif expected == "TYPE_ANTONYM":
            op = retrieve_antonym_axis(src, attribute)
        else:
            op = retrieve_adjacent(src)
        if op == tgt: oracle_c += 1
    oracle_acc = oracle_c / len(test_ok)
    all_results[domain_name]["retrieval_accuracy"] = acc
    all_results[domain_name]["oracle_accuracy"]    = oracle_acc
    print(f"  {domain_name:<24}  n={len(test_ok):>2}  "
          f"pred={pred:<14}  acc={acc:.3f}  oracle={oracle_acc:.3f}")

overall_acc = total_c / total_n if total_n else 0
print(f"\n  OVERALL: {total_c}/{total_n}  acc={overall_acc:.3f}")

print()
print("=" * 74)
print("COMPARISON ACROSS PIPELINE VERSIONS")
print("=" * 74)
print("  Day 198 (v1): 0.779  (7 domains, TYPE_BC + TYPE_ADJACENT only)")
print("  Day 208 (v2): 0.870  (10 domains, + IDENTITY; numbers missed)")
print(f"  Day 212 (v3): {overall_acc:.3f}  (12 domains, + numbers + antonym axes)")

cls_correct = sum(1 for d in all_results.values() if d["correct_classification"])
print(f"\n  Classification: {cls_correct}/{len(all_results)} correct")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 212 complete.")
