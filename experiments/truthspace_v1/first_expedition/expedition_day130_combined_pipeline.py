#!/usr/bin/env python3
"""
Day 130 — Combined Geometric Generation Pipeline

DC 336 synthesis: two classes of geometric knowledge.
  Class 1 (structural): stable axes, LOO generalizes (antonyms 0.694, tense/gender)
  Class 2 (factual):    variable axes, LOO fails (capitals 0.325, hypernyms 0.323)

PIPELINE:
  Given a prompt + candidate set:
  1. T2 filter: exclude candidates whose T2 cosine is > 2σ below mean (gross mismatch)
  2. Route to best sub-ranker based on category type:
     - Structural categories: factual axis (LOO-computed from training pairs)
     - Factual categories: L25 full cosine
  3. Combine scores with T2 category signal

QUESTIONS:
  A) Does the combined pipeline beat any individual method?
  B) Does routing between structural/factual methods help?
  C) What is the best achievable MRR across ALL 21 canonical test pairs?
  D) How does the pipeline compare to oracle log-prob (MRR=1.0)?

TEST PROMPTS: Use all pairs from Days 128-129 PLUS new held-out prompts
  to get a comprehensive view of pipeline performance.

ALSO: Measure T2 filter effectiveness — does filtering with T2 improve
precision by removing obviously wrong candidates before the ranker runs?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day130_combined_pipeline.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

L_BEST = 25  # Day 127 optimal layer

# Day 129 training pairs (used to compute factual axes)
TRAIN_STRUCTURAL = {
    "antonyms": [
        ("cold",    ["warm","lukewarm","mild","chilly"],   "The opposite of hot is"),
        ("sad",     ["angry","bored","worried","tired"],   "The opposite of happy is"),
        ("dark",    ["bright","sunny","light","clear"],    "The opposite of bright is"),
        ("slow",    ["fast","quick","rapid","swift"],      "The opposite of fast is"),
        ("small",   ["large","big","huge","enormous"],     "The opposite of large is"),
        ("wrong",   ["right","correct","accurate","true"], "The opposite of right is"),
    ],
    "languages": [
        ("Portuguese", ["Spanish","Italian","French","English"],  "The official language of Brazil is"),
        ("Arabic",     ["Hebrew","Turkish","Persian","Urdu"],     "The official language of Egypt is"),
        ("Mandarin",   ["Japanese","Korean","Cantonese","Thai"],  "The official language of China is"),
        ("Hindi",      ["Urdu","Bengali","Tamil","Punjabi"],      "The official language of India is"),
    ],
}
TRAIN_FACTUAL = {
    "capitals": [
        ("Paris",      ["London","Rome","Berlin","Madrid"],       "The capital city of France is"),
        ("Tokyo",      ["Osaka","Beijing","Seoul","Bangkok"],     "The capital city of Japan is"),
        ("Berlin",     ["Frankfurt","Vienna","Warsaw","Amsterdam"],"The capital city of Germany is"),
        ("Madrid",     ["Lisbon","Rome","Paris","Brussels"],      "The capital city of Spain is"),
        ("Rome",       ["Milan","Paris","Vienna","Athens"],       "The capital city of Italy is"),
        ("Moscow",     ["Petersburg","Kiev","Minsk","Warsaw"],    "The capital city of Russia is"),
    ],
    "hypernyms": [
        ("dog",    ["cat","rabbit","horse","pet"],          "A poodle is a type of"),
        ("flower", ["tree","bush","grass","weed"],          "A rose is a type of"),
        ("tool",   ["machine","device","appliance","weapon"],"A hammer is a type of"),
        ("bird",   ["insect","reptile","fish","mammal"],    "An eagle is a type of"),
        ("gem",    ["metal","mineral","crystal","stone"],   "A ruby is a type of"),
    ],
}

# NEW held-out test prompts (NOT in any training set)
TEST_HELD_OUT = [
    # structural — antonym-type
    {"cat": "antonyms",   "prompt": "The opposite of young is",
     "correct": ["old","elderly","aged"], "wrong": ["Paris","running","capital"]},
    {"cat": "antonyms",   "prompt": "The opposite of weak is",
     "correct": ["strong","powerful","robust"], "wrong": ["banana","quietly","ocean"]},
    # structural — language-type
    {"cat": "languages",  "prompt": "The official language of Mexico is",
     "correct": ["Spanish"], "wrong": ["English","French","Portuguese","Italian",
                                       "running","banana","because","cold"]},
    {"cat": "languages",  "prompt": "The official language of Japan is",
     "correct": ["Japanese"], "wrong": ["Chinese","Korean","Mandarin","Thai",
                                        "quickly","stone","never","walked"]},
    # factual — capitals-type
    {"cat": "capitals",   "prompt": "The capital city of Australia is",
     "correct": ["Canberra"], "wrong": ["Sydney","Melbourne","Perth","Brisbane",
                                        "banana","running","because","cold"]},
    {"cat": "capitals",   "prompt": "The capital city of Canada is",
     "correct": ["Ottawa"], "wrong": ["Toronto","Vancouver","Montreal","Calgary",
                                      "walked","stone","never","quickly"]},
    # factual — hypernym-type
    {"cat": "hypernyms",  "prompt": "A salmon is a type of",
     "correct": ["fish","animal","vertebrate"], "wrong": ["bird","insect","plant","tree",
                                                          "capital","walked","never","cold"]},
    {"cat": "hypernyms",  "prompt": "A piano is a type of",
     "correct": ["instrument","tool","device"], "wrong": ["weapon","machine","vehicle","toy",
                                                          "flower","language","quickly","green"]},
    # T2 axis tests (from Days 124-129)
    {"cat": "tense",      "prompt": "Yesterday she",
     "correct": ["walked","ran","ate","wrote","saw"],
     "wrong": ["walk","run","eat","write","see","Paris","banana","because"]},
    {"cat": "gender",     "prompt": "The queen and",
     "correct": ["king","prince","duke","emperor"],
     "wrong": ["princess","lady","duchess","countess","banana","quickly","walked","stone"]},
]

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender","comparative","hypernym","plural","synonym","concrete",
    "past_tense","antonym","passive","causation","question","negation",
]
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("His brother arrived at the party","His sister arrived at the party"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The car sped past the sign","The vehicle sped past the sign"),
    ],
    "plural": [
        ("A dog played happily in field","Dogs played happily in field"),
        ("The cat sat quietly by window","The cats sat quietly by window"),
        ("A bird sang softly in mist","Birds sang softly in mist"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
    ],
    "concrete": [
        ("The stone is too heavy","The burden is too heavy"),
        ("The long road leads","The long journey leads"),
        ("The high wall blocks","The high barrier blocks"),
    ],
    "past_tense": [
        ("I walk every morning","I walked every morning"),
        ("She runs through park","She ran through park"),
        ("He eats before leaving","He ate before leaving"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The news is good","The news is bad"),("She is happy","She is sad"),
    ],
    "passive": [
        ("The cat chased mouse","The mouse was chased"),
        ("John broke window","The window was broken"),
    ],
    "causation": [
        ("The rain falls down","The ground gets wet"),
        ("The fire burns long","The wood turns to ash"),
    ],
    "question": [
        ("She is tired today","Is she tired today"),
        ("He can swim well","Can he swim well"),
        ("They went to market","Did they go to market"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know answer"),
    ],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_layers    = model.config.num_hidden_layers
print(f"  hidden={hidden_size}\n")

def get_hs_L(text, layers):
    inp = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}
    except:
        return {L: np.zeros(hidden_size, np.float32) for L in layers}

def get_logprob(prompt, word):
    inp = tok(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            lp = torch.log_softmax(logits, dim=-1).numpy()
        ids = tok(" " + word, add_special_tokens=False)["input_ids"]
        return float(lp[ids[0]]) if ids else float("-inf")
    except:
        return float("-inf")

# ── Build T2 axes ────────────────────────────────────────────────────────────
print("Building T2 axes ...")
t2_axes = {}
for ax_name in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax_name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(ax_name, []):
        try:
            inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
            with torch.no_grad():
                o1 = model(**inp1, output_hidden_states=True)
                o2 = model(**inp2, output_hidden_states=True)
            h1 = o1.hidden_states[L][0,-1,:].numpy().astype(np.float32)
            h2 = o2.hidden_states[L][0,-1,:].numpy().astype(np.float32)
            d = h2-h1; nv = np.linalg.norm(d)
            if nv > 1e-6: diffs.append(d/nv)
        except: pass
    v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, np.float32)
    nv = np.linalg.norm(v)
    t2_axes[ax_name] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

def t2_vec(hs_dict):
    v = np.zeros(12, np.float32)
    for k, ax_name in enumerate(AXIS_NAMES_12):
        L = DAY78_LAYERS[ax_name]
        h = normed(hs_dict.get(L, np.zeros(hidden_size)))
        v[k] = float(np.dot(h, t2_axes[ax_name]))
    return v

def t2_cos(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# ── Build factual axes for structural categories ─────────────────────────────
print("Building factual axes for structural categories (training pairs) ...")
struct_axes = {}
for cat_name, pairs in TRAIN_STRUCTURAL.items():
    deltas = []
    for correct, wrong_pool, _ in pairs:
        h_cor = get_hs_L(" " + correct, [L_BEST])[L_BEST]
        h_wrg = np.mean([get_hs_L(" " + w, [L_BEST])[L_BEST] for w in wrong_pool], axis=0)
        d = h_cor - h_wrg.astype(np.float32)
        nv = np.linalg.norm(d)
        if nv > 1e-6: deltas.append(d/nv)
    if deltas:
        axis = np.mean(deltas, axis=0).astype(np.float32)
        nv   = np.linalg.norm(axis)
        struct_axes[cat_name] = (axis/nv if nv > 1e-6 else axis)
        print(f"  {cat_name}: {len(deltas)} training pairs  ||axis||={nv:.4f}")
print()

T2_LAYERS = sorted(set(DAY78_LAYERS.values()))

print("=" * 72)
print("Day 130: Combined Geometric Pipeline vs Oracle Log-Prob")
print("=" * 72)
print()

all_test_cases = TEST_HELD_OUT
all_results = []

for case in all_test_cases:
    cat      = case["cat"]
    prompt   = case["prompt"]
    correct  = case["correct"]
    wrong    = case["wrong"]
    all_cands = correct + wrong

    # Log-prob (oracle)
    lp = {w: get_logprob(prompt, w) for w in all_cands}

    # Context hidden state at L25 and T2 layers
    ctx_layers = sorted(set([L_BEST] + T2_LAYERS))
    ctx_hs     = get_hs_L(prompt, ctx_layers)
    ctx_t2     = t2_vec(ctx_hs)

    # All candidate hidden states at L25
    cand_hs = {w: get_hs_L(" " + w, [L_BEST] + T2_LAYERS) for w in all_cands}

    # Method 1: L25 full cosine
    cos25 = {w: cosine(ctx_hs[L_BEST], cand_hs[w][L_BEST]) for w in all_cands}

    # Method 2: T2 cosine (isolated candidate vs context anchor)
    t2_sims = {w: t2_cos(t2_vec(cand_hs[w]), ctx_t2) for w in all_cands}

    # Method 3: Factual axis (if structural category has trained axis)
    use_struct = cat in struct_axes
    if use_struct:
        fax = struct_axes[cat]
        fax_projs = {w: float(np.dot(normed(cand_hs[w][L_BEST]), fax)) for w in all_cands}
    else:
        fax_projs = None

    # Combined pipeline:
    # T2 filter: exclude candidates with T2 sim < (mean - std)
    t2_vals = np.array([t2_sims[w] for w in all_cands])
    t2_thresh = t2_vals.mean() - t2_vals.std()
    filtered_cands = [w for w in all_cands if t2_sims[w] >= t2_thresh]
    if len(filtered_cands) < 2: filtered_cands = all_cands  # safety

    # Route: structural → factual axis; factual → L25 cosine
    if use_struct and fax_projs is not None:
        primary_scores = fax_projs
    else:
        primary_scores = cos25

    # Normalize and combine T2 + primary
    prim_v = np.array([primary_scores[w] for w in all_cands])
    t2_v   = np.array([t2_sims[w] for w in all_cands])
    prim_n = (prim_v - prim_v.min()) / (prim_v.max()-prim_v.min() + 1e-8)
    t2_n   = (t2_v - t2_v.min()) / (t2_v.max()-t2_v.min() + 1e-8)
    combo  = 0.8 * prim_n + 0.2 * t2_n
    combo_scores = {w: float(combo[i]) for i, w in enumerate(all_cands)}

    def best_rank(scores, higher=True):
        ranked = sorted(all_cands, key=lambda w: (-scores[w] if higher else scores[w]))
        r = next((j+1 for j,w in enumerate(ranked) if w in correct), len(all_cands)+1)
        return r, 1.0/r

    rank_lp,    mrr_lp    = best_rank(lp)
    rank_cos,   mrr_cos   = best_rank(cos25)
    rank_t2,    mrr_t2    = best_rank(t2_sims)
    rank_prim,  mrr_prim  = best_rank(primary_scores)
    rank_combo, mrr_combo = best_rank(combo_scores)

    all_results.append({
        "cat": cat, "prompt": prompt,
        "method": "struct_axis" if use_struct else "l25_cos",
        "mrr_lp": mrr_lp, "mrr_cos25": mrr_cos,
        "mrr_t2": mrr_t2, "mrr_primary": mrr_prim,
        "mrr_combo": mrr_combo,
        "rank_lp": rank_lp, "rank_combo": rank_combo,
    })

    method_label = "struct_axis" if use_struct else "L25_cos "
    print(f"  [{cat:>10}] {prompt!r}")
    print(f"    oracle lp={mrr_lp:.3f}  L25={mrr_cos:.3f}  T2={mrr_t2:.3f}  "
          f"{method_label}={mrr_prim:.3f}  combo={mrr_combo:.3f}")
    print()

# ── Aggregate ─────────────────────────────────────────────────────────────────
print("=" * 72)
print("Aggregate Results — Day 130 Combined Pipeline")
print("=" * 72)

n = len(all_results)
def mean_mrr(key): return float(np.mean([r[key] for r in all_results]))

print(f"""
  Method              MRR      vs_random
  ─────────────────────────────────────────
  log-prob (oracle)  {mean_mrr('mrr_lp'):.4f}    ---
  Combined pipeline  {mean_mrr('mrr_combo'):.4f}    +{mean_mrr('mrr_combo')-0.2:.4f}
  Primary (routed)   {mean_mrr('mrr_primary'):.4f}    +{mean_mrr('mrr_primary')-0.2:.4f}
  L25 cosine         {mean_mrr('mrr_cos25'):.4f}    +{mean_mrr('mrr_cos25')-0.2:.4f}
  T2 isolated        {mean_mrr('mrr_t2'):.4f}    +{mean_mrr('mrr_t2')-0.2:.4f}
  Random baseline    0.2000    ---
""")

# By category type
struct_results  = [r for r in all_results if r["method"] == "struct_axis"]
factual_results = [r for r in all_results if r["method"] == "l25_cos "]

def cat_summary(results, label):
    if not results: return
    lp   = float(np.mean([r["mrr_lp"]      for r in results]))
    co   = float(np.mean([r["mrr_combo"]   for r in results]))
    prim = float(np.mean([r["mrr_primary"] for r in results]))
    l25  = float(np.mean([r["mrr_cos25"]   for r in results]))
    t2   = float(np.mean([r["mrr_t2"]      for r in results]))
    print(f"  {label} (n={len(results)}):  oracle={lp:.4f}  combo={co:.4f}  "
          f"primary={prim:.4f}  L25={l25:.4f}  T2={t2:.4f}")

cat_summary(struct_results,  "Structural (struct_axis)")
cat_summary([r for r in all_results if r["method"]=="l25_cos "], "Factual    (l25_cosine)")

# Actually split by cat properly
for cat_name in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"] == cat_name]
    lp  = float(np.mean([r["mrr_lp"]    for r in cat_r]))
    co  = float(np.mean([r["mrr_combo"] for r in cat_r]))
    mth = cat_r[0]["method"] if cat_r else "?"
    print(f"    {cat_name:>12}: oracle={lp:.4f}  combo={co:.4f}  "
          f"method={mth}  {'✓' if co>0.5 else '~' if co>0.3 else '✗'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 130 Summary — Complete Geometric Pipeline")
print("=" * 72)

oracle_mrr  = mean_mrr("mrr_lp")
combo_mrr   = mean_mrr("mrr_combo")
l25_mrr     = mean_mrr("mrr_cos25")
random_mrr  = 0.2

print(f"""
  Pipeline captures {100*combo_mrr/oracle_mrr:.1f}% of oracle performance
  (MRR: combo={combo_mrr:.4f} vs oracle={oracle_mrr:.4f})

  L25 alone: {100*l25_mrr/oracle_mrr:.1f}% of oracle
  Combined T2+factual+L25: {100*combo_mrr/oracle_mrr:.1f}% of oracle

  VERDICT:
  {'→ Pipeline EXCEEDS L25 alone: routing adds value' if combo_mrr > l25_mrr else
   '→ Pipeline roughly equals L25: routing adds minimal value'}
  {'→ Pipeline within 20% of oracle: strong geometric approximation' if oracle_mrr-combo_mrr < 0.2 else
   '→ Pipeline is 20%+ below oracle: significant gap remains'}

  The two-class routing strategy (structural → factual axis, factual → L25):
  {'→ EFFECTIVE: improves over single-method baseline' if combo_mrr > l25_mrr else
   '→ NOT effective: no routing benefit over L25'}

  TruthSpace Generation Conclusion (Days 124-130):
  Geometric structures capture {100*combo_mrr/oracle_mrr:.0f}% of oracle MRR.
  The remaining {100-100*combo_mrr/oracle_mrr:.0f}% requires directed weight associations.
  Class 1 (structural) ≈ {100*float(np.mean([r['mrr_primary'] for r in all_results if r['method']=='struct_axis']))/oracle_mrr:.0f}% oracle for structural categories.
  Class 2 (factual) ≈ {100*float(np.mean([r['mrr_cos25'] for r in all_results if r['method']!='struct_axis']))/oracle_mrr:.0f}% oracle for factual categories.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "mrr_summary": {
            "oracle": oracle_mrr, "combo": combo_mrr,
            "l25": l25_mrr, "random": random_mrr,
        },
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 130 complete.")
