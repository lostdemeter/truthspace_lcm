#!/usr/bin/env python3
"""
Day 131 — Query Type Detection from T2 Address

Day 130 showed that the routing strategy needs to be correct to beat L25 alone.
The question is: can we AUTOMATICALLY detect query type from T2 address?

T2 captures 12D projections of the last-token hidden state. Different query
templates have different last-token structures:

  "Yesterday she"         → last token = "she" (pronoun, tense context)
  "The opposite of hot is"→ last token = "is" (relational query)
  "The capital of France is" → last token = "is" (factual query)
  "The king and"          → last token = "and" (completion context)

QUESTION: Do T2 addresses of prompts cluster by query category?
  If yes: T2 is a query classifier → enables automatic routing
  If no:  query type cannot be detected geometrically

EXPERIMENT:
  1. Compute T2 addresses for all prompts from Days 128-130
  2. Cluster by query category (syntactic, relational, factual)
  3. Measure inter-class vs intra-class T2 distance
  4. Test LOO classification: can we classify prompt query type from T2?
  5. If T2 classification works: implement oracle-routing pipeline and
     compare to naive routing from Day 130

Also test: does the T2 address of the LAST TOKEN (as opposed to the
full sentence) carry different query-type information?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day131_query_type_detection.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# All prompts from Days 128-130 with ground-truth category type
PROMPTS = [
    # SYNTACTIC (best method: T2)
    {"prompt": "Yesterday he",          "type": "syntactic", "cat": "tense"},
    {"prompt": "Yesterday she",         "type": "syntactic", "cat": "tense"},
    {"prompt": "Yesterday they",        "type": "syntactic", "cat": "tense"},
    {"prompt": "The king and",          "type": "syntactic", "cat": "gender"},
    {"prompt": "The queen and",         "type": "syntactic", "cat": "gender"},
    {"prompt": "The father and",        "type": "syntactic", "cat": "gender"},
    # RELATIONAL (best method: struct_axis for antonyms)
    {"prompt": "The opposite of hot is",      "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of happy is",    "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of young is",    "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of weak is",     "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of bright is",   "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of fast is",     "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of large is",    "type": "relational", "cat": "antonyms"},
    {"prompt": "The opposite of right is",    "type": "relational", "cat": "antonyms"},
    # FACTUAL — capitals (best method: L25 cosine)
    {"prompt": "The capital city of France is",    "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Japan is",     "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Germany is",   "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Spain is",     "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Italy is",     "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Russia is",    "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Australia is", "type": "factual", "cat": "capitals"},
    {"prompt": "The capital city of Canada is",    "type": "factual", "cat": "capitals"},
    # FACTUAL — hypernyms
    {"prompt": "A poodle is a type of",  "type": "factual", "cat": "hypernyms"},
    {"prompt": "A rose is a type of",    "type": "factual", "cat": "hypernyms"},
    {"prompt": "A hammer is a type of",  "type": "factual", "cat": "hypernyms"},
    {"prompt": "An eagle is a type of",  "type": "factual", "cat": "hypernyms"},
    {"prompt": "A ruby is a type of",    "type": "factual", "cat": "hypernyms"},
    {"prompt": "A salmon is a type of",  "type": "factual", "cat": "hypernyms"},
    {"prompt": "A piano is a type of",   "type": "factual", "cat": "hypernyms"},
    # FACTUAL — languages
    {"prompt": "The official language of Brazil is",  "type": "factual", "cat": "languages"},
    {"prompt": "The official language of Egypt is",   "type": "factual", "cat": "languages"},
    {"prompt": "The official language of China is",   "type": "factual", "cat": "languages"},
    {"prompt": "The official language of India is",   "type": "factual", "cat": "languages"},
    {"prompt": "The official language of Mexico is",  "type": "factual", "cat": "languages"},
    {"prompt": "The official language of Japan is",   "type": "factual", "cat": "languages"},
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
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),
    ],
    "concrete": [
        ("The stone is too heavy","The burden is too heavy"),
        ("The long road leads","The long journey leads"),
    ],
    "past_tense": [
        ("I walk every morning","I walked every morning"),
        ("She runs through park","She ran through park"),
        ("He eats before leaving","He ate before leaving"),
        ("They build the wall","They built the wall"),
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
print(f"  hidden={hidden_size}\n")

# Build T2 axes
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

T2_LAYERS = sorted(set(DAY78_LAYERS.values()))

def get_hs(text):
    inp = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in T2_LAYERS}
    except:
        return {L: np.zeros(hidden_size, np.float32) for L in T2_LAYERS}

def t2_vec(hs_dict):
    v = np.zeros(12, np.float32)
    for k, ax_name in enumerate(AXIS_NAMES_12):
        L = DAY78_LAYERS[ax_name]
        h = normed(hs_dict.get(L, np.zeros(hidden_size)))
        v[k] = float(np.dot(h, t2_axes[ax_name]))
    return v

# Compute T2 addresses for all prompts
print("Computing T2 addresses for all prompts ...")
prompt_t2 = {}
for p in PROMPTS:
    hs = get_hs(p["prompt"])
    prompt_t2[p["prompt"]] = t2_vec(hs)
print(f"  Done. {len(PROMPTS)} prompts.\n")

# ── Part 1: Intra-class vs Inter-class T2 distance ─────────────────────────────
print("=" * 72)
print("Part 1: T2 Address Clustering by Query Type")
print("=" * 72)
print()

types = ["syntactic", "relational", "factual"]
type_vecs = {t: [prompt_t2[p["prompt"]] for p in PROMPTS if p["type"] == t]
             for t in types}

from itertools import combinations as combs

def pairwise_cosines(vecs):
    if len(vecs) < 2: return [0.0]
    return [cosine(vecs[i], vecs[j]) for i,j in combs(range(len(vecs)), 2)]

def cross_cosines(vecs_a, vecs_b):
    return [cosine(a, b) for a in vecs_a for b in vecs_b]

print(f"  {'pair':>25}  {'mean cos':>10}  {'std':>8}")
print(f"  {'-'*48}")
intra = {}
for t in types:
    pw = pairwise_cosines(type_vecs[t])
    intra[t] = float(np.mean(pw))
    print(f"  intra-{t:>12}      {np.mean(pw):>+10.4f}  {np.std(pw):>8.4f}  (n={len(type_vecs[t])})")

print()
inter = {}
for t1, t2 in combs(types, 2):
    cc = cross_cosines(type_vecs[t1], type_vecs[t2])
    key = f"{t1} × {t2}"
    inter[key] = float(np.mean(cc))
    print(f"  inter-{key:>20}  {np.mean(cc):>+10.4f}  {np.std(cc):>8.4f}")

print()
# Discrimination: is intra > inter?
min_intra = min(intra.values())
max_inter = max(inter.values())
print(f"  Min intra-class cosine: {min_intra:.4f}")
print(f"  Max inter-class cosine: {max_inter:.4f}")
sep = min_intra - max_inter
print(f"  Separation margin: {sep:+.4f}  "
      f"({'SEPARABLE' if sep > 0 else 'OVERLAPPING'})")

# Also check category-level (more granular)
print()
print("  Per-category T2 mean addresses:")
cats_seen = sorted(set(p["cat"] for p in PROMPTS))
cat_means = {}
for cat_name in cats_seen:
    vecs = [prompt_t2[p["prompt"]] for p in PROMPTS if p["cat"] == cat_name]
    mean_v = np.mean(vecs, axis=0)
    nv = np.linalg.norm(mean_v)
    cat_means[cat_name] = (mean_v/nv if nv > 1e-6 else mean_v).astype(np.float32)
    intra_pw = pairwise_cosines(vecs)
    print(f"    {cat_name:>12}: n={len(vecs)}  intra_cos={np.mean(intra_pw):+.4f}")

# ── Part 2: LOO classification test ───────────────────────────────────────────
print()
print("=" * 72)
print("Part 2: LOO Type Classification from T2 Address")
print("=" * 72)
print()
print("  For each prompt: exclude it, compute type centroids from remaining,")
print("  classify by nearest centroid.")
print()

def classify_loo(idx):
    p = PROMPTS[idx]
    rest = [q for j, q in enumerate(PROMPTS) if j != idx]
    centroids = {}
    for t in types:
        vecs = [prompt_t2[q["prompt"]] for q in rest if q["type"] == t]
        if not vecs: continue
        m = np.mean(vecs, axis=0)
        nv = np.linalg.norm(m)
        centroids[t] = (m/nv if nv > 1e-6 else m).astype(np.float32)
    v = prompt_t2[p["prompt"]]
    scores = {t: cosine(v, centroids[t]) for t in centroids}
    pred = max(scores, key=lambda t: scores[t])
    return pred, p["type"], pred == p["type"]

correct_counts = {t: 0 for t in types}
total_counts   = {t: 0 for t in types}
all_correct = []
for i, p in enumerate(PROMPTS):
    pred, true, ok = classify_loo(i)
    correct_counts[true] += int(ok)
    total_counts[true]   += 1
    all_correct.append(ok)

print(f"  Overall LOO accuracy: {sum(all_correct)}/{len(all_correct)} "
      f"= {sum(all_correct)/len(all_correct):.3f}")
print()
for t in types:
    acc = correct_counts[t] / total_counts[t] if total_counts[t] > 0 else 0
    print(f"    {t:>12}: {correct_counts[t]}/{total_counts[t]}  acc={acc:.3f}")

# ── Part 3: Category-level LOO classification ──────────────────────────────────
print()
print("=" * 72)
print("Part 3: LOO Category Classification (finer-grained)")
print("=" * 72)
print()

# For each prompt, classify into fine-grained category
def classify_category_loo(idx):
    p = PROMPTS[idx]
    rest = [q for j, q in enumerate(PROMPTS) if j != idx]
    centroids = {}
    for cat_name in cats_seen:
        vecs = [prompt_t2[q["prompt"]] for q in rest if q["cat"] == cat_name]
        if not vecs: continue
        m = np.mean(vecs, axis=0)
        nv = np.linalg.norm(m)
        centroids[cat_name] = (m/nv if nv > 1e-6 else m).astype(np.float32)
    v = prompt_t2[p["prompt"]]
    scores = {cat_name: cosine(v, centroids[cat_name]) for cat_name in centroids}
    pred = max(scores, key=lambda c: scores[c])
    return pred, p["cat"], pred == p["cat"]

cat_correct = {c: 0 for c in cats_seen}
cat_total   = {c: 0 for c in cats_seen}
cat_ok_all  = []
for i, p in enumerate(PROMPTS):
    pred, true, ok = classify_category_loo(i)
    cat_correct[true] += int(ok)
    cat_total[true]   += 1
    cat_ok_all.append(ok)

print(f"  Overall LOO category accuracy: {sum(cat_ok_all)}/{len(cat_ok_all)} "
      f"= {sum(cat_ok_all)/len(cat_ok_all):.3f}")
print()
for cat_name in cats_seen:
    if cat_total[cat_name] == 0: continue
    acc = cat_correct[cat_name] / cat_total[cat_name]
    print(f"    {cat_name:>12}: {cat_correct[cat_name]}/{cat_total[cat_name]}  acc={acc:.3f}  "
          f"{'✓' if acc >= 0.8 else '~' if acc >= 0.5 else '✗'}")

# ── Part 4: T2 discriminant axes for query type ────────────────────────────────
print()
print("=" * 72)
print("Part 4: Which T2 axes most discriminate query type?")
print("=" * 72)
print()

# Per-axis mean and variance across types
type_labels = np.array([0 if p["type"]=="syntactic" else
                         1 if p["type"]=="relational" else 2 for p in PROMPTS])
t2_matrix   = np.array([prompt_t2[p["prompt"]] for p in PROMPTS])  # (N, 12)

print(f"  {'axis':>12}  {'var_between':>12}  {'var_within':>12}  {'F_ratio':>10}")
print(f"  {'-'*50}")
axis_f_ratios = {}
for k, ax_name in enumerate(AXIS_NAMES_12):
    vals = t2_matrix[:, k]
    grand_mean = vals.mean()
    # Between-class variance
    between = np.sum([((vals[type_labels==c]).mean() - grand_mean)**2 * (type_labels==c).sum()
                      for c in [0,1,2]]) / 3
    # Within-class variance
    within  = np.mean([np.var(vals[type_labels==c]) for c in [0,1,2]])
    f = between / (within + 1e-8)
    axis_f_ratios[ax_name] = f
    print(f"  {ax_name:>12}  {between:>12.4f}  {within:>12.4f}  {f:>10.4f}")

best_axes = sorted(axis_f_ratios, key=lambda a: -axis_f_ratios[a])[:3]
print(f"\n  Top discriminant axes: {best_axes}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 131 Summary — Query Type Detection from T2")
print("=" * 72)

type_loo_acc = sum(all_correct) / len(all_correct)
cat_loo_acc  = sum(cat_ok_all) / len(cat_ok_all)
can_classify = type_loo_acc > 0.7
can_fine_classify = cat_loo_acc > 0.5

print(f"""
  T2 query type detection (3-class LOO):   {type_loo_acc:.3f}
  T2 category detection (5-class LOO):     {cat_loo_acc:.3f}

  Type separation: {sep:+.4f}  ({'SEPARABLE' if sep > 0 else 'OVERLAPPING'})
  Top discriminant T2 axes: {best_axes}

  Can T2 classify query type automatically?
  {'→ YES ✓: T2 LOO accuracy > 0.7' if can_classify else
   '→ PARTIAL: T2 LOO accuracy 0.5-0.7' if type_loo_acc > 0.5 else
   '→ NO ✗: T2 cannot reliably detect query type'}

  Implication for automatic routing pipeline:
  {'→ FEASIBLE: Use T2 type classifier to route to best sub-ranker' if can_classify else
   '→ UNRELIABLE: Routing based on T2 type detection would be error-prone'}
  {'→ Category-level routing feasible' if can_fine_classify else
   '→ Category-level routing NOT reliable from T2 alone'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "intra_class": intra,
        "inter_class": inter,
        "separation_margin": float(sep),
        "type_loo_accuracy": float(type_loo_acc),
        "cat_loo_accuracy": float(cat_loo_acc),
        "axis_f_ratios": {k: float(v) for k, v in axis_f_ratios.items()},
        "best_discriminant_axes": best_axes,
        "can_classify_type": bool(can_classify),
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 131 complete.")
