#!/usr/bin/env python3
"""
Day 162 — Domain Extension: Scientific Facts in W_E

Geography/antonyms/gender are well-represented in general web text.
But do scientific facts also live in W_E geometry?

TEST DOMAINS:
  1. Periodic table: element → symbol (hydrogen→H, oxygen→O...)
     PROBLEM: most symbols are 1-2 letter tokens; tokenization may differ
  2. Element → atomic number category (light/heavy/metal/nonmetal)
  3. Biology: animal → category (mammal/bird/fish/reptile/insect)
  4. Animal → diet (carnivore/herbivore/omnivore)
  5. Planet → property (rocky/gas/inner/outer)
  6. Color → property (warm/cool/primary/secondary)

TOKENIZATION CHALLENGE:
  Many scientific terms require multiple tokens in LLMs.
  We identify which test words are single-token in Qwen2.

METHOD:
  entity_excl: for each source word, find nearest vocab neighbor
  Universal direction: mean(target - source) → does it generalize?
  We test both: raw proximity AND direction-augmented lookup

HYPOTHESIS:
  Scientific categories that are densely represented in web text
  (animals, planets, colors) will show W_E structure.
  Domain-specific facts (element→symbol) will fail due to rare co-occurrence.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day162_domain_extension.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Test data ───────────────────────────────────────────────

# Biology: animal → class
ANIMAL_CLASS = [
    ("dog","mammal"),("cat","mammal"),("horse","mammal"),("whale","mammal"),
    ("eagle","bird"),("parrot","bird"),("penguin","bird"),("sparrow","bird"),
    ("salmon","fish"),("shark","fish"),("trout","fish"),
    ("frog","amphibian"),("toad","amphibian"),
    ("cobra","reptile"),("lizard","reptile"),
    ("ant","insect"),("bee","insect"),("moth","insect"),("beetle","insect"),
]

# Animal → diet
ANIMAL_DIET = [
    ("lion","carnivore"),("tiger","carnivore"),("wolf","carnivore"),("shark","carnivore"),
    ("rabbit","herbivore"),("deer","herbivore"),("cow","herbivore"),("horse","herbivore"),
    ("bear","omnivore"),("pig","omnivore"),("crow","omnivore"),("rat","omnivore"),
]

# Planet → type
PLANET_TYPE = [
    ("Mercury","rocky"),("Venus","rocky"),("Earth","rocky"),("Mars","rocky"),
    ("Jupiter","gas"),("Saturn","gas"),("Uranus","gas"),("Neptune","gas"),
]

# Color → temperature
COLOR_TEMP = [
    ("red","warm"),("orange","warm"),("yellow","warm"),
    ("blue","cool"),("green","cool"),("purple","cool"),
    ("white","neutral"),("black","neutral"),("gray","neutral"),
]

# Color → primary/secondary
COLOR_KIND = [
    ("red","primary"),("blue","primary"),("yellow","primary"),
    ("green","secondary"),("orange","secondary"),("purple","secondary"),
]

# Chemical elements (those likely single-token)
ELEMENT_METAL = [
    ("iron","metal"),("copper","metal"),("gold","metal"),("silver","metal"),
    ("aluminum","metal"),("zinc","metal"),("lead","metal"),("tin","metal"),
    ("carbon","nonmetal"),("nitrogen","nonmetal"),("oxygen","nonmetal"),
    ("hydrogen","nonmetal"),("sulfur","nonmetal"),("phosphorus","nonmetal"),
]

# Geographic + cultural (known to work — use as positive control)
CONTROL_CAPS = [
    ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("China","Beijing"),
    ("Italy","Rome"),("Spain","Madrid"),
]
CONTROL_ANTONYMS = [
    ("hot","cold"),("big","small"),("fast","slow"),("good","bad"),
    ("young","old"),("rich","poor"),
]

# ─── Extended vocabulary for this experiment ─────────────────
VOCAB_SCIENCE = [
    # animals
    "mammal","bird","fish","reptile","amphibian","insect","animal","plant",
    "carnivore","herbivore","omnivore",
    "dog","cat","horse","whale","eagle","parrot","penguin","salmon","shark",
    "frog","cobra","ant","bee","lion","tiger","wolf","rabbit","deer","bear",
    "pig","crow","rat","mouse","duck","owl","snake","turtle","beetle","moth",
    # planets
    "Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
    "rocky","gas","inner","outer",
    # colors
    "red","blue","yellow","green","orange","purple","white","black","gray",
    "warm","cool","primary","secondary","neutral",
    # elements
    "iron","copper","gold","silver","aluminum","zinc","lead","tin",
    "carbon","nitrogen","oxygen","hydrogen","sulfur","phosphorus",
    "metal","nonmetal","element","solid","liquid","gas",
    # general
    "heavy","light","large","small","fast","slow","hot","cold",
    "hard","soft","rough","smooth","bright","dark",
    # capitals (control)
    "Paris","Berlin","Tokyo","Beijing","Rome","Madrid","London","Moscow",
    # languages (control)
    "French","German","Japanese","Chinese","Italian","Spanish",
]

VOCAB_FULL = list(dict.fromkeys(VOCAB_SCIENCE))  # dedup

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
H   = W_E.shape[1]
del model
print(f"  H={H}, vocab={W_E.shape[0]}\n")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

# Build vocabulary
vocab_ok = [w for w in VOCAB_FULL if tid(w)]
vocab_embs = {w: W_E[tid(w)] for w in vocab_ok}
N = len(vocab_ok)
print(f"Single-token vocabulary: {N}/{len(VOCAB_FULL)}")

# Check which test words are single-token
def check_coverage(pairs, name):
    n_ok = sum(1 for a, b in pairs if tid(a) and tid(b))
    print(f"  {name:>20}: {n_ok}/{len(pairs)} pairs single-token")
    return n_ok

print("\nTest data single-token coverage:")
check_coverage(ANIMAL_CLASS,  "animal→class")
check_coverage(ANIMAL_DIET,   "animal→diet")
check_coverage(PLANET_TYPE,   "planet→type")
check_coverage(COLOR_TEMP,    "color→temp")
check_coverage(COLOR_KIND,    "color→primary")
check_coverage(ELEMENT_METAL, "element→metal/nonmetal")
check_coverage(CONTROL_CAPS,  "capitals (control)")
check_coverage(CONTROL_ANTONYMS, "antonyms (control)")

def entity_excl(source, direction, exclude_set):
    eid = tid(source)
    if eid is None: return None, 0.0
    e = W_E[eid].copy()
    if direction is not None: e = e + direction
    excl = [w for w in vocab_ok if w not in exclude_set]
    if not excl: return None, 0.0
    scores = {w: cosine(e, vocab_embs[w]) for w in excl}
    top1 = max(excl, key=lambda w: scores[w])
    return top1, scores[top1]

def make_dir(pairs):
    ds = [normed(W_E[tid(b)] - W_E[tid(a)])
          for a, b in pairs if tid(a) and tid(b)]
    return normed(np.mean(ds, axis=0)) if ds else None

def test_domain(pairs, name, direction=None, k_shot=0):
    """Test entity_excl on a domain. k_shot: use first k pairs to build direction."""
    if k_shot > 0:
        train  = pairs[:k_shot]
        test   = pairs[k_shot:]
        d_vec  = make_dir(train)
    else:
        test  = pairs
        d_vec = direction

    n_correct = 0
    results = []
    for source, target in test:
        if not (tid(source) and tid(target)): continue
        pred, score = entity_excl(source, d_vec, {source})
        correct = (pred == target)
        if correct: n_correct += 1
        results.append({"source": source, "target": target,
                         "pred": pred, "score": round(float(score),4),
                         "correct": correct})
    n = len(results)
    if n == 0: return results, 0, 0
    return results, n_correct, n

print()
print("="*68)
print("DOMAIN TESTS")
print("="*68)
print()

all_results = {}

# ─── Control: known-working domains ──────────────────────────
print("CONTROL DOMAINS (should work)")
print("-"*40)
cap_dir = make_dir(CONTROL_CAPS)
antonym_dir = make_dir(CONTROL_ANTONYMS)

res, nc, n = test_domain(CONTROL_CAPS, "capitals", cap_dir)
print(f"  capitals:    {nc}/{n} = {nc/n:.2f}" if n else "  capitals: no data")
all_results["control_caps"] = {"n": n, "nc": nc, "cases": res}

res, nc, n = test_domain(CONTROL_ANTONYMS, "antonyms", antonym_dir)
print(f"  antonyms:    {nc}/{n} = {nc/n:.2f}" if n else "  antonyms: no data")
all_results["control_antonyms"] = {"n": n, "nc": nc, "cases": res}

# ─── Scientific domains ───────────────────────────────────────
print()
print("SCIENTIFIC DOMAINS")
print("-"*40)

# Animal → class (no direction, pure proximity)
res, nc, n = test_domain(ANIMAL_CLASS, "animal→class")
print(f"  animal→class (proximity only): {nc}/{n} = {nc/n:.2f}" if n else "  0 pairs")
for r in res: print(f"    {r['source']:>12} → pred:{r['pred']:<12} target:{r['target']}  {'✓' if r['correct'] else '✗'}")
all_results["animal_class_ndir"] = {"n": n, "nc": nc, "cases": res}
print()

# Animal → class (with LOO direction built from training set — 3-shot)
res3, nc3, n3 = test_domain(ANIMAL_CLASS, "animal→class (3-shot dir)", k_shot=3)
print(f"  animal→class (3-shot direction): {nc3}/{n3} = {nc3/n3:.2f}" if n3 else "")
all_results["animal_class_3shot"] = {"n": n3, "nc": nc3, "cases": res3}
print()

# Animal → diet
res, nc, n = test_domain(ANIMAL_DIET, "animal→diet")
print(f"  animal→diet (proximity only): {nc}/{n} = {nc/n:.2f}" if n else "")
for r in res: print(f"    {r['source']:>10} → pred:{r['pred']:<12} target:{r['diet'] if 'diet' in r else r['target']}  {'✓' if r['correct'] else '✗'}")
all_results["animal_diet"] = {"n": n, "nc": nc, "cases": res}
print()

# Planet → type
res, nc, n = test_domain(PLANET_TYPE, "planet→type")
print(f"  planet→type (proximity only): {nc}/{n} = {nc/n:.2f}" if n else "")
for r in res: print(f"    {r['source']:>10} → pred:{r['pred']:<10} target:{r['target']}  {'✓' if r['correct'] else '✗'}")
all_results["planet_type"] = {"n": n, "nc": nc, "cases": res}
print()

# Color → temp
res, nc, n = test_domain(COLOR_TEMP, "color→temp")
print(f"  color→temp (proximity only): {nc}/{n} = {nc/n:.2f}" if n else "")
for r in res: print(f"    {r['source']:>10} → pred:{r['pred']:<10} target:{r['target']}  {'✓' if r['correct'] else '✗'}")
all_results["color_temp"] = {"n": n, "nc": nc, "cases": res}
print()

# Color → primary/secondary
res, nc, n = test_domain(COLOR_KIND, "color→primary/secondary")
print(f"  color→primary (proximity only): {nc}/{n} = {nc/n:.2f}" if n else "")
for r in res: print(f"    {r['source']:>10} → pred:{r['pred']:<10} target:{r['target']}  {'✓' if r['correct'] else '✗'}")
all_results["color_kind"] = {"n": n, "nc": nc, "cases": res}
print()

# Elements → metal/nonmetal
res, nc, n = test_domain(ELEMENT_METAL, "element→metal/nonmetal")
print(f"  element→metal/nonmetal (proximity only): {nc}/{n} = {nc/n:.2f}" if n else "")
for r in res: print(f"    {r['source']:>14} → pred:{r['pred']:<12} target:{r['target']}  {'✓' if r['correct'] else '✗'}")
all_results["element_metal"] = {"n": n, "nc": nc, "cases": res}

# ─── Universal directions for scientific domains ──────────────
print()
print("="*68)
print("UNIVERSAL DIRECTIONS FOR SCIENTIFIC DOMAINS")
print("="*68)
print()

# Does a "class direction" generalize? Build from mammals, test on birds
mammal_pairs = [(a,b) for a,b in ANIMAL_CLASS if b=="mammal"][:3]
bird_pairs   = [(a,b) for a,b in ANIMAL_CLASS if b=="bird"][:2]

mammal_dir = make_dir(mammal_pairs)
if mammal_dir is not None:
    print("  mammal_dir (from dog,cat,horse → mammal):")
    for a, b in ANIMAL_CLASS:
        if not (tid(a) and tid(b)): continue
        pred, score = entity_excl(a, mammal_dir, {a})
        print(f"    {a:>10}: pred={pred:<12} target={b}  cos={score:.3f}")

print()
# Cross-domain: does antonym direction help with warm/cool colors?
print("  antonym_dir applied to color→temp:")
for a, b in COLOR_TEMP:
    if not (tid(a) and tid(b)): continue
    pred, score = entity_excl(a, antonym_dir, {a})
    print(f"    {a:>10}: pred={pred:<10} target={b}  cos={score:.3f}")

# ─── SVD structure for scientific vocabulary ─────────────────
print()
print("="*68)
print("SVD STRUCTURE OF SCIENTIFIC VOCABULARY")
print("="*68)
print()

M = np.array([vocab_embs[w] for w in vocab_ok], dtype=np.float32)
M_c = M - M.mean(axis=0)
_, S, Vt = np.linalg.svd(M_c, full_matrices=False)
print(f"  S[:6] = {S[:6].round(2)}\n")

print("  Top-5 SVD components:")
for k in range(5):
    scores = [(vocab_ok[i], float(M_c[i] @ Vt[k])) for i in range(N)]
    scores.sort(key=lambda x: -x[1])
    top = [w for w,_ in scores[:5]]
    bot = [w for w,_ in scores[-4:]]
    print(f"  PC{k}: +{top}  -{bot}")
print()

# Check if any SVD component aligns with scientific directions
class_dir = make_dir([(a,b) for a,b in ANIMAL_CLASS if tid(a) and tid(b)])
diet_dir  = make_dir([(a,b) for a,b in ANIMAL_DIET  if tid(a) and tid(b)])
planet_dir= make_dir([(a,b) for a,b in PLANET_TYPE  if tid(a) and tid(b)])

print("  Scientific direction → SVD alignment (top-10 PCs):")
for dname, dvec in [("animal→class", class_dir), ("animal→diet", diet_dir),
                     ("planet→type", planet_dir)]:
    if dvec is None: continue
    aligns = [(k, cosine(Vt[k], dvec)) for k in range(20)]
    best_k, best_c = max(aligns, key=lambda x: abs(x[1]))
    print(f"    {dname:>16}: best PC{best_k}, cos={best_c:.3f}")

# ─── Summary ─────────────────────────────────────────────────
print()
print("="*68)
print("Summary")
print("="*68)
print(f"""
  Domain                   Accuracy    (proximity only)
  ────────────────────────────────────────────────────────
  capitals (control):      {all_results['control_caps']['nc']}/{all_results['control_caps']['n']}
  antonyms (control):      {all_results['control_antonyms']['nc']}/{all_results['control_antonyms']['n']}
  animal→class:            {all_results['animal_class_ndir']['nc']}/{all_results['animal_class_ndir']['n']}
  animal→class (3-shot):   {all_results['animal_class_3shot']['nc']}/{all_results['animal_class_3shot']['n']}
  animal→diet:             {all_results['animal_diet']['nc']}/{all_results['animal_diet']['n']}
  planet→type:             {all_results['planet_type']['nc']}/{all_results['planet_type']['n']}
  color→temperature:       {all_results['color_temp']['nc']}/{all_results['color_temp']['n']}
  color→primary/secondary: {all_results['color_kind']['nc']}/{all_results['color_kind']['n']}
  element→metal/nonmetal:  {all_results['element_metal']['nc']}/{all_results['element_metal']['n']}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 162 complete.")
