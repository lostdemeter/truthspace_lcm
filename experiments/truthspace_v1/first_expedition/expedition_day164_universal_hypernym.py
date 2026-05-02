#!/usr/bin/env python3
"""
Day 164 — Universal Hypernym Direction

Day 162 showed that domain-specific directions don't generalize
(mammal direction doesn't help identify birds).

QUESTION: Is there a UNIVERSAL "instance → category" direction
in W_E that works across ALL domains?

Build direction from mixed cross-domain pairs:
  dog→animal, Paris→city, iron→metal, red→color,
  ant→insect, salmon→fish, rose→flower, oak→tree

Then test if this universal direction helps:
  - Known-good domains (capitals, antonyms, gender)
  - Partially-working domains (insects, metals)
  - Failing domains (planets, colors→temp)

ALSO TEST: within-domain vs cross-domain direction transfer
  Does animal→class direction help planet→type?
  Does antonym direction help color→temperature?

THEORY (if universal direction exists):
  There is a single "is-a" axis in W_E that encodes the
  instance-to-category relationship across all domains.
  This would be the strongest version of the geometry-is-knowledge claim.

THEORY (if no universal direction):
  Each domain has its own geometric structure.
  The "is-a" relationship is not a single axis but a manifold.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day164_universal_hypernym.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Diverse training pairs for universal hypernym direction ─────
# Each row: (instance, category) — mixed across all working domains
HYPERNYM_TRAIN = [
    # geography
    ("Paris",   "city"),
    ("Berlin",  "city"),
    ("Tokyo",   "city"),
    ("France",  "country"),
    ("Germany", "country"),
    # animals
    ("dog",     "animal"),
    ("cat",     "animal"),
    ("horse",   "animal"),
    ("ant",     "insect"),
    ("bee",     "insect"),
    ("salmon",  "fish"),
    # materials/objects
    ("iron",    "metal"),
    ("copper",  "metal"),
    ("gold",    "metal"),
    # nature
    ("oak",     "tree"),
    ("rose",    "flower"),
    ("eagle",   "bird"),
]

# ─── Test set: held-out across all domains ───────────────────────
HYPERNYM_TEST = [
    # geo (held out)
    ("Rome",     "city",    {"Rome"}),
    ("London",   "city",    {"London"}),
    ("Japan",    "country", {"Japan"}),
    # animals (held out)
    ("whale",    "animal",  {"whale"}),
    ("shark",    "animal",  {"shark"}),
    ("beetle",   "insect",  {"beetle"}),
    ("moth",     "insect",  {"moth"}),
    ("trout",    "fish",    {"trout"}),
    # materials (held out)
    ("tin",      "metal",   {"tin"}),
    ("aluminum", "metal",   {"aluminum"}),
    # plants (held out)
    ("elm",      "tree",    {"elm"}),
    ("tulip",    "flower",  {"tulip"}),
    ("sparrow",  "bird",    {"sparrow"}),
]

# ─── Direction transfer test: does domain A direction help domain B? ─
DIRECTION_PAIRS = {
    "capitals": [("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),
                 ("China","Beijing"),("Italy","Rome"),("Spain","Madrid")],
    "antonyms": [("hot","cold"),("big","small"),("fast","slow"),
                 ("good","bad"),("young","old"),("rich","poor")],
    "gender":   [("king","queen"),("man","woman"),("boy","girl"),
                 ("son","daughter"),("actor","actress"),("father","mother")],
    "animals":  [("dog","animal"),("cat","animal"),("horse","animal"),
                 ("ant","insect"),("bee","insect"),("salmon","fish")],
    "metals":   [("iron","metal"),("copper","metal"),("gold","metal"),("tin","metal")],
    "planets":  [("Mercury","rocky"),("Mars","rocky"),("Jupiter","gas"),("Saturn","gas")],
    "colors":   [("red","warm"),("yellow","warm"),("blue","cool"),("green","cool")],
}

# ─── Test prompts that span domains ──────────────────────────────
TEST_PROMPTS = [
    # geo
    ("capitals",  [("France","Paris"),("Germany","Berlin"),("Greece","Athens"),
                   ("Poland","Warsaw"),("Sweden","Stockholm")]),
    # antonyms
    ("antonyms",  [("hot","cold"),("big","small"),("fast","slow"),
                   ("dark","light"),("rich","poor"),("loud","quiet")]),
    # animals
    ("animals",   [("ant","insect"),("bee","insect"),("beetle","insect"),
                   ("salmon","fish"),("dog","animal"),("eagle","bird")]),
    # metals
    ("metals",    [("iron","metal"),("copper","metal"),("aluminum","metal"),("tin","metal")]),
    # planets
    ("planets",   [("Mercury","rocky"),("Mars","rocky"),("Jupiter","gas"),
                   ("Saturn","gas"),("Uranus","gas")]),
    # colors
    ("colors_temp",[("red","warm"),("orange","warm"),("blue","cool"),
                    ("green","cool"),("white","neutral")]),
]

# ─── Extended vocabulary ─────────────────────────────────────────
VOCAB = [
    "animal","insect","bird","fish","mammal","reptile","plant","tree","flower",
    "metal","mineral","element","material","substance","object","device","tool",
    "city","country","capital","language","region","continent","island",
    "warm","cool","cold","hot","neutral","primary","secondary","opposite",
    "rocky","gas","solid","liquid","heavy","light","hard","soft",
    "dog","cat","horse","whale","shark","eagle","ant","bee","beetle",
    "moth","salmon","trout","sparrow","owl","crow","duck","frog",
    "lion","tiger","wolf","bear","pig","rat","rabbit","deer",
    "iron","copper","gold","silver","aluminum","zinc","tin","lead",
    "oak","elm","pine","rose","tulip","lily","daisy",
    "Paris","London","Rome","Berlin","Tokyo","Beijing","Moscow","Madrid",
    "Athens","Warsaw","Stockholm","Vienna","Lisbon","Brussels","Oslo",
    "France","Germany","Japan","China","Italy","Spain","Russia","Greece",
    "Poland","Sweden","England","Australia","Korea","India","Brazil",
    "French","German","Japanese","Chinese","Italian","Spanish","Russian",
    "English","Greek","Polish","Swedish","Korean","Hindi","Arabic",
    "Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
    "red","blue","yellow","green","orange","purple","white","black","gray",
    "hot","cold","big","small","fast","slow","dark","light","good","bad",
    "young","old","rich","poor","loud","quiet","clean","dirty",
    "king","queen","man","woman","boy","girl","actor","actress",
    "son","daughter","father","mother","brother","sister",
]

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
print(f"Vocabulary: {len(vocab_ok)} single-token words\n")

def make_dir(pairs):
    ds = [normed(W_E[tid(b)] - W_E[tid(a)])
          for a, b in pairs if tid(a) and tid(b)]
    return normed(np.mean(ds, axis=0)) if ds else None

def entity_excl(src, direction, exclude):
    eid = tid(src)
    if eid is None: return None, 0.0
    e = W_E[eid].copy()
    if direction is not None: e = e + direction
    cands = [w for w in vocab_ok if w not in exclude]
    scores = {w: cosine(e, vocab_embs[w]) for w in cands}
    top1 = max(cands, key=lambda w: scores[w])
    return top1, scores[top1]

def test_pairs(pairs, direction, label):
    n, nc = 0, 0
    for src, tgt in pairs:
        if not (tid(src) and tid(tgt)): continue
        pred, score = entity_excl(src, direction, {src})
        ok = pred == tgt
        if ok: nc += 1
        n += 1
    pct = nc/n if n else 0
    print(f"    {label:>30}: {nc}/{n} = {pct:.2f}")
    return nc, n

# ─── Part 1: Universal hypernym direction ────────────────────────
print("="*68)
print("PART 1: Universal Hypernym Direction")
print("="*68)
print()

hyp_dir = make_dir([(a,b) for a,b in HYPERNYM_TRAIN if tid(a) and tid(b)])
print(f"  Training pairs: {sum(1 for a,b in HYPERNYM_TRAIN if tid(a) and tid(b))}/{len(HYPERNYM_TRAIN)}")

# Alignment with SVD
M = np.array([vocab_embs[w] for w in vocab_ok])
M_c = M - M.mean(axis=0)
_, S, Vt = np.linalg.svd(M_c, full_matrices=False)
if hyp_dir is not None:
    aligns = [(k, cosine(Vt[k], hyp_dir)) for k in range(20)]
    best_k, best_c = max(aligns, key=lambda x: abs(x[1]))
    print(f"  Universal hypernym dir → SVD: PC{best_k}, cos={best_c:.3f}\n")

# Test on held-out hypernym pairs
print("  Held-out hypernym test (universal direction):")
n_total, nc_total = 0, 0
for src, tgt, excl in HYPERNYM_TEST:
    if not (tid(src) and tid(tgt)): continue
    pred, score = entity_excl(src, hyp_dir, excl)
    ok = pred == tgt
    if ok: nc_total += 1
    n_total += 1
    print(f"    {src:>12} → pred:{pred:<12} target:{tgt}  {'✓' if ok else '✗'}  cos={score:.3f}")
print(f"  Total: {nc_total}/{n_total} = {nc_total/n_total:.3f}\n")

# ─── Part 2: No direction baseline (proximity only) ──────────────
print("="*68)
print("PART 2: Baseline (no direction) vs Universal Direction")
print("="*68)
print()
for domain_name, pairs in TEST_PROMPTS:
    pairs_ok = [(a,b) for a,b in pairs if tid(a) and tid(b)]
    if not pairs_ok: continue
    nc0, n0 = 0, 0
    nc1, n1 = 0, 0
    for src, tgt in pairs_ok:
        p0, _ = entity_excl(src, None, {src})
        p1, _ = entity_excl(src, hyp_dir, {src})
        if p0 == tgt: nc0 += 1
        if p1 == tgt: nc1 += 1
        n0 += 1; n1 += 1
    print(f"  {domain_name:>16}: no_dir={nc0}/{n0}={nc0/n0:.2f}  hyp_dir={nc1}/{n1}={nc1/n1:.2f}")

# ─── Part 3: Domain-to-domain direction transfer ─────────────────
print()
print("="*68)
print("PART 3: Direction Transfer Matrix")
print("="*68)
print()
print("  For each domain pair (train→test), how much does using the WRONG")
print("  domain's direction help or hurt?")
print()

all_dirs = {}
for dname, dpairs in DIRECTION_PAIRS.items():
    d = make_dir([(a,b) for a,b in dpairs if tid(a) and tid(b)])
    all_dirs[dname] = d

domain_names = list(DIRECTION_PAIRS.keys())
print(f"  {'train→':>12} " + "  ".join(f"{d:>12}" for d in domain_names))

for test_name, test_pairs_list in TEST_PROMPTS:
    pairs_ok = [(a,b) for a,b in test_pairs_list if tid(a) and tid(b)]
    if not pairs_ok: continue
    row = []
    for train_name in domain_names:
        d = all_dirs.get(train_name)
        nc = sum(1 for src, tgt in pairs_ok
                 if entity_excl(src, d, {src})[0] == tgt)
        n  = len(pairs_ok)
        row.append(f"{nc}/{n}={nc/n:.2f}")
    print(f"  {test_name:>16}: " + "  ".join(f"{r:>12}" for r in row))

# ─── Part 4: Universal direction composition ─────────────────────
print()
print("="*68)
print("PART 4: Can We Compose Domain Directions to Improve Hypernym?")
print("="*68)
print()

# Try: cap_dir + animal_dir + metal_dir as composite hypernym
cap_dir    = all_dirs.get("capitals")
animal_dir = all_dirs.get("animals")
metal_dir  = all_dirs.get("metals")

if all([cap_dir is not None, animal_dir is not None, metal_dir is not None]):
    composite = normed(normed(cap_dir) + normed(animal_dir) + normed(metal_dir))
    print(f"  Composite direction (cap + animal + metal):")
    nc, n = 0, 0
    for src, tgt, excl in HYPERNYM_TEST:
        if not (tid(src) and tid(tgt)): continue
        pred, score = entity_excl(src, composite, excl)
        ok = pred == tgt
        if ok: nc += 1
        n += 1
        print(f"    {src:>12} → pred:{pred:<12} target:{tgt}  {'✓' if ok else '✗'}")
    print(f"  Composite: {nc}/{n} = {nc/n:.3f}")
    print(f"  Universal: {nc_total}/{n_total} = {nc_total/n_total:.3f}")

# ─── Summary ─────────────────────────────────────────────────────
print()
print("="*68)
print("Summary")
print("="*68)
print(f"  Universal hypernym direction (held-out): {nc_total}/{n_total} = {nc_total/n_total:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"nc_total": nc_total, "n_total": n_total}, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 164 complete.")
