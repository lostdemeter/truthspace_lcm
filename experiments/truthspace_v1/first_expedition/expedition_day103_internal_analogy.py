#!/usr/bin/env python3
"""
Day 103 — Trie-Internal Analogy Accuracy

Day 102: external human analogies hit 35% top-5. The trie is a cluster-local
analogy solver. Day 103 tests the limit: when BOTH the template pair (A,B)
AND the target pair (C,D) are in the navigability graph via the SAME axis,
what is the analogy accuracy?

PREDICTION: Near 100% — since (A→B) and (C→D) are both confirmed navigable
via axis k, the composed analogy A:B::C:D should work by the same geometric
logic.

ALGORITHM:
  1. Load Day 99 navigability graph
  2. Group confirmed pairs by axis: for each axis k, collect all pairs (X,Y,k)
  3. For each axis k, enumerate all analogy problems (A,B,C,D) where:
     - (A→B) is confirmed navigable via axis k (rank < 5)
     - (C→D) is confirmed navigable via axis k (rank < 5)
     - A ≠ C, B ≠ D
  4. Solve each as: flip axis k in addr(C) → predict D
  5. Measure rank of D

This tests whether the trie's coordinate system interpretation holds
INTERNALLY: within the navigability graph, are analogies ~100% accurate?

SECONDARY: Stratify by na_hamming of the template pair (na_ham=0 vs na_ham=1)
and the target pair. Does template/target na_hamming predict accuracy?
"""
import json, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day103_internal_analogy.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]

AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("His brother arrived at the party","His sister arrived at the party"),
        ("The father worked to feed family","The mother worked to feed family"),
        ("A son was born in the winter","A daughter was born in the winter"),
        ("The prince rode across the land","The princess rode across the land"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
        ("The dark room","The darker room"),("A hard rock","A harder rock"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The oak crashed in the storm","The tree crashed in the storm"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The soldier marched into fight","The person marched into fight"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk","Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road","The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky","Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text","The words appeared clearly in the printed text"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He spoke loudly","He spoke noisily"),
        ("It is hard","It is difficult"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The iron chain has broken now","The bond between them has broken"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
        ("The strong root grips the soil","The strong base grips the earth"),
        ("The bridge connects two banks","The bond connects two communities"),
        ("The small key opens the door","The small answer opens the path"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days","We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend","She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire","They sang together around the evening campfire"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The light is on","The dark is on"),("The news is good","The news is bad"),
        ("It is hard","It is soft"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is the first","It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("John broke the window","The window was broken by John"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The dog bit the man","The man was bitten by the dog"),
        ("The teacher helped the student","The student was helped by the teacher"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
        ("The king signed the document","The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The sun heats the cold earth","The ice melts quickly in spring"),
        ("The wind blows the tree branches","The leaves fall to the ground"),
        ("The child cries very loudly","The mother comes running in"),
        ("The ball rolls off the tall edge","The ball falls to the floor"),
        ("The teacher praises the student","The student feels very proud"),
        ("The glass breaks on hard stone","The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The car broke down again","Did the car break down again"),
        ("The dog is hungry now","Is the dog hungry now"),
        ("She wrote the letter herself","Did she write the letter herself"),
        ("He knows the right answer","Does he know the right answer"),
        ("The house looks very old","Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today","It will not rain today"),
    ],
}

PROBE_TOKENS = list(dict.fromkeys([
    "dog","cat","bird","fish","horse","wolf","lion","tiger","elephant","mouse",
    "rabbit","deer","bear","fox","eagle","whale","shark","frog","ant","bee",
    "snake","monkey","cow","pig","sheep","goat","duck","hen","crow","owl",
    "turtle","lizard","crab","lobster","octopus","beetle","butterfly","worm",
    "fly","mosquito","cricket","spider","salmon","tuna","herring","sparrow",
    "robin","finch","parrot","tree","flower","rock","stone","wood","leaf",
    "grass","root","river","mountain","ocean","forest","desert","cloud","rain",
    "snow","wind","sun","moon","star","sky","earth","soil","seed","branch",
    "bark","thorn","moss","mushroom","coral","house","door","window","table",
    "chair","book","cup","key","car","road","bridge","boat","ship","plane",
    "train","bike","knife","fork","spoon","plate","bowl","glass","bottle","box",
    "bag","rope","wire","nail","hammer","wheel","clock","lamp","pen","paper",
    "cloth","thread","button","ring","coin","mirror","hand","foot","eye","ear",
    "nose","mouth","arm","leg","head","heart","blood","bone","skin","hair",
    "finger","toe","back","chest","neck","shoulder","run","walk","jump","swim",
    "fly","eat","sleep","talk","write","read","build","break","open","close",
    "start","stop","think","know","see","hear","feel","love","hate","want",
    "give","take","make","find","lose","push","pull","turn","move","go","come",
    "fall","rise","grow","kill","help","ran","walked","jumped","flew","ate",
    "saw","heard","broke","built","wrote","fast","slow","big","small","hot",
    "cold","old","new","hard","soft","bright","dark","strong","weak","happy",
    "sad","good","bad","right","wrong","high","low","long","short","wide",
    "narrow","deep","shallow","thick","thin","heavy","light","clean","dirty",
    "sweet","bitter","sharp","dull","loud","quiet","faster","slower","bigger",
    "smaller","better","worse","biggest","smallest","best","worst","quickly",
    "slowly","often","never","always","very","quite","really","just","still",
    "the","a","and","or","not","is","was","in","on","of","to","from","with",
    "for","he","she","it","they","we","I","you","his","her","their","my","your",
    "its","our","but","if","one","two","three","four","five","six","seven",
    "eight","nine","ten","hundred","thousand","many","few","more","less","most",
    "least","all","some","king","queen","man","woman","boy","girl","child",
    "parent","brother","sister","father","mother","son","daughter","husband",
    "wife","prince","princess","actor","actress","red","blue","green","yellow",
    "white","black","brown","orange","purple","pink","gray","gold","love","hate",
    "truth","beauty","freedom","power","time","space","mind","body","soul",
    "life","death","hope","fear","joy","pain","trust","faith","peace","war",
    "law","right","duty","honor","shame","pride","guilt","anger","grief","city",
    "town","village","country","island","valley","cave","bridge","castle",
    "market","church","school","hospital","garden","field","park","lake",
    "coast","cliff","path","bread","meat","fruit","milk","water","fire","oil",
    "salt","sugar","coffee","wine","beer","tea","egg","cheese","dogs","cats",
    "trees","birds","horses","men","women","children","hands","eyes",
    "animal","vehicle","tool","gem","burden","barrier","journey","bond",
    "large","tiny","quick","frigid","joyful","difficult","aged","noisy",
    "oak","rose","ruby",
]))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}\n")

def get_h(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_h(s1, [L])[L]; h2 = get_h(s2, [L])[L]
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
print()

print("Extracting probe token hidden states ...")
hs_by_layer = {L: [] for L in ALL_LAYERS}
valid_words = []
for word in PROBE_TOKENS:
    try:
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        for L in ALL_LAYERS:
            hs_by_layer[L].append(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
        valid_words.append(word)
    except: pass
for L in ALL_LAYERS:
    hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

def classify_all(axis_vec, layer_hs, N):
    if np.linalg.norm(axis_vec) < 1e-6: return ["U"] * N
    projs = [float(np.dot(layer_hs[i], axis_vec)) for i in range(N)]
    max_p = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * N
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

classes  = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    classes[name] = classify_all(t2_axes[name], hs_by_layer[L], N)
addresses = ["".join(classes[n][i] for n in AXIS_NAMES_12) for i in range(N)]
addr_int  = np.array([[{"H": 2, "U": 1, "L": 0}[c] for c in a] for a in addresses], dtype=np.int8)

FLIP_MAP = {"H": "L", "L": "H", "U": "H"}
def hamming_str(a, b): return sum(x != y for x, y in zip(a, b))
def nonaxis_hamming(ia, ib, skip):
    return int(np.sum(addr_int[ia] != addr_int[ib])) - int(addr_int[ia][skip] != addr_int[ib][skip])

# ── Load Day 99 navigability graph ─────────────────────────────────────────────
day99_path = SCRIPT_DIR / "day99_navigable_pairs.json"
if not day99_path.exists():
    print("ERROR: day99_navigable_pairs.json not found."); exit(1)
with open(day99_path) as f:
    day99 = json.load(f)
nav_pairs = [r for r in day99.get("confirmed", [])
             if r["src"] in word_idx and r["tgt"] in word_idx]
print(f"Loaded {len(nav_pairs)} navigable pairs\n")

# Group by axis
by_axis = defaultdict(list)
for r in nav_pairs:
    na = nonaxis_hamming(word_idx[r["src"]], word_idx[r["tgt"]],
                         AXIS_NAMES_12.index(r["axis"]))
    by_axis[r["axis"]].append({
        "src": r["src"], "tgt": r["tgt"], "rank": r["rank"], "na_ham": na
    })

# ── Build internal analogy problems ───────────────────────────────────────────
print("=" * 72)
print("Trie-Internal Analogy Test")
print("=" * 72)
print(f"\nFor each axis: sample up to 500 analogy problems (A:B::C:D)")
print(f"where both A→B and C→D are confirmed navigable via that axis.\n")

rng = random.Random(42)
all_results = []

for axis_name in AXIS_NAMES_12:
    pairs = by_axis[axis_name]
    if len(pairs) < 2: continue
    k = AXIS_NAMES_12.index(axis_name)

    # Build analogy problems from all (A→B, C→D) pairs
    problems = []
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i == j: continue
            A = pairs[i]["src"]; B = pairs[i]["tgt"]
            C = pairs[j]["src"]; D = pairs[j]["tgt"]
            if A == C or B == D or A == D or B == C: continue
            problems.append((A, B, C, D, pairs[i]["na_ham"], pairs[j]["na_ham"],
                             pairs[i]["rank"], pairs[j]["rank"]))

    if len(problems) > 500:
        problems = rng.sample(problems, 500)

    hits_exact = 0; hits_top5 = 0; ranks = []
    for A, B, C, D, na_AB, na_CD, rank_AB, rank_CD in problems:
        ic = word_idx[C]; di = word_idx[D]
        # Flip axis k in C's address
        fl = list(addresses[ic]); fl[k] = FLIP_MAP[fl[k]]; fl = "".join(fl)
        ranked = sorted([(j, hamming_str(fl, addresses[j]))
                         for j in range(N) if j != ic], key=lambda x: x[1])
        rank_D = next((r for r, (j, _) in enumerate(ranked) if j == di), -1)
        top5   = [valid_words[j] for j, _ in ranked[:5]]
        exact  = (rank_D == 0)
        top5h  = (0 <= rank_D < 5)
        if exact: hits_exact += 1
        if top5h: hits_top5 += 1
        ranks.append(rank_D if rank_D >= 0 else N)
        all_results.append({
            "axis": axis_name, "A": A, "B": B, "C": C, "D": D,
            "na_AB": na_AB, "na_CD": na_CD,
            "rank_AB": rank_AB, "rank_CD": rank_CD,
            "rank_D": rank_D, "exact": exact, "top5": top5h,
        })

    total = len(problems)
    print(f"  {axis_name:>15}: {total:>5} problems, "
          f"exact={hits_exact}/{total} ({100*hits_exact/max(1,total):.0f}%), "
          f"top5={hits_top5}/{total} ({100*hits_top5/max(1,total):.0f}%), "
          f"mean_rank={np.mean(ranks):.1f}")

# ── Aggregate stats ───────────────────────────────────────────────────────────
total_all    = len(all_results)
exact_all    = sum(1 for r in all_results if r["exact"])
top5_all     = sum(1 for r in all_results if r["top5"])
ranks_all    = [r["rank_D"] if r["rank_D"] >= 0 else N for r in all_results]

print(f"\n  TOTAL: {total_all} problems")
print(f"  Exact match: {exact_all}/{total_all} ({100*exact_all/max(1,total_all):.0f}%)")
print(f"  Top-5 match: {top5_all}/{total_all} ({100*top5_all/max(1,total_all):.0f}%)")
print(f"  Mean rank:   {np.mean(ranks_all):.1f}")
print(f"  Median rank: {np.median(ranks_all):.1f}")

# ── Stratify by na_hamming ────────────────────────────────────────────────────
print()
print("=" * 72)
print("Analogy accuracy by na_hamming of target pair (C→D)")
print("=" * 72)
by_na = defaultdict(lambda: {"exact": 0, "top5": 0, "total": 0})
for r in all_results:
    na = r["na_CD"]
    by_na[na]["total"] += 1
    if r["exact"]: by_na[na]["exact"] += 1
    if r["top5"]:  by_na[na]["top5"] += 1
print(f"\n  {'na_CD':>6}  {'total':>6}  {'exact%':>7}  {'top5%':>6}")
for na in sorted(by_na):
    d = by_na[na]
    if d["total"] > 0:
        print(f"  {na:>6}  {d['total']:>6}  {100*d['exact']/d['total']:>6.0f}%  "
              f"{100*d['top5']/d['total']:>5.0f}%")

# ── Sample analogy problems (exact hits) ─────────────────────────────────────
print()
print("=" * 72)
print("Sample exact-match internal analogies")
print("=" * 72)
exact_hits = [r for r in all_results if r["exact"]]
rng2 = random.Random(99)
sample = rng2.sample(exact_hits, min(30, len(exact_hits)))
print(f"\n  {'A':>10}:{'{B}':>10} :: {'C':>10}:{'{D}':>10}  axis  na_AB na_CD")
print(f"  {'-'*70}")
for r in sorted(sample, key=lambda x: x["axis"]):
    print(f"  {r['A']:>10}:{r['B']:>10} :: {r['C']:>10}:{r['D']:>10}  "
          f"{r['axis']:>12}  {r['na_AB']:>5} {r['na_CD']:>5}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 103 Summary")
print("=" * 72)
print(f"""
  Trie-internal analogy accuracy (both A→B and C→D in navigability graph):
  Exact match: {exact_all}/{total_all} ({100*exact_all/max(1,total_all):.0f}%)
  Top-5 match: {top5_all}/{total_all} ({100*top5_all/max(1,total_all):.0f}%)
  Mean rank:   {np.mean(ranks_all):.1f}

  Comparison with Day 102 (human analogies):
  Human:        1/20 exact (5%),  7/20 top-5 (35%)
  Trie-internal: {100*exact_all/max(1,total_all):.0f}% exact, {100*top5_all/max(1,total_all):.0f}% top-5

  CONCLUSION:
  {'→ Trie IS a consistent internal coordinate system for analogy' if exact_all/max(1,total_all) >= 0.6 else
   '→ Moderate internal consistency — not all same-axis pairs are analogous' if exact_all/max(1,total_all) >= 0.3 else
   '→ Internal analogy accuracy is low — navigability does not imply analogy'}

  The gap between internal ({100*exact_all/max(1,total_all):.0f}%) and human (5%) accuracy
  {'confirms that the trie encodes a different semantic structure than human intuition.' if exact_all/max(1,total_all) > 20 else
   'is smaller than expected — the trie may not be as internally consistent as predicted.'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "total": total_all, "exact_pct": 100*exact_all/max(1,total_all),
        "top5_pct": 100*top5_all/max(1,total_all),
        "mean_rank": float(np.mean(ranks_all)),
        "by_na": {str(k): dict(v) for k, v in by_na.items()},
        "sample_exact": exact_hits[:100],
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 103 complete.")
