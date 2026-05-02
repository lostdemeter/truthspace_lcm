#!/usr/bin/env python3
"""
Day 101 — Multi-Hop Semantic Navigation

If the φ-trie is a true geometric coordinate system, then two sequential
bit flips should be composable: A→B via axis k1, B→C via axis k2 implies
A→C via simultaneous double flip (flip k1 and k2 in A's address).

TEST DESIGN:
  1. Load Day 99 navigability graph (5,977 confirmed A→B edges)
  2. Find chains A→B→C where:
     - A→B is a confirmed navigable pair (rank < 5) via axis k1
     - B→C is a confirmed navigable pair (rank < 5) via axis k2, k2 ≠ k1
  3. For each chain, test if:
     - Single hop: flip k1 in addr(A) → find B (expected: rank < 5)
     - Single hop: flip k2 in addr(B) → find C (expected: rank < 5)
     - Composed hop: flip BOTH k1+k2 in addr(A) → find C (new test)
  4. Compare: does rank(C | flip k1+k2 from A) correlate with
     rank(B | flip k1 from A) + rank(C | flip k2 from B)?

SECONDARY TEST: Triple-hop A→B→C→D (three bit flips). Does rank degrade?

PREDICTION (coordinate system):
  - Composed hop rank should be low (< 20) for confirmed two-hop chains
  - Rank should increase with number of hops but remain better than random
  - This confirms: the trie's 12D address space supports compositional navigation

PREDICTION (fingerprint only):
  - Composed hop rank should be high (random) because the double-flipped
    address may not correspond to any stable semantic position
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day101_multihop.json")
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

# T2 axes
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

# Probe token hidden states
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

# 12D addresses
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

def hamming_str(a, b): return sum(x != y for x, y in zip(a, b))
FLIP_MAP = {"H": "L", "L": "H", "U": "H"}

def flip_bits(addr_str, bit_positions):
    fl = list(addr_str)
    for k in bit_positions:
        fl[k] = FLIP_MAP[fl[k]]
    return "".join(fl)

def get_rank(src_idx, tgt_idx, flipped_addr):
    ranked = sorted([(j, hamming_str(flipped_addr, addresses[j]))
                     for j in range(N) if j != src_idx], key=lambda x: x[1])
    rank = next((r for r, (j, _) in enumerate(ranked) if j == tgt_idx), -1)
    top5 = [valid_words[j] for j, _ in ranked[:5]]
    return rank, top5

# ── Load Day 99 navigability graph ─────────────────────────────────────────────
day99_path = SCRIPT_DIR / "day99_navigable_pairs.json"
if not day99_path.exists():
    print("ERROR: day99_navigable_pairs.json not found. Run Day 99 first.")
    exit(1)

with open(day99_path) as f:
    day99 = json.load(f)

nav_pairs = day99.get("confirmed", [])
print(f"Loaded {len(nav_pairs)} confirmed navigable pairs from Day 99\n")

# Build index: word → list of (tgt, axis_bit_idx) that are navigable from it
nav_from = defaultdict(list)
for r in nav_pairs:
    src, tgt, axis = r["src"], r["tgt"], r["axis"]
    if src in word_idx and tgt in word_idx:
        k = AXIS_NAMES_12.index(axis)
        nav_from[src].append((tgt, k, r["rank"]))

# ── Find two-hop chains A→B→C ──────────────────────────────────────────────────
print("=" * 72)
print("Finding two-hop chains A → B → C")
print("=" * 72)

chains2 = []
for A in nav_from:
    for B, k1, rank_AB in nav_from[A]:
        if B not in nav_from: continue
        for C, k2, rank_BC in nav_from[B]:
            if C == A: continue
            if k2 == k1: continue  # must use different axes
            if C not in word_idx: continue
            chains2.append((A, B, C, k1, k2, rank_AB, rank_BC))

print(f"  Found {len(chains2)} two-hop chains (k1 ≠ k2)\n")

# Sample up to 500 chains for testing (avoid too long runtime)
rng = np.random.default_rng(42)
if len(chains2) > 500:
    indices = rng.choice(len(chains2), 500, replace=False)
    sample2 = [chains2[i] for i in indices]
else:
    sample2 = chains2
print(f"  Testing {len(sample2)} sampled chains\n")

# Test: compose two bit flips
results2 = []
rank_AB_list = []; rank_BC_list = []; rank_AC_list = []
for A, B, C, k1, k2, rank_AB, rank_BC in sample2:
    si = word_idx[A]; bi = word_idx[B]; ci = word_idx[C]
    # Composed: flip k1 AND k2 in A's address
    fl_AC = flip_bits(addresses[si], [k1, k2])
    rank_AC, top5_AC = get_rank(si, ci, fl_AC)
    # Verify single hops still work
    fl_AB = flip_bits(addresses[si], [k1])
    rank_AB_v, _ = get_rank(si, bi, fl_AB)
    fl_BC = flip_bits(addresses[bi], [k2])
    rank_BC_v, _ = get_rank(bi, ci, fl_BC)
    results2.append({
        "A": A, "B": B, "C": C,
        "axis1": AXIS_NAMES_12[k1], "axis2": AXIS_NAMES_12[k2],
        "rank_AB": rank_AB_v, "rank_BC": rank_BC_v, "rank_AC": rank_AC,
        "top5_AC": top5_AC,
        "hit_AB": 0 <= rank_AB_v < 5,
        "hit_BC": 0 <= rank_BC_v < 5,
        "hit_AC": 0 <= rank_AC < 5,
    })
    rank_AB_list.append(rank_AB_v); rank_BC_list.append(rank_BC_v)
    rank_AC_list.append(rank_AC if rank_AC >= 0 else N)

# Filter to chains where both single hops actually work
verified = [r for r in results2 if r["hit_AB"] and r["hit_BC"]]
print(f"  Chains where BOTH single hops verified: {len(verified)}/{len(sample2)}")
if verified:
    hit_AC_v = sum(1 for r in verified if r["hit_AC"])
    print(f"  Two-hop composed navigation (rank<5): {hit_AC_v}/{len(verified)} ({100*hit_AC_v/len(verified):.0f}%)")
    rank_AC_verified = [r["rank_AC"] if r["rank_AC"] >= 0 else N for r in verified]
    print(f"  Mean rank for composed hop: {np.mean(rank_AC_verified):.1f}")
    print(f"  Median rank for composed hop: {np.median(rank_AC_verified):.1f}")

# Single hop stats for comparison
hit_AB_all = sum(1 for r in results2 if r["hit_AB"])
hit_BC_all = sum(1 for r in results2 if r["hit_BC"])
hit_AC_all = sum(1 for r in results2 if r["hit_AC"])
print(f"\n  All sampled chains:")
print(f"  Single hop A→B (rank<5):   {hit_AB_all}/{len(results2)} ({100*hit_AB_all/len(results2):.0f}%)")
print(f"  Single hop B→C (rank<5):   {hit_BC_all}/{len(results2)} ({100*hit_BC_all/len(results2):.0f}%)")
print(f"  Composed A→C (rank<5):     {hit_AC_all}/{len(results2)} ({100*hit_AC_all/len(results2):.0f}%)")

# Print sample chains
print()
print("=" * 72)
print("Sample two-hop chains (verified both hops)")
print("=" * 72)
print(f"\n  {'A':>10} \u2192 {'B':>10} \u2192 {'C':>10}  via {'ax1':>12}/{'ax2':>12}  "
      f"{'r_AB':>5} {'r_BC':>5} {'r_AC':>5}")
shown = 0
for r in sorted(verified, key=lambda x: x["rank_AC"] if x["rank_AC"] >= 0 else N):
    if shown >= 30: break
    flag = "\u2713" if r["hit_AC"] else "\u2717"
    rac = r["rank_AC"] if r["rank_AC"] >= 0 else "miss"
    print(f"  {r['A']:>10} \u2192 {r['B']:>10} \u2192 {r['C']:>10}  "
          f"via {r['axis1']:>12}/{r['axis2']:>12}  "
          f"{r['rank_AB']:>5} {r['rank_BC']:>5} {str(rac):>5} {flag}")
    shown += 1

# ── Three-hop chains A→B→C→D ──────────────────────────────────────────────────
print()
print("=" * 72)
print("Finding three-hop chains A → B → C → D")
print("=" * 72)

chains3 = []
count = 0
for A, B, C, k1, k2, rAB, rBC in chains2:
    if C not in nav_from: continue
    for D, k3, rCD in nav_from[C]:
        if D == A or D == B: continue
        if k3 == k1 or k3 == k2: continue
        if D not in word_idx: continue
        chains3.append((A, B, C, D, k1, k2, k3))
        count += 1
        if count > 10000: break
    if count > 10000: break

print(f"  Found {min(len(chains3), 10000)}+ three-hop chains\n")

# Sample 200 for testing
if len(chains3) > 200:
    indices3 = rng.choice(len(chains3), 200, replace=False)
    sample3 = [chains3[i] for i in indices3]
else:
    sample3 = chains3

results3 = []
for A, B, C, D, k1, k2, k3 in sample3:
    si = word_idx[A]; di = word_idx[D]
    fl_AD = flip_bits(addresses[si], [k1, k2, k3])
    rank_AD, top5_AD = get_rank(si, di, fl_AD)
    results3.append({
        "A": A, "B": B, "C": C, "D": D,
        "k1": AXIS_NAMES_12[k1], "k2": AXIS_NAMES_12[k2], "k3": AXIS_NAMES_12[k3],
        "rank_AD": rank_AD,
        "hit_AD": 0 <= rank_AD < 5,
    })

hit_AD = sum(1 for r in results3 if r["hit_AD"])
rank_AD_list = [r["rank_AD"] if r["rank_AD"] >= 0 else N for r in results3]
print(f"  Three-hop composed navigation (rank<5): {hit_AD}/{len(results3)} ({100*hit_AD/max(1,len(results3)):.0f}%)")
print(f"  Mean rank: {np.mean(rank_AD_list):.1f}")

# ── Rank degradation by number of hops ────────────────────────────────────────
print()
print("=" * 72)
print("Rank degradation by number of hops")
print("=" * 72)
# Single hop baseline
single_ranks = [r["rank_AB"] if r["rank_AB"] >= 0 else N for r in results2]
double_ranks = [r["rank_AC"] if r["rank_AC"] >= 0 else N for r in results2]
triple_ranks = rank_AD_list if results3 else []

print(f"\n  Hops  mean_rank  median_rank  top5_pct")
if single_ranks:
    top5_1 = sum(1 for r in single_ranks if r < 5)
    print(f"     1  {np.mean(single_ranks):>9.1f}  {np.median(single_ranks):>11.1f}  {100*top5_1/len(single_ranks):>7.0f}%")
if double_ranks:
    top5_2 = sum(1 for r in double_ranks if r < 5)
    print(f"     2  {np.mean(double_ranks):>9.1f}  {np.median(double_ranks):>11.1f}  {100*top5_2/len(double_ranks):>7.0f}%")
if triple_ranks:
    top5_3 = sum(1 for r in triple_ranks if r < 5)
    print(f"     3  {np.mean(triple_ranks):>9.1f}  {np.median(triple_ranks):>11.1f}  {100*top5_3/len(triple_ranks):>7.0f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 101 Summary")
print("=" * 72)

two_hop_pct  = 100*hit_AC_all/max(1,len(results2))
three_hop_pct = 100*hit_AD/max(1,len(results3))
one_hop_pct  = 100*hit_AB_all/max(1,len(results2))

interp = (
    "COORDINATE SYSTEM CONFIRMED: Multi-hop navigation degrades gracefully."
    if two_hop_pct > 20 else
    "PARTIAL: Two-hop navigation works for some chains but degrades significantly."
    if two_hop_pct > 5 else
    "FINGERPRINT ONLY: Two-hop navigation fails — composed flips don't reach stable positions."
)

print(f"""
  Two-hop chains tested:    {len(sample2)}
  Verified (both hops ok):  {len(verified)}

  1-hop A→B accuracy:    {one_hop_pct:.0f}%
  2-hop A→C accuracy:    {two_hop_pct:.0f}%
  3-hop A→D accuracy:    {three_hop_pct:.0f}%

  Mean rank 1-hop: {np.mean(single_ranks):.1f}
  Mean rank 2-hop: {np.mean(double_ranks):.1f}
  Mean rank 3-hop: {f"{np.mean(triple_ranks):.1f}" if triple_ranks else "N/A"}

  {interp}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_chains2": len(sample2),
        "n_verified": len(verified),
        "hit_AC_pct": two_hop_pct,
        "hit_AB_pct": one_hop_pct,
        "hit_AD_pct": three_hop_pct,
        "mean_rank_1hop": float(np.mean(single_ranks)),
        "mean_rank_2hop": float(np.mean(double_ranks)),
        "mean_rank_3hop": float(np.mean(triple_ranks)) if triple_ranks else None,
        "sample_results2": results2[:100],
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 101 complete.")
