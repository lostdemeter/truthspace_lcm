#!/usr/bin/env python3
"""
Day 99 — Mine All Navigable Pairs from Vocabulary

Day 98 confirmed: na_hamming(src, tgt, excl_axis) ≤ 1 AND src_bit ≠ tgt_bit
is the predictive condition for trie traversal success (rank < 5).

GOAL: Systematically find ALL pairs in the 401-token vocabulary satisfying
this condition. Then verify: do they ALL navigate with rank < 5?

PREDICTION (strong):
  If na_hamming=0 AND bit_sep=True:  rank = 0  (100% guaranteed)
  If na_hamming=1 AND bit_sep=True:  rank < 5  (mostly guaranteed)

SECONDARY GOAL: Build a "navigability map" of the trie — for each token,
which tokens can it navigate TO, and via which axis?

ALGORITHM:
  For each ordered pair (src, tgt) in vocabulary × vocabulary:
    For each axis k in 0..11:
      na_ham = Hamming(addr[src], addr[tgt], skip=k)
      bit_sep = (addr[src][k] ≠ addr[tgt][k])
      If na_ham ≤ 1 AND bit_sep:
        Add (src, tgt, axis_k, na_ham) to candidate list

  For each candidate, run traversal and measure rank.
  Report all candidates with rank < 5 as "confirmed navigable pairs."

COMPLEXITY: N² × 12 = 401² × 12 ≈ 1.9M comparisons (fast, no model calls needed)
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day99_navigable_pairs.json")
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

# Build 12D addresses
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
addr_arr  = [list(a) for a in addresses]

def hamming(a, b): return sum(x != y for x, y in zip(a, b))
def nonaxis_hamming(ia, ib, skip):
    return sum(addr_arr[ia][k] != addr_arr[ib][k] for k in range(12) if k != skip)

FLIP_MAP = {"H": "L", "L": "H", "U": "H"}

# ── Phase 1: Mine all candidate pairs (N² × 12, pure Python, fast) ───────────
print("=" * 72)
print("Phase 1: Mining candidate pairs (na_hamming ≤ 1, bit_sep=True)")
print("=" * 72)

# Precompute projections as integer arrays for fast comparison
addr_int = np.array([[{"H": 2, "U": 1, "L": 0}[c] for c in a] for a in addresses],
                     dtype=np.int8)  # shape (N, 12)

candidates = []  # (si, ti, axis_idx, na_ham)
for k in range(12):
    # For axis k: find pairs where addr[si][k] != addr[ti][k]
    # and all OTHER 11 bits differ by ≤ 1
    col_k = addr_int[:, k]
    other  = np.concatenate([addr_int[:, :k], addr_int[:, k+1:]], axis=1)  # (N, 11)
    for si in range(N):
        src_k     = col_k[si]
        src_other = other[si]  # (11,)
        # compute Hamming on other 11 bits for all targets
        na_hams   = np.sum(other != src_other[None, :], axis=1)  # (N,)
        # filter: na_ham ≤ 1, bit different
        mask = (na_hams <= 1) & (col_k != src_k)
        for ti in np.where(mask)[0]:
            if si != ti:
                candidates.append((int(si), int(ti), k, int(na_hams[ti])))

print(f"  Found {len(candidates)} candidate pairs (na_ham≤1, bit_sep=True)")

# Deduplicate by (si, ti, k)
seen = set()
unique_cands = []
for c in candidates:
    key = (c[0], c[1], c[2])
    if key not in seen:
        seen.add(key)
        unique_cands.append(c)
candidates = unique_cands
print(f"  After dedup: {len(candidates)} unique candidates\n")

# ── Phase 2: Run traversal for ALL candidates ─────────────────────────────────
print("Phase 2: Running traversal for all candidates ...")
results = []
hits_na0 = 0; total_na0 = 0
hits_na1 = 0; total_na1 = 0

for si, ti, k, na_ham in candidates:
    fl = list(addresses[si])
    fl[k] = FLIP_MAP[fl[k]]
    fl_str = "".join(fl)
    ranked = sorted([(j, hamming(fl_str, addresses[j])) for j in range(N) if j != si],
                     key=lambda x: x[1])
    rank = next((r for r, (j, _) in enumerate(ranked) if j == ti), -1)
    top5 = [valid_words[j] for j, _ in ranked[:5]]
    hit  = 0 <= rank < 5
    if na_ham == 0:
        if hit: hits_na0 += 1
        total_na0 += 1
    else:
        if hit: hits_na1 += 1
        total_na1 += 1
    results.append({
        "src": valid_words[si], "tgt": valid_words[ti],
        "axis": AXIS_NAMES_12[k], "na_hamming": na_ham,
        "rank": rank, "top5": top5, "hit": hit,
        "src_bit": addresses[si][k], "tgt_bit": addresses[ti][k],
    })

# ── Print confirmed navigable pairs ───────────────────────────────────────────
confirmed = [r for r in results if r["hit"]]
confirmed.sort(key=lambda r: (r["na_hamming"], r["rank"]))

print(f"\n  Confirmed navigable (rank<5): {len(confirmed)}/{len(results)}")
print(f"  na_ham=0: {hits_na0}/{total_na0} ({100*hits_na0/max(1,total_na0):.0f}%)")
print(f"  na_ham=1: {hits_na1}/{total_na1} ({100*hits_na1/max(1,total_na1):.0f}%)")

print()
print("=" * 72)
print("Confirmed navigable pairs:")
print("=" * 72)
print(f"\n  {'src':>12} {'tgt':>12} {'axis':>15}  na_ham  rank")
for r in confirmed[:60]:
    print(f"  {r['src']:>12} {r['tgt']:>12} {r['axis']:>15}      {r['na_hamming']}  {r['rank']:>4}")

if len(confirmed) > 60:
    print(f"  ... and {len(confirmed)-60} more")

# ── Axis breakdown ────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Navigable pairs by axis")
print("=" * 72)
by_axis = defaultdict(lambda: {"hits": 0, "total": 0})
for r in results:
    by_axis[r["axis"]]["total"] += 1
    if r["hit"]: by_axis[r["axis"]]["hits"] += 1
print(f"\n  {'axis':>15}  {'cands':>6}  {'hits':>5}  {'%':>6}")
for name in AXIS_NAMES_12:
    d = by_axis[name]
    if d["total"] > 0:
        print(f"  {name:>15}  {d['total']:>6}  {d['hits']:>5}  {100*d['hits']/d['total']:>5.0f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 99 Summary")
print("=" * 72)
print(f"""
  Total candidates (na_ham≤1, bit_sep=True): {len(results)}
  Confirmed navigable (rank<5):              {len(confirmed)} ({100*len(confirmed)/max(1,len(results)):.0f}%)

  na_ham=0 accuracy: {hits_na0}/{total_na0} ({100*hits_na0/max(1,total_na0):.0f}%)
  na_ham=1 accuracy: {hits_na1}/{total_na1} ({100*hits_na1/max(1,total_na1):.0f}%)

  PREDICTION RESULT:
  {'CONFIRMED: na_ham=0 → rank=0 holds for all pairs' if total_na0 > 0 and hits_na0 == total_na0 else
   f'PARTIAL: na_ham=0 → {hits_na0}/{total_na0} pairs rank<5'}
  {'CONFIRMED: na_ham=1 → rank<5 holds for majority' if total_na1 > 0 and hits_na1/total_na1 >= 0.7 else
   f'PARTIAL: na_ham=1 → {hits_na1}/{total_na1} pairs rank<5'}

  Most navigable axis: {max(by_axis, key=lambda k: by_axis[k]['hits']) if by_axis else 'N/A'}
  Total vocab pairs checked: {N*N*12:,} (N={N}, 12 axes)
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_candidates": len(results),
        "n_confirmed": len(confirmed),
        "hits_na0": hits_na0, "total_na0": total_na0,
        "hits_na1": hits_na1, "total_na1": total_na1,
        "confirmed": confirmed[:200],
        "axis_stats": {k: dict(v) for k, v in by_axis.items()},
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 99 complete.")
