#!/usr/bin/env python3
"""
Day 98 — Semantic Parallelism vs Traversal Rank

DC 327 proposes the semantic parallelism condition:
  Traversal succeeds when source and target are near-identical on
  K-1 non-traversed axes while differing on exactly 1.

Hypothesis: Hamming distance on the K-1 non-traversed axes is the
primary predictor of traversal rank. Low non-axis Hamming = low rank.

EXPERIMENT:
  1. For all ground truth pairs tested (Days 92-97, 80+ pairs),
     compute non-axis Hamming distance (Hamming on 11/12 bits, excluding
     the traversal axis bit)
  2. Compute traversal rank for each pair
  3. Compute Spearman correlation: non-axis Hamming vs log(rank+1)
  4. Test whether the correlation is significant

PREDICTION:
  Strong negative correlation: r ≈ -0.7 to -0.9
  Pairs with non-axis Hamming=0 should have rank≈0
  Pairs with non-axis Hamming≥4 should have rank>>50

SECONDARY TEST:
  Among all vocabulary pairs sharing the same axis bit value,
  which pairs have the lowest non-axis Hamming? These SHOULD be
  the navigable pairs (just not in our current GT set).
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day98_parallelism.json")
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

# All ground truth pairs with their axis label (accumulated from Days 92-97)
ALL_GT = [
    # gender (bit 0)
    ("king","queen","gender"),("man","woman","gender"),("boy","girl","gender"),
    ("brother","sister","gender"),("father","mother","gender"),("son","daughter","gender"),
    ("prince","princess","gender"),("actor","actress","gender"),
    # comparative (bit 1)
    ("fast","faster","comparative"),("big","bigger","comparative"),
    ("slow","slower","comparative"),("small","smaller","comparative"),
    ("good","better","comparative"),("bad","worse","comparative"),
    # hypernym (bit 2)
    ("dog","animal","hypernym"),("eagle","bird","hypernym"),("car","vehicle","hypernym"),
    ("rose","flower","hypernym"),("oak","tree","hypernym"),("salmon","fish","hypernym"),
    ("hammer","tool","hypernym"),("wolf","animal","hypernym"),("bear","animal","hypernym"),
    ("sparrow","bird","hypernym"),("owl","bird","hypernym"),("crow","bird","hypernym"),
    # plural (bit 3)
    ("dog","dogs","plural"),("cat","cats","plural"),("tree","trees","plural"),
    ("bird","birds","plural"),("hand","hands","plural"),("eye","eyes","plural"),
    # synonym (bit 4)
    ("big","large","synonym"),("small","tiny","synonym"),("fast","quick","synonym"),
    ("cold","frigid","synonym"),("happy","joyful","synonym"),("hard","difficult","synonym"),
    ("old","aged","synonym"),("loud","noisy","synonym"),
    # concrete (bit 5)
    ("road","path","concrete"),("fire","anger","concrete"),
    ("stone","burden","concrete"),("river","journey","concrete"),("rock","burden","concrete"),
    # past_tense (bit 6)
    ("run","ran","past_tense"),("walk","walked","past_tense"),("jump","jumped","past_tense"),
    ("fly","flew","past_tense"),("eat","ate","past_tense"),("build","built","past_tense"),
    ("write","wrote","past_tense"),("break","broke","past_tense"),
    # antonym (bit 7)
    ("hot","cold","antonym"),("big","small","antonym"),("fast","slow","antonym"),
    ("hard","soft","antonym"),("happy","sad","antonym"),("strong","weak","antonym"),
    ("good","bad","antonym"),("old","new","antonym"),
    # passive (bit 8)
    ("cat","mouse","passive"),("dog","man","passive"),
    # negation (bit 11)
    ("fast","slow","negation"),("good","bad","negation"),
    ("strong","weak","negation"),("hot","cold","negation"),
]

# T2 sentence pairs (same as before)
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
logits_list = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        for L in ALL_LAYERS:
            hs_by_layer[L].append(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
        logits_list.append(out.logits[0, pos, :].numpy().astype(np.float32))
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

def hamming(a, b): return sum(x != y for x, y in zip(a, b))
def nonaxis_hamming(addr_a, addr_b, skip_bit):
    return sum(x != y for k, (x, y) in enumerate(zip(addr_a, addr_b)) if k != skip_bit)

FLIP_MAP = {"H": "L", "L": "H", "U": "H"}

# ── Main analysis ──────────────────────────────────────────────────────────────
print("=" * 72)
print("Semantic Parallelism vs Traversal Rank")
print("=" * 72)

data = []
for src, tgt, axis in ALL_GT:
    if src not in word_idx or tgt not in word_idx:
        continue
    si = word_idx[src]; ti = word_idx[tgt]
    axis_bit = AXIS_NAMES_12.index(axis)
    addr_s = addresses[si]; addr_t = addresses[ti]
    src_b  = addr_s[axis_bit]; tgt_b  = addr_t[axis_bit]
    bit_sep = (src_b != tgt_b)

    # Non-axis Hamming (how similar are they on 11/12 dimensions)
    na_ham = nonaxis_hamming(addr_s, addr_t, axis_bit)

    # Traversal rank
    fl = list(addr_s); fl[axis_bit] = FLIP_MAP[fl[axis_bit]]; fl = "".join(fl)
    ranked = sorted([(j, hamming(fl, addresses[j])) for j in range(N) if j != si],
                     key=lambda x: x[1])
    rank = next((k for k, (j, _) in enumerate(ranked) if j == ti), -1)
    top5 = [valid_words[j] for j, _ in ranked[:5]]

    data.append({
        "src": src, "tgt": tgt, "axis": axis,
        "bit_sep": bit_sep, "src_bit": src_b, "tgt_bit": tgt_b,
        "na_hamming": na_ham, "rank": rank,
        "top5": top5,
    })

# ── Print table sorted by non-axis Hamming ────────────────────────────────────
print(f"\n  {'src':>10} {'tgt':>10} {'axis':>15}  sep  na_ham  rank")
print(f"  {'-'*65}")
sorted_data = sorted(data, key=lambda d: d["na_hamming"])
for d in sorted_data:
    flag = "✓" if d["rank"] >= 0 and d["rank"] < 5 else "✗"
    rank_str = str(d["rank"]) if d["rank"] >= 0 else "miss"
    print(f"  {d['src']:>10} {d['tgt']:>10} {d['axis']:>15}  "
          f"{'Y' if d['bit_sep'] else 'N'}    "
          f"{d['na_hamming']:>5}  {rank_str:>6} {flag}")

# ── Statistics ────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Statistical Analysis")
print("=" * 72)

valid = [d for d in data if d["rank"] >= 0]
na_vals = [d["na_hamming"] for d in valid]
ranks   = [d["rank"] for d in valid]
log_ranks = [math.log(r + 1) for r in ranks]

if HAS_SCIPY and len(valid) > 3:
    rho, pval = spearmanr(na_vals, log_ranks)
    print(f"\n  Spearman ρ (na_hamming vs log(rank+1)): {rho:.3f}  p={pval:.4f}")
    print(f"  N pairs with rank found: {len(valid)}")
else:
    # Manual Spearman
    def manual_spearman(x, y):
        n = len(x)
        rank_x = sorted(range(n), key=lambda i: x[i])
        rank_y = sorted(range(n), key=lambda i: y[i])
        rx = [0]*n; ry = [0]*n
        for r, i in enumerate(rank_x): rx[i] = r
        for r, i in enumerate(rank_y): ry[i] = r
        d2 = sum((rx[i]-ry[i])**2 for i in range(n))
        rho = 1 - 6*d2 / (n*(n**2-1))
        return rho
    rho = manual_spearman(na_vals, log_ranks)
    print(f"\n  Spearman ρ (na_hamming vs log(rank+1)): {rho:.3f}")
    print(f"  N pairs with rank found: {len(valid)}")

# Bin by na_hamming
print(f"\n  Non-axis Hamming  mean_rank  median_rank  n_pairs")
bins = defaultdict(list)
for d in data:
    if d["rank"] >= 0:
        bins[d["na_hamming"]].append(d["rank"])
    elif d["rank"] == -1:
        bins[d["na_hamming"]].append(N)  # treat miss as N
for h in sorted(bins):
    vals = bins[h]
    print(f"  {h:>16}  {np.mean(vals):>9.1f}  {np.median(vals):>11.1f}  {len(vals):>7}")

# ── Best navigable pairs (lowest na_hamming AND bit separated) ────────────────
print()
print("=" * 72)
print("Best candidates (low na_hamming AND bit_sep=True)")
print("=" * 72)
candidates = [d for d in data if d["bit_sep"] and d["na_hamming"] <= 2]
candidates.sort(key=lambda d: (d["na_hamming"], d["rank"] if d["rank"] >= 0 else N))
for d in candidates[:20]:
    rank_str = str(d["rank"]) if d["rank"] >= 0 else "miss"
    flag = "✓" if d["rank"] >= 0 and d["rank"] < 5 else "✗"
    print(f"  {d['src']:>10}→{d['tgt']:<10}  axis={d['axis']:>15}  "
          f"na_ham={d['na_hamming']}  rank={rank_str:>6} {flag}")

# ── All pairs where na_hamming=0 (perfect parallelism) ───────────────────────
print()
print("=" * 72)
print("Pairs with na_hamming=0 (perfect semantic parallelism)")
print("=" * 72)
perfect = [d for d in data if d["na_hamming"] == 0]
if perfect:
    for d in sorted(perfect, key=lambda d: d["rank"] if d["rank"] >= 0 else N):
        rank_str = str(d["rank"]) if d["rank"] >= 0 else "miss"
        flag = "✓" if d["rank"] >= 0 and d["rank"] < 5 else "✗"
        print(f"  {d['src']:>10}→{d['tgt']:<10}  axis={d['axis']:>15}  "
              f"bit_sep={'Y' if d['bit_sep'] else 'N'}  rank={rank_str:>6} {flag}")
else:
    print("  None found in current GT set")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 98 Summary")
print("=" * 72)
sep_data    = [d for d in data if d["bit_sep"]]
nonsep_data = [d for d in data if not d["bit_sep"]]
low_ham  = [d for d in data if d["na_hamming"] <= 2 and d["rank"] >= 0]
high_ham = [d for d in data if d["na_hamming"] >= 4 and d["rank"] >= 0]

print(f"""
  Total GT pairs evaluated:  {len(data)}
  Pairs with bit separation: {len(sep_data)} ({100*len(sep_data)/max(1,len(data)):.0f}%)
  Pairs found (rank≥0):      {len(valid)}
  Pairs in top-5 (rank<5):   {sum(1 for d in data if 0 <= d.get('rank',-1) < 5)}

  Non-axis Hamming distribution:
  na_hamming=0:  {sum(1 for d in data if d['na_hamming']==0)} pairs
  na_hamming=1:  {sum(1 for d in data if d['na_hamming']==1)} pairs
  na_hamming=2:  {sum(1 for d in data if d['na_hamming']==2)} pairs
  na_hamming≥3:  {sum(1 for d in data if d['na_hamming']>=3)} pairs

  Mean rank for na_ham<=2: {f"{np.mean([d['rank'] for d in low_ham]):.1f}" if low_ham else 'N/A'}
  Mean rank for na_ham>=4: {f"{np.mean([d['rank'] for d in high_ham]):.1f}" if high_ham else 'N/A'}

  Spearman ρ: {rho:.3f}
  {'STRONG negative correlation CONFIRMED' if rho < -0.5 else
   'MODERATE negative correlation' if rho < -0.3 else
   'WEAK or no correlation — hypothesis not supported'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"data": data, "spearman_rho": rho, "n_valid": len(valid)},
              f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 98 complete.")
