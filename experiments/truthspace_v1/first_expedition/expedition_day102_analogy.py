#!/usr/bin/env python3
"""
Day 102 — Trie-Based Word Analogy Solver

If the φ-trie is a coordinate system, it should solve word analogy problems:
  "A is to B as C is to ?" → flip the axis separating A from B in C's address

ALGORITHM:
  1. Given (A, B, C), find the axis k where addr(A)[k] ≠ addr(B)[k]
     and na_hamming(A, B, k) = 0 (A and B are perfectly parallel on 11/12 axes)
  2. Flip bit k in addr(C)
  3. Find nearest token D to the flipped address
  4. Return D as the predicted answer

ANALOGY TEST CASES:
  Human-intuitive analogies (from standard word analogy datasets):
    king:queen :: man:woman (gender)
    king:queen :: brother:sister (gender)
    fast:faster :: slow:slower (comparative)
    fast:faster :: big:bigger (comparative)
    hot:cold :: happy:sad (antonym)
    hot:cold :: strong:weak (antonym)
    good:better :: bad:worse (comparative)
    most:least :: many:few (scalar-polarity — what the gender axis captures?)

  Trie-discovered analogies (from Day 99 navigability graph — pairs with na_ham=0):
    Using na_ham=0 confirmed pairs as the A:B template

EVALUATION:
  - Exact match: predicted D = expected answer
  - Top-5: expected answer in top-5 nearest tokens to flipped address
  - Rank: rank of expected answer

PREDICTION:
  - Human-intuitive analogies: should work for pairs with low na_hamming
  - Novel analogies (trie-discovered): the trie should complete them by
    following the same geometric logic
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day102_analogy.json")
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

# Human word analogy test cases: (A, B, C, expected_D, note)
ANALOGY_TESTS = [
    # Standard gender analogies
    ("king",    "queen",   "man",      "woman",    "king:queen::man:?"),
    ("king",    "queen",   "brother",  "sister",   "king:queen::brother:?"),
    ("king",    "queen",   "father",   "mother",   "king:queen::father:?"),
    ("king",    "queen",   "son",      "daughter", "king:queen::son:?"),
    ("king",    "queen",   "prince",   "princess", "king:queen::prince:?"),
    ("king",    "queen",   "actor",    "actress",  "king:queen::actor:?"),
    ("king",    "queen",   "husband",  "wife",     "king:queen::husband:?"),
    ("man",     "woman",   "boy",      "girl",     "man:woman::boy:?"),
    ("brother", "sister",  "son",      "daughter", "brother:sister::son:?"),
    # Comparative analogies
    ("good",    "better",  "bad",      "worse",    "good:better::bad:?"),
    ("fast",    "faster",  "slow",     "slower",   "fast:faster::slow:?"),
    ("big",     "bigger",  "small",    "smaller",  "big:bigger::small:?"),
    ("good",    "better",  "fast",     "faster",   "good:better::fast:?"),
    # Scalar-polarity pairs (Day 100: gender axis captures these)
    ("most",    "least",   "many",     "few",      "most:least::many:?"),
    ("most",    "least",   "more",     "less",     "most:least::more:?"),
    ("best",    "worst",   "most",     "least",    "best:worst::most:?"),
    # Locomotion cluster
    ("run",     "walk",    "fly",      "swim",     "run:walk::fly:?"),
    # Degradation cluster
    ("dirty",   "old",     "loud",     "quiet",    "dirty:old::loud:?"),
    # Novel: trie-discovered pairs as templates
    ("mouse",   "monkey",  "spider",   "frog",     "mouse:monkey::spider:?"),
    ("least",   "most",    "few",      "many",     "least:most::few:?"),
]

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

def nonaxis_hamming(ia, ib, skip):
    return int(np.sum(addr_int[ia] != addr_int[ib])) - int(addr_int[ia][skip] != addr_int[ib][skip])

def find_axis(ia, ib):
    """Find axis where addr[A] ≠ addr[B] and na_hamming is minimal."""
    diffs = [(k, int(addr_int[ia][k] != addr_int[ib][k]),
              nonaxis_hamming(ia, ib, k))
             for k in range(12)]
    # Axes where bits differ
    candidates = [(k, na) for k, d, na in diffs if d == 1]
    if not candidates: return None, None
    # Pick axis with lowest na_hamming
    best_k, best_na = min(candidates, key=lambda x: x[1])
    return best_k, best_na

def solve_analogy(A, B, C):
    """A:B :: C:? → return (predicted_D, axis_used, na_hamming_AB, rank_D_in_top)"""
    if A not in word_idx or B not in word_idx or C not in word_idx:
        return None, None, None, None, []
    ia = word_idx[A]; ib = word_idx[B]; ic = word_idx[C]
    # Step 1: identify the axis separating A from B
    best_k, na_AB = find_axis(ia, ib)
    if best_k is None:
        return None, None, None, None, []
    # Step 2: flip that axis in C's address
    fl = list(addresses[ic])
    fl[best_k] = FLIP_MAP[fl[best_k]]
    flipped = "".join(fl)
    # Step 3: find nearest token to flipped address (excluding C itself)
    ranked = sorted([(j, hamming_str(flipped, addresses[j]))
                     for j in range(N) if j != ic], key=lambda x: x[1])
    predicted = valid_words[ranked[0][0]]
    top5      = [valid_words[j] for j, _ in ranked[:5]]
    return predicted, AXIS_NAMES_12[best_k], na_AB, ranked[0][1], top5

def rank_of(word, A_idx, flipped_addr):
    if word not in word_idx: return -1
    ti = word_idx[word]
    ranked = sorted([(j, hamming_str(flipped_addr, addresses[j]))
                     for j in range(N) if j != A_idx], key=lambda x: x[1])
    return next((r for r, (j, _) in enumerate(ranked) if j == ti), -1)

# ── Run analogy tests ─────────────────────────────────────────────────────────
print("=" * 72)
print("Word Analogy Tests")
print("=" * 72)
print(f"\n  {'Analogy':>35}  {'axis':>15}  na_AB  predicted  expected  rank")
print(f"  {'-'*90}")

results = []
correct_exact = 0; correct_top5 = 0; total_valid = 0

for A, B, C, expected, note in ANALOGY_TESTS:
    pred, axis, na_AB, pred_ham, top5 = solve_analogy(A, B, C)
    if pred is None:
        print(f"  {note:>35}  (missing token)")
        continue
    # Rank of expected in the top list
    if C in word_idx:
        ic = word_idx[C]
        fl = list(addresses[ic]); fl[AXIS_NAMES_12.index(axis)] = FLIP_MAP[fl[AXIS_NAMES_12.index(axis)]]
        flipped = "".join(fl)
        exp_rank = rank_of(expected, ic, flipped) if expected in word_idx else -1
    else:
        exp_rank = -1
    exact = (pred == expected)
    top5_hit = (expected in top5)
    if exact: correct_exact += 1
    if top5_hit: correct_top5 += 1
    total_valid += 1
    flag = "✓" if exact else ("~" if top5_hit else "✗")
    rank_str = str(exp_rank) if exp_rank >= 0 else "miss"
    print(f"  {note:>35}  {axis:>15}  {na_AB:>5}  "
          f"{pred:>9}  {expected:>8}  {rank_str:>4} {flag}")
    results.append({
        "analogy": note, "A": A, "B": B, "C": C, "expected": expected,
        "predicted": pred, "axis": axis, "na_AB": na_AB,
        "exact": exact, "top5_hit": top5_hit, "exp_rank": exp_rank,
        "top5": top5,
    })

# ── Analogy accuracy by axis ──────────────────────────────────────────────────
print()
print("=" * 72)
print("Analogy accuracy by axis used")
print("=" * 72)
by_axis = defaultdict(lambda: {"total": 0, "exact": 0, "top5": 0})
for r in results:
    if r["axis"]:
        by_axis[r["axis"]]["total"] += 1
        if r["exact"]: by_axis[r["axis"]]["exact"] += 1
        if r["top5_hit"]: by_axis[r["axis"]]["top5"] += 1
print(f"\n  {'axis':>15}  {'total':>6}  {'exact':>6}  {'top5':>5}")
for axis in sorted(by_axis, key=lambda k: -by_axis[k]["exact"]):
    d = by_axis[axis]
    if d["total"] > 0:
        print(f"  {axis:>15}  {d['total']:>6}  {d['exact']:>6}  {d['top5']:>5}")

# ── Show all top-5 predictions ────────────────────────────────────────────────
print()
print("=" * 72)
print("Top-5 predictions for each analogy")
print("=" * 72)
for r in results:
    flag = "✓" if r["exact"] else ("~" if r["top5_hit"] else "✗")
    print(f"  {flag} {r['analogy']}")
    print(f"    Top-5: {r['top5']}")
    print(f"    Expected: {r['expected']}  (rank={r['exp_rank']})")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 102 Summary")
print("=" * 72)
print(f"""
  Analogy tests run:    {total_valid}
  Exact match:         {correct_exact}/{total_valid} ({100*correct_exact/max(1,total_valid):.0f}%)
  Top-5 match:         {correct_top5}/{total_valid} ({100*correct_top5/max(1,total_valid):.0f}%)

  INTERPRETATION:
  {'→ Trie IS an analogy solver — exact match majority' if correct_exact/max(1,total_valid) >= 0.5 else
   '→ Trie partially solves analogies (top-5 match)' if correct_top5/max(1,total_valid) >= 0.5 else
   '→ Trie does not reliably solve analogies — geometry is not the semantics humans expect'}

  Key insight:
  The trie solves analogies by identifying which T2 axis separates A from B
  (lowest non-axis Hamming), then flipping that axis in C's address.
  This works when (A,B) and (C,D) are geometrically parallel in the trie's
  12D address space — i.e., when human-intuitive semantic pairs happen to
  also be geometrically parallel in the model's representational geometry.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "results": results,
        "exact_pct": 100*correct_exact/max(1,total_valid),
        "top5_pct": 100*correct_top5/max(1,total_valid),
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 102 complete.")
