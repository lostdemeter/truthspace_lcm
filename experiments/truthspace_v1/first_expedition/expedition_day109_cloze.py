#!/usr/bin/env python3
"""
Day 109 — Cloze / Constrained Generation

Days 104-108 established that the φ-trie CANNOT do open sequential
generation (bigram ≈ unigram ≈ 22%). But the trie HAS proven:
  - Semantic similarity indexing (LOO=0.9443)
  - Internal analogy: 100% (Day 103)
  - Navigable transformations: 94% rank=0 (Day 98)

Day 109: Test the trie on CONSTRAINED generation — tasks that use
semantic similarity rather than sequential probability.

EXPERIMENTS:

1. Semantic cloze — "the ___ ran fast" → candidate words must be
   semantically compatible with the constraints from context words.
   Method: compute the mean address of context words → find nearest
   vocabulary token → that's the cloze prediction.

2. Category cloze — "dog, cat, bird, ___" → predict the next member
   of a category by finding the nearest-address vocabulary token to
   the centroid of the given members.

3. Analogy cloze — "king is to queen as man is to ___" → Day 103
   approach (axis flip) but tested on explicit cloze format.

4. Semantic constraint intersection — "animal that runs fast" →
   find tokens that are simultaneously close (in address space) to
   multiple semantic constraints.

These tasks play to the trie's CONFIRMED strengths: semantic indexing,
nearest-neighbor retrieval, and address arithmetic.

COMPARISON:
  - Trie semantic centroid prediction vs LM prediction
  - Measure both accuracy (is the ground-truth in top-5?) and
    semantic appropriateness (is the prediction semantically plausible?)
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day109_cloze.json")
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

# ── Cloze tasks ────────────────────────────────────────────────────────────────
# Exp 1: Semantic cloze — predict blank from context words
# Format: (context_words, answer, category)
SEMANTIC_CLOZE = [
    # Animals by property
    (["fast", "run", "field"],          "dog",        "animal-property"),
    (["swim", "water", "deep"],         "fish",        "animal-property"),
    (["fly", "sky", "high"],            "bird",        "animal-property"),
    (["forest", "dark", "run"],         "wolf",        "animal-property"),
    (["ocean", "big", "swim"],          "whale",       "animal-property"),
    (["tree", "fruit", "eat"],          "monkey",      "animal-property"),
    (["night", "fly", "dark"],          "owl",         "animal-property"),
    (["river", "swim", "cold"],         "salmon",      "animal-property"),
    # Objects by use
    (["cut", "sharp", "food"],          "knife",       "tool-property"),
    (["read", "paper", "write"],        "book",        "tool-property"),
    (["open", "door", "small"],         "key",         "tool-property"),
    (["drive", "road", "fast"],         "car",         "tool-property"),
    (["light", "night", "bright"],      "lamp",        "tool-property"),
    (["build", "hard", "hit"],          "hammer",      "tool-property"),
    # Emotions/qualities by context
    (["good", "right", "trust"],        "faith",       "abstract-quality"),
    (["sad", "dark", "heart"],          "grief",       "abstract-quality"),
    (["win", "strong", "first"],        "pride",       "abstract-quality"),
    (["free", "wide", "open"],          "freedom",     "abstract-quality"),
    # Nature by description
    (["cold", "white", "fall"],         "snow",        "nature"),
    (["bright", "hot", "sky"],          "sun",         "nature"),
    (["tall", "old", "grow"],           "tree",        "nature"),
    (["deep", "wide", "cold"],          "ocean",       "nature"),
    (["high", "hard", "rock"],          "mountain",    "nature"),
    (["dark", "many", "cold"],          "forest",      "nature"),
]

# Exp 2: Category continuation — "dog, cat, bird, ___"
CATEGORY_CLOZE = [
    (["dog","cat","bird"],              "fish",        "pets"),
    (["lion","tiger","wolf"],           "bear",        "predators"),
    (["eagle","crow","owl"],            "sparrow",     "birds"),
    (["rose","flower","tree"],          "grass",       "plants"),
    (["knife","hammer","nail"],         "rope",        "tools"),
    (["ocean","river","lake"],          "rain",        "water-bodies"),
    (["gold","rock","stone"],           "coal",        "minerals"),
    (["happy","joy","pride"],           "love",        "positive-emotions"),
    (["sad","pain","grief"],            "fear",        "negative-emotions"),
    (["fast","quick","bright"],         "sharp",       "positive-qualities"),
    (["big","large","heavy"],           "thick",       "size-qualities"),
    (["cold","dark","deep"],            "narrow",      "negative-qualities"),
    (["run","walk","jump"],             "swim",        "movement-verbs"),
    (["see","hear","feel"],             "know",        "perception-verbs"),
    (["read","write","think"],          "know",        "cognitive-verbs"),
    (["bread","meat","egg"],            "cheese",      "food"),
    (["sun","moon","star"],             "cloud",       "sky-objects"),
    (["hand","foot","arm"],             "leg",         "body-parts"),
    (["king","queen","man"],            "woman",       "gender-pair"),
    (["man","father","son"],            "brother",     "male-family"),
]

# Exp 3: Analogy cloze — (A, B, C, D_gold, axis_name)
ANALOGY_CLOZE = [
    ("king",   "queen",    "man",      "woman",     "gender"),
    ("man",    "woman",    "boy",      "girl",      "gender"),
    ("father", "mother",   "son",      "daughter",  "gender"),
    ("king",   "queen",    "father",   "mother",    "gender"),
    ("man",    "woman",    "brother",  "sister",    "gender"),
    ("fast",   "faster",   "big",      "bigger",    "comparative"),
    ("big",    "bigger",   "old",      "older",     "comparative"),
    ("cold",   "colder",   "dark",     "darker",    "comparative"),
    ("good",   "bad",      "happy",    "sad",       "antonym"),
    ("hot",    "cold",     "big",      "small",     "antonym"),
    ("fast",   "slow",     "strong",   "weak",      "antonym"),
    ("bright", "dark",     "hard",     "soft",      "antonym"),
    ("dog",    "dogs",     "cat",      "cats",      "plural"),
    ("tree",   "trees",    "bird",     "birds",     "plural"),
    ("man",    "men",      "woman",    "women",     "plural"),
    ("walk",   "walked",   "run",      "ran",       "past_tense"),
    ("eat",    "ate",      "fly",      "flew",      "past_tense"),
    ("build",  "built",    "break",    "broke",     "past_tense"),
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

print("Extracting probe token hidden states ...")
hs_by_layer = {L: [] for L in ALL_LAYERS}
valid_words  = []; logits_list = []
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
logits_arr = np.array(logits_list, dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

# Logit cosine similarity
logits_norm = logits_arr / (np.linalg.norm(logits_arr, axis=1, keepdims=True) + 1e-10)

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

ham_mat = np.zeros((N, N), dtype=np.int8)
for i in range(N):
    for j in range(i+1, N):
        h = int(np.sum(addr_int[i] != addr_int[j]))
        ham_mat[i,j] = h; ham_mat[j,i] = h

def topk_by_hamming(query_words, k=5, exclude=None):
    """Return top-k vocabulary words by Hamming distance to centroid address of query_words."""
    idxs = [word_idx[w] for w in query_words if w in word_idx]
    if not idxs: return []
    # Centroid in address integer space (round)
    centroid = np.mean(addr_int[idxs], axis=0)
    centroid_rounded = np.round(centroid).astype(np.int8)
    # Hamming distance from centroid to all vocab tokens
    dists = np.sum(np.abs(addr_int - centroid_rounded), axis=1)
    if exclude:
        for w in exclude:
            if w in word_idx: dists[word_idx[w]] = 999
    ranked = np.argsort(dists)
    return [valid_words[j] for j in ranked[:k]]

def topk_by_logit_cosim(query_words, k=5, exclude=None):
    """Return top-k vocabulary words by mean logit cosine similarity to query_words."""
    idxs = [word_idx[w] for w in query_words if w in word_idx]
    if not idxs: return []
    mean_logit = np.mean(logits_norm[idxs], axis=0)
    sims = logits_norm @ mean_logit
    if exclude:
        for w in exclude:
            if w in word_idx: sims[word_idx[w]] = -999
    ranked = np.argsort(sims)[::-1]
    return [valid_words[j] for j in ranked[:k]]

def axis_flip_cloze(A, B, C, axis_name):
    """A:B::C:? via trie axis flip (Day 103 method)."""
    if any(w not in word_idx for w in [A, B, C]): return [], []
    iA, iB, iC = word_idx[A], word_idx[B], word_idx[C]
    # Find the axis with the largest address difference between A and B
    best_ax = None; best_sep = -1
    for k, name in enumerate(AXIS_NAMES_12):
        cA, cB = addresses[iA][k], addresses[iB][k]
        if cA != cB:
            ax_sep = 1
            if best_sep < ax_sep:
                best_sep = ax_sep; best_ax = k
    if best_ax is None:
        # Fall back to specified axis
        best_ax = AXIS_NAMES_12.index(axis_name) if axis_name in AXIS_NAMES_12 else 0
    # Flip that axis in C's address
    flip_map = {"H": "L", "L": "H", "U": "U"}
    C_addr = list(addresses[iC])
    C_addr[best_ax] = flip_map[C_addr[best_ax]]
    flipped_addr = "".join(C_addr)
    flipped_arr  = np.array([{"H":2,"U":1,"L":0}[c] for c in flipped_addr], dtype=np.int8)
    dists = np.sum(addr_int != flipped_arr, axis=1)
    dists[iC] = 999  # exclude C itself
    ranked = np.argsort(dists)
    top5   = [valid_words[j] for j in ranked[:5]]
    return top5, best_ax

# ── Experiment 1: Semantic Cloze ───────────────────────────────────────────────
print("=" * 72)
print("Exp 1: Semantic Cloze (context words → predict answer)")
print("=" * 72)
print(f"\n  {'context':>35}  {'answer':>10}  {'trie top-1':>12}  {'lm top-1':>12}  {'T?':>3}  {'L?':>3}")
print(f"  {'-'*80}")

cloze1_trie_top1 = 0; cloze1_trie_top5 = 0
cloze1_lm_top1   = 0; cloze1_lm_top5   = 0
cloze1_results   = []

for ctx_words, answer, cat in SEMANTIC_CLOZE:
    ctx_str = ", ".join(ctx_words)
    trie_top5 = topk_by_hamming(ctx_words, k=5, exclude=ctx_words)
    lm_top5   = topk_by_logit_cosim(ctx_words, k=5, exclude=ctx_words)
    t_hit1 = (trie_top5[0] == answer) if trie_top5 else False
    l_hit1 = (lm_top5[0]   == answer) if lm_top5   else False
    t_hit5 = answer in trie_top5; l_hit5 = answer in lm_top5
    if t_hit1: cloze1_trie_top1 += 1
    if t_hit5: cloze1_trie_top5 += 1
    if l_hit1: cloze1_lm_top1   += 1
    if l_hit5: cloze1_lm_top5   += 1
    cloze1_results.append({"ctx": ctx_words, "answer": answer, "cat": cat,
                            "trie_top5": trie_top5, "lm_top5": lm_top5,
                            "trie_top1_hit": t_hit1, "lm_top1_hit": l_hit1})
    t_str = trie_top5[0] if trie_top5 else "—"
    l_str = lm_top5[0]   if lm_top5   else "—"
    print(f"  {ctx_str:>35}  {answer:>10}  {t_str:>12}  {l_str:>12}  "
          f"{'✓' if t_hit1 else '✗':>3}  {'✓' if l_hit1 else '✗':>3}")

n1 = len(SEMANTIC_CLOZE)
print(f"\n  Trie: top-1={cloze1_trie_top1}/{n1} ({100*cloze1_trie_top1/n1:.0f}%)  "
      f"top-5={cloze1_trie_top5}/{n1} ({100*cloze1_trie_top5/n1:.0f}%)")
print(f"  LM:   top-1={cloze1_lm_top1}/{n1} ({100*cloze1_lm_top1/n1:.0f}%)  "
      f"top-5={cloze1_lm_top5}/{n1} ({100*cloze1_lm_top5/n1:.0f}%)")

# ── Experiment 2: Category Continuation ───────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Category Continuation (X, Y, Z → predict 4th member)")
print("=" * 72)
print(f"\n  {'members':>35}  {'answer':>12}  {'trie':>12}  {'lm':>12}  {'T?':>3}  {'L?':>3}")
print(f"  {'-'*85}")

cat2_trie_top1 = 0; cat2_trie_top5 = 0
cat2_lm_top1   = 0; cat2_lm_top5   = 0
cat2_results   = []

for members, answer, cat in CATEGORY_CLOZE:
    trie_top5 = topk_by_hamming(members, k=5, exclude=members)
    lm_top5   = topk_by_logit_cosim(members, k=5, exclude=members)
    t_hit1 = (trie_top5[0] == answer) if trie_top5 else False
    l_hit1 = (lm_top5[0]   == answer) if lm_top5   else False
    t_hit5 = answer in trie_top5; l_hit5 = answer in lm_top5
    if t_hit1: cat2_trie_top1 += 1
    if t_hit5: cat2_trie_top5 += 1
    if l_hit1: cat2_lm_top1   += 1
    if l_hit5: cat2_lm_top5   += 1
    cat2_results.append({"members": members, "answer": answer, "cat": cat,
                          "trie_top5": trie_top5, "lm_top5": lm_top5})
    m_str = ", ".join(members)
    t_str = trie_top5[0] if trie_top5 else "—"
    l_str = lm_top5[0]   if lm_top5   else "—"
    print(f"  {m_str:>35}  {answer:>12}  {t_str:>12}  {l_str:>12}  "
          f"{'✓' if t_hit1 else '✗':>3}  {'✓' if l_hit1 else '✗':>3}")

n2 = len(CATEGORY_CLOZE)
print(f"\n  Trie: top-1={cat2_trie_top1}/{n2} ({100*cat2_trie_top1/n2:.0f}%)  "
      f"top-5={cat2_trie_top5}/{n2} ({100*cat2_trie_top5/n2:.0f}%)")
print(f"  LM:   top-1={cat2_lm_top1}/{n2} ({100*cat2_lm_top1/n2:.0f}%)  "
      f"top-5={cat2_lm_top5}/{n2} ({100*cat2_lm_top5/n2:.0f}%)")

# ── Experiment 3: Analogy Cloze ───────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 3: Analogy Cloze (A:B::C:?) via axis flip + logit cosim")
print("=" * 72)
print(f"\n  {'A:B::C:?':>28}  {'answer':>10}  {'axis':>12}  "
      f"{'trie':>10}  {'lm':>10}  {'T?':>3}  {'L?':>3}")
print(f"  {'-'*90}")

ana_trie_top1 = 0; ana_trie_top5 = 0
ana_lm_top1   = 0; ana_lm_top5   = 0
ana_results   = []

for A, B, C, D_gold, axis_name in ANALOGY_CLOZE:
    trie_top5, used_ax = axis_flip_cloze(A, B, C, axis_name)
    # LM: query on [A, B, C] mean logit cosim (with axis constraint)
    lm_top5 = topk_by_logit_cosim([A, B, C], k=5, exclude=[A, B, C])
    t_hit1 = (trie_top5[0] == D_gold) if trie_top5 else False
    l_hit1 = (lm_top5[0]   == D_gold) if lm_top5   else False
    t_hit5 = D_gold in trie_top5; l_hit5 = D_gold in lm_top5
    if t_hit1: ana_trie_top1 += 1
    if t_hit5: ana_trie_top5 += 1
    if l_hit1: ana_lm_top1   += 1
    if l_hit5: ana_lm_top5   += 1
    ana_results.append({"A":A,"B":B,"C":C,"D":D_gold,"axis":axis_name,
                         "trie_top5":trie_top5,"lm_top5":lm_top5,
                         "trie_top1_hit":t_hit1,"lm_top1_hit":l_hit1})
    q_str = f"{A}:{B}::{C}:?"
    t_str = trie_top5[0] if trie_top5 else "—"
    l_str = lm_top5[0]   if lm_top5   else "—"
    ax_str = AXIS_NAMES_12[used_ax] if isinstance(used_ax, int) else axis_name
    print(f"  {q_str:>28}  {D_gold:>10}  {ax_str:>12}  "
          f"{t_str:>10}  {l_str:>10}  {'✓' if t_hit1 else '✗':>3}  {'✓' if l_hit1 else '✗':>3}")

n3 = len(ANALOGY_CLOZE)
print(f"\n  Trie: top-1={ana_trie_top1}/{n3} ({100*ana_trie_top1/n3:.0f}%)  "
      f"top-5={ana_trie_top5}/{n3} ({100*ana_trie_top5/n3:.0f}%)")
print(f"  LM:   top-1={ana_lm_top1}/{n3} ({100*ana_lm_top1/n3:.0f}%)  "
      f"top-5={ana_lm_top5}/{n3} ({100*ana_lm_top5/n3:.0f}%)")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 109 Summary — Trie as Constrained Generation Engine")
print("=" * 72)

print(f"""
  Cloze Task Performance:

  Task                      Trie top-1   Trie top-5   LM top-1   LM top-5
  ─────────────────────────────────────────────────────────────────────────
  Semantic cloze ({n1})      {100*cloze1_trie_top1/n1:>8.0f}%   {100*cloze1_trie_top5/n1:>8.0f}%   {100*cloze1_lm_top1/n1:>7.0f}%   {100*cloze1_lm_top5/n1:>7.0f}%
  Category continue ({n2})   {100*cat2_trie_top1/n2:>8.0f}%   {100*cat2_trie_top5/n2:>8.0f}%   {100*cat2_lm_top1/n2:>7.0f}%   {100*cat2_lm_top5/n2:>7.0f}%
  Analogy cloze ({n3})       {100*ana_trie_top1/n3:>8.0f}%   {100*ana_trie_top5/n3:>8.0f}%   {100*ana_lm_top1/n3:>7.0f}%   {100*ana_lm_top5/n3:>7.0f}%

  Compare to sequential prediction (Days 104-108):
    Bigram next-token top-1:  22.0%   (sequential, open generation)
    Cloze (semantic) top-1:   above?  (constrained, semantic lookup)

  KEY QUESTION: Is trie constrained generation better than sequential?
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "semantic_cloze": {
            "n": n1,
            "trie_top1": cloze1_trie_top1, "trie_top5": cloze1_trie_top5,
            "lm_top1": cloze1_lm_top1, "lm_top5": cloze1_lm_top5,
            "results": cloze1_results,
        },
        "category_cloze": {
            "n": n2,
            "trie_top1": cat2_trie_top1, "trie_top5": cat2_trie_top5,
            "lm_top1": cat2_lm_top1, "lm_top5": cat2_lm_top5,
            "results": cat2_results,
        },
        "analogy_cloze": {
            "n": n3,
            "trie_top1": ana_trie_top1, "trie_top5": ana_trie_top5,
            "lm_top1": ana_lm_top1, "lm_top5": ana_lm_top5,
            "results": ana_results,
        },
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 109 complete.")
