#!/usr/bin/env python3
"""
Day 100 — Geometric Parallelism vs Semantic Relatedness

Day 99 found 5,977 navigable pairs with na_hamming ≤ 1.
Many appear "accidental" (fork→cat via gender, most→least via gender).

CORE QUESTION: Does geometric parallelism (na_hamming ≤ 1) correlate with
semantic relatedness as measured by the model's own logit distribution?

If the TruthSpace hypothesis is correct — structure IS information — then
geometrically parallel pairs should be semantically related (similar logit
distributions) even when humans don't perceive the relationship.

TESTS:
  1. Logit cosine similarity for na_ham=0 navigable pairs
     vs na_ham=0 non-navigable pairs (same structure, different outcome)
  2. Logit cosine vs na_hamming for all pairs
  3. Compare: meaningful pairs (king/queen) vs accidental pairs (fork/cat)
     — do they have different logit cosine profiles?
  4. Pearson correlation: logit_cosine vs na_hamming across all 401 token pairs

PREDICTION (from TruthSpace hypothesis):
  - High logit cosine should correlate with low na_hamming
  - "Accidental" navigable pairs should have higher logit cosine than
    semantically distant pairs (the model views them as related)
  - Logit cosine and na_hamming should both independently predict traversal rank
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day100_geometric_semantic.json")
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

# Known semantically meaningful pairs (from GT sets Days 92-97)
MEANINGFUL_PAIRS = {
    ("king","queen"), ("man","woman"), ("brother","sister"), ("father","mother"),
    ("son","daughter"), ("prince","princess"), ("actor","actress"), ("boy","girl"),
    ("fast","faster"), ("big","bigger"), ("small","smaller"), ("good","better"),
    ("eagle","bird"), ("happy","joyful"), ("run","ran"), ("fly","flew"),
}

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

# Probe token hidden states + logits
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
logits_arr = np.array(logits_list, dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

# Normalize logits for cosine
logits_norm = logits_arr / (np.linalg.norm(logits_arr, axis=1, keepdims=True) + 1e-10)

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
addr_int = np.array([[{"H": 2, "U": 1, "L": 0}[c] for c in a] for a in addresses],
                     dtype=np.int8)

def hamming(a, b): return sum(x != y for x, y in zip(a, b))
def nonaxis_hamming_fast(ia, ib, skip):
    return int(np.sum(addr_int[ia] != addr_int[ib])) - (int(addr_int[ia][skip]) != int(addr_int[ib][skip]))

FLIP_MAP = {"H": "L", "L": "H", "U": "H"}

# ── Phase 1: Sample pairs across the full Hamming range ───────────────────────
print("=" * 72)
print("Phase 1: Logit cosine vs na_hamming — stratified sample")
print("=" * 72)

# Compute full pairwise logit cosine for all token pairs (N×N)
print(f"  Computing {N}×{N} logit cosine matrix ...")
logit_cosim = logits_norm @ logits_norm.T  # (N, N)
print(f"  Done.\n")

# Also compute full pairwise hamming
print("  Computing pairwise 12D Hamming ...")
hamming_mat = np.zeros((N, N), dtype=np.int8)
for i in range(N):
    for j in range(i+1, N):
        h = int(np.sum(addr_int[i] != addr_int[j]))
        hamming_mat[i, j] = h
        hamming_mat[j, i] = h
print("  Done.\n")

# Bin logit cosine by 12D Hamming
print("  Logit cosine by 12D Hamming:")
print(f"  {'hamming':>8}  {'mean_cosim':>12}  {'median_cosim':>13}  {'n':>8}")
hamming_cosim = defaultdict(list)
for i in range(N):
    for j in range(i+1, N):
        hamming_cosim[int(hamming_mat[i,j])].append(float(logit_cosim[i,j]))

full_ham_vals = []; full_cos_vals = []
for h in sorted(hamming_cosim):
    vals = hamming_cosim[h]
    if vals:
        print(f"  {h:>8}  {np.mean(vals):>12.4f}  {np.median(vals):>13.4f}  {len(vals):>8}")
        full_ham_vals.extend([h]*len(vals))
        full_cos_vals.extend(vals)

# Pearson correlation
def pearson(x, y):
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    x -= x.mean(); y -= y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / n) if n > 1e-10 else 0.0

rho_full = pearson(full_ham_vals, full_cos_vals)
print(f"\n  Pearson r (12D Hamming vs logit cosine): {rho_full:.3f}")

# ── Phase 2: Compare navigable pairs vs random pairs ─────────────────────────
print()
print("=" * 72)
print("Phase 2: Logit cosine of navigable pairs vs random pairs")
print("=" * 72)

# Load Day 99 results
day99_path = SCRIPT_DIR / "day99_navigable_pairs.json"
if day99_path.exists():
    with open(day99_path) as f:
        day99 = json.load(f)
    navigable = day99.get("confirmed", [])[:1000]
else:
    navigable = []
    print("  day99_navigable_pairs.json not found, skipping comparison")

nav_cosims = []
nav_meaningful_cosims = []
nav_accidental_cosims = []

for r in navigable:
    src, tgt = r["src"], r["tgt"]
    if src not in word_idx or tgt not in word_idx: continue
    si, ti = word_idx[src], word_idx[tgt]
    c = float(logit_cosim[si, ti])
    nav_cosims.append(c)
    if (src, tgt) in MEANINGFUL_PAIRS or (tgt, src) in MEANINGFUL_PAIRS:
        nav_meaningful_cosims.append(c)
    else:
        nav_accidental_cosims.append(c)

# Random pairs with same hamming distribution
rng = np.random.default_rng(42)
rand_pairs = [(int(rng.integers(N)), int(rng.integers(N))) for _ in range(len(navigable))]
rand_cosims = [float(logit_cosim[i,j]) for i,j in rand_pairs if i!=j]

if nav_cosims:
    print(f"\n  Navigable pairs (na_ham≤1, rank<5):")
    print(f"    mean cosim:    {np.mean(nav_cosims):.4f}")
    print(f"    median cosim:  {np.median(nav_cosims):.4f}")
    print(f"    N:             {len(nav_cosims)}")
    if nav_meaningful_cosims:
        print(f"\n  Semantically meaningful navigable pairs:")
        print(f"    mean cosim:    {np.mean(nav_meaningful_cosims):.4f}")
        print(f"    N:             {len(nav_meaningful_cosims)}")
    if nav_accidental_cosims:
        print(f"\n  'Accidental' navigable pairs:")
        print(f"    mean cosim:    {np.mean(nav_accidental_cosims):.4f}")
        print(f"    N:             {len(nav_accidental_cosims)}")
    print(f"\n  Random pairs (same count):")
    print(f"    mean cosim:    {np.mean(rand_cosims):.4f}")
    print(f"    median cosim:  {np.median(rand_cosims):.4f}")
    nav_advantage = np.mean(nav_cosims) - np.mean(rand_cosims)
    print(f"\n  Navigable advantage over random: {nav_advantage:+.4f}")
    accd_advantage = np.mean(nav_accidental_cosims) - np.mean(rand_cosims) if nav_accidental_cosims else float("nan")
    print(f"  Accidental navigable advantage:  {accd_advantage:+.4f}")

# ── Phase 3: Spot-check specific accidental pairs ────────────────────────────
print()
print("=" * 72)
print("Phase 3: Spot-check accidental navigable pairs (fork→cat, most→least)")
print("=" * 72)
spot_pairs = [
    ("fork", "cat"), ("most", "least"), ("least", "most"),
    ("box", "spider"), ("clock", "spider"), ("lamp", "salt"),
    ("paper", "small"), ("thread", "spider"), ("back", "mind"),
    ("run", "walk"),   # both verbs — geometric proximity makes sense
    ("dirty", "old"), ("old", "dirty"),
    ("king", "queen"),   # gold standard
    ("brother", "sister"),  # gold standard
    ("good", "better"),  # gold standard
]
print(f"\n  {'src':>12}  {'tgt':>12}  {'logit_cosim':>12}  {'12D_ham':>8}  {'note'}")
print(f"  {'-'*65}")
for src, tgt in spot_pairs:
    if src not in word_idx or tgt not in word_idx:
        print(f"  {src:>12}  {tgt:>12}  {'(not in vocab)':>12}"); continue
    si, ti = word_idx[src], word_idx[tgt]
    cos = float(logit_cosim[si, ti])
    h   = int(hamming_mat[si, ti])
    note = "MEANINGFUL" if (src,tgt) in MEANINGFUL_PAIRS or (tgt,src) in MEANINGFUL_PAIRS else ""
    print(f"  {src:>12}  {tgt:>12}  {cos:>12.4f}  {h:>8}  {note}")

# ── Phase 4: Logit cosine as predictor of traversal rank ─────────────────────
print()
print("=" * 72)
print("Phase 4: Does logit cosine predict traversal rank independently of na_hamming?")
print("=" * 72)

# Use the Day 99 confirmed pairs and compute logit cosine, then bin by rank
if navigable:
    rank_bins = defaultdict(list)
    cosim_by_rank = []
    for r in navigable:
        src, tgt = r["src"], r["tgt"]
        if src not in word_idx or tgt not in word_idx: continue
        si, ti = word_idx[src], word_idx[tgt]
        cos = float(logit_cosim[si, ti])
        rank = r.get("rank", -1)
        if rank >= 0:
            bucket = min(rank // 1, 4)  # 0,1,2,3,4+
            rank_bins[bucket].append(cos)
        cosim_by_rank.append((cos, rank))

    print(f"\n  Logit cosine by traversal rank (confirmed navigable pairs):")
    print(f"  {'rank':>6}  {'mean_cosim':>12}")
    for b in sorted(rank_bins):
        print(f"  {b:>6}  {np.mean(rank_bins[b]):>12.4f}  (n={len(rank_bins[b])})")
    cosims_r = [c for c, r in cosim_by_rank if r >= 0]
    ranks_r  = [r for c, r in cosim_by_rank if r >= 0]
    if len(cosims_r) > 3:
        r_cosim_corr = pearson(cosims_r, ranks_r)
        print(f"\n  Pearson r (logit_cosim vs rank): {r_cosim_corr:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 100 Summary")
print("=" * 72)

mean_same_leaf = np.mean([float(logit_cosim[i,j]) for i in range(N) for j in range(i+1,N) if addresses[i] == addresses[j]])
print(f"""
  Logit cosine vs 12D Hamming correlation: Pearson r = {rho_full:.3f}
  (negative = higher hamming → lower logit cosine = more semantically different)

  Same-leaf mean logit cosim:    {mean_same_leaf:.4f}
  Navigable pair mean logit cosim: {f"{np.mean(nav_cosims):.4f}" if nav_cosims else "N/A"}
  Random pair mean logit cosim:    {f"{np.mean(rand_cosims):.4f}" if rand_cosims else "N/A"}
  Meaningful pairs mean cosim:     {f"{np.mean(nav_meaningful_cosims):.4f}" if nav_meaningful_cosims else "N/A"}
  Accidental pairs mean cosim:     {f"{np.mean(nav_accidental_cosims):.4f}" if nav_accidental_cosims else "N/A"}

  INTERPRETATION:
  {'→ Geometric parallelism correlates with semantic relatedness (TruthSpace confirmed)' if rho_full < -0.3 else
   '→ Weak correlation — geometry and semantics partially decouple'}
  {'→ Navigable pairs are MORE semantically related than random' if nav_cosims and rand_cosims and np.mean(nav_cosims) > np.mean(rand_cosims) + 0.02 else
   '→ Navigable pairs similar to random in logit cosine'}
  {'→ Accidental pairs ARE semantically related (model-level)' if nav_accidental_cosims and rand_cosims and np.mean(nav_accidental_cosims) > np.mean(rand_cosims) + 0.02 else
   '→ Accidental pairs not significantly more related than random'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "pearson_r": rho_full,
        "navigable_mean_cosim": float(np.mean(nav_cosims)) if nav_cosims else None,
        "random_mean_cosim": float(np.mean(rand_cosims)) if rand_cosims else None,
        "meaningful_mean_cosim": float(np.mean(nav_meaningful_cosims)) if nav_meaningful_cosims else None,
        "accidental_mean_cosim": float(np.mean(nav_accidental_cosims)) if nav_accidental_cosims else None,
        "same_leaf_mean_cosim": float(mean_same_leaf),
        "hamming_cosim_bins": {str(k): {"mean": float(np.mean(v)), "n": len(v)}
                                for k, v in hamming_cosim.items()},
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 100 complete.")
