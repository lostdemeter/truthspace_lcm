#!/usr/bin/env python3
"""
Day 97 — Complete Axis Traversal Survey

DC 327 has a partial classification table for trie navigability:
  Confirmed navigable: gender, comparative (partial), past_tense (conditional)
  Not navigable:       plural, antonym
  UNTESTED:            hypernym, concrete, synonym, passive, causation,
                       question, negation

Day 97 goal: test ALL untested axes for traversal accuracy, using:
  1. Standard 12D full 401-token isolated trie (Day 92 method)
  2. POS-restricted isolated tries where relevant (nouns for hypernym/concrete,
     verbs for passive/causation, sentence pairs for negation/question)

GROUND TRUTH pairs to test:
  hypernym:  dog→animal, rose→flower, eagle→bird, car→vehicle,
             soldier→person, hammer→tool, oak→tree, ruby→gem
  concrete:  road→journey, chain→bond, wall→barrier, flame→hope
             (concrete→abstract, same sentence context)
  synonym:   big→large, small→tiny, fast→quick, cold→frigid,
             happy→joyful, hard→difficult
  passive:   (active→passive voice — requires sentence framing)
  negation:  fast→(not fast), good→(not good), etc.
  question:  (declarative→interrogative — requires sentence framing)
  causation: (cause→effect — requires sentence framing)

NOTE: passive, question, causation are SENTENCE-LEVEL transformations.
Individual tokens don't have inherent passive/question/causation bits.
We'll test if any token in the vocabulary acts as a marker.
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day97_all_axes_traversal.json")
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

# Ground truth traversal pairs for NEW axes
# Each pair: (src_token, tgt_token) — isolated token comparison
GT_NEW = {
    "hypernym": [
        ("dog", "animal"), ("eagle", "bird"), ("car", "vehicle"),
        ("rose", "flower"), ("oak", "tree"), ("salmon", "fish"),
        ("hammer", "tool"), ("ruby", "gem"),
        ("wolf", "animal"), ("bear", "animal"),
        ("sparrow", "bird"), ("owl", "bird"),
    ],
    "concrete": [
        ("road", "path"), ("wall", "barrier"), ("fire", "anger"),
        ("stone", "burden"), ("chain", "bond"), ("root", "base"),
        ("river", "journey"), ("rock", "burden"),
    ],
    "synonym": [
        ("big", "large"), ("small", "tiny"), ("fast", "quick"),
        ("cold", "frigid"), ("happy", "joyful"), ("hard", "difficult"),
        ("old", "aged"), ("loud", "noisy"),
    ],
    # sentence-level axes — testing token-level markers for these is a stretch
    # but let's check if function words / auxiliaries have different bits
    "negation": [
        ("fast", "slow"),    # not fast → opposite? (NO — negation ≠ antonym)
        ("good", "bad"),     # negation might flip polarity
        ("strong", "weak"),
        ("hot", "cold"),
    ],
    "passive": [
        ("cat", "mouse"),    # "the cat chased" vs "the mouse was chased"
        ("chef", "meal"),    # "the chef cooked" vs "the meal was cooked"
        ("dog", "man"),
        ("artist", "picture"),
    ],
}

# Also include Day 92 ground truth for reference
GT_DAY92 = {
    "gender":     [("king","queen"),("man","woman"),("boy","girl"),
                   ("brother","sister"),("father","mother"),("son","daughter"),
                   ("prince","princess"),("actor","actress")],
    "plural":     [("dog","dogs"),("cat","cats"),("tree","trees"),
                   ("bird","birds"),("hand","hands"),("eye","eyes")],
    "past_tense": [("run","ran"),("walk","walked"),("jump","jumped"),
                   ("fly","flew"),("eat","ate"),("build","built"),
                   ("write","wrote"),("break","broke")],
    "comparative":[("fast","faster"),("big","bigger"),("slow","slower"),
                   ("small","smaller"),("good","better"),("bad","worse")],
    "antonym":    [("hot","cold"),("big","small"),("fast","slow"),
                   ("hard","soft"),("happy","sad"),("strong","weak"),
                   ("good","bad"),("old","new")],
}

# T2 sentence pairs
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom",   "The queen ruled with great wisdom"),
        ("A man walked through the forest",    "A woman walked through the forest"),
        ("The boy kicked the ball hard",       "The girl kicked the ball hard"),
        ("His brother arrived at the party",   "His sister arrived at the party"),
        ("The father worked to feed family",   "The mother worked to feed family"),
        ("A son was born in the winter",       "A daughter was born in the winter"),
        ("The prince rode across the land",    "The princess rode across the land"),
        ("The actor played a leading role",    "The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car", "The faster car"), ("A big dog", "A bigger dog"),
        ("The cold wind", "The colder wind"), ("A tall tree", "A taller tree"),
        ("The old house", "The older house"), ("A bright star", "A brighter star"),
        ("The dark room", "The darker room"), ("A hard rock", "A harder rock"),
    ],
    "hypernym": [
        ("The dog ran away from danger",    "The animal ran away from danger"),
        ("A rose bloomed in the garden",    "A flower bloomed in the garden"),
        ("The oak crashed in the storm",    "The tree crashed in the storm"),
        ("The car sped past the sign",      "The vehicle sped past the sign"),
        ("The eagle soared above the hill", "The bird soared above the hill"),
        ("The ruby gleamed in the light",   "The gem gleamed in the light"),
        ("The soldier marched into fight",  "The person marched into fight"),
        ("The hammer struck the nail",      "The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field",    "Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window", "The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist",    "Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm",   "The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk",          "Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road",   "The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky",     "Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text",   "The words appeared clearly in the printed text"),
    ],
    "synonym": [
        ("He is big", "He is large"), ("She is small", "She is tiny"),
        ("He runs fast", "He runs quick"), ("It is cold", "It is frigid"),
        ("She is happy", "She is joyful"), ("He spoke loudly", "He spoke noisily"),
        ("It is hard", "It is difficult"), ("He is old", "He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift",  "The burden is too heavy to lift"),
        ("The iron chain has broken now",   "The bond between them has broken"),
        ("The long road leads to the sea",  "The long journey leads to the sea"),
        ("The high wall blocks the view",   "The high barrier blocks the view"),
        ("The flame slowly fades away",     "The hope slowly fades away"),
        ("The strong root grips the soil",  "The strong base grips the earth"),
        ("The bridge connects two banks",   "The bond connects two communities"),
        ("The small key opens the door",    "The small answer opens the path"),
    ],
    "past_tense": [
        ("I walk to the market every single morning",        "I walked to the market every single morning"),
        ("She runs through the park after her long work",    "She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house",   "He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden",        "They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days",          "We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend",       "She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting", "He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire",   "They sang together around the evening campfire"),
    ],
    "antonym": [
        ("It is hot", "It is cold"), ("He runs fast", "He runs slow"),
        ("The light is on", "The dark is on"), ("The news is good","The news is bad"),
        ("It is hard", "It is soft"), ("She is happy", "She is sad"),
        ("He is strong", "He is weak"), ("It is the first", "It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse",         "The mouse was chased by the cat"),
        ("John broke the window",            "The window was broken by John"),
        ("The chef cooked the meal",         "The meal was cooked by the chef"),
        ("The dog bit the man",              "The man was bitten by the dog"),
        ("The teacher helped the student",   "The student was helped by the teacher"),
        ("The storm destroyed the house",    "The house was destroyed by the storm"),
        ("The artist painted the picture",   "The picture was painted by the artist"),
        ("The king signed the document",     "The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day",    "The ground gets completely wet"),
        ("The fire burns for a long time",  "The wood turns to ash slowly"),
        ("The sun heats the cold earth",    "The ice melts quickly in spring"),
        ("The wind blows the tree branches","The leaves fall to the ground"),
        ("The child cries very loudly",     "The mother comes running in"),
        ("The ball rolls off the tall edge","The ball falls to the floor"),
        ("The teacher praises the student", "The student feels very proud"),
        ("The glass breaks on hard stone",  "The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today",         "Is she very tired today"),
        ("He can swim really well",         "Can he swim really well"),
        ("They went to the market",         "Did they go to the market"),
        ("The car broke down again",        "Did the car break down again"),
        ("The dog is hungry now",           "Is the dog hungry now"),
        ("She wrote the letter herself",    "Did she write the letter herself"),
        ("He knows the right answer",       "Does he know the right answer"),
        ("The house looks very old",        "Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast",    "The dog is not fast"),
        ("She can swim well",  "She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good",   "The food is not good"),
        ("They work hard",     "They do not work hard"),
        ("The water is cold",  "The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today", "It will not rain today"),
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
    # New tokens for Day 97 hypernym tests
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
print(f"  hidden={hidden_size}  layers={ALL_LAYERS}\n")

def get_h(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

# ── Compute T2 axes ───────────────────────────────────────────────────────────
print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L  = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_h(s1, [L])[L]; h2 = get_h(s2, [L])[L]
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    print(f"  {name:<15} L{L}")
print()

# ── Extract probe token hidden states ────────────────────────────────────────
print("Extracting probe token hidden states ...")
hs_by_layer = {L: [] for L in ALL_LAYERS}
logits_list = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        hs, lg = {}, None
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        for L in ALL_LAYERS: hs[L] = out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
        lg = out.logits[0, pos, :].numpy().astype(np.float32)
        for L in ALL_LAYERS: hs_by_layer[L].append(hs[L])
        logits_list.append(lg); valid_words.append(word)
    except: pass
for L in ALL_LAYERS:
    hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

def classify_all(axis_vec, layer_hs, N):
    if np.linalg.norm(axis_vec) < 1e-6: return ["U"] * N
    projs    = [float(np.dot(layer_hs[i], axis_vec)) for i in range(N)]
    max_p    = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * N
    hi, lo   = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

# Build 12D addresses
classes  = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    classes[name] = classify_all(t2_axes[name], hs_by_layer[L], N)
addresses = ["".join(classes[n][i] for n in AXIS_NAMES_12) for i in range(N)]

def hamming(a, b): return sum(x != y for x, y in zip(a, b))
FLIP_MAP = {"H": "L", "L": "H", "U": "H"}

def traversal(gt_pairs, axis_bit_idx, label=""):
    hits = 0; total = 0; details = []
    for src, tgt in gt_pairs:
        if src not in word_idx or tgt not in word_idx:
            details.append({"src": src, "tgt": tgt, "rank": -99, "skip": True})
            continue
        si = word_idx[src]; ti = word_idx[tgt]
        fl = list(addresses[si])
        fl[axis_bit_idx] = FLIP_MAP[fl[axis_bit_idx]]
        fl = "".join(fl)
        ranked = sorted([(j, hamming(fl, addresses[j])) for j in range(N) if j != si],
                         key=lambda x: x[1])
        top5   = [valid_words[j] for j, _ in ranked[:5]]
        rank   = next((k for k, (j, _) in enumerate(ranked) if j == ti), -1)
        hit    = 0 <= rank < 5
        if hit: hits += 1
        total += 1
        src_b = addresses[si][axis_bit_idx]; tgt_b = addresses[ti][axis_bit_idx]
        details.append({"src": src, "tgt": tgt, "rank": rank, "top5": top5,
                         "hit": hit, "src_bit": src_b, "tgt_bit": tgt_b})
        print(f"    {src:>10}({src_b})\u2192{tgt:>10}({tgt_b})  "
              f"rank={rank if rank >= 0 else 'miss':>5}  top3={'/'.join(top5[:3])}")
    return hits, total, details

# ── Run traversal for ALL axes ─────────────────────────────────────────────────
print("=" * 72)
print("Traversal survey — all 12 axes")
print("=" * 72)

all_traversal = {}

# --- Previously tested axes (Day 92 results for reference) ---
print("\n[Previously tested axes — Day 92 reference]")
for name in ["gender", "plural", "past_tense", "comparative", "antonym"]:
    idx = AXIS_NAMES_12.index(name)
    print(f"\n  {name.upper()} (bit {idx}, L{DAY78_LAYERS[name]}):")
    gt = GT_DAY92.get(name, [])
    h, t, det = traversal(gt, idx)
    all_traversal[name] = {"hits": h, "total": t, "pct": 100*h/max(1,t), "details": det}
    print(f"  → {h}/{t} ({100*h/max(1,t):.0f}%)")

# --- New axes ---
print("\n[New axes — Day 97]")
for name in ["hypernym", "concrete", "synonym", "passive", "negation",
             "causation", "question"]:
    idx = AXIS_NAMES_12.index(name)
    gt  = GT_NEW.get(name, [])
    if not gt:
        all_traversal[name] = {"hits": 0, "total": 0, "pct": 0, "details": []}
        continue
    print(f"\n  {name.upper()} (bit {idx}, L{DAY78_LAYERS[name]}):")
    h, t, det = traversal(gt, idx)
    all_traversal[name] = {"hits": h, "total": t, "pct": 100*h/max(1,t), "details": det}
    print(f"  → {h}/{t} ({100*h/max(1,t):.0f}%)")

# ── Bit separation analysis for new axes ─────────────────────────────────────
print()
print("=" * 72)
print("Bit separation analysis for new axes")
print("=" * 72)
for name in ["hypernym", "concrete", "synonym", "passive", "negation"]:
    idx = AXIS_NAMES_12.index(name)
    gt  = GT_NEW.get(name, [])
    sep = 0; tot = 0
    for src, tgt in gt:
        if src not in word_idx or tgt not in word_idx: continue
        si = word_idx[src]; ti = word_idx[tgt]
        sb = addresses[si][idx]; tb = addresses[ti][idx]
        sep += (sb != tb); tot += 1
    print(f"  {name:<15} {sep}/{tot} ({100*sep/max(1,tot):.0f}%) pairs separated")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 97 Summary: Complete Axis Traversal Map")
print("=" * 72)
print(f"\n  {'axis':>15}  {'traversal':>14}  type")
print(f"  {'-'*55}")

axis_types = {
    "gender": "category", "comparative": "relational-degree",
    "hypernym": "relational-semantic", "plural": "relational-morphological",
    "synonym": "relational-semantic", "concrete": "category-semantic",
    "past_tense": "relational-morphological", "antonym": "relational-semantic",
    "passive": "relational-syntactic", "causation": "relational-semantic",
    "question": "relational-syntactic", "negation": "relational-semantic",
}
for name in AXIS_NAMES_12:
    r = all_traversal.get(name)
    if r and r["total"] > 0:
        pct_str = f"{r['hits']:>2}/{r['total']:>2} ({r['pct']:.0f}%)"
    else:
        pct_str = "not tested"
    atype = axis_types.get(name, "unknown")
    print(f"  {name:>15}  {pct_str:>14}  {atype}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"traversal": all_traversal}, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 97 complete.")
