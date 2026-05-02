#!/usr/bin/env python3
"""
Day 106 — Geometric LCM with Probabilistic Decoder

Day 105 bottleneck: deterministic nearest-neighbor decoder always maps the
same address to the same word (usually "the" or "old"). Fix: learn P(word|addr)
from training text, then sample from it during generation/prediction.

ARCHITECTURE v2:
  ENCODER:    word → 12D trie address (pure geometry, zero parameters)
  TRANSITION: P(addr_{t+1} | addr_t) — bigram over addresses (learned)
  DECODER:    P(word | addr) — empirical word distribution per address (learned)

The decoder is now: given predicted address A, sample from the empirical
distribution of words observed at address A in training text.

PREDICTION:
  - Probabilistic decoder eliminates the vocabulary bottleneck
  - Next-token accuracy should improve significantly over Day 105 (22.6%)
  - Generated text should be more diverse and less looping

ADDITIONAL EXPERIMENT:
  Full pipeline accuracy:
    1. Given current word, encode to address
    2. Predict next address via bigram transition
    3. Decode to word via P(word | addr)
    Measure: top-1, top-5 accuracy of predicted word

  Compare to:
    A. Deterministic decoder (Day 105 baseline: 22.6%)
    B. Probabilistic decoder (this experiment)
    C. Oracle decoder: P(word | addr, true_next_addr)
    D. Word-level bigram: P(word_t+1 | word_t) directly

  Also test: does the probabilistic decoder fix greedy generation loops?
"""
import json, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day106_prob_decoder.json")
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

TRAIN_SENTS = [
    "the dog ran fast through the green field",
    "a man and a woman walked by the old river",
    "the king and queen ruled the land with great power",
    "the bird flew high above the tall tree and the bright sun",
    "he ate the bread and she drank the cold water",
    "the house was old and the door was hard to open",
    "the sun is bright and the moon is cold and still",
    "the big dog and the small cat ran away quickly",
    "she is happy and he is sad but they go on",
    "the red rose and the old oak tree grow in the garden",
    "blood and bone and skin and hair and hand are the body",
    "the good king and the bad queen fought over the land",
    "he ran fast and she walked slow to the old house",
    "the strong man and the weak boy worked hard all day",
    "the hot fire and the cold water are far from each other",
    "the eye and the ear and the mouth and the nose are on the head",
    "the cat and the dog and the bird all ran from the wolf",
    "the heart of the man and the soul of the woman are deep",
    "the old tree and the new flower are beautiful in the field",
    "he thinks and she knows and they see and hear the truth",
    "the wolf ran through the dark forest to find food",
    "a child found a small key near the old stone bridge",
    "the blue sky and the white cloud and the bright star",
    "she read the book and he wrote the long letter",
    "the fish swam deep in the cold dark river water",
    "the horse ran fast over the high mountain path",
    "a sharp knife cut the thick bread on the old table",
    "the heavy rain fell from the dark cloud over the field",
    "the young boy and the old man worked in the dry soil",
    "a bright flame rose from the dry wood by the river",
    "the bird sang softly in the tall tree near the house",
    "they built a strong wall around the small garden",
    "the deep ocean and the high mountain are far apart",
    "she felt joy and he felt pain but both found hope",
    "the long road led to the small city by the lake",
    "a loud sound broke the quiet of the cold night",
    "the weak rose from the hard ground in the warm sun",
    "he gave the coin to the poor man near the market",
    "the sweet fruit and the bitter leaf grew on the same tree",
    "a thin rope held the heavy stone above the deep water",
    "the slow turtle and the fast eagle lived by the coast",
    "she found the bright gem near the old stone wall",
    "the warm blood ran through the cold bone and the soft skin",
    "they lost their way in the dark forest but found the path",
    "the proud king and the sad queen walked in the wide field",
    "he knew the right answer but she found the wrong one",
    "the soft cloth and the sharp needle lay on the old chair",
    "a small boat sailed on the wide still blue ocean",
    "the dry earth cracked in the bright hot summer sun",
    "he rose early and she went to sleep late that night",
]

TEST_SENTS = [
    "the dog walked slowly through the cold rain",
    "a woman and a man sat near the warm fire",
    "the big eagle flew over the deep blue ocean",
    "she knew the truth and he found the right path",
    "the old house had a hard door and a small window",
    "the wolf and the bear ran through the dark forest",
    "he ate the sweet fruit near the bright red flower",
    "the long river ran from the high mountain to the sea",
    "a strong man built a high wall near the old city",
    "the cat sat on the soft cloth near the warm fire",
    "she read the short book and he wrote many words",
    "the bright moon and the cold star were in the dark sky",
    "the thin rope held the heavy boat near the coast",
    "he ran fast but she walked slow and they went home",
    "the young girl and the old woman sat by the still lake",
    "the sharp knife cut the hard bread on the table",
    "they found the long path to the small garden in the field",
    "the loud wind blew the dry leaf from the old tree",
    "he lost the gold coin near the deep dark river",
    "the sad man and the happy woman walked on the road",
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

def hamming_str(a, b): return sum(x != y for x, y in zip(a, b))
ham_mat = np.zeros((N, N), dtype=np.int8)
for i in range(N):
    for j in range(i+1, N):
        h = int(np.sum(addr_int[i] != addr_int[j]))
        ham_mat[i,j] = h; ham_mat[j,i] = h

def sent_to_tokens(s): return [w for w in s.split() if w in word_idx]
train_seqs = [sent_to_tokens(s) for s in TRAIN_SENTS]
test_seqs  = [sent_to_tokens(s) for s in TEST_SENTS]
train_addr_seqs = [[addresses[word_idx[w]] for w in seq] for seq in train_seqs]
test_addr_seqs  = [[addresses[word_idx[w]] for w in seq] for seq in test_seqs]

# ── Build address-level bigram model (same as Day 105) ─────────────────────────
unigram_addr = Counter(a for seq in train_addr_seqs for a in seq)
total_uni = sum(unigram_addr.values())
bigram_addr = defaultdict(Counter)
for seq in train_addr_seqs:
    for i in range(len(seq)-1):
        bigram_addr[seq[i]][seq[i+1]] += 1

LAMBDA = 0.8
def smoothed_bigram_predict_addr(prev_addr):
    all_addrs = set(unigram_addr.keys())
    if prev_addr in bigram_addr: all_addrs.update(bigram_addr[prev_addr].keys())
    scores = {}
    for a in all_addrs:
        bi = bigram_addr[prev_addr][a] / sum(bigram_addr[prev_addr].values()) \
             if prev_addr in bigram_addr and sum(bigram_addr[prev_addr].values()) > 0 else 0.0
        uni = unigram_addr[a] / total_uni if total_uni > 0 else 1/N
        scores[a] = LAMBDA * bi + (1-LAMBDA) * uni
    return max(scores, key=scores.get) if scores else None

# ── Probabilistic decoder: P(word | addr) ─────────────────────────────────────
# Learn empirical word distribution for each address from training text
addr_to_words = defaultdict(Counter)
for seq in train_seqs:
    for w in seq:
        addr_to_words[addresses[word_idx[w]]][w] += 1

# For addresses unseen in training, fall back to nearest-address words
def find_nearest_addr(query_addr):
    """Find closest training address to query_addr by Hamming distance."""
    best_addr = None; best_dist = 13
    for a in addr_to_words:
        d = hamming_str(query_addr, a)
        if d < best_dist: best_dist = d; best_addr = a
    return best_addr

def decode_addr_prob(addr, top_k=5, rng=None):
    """Return top-k words by P(word | addr), with fallback."""
    if addr in addr_to_words:
        dist = addr_to_words[addr]
    else:
        fallback = find_nearest_addr(addr)
        dist = addr_to_words[fallback] if fallback else Counter()
    if not dist: return [], []
    total = sum(dist.values())
    words = [w for w, _ in dist.most_common(top_k)]
    probs = [dist[w] / total for w in words]
    return words, probs

def decode_addr_top1(addr):
    words, _ = decode_addr_prob(addr, top_k=1)
    return words[0] if words else None

def decode_addr_sample(addr, rng):
    words, probs = decode_addr_prob(addr, top_k=20)
    if not words: return None
    r = rng.random()
    cumsum = 0.0
    for w, p in zip(words, probs):
        cumsum += p
        if cumsum >= r: return w
    return words[-1]

# ── Word-level bigram (oracle comparison) ─────────────────────────────────────
word_bigram = defaultdict(Counter)
for seq in train_seqs:
    for i in range(len(seq)-1):
        word_bigram[seq[i]][seq[i+1]] += 1
word_unigram = Counter(w for seq in train_seqs for w in seq)
total_wu = sum(word_unigram.values())

def word_bigram_predict(prev_word, top_k=5):
    LAMBDA_W = 0.8
    all_words = set(word_unigram.keys())
    if prev_word in word_bigram: all_words.update(word_bigram[prev_word].keys())
    scores = {}
    for w in all_words:
        bi = word_bigram[prev_word][w] / sum(word_bigram[prev_word].values()) \
             if prev_word in word_bigram and sum(word_bigram[prev_word].values()) > 0 else 0.0
        uni = word_unigram[w] / total_wu if total_wu > 0 else 0.0
        scores[w] = LAMBDA_W * bi + (1-LAMBDA_W) * uni
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return ranked

# ── Evaluate all models on test bigrams ──────────────────────────────────────
test_bigrams = [(w1, w2) for seq in test_seqs for w1, w2 in zip(seq, seq[1:])]
n_test = len(test_bigrams)
print(f"Test bigrams: {n_test}\n")

hits = {"det_top1": 0, "det_top5": 0,
        "prob_top1": 0, "prob_top5": 0,
        "word_bigram_top1": 0, "word_bigram_top5": 0,
        "word_unigram": 0}

for w1, w2 in test_bigrams:
    a1 = addresses[word_idx[w1]]
    pred_addr = smoothed_bigram_predict_addr(a1)
    if pred_addr:
        # Deterministic decoder (Day 105 method: nearest vocab token)
        dists = [(j, int(ham_mat[j] @ np.ones(N, dtype=np.int8)))
                 for j in range(N)]
        # Actually: find nearest word in vocabulary to pred_addr
        # Use hamming from pred_addr to all vocab addresses
        pred_addr_arr = np.array([{"H":2,"U":1,"L":0}[c] for c in pred_addr], dtype=np.int8)
        ham_to_pred = np.sum(addr_int != pred_addr_arr, axis=1)
        det_top1_j = int(np.argmin(ham_to_pred))
        det_top5_j = np.argsort(ham_to_pred)[:5].tolist()
        if valid_words[det_top1_j] == w2: hits["det_top1"] += 1
        if w2 in [valid_words[j] for j in det_top5_j]: hits["det_top5"] += 1
        # Probabilistic decoder
        prob_words_top5, _ = decode_addr_prob(pred_addr, top_k=5)
        if prob_words_top5 and prob_words_top5[0] == w2: hits["prob_top1"] += 1
        if w2 in prob_words_top5: hits["prob_top5"] += 1
    # Word-level bigram
    wb_top5 = word_bigram_predict(w1, top_k=5)
    if wb_top5 and wb_top5[0] == w2: hits["word_bigram_top1"] += 1
    if w2 in wb_top5: hits["word_bigram_top5"] += 1
    # Word unigram
    if word_unigram.most_common(1)[0][0] == w2: hits["word_unigram"] += 1

print("=" * 72)
print("Next-Token Accuracy Comparison")
print("=" * 72)
print(f"\n  {'Model':>35}  {'top1%':>7}  {'top5%':>7}")
print(f"  {'-'*55}")
print(f"  {'Deterministic decoder (Day 105)':>35}  {100*hits['det_top1']/n_test:>6.1f}%  "
      f"{100*hits['det_top5']/n_test:>6.1f}%")
print(f"  {'Probabilistic decoder (Day 106)':>35}  {100*hits['prob_top1']/n_test:>6.1f}%  "
      f"{100*hits['prob_top5']/n_test:>6.1f}%")
print(f"  {'Word bigram (oracle)':>35}  {100*hits['word_bigram_top1']/n_test:>6.1f}%  "
      f"{100*hits['word_bigram_top5']/n_test:>6.1f}%")
print(f"  {'Word unigram':>35}  {100*hits['word_unigram']/n_test:>6.1f}%  {'N/A':>7}")
print(f"  {'Random (1/N)':>35}  {100/N:>6.1f}%  {500/N:>6.1f}%")

# ── Oracle decoder: given true next address, decode probabilistically ─────────
oracle_hits_top1 = 0; oracle_hits_top5 = 0
for w1, w2 in test_bigrams:
    if w2 not in word_idx: continue
    true_addr = addresses[word_idx[w2]]
    oracle_words, _ = decode_addr_prob(true_addr, top_k=5)
    if oracle_words and oracle_words[0] == w2: oracle_hits_top1 += 1
    if w2 in oracle_words: oracle_hits_top5 += 1
print(f"  {'Oracle decoder (true next addr)':>35}  {100*oracle_hits_top1/n_test:>6.1f}%  "
      f"{100*oracle_hits_top5/n_test:>6.1f}%")

# ── Diagnostic: address→words distribution ────────────────────────────────────
print()
print("=" * 72)
print("Address→Words decoder distribution (top-5 addresses by token count)")
print("=" * 72)
addr_usage = Counter(a for seq in train_addr_seqs for a in seq)
print(f"\n  {'address':>12}  {'count':>6}  top-3 words with frequencies")
print(f"  {'-'*65}")
for addr, cnt in addr_usage.most_common(10):
    dist = addr_to_words[addr]
    top3 = dist.most_common(3)
    top3_str = ", ".join(f"{w}×{c}" for w, c in top3)
    print(f"  {addr:>12}  {cnt:>6}  {top3_str}")

# ── Free generation with probabilistic decoder ────────────────────────────────
print()
print("=" * 72)
print("Free Generation — Probabilistic Decoder (sampled)")
print("=" * 72)
print()

SEED_WORDS = ["the", "dog", "king", "woman", "fire", "good", "run", "tree"]
rng_gen = random.Random(42)

def generate_prob(seed_word, length=12, greedy=True):
    if seed_word not in word_idx: return [seed_word]
    seq = [seed_word]; prev_addr = addresses[word_idx[seed_word]]
    for _ in range(length - 1):
        all_addrs = set(unigram_addr.keys())
        if prev_addr in bigram_addr: all_addrs.update(bigram_addr[prev_addr].keys())
        if not all_addrs: break
        scores = {}
        for a in all_addrs:
            bi = bigram_addr[prev_addr][a] / sum(bigram_addr[prev_addr].values()) \
                 if prev_addr in bigram_addr and sum(bigram_addr[prev_addr].values()) > 0 else 0.0
            uni = unigram_addr[a] / total_uni if total_uni > 0 else 1/N
            scores[a] = LAMBDA * bi + (1-LAMBDA) * uni
        if greedy:
            next_addr = max(scores, key=scores.get)
            next_word = decode_addr_top1(next_addr) or valid_words[0]
        else:
            total_s = sum(scores.values())
            r = rng_gen.random() * total_s
            cumsum = 0.0; next_addr = list(scores.keys())[0]
            for a, s in scores.items():
                cumsum += s
                if cumsum >= r: next_addr = a; break
            next_word = decode_addr_sample(next_addr, rng_gen) or valid_words[0]
        seq.append(next_word); prev_addr = next_addr
    return seq

gen_results = {}
for seed in SEED_WORDS:
    greedy = generate_prob(seed, length=10, greedy=True)
    sampled = generate_prob(seed, length=10, greedy=False)
    gen_results[seed] = {"greedy": greedy, "sampled": sampled}
    print(f"  Seed: '{seed}'")
    print(f"    Greedy:  {' '.join(greedy)}")
    print(f"    Sampled: {' '.join(sampled)}")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 106 Summary — Geometric LCM v2 with Probabilistic Decoder")
print("=" * 72)

prob_top1 = 100*hits['prob_top1']/n_test
det_top1  = 100*hits['det_top1']/n_test
wb_top1   = 100*hits['word_bigram_top1']/n_test

print(f"""
  Decoder comparison (next-token top-1 accuracy):
    Deterministic (Day 105): {det_top1:.1f}%
    Probabilistic (Day 106): {prob_top1:.1f}%
    Word bigram (oracle):    {wb_top1:.1f}%
    Random:                   {100/N:.1f}%

  Probabilistic decoder gain vs deterministic: {prob_top1 - det_top1:+.1f}pp

  Oracle analysis:
    Oracle top-1 (true addr → prob decode): {100*oracle_hits_top1/n_test:.1f}%
    This is the ceiling of the address+prob_decode pipeline.
    Gap from oracle = transition model error (wrong next address)

  VERDICT:
  {'→ Probabilistic decoder IMPROVES accuracy' if prob_top1 > det_top1 else
   '→ Probabilistic decoder is similar to deterministic' if abs(prob_top1 - det_top1) <= 1 else
   '→ Probabilistic decoder DEGRADES accuracy (data sparsity issue)'}

  Architecture v2 status:
  - Encoder: 12D trie address (pure geometry) ✓
  - Transition: smoothed bigram over addresses ✓
  - Decoder: P(word|addr) empirical distribution ✓ (fixed vocabulary bottleneck)

  Remaining bottlenecks:
  1. Transition accuracy: wrong next address → wrong next word
  2. Data sparsity: many address pairs unseen in 50-sentence corpus
  3. Context window: bigram only (no attention-equivalent history)
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_test": n_test,
        "det_top1": det_top1, "prob_top1": prob_top1,
        "word_bigram_top1": wb_top1,
        "oracle_top1": 100*oracle_hits_top1/n_test,
        "oracle_top5": 100*oracle_hits_top5/n_test,
        "generated": {k: {"greedy": v["greedy"], "sampled": v["sampled"]}
                      for k, v in gen_results.items()},
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 106 complete.")
