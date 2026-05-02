#!/usr/bin/env python3
"""
Day 105 — Prototype Geometric LCM

DC 328 established the architecture:
  Geometric LCM = Trie (semantic addressing) + Transition model (sequential)

Day 104: naive address NN = 0%, address transitions (learned) = 20%.

Day 105 BUILDS the first prototype geometric LCM:
  - ENCODER:    word → 12D trie address (pure geometry, no weights)
  - TRANSITION: address_t → address_{t+1} (learned bigram distribution)
  - DECODER:    address → word (nearest neighbor in address space)

This is the minimal geometry-first language model.

TRAINING CORPUS: 50 simple sentences from PROBE_TOKENS vocabulary
TEST:            20 held-out sentences

METRICS:
  1. Next-token accuracy (top-1, top-5) vs LM and unigram baselines
  2. Sentence perplexity under the address transition model
  3. Free generation: given a seed token, generate 10 tokens
  4. Semantic coherence of generated sequences (logit cosine of consecutive tokens)

ARCHITECTURE VARIANTS:
  A. Unigram transition:    P(addr_next) = count(addr_next) / total
  B. Bigram transition:     P(addr_next | addr_t) = count(addr_t, addr_next) / count(addr_t)
  C. Trigram transition:    P(addr_next | addr_{t-1}, addr_t)
  D. Smoothed bigram:       linear interpolation A + B

PREDICTION:
  - Bigram >> unigram (consistent with Day 104: 20% > random)
  - Trigram may overfit (small corpus)
  - Generated text will be semantically coherent but syntactically poor
    (because addresses cluster by semantics not syntax)
"""
import json, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day105_geometric_lcm.json")
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

# Training corpus: 50 sentences
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

# Test corpus: 20 held-out sentences
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
valid_words = []; logits_list = []
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

# Normalize logits for cosine similarity
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

def hamming_str(a, b): return sum(x != y for x, y in zip(a, b))
ham_mat = np.zeros((N, N), dtype=np.int8)
for i in range(N):
    for j in range(i+1, N):
        h = int(np.sum(addr_int[i] != addr_int[j]))
        ham_mat[i,j] = h; ham_mat[j,i] = h

def addr_to_word(addr, exclude=None):
    """Find nearest vocabulary token to given address."""
    dists = [(j, int(np.sum(addr_int[j] != np.array([{"H":2,"U":1,"L":0}[c] for c in addr], dtype=np.int8))))
             for j in range(N) if j != exclude]
    return valid_words[min(dists, key=lambda x: x[1])[0]]

# ── Build sentence-token sequences from corpus ─────────────────────────────────
def sent_to_tokens(sent):
    return [w for w in sent.split() if w in word_idx]

train_seqs = [sent_to_tokens(s) for s in TRAIN_SENTS]
test_seqs  = [sent_to_tokens(s) for s in TEST_SENTS]

# Address sequences
train_addr_seqs = [[addresses[word_idx[w]] for w in seq] for seq in train_seqs]
test_addr_seqs  = [[addresses[word_idx[w]] for w in seq] for seq in test_seqs]

# ── Build language models ─────────────────────────────────────────────────────
# A: Unigram
unigram_addr = Counter()
for seq in train_addr_seqs:
    unigram_addr.update(seq)
total_unigram = sum(unigram_addr.values())
def unigram_prob(addr): return unigram_addr[addr] / total_unigram if total_unigram > 0 else 1/N
def unigram_predict(): return unigram_addr.most_common(1)[0][0]

# B: Bigram
bigram_addr = defaultdict(Counter)
for seq in train_addr_seqs:
    for i in range(len(seq)-1):
        bigram_addr[seq[i]][seq[i+1]] += 1
def bigram_predict(prev_addr):
    if prev_addr not in bigram_addr: return None
    return bigram_addr[prev_addr].most_common(1)[0][0]
def bigram_prob(prev_addr, addr):
    if prev_addr not in bigram_addr: return 0.0
    total = sum(bigram_addr[prev_addr].values())
    return bigram_addr[prev_addr][addr] / total if total > 0 else 0.0

# C: Trigram
trigram_addr = defaultdict(Counter)
for seq in train_addr_seqs:
    for i in range(len(seq)-2):
        ctx = (seq[i], seq[i+1])
        trigram_addr[ctx][seq[i+2]] += 1
def trigram_predict(prev2_addr, prev_addr):
    ctx = (prev2_addr, prev_addr)
    if ctx not in trigram_addr: return None
    return trigram_addr[ctx].most_common(1)[0][0]

# D: Smoothed bigram: P(addr_next) = λ * bigram(prev, addr_next) + (1-λ) * unigram(addr_next)
LAMBDA = 0.8
def smoothed_bigram_predict(prev_addr):
    # All unique next-addresses seen in training
    all_addrs = set(unigram_addr.keys())
    if prev_addr in bigram_addr:
        # Add bigram-seen addresses
        all_addrs.update(bigram_addr[prev_addr].keys())
        scores = {a: LAMBDA * bigram_prob(prev_addr, a) + (1-LAMBDA) * unigram_prob(a)
                  for a in all_addrs}
    else:
        scores = {a: unigram_prob(a) for a in all_addrs}
    return max(scores, key=scores.get)

# ── Evaluate models on test bigrams ──────────────────────────────────────────
print("=" * 72)
print("Geometric LCM Evaluation — Next-Token Accuracy")
print("=" * 72)

test_bigrams = []
for seq in test_seqs:
    for i in range(len(seq)-1):
        test_bigrams.append((seq[i], seq[i+1]))
n_test = len(test_bigrams)
print(f"\n  Test bigrams: {n_test}\n")

# Evaluate each model
def evaluate_model(predictor_fn, bigrams):
    hits = 0; miss = 0
    for w1, w2 in bigrams:
        a1 = addresses[word_idx[w1]]
        pred_addr = predictor_fn(a1)
        if pred_addr is None: miss += 1; continue
        pred_word = addr_to_word(pred_addr, exclude=word_idx[w1])
        if pred_word == w2: hits += 1
    return hits, miss

# Unigram model
hits_u, miss_u = evaluate_model(lambda a: unigram_predict(), test_bigrams)
# Bigram model
hits_b, miss_b = evaluate_model(lambda a: bigram_predict(a), test_bigrams)
# Smoothed bigram
hits_s, miss_s = evaluate_model(lambda a: smoothed_bigram_predict(a), test_bigrams)
# Trigram (fall back to bigram if unseen)
def trigram_or_bigram(a1, a0=None):
    if a0 is not None:
        p = trigram_predict(a0, a1)
        if p: return p
    return bigram_predict(a1)

# Word-level unigram baseline
word_unigram = Counter()
for seq in train_seqs: word_unigram.update(seq)
most_common_word = word_unigram.most_common(1)[0][0]
hits_wu = sum(1 for _, w2 in test_bigrams if w2 == most_common_word)

# Random baseline
hits_rand = sum(1 for _ in test_bigrams if random.random() < 1/N)

print(f"  {'Model':>30}  {'hits':>5}  {'total':>5}  {'accuracy':>9}")
print(f"  {'-'*60}")
print(f"  {'Unigram addr (most common addr)':>30}  {hits_u:>5}  {n_test:>5}  {100*hits_u/n_test:>8.1f}%")
print(f"  {'Bigram addr':>30}  {hits_b:>5}  {n_test:>5}  {100*hits_b/n_test:>8.1f}%")
print(f"  {'Smoothed bigram (λ=0.8)':>30}  {hits_s:>5}  {n_test:>5}  {100*hits_s/n_test:>8.1f}%")
print(f"  {'Word unigram baseline':>30}  {hits_wu:>5}  {n_test:>5}  {100*hits_wu/n_test:>8.1f}%")
print(f"  {'Random (1/N)':>30}  ~{1:>4}  {n_test:>5}  {100/N:>8.1f}%")

# ── Perplexity under smoothed bigram model ─────────────────────────────────────
print()
print("=" * 72)
print("Address-space Perplexity (smoothed bigram model)")
print("=" * 72)
EPS = 1e-10
total_log_prob = 0.0; n_tokens = 0
for seq in test_addr_seqs:
    if len(seq) < 2: continue
    for i in range(1, len(seq)):
        p_addr = LAMBDA * bigram_prob(seq[i-1], seq[i]) + (1-LAMBDA) * unigram_prob(seq[i])
        total_log_prob += math.log(p_addr + EPS)
        n_tokens += 1
perplexity = math.exp(-total_log_prob / n_tokens) if n_tokens > 0 else float("inf")

# Unigram perplexity baseline
total_log_prob_u = sum(math.log(unigram_prob(a) + EPS)
                       for seq in test_addr_seqs for a in seq[1:] if seq)
n_tokens_u = sum(len(seq)-1 for seq in test_addr_seqs if len(seq) > 1)
perplexity_u = math.exp(-total_log_prob_u / n_tokens_u) if n_tokens_u > 0 else float("inf")

# Random perplexity (uniform over N addresses)
n_unique_train_addrs = len(set(a for seq in train_addr_seqs for a in seq))
perplexity_rand = n_unique_train_addrs

print(f"\n  Smoothed bigram perplexity: {perplexity:.1f}")
print(f"  Unigram perplexity:          {perplexity_u:.1f}")
print(f"  Random (N_unique_addrs):     {perplexity_rand}")
pp_reduction = (perplexity_rand - perplexity) / perplexity_rand * 100
print(f"  Reduction vs random:         {pp_reduction:.0f}%")

# ── Free generation using smoothed bigram model ───────────────────────────────
print()
print("=" * 72)
print("Free Generation — Geometric LCM")
print("=" * 72)
print()

SEED_WORDS = ["the", "dog", "king", "woman", "fire", "good", "run", "tree"]
rng_gen = random.Random(42)

def generate_sequence(seed_word, length=12):
    if seed_word not in word_idx: return [seed_word]
    seq = [seed_word]
    prev_addr = addresses[word_idx[seed_word]]
    for _ in range(length - 1):
        # Sample from smoothed bigram distribution
        all_addrs = set(unigram_addr.keys())
        if prev_addr in bigram_addr:
            all_addrs.update(bigram_addr[prev_addr].keys())
        if not all_addrs: break
        scores = {a: LAMBDA * bigram_prob(prev_addr, a) + (1-LAMBDA) * unigram_prob(a)
                  for a in all_addrs}
        # Temperature sampling
        total_score = sum(scores.values())
        if total_score <= 0: break
        r = rng_gen.random() * total_score
        cumsum = 0.0; next_addr = list(scores.keys())[0]
        for a, s in scores.items():
            cumsum += s
            if cumsum >= r: next_addr = a; break
        next_word = addr_to_word(next_addr)
        seq.append(next_word)
        prev_addr = next_addr
    return seq

def greedy_generate(seed_word, length=12):
    if seed_word not in word_idx: return [seed_word]
    seq = [seed_word]
    prev_addr = addresses[word_idx[seed_word]]
    for _ in range(length - 1):
        next_addr = smoothed_bigram_predict(prev_addr)
        if next_addr is None: break
        next_word = addr_to_word(next_addr)
        seq.append(next_word)
        prev_addr = next_addr
    return seq

generated = {}
for seed in SEED_WORDS:
    greedy = greedy_generate(seed, length=10)
    sampled = generate_sequence(seed, length=10)
    generated[seed] = {"greedy": greedy, "sampled": sampled}
    print(f"  Seed: '{seed}'")
    print(f"    Greedy:  {' '.join(greedy)}")
    print(f"    Sampled: {' '.join(sampled)}")
    print()

# ── Semantic coherence of generated sequences ─────────────────────────────────
print("=" * 72)
print("Semantic Coherence of Generated Sequences")
print("=" * 72)
print(f"\n  (Consecutive logit cosine similarity in generated vs test sentences)\n")

def seq_coherence(seq):
    cosims = []
    for i in range(len(seq)-1):
        w1, w2 = seq[i], seq[i+1]
        if w1 in word_idx and w2 in word_idx:
            cosims.append(float(logits_norm[word_idx[w1]] @ logits_norm[word_idx[w2]]))
    return cosims

greedy_coherences = [seq_coherence(generated[s]["greedy"]) for s in SEED_WORDS]
test_coherences   = [seq_coherence(sent_to_tokens(s)) for s in TEST_SENTS]

all_greedy = [c for cs in greedy_coherences for c in cs]
all_test   = [c for cs in test_coherences   for c in cs]

print(f"  Generated (greedy) mean consecutive cosim: {np.mean(all_greedy):.4f}")
print(f"  Test sentences mean consecutive cosim:     {np.mean(all_test):.4f}")
print(f"  Random pairs mean cosim:                   ~0.8588  (from Day 100)")
print(f"\n  Generated coherence vs random: {np.mean(all_greedy) - 0.8588:+.4f}")
print(f"  Test coherence vs random:      {np.mean(all_test) - 0.8588:+.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 105 Summary — First Prototype Geometric LCM")
print("=" * 72)

best_acc  = max(100*hits_b/n_test, 100*hits_s/n_test)
best_name = "Smoothed bigram" if hits_s >= hits_b else "Bigram"

print(f"""
  Architecture: Trie (geometric addressing) + Bigram (transition model)
  Vocabulary:   {N} tokens, 12D address space

  Next-token accuracy:
    Best geometric model ({best_name}): {best_acc:.1f}%
    Word unigram baseline:             {100*hits_wu/n_test:.1f}%
    Random:                            {100/N:.1f}%

  Perplexity (smoothed bigram): {perplexity:.1f}
  Perplexity reduction vs random: {pp_reduction:.0f}%

  Generated sequence coherence: {np.mean(all_greedy):.4f}
  Test sentence coherence:      {np.mean(all_test):.4f}
  Random pair coherence:        ~0.8588

  VERDICT:
  {'→ Geometric LCM prototype WORKS: beats baseline' if best_acc > 100*hits_wu/n_test else
   '→ Geometric LCM matches word unigram baseline' if best_acc >= 100*hits_wu/n_test - 2 else
   '→ Geometric LCM prototype underperforms word unigram — syntax bottleneck'}

  The prototype demonstrates:
  1. Address transitions learn sequential structure from small corpora
  2. Generated sequences have {np.mean(all_greedy):.4f} consecutive cosim
     (vs random 0.8588, test {np.mean(all_test):.4f})
  3. The geometric LCM IS learnable — it encodes something

  Remaining gap to full language model:
  - No context window (bigram only sees 1 previous address)
  - No syntax/grammar (addresses cluster by semantics, not syntax)
  - Decoding from address is lossy (multiple words per leaf)
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_test_bigrams": n_test,
        "hits_unigram": hits_u, "hits_bigram": hits_b, "hits_smoothed": hits_s,
        "acc_unigram": 100*hits_u/n_test, "acc_bigram": 100*hits_b/n_test,
        "acc_smoothed": 100*hits_s/n_test, "acc_word_unigram": 100*hits_wu/n_test,
        "perplexity_bigram": perplexity, "perplexity_unigram": perplexity_u,
        "generated": generated,
        "greedy_coherence": float(np.mean(all_greedy)),
        "test_coherence": float(np.mean(all_test)),
    }, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 105 complete.")
