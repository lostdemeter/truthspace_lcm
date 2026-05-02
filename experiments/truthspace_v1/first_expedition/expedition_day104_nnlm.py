#!/usr/bin/env python3
"""
Day 104 — Address-Based Nearest-Neighbor Language Modeling (NNLM)

THE CORE TEST: Does the 12D trie address carry enough information to
predict the next token in a sequence, WITHOUT running the language model?

If the TruthSpace hypothesis is correct — "the shape IS the knowledge" —
then tokens with similar 12D addresses should co-occur in similar contexts.
This means: given token T_t, the token T_{t+1} most likely to follow it
should have a 12D address close to T_t's address (or to T_{t-1}→T_t delta).

EXPERIMENTS:

1. Static address similarity (baseline):
   For each token in vocabulary, find its 5 nearest neighbors by 12D Hamming.
   Measure: does the nearest neighbor appear MORE often as the next word in
   actual text than a random vocabulary token?

2. Same-leaf prediction:
   For each token, if we predict "the next token is in the same leaf", how
   often is the actual next token from the same leaf?

3. Navigated prediction (address arithmetic):
   Given T_t, predict T_{t+1} by:
   - Finding T_t's 12D address
   - Computing the most frequent address → next-address transition
     learned from a small corpus
   - Finding the nearest vocabulary token to the predicted next address

4. Comparison baseline:
   - Unigram frequency: predict the most common token always
   - Random: predict a random vocabulary token
   - LM greedy: predict the argmax of the actual LM logits

TEST CORPUS: 20 simple English sentences covering vocabulary tokens.

PREDICTION:
  - Same-leaf hit rate > random (matches LOO finding: same-leaf cosim=0.9123)
  - Address neighbor hit rate > random (matches Hamming-semantic correlation)
  - Address arithmetic may predict better than unigram for content tokens
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day104_nnlm.json")
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

# Test sentences — all words must be in vocabulary
TEST_SENTENCES = [
    "the dog ran fast and the cat walked slowly",
    "a man and a woman walked through the forest",
    "the king and queen ruled the land well",
    "the bird flew high above the tall tree",
    "he ate the bread and she drank the water",
    "the house was old and the door was broken",
    "the sun is bright and the moon is cold",
    "the big dog and the small cat ran away",
    "she is happy and he is sad and they go",
    "the red rose and the old oak tree grow",
    "blood and bone and skin and hair and hand",
    "the good king and the bad queen fought hard",
    "he ran fast and she walked slow to the house",
    "the strong man and the weak boy worked well",
    "the hot fire and the cold water are far",
    "the eye and the ear and the mouth and the nose",
    "the cat and the dog and the bird all ran",
    "the heart of the man and the soul of the woman",
    "the old tree and the new flower are beautiful",
    "he thinks and she knows and they see and hear",
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

# Unigram frequencies from test sentences
unigram = Counter()
for sent in TEST_SENTENCES:
    for w in sent.split(): unigram[w] += 1
most_common_word = unigram.most_common(1)[0][0]

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

# Precompute full hamming matrix
ham_mat = np.zeros((N, N), dtype=np.int8)
for i in range(N):
    for j in range(i+1, N):
        h = int(np.sum(addr_int[i] != addr_int[j]))
        ham_mat[i,j] = h; ham_mat[j,i] = h

def nearest_neighbors(idx, k=5, exclude_self=True):
    dists = [(j, int(ham_mat[idx,j])) for j in range(N) if j != idx or not exclude_self]
    return sorted(dists, key=lambda x: x[1])[:k]

# ── Build test bigrams ─────────────────────────────────────────────────────────
bigrams = []  # (prev_word, next_word)
for sent in TEST_SENTENCES:
    words = sent.split()
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if w1 in word_idx and w2 in word_idx:
            bigrams.append((w1, w2))

print(f"Test bigrams from sentences: {len(bigrams)}")
print()

# ── Experiment 1: Same-leaf prediction ────────────────────────────────────────
print("=" * 72)
print("Exp 1: Same-leaf prediction")
print("=" * 72)
same_leaf_hits = 0
for w1, w2 in bigrams:
    i1, i2 = word_idx[w1], word_idx[w2]
    if addresses[i1] == addresses[i2]: same_leaf_hits += 1
total = len(bigrams)
# Random baseline: fraction of bigrams where next token is in same leaf
# = (leaf_size - 1) / (N - 1) averaged over all leaves
leaf_sizes = Counter(addresses)
avg_leaf_size = sum(s*s for s in leaf_sizes.values()) / N  # expected same-leaf count
random_same_leaf = (avg_leaf_size - 1) / (N - 1) if N > 1 else 0
print(f"  Same-leaf bigram hits: {same_leaf_hits}/{total} ({100*same_leaf_hits/total:.1f}%)")
print(f"  Random baseline:       {100*random_same_leaf:.1f}%")
print(f"  Lift:                  {100*(same_leaf_hits/total - random_same_leaf):.1f}pp")

# ── Experiment 2: Nearest-address prediction ──────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Nearest-address next-token prediction")
print("=" * 72)
nn_hits_top1 = 0; nn_hits_top5 = 0; nn_hits_top10 = 0
lm_hits_top1 = 0; unigram_hits = 0; random_hits = 0
lm_ranks = []; nn_ranks = []

rng_rand = np.random.default_rng(42)

for w1, w2 in bigrams:
    i1, i2 = word_idx[w1], word_idx[w2]
    # NN prediction: nearest 12D address neighbors (excluding self)
    nn = [j for j, _ in nearest_neighbors(i1, k=10)]
    if i2 in nn[:1]: nn_hits_top1 += 1
    if i2 in nn[:5]: nn_hits_top5 += 1
    if i2 in nn[:10]: nn_hits_top10 += 1
    nn_ranks.append(nn.index(i2) if i2 in nn else N)
    # LM greedy prediction (argmax of logits into full tokenizer vocab)
    lm_top20_ids = np.argsort(logits_arr[i1])[::-1][:200]
    lm_top20_words = [tok.decode([tid]).strip().lower() for tid in lm_top20_ids]
    if w2 in lm_top20_words[:1]: lm_hits_top1 += 1
    lm_rank = lm_top20_words.index(w2) if w2 in lm_top20_words else N
    lm_ranks.append(lm_rank)
    # Unigram: always predict most common word
    if w2 == most_common_word: unigram_hits += 1
    # Random
    rand_idx = int(rng_rand.integers(N))
    if rand_idx == i2: random_hits += 1

print(f"\n  Prediction accuracy (top-1, top-5, top-10):")
print(f"  {'method':>25}  {'top1%':>6}  {'top5%':>6}  {'top10%':>7}")
print(f"  {'-'*55}")
print(f"  {'Nearest address (Hamming)':>25}  "
      f"{100*nn_hits_top1/total:>6.1f}  "
      f"{100*nn_hits_top5/total:>6.1f}  "
      f"{100*nn_hits_top10/total:>7.1f}")
print(f"  {'LM greedy (logit argmax)':>25}  "
      f"{100*lm_hits_top1/total:>6.1f}  "
      f"{'N/A':>6}  {'N/A':>7}")
print(f"  {'Unigram (most frequent)':>25}  "
      f"{100*unigram_hits/total:>6.1f}  "
      f"{'N/A':>6}  {'N/A':>7}")
print(f"  {'Random':>25}  "
      f"{100/N:>6.1f}  "
      f"{500/N:>6.1f}  "
      f"{1000/N:>7.1f}")

# ── Experiment 3: Logit-space nearest neighbor ────────────────────────────────
print()
print("=" * 72)
print("Exp 3: Logit cosine similarity as next-token predictor")
print("=" * 72)
logits_norm = logits_arr / (np.linalg.norm(logits_arr, axis=1, keepdims=True) + 1e-10)
logit_cosim = logits_norm @ logits_norm.T

logit_nn_top1 = 0; logit_nn_top5 = 0
for w1, w2 in bigrams:
    i1, i2 = word_idx[w1], word_idx[w2]
    sims = [(j, float(logit_cosim[i1, j])) for j in range(N) if j != i1]
    sims_sorted = sorted(sims, key=lambda x: -x[1])
    top5_j = [j for j, _ in sims_sorted[:5]]
    if i2 == sims_sorted[0][0]: logit_nn_top1 += 1
    if i2 in top5_j: logit_nn_top5 += 1
print(f"\n  Logit cosine NN top-1: {logit_nn_top1}/{total} ({100*logit_nn_top1/total:.1f}%)")
print(f"  Logit cosine NN top-5: {logit_nn_top5}/{total} ({100*logit_nn_top5/total:.1f}%)")
print(f"  Compared to address NN top-1: {100*nn_hits_top1/total:.1f}%")

# ── Experiment 4: Address transition learning ─────────────────────────────────
print()
print("=" * 72)
print("Exp 4: Address transition learning (in-context prediction)")
print("=" * 72)
# Learn: given addr(T_t) → predict addr(T_{t+1})
# Use first 15 sentences to learn transitions, last 5 to evaluate
train_sents = TEST_SENTENCES[:15]; test_sents = TEST_SENTENCES[15:]
# Count address bigrams in training
addr_bigram = Counter()
for sent in train_sents:
    words = sent.split()
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if w1 in word_idx and w2 in word_idx:
            a1, a2 = addresses[word_idx[w1]], addresses[word_idx[w2]]
            addr_bigram[(a1, a2)] += 1

# Build: given addr_prev, what's the most likely addr_next?
addr_transition = defaultdict(Counter)
for (a1, a2), cnt in addr_bigram.items():
    addr_transition[a1][a2] += cnt

# Evaluate on test sentences
test_bigrams = []
for sent in test_sents:
    words = sent.split()
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if w1 in word_idx and w2 in word_idx:
            test_bigrams.append((w1, w2))

trans_hits = 0; trans_miss = 0
for w1, w2 in test_bigrams:
    i1, i2 = word_idx[w1], word_idx[w2]
    a1 = addresses[i1]; a2_actual = addresses[i2]
    if a1 not in addr_transition:
        trans_miss += 1; continue
    # Predict: find vocab word at most likely next address
    pred_addr = addr_transition[a1].most_common(1)[0][0]
    # Find nearest token to pred_addr
    candidates = [(j, hamming_str(pred_addr, addresses[j])) for j in range(N)]
    predicted_j = min(candidates, key=lambda x: x[1])[0]
    if predicted_j == i2: trans_hits += 1

n_test = len(test_bigrams)
print(f"\n  Address-transition learned predictor:")
print(f"  Hit rate: {trans_hits}/{n_test} ({100*trans_hits/max(1,n_test):.1f}%)")
print(f"  Missed (unseen address): {trans_miss}/{n_test}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 104 Summary")
print("=" * 72)

nn_lift = 100*nn_hits_top5/total - 500/N
same_leaf_lift = 100*same_leaf_hits/total - 100*random_same_leaf

print(f"""
  NNLM Test — Does the 12D address predict next tokens?

  Same-leaf hit rate:         {100*same_leaf_hits/total:.1f}%  (random: {100*random_same_leaf:.1f}%, lift: {same_leaf_lift:+.1f}pp)
  Nearest-address top-1:      {100*nn_hits_top1/total:.1f}%  (random: {100/N:.1f}%)
  Nearest-address top-5:      {100*nn_hits_top5/total:.1f}%  (random: {500/N:.1f}%, lift: {nn_lift:+.1f}pp)
  Logit cosine NN top-1:      {100*logit_nn_top1/total:.1f}%
  LM greedy top-1:            {100*lm_hits_top1/total:.1f}%  ← oracle upper bound
  Unigram top-1:              {100*unigram_hits/total:.1f}%

  KEY QUESTION: Does the 12D address contain generation-relevant info?
  {'→ YES: Address NN significantly beats random on next-token prediction' if nn_hits_top5/total > 5*1/N else
   '→ MARGINAL: Address NN slightly above random' if nn_hits_top5/total > 2*1/N else
   '→ NO: Address NN is at or near random — address does not predict next token'}

  Interpretation:
  The 12D trie address encodes TOKEN IDENTITY in context, not next-token
  prediction. Nearest-address tokens are semantically related, not
  sequentially likely to follow. For generation, address SIMILARITY ≠
  sequential COMPATIBILITY. The trie is a semantic index, not a
  language model in miniature.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_bigrams": total,
        "same_leaf_pct": 100*same_leaf_hits/total,
        "same_leaf_random_pct": 100*random_same_leaf,
        "nn_top1_pct": 100*nn_hits_top1/total,
        "nn_top5_pct": 100*nn_hits_top5/total,
        "nn_top10_pct": 100*nn_hits_top10/total,
        "lm_top1_pct": 100*lm_hits_top1/total,
        "logit_nn_top1_pct": 100*logit_nn_top1/total,
        "unigram_pct": 100*unigram_hits/total,
        "random_top5_pct": 500/N,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 104 complete.")
