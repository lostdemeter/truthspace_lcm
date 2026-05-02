#!/usr/bin/env python3
"""
Day 112 — Axis-by-Axis Sequential Prediction Sweep

Day 107 showed per-axis sequential signal:
  passive  → 65.9% next-token bit prediction from h_t
  antonym  → 40.7%
  negation → 23.7% (below random!)
  question →  0.0%

Day 111 showed address bigram ≈ word bigram at any scale.

Day 112 asks: which individual axes, or which SUBSETS of axes,
carry the most sequential predictive power?

EXPERIMENT:
  For each axis a (1 of 12):
    1. Build a "1D bigram": P(bit_{t+1,a} | bit_{t,a})
    2. Predict the next token's address using ONLY this axis's bit
    3. Decode to vocabulary word
    4. Measure top-1 next-token accuracy

  For subsets of k axes (k=1..12):
    - Best-k: use top-k axes by single-axis accuracy
    - Random-k: baseline
    - Full-12: baseline

  KEY QUESTIONS:
  1. Which axis has highest sequential signal? (Day 107 was from h_t;
     here we use the ACTUAL prev-token address bit)
  2. Is there a subset of k<12 axes that outperforms full 12D address?
  3. Are "semantic" axes (gender, hypernym) sequentially USELESS?
  4. Are "functional" axes (question, negation, passive) sequentially
     USEFUL — i.e., does text exhibit run-length structure in those bits?
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day112_axis_sweep.json")
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

print("Extracting probe token hidden states ...")
hs_by_layer = {L: [] for L in ALL_LAYERS}
valid_words  = []
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

def classify_all(axis_vec, layer_hs):
    if np.linalg.norm(axis_vec) < 1e-6: return ["U"] * N
    projs = [float(np.dot(layer_hs[i], axis_vec)) for i in range(N)]
    max_p = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * N
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

classes  = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    classes[name] = classify_all(t2_axes[name], hs_by_layer[L])
addresses = ["".join(classes[n][i] for n in AXIS_NAMES_12) for i in range(N)]
addr_int  = np.array([[{"H": 2, "U": 1, "L": 0}[c] for c in a] for a in addresses], dtype=np.int8)

def sent_to_tokens(s): return [w for w in s.split() if w in word_idx]
train_seqs = [sent_to_tokens(s) for s in TRAIN_SENTS]
test_seqs  = [sent_to_tokens(s) for s in TEST_SENTS]
train_addr_seqs = [[addresses[word_idx[w]] for w in seq] for seq in train_seqs]
test_addr_seqs  = [[addresses[word_idx[w]] for w in seq] for seq in test_seqs]

test_bigrams = [(seq[i], seq[i+1]) for seq in test_seqs
                for i in range(len(seq)-1)]
n_test = len(test_bigrams)

# ── Full 12D address bigram baseline ──────────────────────────────────────────
LAMBDA = 0.8

def build_bigram(addr_seqs):
    uni = Counter(a for seq in addr_seqs for a in seq)
    bi  = defaultdict(Counter)
    for seq in addr_seqs:
        for i in range(len(seq)-1): bi[seq[i]][seq[i+1]] += 1
    return uni, bi

def bigram_predict_addr(a1, uni, bi):
    all_a = set(uni.keys())
    if a1 in bi: all_a.update(bi[a1].keys())
    if not all_a: return None
    total_uni = sum(uni.values())
    scores = {}
    for a in all_a:
        b = bi[a1][a] / sum(bi[a1].values()) if a1 in bi and sum(bi[a1].values()) > 0 else 0.0
        u = uni[a] / total_uni if total_uni > 0 else 1/N
        scores[a] = LAMBDA * b + (1-LAMBDA) * u
    return max(scores, key=scores.get)

def decode_addr_subset(partial_addr, axis_subset):
    """Find nearest vocab token minimizing Hamming distance on axis_subset only."""
    idx_list = [AXIS_NAMES_12.index(ax) for ax in axis_subset]
    query = np.array([{"H":2,"U":1,"L":0}[partial_addr[j]] for j in range(len(idx_list))], dtype=np.int8)
    vocab_subset = addr_int[:, idx_list]
    dists = np.sum(vocab_subset != query, axis=1)
    return valid_words[int(np.argmin(dists))]

def decode_full_addr(addr):
    pred = np.array([{"H":2,"U":1,"L":0}[c] for c in addr], dtype=np.int8)
    return valid_words[int(np.argmin(np.sum(addr_int != pred, axis=1)))]

uni_full, bi_full = build_bigram(train_addr_seqs)
full_hits = sum(1 for w1,w2 in test_bigrams
                if decode_full_addr(bigram_predict_addr(addresses[word_idx[w1]], uni_full, bi_full) or addresses[word_idx[w1]]) == w2)
full_acc = 100*full_hits/n_test

print(f"Full 12D address bigram: {full_acc:.1f}% ({full_hits}/{n_test})")
print()

# ── Exp 1: Per-axis bigram (1D) ───────────────────────────────────────────────
print("=" * 72)
print("Exp 1: Single-Axis Sequential Prediction")
print("=" * 72)
print(f"\n  {'axis':>12}  {'vocab_bits':>10}  {'axis_acc%':>10}  "
      f"{'bigram_acc%':>12}  {'bit_diversity':>14}")
print(f"  {'-'*65}")

axis_results = {}
for k, ax_name in enumerate(AXIS_NAMES_12):
    # 1D address: just the single axis bit
    axis_train_seqs = [[a[k] for a in seq] for seq in train_addr_seqs]
    axis_test_seqs  = [[a[k] for a in seq] for seq in test_addr_seqs]

    # Bit distribution in vocabulary
    bit_counts = Counter(addresses[i][k] for i in range(N))
    diversity  = 1 - max(bit_counts.values()) / N  # 0=all same, 0.67=uniform

    # 1D bigram
    uni_1d = Counter(b for seq in axis_train_seqs for b in seq)
    bi_1d  = defaultdict(Counter)
    for seq in axis_train_seqs:
        for i in range(len(seq)-1): bi_1d[seq[i]][seq[i+1]] += 1
    total_1d = sum(uni_1d.values())

    # Evaluate: predict next 1D bit, decode to nearest word
    hits_ax  = 0; hits_word = 0
    for w1, w2 in test_bigrams:
        b1 = addresses[word_idx[w1]][k]
        b2_true = addresses[word_idx[w2]][k]
        # Predict next bit
        all_b = set(uni_1d.keys())
        if b1 in bi_1d: all_b.update(bi_1d[b1].keys())
        if not all_b: continue
        scores_b = {}
        for b in all_b:
            bi_  = bi_1d[b1][b] / sum(bi_1d[b1].values()) if b1 in bi_1d and sum(bi_1d[b1].values()) > 0 else 0.0
            uni_ = uni_1d[b] / total_1d if total_1d > 0 else 1/3
            scores_b[b] = LAMBDA * bi_ + (1-LAMBDA) * uni_
        pred_b = max(scores_b, key=scores_b.get)
        if pred_b == b2_true: hits_ax += 1
        # Decode word from this single axis bit
        pred_word = decode_addr_subset(pred_b, [ax_name])
        if pred_word == w2: hits_word += 1

    ax_acc   = 100*hits_ax/n_test
    word_acc = 100*hits_word/n_test
    axis_results[ax_name] = {"ax_acc": ax_acc, "word_acc": word_acc, "diversity": diversity}
    print(f"  {ax_name:>12}  {str(dict(bit_counts)):>10}  "
          f"{ax_acc:>9.1f}%  {word_acc:>11.1f}%  {diversity:>13.3f}")

print()
print(f"  Full 12D baseline: {full_acc:.1f}%")

# ── Exp 2: Best-k subset prediction ───────────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Best-k Axis Subset Sequential Prediction")
print("=" * 72)

# Sort axes by single-axis word accuracy
sorted_axes = sorted(axis_results.keys(), key=lambda a: -axis_results[a]["word_acc"])
print(f"\n  Axes ranked by word accuracy:")
for ax in sorted_axes:
    print(f"    {ax:>12}: word_acc={axis_results[ax]['word_acc']:.1f}%  ax_acc={axis_results[ax]['ax_acc']:.1f}%")

print()
print(f"  {'k':>3}  {'axes_subset':>45}  {'word_acc%':>10}")
print(f"  {'-'*65}")

subset_results = {}
for k in range(1, 13):
    best_k_axes = sorted_axes[:k]
    # Build k-dimensional bigram
    axis_idxs = [AXIS_NAMES_12.index(ax) for ax in best_k_axes]
    kd_train_seqs = [["".join(a[i] for i in axis_idxs) for a in seq]
                     for seq in train_addr_seqs]
    uni_k = Counter(b for seq in kd_train_seqs for b in seq)
    bi_k  = defaultdict(Counter)
    for seq in kd_train_seqs:
        for i in range(len(seq)-1): bi_k[seq[i]][seq[i+1]] += 1
    total_k = sum(uni_k.values())

    hits = 0
    for w1, w2 in test_bigrams:
        key1 = "".join(addresses[word_idx[w1]][i] for i in axis_idxs)
        all_b = set(uni_k.keys())
        if key1 in bi_k: all_b.update(bi_k[key1].keys())
        if not all_b: continue
        scores = {}
        for b in all_b:
            bi_  = bi_k[key1][b] / sum(bi_k[key1].values()) if key1 in bi_k and sum(bi_k[key1].values()) > 0 else 0.0
            uni_ = uni_k[b] / total_k if total_k > 0 else 0
            scores[b] = LAMBDA * bi_ + (1-LAMBDA) * uni_
        pred_b = max(scores, key=scores.get)
        # Decode: find nearest vocab token on these axes
        query = np.array([{"H":2,"U":1,"L":0}[c] for c in pred_b], dtype=np.int8)
        dists = np.sum(addr_int[:, axis_idxs] != query, axis=1)
        pred_w = valid_words[int(np.argmin(dists))]
        if pred_w == w2: hits += 1

    acc = 100*hits/n_test
    subset_results[k] = {"acc": acc, "axes": best_k_axes[:k]}
    ax_str = "+".join(best_k_axes[:min(k,4)]) + ("..." if k > 4 else "")
    print(f"  {k:>3}  {ax_str:>45}  {acc:>9.1f}%")

print()
print(f"  Full 12D baseline: {full_acc:.1f}%")

# ── Exp 3: Axis type analysis ──────────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 3: Axis Type Analysis")
print("=" * 72)

# Classify axes by their role
SEMANTIC_AXES = ["gender", "hypernym", "synonym", "antonym", "concrete"]
MORPHOLOGICAL  = ["comparative", "plural", "past_tense"]
FUNCTIONAL     = ["passive", "causation", "question", "negation"]

def eval_axis_group(axes, label):
    if not axes: return 0
    axis_idxs = [AXIS_NAMES_12.index(ax) for ax in axes if ax in AXIS_NAMES_12]
    if not axis_idxs: return 0
    kd_train_seqs = [["".join(a[i] for i in axis_idxs) for a in seq]
                     for seq in train_addr_seqs]
    uni_k = Counter(b for seq in kd_train_seqs for b in seq)
    bi_k  = defaultdict(Counter)
    for seq in kd_train_seqs:
        for i in range(len(seq)-1): bi_k[seq[i]][seq[i+1]] += 1
    total_k = sum(uni_k.values())
    hits = 0
    for w1, w2 in test_bigrams:
        key1 = "".join(addresses[word_idx[w1]][i] for i in axis_idxs)
        all_b = set(uni_k.keys())
        if key1 in bi_k: all_b.update(bi_k[key1].keys())
        if not all_b: continue
        scores = {}
        for b in all_b:
            bi_  = bi_k[key1][b] / sum(bi_k[key1].values()) if key1 in bi_k and sum(bi_k[key1].values()) > 0 else 0.0
            uni_ = uni_k[b] / total_k if total_k > 0 else 0
            scores[b] = LAMBDA * bi_ + (1-LAMBDA) * uni_
        pred_b = max(scores, key=scores.get)
        query = np.array([{"H":2,"U":1,"L":0}[c] for c in pred_b], dtype=np.int8)
        dists = np.sum(addr_int[:, axis_idxs] != query, axis=1)
        if valid_words[int(np.argmin(dists))] == w2: hits += 1
    return 100*hits/n_test

print()
for group_name, axes in [("Semantic (5D)", SEMANTIC_AXES),
                          ("Morphological (3D)", MORPHOLOGICAL),
                          ("Functional (4D)", FUNCTIONAL)]:
    acc = eval_axis_group(axes, group_name)
    per_ax = ", ".join(f"{a}={axis_results[a]['word_acc']:.0f}%" for a in axes if a in axis_results)
    print(f"  {group_name:>20}: {acc:.1f}%  [{per_ax}]")

print(f"  {'Full 12D':>20}: {full_acc:.1f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 112 Summary — Axis-by-Axis Sequential Prediction Sweep")
print("=" * 72)

best_ax  = max(axis_results, key=lambda a: axis_results[a]["word_acc"])
worst_ax = min(axis_results, key=lambda a: axis_results[a]["word_acc"])
best_k_acc = max(subset_results[k]["acc"] for k in subset_results)
best_k_val = max(subset_results, key=lambda k: subset_results[k]["acc"])

print(f"""
  Single-axis best:  {best_ax} = {axis_results[best_ax]['word_acc']:.1f}%
  Single-axis worst: {worst_ax} = {axis_results[worst_ax]['word_acc']:.1f}%
  Full 12D bigram:   {full_acc:.1f}%
  Best subset k={best_k_val}: {best_k_acc:.1f}% using {subset_results[best_k_val]['axes']}

  VERDICT:
  {'→ A subset of axes outperforms the full 12D address' if best_k_acc > full_acc + 1 else
   '→ No subset outperforms the full 12D address (compression hurts)'}

  AXIS TYPE RESULTS:
  {'→ Functional axes carry most sequential signal' if eval_axis_group(FUNCTIONAL, '') > eval_axis_group(SEMANTIC_AXES, '') else
   '→ Semantic axes carry most sequential signal'}

  KEY INSIGHT:
  Per-axis word accuracy ranges {axis_results[worst_ax]['word_acc']:.1f}% to {axis_results[best_ax]['word_acc']:.1f}%.
  Full 12D address: {full_acc:.1f}%.
  The best axis {'matches' if abs(axis_results[best_ax]['word_acc'] - full_acc) < 2 else
                 'beats' if axis_results[best_ax]['word_acc'] > full_acc else 'underperforms'} the full address.
  Sequential information is {'distributed across axes' if best_k_acc > axis_results[best_ax]['word_acc'] + 2 else
                              'concentrated in a few axes'}.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "axis_results": axis_results,
        "subset_results": {str(k): v for k, v in subset_results.items()},
        "full_acc": full_acc,
        "sorted_axes": sorted_axes,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 112 complete.")
