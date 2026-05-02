#!/usr/bin/env python3
"""
Day 111 — Large-Corpus Scaling Test

DC 330 identified two bottlenecks for the Geometric LCM:
  1. Address representation too coarse (loses 48pp of LM accuracy)
  2. Data sparsity (50 sentences, 374 addresses, bigram = attention)

Day 111 directly tests bottleneck 2: does bigram accuracy IMPROVE
with more training data? This separates data sparsity from the
address representation limit.

METHOD:
  - Download Pride and Prejudice from Project Gutenberg (~6000 sentences)
  - Extract sentences, tokenize against vocabulary (~420 words)
  - Measure bigram address accuracy vs training corpus size:
    N_train = 50, 100, 200, 500, 1000, 2000, 5000
  - Fixed test set: same 20 sentences as Days 105-110

PREDICTION:
  - If bigram accuracy increases with N_train → data sparsity is the
    primary bottleneck → more data will close the gap
  - If bigram accuracy plateaus near 22% → address representation
    itself is the bottleneck (the 12D address can't predict sequence)
  - The oracle ceiling is fixed at 93.1% regardless of training data
    (it only depends on the encoder+decoder fidelity)
"""
import json, math, re, random
from pathlib import Path
from collections import Counter, defaultdict
import urllib.request
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day111_scaling.json")
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

# ── Download and parse large corpus ──────────────────────────────────────────
CACHE_FILE = SCRIPT_DIR / "day111_corpus_cache.txt"
print("Fetching corpus (Pride and Prejudice, Project Gutenberg) ...")
if CACHE_FILE.exists():
    raw_text = CACHE_FILE.read_text(encoding="utf-8")
    print(f"  Loaded from cache: {len(raw_text):,} chars")
else:
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw_text = resp.read().decode("utf-8", errors="replace")
    CACHE_FILE.write_text(raw_text, encoding="utf-8")
    print(f"  Downloaded: {len(raw_text):,} chars")

def extract_sentences(text, min_words=4, max_words=20):
    """Extract clean sentences with min/max word count."""
    # Strip Project Gutenberg header/footer
    start = text.find("*** START OF THE PROJECT")
    end   = text.find("*** END OF THE PROJECT")
    if start != -1: text = text[start+50:]
    if end   != -1: text = text[:end]
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text)
    sents = []
    for s in raw:
        s = re.sub(r'\s+', ' ', s).strip()
        words = [w.lower().rstrip('.,;:!?"\')') for w in s.split()]
        words = [re.sub(r"[^a-z'-]", '', w) for w in words]
        words = [w for w in words if w]
        if min_words <= len(words) <= max_words:
            sents.append(words)
    return sents

all_sentences = extract_sentences(raw_text)
print(f"  Extracted {len(all_sentences):,} sentences ({min(4,len(all_sentences))}-{max(4,len(all_sentences))} words)")

# ── Load model and compute addresses ─────────────────────────────────────────
print(f"\nLoading {MODEL_ID} ...")
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

def decode_addr(addr):
    pred = np.array([{"H":2,"U":1,"L":0}[c] for c in addr], dtype=np.int8)
    return valid_words[int(np.argmin(np.sum(addr_int != pred, axis=1)))]

# ── Build corpus token sequences ──────────────────────────────────────────────
def sent_words_to_idx(words):
    return [w for w in words if w in word_idx]

vocab_sents = [sent_words_to_idx(s) for s in all_sentences]
vocab_sents = [s for s in vocab_sents if len(s) >= 2]
print(f"Corpus sentences with ≥2 vocab words: {len(vocab_sents):,}")

# Convert to address sequences
addr_sents = [[addresses[word_idx[w]] for w in seq] for seq in vocab_sents]

# Fixed test set
test_seqs = [sent_words_to_idx(s.split()) for s in TEST_SENTS]
test_addr_seqs = [[addresses[word_idx[w]] for w in seq if w in word_idx] for seq in test_seqs]
test_bigrams = [(seq[i], seq[i+1]) for seq in test_seqs
                for i in range(len(seq)-1) if seq[i] in word_idx and seq[i+1] in word_idx]
n_test = len(test_bigrams)

# ── Scaling experiment ────────────────────────────────────────────────────────
LAMBDA = 0.8
N_TRAIN_SIZES = [50, 100, 200, 500, 1000, 2000, min(5000, len(vocab_sents))]
N_TRAIN_SIZES = [s for s in N_TRAIN_SIZES if s <= len(vocab_sents)]

print(f"\nFixed test set: {n_test} bigrams from {len(TEST_SENTS)} test sentences")
print(f"Scaling N_train: {N_TRAIN_SIZES}")
print()
print("=" * 72)
print("Bigram Accuracy vs Training Corpus Size")
print("=" * 72)
print(f"\n  {'N_train':>8}  {'n_addr_bigrams':>15}  {'unique_addrs':>13}  "
      f"{'bigram_acc%':>12}  {'unique_addr_bigrams':>20}")
print(f"  {'-'*80}")

scaling_results = {}
rng = random.Random(42)

for n_train in N_TRAIN_SIZES:
    # Sample n_train sentences from corpus
    sampled = rng.sample(vocab_sents[:min(10000, len(vocab_sents))], n_train) \
              if n_train <= len(vocab_sents) else vocab_sents
    sampled_addr = [[addresses[word_idx[w]] for w in seq] for seq in sampled]

    # Build unigram and bigram
    uni_c     = Counter(a for seq in sampled_addr for a in seq)
    uni_total = sum(uni_c.values())
    bi_c      = defaultdict(Counter)
    for seq in sampled_addr:
        for i in range(len(seq)-1): bi_c[seq[i]][seq[i+1]] += 1
    n_addr_bigrams = sum(sum(v.values()) for v in bi_c.values())
    unique_addrs   = len(uni_c)
    unique_bi      = sum(len(v) for v in bi_c.values())

    # Evaluate on test set
    hits = 0
    for w1, w2 in test_bigrams:
        a1 = addresses[word_idx[w1]]
        all_a = set(uni_c.keys())
        if a1 in bi_c: all_a.update(bi_c[a1].keys())
        if not all_a: continue
        scores = {}
        for a in all_a:
            bi  = bi_c[a1][a] / sum(bi_c[a1].values()) \
                  if a1 in bi_c and sum(bi_c[a1].values()) > 0 else 0.0
            uni = uni_c[a] / uni_total if uni_total > 0 else 1/N
            scores[a] = LAMBDA * bi + (1-LAMBDA) * uni
        pred_addr = max(scores, key=scores.get)
        pred_word = decode_addr(pred_addr)
        if pred_word == w2: hits += 1

    acc = 100*hits/n_test if n_test > 0 else 0
    scaling_results[n_train] = {
        "acc": acc, "n_addr_bigrams": n_addr_bigrams,
        "unique_addrs": unique_addrs, "unique_addr_bigrams": unique_bi,
    }
    print(f"  {n_train:>8}  {n_addr_bigrams:>15,}  {unique_addrs:>13}  "
          f"{acc:>11.1f}%  {unique_bi:>20,}")

# ── Comparison with per-word bigram at max scale ───────────────────────────────
print()
print("=" * 72)
print("Comparison: Address Bigram vs Word Bigram at Max Scale")
print("=" * 72)

n_max = max(N_TRAIN_SIZES)
sampled_max = random.Random(42).sample(vocab_sents[:min(10000,len(vocab_sents))], n_max) \
              if n_max <= len(vocab_sents) else vocab_sents

word_bi = defaultdict(Counter)
for seq in sampled_max:
    for i in range(len(seq)-1): word_bi[seq[i]][seq[i+1]] += 1
word_uni = Counter(w for seq in sampled_max for w in seq)
word_uni_total = sum(word_uni.values())

word_hits = 0
for w1, w2 in test_bigrams:
    all_w = set(word_uni.keys())
    if w1 in word_bi: all_w.update(word_bi[w1].keys())
    scores = {}
    for w in all_w:
        bi  = word_bi[w1][w] / sum(word_bi[w1].values()) \
              if w1 in word_bi and sum(word_bi[w1].values()) > 0 else 0.0
        uni = word_uni[w] / word_uni_total if word_uni_total > 0 else 1/N
        scores[w] = LAMBDA * bi + (1-LAMBDA) * uni
    pred_w = max(scores, key=scores.get)
    if pred_w == w2: word_hits += 1
word_acc = 100*word_hits/n_test

print(f"\n  N_train = {n_max}")
print(f"  Address bigram accuracy:  {scaling_results[n_max]['acc']:.1f}%")
print(f"  Word bigram accuracy:     {word_acc:.1f}%")
print(f"  Address vs word gap:      {scaling_results[n_max]['acc'] - word_acc:+.1f}pp")
print(f"  Oracle ceiling:           93.1%")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 111 Summary — Large-Corpus Scaling Test")
print("=" * 72)

acc_50  = scaling_results.get(50,  {}).get("acc", 0)
acc_max = scaling_results[max(N_TRAIN_SIZES)]["acc"]
gain    = acc_max - acc_50

print(f"""
  Bigram accuracy scaling curve:
    N=50  (Day 105 baseline): {acc_50:.1f}%
    N={max(N_TRAIN_SIZES):<5}  (max scale):     {acc_max:.1f}%
    Gain with more data:      {gain:+.1f}pp

  Word bigram at N={n_max}: {word_acc:.1f}%
  Oracle ceiling:          93.1%

  VERDICT:
  {'→ Scaling HELPS: address bigram improves significantly with more data' if gain > 5 else
   '→ Scaling has MARGINAL effect: address bigram plateaus near 22%' if gain > 1 else
   '→ Scaling does NOT help: address bigram is flat regardless of corpus size'}

  INTERPRETATION:
  {'The data sparsity bottleneck (bottleneck 2) is REAL and addressable.' if gain > 5 else
   'The address representation bottleneck (bottleneck 1) dominates.' if gain < 2 else
   'Both bottlenecks contribute; more data helps marginally.'}

  Address space statistics at N={n_max}:
    Unique addresses:       {scaling_results[n_max]['unique_addrs']}
    Unique address bigrams: {scaling_results[n_max]['unique_addr_bigrams']:,}
    Total address bigrams:  {scaling_results[n_max]['n_addr_bigrams']:,}

  The address representation compresses {N} vocabulary tokens into
  {scaling_results[n_max]['unique_addrs']} unique addresses.
  Average tokens per address: {N / max(1, scaling_results[n_max]['unique_addrs']):.1f}
  (High = coarse compression = information loss)
""")

scaling_results["word_bigram_acc"] = word_acc
scaling_results["n_test"] = n_test
scaling_results["oracle"] = 93.1

with open(OUTPUT_FILE, "w") as f:
    json.dump(scaling_results, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 111 complete.")
