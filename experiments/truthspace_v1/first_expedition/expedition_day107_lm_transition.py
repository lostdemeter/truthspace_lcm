#!/usr/bin/env python3
"""
Day 107 — LM Hidden State as Transition Oracle

Day 106 revealed the bottleneck:
  - Oracle (true addr → decoder):  93.1%
  - Bigram address model:           22.0%
  - Gap:                            71pp  ← entirely in transition model

The hypothesis: the LM's hidden state h_t at position t already encodes
"what follows" — this is the essence of causal language modeling.
If true, h_t should be predictive of addr(w_{t+1}).

EXPERIMENTS:

1. Direct correlation: For each bigram (w_t, w_{t+1}), compute:
   - h_t at layer L (last token position, layer 1-28)
   - addr(w_{t+1}) as a 12-dim categorical vector
   - Measure: can a linear probe on h_t predict addr(w_{t+1})?

2. T2 axis projection: Project h_t onto each T2 axis direction.
   Does the projection predict the corresponding axis class of w_{t+1}?
   (E.g., if h_t projects HIGH on gender axis → w_{t+1} has H on gender?)

3. LM logits as address predictor:
   - Run LM on context, get logits over full vocabulary
   - Compute: for each vocabulary token in top-k logits, what's its address?
   - Predict next address as the majority address in top-k LM logits
   - This uses the LM's full predictive distribution as a transition oracle

4. Comparison:
   - Bigram address baseline:       22.0%
   - T2-projection transition:      ???
   - Top-k logit majority vote:     ???
   - Full LM greedy next-word:      ???

PREDICTION:
  - LM logit top-k should dramatically outperform bigram (it IS the LM)
  - T2 projection transition may also beat bigram (partial encoding)
  - If T2 projection beats bigram: confirms LM hidden state encodes next-addr
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day107_lm_transition.json")
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

def get_full_context(sentence, layers):
    """Get hidden states at ALL token positions in a sentence."""
    inp = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # Return (n_positions, hidden_size) for each layer, plus logits
    hs   = {L: out.hidden_states[L][0, :, :].numpy().astype(np.float32) for L in layers}
    logits = out.logits[0, :, :].numpy().astype(np.float32)
    tokens = inp["input_ids"][0].tolist()
    return hs, logits, tokens

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
valid_words  = []
logits_arr_list = []
for word in PROBE_TOKENS:
    try:
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        for L in ALL_LAYERS:
            hs_by_layer[L].append(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
        logits_arr_list.append(out.logits[0, pos, :].numpy().astype(np.float32))
        valid_words.append(word)
    except: pass
for L in ALL_LAYERS:
    hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
logits_arr = np.array(logits_arr_list, dtype=np.float32)
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

# ── Build test bigrams from full context ──────────────────────────────────────
# Each test bigram: (context_sentence, position_of_w_t, w_t, w_{t+1})
print("Building test bigrams from context sentences ...")
test_bigrams_ctx = []  # (sentence, tok_pos, w_t, w_next)
for sent in TEST_SENTS:
    words = sent.split()
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if w1 in word_idx and w2 in word_idx:
            test_bigrams_ctx.append((sent, i, w1, w2))
n_test = len(test_bigrams_ctx)
print(f"  {n_test} test bigrams\n")

# ── Exp 1: T2 projection of h_t predicts addr(w_{t+1}) ───────────────────────
print("=" * 72)
print("Exp 1: T2 axis projection of h_t → predict addr(w_{t+1})")
print("=" * 72)
print()

# For each test bigram, run the full sentence through the LM and extract h_t
# at the appropriate layer for each axis. Then classify h_t on each axis
# and compare to addr(w_{t+1}).

per_axis_correct = {name: 0 for name in AXIS_NAMES_12}
per_axis_total   = {name: 0 for name in AXIS_NAMES_12}
t2_proj_addr_correct = 0
t2_proj_addr_total   = 0

# Process each unique sentence once
processed = {}  # sent → (hs, logits, tokens)

print("  Running LM on test sentences ...")
for sent in TEST_SENTS:
    try:
        hs, logits_ctx, tokens = get_full_context(sent, ALL_LAYERS)
        processed[sent] = (hs, logits_ctx, tokens)
    except Exception as e:
        print(f"  Error on '{sent[:40]}...': {e}")

print()
print("  Evaluating per-axis T2 projection accuracy ...")
for sent, word_pos, w1, w2 in test_bigrams_ctx:
    if sent not in processed: continue
    hs, logits_ctx, tokens = processed[sent]

    # Find the token position in the LM's tokenized output that corresponds
    # to word_pos in the word-split sentence.
    # We'll map word positions to token positions by re-tokenizing.
    words = sent.split()
    prefix = " ".join(words[:word_pos+1])
    prefix_tokens = tok(prefix, return_tensors="pt")["input_ids"][0].tolist()
    t_pos = len(prefix_tokens) - 1  # position of last token of w_t

    if t_pos >= hs[list(hs.keys())[0]].shape[0]: continue

    # True address of w_{t+1}
    true_addr = addresses[word_idx[w2]]

    # Predict each axis of addr(w_{t+1}) by projecting h_t
    pred_chars = []
    for name in AXIS_NAMES_12:
        L = DAY78_LAYERS[name]
        ax_vec = t2_axes[name]
        if hs[L].shape[0] <= t_pos: pred_chars.append("U"); continue
        h_t = hs[L][t_pos]
        proj = float(np.dot(h_t, ax_vec))
        # Use same thresholds as vocabulary encoding
        # (recompute thresholds from full vocabulary projection distribution)
        all_projs = [float(np.dot(hs_by_layer[L][j], ax_vec)) for j in range(N)]
        max_p = float(np.percentile(all_projs, 95))
        if max_p < 1e-6: pred_chars.append("U"); continue
        hi = max_p * INV_PHI; lo = max_p * INV_PHI2
        if proj > hi:   pred_chars.append("H")
        elif proj < lo: pred_chars.append("L")
        else:           pred_chars.append("U")
        # Per-axis accuracy
        per_axis_correct[name] += 1 if pred_chars[-1] == true_addr[AXIS_NAMES_12.index(name)] else 0
        per_axis_total[name]   += 1

    pred_addr = "".join(pred_chars)
    # Full address match
    if pred_addr == true_addr: t2_proj_addr_correct += 1
    t2_proj_addr_total += 1

print(f"  T2 projection full-address accuracy: "
      f"{t2_proj_addr_correct}/{t2_proj_addr_total} "
      f"({100*t2_proj_addr_correct/max(1,t2_proj_addr_total):.1f}%)\n")
print(f"  Per-axis accuracy (h_t → predicted axis class of w_{{t+1}}):")
print(f"  {'axis':>15}  {'correct':>7}  {'total':>7}  {'acc%':>6}")
print(f"  {'-'*45}")
for name in AXIS_NAMES_12:
    n_c = per_axis_correct[name]; n_t = per_axis_total[name]
    acc = 100*n_c/n_t if n_t > 0 else 0
    print(f"  {name:>15}  {n_c:>7}  {n_t:>7}  {acc:>5.1f}%")

# ── Exp 2: LM logit top-k majority vote for next address ─────────────────────
print()
print("=" * 72)
print("Exp 2: LM logit top-k → majority vote for next address")
print("=" * 72)
print()

# For each test bigram: take the LM logits at position t (from full context run),
# find the top-k vocabulary tokens with highest logit,
# decode each to its trie address, take the plurality address as prediction.
TOP_K_VALS = [1, 5, 10, 20, 50]
hits_by_k = {k: 0 for k in TOP_K_VALS}
total_by_k = {k: 0 for k in TOP_K_VALS}
lm_word_hits = 0  # direct LM greedy word prediction

for sent, word_pos, w1, w2 in test_bigrams_ctx:
    if sent not in processed: continue
    hs, logits_ctx, tokens = processed[sent]
    words = sent.split()
    prefix = " ".join(words[:word_pos+1])
    prefix_toks = tok(prefix, return_tensors="pt")["input_ids"][0].tolist()
    t_pos = len(prefix_toks) - 1
    if t_pos >= logits_ctx.shape[0]: continue

    logits_t = logits_ctx[t_pos]  # logits over full vocab at position t
    true_addr = addresses[word_idx[w2]]

    # LM greedy word prediction
    top1_tok_id = int(np.argmax(logits_t))
    top1_word   = tok.decode([top1_tok_id]).strip().lower()
    if top1_word == w2: lm_word_hits += 1

    for K in TOP_K_VALS:
        total_by_k[K] += 1
        top_k_ids = np.argsort(logits_t)[::-1][:K]
        # For each top-k token id, decode to word, look up address
        candidate_addrs = []
        for tid in top_k_ids:
            w = tok.decode([tid]).strip().lower()
            if w in word_idx:
                candidate_addrs.append(addresses[word_idx[w]])
        if not candidate_addrs: continue
        # Majority vote: most common address among candidates
        voted_addr = Counter(candidate_addrs).most_common(1)[0][0]
        if voted_addr == true_addr: hits_by_k[K] += 1

print(f"  LM greedy word (top-1 logit):      "
      f"{lm_word_hits}/{n_test} ({100*lm_word_hits/n_test:.1f}%)")
print()
print(f"  {'K':>6}  {'hits':>6}  {'total':>6}  {'addr_acc%':>10}  note")
print(f"  {'-'*55}")
for K in TOP_K_VALS:
    h = hits_by_k[K]; t = total_by_k[K]
    acc = 100*h/t if t > 0 else 0
    note = " ← predicted addr from majority vote"
    print(f"  {K:>6}  {h:>6}  {t:>6}  {acc:>9.1f}%  {note}")

# ── Exp 3: LM logit top-k → decode word → accuracy ────────────────────────────
print()
print("=" * 72)
print("Exp 3: LM logit top-k word → direct word accuracy")
print("=" * 72)
print()
word_hits_by_k = {k: 0 for k in TOP_K_VALS}
for sent, word_pos, w1, w2 in test_bigrams_ctx:
    if sent not in processed: continue
    hs, logits_ctx, tokens = processed[sent]
    words = sent.split()
    prefix = " ".join(words[:word_pos+1])
    prefix_toks = tok(prefix, return_tensors="pt")["input_ids"][0].tolist()
    t_pos = len(prefix_toks) - 1
    if t_pos >= logits_ctx.shape[0]: continue
    logits_t = logits_ctx[t_pos]
    for K in TOP_K_VALS:
        top_k_ids = np.argsort(logits_t)[::-1][:K]
        top_k_words = [tok.decode([tid]).strip().lower() for tid in top_k_ids]
        if w2 in top_k_words: word_hits_by_k[K] += 1

print(f"  {'K':>6}  {'hits':>6}  {'total':>6}  {'word_acc%':>10}")
print(f"  {'-'*38}")
for K in TOP_K_VALS:
    h = word_hits_by_k[K]; t = n_test
    acc = 100*h/t if t > 0 else 0
    print(f"  {K:>6}  {h:>6}  {t:>6}  {acc:>9.1f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 107 Summary — LM Hidden State as Transition Oracle")
print("=" * 72)

# Hamming accuracy of T2 projection
proj_acc = 100*t2_proj_addr_correct/max(1,t2_proj_addr_total)
lm_top1_addr = 100*hits_by_k[1]/max(1,total_by_k[1])
lm_top5_addr = 100*hits_by_k[5]/max(1,total_by_k[5])
lm_top5_word = 100*word_hits_by_k[5]/n_test
lm_top1_word = 100*lm_word_hits/n_test
bigram_baseline = 22.0
oracle_ceiling  = 93.1

print(f"""
  Experiment:      Transition oracle quality (addr accuracy)
  Bigram baseline:  22.0%  (Day 105)
  Oracle ceiling:   93.1%  (Day 106)

  Method                           addr_acc%   word_acc%
  -------------------------------------------------------
  T2 projection (full addr match)  {proj_acc:>8.1f}%   N/A
  LM top-1 majority vote           {lm_top1_addr:>8.1f}%   {lm_top1_word:>7.1f}%
  LM top-5 majority vote           {lm_top5_addr:>8.1f}%   {lm_top5_word:>7.1f}%
  LM top-20 majority vote          {100*hits_by_k[20]/max(1,total_by_k[20]):>8.1f}%   {100*word_hits_by_k[20]/n_test:>7.1f}%
  LM top-50 majority vote          {100*hits_by_k[50]/max(1,total_by_k[50]):>8.1f}%   {100*word_hits_by_k[50]/n_test:>7.1f}%
  Bigram address (Day 105)          22.0%   22.0%
  Random                             0.2%    0.2%

  KEY FINDING:
  LM top-k addr accuracy = {lm_top5_addr:.1f}% (top-5), vs bigram 22.0%
  → {'LM logits ARE a superior transition oracle (+' + f'{lm_top5_addr - bigram_baseline:.0f}pp)' if lm_top5_addr > bigram_baseline + 5 else
     'LM logits are comparable to bigram (' + f'{lm_top5_addr:.0f}% vs 22%)' if lm_top5_addr > bigram_baseline - 3 else
     'LM logits UNDERPERFORM bigram (context mismatch?)'}

  T2 projection addr accuracy = {proj_acc:.1f}%
  → {'T2 projection predicts next address: hidden state encodes future addr' if proj_acc > 30 else
     'T2 projection is near chance: hidden state does NOT encode next addr directly' if proj_acc < 15 else
     'T2 projection marginally above chance: weak next-addr signal'}

  INTERPRETATION:
  The LM's logits at position t are the LM's own prediction of what
  follows — using them directly as the transition oracle bypasses the
  bigram and uses the full LM distribution. If LM top-k >> bigram,
  the LM IS the transition model, and the geometric LCM's bottleneck
  is that bigrams are a poor substitute for attention.

  To close the 22%→93% gap: replace the bigram with a single forward
  pass of the LM to get the transition address distribution.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_test": n_test,
        "t2_proj_full_addr_acc": proj_acc,
        "per_axis_acc": {n: 100*per_axis_correct[n]/max(1,per_axis_total[n]) for n in AXIS_NAMES_12},
        "lm_top1_word_acc": lm_top1_word,
        "lm_top5_word_acc": lm_top5_word,
        "lm_top1_addr_acc": lm_top1_addr,
        "lm_top5_addr_acc": lm_top5_addr,
        "hits_by_k_addr": {str(k): hits_by_k[k] for k in TOP_K_VALS},
        "hits_by_k_word": {str(k): word_hits_by_k[k] for k in TOP_K_VALS},
        "bigram_baseline": bigram_baseline,
        "oracle_ceiling": oracle_ceiling,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 107 complete.")
