#!/usr/bin/env python3
"""
Day 110 — Attention-over-Addresses

DC 330 identified the open problem: the Geometric LCM needs an attention-
equivalent transition model over the address sequence to close the
22% → 93% gap.

ARCHITECTURE:
  Encoder:    word → 12D ternary addr   (trie, deterministic)
  Transition: attention over addr history → next addr  ← THIS
  Decoder:    addr → word               (empirical P(w|addr))

ADDRESS ENCODING FOR ATTENTION:
  12D ternary (H/U/L) → 36D per-axis one-hot vector
  axis i class j → position 3*i + j set to 1
  This is explicit, compact, preserves axis independence.

TRANSITION MODEL:
  A single-layer causal self-attention block (tiny transformer)
  operating on sequences of 36D address embeddings.
  - Input:  sequence of T addr embeddings, shape (T, 36)
  - Output: at each position, predict next addr distribution
  - Train:  cross-entropy loss on next-address prediction

COMPARISON:
  1. Bigram baseline:            22.0%  (Day 105)
  2. Word-level bigram oracle:   22.6%  (Day 105)
  3. Attention 1-layer, 1-head:  ???
  4. Attention 1-layer, 4-head:  ???
  5. Attention 2-layer, 4-head:  ???

KEY QUESTION:
  Does learned attention over the address sequence improve next-address
  accuracy over the bigram, using the same 50-sentence training corpus?
  If yes → attention-over-addresses is a viable transition model.
  If no  → the address representation itself is the bottleneck.
"""
import json, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day110_attn_addr.json")
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

def sent_to_tokens(s): return [w for w in s.split() if w in word_idx]
train_seqs = [sent_to_tokens(s) for s in TRAIN_SENTS]
test_seqs  = [sent_to_tokens(s) for s in TEST_SENTS]
train_addr_seqs = [[addresses[word_idx[w]] for w in seq] for seq in train_seqs]
test_addr_seqs  = [[addresses[word_idx[w]] for w in seq] for seq in test_seqs]

# ── Address encoding: 12D ternary → 36D one-hot + learned embeddings ─────────
CLASS_MAP = {"H": 0, "U": 1, "L": 2}
ADDR_DIM  = 36   # 12 axes × 3 classes

def addr_to_vec(addr_str):
    """12-char address string → 36D one-hot float tensor."""
    v = torch.zeros(ADDR_DIM)
    for i, c in enumerate(addr_str):
        v[3*i + CLASS_MAP[c]] = 1.0
    return v

# Build unique address set and address→int mapping
all_train_addrs = sorted(set(a for seq in train_addr_seqs for a in seq))
all_vocab_addrs = sorted(set(addresses))
all_addrs_set   = sorted(set(all_train_addrs + all_vocab_addrs))
addr_to_id      = {a: i for i, a in enumerate(all_addrs_set)}
id_to_addr      = {i: a for a, i in addr_to_id.items()}
VOCAB_SIZE      = len(all_addrs_set)
print(f"Address vocabulary: {VOCAB_SIZE} unique addresses\n")

# ── Bigram baseline (for comparison) ──────────────────────────────────────────
uni_c    = Counter(a for seq in train_addr_seqs for a in seq)
uni_total = sum(uni_c.values())
bi_c     = defaultdict(Counter)
for seq in train_addr_seqs:
    for i in range(len(seq)-1): bi_c[seq[i]][seq[i+1]] += 1

LAMBDA = 0.8
def bigram_predict(prev_addr):
    all_a = set(uni_c.keys())
    if prev_addr in bi_c: all_a.update(bi_c[prev_addr].keys())
    scores = {}
    for a in all_a:
        bi  = bi_c[prev_addr][a] / sum(bi_c[prev_addr].values()) \
              if prev_addr in bi_c and sum(bi_c[prev_addr].values()) > 0 else 0.0
        uni = uni_c[a] / uni_total if uni_total > 0 else 1/VOCAB_SIZE
        scores[a] = LAMBDA * bi + (1-LAMBDA) * uni
    return max(scores, key=scores.get) if scores else None

# ── Decoder: nearest-Hamming word ─────────────────────────────────────────────
def decode_addr(addr):
    pred_arr = np.array([{"H":2,"U":1,"L":0}[c] for c in addr], dtype=np.int8)
    dists    = np.sum(addr_int != pred_arr, axis=1)
    return valid_words[int(np.argmin(dists))]

# ── Tiny transformer for attention-over-addresses ─────────────────────────────
class AddrAttention(nn.Module):
    """Causal self-attention block operating on 36D address embeddings."""
    def __init__(self, d_model=64, n_heads=4, n_layers=1, dropout=0.1, vocab=VOCAB_SIZE):
        super().__init__()
        self.embed    = nn.Linear(ADDR_DIM, d_model, bias=False)
        self.pos_enc  = nn.Embedding(64, d_model)   # up to 64 positions
        layers = []
        for _ in range(n_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4, dropout=dropout,
                batch_first=True, norm_first=True,
            ))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True,
        ), num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x_vecs, src_key_padding_mask=None):
        # x_vecs: (B, T, ADDR_DIM)
        T = x_vecs.shape[1]
        emb = self.embed(x_vecs)
        pos = self.pos_enc(torch.arange(T, device=x_vecs.device)).unsqueeze(0)
        emb = emb + pos
        causal_mask = torch.triu(torch.ones(T, T, device=x_vecs.device), diagonal=1).bool()
        out = self.transformer(emb, mask=causal_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               is_causal=True)
        return self.head(out)  # (B, T, vocab)

def make_batches(addr_seqs, max_len=32):
    """Make (input, target) pairs for next-address prediction."""
    inputs, targets = [], []
    for seq in addr_seqs:
        ids = [addr_to_id[a] for a in seq if a in addr_to_id]
        if len(ids) < 2: continue
        for start in range(0, len(ids)-1, max_len):
            chunk = ids[start:start+max_len+1]
            if len(chunk) < 2: continue
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
    return inputs, targets

def pad_batch(seqs):
    max_len = max(len(s) for s in seqs)
    padded  = [s + [0]*(max_len-len(s)) for s in seqs]
    mask    = [[False]*len(s) + [True]*(max_len-len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

train_in, train_tgt = make_batches(train_addr_seqs)
test_in,  test_tgt  = make_batches(test_addr_seqs)

def addr_vec_batch(id_batch):
    """Convert batch of id sequences to one-hot address vectors."""
    vecs = []
    for ids in id_batch:
        row = []
        for aid in ids:
            row.append(addr_to_vec(id_to_addr[aid]))
        vecs.append(torch.stack(row))
    return torch.stack(vecs)  # (B, T, ADDR_DIM)

# ── Train and evaluate attention models ───────────────────────────────────────
CONFIGS = [
    {"name": "Attn 1L-1H  d=32",  "n_layers": 1, "n_heads": 1, "d_model": 32},
    {"name": "Attn 1L-4H  d=64",  "n_layers": 1, "n_heads": 4, "d_model": 64},
    {"name": "Attn 2L-4H  d=64",  "n_layers": 2, "n_heads": 4, "d_model": 64},
    {"name": "Attn 2L-4H  d=128", "n_layers": 2, "n_heads": 4, "d_model": 128},
]
EPOCHS     = 200
LR         = 3e-3
BATCH_SIZE = 8

results_table = {}

for cfg in CONFIGS:
    print(f"Training {cfg['name']} ...")
    attn_model = AddrAttention(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], vocab=VOCAB_SIZE
    )
    opt = torch.optim.Adam(attn_model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Training
    attn_model.train()
    indices = list(range(len(train_in)))
    for epoch in range(EPOCHS):
        random.shuffle(indices)
        total_loss = 0.0; n_batches = 0
        for b_start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[b_start:b_start+BATCH_SIZE]
            inp_seqs = [train_in[i]  for i in batch_idx]
            tgt_seqs = [train_tgt[i] for i in batch_idx]
            inp_ids, inp_mask = pad_batch(inp_seqs)
            tgt_ids, _        = pad_batch(tgt_seqs)
            inp_vecs = addr_vec_batch([[j for j in seq] for seq in inp_ids.tolist()])
            logits   = attn_model(inp_vecs, src_key_padding_mask=inp_mask)
            loss     = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tgt_ids.reshape(-1),
                ignore_index=0,
            )
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        sched.step()

    # Evaluation on test bigrams
    attn_model.eval()
    test_bigrams = [(seq[i], seq[i+1]) for seq in test_seqs
                    for i in range(len(seq)-1)
                    if seq[i] in word_idx and seq[i+1] in word_idx]
    n_test = len(test_bigrams)

    # For attention model: predict next address from single-token context
    attn_hits_top1 = 0; bigram_hits = 0
    for w1, w2 in test_bigrams:
        a1 = addresses[word_idx[w1]]; a2 = addresses[word_idx[w2]]
        # Attention: condition on single address
        if a1 in addr_to_id:
            inp_vec = addr_to_vec(a1).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logit = attn_model(inp_vec)[0, 0, :]
            pred_aid = int(torch.argmax(logit).item())
            pred_addr_attn = id_to_addr.get(pred_aid, a1)
            pred_word_attn = decode_addr(pred_addr_attn)
            if pred_word_attn == w2: attn_hits_top1 += 1
        # Bigram
        pred_addr_bi = bigram_predict(a1)
        if pred_addr_bi and decode_addr(pred_addr_bi) == w2: bigram_hits += 1

    attn_acc = 100*attn_hits_top1/n_test if n_test > 0 else 0
    bi_acc   = 100*bigram_hits/n_test if n_test > 0 else 0
    n_params = sum(p.numel() for p in attn_model.parameters())
    print(f"  {cfg['name']:25s}  attn={attn_acc:.1f}%  bigram={bi_acc:.1f}%  "
          f"params={n_params}")
    results_table[cfg['name']] = {
        "attn_acc": attn_acc, "bigram_acc": bi_acc, "n_params": n_params,
        "n_test": n_test, "epochs": EPOCHS,
    }

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 110 Summary — Attention-over-Addresses")
print("=" * 72)
print()
print(f"  Bigram baseline (Day 105):  22.0%")
print(f"  Oracle ceiling (Day 106):   93.1%")
print()
print(f"  {'Model':30s}  {'attn_acc%':>10}  {'vs bigram':>10}  params")
print(f"  {'-'*65}")
for name, r in results_table.items():
    delta = r["attn_acc"] - 22.0
    print(f"  {name:30s}  {r['attn_acc']:>9.1f}%  {delta:>+9.1f}pp  {r['n_params']}")

best_name = max(results_table, key=lambda k: results_table[k]["attn_acc"])
best_acc  = results_table[best_name]["attn_acc"]
print(f"""
  Best attention model: {best_name} at {best_acc:.1f}%
  Bigram baseline:       22.0%
  Gap to oracle:         {93.1 - best_acc:.1f}pp

  VERDICT:
  {'→ Attention-over-addresses SIGNIFICANTLY beats bigram (+' + f'{best_acc-22.0:.0f}pp)' if best_acc > 27 else
   '→ Attention-over-addresses marginally beats bigram' if best_acc > 22.5 else
   '→ Attention-over-addresses does NOT beat bigram (underfitting/overfitting)'}

  KEY FINDING:
  The attention module is operating on a {'compressed' if best_acc > 27 else 'sparse'} address sequence.
  With {len(train_in)} training sequences and VOCAB_SIZE={VOCAB_SIZE} addresses,
  the attention model {'learns sequential structure' if best_acc > 25 else 'struggles to generalize (data sparsity)'}.

  Next step:
  {'→ Increase training data or use transfer from LM attention weights' if best_acc < 25 else
   '→ Test with longer context window (full sentence, not just bigram)'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"results": results_table, "bigram_baseline": 22.0, "oracle": 93.1}, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 110 complete.")
