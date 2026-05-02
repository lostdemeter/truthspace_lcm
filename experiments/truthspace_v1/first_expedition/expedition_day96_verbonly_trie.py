#!/usr/bin/env python3
"""
Day 96 — Verb-Only Trie: Testing Address Uniqueness Hypothesis

Day 95 identified two conditions for trie navigability:
  1. Bit discrimination: src_bit ≠ tgt_bit  (achievable with contextualization)
  2. Address uniqueness: tgt must be uniquely near src's flipped address

For run→ran in the full 401-token trie:
  - run has L in past_tense bit (isolated L28)
  - ran has H in past_tense bit
  - But many other tokens also have H in past_tense, and share 11/12 bits
    with run → ran isn't the nearest neighbor

UNIQUENESS HYPOTHESIS: In a verb-only trie (~50 tokens), the address
space is much denser per token. ran should be uniquely identified at
run's flipped address because there are fewer competing tokens.

Additionally, we use CONTEXTUALIZED addresses: embed each verb in
"I [VERB] to the market every single morning" → last-token hidden state.
Past_tense already gives 7/8 isolated bit separation. With context: 8/8.

ALSO TEST: adjective-only trie for comparative (6/6 with contextualization).
Template: "The [ADJ] car"

PREDICTION:
  - past_tense traversal in verb-only trie: > 0/8 (vs 0/8 in full trie)
  - comparative traversal in adj-only trie:  > 2/6 (vs 2/6 in full trie)
  - If prediction holds: address uniqueness IS the bottleneck

TEST DESIGN:
  For past_tense verb-only trie:
    - Vocabulary: all verbs in probe set (base forms + past forms)
    - Address: contextualized "I [VERB] to the market every single morning"
    - Trie: 12D but using ALL sentence-level axes (most axes will be noisy
      for verbs, providing residual context for uniqueness)
    - Traversal: flip past_tense bit, find nearest verb

  For comparative adj-only trie:
    - Vocabulary: all adjectives in probe set
    - Address: "The [ADJ] car"
    - Traversal: flip comparative bit
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day96_verbonly_trie.json")
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

# Verb vocabulary (base + past forms)
VERB_VOCAB = [
    "run", "ran", "walk", "walked", "jump", "jumped", "swim", "swam",
    "fly", "flew", "eat", "ate", "sleep", "slept", "talk", "talked",
    "write", "wrote", "read", "build", "built", "break", "broke",
    "open", "opened", "close", "closed", "start", "started", "stop", "stopped",
    "think", "thought", "know", "knew", "see", "saw", "hear", "heard",
    "feel", "felt", "love", "loved", "hate", "hated", "want", "wanted",
    "give", "gave", "take", "took", "make", "made", "find", "found",
    "lose", "lost", "push", "pushed", "pull", "pulled", "turn", "turned",
    "move", "moved", "go", "went", "come", "came", "fall", "fell",
    "rise", "rose", "grow", "grew", "kill", "killed", "help", "helped",
    "speak", "spoke", "sing", "sang", "drive", "drove", "ride", "rode",
    "throw", "threw", "catch", "caught", "hit", "cut", "put",
]

# Adjective vocabulary (base + comparative)
ADJ_VOCAB = [
    "fast", "faster", "fastest",
    "slow", "slower", "slowest",
    "big", "bigger", "biggest",
    "small", "smaller", "smallest",
    "hot", "hotter", "hottest",
    "cold", "colder", "coldest",
    "old", "older", "oldest",
    "new", "newer", "newest",
    "hard", "harder", "hardest",
    "soft", "softer", "softest",
    "bright", "brighter", "brightest",
    "dark", "darker", "darkest",
    "strong", "stronger", "strongest",
    "weak", "weaker", "weakest",
    "happy", "happier", "happiest",
    "sad", "sadder", "saddest",
    "good", "better", "best",
    "bad", "worse", "worst",
    "long", "longer", "longest",
    "short", "shorter", "shortest",
    "high", "higher", "highest",
    "low", "lower", "lowest",
    "loud", "louder", "loudest",
    "quiet", "quieter", "quietest",
    "clean", "cleaner", "cleanest",
    "dirty", "dirtier", "dirtiest",
    "heavy", "heavier", "heaviest",
    "light", "lighter", "lightest",
    "deep", "deeper", "deepest",
    "wide", "wider", "widest",
    "narrow", "narrower", "narrowest",
    "thick", "thicker", "thickest",
    "thin", "thinner", "thinnest",
]

# Ground truth traversal pairs
GT_PAST_TENSE = [
    ("run","ran"), ("walk","walked"), ("jump","jumped"),
    ("fly","flew"), ("eat","ate"), ("build","built"),
    ("write","wrote"), ("break","broke"),
]
GT_COMPARATIVE = [
    ("fast","faster"), ("big","bigger"), ("slow","slower"),
    ("small","smaller"), ("good","better"), ("bad","worse"),
]

# T2 axis sentence pairs
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

# ── T2 axes ───────────────────────────────────────────────────────────────────
print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_h(s1, [L])[L]; h2 = get_h(s2, [L])[L]
            d = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    if diffs:
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    else:
        t2_axes[name] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {name:<15} L{L}")
print()

def extract_vocab(words, template_fn=None):
    """Extract hidden states for a vocabulary. template_fn: word -> sentence."""
    hs = {}
    for w in words:
        try:
            sent = template_fn(w) if template_fn else " " + w.strip()
            h = get_h(sent, ALL_LAYERS)
            hs[w] = h
        except: pass
    return hs

def classify_and_build(hs_dict, words):
    """Build 12D ternary addresses. Returns (classes_dict, addresses_dict)."""
    classes = {name: {} for name in AXIS_NAMES_12}
    for name in AXIS_NAMES_12:
        L  = DAY78_LAYERS[name]
        ax = t2_axes[name]
        if np.linalg.norm(ax) < 1e-6:
            for w in words: classes[name][w] = "U"
            continue
        projs = {w: float(np.dot(hs_dict[w][L], ax))
                 for w in words if w in hs_dict and L in hs_dict[w]}
        if not projs:
            for w in words: classes[name][w] = "U"; continue
        max_p = float(np.percentile(list(projs.values()), 95))
        if max_p < 1e-6:
            for w in words: classes[name][w] = "U"; continue
        hi, lo = max_p * INV_PHI, max_p * INV_PHI2
        for w in words:
            p = projs.get(w, 0.0)
            classes[name][w] = "H" if p > hi else "L" if p < lo else "U"
    addrs = {w: "".join(classes[n].get(w, "U") for n in AXIS_NAMES_12) for w in words}
    return classes, addrs

FLIP_MAP = {"H": "L", "L": "H", "U": "H"}
def hamming(a, b): return sum(x != y for x, y in zip(a, b))

def run_traversal(addrs, gt_pairs, axis_bit_idx, vocab_words):
    hits = 0; total = 0; details = []
    for src, tgt in gt_pairs:
        if src not in addrs or tgt not in addrs: continue
        fl = list(addrs[src])
        fl[axis_bit_idx] = FLIP_MAP[fl[axis_bit_idx]]
        fl = "".join(fl)
        ranked = sorted([(w, hamming(fl, addrs[w]))
                         for w in vocab_words if w != src], key=lambda x: x[1])
        top5 = [w for w, _ in ranked[:5]]
        rank = next((k for k, (w, _) in enumerate(ranked) if w == tgt), -1)
        hit  = 0 <= rank < 5
        if hit: hits += 1
        total += 1
        src_bit = addrs[src][axis_bit_idx]
        tgt_bit = addrs[tgt][axis_bit_idx]
        details.append({"src": src, "tgt": tgt, "rank": rank,
                         "top5": top5, "hit": hit,
                         "src_bit": src_bit, "tgt_bit": tgt_bit})
        print(f"    {src:>8}({src_bit})\u2192{tgt:>8}({tgt_bit})  "
              f"rank={rank if rank >= 0 else 'miss':>5}  top3={'/'.join(top5[:3])}")
    return hits, total, details

all_results = {}

# ── TEST 1: VERB-ONLY trie, isolated addressing ───────────────────────────────
print("=" * 72)
print("TEST 1: Verb-only trie — ISOLATED addressing")
print("=" * 72)
verb_hs_iso = extract_vocab(VERB_VOCAB)
valid_verbs_iso = [w for w in VERB_VOCAB if w in verb_hs_iso]
print(f"  {len(valid_verbs_iso)} verbs extracted\n")
_, verb_addrs_iso = classify_and_build(verb_hs_iso, valid_verbs_iso)
pt_idx = AXIS_NAMES_12.index("past_tense")
h, t, det = run_traversal(verb_addrs_iso, GT_PAST_TENSE, pt_idx, valid_verbs_iso)
print(f"\n  Isolated verb-only trie: {h}/{t} ({100*h/max(1,t):.0f}%)")
all_results["verb_iso"] = {"hits": h, "total": t, "details": det}

# ── TEST 2: VERB-ONLY trie, contextualized addressing ─────────────────────────
print()
print("=" * 72)
print("TEST 2: Verb-only trie — CONTEXTUALIZED addressing")
print("         Template: 'I [VERB] to the market every single morning'")
print("=" * 72)
verb_tmpl = lambda v: f"I {v} to the market every single morning"
verb_hs_ctx = extract_vocab(VERB_VOCAB, template_fn=verb_tmpl)
valid_verbs_ctx = [w for w in VERB_VOCAB if w in verb_hs_ctx]
print(f"  {len(valid_verbs_ctx)} verbs extracted\n")
_, verb_addrs_ctx = classify_and_build(verb_hs_ctx, valid_verbs_ctx)
h, t, det = run_traversal(verb_addrs_ctx, GT_PAST_TENSE, pt_idx, valid_verbs_ctx)
print(f"\n  Contextual verb-only trie: {h}/{t} ({100*h/max(1,t):.0f}%)")
all_results["verb_ctx"] = {"hits": h, "total": t, "details": det}

# ── TEST 3: ADJ-ONLY trie, isolated addressing ────────────────────────────────
print()
print("=" * 72)
print("TEST 3: Adjective-only trie — ISOLATED addressing")
print("=" * 72)
adj_hs_iso = extract_vocab(ADJ_VOCAB)
valid_adjs_iso = [w for w in ADJ_VOCAB if w in adj_hs_iso]
print(f"  {len(valid_adjs_iso)} adjectives extracted\n")
_, adj_addrs_iso = classify_and_build(adj_hs_iso, valid_adjs_iso)
cmp_idx = AXIS_NAMES_12.index("comparative")
h, t, det = run_traversal(adj_addrs_iso, GT_COMPARATIVE, cmp_idx, valid_adjs_iso)
print(f"\n  Isolated adj-only trie: {h}/{t} ({100*h/max(1,t):.0f}%)")
all_results["adj_iso"] = {"hits": h, "total": t, "details": det}

# ── TEST 4: ADJ-ONLY trie, contextualized addressing ─────────────────────────
print()
print("=" * 72)
print("TEST 4: Adjective-only trie — CONTEXTUALIZED addressing")
print("         Template: 'The [ADJ] car'")
print("=" * 72)
adj_tmpl = lambda a: f"The {a} car"
adj_hs_ctx = extract_vocab(ADJ_VOCAB, template_fn=adj_tmpl)
valid_adjs_ctx = [w for w in ADJ_VOCAB if w in adj_hs_ctx]
print(f"  {len(valid_adjs_ctx)} adjectives extracted\n")
_, adj_addrs_ctx = classify_and_build(adj_hs_ctx, valid_adjs_ctx)
h, t, det = run_traversal(adj_addrs_ctx, GT_COMPARATIVE, cmp_idx, valid_adjs_ctx)
print(f"\n  Contextual adj-only trie: {h}/{t} ({100*h/max(1,t):.0f}%)")
all_results["adj_ctx"] = {"hits": h, "total": t, "details": det}

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 96 Summary")
print("=" * 72)
print(f"""
  Traversal accuracy across trie configurations:

  Config                          hits/total   %
  ────────────────────────────────────────────────
  verb-only isolated (past_tense): {all_results['verb_iso']['hits']}/{all_results['verb_iso']['total']}          {100*all_results['verb_iso']['hits']/max(1,all_results['verb_iso']['total']):.0f}%
  verb-only context  (past_tense): {all_results['verb_ctx']['hits']}/{all_results['verb_ctx']['total']}          {100*all_results['verb_ctx']['hits']/max(1,all_results['verb_ctx']['total']):.0f}%
  full 401-token     (past_tense):  0/8           0%  [Day 92]

  adj-only isolated  (comparative):{all_results['adj_iso']['hits']}/{all_results['adj_iso']['total']}          {100*all_results['adj_iso']['hits']/max(1,all_results['adj_iso']['total']):.0f}%
  adj-only context   (comparative):{all_results['adj_ctx']['hits']}/{all_results['adj_ctx']['total']}          {100*all_results['adj_ctx']['hits']/max(1,all_results['adj_ctx']['total']):.0f}%
  full 401-token     (comparative): 2/6          33%  [Day 92]

  INTERPRETATION:
  If verb-only > full trie:
    → Address uniqueness IS the bottleneck. Dense vocabulary = unique addresses.
    → Solution: POS-stratified tries with contextualized addressing.
  If verb-only ≈ full trie:
    → Address uniqueness is NOT the bottleneck. A different issue prevents traversal.
    → The T2 axes simply don't discriminate verb forms at the individual word level.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"results": all_results}, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 96 complete.")
