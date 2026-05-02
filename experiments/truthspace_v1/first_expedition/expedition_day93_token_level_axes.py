#!/usr/bin/env python3
"""
Day 93 — Token-Level Axes: Direct Token Pairs for Navigable Trie

Day 92 discovered two axis types:
  CATEGORY axes (sentence-level, navigable):  gender 50%, comparative 33%
  RELATIONAL axes (sentence-level, NOT nav): past_tense 0%, plural 0%, antonym 0%

Hypothesis: relational axes fail because sentence-difference T2 vectors
capture HOW A SENTENCE CHANGES when a word is transformed, not what
makes the word itself different. The token " run" and " ran" in isolation
have similar hidden states despite their different tenses — the tense
information is sentence-contextual.

FIX: Build T2 axes from DIRECT TOKEN PAIRS:
  past_tense: T2 = mean(hidden(" ran") - hidden(" run")) over many pairs
  plural:     T2 = mean(hidden(" dogs") - hidden(" dog")) over many pairs
  antonym:    T2 = mean(hidden(" cold") - hidden(" hot")) over many pairs

These token-level axes should capture the actual hidden-state direction
that separates tense, number, and polarity at the TOKEN EMBEDDING LEVEL.

TEST: For each axis, compute the token-level T2 axis, then build a 12D
trie where the relational axes are REPLACED with token-level variants.
Re-run the address traversal and compare navigability.

ALSO TEST: Is there a single universal "morphological complexity" axis
that separates base forms from derived forms (run<ran, dog<dogs, fast<faster)?
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day93_token_level_axes.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

# Token-level pairs for building T2 axes directly from token hidden states
TOKEN_PAIRS = {
    # Sentence-level (Day 78 originals)
    "gender_sent": [
        ("The king ruled with great wisdom",   "The queen ruled with great wisdom"),
        ("A man walked through the forest",    "A woman walked through the forest"),
        ("The boy kicked the ball hard",       "The girl kicked the ball hard"),
        ("His brother arrived at the party",   "His sister arrived at the party"),
        ("The father worked to feed family",   "The mother worked to feed family"),
        ("A son was born in the winter",       "A daughter was born in the winter"),
        ("The prince rode across the land",    "The princess rode across the land"),
        ("The actor played a leading role",    "The actress played a leading role"),
    ],
    # Token-level: direct hidden state differences at last token position
    "gender_tok": [
        (" king",     " queen"),
        (" man",      " woman"),
        (" boy",      " girl"),
        (" brother",  " sister"),
        (" father",   " mother"),
        (" son",      " daughter"),
        (" prince",   " princess"),
        (" actor",    " actress"),
        (" husband",  " wife"),
        (" he",       " she"),
        (" his",      " her"),
    ],
    "past_tense_tok": [
        (" run",   " ran"),
        (" walk",  " walked"),
        (" jump",  " jumped"),
        (" fly",   " flew"),
        (" eat",   " ate"),
        (" see",   " saw"),
        (" build", " built"),
        (" write", " wrote"),
        (" break", " broke"),
        (" hear",  " heard"),
        (" go",    " went"),
        (" come",  " came"),
        (" give",  " gave"),
        (" find",  " found"),
        (" make",  " made"),
        (" fall",  " fell"),
    ],
    "plural_tok": [
        (" dog",   " dogs"),
        (" cat",   " cats"),
        (" tree",  " trees"),
        (" bird",  " birds"),
        (" hand",  " hands"),
        (" eye",   " eyes"),
        (" house", " houses"),
        (" horse", " horses"),
        (" man",   " men"),
        (" woman", " women"),
        (" child", " children"),
        (" foot",  " feet"),
    ],
    "comparative_tok": [
        (" fast",   " faster"),
        (" slow",   " slower"),
        (" big",    " bigger"),
        (" small",  " smaller"),
        (" hot",    " hotter"),
        (" cold",   " colder"),
        (" old",    " older"),
        (" young",  " younger"),
        (" good",   " better"),
        (" bad",    " worse"),
        (" hard",   " harder"),
        (" soft",   " softer"),
    ],
    "antonym_tok": [
        (" hot",    " cold"),
        (" big",    " small"),
        (" fast",   " slow"),
        (" hard",   " soft"),
        (" happy",  " sad"),
        (" strong", " weak"),
        (" good",   " bad"),
        (" old",    " new"),
        (" high",   " low"),
        (" long",   " short"),
        (" dark",   " bright"),
        (" heavy",  " light"),
    ],
}

# Ground truth traversal pairs
GROUND_TRUTH = {
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

PROBE_TOKENS = list(dict.fromkeys([
    "dog", "cat", "bird", "fish", "horse", "wolf", "lion", "tiger",
    "elephant", "mouse", "rabbit", "deer", "bear", "fox", "eagle",
    "whale", "shark", "frog", "ant", "bee", "snake", "monkey", "cow",
    "pig", "sheep", "goat", "duck", "hen", "crow", "owl",
    "turtle", "lizard", "crab", "lobster", "octopus", "beetle",
    "butterfly", "worm", "fly", "mosquito", "cricket", "spider",
    "salmon", "tuna", "herring", "sparrow", "robin", "finch", "parrot",
    "tree", "flower", "rock", "stone", "wood", "leaf", "grass", "root",
    "river", "mountain", "ocean", "forest", "desert", "cloud", "rain",
    "snow", "wind", "sun", "moon", "star", "sky", "earth", "soil",
    "seed", "branch", "bark", "thorn", "moss", "mushroom", "coral",
    "house", "door", "window", "table", "chair", "book", "cup", "key",
    "car", "road", "bridge", "boat", "ship", "plane", "train", "bike",
    "knife", "fork", "spoon", "plate", "bowl", "glass", "bottle", "box",
    "bag", "rope", "wire", "nail", "hammer", "wheel", "clock", "lamp",
    "pen", "paper", "cloth", "thread", "button", "ring", "coin", "mirror",
    "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg",
    "head", "heart", "blood", "bone", "skin", "hair", "finger", "toe",
    "back", "chest", "neck", "shoulder",
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "talk",
    "write", "read", "build", "break", "open", "close", "start", "stop",
    "think", "know", "see", "hear", "feel", "love", "hate", "want",
    "give", "take", "make", "find", "lose", "push", "pull", "turn",
    "move", "go", "come", "fall", "rise", "grow", "kill", "help",
    "ran", "walked", "jumped", "flew", "ate", "saw", "heard", "broke",
    "built", "wrote",
    "fast", "slow", "big", "small", "hot", "cold", "old", "new",
    "hard", "soft", "bright", "dark", "strong", "weak", "happy", "sad",
    "good", "bad", "right", "wrong", "high", "low", "long", "short",
    "wide", "narrow", "deep", "shallow", "thick", "thin", "heavy", "light",
    "clean", "dirty", "sweet", "bitter", "sharp", "dull", "loud", "quiet",
    "faster", "slower", "bigger", "smaller", "better", "worse",
    "biggest", "smallest", "best", "worst",
    "quickly", "slowly", "often", "never", "always", "very", "quite",
    "really", "just", "still",
    "the", "a", "and", "or", "not", "is", "was", "in", "on", "of",
    "to", "from", "with", "for", "he", "she", "it", "they", "we",
    "I", "you", "his", "her", "their", "my", "your", "its", "our",
    "but", "if",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand",
    "many", "few", "more", "less", "most", "least", "all", "some",
    "king", "queen", "man", "woman", "boy", "girl", "child", "parent",
    "brother", "sister", "father", "mother", "son", "daughter",
    "husband", "wife", "prince", "princess", "actor", "actress",
    "red", "blue", "green", "yellow", "white", "black", "brown",
    "orange", "purple", "pink", "gray", "gold",
    "love", "hate", "truth", "beauty", "freedom", "power",
    "time", "space", "mind", "body", "soul", "life", "death", "hope",
    "fear", "joy", "pain", "trust", "faith", "peace",
    "war", "law", "right", "duty", "honor", "shame", "pride", "guilt",
    "anger", "grief",
    "city", "town", "village", "country", "island", "valley", "cave",
    "bridge", "castle", "market", "church", "school", "hospital",
    "garden", "field", "park", "lake", "coast", "cliff", "path",
    "bread", "meat", "fruit", "milk", "water", "fire", "oil", "salt",
    "sugar", "coffee", "wine", "beer", "tea", "egg", "cheese",
    "dogs", "cats", "trees", "birds", "horses", "men", "women",
    "children", "hands", "eyes",
]))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

TARGET_LAYER = 28

def get_h28(text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    h   = out.hidden_states[TARGET_LAYER][0, pos, :].numpy().astype(np.float32)
    lg  = out.logits[0, pos, :].numpy().astype(np.float32)
    return h, lg

def compute_axis(pairs, is_sentence=False):
    """Compute T2 axis. pairs: list of (src, tgt) strings."""
    diffs = []
    for s1, s2 in pairs:
        try:
            h1, _ = get_h28(s1); h2, _ = get_h28(s2)
            d = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    if not diffs: return np.zeros(hidden_size, dtype=np.float32)
    v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
    return (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

# ── Compute all axes ──────────────────────────────────────────────────────────
print("Computing sentence-level and token-level axes at L28 ...")
axes = {}
for name, pairs in TOKEN_PAIRS.items():
    axes[name] = compute_axis(pairs)
    print(f"  {name:<25}")

print()

# ── Compare sentence vs token axes ───────────────────────────────────────────
def cos_angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return float("nan")
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1, 1)))))

print("=" * 60)
print("Angle between sentence-level and token-level axes")
print("=" * 60)
AXIS_PAIRS_COMPARE = [
    ("gender_sent",    "gender_tok"),
    ("past_tense_tok", None),  # no sentence-level counterpart directly
]
angle_gend = cos_angle(axes["gender_sent"], axes["gender_tok"])
print(f"  gender: sentence vs token axis angle = {angle_gend:.1f}°")
print(f"  (small angle → same information; 90° → completely different)")
print()

# ── Extract token hidden states ───────────────────────────────────────────────
print("Extracting h28 + logits for all probe tokens ...")
hiddens = []; logits_all = []; valid_words = []
for word in PROBE_TOKENS:
    try:
        h, lg = get_h28(" " + word.strip())
        hiddens.append(h); logits_all.append(lg); valid_words.append(word)
    except: pass
hiddens    = np.array(hiddens, dtype=np.float32)
logits_all = np.array(logits_all, dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

# ── Per-axis thresholding ─────────────────────────────────────────────────────
def classify(axis_vec, hiddens):
    if np.linalg.norm(axis_vec) < 1e-6: return ["U"] * len(hiddens)
    projs  = [float(np.dot(h, axis_vec)) for h in hiddens]
    max_p  = float(np.percentile(projs, 95))
    if max_p < 1e-6: return ["U"] * len(hiddens)
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

def run_traversal(axis_name, axis_vec, gt_pairs, label=""):
    """Run address traversal test for a single axis."""
    classes = classify(axis_vec, hiddens)
    # Build simple 1D addresses for focused test
    hits = 0; total = 0; pair_results = []
    for src, tgt in gt_pairs:
        if src not in word_idx or tgt not in word_idx: continue
        si, ti = word_idx[src], word_idx[tgt]
        src_bit = classes[si]; tgt_bit = classes[ti]
        # For traversal: find tokens with different bit than src (prefer tgt's bit)
        target_bit = tgt_bit if tgt_bit != src_bit else ("H" if src_bit == "L" else "L")
        # Rank by projection value in target_bit direction
        projs_all = [float(np.dot(hiddens[j], axis_vec)) for j in range(N)]
        max_p = float(np.percentile(projs_all, 95))
        if src_bit == "L":
            # Navigate toward H: rank by decreasing projection
            ranked = sorted([j for j in range(N) if j != si], key=lambda j: -projs_all[j])
        else:
            # Navigate toward L: rank by increasing projection
            ranked = sorted([j for j in range(N) if j != si], key=lambda j: projs_all[j])
        top5 = [valid_words[j] for j in ranked[:5]]
        tgt_rank = ranked.index(ti) if ti in ranked else -1
        hit = tgt_rank >= 0 and tgt_rank < 5
        if hit: hits += 1
        total += 1
        pair_results.append({
            "src": src, "tgt": tgt, "src_bit": src_bit, "tgt_bit": tgt_bit,
            "top5": top5, "tgt_rank": tgt_rank, "hit": hit
        })
    return hits, total, pair_results

# ── Compare sentence-level vs token-level axes ────────────────────────────────
print("=" * 72)
print("Traversal accuracy: sentence-level vs token-level axes")
print("=" * 72)

results = {}
COMPARISONS = [
    ("gender",     "gender_sent",     "gender_tok",     "gender"),
    ("past_tense", None,              "past_tense_tok",  "past_tense"),
    ("plural",     None,              "plural_tok",      "plural"),
    ("comparative",None,              "comparative_tok", "comparative"),
    ("antonym",    None,              "antonym_tok",     "antonym"),
]

for label, sent_key, tok_key, gt_key in COMPARISONS:
    gt_pairs = GROUND_TRUTH[gt_key]
    print(f"\n{label.upper()}")
    if sent_key and sent_key in axes:
        h, t, pr = run_traversal(label+"_sent", axes[sent_key], gt_pairs)
        print(f"  sentence-level axis: {h}/{t} ({100*h/t:.0f}% if t>0)")
        results[label+"_sent"] = {"hits": h, "total": t, "pairs": pr}
    h, t, pr = run_traversal(label+"_tok", axes[tok_key], gt_pairs)
    print(f"  token-level axis:    {h}/{t} ({100*h/t:.0f}% if t>0)")
    results[label+"_tok"] = {"hits": h, "total": t, "pairs": pr}
    # Show top5 for first pair
    if pr:
        p = pr[0]
        print(f"  e.g. {p['src']}({p['src_bit']})→{p['tgt']}({p['tgt_bit']}): "
              f"rank={p['tgt_rank']} top5={p['top5']}")

# ── Measure axis coherence (how "clean" is each token-level axis?) ────────────
print()
print("=" * 72)
print("Token-level axis coherence (mean cosine of difference vectors)")
print("=" * 72)
for name in ["gender_tok", "past_tense_tok", "plural_tok",
             "comparative_tok", "antonym_tok"]:
    pairs = TOKEN_PAIRS[name]
    individual_axes = []
    for s1, s2 in pairs:
        try:
            h1, _ = get_h28(s1); h2, _ = get_h28(s2)
            d = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: individual_axes.append(d / n)
        except: pass
    if len(individual_axes) < 2:
        print(f"  {name:<25}: insufficient data"); continue
    # Pairwise cosines between individual difference vectors
    cosines = []
    for i in range(len(individual_axes)):
        for j in range(i+1, len(individual_axes)):
            a, b = individual_axes[i], individual_axes[j]
            cosines.append(float(np.dot(a, b)))
    mean_cos = float(np.mean(cosines))
    mean_angle = float(math.degrees(math.acos(float(np.clip(mean_cos, -1, 1)))))
    print(f"  {name:<25}: mean pairwise cos={mean_cos:.3f}  angle={mean_angle:.1f}°")

# ── Gender axis comparison ────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"Gender sentence vs token axes: angle = {angle_gend:.1f}°")
print("=" * 72)
if angle_gend < 30:
    print("  SAME DIRECTION: token-level and sentence-level encode identical info")
elif angle_gend < 60:
    print("  SIMILAR: significant overlap between token-level and sentence-level")
elif angle_gend < 80:
    print("  DIFFERENT: token-level and sentence-level mostly different")
else:
    print("  ORTHOGONAL: completely different information sources")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 93 Summary")
print("=" * 72)
print(f"""
  TRAVERSAL ACCURACY (1D axis projection ranking):

  Axis           sentence-level   token-level
  ─────────────────────────────────────────────""")
for label, sent_key, tok_key, gt_key in COMPARISONS:
    s_str = f"{results[label+'_sent']['hits']}/{results[label+'_sent']['total']} ({100*results[label+'_sent']['hits']/max(1,results[label+'_sent']['total']):.0f}%)" if label+"_sent" in results else "N/A"
    t_str = f"{results[label+'_tok']['hits']}/{results[label+'_tok']['total']} ({100*results[label+'_tok']['hits']/max(1,results[label+'_tok']['total']):.0f}%)"
    print(f"  {label:<15}  {s_str:>15}  {t_str}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"results": results,
               "gender_sent_vs_tok_angle": float(angle_gend)},
              f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 93 complete.")
