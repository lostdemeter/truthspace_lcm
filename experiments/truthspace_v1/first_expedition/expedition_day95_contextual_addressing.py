#!/usr/bin/env python3
"""
Day 95 — Contextualized Addressing: Template-Embedded Token Representations

DC 326 identified the fundamental limit: sentence-level T2 axes vs
token-level classification. The trie navigates well for category axes
(gender 50%) but fails for relational axes (past_tense, plural: 0%).

Hypothesis: if we embed each token in an AXIS-APPROPRIATE SENTENCE TEMPLATE
and extract the LAST-TOKEN hidden state (as in the T2 axis computation),
the token's representation will better reflect the transformation axis
because we're using the SAME sentence frame that generated the axis.

For plural axis (built from "A dog played..." → "Dogs played..."):
  SINGULAR frame: "A [TOKEN] played happily in the open green field"
  PLURAL frame:   "[TOKEN] played happily in the open green field"

For past_tense (built from "I walk..." → "I walked..."):
  PRESENT frame: "I [TOKEN] to the market every single morning"
  PAST frame:    "I [TOKEN] to the market every single morning"
    (same frame, different token form)

For gender (built from "The king ruled..." → "The queen ruled..."):
  Frame: "The [TOKEN] ruled with great wisdom"

For comparative (built from "The fast car" → "The faster car"):
  Frame: "The [TOKEN] car"

For antonym: no clean frame (antonymy is relational, not positional)

PREDICTION:
  - Contextualized addressing SHOULD dramatically improve traversal for
    relational axes (plural, past_tense)
  - But: the same token embedded in different frames will have different
    addresses → context-dependent trie (loss of universality)
  - LOO may degrade because context-dependent addresses capture frame
    semantics, not pure token semantics

SCOPE: Test on ground truth traversal pairs only (80 tokens), comparing:
  1. Isolated: " [TOKEN]" (current Day 92 approach)
  2. Axis-contextualized: "[FRAME_WITH_TOKEN]" last-token position
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day95_contextual_addressing.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI      = (1 + math.sqrt(5)) / 2
INV_PHI  = 1 / PHI
INV_PHI2 = 1 / PHI**2

# Sentence templates used to generate each T2 axis
# (first pair from each axis's sentence list, as representative frame)
AXIS_TEMPLATES = {
    "gender":      "The {token} ruled with great wisdom",
    "plural":      "A {token} played happily in the open green field",
    "plural_p":    "{token} played happily in the open green field",  # plural form
    "past_tense":  "I {token} to the market every single morning",
    "comparative": "The {token} car",
    "antonym":     "It is {token}",
}

# Day 78 sentence-level T2 pairs for axis computation
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

# Ground truth traversal pairs (same as Day 92)
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

# Template sentences for contextualized addressing
# Maps axis → (singular_template, plural/transformed_template)
# The token is placed in the SAME structural position as in the axis pairs
CONTEXT_TEMPLATES = {
    "gender": {
        "src":  lambda t: f"The {t} ruled with great wisdom",
        "tgt":  lambda t: f"The {t} ruled with great wisdom",
    },
    "plural": {
        "src":  lambda t: f"A {t} played happily in the open green field",
        "tgt":  lambda t: f"{t.capitalize()} played happily in the open green field",
    },
    "past_tense": {
        "src":  lambda t: f"I {t} to the market every single morning",
        "tgt":  lambda t: f"I {t} to the market every single morning",
    },
    "comparative": {
        "src":  lambda t: f"The {t} car",
        "tgt":  lambda t: f"The {t} car",
    },
    "antonym": {
        "src":  lambda t: f"It is {t}",
        "tgt":  lambda t: f"It is {t}",
    },
}

print(f"Loading {MODEL_ID} ...")
tok_model = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
ALL_LAYERS = list(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}  layers={sorted(ALL_LAYERS)}\n")

def get_h_layers(text, layers):
    inp = tok_model(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

# ── Compute T2 axes (sentence-level, Day78 method) ────────────────────────────
print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_h_layers(s1, [L])[L]
            h2 = get_h_layers(s2, [L])[L]
            d = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    if diffs:
        v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
        t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
    else:
        t2_axes[name] = np.zeros(hidden_size, dtype=np.float32)
    print(f"  {name:<15} L{DAY78_LAYERS[name]}")
print()

# ── Phase 1: Isolated addresses for ground truth pairs ───────────────────────
print("Phase 1: Isolated token representations for GT pairs ...")
# Collect unique tokens from ground truth
gt_tokens = set()
for pairs in GROUND_TRUTH.values():
    for s, t in pairs:
        gt_tokens.update([s, t])
gt_tokens = sorted(gt_tokens)

iso_hs = {}  # {word: {L: hidden}}
for word in gt_tokens:
    try:
        h = get_h_layers(" " + word.strip(), ALL_LAYERS)
        iso_hs[word] = h
    except: pass
print(f"  Extracted isolated hs for {len(iso_hs)} GT tokens\n")

# ── Phase 2: Contextualized addresses ────────────────────────────────────────
print("Phase 2: Contextualized representations for GT pairs ...")
ctx_hs = {}  # {word: {axis_name: {L: hidden}}}
# For each axis and each GT token, embed in the axis's template sentence
for axis_name, templates in CONTEXT_TEMPLATES.items():
    if axis_name not in GROUND_TRUTH: continue
    for src, tgt in GROUND_TRUTH[axis_name]:
        for role, word in [("src", src), ("tgt", tgt)]:
            try:
                sent = templates[role](word)
                h    = get_h_layers(sent, ALL_LAYERS)
                key  = (word, axis_name)
                ctx_hs[key] = h
            except Exception as e:
                pass

print(f"  Extracted contextualized hs for {len(ctx_hs)} (word, axis) pairs\n")

# ── Helper: build per-axis threshold from all tokens' projections ─────────────
def classify_tokens(hiddens_dict, axis_vec, layer):
    """hiddens_dict: {word: {L: vec}}"""
    projs = {w: float(np.dot(hiddens_dict[w][layer], axis_vec))
             for w in hiddens_dict if layer in hiddens_dict[w]}
    if not projs: return {}
    vals = list(projs.values())
    max_p = float(np.percentile(vals, 95))
    if max_p < 1e-6: return {w: "U" for w in projs}
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return {w: ("H" if p > hi else "L" if p < lo else "U") for w, p in projs.items()}

# ── Phase 3: Compare isolated vs contextualized axis bit assignments ───────────
print("=" * 72)
print("Phase 3: Isolated vs Contextualized axis bit assignment")
print("=" * 72)

results = {}
for axis_name, gt_pairs in GROUND_TRUTH.items():
    if axis_name not in CONTEXT_TEMPLATES: continue
    axis_vec = t2_axes.get(axis_name, np.zeros(hidden_size))
    L = DAY78_LAYERS.get(axis_name, 28)
    print(f"\n  {axis_name.upper()} (L{L}):")

    # Isolated classifications
    iso_class = classify_tokens(iso_hs, axis_vec, L)

    # Contextualized classifications for this axis
    ctx_class_src = {}; ctx_class_tgt = {}
    for src, tgt in gt_pairs:
        key_s = (src, axis_name); key_t = (tgt, axis_name)
        if key_s in ctx_hs:
            projs_s = float(np.dot(ctx_hs[key_s][L], axis_vec))
            ctx_class_src[src] = projs_s
        if key_t in ctx_hs:
            projs_t = float(np.dot(ctx_hs[key_t][L], axis_vec))
            ctx_class_tgt[tgt] = projs_t

    # Compute thresholds from all ctx projections for this axis
    all_ctx_projs = list(ctx_class_src.values()) + list(ctx_class_tgt.values())
    if all_ctx_projs:
        max_p = float(np.percentile(all_ctx_projs, 95))
        hi, lo = max_p * INV_PHI, max_p * INV_PHI2 if max_p > 1e-6 else (0, 0)
        def ctx_bit(p): return "H" if p > hi else "L" if p < lo else "U"
    else:
        def ctx_bit(p): return "U"

    pair_results = []
    iso_correct = 0; ctx_correct = 0; total = 0
    print(f"  {'src':>10} iso_src  ctx_src  {'tgt':>10} iso_tgt  ctx_tgt  "
          f"iso_sep  ctx_sep")
    for src, tgt in gt_pairs:
        if src not in iso_hs or tgt not in iso_hs: continue
        if (src, axis_name) not in ctx_hs or (tgt, axis_name) not in ctx_hs: continue

        i_src = iso_class.get(src, "?"); i_tgt = iso_class.get(tgt, "?")
        c_src = ctx_bit(ctx_class_src.get(src, 0))
        c_tgt = ctx_bit(ctx_class_tgt.get(tgt, 0))

        iso_sep = (i_src != i_tgt)
        ctx_sep = (c_src != c_tgt)
        if iso_sep: iso_correct += 1
        if ctx_sep: ctx_correct += 1
        total += 1

        print(f"  {src:>10} {i_src:>7}  {c_src:>7}  {tgt:>10} "
              f"{i_tgt:>7}  {c_tgt:>7}  "
              f"{'✓' if iso_sep else '✗':>7}  {'✓' if ctx_sep else '✗':>7}")
        pair_results.append({
            "src": src, "tgt": tgt,
            "iso_src_bit": i_src, "iso_tgt_bit": i_tgt,
            "ctx_src_bit": c_src, "ctx_tgt_bit": c_tgt,
            "iso_separated": iso_sep, "ctx_separated": ctx_sep,
        })

    iso_rate = iso_correct / max(1, total)
    ctx_rate = ctx_correct / max(1, total)
    print(f"\n  Isolated separation:      {iso_correct}/{total} ({100*iso_rate:.0f}%)")
    print(f"  Contextualized separation:{ctx_correct}/{total} ({100*ctx_rate:.0f}%)")
    improvement = ctx_rate - iso_rate
    print(f"  Improvement: {improvement:+.1%}")
    results[axis_name] = {
        "pairs": pair_results, "iso_rate": iso_rate, "ctx_rate": ctx_rate,
        "improvement": improvement, "iso_correct": iso_correct,
        "ctx_correct": ctx_correct, "total": total,
    }

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 95 Summary: Does Contextualized Addressing Improve Bit Separation?")
print("=" * 72)
print(f"\n  {'axis':>15}  {'isolated':>12}  {'contextualized':>15}  {'change':>8}")
for name, r in results.items():
    print(f"  {name:>15}  {r['iso_correct']:>2}/{r['total']:>2} ({100*r['iso_rate']:.0f}%)"
          f"  {r['ctx_correct']:>2}/{r['total']:>2} ({100*r['ctx_rate']:.0f}%)   "
          f"   {r['improvement']:>+.1%}")

total_iso = sum(r["iso_correct"] for r in results.values())
total_ctx = sum(r["ctx_correct"] for r in results.values())
total_tot = sum(r["total"] for r in results.values())
print(f"\n  {'OVERALL':>15}  {total_iso:>2}/{total_tot:>2} ({100*total_iso/max(1,total_tot):.0f}%)  "
      f"{total_ctx:>2}/{total_tot:>2} ({100*total_ctx/max(1,total_tot):.0f}%)   "
      f"   {(total_ctx-total_iso)/max(1,total_tot):>+.1%}")

print(f"""
  INTERPRETATION:
  Bit separation = src_bit ≠ tgt_bit for a ground truth pair.
  If contextualized addressing improves separation significantly:
    → T2 axes DO activate better when tokens are in sentence context
    → The trie should use contextualized (sentence-embedded) hidden states
  If no improvement:
    → The T2 axis direction is robust to context; isolated is sufficient
    → Navigability is limited by axis coherence, not addressing method
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"results": results}, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 95 complete.")
