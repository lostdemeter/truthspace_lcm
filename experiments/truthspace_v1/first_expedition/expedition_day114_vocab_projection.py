#!/usr/bin/env python3
"""
Day 114 — Vocabulary Projection onto Both Geometric Subspaces

DC 331 established two orthogonal geometric structures in the LM:
  1. T2 categorical axes  (~12D): semantic property space
  2. Entity selector axes (~66D): factual entity identity space

Day 114 projects all 420 probe tokens onto both subspaces and tests:
  A. Do proper nouns (king, queen, actor, etc.) project strongly onto d_k?
  B. Do semantic categories (animals, verbs, adjectives) cluster near T2 axes?
  C. Are the two projections complementary — different tokens characterized
     by different subspaces?
  D. Can we predict token type (proper noun vs common noun vs verb vs adj)
     from its projection profile?

PREDICTION (under two-structure theory):
  - Entities/proper nouns: high d_k projection, moderate T2 projection
  - Semantic categories: high T2 projection (category-specific axes)
  - Function words (the, and, is): low both (not entity, not semantic content)
  - Morphological variants (ran, walked, faster): high morphological T2 axes
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day114_vocab_projection.json")
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

# Token categories for projection analysis
TOKEN_CATEGORIES = {
    "animal": ["dog","cat","bird","fish","horse","wolf","lion","tiger","elephant","mouse",
               "rabbit","deer","bear","fox","eagle","whale","shark","frog","snake","monkey",
               "cow","pig","sheep","goat","duck","hen","crow","owl","turtle","lizard"],
    "nature": ["tree","flower","rock","stone","wood","leaf","grass","root","river","mountain",
               "ocean","forest","cloud","rain","snow","wind","sun","moon","star","sky","earth"],
    "artifact": ["house","door","window","table","chair","book","cup","key","car","road",
                 "bridge","boat","ship","plane","train","bike","knife","hammer","clock","lamp"],
    "verb_base": ["run","walk","jump","swim","fly","eat","sleep","talk","write","read",
                  "build","break","open","close","think","know","see","hear","feel","love"],
    "verb_past": ["ran","walked","jumped","flew","ate","saw","heard","broke","built","wrote"],
    "adjective": ["fast","slow","big","small","hot","cold","old","new","hard","soft",
                  "bright","dark","strong","weak","happy","sad","good","bad","right","wrong"],
    "comparative": ["faster","slower","bigger","smaller","better","worse",
                    "biggest","smallest","best","worst"],
    "function": ["the","a","and","or","not","is","was","in","on","of","to","from",
                 "with","for","he","she","it","they","we","I","you","his","her"],
    "gender_pair": ["king","queen","man","woman","boy","girl","brother","sister",
                    "father","mother","son","daughter","husband","wife","prince","princess"],
    "abstract": ["love","truth","beauty","freedom","power","time","space","mind","body",
                 "soul","life","death","hope","fear","joy","pain","trust","faith","peace"],
    "plural_noun": ["dogs","cats","trees","birds","horses","men","women","children","hands","eyes"],
    "hypernym": ["animal","vehicle","tool","gem","burden","barrier","journey","bond"],
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

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_heads     = model.config.num_attention_heads
n_kv_heads  = model.config.num_key_value_heads
head_dim    = hidden_size // n_heads
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}, n_heads={n_heads}, head_dim={head_dim}\n")

def get_last_h(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_last_h(s1, L); h2 = get_last_h(s2, L)
            d  = h2 - h1; nrm = np.linalg.norm(d)
            if nrm > 1e-6: diffs.append(d / nrm)
        except: pass
    v  = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

print("Computing d_k (H6 L23) entity selector direction ...")
L23      = model.model.layers[22]
W_k_L23  = L23.self_attn.k_proj.weight.data.float().numpy()
kv_group = n_heads // n_kv_heads
kvi      = 6 // kv_group
h6k      = W_k_L23[kvi*head_dim : (kvi+1)*head_dim, :]
Uk, _, _ = np.linalg.svd(h6k, full_matrices=False)
d_k      = h6k.T @ Uk[:, 0]
d_k      = (d_k / np.linalg.norm(d_k)).astype(np.float32)

# T2 matrix for subspace projection
t2_matrix = np.stack([t2_axes[ax] for ax in AXIS_NAMES_12], axis=0)  # (12, 1536)

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
for L in ALL_LAYERS: hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

# Use the primary T2 layer (28 = layer index for most axes) for projection
# For each token, use its hidden state at the most relevant layer
# Use L28 (last) as canonical representation for all tokens
L_CANON = max(ALL_LAYERS)  # 28 for most axes, use highest
hs_canon = hs_by_layer[L_CANON]  # (N, 1536)

# Normalize hidden states
hs_norms = np.linalg.norm(hs_canon, axis=1, keepdims=True)
hs_normed = hs_canon / np.maximum(hs_norms, 1e-8)

# ── Projection computations ───────────────────────────────────────────────────
# T2 subspace projection: project onto each T2 axis, compute 12D coordinate
t2_projections = hs_normed @ t2_matrix.T  # (N, 12) — dot with each axis
t2_magnitude   = np.linalg.norm(t2_projections, axis=1)  # (N,) — L2 in T2 subspace

# Entity selector projection: project onto d_k
dk_projection  = hs_normed @ d_k  # (N,) — scalar projection
dk_abs         = np.abs(dk_projection)

# ── Exp 1: Per-category projection statistics ─────────────────────────────────
print("=" * 72)
print("Exp 1: Category Projection onto T2 vs d_k Subspace")
print("=" * 72)
print(f"\n  {'category':>15}  {'n':>4}  {'T2_mag_mean':>12}  {'dk_abs_mean':>12}  "
      f"{'T2/dk_ratio':>12}")
print(f"  {'-'*60}")

cat_results = {}
for cat_name, words in TOKEN_CATEGORIES.items():
    idxs = [word_idx[w] for w in words if w in word_idx]
    if not idxs: continue
    t2_m  = float(np.mean(t2_magnitude[idxs]))
    dk_m  = float(np.mean(dk_abs[idxs]))
    ratio = t2_m / max(dk_m, 1e-8)
    cat_results[cat_name] = {"t2_mag": t2_m, "dk_abs": dk_m, "ratio": ratio, "n": len(idxs)}
    print(f"  {cat_name:>15}  {len(idxs):>4}  {t2_m:>12.4f}  {dk_m:>12.4f}  {ratio:>12.2f}")

# All-token stats
overall_t2 = float(np.mean(t2_magnitude))
overall_dk = float(np.mean(dk_abs))
print(f"  {'ALL':>15}  {N:>4}  {overall_t2:>12.4f}  {overall_dk:>12.4f}  "
      f"{overall_t2/max(overall_dk,1e-8):>12.2f}")

# ── Exp 2: Top tokens by d_k projection ──────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Top/Bottom Tokens by d_k (Entity Selector) Projection")
print("=" * 72)

sorted_dk = np.argsort(-dk_abs)
print(f"\n  Top 20 tokens by |d_k projection|:")
print(f"  {'rank':>5}  {'word':>12}  {'dk_proj':>10}  {'t2_mag':>10}")
for rank, idx in enumerate(sorted_dk[:20]):
    print(f"  {rank+1:>5}  {valid_words[idx]:>12}  {dk_abs[idx]:>10.4f}  {t2_magnitude[idx]:>10.4f}")

print(f"\n  Bottom 20 tokens by |d_k projection| (function words expected):")
for rank, idx in enumerate(sorted_dk[-20:]):
    r = N - 20 + rank + 1
    print(f"  {r:>5}  {valid_words[idx]:>12}  {dk_abs[idx]:>10.4f}  {t2_magnitude[idx]:>10.4f}")

# ── Exp 3: Top tokens by T2 magnitude ────────────────────────────────────────
print()
print("=" * 72)
print("Exp 3: Top Tokens by T2 Subspace Magnitude")
print("=" * 72)

sorted_t2 = np.argsort(-t2_magnitude)
print(f"\n  Top 20 tokens by T2 subspace magnitude:")
print(f"  {'rank':>5}  {'word':>12}  {'t2_mag':>10}  {'dk_proj':>10}  {'dominant_axis':>15}")
for rank, idx in enumerate(sorted_t2[:20]):
    dom_ax = AXIS_NAMES_12[int(np.argmax(np.abs(t2_projections[idx])))]
    print(f"  {rank+1:>5}  {valid_words[idx]:>12}  {t2_magnitude[idx]:>10.4f}  "
          f"{dk_abs[idx]:>10.4f}  {dom_ax:>15}")

# ── Exp 4: Per-axis top words ─────────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 4: Top 5 Words per T2 Axis (H-pole)")
print("=" * 72)
print()

axis_top_words = {}
for k, ax_name in enumerate(AXIS_NAMES_12):
    projs_ax = t2_projections[:, k]  # (N,)
    top_pos  = np.argsort(-projs_ax)[:5]
    top_neg  = np.argsort(projs_ax)[:5]
    h_words  = [(valid_words[i], float(projs_ax[i])) for i in top_pos]
    l_words  = [(valid_words[i], float(projs_ax[i])) for i in top_neg]
    axis_top_words[ax_name] = {"H_pole": h_words, "L_pole": l_words}
    h_str = ", ".join(f"{w}({v:+.2f})" for w,v in h_words)
    l_str = ", ".join(f"{w}({v:+.2f})" for w,v in l_words)
    print(f"  {ax_name:>14}  H-pole: {h_str}")
    print(f"  {' '*14}  L-pole: {l_str}")
    print()

# ── Exp 5: Quadrant analysis — high T2 vs high d_k ───────────────────────────
print()
print("=" * 72)
print("Exp 5: Quadrant Analysis — T2 Magnitude vs d_k Projection")
print("=" * 72)

t2_med = np.median(t2_magnitude)
dk_med = np.median(dk_abs)
quadrants = {"high_t2_high_dk": [], "high_t2_low_dk": [],
             "low_t2_high_dk":  [], "low_t2_low_dk":  []}
for i in range(N):
    t2h = t2_magnitude[i] >= t2_med
    dkh = dk_abs[i]       >= dk_med
    if   t2h and dkh: quadrants["high_t2_high_dk"].append(valid_words[i])
    elif t2h        : quadrants["high_t2_low_dk"].append(valid_words[i])
    elif dkh        : quadrants["low_t2_high_dk"].append(valid_words[i])
    else            : quadrants["low_t2_low_dk"].append(valid_words[i])

print(f"\n  Median T2 = {t2_med:.4f},  Median d_k = {dk_med:.4f}")
print()
for q_name, words in quadrants.items():
    sample = ", ".join(words[:15]) + ("..." if len(words) > 15 else "")
    print(f"  {q_name:>20}  (n={len(words):>3})  {sample}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 114 Summary — Vocabulary Projection onto Both Subspaces")
print("=" * 72)

# Rank categories by T2/dk ratio
sorted_cats = sorted(cat_results.items(), key=lambda x: -x[1]["ratio"])
highest_t2_cat = sorted_cats[0][0]
lowest_t2_cat  = sorted_cats[-1][0]

# Top d_k token
top_dk_word = valid_words[sorted_dk[0]]
top_t2_word = valid_words[sorted_t2[0]]

print(f"""
  d_k (entity selector) projection:
    Top token:   {top_dk_word} (|d_k|={dk_abs[sorted_dk[0]]:.4f})
    Overall mean: {overall_dk:.4f}
    Highest d_k category: {max(cat_results, key=lambda c: cat_results[c]['dk_abs'])}

  T2 subspace magnitude:
    Top token:   {top_t2_word} (T2_mag={t2_magnitude[sorted_t2[0]]:.4f})
    Overall mean: {overall_t2:.4f}
    Highest T2 category: {max(cat_results, key=lambda c: cat_results[c]['t2_mag'])}

  Complementarity:
    Highest T2/dk ratio: {highest_t2_cat} (ratio={cat_results[highest_t2_cat]['ratio']:.2f})
    Lowest  T2/dk ratio: {lowest_t2_cat}  (ratio={cat_results[lowest_t2_cat]['ratio']:.2f})

  Quadrant sizes:
    High T2, High d_k: {len(quadrants['high_t2_high_dk'])} tokens
    High T2, Low  d_k: {len(quadrants['high_t2_low_dk'])} tokens
    Low  T2, High d_k: {len(quadrants['low_t2_high_dk'])} tokens
    Low  T2, Low  d_k: {len(quadrants['low_t2_low_dk'])} tokens

  VERDICT:
  {'→ T2 and d_k capture COMPLEMENTARY properties (different categories rank differently)' if
   cat_results[highest_t2_cat]['ratio'] > 2 * cat_results[lowest_t2_cat]['ratio'] else
   '→ T2 and d_k have CORRELATED projections (same tokens rank high on both)'}

  KEY FINDING:
  The two geometric subspaces {'clearly separate token categories' if
  cat_results[highest_t2_cat]['ratio'] > 2 * cat_results[lowest_t2_cat]['ratio'] else
  'do not cleanly separate token categories'}.
  {'Semantic categories and proper nouns occupy different subspaces.' if
   cat_results[highest_t2_cat]['ratio'] > 2 * cat_results[lowest_t2_cat]['ratio'] else
   'Both subspaces may capture similar information about the same tokens.'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "cat_results": cat_results,
        "axis_top_words": axis_top_words,
        "quadrant_sizes": {k: len(v) for k,v in quadrants.items()},
        "quadrant_samples": {k: v[:10] for k,v in quadrants.items()},
        "top_dk_tokens": [(valid_words[i], float(dk_abs[i])) for i in sorted_dk[:20]],
        "top_t2_tokens": [(valid_words[i], float(t2_magnitude[i])) for i in sorted_t2[:20]],
        "overall_t2": overall_t2, "overall_dk": overall_dk,
        "t2_median": float(t2_med), "dk_median": float(dk_med),
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 114 complete.")
