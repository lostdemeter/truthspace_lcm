#!/usr/bin/env python3
"""
Day 116 — Entity Selector Validation with Proper Nouns

DC 331 claims the LM has TWO orthogonal geometric structures:
  1. T2 categorical axes (~12D): semantic category properties
  2. Entity selector direction (d_k at H6 L23): entity identity

Day 113 found T2 ⊥ d_k (cosine 0.014, below random 0.021).
Day 114b found d_k best token = "sparrow" from common-word vocabulary.

The entity selector was validated on PROPER NOUNS in model rev-eng:
  France → Paris, Germany → Berlin, Japan → Tokyo (Finding 40)

But our probe vocabulary contains only COMMON WORDS (dog, cat, king...).
To properly validate the entity selector, we need PROPER NOUNS.

EXPERIMENT:
  Expand the vocabulary to include:
    - Proper nouns: countries, capitals, cities, famous people's names
    - Common nouns for comparison

  Then measure:
  A. Do proper nouns have higher d_k projection than common words?
  B. Do proper nouns have different T2 addresses than common words?
  C. Are proper nouns + their related concepts geometrically close
     in EACH subspace independently?
     - T2 space: France ↔ Germany (both countries, should be close)
     - d_k space: France ↔ Paris (entity + answer, should be close)

PREDICTION (two-structure theory):
  - Proper nouns: HIGH d_k projection (they ARE entity types)
  - Proper noun pairs (France/Paris): d_k similarity > T2 similarity
  - Category pairs (dog/cat): T2 similarity > d_k similarity

Also tests: can the d_k direction distinguish entity-type tokens from
common tokens, validating its role as an "entity identity selector"?
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day116_entity_selector_validation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

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

# Vocabulary: mix of proper nouns and common words
# Proper nouns — countries + capitals (from model rev-eng Finding 40)
COUNTRY_CAPITAL_PAIRS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
    ("Italy", "Rome"), ("Spain", "Madrid"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Mexico", "Mexico"),
    ("Brazil", "Brasilia"), ("India", "Delhi"), ("China", "Beijing"),
    ("Russia", "Moscow"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ("Egypt", "Cairo"), ("Turkey", "Ankara"), ("Greece", "Athens"),
]

PROPER_NOUNS = (
    [c for c,_ in COUNTRY_CAPITAL_PAIRS] +
    [cap for _,cap in COUNTRY_CAPITAL_PAIRS] +
    ["London", "Washington", "Amsterdam", "Vienna", "Prague", "Budapest",
     "Lisbon", "Dublin", "Helsinki", "Copenhagen", "Brussels", "Zurich",
     "Newton", "Einstein", "Darwin", "Plato", "Aristotle",
     "Shakespeare", "Mozart", "Beethoven", "Picasso",
     "Europe", "Asia", "Africa", "America", "Pacific", "Atlantic",
     "English", "French", "German", "Spanish", "Italian", "Japanese"]
)

COMMON_NOUNS = [
    "dog","cat","bird","horse","tree","river","mountain","house","book","door",
    "man","woman","king","queen","city","village","garden","school","bridge","road",
]
COMMON_VERBS = ["run","walk","eat","sleep","think","know","love","build","break","find"]
COMMON_ADJS  = ["fast","slow","big","small","hot","cold","old","new","good","bad"]
FUNCTION_WDS = ["the","a","and","or","is","was","to","of","in","on"]

ALL_TOKENS = list(dict.fromkeys(
    PROPER_NOUNS + COMMON_NOUNS + COMMON_VERBS + COMMON_ADJS + FUNCTION_WDS
))

# For T2 axes, use the same all-layers extraction approach
INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_heads     = model.config.num_attention_heads
n_kv_heads  = model.config.num_key_value_heads
head_dim    = hidden_size // n_heads
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}\n")

def get_last_h(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].numpy().astype(np.float32)

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_last_h(s1, L); h2 = get_last_h(s2, L)
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    v  = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

print("Computing d_k (H6 L23) entity selector direction ...")
L23     = model.model.layers[22]
W_k_L23 = L23.self_attn.k_proj.weight.data.float().numpy()
kv_grp  = n_heads // n_kv_heads
kvi     = 6 // kv_grp
h6k     = W_k_L23[kvi*head_dim : (kvi+1)*head_dim, :]
Uk,_,_  = np.linalg.svd(h6k, full_matrices=False)
d_k     = (h6k.T @ Uk[:, 0]).astype(np.float32)
d_k    /= np.linalg.norm(d_k)

print("Extracting hidden states for expanded vocabulary ...")
hs_by_layer = {L: {} for L in ALL_LAYERS}  # {layer: {word: vec}}
hs_L23      = {}
valid_words = []
for word in ALL_TOKENS:
    try:
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        for L in ALL_LAYERS:
            hs_by_layer[L][word] = out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
        hs_L23[word] = out.hidden_states[23][0, pos, :].numpy().astype(np.float32)
        valid_words.append(word)
    except: pass

N = len(valid_words)
print(f"  {N} tokens\n")

def normed_vec(v): return v / (np.linalg.norm(v) + 1e-8)

# T2 projection per word (per-axis correct layer)
def t2_proj(word):
    return np.array([
        float(np.dot(normed_vec(hs_by_layer[DAY78_LAYERS[ax]][word]), t2_axes[ax]))
        for ax in AXIS_NAMES_12
    ], dtype=np.float32)

# d_k projection per word (at L23)
def dk_proj(word):
    return float(abs(np.dot(normed_vec(hs_L23[word]), d_k)))

print("Computing projections for all tokens ...")
t2_projs = {w: t2_proj(w) for w in valid_words}
dk_projs = {w: dk_proj(w) for w in valid_words}
t2_mags  = {w: float(np.linalg.norm(t2_projs[w])) for w in valid_words}

# Token type assignment
def get_type(w):
    if w in [c for c,_ in COUNTRY_CAPITAL_PAIRS]: return "country"
    if w in [cap for _,cap in COUNTRY_CAPITAL_PAIRS]: return "capital"
    if w in PROPER_NOUNS: return "proper_noun"
    if w in COMMON_NOUNS: return "common_noun"
    if w in COMMON_VERBS: return "common_verb"
    if w in COMMON_ADJS:  return "adjective"
    if w in FUNCTION_WDS: return "function"
    return "other"

# ── Exp 1: d_k projection by token type ──────────────────────────────────────
print("=" * 72)
print("Exp 1: d_k Projection by Token Type")
print("=" * 72)
print(f"\n  {'type':>15}  {'n':>4}  {'dk_mean':>10}  {'dk_max':>10}  {'T2_mean':>10}  {'T2/dk':>8}")
print(f"  {'-'*60}")

type_groups = {}
for w in valid_words:
    t = get_type(w)
    type_groups.setdefault(t, []).append(w)

type_results = {}
for t_name in ["country","capital","proper_noun","common_noun","common_verb","adjective","function"]:
    words = type_groups.get(t_name, [])
    if not words: continue
    dk_vals = [dk_projs[w] for w in words]
    t2_vals = [t2_mags[w] for w in words]
    dk_m = float(np.mean(dk_vals)); dk_x = float(np.max(dk_vals))
    t2_m = float(np.mean(t2_vals))
    ratio = t2_m / max(dk_m, 1e-8)
    type_results[t_name] = {"dk_mean": dk_m, "dk_max": dk_x, "t2_mean": t2_m,
                             "ratio": ratio, "n": len(words)}
    print(f"  {t_name:>15}  {len(words):>4}  {dk_m:>10.4f}  {dk_x:>10.4f}  {t2_m:>10.4f}  {ratio:>8.2f}")

# ── Exp 2: Top tokens by d_k projection ──────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Top 20 Tokens by d_k Projection")
print("=" * 72)
sorted_dk = sorted(valid_words, key=lambda w: -dk_projs[w])
print(f"\n  {'rank':>5}  {'word':>15}  {'type':>12}  {'dk_proj':>10}  {'T2_mag':>10}")
for rank, w in enumerate(sorted_dk[:20]):
    print(f"  {rank+1:>5}  {w:>15}  {get_type(w):>12}  {dk_projs[w]:>10.4f}  {t2_mags[w]:>10.4f}")

# ── Exp 3: Country ↔ Capital similarity in T2 vs d_k subspace ────────────────
print()
print("=" * 72)
print("Exp 3: Country ↔ Capital Pair Similarity (T2 vs d_k subspace)")
print("=" * 72)
print(f"\n  {'pair':>20}  {'T2_cosim':>10}  {'dk_proj_c':>12}  {'dk_proj_cap':>12}  "
      f"{'dk_diff':>10}  {'which_closer?':>14}")
print(f"  {'-'*80}")

pair_results = {}
for country, capital in COUNTRY_CAPITAL_PAIRS:
    if country not in valid_words or capital not in valid_words: continue
    # T2 cosine similarity between country and capital
    t2_c = t2_projs[country]; t2_cap = t2_projs[capital]
    t2_cos = float(np.dot(t2_c, t2_cap) / (np.linalg.norm(t2_c) * np.linalg.norm(t2_cap) + 1e-8))
    dk_c   = dk_projs[country]; dk_cap = dk_projs[capital]
    dk_diff = abs(dk_c - dk_cap)
    which = "T2" if t2_cos > 0.5 else "dk" if dk_diff < 0.01 else "neither"
    pair_results[f"{country}/{capital}"] = {"t2_cosim": t2_cos, "dk_country": dk_c,
                                             "dk_capital": dk_cap, "dk_diff": dk_diff}
    print(f"  {country+'/'+capital:>20}  {t2_cos:>10.4f}  {dk_c:>12.4f}  {dk_cap:>12.4f}  "
          f"{dk_diff:>10.4f}  {which:>14}")

# Compare to same-category common word pairs
print()
print("  Reference: same-category common word pairs (T2 similarity expected)")
ref_pairs = [("dog","cat"), ("run","walk"), ("big","small"), ("good","bad"),
             ("man","woman"), ("king","queen")]
for w1, w2 in ref_pairs:
    if w1 not in valid_words or w2 not in valid_words: continue
    t2_c = t2_projs[w1]; t2_c2 = t2_projs[w2]
    t2_cos = float(np.dot(t2_c, t2_c2) / (np.linalg.norm(t2_c) * np.linalg.norm(t2_c2) + 1e-8))
    dk_d   = abs(dk_projs[w1] - dk_projs[w2])
    print(f"  {w1+'/'+w2:>20}  {t2_cos:>10.4f}  {dk_projs[w1]:>12.4f}  {dk_projs[w2]:>12.4f}  {dk_d:>10.4f}")

# ── Exp 4: Proper noun cluster analysis ──────────────────────────────────────
print()
print("=" * 72)
print("Exp 4: Proper Noun vs Common Word Cluster Separation")
print("=" * 72)

proper_dk = [dk_projs[w] for w in valid_words if get_type(w) in ("country","capital","proper_noun")]
common_dk  = [dk_projs[w] for w in valid_words if get_type(w) in ("common_noun","common_verb","adjective","function")]

if proper_dk and common_dk:
    # Cohen's d effect size
    pooled_std = math.sqrt((np.var(proper_dk) + np.var(common_dk)) / 2)
    cohens_d   = (np.mean(proper_dk) - np.mean(common_dk)) / max(pooled_std, 1e-8)
    print(f"\n  Proper nouns d_k: mean={np.mean(proper_dk):.4f}, std={np.std(proper_dk):.4f}")
    print(f"  Common words d_k: mean={np.mean(common_dk):.4f}, std={np.std(common_dk):.4f}")
    print(f"  Cohen's d (effect size): {cohens_d:.4f}")
    print(f"  Verdict: {'LARGE effect (>0.8)' if abs(cohens_d)>0.8 else 'MEDIUM effect (>0.5)' if abs(cohens_d)>0.5 else 'SMALL effect (>0.2)' if abs(cohens_d)>0.2 else 'NO effect (<0.2)'}")
    print(f"  Direction: {'proper nouns HIGHER d_k' if np.mean(proper_dk)>np.mean(common_dk) else 'proper nouns LOWER d_k (unexpected)'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 116 Summary — Entity Selector Validation with Proper Nouns")
print("=" * 72)

country_dk  = type_results.get("country", {}).get("dk_mean", 0)
capital_dk  = type_results.get("capital", {}).get("dk_mean", 0)
common_n_dk = type_results.get("common_noun", {}).get("dk_mean", 0)
func_dk     = type_results.get("function", {}).get("dk_mean", 0)
top_dk_word = sorted_dk[0]
top_dk_type = get_type(top_dk_word)

print(f"""
  d_k by type (countries vs common):
    Countries:    {country_dk:.4f}
    Capitals:     {capital_dk:.4f}
    Common nouns: {common_n_dk:.4f}
    Function wds: {func_dk:.4f}

  Top d_k token: {top_dk_word} (type={top_dk_type}, dk={dk_projs[top_dk_word]:.4f})

  Country/capital T2 cosine similarity:
    Mean: {np.mean([r['t2_cosim'] for r in pair_results.values()]):.4f}
    (Reference: dog/cat = {float(np.dot(t2_projs['dog'],t2_projs['cat'])/(np.linalg.norm(t2_projs['dog'])*np.linalg.norm(t2_projs['cat'])+1e-8)):.4f} if available)

  VERDICT:
  {'→ Entity selector (d_k) VALIDATED: proper nouns have higher d_k projection' if proper_dk and common_dk and np.mean(proper_dk) > np.mean(common_dk) + np.std(common_dk)*0.5 else
   '→ Entity selector (d_k) NOT validated with this vocabulary' if proper_dk and common_dk else
   '→ Insufficient data for verdict'}

  KEY FINDING:
  {'→ DC 331 two-structure theory VALIDATED: d_k discriminates entity tokens from common words' if proper_dk and common_dk and np.mean(proper_dk) > np.mean(common_dk) else
   '→ d_k direction does not preferentially select proper nouns in this test'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "type_results": type_results,
        "pair_results": pair_results,
        "top_dk_tokens": [(w, float(dk_projs[w]), get_type(w)) for w in sorted_dk[:20]],
        "cohens_d": float(cohens_d) if proper_dk and common_dk else None,
        "proper_dk_mean": float(np.mean(proper_dk)) if proper_dk else None,
        "common_dk_mean": float(np.mean(common_dk)) if common_dk else None,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 116 complete.")
