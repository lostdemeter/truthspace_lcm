#!/usr/bin/env python3
"""
Day 115 — Threshold-Amplification Test

Day 114b showed: T2 axis projections for semantic categories are weak
(Δ = 0.003 to 0.040) yet the trie achieves 94% LOO accuracy.

HYPOTHESIS: The φ-threshold classifier amplifies weak continuous signals
into strong discrete H/U/L class separations.

MECHANISM:
  1. Continuous projection p ∈ ℝ  (weak signal, Δ ~ 0.03)
  2. φ-threshold binning: H if p > φ-cutoff, L if p < lower-cutoff, else U
  3. Discrete H/U/L label (strong separation — addresses are unique)

The amplification happens because:
  - The threshold separates a continuum into 3 bins
  - Words on opposite sides of a threshold get maximally different labels
  - The 12-dimensional product of 3-class labels gives 3^12 = 531,441
    possible addresses — enough to uniquely label all vocabulary words

EXPERIMENT:
  For each axis × category pair:
  1. Compute the φ-threshold cutoffs (using the classify_all() method)
  2. Assign H/U/L to every vocabulary token
  3. Measure the H/U/L distribution FOR category words vs all others
  4. Compute Cramér's V (association strength between category × H/U/L)

  Compare:
  - Continuous projection Δ (Day 114b)
  - Discrete H/U/L purity (Day 115)

  If φ-thresholding AMPLIFIES: discrete purity >> continuous delta
  If φ-thresholding does NOT amplify: discrete purity ≈ continuous delta

ALSO:
  - Measure how many unique addresses exist in the vocabulary
  - Measure how the uniqueness changes with k axes (1→12)
  - Show that uniqueness (the key to LOO=94%) is a THRESHOLD property
"""
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day115_threshold_amplification.json")
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

TOKEN_CATEGORIES = {
    "gender_pair": ["king","queen","man","woman","boy","girl","brother","sister",
                    "father","mother","son","daughter","husband","wife","prince","princess",
                    "actor","actress"],
    "verb_past":   ["ran","walked","jumped","flew","ate","saw","heard","broke","built","wrote"],
    "verb_base":   ["run","walk","jump","swim","eat","sleep","talk","write","read","build",
                    "break","open","close","think","know","see","hear","feel","love","hate"],
    "plural_noun": ["dogs","cats","trees","birds","horses","men","women","children","hands","eyes"],
    "comparative": ["faster","slower","bigger","smaller","better","worse",
                    "biggest","smallest","best","worst"],
    "hypernym":    ["animal","vehicle","tool","gem","burden","barrier","journey","bond"],
    "abstract":    ["love","truth","beauty","freedom","power","time","space","mind","body",
                    "soul","life","death","hope","fear","joy","pain","trust","faith","peace"],
    "function":    ["the","a","and","or","not","is","was","in","on","of","to","from",
                    "with","for","he","she","it","they","we","I","you","his","her"],
    "animal":      ["dog","cat","bird","fish","horse","wolf","lion","tiger","elephant","mouse",
                    "rabbit","deer","bear","fox","eagle","whale","shark","frog","snake","monkey"],
    "adjective":   ["fast","slow","big","small","hot","cold","old","new","hard","soft",
                    "bright","dark","strong","weak","happy","sad","good","bad","right","wrong"],
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

# ── classify_all() exactly as the trie uses it ────────────────────────────────
def classify_axis(axis_name):
    """Returns list of N 'H'/'U'/'L' strings, using the trie's exact method."""
    axis_vec  = t2_axes[axis_name]
    layer_hs  = hs_by_layer[DAY78_LAYERS[axis_name]]
    if np.linalg.norm(axis_vec) < 1e-6:
        return ["U"] * N
    projs  = layer_hs @ axis_vec              # (N,)
    max_p  = float(np.percentile(projs, 95))
    if max_p < 1e-6:
        return ["U"] * N
    hi, lo = max_p * INV_PHI, max_p * INV_PHI2
    return ["H" if p > hi else "L" if p < lo else "U" for p in projs]

# Compute all 12-axis classifications
print("Classifying all tokens on all 12 axes ...")
all_classes = {}
for name in AXIS_NAMES_12:
    all_classes[name] = classify_axis(name)

addresses = ["".join(all_classes[n][i] for n in AXIS_NAMES_12) for i in range(N)]

# ── Exp 1: H/U/L distribution per axis per category ──────────────────────────
print("=" * 72)
print("Exp 1: H/U/L Class Distributions per Axis × Category")
print("=" * 72)

def cramers_v(cat_labels, hul_labels):
    """Cramér's V: 0=no association, 1=perfect association."""
    n = len(cat_labels)
    contingency = {}
    for c, h in zip(cat_labels, hul_labels):
        contingency[(c, h)] = contingency.get((c, h), 0) + 1
    cats = sorted(set(cat_labels))
    huls = ["H", "U", "L"]
    chi2 = 0.0
    for c in cats:
        for h in huls:
            obs = contingency.get((c, h), 0)
            row = sum(contingency.get((c, hh), 0) for hh in huls)
            col = sum(contingency.get((cc, h), 0) for cc in cats)
            exp = row * col / n
            if exp > 0: chi2 += (obs - exp) ** 2 / exp
    k = min(len(cats), len(huls))
    if k <= 1 or n == 0: return 0.0
    return math.sqrt(chi2 / (n * (k - 1)))

axis_category_results = {}
FOCUS_AXES = {
    "gender":     "gender_pair",
    "past_tense": "verb_past",
    "plural":     "plural_noun",
    "comparative":"comparative",
    "hypernym":   "hypernym",
}

print()
print(f"  {'axis':>14}  {'category':>12}  "
      f"{'in_H%':>7}  {'in_U%':>7}  {'in_L%':>7}  "
      f"{'out_H%':>7}  {'out_U%':>7}  {'out_L%':>7}  "
      f"{'CramérV':>8}  {'day114b_Δ':>10}")
print(f"  {'-'*100}")

DAY114B_DELTAS = {
    "gender": 0.003, "comparative": 0.001, "past_tense": 0.040,
    "plural": 0.028, "hypernym": 0.029,
}

for ax_name, cat_name in FOCUS_AXES.items():
    if cat_name not in TOKEN_CATEGORIES: continue
    cat_idxs   = [word_idx[w] for w in TOKEN_CATEGORIES[cat_name] if w in word_idx]
    other_idxs = [i for i in range(N) if i not in set(cat_idxs)]
    if not cat_idxs or not other_idxs: continue

    cls = all_classes[ax_name]
    cat_hul   = Counter(cls[i] for i in cat_idxs)
    other_hul = Counter(cls[i] for i in other_idxs)

    n_cat   = len(cat_idxs)
    n_other = len(other_idxs)

    cat_h = 100 * cat_hul.get("H", 0) / n_cat
    cat_u = 100 * cat_hul.get("U", 0) / n_cat
    cat_l = 100 * cat_hul.get("L", 0) / n_cat
    oth_h = 100 * other_hul.get("H", 0) / n_other
    oth_u = 100 * other_hul.get("U", 0) / n_other
    oth_l = 100 * other_hul.get("L", 0) / n_other

    # Cramér's V
    cat_labels = ["in"]*n_cat + ["out"]*n_other
    hul_labels = [cls[i] for i in cat_idxs] + [cls[i] for i in other_idxs]
    v = cramers_v(cat_labels, hul_labels)

    delta = DAY114B_DELTAS.get(ax_name, float("nan"))
    axis_category_results[ax_name] = {
        "category": cat_name, "in_H": cat_h, "in_U": cat_u, "in_L": cat_l,
        "out_H": oth_h, "out_U": oth_u, "out_L": oth_l, "cramers_v": v,
        "day114b_delta": delta,
    }
    print(f"  {ax_name:>14}  {cat_name:>12}  "
          f"{cat_h:>7.1f}  {cat_u:>7.1f}  {cat_l:>7.1f}  "
          f"{oth_h:>7.1f}  {oth_u:>7.1f}  {oth_l:>7.1f}  "
          f"{v:>8.4f}  {delta:>10.4f}")

# ── Exp 2: Full vocabulary class distributions ────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Full Vocabulary H/U/L Distributions per Axis")
print("=" * 72)
print(f"\n  {'axis':>14}  {'H%':>7}  {'U%':>7}  {'L%':>7}  "
      f"{'entropy_bits':>13}  {'max_class':>10}")
print(f"  {'-'*65}")

axis_distributions = {}
for name in AXIS_NAMES_12:
    cls = all_classes[name]
    ctr = Counter(cls)
    h_pct = 100 * ctr.get("H", 0) / N
    u_pct = 100 * ctr.get("U", 0) / N
    l_pct = 100 * ctr.get("L", 0) / N
    # Entropy
    ent = 0.0
    for k, v in ctr.items():
        p = v / N
        if p > 0: ent -= p * math.log2(p)
    max_cls = max(ctr, key=ctr.get)
    axis_distributions[name] = {"H": h_pct, "U": u_pct, "L": l_pct,
                                  "entropy": ent, "max_class": max_cls}
    print(f"  {name:>14}  {h_pct:>7.1f}  {u_pct:>7.1f}  {l_pct:>7.1f}  "
          f"{ent:>13.4f}  {max_cls:>10}")

# ── Exp 3: Address uniqueness as a function of k axes ────────────────────────
print()
print("=" * 72)
print("Exp 3: Address Uniqueness vs Number of Axes")
print("=" * 72)
print(f"\n  {'k':>3}  {'unique_addrs':>13}  {'unique_pct':>11}  "
      f"{'max_loo_acc':>12}  {'note':>20}")
print(f"  {'-'*65}")

uniqueness_results = {}
# Use axes ranked by entropy (most informative first)
sorted_by_entropy = sorted(AXIS_NAMES_12, key=lambda n: -axis_distributions[n]["entropy"])
for k in range(1, 13):
    subset = sorted_by_entropy[:k]
    subset_addrs = ["".join(all_classes[n][i] for n in subset) for i in range(N)]
    addr_ctr = Counter(subset_addrs)
    n_unique = len(addr_ctr)
    unique_pct = 100 * n_unique / N
    # Max LOO accuracy = fraction of tokens with a UNIQUE address
    n_unique_tokens = sum(1 for c in addr_ctr.values() if c == 1)
    max_loo = 100 * n_unique_tokens / N
    note = f"+{sorted_by_entropy[k-1][:8]}"
    uniqueness_results[k] = {"n_unique": n_unique, "pct": unique_pct,
                              "max_loo": max_loo, "axes": subset[:k]}
    print(f"  {k:>3}  {n_unique:>13}  {unique_pct:>11.1f}  "
          f"{max_loo:>12.1f}  {note:>20}")

print(f"\n  Full 3^12 space: {3**12:,} possible addresses")
print(f"  Vocabulary size: {N}")
print(f"  Observed addresses at k=12: {uniqueness_results[12]['n_unique']}")

# ── Exp 4: Threshold amplification quantification ────────────────────────────
print()
print("=" * 72)
print("Exp 4: Threshold Amplification — Continuous vs Discrete Signal")
print("=" * 72)
print()
print("  Comparing Day 114b continuous delta to Day 115 Cramér's V:")
print()
print(f"  {'axis':>14}  {'continuous_Δ':>13}  {'cramers_V':>10}  {'amplification':>14}  "
      f"{'verdict':>12}")
print(f"  {'-'*70}")

for ax_name in FOCUS_AXES:
    if ax_name not in axis_category_results: continue
    r     = axis_category_results[ax_name]
    delta = r["day114b_delta"]
    v     = r["cramers_v"]
    amp   = v / max(delta, 1e-6)
    print(f"  {ax_name:>14}  {delta:>13.4f}  {v:>10.4f}  {amp:>14.1f}×  "
          f"{'AMPLIFIED' if v > 0.05 else 'weak':>12}")

# ── Exp 5: Why does LOO work? — address entropy model ─────────────────────────
print()
print("=" * 72)
print("Exp 5: Why LOO=94%? — Address Entropy Analysis")
print("=" * 72)
print()

# For LOO: each token is looked up by its address → returns nearest neighbor
# If address is unique: correct nearest neighbor = self (100%)
# If address is shared: must use distance to all sharing tokens
unique_addrs = Counter(addresses)
n_unique_addr = sum(1 for c in unique_addrs.values() if c == 1)
n_shared_1   = sum(1 for c in unique_addrs.values() if c == 2)
n_shared_more = sum(1 for c in unique_addrs.values() if c >= 3)

print(f"  Total vocabulary tokens: {N}")
print(f"  Total unique addresses:  {len(unique_addrs)}")
print(f"  Tokens with unique address:   {n_unique_addr}  ({100*n_unique_addr/N:.1f}%)")
print(f"  Tokens sharing with 1 other:  {2*n_shared_1}  ({100*2*n_shared_1/N:.1f}%)")
print(f"  Tokens sharing with 2+ others:{N-n_unique_addr-2*n_shared_1}  "
      f"({100*(N-n_unique_addr-2*n_shared_1)/N:.1f}%)")
print()
print(f"  Expected LOO ceiling (unique tokens only): {100*n_unique_addr/N:.1f}%")
print(f"  Reported LOO accuracy (Day 105):           94.0%")
print()

# Show some shared-address groups
shared_groups = [(addr, [valid_words[i] for i in range(N) if addresses[i] == addr])
                 for addr, cnt in unique_addrs.items() if cnt >= 2]
shared_groups.sort(key=lambda x: -len(x[1]))
print(f"  Largest shared-address groups (most confusable pairs):")
for addr, words in shared_groups[:8]:
    print(f"    {addr}: {', '.join(words)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 115 Summary — Threshold Amplification Mechanism")
print("=" * 72)

best_v_axis = max(axis_category_results, key=lambda a: axis_category_results[a]["cramers_v"])
best_v      = axis_category_results[best_v_axis]["cramers_v"]
weakest_delta = min(axis_category_results[a]["day114b_delta"] for a in axis_category_results)
max_loo_k5  = uniqueness_results.get(5, {}).get("max_loo", 0)
max_loo_k12 = uniqueness_results.get(12, {}).get("max_loo", 0)

amplified = [a for a in axis_category_results if axis_category_results[a]["cramers_v"] > 0.05]

print(f"""
  Continuous projection Δ (Day 114b): {weakest_delta:.4f} to {max(r['day114b_delta'] for r in axis_category_results.values()):.4f}
  Discrete Cramér's V (Day 115):      {min(r['cramers_v'] for r in axis_category_results.values()):.4f} to {best_v:.4f}
  Best association: {best_v_axis} (V={best_v:.4f})

  Amplification confirmed: {', '.join(amplified) if amplified else 'none (V < 0.05 for all)'}

  Address uniqueness:
    k=1 axis:  {uniqueness_results[1]['max_loo']:.1f}% unique tokens
    k=5 axes:  {max_loo_k5:.1f}% unique tokens
    k=12 axes: {max_loo_k12:.1f}% unique tokens

  Tokens with unique address (k=12): {n_unique_addr}/{N} ({100*n_unique_addr/N:.1f}%)
  Expected LOO ceiling from uniqueness: {100*n_unique_addr/N:.1f}%
  Reported LOO accuracy (Day 105):      94.0%

  THE MECHANISM:
  {'→ φ-thresholding AMPLIFIES weak continuous signals into strong class separation' if amplified else
   '→ φ-thresholding does NOT strongly amplify continuous signals (V still < 0.05)'}

  {'→ LOO accuracy comes from ADDRESS UNIQUENESS: ' + str(n_unique_addr) + '/' + str(N) +
   ' tokens have unique addresses' if abs(100*n_unique_addr/N - 94.0) < 5 else
   '→ LOO accuracy mechanism requires further investigation (uniqueness != 94%)'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "axis_category_results": axis_category_results,
        "axis_distributions": axis_distributions,
        "uniqueness_results": {str(k): v for k,v in uniqueness_results.items()},
        "n_unique_addr": n_unique_addr,
        "n_shared_1": n_shared_1,
        "shared_groups": [(a, ws) for a, ws in shared_groups[:20]],
        "sorted_by_entropy": sorted_by_entropy,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 115 complete.")
