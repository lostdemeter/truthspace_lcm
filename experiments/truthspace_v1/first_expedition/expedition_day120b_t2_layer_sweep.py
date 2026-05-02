#!/usr/bin/env python3
"""
Day 120b — T2 Layer-Sweep: When Do Semantic Axes Emerge? (Fixed)

Day 120 used phi_bin thresholding with 12 tokens — too coarse (all fall in U).
This version uses Cohen's d (delta / pooled_std) as the discrimination metric,
which is continuous and doesn't require phi_bin binning.

Cohen's d = (mean_class_A - mean_class_B) / pooled_std
  > 0.8 = large, 0.5-0.8 = medium, 0.2-0.5 = small, < 0.2 = tiny

Also measures:
  1. Cross-layer axis-direction cosine (does the axis direction rotate?)
  2. Gram matrix off-diagonal mean per layer (are axes orthogonal at each layer?)
  3. Peak discrimination layer vs Day78 optimal layer

Test layers: 1, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 27, 28
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day120b_t2_layer_sweep.json")
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
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk","Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road","The cars drove slowly down the long empty road"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
        ("The strong root grips the soil","The strong base grips the earth"),
        ("The bridge connects two banks","The bond connects two communities"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days","We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend","She wrote a letter to her dear old friend"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The news is good","The news is bad"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is the first","It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
        ("The teacher helped the student","The student was helped by the teacher"),
        ("The king signed the document","The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The child cries very loudly","The mother comes running in"),
        ("The glass breaks on hard stone","The water spills everywhere"),
        ("The sun heats the cold earth","The ice melts quickly in spring"),
        ("The teacher praises the student","The student feels very proud"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The dog is hungry now","Is the dog hungry now"),
        ("She wrote the letter herself","Did she write the letter herself"),
        ("The house looks very old","Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
    ],
}

# Paired test tokens: (class_A, class_B) — class A is the "base", B is the "transformed"
AXIS_TOKEN_PAIRS = {
    "gender":     [("king","queen"),("man","woman"),("boy","girl"),
                   ("father","mother"),("son","daughter"),("actor","actress"),
                   ("brother","sister"),("prince","princess")],
    "comparative":[("fast","faster"),("big","bigger"),("old","older"),
                   ("cold","colder"),("tall","taller"),("bright","brighter"),
                   ("dark","darker"),("hard","harder")],
    "hypernym":   [("dog","animal"),("rose","flower"),("car","vehicle"),
                   ("eagle","bird"),("ruby","gem"),("hammer","tool"),
                   ("oak","tree"),("salmon","fish")],
    "plural":     [("dog","dogs"),("cat","cats"),("tree","trees"),
                   ("bird","birds"),("book","books"),("car","cars"),
                   ("star","stars"),("hand","hands")],
    "synonym":    [("big","large"),("small","tiny"),("fast","quick"),
                   ("cold","frigid"),("happy","joyful"),("old","aged"),
                   ("hard","difficult"),("sad","unhappy")],
    "concrete":   [("stone","burden"),("road","journey"),("wall","barrier"),
                   ("flame","hope"),("root","base"),("bridge","bond"),
                   ("chain","constraint"),("key","solution")],
    "past_tense": [("walk","walked"),("run","ran"),("eat","ate"),
                   ("see","saw"),("build","built"),("swim","swam"),
                   ("write","wrote"),("fly","flew")],
    "antonym":    [("hot","cold"),("fast","slow"),("good","bad"),
                   ("happy","sad"),("strong","weak"),("old","new"),
                   ("big","small"),("hard","soft")],
    "passive":    [("breaks","broken"),("chases","chased"),("cooks","cooked"),
                   ("builds","built"),("writes","written"),("sees","seen"),
                   ("takes","taken"),("gives","given")],
    "causation":  [("rain","wet"),("fire","ash"),("wind","fall"),
                   ("sun","melt"),("cry","comfort"),("break","spill"),
                   ("heat","evaporate"),("cool","freeze")],
    "question":   [("is","Is"),("can","Can"),("does","Does"),
                   ("was","Was"),("will","Will"),("has","Has"),
                   ("are","Are"),("did","Did")],
    "negation":   [("fast","slow"),("good","bad"),("strong","weak"),
                   ("happy","sad"),("old","young"),("big","small"),
                   ("hot","cold"),("rich","poor")],
}

TEST_LAYERS = [1, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 27, 28]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def cohens_d(a_vals, b_vals):
    """Cohen's d: (mean_A - mean_B) / pooled_std."""
    a = np.array(a_vals); b = np.array(b_vals)
    if len(a) < 2 or len(b) < 2: return 0.0
    pooled_std = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std < 1e-10: return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

print("Computing T2 axes at each test layer ...")
axes_at_layer = {ax: {} for ax in AXIS_NAMES_12}
for ax_name in AXIS_NAMES_12:
    pairs = AXIS_SENTENCE_PAIRS.get(ax_name, [])
    for L in TEST_LAYERS:
        diffs = []
        for s1, s2 in pairs:
            try:
                inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
                with torch.no_grad():
                    o1 = model(**inp1, output_hidden_states=True)
                    o2 = model(**inp2, output_hidden_states=True)
                h1 = o1.hidden_states[L][0,-1,:].numpy().astype(np.float32)
                h2 = o2.hidden_states[L][0,-1,:].numpy().astype(np.float32)
                d = h2-h1; n = np.linalg.norm(d)
                if n > 1e-6: diffs.append(d/n)
            except: pass
        v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, np.float32)
        nv = np.linalg.norm(v)
        axes_at_layer[ax_name][L] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

print("Extracting token hidden states at all test layers ...")
all_words = set()
for pairs in AXIS_TOKEN_PAIRS.values():
    for a, b in pairs: all_words.add(a); all_words.add(b)

word_hs = {}
for w in all_words:
    inp = tok(" " + w, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        word_hs[w] = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                      for L in TEST_LAYERS}
    except: pass
print(f"  {len(word_hs)} words extracted.\n")

# ── Exp 1: Cohen's d per axis per layer ───────────────────────────────────────
print("=" * 90)
print("Exp 1: Cohen's d per Axis per Layer")
print("       (d = (mean_base - mean_transformed) / pooled_std; sign indicates direction)")
print("       Optimal layer marked with *")
print("=" * 90)

header_layers = "  ".join(f"L{L:02d}" for L in TEST_LAYERS)
print(f"\n  {'axis':>14}  OL  {header_layers}")
print("  " + "-" * (14 + 6 + len(header_layers) + 10))

cohens_d_results = {}
for ax_name in AXIS_NAMES_12:
    L_opt = DAY78_LAYERS[ax_name]
    pairs = AXIS_TOKEN_PAIRS[ax_name]
    row = {}
    for L in TEST_LAYERS:
        axis = axes_at_layer[ax_name][L]
        a_projs = []; b_projs = []
        for a, b in pairs:
            if a not in word_hs or b not in word_hs: continue
            ha = normed(word_hs[a][L]); hb = normed(word_hs[b][L])
            a_projs.append(float(np.dot(ha, axis)))
            b_projs.append(float(np.dot(hb, axis)))
        d = cohens_d(a_projs, b_projs)
        row[L] = d
    cohens_d_results[ax_name] = row
    vals_str = "  ".join(
        f"{'*' if L == L_opt else ' '}{abs(row[L]):.2f}" for L in TEST_LAYERS
    )
    print(f"  {ax_name:>14}  {L_opt:2d}  {vals_str}")

# ── Exp 2: Cross-layer axis direction cosine ──────────────────────────────────
print()
print("=" * 90)
print("Exp 2: Cross-Layer Axis Direction Cosine (anchored to optimal layer)")
print("       1.0 = same direction, 0.0 = orthogonal")
print("=" * 90)
print(f"\n  {'axis':>14}  OL  {header_layers}")
print("  " + "-" * (14 + 6 + len(header_layers) + 10))

cross_layer_results = {}
for ax_name in AXIS_NAMES_12:
    L_opt = DAY78_LAYERS[ax_name]
    ax_opt = axes_at_layer[ax_name][L_opt]
    row = {}
    for L in TEST_LAYERS:
        ax_L = axes_at_layer[ax_name][L]
        row[L] = float(abs(np.dot(ax_opt, ax_L)))
    cross_layer_results[ax_name] = row
    vals_str = "  ".join(f"{'*' if L == L_opt else ' '}{row[L]:.2f}" for L in TEST_LAYERS)
    print(f"  {ax_name:>14}  {L_opt:2d}  {vals_str}")

# ── Exp 3: T2 Gram matrix off-diagonal per layer ──────────────────────────────
print()
print("=" * 90)
print("Exp 3: T2 Gram Matrix Off-Diagonal Mean per Layer")
print("       Lower = more orthogonal")
print("=" * 90)
print(f"\n  {'layer':>6}  {'offdiag_mean':>14}  {'offdiag_max':>13}  {'quality':>12}")
print("  " + "-" * 55)

gram_results = {}
for L in TEST_LAYERS:
    axvecs = np.stack([axes_at_layer[ax][L] for ax in AXIS_NAMES_12])
    G = axvecs @ axvecs.T
    np.fill_diagonal(G, 0)
    upper = np.abs(G[np.triu_indices(12, k=1)])
    off_mean = float(np.mean(upper)); off_max = float(np.max(upper))
    gram_results[L] = {"offdiag_mean": off_mean, "offdiag_max": off_max}
    tag = "NEAR-ORTHO" if off_mean < 0.08 else "MODERATE" if off_mean < 0.15 else "CORRELATED"
    print(f"  {'L'+str(L):>6}  {off_mean:>14.4f}  {off_max:>13.4f}  {tag:>12}")

# ── Exp 4: Emergence profile per axis ────────────────────────────────────────
print()
print("=" * 90)
print("Exp 4: Axis Emergence Profile")
print("=" * 90)
print(f"\n  {'axis':>14}  {'OL':>4}  {'OL_d':>6}  {'peak_L':>8}  {'peak_d':>8}  "
      f"{'match':>6}  {'etype':>16}")
print("  " + "-" * 75)

emergence_results = {}
for ax_name in AXIS_NAMES_12:
    L_opt = DAY78_LAYERS[ax_name]
    row   = {L: abs(cohens_d_results[ax_name][L]) for L in TEST_LAYERS}
    d_opt = row.get(L_opt, 0)
    peak_L = max(row, key=row.get); peak_d = row[peak_L]

    early_d = max(row.get(L,0) for L in [1,3,5])
    mid_d   = max(row.get(L,0) for L in [10,13,15])
    late_d  = max(row.get(L,0) for L in [23,25,27,28])

    if late_d >= mid_d and late_d >= early_d and late_d > 0.2:
        etype = "late-emerging"
    elif early_d >= mid_d and early_d >= late_d and early_d > 0.2:
        etype = "early-emerging"
    elif mid_d >= early_d and mid_d >= late_d and mid_d > 0.2:
        etype = "mid-emerging"
    else:
        etype = "plateau/weak"

    match = "YES" if abs(peak_L - L_opt) <= 3 else "NO"
    emergence_results[ax_name] = {
        "L_opt": L_opt, "d_opt": d_opt, "peak_L": peak_L, "peak_d": peak_d,
        "match": match, "etype": etype
    }
    print(f"  {ax_name:>14}  {L_opt:>4}  {d_opt:>6.3f}  "
          f"  {f'L{peak_L:02d}':>6}  {peak_d:>8.3f}  {match:>6}  {etype:>16}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("Day 120b Summary — T2 Layer Emergence Profile")
print("=" * 90)

n_match    = sum(1 for r in emergence_results.values() if r["match"] == "YES")
n_total    = len(emergence_results)
late_axes  = [ax for ax, r in emergence_results.items() if r["etype"] == "late-emerging"]
early_axes = [ax for ax, r in emergence_results.items() if r["etype"] == "early-emerging"]
mid_axes   = [ax for ax, r in emergence_results.items() if r["etype"] == "mid-emerging"]
weak_axes  = [ax for ax, r in emergence_results.items() if r["etype"] == "plateau/weak"]
best_orth_L = min(gram_results, key=lambda L: gram_results[L]["offdiag_mean"])
best_orth_v = gram_results[best_orth_L]["offdiag_mean"]

# Find axes that PEAK at their optimal layer (within 3 layers)
strong_axes = [ax for ax, r in emergence_results.items() if r["peak_d"] > 0.5]

print(f"""
  Day78 optimal layer matches peak Cohen's d layer (±3): {n_match}/{n_total}
  Strong axes (peak |d| > 0.5): {len(strong_axes)}: {', '.join(strong_axes) or 'none'}

  Emergence types:
    Early-emerging (peak L1-5):   {len(early_axes)}: {', '.join(early_axes) or 'none'}
    Mid-emerging (peak L10-15):   {len(mid_axes)}: {', '.join(mid_axes) or 'none'}
    Late-emerging (peak L23-28):  {len(late_axes)}: {', '.join(late_axes) or 'none'}
    Plateau/weak (<0.2 at all):   {len(weak_axes)}: {', '.join(weak_axes) or 'none'}

  T2 Gram matrix most orthogonal at: L{best_orth_L} (offdiag_mean={best_orth_v:.4f})

  VERDICT:
  {'→ T2 axes emerge at FIXED layers (phase transitions): Day78 optimal layers confirmed' if n_match >= 8 else
   '→ T2 axes show MIXED emergence: some at optimal layers, some shift' if n_match >= 5 else
   '→ T2 axes peak at DIFFERENT layers than Day78 optimal: emergence profile reveals new structure'}

  ARCHITECTURE INSIGHT:
  {'→ LATE-DOMINANCE: semantic axes form primarily in late layers (L23-28)' if len(late_axes) > 6 else
   '→ DISTRIBUTED: different semantic properties emerge at different depths' if len(early_axes) > 1 and len(late_axes) > 1 else
   '→ UNIFORM: most axes show similar layer-emergence profiles'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "cohens_d_results": cohens_d_results,
        "cross_layer_results": cross_layer_results,
        "gram_results": gram_results,
        "emergence_results": emergence_results,
        "test_layers": TEST_LAYERS,
        "day78_layers": DAY78_LAYERS,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 120b complete.")
