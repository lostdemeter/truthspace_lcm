#!/usr/bin/env python3
"""
Day 121 — Layer-Optimal LOO: Does Using Cohen's d-Optimal Layers Beat Day78?

Day 78 found optimal layers from sentence-level last-token representations.
Day 120b found different peak Cohen's d layers for isolated words.
  7/12 axes peak EARLIER in isolated-word measurement.

QUESTION: If we use Day120b's Cohen's d-optimal layers for the trie address
system, does LOO accuracy improve or degrade vs Day78's 94% baseline?

CONFIGURATIONS:
  Config A: Day78 layers (baseline, 94% LOO)
  Config B: Day120b peak-Cohen's-d layers
  Config C: All-L1 (every axis projected at L1)
  Config D: All-L28 (every axis projected at L28 — late-layer baseline)
  Config E: Mixed-optimal (best per-axis from A vs B individually)

For each config, compute:
  1. phi-threshold 12-symbol ternary address per token
  2. Address uniqueness rate (fraction with unique 12-char address)
  3. LOO accuracy = unique + euclidean fallback for collisions
  4. Mean T2 inter-axis discrimination (Cohen's d averaged over all axes)
"""
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day121_layer_optimal_loo.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Configuration: axis -> layer mapping
DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
DAY120B_PEAK = {
    "gender": 27, "comparative": 1, "hypernym": 1, "plural": 27,
    "synonym": 1, "concrete": 1, "past_tense": 1, "antonym": 10,
    "passive": 28, "causation": 1, "question": 28, "negation": 5,
}
ALL_L1  = {ax: 1  for ax in DAY78_LAYERS}
ALL_L28 = {ax: 28 for ax in DAY78_LAYERS}
# Mixed: per-axis choose whichever of day78 vs day120b had higher peak_d
# From day120b results (using Day78 OL_d vs Day120b peak_d):
DAY120B_PEAK_D = {
    "gender": 6.257, "comparative": 8.453, "hypernym": 2.167, "plural": 2.411,
    "synonym": 2.973, "concrete": 1.694, "past_tense": 2.848, "antonym": 2.347,
    "passive": 0.933, "causation": 1.314, "question": 2.306, "negation": 1.552,
}
DAY78_OL_D = {
    "gender": 6.257, "comparative": 3.060, "hypernym": 0.412, "plural": 0.654,
    "synonym": 1.019, "concrete": 0.127, "past_tense": 1.629, "antonym": 0.730,
    "passive": 0.933, "causation": 0.193, "question": 2.306, "negation": 0.664,
}
MIXED_LAYERS = {
    ax: (DAY78_LAYERS[ax] if DAY78_OL_D[ax] >= DAY120B_PEAK_D[ax]
         else DAY120B_PEAK[ax])
    for ax in DAY78_LAYERS
}

CONFIGS = {
    "A_day78":      DAY78_LAYERS,
    "B_day120b":    DAY120B_PEAK,
    "C_all_L1":     ALL_L1,
    "D_all_L28":    ALL_L28,
    "E_mixed":      MIXED_LAYERS,
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

# 420 probe vocabulary (same as Day 114b)
VOCAB_420 = [
    "king","queen","man","woman","boy","girl","father","mother","son","daughter",
    "actor","actress","brother","sister","prince","princess","husband","wife",
    "dog","cat","tree","bird","rose","flower","car","vehicle","hammer","tool",
    "eagle","gem","ruby","salmon","fish","oak","horse","wolf","lion","deer",
    "run","ran","walk","walked","eat","ate","see","saw","build","built",
    "swim","swam","write","wrote","fly","flew","speak","spoke","break","broke",
    "fast","faster","big","bigger","old","older","cold","colder","tall","taller",
    "bright","brighter","dark","darker","hard","harder","warm","warmer",
    "hot","cold","good","bad","happy","sad","strong","weak","old","new",
    "big","small","rich","poor","light","dark","early","late","open","closed",
    "big","large","small","tiny","fast","quick","cold","frigid","happy","joyful",
    "hard","difficult","sad","unhappy","tired","exhausted","smart","intelligent",
    "stone","burden","road","journey","wall","barrier","flame","hope",
    "root","base","bridge","bond","chain","constraint","key","solution",
    "dogs","cats","trees","birds","books","cars","stars","hands","eyes",
    "men","women","children","words","thoughts","years","days","hours",
    "the","a","an","is","was","are","were","be","been","being",
    "and","or","but","not","so","yet","for","nor","after","before",
    "he","she","it","they","we","you","I","me","him","her","us","them",
    "his","her","its","their","our","your","my","this","that","these","those",
    "very","quite","rather","almost","nearly","just","only","also","even","still",
    "never","always","often","sometimes","usually","rarely","already","soon",
    "run","jump","sing","dance","think","feel","know","want","need","like",
    "take","give","make","get","go","come","see","hear","say","tell",
    "France","Paris","Germany","Berlin","Japan","Tokyo","Italy","Rome",
    "Spain","Madrid","China","Beijing","Russia","Moscow","Canada","Ottawa",
    "England","London","India","Delhi","Brazil","Brasilia","Egypt","Cairo",
    "red","blue","green","yellow","white","black","orange","purple","pink","brown",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "first","second","third","last","next","previous","following","prior",
    "up","down","left","right","in","out","on","off","over","under","through",
    "science","art","music","history","math","language","culture","nature",
    "water","fire","earth","wind","light","dark","time","space","mind","soul",
    "beautiful","ugly","strange","normal","simple","complex","clear","confused",
    "angry","calm","brave","afraid","proud","ashamed","free","trapped",
]
# Deduplicate while preserving order
seen = set(); VOCAB = []
for w in VOCAB_420:
    if w.lower() not in seen: seen.add(w.lower()); VOCAB.append(w)

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

def phi_bin(x):
    if   x >  INV_PHI:  return "H"
    elif x < -INV_PHI2: return "L"
    else:               return "U"

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
ALL_LAYERS  = sorted(set(
    list(DAY78_LAYERS.values()) +
    list(DAY120B_PEAK.values()) +
    [1, 5, 10, 28]
))
print(f"  hidden={hidden_size}, vocab={len(VOCAB)}, layers={ALL_LAYERS}\n")

print("Computing T2 axes for all needed layers ...")
axes = {}  # {ax_name: {layer: unit_vec}}
for ax_name in AXIS_NAMES_12:
    axes[ax_name] = {}
    for L in ALL_LAYERS:
        diffs = []
        for s1, s2 in AXIS_SENTENCE_PAIRS.get(ax_name, []):
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
        axes[ax_name][L] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

print(f"Extracting hidden states for {len(VOCAB)} vocabulary tokens ...")
word_hs = {}
for w in VOCAB:
    inp = tok(" " + w, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        word_hs[w] = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                      for L in ALL_LAYERS}
    except: pass
valid_vocab = [w for w in VOCAB if w in word_hs]
N = len(valid_vocab)
print(f"  {N} valid tokens.\n")

def compute_t2_matrix(layer_map):
    """Compute T2 12D continuous projection matrix for all vocab tokens."""
    mat = np.zeros((N, 12), dtype=np.float32)
    for k, ax_name in enumerate(AXIS_NAMES_12):
        L   = layer_map[ax_name]
        ax  = axes[ax_name][L]
        for i, w in enumerate(valid_vocab):
            h = normed(word_hs[w][L])
            mat[i, k] = float(np.dot(h, ax))
    return mat

def compute_addresses(t2_mat):
    """Convert 12D projections to 12-char H/U/L addresses."""
    return ["".join(phi_bin(t2_mat[i, k]) for k in range(12)) for i in range(N)]

def loo_accuracy(addresses, t2_mat):
    """
    LOO accuracy: for each token, find nearest neighbor (excluding self).
    'Correct' if nearest neighbor shares at least 6 address chars (loose)
    OR if the address is unique.
    Better metric: what fraction of addresses are unique?
    """
    addr_counts = Counter(addresses)
    n_unique    = sum(1 for a in addresses if addr_counts[a] == 1)
    unique_rate = n_unique / N

    # Euclidean nearest-neighbor LOO for collision tokens
    t2_norms = t2_mat / (np.linalg.norm(t2_mat, axis=1, keepdims=True) + 1e-8)
    correct_loo = 0
    for i in range(N):
        addr_i = addresses[i]
        if addr_counts[addr_i] == 1:
            correct_loo += 1  # unique = always "correct" by address
        else:
            # Find nearest neighbor in T2 space (excluding self)
            dists = np.linalg.norm(t2_mat - t2_mat[i], axis=1)
            dists[i] = 1e9
            nn_idx = int(np.argmin(dists))
            # "Correct" if NN has same address (within the collision group)
            if addresses[nn_idx] == addr_i:
                correct_loo += 1
    return n_unique, unique_rate, correct_loo / N

print("=" * 72)
print("Computing T2 addresses and LOO accuracy for all configurations ...")
print("=" * 72)
print()

results = {}
for cfg_name, layer_map in CONFIGS.items():
    print(f"  Config {cfg_name}: layers = {[layer_map[ax] for ax in AXIS_NAMES_12]}")
    t2_mat   = compute_t2_matrix(layer_map)
    addresses = compute_addresses(t2_mat)
    n_unique, unique_rate, loo_acc = loo_accuracy(addresses, t2_mat)
    results[cfg_name] = {
        "layer_map": layer_map,
        "n_unique": n_unique, "unique_rate": unique_rate, "loo_acc": loo_acc,
        "n_tokens": N,
    }
    print(f"    unique: {n_unique}/{N} ({100*unique_rate:.1f}%)  LOO: {100*loo_acc:.1f}%")

# ── Address entropy per config ─────────────────────────────────────────────
print()
print("=" * 72)
print("Per-Config Summary Table")
print("=" * 72)
print(f"\n  {'config':>15}  {'unique_rate':>12}  {'loo_acc':>9}  "
      f"{'vs_day78':>10}  {'layers':>30}")
print(f"  {'-'*82}")

day78_loo = results["A_day78"]["loo_acc"]
day78_uniq = results["A_day78"]["unique_rate"]
for cfg_name, r in results.items():
    delta_loo  = 100*(r["loo_acc"] - day78_loo)
    delta_uniq = 100*(r["unique_rate"] - day78_uniq)
    layer_str  = ",".join(str(CONFIGS[cfg_name][ax]) for ax in AXIS_NAMES_12)
    print(f"  {cfg_name:>15}  {100*r['unique_rate']:>11.1f}%  "
          f"{100*r['loo_acc']:>8.1f}%  {delta_loo:>+9.1f}pp  [{layer_str}]")

# ── Per-axis: which layer setting contributes most? ──────────────────────────
print()
print("=" * 72)
print("Per-Axis Layer Choice Analysis: Which Config Uses Each Layer Best?")
print("=" * 72)
print(f"\n  For each axis, test using day78 vs day120b optimal while keeping all")
print(f"  others at day78. Measure LOO impact of swapping ONE axis at a time.")
print()
print(f"  {'axis':>14}  {'day78_L':>8}  {'d120b_L':>8}  "
      f"{'loo_day78_only':>16}  {'loo_d120b_only':>16}  {'best':>10}")
print(f"  {'-'*78}")

per_axis_results = {}
for ax_name in AXIS_NAMES_12:
    L_78   = DAY78_LAYERS[ax_name]
    L_120b = DAY120B_PEAK[ax_name]
    if L_78 == L_120b:
        per_axis_results[ax_name] = {"same": True, "L_78": L_78}
        print(f"  {ax_name:>14}  {'L'+str(L_78):>8}  {'L'+str(L_120b):>8}  "
              f"{'(same)':>16}  {'(same)':>16}  {'—':>10}")
        continue
    # Config: day78 for all except this axis at day78
    map_78 = dict(DAY78_LAYERS); map_78[ax_name] = L_78
    mat_78 = compute_t2_matrix(map_78)
    addr_78 = compute_addresses(mat_78)
    _, _, loo_78 = loo_accuracy(addr_78, mat_78)

    # Config: day78 for all except this axis at day120b peak
    map_120b = dict(DAY78_LAYERS); map_120b[ax_name] = L_120b
    mat_120b = compute_t2_matrix(map_120b)
    addr_120b = compute_addresses(mat_120b)
    _, _, loo_120b = loo_accuracy(addr_120b, mat_120b)

    best = f"L{L_78}" if loo_78 >= loo_120b else f"L{L_120b}"
    delta = 100*(loo_120b - loo_78)
    per_axis_results[ax_name] = {
        "L_78": L_78, "L_120b": L_120b,
        "loo_78": loo_78, "loo_120b": loo_120b,
        "delta_pp": delta, "best": best
    }
    print(f"  {ax_name:>14}  {'L'+str(L_78):>8}  {'L'+str(L_120b):>8}  "
          f"{100*loo_78:>15.1f}%  {100*loo_120b:>15.1f}%  "
          f"{best:>10} ({delta:+.1f}pp)")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 121 Summary — Layer-Optimal LOO")
print("=" * 72)

best_cfg    = max(results, key=lambda c: results[c]["loo_acc"])
best_loo    = results[best_cfg]["loo_acc"]
axes_day78_better   = [ax for ax, r in per_axis_results.items()
                        if not r.get("same") and r.get("loo_78",0) >= r.get("loo_120b",0)]
axes_day120b_better = [ax for ax, r in per_axis_results.items()
                        if not r.get("same") and r.get("loo_120b",0) > r.get("loo_78",0)]

print(f"""
  Best config:    {best_cfg}  ({100*best_loo:.1f}% LOO)
  Day78 baseline: {100*day78_loo:.1f}% LOO

  Per-axis: Day78 layer better for:   {', '.join(axes_day78_better) or 'none'}
  Per-axis: Day120b layer better for: {', '.join(axes_day120b_better) or 'none'}

  VERDICT:
  {f'→ Day78 sentence-level layers ARE optimal for the trie ({100*day78_loo:.1f}%)' if best_cfg == 'A_day78' else
   f'→ Day120b early-layer axes IMPROVE LOO by {100*(best_loo-day78_loo):+.1f}pp ({best_cfg}: {100*best_loo:.1f}%)'}

  KEY FINDING:
  {'→ Sentence-level optimal layers outperform isolated-word optimal layers for the trie' if best_cfg == 'A_day78' else
   '→ Isolated-word optimal layers improve address uniqueness for the trie' if best_cfg == 'B_day120b' else
   '→ Mixed layer strategy (best of both) gives the highest LOO accuracy'}

  IMPLICATION:
  {'→ The trie should continue using Day78 sentence-level optimal layers' if best_cfg == 'A_day78' else
   '→ Re-optimize the trie axes using isolated-word peak-Cohen-d layers'})
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "results": results,
        "per_axis_results": per_axis_results,
        "best_config": best_cfg,
        "day78_loo": day78_loo,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 121 complete.")
